"""Watch a trained checkpoint play, in a window.

Lives beside app.py rather than inside snake/, because snake/ is deliberately
free of pygame so the headless code stays importable without a display. This is
the rendering layer; the rules and the network come from the package.

    ./venv/bin/python watch.py --checkpoint runs/6x6-qnorm/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np

BACKGROUND = (24, 24, 28)
GRID_DARK = (40, 40, 46)
GRID_LIGHT = (50, 50, 58)
WALL = (30, 30, 34)
FOOD = (214, 69, 69)
TEXT = (235, 235, 240)
DIM = (140, 140, 150)


def _snake_colour(position, length):
    """Head bright, tail dark, so the direction of travel reads at a glance."""
    shade = 90 + 140 * (1 - position / max(1, length))
    return (40, int(shade), 70)


def _draw(surface, pygame, font, env, cell, margin, score, index, position, total, solved):
    surface.fill(BACKGROUND)
    dim = env.grid_dim
    for x in range(dim):
        for y in range(dim):
            rect = (margin + x * cell, margin + y * cell, cell, cell)
            if x == 0 or y == 0 or x == dim - 1 or y == dim - 1:
                pygame.draw.rect(surface, WALL, rect)
            else:
                shade = GRID_DARK if (x + y) % 2 == 0 else GRID_LIGHT
                pygame.draw.rect(surface, shade, rect)

    if env.food is not None:
        fx, fy = env.food
        pygame.draw.rect(
            surface, FOOD, (margin + fx * cell + cell // 4, margin + fy * cell + cell // 4,
                            cell // 2, cell // 2))

    length = len(env.snake)
    for index, (x, y) in enumerate(env.snake):
        pygame.draw.rect(
            surface, _snake_colour(index, length),
            (margin + x * cell + 1, margin + y * cell + 1, cell - 2, cell - 2))

    bar = margin + dim * cell + 8
    surface.blit(font.render(
        f"score {score}   length {length}/{env.total_cells}", True, TEXT), (margin, bar))
    surface.blit(font.render(
        f"game {index}   ({position}/{total})   solved {solved}", True, DIM),
        (margin, bar + 24))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Watch a checkpoint play snake.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=5,
                        help="play game indices 0..N-1")
    parser.add_argument("--only", default=None,
                        help="replay specific game indices, comma separated, "
                             "as reported by `snake.cli eval`")
    parser.add_argument("--moves-per-second", type=float, default=12.0)
    parser.add_argument("--cell", type=int, default=56, help="pixels per board cell")
    parser.add_argument("--simulations", type=int, default=None,
                        help="override the checkpoint's search budget")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true",
                        help="run with no window, for checking it completes")
    args = parser.parse_args(argv)

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    # Imported here so --help works without a display or the training stack.
    import dataclasses
    import pygame
    import torch

    from snake.env import SnakeEnv, default_starvation_limit
    from snake.evaluator import TorchEvaluator
    from snake.mcts import run_search
    from snake.train import NetworkAgent, load_for_inference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_for_inference(args.checkpoint, device)
    evaluator = TorchEvaluator(model, device)
    size = cfg.train.board_size
    if args.simulations:
        cfg = dataclasses.replace(
            cfg, search=dataclasses.replace(cfg.search, simulations=args.simulations)
        )
    simulations = cfg.search.simulations
    indices = ([int(part) for part in args.only.split(",")] if args.only
               else list(range(args.games)))

    pygame.init()
    font = pygame.font.SysFont("monospace", 18)
    margin, cell = 16, args.cell
    side = margin * 2 + (size + 2) * cell
    surface = pygame.display.set_mode((side, side + 56))
    pygame.display.set_caption(f"snake {size}x{size} - {os.path.basename(args.checkpoint)}")
    clock = pygame.time.Clock()

    print(f"{args.checkpoint}: {size}x{size}, {simulations} sims/move, {device}")
    print(f"replaying game indices {indices} at seed {args.seed}")
    solved, scores, running = 0, [], True
    for position, index in enumerate(indices, start=1):
        # Seeded exactly as snake/arena.py seeds it, so game N here is the same
        # game N that `snake.cli eval` reported. NetworkAgent draws its per move
        # seed from its own generator in the same sequence, so the replay is the
        # game that was scored, not a fresh one from the same position.
        env = SnakeEnv(size, np.random.default_rng([args.seed, index, 0]),
                       default_starvation_limit(size))
        agent = NetworkAgent(evaluator, cfg, np.random.default_rng([args.seed, index, 1]))
        move = 0
        while running and not env.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key in
                        (pygame.K_ESCAPE, pygame.K_q)):
                    running = False
            search = agent.new_search(env)
            run_search([search], evaluator, simulations)
            env.step(agent.choose(env, search))
            move += 1
            _draw(surface, pygame, font, env, cell, margin,
                  env.score, index, position, len(indices), solved)
            pygame.display.flip()
            clock.tick(args.moves_per_second)
        if not running:
            break
        scores.append(env.score)
        solved += env.death_cause == "solved"
        print(f"  game {index:>3}: score {env.score:>3}  length {env.length:>3}  "
              f"{move:>4} moves  {env.death_cause}")

    pygame.quit()
    if scores:
        print(f"mean {np.mean(scores):.2f} over {len(scores)} games, {solved} solved")


if __name__ == "__main__":
    main()
