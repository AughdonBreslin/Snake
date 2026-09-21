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


def _draw(surface, pygame, font, env, cell, margin, score, game, games, solved):
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
        f"game {game}/{games}   solved {solved}", True, DIM), (margin, bar + 24))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Watch a checkpoint play snake.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=5)
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
    from snake.mcts import Search, run_search
    from snake.train import load_for_inference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_for_inference(args.checkpoint, device)
    evaluator = TorchEvaluator(model, device)
    size = cfg.train.board_size
    search_cfg = dataclasses.replace(
        cfg.search,
        dirichlet_epsilon=0.0,  # evaluation is deterministic, no exploration noise
        simulations=args.simulations or cfg.search.simulations,
    )

    pygame.init()
    font = pygame.font.SysFont("monospace", 18)
    margin, cell = 16, args.cell
    side = margin * 2 + (size + 2) * cell
    surface = pygame.display.set_mode((side, side + 56))
    pygame.display.set_caption(f"snake {size}x{size} - {os.path.basename(args.checkpoint)}")
    clock = pygame.time.Clock()

    print(f"{args.checkpoint}: {size}x{size}, {search_cfg.simulations} sims/move, {device}")
    solved, scores, running = 0, [], True
    for game in range(1, args.games + 1):
        env = SnakeEnv(size, np.random.default_rng([args.seed, game]),
                       default_starvation_limit(size))
        rng = np.random.default_rng([args.seed, game, 1])
        move = 0
        while running and not env.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key in
                        (pygame.K_ESCAPE, pygame.K_q)):
                    running = False
            search = Search(env, search_cfg, np.random.default_rng([args.seed, game, move + 2]))
            run_search([search], evaluator, search_cfg.simulations)
            counts = search.visit_counts()
            env.step(int(np.argmax(np.where(env.legal_actions(), counts, -1.0))))
            move += 1
            _draw(surface, pygame, font, env, cell, margin,
                  env.score, game, args.games, solved)
            pygame.display.flip()
            clock.tick(args.moves_per_second)
        if not running:
            break
        scores.append(env.score)
        solved += env.death_cause == "solved"
        print(f"  game {game}: score {env.score:>3}  length {env.length:>3}  "
              f"{move:>4} moves  {env.death_cause}")

    pygame.quit()
    if scores:
        print(f"mean {np.mean(scores):.2f} over {len(scores)} games, {solved} solved")


if __name__ == "__main__":
    main()
