"""Entry points. Import torch lazily so the baseline commands run without it."""

from __future__ import annotations

import argparse
import json
import pathlib

from snake.arena import play_games, summarize
from snake.baselines import BFSSafeAgent, GreedyAgent, HamiltonianAgent, RandomAgent


def _baselines(args):
    agents = {
        "random": lambda size: RandomAgent,
        "greedy": lambda size: GreedyAgent,
        "bfs_safe": lambda size: BFSSafeAgent,
        "hamiltonian": lambda size: (lambda rng: HamiltonianAgent(size, args.seed)),
    }
    report = {}
    for size in args.sizes:
        report[str(size)] = {}
        for name, factory_for in agents.items():
            if name == "hamiltonian" and size % 2 == 1:
                continue
            results = play_games(
                factory_for(size), size=size, n_games=args.games, seed=args.seed
            )
            summary = summarize(results, total_cells=size * size)
            report[str(size)][name] = summary
            print(
                f"size {size:>3} {name:<12} "
                f"mean score {summary['mean_score']:8.2f} "
                f"fill {summary['mean_fill_fraction']:6.3f} "
                f"outcomes {summary['outcomes']}"
            )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


def _train(args):
    # Imported here so the baselines subcommand keeps working without torch.
    from snake.config import RunConfig
    from snake.train import Trainer

    cfg = RunConfig.build(
        net={"channels": args.channels, "blocks": args.blocks, "groups": args.groups},
        search={"simulations": args.simulations},
        train={
            "board_size": args.board_size,
            "iterations": args.iterations,
            "concurrent_games": args.concurrent_games,
            "train_steps_per_iteration": args.train_steps,
            "batch_size": args.batch_size,
            "eval_games": args.eval_games,
            "eval_every": args.eval_every,
            "checkpoint_every": args.checkpoint_every,
            "seed": args.seed,
            "device": args.device,
            "run_dir": args.run_dir,
        },
    )
    trainer = Trainer(cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"resumed from {args.resume} at iteration {trainer.iteration}")

    for _ in range(cfg.train.iterations):
        metrics = trainer.run_iteration()
        print(
            f"iter {trainer.iteration:>4} "
            f"loss {metrics['total']:7.4f} "
            f"entropy {metrics['entropy']:6.4f} "
            f"value mae {metrics['value_mae']:6.4f} "
            f"score {metrics['mean_score']:6.2f} "
            f"games/s {metrics['games_per_second']:6.2f}"
        )
        if trainer.iteration % cfg.train.eval_every == 0:
            summary = trainer.evaluate()
            print(f"  eval mean score {summary['mean_score']:.2f} "
                  f"fill {summary['mean_fill_fraction']:.3f} "
                  f"outcomes {summary['outcomes']}")
            trainer.record_best(summary["mean_score"])
        if trainer.iteration % cfg.train.checkpoint_every == 0:
            trainer.save_checkpoint(f"iter{trainer.iteration:06d}")
    trainer.save_checkpoint(f"iter{trainer.iteration:06d}")


def _eval(args):
    import torch

    from snake.arena import play_games, summarize
    from snake.evaluator import TorchEvaluator
    from snake.train import NetworkAgent, load_for_inference

    device = torch.device(args.device)
    model, cfg = load_for_inference(args.checkpoint, device)
    evaluator = TorchEvaluator(model, device)
    size = cfg.train.board_size
    summary = summarize(
        play_games(
            lambda rng: NetworkAgent(evaluator, cfg, rng),
            size=size,
            n_games=args.games,
            seed=args.seed,
        ),
        total_cells=size * size,
    )
    print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="snake")
    sub = parser.add_subparsers(dest="command", required=True)

    baselines = sub.add_parser("baselines", help="score the reference agents")
    baselines.add_argument("--sizes", type=int, nargs="+", default=[6, 10, 20])
    baselines.add_argument("--games", type=int, default=100)
    baselines.add_argument("--seed", type=int, default=0)
    baselines.add_argument("--out", default="results/baselines.json")
    baselines.set_defaults(func=_baselines)

    train = sub.add_parser("train", help="run the self play learning loop")
    train.add_argument("--board-size", type=int, default=6)
    train.add_argument("--iterations", type=int, default=200)
    train.add_argument("--simulations", type=int, default=100)
    train.add_argument("--concurrent-games", type=int, default=32)
    train.add_argument("--train-steps", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--channels", type=int, default=64)
    train.add_argument("--blocks", type=int, default=6)
    train.add_argument("--groups", type=int, default=8)
    train.add_argument("--eval-games", type=int, default=100)
    train.add_argument("--eval-every", type=int, default=5)
    train.add_argument("--checkpoint-every", type=int, default=5)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cuda")
    train.add_argument("--run-dir", default="runs/default")
    train.add_argument("--resume", default=None)
    train.set_defaults(func=_train)

    evaluate = sub.add_parser("eval", help="score a checkpoint")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--games", type=int, default=100)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--device", default="cuda")
    evaluate.set_defaults(func=_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
