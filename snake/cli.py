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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="snake")
    sub = parser.add_subparsers(dest="command", required=True)

    baselines = sub.add_parser("baselines", help="score the reference agents")
    baselines.add_argument("--sizes", type=int, nargs="+", default=[6, 10, 20])
    baselines.add_argument("--games", type=int, default=100)
    baselines.add_argument("--seed", type=int, default=0)
    baselines.add_argument("--out", default="results/baselines.json")
    baselines.set_defaults(func=_baselines)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
