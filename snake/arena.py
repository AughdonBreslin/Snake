"""Deterministic evaluation. No torch, no pygame."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from snake.env import SnakeEnv, default_starvation_limit

OUTCOMES = ("wall", "self", "starvation", "solved")


@dataclass(frozen=True)
class GameResult:
    score: int
    length: int
    steps: int
    outcome: str


def play_games(agent_factory, size, n_games, seed, starvation_limit=None):
    """Run n_games with one agent, one game per seed offset.

    agent_factory takes a generator and returns a fresh agent, so stateful
    agents cannot leak information between games.
    """
    if starvation_limit is None:
        starvation_limit = default_starvation_limit(size)
    results = []
    for game_index in range(n_games):
        env_rng = np.random.default_rng([seed, game_index, 0])
        agent_rng = np.random.default_rng([seed, game_index, 1])
        env = SnakeEnv(size, env_rng, starvation_limit=starvation_limit)
        agent = agent_factory(agent_rng)
        steps = 0
        while not env.game_over:
            env.step(agent.act(env))
            steps += 1
        results.append(
            GameResult(
                score=env.score,
                length=env.length,
                steps=steps,
                outcome=env.death_cause,
            )
        )
    return results


def summarize(results, total_cells):
    scores = [r.score for r in results]
    outcomes = {name: 0 for name in OUTCOMES}
    for r in results:
        outcomes[r.outcome] += 1
    return {
        "games": len(results),
        "mean_score": float(statistics.fmean(scores)),
        "median_score": float(statistics.median(scores)),
        "max_score": int(max(scores)),
        "mean_steps": float(statistics.fmean(r.steps for r in results)),
        "mean_fill_fraction": float(
            statistics.fmean(r.length / total_cells for r in results)
        ),
        "outcomes": outcomes,
    }
