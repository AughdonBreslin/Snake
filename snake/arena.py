"""Deterministic evaluation. No torch, no pygame."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from snake.env import SnakeEnv, default_starvation_limit
from snake.mcts import run_search

OUTCOMES = ("wall", "self", "starvation", "solved")


@dataclass(frozen=True)
class GameResult:
    score: int
    length: int
    steps: int
    outcome: str
    # Which game of the run this was. Games finish out of order when played in
    # lockstep, so without this a result cannot be traced back to the seed that
    # produced it, and a failure cannot be replayed.
    game_index: int = -1


def play_games(agent_factory, size, n_games, seed, starvation_limit=None, on_game=None):
    """Run n_games with one agent, one game per seed offset.

    agent_factory takes a generator and returns a fresh agent, so stateful
    agents cannot leak information between games.

    on_game, if given, is called as each game finishes with the running count,
    the total, and that game's result.
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
                game_index=game_index,
            )
        )
        if on_game is not None:
            on_game(len(results), n_games, results[-1])
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


def play_games_batched(
    agent_factory,
    size,
    n_games,
    seed,
    evaluator,
    simulations,
    starvation_limit=None,
    on_game=None,
):
    """Same experiment as play_games, run in lockstep so the leaf evaluations
    at each simulation index batch into one call.

    play_games asks each agent for a whole move, so every search runs alone and
    the network is called one sample at a time. That is fine for the hand
    written baselines, which do no search at all, and very slow for a searching
    agent once games get long.

    This produces identical results rather than merely similar ones. Each
    game's environment and agent generators are derived from the seed and the
    game index exactly as play_games derives them, so games are independent of
    each other and of their ordering, and the agent draws its per move seed from
    its own generator in the same sequence either way. Batching therefore
    changes when the network is called, never what it returns.

    Requires agents exposing new_search and choose. The baselines do not search
    and should keep using play_games.

    on_game, if given, is called as each game finishes with the running count,
    the total, and that game's result. Games end whenever they end, so the count
    is monotonic but the game index is not.
    """
    if starvation_limit is None:
        starvation_limit = default_starvation_limit(size)

    envs, agents = [], []
    for game_index in range(n_games):
        envs.append(
            SnakeEnv(
                size,
                np.random.default_rng([seed, game_index, 0]),
                starvation_limit=starvation_limit,
            )
        )
        agents.append(agent_factory(np.random.default_rng([seed, game_index, 1])))

    steps = [0] * n_games
    results = [None] * n_games
    active = list(range(n_games))
    completed = 0

    while active:
        searches = [agents[index].new_search(envs[index]) for index in active]
        run_search(searches, evaluator, simulations)
        still_active = []
        for index, search in zip(active, searches, strict=True):
            env = envs[index]
            env.step(agents[index].choose(env, search))
            steps[index] += 1
            if env.game_over:
                results[index] = GameResult(
                    score=env.score,
                    length=env.length,
                    steps=steps[index],
                    outcome=env.death_cause,
                    game_index=index,
                )
                completed += 1
                if on_game is not None:
                    on_game(completed, n_games, results[index])
            else:
                still_active.append(index)
        active = still_active

    return results
