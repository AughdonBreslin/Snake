"""Reference agents. No torch, no pygame.

These exist so that a learned agent's score means something. A mean score of 30
on a 20x20 board is uninterpretable until you know what a greedy heuristic and a
BFS solver reach on the same seeds.
"""

from __future__ import annotations

import numpy as np

from snake.env import DELTAS


def _probe(env):
    """A throwaway copy for one step of lookahead.

    The generator is fresh rather than shared, so probing is order independent
    and two calls on the same state always agree. It is only consumed at all if
    the probe move happens to eat, and where the replacement food lands does not
    affect any question we ask of a probe.
    """
    return env.clone(np.random.default_rng(0))


def survivable(env, action):
    """Whether the action does not immediately end the episode."""
    return not _probe(env).step(action)


def _legal(env):
    mask = env.legal_actions()
    return [a for a in range(4) if mask[a]]


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class RandomAgent:
    """Uniform over legal actions. The floor."""

    def __init__(self, rng):
        self.rng = rng

    def act(self, env):
        return int(self.rng.choice(_legal(env)))


class GreedyAgent:
    """Move toward the food, preferring moves that are not immediately fatal.

    The survival filter matters: without it a greedy agent walks into the first
    wall it meets and scores no better than random, which makes it useless as a
    reference point.
    """

    def __init__(self, rng):
        self.rng = rng

    def act(self, env):
        legal = _legal(env)
        safe = [a for a in legal if survivable(env, a)]
        options = safe or legal
        if env.food is None:
            return int(self.rng.choice(options))
        head = env.snake[0]
        best = min(
            _distance((head[0] + DELTAS[a][0], head[1] + DELTAS[a][1]), env.food)
            for a in options
        )
        tied = [
            a
            for a in options
            if _distance((head[0] + DELTAS[a][0], head[1] + DELTAS[a][1]), env.food)
            == best
        ]
        return int(self.rng.choice(tied))
