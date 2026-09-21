"""Reference agents. No torch, no pygame.

These exist so that a learned agent's score means something. A mean score of 30
on a 20x20 board is uninterpretable until you know what a greedy heuristic and a
BFS solver reach on the same seeds.
"""

from __future__ import annotations

import random as _stdlib_random
from collections import deque
from typing import Protocol

import numpy as np

from graph import HamiltonianCycle
from snake.env import DELTAS


class Agent(Protocol):
    """The contract every reference and learned agent implements.

    An agent looks at the current environment state and returns one legal
    action. It must never return an action that env.legal_actions() forbids.
    """

    def act(self, env) -> int:
        ...


def _probe(env):
    """A throwaway copy for one step of lookahead.

    The generator is fresh rather than shared, so probing is order independent
    and two calls on the same state always agree. It is only consumed at all if
    the probe move happens to eat, and where the replacement food lands does not
    affect any question we ask of a probe.
    """
    return env.clone(np.random.default_rng(0))


def survivable(env, action):
    """Whether the action does not end the episode badly.

    Solving the board is a terminal state but it is the best possible outcome,
    so it must not be treated the same as a wall, a self collision, or
    starvation.
    """
    probe = _probe(env)
    done = probe.step(action)
    return not done or probe.death_cause == "solved"


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


def _bfs(env, start, goal, blocked):
    """Shortest path from start to goal over cells not in blocked. Returns the
    list of cells after start, or None if unreachable."""
    if start == goal:
        return []
    limit = env.grid_dim - 1
    seen = {start}
    queue = deque([(start, [])])
    while queue:
        (x, y), path = queue.popleft()
        for dx, dy in DELTAS:
            nxt = (x + dx, y + dy)
            if nxt in seen:
                continue
            if not (0 < nxt[0] < limit and 0 < nxt[1] < limit):
                continue
            if nxt in blocked and nxt != goal:
                continue
            if nxt == goal:
                return path + [nxt]
            seen.add(nxt)
            queue.append((nxt, path + [nxt]))
    return None


def _tail_reachable_after(env, action):
    """Whether, having taken this action, the head can still reach its own tail.

    This is the safety property that separates a BFS agent that solves boards
    from one that walks into a pocket and starves. The tail is excluded from the
    blocked set because it will have moved by the time the head arrives.
    """
    probe = _probe(env)
    if probe.step(action):
        return probe.death_cause == "solved"
    body = set(probe.snake)
    tail = probe.snake[-1]
    body.discard(tail)
    return _bfs(probe, probe.snake[0], tail, body) is not None


class BFSSafeAgent:
    """Head for the food along the shortest path, but only when doing so leaves
    the head able to reach its tail afterward. Otherwise stall by following the
    tail, which is always safe while the body forms a single connected path."""

    def __init__(self, rng):
        self.rng = rng

    def act(self, env):
        legal = _legal(env)
        head = env.snake[0]

        if env.food is not None:
            body = set(env.snake)
            body.discard(env.snake[-1])
            path = _bfs(env, head, env.food, body)
            if path:
                step = path[0]
                action = _action_between(head, step)
                if action in legal and _tail_reachable_after(env, action):
                    return action

        stalling = [a for a in legal if _tail_reachable_after(env, a)]
        if stalling:
            # Move away from the food while stalling, so the tail has time to
            # clear the route rather than being chased into a dead end.
            if env.food is None:
                return int(self.rng.choice(stalling))
            best = max(
                _distance((head[0] + DELTAS[a][0], head[1] + DELTAS[a][1]), env.food)
                for a in stalling
            )
            tied = [
                a
                for a in stalling
                if _distance((head[0] + DELTAS[a][0], head[1] + DELTAS[a][1]), env.food)
                == best
            ]
            return int(self.rng.choice(tied))

        safe = [a for a in legal if survivable(env, a)]
        return int(self.rng.choice(safe or legal))


def _action_between(origin, target):
    delta = (target[0] - origin[0], target[1] - origin[1])
    return DELTAS.index(delta)


class HamiltonianAgent:
    """Follow a fixed Hamiltonian cycle. Perfect by construction, so it is the
    ceiling every other agent is measured against.

    graph.HamiltonianCycle uses the global random module, which is the one
    documented exception to this package's no global randomness rule. The
    global state is saved and restored around the one call that needs it, so
    constructing this agent cannot change randomness anywhere else, such as
    app.py's menu, which imports from the same global module.
    """

    def __init__(self, size, seed):
        if size % 2 == 1:
            raise ValueError(
                f"size {size} is odd; a grid graph with both dimensions odd has "
                "no Hamiltonian cycle"
            )

        state = _stdlib_random.getstate()
        _stdlib_random.seed(seed)
        try:
            cycle = HamiltonianCycle(size, size, 1).cycle_positions()
        finally:
            _stdlib_random.setstate(state)

        # graph.py yields (row, col), both zero based over the playable area.
        # Playable coordinates are (x, y) = (col + 1, row + 1).
        self.cells = [(col + 1, row + 1) for row, col in cycle]
        self.order = {cell: i for i, cell in enumerate(self.cells)}

    def act(self, env):
        legal = _legal(env)
        head = env.snake[0]
        nxt = self.cells[(self.order[head] + 1) % len(self.cells)]
        action = _action_between(head, nxt)
        if action in legal and survivable(env, action):
            return action
        # The env starts the snake in a straight line that is not aligned to
        # the cycle, so the first move or two may deviate. Once the body
        # trails along the cycle the successor cell is always free.
        safe = [a for a in legal if survivable(env, a)]
        return int(safe[0] if safe else legal[0])
