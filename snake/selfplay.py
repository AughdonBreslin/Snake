"""Self play, value targets, and the replay buffer.

Positions are stored compactly and encoded at sample time rather than stored as
observation arrays. On a 20x20 board an observation is about 21 KB, so a 200k
buffer of them would need roughly 4 GB; the compact form is an order of
magnitude smaller and costs one cheap encode per sampled row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snake.arena import GameResult
from snake.encoding import (
    SYMMETRIES,
    encode,
    transform_observation,
    transform_policy,
)
from snake.env import SnakeEnv, default_starvation_limit
from snake.mcts import Search, run_search

_SEED_MAX = 2**63 - 1


@dataclass(frozen=True)
class Position:
    snake: tuple
    food: tuple | None
    direction: int
    size: int
    pi: np.ndarray
    z: float


def value_targets(lengths, final_length, total_cells):
    """z(s) is the fraction of the whole board still to be filled from s.

    Normalizing by the constant board area rather than by remaining capacity is
    what makes z the undiscounted return of an MDP with a reward of 1/C per
    food, which is exactly what search backs up. Dividing by remaining capacity
    would put parent and child in different units.
    """
    return [(final_length - length) / total_cells for length in lengths]


def select_move(pi, move_index, cfg, rng):
    tau = 1.0 if move_index < cfg.tau_threshold else cfg.tau_final
    if tau <= 0:
        return int(np.argmax(pi))
    weights = np.power(pi, 1.0 / tau)
    total = weights.sum()
    if total <= 0:
        weights = pi
        total = weights.sum()
    return int(rng.choice(4, p=weights / total))


def play_batch(cfg, evaluator, rng):
    """Play cfg.train.concurrent_games games in lockstep, batching every leaf
    evaluation at a given simulation index into one call."""
    size = cfg.train.board_size
    total_cells = size * size
    limit = default_starvation_limit(size)

    envs = [
        SnakeEnv(size, np.random.default_rng(int(rng.integers(_SEED_MAX))), limit)
        for _ in range(cfg.train.concurrent_games)
    ]
    pending = [[] for _ in envs]      # (snake, food, direction, pi, length) per move
    steps = [0] * len(envs)
    results = [None] * len(envs)

    active = list(range(len(envs)))
    move_index = 0
    while active:
        # A fresh tree every move. There is no tree reuse in this design.
        searches = {
            index: Search(
                envs[index], cfg.search, np.random.default_rng(int(rng.integers(_SEED_MAX)))
            )
            for index in active
        }
        run_search(list(searches.values()), evaluator, cfg.search.simulations)

        still_active = []
        for index in active:
            env = envs[index]
            counts = searches[index].visit_counts()
            total = counts.sum()
            pi = counts / total if total > 0 else env.legal_actions() / env.legal_actions().sum()
            # The length is recorded before the step, because z is measured from
            # this state. Recording it after would shift every target by a move.
            pending[index].append(
                (tuple(env.snake), env.food, env.direction, pi.astype(np.float32), env.length)
            )
            action = select_move(pi, move_index, cfg.search, rng)
            done = env.step(action)
            steps[index] += 1
            if done:
                results[index] = GameResult(
                    score=env.score,
                    length=env.length,
                    steps=steps[index],
                    outcome=env.death_cause,
                )
            else:
                still_active.append(index)
        active = still_active
        move_index += 1

    positions = []
    for index, env in enumerate(envs):
        lengths = [row[4] for row in pending[index]]
        targets = value_targets(lengths, env.length, total_cells)
        for (snake, food, direction, pi, _), z in zip(pending[index], targets, strict=True):
            positions.append(
                Position(
                    snake=snake, food=food, direction=direction,
                    size=size, pi=pi, z=z,
                )
            )
    return positions, results


class ReplayBuffer:
    """A ring buffer over positions.

    A deque would be the obvious choice, but indexing one is linear, and sampling
    a batch of 512 from a 200k buffer would then cost tens of millions of steps
    per batch. A list plus a write cursor keeps indexing constant time.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self._items = []
        self._cursor = 0

    def __len__(self):
        return len(self._items)

    def items(self):
        return list(self._items)

    def load(self, items):
        self._items = list(items)[-self.capacity :]
        # Index 0 holds the oldest entry after a load, whether the load filled
        # the buffer or left room for appends, so the next eviction starts there.
        self._cursor = 0

    def extend(self, positions):
        for position in positions:
            if len(self._items) < self.capacity:
                self._items.append(position)
            else:
                self._items[self._cursor] = position
                self._cursor = (self._cursor + 1) % self.capacity

    def sample(self, batch_size, rng):
        """Draw a batch, applying one of the eight dihedral symmetries per row.

        Augmenting at sample time rather than at store time keeps the buffer
        eight times smaller and still shows the network every symmetry.
        """
        indices = rng.integers(0, len(self._items), size=batch_size)
        symmetry_choices = rng.integers(0, len(SYMMETRIES), size=batch_size)

        obs_batch, pi_batch, z_batch = [], [], []
        for index, symmetry_index in zip(indices, symmetry_choices, strict=True):
            item = self._items[int(index)]  # constant time, see the class docstring
            k, flip = SYMMETRIES[int(symmetry_index)]
            obs = encode(item.size, item.snake, item.food, item.direction)
            obs_batch.append(transform_observation(obs, k, flip))
            pi_batch.append(transform_policy(item.pi, k, flip))
            z_batch.append(item.z)

        return (
            np.stack(obs_batch).astype(np.float32),
            np.stack(pi_batch).astype(np.float32),
            np.array(z_batch, dtype=np.float32),
        )
