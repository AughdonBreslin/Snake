"""Headless snake rules. No pygame, no torch, no global randomness.

Coordinates are (x, y) with the origin at the top left of the full grid, which
matches app.py. The full grid is (size + 2) square: a one cell wall border
surrounds a size by size playable area, so playable coordinates run from 1 to
size inclusive on both axes.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from snake.encoding import encode

RIGHT, DOWN, LEFT, UP = 0, 1, 2, 3

# (dx, dy) indexed by action. y increases downward, matching app.py.
DELTAS = ((1, 0), (0, 1), (-1, 0), (0, -1))
OPPOSITE = (LEFT, UP, RIGHT, DOWN)


def default_starvation_limit(size: int) -> int:
    """Steps allowed without eating before an episode is truncated."""
    return 2 * size * size


class SnakeEnv:
    def __init__(self, size, rng, starvation_limit=None):
        if size < 3:
            raise ValueError(f"size must be at least 3, got {size}")
        self.size = size
        self.grid_dim = size + 2
        self.rng = rng
        self.starvation_limit = starvation_limit
        self.reset()

    # Construction and copying

    def reset(self):
        mid = self.grid_dim // 2
        self.snake = deque([(3, mid), (2, mid), (1, mid)])
        self._occupied = set(self.snake)
        self.direction = RIGHT
        self.score = 0
        self.game_over = False
        self.death_cause = None
        self.steps_since_food = 0
        self.food = self._spawn_food()

    def clone(self, rng):
        """Copy the state. The caller supplies the generator, so two envs can
        never share one and advance each other's stream."""
        twin = SnakeEnv.__new__(SnakeEnv)
        twin.size = self.size
        twin.grid_dim = self.grid_dim
        twin.rng = rng
        twin.starvation_limit = self.starvation_limit
        twin.snake = deque(self.snake)
        twin._occupied = set(self._occupied)
        twin.direction = self.direction
        twin.score = self.score
        twin.game_over = self.game_over
        twin.death_cause = self.death_cause
        twin.steps_since_food = self.steps_since_food
        twin.food = self.food
        return twin

    # Properties

    @property
    def length(self):
        return len(self.snake)

    @property
    def total_cells(self):
        return self.size * self.size

    # Rules

    def _spawn_food(self):
        if self.total_cells - len(self.snake) <= 0:
            return None
        while True:
            cell = (
                int(self.rng.integers(1, self.size + 1)),
                int(self.rng.integers(1, self.size + 1)),
            )
            if cell not in self._occupied:
                return cell

    def step(self, action):
        """Apply one move. Returns whether the episode ended."""
        if self.game_over:
            raise RuntimeError("step() called on a finished episode")
        if action == OPPOSITE[self.direction]:
            raise ValueError(
                f"action {action} reverses direction {self.direction}; "
                "callers must respect legal_actions()"
            )

        self.direction = action
        dx, dy = DELTAS[action]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy

        # The tail is popped before the collision check, so the head may legally
        # move into the cell the tail is vacating. This mirrors app.py:375.
        tail = self.snake.pop()
        self._occupied.discard(tail)

        if not (0 < nx < self.grid_dim - 1 and 0 < ny < self.grid_dim - 1):
            return self._die("wall", tail)
        if (nx, ny) in self._occupied:
            return self._die("self", tail)

        self.snake.appendleft((nx, ny))
        self._occupied.add((nx, ny))

        if (nx, ny) == self.food:
            self.snake.append(tail)
            self._occupied.add(tail)
            self.score += 1
            self.steps_since_food = 0
            self.food = self._spawn_food()
            if self.food is None:
                self.game_over = True
                self.death_cause = "solved"
                return True
            return False

        self.steps_since_food += 1
        if (
            self.starvation_limit is not None
            and self.steps_since_food >= self.starvation_limit
        ):
            self.game_over = True
            self.death_cause = "starvation"
            return True
        return False

    def _die(self, cause, tail):
        # Put the tail back so length reports the length the snake died at.
        self.snake.append(tail)
        self._occupied.add(tail)
        self.game_over = True
        self.death_cause = cause
        return True

    # Agent interface

    def legal_actions(self):
        """Only reversal is illegal. Fatal moves stay legal, because removing
        them would delete the learning signal about death."""
        mask = np.ones(4, dtype=bool)
        mask[OPPOSITE[self.direction]] = False
        return mask

    def would_eat(self, action):
        """Whether this action lands on food. Deterministic given the current
        state, which is what lets search treat the food reward as known."""
        if self.food is None:
            return False
        dx, dy = DELTAS[action]
        hx, hy = self.snake[0]
        return (hx + dx, hy + dy) == self.food

    def observation(self):
        return encode(self.size, self.snake, self.food, self.direction)
