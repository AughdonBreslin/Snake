# AlphaZero Snake, Phases 0 and 1, Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless snake environment with interpretable baselines, then a single-player AlphaZero learning loop that beats the greedy baseline on a 6x6 board.

**Architecture:** `snake/env.py` becomes the single definition of the game rules and `app.py` delegates to it, so the agent trains on the rules the game ships. Search is open-loop determinized PUCT: nodes are keyed by action path and every simulation replays from the root with a fresh RNG, which averages over the random food spawn instead of committing to one sampled future. Self-play runs many games in lockstep so the leaf evaluations at each simulation index batch into a single GPU forward.

**Tech Stack:** Python 3.12, numpy, pygame (rendering only), PyTorch, TensorBoard, pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-alphazero-snake-design.md`

## Global Constraints

- Python 3.12. All work happens inside `venv/` at the repo root, which is already gitignored.
- `snake/env.py`, `snake/encoding.py`, `snake/baselines.py`, `snake/arena.py`, and `snake/config.py` must not import `torch` or `pygame`. `app.py` must keep running with only numpy and pygame installed.
- `requirements.txt` stays at numpy and pygame. Training dependencies go in `requirements-train.txt`, test dependencies in `requirements-dev.txt`.
- Observations are `float32` of shape `[11, size + 2, size + 2]`, indexed `[channel, y, x]`.
- Actions are absolute: `RIGHT = 0`, `DOWN = 1`, `LEFT = 2`, `UP = 3`. Deltas are `(dx, dy)`.
- Positions are `(x, y)` with the origin at the top-left of the full grid, matching `app.py`. The playable area is `1 <= x <= size` and `1 <= y <= size`.
- The value target is `z(s) = (final_length - length(s)) / total_cells`, where `total_cells = size * size`.
- Normalization layers are `GroupNorm`. `BatchNorm` must not appear anywhere.
- Every source of randomness is seeded explicitly. No use of the global `random` module or `np.random` legacy functions in `snake/`, with one documented exception in the Hamiltonian baseline, which wraps `graph.py`.
- No em-dashes anywhere, including code comments and docstrings. No bold or italic markup in generated documents.
- Commit after every task. Do not merge to `main`.

## Deviations from the spec, agreed before planning

These are recorded so a reviewer does not read them as mistakes.

1. The spec's section 16 describes a step-for-step equivalence test between `SnakeGame` and `SnakeEnv`. After the refactor in Task 6, `SnakeGame` delegates to `SnakeEnv`, so such a test would be tautological. The protection is instead Task 1: rule behavior is pinned against the current `app.py` before any refactor, and those same tests must still pass after it. Task 6 adds a delegation test proving no rule logic was left behind in `SnakeGame`.
2. The spec's section 12 lists three death causes. A fourth, `solved`, is added, because a full board is a distinct outcome and the Hamiltonian baseline reaches it every game.
3. The spec's section 10.2 places batched leaf evaluation in Phase 1, and this plan implements it there (Task 16 and Task 17), because single-leaf inference makes a 6x6 iteration take roughly an hour. Multi-process self-play remains a Phase 2 item.
4. `SnakeEnv` restores the popped tail when a move is fatal, so `length` after game over is the length the snake died at. `app.py` does not restore it, but `app.py` never reads the body after game over, so this is invisible to rendering.

---

## Phase 0: environment and measuring stick

### Task 1: Project scaffolding and pinned rule tests

Pin the behavior of the current `app.py` before touching it. Every assertion here must pass against `app.py` exactly as it stands today, and must still pass after Task 6.

**Files:**
- Create: `requirements-dev.txt`
- Create: `pytest.ini`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_app_rules_pinned.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a `game` pytest fixture yielding a fresh `app.SnakeGame` on a headless pygame display, used by Task 6.

- [ ] **Step 1: Create the virtualenv and install dependencies**

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
printf 'pytest==8.3.4\n' > requirements-dev.txt
./venv/bin/pip install -r requirements-dev.txt
```

- [ ] **Step 2: Configure pytest**

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
addopts = -q
```

Create empty `tests/__init__.py`.

- [ ] **Step 3: Write the headless pygame fixture**

`app.py` builds fonts and a display at import and construction time, so tests need a dummy video driver. Create `tests/conftest.py`:

```python
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest


@pytest.fixture(scope="session")
def window():
    pygame.init()
    surface = pygame.display.set_mode((440, 440), pygame.NOFRAME)
    yield surface
    pygame.quit()


@pytest.fixture
def game(window):
    import app

    g = app.SnakeGame(window)
    g.reset()
    return g
```

- [ ] **Step 4: Write the failing pinned rule tests**

Create `tests/test_app_rules_pinned.py`. These describe the rules as they are today, in terms of `SnakeGame`'s public surface only:

```python
import app
from app import DOWN, LEFT, RIGHT, UP


def test_reset_places_three_cells_facing_right(game):
    mid = app.GRID_HEIGHT // 2
    assert game.snake == [(3, mid), (2, mid), (1, mid)]
    assert game.direction == RIGHT
    assert game.score == 0


def test_straight_move_shifts_head_and_drops_tail(game):
    mid = app.GRID_HEIGHT // 2
    game.food = (10, 1)
    over, _ = game.move()
    assert over is False
    assert game.snake == [(4, mid), (3, mid), (2, mid)]


def test_walking_into_the_right_wall_ends_the_game(game):
    mid = app.GRID_HEIGHT // 2
    game.food = (10, 1)
    game.snake = [(app.GRID_WIDTH - 2, mid)]
    over, _ = game.move()
    assert over is True


def test_walking_into_the_top_wall_ends_the_game(game):
    game.food = (10, 5)
    game.snake = [(5, 1)]
    game.direction = UP
    game.move_queue = []
    over, _ = game.move()
    assert over is True


def test_eating_grows_the_snake_and_scores(game):
    mid = app.GRID_HEIGHT // 2
    game.food = (4, mid)
    over, _ = game.move()
    assert over is False
    assert game.score == 1
    assert len(game.snake) == 4
    assert game.snake[0] == (4, mid)


def test_new_food_never_lands_on_the_snake(game):
    for _ in range(200):
        assert game.spawn_food() not in game.snake


def test_head_may_enter_the_cell_the_tail_is_vacating(game):
    # A closed loop of four cells. Moving down from (5, 5) targets (5, 6), which
    # is the tail, and that is legal because the tail is popped before the
    # collision check. This is app.py:375 and it is the single most load bearing
    # rule in the game.
    game.snake = [(5, 5), (6, 5), (6, 6), (5, 6)]
    game.direction = DOWN
    game.move_queue = []
    game.food = (12, 12)
    over, _ = game.move()
    assert over is False
    assert game.snake[0] == (5, 6)


def test_running_into_the_body_ends_the_game(game):
    # Head (5, 5) moving down targets (5, 6), which is a mid body cell and stays
    # occupied after the tail (6, 6) is popped.
    game.snake = [(5, 5), (4, 5), (3, 5), (3, 6), (4, 6), (5, 6), (6, 6)]
    game.direction = DOWN
    game.move_queue = []
    game.food = (12, 12)
    over, _ = game.move()
    assert over is True
```

- [ ] **Step 5: Run the pinned tests**

```bash
./venv/bin/python -m pytest tests/test_app_rules_pinned.py -v
```

Expected: all PASS. These describe existing behavior, so a failure means the assertion is wrong about `app.py`, not that `app.py` is wrong. Read `app.py:361-391` and fix the test.

- [ ] **Step 6: Commit**

```bash
git add requirements-dev.txt pytest.ini tests/
git commit -m "test: pin current snake rule behavior before refactor"
```

---

### Task 2: SnakeEnv core rules

**Files:**
- Create: `snake/__init__.py`
- Create: `snake/env.py`
- Create: `tests/test_env_rules.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `snake.env.RIGHT, DOWN, LEFT, UP` as ints 0, 1, 2, 3
  - `snake.env.DELTAS: tuple[tuple[int, int], ...]`, four `(dx, dy)` pairs indexed by action
  - `snake.env.OPPOSITE: tuple[int, ...]`, the reversal of each action
  - `SnakeEnv(size: int, rng: np.random.Generator, starvation_limit: int | None = None)`
  - `SnakeEnv.reset() -> None`
  - `SnakeEnv.step(action: int) -> bool`, returns whether the episode ended
  - `SnakeEnv.clone(rng: np.random.Generator) -> SnakeEnv`
  - `SnakeEnv.snake: collections.deque[tuple[int, int]]`, head first
  - `SnakeEnv.food: tuple[int, int] | None`
  - `SnakeEnv.direction: int`, `SnakeEnv.score: int`, `SnakeEnv.game_over: bool`
  - `SnakeEnv.death_cause: str | None`, one of `"wall"`, `"self"`, `"starvation"`, `"solved"`
  - `SnakeEnv.length: int`, `SnakeEnv.total_cells: int`, `SnakeEnv.grid_dim: int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_env_rules.py`:

```python
import numpy as np
import pytest

from snake.env import DOWN, LEFT, RIGHT, UP, SnakeEnv


def make(size=8, seed=0, starvation_limit=None):
    return SnakeEnv(size, np.random.default_rng(seed), starvation_limit)


def test_reset_matches_app_layout():
    env = make(size=20)
    mid = 22 // 2
    assert list(env.snake) == [(3, mid), (2, mid), (1, mid)]
    assert env.direction == RIGHT
    assert env.score == 0
    assert env.length == 3
    assert env.total_cells == 400
    assert env.grid_dim == 22


def test_straight_move_shifts_head_and_drops_tail():
    env = make()
    env.food = (7, 7)
    mid = env.grid_dim // 2
    assert env.step(RIGHT) is False
    assert list(env.snake) == [(4, mid), (3, mid), (2, mid)]


def test_wall_collision_reports_wall():
    env = make(size=5)
    env.food = (5, 5)
    env.snake.clear()
    env.snake.append((5, 3))
    env._occupied = {(5, 3)}
    assert env.step(RIGHT) is True
    assert env.death_cause == "wall"


def test_self_collision_reports_self():
    # Head (5, 5) facing right. Turning down targets (5, 6), a mid body cell that
    # is still occupied after the tail (6, 6) is popped.
    env = make()
    env.food = (7, 7)
    env.snake.clear()
    env.snake.extend([(5, 5), (4, 5), (3, 5), (3, 6), (4, 6), (5, 6), (6, 6)])
    env._occupied = set(env.snake)
    env.direction = RIGHT
    assert env.step(DOWN) is True
    assert env.death_cause == "self"


def test_head_may_enter_the_cell_the_tail_is_vacating():
    env = make()
    env.food = (7, 7)
    env.snake.clear()
    env.snake.extend([(5, 5), (6, 5), (6, 6), (5, 6)])
    env._occupied = set(env.snake)
    env.direction = RIGHT
    assert env.step(DOWN) is False
    assert env.snake[0] == (5, 6)


def test_eating_grows_and_scores():
    env = make()
    mid = env.grid_dim // 2
    env.food = (4, mid)
    assert env.step(RIGHT) is False
    assert env.score == 1
    assert env.length == 4
    assert env.food != (4, mid)


def test_food_never_spawns_on_the_body():
    env = make(size=4, seed=3)
    for _ in range(500):
        assert env._spawn_food() not in set(env.snake)


def test_fatal_move_preserves_length():
    env = make(size=5)
    env.food = (5, 5)
    env.snake.clear()
    env.snake.extend([(5, 3), (4, 3), (3, 3)])
    env._occupied = set(env.snake)
    assert env.step(RIGHT) is True
    assert env.length == 3


def test_clone_is_independent():
    env = make()
    twin = env.clone(np.random.default_rng(1))
    twin.step(RIGHT)
    twin.step(RIGHT)
    assert env.length == 3
    assert list(env.snake) == [(3, 5), (2, 5), (1, 5)]
    assert twin.snake[0] != env.snake[0]


def test_clone_does_not_share_the_generator():
    env = make(size=4, seed=7)
    before = env.rng.bit_generator.state
    twin = env.clone(np.random.default_rng(99))
    for _ in range(20):
        if not twin.game_over:
            twin.step(twin.direction)
    assert env.rng.bit_generator.state == before


def test_step_on_a_finished_episode_raises():
    env = make(size=5)
    env.snake.clear()
    env.snake.append((5, 3))
    env._occupied = {(5, 3)}
    env.step(RIGHT)
    with pytest.raises(RuntimeError):
        env.step(RIGHT)


def test_size_below_three_is_rejected():
    with pytest.raises(ValueError):
        make(size=2)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/test_env_rules.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'snake'`.

- [ ] **Step 3: Write the implementation**

Create empty `snake/__init__.py`, then `snake/env.py`:

```python
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
```

Note that `snake/env.py` imports `encode` from `snake/encoding.py`, which does not exist until Task 5. Create a placeholder now so imports resolve:

```bash
cat > snake/encoding.py <<'EOF'
"""Observation planes. Filled in by Task 5."""


def encode(size, snake, food, direction):
    raise NotImplementedError("implemented in Task 5")
EOF
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_env_rules.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add snake/ tests/test_env_rules.py
git commit -m "feat: add headless SnakeEnv with app.py's rules"
```

---

### Task 3: Legal actions and the would-eat query

Task 2 shipped both methods. This task proves their contract, because search correctness depends on both and neither is exercised by the rule tests.

**Files:**
- Create: `tests/test_env_actions.py`

**Interfaces:**
- Consumes: `SnakeEnv.legal_actions`, `SnakeEnv.would_eat`, `snake.env.OPPOSITE` from Task 2.
- Produces: nothing new.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_env_actions.py`:

```python
import numpy as np

from snake.env import DELTAS, DOWN, LEFT, OPPOSITE, RIGHT, UP, SnakeEnv


def make(size=8, seed=0):
    return SnakeEnv(size, np.random.default_rng(seed))


def test_only_the_reversal_is_masked():
    env = make()
    for direction in (RIGHT, DOWN, LEFT, UP):
        env.direction = direction
        mask = env.legal_actions()
        assert mask.sum() == 3
        assert not mask[OPPOSITE[direction]]


def test_fatal_moves_stay_legal():
    # The head is against the right wall. Moving right is fatal but still legal,
    # because masking it away would remove the signal that it kills you.
    env = make(size=5)
    env.snake.clear()
    env.snake.append((5, 3))
    env._occupied = {(5, 3)}
    env.direction = RIGHT
    assert env.legal_actions()[RIGHT]


def test_would_eat_is_true_only_for_the_food_cell():
    env = make()
    head = env.snake[0]
    for action in (RIGHT, DOWN, UP):
        dx, dy = DELTAS[action]
        env.food = (head[0] + dx, head[1] + dy)
        assert env.would_eat(action) is True
        others = [a for a in (RIGHT, DOWN, UP) if a != action]
        assert all(env.would_eat(a) is False for a in others)


def test_would_eat_is_false_on_a_solved_board():
    env = make()
    env.food = None
    assert all(env.would_eat(a) is False for a in range(4))


def test_would_eat_predicts_the_score_change():
    rng = np.random.default_rng(11)
    env = SnakeEnv(6, rng)
    for _ in range(300):
        if env.game_over:
            break
        legal = [a for a in range(4) if env.legal_actions()[a]]
        action = int(rng.choice(legal))
        predicted = env.would_eat(action)
        before = env.score
        done = env.step(action)
        if not done or env.death_cause == "solved":
            assert (env.score > before) == predicted
```

- [ ] **Step 2: Run the tests**

```bash
./venv/bin/python -m pytest tests/test_env_actions.py -v
```

Expected: all PASS, since Task 2 implemented both methods. If `test_only_the_reversal_is_masked` fails on the `is np.False_` comparison, the mask dtype is wrong; it must be `bool`, not `int`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_env_actions.py
git commit -m "test: pin legal action masking and the would-eat query"
```

---

### Task 4: Starvation cap

**Files:**
- Modify: `snake/env.py` (already implements the cap; this task only adds tests)
- Create: `tests/test_env_starvation.py`

**Interfaces:**
- Consumes: `SnakeEnv`, `default_starvation_limit` from Task 2.
- Produces: nothing new.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_env_starvation.py`:

```python
import numpy as np

from snake.env import DOWN, LEFT, RIGHT, SnakeEnv, default_starvation_limit


def test_default_limit_is_twice_the_board():
    assert default_starvation_limit(6) == 72
    assert default_starvation_limit(20) == 800


def test_no_limit_means_no_truncation():
    env = SnakeEnv(8, np.random.default_rng(0), starvation_limit=None)
    env.food = (8, 8)
    steps = 0
    while not env.game_over and steps < 500:
        env.step(_circle(steps))
        steps += 1
    assert env.death_cause != "starvation"


def test_the_limit_truncates_and_reports_starvation():
    env = SnakeEnv(8, np.random.default_rng(0), starvation_limit=10)
    env.food = (8, 8)
    steps = 0
    while not env.game_over:
        env.step(_circle(steps))
        steps += 1
        assert steps <= 11
    assert env.death_cause == "starvation"
    assert env.steps_since_food == 10


def test_eating_resets_the_starvation_counter():
    env = SnakeEnv(8, np.random.default_rng(0), starvation_limit=10)
    mid = env.grid_dim // 2
    env.food = (7, mid)
    for _ in range(3):
        env.step(RIGHT)
    assert env.steps_since_food == 3
    env.food = (env.snake[0][0] + 1, mid)
    env.step(RIGHT)
    assert env.steps_since_food == 0


def _circle(step):
    # A four move loop that never eats: right, down, left, up is illegal as a
    # sequence because up reverses down, so use a wider rectangle.
    return (RIGHT, RIGHT, DOWN, LEFT, LEFT, DOWN, RIGHT, RIGHT)[step % 8]
```

- [ ] **Step 2: Run the tests**

```bash
./venv/bin/python -m pytest tests/test_env_starvation.py -v
```

Expected: all PASS. If `_circle` raises `ValueError` about reversing, the sequence is wrong; every consecutive pair must differ by a turn, not a reversal.

- [ ] **Step 3: Commit**

```bash
git add tests/test_env_starvation.py
git commit -m "test: pin the starvation truncation rule"
```

---

### Task 5: Observation encoding

**Files:**
- Modify: `snake/encoding.py` (replace the Task 2 placeholder)
- Create: `tests/test_encoding.py`

**Interfaces:**
- Consumes: `snake.env.DELTAS` for the direction convention.
- Produces:
  - `snake.encoding.N_PLANES: int` equal to 11
  - `snake.encoding.encode(size: int, snake, food, direction) -> np.ndarray` of shape `[11, size + 2, size + 2]`, dtype `float32`. `snake` is any head-first sequence of `(x, y)`, `food` is `(x, y)` or `None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_encoding.py`:

```python
import numpy as np

from snake.encoding import N_PLANES, encode
from snake.env import RIGHT, UP, SnakeEnv


def sample():
    snake = [(3, 2), (2, 2), (1, 2)]
    return encode(size=4, snake=snake, food=(4, 4), direction=RIGHT), snake


def test_shape_and_dtype():
    obs, _ = sample()
    assert obs.shape == (N_PLANES, 6, 6)
    assert obs.dtype == np.float32


def test_wall_plane_is_the_border_only():
    obs, _ = sample()
    walls = obs[0]
    assert walls[0].all() and walls[-1].all()
    assert walls[:, 0].all() and walls[:, -1].all()
    assert walls[1:-1, 1:-1].sum() == 0


def test_body_plane_excludes_the_head():
    obs, snake = sample()
    hx, hy = snake[0]
    assert obs[1, hy, hx] == 0.0
    for x, y in snake[1:]:
        assert obs[1, y, x] == 1.0
    assert obs[1].sum() == len(snake) - 1


def test_head_and_food_and_tail_planes():
    obs, snake = sample()
    assert obs[2].sum() == 1.0
    assert obs[2, snake[0][1], snake[0][0]] == 1.0
    assert obs[3].sum() == 1.0
    assert obs[3, 4, 4] == 1.0
    assert obs[4].sum() == 1.0
    assert obs[4, snake[-1][1], snake[-1][0]] == 1.0


def test_missing_food_gives_an_empty_plane():
    obs = encode(size=4, snake=[(3, 2), (2, 2)], food=None, direction=RIGHT)
    assert obs[3].sum() == 0.0


def test_body_age_counts_down_from_the_head():
    obs, snake = sample()
    length = len(snake)
    for i, (x, y) in enumerate(snake):
        assert obs[5, y, x] == np.float32((length - i) / length)
    assert obs[5, snake[0][1], snake[0][0]] == 1.0
    assert obs[5, snake[-1][1], snake[-1][0]] == np.float32(1 / length)


def test_heading_planes_are_one_hot_and_constant():
    for direction in range(4):
        obs = encode(4, [(3, 2), (2, 2)], (4, 4), direction)
        active = [p for p in range(6, 10) if obs[p].any()]
        assert active == [6 + direction]
        assert obs[6 + direction].all()


def test_fill_fraction_plane():
    obs, snake = sample()
    assert np.allclose(obs[10], len(snake) / 16)


def test_env_observation_matches_encode():
    env = SnakeEnv(6, np.random.default_rng(0))
    assert np.array_equal(
        env.observation(),
        encode(env.size, env.snake, env.food, env.direction),
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/test_encoding.py -v
```

Expected: FAIL with `NotImplementedError: implemented in Task 5`.

- [ ] **Step 3: Write the implementation**

Replace `snake/encoding.py`:

```python
"""Observation planes and dihedral symmetry. No torch, no pygame.

Planes are indexed [channel, y, x]. Every plane is board size invariant in
meaning, which is what lets one set of network weights serve every board in the
curriculum.

  0      walls, the one cell border
  1      body excluding the head
  2      head
  3      food
  4      tail cell
  5      body age: steps until that cell vacates, divided by current length
  6..9   heading, one hot as four constant planes
  10     fill fraction, constant, length divided by total playable cells
"""

from __future__ import annotations

import numpy as np

N_PLANES = 11
HEADING_PLANE_0 = 6


def encode(size, snake, food, direction):
    dim = size + 2
    obs = np.zeros((N_PLANES, dim, dim), dtype=np.float32)

    obs[0, 0, :] = 1.0
    obs[0, dim - 1, :] = 1.0
    obs[0, :, 0] = 1.0
    obs[0, :, dim - 1] = 1.0

    length = len(snake)
    for i, (x, y) in enumerate(snake):
        if i > 0:
            obs[1, y, x] = 1.0
        # The cell at index i vacates in (length - i) steps.
        obs[5, y, x] = (length - i) / length

    hx, hy = snake[0]
    obs[2, hy, hx] = 1.0

    if food is not None:
        fx, fy = food
        obs[3, fy, fx] = 1.0

    tx, ty = snake[-1]
    obs[4, ty, tx] = 1.0

    obs[HEADING_PLANE_0 + direction, :, :] = 1.0
    obs[10, :, :] = length / (size * size)

    return obs
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_encoding.py -v
```

Expected: all PASS.

- [ ] **Step 5: Run the whole suite**

```bash
./venv/bin/python -m pytest -v
```

Expected: all PASS, including the Task 2 tests that were exercising the placeholder only indirectly.

- [ ] **Step 6: Commit**

```bash
git add snake/encoding.py tests/test_encoding.py
git commit -m "feat: add the 11 plane observation encoding"
```

---

### Task 6: Wire app.py to SnakeEnv

`SnakeGame` keeps rendering, fonts, score text, highscore persistence, and keyboard handling. Every rule decision moves to `SnakeEnv`.

**Files:**
- Modify: `app.py:294-442` (the `SnakeGame` class)
- Create: `tests/test_app_delegates.py`

**Interfaces:**
- Consumes: `SnakeEnv`, `snake.env.RIGHT/DOWN/LEFT/UP` from Task 2.
- Produces: `app.SnakeGame.env: SnakeEnv`, plus `snake`, `food`, `score`, `game_over`, and `direction` as read-only properties forwarding to it.

- [ ] **Step 1: Write the failing delegation test**

Create `tests/test_app_delegates.py`:

```python
import inspect

import numpy as np

import app
from snake.env import RIGHT, SnakeEnv


def test_snakegame_holds_an_env(game):
    assert isinstance(game.env, SnakeEnv)
    assert game.env.size == app.GRID_WIDTH - 2


def test_snakegame_has_no_rule_logic_left():
    # If any of these names still exist on SnakeGame, rules were left behind and
    # the two definitions will drift.
    source = inspect.getsource(app.SnakeGame)
    for leftover in ("def spawn_food", "GRID_WIDTH - 1", "self.snake.pop()"):
        assert leftover not in source, f"rule logic still in SnakeGame: {leftover}"


def test_the_renderer_never_truncates_on_starvation(game):
    assert game.env.starvation_limit is None


def test_delegation_reproduces_a_bare_env(window):
    # Same seed, same actions, identical trajectory. This proves SnakeGame adds
    # no rule behavior of its own.
    reference = SnakeEnv(app.GRID_WIDTH - 2, np.random.default_rng(1234))
    g = app.SnakeGame(window)
    g.reset_with_env(SnakeEnv(app.GRID_WIDTH - 2, np.random.default_rng(1234)))

    rng = np.random.default_rng(5)
    for _ in range(400):
        if reference.game_over:
            break
        legal = [a for a in range(4) if reference.legal_actions()[a]]
        action = int(rng.choice(legal))
        reference.step(action)
        g.step_action(action)
        assert list(g.snake) == list(reference.snake)
        assert g.food == reference.food
        assert g.score == reference.score
        assert g.game_over == reference.game_over
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./venv/bin/python -m pytest tests/test_app_delegates.py -v
```

Expected: FAIL with `AttributeError: 'SnakeGame' object has no attribute 'env'`.

- [ ] **Step 3: Rewrite SnakeGame's rule surface**

In `app.py`, add the import near the top:

```python
from snake.env import DOWN, LEFT, RIGHT, UP, SnakeEnv
```

Delete the `Direction` and `Action` enums and the `RIGHT, DOWN, LEFT, UP = ...` line at `app.py:268-292`. They are replaced by the plain ints from `snake.env`, which is what removes the `Direction.__add__` returning an `int` defect described in the spec's section 2.

Replace `SnakeGame.reset`, `spawn_food`, `move`, `step`, `get_state`, and `get_valid_inputs` with:

```python
    def reset(self):
        self.reset_with_env(
            SnakeEnv(GRID_WIDTH - 2, np.random.default_rng(), starvation_limit=None)
        )

    def reset_with_env(self, env):
        self.env = env
        self.move_queue = []
        self.score_text = self.font.render(str(self.score), True, (255, 255, 255))

    @property
    def snake(self):
        return self.env.snake

    @property
    def food(self):
        return self.env.food

    @property
    def score(self):
        return self.env.score

    @property
    def game_over(self):
        return self.env.game_over

    @property
    def direction(self):
        return self.env.direction

    def step_action(self, action):
        """Apply one move and refresh the score text. Returns whether the game
        ended. All rules live in SnakeEnv."""
        if self.env.game_over:
            return True
        done = self.env.step(action)
        self.score_text = self.font.render(str(self.score), True, (255, 255, 255))
        if self.score > self.highscore:
            self.highscore = self.score
            self.highscore_text = self.font.render(
                f"Highscore: {self.highscore}", True, (255, 255, 255)
            )
        return done

    def move(self):
        action = self.move_queue.pop(0) if self.move_queue else self.env.direction
        return self.step_action(action)
```

Update `SnakeGame.event` at `app.py:393-404` so the guards compare against the new int constants rather than the deleted enum members. The four branches become, with `RIGHT` shown and the rest following the same shape:

```python
        elif (key == pygame.K_RIGHT or key == pygame.K_d) and (
            (not self.move_queue and self.direction != LEFT)
            or (self.move_queue and self.move_queue[-1] != LEFT)
        ):
            self.move_queue.append(RIGHT)
```

Update `SnakeGame.play` at `app.py:435-442`, which previously unpacked a two element return:

```python
    def play(self):
        if not self.game_over:
            self.move()
        if self.game_over:
            self.write_highscore(self.highscore)
            self.draw_game_over()
        else:
            self.draw_background()
```

- [ ] **Step 4: Run the delegation tests and the pinned tests together**

```bash
./venv/bin/python -m pytest tests/test_app_delegates.py tests/test_app_rules_pinned.py -v
```

Expected: all PASS. The pinned tests from Task 1 assign to `game.snake` and `game.food` directly, which the new properties do not allow. Fix them by assigning to `game.env.snake`, `game.env.food`, and `game.env.direction` instead, and by rebuilding `game.env._occupied` after any direct body assignment. The assertions themselves must not change: they are the contract, and changing one means the refactor altered a rule.

- [ ] **Step 5: Play the game by hand**

```bash
./venv/bin/python app.py
```

Confirm: the snake moves, arrow keys and WASD steer, eating increases the score, hitting a wall or the body ends the game, Enter restarts, Escape returns to the menu, and Q and Alt still reach settings and leaderboards.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app_delegates.py tests/test_app_rules_pinned.py
git commit -m "refactor: app.py delegates all snake rules to SnakeEnv"
```

---

### Task 7: Random and greedy baselines

**Files:**
- Create: `snake/baselines.py`
- Create: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `SnakeEnv`, `DELTAS`, `legal_actions`, `clone` from Task 2.
- Produces:
  - `snake.baselines.Agent` protocol: `act(env: SnakeEnv) -> int`
  - `RandomAgent(rng: np.random.Generator)`
  - `GreedyAgent(rng: np.random.Generator)`
  - `snake.baselines.survivable(env, action) -> bool`, whether the action is not immediately fatal

- [ ] **Step 1: Write the failing tests**

Create `tests/test_baselines.py`:

```python
import numpy as np

from snake.baselines import GreedyAgent, RandomAgent, survivable
from snake.env import RIGHT, SnakeEnv


def run(agent, size=6, seed=0, limit=2000):
    env = SnakeEnv(size, np.random.default_rng(seed), starvation_limit=2 * size * size)
    for _ in range(limit):
        if env.game_over:
            break
        env.step(agent.act(env))
    return env


def test_survivable_flags_a_wall():
    env = SnakeEnv(5, np.random.default_rng(0))
    env.snake.clear()
    env.snake.append((5, 3))
    env._occupied = {(5, 3)}
    assert survivable(env, RIGHT) is False


def test_random_agent_only_returns_legal_actions():
    rng = np.random.default_rng(0)
    agent = RandomAgent(rng)
    env = SnakeEnv(6, np.random.default_rng(1), starvation_limit=72)
    for _ in range(300):
        if env.game_over:
            break
        action = agent.act(env)
        assert env.legal_actions()[action]
        env.step(action)


def test_greedy_beats_random_over_many_games():
    greedy = sum(run(GreedyAgent(np.random.default_rng(s)), seed=s).score for s in range(30))
    random_ = sum(run(RandomAgent(np.random.default_rng(s)), seed=s).score for s in range(30))
    assert greedy > random_


def test_greedy_avoids_an_immediately_fatal_move_when_it_can():
    # Food is straight ahead through the wall. Greedy must turn instead.
    env = SnakeEnv(6, np.random.default_rng(0))
    env.snake.clear()
    env.snake.extend([(6, 3), (5, 3), (4, 3)])
    env._occupied = set(env.snake)
    env.direction = RIGHT
    env.food = (6, 6)
    action = GreedyAgent(np.random.default_rng(0)).act(env)
    assert survivable(env, action)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/test_baselines.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'snake.baselines'`.

- [ ] **Step 3: Write the implementation**

Create `snake/baselines.py`:

```python
"""Reference agents. No torch, no pygame.

These exist so that a learned agent's score means something. A mean score of 30
on a 20x20 board is uninterpretable until you know what a greedy heuristic and a
BFS solver reach on the same seeds.
"""

from __future__ import annotations

import numpy as np

from snake.env import DELTAS, SnakeEnv

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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_baselines.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add snake/baselines.py tests/test_baselines.py
git commit -m "feat: add random and greedy baseline agents"
```

---

### Task 8: BFS baseline with tail safety

This is the reference that decides whether AlphaZero learned anything a heuristic did not already have.

**Files:**
- Modify: `snake/baselines.py`
- Modify: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `survivable`, `_legal`, `_distance` from Task 7.
- Produces: `snake.baselines.BFSSafeAgent(rng: np.random.Generator)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_baselines.py`:

```python
from snake.baselines import BFSSafeAgent


def test_bfs_beats_greedy_over_many_games():
    bfs = sum(run(BFSSafeAgent(np.random.default_rng(s)), seed=s).score for s in range(30))
    greedy = sum(run(GreedyAgent(np.random.default_rng(s)), seed=s).score for s in range(30))
    assert bfs > greedy


def test_bfs_fills_most_of_a_small_board():
    scores = [run(BFSSafeAgent(np.random.default_rng(s)), size=6, seed=s).score for s in range(20)]
    assert np.mean(scores) > 15


def test_bfs_only_returns_legal_actions():
    agent = BFSSafeAgent(np.random.default_rng(0))
    env = SnakeEnv(8, np.random.default_rng(2), starvation_limit=128)
    for _ in range(500):
        if env.game_over:
            break
        action = agent.act(env)
        assert env.legal_actions()[action]
        env.step(action)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/test_baselines.py -k bfs -v
```

Expected: FAIL with `ImportError: cannot import name 'BFSSafeAgent'`.

- [ ] **Step 3: Write the implementation**

Append to `snake/baselines.py`:

```python
from collections import deque as _deque


def _bfs(env, start, goal, blocked):
    """Shortest path from start to goal over cells not in blocked. Returns the
    list of cells after start, or None if unreachable."""
    if start == goal:
        return []
    limit = env.grid_dim - 1
    seen = {start}
    queue = _deque([(start, [])])
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
        return False
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_baselines.py -v
```

Expected: all PASS. If `test_bfs_fills_most_of_a_small_board` falls short, the stalling branch is the likely cause: check that `_tail_reachable_after` excludes the tail cell from the blocked set, since including it makes every move look unsafe once the snake is long.

- [ ] **Step 5: Commit**

```bash
git add snake/baselines.py tests/test_baselines.py
git commit -m "feat: add BFS baseline with tail reachability safety"
```

---

### Task 9: Hamiltonian baseline

**Files:**
- Modify: `snake/baselines.py`
- Modify: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `graph.HamiltonianCycle(r, c, cycles).cycle_positions()`, which returns `r * c` tuples of `(row, col)` with both zero based, in cycle order. It uses the global `random` module, which is why the constructor seeds it.
- Produces: `snake.baselines.HamiltonianAgent(size: int, seed: int)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_baselines.py`:

```python
import pytest

from snake.baselines import HamiltonianAgent


def test_hamiltonian_solves_a_small_board():
    env = SnakeEnv(6, np.random.default_rng(0), starvation_limit=None)
    agent = HamiltonianAgent(size=6, seed=0)
    for _ in range(10000):
        if env.game_over:
            break
        env.step(agent.act(env))
    assert env.death_cause == "solved"
    assert env.length == 36


def test_hamiltonian_rejects_odd_boards():
    # A grid graph with both dimensions odd has no Hamiltonian cycle.
    with pytest.raises(ValueError):
        HamiltonianAgent(size=5, seed=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/test_baselines.py -k hamiltonian -v
```

Expected: FAIL with `ImportError: cannot import name 'HamiltonianAgent'`.

- [ ] **Step 3: Write the implementation**

Append to `snake/baselines.py`:

```python
import random as _stdlib_random


class HamiltonianAgent:
    """Follow a fixed Hamiltonian cycle. Perfect by construction, so it is the
    ceiling every other agent is measured against.

    graph.HamiltonianCycle uses the global random module, which is the one
    documented exception to this package's no global randomness rule. The seed
    is set here so runs stay reproducible.
    """

    def __init__(self, size, seed):
        if size % 2 == 1:
            raise ValueError(
                f"size {size} is odd; a grid graph with both dimensions odd has "
                "no Hamiltonian cycle"
            )
        from graph import HamiltonianCycle

        _stdlib_random.seed(seed)
        cycle = HamiltonianCycle(size, size, 1).cycle_positions()
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
        # The env starts the snake in a straight line that is not aligned to the
        # cycle, so the first move or two may deviate. Once the body trails along
        # the cycle the successor cell is always free.
        safe = [a for a in legal if survivable(env, a)]
        return int(safe[0] if safe else legal[0])
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_baselines.py -v
```

Expected: all PASS. `test_hamiltonian_solves_a_small_board` is the one that matters: if it fails with a `self` death cause, the row and column mapping is transposed; print `self.cells[:6]` and check the first cells are adjacent.

- [ ] **Step 5: Commit**

```bash
git add snake/baselines.py tests/test_baselines.py
git commit -m "feat: add Hamiltonian cycle baseline as the score ceiling"
```

---

### Task 10: Evaluation harness

**Files:**
- Create: `snake/arena.py`
- Create: `tests/test_arena.py`

**Interfaces:**
- Consumes: `SnakeEnv`, `default_starvation_limit`, and any agent exposing `act(env) -> int`.
- Produces:
  - `snake.arena.GameResult` dataclass with `score: int`, `length: int`, `steps: int`, `outcome: str`
  - `snake.arena.play_games(agent_factory, size, n_games, seed, starvation_limit=None) -> list[GameResult]`, where `agent_factory(rng)` returns a fresh agent
  - `snake.arena.summarize(results, total_cells) -> dict`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_arena.py`:

```python
import numpy as np

from snake.arena import GameResult, play_games, summarize
from snake.baselines import GreedyAgent, HamiltonianAgent, RandomAgent


def test_play_games_is_deterministic_for_a_seed():
    first = play_games(RandomAgent, size=6, n_games=10, seed=7)
    second = play_games(RandomAgent, size=6, n_games=10, seed=7)
    assert [r.score for r in first] == [r.score for r in second]


def test_different_seeds_give_different_games():
    first = play_games(RandomAgent, size=6, n_games=10, seed=1)
    second = play_games(RandomAgent, size=6, n_games=10, seed=2)
    assert [r.score for r in first] != [r.score for r in second]


def test_every_game_reports_a_known_outcome():
    results = play_games(GreedyAgent, size=6, n_games=20, seed=0)
    assert len(results) == 20
    assert all(r.outcome in {"wall", "self", "starvation", "solved"} for r in results)


def test_summary_fields():
    results = [
        GameResult(score=3, length=6, steps=40, outcome="wall"),
        GameResult(score=5, length=8, steps=60, outcome="self"),
    ]
    summary = summarize(results, total_cells=36)
    assert summary["games"] == 2
    assert summary["mean_score"] == 4.0
    assert summary["median_score"] == 4.0
    assert summary["max_score"] == 5
    assert summary["mean_steps"] == 50.0
    assert np.isclose(summary["mean_fill_fraction"], (6 / 36 + 8 / 36) / 2)
    assert summary["outcomes"] == {"wall": 1, "self": 1, "starvation": 0, "solved": 0}


def test_hamiltonian_summary_is_a_perfect_ceiling():
    summary = summarize(
        play_games(lambda rng: HamiltonianAgent(6, 0), size=6, n_games=5, seed=0),
        total_cells=36,
    )
    assert summary["mean_fill_fraction"] == 1.0
    assert summary["outcomes"]["solved"] == 5
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/test_arena.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'snake.arena'`.

- [ ] **Step 3: Write the implementation**

Create `snake/arena.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_arena.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add snake/arena.py tests/test_arena.py
git commit -m "feat: add deterministic evaluation harness"
```

---

### Task 11: Record baseline results, Phase 0 acceptance

**Files:**
- Create: `snake/cli.py`
- Create: `results/.gitkeep`
- Create: `results/baselines.json` (generated)

**Interfaces:**
- Consumes: everything from Tasks 7 through 10.
- Produces: `python -m snake.cli baselines --sizes 6 10 20 --games 100 --seed 0 --out results/baselines.json`.

- [ ] **Step 1: Write the CLI**

Create `snake/cli.py`:

```python
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
```

- [ ] **Step 2: Run the baselines on 6x6 only, as a smoke test**

```bash
./venv/bin/python -m snake.cli baselines --sizes 6 --games 20 --seed 0 --out /tmp/smoke.json
```

Expected: four lines printed, with `random` scoring lowest, `greedy` above it, `bfs_safe` above that, and `hamiltonian` at fill 1.000 with every outcome `solved`. If that ordering does not hold, a baseline is wrong and the following full run is not worth doing.

- [ ] **Step 3: Record the full baseline table**

```bash
mkdir -p results && touch results/.gitkeep
./venv/bin/python -m snake.cli baselines --sizes 6 10 20 --games 100 --seed 0 --out results/baselines.json
```

The 20x20 Hamiltonian run does real work in `graph.py`, so allow several minutes.

- [ ] **Step 4: Run the whole suite**

```bash
./venv/bin/python -m pytest -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add snake/cli.py results/
git commit -m "feat: record baseline scores for 6x6, 10x10 and 20x20"
```

**Phase 0 is accepted when:** the pinned tests from Task 1 still pass, `app.py` plays correctly by hand, and `results/baselines.json` holds scores for all four reference agents at all three board sizes.

---

## Phase 1: learning loop on 6x6

### Task 12: Run configuration and seeding

**Files:**
- Create: `snake/config.py`
- Create: `requirements-train.txt`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `snake.config.NetConfig`, `SearchConfig`, `TrainConfig`, `RunConfig` frozen dataclasses
  - `RunConfig.to_json() -> str` and `RunConfig.from_json(text: str) -> RunConfig`
  - `snake.config.seed_everything(seed: int) -> None`

- [ ] **Step 1: Declare the training dependencies**

```bash
cat > requirements-train.txt <<'EOF'
torch==2.5.1
tensorboard==2.18.0
EOF
./venv/bin/pip install -r requirements-train.txt
./venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: the version and `True`. If CUDA reports `False`, install the CUDA build for this machine's driver from pytorch.org before continuing, because the whole plan assumes GPU inference.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_config.py`:

```python
import numpy as np
import pytest

from snake.config import RunConfig, seed_everything


def test_defaults_match_the_spec():
    cfg = RunConfig()
    assert cfg.net.channels == 64
    assert cfg.net.blocks == 6
    assert cfg.search.simulations == 100
    assert cfg.search.c_puct == 1.5
    assert cfg.search.dirichlet_alpha == 0.8
    assert cfg.search.dirichlet_epsilon == 0.25
    assert cfg.search.tau_threshold == 40
    assert cfg.search.tau_final == 0.2
    assert cfg.train.board_size == 6
    assert cfg.train.batch_size == 512
    assert cfg.train.replay_capacity == 200_000
    assert cfg.train.learning_rate == 2e-3
    assert cfg.train.weight_decay == 1e-4
    assert cfg.train.concurrent_games == 32
    assert cfg.train.eval_games == 100


def test_channels_must_divide_into_groups():
    with pytest.raises(ValueError):
        RunConfig.build(net={"channels": 65, "groups": 8})


def test_json_round_trip():
    cfg = RunConfig.build(train={"board_size": 10, "seed": 3}, search={"simulations": 25})
    restored = RunConfig.from_json(cfg.to_json())
    assert restored == cfg
    assert restored.train.board_size == 10
    assert restored.search.simulations == 25


def test_seed_everything_makes_torch_reproducible():
    import torch

    seed_everything(5)
    first = torch.randn(4)
    seed_everything(5)
    assert torch.equal(first, torch.randn(4))


def test_seed_everything_makes_numpy_legacy_reproducible():
    seed_everything(5)
    first = np.random.rand(4)
    seed_everything(5)
    assert np.array_equal(first, np.random.rand(4))
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'snake.config'`.

- [ ] **Step 4: Write the implementation**

Create `snake/config.py`:

```python
"""Run configuration. Serialized next to every checkpoint so a result three
weeks old can still be explained. No torch import at module level, so the
baseline commands keep working without the training stack."""

from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class NetConfig:
    channels: int = 64
    blocks: int = 6
    groups: int = 8


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.8
    dirichlet_epsilon: float = 0.25
    tau_threshold: int = 40
    tau_final: float = 0.2


@dataclass(frozen=True)
class TrainConfig:
    board_size: int = 6
    iterations: int = 200
    concurrent_games: int = 32
    train_steps_per_iteration: int = 200
    batch_size: int = 512
    replay_capacity: int = 200_000
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    eval_games: int = 100
    eval_every: int = 5
    checkpoint_every: int = 5
    keep_last: int = 3
    seed: int = 0
    device: str = "cuda"
    run_dir: str = "runs/default"


@dataclass(frozen=True)
class RunConfig:
    net: NetConfig = field(default_factory=NetConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        if self.net.channels % self.net.groups != 0:
            raise ValueError(
                f"channels {self.net.channels} must be divisible by "
                f"groups {self.net.groups}"
            )
        if self.train.board_size < 3:
            raise ValueError(f"board_size must be at least 3, got {self.train.board_size}")

    @classmethod
    def build(cls, net=None, search=None, train=None):
        return cls(
            net=NetConfig(**(net or {})),
            search=SearchConfig(**(search or {})),
            train=TrainConfig(**(train or {})),
        )

    def to_json(self):
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text):
        raw = json.loads(text)
        return cls.build(net=raw["net"], search=raw["search"], train=raw["train"])


def seed_everything(seed):
    """Seed every global source of randomness. Envs still take an explicit
    generator; this covers torch, the legacy numpy global, and the stdlib module
    that graph.py reaches for."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_config.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add snake/config.py requirements-train.txt tests/test_config.py
git commit -m "feat: add run configuration and explicit seeding"
```

---

### Task 13: Dihedral symmetry transforms

Eight times the training data at negligible cost. The action permutation is derived from the geometry rather than typed from a table, because a reversed permutation degrades the agent silently instead of raising.

**Files:**
- Modify: `snake/encoding.py`
- Create: `tests/test_symmetry.py`

**Interfaces:**
- Consumes: `N_PLANES`, `HEADING_PLANE_0`, `encode` from Task 5.
- Produces:
  - `snake.encoding.SYMMETRIES: tuple[tuple[int, bool], ...]`, the eight `(k, flip)` pairs
  - `snake.encoding.action_permutation(k: int, flip: bool) -> list[int]`
  - `snake.encoding.transform_observation(obs, k, flip) -> np.ndarray`
  - `snake.encoding.transform_policy(pi, k, flip) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_symmetry.py`:

```python
import numpy as np

from snake.encoding import (
    SYMMETRIES,
    action_permutation,
    encode,
    transform_observation,
    transform_policy,
)
from snake.env import DELTAS, DOWN, LEFT, RIGHT, UP


def sample():
    return encode(size=6, snake=[(4, 3), (3, 3), (2, 3)], food=(6, 6), direction=RIGHT)


def test_there_are_eight_symmetries():
    assert len(SYMMETRIES) == 8
    assert len(set(SYMMETRIES)) == 8


def test_identity_changes_nothing():
    obs = sample()
    assert np.array_equal(transform_observation(obs, 0, False), obs)
    assert action_permutation(0, False) == [0, 1, 2, 3]


def test_one_rotation_maps_right_to_up():
    # np.rot90 on a [.., y, x] array sends displacement (dx, dy) to (dy, -dx).
    assert action_permutation(1, False) == [UP, RIGHT, DOWN, LEFT]


def test_flip_swaps_left_and_right_only():
    assert action_permutation(0, True) == [LEFT, DOWN, RIGHT, UP]


def test_permutation_is_a_bijection():
    for k, flip in SYMMETRIES:
        assert sorted(action_permutation(k, flip)) == [0, 1, 2, 3]


def test_observation_round_trip():
    obs = sample()
    for k in range(4):
        there = transform_observation(obs, k, False)
        back = transform_observation(there, (4 - k) % 4, False)
        assert np.array_equal(back, obs)


def test_policy_round_trip():
    pi = np.array([0.1, 0.2, 0.3, 0.4])
    for k in range(4):
        there = transform_policy(pi, k, False)
        back = transform_policy(there, (4 - k) % 4, False)
        assert np.allclose(back, pi)


def test_policy_moves_mass_the_same_way_as_the_board():
    pi = np.zeros(4)
    pi[RIGHT] = 1.0
    for k, flip in SYMMETRIES:
        moved = transform_policy(pi, k, flip)
        assert moved[action_permutation(k, flip)[RIGHT]] == 1.0


def test_heading_plane_agrees_with_the_transformed_geometry():
    # The single most load bearing symmetry test. In every transformed frame,
    # the heading channel must match the actual head minus neck displacement.
    # A reversed permutation fails here and nowhere else.
    obs = sample()
    for k, flip in SYMMETRIES:
        out = transform_observation(obs, k, flip)
        head = _one_hot_position(out[2])
        neck = _highest_body_age_position(out)
        delta = (head[1] - neck[1], head[0] - neck[0])  # (dx, dy)
        encoded = [p for p in range(4) if out[6 + p].any()]
        assert len(encoded) == 1
        assert DELTAS[encoded[0]] == delta, f"k={k} flip={flip}"


def test_transform_preserves_plane_sums():
    obs = sample()
    for k, flip in SYMMETRIES:
        out = transform_observation(obs, k, flip)
        for plane in (0, 1, 2, 3, 4):
            assert np.isclose(out[plane].sum(), obs[plane].sum())


def _one_hot_position(plane):
    ys, xs = np.nonzero(plane)
    assert len(ys) == 1
    return int(ys[0]), int(xs[0])


def _highest_body_age_position(obs):
    # Among body cells excluding the head, the neck has the largest age value.
    body = obs[5] * (obs[1] > 0)
    ys, xs = np.nonzero(body == body.max())
    return int(ys[0]), int(xs[0])
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/test_symmetry.py -v
```

Expected: FAIL with `ImportError: cannot import name 'SYMMETRIES'`.

- [ ] **Step 3: Write the implementation**

Append to `snake/encoding.py`:

```python
# The eight elements of the dihedral group of the square, as (rotations, flip).
SYMMETRIES = tuple((k, flip) for flip in (False, True) for k in range(4))

# (dx, dy) per action. Duplicated from snake.env deliberately: encoding.py must
# not import env.py, since env.py imports this module.
_DELTAS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def action_permutation(k, flip):
    """Where each action ends up after k rotations and an optional x flip.

    Derived from the geometry rather than tabulated. np.rot90 on an array
    indexed [.., y, x] sends a displacement (dx, dy) to (dy, -dx); np.flip on
    the x axis sends it to (-dx, dy).
    """
    perm = [0] * 4
    for action, (dx, dy) in enumerate(_DELTAS):
        for _ in range(k % 4):
            dx, dy = dy, -dx
        if flip:
            dx = -dx
        perm[action] = _DELTAS.index((dx, dy))
    return perm


def transform_observation(obs, k, flip):
    out = np.rot90(obs, k=k, axes=(1, 2))
    if flip:
        out = np.flip(out, axis=2)
    out = np.ascontiguousarray(out)

    # The heading planes are spatially constant, so rotating them is a no op.
    # Their channel identity is what has to move.
    perm = action_permutation(k, flip)
    heading = out[HEADING_PLANE_0 : HEADING_PLANE_0 + 4].copy()
    for action in range(4):
        out[HEADING_PLANE_0 + perm[action]] = heading[action]
    return out


def transform_policy(pi, k, flip):
    perm = action_permutation(k, flip)
    out = np.zeros_like(pi)
    for action in range(4):
        out[perm[action]] = pi[action]
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_symmetry.py -v
```

Expected: all PASS. If `test_heading_plane_agrees_with_the_transformed_geometry` fails, the permutation is applied in the wrong direction: the assignment must be `out[HEADING_PLANE_0 + perm[a]] = heading[a]`, not `out[HEADING_PLANE_0 + a] = heading[perm[a]]`.

- [ ] **Step 5: Commit**

```bash
git add snake/encoding.py tests/test_symmetry.py
git commit -m "feat: add D4 symmetry transforms for observations and policies"
```

---

### Task 14: The network

**Files:**
- Create: `snake/model.py`
- Create: `tests/test_model.py`

**Interfaces:**
- Consumes: `snake.encoding.N_PLANES`, `snake.config.NetConfig`.
- Produces: `snake.model.SnakeNet(cfg: NetConfig)` with `forward(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`, returning `logits[B, 4]` and `value[B]` in `[0, 1]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_model.py`:

```python
import numpy as np
import torch

from snake.config import NetConfig
from snake.encoding import N_PLANES, encode
from snake.env import RIGHT
from snake.model import SnakeNet


def net(**kwargs):
    return SnakeNet(NetConfig(channels=16, blocks=2, groups=4, **kwargs))


def batch(size, n=3):
    obs = np.stack(
        [encode(size, [(3, 2), (2, 2), (1, 2)], (size, size), RIGHT) for _ in range(n)]
    )
    return torch.from_numpy(obs)


def test_output_shapes():
    logits, value = net()(batch(6))
    assert logits.shape == (3, 4)
    assert value.shape == (3,)


def test_value_is_in_the_unit_interval():
    _, value = net()(batch(6))
    assert torch.all(value >= 0) and torch.all(value <= 1)


def test_policy_head_emits_logits_not_probabilities():
    # Logits must be free to be negative and must not already sum to one, or the
    # training loss would softmax twice.
    logits, _ = net()(batch(6, n=64))
    assert not torch.allclose(logits.exp().sum(dim=1), torch.ones(64))


def test_one_set_of_weights_runs_on_any_board_size():
    model = net()
    for size in (6, 10, 20):
        logits, value = model(batch(size))
        assert logits.shape == (3, 4)
        assert value.shape == (3,)


def test_there_is_no_batchnorm_anywhere():
    # BatchNorm plus batch of one inference is the classic AlphaZero footgun.
    for module in net().modules():
        assert not isinstance(module, torch.nn.modules.batchnorm._BatchNorm)


def test_batch_of_one_matches_the_same_row_in_a_batch():
    model = net().eval()
    many = batch(6, n=4)
    with torch.no_grad():
        logits_many, value_many = model(many)
        logits_one, value_one = model(many[:1])
    assert torch.allclose(logits_many[0], logits_one[0], atol=1e-5)
    assert torch.allclose(value_many[0], value_one[0], atol=1e-5)


def test_the_policy_head_reads_the_head_cell():
    # Moving only the head changes the policy. If it does not, the head cell
    # gather is broken and the head is being ignored.
    model = net().eval()
    a = torch.from_numpy(encode(6, [(3, 2), (2, 2), (1, 2)], (6, 6), RIGHT))[None]
    b = torch.from_numpy(encode(6, [(3, 5), (2, 5), (1, 5)], (6, 6), RIGHT))[None]
    with torch.no_grad():
        assert not torch.allclose(model(a)[0], model(b)[0], atol=1e-4)


def test_gradients_reach_every_parameter():
    model = net()
    logits, value = model(batch(6))
    (logits.sum() + value.sum()).backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/test_model.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'snake.model'`.

- [ ] **Step 3: Write the implementation**

Create `snake/model.py`:

```python
"""Fully convolutional policy and value network.

Nothing may be sized from the board dimensions, because one set of weights has
to serve every board in the curriculum. GroupNorm rather than BatchNorm, so a
batch of one during search behaves exactly like the same row inside a batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from snake.encoding import N_PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.relu(x + h)


class SnakeNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels, groups = cfg.channels, cfg.groups

        self.stem_conv = nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False)
        self.stem_norm = nn.GroupNorm(groups, channels)
        self.blocks = nn.ModuleList(
            ResidualBlock(channels, groups) for _ in range(cfg.blocks)
        )

        self.policy_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.policy_norm = nn.GroupNorm(groups, channels)
        self.policy_fc = nn.Sequential(
            nn.Linear(2 * channels, channels), nn.ReLU(), nn.Linear(channels, 4)
        )

        self.value_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.value_norm = nn.GroupNorm(groups, channels)
        self.value_fc = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, 1)
        )

    def forward(self, obs):
        """obs is float32 [B, N_PLANES, H, W]. Returns (logits[B, 4], value[B])."""
        # Plane 2 is the head one hot, so multiplying by it and summing over
        # space gathers the head cell's feature vector. This keeps the readout
        # differentiable, batched, and independent of the board size.
        head_mask = obs[:, 2:3]

        x = F.relu(self.stem_norm(self.stem_conv(obs)))
        for block in self.blocks:
            x = block(x)

        p = F.relu(self.policy_norm(self.policy_conv(x)))
        at_head = (p * head_mask).sum(dim=(2, 3))
        pooled = p.mean(dim=(2, 3))
        logits = self.policy_fc(torch.cat([at_head, pooled], dim=1))

        v = F.relu(self.value_norm(self.value_conv(x)))
        value = torch.sigmoid(self.value_fc(v.mean(dim=(2, 3)))).squeeze(1)

        return logits, value
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_model.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add snake/model.py tests/test_model.py
git commit -m "feat: add fully convolutional policy and value network"
```

---

### Task 15: Evaluator

The interface search calls instead of the network. This is the seam that later carries batched and multi-process inference without search changing.

**Files:**
- Create: `snake/evaluator.py`
- Create: `tests/test_evaluator.py`

**Interfaces:**
- Consumes: `SnakeNet` from Task 14.
- Produces:
  - `snake.evaluator.Evaluator` protocol with `evaluate_batch(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]`, taking `[B, C, H, W]` and returning priors `[B, 4]` summing to one per row and values `[B]` in `[0, 1]`
  - `snake.evaluator.TorchEvaluator(model, device)`
  - `snake.evaluator.UniformEvaluator(value: float = 0.0)`, a stub for testing search in isolation

- [ ] **Step 1: Write the failing tests**

Create `tests/test_evaluator.py`:

```python
import numpy as np
import torch

from snake.config import NetConfig
from snake.encoding import encode
from snake.env import RIGHT
from snake.evaluator import TorchEvaluator, UniformEvaluator
from snake.model import SnakeNet


def obs_batch(n=5, size=6):
    return np.stack(
        [encode(size, [(3, 2), (2, 2), (1, 2)], (size, size), RIGHT) for _ in range(n)]
    )


def test_uniform_evaluator_shapes_and_values():
    priors, values = UniformEvaluator(value=0.25).evaluate_batch(obs_batch(3))
    assert priors.shape == (3, 4)
    assert np.allclose(priors, 0.25)
    assert np.allclose(values, 0.25)


def test_torch_evaluator_returns_a_distribution():
    model = SnakeNet(NetConfig(channels=16, blocks=2, groups=4))
    priors, values = TorchEvaluator(model, torch.device("cpu")).evaluate_batch(obs_batch(7))
    assert priors.shape == (7, 4)
    assert np.allclose(priors.sum(axis=1), 1.0, atol=1e-5)
    assert values.shape == (7,)
    assert ((values >= 0) & (values <= 1)).all()


def test_torch_evaluator_leaves_the_model_in_eval_mode_and_takes_no_gradients():
    model = SnakeNet(NetConfig(channels=16, blocks=2, groups=4))
    model.train()
    evaluator = TorchEvaluator(model, torch.device("cpu"))
    evaluator.evaluate_batch(obs_batch(2))
    assert model.training is False
    assert all(p.grad is None for p in model.parameters())


def test_torch_evaluator_output_is_float32_numpy():
    model = SnakeNet(NetConfig(channels=16, blocks=2, groups=4))
    priors, values = TorchEvaluator(model, torch.device("cpu")).evaluate_batch(obs_batch(2))
    assert isinstance(priors, np.ndarray) and priors.dtype == np.float32
    assert isinstance(values, np.ndarray) and values.dtype == np.float32
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/test_evaluator.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'snake.evaluator'`.

- [ ] **Step 3: Write the implementation**

Create `snake/evaluator.py`:

```python
"""The seam between search and the network.

MCTS never calls a model directly. Swapping single process inference for
multi process workers later is a new class here and no change to mcts.py.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch


class Evaluator(Protocol):
    def evaluate_batch(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """obs is [B, C, H, W]. Returns priors [B, 4] summing to one per row and
        values [B] in [0, 1]."""


class UniformEvaluator:
    """Uniform priors and a constant value. Lets search be tested without a net."""

    def __init__(self, value=0.0):
        self.value = value

    def evaluate_batch(self, obs):
        n = len(obs)
        priors = np.full((n, 4), 0.25, dtype=np.float32)
        values = np.full(n, self.value, dtype=np.float32)
        return priors, values


class TorchEvaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def evaluate_batch(self, obs):
        tensor = torch.from_numpy(np.ascontiguousarray(obs)).to(
            self.device, non_blocking=True
        )
        logits, values = self.model(tensor)
        priors = torch.softmax(logits, dim=1)
        return (
            priors.float().cpu().numpy(),
            values.float().cpu().numpy(),
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_evaluator.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add snake/evaluator.py tests/test_evaluator.py
git commit -m "feat: add the evaluator seam between search and the network"
```

---

### Task 16: Open-loop determinized MCTS

The core of the project. `Search` is split into `descend` and `apply` so a driver can run many games in lockstep and batch every leaf evaluation at a given simulation index into one forward pass.

**Files:**
- Create: `snake/mcts.py`
- Create: `tests/test_mcts.py`

**Interfaces:**
- Consumes: `SnakeEnv`, `SearchConfig`, the `Evaluator` protocol.
- Produces:
  - `snake.mcts.Node` with `prior`, `visits`, `value_sum`, `children`, `terminal`, `expanded`
  - `snake.mcts.Search(env: SnakeEnv, cfg: SearchConfig, rng: np.random.Generator)`
  - `Search.descend() -> np.ndarray | None`, walks to a leaf and returns the observation needing evaluation, or `None` when the leaf was terminal and was backed up already
  - `Search.apply(priors: np.ndarray, value: float) -> None`, completes the pending simulation
  - `Search.visit_counts() -> np.ndarray` of shape `[4]`
  - `snake.mcts.run_search(searches: list[Search], evaluator, simulations: int) -> None`, the lockstep driver

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcts.py`:

```python
import numpy as np

from snake.config import SearchConfig
from snake.env import LEFT, OPPOSITE, RIGHT, SnakeEnv
from snake.evaluator import UniformEvaluator
from snake.mcts import Search, run_search


def make_env(size=6, seed=0):
    return SnakeEnv(size, np.random.default_rng(seed), starvation_limit=2 * size * size)


def make_search(env=None, **overrides):
    settings = {"simulations": 64, "dirichlet_epsilon": 0.0}
    settings.update(overrides)
    return Search(env or make_env(), SearchConfig(**settings), np.random.default_rng(0))


def test_visits_are_spent_on_legal_actions_only():
    env = make_env()
    search = make_search(env)
    run_search([search], UniformEvaluator(0.5), simulations=64)
    counts = search.visit_counts()
    assert counts[OPPOSITE[env.direction]] == 0
    assert counts.sum() > 0


def test_total_visits_match_the_simulation_budget():
    search = make_search()
    run_search([search], UniformEvaluator(0.5), simulations=64)
    # The first simulation expands the root and spends no child visit.
    assert search.visit_counts().sum() == 63


def test_search_never_mutates_the_root_env():
    env = make_env()
    before = (list(env.snake), env.food, env.score, env.direction)
    run_search([make_search(env)], UniformEvaluator(0.5), simulations=64)
    assert (list(env.snake), env.food, env.score, env.direction) == before


def test_a_forced_win_is_found():
    # Food is one step to the right. With a value of zero everywhere, the only
    # thing separating actions is the 1/C reward on the eating edge, so the
    # search must concentrate on RIGHT.
    env = make_env()
    env.food = (env.snake[0][0] + 1, env.snake[0][1])
    search = make_search(env, simulations=200)
    run_search([search], UniformEvaluator(0.0), simulations=200)
    counts = search.visit_counts()
    assert counts.argmax() == RIGHT


def test_death_is_avoided():
    # The head is against the right wall with a clear board otherwise. Moving
    # right is instant death and must attract the fewest visits.
    env = make_env(size=8)
    env.snake.clear()
    env.snake.extend([(8, 4), (7, 4), (6, 4)])
    env._occupied = set(env.snake)
    env.direction = RIGHT
    env.food = (2, 2)
    search = make_search(env, simulations=300)
    run_search([search], UniformEvaluator(0.5), simulations=300)
    counts = search.visit_counts()
    assert counts.argmax() != RIGHT
    assert counts[RIGHT] < counts.max()


def test_terminal_nodes_are_never_expanded():
    env = make_env(size=8)
    env.snake.clear()
    env.snake.extend([(8, 4), (7, 4), (6, 4)])
    env._occupied = set(env.snake)
    env.direction = RIGHT
    search = make_search(env, simulations=100)
    run_search([search], UniformEvaluator(0.5), simulations=100)
    dead = search.root.children[RIGHT]
    assert dead.terminal is True
    assert dead.expanded is False
    assert dead.value_sum == 0.0


def test_root_value_equals_the_leaf_value_when_nothing_is_eaten():
    # With no food reachable inside the horizon, every backup is the identity,
    # so the root's mean value is exactly the constant the evaluator returns.
    env = make_env(size=8)
    env.food = (8, 8)
    search = make_search(env, simulations=32)
    run_search([search], UniformEvaluator(0.4), simulations=32)
    assert np.isclose(search.root.value_sum / search.root.visits, 0.4, atol=1e-6)


def test_eating_adds_exactly_one_over_total_cells_on_backup():
    # One simulation, forced onto the eating edge by making every other action
    # illegal is not possible, so instead check the analytic relation directly:
    # the root value must exceed the leaf constant by at most 1/C per eat.
    env = make_env(size=6)
    env.food = (env.snake[0][0] + 1, env.snake[0][1])
    search = make_search(env, simulations=200)
    run_search([search], UniformEvaluator(0.3), simulations=200)
    root_value = search.root.value_sum / search.root.visits
    assert root_value > 0.3
    assert root_value < 0.3 + 1.0 / env.total_cells * 3


def test_dirichlet_noise_changes_the_root_priors():
    env = make_env()
    quiet = make_search(env, simulations=8, dirichlet_epsilon=0.0)
    noisy = Search(env, SearchConfig(simulations=8, dirichlet_epsilon=0.25),
                   np.random.default_rng(0))
    run_search([quiet], UniformEvaluator(0.5), simulations=8)
    run_search([noisy], UniformEvaluator(0.5), simulations=8)
    quiet_priors = [c.prior for c in quiet.root.children if c is not None]
    noisy_priors = [c.prior for c in noisy.root.children if c is not None]
    assert not np.allclose(quiet_priors, noisy_priors)
    assert np.isclose(sum(noisy_priors), 1.0)


def test_lockstep_driver_runs_many_games_together():
    searches = [make_search(make_env(seed=s)) for s in range(4)]
    run_search(searches, UniformEvaluator(0.5), simulations=32)
    for search in searches:
        assert search.visit_counts().sum() == 31


def test_open_loop_visits_different_food_futures():
    # Two simulations that both eat should see different replacement food, which
    # is the whole point of reseeding per simulation.
    env = make_env(size=6)
    env.food = (env.snake[0][0] + 1, env.snake[0][1])
    search = make_search(env, simulations=400)
    seen = set()
    original_clone = env.clone

    def spy(rng):
        twin = original_clone(rng)
        seen.add(twin.rng.integers(0, 2**31))
        return twin

    env.clone = spy
    run_search([search], UniformEvaluator(0.5), simulations=400)
    assert len(seen) > 100
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/test_mcts.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'snake.mcts'`.

- [ ] **Step 3: Write the implementation**

Create `snake/mcts.py`:

```python
"""Open loop determinized PUCT for a single player game.

Nodes are keyed by action path rather than by state. Every simulation replays
from the root with a fresh generator, so repeated visits to a node average over
different food futures instead of committing to one sampled world. Environment
steps are constant time and the descent is linear in depth regardless, so
replaying costs essentially nothing.

Values are the fraction of the whole board still to be filled, which is the
value function of an undiscounted MDP with a reward of 1/C per food. Backup
therefore adds that reward on eating edges, and every value in the tree is in
the same units, so comparing siblings is well defined even when one was reached
by eating and another was not.
"""

from __future__ import annotations

import math

import numpy as np

_SEED_MAX = 2**63 - 1


class Node:
    __slots__ = ("prior", "visits", "value_sum", "children", "terminal")

    def __init__(self, prior):
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children = None
        self.terminal = False

    @property
    def expanded(self):
        return self.children is not None

    def expand(self, priors, legal):
        masked = np.where(legal, priors, 0.0)
        total = masked.sum()
        if total > 0:
            masked = masked / total
        else:
            masked = legal.astype(np.float64) / legal.sum()
        self.children = [
            Node(float(masked[a])) if legal[a] else None for a in range(4)
        ]


class Search:
    def __init__(self, env, cfg, rng):
        self.env = env
        self.cfg = cfg
        self.rng = rng
        self.root = Node(1.0)
        self._reward = 1.0 / env.total_cells
        self._pending = None

    # Lockstep interface

    def descend(self):
        """Walk to a leaf. Returns the observation needing evaluation, or None
        when the leaf was terminal, in which case backup has already run."""
        scratch = self.env.clone(np.random.default_rng(int(self.rng.integers(_SEED_MAX))))
        node = self.root
        path = [node]
        rewards = []

        while node.expanded and not node.terminal:
            action = self._select(node, scratch)
            ate = scratch.would_eat(action)
            done = scratch.step(action)
            rewards.append(self._reward if ate else 0.0)
            node = node.children[action]
            path.append(node)
            if done:
                node.terminal = True
                break

        if node.terminal:
            self._backup(path, rewards, 0.0)
            return None

        self._pending = (path, rewards, scratch)
        return scratch.observation()

    def apply(self, priors, value):
        """Complete the pending simulation with the evaluator's answer."""
        path, rewards, scratch = self._pending
        self._pending = None
        leaf = path[-1]
        leaf.expand(priors, scratch.legal_actions())
        if leaf is self.root and self.cfg.dirichlet_epsilon > 0:
            self._add_root_noise()
        self._backup(path, rewards, float(value))

    def visit_counts(self):
        return np.array(
            [0 if child is None else child.visits for child in self.root.children],
            dtype=np.float64,
        )

    # Internals

    def _select(self, node, scratch):
        sqrt_parent = math.sqrt(max(1, node.visits))
        parent_value = node.value_sum / node.visits if node.visits else 0.0
        best_action, best_score = -1, -math.inf
        for action in range(4):
            child = node.children[action]
            if child is None:
                continue
            # An unvisited child inherits the parent's estimate rather than
            # infinity, so the prior still shapes the first few visits.
            child_value = (
                child.value_sum / child.visits if child.visits else parent_value
            )
            reward = self._reward if scratch.would_eat(action) else 0.0
            q = reward + child_value
            score = q + self.cfg.c_puct * child.prior * sqrt_parent / (1 + child.visits)
            if score > best_score:
                best_score, best_action = score, action
        return best_action

    def _backup(self, path, rewards, value):
        # value is V(leaf) in the shared units. Walking up, V(parent) is the
        # edge reward plus V(child), with no sign alternation because there is
        # no opponent.
        current = value
        for index in range(len(path) - 1, -1, -1):
            path[index].visits += 1
            path[index].value_sum += current
            if index > 0:
                current = current + rewards[index - 1]

    def _add_root_noise(self):
        legal = [a for a in range(4) if self.root.children[a] is not None]
        noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(legal))
        epsilon = self.cfg.dirichlet_epsilon
        for action, sample in zip(legal, noise):
            child = self.root.children[action]
            child.prior = (1 - epsilon) * child.prior + epsilon * float(sample)


def run_search(searches, evaluator, simulations):
    """Advance every search in lockstep, batching the leaf evaluations at each
    simulation index into a single call. This is what keeps a GPU busy without
    introducing threads or processes."""
    for _ in range(simulations):
        pending = []
        for search in searches:
            obs = search.descend()
            if obs is not None:
                pending.append((search, obs))
        if not pending:
            continue
        batch = np.stack([obs for _, obs in pending])
        priors, values = evaluator.evaluate_batch(batch)
        for (search, _), prior, value in zip(pending, priors, values):
            search.apply(prior, value)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_mcts.py -v
```

Expected: all PASS. Two likely failures and their causes:

- `test_root_value_equals_the_leaf_value_when_nothing_is_eaten` failing means `_backup` is adding a reward where there was none. The reward index is `rewards[index - 1]`, because `rewards[i]` describes the edge into `path[i + 1]`.
- `test_total_visits_match_the_simulation_budget` off by more than one means terminal simulations are being counted twice: `descend` must back up and return `None` without also leaving a pending entry.

- [ ] **Step 5: Commit**

```bash
git add snake/mcts.py tests/test_mcts.py
git commit -m "feat: add open loop determinized MCTS with reward carrying backup"
```

---

### Task 17: Self-play, value targets, and the replay buffer

**Files:**
- Create: `snake/selfplay.py`
- Create: `tests/test_selfplay.py`

**Interfaces:**
- Consumes: `Search`, `run_search`, `SnakeEnv`, `default_starvation_limit`, `SYMMETRIES`, `transform_observation`, `transform_policy`, `encode`.
- Produces:
  - `snake.selfplay.Position` dataclass with `snake: tuple`, `food`, `direction: int`, `size: int`, `pi: np.ndarray`, `z: float`
  - `snake.selfplay.value_targets(lengths: list[int], final_length: int, total_cells: int) -> list[float]`
  - `snake.selfplay.select_move(pi, move_index, search_cfg: SearchConfig, rng) -> int`
  - `snake.selfplay.play_batch(cfg, evaluator, rng) -> tuple[list[Position], list[GameResult]]`
  - `snake.selfplay.ReplayBuffer(capacity: int)` with `extend(positions)`, `sample(batch_size, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]`, and `__len__`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_selfplay.py`:

```python
import numpy as np

from snake.config import RunConfig, SearchConfig
from snake.env import RIGHT, SnakeEnv
from snake.evaluator import UniformEvaluator
from snake.selfplay import Position, ReplayBuffer, play_batch, select_move, value_targets


def cfg(**train):
    settings = {"board_size": 6, "concurrent_games": 2}
    settings.update(train)
    return RunConfig.build(train=settings, search={"simulations": 8, "dirichlet_epsilon": 0.0})


def test_value_targets_are_return_to_go_over_total_cells():
    # Lengths 3, 3, 4, 5 with a final length of 5 on a 36 cell board.
    assert value_targets([3, 3, 4, 5], final_length=5, total_cells=36) == [
        2 / 36,
        2 / 36,
        1 / 36,
        0.0,
    ]


def test_value_targets_are_zero_when_nothing_more_is_eaten():
    assert value_targets([7, 7, 7], final_length=7, total_cells=36) == [0.0, 0.0, 0.0]


def test_value_targets_stay_in_the_unit_interval():
    z = value_targets([3], final_length=36, total_cells=36)
    assert 0.0 <= z[0] <= 1.0


def test_select_move_is_greedy_at_zero_temperature():
    pi = np.array([0.1, 0.6, 0.0, 0.3])
    settings = SearchConfig(tau_threshold=0, tau_final=0.0)
    picks = {select_move(pi, 5, settings, np.random.default_rng(s)) for s in range(20)}
    assert picks == {1}


def test_select_move_explores_at_temperature_one():
    pi = np.array([0.25, 0.25, 0.25, 0.25])
    settings = SearchConfig(tau_threshold=100)
    picks = {select_move(pi, 0, settings, np.random.default_rng(s)) for s in range(40)}
    assert len(picks) > 1


def test_play_batch_returns_finished_games():
    positions, results = play_batch(cfg(), UniformEvaluator(0.5), np.random.default_rng(0))
    assert len(results) == 2
    assert all(r.outcome in {"wall", "self", "starvation", "solved"} for r in results)
    assert len(positions) == sum(r.steps for r in results)


def test_play_batch_positions_carry_normalized_policies():
    positions, _ = play_batch(cfg(), UniformEvaluator(0.5), np.random.default_rng(1))
    for position in positions[:50]:
        assert np.isclose(position.pi.sum(), 1.0)
        assert 0.0 <= position.z <= 1.0


def test_play_batch_is_reproducible():
    first, _ = play_batch(cfg(), UniformEvaluator(0.5), np.random.default_rng(3))
    second, _ = play_batch(cfg(), UniformEvaluator(0.5), np.random.default_rng(3))
    assert [p.z for p in first] == [p.z for p in second]


def test_buffer_evicts_oldest_first():
    buffer = ReplayBuffer(capacity=10)
    buffer.extend([_position(i) for i in range(25)])
    assert len(buffer) == 10


def test_buffer_sample_shapes():
    buffer = ReplayBuffer(capacity=100)
    buffer.extend([_position(i) for i in range(50)])
    obs, pi, z = buffer.sample(8, np.random.default_rng(0))
    assert obs.shape == (8, 11, 8, 8)
    assert pi.shape == (8, 4)
    assert z.shape == (8,)
    assert obs.dtype == np.float32
    assert np.allclose(pi.sum(axis=1), 1.0)


def test_buffer_sampling_applies_symmetries():
    # One position sampled many times must not always come back identical, or
    # the eightfold augmentation is not happening.
    buffer = ReplayBuffer(capacity=10)
    buffer.extend([_position(0)])
    seen = set()
    for seed in range(50):
        obs, _, _ = buffer.sample(1, np.random.default_rng(seed))
        seen.add(obs.tobytes())
    assert len(seen) > 1


def _position(index):
    return Position(
        snake=((3, 3), (2, 3), (1, 3)),
        food=(6, 6),
        direction=RIGHT,
        size=6,
        pi=np.array([0.4, 0.3, 0.0, 0.3]),
        z=index / 100,
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/test_selfplay.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'snake.selfplay'`.

- [ ] **Step 3: Write the implementation**

Create `snake/selfplay.py`:

```python
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
    """z(s) is the fraction of the whole board still to be filled from s."""
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
        for (snake, food, direction, pi, _), z in zip(pending[index], targets):
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
        self._cursor = len(self._items) % self.capacity

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
        for index, symmetry_index in zip(indices, symmetry_choices):
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_selfplay.py -v
```

Expected: all PASS. If `test_play_batch_returns_finished_games` hangs, `starvation_limit` is not reaching the env: with a uniform evaluator the agent will loop forever without it.

- [ ] **Step 5: Commit**

```bash
git add snake/selfplay.py tests/test_selfplay.py
git commit -m "feat: add self play, value targets, and augmenting replay buffer"
```

---

### Task 18: Training loop and metrics

**Files:**
- Create: `snake/train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes: everything from Tasks 12 through 17.
- Produces:
  - `snake.train.losses(logits, value, pi_target, z_target) -> dict[str, torch.Tensor]` with keys `policy`, `value`, `total`, `entropy`, `value_mae`
  - `snake.train.Trainer(cfg: RunConfig)` with `train_step(batch) -> dict[str, float]`, `run_iteration() -> dict[str, float]`, and `evaluate() -> dict`
  - `snake.train.NetworkAgent(evaluator, cfg: RunConfig, rng)` implementing `act(env) -> int` for arena evaluation

- [ ] **Step 1: Write the failing tests**

Create `tests/test_train.py`:

```python
import numpy as np
import torch

from snake.config import RunConfig
from snake.train import Trainer, losses


def cfg(tmp_path, **train):
    settings = {
        "board_size": 6,
        "concurrent_games": 2,
        "train_steps_per_iteration": 3,
        "batch_size": 8,
        "replay_capacity": 500,
        "eval_games": 2,
        "device": "cpu",
        "run_dir": str(tmp_path / "run"),
    }
    settings.update(train)
    return RunConfig.build(
        net={"channels": 16, "blocks": 2, "groups": 4},
        search={"simulations": 8},
        train=settings,
    )


def test_policy_loss_is_zero_when_the_prediction_is_perfect():
    target = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    logits = torch.tensor([[-50.0, 50.0, -50.0, -50.0]])
    value = torch.tensor([0.3])
    out = losses(logits, value, target, torch.tensor([0.3]))
    assert out["policy"].item() < 1e-4
    assert out["value"].item() < 1e-8


def test_policy_loss_does_not_softmax_twice():
    # Feeding an already normalized distribution as logits must NOT give zero
    # loss. If it does, a softmax is being applied to a softmax somewhere.
    target = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    out = losses(target.clone(), torch.tensor([0.5]), target, torch.tensor([0.5]))
    assert out["policy"].item() > 0.5


def test_entropy_is_reported_and_maximal_for_a_uniform_policy():
    logits = torch.zeros(1, 4)
    out = losses(logits, torch.tensor([0.5]), torch.full((1, 4), 0.25), torch.tensor([0.5]))
    assert np.isclose(out["entropy"].item(), np.log(4), atol=1e-5)


def test_value_mae_is_reported():
    out = losses(
        torch.zeros(2, 4),
        torch.tensor([0.2, 0.8]),
        torch.full((2, 4), 0.25),
        torch.tensor([0.4, 0.4]),
    )
    assert np.isclose(out["value_mae"].item(), 0.3, atol=1e-6)


def test_a_fixed_batch_can_be_overfit(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    rng = np.random.default_rng(0)
    obs = rng.random((8, 11, 8, 8)).astype(np.float32)
    pi = np.full((8, 4), 0.25, dtype=np.float32)
    pi[:, 1] = 0.7
    pi[:, [0, 2, 3]] = 0.1
    z = rng.random(8).astype(np.float32)
    batch = (obs, pi, z)

    first = trainer.train_step(batch)["total"]
    for _ in range(200):
        last = trainer.train_step(batch)["total"]
    assert last < first * 0.5


def test_run_iteration_fills_the_buffer_and_reports_metrics(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    metrics = trainer.run_iteration()
    assert len(trainer.buffer) > 0
    for key in ("policy", "value", "total", "entropy", "value_mae", "mean_score", "games_per_second"):
        assert key in metrics


def test_evaluate_returns_an_arena_summary(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    summary = trainer.evaluate()
    assert summary["games"] == 2
    assert set(summary["outcomes"]) == {"wall", "self", "starvation", "solved"}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/test_train.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'snake.train'`.

- [ ] **Step 3: Write the implementation**

Create `snake/train.py`:

```python
"""The optimizer loop, metrics, and evaluation."""

from __future__ import annotations

import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from snake.arena import play_games, summarize
from snake.config import seed_everything
from snake.evaluator import TorchEvaluator
from snake.mcts import Search, run_search
from snake.model import SnakeNet
from snake.selfplay import ReplayBuffer, play_batch, select_move

_SEED_MAX = 2**63 - 1


def losses(logits, value, pi_target, z_target):
    """Cross entropy against the visit distribution plus MSE on the value.

    log_softmax is applied here, so the policy head must emit raw logits. This
    is the double softmax that the earlier god.py sketch had.
    """
    log_probs = F.log_softmax(logits, dim=1)
    policy = -(pi_target * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(value, z_target)
    with torch.no_grad():
        entropy = -(log_probs.exp() * log_probs).sum(dim=1).mean()
        value_mae = (value - z_target).abs().mean()
    return {
        "policy": policy,
        "value": value_loss,
        "total": policy + value_loss,
        "entropy": entropy,
        "value_mae": value_mae,
    }


class NetworkAgent:
    """Wraps search so the arena can evaluate a checkpoint like any baseline."""

    def __init__(self, evaluator, cfg, rng):
        self.evaluator = evaluator
        self.cfg = cfg
        self.rng = rng
        self.move_index = 0

    def act(self, env):
        search = Search(
            env, self.cfg.search, np.random.default_rng(int(self.rng.integers(_SEED_MAX)))
        )
        run_search([search], self.evaluator, self.cfg.search.simulations)
        counts = search.visit_counts()
        self.move_index += 1
        return int(np.argmax(counts))


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        seed_everything(cfg.train.seed)
        self.device = torch.device(cfg.train.device)
        self.model = SnakeNet(cfg.net).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        self.evaluator = TorchEvaluator(self.model, self.device)
        self.buffer = ReplayBuffer(cfg.train.replay_capacity)
        self.rng = np.random.default_rng(cfg.train.seed)
        self.iteration = 0
        self.step = 0

        self.run_dir = pathlib.Path(cfg.train.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(cfg.to_json() + "\n")
        self.writer = SummaryWriter(str(self.run_dir / "tb"))

    def train_step(self, batch):
        obs, pi, z = batch
        self.model.train()
        tensors = [torch.from_numpy(a).to(self.device) for a in (obs, pi, z)]
        logits, value = self.model(tensors[0])
        out = losses(logits, value, tensors[1], tensors[2])
        self.optimizer.zero_grad(set_to_none=True)
        out["total"].backward()
        self.optimizer.step()
        self.step += 1
        return {key: float(tensor.item()) for key, tensor in out.items()}

    def run_iteration(self):
        started = time.perf_counter()
        self.model.eval()
        positions, results = play_batch(self.cfg, self.evaluator, self.rng)
        self.buffer.extend(positions)
        elapsed = time.perf_counter() - started

        metrics = {}
        for _ in range(self.cfg.train.train_steps_per_iteration):
            if len(self.buffer) < self.cfg.train.batch_size:
                break
            metrics = self.train_step(
                self.buffer.sample(self.cfg.train.batch_size, self.rng)
            )

        metrics.setdefault("policy", float("nan"))
        metrics.setdefault("value", float("nan"))
        metrics.setdefault("total", float("nan"))
        metrics.setdefault("entropy", float("nan"))
        metrics.setdefault("value_mae", float("nan"))
        metrics["mean_score"] = float(np.mean([r.score for r in results]))
        metrics["mean_steps"] = float(np.mean([r.steps for r in results]))
        metrics["games_per_second"] = len(results) / max(elapsed, 1e-9)
        metrics["buffer_size"] = float(len(self.buffer))
        metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        for key, value in metrics.items():
            self.writer.add_scalar(f"train/{key}", value, self.iteration)
        self.iteration += 1
        return metrics

    def evaluate(self):
        self.model.eval()
        size = self.cfg.train.board_size
        summary = summarize(
            play_games(
                lambda rng: NetworkAgent(self.evaluator, self.cfg, rng),
                size=size,
                n_games=self.cfg.train.eval_games,
                seed=self.cfg.train.seed,
            ),
            total_cells=size * size,
        )
        for key in ("mean_score", "median_score", "max_score", "mean_fill_fraction", "mean_steps"):
            self.writer.add_scalar(f"eval/{key}", summary[key], self.iteration)
        for outcome, count in summary["outcomes"].items():
            self.writer.add_scalar(f"eval/outcome_{outcome}", count, self.iteration)
        return summary
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_train.py -v
```

Expected: all PASS. `test_a_fixed_batch_can_be_overfit` is the one that proves the loss and optimizer are wired correctly; if it plateaus, check that `zero_grad` runs before `backward` and that `losses` returns tensors rather than floats for `policy` and `value`.

- [ ] **Step 5: Commit**

```bash
git add snake/train.py tests/test_train.py
git commit -m "feat: add the training loop, losses, and TensorBoard metrics"
```

---

### Task 19: Checkpointing and resume

**Files:**
- Modify: `snake/train.py`
- Create: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: `Trainer` from Task 18.
- Produces:
  - `Trainer.save_checkpoint(tag: str) -> pathlib.Path`
  - `Trainer.load_checkpoint(path) -> None`
  - `snake.train.load_for_inference(path, device) -> tuple[SnakeNet, RunConfig]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_checkpoint.py`:

```python
import numpy as np
import torch

from snake.config import RunConfig
from snake.train import Trainer, load_for_inference


def cfg(tmp_path):
    return RunConfig.build(
        net={"channels": 16, "blocks": 2, "groups": 4},
        search={"simulations": 4},
        train={
            "board_size": 6,
            "concurrent_games": 2,
            "train_steps_per_iteration": 2,
            "batch_size": 8,
            "replay_capacity": 200,
            "eval_games": 1,
            "keep_last": 2,
            "device": "cpu",
            "run_dir": str(tmp_path / "run"),
        },
    )


def test_save_then_load_restores_identical_outputs(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    trainer.run_iteration()
    path = trainer.save_checkpoint("test")

    probe = torch.from_numpy(np.random.default_rng(0).random((2, 11, 8, 8)).astype(np.float32))
    trainer.model.eval()
    with torch.no_grad():
        before = trainer.model(probe)

    fresh = Trainer(cfg(tmp_path))
    fresh.load_checkpoint(path)
    fresh.model.eval()
    with torch.no_grad():
        after = fresh.model(probe)

    assert torch.allclose(before[0], after[0], atol=1e-6)
    assert torch.allclose(before[1], after[1], atol=1e-6)


def test_resume_restores_the_counters_and_buffer(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    trainer.run_iteration()
    trainer.run_iteration()
    path = trainer.save_checkpoint("test")

    fresh = Trainer(cfg(tmp_path))
    fresh.load_checkpoint(path)
    assert fresh.iteration == trainer.iteration
    assert fresh.step == trainer.step
    assert len(fresh.buffer) == len(trainer.buffer)


def test_optimizer_state_survives_a_round_trip(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    trainer.run_iteration()
    path = trainer.save_checkpoint("test")
    fresh = Trainer(cfg(tmp_path))
    fresh.load_checkpoint(path)
    original = trainer.optimizer.state_dict()["state"]
    restored = fresh.optimizer.state_dict()["state"]
    assert set(original) == set(restored)
    for key in original:
        assert torch.allclose(original[key]["exp_avg"], restored[key]["exp_avg"])


def test_only_keep_last_checkpoints_are_retained(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    for index in range(5):
        trainer.run_iteration()
        trainer.save_checkpoint(f"iter{index}")
    kept = sorted((trainer.run_dir / "checkpoints").glob("iter*.pt"))
    assert len(kept) == 2


def test_load_for_inference_needs_no_trainer(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    trainer.run_iteration()
    path = trainer.save_checkpoint("test")
    model, restored_cfg = load_for_inference(path, torch.device("cpu"))
    assert restored_cfg.net.channels == 16
    assert model.training is False
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/test_checkpoint.py -v
```

Expected: FAIL with `AttributeError: 'Trainer' object has no attribute 'save_checkpoint'`.

- [ ] **Step 3: Write the implementation**

Append to `snake/train.py`:

```python
    def save_checkpoint(self, tag):
        directory = self.run_dir / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{tag}.pt"
        torch.save(
            {
                "config": self.cfg.to_json(),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iteration": self.iteration,
                "step": self.step,
                "buffer": self.buffer.items(),
            },
            path,
        )
        self._prune_checkpoints(directory)
        return path

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.iteration = payload["iteration"]
        self.step = payload["step"]
        self.buffer.load(payload["buffer"])

    def _prune_checkpoints(self, directory):
        # "best.pt" is kept regardless of age; only the rolling ones are pruned.
        rolling = sorted(
            (p for p in directory.glob("*.pt") if p.stem != "best"),
            key=lambda p: (p.stat().st_mtime, p.name),
        )
        for stale in rolling[: max(0, len(rolling) - self.cfg.train.keep_last)]:
            stale.unlink()


def load_for_inference(path, device):
    """Rebuild a network from a checkpoint without constructing a Trainer, which
    is what the menu integration in a later phase will use."""
    from snake.config import RunConfig

    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = RunConfig.from_json(payload["config"])
    model = SnakeNet(cfg.net).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_checkpoint.py -v
```

Expected: all PASS. If `test_only_keep_last_checkpoints_are_retained` sees three files, the modification times are too close to order reliably; make `save_checkpoint` sort by an explicit saved iteration instead of `st_mtime`.

- [ ] **Step 5: Commit**

```bash
git add snake/train.py tests/test_checkpoint.py
git commit -m "feat: add checkpoint saving, pruning, and resume"
```

---

### Task 20: Training CLI and removing god.py

**Files:**
- Modify: `snake/cli.py`
- Delete: `god.py`
- Modify: `requirements.txt`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Trainer`, `RunConfig`, `load_for_inference`.
- Produces:
  - `python -m snake.cli train --board-size 6 --iterations N --run-dir runs/NAME [--resume PATH]`
  - `python -m snake.cli eval --checkpoint PATH --games N`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
import json
import pathlib

from snake.cli import main


def test_train_writes_a_config_and_a_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    main([
        "train",
        "--board-size", "6",
        "--iterations", "1",
        "--simulations", "4",
        "--concurrent-games", "2",
        "--train-steps", "2",
        "--batch-size", "8",
        "--channels", "16",
        "--blocks", "2",
        "--groups", "4",
        "--eval-games", "1",
        "--device", "cpu",
        "--run-dir", str(run_dir),
    ])
    config = json.loads((run_dir / "config.json").read_text())
    assert config["train"]["board_size"] == 6
    assert list((run_dir / "checkpoints").glob("*.pt"))


REPO = pathlib.Path(__file__).resolve().parent.parent


def test_god_py_is_gone():
    assert not (REPO / "god.py").exists()


def test_requirements_stay_free_of_torch():
    text = (REPO / "requirements.txt").read_text()
    assert "torch" not in text
    assert "numpy" in text and "pygame" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./venv/bin/python -m pytest tests/test_cli.py -v
```

Expected: FAIL on the missing `train` subcommand and on `god.py` still existing.

- [ ] **Step 3: Extend the CLI**

Add to `snake/cli.py`:

```python
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
            "seed": args.seed,
            "device": args.device,
            "run_dir": args.run_dir,
        },
    )
    trainer = Trainer(cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"resumed from {args.resume} at iteration {trainer.iteration}")

    best = -1.0
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
            if summary["mean_score"] > best:
                best = summary["mean_score"]
                trainer.save_checkpoint("best")
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
```

Register both in `main`, before `args = parser.parse_args(argv)`:

```python
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
```

- [ ] **Step 4: Delete the superseded sketch**

`god.py` has never run end to end and every part of it is now replaced: the network by `snake/model.py`, the search by `snake/mcts.py`, the trainer by `snake/train.py`. Git history keeps it.

```bash
git rm god.py
```

- [ ] **Step 5: Confirm the runtime dependencies did not grow**

```bash
cat requirements.txt
```

Expected: numpy and pygame only. If torch appears, remove it; `app.py` must keep running without the training stack.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_cli.py -v
```

Expected: all PASS.

- [ ] **Step 7: Confirm the game still runs without torch on the import path**

```bash
./venv/bin/python -c "import app; print('app imports cleanly')"
./venv/bin/python -m snake.cli baselines --sizes 6 --games 5 --out /tmp/nodeps.json
```

Expected: both succeed.

- [ ] **Step 8: Commit**

```bash
git add snake/cli.py tests/test_cli.py requirements.txt
git commit -m "feat: add train and eval commands, remove the superseded god.py sketch"
```

---

### Task 21: Phase 1 acceptance run

**Files:**
- Create: `results/phase1-6x6.json` (generated)
- Modify: `docs/superpowers/plans/2026-08-24-alphazero-snake-phase-0-1.md` (record the result)

**Interfaces:**
- Consumes: the whole package.
- Produces: a trained 6x6 checkpoint and its scored comparison against the baselines.

- [ ] **Step 1: Run the full test suite**

```bash
./venv/bin/python -m pytest -v
```

Expected: every test passes. Do not start a long training run on a red suite.

- [ ] **Step 2: Smoke test the loop on the GPU for two iterations**

```bash
./venv/bin/python -m snake.cli train --board-size 6 --iterations 2 \
  --simulations 25 --concurrent-games 8 --train-steps 20 \
  --run-dir runs/smoke
```

Expected: two iteration lines print, `runs/smoke/config.json` exists, and a checkpoint appears. Note the `games/s` figure: it is the number that decides whether the full run is hours or days.

- [ ] **Step 3: Launch the real run**

```bash
./venv/bin/python -m snake.cli train --board-size 6 --iterations 200 \
  --run-dir runs/6x6-seed0 2>&1 | tee runs/6x6-seed0/train.log
```

- [ ] **Step 4: Watch the metrics**

```bash
./venv/bin/tensorboard --logdir runs
```

Three things to check as it goes:

- `train/entropy` should fall from near `log(4)` and settle well above zero. Reaching zero early means policy collapse, and the fix is a larger `dirichlet_epsilon` or a longer `tau_threshold`.
- `train/value_mae` should fall steadily. A flat line means the value head is not learning and the target computation is suspect.
- `eval/mean_score` should rise. `eval/outcome_starvation` dominating means the agent has learned to survive without eating, which the value target should penalize; if it persists, that is a genuine finding worth reporting rather than a bug to patch.

- [ ] **Step 5: Score the result against the baselines**

```bash
./venv/bin/python -m snake.cli eval --checkpoint runs/6x6-seed0/checkpoints/best.pt \
  --games 100 --seed 0 > results/phase1-6x6.json
cat results/phase1-6x6.json
./venv/bin/python -c "
import json
baselines = json.load(open('results/baselines.json'))['6']
agent = json.load(open('results/phase1-6x6.json'))
print(f\"agent    {agent['mean_score']:8.2f}\")
for name, summary in baselines.items():
    print(f\"{name:<8} {summary['mean_score']:8.2f}\")
"
```

- [ ] **Step 6: Record the outcome in this plan**

Append a short section to the end of this file stating the agent's mean score, the four baseline scores on the same seeds, and which of them the agent beat. Report it plainly whether or not it cleared the bar.

- [ ] **Step 7: Commit**

```bash
git add results/phase1-6x6.json docs/superpowers/plans/2026-08-24-alphazero-snake-phase-0-1.md
git commit -m "chore: record the phase 1 result on 6x6 against the baselines"
```

**Phase 1 is accepted when:** the trained agent's mean score on 6x6 exceeds the greedy baseline on the same seeds, `--resume` restores a run correctly, and TensorBoard shows `train/value_mae` decreasing without `train/entropy` collapsing to zero.

If the agent does not beat greedy, that is a result, not a failure of the plan. Report it with the death cause histogram, which says which failure mode it is stuck on, and treat the next step as a research question rather than a bug hunt.

---

## Phase 1 result, recorded 2026-08-30

Run: `runs/6x6-seed0`, 200 iterations on a 6x6 board, seed 0, default hyperparameters.
Scored with `snake.cli eval` on `best.pt` over 100 deterministic games at seed 0, against the baselines recorded in `results/baselines.json` on the same seeds.

| Agent | Mean | Median | Max | Fill | Outcomes |
|---|---|---|---|---|---|
| alphazero | 7.29 | 6.0 | 26 | 0.286 | wall 7, self 15, starvation 78, solved 0 |
| random | 0.23 | 0.0 | 2 | 0.090 | wall 99, self 1 |
| greedy | 12.53 | 12.0 | 25 | 0.431 | wall 1, self 99 |
| bfs_safe | 20.92 | 23.0 | 33 | 0.664 | wall 16, self 22, starvation 48, solved 14 |
| hamiltonian | 33.00 | 33.0 | 33 | 1.000 | solved 100 |

The agent beats random decisively and does not beat greedy.
Phase 1's acceptance bar was beating greedy, so the bar was not met.

The learning curve is real but plateaus early.
Evaluation mean score rose from 0.92 to about 6.1 by iteration 50, then sat in the 6.6 to 7.1 band for the remaining 150 iterations.
Policy entropy settled at 1.042 against a uniform value of log 4 = 1.386, so the policy stayed comparatively undecided rather than collapsing.
Value mean absolute error settled around 0.065 and stopped improving.

The death cause histogram is the informative part, and it is why Phase 0 built one.
Starvation accounts for 78% of the agent's games, against 48% for BFS and 0% for greedy.
The agent learned to stop dying and did not learn to forage: its wall and self collision rates, at 7% and 15%, are far below greedy's 99% self collision rate, so it is genuinely better at survival than the heuristic that beats it on score.
Its maximum of 26 also exceeds greedy's 25, so it can occasionally play well; the median of 6 against greedy's 12 is where the gap lives.

This is a result rather than a defect, and the next step is a research question.
The most likely candidates, in the order the evidence supports them: the value target rewards eating eventually but never sooner, so nothing distinguishes a state that eats in three moves from one that eats in thirty; a search budget of 100 simulations may not see far enough on a 6x6 board for the food to enter the horizon at all once the snake is short; and the starvation truncation at 72 steps may be arriving before the agent's own search would have resolved the position.
None of these were investigated here, because Phase 1's scope ended at measuring the result honestly.
