import numpy as np
import pytest

from snake.env import DOWN, RIGHT, SnakeEnv


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
