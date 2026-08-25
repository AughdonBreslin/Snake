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
