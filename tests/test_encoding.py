import numpy as np

from snake.encoding import N_PLANES, encode
from snake.env import RIGHT, SnakeEnv


FOOD = (4, 1)


def sample():
    snake = [(3, 2), (2, 2), (1, 2)]
    return encode(size=4, snake=snake, food=FOOD, direction=RIGHT), snake


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
    assert obs[3, FOOD[1], FOOD[0]] == 1.0
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
