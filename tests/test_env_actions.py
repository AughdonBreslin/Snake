import numpy as np
import pytest

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


def test_stepping_a_reversal_raises():
    env = make()
    with pytest.raises(ValueError):
        env.step(OPPOSITE[env.direction])
