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
