"""Headless smoke test for App/SnakeGame wiring.

This does not re-test snake rules (that is tests/test_app_rules_pinned.py and
tests/test_env_rules.py). It proves that the pygame wiring around SnakeEnv
still holds after the delegation refactor: screens switch, the full run loop
does not raise, score text is refreshed, and game over followed by a restart
both work, all driven by synthetic pygame events rather than a human at a
keyboard.

Steering itself (that each arrow key and its WASD alias actually turns the
snake) is covered directly against SnakeGame by
test_direction_updates_after_each_arrow_key and
test_direction_updates_after_each_wasd_key below, which check game.direction
after every single key rather than once at the end of a long script.
"""

from collections import deque

import pygame
import pytest

import app
from snake.env import DOWN, LEFT, RIGHT, UP


class ScriptedEvents:
    """Stands in for pygame.event.get(). Each call returns the next scripted
    batch of events, so one call to App.run()'s loop body advances exactly one
    simulated frame. Once the script is exhausted, no more calls are expected
    because the batch that sets running to False is always the last one."""

    def __init__(self, batches):
        self._batches = list(batches)

    def __call__(self):
        return self._batches.pop(0)


def keydown(key):
    return [pygame.event.Event(pygame.KEYDOWN, key=key)]


@pytest.fixture
def application(window, monkeypatch):
    # App.run() calls pygame.quit() when its loop ends, which would tear down
    # the session-wide dummy display that later tests rely on. Neutralize it
    # for the duration of this test only.
    monkeypatch.setattr(pygame, "quit", lambda: None)
    a = app.App()
    # Uncapped: every scripted frame reaches play()/draw regardless of real
    # wall clock time between iterations. Settings.event() persists self.fps
    # to settings.txt on every keypress while the settings screen is active,
    # so silence that write for the test; settings persistence is pre-existing
    # behavior unrelated to this task and must not touch the repo's file.
    a.settings.fps = 0
    monkeypatch.setattr(a.settings, "write_settings", lambda difficulty: None)
    return a


def test_screens_switch_and_the_run_loop_does_not_raise(application, monkeypatch):
    # This exercises the full App.run() loop across every screen transition,
    # including the arrow keys reaching the game screen along the way. It does
    # not itself assert anything about steering: the scripted turns below
    # cycle RIGHT -> DOWN -> LEFT -> UP -> RIGHT, so the direction at the end
    # is indistinguishable from the never-turned default, and a final-value
    # check here would prove nothing. Per-key steering is covered separately
    # by test_direction_updates_after_each_arrow_key and its WASD twin, which
    # assert direction after each individual key.
    script = (
        [keydown(pygame.K_RETURN)]  # home -> game
        + [keydown(pygame.K_DOWN)]
        + [keydown(pygame.K_LEFT)]
        + [keydown(pygame.K_UP)]
        + [keydown(pygame.K_RIGHT)]
        + [[] for _ in range(10)]  # let the queued turns actually apply
        + [keydown(pygame.K_q)]  # game -> settings
        + [keydown(pygame.K_ESCAPE)]  # settings -> home
        + [keydown(pygame.K_LALT)]  # home -> leaderboard
        + [keydown(pygame.K_ESCAPE)]  # leaderboard -> home
        + [keydown(pygame.K_BACKSPACE)]  # exit App.run()
    )
    monkeypatch.setattr(pygame.event, "get", ScriptedEvents(script))

    application.run()  # must not raise

    assert application.curr_screen == "home"


def test_direction_updates_after_each_arrow_key(game):
    # Drive SnakeGame directly rather than through App.run(), since the run
    # loop gives no place to assert mid-script. Plant the snake in the middle
    # of the board, alone, with food tucked in a far corner, so a full
    # RIGHT -> DOWN -> LEFT -> UP -> RIGHT loop of turns can run without
    # hitting a wall, running into its own body, or eating and growing.
    mid = app.GRID_HEIGHT // 2
    game.env.snake = deque([(mid, mid)])
    game.env._occupied = set(game.env.snake)
    game.env.direction = RIGHT
    game.env.food = (1, 1)
    game.move_queue = []

    for key, expected in (
        (pygame.K_DOWN, DOWN),
        (pygame.K_LEFT, LEFT),
        (pygame.K_UP, UP),
        (pygame.K_RIGHT, RIGHT),
    ):
        game.event(key)
        over = game.move()
        assert over is False
        assert game.direction == expected


def test_direction_updates_after_each_wasd_key(game):
    # Same legal turn sequence as the arrow-key test above, but through the
    # WASD aliases that SnakeGame.event also accepts.
    mid = app.GRID_HEIGHT // 2
    game.env.snake = deque([(mid, mid)])
    game.env._occupied = set(game.env.snake)
    game.env.direction = RIGHT
    game.env.food = (1, 1)
    game.move_queue = []

    for key, expected in (
        (pygame.K_s, DOWN),
        (pygame.K_a, LEFT),
        (pygame.K_w, UP),
        (pygame.K_d, RIGHT),
    ):
        game.event(key)
        over = game.move()
        assert over is False
        assert game.direction == expected


def test_game_over_then_restart(application):
    application.curr_screen = "game"
    game = application.game
    mid = app.GRID_HEIGHT // 2

    # Plant the snake one cell from the right wall, heading toward it, using
    # the same technique as the pinned rule tests: env.snake is a deque and
    # _occupied must be rebuilt after a direct body assignment.
    game.env.snake = deque([(app.GRID_WIDTH - 2, mid)])
    game.env._occupied = set(game.env.snake)
    game.env.direction = RIGHT
    game.move_queue = []

    assert game.game_over is False
    application.game.play()
    assert game.game_over is True

    # Enter restarts from the game over screen.
    score_text_before_restart = game.score_text
    game.event(pygame.K_RETURN)
    assert game.game_over is False
    assert game.score == 0
    assert list(game.snake) == [(3, mid), (2, mid), (1, mid)]
    # A fresh render happened; the score text object was rebuilt, not reused.
    assert game.score_text is not score_text_before_restart

    # The game keeps playing normally after the restart.
    application.game.play()
    assert game.game_over is False
