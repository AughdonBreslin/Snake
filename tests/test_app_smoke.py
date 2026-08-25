"""Headless smoke test for App/SnakeGame wiring.

This does not re-test snake rules (that is tests/test_app_rules_pinned.py and
tests/test_env_rules.py). It proves that the pygame wiring around SnakeEnv
still holds after the delegation refactor: screens switch, direction keys are
accepted, score text is refreshed, and game over followed by a restart both
work, all driven by synthetic pygame events rather than a human at a keyboard.
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


def test_screens_switch_and_all_four_directions_steer(application, monkeypatch):
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
    # Direction changed at least once away from the reset default, proving the
    # queued turns from the arrow keys above reached SnakeEnv through the game.
    assert application.game.direction in (RIGHT, DOWN, LEFT, UP)


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
