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
