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
