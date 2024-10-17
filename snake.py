import pygame
import random
from enum import Enum
from collections import deque
import os
import sys


# Constants
WINDOW_WIDTH = 440
WINDOW_HEIGHT = 440
CELL_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
PATH = "/mnt/c/Users/aughb/Personal_Projects/Snake/"
HAMILTONIAN_CYCLE = [
    ( 7,  9), ( 7, 10), ( 7, 11), ( 7, 12), ( 8, 12),
    ( 8, 11), ( 8, 10), ( 8,  9), ( 8,  8), ( 9,  8),
    ( 9,  9), ( 9, 10), (10, 10), (11, 10), (11, 11),
    (11, 12), (11, 13), (11, 14), (10, 14), (10, 13),
    (10, 12), (10, 11), ( 9, 11), ( 9, 12), ( 9, 13),
    ( 8, 13), ( 8, 14), ( 9, 14), ( 9, 15), (10, 15),
    (11, 15), (12, 15), (12, 14), (13, 14), (14, 14),
    (15, 14), (15, 13), (14, 13), (14, 12), (14, 11),
    (13, 11), (13, 12), (13, 13), (12, 13), (12, 12),
    (12, 11), (12, 10), (13, 10), (14, 10), (15, 10),
    (15, 11), (15, 12), (16, 12), (16, 11), (16, 10),
    (16,  9), (15,  9), (15,  8), (15,  7), (14,  7),
    (14,  8), (14,  9), (13,  9), (13,  8), (13,  7),
    (13,  6), (12,  6), (12,  7), (11,  7), (11,  8),
    (12,  8), (12,  9), (11,  9), (10,  9), (10,  8),
    (10,  7), ( 9,  7), ( 8,  7), ( 7,  7), ( 6,  7),
    ( 6,  6), ( 6,  5), ( 7,  5), ( 7,  6), ( 8,  6),
    ( 9,  6), (10,  6), (11,  6), (11,  5), (10,  5),
    (10,  4), ( 9,  4), ( 9,  5), ( 8,  5), ( 8,  4),
    ( 8,  3), ( 7,  3), ( 7,  4), ( 6,  4), ( 6,  3),
    ( 5,  3), ( 4,  3), ( 4,  2), ( 3,  2), ( 2,  2),
    ( 2,  1), ( 2,  0), ( 3,  0), ( 3,  1), ( 4,  1),
    ( 4,  0), ( 5,  0), ( 6,  0), ( 6,  1), ( 5,  1),
    ( 5,  2), ( 6,  2), ( 7,  2), ( 8,  2), ( 8,  1),
    ( 7,  1), ( 7,  0), ( 8,  0), ( 9,  0), ( 9,  1),
    (10,  1), (10,  0), (11,  0), (12,  0), (13,  0),
    (13,  1), (13,  2), (13,  3), (12,  3), (12,  2),
    (12,  1), (11,  1), (11,  2), (10,  2), ( 9,  2),
    ( 9,  3), (10,  3), (11,  3), (11,  4), (12,  4),
    (12,  5), (13,  5), (13,  4), (14,  4), (14,  3),
    (14,  2), (14,  1), (14,  0), (15,  0), (15,  1),
    (15,  2), (15,  3), (15,  4), (16,  4), (16,  3),
    (17,  3), (18,  3), (18,  2), (17,  2), (16,  2),
    (16,  1), (16,  0), (17,  0), (17,  1), (18,  1),
    (18,  0), (19,  0), (19,  1), (19,  2), (19,  3),
    (19,  4), (18,  4), (17,  4), (17,  5), (16,  5),
    (15,  5), (14,  5), (14,  6), (15,  6), (16,  6),
    (17,  6), (18,  6), (18,  5), (19,  5), (19,  6),
    (19,  7), (18,  7), (17,  7), (16,  7), (16,  8),
    (17,  8), (18,  8), (19,  8), (19,  9), (18,  9),
    (17,  9), (17, 10), (18, 10), (19, 10), (19, 11),
    (19, 12), (19, 13), (19, 14), (19, 15), (19, 16),
    (18, 16), (18, 15), (17, 15), (17, 14), (18, 14),
    (18, 13), (18, 12), (18, 11), (17, 11), (17, 12),
    (17, 13), (16, 13), (16, 14), (16, 15), (15, 15),
    (15, 16), (16, 16), (17, 16), (17, 17), (17, 18),
    (18, 18), (18, 17), (19, 17), (19, 18), (19, 19),
    (18, 19), (17, 19), (16, 19), (16, 18), (16, 17),
    (15, 17), (15, 18), (15, 19), (14, 19), (14, 18),
    (13, 18), (13, 19), (12, 19), (12, 18), (11, 18),
    (11, 19), (10, 19), (10, 18), (10, 17), (11, 17),
    (12, 17), (13, 17), (14, 17), (14, 16), (14, 15),
    (13, 15), (13, 16), (12, 16), (11, 16), (10, 16),
    ( 9, 16), ( 8, 16), ( 8, 15), ( 7, 15), ( 7, 14),
    ( 7, 13), ( 6, 13), ( 5, 13), ( 5, 12), ( 6, 12),
    ( 6, 11), ( 5, 11), ( 4, 11), ( 4, 12), ( 4, 13),
    ( 3, 13), ( 3, 12), ( 3, 11), ( 3, 10), ( 4, 10),
    ( 5, 10), ( 6, 10), ( 6,  9), ( 5,  9), ( 4,  9),
    ( 3,  9), ( 2,  9), ( 1,  9), ( 1, 10), ( 2, 10),
    ( 2, 11), ( 1, 11), ( 1, 12), ( 2, 12), ( 2, 13),
    ( 2, 14), ( 2, 15), ( 1, 15), ( 1, 16), ( 1, 17),
    ( 1, 18), ( 2, 18), ( 3, 18), ( 3, 17), ( 2, 17),
    ( 2, 16), ( 3, 16), ( 3, 15), ( 3, 14), ( 4, 14),
    ( 4, 15), ( 5, 15), ( 5, 14), ( 6, 14), ( 6, 15),
    ( 6, 16), ( 7, 16), ( 7, 17), ( 8, 17), ( 9, 17),
    ( 9, 18), ( 9, 19), ( 8, 19), ( 8, 18), ( 7, 18),
    ( 7, 19), ( 6, 19), ( 6, 18), ( 6, 17), ( 5, 17),
    ( 5, 16), ( 4, 16), ( 4, 17), ( 4, 18), ( 5, 18),
    ( 5, 19), ( 4, 19), ( 3, 19), ( 2, 19), ( 1, 19),
    ( 0, 19), ( 0, 18), ( 0, 17), ( 0, 16), ( 0, 15),
    ( 0, 14), ( 1, 14), ( 1, 13), ( 0, 13), ( 0, 12),
    ( 0, 11), ( 0, 10), ( 0,  9), ( 0,  8), ( 0,  7),
    ( 0,  6), ( 1,  6), ( 1,  7), ( 1,  8), ( 2,  8),
    ( 2,  7), ( 2,  6), ( 2,  5), ( 2,  4), ( 1,  4),
    ( 1,  5), ( 0,  5), ( 0,  4), ( 0,  3), ( 0,  2),
    ( 0,  1), ( 0,  0), ( 1,  0), ( 1,  1), ( 1,  2),
    ( 1,  3), ( 2,  3), ( 3,  3), ( 3,  4), ( 3,  5),
    ( 4,  5), ( 4,  4), ( 5,  4), ( 5,  5), ( 5,  6),
    ( 5,  7), ( 4,  7), ( 4,  6), ( 3,  6), ( 3,  7),
    ( 3,  8), ( 4,  8), ( 5,  8), ( 6,  8), ( 7,  8)
]

# Define Direction enum
class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

UP, DOWN, LEFT, RIGHT = Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT

class SnakeGame:
    def __init__(self, left):
        pygame.init()
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()

        self.setup()
        self.font = pygame.font.Font(os.path.join(PATH, "fira.ttf"), 20)
        
        if left == "l":
            pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.NOFRAME)
            os.environ['SDL_VIDEO_WINDOW_POS'] = "100,100"
        else:
            pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.NOFRAME)
            os.environ['SDL_VIDEO_WINDOW_POS'] = "1380,100"

        self.score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.high_score_text = self.font.render(f"High Score: {self.high_score}", True, (255, 255, 255))
        
        menu_font = pygame.font.Font(os.path.join(PATH, "fira.ttf"), 30)

    def render_multiline_text(self, text, color, x, y, line_spacing=5):
        lines = text.split('\n')
        for i, line in enumerate(lines):
            text_surface = self.font.render(line, True, color)
            self.window.blit(text_surface, (x, y + i * (self.font.get_height() + line_spacing)))

    def setup(self):
        self.snake = [(3, 3), (2, 3), (1, 3)]
        self.direction = RIGHT
        self.move_queue = deque()
        self.game_over = False
        self.game_started = False
        self.settings = False
        self.leaderboard = False
        self.score = 0
        self.high_score = 0
        self.offset = 0
        self.frame_rate = 10
        self.spawn_food()
        self.read_settings()
        self.read_high_score()

    def spawn_food(self):
        while True:
            self.food = (random.randint(1, GRID_WIDTH), random.randint(1, GRID_HEIGHT))
            if self.food not in self.snake:
                break

    def read_high_score(self):
        try:
            with open(os.path.join(PATH, "highscore.txt"), "r") as file:
                self.high_score = int(file.read())
        except (FileNotFoundError, ValueError):
            self.high_score = 0

    def write_high_score(self):
        with open(os.path.join(PATH, "highscore.txt"), "w") as file:
            file.write(str(self.high_score))

    def read_settings(self):
        try:
            with open(os.path.join(PATH, "settings.txt"), "r") as file:
                self.frame_rate = int(file.read())
        except (FileNotFoundError, ValueError):
            self.frame_rate = 10

    def write_settings(self):
        with open(os.path.join(PATH, "settings.txt"), "w") as file:
            file.write(str(self.frame_rate))

    def move(self):
        if self.move_queue:
            self.direction = self.move_queue.popleft()

        head = self.snake[0]
        if self.direction == UP:
            new_head = (head[0], head[1] - 1)
        elif self.direction == DOWN:
            new_head = (head[0], head[1] + 1)
        elif self.direction == LEFT:
            new_head = (head[0] - 1, head[1])
        else:  # RIGHT
            new_head = (head[0] + 1, head[1])

        if (new_head[0] < 1 or new_head[0] > GRID_WIDTH or
            new_head[1] < 1 or new_head[1] > GRID_HEIGHT or
            new_head in self.snake[1:]):
            self.game_over = True
            return

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.spawn_food()
        else:
            self.snake.pop()

    def draw_background(self):
        for i in range(1, GRID_WIDTH + 1):
            for j in range(1, GRID_HEIGHT + 1):
                color = (50, 50, 50) if (i + j) % 2 == 0 else (40, 40, 40)
                pygame.draw.rect(self.window, color, (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def draw_walls(self):
        wall_color = (100, 100, 100)
        pygame.draw.rect(self.window, wall_color, (0, 0, WINDOW_WIDTH, CELL_SIZE))
        pygame.draw.rect(self.window, wall_color, (0, WINDOW_HEIGHT - CELL_SIZE, WINDOW_WIDTH, CELL_SIZE))
        pygame.draw.rect(self.window, wall_color, (0, 0, CELL_SIZE, WINDOW_HEIGHT))
        pygame.draw.rect(self.window, wall_color, (WINDOW_WIDTH - CELL_SIZE, 0, CELL_SIZE, WINDOW_HEIGHT))

    def draw_scores(self):
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        high_score_text = self.font.render(f"High Score: {self.high_score}", True, (255, 255, 255))
        self.window.blit(score_text, (10, -3))
        self.window.blit(high_score_text, (WINDOW_WIDTH - 200, -3))

    def draw_snake(self):
        for i, pos in enumerate(self.snake):
            color = (50 + pos[0] * 10, 250 - pos[1] * 10, 0)
            pygame.draw.rect(self.window, color, (pos[0] * CELL_SIZE, pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def draw_food(self):
        pygame.draw.rect(self.window, (255, 0, 0), (self.food[0] * CELL_SIZE, self.food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if not self.game_started:
                        if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                            self.game_started = True
                        elif event.key in (pygame.K_LALT, pygame.K_RALT):
                            self.leaderboard = True
                        elif event.key == pygame.K_q:
                            self.settings = True
                        elif event.key == pygame.K_ESCAPE and (self.settings or self.leaderboard):
                            self.settings = False
                            self.leaderboard = False
                    elif self.game_started and not self.game_over:
                        if event.key in (pygame.K_UP, pygame.K_w) and self.direction != DOWN:
                            self.move_queue.append(UP)
                        elif event.key in (pygame.K_DOWN, pygame.K_s) and self.direction != UP:
                            self.move_queue.append(DOWN)
                        elif event.key in (pygame.K_LEFT, pygame.K_a) and self.direction != RIGHT:
                            self.move_queue.append(LEFT)
                        elif event.key in (pygame.K_RIGHT, pygame.K_d) and self.direction != LEFT:
                            self.move_queue.append(RIGHT)
                    elif self.game_over:
                        if event.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_ESCAPE):
                            self.setup()
                    if event.key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                        running = False

            self.window.fill((0, 0, 0))

            if not self.game_started:
                if self.settings:
                    self.draw_background()
                    self.draw_settings_background()
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_1]:
                        self.frame_rate = 5
                    elif keys[pygame.K_2]:
                        self.frame_rate = 10
                    elif keys[pygame.K_3]:
                        self.frame_rate = 15
                    elif keys[pygame.K_4]:
                        self.frame_rate = 20
                    elif keys[pygame.K_5]:
                        self.frame_rate = 25
                    self.write_settings()
                elif self.leaderboard:
                    self.draw_leaderboard_background()
                else:
                    self.draw_menu_background()
            elif self.game_started and not self.game_over:
                self.draw_walls()
                self.draw_background()
                self.move()
                self.draw_snake()
                self.draw_food()
                self.draw_scores()
            else:
                self.draw_game_over_background()
                if self.score > self.high_score:
                    self.high_score = self.score
                    self.write_high_score()

            pygame.display.flip()
            self.clock.tick(self.frame_rate)

        pygame.quit()

    def draw_settings_background(self):
        for i in range(400):
            mod_offset = (i + self.offset) % 400
            
            if 0 < mod_offset < 10:
                color = (150 + 10 * mod_offset, 0, 0)
                x = (HAMILTONIAN_CYCLE[i % 400][0] + 1) * CELL_SIZE
                y = (HAMILTONIAN_CYCLE[i % 400][1] + 1) * CELL_SIZE + 1

            elif 50 < mod_offset < 60:
                color = (0, 150 + 10 * (mod_offset - 50), 0)
                x = (HAMILTONIAN_CYCLE[i % 400][0] + 1) * CELL_SIZE
                y = (HAMILTONIAN_CYCLE[i % 400][1] + 1) * CELL_SIZE + 1

            elif 100 < mod_offset < 110:
                color = (0, 0, 150 + 10 * (mod_offset - 100))
                x = (HAMILTONIAN_CYCLE[i % 400][0] + 1) * CELL_SIZE
                y = (HAMILTONIAN_CYCLE[i % 400][1] + 1) * CELL_SIZE + 1

            elif 150 < mod_offset < 160:
                color = (150 + 10 * (mod_offset - 150), 150 + 10 * (mod_offset - 150), 0)
                x = (HAMILTONIAN_CYCLE[i % 400][0] + 1) * CELL_SIZE
                y = (HAMILTONIAN_CYCLE[i % 400][1] + 1) * CELL_SIZE + 1

            elif 200 < mod_offset < 210:
                color = (150 + 10 * (mod_offset - 200), 0, 150 + 10 * (mod_offset - 200))
                x = (HAMILTONIAN_CYCLE[i % 400][0] + 1) * CELL_SIZE
                y = (HAMILTONIAN_CYCLE[i % 400][1] + 1) * CELL_SIZE + 1

            elif 250 < mod_offset < 260:
                color = (0, 150 + 10 * (mod_offset - 250), 150 + 10 * (mod_offset - 250))
                x = (HAMILTONIAN_CYCLE[i % 400][0] + 1) * CELL_SIZE
                y = (HAMILTONIAN_CYCLE[i % 400][1] + 1) * CELL_SIZE + 1

            elif 300 < mod_offset < 310:
                color = (150 + 10 * (mod_offset - 300), 150 + 10 * (mod_offset - 300), 150 + 10 * (mod_offset - 300))
                x = (HAMILTONIAN_CYCLE[i % 400][0] + 1) * CELL_SIZE
                y = (HAMILTONIAN_CYCLE[i % 400][1] + 1) * CELL_SIZE + 1

            elif 350 < mod_offset < 360:
                color = (150 - 10 * (mod_offset - 350), 150 - 10 * (mod_offset - 350), 150 - 10 * (mod_offset - 350))
                x = (HAMILTONIAN_CYCLE[i % 400][0] + 1) * CELL_SIZE
                y = (HAMILTONIAN_CYCLE[i % 400][1] + 1) * CELL_SIZE + 1

            else:
                color = (0, 0, 0)
                x, y = 0, 0
            
            pygame.draw.rect(self.window, color, (x, y, CELL_SIZE, CELL_SIZE))
        
        self.offset = (self.offset + 1) % 400
        self.render_multiline_text("Settings\nDifficulty: 1 2 3 4 5\nMove: WASD/Arrows\nBack: Escape", (255, 255, 255), 10, 10)

    def draw_leaderboard_background(self):
        for i in range(1, GRID_WIDTH + 1):
            for j in range(1, GRID_HEIGHT + 1):
                if (i + j) % 3 == 0:
                    color = (255, 215, 0)  # Gold
                elif (i + j) % 3 == 1:
                    color = (192, 192, 192)  # Silver
                else:
                    color = (205, 127, 50)  # Bronze
                pygame.draw.rect(self.window, color, (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        self.render_multiline_text("Leaderboard\n1. 0\n2. 0\n3. 0\nBack: Escape", (0, 0, 0), 10, 10)

    def draw_menu_background(self):
        self.draw_background()
        for i in range(15, 20):
            color = (50 + i * 10, 250 - i * 10, 0)
            pygame.draw.rect(self.window, color, (i * CELL_SIZE, 18 * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        for i in range(2, 7):
            color = (50 + i * 10, 250 - i * 10, 0)
            pygame.draw.rect(self.window, color, (4 * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        self.render_multiline_text("Snake Game\nStart: Enter/Space\nLeaderboard: Alt\nSettings: Q\nQuit: Delete/Backspace", (255, 255, 255), 10, 10)


    def draw_game_over_background(self):
        for i in range(1, GRID_WIDTH + 1):
            for j in range(1, GRID_HEIGHT + 1):
                distance = min(min(i, GRID_WIDTH - i), min(j, GRID_HEIGHT - j))
                color = 120 - distance * 10
                pygame.draw.rect(self.window, (color + random.randint(0, 5), color + random.randint(0, 5), color + random.randint(0, 5)),
                                 (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                
        self.render_multiline_text(f"Game Over\nScore: {self.score}\nHigh Score: {self.high_score}\nContinue: Space", (255, 255, 255), 10, 10)

if __name__ == "__main__":
    left = "l" if len(sys.argv) > 1 and sys.argv[1] == "l" else "r"
    game = SnakeGame(left)
    game.run()