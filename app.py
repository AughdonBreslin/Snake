import os
import pygame

from enum import Enum
from random import randint

from graph import HamiltonianCycle

CELL_SIZE = 20
WINDOW_WIDTH, WINDOW_HEIGHT = 440, 440
GRID_WIDTH, GRID_HEIGHT = WINDOW_WIDTH // CELL_SIZE, WINDOW_HEIGHT // CELL_SIZE
CELL_COUNT = (GRID_WIDTH - 2) * (GRID_HEIGHT - 2)
PATH = "/mnt/c/Users/aughb/Personal_Projects/Snake/"

class App:
    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.NOFRAME)
        self.clock = pygame.time.Clock()

        self.fps = 60
        self.curr_screen = "home"
        self.background = Background(self.window)
        self.home = Home(self.window)
        self.game = SnakeGame(self.window)
        self.settings = Settings(self.window)
        self.leaderboard = Leaderboard(self.window)
        self.controls = False

    def run(self):
        running = True
        frame = 1
        time = 0
        while running:
            self.clock.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    match event.key:
                        case pygame.K_BACKSPACE | pygame.K_DELETE:
                            running = False
                        case pygame.K_SPACE | pygame.K_RETURN:
                            self.curr_screen = "game"
                        case pygame.K_q:
                            self.background.draw_background()
                            self.game.reset()
                            self.curr_screen = "settings"
                        case pygame.K_LALT | pygame.K_RALT:
                            self.game.reset()
                            self.curr_screen = "leaderboard"
                        case pygame.K_ESCAPE:
                            self.curr_screen = "home"
                    if self.curr_screen == "game":
                        self.game.event(event.key)
                    elif self.curr_screen == "settings":
                        self.settings.event(event.key)
                    elif self.curr_screen == "leaderboard":
                        pass
            # Stats            
            print(f"FPS: {self.clock.get_fps():.2f}, igFPS: {self.settings.fps} Move Queue: {self.game.move_queue}                                  ", end='\r')
            if time / self.fps > frame:
                match self.curr_screen:
                    case "home":
                        self.home.draw_background()
                    case "game":
                        self.game.play()
                    case "settings":
                        self.settings.draw_background()
                    case "leaderboard":
                        self.leaderboard.draw_background()
                    case "controls":
                        pass
                frame += 1
            time += self.settings.fps
            
            # Updates screen
            pygame.display.flip()

        pygame.quit()

class Background:
    def __init__(self, window):
        self.window = window
        self.title_font = pygame.font.Font(os.path.join(PATH, "cpp/data/fira.ttf"), 2*CELL_SIZE)
        self.font = pygame.font.Font(os.path.join(PATH, "cpp/data/fira.ttf"), CELL_SIZE)
    
    def draw_background(self):
        for i in range(GRID_WIDTH):
            for j in range(GRID_HEIGHT):
                if (i + j) % 2 == 0:
                    pygame.draw.rect(self.window, (40, 40, 40), (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                else:
                    pygame.draw.rect(self.window, (50, 50, 50), (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.window, (30, 30, 30), (0, 0, WINDOW_WIDTH, WINDOW_HEIGHT), CELL_SIZE)
    
    def _render_multiline_text(self, text, x, y, color=(255, 255, 255), line_spacing=5):
        lines = text.split('\n')
        for i, line in enumerate(lines):
            text_surface = self.font.render(line, True, color)
            self.window.blit(text_surface, (x, y + i * (self.font.get_height() + line_spacing)))

    def draw_text(self, text, x, y, size=CELL_SIZE, selected=None):
        if '\n' in text:
            self._render_multiline_text(text, x, y)
        elif selected:
            unselected_beginning = self.font.render(text[:selected[0]], True, (255, 255, 255))
            unselected_ending = self.font.render(text[selected[1]:], True, (255, 255, 255))
            selected_text = self.font.render(text[selected[0]:selected[1]], True, (0, 255, 0))
            self.window.blit(unselected_beginning, (x, y))
            self.window.blit(selected_text, (x + unselected_beginning.get_width(), y))
            self.window.blit(unselected_ending, (x + unselected_beginning.get_width() + selected_text.get_width(), y))
        else:
            text_surface = self.font.render(text, True, (255, 255, 255))
            self.window.blit(text_surface, (x, y))

class Home(Background):
    def __init__(self, window):
        super().__init__(window)
        self.snake_text = self.title_font.render("Snake", True, (255, 255, 255))
        self.start_text = self.font.render("Space or Enter to start game", True, (255, 255, 255))
        self.leaderboard_text = self.font.render("Alt to view leaderboards", True, (255, 255, 255))
        self.settings_text = self.font.render("Q to enter settings", True, (255, 255, 255))
        self.exit_text = self.font.render("Backspace/Delete to exit", True, (255, 255, 255))
        self.hamiltonian_cycle = HamiltonianCycle(GRID_HEIGHT - 2, GRID_WIDTH - 2, 25).cycle_positions()
        self.offset = 0
        self.snake_len = 3
        self.food = (randint(0, GRID_HEIGHT - 2 - 1), randint(0, GRID_WIDTH - 2 - 1))


    def draw_background(self):
        super().draw_background()

        snake_pos = []
        for i in range(self.snake_len):
            index = (i + self.offset) % CELL_COUNT
            pos = self.hamiltonian_cycle[index]
            if pos == self.food and i == self.snake_len - 1:
                self.snake_len += 1
                break
            snake_pos.append(pos)
            pygame.draw.rect(self.window, (0, 75 + 125*i/self.snake_len, 0), ((pos[0] + 1) * CELL_SIZE, (pos[1] + 1) * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        if self.snake_len == CELL_COUNT:
            self.snake_len = 3
            self.hamiltonian_cycle = HamiltonianCycle(GRID_HEIGHT - 2, GRID_WIDTH - 2, 25).cycle_positions()

        while self.food in snake_pos:
            self.food = (randint(0, GRID_HEIGHT - 2 - 1), randint(0, GRID_WIDTH - 2 - 1))
        pygame.draw.rect(self.window, (200, 0, 0), ((self.food[0] + 1) * CELL_SIZE, (self.food[1] + 1) * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        self.offset = (self.offset + 1) % CELL_COUNT

        self.window.blit(self.snake_text, (WINDOW_WIDTH * 0.36, WINDOW_HEIGHT * 0.1))
        self.window.blit(self.start_text, (WINDOW_WIDTH * 0.1, WINDOW_HEIGHT * 0.75))
        self.window.blit(self.leaderboard_text, (WINDOW_WIDTH * 0.1, WINDOW_HEIGHT * 0.80))
        self.window.blit(self.settings_text, (WINDOW_WIDTH * 0.1, WINDOW_HEIGHT * 0.85))
        self.window.blit(self.exit_text, (WINDOW_WIDTH * 0.1, WINDOW_HEIGHT * 0.9))

class Settings(Background):
    def __init__(self, window):
        super().__init__(window)
        self.offset = 0
        self.fps = self.read_settings()
        self.selected = [10 + self.fps * 2//5, 11 + self.fps * 2//5]
        self.settings_text = self.title_font.render("Settings", True, (255, 255, 255))
        self.difficulty_text = self.font.render("Difficulty: 1 2 3 4 5 6 7 8 9 0", True, (255, 255, 255))
        self.selected_text = self.font.render(f"{self.fps//5%10}", True, (0, 255, 0))
        self.escape_text = self.font.render("ESC to return to menu", True, (255, 255, 255))
        self.hamiltonian_cycle = HamiltonianCycle(GRID_HEIGHT - 2, GRID_WIDTH - 2, 25).cycle_positions()

    def draw_background(self):
        for i in range(8):
            progress = i / 8
            for j in range(10):
                color = ((i>>2&1) * (150 + 100 * j/10), (i>>1&1) * (150 + 100 * j/10), (i&1) * (150 + 100 * j/10))
                x = (self.hamiltonian_cycle[int(progress * CELL_COUNT + j + self.offset) % CELL_COUNT][0] + 1) * CELL_SIZE
                y = (self.hamiltonian_cycle[int(progress * CELL_COUNT + j + self.offset) % CELL_COUNT][1] + 1) * CELL_SIZE
                pygame.draw.rect(self.window, color, (x, y, CELL_SIZE, CELL_SIZE))
            x = (self.hamiltonian_cycle[int(progress * CELL_COUNT - 1 + self.offset) % CELL_COUNT][0] + 1) * CELL_SIZE
            y = (self.hamiltonian_cycle[int(progress * CELL_COUNT - 1 + self.offset) % CELL_COUNT][1] + 1) * CELL_SIZE
            color = (40, 40, 40) if (x + y) / 20 % 2 == 0 else (50, 50, 50)
            pygame.draw.rect(self.window, color, (x, y, CELL_SIZE, CELL_SIZE))
        self.offset = (self.offset + 1) % CELL_COUNT

        self.window.blit(self.settings_text, (WINDOW_WIDTH * 0.30, WINDOW_HEIGHT * 0.1))
        self.window.blit(self.difficulty_text, (WINDOW_WIDTH * 0.1, WINDOW_HEIGHT * 0.4))
        self.window.blit(self.selected_text, (WINDOW_WIDTH * 0.1 + self.difficulty_text.get_width() - self.selected_text.get_width() * (50 - self.fps)/2.5 - 12, WINDOW_HEIGHT * 0.4))

        self.window.blit(self.escape_text, (WINDOW_WIDTH * 0.1, WINDOW_HEIGHT * 0.9))

    def read_settings(self):
        try:
            with open(os.path.join(PATH, "settings.txt"), "r") as f:
                return int(f.read())
        except FileNotFoundError:
            print("Settings file not found, creating new one.")
            self.write_settings(10)
            return 10

    def write_settings(self, difficulty):
        with open(os.path.join(PATH, "settings.txt"), "w") as f:
            f.write(str(difficulty))

    def event(self, key):
        match key:
            case pygame.K_1:
                self.fps = 5
            case pygame.K_2:
                self.fps = 10
            case pygame.K_3:
                self.fps = 15
            case pygame.K_4:
                self.fps = 20
            case pygame.K_5:
                self.fps = 25
            case pygame.K_6:
                self.fps = 30
            case pygame.K_7:
                self.fps = 35
            case pygame.K_8:
                self.fps = 40
            case pygame.K_9:
                self.fps = 45
            case pygame.K_0:
                self.fps = 50

        self.selected_text = self.font.render(f"{self.fps//5%10}", True, (0, 255, 0))
        self.write_settings(self.fps)

class Leaderboard(Background):
    def __init__(self, window):
        super().__init__(window)
        self.leaderboard = self.read_leaderboard()
        self.leaderboard_text = self.title_font.render("Leaderboards", True, (255, 255, 255))
        self.first_place = self.font.render(f"1. {self.leaderboard[0]}", True, (255, 255, 255))
        self.second_place = self.font.render(f"2. {self.leaderboard[1]}", True, (255, 255, 255))
        self.third_place = self.font.render(f"3. {self.leaderboard[2]}", True, (255, 255, 255))
        self.escape_text = self.font.render("ESC to return to menu", True, (255, 255, 255))

    def read_leaderboard(self):
        try:
            with open(os.path.join(PATH, "leaderboard.txt"), "r") as f:
                lines = f.readlines()
                return [line.strip() for line in lines]
        except FileNotFoundError:
            print("Leaderboard file not found, creating new one.")
            self.write_leaderboard(["0\n", "0\n", "0\n"])
            return ["0\n", "0\n", "0\n"]
        
    def write_leaderboard(self, leaderboard):
        with open(os.path.join(PATH, "leaderboard.txt"), "w") as f:
            f.writelines(leaderboard)
    
    def draw_background(self):
        for i in range(1, GRID_WIDTH - 1):
            for j in range(1, GRID_HEIGHT - 1):
                if (i + j) % 6 < 2:
                    color = (115, 71, 28)  # Bronze
                elif (i + j) % 6 < 4:
                    color = (143, 121, 0)  # Gold
                else:
                    color = (108, 108, 108)  # Silver
                pygame.draw.rect(self.window, color, (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        self.window.blit(self.leaderboard_text, (WINDOW_WIDTH * 0.16, WINDOW_HEIGHT * 0.1))
        self.window.blit(self.first_place, (WINDOW_WIDTH * 0.1, WINDOW_HEIGHT * 0.4))
        self.window.blit(self.second_place, (WINDOW_WIDTH * 0.1, WINDOW_HEIGHT * 0.45))
        self.window.blit(self.third_place, (WINDOW_WIDTH * 0.1, WINDOW_HEIGHT * 0.5))
        self.window.blit(self.escape_text, (WINDOW_WIDTH * 0.1, WINDOW_HEIGHT * 0.9))

class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

    def __repr__(self):
        return self.name

    def __add__(self, other):
        return (self.value + other) % 4

class Action(Enum):
    TURN_LEFT = -1
    STRAIGHT = 0
    TURN_RIGHT = 1

    def __repr__(self):
        return self.name

    def __radd__(self, other):
        return other + self.value
    
RIGHT, DOWN, LEFT, UP = Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP
TURN_LEFT, STRAIGHT, TURN_RIGHT = Action.TURN_LEFT, Action.STRAIGHT, Action.TURN_RIGHT

class SnakeGame(Background):
    def __init__(self, window):
        super().__init__(window)
        self.score = 0
        self.highscore = self.read_highscore()
        self.score_prefix = self.font.render("Score: ", True, (255, 255, 255))
        self.score_text = self.font.render(str(self.score), True, (255, 255, 255))
        self.highscore_text = self.font.render(f"Highscore: {self.highscore}", True, (255, 255, 255))
        self.game_over_text = self.title_font.render("Game Over", True, (255, 255, 255))
        self.retry_text = self.font.render("SPACE or Enter to play again", True, (255, 255, 255))
        self.escape_text = self.font.render("ESC to return to menu", True, (255, 255, 255))
        self.reset()
    
    def reset(self):
        self.snake = [(3, GRID_HEIGHT // 2), (2, GRID_HEIGHT // 2), (1, GRID_HEIGHT // 2)]
        self.direction = RIGHT
        self.move_queue = []
        self.food = self.spawn_food()
        self.score = 0
        self.game_over = False
        self.score_text = self.font.render(str(self.score), True, (255, 255, 255))


    def read_highscore(self):
        try:
            with open(os.path.join(PATH, "highscore.txt"), "r") as f:
                return int(f.read())
        except FileNotFoundError:
            print("Highscore file not found, creating new one.")
            self.write_highscore(0)
            return 0
    
    def write_highscore(self, highscore):
        with open(os.path.join(PATH, "highscore.txt"), "w") as f:
            f.write(str(highscore))

    def draw_background(self):
        super().draw_background()
        for x, y in self.snake:
            pygame.draw.rect(self.window, (255 - 150*x/GRID_WIDTH, 255 - 150*y/GRID_HEIGHT, 0), (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.window, (255, 0, 0), (self.food[0] * CELL_SIZE, self.food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        self.window.blit(self.score_prefix, (WINDOW_WIDTH * 0.1, WINDOW_HEIGHT * -0.01))
        self.window.blit(self.score_text, (WINDOW_WIDTH * 0.1 + self.score_prefix.get_width(), WINDOW_HEIGHT * -0.01))
        self.window.blit(self.highscore_text, (WINDOW_WIDTH * 0.5, WINDOW_HEIGHT * -0.01))

    def draw_game_over(self):
        for i in range(1, GRID_WIDTH - 1):
            for j in range(1, GRID_HEIGHT - 1):
                distance = min(min(i, GRID_WIDTH - i - 1), min(j, GRID_HEIGHT - j - 1))
                color = max(0,  120 - distance * 10)
                pygame.draw.rect(self.window, (color + randint(0, 5), color + randint(0, 5), color + randint(0, 5)),
                                 (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                
        self.window.blit(self.game_over_text, (WINDOW_WIDTH * 0.25, WINDOW_HEIGHT * 0.1))
        self.window.blit(self.score_prefix, (WINDOW_WIDTH * 0.1, WINDOW_HEIGHT * 0.4))
        self.window.blit(self.score_text, (WINDOW_WIDTH * 0.1 + self.score_prefix.get_width(), WINDOW_HEIGHT * 0.4))
        self.window.blit(self.highscore_text, (WINDOW_WIDTH * 0.1, WINDOW_HEIGHT * 0.45))
        self.window.blit(self.retry_text, (WINDOW_WIDTH * 0.1, WINDOW_HEIGHT * 0.85))
        self.window.blit(self.escape_text, (WINDOW_WIDTH * 0.1, WINDOW_HEIGHT * 0.9))

    def spawn_food(self):
        while True:
            food = (randint(1, GRID_WIDTH - 2), randint(1, GRID_HEIGHT - 2))
            if food not in self.snake:
                return food
    
    def move(self):
        if self.move_queue:
            self.direction = self.move_queue.pop(0)
        x, y = self.snake[0]
        if self.direction == RIGHT:
            x += 1
        elif self.direction == DOWN:
            y += 1
        elif self.direction == LEFT:
            x -= 1
        elif self.direction == UP:
            y -= 1
        if (x, y) in self.snake or x == 0 or x == GRID_WIDTH - 1 or y == 0 or y == GRID_HEIGHT - 1:
            return False
        self.snake.insert(0, (x, y))
        if (x, y) == self.food:
            self.food = self.spawn_food()
            self.score += 1
            self.score_text = self.font.render(str(self.score), True, (255, 255, 255))
            self.highscore = max(self.highscore, self.score)
            self.highscore_text = self.font.render(f"Highscore: {self.highscore}", True, (255, 255, 255))
        else:
            self.snake.pop()
        return True
    
    def event(self, key):
        if self.game_over:
            if key == pygame.K_SPACE or key == pygame.K_RETURN:
                self.reset()
        elif (key == pygame.K_RIGHT or key == pygame.K_d) and ((not self.move_queue and self.direction != LEFT) or (self.move_queue and self.move_queue[-1] != LEFT)):
            self.move_queue.append(RIGHT)
        elif (key == pygame.K_DOWN or key == pygame.K_s) and ((not self.move_queue and self.direction != UP) or (self.move_queue and self.move_queue[-1] != UP)):
            self.move_queue.append(DOWN)
        elif (key == pygame.K_LEFT or key == pygame.K_a) and ((not self.move_queue and self.direction != RIGHT) or (self.move_queue and self.move_queue[-1] != RIGHT)):
            self.move_queue.append(LEFT)
        elif (key == pygame.K_UP or key == pygame.K_w) and ((not self.move_queue and self.direction != DOWN) or (self.move_queue and self.move_queue[-1] != DOWN)):
            self.move_queue.append(UP)
    
    def play(self):
        self.game_over = not self.move()
        if self.game_over:
            self.write_highscore(self.highscore)
            self.draw_game_over()
        else:
            self.draw_background()

if __name__ == "__main__":
    app = App()
    app.run()