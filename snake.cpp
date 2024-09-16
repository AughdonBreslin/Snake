#include <SFML/Graphics.hpp>
#include <vector>
#include <random>
#include <ctime>
#include <iostream>
#include <fstream>
#include <queue>
#include <string>

const int WINDOW_WIDTH = 440;
const int WINDOW_HEIGHT = 440;
const int CELL_SIZE = 20;
const int GRID_WIDTH = 20;
const int GRID_HEIGHT = 20;
const std::string path = "/mnt/c/Users/aughb/Personal_Projects/Snake/";

enum class Direction { Up, Down, Left, Right };

class SnakeGame {
private:
    sf::RenderWindow window;
    std::vector<sf::Vector2i> snake;
    sf::Vector2i food;
    Direction direction;
    std::queue<Direction> moveQueue;
    bool gameOver;
    bool gameStarted;
    int score;
    int highScore;
    sf::Font font;
    sf::Text scoreText;
    sf::Text highScoreText;
    sf::Text instructionsText;

    void setup() {
        snake = { {3, 3}, {2, 3}, {1, 3} };
        direction = Direction::Right;
        moveQueue = std::queue<Direction>();
        gameOver = false;
        gameStarted = false;
        score = 0;
        highScore = 0;
        spawnFood();
        updateScoreText();
        readHighScore();
        updateHighScoreText();
    }

    void spawnFood() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> disX(1, GRID_WIDTH);
        std::uniform_int_distribution<> disY(1, GRID_HEIGHT);

        do {
            food.x = disX(gen);
            food.y = disY(gen);
        } while (std::find(snake.begin(), snake.end(), food) != snake.end());
    }

    void updateScoreText() {
        scoreText.setString("Score: " + std::to_string(score));
    }

    void readHighScore() {
        std::ifstream file(path + "highscore.txt");
        if (file.is_open()) {
            if (!(file >> highScore)) {
                highScore = 0;
            }
            file.close();
        } else {
            highScore = 0;
        }
    }

    void writeHighScore() {
        std::ofstream file(path + "highscore.txt");
        if (file.is_open()) {
            file << highScore;
            file.close();
        }
    }

    void updateHighScoreText() {
        highScoreText.setString("High Score: " + std::to_string(highScore));
    }

    void move() {
        if (!moveQueue.empty()) {
            direction = moveQueue.front();
            moveQueue.pop();
        }

        sf::Vector2i newHead = snake.front();
        switch (direction) {
            case Direction::Up: newHead.y--; break;
            case Direction::Down: newHead.y++; break;
            case Direction::Left: newHead.x--; break;
            case Direction::Right: newHead.x++; break;
        }

        if (newHead.x < 1 || newHead.x > GRID_WIDTH || newHead.y < 1 || newHead.y > GRID_HEIGHT ||
            std::find(snake.begin() + 1, snake.end(), newHead) != snake.end()) {
            gameOver = true;
            return;
        }

        snake.insert(snake.begin(), newHead);

        if (newHead == food) {
            score++;
            updateScoreText();
            spawnFood();
        } else {
            snake.pop_back();
        }
    }

    void drawGrid() {
        sf::RectangleShape cell(sf::Vector2f(CELL_SIZE, CELL_SIZE));
        cell.setFillColor(sf::Color::Black);
        cell.setOutlineColor(sf::Color(0, 100, 0));
        cell.setOutlineThickness(1);

        for (int x = 1; x <= GRID_WIDTH; x++) {
            for (int y = 1; y <= GRID_HEIGHT; y++) {
                cell.setPosition(x * CELL_SIZE, y * CELL_SIZE);
                window.draw(cell);
            }
        }
    }

    void drawWalls() {
        sf::RectangleShape wall(sf::Vector2f(WINDOW_WIDTH, CELL_SIZE));
        wall.setFillColor(sf::Color(100, 100, 100));

        // Top wall
        wall.setPosition(0, 0);
        window.draw(wall);

        // Bottom wall
        wall.setPosition(0, WINDOW_HEIGHT - CELL_SIZE);
        window.draw(wall);

        // Left wall
        wall.setSize(sf::Vector2f(CELL_SIZE, WINDOW_HEIGHT));
        wall.setPosition(0, 0);
        window.draw(wall);

        // Right wall
        wall.setPosition(WINDOW_WIDTH - CELL_SIZE, 0);
        window.draw(wall);
    }

    void drawSnake() {
        sf::RectangleShape segment(sf::Vector2f(CELL_SIZE, CELL_SIZE));

        for (const auto& pos : snake) {
            segment.setFillColor(sf::Color(50 + pos.x * 10, 250 - pos.y * 10, 0));
            segment.setPosition(pos.x * CELL_SIZE, pos.y * CELL_SIZE);
            window.draw(segment);
        }
    }

    void drawFood() {
        sf::RectangleShape foodShape(sf::Vector2f(CELL_SIZE, CELL_SIZE));
        foodShape.setFillColor(sf::Color::Red);
        foodShape.setPosition(food.x * CELL_SIZE, food.y * CELL_SIZE);
        window.draw(foodShape);
    }

public:
    SnakeGame() : window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Snake Game") {
        window.setPosition(sf::Vector2i(1380, 100));
        window.setFramerateLimit(10);
        if (!font.loadFromFile(path + "fira.ttf")) {
            throw std::runtime_error("Unable to load font");
        }
        scoreText.setFont(font);
        scoreText.setCharacterSize(20);
        scoreText.setFillColor(sf::Color::White);
        scoreText.setPosition(10, 5);

        highScoreText.setFont(font);
        highScoreText.setCharacterSize(20);
        highScoreText.setFillColor(sf::Color::White);
        highScoreText.setPosition(WINDOW_WIDTH - 170, 5);

        instructionsText.setFont(font);
        instructionsText.setCharacterSize(20);
        instructionsText.setFillColor(sf::Color::White);
        readHighScore();
        instructionsText.setString("Snake Game!\nHigh Score: " +std::to_string(highScore) + "\nPress 'S' to start");
        instructionsText.setPosition(WINDOW_WIDTH / 2 - instructionsText.getLocalBounds().width / 2,
                                     WINDOW_HEIGHT / 2 - instructionsText.getLocalBounds().height / 2);

        setup();
    }

    void run() {
        while (window.isOpen()) {
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed) {
                    window.close();
                }
                if (event.type == sf::Event::KeyPressed) {
                    if (event.key.code == sf::Keyboard::S && !gameStarted) {
                        gameStarted = true;
                    }
                    if (event.key.code == sf::Keyboard::R) {
                        setup();
                        gameStarted = true;
                    }
                    if (gameStarted && !gameOver) {
                        if (event.key.code == sf::Keyboard::Up && ((moveQueue.empty() && direction != Direction::Down) || moveQueue.back() != Direction::Down)) {
                            moveQueue.push(Direction::Up);
                        }
                        if (event.key.code == sf::Keyboard::Down && ((moveQueue.empty() && direction != Direction::Up) || moveQueue.back() != Direction::Up)) {
                            moveQueue.push(Direction::Down);
                        }
                        if (event.key.code == sf::Keyboard::Left && ((moveQueue.empty() && direction != Direction::Right) || moveQueue.back() != Direction::Right)) {
                            moveQueue.push(Direction::Left);
                        }
                        if (event.key.code == sf::Keyboard::Right && ((moveQueue.empty() && direction != Direction::Left) || moveQueue.back() != Direction::Left)) {
                            moveQueue.push(Direction::Right);
                        }
                    }
                }
            }

            window.clear(sf::Color::Black);

            drawWalls();
            drawGrid();

            if (gameStarted && !gameOver) {
                move();
                drawSnake();
                drawFood();
                window.draw(scoreText);
                window.draw(highScoreText);
            } else if (!gameStarted) {
                window.draw(instructionsText);
            } else {
                if (score > highScore) {
                    highScore = score;
                    writeHighScore();
                    updateHighScoreText();
                }
                sf::Text gameOverText;
                gameOverText.setFont(font);
                gameOverText.setCharacterSize(30);
                gameOverText.setFillColor(sf::Color::White);
                gameOverText.setString("Game Over!\nScore: " + std::to_string(score) + 
                                       "\nHigh Score: " + std::to_string(highScore) + "\nPress 'R' to restart");
                gameOverText.setPosition(WINDOW_WIDTH / 2 - gameOverText.getLocalBounds().width / 2,
                                         WINDOW_HEIGHT / 2 - gameOverText.getLocalBounds().height / 2);
                window.draw(gameOverText);
            }

            window.display();
        }
    }
};

int main() {
    try {
        SnakeGame game;
        game.run();
    } catch (const std::exception& e) {
        std::cout << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}