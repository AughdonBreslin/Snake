#include <SFML/Graphics.hpp>
#include <vector>
#include <random>
#include <ctime>
#include <iostream>
#include <sstream>
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
    bool settings;
    bool leaderboard;
    int score;
    int highScore;
    int frameRate;
    sf::Font font;
    sf::Text scoreText;
    sf::Text highScoreText;
    sf::Text menuText;
    sf::Text settingsText;
    sf::Text leaderboardText;
    
    void setup() {
        snake = { {3, 3}, {2, 3}, {1, 3} };
        direction = Direction::Right;
        moveQueue = std::queue<Direction>();
        gameOver = false;
        gameStarted = false;
        settings = false;
        leaderboard = false;
        score = 0;
        highScore = 0;
        spawnFood();
        readSettings();
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

    void readSettings() {
        std::ifstream file(path + "settings.txt");
        if (file.is_open()) {
            if (!(file >> frameRate)) {
                frameRate = 10;
            }
            file.close();
        } else {
            frameRate = 10;
        }
    }

    void writeSettings() {
        std::ofstream file(path + "settings.txt");
        if (file.is_open()) {
            file << frameRate;
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

    void drawBackground() {
        // Checkerboard pattern
        sf::RectangleShape cell(sf::Vector2f(CELL_SIZE, CELL_SIZE));
        for (int i = 1; i <= GRID_WIDTH; i++) {
            for (int j = 1; j <= GRID_HEIGHT; j++) {
                if ((i + j) % 2 == 0) {
                    cell.setFillColor(sf::Color(50, 50, 50));
                } else {
                    cell.setFillColor(sf::Color(40, 40, 40));
                }
                cell.setPosition(i * CELL_SIZE, j * CELL_SIZE);
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

    void drawLeaderboardBackground() {
        // Gold to silver to bronze gradient
        sf::RectangleShape cell(sf::Vector2f(CELL_SIZE, CELL_SIZE));
        for (int i = 1; i <= GRID_WIDTH; i++) {
            for (int j = 1; j <= GRID_HEIGHT; j++) {
                if ((i + j) % 3 == 0) {
                    cell.setFillColor(sf::Color(255, 215, 0));
                } else if ((i + j) % 3 == 1) {
                    cell.setFillColor(sf::Color(192, 192, 192));
                } else {
                    cell.setFillColor(sf::Color(205, 127, 50));
                }
                cell.setPosition(i * CELL_SIZE, j * CELL_SIZE);
                window.draw(cell);
            }
        }
    }

    void drawSettingsBackground() {
        // Random gray gradient
        sf::RectangleShape cell(sf::Vector2f(CELL_SIZE, CELL_SIZE));
        for (int i = 1; i <= GRID_WIDTH; i++) {
            for (int j = 1; j <= GRID_HEIGHT; j++) {
                cell.setFillColor(sf::Color(10 + rand() % 100, 10 + rand() % 100, 10 + rand() % 100));
                cell.setPosition(i * CELL_SIZE, j * CELL_SIZE);
                window.draw(cell);
            }
        }
    }

    void drawMenuBackground() {
        sf::RectangleShape cell(sf::Vector2f(CELL_SIZE, CELL_SIZE));
        for (int i = 1; i <= GRID_WIDTH; i++) {
            for (int j = 1; j <= GRID_HEIGHT; j++) {
                if ((i + j) % 2 == 0) {
                    cell.setFillColor(sf::Color(50, 50, 50));
                } else {
                    cell.setFillColor(sf::Color(40, 40, 40));
                }
                cell.setPosition(i * CELL_SIZE, j * CELL_SIZE);
                window.draw(cell);
            }
        }
        sf::RectangleShape segment(sf::Vector2f(CELL_SIZE, CELL_SIZE));
        for (int i = 15; i < 20; i++) {
            segment.setFillColor(sf::Color(50 + i * 10, 250 - i * 10, 0));
            segment.setPosition(i * CELL_SIZE, 18 * CELL_SIZE);
            window.draw(segment);
        }
        for (int i = 2; i < 7; i++) {
            segment.setFillColor(sf::Color(50 + i * 10, 250 - i * 10, 0));
            segment.setPosition(4 * CELL_SIZE, i * CELL_SIZE);
            window.draw(segment);
        }
    }

    void drawGameOverBackground() {
        // Radial heat map
        sf::RectangleShape cell(sf::Vector2f(CELL_SIZE, CELL_SIZE));
        for (int i = 1; i <= GRID_WIDTH; i++) {
            for (int j = 1; j <= GRID_HEIGHT; j++) {
                int distance = std::min(std::min(i, GRID_WIDTH - i), std::min(j, GRID_HEIGHT - j));
                int color = 120 - distance * 10;
                cell.setFillColor(sf::Color(color + rand() % 5, color + rand() % 5, color + rand() % 5));
                cell.setPosition(i * CELL_SIZE, j * CELL_SIZE);
                window.draw(cell);
            }
        }
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
    SnakeGame(std::string left) : window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Snake Game", sf::Style::None) {
        setup();
        window.setFramerateLimit(frameRate);
        if (!font.loadFromFile(path + "fira.ttf")) {
            throw std::runtime_error("Unable to load font");
        }
        if (left == "l") {
            window.setPosition(sf::Vector2i(100, 100));
        } else {
            window.setPosition(sf::Vector2i(1380, 100));
        }

        scoreText.setFont(font);
        scoreText.setCharacterSize(20);
        scoreText.setFillColor(sf::Color::White);
        scoreText.setPosition(10, -3);

        readHighScore();
        highScoreText.setFont(font);
        highScoreText.setCharacterSize(20);
        highScoreText.setFillColor(sf::Color::White);
        highScoreText.setPosition(WINDOW_WIDTH - 200, -3);

        menuText.setFont(font);
        menuText.setCharacterSize(30);
        menuText.setFillColor(sf::Color::White);
        menuText.setString(" Snake Game\n Start Game: Enter/Space\n Leaderboard: Alt\n Settings: Q");
        menuText.setPosition(WINDOW_WIDTH / 2 - menuText.getLocalBounds().width / 2,
                             WINDOW_HEIGHT / 2 - menuText.getLocalBounds().height / 2);
        
        settingsText.setFont(font);
        settingsText.setCharacterSize(30);
        settingsText.setFillColor(sf::Color::White);
        settingsText.setString(" Settings\n Difficulty: 1 2 3 4 5\n Back: Escape");
        settingsText.setPosition(WINDOW_WIDTH / 2 - settingsText.getLocalBounds().width / 2,
                                WINDOW_HEIGHT / 2 - settingsText.getLocalBounds().height / 2);

        leaderboardText.setFont(font);
        leaderboardText.setCharacterSize(30);
        leaderboardText.setFillColor(sf::Color::Black);
        leaderboardText.setString(" Leaderboard\n 1. 0\n 2. 0\n 3. 0\n Back: Escape");
        leaderboardText.setPosition(WINDOW_WIDTH / 2 - leaderboardText.getLocalBounds().width / 2,
                                   WINDOW_HEIGHT / 2 - leaderboardText.getLocalBounds().height / 2);
    }

    void run() {
        while (window.isOpen()) {
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed) {
                    window.close();
                }
                if (event.type == sf::Event::KeyPressed) {
                    if (!gameStarted) {
                        if (event.key.code == sf::Keyboard::Enter || event.key.code == sf::Keyboard::Space) {
                            gameStarted = true;
                        }
                        if (event.key.code == sf::Keyboard::LAlt || event.key.code == sf::Keyboard::RAlt) {
                            leaderboard = true;
                        }
                        if (event.key.code == sf::Keyboard::Q) {
                            settings = true;
                        }
                        if (event.key.code == sf::Keyboard::Escape && (settings || leaderboard)) {
                            settings = false;
                            leaderboard = false;
                        }
                    }

                    if (gameStarted && !gameOver) {
                        if ((event.key.code == sf::Keyboard::Up || event.key.code == sf::Keyboard::W) &&
                         ((moveQueue.empty() && direction != Direction::Down) || moveQueue.back() != Direction::Down)) {
                            moveQueue.push(Direction::Up);
                        }
                        if ((event.key.code == sf::Keyboard::Down || event.key.code == sf::Keyboard::S) &&
                         ((moveQueue.empty() && direction != Direction::Up) || moveQueue.back() != Direction::Up)) {
                            moveQueue.push(Direction::Down);
                        }
                        if ((event.key.code == sf::Keyboard::Left || event.key.code == sf::Keyboard::A) &&
                         ((moveQueue.empty() && direction != Direction::Right) || moveQueue.back() != Direction::Right)) {
                            moveQueue.push(Direction::Left);
                        }
                        if ((event.key.code == sf::Keyboard::Right || event.key.code == sf::Keyboard::D) &&
                         ((moveQueue.empty() && direction != Direction::Left) || moveQueue.back() != Direction::Left)) {
                            moveQueue.push(Direction::Right);
                        }
                    }
                    if (gameOver) {
                        if (event.key.code == sf::Keyboard::Enter || event.key.code == sf::Keyboard::Space) {
                            setup();
                        }
                    }
                }
            }

            window.clear(sf::Color::Black);
            // Before Game
            if (!gameStarted) {
                if (settings) {
                    drawSettingsBackground();
                    window.draw(settingsText);
                    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num1)) {
                        frameRate = 5;
                    }
                    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num2)) {
                        frameRate = 10;
                    }
                    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num3)) {
                        frameRate = 15;
                    }
                    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num4)) {
                        frameRate = 20;
                    }
                    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num5)) {
                        frameRate = 25;
                    }
                    window.setFramerateLimit(frameRate);
                    writeSettings();
                } else if (leaderboard) {
                    drawLeaderboardBackground();
                    window.draw(leaderboardText);
                } else {
                    drawMenuBackground();
                    window.draw(menuText);
                }
            // During Game
            } else if (gameStarted && !gameOver) {
                drawWalls();
                drawBackground();
                move();
                drawSnake();
                drawFood();
                window.draw(scoreText);
                window.draw(highScoreText);
            // After Game
            } else {
                drawGameOverBackground();
                if (score > highScore) {
                    highScore = score;
                    writeHighScore();
                    updateHighScoreText();
                }
                sf::Text gameOverText;
                gameOverText.setFont(font);
                gameOverText.setCharacterSize(30);
                gameOverText.setFillColor(sf::Color::White);
                gameOverText.setString(" Game Over\n Score: " + std::to_string(score) + 
                                       "\n High Score: " + std::to_string(highScore) + "\n Continue: Space");
                gameOverText.setPosition(WINDOW_WIDTH / 2 - gameOverText.getLocalBounds().width / 2,
                                         WINDOW_HEIGHT / 2 - gameOverText.getLocalBounds().height / 2);
                window.draw(gameOverText);
            }

            window.display();
        }
    }
};
// Add left/right boolean arg
int main(int argc, char** argv) {
    std::string left;
    std::istringstream iss;
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [left]" << std::endl;
        return 1;
    }
    if (argc > 1) {
        iss.str(argv[1]);
        if (!(iss >> left) || (left != "l" && left != "r")) {
            std::cerr << "Invalid argument: " << argv[1] << std::endl;
            return 1;
        }
        iss.clear();
    } 

    try {
        SnakeGame game(left);
        game.run();
    } catch (const std::exception& e) {
        std::cout << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}