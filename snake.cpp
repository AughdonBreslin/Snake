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
    int offset;
    sf::Font font;
    sf::Text scoreText;
    sf::Text highScoreText;
    sf::Text menuText;
    sf::Text settingsText;
    sf::Text leaderboardText;
    
    void setup() {
        snake = { { 3,  3}, { 2,  3}, {1,  3} };
        direction = Direction::Right;
        moveQueue = std::queue<Direction>();
        gameOver = false;
        gameStarted = false;
        settings = false;
        leaderboard = false;
        score =  0;
        highScore =  0;
        offset =  0;
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
                highScore =  0;
            }
            file.close();
        } else {
            highScore =  0;
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
                if ((i + j) %  2 ==  0) {
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
        wall.setPosition( 0, WINDOW_HEIGHT - CELL_SIZE);
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
                if ((i + j) %  3 ==  0) {
                    cell.setFillColor(sf::Color(255, 215,  0));
                } else if ((i + j) %  3 == 1) {
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
        std::vector<std::pair<int, int>> hamiltonianCycle = {
            { 7,  9}, { 7, 10}, { 7, 11}, { 7, 12}, { 8, 12},
            { 8, 11}, { 8, 10}, { 8,  9}, { 8,  8}, { 9,  8},
            { 9,  9}, { 9, 10}, {10, 10}, {11, 10}, {11, 11},
            {11, 12}, {11, 13}, {11, 14}, {10, 14}, {10, 13},
            {10, 12}, {10, 11}, { 9, 11}, { 9, 12}, { 9, 13},
            { 8, 13}, { 8, 14}, { 9, 14}, { 9, 15}, {10, 15},
            {11, 15}, {12, 15}, {12, 14}, {13, 14}, {14, 14},
            {15, 14}, {15, 13}, {14, 13}, {14, 12}, {14, 11},
            {13, 11}, {13, 12}, {13, 13}, {12, 13}, {12, 12},
            {12, 11}, {12, 10}, {13, 10}, {14, 10}, {15, 10},
            {15, 11}, {15, 12}, {16, 12}, {16, 11}, {16, 10},
            {16,  9}, {15,  9}, {15,  8}, {15,  7}, {14,  7},
            {14,  8}, {14,  9}, {13,  9}, {13,  8}, {13,  7},
            {13,  6}, {12,  6}, {12,  7}, {11,  7}, {11,  8},
            {12,  8}, {12,  9}, {11,  9}, {10,  9}, {10,  8},
            {10,  7}, { 9,  7}, { 8,  7}, { 7,  7}, { 6,  7},
            { 6,  6}, { 6,  5}, { 7,  5}, { 7,  6}, { 8,  6},
            { 9,  6}, {10,  6}, {11,  6}, {11,  5}, {10,  5},
            {10,  4}, { 9,  4}, { 9,  5}, { 8,  5}, { 8,  4},
            { 8,  3}, { 7,  3}, { 7,  4}, { 6,  4}, { 6,  3},
            { 5,  3}, { 4,  3}, { 4,  2}, { 3,  2}, { 2,  2},
            { 2,  1}, { 2,  0}, { 3,  0}, { 3,  1}, { 4,  1},
            { 4,  0}, { 5,  0}, { 6,  0}, { 6,  1}, { 5,  1},
            { 5,  2}, { 6,  2}, { 7,  2}, { 8,  2}, { 8,  1},
            { 7,  1}, { 7,  0}, { 8,  0}, { 9,  0}, { 9,  1},
            {10,  1}, {10,  0}, {11,  0}, {12,  0}, {13,  0},
            {13,  1}, {13,  2}, {13,  3}, {12,  3}, {12,  2},
            {12,  1}, {11,  1}, {11,  2}, {10,  2}, { 9,  2},
            { 9,  3}, {10,  3}, {11,  3}, {11,  4}, {12,  4},
            {12,  5}, {13,  5}, {13,  4}, {14,  4}, {14,  3},
            {14,  2}, {14,  1}, {14,  0}, {15,  0}, {15,  1},
            {15,  2}, {15,  3}, {15,  4}, {16,  4}, {16,  3},
            {17,  3}, {18,  3}, {18,  2}, {17,  2}, {16,  2},
            {16,  1}, {16,  0}, {17,  0}, {17,  1}, {18,  1},
            {18,  0}, {19,  0}, {19,  1}, {19,  2}, {19,  3},
            {19,  4}, {18,  4}, {17,  4}, {17,  5}, {16,  5},
            {15,  5}, {14,  5}, {14,  6}, {15,  6}, {16,  6},
            {17,  6}, {18,  6}, {18,  5}, {19,  5}, {19,  6},
            {19,  7}, {18,  7}, {17,  7}, {16,  7}, {16,  8},
            {17,  8}, {18,  8}, {19,  8}, {19,  9}, {18,  9},
            {17,  9}, {17, 10}, {18, 10}, {19, 10}, {19, 11},
            {19, 12}, {19, 13}, {19, 14}, {19, 15}, {19, 16},
            {18, 16}, {18, 15}, {17, 15}, {17, 14}, {18, 14},
            {18, 13}, {18, 12}, {18, 11}, {17, 11}, {17, 12},
            {17, 13}, {16, 13}, {16, 14}, {16, 15}, {15, 15},
            {15, 16}, {16, 16}, {17, 16}, {17, 17}, {17, 18},
            {18, 18}, {18, 17}, {19, 17}, {19, 18}, {19, 19},
            {18, 19}, {17, 19}, {16, 19}, {16, 18}, {16, 17},
            {15, 17}, {15, 18}, {15, 19}, {14, 19}, {14, 18},
            {13, 18}, {13, 19}, {12, 19}, {12, 18}, {11, 18},
            {11, 19}, {10, 19}, {10, 18}, {10, 17}, {11, 17},
            {12, 17}, {13, 17}, {14, 17}, {14, 16}, {14, 15},
            {13, 15}, {13, 16}, {12, 16}, {11, 16}, {10, 16},
            { 9, 16}, { 8, 16}, { 8, 15}, { 7, 15}, { 7, 14},
            { 7, 13}, { 6, 13}, { 5, 13}, { 5, 12}, { 6, 12},
            { 6, 11}, { 5, 11}, { 4, 11}, { 4, 12}, { 4, 13},
            { 3, 13}, { 3, 12}, { 3, 11}, { 3, 10}, { 4, 10},
            { 5, 10}, { 6, 10}, { 6,  9}, { 5,  9}, { 4,  9},
            { 3,  9}, { 2,  9}, { 1,  9}, { 1, 10}, { 2, 10},
            { 2, 11}, { 1, 11}, { 1, 12}, { 2, 12}, { 2, 13},
            { 2, 14}, { 2, 15}, { 1, 15}, { 1, 16}, { 1, 17},
            { 1, 18}, { 2, 18}, { 3, 18}, { 3, 17}, { 2, 17},
            { 2, 16}, { 3, 16}, { 3, 15}, { 3, 14}, { 4, 14},
            { 4, 15}, { 5, 15}, { 5, 14}, { 6, 14}, { 6, 15},
            { 6, 16}, { 7, 16}, { 7, 17}, { 8, 17}, { 9, 17},
            { 9, 18}, { 9, 19}, { 8, 19}, { 8, 18}, { 7, 18},
            { 7, 19}, { 6, 19}, { 6, 18}, { 6, 17}, { 5, 17},
            { 5, 16}, { 4, 16}, { 4, 17}, { 4, 18}, { 5, 18},
            { 5, 19}, { 4, 19}, { 3, 19}, { 2, 19}, { 1, 19},
            { 0, 19}, { 0, 18}, { 0, 17}, { 0, 16}, { 0, 15},
            { 0, 14}, { 1, 14}, { 1, 13}, { 0, 13}, { 0, 12},
            { 0, 11}, { 0, 10}, { 0,  9}, { 0,  8}, { 0,  7},
            { 0,  6}, { 1,  6}, { 1,  7}, { 1,  8}, { 2,  8},
            { 2,  7}, { 2,  6}, { 2,  5}, { 2,  4}, { 1,  4},
            { 1,  5}, { 0,  5}, { 0,  4}, { 0,  3}, { 0,  2},
            { 0,  1}, { 0,  0}, { 1,  0}, { 1,  1}, { 1,  2},
            { 1,  3}, { 2,  3}, { 3,  3}, { 3,  4}, { 3,  5},
            { 4,  5}, { 4,  4}, { 5,  4}, { 5,  5}, { 5,  6},
            { 5,  7}, { 4,  7}, { 4,  6}, { 3,  6}, { 3,  7},
            { 3,  8}, { 4,  8}, { 5,  8}, { 6,  8}, { 7,  8}
            };
        sf::RectangleShape cell(sf::Vector2f(CELL_SIZE, CELL_SIZE));
        for (int i =  0; i < 400; i++) {
            if ((i + offset)%400 > 0 && (i + offset)%400 < 10) {
                cell.setFillColor(sf::Color(150 + 10*((i + offset)%400), 0, 0));
                cell.setPosition((hamiltonianCycle[i%400].first + 1) * CELL_SIZE, 
                    (hamiltonianCycle[i%400].second + 1) * CELL_SIZE + 1);

            } else if ((i + offset)%400 > 50 && (i + offset)%400 < 60) {
                cell.setFillColor(sf::Color(0, 150 + 10*((i + offset)%400-50), 0));
                cell.setPosition((hamiltonianCycle[i%400].first + 1) * CELL_SIZE, 
                    (hamiltonianCycle[i%400].second + 1) * CELL_SIZE + 1);

            } else if ((i + offset)%400 > 100 && (i + offset)%400 < 110) {
                cell.setFillColor(sf::Color(0, 0, 150 + 10*((i + offset)%400-100)));
                cell.setPosition((hamiltonianCycle[i%400].first + 1) * CELL_SIZE, 
                    (hamiltonianCycle[i%400].second + 1) * CELL_SIZE + 1);

            } else if ((i + offset)%400 > 150 && (i + offset)%400 < 160) {
                cell.setFillColor(sf::Color(150 + 10*((i + offset)%400-150), 150 + 10*((i + offset)%400-150), 0));
                cell.setPosition((hamiltonianCycle[i%400].first + 1) * CELL_SIZE, 
                    (hamiltonianCycle[i%400].second + 1) * CELL_SIZE + 1);

            } else if ((i + offset)%400 > 200 && (i + offset)%400 < 210) {
                cell.setFillColor(sf::Color(150 + 10*((i + offset)%400-200), 0, 150 + 10*((i + offset)%400-200)));
                cell.setPosition((hamiltonianCycle[i%400].first + 1) * CELL_SIZE, 
                    (hamiltonianCycle[i%400].second + 1) * CELL_SIZE + 1);

            } else if ((i + offset)%400 > 250 && (i + offset)%400 < 260) {
                cell.setFillColor(sf::Color(0, 150 + 10*((i + offset)%400-250), 150 + 10*((i + offset)%400-250)));
                cell.setPosition((hamiltonianCycle[i%400].first + 1) * CELL_SIZE, 
                    (hamiltonianCycle[i%400].second + 1) * CELL_SIZE + 1);

            } else if ((i + offset)%400 > 300 && (i + offset)%400 < 310) {
                cell.setFillColor(sf::Color(150 + 10*((i + offset)%400-300), 150 + 10*((i + offset)%400-300), 150 + 10*((i + offset)%400-300)));
                cell.setPosition((hamiltonianCycle[i%400].first + 1) * CELL_SIZE, 
                    (hamiltonianCycle[i%400].second + 1) * CELL_SIZE + 1);

            } else if ((i + offset)%400 > 350 && (i + offset)%400 < 360) {
                cell.setFillColor(sf::Color(150 - 10*((i + offset)%400-350), 150 - 10*((i + offset)%400-350), 150 - 10*((i + offset)%400-350)));
                cell.setPosition((hamiltonianCycle[i%400].first + 1) * CELL_SIZE, 
                    (hamiltonianCycle[i%400].second + 1) * CELL_SIZE + 1);
            } else {
                cell.setFillColor(sf::Color::Black);
                cell.setPosition(0, 0);
            }
            window.draw(cell);
        }
        offset = (offset + 1) % 400;
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
        leaderboardText.setString(" Leaderboard\n 1. 0\n  2. 0\n 3. 0\n Back: Escape");
        leaderboardText.setPosition(WINDOW_WIDTH / 2 - leaderboardText.getLocalBounds().width /  2,
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
                    } else if (gameStarted && !gameOver) {
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
                    } else if (gameOver) {
                        if (event.key.code == sf::Keyboard::Enter || event.key.code == sf::Keyboard::Space || event.key.code == sf::Keyboard::Escape) {
                            setup();
                        }
                    }
                    if (event.key.code == sf::Keyboard::Delete || event.key.code == sf::Keyboard::BackSpace) {
                        window.close();
                    }
                }
            }

            window.clear(sf::Color::Black);
            // Before Game
            if (!gameStarted) {
                if (settings) {
                    drawBackground();
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