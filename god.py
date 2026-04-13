
# Write a CNN that takes in a 4 channel image and outputs a Direction enum from app.py


import copy
import math
import numpy as np
import pygame
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from app import Direction, SnakeGame, WINDOW_WIDTH, WINDOW_HEIGHT

class CNN(nn.Module):
    def __init__(self, grid_size=10):
        super(CNN, self).__init__()

        # Feature extraction layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        
        # Policy head with action probabilities for left, right, up, down
        self.policy_conv = nn.Conv2d(128, 4, kernel_size=1)
        self.policy_fc = nn.Linear(4 * grid_size * grid_size, 4)

        # Value head for game state evaluation
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(grid_size * grid_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Feature extraction
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.relu(self.batch_norm3(self.conv3(x)))

        # Policy head (which move to make)
        policy = self.policy_conv(x) # (batch_size, 4, H, W)
        policy = policy.view(policy.size(0), -1)
        policy = F.relu(self.policy_fc(policy))
        policy = F.softmax(policy, dim=1)

        # Value head (how good the current state is)
        value = self.value_conv(x) # (batch_size, 1, H, W)
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0 # Expected reward
        self.prior = 0 # Prior probability of this node being the best move

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_valid_inputs())

    def best_child(self, c_puct=1.0):
        return max(self.children.values(), key=lambda child: child.uct_score(c_puct))
    
    def uct_score(self, c_puct):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)


class MCTS:
    def __init__(self, model, simulations=50, c_puct=1.0):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct

    def run(self, root_state):
        root = Node(root_state)

        # Get policy and value from CNN
        state_tensor = torch.tensor(root_state.get_state(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad(): # Disable gradient calculation since we don't need it for MCTS inference
            policy, value = self.model(state_tensor)
            policy = policy.squeeze().numpy()

        valid_actions = root_state.get_valid_inputs()
        for action in valid_actions:
            root.children[action] = Node(root_state.make_move(action), parent=root)
            root.children[action].prior = policy[action]

        for _ in range(self.simulations):
            self.simulate(root)

        return self.get_policy(root)
    
    def simulate(self, node):
        # Selection
        while node.is_fully_expanded():
            node = node.best_child(self.c_puct)

        # Expansion
        if node.visits > 0:
            valid_actions = node.state.get_valid_inputs()
            for action in valid_actions:
                if action not in node.children:
                    node.children[action] = Node(node.state.get_next_state(action), parent=node)
                    break
            node = node.children[action]

        # Simulation
        state_tensor = torch.tensor(node.state.to_tensor(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            policy, value = self.model(state_tensor)
            policy = policy.squeeze().numpy()
            value = value.item()

        # Backpropagation
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
            # no need to alternate value since it's a single player game
    
    def get_policy(self, root):
        visits = np.array([child.visits for child in root.children.values()])
        return visits / visits.sum()

class SnakeTrainer:
    def __init__(self, model, games=1000, simulations=50, batch_size=64, lr=0.001):
        self.model = model
        self.games = games
        self.mcts = MCTS(model, simulations)
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.training_data = []

    def self_play(self):
        for _ in range(self.games):
            game_states = []
            policies = []
            values = []
            env = SnakeGame(pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.NOFRAME))
            done = False

            while not done:
                state_tensor = torch.tensor(env.get_state().to_tensor(), dtype=torch.float32).unsqueeze(0)
                policy = self.mcts.run(env)
                action = np.random.choice(len(policy), p=policy)

                game_states.append(state_tensor)
                policies.append(policy)

                done, reward = env.step(action)
                values.append(reward)

            # Compute final rewards for value function
            final_result = sum(values)
            self.training_data.extend([(state, policy, final_result) for state, policy in zip(game_states, policies)])

    def train(self, epochs=10):
        self.self_play()
        for epoch in range(epochs):
            random.shuffle(self.training_data)

            batches = [self.training_data[i:i + self.batch_size] for i in range(0, len(self.training_data), self.batch_size)]
            for batch in batches:
                states, policies, values = zip(*batch)
                states = torch.cat(states)
                policies = torch.tensor(policies, dtype=torch.float32)
                values = torch.tensor(values, dtype=torch.float32)

                self.optimizer.zero_grad()
                predicted_policies, predicted_values = self.model(states)

                # Policy loss: Cross-entropy between predicted and actual policies
                policy_loss = F.cross_entropy(predicted_policies, policies)

                # Value loss: Mean squared error between predicted and actual values
                value_loss = F.mse_loss(predicted_values.squeeze(), values)

                # Total loss
                loss = policy_loss + value_loss
                loss.backward()
                self.optimizer.step()

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def save_model(self, path="snake_model.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="snake_model.pth"):
        self.model.load_state_dict(torch.load(path))

if __name__ == "__main__":
    model = CNN()
    trainer = SnakeTrainer(model)
    trainer.train(epochs=10)
    trainer.save_model()

