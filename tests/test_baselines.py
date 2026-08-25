import numpy as np

from snake.baselines import GreedyAgent, RandomAgent, survivable
from snake.env import DOWN, RIGHT, SnakeEnv


def run(agent, size=6, seed=0, limit=2000):
    env = SnakeEnv(size, np.random.default_rng(seed), starvation_limit=2 * size * size)
    for _ in range(limit):
        if env.game_over:
            break
        env.step(agent.act(env))
    return env


def test_survivable_flags_a_wall():
    env = SnakeEnv(5, np.random.default_rng(0))
    env.snake.clear()
    env.snake.append((5, 3))
    env._occupied = {(5, 3)}
    assert survivable(env, RIGHT) is False


def test_random_agent_only_returns_legal_actions():
    rng = np.random.default_rng(0)
    agent = RandomAgent(rng)
    env = SnakeEnv(6, np.random.default_rng(1), starvation_limit=72)
    for _ in range(300):
        if env.game_over:
            break
        action = agent.act(env)
        assert env.legal_actions()[action]
        env.step(action)


def test_greedy_beats_random_over_many_games():
    greedy = sum(run(GreedyAgent(np.random.default_rng(s)), seed=s).score for s in range(30))
    random_ = sum(run(RandomAgent(np.random.default_rng(s)), seed=s).score for s in range(30))
    assert greedy > random_


def test_greedy_avoids_an_immediately_fatal_move_when_it_can():
    # RIGHT is the naively closest move to the food, but it collides with the
    # snake's own body. Greedy must turn instead of taking the shortest path
    # into itself.
    env = SnakeEnv(6, np.random.default_rng(0))
    env.snake.clear()
    env.snake.extend([(3, 3), (3, 2), (4, 2), (4, 3), (4, 4)])
    env._occupied = set(env.snake)
    env.direction = DOWN
    env.food = (6, 3)
    action = GreedyAgent(np.random.default_rng(0)).act(env)
    assert survivable(env, action)
