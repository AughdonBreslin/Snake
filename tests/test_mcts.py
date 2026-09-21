import numpy as np
import pytest

from snake.config import SearchConfig
from snake.env import DOWN, OPPOSITE, RIGHT, SnakeEnv
from snake.evaluator import UniformEvaluator
from snake.mcts import Search, run_search


def make_env(size=6, seed=0):
    return SnakeEnv(size, np.random.default_rng(seed), starvation_limit=2 * size * size)


def make_search(env=None, **overrides):
    settings = {"simulations": 64, "dirichlet_epsilon": 0.0}
    settings.update(overrides)
    return Search(env or make_env(), SearchConfig(**settings), np.random.default_rng(0))


def test_visits_are_spent_on_legal_actions_only():
    env = make_env()
    search = make_search(env)
    run_search([search], UniformEvaluator(0.5), simulations=64)
    counts = search.visit_counts()
    assert counts[OPPOSITE[env.direction]] == 0
    assert counts.sum() > 0


def test_total_visits_match_the_simulation_budget():
    search = make_search()
    run_search([search], UniformEvaluator(0.5), simulations=64)
    # The first simulation expands the root and spends no child visit.
    assert search.visit_counts().sum() == 63


def test_search_never_mutates_the_root_env():
    env = make_env()
    before = (list(env.snake), env.food, env.score, env.direction)
    run_search([make_search(env)], UniformEvaluator(0.5), simulations=64)
    assert (list(env.snake), env.food, env.score, env.direction) == before


def test_a_forced_win_is_found():
    # Food is one step to the right. With a value of zero everywhere, the only
    # thing separating actions is the 1/C reward on the eating edge, so the
    # search must concentrate on RIGHT.
    env = make_env()
    env.food = (env.snake[0][0] + 1, env.snake[0][1])
    search = make_search(env, simulations=200)
    run_search([search], UniformEvaluator(0.0), simulations=200)
    counts = search.visit_counts()
    assert counts.argmax() == RIGHT


def test_death_is_avoided():
    # The head is against the right wall with a clear board otherwise. Moving
    # right is instant death and must attract the fewest visits.
    env = make_env(size=8)
    env.snake.clear()
    env.snake.extend([(8, 4), (7, 4), (6, 4)])
    env._occupied = set(env.snake)
    env.direction = RIGHT
    env.food = (2, 2)
    search = make_search(env, simulations=300)
    run_search([search], UniformEvaluator(0.5), simulations=300)
    counts = search.visit_counts()
    assert counts.argmax() != RIGHT
    assert counts[RIGHT] < counts.max()


def test_terminal_nodes_are_never_expanded():
    env = make_env(size=8)
    env.snake.clear()
    env.snake.extend([(8, 4), (7, 4), (6, 4)])
    env._occupied = set(env.snake)
    env.direction = RIGHT
    search = make_search(env, simulations=100)
    run_search([search], UniformEvaluator(0.5), simulations=100)
    dead = search.root.children[RIGHT]
    assert dead.terminal is True
    assert dead.expanded is False
    assert dead.value_sum == 0.0


def test_root_value_equals_the_leaf_value_when_nothing_is_eaten():
    # With no food reachable inside the horizon, every backup is the identity,
    # so the root's mean value is exactly the constant the evaluator returns.
    env = make_env(size=8)
    env.food = (8, 8)
    search = make_search(env, simulations=32)
    run_search([search], UniformEvaluator(0.4), simulations=32)
    assert np.isclose(search.root.value_sum / search.root.visits, 0.4, atol=1e-6)


def test_eating_adds_exactly_one_over_total_cells_on_backup():
    # Two simulations pin the reward down exactly. The first expands the root
    # and backs up the constant v. The second has every root child unvisited
    # and equally weighted, so the only separator is the edge reward and RIGHT,
    # the eating action, is selected. The root then holds v + (v + 1/C) and the
    # child holds v, because the reward belongs to the edge and is added on the
    # way up, not banked inside the child.
    #
    # A statistical version of this over hundreds of simulations does not work:
    # terminal leaves back up 0.0 rather than v, which pulls the root mean below
    # v by more than the eating rewards push it above.
    env = make_env(size=6)
    env.food = (env.snake[0][0] + 1, env.snake[0][1])
    value = float(np.float32(0.3))
    reward = 1.0 / env.total_cells

    eating = make_search(env, simulations=2)
    run_search([eating], UniformEvaluator(0.3), simulations=2)
    assert eating.visit_counts()[RIGHT] == 1
    assert np.isclose(eating.root.value_sum, 2 * value + reward, atol=1e-12)
    assert np.isclose(eating.root.children[RIGHT].value_sum, value, atol=1e-12)

    # The same position with the food out of reach takes the identical edge and
    # must back up the constant twice with nothing added.
    far = make_env(size=6)
    far.food = (far.size, far.size)
    quiet = make_search(far, simulations=2)
    run_search([quiet], UniformEvaluator(0.3), simulations=2)
    assert quiet.visit_counts()[RIGHT] == 1
    assert np.isclose(quiet.root.value_sum, 2 * value, atol=1e-12)


def test_dirichlet_noise_changes_the_root_priors():
    env = make_env()
    quiet = make_search(env, simulations=8, dirichlet_epsilon=0.0)
    noisy = Search(env, SearchConfig(simulations=8, dirichlet_epsilon=0.25),
                   np.random.default_rng(0))
    run_search([quiet], UniformEvaluator(0.5), simulations=8)
    run_search([noisy], UniformEvaluator(0.5), simulations=8)
    quiet_priors = [c.prior for c in quiet.root.children if c is not None]
    noisy_priors = [c.prior for c in noisy.root.children if c is not None]
    assert not np.allclose(quiet_priors, noisy_priors)
    assert np.isclose(sum(noisy_priors), 1.0)


def test_lockstep_driver_runs_many_games_together():
    searches = [make_search(make_env(seed=s)) for s in range(4)]
    run_search(searches, UniformEvaluator(0.5), simulations=32)
    for search in searches:
        assert search.visit_counts().sum() == 31


def test_open_loop_visits_different_food_futures():
    # Two simulations that both eat should see different replacement food, which
    # is the whole point of reseeding per simulation.
    env = make_env(size=6)
    env.food = (env.snake[0][0] + 1, env.snake[0][1])
    search = make_search(env, simulations=400)
    seen = set()
    original_clone = env.clone

    def spy(rng):
        twin = original_clone(rng)
        seen.add(twin.rng.integers(0, 2**31))
        return twin

    env.clone = spy
    run_search([search], UniformEvaluator(0.5), simulations=400)
    assert len(seen) > 100


def test_a_node_that_died_once_is_still_descended_again():
    # Terminality is a property of one determinization, not of the node. Death
    # is absorbing and survival is not, so caching the flag would record "has
    # ever died in any sampled world" and the pessimism would grow with the
    # simulation budget. Freezing a safe child by hand stands in for the
    # unlucky sample that would otherwise strand it.
    search = make_search()
    run_search([search], UniformEvaluator(0.5), simulations=1)
    frozen = search.root.children[DOWN]
    frozen.terminal = True
    run_search([search], UniformEvaluator(0.5), simulations=63)
    assert frozen.expanded is True
    assert frozen.visits > 0
    assert frozen.value_sum > 0.0
    assert search.visit_counts().sum() == 63


def test_a_short_evaluator_batch_is_rejected():
    # A silent zip truncation would leave the tail searches with their pending
    # simulation still set, and the next descend would discard that backup.
    class ShortEvaluator:
        def evaluate_batch(self, obs):
            rows = len(obs) - 1
            return (
                np.full((rows, 4), 0.25, dtype=np.float32),
                np.full(rows, 0.5, dtype=np.float32),
            )

    searches = [make_search(make_env(seed=s)) for s in range(2)]
    with pytest.raises(ValueError):
        run_search(searches, ShortEvaluator(), simulations=4)


def test_descend_refuses_to_run_with_a_simulation_pending():
    search = make_search()
    assert search.descend() is not None
    with pytest.raises(RuntimeError):
        search.descend()
