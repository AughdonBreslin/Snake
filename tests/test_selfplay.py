import numpy as np

from snake.config import RunConfig, SearchConfig
from snake.encoding import SYMMETRIES, transform_observation
from snake.env import RIGHT, SnakeEnv
from snake.evaluator import UniformEvaluator
from snake.selfplay import Position, ReplayBuffer, play_batch, select_move, value_targets


def cfg(**train):
    settings = {"board_size": 6, "concurrent_games": 2}
    settings.update(train)
    return RunConfig.build(train=settings, search={"simulations": 8, "dirichlet_epsilon": 0.0})


def test_value_targets_are_return_to_go_over_total_cells():
    # Lengths 3, 3, 4, 5 with a final length of 5 on a 36 cell board.
    assert value_targets([3, 3, 4, 5], final_length=5, total_cells=36) == [
        2 / 36,
        2 / 36,
        1 / 36,
        0.0,
    ]


def test_value_targets_are_zero_when_nothing_more_is_eaten():
    assert value_targets([7, 7, 7], final_length=7, total_cells=36) == [0.0, 0.0, 0.0]


def test_value_targets_stay_in_the_unit_interval():
    z = value_targets([3], final_length=36, total_cells=36)
    assert 0.0 <= z[0] <= 1.0


def test_select_move_is_greedy_at_zero_temperature():
    pi = np.array([0.1, 0.6, 0.0, 0.3])
    settings = SearchConfig(tau_threshold=0, tau_final=0.0)
    picks = {select_move(pi, 5, settings, np.random.default_rng(s)) for s in range(20)}
    assert picks == {1}


def test_select_move_explores_at_temperature_one():
    pi = np.array([0.25, 0.25, 0.25, 0.25])
    settings = SearchConfig(tau_threshold=100)
    picks = {select_move(pi, 0, settings, np.random.default_rng(s)) for s in range(40)}
    assert len(picks) > 1


def test_select_move_never_picks_a_zero_probability_action():
    pi = np.array([0.0, 0.5, 0.5, 0.0])
    settings = SearchConfig(tau_threshold=100)
    picks = {select_move(pi, 0, settings, np.random.default_rng(s)) for s in range(60)}
    assert picks == {1, 2}


def test_play_batch_returns_finished_games():
    positions, results = play_batch(cfg(), UniformEvaluator(0.5), np.random.default_rng(0))
    assert len(results) == 2
    assert all(r.outcome in {"wall", "self", "starvation", "solved"} for r in results)
    assert len(positions) == sum(r.steps for r in results)


def test_play_batch_positions_carry_normalized_policies():
    positions, _ = play_batch(cfg(), UniformEvaluator(0.5), np.random.default_rng(1))
    for position in positions[:50]:
        assert np.isclose(position.pi.sum(), 1.0)
        assert 0.0 <= position.z <= 1.0


def test_play_batch_value_targets_are_return_to_go_of_their_own_game():
    positions, results = play_batch(cfg(), UniformEvaluator(0.5), np.random.default_rng(7))
    start = 0
    for result in results:
        game = positions[start : start + result.steps]
        start += result.steps
        # The last recorded position of a game has nothing left to gain.
        assert game[-1].z == 0.0
        # Every earlier target is the remaining growth over the board area.
        total_cells = game[0].size * game[0].size
        for position in game:
            expected = (result.length - len(position.snake)) / total_cells
            assert position.z == expected
    assert start == len(positions)


def test_play_batch_is_reproducible():
    first, _ = play_batch(cfg(), UniformEvaluator(0.5), np.random.default_rng(3))
    second, _ = play_batch(cfg(), UniformEvaluator(0.5), np.random.default_rng(3))
    assert [p.z for p in first] == [p.z for p in second]


def test_buffer_evicts_oldest_first():
    buffer = ReplayBuffer(capacity=10)
    buffer.extend([_position(i) for i in range(25)])
    assert len(buffer) == 10


def test_buffer_keeps_the_newest_entries_after_eviction():
    buffer = ReplayBuffer(capacity=10)
    buffer.extend([_position(i) for i in range(25)])
    assert sorted(round(p.z * 100) for p in buffer.items()) == list(range(15, 25))


def test_buffer_eviction_stays_oldest_first_after_a_partial_load():
    buffer = ReplayBuffer(capacity=10)
    buffer.load([_position(i) for i in range(4)])
    buffer.extend([_position(i) for i in range(4, 16)])
    assert sorted(round(p.z * 100) for p in buffer.items()) == list(range(6, 16))


def test_buffer_sample_shapes():
    buffer = ReplayBuffer(capacity=100)
    buffer.extend([_position(i) for i in range(50)])
    obs, pi, z = buffer.sample(8, np.random.default_rng(0))
    assert obs.shape == (8, 11, 8, 8)
    assert pi.shape == (8, 4)
    assert z.shape == (8,)
    assert obs.dtype == np.float32
    assert np.allclose(pi.sum(axis=1), 1.0)


def test_buffer_sampling_reaches_the_whole_buffer():
    buffer = ReplayBuffer(capacity=50)
    buffer.extend([_position(i) for i in range(50)])
    _, _, z = buffer.sample(400, np.random.default_rng(0))
    drawn = {round(float(value) * 100) for value in z}
    assert min(drawn) < 5
    assert max(drawn) > 44


def test_buffer_sampling_applies_symmetries():
    # One position sampled many times must not always come back identical, or
    # the eightfold augmentation is not happening.
    buffer = ReplayBuffer(capacity=10)
    buffer.extend([_position(0)])
    seen = set()
    for seed in range(50):
        obs, _, _ = buffer.sample(1, np.random.default_rng(seed))
        seen.add(obs.tobytes())
    assert len(seen) > 1


def test_buffer_reencodes_the_observation_the_env_would_have_produced():
    env = SnakeEnv(6, np.random.default_rng(0))
    buffer = ReplayBuffer(capacity=10)
    buffer.extend(
        [
            Position(
                snake=tuple(env.snake),
                food=env.food,
                direction=env.direction,
                size=env.size,
                pi=np.array([0.4, 0.3, 0.0, 0.3]),
                z=0.5,
            )
        ]
    )
    expected = {
        transform_observation(env.observation(), k, flip).tobytes()
        for k, flip in SYMMETRIES
    }
    seen = set()
    for seed in range(80):
        obs, _, _ = buffer.sample(1, np.random.default_rng(seed))
        seen.add(obs[0].tobytes())
    assert seen <= expected
    assert seen == expected


def _position(index):
    return Position(
        snake=((3, 3), (2, 3), (1, 3)),
        food=(6, 6),
        direction=RIGHT,
        size=6,
        pi=np.array([0.4, 0.3, 0.0, 0.3]),
        z=index / 100,
    )
