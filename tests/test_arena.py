import numpy as np

from snake.arena import OUTCOMES, GameResult, play_games, play_games_batched, summarize
from snake.baselines import GreedyAgent, HamiltonianAgent, RandomAgent


def test_play_games_is_deterministic_for_a_seed():
    first = play_games(RandomAgent, size=6, n_games=10, seed=7)
    second = play_games(RandomAgent, size=6, n_games=10, seed=7)
    assert [r.score for r in first] == [r.score for r in second]


def test_different_seeds_give_different_games():
    first = play_games(RandomAgent, size=6, n_games=10, seed=1)
    second = play_games(RandomAgent, size=6, n_games=10, seed=2)
    assert [r.score for r in first] != [r.score for r in second]


def test_every_game_reports_a_known_outcome():
    results = play_games(GreedyAgent, size=6, n_games=20, seed=0)
    assert len(results) == 20
    assert all(r.outcome in {"wall", "self", "starvation", "solved"} for r in results)


def test_summary_fields():
    results = [
        GameResult(score=3, length=6, steps=40, outcome="wall"),
        GameResult(score=5, length=8, steps=60, outcome="self"),
    ]
    summary = summarize(results, total_cells=36)
    assert summary["games"] == 2
    assert summary["mean_score"] == 4.0
    assert summary["median_score"] == 4.0
    assert summary["max_score"] == 5
    assert summary["mean_steps"] == 50.0
    assert np.isclose(summary["mean_fill_fraction"], (6 / 36 + 8 / 36) / 2)
    assert summary["outcomes"] == {"wall": 1, "self": 1, "starvation": 0, "solved": 0}


def test_hamiltonian_summary_is_a_perfect_ceiling():
    summary = summarize(
        play_games(lambda rng: HamiltonianAgent(6, 0), size=6, n_games=5, seed=0),
        total_cells=36,
    )
    assert summary["mean_fill_fraction"] == 1.0
    assert summary["outcomes"]["solved"] == 5


def _torch_agent_setup(tmp_path):
    import torch

    from snake.config import RunConfig
    from snake.evaluator import TorchEvaluator
    from snake.model import SnakeNet
    from snake.train import NetworkAgent

    cfg = RunConfig.build(
        net={"channels": 16, "blocks": 2, "groups": 4},
        search={"simulations": 8},
        train={"board_size": 6, "device": "cpu", "run_dir": str(tmp_path)},
    )
    evaluator = TorchEvaluator(SnakeNet(cfg.net), torch.device("cpu"))
    return cfg, evaluator, (lambda rng: NetworkAgent(evaluator, cfg, rng))


def test_batched_play_reproduces_the_unbatched_games_exactly(tmp_path):
    # The batched path exists only for speed, so it has to be the same
    # experiment. Games are independent of each other and of their ordering,
    # and the network is deterministic, so batching the evaluations changes
    # when work happens and never what it computes.
    cfg, evaluator, factory = _torch_agent_setup(tmp_path)
    one_at_a_time = play_games(factory, size=6, n_games=5, seed=0)
    lockstep = play_games_batched(
        factory, size=6, n_games=5, seed=0,
        evaluator=evaluator, simulations=cfg.search.simulations,
    )
    assert lockstep == one_at_a_time


def test_batched_play_is_deterministic_for_a_seed(tmp_path):
    cfg, evaluator, factory = _torch_agent_setup(tmp_path)
    kwargs = dict(size=6, n_games=4, evaluator=evaluator,
                  simulations=cfg.search.simulations)
    assert (play_games_batched(factory, seed=3, **kwargs)
            == play_games_batched(factory, seed=3, **kwargs))


def test_batched_play_reports_the_same_outcome_vocabulary(tmp_path):
    cfg, evaluator, factory = _torch_agent_setup(tmp_path)
    results = play_games_batched(
        factory, size=6, n_games=4, seed=0,
        evaluator=evaluator, simulations=cfg.search.simulations,
    )
    assert len(results) == 4
    assert all(r.outcome in set(OUTCOMES) for r in results)


def test_play_games_reports_each_game_as_it_completes():
    # A hundred games with no output until the very end is unreadable while it
    # runs, and that matters more the longer games get.
    seen = []
    results = play_games(
        RandomAgent, size=6, n_games=4, seed=0,
        on_game=lambda completed, total, result: seen.append((completed, total, result)),
    )
    assert [c for c, _, _ in seen] == [1, 2, 3, 4]
    assert all(total == 4 for _, total, _ in seen)
    assert [r for _, _, r in seen] == results


def test_batched_play_reports_each_game_as_it_completes(tmp_path):
    cfg, evaluator, factory = _torch_agent_setup(tmp_path)
    seen = []
    results = play_games_batched(
        factory, size=6, n_games=5, seed=0,
        evaluator=evaluator, simulations=cfg.search.simulations,
        on_game=lambda completed, total, result: seen.append((completed, total, result)),
    )
    # Games end whenever they end, so the running count is what is monotonic,
    # not the game index.
    assert [c for c, _, _ in seen] == [1, 2, 3, 4, 5]
    assert all(total == 5 for _, total, _ in seen)
    assert sorted(r.score for _, _, r in seen) == sorted(r.score for r in results)


def test_progress_reporting_does_not_change_the_games(tmp_path):
    cfg, evaluator, factory = _torch_agent_setup(tmp_path)
    kwargs = dict(size=6, n_games=4, seed=0, evaluator=evaluator,
                  simulations=cfg.search.simulations)
    assert (play_games_batched(factory, on_game=lambda *a: None, **kwargs)
            == play_games_batched(factory, **kwargs))
