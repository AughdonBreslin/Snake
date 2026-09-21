import numpy as np

from snake.arena import GameResult, play_games, summarize
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
