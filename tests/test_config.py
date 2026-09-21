import numpy as np
import pytest

from snake.config import RunConfig, seed_everything


def test_defaults_match_the_spec():
    cfg = RunConfig()
    assert cfg.net.channels == 64
    assert cfg.net.blocks == 6
    assert cfg.search.simulations == 100
    assert cfg.search.c_puct == 1.5
    assert cfg.search.dirichlet_alpha == 0.8
    assert cfg.search.dirichlet_epsilon == 0.25
    assert cfg.search.tau_threshold == 40
    assert cfg.search.tau_final == 0.2
    assert cfg.train.board_size == 6
    assert cfg.train.batch_size == 512
    assert cfg.train.replay_capacity == 200_000
    assert cfg.train.learning_rate == 2e-3
    assert cfg.train.weight_decay == 1e-4
    assert cfg.train.concurrent_games == 32
    assert cfg.train.eval_games == 100


def test_channels_must_divide_into_groups():
    with pytest.raises(ValueError):
        RunConfig.build(net={"channels": 65, "groups": 8})


def test_groups_must_be_positive():
    with pytest.raises(ValueError):
        RunConfig.build(net={"channels": 64, "groups": 0})


def test_simulations_below_one_are_rejected():
    with pytest.raises(ValueError):
        RunConfig.build(search={"simulations": 0})


def test_json_round_trip():
    cfg = RunConfig.build(train={"board_size": 10, "seed": 3}, search={"simulations": 25})
    restored = RunConfig.from_json(cfg.to_json())
    assert restored == cfg
    assert restored.train.board_size == 10
    assert restored.search.simulations == 25


def test_seed_everything_makes_torch_reproducible():
    import torch

    seed_everything(5)
    first = torch.randn(4)
    seed_everything(5)
    assert torch.equal(first, torch.randn(4))


def test_seed_everything_makes_numpy_legacy_reproducible():
    seed_everything(5)
    first = np.random.rand(4)
    seed_everything(5)
    assert np.array_equal(first, np.random.rand(4))
