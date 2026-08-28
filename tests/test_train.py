import numpy as np
import torch

from snake.config import RunConfig
from snake.train import Trainer, losses


def cfg(tmp_path, **train):
    settings = {
        "board_size": 6,
        "concurrent_games": 2,
        "train_steps_per_iteration": 3,
        "batch_size": 8,
        "replay_capacity": 500,
        "eval_games": 2,
        "device": "cpu",
        "run_dir": str(tmp_path / "run"),
    }
    settings.update(train)
    return RunConfig.build(
        net={"channels": 16, "blocks": 2, "groups": 4},
        search={"simulations": 8},
        train=settings,
    )


def test_policy_loss_is_zero_when_the_prediction_is_perfect():
    target = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    logits = torch.tensor([[-50.0, 50.0, -50.0, -50.0]])
    value = torch.tensor([0.3])
    out = losses(logits, value, target, torch.tensor([0.3]))
    assert out["policy"].item() < 1e-4
    assert out["value"].item() < 1e-8


def test_policy_loss_does_not_softmax_twice():
    # Feeding an already normalized distribution as logits must NOT give zero
    # loss. If it does, a softmax is being applied to a softmax somewhere.
    target = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    out = losses(target.clone(), torch.tensor([0.5]), target, torch.tensor([0.5]))
    assert out["policy"].item() > 0.5


def test_entropy_is_reported_and_maximal_for_a_uniform_policy():
    logits = torch.zeros(1, 4)
    out = losses(logits, torch.tensor([0.5]), torch.full((1, 4), 0.25), torch.tensor([0.5]))
    assert np.isclose(out["entropy"].item(), np.log(4), atol=1e-5)


def test_value_mae_is_reported():
    out = losses(
        torch.zeros(2, 4),
        torch.tensor([0.2, 0.8]),
        torch.full((2, 4), 0.25),
        torch.tensor([0.4, 0.4]),
    )
    assert np.isclose(out["value_mae"].item(), 0.3, atol=1e-6)


def test_a_fixed_batch_can_be_overfit(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    rng = np.random.default_rng(0)
    obs = rng.random((8, 11, 8, 8)).astype(np.float32)
    pi = np.full((8, 4), 0.25, dtype=np.float32)
    pi[:, 1] = 0.7
    pi[:, [0, 2, 3]] = 0.1
    z = rng.random(8).astype(np.float32)
    batch = (obs, pi, z)

    first = trainer.train_step(batch)["total"]
    for _ in range(200):
        last = trainer.train_step(batch)["total"]
    assert last < first * 0.5


def test_run_iteration_fills_the_buffer_and_reports_metrics(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    metrics = trainer.run_iteration()
    assert len(trainer.buffer) > 0
    for key in ("policy", "value", "total", "entropy", "value_mae", "mean_score", "games_per_second"):
        assert key in metrics


def test_evaluate_returns_an_arena_summary(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    summary = trainer.evaluate()
    assert summary["games"] == 2
    assert set(summary["outcomes"]) == {"wall", "self", "starvation", "solved"}
