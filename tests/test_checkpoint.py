import numpy as np
import torch

from snake.config import RunConfig
from snake.train import Trainer, load_for_inference


def cfg(tmp_path):
    return RunConfig.build(
        net={"channels": 16, "blocks": 2, "groups": 4},
        search={"simulations": 4},
        train={
            "board_size": 6,
            "concurrent_games": 2,
            "train_steps_per_iteration": 2,
            "batch_size": 8,
            "replay_capacity": 200,
            "eval_games": 1,
            "keep_last": 2,
            "device": "cpu",
            "run_dir": str(tmp_path / "run"),
        },
    )


def test_save_then_load_restores_identical_outputs(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    trainer.run_iteration()
    path = trainer.save_checkpoint("test")

    probe = torch.from_numpy(np.random.default_rng(0).random((2, 11, 8, 8)).astype(np.float32))
    trainer.model.eval()
    with torch.no_grad():
        before = trainer.model(probe)

    fresh = Trainer(cfg(tmp_path))
    fresh.load_checkpoint(path)
    fresh.model.eval()
    with torch.no_grad():
        after = fresh.model(probe)

    assert torch.allclose(before[0], after[0], atol=1e-6)
    assert torch.allclose(before[1], after[1], atol=1e-6)


def test_resume_restores_the_counters_and_buffer(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    trainer.run_iteration()
    trainer.run_iteration()
    path = trainer.save_checkpoint("test")

    fresh = Trainer(cfg(tmp_path))
    fresh.load_checkpoint(path)
    assert fresh.iteration == trainer.iteration
    assert fresh.step == trainer.step
    assert len(fresh.buffer) == len(trainer.buffer)


def test_optimizer_state_survives_a_round_trip(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    trainer.run_iteration()
    path = trainer.save_checkpoint("test")
    fresh = Trainer(cfg(tmp_path))
    fresh.load_checkpoint(path)
    original = trainer.optimizer.state_dict()["state"]
    restored = fresh.optimizer.state_dict()["state"]
    assert set(original) == set(restored)
    for key in original:
        assert torch.allclose(original[key]["exp_avg"], restored[key]["exp_avg"])


def test_only_keep_last_checkpoints_are_retained(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    for index in range(5):
        trainer.run_iteration()
        trainer.save_checkpoint(f"iter{index}")
    kept = sorted((trainer.run_dir / "checkpoints").glob("iter*.pt"))
    assert len(kept) == 2


def test_load_for_inference_needs_no_trainer(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    trainer.run_iteration()
    path = trainer.save_checkpoint("test")
    model, restored_cfg = load_for_inference(path, torch.device("cpu"))
    assert restored_cfg.net.channels == 16
    assert model.training is False
