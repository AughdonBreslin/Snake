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


def test_resume_continues_the_rng_stream_rather_than_restarting_it(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    trainer.run_iteration()
    trainer.run_iteration()
    path = trainer.save_checkpoint("test")

    # Draw the values the original trainer's generator would produce next, right
    # after the point that was checkpointed.
    expected_continuation = trainer.rng.random(8)

    fresh = Trainer(cfg(tmp_path))
    fresh.load_checkpoint(path)
    actual_continuation = fresh.rng.random(8)

    # A fresh Trainer built from the same seed starts its rng at position zero,
    # the same position the original trainer's rng was in before it ever drew
    # anything. If load_checkpoint failed to restore rng_state, fresh.rng would
    # still be sitting at that zero position, so its first 8 draws would equal
    # the *original* trainer's first-ever 8 draws, not the draws that follow two
    # run_iteration() calls worth of consumption. Guard against that false-pass
    # shape explicitly:
    restarted_stream = np.random.default_rng(cfg(tmp_path).train.seed).random(8)
    assert not np.allclose(expected_continuation, restarted_stream)

    assert np.allclose(actual_continuation, expected_continuation)


def test_load_for_inference_needs_no_trainer(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    trainer.run_iteration()
    path = trainer.save_checkpoint("test")
    model, restored_cfg = load_for_inference(path, torch.device("cpu"))
    assert restored_cfg.net.channels == 16
    assert model.training is False


def _scores_on_disk(trainer):
    """Read each best slot's own eval score back off disk, in rank order."""
    import torch as _t

    out = []
    for rank in range(1, 6):
        name = "best.pt" if rank == 1 else f"best{rank}.pt"
        p = trainer.run_dir / "checkpoints" / name
        if not p.exists():
            break
        out.append(_t.load(p, map_location="cpu", weights_only=False)["eval_score"])
    return out


def test_the_first_eval_becomes_the_top_slot(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    assert trainer.record_best(10.0) is True
    assert _scores_on_disk(trainer) == [10.0]


def test_a_better_score_demotes_the_previous_best(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    trainer.record_best(10.0)
    assert trainer.record_best(20.0) is True
    # best.pt is the new peak, the old peak moved down a slot rather than
    # being overwritten and lost.
    assert _scores_on_disk(trainer) == [20.0, 10.0]


def test_the_top_three_are_kept_in_descending_order(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    for score in (10.0, 30.0, 20.0):
        trainer.record_best(score)
    assert _scores_on_disk(trainer) == [30.0, 20.0, 10.0]


def test_a_score_below_the_whole_list_is_rejected(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    for score in (30.0, 20.0, 10.0):
        trainer.record_best(score)
    assert trainer.record_best(5.0) is False
    assert _scores_on_disk(trainer) == [30.0, 20.0, 10.0]


def test_a_middling_score_inserts_and_evicts_the_worst(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    for score in (30.0, 20.0, 10.0):
        trainer.record_best(score)
    assert trainer.record_best(25.0) is True
    assert _scores_on_disk(trainer) == [30.0, 25.0, 20.0]


def test_the_ranking_survives_a_resume(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    for score in (30.0, 20.0, 10.0):
        trainer.record_best(score)
    path = trainer.save_checkpoint("roll")

    fresh = Trainer(cfg(tmp_path))
    fresh.load_checkpoint(path)
    assert fresh.best_scores == [30.0, 20.0, 10.0]
    # and it still ranks correctly afterwards
    assert fresh.record_best(5.0) is False
    assert fresh.record_best(25.0) is True


def test_rolling_pruning_never_touches_a_best_slot(tmp_path):
    trainer = Trainer(cfg(tmp_path))
    for score in (30.0, 20.0, 10.0):
        trainer.record_best(score)
    for index in range(5):
        trainer.save_checkpoint(f"iter{index:06d}")
    kept = {p.name for p in (trainer.run_dir / "checkpoints").glob("*.pt")}
    assert {"best.pt", "best2.pt", "best3.pt"} <= kept


def test_a_checkpoint_written_before_top_n_still_loads(tmp_path):
    # Checkpoints from the single slot era carry best_eval_score but no
    # best_scores. That older format is unambiguously representable in the new
    # one, so it migrates rather than failing.
    import torch as _t

    trainer = Trainer(cfg(tmp_path))
    trainer.best_eval_score = 12.5
    path = trainer.save_checkpoint("legacy")
    payload = _t.load(path, map_location="cpu", weights_only=False)
    del payload["best_scores"]
    _t.save(payload, path)

    fresh = Trainer(cfg(tmp_path))
    fresh.load_checkpoint(path)
    assert fresh.best_scores == [12.5]
