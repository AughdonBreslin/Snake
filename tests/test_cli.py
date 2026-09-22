import json
import pathlib

import torch

from snake.cli import main

TINY_NET = [
    "--board-size", "6",
    "--simulations", "4",
    "--concurrent-games", "2",
    "--train-steps", "2",
    "--batch-size", "8",
    "--channels", "16",
    "--blocks", "2",
    "--groups", "4",
    "--eval-games", "1",
    "--device", "cpu",
]


def test_train_writes_a_config_and_a_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    main([
        "train", *TINY_NET,
        "--iterations", "2",
        "--eval-every", "1",
        "--checkpoint-every", "1",
        "--run-dir", str(run_dir),
    ])
    config = json.loads((run_dir / "config.json").read_text())
    assert config["train"]["board_size"] == 6

    checkpoints = run_dir / "checkpoints"
    # best.pt only appears if the eval branch ran and the best-tracking
    # comparison fired; a periodic iter000001.pt only appears if the
    # checkpoint_every branch fired mid loop, not just the unconditional
    # save after the final iteration (which would only produce iter000002.pt).
    assert (checkpoints / "best.pt").exists()
    assert (checkpoints / "iter000001.pt").exists()


def test_resume_does_not_reset_the_best_eval_score(tmp_path):
    run_dir = tmp_path / "run"
    main([
        "train", *TINY_NET,
        "--iterations", "1",
        "--eval-every", "1",
        "--checkpoint-every", "1",
        "--run-dir", str(run_dir),
    ])
    original = torch.load(
        run_dir / "checkpoints" / "iter000001.pt", weights_only=False
    )
    original_best = original["best_eval_score"]
    # Any real evaluated game score beats the -1.0 the trainer starts at, so
    # the eval branch on iteration 1 must have already raised it above that.
    assert original_best > -1.0

    resumed_dir = tmp_path / "resumed"
    main([
        "train", *TINY_NET,
        "--iterations", "0",
        "--run-dir", str(resumed_dir),
        "--resume", str(run_dir / "checkpoints" / "iter000001.pt"),
    ])
    # Zero further iterations run, so the only checkpoint written is the
    # unconditional save right after resuming. If resume silently reset the
    # best score tracker to -1.0 instead of restoring it, this would come
    # back as -1.0 rather than the value carried over from the original run.
    resumed = torch.load(
        resumed_dir / "checkpoints" / "iter000001.pt", weights_only=False
    )
    assert resumed["best_eval_score"] == original_best


REPO = pathlib.Path(__file__).resolve().parent.parent


def test_god_py_is_gone():
    assert not (REPO / "god.py").exists()


def test_requirements_stay_free_of_torch():
    text = (REPO / "requirements.txt").read_text()
    assert "torch" not in text
    assert "numpy" in text and "pygame" in text


def _parse_train(extra):
    import argparse
    from snake import cli

    captured = {}
    real = cli._train
    cli._train = lambda args: captured.setdefault("args", args)
    try:
        main(["train", *extra])
    finally:
        cli._train = real
    return captured["args"]


def test_discount_flag_defaults_to_undiscounted():
    assert _parse_train([]).discount == 1.0


def test_discount_flag_reaches_the_search_config(tmp_path):
    run_dir = tmp_path / "run"
    main([
        "train", "--board-size", "6", "--iterations", "0",
        "--channels", "16", "--blocks", "2", "--groups", "4",
        "--device", "cpu", "--run-dir", str(run_dir), "--discount", "0.97",
    ])
    config = json.loads((run_dir / "config.json").read_text())
    assert config["search"]["discount"] == 0.97


def test_resume_and_init_weights_are_mutually_exclusive(tmp_path, capsys):
    # Asserting only SystemExit would pass for the wrong reason whenever either
    # flag is unrecognised, since argparse exits on unknown arguments too. Check
    # that the refusal is specifically about the combination.
    import pytest

    with pytest.raises(SystemExit):
        main(["train", "--resume", "a.pt", "--init-weights", "b.pt",
              "--run-dir", str(tmp_path / "run")])
    assert "not allowed with" in capsys.readouterr().err
