import json
import pathlib

from snake.cli import main


def test_train_writes_a_config_and_a_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    main([
        "train",
        "--board-size", "6",
        "--iterations", "1",
        "--simulations", "4",
        "--concurrent-games", "2",
        "--train-steps", "2",
        "--batch-size", "8",
        "--channels", "16",
        "--blocks", "2",
        "--groups", "4",
        "--eval-games", "1",
        "--device", "cpu",
        "--run-dir", str(run_dir),
    ])
    config = json.loads((run_dir / "config.json").read_text())
    assert config["train"]["board_size"] == 6
    assert list((run_dir / "checkpoints").glob("*.pt"))


REPO = pathlib.Path(__file__).resolve().parent.parent


def test_god_py_is_gone():
    assert not (REPO / "god.py").exists()


def test_requirements_stay_free_of_torch():
    text = (REPO / "requirements.txt").read_text()
    assert "torch" not in text
    assert "numpy" in text and "pygame" in text
