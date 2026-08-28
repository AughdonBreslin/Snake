import numpy as np
import torch

from snake.config import NetConfig
from snake.encoding import encode
from snake.env import RIGHT
from snake.evaluator import TorchEvaluator, UniformEvaluator
from snake.model import SnakeNet


def obs_batch(n=5, size=6):
    return np.stack(
        [encode(size, [(3, 2), (2, 2), (1, 2)], (size, size), RIGHT) for _ in range(n)]
    )


def test_uniform_evaluator_shapes_and_values():
    priors, values = UniformEvaluator(value=0.25).evaluate_batch(obs_batch(3))
    assert priors.shape == (3, 4)
    assert np.allclose(priors, 0.25)
    assert np.allclose(values, 0.25)


def test_torch_evaluator_returns_a_distribution():
    model = SnakeNet(NetConfig(channels=16, blocks=2, groups=4))
    priors, values = TorchEvaluator(model, torch.device("cpu")).evaluate_batch(obs_batch(7))
    assert priors.shape == (7, 4)
    assert np.allclose(priors.sum(axis=1), 1.0, atol=1e-5)
    assert values.shape == (7,)
    assert ((values >= 0) & (values <= 1)).all()


def test_torch_evaluator_leaves_the_model_in_eval_mode_and_takes_no_gradients():
    model = SnakeNet(NetConfig(channels=16, blocks=2, groups=4))
    model.train()
    evaluator = TorchEvaluator(model, torch.device("cpu"))
    evaluator.evaluate_batch(obs_batch(2))
    assert model.training is False
    assert all(p.grad is None for p in model.parameters())


def test_torch_evaluator_output_is_float32_numpy():
    model = SnakeNet(NetConfig(channels=16, blocks=2, groups=4))
    priors, values = TorchEvaluator(model, torch.device("cpu")).evaluate_batch(obs_batch(2))
    assert isinstance(priors, np.ndarray) and priors.dtype == np.float32
    assert isinstance(values, np.ndarray) and values.dtype == np.float32
