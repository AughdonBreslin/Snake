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


def test_torch_evaluator_leaves_the_model_in_eval_mode():
    model = SnakeNet(NetConfig(channels=16, blocks=2, groups=4))
    model.train()
    evaluator = TorchEvaluator(model, torch.device("cpu"))
    evaluator.evaluate_batch(obs_batch(2))
    assert model.training is False


def test_torch_evaluator_runs_with_grad_disabled():
    # Asserting p.grad is None proves nothing: .grad is only populated by
    # .backward(). Check the actual guarantee, which is that autograd is off
    # while the model runs, so no graph is built during a tree search.
    model = SnakeNet(NetConfig(channels=16, blocks=2, groups=4))
    seen = {}

    def record(module, inputs, output):
        seen["grad_enabled"] = torch.is_grad_enabled()

    handle = model.register_forward_hook(record)
    try:
        try:
            TorchEvaluator(model, torch.device("cpu")).evaluate_batch(obs_batch(2))
        except RuntimeError:
            # The hook already recorded the grad mode seen during forward.
            # A RuntimeError here means grad tracking leaked into the
            # output tensors and .numpy() rejected them downstream; that
            # is itself evidence of the bug this test exists to catch, so
            # it must not hide the assertion below.
            pass
    finally:
        handle.remove()

    assert seen["grad_enabled"] is False


def test_torch_evaluator_output_is_float32_numpy():
    model = SnakeNet(NetConfig(channels=16, blocks=2, groups=4))
    priors, values = TorchEvaluator(model, torch.device("cpu")).evaluate_batch(obs_batch(2))
    assert isinstance(priors, np.ndarray) and priors.dtype == np.float32
    assert isinstance(values, np.ndarray) and values.dtype == np.float32
