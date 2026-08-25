import numpy as np
import torch

from snake.config import NetConfig
from snake.encoding import encode
from snake.env import RIGHT
from snake.model import SnakeNet


def net(**kwargs):
    return SnakeNet(NetConfig(channels=16, blocks=2, groups=4, **kwargs))


def batch(size, n=3):
    obs = np.stack(
        [encode(size, [(3, 2), (2, 2), (1, 2)], (size, size), RIGHT) for _ in range(n)]
    )
    return torch.from_numpy(obs)


def test_output_shapes():
    logits, value = net()(batch(6))
    assert logits.shape == (3, 4)
    assert value.shape == (3,)


def test_value_is_in_the_unit_interval():
    _, value = net()(batch(6))
    assert torch.all(value >= 0) and torch.all(value <= 1)


def test_policy_head_emits_logits_not_probabilities():
    # Logits must be free to be negative and must not already sum to one, or the
    # training loss would softmax twice.
    logits, _ = net()(batch(6, n=64))
    assert not torch.allclose(logits.exp().sum(dim=1), torch.ones(64))


def test_one_set_of_weights_runs_on_any_board_size():
    model = net()
    for size in (6, 10, 20):
        logits, value = model(batch(size))
        assert logits.shape == (3, 4)
        assert value.shape == (3,)


def test_there_is_no_batchnorm_anywhere():
    # BatchNorm plus batch of one inference is the classic AlphaZero footgun.
    for module in net().modules():
        assert not isinstance(module, torch.nn.modules.batchnorm._BatchNorm)


def test_batch_of_one_matches_the_same_row_in_a_batch():
    model = net().eval()
    many = batch(6, n=4)
    with torch.no_grad():
        logits_many, value_many = model(many)
        logits_one, value_one = model(many[:1])
    assert torch.allclose(logits_many[0], logits_one[0], atol=1e-5)
    assert torch.allclose(value_many[0], value_one[0], atol=1e-5)


def test_the_policy_head_reads_the_head_cell():
    # Moving only the head changes the policy. If it does not, the head cell
    # gather is broken and the head is being ignored.
    model = net().eval()
    a = torch.from_numpy(encode(6, [(3, 2), (2, 2), (1, 2)], (6, 6), RIGHT))[None]
    b = torch.from_numpy(encode(6, [(3, 5), (2, 5), (1, 5)], (6, 6), RIGHT))[None]
    with torch.no_grad():
        assert not torch.allclose(model(a)[0], model(b)[0], atol=1e-4)


def test_gradients_reach_every_parameter():
    model = net()
    logits, value = model(batch(6))
    (logits.sum() + value.sum()).backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []
