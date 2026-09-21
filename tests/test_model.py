import numpy as np
import torch
import torch.nn.functional as F

from snake.config import NetConfig
from snake.encoding import N_PLANES, encode
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
    # Raw logits take both signs. A trailing softmax would make every value
    # positive, and a trailing log_softmax would make every value negative.
    # This needs a batch with real spread to be reliable: batch(6, n=64) repeats
    # one encoded position 64 times, so it collapses to 4 raw values, and an
    # untrained network can by chance put all 4 on the same side of zero
    # (measured empirically at roughly a 1 in 10 chance across fresh random
    # inits). Random noise across many rows gives each output unit many
    # independent draws, which is not flaky in the same way (0 spurious
    # failures across 3000 trials versus roughly 300 across the same number
    # of trials with the repeated-position batch).
    wide_logits, _ = net()(torch.randn(64, N_PLANES, 8, 8))
    assert (wide_logits < 0).any()
    assert (wide_logits > 0).any()


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
    # The policy output depends on where the snake sits on the board. This
    # does not isolate the head gather specifically: both encodings move the
    # whole snake, so the pooled term can account for the difference on its
    # own. The gather itself is tested directly in
    # test_head_gather_extracts_the_head_cell_feature_vector.
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


def test_head_gather_extracts_the_head_cell_feature_vector():
    # The policy head is supposed to gather its per-head-cell term by
    # multiplying the feature map by plane 2 (the head one hot) and summing
    # over space. A version of this test that only recomputes that formula
    # itself and checks it against an index into the recomputed feature map
    # is a tautology: it would hold even if forward() gathered from the wrong
    # plane entirely, because it never touches forward()'s own head_mask. So
    # instead this drives the actual model.forward() and compares its real
    # output against a manual recomputation that explicitly uses plane 2 for
    # the gather, reusing the model's own submodules and weights throughout.
    # If forward() gathered from a different plane, the two would diverge.
    model = net().eval()
    obs = batch(6, n=3)
    with torch.no_grad():
        actual_logits, _ = model(obs)

        x = F.relu(model.stem_norm(model.stem_conv(obs)))
        for block in model.blocks:
            x = block(x)
        p = F.relu(model.policy_norm(model.policy_conv(x)))
        head_mask = obs[:, 2:3]
        at_head = (p * head_mask).sum(dim=(2, 3))
        pooled = p.mean(dim=(2, 3))
        expected_logits = model.policy_fc(torch.cat([at_head, pooled], dim=1))

    assert torch.allclose(actual_logits, expected_logits, atol=1e-6)
