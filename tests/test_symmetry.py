import numpy as np

from snake.encoding import (
    SYMMETRIES,
    action_permutation,
    encode,
    transform_observation,
    transform_policy,
)
from snake.env import DELTAS, DOWN, LEFT, RIGHT, UP


def sample():
    return encode(size=6, snake=[(4, 3), (3, 3), (2, 3)], food=(6, 6), direction=RIGHT)


def test_there_are_eight_symmetries():
    assert len(SYMMETRIES) == 8
    assert len(set(SYMMETRIES)) == 8


def test_identity_changes_nothing():
    obs = sample()
    assert np.array_equal(transform_observation(obs, 0, False), obs)
    assert action_permutation(0, False) == [0, 1, 2, 3]


def test_one_rotation_maps_right_to_up():
    # np.rot90 on a [.., y, x] array sends displacement (dx, dy) to (dy, -dx).
    assert action_permutation(1, False) == [UP, RIGHT, DOWN, LEFT]


def test_flip_swaps_left_and_right_only():
    assert action_permutation(0, True) == [LEFT, DOWN, RIGHT, UP]


def test_permutation_is_a_bijection():
    for k, flip in SYMMETRIES:
        assert sorted(action_permutation(k, flip)) == [0, 1, 2, 3]


def test_observation_round_trip():
    obs = sample()
    for k in range(4):
        there = transform_observation(obs, k, False)
        back = transform_observation(there, (4 - k) % 4, False)
        assert np.array_equal(back, obs)


def test_policy_round_trip():
    pi = np.array([0.1, 0.2, 0.3, 0.4])
    for k in range(4):
        there = transform_policy(pi, k, False)
        back = transform_policy(there, (4 - k) % 4, False)
        assert np.allclose(back, pi)


def test_policy_moves_mass_the_same_way_as_the_board():
    pi = np.zeros(4)
    pi[RIGHT] = 1.0
    for k, flip in SYMMETRIES:
        moved = transform_policy(pi, k, flip)
        assert moved[action_permutation(k, flip)[RIGHT]] == 1.0


def test_heading_plane_agrees_with_the_transformed_geometry():
    # The single most load bearing symmetry test. In every transformed frame,
    # the heading channel must match the actual head minus neck displacement.
    # A reversed permutation fails here and nowhere else.
    obs = sample()
    for k, flip in SYMMETRIES:
        out = transform_observation(obs, k, flip)
        head = _one_hot_position(out[2])
        neck = _highest_body_age_position(out)
        delta = (head[1] - neck[1], head[0] - neck[0])  # (dx, dy)
        encoded = [p for p in range(4) if out[6 + p].any()]
        assert len(encoded) == 1
        assert DELTAS[encoded[0]] == delta, f"k={k} flip={flip}"


def test_transform_preserves_plane_sums():
    obs = sample()
    for k, flip in SYMMETRIES:
        out = transform_observation(obs, k, flip)
        for plane in (0, 1, 2, 3, 4):
            assert np.isclose(out[plane].sum(), obs[plane].sum())


def _one_hot_position(plane):
    ys, xs = np.nonzero(plane)
    assert len(ys) == 1
    return int(ys[0]), int(xs[0])


def _highest_body_age_position(obs):
    # Among body cells excluding the head, the neck has the largest age value.
    body = obs[5] * (obs[1] > 0)
    ys, xs = np.nonzero(body == body.max())
    return int(ys[0]), int(xs[0])


def test_no_symmetry_aliases_its_input():
    # np.rot90 and np.flip return views. The identity symmetry in particular
    # would hand back a view of the caller's array, so an in place operation on
    # a transformed observation would write through into the original.
    obs = sample()
    for k, flip in SYMMETRIES:
        assert not np.shares_memory(transform_observation(obs, k, flip), obs), f"k={k} flip={flip}"
