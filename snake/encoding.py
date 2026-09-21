"""Observation planes and dihedral symmetry. No torch, no pygame.

Planes are indexed [channel, y, x]. Every plane is board size invariant in
meaning, which is what lets one set of network weights serve every board in the
curriculum.

  0      walls, the one cell border
  1      body excluding the head
  2      head
  3      food
  4      tail cell
  5      body age: steps until that cell vacates, divided by current length
  6..9   heading, one hot as four constant planes
  10     fill fraction, constant, length divided by total playable cells
"""

from __future__ import annotations

import numpy as np

N_PLANES = 11
HEADING_PLANE_0 = 6


def encode(size, snake, food, direction):
    dim = size + 2
    obs = np.zeros((N_PLANES, dim, dim), dtype=np.float32)

    obs[0, 0, :] = 1.0
    obs[0, dim - 1, :] = 1.0
    obs[0, :, 0] = 1.0
    obs[0, :, dim - 1] = 1.0

    length = len(snake)
    for i, (x, y) in enumerate(snake):
        if i > 0:
            obs[1, y, x] = 1.0
        # The cell at index i vacates in (length - i) steps.
        obs[5, y, x] = (length - i) / length

    hx, hy = snake[0]
    obs[2, hy, hx] = 1.0

    if food is not None:
        fx, fy = food
        obs[3, fy, fx] = 1.0

    tx, ty = snake[-1]
    obs[4, ty, tx] = 1.0

    obs[HEADING_PLANE_0 + direction, :, :] = 1.0
    obs[10, :, :] = length / (size * size)

    return obs


# The eight elements of the dihedral group of the square, as (rotations, flip).
SYMMETRIES = tuple((k, flip) for flip in (False, True) for k in range(4))

# (dx, dy) per action. Duplicated from snake.env deliberately: encoding.py must
# not import env.py, since env.py imports this module.
_DELTAS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def action_permutation(k, flip):
    """Where each action ends up after k rotations and an optional x flip.

    Derived from the geometry rather than tabulated. np.rot90 on an array
    indexed [.., y, x] sends a displacement (dx, dy) to (dy, -dx); np.flip on
    the x axis sends it to (-dx, dy).
    """
    perm = [0] * 4
    for action, (dx, dy) in enumerate(_DELTAS):
        for _ in range(k % 4):
            dx, dy = dy, -dx
        if flip:
            dx = -dx
        perm[action] = _DELTAS.index((dx, dy))
    return perm


def transform_observation(obs, k, flip):
    out = np.rot90(obs, k=k, axes=(1, 2))
    if flip:
        out = np.flip(out, axis=2)
    # Always copy. np.rot90 and np.flip return views, and ascontiguousarray is a
    # no-op on an already contiguous array, so the identity symmetry would
    # otherwise hand back a view aliasing the caller's observation.
    out = np.array(out, copy=True, order="C")

    # The heading planes are spatially constant, so rotating them is a no op.
    # Their channel identity is what has to move.
    perm = action_permutation(k, flip)
    heading = out[HEADING_PLANE_0 : HEADING_PLANE_0 + 4].copy()
    for action in range(4):
        out[HEADING_PLANE_0 + perm[action]] = heading[action]
    return out


def transform_policy(pi, k, flip):
    perm = action_permutation(k, flip)
    out = np.zeros_like(pi)
    for action in range(4):
        out[perm[action]] = pi[action]
    return out
