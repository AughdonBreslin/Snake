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
