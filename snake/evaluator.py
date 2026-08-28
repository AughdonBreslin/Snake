"""The seam between search and the network.

MCTS never calls a model directly. Swapping single process inference for
multi process workers later is a new class here and no change to mcts.py.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch


class Evaluator(Protocol):
    def evaluate_batch(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """obs is [B, C, H, W]. Returns priors [B, 4] summing to one per row and
        values [B] in [0, 1]."""


class UniformEvaluator:
    """Uniform priors and a constant value. Lets search be tested without a net."""

    def __init__(self, value=0.0):
        self.value = value

    def evaluate_batch(self, obs):
        n = len(obs)
        priors = np.full((n, 4), 0.25, dtype=np.float32)
        values = np.full(n, self.value, dtype=np.float32)
        return priors, values


class TorchEvaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def evaluate_batch(self, obs):
        tensor = torch.from_numpy(np.ascontiguousarray(obs)).to(
            self.device, non_blocking=True
        )
        logits, values = self.model(tensor)
        priors = torch.softmax(logits, dim=1)
        return (
            priors.float().cpu().numpy(),
            values.float().cpu().numpy(),
        )
