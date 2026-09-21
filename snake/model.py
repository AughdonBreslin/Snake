"""Fully convolutional policy and value network.

Nothing may be sized from the board dimensions, because one set of weights has
to serve every board in the curriculum. GroupNorm rather than BatchNorm, so a
batch of one during search behaves exactly like the same row inside a batch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from snake.encoding import N_PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x):
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.relu(x + h)


class SnakeNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels, groups = cfg.channels, cfg.groups

        self.stem_conv = nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False)
        self.stem_norm = nn.GroupNorm(groups, channels)
        self.blocks = nn.ModuleList(
            ResidualBlock(channels, groups) for _ in range(cfg.blocks)
        )

        self.policy_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.policy_norm = nn.GroupNorm(groups, channels)
        self.policy_fc = nn.Sequential(
            nn.Linear(2 * channels, channels), nn.ReLU(), nn.Linear(channels, 4)
        )

        self.value_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.value_norm = nn.GroupNorm(groups, channels)
        self.value_fc = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, 1)
        )

    def forward(self, obs):
        """obs is float32 [B, N_PLANES, H, W]. Returns (logits[B, 4], value[B])."""
        # Plane 2 is the head one hot, so multiplying by it and summing over
        # space gathers the head cell's feature vector. This keeps the readout
        # differentiable, batched, and independent of the board size.
        head_mask = obs[:, 2:3]

        x = F.relu(self.stem_norm(self.stem_conv(obs)))
        for block in self.blocks:
            x = block(x)

        p = F.relu(self.policy_norm(self.policy_conv(x)))
        at_head = (p * head_mask).sum(dim=(2, 3))
        pooled = p.mean(dim=(2, 3))
        logits = self.policy_fc(torch.cat([at_head, pooled], dim=1))

        v = F.relu(self.value_norm(self.value_conv(x)))
        value = torch.sigmoid(self.value_fc(v.mean(dim=(2, 3)))).squeeze(1)

        return logits, value
