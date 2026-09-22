"""Run configuration. Serialized next to every checkpoint so a result three
weeks old can still be explained. No torch import at module level, so the
baseline commands keep working without the training stack."""

from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class NetConfig:
    channels: int = 64
    blocks: int = 6
    groups: int = 8


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.8
    dirichlet_epsilon: float = 0.25
    tau_threshold: int = 40
    tau_final: float = 0.2
    # Per move discount on future food. At 1.0 the target is the undiscounted
    # return to go, which makes a state one move from food and one thirty moves
    # from food identical, so the value head has no gradient to follow and a
    # trained agent wanders until it starves. Below 1.0 nearer food is worth
    # more. It lives here because search must back up the same quantity the
    # training target measures, and search only receives this config.
    discount: float = 1.0


@dataclass(frozen=True)
class TrainConfig:
    board_size: int = 6
    iterations: int = 200
    concurrent_games: int = 32
    train_steps_per_iteration: int = 200
    batch_size: int = 512
    replay_capacity: int = 200_000
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    eval_games: int = 100
    eval_every: int = 5
    checkpoint_every: int = 5
    keep_last: int = 3
    keep_best: int = 3
    seed: int = 0
    device: str = "cuda"
    run_dir: str = "runs/default"


@dataclass(frozen=True)
class RunConfig:
    net: NetConfig = field(default_factory=NetConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        if self.net.groups < 1:
            raise ValueError(f"groups must be at least 1, got {self.net.groups}")
        if self.net.channels % self.net.groups != 0:
            raise ValueError(
                f"channels {self.net.channels} must be divisible by "
                f"groups {self.net.groups}"
            )
        if self.train.board_size < 3:
            raise ValueError(f"board_size must be at least 3, got {self.train.board_size}")
        if self.search.simulations < 1:
            raise ValueError(
                f"simulations must be at least 1, got {self.search.simulations}"
            )

    @classmethod
    def build(cls, net=None, search=None, train=None):
        return cls(
            net=NetConfig(**(net or {})),
            search=SearchConfig(**(search or {})),
            train=TrainConfig(**(train or {})),
        )

    def to_json(self):
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text):
        raw = json.loads(text)
        return cls.build(net=raw["net"], search=raw["search"], train=raw["train"])


def seed_everything(seed):
    """Seed every global source of randomness. Envs still take an explicit
    generator; this covers torch, the legacy numpy global, and the stdlib module
    that graph.py reaches for."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
