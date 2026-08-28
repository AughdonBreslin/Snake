"""The optimizer loop, metrics, and evaluation."""

from __future__ import annotations

import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from snake.arena import play_games, summarize
from snake.config import seed_everything
from snake.evaluator import TorchEvaluator
from snake.mcts import Search, run_search
from snake.model import SnakeNet
from snake.selfplay import ReplayBuffer, play_batch

_SEED_MAX = 2**63 - 1


def losses(logits, value, pi_target, z_target):
    """Cross entropy against the visit distribution plus MSE on the value.

    log_softmax is applied here, so the policy head must emit raw logits. This
    is the double softmax that the earlier god.py sketch had.
    """
    log_probs = F.log_softmax(logits, dim=1)
    policy = -(pi_target * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(value, z_target)
    with torch.no_grad():
        entropy = -(log_probs.exp() * log_probs).sum(dim=1).mean()
        value_mae = (value - z_target).abs().mean()
    return {
        "policy": policy,
        "value": value_loss,
        "total": policy + value_loss,
        "entropy": entropy,
        "value_mae": value_mae,
    }


class NetworkAgent:
    """Wraps search so the arena can evaluate a checkpoint like any baseline."""

    def __init__(self, evaluator, cfg, rng):
        self.evaluator = evaluator
        self.cfg = cfg
        self.rng = rng
        self.move_index = 0

    def act(self, env):
        search = Search(
            env, self.cfg.search, np.random.default_rng(int(self.rng.integers(_SEED_MAX)))
        )
        run_search([search], self.evaluator, self.cfg.search.simulations)
        counts = search.visit_counts()
        self.move_index += 1
        return int(np.argmax(counts))


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        seed_everything(cfg.train.seed)
        self.device = torch.device(cfg.train.device)
        self.model = SnakeNet(cfg.net).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        self.evaluator = TorchEvaluator(self.model, self.device)
        self.buffer = ReplayBuffer(cfg.train.replay_capacity)
        self.rng = np.random.default_rng(cfg.train.seed)
        self.iteration = 0
        self.step = 0

        self.run_dir = pathlib.Path(cfg.train.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(cfg.to_json() + "\n")
        self.writer = SummaryWriter(str(self.run_dir / "tb"))

    def train_step(self, batch):
        obs, pi, z = batch
        self.model.train()
        tensors = [torch.from_numpy(a).to(self.device) for a in (obs, pi, z)]
        logits, value = self.model(tensors[0])
        out = losses(logits, value, tensors[1], tensors[2])
        self.optimizer.zero_grad(set_to_none=True)
        out["total"].backward()
        self.optimizer.step()
        self.step += 1
        return {key: float(tensor.item()) for key, tensor in out.items()}

    def run_iteration(self):
        started = time.perf_counter()
        self.model.eval()
        positions, results = play_batch(self.cfg, self.evaluator, self.rng)
        self.buffer.extend(positions)
        elapsed = time.perf_counter() - started

        metrics = {}
        for _ in range(self.cfg.train.train_steps_per_iteration):
            if len(self.buffer) < self.cfg.train.batch_size:
                break
            metrics = self.train_step(
                self.buffer.sample(self.cfg.train.batch_size, self.rng)
            )

        metrics.setdefault("policy", float("nan"))
        metrics.setdefault("value", float("nan"))
        metrics.setdefault("total", float("nan"))
        metrics.setdefault("entropy", float("nan"))
        metrics.setdefault("value_mae", float("nan"))
        metrics["mean_score"] = float(np.mean([r.score for r in results]))
        metrics["mean_steps"] = float(np.mean([r.steps for r in results]))
        metrics["games_per_second"] = len(results) / max(elapsed, 1e-9)
        metrics["buffer_size"] = float(len(self.buffer))
        metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        for key, value in metrics.items():
            self.writer.add_scalar(f"train/{key}", value, self.iteration)
        self.iteration += 1
        return metrics

    def evaluate(self):
        self.model.eval()
        size = self.cfg.train.board_size
        summary = summarize(
            play_games(
                lambda rng: NetworkAgent(self.evaluator, self.cfg, rng),
                size=size,
                n_games=self.cfg.train.eval_games,
                seed=self.cfg.train.seed,
            ),
            total_cells=size * size,
        )
        for key in ("mean_score", "median_score", "max_score", "mean_fill_fraction", "mean_steps"):
            self.writer.add_scalar(f"eval/{key}", summary[key], self.iteration)
        for outcome, count in summary["outcomes"].items():
            self.writer.add_scalar(f"eval/outcome_{outcome}", count, self.iteration)
        return summary
