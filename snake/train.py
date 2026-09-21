"""The optimizer loop, metrics, and evaluation."""

from __future__ import annotations

import dataclasses
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from snake.arena import play_games, summarize
from snake.config import RunConfig, seed_everything
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
        # Evaluation must be deterministic on a fixed seed, so strip the root
        # exploration noise the training config carries. The arena compares this
        # agent against baselines that have no such noise.
        self.search_cfg = dataclasses.replace(cfg.search, dirichlet_epsilon=0.0)
        self.rng = rng

    def act(self, env):
        search = Search(
            env, self.search_cfg, np.random.default_rng(int(self.rng.integers(_SEED_MAX)))
        )
        run_search([search], self.evaluator, self.search_cfg.simulations)
        counts = search.visit_counts()
        # A child with zero visits and an illegal (None) child both read as 0
        # here, so an all-zero count vector must not fall back to argmax's
        # index-0 tie break: mask out illegal actions before choosing.
        counts = np.where(env.legal_actions(), counts, -1.0)
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
        self.best_eval_score = -1.0
        # Eval scores of the retained best slots, descending. Slot i is
        # best.pt for i == 0 and best{i+1}.pt thereafter.
        self.best_scores = []
        self._pending_eval_score = None

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
                # Offset from the run seed so the games used to pick best.pt
                # during training are never the same games `cli eval` later
                # reports on with the bare run seed; otherwise the headline
                # number would be a max over many attempts on its own test set.
                seed=self.cfg.train.seed + 1_000_000,
            ),
            total_cells=size * size,
        )
        for key in ("mean_score", "median_score", "max_score", "mean_fill_fraction", "mean_steps"):
            self.writer.add_scalar(f"eval/{key}", summary[key], self.iteration)
        for outcome, count in summary["outcomes"].items():
            self.writer.add_scalar(f"eval/outcome_{outcome}", count, self.iteration)
        return summary

    def save_checkpoint(self, tag):
        directory = self.run_dir / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{tag}.pt"
        torch.save(
            {
                "config": self.cfg.to_json(),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iteration": self.iteration,
                "step": self.step,
                "buffer": self.buffer.items(),
                "rng_state": self.rng.bit_generator.state,
                "best_eval_score": self.best_eval_score,
                "best_scores": list(self.best_scores),
                "eval_score": self._pending_eval_score,
            },
            path,
        )
        self._prune_checkpoints(directory)
        return path

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.iteration = payload["iteration"]
        self.step = payload["step"]
        self.buffer.load(payload["buffer"])
        self.rng.bit_generator.state = payload["rng_state"]
        self.best_eval_score = payload["best_eval_score"]
        # Checkpoints from before ranked retention carry best_eval_score
        # but no list. That older format is exactly a one entry ranking,
        # so it migrates rather than failing.
        if "best_scores" in payload:
            self.best_scores = list(payload["best_scores"])
        else:
            self.best_scores = (
                [payload["best_eval_score"]]
                if payload["best_eval_score"] >= 0
                else []
            )

    @staticmethod
    def _best_name(index):
        """Slot 0 stays best.pt so existing tooling and docs keep working."""
        return "best.pt" if index == 0 else f"best{index + 1}.pt"

    def record_best(self, score):
        """Offer an eval score to the ranking. Saves and returns True when it
        lands in the top keep_best, otherwise leaves the slots untouched.

        Keeping several peaks matters because a single slot loses every
        near-peak policy the moment a marginally better one appears, and a run
        that flatlines spends most of its life producing near-peaks.
        """
        keep = self.cfg.train.keep_best
        if len(self.best_scores) >= keep and score <= self.best_scores[-1]:
            return False

        directory = self.run_dir / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        position = sum(1 for existing in self.best_scores if existing >= score)

        # Shift the tail down one slot, starting from the bottom so nothing is
        # overwritten before it has been moved.
        last = min(len(self.best_scores), keep - 1)
        for index in range(last, position, -1):
            source = directory / self._best_name(index - 1)
            if source.exists():
                source.replace(directory / self._best_name(index))

        self.best_scores.insert(position, score)
        del self.best_scores[keep:]
        self.best_eval_score = self.best_scores[0]

        self._pending_eval_score = score
        try:
            self.save_checkpoint(self._best_name(position).removesuffix(".pt"))
        finally:
            self._pending_eval_score = None

        for index in range(keep, keep + 4):
            stale = directory / self._best_name(index)
            if stale.exists():
                stale.unlink()
        return True

    def _prune_checkpoints(self, directory):
        # "best.pt" is kept regardless of age; only the rolling ones are pruned.
        rolling = sorted(
            (p for p in directory.glob("*.pt") if not p.stem.startswith("best")),
            key=lambda p: (p.stat().st_mtime, p.name),
        )
        for stale in rolling[: max(0, len(rolling) - self.cfg.train.keep_last)]:
            stale.unlink()


def load_for_inference(path, device):
    """Rebuild a network from a checkpoint without constructing a Trainer, which
    is what the menu integration in a later phase will use."""
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = RunConfig.from_json(payload["config"])
    model = SnakeNet(cfg.net).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg
