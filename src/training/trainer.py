"""BC-Z training loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.training.losses import compute_bcz_loss

LogFn = Callable[[dict[str, Any]], None]

_TARGET_KEYS = (
    "future_xyz_residual",
    "future_axis_angle_residual",
    "future_target_close",
)


class Trainer:
    """Drives training and validation for a BC-Z policy.

    Per BC-Z Appendix D, Gaussian noise (std=0.1 by default) is added to the
    sentence embedding during training. The Trainer is logger-agnostic: pass
    ``log_fn`` (e.g. a trackio-wrapping callable) to record metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        val_loader: DataLoader | None = None,
        scheduler: LRScheduler | None = None,
        device: torch.device | str = "cpu",
        loss_weights: dict[str, float] | None = None,
        embedding_noise_std: float = 0.1,
        log_fn: LogFn | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.loss_weights = loss_weights
        self.embedding_noise_std = embedding_noise_std
        self.log_fn = log_fn
        self.global_step = 0

    # ---------- public API ----------

    def train_step(
        self,
        batch: dict[str, Tensor],
    ) -> dict[str, float]:
        self.model.train()
        batch = self._to_device(batch)

        state = self._build_state(batch)
        embedding = self._apply_embedding_noise(batch["sentence_embedding"])
        predictions = self.model(batch["image"], embedding, state)

        targets = {k: batch[k] for k in _TARGET_KEYS}
        total, components = compute_bcz_loss(predictions, targets, weights=self.loss_weights)

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()
        self.global_step += 1

        return _pack_metrics(total, components, prefix="loss")

    def validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()
        sums = {"total": 0.0, "xyz": 0.0, "axis_angle": 0.0, "gripper": 0.0}
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._to_device(batch)
                state = self._build_state(batch)
                predictions = self.model(batch["image"], batch["sentence_embedding"], state)
                targets = {k: batch[k] for k in _TARGET_KEYS}
                total, components = compute_bcz_loss(
                    predictions, targets, weights=self.loss_weights
                )
                sums["total"] += total.item()
                for k, v in components.items():
                    sums[k] += v.item()
                n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in sums.items()}
        return {f"val/loss/{k}": v for k, v in avg.items()}

    def fit(
        self,
        num_epochs: int,
    ) -> None:
        for epoch in range(num_epochs):
            for batch in self.train_loader:
                metrics = self.train_step(batch)
                if self.log_fn is not None:
                    self.log_fn({**metrics, "epoch": epoch, "global_step": self.global_step})

            val_metrics = self.validate()
            if val_metrics and self.log_fn is not None:
                self.log_fn({**val_metrics, "epoch": epoch, "global_step": self.global_step})

            if self.scheduler is not None:
                self.scheduler.step()

    # ---------- helpers ----------

    def _apply_embedding_noise(
        self,
        embedding: Tensor,
    ) -> Tensor:
        if self.embedding_noise_std <= 0.0:
            return embedding
        return embedding + self.embedding_noise_std * torch.randn_like(embedding)

    def _to_device(
        self,
        batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    @staticmethod
    def _build_state(
        batch: dict[str, Tensor],
    ) -> Tensor:
        return torch.cat(
            [batch["present_xyz"], batch["present_axis_angle"], batch["present_gripper"]],
            dim=-1,
        )


def _pack_metrics(
    total: Tensor,
    components: dict[str, Tensor],
    prefix: str,
) -> dict[str, float]:
    out = {f"{prefix}/total": total.item()}
    for k, v in components.items():
        out[f"{prefix}/{k}"] = v.item()
    return out
