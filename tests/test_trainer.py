"""Tests for the BC-Z Trainer."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.training.trainer import Trainer


class _StubPolicy(nn.Module):
    """Minimal policy with BCZPolicy's forward signature and output keys.

    Uses a single linear layer so tests are fast; Trainer logic is independent
    of the backbone, which is already covered by test_models.py.
    """

    def __init__(
        self,
        embedding_dim: int = 8,
        state_dim: int = 7,
        num_waypoints: int = 4,
        image_feat_dim: int = 4,
    ):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.image_feat_dim = image_feat_dim
        self.proj = nn.Linear(image_feat_dim + embedding_dim + state_dim, num_waypoints * 7)

    def forward(
        self,
        image: Tensor,
        sentence_embedding: Tensor,
        state: Tensor,
    ) -> dict[str, Tensor]:
        b = image.shape[0]
        img_feat = image.flatten(1).mean(dim=1, keepdim=True).expand(-1, self.image_feat_dim)
        x = torch.cat([img_feat, sentence_embedding, state], dim=-1)
        out = self.proj(x).view(b, self.num_waypoints, 7)
        return {
            "future_xyz_residual": out[..., 0:3],
            "future_axis_angle_residual": out[..., 3:6],
            "future_target_close": out[..., 6:7],
        }


class _StubDataset(Dataset):
    """Random tensors matching BCZDataset's output dict shape."""

    def __init__(
        self,
        size: int = 8,
        num_waypoints: int = 4,
        embedding_dim: int = 8,
        seed: int = 0,
    ):
        self.size = size
        g = torch.Generator().manual_seed(seed)
        self.samples = [
            {
                "image": torch.rand(3, 16, 16, generator=g),
                "sentence_embedding": torch.randn(embedding_dim, generator=g),
                "present_xyz": torch.randn(3, generator=g),
                "present_axis_angle": torch.randn(3, generator=g),
                "present_gripper": torch.rand(1, generator=g),
                "future_xyz_residual": torch.randn(num_waypoints, 3, generator=g),
                "future_axis_angle_residual": torch.randn(num_waypoints, 3, generator=g),
                "future_target_close": torch.bernoulli(
                    torch.full((num_waypoints, 1), 0.5), generator=g
                ),
            }
            for _ in range(size)
        ]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return self.samples[idx]


@pytest.fixture
def trainer_components():
    torch.manual_seed(0)
    model = _StubPolicy(embedding_dim=8, num_waypoints=4)
    train_loader = DataLoader(_StubDataset(size=4, embedding_dim=8, num_waypoints=4), batch_size=2)
    val_loader = DataLoader(
        _StubDataset(size=4, embedding_dim=8, num_waypoints=4, seed=1),
        batch_size=2,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    return model, train_loader, val_loader, optimizer


# ---------- Training loop ----------


def test_reduces_loss_on_repeated_steps(trainer_components):
    """Sanity: optimizer.step() actually updates weights and lowers loss on one batch."""
    model, train_loader, _, optimizer = trainer_components
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        embedding_noise_std=0.0,
    )
    batch = next(iter(train_loader))

    initial = trainer.train_step(batch)["loss/total"]
    for _ in range(30):
        trainer.train_step(batch)
    final = trainer.train_step(batch)["loss/total"]

    assert final < initial * 0.5, f"loss did not decrease: {initial} -> {final}"


# ---------- Embedding noise ----------


def test_nonzero_noise_is_stochastic(trainer_components):
    model, train_loader, _, optimizer = trainer_components
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        embedding_noise_std=0.1,
    )
    emb = torch.randn(8, 8)
    torch.manual_seed(1)
    a = trainer._apply_embedding_noise(emb)
    torch.manual_seed(2)
    b = trainer._apply_embedding_noise(emb)
    assert not torch.equal(a, b)
    # Empirical std of (a - emb) should be near 0.1 for a reasonably sized tensor.
    emp = (a - emb).std().item()
    assert 0.05 < emp < 0.2, f"empirical noise std = {emp}"


# ---------- Validation ----------


def test_validate_sets_eval_mode_and_no_grad(trainer_components):
    model, train_loader, val_loader, optimizer = trainer_components
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
    )
    model.train()

    # Snapshot params; validate must not change them.
    before = [p.detach().clone() for p in model.parameters()]
    metrics = trainer.validate()
    after = list(model.parameters())

    assert not model.training
    assert set(metrics.keys()) == {
        "val/loss/total",
        "val/loss/xyz",
        "val/loss/axis_angle",
        "val/loss/gripper",
    }
    for b, a in zip(before, after, strict=True):
        assert torch.equal(b, a), "validate must not update parameters"


# ---------- Fit integration ----------


def test_fit_calls_log_fn_with_metrics(trainer_components):
    model, train_loader, val_loader, optimizer = trainer_components
    logged: list[dict] = []
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        log_fn=logged.append,
    )

    trainer.fit(num_epochs=2)

    assert len(logged) > 0
    # Every log record carries an "epoch" field.
    assert all("epoch" in r for r in logged)
    # At least one record contains a train-loss key; at least one contains a val-loss key.
    assert any("loss/total" in r for r in logged)
    assert any("val/loss/total" in r for r in logged)
