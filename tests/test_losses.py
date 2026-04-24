"""Tests for BC-Z training losses (Huber on xyz/axis-angle, BCE on gripper)."""

import math

import pytest
import torch
import torch.nn.functional as F

from src.training.losses import DEFAULT_LOSS_WEIGHTS, compute_bcz_loss


def _make_batch(
    batch: int = 4,
    num_waypoints: int = 10,
    gripper_target: float = 0.5,
):
    pred = {
        "future_xyz_residual": torch.randn(batch, num_waypoints, 3),
        "future_axis_angle_residual": torch.randn(batch, num_waypoints, 3),
        "future_target_close": torch.randn(batch, num_waypoints, 1),
    }
    tgt = {
        "future_xyz_residual": torch.randn(batch, num_waypoints, 3),
        "future_axis_angle_residual": torch.randn(batch, num_waypoints, 3),
        "future_target_close": torch.full((batch, num_waypoints, 1), gripper_target),
    }
    return pred, tgt


class TestBCZLoss:
    def test_returns_scalar_and_components(self):
        pred, tgt = _make_batch()
        total, components = compute_bcz_loss(pred, tgt)

        assert total.ndim == 0
        assert set(components.keys()) == {"xyz", "axis_angle", "gripper"}
        for v in components.values():
            assert v.ndim == 0

    def test_default_weights_match_paper(self):
        assert DEFAULT_LOSS_WEIGHTS == {"xyz": 100.0, "axis_angle": 10.0, "gripper": 0.5}

    def test_weights_are_applied(self):
        pred, tgt = _make_batch()
        weights = {"xyz": 2.0, "axis_angle": 3.0, "gripper": 5.0}
        total, comp = compute_bcz_loss(pred, tgt, weights=weights)

        expected = 2.0 * comp["xyz"] + 3.0 * comp["axis_angle"] + 5.0 * comp["gripper"]
        assert torch.allclose(total, expected, atol=1e-6)

    def test_components_are_unweighted(self):
        """Raw per-component values should not depend on weights (weights are for logging)."""
        pred, tgt = _make_batch()
        _, comp_a = compute_bcz_loss(
            pred, tgt, weights={"xyz": 1.0, "axis_angle": 1.0, "gripper": 1.0}
        )
        _, comp_b = compute_bcz_loss(
            pred, tgt, weights={"xyz": 100.0, "axis_angle": 10.0, "gripper": 0.5}
        )

        for k in comp_a:
            assert torch.allclose(comp_a[k], comp_b[k], atol=1e-6)

    @pytest.mark.parametrize(
        "key,comp_name",
        [
            ("future_xyz_residual", "xyz"),
            ("future_axis_angle_residual", "axis_angle"),
        ],
    )
    def test_regression_head_uses_huber_delta_1(self, key, comp_name):
        """Regression components equal F.smooth_l1_loss (Huber with delta=1.0, beta=1.0)."""
        pred, tgt = _make_batch()
        _, comp = compute_bcz_loss(
            pred, tgt, weights={"xyz": 1.0, "axis_angle": 1.0, "gripper": 1.0}
        )

        expected = F.smooth_l1_loss(pred[key], tgt[key], beta=1.0)
        assert torch.allclose(comp[comp_name], expected, atol=1e-6)

    def test_gripper_uses_bce_with_logits(self):
        """Zero logit + target=0.5 → BCE = ln(2); confirms logit interpretation."""
        pred = {
            "future_xyz_residual": torch.zeros(1, 1, 3),
            "future_axis_angle_residual": torch.zeros(1, 1, 3),
            "future_target_close": torch.zeros(1, 1, 1),
        }
        tgt = {
            "future_xyz_residual": torch.zeros(1, 1, 3),
            "future_axis_angle_residual": torch.zeros(1, 1, 3),
            "future_target_close": torch.full((1, 1, 1), 0.5),
        }
        _, comp = compute_bcz_loss(pred, tgt)
        assert comp["gripper"].item() == pytest.approx(math.log(2.0), abs=1e-5)

    def test_gradient_flows_to_predictions(self):
        pred, tgt = _make_batch()
        for v in pred.values():
            v.requires_grad_(True)

        total, _ = compute_bcz_loss(pred, tgt)
        total.backward()

        for k, v in pred.items():
            assert v.grad is not None, f"no grad for {k}"
            assert torch.isfinite(v.grad).all()
