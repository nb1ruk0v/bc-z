"""Tests for metrics utilities."""

import pytest
import torch

from src.utils.metrics import (
    compute_action_metrics,
    gripper_accuracy,
    huber_loss,
    mse_loss,
    quaternion_angular_error,
    quaternion_distance,
    xyz_position_error,
)


class TestBasicMetrics:
    """Test basic loss and error functions."""

    def test_mse_loss(self):
        """Test MSE loss computation."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])

        loss = mse_loss(pred, target)
        assert loss.item() == 0.0

        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        loss = mse_loss(pred, target)
        assert loss.item() == 1.0

    def test_mse_loss_shape_mismatch(self):
        """Test MSE loss raises error on shape mismatch."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0])

        with pytest.raises(ValueError, match="Shape mismatch"):
            mse_loss(pred, target)

    def test_huber_loss(self):
        """Test Huber loss computation."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])

        loss = huber_loss(pred, target)
        assert loss.item() == 0.0

    def test_xyz_position_error(self):
        """Test XYZ position error computation."""
        # Same positions -> zero error
        pred = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.0, 2.0, 3.0]])

        error = xyz_position_error(pred, target)
        assert error.item() == pytest.approx(0.0, abs=1e-6)

        # Unit distance
        pred = torch.tensor([[0.0, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0]])

        error = xyz_position_error(pred, target)
        assert error.item() == pytest.approx(1.0, abs=1e-6)

    def test_xyz_position_error_invalid_shape(self):
        """Test XYZ error raises on invalid shapes."""
        pred = torch.tensor([[1.0, 2.0]])  # Only 2D
        target = torch.tensor([[1.0, 2.0]])

        with pytest.raises(ValueError, match="Expected last dimension to be 3"):
            xyz_position_error(pred, target)


class TestQuaternionMetrics:
    """Test quaternion-related metrics."""

    def test_quaternion_distance_identical(self):
        """Test quaternion distance for identical quaternions."""
        # Identity quaternion [w, x, y, z] = [1, 0, 0, 0]
        pred = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        distance = quaternion_distance(pred, target)
        assert distance.item() == pytest.approx(0.0, abs=1e-6)

    def test_quaternion_distance_sign_invariant(self):
        """Test that quaternion distance is invariant to sign flip."""
        pred = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        target = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])  # Same rotation

        distance = quaternion_distance(pred, target)
        assert distance.item() == pytest.approx(0.0, abs=1e-6)

    def test_quaternion_angular_error_identical(self):
        """Test angular error for identical quaternions."""
        pred = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        error = quaternion_angular_error(pred, target)
        assert error.item() == pytest.approx(0.0, abs=1e-3)

    def test_quaternion_angular_error_90deg(self):
        """Test angular error for 90 degree rotation."""
        # 90 degree rotation around Z-axis
        import math

        angle = math.pi / 4  # Half-angle for quaternion
        pred = torch.tensor([[math.cos(angle), 0.0, 0.0, math.sin(angle)]])
        target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity

        error = quaternion_angular_error(pred, target)
        assert error.item() == pytest.approx(90.0, abs=1.0)


class TestGripperMetrics:
    """Test gripper-related metrics."""

    def test_gripper_accuracy_perfect(self):
        """Test gripper accuracy with perfect predictions."""
        pred = torch.tensor([0.1, 0.9, 0.2, 0.8])
        target = torch.tensor([0.0, 1.0, 0.0, 1.0])

        acc = gripper_accuracy(pred, target, threshold=0.5)
        assert acc.item() == 1.0

    def test_gripper_accuracy_partial(self):
        """Test gripper accuracy with partial correctness."""
        pred = torch.tensor([0.1, 0.9, 0.8, 0.2])  # Last two are wrong
        target = torch.tensor([0.0, 1.0, 0.0, 1.0])

        acc = gripper_accuracy(pred, target, threshold=0.5)
        assert acc.item() == 0.5


class TestActionMetrics:
    """Test comprehensive action metrics."""

    def test_compute_action_metrics_complete(self):
        """Test computing all action metrics."""
        batch_size = 4

        pred_actions = {
            "xyz": torch.randn(batch_size, 3),
            "quaternion": torch.randn(batch_size, 4),
            "gripper": torch.rand(batch_size, 1),
        }

        target_actions = {
            "xyz": torch.randn(batch_size, 3),
            "quaternion": torch.randn(batch_size, 4),
            "gripper": torch.randint(0, 2, (batch_size, 1)).float(),
        }

        metrics = compute_action_metrics(pred_actions, target_actions)

        # Check all expected metrics are present
        assert "xyz_mse" in metrics
        assert "xyz_error" in metrics
        assert "quat_distance" in metrics
        assert "quat_angular_error" in metrics
        assert "gripper_accuracy" in metrics

        # Check all metrics are scalars
        for value in metrics.values():
            assert isinstance(value, float)

    def test_compute_action_metrics_partial(self):
        """Test computing metrics with partial data."""
        pred_actions = {"xyz": torch.randn(4, 3)}
        target_actions = {"xyz": torch.randn(4, 3)}

        metrics = compute_action_metrics(pred_actions, target_actions)

        # Should only have XYZ metrics
        assert "xyz_mse" in metrics
        assert "xyz_error" in metrics
        assert "quat_distance" not in metrics
        assert "gripper_accuracy" not in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
