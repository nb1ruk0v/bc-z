"""Metrics utilities for BC-Z model evaluation and training."""

import torch
from torch import Tensor


def mse_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """
    Compute Mean Squared Error loss.

    Args:
        pred: Predicted values (B, ...)
        target: Ground truth values (B, ...)
        reduction: Reduction method ('mean', 'sum', 'none')

    Returns:
        MSE loss value
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    loss = torch.nn.functional.mse_loss(pred, target, reduction=reduction)
    return loss


def huber_loss(pred: Tensor, target: Tensor, delta: float = 1.0, reduction: str = "mean") -> Tensor:
    """
    Compute Huber loss (smooth L1 loss).

    Huber loss is less sensitive to outliers than MSE.
    It's quadratic for small errors and linear for large errors.

    Args:
        pred: Predicted values (B, ...)
        target: Ground truth values (B, ...)
        delta: Threshold where loss transitions from quadratic to linear
        reduction: Reduction method ('mean', 'sum', 'none')

    Returns:
        Huber loss value
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    loss = torch.nn.functional.huber_loss(pred, target, delta=delta, reduction=reduction)
    return loss


def xyz_position_error(pred: Tensor, target: Tensor) -> Tensor:
    """
    Compute Euclidean distance error for XYZ positions.

    Args:
        pred: Predicted XYZ positions (B, 3) or (B, N, 3)
        target: Ground truth XYZ positions (B, 3) or (B, N, 3)

    Returns:
        Euclidean distance error (B,) or (B, N)
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    if pred.shape[-1] != 3:
        raise ValueError(f"Expected last dimension to be 3, got {pred.shape[-1]}")

    # Compute Euclidean distance
    error = torch.norm(pred - target, dim=-1)
    return error


def quaternion_distance(pred: Tensor, target: Tensor) -> Tensor:
    """
    Compute quaternion distance metric.

    Uses the inner product distance: d(q1, q2) = 1 - |<q1, q2>|
    This metric is invariant to the sign ambiguity of quaternions (q and -q
    represent the same rotation).

    Args:
        pred: Predicted quaternions (B, 4) or (B, N, 4), normalized
        target: Ground truth quaternions (B, 4) or (B, N, 4), normalized

    Returns:
        Quaternion distance (B,) or (B, N), in range [0, 1]
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    if pred.shape[-1] != 4:
        raise ValueError(f"Expected last dimension to be 4, got {pred.shape[-1]}")

    # Normalize quaternions to unit length
    pred_norm = pred / (torch.norm(pred, dim=-1, keepdim=True) + 1e-8)
    target_norm = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)

    # Compute inner product
    inner_product = torch.sum(pred_norm * target_norm, dim=-1)

    # Distance metric: 1 - |<q1, q2>|
    # The absolute value handles the sign ambiguity
    distance = 1.0 - torch.abs(inner_product)

    return distance


def quaternion_angular_error(pred: Tensor, target: Tensor) -> Tensor:
    """
    Compute angular error in degrees between predicted and target quaternions.

    Args:
        pred: Predicted quaternions (B, 4) or (B, N, 4)
        target: Ground truth quaternions (B, 4) or (B, N, 4)

    Returns:
        Angular error in degrees (B,) or (B, N)
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    if pred.shape[-1] != 4:
        raise ValueError(f"Expected last dimension to be 4, got {pred.shape[-1]}")

    # Normalize quaternions
    pred_norm = pred / (torch.norm(pred, dim=-1, keepdim=True) + 1e-8)
    target_norm = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)

    # Compute inner product
    inner_product = torch.sum(pred_norm * target_norm, dim=-1)

    # Clamp to avoid numerical issues with acos
    inner_product = torch.clamp(torch.abs(inner_product), -1.0, 1.0)

    # Convert to angular error in degrees
    # θ = 2 * arccos(|<q1, q2>|)
    angular_error = 2.0 * torch.acos(inner_product) * (180.0 / torch.pi)

    return angular_error


def gripper_accuracy(pred: Tensor, target: Tensor, threshold: float = 0.5) -> Tensor:
    """
    Compute binary accuracy for gripper open/close prediction.

    Args:
        pred: Predicted gripper values (B,) or (B, N), in range [0, 1]
        target: Ground truth gripper values (B,) or (B, N), binary {0, 1}
        threshold: Threshold for binarizing predictions (default: 0.5)

    Returns:
        Accuracy value in [0, 1]
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    # Binarize predictions
    pred_binary = (pred > threshold).float()

    # Compute accuracy
    correct = (pred_binary == target).float()
    accuracy = correct.mean()

    return accuracy


def compute_action_metrics(
    pred_actions: dict[str, Tensor],
    target_actions: dict[str, Tensor],
) -> dict[str, float]:
    """
    Compute comprehensive metrics for action predictions.

    Args:
        pred_actions: Dictionary with predicted actions
            - 'xyz': (B, N, 3) or (B, 3)
            - 'quaternion': (B, N, 4) or (B, 4)
            - 'gripper': (B, N, 1) or (B, 1)
        target_actions: Dictionary with ground truth actions (same structure)

    Returns:
        Dictionary with computed metrics:
            - 'xyz_mse': Mean squared error for XYZ
            - 'xyz_error': Mean Euclidean distance
            - 'quat_distance': Mean quaternion distance
            - 'quat_angular_error': Mean angular error in degrees
            - 'gripper_accuracy': Binary accuracy for gripper

    Raises:
        KeyError: If required keys are missing
        ValueError: If shapes don't match
    """
    metrics = {}

    # XYZ metrics
    if "xyz" in pred_actions and "xyz" in target_actions:
        pred_xyz = pred_actions["xyz"]
        target_xyz = target_actions["xyz"]

        metrics["xyz_mse"] = mse_loss(pred_xyz, target_xyz).item()
        metrics["xyz_error"] = xyz_position_error(pred_xyz, target_xyz).mean().item()

    # Quaternion metrics
    if "quaternion" in pred_actions and "quaternion" in target_actions:
        pred_quat = pred_actions["quaternion"]
        target_quat = target_actions["quaternion"]

        metrics["quat_distance"] = quaternion_distance(pred_quat, target_quat).mean().item()
        metrics["quat_angular_error"] = (
            quaternion_angular_error(pred_quat, target_quat).mean().item()
        )

    # Gripper metrics
    if "gripper" in pred_actions and "gripper" in target_actions:
        pred_gripper = pred_actions["gripper"]
        target_gripper = target_actions["gripper"]

        # Squeeze if needed to handle (B, 1) -> (B,)
        if pred_gripper.dim() > 1 and pred_gripper.shape[-1] == 1:
            pred_gripper = pred_gripper.squeeze(-1)
        if target_gripper.dim() > 1 and target_gripper.shape[-1] == 1:
            target_gripper = target_gripper.squeeze(-1)

        metrics["gripper_accuracy"] = gripper_accuracy(pred_gripper, target_gripper).item()

    return metrics


def compute_per_waypoint_metrics(
    pred_actions: dict[str, Tensor],
    target_actions: dict[str, Tensor],
) -> dict[str, list[float]]:
    """
    Compute metrics separately for each waypoint in trajectory.

    Args:
        pred_actions: Dictionary with predicted action trajectories
            - 'xyz': (B, N, 3)
            - 'quaternion': (B, N, 4)
            - 'gripper': (B, N, 1)
        target_actions: Dictionary with ground truth trajectories

    Returns:
        Dictionary with per-waypoint metrics, each value is a list of length N
    """
    metrics = {}

    if "xyz" not in pred_actions or "xyz" not in target_actions:
        raise KeyError("'xyz' key required in both pred_actions and target_actions")

    num_waypoints = pred_actions["xyz"].shape[1]

    # Initialize metric lists
    metrics["xyz_error_per_waypoint"] = []
    metrics["quat_angular_error_per_waypoint"] = []
    metrics["gripper_accuracy_per_waypoint"] = []

    # Compute metrics for each waypoint
    for i in range(num_waypoints):
        # Extract waypoint
        pred_waypoint = {key: value[:, i] for key, value in pred_actions.items()}
        target_waypoint = {key: value[:, i] for key, value in target_actions.items()}

        # Compute metrics
        waypoint_metrics = compute_action_metrics(pred_waypoint, target_waypoint)

        metrics["xyz_error_per_waypoint"].append(waypoint_metrics.get("xyz_error", 0.0))
        metrics["quat_angular_error_per_waypoint"].append(
            waypoint_metrics.get("quat_angular_error", 0.0)
        )
        metrics["gripper_accuracy_per_waypoint"].append(
            waypoint_metrics.get("gripper_accuracy", 0.0)
        )

    return metrics
