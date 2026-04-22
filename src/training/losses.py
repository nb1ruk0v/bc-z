"""BC-Z training losses.

Per the BC-Z paper (Jang et al., CoRL 2021), §5.2 and Appendix D:
    - XYZ and axis-angle residuals are trained with Huber loss (delta=1.0).
    - Gripper target (normalized in [0, 1]) is trained with log loss (BCE).
    - Component weights: xyz=100, axis_angle=10, gripper=0.5 — chosen so that
      the three loss magnitudes are comparable.
"""

import torch.nn.functional as F
from torch import Tensor

DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "xyz": 100.0,
    "axis_angle": 10.0,
    "gripper": 0.5,
}


def compute_bcz_loss(
    predictions: dict[str, Tensor],
    targets: dict[str, Tensor],
    weights: dict[str, float] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute weighted BC-Z multi-head loss.

    Args:
        predictions: Policy outputs with keys
            ``future_xyz_residual`` (B, W, 3),
            ``future_axis_angle_residual`` (B, W, 3),
            ``future_target_close`` (B, W, 1) — raw logits.
        targets: Ground-truth tensors with the same keys and shapes; gripper
            target is expected in [0, 1].
        weights: Optional override for component weights. Defaults to
            ``DEFAULT_LOSS_WEIGHTS``.

    Returns:
        total_loss: Scalar tensor — weighted sum of components.
        components: Dict of unweighted per-component scalar losses
            ``{"xyz", "axis_angle", "gripper"}`` (useful for logging).
    """
    w = weights if weights is not None else DEFAULT_LOSS_WEIGHTS

    xyz_loss = F.smooth_l1_loss(
        predictions["future_xyz_residual"],
        targets["future_xyz_residual"],
        beta=1.0,
    )
    aa_loss = F.smooth_l1_loss(
        predictions["future_axis_angle_residual"],
        targets["future_axis_angle_residual"],
        beta=1.0,
    )
    gripper_loss = F.binary_cross_entropy_with_logits(
        predictions["future_target_close"],
        targets["future_target_close"],
    )

    components = {"xyz": xyz_loss, "axis_angle": aa_loss, "gripper": gripper_loss}
    total = w["xyz"] * xyz_loss + w["axis_angle"] * aa_loss + w["gripper"] * gripper_loss
    return total, components
