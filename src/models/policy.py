"""BC-Z policy: ResNet + FiLM + action head."""

import torch
from torch import Tensor, nn

from src.models.backbone import ResNetBackbone
from src.models.film import FiLMLayer


class BCZPolicy(nn.Module):
    """
    Language/embedding-conditioned policy for BC-Z.

    Architecture (per BC-Z paper §5.3):
        image -> ResNet stem
              -> [stage_i -> FiLM(sentence_embedding)] x 4
              -> global avg pool
              -> concat(pooled_features, robot_state)
              -> 3 independent MLP heads (each: Linear -> ReLU -> Linear -> ReLU -> Linear):
                  - xyz_head        -> (num_waypoints, 3)
                  - axis_angle_head -> (num_waypoints, 3)
                  - gripper_head    -> (num_waypoints, 1)  (logits)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        embedding_dim: int = 512,
        state_dim: int = 7,
        num_waypoints: int = 10,
        hidden_dim: int = 256,
    ):
        """
        Args:
            backbone: "resnet18" or "resnet50".
            pretrained: Use ImageNet-pretrained backbone weights.
            embedding_dim: Task embedding dimensionality.
            state_dim: Robot state dimensionality (xyz=3, axis_angle=3, gripper=1 → 7).
            num_waypoints: Number of future waypoints to predict.
            hidden_dim: Hidden size of each MLP head.
        """
        super().__init__()
        self.num_waypoints = num_waypoints
        self.state_dim = state_dim

        self.backbone = ResNetBackbone(arch=backbone, pretrained=pretrained)

        self.film_layers = nn.ModuleList(
            [
                FiLMLayer(embedding_dim=embedding_dim, num_features=c)
                for c in self.backbone.feature_channels
            ]
        )

        feature_dim = self.backbone.feature_channels[-1]
        in_dim = feature_dim + state_dim
        self.xyz_head = self._make_head(in_dim, hidden_dim, num_waypoints * 3)
        self.axis_angle_head = self._make_head(in_dim, hidden_dim, num_waypoints * 3)
        self.gripper_head = self._make_head(in_dim, hidden_dim, num_waypoints * 1)

    @staticmethod
    def _make_head(
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        image: Tensor,
        sentence_embedding: Tensor,
        state: Tensor,
    ) -> dict[str, Tensor]:
        """
        Args:
            image: (B, 3, H, W) normalized image.
            sentence_embedding: (B, embedding_dim) task embedding.
            state: (B, state_dim) concatenated robot state
                [present_xyz(3), present_axis_angle(3), present_gripper(1)].

        Returns:
            Dict with keys:
                - future_xyz_residual: (B, num_waypoints, 3)
                - future_axis_angle_residual: (B, num_waypoints, 3)
                - future_target_close: (B, num_waypoints, 1)  (logits)
        """
        x = self.backbone.stem(image)
        stages = [
            self.backbone.stage1,
            self.backbone.stage2,
            self.backbone.stage3,
            self.backbone.stage4,
        ]
        for stage, film in zip(stages, self.film_layers, strict=True):
            x = stage(x)
            x = film(x, sentence_embedding)
        pooled = x.mean(dim=(2, 3))

        joint = torch.cat([pooled, state], dim=-1)
        batch = joint.shape[0]

        xyz = self.xyz_head(joint).view(batch, self.num_waypoints, 3)
        axis_angle = self.axis_angle_head(joint).view(batch, self.num_waypoints, 3)
        gripper = self.gripper_head(joint).view(batch, self.num_waypoints, 1)

        return {
            "future_xyz_residual": xyz,
            "future_axis_angle_residual": axis_angle,
            "future_target_close": gripper,
        }
