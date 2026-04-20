"""BC-Z policy: ResNet + FiLM + action head."""

import torch
from torch import Tensor, nn

from src.models.backbone import ResNetBackbone
from src.models.film import FiLMLayer


class BCZPolicy(nn.Module):
    """
    Language/embedding-conditioned policy for BC-Z.

    Architecture:
        image -> ResNet stem
              -> [stage_i -> FiLM(sentence_embedding)] x 4
              -> global avg pool
              -> concat(pooled_features, robot_state)
              -> MLP head -> (num_waypoints * 7) -> split into xyz/axis_angle/gripper
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
            hidden_dim: Hidden size of the MLP head.
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
        # TODO(week2-follow-up): split into 3 separate heads (xyz, axis_angle, gripper)
        # — they have different output natures (regression vs. binary logits) and may
        # benefit from independent capacity / loss weighting.
        self.head = nn.Sequential(
            nn.Linear(feature_dim + state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_waypoints * 7),
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
        out = self.head(joint)
        out = out.view(-1, self.num_waypoints, 7)

        return {
            "future_xyz_residual": out[..., 0:3],
            "future_axis_angle_residual": out[..., 3:6],
            "future_target_close": out[..., 6:7],
        }
