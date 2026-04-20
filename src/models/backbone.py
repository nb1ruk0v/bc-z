"""ResNet backbone that exposes feature maps from all four stages."""

from torch import Tensor, nn
from torchvision import models

_ARCH_CHANNELS = {
    "resnet18": [64, 128, 256, 512],
    "resnet50": [256, 512, 1024, 2048],
}


class ResNetBackbone(nn.Module):
    """
    ResNet18/50 backbone returning intermediate features after each of the 4 stages.
    """

    def __init__(
        self,
        arch: str = "resnet18",
        pretrained: bool = True,
    ):
        """
        Args:
            arch: Either "resnet18" or "resnet50".
            pretrained: Use ImageNet-pretrained weights.
        """
        super().__init__()
        if arch not in _ARCH_CHANNELS:
            raise ValueError(f"Unsupported arch '{arch}'. Expected one of {list(_ARCH_CHANNELS)}.")

        if arch == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
        else:
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = models.resnet50(weights=weights)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.stage1 = resnet.layer1
        self.stage2 = resnet.layer2
        self.stage3 = resnet.layer3
        self.stage4 = resnet.layer4

        self.feature_channels = _ARCH_CHANNELS[arch]

    def forward(
        self,
        x: Tensor,
    ) -> list[Tensor]:
        """
        Args:
            x: Image tensor (B, 3, H, W).

        Returns:
            List of 4 feature maps, one per stage.
        """
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]
