"""Tests for BC-Z model components: FiLM layer, backbone, policy."""

import pytest
import torch
from torch import nn

from src.models.backbone import ResNetBackbone
from src.models.film import FiLMLayer
from src.models.policy import BCZPolicy

# ---------- FiLMLayer ----------


class TestFiLMLayer:
    def test_output_shape_matches_input(self):
        layer = FiLMLayer(embedding_dim=512, num_features=64)
        x = torch.randn(2, 64, 8, 8)
        emb = torch.randn(2, 512)
        out = layer(x, emb)
        assert out.shape == x.shape

    def test_affine_modulation(self):
        """FiLM applies gamma * x + beta; with gamma=1, beta=0 the output equals input."""
        layer = FiLMLayer(embedding_dim=512, num_features=16)
        with torch.no_grad():
            # Make projection produce gamma=1, beta=0 regardless of input
            layer.projection.weight.zero_()
            layer.projection.bias.zero_()
            # gamma is first half of output; FiLM uses (1 + gamma_raw) so zeros -> gamma=1
            # beta is second half -> 0
        x = torch.randn(4, 16, 5, 5)
        emb = torch.randn(4, 512)
        out = layer(x, emb)
        assert torch.allclose(out, x, atol=1e-6)

    def test_gradient_flow(self):
        layer = FiLMLayer(embedding_dim=32, num_features=8)
        x = torch.randn(2, 8, 4, 4, requires_grad=True)
        emb = torch.randn(2, 32, requires_grad=True)
        out = layer(x, emb)
        out.sum().backward()
        assert x.grad is not None and emb.grad is not None
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(emb.grad).all()


# ---------- ResNetBackbone ----------


class TestResNetBackbone:
    @pytest.mark.parametrize("arch", ["resnet18", "resnet50"])
    def test_returns_four_stage_features(self, arch):
        backbone = ResNetBackbone(arch=arch, pretrained=False)
        x = torch.randn(2, 3, 100, 100)
        feats = backbone(x)
        assert isinstance(feats, list)
        assert len(feats) == 4

    def test_resnet18_channels(self):
        backbone = ResNetBackbone(arch="resnet18", pretrained=False)
        assert backbone.feature_channels == [64, 128, 256, 512]

    def test_resnet50_channels(self):
        backbone = ResNetBackbone(arch="resnet50", pretrained=False)
        assert backbone.feature_channels == [256, 512, 1024, 2048]

    def test_invalid_arch_raises(self):
        with pytest.raises(ValueError):
            ResNetBackbone(arch="resnet999", pretrained=False)


# ---------- BCZPolicy ----------


@pytest.fixture
def default_policy():
    return BCZPolicy(
        backbone="resnet18",
        pretrained=False,
        embedding_dim=512,
        state_dim=7,
        num_waypoints=10,
    )


class TestBCZPolicy:
    def test_forward_output_shapes(self, default_policy):
        batch = 4
        image = torch.randn(batch, 3, 100, 100)
        embedding = torch.randn(batch, 512)
        state = torch.randn(batch, 7)
        out = default_policy(image, embedding, state)

        assert set(out.keys()) == {
            "future_xyz_residual",
            "future_axis_angle_residual",
            "future_target_close",
        }
        assert out["future_xyz_residual"].shape == (batch, 10, 3)
        assert out["future_axis_angle_residual"].shape == (batch, 10, 3)
        assert out["future_target_close"].shape == (batch, 10, 1)

    def test_gradient_flow(self, default_policy):
        image = torch.randn(2, 3, 100, 100)
        embedding = torch.randn(2, 512)
        state = torch.randn(2, 7)
        out = default_policy(image, embedding, state)
        loss = sum(v.sum() for v in out.values())
        loss.backward()

        has_grad = [
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in default_policy.parameters()
            if p.requires_grad
        ]
        assert all(has_grad)

    def test_batch_independence(self, default_policy):
        """Different samples in batch should produce different outputs."""
        default_policy.eval()
        image = torch.randn(2, 3, 100, 100)
        embedding = torch.randn(2, 512)
        state = torch.randn(2, 7)
        with torch.no_grad():
            out = default_policy(image, embedding, state)
        assert not torch.allclose(out["future_xyz_residual"][0], out["future_xyz_residual"][1])

    def test_resnet50_variant(self):
        policy = BCZPolicy(
            backbone="resnet50",
            pretrained=False,
            embedding_dim=512,
            state_dim=7,
            num_waypoints=10,
        )
        image = torch.randn(2, 3, 100, 100)
        embedding = torch.randn(2, 512)
        state = torch.randn(2, 7)
        out = policy(image, embedding, state)
        assert out["future_xyz_residual"].shape == (2, 10, 3)

    def test_is_nn_module(self, default_policy):
        assert isinstance(default_policy, nn.Module)

    def test_film_layers_registered(self, default_policy):
        film_layers = [m for m in default_policy.modules() if isinstance(m, FiLMLayer)]
        assert len(film_layers) == 4
