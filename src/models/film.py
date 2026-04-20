"""FiLM (Feature-wise Linear Modulation) layer for task conditioning."""

import torch
from torch import Tensor, nn


class FiLMLayer(nn.Module):
    """
    FiLM layer: modulates a feature map with (gamma, beta) derived from a task embedding.

    Output = (1 + gamma) * x + beta, where gamma, beta are projected from embedding.
    The (1 + gamma) form makes the identity (gamma_raw=0, beta=0) a natural init point.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_features: int,
    ):
        """
        Args:
            embedding_dim: Dimensionality of the conditioning embedding.
            num_features: Number of channels in the feature map to modulate.
        """
        super().__init__()
        self.num_features = num_features
        self.projection = nn.Linear(embedding_dim, 2 * num_features)

    def forward(
        self,
        x: Tensor,
        embedding: Tensor,
    ) -> Tensor:
        """
        Args:
            x: Feature map of shape (B, C, H, W).
            embedding: Task embedding of shape (B, embedding_dim).

        Returns:
            Modulated feature map of shape (B, C, H, W).
        """
        params = self.projection(embedding)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * x + beta
