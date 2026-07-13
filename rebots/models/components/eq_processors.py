import copy
from typing import Optional, Union

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig

from rebots import config


def _build(mod_or_cfg: Union[nn.Module, DictConfig, dict], *, expected: str, copy_mod: bool = False) -> nn.Module:
    if isinstance(mod_or_cfg, nn.Module):
        if copy_mod:
            return copy.deepcopy(mod_or_cfg)
        return mod_or_cfg
    elif isinstance(mod_or_cfg, DictConfig):
        return config.instantiate(mod_or_cfg)
    elif isinstance(mod_or_cfg, dict):
        try:
            mod_or_cfg = DictConfig(mod_or_cfg)
        except Exception as e:
            raise ValueError(f"Could not convert dict to DictConfig: {e}")
        return config.instantiate(mod_or_cfg)
    else:
        raise TypeError(f"{expected} must be an nn.Module or DictConfig, got {type(mod_or_cfg).__name__}.")


class ScalarOpConditioner(nn.Module):
    """Embeds a parametric operation into to condition representations."""

    def __init__(
        self,
        condition_dim: int,
        num_controls: int,
        embedder: Union[nn.Module, DictConfig, dict],
        mapper: Union[nn.Module, DictConfig, dict],
    ):
        super().__init__()

        self.condition_dim = condition_dim
        self.num_controls = num_controls

        self.embedder = _build(embedder, expected="embedder")
        self.mapper = _build(mapper, expected="mapper")

    def embed(self, controls: torch.Tensor) -> torch.Tensor:
        """
        Embed control parameters to a higher-dimensional space.

        Args:
            controls (torch.Tensor): Control parameters for the operator (B, steps, num_controls, 1). Normalized in the [-1, 1] range.

        Returns:
            torch.Tensor: Embedded control parameters of shape (B, steps, num_controls * D_cond).
        """
        if controls.dim() != 4:
            controls = controls.unsqueeze(1)  # (B, 1, num_bands, 1)
        B, S, num_controls, _ = controls.shape
        assert num_controls == self.num_controls, f"Expected {self.num_controls} bands, got {num_controls}."

        # Embed control parameters
        x = self.embedder(controls)
        # x shape: (B, S, num_controls, D_embed)

        x = rearrange(x, "B S num_controls D_embed -> B S (num_controls D_embed)")
        return x

    def map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map embedded control parameters to a condition representation.

        Args:
            x (torch.Tensor): Input embeddings of shape (B, S, num_bands * D_cond).

        Returns:
            torch.Tensor: Mapped output of shape (B, S, condition_dim).
        """
        if x.dim() != 3:
            x = x.unsqueeze(1)  # (B, 1, num_bands * D_cond)
        B, S, D = x.shape

        x = rearrange(x, "B S D -> (B S) D")
        x = self.mapper(x)
        x = rearrange(x, "(B S) d -> B S d", B=B, S=S)
        # x shape: (B, S, condition_dim)

        return x

    def forward(self, controls: torch.Tensor) -> torch.Tensor:
        """
        Apply the EQ processor to the latent representations.

        Args:
            controls (torch.Tensor): Control tensor of shape (B, num_controls, 1).

        Returns:
            torch.Tensor: Condition representations (B, condition_dim).

        Shape notation:
            B: Batch size
            C: Condition dimension
            T: Time dimension
        """

        controls = controls.unsqueeze(1)  # (B, 1, num_controls, 1)
        embeddings = self.embed(controls)  # (B, 1, num_controls * D_cond)

        return self.map(embeddings)[:, 0, :]  # (B, C)
