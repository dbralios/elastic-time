from typing import Optional

import torch
import torch.nn as nn


class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate condition embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        condition_dim (`int`): The size of the conditioning vector.
    """

    def __init__(self, embedding_dim: int, condition_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, embedding_dim * 2),
        )
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

        # Initialize the last linear layer to zero
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        emb = self.mlp(condition)
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        """
        Transpose two dimensions of the input tensor.

        Args:
            dim0 (int): First dimension to swap.
            dim1 (int): Second dimension to swap.
        """
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Gx = torch.norm(x, p=2, dim=(1,), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int = None,
        layer_scale_init_value: float = None,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = dim * 2

        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value is not None and layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()

        self.residual_block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim),
            Transpose(1, 2),
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            GRN(2 * dim),
            nn.Linear(2 * dim, dim),
            Transpose(1, 2),
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.residual_block(x)
        x = input + x
        return x


class ConditionalConvNeXtV2Block(nn.Module):
    """Conditional ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        cond_dim (int): Number of condition channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim: int, cond_dim: int, drop_path: float = 0.0):
        super().__init__()

        self.pre_block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim),
            Transpose(1, 2),
        )
        self.norm = AdaLayerNorm(dim, cond_dim)
        self.post_block = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            GRN(2 * dim),
            nn.Linear(2 * dim, dim),
            Transpose(1, 2),
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).
            cond (torch.Tensor): Condition tensor of shape (B, cond_dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, T).
        """
        input = x
        cond = cond.unsqueeze(1)  # (B, cond_dim) -> (B, 1, cond_dim)
        x = self.pre_block(x)
        x = self.norm(x, cond)
        x = self.post_block(x)
        x = input + x
        return x
