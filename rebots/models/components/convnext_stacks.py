import math
from typing import Optional, Union

import torch
import torch.nn as nn

from rebots.modules.convnext import ConditionalConvNeXtV2Block, ConvNeXtV2Block


class ConvNeXtStack(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_blocks: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
    ):
        super(ConvNeXtStack, self).__init__()

        if output_dim is None:
            output_dim = input_dim

        self.blocks = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            *[ConvNeXtV2Block(hidden_dim) for _ in range(num_blocks)],
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvNeXt stack.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, T).
        """
        return self.blocks(x)


class ConditionalConvNeXtStack(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        num_blocks: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
    ):
        super(ConditionalConvNeXtStack, self).__init__()

        if output_dim is None:
            output_dim = input_dim

        self.pre_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.post_conv = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            *[ConditionalConvNeXtV2Block(hidden_dim, cond_dim) for _ in range(num_blocks)],
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Conditional ConvNeXt stack.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).
            cond (torch.Tensor): Condition tensor of shape (B, cond_dim).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, T).
        """
        x = self.pre_conv(x)
        for block in self.blocks:
            x = block(x, cond)
        x = self.post_conv(x)
        return x


# CAUTION: The downsample/upsample stacks assume even stride for predictable output lengths.
# Odd input sizes may lead to rounding issues. Use with even input sizes or be mindful of potential off-by-one length differences.


class ConvNeXtStackDownsample(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_blocks: int,
        hidden_dim: int,
        stride: int,
        output_dim: Optional[int] = None,
    ):
        super(ConvNeXtStackDownsample, self).__init__()

        if stride % 2 != 0:
            raise ValueError("Use even stride (2,4,8,...) for predictable lengths.")

        if output_dim is None:
            output_dim = input_dim

        self.blocks = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            *[ConvNeXtV2Block(hidden_dim) for _ in range(num_blocks)],
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvNeXt stack.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, T).
        """
        return self.blocks(x)


class ConvNeXtStackUpsample(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_blocks: int,
        hidden_dim: int,
        stride: int,
        output_dim: Optional[int] = None,
        use_nearest_upsample: bool = False,
    ):
        super(ConvNeXtStackUpsample, self).__init__()

        if output_dim is None:
            output_dim = input_dim

        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=hidden_dim,
                    kernel_size=2 * stride,
                    stride=1,
                    bias=False,
                    padding="same",
                ),
            )
        else:
            upsample_layer = nn.ConvTranspose1d(
                in_channels=input_dim,
                out_channels=hidden_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )

        self.blocks = nn.Sequential(
            upsample_layer,
            *[ConvNeXtV2Block(hidden_dim) for _ in range(num_blocks)],
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvNeXt stack.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, T).
        """
        return self.blocks(x)


# class ConvNeXtStackDownsampleFlexible(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         num_blocks: int,
#         hidden_dim: int,
#         downsample_ratio: float,
#         output_dim: Optional[int] = None,
#         interpolation_mode: str = "linear",
#         align_corners: bool = False,
#     ):
#         super(ConvNeXtStackDownsampleFlexible, self).__init__()

#         if not 0.0 < downsample_ratio < 1.0:
#             raise ValueError(f"downsample_ratio must be in (0, 1), got {downsample_ratio}")

#         if output_dim is None:
#             output_dim = input_dim

#         self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
#         self.output_proj = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

#         self.blocks = nn.Sequential(*[ConvNeXtV2Block(hidden_dim) for _ in range(num_blocks)])
#         self.downsample_ratio = downsample_ratio
#         self.interpolation_mode = interpolation_mode
#         self.align_corners = align_corners

#     def forward(self, x: torch.Tensor, target_length: Optional[int] = None) -> torch.Tensor:
#         """
#         Forward pass of the flexible ConvNeXt downsampling stack.

#         Args:
#             x (torch.Tensor): Input tensor of shape (B, C, T).
#             target_length (Optional[int]): Explicit output temporal length. If not provided,
#                 uses round(T * downsample_ratio).

#         Returns:
#             torch.Tensor: Output tensor of shape (B, C_out, T_out).
#         """
#         input_length = x.shape[-1]

#         if target_length is None:
#             target_length = max(1, int(round(input_length * self.downsample_ratio)))

#         x = self.input_proj(x)
#         x = self.blocks(x)

#         kwargs = {}
#         if self.interpolation_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
#             kwargs["align_corners"] = self.align_corners
#         x = F.interpolate(x, size=target_length, mode=self.interpolation_mode, **kwargs)
#         return self.output_proj(x)


# class ConvNeXtStackUpsampleFlexible(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         num_blocks: int,
#         hidden_dim: int,
#         output_dim: Optional[int] = None,
#         upsample_ratio: Optional[float] = None,
#         interpolation_mode: str = "linear",
#         align_corners: bool = False,
#     ):
#         super(ConvNeXtStackUpsampleFlexible, self).__init__()

#         if upsample_ratio is not None and upsample_ratio <= 1.0:
#             raise ValueError(f"upsample_ratio must be > 1 when provided, got {upsample_ratio}")

#         if output_dim is None:
#             output_dim = input_dim

#         self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
#         self.output_proj = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

#         self.blocks = nn.Sequential(*[ConvNeXtV2Block(hidden_dim) for _ in range(num_blocks)])
#         self.upsample_ratio = upsample_ratio
#         self.interpolation_mode = interpolation_mode
#         self.align_corners = align_corners

#     def forward(self, x: torch.Tensor, target_length: Optional[int] = None) -> torch.Tensor:
#         """
#         Forward pass of the flexible ConvNeXt upsampling stack.

#         Args:
#             x (torch.Tensor): Input tensor of shape (B, C, T).
#             target_length (Optional[int]): Explicit output temporal length. If not provided,
#                 uses round(T * upsample_ratio).

#         Returns:
#             torch.Tensor: Output tensor of shape (B, C_out, T_out).
#         """
#         input_length = x.shape[-1]
#         if target_length is None:
#             target_length = max(1, int(round(input_length * self.upsample_ratio)))

#         x = self.input_proj(x)

#         kwargs = {}
#         if self.interpolation_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
#             kwargs["align_corners"] = self.align_corners
#         x = F.interpolate(x, size=target_length, mode=self.interpolation_mode, **kwargs)

#         x = self.blocks(x)
#         return self.output_proj(x)
