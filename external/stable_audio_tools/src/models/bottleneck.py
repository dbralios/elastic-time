import random

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    def __init__(self, is_discrete: bool = False):
        super().__init__()

        self.is_discrete = is_discrete

    def encode(self, x, return_info=False, **kwargs):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError


class DiscreteBottleneck(Bottleneck):
    def __init__(self, num_quantizers, codebook_size, tokens_id):
        super().__init__(is_discrete=True)

        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.tokens_id = tokens_id

    def decode_tokens(self, codes, **kwargs):
        raise NotImplementedError


class TanhBottleneck(Bottleneck):
    def __init__(self, scale=1.0):
        super().__init__(is_discrete=False)
        self.tanh = nn.Tanh()

        self.scale = scale

    def encode(self, x, return_info=False):
        info = {}

        x = x / self.scale

        x = torch.tanh(x)

        x = x * self.scale

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x


def vae_sample(mean, scale):
    stdev = nn.functional.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean

    kl = (mean * mean + var - logvar - 1).sum(1).mean()

    return latents, kl


class VAEBottleneck(Bottleneck):
    def __init__(self):
        super().__init__(is_discrete=False)

    def encode(self, x, return_info=False, **kwargs):
        info = {}

        mean, scale = x.chunk(2, dim=1)

        x, kl = vae_sample(mean, scale)

        info["kl"] = kl

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x


def compute_mean_kernel(x, y):
    kernel_input = (x[:, None] - y[None]).pow(2).mean(2) / x.shape[-1]
    return torch.exp(-kernel_input).mean()


def compute_mmd(latents):
    latents_reshaped = latents.permute(0, 2, 1).reshape(-1, latents.shape[1])
    noise = torch.randn_like(latents_reshaped)

    latents_kernel = compute_mean_kernel(latents_reshaped, latents_reshaped)
    noise_kernel = compute_mean_kernel(noise, noise)
    latents_noise_kernel = compute_mean_kernel(latents_reshaped, noise)

    mmd = latents_kernel + noise_kernel - 2 * latents_noise_kernel
    return mmd.mean()


class WassersteinBottleneck(Bottleneck):
    def __init__(
        self,
        noise_augment_dim: int = 0,
        bypass_mmd: bool = False,
        use_tanh: bool = False,
        tanh_scale: float = 5.0,
    ):
        super().__init__(is_discrete=False)

        self.noise_augment_dim = noise_augment_dim
        self.bypass_mmd = bypass_mmd
        self.use_tanh = use_tanh
        self.tanh_scale = tanh_scale

    def encode(self, x, return_info=False):
        info = {}

        if self.training and return_info:
            if self.bypass_mmd:
                mmd = torch.tensor(0.0)
            else:
                mmd = compute_mmd(x)

            info["mmd"] = mmd

        if self.use_tanh:
            x = torch.tanh(x / self.tanh_scale) * self.tanh_scale

        if return_info:
            return x, info

        return x

    def decode(self, x):

        if self.noise_augment_dim > 0:
            noise = torch.randn(x.shape[0], self.noise_augment_dim, x.shape[-1]).type_as(x)
            x = torch.cat([x, noise], dim=1)

        return x


class L2Bottleneck(Bottleneck):
    def __init__(self):
        super().__init__(is_discrete=False)

    def encode(self, x, return_info=False):
        info = {}

        x = F.normalize(x, dim=1)

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return F.normalize(x, dim=1)

    def __init__(
        self,
        dim,
        levels,
        num_codebooks=1,
        dither_inference=True,
        noise_dropout: float = 0.05,
    ):
        from .fsq import DitheredFSQ

        # Determine codebook size and levels configuration based on the type of 'levels'
        if isinstance(levels, int):
            codebook_size = levels**dim
            quantizer_levels = [levels] * dim

        elif isinstance(levels, list):
            if len(levels) != dim:
                raise ValueError(f"Length of levels list ({len(levels)}) must match dim ({dim}).")
            codebook_size = 1
            for level in levels:
                codebook_size *= level
            quantizer_levels = levels
        else:
            raise TypeError("Levels must be either an int or a list of ints.")

        # Initialize parent class with the determined codebook size
        super().__init__(
            num_quantizers=num_codebooks,
            codebook_size=codebook_size,
            tokens_id="quantizer_indices",
        )

        # Initialize the quantizer with the correct levels
        self.quantizer = DitheredFSQ(
            levels=quantizer_levels,
            dither_inference=dither_inference,
            num_codebooks=num_codebooks,
            noise_dropout=noise_dropout,
        )

    def norm_std_loss(self, x):
        return (x.std() - 1.0) ** 2

    def encode(self, x, return_info=False):
        info = {}

        x = rearrange(x, "b c n -> b n c")
        x, indices = self.quantizer(x)
        x = rearrange(x, "b n c -> b c n")

        info["quantizer_indices"] = indices

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x

    def decode_tokens(self, tokens, **kwargs):
        latents = self.quantizer.indices_to_codes(tokens)

        return self.decode(latents, **kwargs)
