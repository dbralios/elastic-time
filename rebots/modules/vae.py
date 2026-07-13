import torch
import torch.nn as nn


def vae_sample(mean, scale):
    stdev = nn.functional.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean

    kl = (mean * mean + var - logvar - 1).sum(1).mean()

    return latents, kl


class VAEBottleneck(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, x, return_info=False, **kwargs):
        info = {}

        mean, scale = x.chunk(2, dim=1)

        x, kl = vae_sample(mean, scale, **kwargs)

        info["kl"] = kl

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x

    def forward(self, x, return_info=False, **kwargs):
        return self.encode(x, return_info=return_info, **kwargs)
