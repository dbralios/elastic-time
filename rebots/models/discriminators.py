import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


def WNConv1d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv1d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


class LatentDiscriminator(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        """
        Latent discriminator.
        Args:
            hidden_dim (int): hidden dimension
        """
        super().__init__()

        self.h_dim = hidden_dim

        self.conv_stack = nn.ModuleList(
            [
                WNConv2d(1, self.h_dim, (3, 3), (1, 1), padding=(1, 1)),
                WNConv2d(self.h_dim, self.h_dim, (3, 7), (1, 1), padding=(1, 1)),
                WNConv2d(self.h_dim, self.h_dim, (3, 7), (1, 1), padding=(1, 1)),
                WNConv2d(self.h_dim, self.h_dim, (3, 7), (1, 1), padding=(1, 1)),
                WNConv2d(self.h_dim, self.h_dim, (3, 7), (1, 1), padding=(1, 1)),
                WNConv2d(self.h_dim, 1, (3, 3), (1, 1), padding=(1, 1), act=False),
            ]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x (Tensor): (B, C, T)
        Returns:
            tuple[Tensor, list[Tensor]]: output tensor and list of feature maps
        """
        x = x.unsqueeze(1)

        fmap = []

        for layer in self.conv_stack:
            x = layer(x)
            fmap.append(x)

        return x, fmap[:-1]


if __name__ == "__main__":
    disc = LatentDiscriminator()
    x = torch.zeros(1, 64, 430)
    results = disc(x)
    for i, result in enumerate(results):
        print(f"disc{i}")
        for i, r in enumerate(result):
            print(r.shape, r.mean(), r.min(), r.max())
        print()

    # Print model size
    print(f"Model size: {sum(p.numel() for p in disc.parameters() if p.requires_grad)}")
    x = disc.parameters()
    print(x)
