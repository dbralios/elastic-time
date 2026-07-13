import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    """
    Computes the loss for both the discriminator and the generator in separate functions.
    """

    def __init__(self):
        super().__init__()

    def discriminator_loss(self, d_fake: torch.Tensor, d_real: torch.Tensor) -> torch.Tensor:
        """
        Computes the discriminator loss.
        Args:
            d_fake (torch.Tensor): (B, 1, C, T)
            d_real (torch.Tensor): (B, 1, C, T)
        Returns:
            loss_d (torch.Tensor): Discriminator loss
        """
        loss_d = 0.0
        real_loss = torch.mean((1 - d_real) ** 2)
        fake_loss = torch.mean(d_fake**2)
        loss_d = real_loss + fake_loss
        return loss_d

    def generator_loss(self, d_fake: torch.Tensor, d_real: torch.Tensor) -> torch.Tensor:
        """
        Computes the generator loss.
        Args:
            d_fake (torch.Tensor): (B, 1, C, T)
        Returns:
            loss_g (torch.Tensor): Generator loss
        """
        loss_g = 0.0
        loss_g = torch.mean((1 - d_fake) ** 2)
        return loss_g

    def feature_matching_loss(self, fm_fake: list[torch.Tensor], fm_real: list[torch.Tensor]) -> torch.Tensor:
        """
        Computes the feature matching loss.
        Args:
            fm_fake (List[torch.Tensor]):
            fm_real (List[torch.Tensor]):
        Returns:
            loss_feature (torch.Tensor): Feature matching loss
        """
        loss_feature = 0.0
        for i in range(len(fm_fake)):
            f = fm_fake[i]
            r = fm_real[i].detach()
            loss_feature += nn.functional.l1_loss(f, r) / r.abs().mean().clamp(min=1e-6)

        return loss_feature
