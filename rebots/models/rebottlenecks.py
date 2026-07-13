import copy
from typing import Optional, Union

import torch
import torch.nn as nn
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


class ReBottleneck(nn.Module):
    def __init__(
        self,
        encoder: Union[nn.Module, DictConfig, dict] = None,
        decoder: Union[nn.Module, DictConfig, dict] = None,
        bottleneck: Optional[Union[nn.Module, DictConfig, dict]] = None,
        *,
        masking_cfg: Union[DictConfig, dict] = None,
        mean: Optional[list[float]] = None,
        scale: Optional[list[float]] = None,
    ):
        super(ReBottleneck, self).__init__()

        # Instantiate encoder, decoder and bottleneck modules
        self.decoder = _build(decoder, expected="Decoder")
        self.encoder = _build(encoder, expected="Encoder")

        self.bottleneck = None
        if bottleneck is not None:
            self.bottleneck = _build(bottleneck, expected="Bottleneck")

        # Masking modules
        self.masking_module = None
        self.latent_masking_module = None
        if masking_cfg is not None:
            self.masking_module = config.instantiate(DictConfig(masking_cfg))

        # Normalization parameters
        if mean is not None:
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1))
        else:
            self.mean = None
        if scale is not None:
            self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32).view(1, -1, 1))
        else:
            self.scale = None

    def forward(
        self, x: torch.Tensor, *, return_info: Optional[bool] = True, **kwargs
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).
            return_info (bool): Whether to return additional information. Defaults to True.
        Returns:
            torch.Tensor: Output tensor of shape (B, C, T).
            dict: Additional information if return_info is True.

        Notation:
            B: Batch size
            C: Number of channels (input_dim)
            T: Time dimension
        """

        if self.masking_module is not None:
            x, _ = self.masking_module(x)

        z = self.encoder(x)
        info = {}
        if self.bottleneck is not None:
            bottleneck_ret = self.bottleneck(z, return_info=return_info)
            if isinstance(bottleneck_ret, tuple):
                z, bottleneck_info = bottleneck_ret
                info.update(bottleneck_info)
            else:
                z = bottleneck_ret

        if hasattr(self, "latent_probe") and return_info:
            z_probe = self.linear_probe(z)
            info["latent_probe"] = z_probe

        if self.latent_masking_module is not None:
            z = self.latent_masking_module(z)

        x_hat = self.decoder(z)

        if return_info:
            return x_hat, info
        else:
            return x_hat

    def encode(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[bool] = False,
        return_info: Optional[bool] = True,
        **kwargs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        """
        Encode the input tensor using the encoder blocks. To be used only for inference.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).

        Returns:
            torch.Tensor: Encoded tensor of shape (B, bottleneck_dim, T).
            dict: Additional information if return_info is True.
        """
        if self.masking_module is not None and mask:
            x, _ = self.masking_module(x)

        z = self.encoder(x)

        info = {}
        if self.bottleneck is not None:
            z, info = self.bottleneck(z, return_info=True)

        if hasattr(self, "linear_probe"):
            z_probe = self.linear_probe(z)
            info["linear_probe"] = z_probe

        if self.latent_masking_module is not None and mask:
            z = self.latent_masking_module(z)
        elif mask:
            raise ValueError("Masking module is not defined. Please check the configuration.")

        if self.mean is not None:
            z = z - self.mean
        if self.scale is not None:
            z = z / self.scale

        if return_info:
            return z, info
        else:
            return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the input tensor using the decoder blocks. To be used only for inference.

        Args:
            z (torch.Tensor): Input tensor of shape (B, bottleneck_dim, T).

        Returns:
            torch.Tensor: Decoded tensor of shape (B, C, T).
        """
        if self.scale is not None:
            z = z * self.scale
        if self.mean is not None:
            z = z + self.mean

        x_hat = self.decoder(z)
        return x_hat


class ElasticTimeReBottleneck(nn.Module):
    def __init__(
        self,
        encoder: Union[nn.Module, DictConfig, dict] = None,
        decoder: Union[nn.Module, DictConfig, dict] = None,
        predictor: Union[nn.Module, DictConfig, dict] = None,
        bottleneck: Optional[Union[nn.Module, DictConfig, dict]] = None,
        conditioner: Optional[Union[nn.Module, DictConfig, dict]] = None,
        *,
        mean: Optional[list[float]] = None,
        scale: Optional[list[float]] = None,
    ):
        super(ElasticTimeReBottleneck, self).__init__()

        # Instantiate encoder, decoder and bottleneck modules
        self.decoder = _build(decoder, expected="Decoder")
        self.encoder = _build(encoder, expected="Encoder")
        self.predictor = _build(predictor, expected="Predictor")

        self.bottleneck = None
        if bottleneck is not None:
            self.bottleneck = _build(bottleneck, expected="Bottleneck")

        self.conditioner = None
        if conditioner is not None:
            self.conditioner = _build(conditioner, expected="Conditioner")

        # Normalization parameters
        if mean is not None:
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1))
        else:
            self.mean = None
        if scale is not None:
            self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32).view(1, -1, 1))
        else:
            self.scale = None

    def _get_condition(
        self,
        cond_input: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if cond_input is None:
            return None
        if self.conditioner is None:
            raise ValueError("cond_input provided but conditioner is not set.")

        if not torch.is_tensor(cond_input):
            cond_input = torch.tensor(cond_input, device=device, dtype=dtype)
        cond_input = cond_input.to(device=device, dtype=dtype)

        if cond_input.ndim == 0:
            cond_input = cond_input.view(1)
        if cond_input.ndim == 1:
            cond_input = cond_input[:, None, None]
        elif cond_input.ndim == 2:
            cond_input = cond_input[:, :, None]

        cond_input = cond_input.clamp(0.0, 1.0)

        if cond_input.shape[0] != batch_size:
            if cond_input.shape[0] != 1:
                raise ValueError(f"cond_input batch size mismatch: expected {batch_size}, got {cond_input.shape[0]}")
            cond_input = cond_input.expand(batch_size, -1, -1)

        return self.conditioner(cond_input)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_info: Optional[bool] = True,
        cond_input_enc: Optional[torch.Tensor] = None,
        cond_input_dec: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).
            return_info (bool): Whether to return additional information. Defaults to True.
        Returns:
            torch.Tensor: Output tensor of shape (B, C, T).
            dict: Additional information if return_info is True.

        Notation:
            B: Batch size
            C: Number of channels (input_dim)
            T: Time dimension
        """

        cond_input_enc = self._get_condition(cond_input_enc, batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        if cond_input_enc is None:
            z = self.encoder(x)
        else:
            z = self.encoder(x, cond_input_enc)
        info = {}
        if self.bottleneck is not None:
            z, bottleneck_info = self.bottleneck(z, return_info=return_info)
            info.update(bottleneck_info)

        cond_input_dec = self._get_condition(cond_input_dec, batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        if cond_input_dec is None:
            x_hat = self.decoder(z)
        else:
            x_hat = self.decoder(z, cond_input_dec)

        if return_info:
            return x_hat, info
        else:
            return x_hat

    def encode(
        self,
        x: torch.Tensor,
        *,
        return_info: Optional[bool] = True,
        cond_input: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        """
        Encode the input tensor using the encoder blocks. To be used only for inference.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).
        Returns:
            torch.Tensor: Encoded tensor of shape (B, bottleneck_dim, T).
            dict: Additional information if return_info is True.
        """
        cond = self._get_condition(cond_input, batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        if cond is None:
            z = self.encoder(x)
        else:
            z = self.encoder(x, cond)

        info = {}
        if self.bottleneck is not None:
            z, info = self.bottleneck(z, return_info=True)

        if return_info:
            return z, info
        else:
            return z

    def decode(self, z: torch.Tensor, *, cond_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode the input tensor using the decoder blocks. To be used only for inference.

        Args:
            z (torch.Tensor): Input tensor of shape (B, bottleneck_dim, T).
        Returns:
            torch.Tensor: Decoded tensor of shape (B, C, T).
        """
        cond = self._get_condition(cond_input, batch_size=z.shape[0], device=z.device, dtype=z.dtype)
        if cond is None:
            x_hat = self.decoder(z)
        else:
            x_hat = self.decoder(z, cond)
        return x_hat


class HNetChunker(nn.Module):
    def __init__(self, latent_dim: int, query_dim: int):
        super(HNetChunker, self).__init__()

        self.q_layer = nn.Linear(latent_dim, query_dim)
        self.k_layer = nn.Linear(latent_dim, query_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        q = self.q_layer(x.permute(0, 2, 1))  # (B, T, query_dim)
        k = self.k_layer(x.permute(0, 2, 1))  # (B, T, query_dim)

        p_t_out = torch.ones_like(q[:, :, 0])  # (B, T)

        cos_sim = torch.nn.functional.cosine_similarity(q[:, 1:, :], k[:, :-1, :], dim=-1)  # (B, T)
        p_t = 0.5 * (1 - cos_sim)  # (B, T), values in [0, 1]

        p_t_out[:, 1:] = p_t
        return p_t_out


class HNetReBottleneck(nn.Module):
    def __init__(
        self,
        encoder: Union[nn.Module, DictConfig, dict] = None,
        decoder: Union[nn.Module, DictConfig, dict] = None,
        chunker: Union[nn.Module, DictConfig, dict] = None,
        bottleneck: Optional[Union[nn.Module, DictConfig, dict]] = None,
        conditioner: Optional[Union[nn.Module, DictConfig, dict]] = None,
        *,
        mean: Optional[list[float]] = None,
        scale: Optional[list[float]] = None,
    ):
        super(HNetReBottleneck, self).__init__()

        # Instantiate encoder, decoder and bottleneck modules
        self.decoder = _build(decoder, expected="Decoder")
        self.encoder = _build(encoder, expected="Encoder")
        self.chunker = _build(chunker, expected="HNetChunker")

        self.bottleneck = None
        if bottleneck is not None:
            self.bottleneck = _build(bottleneck, expected="Bottleneck")

        self.conditioner = None
        if conditioner is not None:
            self.conditioner = _build(conditioner, expected="Conditioner")

        # Normalization parameters
        if mean is not None:
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1))
        else:
            self.mean = None
        if scale is not None:
            self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32).view(1, -1, 1))
        else:
            self.scale = None

    def _get_condition(
        self,
        cond_input: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if cond_input is None:
            return None
        if self.conditioner is None:
            raise ValueError("cond_input provided but conditioner is not set.")

        if not torch.is_tensor(cond_input):
            cond_input = torch.tensor(cond_input, device=device, dtype=dtype)
        cond_input = cond_input.to(device=device, dtype=dtype)

        if cond_input.ndim == 0:
            cond_input = cond_input.view(1)
        if cond_input.ndim == 1:
            cond_input = cond_input[:, None, None]
        elif cond_input.ndim == 2:
            cond_input = cond_input[:, :, None]

        cond_input = cond_input.clamp(0.0, 1.0)

        if cond_input.shape[0] != batch_size:
            if cond_input.shape[0] != 1:
                raise ValueError(f"cond_input batch size mismatch: expected {batch_size}, got {cond_input.shape[0]}")
            cond_input = cond_input.expand(batch_size, -1, -1)

        return self.conditioner(cond_input)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_info: Optional[bool] = True,
        cond_input_enc: Optional[torch.Tensor] = None,
        cond_input_dec: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).
            return_info (bool): Whether to return additional information. Defaults to True.
        Returns:
            torch.Tensor: Output tensor of shape (B, C, T).
            dict: Additional information if return_info is True.

        Notation:
            B: Batch size
            C: Number of channels (input_dim)
            T: Time dimension
        """

        cond_input_enc = self._get_condition(cond_input_enc, batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        if cond_input_enc is None:
            z = self.encoder(x)
        else:
            z = self.encoder(x, cond_input_enc)
        info = {}
        if self.bottleneck is not None:
            z, bottleneck_info = self.bottleneck(z, return_info=return_info)
            info.update(bottleneck_info)

        p_t = self.chunker(z)
        threhold = kwargs.get("hnet_threshold", 0.5)
        ratio = kwargs.get("hnet_ratio", None)  # Keep ratio
        if ratio is not None:
            threhold = torch.quantile(p_t, 1 - ratio, dim=1, keepdim=True)

        chunk_mask = (p_t > threhold).float()  # (B, T)

        # Confidence
        c_t = p_t * chunk_mask + (1 - p_t) * (1 - chunk_mask)  # (B, T)
        c_t = c_t + (1 - c_t).detach()  # STE

        # EMA mixing
        B, C, T = z.shape

        p_t_sparse = p_t * chunk_mask

        z_bar = torch.empty_like(z)

        # With your guarantee mask[:,0]==1, you can initialize this way
        prev = z[:, :, 0]
        z_bar[:, :, 0] = prev

        for t in range(1, T):
            w = p_t_sparse[:, t].unsqueeze(1)  # (B,1)
            prev = (1.0 - w) * prev + w * z[:, :, t]
            z_bar[:, :, t] = prev
        z = z_bar

        # Keep chunk representatives
        B, C, T = z.shape

        t = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # (B,T)
        marked = torch.where(chunk_mask.bool(), t, torch.full_like(t, -1))  # (B,T)
        idx = torch.cummax(marked, dim=1).values.clamp(min=0)  # (B,T)
        z_out = z.gather(dim=2, index=idx.unsqueeze(1).expand(B, C, T))  # (B,C,T)

        # Multiply by confidence for bwd
        z_out = c_t.unsqueeze(1) * z_out

        cond_input_dec = self._get_condition(
            cond_input_dec, batch_size=z_out.shape[0], device=z_out.device, dtype=z_out.dtype
        )
        if cond_input_dec is None:
            x_hat = self.decoder(z_out)
        else:
            x_hat = self.decoder(z_out, cond_input_dec)

        F = chunk_mask.float().mean(dim=1)
        G = p_t.mean(dim=1)
        info["F"] = F
        info["G"] = G

        if return_info:
            return x_hat, info
        else:
            return x_hat

    def encode(
        self,
        x: torch.Tensor,
        *,
        return_info: Optional[bool] = True,
        cond_input: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict]]:
        """
        Encode the input tensor using the encoder blocks. To be used only for inference.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T).
        Returns:
            torch.Tensor: Encoded tensor of shape (B, bottleneck_dim, T).
            dict: Additional information if return_info is True.
        """
        cond = self._get_condition(cond_input, batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        if cond is None:
            z = self.encoder(x)
        else:
            z = self.encoder(x, cond)

        info = {}
        if self.bottleneck is not None:
            z, info = self.bottleneck(z, return_info=True)

        if return_info:
            return z, info
        else:
            return z

    def decode(self, z: torch.Tensor, *, cond_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode the input tensor using the decoder blocks. To be used only for inference.

        Args:
            z (torch.Tensor): Input tensor of shape (B, bottleneck_dim, T).
        Returns:
            torch.Tensor: Decoded tensor of shape (B, C, T).
        """
        cond = self._get_condition(cond_input, batch_size=z.shape[0], device=z.device, dtype=z.dtype)
        if cond is None:
            x_hat = self.decoder(z)
        else:
            x_hat = self.decoder(z, cond)
        return x_hat
