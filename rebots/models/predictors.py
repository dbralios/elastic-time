import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward block with optional RMSNorm and ReZero alpha.

    Shapes:
      - x: (N, C) -> (N, C)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        bias: bool = True,
        dropout: float = 0.0,
        use_rmsnorm: bool = True,
        rezero_alpha: bool = False,
        rezero_alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.use_rmsnorm = bool(use_rmsnorm)
        self.rezero_alpha = bool(rezero_alpha)

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")

        self.norm = nn.RMSNorm(self.dim) if self.use_rmsnorm else nn.Identity()
        self.w1a = nn.Linear(self.dim, self.hidden_dim, bias=bias)
        self.w1b = nn.Linear(self.dim, self.hidden_dim, bias=bias)
        self.w2 = nn.Linear(self.hidden_dim, self.dim, bias=bias)
        self.drop = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

        if self.rezero_alpha:
            self.alpha = nn.Parameter(torch.tensor(rezero_alpha_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        gated = F.silu(self.w1a(x_norm)) * self.w1b(x_norm)
        delta = self.drop(self.w2(gated))
        if self.rezero_alpha:
            return x + self.alpha * delta
        return x + delta


class ElasticTimeAutoregressiveGRU(nn.Module):
    """
    Single-step autoregressive predictor built from stacked GRUCell layers.

    For each forward call, we perform one recurrent update. The temporal axis `T`
    is treated as batch-like, so each timestep is updated independently within the
    same call.

    Update rule:
        x_0 = in_proj(z)
        h_0_next = GRUCell_0(x_0, h_0)
        h_l_next = GRUCell_l(h_{l-1}_next, h_l), for l = 1..L-1
        z_next = out_proj(h_{L-1}_next)

    Shapes:
      - z: (B, C, T) or (C, T)
      - h: (B, L, H, T) or (L, H, T), optional
      - out: same shape as z
      - h_next: same shape as h
    """

    def __init__(
        self,
        hidden_dim: int,
        input_dim: int,
        num_layers: int = 2,
        bias: bool = True,
        learned_h_initial: bool = True,
        out_swiglu_ffn: bool = False,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.input_dim = int(input_dim)
        self.num_layers = int(num_layers)

        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        # Projections
        self.in_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=bias)
        self.out_proj = nn.Linear(self.hidden_dim, self.input_dim, bias=bias)
        if out_swiglu_ffn:
            self.out_proj = nn.Sequential(
                SwiGLUFFN(dim=self.hidden_dim, hidden_dim=self.hidden_dim * 4, bias=bias),
                self.out_proj,
            )

        self.cells = nn.ModuleList(
            [
                nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim, bias=bias)
                for _ in range(self.num_layers)
            ]
        )

        if learned_h_initial:
            self.h_initial = nn.Parameter(torch.zeros(self.hidden_dim))
        else:
            self.register_buffer("h_initial", torch.zeros(self.hidden_dim), persistent=False)

    def initial_hidden(
        self,
        batch_size: int,
        time_steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create initial hidden state cache with shape (B, L, H, T)."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if time_steps < 1:
            raise ValueError(f"time_steps must be >= 1, got {time_steps}")

        return (
            self.h_initial.view(1, 1, self.hidden_dim, 1)
            .expand(batch_size, self.num_layers, self.hidden_dim, time_steps)
            .to(device=device, dtype=dtype)
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one autoregressive prediction step.

        Args:
            z (torch.Tensor): Current latent tensor of shape (B, C, T) or
                (C, T), where C == input_dim.
            h (torch.Tensor | None, optional): Previous hidden state of shape
                (B, L, H, T) or (L, H, T), where L == num_layers and
                H == hidden_dim. If None, a broadcasted `h_initial` is used.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - out: Predicted next latent with same shape as z.
                - h_next: Updated hidden state with shape (B, L, H, T) or
                  (L, H, T), matching batched/single input form.
        """

        single = False
        if z.ndim == 2:
            z = z.unsqueeze(0)
            h = h.unsqueeze(0) if h is not None else None
            single = True
        elif z.ndim != 3:
            raise ValueError(f"z must be (B,C,T) or (C,T); got {tuple(z.shape)}")

        B, Cin, T = z.shape
        if Cin != self.input_dim:
            raise ValueError(f"Expected input C={self.input_dim}, got C={Cin}")

        if h is not None and h.shape != (B, self.num_layers, self.hidden_dim, T):
            raise ValueError(f"h must be (B,L,H,T)=({B},{self.num_layers},{self.hidden_dim},{T}), got {tuple(h.shape)}")

        # Treat T as batch-like: reshape (B, Cin, T) -> (B*T, Cin)
        z = z.permute(0, 2, 1).contiguous().view(B * T, self.input_dim)  # (B*T, Cin)
        if h is None:
            h_tensor = self.initial_hidden(B, T, device=z.device, dtype=z.dtype)
        else:
            h_tensor = h
        h_bt = h_tensor.permute(0, 3, 1, 2).contiguous().view(B * T, self.num_layers, self.hidden_dim)

        # Project into hidden dim
        z_proj = self.in_proj(z)  # (B*T, dim)

        h_next = []

        # Apply stacked GRUCells at each timestep independently
        x_l = z_proj
        for l, cell in enumerate(self.cells):
            h_l = cell(x_l, h_bt[:, l, :])
            h_next.append(h_l)
            x_l = h_l

        h_next = torch.stack(h_next, dim=1)  # (B*T, num_layers, hidden_dim)

        # Project back to input dim (if needed) and restore shape
        y = self.out_proj(x_l)  # (B*T, Cin)
        out = y.view(B, T, Cin).permute(0, 2, 1).contiguous()  # (B, Cin, T)
        h_next = (
            h_next.view(B, T, self.num_layers, self.hidden_dim).permute(0, 2, 3, 1).contiguous()
        )  # (B, num_layers, hidden_dim, T)

        return out.squeeze(0) if single else out, h_next.squeeze(0) if single else h_next
