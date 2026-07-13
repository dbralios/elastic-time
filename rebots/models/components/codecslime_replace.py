from __future__ import annotations

from typing import Optional

import torch


def _normalize_target_segments(
    target_segments: torch.Tensor | int,
    *,
    batch_size: int,
    time_steps: int,
    max_segment_length: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(target_segments, int):
        target = torch.full((batch_size,), target_segments, device=device, dtype=torch.long)
    elif torch.is_tensor(target_segments):
        target = target_segments.to(device=device, dtype=torch.long)
        if target.ndim == 0:
            target = target.view(1)
        if target.shape[0] == 1 and batch_size > 1:
            target = target.expand(batch_size)
        if target.shape[0] != batch_size:
            raise ValueError(f"target_segments batch mismatch: expected {batch_size}, got {target.shape[0]}")
    else:
        raise TypeError("target_segments must be an int or torch.Tensor")

    min_segments = (time_steps + max_segment_length - 1) // max_segment_length
    if (target < min_segments).any() or (target > time_steps).any():
        raise ValueError(
            f"target_segments must be in [{min_segments}, {time_steps}] for T={time_steps}, "
            f"max_segment_length={max_segment_length}"
        )
    return target


def _validate_segment_offsets(h: torch.Tensor) -> None:
    if h.ndim != 2:
        raise ValueError(f"h must be (B, T), got {tuple(h.shape)}")
    if h.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("h must be an integer tensor")

    _, time_steps = h.shape
    t = torch.arange(time_steps, device=h.device, dtype=torch.long).view(1, time_steps)
    h_long = h.to(dtype=torch.long)
    if (h_long < 0).any() or (h_long > t).any():
        raise ValueError("h must satisfy 0 <= h[:, t] <= t")

    if not torch.all(h_long[:, 0] == 0):
        raise ValueError("h must satisfy h[:, 0] == 0")

    starts = h_long == 0
    start_positions = torch.where(starts, t, torch.full_like(t, -1))
    last_start = start_positions.cummax(dim=1).values
    expected = t - last_start
    if not torch.equal(h_long, expected):
        raise ValueError("h must encode offsets from the most recent segment start")


def mix_with_codecslime_segments(z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    Replace each contiguous segment in ``z`` with that segment's mean.

    Args:
        z: Latents of shape ``(B, C, T)``.
        h: Segment-offset matrix of shape ``(B, T)`` with strict semantics:
            - ``h[b, 0] == 0``
            - ``h[b, t]`` equals the offset from the most recent segment start
              (the last index ``s <= t`` where ``h[b, s] == 0``)
            - starts are therefore exactly the positions where ``h == 0``.

            The function validates this structure before mixing.

    Returns:
        Mixed latents of shape ``(B, C, T)`` where every timestep in a segment
        is replaced by the mean latent over that segment.

    Notes:
        Implementation detail: segment start indices are reconstructed from
        ``h == 0`` using a cumulative max, then segment sums/counts are computed
        with ``scatter_add_``, and means are expanded back with ``gather``.
    """
    if z.ndim != 3:
        raise ValueError(f"z must be (B, C, T), got {tuple(z.shape)}")
    _validate_segment_offsets(h)

    batch_size, channels, time_steps = z.shape
    if h.shape != (batch_size, time_steps):
        raise ValueError(f"h shape mismatch: expected {(batch_size, time_steps)}, got {tuple(h.shape)}")

    h_long = h.to(device=z.device, dtype=torch.long)
    t = torch.arange(time_steps, device=z.device, dtype=torch.long).view(1, time_steps)
    starts = h_long == 0
    start_positions = torch.where(starts, t, torch.full_like(t, -1))
    start_idx = start_positions.cummax(dim=1).values

    segment_sum = torch.zeros(batch_size, channels, time_steps, device=z.device, dtype=z.dtype)
    segment_sum.scatter_add_(2, start_idx.unsqueeze(1).expand(-1, channels, -1), z)

    segment_count = torch.zeros(batch_size, 1, time_steps, device=z.device, dtype=z.dtype)
    ones = torch.ones(batch_size, 1, time_steps, device=z.device, dtype=z.dtype)
    segment_count.scatter_add_(2, start_idx.unsqueeze(1), ones)

    segment_mean = segment_sum / segment_count.clamp_min(1.0)
    return segment_mean.gather(2, start_idx.unsqueeze(1).expand(-1, channels, -1))


@torch.no_grad()
def random_codecslime_h(
    target_segments: torch.Tensor | int,
    *,
    time_steps: int,
    max_segment_length: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample random segment-offset matrices with bounded segment lengths.

    Args:
        target_segments: Number of kept segments per sample.
        time_steps: Sequence length ``T``.
        max_segment_length: Maximum segment length ``U``.
        device: Target device for the returned tensor.
        generator: Optional random generator.

    Returns:
        Segment offsets ``h`` of shape ``(B, T)``.
    """
    if time_steps < 1:
        raise ValueError(f"time_steps must be >= 1, got {time_steps}")
    if max_segment_length < 1:
        raise ValueError(f"max_segment_length must be >= 1, got {max_segment_length}")

    if isinstance(target_segments, int):
        batch_size = 1
    elif torch.is_tensor(target_segments):
        batch_size = int(target_segments.numel()) if target_segments.ndim == 0 else target_segments.shape[0]
    else:
        raise TypeError("target_segments must be an int or torch.Tensor")

    target = _normalize_target_segments(
        target_segments,
        batch_size=batch_size,
        time_steps=time_steps,
        max_segment_length=max_segment_length,
        device=device,
    )

    h = torch.zeros(batch_size, time_steps, device=device, dtype=torch.long)
    for b in range(batch_size):
        num_segments = int(target[b].item())
        remaining = time_steps
        cursor = 0
        for i in range(num_segments):
            segments_left = num_segments - i
            min_len = max(1, remaining - max_segment_length * (segments_left - 1))
            max_len = min(max_segment_length, remaining - (segments_left - 1))
            seg_len = int(torch.randint(min_len, max_len + 1, (1,), device=device, generator=generator).item())

            h[b, cursor : cursor + seg_len] = torch.arange(seg_len, device=device, dtype=torch.long)
            cursor += seg_len
            remaining -= seg_len

        if cursor != time_steps or remaining != 0:
            raise RuntimeError("Random segmentation construction failed")

    return h


def _segment_cost_table(z: torch.Tensor, max_segment_length: int, distance_mode: str) -> torch.Tensor:
    """
    Precompute segment distortion costs.

    Returns:
        Tensor of shape ``(U+1, B, T+1)`` where entry ``[s, b, j]`` stores
        segment cost for length ``s`` ending at index ``j`` (1-based in DP).
    """
    if distance_mode not in {"l2", "l2_squared"}:
        raise ValueError(f"Unsupported distance_mode: {distance_mode}")

    batch_size, _, time_steps = z.shape
    max_seg = min(max_segment_length, time_steps)
    cost = torch.full((max_seg + 1, batch_size, time_steps + 1), float("inf"), device=z.device, dtype=torch.float32)

    z_float = z.float()
    for seg_len in range(1, max_seg + 1):
        windows = z_float.unfold(2, seg_len, 1).permute(0, 2, 3, 1)
        means = windows.mean(dim=2, keepdim=True)
        diffs = windows - means

        if distance_mode == "l2":
            window_cost = torch.linalg.vector_norm(diffs, ord=2, dim=-1).sum(dim=-1)
        else:
            window_cost = diffs.square().sum(dim=-1).sum(dim=-1)

        cost[seg_len, :, seg_len:] = window_cost

    return cost


@torch.no_grad()
def codecslime_dp_h(
    z: torch.Tensor,
    target_segments: torch.Tensor | int,
    *,
    max_segment_length: int,
    distance_mode: str = "l2",
    return_cost: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Compute optimal Codec-Slime segment offsets with dynamic programming.

    This solves the constrained segmentation problem from Codec-Slime with
    objective based on segment mean distortion.

    Args:
        z: Latents of shape ``(B, C, T)``.
        target_segments: Number of output segments ``T'`` per sample.
        max_segment_length: Maximum segment length ``U``.
        distance_mode: ``"l2"`` or ``"l2_squared"`` surrogate.
        return_cost: If True, also return best DP cost for each sample.

    Returns:
        ``h`` of shape ``(B, T)``, and optionally DP costs of shape ``(B,)``.
    """
    if z.ndim != 3:
        raise ValueError(f"z must be (B, C, T), got {tuple(z.shape)}")
    if max_segment_length < 1:
        raise ValueError(f"max_segment_length must be >= 1, got {max_segment_length}")

    batch_size, _, time_steps = z.shape
    target = _normalize_target_segments(
        target_segments,
        batch_size=batch_size,
        time_steps=time_steps,
        max_segment_length=max_segment_length,
        device=z.device,
    )

    max_target_segments = int(target.max().item())
    max_seg = min(max_segment_length, time_steps)

    segment_cost = _segment_cost_table(z, max_segment_length=max_seg, distance_mode=distance_mode)

    dp = torch.full(
        (max_target_segments + 1, batch_size, time_steps + 1),
        float("inf"),
        device=z.device,
        dtype=torch.float32,
    )
    back = torch.zeros(
        (max_target_segments + 1, batch_size, time_steps + 1),
        device=z.device,
        dtype=torch.int16,
    )
    dp[0, :, 0] = 0.0

    for i in range(1, max_target_segments + 1):
        best = torch.full((batch_size, time_steps + 1), float("inf"), device=z.device, dtype=torch.float32)
        best_seg = torch.zeros((batch_size, time_steps + 1), device=z.device, dtype=torch.long)

        for seg_len in range(1, max_seg + 1):
            prev = dp[i - 1, :, : time_steps + 1 - seg_len]
            cur_seg_cost = segment_cost[seg_len, :, seg_len:]
            candidate = prev + cur_seg_cost

            improved = candidate < best[:, seg_len:]
            best[:, seg_len:] = torch.where(improved, candidate, best[:, seg_len:])
            seg_len_tensor = torch.full_like(best_seg[:, seg_len:], seg_len)
            best_seg[:, seg_len:] = torch.where(improved, seg_len_tensor, best_seg[:, seg_len:])

        dp[i] = best
        back[i] = best_seg.to(torch.int16)

    batch_idx = torch.arange(batch_size, device=z.device)
    best_cost = dp[target, batch_idx, time_steps]
    if not torch.isfinite(best_cost).all():
        raise RuntimeError("No feasible DP solution for one or more samples")

    h = torch.zeros(batch_size, time_steps, device=z.device, dtype=torch.long)
    for b in range(batch_size):
        segments_left = int(target[b].item())
        end = time_steps

        while segments_left > 0:
            seg_len = int(back[segments_left, b, end].item())
            if seg_len <= 0:
                raise RuntimeError("Backtracking failed: invalid segment length")

            start = end - seg_len
            h[b, start:end] = torch.arange(seg_len, device=z.device, dtype=torch.long)
            end = start
            segments_left -= 1

        if end != 0:
            raise RuntimeError("Backtracking failed: segmentation does not cover full sequence")

    if return_cost:
        return h, best_cost
    return h
