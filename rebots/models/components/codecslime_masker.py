from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

import torch

from rebots.models.components.codecslime_replace import codecslime_dp_h


@dataclass
class StrategySpec:
    name: Literal["random", "dp", "keep"]
    weight: float
    params: Optional[dict[str, Any]] = None


@dataclass
class CodecSlimeMaskSample:
    h_prefilled: torch.Tensor
    dp_mask: torch.Tensor
    dp_target_segments: torch.Tensor
    target_replace_fraction: torch.Tensor
    strategy_indices: torch.Tensor


class CodecSlimeMasker:
    """
    Sample and build segment-offset masks for CodecSlime style averaging.

    ``h`` has shape ``(B, T)`` with entries ``h[b, t] = t - segment_start``.
    """

    def __init__(
        self,
        max_segment_length: int,
        strategies: Iterable[StrategySpec | dict[str, Any]],
        *,
        distance_mode: str = "l2",
    ) -> None:
        if max_segment_length < 1:
            raise ValueError(f"max_segment_length must be >= 1, got {max_segment_length}")

        self.max_segment_length = max_segment_length
        self.distance_mode = distance_mode
        self._strategies = self._normalize_strategies(strategies)
        self._validate_strategies(self._strategies)

    def sample(
        self,
        batch_size: int,
        time_steps: int,
        device: torch.device,
        *,
        global_step: Optional[int] = None,
    ) -> CodecSlimeMaskSample:
        """
        Sample masking plans without inspecting latent values.

        Random/keep strategies directly produce masks, while DP rows are deferred
        until ``replace`` when latents are available.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if time_steps < 1:
            raise ValueError(f"time_steps must be >= 1, got {time_steps}")

        weights = torch.tensor([spec.weight for spec in self._strategies], device=device)
        probs = weights / weights.sum()
        strategy_indices = torch.multinomial(probs, num_samples=batch_size, replacement=True)

        h_prefilled = torch.zeros(batch_size, time_steps, device=device, dtype=torch.long)
        dp_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        dp_target_segments = torch.full((batch_size,), time_steps, device=device, dtype=torch.long)
        target_replace_fraction = torch.zeros(batch_size, device=device, dtype=torch.float32)

        for idx, spec in enumerate(self._strategies):
            row_mask = strategy_indices == idx
            if not row_mask.any():
                continue

            cur_batch = int(row_mask.sum().item())
            params = spec.params or {}

            if spec.name == "random":
                mode = params.get("mode", "melt_manager")
                if mode != "melt_manager":
                    raise ValueError(f"Unsupported random strategy mode: {mode}")

                h_random, target_segments = self._sample_random_h_melt_manager(
                    cur_batch,
                    time_steps,
                    device,
                    params,
                    global_step=global_step,
                )

                h_prefilled[row_mask] = h_random
                target_replace_fraction[row_mask] = (time_steps - target_segments).float() / float(time_steps)
            elif spec.name == "dp":
                dp_mask[row_mask] = True
                target_segments = self._sample_target_segments(cur_batch, time_steps, device, params)
                dp_target_segments[row_mask] = target_segments
                target_replace_fraction[row_mask] = (time_steps - target_segments).float() / float(time_steps)
            elif spec.name == "keep":
                h_prefilled[row_mask] = 0
                target_replace_fraction[row_mask] = 0.0
            else:
                raise ValueError(f"Unknown strategy: {spec.name}")

        return CodecSlimeMaskSample(
            h_prefilled=h_prefilled,
            dp_mask=dp_mask,
            dp_target_segments=dp_target_segments,
            target_replace_fraction=target_replace_fraction,
            strategy_indices=strategy_indices,
        )

    def replace(self, z: torch.Tensor, sample: CodecSlimeMaskSample) -> torch.Tensor:
        """
        Build final segment offsets from a sampled plan.

        DP rows are optimized on-the-fly from latent values.
        """
        if z.ndim != 3:
            raise ValueError(f"z must be (B, C, T), got {tuple(z.shape)}")

        batch_size, _, time_steps = z.shape
        if sample.h_prefilled.shape != (batch_size, time_steps):
            raise ValueError(
                f"sample.h_prefilled shape mismatch: expected {(batch_size, time_steps)}, got {tuple(sample.h_prefilled.shape)}"
            )
        if sample.dp_mask.shape != (batch_size,):
            raise ValueError(
                f"sample.dp_mask shape mismatch: expected {(batch_size,)}, got {tuple(sample.dp_mask.shape)}"
            )
        if sample.dp_target_segments.shape != (batch_size,):
            raise ValueError(
                "sample.dp_target_segments shape mismatch: "
                f"expected {(batch_size,)}, got {tuple(sample.dp_target_segments.shape)}"
            )

        h = sample.h_prefilled.clone()
        if sample.dp_mask.any():
            h_dp = codecslime_dp_h(
                z[sample.dp_mask],
                sample.dp_target_segments[sample.dp_mask],
                max_segment_length=self.max_segment_length,
                distance_mode=self.distance_mode,
            )
            h[sample.dp_mask] = h_dp

        return h

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """Backward-compatible one-shot API."""
        batch_size, _, time_steps = z.shape
        sample = self.sample(batch_size, time_steps, z.device)
        return self.replace(z, sample)

    def _sample_random_h_melt_manager(
        self,
        batch_size: int,
        time_steps: int,
        device: torch.device,
        params: dict[str, Any],
        *,
        global_step: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, time_steps, device=device, dtype=torch.long)
        target_segments = torch.full((batch_size,), time_steps, device=device, dtype=torch.long)

        for b in range(batch_size):
            proportions = self._sample_melt_proportions(
                params,
                device=device,
                global_step=global_step,
            )
            if proportions is None:
                continue

            h_row = self._sample_h_from_length_proportions(
                proportions,
                time_steps=time_steps,
                device=device,
            )
            h[b] = h_row
            target_segments[b] = int((h_row == 0).sum().item())

        return h, target_segments

    def _sample_melt_proportions(
        self,
        params: dict[str, Any],
        *,
        device: torch.device,
        global_step: Optional[int],
    ) -> Optional[torch.Tensor]:
        step = max(int(global_step) if global_step is not None else 0, 0)
        skip_probability = float(params.get("skip_probability", 0.5))
        if torch.rand((), device=device).item() < skip_probability:
            return None

        target_mix = torch.tensor(params["target_mix"], device=device, dtype=torch.float32)
        target_mix = target_mix / target_mix.sum().clamp_min(1e-8)

        target_steps = float(params.get("target_steps", 100_000.0))
        concentration = float(params.get("concentration", 30.0))
        epsilon = float(params.get("epsilon", 1.0e-6))

        progress = min(step / max(target_steps, 1.0), 1.0)
        keep_mix = torch.zeros_like(target_mix)
        keep_mix[0] = 1.0

        d = (1.0 - progress) * keep_mix + progress * target_mix
        d = d.clamp_min(epsilon)
        d = d / d.sum().clamp_min(epsilon)

        cooling = max(1.0, step / max(target_steps, 1.0)) ** 2.5
        alpha = d * concentration / cooling
        alpha = alpha.clamp_min(epsilon)

        return torch.distributions.Dirichlet(alpha).sample()

    def _sample_h_from_length_proportions(
        self,
        proportions: torch.Tensor,
        *,
        time_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        T = int(time_steps)
        U = min(int(proportions.numel()), int(self.max_segment_length), T)

        p = proportions[:U].to(device=device, dtype=torch.float32).clamp_min(1e-8)
        p = p / p.sum().clamp_min(1e-8)

        lens = torch.arange(1, U + 1, device=device, dtype=torch.long)  # 1..U

        # c so that sum_k k*(c p_k) = T in expectation
        mu = (p * lens.float()).sum().clamp_min(1e-8)
        c = float(T) / float(mu)

        # ceil counts => total length >= T
        counts = torch.ceil(c * p).to(torch.long).clamp_min(0)

        # build + shuffle segment-length list
        segs = torch.repeat_interleave(lens, counts)  # (N,)
        segs = segs[torch.randperm(segs.numel(), device=device)]

        # fill h, truncating the last segment to hit T exactly
        h = torch.zeros(T, device=device, dtype=torch.long)
        cursor = 0
        for L in segs.tolist():
            if cursor >= T:
                break
            L_eff = min(int(L), T - cursor)  # "trim"
            h[cursor : cursor + L_eff] = torch.arange(L_eff, device=device, dtype=torch.long)
            cursor += L_eff

        # cursor will always reach T because total length(segs) >= T
        return h

    def _sample_target_segments(
        self,
        batch_size: int,
        time_steps: int,
        device: torch.device,
        params: dict[str, Any],
    ) -> torch.Tensor:
        range_cfg = params.get("target_length_range", None)
        if range_cfg is None:
            raise ValueError("strategy requires params.target_length_range")

        if not isinstance(range_cfg, (list, tuple)) or len(range_cfg) != 2:
            raise ValueError("target_length_range must be a 2-tuple or list")

        low, high = float(range_cfg[0]), float(range_cfg[1])
        if not (0.0 < low <= high <= 1.0):
            raise ValueError("target_length_range must satisfy 0 < low <= high <= 1")

        min_segments = (time_steps + self.max_segment_length - 1) // self.max_segment_length
        low_idx = max(min_segments, int(time_steps * low))
        high_idx = min(time_steps, int(time_steps * high))
        if high_idx < low_idx:
            high_idx = low_idx

        return torch.randint(low_idx, high_idx + 1, (batch_size,), device=device)

    def _normalize_strategies(self, strategies: Iterable[StrategySpec | dict[str, Any]]) -> list[StrategySpec]:
        normalized: list[StrategySpec] = []
        for spec in strategies:
            if isinstance(spec, StrategySpec):
                normalized.append(spec)
            elif isinstance(spec, dict):
                normalized.append(
                    StrategySpec(
                        name=spec["name"],
                        weight=spec["weight"],
                        params=spec.get("params", None),
                    )
                )
            else:
                raise ValueError("Each strategy must be a StrategySpec or a dict")
        return normalized

    def _validate_strategies(self, strategies: list[StrategySpec]) -> None:
        if not strategies:
            raise ValueError("strategies must be a non-empty list")

        if sum(spec.weight for spec in strategies) <= 0:
            raise ValueError("strategy weights must sum to a positive value")

        for spec in strategies:
            if spec.weight < 0:
                raise ValueError("strategy weight must be >= 0")

            if spec.name == "random":
                params = spec.params or {}
                mode = params.get("mode", "melt_manager")
                if mode != "melt_manager":
                    raise ValueError(f"Unsupported random strategy mode: {mode}")

                target_mix = params.get("target_mix", None)
                if target_mix is None:
                    raise ValueError("melt_manager mode requires params.target_mix")
                if not isinstance(target_mix, (list, tuple)) or len(target_mix) != self.max_segment_length:
                    raise ValueError("params.target_mix must be a list/tuple with length equal to max_segment_length")

                total = float(sum(float(x) for x in target_mix))
                if total <= 0:
                    raise ValueError("params.target_mix must sum to a positive value")

                target_steps = float(params.get("target_steps", 100_000.0))
                if target_steps <= 0:
                    raise ValueError("params.target_steps must be > 0")

                skip_probability = float(params.get("skip_probability", 0.5))
                if not (0.0 <= skip_probability <= 1.0):
                    raise ValueError("params.skip_probability must be in [0, 1]")

                concentration = float(params.get("concentration", 30.0))
                if concentration <= 0:
                    raise ValueError("params.concentration must be > 0")
            elif spec.name == "dp":
                params = spec.params or {}
                range_cfg = params.get("target_length_range", None)
                if range_cfg is None:
                    raise ValueError("dp strategy requires params.target_length_range")
                if not isinstance(range_cfg, (list, tuple)) or len(range_cfg) != 2:
                    raise ValueError("target_length_range must be a 2-tuple or list")
            elif spec.name == "keep":
                continue
            else:
                raise ValueError(f"Unknown strategy: {spec.name}")
