from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

import torch

from rebots.models.components.elastic_time_replace import binary_to_segment_matrix, dp_h_batched, greedy_h_for_predictor


@dataclass
class StrategySpec:
    name: Literal["random", "random_proportion", "greedy", "dp", "downsample", "keep"]
    weight: float
    params: Optional[dict[str, Any]] = None


@dataclass
class ElasticTimeMaskSample:
    h_prefilled: torch.Tensor
    greedy_mask: torch.Tensor
    greedy_target_length: torch.Tensor
    target_replace_fraction: torch.Tensor
    strategy_indices: torch.Tensor
    dp_mask: Optional[torch.Tensor] = None
    dp_target_length: Optional[torch.Tensor] = None


class ElasticTimeMasker:
    """
    Produce an Elastic Time offset mask h for mixing z with Elastic Time predictions.

    h has shape (B, T) with values in [0..K]. h=0 means keep z_t.
    """

    def __init__(
        self,
        K: int,
        strategies: Iterable[StrategySpec | dict[str, Any]],
    ) -> None:
        self.K = K
        self._strategies = self._normalize_strategies(strategies)
        self._validate_strategies(self._strategies)

    def sample(
        self,
        batch_size: int,
        T: int,
        device: torch.device,
        *,
        progress: Optional[float] = None,
    ) -> ElasticTimeMaskSample:
        """
        Sample masking plans for a batch without using latents.

        Random/random_proportion/keep/downsample strategies directly populate h;
        greedy/dp rows are deferred
        and only store target lengths.

        Args:
            batch_size: Batch size.
            T: Sequence length.
            device: Device for sampled tensors.
            progress: Optional normalized training progress in [0, 1] used by
                curriculum target length ranges.
        """
        weights = torch.tensor([s.weight for s in self._strategies], device=device)
        probs = weights / weights.sum()
        strategy_indices = torch.multinomial(probs, num_samples=batch_size, replacement=True)

        h_prefilled = torch.zeros(batch_size, T, dtype=torch.long, device=device)
        greedy_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        greedy_target_length = torch.full((batch_size,), T, dtype=torch.long, device=device)
        dp_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        dp_target_length = torch.full((batch_size,), T, dtype=torch.long, device=device)
        target_replace_fraction = torch.zeros(batch_size, dtype=torch.float32, device=device)

        for idx, spec in enumerate(self._strategies):
            mask = strategy_indices == idx
            if not mask.any():
                continue

            params = spec.params or {}
            cur_batch_size = int(mask.sum().item())
            if spec.name == "random":
                h_candidate = self._random_segments_h_from_shape(cur_batch_size, T, device, params)
                h_prefilled[mask] = h_candidate
                target_replace_fraction[mask] = (h_candidate != 0).float().mean(dim=1)
            elif spec.name == "random_proportion":
                h_candidate = self._random_proportion_h_from_shape(cur_batch_size, T, device, params)
                h_prefilled[mask] = h_candidate
                target_replace_fraction[mask] = (h_candidate != 0).float().mean(dim=1)
            elif spec.name == "greedy":
                greedy_mask[mask] = True
                target_length = self._sample_greedy_target_length(cur_batch_size, T, device, params, progress=progress)
                greedy_target_length[mask] = target_length
                target_replace_fraction[mask] = (T - target_length).float() / float(T)
            elif spec.name == "dp":
                dp_mask[mask] = True
                target_length = self._sample_dp_target_length(cur_batch_size, T, device, params, progress=progress)
                dp_target_length[mask] = target_length
                target_replace_fraction[mask] = (T - target_length).float() / float(T)
            elif spec.name == "downsample":
                factor = self._get_downsample_factor(params)
                h_candidate = self._downsample_h_from_shape(cur_batch_size, T, device, factor=factor)
                h_prefilled[mask] = h_candidate
                target_replace_fraction[mask] = (h_candidate != 0).float().mean(dim=1)
            elif spec.name == "keep":
                h_prefilled[mask] = 0
                target_replace_fraction[mask] = 0.0
            else:
                raise ValueError(f"Unknown strategy: {spec.name}")

        return ElasticTimeMaskSample(
            h_prefilled=h_prefilled,
            greedy_mask=greedy_mask,
            greedy_target_length=greedy_target_length,
            target_replace_fraction=target_replace_fraction,
            strategy_indices=strategy_indices,
            dp_mask=dp_mask,
            dp_target_length=dp_target_length,
        )

    def replace(self, z: torch.Tensor, predictor: torch.nn.Module, sample: ElasticTimeMaskSample) -> torch.Tensor:
        """
        Build final h from sampled plans.

        Random/keep rows use prefilled h from `sample`; greedy/dp rows are
        computed from latents and predictor.
        """
        h = sample.h_prefilled.clone()
        if sample.greedy_mask.any():
            h_greedy = self._greedy_h(
                z[sample.greedy_mask],
                predictor,
                target_length=sample.greedy_target_length[sample.greedy_mask],
            )
            h[sample.greedy_mask] = h_greedy

        dp_mask = sample.dp_mask
        if dp_mask is not None and dp_mask.any():
            if sample.dp_target_length is None:
                raise ValueError("dp_mask is set but dp_target_length is missing in ElasticTimeMaskSample")
            h_dp = self._dp_h(
                z[dp_mask],
                predictor,
                target_length=sample.dp_target_length[dp_mask],
            )
            h[dp_mask] = h_dp
        return h

    def __call__(self, z: torch.Tensor, predictor: torch.nn.Module) -> torch.Tensor:
        """
        Args:
            z: (B, C, T) float  -- Latent states
            predictor: Module  -- Predicts A z_{t-1} from z_{t-1}
        Returns:
            h: (B, T) long  -- Elastic Time offset, for mix_with_elastic_time_segments:
                               h[t]=0 -> keep z_t
                               h[t]=k -> use A^k z_{t-k}
        """
        B, _, T = z.shape
        sample = self.sample(B, T, z.device)
        return self.replace(z, predictor, sample)

    def _random_segments_h(self, z: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        B, _, T = z.shape
        return self._random_segments_h_from_shape(B, T, z.device, params)

    def _get_downsample_factor(self, params: dict[str, Any]) -> int:
        factor = params.get("factor", params.get("ds", None))
        if factor is None:
            raise ValueError("downsample strategy requires params.factor >= 2")
        factor = int(factor)
        if factor < 2:
            raise ValueError("downsample strategy requires params.factor >= 2")
        return factor

    def _downsample_h_from_shape(self, B: int, T: int, device: torch.device, factor: int) -> torch.Tensor:
        result = torch.zeros(B, T, dtype=torch.long, device=device)
        result[:, :] = torch.arange(T, device=device, dtype=torch.long).remainder(factor)
        return result

    def _random_segments_h_from_shape(
        self, B: int, T: int, device: torch.device, params: dict[str, Any]
    ) -> torch.Tensor:
        N_raw = params.get("N", None)
        if N_raw is None:
            raise ValueError("random strategy requires params.N >= 1")
        N = int(N_raw)
        if N < 1:
            raise ValueError("random strategy requires params.N >= 1")

        # Start with zeros
        result = torch.zeros(B, T, dtype=torch.long, device=device)

        # Random segment lengths (B, N)
        max_len = self.K + 1
        lengths = torch.randint(1, max_len + 1, (B, N), device=device)

        # Random starting positions (B, N)
        max_start = max(0, T - max_len)
        starts = torch.randint(0, max(1, max_start + 1), (B, N), device=device)

        # sort in the last dimension to avoid overlapping segments
        starts, _ = torch.sort(starts, dim=-1)

        # crop lengths to avoid overflow into the next segment
        max_lengths = torch.empty_like(lengths)
        max_lengths[:, :-1] = starts[:, 1:] - starts[:, :-1]
        max_lengths[:, -1] = T - starts[:, -1]
        lengths = torch.min(lengths, max_lengths)

        # Create segment indices (0, 1, 2, ..., K) for each segment
        segment_indices = torch.arange(max_len, device=device).view(1, 1, -1)

        # Create absolute positions for each value (B, N, K+1)
        positions = starts.unsqueeze(-1) + segment_indices

        # Mask valid positions (within segment length and within T)
        mask = (segment_indices < lengths.unsqueeze(-1)) & (positions < T)

        # For each position, compute which segment index it belongs to (0, 1, 2, ...)
        segment_values = segment_indices.expand(B, N, -1)[mask]

        # Flatten indices for scatter
        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, max_len)[mask]
        position_indices = positions[mask]

        # Scatter - later segments will overwrite earlier ones at same position
        result[batch_indices, position_indices] = segment_values

        return result

    def _resolve_proportion_range(self, params: dict[str, Any]) -> tuple[float, float]:
        proportion_range = params.get("proportion_range", None)
        if proportion_range is None:
            raise ValueError("random_proportion strategy requires params.proportion_range")
        self._validate_target_length_range(proportion_range, name="proportion_range")
        low, high = proportion_range
        return float(low), float(high)

    def _random_proportion_h_from_shape(
        self,
        B: int,
        T: int,
        device: torch.device,
        params: dict[str, Any],
    ) -> torch.Tensor:
        result = torch.zeros(B, T, dtype=torch.long, device=device)
        if T <= 1:
            return result

        low, high = self._resolve_proportion_range(params)
        sampled_proportions = torch.empty(B, device=device, dtype=torch.float32).uniform_(low, high)
        target_replace_count = torch.round(sampled_proportions * float(T)).to(dtype=torch.long).clamp(min=0, max=T - 1)

        if int(target_replace_count.max().item()) == 0:
            return result

        random_scores = torch.rand(B, T - 1, device=device)
        random_ranks = torch.argsort(random_scores, dim=1).argsort(dim=1)

        replace_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        replace_mask[:, 1:] = random_ranks < target_replace_count.unsqueeze(1)

        result = binary_to_segment_matrix(replace_mask).clamp(max=self.K)
        return result.to(dtype=torch.long)

    def _resolve_target_length_range(
        self,
        params: dict[str, Any],
        progress: Optional[float],
    ) -> tuple[float, float]:
        target_length_range = params.get("target_length_range", None)
        if target_length_range is None:
            raise ValueError("strategy requires params.target_length_range")

        curriculum = params.get("curriculum", None)
        if curriculum is None:
            low, high = target_length_range
            return float(low), float(high)

        start_range = curriculum.get("start_target_length_range", target_length_range)
        end_range = curriculum.get("end_target_length_range", target_length_range)
        start_progress = float(curriculum.get("start_progress", 0.0))
        end_progress = float(curriculum.get("end_progress", 1.0))

        p = 0.0 if progress is None else float(progress)
        p = max(0.0, min(1.0, p))

        if p <= start_progress:
            alpha = 0.0
        elif p >= end_progress:
            alpha = 1.0
        else:
            alpha = (p - start_progress) / max(1e-8, end_progress - start_progress)

        low0, high0 = float(start_range[0]), float(start_range[1])
        low1, high1 = float(end_range[0]), float(end_range[1])
        low = (1.0 - alpha) * low0 + alpha * low1
        high = (1.0 - alpha) * high0 + alpha * high1
        return low, high

    def _sample_greedy_target_length(
        self,
        B: int,
        T: int,
        device: torch.device,
        params: dict[str, Any],
        *,
        progress: Optional[float] = None,
    ) -> torch.Tensor:
        low, high = self._resolve_target_length_range(params, progress)
        low_idx = max(1, int(T * low))
        high_idx = min(T, int(T * high))
        if high_idx < low_idx:
            high_idx = low_idx
        return torch.randint(low_idx, high_idx + 1, (B,), device=device)

    def _sample_dp_target_length(
        self,
        B: int,
        T: int,
        device: torch.device,
        params: dict[str, Any],
        *,
        progress: Optional[float] = None,
    ) -> torch.Tensor:
        target_length = self._sample_greedy_target_length(B, T, device, params, progress=progress)
        min_feasible_keep = (T + self.K) // (self.K + 1)
        return target_length.clamp(min=min_feasible_keep, max=T)

    @torch.no_grad()
    def _greedy_h(
        self,
        z: torch.Tensor,
        predictor: torch.nn.Module,
        params: Optional[dict[str, Any]] = None,
        target_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Greedy selection to decide which timesteps t to overwrite with A z_{t-1}.
        Discrete; runs in no_grad.

        Returns:
        h: (B, T) long  -- Elastic Time offset, for mix_with_elastic_time_segments:
                           h[t]=0 -> keep z_t
                           h[t]=k -> use A^k z_{t-k}
        """
        B, _, T = z.shape

        # Get target length
        if target_length is None:
            params = params or {}
            target_length = self._sample_greedy_target_length(B, T, z.device, params)
        else:
            target_length = target_length.to(device=z.device, dtype=torch.long)

        return greedy_h_for_predictor(
            z,
            predictor,
            target_length=target_length,
            K=self.K,
        )

    @torch.no_grad()
    def _dp_h(
        self,
        z: torch.Tensor,
        predictor: torch.nn.Module,
        params: Optional[dict[str, Any]] = None,
        target_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Exact DP selection to decide which timesteps to overwrite.

        Returns:
            h: (B, T) long, with h[t]=0 keep and h[t]=k use A^k z_{t-k}
        """
        B, _, T = z.shape

        if target_length is None:
            params = params or {}
            target_length_tensor = self._sample_dp_target_length(B, T, z.device, params)
        else:
            target_length_tensor = target_length.to(device=z.device, dtype=torch.long)

        min_feasible_keep = (T + self.K) // (self.K + 1)
        target_length_tensor = target_length_tensor.clamp(min=min_feasible_keep, max=T)

        return dp_h_batched(
            z,
            predictor,
            target_length=target_length_tensor,
            K=self.K,
        )

    def _normalize_strategies(self, strategies: Iterable[StrategySpec | dict[str, Any]]) -> list[StrategySpec]:
        normalized: list[StrategySpec] = []
        for spec in strategies:
            if isinstance(spec, StrategySpec):
                normalized.append(spec)
            elif isinstance(spec, dict):
                params = spec.get("params", None)
                normalized.append(
                    StrategySpec(
                        name=spec["name"],
                        weight=spec["weight"],
                        params=params,
                    )
                )
            else:
                raise ValueError("Each strategy must be a StrategySpec or a dict")
        return normalized

    def _validate_target_length_range(self, value: Any, *, name: str) -> None:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{name} must be a 2-tuple or list")
        low, high = float(value[0]), float(value[1])
        if not (0.0 <= low <= 1.0 and 0.0 <= high <= 1.0 and low <= high):
            raise ValueError(f"{name} must satisfy 0 <= low <= high <= 1")

    def _validate_curriculum(self, params: dict[str, Any]) -> None:
        curriculum = params.get("curriculum", None)
        if curriculum is None:
            return
        if not isinstance(curriculum, dict):
            raise ValueError("curriculum must be a dict")

        self._validate_target_length_range(
            curriculum.get("start_target_length_range", params["target_length_range"]),
            name="curriculum.start_target_length_range",
        )
        self._validate_target_length_range(
            curriculum.get("end_target_length_range", params["target_length_range"]),
            name="curriculum.end_target_length_range",
        )

        start_progress = float(curriculum.get("start_progress", 0.0))
        end_progress = float(curriculum.get("end_progress", 1.0))
        if not (0.0 <= start_progress <= 1.0 and 0.0 <= end_progress <= 1.0 and start_progress <= end_progress):
            raise ValueError("curriculum progress must satisfy 0 <= start_progress <= end_progress <= 1")

    def _validate_strategies(self, strategies: list[StrategySpec]) -> None:
        if not strategies:
            raise ValueError("strategies must be a non-empty list")
        for spec in strategies:
            if spec.weight < 0:
                raise ValueError("strategy weight must be >= 0")
            if spec.name == "random":
                params = spec.params or {}
                N = params.get("N", None)
                if N is None or N < 1:
                    raise ValueError("random strategy requires params.N >= 1")
            elif spec.name == "random_proportion":
                params = spec.params or {}
                proportion_range = params.get("proportion_range", None)
                if proportion_range is None:
                    raise ValueError("random_proportion strategy requires params.proportion_range")
                self._validate_target_length_range(proportion_range, name="proportion_range")
            elif spec.name == "greedy":
                params = spec.params or {}
                tlr = params.get("target_length_range", None)
                if tlr is None:
                    raise ValueError("greedy strategy requires params.target_length_range")
                self._validate_target_length_range(tlr, name="target_length_range")
                self._validate_curriculum(params)
            elif spec.name == "dp":
                params = spec.params or {}
                tlr = params.get("target_length_range", None)
                if tlr is None:
                    raise ValueError("dp strategy requires params.target_length_range")
                self._validate_target_length_range(tlr, name="target_length_range")
                self._validate_curriculum(params)
            elif spec.name == "downsample":
                params = spec.params or {}
                factor = self._get_downsample_factor(params)
                if self.K < factor - 1:
                    raise ValueError(f"downsample strategy requires K >= {factor - 1} for factor={factor}")
            elif spec.name == "keep":
                continue
            else:
                raise ValueError(f"Unknown strategy: {spec.name}")
