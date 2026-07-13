from __future__ import annotations

from typing import Iterable, Optional

import torch


def mix_with_elastic_time_segments(z: torch.Tensor, z_preds: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    Mix original latents with Elastic Time rollouts using offset mask ``h``.

    Args:
        z: Latents of shape (B, C, T).
        z_preds: Elastic Time rollouts of shape (B, K, C, T), where
            ``z_preds[b, k-1, :, t0] = A^k z[b, :, t0]`` for ``k in [1..K]``.
        h: Offset mask of shape (B, T) with values in [0..K].
            ``h[t] = 0`` keeps ``z_t`` and ``h[t] = k`` uses ``A^k z_{t-k}``.

    Returns:
        Mixed latents of shape (B, C, T).
    """
    if z.ndim != 3:
        raise ValueError(f"z must be (B, C, T), got {tuple(z.shape)}")
    if z_preds.ndim != 4:
        raise ValueError(f"z_preds must be (B, K, C, T), got {tuple(z_preds.shape)}")
    if h.ndim != 2:
        raise ValueError(f"h must be (B, T), got {tuple(h.shape)}")

    device = z.device
    B, C, T = z.shape
    Bp, K, Cp, Tp = z_preds.shape
    if Bp != B or Cp != C or Tp != T:
        raise ValueError(f"z_preds shape mismatch: expected ({B}, K, {C}, {T}), got {tuple(z_preds.shape)}")
    if h.shape != (B, T):
        raise ValueError(f"h shape mismatch: expected {(B, T)}, got {tuple(h.shape)}")

    h = h.to(device=device, dtype=torch.long)
    keep_mask = h == 0
    if K == 0:
        if not bool(keep_mask.all()):
            raise ValueError("h contains non-zero offsets but z_preds has K=0")
        return z

    t = torch.arange(T, device=device).view(1, T).expand(B, T)
    s_idx = t - h

    k_idx = (h - 1).clamp(min=0)
    lin_idx = k_idx * T + s_idx

    z_preds_flat = z_preds.permute(0, 2, 1, 3).reshape(B, C, K * T)
    pred_from_rollouts = z_preds_flat.gather(2, lin_idx.unsqueeze(1).expand(-1, C, -1))

    return torch.where(keep_mask.unsqueeze(1), z, pred_from_rollouts)


def binary_to_segment_matrix(replaced_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a binary replacement mask into Elastic Time offset matrix ``h``.

    Args:
        replaced_mask: Bool or int mask of shape (B, T).

    Returns:
        Offset matrix ``h`` of shape (B, T).
    """
    if replaced_mask.ndim != 2:
        raise ValueError(f"replaced_mask must be (B, T), got {tuple(replaced_mask.shape)}")

    s = replaced_mask.to(dtype=torch.int64)
    cs = s.cumsum(dim=-1)

    cs_mask = cs.clone()
    cs_mask[s > 0] = 0

    return cs - cs_mask.cummax(-1).values


def segment_matrix_to_segment_length_histogram(h: torch.Tensor, K: int) -> torch.Tensor:
    """
    Convert integer segment matrix ``h`` into a histogram.

    Args:
        h: Segment matrix of shape (B, T) with values in [0..K].
        K: Maximum value in ``h``.

    Returns:
        Histogram of shape (K+1,), averaged per sample (divided by ``B``).
    """
    if h.ndim != 2:
        raise ValueError(f"h must be (B, T), got {tuple(h.shape)}")
    if h.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("h must be an integer tensor")
    if K < 0:
        raise ValueError(f"K must be >= 0, got {K}")
    if (h < 0).any() or (h > K).any():
        h = h.clamp(0, K)

    B = h.shape[0]
    hist = torch.bincount(h.reshape(-1), minlength=K + 1).to(dtype=torch.float32)
    hist = hist / float(B)
    return hist


def greedy_h(
    z: torch.Tensor,
    predictor: torch.nn.Module,
    target_length: Optional[torch.Tensor | int] = None,
    target_length_range: Optional[tuple[float, float]] = None,
    K: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute Elastic Time greedy offset mask ``h``.

    Args:
        z: Latents of shape (B, C, T).
        predictor: Elastic Time predictor module.
        target_length: Desired kept length per sample. Can be scalar int,
            scalar tensor, or shape (B,) tensor.
        target_length_range: If ``target_length`` is None, sample per-sample
            target lengths from this normalized range ``[low, high]``.
        K: Optional maximum predictor rollout depth for chaining. If provided,
            chain extension is capped to keep ``h`` in ``[0..K]``.

    Returns:
        Offset mask ``h`` of shape (B, T) with values in [0..T-1].
    """
    if z.ndim != 3:
        raise ValueError(f"z must be (B, C, T), got {tuple(z.shape)}")

    B, C, T = z.shape

    if target_length is None:
        if target_length_range is None:
            raise ValueError("Either target_length or target_length_range must be provided")
        low, high = target_length_range
        low_idx = max(1, int(T * low))
        high_idx = min(T, int(T * high))
        target_length_tensor = torch.randint(low_idx, high_idx + 1, (B,), device=z.device, dtype=torch.long)
    else:
        if isinstance(target_length, int):
            target_length_tensor = torch.full((B,), target_length, device=z.device, dtype=torch.long)
        elif torch.is_tensor(target_length):
            target_length_tensor = target_length.to(device=z.device, dtype=torch.long)
            if target_length_tensor.ndim == 0:
                target_length_tensor = target_length_tensor.view(1)
            if target_length_tensor.shape[0] == 1 and B > 1:
                target_length_tensor = target_length_tensor.expand(B)
            if target_length_tensor.shape[0] != B:
                raise ValueError(f"target_length batch mismatch: expected {B}, got {target_length_tensor.shape[0]}")
        else:
            raise ValueError("target_length must be int, tensor, or None")

    target_length_tensor = target_length_tensor.clamp(1, T)
    k_cap = T if K is None else int(K)
    if k_cap < 0:
        raise ValueError(f"K must be >= 0, got {k_cap}")

    inf = float("inf")
    h = torch.zeros(B, T, dtype=torch.long, device=z.device)

    seq = z.detach().clone()

    err = torch.full((B, T), inf, device=z.device, dtype=torch.float32)
    pred_all = predictor(seq[:, :, :-1])
    err[:, 1:] = (seq[:, :, 1:] - pred_all).float().pow(2).mean(dim=1)

    num_to_replace = T - target_length_tensor
    max_steps = int(num_to_replace.max().item())

    for step in range(max_steps):
        cur_err, t = err.min(dim=1)
        active = (step < num_to_replace) & torch.isfinite(cur_err)
        if not active.any():
            break

        b_idx = active.nonzero(as_tuple=True)[0]
        t_sel = t[b_idx]

        prev = seq[b_idx].gather(2, (t_sel - 1).view(-1, 1, 1).expand(-1, C, 1))
        pred_sel = predictor(prev).squeeze(-1)

        # seq[b_idx].scatter_(
        #     2,
        #     t_sel.view(-1, 1, 1).expand(-1, C, 1),
        #     pred_sel.unsqueeze(-1),
        # )
        seq[b_idx, :, t_sel] = pred_sel

        h[b_idx, t_sel] = h[b_idx, t_sel - 1] + 1

        err[b_idx, t_sel] = inf
        err[b_idx, t_sel - 1] = inf

        t_next = t_sel + 1
        has_next = t_next < T
        if has_next.any():
            b2 = b_idx[has_next]
            t_sel2 = t_sel[has_next]
            t_next2 = t_next[has_next]

            # Enforce rollout depth cap on chain extension. If current depth
            # already reached K, disallow selecting t+1 next.
            can_extend = h[b2, t_sel2] < k_cap
            if (~can_extend).any():
                err[b2[~can_extend], t_next2[~can_extend]] = inf

            finite_next = can_extend & torch.isfinite(err[b2, t_next2])
            if finite_next.any():
                b3 = b2[finite_next]
                t_sel3 = t_sel2[finite_next]
                t_next3 = t_next2[finite_next]

                cur = seq[b3].gather(2, t_sel3.view(-1, 1, 1).expand(-1, C, 1))
                pred_next = predictor(cur).squeeze(-1)

                nxt = seq[b3].gather(2, t_next3.view(-1, 1, 1).expand(-1, C, 1)).squeeze(-1)

                err[b3, t_next3] = (nxt - pred_next).float().pow(2).mean(dim=1)

    return h


@torch.no_grad()
def greedy_h_rnn(
    z: torch.Tensor,
    predictor: torch.nn.Module,
    target_length: Optional[torch.Tensor | int] = None,
    target_length_range: Optional[tuple[float, float]] = None,
    K: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute Elastic Time greedy offset mask ``h`` for recurrent predictors.

    Args:
        z: Latents of shape (B, C, T).
        predictor: Recurrent Elastic Time predictor module. Must expose
            ``initial_hidden(batch_size, time_steps, *, device, dtype)`` and
            return ``(pred, h_next)`` from ``forward``.
        target_length: Desired kept length per sample. Can be scalar int,
            scalar tensor, or shape (B,) tensor.
        target_length_range: If ``target_length`` is None, sample per-sample
            target lengths from this normalized range ``[low, high]``.
        K: Optional maximum predictor rollout depth for chaining. If provided,
            chain extension is capped to keep ``h`` in ``[0..K]``.

    Returns:
        Offset mask ``h`` of shape (B, T) with values in [0..T-1].
    """
    if z.ndim != 3:
        raise ValueError(f"z must be (B, C, T), got {tuple(z.shape)}")

    B, C, T = z.shape

    if target_length is None:
        if target_length_range is None:
            raise ValueError("Either target_length or target_length_range must be provided")
        low, high = target_length_range
        low_idx = max(1, int(T * low))
        high_idx = min(T, int(T * high))
        target_length_tensor = torch.randint(low_idx, high_idx + 1, (B,), device=z.device, dtype=torch.long)
    else:
        if isinstance(target_length, int):
            target_length_tensor = torch.full((B,), target_length, device=z.device, dtype=torch.long)
        elif torch.is_tensor(target_length):
            target_length_tensor = target_length.to(device=z.device, dtype=torch.long)
            if target_length_tensor.ndim == 0:
                target_length_tensor = target_length_tensor.view(1)
            if target_length_tensor.shape[0] == 1 and B > 1:
                target_length_tensor = target_length_tensor.expand(B)
            if target_length_tensor.shape[0] != B:
                raise ValueError(f"target_length batch mismatch: expected {B}, got {target_length_tensor.shape[0]}")
        else:
            raise ValueError("target_length must be int, tensor, or None")

    target_length_tensor = target_length_tensor.clamp(1, T)
    k_cap = T if K is None else int(K)
    if k_cap < 0:
        raise ValueError(f"K must be >= 0, got {k_cap}")

    inf = float("inf")
    h = torch.zeros(B, T, dtype=torch.long, device=z.device)

    seq = z.detach().clone()

    hidden_states = predictor.initial_hidden(B, T, device=z.device, dtype=z.dtype)
    hidden_states = hidden_states.clone()

    err = torch.full((B, T), inf, device=z.device, dtype=torch.float32)
    pred_all, _ = predictor(seq[:, :, :-1], hidden_states[:, :, :, :-1])
    err[:, 1:] = (seq[:, :, 1:] - pred_all).float().pow(2).mean(dim=1)

    num_to_replace = T - target_length_tensor
    max_steps = int(num_to_replace.max().item())

    for step in range(max_steps):
        cur_err, t = err.min(dim=1)
        active = (step < num_to_replace) & torch.isfinite(cur_err)
        if not active.any():
            break

        b_idx = active.nonzero(as_tuple=True)[0]
        t_sel = t[b_idx]

        prev = seq[b_idx, :, t_sel - 1].unsqueeze(-1)
        prev_hidden = hidden_states[b_idx, :, :, t_sel - 1].unsqueeze(-1)
        pred_sel, next_hidden = predictor(prev, prev_hidden)
        pred_sel = pred_sel.squeeze(-1)
        next_hidden = next_hidden.squeeze(-1)

        seq[b_idx, :, t_sel] = pred_sel
        hidden_states[b_idx, :, :, t_sel] = next_hidden

        h[b_idx, t_sel] = h[b_idx, t_sel - 1] + 1

        err[b_idx, t_sel] = inf
        err[b_idx, t_sel - 1] = inf

        t_next = t_sel + 1
        has_next = t_next < T
        if has_next.any():
            b2 = b_idx[has_next]
            t_sel2 = t_sel[has_next]
            t_next2 = t_next[has_next]

            # Enforce rollout depth cap on chain extension. If current depth
            # already reached K, disallow selecting t+1 next.
            can_extend = h[b2, t_sel2] < k_cap
            if (~can_extend).any():
                err[b2[~can_extend], t_next2[~can_extend]] = inf

            finite_next = can_extend & torch.isfinite(err[b2, t_next2])
            if finite_next.any():
                b3 = b2[finite_next]
                t_sel3 = t_sel2[finite_next]
                t_next3 = t_next2[finite_next]

                cur = seq[b3, :, t_sel3].unsqueeze(-1)
                cur_hidden = hidden_states[b3, :, :, t_sel3].unsqueeze(-1)
                pred_next, _ = predictor(cur, cur_hidden)
                pred_next = pred_next.squeeze(-1)

                nxt = seq[b3, :, t_next3]

                err[b3, t_next3] = (nxt - pred_next).float().pow(2).mean(dim=1)

    return h


def _predictor_supports_rollout_state(predictor: torch.nn.Module) -> bool:
    """Return True when predictor exposes recurrent rollout-state API."""
    return callable(getattr(predictor, "initial_hidden", None))


def greedy_h_for_predictor(
    z: torch.Tensor,
    predictor: torch.nn.Module,
    target_length: Optional[torch.Tensor | int] = None,
    target_length_range: Optional[tuple[float, float]] = None,
    K: Optional[int] = None,
) -> torch.Tensor:
    """Route greedy mask computation to stateless or recurrent implementation."""
    if _predictor_supports_rollout_state(predictor):
        return greedy_h_rnn(
            z=z,
            predictor=predictor,
            target_length=target_length,
            target_length_range=target_length_range,
            K=K,
        )

    return greedy_h(
        z=z,
        predictor=predictor,
        target_length=target_length,
        target_length_range=target_length_range,
        K=K,
    )


@torch.no_grad()
def shrink_with_elastic_time_multi_rnn(
    z: torch.Tensor,
    predictor: torch.nn.Module,
    Ns: Iterable[int],
    return_h: bool = False,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Greedy replacement snapshots for multiple target kept lengths with RNN predictors."""
    if z.ndim != 3:
        raise ValueError(f"z must be (B, C, T), got {tuple(z.shape)}")

    B, C, T = z.shape
    ns_list = list(Ns)
    if len(ns_list) == 0:
        raise ValueError("Ns must be a non-empty iterable")
    for n in ns_list:
        if not (1 <= n <= T):
            raise ValueError(f"Each N must satisfy 1 <= N <= T, got N={n}, T={T}")

    if not callable(getattr(predictor, "initial_hidden", None)):
        raise ValueError("predictor must define initial_hidden for shrink_with_elastic_time_multi_rnn")

    k_targets = [T - n for n in ns_list]
    unique_k = sorted(set(k_targets))
    max_k = max(unique_k)

    k_to_indices: dict[int, list[int]] = {}
    for i, k in enumerate(k_targets):
        k_to_indices.setdefault(k, []).append(i)

    z_outs = [torch.empty_like(z) for _ in ns_list]
    h_outs = [torch.empty((B, T), dtype=torch.long, device=z.device) for _ in ns_list] if return_h else None
    predictor = predictor.to(device=z.device, dtype=z.dtype)

    def _predict_step(x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred = predictor(x, state)
        if not isinstance(pred, tuple) or len(pred) != 2:
            raise ValueError("predictor must return (pred, h_next) for shrink_with_elastic_time_multi_rnn")
        return pred

    for b in range(B):
        seq = z[b].clone()
        replaced = torch.zeros(T, dtype=torch.bool, device=z.device)
        h = torch.zeros(T, dtype=torch.long, device=z.device)

        hidden_states = predictor.initial_hidden(1, T, device=z.device, dtype=z.dtype).clone()

        err = torch.full((T,), float("inf"), device=z.device)
        for t in range(1, T):
            pred_t, _ = _predict_step(seq[:, t - 1].view(1, C, 1), hidden_states[:, :, :, t - 1 : t])
            err[t] = (seq[:, t] - pred_t.squeeze(0).squeeze(-1)).pow(2).mean()

        replaced_count = 0
        snapped: set[int] = set()

        def snapshot(k: int) -> None:
            for idx in k_to_indices.get(k, []):
                z_outs[idx][b].copy_(seq)
                if return_h and h_outs is not None:
                    h_outs[idx][b].copy_(h)

        if 0 in k_to_indices:
            snapshot(0)
            snapped.add(0)

        while replaced_count < max_k:
            t = int(err.argmin().item())
            if t == 0 or torch.isinf(err[t]):
                break

            if replaced[t]:
                err[t] = float("inf")
                continue

            pred_sel, next_hidden = _predict_step(seq[:, t - 1].view(1, C, 1), hidden_states[:, :, :, t - 1 : t])
            seq[:, t] = pred_sel.squeeze(0).squeeze(-1)
            hidden_states[:, :, :, t] = next_hidden.squeeze(-1)
            h[t] = h[t - 1] + 1

            replaced[t] = True
            if t - 1 >= 0:
                replaced[t - 1] = True

            replaced_count += 1

            err[t] = float("inf")
            if t - 1 >= 0:
                err[t - 1] = float("inf")

            if t + 1 < T and not torch.isinf(err[t + 1]):
                pred_next, _ = _predict_step(seq[:, t].view(1, C, 1), hidden_states[:, :, :, t : t + 1])
                err[t + 1] = (seq[:, t + 1] - pred_next.squeeze(0).squeeze(-1)).pow(2).mean()

            if replaced_count in k_to_indices and replaced_count not in snapped:
                snapshot(replaced_count)
                snapped.add(replaced_count)

        for k in unique_k:
            if k not in snapped:
                snapshot(k)
                snapped.add(k)

    if return_h and h_outs is not None:
        return list(z_outs), [h_cur for h_cur in h_outs]
    return list(z_outs)


@torch.no_grad()
def shrink_with_elastic_time_multi_for_predictor(
    z: torch.Tensor,
    predictor: torch.nn.Module,
    Ns: Iterable[int],
    return_h: bool = False,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Greedy multi-target shrink that dispatches based on predictor API."""
    if _predictor_supports_rollout_state(predictor):
        return shrink_with_elastic_time_multi_rnn(z, predictor, Ns, return_h=return_h)
    return shrink_with_elastic_time_multi(z, predictor, Ns, return_h=return_h)


@torch.no_grad()
def shrink_with_elastic_time_multi(
    z: torch.Tensor,
    predictor: torch.nn.Module,
    Ns: Iterable[int],
    return_h: bool = False,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Greedy replacement snapshots for multiple target kept lengths.

    Args:
        z: Latents of shape (B, C, T).
        predictor: Elastic Time predictor mapping (C, 1) -> (C, 1), or a module
            compatible with that call pattern.
        Ns: Target kept lengths. For each N, performs exactly ``T - N`` greedy
            replacements and snapshots the resulting sequence.

    Returns:
        List of ``z_new`` tensors aligned with ``Ns`` order, each shape (B, C, T).
    """
    if z.ndim != 3:
        raise ValueError(f"z must be (B, C, T), got {tuple(z.shape)}")

    B, C, T = z.shape
    ns_list = list(Ns)
    if len(ns_list) == 0:
        raise ValueError("Ns must be a non-empty iterable")
    for n in ns_list:
        if not (1 <= n <= T):
            raise ValueError(f"Each N must satisfy 1 <= N <= T, got N={n}, T={T}")

    k_targets = [T - n for n in ns_list]
    unique_k = sorted(set(k_targets))
    max_k = max(unique_k)

    k_to_indices: dict[int, list[int]] = {}
    for i, k in enumerate(k_targets):
        k_to_indices.setdefault(k, []).append(i)

    z_outs = [torch.empty_like(z) for _ in ns_list]
    h_outs = [torch.empty((B, T), dtype=torch.long, device=z.device) for _ in ns_list] if return_h else None
    predictor = predictor.to(device=z.device, dtype=z.dtype)

    for b in range(B):
        seq = z[b].clone()
        replaced = torch.zeros(T, dtype=torch.bool, device=z.device)
        h = torch.zeros(T, dtype=torch.long, device=z.device)

        err = torch.full((T,), float("inf"), device=z.device)
        for t in range(1, T):
            pred = predictor(seq[:, t - 1].unsqueeze(-1)).squeeze(-1)
            err[t] = (seq[:, t] - pred).pow(2).mean()

        replaced_count = 0
        snapped: set[int] = set()

        def snapshot(k: int) -> None:
            for idx in k_to_indices.get(k, []):
                z_outs[idx][b].copy_(seq)
                if return_h and h_outs is not None:
                    h_outs[idx][b].copy_(h)

        if 0 in k_to_indices:
            snapshot(0)
            snapped.add(0)

        while replaced_count < max_k:
            t = int(err.argmin().item())
            if t == 0 or torch.isinf(err[t]):
                break

            if replaced[t]:
                err[t] = float("inf")
                continue

            pred = predictor(seq[:, t - 1].unsqueeze(-1)).squeeze(-1)
            seq[:, t] = pred.clone()
            h[t] = h[t - 1] + 1

            replaced[t] = True
            if t - 1 >= 0:
                replaced[t - 1] = True

            replaced_count += 1

            err[t] = float("inf")
            if t - 1 >= 0:
                err[t - 1] = float("inf")

            if t + 1 < T and not torch.isinf(err[t + 1]):
                pred_next = predictor(seq[:, t].unsqueeze(-1)).squeeze(-1)
                err[t + 1] = (seq[:, t + 1] - pred_next).pow(2).mean()

            if replaced_count in k_to_indices and replaced_count not in snapped:
                snapshot(replaced_count)
                snapped.add(replaced_count)

        for k in unique_k:
            if k not in snapped:
                snapshot(k)
                snapped.add(k)

    if return_h and h_outs is not None:
        return list(z_outs), [h_cur for h_cur in h_outs]
    return list(z_outs)


def _precompute_segment_costs(
    seq: torch.Tensor,
    predictor: torch.nn.Module,
    max_k: int,
) -> torch.Tensor:
    """
    Precompute segment replacement costs for the DP solver.

    seg_cost[a, L] is the cumulative replacement cost for replacing positions
    ``(a+1 .. a+L)`` anchored at kept position ``a``.
    """
    C, T = seq.shape
    max_l = min(max_k, T - 1)
    seg_cost = torch.full((T, max_l + 1), float("inf"), device=seq.device, dtype=torch.float32)
    seg_cost[:, 0] = 0.0

    for a in range(T):
        lmax_a = min(max_l, T - 1 - a)
        if lmax_a <= 0:
            continue

        parent = seq[:, a]
        running = 0.0
        for l in range(1, lmax_a + 1):
            pred = predictor(parent.unsqueeze(-1)).squeeze(-1)
            target = seq[:, a + l]
            step = (pred.float() - target.float()).pow(2).mean()
            running = running + step
            seg_cost[a, l] = running
            parent = pred

    return seg_cost


def _precompute_segment_costs_batched(
    z: torch.Tensor,
    predictor: torch.nn.Module,
    max_k: int,
) -> torch.Tensor:
    """
    Batched version of ``_precompute_segment_costs``.

    Args:
        z: Latents of shape (B, C, T).
        predictor: Predictor module. Supports both stateless predictors and
            recurrent predictors returning ``(pred, h_next)``.
        max_k: Maximum segment length to evaluate.

    Returns:
        Segment cost tensor of shape (B, T, Lmax + 1), where
        ``Lmax = min(max_k, T - 1)``.
    """
    if z.ndim != 3:
        raise ValueError(f"z must be (B,C,T), got {tuple(z.shape)}")

    B, C, T = z.shape
    max_l = min(max_k, T - 1)

    seg_cost = torch.full((B, T, max_l + 1), float("inf"), device=z.device, dtype=torch.float32)
    seg_cost[:, :, 0] = 0.0

    if max_l == 0:
        return seg_cost

    supports_state = _predictor_supports_rollout_state(predictor)

    current = z
    h_state = None
    running = torch.zeros(B, T, device=z.device, dtype=torch.float32)

    for l in range(1, max_l + 1):
        pred = predictor(current, h_state) if supports_state and h_state is not None else predictor(current)

        if isinstance(pred, tuple):
            if len(pred) != 2:
                raise ValueError(f"Expected predictor output tuple of length 2, got {len(pred)}")
            current, h_state = pred
        else:
            current, h_state = pred, None

        if current.ndim != 3 or current.shape != (B, C, T):
            raise ValueError(f"predictor output shape mismatch: expected {(B, C, T)}, got {tuple(current.shape)}")

        # Valid anchors satisfy a + l <= T - 1  =>  a < T - l.
        valid_len = T - l
        pred_valid = current[:, :, :valid_len]
        target_valid = z[:, :, l:]

        step = (pred_valid.float() - target_valid.float()).pow(2).mean(dim=1)
        running[:, :valid_len] = running[:, :valid_len] + step
        seg_cost[:, :valid_len, l] = running[:, :valid_len]

    return seg_cost


def _rollout_predictor_latents_for_predictor(
    z: torch.Tensor,
    predictor: torch.nn.Module,
    max_k: int,
) -> torch.Tensor:
    """
    Roll out predictor for ``k=1..max_k`` and return ``(B, max_k, C, T)``.

    Supports both stateless predictors and stateful predictors returning
    ``(pred, h_next)``.
    """
    if max_k < 0:
        raise ValueError(f"max_k must be >= 0, got {max_k}")
    if z.ndim != 3:
        raise ValueError(f"z must be (B, C, T), got {tuple(z.shape)}")

    B, C, T = z.shape
    if max_k == 0:
        return torch.zeros(B, 0, C, T, device=z.device, dtype=z.dtype)

    supports_state = _predictor_supports_rollout_state(predictor)

    current = z
    h_state = None
    z_preds = torch.zeros(B, max_k, C, T, device=z.device, dtype=z.dtype)

    for k in range(max_k):
        pred = predictor(current, h_state) if supports_state and h_state is not None else predictor(current)

        if isinstance(pred, tuple):
            if len(pred) != 2:
                raise ValueError(f"Expected predictor output tuple of length 2, got {len(pred)}")
            current, h_state = pred
        else:
            current, h_state = pred, None

        if current.ndim != 3 or current.shape != (B, C, T):
            raise ValueError(f"predictor output shape mismatch: expected {(B, C, T)}, got {tuple(current.shape)}")

        z_preds[:, k, :, :] = current

    return z_preds


def _apply_segments(
    seq: torch.Tensor,
    predictor: torch.nn.Module,
    segments: list[tuple[int, int]],
) -> torch.Tensor:
    """
    Apply replacement segments to a single sequence.

    Args:
        seq: Single latent sequence of shape (C, T).
        predictor: Elastic Time predictor.
        segments: ``[(anchor_index, segment_length), ...]``.
    """
    out = seq.clone()
    for a, l in segments:
        parent = out[:, a]
        for k in range(1, l + 1):
            pred = predictor(parent.unsqueeze(-1)).squeeze(-1)
            out[:, a + k] = pred
            parent = pred
    return out


@torch.no_grad()
def shrink_with_elastic_time_multi_dp(
    z: torch.Tensor,
    predictor: torch.nn.Module,
    Ns: Iterable[int],
) -> list[torch.Tensor]:
    """
    Exact DP/shortest-path replacement for multiple target kept lengths.

    Args:
        z: Latents of shape (B, C, T).
        predictor: Elastic Time predictor module. Supports stateless and stateful
            predictors.
        Ns: Target kept lengths. For each N, solves exactly ``k = T - N``
            replacements with the segment-constrained optimum.

    Returns:
        List of ``z_new`` tensors aligned with ``Ns`` order, each shape
        (B, C, T).
    """
    if z.ndim != 3:
        raise ValueError(f"z must be (B,C,T); got {tuple(z.shape)}")
    B, C, T = z.shape

    ns_list = list(Ns)
    if len(ns_list) == 0:
        raise ValueError("Ns must be a non-empty iterable")
    for n in ns_list:
        if not (1 <= n <= T):
            raise ValueError(f"Each N must satisfy 1 <= N <= T, got N={n}, T={T}")

    k_targets = [T - n for n in ns_list]
    max_k = max(k_targets)
    if max_k > T - 1:
        raise ValueError(f"Infeasible: max replacements {max_k} > T-1 {T - 1} (t=0 has no parent).")

    predictor = predictor.to(device=z.device, dtype=z.dtype).eval()

    num_targets = len(ns_list)
    z_stacked = z.repeat(num_targets, 1, 1)

    target_lengths = torch.tensor(ns_list, device=z.device, dtype=torch.long)
    target_lengths = target_lengths.repeat_interleave(B)

    h_stacked = dp_h_batched(
        z_stacked,
        predictor,
        target_length=target_lengths,
    )

    max_h = int(h_stacked.max().item())
    z_preds_stacked = _rollout_predictor_latents_for_predictor(z_stacked, predictor, max_h)
    z_mixed_stacked = mix_with_elastic_time_segments(z_stacked, z_preds_stacked, h_stacked)

    z_outs = [torch.empty((B, C, T), device=z.device, dtype=z.dtype) for _ in ns_list]
    for i in range(num_targets):
        start = i * B
        end = (i + 1) * B
        z_outs[i].copy_(z_mixed_stacked[start:end])

    return z_outs


@torch.no_grad()
def dp_h_batched(
    z: torch.Tensor,
    predictor: torch.nn.Module,
    target_length: Optional[torch.Tensor | int] = None,
    target_length_range: Optional[tuple[float, float]] = None,
    K: Optional[int] = None,
) -> torch.Tensor:
    """
    Batched DP to compute optimal segment-based replacement mask ``h``.

    Args:
        z: Latents of shape (B, C, T).
        predictor: Predictor module. Supports both stateless predictors and
            recurrent predictors that follow the same step API used in
            ``train_rebottleneck_elastic_time``.
        target_length: Desired kept length per sample. Can be scalar int, scalar tensor, or shape (B,) tensor.
        target_length_range: If ``target_length`` is None, sample per-sample target lengths from this normalized range ``[low, high]``.
        K: Optional maximum predictor rollout depth for chaining. If provided, chain extension is capped to keep ``h`` in ``[0..K]``.

    Returns:
        Offset mask ``h`` of shape (B, T) with values in [0..T-1].
    """

    if z.ndim != 3:
        raise ValueError(f"z must be (B, C, T), got {tuple(z.shape)}")

    B, _, T = z.shape
    device = z.device

    if target_length is None:
        if target_length_range is None:
            raise ValueError("Either target_length or target_length_range must be provided")
        low, high = target_length_range
        low_idx = max(1, int(T * low))
        high_idx = min(T, int(T * high))
        if high_idx < low_idx:
            high_idx = low_idx
        target_length_tensor = torch.randint(low_idx, high_idx + 1, (B,), device=device, dtype=torch.long)
    else:
        if isinstance(target_length, int):
            target_length_tensor = torch.full((B,), target_length, device=device, dtype=torch.long)
        elif torch.is_tensor(target_length):
            target_length_tensor = target_length.to(device=device, dtype=torch.long)
            if target_length_tensor.ndim == 0:
                target_length_tensor = target_length_tensor.view(1)
            if target_length_tensor.shape[0] == 1 and B > 1:
                target_length_tensor = target_length_tensor.expand(B)
            if target_length_tensor.shape[0] != B:
                raise ValueError(f"target_length batch mismatch: expected {B}, got {target_length_tensor.shape[0]}")
        else:
            raise ValueError("target_length must be int, tensor, or None")

    target_length_tensor = target_length_tensor.clamp(1, T)
    k_targets = T - target_length_tensor

    k_chain_cap = T - 1 if K is None else int(K)
    if k_chain_cap < 0:
        raise ValueError(f"K must be >= 0, got {k_chain_cap}")
    k_chain_cap = min(k_chain_cap, T - 1)

    if int(k_targets.max().item()) == 0:
        return torch.zeros(B, T, device=device, dtype=torch.long)

    predictor = predictor.to(device=device, dtype=z.dtype)
    seg_cost = _precompute_segment_costs_batched(z, predictor, k_chain_cap)  # (B, T, Lmax + 1)

    _, _, L1 = seg_cost.shape
    max_l = min(L1 - 1, k_chain_cap, T - 1)
    max_target_k = int(k_targets.max().item())

    inf = float("inf")
    dp = torch.full((B, T + 1, max_target_k + 1), inf, device=device, dtype=torch.float32)
    dp[:, 0, 0] = 0.0

    # prev_l is sufficient for backtracking:
    # a_prev = a_cur - l - 1, k_prev = k_cur - l
    prev_l = torch.full((B, T + 1, max_target_k + 1), -1, device=device, dtype=torch.int32)

    kk = torch.arange(max_target_k + 1, device=device, dtype=torch.int64)[None, None, :]  # (1,1,Kmax+1)

    for a in range(T):
        L = min(max_l, T - 1 - a) + 1  # includes l=0
        ll = torch.arange(L, device=device, dtype=torch.int64)[None, :, None]  # (1,L,1)

        row_slice = slice(a + 1, a + 1 + L)
        base = dp[:, a, :].unsqueeze(1)  # (B,1,Kmax+1)

        idx = kk - ll  # (1,L,Kmax+1)
        valid = idx >= 0
        idx_clamped = idx.clamp(min=0).expand(B, L, max_target_k + 1)

        gathered = base.expand(B, L, max_target_k + 1).gather(2, idx_clamped)
        gathered = gathered.masked_fill(~valid.expand(B, L, max_target_k + 1), inf)

        cost = seg_cost[:, a, :L].to(torch.float32).unsqueeze(-1)  # (B,L,1)
        cand = gathered + cost

        old = dp[:, row_slice, :]
        better = cand < old
        dp[:, row_slice, :] = torch.where(better, cand, old)

        old_prev = prev_l[:, row_slice, :]
        l_broadcast = ll.expand(B, L, max_target_k + 1).to(torch.int32)
        prev_l[:, row_slice, :] = torch.where(better, l_broadcast, old_prev)

    h = torch.zeros(B, T, device=device, dtype=torch.long)
    for b in range(B):
        k_target = int(k_targets[b].item())
        if not torch.isfinite(dp[b, T, k_target]):
            raise RuntimeError(
                "No feasible DP solution. "
                f"sample={b}, target_length={int(target_length_tensor[b].item())}, K={k_chain_cap}, T={T}."
            )

        a_cur, k_cur = T, k_target
        while not (a_cur == 0 and k_cur == 0):
            l = int(prev_l[b, a_cur, k_cur].item())
            if l < 0:
                raise RuntimeError("Backtracking failed: missing predecessor.")

            a_prev = a_cur - l - 1
            k_prev = k_cur - l
            if a_prev < 0 or k_prev < 0:
                raise RuntimeError("Backtracking failed: invalid predecessor state.")

            if l > 0:
                h[b, a_prev + 1 : a_prev + l + 1] = torch.arange(1, l + 1, device=device, dtype=torch.long)

            a_cur, k_cur = a_prev, k_prev

    return h


@torch.no_grad()
def replace_a_random_multi(
    z: torch.Tensor,
    predictor: torch.nn.Module,
    Ns: Iterable[int],
    seed: Optional[int] = None,
    rand_scores: Optional[torch.Tensor] = None,
) -> list[torch.Tensor]:
    """
    Random replacement snapshots for multiple target kept lengths.

    For each N in ``Ns``, selects exactly ``k = T - N`` random replacement
    positions (excluding t=0), builds segment offsets ``h``, and mixes Elastic Time
    rollouts with ``mix_with_elastic_time_segments``.

    Args:
        z: Latents of shape (B, C, T).
        predictor: Elastic Time predictor module.
        Ns: Target kept lengths.
        seed: Optional random seed.
        rand_scores: Optional deterministic scores with shape (len(Ns), B, T).

    Returns:
        List of mixed latents aligned with ``Ns`` order, each shape (B, C, T).
    """
    if z.ndim != 3:
        raise ValueError(f"z must be (B, C, T), got {tuple(z.shape)}")

    B, C, T = z.shape
    ns_list = list(Ns)
    if len(ns_list) == 0:
        raise ValueError("Ns must be a non-empty iterable")
    for n in ns_list:
        if not (1 <= n <= T):
            raise ValueError(f"Each N must satisfy 1 <= N <= T, got N={n}, T={T}")

    predictor = predictor.to(device=z.device, dtype=z.dtype)

    k_targets = [T - n for n in ns_list]
    max_k = max(k_targets)
    if max_k > T - 1:
        raise ValueError(f"max_k={max_k} is infeasible for T={T}")

    z_outs = [torch.empty_like(z) for _ in ns_list]

    generator = None
    if seed is not None:
        generator = torch.Generator(device=z.device)
        generator.manual_seed(seed)

    if max_k > 0:
        current = z
        z_preds = torch.zeros(B, max_k, C, T, device=z.device, dtype=z.dtype)
        for k in range(max_k):
            current = predictor(current)
            z_preds[:, k, :, :] = current
    else:
        z_preds = torch.zeros(B, 0, C, T, device=z.device, dtype=z.dtype)

    if rand_scores is None:
        rand_scores_tensor = torch.rand((len(ns_list), B, T), generator=generator, device=z.device)
        rand_scores_tensor[:, :, 0] = float("-inf")
    else:
        expected = (len(ns_list), B, T)
        if rand_scores.shape != expected:
            raise ValueError(f"rand_scores must have shape {expected}, got {tuple(rand_scores.shape)}")
        rand_scores_tensor = rand_scores.to(device=z.device)

    for i, k in enumerate(k_targets):
        if k == 0:
            z_outs[i].copy_(z)
            continue

        scores = rand_scores_tensor[i]
        _, idx = scores.topk(k, dim=1)
        mask = torch.zeros(B, T, dtype=torch.bool, device=z.device)
        mask.scatter_(1, idx, True)

        h = binary_to_segment_matrix(mask)
        z_outs[i].copy_(mix_with_elastic_time_segments(z, z_preds, h))

    return list(z_outs)


@torch.no_grad()
def replace_a_random_multi_for_predictor(
    z: torch.Tensor,
    predictor: torch.nn.Module,
    Ns: Iterable[int],
    seed: Optional[int] = None,
    rand_scores: Optional[torch.Tensor] = None,
) -> list[torch.Tensor]:
    """Random multi-target replacement that supports stateless and recurrent predictors."""
    if not _predictor_supports_rollout_state(predictor):
        return replace_a_random_multi(z, predictor, Ns, seed=seed, rand_scores=rand_scores)

    if z.ndim != 3:
        raise ValueError(f"z must be (B, C, T), got {tuple(z.shape)}")

    B, C, T = z.shape
    ns_list = list(Ns)
    if len(ns_list) == 0:
        raise ValueError("Ns must be a non-empty iterable")
    for n in ns_list:
        if not (1 <= n <= T):
            raise ValueError(f"Each N must satisfy 1 <= N <= T, got N={n}, T={T}")

    k_targets = [T - n for n in ns_list]
    max_k = max(k_targets)
    if max_k > T - 1:
        raise ValueError(f"max_k={max_k} is infeasible for T={T}")

    predictor = predictor.to(device=z.device, dtype=z.dtype)

    if max_k > 0:
        current = z
        hidden = predictor.initial_hidden(B, T, device=z.device, dtype=z.dtype)
        z_preds = torch.zeros(B, max_k, C, T, device=z.device, dtype=z.dtype)
        for k in range(max_k):
            pred = predictor(current, hidden)
            if not isinstance(pred, tuple) or len(pred) != 2:
                raise ValueError("predictor must return (pred, h_next) for replace_a_random_multi_for_predictor")
            current, hidden = pred
            z_preds[:, k, :, :] = current
    else:
        z_preds = torch.zeros(B, 0, C, T, device=z.device, dtype=z.dtype)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=z.device)
        generator.manual_seed(seed)

    if rand_scores is None:
        rand_scores_tensor = torch.rand((len(ns_list), B, T), generator=generator, device=z.device)
        rand_scores_tensor[:, :, 0] = float("-inf")
    else:
        expected = (len(ns_list), B, T)
        if rand_scores.shape != expected:
            raise ValueError(f"rand_scores must have shape {expected}, got {tuple(rand_scores.shape)}")
        rand_scores_tensor = rand_scores.to(device=z.device)

    z_outs = [torch.empty_like(z) for _ in ns_list]
    for i, k in enumerate(k_targets):
        if k == 0:
            z_outs[i].copy_(z)
            continue

        scores = rand_scores_tensor[i]
        _, idx = scores.topk(k, dim=1)
        mask = torch.zeros(B, T, dtype=torch.bool, device=z.device)
        mask.scatter_(1, idx, True)

        h = binary_to_segment_matrix(mask)
        z_outs[i].copy_(mix_with_elastic_time_segments(z, z_preds, h))

    return z_outs
