from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class PeriodAwareFourier(nn.Module):
    """Continuous-time Fourier features shared by encoder and decoder."""

    def __init__(self, num_geometric: int = 6, base_freq: float = 1.0,
                 scale_factor: float = 1.5, include_input: bool = True):
        super().__init__()
        bands = sorted({base_freq * scale_factor**k for k in range(num_geometric)})
        self.register_buffer("bands", torch.tensor(bands, dtype=torch.float32))
        self.include_input = include_input
        self.out_dim = 2 * len(bands) + int(include_input)

    def forward(self, t: Tensor) -> Tensor:
        frequency = self.bands.to(dtype=t.dtype)
        frequency = frequency.view(*([1] * (t.dim() - 1)), -1)
        phase = 2.0 * math.pi * t * frequency
        features = [torch.sin(phase), torch.cos(phase)]
        if self.include_input:
            features.insert(0, t)
        return torch.cat(features, dim=-1)


def _masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Softmax that returns exact zeros for an all-masked row."""
    weights = torch.softmax(logits.masked_fill(~mask, -1e4), dim=dim)
    weights = weights * mask.to(dtype=logits.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp(min=1e-8)


@torch.no_grad()
def budgeted_variable_rate_patching(
    x: Tensor,
    observed_tp: Tensor,
    x_mask: Tensor,
    max_tokens_per_variable: int = 8,
    total_token_budget: int | None = None,
    target_patch_mass: int = 4,
    count_penalty: float = 1.0,
    span_penalty: float = 1.0,
    gap_penalty: float = 1.0,
    k_min: int = 1,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build hard variable-rate supports under a sample-level token budget.

    Every observation belongs to exactly one contiguous support. Each non-empty
    variable receives one token first. Remaining tokens compete globally based
    on the reduction in count, temporal-span, and internal-gap distortion.
    Batch statistics affect tensor padding only, never a sample's partition.

    ``total_token_budget`` is a required positive, absolute per-sample cap. It
    must be large enough to give every observed variable one support; the
    function never silently expands it from sample statistics.

    Returns values, times, six-dimensional geometry metadata, support mask,
    slot mask, and the variable-specific token counts.
    """
    if max_tokens_per_variable <= 0:
        raise ValueError("max_tokens_per_variable must be positive")
    if total_token_budget is None or total_token_budget <= 0:
        raise ValueError("total_token_budget must be a fixed positive integer")
    if target_patch_mass <= 0:
        raise ValueError("target_patch_mass must be positive")
    if min(count_penalty, span_penalty, gap_penalty) < 0:
        raise ValueError("tokenizer penalties must be non-negative")

    B, L, N = x.shape
    P = max_tokens_per_variable
    dtype, device = x.dtype, x.device
    mask_bnl = x_mask.permute(0, 2, 1) > 0
    value_bnl = x.permute(0, 2, 1)
    time_bnl = observed_tp[:, None, :].expand(B, N, L)

    inf = torch.full_like(time_bnl, float("inf"))
    order = torch.where(mask_bnl, time_bnl, inf).argsort(dim=-1)
    values = torch.gather(value_bnl, -1, order)
    times = torch.gather(time_bnl, -1, order)
    valid = torch.gather(mask_bnl, -1, order)
    values = torch.where(valid, values, torch.zeros_like(values))
    times = torch.where(valid, times, torch.zeros_like(times))

    n_obs = valid.sum(dim=-1).long()
    has_obs = n_obs > 0
    token_counts = has_obs.long()
    active_variables = has_obs.sum(dim=-1)
    max_possible = torch.minimum(n_obs, torch.full_like(n_obs, P)).sum(dim=-1)
    if (active_variables > total_token_budget).any():
        required = int(active_variables.max().item())
        raise ValueError(
            "total_token_budget is smaller than the number of active variables "
            f"in a sample (budget={total_token_budget}, required>={required})"
        )
    sample_budget = torch.full_like(active_variables, total_token_budget)
    sample_budget = torch.minimum(sample_budget, max_possible)

    rank = torch.arange(L, device=device).view(1, 1, L)
    segment_start = torch.zeros(B, N, L, device=device, dtype=torch.long)
    segment_end = (n_obs - 1).clamp(min=0).unsqueeze(-1).expand(B, N, L).clone()
    overall_start = times[..., 0]
    last_rank = (n_obs - 1).clamp(min=0).unsqueeze(-1)
    overall_end = torch.gather(times, -1, last_rank).squeeze(-1)
    overall_span = (overall_end - overall_start).clamp(min=0.0)
    typical_gap = overall_span / (n_obs - 1).clamp(min=1).to(dtype)
    typical_gap = typical_gap.clamp(min=1e-4)

    remaining = sample_budget - token_counts.sum(dim=-1)
    max_steps = int(remaining.max().item()) if remaining.numel() else 0
    if L > 1:
        cut_rank = torch.arange(L - 1, device=device).view(1, 1, L - 1)
        batch_index = torch.arange(B, device=device)
        variable_grid = torch.arange(N, device=device).view(1, N, 1)
        rank_grid = torch.arange(L, device=device).view(1, 1, L)

        for step in range(max_steps):
            start_at_cut = segment_start[..., :-1]
            end_at_cut = segment_end[..., :-1]
            left_mass = cut_rank - start_at_cut + 1
            right_mass = end_at_cut - cut_rank
            start_time = torch.gather(times, -1, start_at_cut.clamp(0, L - 1))
            end_time = torch.gather(times, -1, end_at_cut.clamp(0, L - 1))
            left_end_time, right_start_time = times[..., :-1], times[..., 1:]
            current_span = (end_time - start_time).clamp(min=0.0)
            left_span = (left_end_time - start_time).clamp(min=0.0)
            right_span = (end_time - right_start_time).clamp(min=0.0)
            boundary_gap = (right_start_time - left_end_time).clamp(min=0.0)

            mass_gain = 2.0 * left_mass.to(dtype) * right_mass.to(dtype)
            mass_gain = mass_gain / float(target_patch_mass**2)
            span_scale = overall_span.unsqueeze(-1).clamp(min=1e-4)
            span_gain = (
                current_span.square() - left_span.square() - right_span.square()
            ).clamp(min=0.0) / span_scale.square()
            gap_gain = (boundary_gap / typical_gap.unsqueeze(-1)).square().clamp(max=100.0)
            gain = count_penalty * mass_gain + span_penalty * span_gain + gap_penalty * gap_gain
            can_split = (
                (cut_rank < (n_obs - 1).unsqueeze(-1))
                & (right_mass > 0)
                & (token_counts < P).unsqueeze(-1)
            )
            gain = gain.masked_fill(~can_split, -1e9)
            flat_gain = gain.reshape(B, N * (L - 1))
            best_gain, best_flat = flat_gain.max(dim=-1)
            best_variable = torch.div(best_flat, L - 1, rounding_mode="floor")
            best_cut = torch.remainder(best_flat, L - 1)
            allocate = (step < remaining) & (best_gain > -1e8)

            chosen_start = segment_start[batch_index, best_variable, best_cut]
            chosen_end = segment_end[batch_index, best_variable, best_cut]
            selected_variable = variable_grid == best_variable[:, None, None]
            inside_segment = (
                (rank_grid >= chosen_start[:, None, None])
                & (rank_grid <= chosen_end[:, None, None])
            )
            selected = selected_variable & inside_segment & allocate[:, None, None]
            left_side = rank_grid <= best_cut[:, None, None]
            segment_end = torch.where(
                selected & left_side,
                best_cut[:, None, None].expand_as(segment_end), segment_end,
            )
            segment_start = torch.where(
                selected & ~left_side,
                (best_cut + 1)[:, None, None].expand_as(segment_start), segment_start,
            )
            token_counts[batch_index[allocate], best_variable[allocate]] += 1

    is_start = rank == segment_start
    support_id = is_start.long().cumsum(dim=-1) - 1
    support_id = torch.where(valid, support_id, torch.zeros_like(support_id))
    within_support = torch.where(valid, rank - segment_start, torch.zeros_like(rank))
    group_sizes = torch.zeros(B, N, P, device=device, dtype=torch.long)
    group_sizes.scatter_add_(-1, support_id.clamp(0, P - 1), valid.long())
    K = max(k_min, int(group_sizes.max().item()) if group_sizes.numel() else k_min)

    flat_index = support_id.clamp(0, P - 1) * K + within_support.clamp(0, K - 1)
    out_values = x.new_zeros(B, N, P * K)
    out_times = x.new_zeros(B, N, P * K)
    out_alive = x.new_zeros(B, N, P * K)
    out_values.scatter_add_(-1, flat_index, torch.where(valid, values, torch.zeros_like(values)))
    out_times.scatter_add_(-1, flat_index, torch.where(valid, times, torch.zeros_like(times)))
    out_alive.scatter_add_(-1, flat_index, valid.to(dtype))
    out_values = out_values.view(B, N, P, K)
    out_times = out_times.view(B, N, P, K)
    slot_alive = out_alive.view(B, N, P, K)
    support_alive = (slot_alive.sum(dim=-1) > 0).to(dtype)

    count = slot_alive.sum(dim=-1)
    center = (out_times * slot_alive).sum(dim=-1) / count.clamp(min=1.0)
    first = out_times[..., 0]
    last_index = (count.long() - 1).clamp(min=0).unsqueeze(-1)
    last = torch.gather(out_times, -1, last_index).squeeze(-1)
    span = (last - first).clamp(min=0.0)
    if K > 1:
        slot_gap = (out_times[..., 1:] - out_times[..., :-1]).clamp(min=0.0)
        adjacent = (slot_alive[..., 1:] > 0) & (slot_alive[..., :-1] > 0)
        max_gap = torch.where(adjacent, slot_gap, torch.zeros_like(slot_gap)).amax(dim=-1)
    else:
        max_gap = torch.zeros_like(center)
    meta = torch.stack([
        center, first, last, torch.log1p(count),
        torch.log1p(1e3 * span), torch.log1p(1e3 * max_gap),
    ], dim=-1) * support_alive.unsqueeze(-1)
    return out_values, out_times, meta, support_alive, slot_alive, token_counts


