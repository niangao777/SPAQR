from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from layers.SPAQR.components import PeriodAwareFourier, _masked_softmax


class TimeScaleStateScan(nn.Module):
    """Order-preserving state transport at an externally supplied time scale."""

    def __init__(self, d_model: int):
        super().__init__()
        state_input_dim = 2 * d_model + 1
        self.candidate = nn.Linear(state_input_dim, d_model)
        self.update_gate = nn.Linear(state_input_dim, d_model)

    def forward(
        self,
        observations: Tensor,
        times: Tensor,
        alive: Tensor,
        time_scale: Tensor,
        reverse: bool = False,
    ) -> Tensor:
        *leading, length, d_model = observations.shape
        hidden = observations.new_zeros(*leading, d_model)
        reference_time = times.new_zeros(*leading)
        seen = torch.zeros(*leading, dtype=torch.bool, device=times.device)
        indices = range(length - 1, -1, -1) if reverse else range(length)
        scale = time_scale.to(dtype=times.dtype).clamp(min=1e-4)

        for index in indices:
            valid = alive[..., index] > 0
            current_time = times[..., index]
            delta_time = torch.where(
                seen,
                (current_time - reference_time).abs(),
                torch.zeros_like(current_time),
            )
            retention = torch.exp(-delta_time / scale).unsqueeze(-1)
            retained_hidden = hidden * retention
            delta_feature = torch.log1p(delta_time / scale).unsqueeze(-1)
            state_input = torch.cat([
                observations[..., index, :],
                retained_hidden,
                delta_feature,
            ], dim=-1)
            candidate = torch.tanh(self.candidate(state_input))
            gate = torch.sigmoid(self.update_gate(state_input))
            transported = (
                (1.0 - gate) * retained_hidden + gate * candidate
            )
            hidden = torch.where(valid.unsqueeze(-1), transported, hidden)
            reference_time = torch.where(valid, current_time, reference_time)
            seen = seen | valid

        return hidden


class MultiScaleSupportFieldEncoder(nn.Module):
    """Encode one fixed-budget support as four fixed-capacity field states.

    The representation capacity is identical for every support: one coarse
    state, two fine boundary states, and one coarse-to-fine transition state.
    No additional support tokens or alternate partitions are constructed.
    """

    n_support_states = 4
    coarse_state_index = 0

    def __init__(
        self,
        n_variables: int,
        d_model: int,
        shared_fourier: PeriodAwareFourier,
        var_embed_dim: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.fourier = shared_fourier
        observation_dim = 1 + 2 * self.fourier.out_dim
        self.observation_encoder = nn.Sequential(
            nn.Linear(observation_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.coarse_forward_scan = TimeScaleStateScan(d_model)
        self.coarse_backward_scan = TimeScaleStateScan(d_model)
        self.fine_forward_scan = TimeScaleStateScan(d_model)
        self.fine_backward_scan = TimeScaleStateScan(d_model)

        # Constrain coarse_scale > fine_scale while allowing both to learn.
        fine_fraction = 0.1 / 0.5
        self.raw_fine_scale = nn.Parameter(torch.tensor(
            math.log(fine_fraction / (1.0 - fine_fraction))
        ))
        self.raw_scale_gap = nn.Parameter(torch.tensor(
            math.log(math.expm1(0.9))
        ))

        self.geometry_encoder = nn.Sequential(
            nn.Linear(self.fourier.out_dim + 5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.variable_embedding = nn.Embedding(n_variables, var_embed_dim)
        self.variable_projection = nn.Linear(var_embed_dim, d_model)

        self.coarse_projection = self._state_projection(4 * d_model, d_model)
        self.left_projection = self._state_projection(4 * d_model, d_model)
        self.right_projection = self._state_projection(4 * d_model, d_model)
        self.transition_projection = self._state_projection(
            6 * d_model + 7, d_model
        )

    @staticmethod
    def _state_projection(input_dim: int, d_model: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def time_scales(self) -> tuple[Tensor, Tensor]:
        fine_scale = 0.5 * torch.sigmoid(self.raw_fine_scale) + 1e-3
        coarse_scale = fine_scale + functional.softplus(self.raw_scale_gap)
        return fine_scale, coarse_scale

    def forward(
        self,
        support_values: Tensor,
        support_times: Tensor,
        support_meta: Tensor,
        support_alive: Tensor,
        slot_alive: Tensor,
    ) -> Tensor:
        _, n_variables, _, length = support_values.shape
        center = support_meta[..., 0]
        absolute_time = self.fourier(support_times.unsqueeze(-1))
        relative_time = self.fourier(
            (support_times - center.unsqueeze(-1)).unsqueeze(-1)
        )
        observation_features = torch.cat([
            support_values.unsqueeze(-1),
            absolute_time,
            relative_time,
        ], dim=-1)
        observation_states = self.observation_encoder(observation_features)

        fine_scale, coarse_scale = self.time_scales()
        coarse_right = self.coarse_forward_scan(
            observation_states,
            support_times,
            slot_alive,
            coarse_scale,
            reverse=False,
        )
        coarse_left = self.coarse_backward_scan(
            observation_states,
            support_times,
            slot_alive,
            coarse_scale,
            reverse=True,
        )
        fine_right = self.fine_forward_scan(
            observation_states,
            support_times,
            slot_alive,
            fine_scale,
            reverse=False,
        )
        fine_left = self.fine_backward_scan(
            observation_states,
            support_times,
            slot_alive,
            fine_scale,
            reverse=True,
        )

        geometry_features = torch.cat([
            self.fourier(center.unsqueeze(-1)),
            support_meta[..., 1:],
        ], dim=-1)
        geometry_state = self.geometry_encoder(geometry_features)
        variable_ids = torch.arange(n_variables, device=support_values.device)
        variable_state = self.variable_projection(
            self.variable_embedding(variable_ids)
        ).view(1, n_variables, 1, self.d_model)
        variable_state = variable_state.expand_as(coarse_left)

        coarse_state = self.coarse_projection(torch.cat([
            coarse_left,
            coarse_right,
            geometry_state,
            variable_state,
        ], dim=-1))
        left_state = self.left_projection(torch.cat([
            fine_left,
            coarse_left,
            geometry_state,
            variable_state,
        ], dim=-1))
        right_state = self.right_projection(torch.cat([
            fine_right,
            coarse_right,
            geometry_state,
            variable_state,
        ], dim=-1))

        count = slot_alive.sum(dim=-1)
        last_index = (count.long() - 1).clamp(
            min=0, max=length - 1
        ).unsqueeze(-1)
        first_value = support_values[..., 0]
        last_value = torch.gather(
            support_values, -1, last_index
        ).squeeze(-1)
        value_delta = last_value - first_value
        weights = slot_alive / count.unsqueeze(-1).clamp(min=1.0)
        mean_time = (weights * support_times).sum(dim=-1, keepdim=True)
        mean_value = (weights * support_values).sum(dim=-1, keepdim=True)
        centered_time = support_times - mean_time
        centered_value = support_values - mean_value
        covariance = (
            slot_alive * centered_time * centered_value
        ).sum(dim=-1)
        time_variance = (
            slot_alive * centered_time.square()
        ).sum(dim=-1)
        slope = covariance / time_variance.clamp(min=1e-6)
        signed_log_slope = torch.sign(slope) * torch.log1p(slope.abs())
        dynamics = torch.stack([
            first_value,
            last_value,
            value_delta,
            signed_log_slope,
            support_meta[..., 3],
            support_meta[..., 4],
            support_meta[..., 5],
        ], dim=-1)
        transition_state = self.transition_projection(torch.cat([
            coarse_left,
            coarse_right,
            fine_left,
            fine_right,
            coarse_right - coarse_left,
            fine_right - fine_left,
            dynamics,
        ], dim=-1))

        state_memory = torch.stack([
            coarse_state,
            left_state,
            right_state,
            transition_state,
        ], dim=-2)
        return state_memory * support_alive.unsqueeze(-1).unsqueeze(-1)


def pack_fixed_budget_supports(
    support_states: Tensor,
    support_meta: Tensor,
    support_alive: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Remove padding while preserving every valid fixed-budget support."""

    batch, n_variables, max_supports, n_states, d_model = support_states.shape
    flat_states = support_states.reshape(
        batch, n_variables * max_supports, n_states, d_model
    )
    flat_meta = support_meta.reshape(batch, n_variables * max_supports, 6)
    flat_alive = support_alive.reshape(batch, n_variables * max_supports)
    original_index = torch.arange(
        n_variables * max_supports, device=support_states.device
    ).view(1, n_variables * max_supports)
    packing_key = torch.where(
        flat_alive > 0,
        original_index,
        original_index + n_variables * max_supports,
    )
    memory_order = packing_key.argsort(dim=-1)
    packed_length = max(
        1, int((flat_alive > 0).sum(dim=-1).max().item())
    )
    memory_order = memory_order[:, :packed_length]
    state_memory = torch.gather(
        flat_states,
        1,
        memory_order.unsqueeze(-1).unsqueeze(-1).expand(
            batch, packed_length, n_states, d_model
        ),
    )
    memory_meta = torch.gather(
        flat_meta,
        1,
        memory_order.unsqueeze(-1).expand(batch, packed_length, 6),
    )
    memory_alive = torch.gather(flat_alive, 1, memory_order)
    source_variable = torch.div(
        memory_order, max_supports, rounding_mode="floor"
    )
    return (
        state_memory,
        memory_meta,
        memory_alive,
        source_variable,
        memory_order,
    )


class GatedStateUpdate(nn.Module):
    """Conditionally replace a state instead of applying a blind residual."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.context_gate = nn.Linear(2 * d_model, d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.feed_forward_gate = nn.Linear(2 * d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.context_gate.bias, -1.0)
        nn.init.constant_(self.feed_forward_gate.bias, -1.0)

    def forward(self, state: Tensor, proposal: Tensor) -> Tensor:
        proposal = self.dropout(proposal)
        gate = torch.sigmoid(
            self.context_gate(torch.cat([state, proposal], dim=-1))
        )
        mixed = self.context_norm(
            (1.0 - gate) * state + gate * proposal
        )
        feed_forward = self.feed_forward(mixed)
        feed_forward_gate = torch.sigmoid(
            self.feed_forward_gate(torch.cat([
                mixed, feed_forward,
            ], dim=-1))
        )
        return self.output_norm(
            (1.0 - feed_forward_gate) * mixed
            + feed_forward_gate * feed_forward
        )


class CoarseGlobalQueryLayer(nn.Module):
    """Read every coarse support state using the prediction target."""

    def __init__(
        self,
        n_variables: int,
        d_model: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_variables = n_variables
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        hidden = max(d_model // 2, 8)
        self.geometry_bias = nn.Sequential(
            nn.Linear(8, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_heads),
        )
        self.variable_relation = nn.Embedding(
            n_variables * n_variables, n_heads
        )
        nn.init.zeros_(self.variable_relation.weight)
        self.dropout = nn.Dropout(dropout)
        self.update = GatedStateUpdate(d_model, dropout)

    def forward(
        self,
        query: Tensor,
        query_time: Tensor,
        target_variable: Tensor,
        memory: Tensor,
        memory_meta: Tensor,
        memory_alive: Tensor,
        source_variable: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, targets, d_model = query.shape
        supports = memory.shape[1]
        projected_query = self.query_projection(query).view(
            batch, targets, self.n_heads, self.head_dim
        ).transpose(1, 2)
        projected_key = self.key_projection(memory).view(
            batch, supports, self.n_heads, self.head_dim
        ).transpose(1, 2)
        projected_value = self.value_projection(memory).view(
            batch, supports, self.n_heads, self.head_dim
        ).transpose(1, 2)
        logits = torch.einsum(
            "bhtd,bhsd->bhts", projected_query, projected_key
        ) / math.sqrt(self.head_dim)
        logits = logits + self._relation_bias(
            query,
            query_time,
            target_variable,
            memory_meta,
            source_variable,
        )
        attention = _masked_softmax(
            logits, memory_alive[:, None, None, :] > 0, dim=-1
        )
        context = torch.einsum(
            "bhts,bhsd->bhtd", self.dropout(attention), projected_value
        )
        context = context.transpose(1, 2).reshape(batch, targets, d_model)
        query = self.update(query, self.output_projection(context))
        return query, attention.mean(dim=1)

    def _relation_bias(
        self,
        query: Tensor,
        query_time: Tensor,
        target_variable: Tensor,
        memory_meta: Tensor,
        source_variable: Tensor,
    ) -> Tensor:
        batch, targets = query_time.shape
        supports = memory_meta.shape[1]
        center = memory_meta[..., 0]
        start = memory_meta[..., 1]
        end = memory_meta[..., 2]
        delta = query_time.unsqueeze(-1) - center.unsqueeze(1)
        interval_distance = torch.maximum(
            start.unsqueeze(1) - query_time.unsqueeze(-1),
            query_time.unsqueeze(-1) - end.unsqueeze(1),
        ).clamp(min=0.0)
        inside = (interval_distance <= 1e-8).to(query.dtype)
        same_variable = (
            target_variable.unsqueeze(-1) == source_variable.unsqueeze(1)
        ).to(query.dtype)
        geometry_features = torch.stack([
            delta,
            delta.abs(),
            interval_distance,
            inside,
            memory_meta[..., 3].unsqueeze(1).expand(batch, targets, supports),
            memory_meta[..., 4].unsqueeze(1).expand(batch, targets, supports),
            memory_meta[..., 5].unsqueeze(1).expand(batch, targets, supports),
            same_variable,
        ], dim=-1)
        bias = self.geometry_bias(geometry_features).permute(0, 3, 1, 2)
        relation_id = target_variable.unsqueeze(-1) * self.n_variables
        relation_id = relation_id + source_variable.unsqueeze(1)
        return bias + self.variable_relation(relation_id).permute(0, 3, 1, 2)


class QueryConditionedSupportUnfolding(nn.Module):
    """Lazily resolve support scale, then select supports for each target."""

    n_state_types = 4

    def __init__(
        self,
        n_variables: int,
        d_model: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_variables = n_variables
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.state_query = nn.Linear(d_model, d_model)
        self.state_key = nn.Linear(d_model, d_model)
        state_hidden = max(d_model // 2, 8)
        self.state_geometry_bias = nn.Sequential(
            nn.Linear(9, state_hidden),
            nn.SiLU(),
            nn.Linear(state_hidden, 1),
        )
        self.state_variable_relation = nn.Embedding(
            n_variables * n_variables, 1
        )
        self.state_type_bias = nn.Embedding(self.n_state_types, 1)
        nn.init.zeros_(self.state_variable_relation.weight)
        nn.init.zeros_(self.state_type_bias.weight)

        self.support_query = nn.Linear(d_model, d_model)
        self.support_key = nn.Linear(d_model, d_model)
        self.state_value = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        support_hidden = max(d_model // 2, 8)
        self.support_geometry_bias = nn.Sequential(
            nn.Linear(8, support_hidden),
            nn.SiLU(),
            nn.Linear(support_hidden, n_heads),
        )
        self.support_variable_relation = nn.Embedding(
            n_variables * n_variables, n_heads
        )
        nn.init.zeros_(self.support_variable_relation.weight)
        self.dropout = nn.Dropout(dropout)
        self.update = GatedStateUpdate(d_model, dropout)

    def forward(
        self,
        query: Tensor,
        query_time: Tensor,
        target_variable: Tensor,
        state_memory: Tensor,
        memory_meta: Tensor,
        memory_alive: Tensor,
        source_variable: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, targets, d_model = query.shape
        supports, state_types = state_memory.shape[1:3]
        if state_types != self.n_state_types:
            raise ValueError(
                f"Expected {self.n_state_types} support states, got {state_types}"
            )

        state_query = self.state_query(query)
        state_key = self.state_key(state_memory)
        state_logits = torch.einsum(
            "btd,bsrd->btsr", state_query, state_key
        ) / math.sqrt(d_model)
        state_logits = state_logits + self._state_relation_bias(
            query,
            query_time,
            target_variable,
            memory_meta,
            source_variable,
        )
        state_mask = memory_alive[:, None, :, None].expand(
            batch, targets, supports, state_types
        ) > 0
        state_attention = _masked_softmax(
            state_logits, state_mask, dim=-1
        )

        coarse_memory = state_memory[:, :, 0, :]
        projected_query = self.support_query(query).view(
            batch, targets, self.n_heads, self.head_dim
        ).transpose(1, 2)
        projected_key = self.support_key(coarse_memory).view(
            batch, supports, self.n_heads, self.head_dim
        ).transpose(1, 2)
        support_logits = torch.einsum(
            "bhtd,bhsd->bhts", projected_query, projected_key
        ) / math.sqrt(self.head_dim)
        support_logits = support_logits + self._support_relation_bias(
            query,
            query_time,
            target_variable,
            memory_meta,
            source_variable,
        )
        # Marginalize the query-conditioned internal states into one support
        # score. Subtracting log(R) keeps the evidence neutral when all four
        # states are equally compatible and avoids rewarding state count.
        state_evidence = (
            torch.logsumexp(state_logits, dim=-1) - math.log(state_types)
        ).unsqueeze(1)
        support_logits = support_logits + state_evidence
        support_attention = _masked_softmax(
            support_logits,
            memory_alive[:, None, None, :] > 0,
            dim=-1,
        )

        projected_value = self.state_value(state_memory).view(
            batch,
            supports,
            state_types,
            self.n_heads,
            self.head_dim,
        ).permute(0, 3, 1, 2, 4)
        joint_attention = (
            support_attention.unsqueeze(-1)
            * state_attention.unsqueeze(1)
        )
        context = torch.einsum(
            "bhtsr,bhsrd->bhtd",
            self.dropout(joint_attention),
            projected_value,
        )
        context = context.transpose(1, 2).reshape(batch, targets, d_model)
        query = self.update(query, self.output_projection(context))
        return query, support_attention.mean(dim=1), state_attention

    def _state_relation_bias(
        self,
        query: Tensor,
        query_time: Tensor,
        target_variable: Tensor,
        memory_meta: Tensor,
        source_variable: Tensor,
    ) -> Tensor:
        batch, targets = query_time.shape
        supports = memory_meta.shape[1]
        center = memory_meta[..., 0]
        start = memory_meta[..., 1]
        end = memory_meta[..., 2]
        anchors = torch.stack([center, start, end, center], dim=-1)
        delta = query_time[:, :, None, None] - anchors[:, None, :, :]
        interval_distance = torch.maximum(
            start.unsqueeze(1) - query_time.unsqueeze(-1),
            query_time.unsqueeze(-1) - end.unsqueeze(1),
        ).clamp(min=0.0)
        inside = (interval_distance <= 1e-8).to(query.dtype)
        same_variable = (
            target_variable.unsqueeze(-1) == source_variable.unsqueeze(1)
        ).to(query.dtype)
        state_types = self.n_state_types
        geometry_features = torch.stack([
            delta,
            delta.abs(),
            interval_distance.unsqueeze(-1).expand(
                batch, targets, supports, state_types
            ),
            inside.unsqueeze(-1).expand(
                batch, targets, supports, state_types
            ),
            memory_meta[..., 3].unsqueeze(1).unsqueeze(-1).expand(
                batch, targets, supports, state_types
            ),
            memory_meta[..., 4].unsqueeze(1).unsqueeze(-1).expand(
                batch, targets, supports, state_types
            ),
            memory_meta[..., 5].unsqueeze(1).unsqueeze(-1).expand(
                batch, targets, supports, state_types
            ),
            same_variable.unsqueeze(-1).expand(
                batch, targets, supports, state_types
            ),
            torch.log1p(
                interval_distance
                / (end - start).clamp(min=1e-4).unsqueeze(1)
            ).unsqueeze(-1).expand(
                batch, targets, supports, state_types
            ),
        ], dim=-1)
        bias = self.state_geometry_bias(geometry_features).squeeze(-1)
        relation_id = target_variable.unsqueeze(-1) * self.n_variables
        relation_id = relation_id + source_variable.unsqueeze(1)
        relation_bias = self.state_variable_relation(relation_id)
        state_ids = torch.arange(state_types, device=query.device)
        type_bias = self.state_type_bias(state_ids).view(
            1, 1, 1, state_types
        )
        return bias + relation_bias + type_bias

    def _support_relation_bias(
        self,
        query: Tensor,
        query_time: Tensor,
        target_variable: Tensor,
        memory_meta: Tensor,
        source_variable: Tensor,
    ) -> Tensor:
        batch, targets = query_time.shape
        supports = memory_meta.shape[1]
        center = memory_meta[..., 0]
        start = memory_meta[..., 1]
        end = memory_meta[..., 2]
        delta = query_time.unsqueeze(-1) - center.unsqueeze(1)
        interval_distance = torch.maximum(
            start.unsqueeze(1) - query_time.unsqueeze(-1),
            query_time.unsqueeze(-1) - end.unsqueeze(1),
        ).clamp(min=0.0)
        inside = (interval_distance <= 1e-8).to(query.dtype)
        same_variable = (
            target_variable.unsqueeze(-1) == source_variable.unsqueeze(1)
        ).to(query.dtype)
        geometry_features = torch.stack([
            delta,
            delta.abs(),
            interval_distance,
            inside,
            memory_meta[..., 3].unsqueeze(1).expand(batch, targets, supports),
            memory_meta[..., 4].unsqueeze(1).expand(batch, targets, supports),
            memory_meta[..., 5].unsqueeze(1).expand(batch, targets, supports),
            same_variable,
        ], dim=-1)
        bias = self.support_geometry_bias(geometry_features).permute(
            0, 3, 1, 2
        )
        relation_id = target_variable.unsqueeze(-1) * self.n_variables
        relation_id = relation_id + source_variable.unsqueeze(1)
        relation_bias = self.support_variable_relation(
            relation_id
        ).permute(0, 3, 1, 2)
        return bias + relation_bias


class QueryConditionedSupportUnfoldingDecoder(nn.Module):
    """Globally retrieve coarse supports, then unfold their internal scale."""

    def __init__(
        self,
        n_variables: int,
        d_model: int,
        shared_fourier: PeriodAwareFourier,
        n_heads: int,
        n_layers: int,
        dropout: float,
        var_embed_dim: int = 16,
    ):
        super().__init__()
        if n_layers < 0:
            raise ValueError("n_layers cannot be negative")
        self.n_variables = n_variables
        self.d_model = d_model
        self.fourier = shared_fourier
        self.query_variable = nn.Embedding(n_variables, var_embed_dim)
        self.query_projection = nn.Sequential(
            nn.Linear(self.fourier.out_dim + var_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.coarse_query_layers = nn.ModuleList([
            CoarseGlobalQueryLayer(
                n_variables, d_model, n_heads, dropout
            )
            for _ in range(n_layers)
        ])
        self.support_unfolding = QueryConditionedSupportUnfolding(
            n_variables, d_model, n_heads, dropout
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        query_t: Tensor,
        state_memory: Tensor,
        memory_meta: Tensor,
        memory_alive: Tensor,
        source_variable: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = state_memory.shape[0]
        n_queries = query_t.shape[1]
        target_variable = torch.arange(
            self.n_variables, device=query_t.device
        ).view(1, self.n_variables, 1).expand(
            batch, self.n_variables, n_queries
        )
        query_time = query_t[:, None, :].expand(
            batch, self.n_variables, n_queries
        )
        query = self.query_projection(torch.cat([
            self.fourier(query_time.unsqueeze(-1)),
            self.query_variable(target_variable),
        ], dim=-1)).reshape(
            batch, self.n_variables * n_queries, self.d_model
        )
        flat_time = query_time.reshape(batch, self.n_variables * n_queries)
        flat_target = target_variable.reshape(
            batch, self.n_variables * n_queries
        )

        coarse_memory = state_memory[:, :, 0, :]
        coarse_attention_maps = []
        for layer in self.coarse_query_layers:
            query, attention = layer(
                query,
                flat_time,
                flat_target,
                coarse_memory,
                memory_meta,
                memory_alive,
                source_variable,
            )
            if return_attention:
                coarse_attention_maps.append(attention.reshape(
                    batch,
                    self.n_variables,
                    n_queries,
                    coarse_memory.shape[1],
                ))

        query, support_attention, state_attention = self.support_unfolding(
            query,
            flat_time,
            flat_target,
            state_memory,
            memory_meta,
            memory_alive,
            source_variable,
        )
        prediction = self.head(query).squeeze(-1)
        prediction = prediction.reshape(
            batch, self.n_variables, n_queries
        ).transpose(1, 2).contiguous()

        if return_attention:
            if coarse_attention_maps:
                coarse_attention = torch.stack(
                    coarse_attention_maps, dim=0
                )
            else:
                coarse_attention = query.new_zeros(
                    0,
                    batch,
                    self.n_variables,
                    n_queries,
                    coarse_memory.shape[1],
                )
            support_attention = support_attention.reshape(
                batch,
                self.n_variables,
                n_queries,
                coarse_memory.shape[1],
            )
            state_attention = state_attention.reshape(
                batch,
                self.n_variables,
                n_queries,
                coarse_memory.shape[1],
                MultiScaleSupportFieldEncoder.n_support_states,
            )
            return (
                prediction,
                coarse_attention,
                support_attention,
                state_attention,
            )
        return prediction
