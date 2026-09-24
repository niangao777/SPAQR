from __future__ import annotations

import torch
import torch.nn as nn
from einops import repeat
from torch import Tensor

from layers.SPAQR import (
    MultiScaleSupportFieldEncoder,
    PeriodAwareFourier,
    QueryConditionedSupportUnfoldingDecoder,
    budgeted_variable_rate_patching,
    pack_fixed_budget_supports,
)
from utils.ExpConfigs import ExpConfigs
from utils.globals import logger


class Model(nn.Module):
    """Fixed-budget query-conditioned support unfolding model."""

    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.configs = configs
        self.n_variables = configs.enc_in
        self.d_model = getattr(configs, "spaqr_d_model", 128)
        self.var_embed_dim = getattr(configs, "spaqr_var_embed_dim", 16)
        self.n_heads = getattr(configs, "spaqr_n_heads", 4)
        self.dropout = getattr(configs, "spaqr_dropout", 0.1)
        self.decoder_layers = getattr(configs, "spaqr_decoder_layers", 4)

        self.max_tokens_per_variable = getattr(configs, "spaqr_p_target", 8)
        self.total_token_budget = getattr(configs, "spaqr_token_budget", None)
        if self.total_token_budget is None or self.total_token_budget <= 0:
            raise ValueError(
                "spaqr_token_budget must be explicitly set to a positive fixed budget"
            )
        self.target_patch_mass = getattr(configs, "spaqr_min_patch_mass", 4)
        self.count_penalty = getattr(configs, "spaqr_count_penalty", 1.0)
        self.span_penalty = getattr(configs, "spaqr_span_penalty", 1.0)
        self.gap_penalty = getattr(configs, "spaqr_gap_penalty", 1.0)

        self.shared_fourier = PeriodAwareFourier(
            num_geometric=getattr(configs, "spaqr_num_geometric_freq", 6),
            base_freq=getattr(configs, "spaqr_fourier_base_freq", 1.0),
            scale_factor=getattr(configs, "spaqr_fourier_scale_factor", 1.5),
        )
        self.support_encoder = MultiScaleSupportFieldEncoder(
            n_variables=self.n_variables,
            d_model=self.d_model,
            shared_fourier=self.shared_fourier,
            var_embed_dim=self.var_embed_dim,
        )
        self.decoder = QueryConditionedSupportUnfoldingDecoder(
            n_variables=self.n_variables,
            d_model=self.d_model,
            shared_fourier=self.shared_fourier,
            n_heads=self.n_heads,
            n_layers=self.decoder_layers,
            dropout=self.dropout,
            var_embed_dim=self.var_embed_dim,
        )
        self.debug_print = bool(getattr(configs, "spaqr_debug", 0))
        self._debug_batch_count = 0

    def forward(
        self,
        x: Tensor,
        x_mark: Tensor | None = None,
        x_mask: Tensor | None = None,
        y: Tensor | None = None,
        y_mark: Tensor | None = None,
        y_mask: Tensor | None = None,
        exp_stage: str = "train",
        **kwargs,
    ):
        batch_size, seq_len, n_variables = x.shape
        y_len = self.configs.pred_len if self.configs.pred_len != 0 else seq_len
        return_attention = bool(kwargs.get("return_attention_vis", False))

        if x_mark is None:
            x_mark = repeat(
                torch.arange(seq_len, dtype=x.dtype, device=x.device) / max(seq_len, 1),
                "L -> B L 1", B=batch_size,
            )
        if x_mask is None:
            x_mask = torch.ones_like(x)
        if y is None:
            logger.warning("y is missing. This is only reasonable when testing FLOPs.")
            y = torch.ones((batch_size, y_len, n_variables), dtype=x.dtype, device=x.device)
        if y_mark is None:
            y_mark = repeat(
                torch.arange(y.shape[1], dtype=y.dtype, device=y.device) / max(y.shape[1], 1),
                "L -> B L 1", B=batch_size,
            )
        if y_mask is None:
            y_mask = torch.ones_like(y)

        observed_tp = x_mark[:, :, 0].clamp(0, 1)
        query_tp = y_mark[:, :, 0].clamp(0, 1)
        (
            support_values,
            support_times,
            support_meta,
            support_alive,
            slot_alive,
            token_counts,
        ) = budgeted_variable_rate_patching(
            x=x,
            observed_tp=observed_tp,
            x_mask=x_mask,
            max_tokens_per_variable=self.max_tokens_per_variable,
            total_token_budget=self.total_token_budget,
            target_patch_mass=self.target_patch_mass,
            count_penalty=self.count_penalty,
            span_penalty=self.span_penalty,
            gap_penalty=self.gap_penalty,
        )
        support_states = self.support_encoder(
            support_values, support_times, support_meta, support_alive, slot_alive
        )
        (
            state_memory,
            memory_meta,
            memory_alive,
            source_variable,
            packed_support_index,
        ) = pack_fixed_budget_supports(
            support_states,
            support_meta,
            support_alive,
        )
        decoder_output = self.decoder(
            query_t=query_tp,
            state_memory=state_memory,
            memory_meta=memory_meta,
            memory_alive=memory_alive,
            source_variable=source_variable,
            return_attention=return_attention,
        )
        if return_attention:
            (
                prediction,
                attention_maps,
                unfolded_support_attention,
                support_state_attention,
            ) = decoder_output
        else:
            prediction = decoder_output
            attention_maps = None
            unfolded_support_attention = None
            support_state_attention = None

        if self.debug_print and self._debug_batch_count < 3:
            active = (token_counts > 0).sum(dim=-1).float().mean().item()
            total = token_counts.sum(dim=-1).float().mean().item()
            minimum = token_counts[token_counts > 0].min().item() if (token_counts > 0).any() else 0
            maximum = token_counts.max().item()
            fine_scale, coarse_scale = self.support_encoder.time_scales()
            logger.info(
                "Variable-rate tokenizer: active_vars=%.2f, tokens=%.2f, "
                "tokens_per_nonempty_var=[%d,%d], support_time_scales=[%.4f,%.4f]",
                active,
                total,
                minimum,
                maximum,
                fine_scale.item(),
                coarse_scale.item(),
            )
            self._debug_batch_count += 1

        f_dim = -1 if self.configs.features == "MS" else 0
        if self.configs.task_name in ["long_term_forecast", "short_term_forecast"]:
            output = {
                "pred": prediction[:, :, f_dim:],
                "true": y[:, :, f_dim:],
                "mask": y_mask[:, :, f_dim:],
            }
        elif self.configs.task_name == "imputation":
            output = {"pred": prediction, "true": y, "mask": y_mask}
        else:
            raise ValueError(f"Unsupported task_name={self.configs.task_name!r}")

        if return_attention:
            fine_scale, coarse_scale = self.support_encoder.time_scales()
            output.update({
                "attn_maps": attention_maps,
                "unfolded_support_attn_maps": unfolded_support_attention,
                "support_state_attn_maps": support_state_attention,
                "support_fine_time_scale": fine_scale.detach(),
                "support_coarse_time_scale": coarse_scale.detach(),
                "packed_support_index": packed_support_index,
                "support_alive": support_alive,
                "support_meta": support_meta,
                "support_times": support_times,
                "slot_alive": slot_alive,
                "token_counts": token_counts,
                "query_t": query_tp,
                "x": x,
                "x_mark": x_mark,
                "x_mask": x_mask,
                "y_mark": y_mark,
            })
        return output
