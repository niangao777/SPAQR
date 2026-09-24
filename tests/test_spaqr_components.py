from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from layers.SPAQR import (  # noqa: E402
    MultiScaleSupportFieldEncoder,
    PeriodAwareFourier,
    QueryConditionedSupportUnfoldingDecoder,
    budgeted_variable_rate_patching,
    pack_fixed_budget_supports,
)


class FixedBudgetTokenizerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.x = torch.randn(2, 6, 3)
        self.times = torch.linspace(0.0, 0.8, 6).repeat(2, 1)
        self.mask = torch.ones_like(self.x)

    def test_fixed_budget_is_respected_and_observations_are_conserved(self) -> None:
        values, times, meta, support_alive, slot_alive, token_counts = (
            budgeted_variable_rate_patching(
                self.x,
                self.times,
                self.mask,
                max_tokens_per_variable=4,
                total_token_budget=6,
            )
        )
        self.assertEqual(values.shape, times.shape)
        self.assertEqual(meta.shape[-1], 6)
        self.assertTrue(torch.equal(token_counts.sum(dim=-1), torch.tensor([6, 6])))
        self.assertTrue(torch.equal(slot_alive.sum(dim=(1, 2, 3)), torch.tensor([18.0, 18.0])))
        self.assertTrue(torch.equal(support_alive.sum(dim=(1, 2)), torch.tensor([6.0, 6.0])))

    def test_adaptive_or_infeasible_budgets_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed positive"):
            budgeted_variable_rate_patching(
                self.x, self.times, self.mask, total_token_budget=0
            )
        with self.assertRaisesRegex(ValueError, "active variables"):
            budgeted_variable_rate_patching(
                self.x, self.times, self.mask, total_token_budget=2
            )


class QueryConditionedSupportUnfoldingPathTest(unittest.TestCase):
    def test_forward_attention_and_backward(self) -> None:
        torch.manual_seed(11)
        batch, length, variables = 2, 7, 3
        x = torch.randn(batch, length, variables)
        times = torch.linspace(0.0, 0.75, length).repeat(batch, 1)
        mask = torch.ones_like(x)
        support = budgeted_variable_rate_patching(
            x,
            times,
            mask,
            max_tokens_per_variable=4,
            total_token_budget=6,
        )
        values, support_times, meta, alive, slot_alive, _ = support

        fourier = PeriodAwareFourier(num_geometric=2)
        support_encoder = MultiScaleSupportFieldEncoder(
            n_variables=variables,
            d_model=16,
            shared_fourier=fourier,
            var_embed_dim=4,
        )
        decoder = QueryConditionedSupportUnfoldingDecoder(
            n_variables=variables,
            d_model=16,
            shared_fourier=fourier,
            n_heads=2,
            n_layers=2,
            dropout=0.0,
            var_embed_dim=4,
        )

        support_states = support_encoder(
            values, support_times, meta, alive, slot_alive
        )
        fine_scale, coarse_scale = support_encoder.time_scales()
        self.assertGreater(coarse_scale.item(), fine_scale.item())
        self.assertGreater(fine_scale.item(), 0.0)
        (
            state_memory,
            memory_meta,
            memory_alive,
            source_variable,
            packed_index,
        ) = pack_fixed_budget_supports(
            support_states,
            meta,
            alive,
        )
        (
            prediction,
            coarse_attention,
            support_attention,
            state_attention,
        ) = decoder(
            query_t=torch.tensor([[0.8, 0.9], [0.82, 0.92]]),
            state_memory=state_memory,
            memory_meta=memory_meta,
            memory_alive=memory_alive,
            source_variable=source_variable,
            return_attention=True,
        )

        self.assertEqual(prediction.shape, (batch, 2, variables))
        self.assertEqual(coarse_attention.shape, (2, batch, variables, 2, 6))
        self.assertEqual(support_attention.shape, (batch, variables, 2, 6))
        self.assertEqual(state_attention.shape, (batch, variables, 2, 6, 4))
        self.assertEqual(support_states.shape, (batch, variables, 4, 4, 16))
        self.assertEqual(packed_index.shape, (batch, 6))
        self.assertTrue(torch.equal(memory_alive.sum(dim=-1), torch.tensor([6.0, 6.0])))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(state_attention).all())
        self.assertTrue(torch.allclose(
            support_attention.sum(dim=-1),
            torch.ones(batch, variables, 2),
            atol=1e-5,
        ))
        self.assertTrue(torch.allclose(
            state_attention.sum(dim=-1),
            torch.ones(batch, variables, 2, 6),
            atol=1e-5,
        ))
        prediction.square().mean().backward()
        self.assertIsNotNone(support_encoder.raw_fine_scale.grad)
        self.assertIsNotNone(support_encoder.raw_scale_gap.grad)
        self.assertIsNotNone(
            support_encoder.fine_forward_scan.update_gate.weight.grad
        )
        self.assertIsNotNone(
            decoder.support_unfolding.state_type_bias.weight.grad
        )


if __name__ == "__main__":
    unittest.main()
