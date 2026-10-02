"""Tests for v8 causal forecast residualization and joint search."""

from __future__ import annotations

import unittest

import run_control_corrected_rollout_v8 as v8


class CorrectedRolloutTests(unittest.TestCase):
    def test_observed_counts_include_current_step_only_within_minute(self):
        arrivals = {
            11: [(('old', 'x'), 0)],
            12: [(('a', 'b'), 1)],
            13: [(('a', 'b'), 2), (('a', 'c'), 3)],
        }
        counts = v8.observed_current_minute_counts(arrivals, 13)
        self.assertEqual(counts[("a", "b")], 2.0)
        self.assertEqual(counts[("a", "c")], 1.0)
        self.assertNotIn(("old", "x"), counts)

    def test_current_minute_forecast_is_residualized(self):
        pairs = [["a", "b"]]
        inputs = {"movement_pairs": pairs, "forecast": {"419": [[12.0], [24.0]]}}
        arrivals = {360: [(('a', 'b'), 0)], 361: [(('a', 'b'), 1)]}
        rates = v8.causal_forecast_rates(inputs, arrivals, 361, 390)
        self.assertEqual(rates[0], {})
        # 10 predicted vehicles remain over ten future 5-second slots.
        self.assertAlmostEqual(rates[1][("a", "b")], 0.2)

    def test_half_minute_crosses_to_next_forecast_row(self):
        inputs = {"movement_pairs": [["a", "b"]], "forecast": {"419": [[6.0], [12.0]]}}
        rates = v8.causal_forecast_rates(inputs, {}, 366, 390)
        self.assertAlmostEqual(rates[1][("a", "b")], 0.24)  # 6 / unobserved 25 seconds
        self.assertAlmostEqual(rates[6][("a", "b")], 0.2)  # next minute 12 / 60


if __name__ == "__main__":
    unittest.main()
