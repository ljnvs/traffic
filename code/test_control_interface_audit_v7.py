"""Focused unit tests for the corrected v7 control interface."""

from __future__ import annotations

import unittest

import run_control_interface_audit_v7 as v7


class ControlInterfaceAuditTests(unittest.TestCase):
    def test_exact_oracle_excludes_current_step_and_maps_future_events(self):
        arrivals = {
            10: [(('a', 'b'), 0)],
            11: [(('a', 'b'), 1), (('a', 'c'), 2)],
        }
        oracle = v7.exact_event_oracle(arrivals, 10, 15)
        self.assertEqual(oracle[0], {})
        self.assertAlmostEqual(oracle[1][('a', 'b')], 0.2)
        self.assertAlmostEqual(oracle[1][('a', 'c')], 0.2)

    def test_minute_forecast_uses_previous_origin_at_minute_boundary(self):
        inputs = {
            "movement_pairs": [["a", "b"]],
            "forecast": {"419": [[60.0], [120.0], [180.0]]},
        }
        rates, audit = v7.aligned_minute_forecast(inputs, 420, 0, 90)
        self.assertEqual(rates[0], {})
        self.assertEqual(audit["forecast_origin_minute"], 419)
        self.assertFalse(audit["uses_current_minute_lag0"])
        self.assertAlmostEqual(rates[1][("a", "b")], 1.0)
        self.assertAlmostEqual(rates[12][("a", "b")], 2.0)

    def test_minute_forecast_switches_row_after_half_minute(self):
        inputs = {
            "movement_pairs": [["a", "b"]],
            "forecast": {"419": [[60.0], [120.0], [180.0]]},
        }
        rates, audit = v7.aligned_minute_forecast(inputs, 420, 30, 60)
        self.assertEqual(audit["projection_row_indices"][:2], [None, 0])
        self.assertAlmostEqual(rates[5][("a", "b")], 1.0)
        self.assertAlmostEqual(rates[6][("a", "b")], 2.0)

    def test_effective_actions_deduplicate_equal_movement_sets(self):
        phases = [
            {"phase_index": 0, "movements": [], "served_entry_indices": []},
            {"phase_index": 1, "movements": [("a", "b", 1)], "served_entry_indices": []},
            {"phase_index": 2, "movements": [("a", "b", 2)], "served_entry_indices": []},
        ]
        rows, summary = v7.action_space_audit({"i": phases})
        self.assertEqual(rows[0]["distinct_nonempty_action_count"], 1)
        self.assertEqual(summary["intersections_with_control_freedom"], 0)

    def test_positive_control_and_projection_consistency(self):
        positive = v7.synthetic_positive_control()
        consistency = v7.projection_consistency_checks()
        self.assertTrue(positive["passed"])
        self.assertLess(
            positive["oracle_selected_cost_under_oracle"],
            positive["no_future_choice_cost_under_oracle"],
        )
        self.assertTrue(consistency["single_movement_match"])
        self.assertTrue(consistency["competing_movement_difference_expected"])


if __name__ == "__main__":
    unittest.main()
