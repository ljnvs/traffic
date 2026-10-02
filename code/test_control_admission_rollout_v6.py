"""Focused tests for the v6 network rollout projector."""

from __future__ import annotations

import unittest

import run_control_admission_rollout_v6 as v6


class RolloutProjectionTests(unittest.TestCase):
    def setUp(self):
        self.network = {
            "road_end": {"r0": "i0", "r1": "i1", "r2": "sink"},
            "controlled": {"i0"},
            "successors": {"r0": ("r1",), "r1": ("r2",), "r2": ()},
            "capacities": {"r0": 20.0, "r1": 20.0, "r2": 20.0},
            "lane_counts": {"r0": 1, "r1": 1, "r2": 1},
            "monitored": {"r0", "r1"},
            "exit_roads": {"r2"},
        }
        self.actions = {"i0": {("r0", "r1")}}

    def test_service_propagates_to_downstream_movement(self):
        road_q, movement_q = v6.project_one_step(
            {"r0": 10.0, "r1": 0.0, "r2": 0.0},
            {("r0", "r1"): 10.0},
            self.actions,
            self.network,
            {},
            switched=set(),
            step_index=0,
        )
        self.assertAlmostEqual(road_q["r0"], 7.5)
        self.assertAlmostEqual(movement_q[("r1", "r2")], 2.5)

    def test_switch_loss_blocks_first_projection_step(self):
        road_q, movement_q = v6.project_one_step(
            {"r0": 10.0, "r1": 0.0, "r2": 0.0},
            {("r0", "r1"): 10.0},
            self.actions,
            self.network,
            {},
            switched={"i0"},
            step_index=0,
        )
        self.assertEqual(road_q["r0"], 10.0)
        self.assertEqual(movement_q[("r0", "r1")], 10.0)

    def test_future_arrival_rate_is_added_before_service(self):
        road_q, movement_q = v6.project_one_step(
            {"r0": 0.0, "r1": 0.0, "r2": 0.0},
            {},
            {"i0": set()},
            self.network,
            {("r0", "r1"): 0.2},
            switched=set(),
            step_index=0,
        )
        self.assertAlmostEqual(road_q["r0"], 1.0)
        self.assertAlmostEqual(movement_q[("r0", "r1")], 1.0)


if __name__ == "__main__":
    unittest.main()
