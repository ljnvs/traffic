"""Capacity-normalized store-and-forward admission screen with switch losses."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import run_control_admission_queue_v1 as v1


SIMULATION_STEP_SECONDS = 5
CONTROL_INTERVAL_SECONDS = 30
MIN_GREEN_SECONDS = 30
SWITCH_LOSS_SECONDS = 5
BETA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def normalized_phase_score(
    phase,
    queues,
    movement_queues,
    capacities,
    lane_counts,
    entry_roads,
    demand,
    beta,
):
    pressure = 0.0
    for start, end, start_lanes in phase["movements"]:
        upstream = movement_queues.get((start, end), 0.0) / capacities[start]
        downstream = queues.get(end, 0.0) / capacities[end]
        lane_share = start_lanes / lane_counts[start]
        pressure += max(0.0, upstream - downstream) * lane_share
    demand_bonus = 0.0
    for index in phase["served_entry_indices"]:
        road = entry_roads[index]
        per_minute_service = (
            v1.SATURATION_FLOW_PER_LANE_PER_SECOND * lane_counts[road] * 60.0
        )
        demand_bonus += demand[index] / per_minute_service
    return pressure + beta * demand_bonus


def simulate(date_dir, roadnet, metadata, mode, beta):
    simulation = metadata["simulation"]
    step_seconds = SIMULATION_STEP_SECONDS
    warmup_steps = simulation["warmup_seconds"] // step_seconds
    evaluation_steps = simulation["evaluation_seconds"] // step_seconds
    total_steps = warmup_steps + evaluation_steps
    inputs = json.loads((date_dir / "demand_inputs.json").read_text(encoding="utf-8"))
    arrivals, input_vehicles = v1.load_arrivals(
        date_dir / "flow_window.json", step_seconds
    )
    (
        roads,
        intersections,
        controllers,
        uncontrolled,
        capacities,
        lane_counts,
        monitored_roads,
    ) = v1.build_network(roadnet, metadata)

    state = {road: {} for road in roads}
    exit_roads = set(metadata["exit_roads"])
    entry_roads = metadata["entry_roads"]
    queue_vehicle_seconds = 0.0
    vehicle_seconds = 0.0
    spillback_road_seconds = 0.0
    max_occupancy = 0.0
    throughput = 0.0
    completed = 0.0
    completed_travel_time = 0.0
    phase_switches = 0
    active_phase = {}
    phase_started_second = {}
    lost_until_second = defaultdict(lambda: -1)

    for step in range(total_steps):
        second = step * step_seconds
        in_evaluation = step >= warmup_steps
        for route, sequence in arrivals.get(step, []):
            key = (route, 0, step, sequence)
            state[route[0]][key] = state[route[0]].get(key, 0.0) + 1.0

        queues = {road: sum(packets.values()) for road, packets in state.items()}
        movement_queues = defaultdict(float)
        for road, packets in state.items():
            for (route, position, _, _), mass in packets.items():
                if position < len(route) - 1:
                    movement_queues[(road, route[position + 1])] += mass

        chosen_phases = {}
        for intersection_id, phases in controllers.items():
            if not in_evaluation or mode == "FixedTime":
                chosen = v1.fixed_phase_at_second(phases, second)
                active_phase[intersection_id] = chosen
                phase_started_second[intersection_id] = second
                lost_until_second[intersection_id] = -1
            else:
                if intersection_id not in active_phase:
                    current = v1.fixed_phase_at_second(phases, second)
                    active_phase[intersection_id] = current
                    phase_started_second[intersection_id] = second
                if (second - simulation["warmup_seconds"]) % CONTROL_INTERVAL_SECONDS == 0:
                    candidates = [phase for phase in phases if phase["movements"]] or phases
                    global_minute = (
                        simulation["evaluation_start_minute"]
                        + (second - simulation["warmup_seconds"]) // 60
                    )
                    demand = v1.demand_vector(inputs, mode, global_minute)
                    applied_beta = (
                        beta if mode in ("ForecastPressure", "OraclePressure") else 0.0
                    )
                    scores = [
                        normalized_phase_score(
                            phase,
                            queues,
                            movement_queues,
                            capacities,
                            lane_counts,
                            entry_roads,
                            demand,
                            applied_beta,
                        )
                        for phase in candidates
                    ]
                    desired = candidates[
                        max(range(len(scores)), key=lambda index: (scores[index], -index))
                    ]
                    current = active_phase[intersection_id]
                    green_age = second - phase_started_second[intersection_id]
                    if (
                        desired["phase_index"] != current["phase_index"]
                        and green_age >= MIN_GREEN_SECONDS
                    ):
                        active_phase[intersection_id] = desired
                        phase_started_second[intersection_id] = second
                        lost_until_second[intersection_id] = second + SWITCH_LOSS_SECONDS
                        phase_switches += 1
                chosen = active_phase[intersection_id]
            chosen_phases[intersection_id] = chosen

        removals = defaultdict(lambda: defaultdict(float))
        additions = defaultdict(lambda: defaultdict(float))
        reserved = defaultdict(float)
        step_throughput = 0.0
        step_completed = 0.0
        step_travel_time = 0.0

        for start_road, packets in state.items():
            if not packets:
                continue
            end_intersection = roads[start_road]["endIntersection"]
            budget = (
                v1.SATURATION_FLOW_PER_LANE_PER_SECOND
                * lane_counts[start_road]
                * step_seconds
            )
            if end_intersection in controllers:
                if second < lost_until_second[end_intersection]:
                    allowed = set()
                else:
                    allowed = {
                        end
                        for start, end, _ in chosen_phases[end_intersection]["movements"]
                        if start == start_road
                    }
            else:
                allowed = uncontrolled.get(end_intersection, {}).get(start_road, set())

            for key, available in sorted(
                packets.items(), key=lambda item: (item[0][2], item[0][3])
            ):
                if budget <= 1e-12:
                    break
                route, position, source_step, sequence = key
                final_road = position == len(route) - 1
                if not final_road and route[position + 1] not in allowed:
                    continue
                moved = min(available, budget)
                if not final_road:
                    next_road = route[position + 1]
                    storage = max(
                        0.0,
                        capacities[next_road]
                        - queues.get(next_road, 0.0)
                        - reserved[next_road],
                    )
                    moved = min(moved, storage)
                if moved <= 1e-12:
                    continue
                removals[start_road][key] += moved
                budget -= moved
                if final_road:
                    if in_evaluation:
                        step_completed += moved
                        step_travel_time += moved * (step + 1 - source_step) * step_seconds
                        if start_road in exit_roads:
                            step_throughput += moved
                else:
                    new_key = (route, position + 1, source_step, sequence)
                    additions[route[position + 1]][new_key] += moved
                    reserved[route[position + 1]] += moved

        for road, changes in removals.items():
            for key, value in changes.items():
                remaining = state[road][key] - value
                if remaining <= 1e-12:
                    del state[road][key]
                else:
                    state[road][key] = remaining
        for road, changes in additions.items():
            for key, value in changes.items():
                state[road][key] = state[road].get(key, 0.0) + value

        if in_evaluation:
            queues = {road: sum(packets.values()) for road, packets in state.items()}
            monitored_queue = sum(queues[road] for road in monitored_roads)
            all_present = sum(queues.values())
            queue_vehicle_seconds += monitored_queue * step_seconds
            vehicle_seconds += all_present * step_seconds
            for road in monitored_roads:
                occupancy = queues[road] / capacities[road]
                max_occupancy = max(max_occupancy, occupancy)
                if occupancy >= 0.8:
                    spillback_road_seconds += step_seconds
            throughput += step_throughput
            completed += step_completed
            completed_travel_time += step_travel_time

    evaluation_seconds = simulation["evaluation_seconds"]
    return {
        "mode": mode,
        "beta": beta if mode in ("ForecastPressure", "OraclePressure") else None,
        "queue_vehicle_seconds": queue_vehicle_seconds,
        "vehicle_seconds": vehicle_seconds,
        "mean_waiting_vehicles": queue_vehicle_seconds / evaluation_seconds,
        "mean_present_vehicles": vehicle_seconds / evaluation_seconds,
        "spillback_road_time_fraction": spillback_road_seconds
        / (len(monitored_roads) * evaluation_seconds),
        "max_road_occupancy": max_occupancy,
        "throughput_exit_vehicles": throughput,
        "completed_trips": completed,
        "mean_completed_travel_time": (
            completed_travel_time / completed if completed else None
        ),
        "phase_switches": phase_switches,
        "input_vehicles": input_vehicles,
        "simulator": "store_and_forward_queue_v2",
    }


def summarize(test_results):
    return v1.summarize(test_results)


def build(assets_dir, output_dir, roadnet_path):
    final_dir = output_dir.resolve()
    building = final_dir.with_name(final_dir.name + ".building")
    if final_dir.exists() or building.exists():
        raise FileExistsError(f"Output/build directory already exists: {final_dir}, {building}")
    building.mkdir(parents=True)
    metadata = json.loads((assets_dir / "ASSETS.json").read_text(encoding="utf-8"))
    roadnet = json.loads(roadnet_path.read_text(encoding="utf-8"))
    validation_results = []
    validation_modes = [("FixedTime", 0.0), ("MaxPressure", 0.0)] + [
        ("ForecastPressure", beta) for beta in BETA_GRID
    ]
    for date in metadata["validation_dates"]:
        for mode, beta in validation_modes:
            row = simulate(assets_dir / date, roadnet, metadata, mode, beta)
            row.update({"date": date, "split": "validation"})
            validation_results.append(row)
            print(
                f"validation date={date} mode={mode} beta={beta} "
                f"queue={row['queue_vehicle_seconds']:.1f}",
                flush=True,
            )
    beta_scores = {
        str(beta): statistics.mean(
            row["queue_vehicle_seconds"]
            for row in validation_results
            if row["mode"] == "ForecastPressure" and row["beta"] == beta
        )
        for beta in BETA_GRID
    }
    selected_beta = min(BETA_GRID, key=lambda beta: (beta_scores[str(beta)], beta))
    test_results = []
    for date in metadata["test_dates"]:
        for mode in ("FixedTime", "MaxPressure", "ForecastPressure", "OraclePressure"):
            row = simulate(assets_dir / date, roadnet, metadata, mode, selected_beta)
            row.update({"date": date, "split": "test"})
            test_results.append(row)
            print(
                f"test date={date} mode={mode} queue={row['queue_vehicle_seconds']:.1f} "
                f"throughput={row['throughput_exit_vehicles']:.1f}",
                flush=True,
            )
    summary = summarize(test_results)
    result = {
        "experiment_id": "xuancheng_control_admission_queue_v2",
        "status": "COMPLETED_PENDING_REPRODUCIBILITY_RERUN",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selected_beta": selected_beta,
        "validation_beta_mean_queue_vehicle_seconds": beta_scores,
        "validation_results": validation_results,
        "test_results": test_results,
        "summary": summary,
        "protocol": {
            "simulator": "5-second capacity-normalized store-and-forward point-queue",
            "simulation_step_seconds": SIMULATION_STEP_SECONDS,
            "control_interval_seconds": CONTROL_INTERVAL_SECONDS,
            "minimum_green_seconds": MIN_GREEN_SECONDS,
            "switch_loss_seconds": SWITCH_LOSS_SECONDS,
            "saturation_flow_per_lane_per_second": v1.SATURATION_FLOW_PER_LANE_PER_SECOND,
            "beta_grid": list(BETA_GRID),
            "controlled_intersections": len(metadata["controlled_intersections"]),
            "validation_dates": metadata["validation_dates"],
            "test_dates": metadata["test_dates"],
            "warmup_seconds": metadata["simulation"]["warmup_seconds"],
            "evaluation_seconds": metadata["simulation"]["evaluation_seconds"],
        },
        "scope_warning": (
            "Admission screen only: boundary-injected traffic, point queues, deterministic "
            "service, no microscopic car-following or peripheral background traffic."
        ),
    }
    (building / "RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    v1.write_csv(building / "validation_results.csv", validation_results)
    v1.write_csv(building / "test_results.csv", test_results)
    v1.write_csv(building / "summary.csv", summary)
    os.replace(building, final_dir)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roadnet", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.assets_dir.resolve(), args.output_dir.resolve(), args.roadnet.resolve())
    print(
        json.dumps(
            {"status": result["status"], "selected_beta": result["selected_beta"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
