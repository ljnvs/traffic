"""Fast store-and-forward admission screen for forecast-aware signal control."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path


VEHICLE_SPACE_M = 7.5
SATURATION_FLOW_PER_LANE_PER_SECOND = 0.5


def road_length(points):
    return sum(
        math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        for a, b in zip(points, points[1:])
    )


def build_network(roadnet, metadata):
    roads = {road["id"]: road for road in roadnet["roads"]}
    intersections = {item["id"]: item for item in roadnet["intersections"]}
    controlled = set(metadata["controlled_intersections"])
    entry_index = {road: index for index, road in enumerate(metadata["entry_roads"])}
    controllers = {}
    for intersection_id in controlled:
        intersection = intersections[intersection_id]
        phases = []
        for phase_index, phase in enumerate(intersection["trafficLight"]["lightphases"]):
            movements = []
            served_entries = set()
            for link_index in phase.get("availableRoadLinks", []):
                link = intersection["roadLinks"][link_index]
                start_lanes = len({lane["startLaneIndex"] for lane in link["laneLinks"]})
                movements.append((link["startRoad"], link["endRoad"], max(1, start_lanes)))
                if link["startRoad"] in entry_index:
                    served_entries.add(entry_index[link["startRoad"]])
            phases.append(
                {
                    "phase_index": phase_index,
                    "duration": max(1, int(phase.get("time", 30))),
                    "movements": movements,
                    "served_entry_indices": sorted(served_entries),
                }
            )
        controllers[intersection_id] = phases
    allowed_at_uncontrolled = {}
    for intersection_id, intersection in intersections.items():
        if intersection_id in controlled:
            continue
        mapping = defaultdict(set)
        for link in intersection.get("roadLinks", []):
            mapping[link["startRoad"]].add(link["endRoad"])
        allowed_at_uncontrolled[intersection_id] = mapping
    capacities = {
        road_id: max(
            1.0,
            road_length(road["points"]) * len(road["lanes"]) / VEHICLE_SPACE_M,
        )
        for road_id, road in roads.items()
    }
    lane_counts = {road_id: len(road["lanes"]) for road_id, road in roads.items()}
    monitored = [
        road_id
        for road_id, road in roads.items()
        if road["endIntersection"] in controlled
    ]
    return roads, intersections, controllers, allowed_at_uncontrolled, capacities, lane_counts, monitored


def fixed_phase_at_second(phases, second):
    cycle = sum(phase["duration"] for phase in phases)
    offset = second % cycle
    elapsed = 0
    for phase in phases:
        elapsed += phase["duration"]
        if offset < elapsed:
            return phase
    return phases[-1]


def demand_vector(inputs, mode, global_minute):
    if mode not in ("ForecastPressure", "OraclePressure"):
        return [0.0] * len(inputs["entry_roads"])
    key = "forecast" if mode == "ForecastPressure" else "oracle"
    block = inputs[key][str(global_minute)]
    return [
        sum(row[index] for row in block) / len(block)
        for index in range(len(inputs["entry_roads"]))
    ]


def phase_score(phase, queues, movement_queues, lane_counts, demand, beta):
    pressure = 0.0
    for start, end, _ in phase["movements"]:
        upstream = movement_queues.get((start, end), 0.0) / lane_counts[start]
        downstream = queues.get(end, 0.0) / lane_counts[end]
        pressure += max(0.0, upstream - downstream)
    bonus = beta * sum(demand[index] for index in phase["served_entry_indices"])
    return pressure + bonus


def load_arrivals(flow_path, step_seconds):
    flows = json.loads(flow_path.read_text(encoding="utf-8"))
    arrivals = defaultdict(list)
    for sequence, flow in enumerate(flows):
        route = tuple(flow["route"])
        step = int(float(flow["startTime"]) // step_seconds)
        arrivals[step].append((route, sequence))
    return arrivals, len(flows)


def simulate(date_dir, roadnet, metadata, mode, beta):
    simulation = metadata["simulation"]
    step_seconds = simulation["control_interval_seconds"]
    warmup_steps = simulation["warmup_seconds"] // step_seconds
    evaluation_steps = simulation["evaluation_seconds"] // step_seconds
    total_steps = warmup_steps + evaluation_steps
    inputs = json.loads((date_dir / "demand_inputs.json").read_text(encoding="utf-8"))
    arrivals, input_vehicles = load_arrivals(date_dir / "flow_window.json", step_seconds)
    (
        roads,
        intersections,
        controllers,
        uncontrolled,
        capacities,
        lane_counts,
        monitored_roads,
    ) = build_network(roadnet, metadata)

    # state[road][(route, position, source_step, sequence)] = vehicle mass
    state = {road: {} for road in roads}
    exit_roads = set(metadata["exit_roads"])
    queue_vehicle_seconds = 0.0
    vehicle_seconds = 0.0
    spillback_road_seconds = 0.0
    max_occupancy = 0.0
    throughput = 0.0
    completed = 0.0
    completed_travel_time = 0.0
    phase_switches = 0
    last_phase = {}

    for step in range(total_steps):
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
        second = step * step_seconds
        in_evaluation = step >= warmup_steps
        for intersection_id, phases in controllers.items():
            if not in_evaluation or mode == "FixedTime":
                chosen = fixed_phase_at_second(phases, second)
            else:
                candidates = [phase for phase in phases if phase["movements"]] or phases
                global_minute = (
                    simulation["evaluation_start_minute"]
                    + (step - warmup_steps) * step_seconds // 60
                )
                demand = demand_vector(inputs, mode, global_minute)
                applied_beta = beta if mode in ("ForecastPressure", "OraclePressure") else 0.0
                scores = [
                    phase_score(
                        phase, queues, movement_queues, lane_counts, demand, applied_beta
                    )
                    for phase in candidates
                ]
                chosen = candidates[max(range(len(scores)), key=lambda index: (scores[index], -index))]
            chosen_phases[intersection_id] = chosen
            current_index = chosen["phase_index"]
            if (
                in_evaluation
                and intersection_id in last_phase
                and last_phase[intersection_id] != current_index
            ):
                phase_switches += 1
            last_phase[intersection_id] = current_index

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
                SATURATION_FLOW_PER_LANE_PER_SECOND
                * lane_counts[start_road]
                * step_seconds
            )
            if end_intersection in controllers:
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
                route, position, source_step, _ = key
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
                    new_key = (route, position + 1, source_step, key[3])
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
    road_time = len(monitored_roads) * evaluation_seconds
    return {
        "mode": mode,
        "beta": beta if mode in ("ForecastPressure", "OraclePressure") else None,
        "queue_vehicle_seconds": queue_vehicle_seconds,
        "vehicle_seconds": vehicle_seconds,
        "mean_waiting_vehicles": queue_vehicle_seconds / evaluation_seconds,
        "mean_present_vehicles": vehicle_seconds / evaluation_seconds,
        "spillback_road_time_fraction": spillback_road_seconds / road_time,
        "max_road_occupancy": max_occupancy,
        "throughput_exit_vehicles": throughput,
        "completed_trips": completed,
        "mean_completed_travel_time": (
            completed_travel_time / completed if completed else None
        ),
        "phase_switches": phase_switches,
        "input_vehicles": input_vehicles,
        "simulator": "store_and_forward_queue_v1",
    }


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(test_results):
    by_mode = defaultdict(list)
    for row in test_results:
        by_mode[row["mode"]].append(row)
    summary = []
    for mode in ("FixedTime", "MaxPressure", "ForecastPressure", "OraclePressure"):
        rows = by_mode[mode]
        summary.append(
            {
                "mode": mode,
                "days": len(rows),
                "mean_queue_vehicle_seconds": statistics.mean(
                    row["queue_vehicle_seconds"] for row in rows
                ),
                "mean_waiting_vehicles": statistics.mean(
                    row["mean_waiting_vehicles"] for row in rows
                ),
                "mean_throughput_exit_vehicles": statistics.mean(
                    row["throughput_exit_vehicles"] for row in rows
                ),
                "mean_spillback_road_time_fraction": statistics.mean(
                    row["spillback_road_time_fraction"] for row in rows
                ),
                "mean_completed_travel_time": statistics.mean(
                    row["mean_completed_travel_time"] for row in rows
                ),
            }
        )
    return summary


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
        ("ForecastPressure", beta) for beta in metadata["beta_grid"]
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
        for beta in metadata["beta_grid"]
    }
    selected_beta = min(
        metadata["beta_grid"], key=lambda beta: (beta_scores[str(beta)], beta)
    )
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
        "experiment_id": "xuancheng_control_admission_queue_v1",
        "status": "COMPLETED_PENDING_REPRODUCIBILITY_RERUN",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selected_beta": selected_beta,
        "validation_beta_mean_queue_vehicle_seconds": beta_scores,
        "validation_results": validation_results,
        "test_results": test_results,
        "summary": summary,
        "protocol": {
            "simulator": "30-second deterministic store-and-forward point-queue",
            "saturation_flow_per_lane_per_second": SATURATION_FLOW_PER_LANE_PER_SECOND,
            "controlled_intersections": len(metadata["controlled_intersections"]),
            "validation_dates": metadata["validation_dates"],
            "test_dates": metadata["test_dates"],
            **metadata["simulation"],
        },
        "scope_warning": (
            "Admission screen only: boundary-injected traffic, point queues, deterministic "
            "service, no microscopic car-following or peripheral background traffic."
        ),
    }
    (building / "RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(building / "validation_results.csv", validation_results)
    write_csv(building / "test_results.csv", test_results)
    write_csv(building / "summary.csv", summary)
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
