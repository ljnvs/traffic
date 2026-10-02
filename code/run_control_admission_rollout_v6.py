"""Explicit short-horizon movement-queue rollout controller and oracle screen v6."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import run_control_admission_movement_v4 as v4
import run_control_admission_queue_v1 as v1
import run_control_admission_queue_v2 as v2


HORIZON_GRID_SECONDS = (30, 60, 90)
PROJECTION_STEP_SECONDS = 5
SPILLBACK_THRESHOLD = 0.8
SPILLBACK_PENALTY = 10.0


def transition_shares(movement_q, successors):
    shares = {}
    for road, next_roads in successors.items():
        if not next_roads:
            continue
        masses = [max(0.0, movement_q.get((road, nxt), 0.0)) for nxt in next_roads]
        total = sum(masses)
        if total > 1e-12:
            shares[road] = {nxt: mass / total for nxt, mass in zip(next_roads, masses)}
        else:
            shares[road] = {nxt: 1.0 / len(next_roads) for nxt in next_roads}
    return shares


def project_one_step(
    road_q,
    movement_q,
    actions,
    network,
    arrival_rates,
    switched,
    step_index,
):
    road_q = defaultdict(float, {road: float(value) for road, value in road_q.items()})
    movement_q = defaultdict(
        float, {tuple(pair): float(value) for pair, value in movement_q.items()}
    )
    for pair, rate in arrival_rates.items():
        mass = max(0.0, float(rate)) * PROJECTION_STEP_SECONDS
        movement_q[tuple(pair)] += mass
        road_q[pair[0]] += mass

    initial_road = dict(road_q)
    initial_movement = dict(movement_q)
    shares = transition_shares(initial_movement, network["successors"])
    removals = defaultdict(float)
    movement_removals = defaultdict(float)
    road_additions = defaultdict(float)
    movement_additions = defaultdict(float)
    reserved = defaultdict(float)

    for road in sorted(network["road_end"]):
        budget = (
            v1.SATURATION_FLOW_PER_LANE_PER_SECOND
            * network["lane_counts"][road]
            * PROJECTION_STEP_SECONDS
        )
        end_intersection = network["road_end"][road]
        if end_intersection in network["controlled"]:
            in_switch_loss = (
                end_intersection in switched
                and step_index * PROJECTION_STEP_SECONDS < v2.SWITCH_LOSS_SECONDS
            )
            allowed = set() if in_switch_loss else actions.get(end_intersection, set())
        else:
            allowed = {(road, nxt) for nxt in network["successors"].get(road, ())}

        pairs = [
            (road, nxt)
            for nxt in network["successors"].get(road, ())
            if (road, nxt) in allowed and initial_movement.get((road, nxt), 0.0) > 0
        ]
        eligible_total = sum(initial_movement[pair] for pair in pairs)
        if eligible_total > 1e-12 and budget > 1e-12:
            for pair in pairs:
                desired = min(
                    initial_movement[pair],
                    budget * initial_movement[pair] / eligible_total,
                )
                next_road = pair[1]
                storage = max(
                    0.0,
                    network["capacities"][next_road]
                    - initial_road.get(next_road, 0.0)
                    - reserved[next_road],
                )
                moved = min(desired, storage)
                if moved <= 1e-12:
                    continue
                movement_removals[pair] += moved
                removals[road] += moved
                road_additions[next_road] += moved
                reserved[next_road] += moved
                if next_road not in network["exit_roads"]:
                    for subsequent, share in shares.get(next_road, {}).items():
                        movement_additions[(next_road, subsequent)] += moved * share
            budget = max(0.0, budget - removals[road])

        movement_mass = sum(
            initial_movement.get((road, nxt), 0.0)
            for nxt in network["successors"].get(road, ())
        )
        terminal_mass = max(0.0, initial_road.get(road, 0.0) - movement_mass)
        if budget > 1e-12 and terminal_mass > 0:
            removals[road] += min(budget, terminal_mass)

    for road, value in removals.items():
        road_q[road] = max(0.0, road_q[road] - value)
    for pair, value in movement_removals.items():
        movement_q[pair] = max(0.0, movement_q[pair] - value)
    for road, value in road_additions.items():
        road_q[road] += value
    for pair, value in movement_additions.items():
        movement_q[pair] += value
    return dict(road_q), dict(movement_q)


def rollout_cost(road_q, movement_q, actions, network, arrival_rates_by_step, switched):
    cost = 0.0
    projected_road = dict(road_q)
    projected_movement = dict(movement_q)
    for step_index, arrival_rates in enumerate(arrival_rates_by_step):
        projected_road, projected_movement = project_one_step(
            projected_road,
            projected_movement,
            actions,
            network,
            arrival_rates,
            switched,
            step_index,
        )
        queue_cost = sum(projected_road.get(road, 0.0) for road in network["monitored"])
        spillback_cost = sum(
            max(
                0.0,
                projected_road.get(road, 0.0) / network["capacities"][road]
                - SPILLBACK_THRESHOLD,
            )
            ** 2
            * network["capacities"][road]
            for road in network["monitored"]
        )
        cost += PROJECTION_STEP_SECONDS * (
            queue_cost + SPILLBACK_PENALTY * spillback_cost
        )
    return cost


def oracle_arrival_rates(movement_inputs, global_minute, horizon_seconds, use_future):
    steps = horizon_seconds // PROJECTION_STEP_SECONDS
    if not use_future:
        return [{} for _ in range(steps)]
    block = movement_inputs["oracle"][str(global_minute)]
    pairs = [tuple(pair) for pair in movement_inputs["movement_pairs"]]
    output = []
    for step in range(steps):
        minute_offset = min(step // (60 // PROJECTION_STEP_SECONDS), len(block) - 1)
        output.append(
            {
                pair: float(block[minute_offset][index]) / 60.0
                for index, pair in enumerate(pairs)
            }
        )
    return output


def make_projection_network(roadnet, metadata):
    roads, _intersections, controllers, _uncontrolled, capacities, lane_counts, monitored = (
        v1.build_network(roadnet, metadata)
    )
    successors = defaultdict(set)
    road_ids = set(roads)
    for intersection in roadnet["intersections"]:
        for link in intersection.get("roadLinks", []):
            if link["startRoad"] in road_ids and link["endRoad"] in road_ids:
                successors[link["startRoad"]].add(link["endRoad"])
    return {
        "road_end": {road: details["endIntersection"] for road, details in roads.items()},
        "controlled": set(controllers),
        "successors": {road: tuple(sorted(successors.get(road, set()))) for road in roads},
        "capacities": capacities,
        "lane_counts": lane_counts,
        "monitored": set(monitored),
        "exit_roads": set(metadata["exit_roads"]),
    }


def maxpressure_phase_choices(
    controllers, active_phase, phase_started_second, second,
    queues, movement_queues, capacities, lane_counts, entry_roads,
):
    choices = {}
    for intersection_id, phases in controllers.items():
        candidates = [phase for phase in phases if phase["movements"]] or phases
        current = active_phase[intersection_id]
        if second - phase_started_second[intersection_id] < v2.MIN_GREEN_SECONDS:
            choices[intersection_id] = current
            continue
        scores = [
            v2.normalized_phase_score(
                phase, queues, movement_queues, capacities, lane_counts,
                entry_roads, [0.0] * len(entry_roads), 0.0,
            )
            for phase in candidates
        ]
        choices[intersection_id] = candidates[
            max(range(len(scores)), key=lambda index: (scores[index], -index))
        ]
    return choices


def select_rollout_phases(
    controllers,
    active_phase,
    phase_started_second,
    second,
    queues,
    movement_queues,
    capacities,
    lane_counts,
    entry_roads,
    projection_network,
    arrival_rates_by_step,
):
    choices = maxpressure_phase_choices(
        controllers, active_phase, phase_started_second, second,
        queues, movement_queues, capacities, lane_counts, entry_roads,
    )
    for intersection_id in sorted(controllers):
        current = active_phase[intersection_id]
        if second - phase_started_second[intersection_id] < v2.MIN_GREEN_SECONDS:
            choices[intersection_id] = current
            continue
        candidates = [phase for phase in controllers[intersection_id] if phase["movements"]]
        if not candidates:
            candidates = controllers[intersection_id]
        scored = []
        for phase in candidates:
            trial = dict(choices)
            trial[intersection_id] = phase
            actions = {
                iid: {(start, end) for start, end, _ in selected["movements"]}
                for iid, selected in trial.items()
            }
            switched = {
                iid
                for iid, selected in trial.items()
                if selected["phase_index"] != active_phase[iid]["phase_index"]
            }
            cost = rollout_cost(
                queues, movement_queues, actions, projection_network,
                arrival_rates_by_step, switched,
            )
            scored.append((cost, phase["phase_index"], phase))
        choices[intersection_id] = min(scored, key=lambda item: (item[0], item[1]))[2]
    return choices


def simulate(date_dir, roadnet, metadata, movement_inputs, mode, horizon_seconds):
    simulation = metadata["simulation"]
    step_seconds = v2.SIMULATION_STEP_SECONDS
    warmup_steps = simulation["warmup_seconds"] // step_seconds
    evaluation_steps = simulation["evaluation_seconds"] // step_seconds
    total_steps = warmup_steps + evaluation_steps
    arrivals, input_vehicles = v1.load_arrivals(date_dir / "flow_window.json", step_seconds)
    (
        roads, _intersections, controllers, uncontrolled,
        capacities, lane_counts, monitored_roads,
    ) = v1.build_network(roadnet, metadata)
    projection_network = make_projection_network(roadnet, metadata)
    day_inputs = movement_inputs[date_dir.name]

    state = {road: {} for road in roads}
    exit_roads = set(metadata["exit_roads"])
    entry_roads = metadata["entry_roads"]
    queue_vehicle_seconds = vehicle_seconds = spillback_road_seconds = 0.0
    max_occupancy = throughput = completed = completed_travel_time = 0.0
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

        for intersection_id, phases in controllers.items():
            if intersection_id not in active_phase:
                active_phase[intersection_id] = v1.fixed_phase_at_second(phases, second)
                phase_started_second[intersection_id] = second

        if in_evaluation and mode != "FixedTime" and (
            (second - simulation["warmup_seconds"]) % v2.CONTROL_INTERVAL_SECONDS == 0
        ):
            if mode == "MaxPressure":
                choices = maxpressure_phase_choices(
                    controllers, active_phase, phase_started_second, second,
                    queues, movement_queues, capacities, lane_counts, entry_roads,
                )
            else:
                global_minute = (
                    simulation["evaluation_start_minute"]
                    + (second - simulation["warmup_seconds"]) // 60
                )
                arrival_rates = oracle_arrival_rates(
                    day_inputs, global_minute, horizon_seconds,
                    use_future=mode == "RolloutOracle",
                )
                choices = select_rollout_phases(
                    controllers, active_phase, phase_started_second, second,
                    queues, movement_queues, capacities, lane_counts, entry_roads,
                    projection_network, arrival_rates,
                )
            for intersection_id, desired in choices.items():
                current = active_phase[intersection_id]
                if desired["phase_index"] != current["phase_index"]:
                    active_phase[intersection_id] = desired
                    phase_started_second[intersection_id] = second
                    lost_until_second[intersection_id] = second + v2.SWITCH_LOSS_SECONDS
                    phase_switches += 1

        chosen_phases = {}
        for intersection_id, phases in controllers.items():
            if not in_evaluation or mode == "FixedTime":
                chosen = v1.fixed_phase_at_second(phases, second)
                active_phase[intersection_id] = chosen
                phase_started_second[intersection_id] = second
                lost_until_second[intersection_id] = -1
            else:
                chosen = active_phase[intersection_id]
            chosen_phases[intersection_id] = chosen

        removals = defaultdict(lambda: defaultdict(float))
        additions = defaultdict(lambda: defaultdict(float))
        reserved = defaultdict(float)
        step_throughput = step_completed = step_travel_time = 0.0
        for start_road, packets in state.items():
            if not packets:
                continue
            end_intersection = roads[start_road]["endIntersection"]
            budget = v1.SATURATION_FLOW_PER_LANE_PER_SECOND * lane_counts[start_road] * step_seconds
            if end_intersection in controllers:
                allowed = set() if second < lost_until_second[end_intersection] else {
                    end for start, end, _ in chosen_phases[end_intersection]["movements"]
                    if start == start_road
                }
            else:
                allowed = uncontrolled.get(end_intersection, {}).get(start_road, set())
            for key, available in sorted(packets.items(), key=lambda item: (item[0][2], item[0][3])):
                if budget <= 1e-12:
                    break
                route, position, source_step, sequence = key
                final_road = position == len(route) - 1
                if not final_road and route[position + 1] not in allowed:
                    continue
                moved = min(available, budget)
                if not final_road:
                    next_road = route[position + 1]
                    storage = max(0.0, capacities[next_road] - queues.get(next_road, 0.0) - reserved[next_road])
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
            queue_vehicle_seconds += sum(queues[road] for road in monitored_roads) * step_seconds
            vehicle_seconds += sum(queues.values()) * step_seconds
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
        "horizon_seconds": horizon_seconds if mode.startswith("Rollout") else None,
        "queue_vehicle_seconds": queue_vehicle_seconds,
        "vehicle_seconds": vehicle_seconds,
        "mean_waiting_vehicles": queue_vehicle_seconds / evaluation_seconds,
        "mean_present_vehicles": vehicle_seconds / evaluation_seconds,
        "spillback_road_time_fraction": spillback_road_seconds / (len(monitored_roads) * evaluation_seconds),
        "max_road_occupancy": max_occupancy,
        "throughput_exit_vehicles": throughput,
        "completed_trips": completed,
        "mean_completed_travel_time": completed_travel_time / completed if completed else None,
        "phase_switches": phase_switches,
        "input_vehicles": input_vehicles,
        "simulator": "store_and_forward_queue_v2_rollout_v6",
    }


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)
    result = []
    for mode in ("FixedTime", "MaxPressure", "RolloutNoFuture", "RolloutOracle"):
        selected = grouped[mode]
        result.append({
            "mode": mode,
            "days": len(selected),
            "mean_queue_vehicle_seconds": statistics.mean(r["queue_vehicle_seconds"] for r in selected),
            "mean_waiting_vehicles": statistics.mean(r["mean_waiting_vehicles"] for r in selected),
            "mean_throughput_exit_vehicles": statistics.mean(r["throughput_exit_vehicles"] for r in selected),
            "mean_spillback_road_time_fraction": statistics.mean(r["spillback_road_time_fraction"] for r in selected),
            "mean_completed_travel_time": statistics.mean(r["mean_completed_travel_time"] for r in selected),
        })
    return result


def build(assets_dir, demand_dir, split_path, prediction_dir, output_dir, roadnet_path):
    final_dir = output_dir.resolve()
    building = final_dir.with_name(final_dir.name + ".building")
    if final_dir.exists() or building.exists():
        raise FileExistsError(f"Output/build directory already exists: {final_dir}, {building}")
    building.mkdir(parents=True)
    metadata = json.loads((assets_dir / "ASSETS.json").read_text(encoding="utf-8"))
    roadnet = json.loads(roadnet_path.read_text(encoding="utf-8"))
    movement_inputs, _manifest = v4.prepare_movement_inputs(
        demand_dir, split_path, prediction_dir,
        metadata["validation_dates"], metadata["test_dates"],
        metadata["simulation"]["evaluation_start_minute"],
        metadata["simulation"]["evaluation_end_minute"],
    )
    validation_results = []
    for date in metadata["validation_dates"]:
        for mode, horizon in [("FixedTime", 0), ("MaxPressure", 0)] + [
            (mode, horizon) for horizon in HORIZON_GRID_SECONDS
            for mode in ("RolloutNoFuture", "RolloutOracle")
        ]:
            row = simulate(assets_dir / date, roadnet, metadata, movement_inputs, mode, horizon)
            row.update({"date": date, "split": "validation"})
            validation_results.append(row)
            print(f"validation date={date} mode={mode} horizon={horizon} queue={row['queue_vehicle_seconds']:.1f}", flush=True)
    oracle_scores = {
        str(horizon): statistics.mean(
            row["queue_vehicle_seconds"] for row in validation_results
            if row["mode"] == "RolloutOracle" and row["horizon_seconds"] == horizon
        ) for horizon in HORIZON_GRID_SECONDS
    }
    selected_horizon = min(HORIZON_GRID_SECONDS, key=lambda h: (oracle_scores[str(h)], h))
    test_results = []
    for date in metadata["test_dates"]:
        for mode in ("FixedTime", "MaxPressure", "RolloutNoFuture", "RolloutOracle"):
            row = simulate(assets_dir / date, roadnet, metadata, movement_inputs, mode, selected_horizon)
            row.update({"date": date, "split": "test"})
            test_results.append(row)
            print(f"test date={date} mode={mode} queue={row['queue_vehicle_seconds']:.1f} throughput={row['throughput_exit_vehicles']:.1f}", flush=True)
    summary = summarize(test_results)
    result = {
        "experiment_id": "xuancheng_control_admission_rollout_v6",
        "status": "COMPLETED_PENDING_REPRODUCIBILITY_RERUN",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selected_horizon_seconds": selected_horizon,
        "validation_oracle_horizon_mean_queue_vehicle_seconds": oracle_scores,
        "validation_results": validation_results,
        "test_results": test_results,
        "summary": summary,
        "protocol": {
            "controller": "network movement-queue coordinate-rollout",
            "horizon_grid_seconds": list(HORIZON_GRID_SECONDS),
            "projection_step_seconds": PROJECTION_STEP_SECONDS,
            "spillback_threshold": SPILLBACK_THRESHOLD,
            "spillback_penalty": SPILLBACK_PENALTY,
            "action_search": "one deterministic coordinate-descent sweep initialized by MaxPressure",
            "simulation_step_seconds": v2.SIMULATION_STEP_SECONDS,
            "control_interval_seconds": v2.CONTROL_INTERVAL_SECONDS,
            "minimum_green_seconds": v2.MIN_GREEN_SECONDS,
            "switch_loss_seconds": v2.SWITCH_LOSS_SECONDS,
            "validation_dates": metadata["validation_dates"],
            "test_dates": metadata["test_dates"],
            "sealed_split": "calibration",
        },
        "source_hashes": {
            "assets_manifest_sha256": v4.sha256(assets_dir / "ASSETS.json"),
            "roadnet_sha256": v4.sha256(roadnet_path),
            "movement_demand_sha256": v4.sha256(demand_dir / "demand_by_movement_1min.csv"),
            "split_sha256": v4.sha256(split_path),
        },
        "scope_warning": "Low-cost deterministic queue rollout, not microscopic MPC.",
    }
    (building / "RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    v1.write_csv(building / "validation_results.csv", validation_results)
    v1.write_csv(building / "test_results.csv", test_results)
    v1.write_csv(building / "summary.csv", summary)
    os.replace(building, final_dir)
    return result


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, default=project / "04_实验" / "控制基线" / "control_admission_v1" / "assets_subnetwork_v2")
    parser.add_argument("--demand-dir", type=Path, default=project / "02_数据" / "派生数据" / "宣城_movement_demand_v1")
    parser.add_argument("--split-path", type=Path, default=project / "02_数据" / "派生数据" / "宣城_clean_v2" / "SPLIT.json")
    parser.add_argument("--prediction-dir", type=Path, default=project / "04_实验" / "预测基线" / "movement_point_baseline_v1" / "results_v1")
    parser.add_argument("--output-dir", type=Path, default=project / "04_实验" / "控制基线" / "control_admission_v6" / "results_v1")
    parser.add_argument("--roadnet", type=Path, default=project / "04_实验" / "控制基线" / "control_admission_v1" / "assets_subnetwork_v2" / "roadnet_subnetwork.json")
    args = parser.parse_args()
    result = build(args.assets_dir.resolve(), args.demand_dir.resolve(), args.split_path.resolve(), args.prediction_dir.resolve(), args.output_dir.resolve(), args.roadnet.resolve())
    print(json.dumps({"status": result["status"], "selected_horizon_seconds": result["selected_horizon_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
