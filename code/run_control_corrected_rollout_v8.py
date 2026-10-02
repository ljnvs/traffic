"""Leakage-safe corrected rollout control experiment (v8).

The experiment freezes a 60-second horizon and joint enumeration after the v7
audit.  Validation is run first; test is evaluated only after the structure is
frozen.  Calibration is never loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_control_admission_movement_v4 as v4
import run_control_admission_queue_v1 as v1
import run_control_admission_queue_v2 as v2
import run_control_admission_rollout_v6 as v6
import run_control_interface_audit_v7 as v7
import run_movement_point_baselines_v1 as movement_baseline


HORIZON_SECONDS = 60
MODES = (
    "FixedTime",
    "MaxPressure",
    "RolloutZeroFuture",
    "RolloutCausalForecast",
    "RolloutEventOracle",
)


def prepare_forecasts(
    demand_dir, split_path, prediction_dir, requested_dates, fit_scope,
    start_minute, end_minute,
):
    """Prepare only the requested split; validation never opens test predictions."""
    (
        values, entry_values, dates, movements, entries, movement_to_entry,
        movement_map, split_names, _split,
    ) = movement_baseline.load_data(demand_dir, split_path)
    features, target, origins, _ = movement_baseline.feature_matrix(entry_values, "entry")
    features = movement_baseline.add_weekday_features(features, dates, origins)
    point_manifest = json.loads((prediction_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    selected_leaf = int(point_manifest["protocol"]["selected_entry_min_samples_leaf"])
    selected_window = int(point_manifest["protocol"]["selected_share_window_minutes"])
    if selected_window != 60:
        raise AssertionError("Frozen movement interface must use the selected 60-minute share window")
    origin_index = {int(minute): index for index, minute in enumerate(origins)}
    movement_pairs = movement_map[["entry_road", "next_road"]].astype(str).values.tolist()

    if fit_scope == "train_only":
        train_mask = movement_baseline.row_mask(split_names, ["train"], origins)
        validation_mask = movement_baseline.row_mask(split_names, ["validation"], origins)
        model = movement_baseline.fit_extra_trees(
            features[train_mask], target[train_mask], selected_leaf,
            movement_baseline.FINAL_TREES,
        )
        entry_prediction = np.clip(model.predict(features[validation_mask]), 0, None).astype(np.float32)
        train_values = values[split_names == "train"]
        validation_values = values[split_names == "validation"]
        global_share, _minute_share, _totals = movement_baseline.share_parameters(
            train_values, movement_to_entry, len(entries)
        )
        predictions = movement_baseline.hierarchical_rolling(
            entry_prediction, validation_values, origins, selected_window,
            global_share, movement_to_entry, len(entries),
        ).reshape(
            len(validation_values), len(origins), movement_baseline.HORIZON, len(movements)
        )
        available_dates = [date for date, name in zip(dates, split_names) if name == "validation"]
    elif fit_scope == "train_plus_validation_frozen_test_archive":
        with np.load(prediction_dir / "test_predictions.npz") as archive:
            archive_movements = archive["movements"].astype(str).tolist()
            archive_origins = archive["origins"].astype(int)
            available_dates = archive["dates"].astype(str).tolist()
            predictions = archive["hierarchical_rolling_share"].reshape(
                len(available_dates), len(archive_origins), movement_baseline.HORIZON, len(movements)
            )
        if archive_movements != movements or not np.array_equal(archive_origins, origins):
            raise AssertionError("Frozen test archive order differs from current movement data")
    else:
        raise ValueError(f"Unknown fit scope: {fit_scope}")

    date_index = {date: index for index, date in enumerate(available_dates)}
    output = {}
    for date in requested_dates:
        if date not in date_index:
            raise AssertionError(f"No frozen prediction for requested date {date}")
        matrix = predictions[date_index[date]]
        blocks = {}
        for origin in range(start_minute - 1, end_minute):
            blocks[str(origin)] = matrix[origin_index[origin]].tolist()
        output[date] = {
            "date": date,
            "fit_scope": fit_scope,
            "movement_pairs": movement_pairs,
            "forecast": blocks,
        }
    return output, point_manifest


def observed_current_minute_counts(arrivals, current_step):
    minute_start_step = current_step - (current_step % (60 // v2.SIMULATION_STEP_SECONDS))
    counts = defaultdict(float)
    for step in range(minute_start_step, current_step + 1):
        for route, _sequence in arrivals.get(step, []):
            if len(route) >= 2:
                counts[(route[0], route[1])] += 1.0
    return counts


def causal_forecast_rates(day_inputs, arrivals, current_step, simulation_start_minute):
    """Residualize current-minute forecasts by arrivals already observed online."""
    step_seconds = v2.SIMULATION_STEP_SECONDS
    slots = HORIZON_SECONDS // step_seconds
    steps_per_minute = 60 // step_seconds
    global_minute = simulation_start_minute + current_step // steps_per_minute
    step_in_minute = current_step % steps_per_minute
    origin = global_minute - 1
    block = day_inputs["forecast"][str(origin)]
    pairs = [tuple(pair) for pair in day_inputs["movement_pairs"]]
    observed = observed_current_minute_counts(arrivals, current_step)
    remaining_current_slots = steps_per_minute - step_in_minute - 1
    current_remaining = {
        pair: max(0.0, float(block[0][index]) - observed.get(pair, 0.0))
        for index, pair in enumerate(pairs)
    }
    output = []
    for offset in range(slots):
        if offset == 0:
            output.append({})
            continue
        target_step_in_day = current_step + offset
        target_minute = simulation_start_minute + target_step_in_day // steps_per_minute
        if target_minute == global_minute:
            denominator = remaining_current_slots * step_seconds
            output.append({
                pair: mass / denominator
                for pair, mass in current_remaining.items()
                if denominator > 0 and mass > 0
            })
        else:
            row_index = target_minute - origin - 1
            output.append({
                pair: float(block[row_index][index]) / 60.0
                for index, pair in enumerate(pairs)
            })
    return output


def allowed_candidates(controllers, active_phase, phase_started_second, second):
    effective = v7.effective_phases(controllers)
    candidates = {}
    for iid, phases in effective.items():
        if second - phase_started_second[iid] < v2.MIN_GREEN_SECONDS:
            candidates[iid] = [active_phase[iid]]
        else:
            candidates[iid] = phases or [active_phase[iid]]
    return candidates


def joint_rollout_choice(
    controllers, active_phase, phase_started_second, second,
    queues, movement_queues, projection_network, arrival_rates,
):
    candidates = allowed_candidates(controllers, active_phase, phase_started_second, second)
    variable_ids = sorted(iid for iid, phases in candidates.items() if len(phases) > 1)
    fixed = {iid: phases[0] for iid, phases in candidates.items() if len(phases) == 1}
    best = None
    evaluated = 0
    import itertools
    for combination in itertools.product(*(candidates[iid] for iid in variable_ids)):
        choices = dict(fixed)
        choices.update(dict(zip(variable_ids, combination)))
        cost = v7.score_choices(
            choices, active_phase, queues, movement_queues,
            projection_network, arrival_rates,
        )
        signature = tuple(choices[iid]["phase_index"] for iid in sorted(choices))
        candidate = (cost, signature, choices)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
        evaluated += 1
    if best is None:
        raise AssertionError("No feasible rollout action")
    return best[2], best[0], evaluated


def simulate(date_dir, roadnet, metadata, day_inputs, mode):
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
    projection_network = v6.make_projection_network(roadnet, metadata)
    state = {road: {} for road in roads}
    exit_roads = set(metadata["exit_roads"])
    entry_roads = metadata["entry_roads"]
    queue_vehicle_seconds = vehicle_seconds = spillback_road_seconds = 0.0
    max_occupancy = throughput = completed = completed_travel_time = 0.0
    phase_switches = rollout_decisions = evaluated_actions = 0
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
        for iid, phases in controllers.items():
            if iid not in active_phase:
                active_phase[iid] = v1.fixed_phase_at_second(phases, second)
                phase_started_second[iid] = second

        decision = in_evaluation and mode != "FixedTime" and (
            (second - simulation["warmup_seconds"]) % v2.CONTROL_INTERVAL_SECONDS == 0
        )
        if decision:
            if mode == "MaxPressure":
                choices = v6.maxpressure_phase_choices(
                    controllers, active_phase, phase_started_second, second,
                    queues, movement_queues, capacities, lane_counts, entry_roads,
                )
            else:
                if mode == "RolloutZeroFuture":
                    rates = [{} for _ in range(HORIZON_SECONDS // step_seconds)]
                elif mode == "RolloutCausalForecast":
                    rates = causal_forecast_rates(
                        day_inputs, arrivals, step,
                        simulation["simulation_start_minute"],
                    )
                elif mode == "RolloutEventOracle":
                    rates = v7.exact_event_oracle(arrivals, step, HORIZON_SECONDS)
                else:
                    raise ValueError(mode)
                choices, _cost, count = joint_rollout_choice(
                    controllers, active_phase, phase_started_second, second,
                    queues, movement_queues, projection_network, rates,
                )
                rollout_decisions += 1
                evaluated_actions += count
            for iid, desired in choices.items():
                current = active_phase[iid]
                if desired["phase_index"] != current["phase_index"]:
                    active_phase[iid] = desired
                    phase_started_second[iid] = second
                    lost_until_second[iid] = second + v2.SWITCH_LOSS_SECONDS
                    phase_switches += 1

        chosen_phases = {}
        for iid, phases in controllers.items():
            if not in_evaluation or mode == "FixedTime":
                chosen = v1.fixed_phase_at_second(phases, second)
                active_phase[iid] = chosen
                phase_started_second[iid] = second
                lost_until_second[iid] = -1
            else:
                chosen = active_phase[iid]
            chosen_phases[iid] = chosen

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
        "horizon_seconds": HORIZON_SECONDS if mode.startswith("Rollout") else None,
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
        "rollout_decisions": rollout_decisions,
        "mean_joint_actions_evaluated": evaluated_actions / rollout_decisions if rollout_decisions else None,
        "input_vehicles": input_vehicles,
        "simulator": "store_and_forward_queue_v2_corrected_rollout_v8",
    }


def summarize(rows):
    output = []
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        output.append({
            "mode": mode,
            "days": len(selected),
            "mean_queue_vehicle_seconds": statistics.mean(r["queue_vehicle_seconds"] for r in selected),
            "mean_waiting_vehicles": statistics.mean(r["mean_waiting_vehicles"] for r in selected),
            "mean_throughput_exit_vehicles": statistics.mean(r["throughput_exit_vehicles"] for r in selected),
            "mean_spillback_road_time_fraction": statistics.mean(r["spillback_road_time_fraction"] for r in selected),
            "mean_completed_travel_time": statistics.mean(r["mean_completed_travel_time"] for r in selected),
            "mean_phase_switches": statistics.mean(r["phase_switches"] for r in selected),
        })
    return output


def relative_effect(summary, candidate, baseline, metric, lower_better):
    by_mode = {row["mode"]: row for row in summary}
    a = by_mode[candidate][metric]
    b = by_mode[baseline][metric]
    raw = (a - b) / b
    return -raw if lower_better else raw


def run_split(dates, assets_dir, roadnet, metadata, inputs, split):
    rows = []
    for date in dates:
        for mode in MODES:
            row = simulate(assets_dir / date, roadnet, metadata, inputs[date], mode)
            row.update({"date": date, "split": split})
            rows.append(row)
            print(
                f"{split} date={date} mode={mode} queue={row['queue_vehicle_seconds']:.1f} "
                f"throughput={row['throughput_exit_vehicles']:.1f}", flush=True,
            )
    return rows


def report_text(result):
    test = result["test_effects"]
    gate = result["validation_gate"]
    return f"""# v8 修正接口同条件闭环报告

## 状态

实验已完成并冻结。结构仅依据v7审计和validation确定，之后一次性运行test；calibration未使用。

## Validation门

- 精确事件Oracle相对零未来的排队改善：{gate['oracle_vs_zero_queue_improvement']:.3%}
- 因果预测相对零未来的排队改善：{gate['forecast_vs_zero_queue_improvement']:.3%}
- Oracle信息价值门：{gate['oracle_value_gate']}

## Test一次性评估

- 因果预测 vs 零未来：排队 {test['forecast_vs_zero_queue_improvement']:.3%}，吞吐 {test['forecast_vs_zero_throughput_improvement']:.3%}，回溢 {test['forecast_vs_zero_spillback_improvement']:.3%}。
- 精确事件Oracle vs 零未来：排队 {test['oracle_vs_zero_queue_improvement']:.3%}，吞吐 {test['oracle_vs_zero_throughput_improvement']:.3%}，回溢 {test['oracle_vs_zero_spillback_improvement']:.3%}。
- 零未来滚动器 vs MaxPressure：排队 {test['zero_vs_maxpressure_queue_improvement']:.3%}。

## 解释边界

1. 因果预测与事件Oracle都采用相同的60秒时域、5秒推进、联合动作穷举、最小绿和切相损失。
2. 预测是源端出发需求代理；事件Oracle也是源端未来出发事件，不是真实子网边界到达。
3. 该点队列投影仍采用比例服务近似，结果不是微观仿真或严格控制上界。
4. test只用于冻结后的最终评估，不得据此继续调整结构。论文初稿仍未开始。
"""


def build(assets_dir, demand_dir, split_path, prediction_dir, output_dir, roadnet_path):
    final_dir = output_dir.resolve()
    building = final_dir.with_name(final_dir.name + ".building")
    if final_dir.exists() or building.exists():
        raise FileExistsError(f"Output/build directory already exists: {final_dir}, {building}")
    building.mkdir(parents=True)
    metadata = json.loads((assets_dir / "ASSETS.json").read_text(encoding="utf-8"))
    roadnet = json.loads(roadnet_path.read_text(encoding="utf-8"))

    validation_inputs, manifest = prepare_forecasts(
        demand_dir, split_path, prediction_dir, metadata["validation_dates"],
        "train_only", metadata["simulation"]["evaluation_start_minute"],
        metadata["simulation"]["evaluation_end_minute"],
    )
    validation_rows = run_split(
        metadata["validation_dates"], assets_dir, roadnet, metadata,
        validation_inputs, "validation",
    )
    validation_summary = summarize(validation_rows)
    validation_gate = {
        "oracle_vs_zero_queue_improvement": relative_effect(
            validation_summary, "RolloutEventOracle", "RolloutZeroFuture",
            "mean_queue_vehicle_seconds", True,
        ),
        "forecast_vs_zero_queue_improvement": relative_effect(
            validation_summary, "RolloutCausalForecast", "RolloutZeroFuture",
            "mean_queue_vehicle_seconds", True,
        ),
    }
    validation_gate["oracle_value_gate"] = (
        "GO" if validation_gate["oracle_vs_zero_queue_improvement"] > 0 else "NO_GO"
    )

    # Structure is now frozen.  Only here may the frozen test prediction archive be opened.
    test_inputs, _ = prepare_forecasts(
        demand_dir, split_path, prediction_dir, metadata["test_dates"],
        "train_plus_validation_frozen_test_archive",
        metadata["simulation"]["evaluation_start_minute"],
        metadata["simulation"]["evaluation_end_minute"],
    )
    test_rows = run_split(
        metadata["test_dates"], assets_dir, roadnet, metadata, test_inputs, "test"
    )
    test_summary = summarize(test_rows)
    test_effects = {
        "forecast_vs_zero_queue_improvement": relative_effect(test_summary, "RolloutCausalForecast", "RolloutZeroFuture", "mean_queue_vehicle_seconds", True),
        "forecast_vs_zero_throughput_improvement": relative_effect(test_summary, "RolloutCausalForecast", "RolloutZeroFuture", "mean_throughput_exit_vehicles", False),
        "forecast_vs_zero_spillback_improvement": relative_effect(test_summary, "RolloutCausalForecast", "RolloutZeroFuture", "mean_spillback_road_time_fraction", True),
        "oracle_vs_zero_queue_improvement": relative_effect(test_summary, "RolloutEventOracle", "RolloutZeroFuture", "mean_queue_vehicle_seconds", True),
        "oracle_vs_zero_throughput_improvement": relative_effect(test_summary, "RolloutEventOracle", "RolloutZeroFuture", "mean_throughput_exit_vehicles", False),
        "oracle_vs_zero_spillback_improvement": relative_effect(test_summary, "RolloutEventOracle", "RolloutZeroFuture", "mean_spillback_road_time_fraction", True),
        "zero_vs_maxpressure_queue_improvement": relative_effect(test_summary, "RolloutZeroFuture", "MaxPressure", "mean_queue_vehicle_seconds", True),
    }
    result = {
        "experiment_id": "xuancheng_corrected_rollout_v8",
        "status": "COMPLETED_FROZEN_TEST_EVALUATION",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "validation_gate": validation_gate,
        "test_effects": test_effects,
        "validation_results": validation_rows,
        "validation_summary": validation_summary,
        "test_results": test_rows,
        "test_summary": test_summary,
        "protocol": {
            "horizon_seconds": HORIZON_SECONDS,
            "projection_step_seconds": v2.SIMULATION_STEP_SECONDS,
            "control_interval_seconds": v2.CONTROL_INTERVAL_SECONDS,
            "minimum_green_seconds": v2.MIN_GREEN_SECONDS,
            "switch_loss_seconds": v2.SWITCH_LOSS_SECONDS,
            "action_search": "exact joint enumeration of distinct nonempty actions at unlocked intersections",
            "forecast_observability": "decision minute m uses forecast origin m-1; current-minute forecast is residualized by observed arrivals",
            "oracle_observability": "slot 0 excludes arrivals already inserted before decision; slot k uses event step current+k",
            "validation_dates": metadata["validation_dates"],
            "test_dates": metadata["test_dates"],
            "calibration_status": "sealed_not_loaded",
            "test_structure_selection": "none",
            "selected_forecast_leaf": manifest["protocol"]["selected_entry_min_samples_leaf"],
            "selected_share_window_minutes": manifest["protocol"]["selected_share_window_minutes"],
        },
        "source_hashes": {
            "assets_manifest_sha256": v4.sha256(assets_dir / "ASSETS.json"),
            "roadnet_sha256": v4.sha256(roadnet_path),
            "movement_demand_sha256": v4.sha256(demand_dir / "demand_by_movement_1min.csv"),
            "split_sha256": v4.sha256(split_path),
            "prediction_manifest_sha256": v4.sha256(prediction_dir / "MANIFEST.json"),
            "frozen_test_predictions_sha256": v4.sha256(prediction_dir / "test_predictions.npz"),
        },
        "scope_warning": "Source-departure proxy and aggregate queue model; not microscopic control or a strict oracle upper bound.",
    }
    (building / "RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    v1.write_csv(building / "validation_results.csv", validation_rows)
    v1.write_csv(building / "validation_summary.csv", validation_summary)
    v1.write_csv(building / "test_results.csv", test_rows)
    v1.write_csv(building / "test_summary.csv", test_summary)
    (building / "实验报告.md").write_text(report_text(result), encoding="utf-8")
    os.replace(building, final_dir)
    return result


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, default=project / "04_实验" / "控制基线" / "control_admission_v1" / "assets_subnetwork_v2")
    parser.add_argument("--demand-dir", type=Path, default=project / "02_数据" / "派生数据" / "宣城_movement_demand_v1")
    parser.add_argument("--split-path", type=Path, default=project / "02_数据" / "派生数据" / "宣城_clean_v2" / "SPLIT.json")
    parser.add_argument("--prediction-dir", type=Path, default=project / "04_实验" / "预测基线" / "movement_point_baseline_v1" / "results_v1")
    parser.add_argument("--output-dir", type=Path, default=project / "04_实验" / "控制基线" / "corrected_rollout_v8")
    parser.add_argument("--roadnet", type=Path, default=project / "04_实验" / "控制基线" / "control_admission_v1" / "assets_subnetwork_v2" / "roadnet_subnetwork.json")
    args = parser.parse_args()
    result = build(
        args.assets_dir.resolve(), args.demand_dir.resolve(), args.split_path.resolve(),
        args.prediction_dir.resolve(), args.output_dir.resolve(), args.roadnet.resolve(),
    )
    print(json.dumps({
        "status": result["status"],
        "validation_gate": result["validation_gate"],
        "test_effects": result["test_effects"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
