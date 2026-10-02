"""Movement-level predictive-pressure and oracle-value admission screen v4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

import run_control_admission_queue_v1 as v1
import run_control_admission_queue_v2 as v2
import run_movement_point_baselines_v1 as movement_baseline


HORIZON_GRID_MINUTES = (1, 3, 5, 10, 15)
WEIGHT_GRID = (0.25, 0.5, 1.0)
POLICY_GRID = ((0, 0.0),) + tuple(
    (horizon, weight) for horizon in HORIZON_GRID_MINUTES for weight in WEIGHT_GRID
)

_ACTIVE_HORIZON_MINUTES = 0
_ACTIVE_MOVEMENT_INPUTS = None
_ORIGINAL_DEMAND_VECTOR = v1.demand_vector
_ORIGINAL_PHASE_SCORE = v2.normalized_phase_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def movement_demand_vector(_inputs, mode, global_minute):
    if (
        mode not in ("ForecastPressure", "OraclePressure")
        or _ACTIVE_HORIZON_MINUTES <= 0
        or _ACTIVE_MOVEMENT_INPUTS is None
    ):
        return {}
    key = "forecast" if mode == "ForecastPressure" else "oracle"
    block = _ACTIVE_MOVEMENT_INPUTS[key][str(global_minute)]
    horizon_rows = min(len(block), int(_ACTIVE_HORIZON_MINUTES))
    pairs = _ACTIVE_MOVEMENT_INPUTS["movement_pairs"]
    return {
        tuple(pair): float(sum(block[h][index] for h in range(horizon_rows)))
        for index, pair in enumerate(pairs)
    }


def movement_phase_score(
    phase,
    queues,
    movement_queues,
    capacities,
    lane_counts,
    entry_roads,
    future_movements,
    weight,
):
    pressure = 0.0
    for start, end, start_lanes in phase["movements"]:
        projected = movement_queues.get((start, end), 0.0)
        projected += weight * future_movements.get((start, end), 0.0)
        upstream = projected / capacities[start]
        downstream = queues.get(end, 0.0) / capacities[end]
        lane_share = start_lanes / lane_counts[start]
        pressure += max(0.0, upstream - downstream) * lane_share
    return pressure


def prepare_movement_inputs(
    demand_dir: Path,
    split_path: Path,
    prediction_dir: Path,
    validation_dates,
    test_dates,
    start_minute,
    end_minute,
):
    (
        values,
        entry_values,
        dates,
        movements,
        entries,
        movement_to_entry,
        movement_map,
        split_names,
        _split,
    ) = movement_baseline.load_data(demand_dir, split_path)
    features, target, origins, _ = movement_baseline.feature_matrix(entry_values, "entry")
    features = movement_baseline.add_weekday_features(features, dates, origins)
    train_mask = movement_baseline.row_mask(split_names, ["train"], origins)
    validation_mask = movement_baseline.row_mask(split_names, ["validation"], origins)

    point_manifest = json.loads((prediction_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    selected_leaf = int(point_manifest["protocol"]["selected_entry_min_samples_leaf"])
    selected_window = int(point_manifest["protocol"]["selected_share_window_minutes"])
    if selected_window != 60:
        raise AssertionError("Frozen movement interface must use the selected 60-minute share window")

    validation_model = movement_baseline.fit_extra_trees(
        features[train_mask],
        target[train_mask],
        selected_leaf,
        movement_baseline.FINAL_TREES,
    )
    validation_entry_prediction = np.clip(
        validation_model.predict(features[validation_mask]), 0, None
    ).astype(np.float32)
    train_values = values[split_names == "train"]
    validation_values = values[split_names == "validation"]
    global_share, _minute_share, _totals = movement_baseline.share_parameters(
        train_values, movement_to_entry, len(entries)
    )
    validation_prediction = movement_baseline.hierarchical_rolling(
        validation_entry_prediction,
        validation_values,
        origins,
        selected_window,
        global_share,
        movement_to_entry,
        len(entries),
    ).reshape(len(validation_values), len(origins), movement_baseline.HORIZON, len(movements))

    with np.load(prediction_dir / "test_predictions.npz") as archive:
        archive_movements = archive["movements"].astype(str).tolist()
        archive_origins = archive["origins"].astype(int)
        archive_dates = archive["dates"].astype(str).tolist()
        test_prediction = archive["hierarchical_rolling_share"].reshape(
            len(archive_dates), len(archive_origins), movement_baseline.HORIZON, len(movements)
        )
    if archive_movements != movements or not np.array_equal(archive_origins, origins):
        raise AssertionError("Prediction archive movement/origin order differs from current data")

    date_index = {date: index for index, date in enumerate(dates)}
    validation_all = [date for date, name in zip(dates, split_names) if name == "validation"]
    validation_index = {date: index for index, date in enumerate(validation_all)}
    test_index = {date: index for index, date in enumerate(archive_dates)}
    origin_index = {int(minute): index for index, minute in enumerate(origins)}
    movement_pairs = movement_map[["entry_road", "next_road"]].astype(str).values.tolist()
    prepared = {}
    for date in tuple(validation_dates) + tuple(test_dates):
        if date in validation_index:
            forecast_matrix = validation_prediction[validation_index[date]]
            forecast_fit = "train_only"
        elif date in test_index:
            forecast_matrix = test_prediction[test_index[date]]
            forecast_fit = "train_plus_validation"
        else:
            raise AssertionError(f"No frozen forecast for control date {date}")
        actual = values[date_index[date]]
        forecasts = {}
        oracle = {}
        for minute in range(start_minute, end_minute):
            index = origin_index[minute]
            forecasts[str(minute)] = forecast_matrix[index].tolist()
            oracle[str(minute)] = np.stack(
                [actual[minute + horizon] for horizon in range(1, movement_baseline.HORIZON + 1)],
                axis=0,
            ).tolist()
        prepared[date] = {
            "date": date,
            "forecast_fit": forecast_fit,
            "movement_ids": movements,
            "movement_pairs": movement_pairs,
            "forecast": forecasts,
            "oracle": oracle,
        }
    return prepared, point_manifest


def phase_movement_pairs(roadnet, metadata):
    network = v1.build_network(roadnet, metadata)
    controllers = network[2]
    return {
        (start, end)
        for phases in controllers.values()
        for phase in phases
        for start, end, _ in phase["movements"]
    }


def simulate(date_dir, roadnet, metadata, movement_inputs, mode, horizon_minutes, weight):
    global _ACTIVE_HORIZON_MINUTES, _ACTIVE_MOVEMENT_INPUTS
    _ACTIVE_HORIZON_MINUTES = horizon_minutes if mode in ("ForecastPressure", "OraclePressure") else 0
    _ACTIVE_MOVEMENT_INPUTS = movement_inputs[date_dir.name]
    return v2.simulate(date_dir, roadnet, metadata, mode, weight)


def build(assets_dir, demand_dir, split_path, prediction_dir, output_dir, roadnet_path):
    final_dir = output_dir.resolve()
    building = final_dir.with_name(final_dir.name + ".building")
    if final_dir.exists() or building.exists():
        raise FileExistsError(f"Output/build directory already exists: {final_dir}, {building}")
    building.mkdir(parents=True)
    metadata = json.loads((assets_dir / "ASSETS.json").read_text(encoding="utf-8"))
    roadnet = json.loads(roadnet_path.read_text(encoding="utf-8"))
    movement_inputs, point_manifest = prepare_movement_inputs(
        demand_dir,
        split_path,
        prediction_dir,
        metadata["validation_dates"],
        metadata["test_dates"],
        metadata["simulation"]["evaluation_start_minute"],
        metadata["simulation"]["evaluation_end_minute"],
    )
    vocabulary = {tuple(pair) for pair in next(iter(movement_inputs.values()))["movement_pairs"]}
    phase_pairs = phase_movement_pairs(roadnet, metadata)
    missing_pairs = sorted(vocabulary - phase_pairs)
    if missing_pairs:
        raise AssertionError(f"Movement vocabulary contains pairs absent from signal phases: {missing_pairs}")

    v1.demand_vector = movement_demand_vector
    v2.normalized_phase_score = movement_phase_score
    try:
        validation_results = []
        validation_modes = [("FixedTime", 0, 0.0), ("MaxPressure", 0, 0.0)] + [
            ("ForecastPressure", horizon, weight) for horizon, weight in POLICY_GRID
        ]
        for date in metadata["validation_dates"]:
            for mode, horizon, weight in validation_modes:
                row = simulate(
                    assets_dir / date, roadnet, metadata, movement_inputs, mode, horizon, weight
                )
                row.update(
                    {
                        "date": date,
                        "split": "validation",
                        "lookahead_minutes": horizon if mode == "ForecastPressure" else None,
                        "forecast_weight": weight if mode == "ForecastPressure" else None,
                    }
                )
                validation_results.append(row)
                print(
                    f"validation date={date} mode={mode} horizon={horizon} weight={weight} "
                    f"queue={row['queue_vehicle_seconds']:.1f}",
                    flush=True,
                )
        policy_scores = {
            f"{horizon}|{weight}": statistics.mean(
                row["queue_vehicle_seconds"]
                for row in validation_results
                if row["mode"] == "ForecastPressure"
                and row["lookahead_minutes"] == horizon
                and row["forecast_weight"] == weight
            )
            for horizon, weight in POLICY_GRID
        }
        selected_horizon, selected_weight = min(
            POLICY_GRID,
            key=lambda item: (policy_scores[f"{item[0]}|{item[1]}"], item[0], item[1]),
        )

        test_results = []
        for date in metadata["test_dates"]:
            for mode in ("FixedTime", "MaxPressure", "ForecastPressure", "OraclePressure"):
                row = simulate(
                    assets_dir / date,
                    roadnet,
                    metadata,
                    movement_inputs,
                    mode,
                    selected_horizon,
                    selected_weight,
                )
                row.update(
                    {
                        "date": date,
                        "split": "test",
                        "lookahead_minutes": (
                            selected_horizon if mode in ("ForecastPressure", "OraclePressure") else None
                        ),
                        "forecast_weight": (
                            selected_weight if mode in ("ForecastPressure", "OraclePressure") else None
                        ),
                    }
                )
                test_results.append(row)
                print(
                    f"test date={date} mode={mode} queue={row['queue_vehicle_seconds']:.1f} "
                    f"throughput={row['throughput_exit_vehicles']:.1f}",
                    flush=True,
                )
    finally:
        v1.demand_vector = _ORIGINAL_DEMAND_VECTOR
        v2.normalized_phase_score = _ORIGINAL_PHASE_SCORE

    summary = v1.summarize(test_results)
    result = {
        "experiment_id": "xuancheng_control_admission_movement_v4",
        "status": "COMPLETED_PENDING_REPRODUCIBILITY_RERUN",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "selected_lookahead_minutes": selected_horizon,
        "selected_forecast_weight": selected_weight,
        "validation_policy_mean_queue_vehicle_seconds": policy_scores,
        "validation_results": validation_results,
        "test_results": test_results,
        "summary": summary,
        "protocol": {
            "controller": "capacity-normalized movement-level predictive pressure",
            "prediction_interface": "HierarchicalRollingShare",
            "share_window_minutes": point_manifest["protocol"]["selected_share_window_minutes"],
            "policy_grid": [list(item) for item in POLICY_GRID],
            "selection_metric": "mean validation queue_vehicle_seconds",
            "simulation_step_seconds": v2.SIMULATION_STEP_SECONDS,
            "control_interval_seconds": v2.CONTROL_INTERVAL_SECONDS,
            "minimum_green_seconds": v2.MIN_GREEN_SECONDS,
            "switch_loss_seconds": v2.SWITCH_LOSS_SECONDS,
            "movement_count": len(vocabulary),
            "movement_pair_phase_coverage": len(vocabulary & phase_pairs),
            "validation_dates": metadata["validation_dates"],
            "test_dates": metadata["test_dates"],
            "validation_forecast_fit": "train only",
            "test_forecast_fit": "train plus validation",
            "sealed_split": "calibration",
        },
        "source_hashes": {
            "assets_manifest_sha256": sha256(assets_dir / "ASSETS.json"),
            "roadnet_sha256": sha256(roadnet_path),
            "movement_demand_sha256": sha256(demand_dir / "demand_by_movement_1min.csv"),
            "prediction_manifest_sha256": sha256(prediction_dir / "MANIFEST.json"),
            "test_predictions_sha256": sha256(prediction_dir / "test_predictions.npz"),
            "split_sha256": sha256(split_path),
        },
        "scope_warning": (
            "Admission screen only: boundary-injected demand, deterministic point queues, "
            "source-departure-time demand proxy, and no microscopic car-following."
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
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets-dir", type=Path,
        default=project / "04_实验" / "控制基线" / "control_admission_v1" / "assets_subnetwork_v2",
    )
    parser.add_argument(
        "--demand-dir", type=Path,
        default=project / "02_数据" / "派生数据" / "宣城_movement_demand_v1",
    )
    parser.add_argument(
        "--split-path", type=Path,
        default=project / "02_数据" / "派生数据" / "宣城_clean_v2" / "SPLIT.json",
    )
    parser.add_argument(
        "--prediction-dir", type=Path,
        default=project / "04_实验" / "预测基线" / "movement_point_baseline_v1" / "results_v1",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=project / "04_实验" / "控制基线" / "control_admission_v4" / "results_v1",
    )
    parser.add_argument(
        "--roadnet", type=Path,
        default=project / "04_实验" / "控制基线" / "control_admission_v1" / "assets_subnetwork_v2" / "roadnet_subnetwork.json",
    )
    args = parser.parse_args()
    result = build(
        args.assets_dir.resolve(),
        args.demand_dir.resolve(),
        args.split_path.resolve(),
        args.prediction_dir.resolve(),
        args.output_dir.resolve(),
        args.roadnet.resolve(),
    )
    print(json.dumps(
        {
            "status": result["status"],
            "selected_lookahead_minutes": result["selected_lookahead_minutes"],
            "selected_forecast_weight": result["selected_forecast_weight"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
