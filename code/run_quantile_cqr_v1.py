"""Train multi-quantile LightGBM forecasts and static CQR calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_point_baselines_v1 import (
    HORIZON,
    SEED,
    add_weekday_features,
    feature_matrix,
    load_data,
    save_csv,
    sha256,
)


QUANTILES = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
INTERVALS = ((0.50, 0.25, 0.75), (0.80, 0.10, 0.90), (0.90, 0.05, 0.95))
CHILD_GRID = (20, 50, 100)
TUNING_TREES = 140
FINAL_TREES = 220


def masks_for_splits(split_names: np.ndarray, origins: np.ndarray):
    return {
        name: np.repeat(split_names == name, len(origins))
        for name in ("train", "validation", "calibration", "test")
    }


def stacked_design(features, origin_minutes, target_by_horizon):
    rows = len(features)
    horizon = np.tile(np.arange(1, HORIZON + 1), rows)
    future_minute = (np.repeat(origin_minutes, HORIZON) + horizon) % 1440
    angle = 2 * np.pi * future_minute / 1440
    design = np.concatenate(
        [
            np.repeat(features, HORIZON, axis=0),
            (horizon / HORIZON)[:, None].astype(np.float32),
            np.sin(angle)[:, None].astype(np.float32),
            np.cos(angle)[:, None].astype(np.float32),
        ],
        axis=1,
    )
    return design.astype(np.float32), target_by_horizon.reshape(-1).astype(np.float32)


def make_model(alpha: float, child_samples: int, trees: int):
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=trees,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=child_samples,
        subsample=1.0,
        colsample_bytree=0.8,
        reg_lambda=0.1,
        random_state=SEED,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def pinball(actual, predicted, quantile):
    error = actual - predicted
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def conformal_adjustment(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    probability = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, probability, method="higher"))


def interval_metrics(actual, lower, upper, nominal):
    alpha = 1 - nominal
    covered = (actual >= lower) & (actual <= upper)
    score = (upper - lower) + (2 / alpha) * np.maximum(lower - actual, 0) + (2 / alpha) * np.maximum(actual - upper, 0)
    return {
        "coverage": float(covered.mean()),
        "coverage_error": float(covered.mean() - nominal),
        "mean_width": float((upper - lower).mean()),
        "interval_score": float(score.mean()),
    }


def enforce_monotonic_preserve_median(array):
    result = np.maximum(array, 0).copy()
    median_index = int(np.where(np.isclose(QUANTILES, 0.5))[0][0])
    for index in range(median_index - 1, -1, -1):
        result[..., index] = np.minimum(result[..., index], result[..., index + 1])
    for index in range(median_index + 1, len(QUANTILES)):
        result[..., index] = np.maximum(result[..., index], result[..., index - 1])
    return result


def evaluate_pinball(actual, forecast, entries, method):
    overall = []
    by_horizon = []
    by_entry = []
    for q_index, quantile in enumerate(QUANTILES):
        overall.append(
            {
                "method": method,
                "quantile": quantile,
                "pinball_loss": pinball(actual, forecast[..., q_index], quantile),
            }
        )
        for horizon in range(HORIZON):
            by_horizon.append(
                {
                    "method": method,
                    "quantile": quantile,
                    "horizon_min": horizon + 1,
                    "pinball_loss": pinball(
                        actual[:, horizon, :], forecast[:, horizon, :, q_index], quantile
                    ),
                }
            )
        for entry_index, entry in enumerate(entries):
            by_entry.append(
                {
                    "method": method,
                    "quantile": quantile,
                    "entry_road_id": entry,
                    "pinball_loss": pinball(
                        actual[:, :, entry_index], forecast[:, :, entry_index, q_index], quantile
                    ),
                }
            )
    return pd.DataFrame(overall), pd.DataFrame(by_horizon), pd.DataFrame(by_entry)


def evaluate_intervals(actual, forecast, entries, method, high_thresholds=None):
    overall = []
    by_horizon = []
    by_entry = []
    joint = []
    for nominal, q_lower, q_upper in INTERVALS:
        lower_index = int(np.where(np.isclose(QUANTILES, q_lower))[0][0])
        upper_index = int(np.where(np.isclose(QUANTILES, q_upper))[0][0])
        lower = forecast[..., lower_index]
        upper = forecast[..., upper_index]
        overall.append(
            {"method": method, "nominal_coverage": nominal, **interval_metrics(actual, lower, upper, nominal)}
        )
        for horizon in range(HORIZON):
            by_horizon.append(
                {
                    "method": method,
                    "nominal_coverage": nominal,
                    "horizon_min": horizon + 1,
                    **interval_metrics(actual[:, horizon, :], lower[:, horizon, :], upper[:, horizon, :], nominal),
                }
            )
        for entry_index, entry in enumerate(entries):
            row = {
                "method": method,
                "nominal_coverage": nominal,
                "entry_road_id": entry,
                **interval_metrics(actual[:, :, entry_index], lower[:, :, entry_index], upper[:, :, entry_index], nominal),
            }
            if high_thresholds is not None:
                high = actual[:, :, entry_index] >= high_thresholds[:, entry_index][None, :]
                row["high_demand_n"] = int(high.sum())
                row["high_demand_coverage"] = float(
                    (((actual[:, :, entry_index] >= lower[:, :, entry_index]) & (actual[:, :, entry_index] <= upper[:, :, entry_index]))[high]).mean()
                ) if high.any() else math.nan
            by_entry.append(row)
        covered = (actual >= lower) & (actual <= upper)
        joint.append(
            {
                "method": method,
                "nominal_coverage": nominal,
                "all_105_targets_covered_rate": float(covered.reshape(len(actual), -1).all(axis=1).mean()),
                "mean_fraction_of_105_targets_covered": float(covered.mean(axis=(1, 2)).mean()),
                "all_7_entries_covered_mean_over_horizons": float(covered.all(axis=2).mean()),
            }
        )
    return tuple(pd.DataFrame(rows) for rows in (overall, by_horizon, by_entry, joint))


def build(demand_dir: Path, split_path: Path, output_dir: Path):
    started = time.time()
    final_dir = output_dir.resolve()
    building = final_dir.with_name(final_dir.name + ".building")
    if final_dir.exists() or building.exists():
        raise FileExistsError(f"Output/build directory already exists: {final_dir}, {building}")
    building.mkdir(parents=True)
    values, dates, entries, split_names, split = load_data(demand_dir, split_path)
    features, target_flat, origins, feature_names = feature_matrix(values)
    features = add_weekday_features(features, dates, origins)
    row_origin_minutes = np.tile(origins, len(dates))
    masks = masks_for_splits(split_names, origins)
    targets = target_flat.reshape(len(target_flat), HORIZON, len(entries))

    tuning_rows = []
    for child_samples in CHILD_GRID:
        losses = []
        for entry_index, entry in enumerate(entries):
            train_x, train_y = stacked_design(
                features[masks["train"]],
                row_origin_minutes[masks["train"]],
                targets[masks["train"], :, entry_index],
            )
            validation_x, validation_y = stacked_design(
                features[masks["validation"]],
                row_origin_minutes[masks["validation"]],
                targets[masks["validation"], :, entry_index],
            )
            model = make_model(0.5, child_samples, TUNING_TREES)
            model.fit(train_x, train_y)
            prediction = np.maximum(model.predict(validation_x), 0)
            losses.append(pinball(validation_y, prediction, 0.5))
        tuning_rows.append(
            {
                "min_child_samples": child_samples,
                "validation_median_pinball": float(np.mean(losses)),
            }
        )
        print(
            f"tuning min_child_samples={child_samples} median_pinball={np.mean(losses):.6f}",
            flush=True,
        )
    tuning = pd.DataFrame(tuning_rows).sort_values(
        ["validation_median_pinball", "min_child_samples"]
    )
    selected_child = int(tuning.iloc[0]["min_child_samples"])
    save_csv(tuning, building / "validation_hyperparameter_selection.csv")

    fit_mask = masks["train"] | masks["validation"]
    calibration_rows = int(masks["calibration"].sum())
    test_rows = int(masks["test"].sum())
    raw_calibration = np.empty(
        (calibration_rows, HORIZON, len(entries), len(QUANTILES)), dtype=np.float32
    )
    raw_test = np.empty((test_rows, HORIZON, len(entries), len(QUANTILES)), dtype=np.float32)
    for entry_index, entry in enumerate(entries):
        fit_x, fit_y = stacked_design(
            features[fit_mask], row_origin_minutes[fit_mask], targets[fit_mask, :, entry_index]
        )
        calibration_x, _ = stacked_design(
            features[masks["calibration"]],
            row_origin_minutes[masks["calibration"]],
            targets[masks["calibration"], :, entry_index],
        )
        test_x, _ = stacked_design(
            features[masks["test"]],
            row_origin_minutes[masks["test"]],
            targets[masks["test"], :, entry_index],
        )
        for q_index, quantile in enumerate(QUANTILES):
            model = make_model(float(quantile), selected_child, FINAL_TREES)
            model.fit(fit_x, fit_y)
            raw_calibration[:, :, entry_index, q_index] = model.predict(calibration_x).reshape(
                calibration_rows, HORIZON
            )
            raw_test[:, :, entry_index, q_index] = model.predict(test_x).reshape(
                test_rows, HORIZON
            )
        print(f"trained entry={entry} quantile_models={len(QUANTILES)}", flush=True)

    crossing_calibration = float(np.mean(np.any(np.diff(raw_calibration, axis=-1) < 0, axis=-1)))
    crossing_test = float(np.mean(np.any(np.diff(raw_test, axis=-1) < 0, axis=-1)))
    raw_calibration = np.maximum(np.sort(raw_calibration, axis=-1), 0)
    raw_test = np.maximum(np.sort(raw_test, axis=-1), 0)
    actual_calibration = targets[masks["calibration"]]
    actual_test = targets[masks["test"]]

    calibrated_test = raw_test.copy()
    adjustment_rows = []
    for nominal, q_lower, q_upper in INTERVALS:
        alpha = 1 - nominal
        lower_index = int(np.where(np.isclose(QUANTILES, q_lower))[0][0])
        upper_index = int(np.where(np.isclose(QUANTILES, q_upper))[0][0])
        for horizon in range(HORIZON):
            for entry_index, entry in enumerate(entries):
                scores = np.maximum(
                    raw_calibration[:, horizon, entry_index, lower_index]
                    - actual_calibration[:, horizon, entry_index],
                    actual_calibration[:, horizon, entry_index]
                    - raw_calibration[:, horizon, entry_index, upper_index],
                )
                adjustment = conformal_adjustment(scores, alpha)
                calibrated_test[:, horizon, entry_index, lower_index] = (
                    raw_test[:, horizon, entry_index, lower_index] - adjustment
                )
                calibrated_test[:, horizon, entry_index, upper_index] = (
                    raw_test[:, horizon, entry_index, upper_index] + adjustment
                )
                adjustment_rows.append(
                    {
                        "nominal_coverage": nominal,
                        "horizon_min": horizon + 1,
                        "entry_road_id": entry,
                        "calibration_n": len(scores),
                        "adjustment": adjustment,
                    }
                )
    calibrated_test = enforce_monotonic_preserve_median(calibrated_test)
    adjustments = pd.DataFrame(adjustment_rows)
    save_csv(adjustments, building / "cqr_adjustments.csv")

    high_thresholds = np.quantile(actual_calibration, 0.9, axis=0, method="higher")
    pinball_frames = []
    pinball_horizon_frames = []
    pinball_entry_frames = []
    interval_frames = []
    interval_horizon_frames = []
    interval_entry_frames = []
    joint_frames = []
    for method, forecast in (("Raw", raw_test), ("CQR", calibrated_test)):
        p, ph, pe = evaluate_pinball(actual_test, forecast, entries, method)
        i, ih, ie, j = evaluate_intervals(
            actual_test, forecast, entries, method, high_thresholds=high_thresholds
        )
        pinball_frames.append(p)
        pinball_horizon_frames.append(ph)
        pinball_entry_frames.append(pe)
        interval_frames.append(i)
        interval_horizon_frames.append(ih)
        interval_entry_frames.append(ie)
        joint_frames.append(j)
    pinball_overall = pd.concat(pinball_frames, ignore_index=True)
    pinball_by_horizon = pd.concat(pinball_horizon_frames, ignore_index=True)
    pinball_by_entry = pd.concat(pinball_entry_frames, ignore_index=True)
    intervals_overall = pd.concat(interval_frames, ignore_index=True)
    intervals_by_horizon = pd.concat(interval_horizon_frames, ignore_index=True)
    intervals_by_entry = pd.concat(interval_entry_frames, ignore_index=True)
    joint = pd.concat(joint_frames, ignore_index=True)
    save_csv(pinball_overall, building / "test_pinball_by_quantile.csv")
    save_csv(pinball_by_horizon, building / "test_pinball_by_horizon.csv")
    save_csv(pinball_by_entry, building / "test_pinball_by_entry.csv")
    save_csv(intervals_overall, building / "test_interval_metrics.csv")
    save_csv(intervals_by_horizon, building / "test_interval_metrics_by_horizon.csv")
    save_csv(intervals_by_entry, building / "test_interval_metrics_by_entry.csv")
    save_csv(joint, building / "test_joint_coverage.csv")

    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    for method, group in intervals_overall.groupby("method", sort=False):
        ax.plot(
            group["nominal_coverage"],
            group["coverage"],
            marker="o",
            linewidth=2,
            label=method,
        )
    ax.plot([0.45, 0.95], [0.45, 0.95], linestyle="--", color="#555555", label="Ideal")
    ax.set_xlim(0.45, 0.95)
    ax.set_ylim(0.45, 0.95)
    ax.set_xlabel("Nominal marginal coverage")
    ax.set_ylabel("Empirical test coverage")
    ax.set_title("Raw and CQR interval reliability")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(building / "fig_interval_reliability.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    selected = intervals_by_horizon[intervals_by_horizon["nominal_coverage"] == 0.9]
    for method, group in selected.groupby("method", sort=False):
        ax.plot(group["horizon_min"], group["coverage"], marker="o", label=method)
    ax.axhline(0.9, linestyle="--", color="#555555", label="Nominal 90%")
    ax.set_xticks(range(1, HORIZON + 1))
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Test coverage of the nominal 90% interval")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(building / "fig_90pct_coverage_by_horizon.png", dpi=180)
    plt.close(fig)

    np.savez_compressed(
        building / "test_quantile_predictions.npz",
        actual=actual_test.astype(np.float32),
        raw=raw_test.astype(np.float32),
        cqr=calibrated_test.astype(np.float32),
        quantiles=QUANTILES.astype(np.float32),
        entries=np.array(entries),
        dates=np.array([date for date, name in zip(dates, split_names) if name == "test"]),
        origins=origins.astype(np.int16),
    )

    outputs = {}
    for path in sorted(building.iterdir()):
        if path.is_file():
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    summary = {
        "raw_crossing_rate_before_sort_calibration": crossing_calibration,
        "raw_crossing_rate_before_sort_test": crossing_test,
        "mean_pinball": {
            method: float(group["pinball_loss"].mean())
            for method, group in pinball_overall.groupby("method")
        },
        "interval_metrics": intervals_overall.to_dict(orient="records"),
        "joint_coverage": joint.to_dict(orient="records"),
    }
    manifest = {
        "experiment_id": "xuancheng_quantile_cqr_v1",
        "status": "COMPLETED_PENDING_REPRODUCIBILITY_RERUN",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {
            "dataset_id": "xuancheng_subnetwork_demand_v1",
            "demand_detail_sha256": sha256(demand_dir / "demand_by_entry_1min.csv"),
            "split_sha256": sha256(split_path),
            "split_id": split["split_id"],
        },
        "protocol": {
            "quantiles": QUANTILES.tolist(),
            "intervals": [list(item) for item in INTERVALS],
            "history_minutes": 60,
            "horizon_minutes": HORIZON,
            "seed": SEED,
            "min_child_grid": CHILD_GRID,
            "selected_min_child_samples": selected_child,
            "fit_splits": ["train", "validation"],
            "calibration_split": "calibration",
            "evaluation_split": "test",
            "crossing_correction": "sort raw quantiles per target",
            "cqr_scope": "separate by entry, horizon and central interval",
            "median_preserved_by_cqr": True,
        },
        "sample_counts": {
            name: int(mask.sum()) for name, mask in masks.items()
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
        },
        "summary": summary,
        "outputs": outputs,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (building / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(building, final_dir)
    return manifest


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demand-dir",
        type=Path,
        default=project / "02_数据" / "派生数据" / "宣城_subnetwork_demand_v1",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=project / "02_数据" / "派生数据" / "宣城_clean_v2" / "SPLIT.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "04_实验" / "概率预测" / "quantile_cqr_v1" / "results",
    )
    args = parser.parse_args()
    result = build(args.demand_dir, args.split_path, args.output_dir)
    print(json.dumps({"status": result["status"], "summary": result["summary"], "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
