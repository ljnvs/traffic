"""Compare independent and empirical-copula demand scenarios with equal marginals."""

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
from scipy.stats import rankdata

from run_point_baselines_v1 import HORIZON, add_weekday_features, feature_matrix, load_data, save_csv, sha256
from run_quantile_cqr_v1 import FINAL_TREES, QUANTILES, make_model, stacked_design


SEED = 20260920
SCENARIOS = 100
DEPENDENCE_QUANTILES = (0.10, 0.50, 0.90)
VARIOGRAM_POWER = 0.5
EPSILON = 1e-6


def masks_for_splits(split_names, origins):
    return {
        name: np.repeat(split_names == name, len(origins))
        for name in ("train", "validation", "calibration", "test")
    }


def inverse_piecewise_quantile(uniforms: np.ndarray, quantile_values: np.ndarray):
    """Map S x D uniforms through D x Q quantile functions."""
    lower_tail = np.maximum(
        0,
        quantile_values[:, 0]
        - (quantile_values[:, 1] - quantile_values[:, 0]),
    )
    upper_tail = quantile_values[:, -1] + (
        quantile_values[:, -1] - quantile_values[:, -2]
    )
    levels = np.concatenate(([0.0], QUANTILES, [1.0]))
    values = np.concatenate(
        [lower_tail[:, None], quantile_values, upper_tail[:, None]], axis=1
    )
    result = np.empty_like(uniforms, dtype=np.float32)
    for interval in range(len(levels) - 1):
        if interval == len(levels) - 2:
            mask = (uniforms >= levels[interval]) & (uniforms <= levels[interval + 1])
        else:
            mask = (uniforms >= levels[interval]) & (uniforms < levels[interval + 1])
        weight = (uniforms - levels[interval]) / (levels[interval + 1] - levels[interval])
        interpolated = values[:, interval][None, :] + weight * (
            values[:, interval + 1] - values[:, interval]
        )[None, :]
        result[mask] = interpolated[mask]
    return np.maximum(result, 0)


def structured_pairs(entries: int):
    temporal_i = []
    temporal_j = []
    for horizon in range(HORIZON - 1):
        for entry in range(entries):
            temporal_i.append(horizon * entries + entry)
            temporal_j.append((horizon + 1) * entries + entry)
    spatial_i = []
    spatial_j = []
    for horizon in range(HORIZON):
        for first in range(entries):
            for second in range(first + 1, entries):
                spatial_i.append(horizon * entries + first)
                spatial_j.append(horizon * entries + second)
    return (
        np.array(temporal_i + spatial_i, dtype=np.int16),
        np.array(temporal_j + spatial_j, dtype=np.int16),
        len(temporal_i),
        len(spatial_i),
    )


def longest_run(mask: np.ndarray) -> np.ndarray:
    """Longest True run on last axis."""
    current = np.zeros(mask.shape[:-1], dtype=np.int16)
    longest = np.zeros_like(current)
    for index in range(mask.shape[-1]):
        current = np.where(mask[..., index], current + 1, 0)
        longest = np.maximum(longest, current)
    return longest


def scenario_metrics(scenarios, actual, pair_i, pair_j, thresholds):
    difference = scenarios - actual[None, :]
    term_one = np.linalg.norm(difference, axis=1).mean()
    term_two = np.linalg.norm(scenarios - np.roll(scenarios, 1, axis=0), axis=1).mean()
    energy = float(term_one - 0.5 * term_two)
    observed_variogram = np.abs(actual[pair_i] - actual[pair_j]) ** VARIOGRAM_POWER
    forecast_variogram = np.mean(
        np.abs(scenarios[:, pair_i] - scenarios[:, pair_j]) ** VARIOGRAM_POWER,
        axis=0,
    )
    variogram = float(np.mean((observed_variogram - forecast_variogram) ** 2))
    shaped = scenarios.reshape(SCENARIOS, HORIZON, -1)
    total_by_horizon = shaped.sum(axis=2)
    cumulative_probability = float(
        np.mean(total_by_horizon.sum(axis=1) > thresholds["cumulative"])
    )
    peak_probability = float(
        np.mean(total_by_horizon.max(axis=1) > thresholds["peak"])
    )
    persistent_probability = float(
        np.mean(longest_run(total_by_horizon > thresholds["horizon_q75"][None, :]) >= 3)
    )
    return energy, variogram, cumulative_probability, peak_probability, persistent_probability


def event_outcomes(actual, thresholds):
    total = actual.sum(axis=2)
    return {
        "cumulative": total.sum(axis=1) > thresholds["cumulative"],
        "peak": total.max(axis=1) > thresholds["peak"],
        "persistent": longest_run(total > thresholds["horizon_q75"][None, :]) >= 3,
    }


def dependence_diagnostics(templates, rng, pair_i, pair_j):
    sample_n = 20000
    indices = rng.integers(0, len(templates), size=sample_n)
    correlated = templates[indices]
    orders = np.argsort(rng.random(correlated.shape), axis=0)
    independent = np.take_along_axis(correlated, orders, axis=0)
    rows = []
    template_corr = []
    correlated_corr = []
    independent_corr = []
    for first, second in zip(pair_i, pair_j):
        template_corr.append(np.corrcoef(templates[:, first], templates[:, second])[0, 1])
        correlated_corr.append(np.corrcoef(correlated[:, first], correlated[:, second])[0, 1])
        independent_corr.append(np.corrcoef(independent[:, first], independent[:, second])[0, 1])
    template_corr = np.array(template_corr)
    for method, values in (
        ("Correlated", np.array(correlated_corr)),
        ("Independent", np.array(independent_corr)),
    ):
        rows.append(
            {
                "method": method,
                "pair_count": len(pair_i),
                "mean_absolute_pair_correlation": float(np.mean(np.abs(values))),
                "correlation_rmse_vs_calibration_template": float(
                    np.sqrt(np.mean((values - template_corr) ** 2))
                ),
                "mean_signed_pair_correlation": float(np.mean(values)),
                "calibration_template_mean_absolute_correlation": float(
                    np.mean(np.abs(template_corr))
                ),
            }
        )
    return pd.DataFrame(rows)


def build(demand_dir: Path, split_path: Path, forecast_dir: Path, output_dir: Path):
    started = time.time()
    final_dir = output_dir.resolve()
    building = final_dir.with_name(final_dir.name + ".building")
    if final_dir.exists() or building.exists():
        raise FileExistsError(f"Output/build directory already exists: {final_dir}, {building}")
    building.mkdir(parents=True)

    forecast_manifest = json.loads((forecast_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    forecast_verification = json.loads((forecast_dir / "VERIFICATION.json").read_text(encoding="utf-8"))
    if forecast_verification["status"] != "VERIFIED":
        raise AssertionError("Source quantile forecasts are not verified")
    selected_child = int(forecast_manifest["protocol"]["selected_min_child_samples"])
    values, dates, entries, split_names, split = load_data(demand_dir, split_path)
    features, target_flat, origins, _ = feature_matrix(values)
    features = add_weekday_features(features, dates, origins)
    row_origin_minutes = np.tile(origins, len(dates))
    masks = masks_for_splits(split_names, origins)
    targets = target_flat.reshape(len(target_flat), HORIZON, len(entries))
    fit_mask = masks["train"] | masks["validation"]
    calibration_rows = int(masks["calibration"].sum())

    calibration_predictions = np.empty(
        (calibration_rows, HORIZON, len(entries), len(DEPENDENCE_QUANTILES)),
        dtype=np.float32,
    )
    for entry_index, entry in enumerate(entries):
        fit_x, fit_y = stacked_design(
            features[fit_mask], row_origin_minutes[fit_mask], targets[fit_mask, :, entry_index]
        )
        calibration_x, _ = stacked_design(
            features[masks["calibration"]],
            row_origin_minutes[masks["calibration"]],
            targets[masks["calibration"], :, entry_index],
        )
        for quantile_index, quantile in enumerate(DEPENDENCE_QUANTILES):
            model = make_model(float(quantile), selected_child, FINAL_TREES)
            model.fit(fit_x, fit_y)
            calibration_predictions[:, :, entry_index, quantile_index] = model.predict(
                calibration_x
            ).reshape(calibration_rows, HORIZON)
        print(f"dependence-model entry={entry} models={len(DEPENDENCE_QUANTILES)}", flush=True)

    calibration_predictions = np.sort(calibration_predictions, axis=-1)
    actual_calibration = targets[masks["calibration"]]
    scale = (
        calibration_predictions[..., 2] - calibration_predictions[..., 0] + EPSILON
    )
    standardized_residual = (
        actual_calibration - calibration_predictions[..., 1]
    ) / scale
    flat_residual = standardized_residual.reshape(calibration_rows, -1)
    templates = np.empty_like(flat_residual, dtype=np.float32)
    for dimension in range(flat_residual.shape[1]):
        templates[:, dimension] = (
            rankdata(flat_residual[:, dimension], method="average") - 0.5
        ) / calibration_rows

    prediction_archive = np.load(forecast_dir / "test_quantile_predictions.npz")
    actual_test = prediction_archive["actual"].astype(np.float32)
    cqr_test = prediction_archive["cqr"].astype(np.float32)
    if actual_test.shape != (9562, HORIZON, len(entries)):
        raise AssertionError(f"Unexpected test shape: {actual_test.shape}")
    if not np.allclose(prediction_archive["quantiles"], QUANTILES, atol=1e-7, rtol=0):
        raise AssertionError("Quantile grid mismatch")

    calibration_total = actual_calibration.sum(axis=2)
    thresholds = {
        "cumulative": float(np.quantile(calibration_total.sum(axis=1), 0.9, method="higher")),
        "peak": float(np.quantile(calibration_total.max(axis=1), 0.9, method="higher")),
        "horizon_q75": np.quantile(calibration_total, 0.75, axis=0, method="higher"),
    }
    outcomes = event_outcomes(actual_test, thresholds)
    pair_i, pair_j, temporal_pairs, spatial_pairs = structured_pairs(len(entries))
    rng = np.random.default_rng(SEED)
    diagnostics_rng = np.random.default_rng(SEED + 1)
    dependence = dependence_diagnostics(templates, diagnostics_rng, pair_i, pair_j)
    save_csv(dependence, building / "dependence_diagnostics.csv")

    method_values = {
        method: {
            "energy": np.empty(len(actual_test), dtype=np.float32),
            "variogram": np.empty(len(actual_test), dtype=np.float32),
            "cumulative_probability": np.empty(len(actual_test), dtype=np.float32),
            "peak_probability": np.empty(len(actual_test), dtype=np.float32),
            "persistent_probability": np.empty(len(actual_test), dtype=np.float32),
        }
        for method in ("Independent", "Correlated")
    }
    max_sorted_difference = 0.0
    max_mean_difference = 0.0
    audit_correlated = []
    audit_independent = []
    for origin_index in range(len(actual_test)):
        sampled_indices = rng.integers(0, calibration_rows, size=SCENARIOS)
        correlated_u = templates[sampled_indices]
        orders = np.argsort(rng.random(correlated_u.shape), axis=0)
        independent_u = np.take_along_axis(correlated_u, orders, axis=0)
        q_values = cqr_test[origin_index].reshape(-1, len(QUANTILES))
        correlated_scenarios = inverse_piecewise_quantile(correlated_u, q_values)
        independent_scenarios = inverse_piecewise_quantile(independent_u, q_values)
        max_sorted_difference = max(
            max_sorted_difference,
            float(
                np.max(
                    np.abs(
                        np.sort(correlated_scenarios, axis=0)
                        - np.sort(independent_scenarios, axis=0)
                    )
                )
            ),
        )
        max_mean_difference = max(
            max_mean_difference,
            float(
                np.max(
                    np.abs(
                        correlated_scenarios.mean(axis=0)
                        - independent_scenarios.mean(axis=0)
                    )
                )
            ),
        )
        for method, scenarios in (
            ("Independent", independent_scenarios),
            ("Correlated", correlated_scenarios),
        ):
            results = scenario_metrics(
                scenarios,
                actual_test[origin_index].reshape(-1),
                pair_i,
                pair_j,
                thresholds,
            )
            for key, value in zip(method_values[method], results):
                method_values[method][key][origin_index] = value
        if origin_index < 16:
            audit_correlated.append(correlated_scenarios)
            audit_independent.append(independent_scenarios)
        if (origin_index + 1) % 1000 == 0:
            print(f"scenario-origins={origin_index + 1}/{len(actual_test)}", flush=True)

    test_dates = [date for date, name in zip(dates, split_names) if name == "test"]
    date_column = np.repeat(test_dates, len(origins))
    score_rows = []
    daily_rows = []
    event_rows = []
    for method, values_by_metric in method_values.items():
        row = {
            "method": method,
            "energy_score": float(values_by_metric["energy"].mean()),
            "variogram_score": float(values_by_metric["variogram"].mean()),
        }
        for event, probability_key in (
            ("cumulative", "cumulative_probability"),
            ("peak", "peak_probability"),
            ("persistent", "persistent_probability"),
        ):
            probabilities = values_by_metric[probability_key]
            observed = outcomes[event].astype(np.float32)
            brier = float(np.mean((probabilities - observed) ** 2))
            row[f"{event}_brier"] = brier
            row[f"{event}_observed_rate"] = float(observed.mean())
            row[f"{event}_mean_probability"] = float(probabilities.mean())
            for date in test_dates:
                mask = date_column == date
                event_rows.append(
                    {
                        "method": method,
                        "date": date,
                        "event": event,
                        "observed_rate": float(observed[mask].mean()),
                        "mean_probability": float(probabilities[mask].mean()),
                        "brier_score": float(np.mean((probabilities[mask] - observed[mask]) ** 2)),
                    }
                )
        score_rows.append(row)
        for date in test_dates:
            mask = date_column == date
            daily_rows.append(
                {
                    "method": method,
                    "date": date,
                    "energy_score": float(values_by_metric["energy"][mask].mean()),
                    "variogram_score": float(values_by_metric["variogram"][mask].mean()),
                }
            )
    scores = pd.DataFrame(score_rows)
    daily_scores = pd.DataFrame(daily_rows)
    event_scores = pd.DataFrame(event_rows)
    save_csv(scores, building / "test_scenario_scores.csv")
    save_csv(daily_scores, building / "test_scenario_scores_by_day.csv")
    save_csv(event_scores, building / "test_event_scores_by_day.csv")

    marginal_check = {
        "status": "PASS" if max_sorted_difference == 0 else "FAIL",
        "max_absolute_sorted_sample_difference": max_sorted_difference,
        "max_absolute_sample_mean_difference": max_mean_difference,
        "scope": "every test origin and all 105 dimensions",
    }
    (building / "MARGINAL_CHECK.json").write_text(
        json.dumps(marginal_check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        building / "audit_scenarios_first16.npz",
        actual=actual_test[:16],
        correlated=np.stack(audit_correlated),
        independent=np.stack(audit_independent),
        entries=np.array(entries),
        origins=origins[:16],
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for axis, metric, title in (
        (axes[0], "energy_score", "Energy score"),
        (axes[1], "variogram_score", "Variogram score"),
    ):
        values_plot = scores.set_index("method")[metric]
        axis.bar(values_plot.index, values_plot.values, color=["#64748B", "#2563EB"])
        axis.set_title(title)
        axis.set_ylabel("Lower is better")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Equal-marginal scenario comparison")
    fig.tight_layout()
    fig.savefig(building / "fig_scenario_scores.png", dpi=180)
    plt.close(fig)

    outputs = {}
    for path in sorted(building.iterdir()):
        if path.is_file():
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "experiment_id": "xuancheng_copula_scenarios_v1",
        "status": "COMPLETED_PENDING_REPRODUCIBILITY_RERUN",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {
            "dataset_id": "xuancheng_subnetwork_demand_v1",
            "forecast_experiment_id": forecast_manifest["experiment_id"],
            "forecast_archive_sha256": sha256(forecast_dir / "test_quantile_predictions.npz"),
            "split_sha256": sha256(split_path),
        },
        "protocol": {
            "scenarios": SCENARIOS,
            "seed": SEED,
            "dimensions": HORIZON * len(entries),
            "dependence_template_rows": calibration_rows,
            "dependence_residual": "rank of (y-q50)/(q90-q10+1e-6)",
            "correlated_sampling": "same calibration block row across all dimensions",
            "independent_sampling": "dimension-wise permutation of the same marginal samples",
            "marginal_source": "verified CQR test quantiles",
            "variogram_power": VARIOGRAM_POWER,
            "temporal_pairs": temporal_pairs,
            "spatial_pairs": spatial_pairs,
            "test_evaluation_only": True,
        },
        "event_thresholds": {
            "cumulative_q90": thresholds["cumulative"],
            "peak_q90": thresholds["peak"],
            "horizon_q75": thresholds["horizon_q75"].tolist(),
        },
        "marginal_check": marginal_check,
        "scores": scores.to_dict(orient="records"),
        "dependence_diagnostics": dependence.to_dict(orient="records"),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "lightgbm": lgb.__version__,
        },
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
        "--forecast-dir",
        type=Path,
        default=project / "04_实验" / "概率预测" / "quantile_cqr_v1" / "results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "04_实验" / "场景生成" / "copula_scenarios_v1" / "results",
    )
    args = parser.parse_args()
    result = build(args.demand_dir, args.split_path, args.forecast_dir, args.output_dir)
    print(json.dumps({"status": result["status"], "marginal_check": result["marginal_check"], "scores": result["scores"], "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
