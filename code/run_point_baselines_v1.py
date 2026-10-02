"""EDA and leakage-safe point forecasting baselines for Xuancheng demand v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor


SEED = 20260920
HISTORY = 60
HORIZON = 15
LAGS = (0, 1, 4, 14, 29, 59)
ROLLING_WINDOWS = (5, 15, 30, 60)
LEAF_GRID = (1, 3, 8)
TUNING_TREES = 120
FINAL_TREES = 240


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def load_data(demand_dir: Path, split_path: Path):
    split = json.loads(split_path.read_text(encoding="utf-8"))
    date_to_split = {
        date: name
        for name, details in split["splits"].items()
        for date in details["dates"]
    }
    detail = pd.read_csv(
        demand_dir / "demand_by_entry_1min.csv",
        dtype={"date": "string", "split": "string", "entry_road_id": "string"},
    )
    if detail.duplicated(["date", "minute", "entry_road_id"]).any():
        raise AssertionError("Duplicate date-minute-entry key")
    if set(detail["date"]) != set(date_to_split):
        raise AssertionError("Demand dates differ from SPLIT.json")
    expected_split = detail["date"].map(date_to_split)
    if not (expected_split == detail["split"]).all():
        raise AssertionError("Demand split labels differ from SPLIT.json")
    entries = sorted(detail["entry_road_id"].unique().tolist())
    dates = sorted(detail["date"].unique().tolist())
    pivot = detail.pivot(index=["date", "minute"], columns="entry_road_id", values="vehicle_count")
    expected_index = pd.MultiIndex.from_product(
        [dates, range(1440)], names=["date", "minute"]
    )
    pivot = pivot.reindex(expected_index).reindex(columns=entries)
    if pivot.isna().any().any():
        raise AssertionError("Incomplete date-minute-entry grid")
    values = pivot.to_numpy(dtype=np.float32).reshape(len(dates), 1440, len(entries))
    split_names = np.array([date_to_split[date] for date in dates])
    return values, dates, entries, split_names, split


def within_day_autocorr(values: np.ndarray, lag: int) -> np.ndarray:
    left = values[:, :-lag, :].reshape(-1, values.shape[-1])
    right = values[:, lag:, :].reshape(-1, values.shape[-1])
    result = []
    for column in range(values.shape[-1]):
        result.append(float(np.corrcoef(left[:, column], right[:, column])[0, 1]))
    return np.array(result)


def build_eda(values, dates, entries, split_names, output_dir):
    train = values[split_names == "train"]
    rows = []
    flat_train = train.reshape(-1, len(entries))
    for index, entry in enumerate(entries):
        series = flat_train[:, index]
        mean = float(series.mean())
        variance = float(series.var(ddof=1))
        rows.append(
            {
                "entry_road_id": entry,
                "mean": mean,
                "std": float(series.std(ddof=1)),
                "variance": variance,
                "variance_mean_ratio": variance / mean if mean else math.nan,
                "zero_rate": float(np.mean(series == 0)),
                "p50": float(np.quantile(series, 0.50)),
                "p90": float(np.quantile(series, 0.90)),
                "p95": float(np.quantile(series, 0.95)),
                "p99": float(np.quantile(series, 0.99)),
                "maximum": float(series.max()),
            }
        )
    summary = pd.DataFrame(rows)
    save_csv(summary, output_dir / "eda_entry_summary_train.csv")

    autocorr_rows = []
    for lag in (1, 5, 15, 30, 60):
        correlations = within_day_autocorr(train, lag)
        for entry, value in zip(entries, correlations):
            autocorr_rows.append(
                {"entry_road_id": entry, "lag_minutes": lag, "autocorrelation": value}
            )
    autocorr = pd.DataFrame(autocorr_rows)
    save_csv(autocorr, output_dir / "eda_autocorrelation_train.csv")

    correlation = pd.DataFrame(np.corrcoef(flat_train, rowvar=False), index=entries, columns=entries)
    correlation.index.name = "entry_road_id"
    correlation.to_csv(output_dir / "eda_entry_correlation_train.csv", encoding="utf-8-sig")

    daily_rows = []
    for day_index, date in enumerate(dates):
        for entry_index, entry in enumerate(entries):
            daily_rows.append(
                {
                    "date": date,
                    "split": split_names[day_index],
                    "entry_road_id": entry,
                    "daily_total": int(values[day_index, :, entry_index].sum()),
                }
            )
    daily = pd.DataFrame(daily_rows)
    save_csv(daily, output_dir / "eda_daily_totals.csv")

    totals = values.sum(axis=2)
    colors = {"train": "#2563EB", "validation": "#16A34A", "calibration": "#D97706", "test": "#DC2626"}
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(dates))
    bars = ax.bar(x, totals.sum(axis=1), color=[colors[name] for name in split_names])
    ax.set_xticks(x)
    ax.set_xticklabels([date[5:] for date in dates], rotation=55, ha="right")
    ax.set_ylabel("Vehicles per day")
    ax.set_title("Daily first-entry demand by frozen split")
    ax.grid(axis="y", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors.values()]
    ax.legend(handles, list(colors), ncol=4, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_daily_total_by_split.png", dpi=180)
    plt.close(fig)

    train_profile = train.mean(axis=0).sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(np.arange(1440) / 60, train_profile, color="#1D4ED8", linewidth=1.4)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean vehicles per minute")
    ax.set_title("Training-period mean total entry-demand profile")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_train_mean_daily_profile.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(correlation.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    short = [entry[-5:] for entry in entries]
    ax.set_xticks(range(len(entries)), short, rotation=45, ha="right")
    ax.set_yticks(range(len(entries)), short)
    ax.set_title("Training-period contemporaneous correlation")
    fig.colorbar(image, ax=ax, shrink=0.82, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_train_entry_correlation.png", dpi=180)
    plt.close(fig)
    return summary, autocorr, correlation, daily


def feature_matrix(values: np.ndarray):
    days, minutes, entries = values.shape
    origins = np.arange(HISTORY - 1, minutes - HORIZON)
    rows = days * len(origins)
    feature_blocks = []
    names = []
    for lag in LAGS:
        block = values[:, origins - lag, :].reshape(rows, entries)
        feature_blocks.append(block)
        names.extend([f"entry_{entry}_lag_{lag}" for entry in range(entries)])
    cumulative = np.concatenate(
        [np.zeros((days, 1, entries), dtype=np.float32), np.cumsum(values, axis=1)], axis=1
    )
    for window in ROLLING_WINDOWS:
        end = origins + 1
        start = end - window
        block = (cumulative[:, end, :] - cumulative[:, start, :]) / window
        feature_blocks.append(block.reshape(rows, entries))
        names.extend([f"entry_{entry}_mean_{window}" for entry in range(entries)])
    minute = np.tile(origins, days)
    angle = 2 * np.pi * minute / 1440
    feature_blocks.extend([np.sin(angle)[:, None], np.cos(angle)[:, None]])
    names.extend(["minute_sin", "minute_cos"])
    target_blocks = [values[:, origins + h, :] for h in range(1, HORIZON + 1)]
    target = np.stack(target_blocks, axis=2).reshape(rows, entries * HORIZON)
    return np.concatenate(feature_blocks, axis=1).astype(np.float32), target.astype(np.float32), origins, names


def add_weekday_features(features, dates, origins):
    weekdays = np.array([pd.Timestamp(date).weekday() for date in dates])
    repeated = np.repeat(weekdays, len(origins))
    angle = 2 * np.pi * repeated / 7
    return np.concatenate([features, np.sin(angle)[:, None], np.cos(angle)[:, None]], axis=1).astype(np.float32)


def subset_rows(array, split_names, target_split, origins):
    mask_days = split_names == target_split
    mask_rows = np.repeat(mask_days, len(origins))
    return array[mask_rows], mask_rows


def historical_mean_prediction(train_values, selected_values, origins):
    profile = train_values.mean(axis=0)
    blocks = [profile[origins + h, :] for h in range(1, HORIZON + 1)]
    one_day = np.stack(blocks, axis=1).reshape(len(origins), -1)
    return np.tile(one_day, (selected_values.shape[0], 1)).astype(np.float32)


def persistence_prediction(selected_values, origins):
    current = selected_values[:, origins, :]
    return np.repeat(current[:, :, None, :], HORIZON, axis=2).reshape(
        selected_values.shape[0] * len(origins), -1
    ).astype(np.float32)


def metric_values(actual, predicted):
    error = predicted - actual
    absolute = np.abs(error)
    denominator = np.abs(actual) + np.abs(predicted)
    smape_terms = np.divide(2 * absolute, denominator, out=np.zeros_like(absolute), where=denominator > 0)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(absolute.sum() / np.abs(actual).sum()),
        "smape": float(smape_terms.mean()),
        "bias": float(error.mean()),
    }


def evaluate_models(actual, predictions, entries, test_dates, origins):
    rows_horizon = []
    rows_entry = []
    rows_daily = []
    rows_overall = []
    shaped_actual = actual.reshape(-1, HORIZON, len(entries))
    for model, predicted in predictions.items():
        shaped_predicted = predicted.reshape(-1, HORIZON, len(entries))
        rows_overall.append({"model": model, **metric_values(actual, predicted)})
        for horizon in range(HORIZON):
            rows_horizon.append(
                {
                    "model": model,
                    "horizon_min": horizon + 1,
                    **metric_values(shaped_actual[:, horizon, :], shaped_predicted[:, horizon, :]),
                }
            )
        for entry_index, entry in enumerate(entries):
            rows_entry.append(
                {
                    "model": model,
                    "entry_road_id": entry,
                    **metric_values(shaped_actual[:, :, entry_index], shaped_predicted[:, :, entry_index]),
                }
            )
        rows_per_day = len(origins)
        for day_index, date in enumerate(test_dates):
            start = day_index * rows_per_day
            end = start + rows_per_day
            rows_daily.append(
                {
                    "model": model,
                    "date": date,
                    **metric_values(shaped_actual[start:end], shaped_predicted[start:end]),
                }
            )
    return tuple(pd.DataFrame(rows) for rows in (rows_overall, rows_horizon, rows_entry, rows_daily))


def build(demand_dir: Path, split_path: Path, output_dir: Path):
    started = time.time()
    final_dir = output_dir.resolve()
    building = final_dir.with_name(final_dir.name + ".building")
    if final_dir.exists() or building.exists():
        raise FileExistsError(f"Output/build directory already exists: {final_dir}, {building}")
    building.mkdir(parents=True)
    values, dates, entries, split_names, split = load_data(demand_dir, split_path)
    eda_summary, autocorr, correlation, daily = build_eda(
        values, dates, entries, split_names, building
    )
    features, target, origins, feature_names = feature_matrix(values)
    features = add_weekday_features(features, dates, origins)
    feature_names.extend(["weekday_sin", "weekday_cos"])

    x_train, train_mask = subset_rows(features, split_names, "train", origins)
    y_train = target[train_mask]
    x_validation, validation_mask = subset_rows(features, split_names, "validation", origins)
    y_validation = target[validation_mask]
    tuning_rows = []
    for leaf in LEAF_GRID:
        model = ExtraTreesRegressor(
            n_estimators=TUNING_TREES,
            min_samples_leaf=leaf,
            max_features=0.7,
            n_jobs=-1,
            random_state=SEED,
        )
        model.fit(x_train, y_train)
        prediction = np.clip(model.predict(x_validation), 0, None).astype(np.float32)
        tuning_rows.append({"min_samples_leaf": leaf, **metric_values(y_validation, prediction)})
        print(f"tuning leaf={leaf} validation_mae={tuning_rows[-1]['mae']:.6f}", flush=True)
    tuning = pd.DataFrame(tuning_rows).sort_values(["mae", "min_samples_leaf"])
    selected_leaf = int(tuning.iloc[0]["min_samples_leaf"])
    save_csv(tuning, building / "validation_hyperparameter_selection.csv")

    fit_mask = np.repeat(np.isin(split_names, ["train", "validation"]), len(origins))
    x_fit = features[fit_mask]
    y_fit = target[fit_mask]
    test_values = values[split_names == "test"]
    test_dates = [date for date, name in zip(dates, split_names) if name == "test"]
    x_test, test_mask = subset_rows(features, split_names, "test", origins)
    y_test = target[test_mask]
    final_model = ExtraTreesRegressor(
        n_estimators=FINAL_TREES,
        min_samples_leaf=selected_leaf,
        max_features=0.7,
        n_jobs=-1,
        random_state=SEED,
    )
    final_model.fit(x_fit, y_fit)
    tree_prediction = np.clip(final_model.predict(x_test), 0, None).astype(np.float32)
    fit_values = values[np.isin(split_names, ["train", "validation"])]
    predictions = {
        "Persistence": persistence_prediction(test_values, origins),
        "HistoricalMean": historical_mean_prediction(fit_values, test_values, origins),
        "ExtraTrees": tree_prediction,
    }
    for name, prediction in predictions.items():
        if prediction.shape != y_test.shape or not np.isfinite(prediction).all() or (prediction < 0).any():
            raise AssertionError(f"Invalid prediction matrix: {name}")

    overall, by_horizon, by_entry, by_day = evaluate_models(
        y_test, predictions, entries, test_dates, origins
    )
    save_csv(overall, building / "test_metrics_overall.csv")
    save_csv(by_horizon, building / "test_metrics_by_horizon.csv")
    save_csv(by_entry, building / "test_metrics_by_entry.csv")
    save_csv(by_day, building / "test_metrics_by_day.csv")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for model, group in by_horizon.groupby("model", sort=False):
        ax.plot(group["horizon_min"], group["mae"], marker="o", markersize=3, label=model)
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("MAE (vehicles/minute/entry)")
    ax.set_title("Test MAE by forecast horizon")
    ax.set_xticks(range(1, HORIZON + 1))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(building / "fig_test_mae_by_horizon.png", dpi=180)
    plt.close(fig)

    np.savez_compressed(
        building / "test_predictions.npz",
        actual=y_test.astype(np.float32),
        persistence=predictions["Persistence"],
        historical_mean=predictions["HistoricalMean"],
        extra_trees=predictions["ExtraTrees"],
        origins=origins.astype(np.int16),
        entries=np.array(entries),
        dates=np.array(test_dates),
    )

    outputs = {}
    for path in sorted(building.iterdir()):
        if path.is_file():
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "experiment_id": "xuancheng_point_baseline_v1",
        "status": "COMPLETED_PENDING_REPRODUCIBILITY_RERUN",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {
            "demand_dir": str(demand_dir.resolve()),
            "demand_detail_sha256": sha256(demand_dir / "demand_by_entry_1min.csv"),
            "split_sha256": sha256(split_path),
            "split_id": split["split_id"],
        },
        "protocol": {
            "history_minutes": HISTORY,
            "horizon_minutes": HORIZON,
            "lags": LAGS,
            "rolling_windows": ROLLING_WINDOWS,
            "seed": SEED,
            "leaf_grid": LEAF_GRID,
            "selection_metric": "validation MAE",
            "selected_min_samples_leaf": selected_leaf,
            "fit_splits": ["train", "validation"],
            "sealed_split": "calibration",
            "evaluation_split": "test",
            "valid_origins_per_day": len(origins),
            "feature_count": len(feature_names),
            "output_count": len(entries) * HORIZON,
        },
        "sample_counts": {
            "train_origins": int(len(x_train)),
            "validation_origins": int(len(x_validation)),
            "fit_origins": int(len(x_fit)),
            "test_origins": int(len(x_test)),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "test_metrics": overall.to_dict(orient="records"),
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
        default=project / "04_实验" / "预测基线" / "point_baseline_v1" / "results",
    )
    args = parser.parse_args()
    result = build(args.demand_dir, args.split_path, args.output_dir)
    print(json.dumps({"status": result["status"], "test_metrics": result["test_metrics"], "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
