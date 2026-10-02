"""Leakage-safe movement-demand point forecasting baselines for Xuancheng."""

from __future__ import annotations

import argparse
import gc
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
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


SEED = 20260920
HISTORY = 60
HORIZON = 15
LAGS = (0, 1, 4, 14, 29, 59)
ROLLING_WINDOWS = (5, 15, 30, 60)
ENTRY_LEAF_GRID = (3, 8, 20)
DIRECT_LEAF_GRID = (8, 20, 50)
SHARE_WINDOW_GRID = (15, 60, 180)
SHARE_PRIOR_STRENGTH = 20.0
TUNING_TREES = 60
FINAL_TREES = 120

MODEL_KEYS = {
    "Persistence": "persistence",
    "HistoricalMean": "historical_mean",
    "HierarchicalHistoricalShare": "hierarchical_historical_share",
    "HierarchicalRollingShare": "hierarchical_rolling_share",
    "DirectExtraTrees": "direct_extra_trees",
    "DirectReconciled": "direct_reconciled",
}


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
        date: name for name, details in split["splits"].items() for date in details["dates"]
    }
    detail = pd.read_csv(
        demand_dir / "demand_by_movement_1min.csv",
        dtype={
            "date": "string",
            "split": "string",
            "movement_id": "string",
            "entry_road": "string",
            "next_road": "string",
        },
    )
    key = ["date", "minute", "movement_id"]
    if detail.duplicated(key).any():
        raise AssertionError("Duplicate date-minute-movement key")
    if set(detail["date"]) != set(date_to_split):
        raise AssertionError("Movement-demand dates differ from SPLIT.json")
    if not (detail["date"].map(date_to_split) == detail["split"]).all():
        raise AssertionError("Movement-demand split labels differ from SPLIT.json")
    movement_map = (
        detail[["movement_id", "entry_road", "next_road"]]
        .drop_duplicates()
        .sort_values("movement_id")
        .reset_index(drop=True)
    )
    if movement_map["movement_id"].duplicated().any() or len(movement_map) != 20:
        raise AssertionError("Expected 20 uniquely mapped movements")
    movements = movement_map["movement_id"].tolist()
    dates = sorted(detail["date"].unique().tolist())
    pivot = detail.pivot(index=["date", "minute"], columns="movement_id", values="count")
    expected_index = pd.MultiIndex.from_product([dates, range(1440)], names=["date", "minute"])
    pivot = pivot.reindex(expected_index).reindex(columns=movements)
    if pivot.isna().any().any():
        raise AssertionError("Incomplete date-minute-movement grid")
    values = pivot.to_numpy(dtype=np.float32).reshape(len(dates), 1440, len(movements))
    split_names = np.array([date_to_split[date] for date in dates])
    entries = sorted(movement_map["entry_road"].unique().tolist())
    entry_index = {entry: index for index, entry in enumerate(entries)}
    movement_to_entry = np.array(
        [entry_index[entry] for entry in movement_map["entry_road"]], dtype=np.int16
    )
    incidence = np.zeros((len(movements), len(entries)), dtype=np.float32)
    incidence[np.arange(len(movements)), movement_to_entry] = 1.0
    entry_values = values @ incidence
    return (
        values,
        entry_values,
        dates,
        movements,
        entries,
        movement_to_entry,
        movement_map,
        split_names,
        split,
    )


def feature_matrix(values: np.ndarray, prefix: str):
    days, minutes, series_count = values.shape
    origins = np.arange(HISTORY - 1, minutes - HORIZON)
    rows = days * len(origins)
    blocks = []
    names = []
    for lag in LAGS:
        blocks.append(values[:, origins - lag, :].reshape(rows, series_count))
        names.extend([f"{prefix}_{index}_lag_{lag}" for index in range(series_count)])
    cumulative = np.concatenate(
        [np.zeros((days, 1, series_count), dtype=np.float32), np.cumsum(values, axis=1)], axis=1
    )
    for window in ROLLING_WINDOWS:
        end = origins + 1
        start = end - window
        block = (cumulative[:, end, :] - cumulative[:, start, :]) / window
        blocks.append(block.reshape(rows, series_count))
        names.extend([f"{prefix}_{index}_mean_{window}" for index in range(series_count)])
    minute = np.tile(origins, days)
    minute_angle = 2 * np.pi * minute / 1440
    blocks.extend([np.sin(minute_angle)[:, None], np.cos(minute_angle)[:, None]])
    names.extend(["minute_sin", "minute_cos"])
    targets = np.stack([values[:, origins + h, :] for h in range(1, HORIZON + 1)], axis=2)
    target = targets.reshape(rows, HORIZON * series_count)
    return np.concatenate(blocks, axis=1).astype(np.float32), target.astype(np.float32), origins, names


def add_weekday_features(features, dates, origins):
    weekdays = np.array([pd.Timestamp(date).weekday() for date in dates])
    repeated = np.repeat(weekdays, len(origins))
    angle = 2 * np.pi * repeated / 7
    return np.concatenate([features, np.sin(angle)[:, None], np.cos(angle)[:, None]], axis=1).astype(np.float32)


def row_mask(split_names, allowed, origins):
    return np.repeat(np.isin(split_names, list(allowed)), len(origins))


def metric_values(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - actual
    absolute = np.abs(error)
    denominator = np.abs(actual) + np.abs(predicted)
    smape_terms = np.divide(
        2 * absolute, denominator, out=np.zeros_like(absolute), where=denominator > 0
    )
    actual_sum = np.abs(actual).sum()
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(absolute.sum() / actual_sum) if actual_sum else math.nan,
        "smape": float(smape_terms.mean()),
        "bias": float(error.mean()),
    }


def fit_extra_trees(x, y, leaf, trees):
    model = ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=leaf,
        max_features=0.7,
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(x, y)
    return model


def tune_extra_trees(x_train, y_train, x_validation, y_validation, grid, family):
    rows = []
    for leaf in grid:
        model = fit_extra_trees(x_train, y_train, leaf, TUNING_TREES)
        prediction = np.clip(model.predict(x_validation), 0, None).astype(np.float32)
        rows.append(
            {"model_family": family, "min_samples_leaf": leaf, **metric_values(y_validation, prediction)}
        )
        print(f"tuning {family} leaf={leaf} validation_mae={rows[-1]['mae']:.6f}", flush=True)
        del model, prediction
        gc.collect()
    frame = pd.DataFrame(rows).sort_values(["mae", "min_samples_leaf"]).reset_index(drop=True)
    return int(frame.iloc[0]["min_samples_leaf"]), frame


def persistence_prediction(selected_values, origins):
    current = selected_values[:, origins, :]
    return np.repeat(current[:, :, None, :], HORIZON, axis=2).reshape(
        selected_values.shape[0] * len(origins), -1
    ).astype(np.float32)


def historical_mean_prediction(fit_values, selected_values, origins):
    profile = fit_values.mean(axis=0)
    one_day = np.stack([profile[origins + h, :] for h in range(1, HORIZON + 1)], axis=1)
    return np.tile(one_day.reshape(len(origins), -1), (selected_values.shape[0], 1)).astype(np.float32)


def share_parameters(fit_values, movement_to_entry, entry_count):
    total_by_movement = fit_values.sum(axis=(0, 1)).astype(np.float64)
    global_share = np.zeros_like(total_by_movement)
    minute_share = np.zeros((1440, len(total_by_movement)), dtype=np.float64)
    minute_counts = fit_values.sum(axis=0).astype(np.float64)
    for entry_index in range(entry_count):
        indexes = np.flatnonzero(movement_to_entry == entry_index)
        total = total_by_movement[indexes].sum()
        global_share[indexes] = total_by_movement[indexes] / total
        numerator = minute_counts[:, indexes] + SHARE_PRIOR_STRENGTH * global_share[indexes]
        minute_share[:, indexes] = numerator / numerator.sum(axis=1, keepdims=True)
    return global_share.astype(np.float32), minute_share.astype(np.float32), total_by_movement


def hierarchical_historical(entry_prediction, minute_share, origins, days, movement_to_entry):
    entry_shaped = entry_prediction.reshape(days, len(origins), HORIZON, -1)
    result = np.zeros((days, len(origins), HORIZON, len(movement_to_entry)), dtype=np.float32)
    for h in range(HORIZON):
        shares = minute_share[origins + h + 1]
        result[:, :, h, :] = entry_shaped[:, :, h, movement_to_entry] * shares[None, :, :]
    return result.reshape(days * len(origins), -1)


def rolling_shares(selected_values, origins, window, global_share, movement_to_entry, entry_count):
    days, _, movement_count = selected_values.shape
    cumulative = np.concatenate(
        [np.zeros((days, 1, movement_count), dtype=np.float32), np.cumsum(selected_values, axis=1)], axis=1
    )
    end = origins + 1
    start = np.maximum(0, end - window)
    counts = cumulative[:, end, :] - cumulative[:, start, :]
    shares = np.zeros_like(counts, dtype=np.float32)
    for entry_index in range(entry_count):
        indexes = np.flatnonzero(movement_to_entry == entry_index)
        numerator = counts[:, :, indexes] + SHARE_PRIOR_STRENGTH * global_share[indexes]
        shares[:, :, indexes] = numerator / numerator.sum(axis=2, keepdims=True)
    return shares


def hierarchical_rolling(
    entry_prediction, selected_values, origins, window, global_share, movement_to_entry, entry_count
):
    days = selected_values.shape[0]
    shares = rolling_shares(
        selected_values, origins, window, global_share, movement_to_entry, entry_count
    )
    entry_shaped = entry_prediction.reshape(days, len(origins), HORIZON, entry_count)
    result = np.zeros((days, len(origins), HORIZON, len(movement_to_entry)), dtype=np.float32)
    for h in range(HORIZON):
        result[:, :, h, :] = entry_shaped[:, :, h, movement_to_entry] * shares
    return result.reshape(days * len(origins), -1)


def reconcile_direct(direct, entry_prediction, fallback_share, movement_to_entry, entry_count):
    rows = direct.shape[0]
    direct_shaped = direct.reshape(rows, HORIZON, -1).astype(np.float64)
    entry_shaped = entry_prediction.reshape(rows, HORIZON, entry_count).astype(np.float64)
    result = np.zeros_like(direct_shaped)
    for entry_index in range(entry_count):
        indexes = np.flatnonzero(movement_to_entry == entry_index)
        subtotal = direct_shaped[:, :, indexes].sum(axis=2)
        scale = np.divide(
            entry_shaped[:, :, entry_index], subtotal, out=np.zeros_like(subtotal), where=subtotal > 0
        )
        result[:, :, indexes] = direct_shaped[:, :, indexes] * scale[:, :, None]
        zero_mask = subtotal <= 0
        if np.any(zero_mask):
            for movement_index in indexes:
                result[:, :, movement_index][zero_mask] = (
                    entry_shaped[:, :, entry_index][zero_mask] * fallback_share[movement_index]
                )
    return result.reshape(rows, -1).astype(np.float32)


def aggregate_by_entry(matrix, movement_to_entry, entry_count):
    shaped = matrix.reshape(-1, HORIZON, len(movement_to_entry))
    result = np.zeros((len(shaped), HORIZON, entry_count), dtype=np.float64)
    for entry_index in range(entry_count):
        result[:, :, entry_index] = shaped[:, :, movement_to_entry == entry_index].sum(axis=2)
    return result


def movement_tiers(train_totals):
    order = np.argsort(train_totals, kind="stable")
    tiers = np.empty(len(train_totals), dtype=object)
    chunks = np.array_split(order, 3)
    for label, indexes in zip(("low", "medium", "high"), chunks):
        tiers[indexes] = label
    return tiers


def evaluate_models(
    actual,
    predictions,
    movements,
    entries,
    movement_map,
    movement_to_entry,
    test_dates,
    origins,
    train_totals,
    high_thresholds,
    shared_entry_prediction,
):
    shaped_actual = actual.reshape(-1, HORIZON, len(movements))
    tiers = movement_tiers(train_totals)
    overall_rows, horizon_rows, movement_rows, day_rows = [], [], [], []
    state_rows, tier_rows, occurrence_rows, entry_rows = [], [], [], []
    shared_entry = shared_entry_prediction.reshape(-1, HORIZON, len(entries))
    movement_lookup = movement_map.set_index("movement_id")
    rows_per_day = len(origins)
    for model, predicted in predictions.items():
        shaped_predicted = predicted.reshape(-1, HORIZON, len(movements))
        overall_rows.append({"model": model, **metric_values(shaped_actual, shaped_predicted)})
        for h in range(HORIZON):
            horizon_rows.append(
                {"model": model, "horizon_min": h + 1, **metric_values(shaped_actual[:, h], shaped_predicted[:, h])}
            )
        for movement_index, movement in enumerate(movements):
            movement_rows.append(
                {
                    "model": model,
                    "movement_id": movement,
                    "entry_road": movement_lookup.loc[movement, "entry_road"],
                    "next_road": movement_lookup.loc[movement, "next_road"],
                    "demand_tier": tiers[movement_index],
                    "train_total": int(train_totals[movement_index]),
                    **metric_values(shaped_actual[:, :, movement_index], shaped_predicted[:, :, movement_index]),
                }
            )
        for day_index, date in enumerate(test_dates):
            day_slice = slice(day_index * rows_per_day, (day_index + 1) * rows_per_day)
            day_rows.append(
                {"model": model, "date": date, **metric_values(shaped_actual[day_slice], shaped_predicted[day_slice])}
            )
        nonzero_mask = shaped_actual > 0
        threshold_cube = high_thresholds[None, None, :]
        high_mask = shaped_actual >= threshold_cube
        for state, mask in (("nonzero", nonzero_mask), ("high_demand", high_mask)):
            state_rows.append(
                {
                    "model": model,
                    "state": state,
                    "observations": int(mask.sum()),
                    "observation_share": float(mask.mean()),
                    **metric_values(shaped_actual[mask], shaped_predicted[mask]),
                }
            )
        for tier in ("low", "medium", "high"):
            indexes = np.flatnonzero(tiers == tier)
            tier_rows.append(
                {
                    "model": model,
                    "demand_tier": tier,
                    "movement_count": len(indexes),
                    **metric_values(shaped_actual[:, :, indexes], shaped_predicted[:, :, indexes]),
                }
            )
        labels = (shaped_actual.reshape(-1) > 0).astype(np.int8)
        scores = shaped_predicted.reshape(-1)
        binary = scores >= 0.5
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, binary, average="binary", zero_division=0
        )
        occurrence_rows.append(
            {
                "model": model,
                "positive_rate": float(labels.mean()),
                "roc_auc": float(roc_auc_score(labels, scores)),
                "average_precision": float(average_precision_score(labels, scores)),
                "threshold": 0.5,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
        actual_entry = aggregate_by_entry(actual, movement_to_entry, len(entries))
        predicted_entry = aggregate_by_entry(predicted, movement_to_entry, len(entries))
        for entry_index, entry in enumerate(entries):
            gap = predicted_entry[:, :, entry_index] - shared_entry[:, :, entry_index]
            entry_rows.append(
                {
                    "model": model,
                    "entry_road": entry,
                    **metric_values(actual_entry[:, :, entry_index], predicted_entry[:, :, entry_index]),
                    "shared_forecast_gap_mae": float(np.abs(gap).mean()),
                    "shared_forecast_gap_max_abs": float(np.abs(gap).max()),
                }
            )
    return {
        "overall": pd.DataFrame(overall_rows),
        "by_horizon": pd.DataFrame(horizon_rows),
        "by_movement": pd.DataFrame(movement_rows),
        "by_day": pd.DataFrame(day_rows),
        "by_state": pd.DataFrame(state_rows),
        "by_tier": pd.DataFrame(tier_rows),
        "occurrence": pd.DataFrame(occurrence_rows),
        "by_entry": pd.DataFrame(entry_rows),
    }, tiers


def build(demand_dir: Path, split_path: Path, output_dir: Path):
    started = time.time()
    final_dir = output_dir.resolve()
    building = final_dir.with_name(final_dir.name + ".building")
    if final_dir.exists() or building.exists():
        raise FileExistsError(f"Output/build directory already exists: {final_dir}, {building}")
    building.mkdir(parents=True)
    (
        values,
        entry_values,
        dates,
        movements,
        entries,
        movement_to_entry,
        movement_map,
        split_names,
        split,
    ) = load_data(demand_dir, split_path)

    movement_features, movement_target, origins, movement_feature_names = feature_matrix(values, "movement")
    entry_features, entry_target, entry_origins, entry_feature_names = feature_matrix(entry_values, "entry")
    if not np.array_equal(origins, entry_origins):
        raise AssertionError("Movement and entry origins differ")
    movement_features = add_weekday_features(movement_features, dates, origins)
    entry_features = add_weekday_features(entry_features, dates, origins)
    movement_feature_names.extend(["weekday_sin", "weekday_cos"])
    entry_feature_names.extend(["weekday_sin", "weekday_cos"])

    train_mask = row_mask(split_names, ["train"], origins)
    validation_mask = row_mask(split_names, ["validation"], origins)
    fit_mask = row_mask(split_names, ["train", "validation"], origins)
    test_mask = row_mask(split_names, ["test"], origins)

    entry_leaf, entry_tuning = tune_extra_trees(
        entry_features[train_mask], entry_target[train_mask],
        entry_features[validation_mask], entry_target[validation_mask],
        ENTRY_LEAF_GRID, "entry_total",
    )
    direct_leaf, direct_tuning = tune_extra_trees(
        movement_features[train_mask], movement_target[train_mask],
        movement_features[validation_mask], movement_target[validation_mask],
        DIRECT_LEAF_GRID, "direct_movement",
    )
    save_csv(pd.concat([entry_tuning, direct_tuning], ignore_index=True), building / "validation_tree_selection.csv")

    train_values = values[split_names == "train"]
    validation_values = values[split_names == "validation"]
    global_share_train, minute_share_train, train_totals = share_parameters(
        train_values, movement_to_entry, len(entries)
    )
    entry_validation_model = fit_extra_trees(
        entry_features[train_mask], entry_target[train_mask], entry_leaf, TUNING_TREES
    )
    entry_validation_prediction = np.clip(
        entry_validation_model.predict(entry_features[validation_mask]), 0, None
    ).astype(np.float32)
    del entry_validation_model
    gc.collect()
    share_rows = []
    for window in SHARE_WINDOW_GRID:
        prediction = hierarchical_rolling(
            entry_validation_prediction,
            validation_values,
            origins,
            window,
            global_share_train,
            movement_to_entry,
            len(entries),
        )
        share_rows.append({"window_minutes": window, **metric_values(movement_target[validation_mask], prediction)})
        print(f"tuning rolling_share window={window} validation_mae={share_rows[-1]['mae']:.6f}", flush=True)
    share_tuning = pd.DataFrame(share_rows).sort_values(["mae", "window_minutes"]).reset_index(drop=True)
    selected_window = int(share_tuning.iloc[0]["window_minutes"])
    save_csv(share_tuning, building / "validation_share_window_selection.csv")

    entry_model = fit_extra_trees(
        entry_features[fit_mask], entry_target[fit_mask], entry_leaf, FINAL_TREES
    )
    shared_entry_prediction = np.clip(entry_model.predict(entry_features[test_mask]), 0, None).astype(np.float32)
    del entry_model
    gc.collect()
    direct_model = fit_extra_trees(
        movement_features[fit_mask], movement_target[fit_mask], direct_leaf, FINAL_TREES
    )
    direct_prediction = np.clip(direct_model.predict(movement_features[test_mask]), 0, None).astype(np.float32)
    del direct_model
    gc.collect()

    fit_values = values[np.isin(split_names, ["train", "validation"])]
    test_values = values[split_names == "test"]
    test_dates = [date for date, name in zip(dates, split_names) if name == "test"]
    global_share_fit, minute_share_fit, _ = share_parameters(
        fit_values, movement_to_entry, len(entries)
    )
    historical_share_prediction = hierarchical_historical(
        shared_entry_prediction, minute_share_fit, origins, len(test_dates), movement_to_entry
    )
    rolling_share_prediction = hierarchical_rolling(
        shared_entry_prediction,
        test_values,
        origins,
        selected_window,
        global_share_fit,
        movement_to_entry,
        len(entries),
    )
    reconciled_prediction = reconcile_direct(
        direct_prediction,
        shared_entry_prediction,
        global_share_fit,
        movement_to_entry,
        len(entries),
    )
    predictions = {
        "Persistence": persistence_prediction(test_values, origins),
        "HistoricalMean": historical_mean_prediction(fit_values, test_values, origins),
        "HierarchicalHistoricalShare": historical_share_prediction,
        "HierarchicalRollingShare": rolling_share_prediction,
        "DirectExtraTrees": direct_prediction,
        "DirectReconciled": reconciled_prediction,
    }
    y_test = movement_target[test_mask]
    for name, prediction in predictions.items():
        if prediction.shape != y_test.shape or not np.isfinite(prediction).all() or (prediction < 0).any():
            raise AssertionError(f"Invalid prediction matrix: {name}")

    positive_train = np.where(train_values > 0, train_values, np.nan)
    high_thresholds = np.nanquantile(positive_train, 0.90, axis=(0, 1))
    high_thresholds = np.maximum(1.0, np.nan_to_num(high_thresholds, nan=1.0)).astype(np.float32)
    tables, tiers = evaluate_models(
        y_test,
        predictions,
        movements,
        entries,
        movement_map,
        movement_to_entry,
        test_dates,
        origins,
        train_totals,
        high_thresholds,
        shared_entry_prediction,
    )
    output_names = {
        "overall": "test_metrics_overall.csv",
        "by_horizon": "test_metrics_by_horizon.csv",
        "by_movement": "test_metrics_by_movement.csv",
        "by_day": "test_metrics_by_day.csv",
        "by_state": "test_metrics_by_state.csv",
        "by_tier": "test_metrics_by_demand_tier.csv",
        "occurrence": "test_occurrence_metrics.csv",
        "by_entry": "test_entry_aggregation_metrics.csv",
    }
    for key, filename in output_names.items():
        save_csv(tables[key], building / filename)

    metadata = movement_map.copy()
    metadata["train_total"] = train_totals.astype(np.int64)
    metadata["demand_tier"] = tiers
    metadata["high_demand_threshold"] = high_thresholds
    metadata["global_share_train_validation"] = global_share_fit
    save_csv(metadata, building / "movement_metadata.csv")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for model, group in tables["by_horizon"].groupby("model", sort=False):
        ax.plot(group["horizon_min"], group["mae"], marker="o", markersize=2.5, label=model)
    ax.set_xlabel("Forecast horizon (minutes)")
    ax.set_ylabel("MAE (vehicles/minute/movement)")
    ax.set_title("Movement-demand test MAE by forecast horizon")
    ax.set_xticks(range(1, HORIZON + 1))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(building / "fig_test_mae_by_horizon.png", dpi=180)
    plt.close(fig)

    archive = {
        "actual": y_test.astype(np.float32),
        "shared_entry_prediction": shared_entry_prediction,
        "origins": origins.astype(np.int16),
        "movements": np.array(movements),
        "entries": np.array(entries),
        "movement_to_entry": movement_to_entry,
        "dates": np.array(test_dates),
        "high_thresholds": high_thresholds,
        "train_totals": train_totals.astype(np.int64),
    }
    archive.update({MODEL_KEYS[name]: value for name, value in predictions.items()})
    np.savez_compressed(building / "test_predictions.npz", **archive)

    core_files = sorted(path.name for path in building.iterdir() if path.is_file())
    outputs = {
        name: {"bytes": (building / name).stat().st_size, "sha256": sha256(building / name)}
        for name in core_files
    }
    manifest = {
        "experiment_id": "xuancheng_movement_point_baseline_v1",
        "status": "COMPLETED_PENDING_REPRODUCIBILITY_RERUN",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {
            "movement_demand_sha256": sha256(demand_dir / "demand_by_movement_1min.csv"),
            "movement_manifest_sha256": sha256(demand_dir / "MOVEMENT_DEMAND.json"),
            "split_sha256": sha256(split_path),
            "split_id": split["split_id"],
        },
        "protocol": {
            "history_minutes": HISTORY,
            "horizon_minutes": HORIZON,
            "lags": LAGS,
            "rolling_windows": ROLLING_WINDOWS,
            "seed": SEED,
            "entry_leaf_grid": ENTRY_LEAF_GRID,
            "direct_leaf_grid": DIRECT_LEAF_GRID,
            "share_window_grid": SHARE_WINDOW_GRID,
            "share_prior_strength": SHARE_PRIOR_STRENGTH,
            "selected_entry_min_samples_leaf": entry_leaf,
            "selected_direct_min_samples_leaf": direct_leaf,
            "selected_share_window_minutes": selected_window,
            "selection_metric": "validation MAE",
            "fit_splits": ["train", "validation"],
            "sealed_split": "calibration",
            "evaluation_split": "test",
            "valid_origins_per_day": len(origins),
            "movement_feature_count": len(movement_feature_names),
            "entry_feature_count": len(entry_feature_names),
            "movement_output_count": len(movements) * HORIZON,
        },
        "sample_counts": {
            "train_origins": int(train_mask.sum()),
            "validation_origins": int(validation_mask.sum()),
            "fit_origins": int(fit_mask.sum()),
            "test_origins": int(test_mask.sum()),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "test_metrics": tables["overall"].to_dict(orient="records"),
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
        "--demand-dir", type=Path,
        default=project / "02_数据" / "派生数据" / "宣城_movement_demand_v1",
    )
    parser.add_argument(
        "--split-path", type=Path,
        default=project / "02_数据" / "派生数据" / "宣城_clean_v2" / "SPLIT.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=project / "04_实验" / "预测基线" / "movement_point_baseline_v1" / "results_v1",
    )
    args = parser.parse_args()
    result = build(args.demand_dir, args.split_path, args.output_dir)
    print(json.dumps(
        {"status": result["status"], "test_metrics": result["test_metrics"], "elapsed_seconds": result["elapsed_seconds"]},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
