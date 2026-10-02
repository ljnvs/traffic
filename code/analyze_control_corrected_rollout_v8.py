"""Paired-day uncertainty analysis for the frozen v8 test evaluation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import run_control_admission_queue_v1 as v1


COMPARISONS = (
    ("Forecast_vs_ZeroFuture", "RolloutCausalForecast", "RolloutZeroFuture"),
    ("EventOracle_vs_ZeroFuture", "RolloutEventOracle", "RolloutZeroFuture"),
    ("ZeroFuture_vs_MaxPressure", "RolloutZeroFuture", "MaxPressure"),
)
METRICS = (
    ("queue_vehicle_seconds", True),
    ("throughput_exit_vehicles", False),
    ("spillback_road_time_fraction", True),
    ("mean_completed_travel_time", True),
)
BOOTSTRAP_DRAWS = 50_000
SEED = 20260921


def improvement(candidate, baseline, lower_better):
    raw = (candidate - baseline) / baseline
    return -raw if lower_better else raw


def exact_sign_p(values):
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    n = positive + negative
    if n == 0:
        return 1.0, positive, negative
    tail = min(positive, negative)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n)
    return min(1.0, 2 * probability), positive, negative


def analyze(result):
    rows = result["test_results"]
    dates = sorted({row["date"] for row in rows})
    lookup = {(row["date"], row["mode"]): row for row in rows}
    rng = np.random.default_rng(SEED)
    daily = []
    summary = []
    for label, candidate_mode, baseline_mode in COMPARISONS:
        for metric, lower_better in METRICS:
            candidate = np.array([lookup[(date, candidate_mode)][metric] for date in dates], dtype=float)
            baseline = np.array([lookup[(date, baseline_mode)][metric] for date in dates], dtype=float)
            effects = np.array([
                improvement(a, b, lower_better) for a, b in zip(candidate, baseline)
            ])
            for date, a, b, effect in zip(dates, candidate, baseline, effects):
                daily.append({
                    "comparison": label,
                    "metric": metric,
                    "date": date,
                    "candidate": a,
                    "baseline": b,
                    "improvement": effect,
                })
            estimate = improvement(candidate.mean(), baseline.mean(), lower_better)
            indices = rng.integers(0, len(dates), size=(BOOTSTRAP_DRAWS, len(dates)))
            boot_candidate = candidate[indices].mean(axis=1)
            boot_baseline = baseline[indices].mean(axis=1)
            raw = (boot_candidate - boot_baseline) / boot_baseline
            boot = -raw if lower_better else raw
            sign_p, positive, negative = exact_sign_p(effects)
            loo = []
            for omitted in range(len(dates)):
                keep = np.arange(len(dates)) != omitted
                loo.append(improvement(candidate[keep].mean(), baseline[keep].mean(), lower_better))
            summary.append({
                "comparison": label,
                "metric": metric,
                "days": len(dates),
                "mean_ratio_improvement": estimate,
                "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                "improved_days": int(positive),
                "worsened_days": int(negative),
                "ties": int(len(dates) - positive - negative),
                "exact_sign_test_p_two_sided": sign_p,
                "leave_one_out_min": min(loo),
                "leave_one_out_max": max(loo),
            })
    return daily, summary


def report(summary):
    rows = {(row["comparison"], row["metric"]): row for row in summary}
    def line(comparison, metric, label):
        row = rows[(comparison, metric)]
        return (
            f"- {label}：{row['mean_ratio_improvement']:.3%} "
            f"[95% bootstrap {row['bootstrap_ci_low']:.3%}, {row['bootstrap_ci_high']:.3%}]，"
            f"改善{row['improved_days']}/7天，双侧精确符号检验p={row['exact_sign_test_p_two_sided']:.4f}。"
        )
    return """# v8 逐日效应与不确定性

主估计量为候选策略与基线策略的7日算术均值之比，并统一转换为正值表示改善。区间使用固定种子50,000次日期配对bootstrap。样本仅7日，区间和p值只作探索性描述。

## 核心结果

""" + "\n".join([
        line("Forecast_vs_ZeroFuture", "queue_vehicle_seconds", "因果预测 vs 零未来，排队"),
        line("EventOracle_vs_ZeroFuture", "queue_vehicle_seconds", "事件Oracle vs 零未来，排队"),
        line("EventOracle_vs_ZeroFuture", "spillback_road_time_fraction", "事件Oracle vs 零未来，回溢"),
        line("ZeroFuture_vs_MaxPressure", "queue_vehicle_seconds", "零未来 vs MaxPressure，排队"),
    ]) + """

## 解释

修正时间接口和动作搜索后，预测与Oracle仍未稳定降低排队。Oracle对回溢存在平均改善，但逐日样本很少，不能据此声称一般性统计显著。预测策略的排队退化幅度大于Oracle，说明预测误差仍会放大近似控制器的动作偏差；Oracle自身为负则说明控制结构和目标失配仍是主要瓶颈。
"""


def main():
    project = Path(__file__).resolve().parents[1]
    output = project / "04_实验" / "控制基线" / "corrected_rollout_v8"
    result = json.loads((output / "RESULTS.json").read_text(encoding="utf-8"))
    daily, summary = analyze(result)
    v1.write_csv(output / "逐日配对效应.csv", daily)
    v1.write_csv(output / "统计不确定性汇总.csv", summary)
    payload = {
        "status": "COMPLETED_EXPLORATORY_PAIRED_DAY_INFERENCE",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": SEED,
        "paired_unit": "date",
        "days": 7,
        "summary": summary,
        "warning": "Exploratory only: seven test days and no multiplicity correction.",
    }
    (output / "STATISTICS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "统计报告.md").write_text(report(summary), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "comparisons": len(summary)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
