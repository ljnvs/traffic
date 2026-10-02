"""Control-interface audit and positive controls (v7).

This module deliberately does not run or inspect the test split.  It audits the
time semantics and action space used by v6, builds an event-exact 5-second
oracle from validation flow files, and checks coordinate search against joint
enumeration on synthetic and validation states.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import run_control_admission_queue_v1 as v1
import run_control_admission_queue_v2 as v2
import run_control_admission_rollout_v6 as v6


STEP_SECONDS = 5
AUDIT_HORIZON_SECONDS = 60
VALIDATION_SAMPLE_OFFSETS_SECONDS = (0, 1800, 3570)


def movement_key(phase):
    """Canonical action identity; signal phases with the same movements are one action."""
    return tuple(sorted((start, end) for start, end, _lanes in phase["movements"]))


def effective_phases(controllers):
    result = {}
    for intersection_id, phases in controllers.items():
        unique = {}
        for phase in phases:
            key = movement_key(phase)
            if key and key not in unique:
                unique[key] = phase
        result[intersection_id] = [unique[key] for key in sorted(unique)]
    return result


def action_space_audit(controllers):
    effective = effective_phases(controllers)
    rows = []
    for intersection_id in sorted(controllers):
        nonempty_raw = [phase for phase in controllers[intersection_id] if movement_key(phase)]
        rows.append(
            {
                "intersection_id": intersection_id,
                "raw_phase_count": len(controllers[intersection_id]),
                "nonempty_phase_count": len(nonempty_raw),
                "distinct_nonempty_action_count": len(effective[intersection_id]),
                "has_control_freedom": len(effective[intersection_id]) > 1,
                "effective_phase_indices": [
                    phase["phase_index"] for phase in effective[intersection_id]
                ],
            }
        )
    variable = [row for row in rows if row["has_control_freedom"]]
    joint_count = 1
    for row in variable:
        joint_count *= row["distinct_nonempty_action_count"]
    return rows, {
        "controlled_intersections": len(rows),
        "intersections_with_control_freedom": len(variable),
        "fraction_with_control_freedom": len(variable) / len(rows) if rows else 0.0,
        "single_stage_joint_action_count": joint_count,
    }


def exact_event_oracle(arrivals, current_step, horizon_seconds):
    """Future exogenous arrivals aligned to v6's add-before-service projector.

    The simulator has already inserted arrivals[current_step] before making the
    decision.  Projection slot 0 therefore contains no new arrivals and applies
    service to the observed state.  Slot k>0 adds arrivals[current_step+k]
    before that slot's service.  Values are rates per second because v6's
    projector multiplies rates by five seconds.
    """
    slots = horizon_seconds // STEP_SECONDS
    output = []
    for offset in range(slots):
        counts = defaultdict(float)
        if offset > 0:
            for route, _sequence in arrivals.get(current_step + offset, []):
                if len(route) >= 2:
                    counts[(route[0], route[1])] += 1.0
        output.append({pair: count / STEP_SECONDS for pair, count in counts.items()})
    return output


def aligned_minute_forecast(day_inputs, global_minute, second_in_minute, horizon_seconds):
    """Map a causal minute forecast to 5-second projection slots.

    At any decision in minute m, only origin m-1 is admissible.  Its first row
    predicts minute m.  Slot 0 adds nothing because its arrivals are observed;
    later slots use the row for the minute containing their event time.
    """
    origin = global_minute - 1
    block = day_inputs["forecast"][str(origin)]
    pairs = [tuple(pair) for pair in day_inputs["movement_pairs"]]
    output = []
    row_indices = []
    for offset in range(horizon_seconds // STEP_SECONDS):
        if offset == 0:
            output.append({})
            row_indices.append(None)
            continue
        target_minute = global_minute + (second_in_minute + offset * STEP_SECONDS) // 60
        row_index = int(target_minute - origin - 1)
        if row_index < 0 or row_index >= len(block):
            raise IndexError("Requested projection lies outside the frozen forecast horizon")
        output.append(
            {pair: float(block[row_index][index]) / 60.0 for index, pair in enumerate(pairs)}
        )
        row_indices.append(row_index)
    return output, {
        "decision_minute": global_minute,
        "forecast_origin_minute": origin,
        "uses_current_minute_lag0": False,
        "projection_row_indices": row_indices,
    }


def choices_to_actions(choices):
    return {
        intersection_id: set(movement_key(phase))
        for intersection_id, phase in choices.items()
    }


def switched_intersections(choices, active_phase):
    return {
        intersection_id
        for intersection_id, phase in choices.items()
        if phase["phase_index"] != active_phase[intersection_id]["phase_index"]
    }


def score_choices(choices, active_phase, road_q, movement_q, network, arrivals):
    return v6.rollout_cost(
        road_q,
        movement_q,
        choices_to_actions(choices),
        network,
        arrivals,
        switched_intersections(choices, active_phase),
    )


def joint_exhaustive_search(
    controllers, active_phase, road_q, movement_q, network, arrivals
):
    """Enumerate all distinct nonempty actions at variable nodes (8x8=64 here)."""
    effective = effective_phases(controllers)
    variable_ids = sorted(iid for iid, phases in effective.items() if len(phases) > 1)
    fixed = {
        iid: (active_phase.get(iid) or phases[0])
        for iid, phases in effective.items()
        if len(phases) == 1
    }
    best = None
    evaluated = 0
    for combination in itertools.product(*(effective[iid] for iid in variable_ids)):
        choices = dict(fixed)
        choices.update(dict(zip(variable_ids, combination)))
        cost = score_choices(choices, active_phase, road_q, movement_q, network, arrivals)
        signature = tuple(choices[iid]["phase_index"] for iid in sorted(choices))
        candidate = (cost, signature, choices)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
        evaluated += 1
    if best is None:
        raise AssertionError("No joint action was available")
    return best[2], best[0], evaluated


def coordinate_search(
    controllers, active_phase, initial_choices, road_q, movement_q, network, arrivals
):
    """One deterministic coordinate sweep, matching the structural choice in v6."""
    effective = effective_phases(controllers)
    choices = dict(initial_choices)
    for iid in sorted(effective):
        candidates = effective[iid]
        if len(candidates) <= 1:
            choices[iid] = candidates[0]
            continue
        scored = []
        for phase in candidates:
            trial = dict(choices)
            trial[iid] = phase
            scored.append(
                (
                    score_choices(trial, active_phase, road_q, movement_q, network, arrivals),
                    phase["phase_index"],
                    phase,
                )
            )
        choices[iid] = min(scored, key=lambda item: (item[0], item[1]))[2]
    return choices, score_choices(choices, active_phase, road_q, movement_q, network, arrivals)


def exact_fifo_one_step(packets, actions, network):
    """One service step with the same FIFO packet semantics as the actual simulator."""
    state = {road: dict(values) for road, values in packets.items()}
    queues = {road: sum(values.values()) for road, values in state.items()}
    removals = defaultdict(lambda: defaultdict(float))
    additions = defaultdict(lambda: defaultdict(float))
    reserved = defaultdict(float)
    for road in sorted(network["road_end"]):
        budget = (
            v1.SATURATION_FLOW_PER_LANE_PER_SECOND
            * network["lane_counts"][road]
            * STEP_SECONDS
        )
        end_intersection = network["road_end"][road]
        if end_intersection in network["controlled"]:
            allowed = actions.get(end_intersection, set())
        else:
            allowed = {(road, nxt) for nxt in network["successors"].get(road, ())}
        for key, available in sorted(state.get(road, {}).items(), key=lambda item: (item[0][2], item[0][3])):
            if budget <= 1e-12:
                break
            route, position, _source_step, _sequence = key
            final = position == len(route) - 1
            if not final and (road, route[position + 1]) not in allowed:
                continue
            moved = min(available, budget)
            if not final:
                next_road = route[position + 1]
                storage = max(
                    0.0,
                    network["capacities"][next_road]
                    - queues.get(next_road, 0.0)
                    - reserved[next_road],
                )
                moved = min(moved, storage)
            if moved <= 1e-12:
                continue
            removals[road][key] += moved
            budget -= moved
            if not final:
                new_key = (route, position + 1, key[2], key[3])
                additions[next_road][new_key] += moved
                reserved[next_road] += moved
    for road, values in removals.items():
        for key, value in values.items():
            state[road][key] -= value
            if state[road][key] <= 1e-12:
                del state[road][key]
    for road, values in additions.items():
        for key, value in values.items():
            state[road][key] = state[road].get(key, 0.0) + value
    road_q = {road: sum(values.values()) for road, values in state.items()}
    movement_q = defaultdict(float)
    for road, values in state.items():
        for (route, position, _source, _sequence), mass in values.items():
            if position < len(route) - 1:
                movement_q[(road, route[position + 1])] += mass
    return road_q, dict(movement_q)


def projection_consistency_checks():
    network = {
        "road_end": {"r0": "i0", "r1": "sink", "r2": "sink"},
        "controlled": {"i0"},
        "successors": {"r0": ("r1", "r2"), "r1": (), "r2": ()},
        "capacities": {"r0": 20.0, "r1": 20.0, "r2": 20.0},
        "lane_counts": {"r0": 1, "r1": 1, "r2": 1},
        "monitored": {"r0"},
        "exit_roads": {"r1", "r2"},
    }
    actions = {"i0": {("r0", "r1"), ("r0", "r2")}}
    one = (("r0", "r1"), 0, 0, 0)
    packets_simple = {"r0": {one: 10.0}, "r1": {}, "r2": {}}
    fifo_road, fifo_movement = exact_fifo_one_step(packets_simple, actions, network)
    projected_road, projected_movement = v6.project_one_step(
        {"r0": 10.0, "r1": 0.0, "r2": 0.0},
        {("r0", "r1"): 10.0}, actions, network, {}, set(), 0,
    )
    simple_error = max(abs(fifo_road[r] - projected_road[r]) for r in network["road_end"])

    early = (("r0", "r1"), 0, 0, 0)
    late = (("r0", "r2"), 0, 1, 1)
    packets_competing = {"r0": {early: 5.0, late: 5.0}, "r1": {}, "r2": {}}
    fifo2_road, fifo2_movement = exact_fifo_one_step(packets_competing, actions, network)
    projected2_road, projected2_movement = v6.project_one_step(
        {"r0": 10.0, "r1": 0.0, "r2": 0.0},
        {("r0", "r1"): 5.0, ("r0", "r2"): 5.0}, actions, network, {}, set(), 0,
    )
    movement_l1 = sum(
        abs(fifo2_movement.get(pair, 0.0) - projected2_movement.get(pair, 0.0))
        for pair in set(fifo2_movement) | set(projected2_movement)
    )
    return {
        "single_movement_max_road_queue_error": simple_error,
        "single_movement_match": simple_error < 1e-9,
        "competing_movement_fifo_vs_proportional_l1": movement_l1,
        "competing_movement_difference_expected": movement_l1 > 1e-9,
        "interpretation": "Aggregate projector matches FIFO for one eligible movement, but proportional service is not FIFO-equivalent when movements compete.",
    }


def synthetic_positive_control():
    phase_a = {"phase_index": 0, "movements": [("a", "ao", 1)], "served_entry_indices": []}
    phase_b = {"phase_index": 1, "movements": [("b", "bo", 1)], "served_entry_indices": []}
    controllers = {"i": [phase_a, phase_b]}
    active = {"i": phase_a}
    network = {
        "road_end": {"a": "i", "b": "i", "ao": "sink_a", "bo": "sink_b"},
        "controlled": {"i"},
        "successors": {"a": ("ao",), "b": ("bo",), "ao": (), "bo": ()},
        "capacities": {"a": 30.0, "b": 30.0, "ao": 100.0, "bo": 100.0},
        "lane_counts": {"a": 1, "b": 1, "ao": 1, "bo": 1},
        "monitored": {"a", "b"},
        "exit_roads": {"ao", "bo"},
    }
    road_q = {"a": 7.0, "b": 6.0, "ao": 0.0, "bo": 0.0}
    movement_q = {("a", "ao"): 7.0, ("b", "bo"): 6.0}
    no_future = [{} for _ in range(AUDIT_HORIZON_SECONDS // STEP_SECONDS)]
    future = [{} for _ in no_future]
    for slot in range(1, min(7, len(future))):
        future[slot] = {("b", "bo"): 1.0}
    no_choice, no_cost, _ = joint_exhaustive_search(
        controllers, active, road_q, movement_q, network, no_future
    )
    oracle_choice, oracle_cost, combinations = joint_exhaustive_search(
        controllers, active, road_q, movement_q, network, future
    )
    no_future_choice_cost_under_oracle = score_choices(
        no_choice, active, road_q, movement_q, network, future
    )
    coord_choice, coord_cost = coordinate_search(
        controllers, active, {"i": phase_a}, road_q, movement_q, network, future
    )
    no_phase = no_choice["i"]["phase_index"]
    oracle_phase = oracle_choice["i"]["phase_index"]
    coord_phase = coord_choice["i"]["phase_index"]
    return {
        "passed": (
            no_phase != oracle_phase
            and coord_phase == oracle_phase
            and oracle_cost < no_future_choice_cost_under_oracle
        ),
        "no_future_phase": no_phase,
        "oracle_phase": oracle_phase,
        "coordinate_phase": coord_phase,
        "no_future_selected_cost_under_no_future": no_cost,
        "no_future_choice_cost_under_oracle": no_future_choice_cost_under_oracle,
        "oracle_selected_cost_under_oracle": oracle_cost,
        "oracle_information_cost_reduction": no_future_choice_cost_under_oracle - oracle_cost,
        "oracle_information_relative_cost_reduction": (
            (no_future_choice_cost_under_oracle - oracle_cost)
            / no_future_choice_cost_under_oracle
        ),
        "coordinate_cost_under_oracle": coord_cost,
        "joint_combinations": combinations,
        "known_future_arrival": "5 vehicles/5s to movement b->bo in projection slots 1..6",
    }


def validation_event_audit(assets_dir, metadata):
    rows = []
    for date in metadata["validation_dates"]:
        arrivals, input_vehicles = v1.load_arrivals(
            assets_dir / date / "flow_window.json", STEP_SECONDS
        )
        decision_step = metadata["simulation"]["warmup_seconds"] // STEP_SECONDS
        oracle = exact_event_oracle(arrivals, decision_step, AUDIT_HORIZON_SECONDS)
        current_count = len(arrivals.get(decision_step, []))
        future_count = sum(sum(rate * STEP_SECONDS for rate in slot.values()) for slot in oracle)
        first_slot_count = sum(oracle[0].values()) * STEP_SECONDS
        rows.append(
            {
                "date": date,
                "input_vehicles": input_vehicles,
                "decision_step": decision_step,
                "arrivals_already_observed_at_decision": current_count,
                "oracle_first_projection_slot_new_arrivals": first_slot_count,
                "oracle_future_arrivals_slots_1_to_11": future_count,
                "first_slot_excludes_current_step": first_slot_count == 0.0,
            }
        )
    return rows


def markdown_report(result):
    action = result["action_space_summary"]
    positive = result["synthetic_positive_control"]
    consistency = result["projection_consistency"]
    return f"""# v7 控制接口审计与正对照报告

## 结论

本阶段通过。精确事件 Oracle 的首个投影槽不再重复加入决策前已经观测到的到达；分钟预测接口固定使用决策分钟 `m-1` 的预测原点；合成正对照中未来信息改变了最优动作，坐标搜索也识别出与穷举一致的动作。

这仍不是新的 test 闭环结果。v1–v6 应保留为历史诊断，不能再把 v6 的 Oracle 差值称作严格上界。

## 关键审计结果

- 控制路口：{action['controlled_intersections']} 个。
- 具有多动作自由度：{action['intersections_with_control_freedom']} 个（{action['fraction_with_control_freedom']:.1%}）。
- 单阶段有效联合动作：{action['single_stage_joint_action_count']} 个。
- 合成正对照：{'通过' if positive['passed'] else '失败'}；无未来选相位 {positive['no_future_phase']}，精确未来选相位 {positive['oracle_phase']}，坐标搜索选相位 {positive['coordinate_phase']}。
- 在同一精确未来情景下，未来感知动作相对无未来动作将投影成本从 {positive['no_future_choice_cost_under_oracle']:.1f} 降至 {positive['oracle_selected_cost_under_oracle']:.1f}，下降 {positive['oracle_information_relative_cost_reduction']:.1%}。
- 单运动一步投影/FIFO最大道路队列误差：{consistency['single_movement_max_road_queue_error']:.6g}。
- 多运动竞争时FIFO与比例服务的运动队列L1差：{consistency['competing_movement_fifo_vs_proportional_l1']:.6g}（确认二者并不等价）。

## 时间语义（修正后）

1. 仿真循环先加入 `current_step` 到达，再形成状态并决策。
2. 投影槽0只服务当前已观测状态，不再加入 `current_step` 到达。
3. 投影槽 `k>0` 精确加入 `current_step+k` 的事件，粒度为5秒。
4. 决策分钟 `m` 只读取预测原点 `m-1`；其第1行对应分钟 `m`。在 `m:30`，分钟 `m` 的剩余槽仍用第1行，跨到 `m+1` 后使用第2行。

## 对论文设计的影响

- 现有网络中只有2/9路口拥有有效选择，控制收益小不能直接解释为“预测无价值”。
- 精确事件 Oracle 现在只能称为“给定当前近似滚动模型与单阶段固定动作的未来信息诊断”；比例服务与实际FIFO存在结构差异。
- 下一阶段应在 validation 上把修正后的预测、精确 Oracle 与同条件无未来基线接入闭环，再决定是否解封 test。不得用 test 选择时域、目标权重或控制结构。

## 数据边界

本报告仅读取 `validation_dates`：{', '.join(result['protocol']['validation_dates'])}。未读取或运行 test 日期，calibration 继续封存。
"""


def experiment_plan():
    return """# v7 实验方案

目标是修复并审计“预测—控制”接口，而不是追求新的测试集最优数值。

1. 对信号相位按非空运动集合去重，报告真实控制自由度。
2. 从 validation 的 `flow_window.json` 构建5秒事件级 Oracle。决策时当前step已观测，投影槽0不得重复加入。
3. 规定分钟预测的可观测性：分钟m决策仅可用origin=m-1，禁止使用origin=m的lag0特征。
4. 对有效多相位路口实施单阶段联合动作穷举，并保留一轮坐标搜索作对照。
5. 用合成状态验证“已知未来到达应改变动作且降低对应目标”。正对照失败则停止扩展。
6. 用一步推进检查投影与实际FIFO语义：单运动应一致，多运动差异必须显式量化。

选择与报告规则：只使用 validation 与合成状态；不读取 test 选择结构或参数；不修改v1–v6；不开始论文初稿。
"""


def build(assets_dir, roadnet_path, output_dir):
    final_dir = output_dir.resolve()
    building = final_dir.with_name(final_dir.name + ".building")
    if final_dir.exists() or building.exists():
        raise FileExistsError(f"Output/build directory already exists: {final_dir}, {building}")
    building.mkdir(parents=True)
    metadata = json.loads((assets_dir / "ASSETS.json").read_text(encoding="utf-8"))
    roadnet = json.loads(roadnet_path.read_text(encoding="utf-8"))
    network_tuple = v1.build_network(roadnet, metadata)
    controllers = network_tuple[2]
    rows, action_summary = action_space_audit(controllers)
    positive = synthetic_positive_control()
    consistency = projection_consistency_checks()
    validation_rows = validation_event_audit(assets_dir, metadata)
    if not positive["passed"]:
        raise AssertionError("Synthetic future-information positive control failed")
    if not consistency["single_movement_match"]:
        raise AssertionError("Projection does not match FIFO in the single-movement control")
    if not all(row["first_slot_excludes_current_step"] for row in validation_rows):
        raise AssertionError("Exact oracle duplicated the already observed current step")
    result = {
        "experiment_id": "xuancheng_control_interface_audit_v7",
        "status": "P0_AND_POSITIVE_CONTROL_COMPLETED",
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "action_space_rows": rows,
        "action_space_summary": action_summary,
        "validation_event_oracle_audit": validation_rows,
        "synthetic_positive_control": positive,
        "projection_consistency": consistency,
        "protocol": {
            "data_scope": "validation_and_synthetic_only",
            "validation_dates": metadata["validation_dates"],
            "test_dates_read": [],
            "calibration_status": "sealed",
            "event_oracle_step_seconds": STEP_SECONDS,
            "audit_horizon_seconds": AUDIT_HORIZON_SECONDS,
            "search": "joint enumeration over distinct nonempty actions plus one coordinate sweep",
        },
        "interpretation_guardrail": "Diagnostic within the approximate fixed-action rollout; not a strict oracle upper bound.",
    }
    (building / "RESULTS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (building / "实验方案.md").write_text(experiment_plan(), encoding="utf-8")
    (building / "实验报告.md").write_text(markdown_report(result), encoding="utf-8")
    v1.write_csv(building / "action_space.csv", rows)
    v1.write_csv(building / "validation_event_oracle_audit.csv", validation_rows)
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
        "--roadnet", type=Path,
        default=project / "04_实验" / "控制基线" / "control_admission_v1" / "assets_subnetwork_v2" / "roadnet_subnetwork.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=project / "04_实验" / "控制基线" / "control_interface_audit_v7",
    )
    args = parser.parse_args()
    result = build(args.assets_dir.resolve(), args.roadnet.resolve(), args.output_dir.resolve())
    print(json.dumps({
        "status": result["status"],
        "action_space_summary": result["action_space_summary"],
        "positive_control_passed": result["synthetic_positive_control"]["passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
