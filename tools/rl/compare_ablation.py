"""Side-by-side D1/D2 view of the v3.6.0 T1 ablation.

The two runs differ in *data source* (local pilot vs human replays), *fold count*
(9 vs 6) and *movement model* (Border constants vs measured per shot type), so
absolute values are not comparable.  Only two things are comparable and both are
reported here: the **sign/direction** of each increment and whether the
pre-registered gate was passed.

It also checks the six falsifiable predictions that 执行计划-D2补遗 §5 wrote down
*before* D2 existed, because a prediction that turns out wrong is worth more
than one that turns out right.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STRATEGIES = ("LocalDodge", "REGION", "LURE", "ENEMY", "POWER", "BOSS", "EMERGENCY", "HISTORY")
VERDICT_MARK = {
    "pass": "✅ pass",
    "fail": "❌ fail",
    "unstable": "⚠ unstable",
    "underpowered": "⚠ underpowered",
    "not_tested": "— not_tested",
}


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def gate_cell(report: dict, strategy: str) -> dict | None:
    """The cell each strategy's pre-registered *increment* gate was evaluated on."""
    row = report["verdicts"]["per_strategy"].get(strategy)
    if not row or not row.get("increments"):
        return None
    label = row["increments"][0]["label"]
    for cell in report["cells"]:
        if cell["owner"] == strategy and cell["label"] == label:
            return cell
    return None


def fmt(value, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if signed:
        return f"{number:+.{digits}f}"
    if number == int(number) and abs(number) < 1e6:
        return str(int(number))
    return f"{number:.{digits}f}"


def survival_metric(report: dict) -> str:
    fallback = report.get("survival_fallback") or {}
    if fallback.get("fallback"):
        return f"{fallback['fallback_label']} (spearman)"
    return f"{fallback.get('label', 'death_60')} (auc)"


def check_predictions(d1: dict, d2: dict) -> list[tuple[str, str, str, str]]:
    """(id, statement, outcome, evidence) for the six pre-registered predictions."""
    v2 = d2["verdicts"]["per_strategy"]
    c1 = {name: gate_cell(d1, name) for name in STRATEGIES}
    c2 = {name: gate_cell(d2, name) for name in STRATEGIES}
    rows: list[tuple[str, str, str, str]] = []

    p1 = v2.get("POWER", {}).get("verdict")
    rows.append(("P1", "POWER 在 D2 仍然 pass", "命中" if p1 == "pass" else f"未命中（{p1}）", f"D2 verdict={p1}"))

    d1_em = (c1.get("EMERGENCY") or {}).get("delta")
    d2_em_verdict = v2.get("EMERGENCY", {}).get("verdict")
    if d2_em_verdict == "underpowered":
        outcome = "不可判（标签功效不足）"
    elif d1_em is None or (c2.get("EMERGENCY") or {}).get("delta") is None:
        outcome = "不可判"
    else:
        outcome = "命中" if c2["EMERGENCY"]["delta"] > d1_em else "未命中"
    rows.append(
        (
            "P2",
            "EMERGENCY 在 D2 的增量显著高于 D1",
            outcome,
            f"D1 Δ={fmt(d1_em, signed=True)}；D2 判定={d2_em_verdict}",
        )
    )

    d2_region = (c2.get("REGION") or {}).get("delta")
    rows.append(
        (
            "P3",
            "REGION 在 D2 的生存增量仍 ≤ 0",
            "命中" if d2_region is not None and d2_region <= 0 else "未命中",
            f"D2 Δ={fmt(d2_region, signed=True)}",
        )
    )

    d2_boss = v2.get("BOSS", {})
    rows.append(
        (
            "P4",
            "BOSS 在 D2 从 unstable 变为可判（stable）",
            "命中" if d2_boss.get("stability") == "stable" else "未命中",
            f"D2 verdict={d2_boss.get('verdict')} stability={d2_boss.get('stability')}",
        )
    )

    d2_hist = v2.get("HISTORY", {}).get("verdict")
    rows.append(("P5", "HISTORY 仍然 fail", "命中" if d2_hist == "fail" else f"未命中（{d2_hist}）", f"D2 verdict={d2_hist}"))

    fallback = (d2.get("survival_fallback") or {}).get("fallback")
    rows.append(
        (
            "P6",
            "death_60 落入 underpowered 分支（阳性 <100）",
            "命中" if fallback else "未命中",
            f"death_60 阳性 = {(d2.get('survival_fallback') or {}).get('positives')}",
        )
    )
    return rows


def render(d1: dict, d2: dict, d1_name: str, d2_name: str) -> str:
    lines = [
        "# v3.6.0 T1 消融：D1（局部 pilot）vs D2（人类回放）",
        "",
        f"- D1：`{d1_name}` — {d1['inventory']['episodes']} 局 / {d1['inventory']['rows']} 决策；"
        f"速度口径 `{d1['inventory'].get('speed_mode', 'config')}`；LOEO {d1['inventory']['episodes']} 折；"
        f"生存标签 {survival_metric(d1)}",
        f"- D2：`{d2_name}` — {d2['inventory']['episodes']} 局 / {d2['inventory']['rows']} 决策；"
        f"速度口径 `{d2['inventory'].get('speed_mode', 'config')}`；LOEO {d2['inventory']['episodes']} 折；"
        f"生存标签 {survival_metric(d2)}",
        "",
        "> 两个运行的模型、标签定义、阈值与置换次数完全相同；差别只来自**数据源**、**折数**、**速度口径**，",
        "> 以及 D2 因 `death_60` 阳性为 0 而触发的**预注册生存标签回退**。",
        "> 因此**绝对值不可直接比较**，只比较增量方向与是否过门。",
        "",
        "## 1 逐策略对照",
        "",
        "| 策略 | D1 判定 | D1 增量 | D1 p | D2 判定 | D2 增量 | D2 p | 方向一致 | 变化 |",
        "|---|---|---:|---:|---|---:|---:|---|---|",
    ]
    flips = []
    for strategy in STRATEGIES:
        v1 = d1["verdicts"]["per_strategy"].get(strategy, {})
        v2 = d2["verdicts"]["per_strategy"].get(strategy, {})
        c1 = gate_cell(d1, strategy)
        c2 = gate_cell(d2, strategy)
        d1_delta = None if c1 is None else c1.get("delta")
        d2_delta = None if c2 is None else c2.get("delta")
        p1 = None if c1 is None else c1.get("permutation_p")
        p2 = None if c2 is None else c2.get("permutation_p")
        agree = "—"
        if d1_delta is not None and d2_delta is not None:
            if (d1_delta >= 0) == (d2_delta >= 0):
                agree = "是"
            else:
                agree = "**否**"
                flips.append(strategy)
        change = ""
        if v1.get("verdict") != v2.get("verdict"):
            change = f"{v1.get('verdict')} → **{v2.get('verdict')}**"
        lines.append(
            f"| {strategy} | {VERDICT_MARK.get(v1.get('verdict'), '—')} | {fmt(d1_delta, signed=True)} | {fmt(p1, 3)} | "
            f"{VERDICT_MARK.get(v2.get('verdict'), '—')} | {fmt(d2_delta, signed=True)} | {fmt(p2, 3)} | {agree} | {change} |"
        )
    lines += ["", "## 2 全局门", "", "| 门 | D1 | D2 |", "|---|---|---|"]
    g1 = d1["verdicts"]["gates"]
    g2 = d2["verdicts"]["gates"]
    for name in ("G-1", "G-2", "G-3", "G-4"):
        label = g1[name]["name"]
        signed = name == "G-3"
        if name == "G-3":
            label = f"ALL vs BASE {survival_metric(d1)} vs {survival_metric(d2)} ≥ +0.05"
        lines.append(
            f"| {name} {label} | {fmt(g1[name].get('value'), signed=signed)} → {'通过' if g1[name]['pass'] else '未过'} | "
            f"{fmt(g2[name].get('value'), signed=signed)} → {'通过' if g2[name]['pass'] else '未过'} |"
        )
    lines += [
        "",
        f"- D1 `enters_T2` = **{d1['verdicts']['enters_T2']}**；D2 `enters_T2` = **{d2['verdicts']['enters_T2']}**",
        f"- 增量方向翻转的策略：{flips if flips else '无'}",
        "",
        "## 3 预注册预测核对（预测写在 D2 采集之前）",
        "",
        "| 编号 | 预测 | 结果 | 证据 |",
        "|---|---|---|---|",
    ]
    for pid, statement, outcome, evidence in check_predictions(d1, d2):
        lines.append(f"| {pid} | {statement} | {outcome} | {evidence} |")
    lines += [
        "",
        "## 4 口径与诊断差异",
        "",
        "| 项 | D1 | D2 |",
        "|---|---|---|",
        f"| 生存标签 | {survival_metric(d1)} | {survival_metric(d2)} |",
        f"| 空特征 pooled 指标 | {fmt(d1['diagnostics']['degenerate_auc'])} | {fmt(d2['diagnostics']['degenerate_auc'])} |",
        f"| 空特征局均指标 | {fmt(d1['diagnostics']['degenerate_auc_ep'])} | {fmt(d2['diagnostics']['degenerate_auc_ep'])} |",
        f"| 近常量二值标签 | {d1['diagnostics'].get('degenerate_labels')} | {d2['diagnostics'].get('degenerate_labels')} |",
        f"| 零散布置换单元格 | {d1['diagnostics']['degenerate_cells']} | {d2['diagnostics']['degenerate_cells']} |",
        f"| unstable 单元格 | {d1['diagnostics'].get('unstable_cells')} | {d2['diagnostics'].get('unstable_cells')} |",
        f"| pooled/局均符号不一致 | {d1['diagnostics']['pooled_vs_ep_disagreements']} | {d2['diagnostics']['pooled_vs_ep_disagreements']} |",
        "",
        "## 5 各局实测移动速度（D2，`--speed-mode measure`）",
        "",
        "| episode | speed | focus_speed | 轴样本 | 回退 |",
        "|---|---:|---:|---:|---|",
    ]
    for row, speeds in zip(d2["inventory"]["per_episode"], d2["inventory"].get("measured_speeds") or []):
        if not speeds:
            lines.append(f"| {row['episode']} | n/a | n/a | 0 | 未测 |")
            continue
        lines.append(
            f"| {row['episode']} | {fmt(speeds.get('speed'), 2)} | {fmt(speeds.get('focus_speed'), 2)} | "
            f"{speeds.get('samples_axis_normal')} | {speeds.get('fallback') or '—'} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def render_three(d1: dict, d2: dict, d3: dict, names: tuple[str, str, str]) -> str:
    """Three-source table: verdicts and gate deltas side by side."""
    gate_label = {
        source: surveillance_gate_label(report)
        for source, report in zip(("D1", "D2", "D3"), (d1, d2, d3))
    }
    lines = [
        "",
        "## 6 D1 / D2 / D3 三源对照",
        "",
        f"- D1：{names[0]}（{d1['inventory']['episodes']} 局，gate set `{d1['protocol'].get('gate_set', 'v1')}`，"
        f"生存标签 {survival_metric(d1)}）",
        f"- D2：{names[1]}（{d2['inventory']['episodes']} 局，gate set `{d2['protocol'].get('gate_set', 'v1')}`，"
        f"生存标签 {survival_metric(d2)}）",
        f"- D3：{names[2]}（{d3['inventory']['episodes']} 局，gate set `{d3['protocol'].get('gate_set', 'v1')}`，"
        f"生存标签 {survival_metric(d3)}）",
        "",
        "| 策略 | D1 | D2 | D3 | D1 增量 | D2 增量 | D3 增量 | D3 门禁标签 |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for strategy in STRATEGIES:
        cells = []
        for report in (d1, d2, d3):
            verdict = report["verdicts"]["per_strategy"].get(strategy, {}).get("verdict")
            cells.append(VERDICT_MARK.get(verdict, "—"))
        deltas = []
        for report in (d1, d2, d3):
            cell = gate_cell(report, strategy)
            deltas.append(fmt(None if cell is None else cell.get("delta"), signed=True))
        label = D3_LABELS.get(strategy, "")
        lines.append(
            f"| {strategy} | {cells[0]} | {cells[1]} | {cells[2]} | {deltas[0]} | {deltas[1]} | {deltas[2]} | {label} |"
        )
    lines += ["", "| 全局门 | D1 | D2 | D3 |", "|---|---|---|---|"]
    for name in ("G-1", "G-2", "G-3", "G-4"):
        marks = []
        for report in (d1, d2, d3):
            gate = report["verdicts"]["gates"][name]
            marks.append(f"{fmt(gate.get('value'), signed=(name == 'G-3'))} → {'通过' if gate['pass'] else '未过'}")
        lines.append(f"| {name} | {marks[0]} | {marks[1]} | {marks[2]} |")
    lines += [
        "",
        f"- `enters_T2`：D1 **{d1['verdicts']['enters_T2']}** / D2 **{d2['verdicts']['enters_T2']}** / "
        f"D3 **{d3['verdicts']['enters_T2']}**",
        f"- D3 生存指标门禁标签：{gate_label['D3']}",
        "",
    ]
    return "\n".join(lines) + "\n"


D3_LABELS = {
    "ENEMY": "attributed_spawns_60（换标签）",
    "BOSS": "boss_life_delta_60（新相位列）",
}


def surveillance_gate_label(report: dict) -> str:
    fallback = report.get("survival_fallback") or {}
    if fallback.get("fallback"):
        return f"{fallback['fallback_label']} (spearman)"
    return f"{fallback.get('label', 'death_60')} (auc)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/rl/compare_ablation.py")
    parser.add_argument("--d1", type=Path, required=True)
    parser.add_argument("--d2", type=Path, required=True)
    parser.add_argument("--d3", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    d1 = load(args.d1)
    d2 = load(args.d2)
    text = render(d1, d2, str(args.d1), str(args.d2))
    if args.d3 is not None:
        d3 = load(args.d3)
        text += render_three(d1, d2, d3, (str(args.d1), str(args.d2), str(args.d3)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
