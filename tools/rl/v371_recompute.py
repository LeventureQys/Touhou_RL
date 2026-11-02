"""v3.7.1 table recomputation: every reported number comes from a raw JSON file.

Why this file exists
--------------------
v3.7.0's reports were written by copying numbers out of ad-hoc scripts into
prose, so the ``1207 passed`` in the settlement document disagreed with the
``1211 passed`` in the log, and the pair table could not be regenerated.  C-08
forbids repeating that: the reports quote tables printed by this script, and
this script only reads ``data/*.json``.

It also implements the statistical rule the v3.7.1 preregistration fixes *before*
any confirmation run: a paired difference is reported with its paired t interval,
its sign counts, its sample size **and** the number of excluded pairs.  A
cross-zero interval is printed as ``not established``, never as "improved".

    python tools/rl/v371_recompute.py gates --data Document/Update/Alpha/v3.7.1/data
    python tools/rl/v371_recompute.py pairs --data Document/Update/Alpha/v3.7.1/data/live-pairs.json
    python tools/rl/v371_recompute.py report --data Document/Update/Alpha/v3.7.1/data --out <md>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

VERSION = "v3.7.1"


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _t_critical_975(n: int) -> float:
    """Two-sided 95% Student-t critical value for ``df = n - 1``.

    A small table covers the sample sizes this protocol allows (n <= 30); larger
    samples fall back to the normal value, which is the correct limit.
    """

    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    return table.get(int(n), 1.960)


def paired_summary(differences: list[float]) -> dict:
    """Mean paired difference with a 95% t interval and honest wording."""

    n = len(differences)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "verdict": "no data"}
    mean = sum(differences) / n
    if n == 1:
        return {
            "n": 1,
            "mean": round(mean, 4),
            "ci_low": None,
            "ci_high": None,
            "sd": None,
            "verdict": "underpowered (n=1, no interval)",
        }
    variance = sum((value - mean) ** 2 for value in differences) / (n - 1)
    sd = math.sqrt(variance)
    half = _t_critical_975(n) * sd / math.sqrt(n)
    low, high = mean - half, mean + half
    if high < 0:
        verdict = "difference established (interval entirely below 0)"
    elif low > 0:
        verdict = "difference established (interval entirely above 0)"
    else:
        verdict = "not established (interval crosses 0)"
    return {
        "n": n,
        "mean": round(mean, 4),
        "sd": round(sd, 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def gates_table(document: dict) -> str:
    lines = [
        f"| 门禁 | 状态 | 依据 |",
        "|---|---|---|",
    ]
    detail = {
        "C-01": "规划耗时 0/1/3/10 帧时均按 gameplay_frame 终止",
        "C-01b": "回退/停滞/离开游戏被分类为技术失败，曝光不为负",
        "C-01c": "同一动作序列经同一管线回放，轨迹逐帧相同",
        "C-02": "两臂同一窗口、同一终帧、同一 hold 分布",
        "C-02b": "窗口内死亡按帧归属，跨边界标 boundary_unknown",
        "C-03": "send_frame 与 effective_frame 分离且可解析",
        "C-03b": "读回不可解析时输出区间，不伪造点估计",
        "C-04": "注入 0/5/15/30/50 ms 延迟只改变迟到帧数，不改变窗口长度",
    }
    for name, entry in sorted(document.get("gates", {}).items()):
        lines.append(
            f"| {name} | {entry['status']} | {detail.get(name, entry.get('detail', ''))} |"
        )
    return "\n".join(lines)


def gates_headline(document: dict) -> dict:
    gates = document.get("gates", {})
    c01 = gates.get("C-01", {})
    c04 = gates.get("C-04", {})
    c03 = gates.get("C-03", {})
    return {
        "passed": document.get("passed"),
        "count": len(gates),
        "exposure_stable_across_planning": c01.get("exposure_values"),
        "injected_delay": c04.get("injected"),
        "effective_frames_after_send": (
            c03.get("send_vs_effective", {}).get("frames_after_send")
        ),
        "changed_resolution_rate": (
            c03.get("send_vs_effective", {}).get("changed_resolution_rate")
        ),
    }


# ---------------------------------------------------------------------------
# pairs
# ---------------------------------------------------------------------------
def _field(entry: dict, arm: str, key: str, *fallbacks, default=None):
    """Read a per-arm field, tolerating the v3.7.0 schema.

    The v3.7.0 pair file has ``deaths`` and, for the subject only, ``frame_span``;
    it has no exposure field for either arm.  Being able to read both schemas is
    what lets this tool show the v3.7.0 defect numerically instead of asserting it.
    """

    arm_entry = entry.get(arm) or {}
    for name in (key, *fallbacks):
        if name in arm_entry and arm_entry[name] is not None:
            return arm_entry[name]
    return default


def _exposure(entry: dict, arm: str) -> int | None:
    value = _field(entry, arm, "exposure_frames")
    if value is not None:
        return int(value)
    span = _field(entry, arm, "frame_span")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        return int(span[1]) - int(span[0])
    return None


def pair_table(document: dict) -> dict:
    """Paired summary over complete pairs, with exclusions listed, not removed silently."""

    entries = document.get("pairs", [])
    complete = [entry for entry in entries if "baseline" in entry and "subject" in entry]
    excluded: list[dict] = []
    usable: list[dict] = []
    for entry in complete:
        failures = []
        for arm in ("baseline", "subject"):
            failure = (entry.get(arm) or {}).get("technical_failure")
            if failure:
                failures.append(f"{arm}:{failure}")
        if failures:
            excluded.append({"pair": entry["pair"], "reasons": failures})
        else:
            usable.append(entry)

    deaths_base = [
        int(_field(entry, "baseline", "deaths_in_window", "deaths", default=0)) for entry in usable
    ]
    deaths_sub = [
        int(_field(entry, "subject", "deaths_in_window", "deaths", default=0)) for entry in usable
    ]
    diffs = [s - b for b, s in zip(deaths_base, deaths_sub)]
    exposure_base = [_exposure(entry, "baseline") for entry in usable]
    exposure_sub = [_exposure(entry, "subject") for entry in usable]
    missing_exposure = sum(1 for value in exposure_base + exposure_sub if value is None)
    paired_exposure = [
        (_exposure(entry, "baseline"), _exposure(entry, "subject")) for entry in usable
    ]
    bombs_base_raw = [_field(entry, "baseline", "bombs_in_window", "bombs") for entry in usable]
    bombs_sub_raw = [_field(entry, "subject", "bombs_in_window", "bombs") for entry in usable]
    bombs_base = [int(v) for v in bombs_base_raw if v is not None]
    bombs_sub = [int(v) for v in bombs_sub_raw if v is not None]
    bombs_missing = (len(bombs_base_raw) - len(bombs_base)) + (len(bombs_sub_raw) - len(bombs_sub))
    first_death_base = [_field(entry, "baseline", "first_death_frame") for entry in usable]
    first_death_sub = [_field(entry, "subject", "first_death_frame") for entry in usable]
    boundary = sum(
        int(entry[arm].get("boundary_unknown", 0) or 0)
        for entry in usable
        for arm in ("baseline", "subject")
    )

    mean_base = sum(deaths_base) / len(deaths_base) if deaths_base else None
    mean_sub = sum(deaths_sub) / len(deaths_sub) if deaths_sub else None
    measured = [value for value in exposure_base + exposure_sub if value is not None]
    measured_base = [value for value in exposure_base if value is not None]
    measured_sub = [value for value in exposure_sub if value is not None]
    same_exposure = sum(1 for a, b in paired_exposure if a is not None and a == b)
    nominal = document.get("frames_per_episode")
    return {
        "schema": document.get("schema"),
        "pairs_total": len(entries),
        "pairs_complete": len(complete),
        "pairs_usable": len(usable),
        "excluded": excluded,
        "deaths_baseline_mean": None if mean_base is None else round(mean_base, 4),
        "deaths_subject_mean": None if mean_sub is None else round(mean_sub, 4),
        "relative_change": (
            None if not mean_base else round((mean_sub - mean_base) / mean_base, 4)
        ),
        "paired_difference": paired_summary(diffs),
        "positive_pairs": sum(1 for value in diffs if value > 0),
        "negative_pairs": sum(1 for value in diffs if value < 0),
        "tied_pairs": sum(1 for value in diffs if value == 0),
        "nominal_frames_per_episode": nominal,
        "exposure_baseline_mean": (
            None if not measured_base else round(sum(measured_base) / len(measured_base), 2)
        ),
        "exposure_subject_mean": (
            None if not measured_sub else round(sum(measured_sub) / len(measured_sub), 2)
        ),
        "exposure_min": min(measured) if measured else None,
        "exposure_max": max(measured) if measured else None,
        "exposure_missing_arms": int(missing_exposure),
        "exposure_identical_pairs": int(same_exposure),
        "bombs_baseline_mean": (
            None if not bombs_base else round(sum(bombs_base) / len(bombs_base), 4)
        ),
        "bombs_subject_mean": None if not bombs_sub else round(sum(bombs_sub) / len(bombs_sub), 4),
        "bombs_missing_arms": int(bombs_missing),
        "first_death_baseline_pairs_with_death": sum(1 for v in first_death_base if v is not None),
        "first_death_subject_pairs_with_death": sum(1 for v in first_death_sub if v is not None),
        "first_death_baseline_mean": _mean_or_none(first_death_base),
        "first_death_subject_mean": _mean_or_none(first_death_sub),
        "boundary_unknown_events": int(boundary),
        "order_scheme": document.get("order_scheme"),
        "order": document.get("order"),
    }


def _mean_or_none(values: list) -> float | None:
    present = [value for value in values if value is not None]
    return round(sum(present) / len(present), 2) if present else None


def pairs_table_text(summary: dict) -> str:
    paired = summary["paired_difference"]
    nominal = summary.get("nominal_frames_per_episode")
    exposure_note = (
        f"名义 {nominal} 帧；实测范围 "
        f"[{summary['exposure_min']}, {summary['exposure_max']}]；"
        f"两臂曝光完全相同 {summary['exposure_identical_pairs']} 对；"
        f"未记录曝光的臂 {summary['exposure_missing_arms']} 个"
    )
    rows = [
        "| 指标 | baseline | subject | 说明 |",
        "|---|---:|---:|---|",
        f"| 可用配对 | {summary['pairs_usable']} | {summary['pairs_usable']} | "
        f"完整 {summary['pairs_complete']} / 总数 {summary['pairs_total']} |",
        f"| 死亡/局（窗口内） | {summary['deaths_baseline_mean']} | {summary['deaths_subject_mean']} | "
        f"相对变化 {summary['relative_change']} |",
        f"| 配对差 (subject-baseline) | {paired['mean']} | 95% CI | "
        f"[{paired['ci_low']}, {paired['ci_high']}] |",
        f"| 配对符号 | +{summary['positive_pairs']} | -{summary['negative_pairs']} | "
        f"持平 {summary['tied_pairs']} |",
        f"| 实际曝光帧（均值） | {summary['exposure_baseline_mean']} | {summary['exposure_subject_mean']} | "
        f"{exposure_note} |",
        f"| bomb/局 | {summary['bombs_baseline_mean']} | {summary['bombs_subject_mean']} | — |",
        f"| 首次死亡帧（均值） | {summary['first_death_baseline_mean']} | "
        f"{summary['first_death_subject_mean']} | 有死亡的对数 "
        f"{summary['first_death_baseline_pairs_with_death']} / {summary['first_death_subject_pairs_with_death']} |",
        f"| boundary_unknown 事件 | — | {summary['boundary_unknown_events']} | 边界不可归属，不计入死亡 |",
        f"| 判定 | {paired['verdict']} | — | 预注册门槛见 preregistration/ |",
    ]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------
def build_report(data_dir: Path) -> str:
    gates_path = data_dir / "c01-c04-gates.json"
    pairs_path = data_dir / "live-pairs.json"
    parts = [f"# v3.7.1 P0 实验有效性审计（自动生成）", ""]
    parts.append(
        "> 本文件由 `tools/rl/v371_recompute.py` 根据 `data/*.json` 生成。"
        "不要手改数字；重跑脚本即可复现。"
    )
    parts.append("")

    if gates_path.is_file():
        gates = json.loads(gates_path.read_text(encoding="utf-8"))
        headline = gates_headline(gates)
        parts.append("## 1. 门禁结果（假时钟，无需游戏）")
        parts.append("")
        parts.append(
            f"总体：**{'pass' if headline['passed'] else 'fail'}**（{headline['count']} 项门禁）"
        )
        parts.append("")
        parts.append(gates_table(gates))
        parts.append("")
        parts.append("### 1.1 关键测量")
        parts.append("")
        parts.append(
            f"- 规划耗时 0/1/3/10 帧时实际曝光帧：`{headline['exposure_stable_across_planning']}`"
            "（同一窗口长度 ⇒ 终止依据是帧，不是决策数）"
        )
        parts.append(
            f"- 注入延迟下的曝光帧：`{headline['injected_delay']}`"
            "（延迟只改变迟到帧数与 hold 长度）"
        )
        parts.append(
            f"- 发送到生效帧数：`{headline['effective_frames_after_send']}`；"
            f"动作变更决策的生效解析率：`{headline['changed_resolution_rate']}`"
        )
        parts.append("")
    else:
        parts.append("## 1. 门禁结果")
        parts.append("")
        parts.append(f"（未找到 `{gates_path}`：门禁尚未运行。）")
        parts.append("")

    if pairs_path.is_file():
        pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
        summary = pair_table(pairs)
        parts.append("## 2. 配对结果")
        parts.append("")
        parts.append(
            f"顺序方案 `{summary['order_scheme']}`，实际顺序 "
            f"`{summary['order']}`"
        )
        parts.append("")
        parts.append(pairs_table_text(summary))
        parts.append("")
        if summary["excluded"]:
            parts.append("### 2.1 剔除（显式列出，不静默丢弃）")
            parts.append("")
            for entry in summary["excluded"]:
                parts.append(f"- pair {entry['pair']}: {entry['reasons']}")
            parts.append("")
    else:
        parts.append("## 2. 配对结果")
        parts.append("")
        parts.append(f"（未找到 `{pairs_path}`：尚未进行公平对照真机运行。）")
        parts.append("")

    parts.append("## 3. 复现命令")
    parts.append("")
    parts.append("```text")
    parts.append("python -m touhou_rl.escape experiment gates --out <data>/c01-c04-gates.json")
    parts.append("python -m touhou_rl.escape experiment pair --pairs 30 "
                 "--order balanced_random --seed 0 --confirm-foreground --out <data>/live-pairs.json")
    parts.append("python tools/rl/v371_recompute.py report --data <data> --out <report>")
    parts.append("```")
    return "\n".join(parts) + "\n"


def power_table(sd: float, baseline_mean: float, *, power: float = 0.80, alpha: float = 0.05) -> dict:
    """Paired-design power arithmetic for the pre-registered sample size.

    This is the calculation the v3.7.0 preregistration did not do: it fixed 30
    pairs first and asked for a 20% effect afterwards.  With the paired SD
    observed in v3.7.0 (1.442 deaths/episode) n=30 can only detect a ~71%
    relative change, so a 20% claim would need hundreds of pairs.
    """

    z_alpha = 1.959964 if abs(alpha - 0.05) < 1e-9 else _t_critical_975(10**6)
    z_power = {0.80: 0.841621, 0.90: 1.281552}.get(round(power, 2), 0.841621)
    factor = (z_alpha + z_power) ** 2

    def required_n(effect: float) -> int:
        if effect <= 0:
            raise ValueError("effect must be positive")
        return int(math.ceil(factor * (sd**2) / (effect**2)))

    def detectable(n: int) -> float:
        return (z_alpha + z_power) * sd / math.sqrt(n)

    relative_targets = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    return {
        "paired_sd": round(sd, 4),
        "baseline_mean": round(baseline_mean, 4),
        "alpha": alpha,
        "power": power,
        "n_reference": 30,
        "min_detectable_effect_at_30": round(detectable(30), 4),
        "min_detectable_relative_at_30": round(detectable(30) / baseline_mean, 4),
        "required_n": {
            f"{target:.0%} relative ({target * baseline_mean:.3f} deaths/ep)": required_n(
                target * baseline_mean
            )
            for target in relative_targets
        },
    }


def power_text(payload: dict) -> str:
    rows = [
        "| 目标相对变化 | 绝对效应（死/局） | 所需对数（80% 功效） |",
        "|---|---:|---:|",
    ]
    for key, value in payload["required_n"].items():
        relative, absolute = key.split(" (")
        rows.append(f"| {relative} | {absolute.rstrip(')')} | {value} |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="recompute the v3.7.1 tables from raw JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    gates = sub.add_parser("gates", help="print the gate table")
    gates.add_argument("--data", required=True, help="directory holding c01-c04-gates.json")

    pairs = sub.add_parser("pairs", help="print the paired summary")
    pairs.add_argument("--data", required=True, help="path to live-pairs.json")

    report = sub.add_parser("report", help="write the full P0 report")
    report.add_argument("--data", required=True, help="directory holding the data files")
    report.add_argument("--out", required=True)

    power = sub.add_parser("power", help="paired power arithmetic (sample-size justification)")
    power.add_argument("--sd", type=float, required=True, help="paired difference SD")
    power.add_argument("--baseline", type=float, required=True, help="baseline deaths per episode")
    power.add_argument("--power", type=float, default=0.80)
    power.add_argument("--alpha", type=float, default=0.05)

    args = parser.parse_args(argv)
    if args.command == "gates":
        document = json.loads((Path(args.data) / "c01-c04-gates.json").read_text(encoding="utf-8"))
        print(gates_table(document))
        print()
        print(json.dumps(gates_headline(document), ensure_ascii=False, indent=2))
        return 0
    if args.command == "pairs":
        document = json.loads(Path(args.data).read_text(encoding="utf-8"))
        summary = pair_table(document)
        print(pairs_table_text(summary))
        print()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "power":
        payload = power_table(args.sd, args.baseline, power=args.power, alpha=args.alpha)
        print(power_text(payload))
        print()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    text = build_report(Path(args.data))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
