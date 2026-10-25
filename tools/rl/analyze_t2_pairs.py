"""v3.6.1 T2 paired analysis: tactic arm vs baseline arm, pair by pair.

Pre-registered in ``v3.6.1/执行计划-T2.md``.  The design is:

* episodes alternate arms, so pair k is (baseline ep 2k+1, tactic ep 2k+2);
* both arms of a pair share the same seeded warm-up, so they start from the same
  initial condition;
* the primary metric is decided before the run, not after looking at the table.

Everything is computed from the recorded traces by one code path, so both arms are
measured identically:

    python tools/rl/analyze_t2_pairs.py --probe <dir> --tactic clear_enemy \
        --primary attributed_spawns_per_decision --baseline-min 16 \
        --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from touhou_rl import spec
from touhou_rl.bullet_track import attribute_spawns, spawns_by_slot
from touhou_rl.trace35 import read_episodes, read_trace

SPAWN_RADIUS = 32.0
SAFE_CLEARANCE = 6.0
BOOTSTRAP = 5000
BOOTSTRAP_SEED = 3611
WALL_MARGIN = 8.0


def _clearance(px: float, py: float, pos: np.ndarray, half: np.ndarray) -> float:
    """Same definition as ``ablate36.clearance_now`` (cap 60, floor -10)."""
    if pos.shape[0] == 0:
        return 60.0
    dist = np.sqrt((pos[:, 0] - px) ** 2 + (pos[:, 1] - py) ** 2) - half - 0.825 - 0.5
    return float(max(-10.0, min(60.0, dist.min())))


def episode_metrics(probe: Path, episode: str) -> dict:
    """All per-episode quantities, computed the same way for every arm."""
    records = read_trace(probe / "trace" / f"{episode}.jsonl")
    bullets = np.load(probe / "bullets" / f"{episode}.npz")
    offsets = bullets["offsets"]
    slots_all = bullets["slots"] if "slots" in bullets else None
    positions = bullets["positions"]
    velocities = bullets["velocities"]
    halfs = bullets["half"]

    spawns = 0
    attributed = 0
    transitions = 0
    bullets_sum = 0.0
    deaths = 0
    bombs_used = 0
    frames: list[int] = []
    actions: list[tuple[int, int, int, int]] = []
    wall_frames = 0
    player_frames = 0
    safe_gain = 0.0
    safe_frames = 0
    power_series: list[float] = []
    clearance_series: list[float] = []
    intent_changes = 0
    last_intent = None
    warmup = 0

    prev_slots = None
    prev_power = None
    for index, record in enumerate(records):
        start, end = int(offsets[index]), int(offsets[index + 1])
        pos = positions[start:end]
        vel = velocities[start:end]
        half = halfs[start:end]
        cur_slots = slots_all[start:end] if slots_all is not None else None

        player = record.get("player") or {}
        px = float(player.get("x") or 0.0)
        py = float(player.get("y") or 0.0)
        resources = record.get("resources") or {}
        power = float(resources.get("power") or 0.0)
        clearance = _clearance(px, py, pos, half)

        if prev_slots is not None and cur_slots is not None:
            new_mask = spawns_by_slot(prev_slots, cur_slots)
            transitions += 1
            spawns += int(new_mask.sum())
            spawn_pos = pos[new_mask]
            enemies = np.array(
                [[float(e.get("x", 0.0)), float(e.get("y", 0.0))] for e in (record.get("enemies") or [])],
                dtype=np.float32,
            ).reshape(-1, 2)
            if spawn_pos.shape[0]:
                attributed += int(attribute_spawns(spawn_pos, enemies, SPAWN_RADIUS)["counts"].sum())

        bullets_sum += float(pos.shape[0])
        frames.append(int(record.get("frame_post") or 0))
        events = record.get("events") or []
        if "death" in events:
            deaths += 1
        bombs_used = max(bombs_used, int(float(resources.get("bombs_used") or 0)))
        action = record.get("action") or [0, 1, 0, 0]
        actions.append(tuple(int(value) for value in action))
        if px <= spec.MOVE_AREA_X + WALL_MARGIN or px >= spec.MOVE_AREA_X + spec.MOVE_AREA_W - WALL_MARGIN:
            wall_frames += 1
        player_frames += 1

        controller = record.get("controller") or {}
        if controller.get("phase") == "warmup":
            warmup += 1
        intent = controller.get("tactic") or controller.get("override") or controller.get("phase")
        if intent != last_intent:
            intent_changes += 1
            last_intent = intent

        if prev_power is not None and clearance >= SAFE_CLEARANCE:
            safe_gain += max(0.0, power - prev_power)
            safe_frames += 1
        prev_power = power
        power_series.append(power)
        clearance_series.append(clearance)
        prev_slots = cur_slots

    action_counts: dict[tuple[int, int, int, int], int] = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    total = max(1, len(actions))
    entropy = -sum((count / total) * math.log(count / total) for count in action_counts.values())

    return {
        "episode": episode,
        "decisions": len(records),
        "comparisons": transitions,
        "attributed_spawns": attributed,
        "spawns": spawns,
        "attributed_spawns_per_decision": attributed / transitions if transitions else None,
        "spawns_per_decision": spawns / transitions if transitions else None,
        "mean_bullets": bullets_sum / total,
        "final_frame": frames[-1] if frames else 0,
        "deaths": deaths,
        "bombs_used": bombs_used,
        "max_power": max(power_series) if power_series else 0.0,
        "mean_power": float(np.mean(power_series)) if power_series else 0.0,
        "safe_power_gain_per_frame": safe_gain / safe_frames if safe_frames else None,
        "safe_frames": safe_frames,
        "mean_clearance": float(np.mean(clearance_series)) if clearance_series else None,
        "wall_hug_fraction": wall_frames / player_frames if player_frames else 0.0,
        "action_entropy": entropy,
        "distinct_actions": len(action_counts),
        "intent_changes": intent_changes,
        "warmup_decisions": warmup,
    }


def build_pairs(probe: Path, tactic: str) -> tuple[list[dict], list[str]]:
    """Pair consecutive episodes; both members of a pair must share the warm-up."""
    episodes = read_episodes(probe)
    episodes.sort(key=lambda item: int(str(item["episode"]).rsplit("-ep", 1)[-1]))
    problems: list[str] = []
    pairs: list[dict] = []
    for offset in range(0, len(episodes) - 1, 2):
        first, second = episodes[offset], episodes[offset + 1]
        if (first.get("tactic"), second.get("tactic")) != ("none", tactic):
            problems.append(
                f"pair at offset {offset}: expected (none, {tactic}), got "
                f"({first.get('tactic')}, {second.get('tactic')})"
            )
            continue
        if first.get("warmup_decisions") != second.get("warmup_decisions"):
            problems.append(
                f"pair {offset // 2}: warm-up mismatch "
                f"{first.get('warmup_decisions')} vs {second.get('warmup_decisions')}"
            )
        pairs.append({"pair": offset // 2, "baseline": first, "tactic": second})
    if len(episodes) % 2:
        problems.append(f"odd episode count ({len(episodes)}); trailing episode dropped")
    return pairs, problems


def bootstrap_ci(values: np.ndarray, seed: int = BOOTSTRAP_SEED, draws: int = BOOTSTRAP) -> dict:
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(draws)
    for i in range(draws):
        means[i] = values[rng.integers(0, n, n)].mean()
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        # one-sided: probability the reduction is <= 0
        "p_le_zero": float((means <= 0.0).mean()),
        "draws": draws,
        "seed": seed,
    }


def analyse(probe: Path, tactic: str, primary: str) -> dict:
    pairs, problems = build_pairs(probe, tactic)
    rows = []
    for pair in pairs:
        base = episode_metrics(probe, pair["baseline"]["episode"])
        tact = episode_metrics(probe, pair["tactic"]["episode"])
        base_value = base.get(primary)
        tact_value = tact.get(primary)
        reduction = None
        if base_value not in (None, 0) and tact_value is not None:
            reduction = 1.0 - float(tact_value) / float(base_value)
        rows.append(
            {
                "pair": pair["pair"],
                "baseline": base,
                "tactic": tact,
                primary: {"baseline": base_value, "tactic": tact_value, "reduction": reduction},
                "final_frame_ratio": (tact["final_frame"] / base["final_frame"]) if base["final_frame"] else None,
            }
        )
    reductions = np.asarray([row[primary]["reduction"] for row in rows if row[primary]["reduction"] is not None], dtype=float)
    frame_ratios = np.asarray([row["final_frame_ratio"] for row in rows if row["final_frame_ratio"] is not None], dtype=float)
    death_delta = np.asarray(
        [row["tactic"]["deaths"] - row["baseline"]["deaths"] for row in rows], dtype=float
    )

    stats = bootstrap_ci(reductions) if reductions.size >= 3 else {"mean": None, "p_le_zero": None}
    frame_stats = bootstrap_ci(frame_ratios - 1.0) if frame_ratios.size >= 3 else {"mean": None}
    death_stats = bootstrap_ci(death_delta) if death_delta.size else {"mean": None}

    usable = len(rows)
    gate_e1 = None
    if stats.get("mean") is not None and stats.get("p_le_zero") is not None:
        gate_e1 = bool(stats["mean"] >= 0.15 and stats["p_le_zero"] < 0.05)
    # Final frame ratio >= 0.90 means the tactic arm survived at least 90% as long.
    gate_e2 = None
    if frame_stats.get("mean") is not None:
        gate_e2 = bool((1.0 + frame_stats["mean"]) >= 0.90)
    gate_e3 = None
    if death_stats.get("mean") is not None:
        gate_e3 = bool(death_stats["mean"] <= 0.10)
    if usable < 16:
        verdict = "underpowered"
    elif gate_e2 is False or gate_e3 is False:
        verdict = "fail_safety"
    elif gate_e1:
        verdict = "pass"
    else:
        verdict = "fail"

    return {
        "schema": "v361-t2-paired/1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "probe": str(probe),
        "tactic": tactic,
        "primary": primary,
        "design": {
            "spawn_radius_px": SPAWN_RADIUS,
            "safe_clearance_px": SAFE_CLEARANCE,
            "bootstrap_draws": BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "pairs": usable,
        },
        "problems": problems,
        "pairs": rows,
        "summary": {
            "primary_reduction": stats,
            "final_frame_ratio": frame_stats,
            "death_delta": death_stats,
            "mean_final_frame": {
                "baseline": float(np.mean([row["baseline"]["final_frame"] for row in rows])) if rows else None,
                "tactic": float(np.mean([row["tactic"]["final_frame"] for row in rows])) if rows else None,
            },
            "mean_attributed_per_decision": {
                "baseline": float(np.mean([row["baseline"]["attributed_spawns_per_decision"] for row in rows])) if rows else None,
                "tactic": float(np.mean([row["tactic"]["attributed_spawns_per_decision"] for row in rows])) if rows else None,
            },
            "gates": {
                "T2-E1_primary": gate_e1,
                "T2-E2_survival": gate_e2,
                "T2-E3_deaths": gate_e3,
            },
            "verdict": verdict,
        },
    }


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    pairs = report["pairs"]
    primary = report["primary"]
    lines = [
        f"# T2 配对比较：{report['tactic']} vs baseline",
        "",
        f"- 生成：{report['generated']}",
        f"- 数据：`{report['probe']}`（{report['design']['pairs']} 对）",
        f"- 主指标：`{primary}`；归属半径 {report['design']['spawn_radius_px']} px；bootstrap {report['design']['bootstrap_draws']} 次",
        f"- 判定：**{summary['verdict']}**",
        "",
        "## 1 门禁",
        "",
        "| 编号 | 判据 | 阈值 | 实测 | 判定 |",
        "|---|---|---:|---:|---|",
        f"| T2-E1 | 主指标下降（配对均值） | ≥15% | {100 * (summary['primary_reduction'].get('mean') or 0):.1f}% "
        f"(95% CI {100 * (summary['primary_reduction'].get('ci_low') or 0):.1f}% … {100 * (summary['primary_reduction'].get('ci_high') or 0):.1f}%) "
        f"| {'✅' if summary['gates']['T2-E1_primary'] else '❌'} |",
        f"| T2-E2 | 生存帧数比 | ≥90% | {100 * (1 + (summary['final_frame_ratio'].get('mean') or 0)):.1f}% "
        f"| {'✅' if summary['gates']['T2-E2_survival'] else '❌'} |",
        f"| T2-E3 | 死亡次数增量 | ≤+0.10 | {summary['death_delta'].get('mean'):+.3f} "
        f"| {'✅' if summary['gates']['T2-E3_deaths'] else '❌'} |",
        "",
        "## 2 逐对",
        "",
        f"| 对 | baseline {primary} | tactic {primary} | 下降 | baseline 帧 | tactic 帧 | 比 | 死亡 Δ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pairs:
        value = row[primary]
        lines.append(
            f"| {row['pair']} | {value['baseline']:.4f} | {value['tactic']:.4f} | "
            f"{100 * (value['reduction'] or 0):.1f}% | {row['baseline']['final_frame']} | "
            f"{row['tactic']['final_frame']} | {row['final_frame_ratio']:.3f} | "
            f"{row['tactic']['deaths'] - row['baseline']['deaths']:+d} |"
        )
    lines += [
        "",
        "## 3 次指标",
        "",
        "| 指标 | baseline | tactic |",
        "|---|---:|---:|",
        f"| 平均归属发弹/决策 | {summary['mean_attributed_per_decision']['baseline']:.4f} | {summary['mean_attributed_per_decision']['tactic']:.4f} |",
        f"| 平均生存帧 | {summary['mean_final_frame']['baseline']:.0f} | {summary['mean_final_frame']['tactic']:.0f} |",
    ]
    for key in (
        "mean_bullets",
        "mean_power",
        "max_power",
        "safe_power_gain_per_frame",
        "wall_hug_fraction",
        "action_entropy",
        "intent_changes",
        "bombs_used",
    ):
        base_values = [row["baseline"].get(key) for row in pairs if row["baseline"].get(key) is not None]
        tact_values = [row["tactic"].get(key) for row in pairs if row["tactic"].get(key) is not None]
        if not base_values or not tact_values:
            continue
        lines.append(f"| {key} | {np.mean(base_values):.4f} | {np.mean(tact_values):.4f} |")
    if report["problems"]:
        lines += ["", "## 4 问题", ""] + [f"- {item}" for item in report["problems"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/rl/analyze_t2_pairs.py")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--tactic", default="clear_enemy", choices=["clear_enemy", "collect_power"])
    parser.add_argument(
        "--primary",
        default="attributed_spawns_per_decision",
        choices=["attributed_spawns_per_decision", "safe_power_gain_per_frame"],
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    report = analyse(args.probe, args.tactic, args.primary)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "paired.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "paired.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
