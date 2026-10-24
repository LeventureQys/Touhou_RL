"""v3.5.0 S1: offline large-region danger probe over the S0 traces.

This module is analysis only.  It rebuilds a time-expanded danger occupancy from
the full bullet pool already recorded in ``v350-trace/1``, asks whether the
player is inside a *large connected danger region*, whether that region expands
and whether a reachable safe component exists, and compares those features with
the local clearance baseline.

The pre-registered S1 gates (execution plan §4.3) are encoded here; ``verify``
must not change thresholds after seeing results.

Why player-centric components instead of labelling the whole grid
-----------------------------------------------------------------
SciPy/OpenCV are not dependencies of this project.  For S1 the only component
that matters for the gates is the one containing (or nearest to) the player, so
a bounded BFS from the player's cell gives the component area, bounding box and
centroid without a full-grid union-find.  The full-grid danger *area* is still
computed independently, so "large attack somewhere" and "player is inside it"
are separate features.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import spec
from .trace35 import read_episodes, read_trace, sha256_file, stable_split

CELL = 8.0
GRID_W = 48
GRID_H = 56
ORIGIN_X = spec.PLAYFIELD_X
ORIGIN_Y = spec.PLAYFIELD_Y
TIMES = (0, 10, 20, 40, 60)
PLAYER_HALF = 0.825
BULLET_PAD = 0.5
# A "large" region is >=3% of the playfield grid (80 of 2,688 cells); this is
# the S1 operationalisation of "大范围攻击", fixed before running the probe.
MIN_REGION_CELLS = 80
SAFE_MIN_CELLS = 8
DEATH_PREFIX_DECISIONS = 10  # 20 frames at frameskip=2
GATES = {
    "S1-1": 0.80,  # death-before region sample identifiable
    "S1-2": 0.70,  # region identifiable >=20 frames before death
    "S1-3": 0.95,  # safe component success
    "S1-4": 0.05,  # AUC increment over clearance
}


def player_cell(x: float, y: float) -> tuple[int, int]:
    ix = int(math.floor((x - ORIGIN_X) / CELL))
    iy = int(math.floor((y - ORIGIN_Y) / CELL))
    return max(0, min(GRID_H - 1, iy)), max(0, min(GRID_W - 1, ix))


def _splat(points: np.ndarray) -> np.ndarray:
    """Rasterise bullet centres into an 8 px grid with a conservative 3x3 splat."""
    occ = np.zeros((GRID_H, GRID_W), dtype=np.int16)
    if points.shape[0] == 0:
        return occ.astype(bool)
    ix = np.floor((points[:, 0] - ORIGIN_X) / CELL).astype(np.int32)
    iy = np.floor((points[:, 1] - ORIGIN_Y) / CELL).astype(np.int32)
    keep = (ix >= 1) & (ix < GRID_W - 1) & (iy >= 1) & (iy < GRID_H - 1)
    ix, iy = ix[keep], iy[keep]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            np.add.at(occ, (iy + dy, ix + dx), 1)
    return occ > 0


def danger_maps(positions: np.ndarray, velocities: np.ndarray, half: np.ndarray) -> dict[int, np.ndarray]:
    """Time-expanded occupancy for the contract's t=0/10/20/40/60 slices.

    Bullets smaller than half a cell are splatted as points plus their 8
    neighbours; this over-approximates the true circle by at most one cell and is
    intentionally conservative for a danger map.
    """
    maps = {}
    for t in TIMES:
        points = positions + velocities * float(t)
        maps[t] = _splat(points)
    return maps


def bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not cells:
        return None
    ys = [cell[0] for cell in cells]
    xs = [cell[1] for cell in cells]
    return min(ys), min(xs), max(ys), max(xs)


def centroid(cells: list[tuple[int, int]]) -> tuple[float, float] | None:
    if not cells:
        return None
    return (sum(cell[1] for cell in cells) / len(cells), sum(cell[0] for cell in cells) / len(cells))


def component_from(mask: np.ndarray, start: tuple[int, int], limit: int = 1200) -> list[tuple[int, int]]:
    """4-connected flood fill from ``start`` on ``mask`` (bounded)."""
    if not (0 <= start[0] < mask.shape[0] and 0 <= start[1] < mask.shape[1]):
        return []
    if not mask[start]:
        return []
    seen = np.zeros_like(mask, dtype=bool)
    seen[start] = True
    queue = deque([start])
    cells: list[tuple[int, int]] = []
    while queue and len(cells) < limit:
        y, x = queue.popleft()
        cells.append((y, x))
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                queue.append((ny, nx))
    return cells


def safe_component(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    """Reachable free component from the player cell (or its free neighbours)."""
    free = ~mask
    if free[start]:
        return component_from(free, start, limit=2000)
    best: list[tuple[int, int]] = []
    for ny, nx in ((start[0] - 1, start[1]), (start[0] + 1, start[1]),
                   (start[0], start[1] - 1), (start[0], start[1] + 1)):
        if 0 <= ny < free.shape[0] and 0 <= nx < free.shape[1] and free[ny, nx]:
            cells = component_from(free, (ny, nx), limit=2000)
            if len(cells) > len(best):
                best = cells
    return best


def nearest_safe_component(mask: np.ndarray, start: tuple[int, int], radius: int = GRID_H) -> list[tuple[int, int]]:
    """Nearest external free component within ``radius`` cells of the player.

    The player is normally inside the danger region by the time S1 asks the
    question, so looking only at the four immediate neighbours returns "no safe
    component" for a 3x3 conservative splat even when a free corridor is two
    cells away.  Search increasing Chebyshev rings and take the first ring that
    contains free cells; this is the "最近外部安全连通分量" of the contract.
    """
    free = ~mask
    if free[start]:
        return component_from(free, start, limit=2000)
    global_best: list[tuple[int, int]] = []
    for ring in range(1, radius + 1):
        ring_best: list[tuple[int, int]] = []
        for dy in range(-ring, ring + 1):
            for dx in range(-ring, ring + 1):
                if max(abs(dy), abs(dx)) != ring:
                    continue
                ny, nx = start[0] + dy, start[1] + dx
                if 0 <= ny < free.shape[0] and 0 <= nx < free.shape[1] and free[ny, nx]:
                    cells = component_from(free, (ny, nx), limit=2000)
                    if len(cells) > len(ring_best):
                        ring_best = cells
        if len(ring_best) > len(global_best):
            global_best = ring_best
        if len(global_best) >= SAFE_MIN_CELLS:
            return global_best
    return global_best


def min_clearance(player: tuple[float, float], positions: np.ndarray, velocities: np.ndarray, half: np.ndarray) -> float:
    if positions.shape[0] == 0:
        return 60.0
    px = positions[:, 0] + velocities[:, 0]
    py = positions[:, 1] + velocities[:, 1]
    dist = np.sqrt((px - player[0]) ** 2 + (py - player[1]) ** 2) - half - PLAYER_HALF - BULLET_PAD
    return float(max(-10.0, min(60.0, dist.min())))


@dataclass
class DecisionFeature:
    decision: int
    frame: int
    player: tuple[float, float]
    area_frac: float
    in_region_20: bool
    in_region_40: bool
    region_area_20: int
    region_area_40: int
    region_direction: tuple[float, float] | None
    safe_area: int
    clearance: float
    death: bool


def _load_episode(out_dir: Path, name: str) -> tuple[list[dict], np.lib.npyio.NpzFile]:
    records = read_trace(out_dir / "trace" / f"{name}.jsonl")
    bullets = np.load(out_dir / "bullets" / f"{name}.npz")
    return records, bullets


def extract_features(records: list[dict], bullets) -> list[DecisionFeature]:
    positions_all = bullets["positions"]
    velocities_all = bullets["velocities"]
    half_all = bullets["half"]
    offsets = bullets["offsets"]
    features: list[DecisionFeature] = []
    for index, record in enumerate(records):
        start, end = int(offsets[index]), int(offsets[index + 1])
        positions = positions_all[start:end]
        velocities = velocities_all[start:end]
        half = half_all[start:end]
        player = record.get("player") or {}
        px, py = player.get("x"), player.get("y")
        if px is None or py is None:
            continue
        px, py = float(px), float(py)
        maps = danger_maps(positions, velocities, half)
        cell = player_cell(px, py)
        comp20 = component_from(maps[20], cell)
        comp40 = component_from(maps[40], cell)
        area20, area40 = len(comp20), len(comp40)
        direction = None
        if len(comp20) >= MIN_REGION_CELLS and len(comp40) >= MIN_REGION_CELLS:
            c20, c40 = centroid(comp20), centroid(comp40)
            if c20 and c40:
                direction = (c40[0] - c20[0], c40[1] - c20[1])
        safe = nearest_safe_component(maps[20], cell)
        death = "death" in (record.get("events") or [])
        features.append(
            DecisionFeature(
                decision=int(record.get("decision", index + 1)),
                frame=int(record.get("frame_post", 0)),
                player=(px, py),
                area_frac=float(maps[20].mean()),
                in_region_20=area20 >= MIN_REGION_CELLS,
                in_region_40=area40 >= MIN_REGION_CELLS,
                region_area_20=area20,
                region_area_40=area40,
                region_direction=direction,
                safe_area=len(safe),
                clearance=min_clearance((px, py), positions, velocities, half),
                death=death,
            )
        )
    return features


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank-AUC with ties handled by average ranks."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # Average ranks for ties.
    sorted_scores = scores[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = ranks[labels].sum()
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def death_labels(features: list[DecisionFeature], horizon_decisions: int = 30) -> np.ndarray:
    """True when a death event happens within the next ``horizon_decisions`` decisions."""
    labels = np.zeros(len(features), dtype=bool)
    death_indices = [i for i, feature in enumerate(features) if feature.death]
    for index in range(len(features)):
        for death_index in death_indices:
            if index < death_index <= index + horizon_decisions:
                labels[index] = True
                break
    return labels


def gate_report(per_episode: dict[str, list[DecisionFeature]], permutations: int = 500, seed: int = 11) -> dict:
    death_samples = 0
    death_region = 0
    deaths = 0
    deaths_early = 0
    region_decisions = 0
    safe_ok = 0
    all_region: list[float] = []
    all_clearance_risk: list[float] = []
    all_death_label: list[float] = []
    per_episode_rows = []

    for name, features in per_episode.items():
        deaths_here = deaths_early_here = death_samples_here = death_region_here = 0
        for index, feature in enumerate(features):
            if feature.in_region_20 or feature.in_region_40:
                region_decisions += 1
                if feature.safe_area >= SAFE_MIN_CELLS:
                    safe_ok += 1
            if feature.death:
                deaths_here += 1
                prefix = features[max(0, index - DEATH_PREFIX_DECISIONS): index]
                prefix = [item for item in prefix if item.player]
                for item in prefix:
                    death_samples_here += 1
                    if item.in_region_20 or item.in_region_40:
                        death_region_here += 1
                if index - DEATH_PREFIX_DECISIONS >= 0:
                    earlier = features[index - DEATH_PREFIX_DECISIONS]
                    if earlier.in_region_20 or earlier.in_region_40:
                        deaths_early_here += 1
        labels = death_labels(features)
        for feature, label in zip(features, labels):
            all_region.append(1.0 if (feature.in_region_20 or feature.in_region_40) else 0.0)
            all_clearance_risk.append(-feature.clearance)
            all_death_label.append(1.0 if label else 0.0)
        deaths += deaths_here
        deaths_early += deaths_early_here
        death_samples += death_samples_here
        death_region += death_region_here
        per_episode_rows.append(
            {
                "episode": name,
                "decisions": len(features),
                "deaths": deaths_here,
                "death_samples": death_samples_here,
                "death_region_samples": death_region_here,
                "early": deaths_early_here,
                "split": stable_split(name),
            }
        )

    region_auc = auc(np.asarray(all_region), np.asarray(all_death_label))
    clearance_auc = auc(np.asarray(all_clearance_risk), np.asarray(all_death_label))
    labels = np.asarray(all_death_label, dtype=bool)
    region = np.asarray(all_region, dtype=np.float64)
    rng = np.random.default_rng(seed)
    observed = (region_auc - clearance_auc) if region_auc is not None and clearance_auc is not None else None
    p_value = None
    if observed is not None:
        count = 0
        for _ in range(permutations):
            shuffled = rng.permutation(region)
            perm_auc = auc(shuffled, labels)
            if perm_auc is not None and (perm_auc - clearance_auc) >= observed:
                count += 1
        p_value = (count + 1) / (permutations + 1)

    gates = {
        "S1-1": {
            "name": "死亡前区域内样本可识别率",
            "value": death_region / death_samples if death_samples else None,
            "threshold": GATES["S1-1"],
            "numerator": death_region,
            "denominator": death_samples,
            "pass": death_samples > 0 and death_region / death_samples >= GATES["S1-1"],
        },
        "S1-2": {
            "name": "至少 20 帧前可识别的大区域攻击比例",
            "value": deaths_early / deaths if deaths else None,
            "threshold": GATES["S1-2"],
            "numerator": deaths_early,
            "denominator": deaths,
            "pass": deaths > 0 and deaths_early / deaths >= GATES["S1-2"],
        },
        "S1-3": {
            "name": "安全连通分量计算成功率",
            "value": safe_ok / region_decisions if region_decisions else None,
            "threshold": GATES["S1-3"],
            "numerator": safe_ok,
            "denominator": region_decisions,
            "pass": region_decisions > 0 and safe_ok / region_decisions >= GATES["S1-3"],
        },
        "S1-4": {
            "name": "区域特征相对 clearance 的增量",
            "value": observed,
            "threshold": GATES["S1-4"],
            "region_auc": region_auc,
            "clearance_auc": clearance_auc,
            "permutation_p": p_value,
            "permutations": permutations,
            "pass": observed is not None and observed >= GATES["S1-4"] and (p_value is not None and p_value < 0.05),
        },
    }
    failed = [name for name, gate in gates.items() if not gate.get("pass")]
    return {
        "gates": gates,
        "overall": {"pass": not failed, "failed": failed},
        "per_episode": per_episode_rows,
        "totals": {
            "episodes": len(per_episode),
            "decisions": sum(len(features) for features in per_episode.values()),
            "deaths": deaths,
            "death_samples": death_samples,
            "region_decisions": region_decisions,
            "safe_ok": safe_ok,
        },
    }


def probe(out_dir: Path) -> dict:
    episodes = read_episodes(out_dir)
    per_episode: dict[str, list[DecisionFeature]] = {}
    for episode in episodes:
        name = episode.get("episode")
        records, bullets = _load_episode(out_dir, name)
        per_episode[name] = extract_features(records, bullets)
    result = gate_report(per_episode)
    result.update(
        {
            "schema": "v350-region-probe/1",
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": str(out_dir),
            "config": {
                "cell_px": CELL,
                "grid": [GRID_H, GRID_W],
                "times": list(TIMES),
                "min_region_cells": MIN_REGION_CELLS,
                "safe_min_cells": SAFE_MIN_CELLS,
                "death_prefix_decisions": DEATH_PREFIX_DECISIONS,
            },
        }
    )
    return result


def render_markdown(report: dict) -> str:
    lines = [
        "# v3.5.0 S1 区域探针结果",
        "",
        f"- 生成：{report.get('generated')}",
        f"- 数据：`{report.get('source')}`",
        f"- 总判定：**{'通过' if report['overall']['pass'] else '未通过'}**",
        "",
        "## 门禁",
        "",
        "| 编号 | 判据 | 阈值 | 实测 | 判定 |",
        "|---|---|---:|---:|---|",
    ]
    for name, gate in report["gates"].items():
        value = gate.get("value")
        shown = "not_tested" if value is None else f"{value:.4f}"
        lines.append(f"| {name} | {gate['name']} | {gate['threshold']:.2f} | {shown} | {'通过' if gate['pass'] else '未过'} |")
    lines += ["", "## 逐局", "", "| episode | 决策 | deaths | 死亡前样本 | 区域内样本 | 提前 20 帧 |", "|---|---:|---:|---:|---:|---:|"]
    for row in report["per_episode"]:
        lines.append(
            f"| {row['episode']} | {row['decisions']} | {row['deaths']} | "
            f"{row['death_samples']} | {row['death_region_samples']} | {row['early']} |"
        )
    lines += ["", "## 诊断", ""]
    s14 = report["gates"]["S1-4"]
    lines.append(f"- region AUC = {s14.get('region_auc')}, clearance AUC = {s14.get('clearance_auc')}, "
                 f"increment = {s14.get('value')}, permutation p = {s14.get('permutation_p')}")
    lines.append("- S1 只做离线分析，不改动作；未通过则 S2 不启动。")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m touhou_rl.region_probe")
    parser.add_argument("--probe", type=Path, required=True, help="S0 output directory (trace/ + bullets/ + episodes.jsonl)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = probe(args.probe)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "region-probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "region-probe.md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"overall": result["overall"], "gates": {k: v.get("value") for k, v in result["gates"].items()}},
                     ensure_ascii=False, indent=2))
    return 0 if result["overall"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
