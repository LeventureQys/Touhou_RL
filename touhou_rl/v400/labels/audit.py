"""Validity-audit table for the P2 label dataset (12-P2 §2.2, P2-WP2 prerequisite).

The audit is deliberately a *skeleton with real counts*: it enumerates every
label class the dataset is supposed to cover, counts positives / negatives /
missing, and, for each class that fails its threshold, states how many more
frames of the *observed* distribution would be needed.  If the class was never
observed, it says so instead of printing an infinite number — more of the same
data cannot fix a class the collection never produces.

Thresholds are the ones 12-P2 §2.2 fixes (each event >= 500 positives; each
lives/bombs value >= 1,000 frames) plus, for the two label families §2.2 does
not name, a symmetric 500/500 coverage floor proposed here and recorded as such.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

#: 12-P2 §2.2: each event class needs at least this many positives.
EVENT_POSITIVE_THRESHOLD = 500

#: 12-P2 §2.2: each value 0..8 of lives/bombs needs at least this many frames.
HUD_VALUE_FRAME_THRESHOLD = 1000

#: 12-P2 §2.2 asks for a 50-frame overlay review.
OVERLAY_FRAME_THRESHOLD = 50

#: Proposed (not spec-fixed) coverage floors for the P2-specific labels.
PROPOSED_CLASS_THRESHOLD = 500

#: Minimum dataset size for this slice.
MIN_FRAMES = 1000

THRESHOLDS = {
    "frames_total": MIN_FRAMES,
    "event_positives": EVENT_POSITIVE_THRESHOLD,
    "hud_value_frames": HUD_VALUE_FRAME_THRESHOLD,
    "overlay_frames": OVERLAY_FRAME_THRESHOLD,
    "future_danger_positive": PROPOSED_CLASS_THRESHOLD,
    "future_danger_negative": PROPOSED_CLASS_THRESHOLD,
    "occupancy_nonempty": PROPOSED_CLASS_THRESHOLD,
    "occupancy_empty": PROPOSED_CLASS_THRESHOLD,
    "note": (
        "event/hud thresholds are from 12-P2 §2.2; the future_danger and "
        "occupancy floors are proposed here so the skeleton has a fail-closed "
        "criterion, and are marked as proposed in the report"
    ),
}


def load_records(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _counter() -> dict[str, int]:
    return {}


def _bump(counter: dict[str, int], key: Any, amount: int = 1) -> None:
    text = str(key)
    counter[text] = counter.get(text, 0) + amount


def _needed(threshold: int, positives: int, frames: int) -> dict[str, Any]:
    """How many more frames the observed rate needs to reach ``threshold``."""

    if positives >= threshold:
        return {"deficit": 0, "additional_frames": 0, "rate": positives / frames if frames else 0.0}
    rate = (positives / frames) if frames else 0.0
    if rate <= 0.0:
        return {
            "deficit": threshold - positives,
            "additional_frames": None,
            "rate": 0.0,
            "blocker": "class never observed; more frames of the same distribution cannot produce it",
        }
    return {
        "deficit": threshold - positives,
        "additional_frames": int(math.ceil((threshold - positives) / rate)),
        "rate": rate,
    }


def audit(records: Sequence[dict[str, Any]], *, frames_dir: str | Path | None = None) -> dict[str, Any]:
    """Compute the audit table from the JSONL records."""

    total = len(records)
    runs: dict[str, int] = {}
    sources: dict[str, int] = {}
    stages: dict[str, int] = {}
    policies: dict[str, int] = {}

    player_complete = 0
    player_missing_fields: dict[str, int] = {}
    hurtbox_missing = 0

    hud_complete = 0
    hud_missing_fields: dict[str, int] = {}
    lives_hist: dict[str, int] = {}
    bombs_hist: dict[str, int] = {}
    power_hist: dict[str, int] = {}
    score_present = 0

    occ_nonempty = 0
    occ_empty = 0
    occ_bullet_nonempty = 0
    occ_frames_missing = 0
    occupied_cells_hist: dict[str, int] = {}
    nearest_distance: list[float] = []

    danger_applicable = 0
    danger_true = 0
    danger_false = 0
    danger_null = 0
    danger_reasons: dict[str, int] = {}
    danger_min_gap: list[float] = []

    death_true = 0
    death_false = 0
    death_null = 0
    event_source_counts: dict[str, int] = {}

    frame_continuity: dict[str, Any] = {}
    pixel_files_missing = 0

    for record in records:
        run = str(record.get("run"))
        _bump(runs, run)
        _bump(sources, record.get("policy", "?"))
        _bump(stages, record.get("stage", "?"))
        _bump(policies, record.get("policy", "?"))

        player = record.get("player") or {}
        if player.get("complete"):
            player_complete += 1
        for name in player.get("missing") or []:
            _bump(player_missing_fields, name)
        if not player.get("hurtbox"):
            hurtbox_missing += 1

        hud = record.get("hud") or {}
        if hud.get("complete"):
            hud_complete += 1
        for name in hud.get("missing") or []:
            _bump(hud_missing_fields, name)
        if hud.get("score") is not None:
            score_present += 1
        for key, hist in (
            ("lives", lives_hist),
            ("bombs", bombs_hist),
            ("power", power_hist),
        ):
            value = hud.get(key)
            _bump(hist, "missing" if value is None else value)

        occ = record.get("occupancy") or {}
        if not occ:
            occ_frames_missing += 1
        else:
            cells = int(occ.get("occupied_cells", 0))
            bullets = int(occ.get("bullet_cells", 0))
            if cells > 0:
                occ_nonempty += 1
            else:
                occ_empty += 1
            if bullets > 0:
                occ_bullet_nonempty += 1
            _bump(occupied_cells_hist, cells if cells < 50 else "50+")
            if occ.get("nearest_distance") is not None:
                nearest_distance.append(float(occ["nearest_distance"]))

        danger = record.get("future_danger") or {}
        if danger.get("applicable"):
            danger_applicable += 1
        value = danger.get("would_die")
        if value is True:
            danger_true += 1
        elif value is False:
            danger_false += 1
        else:
            danger_null += 1
        _bump(danger_reasons, danger.get("reason") or "ok")
        if danger.get("min_gap") is not None and danger.get("would_die") is not None:
            danger_min_gap.append(float(danger["min_gap"]))

        events = record.get("events") or {}
        death = (events.get("death") or {})
        _bump(event_source_counts, f"death:{death.get('event_source')}")
        if death.get("value") is True:
            death_true += 1
        elif death.get("value") is False:
            death_false += 1
        else:
            death_null += 1

        if frames_dir is not None:
            pixels = record.get("pixels") or {}
            name = pixels.get("file")
            if not name or not (Path(frames_dir) / name).is_file():
                pixel_files_missing += 1

    for run in runs:
        numbers = [int(record["frame"]) for record in records if str(record.get("run")) == run]
        gaps = [b - a for a, b in zip(numbers, numbers[1:])]
        frame_continuity[run] = {
            "frames": len(numbers),
            "first_frame": numbers[0] if numbers else None,
            "last_frame": numbers[-1] if numbers else None,
            "gaps_not_one": sum(1 for gap in gaps if gap != 1),
        }

    def sorted_hist(hist: dict[str, int]) -> dict[str, int]:
        def key(item: str):
            try:
                return (0, int(item))
            except ValueError:
                return (1, item)

        return {k: hist[k] for k in sorted(hist, key=key)}

    events_section: dict[str, Any] = {}
    for name in ("death", "hit", "bomb", "stage_clear", "game_over", "extend"):
        positives = death_true if name == "death" else 0
        negatives = death_false if name == "death" else 0
        source = "proxy" if name == "death" else "unavailable"
        entry = {
            "event_source": source,
            "positives": positives,
            "negatives": negatives,
            "nulls": death_null if name == "death" else total,
            "missing_rate": (death_null / total) if (name == "death" and total) else 1.0,
            "threshold": EVENT_POSITIVE_THRESHOLD,
            "passes": positives >= EVENT_POSITIVE_THRESHOLD,
        }
        if name == "death":
            entry["coverage"] = _needed(EVENT_POSITIVE_THRESHOLD, positives, total)
        else:
            entry["blocker"] = (
                "EVENTS/frameEventFlags not observable in this build; no label can "
                "be generated and no amount of extra frames fixes it"
            )
            entry["coverage"] = {
                "deficit": EVENT_POSITIVE_THRESHOLD,
                "additional_frames": None,
                "blocker": "channel missing (needs the EVENTS export, not more data)",
            }
        events_section[name] = entry

    hud_values = {
        "lives": {
            "distribution": sorted_hist(lives_hist),
            "threshold_per_value": HUD_VALUE_FRAME_THRESHOLD,
            "values_below_threshold": [
                value
                for value in "012345678"
                if lives_hist.get(value, 0) < HUD_VALUE_FRAME_THRESHOLD
            ],
        },
        "bombs": {
            "distribution": sorted_hist(bombs_hist),
            "threshold_per_value": HUD_VALUE_FRAME_THRESHOLD,
            "values_below_threshold": [
                value
                for value in "012345678"
                if bombs_hist.get(value, 0) < HUD_VALUE_FRAME_THRESHOLD
            ],
        },
        "power": {
            "distinct_values": len([key for key in power_hist if key != "missing"]),
            "distribution_head": dict(list(sorted_hist(power_hist).items())[:10]),
        },
    }

    danger_section = {
        "applicable": danger_applicable,
        "positive": danger_true,
        "negative": danger_false,
        "null": danger_null,
        "missing_rate": (danger_null / total) if total else None,
        "reasons": dict(sorted(danger_reasons.items(), key=lambda kv: -kv[1])),
        "min_gap_px": _quantiles(danger_min_gap),
        "threshold_positive": PROPOSED_CLASS_THRESHOLD,
        "threshold_negative": PROPOSED_CLASS_THRESHOLD,
        "coverage_positive": _needed(PROPOSED_CLASS_THRESHOLD, danger_true, total),
        "coverage_negative": _needed(PROPOSED_CLASS_THRESHOLD, danger_false, total),
    }

    occupancy_section = {
        "frames_with_label": total - occ_frames_missing,
        "nonempty": occ_nonempty,
        "empty": occ_empty,
        "bullet_nonempty": occ_bullet_nonempty,
        "missing": occ_frames_missing,
        "missing_rate": (occ_frames_missing / total) if total else None,
        "occupied_cells_hist": sorted_hist(occupied_cells_hist),
        "nearest_distance_px": _quantiles(nearest_distance),
        "threshold_nonempty": PROPOSED_CLASS_THRESHOLD,
        "threshold_empty": PROPOSED_CLASS_THRESHOLD,
        "coverage_nonempty": _needed(PROPOSED_CLASS_THRESHOLD, occ_nonempty, total),
        "coverage_empty": _needed(PROPOSED_CLASS_THRESHOLD, occ_empty, total),
    }

    player_section = {
        "complete": player_complete,
        "complete_rate": (player_complete / total) if total else None,
        "hurtbox_missing": hurtbox_missing,
        "missing_fields": dict(player_missing_fields),
    }

    hud_section = {
        "complete": hud_complete,
        "complete_rate": (hud_complete / total) if total else None,
        "missing_fields": dict(hud_missing_fields),
        "score_present": score_present,
        "unavailable_by_export": {
            "display_score": "not exported (Gui.cpp renders globals.displayScore)",
            "point_total": "nextPointItemExtendThreshold not exported",
            "time_total": "lastSpellTimeOrbThreshold not exported",
        },
        "values": hud_values,
    }

    conclusions: list[dict[str, Any]] = []
    if total < MIN_FRAMES:
        conclusions.append(
            {
                "class": "frames_total",
                "passes": False,
                "deficit": MIN_FRAMES - total,
                "additional_frames": MIN_FRAMES - total,
                "action": "collect more frames",
            }
        )
    else:
        conclusions.append({"class": "frames_total", "passes": True, "deficit": 0, "additional_frames": 0})

    for name, entry in events_section.items():
        if entry["passes"]:
            conclusions.append({"class": f"event:{name}", "passes": True, "deficit": 0, "additional_frames": 0})
        else:
            coverage = entry["coverage"]
            conclusions.append(
                {
                    "class": f"event:{name}",
                    "passes": False,
                    "event_source": entry["event_source"],
                    "deficit": coverage.get("deficit"),
                    "additional_frames": coverage.get("additional_frames"),
                    "blocker": coverage.get("blocker", ""),
                    "action": (
                        "add EVENTS export (code change) — extra frames cannot fix it"
                        if entry["event_source"] == "unavailable"
                        else "collect more frames with deaths"
                    ),
                }
            )

    for field in ("lives", "bombs"):
        missing_values = hud_values[field]["values_below_threshold"]
        if not missing_values:
            conclusions.append({"class": f"hud_values:{field}", "passes": True, "deficit": 0, "additional_frames": 0})
        else:
            # Extra frames needed for the *rarest* missing value to reach 1,000.
            needs = []
            hist = lives_hist if field == "lives" else bombs_hist
            never_observed = [
                value for value in missing_values if hist.get(value, 0) == 0
            ]
            for value in missing_values:
                needs.append(_needed(HUD_VALUE_FRAME_THRESHOLD, hist.get(value, 0), total))
            bounded = [entry["additional_frames"] for entry in needs if entry["additional_frames"] is not None]
            conclusions.append(
                {
                    "class": f"hud_values:{field}",
                    "passes": False,
                    "deficit": needs[0]["deficit"],
                    "additional_frames": None if never_observed else (max(bounded) if bounded else None),
                    "blocker": (
                        f"values {never_observed} never observed; more of the same "
                        "stage/policy mix cannot produce them"
                        if never_observed
                        else ""
                    ),
                    "action": f"collect frames where {field} takes values {missing_values}",
                }
            )

    for name, value, threshold in (
        ("future_danger:positive", danger_true, PROPOSED_CLASS_THRESHOLD),
        ("future_danger:negative", danger_false, PROPOSED_CLASS_THRESHOLD),
        ("occupancy:nonempty", occ_nonempty, PROPOSED_CLASS_THRESHOLD),
        ("occupancy:empty", occ_empty, PROPOSED_CLASS_THRESHOLD),
    ):
        coverage = _needed(threshold, value, total)
        conclusions.append(
            {
                "class": name,
                "passes": value >= threshold,
                "deficit": coverage["deficit"],
                "additional_frames": coverage["additional_frames"],
                "blocker": coverage.get("blocker", ""),
                "threshold": threshold,
                "threshold_proposed": True,
            }
        )

    return {
        "schema": "v400-p2-audit-1",
        "dataset": {
            "frames": total,
            "runs": runs,
            "policy_frames": sources,
            "stage_frames": stages,
            "frame_continuity": frame_continuity,
            "pixel_files_missing": pixel_files_missing if frames_dir is not None else None,
            "label_versions": sorted({
                str((record.get("meta") or {}).get("label_version")) for record in records
            }),
        },
        "player": player_section,
        "hud": hud_section,
        "occupancy": occupancy_section,
        "future_danger": danger_section,
        "events": {"classes": events_section, "event_source_counts": dict(event_source_counts)},
        "thresholds": THRESHOLDS,
        "conclusions": conclusions,
        "passes": all(conclusion.get("passes") for conclusion in conclusions),
    }


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"n": 0, "min": None, "p50": None, "p95": None, "max": None}
    def at(frac: float) -> float:
        return ordered[min(len(ordered) - 1, int(len(ordered) * frac))]
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p50": at(0.5),
        "p95": at(0.95),
        "max": ordered[-1],
    }


def hud_lag(
    records: Sequence[dict[str, Any]],
    frames_dir: str | Path,
    *,
    fields: Sequence[str] = ("lives", "bombs", "power", "graze", "point", "time"),
    max_lag: int = 8,
) -> dict[str, Any]:
    """Measure how many frames the rendered HUD row lags the state (P2 §3.2).

    Pixels are used only as a *change detector*: a row's band digest is compared
    with the previous frame's, and a state change is matched to the first band
    change within ``max_lag``.  No value is read out of the pixels, so this
    cannot contaminate the labels.  Frames where two measured fields change at
    once are skipped (a band change cannot be attributed).  Rows whose band
    changed on every frame (animation) are reported as unattributable.
    """

    from PIL import Image  # local import: only the audit needs pixels

    from touhou_rl.v400.labels.spec import HUD_LAYOUT

    paths = [Path(frames_dir) / record["pixels"]["file"] for record in records]
    digests: list[dict[str, str]] = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        row: dict[str, str] = {}
        for field in fields:
            x0, y0, x1, y1 = HUD_LAYOUT[field]["band"]
            row[field] = hashlib.sha1(array[y0:y1, x0:x1].tobytes()).hexdigest()
        digests.append(row)

    changed: dict[str, list[bool]] = {field: [] for field in fields}
    for index in range(1, len(digests)):
        for field in fields:
            changed[field].append(digests[index][field] != digests[index - 1][field])

    lag_hist: dict[str, dict[str, int]] = {}
    detail: dict[str, Any] = {}
    for field in fields:
        hist: dict[str, int] = {}
        matches = 0
        unresolved = 0
        animated = sum(changed[field])
        state_changes = 0
        for index in range(1, len(records)):
            current = (records[index].get("hud") or {}).get(field)
            previous = (records[index - 1].get("hud") or {}).get(field)
            if current is None or previous is None or current == previous:
                continue
            # Skip frames where another measured field also changed: the band
            # change could belong to the other field.
            others = [
                other
                for other in fields
                if other != field
                and (records[index].get("hud") or {}).get(other)
                != (records[index - 1].get("hud") or {}).get(other)
            ]
            if others:
                continue
            state_changes += 1
            found = None
            for offset in range(0, max_lag + 1):
                step = index - 1 + offset
                if 0 <= step < len(changed[field]) and changed[field][step]:
                    found = offset
                    break
            if found is None:
                unresolved += 1
            else:
                matches += 1
                hist[str(found)] = hist.get(str(found), 0) + 1
        lag_hist[field] = hist
        detail[field] = {
            "state_changes": state_changes,
            "matched": matches,
            "unresolved": unresolved,
            "band_changed_frames": animated,
            "band_animation_rate": (animated / max(1, len(records) - 1)),
            "lag_histogram": hist,
        }
    return {
        "fields": list(fields),
        "max_lag": max_lag,
        "per_field": detail,
        "note": (
            "pixels are used as a change detector only; a row that animates on "
            "every frame is reported through band_animation_rate instead of a "
            "fabricated lag"
        ),
    }


def to_markdown(report: dict[str, Any]) -> str:
    """Render the audit as the markdown table P2-WP2 will extend."""

    lines: list[str] = []
    dataset = report["dataset"]
    lines.append("### 审计表骨架（真实计数）")
    lines.append("")
    lines.append(
        f"- 帧数 **{dataset['frames']}**；策略分布 {dataset['policy_frames']}；"
        f"面分布 {dataset['stage_frames']}；label_version={dataset['label_versions']}"
    )
    lines.append("")
    lines.append("| 标签类 | 正例 | 负例 | 缺失率 | 阈值 | 达标 | 缺口/需补采 |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, entry in report["events"]["classes"].items():
        lines.append(
            f"| event:{name}（{entry['event_source']}） | {entry['positives']} | "
            f"{entry['negatives']} | {entry['missing_rate']:.3f} | "
            f"{entry['threshold']} | {'是' if entry['passes'] else '否'} | "
            f"{_fmt_need(entry['coverage'])} |"
        )
    danger = report["future_danger"]
    lines.append(
        f"| future_danger:positive | {danger['positive']} | {danger['negative']} | "
        f"{danger['missing_rate']:.3f} | {danger['threshold_positive']}（提议） | "
        f"{'是' if danger['positive'] >= danger['threshold_positive'] else '否'} | "
        f"{_fmt_need(danger['coverage_positive'])} |"
    )
    lines.append(
        f"| future_danger:negative | {danger['negative']} | {danger['positive']} | "
        f"{danger['missing_rate']:.3f} | {danger['threshold_negative']}（提议） | "
        f"{'是' if danger['negative'] >= danger['threshold_negative'] else '否'} | "
        f"{_fmt_need(danger['coverage_negative'])} |"
    )
    occupancy = report["occupancy"]
    lines.append(
        f"| occupancy:nonempty | {occupancy['nonempty']} | {occupancy['empty']} | "
        f"{occupancy['missing_rate']:.3f} | {occupancy['threshold_nonempty']}（提议） | "
        f"{'是' if occupancy['nonempty'] >= occupancy['threshold_nonempty'] else '否'} | "
        f"{_fmt_need(occupancy['coverage_nonempty'])} |"
    )
    lines.append(
        f"| occupancy:empty | {occupancy['empty']} | {occupancy['nonempty']} | "
        f"{occupancy['missing_rate']:.3f} | {occupancy['threshold_empty']}（提议） | "
        f"{'是' if occupancy['empty'] >= occupancy['threshold_empty'] else '否'} | "
        f"{_fmt_need(occupancy['coverage_empty'])} |"
    )
    lines.append("")
    lines.append("| HUD 标量 | 分布 / 缺失 | 0–8 每个取值 ≥1000 帧 |")
    lines.append("|---|---|---|")
    for field in ("lives", "bombs"):
        values = report["hud"]["values"][field]
        lines.append(
            f"| {field} | {values['distribution']} | "
            f"{'达标' if not values['values_below_threshold'] else '缺 ' + ','.join(values['values_below_threshold'])} |"
        )
    lines.append(
        f"| power | {report['hud']['values']['power']['distinct_values']} 个不同取值 | — |"
    )
    lines.append("")
    lines.append("**fail-closed 结论（缺什么 → 要补多少帧）**")
    lines.append("")
    for conclusion in report["conclusions"]:
        status = "达标" if conclusion["passes"] else "**未达标**"
        need = conclusion.get("additional_frames")
        blocker = conclusion.get("blocker") or ""
        lines.append(
            f"- `{conclusion['class']}`：{status}；"
            + ("无需补采" if need == 0 else (
                f"按当前出现率还需 **{need}** 帧" if need is not None else f"**不可由补采解决**（{blocker}）"
            ))
        )
    return "\n".join(lines) + "\n"


def _fmt_need(coverage: dict[str, Any]) -> str:
    if coverage.get("deficit", 0) == 0:
        return "达标"
    need = coverage.get("additional_frames")
    if need is None:
        return "不可补采（通道缺失）"
    return f"+{need} 帧"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# v2: real EVENTS bits, EVENTS-vs-proxy consistency, G2 data-volume gap
# ---------------------------------------------------------------------------
#: Event classes of the v2 label set, in report order.
V2_EVENT_CLASSES: tuple[str, ...] = (
    "hit",
    "player_dying_started",
    "effect_spawned",
    "death",
    "bomb",
    "game_over",
    "extend",
    "stage_clear",
)

#: 12-P2 §6 G2 detection gates, restated as *data-volume* requirements only.
G2_TARGETS = {
    "event_rate": 0.99,          # death / bomb / stage_clear recall & precision
    "hud_frame_accuracy": 0.995,  # lives / bombs per-frame accuracy
    "score_sign_rate": 0.99,      # delta-score sign agreement
    "latency_frames": 10,         # median death-detection latency
}


def zero_error_sample_size(target_rate: float, alpha: float = 0.05) -> int:
    """Positives needed so that *zero misses* still bounds the rate at ``alpha``.

    With ``n`` independent positives and no error, the one-sided
    ``1 - alpha`` Clopper-Pearson lower bound on the success rate is
    ``alpha ** (1/n)``; requiring that to reach ``target_rate`` gives
    ``n >= ln(alpha) / ln(target_rate)``.  For 99 % / 95 % one-sided this is 299
    (the familiar "rule of three" rounded up); for 99.5 % it is 598.
    """

    if not 0.0 < target_rate < 1.0:
        raise ValueError(f"target_rate must be in (0,1), got {target_rate!r}")
    return int(math.ceil(math.log(alpha) / math.log(target_rate)))


def event_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count every v2 event class: positives / negatives / nulls / sources."""

    summary: dict[str, Any] = {}
    for name in V2_EVENT_CLASSES:
        positives = negatives = nulls = 0
        sources: dict[str, int] = {}
        for record in records:
            entry = (record.get("events") or {}).get(name) or {}
            value = entry.get("value")
            if value is True:
                positives += 1
            elif value is False:
                negatives += 1
            else:
                nulls += 1
            sources[str(entry.get("event_source"))] = (
                sources.get(str(entry.get("event_source")), 0) + 1
            )
        total = len(records)
        summary[name] = {
            "positives": positives,
            "negatives": negatives,
            "nulls": nulls,
            "events_per_frame": (positives / total) if total else None,
            "event_sources": dict(sorted(sources.items())),
            "threshold": EVENT_POSITIVE_THRESHOLD,
            "passes": positives >= EVENT_POSITIVE_THRESHOLD,
        }
    return summary


def event_consistency(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compare the real ``EVENTS`` bits with the state-derived proxies.

    All comparisons are on the *same frames* of the same dataset, which is the
    only way the two channels can be compared: a hit count from a different run
    proves nothing about a proxy count.  ``hit`` vs ``ALIVE -> DYING`` is not
    expected to be equal — a hit may be absorbed by a deathbomb/border, and the
    recorded run may end before the DYING frame — so the report gives the full
    2x2 table plus the mismatching frames instead of a single "agreement rate".
    """

    def flag(record: dict[str, Any], path: Sequence[str]) -> bool | None:
        node: Any = record.get("events") or {}
        for key in path:
            if not isinstance(node, dict):
                return None
            node = node.get(key)
        return node if isinstance(node, bool) else None

    cells = {
        "hit_true": 0,
        "hit_false": 0,
        "hit_null": 0,
        "alive_to_dying_true": 0,
        "alive_to_dying_false": 0,
        "alive_to_dying_null": 0,
        "hit_and_edge": 0,
        "hit_only": 0,
        "edge_only": 0,
        "neither": 0,
        "death_bit_true": 0,
        "dying_started_bit_true": 0,
        "death_bit_and_dying_started": 0,
        "death_bit_only": 0,
        "dying_started_only": 0,
        "death_proxy_true": 0,
        "death_bit_and_proxy": 0,
        "death_bit_only_vs_proxy": 0,
        "death_proxy_only_vs_bit": 0,
        "bomb_bit_true": 0,
        "bomb_proxy_true": 0,
        "bomb_bit_and_proxy": 0,
        "bomb_bit_only_vs_proxy": 0,
        "bomb_proxy_only_vs_bit": 0,
        "bomb_proxy_ambiguous": 0,
        "frames": len(records),
    }
    edge_only_frames: list[dict[str, Any]] = []
    bomb_bit_only_frames: list[dict[str, Any]] = []
    hit_only_frames: list[dict[str, Any]] = []
    for record in records:
        hit = flag(record, ("hit", "value"))
        edge = flag(record, ("death", "proxy", "components", "alive_to_dying"))
        death_bit = flag(record, ("death", "value"))
        dying = flag(record, ("player_dying_started", "value"))
        death_proxy = flag(record, ("death", "proxy", "value"))
        bomb_bit = flag(record, ("bomb", "value"))
        bomb_proxy = flag(record, ("bomb", "proxy", "value"))
        bomb_proxy_block = ((record.get("events") or {}).get("bomb") or {}).get("proxy") or {}
        bomb_reason = bomb_proxy_block.get("reason")

        for value, key_true, key_false, key_null in (
            (hit, "hit_true", "hit_false", "hit_null"),
            (edge, "alive_to_dying_true", "alive_to_dying_false", "alive_to_dying_null"),
        ):
            if value is True:
                cells[key_true] += 1
            elif value is False:
                cells[key_false] += 1
            else:
                cells[key_null] += 1
        if hit is not None and edge is not None:
            if hit and edge:
                cells["hit_and_edge"] += 1
            elif hit:
                cells["hit_only"] += 1
                if len(hit_only_frames) < 10:
                    hit_only_frames.append(
                        {"shard": record.get("shard"), "frame": record.get("frame")}
                    )
            elif edge:
                cells["edge_only"] += 1
                if len(edge_only_frames) < 10:
                    edge_only_frames.append(
                        {"shard": record.get("shard"), "frame": record.get("frame")}
                    )
            else:
                cells["neither"] += 1

        if death_bit is True:
            cells["death_bit_true"] += 1
        if dying is True:
            cells["dying_started_bit_true"] += 1
        if death_bit and dying:
            cells["death_bit_and_dying_started"] += 1
        elif death_bit:
            cells["death_bit_only"] += 1
        elif dying:
            cells["dying_started_only"] += 1
        if death_proxy is True:
            cells["death_proxy_true"] += 1
        if death_bit and death_proxy:
            cells["death_bit_and_proxy"] += 1
        elif death_bit:
            cells["death_bit_only_vs_proxy"] += 1
        elif death_proxy:
            cells["death_proxy_only_vs_bit"] += 1

        if bomb_bit is True:
            cells["bomb_bit_true"] += 1
        if bomb_proxy is True:
            cells["bomb_proxy_true"] += 1
        if bomb_proxy is None and bomb_reason == "ambiguous_with_death":
            cells["bomb_proxy_ambiguous"] += 1
        if bomb_bit and bomb_proxy:
            cells["bomb_bit_and_proxy"] += 1
        elif bomb_bit:
            cells["bomb_bit_only_vs_proxy"] += 1
            if len(bomb_bit_only_frames) < 10:
                bomb_bit_only_frames.append(
                    {"shard": record.get("shard"), "frame": record.get("frame")}
                )
        elif bomb_proxy:
            cells["bomb_proxy_only_vs_bit"] += 1

    def safe_div(a: int, b: int) -> float | None:
        return (a / b) if b else None

    # The ALIVE->DYING edge is defined on the pair (k, k+1): it says "the ship was
    # alive after frame k and dying after frame k+1", i.e. the transition happened
    # *during* frame k+1 — the same frame whose EVENTS word is read at k+1.  So the
    # correctly aligned comparison is hit(k) vs edge(k-1).  Both keyings are
    # reported: the unaligned one so the off-by-one is visible, the aligned one as
    # the number that would flag a broken channel.
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = (record.get("shard"), record.get("attempt"))
        groups.setdefault(key, []).append(record)
    aligned = {"both": 0, "hit_only": 0, "edge_only": 0, "neither": 0, "pairs": 0}
    aligned_edge_only: list[dict[str, Any]] = []
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda record: int(record["frame"]))
        hits = [flag(record, ("hit", "value")) for record in rows]
        edges = [
            flag(record, ("death", "proxy", "components", "alive_to_dying"))
            for record in rows
        ]
        offsets = [int(rows[i + 1]["frame"]) - int(rows[i]["frame"]) for i in range(len(rows) - 1)]
        for index in range(1, len(rows)):
            if offsets[index - 1] != 1:
                continue  # not a contiguous pair: do not align across a gap
            hit = hits[index]
            edge = edges[index - 1]
            if hit is None or edge is None:
                continue
            aligned["pairs"] += 1
            if hit and edge:
                aligned["both"] += 1
            elif hit:
                aligned["hit_only"] += 1
            elif edge:
                aligned["edge_only"] += 1
                if len(aligned_edge_only) < 10:
                    aligned_edge_only.append(
                        {"shard": rows[index].get("shard"), "frame": rows[index].get("frame")}
                    )
            else:
                aligned["neither"] += 1

    return {
        "schema": "v400-p2-event-consistency-2",
        "frames": cells["frames"],
        "events_available_frames": sum(
            1
            for record in records
            if ((record.get("events") or {}).get("channel") or {}).get("available")
        ),
        "hit_vs_alive_to_dying": {
            "cells": {
                key: cells[key]
                for key in (
                    "hit_true",
                    "hit_false",
                    "hit_null",
                    "alive_to_dying_true",
                    "alive_to_dying_false",
                    "alive_to_dying_null",
                    "hit_and_edge",
                    "hit_only",
                    "edge_only",
                    "neither",
                )
            },
            "hit_count": cells["hit_true"],
            "alive_to_dying_count": cells["alive_to_dying_true"],
            "edge_recall_by_hit": safe_div(cells["hit_and_edge"], cells["alive_to_dying_true"]),
            "hit_is_edge_rate": safe_div(cells["hit_and_edge"], cells["hit_true"]),
            "edge_only_examples": edge_only_frames,
            "hit_only_examples": hit_only_frames,
            "aligned_cells": aligned,
            "aligned_edge_recall_by_hit": safe_div(aligned["both"], aligned["both"] + aligned["edge_only"]),
            "aligned_edge_is_explained_by_hit": safe_div(
                aligned["both"], aligned["both"] + aligned["hit_only"]
            ),
            "aligned_edge_only_examples": aligned_edge_only,
            "keying": (
                "cells = hit(k) vs edge(k) (same-index comparison, reproduces the "
                "obvious-but-wrong pairing); aligned_cells = hit(k) vs edge(k-1), "
                "because edge(k) means the transition happened during frame k+1 and "
                "the EVENTS word read at k+1 is that frame's word"
            ),
            "note": (
                "hit (EVENTS 0x2) and the ALIVE->DYING edge are different events on "
                "purpose: a hit absorbed by a deathbomb/border has no DYING edge.  "
                "hit_is_edge_rate < 1 is therefore expected, not a channel error; "
                "aligned_edge_recall_by_hit is the number that would flag a real "
                "mismatch"
            ),
        },
        "death_bit_vs_dying_started": {
            "death_bit_true": cells["death_bit_true"],
            "dying_started_true": cells["dying_started_bit_true"],
            "both": cells["death_bit_and_dying_started"],
            "death_only": cells["death_bit_only"],
            "dying_started_only": cells["dying_started_only"],
            "note": (
                "0x4 is committed death, 0x200 is the start of the dying animation; "
                "the commit is expected to follow the start by one or more frames"
            ),
        },
        "death_bit_vs_proxy": {
            "death_bit_true": cells["death_bit_true"],
            "death_proxy_true": cells["death_proxy_true"],
            "both": cells["death_bit_and_proxy"],
            "bit_only": cells["death_bit_only_vs_proxy"],
            "proxy_only": cells["death_proxy_only_vs_bit"],
        },
        "bomb_bit_vs_proxy": {
            "bomb_bit_true": cells["bomb_bit_true"],
            "bomb_proxy_true": cells["bomb_proxy_true"],
            "both": cells["bomb_bit_and_proxy"],
            "bit_only": cells["bomb_bit_only_vs_proxy"],
            "proxy_only": cells["bomb_proxy_only_vs_bit"],
            "proxy_ambiguous_with_death": cells["bomb_proxy_ambiguous"],
            "bit_only_examples": bomb_bit_only_frames,
            "note": (
                "BOMB_STARTED (0x1) also fires for a deathbomb, so bit_only is "
                "expected on death frames; the proxy refuses to call a bomb there "
                "(ambiguous_with_death) instead of guessing"
            ),
        },
    }


def g2_gap_estimate(
    records: Sequence[dict[str, Any]],
    events: dict[str, Any],
    *,
    consistency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Frame/positive counts still missing before G2 could even be *measured*.

    This is deliberately **not** a detector score.  It only answers "how much
    data would make the G2 numbers statistically meaningful": with zero errors
    the one-sided 95 % lower bound reaches 99 % at 299 positives and 99.5 % at
    598 frames, so anything below that cannot support the gate no matter how
    good the model is.
    """

    total = len(records)
    n99 = zero_error_sample_size(G2_TARGETS["event_rate"])
    n995 = zero_error_sample_size(G2_TARGETS["hud_frame_accuracy"])
    n99sign = zero_error_sample_size(G2_TARGETS["score_sign_rate"])

    def deficit(needed: int, have: int, rate: float | None) -> dict[str, Any]:
        gap = max(0, needed - have)
        if gap == 0:
            return {
                "needed": needed,
                "have": have,
                "deficit": 0,
                "additional_frames": 0,
                "status": "enough_to_claim",
            }
        if not rate:
            return {
                "needed": needed,
                "have": have,
                "deficit": gap,
                "additional_frames": None,
                "status": "class_not_observed",
                "blocker": (
                    "the class never occurred in this collection; more frames of the "
                    "same distribution cannot produce it"
                ),
            }
        frames_total_needed = int(math.ceil(needed / rate))
        return {
            "needed": needed,
            "have": have,
            "deficit": gap,
            "rate": rate,
            "additional_frames": max(0, frames_total_needed - total),
            "frames_total_needed": frames_total_needed,
            "status": "short",
        }

    def rate_of(name: str) -> float | None:
        value = (events.get(name) or {}).get("events_per_frame")
        return float(value) if value else None

    # HUD value coverage (12-P2 §2.2: every lives/bombs value needs >= 1000 frames)
    lives_hist: dict[str, int] = {}
    bombs_hist: dict[str, int] = {}
    hud_missing = 0
    score_delta_frames = 0
    score_drops = 0
    score_rises = 0
    previous: dict[str, Any] = {}
    for record in records:
        hud = record.get("hud") or {}
        if not hud.get("complete"):
            hud_missing += 1
        for key, hist in (("lives", lives_hist), ("bombs", bombs_hist)):
            value = hud.get(key)
            _bump(hist, "missing" if value is None else value)
        score = hud.get("score")
        shard = record.get("shard")
        if score is not None and shard in previous and previous[shard] is not None:
            delta = int(score) - int(previous[shard])
            if delta != 0:
                score_delta_frames += 1
                if delta < 0:
                    score_drops += 1
                else:
                    score_rises += 1
        if shard is not None:
            previous[shard] = score

    hud_value_gaps = {}
    for field, hist in (("lives", lives_hist), ("bombs", bombs_hist)):
        per_value = {}
        for value in "012345678":
            have = int(hist.get(value, 0))
            deficit_frames = max(0, HUD_VALUE_FRAME_THRESHOLD - have)
            if deficit_frames == 0:
                additional = 0
            elif have == 0 or not total:
                additional = None
            else:
                additional = max(
                    0,
                    int(math.ceil(deficit_frames * total / have)) - total,
                )
            per_value[value] = {
                "frames": have,
                "deficit": deficit_frames,
                "additional_frames": additional,
                "observed": have > 0,
            }
        hud_value_gaps[field] = per_value

    score_rate = (score_delta_frames / total) if total else None
    latency_floor = 30  # proposed: a median needs a handful of events, not one
    deaths_with_pixels = sum(
        1
        for record in records
        if (record.get("events") or {}).get("death", {}).get("value") is True
        and (record.get("pixels") or {}).get("file")
    )

    return {
        "schema": "v400-p2-g2-data-gap-2",
        "note": (
            "DATA-VOLUME gap only. G2 is a detector gate (WP3/WP5); nothing here "
            "claims a recall/precision/latency score. The counts are the smallest "
            "number of positives on which a zero-error run could support the stated "
            "rate at a one-sided 95% confidence bound."
        ),
        "alpha": 0.05,
        "targets": G2_TARGETS,
        "required_positives": {
            "event_rate_99": n99,
            "hud_frame_accuracy_995": n995,
            "score_sign_99": n99sign,
            "death_latency_median_events": latency_floor,
            "hud_value_frames_12_P2_2_2": HUD_VALUE_FRAME_THRESHOLD,
        },
        "frames": total,
        "events": {
            "death": deficit(n99, events["death"]["positives"], rate_of("death")),
            "bomb": deficit(n99, events["bomb"]["positives"], rate_of("bomb")),
            "player_dying_started": deficit(
                n99, events["player_dying_started"]["positives"], rate_of("player_dying_started")
            ),
            "hit": deficit(n99, events["hit"]["positives"], rate_of("hit")),
            "bonus_effect_spawned": deficit(
                n99, events["effect_spawned"]["positives"], rate_of("effect_spawned")
            ),
            "game_over": deficit(n99, events["game_over"]["positives"], rate_of("game_over")),
            "extend": deficit(n99, events["extend"]["positives"], rate_of("extend")),
            "stage_clear": deficit(
                n99, events["stage_clear"]["positives"], rate_of("stage_clear")
            ),
        },
        "hud": {
            "frames_with_complete_label": total - hud_missing,
            "accuracy_frames_needed": n995,
            "accuracy_deficit": max(0, n995 - (total - hud_missing)),
            "values": hud_value_gaps,
            "score_delta": {
                "frames_with_nonzero_delta": score_delta_frames,
                "rises": score_rises,
                "drops": score_drops,
                "needed": n99sign,
                "deficit": max(0, n99sign - score_delta_frames),
                "additional_frames": (
                    None
                    if score_delta_frames == 0
                    else max(
                        0,
                        (int(math.ceil(n99sign / score_rate)) if score_rate else 0) - total,
                    )
                ),
            },
        },
        "death_latency": {
            "death_events_with_pixel": deaths_with_pixels,
            "proposed_median_floor": latency_floor,
            "deficit": max(0, latency_floor - deaths_with_pixels),
            "blocker": (
                "a median latency needs several death events; the HUD-row latency "
                "measured in hud_lag is a separate, per-row quantity"
            ),
        },
        "consistency_available": bool(consistency and consistency.get("events_available_frames")),
    }


def hud_lag_v2(
    records: Sequence[dict[str, Any]],
    frames_dir: str | Path,
    *,
    fields: Sequence[str] = ("lives", "bombs", "score", "graze", "point", "time", "power"),
    max_lag: int = 10,
    quiet_frames: int = 3,
) -> dict[str, Any]:
    """Measure the HUD's state -> picture latency per row (12-P2 §3.2).

    Pixels are only used as a *change detector*: each row's band digest is
    compared with the previous frame's, never decoded.  A state change at frame
    ``k`` (``hud[k] != hud[k-1]``) is attributed to the first band change at
    ``j`` in ``[k, k+max_lag]``, and the measured lag is ``j - k``: if the row
    redraws in the same frame, ``j == k`` and the lag is 0.  The previous
    implementation searched from ``k-1``, which could report a change that
    happened *before* the state moved; that off-by-one is fixed here and the
    convention is stated in the report.

    A row whose band also animates is not attributable, so a candidate change is
    kept only when the band was stable for ``quiet_frames`` frames immediately
    before ``k`` (a freshly rolling score is otherwise indistinguishable from a
    stale one).  Rows with no attributable change are reported ``untested`` with
    the collection condition that would test them, never with a guessed lag.
    """

    from PIL import Image

    from touhou_rl.v400.labels.spec import HUD_LAYOUT

    frames_dir = Path(frames_dir)
    # group by (shard, attempt, stage, difficulty) so a state change is never
    # paired across two different runs.
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            record.get("shard"),
            record.get("attempt"),
            record.get("stage"),
            record.get("difficulty"),
        )
        groups.setdefault(key, []).append(record)

    def pixel_path(record: dict[str, Any]) -> Path | None:
        """Resolve a label's ``pixels.file`` under the dataset root.

        The collector stores ``pixels/frame-NNNNN.png`` relative to its shard
        attempt directory, so a full path is
        ``<root>/shards/<shard>/<attempt>/pixels/...``.  A flat layout
        (``<root>/pixels/...``) is accepted too, which is what the unit test
        builds.
        """

        name = (record.get("pixels") or {}).get("file")
        if not name:
            return None
        candidates = [
            frames_dir / "shards" / str(record.get("shard")) / str(record.get("attempt")) / name,
            frames_dir / name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    per_field: dict[str, Any] = {
        field: {
            "state_changes": 0,
            "candidate_changes": 0,
            "matched": 0,
            "unresolved": 0,
            "skipped_animated_before": 0,
            "lag_histogram": {},
            "lag_values": [],
            "band_changed_frames": 0,
            "frames": 0,
        }
        for field in fields
    }

    for key, rows in groups.items():
        rows = sorted(rows, key=lambda record: int(record["frame"]))
        digests: dict[str, list[str]] = {field: [] for field in fields}
        for record in rows:
            path = pixel_path(record)
            if path is None:
                digests = None
                break
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            for field in fields:
                x0, y0, x1, y1 = HUD_LAYOUT[field]["band"]
                digests[field].append(hashlib.sha1(array[y0:y1, x0:x1].tobytes()).hexdigest())
        if digests is None:
            continue
        for field in fields:
            series = digests[field]
            changed = [False] * len(series)
            for index in range(1, len(series)):
                changed[index] = series[index] != series[index - 1]
            bucket = per_field[field]
            bucket["frames"] += len(series)
            bucket["band_changed_frames"] += sum(changed)
            for index in range(1, len(rows)):
                current = (rows[index].get("hud") or {}).get(field)
                previous = (rows[index - 1].get("hud") or {}).get(field)
                if current is None or previous is None or current == previous:
                    continue
                bucket["state_changes"] += 1
                quiet = all(
                    not changed[j] for j in range(max(1, index - quiet_frames), index)
                )
                if not quiet:
                    bucket["skipped_animated_before"] += 1
                    continue
                bucket["candidate_changes"] += 1
                found = None
                for j in range(index, min(len(series), index + max_lag + 1)):
                    if changed[j]:
                        found = j - index
                        break
                if found is None:
                    bucket["unresolved"] += 1
                    continue
                bucket["matched"] += 1
                bucket["lag_values"].append(int(found))
                bucket["lag_histogram"][str(found)] = (
                    bucket["lag_histogram"].get(str(found), 0) + 1
                )

    report: dict[str, Any] = {
        "schema": "v400-p2-hud-lag-2",
        "fields": list(fields),
        "max_lag": max_lag,
        "quiet_frames": quiet_frames,
        "alignment": (
            "lag = (first band-change frame j) - (state-change frame k), j >= k; "
            "pixel frame k and STATE frame k are the same engine frame in this "
            "collector (SHM header frame = frame - 1 because AfterDrawFrame runs "
            "before ++g_frameCount, and the SHM content is that frame's render)"
        ),
        "pixel_use": "band digest only (change detector); no value is read from pixels",
    }
    for field in fields:
        bucket = per_field[field]
        values = sorted(bucket["lag_values"])
        if values:
            def at(frac: float) -> int:
                return int(values[min(len(values) - 1, int(len(values) * frac))])

            summary = {
                "n": len(values),
                "median": at(0.5),
                "p95": at(0.95),
                "max": values[-1],
                "min": values[0],
            }
        else:
            summary = {"n": 0, "median": None, "p95": None, "max": None, "min": None}
        tested = bucket["matched"] > 0
        entry = {
            **summary,
            "tested": tested,
            "state_changes": bucket["state_changes"],
            "candidate_changes": bucket["candidate_changes"],
            "matched": bucket["matched"],
            "unresolved": bucket["unresolved"],
            "skipped_animated_before": bucket["skipped_animated_before"],
            "band_changed_frames": bucket["band_changed_frames"],
            "band_animation_rate": (
                bucket["band_changed_frames"] / max(1, bucket["frames"] - len(groups))
            ),
            "lag_histogram": dict(sorted(bucket["lag_histogram"].items(), key=lambda kv: int(kv[0]))),
            "satisfies_g2_latency": (
                None if not tested else bool(summary["median"] is not None and summary["median"] <= 10)
            ),
            "gap": (
                "no state change in this window"
                if bucket["state_changes"] == 0
                else (
                    f"{bucket['skipped_animated_before']} change(s) had an animating band "
                    "before them (unattributable)"
                    if not tested
                    else ""
                )
            ),
            "collection_condition": _hud_condition(field),
        }
        report[field] = entry
    report["note"] = (
        "rows with tested=False are explicitly untested; their G2 latency claim is "
        "not supported by this dataset"
    )
    return report


def _hud_condition(field: str) -> str:
    return {
        "lives": "at least one lost life or extend inside the recorded window",
        "bombs": "at least one bomb press (BOMB button) or death-stock change",
        "score": "a frame where the score is constant for >= 3 frames before a jump "
        "(otherwise the rolling display hides the first redraw)",
        "graze": "any graze, which a shooting policy produces",
        "point": "point items collected (top-of-screen play) ",
        "time": "time orbs (spell/boss phase with the time-orb HUD active)",
        "power": "power items collected (item drops from enemies)",
    }


def to_markdown_v2(report: dict[str, Any]) -> str:
    """Render the v2 audit report (events, consistency, HUD, G2 data gap)."""

    lines: list[str] = []
    lines.append("### 真实采集（v2）")
    lines.append("")
    lines.append(
        f"- 帧数 **{report['frames']}**；来源 {report['sources']}；面 {report['stages']}；"
        f"难度 {report['difficulties']}；分片 {len(report['shards'])}"
    )
    lines.append(f"- label_version `{report['label_version']}`；引擎 {report['manifest']['engine_sha256']}")
    lines.append("")
    lines.append("### 事件计数（真实 EVENTS 位 / 代理）")
    lines.append("")
    lines.append("| 事件类 | 正例 | 负例 | 缺失 | 来源 | 阈值(500) |")
    lines.append("|---|---|---|---|---|---|")
    for name, entry in report["events"].items():
        lines.append(
            f"| {name} | {entry['positives']} | {entry['negatives']} | {entry['nulls']} | "
            f"{entry['event_sources']} | {'是' if entry['passes'] else '否'} |"
        )
    consistency = report["event_consistency"]
    hit = consistency["hit_vs_alive_to_dying"]
    lines.append("")
    lines.append("### EVENTS ↔ 代理一致性（同一批帧）")
    lines.append("")
    lines.append("| 比较 | 数字 |")
    lines.append("|---|---|")
    lines.append(f"| EVENTS 可用帧 | {consistency['events_available_frames']} / {consistency['frames']} |")
    lines.append(f"| `EVENTS hit`(0x2) 次数 | {hit['hit_count']} |")
    lines.append(f"| 代理 `ALIVE→DYING` 次数 | {hit['alive_to_dying_count']} |")
    lines.append(f"| 两者同帧 | {hit['cells']['hit_and_edge']} |")
    lines.append(f"| 仅 hit（无 dying 边） | {hit['cells']['hit_only']} |")
    lines.append(f"| 仅 dying 边（无 hit） | {hit['cells']['edge_only']} |")
    lines.append(f"| 两者皆无 | {hit['cells']['neither']} |")
    lines.append(
        f"| `edge_recall_by_hit`（边被 hit 覆盖的比例） | {_fmt_rate(hit['edge_recall_by_hit'])} |"
    )
    aligned = hit["aligned_cells"]
    lines.append(
        f"| **对齐后** `hit(k)` vs `edge(k-1)` | 同时 {aligned['both']}；仅 hit "
        f"{aligned['hit_only']}；仅边 {aligned['edge_only']}；皆无 {aligned['neither']} "
        f"（配对 {aligned['pairs']}） |"
    )
    lines.append(
        f"| 对齐后 `edge_recall_by_hit` | {_fmt_rate(hit['aligned_edge_recall_by_hit'])} |"
    )
    death = consistency["death_bit_vs_dying_started"]
    lines.append(
        f"| `death`(0x4) / `player_dying_started`(0x200) | "
        f"{death['death_bit_true']} / {death['dying_started_true']}（同帧 {death['both']}） |"
    )
    dproxy = consistency["death_bit_vs_proxy"]
    lines.append(
        f"| `death` 位 vs 代理 | 同时 {dproxy['both']}；仅位 {dproxy['bit_only']}；仅代理 {dproxy['proxy_only']} |"
    )
    bproxy = consistency["bomb_bit_vs_proxy"]
    lines.append(
        f"| `bomb`(0x1) vs 代理 | 位 {bproxy['bomb_bit_true']}；代理 {bproxy['bomb_proxy_true']}；"
        f"同时 {bproxy['both']}；仅位 {bproxy['bit_only']}；仅代理 {bproxy['proxy_only']}；"
        f"代理拒判(死亡帧) {bproxy['proxy_ambiguous_with_death']} |"
    )
    lines.append("")
    danger = report["future_danger"]
    occupancy = report["occupancy"]
    lines.append("### 感知标签覆盖")
    lines.append("")
    lines.append("| 标签 | 统计 |")
    lines.append("|---|---|")
    lines.append(
        f"| future_danger(H=16) | 真 {danger['positive']} / 假 {danger['negative']} / "
        f"None {danger['null']}；缺失率 {_fmt_rate(danger['missing_rate'])}；"
        f"min_gap p50 {_fmt_px(danger['min_gap_px']['p50'])} |"
    )
    lines.append(
        f"| occupancy | 非空 {occupancy['nonempty']} / 空 {occupancy['empty']} / 缺 {occupancy['missing']} |"
    )
    hud = report["hud"]
    lines.append("")
    lines.append("### HUD 分布")
    lines.append("")
    lines.append("| 标量 | 分布 |")
    lines.append("|---|---|")
    for field in ("lives", "bombs", "power", "point", "time"):
        lines.append(f"| {field} | {hud[field]} |")
    lines.append(f"| score | {hud['score']} |")
    lines.append(f"| graze | {hud['graze']} |")
    lines.append("")
    gap = report["g2_data_gap"]
    lines.append("### G2 数据量缺口（不是检测器成绩）")
    lines.append("")
    lines.append(
        f"- 统计口径：零错误时单侧 95% 下界达到 99% 需 **{gap['required_positives']['event_rate_99']}** "
        f"个正例；99.5% 需 **{gap['required_positives']['hud_frame_accuracy_995']}** 帧；"
        f"死亡延迟中位数至少 **{gap['required_positives']['death_latency_median_events']}** 个事件（提议下限）。"
    )
    lines.append("")
    lines.append("| 门 | 已有 | 需要 | 缺口 | 按当前出现率还需帧 |")
    lines.append("|---|---|---|---|---|")
    for name, entry in gap["events"].items():
        need = entry.get("additional_frames")
        lines.append(
            f"| event:{name} | {entry['have']} | {entry['needed']} | {entry['deficit']} | "
            f"{'—' if need is None else need} |"
        )
    hud_gap = gap["hud"]
    lines.append(
        f"| 残机/bomb 逐帧准确率 | {hud_gap['frames_with_complete_label']} 帧 | "
        f"{hud_gap['accuracy_frames_needed']} | {hud_gap['accuracy_deficit']} | — |"
    )
    score_delta = hud_gap["score_delta"]
    lines.append(
        f"| 分数增量符号 | {score_delta['frames_with_nonzero_delta']} 帧有非零增量 "
        f"（涨 {score_delta['rises']}/跌 {score_delta['drops']}） | {score_delta['needed']} | "
        f"{score_delta['deficit']} | {score_delta['additional_frames'] if score_delta['additional_frames'] is not None else '—'} |"
    )
    lines.append(
        f"| 死亡延迟（事件数） | {gap['death_latency']['death_events_with_pixel']} | "
        f"{gap['death_latency']['proposed_median_floor']} | {gap['death_latency']['deficit']} | — |"
    )
    lines.append("")
    lines.append("**HUD 取值覆盖（12-P2 §2.2：每个取值 ≥1000 帧）**")
    lines.append("")
    for field in ("lives", "bombs"):
        values = hud_gap["values"][field]
        observed = {value: entry["frames"] for value, entry in values.items() if entry["observed"]}
        never = [value for value, entry in values.items() if not entry["observed"]]
        lines.append(f"- `{field}`：已观测 {observed}；从未出现 {never}")
    return "\n".join(lines) + "\n"


def _fmt_rate(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"


def _fmt_px(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2f} px"

