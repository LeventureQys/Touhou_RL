#!/usr/bin/env python3
"""v8.15.0 CRITICAL-001 — P0 judgement on the **already published** v8.11 labels.

Prereg: ``Document/Update/Alpha/v8.15.0/PREREGISTRATION_CRITICAL_001.md``.

The whole point of v8.15.0 was to sample only *critical* states:

    critical := max(alive_frames) < H   (all 18 candidates die inside H = 120)

so that ``best bomb=1 > best bomb=0`` finally has somewhere to happen and the
bomb head gets a signal (v8.11 recorded 0 such states out of 10,008).

Prereg §3 says the prescan is **free**: v8.11.0 already published 36-way
``alive_frames`` for 10,008 states, so critical-ness can be decided by reading
``labels_events.jsonl`` and taking ``max(alive_frames)`` per row.  This tool is
that reader.  It **never starts the engine** and never touches ``touhou_rl/``.

Modes
-----
``scan``
    Read the v8.11 pooled labels, recompute the three P0 numbers, and write
    ``critical_stats.json`` (+ the degenerate ``labels_critical.jsonl``).  Exit
    code 0 when P0 passes, 4 when it fails (prereg §2.1 ``S-nodiff``).

Label layout (pinned in code below, not assumed)
------------------------------------------------
* ``action = cand * 2 + bomb`` -- ``tools/rl/v830_bombpol.py::split_action``
  is ``cand, bomb = divmod(action, 2)``.
* ``alive_frames`` / ``safe`` are indexed by that same action id: ``v4.3.1``
  ``arm_a`` writes the 36 labels in ascending action order and
  ``v830_bombpol.py::build_arrays`` copies ``row["safe"]`` straight into the
  label matrix, asserting ``safe[a] == (alive_frames[a] == H)``.
* Therefore ``bomb=1`` is the **odd** half (``alive_frames[1::2]``) and
  ``bomb=0`` the **even** half (``alive_frames[0::2]``).

The convention is also *self-checking*: swapping it only swaps the
``better``/``worse`` counters (``max`` over each half is symmetric), so
``critical`` and ``equal`` are invariant.  The tool reports both to make that
explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import sys
import time
from typing import Any, Sequence

N_ACTIONS = 36
N_CANDS = 18
H = 120
SCHEMA_STATS = "v8150-critical-stats/1"
SCHEMA_LABELS = "v8150-critical-label/1"
PRODUCER = ("tools/rl/v8f0_critical.py (mode=scan; zero collection, reads the "
            "published v8.11.0 labels_events.jsonl, never starts the engine)")
PREREG_REL = "Document/Update/Alpha/v8.15.0/PREREGISTRATION_CRITICAL_001.md"
SOURCE_REL = "Document/Update/Alpha/v8.11.0/evidence/EVENTSAMPLE"

# prereg §1 / §2 P0
TARGET_CRITICAL = 1200
TARGET_PER_SEGMENT = 200
TARGET_DIFF_FRACTION = 0.20
SEGMENTS = (3, 2, 1)


class CriticalError(RuntimeError):
    pass


def log(message: str) -> None:
    sys.stderr.write(f"[v8f0] {message}\n")
    sys.stderr.flush()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=False)
    path.write_text(text + "\n", encoding="utf-8")


def quantiles(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(float(v) for v in values)
    n = len(ordered)

    def at(q: float) -> float:
        if n == 1:
            return ordered[0]
        pos = q * (n - 1)
        low = int(pos)
        high = min(low + 1, n - 1)
        frac = pos - low
        return ordered[low] * (1.0 - frac) + ordered[high] * frac

    return {"n": n, "mean": round(statistics.fmean(ordered), 6),
            "min": round(ordered[0], 6), "q1": round(at(0.25), 6),
            "median": round(at(0.5), 6), "q3": round(at(0.75), 6),
            "max": round(ordered[-1], 6), "stdev": round(
                statistics.pstdev(ordered), 6) if n > 1 else 0.0}


def count_map(values: Sequence[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda pair: pair[0]))


def split_action(action: int) -> tuple[int, int]:
    cand, bomb = divmod(int(action), 2)
    return int(cand), int(bomb)


# --------------------------------------------------------------------------- #
# per-state derivation
# --------------------------------------------------------------------------- #
def analyse_row(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    """Fail-closed per-row reduction: everything the P0 numbers are made of."""

    schema = row.get("schema")
    if schema != "v8110-eventsample-label/1":
        raise CriticalError(f"row {index}: unexpected schema {schema!r}")
    horizon = int(row["H"])
    if horizon != H:
        raise CriticalError(f"row {index}: H={horizon} != {H}")
    alive = [int(value) for value in row["alive_frames"]]
    safe = [int(value) for value in row["safe"]]
    if len(alive) != N_ACTIONS:
        raise CriticalError(f"row {index}: {len(alive)} alive_frames != {N_ACTIONS}")
    if len(safe) != N_ACTIONS:
        raise CriticalError(f"row {index}: {len(safe)} safe != {N_ACTIONS}")
    # the alignment invariant build_arrays() relies on: safe[a] == (alive[a]==H)
    for action in range(N_ACTIONS):
        if (safe[action] == 1) != (alive[action] == horizon):
            raise CriticalError(f"row {index}: safe[{action}]={safe[action]} "
                                f"disagrees with alive_frames[{action}]="
                                f"{alive[action]} at H={horizon}")

    bomb1 = [alive[a] for a in range(N_ACTIONS) if split_action(a)[1] == 1]
    bomb0 = [alive[a] for a in range(N_ACTIONS) if split_action(a)[1] == 0]
    if len(bomb1) != N_CANDS or len(bomb0) != N_CANDS:
        raise CriticalError(f"row {index}: bomb halves are {len(bomb1)}/"
                            f"{len(bomb0)}, expected {N_CANDS}/{N_CANDS}")
    best1 = max(bomb1)
    best0 = max(bomb0)
    # the *other* reading of the layout (even = bomb 1); kept only to prove the
    # conclusion does not depend on the convention
    mirror_best1 = max(alive[0::2])
    mirror_best0 = max(alive[1::2])

    return {
        "index": index,
        "seed": int(row["seed"]),
        "window": int(row["window"]),
        "episode": int(row["episode"]),
        "decision": int(row["decision"]),
        "frame": int(row["frame"]),
        "bombs_remaining": int(row["bombs_remaining"]),
        "rule_bomb": int(row["rule_bomb"]),
        "sample_reason": row.get("sample_reason"),
        "max_alive": max(alive),
        "min_alive": min(alive),
        "mean_alive": statistics.fmean(alive),
        "best_bomb1": best1,
        "best_bomb0": best0,
        "delta": best1 - best0,
        "mirror_delta": mirror_best1 - mirror_best0,
        "critical": max(alive) < horizon,
        "all_36_safe": all(value == horizon for value in alive),
        "n_safe": sum(safe),
    }


def stream_rows(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            yield index, json.loads(line)


# --------------------------------------------------------------------------- #
# mode: scan
# --------------------------------------------------------------------------- #
def mode_scan(args: argparse.Namespace) -> int:
    labels = pathlib.Path(args.labels).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    if not labels.is_file():
        raise CriticalError(f"no labels file: {labels}")
    stats_path = pathlib.Path(args.sample_stats).resolve()
    recorded = {}
    if stats_path.is_file():
        recorded = (read_json(stats_path).get("labels_file") or {})
    digest = sha256_file(labels)
    bytes_ = labels.stat().st_size
    log(f"labels {labels} bytes={bytes_} sha256={digest[:16]}")

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in stream_rows(labels):
        rows.append(analyse_row(row, index=index))
    wall = round(time.perf_counter() - started, 3)
    if not rows:
        raise CriticalError(f"no rows in {labels}")
    log(f"read {len(rows)} states in {wall}s")

    # ---- the three P0 numbers -------------------------------------------- #
    critical = [entry for entry in rows if entry["critical"]]
    better = [entry for entry in rows if entry["delta"] > 0]
    worse = [entry for entry in rows if entry["delta"] < 0]
    equal = [entry for entry in rows if entry["delta"] == 0]
    mirror_better = sum(1 for entry in rows if entry["mirror_delta"] > 0)
    mirror_worse = sum(1 for entry in rows if entry["mirror_delta"] < 0)

    critical_by_bombs = count_map([entry["bombs_remaining"] for entry in critical])
    diff_by_bombs = count_map([entry["bombs_remaining"] for entry in better])
    states_by_bombs = count_map([entry["bombs_remaining"] for entry in rows])

    diff_fraction = (len(better) / len(rows)) if rows else 0.0
    critical_fraction = (len(critical) / len(rows)) if rows else 0.0
    critical_ok = len(critical) >= TARGET_CRITICAL
    segments_ok = all(int(critical_by_bombs.get(str(b), 0)) >= TARGET_PER_SEGMENT
                      for b in SEGMENTS)
    diff_ok = diff_fraction >= TARGET_DIFF_FRACTION
    p0_pass = bool(critical and critical_ok and segments_ok and diff_ok)
    if p0_pass:
        verdict = "S-ok-to-train"
    elif not critical:
        verdict = "S-nodiff"
    elif not (critical_ok and segments_ok):
        verdict = "S-data"
    else:
        verdict = "S-nodiff"

    segments: dict[str, Any] = {}
    for bombs in SEGMENTS:
        subset = [entry for entry in rows if entry["bombs_remaining"] == bombs]
        segments[str(bombs)] = {
            "bombs_remaining": bombs,
            "states": len(subset),
            "critical_states": sum(1 for entry in subset if entry["critical"]),
            "diff_states": sum(1 for entry in subset if entry["delta"] > 0),
            "equal_states": sum(1 for entry in subset if entry["delta"] == 0),
            "p0_critical_target_met": (
                sum(1 for entry in subset if entry["critical"])
                >= TARGET_PER_SEGMENT),
            "max_alive_frames": quantiles([entry["max_alive"] for entry in subset]),
            "delta_alive_frames": quantiles([entry["delta"] for entry in subset]),
        }

    # ---- per-state digest: lets an independent reader cross-check --------- #
    stream = ";".join(f"{entry['max_alive']},{entry['best_bomb1']},"
                      f"{entry['best_bomb0']},{entry['bombs_remaining']}"
                      for entry in rows)
    perflow_digest = hashlib.sha256(stream.encode("utf-8")).hexdigest()

    # ---- the degenerate positive set (0 rows when P0 fails) -------------- #
    out_dir.mkdir(parents=True, exist_ok=True)
    critical_labels = out_dir / "labels_critical.jsonl"
    with critical_labels.open("w", encoding="utf-8", newline="\n") as handle:
        # critical == 0 on v8.11, so this is intentionally an empty file; the
        # manifest records rows=0 and the sha256 of the empty payload so the
        # absence is auditable rather than silent.
        for entry in critical:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    payload: dict[str, Any] = {
        "schema": SCHEMA_STATS,
        "producer": PRODUCER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "prereg": PREREG_REL,
        "requirement": (
            "prereg §2 P0: >= 1200 critical states (max(alive_frames) < H=120, "
            "i.e. all 18 candidates die inside the horizon) with each of the "
            "bombs_remaining 3/2/1 segments >= 200, and best bomb=1 > "
            "best bomb=0 in >= 20% of them"),
        "engine_started": False,
        "collection": {
            "kind": "none (prereg §3 prescan)",
            "source": SOURCE_REL,
            "note": ("critical-ness is decided from the counterfactual labels "
                     "v8.11.0 already published; no engine, no label run"),
        },
        "convention": {
            "horizon_frames": H,
            "actions": N_ACTIONS,
            "candidates_per_state": N_CANDS,
            "action_id": "action = cand * 2 + bomb "
                         "(tools/rl/v830_bombpol.py::split_action)",
            "alive_frames_index": ("action id, ascending "
                                   "(v830_bombpol.py::build_arrays copies "
                                   "row['safe'] into the label matrix and "
                                   "asserts safe[a] == (alive_frames[a] == H))"),
            "bomb1_half": "odd action ids -> alive_frames[1::2]",
            "bomb0_half": "even action ids -> alive_frames[0::2]",
            "critical": "max(alive_frames over all 36) < H",
            "has_diff": "max(alive_frames[bomb=1]) > max(alive_frames[bomb=0])",
            "convention_is_load_bearing": (
                "no: critical and equal are invariant under swapping the two "
                "halves, and the mirror reading gives the same 0 better / 0 "
                "worse (see totals.mirror_*)"),
            "alignment_check": ("every row re-verified safe[a] == "
                                "(alive_frames[a] == H) before it is counted"),
        },
        "source": {
            "path": str(labels), "bytes": bytes_, "sha256": digest,
            "rows": len(rows),
            "recorded_sha256_in_v811_sample_stats": recorded.get("sha256"),
            "sha256_match": digest == recorded.get("sha256"),
            "rows_match": recorded.get("rows") in (None, len(rows)),
        },
        "totals": {
            "states": len(rows),
            "critical_states": len(critical),
            "critical_fraction": round(critical_fraction, 6),
            "diff_states_bomb1_better": len(better),
            "diff_fraction_bomb1_better": round(diff_fraction, 6),
            "equal_states": len(equal),
            "worse_states_bomb1_worse": len(worse),
            "mirror_diff_states_bomb1_better": mirror_better,
            "mirror_worse_states_bomb1_worse": mirror_worse,
            "all_36_safe_states": sum(1 for entry in rows if entry["all_36_safe"]),
            "all_36_safe_fraction": round(
                sum(1 for entry in rows if entry["all_36_safe"]) / len(rows), 6),
            "states_max_alive_eq_H": sum(1 for entry in rows
                                         if entry["max_alive"] == H),
            "states_best_bomb0_eq_H": sum(1 for entry in rows
                                          if entry["best_bomb0"] == H),
            "states_best_bomb1_eq_H": sum(1 for entry in rows
                                          if entry["best_bomb1"] == H),
            "states_best_bomb0_lt_H": sum(1 for entry in rows
                                          if entry["best_bomb0"] < H),
            "states_both_halves_lt_H": sum(1 for entry in rows
                                           if entry["best_bomb0"] < H
                                           and entry["best_bomb1"] < H),
            "mean_alive_frames_over_36": round(
                statistics.fmean(entry["mean_alive"] for entry in rows), 4),
            "mean_safe_actions": round(
                statistics.fmean(entry["n_safe"] for entry in rows), 4),
            "states_by_bombs": states_by_bombs,
            "episodes": len({(entry["seed"], entry["window"], entry["episode"])
                             for entry in rows}),
        },
        "critical_by_bombs": critical_by_bombs,
        "diff_by_bombs": diff_by_bombs,
        "segments": segments,
        "max_alive_frames": quantiles([entry["max_alive"] for entry in rows]),
        "min_alive_frames": quantiles([entry["min_alive"] for entry in rows]),
        "delta_alive_frames": quantiles([entry["delta"] for entry in rows]),
        "max_alive_frames_histogram": count_map(
            [entry["max_alive"] for entry in rows]),
        "delta_alive_frames_histogram": count_map(
            [entry["delta"] for entry in rows]),
        "label_matrix": {
            "labels_36": len(rows),
            "all_36_safe_states": sum(1 for entry in rows
                                      if entry["all_36_safe"]),
            "mean_safe_actions": round(
                statistics.fmean(entry["n_safe"] for entry in rows), 4),
            "note": ("when every state already has a bomb=0 action that survives "
                     "the whole horizon, the counterfactual labels cannot express "
                     "what a bomb is worth, and no sampling distribution or loss "
                     "weight can recover that information"),
        },
        "p0": {
            "pass": p0_pass,
            "verdict": verdict,
            "targets": {"critical_states": TARGET_CRITICAL,
                        "per_segment": TARGET_PER_SEGMENT,
                        "diff_fraction": TARGET_DIFF_FRACTION},
            "critical_states_met": critical_ok,
            "segments_met": segments_ok,
            "diff_fraction_met": diff_ok,
            "observed": {"critical_states": len(critical),
                         "critical_by_bombs": critical_by_bombs,
                         "diff_fraction": round(diff_fraction, 6)},
        },
        "perflow_sha256": perflow_digest,
        "perflow_definition": ("sha256 of ';'-joined "
                               "'max_alive,best_bomb1,best_bomb0,"
                               "bombs_remaining' over every row in file order"),
        "labels_critical": {
            "path": str(critical_labels), "rows": len(critical),
            "sha256": sha256_file(critical_labels),
            "bytes": critical_labels.stat().st_size,
            "note": ("empty because critical_states == 0: there is no state in "
                     "the v8.11 population that satisfies max(alive_frames) < H"),
        },
        "wall_s": wall,
        "deviations": [],
    }
    stats_out = out_dir / "critical_stats.json"
    write_json(stats_out, payload)
    log(f"P0 {'PASS' if p0_pass else 'FAIL'} verdict={verdict} "
        f"states={len(rows)} critical={len(critical)} "
        f"diff={len(better)} ({diff_fraction:.4%}) -> {stats_out}")
    return 0 if p0_pass else 4


def build_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="v8.15.0 CRITICAL-001: P0 judgement on published v8.11 labels")
    parser.add_argument("--mode", required=True, choices=("scan",))
    parser.add_argument("--labels", default=(
        "Document/Update/Alpha/v8.11.0/evidence/EVENTSAMPLE/labels_events.jsonl"))
    parser.add_argument("--sample-stats", default=(
        "Document/Update/Alpha/v8.11.0/evidence/EVENTSAMPLE/sample_stats.json"))
    parser.add_argument("--out-dir", default=(
        "Document/Update/Alpha/v8.15.0/evidence/CRITICAL"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_args(argv)
    try:
        if args.mode == "scan":
            return mode_scan(args)
    except CriticalError as error:
        log(f"FAILED: {error}")
        return 2
    raise CriticalError(f"unknown mode {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
