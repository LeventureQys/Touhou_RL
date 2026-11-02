"""Offline preregistered rollout split validation and trace auditing.

This module is deliberately independent of the game/runtime.  It consumes only a
JSON protocol and JSONL traces and returns a reproducible audit report.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "v200-preregistered-rollout/1"
_SPLITS = ("train", "dev", "test")
_HEX64 = set("0123456789abcdef")
_PROTOCOL_KEYS = {
    "schema", "checkpoint_sha256", "teacher_source_sha256", "environment_digest",
    "frameskip", "episodes", "min_samples_each", "purge_windows", "max_attempts", "thresholds",
}
_PURGE_KEYS = {"future_decisions", "history_decisions", "chunk_decisions"}


def _bad(message: str) -> None:
    raise ValueError(message)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _bad(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _bad(f"{name} must be a non-negative integer")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in _HEX64 for c in value):
        _bad(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def protocol_digest(protocol: dict[str, Any]) -> str:
    """Return the SHA-256 digest of canonical protocol JSON."""
    return hashlib.sha256(_canonical(protocol)).hexdigest()


def validate_protocol(path_or_protocol: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Validate and return a protocol loaded from a mapping or JSON file."""
    if isinstance(path_or_protocol, (str, Path)):
        try:
            value = json.loads(Path(path_or_protocol).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid protocol JSON: {exc}") from exc
    else:
        value = path_or_protocol
    if not isinstance(value, dict):
        _bad("protocol must be an object")
    unknown = set(value) - _PROTOCOL_KEYS
    missing = _PROTOCOL_KEYS - set(value)
    if unknown:
        _bad(f"unknown protocol fields: {sorted(unknown)}")
    if missing:
        _bad(f"missing protocol fields: {sorted(missing)}")
    if value["schema"] != SCHEMA or not isinstance(value["schema"], str):
        _bad(f"schema must be {SCHEMA!r}")
    for key in ("checkpoint_sha256", "teacher_source_sha256", "environment_digest"):
        _sha(value[key], key)
    _positive_int(value["frameskip"], "frameskip")
    _positive_int(value["min_samples_each"], "min_samples_each")
    _positive_int(value["max_attempts"], "max_attempts")
    episodes = value["episodes"]
    if not isinstance(episodes, dict) or set(episodes) != set(_SPLITS):
        _bad("episodes must contain exactly train/dev/test")
    seen: set[int] = set()
    for split in _SPLITS:
        ids = episodes[split]
        if not isinstance(ids, list) or not ids:
            _bad(f"episodes.{split} must be a non-empty array")
        local: set[int] = set()
        for episode in ids:
            _positive_int(episode, f"episodes.{split} episode")
            if episode in local:
                _bad(f"duplicate episode {episode} in {split}")
            if episode in seen:
                _bad(f"episode {episode} overlaps splits")
            local.add(episode); seen.add(episode)
    purge = value["purge_windows"]
    if not isinstance(purge, dict) or set(purge) != _PURGE_KEYS:
        _bad("purge_windows must contain exactly future_decisions/history_decisions/chunk_decisions")
    for key in _PURGE_KEYS:
        _nonnegative_int(purge[key], f"purge_windows.{key}")
    thresholds = value["thresholds"]
    if not isinstance(thresholds, dict):
        _bad("thresholds must be an object")
    # Threshold names are intentionally extensible, but values must be JSON numbers.
    for key, val in thresholds.items():
        if not isinstance(key, str) or isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            _bad("thresholds values must be finite numbers")
    return value


def _trace_digest(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.read_bytes())
    return h.hexdigest()


def audit_trace_splits(trace_paths: Iterable[str | Path], protocol: str | Path | dict[str, Any], split_name: str) -> dict[str, Any]:
    """Audit JSONL traces for one preregistered split.

    Sample indices may start at any non-negative value and may be globally
    scattered; within one audited split they must be unique and strictly
    increasing in trace order.  Thresholds are reported as gates, never
    silently treated as passed.
    """
    p = validate_protocol(protocol)
    if split_name not in _SPLITS or split_name == "r1d":
        _bad("split_name must be train, dev, or test; r1d is not a formal holdout")
    paths = [Path(x) for x in trace_paths]
    if not paths:
        _bad("trace_paths must not be empty")
    allowed = set(p["episodes"][split_name])
    ownership = {e: s for s in _SPLITS for e in p["episodes"][s]}
    radius = max(p["purge_windows"].values())
    rows: list[dict[str, Any]] = []
    rollout_id: str | None = None
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"cannot read trace {path}: {exc}") from exc
        for line_no, line in enumerate(lines, 1):
            try: row = json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            if not isinstance(row, dict): _bad(f"trace row {path}:{line_no} is not an object")
            for key in ("episode", "decision", "rollout_id", "sample_index", "frame", "stage"):
                if key not in row: _bad(f"trace row missing {key} at {path}:{line_no}")
            if isinstance(row["episode"], bool) or not isinstance(row["episode"], int) or row["episode"] <= 0: _bad("episode must be positive integer")
            if isinstance(row["decision"], bool) or not isinstance(row["decision"], int) or row["decision"] < 0: _bad("decision must be non-negative integer")
            if not isinstance(row["rollout_id"], str) or not row["rollout_id"]: _bad("rollout_id must be non-empty string")
            if rollout_id is None: rollout_id = row["rollout_id"]
            elif row["rollout_id"] != rollout_id: _bad("rollout_id is inconsistent")
            if row["sample_index"] is not None and (isinstance(row["sample_index"], bool) or not isinstance(row["sample_index"], int) or row["sample_index"] < 0): _bad("sample_index must be non-negative integer or null")
            if isinstance(row["frame"], bool) or not isinstance(row["frame"], int) or row["frame"] < 0: _bad("frame must be non-negative integer")
            if isinstance(row["stage"], bool) or not isinstance(row["stage"], int) or row["stage"] < 0: _bad("stage must be non-negative integer")
            ep = row["episode"]
            if ep not in ownership: _bad(f"episode {ep} is not declared by protocol")
            if ownership[ep] != split_name: _bad(f"episode {ep} belongs to {ownership[ep]}, not {split_name}")
            snap = row.get("raw_safety_snapshot")
            if not isinstance(snap, dict) or snap.get("frame") != row["frame"] or snap.get("stage") != row["stage"]:
                _bad(f"raw snapshot frame/stage mismatch at {path}:{line_no}")
            # Preserve raw skipped rows; they are boundary evidence, not samples.
            rows.append(row)
    if rollout_id == "r1d": _bad("r1d cannot be used as a formal unseen split")
    if not rows: _bad("trace rows must not be empty")
    prev_ep = prev_dec = None
    seen_samples: set[int] = set()
    previous_sample = None
    for row in rows:
        ep, dec = row["episode"], row["decision"]
        if prev_ep is not None:
            if ep < prev_ep or (ep == prev_ep and dec != prev_dec + 1) or (ep > prev_ep and dec != 0):
                _bad("episode/decision order is not strict and contiguous")
        prev_ep, prev_dec = ep, dec
        si = row["sample_index"]
        if si is not None:
            if si in seen_samples: _bad("duplicate sample_index")
            if previous_sample is not None and si <= previous_sample: _bad("sample_index must be strictly increasing")
            seen_samples.add(si); previous_sample = si
    by_ep: dict[int, list[dict[str, Any]]] = {}
    for row in rows: by_ep.setdefault(row["episode"], []).append(row)
    counts = {"decisions": len(rows), "aligned_samples": len(seen_samples), "usable_samples": 0, "purged_samples": 0}
    purge_by_reason = {"episode_boundary": 0, "future_decisions": 0, "history_decisions": 0, "chunk_decisions": 0}
    usable_ids: set[int] = set()
    for ep_rows in by_ep.values():
        first, last = ep_rows[0]["decision"], ep_rows[-1]["decision"]
        for row in ep_rows:
            si = row["sample_index"]
            if si is None: continue
            d = row["decision"]; reasons = set()
            lo = d - p["purge_windows"]["history_decisions"]
            hi = d + max(p["purge_windows"]["future_decisions"], p["purge_windows"]["chunk_decisions"])
            if lo < first or hi > last: reasons.add("episode_boundary")
            window = [x for x in ep_rows if lo <= x["decision"] <= hi]
            if any(x.get("aligned") is not True or x.get("fresh_pixels") is not True or x.get("sample_index") is None for x in window):
                reasons.add("unaligned_or_missing")
            if any(x["stage"] != row["stage"] for x in window): reasons.add("stage_boundary")
            for left, right in zip(window, window[1:]):
                if right["decision"] != left["decision"] + 1 or right["frame"] - left["frame"] != p["frameskip"]:
                    reasons.add("frame_or_decision_gap")
                if left.get("terminated") or left.get("truncated") or left.get("deaths_after", 0) > left.get("deaths_before", 0) or right.get("deaths_before", 0) != left.get("deaths_before", 0):
                    reasons.add("death_or_terminal_boundary")
            if reasons:
                counts["purged_samples"] += 1
                for reason in reasons: purge_by_reason[reason] = purge_by_reason.get(reason, 0) + 1
            else:
                counts["usable_samples"] += 1; usable_ids.add(si)
    gates = {
        "min_samples": "pass" if counts["usable_samples"] >= p["min_samples_each"] else "fail",
        "thresholds": "unknown",  # no metric is evaluated here
        "aligned_fresh": "pass",
    }
    return {"schema": "v200-preregistered-rollout-audit/1", "split": split_name, "rollout_id": rollout_id,
            "episode_ids": sorted(by_ep), "decisions": counts["decisions"], "aligned_samples": counts["aligned_samples"],
            "usable_samples": counts["usable_samples"], "purged_samples": counts["purged_samples"], "purge_by_reason": purge_by_reason,
            "trace_sha256": _trace_digest(paths), "protocol_digest": protocol_digest(p), "leakage": False,
            "purge_radius": radius, "gates": gates, "unmeasured": [k for k,v in gates.items() if v == "unknown"],
            "pass": all(v == "pass" for v in gates.values())}


__all__ = ["SCHEMA", "validate_protocol", "audit_trace_splits", "protocol_digest"]
