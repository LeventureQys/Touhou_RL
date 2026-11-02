"""Pure offline rollout timing contract, schema v200-rollout-timing/1.

No collector, game-process, NumPy, or platform dependencies. Structural errors
raise ValueError; measurable-data quality problems are retained in reports.
Stages are monotonically ordered identifiers, not a common frame clock.
"""
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json

SCHEMA_VERSION = "v200-rollout-timing/1"
TIMESTAMP_KEYS = ("capture_before", "capture_after", "action_issue", "action_effective", "step_end")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _stamp(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a stage/frame mapping")
    return {key: _integer(value.get(key), f"{name}.{key}") for key in ("stage", "frame")}


def _key(stamp):
    return stamp["stage"], stamp["frame"]


def _bullets(row):
    snapshot = row.get("raw_safety_snapshot", row.get("raw_snapshot", {}))
    if not isinstance(snapshot, Mapping):
        raise ValueError("raw_snapshot must be a mapping")
    bullets = snapshot.get("bullets", row.get("bullets"))
    if bullets is None:
        return None
    if not isinstance(bullets, Mapping):
        raise ValueError("bullets must be parallel arrays")
    if "generation" not in bullets and snapshot.get("generation_valid") is False:
        return None
    for name in ("slot_id", "state", "generation"):
        if name not in bullets:
            raise ValueError(f"bullets.{name} is required")
    lengths = []
    for name, array in bullets.items():
        if not isinstance(array, (list, tuple)):
            raise ValueError(f"bullets.{name} must be an array")
        lengths.append(len(array))
    if len(set(lengths)) > 1:
        raise ValueError("bullet arrays must have equal lengths")
    slots = [_integer(slot, "bullets.slot_id") for slot in bullets["slot_id"]]
    if len(slots) != len(set(slots)):
        raise ValueError("bullet slot_id values must be unique")
    for generation in bullets["generation"]:
        _integer(generation, "bullets.generation")
    return bullets


def validate_timing_row(row):
    """Return an independent normalized JSON row, or raise ValueError.

    Transition rows permit a stage increase only at an action boundary (not
    during capture). Cross-stage and zero-frame holds have no duration. A
    readback mismatch is a quality reason, not a structural exception.
    """
    if not isinstance(row, Mapping):
        raise ValueError("row must be a mapping")
    result = deepcopy(dict(row))
    if result.get("schema", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("unsupported schema")
    result["schema"] = SCHEMA_VERSION
    if not isinstance(result.get("rollout_id"), str) or not result["rollout_id"]:
        raise ValueError("rollout_id must be a non-empty string")
    _integer(result.get("episode"), "episode", 1)
    for name in ("decision", "stage", "frame"):
        _integer(result.get(name), name)
    action = result.get("executed_action")
    if not isinstance(action, (list, tuple)) or len(action) != 4:
        raise ValueError("executed_action must contain four integers")
    if any(type(value) is not int for value in action):
        raise ValueError("executed_action must contain four integers")
    if not 0 <= action[0] < 9 or any(value not in (0, 1) for value in action[1:]):
        raise ValueError("executed_action out of range")
    result["executed_action"] = list(action)
    transition = result.get("transition", False)
    if type(transition) is not bool:
        raise ValueError("transition must be a boolean")
    result["transition"] = transition
    stamps = result.get("timestamps")
    if not isinstance(stamps, Mapping):
        raise ValueError("timestamps must be a mapping")
    stamps = {name: (None if name == "action_effective" and stamps.get(name) is None else _stamp(stamps.get(name), name)) for name in TIMESTAMP_KEYS}
    result["timestamps"] = stamps
    ordered = [name for name in TIMESTAMP_KEYS if stamps[name] is not None]
    for left, right in zip(ordered, ordered[1:]):
        before, after = stamps[left], stamps[right]
        if _key(before) > _key(after):
            raise ValueError(f"timestamps out of order: {left} > {right}")
        if before["stage"] != after["stage"] and (not transition or left == "capture_before"):
            raise ValueError("cross-stage timestamps require an action boundary and transition=true")
    effective, end = stamps["action_effective"], stamps["step_end"]
    valid = effective is not None and effective["stage"] == end["stage"] and end["frame"] > effective["frame"]
    reasons = []
    if not valid:
        reasons.append("effective_stamp_missing" if effective is None else "cross_stage_hold" if effective["stage"] != end["stage"] else "zero_frame_hold")
    result["action_hold_frames_valid"] = valid
    result["actual_hold_frames"] = end["frame"] - effective["frame"] if valid else None
    readback = result.get("input_mask_readback")
    result["input_mask_readback_valid"] = None
    if readback is not None:
        stamp = _stamp(readback, "input_mask_readback")
        _integer(readback.get("mask"), "input_mask_readback.mask")
        result["input_mask_readback_valid"] = stamp == effective
        if stamp != effective:
            reasons.append("input_mask_readback_mismatch")
            result["action_hold_frames_valid"] = False
            result["actual_hold_frames"] = None
    result["reasons"] = reasons
    _bullets(result)
    try:
        return json.loads(json.dumps(result, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("row must contain finite JSON data") from exc


def validate_trace(rows):
    """Validate one rollout/episode in decision and capture order.

    Duplicate/reordered decisions or capture stamps and mixed rollout/episode
    identities raise ValueError. Generation regressions and slot reuse without
    an increment are reported with row/slot context rather than discarded.
    Missing snapshots interrupt adjacency; an inactive (state=0) to active
    transition or a slot disappearing then returning constitutes slot reuse.
    """
    normalized = [validate_timing_row(row) for row in rows]
    reasons = Counter()
    issues = []
    previous = None
    previous_slots = None
    seen_slots = {}
    for index, row in enumerate(normalized):
        if previous is not None:
            if (row["rollout_id"], row["episode"]) != (previous["rollout_id"], previous["episode"]):
                raise ValueError("trace must contain exactly one rollout and episode")
            if row["decision"] <= previous["decision"]:
                raise ValueError("duplicate or reordered decision")
            if (row["stage"], row["frame"]) <= (previous["stage"], previous["frame"]):
                raise ValueError("duplicate or reordered row stage/frame")
            if _key(row["timestamps"]["capture_before"]) < _key(previous["timestamps"]["capture_before"]):
                raise ValueError("reordered capture timestamps")
        reasons.update(row["reasons"])
        bullets = _bullets(row)
        current_slots = None
        if bullets is not None:
            current_slots = {}
            for offset, slot in enumerate(bullets["slot_id"]):
                generation, state = bullets["generation"][offset], bullets["state"][offset]
                current_slots[slot] = (generation, state)
                old = seen_slots.get(slot)
                if old is not None and previous_slots is not None:
                    reused = slot not in previous_slots or (old[1] == 0 and state != 0)
                    reason = "generation_regression" if generation < old[0] else (
                        "slot_reuse_without_generation_increment" if reused and generation <= old[0] else None)
                    if reason:
                        reasons[reason] += 1
                        issues.append({"row": index, "slot_id": slot, "reason": reason})
                seen_slots[slot] = (generation, state)
        previous_slots = current_slots
        previous = row
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    valid_count = sum(row["action_hold_frames_valid"] for row in normalized)
    return {"schema": SCHEMA_VERSION, "row_count": len(normalized),
            "duration_valid_count": valid_count, "duration_invalid_count": len(normalized) - valid_count,
            "generation_issue_count": len(issues), "generation_issues": issues,
            "reason_counts": dict(sorted(reasons.items())),
            "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "valid": not reasons}
