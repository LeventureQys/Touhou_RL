"""v3.5.0 S0 trace contract: schema, completeness and event classification.

This module is deliberately pure: it builds dictionaries and decides whether a
decision is ``complete``; it does not launch the game, read memory or train
anything.  The probe (``probe35.py``) supplies live snapshots, the offline
verifier consumes the same dictionaries.

Why a separate module
---------------------
The v3.4.0 Stage 0 report showed that a plausible-looking trace can be 23%
incomplete while still being usable for a loss curve.  The v3.5.0 contract
therefore makes "complete" a pre-registered predicate with named failure
reasons, and makes every field's unknown state explicit.  Keeping that logic
out of the live loop is what makes it unit-testable without a game process.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import spec
from .state import BossInfo, EnemyInfo, GameState, ItemInfo, ShotParams

TRACE_SCHEMA = "v350-trace/1"
EPISODE_SCHEMA = "v350-episode/1"
MANIFEST_SCHEMA = "v350-manifest/1"

# Groups that must be readable for a steady decision to count as complete.
REQUIRED_GROUPS = ("player", "bullets", "enemies", "items", "boss", "power", "shot", "gui")
# Subset used by S0-2 ("enemy/item/power 字段可追溯").
TRACEABILITY_GROUPS = ("enemies", "items", "power")

# Counter-vs-pool tolerance from the contract §7.4 (fixed before results).
BULLET_COUNT_TOLERANCE = 2


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _f(value: float) -> float | None:
    """Round a float for JSON, preserving non-finite values as ``null``."""
    return round(float(value), 3) if _finite(value) else None


def group_status(ok: bool, count: int | None = None) -> str:
    """Map a read outcome to the contract's ``ok | empty | error`` vocabulary."""
    if not ok:
        return "error"
    if count is not None and count == 0:
        return "empty"
    return "ok"


@dataclass
class Snapshot:
    """One post-action state plus the entity/bullet reads taken with it."""

    state: GameState
    wall_ms: int
    enemies: list[EnemyInfo] | None = None
    items: list[ItemInfo] | None = None
    bosses: list[BossInfo] | None = None
    shot: ShotParams | None = None
    bullets: dict | None = None  # keys: position, velocity, half, counter
    frame_event_flags: int | None = None
    status: dict[str, str] = field(default_factory=dict)

    def group(self, name: str) -> str:
        return self.status.get(name, "error")


def _state_dict(state: GameState) -> dict:
    return {
        "supervisor_state": int(state.supervisor_state),
        "stage": int(state.stage),
        "stage_frames": int(state.stage_frames),
        "gameplay_frame": int(state.gameplay_frame),
        "run_frames": int(state.run_frames),
        "flags": int(state.flags) & 0xFFFFFFFF,
        "game_cleared": bool(state.game_cleared),
        "stage_clear_sequence_active": bool(state.stage_clear_sequence_active),
        "stage_transition_state": int(state.stage_transition_state),
        "gui_stage_clear_screen_state": int(state.gui_stage_clear_screen_state),
        "gui_dialogue_present": bool(state.gui_dialogue_present),
        "gui_dialogue_msg_idx": int(state.gui_dialogue_msg_idx),
        "current_spell": int(state.current_spell),
        "power": float(state.power),
        "lives": float(state.lives),
        "bombs": float(state.bombs),
        "bombs_used": int(state.bombs_used),
        "deaths": int(state.deaths),
        "player_state": int(state.player_state),
    }


def state_snapshot(state: GameState) -> dict:
    return _state_dict(state)


def events_between(prev: dict | None, cur: dict) -> list[str]:
    """Rising-edge / delta events between two scalar snapshots.

    ``prev is None`` is the first decision of an episode: no events are emitted
    for the pre-existing state (a run that starts with power 128 is not a
    ``power_max`` event; that would make the event a state fact instead of a
    transition).
    """
    if prev is None:
        return []
    events: list[str] = []
    if cur["deaths"] > prev["deaths"]:
        events.append("death")
    if (cur["stage_clear_sequence_active"] and not prev["stage_clear_sequence_active"]) or (
        cur["gui_stage_clear_screen_state"] != 0 and prev["gui_stage_clear_screen_state"] == 0
    ) or cur["stage"] > prev["stage"]:
        events.append("stage_clear")
    if cur["game_cleared"] and not prev["game_cleared"]:
        events.append("game_clear")
    if cur["lives"] < 0.0 <= prev["lives"]:
        events.append("game_over")
    if cur["supervisor_state"] != spec.STATE_GAME_MANAGER and prev["supervisor_state"] == spec.STATE_GAME_MANAGER:
        events.append("left_game")
    if cur["stage_transition_state"] != prev["stage_transition_state"]:
        events.append("stage_transition")
    if cur["gui_dialogue_present"] and not prev["gui_dialogue_present"]:
        events.append("dialogue_start")
    elif prev["gui_dialogue_present"] and not cur["gui_dialogue_present"]:
        events.append("dialogue_end")
    if cur["power"] >= spec.POWER_MAX > prev["power"]:
        events.append("power_max")
    if cur["bombs_used"] > prev["bombs_used"]:
        events.append("bomb_used")
    if cur["current_spell"] != prev["current_spell"] and cur["current_spell"] >= 0:
        events.append("spell")
    return events


def _enemy_dict(enemy: EnemyInfo) -> dict:
    return {
        "slot": int(enemy.slot),
        "x": _f(enemy.x),
        "y": _f(enemy.y),
        "life": int(enemy.life),
        "max_life": int(enemy.max_life),
        "boss": bool(enemy.boss),
        "sub_id": None if enemy.sub_id is None else int(enemy.sub_id),
        "item_drop": int(enemy.item_drop),
        "boss_slot": int(enemy.boss_slot),
    }


def _item_dict(item: ItemInfo) -> dict:
    return {
        "slot": int(item.slot),
        "x": _f(item.x),
        "y": _f(item.y),
        "type": int(item.item_type),
        "state": int(item.state),
        "onscreen": bool(item.onscreen),
        "is_max_value": bool(item.is_max_value),
    }


def _boss_dict(boss: BossInfo) -> dict:
    return {
        "slot": int(boss.slot),
        "x": _f(boss.x),
        "y": _f(boss.y),
        "life": int(boss.life),
    }


def snapshot_body(snapshot: Snapshot) -> dict:
    """The JSON-serialisable body of one snapshot (without action/teacher)."""
    state = snapshot.state
    return {
        "clock": {
            "stage": int(state.stage),
            "stage_frames": int(state.stage_frames),
            "gameplay_frame": int(state.gameplay_frame),
            "run_frames": int(state.run_frames),
            "flags": int(state.flags) & 0xFFFFFFFF,
            "current_spell": int(state.current_spell),
        },
        "player": {
            "x": _f(state.player_x),
            "y": _f(state.player_y),
            "vx": _f(state.player_vx),
            "vy": _f(state.player_vy),
            "state": int(state.player_state),
            "focus_mode": int(state.focus_mode),
            "hurtbox_half_x": _f(state.hurtbox_half_x),
            "hurtbox_half_y": _f(state.hurtbox_half_y),
        },
        "resources": {
            "power": _f(state.power),
            "power_max": spec.POWER_MAX,
            "lives": _f(state.lives),
            "bombs": _f(state.bombs),
            "bombs_used": int(state.bombs_used),
            "deaths": int(state.deaths),
            "score": int(state.score),
            "graze": int(state.graze),
        },
        "bosses": [_boss_dict(b) for b in (snapshot.bosses or [])],
        "enemies": [_enemy_dict(e) for e in (snapshot.enemies or [])],
        "items": [_item_dict(i) for i in (snapshot.items or [])],
        "shot": None
        if snapshot.shot is None
        else {
            "point_item_value_line": _f(snapshot.shot.point_item_value_line),
            "item_collection_box_size": _f(snapshot.shot.item_collection_box_size),
            "item_auto_collect_speed": _f(snapshot.shot.item_auto_collect_speed),
        },
        "frame_events": None
        if snapshot.frame_event_flags is None
        else {
            "bits": int(snapshot.frame_event_flags),
            "item_collected": bool(snapshot.frame_event_flags & spec.FRAME_EVENT_ITEM_COLLECTED),
        },
        "schema_status": dict(snapshot.status),
    }


def bullets_summary(snapshot: Snapshot) -> dict:
    payload = snapshot.bullets
    count = int(payload["position"].shape[0]) if payload is not None else None
    counter = int(snapshot.state.active_bullets)
    return {
        "count": count,
        "counter": counter,
        # ``activeBulletCount`` is rewritten in place during the game's frame
        # update, so a read can land mid-update.  It is recorded as a diagnostic
        # (see contract §7.4 amendment) but is not a completeness criterion.
        "counter_delta": None if count is None else int(count) - counter,
        "sidecar_index": None,
    }


def build_record(
    episode: str,
    decision: int,
    frameskip: int,
    frame_pre: int,
    wall_ms: int,
    pre_state: dict | None,
    snapshot: Snapshot,
    action: np.ndarray | list[int],
    teacher: dict | None = None,
    controller: dict | None = None,
    events: list[str] | None = None,
    bullets_index: int | None = None,
) -> dict:
    """Build one contract-compliant decision record (no I/O)."""
    state = snapshot.state
    record: dict[str, Any] = {
        "schema": TRACE_SCHEMA,
        "episode": episode,
        "decision": int(decision),
        "frameskip": int(frameskip),
        "frame_pre": int(frame_pre),
        "frame_post": int(state.gameplay_frame),
        "wall_ms": int(wall_ms),
        "schema_status": dict(snapshot.status),
        **snapshot_body(snapshot),
        "action": [int(v) for v in np.asarray(action, dtype=np.int64).reshape(-1)],
        "events": list(events if events is not None else events_between(pre_state, _state_dict(state))),
    }
    if teacher is not None:
        record["teacher"] = teacher
    if controller is not None:
        record["controller"] = controller
    bullets = bullets_summary(snapshot)
    if bullets_index is not None:
        bullets["sidecar_index"] = int(bullets_index)
    record["bullets"] = bullets
    complete, reasons = assess_complete(record, prev=None)
    record["complete"] = complete
    record["incomplete_reasons"] = reasons
    return record


def assess_complete(record: dict, prev: dict | None = None) -> tuple[bool, list[str]]:
    """Contract §7.4 completeness predicate.

    ``prev`` supplies the continuity checks (decision/frame/wall); without it
    only the record-local checks run.  All failures are returned, not just the
    first, so a report can attribute a drop in the completion rate.
    """
    reasons: list[str] = []
    frame_pre = record.get("frame_pre")
    frame_post = record.get("frame_post")
    frameskip = int(record.get("frameskip", 0))
    if frame_pre is None or frame_post is None:
        reasons.append("frame_missing")
    elif ((int(frame_post) - int(frame_pre)) & 0xFFFFFFFF) != frameskip:
        reasons.append("frame_mismatch")

    status = record.get("schema_status", {})
    for name in REQUIRED_GROUPS:
        if status.get(name, "error") == "error":
            reasons.append(f"schema_{name}_error")

    player = record.get("player", {})
    px, py = player.get("x"), player.get("y")
    if not (_finite(px) and _finite(py)):
        reasons.append("player_not_finite")
    else:
        in_x = spec.MOVE_AREA_X - 8.0 <= float(px) <= spec.MOVE_AREA_X + spec.MOVE_AREA_W + 8.0
        in_y = spec.MOVE_AREA_Y - 8.0 <= float(py) <= spec.MOVE_AREA_Y + spec.MOVE_AREA_H + 8.0
        if not (in_x and in_y):
            reasons.append("player_outside_move_area")

    bullets = record.get("bullets", {})
    count = bullets.get("count")
    if count is None:
        reasons.append("bullet_count_missing")
    elif not 0 <= int(count) <= spec.BULLET_COUNT:
        reasons.append("bullet_count_out_of_range")

    if prev is not None:
        if int(record.get("decision", -1)) != int(prev.get("decision", -2)) + 1:
            reasons.append("decision_not_contiguous")
        if record.get("frame_pre") != prev.get("frame_post"):
            reasons.append("frame_gap")
        if int(record.get("wall_ms", 0)) < int(prev.get("wall_ms", 0)):
            reasons.append("wall_not_monotonic")
    return (not reasons, reasons)


def steady_exclusion(record: dict) -> str | None:
    """Return why a record is excluded from the S0-2/S0-4 denominator.

    A decision is steady when the pre-state was ordinary gameplay; the record's
    own post-state must still be in gameplay.  Dialogue/stage-clear frames are
    excluded because the game intentionally stops advancing the frame clock.
    """
    clock = record.get("clock", {})
    player = record.get("player", {})
    if record.get("frame_pre") is None or record.get("frame_post") is None:
        return "frame_missing"
    if int(record.get("frame_post", 0)) <= 0:
        return "not_playing"
    events = set(record.get("events") or [])
    if events & {"stage_clear", "game_clear", "game_over", "left_game"}:
        return "terminal_transition"
    if events & {"dialogue_start", "dialogue_end", "stage_transition"}:
        return "paused_transition"
    flags = clock.get("flags")
    if flags is None:
        return "flags_missing"
    if int(flags) & spec.FLAG_STAGE_CLEAR_SEQUENCE:
        return "stage_clear_sequence"
    if player.get("state") not in (0, 3, None):
        return "player_spawning_or_dying"
    return None


def classify_terminal(
    reason: str,
    final_state: GameState | None,
    post_samples: list[dict] | None,
) -> dict:
    """Classify one termination with post-terminal evidence.

    Priority (contract §6.2): game_clear > stage_clear > game_over > left_game >
    max_steps > frame_stalled.  ``frame_stalled`` is an unknown, not a game
    event; the verifier fails S0-3 on it unless post-terminal evidence resolves
    the reason.
    """
    # Post-terminal polling is only evidence for a *game* terminal.  After a
    # max_steps truncation the probe stops acting while the game keeps running,
    # so a death in the polling window is not the reason this episode ended; it
    # must not turn a max_steps run into a game_over run (observed in S0 ep04/06).
    samples = [] if reason in ("max_steps", "probe_error") else (post_samples or [])
    game_clear = bool(final_state and final_state.game_cleared) or any(s.get("game_cleared") for s in samples)
    stage_advanced = any(
        int(s.get("stage", 0)) > int(final_state.stage) for s in samples
    ) if final_state is not None else False
    clear_sequence = bool(final_state and final_state.stage_clear_sequence_active) or any(
        s.get("stage_clear_sequence_active") for s in samples
    )
    clear_screen = bool(final_state and final_state.gui_stage_clear_screen_state != 0) or any(
        int(s.get("gui_stage_clear_screen_state", 0)) != 0 for s in samples
    )
    game_over = bool(final_state and final_state.lives < 0.0) or any(s.get("lives", 0.0) < 0.0 for s in samples)
    # TH08 stops the frame counter on the game-over screen with ``lives`` at
    # exactly 0.0 (not negative) and ``player_state`` dying/spawning; the v3.4.0
    # boss probe shows this on every teacher death (3 deaths, lives 0.0,
    # player_state 2 -> 1).  Without this branch S0-3 would label a real game
    # over as the unknown ``frame_stalled``.
    stalled_game_over = bool(
        final_state is not None
        and reason not in ("max_steps", "probe_error")
        and final_state.lives <= 0.0
        and final_state.player_state in (1, 2)
        and not clear_sequence
        and not clear_screen
        and not game_clear
        and not final_state.gui_dialogue_present
    )
    game_over = game_over or stalled_game_over or any(
        float(s.get("lives", 1.0)) <= 0.0 and int(s.get("player_state", 0)) in (1, 2)
        for s in samples
    )
    left_game = bool(
        final_state and final_state.supervisor_state != spec.STATE_GAME_MANAGER
    ) or any(int(s.get("supervisor_state", spec.STATE_GAME_MANAGER)) != spec.STATE_GAME_MANAGER for s in samples)

    if game_clear:
        classified = "game_clear"
    elif stage_advanced or clear_sequence or clear_screen:
        classified = "stage_clear"
    elif game_over:
        classified = "game_over"
    elif left_game:
        classified = "left_game"
    elif reason == "max_steps":
        classified = "max_steps"
    elif reason == "probe_error":
        classified = "probe_error"
    else:
        classified = "frame_stalled"
    evidence = {
        "reason": reason,
        "classified": classified,
        "game_clear": game_clear,
        "stage_advanced": stage_advanced,
        "clear_sequence": clear_sequence,
        "clear_screen": clear_screen,
        "game_over": game_over,
        "left_game": left_game,
        "post_samples": len(samples),
    }
    return evidence


def stable_split(episode: str, seed: str = "v350-s0", buckets: int = 4) -> str:
    """Whole-episode deterministic split; the same episode always lands once."""
    digest = hashlib.sha256(f"{seed}:{episode}".encode("utf-8")).hexdigest()
    return "validation" if int(digest[:8], 16) % buckets == buckets - 1 else "train"


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------


class TraceWriter:
    """Writes one episode's decision JSONL and bullet sidecar.

    Bullets are stored once per decision as a concatenated array + offsets so
    the offline region probe can rebuild ``(N, 5)`` snapshots without parsing
    megabytes of JSON.  ``np.savez`` (not compressed) is used on purpose: the
    decision loop is latency-sensitive and the sidecar is deleted after the
    derived evidence is written.
    """

    def __init__(self, out_dir: Path, episode: str):
        self.out_dir = Path(out_dir)
        self.episode = episode
        self.trace_path = self.out_dir / "trace" / f"{episode}.jsonl"
        self.bullets_path = self.out_dir / "bullets" / f"{episode}.npz"
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.bullets_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.trace_path.open("w", encoding="utf-8")
        self._offsets: list[int] = [0]
        self._positions: list[np.ndarray] = []
        self._velocities: list[np.ndarray] = []
        self._half: list[np.ndarray] = []
        self._slots: list[np.ndarray] = []
        self._counters: list[int] = []
        self._closed = False

    def write(self, record: dict, snapshot: Snapshot | None = None) -> None:
        if self._closed:
            raise RuntimeError("TraceWriter is closed")
        index = len(self._counters)
        if snapshot is not None and snapshot.bullets is not None:
            pos = np.asarray(snapshot.bullets["position"], dtype=np.float32).reshape(-1, 2)
            vel = np.asarray(snapshot.bullets["velocity"], dtype=np.float32).reshape(-1, 2)
            half = np.asarray(snapshot.bullets["half"], dtype=np.float32).reshape(-1)
            self._positions.append(pos)
            self._velocities.append(vel)
            self._half.append(half)
            # The pool slot id is what makes "this bullet is new" decidable.  A
            # position-only sidecar cannot separate a spawn from a survivor whose
            # motion is not exactly pos + vel*frameskip: measured spawn rate was
            # 11.3/decision at a 1.5 px match tolerance and 2.2/decision at 10 px,
            # with no convergence (see touhou_rl/bullet_track.py).  -1 means the
            # caller did not supply slot ids.
            slot_ids = snapshot.bullets.get("slot_id")
            if slot_ids is None:
                slots = np.full(pos.shape[0], -1, dtype=np.int32)
            else:
                slots = np.asarray(slot_ids, dtype=np.int32).reshape(-1)
                if slots.size != pos.shape[0]:
                    slots = np.full(pos.shape[0], -1, dtype=np.int32)
            self._slots.append(slots)
            self._offsets.append(self._offsets[-1] + int(pos.shape[0]))
        else:
            self._offsets.append(self._offsets[-1])
        self._counters.append(int(record.get("bullets", {}).get("counter", 0)))
        record.setdefault("bullets", {})["sidecar_index"] = index
        self._handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    def close(self) -> dict:
        if not self._closed:
            self._handle.close()
            positions = (
                np.concatenate(self._positions, axis=0) if self._positions else np.zeros((0, 2), np.float32)
            )
            velocities = (
                np.concatenate(self._velocities, axis=0) if self._velocities else np.zeros((0, 2), np.float32)
            )
            half = np.concatenate(self._half, axis=0) if self._half else np.zeros((0,), np.float32)
            slots = np.concatenate(self._slots, axis=0) if self._slots else np.zeros((0,), np.int32)
            np.savez(
                self.bullets_path,
                positions=positions.astype(np.float32),
                velocities=velocities.astype(np.float32),
                half=half.astype(np.float32),
                slots=slots.astype(np.int32),
                offsets=np.asarray(self._offsets, dtype=np.int64),
                counters=np.asarray(self._counters, dtype=np.int32),
            )
            self._closed = True
        return {
            "trace": str(self.trace_path),
            "bullets": str(self.bullets_path),
            "decisions": len(self._counters),
            "bullets_total": int(self._offsets[-1]),
            "slot_ids": bool(self._slots),
        }


def write_episode(out_dir: Path, episode: dict) -> None:
    path = Path(out_dir) / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(episode, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_episodes(out_dir: Path) -> list[dict]:
    path = Path(out_dir) / "episodes.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
