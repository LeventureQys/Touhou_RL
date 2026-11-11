"""obs-v3 teacher-shard reader (T2/D-14 -> T3 D1, T4).

The v1 loader (:func:`touhou_rl.v400.student.dataset.load_teacher_shard`) is
frozen around ``play64``/``hud`` and the obs-v1 digest, so it must keep refusing an
obs-v3 shard.  This module is the v3 counterpart and nothing else: it reads the
same ``shard.json`` schema, the same ``labels.jsonl`` row schema, the same
observation instant and the same window arithmetic, and adds

* the 96x112 play plane (``obs/play96x112.npy``, ``(N,4,112,96)`` uint8),
* the 8x18 action history (``obs/act_hist.npy``, ``(N,8,18)`` uint8), and
* the obs-v3 spec digest gate (:func:`touhou_rl.v400.student.obs_spec_v3.check_obs_spec_v3`).

Two checks make "the stored ``act_hist`` is the history of the stored trajectory"
a proved statement instead of a description:

1. every label row is re-read and validated with the *same* function the v1 loader
   uses (``dataset._read_teacher_obs_labels``): decision index, frame arithmetic
   ``frame == i * frames_per_decision``, ``render_frames == frames[d-4..d-1]``,
   episode-start padding only on decision 0;
2. the history is rebuilt from those rows with obs-v2's frozen rule
   (``chosen_plan.first_action`` for a teacher-driven shard,
   ``dagger.executed_cand`` for a DAgger shard, cleared at the shard start and at
   every stage change) and compared **byte for byte** with the stored plane.

``require_observations=True`` is the D1 training gate: a label row that cannot be
resolved to a readable v3 shard row raises :class:`DatasetInsufficient` instead of
letting a training script run on labels without pixels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from touhou_rl.v400.student import dataset, obs_spec_v2, obs_spec_v3
from touhou_rl.v400.student.dataset import (
    DatasetError,
    DatasetInsufficient,
    STUDENT_SAMPLE_SCHEMA,
    TEACHER_OBSERVATION_TIME,
    TEACHER_SHARD_SCHEMA,
    V0_DISCLOSURE,
    _read_teacher_obs_labels,
    _safe_relative,
    _sha256_file,
    _validate_teacher_decision,
    scan_teacher_labels,
)

#: Shard/plane contract of an obs-v3 shard (same shard schema, different planes).
SHARD_SCHEMA = TEACHER_SHARD_SCHEMA
PLAY_PLANE = obs_spec_v3.PLAY_PLANE                     # "play96x112"
PLAY_FILE = "play96x112.npy"
HUD_FILE = "hud.npy"
ACT_HIST_FILE = "act_hist.npy"
ACT_HIST_META_KEY = "act_hist"


class ObsV3DatasetError(DatasetError):
    """An obs-v3 shard violates its contract."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ObsV3DatasetError(f"missing shard file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ObsV3DatasetError(f"{path}: not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ObsV3DatasetError(f"{path}: not a JSON object")
    return payload


def expected_action_history(records: Sequence[dict[str, Any]]) -> np.ndarray:
    """``(N,8,18)`` uint8 history rebuilt from validated label records."""

    try:
        cand, stage = obs_spec_v2.reconstruct_from_label_rows(list(records))
        history = obs_spec_v2.build_action_history(cand, stage)
        obs_spec_v2.check_action_history(history)
    except obs_spec_v2.ObsSpecV2Error as error:
        raise ObsV3DatasetError(
            f"act_hist cannot be reconstructed from the label rows: {error}") from error
    return history


def load_teacher_shard_v3(shard_dir: str | Path, *, verify_sha: bool = True,
                          verify_history: bool = True) -> "TeacherShardV3":
    """Load one obs-v3 shard, or raise :class:`DatasetError`.

    Refuses: a missing/incomplete ``shard.json``, a spec version/digest that is not
    the frozen obs-v3 one, plane shapes/sizes/sha256 that do not match
    ``shard.json``, an alignment statement that is not the frozen one, a label row
    whose decision/frame/render arithmetic disagrees, and (``verify_history``) an
    ``act_hist`` plane that is not the exact history of the stored actions.
    """

    shard_dir = Path(shard_dir)
    payload = _load_json(shard_dir / "shard.json")
    if payload.get("schema") != SHARD_SCHEMA:
        raise ObsV3DatasetError(
            f"{shard_dir}: shard schema {payload.get('schema')!r} != {SHARD_SCHEMA!r}")
    if payload.get("complete") is not True:
        raise ObsV3DatasetError(f"{shard_dir}: shard is not complete; refusing to load")
    try:
        obs_spec_v3.check_obs_spec_v3(payload.get("obs_spec_version"),
                                      payload.get("obs_spec_digest"))
    except obs_spec_v3.ObsSpecV3Error as error:
        raise ObsV3DatasetError(f"{shard_dir}: obs spec check failed: {error}") from error
    alignment = payload.get("alignment") or {}
    for key, expected in (
        ("observation_time", TEACHER_OBSERVATION_TIME),
        ("stack", obs_spec_v3.FRAME_STACK),
        ("stack_order", obs_spec_v3.STACK_ORDER),
        ("decision_frames", obs_spec_v3.DECISION_FRAMES),
        ("p3_window_last_row", "i = decision_frame - 1"),
        ("action_history_excludes_current_action", True),
    ):
        if alignment.get(key) != expected:
            raise ObsV3DatasetError(
                f"{shard_dir}: alignment {key}={alignment.get(key)!r} != {expected!r}; "
                "a shard recorded under another observation instant cannot be aligned "
                "with the teacher labels")
    if "frames[i-3..i] -> mask[i+1]" not in str(alignment.get("p3_window")):
        raise ObsV3DatasetError(f"{shard_dir}: alignment does not state the P3 window")
    if "action_history" not in alignment:
        raise ObsV3DatasetError(f"{shard_dir}: alignment does not state act_hist")
    decisions = payload.get("decision_count")
    if not isinstance(decisions, int) or decisions <= 0:
        raise ObsV3DatasetError(
            f"{shard_dir}: decision_count {decisions!r} is not positive")
    frames_per_decision = payload.get("frames_per_decision")
    if (not isinstance(frames_per_decision, int)
            or frames_per_decision < obs_spec_v3.DECISION_FRAMES
            or frames_per_decision % obs_spec_v3.DECISION_FRAMES):
        raise ObsV3DatasetError(
            f"{shard_dir}: frames_per_decision {frames_per_decision!r} is not a "
            f"positive multiple of {obs_spec_v3.DECISION_FRAMES}")
    obs = payload.get("obs") or {}
    planes: dict[str, np.ndarray] = {}
    #: (meta key, expected shard-relative file, expected shape) — the file names are
    #: part of the contract so a renamed plane cannot be read as another field.
    expected_planes = (
        (PLAY_PLANE, f"obs/{PLAY_FILE}",
         (decisions, obs_spec_v3.FRAME_STACK, *obs_spec_v3.PLAY_SHAPE)),
        (obs_spec_v3.HUD_PLANE, f"obs/{HUD_FILE}",
         (decisions, *obs_spec_v3.HUD_SHAPE)),
        (ACT_HIST_META_KEY, f"obs/{ACT_HIST_FILE}",
         (decisions, *obs_spec_v3.ACTION_HISTORY_SHAPE)),
    )
    for name, expected_file, shape in expected_planes:
        meta = obs.get(name) or {}
        if str(meta.get("path", "")) != expected_file:
            raise ObsV3DatasetError(
                f"{shard_dir}: obs.{name}.path {meta.get('path')!r} != "
                f"{expected_file!r}")
        path = shard_dir / _safe_relative(meta.get("path", ""))
        if not path.is_file():
            raise ObsV3DatasetError(f"{shard_dir}: missing plane {path}")
        if verify_sha:
            recorded = meta.get("sha256")
            if not recorded:
                raise ObsV3DatasetError(f"{path}: no sha256 recorded in shard.json")
            actual = _sha256_file(path)
            if actual != recorded:
                raise ObsV3DatasetError(f"{path}: sha256 {actual} != recorded {recorded}")
        size = path.stat().st_size
        if int(meta.get("bytes", -1)) != size:
            raise ObsV3DatasetError(
                f"{path}: {size} B != shard.json bytes {meta.get('bytes')!r}")
        try:
            array = np.load(path, mmap_mode="r")
        except (ValueError, OSError) as error:
            raise ObsV3DatasetError(f"{path}: not a readable npy ({error})") from error
        if array.dtype != np.uint8:
            raise ObsV3DatasetError(f"{path}: dtype {array.dtype} is not uint8")
        if array.shape != shape:
            raise ObsV3DatasetError(f"{path}: shape {array.shape} != {shape}")
        planes[name] = array
    records = _read_teacher_obs_labels(
        shard_dir / "labels.jsonl",
        decisions=decisions,
        frames_per_decision=frames_per_decision,
        shard_key=str(payload.get("shard")),
    )
    if verify_history:
        rebuilt = expected_action_history(records)
        stored = np.asarray(planes[ACT_HIST_META_KEY])
        if stored.shape != rebuilt.shape or not np.array_equal(stored, rebuilt):
            raise ObsV3DatasetError(
                f"{shard_dir}: stored act_hist is not the history of the stored "
                "actions (rebuild mismatch); the shard cannot be trained on")
    return TeacherShardV3(
        dir=shard_dir,
        meta=payload,
        records=records,
        play=planes[PLAY_PLANE],
        hud=planes[obs_spec_v3.HUD_PLANE],
        act_hist=planes[ACT_HIST_META_KEY],
        frames_per_decision=frames_per_decision,
    )


class TeacherShardV3:
    """One loaded obs-v3 shard: labels + mem-mapped v3 planes.

    ``shard[i]`` is one P3 teacher sample: ``play (4,112,96)`` uint8,
    ``hud (56,160)`` uint8, ``act_hist (8,18)`` uint8, the teacher label
    (``scores[18]``/``verified_mask[18]``/``bomb``/``margin``/``chosen_plan``) and
    the policy-facing auxiliary label ``verified_actions`` — the candidates the
    teacher verified as surviving (D-14 / ``obs_spec_v3.POLICY_AUX_LABEL``).
    """

    obs_version = obs_spec_v3.OBS_SPEC_VERSION

    def __init__(
        self,
        *,
        dir: Path,
        meta: dict[str, Any],
        records: Sequence[dict[str, Any]],
        play: np.ndarray,
        hud: np.ndarray,
        act_hist: np.ndarray,
        frames_per_decision: int,
    ) -> None:
        self.dir = Path(dir)
        self.meta = dict(meta)
        self.records = list(records)
        self.play = play
        self.hud = hud
        self.act_hist = act_hist
        self.frames_per_decision = int(frames_per_decision)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        mask = np.asarray(record["verified_mask"], dtype=bool)
        sample = {
            "schema": STUDENT_SAMPLE_SCHEMA,
            "kind": "teacher-observed-obs-v3",
            "decision": int(index),
            "frame": int(record["frame"]),
            "play": np.asarray(self.play[index], dtype=np.uint8),
            "hud": np.asarray(self.hud[index], dtype=np.uint8),
            "act_hist": np.asarray(self.act_hist[index], dtype=np.uint8),
            "verified_actions": [int(cand) for cand in np.nonzero(mask)[0]],
            "scores": record["scores"],
            "verified_mask": record["verified_mask"],
            "verified_count": int(record["verified_count"]),
            "bomb": int(record["bomb"]),
            "margin": record["margin"],
            "chosen_plan": record["chosen_plan"],
            "stage": int(record.get("stage", 0)),
            "difficulty": int(record.get("difficulty", 0)),
            "obs_spec_version": self.meta.get("obs_spec_version"),
            "obs_spec_digest": self.meta.get("obs_spec_digest"),
            "shard": self.meta.get("shard"),
        }
        return sample

    def normalized(self, index: int) -> dict[str, np.ndarray]:
        """Model-ready arrays: play ``(4,112,96)``, hud ``(1,56,160)``, act_hist ``(144,)``."""

        sample = self[index]
        return {
            "play": obs_spec_v3.normalize_play(sample["play"]),
            "hud": obs_spec_v3.normalize_hud(sample["hud"]),
            "act_hist": obs_spec_v3.normalize_action_history(sample["act_hist"]),
        }

    def report(self) -> dict[str, Any]:
        obs = self.meta.get("obs") or {}
        return {
            "shard": self.meta.get("shard"),
            "dir": str(self.dir),
            "obs_version": self.meta.get("obs_version"),
            "reader_version": self.obs_version,
            "decisions": len(self.records),
            "frames": self.meta.get("stage_frames"),
            "frames_per_decision": self.frames_per_decision,
            "obs_spec_version": self.meta.get("obs_spec_version"),
            "obs_spec_digest": self.meta.get("obs_spec_digest"),
            "alignment": (self.meta.get("alignment") or {}).get("observation_time"),
            "play_shape": list(self.play.shape),
            "hud_shape": list(self.hud.shape),
            "act_hist_shape": list(self.act_hist.shape),
            "obs_bytes": obs.get("bytes_total"),
            "obs_bytes_per_decision": obs.get("bytes_per_decision"),
            "labels_sha256": (self.meta.get("labels") or {}).get("sha256"),
            "play_sha256": (obs.get(PLAY_PLANE) or {}).get("sha256"),
            "hud_sha256": (obs.get("hud") or {}).get("sha256"),
            "act_hist_sha256": (obs.get(ACT_HIST_META_KEY) or {}).get("sha256"),
            "complete": self.meta.get("complete"),
        }


def load_teacher_samples_v3(
    paths: Iterable[str | Path],
    *,
    min_samples: int = dataset.MIN_TEACHER_SAMPLES,
    obs_root: str | Path | None = None,
    require_observations: bool = False,
) -> dict[str, Any]:
    """Resolve every obs-v3 label row to its stored observation arrays.

    ``require_observations=True`` is the D1 gate: a label without a readable v3
    shard row raises :class:`DatasetInsufficient` (with the count and the first
    unresolved reference) instead of returning labels a training script could use.
    """

    paths = list(paths)
    if not paths:
        raise DatasetInsufficient("no teacher label files given")
    scan = scan_teacher_labels(paths)
    if scan["with_observation_reference"] == 0:
        raise DatasetInsufficient(
            "teacher labels carry no observation reference: "
            f"{scan['decisions']} decisions across {len(scan['files'])} file(s), "
            "0 with an 'obs'/'play'/'shard' key; an obs-v3 pixel student cannot be "
            "trained from labels whose frames were not stored. " + V0_DISCLOSURE)
    if scan["with_observation_reference"] < min_samples:
        raise DatasetInsufficient(
            f"teacher labels are insufficient: {scan['with_observation_reference']} "
            f"usable decision(s) < required {min_samples}. " + V0_DISCLOSURE)
    records: list[dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("kind") != "decision":
                    continue
                record = _validate_teacher_decision(payload, f"{path}:{line_number}")
                if not record["observation"]:
                    continue
                records.append(record)
    shards: dict[str, TeacherShardV3] = {}
    unresolved: list[dict[str, Any]] = []
    if obs_root is not None:
        for record in records:
            reference = record["observation"].get("obs") or {}
            key = reference.get("shard")
            row = reference.get("row")
            if not key or row is None:
                unresolved.append({"reason": "no shard/row in obs reference",
                                   "obs": reference})
                continue
            shard_dir = Path(obs_root) / "shards" / str(key)
            if not (shard_dir / "shard.json").is_file():
                unresolved.append({"reason": f"no shard.json under {shard_dir}",
                                   "shard": key})
                continue
            cache_key = str(shard_dir)
            if cache_key not in shards:
                shards[cache_key] = load_teacher_shard_v3(shard_dir)
            shard = shards[cache_key]
            index = int(row)
            if not 0 <= index < len(shard):
                raise ObsV3DatasetError(
                    f"{shard_dir}: obs row {index} is outside the shard's "
                    f"{len(shard)} decisions")
            sample = shard[index]
            record["obs_arrays"] = {
                "shard": shard,
                "row": index,
                "play": sample["play"],
                "hud": sample["hud"],
                "act_hist": sample["act_hist"],
                "verified_actions": sample["verified_actions"],
                "play_shape": list(obs_spec_v3.PLAY_SHAPE),
                "play_extent_wh": list(obs_spec_v3.PLAY_EXTENT_WH),
                "hud_shape": list(obs_spec_v3.HUD_SHAPE),
                "act_hist_shape": list(obs_spec_v3.ACTION_HISTORY_SHAPE),
                "stack": obs_spec_v3.FRAME_STACK,
            }
        if require_observations and unresolved:
            raise DatasetInsufficient(
                f"teacher label(s) reference {len(unresolved)} observation row(s) "
                f"that no readable obs-v3 shard under {obs_root} provides "
                f"(first: {unresolved[0]}). " + V0_DISCLOSURE)
    obs_shapes = {
        "play": list(obs_spec_v3.PLAY_SHAPE),
        "hud": list(obs_spec_v3.HUD_SHAPE),
        "act_hist": list(obs_spec_v3.ACTION_HISTORY_SHAPE),
        "stack": obs_spec_v3.FRAME_STACK,
    }
    return {
        "schema": "v400-teacher-label-obs-v3-1",
        "kind": "teacher-obs-v3",
        "samples": records,
        "count": len(records),
        "obs_spec_version": obs_spec_v3.OBS_SPEC_VERSION,
        "obs_spec_digest": obs_spec_v3.obs_spec_v3_digest(),
        "obs_shards": {key: shard.report() for key, shard in shards.items()},
        "obs_shapes": obs_shapes if shards else None,
        "obs_unresolved": unresolved,
        "decision_frames": obs_spec_v3.DECISION_FRAMES,
        "alignment": TEACHER_OBSERVATION_TIME if shards else None,
        "policy_aux_label": obs_spec_v3.POLICY_AUX_LABEL,
        "disclosure": (
            "obs-v3 teacher-soft-label samples with stored 96x112 play planes and "
            "8x18 action history; the occupancy F1 of input B is 0.8502 (< 0.90), "
            "so acceptance is the closed-loop criterion (D-14), not the probe F1"),
    }


__all__ = [
    "ACT_HIST_FILE",
    "ACT_HIST_META_KEY",
    "HUD_FILE",
    "ObsV3DatasetError",
    "PLAY_FILE",
    "PLAY_PLANE",
    "SHARD_SCHEMA",
    "TeacherShardV3",
    "expected_action_history",
    "load_teacher_samples_v3",
    "load_teacher_shard_v3",
]
