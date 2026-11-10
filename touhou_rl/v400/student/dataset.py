"""Student dataset contract (P3-WP1-a) and its fail-closed validators.

Two sample kinds exist in v4.0.0:

``bc-v0`` (behaviour cloning, what this slice can actually train)
    ``(play64[4], hud[56x160], cand[18] hard label, bomb[2])`` built from the
    P2 300k-frame collection, where the label is the *collection policy's*
    button mask (``random`` / ``still`` / ``dive``).  These are **not teacher
    labels and never a P3/G3 score**; every report carries :data:`V0_DISCLOSURE`.

``teacher`` (the P3 target, not yet trainable)
    ``(pixels, TeacherLabel(scores[18], bomb, chosen_plan, margin, verified_mask))``.
    The loader validates the schema but cannot invent observations: the stored
    teacher JSONL (``.tmp_v400_teacher_child_stage1.jsonl`` and siblings) has no
    observation reference and only 1500 decisions, so
    :func:`load_teacher_samples` fails closed with the real counts instead of
    silently training on nothing.

The dataset window convention is fixed here: row ``i`` of a shard stores the
mask that produced the stored frame ``i`` (the collector writes ``mask`` before
``STEP`` and reads pixels after it), so a sample is
``frames[i-3..i] -> mask[i+1]``.  Labelling a stacked frame with the mask that
*produced* it would leak the action through its own immediate effect, so this
module always predicts the next decision's mask.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from touhou_rl.v400.gym.actions import mask_to_cand
from touhou_rl.v400.student.obs_spec import FRAME_STACK, HUD_SHAPE

from touhou_rl.v400.labels import spec as label_spec
from touhou_rl.v400.student import obs_spec

INDEX_SCHEMA = "v400-p2det-index-1"
SPLIT_SCHEMA = "v400-p2det-split-1"
SHARD_SCHEMA = "v400-p2det-shard-1"
STUDENT_SAMPLE_SCHEMA = "v400-student-sample-1"
TEACHER_LABEL_SCHEMA = "v400-teacher-label-1"
LABEL_VERSION = label_spec.LABEL_VERSION
#: Teacher **observation** shard (D0), written by
#: ``touhou_rl/v400/teacher/obs.py``.  The schema strings are duplicated here on
#: purpose: the loader must fail when the writer's contract changes, not follow it.
TEACHER_OBS_SCHEMA = "v400-teacher-obs-1"
TEACHER_SHARD_SCHEMA = "v400-teacher-obs-shard-1"
TEACHER_MANIFEST_SCHEMA = "v400-teacher-obs-manifest-1"
#: The only accepted observation instant.  ``decision_time_before_commit`` means the
#: stored 4-frame stack ends at ``decision_frame - 1`` and the label is the action
#: applied at ``decision_frame`` — i.e. the P3 ``frames[i-3..i] -> mask[i+1]`` window
#: with ``i = decision_frame - 1``.  A shard recorded *after* the commit would label
#: an already-executed action and leak its effect into its own input, so it is refused.
TEACHER_OBSERVATION_TIME = "decision_time_before_commit"
PLANE_SHAPES: dict[str, tuple[int, int]] = {
    obs_spec.PLAY_PLANE: obs_spec.PLAY_SHAPE,
    obs_spec.HUD_PLANE: obs_spec.HUD_SHAPE,
}

#: P3-WP1 target is ~2M frames and DAgger labels 500k decisions per round.  A
#: teacher dataset below this is refused as *training* input, not just warned
#: about: silently training a student on 1500 samples would produce a number
#: that looks like P3 progress and is not.
MIN_TEACHER_SAMPLES = 50_000

#: Mandatory provenance string on every BC-v0 artefact.
V0_DISCLOSURE = (
    "V0 = behaviour cloning of the random/still/dive collection policies. "
    "These labels are the collector's own button masks, not TeacherLabel scores, "
    "and no P3 or G3 result may be claimed from this checkpoint."
)


class DatasetError(RuntimeError):
    """A dataset violates the contract (wrong schema, shape, or meaning)."""


class DatasetInsufficient(DatasetError):
    """A dataset exists but is too small / has no usable observations."""


@dataclass(frozen=True)
class ShardEntry:
    shard: str
    attempt: str
    dir: Path
    frames: int
    source: str
    stage: int
    difficulty: int
    seed: int
    split: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "shard": self.shard,
            "attempt": self.attempt,
            "dir": str(self.dir),
            "frames": self.frames,
            "source": self.source,
            "stage": self.stage,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "split": self.split,
        }


# ---------------------------------------------------------------------------
# index / split
# ---------------------------------------------------------------------------
def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetError(f"dataset file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(f"dataset file is not JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetError(f"dataset file is not a JSON object: {path}")
    return payload


def load_index(path: str | Path) -> dict[str, Any]:
    """Load and validate a P2 ``index.json`` before any plane is touched."""

    payload = _read_json(path)
    if payload.get("schema") != INDEX_SCHEMA:
        raise DatasetError(f"index schema {payload.get('schema')!r} != {INDEX_SCHEMA!r}")
    if payload.get("label_version") != LABEL_VERSION:
        raise DatasetError(
            f"index label_version {payload.get('label_version')!r} != {LABEL_VERSION!r}; "
            "mixing label meanings is refused"
        )
    shapes = payload.get("plane_shapes") or {}
    for name, expected in PLANE_SHAPES.items():
        got = tuple(shapes.get(name, ()))
        if got != expected:
            raise DatasetError(f"index plane {name} shape {got} != {expected}")
    frames = payload.get("frames_total")
    if not isinstance(frames, int) or frames <= 0:
        raise DatasetError(f"index frames_total is not a positive int: {frames!r}")
    if not payload.get("shards"):
        raise DatasetError("index has no shards")
    return payload


def load_split(path: str | Path, index: dict[str, Any] | None = None) -> list[ShardEntry]:
    """Load a P2 ``split.json`` and return one validated entry per shard."""

    payload = _read_json(path)
    if payload.get("schema") != SPLIT_SCHEMA:
        raise DatasetError(f"split schema {payload.get('schema')!r} != {SPLIT_SCHEMA!r}")
    if payload.get("label_version") != LABEL_VERSION:
        raise DatasetError(
            f"split label_version {payload.get('label_version')!r} != {LABEL_VERSION!r}")
    if "stage" not in str(payload.get("machine", "")):
        raise DatasetError(f"split is not stage-level: {payload.get('machine')!r}")
    if index is not None:
        index_shards = {row.get("shard") for row in index.get("shards", [])}
        split_shards = {row.get("shard") for row in payload.get("shards", [])}
        if index_shards and not split_shards <= index_shards:
            missing = sorted(split_shards - index_shards)[:3]
            raise DatasetError(f"split shards not in index: {missing}")
    stage_splits = payload.get("stage_splits") or {}
    seen_stages: dict[int, str] = {}
    for split_name, stages in stage_splits.items():
        for stage in stages:
            if int(stage) in seen_stages:
                raise DatasetError(
                    f"stage {stage} appears in two splits: {seen_stages[int(stage)]!r} "
                    f"and {split_name!r}")
            seen_stages[int(stage)] = split_name
    entries: list[ShardEntry] = []
    for raw in payload.get("shards") or []:
        try:
            entry = ShardEntry(
                shard=str(raw["shard"]),
                attempt=str(raw.get("attempt", "attempt-01")),
                dir=Path(raw["dir"]),
                frames=int(raw["frames"]),
                source=str(raw["source"]),
                stage=int(raw["stage"]),
                difficulty=int(raw["difficulty"]),
                seed=int(raw["seed"]),
                split=str(raw["split"]),
            )
        except KeyError as exc:
            raise DatasetError(f"split shard row missing {exc}: {raw!r}") from exc
        if entry.frames <= 0:
            raise DatasetError(f"shard {entry.shard} has frames={entry.frames}")
        if entry.split not in ("train", "val", "test"):
            raise DatasetError(f"shard {entry.shard} has unknown split {entry.split!r}")
        expected = seen_stages.get(entry.stage)
        if expected is not None and expected != entry.split:
            raise DatasetError(
                f"shard {entry.shard} stage {entry.stage} is {entry.split!r} but "
                f"stage_splits says {expected!r}")
        entries.append(entry)
    if not entries:
        raise DatasetError("split has no shards")
    for split_name in ("train", "val", "test"):
        if not any(entry.split == split_name for entry in entries):
            raise DatasetError(f"split {split_name!r} is empty")
    return entries


# ---------------------------------------------------------------------------
# BC-v0 windows
# ---------------------------------------------------------------------------
def _plane_path(shard_dir: Path, name: str) -> Path:
    return Path(shard_dir) / f"{name}.u8"


def _read_labels(path: Path, expected_frames: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    except FileNotFoundError as exc:
        raise DatasetError(f"missing labels: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(f"bad JSON line in {path}: {exc}") from exc
    if len(rows) != expected_frames:
        raise DatasetError(
            f"{path} has {len(rows)} label rows but the shard stores {expected_frames} frames")
    for index, row in enumerate(rows):
        if "msk" not in row:
            raise DatasetError(f"{path} row {index} has no 'msk' action mask")
        if int(row.get("i", index)) != index:
            raise DatasetError(f"{path} row {index} has i={row.get('i')!r}")
    return rows


def decode_mask(mask: int) -> tuple[int, int]:
    """Button mask -> ``(cand, bomb)``; raises ``DatasetError`` if not encodable."""

    try:
        return mask_to_cand(int(mask))
    except ValueError as exc:
        raise DatasetError(f"mask {mask!r} is not a student action: {exc}") from exc


class BehaviorCloneDataset:
    """Windows of ``bc-v0`` samples over one dataset split.

    Each item is a dict with ``play (4,64,64) uint8``, ``hud (56,160) uint8``,
    ``move`` int, ``bomb`` int, and provenance fields.  Planes are memory-mapped
    and labels are parsed once per shard, so the 6.8 GiB dataset is never copied.
    """

    def __init__(
        self,
        entries: Sequence[ShardEntry],
        *,
        stack: int = FRAME_STACK,
        label_offset: int = 1,
        require_hud: bool = True,
        cache_shards: int = 8,
        verify_sizes: bool = True,
    ) -> None:
        if stack < 1:
            raise DatasetError("stack must be >= 1")
        if label_offset != 1:
            raise DatasetError(
                "label_offset is frozen at 1: sample frames[i-stack+1..i] -> mask[i+1]")
        self.entries = list(entries)
        self.stack = int(stack)
        self.label_offset = int(label_offset)
        self.require_hud = bool(require_hud)
        self.cache_shards = int(cache_shards)
        self._counts: list[int] = []
        self._starts = np.zeros(len(self.entries) + 1, dtype=np.int64)
        for position, entry in enumerate(self.entries):
            count = entry.frames - self.stack
            if count < 0:
                raise DatasetError(f"shard {entry.shard} has {entry.frames} frames, "
                                   f"fewer than stack={self.stack}")
            self._counts.append(count)
            self._starts[position + 1] = self._starts[position] + count
        if self._starts[-1] == 0:
            raise DatasetError("dataset has no valid windows")
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        if verify_sizes:
            self._verify_plane_sizes()

    # -- validation -------------------------------------------------------
    def _verify_plane_sizes(self) -> None:
        for entry in self.entries:
            for name, shape in PLANE_SHAPES.items():
                if name == obs_spec.HUD_PLANE and not self.require_hud:
                    continue
                path = _plane_path(entry.dir, name)
                if not path.is_file():
                    raise DatasetError(f"missing plane {path}")
                expected = entry.frames * shape[0] * shape[1]
                actual = path.stat().st_size
                if actual != expected:
                    raise DatasetError(
                        f"{path} is {actual} B, expected {expected} B "
                        f"({entry.frames} frames x {shape[0]}x{shape[1]})")

    # -- sample access ----------------------------------------------------
    def __len__(self) -> int:
        return int(self._starts[-1])

    def _entry_and_frame(self, index: int) -> tuple[int, int]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        entry_index = int(np.searchsorted(self._starts, index, side="right") - 1)
        offset = int(index - self._starts[entry_index])
        return entry_index, offset + self.stack - 1

    def _shard_state(self, entry_index: int) -> dict[str, Any]:
        cached = self._cache.get(entry_index)
        if cached is not None:
            self._cache.move_to_end(entry_index)
            return cached
        entry = self.entries[entry_index]
        labels = _read_labels(entry.dir / "labels.jsonl", entry.frames)
        cand = np.full(entry.frames, -1, dtype=np.int16)
        bomb = np.zeros(entry.frames, dtype=np.int8)
        for row_index, row in enumerate(labels):
            try:
                cand_value, bomb_value = decode_mask(int(row["msk"]))
            except DatasetError:
                continue
            cand[row_index] = cand_value
            bomb[row_index] = bomb_value
        state = {
            "labels": labels,
            "cand": cand,
            "bomb": bomb,
            "play": np.memmap(_plane_path(entry.dir, obs_spec.PLAY_PLANE), dtype=np.uint8,
                              mode="r", shape=(entry.frames, *obs_spec.PLAY_SHAPE)),
            "hud": (np.memmap(_plane_path(entry.dir, obs_spec.HUD_PLANE), dtype=np.uint8,
                              mode="r", shape=(entry.frames, *obs_spec.HUD_SHAPE))
                    if self.require_hud else None),
        }
        self._cache[entry_index] = state
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return state

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry_index, frame = self._entry_and_frame(index)
        state = self._shard_state(entry_index)
        entry = self.entries[entry_index]
        play = np.stack([state["play"][frame - self.stack + 1 + k] for k in range(self.stack)])
        label_index = frame + self.label_offset
        move = int(state["cand"][label_index])
        if move < 0:
            raise DatasetError(
                f"shard {entry.shard} frame {label_index} has an action mask that is not a "
                "student action (no shoot bit or ambiguous direction)")
        sample = {
            "schema": STUDENT_SAMPLE_SCHEMA,
            "play": play,
            "hud": None if state["hud"] is None else state["hud"][frame].copy(),
            "move": move,
            "bomb": int(state["bomb"][label_index]),
            "shard": entry.shard,
            "source": entry.source,
            "frame": frame,
            "label_frame": label_index,
            "obs_spec_version": obs_spec.OBS_SPEC_VERSION,
        }
        return sample

    # -- reporting --------------------------------------------------------
    def report(self) -> dict[str, Any]:
        windows_by_source: dict[str, int] = {}
        frames_by_source: dict[str, int] = {}
        unmappable = 0
        masked = 0
        for entry_index, entry in enumerate(self.entries):
            windows_by_source[entry.source] = windows_by_source.get(entry.source, 0) + self._counts[entry_index]
            frames_by_source[entry.source] = frames_by_source.get(entry.source, 0) + entry.frames
        for entry in self.entries:
            rows = _read_labels(entry.dir / "labels.jsonl", entry.frames)
            for row in rows:
                try:
                    decode_mask(int(row["msk"]))
                except DatasetError:
                    unmappable += 1
                else:
                    masked += 1
        total = masked + unmappable
        return {
            "schema": STUDENT_SAMPLE_SCHEMA,
            "kind": "bc-v0-behaviour-cloning",
            "shards": len(self.entries),
            "frames": sum(entry.frames for entry in self.entries),
            "windows": len(self),
            "windows_by_source": windows_by_source,
            "frames_by_source": frames_by_source,
            "actions_mappable": masked,
            "actions_unmappable": unmappable,
            "actions_mappable_fraction": (masked / total) if total else 0.0,
            "obs_spec_version": obs_spec.OBS_SPEC_VERSION,
            "obs_spec_digest": obs_spec.obs_spec_digest(),
            "label_kind": "collection policy mask (random/still/dive)",
            "disclosure": V0_DISCLOSURE,
        }


def build_dataset(
    index_path: str | Path,
    split_path: str | Path,
    *,
    split: str = "train",
    require_hud: bool = True,
) -> BehaviorCloneDataset:
    """Load index+split, select one split, return a validated window dataset."""

    index = load_index(index_path)
    entries = load_split(split_path, index)
    selected = [entry for entry in entries if entry.split == split]
    if not selected:
        raise DatasetError(f"split {split!r} has no shards")
    return BehaviorCloneDataset(selected, require_hud=require_hud)


# ---------------------------------------------------------------------------
# teacher labels (P3 target; fail closed until the teacher produces observations)
# ---------------------------------------------------------------------------
_OBSERVATION_KEYS = ("obs", "observation", "play", "play64", "shard", "shard_dir", "shard_key")


def _validate_teacher_decision(payload: dict[str, Any], where: str) -> dict[str, Any]:
    # ``TeacherLabel.as_dict()`` writes ``scores``; the teacher child decision log
    # (``.tmp_v400_teacher_child_stage1.jsonl``) writes the same vector as
    # ``geo_score`` (11-P1 §3 names it ``geo_score[18]``).  Accept both spellings,
    # reject anything else rather than guessing.
    scores = payload.get("scores", payload.get("geo_score"))
    mask = payload.get("verified_mask")
    if scores is None or mask is None:
        raise DatasetError(f"{where}: teacher decision has no scores/geo_score/verified_mask")
    scores_arr = np.asarray(scores, dtype=np.float64)
    mask_arr = np.asarray(mask, dtype=bool)
    if scores_arr.shape != (obs_spec.N_MOVE_CLASSES,):
        raise DatasetError(f"{where}: scores shape {scores_arr.shape} != (18,)")
    if mask_arr.shape != (obs_spec.N_MOVE_CLASSES,):
        raise DatasetError(f"{where}: verified_mask shape {mask_arr.shape} != (18,)")
    if not np.all(np.isfinite(scores_arr)):
        raise DatasetError(f"{where}: scores contain non-finite values")
    bomb = payload.get("bomb")
    if bomb not in (0, 1):
        raise DatasetError(f"{where}: bomb {bomb!r} is not 0/1")
    record = {
        "schema": TEACHER_LABEL_SCHEMA,
        "scores": scores_arr,
        "verified_mask": mask_arr,
        "verified_count": int(mask_arr.sum()),
        "bomb": int(bomb),
        # A decision with no chosen plan carries no margin (the teacher writes
        # ``null``); 0.0 keeps the record numeric without inventing a value.
        "margin": float(payload.get("margin") or 0.0),
        "chosen_plan": payload.get("chosen_plan"),
        "frame": int(payload.get("frame", 0)),
        "stage": int(payload.get("stage", 0)),
        "difficulty": int(payload.get("difficulty", 0)),
    }
    observation = {key: payload[key] for key in _OBSERVATION_KEYS if key in payload}
    record["observation"] = observation
    return record


def scan_teacher_labels(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Count decisions and observation references without training anything."""

    files = []
    decisions = 0
    with_observation = 0
    invalid = 0
    per_file = {}
    for path in paths:
        path = Path(path)
        file_decisions = 0
        file_with_obs = 0
        file_invalid = 0
        try:
            handle = open(path, "r", encoding="utf-8")
        except FileNotFoundError:
            per_file[str(path)] = {"present": False}
            continue
        with handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    file_invalid += 1
                    continue
                if payload.get("kind") != "decision":
                    continue
                file_decisions += 1
                try:
                    record = _validate_teacher_decision(payload, f"{path}:{line_number}")
                except DatasetError:
                    file_invalid += 1
                    continue
                if record["observation"]:
                    file_with_obs += 1
        per_file[str(path)] = {
            "present": True,
            "decisions": file_decisions,
            "with_observation_reference": file_with_obs,
            "invalid": file_invalid,
        }
        decisions += file_decisions
        with_observation += file_with_obs
        invalid += file_invalid
        files.append(str(path))
    return {
        "files": files,
        "decisions": decisions,
        "with_observation_reference": with_observation,
        "invalid": invalid,
        "per_file": per_file,
    }


def load_teacher_samples(
    paths: Iterable[str | Path],
    *,
    min_samples: int = MIN_TEACHER_SAMPLES,
    obs_root: str | Path | None = None,
    require_observations: bool = False,
) -> dict[str, Any]:
    """Load teacher decision labels, or raise with the real reason they are unusable.

    A usable teacher sample needs both the label and the observation it labelled.
    The teacher child JSONL (EXP-P1-003..011) has the label only, so without
    ``--save-observations`` shards this raises :class:`DatasetInsufficient` and
    reports counts rather than returning an empty dataset a training script could
    mistake for success.

    ``obs_root`` points at a directory written by
    ``tools/rl/v400_teacher.py --save-observations``; every label's ``obs``
    reference is then resolved through :func:`load_teacher_shard` (which verifies
    the frozen obs-v1 digest, the plane shapes/sizes and the
    ``frames[i-3..i] -> mask[i+1]`` alignment) and the row's ``play``/``hud`` are
    attached as ``record["obs_arrays"]``.  ``require_observations=True`` turns an
    unresolvable reference into an error instead of a warning.
    """

    paths = list(paths)
    if not paths:
        raise DatasetInsufficient("no teacher label files given")
    scan = scan_teacher_labels(paths)
    if scan["with_observation_reference"] == 0:
        raise DatasetInsufficient(
            "teacher labels carry no observation reference: "
            f"{scan['decisions']} decisions across {len(scan['files'])} file(s), "
            f"0 with an 'obs'/'play'/'shard' key; a pixel student cannot be trained "
            "from labels whose frames were not stored. " + V0_DISCLOSURE
        )
    if scan["with_observation_reference"] < min_samples:
        raise DatasetInsufficient(
            f"teacher labels are insufficient: {scan['with_observation_reference']} usable "
            f"decision(s) < required {min_samples}. " + V0_DISCLOSURE
        )
    records = []
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
    shards: dict[str, Any] = {}
    unresolved: list[dict[str, Any]] = []
    if obs_root is not None:
        shards, unresolved = _attach_teacher_observations(records, Path(obs_root))
        if require_observations and unresolved:
            raise DatasetInsufficient(
                f"teacher label(s) reference {len(unresolved)} observation row(s) that no "
                f"readable shard under {obs_root} provides (first: {unresolved[0]}). "
                + V0_DISCLOSURE)
    obs_shapes = {
        "play": list(obs_spec.PLAY_SHAPE),
        "hud": list(obs_spec.HUD_SHAPE),
        "stack": obs_spec.FRAME_STACK,
    }
    return {
        "schema": TEACHER_LABEL_SCHEMA,
        "kind": "teacher",
        "samples": records,
        "count": len(records),
        "obs_spec_version": obs_spec.OBS_SPEC_VERSION,
        "obs_spec_digest": obs_spec.obs_spec_digest(),
        "obs_shards": {key: shard.report() for key, shard in shards.items()},
        "obs_shapes": obs_shapes if shards else None,
        "obs_unresolved": unresolved,
        "decision_frames": obs_spec.DECISION_FRAMES,
        "alignment": TEACHER_OBSERVATION_TIME if shards else None,
        "disclosure": V0_DISCLOSURE if not shards else (
            "teacher-soft-label samples WITH stored obs-v1 observations; the D0 "
            "training gate (>= %d samples) still applies" % min_samples),
    }


# ---------------------------------------------------------------------------
# teacher observation shards (D0)
# ---------------------------------------------------------------------------
def _sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(path: Any) -> Path:
    candidate = Path(str(path))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise DatasetError(f"shard file path {path!r} escapes the shard directory")
    return candidate


def load_teacher_shard(shard_dir: str | Path, *, verify_sha: bool = True) -> "TeacherShard":
    """Load one ``--save-observations`` shard, or raise :class:`DatasetError`.

    This is the D0 gate.  It refuses: a missing/incomplete ``shard.json``, an obs
    spec version/digest that is not the frozen obs-v1 one, plane shapes or sizes
    that are not ``(N,4,64,64)`` / ``(N,56,160)`` uint8, an alignment statement
    that is not "observation at the decision point, label on the next action",
    a label row whose decision index and frame disagree, and any row whose stored
    render frames are not ``frames[d-4..d-1]``.
    """

    shard_dir = Path(shard_dir)
    payload = _read_json(shard_dir / "shard.json")
    if payload.get("schema") != TEACHER_SHARD_SCHEMA:
        raise DatasetError(
            f"{shard_dir}: shard schema {payload.get('schema')!r} != {TEACHER_SHARD_SCHEMA!r}")
    if payload.get("complete") is not True:
        raise DatasetError(f"{shard_dir}: shard is not complete; refusing to load")
    try:
        obs_spec.check_obs_spec(payload.get("obs_spec_version"), payload.get("obs_spec_digest"))
    except obs_spec.ObsSpecError as error:
        raise DatasetError(f"{shard_dir}: obs spec check failed: {error}") from error
    alignment = payload.get("alignment") or {}
    for key, expected in (
        ("observation_time", TEACHER_OBSERVATION_TIME),
        ("stack", obs_spec.FRAME_STACK),
        ("stack_order", obs_spec.STACK_ORDER),
        ("decision_frames", obs_spec.DECISION_FRAMES),
        ("p3_window_last_row", "i = decision_frame - 1"),
    ):
        if alignment.get(key) != expected:
            raise DatasetError(
                f"{shard_dir}: alignment {key}={alignment.get(key)!r} != {expected!r}; "
                "a shard recorded under another observation instant cannot be aligned "
                "with the teacher labels")
    if not alignment.get("p3_window") or "frames[i-3..i] -> mask[i+1]" not in str(
        alignment.get("p3_window")):
        raise DatasetError(f"{shard_dir}: alignment does not state the P3 window")
    decisions = payload.get("decision_count")
    if not isinstance(decisions, int) or decisions <= 0:
        raise DatasetError(f"{shard_dir}: decision_count {decisions!r} is not positive")
    frames_per_decision = payload.get("frames_per_decision")
    if not isinstance(frames_per_decision, int) or frames_per_decision < obs_spec.DECISION_FRAMES:
        raise DatasetError(
            f"{shard_dir}: frames_per_decision {frames_per_decision!r} is not a positive "
            f"multiple of DECISION_FRAMES={obs_spec.DECISION_FRAMES}")
    if frames_per_decision % obs_spec.DECISION_FRAMES:
        raise DatasetError(
            f"{shard_dir}: frames_per_decision {frames_per_decision} is not a multiple of "
            f"{obs_spec.DECISION_FRAMES}")
    obs = payload.get("obs") or {}
    planes: dict[str, np.ndarray] = {}
    expected_shapes = {
        obs_spec.PLAY_PLANE: (decisions, obs_spec.FRAME_STACK, *obs_spec.PLAY_SHAPE),
        obs_spec.HUD_PLANE: (decisions, *obs_spec.HUD_SHAPE),
    }
    for name, shape in expected_shapes.items():
        meta = obs.get("play64" if name == obs_spec.PLAY_PLANE else "hud") or {}
        relative = _safe_relative(meta.get("path", ""))
        path = shard_dir / relative
        if not path.is_file():
            raise DatasetError(f"{shard_dir}: missing plane {path}")
        if verify_sha:
            recorded = meta.get("sha256")
            if not recorded:
                raise DatasetError(f"{path}: no sha256 recorded in shard.json")
            actual = _sha256_file(path)
            if actual != recorded:
                raise DatasetError(
                    f"{path}: sha256 {actual} != recorded {recorded}")
        size = path.stat().st_size
        if int(meta.get("bytes", -1)) != size:
            raise DatasetError(
                f"{path}: {size} B != shard.json bytes {meta.get('bytes')!r}")
        try:
            array = np.load(path, mmap_mode="r")
        except (ValueError, OSError) as error:
            raise DatasetError(f"{path}: not a readable npy ({error})") from error
        if array.dtype != np.uint8:
            raise DatasetError(f"{path}: dtype {array.dtype} is not uint8")
        if array.shape != shape:
            raise DatasetError(f"{path}: shape {array.shape} != {shape}")
        planes[name] = array
    records = _read_teacher_obs_labels(
        shard_dir / "labels.jsonl",
        decisions=decisions,
        frames_per_decision=frames_per_decision,
        shard_key=str(payload.get("shard")),
    )
    return TeacherShard(
        dir=shard_dir,
        meta=payload,
        records=records,
        play=planes[obs_spec.PLAY_PLANE],
        hud=planes[obs_spec.HUD_PLANE],
        frames_per_decision=frames_per_decision,
    )


def _read_teacher_obs_labels(
    path: Path,
    *,
    decisions: int,
    frames_per_decision: int,
    shard_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = open(path, "r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise DatasetError(f"missing teacher labels: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_number}: bad JSON: {exc}") from exc
            where = f"{path}:{line_number}"
            if payload.get("kind") != "decision":
                raise DatasetError(f"{where}: kind {payload.get('kind')!r} is not 'decision'")
            if payload.get("schema") != TEACHER_OBS_SCHEMA:
                raise DatasetError(
                    f"{where}: label schema {payload.get('schema')!r} != {TEACHER_OBS_SCHEMA!r}")
            record = _validate_teacher_decision(payload, where)
            index = len(rows)
            if int(payload.get("decision", -1)) != index:
                raise DatasetError(
                    f"{where}: decision {payload.get('decision')!r} is not row {index}")
            frame = payload.get("decision_frame", payload.get("frame"))
            if not isinstance(frame, int) or frame != index * frames_per_decision:
                raise DatasetError(
                    f"{where}: decision {index} is at frame {frame!r}, expected "
                    f"{index * frames_per_decision} ({frames_per_decision} frames per "
                    "decision); the frame counter and the decision index disagree")
            observation = record["observation"].get("obs") or {}
            if observation.get("shard") != shard_key:
                raise DatasetError(
                    f"{where}: obs shard {observation.get('shard')!r} != {shard_key!r}")
            if int(observation.get("row", -1)) != index:
                raise DatasetError(f"{where}: obs row {observation.get('row')!r} != {index}")
            if observation.get("observation_time") != TEACHER_OBSERVATION_TIME:
                raise DatasetError(
                    f"{where}: observation_time {observation.get('observation_time')!r} "
                    f"!= {TEACHER_OBSERVATION_TIME!r}")
            render = [int(value) for value in observation.get("render_frames") or []]
            pad = int(observation.get("pad", 0))
            if len(render) != obs_spec.FRAME_STACK:
                raise DatasetError(
                    f"{where}: {len(render)} render frames, need {obs_spec.FRAME_STACK}")
            if pad == 0:
                # The stored stack is the 4 renders *ending at* d-1 (the instant the
                # teacher was queried), so P3's frames[i-3..i] -> mask[i+1] holds
                # with i = d-1.
                expected = [frame - obs_spec.FRAME_STACK + offset
                            for offset in range(obs_spec.FRAME_STACK)]
                if render != expected:
                    raise DatasetError(
                        f"{where}: stored renders {render} != frames[d-4..d-1] {expected}; "
                        "the observation is not the P3 frames[i-3..i] window")
                newest = observation.get("newest_render_frame")
                if newest is None or int(newest) + 1 != frame:
                    raise DatasetError(
                        f"{where}: newest render {newest!r} + 1 != decision frame {frame}")
                if int(observation.get("p3_label_frame", -1)) != frame:
                    raise DatasetError(
                        f"{where}: p3_label_frame {observation.get('p3_label_frame')!r} != "
                        f"{frame}")
            elif not (index == 0 and pad == obs_spec.FRAME_STACK - 1):
                raise DatasetError(
                    f"{where}: pad={pad} is only allowed on decision 0 "
                    f"(episode-start padding, obs-spec-v1 §2)")
            record["row"] = index
            record["obs"] = observation
            rows.append(record)
    if len(rows) != decisions:
        raise DatasetError(
            f"{path}: {len(rows)} label rows but shard.json says {decisions} decisions")
    for index in range(1, len(rows)):
        if int(rows[index]["obs"].get("pad", 0)):
            raise DatasetError(f"{path}: decision {index} has episode-start padding")
    return rows


class TeacherShard:
    """One loaded ``--save-observations`` shard: labels + mem-mapped obs planes.

    ``shard[i]`` returns the raw uint8 ``play (4,64,64)`` + ``hud (56,160)`` next
    to the teacher label, i.e. exactly one P3 teacher sample.  ``normalized(i)``
    applies :func:`obs_spec.normalize_play` / :func:`obs_spec.normalize_hud`, the
    only normalisation obs-v1 allows.
    """

    def __init__(
        self,
        *,
        dir: Path,
        meta: dict[str, Any],
        records: Sequence[dict[str, Any]],
        play: np.ndarray,
        hud: np.ndarray,
        frames_per_decision: int,
    ) -> None:
        self.dir = Path(dir)
        self.meta = dict(meta)
        self.records = list(records)
        self.play = play
        self.hud = hud
        self.frames_per_decision = int(frames_per_decision)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "schema": STUDENT_SAMPLE_SCHEMA,
            "kind": "teacher-observed",
            "decision": int(index),
            "frame": int(record["frame"]),
            "play": np.asarray(self.play[index], dtype=np.uint8),
            "hud": np.asarray(self.hud[index], dtype=np.uint8),
            "scores": record["scores"],
            "verified_mask": record["verified_mask"],
            "verified_count": record["verified_count"],
            "bomb": record["bomb"],
            "margin": record["margin"],
            "chosen_plan": record["chosen_plan"],
            "obs_spec_version": self.meta["obs_spec_version"],
            "obs_spec_digest": self.meta["obs_spec_digest"],
            "shard": self.meta.get("shard"),
        }

    def normalized(self, index: int) -> dict[str, np.ndarray]:
        sample = self[index]
        return {
            "play": obs_spec.normalize_play(sample["play"]),
            "hud": obs_spec.normalize_hud(sample["hud"]),
        }

    def report(self) -> dict[str, Any]:
        obs = self.meta.get("obs") or {}
        return {
            "shard": self.meta.get("shard"),
            "dir": str(self.dir),
            "decisions": len(self.records),
            "frames": self.meta.get("stage_frames"),
            "frames_per_decision": self.frames_per_decision,
            "obs_spec_version": self.meta.get("obs_spec_version"),
            "obs_spec_digest": self.meta.get("obs_spec_digest"),
            "alignment": self.meta.get("alignment", {}).get("observation_time"),
            "play_shape": list(self.play.shape),
            "hud_shape": list(self.hud.shape),
            "obs_bytes": (obs.get("bytes_total")),
            "labels_sha256": (self.meta.get("labels") or {}).get("sha256"),
            "play_sha256": (obs.get("play64") or {}).get("sha256"),
            "hud_sha256": (obs.get("hud") or {}).get("sha256"),
            "complete": self.meta.get("complete"),
        }


def load_teacher_manifest(root: str | Path) -> dict[str, Any]:
    """Read ``<root>/manifest.json`` of a ``--save-observations`` dataset."""

    root = Path(root)
    payload = _read_json(root / "manifest.json")
    if payload.get("schema") != TEACHER_MANIFEST_SCHEMA:
        raise DatasetError(
            f"{root}: manifest schema {payload.get('schema')!r} != {TEACHER_MANIFEST_SCHEMA!r}")
    try:
        obs_spec.check_obs_spec(payload.get("obs_spec_version"), payload.get("obs_spec_digest"))
    except obs_spec.ObsSpecError as error:
        raise DatasetError(f"{root}: manifest obs spec check failed: {error}") from error
    shards = payload.get("shards") or []
    if not shards:
        raise DatasetError(f"{root}: manifest lists no shards")
    totals = payload.get("totals") or {}
    if int(totals.get("decisions", -1)) != sum(int(row.get("decisions", 0)) for row in shards):
        raise DatasetError(f"{root}: manifest totals do not sum the shard rows")
    return payload


def _attach_teacher_observations(
    records: list[dict[str, Any]], obs_root: Path
) -> tuple[dict[str, "TeacherShard"], list[dict[str, Any]]]:
    """Resolve every label's ``obs`` reference to a loaded shard row."""

    shards: dict[str, TeacherShard] = {}
    unresolved: list[dict[str, Any]] = []
    for record in records:
        reference = (record["observation"].get("obs") or {})
        key = reference.get("shard")
        row = reference.get("row")
        if not key or row is None:
            unresolved.append({"reason": "no shard/row in obs reference", "obs": reference})
            continue
        path = Path(obs_root) / "shards" / str(key)
        if not (path / "shard.json").is_file():
            unresolved.append({"reason": f"no shard.json under {path}", "shard": key})
            continue
        cache_key = str(path)
        if cache_key not in shards:
            shards[cache_key] = load_teacher_shard(path)
        shard = shards[cache_key]
        index = int(row)
        if not 0 <= index < len(shard):
            raise DatasetError(
                f"{path}: obs row {index} is outside the shard's {len(shard)} decisions")
        record["obs_arrays"] = {
            "shard": shard,
            "row": index,
            "play": shard.play[index],
            "hud": shard.hud[index],
            "play_shape": list(obs_spec.PLAY_SHAPE),
            "hud_shape": list(obs_spec.HUD_SHAPE),
            "stack": obs_spec.FRAME_STACK,
        }
    return shards, unresolved


__all__ = [
    "BehaviorCloneDataset",
    "DatasetError",
    "DatasetInsufficient",
    "INDEX_SCHEMA",
    "LABEL_VERSION",
    "MIN_TEACHER_SAMPLES",
    "PLANE_SHAPES",
    "SPLIT_SCHEMA",
    "STUDENT_SAMPLE_SCHEMA",
    "ShardEntry",
    "TEACHER_LABEL_SCHEMA",
    "TEACHER_MANIFEST_SCHEMA",
    "TEACHER_OBSERVATION_TIME",
    "TEACHER_OBS_SCHEMA",
    "TEACHER_SHARD_SCHEMA",
    "TeacherShard",
    "V0_DISCLOSURE",
    "build_dataset",
    "decode_mask",
    "load_index",
    "load_split",
    "load_teacher_manifest",
    "load_teacher_samples",
    "load_teacher_shard",
    "scan_teacher_labels",
]
