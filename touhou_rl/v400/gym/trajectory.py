"""``TrajectoryStart``: resume a recorded trajectory at an exact, verified frame.

03 §5 names three ways into a stage: ``StageStart`` (enter the stage by seed),
``ReplayStart`` (fast-forward a human replay) and ``TrajectoryStart`` (fast-forward
**our own** recorded trajectory to frame ``n`` and hand control over there).  This
module implements the third one.

Why the hand-over has to be verified
------------------------------------
A fast-forward is only useful if the state it stops on is *the same state the
trajectory had at that frame*.  If it is not, every label produced after the
take-over describes a state that never existed on the recorded run, and nothing
downstream can see that: the episode runs, the teacher answers, the data looks
fine.  So the hand-over is fail-closed: the frame the replay stops on must carry
a **recorded state hash**, and the engine must report exactly that hash.  A
mismatch raises ``TrajectoryHashMismatch`` with the frame number, the expected
hash and the observed hash, and no episode is continued.

Masks are per frame
-------------------
The recorded corpora store the committed input as ``(mask, count)`` blocks or as
one mask per frame; both expand to the same per-frame tape.  The replay feeds the
tape with ``STEP <mask> <n>`` blocks whose ``n`` never crosses a recorded hash
checkpoint, so every checkpoint the tape passes is verified.  ``STEP mask N`` is
``N`` applications of the same mask inside one ``PumpFrame`` loop in the engine
(``HandleStep``), so a block and ``N`` single-frame steps advance the engine
identically — the ``--batch-frames 1`` mode exists to prove that on the real
engine rather than assume it.

Corpus
------
:func:`load_d0_shard` / :func:`load_d0_shards` read the D0 teacher shards
(``.tmp_v400_d0/d0/shards/*/{shard.json,labels.jsonl}``).  Those shards already
carry what a verified take-over needs: the exact ``RESET`` line, the per-frame
masks (``step_frames``), and the 25-field state hash at every decision boundary
(``hash_pre``/``hash_post`` plus ``shard.json``'s ``hash_checkpoint_pairs``).  The
loader cross-checks the two independent hash sources and refuses a shard that
disagrees with itself, so no hash is synthesised here.

Take-over point sampling
------------------------
:func:`sample_target_frame` picks a recorded frame with the bias T3 asks for
("偏向后半段与 Boss 段"): the late half of the stage and, when the caller knows
where the boss section starts, the boss section.  It samples **only** frames that
carry a recorded hash, so every sampled point is verifiable by construction.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from touhou_rl.v400.gym.env import DIFFICULTIES, StageStart, Th08Env

#: One mask is a u16 bitfield (10-P0 §2).
MASK_MAX = 0xFFFF

#: The engine prints the state hash as 16 lowercase hex digits.
HASH_HEX_LENGTH = 16

#: Frame count of one STEP block while replaying.  A block never crosses a
#: recorded checkpoint, so this only bounds the size of a hash-checked chunk.
DEFAULT_BATCH_FRAMES = 64

#: Frames of one D0 decision: ``decision_frames`` x ``committed frames``.
D0_FRAMES_PER_DECISION = 4

_HEX_DIGITS = frozenset("0123456789abcdef")


# ---------------------------------------------------------------------------
# Errors: every one of them means "do not continue this episode"
# ---------------------------------------------------------------------------
class TrajectoryStartError(RuntimeError):
    """Base class: the recorded trajectory could not be resumed faithfully."""


class TrajectoryMaskError(TrajectoryStartError):
    """The mask tape is unusable (not enough frames, or a mask out of range)."""


class TrajectoryTargetError(TrajectoryStartError):
    """``target_frame`` is outside the tape or negative."""


class TrajectoryHashUnavailable(TrajectoryStartError):
    """A frame that must be verified carries no recorded hash to verify."""


class TrajectoryHashMismatch(TrajectoryStartError):
    """The engine's state hash at ``frame`` is not the recorded hash."""

    def __init__(self, *, frame: int, expected: str, observed: str, where: str) -> None:
        self.frame = int(frame)
        self.expected = expected
        self.observed = observed
        self.where = where
        super().__init__(
            f"trajectory take-over diverged at frame {self.frame} ({where}): "
            f"expected hash={expected}, observed hash={observed}; "
            "refusing to hand over control from an unverified state")

    def as_record(self) -> dict[str, object]:
        return {
            "trajectory_hash_mismatch": True,
            "frame": self.frame,
            "expected": self.expected,
            "observed": self.observed,
            "where": self.where,
        }


class TrajectoryAborted(TrajectoryStartError):
    """The engine left the trajectory: game over, exit, or a short advance."""


class TrajectoryCorpusError(TrajectoryStartError):
    """A recorded shard is incomplete or contradicts itself."""


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------
def normalize_hash(value: object, *, where: str = "state hash") -> str:
    """Return a 16-digit lowercase hex state hash, or raise.

    ``"unknown"``, an empty string, an integer count or a truncated hash is
    rejected instead of coerced: the guard must never be able to compare two
    unverifiable values and call them equal.
    """
    if not isinstance(value, str):
        raise TrajectoryHashUnavailable(
            f"{where} is not a hex string ({type(value).__name__}: {value!r})")
    text = value.strip().lower()
    if len(text) != HASH_HEX_LENGTH or any(character not in _HEX_DIGITS for character in text):
        raise TrajectoryHashUnavailable(f"{where} is not 16 hex digits: {value!r}")
    return text


def _normalize_mask(value: object, *, index: int) -> int:
    try:
        mask = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TrajectoryMaskError(f"mask at frame {index} is not an integer: {value!r}") from exc
    if not 0 <= mask <= MASK_MAX:
        raise TrajectoryMaskError(f"mask at frame {index} is out of range: {mask}")
    return mask


# ---------------------------------------------------------------------------
# TrajectoryStart
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrajectoryStart:
    """Enter a stage by seed, then fast-forward a recorded tape to ``target_frame``.

    03 §5 writes this as ``TrajectoryStart(traj_id, frame)``.  The recorded tape
    is carried along here (``seed``/``stage_index``/``difficulty``/``masks``)
    because a take-over has to be reproducible from the start object alone: the
    same ``RESET`` line as the original run, then the same per-frame masks.
    ``hashes`` maps **stage frame** (``FrameCount``, reset to 0 by ``RESET``) to
    the state hash recorded on that frame; :attr:`target_frame` must be one of
    them, and every checkpoint the replay passes is compared too.

    ``team``/``resources`` default to the D0 configuration (team 0, standard
    resources); ``practice`` starts a trajectory recorded in practice mode.
    """

    seed: int
    stage_index: int
    difficulty: str
    masks: Sequence[int]
    target_frame: int
    hashes: Mapping[int, str] = field(default_factory=dict)
    team: int = 0
    resources: str = "standard"
    traj_id: str = ""
    #: Derived in ``__post_init__`` from the fields above (03 §5's ``RESET`` spec).
    stage_start: StageStart = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        masks = tuple(_normalize_mask(mask, index=index) for index, mask in enumerate(self.masks))
        hashes: dict[int, str] = {}
        for key, value in dict(self.hashes).items():
            frame = int(key)
            if frame < 0:
                raise TrajectoryTargetError(f"recorded hash frame is negative: {frame}")
            hashes[frame] = normalize_hash(value, where=f"recorded hash at frame {frame}")
        target = int(self.target_frame)
        if target < 0:
            raise TrajectoryTargetError(f"target_frame is negative: {target}")
        if target > len(masks):
            raise TrajectoryMaskError(
                f"mask tape holds {len(masks)} frames but target_frame is {target}; "
                "the take-over point is beyond the recorded trajectory")
        if target not in hashes:
            raise TrajectoryHashUnavailable(
                f"target_frame {target} has no recorded state hash; a take-over at a frame "
                "that cannot be verified is refused")
        start = StageStart(stage=self.stage_index, difficulty=self.difficulty,
                           team=self.team, seed=self.seed, resources=self.resources)
        object.__setattr__(self, "masks", masks)
        object.__setattr__(self, "hashes", hashes)
        object.__setattr__(self, "target_frame", target)
        object.__setattr__(self, "stage_start", start)

    @property
    def frames(self) -> int:
        """Length of the recorded mask tape."""
        return len(self.masks)

    @property
    def expected_target_hash(self) -> str:
        return self.hashes[self.target_frame]

    def as_record(self) -> dict[str, object]:
        """Provenance of this start; no number is recomputed."""
        return {
            "traj_id": self.traj_id,
            "seed": self.seed,
            "stage_index": self.stage_index,
            "difficulty": self.difficulty,
            "team": self.team,
            "resources": self.resources,
            "masks": len(self.masks),
            "target_frame": self.target_frame,
            "checkpoints": len(self.hashes),
            "expected_target_hash": self.expected_target_hash,
        }


@dataclass(frozen=True)
class TakeoverReport:
    """What the fast-forward actually did, for the experiment card."""

    traj_id: str
    seed: int
    stage_index: int
    difficulty: str
    target_frame: int
    reset_hash: str
    target_hash: str
    expected_target_hash: str
    frames_advanced: int
    steps: int
    checkpoints_checked: int
    checkpoint_hashes: tuple[tuple[int, str], ...]
    wall_s: float
    reset_s: float
    fast_forward_s: float

    @property
    def matches(self) -> bool:
        return self.target_hash == self.expected_target_hash

    def as_record(self) -> dict[str, object]:
        return {
            "traj_id": self.traj_id,
            "seed": self.seed,
            "stage_index": self.stage_index,
            "difficulty": self.difficulty,
            "target_frame": self.target_frame,
            "reset_hash": self.reset_hash,
            "target_hash": self.target_hash,
            "expected_target_hash": self.expected_target_hash,
            "matches": self.matches,
            "frames_advanced": self.frames_advanced,
            "steps": self.steps,
            "checkpoints_checked": self.checkpoints_checked,
            "wall_s": round(self.wall_s, 6),
            "reset_s": round(self.reset_s, 6),
            "fast_forward_s": round(self.fast_forward_s, 6),
        }


# ---------------------------------------------------------------------------
# The fast-forward itself
# ---------------------------------------------------------------------------
def replay_to_frame(env: Th08Env, start: TrajectoryStart, *,
                    batch_frames: int = DEFAULT_BATCH_FRAMES) -> TakeoverReport:
    """``RESET`` the recorded start, feed the tape, verify, hand over at frame ``n``.

    The environment must already be attached (use it as a context manager).  On
    return the engine sits on ``start.target_frame`` with control available to the
    caller: the verified hash equals the recorded hash, or this function raised.

    Every recorded checkpoint the tape passes is compared, not only the target, so
    a divergence is reported at the frame where it first appears.
    """
    if batch_frames < 1:
        raise ValueError(f"batch_frames must be positive: {batch_frames}")
    target = start.target_frame
    masks = start.masks
    hashes = start.hashes
    started = time.perf_counter()

    env.reset(start.stage_start)
    reset_s = time.perf_counter() - started
    reset_hash = normalize_hash(getattr(env, "last_reset_hash", None),
                                where="RESET reply hash")
    expected_reset = hashes.get(0)
    if expected_reset is not None and reset_hash != expected_reset:
        raise TrajectoryHashMismatch(frame=0, expected=expected_reset, observed=reset_hash,
                                     where="RESET")

    checkpoints: list[tuple[int, str]] = []
    if 0 in hashes:
        checkpoints.append((0, reset_hash))

    frame = 0
    steps = 0
    fast_forward_started = time.perf_counter()
    while frame < target:
        mask = masks[frame]
        count = 1
        while (count < batch_frames and frame + count < target
               and masks[frame + count] == mask
               and (frame + count) not in hashes):
            count += 1
        reply = env.advance(mask, count)
        steps += 1
        if reply.exit or reply.game_over:
            raise TrajectoryAborted(
                f"engine left the trajectory at frame {frame + reply.advanced} "
                f"(exit={reply.exit} game_over={reply.game_over}); the recorded trajectory "
                f"needs frame {target}")
        if reply.advanced != count:
            raise TrajectoryAborted(
                f"STEP advanced {reply.advanced} of {count} frames at frame {frame}")
        if reply.frame != frame + count:
            raise TrajectoryAborted(
                f"engine frame counter {reply.frame} != {frame + count} after a STEP of "
                f"{count} frames; the tape and the engine are not on the same frame")
        frame += reply.advanced
        if frame not in hashes:
            continue
        observed = normalize_hash(reply.hash, where=f"STEP reply hash at frame {frame}")
        expected = hashes[frame]
        if observed != expected:
            raise TrajectoryHashMismatch(frame=frame, expected=expected, observed=observed,
                                         where="fast-forward checkpoint")
        checkpoints.append((frame, observed))
    fast_forward_s = time.perf_counter() - fast_forward_started

    if frame != target:
        raise TrajectoryAborted(f"stopped at frame {frame}, not the target frame {target}")
    shm = env.shm
    if shm is not None and int(shm.header()["frame"]) != target:
        raise TrajectoryAborted(
            f"shared state block holds frame {int(shm.header()['frame'])}, not {target}")
    target_hash = next(value for index, value in reversed(checkpoints) if index == target)

    return TakeoverReport(
        traj_id=start.traj_id,
        seed=start.seed,
        stage_index=start.stage_index,
        difficulty=start.difficulty,
        target_frame=target,
        reset_hash=reset_hash,
        target_hash=target_hash,
        expected_target_hash=start.expected_target_hash,
        frames_advanced=frame,
        steps=steps,
        checkpoints_checked=len(checkpoints),
        checkpoint_hashes=tuple(checkpoints),
        wall_s=time.perf_counter() - started,
        reset_s=reset_s,
        fast_forward_s=fast_forward_s,
    )


# ---------------------------------------------------------------------------
# Take-over point sampling
# ---------------------------------------------------------------------------
def sample_target_frame(rng, candidates: Sequence[int], *, bias: float = 0.65,
                        boss_start: int | None = None,
                        boss_weight: float = 0.25) -> int:
    """Draw one recorded frame, biased to the late half and the boss section.

    ``candidates`` are the frames that carry a recorded state hash, so every draw
    is verifiable.  The draw uses one ``rng.random()`` followed by one
    ``rng.choice()``, which keeps it reproducible for a seeded ``random.Random``.

    Masses (they must sum to at most 1):

    * ``boss_weight`` - frames at or after ``boss_start`` (skipped when
      ``boss_start`` is ``None`` or no candidate reaches it);
    * ``bias`` - frames in the late half, i.e. ``frame >= (lo + hi + 1) // 2``;
    * the remainder - uniform over every candidate.

    ``bias=0.0, boss_weight=0.0`` is a uniform draw over the recorded frames.
    """
    for name, value in (("bias", bias), ("boss_weight", boss_weight)):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    if float(bias) + float(boss_weight) > 1.0 + 1e-9:
        raise ValueError(
            f"bias + boss_weight must be at most 1, got {bias} + {boss_weight}")
    frames = [int(frame) for frame in candidates]
    if not frames:
        raise ValueError("no candidate frames to sample a take-over point from")
    if len(frames) == 1:
        return frames[0]
    low, high = min(frames), max(frames)
    halfway = (low + high + 1) // 2
    late = [frame for frame in frames if frame >= halfway]
    boss = ([frame for frame in frames if frame >= int(boss_start)]
            if boss_start is not None else [])
    roll = float(rng.random())
    if boss and roll < float(boss_weight):
        pool = boss
    elif late and roll < float(boss_weight) + float(bias):
        pool = late
    else:
        pool = frames
    return int(rng.choice(pool))


# ---------------------------------------------------------------------------
# Recorded trajectories (D0 shards)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RecordedTrajectory:
    """One recorded run: the exact start, the per-frame tape, the hash table."""

    traj_id: str
    seed: int
    stage_index: int
    difficulty: str
    team: int
    resources: str
    masks: tuple[int, ...]
    hashes: Mapping[int, str]
    frames: int
    source: str

    @property
    def checkpoint_frames(self) -> tuple[int, ...]:
        return tuple(sorted(self.hashes))

    def start(self, target_frame: int) -> TrajectoryStart:
        return TrajectoryStart(
            seed=self.seed, stage_index=self.stage_index, difficulty=self.difficulty,
            masks=self.masks, target_frame=int(target_frame), hashes=self.hashes,
            team=self.team, resources=self.resources, traj_id=self.traj_id)


def parse_reset_command(text: object) -> dict[str, object]:
    """``"RESET 0 1 0 1 0"`` -> the exact ``StageStart`` fields it asked for."""
    if not isinstance(text, str):
        raise TrajectoryCorpusError(f"reset_command is not a string: {text!r}")
    parts = text.split()
    if len(parts) < 6 or parts[0] != "RESET":
        raise TrajectoryCorpusError(f"unparsable reset_command: {text!r}")
    try:
        stage, difficulty, team, seed, practice = (int(part) for part in parts[1:6])
    except ValueError as exc:
        raise TrajectoryCorpusError(f"unparsable reset_command: {text!r}") from exc
    if not 0 <= difficulty < len(DIFFICULTIES):
        raise TrajectoryCorpusError(f"reset_command difficulty out of range: {text!r}")
    if practice not in (0, 1):
        raise TrajectoryCorpusError(f"reset_command practice is not 0/1: {text!r}")
    return {
        "stage_index": stage,
        "difficulty": DIFFICULTIES[difficulty],
        "team": team,
        "seed": seed,
        "resources": "practice" if practice else "standard",
    }


def _read_decisions(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("rb") as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TrajectoryCorpusError(f"{path.name}: line {number} is not JSON: {exc}") from exc
            if record.get("kind") != "decision":
                raise TrajectoryCorpusError(
                    f"{path.name}: line {number} is kind={record.get('kind')!r}, expected decision")
            records.append(record)
    if not records:
        raise TrajectoryCorpusError(f"{path.name}: no decision rows")
    return records


def load_d0_shard(shard_dir: str | Path) -> RecordedTrajectory:
    """Read one D0 shard into a :class:`RecordedTrajectory`.

    Fail-closed on: an incomplete shard, a ``reset_command`` that is not the D0
    shape, a decision whose frame counter or mask count does not add up, a hash
    checkpoint the shard's own A/B comparison marks unequal, or a disagreement
    between ``labels.jsonl`` and ``shard.json``'s ``hash_checkpoint_pairs``.
    """
    shard_dir = Path(shard_dir)
    shard_path = shard_dir / "shard.json"
    labels_path = shard_dir / "labels.jsonl"
    for path in (shard_path, labels_path):
        if not path.is_file():
            raise TrajectoryCorpusError(f"shard file missing: {path}")
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    if shard.get("complete") is not True:
        raise TrajectoryCorpusError(f"{shard_path}: shard is not complete")
    frames_per_decision = int(shard.get("frames_per_decision") or D0_FRAMES_PER_DECISION)
    if frames_per_decision <= 0:
        raise TrajectoryCorpusError(f"{shard_path}: frames_per_decision={frames_per_decision}")
    start = parse_reset_command(shard.get("reset_command"))

    decisions = _read_decisions(labels_path)
    masks: list[int] = []
    from_labels: dict[int, str] = {}
    previous_post: tuple[int, str] | None = None
    for index, record in enumerate(decisions):
        if int(record.get("decision", index)) != index:
            raise TrajectoryCorpusError(
                f"{labels_path.name}: decision {index} is labelled {record.get('decision')!r}")
        frame = int(record.get("frame", -1))
        if frame != index * frames_per_decision:
            raise TrajectoryCorpusError(
                f"{labels_path.name}: decision {index} frame={frame}, expected "
                f"{index * frames_per_decision}")
        steps = record.get("step_frames")
        if not isinstance(steps, list) or len(steps) != frames_per_decision:
            raise TrajectoryCorpusError(
                f"{labels_path.name}: decision {index} carries {len(steps) if isinstance(steps, list) else steps} "
                f"masks, expected {frames_per_decision}")
        for offset, mask in enumerate(steps):
            masks.append(_normalize_mask(mask, index=frame + offset))
        pre = normalize_hash(record.get("hash_pre"), where=f"decision {index} hash_pre")
        post = normalize_hash(record.get("hash_post"), where=f"decision {index} hash_post")
        from_labels[frame] = pre
        from_labels[frame + frames_per_decision] = post
        if previous_post is not None and previous_post[0] == frame and previous_post[1] != pre:
            raise TrajectoryCorpusError(
                f"{labels_path.name}: decision {index} hash_pre={pre} does not continue "
                f"decision {index - 1} hash_post={previous_post[1]}")
        previous_post = (frame + frames_per_decision, post)

    frames = len(masks)
    from_pairs: dict[int, str] = {}
    pairs = shard.get("hash_checkpoint_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise TrajectoryCorpusError(f"{shard_path}: no hash_checkpoint_pairs")
    for index, pair in enumerate(pairs):
        if pair.get("equal") is not True:
            raise TrajectoryCorpusError(
                f"{shard_path}: checkpoint {index} (decision {pair.get('decision')}, "
                f"{pair.get('boundary')}, frame {pair.get('frame')}) is marked unequal: "
                f"env={pair.get('env')!r} shadow={pair.get('shadow')!r}")
        frame = int(pair.get("frame", -1))
        value = normalize_hash(pair.get("env"), where=f"checkpoint frame {frame} env hash")
        if from_pairs.get(frame, value) != value:
            raise TrajectoryCorpusError(
                f"{shard_path}: frame {frame} carries two different env hashes")
        from_pairs[frame] = value

    for frame, value in from_pairs.items():
        if frame in from_labels and from_labels[frame] != value:
            raise TrajectoryCorpusError(
                f"{shard_dir.name}: frame {frame} hash {from_labels[frame]} in labels.jsonl "
                f"but {value} in hash_checkpoint_pairs")
    missing = [frame for frame in from_labels if frame not in from_pairs]
    if missing:
        raise TrajectoryCorpusError(
            f"{shard_dir.name}: {len(missing)} label checkpoints have no A/B pair "
            f"(first: frame {missing[0]})")

    hashes = dict(from_labels)
    hashes.update(from_pairs)
    beyond = [frame for frame in hashes if frame > frames]
    if beyond:
        raise TrajectoryCorpusError(
            f"{shard_dir.name}: checkpoint at frame {min(beyond)} is beyond the "
            f"{frames}-frame mask tape")

    return RecordedTrajectory(
        traj_id=f"d0/{shard_dir.name}",
        seed=int(start["seed"]),  # type: ignore[arg-type]
        stage_index=int(start["stage_index"]),  # type: ignore[arg-type]
        difficulty=str(start["difficulty"]),
        team=int(start["team"]),  # type: ignore[arg-type]
        resources=str(start["resources"]),
        masks=tuple(masks),
        hashes=hashes,
        frames=frames,
        source=str(shard_path),
    )


def load_d0_shards(root: str | Path) -> list[RecordedTrajectory]:
    """Every shard under ``root`` (a ``.../d0/shards`` directory), by name."""
    root = Path(root)
    if not root.is_dir():
        raise TrajectoryCorpusError(f"shard root missing: {root}")
    shards = sorted(path for path in root.iterdir() if (path / "shard.json").is_file())
    if not shards:
        raise TrajectoryCorpusError(f"no shards under {root}")
    return [load_d0_shard(path) for path in shards]


__all__ = [
    "DEFAULT_BATCH_FRAMES",
    "MASK_MAX",
    "RecordedTrajectory",
    "TakeoverReport",
    "TrajectoryAborted",
    "TrajectoryCorpusError",
    "TrajectoryHashMismatch",
    "TrajectoryHashUnavailable",
    "TrajectoryMaskError",
    "TrajectoryStart",
    "TrajectoryStartError",
    "TrajectoryTargetError",
    "load_d0_shard",
    "load_d0_shards",
    "normalize_hash",
    "parse_reset_command",
    "replay_to_frame",
    "sample_target_frame",
]
