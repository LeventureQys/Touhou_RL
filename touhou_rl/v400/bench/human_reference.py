"""Measured human reference: the source of the WP7 frame caps.

Why this file exists in two halves
----------------------------------
A replay file carries per-stage *input stream extents*, not gameplay lengths, so
``inspect_human_replay`` / ``build_human_reference`` can only fail closed: the
retail codec verifies the file and its streams but certifies no death or
bomb-use event, and stage-clear/final streams include extra input records.

EXP-P0-046 measured the missing quantities the only way they can be measured: it
played two real retail replays in the real Linux engine and read the engine's
own ``STATE`` counters during playback (``GameManager.gameplayFrameCounter``,
``ZunGlobals.deaths``, ``ZunGlobals.bombsRemaining``).  Those measured numbers
are the only input the builder below accepts; nothing is derived from
``inputRecords``.

The benchmark cap rule is::

    cap(stage) = ceil(1.5 * measured human gameplay_frames(stage))

The 1.5 multiplier is a declared benchmark policy, not a measurement.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from touhou_rl.v400.gym.replay_starts import decode_replay

SCHEMA = "v400-wp7-human-reference/1"
COLLECTION_METHOD = "engine STATE during real replay playback"
CAP_MULTIPLIER = 1.5
EVIDENCE_KIND = "real-engine-replay-playback"

#: ScoreDat ``Stage`` enum order, as used by EXP-P0-046 for ``stage_index``.
STAGE_NAMES = ("Stage1", "Stage2", "Stage3", "Stage4A", "Stage4B", "Stage5",
               "Stage6A", "Stage6B", "Extra")
#: ``ZunGlobals.difficulty`` indices that the measured runs cover.
DIFFICULTY_BY_INDEX = {0: "easy", 1: "normal", 2: "hard", 3: "lunatic"}
#: Playable shot indices (retail ``SHOTS`` order); only the two measured ones matter.
SHOT_NAMES = ("Border Team", "Magic Team", "Scarlet Team", "Ghost Team", "Reimu",
              "Yukari", "Marisa", "Alice", "Sakuya", "Remilia", "Youmu", "Yuyuko")
#: Bench suite ``stage_index`` -> measured stage name.  The key is the **0-based**
#: engine stage index (the ``RESET <stage>`` argument, ``STATE.f1`` and the
#: ``ScoreDat.hpp`` ``Stage`` enum, where ``STAGE1 = 0``), so ``0 -> Stage1``.
#: This table is the cap-side half of the stage-index contract: a 1-based key would
#: silently return the *next* stage's cap (EXP-P0-077: a suite that reset stage 1
#: was given ``Stage1``'s cap).  Keys 0..7 are the eight gameplay stages; ``8`` is
#: ``Extra`` so an unmeasured stage fails on coverage, not on naming.
SUITE_STAGE_NAMES = {
    0: "Stage1", 1: "Stage2", 2: "Stage3", 3: "Stage4A", 4: "Stage4B",
    5: "Stage5", 6: "Stage6A", 7: "Stage6B", 8: "Extra",
}
#: ``f3`` (gameplay) minus ``f2`` (replay input) is the stage-clear sequence;
#: EXP-P0-046 bounds it at 600 frames and fails closed beyond that.
MAX_CLEAR_SEQUENCE_FRAMES = 600
#: Every stage the reference would need for a full-coverage claim.
ALL_STAGES = STAGE_NAMES


class MissingHumanMeasurements(ValueError):
    """Replay metadata cannot supply a measured human benchmark reference."""


class HumanReferenceError(ValueError):
    """A measured reference document is absent, malformed, or does not cover a stage."""


REQUIRED_GAME_STATE = (
    "attested retail game playback tied to the replay SHA-256",
    "observed per-stage gameplay calculation and terminal/clear frames",
    "observed player death events or authoritative death counters and reset semantics",
    "observed bomb-use events or authoritative bomb counters and reset semantics",
    "observed stage boundaries and run completion status",
)


# -- replay-stream evidence (fail closed, no measured numbers) -------------
def inspect_human_replay(path: str | Path) -> dict:
    """Return only verified stream extents; report unmeasured fields as null."""
    starts = decode_replay(path, require_retail=True)
    first = starts[0]
    last_index = starts[-1].stage_index
    return {
        "status": "insufficient_real_game_state",
        "source": str(Path(path).resolve()),
        "source_sha256": first.source_sha256,
        "executable": first.executable,
        "shot": first.shot,
        "difficulty": first.difficulty,
        "stages": [
            {
                "stage_index": stage.stage_index,
                "stage_name": stage.stage_name,
                "input_records": stage.input_records,
                "input_record_stride_bytes": 6 if stage.extended else 2,
                "retail_expected_trailing_input_records": [6, 7] if stage.stage_index == last_index else [2, 3],
                "gameplay_frames": None,
                "deaths": None,
                "bombs_used": None,
            }
            for stage in starts
        ],
        "evidence": "TH08 v6 checksum-verified replay decoded with retail executable identity; stream extents only",
        "limitations": "Trailing-input ranges are retail playback expectations, not an attested measurement for this replay. Exact gameplay frames/death/bomb events require game playback.",
        "required_real_game_state": list(REQUIRED_GAME_STATE),
        "human_reference_created": False,
    }


def build_human_reference(path: str | Path, output: str | Path) -> dict:
    """Fail closed without ever creating a fictional reference file."""
    report = inspect_human_replay(path)
    raise MissingHumanMeasurements(
        "human_reference.json not created: replay lacks observed gameplay frames, death and bomb-use events; "
        + "; ".join(report["required_real_game_state"])
    )


# -- measured-reference construction --------------------------------------
def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _is_count(value: object) -> bool:
    """A real counter value: a non-negative int, never a bool or a float guess."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def cap_from_human_frames(frames: int) -> int:
    """``ceil(1.5 * frames)`` in exact integer arithmetic."""
    if not _is_count(frames) or frames <= 0:
        raise HumanReferenceError(f"human gameplay frames must be a positive int, got {frames!r}")
    return -(-3 * int(frames) // 2)


def stage_name(stage_index: int) -> str:
    if not isinstance(stage_index, int) or not 0 <= stage_index < len(STAGE_NAMES):
        raise HumanReferenceError(f"stage index out of range: {stage_index!r}")
    return STAGE_NAMES[stage_index]


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise HumanReferenceError(reason)


def _measured_at_from_evidence(path: Path) -> str:
    """Evidence-file mtime in local time; EXP-P0-046 stored no in-band clock."""
    return datetime.fromtimestamp(Path(path).stat().st_mtime).astimezone().isoformat(timespec="seconds")


def _measured_set(run: dict, *, position: int) -> dict:
    where = f"runs[{position}]"
    _require(isinstance(run, dict), f"{where}: not an object")

    replay = run.get("replay")
    _require(isinstance(replay, dict), f"{where}: missing replay identity")
    sha = replay.get("sha256")
    _require(_is_sha256(sha), f"{where}: replay sha256 is not a 64-char lowercase hex digest")
    file_name = replay.get("file")
    _require(isinstance(file_name, str) and file_name, f"{where}: replay file name missing")

    difficulty = run.get("difficulty")
    _require(isinstance(difficulty, str) and difficulty in DIFFICULTY_BY_INDEX.values(),
             f"{where}: difficulty {difficulty!r} is not a measured difficulty")

    raw_stages = run.get("stages")
    _require(isinstance(raw_stages, list) and raw_stages, f"{where}: no measured stages")

    stages: list[dict] = []
    seen: set[int] = set()
    for index, stage in enumerate(raw_stages):
        place = f"{where}.stages[{index}]"
        _require(isinstance(stage, dict), f"{place}: not an object")
        order = stage.get("stage_index")
        _require(isinstance(order, int) and not isinstance(order, bool), f"{place}: stage_index missing")
        name = stage_name(order)
        _require(order not in seen, f"{place}: duplicate stage_index {order}")
        seen.add(order)
        frames = stage.get("gameplay_frames")
        _require(_is_count(frames) and frames > 0,
                 f"{place}: gameplay_frames is not a measured positive frame count: {frames!r}")
        deaths = stage.get("deaths")
        _require(_is_count(deaths), f"{place}: deaths is not a measured non-negative count: {deaths!r}")
        bombs = stage.get("bombs_used")
        _require(_is_count(bombs), f"{place}: bombs_used is not a measured non-negative count: {bombs!r}")

        index_difficulty = stage.get("difficulty")
        if isinstance(index_difficulty, int) and not isinstance(index_difficulty, bool):
            expected = DIFFICULTY_BY_INDEX.get(index_difficulty)
            _require(expected == difficulty,
                     f"{place}: stage difficulty index {index_difficulty} contradicts run difficulty {difficulty!r}")

        # Cross-check only: the replay input stream stops when the stage-clear
        # sequence starts, so gameplay frames are >= input frames, and the gap
        # is the clear sequence (EXP-P0-046 bounds it at 600).
        cross = stage.get("replay_input_frames")
        if cross is not None:
            _require(_is_count(cross), f"{place}: replay_input_frames is not a count")
            gap = frames - cross
            _require(0 <= gap <= MAX_CLEAR_SEQUENCE_FRAMES,
                     f"{place}: gameplay frames {frames} minus replay input frames {cross} is outside "
                     f"[0, {MAX_CLEAR_SEQUENCE_FRAMES}]; the numbers are not a consistent measurement")

        entry = {"stage_index": order, "stage_name": name,
                 "gameplay_frames": frames, "deaths": deaths, "bombs_used": bombs}
        for extra in ("stage_cleared", "end_reason"):
            if isinstance(stage.get(extra), (bool, str)):
                entry[extra] = stage[extra]
        for extra in ("score_end", "lives_start", "lives_end"):
            if _is_count(stage.get(extra)):
                entry[extra] = stage[extra]
        if cross is not None:
            entry["replay_input_frames_cross_check"] = cross
        stages.append(entry)

    shots = {stage.get("shot") for stage in raw_stages}
    _require(len(shots) == 1, f"{where}: stages disagree on the shot index")
    shot = shots.pop()
    _require(isinstance(shot, int) and 0 <= shot < len(SHOT_NAMES), f"{where}: shot index out of range")

    stages.sort(key=lambda item: item["stage_index"])
    return {
        "source_replay_file": file_name,
        "source_replay_sha256": sha,
        **({"source_replay_size_bytes": replay["size_bytes"]}
           if _is_count(replay.get("size_bytes")) else {}),
        "difficulty": difficulty,
        "difficulty_index": next(index for index, name in DIFFICULTY_BY_INDEX.items() if name == difficulty),
        "shot": shot,
        "shot_name": SHOT_NAMES[shot],
        "stages": stages,
    }


def _coverage(sets: list[dict]) -> dict:
    measured: dict[str, list[str]] = {}
    for entry in sets:
        measured.setdefault(entry["difficulty"], [])
        for stage in entry["stages"]:
            if stage["stage_name"] not in measured[entry["difficulty"]]:
                measured[entry["difficulty"]].append(stage["stage_name"])
    for names in measured.values():
        names.sort(key=ALL_STAGES.index)
    missing = {difficulty: [name for name in ALL_STAGES if name not in names]
               for difficulty, names in measured.items()}
    return {
        "all_stages": list(ALL_STAGES),
        "measured": measured,
        "missing": missing,
        "complete": all(not gaps for gaps in missing.values()),
    }


CAVEAT_TEXT = (
    "stage coverage is incomplete: no measured human play exists for Stage6A, for "
    "Stage4A on lunatic, for Stage4B on normal, or for Extra; a suite that needs one of "
    "those stages has no human-referenced cap and must not be given one",
    "each replay records only six stages: the per-stage numbers come from the stages the "
    "player actually recorded, not from a full six-stage run of every route",
    "single normal corpus: every normal number comes from one replay (Th8_ud2970.rpy) by "
    "one player, so it is a reference point, not a population estimate",
    "gameplay_frames counts logic frames and includes the stage-clear sequence; the replay "
    "input stream stops at stage clear, so gameplay_frames exceeds "
    "replay_input_frames_cross_check by the clear-sequence length (69-500 frames here). "
    "It is not a rendering-frame count, and dialogue fast-forward makes logic frames exceed "
    "rendered frames",
    "deaths is the engine death-counter delta and bombs_used counts bomb activations: a "
    "deathbomb spends two stock and still counts once, and the death-time bomb refill is "
    "not a use; bombs_stock_spent is deliberately not stored here",
    "the measurement is replay playback in the shadow/null-device engine, not live human "
    "play: there is no human input latency, no menu time, and every stage was played by a "
    "separate process started at that stage's own beginning",
)


def build_reference_from_measurements(measured: dict, *, measured_at: str,
                                      measured_at_source: str,
                                      evidence_path: str | None = None,
                                      engine_sha256: str | None = None,
                                      engine_sha256_source: str | None = None) -> dict:
    """Build the validated reference document from EXP-P0-046 measured evidence.

    ``engine_sha256`` binds the reference to the exact executable that produced
    the measurements; it is required, either from the evidence JSON or from the
    caller (who must have measured it).  Raises :class:`HumanReferenceError` for
    anything that is not a real engine-measured record; never fills a missing
    number with a guess.
    """
    _require(isinstance(measured, dict), "measured evidence is not an object")
    _require(measured.get("kind") == EVIDENCE_KIND,
             f"measured evidence kind is {measured.get('kind')!r}, expected {EVIDENCE_KIND!r}")
    _require(measured.get("human_reference_created") is True,
             "measured evidence does not declare human_reference_created=true")
    runs = measured.get("runs")
    _require(isinstance(runs, list) and runs, "measured evidence has no runs")

    try:
        stamp = datetime.fromisoformat(measured_at)
    except (TypeError, ValueError) as exc:
        raise HumanReferenceError(f"measured_at is not an ISO-8601 timestamp: {measured_at!r}") from exc
    _require(stamp.tzinfo is not None, "measured_at must carry a timezone offset")

    sets = [_measured_set(run, position=index) for index, run in enumerate(runs)]
    engines = {json.dumps(run.get("engine"), sort_keys=True) for run in runs}
    _require(len(engines) == 1, "measured runs disagree on the engine identity")
    raw_engine = runs[0].get("engine")
    _require(isinstance(raw_engine, dict) and isinstance(raw_engine.get("binary"), str),
             "measured evidence has no engine identity")
    engine = dict(raw_engine)
    sha256 = engine_sha256 or engine.get("binary_sha256") or engine.get("sha256")
    _require(_is_sha256(sha256),
             "the engine binary sha256 is missing: pass engine_sha256= (measured with sha256sum on the "
             "exact harness binary that produced the numbers)")
    engine["binary_sha256"] = sha256
    engine["binary_sha256_source"] = (engine_sha256_source if engine_sha256
                                      else "recorded in the measured evidence JSON")

    coverage = _coverage(sets)
    document = {
        "schema": SCHEMA,
        "collection_method": COLLECTION_METHOD,
        "not_from_input_records": True,
        "measured_at": stamp.isoformat(timespec="seconds"),
        "measured_at_source": measured_at_source,
        "measured_evidence": {
            "experiment": measured.get("experiment"),
            "kind": measured.get("kind"),
            "source": measured.get("source"),
            "path": evidence_path,
        },
        "engine": engine,
        "cap_multiplier": CAP_MULTIPLIER,
        "cap_rule": "ceil(cap_multiplier * measured human gameplay_frames of that stage)",
        "coverage": coverage,
        "caveats": list(CAVEAT_TEXT),
        "sets": sets,
    }
    validate_reference(document)
    return document


# -- strict loading -------------------------------------------------------
def validate_reference(document: object) -> dict:
    """Fail closed on any reference document that cannot support a cap claim."""
    _require(isinstance(document, dict), "reference is not a JSON object")
    _require(document.get("schema") == SCHEMA,
             f"reference schema is {document.get('schema')!r}, expected {SCHEMA!r}")
    _require(document.get("collection_method") == COLLECTION_METHOD,
             f"reference collection_method is {document.get('collection_method')!r}")
    _require(document.get("not_from_input_records") is True,
             "reference must declare not_from_input_records=true")
    _require(document.get("cap_multiplier") == CAP_MULTIPLIER,
             f"reference cap_multiplier is {document.get('cap_multiplier')!r}, expected {CAP_MULTIPLIER}")
    try:
        stamp = datetime.fromisoformat(document.get("measured_at", ""))
    except (TypeError, ValueError) as exc:
        raise HumanReferenceError("reference measured_at is not an ISO-8601 timestamp") from exc
    _require(stamp.tzinfo is not None, "reference measured_at must carry a timezone offset")
    caveats = document.get("caveats")
    _require(isinstance(caveats, list) and caveats
             and all(isinstance(item, str) and item for item in caveats),
             "reference must carry non-empty caveats")
    engine = document.get("engine")
    _require(isinstance(engine, dict) and isinstance(engine.get("binary"), str),
             "reference must carry the engine identity")
    _require(_is_sha256(engine.get("binary_sha256")),
             "reference must record the sha256 of the engine binary that produced the numbers")

    sets = document.get("sets")
    _require(isinstance(sets, list) and sets, "reference has no measured sets")
    for position, entry in enumerate(sets):
        where = f"sets[{position}]"
        _require(isinstance(entry, dict), f"{where}: not an object")
        _require(_is_sha256(entry.get("source_replay_sha256")),
                 f"{where}: source_replay_sha256 missing or malformed")
        _require(entry.get("difficulty") in DIFFICULTY_BY_INDEX.values(),
                 f"{where}: difficulty {entry.get('difficulty')!r} is not measured")
        _require(_is_count(entry.get("shot")), f"{where}: shot missing")
        stages = entry.get("stages")
        _require(isinstance(stages, list) and stages, f"{where}: no measured stages")
        indices: set[int] = set()
        for stage_position, stage in enumerate(stages):
            place = f"{where}.stages[{stage_position}]"
            _require(isinstance(stage, dict), f"{place}: not an object")
            order = stage.get("stage_index")
            _require(isinstance(order, int) and 0 <= order < len(STAGE_NAMES), f"{place}: bad stage_index")
            _require(order not in indices, f"{place}: duplicate stage_index {order}")
            indices.add(order)
            _require(stage.get("stage_name") == STAGE_NAMES[order],
                     f"{place}: stage_name does not match stage_index {order}")
            frames = stage.get("gameplay_frames")
            _require(_is_count(frames) and frames > 0, f"{place}: gameplay_frames is not measured")
            _require(_is_count(stage.get("deaths")), f"{place}: deaths is not measured")
            _require(_is_count(stage.get("bombs_used")), f"{place}: bombs_used is not measured")
    return document


def load_reference(path: str | Path) -> dict:
    """Read and validate a reference document; raises if it cannot be trusted."""
    path = Path(path)
    if not path.is_file():
        raise HumanReferenceError(f"no measured human reference at {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HumanReferenceError(f"{path}: not valid JSON ({exc})") from exc
    return validate_reference(document)


def reference_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_reference(document: dict, output: str | Path) -> dict:
    """Validate, then write; an invalid document never reaches the filesystem."""
    validate_reference(document)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    output.write_text(text, encoding="utf-8")
    reloaded = load_reference(output)
    return {"path": str(output), "sha256": reference_sha256(output),
            "measured_at": reloaded["measured_at"], "sets": len(reloaded["sets"]),
            "coverage": reloaded["coverage"]}


# -- cap derivation -------------------------------------------------------
@dataclass(frozen=True)
class HumanCap:
    """One measured stage cap, with the exact human number it came from."""

    stage_name: str
    stage_index: int
    difficulty: str
    shot: int
    shot_name: str
    human_frames: int
    multiplier: float
    cap: int
    source_replay_sha256: str
    source_replay_file: str

    def as_dict(self) -> dict:
        return {
            "stage_name": self.stage_name, "stage_index": self.stage_index,
            "difficulty": self.difficulty, "shot": self.shot, "shot_name": self.shot_name,
            "human_frames": self.human_frames, "cap_multiplier": self.multiplier,
            "cap": self.cap, "source_replay_sha256": self.source_replay_sha256,
            "source_replay_file": self.source_replay_file,
        }


def cap_for(reference: dict, *, difficulty: str, stage_name_value: str) -> HumanCap:
    """The measured cap for one stage; fails closed when the stage was not measured."""
    validate_reference(reference)
    for entry in reference["sets"]:
        if entry["difficulty"] != difficulty:
            continue
        for stage in entry["stages"]:
            if stage["stage_name"] == stage_name_value:
                return HumanCap(
                    stage_name=stage["stage_name"], stage_index=stage["stage_index"],
                    difficulty=entry["difficulty"], shot=entry["shot"],
                    shot_name=SHOT_NAMES[entry["shot"]],
                    human_frames=stage["gameplay_frames"], multiplier=CAP_MULTIPLIER,
                    cap=cap_from_human_frames(stage["gameplay_frames"]),
                    source_replay_sha256=entry["source_replay_sha256"],
                    source_replay_file=entry.get("source_replay_file", "unrecorded"),
                )
    measured = reference.get("coverage", {}).get("measured", {})
    raise HumanReferenceError(
        f"the human reference has no measured play for {stage_name_value} on {difficulty}; "
        f"measured coverage is {json.dumps(measured, sort_keys=True)}"
    )


def cap_for_suite(reference: dict, suite) -> HumanCap:
    """The measured cap for a bench suite; multi-stage suites have no single cap.

    ``suite.stage_indexes`` (or its 0-based alias ``suite.stages``) are engine
    stage indices: ``0`` is ``Stage1``.  A suite therefore gets the cap of the
    stage it actually resets, not of the stage with the same 1-based ordinal.
    """
    stage_indexes = tuple(getattr(suite, "stage_indexes", None) or suite.stages)
    if len(stage_indexes) != 1:
        raise HumanReferenceError(
            f"suite {suite.name} declares {len(stage_indexes)} stages; a per-stage human cap cannot be "
            "collapsed into one number by this module"
        )
    name = SUITE_STAGE_NAMES.get(int(stage_indexes[0]))
    if name is None:
        raise HumanReferenceError(
            f"suite {suite.name}: stage_index {stage_indexes[0]} has no measured name")
    return cap_for(reference, difficulty=suite.difficulty, stage_name_value=name)


def caps_for(reference: dict) -> list[dict]:
    """Every derivable cap in the reference, for CLI reporting."""
    return [
        dict(cap_for(reference, difficulty=entry["difficulty"],
                     stage_name_value=stage["stage_name"]).as_dict(),
             collection_method=reference["collection_method"])
        for entry in reference["sets"]
        for stage in entry["stages"]
    ]


__all__ = [
    "SCHEMA", "COLLECTION_METHOD", "CAP_MULTIPLIER", "STAGE_NAMES", "SUITE_STAGE_NAMES",
    "MissingHumanMeasurements", "HumanReferenceError", "HumanCap",
    "inspect_human_replay", "build_human_reference", "build_reference_from_measurements",
    "validate_reference", "load_reference", "reference_sha256", "write_reference",
    "cap_from_human_frames", "cap_for", "cap_for_suite", "caps_for", "stage_name",
]
