"""Offline tests for ``tools/rl/v8_32_relabel.py`` -- no engine is started.

Every engine interaction is a *fake object*: the tool's client seam
(``RelabelDeps.client_factory``) returns :class:`FakeTeacherEngine`, the teacher
seam (``RelabelDeps.decide_call``) returns scripted :class:`FakeDecision` objects,
and the label sink is replaceable too.  So the orchestration is pinned without a
process, which is the only way these checks can run in CI.

What the fake engine models, and why these are not vacuous
----------------------------------------------------------
``hash_at(frame)`` is the tape's own rule for the synthesized shard
(``(frame - 16) // 2`` as 16 hex digits) and the student corpus is written with
the **same** rule, so a correct replay chains and any single override makes one
slot disagree.  A test that expects a failure therefore fails for the intended
reason, not because the fixture was broken to begin with: the happy path uses the
identical fixture and asserts the labels were written.

Test map (the numbering is the work package's)
---------------------------------------------
1  prefix replay call sequence (lead-in + tape slots, STEP by STEP)
2  a hash mismatch is fail-closed and writes no label
3  exactly one ``decide(record_source="child")`` per student decision
4  the 18 scores / 18 mask flags / the count are written back
5  ``chosen`` is the argmax **inside** ``verified_mask``
6  the ``executed == chosen`` agreement statistics
7  ``--limit`` truncates (and says so)
8  the source self-check refuses a bad row count / hash chain / seed
9  a write failure is fail-closed and keeps the rows already written
10 ``meta.json`` carries every documented field
11 the label fields pass ``touhou_rl.v420.reslot.check_label_record``
12 the entry gate (frame 16: nothing else checks that hash)
13 ``decide`` must not move the shadow
14 a teacher exception is fail-closed
15 the child-hazard probe runs once, before the first query
16 the mute preflight refuses before the engine exists
17 the engine sha gate refuses before the engine exists
18 a non-finite score is written as ``null`` and listed, never floored
19 ``--out`` refuses a non-empty directory / the source directory
20 the CLI surface maps onto ``RelabelConfig``
21 ``check_label_fields`` bites on a fabricated record
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for entry in (str(REPO), str(REPO / "tools" / "rl")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import v8_32_relabel as rl  # noqa: E402
import v8_32_takeover as tk  # noqa: E402
import v400_make_cfg  # noqa: E402
from touhou_rl.v400.gym.actions import cand_to_mask  # noqa: E402
from touhou_rl.v420 import contracts as C  # noqa: E402
from touhou_rl.v420 import reslot  # noqa: E402
from touhou_rl.v400.teacher.shoot import release_due  # noqa: E402

SEED = 12
ENTRY = 64
ENTRY_INDEX = ENTRY // 2
TAPE_SLOTS = 40
LEAD_IN = 8
PERIOD = 48
#: The candidate cycle the corpus executes (all distinct, all in range).
CANDS = (7, 0, 3, 12, 16, 1, 5, 9, 2, 4, 6, 8)


# ---------------------------------------------------------------------------
# Fixtures: the tape, the student corpus, a muted data dir, the engine file
# ---------------------------------------------------------------------------
def tape_hash(frame: int) -> str:
    """The synthesized tape's hash rule: one hash per two-frame slot."""

    if frame < tk.TAPE_FIRST_FRAME:
        return "ffffffffffffffff"
    return f"{(frame - tk.TAPE_FIRST_FRAME) // tk.DECISION_FRAMES:016x}"


def write_tape(path: Path, *, slots: int = TAPE_SLOTS, seed: int = SEED,
               lead_in: int = LEAD_IN) -> Path:
    """One synthetic T04 ``slots.jsonl`` on the 2-frame grid, chained."""

    lines: list[str] = []
    for index in range(slots):
        frame = tk.TAPE_FIRST_FRAME + tk.DECISION_FRAMES * index
        cand = CANDS[index % len(CANDS)]
        row = {
            "schema": "v420-dataset-2f/1/slot",
            "episode_seed": int(seed),
            "slot_id": index,
            "decision_frame": frame,
            "commit_frames": [frame, frame + 2],
            "duration_frames": 2,
            "mask": int(cand_to_mask(cand, 0)),
            "executed_cand": cand,
            "bomb": 0,
            "hash_pre": tape_hash(frame),
            "hash_post": tape_hash(frame + 2),
            "history_frames": [2 * step for step in range(lead_in)],
            "history": [7] * lead_in,
            "chosen": cand,
        }
        lines.append(json.dumps(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def student_row(*, frame: int, cand: int, bomb: int = 0, seed: int = SEED,
                period: int = PERIOD) -> dict:
    """One ``--save-obs`` row, written the way the takeover runner writes it."""

    masks = tk.checkpoint_masks(cand, bomb, frame // tk.DECISION_FRAMES, period)
    return {
        "schema": tk.SCHEMA_OBS_SLOT,
        "dataset_version": tk.SCHEMA_OBS_SLOT,
        "contract": tk.SCHEMA_OBS_SLOT,
        "obs_contract": "obs-v3",
        "action_contract": "cand 0..17 + bomb 0..1, 2 frames per decision",
        "episode_seed": int(seed),
        "slot_id": None,
        "shard_key": None,
        "provenance": None,
        "decision": frame // tk.DECISION_FRAMES,
        "window_decision": 0,
        "decision_frame": int(frame),
        "commit_frames": [int(frame), int(frame) + tk.DECISION_FRAMES],
        "duration_frames": tk.DECISION_FRAMES,
        "obs_render_frames": [frame - 4 + offset for offset in range(4)],
        "history_frames": [frame - 2 * (8 - position) for position in range(8)],
        "history": list(CANDS[:8]),
        "executed_cand": int(cand),
        "executed": {"cand": int(cand), "bomb": int(bomb), "mask": int(masks[0]),
                     "source": "policy_act"},
        "cand": int(cand),
        "bomb": int(bomb),
        "mask": int(masks[0]),
        "masks": [int(mask) for mask in masks],
        "hash_pre": tape_hash(int(frame)),
        "hash_post": tape_hash(int(frame) + tk.DECISION_FRAMES),
        "state": {"deaths": 0.0, "lives": 3.0, "bombs": 3.0, "score": 0,
                  "stage": 0, "frame": int(frame)},
        "events": [],
        "end": "",
        "runner_frames": {"obs_render": [frame - 4 + offset for offset in range(4)],
                          "commit": [frame + 1, frame + 2]},
        "obs": {"play": "obs/play96x112.npy", "hud": "obs/hud.npy",
                "act_hist": "obs/act_hist.npy", "row": 0, "hud_frame": frame - 1},
        "label_source": tk.OBS_LABEL_SOURCE,
        "label_verified": None,
        "scores": None,
        "verified_mask": None,
        "verified_count": None,
        "chosen": None,
        "chosen_verified": None,
        "chosen_score": None,
        "teacher_bomb": None,
        "margin": None,
        "obs_source": tk.OBS_SOURCE,
        "classification": "measured",
    }


def write_source(root: Path, *, entry_frame: int = ENTRY, decisions: int = 4,
                 cands: "list[int] | None" = None, seed: int = SEED,
                 period: int = PERIOD, mutate=None) -> Path:
    """One synthetic ``--save-obs`` corpus of ``decisions`` rows."""

    cands = list(cands if cands is not None else CANDS[:decisions])
    assert len(cands) == decisions, (len(cands), decisions)
    rows = []
    for index, cand in enumerate(cands):
        frame = entry_frame + tk.DECISION_FRAMES * index
        row = student_row(frame=frame, cand=int(cand), seed=seed, period=period)
        row["obs"] = dict(row["obs"], row=index)
        row["window_decision"] = index
        if mutate is not None:
            mutate(index, row)
        rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    (root / tk.OBS_SLOTS_FILE).write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8")
    meta = {
        "schema": tk.SCHEMA_OBS_META,
        "created_at": "2026-01-01T00:00:00",
        "seed": int(seed),
        "entry_frame": int(entry_frame),
        "frames_per_decision": tk.DECISION_FRAMES,
        "shoot_pulse_period": int(period),
        "cap_frames": 10_000,
        "decisions": len(rows),
        "frames": 0,
        "stop": "frame_cap",
        "complete": True,
        "label_source": tk.OBS_LABEL_SOURCE,
        "obs_source": tk.OBS_SOURCE,
        "slots": {"file": tk.OBS_SLOTS_FILE, "rows": len(rows), "bytes": 0,
                  "sha256": None},
    }
    (root / tk.OBS_META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True),
                                        encoding="utf-8")
    return root


def make_data_dir(path: Path, *, audible: bool = False) -> Path:
    """A private game-data directory whose ``th08.cfg`` is muted (the template)."""

    import struct

    path.mkdir(parents=True, exist_ok=True)
    data = bytearray(v400_make_cfg.build())
    if audible:
        data[0x1F] = 1
        data[0x20] = 1
        struct.pack_into("<b", data, 0x27, 39)
        struct.pack_into("<b", data, 0x28, 36)
    (path / "th08.cfg").write_bytes(bytes(data))
    return path


def make_engine(path: Path, payload: bytes = b"fake th08-modern binary\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


# ---------------------------------------------------------------------------
# The fakes
# ---------------------------------------------------------------------------
class FakeTeacherEngine:
    """Scripted ``TeacherEngine``: the protocol surface, no process, no frames."""

    def __init__(self, *, overrides: "dict[int, str] | None" = None,
                 child_hazards: "dict | None" = None,
                 step_fault: "dict[int, dict] | None" = None) -> None:
        self.frame_now = 0
        self.steps: list[int] = []
        self.hash_frames: list[int] = []
        self.reset_calls: list[dict] = []
        self.probe_calls: list[str] = []
        self.closed = False
        self.close_code = 0
        self.overrides = dict(overrides or {})
        self.step_fault = dict(step_fault or {})
        self.child_hazards = (dict(child_hazards) if child_hazards is not None
                              else {"available": True, "reason": None})

    # -- lifecycle ------------------------------------------------------
    def close(self) -> int:
        self.closed = True
        return self.close_code

    # -- protocol -------------------------------------------------------
    def reset(self, *, stage: int, difficulty: int, team: int, seed: int):
        self.reset_calls.append({"stage": int(stage), "difficulty": int(difficulty),
                                 "team": int(team), "seed": int(seed)})
        self.frame_now = 0
        return {"_verb": "OK", "ready": "1", "stage": str(stage),
                "difficulty": str(difficulty), "team": str(team), "seed": str(seed)}

    def frame(self) -> int:
        return int(self.frame_now)

    def hash(self) -> str:
        self.hash_frames.append(self.frame_now)
        return self.overrides.get(self.frame_now, tape_hash(self.frame_now))

    def step(self, mask: int, frames: int = 1):
        assert int(frames) == 1, "the relabel replay sends one frame per STEP"
        self.steps.append(int(mask))
        fault = self.step_fault.get(self.frame_now)
        if fault is not None:
            return dict(fault)
        self.frame_now += 1
        return {"_verb": "OK", "advanced": "1"}

    def probe_capabilities(self):
        self.probe_calls.append("capabilities")
        return {"child_hazards": "1"}

    def probe_child_hazards(self):
        self.probe_calls.append("child_hazards")
        return dict(self.child_hazards)


@dataclass
class FakePlan:
    """The bits of ``touhou_rl.v400.teacher.search.Plan`` the label reads."""

    first_action: int
    score: float
    duration: int = 2
    death_frame: int | None = None
    verified: bool = True


@dataclass
class FakeLabel:
    """The bits of ``touhou_rl.v400.teacher.label.TeacherLabel`` the label reads."""

    scores: object
    verified_mask: object
    bomb: int = 0
    margin: float = 1.0
    events: list = field(default_factory=list)


@dataclass
class FakeDecision:
    """The bits of ``touhou_rl.v400.teacher.decide.Decision`` the label reads."""

    label: FakeLabel
    chosen: FakePlan | None = None
    timing_ms: dict = field(default_factory=dict)


def make_decision(scores, verified=(), *, bomb: int = 0,
                  margin: float = 1.0) -> FakeDecision:
    """A teacher verdict: the score vector, the verified set, a chosen plan."""

    values = np.asarray(scores, dtype=np.float64).copy()
    mask = np.zeros(rl.N_CAND, dtype=bool)
    for index in verified:
        mask[int(index)] = True
    finite_verified = [int(index) for index in verified
                       if np.isfinite(values[int(index)])]
    chosen = None
    if finite_verified:
        best = max(finite_verified, key=lambda index: (float(values[index]), -index))
        chosen = FakePlan(first_action=best, score=float(values[best]), verified=True)
    return FakeDecision(label=FakeLabel(scores=values, verified_mask=mask, bomb=bomb,
                                        margin=margin),
                        chosen=chosen, timing_ms={"search": 1.0, "verify": 2.0})


class DecideScript:
    """The ``decide`` seam: records every call and returns scripted decisions."""

    def __init__(self, script=None, *, move: int = 0, raises: BaseException | None = None
                 ) -> None:
        self.script = script or (lambda index, engine, kwargs: make_decision(
            np.linspace(0.0, 17.0, rl.N_CAND), verified=(index % rl.N_CAND,)))
        self.calls: list[dict] = []
        self.move = int(move)
        self.raises = raises

    def __call__(self, engine, config, **kwargs):
        self.calls.append({"engine": engine, "config": config, **kwargs})
        if self.move:
            engine.frame_now += self.move
        if self.raises is not None:
            raise self.raises
        return self.script(len(self.calls) - 1, engine, kwargs)


class ClientFactory:
    """The ``client_factory`` seam: hands back one fake engine and counts starts."""

    def __init__(self, engine: FakeTeacherEngine | None) -> None:
        self.engine = engine
        self.calls: list[object] = []

    def __call__(self, config):
        self.calls.append(config)
        if self.engine is None:
            raise AssertionError("the client factory must not be called")
        return self.engine


# ---------------------------------------------------------------------------
# The output directory and the dependency harness
# ---------------------------------------------------------------------------
def base_config(tmp_path: Path, *, entry_frame: int = ENTRY, **overrides
                ) -> rl.RelabelConfig:
    engine = overrides.pop("engine", None) or make_engine(
        tmp_path / "build" / "th08-modern")
    data_dir = overrides.pop("data_dir", None) or make_data_dir(tmp_path / "data")
    fields = {
        "save_obs": tmp_path / "src",
        "tape_slots": tmp_path / "tape" / "slots.jsonl",
        "seed": SEED,
        "entry_frame": int(entry_frame),
        "engine": engine,
        # --engine may be a build directory; the sha is always the binary's
        "expected_sha": overrides.pop("expected_sha", None)
        or rl.sha256_file(tk.resolve_engine(engine)),
        "data_dir": data_dir,
        "out": tmp_path / "out",
        "argv": ("v8_32_relabel.py", "run"),
    }
    fields.update(overrides)
    return rl.RelabelConfig(**fields)


@dataclass
class Harness:
    """A run wired to fakes, plus the recording seams."""

    config: rl.RelabelConfig
    engine: FakeTeacherEngine | None
    factory: ClientFactory
    decide: DecideScript
    deps: rl.RelabelDeps
    sink_factory: object
    ticks: list

    @property
    def out(self) -> Path:
        return Path(self.config.out)

    def meta(self) -> dict:
        return json.loads((self.out / rl.META_FILE).read_text(encoding="utf-8"))

    def labels(self) -> list[dict]:
        path = self.out / rl.LABELS_FILE
        if not path.is_file():
            return []
        return [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def run(self) -> dict:
        return rl.run_relabel(self.config, deps=self.deps, log=lambda _m: None)


def make_harness(tmp_path: Path, *, decisions: int = 4, entry_frame: int = ENTRY,
                 cands=None, mutate=None, engine: FakeTeacherEngine | None = None,
                 decide: DecideScript | None = None, deps_overrides=None,
                 sink_factory=None, **config_overrides) -> Harness:
    """A full run: a tape, a student corpus of ``decisions`` rows, fake seams."""

    write_tape(tmp_path / "tape" / "slots.jsonl")
    cfg_entry = entry_frame
    write_source(tmp_path / "src", entry_frame=cfg_entry, decisions=decisions,
                 cands=cands, mutate=mutate)
    config = base_config(tmp_path, entry_frame=cfg_entry, **config_overrides)
    engine = engine if engine is not None else FakeTeacherEngine()
    factory = ClientFactory(engine)
    script = decide if decide is not None else DecideScript()
    ticks = [0.0]

    def clock() -> float:
        ticks[0] += 1.0
        return ticks[0]

    fields = {
        "client_factory": factory,
        "decide_call": script,
        "writer_factory": (sink_factory if sink_factory is not None
                           else rl.default_writer_factory),
        "clock": clock,
    }
    fields.update(deps_overrides or {})
    deps = rl.RelabelDeps(**fields)
    return Harness(config=config, engine=engine, factory=factory, decide=script,
                   deps=deps, sink_factory=sink_factory, ticks=ticks)


@pytest.fixture()
def happy(tmp_path: Path) -> Harness:
    harness = make_harness(tmp_path)
    harness.result = harness.run()
    return harness


def expected_prefix_masks(tape: dict, entry_frame: int, period: int) -> list[int]:
    """The exact STEP sequence the tape prefix must produce."""

    plan = tk.entry_plan(tape, entry_frame)
    masks: list[int] = []
    for decision in plan["lead_in"]:
        masks.extend(tk.lead_in_masks(decision, period))
    for slot in plan["prefix_slots"]:
        masks.extend(tk.slot_masks(slot, period))
    return masks


# ---------------------------------------------------------------------------
# 1. the prefix replay's call sequence
# ---------------------------------------------------------------------------
def test_prefix_replay_steps_lead_in_then_tape_slots(happy: Harness) -> None:
    tape = tk.load_tape(happy.config.tape_slots)
    expected = expected_prefix_masks(tape, ENTRY, PERIOD)
    assert len(expected) == ENTRY  # 16 lead-in frames + 24 slots * 2 frames
    assert happy.engine.steps[: len(expected)] == expected
    # ... and then the student's own masks, decision by decision
    rows = [json.loads(line) for line in
            happy.config.save_obs.joinpath(tk.OBS_SLOTS_FILE)
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    student = [int(mask) for row in rows for mask in row["masks"]]
    assert happy.engine.steps == expected + student


def test_prefix_replay_hashes_every_tape_slot(happy: Harness) -> None:
    # the RESET hash, then frame 16 (the lead-in anchor) and one hash per prefix
    # slot: 16, 18 .. 64
    assert happy.engine.hash_frames[0] == 0
    assert happy.engine.hash_frames[1:26] == list(range(16, ENTRY + 1, 2))
    assert happy.result["prefix"]["verified_slots"] == (
        (ENTRY - tk.TAPE_FIRST_FRAME) // tk.DECISION_FRAMES)
    assert happy.result["prefix"]["lead_in_verified"] is True
    assert happy.result["counts"]["frames_prefix"] == ENTRY


# ---------------------------------------------------------------------------
# 2. a hash mismatch is fail-closed and writes no label
# ---------------------------------------------------------------------------
def test_entry_hash_mismatch_writes_no_label(tmp_path: Path) -> None:
    # entry at frame 16: nothing but the entry gate checks that hash (replay_prefix
    # is done before it), so this isolates the gate itself.
    engine = FakeTeacherEngine(overrides={16: "deadbeefdeadbeef"})
    harness = make_harness(tmp_path, entry_frame=16, decisions=2, engine=engine)
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_STATE_FAILED
    assert result["stop"] == "entry_hash_mismatch"
    assert result["entry"]["verified"] is False
    assert not (harness.out / rl.LABELS_FILE).exists()
    assert harness.decide.calls == []
    assert harness.engine.closed is True
    meta = harness.meta()
    assert meta["failure"]["stop"] == "entry_hash_mismatch"
    assert meta["counts"]["rows_labelled"] == 0
    assert meta["counts"]["failures"] == 1


def test_prefix_slot_mismatch_abandons_the_run(tmp_path: Path) -> None:
    engine = FakeTeacherEngine(overrides={40: "0123456789abcdef"})
    harness = make_harness(tmp_path, decisions=3, engine=engine)
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_STATE_FAILED
    assert result["stop"] == "entry_hash_mismatch"
    assert result["failure"]["error_kind"] == "RelabelError"
    assert result["failure"]["expected"] == tape_hash(40)
    assert not (harness.out / rl.LABELS_FILE).exists()
    assert harness.decide.calls == []


def test_student_state_mismatch_keeps_the_rows_already_written(tmp_path: Path) -> None:
    # frame 68 is row 1's hash_post: the row that diverges is never written.
    engine = FakeTeacherEngine(overrides={68: "00000000deadbeef"})
    harness = make_harness(tmp_path, decisions=3, engine=engine)
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_STATE_FAILED
    assert result["stop"] == "student_hash_mismatch"
    assert result["counts"]["rows_labelled"] == 1
    assert [row["decision_frame"] for row in harness.labels()] == [ENTRY]
    assert result["labels"]["sha256"]  # the partial file is kept and identified


# ---------------------------------------------------------------------------
# 3. one teacher query per decision, always record_source="child"
# ---------------------------------------------------------------------------
def test_one_child_query_per_decision(happy: Harness) -> None:
    assert len(happy.decide.calls) == 4
    frames = [int(call["frame"]) for call in happy.decide.calls]
    assert frames == [ENTRY, ENTRY + 2, ENTRY + 4, ENTRY + 6]
    for call in happy.decide.calls:
        assert call["record_source"] == "child"
        assert call["committed"] == ()
        assert call["extra_actions"] == ()
        assert call["stage"] == 0 and call["difficulty"] == 1
        assert call["shoot_release"] is False
        assert call["engine"] is happy.engine
    assert happy.engine.reset_calls == [
        {"stage": 0, "difficulty": 1, "team": 0, "seed": SEED}]
    assert happy.result["counts"]["teacher_queries"] == 4


def test_query_happens_before_the_frames_it_labels(tmp_path: Path) -> None:
    """The teacher is asked on the row's hash_pre, i.e. before its masks run."""

    seen: list[tuple] = []
    engine_holder: dict[str, FakeTeacherEngine] = {}

    def script(index, engine, kwargs):
        seen.append((int(kwargs["frame"]), int(engine.frame()), engine.hash()))
        return make_decision(np.zeros(rl.N_CAND), verified=(index,))

    harness = make_harness(tmp_path, decisions=2, decide=DecideScript(script))
    engine_holder["engine"] = harness.engine
    harness.run()
    assert [item[0] for item in seen] == [ENTRY, ENTRY + 2]
    for frame, engine_frame, hash_now in seen:
        assert frame == engine_frame
        assert hash_now == tape_hash(frame)


# ---------------------------------------------------------------------------
# 4. the label fields are written back verbatim
# ---------------------------------------------------------------------------
def test_scores_mask_and_count_are_written_back(tmp_path: Path) -> None:
    scores = [float(index) + 0.5 for index in range(rl.N_CAND)]
    verified = (1, 4, 9)
    harness = make_harness(
        tmp_path, decisions=2,
        decide=DecideScript(lambda index, engine, kwargs: make_decision(
            scores, verified=verified)))
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_OK
    rows = harness.labels()
    assert len(rows) == 2
    for row in rows:
        assert row["scores"] == [round(value, 6) for value in scores]
        assert row["verified_mask"] == [1 if index in verified else 0
                                        for index in range(rl.N_CAND)]
        assert row["verified_count"] == 3
        assert row["label_source"] == "teacher_2f"
        assert row["label_verified"] is True
        assert row["teacher_bomb"] == row["bomb"] == 0
        assert row["margin"] == 1.0
        assert row["schema"] == rl.SCHEMA_LABEL
        assert row["relabel"]["engine_sha256"] == rl.sha256_file(harness.config.engine)
        assert row["relabel"]["tape_sha256"] == tk.load_tape(
            harness.config.tape_slots)["sha256"]
        assert row["relabel"]["entry_hash"] == tape_hash(ENTRY)
        assert row["relabel"]["tolerance"] == rl.PINNED_TOLERANCE
        assert row["relabel"]["record_source"] == "child"
        assert row["relabel"]["at"]
    assert result["counts"]["verified_candidates"] == 6


# ---------------------------------------------------------------------------
# 5. chosen is the argmax inside verified_mask
# ---------------------------------------------------------------------------
def test_chosen_is_the_best_verified_candidate(tmp_path: Path) -> None:
    scores = np.zeros(rl.N_CAND)
    scores[3] = 5.0
    scores[7] = 9.0
    scores[17] = 100.0  # the global maximum, but NOT verified
    harness = make_harness(
        tmp_path, decisions=1,
        decide=DecideScript(lambda index, engine, kwargs: make_decision(
            scores, verified=(3, 7))))
    result = harness.run()
    row = harness.labels()[0]
    assert row["chosen"] == 7
    assert row["chosen_score"] == 9.0
    assert row["chosen_verified"] is True
    assert row["verified_mask"][17] == 0
    assert row["relabel"]["chosen_rule"] == rl.CHOSEN_RULE
    assert result["distribution"]["chosen_verified"] == 1


def test_nothing_verified_falls_back_to_the_best_finite_score(tmp_path: Path) -> None:
    scores = np.full(rl.N_CAND, -np.inf)
    scores[2] = 3.0
    scores[11] = 8.0
    harness = make_harness(
        tmp_path, decisions=1,
        decide=DecideScript(lambda index, engine, kwargs: make_decision(
            scores, verified=())))
    result = harness.run()
    row = harness.labels()[0]
    assert row["chosen"] == 11
    assert row["chosen_verified"] is False
    assert row["label_verified"] is False
    assert row["verified_count"] == 0
    assert result["distribution"]["chosen_unverified_fallback"] == 1


# ---------------------------------------------------------------------------
# 6. the executed == chosen agreement statistics
# ---------------------------------------------------------------------------
def test_agreement_counts_executed_equals_chosen(tmp_path: Path) -> None:
    executed = [7, 0, 3]
    # decision 0 agrees (teacher picks 7), 1 does not (teacher picks 12),
    # 2 agrees (teacher picks 3).
    picks = [7, 12, 3]

    def script(index, engine, kwargs):
        scores = np.zeros(rl.N_CAND)
        scores[picks[index]] = 10.0
        return make_decision(scores, verified=(picks[index],))

    harness = make_harness(tmp_path, decisions=3, cands=executed,
                          decide=DecideScript(script))
    result = harness.run()
    assert [row["chosen"] for row in harness.labels()] == picks
    assert [row["executed"]["cand"] for row in harness.labels()] == executed
    assert result["agreement"]["comparable"] == 3
    assert result["agreement"]["executed_equals_chosen"] == 2
    assert result["agreement"]["ratio"] == pytest.approx(2 / 3, abs=1e-6)
    assert result["agreement"]["verified_only"]["comparable"] == 3
    assert result["agreement"]["verified_only"]["equal"] == 2


# ---------------------------------------------------------------------------
# 7. --limit
# ---------------------------------------------------------------------------
def test_limit_labels_only_the_first_rows(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, decisions=5, limit=2)
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_OK
    assert result["stop"] == "limit_reached"
    assert result["counts"]["rows_labelled"] == 2
    assert result["counts"]["rows_requested"] == 2
    assert result["counts"]["source_rows"] == 5
    assert result["counts"]["truncated"] is True
    assert result["counts"]["teacher_queries"] == 2
    assert [row["decision_frame"] for row in harness.labels()] == [ENTRY, ENTRY + 2]
    assert result["complete"] is True  # the run did what it was asked to do
    assert harness.meta()["counts"]["limit"] == 2


def test_limit_larger_than_the_corpus_is_not_a_truncation(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, decisions=2, limit=99)
    result = harness.run()
    assert result["stop"] == "complete"
    assert result["counts"]["truncated"] is False
    assert result["counts"]["rows_labelled"] == 2


# ---------------------------------------------------------------------------
# 8. the source self-check
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "breakage, stop",
    [
        ("row_count", "source_invalid"),        # meta.decisions != rows on disk
        ("hash_chain", "source_invalid"),       # row 1's hash_pre is not row 0's post
        ("seed", "source_invalid"),             # a row belongs to another episode
        ("masks", "source_invalid"),            # the masks do not reproduce the frames
        ("grid", "source_invalid"),             # an odd decision_frame
        ("no_executed", "source_invalid"),      # the student's own action is gone
        ("labelled", "source_already_labelled"),  # the row already carries a label
    ])
def test_source_self_check_refuses(tmp_path: Path, breakage: str,
                                   stop: str) -> None:
    def mutate(index: int, row: dict) -> None:
        if breakage == "hash_chain" and index == 1:
            row["hash_pre"] = "aaaaaaaaaaaaaaaa"
        elif breakage == "seed" and index == 1:
            row["episode_seed"] = SEED + 1
        elif breakage == "masks" and index == 0:
            row["masks"] = [row["mask"] ^ 0x10, row["mask"] ^ 0x10]
        elif breakage == "grid" and index == 1:
            row["decision_frame"] = int(row["decision_frame"]) + 1
        elif breakage == "no_executed" and index == 0:
            row.pop("executed")
        elif breakage == "labelled" and index == 0:
            row["scores"] = [0.0] * rl.N_CAND

    if breakage == "row_count":
        harness = make_harness(tmp_path, decisions=3)
        slots = harness.config.save_obs / tk.OBS_SLOTS_FILE
        lines = slots.read_text(encoding="utf-8").splitlines()
        slots.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    else:
        harness = make_harness(tmp_path, decisions=3, mutate=mutate)
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_REFUSED
    assert result["stop"] == stop
    assert harness.factory.calls == []           # no engine was started
    assert harness.decide.calls == []
    assert not (harness.out / rl.LABELS_FILE).exists()
    meta = harness.meta()
    assert meta["failure"]["stop"] == stop
    assert meta["counts"]["rows_labelled"] == 0
    assert meta["entry"]["verified"] is False


def test_entry_frame_mismatch_is_refused(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, decisions=2)
    # the corpus was recorded at the entry, so another entry frame is another run
    result = rl.run_relabel(dataclasses_replace(harness.config, entry_frame=ENTRY + 2),
                            deps=harness.deps, log=lambda _m: None)
    assert result["exit_code"] == rl.EXIT_REFUSED
    assert result["stop"] == "source_invalid"
    assert harness.factory.calls == []


def test_broken_tape_is_refused_before_the_engine(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, decisions=2)
    tape_path = harness.config.tape_slots
    lines = tape_path.read_text(encoding="utf-8").splitlines()
    # break the tape's own hash chain: slot 1's hash_pre is no longer slot 0's post
    row = json.loads(lines[1])
    row["hash_pre"] = "1234567890abcdef"
    lines[1] = json.dumps(row)
    tape_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_REFUSED
    assert result["stop"] == "tape_invalid"
    assert harness.factory.calls == []
    assert harness.decide.calls == []


def test_tape_anchor_disagreeing_with_row0_is_refused(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, decisions=2)

    def mutate(index: int, row: dict) -> None:
        # shift the whole chain by a constant: it stays internally consistent, so
        # only the tape-anchor comparison can catch it.
        row["hash_pre"] = f"{0xBBBB + 2 * index:016x}"
        row["hash_post"] = f"{0xBBBB + 2 * index + 2:016x}"

    write_source(harness.config.save_obs, decisions=2, mutate=mutate)
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_STATE_FAILED
    assert result["stop"] == "entry_hash_mismatch"
    assert result["failure"]["tape_anchor_hash"] == tape_hash(ENTRY)
    assert result["failure"]["student_hash_pre"] == f"{0xBBBB:016x}"
    assert harness.factory.calls == []
    assert not (harness.out / rl.LABELS_FILE).exists()


def dataclasses_replace(config: rl.RelabelConfig, **changes) -> rl.RelabelConfig:
    import dataclasses

    return dataclasses.replace(config, **changes)


# ---------------------------------------------------------------------------
# 9. write failures
# ---------------------------------------------------------------------------
def test_sink_creation_failure_is_fail_closed(tmp_path: Path) -> None:
    def refuse(path):
        raise rl.RelabelError(f"cannot create the label file {path}: denied",
                              stop="write_failed")

    harness = make_harness(tmp_path, decisions=2, sink_factory=refuse)
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_TECHNICAL
    assert result["stop"] == "write_failed"
    assert result["counts"]["rows_labelled"] == 0
    assert not (harness.out / rl.LABELS_FILE).exists()
    assert harness.engine.closed is True
    assert harness.meta()["failure"]["stop"] == "write_failed"


def test_sink_write_failure_keeps_the_rows_already_written(tmp_path: Path) -> None:
    class ExplodingSink(rl.LabelSink):
        """Writes the first row, then dies on the second."""

        def write(self, row):
            if self.rows >= 1:
                raise rl.RelabelError(
                    f"cannot append row {self.rows} to {self.path}: disk full",
                    stop="write_failed")
            return super().write(row)

    harness = make_harness(tmp_path, decisions=3,
                           sink_factory=lambda path: ExplodingSink(path))
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_TECHNICAL
    assert result["stop"] == "write_failed"
    assert result["counts"]["rows_labelled"] == 1
    assert len(harness.labels()) == 1
    assert result["labels"]["sha256"] is not None
    meta = harness.meta()
    assert meta["failure"]["stop"] == "write_failed"
    assert meta["labels"]["rows"] == 1
    assert harness.engine.closed is True


# ---------------------------------------------------------------------------
# 10. meta.json
# ---------------------------------------------------------------------------
META_KEYS = (
    "schema", "created_at", "runner", "argv", "source", "tape", "entry", "seed",
    "stage", "difficulty", "team", "engine", "data_dir", "mute", "teacher",
    "reset", "probes", "counts", "prefix", "labels", "distribution", "agreement",
    "stop", "failure", "complete", "exit_code", "wall_s", "field_notes",
    "classification",
)


def test_meta_json_is_complete(happy: Harness) -> None:
    meta = happy.meta()
    for key in META_KEYS:
        assert key in meta, key
    assert meta["schema"] == rl.SCHEMA_META
    assert meta["runner"]["sha256"] == rl.sha256_file(rl.TOOL_PATH)
    assert meta["source"]["rows"] == 4
    assert meta["source"]["slots_sha256"] == rl.sha256_file(
        happy.config.save_obs / tk.OBS_SLOTS_FILE)
    assert meta["entry"]["frame"] == ENTRY
    assert meta["entry"]["verified"] is True
    assert meta["entry"]["hash"] == tape_hash(ENTRY)
    assert meta["entry"]["tape_anchor_hash"] == tape_hash(ENTRY)
    assert meta["entry"]["student_hash_pre"] == tape_hash(ENTRY)
    assert meta["tape"]["sha256"] == tk.load_tape(happy.config.tape_slots)["sha256"]
    assert meta["engine"]["sha256"] == rl.sha256_file(happy.config.engine)
    assert meta["engine"]["matches"] is True
    assert meta["mute"]["before"]["muted"] is True
    assert meta["mute"]["after"]["muted"] is True
    assert meta["teacher"]["queries"] == 4
    assert meta["teacher"]["tolerance"] == rl.PINNED_TOLERANCE
    assert meta["teacher"]["snapshot"]["replan_decisions"] == 1
    assert meta["reset"]["reply"]["ready"] == "1"
    assert meta["probes"]["child_hazards"]["available"] is True
    assert meta["counts"] == {
        "source_rows": 4, "rows_requested": 4, "rows_labelled": 4, "limit": None,
        "truncated": False, "frames_prefix": ENTRY, "frames_student": 8,
        "decisions_replayed": 4, "teacher_queries": 4, "verified_slots": 4,
        "verified_candidates": 4, "nonfinite_slots": 0, "failures": 0,
    }
    assert meta["labels"]["rows"] == 4
    assert meta["labels"]["sha256"] == rl.sha256_file(happy.out / rl.LABELS_FILE)
    assert meta["distribution"]["chosen"]
    assert sum(meta["distribution"]["chosen"].values()) == 4
    assert meta["distribution"]["label_verified"] == 4
    assert meta["agreement"]["comparable"] == 4
    assert 0.0 <= meta["agreement"]["ratio"] <= 1.0
    assert meta["stop"] == "complete"
    assert meta["failure"] is None
    assert meta["complete"] is True
    assert meta["exit_code"] == rl.EXIT_OK
    assert meta["wall_s"] > 0
    assert (happy.out / rl.README_FILE).is_file()
    assert "DAgger" in (happy.out / rl.README_FILE).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 11. the label fields against reslot.check_label_record
# ---------------------------------------------------------------------------
def _reslot_record(row: dict) -> dict:
    """Rebuild the T04 record ``reslot`` would have written for this label."""

    mask = int(row["executed"]["mask"])
    boundary = reslot.SlotBoundary(decision_frame=16, block_index=4,
                                   slot_in_block=0, masks=(mask, mask),
                                   executed_cand=int(row["executed"]["cand"]),
                                   executed_bomb=int(row["executed"]["bomb"]),
                                   executed_mask=mask, previous_cand=None)
    view = reslot.DecisionView(
        scores=tuple(row["scores"]),
        verified_mask=tuple(row["verified_mask"]),
        bomb=int(row["bomb"]),
        margin=(float("nan") if row["margin"] is None else float(row["margin"])),
        chosen=row["chosen"],
        chosen_score=row["chosen_score"],
        chosen_duration=int(row["teacher"]["decision_frames"]),
        chosen_death_frame=row["relabel"]["teacher"].get("chosen_death_frame"),
        chosen_verified=bool(row["chosen_verified"]),
        verified_count=int(row["verified_count"]))
    return reslot.build_slot_record(
        boundary=boundary, view=view, seed=SEED, shard_key="synthetic",
        teacher={"mode": "fast", "horizon": 32, "decision_frames": 2, "beam": 24,
                 "verify_k": 1, "replan_decisions": 1, "shoot_pulse_period": PERIOD,
                 "internal_frames": 64, "profile_sha256": None},
        hash_pre=row["hash_pre"], hash_post=row["hash_post"], events=None,
        bombs_left=None, deaths=None, timing_ms=None, committed_tape=(),
        corpus_label={"mask": mask})


def test_label_fields_pass_reslot_check_label_record(happy: Harness) -> None:
    rows = happy.labels()
    assert len(rows) == 4
    for row in rows:
        record = _reslot_record(row)
        errors = reslot.check_label_record(record)
        assert errors == {}, (row["decision_frame"], errors)
        # the label values are the ones this tool wrote, not re-derived ones
        assert record["verified_count"] == row["verified_count"]
        assert record["bomb"] == row["bomb"]


def test_our_checks_cover_every_label_check_reslot_reports(happy: Harness) -> None:
    """Every ``reslot`` check about *label* values is one of ours, same name."""

    provenance_only = {"decision_frame", "commit_frames", "duration", "slot_id",
                       "executed_mask", "executed_matches_corpus"}
    label_checks = set(reslot.CHECK_NAMES) - provenance_only
    assert label_checks <= set(rl.CHECK_NAMES)
    assert set(rl.CHECK_NAMES) - set(reslot.CHECK_NAMES) == {
        "label_source", "label_verified", "teacher_bomb", "executed_preserved",
        "nonfinite_consistent", "relabel_block"}
    # and the record builds at all, which is what the round trip above proves
    assert rl.check_label_fields(happy.labels()[0]) == {}


def test_the_one_value_difference_from_reslot_is_the_null_score(tmp_path: Path) -> None:
    """A candidate the teacher could not plan is ``null``, not a floored ``-inf``.

    This is the *only* place a label row of this format carries a value a T04
    record could not: ``beam_search`` starts at ``-np.inf``, so a degenerate tape
    leaves non-finite entries that ``reslot.check_label_record``'s
    ``scores_finite`` refuses.  A JSON ``-Infinity`` is not even valid JSON, and a
    floor would be a measurement never taken, so the entry is ``null`` and
    ``relabel.nonfinite_scores`` says which -- and this tool's ``scores_finite``
    accepts that **only** when the list matches.
    """

    scores = np.full(rl.N_CAND, -np.inf)
    scores[0] = 1.0
    scores[9] = 5.0
    harness = make_harness(
        tmp_path, decisions=1,
        decide=DecideScript(lambda index, engine, kwargs: make_decision(
            scores, verified=(9,))))
    harness.run()
    row = harness.labels()[0]
    assert row["scores"][4] is None
    assert rl.check_label_fields(row) == {}

    # a T04 record cannot even be built from a null score (build_slot_record
    # rounds every entry with float()), and the raw -inf values are refused by
    # reslot's own scores_finite -- which is exactly why this format says null.
    with pytest.raises(TypeError):
        _reslot_record(row)
    raw = dict(row, scores=[(-np.inf if value is None else value)
                            for value in row["scores"]])
    assert "scores_finite" in reslot.check_label_record(_reslot_record(raw))
    # and a null that is not listed is refused by us too
    broken = json.loads(json.dumps(row))
    broken["relabel"]["nonfinite_scores"] = []
    assert "scores_finite" in rl.check_label_fields(broken)


# ---------------------------------------------------------------------------
# 12..15 engine / teacher protocol failures
# ---------------------------------------------------------------------------
def test_decide_must_not_move_the_shadow(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, decisions=2, decide=DecideScript(move=2))
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_TECHNICAL
    assert result["stop"] == "protocol_error"
    assert "must not advance" in result["failure"]["detail"]
    assert result["counts"]["rows_labelled"] == 0


def test_teacher_exception_is_fail_closed(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, decisions=2,
                           decide=DecideScript(raises=RuntimeError("spawn-template")))
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_TECHNICAL
    assert result["stop"] == "teacher_query_failed"
    assert "spawn-template" in result["failure"]["detail"]
    assert result["counts"]["rows_labelled"] == 0
    assert harness.engine.closed is True


def test_probe_runs_once_before_the_first_query(tmp_path: Path) -> None:
    order: list[str] = []
    engine = FakeTeacherEngine()

    def script(index, engine_obj, kwargs):
        order.append("decide")
        return make_decision(np.zeros(rl.N_CAND), verified=(index,))

    original = engine.probe_capabilities

    def probe_capabilities():
        order.append("probe")
        return original()

    engine.probe_capabilities = probe_capabilities  # type: ignore[method-assign]
    harness = make_harness(tmp_path, decisions=2, engine=engine,
                           decide=DecideScript(script))
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_OK
    assert engine.probe_calls == ["capabilities", "child_hazards"]
    assert order[0] == "probe" and order.count("probe") == 1
    assert result["probes"]["child_hazards"]["available"] is True


def test_child_hazards_unavailable_is_refused_before_any_query(tmp_path: Path) -> None:
    engine = FakeTeacherEngine(child_hazards={"available": False, "reason": "no CHILD"})
    harness = make_harness(tmp_path, decisions=2, engine=engine)
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_TECHNICAL
    assert result["stop"] == "child_hazards_unavailable"
    assert harness.decide.calls == []
    assert not (harness.out / rl.LABELS_FILE).exists()


def test_step_fault_is_fail_closed(tmp_path: Path) -> None:
    # the second frame of row 1 does not advance
    engine = FakeTeacherEngine(step_fault={67: {"_verb": "OK", "advanced": "0"}})
    harness = make_harness(tmp_path, decisions=2, engine=engine)
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_TECHNICAL
    assert result["stop"] == "step_failed"
    assert result["counts"]["rows_labelled"] == 1
    assert harness.labels()[0]["decision_frame"] == ENTRY


# ---------------------------------------------------------------------------
# 16..17 preflight refusals: no engine may be started
# ---------------------------------------------------------------------------
def test_audible_data_dir_is_refused(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, decisions=2,
                           data_dir=make_data_dir(tmp_path / "loud", audible=True))
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_MUTE
    assert result["stop"] == "mute_refused"
    assert harness.factory.calls == []
    assert result["mute"]["before"]["muted"] is False


def test_missing_cfg_is_not_silent(tmp_path: Path) -> None:
    empty = tmp_path / "no-cfg"
    empty.mkdir()
    harness = make_harness(tmp_path, decisions=2, data_dir=empty)
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_MUTE
    assert result["stop"] == "mute_refused"
    assert harness.factory.calls == []


def test_engine_sha_mismatch_is_refused(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, decisions=2,
                           expected_sha="0" * 64)
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_REFUSED
    assert result["stop"] == "engine_sha_mismatch"
    assert harness.factory.calls == []
    assert result["engine"]["matches"] is False


def test_missing_data_dir_is_refused(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, decisions=2,
                           data_dir=tmp_path / "does-not-exist")
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_REFUSED
    assert result["stop"] == "data_dir_missing"
    assert harness.factory.calls == []


def test_unknown_teacher_profile_is_refused(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, decisions=2, profile="no-such-profile")
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_REFUSED
    assert result["stop"] == "config_invalid"
    assert harness.factory.calls == []


# ---------------------------------------------------------------------------
# 18. non-finite scores
# ---------------------------------------------------------------------------
def test_non_finite_score_becomes_null_and_is_listed(tmp_path: Path) -> None:
    scores = np.full(rl.N_CAND, -np.inf)
    scores[0] = 1.0
    scores[6] = 4.0
    harness = make_harness(
        tmp_path, decisions=1,
        decide=DecideScript(lambda index, engine, kwargs: make_decision(
            scores, verified=(6,))))
    result = harness.run()
    row = harness.labels()[0]
    assert row["scores"][0] == 1.0
    assert row["scores"][6] == 4.0
    assert row["scores"][1] is None
    nonfinite = row["relabel"]["nonfinite_scores"]
    assert nonfinite == [index for index in range(rl.N_CAND) if index not in (0, 6)]
    assert row["chosen"] == 6
    assert result["counts"]["nonfinite_slots"] == 1
    assert rl.check_label_fields(row) == {}
    # a fabricated floor instead of the null is refused
    broken = json.loads(json.dumps(row))
    broken["scores"][1] = -1.0e9
    assert "scores_finite" in rl.check_label_fields(broken)


# ---------------------------------------------------------------------------
# 19. --out
# ---------------------------------------------------------------------------
def test_out_must_be_new_or_empty(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, decisions=2)
    harness.out.mkdir(parents=True)
    (harness.out / "leftover.txt").write_text("x", encoding="utf-8")
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_REFUSED
    assert result["stop"] == "out_dir_not_empty"
    assert harness.factory.calls == []


def test_out_may_not_be_the_source_dir(tmp_path: Path) -> None:
    harness = make_harness(tmp_path, decisions=2)
    result = rl.run_relabel(dataclasses_replace(harness.config,
                                                out=harness.config.save_obs),
                            deps=harness.deps, log=lambda _m: None)
    assert result["exit_code"] == rl.EXIT_REFUSED
    assert result["stop"] == "out_dir_invalid"
    assert harness.factory.calls == []


# ---------------------------------------------------------------------------
# The real seams, still without a process
# ---------------------------------------------------------------------------
def test_default_client_factory_starts_a_shadow_engine(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--engine <build-dir>`` must reach ``TeacherEngine`` as a binary file."""

    build = tmp_path / "build"
    binary = make_engine(build / "th08-modern")
    config = base_config(tmp_path, engine=build)
    started: list[dict] = []

    class FakeEngine:
        def __init__(self, binary_arg, data_dir, *, role="shadow", timeout=180.0):
            started.append({"binary": Path(binary_arg), "data_dir": Path(data_dir),
                            "role": role, "timeout": timeout, "hello": 0})

        def start(self):
            started[-1]["started"] = True
            return self

        def hello(self):
            started[-1]["hello"] += 1
            return {}

    monkeypatch.setattr("touhou_rl.v400.teacher.engine.TeacherEngine", FakeEngine)
    engine = rl.default_client_factory(config)
    assert isinstance(engine, FakeEngine)
    assert started[0]["binary"] == binary.resolve()
    assert started[0]["data_dir"] == Path(config.data_dir)
    assert started[0]["role"] == "shadow"   # never the env role: no pixels, no GL
    assert started[0]["timeout"] == rl.DEFAULT_TIMEOUT_S
    assert started[0]["started"] is True
    assert started[0]["hello"] == 1


def test_default_engine_gate_hashes_the_binary(tmp_path: Path) -> None:
    config = base_config(tmp_path)
    gate = rl.default_engine_gate(config)
    assert gate["sha256"] == rl.sha256_file(config.engine)
    assert gate["matches"] is True
    assert gate["role"] == "shadow"
    assert Path(gate["path"]).is_file()


def test_default_engine_gate_refuses_a_wrong_sha(tmp_path: Path) -> None:
    config = base_config(tmp_path, expected_sha="1" * 64)
    with pytest.raises(rl.RelabelError) as caught:
        rl.default_engine_gate(config)
    assert caught.value.stop == "engine_sha_mismatch"
    assert caught.value.detail["engine"]["matches"] is False
    assert caught.value.detail["engine"]["sha256"] == rl.sha256_file(config.engine)


def test_default_teacher_config_is_the_two_frame_profile() -> None:
    config = rl.RelabelConfig(
        save_obs=Path("src"), tape_slots=Path("tape.jsonl"), seed=12,
        entry_frame=64, engine=Path("th08-modern"), expected_sha="ab" * 32,
        data_dir=Path("data"), out=Path("out"))
    teacher = rl.default_teacher_config(config)
    assert teacher.replan_decisions == 1
    assert teacher.decision_frames == tk.DECISION_FRAMES
    assert teacher.shoot_pulse_period == PERIOD
    assert teacher.mode == rl.DEFAULT_PROFILE


def test_default_decide_call_is_the_shipped_teacher(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The teacher seam must be ``touhou_rl.v400.teacher.decide.decide`` itself."""

    seen: dict = {}

    def fake_decide(engine, config, **kwargs):
        seen.update(engine=engine, config=config, kwargs=kwargs)
        return "decision"

    import importlib

    module = importlib.import_module("touhou_rl.v400.teacher.decide")
    monkeypatch.setattr(module, "decide", fake_decide)
    out = rl.default_decide_call("E", "C", frame=16, record_source="child",
                                 committed=())
    assert out == "decision"
    assert seen["engine"] == "E" and seen["config"] == "C"
    assert seen["kwargs"] == {"frame": 16, "record_source": "child",
                              "committed": ()}


def test_prefix_client_adapter_speaks_the_replayer_shape() -> None:
    """``replay_prefix`` needs ``step``/``state_hash``; the teacher engine has
    ``step``/``hash``/``frame``.  The adapter is the only place they meet."""

    engine = FakeTeacherEngine()
    engine.frame_now = 32
    client = rl.PrefixClient(engine)
    assert client.step(0x0041, 1) == {"_verb": "OK", "advanced": "1"}
    assert engine.steps == [0x0041]
    value, frame = client.state_hash()
    assert frame == 33 and value == tape_hash(33)
    assert client.frame() == 33


# ---------------------------------------------------------------------------
# 20. the CLI surface
# ---------------------------------------------------------------------------
def test_cli_maps_onto_the_config(tmp_path: Path) -> None:
    parser = rl.build_parser()
    argv = ["run", "--save-obs", str(tmp_path / "src"), "--tape-slots",
            str(tmp_path / "tape.jsonl"), "--seed", "12", "--entry-frame", "64",
            "--engine", str(tmp_path / "th08-modern"), "--expected-sha", "ab" * 32,
            "--data-dir", str(tmp_path / "data"), "--out", str(tmp_path / "out"),
            "--limit", "3", "--no-xvfb"]
    args = parser.parse_args(argv)
    config = rl.config_from_args(args)
    assert config.seed == 12 and config.entry_frame == 64
    assert config.limit == 3 and config.team == 0
    assert config.shoot_pulse_period == rl.DEFAULT_PULSE_PERIOD
    assert config.profile == rl.DEFAULT_PROFILE
    assert config.timeout_s == rl.DEFAULT_TIMEOUT_S
    assert isinstance(config.out, Path)
    assert config.argv == ()  # main() stamps it


def test_run_subcommand_is_required() -> None:
    parser = rl.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


# ---------------------------------------------------------------------------
# 21. the write-time validator bites
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("broken, check", [
    ("chosen_outside_mask", "verified_argmax"),
    ("chosen_out_of_range", "chosen_range"),
    ("wrong_count", "verified_count"),
    ("short_scores", "scores_length"),
    ("bad_label_source", "label_source"),
    ("label_verified_wrong", "label_verified"),
    ("bad_bomb", "bomb"),
    ("nan_margin", "margin"),
    ("executed_mask", "executed_preserved"),
    ("no_relabel", "relabel_block"),
    ("teacher_bomb_off", "teacher_bomb"),
])
def test_check_label_fields_bites(happy: Harness, broken: str, check: str) -> None:
    row = json.loads(json.dumps(happy.labels()[0]))
    assert rl.check_label_fields(row) == {}
    if broken == "chosen_outside_mask":
        row["verified_mask"] = [0] * rl.N_CAND
        row["verified_mask"][0] = 1
        row["verified_count"] = 1
        row["chosen"] = 5
        row["chosen_verified"] = True
    elif broken == "chosen_out_of_range":
        row["chosen"] = 99
    elif broken == "wrong_count":
        row["verified_count"] = 7
    elif broken == "short_scores":
        row["scores"] = row["scores"][:17]
    elif broken == "bad_label_source":
        row["label_source"] = "dagger_student_rollout"
    elif broken == "label_verified_wrong":
        row["label_verified"] = not row["label_verified"]
    elif broken == "bad_bomb":
        row["bomb"] = 2
    elif broken == "nan_margin":
        row["margin"] = float("nan")
    elif broken == "executed_mask":
        row["executed"]["mask"] = row["executed"]["mask"] ^ 0x8000
    elif broken == "no_relabel":
        row.pop("relabel")
    elif broken == "teacher_bomb_off":
        row["teacher_bomb"] = 1 - row["teacher_bomb"]
    errors = rl.check_label_fields(row)
    assert check in errors, (broken, errors)


def test_a_label_that_fails_the_checks_is_never_written(tmp_path: Path) -> None:
    """A teacher verdict with no finite verified score is refused, not floored."""

    scores = np.full(rl.N_CAND, -np.inf)
    scores[5] = 2.0
    mask = np.zeros(rl.N_CAND, dtype=bool)
    mask[3] = True  # verified, but its score is -inf
    decision = FakeDecision(
        label=FakeLabel(scores=scores, verified_mask=mask, bomb=0, margin=1.0),
        chosen=FakePlan(first_action=3, score=float("-inf")), timing_ms={})
    harness = make_harness(tmp_path, decisions=1,
                           decide=DecideScript(lambda index, engine, kwargs: decision))
    result = harness.run()
    assert result["exit_code"] == rl.EXIT_TECHNICAL
    assert result["stop"] == "teacher_output_invalid"
    assert result["counts"]["rows_labelled"] == 0
    # the sink was opened (so an unwritable --out fails early) but stayed empty
    assert harness.labels() == []
    assert result["labels"]["rows"] == 0


def test_decision_without_label_is_refused(tmp_path: Path) -> None:
    harness = make_harness(
        tmp_path, decisions=1,
        decide=DecideScript(lambda index, engine, kwargs: SimpleNamespace(chosen=None)))
    result = harness.run()
    assert result["stop"] == "teacher_output_invalid"
    assert result["counts"]["rows_labelled"] == 0
