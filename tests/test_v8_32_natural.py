"""V832-RUN-01: the constant arm and the per-decision telemetry (offline only).

Everything here runs without an engine, a checkpoint or a shared block: the arm
selector and the record/telemetry helpers are pure, and :func:`run_episode` is
exercised against a scripted stand-in for the harness client.  The properties
these tests pin are the ones the default path must keep:

* with neither ``--arm constant`` nor ``--decision-log``, no file is written and no
  extra engine command is sent (the fake client counts them);
* the constant arm and a policy emitting the same candidate produce the same
  frames / deaths / score / end on the same scripted engine;
* a decision-log block is a header line plus exactly one line per decision.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from touhou_rl.v400.gym.protocol import EnemyRow, PlayerSnapshot  # noqa: E402
from touhou_rl.v400.gym.shm import FLAG_PIXELS_VALID  # noqa: E402

TOOL_PATH = REPO_ROOT / "tools" / "rl" / "v400_student_eval.py"
DECISION_LOG_SCHEMA = "v400-student-eval-decision-log/1"
DECISION_SCHEMA = "v400-student-eval-decision-1"

#: A real ``PLAYER`` snapshot: the row keeps its field names, so a rename here is a
#: test failure rather than a silent change of the telemetry contract.
PLAYER_SNAPSHOT = PlayerSnapshot(
    frame=24, hp=3.0, lives=3, bombs=2.0, power=100.0, focus=0, is_youkai=0,
    player_state=0, invuln_state=0, position=(192.0, 384.0),
    hurtbox_min=(190.0, 382.0), hurtbox_max=(194.0, 386.0),
    speed_normal_axial=4.0, speed_normal_diagonal=2.828,
    speed_focused_axial=1.6, speed_focused_diagonal=1.13)

#: The keys every decision row must carry, whatever the arm.
REQUIRED_ROW_KEYS = {
    "schema", "seed", "repeat", "decision", "frame", "frames_per_decision", "cand",
    "mask", "deaths", "lives", "bombs", "score", "stage", "events", "end",
}
HEADER_KEYS = {
    "schema", "arm", "seed", "repeat", "frames_per_decision", "checkpoint",
    "checkpoint_sha256", "obs_spec_version", "obs_spec_digest", "reset",
}


@pytest.fixture(scope="module")
def ev():
    """The eval tool, loaded exactly the way the other v400 tests load it."""

    spec = importlib.util.spec_from_file_location("v400_student_eval_v832", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# A scripted stand-in for EnvHarnessClient + SharedBlock
# ---------------------------------------------------------------------------
RESET_REPLY = {"ready": "1", "role": "env", "stage": "0", "seed": "1000",
               "shm": "p3eval-v832-test", "hash": "0123456789abcdef"}


class FakeEngine:
    """One scripted episode: a STATE stream, no engine and no shared memory.

    ``request("STATE")`` returns ``OK state-<i>`` and the test patches
    ``state_from_reply`` to hand back ``states[i]`` (the last entry repeats), so
    the arithmetic under test -- frames, deaths, score, end -- is fully scripted.
    The counters prove which commands the tool actually sent.
    """

    def __init__(self, states, *, events=("hit",), events_error=None, player=None,
                 player_error=None, enemies=(), enemies_error=None,
                 advanced=1, reset_reply=None, shm_name="p3eval-v832-test"):
        self.states = list(states)
        self.shm_name = shm_name
        self.state_calls = 0
        self.step_calls: list[tuple[int, int]] = []
        self.events_calls = 0
        self.player_calls = 0
        self.enemies_calls = 0
        self.closed = False
        self.reset_args: dict = {}
        self.events_names = tuple(events)
        self.events_error = events_error
        self.player_reading = player
        self.player_error = player_error
        self.enemy_rows = list(enemies)
        self.enemies_error = enemies_error
        self.advanced = int(advanced)
        self.reset_reply = dict(RESET_REPLY if reset_reply is None else reset_reply)

    # -- client surface used by run_episode --------------------------------
    def start(self) -> None:
        return None

    def hello(self) -> dict:
        return {"role": "env"}

    def attach_shm(self) -> None:
        return None

    def readback(self, flag: bool) -> None:
        return None

    def reset(self, **kwargs) -> dict:
        self.reset_args = dict(kwargs)
        return dict(self.reset_reply)

    def request(self, line: str, timeout: float | None = None) -> str:
        if str(line) == "STATE":
            index = self.state_calls
            self.state_calls += 1
            return f"OK state-{index}"
        raise AssertionError(f"unexpected command {line!r}")

    def step(self, mask: int, want: int) -> dict:
        self.step_calls.append((int(mask), int(want)))
        return {"advanced": str(int(want) * self.advanced), "exit": "0",
                "game_over": "0"}

    def events(self):
        self.events_calls += 1
        if self.events_error is not None:
            raise self.events_error
        return SimpleNamespace(names=self.events_names)

    def player(self):
        self.player_calls += 1
        if self.player_error is not None:
            raise self.player_error
        return self.player_reading

    def enemies(self, offset: int = 0, limit: int = 16):
        self.enemies_calls += 1
        if self.enemies_error is not None:
            raise self.enemies_error
        return list(self.enemy_rows)

    def close(self) -> None:
        self.closed = True


class FakeShm:
    """Just enough ``SharedBlock``: a valid-pixels header and a 640x480 frame."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self.closed = False

    def header(self) -> dict:
        return {"flags": FLAG_PIXELS_VALID}      # no warmup step needed

    def pixels(self) -> np.ndarray:
        return np.zeros((480, 640, 4), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class FakeModel:
    """A policy that always emits the same candidate; counts its forward calls."""

    def __init__(self, cand: int = 5, bomb: int = 0) -> None:
        self.cand = int(cand)
        self.bomb = int(bomb)
        self.calls = 0

    def act(self, play, hud):
        self.calls += 1
        return self.cand, self.bomb


class UntouchableModel:
    """Any attribute access fails: the constant arm must never touch a model."""

    def __getattr__(self, name: str):
        raise AssertionError(f"the constant arm touched model.{name}")


def _state(frame: int, *, deaths: float = 0.0, lives: float = 3.0,
           bombs: float = 2.0, score: int | None = None, stage: int = 0) -> dict:
    return {
        "deaths": float(deaths),
        "lives_remaining": float(lives),
        "bombs_remaining": float(bombs),
        "score": int(frame * 100 if score is None else score),
        "stage": int(stage),
        "frame": int(frame * 2),
        "gameplay_frame": int(frame * 2),
    }


def _state_reader(states):
    """Patched ``state_from_reply``: ``OK state-<i>`` -> ``states[i]``."""

    def read(reply: str) -> dict:
        index = int(str(reply).rsplit("-", 1)[-1])
        return dict(states[min(index, len(states) - 1)])

    return read


def _states_until(ev, max_frames: int) -> list[dict]:
    """One state per game frame, then the final read (index == frames)."""

    assert max_frames % ev.DECISION_FRAMES == 0
    return [_state(frame=frame) for frame in range(int(max_frames) + 2)]


def _install(monkeypatch, ev, fake: FakeEngine, states: list[dict]) -> FakeEngine:
    monkeypatch.setattr(ev, "EnvHarnessClient", lambda *a, **k: fake)
    monkeypatch.setattr(ev, "SharedBlock", FakeShm)
    monkeypatch.setattr(ev, "state_from_reply", _state_reader(states))
    return fake


def _run(ev, monkeypatch, tmp_path, *, arm: str = "policy",
         max_frames: int | None = None, cand: int = 5, model=None,
         stop_on_death: bool = False, decision_log=None,
         states: list[dict] | None = None, **fake_kwargs) -> tuple[dict, FakeEngine]:
    max_frames = (ev.DECISION_FRAMES * 2) if max_frames is None else int(max_frames)
    if states is None:
        states = _states_until(ev, max_frames)
    fake = FakeEngine(states, **fake_kwargs)
    engine = tmp_path / "engine.bin"
    engine.write_bytes(b"v832 fake engine")
    if arm == "policy":
        checkpoint = tmp_path / "checkpoint.pt"
        checkpoint.write_bytes(b"v832 fake checkpoint")
        checkpoint_path: str | None = str(checkpoint)
        if model is None:
            model = FakeModel(cand=cand)
    else:
        checkpoint_path = None
    _install(monkeypatch, ev, fake, states)
    record = ev.run_episode(
        engine=str(engine),
        data_dir=str(tmp_path),
        checkpoint_path=checkpoint_path,
        model=model,
        seed=1000,
        stage=0,
        difficulty="normal",
        max_frames=max_frames,
        shm_name="p3eval-v832-test",
        stop_on_death=stop_on_death,
        arm=arm,
        constant_cand=cand,
        decision_log=decision_log,
    )
    assert record["end"] != "error", record.get("error")
    return record, fake


def _read_log(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# 1. parser defaults and the arm contract
# ---------------------------------------------------------------------------
def test_parser_defaults_keep_the_existing_ruler(ev):
    args = ev.build_parser().parse_args(
        ["--engine", "e", "--data-dir", "d", "--checkpoint", "c"])
    assert args.arm == "policy"
    assert args.constant_cand == 0
    assert args.decision_log is None
    # nothing else moved
    assert args.seeds == "1000,1001,1002,1003,1004"
    assert args.repeats == 1
    assert args.max_frames == 19887
    assert args.stage == 0
    assert args.difficulty == "normal"
    assert args.obs_window == "consecutive"
    assert args.stop_on_death is False
    assert args.out_dir == ".tmp_v400_p3/eval"
    assert args.bootstrap_iters == ev.BOOTSTRAP_ITERS
    assert args.ci_level == ev.CI_LEVEL


def test_parser_accepts_the_constant_arm_without_a_checkpoint(ev):
    args = ev.build_parser().parse_args(
        ["--engine", "e", "--data-dir", "d", "--arm", "constant"])
    assert args.checkpoint is None
    assert ev.resolve_arm(arm=args.arm, constant_cand=args.constant_cand,
                          checkpoint=args.checkpoint) == {
        "arm": "constant", "constant_cand": 0, "loads_checkpoint": False}


def test_resolve_arm_rejects_out_of_range_candidates(ev):
    for cand in (18, -1, 99):
        with pytest.raises(ev.EvalError, match="constant-cand"):
            ev.resolve_arm(arm="constant", constant_cand=cand, checkpoint=None)
    # the legal edges are accepted
    assert ev.resolve_arm(arm="constant", constant_cand=0,
                          checkpoint=None)["constant_cand"] == 0
    assert ev.resolve_arm(arm="constant", constant_cand=17,
                          checkpoint=None)["constant_cand"] == 17


def test_resolve_arm_rejects_constant_with_a_checkpoint(ev):
    with pytest.raises(ev.EvalError, match="must not be given --checkpoint"):
        ev.resolve_arm(arm="constant", constant_cand=0, checkpoint="ckpt.pt")


def test_resolve_arm_requires_a_checkpoint_for_the_policy_arm(ev):
    with pytest.raises(ev.EvalError, match="--checkpoint"):
        ev.resolve_arm(arm="policy", constant_cand=0, checkpoint=None)
    with pytest.raises(ev.EvalError, match="--checkpoint"):
        ev.resolve_arm(arm="policy", constant_cand=0, checkpoint="")
    assert ev.resolve_arm(arm="policy", constant_cand=0, checkpoint="c.pt") == {
        "arm": "policy", "constant_cand": 0, "loads_checkpoint": True}


def test_resolve_arm_rejects_an_unknown_arm(ev):
    with pytest.raises(ev.EvalError, match="arm"):
        ev.resolve_arm(arm="teacher", constant_cand=0, checkpoint="c.pt")


def test_main_refuses_bad_arm_inputs_before_creating_anything(ev, tmp_path):
    out_dir = tmp_path / "eval"
    cases = [
        ["--engine", "e", "--data-dir", "d", "--out-dir", str(out_dir)],
        ["--engine", "e", "--data-dir", "d", "--out-dir", str(out_dir),
         "--arm", "constant", "--checkpoint", "c.pt"],
        ["--engine", "e", "--data-dir", "d", "--out-dir", str(out_dir),
         "--arm", "constant", "--constant-cand", "18"],
        ["--engine", "e", "--data-dir", "d", "--out-dir", str(out_dir),
         "--arm", "constant", "--constant-cand", "-1"],
    ]
    for argv in cases:
        with pytest.raises(SystemExit) as caught:
            ev.main(argv)
        assert caught.value.code == 2, f"{argv} exited {caught.value.code}"
        assert not out_dir.exists(), f"{argv} created {out_dir}"


def test_main_refuses_a_missing_decision_log_parent(ev, tmp_path):
    out_dir = tmp_path / "eval"
    log_path = tmp_path / "absent" / "decisions.jsonl"
    with pytest.raises(SystemExit) as caught:
        ev.main(["--engine", "e", "--data-dir", "d", "--checkpoint", "c.pt",
                 "--out-dir", str(out_dir), "--decision-log", str(log_path)])
    assert caught.value.code == 2
    assert not log_path.parent.exists()
    assert not out_dir.exists()


# ---------------------------------------------------------------------------
# 2. the decision row and the log file
# ---------------------------------------------------------------------------
def test_decision_record_has_the_frozen_schema_and_keys(ev):
    mask = ev.gym_actions.cand_to_mask(7, 0)
    row = ev.decision_record(
        seed=1000, repeat=0, decision=3, frame=6, frames_per_decision=2, cand=7,
        mask=mask, readings={"deaths": 1.0, "lives": 2.0, "bombs": 1.0, "score": 4242},
        stage=0, events=("hit", "graze"), end="")
    assert row["schema"] == DECISION_SCHEMA == ev.DECISION_SCHEMA
    assert set(row) == REQUIRED_ROW_KEYS
    assert row["events"] == ["hit", "graze"]
    assert row["end"] == ""
    assert row["cand"] == 7 and row["mask"] == mask
    assert row["deaths"] == 1.0 and row["lives"] == 2.0 and row["bombs"] == 1.0
    assert row["score"] == 4242 and row["stage"] == 0
    assert row["decision"] == 3 and row["frame"] == 6
    assert row["frames_per_decision"] == 2
    assert json.loads(json.dumps(row)) == row      # JSON-native


def test_decision_record_keeps_player_raw_and_enemies_diagnostic(ev):
    player = ev.player_payload(PLAYER_SNAPSHOT)
    row = ev.decision_record(
        seed=1000, repeat=1, decision=0, frame=0, frames_per_decision=2, cand=0,
        mask=ev.gym_actions.cand_to_mask(0, 0),
        readings={"deaths": 0.0, "lives": 3.0, "bombs": 2.0, "score": 0}, stage=0,
        events=(), end="death_stop", player=player,
        enemies=ev.enemies_payload([]))
    assert row["player"]["position"] == [192.0, 384.0]
    assert row["player"]["x"] == 192.0 and row["player"]["y"] == 384.0
    assert row["player"]["hp"] == 3.0 and row["player"]["player_state"] == 0
    assert row["enemies"]["available"] is True
    assert row["enemies"]["count"] == 0 and row["enemies"]["boss_life"] == 0
    assert row["end"] == "death_stop"
    # without a reading the keys are absent, never forged
    bare = ev.decision_record(
        seed=1000, repeat=0, decision=0, frame=0, frames_per_decision=2, cand=0,
        mask=1, readings={"deaths": 0.0, "lives": 3.0, "bombs": 2.0, "score": 0},
        stage=0, events=())
    assert "player" not in bare and "enemies" not in bare


def test_decision_log_block_is_a_header_then_one_row_per_decision(ev, tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    log = ev.open_decision_log(log_path)
    assert log is not None
    log.write(ev.decision_header(
        arm="policy", seed=1000, repeat=0,
        frames_per_decision=ev.DECISION_FRAMES, checkpoint="ckpt.pt",
        checkpoint_sha256="a" * 64, obs_spec_version="obs-v1",
        obs_spec_digest="b" * 64, reset=RESET_REPLY))
    for decision in range(2):
        log.write(ev.decision_record(
            seed=1000, repeat=0, decision=decision, frame=decision * 2,
            frames_per_decision=2, cand=decision, mask=decision,
            readings={"deaths": 0.0, "lives": 3.0, "bombs": 2.0, "score": 0},
            stage=0, events=(), end=""))
    log.close()
    lines = _read_log(log_path)
    assert len(lines) == 3
    header, first, second = lines
    assert header["schema"] == DECISION_LOG_SCHEMA == ev.DECISION_LOG_SCHEMA
    assert set(header) == HEADER_KEYS
    assert header["reset"] == RESET_REPLY
    assert header["arm"] == "policy" and header["seed"] == 1000
    assert first["schema"] == second["schema"] == DECISION_SCHEMA
    assert [first["decision"], second["decision"]] == [0, 1]


def test_decision_log_appends_so_two_runs_do_not_overwrite(ev, tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    with ev.open_decision_log(log_path) as log:
        log.write({"schema": DECISION_LOG_SCHEMA, "arm": "constant"})
    with ev.open_decision_log(log_path) as log:
        log.write({"schema": DECISION_LOG_SCHEMA, "arm": "policy"})
    assert [line["arm"] for line in _read_log(log_path)] == ["constant", "policy"]


def test_decision_log_missing_parent_fails_closed(ev, tmp_path):
    log_path = tmp_path / "absent" / "decisions.jsonl"
    with pytest.raises(ev.DecisionLogError, match="parent directory"):
        ev.open_decision_log(log_path)
    assert not log_path.parent.exists()
    assert not log_path.exists()


def test_decision_log_write_after_close_fails_closed(ev, tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    log = ev.open_decision_log(log_path)
    log.close()
    assert log.closed is True
    with pytest.raises(ev.DecisionLogError, match="write failed"):
        log.write({"schema": DECISION_SCHEMA})
    with pytest.raises(ev.DecisionLogError):
        ev.open_decision_log(tmp_path)          # a directory is not a log file


def test_open_decision_log_none_creates_no_file(ev, tmp_path):
    assert ev.open_decision_log(None) is None
    assert ev.open_decision_log("") is None
    assert list(tmp_path.iterdir()) == []


def test_module_docstring_documents_v832(ev):
    doc = ev.__doc__ or ""
    assert "V832-RUN-01" in doc
    assert "--decision-log" in doc and "--arm {policy,constant}" in doc
    assert "HAZARDS" in doc and "ENEMIES" in doc
    assert "referee-side" in doc


# ---------------------------------------------------------------------------
# 3. ENEMIES diagnostics
# ---------------------------------------------------------------------------
def test_enemies_payload_summarises_the_boss_rows(ev):
    rows = [
        EnemyRow(slot=0, flags1=1 << 0, position=(10.0, 20.0, 0.0), life=100,
                 timer=1),
        EnemyRow(slot=3, flags1=(1 << 0) | (1 << 1), position=(30.0, 40.0, 0.0),
                 life=0x2BC, timer=9),
    ]
    payload = ev.enemies_payload(rows)
    assert payload == {"available": True, "count": 2, "boss_count": 1,
                       "boss_life": 0x2BC, "boss_slots": [3]}


def test_probe_enemies_records_a_refusal_without_raising(ev):
    class Refusing:
        def enemies(self, offset: int = 0, limit: int = 16):
            raise RuntimeError("'ENEMIES 0 16' -> ERR enemies-unavailable")

    payload = ev.probe_enemies(Refusing())
    assert payload["available"] is False
    assert "ERR enemies-unavailable" in payload["reason"]
    assert set(payload) == {"available", "reason"}


def test_probe_enemies_records_an_undecodable_page(ev):
    class Broken:
        def enemies(self, offset: int = 0, limit: int = 16):
            return [object()]          # no flags1/slot/life at all

    payload = ev.probe_enemies(Broken())
    assert payload["available"] is False
    assert payload["reason"]


# ---------------------------------------------------------------------------
# 4. episode identity per arm
# ---------------------------------------------------------------------------
def _base_record(ev, **overrides) -> dict:
    record = {
        "schema": ev.EPISODE_SCHEMA,
        "checkpoint": "c.pt",
        "checkpoint_sha256": "a" * 64,
        "obs_spec_version": ev.obs_spec.OBS_SPEC_VERSION,
        "obs_spec_digest": ev.obs_spec.obs_spec_digest(),
        "seed": 1000,
        "suite": "S1N",
        "stage": 0,
        "difficulty": "normal",
        "max_frames": 19887,
        "frames": 321,
        "deaths": 1.0,
        "score": 1234,
        "end": "frame_cap",
        "privilege": "deploy_pixel",
        "wall_s": 1.0,
        "engine_sha256": "b" * 64,
        "policy_input": "play64 4-frame stack + hud strip; no privileged state",
        "not_a_gate": True,
        "stage_start": 0,
        "stage_end": 0,
        "cleared": False,
        "exit": "controller",
        "stop_on_death": False,
    }
    record.update(overrides)
    return record


def test_policy_identity_adds_only_the_arm_key(ev, tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"policy checkpoint")
    identity = ev.arm_identity(arm="policy", constant_cand=0,
                               checkpoint_path=str(checkpoint))
    assert identity["arm"] == "policy"
    assert identity["checkpoint"] == str(checkpoint)
    assert identity["checkpoint_sha256"] == ev.hashlib.sha256(
        checkpoint.read_bytes()).hexdigest()
    assert "constant_cand" not in identity

    before = _base_record(ev)
    after = {**before, **identity}
    assert set(after) - set(before) == {"arm"}
    for key, value in before.items():
        if key in ("checkpoint", "checkpoint_sha256"):
            continue
        assert after[key] == value
    ev.validate_episode_record(after)


def test_constant_identity_has_empty_checkpoint_and_validates(ev):
    identity = ev.arm_identity(arm="constant", constant_cand=7,
                               checkpoint_path=None)
    assert identity == {"arm": "constant", "constant_cand": 7,
                        "checkpoint": "", "checkpoint_sha256": ""}
    record = {**_base_record(ev), **identity}
    record.update(checkpoint="", checkpoint_sha256="")
    ev.validate_episode_record(record)


def test_validator_refuses_a_constant_arm_that_claims_a_checkpoint(ev):
    record = {**_base_record(ev),
              "arm": "constant", "constant_cand": 3, "checkpoint_sha256": "c" * 64}
    with pytest.raises(ev.EvalError, match="checkpoint identity"):
        ev.validate_episode_record(record)
    no_cand = {**_base_record(ev), "arm": "constant", "checkpoint": "",
               "checkpoint_sha256": ""}
    with pytest.raises(ev.EvalError, match="constant_cand"):
        ev.validate_episode_record(no_cand)
    bad_cand = {**_base_record(ev), "arm": "constant", "constant_cand": 18,
                "checkpoint": "", "checkpoint_sha256": ""}
    with pytest.raises(ev.EvalError, match="constant_cand"):
        ev.validate_episode_record(bad_cand)
    policy_with_cand = {**_base_record(ev), "arm": "policy", "constant_cand": 1}
    with pytest.raises(ev.EvalError, match="constant_cand"):
        ev.validate_episode_record(policy_with_cand)


# ---------------------------------------------------------------------------
# 5. a scripted episode: the two arms agree on everything but the identity
# ---------------------------------------------------------------------------
def test_constant_arm_needs_no_model_and_matches_the_policy_ruler(ev, monkeypatch,
                                                                  tmp_path):
    workdir = tmp_path / "run"
    workdir.mkdir()
    constant, fake = _run(ev, monkeypatch, tmp_path, arm="constant", cand=5,
                          model=UntouchableModel())
    assert fake.step_calls, "the constant arm must still spend the frames"
    assert constant["arm"] == "constant"
    assert constant["constant_cand"] == 5
    assert constant["checkpoint"] == "" and constant["checkpoint_sha256"] == ""
    assert constant["frames"] == ev.DECISION_FRAMES * 2
    assert constant["end"] == "frame_cap"
    assert constant["steps"] == 2 and constant["renders"] == 1 + 4
    assert constant["action_histogram"][5] == 2
    ev.validate_episode_record(constant)

    monkeypatch.undo()
    model = FakeModel(cand=5)
    policy, _fake = _run(ev, monkeypatch, tmp_path, arm="policy", cand=5, model=model)
    assert model.calls == 2
    assert policy["arm"] == "policy"
    assert "constant_cand" not in policy
    assert policy["checkpoint"].endswith("checkpoint.pt")
    # the existing fields are untouched by the arm switch
    for key in ("frames", "steps", "renders", "deaths", "score", "stage_start",
                "stage_end", "cleared", "end", "exit", "action_histogram",
                "action0_share", "focus_classes_covered", "max_frames"):
        assert policy[key] == constant[key], key
    ev.validate_episode_record(policy)
    # ... and running in an empty directory created nothing
    assert list(workdir.iterdir()) == []


def test_run_episode_without_a_decision_log_sends_no_telemetry_command(ev,
                                                                       monkeypatch,
                                                                       tmp_path):
    workdir = tmp_path / "plain"
    workdir.mkdir()
    record, fake = _run(ev, monkeypatch, tmp_path, arm="policy")
    assert record["frames"] == ev.DECISION_FRAMES * 2
    assert fake.events_calls == 0
    assert fake.player_calls == 0
    assert fake.enemies_calls == 0
    assert list(workdir.iterdir()) == []


#: The episode-record key set frozen before V832-RUN-01, copied from the tool as it
#: stood at 1038 lines.  A default run may add nothing but ``arm``.
PRE_V832_EPISODE_KEYS = {
    "schema", "checkpoint", "checkpoint_sha256", "obs_spec_version", "obs_spec_digest",
    "seed", "repeat", "suite", "stage", "difficulty", "max_frames", "decision_frames",
    "obs_window", "window_alignment", "stage_argument_semantics", "stop_on_death",
    "privilege", "policy_input", "referee_state_source", "engine", "engine_sha256",
    "not_a_gate", "not_the_confirm_split", "controller", "final_state_source",
    "frames", "steps", "renders", "pixels_warmup_frames", "deaths", "lives_end",
    "bombs_start", "bombs_end", "bombs_used", "score", "score_start", "state_frame",
    "gameplay_frame", "stage_start", "stage_end", "cleared", "action_histogram",
    "action0_share", "focus_classes_covered", "exit", "end", "wall_s", "harness_role",
    "error",
}


def test_policy_record_adds_only_the_arm_key(ev, monkeypatch, tmp_path):
    record, _fake = _run(ev, monkeypatch, tmp_path, arm="policy")
    assert set(record) - {"arm"} == PRE_V832_EPISODE_KEYS
    assert record["arm"] == "policy"
    # the historical values still mean the same thing
    assert record["suite"] == "S1N"
    assert record["decision_frames"] == ev.DECISION_FRAMES
    assert record["privilege"] == "deploy_pixel"
    assert record["not_a_gate"] is True
    assert record["frames"] == ev.DECISION_FRAMES * 2
    assert record["deaths"] == 0.0
    assert record["score"] == record["state_frame"] * 50      # scripted: 100/frame
    assert record["end"] == "frame_cap"
    assert record["exit"] == "controller"


def test_a_failed_log_write_escapes_the_episode(ev, monkeypatch, tmp_path):
    """Telemetry failure must propagate, not become a quiet ``end="error"`` row."""

    log_path = tmp_path / "decisions.jsonl"
    log = ev.open_decision_log(log_path)
    log.close()                                   # every later write raises
    with pytest.raises(ev.DecisionLogError, match="write failed"):
        _run(ev, monkeypatch, tmp_path, arm="constant", cand=1,
             model=UntouchableModel(), decision_log=log)


def test_decision_log_covers_every_decision_of_a_real_episode(ev, monkeypatch,
                                                              tmp_path):
    log_path = tmp_path / "logs" / "decisions.jsonl"
    log_path.parent.mkdir()
    log = ev.open_decision_log(log_path)
    record, fake = _run(ev, monkeypatch, tmp_path, arm="constant", cand=4,
                        model=UntouchableModel(), decision_log=log,
                        player=PLAYER_SNAPSHOT)
    log.close()
    lines = _read_log(log_path)
    header, rows = lines[0], lines[1:]
    assert header["schema"] == DECISION_LOG_SCHEMA
    assert set(header) == HEADER_KEYS
    assert header["arm"] == "constant"
    assert header["seed"] == 1000 and header["repeat"] == 0
    assert header["frames_per_decision"] == ev.DECISION_FRAMES
    assert header["checkpoint"] == "" and header["checkpoint_sha256"] == ""
    assert header["reset"] == fake.reset_reply
    assert len(rows) == record["steps"] == 2
    assert [row["decision"] for row in rows] == [0, 1]
    assert [row["frame"] for row in rows] == [0, ev.DECISION_FRAMES]
    assert sum(row["frames_per_decision"] for row in rows) == record["frames"]
    for row in rows:
        assert set(row) == REQUIRED_ROW_KEYS | {"player", "enemies"}
        assert row["cand"] == 4
        assert row["mask"] == ev.gym_actions.cand_to_mask(4, 0)
        assert row["end"] == ""
        assert row["enemies"] == {"available": True, "count": 0, "boss_count": 0,
                                  "boss_life": 0, "boss_slots": []}
        assert row["player"]["position"] == [192.0, 384.0]
        assert row["events"] == ["hit"]


def test_the_terminating_decision_carries_the_end_label(ev, monkeypatch, tmp_path):
    states = [
        _state(0),                            # after RESET
        _state(1),                            # after the first unit
        _state(2, deaths=1.0, lives=2.0),     # the death this decision caused
        _state(3, deaths=1.0, lives=2.0),     # the final read
    ]
    log_path = tmp_path / "decisions.jsonl"
    log = ev.open_decision_log(log_path)
    record, _fake = _run(ev, monkeypatch, tmp_path, arm="constant", cand=0,
                         model=UntouchableModel(), states=states,
                         stop_on_death=True, decision_log=log)
    log.close()
    assert record["end"] == "death_stop"
    assert record["frames"] == ev.DECISION_FRAMES
    assert record["deaths"] == 1.0 and record["lives_end"] == 2.0
    rows = _read_log(log_path)[1:]
    assert len(rows) == 1
    assert rows[0]["end"] == "death_stop"
    assert rows[0]["deaths"] == 1.0 and rows[0]["lives"] == 2.0
    assert rows[0]["frames_per_decision"] == ev.DECISION_FRAMES
    ev.validate_episode_record(record)


def test_player_and_enemies_absences_never_drop_the_row(ev, monkeypatch, tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    log = ev.open_decision_log(log_path)
    boss = EnemyRow(slot=2, flags1=(1 << 0) | (1 << 1),
                    position=(1.0, 2.0, 0.0), life=777, timer=5)
    _record, fake = _run(
        ev, monkeypatch, tmp_path, arm="constant", cand=1,
        model=UntouchableModel(), decision_log=log,
        player_error=RuntimeError("'PLAYER' -> ERR player-not-ready reset-required"),
        enemies=[boss], events=("graze",))
    log.close()
    rows = _read_log(log_path)[1:]
    assert len(rows) == 2
    for row in rows:
        assert "player" not in row                    # unavailable -> omitted
        assert row["enemies"]["available"] is True
        assert row["enemies"]["boss_slots"] == [2]
        assert row["enemies"]["boss_life"] == 777
        assert row["events"] == ["graze"]
    assert fake.player_calls == 1                     # probed once, then dropped

    # the engine refusing ENEMIES keeps the row and records the refusal
    monkeypatch.undo()
    log_path = tmp_path / "refused.jsonl"
    log = ev.open_decision_log(log_path)
    _record, _fake = _run(
        ev, monkeypatch, tmp_path, arm="constant", cand=2,
        model=UntouchableModel(), decision_log=log,
        enemies_error=RuntimeError("'ENEMIES 0 16' -> ERR enemies-unavailable"))
    log.close()
    rows = _read_log(log_path)[1:]
    assert len(rows) == 2
    for row in rows:
        assert row["enemies"]["available"] is False
        assert "ERR enemies-unavailable" in row["enemies"]["reason"]
        assert row["cand"] == 2 and row["schema"] == DECISION_SCHEMA
