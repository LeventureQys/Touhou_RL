"""CLI contract tests; no synthetic fixture is a measured episode.

The doubles below speak the real reply *format* so the runner plumbing can be
checked without a game process.  They are never written to the repository bench
directory: every test that stores a row uses a temporary path.
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tools.rl import v400_bench_run
from touhou_rl.v400.bench import reset_guard as reset_guard_mod
from touhou_rl.v400.bench.human_reference import (
    build_reference_from_measurements, load_reference, write_reference,
)
from touhou_rl.v400.bench.results import validate

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "rl" / "v400_bench_run.py"
NORMAL_REPLAY_SHA = "e621086d814fba4838a2c84e4081162524c3c7e84d1bc4ca3f6621f9ca52d96f"


def invoke(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def state_reply(frames: int, *, deaths: float, lives: float, bombs: float, score: int,
                stage: int = 0) -> str:
    """One real-format STATE line: f6..f12 are %08x float bits, the rest are %u.

    ``stage`` is ``f1``, the engine's **0-based** stage index: the double reports
    the stage its ``RESET`` actually asked for, so a 1-based/0-based mix-up cannot
    pass unnoticed (EXP-P0-077).
    """
    values = [0] * 25
    values[0] = frames
    values[1] = int(stage)
    values[3] = frames + 2  # gameplayFrameCounter includes the two setup frames
    values[4] = score
    values[6] = float_bits(deaths)
    values[7] = float_bits(lives)
    values[8] = float_bits(bombs)
    parts = [f"frames={frames}"]
    for index, value in enumerate(values):
        parts.append(f"f{index}={value:08x}" if 6 <= index <= 12 else f"f{index}={value}")
    return "OK " + " ".join(parts)


class FakeEngine:
    """Protocol test double: real reply shapes, no gameplay.

    ``deaths_after_step`` scripts the engine's own game-over rule (a death
    committed with no spare lives left) at that decision step, which is the real
    transition the runner must stop on.
    """

    deaths_after_step: int | None = None
    #: Hash the fake engine reports for ``RESET``; overridden per subclass.
    reset_hash = "0" * 16

    def __init__(self, binary, data_dir, *, distro="Ubuntu", timeout=120.0):
        self.binary = binary
        self.data_dir = data_dir
        self.frames = 0
        self.steps: list[tuple[int, int]] = []
        self.started = False
        self.closed = False
        self.reset_calls: list[dict] = []
        self.step_index = 0

    def start(self) -> None:
        self.started = True

    def hello(self) -> dict:
        return {"_verb": "OK", "proto": "1", "harness": "1", "role": "shadow", "frames": "0"}

    def reset(self, **kwargs) -> dict:
        self.reset_calls.append(kwargs)
        self.frames = 0
        self.step_index = 0
        return {"_verb": "OK", "reset": "1", "ready": "1", "setup_frames": "330",
                "frames": "0", "stage": str(kwargs["stage"]), "hash": self.reset_hash}

    def _counters(self) -> tuple[float, float]:
        if self.deaths_after_step is None or self.step_index < self.deaths_after_step:
            return 0.0, 2.0
        return 3.0, 0.0

    def request(self, line: str) -> str:
        assert line == "STATE"
        deaths, lives = self._counters()
        stage = int(self.reset_calls[-1]["stage"]) if self.reset_calls else 0
        return state_reply(self.frames, deaths=deaths, lives=lives, bombs=3.0, score=12345,
                           stage=stage)

    def step(self, mask: int, frames: int) -> dict:
        self.steps.append((mask, frames))
        self.step_index += 1
        self.frames += frames
        return {"_verb": "OK", "advanced": str(frames), "frame": str(self.frames), "hash": "0" * 16}

    def close(self) -> None:
        self.closed = True


class DyingEngine(FakeEngine):
    """Fake engine that reaches game over on the second decision step."""

    deaths_after_step = 2


class ExplodingEngine(FakeEngine):
    """Reply shape is irrelevant: the engine dies on the first STEP."""

    def step(self, mask: int, frames: int) -> dict:
        raise RuntimeError("harness closed the control pipe")


def provenance() -> dict:
    return {
        "harness_binary": "/tmp/th08-modern",
        "harness_binary_sha256": "0" * 64,
        "harness_commit": "deadbeefcafe",
        "harness_distro": "Ubuntu",
        "harness_wsl_kernel": "6.18-test",
    }


def measured_row(agent: str = "random", seed: int = 1000, **overrides) -> dict:
    hour = 0 if agent == "random" else 1
    row = {
        "schema": v400_bench_run.RESULT_SCHEMA,
        "status": "ok",
        "measured": True,
        "ckpt": "none",
        "agent": agent,
        "suite": "S1N",
        "seed": seed,
        "stages": [0],
        # EXP-P0-077: the declared stage is a 0-based engine stage index.
        "stage_index": 0,
        "stage_name": "Stage1",
        "stage_observed": 0,
        "stage_start_observed": 0,
        "stage_consistent": True,
        "frames": 1748,
        "gameplay_frame": 1753,
        "steps": 874,
        "deaths": 3.0,
        "bombs_used": 0.0,
        "lives_end": 0.0,
        "score": 12345,
        "cleared": False,
        "end": "game_over",
        "terminal_rule": "death committed while livesRemaining<=0",
        "cap_hit": False,
        "wall_s": 2.0,
        "max_frames": 3000,
        "cap_source": "provisional",
        "harness_commit": "deadbeefcafe",
        "harness_binary_sha256": "0" * 64,
        "harness_binary": "/tmp/th08-modern",
        "harness_distro": "Ubuntu",
        "harness_wsl_kernel": "6.18-test",
        "harness_role": "shadow",
        "privilege": "deploy_pixel",
        "started_at": f"2026-11-08T0{hour}:00:{seed - 999:02d}+00:00",
        "not_a_gate": True,
        "not_the_20_seed_dev_set": True,
    }
    row.update(overrides)
    return row


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def fixture_reference(directory: Path, frames: int = 100) -> Path:
    """A reference document for plumbing tests.

    ``frames`` is a test fixture value, not a human measurement: these tests check
    the cap arithmetic and the record plumbing, never a human number.
    """
    document = build_reference_from_measurements(
        {"kind": "real-engine-replay-playback", "human_reference_created": True,
         "runs": [{"difficulty": "normal",
                   "replay": {"file": "fixture.rpy", "sha256": "a" * 64},
                   "engine": {"binary": "/tmp/th08-modern", "binary_sha256": "c" * 64},
                   "stages": [{"stage_index": 0, "difficulty": 1, "shot": 6,
                               "gameplay_frames": frames, "deaths": 0, "bombs_used": 0}]}]},
        measured_at="2026-11-08T23:43:36+08:00", measured_at_source="test fixture",
    )
    path = directory / "human_reference.json"
    write_reference(document, path)
    return path


def dev20_row(agent: str = "random", seed: int = 1000, **overrides) -> dict:
    """One row shaped like a formal 20-seed dev episode (test fixture, not a run)."""
    row = measured_row(agent=agent, seed=seed)
    row.update(
        max_frames=19887, cap_source="human_reference",
        cap_reason="ceil(1.5 x 13258 measured human gameplay frames of Stage1 (normal, shot Marisa))",
        human_frames=13258, cap_multiplier=1.5,
        human_reference_stage="Stage1", human_reference_stage_index=0,
        human_reference_difficulty="normal", human_reference_shot=6,
        human_reference_source_sha256=NORMAL_REPLAY_SHA,
        human_reference_path=str(ROOT / "Document/Update/Alpha/v4.0.0/bench/human_reference.json"),
        human_reference_sha256="b" * 64,
        human_reference_present=True,
        collection="dev20", not_the_20_seed_dev_set=False,
    )
    row.update(overrides)
    return row


# -- pre-existing fail-closed contract ------------------------------------
@pytest.mark.parametrize("agent,suite,seed", [
    ("random", "S1N", "1000"),
    ("still", "PSN", "1099"),
])
def test_missing_reference_blocks_without_creating_it(tmp_path, agent, suite, seed):
    reference = tmp_path / "human_reference.json"
    result = invoke("--agent", agent, "--suite", suite, "--seed", seed,
                    "--human-reference", str(reference))
    assert result.returncode == 3
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert "human_reference.json" in payload["reason"]
    assert not reference.exists()
    assert "score" not in result.stdout and "episode" not in result.stdout


@pytest.mark.parametrize("flag,value", [
    ("--agent", "other"),
    ("--suite", "S1L"),
    ("--seed", "60000"),
    ("--seed", "1100"),
    ("--seed", "unknown"),
])
def test_rejects_unregistered_arguments(tmp_path, flag, value):
    args = {"--agent": "random", "--suite": "S1N", "--seed": "1000"}
    args[flag] = value
    command = [item for pair in args.items() for item in pair]
    result = invoke(*command, "--human-reference", str(tmp_path / "human_reference.json"))
    assert result.returncode == 2
    assert not (tmp_path / "human_reference.json").exists()


@pytest.mark.parametrize("missing", ["--agent", "--suite", "--seed"])
def test_requires_selection(tmp_path, missing):
    args = {"--agent": "random", "--suite": "S1N", "--seed": "1000"}
    del args[missing]
    result = invoke(*(item for pair in args.items() for item in pair),
                    "--human-reference", str(tmp_path / "human_reference.json"))
    assert result.returncode == 2


def test_existing_reference_does_not_claim_measured_result(tmp_path):
    reference = tmp_path / "human_reference.json"
    reference.write_text("{}", encoding="utf-8")
    result = invoke("--agent", "random", "--suite", "S1N", "--seed", "1000",
                    "--human-reference", str(reference))
    assert result.returncode == 3
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert "score" not in payload and "episodes" not in payload


def test_execute_without_reference_is_allowed_and_stays_provisional(tmp_path, monkeypatch):
    """The provisional cap is exactly why a missing reference must not block."""
    row = measured_row()
    calls: list[dict] = []

    def fake_run_episode(**kwargs):
        calls.append(kwargs)
        return row

    monkeypatch.setattr(v400_bench_run, "run_episode", fake_run_episode)
    monkeypatch.setattr(v400_bench_run, "collect_provenance", lambda *a, **k: provenance())
    results = tmp_path / "results.jsonl"
    code = v400_bench_run.main([
        "--agent", "random", "--suite", "S1N", "--seed", "1000", "--execute",
        "--human-reference", str(tmp_path / "missing.json"),
        "--results", str(results),
    ])
    assert code == 0
    assert calls[0]["human_reference_present"] is False
    stored = json.loads(results.read_text(encoding="utf-8"))
    assert stored["cap_source"] == "provisional"
    assert stored["max_frames"] == 3000
    assert not (tmp_path / "missing.json").exists()


# -- real STATE parsing ---------------------------------------------------
def test_parse_state_reads_hex_float_fields():
    reply = state_reply(3000, deaths=3.0, lives=0.0, bombs=2.5, score=98765)
    fields = v400_bench_run.parse_state(reply)
    assert fields["frames"] == 3000
    assert fields["f6"] == float_bits(3.0)
    assert fields["f8"] == float_bits(2.5)
    readings = v400_bench_run._readings(fields)
    assert readings == {"deaths": 3.0, "lives": 0.0, "bombs": 2.5, "score": 98765}


def test_parse_state_rejects_incomplete_replies():
    with pytest.raises(ValueError, match="incomplete"):
        v400_bench_run.parse_state("OK frames=0 f0=0")
    with pytest.raises(ValueError, match="did not return OK"):
        v400_bench_run.parse_state("ERR state-undefined")


def test_repository_path_stays_posix_for_wsl_git():
    assert v400_bench_run.repository_path(
        "/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern"
    ) == "/root/Touhou_RL/third_party/th08-rl"
    assert "\\" not in v400_bench_run.repository_path(
        "/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern"
    )
    assert v400_bench_run.repository_path("th08-modern") is None


# -- one episode through an injected engine -------------------------------
def test_execute_episode_records_measured_shape():
    row = v400_bench_run.run_episode(
        agent="still", suite_name="S1N", seed=1000, max_frames=6,
        binary="/tmp/th08-modern", data_dir="/tmp/game",
        client_factory=FakeEngine, provenance=provenance(),
    )
    assert row["status"] == "ok" and row["measured"] is True
    assert row["frames"] == 6 and row["steps"] == 3 and row["state_frame"] == 6
    assert (row["deaths"], row["lives_end"], row["bombs_used"], row["score"]) == (0.0, 2.0, 0.0, 12345)
    assert row["end"] == "frame_cap" and row["cap_hit"] is True
    assert row["gameplay_frame"] == 8
    assert row["cap_source"] == "provisional" and row["max_frames"] == 6
    assert row["not_a_gate"] is True and row["not_the_20_seed_dev_set"] is True
    assert row["privilege"] == "deploy_pixel"
    assert row["harness_commit"] == "deadbeefcafe" and row["harness_role"] == "shadow"
    validate(row)  # it must be a shape the bench itself accepts


def test_execute_stops_on_the_engine_game_over_rule():
    """A death with no spare lives must end the episode, not the frame cap."""
    row = v400_bench_run.run_episode(
        agent="still", suite_name="S1N", seed=1000, max_frames=100,
        binary="/tmp/th08-modern", data_dir="/tmp/game",
        client_factory=DyingEngine, provenance=provenance(),
    )
    assert row["status"] == "ok"
    assert row["end"] == "game_over" and row["cap_hit"] is False
    assert row["frames"] == 4 and row["steps"] == 2
    assert (row["deaths"], row["lives_end"]) == (3.0, 0.0)
    assert "showRetryMenu" in row["terminal_rule"]
    validate(row)


def test_execute_refuses_multi_stage_suites_without_a_detector():
    row = v400_bench_run.run_episode(
        agent="still", suite_name="PSN", seed=1000, max_frames=6,
        binary="/tmp/th08-modern", data_dir="/tmp/game",
        client_factory=FakeEngine, provenance=provenance(),
    )
    assert row["status"] == "error" and row["measured"] is False
    assert "multi-stage" in row["error"]
    assert "score" not in row and "frames" not in row


def test_failed_attempt_is_recorded_without_numbers(tmp_path):
    results = tmp_path / "results.jsonl"
    row = v400_bench_run.run_episode(
        agent="random", suite_name="S1N", seed=1000, max_frames=6,
        binary="/tmp/th08-modern", data_dir="/tmp/game",
        client_factory=ExplodingEngine, provenance=provenance(),
    )
    assert row["status"] == "error" and row["measured"] is False
    assert "harness closed the control pipe" in row["error"]
    for invented in ("frames", "deaths", "bombs_used", "lives_end", "score", "cleared", "end"):
        assert invented not in row
    v400_bench_run.append_row(results, row)
    stored = results.read_text(encoding="utf-8").strip().splitlines()
    assert len(stored) == 1 and json.loads(stored[0])["status"] == "error"


def test_all_failed_attempts_exit_nonzero(tmp_path, monkeypatch):
    error_row = {
        "schema": v400_bench_run.RESULT_SCHEMA, "status": "error", "measured": False,
        "agent": "random", "suite": "S1N", "seed": 1000, "stages": [0],
        "max_frames": 3000, "cap_source": "provisional", "wall_s": 0.5,
        "error": "RuntimeError: harness closed the control pipe",
        "error_type": "RuntimeError",
    }
    monkeypatch.setattr(v400_bench_run, "run_episode", lambda **kwargs: error_row)
    monkeypatch.setattr(v400_bench_run, "collect_provenance", lambda *a, **k: provenance())
    results = tmp_path / "results.jsonl"
    code = v400_bench_run.main([
        "--agent", "random", "--suite", "S1N", "--seed", "1000", "--execute",
        "--results", str(results), "--scoreboard", str(tmp_path / "scoreboard.md"),
    ])
    assert code == 1
    assert json.loads(results.read_text(encoding="utf-8"))["status"] == "error"


def test_append_row_keeps_existing_bytes():
    with tempfile.TemporaryDirectory() as directory:
        results = Path(directory) / "results.jsonl"
        results.write_text('{"pre-existing": true}\n', encoding="utf-8")
        before = results.read_bytes()
        v400_bench_run.append_row(results, measured_row(seed=1000))
        v400_bench_run.append_row(results, measured_row(seed=1001))
        after = results.read_bytes()
        assert after.startswith(before)
        lines = after.decode("utf-8").strip().splitlines()
        assert len(lines) == 3
        assert json.loads(lines[0]) == {"pre-existing": True}


# -- scoreboard -----------------------------------------------------------
def test_summarize_states_provisional_cap_and_sample_limits(tmp_path):
    rows = [measured_row(agent="random", seed=seed) for seed in (1000, 1001, 1002)]
    rows += [measured_row(agent="still", seed=seed) for seed in (1000, 1001, 1002)]
    results = tmp_path / "results.jsonl"
    scoreboard = tmp_path / "scoreboard.md"
    write_rows(results, rows)
    before = results.read_bytes()

    code = v400_bench_run.main(["--summarize", "--results", str(results),
                                "--scoreboard", str(scoreboard)])
    assert code == 0
    text = scoreboard.read_text(encoding="utf-8")
    assert text.startswith(v400_bench_run.SCOREBOARD_MARKER)
    assert "cap_source" in text and "provisional" in text and "3000" in text
    assert "gate evaluation" in text.lower()
    plain = text.replace("**", "")
    assert "not a gate evaluation" in plain and "not the 20-seed dev set" in plain
    assert "not_a_gate" not in text  # the scoreboard prose, not raw JSON
    assert "3" in text and "seeds per agent" in text
    assert "| random |" in text and "| still |" in text
    assert "| 1002 |" in text
    assert results.read_bytes() == before  # append-only source untouched


def test_summarize_refuses_to_overwrite_a_foreign_scoreboard(tmp_path):
    results = tmp_path / "results.jsonl"
    scoreboard = tmp_path / "scoreboard.md"
    write_rows(results, [measured_row()])
    scoreboard.write_text("hand-written notes\n", encoding="utf-8")
    code = v400_bench_run.main(["--summarize", "--results", str(results),
                                "--scoreboard", str(scoreboard)])
    assert code == 1
    assert scoreboard.read_text(encoding="utf-8") == "hand-written notes\n"


def test_summarize_reports_error_attempts_and_refuses_all_error_results(tmp_path):
    results = tmp_path / "results.jsonl"
    scoreboard = tmp_path / "scoreboard.md"
    error_row = {
        "schema": v400_bench_run.RESULT_SCHEMA, "status": "error", "measured": False,
        "agent": "still", "suite": "S1N", "seed": 1002, "stages": [0],
        "max_frames": 3000, "cap_source": "provisional", "wall_s": 0.25,
        "error": "HarnessError: harness exited before replying (code 1)",
        "error_type": "HarnessError",
    }
    write_rows(results, [measured_row(agent="random", seed=1000), error_row])
    code = v400_bench_run.main(["--summarize", "--results", str(results),
                                "--scoreboard", str(scoreboard)])
    assert code == 0
    text = scoreboard.read_text(encoding="utf-8")
    assert "HarnessError: harness exited before replying" in text
    assert "| random |" in text

    only_errors = tmp_path / "only_errors.jsonl"
    unwritten = tmp_path / "never.md"
    write_rows(only_errors, [error_row])
    code = v400_bench_run.main(["--summarize", "--results", str(only_errors),
                                "--scoreboard", str(unwritten)])
    assert code == 1
    assert not unwritten.exists()


# -- invalidation is additive, never destructive ---------------------------
def test_invalidate_marks_old_rows_without_editing_them(tmp_path):
    stale = measured_row(agent="random", seed=1000, end="frame_cap", cap_hit=True,
                         frames=2998, gameplay_frame=1753,
                         started_at="2026-11-08T00:00:00+00:00")
    fresh = measured_row(agent="random", seed=1000, frames=1748, gameplay_frame=1753)
    results = tmp_path / "results.jsonl"
    scoreboard = tmp_path / "scoreboard.md"
    write_rows(results, [stale, fresh])
    before = results.read_bytes()

    stamp = stale["started_at"]
    code = v400_bench_run.main(["--invalidate", stamp, "--reason", "frame cap inside retry menu",
                                "--results", str(results)])
    assert code == 0
    after = results.read_bytes()
    assert after.startswith(before)  # the old row is untouched
    records = [json.loads(line) for line in after.decode("utf-8").strip().splitlines()]
    assert records[-1]["status"] == "invalid"
    assert records[-1]["invalidated_started_at"] == [stamp]
    assert records[-1]["reason"] == "frame cap inside retry menu"
    assert records[-1]["invalidated_rows"][0]["end"] == "frame_cap"

    code = v400_bench_run.main(["--summarize", "--results", str(results),
                                "--scoreboard", str(scoreboard)])
    assert code == 0
    text = scoreboard.read_text(encoding="utf-8")
    assert "Superseded attempts" in text and "frame cap inside retry menu" in text
    assert "| random | S1N | 0 | Stage1 | 1000 | 1748 |" in text
    assert "2998" not in text  # the invalidated row is not in the table


def test_invalidate_rejects_unknown_rows_and_requires_a_reason(tmp_path):
    results = tmp_path / "results.jsonl"
    write_rows(results, [measured_row()])
    code = v400_bench_run.main(["--invalidate", "2026-01-01T00:00:00+00:00",
                                "--reason", "nope", "--results", str(results)])
    assert code == 1
    with pytest.raises(SystemExit) as caught:
        v400_bench_run.main(["--invalidate", measured_row()["started_at"],
                             "--results", str(results)])
    assert caught.value.code == 2


def test_summarize_excludes_invalid_rows_but_keeps_audit_trail(tmp_path):
    stale = measured_row(seed=1000, end="frame_cap", frames=2998,
                         started_at="2026-11-08T00:00:00+00:00")
    fresh = measured_row(seed=1000, frames=1748)
    results = tmp_path / "results.jsonl"
    scoreboard = tmp_path / "scoreboard.md"
    write_rows(results, [stale, fresh])
    v400_bench_run.main(["--invalidate", stale["started_at"], "--reason", "superseded",
                         "--results", str(results)])
    report = v400_bench_run.summarize_results(results, scoreboard)
    assert report == {"status": "ok", "scoreboard": str(scoreboard),
                      "episodes": 1, "errors": 0, "invalidated": 1,
                      "stage_corrections": 0}
    text = scoreboard.read_text(encoding="utf-8")
    assert "| 1748 |" in text and "2998" not in text


# -- human-referenced cap -------------------------------------------------
def test_resolve_cap_reads_the_reference_and_labels_its_source(tmp_path):
    reference = load_reference(fixture_reference(tmp_path, frames=100))
    cap = v400_bench_run.resolve_cap(reference=reference, suite_name="S1N")
    assert cap["cap_source"] == "human_reference"
    assert (cap["max_frames"], cap["human_frames"], cap["cap_multiplier"]) == (150, 100, 1.5)
    assert cap["human_reference_stage"] == "Stage1"
    assert cap["human_reference_source_sha256"] == "a" * 64
    assert "ceil(1.5 x 100" in cap["cap_reason"]

    provisional = v400_bench_run.resolve_cap(reference=None, suite_name="S1N")
    assert provisional == {"max_frames": 3000, "cap_source": "provisional",
                           "cap_reason": "no measured human reference; operator-chosen provisional cap"}

    override = v400_bench_run.resolve_cap(reference=reference, suite_name="S1N",
                                          explicit_max_frames=42)
    assert override["max_frames"] == 42 and override["cap_source"] == "operator_override"
    assert "human_frames" not in override

    # A suite whose stages the reference never measured keeps the provisional cap,
    # and the reason names the missing coverage instead of inventing a number.
    uncovered = v400_bench_run.resolve_cap(reference=reference, suite_name="PSN")
    assert uncovered["cap_source"] == "provisional" and uncovered["max_frames"] == 3000
    assert "no measured cap for suite PSN" in uncovered["cap_reason"]


def test_run_episode_records_the_human_cap_and_dev20_labels():
    cap = {
        "max_frames": 6, "cap_source": "human_reference",
        "cap_reason": "ceil(1.5 x 4 measured human gameplay frames of Stage1 (normal, shot Marisa))",
        "human_frames": 4, "cap_multiplier": 1.5, "human_reference_stage": "Stage1",
        "human_reference_stage_index": 0, "human_reference_difficulty": "normal",
        "human_reference_shot": 6, "human_reference_source_sha256": NORMAL_REPLAY_SHA,
    }
    row = v400_bench_run.run_episode(
        agent="still", suite_name="S1N", seed=1000, cap=cap, collection="dev20",
        binary="/tmp/th08-modern", data_dir="/tmp/game", client_factory=FakeEngine,
        provenance=provenance(), human_reference_present=True,
        human_reference_path="/tmp/human_reference.json", human_reference_sha256="b" * 64,
    )
    assert row["status"] == "ok" and row["frames"] == 6 and row["end"] == "frame_cap"
    assert (row["max_frames"], row["cap_source"]) == (6, "human_reference")
    assert (row["human_frames"], row["cap_multiplier"]) == (4, 1.5)
    assert row["human_reference_stage"] == "Stage1"
    assert row["human_reference_source_sha256"] == NORMAL_REPLAY_SHA
    assert row["human_reference_present"] is True and row["human_reference_sha256"] == "b" * 64
    assert row["collection"] == "dev20" and row["not_the_20_seed_dev_set"] is False
    assert row["dev_split"] is True and row["not_a_gate"] is True
    validate(row)


def test_cli_execute_resolves_the_cap_from_the_reference(tmp_path, monkeypatch):
    reference = fixture_reference(tmp_path, frames=100)
    calls: list[dict] = []

    def fake_run_episode(**kwargs):
        calls.append(kwargs)
        return dev20_row(seed=1000, max_frames=kwargs["cap"]["max_frames"])

    monkeypatch.setattr(v400_bench_run, "run_episode", fake_run_episode)
    monkeypatch.setattr(v400_bench_run, "collect_provenance", lambda *a, **k: provenance())
    results = tmp_path / "results.jsonl"
    code = v400_bench_run.main([
        "--agent", "still", "--suite", "S1N", "--seed", "1000", "--execute", "--dev20",
        "--human-reference", str(reference), "--results", str(results),
    ])
    assert code == 0
    assert calls[0]["cap"]["cap_source"] == "human_reference"
    assert calls[0]["cap"]["max_frames"] == 150
    assert calls[0]["collection"] == "dev20"
    assert calls[0]["human_reference_present"] is True
    assert len(calls[0]["human_reference_sha256"]) == 64
    stored = json.loads(results.read_text(encoding="utf-8"))
    assert stored["cap_source"] == "human_reference" and stored["max_frames"] == 150


def test_cli_refuses_an_unusable_reference_instead_of_downgrading(tmp_path):
    reference = tmp_path / "human_reference.json"
    reference.write_text('{"schema": "v400-wp7-human-reference/1"}', encoding="utf-8")
    results = tmp_path / "results.jsonl"
    code = v400_bench_run.main([
        "--agent", "random", "--suite", "S1N", "--seed", "1000", "--execute",
        "--human-reference", str(reference), "--results", str(results),
    ])
    assert code == 1
    assert not results.exists()  # nothing measured, nothing recorded


def test_scoreboard_marks_the_formal_dev20_board_and_keeps_legacy_rows(tmp_path):
    rows = [dev20_row(agent="random", seed=1000), dev20_row(agent="still", seed=1000),
            measured_row(agent="random", seed=1000)]
    results = tmp_path / "results.jsonl"
    scoreboard = tmp_path / "scoreboard.md"
    write_rows(results, rows)
    assert v400_bench_run.main(["--summarize", "--results", str(results),
                                "--scoreboard", str(scoreboard)]) == 0
    text = scoreboard.read_text(encoding="utf-8")
    plain = text.replace("**", "")
    assert text.startswith(v400_bench_run.SCOREBOARD_MARKER)
    assert "formal 20-seed dev" in text.lower()
    assert "20-seed dev set" in plain and "not the confirm split" in plain
    assert "not the 20-seed dev set" not in plain  # that claim belongs to the old slice
    assert "gate evaluation" in plain.lower()
    assert "human_reference" in text and "19887" in text and "13258" in text
    assert "not_from_input_records" in text
    assert "Provisional-cap table" in text and "3000" in text
    assert "| random | S1N | 0 | Stage1 | 1000 | 1748 |" in text


# -- EXP-P0-077: stage index (0-based) -------------------------------------
def test_run_episode_records_the_zero_based_stage_and_the_observed_start_stage():
    """S1N resets ``stage_index 0`` and the row proves the engine agreed."""
    row = v400_bench_run.run_episode(
        agent="still", suite_name="S1N", seed=1000, max_frames=100,
        binary="/tmp/th08-modern", data_dir="/tmp/game",
        client_factory=DyingEngine, provenance=provenance(),
    )
    assert row["status"] == "ok"
    assert row["stages"] == [0]
    assert row["stage_index"] == 0 and row["stage_name"] == "Stage1"
    assert "0-based" in row["stage_index_semantics"]
    assert row["reset_stage_start"] == "RESET 0 1 0 1000 0"
    # The engine's own STATE.f1 agrees with the declared index: this is the
    # evidence that used to record 1 (the real Stage 2) while the suite was read
    # as Stage 1.
    assert row["stage_start_observed"] == 0 and row["stage_observed"] == 0
    assert row["stage_consistent"] is True
    validate(row)


def test_a_stage_the_engine_did_not_start_is_flagged_not_silently_relabelled():
    """A double that always reports Stage 2 must make the row say so."""
    class StageTwoEngine(DyingEngine):
        def request(self, line: str) -> str:
            return state_reply(self.frames, deaths=0.0, lives=2.0, bombs=3.0,
                               score=12345, stage=1)

    row = v400_bench_run.run_episode(
        agent="still", suite_name="S1N", seed=1000, max_frames=100,
        binary="/tmp/th08-modern", data_dir="/tmp/game",
        client_factory=StageTwoEngine, provenance=provenance(),
    )
    assert row["stage_index"] == 0           # what the suite asked for
    assert row["stage_start_observed"] == 1  # what the engine actually started
    assert row["stage_consistent"] is False


def test_multi_stage_suite_refuses_and_records_no_stage_index():
    row = v400_bench_run.run_episode(
        agent="still", suite_name="PSN", seed=1000, max_frames=6,
        binary="/tmp/th08-modern", data_dir="/tmp/game",
        client_factory=FakeEngine, provenance=provenance(),
    )
    assert row["status"] == "error" and row["measured"] is False
    assert "multi-stage execution needs a stage-clear detector" in row["error"]
    assert "stage_index" not in row


def test_scoreboard_annotates_every_table_by_zero_based_stage_index(tmp_path):
    rows = [dev20_row(agent="random", seed=1000),
            measured_row(agent="still", seed=1000)]
    results = tmp_path / "results.jsonl"
    scoreboard = tmp_path / "scoreboard.md"
    write_rows(results, rows)
    assert v400_bench_run.main(["--summarize", "--results", str(results),
                                "--scoreboard", str(scoreboard)]) == 0
    text = scoreboard.read_text(encoding="utf-8")
    assert "## Stage index (0-based)" in text
    assert "| S1N | 0 | Stage 1 | Stage1 | human_reference | 1 | declared |" in text
    assert "| S1N | 0 | Stage 1 | Stage1 | provisional | 1 | declared |" in text
    assert "| Agent | Suite | Stage idx | Stage | Seed |" in text
    assert "None: no stage label has been corrected." in text


def test_correct_stage_lists_rows_that_ran_another_stage_and_edits_nothing(tmp_path):
    """The audit record restates the stage; the measured numbers are untouched."""
    legacy = dev20_row(agent="random", seed=1000)
    # A pre-EXP-P0-077 row: no declared index, and the engine reported stage 1.
    for key in ("stage_index", "stage_name", "stage_start_observed", "stage_consistent"):
        legacy.pop(key)
    legacy["stages"] = [1]
    legacy["stage_observed"] = 1
    corrected = dev20_row(agent="still", seed=1000)
    results = tmp_path / "results.jsonl"
    scoreboard = tmp_path / "scoreboard.md"
    write_rows(results, [legacy, corrected])
    before = results.read_bytes()

    code = v400_bench_run.main(["--correct-stage", "--suite", "S1N",
                                "--reason", "suite declared 1; the engine stage is 0-based",
                                "--results", str(results)])
    assert code == 0
    after = results.read_bytes()
    assert after.startswith(before)  # append-only: no historical byte changed
    record = json.loads(after.decode("utf-8").strip().splitlines()[-1])
    assert record["schema"] == v400_bench_run.CORRECTION_SCHEMA
    assert record["status"] == "corrected" and record["measured"] is False
    assert record["observed_stage_indexes"] == [1]
    assert record["stage_index"] == 1 and record["stage_name"] == "Stage2"
    assert record["declared_stage_index"] == 0
    assert record["affected_rows"] == 1
    assert record["affected_started_at"] == [legacy["started_at"]]
    assert "invalidated_started_at" not in record  # never voids a measured row

    assert v400_bench_run.main(["--summarize", "--results", str(results),
                                "--scoreboard", str(scoreboard)]) == 0
    text = scoreboard.read_text(encoding="utf-8")
    assert "## Stage-index corrections (append-only audit records)" in text
    assert "Rows restated" in text and "`stage_index 1` (Stage2)" in text
    assert "never a measured number" in text
    # Both measured rows are still there, with their numbers.
    assert "| 1748 |" in text


def test_correct_stage_refuses_when_every_row_already_carries_the_right_stage(tmp_path):
    results = tmp_path / "results.jsonl"
    write_rows(results, [dev20_row(agent="random", seed=1000)])
    before = results.read_bytes()
    code = v400_bench_run.main(["--correct-stage", "--suite", "S1N", "--reason", "nothing",
                                "--results", str(results)])
    assert code == 1
    assert results.read_bytes() == before


def test_correct_stage_requires_a_reason_and_a_suite(tmp_path):
    results = tmp_path / "results.jsonl"
    write_rows(results, [measured_row()])
    with pytest.raises(SystemExit) as caught:
        v400_bench_run.main(["--correct-stage", "--suite", "S1N", "--results", str(results)])
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as caught:
        v400_bench_run.main(["--correct-stage", "--reason", "x", "--results", str(results)])
    assert caught.value.code == 2


def test_a_correction_record_is_not_read_back_as_a_measured_episode(tmp_path):
    """``tools/rl/v400_bench.py --summarize`` must not validate it as an episode."""
    from touhou_rl.v400.bench.results import load_results

    legacy = measured_row(agent="random", seed=1000)
    for key in ("stage_index", "stage_name", "stage_start_observed", "stage_consistent"):
        legacy.pop(key)
    legacy["stages"] = [1]
    legacy["stage_observed"] = 1
    results = tmp_path / "results.jsonl"
    write_rows(results, [legacy])
    assert v400_bench_run.main(["--correct-stage", "--suite", "S1N", "--reason", "label only",
                                "--results", str(results)]) == 0
    rows = load_results(results)
    assert len(rows) == 1 and rows[0]["seed"] == 1000


# -- reset-hash guard (EXP-P0-071) ----------------------------------------
#: The two hashes EXP-P0-070 run c really recorded for the identical
#: ``RESET 0 1 0 1 0`` (``7f04...`` in five of six processes, ``795b...`` in the
#: divergent one).  They are test constants and the engines below are protocol
#: doubles: no test here claims to have measured an episode.
EXPECTED_RESET_HASH = "7f04c7361749ed9f"
DIVERGENT_RESET_HASH = "795b9441ed0c8056"


class DivergingResetEngine(FakeEngine):
    """Protocol double whose RESET starts from a different StageStart state."""

    reset_hash = DIVERGENT_RESET_HASH


class HashlessResetEngine(FakeEngine):
    """Protocol double that answers RESET without the engine's state hash."""

    def reset(self, **kwargs) -> dict:
        reply = super().reset(**kwargs)
        reply.pop("hash", None)
        return reply


def retrying_engine_factory(*hashes: str):
    """A factory whose n-th *process* reports ``hashes[n]``.

    The runner retries a divergent start with a fresh engine process (a repeated
    RESET on one process answers ``ready=0`` on the real engine), so each factory
    call must be a new double instance with its own single RESET.
    """
    created: list[FakeEngine] = []

    def factory(binary, data_dir, *, distro="Ubuntu", timeout=120.0):
        engine = FakeEngine(binary, data_dir, distro=distro, timeout=timeout)
        engine.reset_hash = hashes[min(len(created), len(hashes) - 1)]
        created.append(engine)
        return engine

    factory.created = created
    return factory


def run_cli_episode(tmp_path, monkeypatch, capsys, engine, *extra: str):
    """Run the real CLI once against an injected protocol double."""
    monkeypatch.setattr(v400_bench_run, "WslHarnessClient", engine)
    monkeypatch.setattr(v400_bench_run, "collect_provenance", lambda *a, **k: provenance())
    results = tmp_path / "results.jsonl"
    code = v400_bench_run.main([
        "--agent", "random", "--suite", "S1N", "--seed", "1000", "--execute",
        "--human-reference", str(tmp_path / "missing.json"),
        "--results", str(results),
        "--scoreboard", str(tmp_path / "scoreboard.md"),
        *extra,
    ])
    captured = capsys.readouterr()
    return code, results, captured.out, captured.err


def test_expectation_precedence_is_cli_then_environment_then_first_observed():
    environment = {reset_guard_mod.RESET_HASH_ENV: " 7F04C7361749ED9F "}
    assert reset_guard_mod.resolve_expected_reset_hash(DIVERGENT_RESET_HASH, environment) == (
        DIVERGENT_RESET_HASH, reset_guard_mod.SOURCE_CLI)
    assert reset_guard_mod.resolve_expected_reset_hash(None, environment) == (
        EXPECTED_RESET_HASH, reset_guard_mod.SOURCE_ENV)
    assert reset_guard_mod.resolve_expected_reset_hash(None, {}) == (None, None)

    # Precedence 3: the first verified reset of a run becomes the expectation.
    guard = reset_guard_mod.guard_from_config(environ={})
    assert guard.expected is None and guard.source is None
    guard.check({"_verb": "OK", "ready": "1", "hash": EXPECTED_RESET_HASH},
                stage_start="RESET 0 1 0 1000 0")
    assert (guard.expected, guard.source) == (EXPECTED_RESET_HASH,
                                             reset_guard_mod.SOURCE_FIRST)
    with pytest.raises(reset_guard_mod.ResetHashMismatch):
        guard.check({"_verb": "OK", "ready": "1", "hash": DIVERGENT_RESET_HASH},
                    stage_start="RESET 0 1 0 1000 0")

    # A malformed expectation must fail the launch, never be ignored.
    with pytest.raises(ValueError, match=reset_guard_mod.RESET_HASH_ENV):
        reset_guard_mod.resolve_expected_reset_hash(
            None, {reset_guard_mod.RESET_HASH_ENV: "not-a-hash"})


def test_guard_rejects_a_different_hash_with_both_values_and_stage_start():
    guard = reset_guard_mod.guard_from_config(explicit="7F04C7361749ED9F", environ={})
    assert guard.expected == EXPECTED_RESET_HASH and guard.source == "cli"
    assert guard.check({"_verb": "OK", "ready": "1", "hash": EXPECTED_RESET_HASH},
                       stage_start="RESET 0 1 0 1000 0") == EXPECTED_RESET_HASH

    with pytest.raises(reset_guard_mod.ResetHashMismatch) as caught:
        guard.check({"_verb": "OK", "ready": "1", "hash": DIVERGENT_RESET_HASH},
                    stage_start="RESET 0 1 0 1000 0")
    message = str(caught.value)
    assert EXPECTED_RESET_HASH in message and DIVERGENT_RESET_HASH in message
    assert "RESET 0 1 0 1000 0" in message
    record = caught.value.as_record()
    assert record["reset_hash_mismatch"] is True
    assert record["reset_hash_expected"] == EXPECTED_RESET_HASH
    assert record["reset_hash_observed"] == DIVERGENT_RESET_HASH
    assert record["reset_stage_start"] == "RESET 0 1 0 1000 0"


def test_divergent_reset_hash_is_fail_closed_and_writes_no_row(tmp_path, monkeypatch, capsys):
    code, results, out, err = run_cli_episode(
        tmp_path, monkeypatch, capsys, DivergingResetEngine,
        "--expect-reset-hash", EXPECTED_RESET_HASH, "--reset-retries", "0")
    assert code == v400_bench_run.EXIT_RESET_HASH_MISMATCH == 4
    assert not results.exists()  # fail-closed: no bench line at all
    assert EXPECTED_RESET_HASH in err and DIVERGENT_RESET_HASH in err
    payload = json.loads(out)
    assert payload["status"] == "error" and payload["measured"] is False
    assert payload["reset_guard_failed"] is True and payload["reset_hash_mismatch"] is True
    assert payload["reset_hash_expected"] == EXPECTED_RESET_HASH
    assert payload["reset_hash_observed"] == DIVERGENT_RESET_HASH
    assert payload["reset_stage_start"] == "RESET 0 1 0 1000 0"
    assert payload["reset_attempts"] == 1
    assert "795b9441ed0c8056" in payload["error"]
    for invented in ("frames", "deaths", "bombs_used", "lives_end", "score", "end"):
        assert invented not in payload


def test_matching_reset_hash_runs_and_records_the_guard_provenance(tmp_path, monkeypatch,
                                                                   capsys):
    code, results, _out, _err = run_cli_episode(
        tmp_path, monkeypatch, capsys, FakeEngine,
        "--expect-reset-hash", "0" * 16)
    assert code == 0
    stored = json.loads(results.read_text(encoding="utf-8"))
    assert stored["status"] == "ok" and stored["reset_verified"] is True
    assert stored["reset_hash"] == "0" * 16
    assert stored["reset_hash_expected"] == "0" * 16
    assert stored["reset_hash_source"] == "cli"
    assert stored["reset_attempts"] == 1
    validate(stored)


def test_a_mismatch_retried_once_succeeds_and_records_both_hashes(tmp_path, monkeypatch,
                                                                  capsys):
    factory = retrying_engine_factory(DIVERGENT_RESET_HASH, EXPECTED_RESET_HASH)
    code, results, _out, _err = run_cli_episode(
        tmp_path, monkeypatch, capsys, factory,
        "--expect-reset-hash", EXPECTED_RESET_HASH)  # default --reset-retries 1
    assert code == 0
    stored = json.loads(results.read_text(encoding="utf-8"))
    assert stored["status"] == "ok" and stored["reset_verified"] is True
    assert stored["reset_attempts"] == 2
    assert stored["reset_observed_hashes"] == [DIVERGENT_RESET_HASH, EXPECTED_RESET_HASH]
    assert stored["reset_hash"] == EXPECTED_RESET_HASH
    # The retry is a fresh engine process with one RESET each; the divergent
    # process is closed without ever stepping.
    assert len(factory.created) == 2
    assert [len(engine.reset_calls) for engine in factory.created] == [1, 1]
    assert factory.created[0].steps == [] and factory.created[0].closed is True
    assert factory.created[1].steps and factory.created[1].closed is True
    validate(stored)


def test_retries_exhausted_still_fails_closed_without_a_row(tmp_path, monkeypatch, capsys):
    code, results, _out, err = run_cli_episode(
        tmp_path, monkeypatch, capsys, DivergingResetEngine,
        "--expect-reset-hash", EXPECTED_RESET_HASH, "--reset-retries", "2")
    assert code == v400_bench_run.EXIT_RESET_HASH_MISMATCH
    assert not results.exists()
    assert "after 3 attempt(s)" in err
    assert EXPECTED_RESET_HASH in err and DIVERGENT_RESET_HASH in err


def test_reply_without_a_hash_is_fail_closed(tmp_path, monkeypatch, capsys):
    code, results, out, err = run_cli_episode(
        tmp_path, monkeypatch, capsys, HashlessResetEngine,
        "--expect-reset-hash", EXPECTED_RESET_HASH)
    assert code == v400_bench_run.EXIT_RESET_HASH_MISMATCH
    assert not results.exists()
    payload = json.loads(out)
    assert payload["reset_guard_failed"] is True
    assert payload["reset_hash_mismatch"] is False
    assert payload["reset_stage_start"] == "RESET 0 1 0 1000 0"
    assert "no hash to verify" in err


def test_environment_hash_expectation_is_used_when_no_cli_flag(monkeypatch):
    monkeypatch.setenv(reset_guard_mod.RESET_HASH_ENV, "0" * 16)
    row = v400_bench_run.run_episode(
        agent="still", suite_name="S1N", seed=1000, max_frames=6,
        binary="/tmp/th08-modern", data_dir="/tmp/game",
        client_factory=FakeEngine, provenance=provenance(),
    )
    assert row["status"] == "ok" and row["reset_verified"] is True
    assert row["reset_hash_source"] == "env"
    assert row["reset_hash_expected"] == "0" * 16
    validate(row)


def test_explicit_hash_outranks_the_environment(monkeypatch):
    monkeypatch.setenv(reset_guard_mod.RESET_HASH_ENV, DIVERGENT_RESET_HASH)
    row = v400_bench_run.run_episode(
        agent="still", suite_name="S1N", seed=1000, max_frames=6,
        binary="/tmp/th08-modern", data_dir="/tmp/game",
        client_factory=FakeEngine, provenance=provenance(),
        expect_reset_hash="0" * 16,
    )
    assert row["status"] == "ok" and row["reset_hash_source"] == "cli"


def test_env_hash_mismatch_is_fail_closed_without_a_row(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(reset_guard_mod.RESET_HASH_ENV, EXPECTED_RESET_HASH)
    code, results, _out, err = run_cli_episode(
        tmp_path, monkeypatch, capsys, FakeEngine, "--reset-retries", "0")
    assert code == v400_bench_run.EXIT_RESET_HASH_MISMATCH
    assert not results.exists()
    assert "source=env" in err
    assert "0" * 16 in err and EXPECTED_RESET_HASH in err


def test_invalid_expected_hash_is_rejected_before_any_launch(tmp_path):
    with pytest.raises(SystemExit) as caught:
        v400_bench_run.main([
            "--agent", "random", "--suite", "S1N", "--seed", "1000", "--execute",
            "--expect-reset-hash", "not-a-hash",
            "--results", str(tmp_path / "results.jsonl"),
        ])
    assert caught.value.code == 2
    assert not (tmp_path / "results.jsonl").exists()


def test_a_shared_guard_makes_two_episodes_of_one_run_agree():
    guard = reset_guard_mod.guard_from_config(environ={}, retries=0)
    first = v400_bench_run.run_episode(
        agent="still", suite_name="S1N", seed=1000, max_frames=6,
        binary="/tmp/th08-modern", data_dir="/tmp/game",
        client_factory=FakeEngine, provenance=provenance(), reset_guard=guard,
    )
    assert first["status"] == "ok" and first["reset_hash_source"] == "first-observed"

    second = v400_bench_run.run_episode(
        agent="still", suite_name="S1N", seed=1000, max_frames=6,
        binary="/tmp/th08-modern", data_dir="/tmp/game",
        client_factory=DivergingResetEngine, provenance=provenance(), reset_guard=guard,
    )
    assert second["status"] == "error" and second["measured"] is False
    assert second["reset_hash_mismatch"] is True
    assert second["reset_hash_expected"] == "0" * 16
    assert second["reset_hash_observed"] == DIVERGENT_RESET_HASH
    assert "RESET 0 1 0 1000 0" in second["error"]

