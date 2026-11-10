"""Synthetic tests for tools/rl/v400_wp5_determinism.py (no real engine).

These tests pin the *logic* of the WP5/G0.2-item-1 tool: the reproducible mask
sequence, the per-frame comparison, the fresh-data-dir contract, the fail-closed
verdict and the STEP loop's handling of every protocol outcome.  They are not
evidence for real-engine determinism; the real 5,000-frame acceptance run is
recorded in the experiment card and its JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.rl.v400_wp5_determinism import (  # noqa: E402
    MASK_SEED,
    MASK_WHITELIST,
    REQUIRED_FRAMES,
    RESET_COMMAND,
    DeterminismError,
    build_report,
    child_env,
    collect_frames,
    compare_hashes,
    create_data_dir,
    engine_argv,
    evaluate,
    main,
    mask_sequence,
    mask_sequence_document,
    mask_sequence_sha256,
    parse_reply,
    run_failure_reasons,
    validate_hash,
)

#: Frozen first masks and full-sequence digest: changing the generator must be a
#: deliberate act, because the report promises this sequence is reproducible.
EXPECTED_HEAD = [0xB1, 0xF6, 0x40, 0x73, 0xA6, 0x20, 0xB4, 0x94, 0xB4, 0xC7, 0x50, 0x46]
EXPECTED_SHA256 = "4173ff413d4bc18e27b771e86de16a441f765ac92b273c68ab7e32f5ee585e1f"


class FakeEngine:
    """Replays canned protocol replies; never starts a process."""

    def __init__(self, hashes, *, role="shadow", reset_hash="0123456789abcdef",
                 stop_at=None, bad_frame_at=None, bad_hash_at=None):
        self.hashes = list(hashes)
        self.role = role
        self.reset_hash = reset_hash
        self.stop_at = stop_at
        self.bad_frame_at = bad_frame_at
        self.bad_hash_at = bad_hash_at
        self.step_index = 0
        self.commands: list[str] = []

    def ask(self, command, timeout=None):
        self.commands.append(command)
        if command == "HELLO":
            return {"_verb": "OK", "proto": "1", "harness": "1", "role": self.role, "frames": "0"}
        if command == "DEBUG":
            return {"_verb": "OK", "curState": "2", "subthread": "0", "threads": "1",
                    "frames": str(self.step_index)}
        if command.startswith("RESET"):
            return {"_verb": "OK", "reset": "1", "ready": "1", "frames": "0",
                    "hash": self.reset_hash}
        if command.startswith("STEP"):
            index = self.step_index + 1
            if self.stop_at == index:
                return {"_verb": "OK", "advanced": "0", "frame": str(self.step_index),
                        "exit": "1", "game_over": "1"}
            digest = self.hashes[self.step_index]
            self.step_index += 1
            if self.bad_frame_at == index:
                return {"_verb": "OK", "advanced": "1", "frame": str(index + 1), "hash": digest}
            if self.bad_hash_at == index:
                return {"_verb": "OK", "advanced": "1", "frame": str(index), "hash": digest[:15]}
            return {"_verb": "OK", "advanced": "1", "frame": str(index), "hash": digest}
        if command == "HASH":
            return {"_verb": "OK", "hash": self.hashes[self.step_index - 1],
                    "frames": str(self.step_index)}
        if command == "QUIT":
            return {"_verb": "OK"}
        raise AssertionError(f"unexpected command {command!r}")


def make_run(label, hashes, *, role="shadow", exit_code=0, frame0="d97b43478cc24bae",
             curState="2", threads="1", subthread="0", final_hash_crosscheck=True):
    return {
        "label": label,
        "role": role,
        "hashes": list(hashes),
        "frames_stepped": len(hashes),
        "frame0_hash": frame0,
        "reset_ready": True,
        "reset": {"ready": "1"},
        "debug_after_reset": {"curState": curState, "threads": threads, "subthread": subthread},
        "debug_after": {"curState": curState, "frames": str(len(hashes))},
        "stop": None,
        "exit_code": exit_code,
        "final_hash_crosscheck": final_hash_crosscheck,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Mask sequence
# ---------------------------------------------------------------------------
def test_mask_sequence_is_reproducible_and_whitelisted():
    masks = mask_sequence(MASK_SEED, REQUIRED_FRAMES)
    assert len(masks) == REQUIRED_FRAMES == 5000
    assert all(0 <= mask <= 0xFFFF for mask in masks)
    assert all(mask & ~MASK_WHITELIST == 0 for mask in masks)
    assert masks[:12] == EXPECTED_HEAD
    assert mask_sequence(MASK_SEED, REQUIRED_FRAMES) == masks
    assert len(set(masks)) == 128
    assert mask_sequence_sha256(masks) == EXPECTED_SHA256


def test_mask_sequence_depends_on_seed():
    assert mask_sequence(400, 32) != mask_sequence(401, 32)
    assert mask_sequence(400, 32) == mask_sequence(400, 32)


def test_mask_sequence_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        mask_sequence(400, 0)
    with pytest.raises(ValueError):
        mask_sequence(-1, 8)


def test_mask_sequence_document_is_self_describing():
    masks = mask_sequence(400, 64)
    document = mask_sequence_document(masks)
    assert document["seed"] == 400
    assert document["length"] == 64
    assert document["whitelist"] == "0x00F7"
    assert document["masks"] == masks
    assert document["head"] == masks[:16]
    assert document["tail"] == masks[-8:]
    assert document["all_masks_whitelisted"] is True
    assert document["sha256"] == mask_sequence_sha256(masks)
    assert "SplitMix" in document["algorithm"] or "0x9E3779B97F4A7C15" in document["algorithm"]


# ---------------------------------------------------------------------------
# Comparison / verdict
# ---------------------------------------------------------------------------
def test_compare_hashes_all_equal():
    hashes = [f"{index:016x}" for index in range(10)]
    result = compare_hashes(hashes, list(hashes))
    assert result["frames_compared"] == 10
    assert result["equal_frames"] == 10
    assert result["different_frames"] == 0
    assert result["first_differing_frame"] is None
    assert result["all_equal"] is True


def test_compare_hashes_reports_first_divergence():
    a = [f"{index:016x}" for index in range(10)]
    b = list(a)
    b[7] = "ffffffffffffffff"
    result = compare_hashes(a, b)
    assert result["equal_frames"] == 9
    assert result["different_frames"] == 1
    assert result["first_differing_frame"] == 8  # 1-based frame number
    assert result["differences"] == [{"frame": 8, "a": a[7], "b": b[7]}]
    assert result["all_equal"] is False


def test_compare_hashes_length_mismatch_is_not_equal():
    a = [f"{index:016x}" for index in range(10)]
    result = compare_hashes(a, a[:9])
    assert result["frames_compared"] == 9
    assert result["different_frames"] == 0
    assert result["lengths"] == [10, 9]
    assert result["all_equal"] is False


def test_evaluate_passes_only_when_everything_matches():
    hashes = [f"{index:016x}" for index in range(5)]
    runs = [make_run("run-a", hashes), make_run("run-b", hashes)]
    compare = compare_hashes(hashes, hashes)
    compare["frame0_equal"] = True
    verdict = evaluate(runs, compare, 5)
    assert verdict["pass"] is True
    assert verdict["reasons"] == []
    assert verdict["warnings"] == []


def test_evaluate_flags_divergence_exit_code_and_role():
    a = [f"{index:016x}" for index in range(5)]
    b = list(a)
    b[2] = "ffffffffffffffff"
    runs = [make_run("run-a", a, exit_code=1), make_run("run-b", b, role="env")]
    compare = compare_hashes(a, b)
    compare["frame0_equal"] = False
    verdict = evaluate(runs, compare, 5)
    assert verdict["pass"] is False
    joined = " | ".join(verdict["reasons"])
    assert "run-a: engine exit_code=1" in joined
    assert "run-b: HELLO role='env'" in joined
    assert "first differing frame=3" in joined
    assert "frame-0 (RESET) hashes differ" in joined


def test_evaluate_flags_short_run_and_missing_crosscheck():
    hashes = [f"{index:016x}" for index in range(4)]
    runs = [make_run("run-a", hashes, final_hash_crosscheck=False),
            make_run("run-b", hashes[:3])]
    compare = compare_hashes(hashes, hashes[:3])
    compare["frame0_equal"] = True
    verdict = evaluate(runs, compare, 5)
    assert verdict["pass"] is False
    joined = " | ".join(verdict["reasons"])
    assert "final HASH disagrees" in joined
    assert "run-b: frames_stepped=3" in joined
    assert "frames compared=3" in joined


def test_run_failure_reasons_covers_reset_state():
    run = make_run("run-a", ["0" * 16], curState="6")
    reasons = run_failure_reasons(run, 1)
    assert any("curState='6'" in reason for reason in reasons)


def test_validate_hash_accepts_engine_form_and_rejects_junk():
    assert validate_hash("0123456789abcdef", where="t") == "0123456789abcdef"
    for bad in ("0123456789abcde", "0123456789ABCDEF", "", None, 7):
        with pytest.raises(DeterminismError):
            validate_hash(bad, where="t")


def test_parse_reply():
    assert parse_reply("OK advanced=1 frame=2 hash=deadbeefdeadbeef") == {
        "_verb": "OK", "advanced": "1", "frame": "2", "hash": "deadbeefdeadbeef"}
    with pytest.raises(DeterminismError):
        parse_reply("ERR step-usage")


# ---------------------------------------------------------------------------
# Data directory contract
# ---------------------------------------------------------------------------
def make_template(path: Path) -> Path:
    path.mkdir()
    for name in ("th08.dat", "thbgm.dat", "th08.cfg", "score.dat"):
        (path / name).write_bytes(f"template:{name}".encode())
    return path


def test_create_data_dir_links_archives_and_copies_writable_files(tmp_path):
    template = make_template(tmp_path / "template")
    target = tmp_path / "fresh" / "data-a"
    links: list[tuple[str, str]] = []

    def fake_link(source, destination):
        links.append((source, destination))
        Path(destination).write_text(f"linked:{source}")

    record = create_data_dir(template, target, link=fake_link)
    assert target.is_dir()
    assert [Path(destination).name for _, destination in links] == ["th08.dat", "thbgm.dat"]
    assert [Path(source).name for source, _ in links] == ["th08.dat", "thbgm.dat"]
    assert (target / "th08.cfg").read_bytes() == b"template:th08.cfg"
    assert (target / "score.dat").read_bytes() == b"template:score.dat"
    assert record["created"] is True
    assert set(record["copies"]) == {"th08.cfg", "score.dat"}
    # Copies are independent of the template.
    (target / "score.dat").write_bytes(b"changed")
    assert (template / "score.dat").read_bytes() == b"template:score.dat"


def test_create_data_dir_refuses_existing_path(tmp_path):
    template = make_template(tmp_path / "template")
    target = tmp_path / "already-there"
    target.mkdir()
    with pytest.raises(DeterminismError):
        create_data_dir(template, target, link=lambda *_: None)


def test_create_data_dir_requires_every_template_file(tmp_path):
    template = tmp_path / "template"
    template.mkdir()
    (template / "th08.dat").write_bytes(b"x")
    with pytest.raises(DeterminismError):
        create_data_dir(template, tmp_path / "data", link=lambda *_: None)


# ---------------------------------------------------------------------------
# Command line / environment
# ---------------------------------------------------------------------------
def test_engine_argv_is_the_proven_shadow_invocation(tmp_path):
    argv = engine_argv(tmp_path / "th08-modern", tmp_path / "data")
    assert argv[1:] == ["--data-dir", str(tmp_path / "data"),
                        "--rl-control", "0,1", "--rl-role", "shadow"]


def test_child_env_removes_display_and_pins_dummy_audio():
    env = child_env({"DISPLAY": ":0", "PATH": "/usr/bin", "KEEP": "1"})
    assert "DISPLAY" not in env
    assert env["SDL_AUDIODRIVER"] == "dummy"
    assert env["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert env["LP_NUM_THREADS"] == "1"
    assert env["KEEP"] == "1"


def test_reset_command_matches_the_documented_stage_start():
    assert RESET_COMMAND == "RESET 0 1 0 1234 1"


# ---------------------------------------------------------------------------
# STEP loop
# ---------------------------------------------------------------------------
def test_collect_frames_records_every_hash():
    hashes = [f"{index:016x}" for index in range(6)]
    engine = FakeEngine(hashes)
    result = collect_frames(engine, mask_sequence(400, 6))
    assert result["role"] == "shadow"
    assert result["reset_ready"] is True
    assert result["frame0_hash"] == "0123456789abcdef"
    assert result["hashes"] == hashes
    assert result["frames_stepped"] == 6
    assert result["stop"] is None
    assert result["final_hash_crosscheck"] is True
    assert result["debug_after_reset"]["threads"] == "1"
    assert result["debug_after_reset"]["subthread"] == "0"
    assert result["debug_after"]["frames"] == "6"
    # One STEP per frame, carrying that frame's mask.
    steps = [command for command in engine.commands if command.startswith("STEP")]
    assert steps[0] == f"STEP {mask_sequence(400, 6)[0]} 1"
    assert len(steps) == 6


def test_collect_frames_stops_at_game_over_without_inventing_hashes():
    hashes = [f"{index:016x}" for index in range(6)]
    engine = FakeEngine(hashes, stop_at=3)
    result = collect_frames(engine, mask_sequence(400, 6))
    assert result["hashes"] == hashes[:2]
    assert result["frames_stepped"] == 2
    assert result["stop"]["frame"] == 3
    assert result["stop"]["reply"]["exit"] == "1"
    assert "exit=1" in result["stop"]["reason"]
    assert result["final_hash_crosscheck"] is None


def test_collect_frames_treats_a_wrong_frame_number_as_a_stop():
    hashes = [f"{index:016x}" for index in range(5)]
    engine = FakeEngine(hashes, bad_frame_at=3)
    result = collect_frames(engine, mask_sequence(400, 5))
    assert result["frames_stepped"] == 2
    assert result["stop"]["frame"] == 3


def test_collect_frames_rejects_a_malformed_hash():
    hashes = [f"{index:016x}" for index in range(5)]
    engine = FakeEngine(hashes, bad_hash_at=2)
    with pytest.raises(DeterminismError):
        collect_frames(engine, mask_sequence(400, 5))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def test_build_report_shape_and_pass_status(tmp_path):
    hashes = [f"{index:016x}" for index in range(3)]
    runs = [make_run("run-a", hashes), make_run("run-b", hashes)]
    report = build_report(
        runs=runs,
        masks=mask_sequence(400, 3),
        binary_info={"path": "x", "sha256": "0" * 64, "size": 1},
        harness_info={"repo": "r", "head": "h", "dirty": True},
        data_info={"template": "t", "dirs": ["a", "b"], "created_fresh": True},
        environment={"display": "unset"},
        required_frames=3,
        wall_s=1.5,
        invocation={"argv": ["python"], "cwd": "."},
    )
    assert report["schema"] == "v400-wp5-determinism-5000/1"
    assert report["status"] == "pass"
    assert report["required_frames"] == 3
    assert report["frames_per_run"] == [3, 3]
    assert report["compare"]["equal_frames"] == 3
    assert report["compare"]["different_frames"] == 0
    assert report["compare"]["frame0_equal"] is True
    assert report["debug"]["run-a"]["threads"] == "1"
    assert report["debug"]["run-a"]["subthread"] == "0"
    assert report["verdict"]["pass"] is True
    assert report["mask"]["length"] == 3
    assert len(report["limits"]) >= 2
    json.dumps(report)  # must be serialisable for the evidence file


def test_build_report_fails_on_divergence(tmp_path):
    a = [f"{index:016x}" for index in range(3)]
    b = list(a)
    b[1] = "ffffffffffffffff"
    report = build_report(
        runs=[make_run("run-a", a), make_run("run-b", b)],
        masks=mask_sequence(400, 3),
        binary_info={}, harness_info={}, data_info={}, environment={},
        required_frames=3,
    )
    assert report["status"] == "failed"
    assert report["compare"]["different_frames"] == 1
    assert report["compare"]["first_differing_frame"] == 2
    assert report["verdict"]["pass"] is False
    assert report["invocation"] is None


# ---------------------------------------------------------------------------
# CLI guard rails (no engine is started)
# ---------------------------------------------------------------------------
def test_main_refuses_to_overwrite_an_existing_report(tmp_path):
    existing = tmp_path / "out.json"
    existing.write_text("{}")
    code = main(["--binary", str(tmp_path / "bin"), "--data-template", str(tmp_path / "data"),
                 "--run-root", str(tmp_path / "root"), "--out", str(existing)])
    assert code == 2


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows guard is explicit")
def test_main_blocks_on_windows_without_touching_the_output(tmp_path):
    out = tmp_path / "out.json"
    code = main(["--binary", str(tmp_path / "bin"), "--data-template", str(tmp_path / "data"),
                 "--run-root", str(tmp_path / "root"), "--out", str(out)])
    assert code == 3
    assert not out.exists()
    assert not (tmp_path / "root").exists()
