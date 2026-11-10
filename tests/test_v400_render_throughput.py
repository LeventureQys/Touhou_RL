"""Accounting tests for the WP2 render+readback probe.

These tests only validate the tool's *arithmetic and fail-closed logic*; a
synthetic client is never evidence of engine throughput (that comes from the
real harness runs recorded in EXP-P0-052).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "rl" / "v400_render_throughput.py"


def _load():
    spec = importlib.util.spec_from_file_location("v400_render_throughput", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load()


# ------------------------------------------------------------------ environment
def test_software_backend_forces_all_knobs():
    env = probe.build_child_env("software", display=":98", lp_threads=1,
                                base_env={"PATH": "/usr/bin"})
    assert env["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert env["GALLIUM_DRIVER"] == "llvmpipe"
    assert env["LP_NUM_THREADS"] == "1"
    assert env["DISPLAY"] == ":98"
    assert env["SDL_AUDIODRIVER"] == "dummy"


def test_hardware_backend_strips_ambient_software_knobs():
    env = probe.build_child_env(
        "hardware", display=":0", lp_threads=1,
        base_env={"LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe",
                  "LP_NUM_THREADS": "1", "DISPLAY": ":98", "PATH": "/usr/bin"})
    assert "LIBGL_ALWAYS_SOFTWARE" not in env
    assert "GALLIUM_DRIVER" not in env
    assert "LP_NUM_THREADS" not in env
    assert env["DISPLAY"] == ":0"


def test_hardware_backend_refuses_a_surviving_software_knob():
    """A forced software run must never be reported as the hardware backend."""
    assert probe.fingerprint_conflict("hardware", "LIBGL_ALWAYS_SOFTWARE", "1")
    assert probe.fingerprint_conflict("hardware", "GALLIUM_DRIVER", "llvmpipe")
    assert probe.fingerprint_conflict("hardware", "MESA_LOADER_DRIVER_OVERRIDE", "swrast")
    assert not probe.fingerprint_conflict("hardware", "LP_NUM_THREADS", "1")
    assert not probe.fingerprint_conflict("hardware", "LIBGL_ALWAYS_SOFTWARE", None)
    assert not probe.fingerprint_conflict("software", "LIBGL_ALWAYS_SOFTWARE", "1")


def test_unknown_backend_and_bad_thread_count_rejected():
    with pytest.raises(ValueError):
        probe.build_child_env("gpu", display=":0", lp_threads=1)
    with pytest.raises(ValueError):
        probe.build_child_env("software", display=":98", lp_threads=0)


# ------------------------------------------------------- experiment switches
def test_child_env_overlay_carries_experiment_switches():
    overlay = probe.parse_child_env(["TH08_ASYNC_READBACK=1",
                                     "TH08_PBO_MODE=async"])
    assert overlay == {"TH08_ASYNC_READBACK": "1", "TH08_PBO_MODE": "async"}


def test_child_env_overlay_is_empty_by_default():
    assert probe.parse_child_env([]) == {}
    assert probe.parse_child_env(None) == {}


@pytest.mark.parametrize("bad", ["TH08_ASYNC_READBACK", "=1", ""])
def test_child_env_overlay_rejects_malformed_pairs(bad):
    with pytest.raises(probe.ProbeUnavailable):
        probe.parse_child_env([bad])


def test_child_env_overlay_cannot_fake_the_gl_backend():
    """The backend is decided by --backend and proven from glGetString."""
    for knob in probe.GL_KNOBS:
        with pytest.raises(probe.ProbeUnavailable):
            probe.parse_child_env([f"{knob}=1"])


def test_child_env_overlay_reaches_the_child_environment():
    env = probe.build_child_env("software", display=":98", lp_threads=1,
                                base_env={"PATH": "/usr/bin"})
    env.update(probe.parse_child_env(["TH08_ASYNC_READBACK=1"]))
    assert env["TH08_ASYNC_READBACK"] == "1"
    assert env["LIBGL_ALWAYS_SOFTWARE"] == "1"  # backend untouched


def test_cli_accepts_repeated_child_env_and_records_it():
    parser = probe.build_parser()
    args = parser.parse_args(["--binary", "b", "--data-template", "d", "--run-root", "r",
                              "--backend", "software", "--out", "o",
                              "--child-env", "TH08_ASYNC_READBACK=1",
                              "--child-env", "TH08_PBO_MODE=sync"])
    assert probe.parse_child_env(args.child_env) == {"TH08_ASYNC_READBACK": "1",
                                                     "TH08_PBO_MODE": "sync"}


# ------------------------------------------------------------------ host load
def test_read_loadavg_parses_the_real_proc_line(tmp_path):
    path = tmp_path / "loadavg"
    path.write_text("0.05 0.06 0.02 1/311 20270\n", encoding="utf-8")
    sample = probe.read_loadavg(str(path))
    assert sample["available"] is True
    assert sample["load1"] == pytest.approx(0.05)
    assert sample["load5"] == pytest.approx(0.06)
    assert sample["load15"] == pytest.approx(0.02)
    assert sample["runnable"] == "1/311"
    assert sample["raw"] == "0.05 0.06 0.02 1/311 20270"


def test_read_loadavg_reports_a_missing_file_instead_of_guessing(tmp_path):
    sample = probe.read_loadavg(str(tmp_path / "absent"))
    assert sample["available"] is False
    assert sample["raw"] is None
    assert "error" in sample


def test_read_loadavg_rejects_a_non_loadavg_line(tmp_path):
    path = tmp_path / "loadavg"
    path.write_text("not a load average\n", encoding="utf-8")
    sample = probe.read_loadavg(str(path))
    assert sample["available"] is False
    assert sample["raw"] == "not a load average"


# ------------------------------------------------------------------ throughput
def test_compute_throughput_is_frames_over_seconds():
    assert probe.compute_throughput(1000, 2.0) == 500.0


@pytest.mark.parametrize("frames,seconds", [(0, 0.0), (10, -1.0)])
def test_compute_throughput_rejects_nonpositive_interval(frames, seconds):
    with pytest.raises(ValueError):
        probe.compute_throughput(frames, seconds)


def test_aggregate_uses_one_shared_wall_window_not_the_sum_of_runs():
    instances = [
        {"frames": 640, "step_seconds": 9.0, "frames_per_second": 71.1},
        {"frames": 600, "step_seconds": 9.5, "frames_per_second": 63.2},
    ]
    out = probe.aggregate_throughput(instances, window_seconds=10.0)
    assert out["frames"] == 1240
    assert out["frames_per_second"] == pytest.approx(124.0)
    assert out["instance_fps_sum"] == pytest.approx(134.3)
    assert out["parallel_efficiency"] == pytest.approx(1.85)


def test_aggregate_requires_instances_and_positive_window():
    with pytest.raises(probe.ProbeUnavailable):
        probe.aggregate_throughput([], window_seconds=1.0)
    with pytest.raises(ValueError):
        probe.aggregate_throughput([{"frames": 1, "step_seconds": 1.0,
                                     "frames_per_second": 1.0}], window_seconds=0.0)


# ---------------------------------------------------------------------- gate
def test_gate_thresholds_follow_the_wp2_decision_table():
    assert probe.gate_verdict(2500.0)["decision"] == "continue_as_planned"
    assert probe.gate_verdict(2500.0)["passed"] is True
    assert probe.gate_verdict(1200.0)["decision"] == "continue_but_rebudget_by_measurement"
    assert probe.gate_verdict(1200.0)["passed"] is False
    assert probe.gate_verdict(499.0)["decision"] == "stop_and_ask_owner"
    assert probe.gate_verdict(2000.0)["passed"] is True


def test_gate_is_not_applicable_without_per_frame_readback():
    verdict = probe.gate_verdict(5000.0, readback=False)
    assert verdict["gate_applicable"] is False
    assert verdict["passed"] is False
    assert verdict["decision"] == "diagnostic_only_readback_disabled"


# ------------------------------------------------------------------ GL proof
FINGERPRINT = """pid=4242
GL_VENDOR=Microsoft Corporation
GL_RENDERER=D3D12 (NVIDIA GeForce RTX 4070 SUPER)
GL_VERSION=4.2 (Compatibility Profile) Mesa 23.2.1-1ubuntu3.1~22.04.4
GL_SHADING_LANGUAGE_VERSION=4.20
"""


def test_parse_gl_fingerprint_reads_the_engine_context():
    parsed = probe.parse_gl_fingerprint(FINGERPRINT, pid=4242)
    assert parsed["GL_RENDERER"].startswith("D3D12")
    assert parsed["GL_VENDOR"] == "Microsoft Corporation"


def test_parse_gl_fingerprint_picks_the_matching_pid_block():
    text = FINGERPRINT + "pid=99\nGL_RENDERER=llvmpipe (LLVM 15.0.7, 256 bits)\n"
    assert probe.parse_gl_fingerprint(text, pid=99)["GL_RENDERER"].startswith("llvmpipe")
    assert probe.parse_gl_fingerprint(text, pid=4242)["GL_RENDERER"].startswith("D3D12")
    assert probe.parse_gl_fingerprint("", pid=1) == {}


def test_renderer_must_match_the_declared_backend():
    assert probe.renderer_matches("D3D12", probe.parse_gl_fingerprint(FINGERPRINT))
    assert not probe.renderer_matches("llvmpipe", probe.parse_gl_fingerprint(FINGERPRINT))
    assert not probe.renderer_matches("D3D12", {})  # unproven is never a match
    assert probe.renderer_matches(None, {})


def test_mapped_driver_name_is_not_backend_proof():
    """This Mesa build hardlinks every gallium driver: the path lies, GL doesn't."""
    maps = "7f00-7f01 r-xp /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so"
    seen = probe.classify_gl(maps)
    assert seen["swrast"] is True and seen["d3d12"] is False
    # the same maps text must be accepted by the llvmpipe backend and rejected
    # for hardware by the *fallback* check only -- which is why the tool gates on
    # the in-process renderer string instead.
    assert probe.backend_matches("software", seen) is True
    assert probe.backend_matches("hardware", seen) is False


def test_parse_glxinfo_extracts_renderer_and_acceleration():
    text = ("Extended renderer info (GLX_MESA_query_renderer):\n"
            "    Accelerated: yes\n"
            "OpenGL renderer string: D3D12 (NVIDIA GeForce RTX 4070 SUPER)\n")
    parsed = probe.parse_glxinfo(text)
    assert parsed["accelerated"] is True
    assert parsed["d3d12"] is True
    assert parsed["renderer"].startswith("D3D12")


def test_parse_glxinfo_software():
    parsed = probe.parse_glxinfo("    Accelerated: no\n"
                                 "OpenGL renderer string: llvmpipe (LLVM 15.0.7, 256 bits)\n")
    assert parsed["accelerated"] is False
    assert parsed["llvmpipe"] is True


# ----------------------------------------------------------------- protocol
def test_parse_reply_keeps_keys_and_verb():
    parsed = probe.parse_reply("OK readback=1 frames=120 micros=430014")
    assert parsed["_verb"] == "OK"
    assert parsed["frames"] == "120"
    assert parsed["micros"] == "430014"


# ------------------------------------------------------------------ data dirs
def _template(root: Path) -> Path:
    template = root / "template"
    template.mkdir()
    for name in ("th08.dat", "thbgm.dat", "th08.cfg", "score.dat"):
        (template / name).write_bytes(b"x")
    return template


def test_prepare_data_dirs_isolates_writable_files(tmp_path):
    template = _template(tmp_path)
    result = probe.prepare_data_dirs(template, tmp_path / "run", 2)
    link_actions = {a["action"] for a in result["actions"]
                    if a["file"] in probe.DATA_FILES_LINK}
    assert link_actions <= {"symlink", "hardlink", "copy"}
    for index, path in enumerate(result["dirs"]):
        directory = Path(path)
        assert (directory / "th08.dat").is_file()
        assert not (directory / "score.dat").is_symlink()
        assert (directory / "score.dat").is_file()
        assert directory.name == f"i-{index}"


def test_prepare_data_dirs_refuses_to_reuse_an_existing_instance(tmp_path):
    template = _template(tmp_path)
    probe.prepare_data_dirs(template, tmp_path / "run", 1)
    with pytest.raises(probe.ProbeUnavailable, match="refusing to reuse"):
        probe.prepare_data_dirs(template, tmp_path / "run", 1)


def test_prepare_data_dirs_requires_a_th08_dat_template(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(probe.ProbeUnavailable, match="no th08.dat"):
        probe.prepare_data_dirs(empty, tmp_path / "run", 1)


# ---------------------------------------------------------------------- CLI
def test_cli_rejects_nonpositive_instances_and_seconds():
    parser = probe.build_parser()
    with pytest.raises(SystemExit):
        probe.main(["--binary", __file__, "--data-template", ".", "--run-root", "x",
                    "--backend", "software", "--instances", "0", "--out", "o.json"])
    with pytest.raises(SystemExit):
        probe.main(["--binary", __file__, "--data-template", ".", "--run-root", "x",
                    "--backend", "software", "--seconds", "0", "--out", "o.json"])


def test_cli_requires_a_known_backend():
    parser = probe.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--binary", "b", "--data-template", "d", "--run-root", "r",
                           "--backend", "vulkan", "--out", "o"])
