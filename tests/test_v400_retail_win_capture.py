"""Unit tests for the retail-image Windows capture probe (EXP-P0-060).

None of these tests start ``th08.exe``: they cover the parts of
``tools/rl/v400_retail_win_capture.py`` that decide *whether* a run may happen
(identity, single instance, out-dir reuse), what gets copied into the private
data dir, and the pure reducers that judge a sample sequence.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.rl import v400_retail_win_capture as mod


# ---------------------------------------------------------------------------
# retail identity
# ---------------------------------------------------------------------------
def test_retail_identity_matches_the_documented_image():
    identity = mod.retail_identity()
    if not identity["present"]:
        pytest.skip("retail image not present in this checkout")
    assert identity["size"] == mod.RETAIL_SIZE
    assert identity["sha256"] == mod.RETAIL_SHA256
    assert identity["verified"] is True


def test_retail_image_is_a_fixed_base_32bit_gui_executable():
    if not mod.RETAIL_EXE.is_file():
        pytest.skip("retail image not present in this checkout")
    summary = mod.pe_summary(mod.RETAIL_EXE)
    assert summary["machineIsI386"] is True
    assert summary["subsystemIsGui"] is True
    assert summary["imageBase"] == "0x400000"
    # No DYNAMIC_BASE: linker-map VAs are runtime VAs, which is what the
    # address table and touhou_rl.spec both assume.
    assert summary["dynamicBase"] is False


def test_identity_rejects_a_non_retail_image(tmp_path):
    fake = tmp_path / "th08.exe"
    fake.write_bytes(b"MZ" + b"\x00" * 200)
    identity = mod.retail_identity(fake)
    assert identity["verified"] is False
    assert "refusing to run" in identity["reason"]


# ---------------------------------------------------------------------------
# copy plan
# ---------------------------------------------------------------------------
def test_plan_assets_copies_the_retail_exe_and_the_four_assets():
    plan = mod.plan_assets()
    names = [entry["name"] for entry in plan]
    assert names == ["th08.exe", "th08.dat", "thbgm.dat", "th08.cfg", "score.dat"]
    assert plan[0]["source"] == str(mod.RETAIL_EXE)
    # The engine rewrites these two; they must never be hard-linked into a
    # directory the game can write into.
    assert {entry["name"] for entry in plan if entry["mutable"]} == {"th08.cfg", "score.dat"}


def _fake_template(tmp_path: Path) -> Path:
    """A stand-in asset dir with the same file names and a valid 60-byte cfg."""
    template = tmp_path / "template"
    template.mkdir()
    for name in ("th08.dat", "thbgm.dat", "score.dat"):
        (template / name).write_bytes(b"asset:" + name.encode())
    # prepare_config only accepts a 60-byte GameConfiguration.
    (template / "th08.cfg").write_bytes(bytes(60))
    return template


def test_assemble_run_dir_copies_instead_of_linking(tmp_path, monkeypatch):
    template = _fake_template(tmp_path)
    fake_exe = tmp_path / "resources-th08.exe"
    fake_exe.write_bytes(b"MZ fake retail image")
    monkeypatch.setattr(mod, "RETAIL_EXE", fake_exe)

    run_dir = tmp_path / "run"
    report = mod.assemble_run_dir(run_dir, template, force=True)

    assert (run_dir / "th08.exe").read_bytes() == fake_exe.read_bytes()
    assert (run_dir / "replay").is_dir()
    actions = {entry["name"]: entry["action"] for entry in report["files"]}
    assert set(actions.values()) == {"copied"}
    # A copy, not a hard link: the two paths must not share an inode.
    assert (run_dir / "th08.dat").stat().st_ino != (template / "th08.dat").stat().st_ino
    # The config is normalised for a foreground SendInput session and windowed mode.
    raw = (run_dir / "th08.cfg").read_bytes()
    assert len(raw) == 60
    assert raw[0x22] == 1  # windowed
    assert raw[0x38] & 0x01 == 0  # dontUseDirectInput cleared


def test_assemble_run_dir_can_skip_score_dat(tmp_path, monkeypatch):
    template = _fake_template(tmp_path)
    fake_exe = tmp_path / "resources-th08.exe"
    fake_exe.write_bytes(b"MZ fake retail image")
    monkeypatch.setattr(mod, "RETAIL_EXE", fake_exe)

    run_dir = tmp_path / "run"
    report = mod.assemble_run_dir(run_dir, template, force=True, skip=("score.dat",))
    assert not (run_dir / "score.dat").exists()
    actions = {entry["name"]: entry["action"] for entry in report["files"]}
    assert actions["score.dat"] == "skipped"


# ---------------------------------------------------------------------------
# pure reducers
# ---------------------------------------------------------------------------
def test_monotonic_problems_accepts_a_healthy_sequence():
    assert mod.monotonic_problems([1, 2, 3, 5, 5, 6], [0, 0, 10, 10], [0, 0, 0]) == []


def test_monotonic_problems_reports_each_regression():
    problems = mod.monotonic_problems([5, 4], [100, 90], [1, 0])
    assert len(problems) == 3
    assert "gameplayFrame decreased" in problems[0]
    assert "score decreased" in problems[1]
    assert "deaths decreased" in problems[2]


def test_consecutive_run_finds_the_longest_plus_one_chain():
    assert mod.consecutive_run([9, 1, 2, 3, 7, 8]) == (1, 3)
    assert mod.consecutive_run([]) == (0, 0)
    assert mod.consecutive_run([4]) == (0, 1)
    # Duplicated frame numbers break a chain; the capture phase dedupes first.
    assert mod.consecutive_run([1, 2, 2, 3, 4]) == (2, 3)


def test_frame_deltas():
    assert mod.frame_deltas([10, 11, 13]) == [1, 2]
    assert mod.frame_deltas([10]) == []


# ---------------------------------------------------------------------------
# gatekeepers
# ---------------------------------------------------------------------------
def _minimal_report() -> dict:
    return {"events": [], "result": {}, "status": "running"}


def test_check_single_instance_refuses_when_a_game_is_running(monkeypatch):
    from touhou_rl import winapi

    monkeypatch.setattr(winapi, "list_process_ids", lambda name="th08.exe": [4242])
    report = _minimal_report()
    assert mod._check_single_instance(report) == mod.EXIT_INSTANCE
    assert report["status"] == "blocked"
    assert "4242" in report["result"]["reason"]


def test_check_single_instance_allows_a_clean_slate(monkeypatch):
    from touhou_rl import winapi

    monkeypatch.setattr(winapi, "list_process_ids", lambda name="th08.exe": [])
    report = _minimal_report()
    assert mod._check_single_instance(report) is None
    assert report["otherInstances"] == []


def test_main_refuses_to_reuse_a_non_empty_out_dir(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "already-here.png").write_bytes(b"x")
    code = mod.main(["--phase", "launch", "--out", str(out)])
    assert code == mod.EXIT_USAGE
    assert not (out / "report.json").exists()


def test_main_blocks_on_a_missing_replay_before_any_launch(tmp_path):
    out = tmp_path / "out"
    addresses = tmp_path / "addr.json"
    addresses.write_text("{}", encoding="utf-8")
    code = mod.main(
        [
            "--phase",
            "launch",
            "--out",
            str(out),
            "--addresses",
            str(addresses),
            "--replay",
            str(tmp_path / "missing.rpy"),
        ]
    )
    assert code == mod.EXIT_USAGE
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert "missing replay" in report["result"]["reason"]
    assert report["retailImage"]["verified"] is True


# ---------------------------------------------------------------------------
# retail address table
# ---------------------------------------------------------------------------
def test_retail_address_table_is_a_complete_retail_map():
    table_path = Path(mod.DEFAULT_ADDRESSES)
    if not table_path.is_file():
        pytest.skip("retail address table not present")
    payload = json.loads(table_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "th08-map-addresses/1"
    assert int(payload["imageBase"], 16) == 0x00400000
    symbols = {name: int(entry["address"], 16) for name, entry in payload["symbols"].items()}
    for required in (
        "g_Supervisor",
        "g_GameManager",
        "g_Player",
        "g_TitleScreen",
        "g_Gui",
        "g_CurFrameInput",
        "g_LastFrameInput",
        "g_GuiMessageInputCurrent",
        "g_GuiMessageInputPrevious",
    ):
        assert required in symbols, required
    # Every address has to be inside the retail image's mapped .data/.data1 range,
    # i.e. above .text and below the end of .data1 (0x400000 + 0x014DA8E0).
    for name, address in symbols.items():
        assert 0x00400000 < address < 0x018DA8E0, name
    # The retail layout differs from the reconstruction's; a table that merely
    # repeated spec.py's defaults would be silently wrong.
    from touhou_rl import spec

    if spec.ADDRESS_SOURCE.get("addressFile"):
        pytest.skip("TH08_ADDRESS_FILE already points at a table for this process")
    assert symbols["g_GameManager"] != spec.ADDR_GAME_MANAGER
    # The reconstruction assumes this pair is 4 bytes apart, retail places it 8
    # apart -- that is why the reader shim reads both by absolute address.
    assert symbols["g_GuiMessageInputPrevious"] - symbols["g_GuiMessageInputCurrent"] == 8
    assert symbols["g_LastFrameInput"] - symbols["g_CurFrameInput"] == 8


# ---------------------------------------------------------------------------
# windows-retail manifest for the Linux pairing (EXP-P0-062)
# ---------------------------------------------------------------------------
def _probe(**overrides) -> dict:
    probe = {
        "score": 12340,
        "livesF32": 2.0,
        "bombsF32": 3.0,
        "deathsF32": 0.0,
        "powerF32": 0.0,
        "graze": 7,
        "stage": 0,
        "gameplayFrame": 59,
        "flags": 1,
        "replayMode": 1,
        "currentSpell": -1,
        "playerState": 3,
        "playerX": 100.0,
        "playerY": 200.0,
        "stageFrames": 59,
        "runFrames": 59,
    }
    probe.update(overrides)
    return probe


def test_retail_state_projection_is_exact_over_the_seven_fields():
    projection = mod.retail_state_projection(_probe())
    assert set(projection) == set(mod.PROJECTION_FIELDS)
    assert projection["score"] == {"value": 12340, "bits": "0x00003034"}
    assert projection["lives"] == {"value": 2.0, "bits": "0x40000000"}
    assert projection["bombs"] == {"value": 3.0, "bits": "0x40400000"}
    assert projection["power"] == {"value": 0.0, "bits": "0x00000000"}
    assert projection["graze"] == {"value": 7, "bits": "0x00000007"}
    assert projection["stage"] == {"value": 0, "bits": "0x00000000"}


def test_retail_state_projection_refuses_a_partial_probe():
    probe = _probe()
    del probe["graze"]
    with pytest.raises(ValueError, match="graze"):
        mod.retail_state_projection(probe)


def _frame_png(out: Path, frame: int) -> Path:
    path = out / "frames" / f"frame-{frame:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((480, 640, 4), 7, dtype=np.uint8), "RGBA").save(path)
    return path


def _records(out: Path, frames=(59, 60)) -> list[dict]:
    rows = []
    for frame in frames:
        rows.append(
            {
                "logicFrame": frame,
                "image": _frame_png(out, frame),
                "state": _probe(gameplayFrame=frame),
                "frameBefore": frame,
                "frameAfter": frame,
            }
        )
    return rows


def _retail_image() -> dict:
    return {"path": "retail/th08.exe", "size": mod.RETAIL_SIZE, "sha256": mod.RETAIL_SHA256, "verified": True}


def test_build_retail_manifest_publishes_source_frames_and_projection(tmp_path):
    out = tmp_path / "out"
    manifest = mod.build_retail_manifest(
        _records(out),
        replay_path=tmp_path / "a.rpy",
        replay_sha256="a" * 64,
        stage=0,
        retail_image=_retail_image(),
        out_dir=out,
    )
    assert manifest["source"] == "windows-retail"
    assert manifest["replay_sha256"] == "a" * 64
    assert manifest["replay_stage"] == 0
    assert [row["logicFrame"] for row in manifest["frames"]] == [59, 60]
    first = manifest["frames"][0]
    assert first["image"] == "frames/frame-000059.png"
    assert len(first["state_sha256"]) == 64
    assert first["state_sha256"] != manifest["frames"][1]["state_sha256"]
    assert first["state_evidence"]["fields"]["lives"] == 2.0
    assert set(first["state_evidence"]["field_bits"]) == set(mod.PROJECTION_FIELDS)
    assert first["state_evidence"]["sameFrameProven"] is True
    assert "frame fence" in manifest["capture"]["frameFence"]
    assert manifest["capture"]["projectionFields"] == list(mod.PROJECTION_FIELDS)


def test_build_retail_manifest_rejects_non_consecutive_frames(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="consecutive"):
        mod.build_retail_manifest(
            _records(out, frames=(59, 61)),
            replay_path=tmp_path / "a.rpy",
            replay_sha256="a" * 64,
            stage=0,
            retail_image=_retail_image(),
            out_dir=out,
        )


def test_build_retail_manifest_rejects_an_unproven_frame(tmp_path):
    out = tmp_path / "out"
    rows = _records(out, frames=(59,))
    rows[0]["frameAfter"] = 60
    with pytest.raises(ValueError, match="same frame"):
        mod.build_retail_manifest(
            rows,
            replay_path=tmp_path / "a.rpy",
            replay_sha256="a" * 64,
            stage=0,
            retail_image=_retail_image(),
            out_dir=out,
        )


def test_build_retail_manifest_refuses_an_empty_run(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        mod.build_retail_manifest(
            [],
            replay_path=tmp_path / "a.rpy",
            replay_sha256="a" * 64,
            stage=0,
            retail_image=_retail_image(),
            out_dir=tmp_path,
        )


def test_to_rgba_synthesizes_an_opaque_alpha_channel():
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgba = mod.to_rgba(rgb)
    assert rgba.shape == (4, 5, 4)
    assert int(rgba[..., 3].min()) == 255 and int(rgba[..., 3].max()) == 255
    given = np.zeros((2, 2, 4), dtype=np.uint8)
    given[..., 3] = 9
    assert np.array_equal(mod.to_rgba(given), given)
    with pytest.raises(ValueError):
        mod.to_rgba(np.zeros((4, 5), dtype=np.uint8))


def test_write_retail_manifest_records_the_digest(tmp_path):
    path = tmp_path / "windows-retail-manifest.json"
    info = mod.write_retail_manifest(path, {"source": "windows-retail", "frames": [1, 2, 3]})
    assert path.is_file()
    assert info["frames"] == 3
    assert info["sha256"] == mod.sha256_file(path)
    assert json.loads(path.read_text(encoding="utf-8"))["source"] == "windows-retail"


def test_retail_state_digest_uses_the_projection_bits():
    projection = mod.retail_state_projection(_probe())
    digest = mod.retail_state_digest(59, projection)
    assert len(digest) == 64
    # A different logic frame must not share the digest, even with equal fields.
    assert digest != mod.retail_state_digest(60, projection)
    # Neither must a different float pattern with the same printed value.
    other = mod.retail_state_projection(_probe(powerF32=-0.0))
    assert mod.retail_state_digest(59, other) != digest


def test_float_bits_is_the_ieee32_pattern():
    assert mod.float_bits(2.0) == 0x40000000
    assert mod.float_bits(0.0) == 0
    assert mod.float_bits(-0.0) == 0x80000000
    assert struct.unpack("<f", struct.pack("<I", mod.float_bits(220.0)))[0] == 220.0

# ---------------------------------------------------------------------------
# presentation lag (EXP-P0-065)
# ---------------------------------------------------------------------------
def test_seek_counter_target_moves_only_for_a_negative_lag():
    assert mod.seek_counter_target(2352, 0) == 2352
    # A positive lag still needs the probe of the requested frame, so it seeks there.
    assert mod.seek_counter_target(2352, 1) == 2352
    assert mod.seek_counter_target(2352, -1) == 2351
    # The counter never goes below zero, so a negative lag clamps instead of wrapping.
    assert mod.seek_counter_target(0, -1) == 0


def test_present_lag_label_is_the_grab_counter_minus_the_lag():
    assert mod.present_lag_label(2352, 0) == 2352
    assert mod.present_lag_label(2353, 1) == 2352
    assert mod.present_lag_label(2351, -1) == 2352


def _samples(first: int, count: int) -> list[dict]:
    return [
        {
            "counter": first + i,
            "frameBefore": first + i,
            "frameAfter": first + i,
            "probe": _probe(gameplayFrame=first + i),
            "image": np.full((4, 4, 3), i, dtype=np.uint8),
            "t": float(i),
        }
        for i in range(count)
    ]


def test_align_present_lag_takes_pixels_and_state_from_different_samples():
    rows, missing, counters = mod.align_present_lag(_samples(61, 3), needed=2, present_lag=1)
    assert missing == []
    assert counters == [61, 62, 63]
    assert [row["frame"] for row in rows] == [61, 62]
    # label 61: pixels from the grab at counter 62, memory from the probe at 61.
    assert rows[0]["frameBefore"] == 62
    assert rows[0]["stateFrame"] == 61
    assert rows[0]["probe"]["gameplayFrame"] == 61
    assert int(rows[0]["image"][0, 0, 0]) == 1


def test_align_present_lag_reports_missing_labels_instead_of_filling_them():
    rows, missing, _ = mod.align_present_lag(_samples(61, 2), needed=3, present_lag=1)
    assert [row["frame"] for row in rows] == [61]
    assert missing == [62, 63]


def test_build_retail_manifest_records_the_present_lag(tmp_path):
    out = tmp_path / "out"
    rows = _records(out, frames=(59, 60))
    for row in rows:  # every grab now happens one counter after the labelled frame
        row["frameBefore"] = row["frameAfter"] = row["logicFrame"] + 1
    manifest = mod.build_retail_manifest(
        rows,
        replay_path=tmp_path / "a.rpy",
        replay_sha256="a" * 64,
        stage=0,
        retail_image=_retail_image(),
        out_dir=out,
        present_lag=1,
    )
    assert [row["logicFrame"] for row in manifest["frames"]] == [59, 60]
    first = manifest["frames"][0]
    assert first["state_evidence"]["frameCounterBefore"] == 60
    assert first["state_evidence"]["stateFrameCounter"] == 59
    assert first["state_evidence"]["presentLagFrames"] == 1
    assert manifest["capture"]["presentLagFrames"] == 1
    assert "frameCounter - presentLag" in manifest["capture"]["presentLagNote"]


def test_build_retail_manifest_rejects_a_grab_that_ignores_the_present_lag(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=r"logicFrame\+presentLag"):
        mod.build_retail_manifest(
            _records(out, frames=(59,)),
            replay_path=tmp_path / "a.rpy",
            replay_sha256="a" * 64,
            stage=0,
            retail_image=_retail_image(),
            out_dir=out,
            present_lag=1,
        )


def test_build_retail_manifest_rejects_state_from_another_counter(tmp_path):
    out = tmp_path / "out"
    rows = _records(out, frames=(59,))
    rows[0]["stateFrame"] = 60
    with pytest.raises(ValueError, match="describes counter"):
        mod.build_retail_manifest(
            rows,
            replay_path=tmp_path / "a.rpy",
            replay_sha256="a" * 64,
            stage=0,
            retail_image=_retail_image(),
            out_dir=out,
        )


# ---------------------------------------------------------------------------
# double-grab (EXP-P0-066)
# ---------------------------------------------------------------------------
def _grab(index, sha, start=0.0, duration=2.0, counter=59):
    return {
        "index": index,
        "sha256": sha,
        "startMs": start,
        "durationMs": duration,
        "counterBefore": counter,
        "counterAfter": counter,
        "sameFrameProven": True,
    }


def _series(rows, pick="last", shas=("aa", "bb")):
    out = []
    for row in rows:
        grabs = [
            _grab(index, sha, start=index * 6.0, counter=row["logicFrame"])
            for index, sha in enumerate(shas)
        ]
        summary = mod.grab_series_summary(grabs)
        out.append(
            {
                **summary,
                "grabs": grabs,
                "publishedPick": pick,
                "publishedIndex": len(grabs) - 1 if pick == "last" else 0,
                "publishedSha256": shas[-1] if pick == "last" else shas[0],
            }
        )
    return out


def test_grab_series_summary_flags_the_frames_whose_present_is_visible():
    stable = mod.grab_series_summary([_grab(0, "aa"), _grab(1, "aa", start=6.0)])
    split = mod.grab_series_summary(
        [_grab(0, "aa"), _grab(1, "bb", start=6.0), _grab(2, "bb", start=12.0)]
    )
    assert stable["identical"] is True and stable["phaseDecidable"] is False
    assert "not decidable" in stable["note"]
    assert split["identical"] is False and split["phaseDecidable"] is True
    assert split["distinctSha256"] == 2
    assert split["changeIndices"] == [1]
    assert split["firstSha256"] == "aa" and split["lastSha256"] == "bb"
    assert split["spanMs"] == 12.0 and split["maxGapMs"] == 6.0
    assert "post-present" in split["note"]
    assert mod.grab_series_summary([])["count"] == 0


def test_pick_grab_index_is_the_first_or_the_last_grab():
    grabs = [_grab(0, "aa"), _grab(1, "bb", start=6.0), _grab(2, "cc", start=12.0)]
    assert mod.pick_grab_index(grabs, "first") == 0
    assert mod.pick_grab_index(grabs, "last") == 2
    assert mod.other_grab_pick("first") == "last"
    assert mod.other_grab_pick("last") == "first"
    with pytest.raises(ValueError, match="unknown double-grab pick"):
        mod.pick_grab_index(grabs, "middle")
    with pytest.raises(ValueError, match="empty grab series"):
        mod.pick_grab_index([], "first")


class _FakeMem:
    def __init__(self, counters):
        self.counters = list(counters)
        self.reads = 0

    def u32(self, _address):
        value = self.counters[min(self.reads, len(self.counters) - 1)]
        self.reads += 1
        return value


class _FakeCapture:
    def __init__(self):
        self.calls = 0

    def grab(self, _hwnd):
        self.calls += 1
        return np.full((480, 640, 3), self.calls, dtype=np.uint8)


class _FakeSession:
    def __init__(self, counters):
        self.mem = _FakeMem(counters)
        self.capture = _FakeCapture()
        self.hwnd = 1


class _FakeSpec:
    ADDR_GAME_MANAGER = 0x1000
    OFF_GM_GAMEPLAY_FRAME = 0x20


def test_collect_grab_series_stops_when_the_counter_advances():
    # Two counter reads per grab: frame 7 is grabbed three times, then the
    # counter moves on and the fourth grab is never filed under frame 7.
    session = _FakeSession([7, 7] * 3 + [8, 8])
    result = mod.collect_grab_series(
        session, _FakeSpec, 7, 2, 0.0, True, 12, float("inf")
    )
    assert [grab["index"] for grab in result["grabs"]] == [0, 1, 2]
    assert result["stopReason"] == "advanced"
    assert len({grab["sha256"] for grab in result["grabs"]}) == 3  # every grab differs
    assert int(np.asarray(result["images"]["first"]).flat[0]) == 1
    assert int(np.asarray(result["images"]["last"]).flat[0]) == 3


def test_collect_grab_series_stops_at_the_count_without_until_change():
    session = _FakeSession([7, 7] * 8)
    result = mod.collect_grab_series(
        session, _FakeSpec, 7, 2, 0.0, False, 12, float("inf")
    )
    assert len(result["grabs"]) == 2
    assert result["stopReason"] == "count"
    assert session.capture.calls == 2


def test_collect_grab_series_drops_a_grab_that_straddles_the_counter():
    # The first grab is bracketed by (7, 8): it straddled the update and may not
    # be filed under either frame.
    session = _FakeSession([7, 8, 8, 8])
    result = mod.collect_grab_series(
        session, _FakeSpec, 7, 2, 0.0, True, 12, float("inf")
    )
    assert result["grabs"] == []
    assert result["stopReason"] == "straddled"
    assert result["images"] == {"first": None, "last": None}


def test_build_retail_manifest_publishes_the_double_grab_series(tmp_path):
    out = tmp_path / "out"
    rows = _records(out, frames=(59, 60))
    manifest = mod.build_retail_manifest(
        rows,
        replay_path=tmp_path / "a.rpy",
        replay_sha256="a" * 64,
        stage=0,
        retail_image=_retail_image(),
        out_dir=out,
        double_grab={
            "count": 2,
            "intervalMs": 4.0,
            "untilChange": True,
            "pick": "last",
            "maxPerFrame": 12,
            "series": _series(rows),
        },
    )
    evidence = manifest["frames"][0]["state_evidence"]["doubleGrab"]
    assert evidence["publishedPick"] == "last"
    assert evidence["publishedIndex"] == 1
    assert evidence["publishedSha256"] == "bb"
    assert evidence["grabCount"] == 2
    assert evidence["phaseDecidable"] is True
    assert [grab["sha256"] for grab in evidence["grabs"]] == ["aa", "bb"]
    assert evidence["grabs"][0]["counterBefore"] == 59
    block = manifest["capture"]["doubleGrab"]
    assert block["frames"] == 2
    assert block["framesPhaseDecidable"] == 2
    assert block["framesIdentical"] == 0
    assert block["phaseDecidableFraction"] == 1.0
    assert block["pick"] == "last" and block["untilChange"] is True
    assert "single rule applied to every frame" in block["pickRule"]


def test_build_retail_manifest_refuses_a_published_digest_outside_the_series(tmp_path):
    out = tmp_path / "out"
    rows = _records(out, frames=(59,))
    series = _series(rows)
    series[0]["publishedSha256"] = "zz"
    with pytest.raises(ValueError, match="is not one of"):
        mod.build_retail_manifest(
            rows,
            replay_path=tmp_path / "a.rpy",
            replay_sha256="a" * 64,
            stage=0,
            retail_image=_retail_image(),
            out_dir=out,
            double_grab={"series": series},
        )


def test_build_retail_manifest_refuses_a_short_double_grab_series(tmp_path):
    out = tmp_path / "out"
    rows = _records(out, frames=(59, 60))
    with pytest.raises(ValueError, match="covers 1 frames"):
        mod.build_retail_manifest(
            rows,
            replay_path=tmp_path / "a.rpy",
            replay_sha256="a" * 64,
            stage=0,
            retail_image=_retail_image(),
            out_dir=out,
            double_grab={"series": _series(rows[:1])},
        )


def test_build_retail_manifest_refuses_double_grab_with_a_present_lag(tmp_path):
    out = tmp_path / "out"
    rows = _records(out, frames=(59,))
    rows[0]["frameBefore"] = rows[0]["frameAfter"] = 60
    with pytest.raises(ValueError, match="cannot be combined"):
        mod.build_retail_manifest(
            rows,
            replay_path=tmp_path / "a.rpy",
            replay_sha256="a" * 64,
            stage=0,
            retail_image=_retail_image(),
            out_dir=out,
            present_lag=1,
            double_grab={"series": _series(rows)},
        )


def test_main_refuses_double_grab_with_a_present_lag(tmp_path):
    out = tmp_path / "out"
    assert mod.main(
        ["--phase", "capture", "--out", str(out), "--double-grab", "--present-lag", "1"]
    ) == mod.EXIT_USAGE
    assert not out.exists()


def test_main_refuses_a_single_grab_series(tmp_path):
    out = tmp_path / "out"
    assert mod.main(
        [
            "--phase", "capture", "--out", str(out),
            "--double-grab", "--double-grab-count", "1",
        ]
    ) == mod.EXIT_USAGE
    assert not out.exists()


def test_main_refuses_a_max_per_frame_below_the_series_length(tmp_path):
    out = tmp_path / "out"
    assert mod.main(
        [
            "--phase", "capture", "--out", str(out),
            "--double-grab", "--double-grab-count", "4",
            "--double-grab-max-per-frame", "3",
        ]
    ) == mod.EXIT_USAGE
    assert not out.exists()


