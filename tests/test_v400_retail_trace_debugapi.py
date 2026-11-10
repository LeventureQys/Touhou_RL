"""Unit tests for the native Debug API retail trace observer (EXP-P0-068).

No game is launched: every test drives a pure reducer or a fake memory image.
The fake image is laid out exactly like the retail addresses the observer reads,
so the row composition path is exercised end to end without a running game.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import re
import struct
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/rl/v400_retail_trace_debugapi.py"
OFFICIAL_OBSERVER = ROOT / "third_party/th08/scripts/replay-external-observer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("v400_retail_trace_debugapi", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


obs = _load_module()


# ---------------------------------------------------------------------------
# fake retail memory image
# ---------------------------------------------------------------------------
class FakeMemory:
    """``ProcessMemory``-compatible read-only view over explicit regions."""

    def __init__(self) -> None:
        self.regions: list[tuple[int, bytearray]] = []
        self.reads = 0

    def region(self, base: int, size: int) -> bytearray:
        buffer = bytearray(size)
        self.regions.append((base, buffer))
        return buffer

    def read(self, address: int, size: int) -> bytes:
        self.reads += 1
        for base, buffer in self.regions:
            if base <= address and address + size <= base + len(buffer):
                start = address - base
                return bytes(buffer[start:start + size])
        raise OSError(f"ReadProcessMemory 0x{address:08X}+{size} failed (fake)")

    def u32(self, address: int) -> int:
        return struct.unpack("<I", self.read(address, 4))[0]

    def close(self) -> None:
        return None


def build_fake_image(*, globals_pointer: int = 0x03000000, manager: int = 0x03100000,
                     flags: int = 8, stage: int = 0, game_frame: int = 0):
    """Lay out every address the observer reads, with ``flags``/``stage`` preset."""
    mem = FakeMemory()
    gm = mem.region(obs.GM, 0x3DE30)
    globals_block = mem.region(globals_pointer, 0xA0)
    player = mem.region(obs.PLAYER, 0x2C0)
    rng = mem.region(obs.RNG, 8)
    input_cell = mem.region(obs.INPUT, 2)
    replay_manager_cell = mem.region(obs.REPLAY_MANAGER, 4)
    manager_block = mem.region(manager, 4)

    # The ``gm`` bytearray is the whole GameManager, so the observer's block at
    # GM+0x3DBA8 maps to offset 0x3DBA8 here (flags at +4, frame at +0x218).
    struct.pack_into("<I", gm, 0x3DBA8 + 4, flags)
    struct.pack_into("<I", gm, 0x3DDC0, game_frame)
    struct.pack_into("<I", gm, 0x3DDC4, stage)
    struct.pack_into("<I", gm, 8, globals_pointer)        # g_GameManager.globals
    struct.pack_into("<I", gm, 0x30, 0x2A)                # difficulty: Easy
    gm[0x3DBA8] = 0  # character: Border Team
    gm[0x3DBA9] = 0  # shot
    struct.pack_into("<I", replay_manager_cell, 0, manager)
    struct.pack_into("<I", manager_block, 0, 0)           # ReplayManager::frameCounter
    return {
        "mem": mem,
        "gm": gm,
        "globals": globals_block,
        "player": player,
        "rng": rng,
        "input": input_cell,
        "replayManagerCell": replay_manager_cell,
        "managerBlock": manager_block,
        "manager": manager,
    }


# ---------------------------------------------------------------------------
# structure / ABI sanity
# ---------------------------------------------------------------------------
def test_wow64_context_layout_matches_x86_context():
    # sizeof(CONTEXT) on x86 is 716 bytes; the debug-register fields must be the
    # first six dwords after ContextFlags.
    assert ctypes.sizeof(obs.WOW64_CONTEXT) == 716
    context = obs.WOW64_CONTEXT()
    assert obs.WOW64_CONTEXT.Dr7.offset == 0x18
    assert obs.WOW64_CONTEXT.Eip.offset == 0xB8
    assert obs.WOW64_CONTEXT.EFlags.offset == 0xC0
    assert obs.WOW64_CONTEXT.Esp.offset == 0xC4
    assert context.Dr7 == 0


def test_debug_event_is_large_enough_for_the_windows_layout():
    # The native DEBUG_EVENT is 168 bytes on x64; a smaller ctypes struct would
    # let WaitForDebugEvent overflow it.
    assert ctypes.sizeof(obs.DEBUG_EVENT) >= 168
    assert obs.DEBUG_EVENT.dwThreadId.offset == 8
    assert obs.DEBUG_EVENT.u.offset >= 12


def test_dr7_arming_bits():
    assert obs.dr7_for(1) == 0x401
    assert obs.dr7_for(3) == 0x415
    assert obs.dr7_for(4) == 0x455
    assert obs.DR7_RESERVED & 0x400
    # R/W and LEN stay 00 (execute, 1 byte) for every slot.
    for count in range(1, 5):
        value = obs.dr7_for(count)
        assert value & ~(obs.DR7_RESERVED | 0x55) == 0
        assert bin(value & 0x55).count("1") == count
    with pytest.raises(ValueError):
        obs.dr7_for(5)


# ---------------------------------------------------------------------------
# WOW64 exception codes
# ---------------------------------------------------------------------------
def test_wow64_status_codes_are_treated_as_breakpoint_traps():
    # EXP-P0-068: the retail x86 process under WOW64 reports its DR0 hit as
    # STATUS_WX86_SINGLE_STEP (0x4000001E).  Routing that to the application
    # killed the game with exit code 0x4000001E on the first live POC run.
    assert obs.STATUS_WX86_SINGLE_STEP == 0x4000001E
    assert obs.STATUS_WX86_BREAKPOINT == 0x4000001F
    assert obs.EXCEPTION_SINGLE_STEP in obs.SINGLE_STEP_CODES
    assert obs.STATUS_WX86_SINGLE_STEP in obs.SINGLE_STEP_CODES
    assert obs.EXCEPTION_BREAKPOINT in obs.BREAKPOINT_CODES
    assert obs.STATUS_WX86_BREAKPOINT in obs.BREAKPOINT_CODES
    assert set(obs.SINGLE_STEP_CODES).isdisjoint(obs.BREAKPOINT_CODES)


# ---------------------------------------------------------------------------
# row composition == replay-external-observer.py:201-208
# ---------------------------------------------------------------------------
def test_compose_row_is_the_official_25_field_row():
    gm = bytearray(0x220)
    globals_data = bytearray(0xA0)
    struct.pack_into("<I", gm, 0x218, 1234)
    gm[0], gm[1] = 2, 3
    for offset, value in ((8, 90_941_860), (12, 4321), (0x64, 0x40000000),
                          (0x74, 0x3F800000), (0x80, 0x40400000), (0x98, 0x41200000),
                          (0x30, 17), (0x3C, 23), (0x24, 70000)):
        struct.pack_into("<I", globals_data, offset, value)
    struct.pack_into("<h", globals_data, 0x22, -3)
    globals_data[0x28] = 0xFF
    player_tail = struct.pack("<3I", 0x3F000000, 0x40000000, 0x40400000)
    rng = struct.pack("<HHI", 0x1234, 0, 900)  # seed u16, pad, generation u32

    row = obs.compose_row(
        calc=7, stage=6, frame=22291, gm=bytes(gm), globals_data=bytes(globals_data),
        player_tail=player_tail, player_first=4, rng=rng, rng_origin=100,
        input_value=5, gm_plus_30=0x2A,
    )
    assert len(row) == len(obs.FIELDS) == 25
    assert row == [
        7, 6, 22291, 1234, 90_941_860, 4321,
        0x40000000, 0x3F800000, 0x40400000, 0x41200000,
        0x3F000000, 0x40000000, 0x40400000, 4,
        0x1234, 800, 5, 17, 23, (-3) & 0xFFFFFFFF, 0xFF, 70000,
        2, 3, 0x2A,
    ]
    assert dict(zip(obs.FIELDS, row))["rngGeneration"] == 800


def test_compose_row_rejects_short_blocks():
    with pytest.raises(ValueError):
        obs.compose_row(calc=1, stage=0, frame=0, gm=b"\0" * 8,
                        globals_data=b"\0" * 0xA0, player_tail=b"\0" * 12,
                        player_first=0, rng=b"\0" * 8, rng_origin=0,
                        input_value=0, gm_plus_30=0)
    with pytest.raises(ValueError):
        obs.compose_row(calc=1, stage=0, frame=0, gm=b"\0" * 0x220,
                        globals_data=b"\0" * 4, player_tail=b"\0" * 12,
                        player_first=0, rng=b"\0" * 8, rng_origin=0,
                        input_value=0, gm_plus_30=0)


def test_row_line_is_json_dumps_with_lf_only():
    row = [1, 2, 3]
    assert obs.row_line(row) == "[1, 2, 3]\n"
    assert obs.row_line(row).endswith("\n") and "\r" not in obs.row_line(row)


def official_row_writer():
    """``(callable, line number)`` for the official ``stream.write(...)`` expression.

    The expression is read out of the official observer itself, so a change to its
    line format fails this test instead of silently drifting from it.
    """
    for number, line in enumerate(OFFICIAL_OBSERVER.read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"\s*stream\.write\((.+)\)\s*", line)
        if match:
            expression = match.group(1)
            assert expression == r'json.dumps(row) + "\n"', expression
            return eval(f"lambda row: {expression}", {"json": json}), number  # noqa: S307
    raise AssertionError("replay-external-observer.py has no stream.write(...) line")


def test_row_line_is_byte_identical_to_the_official_write_expression():
    # The captured row format is the official observer's own expression
    # (replay-external-observer.py:209): json.dumps with the default separators
    # (", ", ": "), ASCII only, one LF per row, no padding field, no trailing space.
    writer, line_number = official_row_writer()
    assert line_number == 209
    # Literal bytes of row 1 of the real 99,113-row capture (.tmp_v400_trace_full_1).
    row = [1, 0, 1, 1, 0, 0, 0, 1073741824, 1077936128, 0, 1128267776, 1136656384,
           1056629064, 3, 32485, 0, 0, 0, 0, 0, 0, 60000, 0, 0, 0]
    line = obs.row_line(row)
    assert line == writer(row)
    assert line == ("[1, 0, 1, 1, 0, 0, 0, 1073741824, 1077936128, 0, 1128267776, "
                    "1136656384, 1056629064, 3, 32485, 0, 0, 0, 0, 0, 0, 60000, 0, 0, 0]\n")
    assert line.encode("utf-8").isascii()
    assert line.count("\n") == 1 and line.endswith("\n") and "\r" not in line
    assert line.startswith("[") and line.endswith("]\n")   # a JSON array, not an object
    assert ", " in line and ": " not in line               # default list separators


def test_composed_row_serializes_signed_fields_as_unsigned():
    # gauge is a signed 16-bit value the official observer masks with 0xffffffff
    # (replay-external-observer.py:207); 45,887 rows of the real capture carry
    # values >= 2^31, so a raw s16 would print "-..." and change the file bytes.
    writer, _ = official_row_writer()
    globals_data = bytearray(0xA0)
    struct.pack_into("<h", globals_data, 0x22, -1)
    struct.pack_into("<I", globals_data, 0x30, 0xFFFFFFFF)
    gm = bytearray(0x220)
    row = obs.compose_row(calc=1, stage=0, frame=0, gm=bytes(gm),
                          globals_data=bytes(globals_data), player_tail=b"\0" * 12,
                          player_first=0, rng=struct.pack("<HHI", 1, 0, 0xFFFFFFFF),
                          rng_origin=0, input_value=0, gm_plus_30=0)
    assert row[19] == 0xFFFFFFFF and row[17] == 0xFFFFFFFF
    assert row[15] == 0xFFFFFFFF
    line = obs.row_line(row)
    assert line == writer(row)
    assert "-" not in line
    assert "4294967295" in line


def test_two_hundred_frame_slice_is_byte_identical_to_rebuilt_official_lines(tmp_path):
    # A 200-frame slice (the G0.1 short-slice size) driven through the real
    # collector: every emitted byte must equal the official observer's own
    # serialization of the same 25 values.
    writer, _ = official_row_writer()
    image = build_fake_image(flags=8, stage=0)
    gm, globals_block = image["gm"], image["globals"]
    stages = [{"index": 0, "inputRecords": 400, "endScore": 1_000}]
    collector = obs.TraceCollector(image["mem"], tmp_path, stages, 0, lambda _m: None)
    collector.set_rng_origin(0, 90)
    for frame in range(1, 201):
        struct.pack_into("<I", image["managerBlock"], 0, frame)
        struct.pack_into("<I", gm, 0x3DDC0, frame)
        struct.pack_into("<I", globals_block, 8, 2_447_070 + frame)
        struct.pack_into("<I", globals_block, 12, frame)
        struct.pack_into("<I", globals_block, 0x30, frame)
        struct.pack_into("<h", globals_block, 0x22, -frame)      # signed gauge
        struct.pack_into("<H", image["rng"], 0, 0x1234 + frame)
        struct.pack_into("<I", image["rng"], 4, 100 + 3 * frame)
        struct.pack_into("<H", image["input"], 0, 0x0100 | (frame & 0xFF))
        collector.on_frame()
    collector.flush()

    raw = (tmp_path / "rows.jsonl").read_bytes()
    lines = raw.split(b"\n")
    assert lines[-1] == b""                     # trailing LF, no blank last line
    assert len(lines) == 201
    assert raw.isascii() and b"\r" not in raw
    rebuilt = b"".join(writer(json.loads(line)).encode("utf-8") for line in lines[:-1])
    assert raw == rebuilt
    rows = [json.loads(line) for line in lines[:-1]]
    assert all(len(row) == len(obs.FIELDS) == 25 for row in rows)
    assert [row[0] for row in rows] == list(range(1, 201))
    assert [row[2] for row in rows] == list(range(1, 201))
    assert [row[19] for row in rows] == [(-frame) & 0xFFFFFFFF for frame in range(1, 201)]
    assert collector.count == 200


# ---------------------------------------------------------------------------
# official menu state machine
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "screen,substate,cursor,state,timer,expected",
    [
        (0, 1, 4, 0, 20, "z"),        # StartMenu item 4 == Replay
        (0, 1, 0, 0, 20, "down"),
        (7, 1, 0, 0, 20, "z"),        # replay list -> enter
        (7, 2, 0, 0, 20, "z"),        # stage select, cursor on startStage
        (7, 2, 3, 0, 20, "down"),
        (7, 3, 0, 0, 20, "down"),     # playback mode, target is 2
        (7, 3, 2, 0, 20, "z"),
        (7, 1, 0, 1, 20, None),       # title screen state != 0
        (7, 1, 0, 0, 11, None),       # sub_timer < 12
        (3, 1, 0, 0, 20, None),       # not a handled screen
    ],
)
def test_menu_key_matches_the_official_observer(screen, substate, cursor, state, timer, expected):
    assert obs.menu_key(screen, substate, cursor, state, timer, 0, 2) == expected


# ---------------------------------------------------------------------------
# identity / manifest
# ---------------------------------------------------------------------------
def test_retail_image_identity_is_pinned():
    info = obs.retail_identity()
    assert info["verified"] is True
    assert info["size"] == obs.RETAIL_SIZE
    assert info["sha256"] == obs.RETAIL_SHA256
    assert info["imageBase"] == "0x400000"
    assert info["dynamicBase"] is False, "a dynamic base would invalidate every RVA"


def test_pe_rva_bytes_reproduces_the_attestation_strings():
    raw = obs.RETAIL_EXE.read_bytes()
    assert obs.pe_rva_bytes(raw, 0x452490 - 0x400000, 13).hex() == "558bec51894dfca1b4d06401c1"
    assert obs.pe_rva_bytes(raw, 0x441F48 - 0x400000, 13).hex() == "b948f56401e8feaaffff8945fc"
    assert obs.pe_rva_bytes(raw, 0xFFFFFF, 13) is None


def test_manifest_expectations_for_the_poc_fixture():
    info = obs.manifest_expectations(obs.DEFAULT_REPLAY)
    assert info["file"] == "th8_ud2cd2.rpy"
    assert info["calculationFrames"] == 99113
    assert info["retailTraceSha256"] == (
        "81f535e3e01743255fe1edbe71740f0b27050702ce4d28848ce934d38be1643c"
    )
    assert info["fixtureId"] == "easy-0-a"
    assert [stage["inputRecords"] for stage in info["stages"]] == [
        9389, 13118, 13175, 20123, 20158, 22292,
    ]


def test_attestation_verifies_a_faithful_mapping_and_rejects_a_tampered_one():
    image = build_fake_image()
    raw = obs.RETAIL_EXE.read_bytes()
    windows = {}
    for _key, address, _a, _b in obs.ATTEST:
        windows[address] = image["mem"].region(address, 13)
        windows[address][:] = obs.pe_rva_bytes(raw, address - 0x400000, 13)
    verdict = obs.attest(image["mem"], obs.RETAIL_EXE)
    assert verdict["ok"] is True
    assert all(check["ok"] for check in verdict["checks"])
    # One flipped byte anywhere in the attested window must fail closed.
    windows[0x00452490][0] ^= 0xFF
    assert obs.attest(image["mem"], obs.RETAIL_EXE)["ok"] is False


# ---------------------------------------------------------------------------
# collector == Observer.stop() row path
# ---------------------------------------------------------------------------
def test_collector_writes_one_official_row_per_calculation_frame(tmp_path):
    image = build_fake_image(stage=0, game_frame=0, flags=8)
    gm = image["gm"]
    globals_block = image["globals"]
    struct.pack_into("<I", globals_block, 8, 1_000)          # score
    struct.pack_into("<I", globals_block, 12, 5)             # graze
    struct.pack_into("<H", image["rng"], 0, 0x2222)          # rng seed
    struct.pack_into("<I", image["rng"], 4, 500)             # rng generation
    stages = [
        {"index": 0, "inputRecords": 4, "endScore": 1_000},
        {"index": 1, "inputRecords": 3, "endScore": 2_000},
    ]
    collector = obs.TraceCollector(image["mem"], tmp_path, stages, 0, lambda _m: None)
    collector.set_rng_origin(0, 100)
    emitted = []
    for frame in range(3):
        struct.pack_into("<I", image["managerBlock"], 0, frame)
        struct.pack_into("<I", gm, 0x3DDC0, 1000 + frame)
        collector.on_frame()
        collector.flush()
        emitted.append(json.loads((tmp_path / "rows.jsonl").read_text().splitlines()[-1]))
    assert collector.count == 3
    assert [row[0] for row in emitted] == [1, 2, 3]     # calc index is 1-based
    assert [row[2] for row in emitted] == [0, 1, 2]     # replay frame counter
    assert [row[3] for row in emitted] == [1000, 1001, 1002]
    assert all(len(row) == 25 for row in emitted)
    assert emitted[0][14] == 0x2222
    assert emitted[0][15] == 400                         # 500 - 100
    assert emitted[0][4] == 1_000                        # score
    raw = (tmp_path / "rows.jsonl").read_bytes()
    assert b"\r" not in raw
    assert raw.count(b"\n") == 3
    collector.finish(reason="unit test")
    payload = json.loads((tmp_path / "complete.json").read_text())
    assert list(payload) == ["rows", "ended", "errors", "stages", "stageFrames",
                            "endScores", "rngOrigins", "naturalReplayExit"]
    assert payload["rows"] == 3
    assert payload["ended"] is False                     # the budget reason is an error
    assert payload["naturalReplayExit"] is False
    assert payload["stages"] == [0]
    assert payload["stageFrames"] == {"0": 2}
    assert payload["endScores"] == {"0": 1_000}
    assert payload["rngOrigins"] == {"0": 100}
    assert payload["errors"] == ["unit test", "Stage sequence [0] differs from [0, 1]"]
    assert collector.stage_rows == {"0": 3}


def test_collector_natural_end_is_not_recorded_as_an_error(tmp_path):
    # Official Observer.finish() is called with error=None from
    # DeleteReplayManager (0x453080), so a completed replay must end with
    # ended=True; the valid tails are replay_file.valid_input_tail's 2/3 for a
    # normal stage and 6/7 for the final stage.
    image = build_fake_image(flags=8, stage=0)
    gm = image["gm"]
    struct.pack_into("<I", image["globals"], 8, 1_000)       # score for every row
    stages = [
        {"index": 0, "inputRecords": 4, "endScore": 1_000},
        {"index": 1, "inputRecords": 8, "endScore": 1_000},
    ]
    collector = obs.TraceCollector(image["mem"], tmp_path, stages, 0, lambda _m: None)
    collector.set_rng_origin(0, 100)
    collector.set_rng_origin(1, 100)
    for stage, frames in ((0, (0, 1, 2)), (1, (0, 1, 2))):
        struct.pack_into("<I", gm, 0x3DDC4, stage)
        for frame in frames:
            struct.pack_into("<I", image["managerBlock"], 0, frame)
            collector.on_frame()
    collector.finish(natural_exit=True)
    payload = json.loads((tmp_path / "complete.json").read_text())
    assert payload["ended"] is True
    assert payload["naturalReplayExit"] is True
    assert payload["errors"] == []
    assert payload["stages"] == [0, 1]
    assert payload["stageFrames"] == {"0": 2, "1": 2}
    assert collector.stage_rows == {"0": 3, "1": 3}


def test_collector_skips_frames_without_replay_input(tmp_path):
    image = build_fake_image(flags=8 | 2)  # isReplay + isDemoMode
    struct.pack_into("<I", image["managerBlock"], 0, 0)
    collector = obs.TraceCollector(image["mem"], tmp_path,
                                   [{"index": 0, "inputRecords": 10, "endScore": 0}], 0,
                                   lambda _m: None)
    collector.set_rng_origin(0, 0)
    for _ in range(3):
        collector.on_frame()
    assert collector.count == 0
    assert collector.filtered == 3
    collector.finish(reason="demo filtered")


def test_collector_refuses_a_stage_without_its_initialization_callback(tmp_path):
    image = build_fake_image(flags=8, stage=1)
    struct.pack_into("<I", image["managerBlock"], 0, 0)
    collector = obs.TraceCollector(image["mem"], tmp_path,
                                   [{"index": 1, "inputRecords": 10, "endScore": 0}], 0,
                                   lambda _m: None)
    collector.on_frame()
    assert collector.finished is True
    assert collector.count == 0
    payload = json.loads((tmp_path / "complete.json").read_text())
    assert any("initialization callback" in error for error in payload["errors"])


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------
def test_grade_reports_a_partial_trace_honestly(tmp_path):
    raw = (json.dumps([0] * 25) + "\n").encode()
    (tmp_path / "rows.jsonl").write_bytes(raw)

    class Collector:
        count = 1
        rows_path = tmp_path / "rows.jsonl"

    class Args:
        phase = "rows"
        budget_seconds = 175.0
        start_stage = 0

    class Debugger:
        hits = {"frame": 1}

    report: dict = {"result": {}}
    expectations = {"calculationFrames": 99113,
                    "retailTraceSha256": "81f535e3e01743255fe1edbe71740f0b27050702ce4d28848ce934d38be1643c"}
    code = obs._grade(Args, report, Debugger, Collector, expectations, tmp_path, lambda _m: None)
    assert code == obs.EXIT_STATE
    verdict = report["verdict"]
    assert verdict["rows"] == 1
    assert verdict["calculationFrames"] == 99113
    assert verdict["rowsMatch"] is False
    assert verdict["shaMatch"] is False
    assert verdict["rowsDifference"] == 99112
    assert verdict["rowsFileSha256"] == obs.sha256_bytes(raw)
    assert verdict["canonicalError"] is None
    assert verdict["shaMatch"] is False                  # not the manifest trace


def test_canonical_trace_hash_is_the_official_compare_function(tmp_path):
    # retailTrace.sha256 is defined by compare-replay-traces.trace_sha256 as
    # sha256(compact schema JSON + "\n" + every row packed as 25 "<I"), i.e.
    # independent of JSON spacing.  Locks the imported definition down.
    import hashlib
    import struct

    rows = [[index] * 25 for index in range(1, 4)]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    schema = json.loads(obs.OFFICIAL_SCHEMA_JSON.read_text(encoding="utf-8"))
    expected = hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        + b"".join(struct.pack("<" + "I" * 25, *row) for row in rows)
    ).hexdigest()
    assert obs.canonical_trace_sha256(path) == expected
    assert obs.canonical_trace_sha256(path) != obs.sha256_file(path)
    # A single changed field must change the canonical hash.
    rows[0][4] += 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert obs.canonical_trace_sha256(path) != expected


def test_grade_accepts_only_an_exact_match(tmp_path):
    rows = json.dumps([0] * 25) + "\n"
    (tmp_path / "rows.jsonl").write_bytes(rows.encode())
    digest = obs.canonical_trace_sha256(tmp_path / "rows.jsonl")

    class Collector:
        count = 1
        rows_path = tmp_path / "rows.jsonl"

    class Args:
        phase = "rows"
        budget_seconds = 175.0
        start_stage = 0

    class Debugger:
        hits = {"frame": 1}

    report: dict = {"result": {}}
    expectations = {"calculationFrames": 1, "retailTraceSha256": digest}
    code = obs._grade(Args, report, Debugger, Collector, expectations, tmp_path, lambda _m: None)
    assert code == obs.EXIT_OK
    verdict = report["verdict"]
    assert verdict["shaMatch"] is True
    assert verdict["canonicalTraceSha256"] == digest
    assert verdict["rowsFileEqualsRetailTraceField"] is False   # the field is not a file hash


def test_grade_calls_a_count_match_with_a_byte_difference_a_mismatch(tmp_path):
    rows = json.dumps([0] * 25) + "\n"
    (tmp_path / "rows.jsonl").write_bytes(rows.encode())

    class Collector:
        count = 1
        rows_path = tmp_path / "rows.jsonl"
        stages = {0: {"index": 0, "name": "Stage 1", "inputRecords": 6, "endScore": 2_447_070}}
        stage_rows = {"0": 1}
        stage_frames = {"0": 0}
        end_scores = {"0": 2_447_070}
        payload = {"ended": True, "naturalReplayExit": True, "errors": [], "stages": [0]}

    class Args:
        phase = "rows"
        budget_seconds = 1500.0
        start_stage = 0

    class Debugger:
        hits = {"frame": 1}

    report: dict = {"result": {}}
    expectations = {"calculationFrames": 1, "retailTraceSha256": "00" * 32}
    code = obs._grade(Args, report, Debugger, Collector, expectations, tmp_path, lambda _m: None)
    assert code == obs.EXIT_STATE
    assert report["status"] == "mismatch"
    verdict = report["verdict"]
    assert verdict["rowsMatch"] is True
    assert verdict["shaMatch"] is False
    assert verdict["canonicalTraceSha256"] is not None
    assert verdict["ended"] is True and verdict["naturalReplayExit"] is True
    assert verdict["stageTable"][0]["tailOk"] is True     # 6 % 6 == final-stage tail
    assert verdict["stageTable"][0]["endScoreMatch"] is True


def test_long_run_cap_is_authorized_for_the_g01_gate():
    # The G0.1 gate authorizes one continuous ~16-25 min run; the cap is what
    # makes --budget-seconds 1500 legal instead of an argparse error.
    assert obs.MAX_BUDGET_SECONDS >= 1500.0


def test_budget_cap_covers_the_adaptive_per_fixture_budget():
    """EXP-P0-076: the batch's largest plan must be a legal observer argument."""
    # 149,986 calculation frames -> ceil(149986 / 50) + 300 = 3300 s
    assert obs.MAX_BUDGET_SECONDS >= 3300.0
    # ... and the two caps are the same number, so the batch can never plan a
    # budget the observer would reject at argument parsing time.
    spec = importlib.util.spec_from_file_location(
        "v400_retail_trace_batch_for_cap_test",
        ROOT / "tools/rl/v400_retail_trace_batch.py")
    batch_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch_module)
    assert batch_module.BUDGET_MAX_SECONDS == obs.MAX_BUDGET_SECONDS
    largest = max(fixture["calculationFrames"]
                  for fixture in batch_module.load_fixtures())
    planned = batch_module.fixture_budget_seconds(largest)
    assert planned == 3300.0
    assert planned <= obs.MAX_BUDGET_SECONDS


# ---------------------------------------------------------------------------
# EXP-P0-074: per-row read windows (fewer ReadProcessMemory calls per row)
# ---------------------------------------------------------------------------
class _FakeWindowReader:
    """``WindowReader`` stand-in over a fake image (no process handle needed)."""

    def __init__(self, mem):
        self.mem = mem
        self.reads = 0

    def read(self, address, size):
        self.reads += 1
        return self.mem.read(address, size)


def build_wide_fake_image(*, globals_pointer: int = 0x03000000, manager: int = 0x03100000,
                          flags: int = 8, stage: int = 0):
    """``build_fake_image`` with the *real* contiguous GM..INPUT span committed.

    The merged window read is ``GM+0x3DBA8 .. INPUT+2``; the real process has
    that whole run mapped, so the fake image must too.  Everything the row reads
    lives inside the single ``gm`` region here, which is exactly the property the
    merged path relies on.
    """
    mem = FakeMemory()
    span = obs.WINDOW_OFFSET + obs.WINDOW_BYTES
    gm = mem.region(obs.GM, span)
    globals_block = mem.region(globals_pointer, 0xA0)
    player = mem.region(obs.PLAYER, obs.PLAYER_BYTES)
    replay_manager_cell = mem.region(obs.REPLAY_MANAGER, 4)
    manager_block = mem.region(manager, 4)
    struct.pack_into("<I", gm, obs.WINDOW_OFFSET + 4, flags)
    struct.pack_into("<I", gm, obs.WINDOW_OFFSET + 0x21C, stage)
    struct.pack_into("<I", gm, 8, globals_pointer)
    struct.pack_into("<I", gm, 0x30, 0x2A)
    struct.pack_into("<I", replay_manager_cell, 0, manager)
    return {
        "mem": mem,
        "gm": gm,
        "globals": globals_block,
        "player": player,
        "replayManagerCell": replay_manager_cell,
        "managerBlock": manager_block,
        "manager": manager,
        "rng_offset": obs.WINDOW_OFFSET + obs.RNG_IN_WINDOW,
        "input_offset": obs.WINDOW_OFFSET + obs.INPUT_IN_WINDOW,
        "tail_offset": obs.PLAYER_TAIL_OFFSET,
    }


def test_merged_row_windows_cover_exactly_the_official_read_points():
    assert obs.WINDOW_OFFSET == 0x3DBA8
    assert obs.GM + obs.WINDOW_OFFSET + obs.RNG_IN_WINDOW == obs.RNG
    assert obs.GM + obs.WINDOW_OFFSET + obs.INPUT_IN_WINDOW == obs.INPUT
    assert obs.GM + obs.WINDOW_OFFSET + obs.WINDOW_BYTES == obs.INPUT + 2
    assert obs.WINDOW_BYTES == 0x47E
    # GM+8 (globals pointer) .. GM+0x30+4 (difficulty)
    assert obs.GM_HEAD_OFFSET == 8
    assert obs.GM_HEAD_BYTES == 0x2C
    assert obs.GM + obs.GM_HEAD_OFFSET + obs.GM_HEAD_BYTES == obs.GM + 0x34
    # PLAYER[0] .. PLAYER+0x2B4+12
    assert obs.PLAYER_TAIL_OFFSET == 0x2B4
    assert obs.PLAYER_BYTES == 0x2C0
    assert (obs.PLAYER + obs.PLAYER_TAIL_OFFSET + obs.PLAYER_TAIL_BYTES
            == obs.PLAYER + obs.PLAYER_BYTES)
    assert obs.ROW_BATCH >= 1


def _prime_wide_image(image, frame: int) -> None:
    """Distinct, frame-dependent values at every row address (both read paths)."""
    gm = image["gm"]
    struct.pack_into("<I", image["managerBlock"], 0, frame)
    struct.pack_into("<I", gm, obs.WINDOW_OFFSET + 0x218, 9_000 + frame)   # gameFrame
    struct.pack_into("<I", gm, image["rng_offset"], 0x1100 + frame)        # rng seed
    struct.pack_into("<I", gm, image["rng_offset"] + 4, 700 + 3 * frame)   # rng generation
    struct.pack_into("<H", gm, image["input_offset"], 0x0200 | (frame & 0xFF))
    player = image["player"]
    player[0] = frame & 0xFF
    struct.pack_into("<3I", player, image["tail_offset"], frame, 2 * frame, 3 * frame)
    globals_block = image["globals"]
    struct.pack_into("<I", globals_block, 8, 5_000 + frame)
    struct.pack_into("<I", globals_block, 12, frame)
    struct.pack_into("<h", globals_block, 0x22, -frame)
    globals_block[0x28] = frame & 0xFF
    struct.pack_into("<I", globals_block, 0x24, 40 + frame)


def test_merged_row_path_is_byte_identical_to_per_address_reads(tmp_path):
    stages = [{"index": 0, "inputRecords": 400, "endScore": 5_001}]
    (tmp_path / "legacy").mkdir()
    (tmp_path / "merged").mkdir()
    legacy_image = build_wide_fake_image()
    merged_image = build_wide_fake_image()
    legacy = obs.TraceCollector(legacy_image["mem"], tmp_path / "legacy", stages, 0,
                                lambda _m: None, merged_reads=False)
    merged = obs.TraceCollector(merged_image["mem"], tmp_path / "merged", stages, 0,
                                lambda _m: None)
    merged._reader = _FakeWindowReader(merged_image["mem"])
    merged._merged = True
    legacy.set_rng_origin(0, 100)
    merged.set_rng_origin(0, 100)
    for frame in range(1, 201):
        _prime_wide_image(legacy_image, frame)
        _prime_wide_image(merged_image, frame)
        legacy.on_frame()
        merged.on_frame()
    legacy.flush()
    merged.flush()
    legacy_bytes = (tmp_path / "legacy" / "rows.jsonl").read_bytes()
    merged_bytes = (tmp_path / "merged" / "rows.jsonl").read_bytes()
    assert legacy_bytes == merged_bytes
    assert legacy.count == merged.count == 200
    # 10 ReadProcessMemory per row the official way, 6 with the merged windows
    # (replay-manager pointer, manager frame, 3 windows, globals block).
    assert legacy_image["mem"].reads == 200 * 10
    # merged: 6 ReadProcessMemory per row (2 dependent reads through mem.read,
    # 3 windows + the globals block through the buffered reader).
    assert merged_image["mem"].reads == 200 * 6
    assert merged._reader.reads == 200 * 4
    assert merged.merged_checks == ()


def test_auto_row_read_mode_falls_back_without_a_process_handle(tmp_path):
    image = build_fake_image()
    collector = obs.TraceCollector(image["mem"], tmp_path,
                                   [{"index": 0, "inputRecords": 4, "endScore": 0}],
                                   0, lambda _m: None)
    assert collector.use_merged_windows() is False
    assert collector._merged is False
    assert collector._reader is None
    assert collector.merged_error is not None
    # deciding twice must not re-probe
    assert collector.use_merged_windows() is False


def test_rows_are_joined_in_memory_and_written_in_batches(tmp_path):
    image = build_fake_image()
    stages = [{"index": 0, "inputRecords": 8, "endScore": 0}]
    collector = obs.TraceCollector(image["mem"], tmp_path, stages, 0, lambda _m: None)
    collector.set_rng_origin(0, 0)
    rows = tmp_path / "rows.jsonl"
    for frame in range(5):
        struct.pack_into("<I", image["managerBlock"], 0, frame)
        collector.on_frame()
        assert rows.stat().st_size == 0        # buffered, not written per row
    collector.flush()
    assert rows.read_bytes().count(b"\n") == 5
    assert collector.rows_written == 5


# ---------------------------------------------------------------------------
# EXP-P0-075: independent budget watchdog + navigation fast-fail
# ---------------------------------------------------------------------------
class FakeCollector:
    """``TraceCollector`` stand-in: records the finish/flush the watchdog asks for."""

    def __init__(self, rows_path: Path, count: int = 0) -> None:
        self.rows_path = rows_path
        self.count = count
        self.finished = False
        self.finished_with: dict | None = None
        self.flushes = 0
        self.raise_on_finish = False

    def finish(self, error=None, reason=None, natural_exit=False) -> None:  # noqa: ANN001
        if self.raise_on_finish:
            raise RuntimeError("row stream is wedged")
        if self.finished:
            return
        self.finished = True
        self.finished_with = {"error": error, "reason": reason,
                              "natural_exit": natural_exit}
        self.rows_path.write_text('{"calc": 1}\n', encoding="utf-8")

    def flush(self) -> None:
        self.flushes += 1


class FakeDebugger:
    def __init__(self) -> None:
        self.hard_stops = 0

    def hard_stop(self) -> dict:
        self.hard_stops += 1
        return {"clearedThreads": 3, "debugActiveProcessStop": True, "error": None}


class FakeGameProcess:
    """``subprocess.Popen`` stand-in for the retail game."""

    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.terminated = 0
        self.killed = 0
        self.returncode: int | None = None
        self._alive = True

    def poll(self):
        return None if self._alive else self.returncode

    def terminate(self) -> None:
        self.terminated += 1
        self._alive = False
        self.returncode = 1

    def kill(self) -> None:
        self.killed += 1
        self._alive = False
        self.returncode = -9

    def wait(self, timeout=None):  # noqa: ANN001
        return self.returncode


class FakeStatus:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    def update(self, **fields) -> None:
        self.updates.append(fields)


def test_watchdog_fires_while_a_navigation_is_blocked_forever(tmp_path):
    """The reported bug: a blocked main thread used to outlive its own budget.

    The fake navigation never returns (no timeout, no stop flag), exactly like the
    39-minute ``easy-2-b`` freeze, and the watchdog still finishes the run.
    """
    report = {"status": "running", "result": {}, "navigation": {}}
    rows_path = tmp_path / "rows.jsonl"
    collector = FakeCollector(rows_path, count=17)
    debugger = FakeDebugger()
    game = FakeGameProcess()
    status = FakeStatus()
    exited: dict = {}
    written: dict = {}

    def write_report() -> None:
        written["payload"] = json.loads(json.dumps(report))

    def on_fire() -> None:
        obs.emit_timeout_state(out=tmp_path, report=report, status=status,
                               collector=collector, debugger=debugger, process=game,
                               write_report=write_report,
                               reason="watchdog: test budget",
                               timer_seconds=0.25)
        exited["code"] = obs.EXIT_TIMEOUT      # stands in for os._exit

    watchdog = obs.Watchdog(0.25, on_fire).start()
    entered = threading.Event()

    def blocking_navigation() -> None:
        entered.set()
        threading.Event().wait()               # never returns

    navigation = threading.Thread(target=blocking_navigation, daemon=True)
    navigation.start()
    assert entered.wait(5.0)

    deadline = time.perf_counter() + 10.0
    while "code" not in exited and time.perf_counter() < deadline:
        time.sleep(0.02)

    assert exited.get("code") == obs.EXIT_TIMEOUT
    assert watchdog.fired.is_set()
    assert navigation.is_alive()               # the fake navigation really is stuck
    # the timeout ending: rows flushed, DRs cleared, detached, game killed, report
    assert collector.finished_with["reason"] == "watchdog: test budget"
    assert collector.flushes >= 1
    assert rows_path.is_file()
    assert debugger.hard_stops == 1
    assert game.terminated == 1
    assert game.poll() is not None
    assert status.updates[-1]["phase"] == "timeout"
    assert written["payload"]["status"] == "timeout"
    assert written["payload"]["result"]["exitCode"] == obs.EXIT_TIMEOUT
    stacks = json.loads((tmp_path / "watchdog-stacks.json").read_text(encoding="utf-8"))
    assert any("blocking_navigation" in line
               for lines in stacks["threads"].values() for line in lines)


def test_watchdog_cancel_before_the_deadline_prevents_a_stale_timeout():
    fired: list[bool] = []
    watchdog = obs.Watchdog(0.2, lambda: fired.append(True)).start()
    assert watchdog.armed
    assert watchdog.cancel() is True
    time.sleep(0.4)
    assert fired == []
    assert watchdog.cancel() is True


def test_watchdog_rearm_gives_a_finishing_run_its_grace():
    fired: list[int] = []
    watchdog = obs.Watchdog(0.1, lambda: fired.append(1)).start()
    deadline = time.perf_counter() + 5.0
    while not fired and time.perf_counter() < deadline:
        time.sleep(0.02)
    assert fired == [1]
    watchdog.rearm(0.1)
    deadline = time.perf_counter() + 5.0
    while len(fired) < 2 and time.perf_counter() < deadline:
        time.sleep(0.02)
    assert fired == [1, 1]


def test_timeout_ending_survives_a_wedged_row_stream(tmp_path):
    report = {"status": "running", "result": {}}
    collector = FakeCollector(tmp_path / "rows.jsonl", count=0)
    collector.raise_on_finish = True
    debugger = FakeDebugger()
    game = FakeGameProcess()
    status = FakeStatus()

    result = obs.emit_timeout_state(out=tmp_path, report=report, status=status,
                                    collector=collector, debugger=debugger,
                                    process=game, reason="watchdog: wedged rows")

    assert any(error.startswith("rows:") for error in result["errors"])
    assert debugger.hard_stops == 1          # the detach still happens
    assert game.terminated == 1              # the game is still killed
    assert status.updates[-1]["phase"] == "timeout"
    assert report["status"] == "timeout"
    assert (tmp_path / "report.json").is_file()
    assert report["result"]["exitCode"] == obs.EXIT_TIMEOUT


def test_navigation_deadline_fails_fast_without_waiting_for_the_budget():
    assert obs.navigation_deadline_hit(False, 91.0, 90.0) is True
    assert obs.navigation_deadline_hit(False, 89.9, 90.0) is False
    assert obs.navigation_deadline_hit(True, 9000.0, 90.0) is False   # playing
    assert obs.navigation_deadline_hit(False, 9000.0, 0.0) is False   # disabled


def test_foreground_transfer_is_bounded_so_a_blocked_attach_cannot_hang_the_run():
    class BlockingWinapi:
        def force_foreground(self, hwnd):  # noqa: ANN001
            threading.Event().wait()       # AttachThreadInput-style permanent block
            return True

    started = time.perf_counter()
    ok, timed_out = obs.foreground_with_timeout(BlockingWinapi(), 0x1234, 0.25)
    assert (ok, timed_out) == (False, True)
    assert time.perf_counter() - started < 5.0


class FakeKeyboard:
    def __init__(self) -> None:
        self.keys: list[set] = []

    def set_keys(self, keys) -> None:  # noqa: ANN001
        self.keys.append(set(keys))

    def tap(self, name: str, hold_s: float = 0.06) -> None:
        self.keys.append({name})

    def release_all(self) -> None:
        self.keys.append(set())


def test_navigator_always_returns_even_when_playback_never_starts(monkeypatch):
    """A title screen that never reaches the replay must not own the thread."""
    image = build_fake_image(flags=0)          # not a replay: playback never starts
    monkeypatch.setattr(obs.Navigator, "_blind", lambda self: None)
    navigator = obs.Navigator(image["mem"], FakeKeyboard(), 0, 2, threading.Event(),
                              lambda _m: None, closed_timeout=0.2)
    navigator.blind_timeout = 0.2
    started = time.perf_counter()
    navigator.run()                            # must not raise, must not hang
    assert time.perf_counter() - started < 10.0
    assert navigator.playback_started is False
    assert navigator.error is None



