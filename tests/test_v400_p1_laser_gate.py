"""Unit tests for the P1-S1 laser gate tool.

The gate itself needs the real engine, so these tests pin down the parts that
must not be wrong when it runs: reply parsing, the field-layout *evidence*
rules (a token is only named when its value fits that name), the per-frame
aggregation, pagination, the game-over stop, and the INVALID verdict.

No engine, no WSL, no network.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.rl import v400_p1_laser_gate as gate  # noqa: E402


def bits(value: float) -> str:
    return f"{struct.unpack('<I', struct.pack('<f', value))[0]:08x}"


def laser_row(slot: int, *, state: int = 1, angle: float = 1.5707964) -> str:
    return ":".join([
        str(slot), str(state), bits(100.0), bits(200.0), bits(0.0),
        bits(angle), bits(0.0), bits(120.0), bits(2.0), bits(8.0), "10", "1",
    ])


def bullet_row(slot: int) -> str:
    return ":".join([
        str(slot), "1", "0", bits(50.0), bits(60.0), bits(0.0),
        bits(1.0), bits(-2.0), "7", bits(3.0), "0",
    ])


SENTINEL_ROW = ":".join(["1536", "6", "0", "00000000", "00000000", "00000000",
                         "00000000", "00000000", "0"])


class FakeClient:
    """Scripted engine: STEP advances one frame, snapshots reflect that frame."""

    def __init__(self, laser_counts, bullet_counts, *, stop_at_frame=None,
                 state_reply="OK frames=0 f0=0 f1=0 f13=0", sentinel=False):
        self.laser_counts = list(laser_counts)
        self.bullet_counts = list(bullet_counts)
        self.stop_at_frame = stop_at_frame
        self.sentinel = sentinel
        self.frame = 0
        self.log: list[str] = []
        self._state_reply = state_reply

    def request(self, line: str, timeout=None) -> str:
        self.log.append(line)
        verb = line.split()[0]
        if verb == "STEP":
            if self.stop_at_frame is not None and self.frame + 1 >= self.stop_at_frame:
                self.frame = self.stop_at_frame
                return f"OK advanced=0 frame={self.frame} exit=1 game_over=1"
            self.frame += 1
            return f"OK advanced=1 frame={self.frame} hash=0000000000000000"
        if verb in ("LASERS", "BULLETS"):
            parts = line.split()
            offset = int(parts[1]) if len(parts) > 1 else 0
            limit = int(parts[2]) if len(parts) > 2 else 16
            counts = self.laser_counts if verb == "LASERS" else self.bullet_counts
            count = counts[min(max(self.frame - 1, 0), len(counts) - 1)]
            rows = [bullet_row(i) for i in range(count)]
            if verb == "BULLETS" and self.sentinel:
                # the engine's last pool slot is SENTINEL and counts as live
                rows = [bullet_row(i) for i in range(count)] + [SENTINEL_ROW]
                count += 1
            if verb == "LASERS":
                rows = [laser_row(i) for i in range(count)]
            page = rows[offset:offset + limit]
            key = "lasers" if verb == "LASERS" else "bullets"
            return (f"OK frames={self.frame} count={count} offset={offset} "
                    f"limit={limit} {key}=" + ",".join(page))
        if verb == "STATE":
            return self._state_reply
        if verb == "DEBUG":
            return "OK curState=2 setupState=0 subthread=0 globals=1 stage=4 difficulty=3"
        return "OK ack"

    def fields(self, line: str, timeout=None) -> dict[str, str]:
        return gate.parse_kv(self.request(line, timeout=timeout))


# ---------------------------------------------------------------- parsing
def test_parse_kv_keeps_empty_value():
    kv = gate.parse_kv("OK frames=0 count=0 offset=0 limit=4 lasers=")
    assert kv["lasers"] == ""
    assert kv["count"] == "0"


def test_parse_page_empty_lasers():
    page = gate.parse_page("OK frames=30 count=0 offset=0 limit=4 lasers=")
    assert page["count"] == 0
    assert page["rows"] == []


def test_parse_page_non_empty():
    reply = "OK frames=7 count=2 offset=0 limit=16 lasers=" + laser_row(3) + "," + laser_row(4)
    page = gate.parse_page(reply)
    assert page["count"] == 2
    assert len(page["rows"]) == 2
    assert page["rows"][0][0] == "3"


def test_parse_page_undefined_count():
    page = gate.parse_page("OK count=undefined lasers=undefined frames=0")
    assert page["count"] is None


def test_parse_state_reads_hex_and_decimal():
    reply = "OK frames=12 f0=12 f1=0 f6=7f04c736 f13=2 f24=1"
    state = gate.parse_state(reply)
    assert state["f0"] == 12
    assert state["f6"] == 0x7F04C736
    assert state["f13"] == 2


def test_state_field_names_are_the_reply_keys():
    names = gate.state_field_names("OK frames=1 f0=1 f13=0 hash=00")
    assert names == ["frames", "f0", "f13", "hash"]
    assert not any("event" in name.lower() for name in names)


def test_bits_to_float_round_trip():
    assert gate.bits_to_float(bits(1.5707964)) == pytest.approx(1.5707964, rel=1e-6)
    assert gate.bits_to_float("zzzz") is None


# ---------------------------------------------------------------- layout
def test_layout_verified_for_documented_laser_row():
    rows = [laser_row(0).split(":"), laser_row(1, angle=0.5).split(":")]
    layout = gate.classify_layout(rows, gate.LASER_DOC_FIELDS, kind="laser")
    assert layout["verified"] is True
    assert layout["fieldNames"] == gate.LASER_DOC_FIELDS
    assert layout["lastTokenAlwaysOne"] is True


def test_layout_rejects_short_row_instead_of_padding():
    row = laser_row(0).split(":")[:9]
    layout = gate.classify_layout([row], gate.LASER_DOC_FIELDS, kind="laser")
    assert layout["verified"] is False
    assert layout["tokenCounts"] == [9]
    assert "width" not in layout["fieldNames"]


def test_layout_unconfirms_token_outside_its_range():
    row = laser_row(0).split(":")
    row[2] = bits(99999.0)  # posX far outside the playfield
    layout = gate.classify_layout([row], gate.LASER_DOC_FIELDS, kind="laser")
    assert layout["verified"] is False
    assert layout["fields"][2]["name"] == "unconfirmed"
    assert layout["fields"][2]["documentedName"] == "posX"


def test_layout_prefix_is_confirmed_when_the_build_omits_appended_fields():
    # the pinned binary emits 11 laser tokens: documented minus the trailing inUse
    row = laser_row(0).split(":")[:11]
    layout = gate.classify_layout([row], gate.LASER_DOC_FIELDS, kind="laser")
    assert layout["verified"] is False              # full documented layout unproven
    assert layout["verifiedPrefix"] is True         # the 11 present names are confirmed
    assert layout["verifiedNames"] == gate.LASER_DOC_FIELDS[:11]
    assert layout["missingDocumentedSuffix"] == ["inUse"]


def test_layout_without_rows_is_not_a_pass():
    layout = gate.classify_layout([], gate.LASER_DOC_FIELDS, kind="laser")
    assert layout["status"] == "no-nonempty-row"
    assert layout["verified"] is False


# ---------------------------------------------------------------- frames
def test_collect_frames_counts_and_paginates():
    # frame 1 -> 20 lasers (two pages), frame 2 -> 3, frame 3 -> 0
    client = FakeClient([20, 3, 0], [6, 7, 5])
    source = gate._new_source("reset-stage1-normal", "reset", {})
    gate.collect_frames(client, source, 3, sample_every=2)

    lasers = source["lasers"]
    assert source["framesObserved"] == 3
    assert lasers["nonEmptyFrames"] == 2
    assert lasers["totalEntries"] == 23
    assert lasers["maxCount"] == 20
    assert lasers["framesWithPaging"] == 1
    assert lasers["headerPagedMismatchFrames"] == 0
    assert source["bullets"]["nonEmptyFrames"] == 3
    assert source["bullets"]["maxCount"] == 7
    # one sample at frame 2 (every 2 frames), plus the first frame
    assert [sample["frame"] for sample in source["samples"]] == [1, 2]


def test_collect_frames_stops_on_game_over():
    client = FakeClient([1, 5], [3, 4], stop_at_frame=2)
    source = gate._new_source("practice-stage6", "reset", {})
    gate.collect_frames(client, source, 10, sample_every=500)
    assert source["gameOver"] is True
    assert source["endReason"] == "game_over"
    assert source["framesObserved"] == 1
    assert source["lasers"]["nonEmptyFrames"] == 1


def test_sentinel_slot_is_removed_from_effective_bullet_numbers():
    # the engine counts the pool's SENTINEL slot as a live bullet every frame
    client = FakeClient([0, 0, 0], [2, 3, 0], sentinel=True)
    source = gate._new_source("reset-stage1-normal", "reset", {})
    gate.collect_frames(client, source, 3, sample_every=2)
    bullets = source["bullets"]
    assert bullets["sentinelObserved"] is True
    assert bullets["sentinelAdjustment"] == 1
    assert [e["row"] for e in bullets["sentinelEvidence"]][0] == SENTINEL_ROW.split(":")
    assert bullets["rawNonEmptyFrames"] == 3          # 3, 4, 1 -- all non-zero
    assert bullets["rawMaxCount"] == 4
    assert bullets["nonEmptyFrames"] == 2             # 2, 3, 0
    assert bullets["maxCount"] == 3
    assert "3" in bullets["countHistogram"]


def test_measure_timing_reports_per_call_stats():
    client = FakeClient([2], [9])
    timing = gate.measure_timing(client, frames=4)
    assert timing["frames"] == 4
    assert timing["lasers"]["frames"] == 4
    assert timing["bullets"]["frames"] == 4
    assert timing["pair"]["meanMs"] >= 0.0
    assert timing["laserRowsSeen"] == 8
    assert timing["bulletCountSum"] == 36


# ---------------------------------------------------------------- verdict
def _source(source_id, laser_frames=0, laser_rows=None):
    source = gate._new_source(source_id, "reset", {})
    source["framesObserved"] = 10
    source["lasers"]["nonEmptyFrames"] = laser_frames
    source["lasers"]["maxCount"] = 3 if laser_frames else 0
    source["lasers"]["totalEntries"] = 6 if laser_frames else 0
    if laser_rows:
        source["lasers"]["firstRows"] = [{"frame": 1, "count": len(laser_rows),
                                          "rows": laser_rows}]
    return source


def test_verdict_invalid_when_every_source_is_empty():
    sources = [_source("reset-stage1-normal"), _source("replay-stage5"),
               _source("practice-stage6")]
    verdict = gate.decide_verdict(sources)
    assert verdict["status"] == "INVALID"
    assert verdict["lasersAvailable"] is False
    assert "INVALID" in verdict["note"]
    assert verdict["laserFieldLayoutVerified"] is False


def test_verdict_valid_and_layout_verified_when_rows_exist():
    rows = [laser_row(0).split(":")]
    sources = [_source("reset-stage1-normal"),
               _source("replay-stage5", laser_frames=4, laser_rows=rows),
               _source("practice-stage6")]
    verdict = gate.decide_verdict(sources)
    assert verdict["status"] == "VALID"
    assert verdict["laserFieldLayoutVerified"] is True
    assert verdict["primarySourcesWithLasers"] == ["replay-stage5"]
    assert verdict["maxLaserCount"] == 3


def test_parse_scan_spec_uses_stage_slots():
    assert gate.parse_scan_spec("replay:0,4;practice:6") == [
        ("replay", 0), ("replay", 4), ("practice", 6)]
    assert gate.parse_scan_spec("") == []
    with pytest.raises(SystemExit):
        gate.parse_scan_spec("wine:1")
    with pytest.raises(SystemExit):
        gate.parse_scan_spec("practice:99")


def test_stage_names_match_replay_slots():
    assert gate.STAGE_NAMES[0] == "Stage 1"
    assert gate.STAGE_NAMES[5] == "Stage 5"
    assert gate.STAGE_NAMES[6] == "Final A"
    assert len(gate.STAGE_NAMES) == 9


def test_verdict_says_gate_still_empty_when_only_scan_has_lasers():
    sources = [_source("reset-stage1-normal"), _source("replay-stage5"),
               _source("practice-stage6")]
    scan = [_source("scan-practice-finala", laser_frames=12, laser_rows=[laser_row(1).split(":")])]
    verdict = gate.decide_verdict(sources, scan)
    assert verdict["status"] == "VALID"
    assert verdict["primarySourcesWithLasers"] == []
    assert verdict["scanSourcesWithLasers"] == ["scan-practice-finala"]
    assert "NOT in any of the explicitly requested sources" in verdict["note"]


def test_next_steps_point_at_more_stages_not_at_guessing():
    steps = gate.next_steps()
    assert any("th08_replay" in step for step in steps)
    assert any("practice" in step.lower() for step in steps)
    assert all(isinstance(step, str) and step for step in steps)


# ---------------------------------------------------------------- report
def test_report_json_is_serialisable(tmp_path):
    payload = {"sources": [_source("reset-stage1-normal")], "note": "空"}
    path = tmp_path / "laser_gate.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["note"] == "空"


def test_probe_capabilities_flags_missing_event_field():
    client = FakeClient([0], [0], state_reply="OK frames=0 f0=0 f1=0")
    client.request = _probe_router(client.request)
    caps = gate.probe_capabilities(client)
    assert caps["stateFieldNames"] == ["frames", "f0", "f1"]
    assert caps["stateEventFieldPresent"] is False
    assert caps["commands"]["EVENTS"]["ok"] is False
    assert caps["hazardsCommandPresent"] is False


def _probe_router(inner):
    def request(line, timeout=None):
        if line.startswith("EVENTS"):
            raise gate.EngineError("'EVENTS' -> ERR unknown-command EVENTS")
        if line.startswith("HAZARDS"):
            raise gate.EngineError("'HAZARDS 0' -> ERR unknown-command HAZARDS")
        if line.startswith("PLAYER"):
            return "OK frames=0 hp=1 lives=2"
        return inner(line, timeout=timeout)
    return request
