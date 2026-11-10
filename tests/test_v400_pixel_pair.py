"""Non-acceptance unit checks of the WP8 manifest pairing contract."""
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.rl.v400_pixel_pair import (
    PROJECTION_FIELDS,
    EvidenceError,
    compare,
    main,
    project_linux,
    project_retail,
)


def manifest(root: Path, source: str, *, logic=7, replay="a" * 64, state="b" * 64, pixel=0, evidence=None):
    root.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((480, 640, 4), pixel, dtype=np.uint8), "RGBA").save(root / "frame.png")
    row = {"logicFrame": logic, "image": "frame.png", "state_sha256": state}
    if evidence is not None:
        row["state_evidence"] = evidence
    path = root / "frames.json"
    path.write_text(json.dumps({"source": source, "replay_sha256": replay, "frames": [row]}))
    return path


def test_pixel_pair_exact_frame_and_regions(tmp_path):
    a = manifest(tmp_path / "w", "windows-fence")
    b = manifest(tmp_path / "l", "linux-harness", pixel=1)
    result = compare(a, b, frames=1)
    assert result["status"] == "measured-not-G0-pass"
    assert result["pairs"] == 1
    assert result["frames"][0]["regions"]["playfield"]["mean_absolute_difference"] == 1.0
    assert result["frames"][0]["regions"]["hud"]["different_pixel_ratio"] == 1.0


@pytest.mark.parametrize("change", ["logic", "replay", "state", "missing"])
def test_pixel_pair_rejects_unpaired_or_unproven(tmp_path, change):
    a = manifest(tmp_path / "w", "windows-fence")
    kwargs = {"logic": 8} if change == "logic" else {"replay": "c" * 64} if change == "replay" else {"state": "c" * 64} if change == "state" else {}
    b = manifest(tmp_path / "l", "linux-harness", **kwargs)
    if change == "missing":
        (b.parent / "frame.png").unlink()
    with pytest.raises(EvidenceError):
        compare(a, b, frames=1)


def test_pixel_pair_rejects_insufficient_wp8_frames(tmp_path):
    a = manifest(tmp_path / "w", "windows-fence")
    b = manifest(tmp_path / "l", "linux-harness")
    with pytest.raises(EvidenceError, match="200"):
        compare(a, b)


def test_default_pairing_stays_strict(tmp_path):
    mismatched = manifest(tmp_path / "w", "windows-fence", state="b" * 64)
    other = manifest(tmp_path / "l", "linux-harness", state="c" * 64)
    with pytest.raises(EvidenceError, match="state mismatch"):
        compare(mismatched, other, frames=1)
    matching = manifest(tmp_path / "w2", "windows-fence", state="b" * 64)
    same = manifest(tmp_path / "l2", "linux-harness", state="b" * 64)
    result = compare(matching, same, frames=1)
    assert result["state_pairing"] == "strict"
    assert result["state_consistency"] == "enforced-equal"


def test_same_logic_frame_keeps_evidence_and_refuses_state_claim(tmp_path):
    windows_evidence = {"bullet_digest": "0badc0de", "enemy_digest": "12345678"}
    linux_evidence = {"algorithm": "fnv1a-64", "fields": {"f3": 7}, "fnv1a64": "0011223344556677"}
    a = manifest(tmp_path / "w", "windows-fence", state="b" * 64, evidence=windows_evidence)
    b = manifest(tmp_path / "l", "linux-harness", state="c" * 64, evidence=linux_evidence)
    result = compare(a, b, frames=1, state_pairing="same-logic-frame")
    assert result["state_pairing"] == "same-logic-frame"
    # Never claim the states agree.
    assert result["state_consistency"] == "not-asserted"
    assert result["frames"][0]["state_sha256_equal"] is False
    assert result["frames"][0]["windows_state_evidence"] == windows_evidence
    assert result["frames"][0]["linux_state_evidence"] == linux_evidence
    warnings = " ".join(result["warnings"]).lower()
    assert "cannot be fully attributed to rendering" in warnings
    assert "not established" in warnings
    assert "confirmed identical" in warnings
    assert result["frames"][0]["regions"]["playfield"]["mean_absolute_difference"] == 0.0


def test_same_logic_frame_still_enforces_replay_and_frame_set(tmp_path):
    a = manifest(tmp_path / "w", "windows-fence", state="b" * 64)
    replay = manifest(tmp_path / "l1", "linux-harness", state="c" * 64, replay="d" * 64)
    with pytest.raises(EvidenceError, match="replay hash"):
        compare(a, replay, frames=1, state_pairing="same-logic-frame")
    logic = manifest(tmp_path / "l2", "linux-harness", state="c" * 64, logic=8)
    with pytest.raises(EvidenceError, match="logicFrame pairs"):
        compare(a, logic, frames=1, state_pairing="same-logic-frame")


def test_cli_writes_report_and_exposes_the_mode(tmp_path):
    matching_w = manifest(tmp_path / "w", "windows-fence", state="b" * 64)
    matching_l = manifest(tmp_path / "l", "linux-harness", state="b" * 64)
    out = tmp_path / "report" / "pixel-report.json"
    assert main(["--windows", str(matching_w), "--linux", str(matching_l), "--frames", "1", "--out", str(out)]) == 0
    strict = json.loads(out.read_text())
    assert strict["state_pairing"] == "strict"
    other_w = manifest(tmp_path / "w2", "windows-fence", state="b" * 64)
    other_l = manifest(tmp_path / "l2", "linux-harness", state="c" * 64)
    assert main([
        "--windows", str(other_w), "--linux", str(other_l), "--frames", "1", "--out", str(out),
        "--state-pairing", "same-logic-frame",
    ]) == 0
    relaxed = json.loads(out.read_text())
    assert relaxed["state_pairing"] == "same-logic-frame"
    assert relaxed["state_consistency"] == "not-asserted"


def test_cli_rejects_unknown_pairing_mode(tmp_path):
    a = manifest(tmp_path / "w", "windows-fence")
    b = manifest(tmp_path / "l", "linux-harness")
    with pytest.raises(SystemExit):
        main(["--windows", str(a), "--linux", str(b), "--frames", "1",
              "--out", str(tmp_path / "r.json"), "--state-pairing", "equal-ish"])


# ---------------------------------------------------------------------------
# common-projection (EXP-P0-062: retail original vs Linux harness)
# ---------------------------------------------------------------------------
FLOAT_PROJECTION = ("lives", "bombs", "deaths", "power")
LINUX_INDEX = {"score": 4, "graze": 5, "deaths": 6, "lives": 7, "bombs": 8, "power": 9, "stage": 1}
DEFAULT_PROJECTION = {
    "score": 0,
    "lives": 2.0,
    "bombs": 3.0,
    "deaths": 0.0,
    "power": 0.0,
    "graze": 0,
    "stage": 0,
}


def float_bits(value) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def field_hex(name: str, value) -> str:
    if name in FLOAT_PROJECTION:
        return f"0x{float_bits(value):08x}"
    return f"0x{int(value) & 0xFFFFFFFF:08x}"


def retail_frame(projection=None, *, logic=7):
    projection = dict(DEFAULT_PROJECTION if projection is None else projection)
    return {
        "algorithm": "retail-memory-projection/1",
        "fields": projection,
        "field_bits": {name: field_hex(name, value) for name, value in projection.items()},
        "frameCounterBefore": logic,
        "frameCounterAfter": logic,
        "sameFrameProven": True,
        "memory": {"gameplayFrame": logic, "stage": projection["stage"], "flags": 0},
    }


def linux_frame(projection=None, *, logic=7):
    projection = dict(DEFAULT_PROJECTION if projection is None else projection)
    fields = {}
    for name, index in LINUX_INDEX.items():
        value = projection[name]
        fields[f"f{index}"] = float_bits(value) if name in FLOAT_PROJECTION else int(value)
    return {
        "algorithm": "fnv1a-64 over the 25 tracked u32 fields (linux harness)",
        "fields": fields,
        "fnv1a64": "0011223344556677",
        "harness_frame": 99,
        "replay_frame_counter": logic,
        "stage": projection["stage"],
    }


def projected_manifest(root: Path, source: str, *, projection=None, logic=7, replay="a" * 64, state="e" * 64):
    root.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((480, 640, 4), 0, dtype=np.uint8), "RGBA").save(root / "frame.png")
    evidence = (
        retail_frame(projection, logic=logic)
        if source == "windows-retail"
        else linux_frame(projection, logic=logic)
    )
    row = {"logicFrame": logic, "image": "frame.png", "state_sha256": state, "state_evidence": evidence}
    path = root / "frames.json"
    path.write_text(json.dumps({"source": source, "replay_sha256": replay, "frames": [row]}))
    return path


def test_projections_relabel_the_same_fields_on_both_sides():
    projection = dict(DEFAULT_PROJECTION, score=12340, graze=7, power=220.0, lives=3.0)
    retail = project_retail(retail_frame(projection), 7)
    linux = project_linux(linux_frame(projection), 7)
    assert retail == linux
    assert set(retail) == set(PROJECTION_FIELDS)
    assert retail["power"] == float_bits(220.0)


def test_common_projection_asserts_equality_on_the_seven_fields(tmp_path):
    a = projected_manifest(tmp_path / "w", "windows-retail", state="e" * 64)
    b = projected_manifest(tmp_path / "l", "linux-harness", state="f" * 64)
    result = compare(a, b, frames=1, state_pairing="common-projection")
    assert result["state_pairing"] == "common-projection"
    assert result["state_consistency"] == "asserted-on-projection"
    assert result["state_projection"]["fields"] == list(PROJECTION_FIELDS)
    assert result["state_projection"]["frames_equal"] == 1
    assert result["state_projection"]["frames_total"] == 1
    assert set(result["state_projection"]["field_equality"]) == set(PROJECTION_FIELDS)
    assert result["frames"][0]["state_projection"]["equal"] is True
    # Different digests are expected on this path and must not block the projection.
    assert result["frames"][0]["state_sha256_equal"] is False
    warnings = " ".join(result["warnings"]).lower()
    assert "seven-field projection" in warnings
    assert "frame fence" in warnings


def test_common_projection_fails_closed_on_the_first_unequal_field(tmp_path):
    a = projected_manifest(tmp_path / "w", "windows-retail")
    b = projected_manifest(tmp_path / "l", "linux-harness", projection=dict(DEFAULT_PROJECTION, lives=3.0))
    with pytest.raises(EvidenceError, match=r"frame 7: common projection mismatch.*\['lives'\]"):
        compare(a, b, frames=1, state_pairing="common-projection")


def test_common_projection_compares_float_patterns_not_decimals(tmp_path):
    # -0.0 and 0.0 print the same but are different 32-bit patterns.
    a = projected_manifest(tmp_path / "w", "windows-retail", projection=dict(DEFAULT_PROJECTION, power=-0.0))
    b = projected_manifest(tmp_path / "l", "linux-harness", projection=dict(DEFAULT_PROJECTION, power=0.0))
    with pytest.raises(EvidenceError, match=r"\['power'\]"):
        compare(a, b, frames=1, state_pairing="common-projection")


def test_common_projection_rejects_missing_projection_fields(tmp_path):
    a = projected_manifest(tmp_path / "w", "windows-retail")
    b = projected_manifest(tmp_path / "l", "linux-harness")
    data = json.loads(b.read_text())
    del data["frames"][0]["state_evidence"]["fields"]["f8"]  # bombs
    b.write_text(json.dumps(data))
    with pytest.raises(EvidenceError, match="f8"):
        compare(a, b, frames=1, state_pairing="common-projection")
    stripped = json.loads(a.read_text())
    del stripped["frames"][0]["state_evidence"]["field_bits"]["graze"]
    a.write_text(json.dumps(stripped))
    with pytest.raises(EvidenceError, match="graze"):
        compare(a, b, frames=1, state_pairing="common-projection")


def test_common_projection_still_enforces_replay_and_frame_set(tmp_path):
    a = projected_manifest(tmp_path / "w", "windows-retail")
    other_replay = projected_manifest(tmp_path / "l1", "linux-harness", replay="d" * 64)
    with pytest.raises(EvidenceError, match="replay hash"):
        compare(a, other_replay, frames=1, state_pairing="common-projection")
    other_frame = projected_manifest(tmp_path / "l2", "linux-harness", logic=8)
    with pytest.raises(EvidenceError, match="logicFrame pairs"):
        compare(a, other_frame, frames=1, state_pairing="common-projection")


def test_default_strict_stays_closed_to_retail_manifests(tmp_path):
    a = projected_manifest(tmp_path / "w", "windows-retail")
    b = projected_manifest(tmp_path / "l", "linux-harness")
    with pytest.raises(EvidenceError, match="source"):
        compare(a, b, frames=1)
    # same-logic-frame keeps its old source requirement too.
    with pytest.raises(EvidenceError, match="source"):
        compare(a, b, frames=1, state_pairing="same-logic-frame")


def test_common_projection_cli_writes_projection_report(tmp_path):
    a = projected_manifest(tmp_path / "w", "windows-retail")
    b = projected_manifest(tmp_path / "l", "linux-harness")
    out = tmp_path / "report" / "pixel-report.json"
    assert main([
        "--windows", str(a), "--linux", str(b), "--frames", "1", "--out", str(out),
        "--state-pairing", "common-projection",
    ]) == 0
    report = json.loads(out.read_text())
    assert report["state_consistency"] == "asserted-on-projection"
    assert report["state_projection"]["fields"] == list(PROJECTION_FIELDS)


def test_common_projection_cli_is_fail_closed_without_a_report(tmp_path):
    a = projected_manifest(tmp_path / "w", "windows-retail")
    b = projected_manifest(tmp_path / "l", "linux-harness", projection=dict(DEFAULT_PROJECTION, bombs=2.0))
    out = tmp_path / "report" / "pixel-report.json"
    assert main([
        "--windows", str(a), "--linux", str(b), "--frames", "1", "--out", str(out),
        "--state-pairing", "common-projection",
    ]) == 3
    assert not out.exists()

# ---------------------------------------------------------------------------
# global presentation phase (EXP-P0-065)
# ---------------------------------------------------------------------------
def sequence_manifest(root: Path, source: str, frames: dict, *, replay="a" * 64, state="e" * 64):
    """One manifest whose per-frame PNG is a constant grey level.

    ``frames`` maps logicFrame -> grey level; a frame ``n`` in the retail manifest
    with level ``n-1`` is exactly the "the grab filed under n shows Linux n-1"
    situation, which is what ``--global-phase`` has to detect.
    """
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for logic, level in frames.items():
        Image.fromarray(np.full((480, 640, 4), level, dtype=np.uint8), "RGBA").save(
            root / f"frame-{logic}.png"
        )
        evidence = (
            retail_frame(logic=logic) if source == "windows-retail" else linux_frame(logic=logic)
        )
        rows.append(
            {
                "logicFrame": logic,
                "image": f"frame-{logic}.png",
                "state_sha256": state,
                "state_evidence": evidence,
            }
        )
    path = root / "frames.json"
    path.write_text(json.dumps({"source": source, "replay_sha256": replay, "frames": rows}))
    return path


def test_global_phase_picks_the_single_stream_offset(tmp_path):
    linux = sequence_manifest(tmp_path / "l", "linux-harness", {n: n for n in range(5, 10)})
    # retail[n] shows Linux n-1: the true stream offset is -1 for every frame.
    retail = sequence_manifest(
        tmp_path / "w", "windows-retail", {n: n - 1 for n in range(5, 10)}
    )
    report = compare(retail, linux, frames=5, state_pairing="common-projection", global_phase=True)
    phase = report["global_phase"]
    assert phase["best_shift"] == -1
    assert phase["best_shift_sum"] == 0.0
    assert phase["runner_up_shift"] in (-1, 0, 1) and phase["runner_up_shift"] != -1
    assert phase["gap_sum"] > 0
    # s=0 is one full grey level off on every comparable frame.
    assert phase["per_shift"]["0"]["playfield_mad_mean"] == 1.0
    assert phase["per_frame_best_shift_counts"]["-1"] == phase["comparable_frames"]
    assert phase["per_frame_best_oracle"]["playfield_mad_mean"] == 0.0
    # Only the frames that have a partner for *every* shift may decide the winner.
    assert phase["comparable_frames"] == 3
    assert "cherry-picking" in phase["note"]
    assert phase["retail_manifest_present_lag_frames"] == 0


def test_global_phase_reports_zero_when_the_retail_grab_is_aligned(tmp_path):
    levels = {n: n for n in range(5, 10)}
    linux = sequence_manifest(tmp_path / "l", "linux-harness", levels)
    retail = sequence_manifest(tmp_path / "w", "windows-retail", levels)
    phase = compare(
        retail, linux, frames=5, state_pairing="common-projection", global_phase=True
    )["global_phase"]
    assert phase["best_shift"] == 0
    assert phase["best_shift_sum"] == 0.0
    assert phase["per_frame_best_shift_counts"]["0"] == phase["comparable_frames"]


def test_global_phase_is_absent_unless_requested(tmp_path):
    levels = {7: 7}
    linux = sequence_manifest(tmp_path / "l", "linux-harness", levels)
    retail = sequence_manifest(tmp_path / "w", "windows-retail", levels)
    report = compare(retail, linux, frames=1, state_pairing="common-projection")
    assert "global_phase" not in report


def test_global_phase_cli_flag_writes_the_block(tmp_path):
    levels = {n: n for n in range(5, 10)}
    linux = sequence_manifest(tmp_path / "l", "linux-harness", levels)
    retail = sequence_manifest(tmp_path / "w", "windows-retail", levels)
    out = tmp_path / "report" / "pixel-report.json"
    assert main([
        "--windows", str(retail), "--linux", str(linux), "--frames", "5",
        "--out", str(out), "--state-pairing", "common-projection", "--global-phase",
    ]) == 0
    report = json.loads(out.read_text())
    assert report["global_phase"]["best_shift"] == 0
    assert report["global_phase"]["comparable_frames"] == 3


def test_global_phase_counts_the_clean_frames_of_every_shift(tmp_path):
    levels = {n: n for n in range(5, 10)}
    linux = sequence_manifest(tmp_path / "l", "linux-harness", levels)
    retail = sequence_manifest(tmp_path / "w", "windows-retail", levels)
    phase = compare(
        retail, linux, frames=5, state_pairing="common-projection", global_phase=True
    )["global_phase"]
    clean = phase["clean"]
    assert clean["threshold"] == 0.2
    # The retail grab is the same frame here, so only s=0 is clean.
    assert clean["per_shift_on_comparable_frames"]["0"]["clean_frames"] == phase["comparable_frames"]
    assert clean["per_shift_on_comparable_frames"]["0"]["clean_fraction"] == 1.0
    assert clean["per_shift_on_comparable_frames"]["-1"]["clean_frames"] == 0
    assert clean["per_shift_on_comparable_frames"]["1"]["clean_frames"] == 0
    assert phase["per_shift_on_comparable_frames"]["0"]["playfield_mad_clean_fraction"] == 1.0
    assert (
        phase["per_frame_best_oracle"]["playfield_mad_clean_frames"] == phase["comparable_frames"]
    )
    assert "oracle bound" in clean["definition"]


def test_global_phase_counts_clean_frames_when_the_stream_offset_is_minus_one(tmp_path):
    linux = sequence_manifest(tmp_path / "l", "linux-harness", {n: n for n in range(5, 10)})
    retail = sequence_manifest(
        tmp_path / "w", "windows-retail", {n: n - 1 for n in range(5, 10)}
    )
    phase = compare(
        retail, linux, frames=5, state_pairing="common-projection", global_phase=True
    )["global_phase"]
    clean = phase["clean"]["per_shift_on_comparable_frames"]
    assert clean["-1"]["clean_frames"] == phase["comparable_frames"]
    assert clean["0"]["clean_frames"] == 0


def test_global_phase_cli_prints_the_clean_counts(tmp_path, capsys):
    levels = {n: n for n in range(5, 10)}
    linux = sequence_manifest(tmp_path / "l", "linux-harness", levels)
    retail = sequence_manifest(tmp_path / "w", "windows-retail", levels)
    assert main([
        "--windows", str(retail), "--linux", str(linux), "--frames", "5",
        "--out", str(tmp_path / "p.json"), "--state-pairing", "common-projection",
        "--global-phase",
    ]) == 0
    printed = capsys.readouterr().out
    assert "CLEAN (playfield MAD <= 0.2)" in printed
    assert "s=0 3/3" in printed
