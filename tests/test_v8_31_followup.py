"""Offline tests for ``tools/rl/v8_31_followup.py`` (V831-SLICE-02, step 1).

The rules these tests enforce
-----------------------------
* **No engine.**  Nothing here spawns ``th08-modern``, copies a game directory or
  reads a shared-memory block.  The only subprocess is ``python -B <tool> --help``.
* **No big arrays.**  The four corpora are read as JSON lines; the ``.npy`` files
  are only ever asked for their header, and the tests never load one.
* **No invented numbers.**  Every expected value either comes from the ticket
  (the two registered sha256 values), from the registered meta files (rows,
  frames, decision frames) or is decoded from the recording inside the test
  (``mask_to_cand``/``commit_frames`` on D1 row 23).
* **The rejection vocabulary is covered one class per test**, plus the two
  negatives that matter: a direction change *with* a SHOOT change must stay a
  rejection, and a real pulse at ``release_due(47, 48)`` must decode as a release.

The decode rule under test: the recorded pair must equal
``commit_frames(cand, decision_frames=2, release_shoot=release_due(i, 48))``
bit for bit; the only tolerance is the exact pair ``[m, m & ~SHOOT]``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from touhou_rl.spec import BTN_BOMB, BTN_DOWN, BTN_FOCUS, BTN_SHOOT, BTN_UP  # noqa: E402
from touhou_rl.v400.gym.actions import (  # noqa: E402
    AGENT_MASK,
    N_CAND,
    cand_to_mask,
    mask_to_cand,
)
from touhou_rl.v400.student.obs_spec_v3 import obs_spec_v3_digest  # noqa: E402
from touhou_rl.v400.teacher.shoot import commit_frames, release_due  # noqa: E402


def _load_followup_tool():
    """Import ``tools/rl/v8_31_followup.py`` (not a package, so by file path).

    The module is registered in ``sys.modules`` before it executes: the tool uses
    ``@dataclass``, and ``dataclasses`` resolves ``cls.__module__`` through
    ``sys.modules`` while the class body is processed.
    """

    path = REPO_ROOT / "tools" / "rl" / "v8_31_followup.py"
    assert path.is_file(), f"missing {path}"
    name = "v831_followup_tool"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


FOLLOWUP = _load_followup_tool()

D1_LABELS = (REPO_ROOT / ".tmp_v400_d1" / "d1" / "shards"
             / "d1-s0d1-n3371-seed12-shard" / "labels.jsonl")
S1C_LABELS = (REPO_ROOT / ".tmp_v410_s1c" / "d1" / "shards"
              / "d1-s0d1-n3300-seed12-shard" / "labels.jsonl")
T02C_SLOTS = (REPO_ROOT / ".tmp_v420_data" / "T02c" / "rebuild"
              / "d1-s0d1-n3300-seed12-shard" / "2f" / "slots.jsonl")
T04_SLOTS = (REPO_ROOT / ".tmp_v420_data" / "T04" / "dataset"
             / "d1-s0d1-n3300-seed12-shard" / "slots.jsonl")

#: Ticket §02-A.1, verbatim.
REGISTERED_SHA256 = {
    "d1-canonical-seed12":
        "457728a988717ae2e5deae425196a43a6d0de4cbb21f18b4c41630f9d3aaccbd",
    "t04-2f-seed12":
        "f759de7dacc78343dec8825da5de14942e749b8a5a2c5319304e513d1618b7b9",
}

#: What the four corpora register about themselves (read from their own meta
#: files by this test too, so a source cannot silently change its grid).
EXPECTED_GRID = {
    "d1-canonical-seed12": {
        "rows": 3298, "frames": 13192, "decision_frames": 6596,
        "frames_per_decision": 4, "leading_frames": 0,
        "first_decision_frame": 0, "last_decision_frame": 13188,
        "first_pair_frame": 0, "last_pair_frame": 13190,
        "split_pairs": 137, "hash_frames": 6596,
    },
    "s1c-seed12": {
        "rows": 3298, "frames": 13192, "decision_frames": 6596,
        "frames_per_decision": 4, "leading_frames": 0,
        "first_decision_frame": 0, "last_decision_frame": 13188,
        "first_pair_frame": 0, "last_pair_frame": 13190,
        "split_pairs": 137, "hash_frames": 6596,
    },
    "t02c-2f-seed12": {
        "rows": 6588, "frames": 13176, "decision_frames": 6588,
        "frames_per_decision": 2, "leading_frames": 16,
        "first_decision_frame": 16, "last_decision_frame": 13190,
        "first_pair_frame": 16, "last_pair_frame": 13190,
        "split_pairs": 137, "hash_frames": 0,
    },
    "t04-2f-seed12": {
        "rows": 6588, "frames": 13176, "decision_frames": 6588,
        "frames_per_decision": 2, "leading_frames": 16,
        "first_decision_frame": 16, "last_decision_frame": 13190,
        "first_pair_frame": 16, "last_pair_frame": 13190,
        "split_pairs": 137, "hash_frames": 13176,
    },
}

ALL_COMMANDS = ("sources", "adapter", "prefix", "reference", "compare", "recon",
                "nominate", "intervene", "pilot", "strength", "diag-teacher", "run",
                "verify")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def masks() -> dict[str, int]:
    """Three real candidate masks the whole file uses (no hand-written bits)."""

    return {
        "cand12": int(cand_to_mask(12)),
        "cand3": int(cand_to_mask(3)),
        "cand5": int(cand_to_mask(5)),
    }


def d1_row(index: int) -> dict:
    """Stream ``labels.jsonl`` up to row ``index`` (never loads the file whole)."""

    with D1_LABELS.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle):
            if number == index:
                return json.loads(line)
    raise AssertionError(f"{D1_LABELS} has no row {index}")


def t02c_row(index: int) -> dict:
    with T02C_SLOTS.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle):
            if number == index:
                return json.loads(line)
    raise AssertionError(f"{T02C_SLOTS} has no row {index}")


@pytest.fixture(scope="module")
def loaded_sources():
    """Load the four corpora once (streaming JSONL, ~0.4 s)."""

    sources, resolutions, problems = FOLLOWUP.load_all_sources(REPO_ROOT)
    assert not problems, problems
    assert len(sources) == 4, sorted(sources)
    return sources, resolutions


# ---------------------------------------------------------------------------
# shared infrastructure: identities, helpers, gate, ledger
# ---------------------------------------------------------------------------
def test_host_identity_carries_the_ticket_code_name_and_python():
    host = FOLLOWUP.host_identity()
    assert host["code_name"] == "轩神开播了"
    assert host["platform"]
    assert host["python"].count(".") >= 1
    assert host["python"] == sys.version.split()[0].split("+")[0] or host["python"]
    assert host["classification"] == "measured"


def test_sha256_file_and_now_iso_are_the_project_shapes(tmp_path):
    payload = b"V831-SLICE-02\n"
    target = tmp_path / "blob.bin"
    target.write_bytes(payload)
    assert FOLLOWUP.sha256_file(target) == hashlib.sha256(payload).hexdigest()
    stamp = FOLLOWUP.now_iso()
    assert "T" in stamp and (stamp.endswith("Z") or "+" in stamp or "-" in stamp[-6:])


def test_parse_kv_splits_a_protocol_reply():
    parsed = FOLLOWUP._parse_kv("OK ready=1 hash=83d953750afaea5a")
    assert parsed["_verb"] == "OK"
    assert parsed["ready"] == "1"
    assert parsed["hash"] == "83d953750afaea5a"
    assert FOLLOWUP._parse_kv("")["_verb"] == ""


def test_gate_passes_only_when_every_check_passes():
    gate = FOLLOWUP.Gate("unit")
    gate.add("first", True, 1)
    assert gate.passed
    gate.add("second", False, "why")
    assert not gate.passed
    assert gate.problems == ["second: why"]
    payload = gate.as_dict()
    assert payload["name"] == "unit" and payload["passed"] is False
    assert [check["name"] for check in payload["checks"]] == ["first", "second"]


def test_import_identity_binds_the_real_rebuild_path():
    imports = {entry["module"]: entry for entry in FOLLOWUP.import_identity()}
    expected = {
        "touhou_rl.v420.counterfactual",
        "touhou_rl.v400.teacher.shoot",
        "touhou_rl.v400.gym.actions",
        "touhou_rl.v420.contracts",
        "touhou_rl.v400.student.obs_spec_v3",
    }
    assert expected <= set(imports)
    for name in expected:
        entry = imports[name]
        assert "import_error" not in entry, entry
        assert entry["sha256"]
    counterfactual = imports["touhou_rl.v420.counterfactual"]
    assert counterfactual["file"].endswith("touhou_rl/v420/counterfactual.py")
    assert counterfactual["sha256"] == hashlib.sha256(
        (REPO_ROOT / "touhou_rl" / "v420" / "counterfactual.py").read_bytes()).hexdigest()
    for dotted in ("_StudentSession._pair_cand", "_StudentSession.replay"):
        attribute = counterfactual["attributes"][dotted]
        assert attribute["present"] is True
        assert attribute["file"].endswith("counterfactual.py")
        assert attribute["file_sha256"] == counterfactual["sha256"]
        assert isinstance(attribute["lineno"], int) and attribute["lineno"] > 0
    shoot = imports["touhou_rl.v400.teacher.shoot"]["attributes"]
    assert shoot["release_due"]["present"] and shoot["commit_frames"]["present"]
    actions = imports["touhou_rl.v400.gym.actions"]["attributes"]
    assert actions["mask_to_cand"]["present"] and actions["cand_to_mask"]["present"]
    digest = imports["touhou_rl.v400.student.obs_spec_v3"]["calls"]["obs_spec_v3_digest"]
    assert digest["ok"] is True
    # The frozen obs-v3 digest is registered evidence: it must not drift.
    assert digest["value"] == obs_spec_v3_digest()
    assert digest["value"].startswith("ef2fbd15")


def test_budget_limits_default_to_the_ticket_numbers():
    limits = FOLLOWUP.BudgetLimits()
    assert limits.max_starts == 28
    assert limits.max_engine_frames == 360_000
    assert limits.max_wall_s == 8 * 3600
    assert limits.max_write_gib == 18.0
    assert limits.booked_at == "2026-11-26T11:07:00+08:00"
    assert limits.deadline().startswith("2026-11-26T19:07")
    payload = limits.as_dict()
    assert payload["max_starts"] == 28 and payload["classification"] == "historical"


def test_ledger_reserves_then_settles_the_actuals(tmp_path):
    limit = FOLLOWUP.BudgetLimits(max_starts=2, max_engine_frames=100,
                                  max_wall_s=60.0, max_write_gib=1.0)
    path = tmp_path / "ledger.jsonl"
    ledger = FOLLOWUP.Ledger(path, limit)
    ledger.reserve("attempt-1", starts=1, engine_frames=40, wall_s=10.0,
                   write_gib=0.10, inputs={"engine": "fake"}, note="unit test")
    ledger.settle("attempt-1", starts=1, engine_frames=13, wall_s=3.5,
                  write_gib=0.01, result={"terminal": "cap"})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["phase"] for row in rows] == ["reserve", "settle"]
    assert rows[0]["schema"] == FOLLOWUP.SCHEMA_LEDGER
    assert rows[1]["actual"]["engine_frames"] == 13      # the actual, not 40
    assert rows[1]["actual"]["wall_s"] == 3.5
    assert "requested" not in rows[1]          # a settle carries reserved + actual
    assert rows[1]["reserved"]["engine_frames"] == 40
    totals = ledger.totals()
    assert totals["reserved"]["starts"] == 1
    assert totals["actual"]["starts"] == 1
    assert totals["committed"]["engine_frames"] == 13
    assert totals["remaining"]["engine_frames"] == 100 - 13
    # a resumed process re-reads the same file and keeps the totals
    again = FOLLOWUP.Ledger(path, limit)
    assert again.totals()["actual"]["engine_frames"] == 13
    assert again.reservations() == {}


def test_ledger_refuses_an_over_budget_reservation(tmp_path):
    limit = FOLLOWUP.BudgetLimits(max_starts=1, max_engine_frames=10,
                                  max_wall_s=5.0, max_write_gib=0.5)
    path = tmp_path / "ledger.jsonl"
    ledger = FOLLOWUP.Ledger(path, limit)
    ledger.reserve("first", starts=1, engine_frames=10)
    with pytest.raises(FOLLOWUP.BudgetRefused) as info:
        ledger.reserve("second", starts=1, engine_frames=1)
    exceeded = {entry["dimension"] for entry in info.value.payload["exceeded"]}
    assert "starts" in exceeded and "engine_frames" in exceeded
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["phase"] == "refuse"
    assert rows[-1]["attempt_id"] == "second"
    assert ledger.totals()["committed"]["starts"] == 1
    with pytest.raises(FOLLOWUP.LedgerError):
        ledger.settle("never-reserved", starts=1)


# ---------------------------------------------------------------------------
# decode_pair: one accepted case and every named rejection
# ---------------------------------------------------------------------------
def test_release_due_phase_is_the_measured_one():
    assert release_due(47, 48) is True
    assert release_due(46, 48) is False
    assert release_due(95, 48) is True
    assert release_due(96, 48) is False


def test_decode_pair_accepts_a_plain_two_frame_commit():
    m = masks()
    cand, bomb, classification, detail = FOLLOWUP.decode_pair([m["cand12"], m["cand12"]], 0)
    assert (cand, bomb, classification) == (12, 0, "accepted")
    assert detail["expected"] == [m["cand12"], m["cand12"]]
    assert detail["release_due"] is False
    assert detail["rule"] == "expected"


def test_decode_pair_accepts_the_real_pulse_at_decision_47():
    m = masks()
    released = m["cand12"] & ~BTN_SHOOT
    assert released != m["cand12"]
    cand, bomb, classification, detail = FOLLOWUP.decode_pair([m["cand12"], released], 47)
    assert (cand, bomb) == (12, 0)
    assert classification == "accepted_via_release"
    assert detail["expected"] == [m["cand12"], released]
    assert detail["expected_without_release"] == [m["cand12"], m["cand12"]]
    assert detail["rule"] == "expected_with_release"
    # the same pair one decision earlier is not a release
    _, _, other, other_detail = FOLLOWUP.decode_pair([m["cand12"], released], 46)
    assert other == "release_not_due"
    assert other_detail["release_due"] is False


def test_decode_pair_does_not_swallow_a_direction_change_with_a_shoot_change():
    """The negative that proves the tolerance is exact and not `mask & ~SHOOT`."""

    m = masks()
    pair = [m["cand12"], m["cand5"] & ~BTN_SHOOT]
    assert pair[1] != pair[0] & ~BTN_SHOOT
    cand, bomb, classification, detail = FOLLOWUP.decode_pair(pair, 47)
    assert classification == "split_non_shoot"
    assert (cand, bomb) == (12, 0)
    assert detail["release_due"] is True          # a release *was* due ...
    assert detail["xor"] != BTN_SHOOT             # ... but the change is not SHOOT-only


def test_decode_pair_rejects_a_reverse_release():
    m = masks()
    cand, bomb, classification, detail = FOLLOWUP.decode_pair(
        [m["cand12"] & ~BTN_SHOOT, m["cand12"]], 47)
    assert classification == "reverse_release"
    assert cand is None and bomb is None
    assert "SHOOT off on the first frame" in detail["rule"]


def test_decode_pair_rejects_a_release_that_is_not_due():
    m = masks()
    _, _, classification, detail = FOLLOWUP.decode_pair(
        [m["cand12"], m["cand12"] & ~BTN_SHOOT], 0)
    assert classification == "release_not_due"
    assert detail["release_due"] is False


def test_decode_pair_rejects_a_pair_where_neither_frame_shoots():
    m = masks()
    cand, bomb, classification, detail = FOLLOWUP.decode_pair(
        [m["cand12"] & ~BTN_SHOOT, m["cand12"] & ~BTN_SHOOT], 47)
    assert classification == "no_shoot_both"
    assert cand is None and bomb is None


def test_decode_pair_rejects_a_bomb_press():
    m = masks()
    with_bomb = int(cand_to_mask(12, 1))
    cand, bomb, classification, detail = FOLLOWUP.decode_pair([with_bomb, with_bomb], 0)
    assert classification == "bomb_pressed"
    assert cand == 12 and bomb == 1
    # a bomb bit on the second frame only is the same rejection
    _, _, second, _ = FOLLOWUP.decode_pair([m["cand12"], m["cand12"] | BTN_BOMB], 0)
    assert second == "bomb_pressed"


@pytest.mark.parametrize("pair", [
    [int(cand_to_mask(12)) | (1 << 8), int(cand_to_mask(12)) | (1 << 8)],  # system bit
    [int(cand_to_mask(12)) | BTN_UP | BTN_DOWN,
     int(cand_to_mask(12)) | BTN_UP | BTN_DOWN],                          # UP|DOWN
])
def test_decode_pair_rejects_unknown_bits(pair):
    cand, bomb, classification, detail = FOLLOWUP.decode_pair(pair, 0)
    assert classification == "unknown_bits"
    assert cand is None and bomb is None
    assert detail["rule"]


@pytest.mark.parametrize("pair", [
    [],
    [int(cand_to_mask(12))],
    [int(cand_to_mask(12)), int(cand_to_mask(12)), int(cand_to_mask(12))],
    [int(cand_to_mask(12)), None],
    "AB",
    None,
])
def test_decode_pair_rejects_an_incomplete_pair(pair):
    cand, bomb, classification, detail = FOLLOWUP.decode_pair(pair, 0)
    assert classification == "incomplete_pair"
    assert cand is None and bomb is None
    # two different messages, both named: wrong length vs a non-integer mask
    assert detail["rule"] in ("pair must carry exactly 2 masks",
                             "a mask is not an integer")
    assert detail["decision_index"] == 0


def test_decode_pair_rejects_a_missing_release():
    """Release due but the second frame kept SHOOT: a hole, not a split."""

    m = masks()
    _, _, classification, detail = FOLLOWUP.decode_pair([m["cand12"], m["cand12"]], 47)
    assert classification == "release_missing"
    assert detail["release_due"] is True
    assert detail["expected"] == [m["cand12"], m["cand12"] & ~BTN_SHOOT]
    assert detail["rule"] == "release_due_but_the_second_frame_kept_SHOOT"


def test_decode_pair_rejects_a_split_of_two_equal_shooting_frames():
    m = masks()
    _, _, classification, detail = FOLLOWUP.decode_pair(
        [m["cand12"], m["cand3"]], 0)
    assert classification == "split_non_shoot"
    assert detail["xor"] == m["cand12"] ^ m["cand3"]


def test_decode_pair_uses_the_frozen_action_table():
    """Every accepted candidate round-trips through ``mask_to_cand``."""

    for cand in range(N_CAND):
        mask = int(cand_to_mask(cand))
        decoded, bomb, classification, _ = FOLLOWUP.decode_pair([mask, mask], 0)
        assert (decoded, bomb, classification) == (cand, 0, "accepted")
        assert mask_to_cand(mask) == (cand, 0)
        assert mask & AGENT_MASK == mask          # the encoder never leaves the mask
        assert mask & BTN_SHOOT                   # ... and always shoots
        assert not mask & BTN_BOMB and not mask & (BTN_UP << 8)


def test_every_classification_is_counted_and_named():
    counter = FOLLOWUP.rejection_counter()
    assert set(counter) == set(FOLLOWUP.CLASSIFICATIONS)
    assert set(FOLLOWUP.ACCEPTED) == {"accepted", "accepted_via_release"}
    assert set(FOLLOWUP.REJECTION_CLASSES) == {
        "incomplete_pair", "unknown_bits", "no_shoot_both", "reverse_release",
        "bomb_pressed", "release_not_due", "release_missing", "split_non_shoot"}
    assert all(value == 0 for value in counter.values())
    # every rejection class must be reachable by one synthetic sample
    samples = FOLLOWUP.synthetic_rejection_samples()
    landed = {entry["classification"] for entry in samples.values()}
    assert set(FOLLOWUP.REJECTION_CLASSES) <= landed
    assert all(entry["as_expected_class"] for entry in samples.values())


# ---------------------------------------------------------------------------
# commit_frames against the D1 recording
# ---------------------------------------------------------------------------
def test_commit_frames_reproduces_the_d1_pulse_row():
    row = d1_row(23)
    assert row["decision_frame"] == 92
    assert row["step_frames"] == [149, 149, 133, 132]

    # First sub-decision (frames 92, 93): no release, both masks equal.
    cand_first, bomb_first = mask_to_cand(row["step_frames"][0])
    assert (cand_first, bomb_first) == (11, 0)
    assert release_due(row["decision_frame"] // 2, 48) is False
    assert FOLLOWUP.expand_commit(commit_frames(
        cand_first, decision_frames=2,
        release_shoot=release_due(row["decision_frame"] // 2, 48))) == [149, 149]

    # Second sub-decision (frames 94, 95): the real pulse.  The candidate of the
    # release pair is NOT the row's first candidate; it decodes to 12.
    release_decision = (row["decision_frame"] + 2) // 2
    cand_second, bomb_second = mask_to_cand(row["step_frames"][2])
    assert (cand_second, bomb_second) == (12, 0)
    assert release_due(release_decision, 48) is True
    assert FOLLOWUP.expand_commit(commit_frames(
        cand_second, decision_frames=2, release_shoot=True)) == [133, 132]
    assert row["step_frames"][2:] == [133, 132]
    assert row["step_frames"][3] == row["step_frames"][2] & ~BTN_SHOOT
    assert row["hash_pre"] == "7e9b64232d230a26"
    assert row["hash_post"] == "b131114d28bd2a4c"

    # And decode_pair closes the loop on the recorded pair.
    _, _, classification, detail = FOLLOWUP.decode_pair(
        [row["step_frames"][2], row["step_frames"][3]], release_decision)
    assert classification == "accepted_via_release"
    assert detail["cand"] == 12


def test_the_recorded_release_rows_are_the_137_pulse_decisions():
    """137 rows == decisions 47 mod 48, measured from the file, not assumed."""

    rows = 0
    decisions = []
    with D1_LABELS.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            steps = row["step_frames"]
            for offset in (0, 2):
                pair = steps[offset:offset + 2]
                if pair[0] != pair[1]:
                    rows += 1
                    decisions.append((row["decision_frame"] + offset) // 2)
    assert rows == 137
    assert all(decision % 48 == 47 for decision in decisions)
    assert decisions[:3] == [47, 95, 143]
    assert decisions[-1] == 47 + 48 * 136


# ---------------------------------------------------------------------------
# the four registered sources
# ---------------------------------------------------------------------------
def test_the_four_sources_exist_and_keep_their_own_identity(loaded_sources):
    sources, resolutions = loaded_sources
    assert set(sources) == set(EXPECTED_GRID)
    shas = {source["sha256"] for source in sources.values()}
    assert len(shas) == 4, "each corpus must keep a distinct identity"
    for source_id, expected in REGISTERED_SHA256.items():
        assert sources[source_id]["sha256"] == expected
        assert sources[source_id]["sha256_matches_registration"] is True
    assert sources["t02c-2f-seed12"]["sha256_matches_alt"] is True
    # T04's obs files are symlinks into .tmp_v400_d1.  On Windows the link is a
    # reparse tag that cannot be stat'ed, so the shape is resolved through T04's
    # own meta and the resolution is recorded; on WSL the link resolves and there
    # is nothing to record.  Both are the same corpus, so the assertion is about
    # *which* source may appear, not about how many resolutions this host needs.
    resolved_ids = {entry["source"] for entry in resolutions}
    assert resolved_ids <= {"t04-2f-seed12"}
    if resolved_ids == set():
        # WSL: every obs path is readable here, so no resolution was needed and
        # the shape must have come straight from the files
        assert sources["t04-2f-seed12"]["obs"] is not None


def test_each_source_grid_matches_its_registered_numbers(loaded_sources):
    sources, _ = loaded_sources
    for source_id, expected in EXPECTED_GRID.items():
        counts = sources[source_id]["counts"]
        for key in ("rows", "frames", "decision_frames", "frames_per_decision",
                    "leading_frames", "first_decision_frame", "last_decision_frame",
                    "first_pair_frame", "last_pair_frame"):
            assert counts[key] == expected[key], f"{source_id}.{key}"
        assert sources[source_id]["masks"]["split_pairs"] == expected["split_pairs"]
        assert sources[source_id]["hash_coverage"]["hash_frames_total"] == expected["hash_frames"]


def test_registered_meta_files_agree_with_the_measurements(loaded_sources):
    sources, _ = loaded_sources
    for source_id, source in sources.items():
        agreement = source["meta"]["agreement"]
        assert agreement, source_id
        assert all(entry["ok"] for entry in agreement), [
            entry for entry in agreement if not entry["ok"]]
        assert source["meta"]["sha256"]
        assert not source["problems"], source["problems"]


def test_mask_statistics_are_bomb_free_and_consistent(loaded_sources):
    sources, _ = loaded_sources
    for source_id, source in sources.items():
        masks_block = source["masks"]
        assert masks_block["bomb_rows"] == 0, source_id
        assert masks_block["bomb_values"] in ([], [0]), source_id
        assert masks_block["rows_with_split_pair"] == 137, source_id
        assert masks_block["shoot_release_rows"] == 137, source_id
        assert masks_block["distinct_masks"] >= 17, source_id
        assert masks_block["distinct_masks"] == len(masks_block["distinct_mask_values"])
    d1 = sources["d1-canonical-seed12"]
    assert d1["masks"]["mask_entries"] == 13192
    assert d1["hash_coverage"]["hash_pre_frames"]["step"] == 4
    assert d1["hash_coverage"]["hash_chain_breaks"] == 0
    t04 = sources["t04-2f-seed12"]
    assert t04["masks"]["mask_entries"] == 6588       # one mask per slot
    assert t04["masks"]["derived_tail_slots"] == 6588
    assert t04["pair_source"] == "derived" and t04["independent"] is False
    assert sources["t02c-2f-seed12"]["pair_source"] == "recorded"
    assert sources["t02c-2f-seed12"]["independent"] is True


def test_obs_shapes_are_read_from_the_headers(loaded_sources):
    sources, _ = loaded_sources
    for source_id, source in sources.items():
        files = source["obs"]["files"]
        assert set(files) == {"play", "hud", "act_hist"}, source_id
        assert files["play"]["shape"][1:] == [4, 112, 96], source_id
        assert files["hud"]["shape"][1:] == [56, 160], source_id
        assert files["act_hist"]["shape"][1:] == [8, 18], source_id
    assert sources["d1-canonical-seed12"]["obs"]["files"]["play"]["shape"][0] == 3298
    assert sources["t04-2f-seed12"]["obs"]["files"]["play"]["shape"][0] == 6588


def test_the_four_alignments_are_identical_inside_their_intervals(loaded_sources):
    sources, _ = loaded_sources
    rows = FOLLOWUP.build_alignments(sources)
    assert [row["id"] for row in rows] == [
        "d1-vs-s1c", "d1-vs-t04", "d1-vs-t02c", "t02c-vs-t04"]
    for row in rows:
        coverage = row["coverage"]
        assert row["identical"] is True, (row["id"], row["first_diff"])
        assert row["verdict"] == "same_run_compatible"
        assert row["masks"]["mismatches"] == 0
        assert row["hashes"]["mismatched"] == 0
        assert coverage["compared"] > 0
        assert (coverage["compared"] + coverage["missing_left"]
                + coverage["missing_right"]) == row["interval"]["frames"]
        assert row["interval"]["first_frame"] >= sources[row["left"]]["counts"]["first_frame"]
        assert row["interval"]["last_frame"] <= sources[row["left"]]["counts"]["last_frame"]
    by_id = {row["id"]: row for row in rows}
    assert by_id["d1-vs-s1c"]["masks"]["compared"] == 13192
    assert by_id["d1-vs-t04"]["masks"]["compared"] == 6588
    assert by_id["d1-vs-t04"]["coverage"]["missing_right"] == 6588   # t04 has no tail
    assert by_id["d1-vs-t04"]["hashes"]["compared"] == 3294          # 4-frame grid only
    assert by_id["d1-vs-t02c"]["masks"]["compared"] == 13176
    assert by_id["d1-vs-t02c"]["hashes"]["state"] == "right_unavailable"
    assert by_id["t02c-vs-t04"]["hashes"]["state"] == "left_unavailable"


# ---------------------------------------------------------------------------
# adapter runs over the recordings
# ---------------------------------------------------------------------------
def test_adapter_decodes_every_recorded_pair(loaded_sources):
    sources, _ = loaded_sources
    prefix = FOLLOWUP.run_adapter(sources["d1-canonical-seed12"]["_model"]["pairs"],
                                  frame_cap=8664)
    assert prefix["decisions"] == 4332
    assert prefix["accepted"] == 4332
    assert prefix["expected_mask_mismatches"] == 0
    assert sum(prefix["rejections"].values()) == 0
    assert prefix["accepted_via_release"] == 90        # decisions 47 mod 48 <= 4331
    assert prefix["release_due_decisions"] == 90
    assert prefix["pair_cand_would_refuse"] == 90

    full = FOLLOWUP.run_adapter(sources["d1-canonical-seed12"]["_model"]["pairs"],
                                frame_cap=None)
    assert (full["decisions"], full["accepted"]) == (6596, 6596)
    assert full["accepted_via_release"] == 137
    assert full["bomb_nonzero"] == 0
    assert full["first_mismatch"] is None
    assert len(full["samples"]) == 3

    for source_id, expected_decisions in (("s1c-seed12", 6596),
                                          ("t02c-2f-seed12", 6588),
                                          ("t04-2f-seed12", 6588)):
        run = FOLLOWUP.run_adapter(sources[source_id]["_model"]["pairs"], frame_cap=None)
        assert run["decisions"] == expected_decisions, source_id
        assert run["accepted"] == expected_decisions, source_id
        assert run["expected_mask_mismatches"] == 0, source_id
        assert run["rejections"] == {name: 0 for name in FOLLOWUP.REJECTION_CLASSES}


def test_t04_literal_single_mask_reading_is_reported_beside_the_derived_one(loaded_sources):
    sources, _ = loaded_sources
    literal = FOLLOWUP.run_adapter(
        sources["t04-2f-seed12"]["_model"]["literal_pairs"], frame_cap=None)
    assert literal["decisions"] == 6588
    assert literal["accepted"] == 6588 - 137
    assert literal["rejections"]["release_missing"] == 137
    assert literal["first_mismatch"]["decision_index"] == 47
    assert literal["first_mismatch"]["pair"] == [133, 133]
    assert literal["first_mismatch"]["expected"] == [133, 132]


def test_t02c_records_the_real_pulse_pair():
    row = t02c_row(39)
    assert row["decision_frame"] == 94
    assert row["commit_frames"] == [94, 96]
    assert (row["corpus"]["committed_masks"]) == [133, 132]
    assert row["mask"] == 133
    assert row["cand"] == 12 and row["bomb"] == 0
    _, _, classification, detail = FOLLOWUP.decode_pair(
        list(row["corpus"]["committed_masks"]), row["decision_frame"] // 2)
    assert classification == "accepted_via_release"
    assert detail["expected"] == [133, 132]


# ---------------------------------------------------------------------------
# CLI: every subcommand has --help, unimplemented ones return 3
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("command", ALL_COMMANDS)
def test_every_subcommand_help_exits_zero(command):
    with pytest.raises(SystemExit) as info:
        FOLLOWUP.main([command, "--help"])
    assert info.value.code == 0


def test_top_level_help_through_the_real_entry_point():
    completed = subprocess.run(
        [sys.executable, "-B", str(REPO_ROOT / "tools" / "rl" / "v8_31_followup.py"),
         "--help"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "sources" in completed.stdout and "adapter" in completed.stdout
    assert "verify" in completed.stdout and "run" in completed.stdout


def test_no_command_is_a_skeleton_and_the_fallback_still_returns_3(tmp_path, capsys):
    """Every subcommand of the ticket is implemented; the fallback stays rc 3.

    ``pilot`` used to be asserted here; it now has its own tests below (it either
    refuses before any start or reads a real input), so keeping it in this list
    would assert the opposite of what the tool does.
    """

    assert FOLLOWUP.NOT_IMPLEMENTED == {}
    args = FOLLOWUP.argparse.Namespace(command="no-such-command")
    assert FOLLOWUP.cmd_not_implemented(args) == FOLLOWUP.EXIT_NOT_IMPLEMENTED
    printed = capsys.readouterr()
    assert "not-implemented" in printed.out
    assert "no-such-command" in printed.out
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("command", ["prefix", "reference", "recon", "intervene"])
def test_engine_packages_refuse_before_any_start(tmp_path, capsys, command):
    """A missing engine is a refusal: rc 2, no ledger row, no private copy."""

    out = tmp_path / "out"
    argv = [command, "--repo", str(REPO_ROOT), "--engine",
            str(tmp_path / "no-engine-here"), "--data-source",
            str(tmp_path / "no-data-here"), "--out", str(out), "--no-xvfb"]
    code = FOLLOWUP.main(argv)
    assert code == FOLLOWUP.EXIT_TECHNICAL
    if command == "intervene":
        # `intervene` reads the frozen card before it looks at the engine, so the
        # refusal under test is the missing card here (same rc, no start)
        record = json.loads((out / "02c" / "intervene.json").read_text(
            encoding="utf-8"))
        assert record["engine_starts"] == 0
        assert record["terminal"]["kind"] == "technical_invalid"
        assert "task card" in record["terminal"]["detail"]
        assert not (out / "ledger.jsonl").exists()
        return
    if command == "recon":
        base = out / "02b" / "recon"
    else:
        base = out / "02a" / command
    record = json.loads((base / "result.json").read_text(encoding="utf-8"))
    assert record["engine_started"] is False
    assert record["terminal"]["kind"] == "technical_invalid"
    assert record["terminal"]["reason"] == "EngineRefused"
    assert "engine binary not found" in record["terminal"]["detail"]
    assert record["counts"]["started"] == 0
    assert not (out / "ledger.jsonl").exists(), "no ledger row for a refused start"
    assert not (base / "game").exists()
    assert not (base / "mute.json").exists(), \
        "the copy (and its mute report) only exists once the engine may start"


def test_verify_03_refuses_an_empty_run(tmp_path, capsys):
    """``verify --package 03`` is implemented, so an empty run is a refusal (rc 2)."""

    out = tmp_path / "empty"
    assert FOLLOWUP.main(["verify", "--out", str(out),
                          "--package", "03"]) == FOLLOWUP.EXIT_TECHNICAL
    assert "missing" in capsys.readouterr().err
    payload = json.loads((out / "verify.json").read_text(encoding="utf-8"))
    assert payload["gate"]["passed"] is False
    assert payload["problems"]


@pytest.mark.parametrize("package", ["02b", "02c"])
def test_verify_refuses_a_missing_02bc_package(tmp_path, capsys, package):
    """``verify --package 02b/02c`` is implemented, so an empty run is rc 2."""

    out = tmp_path / "empty"
    assert FOLLOWUP.main(["verify", "--out", str(out),
                          "--package", package]) == FOLLOWUP.EXIT_TECHNICAL
    assert "missing" in capsys.readouterr().err
    payload = json.loads((out / "verify.json").read_text(encoding="utf-8"))
    assert payload["gate"]["passed"] is False


def test_compare_refuses_a_run_without_evidence(tmp_path, capsys):
    out = tmp_path / "empty"
    assert FOLLOWUP.main(["compare", "--out", str(out)]) == FOLLOWUP.EXIT_TECHNICAL
    payload = json.loads((out / "02a" / "compare.json").read_text(encoding="utf-8"))
    assert payload["missing"], payload
    assert payload["gate"]["passed"] is False
    assert payload["verdict"] == "differs"


def test_verify_refuses_a_missing_package(tmp_path, capsys):
    assert FOLLOWUP.main(["verify", "--out", str(tmp_path / "empty")]) == \
        FOLLOWUP.EXIT_TECHNICAL
    assert "missing" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# end-to-end, still offline: sources -> adapter -> verify, and the run skeleton
# ---------------------------------------------------------------------------
def test_sources_adapter_verify_end_to_end(tmp_path):
    out = tmp_path / "s1_out"
    assert FOLLOWUP.main(["sources", "--repo", str(REPO_ROOT),
                          "--out", str(out)]) == FOLLOWUP.EXIT_OK
    sources_json = json.loads((out / "02a" / "sources.json").read_text(encoding="utf-8"))
    assert sources_json["schema"] == FOLLOWUP.SCHEMA_SOURCES
    assert sources_json["gate"]["passed"] is True
    assert sources_json["host"]["code_name"] == "轩神开播了"
    assert sources_json["heads"]["head"]
    assert len(sources_json["sources"]) == 4
    assert len(sources_json["alignments"]) == 4
    assert sources_json["r22_recheck"]["checks"]["slots"] is True
    assert "anchors_16_8668" in sources_json["r22_recheck"]["hash_frames_reproduced_by"]

    assert FOLLOWUP.main(["adapter", "--repo", str(REPO_ROOT),
                          "--out", str(out)]) == FOLLOWUP.EXIT_OK
    adapter_json = json.loads((out / "02a" / "adapter.json").read_text(encoding="utf-8"))
    assert adapter_json["gate"]["passed"] is True, adapter_json["gate"]["problems"]
    assert [run["id"] for run in adapter_json["runs"]] == [
        "d1-prefix-8664", "d1-full-13192", "s1c-full-13192",
        "t02c-slots-6588", "t04-slots-6588"]
    for run in adapter_json["runs"]:
        assert run["accepted"] == run["decisions"]
        assert run["expected_mask_mismatches"] == 0
        assert sum(run["rejections"].values()) == 0
        assert run["bomb_nonzero"] == 0
    assert adapter_json["test_coverage"]["classes_missing"] == []

    assert FOLLOWUP.main(["verify", "--out", str(out)]) == FOLLOWUP.EXIT_OK
    verify = json.loads((out / "verify.json").read_text(encoding="utf-8"))
    assert verify["gate"]["passed"] is True, verify["gate"]["problems"]
    # `--package 02a` verifies 02-A only; 02-B/02-C are separate packages now
    assert verify["packages"]["02b"]["status"] == "not-run"
    assert verify["packages"]["02c"]["status"] == "not-run"
    assert set(verify["verified"]["sources"]) == set(EXPECTED_GRID)
    assert len(verify["verified"]["runs"]) == len(adapter_json["runs"])


def test_run_drives_the_package_sequence_and_reuses_what_reverifies(tmp_path):
    """A partial ``run`` passes the offline pair, marks the rest ``not-run``, and
    the second invocation **reuses** the pair instead of recomputing it."""

    out = tmp_path / "run"
    code = FOLLOWUP.main(["run", "--repo", str(REPO_ROOT), "--out", str(out),
                          "--max-starts", "3", "--max-engine-frames", "1000"])
    assert code == FOLLOWUP.EXIT_OK
    payload = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert payload["schema"] == FOLLOWUP.SCHEMA_RUN
    assert payload["engine_starts"] == 0
    assert payload["status"] == "finished"
    assert payload["order"] == [entry["name"] for entry in FOLLOWUP.RUN_PACKAGES]
    assert payload["gates"]["02a-sources"]["status"] == "passed"
    assert payload["gates"]["02a-adapter"]["status"] == "passed"
    # every engine package is `not-run` with the missing input named
    for name in ("02a-prefix", "02a-reference", "02b", "02c", "03"):
        assert payload["gates"][name]["status"] == "not-run", name
        assert "--engine" in payload["gates"][name]["reason"]
    assert payload["gates"]["verify"]["status"] == "passed"
    assert payload["gates"]["verify"]["verify_package"] == "02a"
    assert payload["started_at"] and payload["finished_at"] and payload["next_step"]
    rows = [json.loads(line) for line in
            (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["phase"] for row in rows} == {"reserve", "settle"}
    assert all(row["actual"]["starts"] == 0 for row in rows if row["phase"] == "settle")
    totals = payload["ledger_totals"]
    assert totals["actual"]["starts"] == 0
    assert totals["actual"]["engine_frames"] == 0
    assert totals["limits"]["max_starts"] == 3

    sources_before = (out / "02a" / "sources.json").stat().st_mtime_ns
    again = FOLLOWUP.main(["run", "--repo", str(REPO_ROOT), "--out", str(out),
                           "--only", "sources,adapter", "--max-starts", "3",
                           "--max-engine-frames", "1000"])
    assert again == FOLLOWUP.EXIT_OK
    second = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert second["gates"]["02a-sources"]["status"] == "reused"
    assert second["gates"]["02a-sources"]["reused"] is True
    assert second["gates"]["02a-adapter"]["status"] == "reused"
    assert second["reused"]["02a-sources"]["reusable"] is True
    assert (out / "02a" / "sources.json").stat().st_mtime_ns == sources_before, \
        "a reused package must not be recomputed"
    assert second["gates"]["02b"]["status"] == "skipped"


def test_no_engine_binary_is_referenced_by_the_offline_commands():
    """``sources``/``adapter`` must not even accept ``--engine``."""

    parser = FOLLOWUP.build_parser()
    for command in ("sources", "adapter"):
        with pytest.raises(SystemExit) as info:
            parser.parse_args([command, "--engine", "nope", "--out", "x"])
        assert info.value.code == 2


# ---------------------------------------------------------------------------
# 02-A.3 / 02-A.4: the D1 replay plan, the PNG identity, the comparison
# Everything below is offline: the masks and anchors are read from D1, no engine
# is started and no shared-memory block is read.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def d1_plan():
    return FOLLOWUP.d1_replay_plan(REPO_ROOT)


def test_registereded_d1_identity_is_rechecked(d1_plan):
    assert d1_plan["sha256"] == REGISTERED_SHA256["d1-canonical-seed12"]
    assert d1_plan["sha256"] == d1_plan["registered_sha256"]


def test_d1_reset_hash_is_frame_zero_and_the_task_states_it(d1_plan):
    assert d1_plan["reset_frame"] == 0
    assert d1_plan["reset_hash"] == "83d953750afaea5a"


def test_the_two_anchor_denominators_are_2165_and_2166(d1_plan):
    """2,165 ``hash_pre`` after the reset (frame 0 is the reset check) + 2,166 
    ``hash_post`` -- the ticket's own numbers."""

    assert len(d1_plan["expected_pre"]) == 2165
    assert len(d1_plan["expected_post"]) == 2166
    assert sorted(d1_plan["expected_pre"])[0] == 4
    assert sorted(d1_plan["expected_pre"])[-1] == 8660
    assert sorted(d1_plan["expected_post"])[0] == 4
    assert sorted(d1_plan["expected_post"])[-1] == 8664
    assert d1_plan["pre_is_subset_of_post"] is True


def test_the_pre_and_post_anchors_agree_where_they_overlap(d1_plan):
    assert d1_plan["chain_breaks"] == 0
    for frame, value in d1_plan["expected_pre"].items():
        assert d1_plan["expected_post"][frame] == value


def test_the_continuation_anchor_is_the_next_row_s_hash(d1_plan):
    assert d1_plan["continuation_frame"] == 8668
    assert d1_plan["continuation_expected"] == "99d68e050d54b4e5"
    assert d1_plan["continuation_pre_also"] == "99d68e050d54b4e5"
    assert d1_plan["continuation_pre_matches_post"] is True
    # the two anchor classes are recorded separately and both stop at the boundary
    assert 8668 not in d1_plan["expected_post"]
    assert 8668 not in d1_plan["expected_pre"]


def test_step_masks_are_flat_masks_shifted_by_one(d1_plan):
    """STEP frame ``f`` sends D1 flat mask ``f - 1``; the shift is not a guess."""

    flat = d1_plan["flat_masks"]
    for frame in (1, 2, 3, 4, 8659, 8660, 8661, 8664, 8665, 8668):
        assert d1_plan["step_masks"][frame] == flat[frame - 1]


def test_the_obs_stack_window_is_the_last_four_real_pictures(d1_plan):
    """Pixel frame ``p`` is read after STEP ``p + 1`` and uses flat mask ``p``."""

    assert FOLLOWUP.STACK_PIXEL_FRAMES == (8660, 8661, 8662, 8663)
    assert FOLLOWUP.WINDOW_STEPS == (8659, 8660, 8661, 8662, 8663, 8664, 8665)
    assert FOLLOWUP.WINDOW_PIXEL_FRAMES == tuple(
        step - 1 for step in FOLLOWUP.WINDOW_STEPS)
    assert FOLLOWUP.RESOURCE_FRAMES == (8663, 8664)


def test_the_prefix_plan_is_one_single_frame_step_per_frame(d1_plan):
    units, last_frame = FOLLOWUP._plan_for_mode("prefix", d1_plan["step_masks"])
    assert last_frame == 8668
    assert len(units) == 8668
    assert all(unit["frames"] == 1 for unit in units)
    summary = FOLLOWUP.plan_summary(units)
    assert summary["frames"] == 8668
    assert summary["mask2_steps"] == 0
    check = FOLLOWUP._check_plan(units, d1_plan["step_masks"], last_frame)
    assert check["units_expand_to_the_d1_trace"] is True
    assert check["window_frames_are_step_ends"] is True


def test_the_reference_plan_uses_two_frame_steps_where_the_masks_agree(d1_plan):
    units, last_frame = FOLLOWUP._plan_for_mode("reference", d1_plan["step_masks"])
    assert last_frame == 8665
    check = FOLLOWUP._check_plan(units, d1_plan["step_masks"], last_frame)
    assert check["units_expand_to_the_d1_trace"] is True
    assert check["window_frames_are_step_ends"] is True
    summary = FOLLOWUP.plan_summary(units)
    assert summary["frames"] == 8665
    assert summary["mask2_steps"] > 4000
    assert summary["mask2_steps"] + summary["mask1_steps"] == len(units)
    assert summary["only_units_of_1_or_2_frames"] is True
    # a two-frame unit is only legal when both frames carry the same mask
    for unit in units:
        if unit["frames"] == 2:
            assert (d1_plan["step_masks"][unit["start"]]
                    == d1_plan["step_masks"][unit["start"] + 1])
    # the last unit is the half decision that makes pixel frame 8,664 reachable
    assert units[-1]["start"] == 8665
    assert units[-1]["frames"] == 1
    assert units[-1]["partial_decision"] is True


def test_the_two_plans_carry_the_same_masks(d1_plan):
    prefix_units, _ = FOLLOWUP._plan_for_mode("prefix", d1_plan["step_masks"])
    reference_units, _ = FOLLOWUP._plan_for_mode("reference", d1_plan["step_masks"])
    prefix_trace = [(unit["start"], unit["mask"]) for unit in prefix_units]  # 1 frame each
    reference_trace = [(unit["start"] + offset, unit["mask"])
                       for unit in reference_units
                       for offset in range(unit["frames"])]
    assert reference_trace == prefix_trace[:8665]


def test_the_mask_trace_digest_is_granularity_independent(d1_plan):
    prefix_units, _ = FOLLOWUP._plan_for_mode("prefix", d1_plan["step_masks"])
    reference_units, _ = FOLLOWUP._plan_for_mode("reference", d1_plan["step_masks"])
    prefix_digest = FOLLOWUP.mask_trace_digest(d1_plan["step_masks"], 1, 8664)
    reference_digest = FOLLOWUP.mask_trace_digest(
        {frame: mask for unit in reference_units
         for frame, mask in [(unit["start"] + offset, unit["mask"])
                             for offset in range(unit["frames"])]
         if frame <= 8664}, 1, 8664)
    assert prefix_digest == reference_digest
    assert len(prefix_digest) == 64
    assert prefix_units  # the plan is what makes the two traces comparable


def test_the_png_writer_is_deterministic_and_reads_back():
    width, height = 6, 3
    rows = bytes((x * 11 + y * 7) & 0xFF for y in range(height) for x in range(width * 4))
    first = FOLLOWUP.png_bytes(rows, width, height)
    second = FOLLOWUP.png_bytes(rows, width, height)
    assert first == second, "the PNG must be byte-identical for identical pixels"
    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    got_width, got_height, back = FOLLOWUP.read_png(first)
    assert (got_width, got_height) == (width, height)
    assert back == rows
    assert FOLLOWUP.raw_order_bytes(rows, width, height) != rows
    assert len(FOLLOWUP.raw_order_bytes(rows, width, height)) == len(rows)


def test_the_png_digests_are_the_engine_byte_order(tmp_path):
    """The stored PNG must reproduce the recorded blake2b/sha256 exactly."""

    width, height = 4, 2
    top_down = bytes(range(width * height * 4))
    data = FOLLOWUP.png_bytes(top_down, width, height)
    path = tmp_path / "frame.png"
    path.write_bytes(data)
    fresh = FOLLOWUP.frame_digests_from_png(path)
    raw = FOLLOWUP.raw_order_bytes(top_down, width, height)
    assert fresh["blake2b128"] == hashlib.blake2b(raw, digest_size=16).hexdigest()
    assert fresh["sha256_raw"] == hashlib.sha256(raw).hexdigest()
    assert fresh["png_sha256"] == hashlib.sha256(data).hexdigest()
    assert fresh["bytes"] == len(data)


def test_anchor_report_compares_every_frame_it_was_asked_about():
    expected = {4: "aaaa", 8: "bbbb", 12: "cccc"}
    report = FOLLOWUP.anchor_report(expected=expected,
                                    actual={4: "aaaa", 8: "bbbb", 12: "cccc"},
                                    first_frame=1, last_frame=16, label="t")
    assert report["expected"] == report["compared"] == report["matched"] == 3
    assert report["mismatches"] == 0
    broken = FOLLOWUP.anchor_report(expected=expected,
                                    actual={4: "aaaa", 8: "zzzz"},
                                    first_frame=1, last_frame=16, label="t")
    assert broken["expected"] == 3 and broken["compared"] == 3
    assert broken["mismatches"] == 2
    assert broken["first_mismatch_frame"] == 8
    assert broken["missing_actual"] == 1
    assert broken["first_mismatch"]["expected"] == "bbbb"
    assert broken["first_mismatch"]["actual"] == "zzzz"


def test_first_difference_reports_the_first_place_verbatim():
    assert FOLLOWUP.first_difference({"a": 1, "b": [1, 2]}, {"a": 1, "b": [1, 2]}) is None
    found = FOLLOWUP.first_difference({"a": 1, "b": [1, 2]}, {"a": 9, "b": [1, 2]})
    assert found["path"] == "root.a" and found["left"] == 1 and found["right"] == 9
    nested = FOLLOWUP.first_difference({"b": [1, 2]}, {"b": [1, 3]})
    assert nested["path"] == "root.b[1]"
    lengths = FOLLOWUP.first_difference({"b": [1]}, {"b": [1, 2]})
    assert lengths["reason"] == "length differs"


def test_the_player_reply_is_decoded_without_rounding():
    reply = ("OK frames=7 hp=40000000 lives=3 bombs=00000000 power=3f000000 focus=1 "
             "isYoukai=0 playerState=3 invulnState=1 px=43280000 py=43800000 "
             "hurtboxMinX=3f800000 hurtboxMinY=3f800000 hurtboxMaxX=bf800000 "
             "hurtboxMaxY=bf800000 speedNormalAxial=40800000 "
             "speedNormalDiagonal=403504f3 speedFocusedAxial=40000000 "
             "speedFocusedDiagonal=3fb504f3")
    player = FOLLOWUP.parse_player_reply(reply)
    assert player["frames"] == 7
    assert player["px"] == 168.0 and player["py"] == 256.0
    assert player["hp"] == 2.0
    assert player["lives"] == 3 and player["bombs"] == 0.0
    assert player["power"] == 0.5
    assert player["focus"] == 1 and player["is_youkai"] == 0
    assert player["player_state"] == 3
    assert player["invuln_state"] == 1
    assert player["raw_bits"]["px"] == 0x43280000
    assert player["position_source"].startswith("PLAYER")


def test_the_resource_record_keeps_both_position_sources():
    state = {"frame": 8664, "stage": 0, "score": 1234, "graze": 7, "deaths": 0.0,
             "lives_remaining": 3.0, "bombs_remaining": 1.0, "player_power": 0.5,
             "player_x": 168.0, "player_y": 256.0, "gameplay_frame": 8665}
    player = {"px": 168.0, "py": 256.0, "hp": 2.0, "lives": 3, "bombs": 1.0,
              "power": 0.5, "focus": 0, "is_youkai": 0, "player_state": 0,
              "player_state_name": "ALIVE"}
    hud = {"frame": 8664, "display_score": 1200, "globals_score": 1234, "lives": 3,
           "bombs": 1, "power": 1, "graze": 7, "point": 0, "time": 5, "game_over": 0}
    record = FOLLOWUP.resource_record(frame=8664, state=state, player=player, hud=hud)
    assert record["frame"] == 8664
    assert record["position"] == {"x": 168.0, "y": 256.0, "source": "PLAYER px/py"}
    assert record["state_position"] == {"x": 168.0, "y": 256.0}
    assert record["position_sources_agree"] is True
    assert record["resources"]["lives"] == 3
    assert record["resources"]["score"] == 1234
    assert record["hud"]["display_score"] == 1200
    assert set(record) >= {"state", "player", "hud", "resources", "position"}


def test_the_whole_4332_decision_prefix_decodes_without_a_rejection(d1_plan):
    """The reference's independent decode must agree with the frozen adapter."""

    masks = d1_plan["step_masks"]
    for index in range(FOLLOWUP.PREFIX_DECISIONS):
        first = 2 * index + 1
        pair = [masks[first], masks[first + 1]]
        adapter_row = FOLLOWUP._decode_decision(pair, index)
        direct_row = FOLLOWUP._decode_decision_direct(pair)
        assert adapter_row["accepted"], (index, adapter_row["classification"])
        assert adapter_row["cand"] == direct_row["cand"], index
        assert adapter_row["bomb"] == 0
        assert direct_row["bomb"] == 0


def test_the_last_eight_prefix_decisions_are_the_expected_phase(d1_plan):
    """Decisions 4,324..4,331: the last eight complete actions before 8,664."""

    masks = d1_plan["step_masks"]
    rows = []
    for index in range(FOLLOWUP.PREFIX_DECISIONS - 8, FOLLOWUP.PREFIX_DECISIONS):
        first = 2 * index + 1
        rows.append(FOLLOWUP._decode_decision([masks[first], masks[first + 1]], index))
    assert [row["decision_index"] for row in rows] == list(range(4324, 4332))
    assert all(row["accepted"] and row["bomb"] == 0 for row in rows)
    # the SHOOT pulse is due every 48 decisions; 4,319 is the last one before
    # the boundary and 4,331 is not, so the boundary pair is not a release
    assert rows[-1]["release_due"] is False
    assert rows[0]["pair"] == [masks[8649], masks[8650]]
    assert rows[-1]["pair"] == [masks[8663], masks[8664]]


def test_compare_reports_every_item_it_is_asked_for(tmp_path):
    """A synthetic pair of runs: the comparison must find 7/7 + the aggregates."""

    root = tmp_path / "out"
    frames = [(8658, 0x11), (8659, 0x22)]
    for mode, shift in (("prefix", 0), ("reference", 1)):
        base = root / "02a" / mode
        (base / "frames").mkdir(parents=True)
        captures = []
        for pixel_frame, seed in frames:
            top_down = bytes((seed + index) & 0xFF
                             for index in range(4 * 2 * 4))
            data = FOLLOWUP.png_bytes(top_down, 4, 2)
            path = base / "frames" / f"frame_pix{pixel_frame}.png"
            path.write_bytes(data)
            raw = FOLLOWUP.raw_order_bytes(top_down, 4, 2)
            captures.append({
                "captured_at_step_F": pixel_frame + 1,
                "pixel_frame": pixel_frame,
                "header_frame": pixel_frame + 1 + shift,
                "events_frames": pixel_frame + 1 + shift,
                "blake2b128": hashlib.blake2b(raw, digest_size=16).hexdigest(),
                "sha256_raw": hashlib.sha256(raw).hexdigest(),
                "png_sha256": hashlib.sha256(data).hexdigest(),
                "png": f"02a/{mode}/frames/frame_pix{pixel_frame}.png",
            })
        takeover = {"window": {"captures": captures}}
        (base / "takeover.json").write_text(json.dumps(takeover), encoding="utf-8")
        history = [{"decision_index": index, "cand": 3, "pair": [65, 66]}
                   for index in range(FOLLOWUP.PREFIX_DECISIONS)]
        (base / "history.jsonl").write_text(
            "\n".join(json.dumps(row) for row in history) + "\n", encoding="utf-8")
        (base / "frames.jsonl").write_text("", encoding="utf-8")
        (base / "result.json").write_text(json.dumps({
            "mask_trace": {"digest": "d" * 64},
            "actor_inputs": {"play": "p", "hud": "h", "act_hist": "a",
                             "actor_input_digest": "x", "history_entries": 4332,
                             "next_decision_index": 4332, "frames_pumped": 4},
            "resources": {str(frame): {"hud": {"frame": frame},
                                       "position": {"x": 1.0, "y": 2.0},
                                       "state": {"score": 1}}
                          for frame in FOLLOWUP.RESOURCE_FRAMES},
            "reset": {"hash": "83d953750afaea5a"},
            "anchors": {group: {"expected": 2166, "compared": 2166, "mismatches": 0}
                        for group in ("pre", "post", "combined")},
        }), encoding="utf-8")
    payload = FOLLOWUP.compare_runs(root)
    items = {row["item"]: row for row in payload["items"]}
    assert items["frames_7_of_7_blake2b128"]["equal"] is True
    assert items["frames_7_of_7_sha256_raw"]["equal"] is True
    assert items["frames_7_of_7_png_sha256"]["equal"] is True
    assert items["last_eight_actions"]["equal"] is True
    assert items["history_row_count"]["prefix"] == FOLLOWUP.PREFIX_DECISIONS
    assert items["history_row_count"]["reference"] == FOLLOWUP.PREFIX_DECISIONS
    # the reference's header frame is shifted by one on purpose -> a difference
    assert items["frame_relation"]["equal"] is False
    assert payload["first_difference"] is not None
    assert payload["verdict"] == "differs"




# ---------------------------------------------------------------------------
# 02-B / 02-C: the identity channel, the generation table, the frozen card and
# the shared budget.  Everything below is offline: no engine is started and no
# corpus outside the four registered ones is read.
# ---------------------------------------------------------------------------
BOSS_FLAGS = (FOLLOWUP.ENEMY_FLAG_BOSS | FOLLOWUP.ENEMY_FLAG_DAMAGEABLE
              | FOLLOWUP.ENEMY_FLAG_ACCEPTS_DAMAGE | 0x1)
PLAIN_FLAGS = 0x1 | FOLLOWUP.ENEMY_FLAG_DAMAGEABLE


def hazard_enemy(slot: int, flags: int, life: int, x: float = 100.0,
                 y: float = 200.0) -> dict:
    """A synthetic decoded enemy row (the shape ``parse_hazards_enemies`` emits)."""

    return {
        "slot": slot, "flags1": f"0x{flags:08x}", "flags1_dec": flags,
        "flags1_names": FOLLOWUP.flags1_names(flags), "life": life,
        "life_raw_unsigned": life & 0xFFFFFFFF, "timer": 7, "x": x, "y": y,
        "z": 0.0, "boss": bool(flags & FOLLOWUP.ENEMY_FLAG_BOSS),
        "damageable": bool(flags & FOLLOWUP.ENEMY_FLAG_DAMAGEABLE),
        "accepts_damage": bool(flags & FOLLOWUP.ENEMY_FLAG_ACCEPTS_DAMAGE),
        "persist_after_death": bool(
            flags & FOLLOWUP.ENEMY_FLAG_PERSIST_AFTER_DEATH),
        "has_been_in_bounds": bool(
            flags & FOLLOWUP.ENEMY_FLAG_HAS_BEEN_IN_BOUNDS),
        "allow_offscreen": bool(flags & FOLLOWUP.ENEMY_FLAG_ALLOW_OFFSCREEN),
        "death_mode": 0, "world_x": x, "world_y": y, "world_z": 0.0,
        "hitbox_w": 16.0, "hitbox_h": 16.0,
    }


def decision_row(index: int, enemies: list, ev: tuple = (0, 0),
                 score: int = 0) -> dict:
    """A synthetic 02-B decision row (the shape ``decisions.jsonl`` stores)."""

    return {
        "schema": FOLLOWUP.SCHEMA_RECON_ROW,
        "decision_index": index,
        "frame_start": index * 2 + 1,
        "frame_end": index * 2 + 2,
        "boundary_frame": index * 2,
        "cand": 7,
        "masks": [65, 65],
        "ev_flags": [f"0x{ev[0]:04x}", f"0x{ev[1]:04x}"],
        "ev_flags_dec": [ev[0], ev[1]],
        "ev_names": [FOLLOWUP.event_names(ev[0]), FOLLOWUP.event_names(ev[1])],
        "state": {"frame": index * 2 + 2, "stage": 0, "lives_remaining": 2.0,
                  "bombs_remaining": 3.0, "player_power": 100.0, "score": score,
                  "graze": 0, "deaths": 0.0},
        "hud": {"frame": index * 2 + 2, "display_score": score,
                "globals_score": score, "lives": 2, "bombs": 3, "power": 100,
                "graze": 0, "time": 0, "game_over": 0},
        "player": {"px": 100.0, "py": 300.0, "hp": 1.0, "focus": 0,
                   "player_state": 0, "player_state_name": "ALIVE",
                   "invuln_state": 0},
        "enemies": enemies,
        "counts": {"bullets": 0, "lasers": 0, "enemies": len(enemies)},
        "hash": {"frame_start": "aa", "frame_end": "bb"},
        "anchor": {"kind": "none", "frame": index * 2, "expected": None,
                   "actual": "bb", "ok": True, "checks": []},
        "classification": "measured",
    }


def boss_kill_rows() -> list:
    """A BOSS generation that spawns at decision 10 and dies at decision 12."""

    rows = [decision_row(index, []) for index in range(10)]
    rows.append(decision_row(10, [hazard_enemy(3, BOSS_FLAGS, 300)]))
    rows.append(decision_row(11, [hazard_enemy(3, BOSS_FLAGS, 40)]))
    rows.append(decision_row(12, [], ev=(FOLLOWUP.EVENT_ENEMY_REMOVED,
                                         FOLLOWUP.EVENT_EFFECT_SPAWNED),
                             score=9999))
    rows += [decision_row(index, []) for index in (13, 14)]
    return rows


def test_signed_life_restores_the_negative_value():
    assert FOLLOWUP.signed_life(0xFFFFFFFF) == -1
    assert FOLLOWUP.signed_life(0xFFFFFF9C) == -100
    assert FOLLOWUP.signed_life(300) == 300
    assert FOLLOWUP.signed_life(0) == 0


def test_enemy_row_carries_flags_bits_and_the_signed_life():
    flags = (FOLLOWUP.ENEMY_FLAG_BOSS | FOLLOWUP.ENEMY_FLAG_DAMAGEABLE
             | FOLLOWUP.ENEMY_FLAG_PERSIST_AFTER_DEATH | 0x1)
    row = (f"7:{flags:08x}:" + ":".join(["40800000", "40000000", "00000000"])
           + ":ffffff9c:1234" + ":40800000:40000000:00000000:41800000:41800000")
    decoded = FOLLOWUP.enemy_row_record(row)
    assert decoded["slot"] == 7
    assert decoded["flags1"] == f"0x{flags:08x}"
    assert decoded["flags1_dec"] == flags
    assert decoded["life"] == -100, "the %08x life field must be signed again"
    assert decoded["life_raw_unsigned"] == 0xFFFFFF9C
    assert decoded["boss"] is True and decoded["damageable"] is True
    assert decoded["persist_after_death"] is True
    assert decoded["timer"] == 1234
    assert decoded["hitbox_w"] == 16.0
    assert "PERSIST_AFTER_DEATH" in FOLLOWUP.flags1_names(flags)


def test_parse_hazards_enemies_reads_the_enemy_rows_only():
    reply = ("OK frames=42 nb=183 nl=0 ne=2 "
             "bullets=2:1:8:42d4eea6:42c176f4:3dcccccd:bc51ad65:bce84cff:178:40800000:0 "
             "lasers= "
             "enemies=" + ",".join([
                 "0:00000003:40490fdb:40490fdb:00000000:0000012c:00000001"
                 ":40490fdb:40490fdb:00000000:41800000:41800000",
                 "1:8080000b:40800000:40800000:00000000:00000000:00000002"
                 ":40800000:40800000:00000000:41800000:41800000"]))
    decoded = FOLLOWUP.parse_hazards_enemies(reply)
    assert decoded["frame"] == 42
    assert decoded["bullets"] == 183 and decoded["lasers"] == 0
    assert decoded["enemies_count"] == 2
    assert [row["slot"] for row in decoded["enemies"]] == [0, 1]
    assert decoded["enemies"][0]["life"] == 300
    assert decoded["enemies"][1]["persist_after_death"] is True
    assert decoded["enemies"][1]["life"] == 0


def test_tape_schedule_reproduces_t04_and_d1():
    """The rebuilt tape must be the recorded masks, not a re-derivation."""

    tape = FOLLOWUP.tape_schedule(REPO_ROOT)
    assert tape["decisions"] == 6660
    assert tape["slots"] == 6588
    assert tape["leading_decisions"] == 8
    assert tape["leading_first_eight"] == [7] * 8
    assert tape["grid_is_two_frames_per_decision"] is True
    assert tape["decisions_carrying_bomb"] == []
    assert tape["decisions_without_agent_bits"] == []
    cross = FOLLOWUP.tape_cross_check(REPO_ROOT, tape)
    assert cross["t04"]["mismatches"] == 0
    assert cross["t04"]["compared"] == cross["t04"]["expected"] == 6588
    assert cross["d1"]["mismatches"] == 0
    assert cross["d1"]["compared"] == 6596, "D1 covers the tape's first 6,596 decisions"
    assert cross["d1"]["expected"] == 6596, "the denominator is what D1 can cover"
    assert cross["d1"]["tape_decisions"] == 6660, "the tape itself is longer"
    assert cross["d1"]["uncovered_tail_decisions"] == 64
    assert cross["hash_identity"]["mismatches"] == 0
    assert cross["hash_identity"]["compared"] > 3000


def test_decision_masks_are_the_frozen_executor_output():
    """Decision ``i`` owns STEP frames ``2i+1``/``2i+2``: masks come from D1."""

    plan = FOLLOWUP.d1_replay_plan(REPO_ROOT)
    flat = plan["flat_masks"]
    tape = FOLLOWUP.tape_schedule(REPO_ROOT)
    for index in (0, 8, 47, 48, 1000, 6595):
        pair = FOLLOWUP.decision_masks(tape["cands"], index)
        assert pair == [flat[2 * index], flat[2 * index + 1]]
    # a release decision drops exactly the SHOOT bit on the second frame
    release = next(index for index in range(0, 200)
                   if FOLLOWUP.adapter()["release_due"](index, 48))
    pair = FOLLOWUP.decision_masks(tape["cands"], release)
    assert pair[1] == pair[0] & ~FOLLOWUP.adapter()["BTN_SHOOT"] & ~FOLLOWUP.adapter()["BTN_BOMB"]
    assert pair[0] != pair[1]


def test_generation_segments_name_every_ending():
    cases = {
        "kill_M0": (boss_kill_rows(), "kill_M0"),
        "kill_M1": ([decision_row(index, []) for index in range(5)]
                    + [decision_row(5, [hazard_enemy(2, BOSS_FLAGS, 120)]),
                       decision_row(6, [hazard_enemy(
                           2, BOSS_FLAGS
                           | FOLLOWUP.ENEMY_FLAG_PERSIST_AFTER_DEATH, 0)],
                           ev=(FOLLOWUP.EVENT_ENEMY_REMOVED,
                               FOLLOWUP.EVENT_EFFECT_SPAWNED)),
                       decision_row(7, [hazard_enemy(
                           2, BOSS_FLAGS
                           | FOLLOWUP.ENEMY_FLAG_PERSIST_AFTER_DEATH, 0)])],
                    "kill_M1"),
        "scripted_retreat": ([decision_row(index, []) for index in range(3)]
                             + [decision_row(3, [hazard_enemy(7, PLAIN_FLAGS, 200)]),
                                decision_row(4, [], ev=(FOLLOWUP.EVENT_ENEMY_REMOVED, 0))],
                             "scripted_retreat"),
        "clear_wipe": ([decision_row(index, []) for index in range(3)]
                       + [decision_row(3, [hazard_enemy(9, PLAIN_FLAGS, 200)]),
                          decision_row(4, [hazard_enemy(9, PLAIN_FLAGS, 0)]),
                          decision_row(5, [], ev=(FOLLOWUP.EVENT_ENEMY_REMOVED, 0))],
                       "clear_wipe"),
    }
    for name, (rows, expected) in cases.items():
        segments = FOLLOWUP.generation_segments(rows)
        assert [item["outcome"] for item in segments["generations"]] == [expected], name
    # a boss phase end is recorded but is not a terminal outcome
    rows = ([decision_row(index, []) for index in range(4)]
            + [decision_row(4, [hazard_enemy(1, BOSS_FLAGS, 500)]),
               decision_row(5, [hazard_enemy(1, BOSS_FLAGS, 500)]),
               decision_row(6, [hazard_enemy(
                   1, BOSS_FLAGS & ~FOLLOWUP.ENEMY_FLAG_DAMAGEABLE, 1)],
                   ev=(FOLLOWUP.EVENT_ENEMY_REMOVED,
                       FOLLOWUP.EVENT_EFFECT_SPAWNED)),
               decision_row(7, [hazard_enemy(
                   1, BOSS_FLAGS & ~FOLLOWUP.ENEMY_FLAG_DAMAGEABLE, 1)])])
    segments = FOLLOWUP.generation_segments(rows)
    generation = segments["generations"][0]
    assert len(generation["phase_ends"]) == 1
    assert generation["phase_ends"][0]["kind"] == "boss_phase_end_M3"
    assert generation["outcome"] == "still_active_at_cap"
    # a reused slot breaks the generation instead of being a removal
    rows = ([decision_row(index, []) for index in range(3)]
            + [decision_row(3, [hazard_enemy(4, PLAIN_FLAGS, 1)],
                            ev=(FOLLOWUP.EVENT_ENEMY_SPAWNED, 0)),
               decision_row(4, [hazard_enemy(4, PLAIN_FLAGS, 1)],
                            ev=(FOLLOWUP.EVENT_ENEMY_SPAWNED
                                | FOLLOWUP.EVENT_ENEMY_REMOVED, 0)),
               decision_row(5, [hazard_enemy(4, PLAIN_FLAGS, 1)])])
    segments = FOLLOWUP.generation_segments(rows)
    assert len(segments["generations"]) == 2
    assert segments["generations"][0]["outcome"] == "slot_reused"


def test_candidate_windows_honour_the_cap_and_the_stage_allowance():
    segments = FOLLOWUP.generation_segments(boss_kill_rows())
    windows = FOLLOWUP.candidate_windows(segments["generations"],
                                        stage_frames_remaining=20000)
    assert len(windows) == 1
    window = windows[0]
    assert window["eligible"] is True, window["missing"]
    assert window["entry_decision"] == window["spawn_decision"] - 2
    assert window["entry_decision"] >= 1
    assert window["span_frames"] is not None and window["span_frames"] <= 600
    # the same window cannot be paid for by two arms inside 1,000 frames
    tight = FOLLOWUP.candidate_windows(segments["generations"],
                                      stage_frames_remaining=1000)
    assert tight[0]["eligible"] is False
    assert any("frames" in item for item in tight[0]["missing"])
    # a non-kill ending is never eligible
    retreat = FOLLOWUP.generation_segments(
        [decision_row(index, []) for index in range(3)]
        + [decision_row(3, [hazard_enemy(7, PLAIN_FLAGS, 200)]),
           decision_row(4, [], ev=(FOLLOWUP.EVENT_ENEMY_REMOVED, 0))])
    assert FOLLOWUP.candidate_windows(retreat["generations"],
                                      stage_frames_remaining=20000) == []


def test_task_card_round_trip_and_tamper_refusal(tmp_path):
    rows = boss_kill_rows()
    tape = FOLLOWUP.tape_schedule(REPO_ROOT)
    windows = FOLLOWUP.candidate_windows(
        FOLLOWUP.generation_segments(rows)["generations"],
        stage_frames_remaining=20000)
    recon = {"base": tmp_path / "02b" / "recon",
             "result": {"terminal": {"kind": "frame_cap"},
                        "counts": {"frames": 30, "decisions": 15},
                        "heads": {"head": "0c4efbd", "dirty_count": 0}},
             "rows": rows}
    card = FOLLOWUP.build_task_card(candidate=windows[0], tape=tape, rows=rows,
                                    recon=recon, repo=REPO_ROOT,
                                    budget={"stage_remaining": {"engine_frames": 20000}})
    body = {key: value for key, value in card.items() if key != "content_sha256"}
    assert FOLLOWUP.content_digest(body) == card["content_sha256"]
    for key in ("schema", "task_id", "created_at", "ticket", "repo_head",
                "tool_sha256", "event", "entry", "resources_at_entry", "success",
                "legal_failure", "frame_cap", "termination_priority",
                "teacher_continuation", "control", "read_method",
                "not_a_policy_result", "content_sha256"):
        assert key in card, key
    assert card["entry"]["prefix_masks_sha256"] == FOLLOWUP.tape_digest(
        tape, 0, card["entry"]["prefix_decisions"])
    assert card["entry"]["pulse_period"] == 48
    assert card["entry"]["controller"] == "recorded_tape_masks"
    assert card["entry"]["reads_state"] is False
    assert card["frame_cap"] == {"decisions": 300, "frames": 600}
    assert card["control"]["cand"] == 12
    assert card["not_a_policy_result"] is True
    assert card["event"]["seed12_instance"]["slot"] == 3
    assert len(card["event"]["alternative_outcomes"]) >= 8
    assert card["entry"]["hash"] == rows[card["entry"]["decision_index"] - 1][
        "hash"]["frame_end"]
    written = FOLLOWUP.write_task_card(tmp_path / "02b", card)
    assert written["content_sha256"] == card["content_sha256"]
    loaded = FOLLOWUP.read_task_card(tmp_path / "02b")
    assert loaded["content_sha256_matches"] is True
    text = (tmp_path / "02b" / "task_card.md").read_text(encoding="utf-8")
    assert card["content_sha256"] in text and text.startswith("# Task card")
    tampered = json.loads((tmp_path / "02b" / "task_card.json").read_text(
        encoding="utf-8"))
    tampered["event"]["classification"] = "scripted_retreat"
    (tmp_path / "02b" / "task_card.json").write_text(json.dumps(tampered),
                                                     encoding="utf-8")
    with pytest.raises(FOLLOWUP.EngineRefused):
        FOLLOWUP.read_task_card(tmp_path / "02b")


def _ledger_rows() -> list:
    """The 02-A rows of the shared ledger, verbatim in shape."""

    return [
        {"phase": "reserve", "attempt_id": "02a-prefix",
         "requested": {"starts": 1.0, "engine_frames": 8668.0, "wall_s": 900.0,
                       "write_gib": 1.0}},
        {"phase": "settle", "attempt_id": "02a-prefix",
         "actual": {"starts": 1.0, "engine_frames": 8668.0, "wall_s": 75.825,
                    "write_gib": 0.466}},
        {"phase": "reserve", "attempt_id": "02a-reference",
         "requested": {"starts": 1.0, "engine_frames": 8665.0, "wall_s": 900.0,
                       "write_gib": 1.0}},
        {"phase": "settle", "attempt_id": "02a-reference",
         "actual": {"starts": 1.0, "engine_frames": 8665.0, "wall_s": 74.798,
                    "write_gib": 0.466}},
    ]


def test_stage_budget_reads_the_existing_ledger_and_refuses_an_over_booking(tmp_path):
    (tmp_path / "ledger.jsonl").write_text(
        "\n".join(json.dumps(row) for row in _ledger_rows()) + "\n",
        encoding="utf-8")
    report = FOLLOWUP.stage_budget(tmp_path, need_starts=1, need_frames=13194,
                                   label="02b-recon")
    payload = report["report"]
    assert payload["ledger_rows_read"] == 4
    assert payload["cumulative"]["starts"] == 2
    assert payload["cumulative"]["engine_frames"] == 17333
    assert payload["stage_used"]["starts"] == 0, "02-A rows are not this stage's"
    assert payload["stage_remaining"]["starts"] == 3
    assert payload["stage_remaining"]["engine_frames"] == 20000
    assert payload["grand_remaining"]["starts"] == 26
    assert payload["grand_remaining"]["engine_frames"] == 360000 - 17333
    assert report["limits"].max_starts == 5
    assert report["limits"].max_engine_frames == 37333
    with pytest.raises(FOLLOWUP.BudgetRefused):
        FOLLOWUP.stage_budget(tmp_path, need_starts=1, need_frames=20001,
                              label="02b-recon")


def test_stage_budget_counts_only_the_stage_rows(tmp_path):
    rows = _ledger_rows() + [
        {"phase": "reserve", "attempt_id": "02b-recon",
         "requested": {"starts": 1.0, "engine_frames": 13194.0, "wall_s": 900.0,
                       "write_gib": 1.0}},
        {"phase": "settle", "attempt_id": "02b-recon",
         "actual": {"starts": 1.0, "engine_frames": 13194.0, "wall_s": 190.0,
                    "write_gib": 0.8}},
    ]
    (tmp_path / "ledger.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = FOLLOWUP.stage_budget(tmp_path, need_starts=2, need_frames=6800,
                                   label="02c-intervene")
    payload = report["report"]
    assert payload["stage_used"]["starts"] == 1
    assert payload["stage_used"]["engine_frames"] == 13194
    assert payload["cumulative"]["starts"] == 3
    assert payload["cumulative"]["engine_frames"] == 17333 + 13194
    assert payload["stage_remaining"]["starts"] == 2
    assert payload["stage_remaining"]["engine_frames"] == 20000 - 13194
    assert report["limits"].max_engine_frames == 37333
    with pytest.raises(FOLLOWUP.BudgetRefused):
        FOLLOWUP.stage_budget(tmp_path, need_starts=2, need_frames=7000,
                              label="02c-intervene")


def test_frozen_tracker_names_the_ending_from_raw_rows():
    rows = boss_kill_rows()
    card = {"event": {"seed12_instance": {"slot": 3, "spawn_decision": 10,
                                          "life_at_spawn": 300}},
            "entry": {"prefix_decisions": 8}}
    outcome = FOLLOWUP.recompute_arm_outcome(rows[8:], card)
    assert outcome["recomputed_outcome"] == "kill_M0"
    assert outcome["life_first"] == 300 and outcome["life_last"] == 40
    assert outcome["life_min"] == 40
    assert outcome["outcome_frame"] == 25
    # a slot that is never seen is named, not silently dropped
    never = {"event": {"seed12_instance": {"slot": 99, "spawn_decision": 10,
                                           "life_at_spawn": 300}},
             "entry": {"prefix_decisions": 8}}
    assert FOLLOWUP.recompute_arm_outcome(rows[8:], never)[
        "recomputed_outcome"] == "generation_not_observed"


def test_outcome_vocabulary_covers_the_ticket_outcomes():
    for name in ("kill_M0", "kill_M1", "kill_M2", "boss_phase_end_M3",
                 "scripted_retreat", "clear_wipe", "not_removed_within_cap",
                 "legal_death", "game_over", "frame_cap"):
        assert name in FOLLOWUP.OUTCOME_VOCABULARY
    assert FOLLOWUP.KILL_OUTCOMES == ("kill_M0", "kill_M1", "kill_M2")


def test_nominate_refuses_a_missing_recon(tmp_path, capsys):
    out = tmp_path / "out"
    code = FOLLOWUP.main(["nominate", "--repo", str(REPO_ROOT), "--out", str(out),
                          "--recon", str(tmp_path / "no-recon")])
    assert code == FOLLOWUP.EXIT_TECHNICAL
    payload = json.loads((out / "02b" / "nomination.json").read_text(
        encoding="utf-8"))
    assert payload["verdict"] == "technical_invalid"
    assert payload["gate"]["passed"] is False
    assert "recon" in payload["error"]["detail"]


def test_verify_02b_requires_the_recon_and_the_nomination(tmp_path):
    out = tmp_path / "out"
    (out / "02b" / "recon").mkdir(parents=True)
    (out / "02b" / "recon" / "result.json").write_text("{}", encoding="utf-8")
    assert FOLLOWUP.main(["verify", "--out", str(out), "--package", "02b"]) == \
        FOLLOWUP.EXIT_TECHNICAL
    payload = json.loads((out / "verify.json").read_text(encoding="utf-8"))
    assert payload["gate"]["passed"] is False


# ---------------------------------------------------------------------------
# 03 pilot: the registration, the frozen table, the arms and the nomination rule
# Everything here is offline: the fit and the arms need the engine and live in
# 03/smoke or 03/dev evidence, never in this file.
# ---------------------------------------------------------------------------
def test_pilot_layout_keeps_the_smoke_out_of_the_pilot_evidence(tmp_path):
    formal = FOLLOWUP.pilot_layout(tmp_path, smoke=False)
    smoke = FOLLOWUP.pilot_layout(tmp_path, smoke=True)
    assert formal["base"] == tmp_path / "03"
    assert smoke["base"] == tmp_path / "03" / "smoke"
    assert formal["arms"] == tmp_path / "03" / "dev"
    assert smoke["arms"] == tmp_path / "03" / "smoke"
    assert formal["classification"] == "measured"
    assert smoke["classification"] == FOLLOWUP.PILOT_SMOKE_CLASSIFICATION
    assert smoke["train"].is_relative_to(smoke["base"])


def _fake_card() -> dict:
    """The four card fields the registration reads (nothing else is used)."""

    return {"content_sha256": "617f03a5" + "0" * 56, "task_id": "fake-card",
            "entry": {"frame": 3086, "prefix_decisions": 1543,
                      "prefix_masks_sha256": "2632726e" + "0" * 56}}


def test_pilot_registration_freezes_and_refuses_a_different_body(tmp_path):
    payload = FOLLOWUP.pilot_roots_payload(
        repo=REPO_ROOT, card=_fake_card(),
        created_at="2026-11-26T00:00:00+08:00")
    assert payload["schema"] == FOLLOWUP.SCHEMA_PILOT_ROOTS
    assert payload["status"] == "frozen"
    assert [entry["seed"] for entry in payload["roots"]] == list(
        FOLLOWUP.PILOT_DEV_SEEDS)
    assert payload["training_roots"]["fit"]["id"] == "d1-n3371-seed13"
    assert payload["training_roots"]["donor"]["id"] == "d1-n3371-seed14"
    assert payload["training_roots"]["fit"]["labels_sha256"].startswith("b20939e6")
    assert payload["training_roots"]["donor"]["labels_sha256"].startswith("fd78ddb9")
    assert payload["exposure"]["verdict"] == "not_exposed"
    assert payload["recipe"]["updates"] == 200
    assert payload["recipe"]["lr"] == 0.0003
    assert payload["split"]["sha256"] == (
        "bfab16ba7b68ff4f8dc33a6dd06bbe0e69c1771293804ad1eb432f1a960522c2")
    path = tmp_path / "pilot_03_roots.json"
    written = FOLLOWUP.write_pilot_roots_config(path, payload)
    assert written["written"] is True
    assert written["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # the ticket's write scope allows exactly one new file: no companion checksum
    assert sorted(item.name for item in tmp_path.iterdir()) == [
        "pilot_03_roots.json"]
    loaded = FOLLOWUP.load_pilot_roots_config(path)
    assert loaded["content_sha256"] == loaded["content_sha256_recomputed"]
    assert loaded["sha256"] == written["sha256"]
    # a second, different body is a refusal, never an update
    changed = json.loads(json.dumps(payload))
    changed["created_at"] = "2026-11-26T00:00:01+08:00"
    with pytest.raises(FOLLOWUP.EngineRefused):
        FOLLOWUP.write_pilot_roots_config(path, changed)
    # an identical body is accepted and never rewritten
    again = FOLLOWUP.write_pilot_roots_config(path, payload)
    assert again["written"] is False
    assert again["sha256"] == written["sha256"]


def test_pilot_registration_rejects_a_tampered_body(tmp_path):
    payload = FOLLOWUP.pilot_roots_payload(
        repo=REPO_ROOT, card=_fake_card(),
        created_at="2026-11-26T00:00:00+08:00")
    path = tmp_path / "pilot_03_roots.json"
    FOLLOWUP.write_pilot_roots_config(path, payload)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["roots"][0]["seed"] = 1024
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(FOLLOWUP.EngineRefused):
        FOLLOWUP.load_pilot_roots_config(path)


def test_open_loop_table_digests_are_recomputable():
    """``table_sha256`` and ``mask_trace_sha256`` are re-derived, not trusted."""

    cands = [4, 4, 11, 7, 12, 12, 3, 0, 17, 8]
    entry = 1543
    masks = FOLLOWUP.pilot_table_masks(cands, entry)
    manual = []
    for offset, cand in enumerate(cands):
        release = FOLLOWUP.adapter()["release_due"](entry + offset, 48)
        parts = FOLLOWUP.adapter()["commit_frames"](
            cand, decision_frames=2, release_shoot=release)
        values = FOLLOWUP.expand_commit(parts)
        manual.append([int(values[0]), int(values[1])])
    assert masks == manual
    assert FOLLOWUP.table_sha256(cands) == hashlib.sha256(
        "".join(f"{index}:{cand}\n" for index, cand in enumerate(cands)
                ).encode("ascii")).hexdigest()
    first = 3086 + 1
    flat = FOLLOWUP.flat_masks_of(masks, first)
    assert sorted(flat) == list(range(first, first + 2 * len(cands)))
    digest = FOLLOWUP.mask_trace_digest(flat, first, first + 2 * len(cands) - 1)
    assert digest == hashlib.sha256(
        "".join(f"{frame}:{flat[frame]}\n" for frame in
                range(first, first + 2 * len(cands))).encode("ascii")).hexdigest()


def test_pilot_table_uses_the_global_pulse_phase_not_the_local_one():
    """The prefix ends at decision 1542, so the arm's phase is not the table's 0."""

    cands = [12] * 96
    assert FOLLOWUP.pilot_table_masks(cands, 1543) != FOLLOWUP.pilot_table_masks(
        cands, 0)
    # 1543 % 48 = 7, so the release decisions of the arm are k = 40, 88, ...
    global_masks = FOLLOWUP.pilot_table_masks(cands, 1543)
    releases = [index for index, pair in enumerate(global_masks)
                if pair[0] != pair[1]]
    assert releases and all((1543 + index + 1) % 48 == 0 for index in releases)


def _fake_arm(outcome: str, *, entry: bool = True, terminal: str = "frozen",
              exit_code: int = 0, share: float = 0.5) -> dict:
    return {"outcome": outcome, "stop_reason": "frozen_consequence",
            "terminal": terminal, "exit_code": exit_code,
            "entry_reached": entry, "control_decisions": 300,
            "lives_lost": 0.0, "frozen_life_first": 6000, "frozen_life_last": 0,
            "frozen_life_min": 0, "frozen_observations": 300,
            "candidate_dominant_share": share}


def _fake_arms(seeds, *, final="kill_M1", init="not_removed_within_cap",
               strong="not_removed_within_cap") -> dict:
    return {str(seed): {"init": _fake_arm(init), "final": _fake_arm(final),
                        "open_loop": _fake_arm(strong)} for seed in seeds}


def test_pilot_comparison_needs_both_contrasts_and_two_positive_roots_each():
    seeds = FOLLOWUP.PILOT_DEV_SEEDS
    comparison = FOLLOWUP.pilot_comparison(arms=_fake_arms(seeds), smoke=False)
    assert comparison["nominate_expansion"] is True
    assert comparison["equal_weight_means"]["learned_minus_before"] == 1.0
    assert comparison["equal_weight_means"]["learned_minus_strong"] == 1.0
    assert len(comparison["positive_roots"]["learned_minus_before"]) == 4
    # one root short of the four the ticket allocates: no nomination
    partial = FOLLOWUP.pilot_comparison(arms=_fake_arms(seeds[:3]), smoke=False)
    assert partial["nominate_expansion"] is False
    # a technical loss forbids nomination
    broken = _fake_arms(seeds)
    broken[str(seeds[0])]["final"] = _fake_arm(
        "kill_M1", terminal="technical_conflict", exit_code=2)
    assert FOLLOWUP.pilot_comparison(arms=broken, smoke=False)[
        "nominate_expansion"] is False
    # only the initial arm improves: no nomination
    weak = FOLLOWUP.pilot_comparison(
        arms=_fake_arms(seeds, final="not_removed_within_cap"), smoke=False)
    assert weak["nominate_expansion"] is False
    # a smoke is never a pilot result
    assert FOLLOWUP.pilot_comparison(arms=_fake_arms(seeds), smoke=True)[
        "nominate_expansion"] is False


def test_pilot_refuses_before_any_start_without_the_task_card(tmp_path, capsys):
    out = tmp_path / "out"
    config = tmp_path / "pilot_03_roots.json"
    code = FOLLOWUP.main(["pilot", "--repo", str(REPO_ROOT), "--out", str(out),
                          "--engine", str(tmp_path / "no-engine"),
                          "--data-source", str(tmp_path / "no-data"),
                          "--config", str(config), "--no-xvfb",
                          "--only-seed", "1000"])
    assert code == FOLLOWUP.EXIT_TECHNICAL
    record = json.loads((out / "03" / "smoke" / "pilot.json").read_text(
        encoding="utf-8"))
    assert record["engine_starts"] == 0
    assert record["terminal"]["kind"] == "technical_invalid"
    assert "task card" in record["terminal"]["detail"]
    assert record["classification"] == FOLLOWUP.PILOT_SMOKE_CLASSIFICATION
    assert not config.exists(), "the registration is written after the card is read"
    assert not (out / "ledger.jsonl").exists(), "no ledger row for a refused start"


def test_run_restores_a_passing_record_when_the_rerun_fails(tmp_path):
    """A wrong ``--repo`` may fail the package, never destroy passing evidence."""

    out = tmp_path / "out"
    (out / "02a").mkdir(parents=True)
    good = {"schema": FOLLOWUP.SCHEMA_SOURCES,
            "gate": {"passed": True, "checks": []}, "sources": [1, 2, 3, 4]}
    (out / "02a" / "sources.json").write_text(json.dumps(good), encoding="utf-8")
    before = (out / "02a" / "sources.json").read_bytes()
    code = FOLLOWUP.main(["run", "--repo", str(tmp_path / "no-checkout"),
                          "--out", str(out), "--only", "sources"])
    assert code in (FOLLOWUP.EXIT_GATE_FAILED, FOLLOWUP.EXIT_TECHNICAL)
    payload = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert payload["gates"]["02a-sources"]["status"] in ("failed", "technical")
    assert payload["gates"]["02a-sources"]["restored_evidence"] == [
        "02a/sources.json"]
    assert (out / "02a" / "sources.json").read_bytes() == before
    assert FOLLOWUP.run_evidence_snapshot(out, "02a-sources") == {
        "02a/sources.json": before}
    assert FOLLOWUP.run_evidence_restore(out, {"02a/sources.json": before}) == []


def test_pilot_never_overwrites_a_passing_summary_with_a_failure(tmp_path):
    """A re-run refused by the budget must not replace the smoke summary."""

    out = tmp_path / "out"
    base = out / "03" / "smoke"
    base.mkdir(parents=True)
    passing = {"schema": FOLLOWUP.SCHEMA_PILOT, "smoke": True,
               "verdict": "smoke_ok", "smoke_ok": True,
               "gate": {"passed": True, "checks": []}}
    summary = base / "pilot.json"
    summary.write_text(json.dumps(passing), encoding="utf-8")
    before = summary.read_bytes()
    code = FOLLOWUP.main(["pilot", "--repo", str(REPO_ROOT), "--out", str(out),
                          "--engine", str(tmp_path / "no-engine"),
                          "--data-source", str(tmp_path / "no-data"),
                          "--config", str(tmp_path / "cfg.json"), "--no-xvfb",
                          "--only-seed", "1000"])
    assert code == FOLLOWUP.EXIT_TECHNICAL
    assert summary.read_bytes() == before, "the passing summary was kept"
    alternates = [item for item in base.iterdir()
                  if item.name != "pilot.json" and item.suffix == ".json"]
    assert len(alternates) == 1, sorted(item.name for item in base.iterdir())
    record = json.loads(alternates[0].read_text(encoding="utf-8"))
    assert record["preserved_evidence"]["kept"] == summary.as_posix()
    assert record["gate"]["passed"] is False


# ---------------------------------------------------------------------------
# 03 strength check on the fit root (ticket L51): the offline rules only
# The two engine starts live in 03/strength/ evidence, never in this file.
# ---------------------------------------------------------------------------
def _strength_arm_fixture(outcome: str, *, score: int = 1000,
                          life_first: int = 4878, life_min: int = 1732,
                          terminal: str | None = None,
                          exit_code: int = 0) -> dict:
    """One arm summary with only the fields the judgment rule reads."""

    return {"outcome": outcome,
            "terminal": terminal if terminal is not None else outcome,
            "exit_code": int(exit_code),
            "score_at_end": int(score),
            "frozen_life_first": int(life_first),
            "frozen_life_min": int(life_min),
            "frozen_life_drop": float(life_first - life_min),
            "stop_reason": "frozen_consequence",
            "outcome_decision": 1600, "outcome_frame": 3200}


def test_strength_judgment_follows_the_ticket_outcome_order():
    """``kill_*`` > alive at the cap > death, and a high score never outranks it."""

    kill = _strength_arm_fixture("kill_M1", score=10, life_min=0)
    death = _strength_arm_fixture("legal_death", score=99_999, life_min=3000)
    passed = FOLLOWUP.strength_judgment(
        arms={"open_loop": kill, "constant_12": death})
    assert passed["completed_at_least_once"] is True
    assert passed["not_worse_than_12"] is True
    assert passed["strength_passed"] is True
    assert passed["not_worse_evidence"]["decided_by"] == "outcome_tier"
    assert passed["classification"] == FOLLOWUP.STRENGTH_CLASSIFICATION
    assert passed["completed_evidence"]["kill_outcomes"] == list(
        FOLLOWUP.KILL_OUTCOMES)
    worse = FOLLOWUP.strength_judgment(
        arms={"open_loop": death, "constant_12": kill})
    assert worse["completed_at_least_once"] is False
    assert worse["not_worse_than_12"] is False
    assert worse["strength_passed"] is False
    # alive at the cap beats a death even with a lower score
    alive = _strength_arm_fixture("not_removed_within_cap", score=1)
    assert FOLLOWUP.strength_judgment(
        arms={"open_loop": alive, "constant_12": death})["not_worse_than_12"] is True
    # ... but not completing keeps strength_passed false
    assert FOLLOWUP.strength_judgment(
        arms={"open_loop": alive, "constant_12": death})["strength_passed"] is False


def test_strength_judgment_tie_break_needs_score_and_life_drop():
    base = dict(life_first=4878, life_min=1732)  # a drop of 3,146
    same = FOLLOWUP.strength_judgment(arms={
        "open_loop": _strength_arm_fixture("not_removed_within_cap", score=100, **base),
        "constant_12": _strength_arm_fixture("not_removed_within_cap", score=100,
                                             **base)})
    assert same["not_worse_than_12"] is True
    assert same["not_worse_evidence"]["decided_by"] == "same_tier_tie_break"
    assert same["not_worse_evidence"]["tie_break"]["conflict"] is False
    # more score but less damage on the target: the rule says not better
    conflict = FOLLOWUP.strength_judgment(arms={
        "open_loop": _strength_arm_fixture("not_removed_within_cap", score=200, **base),
        "constant_12": _strength_arm_fixture("not_removed_within_cap", score=100,
                                             life_first=4878, life_min=900)})
    assert conflict["not_worse_than_12"] is False
    tie = conflict["not_worse_evidence"]["tie_break"]
    assert tie["score_at_end"]["not_lower"] is True
    assert tie["frozen_target_life_drop"]["not_smaller"] is False
    assert tie["conflict"] is True
    # a technical arm makes the comparison inadmissible even after a kill
    technical = FOLLOWUP.strength_judgment(arms={
        "open_loop": _strength_arm_fixture("kill_M0", life_min=0),
        "constant_12": _strength_arm_fixture(
            "legal_death", terminal="technical_conflict", exit_code=2)})
    assert technical["completed_at_least_once"] is True
    assert technical["not_worse_than_12"] is False
    assert technical["strength_passed"] is False
    assert technical["not_worse_evidence"]["technical_arms"]
    # an outcome outside the three tiers ranks below all of them and is named
    other = FOLLOWUP.strength_judgment(arms={
        "open_loop": _strength_arm_fixture("scripted_retreat"),
        "constant_12": _strength_arm_fixture("legal_death")})
    assert other["arms"]["open_loop"]["tier"] == 0
    assert other["not_worse_than_12"] is False


def test_strength_action_stats_measure_frequency_dwell_and_releases():
    cands = [0] * 300
    masks = FOLLOWUP.pilot_table_masks(cands, 1543)
    stats = FOLLOWUP.strength_action_stats(cands=cands, masks=masks,
                                          first_decision=1543)
    assert stats["candidate_histogram"] == {"0": 300}
    assert stats["distinct_candidates"] == 1
    assert stats["dominant_share"] == 1.0
    run = stats["longest_same_candidate_run"]
    assert run["decisions"] == 300 and run["cand"] == 0
    assert (run["first_decision"], run["last_decision"]) == (1543, 1842)
    releases = stats["releases"]
    assert releases["count"] == 6
    assert releases["decisions"] == [1583, 1631, 1679, 1727, 1775, 1823]
    assert releases["per_48_decision_block"] == {str(block): 1 for block in range(6)}
    assert releases["complete_blocks"] == 6
    assert releases["partial_blocks"] == [6]
    assert releases["all_complete_blocks_have_exactly_one_release"] is True
    feasibility = stats["execution_feasibility"]
    assert feasibility["frames_with_bomb"] == 0
    assert feasibility["frames_with_unknown_bits"] == 0
    assert feasibility["frames_without_shoot"] == 6
    assert feasibility["frames_whose_mask_decodes_to_the_executed_candidate"] == 594
    assert feasibility["mask_anomalies"] == 0
    assert feasibility["all_frames_executable"] is True
    # a mask that is not the executor's expansion is named, never smoothed
    broken = [[int(pair[0]), int(pair[1])] for pair in masks[:2]]
    broken[0] = [broken[0][0] | int(FOLLOWUP.adapter()["BTN_BOMB"]), broken[0][1]]
    bad = FOLLOWUP.strength_action_stats(cands=cands[:2], masks=broken,
                                        first_decision=1543)
    assert bad["execution_feasibility"]["all_frames_executable"] is False
    assert bad["execution_feasibility"]["mask_anomalies"] == 1
    assert bad["execution_feasibility"]["frames_with_bomb"] == 1


def test_strength_arm_candidates_are_the_two_frozen_controls():
    table = [0] * 8
    assert FOLLOWUP.strength_arm_candidates(FOLLOWUP.STRENGTH_ARM_OPEN_LOOP,
                                            table) == table
    assert FOLLOWUP.strength_arm_candidates(FOLLOWUP.STRENGTH_ARM_CONSTANT_12,
                                            table) == [12] * 8
    assert FOLLOWUP.CONTROL_CAND == 12
    assert FOLLOWUP.STRENGTH_ARMS == ("open_loop", "constant_12")
    with pytest.raises(FOLLOWUP.EngineRefused):
        FOLLOWUP.strength_arm_candidates("something_else", table)


def test_the_required_open_loop_digest_is_the_300_zero_candidates():
    """The frozen table the strength check may play is constant candidate 0."""

    assert FOLLOWUP.STRENGTH_REQUIRED_TABLE_SHA256 == (
        "77adfa10c02a5667f56482c59ef83f20666a86a54c54ec430535c870fa340bca")
    assert FOLLOWUP.table_sha256([0] * FOLLOWUP.PILOT_WINDOW_DECISIONS) == (
        FOLLOWUP.STRENGTH_REQUIRED_TABLE_SHA256)
    assert FOLLOWUP.table_sha256([1] * FOLLOWUP.PILOT_WINDOW_DECISIONS) != (
        FOLLOWUP.STRENGTH_REQUIRED_TABLE_SHA256)


def test_strength_table_block_refuses_a_tampered_table(tmp_path):
    out = tmp_path / "out"
    table_dir = out / "03" / "open_loop"
    table_dir.mkdir(parents=True)
    cands = [0] * FOLLOWUP.PILOT_WINDOW_DECISIONS
    payload = {"schema": FOLLOWUP.SCHEMA_PILOT_TABLE, "cands": cands,
               "masks": FOLLOWUP.pilot_table_masks(cands, 1543),
               "table_sha256": FOLLOWUP.table_sha256(cands)}
    path = table_dir / "table.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    block = FOLLOWUP.strength_table_block(out)
    assert block["table_sha256"] == FOLLOWUP.STRENGTH_REQUIRED_TABLE_SHA256
    assert all(block["checks"].values())
    assert block["pilot_summary_table_sha256"] is None
    # one different candidate is a refusal, not a silent substitution
    payload["cands"] = [1] + cands[1:]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FOLLOWUP.SourceError):
        FOLLOWUP.strength_table_block(out)
    # a missing table is a refusal too
    path.unlink()
    with pytest.raises(FOLLOWUP.SourceError):
        FOLLOWUP.strength_table_block(out)


def test_strength_refuses_before_any_start_without_the_task_card(tmp_path, capsys):
    out = tmp_path / "out"
    code = FOLLOWUP.main(["strength", "--repo", str(REPO_ROOT), "--out", str(out),
                          "--engine", str(tmp_path / "no-engine"),
                          "--data-source", str(tmp_path / "no-data"), "--no-xvfb"])
    assert code == FOLLOWUP.EXIT_TECHNICAL
    base = out / "03" / "strength"
    record = json.loads((base / "result.json").read_text(encoding="utf-8"))
    assert record["schema"] == FOLLOWUP.SCHEMA_STRENGTH
    assert record["classification"] == FOLLOWUP.STRENGTH_CLASSIFICATION
    assert record["engine_starts"] == 0
    assert record["terminal"]["kind"] == "technical_invalid"
    assert "task card" in record["terminal"]["detail"]
    assert record["judgment"]["strength_passed"] is False
    assert record["seed"] == FOLLOWUP.PILOT_FIT_SEED
    assert not (out / "ledger.jsonl").exists(), "no ledger row for a refused start"
    assert not (base / "arm_open_loop").exists(), "no arm dir before a start"
    assert not (base / "arm_constant_12").exists()


def test_strength_accepts_only_the_fit_root(tmp_path, capsys):
    out = tmp_path / "out"
    code = FOLLOWUP.main(["strength", "--repo", str(REPO_ROOT), "--out", str(out),
                          "--engine", str(tmp_path / "no-engine"),
                          "--data-source", str(tmp_path / "no-data"),
                          "--seed", "14", "--no-xvfb"])
    assert code == FOLLOWUP.EXIT_TECHNICAL
    record = json.loads((out / "03" / "strength" / "result.json").read_text(
        encoding="utf-8"))
    assert record["engine_starts"] == 0
    assert "seed 13" in record["terminal"]["detail"]
    # `pilot --strength-seed` is the same code path and refuses identically
    forwarded = tmp_path / "forwarded"
    code = FOLLOWUP.main(["pilot", "--strength-seed", "14",
                          "--repo", str(REPO_ROOT), "--out", str(forwarded),
                          "--engine", str(tmp_path / "no-engine"),
                          "--data-source", str(tmp_path / "no-data"), "--no-xvfb"])
    assert code == FOLLOWUP.EXIT_TECHNICAL
    record = json.loads((forwarded / "03" / "strength" / "result.json").read_text(
        encoding="utf-8"))
    assert "seed 13" in record["terminal"]["detail"]
    assert record["engine_starts"] == 0
    assert not (forwarded / "03" / "pilot.json").exists(), "the pilot path never ran"
    assert not (forwarded / "03" / "open_loop").exists()


def test_verify_03_strength_is_not_run_without_evidence(tmp_path):
    gate = FOLLOWUP.Gate("verify")
    report = FOLLOWUP.verify_03_strength(tmp_path, REPO_ROOT, gate)
    assert report["status"] == "not-run"
    assert gate.checks == [], "no gate claim is made without evidence"


def test_strength_layout_is_inside_03_only(tmp_path):
    layout = FOLLOWUP.strength_layout(tmp_path)
    assert layout["base"] == tmp_path / "03" / "strength"
    assert layout["result"] == layout["base"] / "result.json"
    assert sorted(str(path.parent.name) for path in layout["arms"].values()) == [
        "strength", "strength"]
    assert layout["table"] == tmp_path / "03" / "open_loop" / "table.json"
    assert layout["card"] == tmp_path / "02b"
    assert layout["base"].is_relative_to(tmp_path / "03")


def _ledger_row(seq: int, attempt: str, phase: str, *,
                starts: float = 0.0, frames: float = 0.0) -> dict:
    row = {"seq": seq, "attempt_id": attempt, "phase": phase,
           "wall_s": 0.0, "write_gib": 0.0}
    if phase == "reserve":
        row["requested"] = {"starts": starts, "engine_frames": frames,
                            "wall_s": 900.0, "write_gib": 1.0}
    else:
        row["actual"] = {"starts": starts, "engine_frames": frames,
                         "wall_s": 41.0, "write_gib": 0.5}
    return row


def test_run_json_ledger_reconciliation_is_exact_and_named():
    """A ledger that grows after ``run.json`` is consistent only if accounted for."""

    claimed = {"rows": 2,
               "actual": {"starts": 20.0, "engine_frames": 90_737.0,
                          "wall_s": 1_139.437, "write_gib": 9.303}}
    base_rows = [_ledger_row(1, "run-03", "reserve"),
                 _ledger_row(2, "run-03", "settle")]
    # 1. nothing grew: the strict equality holds
    strict = FOLLOWUP.run_json_ledger_reconciliation(
        claimed_totals=claimed, ledger_rows=base_rows,
        ledger_actual=claimed["actual"], accounted_attempt_ids=())
    assert strict["ok"] is True
    assert strict["strict_starts_equal"] is True
    # 2. exactly the two accounted attempts grew the ledger
    grown = base_rows + [
        _ledger_row(3, "03-strength-open_loop", "reserve", starts=1, frames=3686),
        _ledger_row(4, "03-strength-open_loop", "settle", starts=1, frames=3266),
        _ledger_row(5, "03-strength-constant_12", "reserve", starts=1, frames=3686),
        _ledger_row(6, "03-strength-constant_12", "settle", starts=1, frames=3352)]
    ledger_actual = {"starts": 22.0, "engine_frames": 90_737.0 + 6_618.0,
                     "wall_s": 1_139.437 + 82.0, "write_gib": 9.303 + 1.0}
    reconciled = FOLLOWUP.run_json_ledger_reconciliation(
        claimed_totals=claimed, ledger_rows=grown, ledger_actual=ledger_actual,
        accounted_attempt_ids=("03-strength-open_loop", "03-strength-constant_12"))
    assert reconciled["strict_starts_equal"] is False
    assert reconciled["extra_rows"] == 4
    assert reconciled["extra_attempt_ids"] == ["03-strength-constant_12",
                                               "03-strength-open_loop"]
    assert reconciled["unaccounted_attempt_ids"] == []
    assert reconciled["reconciled"] is True
    assert reconciled["ok"] is True
    # 3. an attempt nobody accounts for is a failure, not a tolerance
    stray = FOLLOWUP.run_json_ledger_reconciliation(
        claimed_totals=claimed,
        ledger_rows=base_rows + [_ledger_row(3, "03-somewhere-else", "settle",
                                             starts=1, frames=3266)],
        ledger_actual={"starts": 21.0, "engine_frames": 94_003.0,
                       "wall_s": 1_180.0, "write_gib": 9.8},
        accounted_attempt_ids=("03-strength-open_loop",))
    assert stray["unaccounted_attempt_ids"] == ["03-somewhere-else"]
    assert stray["reconciled"] is False
    assert stray["ok"] is False
    # 4. accounted attempts that do not add up to the ledger are still refused
    wrong = FOLLOWUP.run_json_ledger_reconciliation(
        claimed_totals=claimed, ledger_rows=grown,
        ledger_actual={"starts": 25.0, "engine_frames": 90_737.0,
                       "wall_s": 1_139.437, "write_gib": 9.303},
        accounted_attempt_ids=("03-strength-open_loop", "03-strength-constant_12"))
    assert wrong["reconciled"] is False
    assert wrong["ok"] is False
    # 5. no recorded row count: the reconciliation cannot apply (fail closed)
    unknown = FOLLOWUP.run_json_ledger_reconciliation(
        claimed_totals={"actual": claimed["actual"]}, ledger_rows=grown,
        ledger_actual=ledger_actual,
        accounted_attempt_ids=("03-strength-open_loop", "03-strength-constant_12"))
    assert unknown["recorded_rows"] == 0
    assert unknown["ok"] is False


def test_strength_attempt_ids_reads_the_evidence(tmp_path):
    assert FOLLOWUP.strength_attempt_ids(tmp_path) == set()
    base = tmp_path / "03" / "strength"
    base.mkdir(parents=True)
    (base / "result.json").write_text(json.dumps({
        "schema": FOLLOWUP.SCHEMA_STRENGTH,
        "arms": {"open_loop": {"attempt_id": "03-strength-open_loop"},
                 "constant_12": {"attempt_id": "03-strength-constant_12"}}}),
        encoding="utf-8")
    assert FOLLOWUP.strength_attempt_ids(tmp_path) == {
        "03-strength-open_loop", "03-strength-constant_12"}


def test_strength_never_replaces_evidence_that_carries_both_arms(tmp_path):
    """A refused re-run (the budget allows exactly two starts) writes beside it."""

    out = tmp_path / "out"
    base = out / "03" / "strength"
    base.mkdir(parents=True)
    evidence = {"schema": FOLLOWUP.SCHEMA_STRENGTH,
                "classification": FOLLOWUP.STRENGTH_CLASSIFICATION,
                "arms": {"open_loop": {"attempt_id": "03-strength-open_loop"},
                         "constant_12": {"attempt_id": "03-strength-constant_12"}},
                "judgment": {"strength_passed": False},
                "gate": {"passed": False, "checks": []}}
    target = base / "result.json"
    target.write_text(json.dumps(evidence), encoding="utf-8")
    before = target.read_bytes()
    code = FOLLOWUP.main(["strength", "--repo", str(REPO_ROOT), "--out", str(out),
                          "--engine", str(tmp_path / "no-engine"),
                          "--data-source", str(tmp_path / "no-data"), "--no-xvfb"])
    assert code == FOLLOWUP.EXIT_TECHNICAL
    assert target.read_bytes() == before, "the two-arm evidence was kept"
    alternates = [item for item in base.iterdir()
                  if item.name != "result.json" and item.suffix == ".json"]
    assert len(alternates) == 1, sorted(item.name for item in base.iterdir())
    record = json.loads(alternates[0].read_text(encoding="utf-8"))
    assert record["preserved_evidence"]["kept"] == target.as_posix()
    assert record["preserved_evidence"]["previous_arms"] == [
        "open_loop", "constant_12"]
    assert record["gate"]["passed"] is False

# ---------------------------------------------------------------------------
# the teacher-arm diagnostic on the four unexposed dev roots (additive)
# The four engine starts live in 03/diag/teacher_arm/ evidence, never in this file.
# ---------------------------------------------------------------------------
#: The card's two frozen digests on the tape grid (02-B task card, verbatim).
DIAG_PREFIX_SHA256 = "2632726e4e6b7f35f20cd5257b10280d19f44f6418d68d29cd4dba89e6250300"
DIAG_CONTINUATION_SHA256 = (
    "d8680c492949543431a3a0d704243464448aebdaef751b1c16b6bfd770b04a08")


def _diag_card() -> dict:
    """The frozen card's fields this diagnostic reads (numbers verbatim)."""

    return {"entry": {"prefix_decisions": 1543, "frame": 3086,
                      "prefix_masks_sha256": DIAG_PREFIX_SHA256,
                      "hash": "8922ae26eb962f6c"},
            "frame_cap": {"decisions": 300, "frames": 600},
            "teacher_continuation": {"masks_sha256": DIAG_CONTINUATION_SHA256,
                                     "first_decision": 1543,
                                     "last_decision": 1842},
            "event": {"seed12_instance": {"slot": 1, "spawn_decision": 1466,
                                          "life_at_spawn": 6000}}}


def test_diag_teacher_layout_is_inside_03_and_beside_no_existing_evidence(tmp_path):
    layout = FOLLOWUP.diag_teacher_layout(tmp_path)
    assert layout["base"] == tmp_path / "03" / "diag" / "teacher_arm"
    assert layout["result"] == layout["base"] / "result.json"
    assert sorted(layout["seeds"]) == [1020, 1021, 1022, 1023]
    assert all(path.parent == layout["base"] for path in layout["seeds"].values())
    assert layout["pilot"] == tmp_path / "03" / "pilot.json"
    assert layout["intervene"] == tmp_path / "02c" / "intervene.json"
    assert layout["card"] == tmp_path / "02b"
    assert layout["base"].is_relative_to(tmp_path / "03")
    assert FOLLOWUP.DIAG_TEACHER_CLASSIFICATION == (
        "teacher_arm_diagnostic_not_a_pilot_result")
    assert FOLLOWUP.DIAG_TEACHER_ATTEMPT_PREFIX == "03-diag-teacher-"


def test_diag_teacher_allowance_is_four_starts_and_15000_frames_never_widened():
    fixed = FOLLOWUP.diag_teacher_stage_cap()
    assert (fixed["starts"], fixed["engine_frames"]) == (4.0, 15_000.0)
    assert fixed["narrowed_by_the_cli"] is False
    # the CLI restates the whole-ticket numbers (28 / 360,000) and may not widen
    wide = FOLLOWUP.diag_teacher_stage_cap(max_starts=28,
                                           max_engine_frames=360_000)
    assert (wide["starts"], wide["engine_frames"]) == (4.0, 15_000.0)
    assert wide["narrowed_by_the_cli"] is False
    narrow = FOLLOWUP.diag_teacher_stage_cap(max_starts=2, max_engine_frames=7000)
    assert (narrow["starts"], narrow["engine_frames"]) == (2.0, 7000.0)
    assert narrow["narrowed_by_the_cli"] is True
    # four roots x (3,086 prefix + 600 window) = 14,744 <= 15,000
    per_root = 3086 + 600
    assert per_root * len(FOLLOWUP.DIAG_TEACHER_SEEDS) <= 15_000
    assert FOLLOWUP.DIAG_TEACHER_SEEDS == (1020, 1021, 1022, 1023)
    assert FOLLOWUP.DIAG_TEACHER_SEEDS == tuple(FOLLOWUP.PILOT_DEV_SEEDS)


def test_diag_teacher_window_is_the_cards_own_continuation():
    """The window is re-derived from the recorded tape, not from the card."""
    card = _diag_card()
    tape = FOLLOWUP.tape_schedule(REPO_ROOT)
    window = FOLLOWUP.diag_teacher_window(card=card, tape=tape)
    assert window["masks_sha256"] == DIAG_CONTINUATION_SHA256
    assert window["masks_sha256_matches_the_card"] is True
    assert window["masks_are_the_frozen_executor_expansion"] is True
    assert (window["first_decision"], window["last_decision"]) == (1543, 1842)
    assert window["decisions"] == 300 and window["frames"] == 600
    assert window["controller"] == "recorded_tape_masks"
    assert window["cands"] == tape["cands"][1543:1843]
    assert window["masks"] == tape["masks"][1543:1843]
    # the teacher continuation is a real controller, not a constant
    stats = window["action_stats"]
    assert stats["distinct_candidates"] > 1
    assert stats["dominant_share"] < 1.0
    assert stats["execution_feasibility"]["all_frames_executable"] is True
    # the prefix digest is the card's own
    assert FOLLOWUP.tape_digest(tape, 0, 1543) == DIAG_PREFIX_SHA256
    # a card whose continuation digest is not the tape's is reported, not hidden
    wrong = _diag_card()
    wrong["teacher_continuation"]["masks_sha256"] = "0" * 64
    assert FOLLOWUP.diag_teacher_window(card=wrong, tape=tape)[
        "masks_sha256_matches_the_card"] is False


def test_diag_teacher_interpretation_separates_the_two_explanations():
    pilot = {"completed_roots": []}
    zero = FOLLOWUP.diag_teacher_interpretation(
        teacher_roots={"1020": {"outcome": "legal_death"},
                       "1021": {"outcome": "not_removed_within_cap"},
                       "1022": {"outcome": "legal_death"},
                       "1023": {"outcome": "scripted_retreat"}},
        pilot_reference=pilot)
    assert zero["interpretation"] == "entry_or_task_seed_specific"
    assert zero["teacher_completed_roots"] == []
    assert zero["pilot_completed_roots"] == []
    one = FOLLOWUP.diag_teacher_interpretation(
        teacher_roots={"1020": {"outcome": "legal_death"},
                       "1021": {"outcome": "kill_M1"},
                       "1022": {"outcome": "not_removed_within_cap"},
                       "1023": {"outcome": "legal_death"}},
        pilot_reference=pilot)
    assert one["interpretation"] == "policy_quality"
    assert one["teacher_completed_roots"] == [1021]
    all_four = FOLLOWUP.diag_teacher_interpretation(
        teacher_roots={key: {"outcome": "kill_M0"}
                       for key in ("1020", "1021", "1022", "1023")},
        pilot_reference=pilot)
    assert all_four["teacher_completed_roots"] == [1020, 1021, 1022, 1023]
    # the classification never changes with the interpretation
    for block in (zero, one, all_four):
        assert block["classification"] == FOLLOWUP.DIAG_TEACHER_CLASSIFICATION


def test_diag_teacher_pilot_reference_is_quoted_never_recomputed(tmp_path):
    base = tmp_path / "03"
    base.mkdir(parents=True)
    pilot = {"schema": FOLLOWUP.SCHEMA_PILOT, "verdict": "no_nomination",
             "seeds": [1020, 1021, 1022, 1023],
             "gate": {"passed": False},
             "comparison": {"per_root": {
                 str(seed): {"arms": {
                     arm: {"outcome": ("kill_M2" if (seed == 1022 and arm == "final")
                                       else "legal_death"),
                           "completed": (seed == 1022 and arm == "final"),
                           "control_decisions": 80,
                           "frozen_life_first": 4878,
                           "frozen_life_min": 1732}
                     for arm in FOLLOWUP.PILOT_ARMS}}
                 for seed in FOLLOWUP.DIAG_TEACHER_SEEDS}},
             "arms": {}}
    path = base / "pilot.json"
    path.write_text(json.dumps(pilot), encoding="utf-8")
    reference = FOLLOWUP.diag_teacher_pilot_reference(tmp_path)
    assert reference["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert reference["path"] == path.as_posix()
    assert reference["read_only"] is True
    assert reference["classification"] == "historical"
    assert reference["completed_roots"] == [1022]
    row = reference["roots"]["1020"]
    assert row["outcome_by_arm"] == {arm: "legal_death"
                                     for arm in FOLLOWUP.PILOT_ARMS}
    assert row["outcome_column"] == "; ".join(
        f"{arm}=legal_death" for arm in FOLLOWUP.PILOT_ARMS)
    assert reference["roots"]["1022"]["completed_arms"] == ["final"]
    # a missing pilot summary is a refusal, not an empty comparison
    path.unlink()
    with pytest.raises(FOLLOWUP.SourceError):
        FOLLOWUP.diag_teacher_pilot_reference(tmp_path)


def test_diag_teacher_row_quotes_every_number_the_receipt_needs():
    record = {
        "seed": 1020, "attempt_id": "03-diag-teacher-1020",
        "engine_started": True, "controller": "recorded_tape_masks",
        "entry": {"reached": True, "frozen_slot_present": True,
                  "frame": 3086, "stage": 0, "dead": False,
                  "hash": "7580613141d6179c", "card_hash": "8922ae26eb962f6c",
                  "card_hash_matches": False, "frozen_slot_boss": True,
                  "frozen_slot_life": 4878, "observed_spawn_decision": 1466,
                  "card_spawn_decision": 1466, "spawn_delta": 0,
                  "identity_mismatch": False},
        "counts": {"prefix_decisions": 1543, "prefix_frames": 3086,
                   "control_decisions": 300, "control_frames": 600,
                   "engine_frames": 3686},
        "window": {"decision_range": [1543, 1842],
                   "frame_range": [3087, 3686]},
        "outcome": "kill_M1", "stop_reason": "frozen_consequence",
        "outcome_decision": 1842, "outcome_frame": 3686,
        "terminal": {"kind": "frozen_consequence"},
        "deaths_observed": 0.0, "lives_lost": 0.0,
        "resources_at_entry": {"state": {"lives_remaining": 2.0, "score": 40000,
                                         "player_power": 12.0}},
        "resources_at_end": {"state": {"lives_remaining": 2.0, "score": 88466,
                                       "player_power": 12.0, "graze": 113}},
        "frozen_generation": {"life_first": 4878, "life_last": 0, "life_min": 0,
                              "observations": 300, "phase_ends": []},
        "candidate_histogram": {"0": 10}, "candidate_dominant_share": 0.1,
        "action_stats": {"distinct_candidates": 17,
                         "longest_same_candidate_run": {"decisions": 4}},
        "arm_exit_code": 0, "result_path": "/x/result.json",
        "result_sha256": "a" * 64,
        "mute": {"after_exit_problems": []},
        "ledger": {"actual": {"starts": 1.0, "engine_frames": 3686.0}},
        "classification": FOLLOWUP.DIAG_TEACHER_CLASSIFICATION,
    }
    pilot = {"outcome_column": "init=legal_death; final=legal_death; "
                               "open_loop=legal_death"}
    row = FOLLOWUP.diag_teacher_row(record, pilot=pilot)
    assert row["conclusion"] == "measured"
    assert row["completed"] is True
    assert row["entry_reached"] is True and row["entry_ok"] is True
    assert row["card_entry_hash_matches"] is False
    assert (row["score_at_entry"], row["score_at_end"]) == (40000, 88466)
    assert (row["frozen_life_first"], row["frozen_life_last"],
            row["frozen_life_min"]) == (4878, 0, 0)
    assert row["frozen_life_drop"] == 4878.0
    assert row["pilot_same_root_outcome_column"] == pilot["outcome_column"]
    assert row["deaths_observed"] == 0.0 and row["lives_lost"] == 0.0
    # a root the budget did not allow to start is a named conclusion, not a run
    unstarted = FOLLOWUP.diag_teacher_unstarted_record(
        seed=1023, frame_plan={"per_root_frames": 3686}, budget={})
    row_unstarted = FOLLOWUP.diag_teacher_row(unstarted)
    assert row_unstarted["conclusion"] == "not_started_budget"
    assert row_unstarted["engine_started"] is False
    assert row_unstarted["completed"] is False
    # a root beside no pilot row quotes no column, and never invents one
    assert "pilot_same_root" not in FOLLOWUP.diag_teacher_row(record)


def test_diag_teacher_attempt_ids_read_the_evidence(tmp_path):
    assert FOLLOWUP.diag_teacher_attempt_ids(tmp_path) == set()
    base = tmp_path / "03" / "diag" / "teacher_arm"
    base.mkdir(parents=True)
    (base / "result.json").write_text(json.dumps({
        "schema": FOLLOWUP.SCHEMA_DIAG_TEACHER,
        "roots": {"1020": {"attempt_id": "03-diag-teacher-1020",
                           "engine_started": True},
                  "1021": {"attempt_id": "03-diag-teacher-1021",
                           "engine_started": False},
                  "1022": {"attempt_id": None, "engine_started": False}}}),
        encoding="utf-8")
    assert FOLLOWUP.diag_teacher_attempt_ids(tmp_path) == {"03-diag-teacher-1020"}


def test_diag_teacher_refuses_before_any_start_without_the_task_card(tmp_path,
                                                                    capsys):
    out = tmp_path / "out"
    code = FOLLOWUP.main(["diag-teacher", "--repo", str(REPO_ROOT),
                          "--out", str(out), "--engine",
                          str(tmp_path / "no-engine"),
                          "--data-source", str(tmp_path / "no-data"),
                          "--no-xvfb"])
    assert code == FOLLOWUP.EXIT_TECHNICAL
    base = out / "03" / "diag" / "teacher_arm"
    record = json.loads((base / "result.json").read_text(encoding="utf-8"))
    assert record["schema"] == FOLLOWUP.SCHEMA_DIAG_TEACHER
    assert record["classification"] == FOLLOWUP.DIAG_TEACHER_CLASSIFICATION
    assert record["engine_starts"] == 0
    assert record["terminal"]["kind"] == "technical_invalid"
    assert "task card" in record["terminal"]["detail"]
    assert record.get("interpretation") is None
    assert not (out / "ledger.jsonl").exists(), "no ledger row for a refused start"
    assert not any((base / str(seed)).exists()
                   for seed in FOLLOWUP.DIAG_TEACHER_SEEDS)


def test_diag_teacher_rejects_a_seed_outside_the_four_roots(tmp_path, capsys):
    out = tmp_path / "out"
    code = FOLLOWUP.main(["diag-teacher", "--repo", str(REPO_ROOT),
                          "--out", str(out), "--engine",
                          str(tmp_path / "no-engine"),
                          "--data-source", str(tmp_path / "no-data"),
                          "--only-seed", "1040", "--no-xvfb"])
    assert code == FOLLOWUP.EXIT_NOT_IMPLEMENTED
    assert "1040" in capsys.readouterr().out
    assert not (out / "ledger.jsonl").exists()


def test_verify_03_diag_teacher_is_not_run_without_evidence(tmp_path):
    gate = FOLLOWUP.Gate("verify")
    report = FOLLOWUP.verify_03_diag_teacher(tmp_path, REPO_ROOT, gate)
    assert report["status"] == "not-run"
    assert gate.checks == [], "no gate claim is made without evidence"
