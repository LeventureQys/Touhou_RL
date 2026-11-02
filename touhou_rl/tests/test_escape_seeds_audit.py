"""Seed list and trace-audit tests (pure, no game).

The seed test pins the *rule*, not just the numbers: if someone changes the
derivation, the formal list changes and every pre-registered denominator changes
with it, which must be a deliberate, visible edit.
"""

from __future__ import annotations

import json
from pathlib import Path

from touhou_rl.escape import audit, seeds

ROOT = Path(__file__).resolve().parents[2]


def test_seed_lists_are_deterministic_and_disjoint():
    first = seeds.build()
    second = seeds.build()
    assert first["digest"] == second["digest"]
    for family in first["families"]:
        assert len(family["seeds"]) == 30
        assert len(set(family["seeds"])) == 30
        assert family["dev"] == family["seeds"][:12]
        assert family["formal"] == family["seeds"][12:]
    # A different master must produce a different list.
    other = seeds.build(master=1)
    assert other["families"][0]["seeds"] != first["families"][0]["seeds"]


def test_seventeen_families_and_twenty_edge_instances():
    document = seeds.build()
    assert [family["family"] for family in document["families"]] == [
        f"SC-{index:02d}" for index in range(1, 18)
    ]
    assert len(document["edge_instances"]) == 20
    assert len({item["id"] for item in document["edge_instances"]}) == 20
    targets = {item["target"] for item in document["edge_instances"]}
    assert targets <= {family["family"] for family in document["families"]}


def test_committed_seed_file_matches_the_rule():
    path = ROOT / "Document/Update/Alpha/v3.7.0/preregistration/seeds-v1.json"
    if not path.is_file():
        import pytest

        pytest.skip("seed file not written yet")
    committed = json.loads(path.read_text(encoding="utf-8"))
    assert committed == seeds.build()


def _write_trace(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


def _record(frame: int, vx: float, vy: float, count: int, counter: int, complete: bool = True) -> dict:
    return {
        "frame_pre": frame,
        "frame_post": frame + 2,
        "complete": complete,
        "incomplete_reasons": [] if complete else ["frame_mismatch"],
        "schema_status": {"player": "ok", "bullets": "ok", "enemies": "empty"},
        "player": {"vx": vx, "vy": vy},
        "bullets": {"count": count, "counter": counter},
    }


def test_audit_reports_frames_speeds_and_pool_delta(tmp_path):
    good = _write_trace(
        tmp_path / "good.jsonl",
        [
            _record(10, 4.0, 0.0, 10, 10),
            _record(12, 2.8284, -2.8284, 12, 12),
            _record(14, 0.0, 0.0, 0, 4),
        ],
    )
    report = audit.audit_episode(good)
    assert report["decisions"] == 3
    assert report["complete"] == 3
    assert report["frame_advance_histogram"] == {"2": 3}
    assert report["speed_class_histogram"]["axis"] == 1
    assert report["speed_class_histogram"]["diagonal"] == 1
    assert report["speed_class_histogram"]["idle"] == 1
    assert report["pool_delta"]["exact_equal"] == 2
    assert report["pool_delta"]["min"] == -4
    assert report["pool_delta"]["max"] == 0
    assert report["first_decision"]["has_player"] is True


def test_audit_flags_incomplete_and_off_model_speed(tmp_path):
    bad = _write_trace(
        tmp_path / "bad.jsonl",
        [_record(10, 3.0, 0.0, 5, 5, complete=False)],
    )
    report = audit.audit_episode(bad)
    assert report["complete_ratio"] == 0.0
    assert report["incomplete_reasons"] == {"frame_mismatch": 1}
    assert "other:(3.000,0.000)" in report["speed_class_histogram"]


def test_audit_aggregates_multiple_episodes(tmp_path):
    first = _write_trace(tmp_path / "a.jsonl", [_record(10, 4.0, 0.0, 1, 1)])
    second = _write_trace(tmp_path / "b.jsonl", [_record(10, 2.0, 0.0, 1, 1, complete=False)])
    report = audit.audit_paths([first, second])
    assert report["totals"]["episodes"] == 2
    assert report["totals"]["decisions"] == 2
    assert report["totals"]["complete_ratio"] == 0.5
    assert report["totals"]["off_model_ratio"] == 0.0
