from pathlib import Path

from touhou_rl.v400.gym.replay_starts import build_start_index


REPLAY = Path(__file__).parents[1] / "th08_replay" / "Th8_ud0059.rpy"


def test_real_replay_start_index_is_provenance_rich():
    rows = build_start_index([REPLAY])
    assert len(rows) == sum((count + 599) // 600 for count in (16665, 18942, 22305, 26820, 30854, 35524))
    assert {row["stage_index"] for row in rows} == {0, 1, 2, 4, 5, 7}
    assert all(row["stage_frame"] % 600 == 0 for row in rows)
    assert all(row["sample_every"] == 600 for row in rows)
    assert all(row["source_sha256"] == "62b02a172780d846f4109fd0dcc762692c5205692670c0f7400c2026c409a5e9" for row in rows)
    assert all(row["input_records"] > 0 and row["takeover_supported"] is False for row in rows)
    assert rows[-1]["global_frame"] == 115586 + 600 * ((35524 - 1) // 600)
    assert rows[0]["resource_metadata"]["codec"] == "TH08-T8RP-v6-LZSS"
