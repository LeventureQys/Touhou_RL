"""Regression tests for the `pipeline train` report reader.

Background: v1.0.3 moved the vision trainer's summary to ``vision-summary/2``,
where the policy aggregates live in ``stages.policy.best.val_regret_px`` and
``stages.policy.final.val_top1`` instead of the old flat
``stages.policy.best_val_regret_px`` / ``stages.policy.final["top1"]`` keys.
`_flat_metrics` is the only consumer of that shape inside the pipeline stage, and
a silent rename there turns two report fields into ``null`` without any error.
These tests pin both shapes.
"""

from __future__ import annotations

from touhou_rl.pipeline.train import _flat_metrics

STATE_ROW = {
    "epoch": 8,
    "val_loss": 0.0673,
    "val_player_pos_px": 70.97,
    "val_danger_corr": 0.475,
    "val_bullet_count_mae": 15.1,
}


def test_flat_metrics_reads_v2_best_and_final_nodes():
    summary = {
        "schema": "vision-summary/2",
        "stages": {
            "state": {"history": [STATE_ROW], "best": STATE_ROW},
            "policy": {
                "history": [{"epoch": 1, "val_top1": 0.1, "val_regret_px": 4.0}],
                "best": {"epoch": 5, "val_top1": 0.11, "val_regret_px": 3.717},
                "final": {"epoch": 8, "val_top1": 0.1062, "val_regret_px": 3.786},
            },
        },
    }
    flat = _flat_metrics("vision", summary)
    assert flat["policy_best_regret_px"] == 3.717
    assert flat["policy_top1"] == 0.1062
    assert flat["player_pos_px"] == 70.97
    assert flat["danger_corr"] == 0.475


def test_flat_metrics_still_reads_the_v1_flat_shape():
    summary = {
        "stages": {
            "state": {"history": [STATE_ROW]},
            "policy": {"best_val_regret_px": 3.717, "final": {"top1": 0.1062}},
        }
    }
    flat = _flat_metrics("vision", summary)
    assert flat["policy_best_regret_px"] == 3.717
    assert flat["policy_top1"] == 0.1062


def test_flat_metrics_never_returns_none_for_a_populated_summary():
    summary = {
        "stages": {
            "state": {"history": [STATE_ROW]},
            "policy": {"best": {"val_regret_px": 3.7}, "final": {"val_top1": 0.2}},
        }
    }
    flat = _flat_metrics("vision", summary)
    assert all(value is not None for value in flat.values()), flat
