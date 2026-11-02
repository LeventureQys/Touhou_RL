"""Shadow re-check tests on a synthetic trace + sidecar (no game files needed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from touhou_rl.escape import shadow


def _write_episode(root: Path, *, bullet_x: float, bullet_vx: float, teacher: float):
    records = [
        {
            "schema": "v350-trace/1",
            "episode": "synthetic",
            "decision": decision,
            "frameskip": 2,
            "frame_pre": frame - 2,
            "frame_post": frame,
            "player": {"x": 192.0, "y": 384.0, "vx": 0.0, "vy": 0.0, "state": 0, "focus_mode": 0,
                       "hurtbox_half_x": 0.825, "hurtbox_half_y": 0.825},
            "resources": {"deaths": 0},
            "bullets": {"count": 1, "counter": 1, "sidecar_index": decision - 1},
            "teacher": {"clearance": teacher, "bullets_used": 1},
            "controller": {"policy": "teacher"},
            "action": [0, 1, 0, 0],
            "events": [],
        }
        for decision, frame in ((1, 10), (2, 12), (3, 14))
    ]
    trace = root / "synthetic.jsonl"
    trace.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    sidecar = root / "synthetic.npz"
    np.savez(
        sidecar,
        positions=np.array([[bullet_x, 384.0]], dtype=np.float32),
        velocities=np.array([[bullet_vx, 0.0]], dtype=np.float32),
        half=np.array([2.0], dtype=np.float32),
        slots=np.array([0], dtype=np.int32),
        offsets=np.array([0, 1, 1, 1], dtype=np.int64),
        counters=np.array([1, 1, 1], dtype=np.int32),
    )
    return trace, sidecar


def test_frame_collision_reported_safe_is_detected(tmp_path):
    trace, sidecar = _write_episode(tmp_path, bullet_x=186.0, bullet_vx=6.0, teacher=5.0)
    report = shadow.shadow_episode(trace, sidecar)
    assert report["decisions_checked"] == 1
    assert report["frame_collisions"] == 1
    assert report["reported_safe_but_frame_collision"] == 1
    assert report["phase_exact_ratio"] == 1.0
    event = report["events"][0]
    assert event["frame_clearance"] <= 0.0
    assert event["frame_index"] == 1
    assert event["teacher_clearance"] == 5.0


def test_passing_bullet_is_not_a_collision(tmp_path):
    trace, sidecar = _write_episode(tmp_path, bullet_x=186.0, bullet_vx=6.0, teacher=5.0)
    # Move the bullet well above the player: same speed, no contact.
    sidecar_data = dict(np.load(sidecar))
    sidecar_data["positions"] = np.array([[186.0, 300.0]], dtype=np.float32)
    np.savez(sidecar, **sidecar_data)
    report = shadow.shadow_episode(trace, sidecar)
    assert report["decisions_checked"] == 1
    assert report["frame_collisions"] == 0


def test_missed_death_window(tmp_path):
    trace, sidecar = _write_episode(tmp_path, bullet_x=186.0, bullet_vx=6.0, teacher=5.0)
    # Mark a death two decisions after the collision: it counts as "seen".
    lines = trace.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[2])
    payload["events"] = ["death"]
    lines[2] = json.dumps(payload)
    trace.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = shadow.shadow_episode(trace, sidecar)
    assert report["collisions_within_3_decisions_of_death"] == 1
    assert report["death_decisions"] == [3]


def test_tunnelling_shows_up_as_swept_only(tmp_path):
    trace, sidecar = _write_episode(tmp_path, bullet_x=180.0, bullet_vx=24.0, teacher=5.0)
    report = shadow.shadow_episode(trace, sidecar)
    assert report["frame_collisions"] == 0
    assert report["swept_collisions"] == 1
    assert report["swept_only_collisions"] == 1
