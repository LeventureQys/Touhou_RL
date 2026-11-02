import numpy as np
import pytest

from touhou_rl.chunk_safety import ChunkSafetyConfig, predict

SCHEMA = "v200-raw-safety-snapshot/1"

def snap(player=(192., 384.), pos=(), vel=(), half=(), complete=True):
    return {"schema": SCHEMA, "player_position": np.asarray(player),
            "bullets": {"position": np.asarray(pos, dtype=float).reshape(-1, 2),
                        "velocity": np.asarray(vel, dtype=float).reshape(-1, 2),
                        "half": np.asarray(half, dtype=float)},
            "complete": complete, "frame": 10, "stage": 1}

def test_empty_field_shapes_and_validity():
    r = predict(snap())
    assert r["clearance"].shape == (18, 3)
    assert r["clearance"].dtype == np.float32
    assert np.all(r["clearance"] > 0)
    assert np.all(r["model_valid"] == 1)
    assert np.all(r["duration_valid"] == 1)
    assert np.all(r["duration_class"] == 2)

def test_approaching_and_fast_crossing_detected():
    r = predict(snap(pos=[[192, 370]], vel=[[0, 4]], half=[1]))
    assert np.any(r["collision_predicted"][:, 0] == 1)
    r = predict(snap(pos=[[150, 384]], vel=[[50, 0]], half=[1]))
    assert np.any(r["collision_predicted"][:, 0] == 1)

def test_boundary_clip_focus_diagonal_and_monotonic():
    # Position outside the legal area is clipped, and focused motion is slower.
    a = predict(snap(player=[-100, 100]))
    b = predict(snap(player=[-100, 100]), ChunkSafetyConfig(focus_speed=1, focus_speed_diag=.7071))
    assert np.all(a["duration_valid"] == 1)
    assert b["clearance"][9 + 2, 0] >= a["clearance"][9 + 2, 0]
    assert np.all(np.diff(a["clearance"], axis=1) <= 1e-5)

def test_invalid_inputs_and_negative_radius():
    r = predict({})
    assert np.all(r["model_valid"] == 0)
    assert np.all(np.isnan(r["clearance"]))
    assert np.all(r["duration_class"] == -1)
    assert np.all(predict(snap(complete=False))["model_valid"] == 0)
    with pytest.raises(ValueError):
        predict(snap(half=[-1]))
    with pytest.raises(ValueError):
        predict(snap(), ChunkSafetyConfig(frameskip=0))

def test_raw_bullets_not_capped():
    pos = [[1000 + i, 384] for i in range(400)]
    pos[399] = [192, 384]
    r = predict(snap(pos=pos, vel=[[0, 0]] * 400, half=[0] * 400))
    assert np.all(r["collision_predicted"] == 1)

def test_structural_errors_raise():
    x = snap(); x["bullets"]["velocity"] = np.zeros((2, 2))
    with pytest.raises(ValueError):
        predict(x)
