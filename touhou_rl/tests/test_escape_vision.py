"""Vision front-end and student tests (D10–D12 skeleton, synthetic frames)."""

from __future__ import annotations

import numpy as np
import pytest

from touhou_rl.escape import families, student, synth, vision


def test_render_state_draws_hazards_and_player():
    scenario = synth.Scenario(
        "wall/0", "TEST", 0, (192.0, 384.0),
        (synth.HazardSpec((300.0, 200.0), (0.0, 0.0), (8.0, 8.0)),),
        horizon_frames=10, hold_frames=0,
    )
    image = vision.render_state(scenario, 0, 192.0, 384.0)
    assert image.shape == vision.IMAGE_SHAPE
    assert image.dtype == np.float32
    assert image.max() >= 1.0  # a hazard pixel
    assert np.isclose(image[image > 0.4].min(), 0.5, atol=1e-6)  # the player pixel


def test_collect_teacher_returns_images_and_labels():
    dataset = student.collect_teacher(
        student.StudentConfig(
            episodes=2,
            planner=student.controller.ControllerConfig(
                planner=student.planner.PlannerConfig(horizon_decisions=4, continuation_decisions=1, beam_width=1)
            ),
        )
    )
    assert dataset["images"].shape[0] == dataset["labels"].shape[0] > 0
    assert dataset["images"].shape[1] == 1  # default: single frame
    assert dataset["images"].shape[2:] == vision.IMAGE_SHAPE
    assert dataset["labels"].min() >= 0 and dataset["labels"].max() < 18


def test_student_trains_and_beats_chance():
    pytest.importorskip("torch")
    document = student.train_and_evaluate(
        student.StudentConfig(
            episodes=3,
            epochs=6,
            seed=3,
            device="cpu",
            planner=student.controller.ControllerConfig(
                planner=student.planner.PlannerConfig(horizon_decisions=4, continuation_decisions=1, beam_width=1)
            ),
        )
    )
    assert document["synthetic_frames"] is True
    assert document["losses"][-1] < document["losses"][0]
    assert document["train"]["accuracy"] > 1.0 / 18.0
    assert document["test"]["samples"] > 0


def test_closed_loop_runs_with_student_actions():
    pytest.importorskip("torch")
    model = student.StudentNet("cpu", seed=1)
    scenario = families.edge_instances()["E-01"]
    result = student.closed_loop_student(model, scenario)
    assert result["scenario"] == "E-01"
    assert result["frames"] > 0
    assert isinstance(result["action_histogram"], dict)


def test_vision_masks_the_hud_area():
    """The rasteriser only draws the playfield, never the HUD strip."""
    scenario = synth.Scenario("empty/0", "TEST", 0, (192.0, 384.0), (),
                              horizon_frames=4, hold_frames=0)
    image = vision.render_state(scenario, 0, 192.0, 384.0)
    assert image.shape == vision.IMAGE_SHAPE
    # Only the player marker is non-zero on an empty field.
    non_zero = np.argwhere(image > 0.0)
    assert non_zero.shape[0] <= 9


def test_render_pair_stacks_two_frames():
    scenario = families.edge_instances()["E-05"]
    pair = vision.render_pair(scenario, 10, (190.0, 384.0), (192.0, 384.0))
    assert pair.shape == (2,) + vision.IMAGE_SHAPE
    assert pair.dtype == np.float32
    # Hazards move between the two frames, so the channels differ.
    assert not np.array_equal(pair[0], pair[1])
