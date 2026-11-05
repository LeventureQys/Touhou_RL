"""Synthetic contract fixtures only: no real capture/F2/G3 acceptance claim."""
from copy import deepcopy
import hashlib
import pytest

from touhou_rl.f2_labels import (audit_observation, build_label, evidence_sha,
                                 validate_event_annotation, validate_label, label_to_tensors)

SHA = "a" * 64


def fixture(mask=1):
    images = [{"frame": f, "sha256": str(f + 1) * 64} for f in range(4)]
    tracks = [{"id": "b1", "samples": [
        {"frame": 0, "xy": [10, 10], "visible": True, "image_sha256": images[0]["sha256"]},
        {"frame": 1, "xy": [10, 20], "visible": True, "image_sha256": images[1]["sha256"]}]},
        {"id": "b2", "samples": [
        {"frame": 2, "xy": [10, 10], "visible": True, "image_sha256": images[2]["sha256"]},
        {"frame": 3, "xy": [10, 20], "visible": True, "image_sha256": images[3]["sha256"]}]}]
    o = dict(frame=3, observed_through=3, units="image_px", image_size=[384, 448],
             player_px=[10, 100], source={k: {"id": k, "sha256": SHA} for k in ("capture", "proof")},
             images=images, bullet_tracks=tracks)
    a = dict(observation_sha256=evidence_sha(o), family="F2", event_type="aimed_stream",
             reviewer={"id": "independent-human", "method": "independent_pixel_annotation"},
             teacher={"producer_id": "memory-tracker", "artifact_sha256": SHA, "tracks": deepcopy(tracks)},
             tolerances={"position_px": 1., "aim_cross_px_per_frame": .1,
                         "aim_miss_px": .5, "aim_lookback_frames": 2, "emission_recency_frames": 4},
             pixel_tracks=deepcopy(tracks),
             emissions=[dict(frame=f, track_id=tr, emitter_id="enemy1", aim_player_frame=f,
                             aim_image_sha256=images[f]["sha256"], origin_px=[10, 10],
                             aim_player_px=[10, 100], velocity_px_per_frame=[0, 10], miss_px=0.,
                             player_pixel_annotation={"frame": f, "image_sha256": images[f]["sha256"], "xy": [10, 100]})
                        for f, tr in ((0, "b1"), (2, "b2"))])
    payload = b"\x01\x00" * 3 + mask.to_bytes(2, "little")
    r = dict(source="replay_decoder_alignment", extended=False, logic_frame=3, record_index=3,
             observation_frame=3, consumed_mask=1, previous_mask=1,
             replay_sha256=SHA, payload_sha256=hashlib.sha256(payload).hexdigest(), payload_hex=payload.hex())
    return o, a, r


def test_valid_demonstration_is_not_gap_or_safety_label():
    row = build_label(*fixture())
    assert validate_label(row) == []
    assert row["action_positive"] == [True] + [False] * 17
    assert row["label_status"] == "demonstration"
    for key in ("deadline", "route", "gap", "gap_center", "relative_dx"):
        assert row[key] is None and row[key + "_valid"] is False


@pytest.mark.parametrize("units", ["normalized", "world", "px", None])
def test_illegal_units(units):
    o, a, r = fixture()
    o["units"] = units
    assert audit_observation(o)
    assert not build_label(o, a, r)["action_valid"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 384, True])
def test_invalid_coordinate(value):
    o, _, _ = fixture()
    o["player_px"][0] = value
    assert audit_observation(o)


def test_future_observation_and_sample():
    o, a, r = fixture()
    o["observed_through"] = 4
    assert audit_observation(o)
    o, a, r = fixture()
    o["bullet_tracks"][0]["samples"][0]["frame"] = 4
    assert audit_observation(o)


@pytest.mark.parametrize("family", ["F1", "F3", "F2 verified"])
def test_other_family_not_f2(family):
    o, a, r = fixture()
    a["family"] = family
    assert validate_event_annotation(a, o)
    assert build_label(o, a, r)["label_status"] == "unknown"


def test_verified_string_is_not_visual_evidence():
    o, a, r = fixture()
    a = {"family": "F2", "verified": "verified"}
    row = build_label(o, a, r)
    assert row["requires_review"] and not row["action_valid"]
    row = build_label(o, None, r)
    assert validate_label(row) == []
    row["action_positive"][0] = True
    assert validate_label(row)
    with pytest.raises(ValueError):
        label_to_tensors([row])


@pytest.mark.parametrize("key", ["position_px", "aim_cross_px_per_frame", "aim_miss_px",
                                 "aim_lookback_frames", "emission_recency_frames"])
def test_explicit_tolerance_required(key):
    o, a, r = fixture()
    del a["tolerances"][key]
    assert validate_event_annotation(a, o)


def test_spent_volley_outside_the_current_stream_is_rejected():
    """A row cannot be justified by aimed bullets fired long before it."""
    o, a, r = fixture()
    a["tolerances"]["emission_recency_frames"] = 2  # track b1 is emitted at frame 0, row is 3
    errors = validate_event_annotation(a, o)
    assert any("recency" in e for e in errors), errors


@pytest.mark.parametrize("value", [-1, 1.5, "2", None, True])
def test_aim_lookback_must_be_a_nonnegative_int(value):
    o, a, r = fixture()
    a["tolerances"]["aim_lookback_frames"] = value
    assert validate_event_annotation(a, o)


def test_aim_is_a_miss_distance_not_a_speed_ratio():
    """A 45-degree aim error at bullet speed stays under the old speed bound.

    The removed criterion compared ``|cross|/|aim-origin|`` (a cross-track
    *speed*) with a speed-sized tolerance, so a shot passing ~85 px to one side
    of the player still "aimed".  The contract must reject it on geometry alone.
    """
    o, a, r = fixture()
    # Player moved 85 px to the side: the path no longer passes the hitbox.
    a["emissions"][0]["aim_player_px"] = [95, 100]
    a["emissions"][0]["player_pixel_annotation"]["xy"] = [95, 100]
    a["tolerances"]["position_px"] = 90.  # keep the position checks out of the way
    a["tolerances"]["aim_miss_px"] = 1.5
    errors = validate_event_annotation(a, o)
    assert any("miss" in e for e in errors), errors


def test_declared_miss_must_match_privileged_geometry():
    o, a, r = fixture()
    a["emissions"][0]["miss_px"] = 0.4
    assert any("declared aim miss" in e for e in validate_event_annotation(a, o))
    o, a, r = fixture()
    del a["emissions"][0]["miss_px"]
    assert any("declared aim miss" in e for e in validate_event_annotation(a, o))


def test_aim_frame_lookback_is_bounded():
    """Searching further into the past for a near player position is refused."""
    o, a, r = fixture()
    a["emissions"][0]["aim_player_frame"] = 0  # the emission frame itself: fine
    assert validate_event_annotation(a, o) == []
    o, a, r = fixture()
    a["tolerances"]["aim_lookback_frames"] = 0
    # Track b2 is emitted at frame 2; cite the frame-0 player position instead.
    a["emissions"][1]["aim_player_frame"] = 0
    a["emissions"][1]["aim_image_sha256"] = o["images"][0]["sha256"]
    a["emissions"][1]["player_pixel_annotation"] = {"frame": 0, "image_sha256": o["images"][0]["sha256"], "xy": [10, 100]}
    errors = validate_event_annotation(a, o)
    assert any("lookback" in e for e in errors), errors


def test_aim_frame_after_emission_rejected():
    o, a, r = fixture()
    a["emissions"][0]["aim_player_frame"] = 1
    a["emissions"][0]["aim_image_sha256"] = o["images"][1]["sha256"]
    a["emissions"][0]["player_pixel_annotation"] = {"frame": 1, "image_sha256": o["images"][1]["sha256"], "xy": [10, 100]}
    errors = validate_event_annotation(a, o)
    assert any("causal track sample" in e or "lookback" in e for e in errors), errors


def test_independent_teacher_check_and_image_binding():
    o, a, r = fixture()
    a["teacher"]["tracks"][0]["samples"][0]["xy"] = [30, 10]
    assert validate_event_annotation(a, o)
    o, a, r = fixture()
    a["reviewer"]["id"] = a["teacher"]["producer_id"]
    assert validate_event_annotation(a, o)
    o, a, r = fixture()
    a["pixel_tracks"][0]["samples"][0]["image_sha256"] = SHA
    assert validate_event_annotation(a, o)


def test_fabricated_aim_and_unbound_annotation_rejected():
    o, a, r = fixture()
    a["emissions"][0]["velocity_px_per_frame"] = [0, -10]
    assert validate_event_annotation(a, o)
    o, a, r = fixture()
    o["source"]["capture"]["id"] = "other"
    assert validate_event_annotation(a, o)


@pytest.mark.parametrize("mask,candidate", [(1, 0), (0x55, 17)])
def test_actual_candidate_boundaries(mask, candidate):
    row = build_label(*fixture(mask))
    assert validate_label(row) == []
    assert row["action_positive"][candidate]
    row["action_positive"] += [False]
    assert validate_label(row)


@pytest.mark.parametrize("mask", [0, 3, 0x31, 0xC1, 0x101, 0x09])
def test_outside_actual_candidate_set_unknown(mask):
    row = build_label(*fixture(mask))
    assert row["label_status"] == "unknown" and not any(row["action_positive"])


@pytest.mark.parametrize("mutation", ["inputReadback", "unaligned", "out_of_range", "bad_sha", "wrong_consumed"])
def test_unaligned_actions_never_supervised(mutation):
    o, a, r = fixture()
    if mutation == "inputReadback":
        r["source"] = mutation
    elif mutation == "unaligned":
        r["logic_frame"] = 2
    elif mutation == "out_of_range":
        r["record_index"] = r["logic_frame"] = 99
    elif mutation == "wrong_consumed":
        # A record index that is not the input the frame actually consumed.
        r["consumed_mask"] = 0x40
    else:
        r["payload_sha256"] = SHA
    row = build_label(o, a, r)
    assert row["label_status"] == "unknown" and validate_label(row) == []


def test_unknown_every_head_has_zero_gradient():
    torch = pytest.importorskip("torch")
    from touhou_rl.tactical_policy import tactical_policy_loss
    o, a, r = fixture()
    labels = label_to_tensors([build_label(o, None, r)])
    output = {k: torch.randn(*shape, requires_grad=True) for k, shape in {
        "policy_logits": (1, 18), "option_logits": (1, 3), "target": (1, 2),
        "termination_logits": (1, 3), "slack": (1,)}.items()}
    output["base_logits"] = torch.zeros(1, 18)
    loss = tactical_policy_loss(output, labels)
    assert not loss["has_supervision"] and all(n == 0 for n in loss["counts"].values())
    loss["loss"].backward()
    for key in ("policy_logits", "option_logits", "target", "termination_logits", "slack"):
        assert torch.count_nonzero(output[key].grad) == 0


def test_bc_only_action_gradient_and_revalidated_actual_action():
    torch = pytest.importorskip("torch")
    from touhou_rl.tactical_policy import tactical_policy_loss
    row = build_label(*fixture())
    labels = label_to_tensors([row])
    output = {k: torch.randn(*shape, requires_grad=True) for k, shape in {
        "policy_logits": (1, 18), "option_logits": (1, 3), "target": (1, 2),
        "termination_logits": (1, 3), "slack": (1,)}.items()}
    output["base_logits"] = torch.zeros(1, 18)
    result = tactical_policy_loss(output, labels)
    assert result["counts"]["action"] == 1 and sum(result["counts"].values()) == 1
    result["loss"].backward()
    assert torch.count_nonzero(output["policy_logits"].grad) > 0
    for key in ("option_logits", "target", "termination_logits", "slack"):
        assert torch.count_nonzero(output[key].grad) == 0
    row["action_positive"] = [False, True] + [False] * 16
    assert validate_label(row)
