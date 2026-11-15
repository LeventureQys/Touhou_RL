"""Behavioural tests for the ``v430-contract/1`` data contract (W01 slice A/B).

Every assertion here pins a *behaviour* the work package promises:

* the 72-combination joint id is an exact bijection (72 distinct values);
* the four action heads are independent and their joint log-prob/entropy are the
  **sums** of hand-computable numbers, with no implicit SHOOT;
* the :class:`ActorObservation` shapes/dtypes/domains are refused when wrong, and
  privileged fields can never enter the actor;
* ``terminated`` / ``truncated`` / ``segment_end`` are three fields, and a
  bootstrap without ``terminal_observation_ref`` is refused;
* the result JSON keeps ``capability_gate`` at ``not_run`` unless an independent
  judgement says otherwise;
* ``configs/v430/contracts.json`` and the module constants agree in both
  directions.

Synthetic: this file proves the contract implementation.  It is **not** game
capability evidence and it starts no engine.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib

import pytest

from touhou_rl.v430 import contracts as C

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "v430" / "contracts.json"

#: ``ln 2``, the hand-computed value used by the probability tests below.
LN2 = math.log(2.0)


class _Array:
    """Minimal duck-typed array: shape + dtype, no numpy, no torch.

    The three image fields are larger than the leaf-scan limit, so their values
    are checked through the declared dtype exactly as a real uint8 tensor would
    be; only the small fields are also scanned element by element.
    """

    def __init__(self, shape, dtype, fill=0):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._fill = fill

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Array(shape={self.shape}, dtype={self.dtype})"


def valid_observation() -> dict:
    """A legal :class:`ActorObservation` (all fields present, all in domain)."""

    return {
        "global_rgb": _Array((3, 240, 320), "uint8"),
        "play_gray": _Array((4, 224, 192), "uint8"),
        "hud_gray": _Array((1, 448, 160), "uint8"),
        "executed_key_history": [[0] * 7 for _ in range(8)],
        "history_valid": [True] * 8,
        "elapsed_capture_s": 2.0 / 60.0,
        "memory": [0.0] * 256,
    }


def valid_transition(**overrides) -> dict:
    """A legal transition record; overrides replace single fields."""

    record = {
        "record_id": 0,
        "episode_id": 7,
        "obs_ref": "shard/0000/obs/0",
        "next_obs_ref": "shard/0000/obs/1",
        "requested_action": C.encode_joint(0, 0, 1, 0),
        "executed_masks": (0x0001, 0x0001),
        "controller_source": "actor",
        "terminated": False,
        "truncated": False,
        "segment_end": False,
        "terminal_observation_ref": None,
        "elapsed_game_frames": 2,
        "actor_loss_mask": 1,
        "value_loss_mask": 1,
        "privileged_ref": None,
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# 1. joint action: 72 combinations, exact bijection
# ---------------------------------------------------------------------------
def test_joint_encoding_is_a_bijection_of_exactly_72_values():
    seen = set()
    for move in range(9):
        for focus in (0, 1):
            for shoot in (0, 1):
                for bomb in (0, 1):
                    joint = C.encode_joint(move, focus, shoot, bomb)
                    assert C.decode_joint(joint) == (move, focus, shoot, bomb)
                    seen.add(joint)
    assert len(seen) == 72
    assert seen == set(range(72))
    assert C.JOINT_COUNT == 72
    assert C.joint_inverse() is True


def test_joint_formula_is_the_plan_spec_formula():
    for move in range(9):
        for focus in (0, 1):
            for shoot in (0, 1):
                for bomb in (0, 1):
                    expected = ((move * 2 + focus) * 2 + shoot) * 2 + bomb
                    assert C.encode_joint(move, focus, shoot, bomb) == expected
    # The first and last combination, spelled out.
    assert C.encode_joint(0, 0, 0, 0) == 0
    assert C.encode_joint(8, 1, 1, 1) == 71


def test_shoot_zero_and_shoot_one_are_different_actions():
    # No "always shoot" collapse: the two requests must never share a joint id.
    for move in range(9):
        for focus in (0, 1):
            for bomb in (0, 1):
                off = C.encode_joint(move, focus, 0, bomb)
                on = C.encode_joint(move, focus, 1, bomb)
                assert off != on
                assert (off - on) % 2 == 0  # they differ in the shoot bit only


@pytest.mark.parametrize(
    "move, focus, shoot, bomb",
    [
        (9, 0, 0, 0),        # move out of range
        (-1, 0, 0, 0),       # negative move
        (0, 2, 0, 0),        # focus is binary
        (0, 0, 3, 0),        # shoot is binary
        (0, 0, 0, -1),       # bomb is binary
        (0, True, 0, 0),     # a bool is not an action index
        (0, 0, False, 0),
        (1.0, 0, 0, 0),      # a float is not a move
        ("1", 0, 0, 0),
    ],
)
def test_joint_encoding_rejects_out_of_range_and_bools(move, focus, shoot, bomb):
    with pytest.raises(C.ContractError):
        C.encode_joint(move, focus, shoot, bomb)


@pytest.mark.parametrize("joint", [-1, 72, 1000, True, 3.5, "3"])
def test_joint_decoding_rejects_anything_outside_the_72(joint):
    with pytest.raises(C.ContractError):
        C.decode_joint(joint)


def test_joint_action_rejects_invalid_heads_at_construction():
    with pytest.raises(C.ContractError):
        C.JointAction(9, 0, 0, 0)
    assert C.JointAction(4, 1, 0, 1).joint == C.encode_joint(4, 1, 0, 1)
    assert len(list(C.iter_joint_actions())) == 72


# ---------------------------------------------------------------------------
# 2. joint probability: four independent heads, summed
# ---------------------------------------------------------------------------
def test_joint_logprob_is_the_sum_of_four_heads_hand_computed():
    # Four heads with probability 1/2 each: the joint is 1/16, log = -4 ln 2.
    assert C.check_joint_logprob(-LN2, -LN2, -LN2, -LN2) == pytest.approx(-4.0 * LN2)
    # Unequal heads: the sum is still just the sum (no renormalisation).
    assert C.check_joint_logprob(-0.5, -1.0, -2.0, -3.0) == pytest.approx(-6.5)
    # A certain head contributes exactly 0.
    assert C.check_joint_logprob(0.0, 0.0, -LN2, -LN2) == pytest.approx(-2.0 * LN2)
    assert C.joint_logprob_of(
        {"move": -LN2, "focus": -LN2, "shoot": -LN2, "bomb": -LN2}
    ) == pytest.approx(-4.0 * LN2)


def test_joint_entropy_is_the_sum_of_four_heads_hand_computed():
    # Four fair binary heads: 4 ln 2 nats.
    assert C.check_joint_entropy(LN2, LN2, LN2, LN2) == pytest.approx(4.0 * LN2)
    # A deterministic head contributes no entropy.
    assert C.check_joint_entropy(0.0, LN2, 0.0, 1.5) == pytest.approx(LN2 + 1.5)
    assert C.joint_entropy_of(
        {"move": LN2, "focus": 0.0, "shoot": LN2, "bomb": 0.0}
    ) == pytest.approx(2.0 * LN2)


def test_every_head_is_mandatory_so_shoot_cannot_be_assumed():
    # Omitting the shoot head would implicitly mean "shoot is certain"; the
    # contract refuses it instead of substituting 0.0 (that is the v4.2.0 defect).
    with pytest.raises(C.ContractError):
        C.check_joint_logprob(-LN2, -LN2, None, -LN2)
    with pytest.raises(C.ContractError):
        C.check_joint_entropy(LN2, LN2, None, LN2)
    with pytest.raises(C.ContractError):
        C.joint_logprob_of({"move": -LN2, "focus": -LN2, "bomb": -LN2})
    with pytest.raises(C.ContractError):
        C.joint_logprob_of({"move": -LN2, "focus": -LN2, "shoot": -LN2,
                            "bomb": -LN2, "extra": -LN2})


@pytest.mark.parametrize("bad", [0.5, 1.0, float("nan"), float("inf"), True, "0"])
def test_head_logprob_rejects_impossible_values(bad):
    with pytest.raises(C.ContractError):
        C.check_head_logprob(bad, "move")


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("-inf"), True])
def test_head_entropy_rejects_negative_or_non_finite(bad):
    with pytest.raises(C.ContractError):
        C.check_head_entropy(bad, "move")


def test_forced_shoot_and_truth_mask_constants_are_false():
    assert C.forced_shoot() is False
    assert C.truth_action_mask() is False
    assert C.FORCED_SHOOT is False
    assert C.TRUTH_ACTION_MASK is False


def test_check_no_forced_shoot_rule_catches_an_or_ed_shoot_bit():
    # TH08 bit 0 is SHOOT; a shoot=0 request whose mask carries bit 0 is the
    # v4.2.0 defect, so the checker must refuse it in both directions.
    with pytest.raises(C.ContractError):
        C.check_no_forced_shoot_rule([0x0001], shoot_requests=[0])
    with pytest.raises(C.ContractError):
        C.check_no_forced_shoot_rule([0x0000], shoot_requests=[1])
    assert C.check_no_forced_shoot_rule([0x0000, 0x0001], shoot_requests=[0, 1]) == (
        0x0000, 0x0001)


# ---------------------------------------------------------------------------
# 3. ActorObservation
# ---------------------------------------------------------------------------
def test_actor_observation_accepts_the_contract_shapes():
    specs = C.check_actor_observation(valid_observation())
    assert set(specs) == set(C.ACTOR_OBSERVATION_KEYS)
    assert specs["global_rgb"] == ((3, 240, 320), "uint8")
    assert specs["play_gray"] == ((4, 224, 192), "uint8")
    assert specs["hud_gray"] == ((1, 448, 160), "uint8")
    assert specs["executed_key_history"] == ((8, 7), "uint8")
    assert specs["history_valid"] == ((8,), "bool")
    assert specs["elapsed_capture_s"] == ((), "float32")
    assert specs["memory"] == ((256,), "float32")
    # The dataclass surface is equivalent to the mapping surface.
    record = C.ActorObservation(**valid_observation())
    assert set(C.check_actor_observation(record)) == set(C.ACTOR_OBSERVATION_KEYS)


def test_actor_observation_rejects_a_missing_field():
    obs = valid_observation()
    del obs["hud_gray"]
    with pytest.raises(C.ContractError, match="missing fields"):
        C.check_actor_observation(obs)


def test_actor_observation_rejects_an_extra_field():
    obs = valid_observation()
    obs["stage_number"] = 1
    with pytest.raises(C.ContractError, match="extra fields"):
        C.check_actor_observation(obs)


@pytest.mark.parametrize(
    "field, shape, dtype",
    [
        ("global_rgb", (3, 240, 319), "uint8"),   # one pixel column short
        ("global_rgb", (3, 320, 240), "uint8"),   # transposed
        ("play_gray", (4, 224, 192), "float32"),  # wrong dtype
        ("play_gray", (3, 224, 192), "uint8"),    # old three-frame stack
        ("hud_gray", (1, 224, 160), "uint8"),     # half the HUD height
        ("executed_key_history", (8, 8), "uint8"),
        ("memory", (128,), "float32"),
    ],
)
def test_actor_observation_rejects_wrong_shape_or_dtype(field, shape, dtype):
    obs = valid_observation()
    obs[field] = _Array(shape, dtype)
    with pytest.raises(C.ContractError):
        C.check_actor_observation(obs)


def test_actor_observation_rejects_key_history_values_outside_zero_one():
    obs = valid_observation()
    history = [[0] * 7 for _ in range(8)]
    history[3][5] = 2  # the shoot column is a flag, not a counter
    obs["executed_key_history"] = history
    with pytest.raises(C.ContractError):
        C.check_actor_observation(obs)


def test_actor_observation_rejects_non_bool_history_valid():
    obs = valid_observation()
    obs["history_valid"] = [1] * 8  # 0/1 ints are not flags
    with pytest.raises(C.ContractError):
        C.check_actor_observation(obs)


def test_actor_observation_rejects_non_finite_memory_or_negative_interval():
    obs = valid_observation()
    obs["memory"] = [0.0] * 255 + [float("nan")]
    with pytest.raises(C.ContractError):
        C.check_actor_observation(obs)
    obs = valid_observation()
    obs["elapsed_capture_s"] = -0.5
    with pytest.raises(C.ContractError):
        C.check_actor_observation(obs)
    obs = valid_observation()
    obs["elapsed_capture_s"] = [0.03]
    with pytest.raises(C.ContractError):
        C.check_actor_observation(obs)


def test_history_valid_must_be_a_leading_false_run():
    assert C.check_history_valid_prefix([False] * 8) == (False,) * 8
    assert C.check_history_valid_prefix([False, False] + [True] * 6) == (
        (False, False) + (True,) * 6)
    # A hole in the middle cannot be represented: a zero row would read as idle.
    with pytest.raises(C.ContractError):
        C.check_history_valid_prefix([True, False, True, True, True, True, True, True])


@pytest.mark.parametrize("name", list(C.FORBIDDEN_ACTOR_FIELDS))
def test_forbidden_actor_fields_are_refused(name):
    obs = valid_observation()
    obs[name] = 123
    with pytest.raises(C.ContractError, match="privileged"):
        C.check_actor_observation(obs)


def test_forbidden_actor_fields_are_found_when_nested():
    obs = valid_observation()
    obs["audit"] = {"meta": {"resource_truth": {"bombs": 3}}}
    found = C.find_privileged_fields(obs)
    assert "audit.meta.resource_truth" in found
    with pytest.raises(C.ContractError):
        C.check_actor_observation(obs)


def test_forbidden_actor_fields_are_found_on_an_object():
    @dataclasses.dataclass
    class Leaky:
        engine_frame: int
        pixels: int

    assert C.find_privileged_fields(Leaky(engine_frame=12, pixels=0)) == ("engine_frame",)
    with pytest.raises(C.ContractError):
        C.assert_not_privileged(Leaky(engine_frame=12, pixels=0))
    clean = {"pixels": 1, "keys": [0] * 7}
    assert C.assert_not_privileged(clean) is clean


def test_privileged_field_list_is_exactly_the_plan_list():
    assert C.FORBIDDEN_ACTOR_FIELDS == (
        "engine_frame", "seed", "script_id", "stage_truth", "enemy_table",
        "resource_truth", "teacher_action", "future_tape")


# ---------------------------------------------------------------------------
# 4. O2 parameters and digest
# ---------------------------------------------------------------------------
def test_o2_parameters_match_the_plan_spec_numbers():
    params = C.o2_parameters()
    assert params["source_rgb_wh"] == [640, 480]
    assert params["play_crop_xywh"] == [32, 16, 384, 448]
    assert params["hud_crop_xywh"] == [480, 16, 160, 448]
    assert params["gray_rule"] == "(77*R+151*G+28*B)>>8"
    assert params["gray_weights_rgb"] == [77, 151, 28]
    assert params["gray_shift"] == 8
    assert params["downsample_factor"] == 2
    assert params["downsample_accumulator"] == "uint32"
    assert params["normalization"] == "divide by 255 in minibatch"
    assert params["storage_dtype"] == "uint8"
    assert params["image_bytes_per_decision"] == 474112
    assert params["forbidden_actor_fields"] == list(C.FORBIDDEN_ACTOR_FIELDS)
    assert params["key_order"] == ["left", "right", "up", "down", "focus",
                                   "shoot", "bomb"]
    assert params["global_rgb_chw"] == [3, 240, 320]
    assert params["play_gray_chw"] == [4, 224, 192]
    assert params["hud_gray_chw"] == [1, 448, 160]
    # No real pixel encoding is implemented in this slice.
    assert not hasattr(C, "encode_o2")
    assert not hasattr(C, "downsample_2x2")


def test_o2_digest_is_stable_and_sensitive_to_one_parameter():
    digest = C.o2_digest()
    assert len(digest) == 64 and digest == digest.lower()
    assert all(char in "0123456789abcdef" for char in digest)
    assert C.o2_digest() == digest
    assert C.o2_digest(C.o2_parameters()) == digest

    changed = C.o2_parameters()
    changed["play_crop_xywh"] = [33, 16, 384, 448]
    assert C.o2_digest(changed) != digest

    changed_gray = C.o2_parameters()
    changed_gray["gray_weights_rgb"] = [76, 150, 29]
    assert C.o2_digest(changed_gray) != digest

    with pytest.raises(C.ContractError):
        C.o2_digest([("not", "a mapping")])


# ---------------------------------------------------------------------------
# 5. time / step contract
# ---------------------------------------------------------------------------
def test_hold_game_frames_is_two():
    assert C.HOLD_GAME_FRAMES == 2
    assert C.time_parameters()["hold_game_frames"] == 2


def test_engine_and_capture_durations_have_independent_sources():
    engine = C.check_duration(duration_source="engine_elapsed_frames",
                              elapsed_game_frames=2)
    assert engine.valid and engine.effective_frames == 2.0
    assert engine.elapsed_capture_s is None

    external = C.check_duration(duration_source="external_controller_clock",
                                elapsed_capture_s=2.0 / 60.0)
    assert external.valid
    assert external.effective_frames == pytest.approx(2.0)
    assert external.elapsed_game_frames is None

    # Neither source may borrow the other field.
    with pytest.raises(C.ContractError):
        C.check_duration(duration_source="engine_elapsed_frames")
    with pytest.raises(C.ContractError):
        C.check_duration(duration_source="external_controller_clock")
    with pytest.raises(C.ContractError):
        C.check_duration(duration_source="external_controller_clock",
                         elapsed_capture_s=2.0 / 60.0, elapsed_game_frames=2)
    with pytest.raises(C.ContractError):
        C.check_duration(duration_source="invalid", elapsed_game_frames=2)


@pytest.mark.parametrize("source", ["invalid", "engine", "", None, "External"])
def test_unknown_or_invalid_duration_sources_are_refused(source):
    with pytest.raises(C.ContractError):
        C.check_duration(duration_source=source, elapsed_game_frames=2)


def test_effective_frames_conversion_is_external_only():
    assert C.effective_frames_from_capture(2.0 / 60.0) == pytest.approx(2.0)
    assert C.effective_frames_from_capture(1.0 / 60.0) == pytest.approx(1.0)
    assert C.effective_frames_from_capture(1.0 / 30.0) == pytest.approx(2.0)
    with pytest.raises(C.ContractError):
        C.effective_frames_from_capture(2.0 / 60.0, duration_source="engine_elapsed_frames")
    with pytest.raises(C.ContractError):
        C.effective_frames_from_capture(2.0 / 60.0, duration_source="invalid")
    with pytest.raises(C.ContractError):
        C.effective_frames_from_capture(0.0)
    with pytest.raises(C.ContractError):
        C.effective_frames_from_capture(-1.0)
    with pytest.raises(C.ContractError):
        C.effective_frames_from_capture(1.0 / 60.0, hz=0)


def test_faults_cannot_masquerade_as_a_normal_two_frame_step():
    invalid = C.check_duration(duration_source="invalid")
    assert invalid.valid is False
    assert invalid.effective_frames is None
    with pytest.raises(C.ContractError):
        C.assert_normal_two_frame(invalid)

    # A dropped frame is 3 frames of wall time for a 2-frame request.
    dropped = C.check_duration(duration_source="external_controller_clock",
                               elapsed_capture_s=3.0 / 60.0)
    assert dropped.effective_frames == pytest.approx(3.0)
    with pytest.raises(C.ContractError):
        C.assert_normal_two_frame(dropped)

    normal = C.check_duration(duration_source="external_controller_clock",
                              elapsed_capture_s=2.0 / 60.0)
    assert C.assert_normal_two_frame(normal) is normal


# ---------------------------------------------------------------------------
# 6. transition fields and bootstrap
# ---------------------------------------------------------------------------
def test_valid_transition_is_accepted():
    checked = C.check_transition(valid_transition())
    assert checked["controller_source"] == "actor"


@pytest.mark.parametrize("alias", ["done", "is_done", "episode_done", "end",
                                   "terminal", "is_terminal"])
def test_a_single_done_field_is_refused(alias):
    record = valid_transition()
    record[alias] = True
    with pytest.raises(C.ContractError, match="three different facts"):
        C.check_transition(record)


@pytest.mark.parametrize("flag", ["terminated", "truncated", "segment_end"])
def test_each_end_flag_is_required_and_may_not_be_omitted(flag):
    record = valid_transition()
    del record[flag]
    with pytest.raises(C.ContractError):
        C.check_transition(record)


@pytest.mark.parametrize("flag", ["terminated", "truncated", "segment_end"])
def test_end_flags_must_be_bools_not_ints(flag):
    record = valid_transition(**{flag: 1})
    with pytest.raises(C.ContractError):
        C.check_transition(record)


def test_terminated_and_truncated_are_mutually_exclusive():
    record = valid_transition(terminated=True, truncated=True,
                              truncation_reason="frame_budget")
    with pytest.raises(C.ContractError, match="mutually exclusive"):
        C.check_transition(record)


def test_truncation_requires_a_registered_reason():
    with pytest.raises(C.ContractError):
        C.check_transition(valid_transition(truncated=True))
    with pytest.raises(C.ContractError):
        C.check_transition(valid_transition(truncated=True, truncation_reason="because"))
    assert C.check_transition(valid_transition(truncated=True,
                                               truncation_reason="frame_budget"))


def test_missing_terminal_observation_forbids_bootstrap():
    # Truncated with a bootstrap but no terminal observation: refused.
    with pytest.raises(C.ContractError, match="terminal_observation_ref"):
        C.check_bootstrap(terminated=False, truncated=True, segment_end=False,
                          bootstrap_mask=1, terminal_observation_ref=None,
                          next_obs_ref="shard/next")
    # With the reference the same record bootstraps.
    assert C.check_bootstrap(terminated=False, truncated=True, segment_end=False,
                             bootstrap_mask=1,
                             terminal_observation_ref="shard/terminal",
                             next_obs_ref="shard/next") == 1
    # Terminated never bootstraps, even with a reference available.
    with pytest.raises(C.ContractError):
        C.check_bootstrap(terminated=True, truncated=False, segment_end=False,
                          bootstrap_mask=1,
                          terminal_observation_ref="shard/terminal")
    assert C.check_bootstrap(terminated=True, truncated=False, segment_end=False,
                             bootstrap_mask=0,
                             terminal_observation_ref="shard/terminal") == 0
    # A continuing step needs its real next observation, not a reset image.
    with pytest.raises(C.ContractError):
        C.check_bootstrap(terminated=False, truncated=False, segment_end=False,
                          bootstrap_mask=1)
    with pytest.raises(C.ContractError, match="post-reset"):
        C.check_bootstrap(terminated=False, truncated=False, segment_end=True,
                          bootstrap_mask=1, next_obs_ref="shard/reset",
                          reset_observation_ref="shard/reset")
    assert C.check_bootstrap(terminated=False, truncated=False, segment_end=False,
                             bootstrap_mask=1, next_obs_ref="shard/next") == 1


def test_bootstrap_mask_cannot_be_two():
    with pytest.raises(C.ContractError):
        C.check_bootstrap(terminated=False, truncated=False, segment_end=False,
                          bootstrap_mask=2, next_obs_ref="shard/next")


def test_session_transitions_carry_no_actor_loss():
    assert C.SESSION_ACTOR_LOSS_MASK == 0
    with pytest.raises(C.ContractError, match="actor_loss_mask"):
        C.check_transition(valid_transition(controller_source="session",
                                            actor_loss_mask=1))
    ok = C.check_transition(valid_transition(controller_source="session",
                                             actor_loss_mask=0))
    assert ok["controller_source"] == "session"
    with pytest.raises(C.ContractError):
        C.check_transition(valid_transition(controller_source="judge"))


def test_transition_rejects_unknown_fields_and_bad_masks():
    record = valid_transition()
    record["engine_frame"] = 42
    with pytest.raises(C.ContractError, match="unregistered fields"):
        C.check_transition(record)
    with pytest.raises(C.ContractError):
        C.check_transition(valid_transition(executed_masks=(0x10000, 0x10000)))
    with pytest.raises(C.ContractError):
        C.check_transition(valid_transition(executed_masks=(1, 1, 1)))
    with pytest.raises(C.ContractError):
        C.check_transition(valid_transition(requested_action=72))


def test_actor_visible_transition_may_not_carry_privileged_ref():
    record = valid_transition(privileged_ref="shard/truth/0")
    assert C.check_transition(record)  # the training side may read it
    with pytest.raises(C.ContractError):
        C.check_transition(record, for_actor=True)
    assert C.check_transition(valid_transition(), for_actor=True)


def test_transition_bootstrap_is_checked_when_supplied():
    with pytest.raises(C.ContractError):
        C.check_transition(valid_transition(truncated=True,
                                            truncation_reason="rollout_limit"),
                           bootstrap_mask=1)
    ok = C.check_transition(
        valid_transition(truncated=True, truncation_reason="rollout_limit",
                         terminal_observation_ref="shard/terminal"),
        bootstrap_mask=1)
    assert ok["truncated"] is True


# ---------------------------------------------------------------------------
# 7. result JSON and exit codes
# ---------------------------------------------------------------------------
def test_exit_codes_match_the_plan_spec():
    assert (C.EXIT_OK, C.EXIT_INVALID, C.EXIT_GATE_FAIL, C.EXIT_INCOMPLETE) == (0, 2, 3, 4)
    assert set(C.EXIT_CODES) == {0, 2, 3, 4}
    assert C.RESULT_FIELDS == ("execution_status", "protocol_validity",
                              "research_result", "capability_gate")


def test_new_result_defaults_capability_gate_to_not_run():
    result = C.new_result()
    assert result["capability_gate"] == "not_run"
    assert result["execution_status"] == "ok"
    assert result["protocol_validity"] == "valid"
    assert result["research_result"] == {"status": "not_run"}
    assert set(C.RESULT_FIELDS) <= set(result)
    assert C.assert_result_schema(result) == result
    assert C.exit_code_for(result) == 0
    # An explicit non-capability sentinel is allowed, a claim is not.
    assert C.new_result(capability_gate="not_applicable")["capability_gate"] == (
        "not_applicable")


def test_exit_zero_cannot_promote_a_capability_gate():
    with pytest.raises(C.ContractError, match="constructor"):
        C.new_result(capability_gate="pass")
    with pytest.raises(C.ContractError, match="constructor"):
        C.new_result(capability_gate="fail")
    # A hand-built training result with exit 0 and gate=pass is refused too.
    forged = {
        "command_class": "training",
        "execution_status": "ok",
        "protocol_validity": "valid",
        "research_result": {"status": "pass"},
        "capability_gate": "pass",
        "capability_judgement": {"judge": "someone", "sha256": "0" * 64},
    }
    with pytest.raises(C.ContractError, match="training"):
        C.assert_result_schema(forged)
    # Even a non-training command cannot self-promote without judge evidence.
    no_evidence = C.new_result()
    no_evidence["capability_gate"] = "pass"
    with pytest.raises(C.ContractError, match="capability_judgement"):
        C.assert_result_schema(no_evidence)


def test_set_capability_gate_requires_independent_evidence():
    result = C.new_result(command_class="settlement")
    with pytest.raises(C.ContractError):
        C.set_capability_gate(result, "pass", judged_from={})
    with pytest.raises(C.ContractError):
        C.set_capability_gate(result, "pass", judged_from={"judge": "j"})
    with pytest.raises(C.ContractError):
        C.set_capability_gate(result, "pass",
                              judged_from={"judge": "", "sha256": "a" * 64})
    promoted = C.set_capability_gate(
        result, "pass", judged_from={"judge": "W12-settle", "sha256": "a" * 64})
    assert promoted["capability_gate"] == "pass"
    assert promoted["execution_status"] == "ok"  # exit 0 keeps saying "completed"
    assert C.assert_result_schema(promoted)["capability_gate"] == "pass"


@pytest.mark.parametrize("missing", list(C.RESULT_FIELDS))
def test_result_schema_requires_all_four_fields(missing):
    result = C.new_result()
    del result[missing]
    with pytest.raises(C.ContractError, match="missing fields"):
        C.assert_result_schema(result)


def test_result_schema_checks_status_consistency():
    with pytest.raises(C.ContractError):
        C.new_result(execution_status="done")
    with pytest.raises(C.ContractError):
        C.new_result(protocol_validity="maybe")
    with pytest.raises(C.ContractError):
        C.new_result(research_result={"status": "great"})
    invalid_ok = C.new_result()
    invalid_ok["protocol_validity"] = "invalid"
    with pytest.raises(C.ContractError):
        C.assert_result_schema(invalid_ok)
    bad_exit = C.new_result()
    bad_exit["exit_code"] = 0
    bad_exit["execution_status"] = "gate_fail"
    with pytest.raises(C.ContractError, match="exit_code"):
        C.assert_result_schema(bad_exit)
    with pytest.raises(C.ContractError):
        C.new_result(execution_status="ok", protocol_validity="valid",
                     capability_gate="victory")


# ---------------------------------------------------------------------------
# 8. the config mirrors the constants in both directions
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_config_exists_and_is_frozen_contract_metadata():
    data = _load_config()
    assert CONFIG_PATH.is_file()
    assert data["schema"] == C.CONFIG_SCHEMA == "v430-contracts/1"
    assert data["spec_revision"] == C.SPEC_REVISION == "v430-plan-r2"
    assert data["source_module"] == "touhou_rl/v430/contracts.py"
    assert data["timezone"] == "Asia/Shanghai"
    frozen_at = data["frozen_at"]
    assert isinstance(frozen_at, str) and "T" in frozen_at
    assert frozen_at.endswith("+08:00")
    assert data["plan_spec"]["revision_expected"] == C.SPEC_REVISION


def test_config_constants_equal_the_module_constants():
    data = _load_config()
    assert data["constants"] == C.contract_constants()
    # Both directions: no key may be added to one side only.
    assert set(data["constants"]) == set(C.contract_constants())
    assert set(data["constants"]["action"]) == set(C.action_parameters())
    assert set(data["constants"]["observation"]) == set(C.observation_parameters())


def test_config_pins_the_mandated_constants():
    data = _load_config()
    action = data["constants"]["action"]
    assert action["joint_count"] == 72
    assert action["hold_game_frames"] == 2
    assert action["forced_shoot"] is False
    assert action["truth_action_mask"] is False
    assert action["owner_exclusive"] is True
    assert action["session_actor_loss_mask"] == 0
    assert action["heads"] == {"move": 9, "focus": 2, "shoot": 2, "bomb": 2}
    observation = data["constants"]["observation"]
    assert observation["source_rgb_wh"] == [640, 480]
    assert observation["play_crop_xywh"] == [32, 16, 384, 448]
    assert observation["hud_crop_xywh"] == [480, 16, 160, 448]
    assert observation["digest"] == C.o2_digest()
    assert data["constants"]["results"]["capability_gate_default"] == "not_run"
    assert data["constants"]["results"]["exit_codes"] == {
        "0": C.EXIT_CODES[0], "2": C.EXIT_CODES[2],
        "3": C.EXIT_CODES[3], "4": C.EXIT_CODES[4],
    }
