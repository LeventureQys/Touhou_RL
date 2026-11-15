"""Behavioural tests for the v4.3.0 pixel session layer and the isolated judge (W01).

``tests/test_v430_session.py`` is the only new W01 test file in the whitelist, so
the ``evaluate.py`` rejection paths live here too (as a separate section) instead
of in a ``test_v430_evaluate.py`` that the work package does not authorise.

Every assertion pins a property the work package promises, and every one of them
is a *behaviour* assertion -- a state transition, a compiled mask, a raised
exception, an exit code, a file that does or does not exist.  Nothing here claims
the real game behaves this way: this file launches no engine, so it is
implementation evidence, not game evidence.

Session section
---------------
* pixel scenes drive the documented state transitions, including
  ``playing -> hit_pending -> respawn -> playing``;
* ``unknown`` is a real state: a scene the pixels do not identify is never
  upgraded to ``playing`` and never reinterpreted as ``game_over``, and it
  presses no key;
* a session-driven decision carries ``controller_source='session'`` and
  ``actor_loss_mask=0`` while the actor's compiler carries ``1``;
* two owners can never hold the keyboard, and acquiring it implicitly is refused;
* engine truth (``stage_id`` / ``game_over`` / ``stage_clear`` / lives / ...) in
  the evidence or the machine context is refused *before* parsing (fail-closed);
* restarts are counted with an auditable reason and source;
* the capture interval, the events and any reward component stamped on a
  ``transition`` span survive into the ledger, and the ledger can *prove* that no
  sample, no frame index and no clock second was dropped.

Judge section
-------------
* an empty result set, a missing field, a zero denominator, an invalid record, a
  hand-written ``mean``/``clears`` and a digest mismatch all produce exit 2 and
  never a passed gate;
* the gate itself is a research result: exit 0/3 with ``capability_gate`` still
  ``not_run``;
* the write path refuses a fabricated capability **before** creating any file,
  both in ``evaluate.write_result`` and in the CLI's ``write_checked_result``;
* text digests are ``CRLF -> LF`` normalised, binary digests are raw, an identity
  without a ``digest_scope`` is refused, and a scope mismatch is reported
  differently from a content mismatch;
* the judge's import graph is asserted to contain no training/teacher/engine
  module.

Regression note: ``session_action_for_state`` used to be written as
``SessionAction(state=state, *request)``, which raised ``TypeError: got multiple
values for argument 'state'`` for every session-driven state -- i.e. the session
could never press a key at all.  ``test_every_state_maps_to_a_session_action`` and
``test_session_driven_states_compile_with_actor_loss_mask_zero`` are the
regression tests for that bug.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from touhou_rl.v430 import contracts as C
from touhou_rl.v430 import session as S
from touhou_rl.v430.actions import ActionCompiler, KeyOwnership, OwnerConflictError

#: Modules that must not appear on the judge's import graph (observed, not just
#: read from a constant, so a runtime side effect would be caught too).
_FORBIDDEN_RUNTIME_MODULES = (
    "touhou_rl.v430.rl", "touhou_rl.v430.teacher", "touhou_rl.v430.curriculum",
    "touhou_rl.v430.learn", "touhou_rl.v430.models", "touhou_rl.v430.data",
    "touhou_rl.v430.observation", "touhou_rl.v430.tactics",
    "touhou_rl.v400.teacher", "torch", "numpy",
)

_BEFORE_EVALUATE = frozenset(sys.modules)
from touhou_rl.v430 import evaluate as E  # noqa: E402  (deliberately after the snapshot)

_NEW_MODULES = frozenset(sys.modules) - _BEFORE_EVALUATE

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERVAL = 1.0 / C.REFERENCE_HZ


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def evidence(index: int, *, scene: str = S.SCENE_PLAY, dark: float = 0.1,
             player: bool = True, hit: bool = False,
             pips: int | None = None, digest: str | None = None,
             captured_at: float | None = None) -> dict:
    """One pixel-evidence mapping; ``elapsed`` is consistent with the clock."""

    return {
        "frame_index": index,
        "frame_digest": digest if digest is not None else f"{index + 1:064x}",
        "captured_at_s": (captured_at if captured_at is not None
                          else (index + 1) * INTERVAL),
        "elapsed_capture_s": INTERVAL,
        "scene": scene,
        "play_field_dark_ratio": dark,
        "player_sprite_visible": player,
        "hit_flash_visible": hit,
        "visible_life_pips": pips,
    }


def live(index: int, **kwargs) -> dict:
    return evidence(index, scene=S.SCENE_PLAY, player=True, **kwargs)


def drive(machine: S.SessionMachine, frames) -> list[S.SessionStep]:
    return [machine.advance(frame) for frame in frames]


def playing_then_banner() -> S.SessionMachine:
    machine = S.SessionMachine()
    drive(machine, [live(0), live(1),
                    evidence(2, scene=S.SCENE_BANNER, player=False),
                    evidence(3, scene=S.SCENE_BANNER, player=False)])
    return machine


# ===========================================================================
# session: states and states -> actions
# ===========================================================================
def test_the_eight_required_states_exist():
    required = {"playing", "hit_pending", "respawn", "transition", "game_over",
                "menu", "ending", "unknown"}
    assert required <= set(S.SESSION_STATES)
    assert len(S.SESSION_STATES) == len(set(S.SESSION_STATES))


def test_every_state_maps_to_a_session_action_or_none():
    """Regression: this raised ``TypeError`` for all four session-driven states."""

    driven = []
    actor_driven = []
    for state in S.SESSION_STATES:
        action = S.session_action_for_state(state)
        if action is None:
            actor_driven.append(state)
            continue
        assert isinstance(action, S.SessionAction), state
        payload = action.as_dict()
        assert payload["state"] == state
        assert payload["controller_source"] == "session"
        assert payload["actor_loss_mask"] == 0
        assert payload["requested"] == [action.move, action.focus, action.shoot,
                                        action.bomb]
        driven.append(state)
    assert set(driven) == set(S.SESSION_ACTION_STATES)
    assert set(actor_driven) == {"playing", "hit_pending", "respawn", "unknown"}


def test_session_action_fields_keep_their_documented_order():
    assert [field.name for field in dataclasses.fields(S.SessionAction)] == [
        "state", "move", "focus", "shoot", "bomb"]
    action = S.session_action_for_state(S.MENU)
    assert action.as_dict()["requested"] == [0, 0, 1, 0]


def test_an_unregistered_state_is_refused():
    with pytest.raises(S.SessionStateError):
        S.session_action_for_state("paused")


@pytest.mark.parametrize("scene,expected", [
    (S.SCENE_BANNER, S.TRANSITION),
    (S.SCENE_MENU, S.MENU),
    (S.SCENE_OVER, S.GAME_OVER),
    (S.SCENE_ENDING, S.ENDING),
])
def test_pixel_scene_drives_the_expected_state(scene, expected):
    machine = S.SessionMachine()
    step = machine.advance(evidence(0, scene=scene, player=False))
    assert step.state == expected
    assert machine.state == expected
    assert step.session_owns_keys is True


def test_playing_hit_respawn_playing():
    machine = S.SessionMachine()
    steps = drive(machine, [
        live(0, pips=3),
        live(1, pips=3, hit=True),
        evidence(2, scene=S.SCENE_PLAY, player=False, pips=3),
        live(3, pips=2),
    ])
    assert [step.state for step in steps] == [
        S.PLAYING, S.HIT_PENDING, S.RESPAWN, S.PLAYING]
    assert steps[1].events == ("hit_flash_seen",)
    assert steps[2].events == ("respawn_wait",)
    assert steps[3].events == ("player_visible_again",)


def test_a_life_pip_drop_is_a_hit_only_when_both_readings_are_known():
    machine = S.SessionMachine()
    drive(machine, [live(0, pips=3)])
    dropped = machine.advance(live(1, pips=2))
    assert dropped.state == S.HIT_PENDING
    assert dropped.events == ("life_pips_dropped",)


def test_none_life_pips_is_never_read_as_zero():
    machine = S.SessionMachine()
    drive(machine, [live(0, pips=3)])
    unknown_reading = machine.advance(live(1, pips=None))
    assert unknown_reading.state == S.PLAYING
    assert unknown_reading.events == ()
    still_unknown = machine.advance(live(2, pips=None))
    assert still_unknown.state == S.PLAYING
    recovery = machine.advance(live(3, pips=3))
    assert recovery.state == S.PLAYING


# ===========================================================================
# session: session-driven states really compile keys
# ===========================================================================
@pytest.mark.parametrize("scene,expected", [
    (S.SCENE_MENU, S.MENU),
    (S.SCENE_BANNER, S.TRANSITION),
    (S.SCENE_OVER, S.GAME_OVER),
    (S.SCENE_ENDING, S.ENDING),
])
def test_session_driven_states_compile_with_actor_loss_mask_zero(scene, expected):
    """End to end: no exception, right state, keys held, no policy gradient."""

    machine = S.SessionMachine()
    step = machine.advance(evidence(0, scene=scene, player=False))
    assert step.state == expected
    assert step.session_owns_keys is True
    assert machine.session_holds_keys is True
    assert step.action is not None
    assert step.action.controller_source == "session"
    assert step.action.actor_loss_mask == 0
    assert step.action.actor_loss_mask == C.SESSION_ACTOR_LOSS_MASK
    assert len(step.action.executed_masks) == C.HOLD_GAME_FRAMES
    for mask in step.action.executed_masks:
        assert mask in {entry[0] for entry in _all_joint_masks()}
    # The ledger attributes the span to the session, with the mask recorded.
    span = machine.finish().spans[0]
    assert span.state == expected
    assert span.key_owner == "session"
    assert span.actor_loss_mask == 0
    assert span.session_decisions == 1


def _all_joint_masks():
    from touhou_rl.v430 import actions as A
    return A.all_joint_masks()


def test_playing_does_not_hold_the_keyboard_and_cannot_compile():
    machine = S.SessionMachine()
    machine.advance(live(0))
    assert machine.state == S.PLAYING
    assert machine.session_holds_keys is False
    with pytest.raises(S.SessionStateError) as excinfo:
        machine.compile_session_action()
    assert "does not hold the keyboard" in str(excinfo.value)


def test_session_compiler_reports_session_and_actor_compiler_reports_actor():
    bus = KeyOwnership()
    machine = S.SessionMachine(ownership=bus)
    step = machine.advance(evidence(0, scene=S.SCENE_MENU, player=False))
    assert step.action.controller_source == "session"
    assert step.action.actor_loss_mask == S.SESSION_ACTOR_LOSS_MASK == 0
    machine.release_control()
    assert bus.holder is None
    actor = ActionCompiler("actor", ownership=bus)
    try:
        compiled = actor.compile_request(0, 0, 1, 0)
        assert compiled.controller_source == "actor"
        assert compiled.actor_loss_mask == C.ACTOR_LOSS_MASK_ON == 1
    finally:
        actor.release()


def test_two_owners_cannot_hold_the_keyboard_at_once():
    bus = KeyOwnership()
    actor = ActionCompiler("actor", ownership=bus)
    try:
        machine = S.SessionMachine(ownership=bus)
        with pytest.raises(OwnerConflictError) as excinfo:
            machine.advance(evidence(0, scene=S.SCENE_MENU, player=False))
        assert "owns the keyboard" in str(excinfo.value) or \
               "cannot take the keyboard" in str(excinfo.value)
        assert bus.holder == "actor"
    finally:
        actor.release()
    # Once the actor releases, the session may take over.
    machine = S.SessionMachine(ownership=bus)
    step = machine.advance(evidence(0, scene=S.SCENE_MENU, player=False))
    assert step.state == S.MENU
    assert bus.holder == "session"


def test_switching_owner_must_be_explicit():
    bus = KeyOwnership()
    machine = S.SessionMachine(ownership=bus)
    machine.advance(evidence(0, scene=S.SCENE_MENU, player=False))
    assert bus.holder == "session"
    with pytest.raises(OwnerConflictError):
        ActionCompiler("actor", ownership=bus)
    machine.advance(live(1))
    assert bus.holder is None, "a live play field must hand the keys back"


# ===========================================================================
# session: unknown is a real state
# ===========================================================================
def test_unknown_is_not_filled_with_playing_or_game_over():
    machine = S.SessionMachine()
    assert machine.state == S.UNKNOWN
    steps = drive(machine, [
        live(0),
        evidence(1, scene=S.SCENE_UNRECOGNISED, player=False),
        # Two pixel features disagree: a "live" label on a >90% black play field.
        evidence(2, scene=S.SCENE_PLAY, dark=0.95, player=False),
        evidence(3, scene=S.SCENE_PLAY, dark=0.95, player=True),
        evidence(4, scene=S.SCENE_UNRECOGNISED, player=False),
    ])
    assert [step.state for step in steps] == [
        S.PLAYING, S.UNKNOWN, S.UNKNOWN, S.UNKNOWN, S.UNKNOWN]
    for step in steps[1:]:
        assert step.state not in (S.PLAYING, S.GAME_OVER)
        assert step.action is None, "unknown must press no key"
        assert step.session_owns_keys is False
    assert machine.state == S.UNKNOWN
    ledger = machine.finish()
    unknown_span = [span for span in ledger.spans if span.state == S.UNKNOWN]
    assert unknown_span
    assert all(span.key_owner is None for span in unknown_span)
    assert all(span.session_decisions == 0 for span in unknown_span)


def test_unknown_is_sticky_until_a_positive_pixel_signal():
    machine = S.SessionMachine()
    drive(machine, [evidence(0, scene=S.SCENE_UNRECOGNISED, player=False)])
    assert machine.state == S.UNKNOWN
    # Still unknown: a live label with no player sprite.
    machine.advance(evidence(1, scene=S.SCENE_PLAY, player=False))
    assert machine.state == S.UNKNOWN
    # Only a positive signal (live field + located player) recovers.
    machine.advance(live(2))
    assert machine.state == S.PLAYING


def test_unrecognised_keeps_a_pending_death_beat():
    machine = S.SessionMachine()
    drive(machine, [live(0), live(1, hit=True)])
    assert machine.state == S.HIT_PENDING
    step = machine.advance(evidence(2, scene=S.SCENE_UNRECOGNISED, player=False))
    assert step.state == S.HIT_PENDING


# ===========================================================================
# session: engine truth is refused (fail-closed sentinels)
# ===========================================================================
@pytest.mark.parametrize("field", [
    "stage_id", "stage", "stage_clear", "stage_truth", "game_over",
    "game_over_flag", "lives", "lives_truth", "bomb_count", "resource_truth",
    "engine_frame", "seed", "teacher_action",
])
def test_truth_fields_are_refused_in_evidence(field):
    machine = S.SessionMachine()
    machine.advance(live(0))
    poisoned = dict(evidence(1))
    poisoned[field] = 3
    with pytest.raises(S.SessionTruthError) as excinfo:
        machine.advance(poisoned)
    message = str(excinfo.value)
    assert "engine truth" in message
    assert field in message
    # Fail closed: the machine did not move and the sample was not recorded.
    assert machine.state == S.PLAYING
    assert len(machine.ledger.samples) == 1


@pytest.mark.parametrize("payload", [
    {"stage_truth": 3},
    {"game_over": True},
    {"stage_clear": True},
    {"lives": 2},
    {"nested": {"engine_frame": 12345}},
    {"list": [{"seed": 7}]},
])
def test_truth_fields_are_refused_in_the_machine_context(payload):
    with pytest.raises(S.SessionTruthError):
        S.SessionMachine(context=payload)


def test_contract_privileged_names_are_also_refused():
    for name in C.FORBIDDEN_ACTOR_FIELDS:
        with pytest.raises(S.SessionTruthError):
            S.SessionMachine(context={name: 1})


def test_unregistered_evidence_fields_are_refused():
    machine = S.SessionMachine()
    with pytest.raises(S.SessionEvidenceError) as excinfo:
        machine.advance({**evidence(0), "weather": "sunny"})
    assert "unregistered fields" in str(excinfo.value)


def test_missing_evidence_fields_are_refused():
    frame = evidence(0)
    frame.pop("scene")
    with pytest.raises(S.SessionEvidenceError) as excinfo:
        S.SessionMachine().advance(frame)
    assert "missing fields" in str(excinfo.value)


def test_a_pixel_digest_must_be_hashed_as_binary():
    frame = evidence(0)
    frame["frame_digest_scope"] = "text/lf-normalized"
    with pytest.raises(S.SessionEvidenceError) as excinfo:
        S.SessionMachine().advance(frame)
    assert "binary" in str(excinfo.value)


def test_a_zero_capture_interval_is_refused():
    frame = evidence(0)
    frame["elapsed_capture_s"] = 0.0
    with pytest.raises(S.SessionEvidenceError) as excinfo:
        S.SessionMachine().advance(frame)
    assert "drop time from the reward chain" in str(excinfo.value)


def test_session_source_imports_no_engine_module():
    audit = S.session_source_audit()
    assert audit["forbidden_present"] == []
    assert "touhou_rl.v400.gym.env" not in audit["imports"]
    assert "touhou_rl.v430.contracts" in audit["imports"]
    assert audit["digest_scope"] == E.DIGEST_SCOPE_TEXT


# ===========================================================================
# session: clock discipline
# ===========================================================================
def test_a_backwards_capture_clock_is_refused():
    machine = S.SessionMachine()
    machine.advance(evidence(0, captured_at=10 * INTERVAL))
    with pytest.raises(S.SessionClockError):
        machine.advance(evidence(1, captured_at=5 * INTERVAL))


def test_a_repeated_frame_index_is_refused():
    machine = S.SessionMachine()
    machine.advance(live(0))
    with pytest.raises(S.SessionClockError):
        machine.advance(live(0))


def test_an_interval_inconsistent_with_the_clock_is_refused():
    machine = S.SessionMachine()
    machine.advance(live(0))
    frame = evidence(1, captured_at=10 * INTERVAL)
    frame["elapsed_capture_s"] = 5 * INTERVAL
    with pytest.raises(S.SessionClockError):
        machine.advance(frame)


def test_advancing_after_finish_is_refused():
    machine = S.SessionMachine()
    machine.advance(live(0))
    machine.finish()
    with pytest.raises(S.SessionStateError):
        machine.advance(live(1))


# ===========================================================================
# session: restarts are counted and auditable
# ===========================================================================
def test_restart_count_and_reasons_are_auditable():
    machine = S.SessionMachine()
    drive(machine, [
        live(0),                                        # playing
        evidence(1, scene=S.SCENE_OVER, player=False),   # game_over
        evidence(2, scene=S.SCENE_MENU, player=False),   # -> restart #1
        live(3),                                        # playing again
        evidence(4, scene=S.SCENE_ENDING, player=False),  # ending
        evidence(5, scene=S.SCENE_MENU, player=False),   # -> restart #2
    ])
    assert machine.restart_count == 2
    first, second = machine.restarts
    assert first.restart_index == 1
    assert first.reason == "game_over_restart"
    assert first.source == "pixels"
    assert (first.from_state, first.to_state) == (S.GAME_OVER, S.MENU)
    assert first.frame_index == 2
    assert first.frame_digest_scope == "binary/raw"
    assert len(first.frame_digest) == 64
    assert first.session_elapsed_s == pytest.approx(2 * INTERVAL)
    assert second.restart_index == 2
    assert second.reason == "ending_restart"
    assert (second.from_state, second.to_state) == (S.ENDING, S.MENU)
    ledger = machine.finish()
    report = ledger.assert_complete()
    assert report["restart_count"] == 2
    assert report["restart_reasons"] == ["game_over_restart", "ending_restart"]
    assert report["restart_sources"] == ["pixels", "pixels"]
    assert [record.reason for record in ledger.restarts] == [
        "game_over_restart", "ending_restart"]


def test_a_manual_restart_requires_an_operator_reason():
    machine = S.SessionMachine()
    drive(machine, [live(0), live(1)])
    with pytest.raises(S.SessionStateError):
        machine.record_manual_restart(live(2), reason="game_over_restart")
    record = machine.record_manual_restart(live(2), reason="manual_reset")
    assert record.source == "operator"
    assert record.reason == "manual_reset"
    assert machine.restart_count == 1


def test_a_plain_menu_visit_is_not_a_restart():
    machine = S.SessionMachine()
    drive(machine, [live(0), evidence(1, scene=S.SCENE_MENU, player=False),
                    live(2)])
    assert machine.restart_count == 0


# ===========================================================================
# session: the transition period keeps its time, events and reward
# ===========================================================================
def test_transition_span_keeps_duration_events_and_reward():
    machine = S.SessionMachine()
    drive(machine, [live(0), live(1)])
    banner = machine.advance(evidence(2, scene=S.SCENE_BANNER, player=False))
    assert banner.state == S.TRANSITION
    # The stage-clear reward is paid *while the banner is up* and the actor is not
    # driving; it must not vanish with the owner switch.
    machine.attach_reward("stage_clear", 1.0)
    silence = machine.advance(evidence(3, scene=S.SCENE_BANNER, player=False))
    assert silence.state == S.TRANSITION
    back = machine.advance(live(4))
    assert back.state == S.PLAYING
    ledger = machine.finish()
    report = ledger.assert_complete()
    assert report["complete"] is True

    transition_spans = [span for span in ledger.spans
                        if span.state == S.TRANSITION]
    assert len(transition_spans) == 1
    span = transition_spans[0]
    assert span.sample_count == 2, "both banner frames belong to the transition"
    assert span.elapsed_capture_s == pytest.approx(2 * INTERVAL)
    assert span.duration["duration_source"] == C.DURATION_SOURCE_EXTERNAL
    assert span.duration["effective_frames"] == pytest.approx(
        C.REFERENCE_HZ * 2 * INTERVAL)
    assert "stage_banner_seen" in span.events
    assert span.reward_components == {"stage_clear": 1.0}
    assert span.key_owner == "session"
    assert span.actor_loss_mask == 0
    assert span.session_decisions == 2

    rollup = ledger.reward_rollup()
    assert rollup["stage_clear"]["sum"] == 1.0
    assert rollup["stage_clear"]["states"] == [S.TRANSITION]
    assert rollup["stage_clear"]["spans"] == [span.span_index]
    assert rollup["stage_clear"]["elapsed_capture_s"] == pytest.approx(2 * INTERVAL)

    totals = ledger.totals()
    assert totals["total_elapsed_capture_s"] == pytest.approx(5 * INTERVAL)
    assert totals["transition_waits"] == 1
    assert totals["samples"] == 5
    assert report["duration_source"] == C.DURATION_SOURCE_EXTERNAL


def test_the_ledger_detects_a_dropped_span():
    machine = playing_then_banner()
    machine.finish()
    ledger = machine.ledger
    broken = S.SessionLedger(spans=ledger.spans[:-1], samples=ledger.samples,
                             restarts=ledger.restarts, hz=ledger.totals()["hz"],
                             finished=True)
    with pytest.raises(S.SessionLedgerError) as excinfo:
        broken.assert_complete()
    assert "samples" in str(excinfo.value)


def test_the_ledger_detects_a_zero_length_span_in_the_middle():
    machine = playing_then_banner()
    machine.finish()
    ledger = machine.ledger
    assert len(ledger.spans) >= 2
    tampered = dataclasses.replace(
        ledger.spans[0], elapsed_capture_s=0.0,
        duration=S._span_duration(0.0, C.REFERENCE_HZ))
    broken = S.SessionLedger(spans=(tampered,) + tuple(ledger.spans[1:]),
                             samples=ledger.samples, restarts=ledger.restarts,
                             hz=ledger.totals()["hz"], finished=True)
    with pytest.raises(S.SessionLedgerError) as excinfo:
        broken.assert_complete()
    assert "zero" in str(excinfo.value)


def test_reward_attached_before_any_evidence_is_refused():
    with pytest.raises(S.SessionStateError):
        S.SessionMachine().attach_reward("stage_clear", 1.0)


def test_a_duplicate_reward_component_on_one_span_is_refused():
    machine = S.SessionMachine()
    machine.advance(live(0))
    machine.attach_reward("clear", 1.0)
    with pytest.raises(S.SessionStateError):
        machine.attach_reward("clear", 1.0)


def test_a_fractional_transition_does_not_lose_the_zero_length_final_span():
    """A state change on the last captured frame is recorded, not rounded away."""

    machine = S.SessionMachine()
    drive(machine, [live(0), live(1)])
    machine.advance(evidence(2, scene=S.SCENE_MENU, player=False))
    ledger = machine.finish()
    report = ledger.assert_complete()
    assert report["complete"] is True
    totals = ledger.totals()
    assert totals["total_elapsed_capture_s"] == pytest.approx(3 * INTERVAL)
    assert totals["spans"] == 2


# ===========================================================================
# evaluate: Wilson interval
# ===========================================================================
def test_wilson_interval_matches_the_v420_formula():
    from touhou_rl.v420.eval import wilson_interval as v420_wilson

    for successes, total in ((0, 1), (3, 4), (50, 100), (100, 100), (1, 7)):
        assert E.wilson_interval(successes, total) == v420_wilson(successes, total)


def test_wilson_interval_has_sane_bounds():
    low, high = E.wilson_interval(0, 10)
    assert low == 0.0 and 0.0 < high < 1.0
    low, high = E.wilson_interval(10, 10)
    assert 0.0 < low < 1.0 and high <= 1.0
    previous = -1.0
    for successes in range(0, 21):
        low, _high = E.wilson_interval(successes, 20)
        assert low >= previous
        previous = low


def test_a_zero_denominator_is_refused():
    with pytest.raises(E.ZeroDenominatorError):
        E.wilson_interval(0, 0)


def test_the_gate_comes_from_the_plan_spec():
    spec = json.loads((REPO_ROOT / "Document/Update/Alpha/v4.3.0/PLAN_SPEC.json")
                      .read_text(encoding="utf-8-sig"))
    gate = E.GateRequest.from_plan_spec(spec)
    formal = spec["evaluation"]["formal"]
    assert gate.episodes == formal["episodes"]
    assert gate.min_successes == formal["min_successes"]
    assert gate.wilson_min_lower_bound == formal["wilson_min_lower_bound"]
    low, _high = E.wilson_interval(gate.min_successes, gate.episodes)
    assert low >= gate.wilson_min_lower_bound, (
        "the frozen plan must be satisfiable at its own minimum")


# ===========================================================================
# evaluate: the sealed input and its rejection paths
# ===========================================================================
def episode(index: int, *, outcome: str = "clear", ancestor: str | None = None,
            split: str = "dev") -> dict:
    return {
        "episode_id": f"ep-{index:04d}",
        "ancestor_id": ancestor or f"anc-{index:04d}",
        "seed": index,
        "split": split,
        "outcome": outcome,
        "protocol_validity": "valid",
        "policy_sha256": "a" * 64,
        "frames": 1000 + index,
    }


def write_batch(path: Path, episodes, *, split: str = "dev", **extra) -> Path:
    document = {
        "schema": E.BATCH_SCHEMA,
        "experiment_id": "v430-w01-test",
        "run_id": "unit-test",
        "policy_sha256": "b" * 64,
        "split": split,
        "episodes": episodes,
    }
    document.update(extra)
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


SMALL_GATE = E.GateRequest(episodes=4, min_successes=2,
                           wilson_min_lower_bound=0.10)


def assert_rejected(result, kind: str) -> None:
    assert result["execution_status"] == "invalid"
    assert result["protocol_validity"] == "invalid"
    assert result["exit_code"] == C.EXIT_INVALID == 2
    assert result["research_result"]["passed"] is False
    assert result["research_result"]["status"] == "invalid"
    assert result["capability_gate"] == "not_run"
    kinds = [row["kind"] for row in result["rejections"]]
    assert kind in kinds, kinds
    assert C.assert_result_schema(result)  # never a half-built document


def test_an_empty_result_set_is_refused(tmp_path):
    batch = write_batch(tmp_path / "empty.json", [])
    assert_rejected(E.evaluate(batch, gate=SMALL_GATE), "empty_result_set")


def test_a_missing_episode_field_is_refused(tmp_path):
    broken = episode(0)
    broken.pop("ancestor_id")
    batch = write_batch(tmp_path / "broken.json", [broken])
    assert_rejected(E.evaluate(batch, gate=SMALL_GATE), "missing_field")


def test_a_zero_denominator_is_refused(tmp_path):
    batch = write_batch(tmp_path / "denominator.json", [])
    result = E.evaluate(batch, gate=SMALL_GATE)
    assert result["statistics"]["denominator"] == 0
    assert result["statistics"]["point_estimate"] is None
    assert result["statistics"]["wilson"] is None
    assert_rejected(result, "empty_result_set")


def test_a_record_flagged_invalid_poisons_the_whole_batch(tmp_path):
    episodes = [episode(0), episode(1), episode(2)]
    episodes[1]["protocol_validity"] = "invalid"
    batch = write_batch(tmp_path / "poisoned.json", episodes)
    assert_rejected(E.evaluate(batch, gate=SMALL_GATE), "invalid_record")


def test_a_hand_written_clears_count_is_refused(tmp_path):
    batch = write_batch(tmp_path / "manual.json", [episode(0), episode(1)],
                        clears=2)
    assert_rejected(E.evaluate(batch, gate=SMALL_GATE), "manual_aggregate")


def test_a_hand_written_success_flag_that_disagrees_is_refused(tmp_path):
    broken = episode(0, outcome="game_over")
    broken["success"] = True
    batch = write_batch(tmp_path / "lied.json", [broken])
    assert_rejected(E.evaluate(batch, gate=SMALL_GATE), "invalid_record")


def test_a_duplicate_episode_id_is_refused(tmp_path):
    duplicate = [episode(0), episode(0)]
    batch = write_batch(tmp_path / "dupe.json", duplicate)
    assert_rejected(E.evaluate(batch, gate=SMALL_GATE), "invalid_record")


def test_a_digest_mismatch_is_refused(tmp_path):
    batch = write_batch(tmp_path / "digest.json",
                        [episode(0), episode(1), episode(2), episode(3)])
    result = E.evaluate(batch, gate=SMALL_GATE,
                        pins={str(batch): {"sha256": "0" * 64,
                                           "digest_scope": E.DIGEST_SCOPE_TEXT}})
    assert_rejected(result, "content_mismatch")


def test_a_matching_pin_is_accepted(tmp_path):
    batch = write_batch(tmp_path / "pinned.json",
                        [episode(0), episode(1), episode(2), episode(3)])
    identity = E.identify_file(batch)
    result = E.evaluate(batch, gate=SMALL_GATE, pins={str(batch): identity})
    assert result["execution_status"] == "ok"
    assert result["inputs"][0]["digest_scope"] == E.DIGEST_SCOPE_TEXT
    assert result["inputs"][0]["sha256"] == identity["sha256"]


def test_a_gate_failure_is_exit_three_not_invalid(tmp_path):
    batch = write_batch(tmp_path / "gate.json", [episode(0, outcome="game_over"),
                                                 episode(1, outcome="game_over"),
                                                 episode(2), episode(3)])
    result = E.evaluate(batch, gate=E.GateRequest(episodes=4, min_successes=4,
                                                 wilson_min_lower_bound=0.5))
    assert result["execution_status"] == "gate_fail"
    assert result["protocol_validity"] == "valid"
    assert result["exit_code"] == C.EXIT_GATE_FAIL == 3
    assert result["research_result"]["status"] == "fail"
    assert result["research_result"]["passed"] is False
    assert result["capability_gate"] == "not_run"


def test_a_passing_gate_is_a_research_result_not_a_capability(tmp_path):
    batch = write_batch(tmp_path / "pass.json",
                        [episode(0), episode(1), episode(2),
                         episode(3, outcome="game_over")])
    result = E.evaluate(batch, gate=SMALL_GATE)
    assert result["execution_status"] == "ok"
    assert result["exit_code"] == C.EXIT_OK == 0
    assert result["research_result"]["status"] == "pass"
    assert result["capability_gate"] == "not_run"
    assert result["capability_gate"] != "pass"
    assert any("capability_gate stays not_run" in gap
               for gap in result["remaining_gaps"])
    # The point estimate is always reported with its denominator and interval.
    statistics = result["statistics"]
    assert statistics["denominator"] == 4
    assert statistics["successes"] == 3
    assert statistics["point_estimate"] == pytest.approx(0.75)
    assert statistics["wilson"]["low"] < 0.75 < statistics["wilson"]["high"]
    assert statistics["wilson"]["method"] == "wilson-score-interval"


def test_every_result_carries_the_four_contract_fields(tmp_path):
    batch = write_batch(tmp_path / "fields.json", [episode(0), episode(1)])
    result = E.evaluate(batch, gate=SMALL_GATE)
    for field in C.RESULT_FIELDS:
        assert field in result
    assert result["exit_code"] == C.exit_code_for(result)
    assert C.assert_result_schema(result)
    assert result["judge_isolation"]["isolated"] is True


def test_no_gate_is_a_summary_not_a_pass(tmp_path):
    batch = write_batch(tmp_path / "summary.json", [episode(0), episode(1)])
    result = E.evaluate(batch)
    assert result["research_result"]["status"] == "inconclusive"
    assert result["research_result"]["passed"] is False
    assert result["capability_gate"] == "not_run"
    assert result["exit_code"] == C.EXIT_OK


# ===========================================================================
# evaluate: the write path validates before it writes
# ===========================================================================
def fabricated_capability() -> dict:
    """A result that claims a capability from a training command."""

    result = C.new_result(command_class="training")
    promoted = C.set_capability_gate(result, "pass",
                                     judged_from={"judge": "self",
                                                  "sha256": "0" * 64})
    promoted["exit_code"] = 0
    return promoted


def test_the_fabricated_result_really_does_claim_a_capability():
    """Control: without the schema check the claim would survive to disk."""

    fabricated = fabricated_capability()
    assert fabricated["capability_gate"] == "pass"
    with pytest.raises(C.ContractError):
        C.assert_result_schema(fabricated)


def test_write_result_blocks_a_fabricated_capability_and_creates_no_file(tmp_path):
    target = tmp_path / "nested" / "result.json"
    with pytest.raises(E.ResultSchemaError):
        E.write_result(fabricated_capability(), target)
    assert not target.exists()
    assert not target.parent.exists(), "the check runs before any directory is made"


def test_the_cli_write_path_blocks_a_fabricated_capability(tmp_path):
    """The CLI must not be a second, schema-free way to write a result."""

    module = load_cli_module()
    target = tmp_path / "cli-result.json"
    ok, message = module.write_checked_result(target, fabricated_capability())
    assert ok is False
    assert message and "assert_result_schema" in message
    assert not target.exists()


def test_write_result_accepts_a_legal_result(tmp_path):
    batch = write_batch(tmp_path / "legal.json", [episode(0), episode(1)])
    result = E.evaluate(batch, gate=SMALL_GATE)
    target = tmp_path / "out" / "legal-result.json"
    written = E.write_result(result, target)
    assert written == target
    assert target.is_file()
    with pytest.raises(E.OutputExistsError):
        E.write_result(result, target)


def load_cli_module():
    """Import ``tools/rl/v430.py`` as a module without touching ``sys.path``."""

    path = REPO_ROOT / "tools" / "rl" / "v430.py"
    spec = importlib.util.spec_from_file_location("v430_cli_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_module_keeps_audit_and_adds_validate_contract():
    module = load_cli_module()
    parser = module.build_parser()
    args = parser.parse_args(["validate-contract",
                              "--config", "configs/v430/contracts.json",
                              "--out", ".tmp/out.json"])
    assert args.command == "validate-contract"
    assert args.handler is module.cmd_validate_contract
    audit = parser.parse_args(["audit", "--config", "configs/v430/assets.json",
                               "--out", ".tmp/a.json"])
    assert audit.handler is module.cmd_audit
    assert audit.command == "audit"
    # validate-contract is no longer a reserved (fails-closed) subcommand.
    assert "validate-contract" not in module.RESERVED
    assert "audit" not in module.RESERVED


# ===========================================================================
# evaluate: digest scopes (CRLF -> LF for text, raw for binary)
# ===========================================================================
def test_text_digests_ignore_line_endings_and_binary_digests_do_not(tmp_path):
    crlf = tmp_path / "crlf.json"
    lf = tmp_path / "lf.json"
    crlf.write_bytes(b'{\r\n  "a": 1\r\n}\r\n')
    lf.write_bytes(b'{\n  "a": 1\n}\n')
    text_crlf = E.identify_file(crlf, scope=E.DIGEST_SCOPE_TEXT)
    text_lf = E.identify_file(lf, scope=E.DIGEST_SCOPE_TEXT)
    binary_crlf = E.identify_file(crlf, scope=E.DIGEST_SCOPE_BINARY)
    binary_lf = E.identify_file(lf, scope=E.DIGEST_SCOPE_BINARY)
    assert text_crlf["sha256"] == text_lf["sha256"]
    assert binary_crlf["sha256"] != binary_lf["sha256"]
    assert text_crlf["line_endings"] == {"crlf": 3, "lf_only": 0,
                                         "has_crlf": True, "mixed": False}
    assert text_crlf["digest_scope"] == E.DIGEST_SCOPE_TEXT


def test_every_identity_carries_a_digest_scope():
    identity = E.identify_bytes(b"hello\n", path="x.txt")
    assert identity["digest_scope"] == E.DIGEST_SCOPE_TEXT
    with pytest.raises(E.MissingDigestScopeError):
        E.check_identity({"sha256": identity["sha256"]})
    with pytest.raises(E.MissingDigestScopeError):
        E.check_identity({"sha256": identity["sha256"],
                          "digest_scope": "sha256-of-whatever"})
    E.check_identity(identity)


def test_scope_mismatch_and_content_mismatch_are_different_failures(tmp_path):
    crlf = tmp_path / "probe.json"
    crlf.write_bytes(b'{\r\n  "a": 1\r\n}\r\n')
    payload = crlf.read_bytes()
    binary_digest = E.digest_bytes(payload, scope=E.DIGEST_SCOPE_BINARY)
    with pytest.raises(E.DigestScopeMismatch) as scope_exc:
        E.verify_identity({"sha256": binary_digest,
                           "digest_scope": E.DIGEST_SCOPE_TEXT,
                           "path": "probe.json"}, payload)
    scope_message = str(scope_exc.value)
    assert "different scope" in scope_message
    assert "not a content change" in scope_message
    with pytest.raises(E.DigestMismatch) as content_exc:
        E.verify_identity({"sha256": "0" * 64,
                           "digest_scope": E.DIGEST_SCOPE_TEXT,
                           "path": "probe.json"}, payload)
    content_message = str(content_exc.value)
    assert "content differs" in content_message
    assert "neither scope reproduces the pin" in content_message
    assert type(scope_exc.value) is not type(content_exc.value)


def test_the_plan_spec_itself_is_a_crlf_file_so_the_scope_matters():
    """The finding is real in this checkout, not hypothetical."""

    plan = REPO_ROOT / "Document/Update/Alpha/v4.3.0/PLAN_SPEC.json"
    identity = E.identify_file(plan)
    if identity["line_endings"]["crlf"] == 0:
        pytest.skip("PLAN_SPEC.json is LF in this checkout")
    binary = E.identify_file(plan, scope=E.DIGEST_SCOPE_BINARY)
    assert identity["sha256"] != binary["sha256"]
    with pytest.raises(E.DigestScopeMismatch):
        E.verify_identity({"sha256": binary["sha256"],
                           "digest_scope": E.DIGEST_SCOPE_TEXT,
                           "path": "PLAN_SPEC.json"}, plan.read_bytes())


def test_the_pixel_digest_scope_matches_the_binary_scope():
    assert S.DIGEST_SCOPE_PIXELS == E.DIGEST_SCOPE_BINARY
    assert E.DIGEST_SCOPES == (E.DIGEST_SCOPE_TEXT, E.DIGEST_SCOPE_BINARY)


# ===========================================================================
# evaluate: judge isolation
# ===========================================================================
def test_the_judge_imports_nothing_from_training_or_the_engine():
    isolation = E.assert_judge_isolation()
    assert isolation["isolated"] is True
    for name in isolation["imports"]:
        assert name in E.JUDGE_IMPORTS_ALLOWED, name
        assert not any(name == bad or name.startswith(bad + ".")
                       for bad in E.JUDGE_IMPORTS_FORBIDDEN), name


def test_importing_the_judge_pulls_in_no_forbidden_module():
    for name in _NEW_MODULES:
        assert not any(name == bad or name.startswith(bad + ".")
                       for bad in _FORBIDDEN_RUNTIME_MODULES), name


def test_the_judge_refuses_a_dynamic_import(tmp_path):
    source = tmp_path / "dynamic.py"
    source.write_text(
        "import importlib\n"
        "importlib.import_module('touhou_rl.v430.rl')\n", encoding="utf-8")
    with pytest.raises(E.EvaluationError) as excinfo:
        E.module_imports(source)
    assert "dynamic import" in str(excinfo.value)


def test_the_judge_refuses_a_training_dependency(tmp_path):
    source = tmp_path / "coupled.py"
    source.write_text("import torch\n", encoding="utf-8")
    with pytest.raises(E.EvaluationError) as excinfo:
        E.assert_judge_isolation(source)
    assert "torch" in str(excinfo.value)
