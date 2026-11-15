"""Behavioural tests for the v4.3.0 action compiler (W01 slice C).

Each assertion pins a property the work package promises:

* ``shoot=0`` really produces masks **without** the SHOOT bit and ``shoot=1``
  produces masks with it (the v4.2.0 forced-SHOOT defect, now impossible);
* the direction and focus bits match the reused v4.0.0 tables exactly;
* a bomb triggers once on the press edge and a held request does not re-trigger,
  with ``requested``/``emitted``/``bomb_edge_consumed`` all recorded;
* only one owner may hold the keyboard and a switch must be explicit;
* a session decision carries ``actor_loss_mask = 0``;
* masks outside ``0..0xFFFF`` are refused;
* replaying the same requests from the same snapshot yields bitwise-identical
  masks (zero-update determinism), and no hidden hit state influences a compile.

Synthetic: this file proves the state machine.  It launches no engine, so it is
**not** evidence that the real game reacts to these masks the way we assume — that
is the engine-backed part of W01 that follows.
"""

from __future__ import annotations

import dataclasses

import pytest

from touhou_rl.spec import BTN_BOMB, BTN_FOCUS, BTN_SHOOT, MOVES
from touhou_rl.v400.gym.actions import DIRECTION_MASK, cand_to_mask
from touhou_rl.v430 import actions as A
from touhou_rl.v430 import contracts as C

SHOOT = BTN_SHOOT
FOCUS = BTN_FOCUS
BOMB = BTN_BOMB


def direction_bits(move: int) -> int:
    bits = 0
    for bit in MOVES[move]:
        bits |= bit
    return bits


def compile_one(request, *, source="actor"):
    compiler = A.ActionCompiler(source)
    try:
        return compiler.compile_request(*request)
    finally:
        compiler.release()


# ---------------------------------------------------------------------------
# shoot: released really means released
# ---------------------------------------------------------------------------
def test_shoot_zero_masks_never_contain_the_shoot_bit():
    for move in range(9):
        for focus in (0, 1):
            compiled = compile_one((move, focus, 0, 0))
            assert len(compiled.executed_masks) == C.HOLD_GAME_FRAMES
            for mask in compiled.executed_masks:
                assert mask & SHOOT == 0, (
                    f"move={move} focus={focus} shoot=0 emitted 0x{mask:04x} "
                    f"with the SHOOT bit set")


def test_shoot_one_masks_always_contain_the_shoot_bit():
    for move in range(9):
        for focus in (0, 1):
            compiled = compile_one((move, focus, 1, 0))
            for mask in compiled.executed_masks:
                assert mask & SHOOT == SHOOT


def test_shoot_released_and_pressed_are_different_masks():
    for move in range(9):
        for focus in (0, 1):
            for bomb in (0, 1):
                released = A.mask_for(move, focus, 0, bomb)
                pressed = A.mask_for(move, focus, 1, bomb)
                assert released != pressed
                assert pressed == released | SHOOT


def test_a_mask_with_an_or_ed_shoot_bit_is_refused_by_the_state_machine():
    # The compiler re-checks the emitted masks against the request, so even a
    # hypothetical forced-SHOOT builder cannot pass.
    with pytest.raises(C.ContractError):
        C.check_no_forced_shoot_rule([0x00B1, 0x00B1], shoot_requests=[0, 0])
    with pytest.raises(C.ContractError):
        A.compile_masks((0, 0, 0, 0), frames=4)


# ---------------------------------------------------------------------------
# direction and focus bits
# ---------------------------------------------------------------------------
def test_direction_bits_match_the_reused_v400_table():
    for move in range(9):
        compiled = compile_one((move, 0, 0, 0))
        assert compiled.executed_masks[0] & DIRECTION_MASK == direction_bits(move)
        # And the v4.0.0 candidate table (which forces SHOOT) minus its shoot bit.
        assert direction_bits(move) == cand_to_mask(move, 0) & ~SHOOT


def test_focus_bit_follows_the_request():
    for move in range(9):
        unfocused = compile_one((move, 0, 1, 0))
        focused = compile_one((move, 1, 1, 0))
        assert unfocused.executed_masks[0] & FOCUS == 0
        assert focused.executed_masks[0] & FOCUS == FOCUS
        assert focused.executed_masks[0] == unfocused.executed_masks[0] | FOCUS


def test_all_72_combinations_have_distinct_masks():
    seen = {}
    for mask, action in A.all_joint_masks():
        assert mask not in seen, f"{action} collides with {seen.get(mask)}"
        seen[mask] = action
    assert len(seen) == 72
    assert len(set(seen.values())) == 72


def test_hold_window_is_two_frames_and_stable():
    compiled = compile_one((3, 1, 0, 0))
    assert compiled.hold_game_frames == C.HOLD_GAME_FRAMES == 2
    assert len(compiled.executed_masks) == 2
    assert len(set(compiled.executed_masks)) == 1
    assert [frame.index for frame in compiled.frames] == [0, 1]
    assert all(frame.bomb_edge is False for frame in compiled.frames)


# ---------------------------------------------------------------------------
# bomb press edge
# ---------------------------------------------------------------------------
def test_bomb_press_edge_triggers_exactly_once():
    compiler = A.ActionCompiler("actor")
    try:
        first = compiler.compile_request(0, 0, 1, 1)
        assert first.bomb_requested == 1
        assert first.bomb_edge_consumed is True
        assert first.bomb_edge_frames == (0,)
        assert first.bomb_held is False
        assert first.bomb_edges_total == 1
        assert first.executed_masks[0] & BOMB == BOMB
        assert first.bomb_emitted == (1, 1)

        # Still pressed: the level stays asserted but no second trigger fires.
        second = compiler.compile_request(0, 0, 1, 1)
        assert second.bomb_requested == 1
        assert second.bomb_emitted == (1, 1)
        assert second.bomb_edge_frames == ()
        assert second.bomb_edge_consumed is False
        assert second.bomb_held is True
        assert second.bomb_edges_total == 1  # one press, one trigger

        third = compiler.compile_request(0, 0, 1, 1)
        assert third.bomb_edges_total == 1
        assert third.bomb_edge_frames == ()
        # A logged decision must show the whole story, not just the level bit.
        logged = third.as_dict()
        assert logged["bomb_requested"] == 1
        assert logged["bomb_emitted"] == [1, 1]
        assert logged["bomb_edge_consumed"] is False
        assert logged["bomb_edges_total"] == 1
    finally:
        compiler.release()


def test_bomb_triggers_again_only_after_a_release():
    compiler = A.ActionCompiler("actor")
    try:
        assert compiler.compile_request(0, 0, 1, 1).bomb_edge_consumed is True
        assert compiler.compile_request(0, 0, 1, 1).bomb_edge_consumed is False
        released = compiler.compile_request(0, 0, 1, 0)
        assert released.executed_masks[0] & BOMB == 0
        assert released.bomb_edge_consumed is False
        assert released.bomb_held is False
        again = compiler.compile_request(0, 0, 1, 1)
        assert again.bomb_edge_consumed is True
        assert again.bomb_edge_frames == (0,)
        assert again.bomb_edges_total == 2
    finally:
        compiler.release()


def test_a_bomb_never_leaks_into_the_next_decision_without_a_request():
    compiler = A.ActionCompiler("actor")
    try:
        compiler.compile_request(0, 0, 0, 1)
        after = compiler.compile_request(0, 0, 0, 0)
        for mask in after.executed_masks:
            assert mask & BOMB == 0
        assert after.bomb_emitted == (0, 0)
    finally:
        compiler.release()


# ---------------------------------------------------------------------------
# no truth: the compiler cannot filter by resources
# ---------------------------------------------------------------------------
def test_a_resource_free_bomb_request_is_still_emitted():
    # truth_action_mask=false: the compiler has no resource input, so a request
    # that would spend a bomb the character may not own is sent and recorded.
    compiled = compile_one((0, 0, 1, 1))
    assert compiled.executed_masks[0] & BOMB == BOMB
    assert compiled.bomb_requested == 1


def test_a_context_cannot_change_the_compiled_action():
    plain = A.ActionCompiler("actor")
    with_context = A.ActionCompiler("actor")
    try:
        request = (2, 1, 0, 1)
        baseline = plain.compile_request(*request)
        contextual = with_context.compile_request(
            *request, context={"note": "hud says 1 bomb left", "frame": 12})
        assert contextual.executed_masks == baseline.executed_masks
        assert contextual.bomb_edge_frames == baseline.bomb_edge_frames
    finally:
        plain.release()
        with_context.release()


@pytest.mark.parametrize(
    "privileged",
    [
        {"resource_truth": {"bombs": 0}},
        {"engine_frame": 400},
        {"stage_truth": {"stage": 2}},
        {"teacher_action": [0, 0, 1, 0]},
        {"future_tape": [0x0001]},
        {"audit": {"meta": {"seed": 1234}}},
    ],
)
def test_a_privileged_context_is_refused(privileged):
    with pytest.raises(A.PrivilegedInputError):
        compiler = A.ActionCompiler("actor")
        try:
            compiler.compile_request(0, 0, 1, 0, context=privileged)
        finally:
            compiler.release()


def test_assert_not_privileged_is_re_exported_by_the_action_layer():
    with pytest.raises(A.PrivilegedInputError):
        A.ActionCompiler("actor", context={"enemy_table": []})
    assert A.assert_not_privileged is C.assert_not_privileged
    clean = {"inputs": {"pixels": 1}}
    assert A.assert_not_privileged(clean) is clean


# ---------------------------------------------------------------------------
# owner exclusivity
# ---------------------------------------------------------------------------
def test_two_owners_cannot_hold_the_keyboard_at_once():
    actor = A.ActionCompiler("actor")
    try:
        with pytest.raises(A.OwnerConflictError):
            A.ActionCompiler("session", ownership=actor.ownership)
        # The actor still owns it and can compile.
        assert actor.compile_request(0, 0, 1, 0).controller_source == "actor"
    finally:
        actor.release()


def test_owner_switch_requires_an_explicit_release():
    bus = A.KeyOwnership()
    actor = A.ActionCompiler("actor", ownership=bus)
    assert bus.holder == "actor"
    with pytest.raises(A.OwnerConflictError):
        A.ActionCompiler("session", ownership=bus)

    actor.release()
    assert bus.holder is None
    session = A.ActionCompiler("session", ownership=bus)
    assert bus.holder == "session"

    # The released actor cannot send keys while the session owns them.
    with pytest.raises(A.OwnerConflictError):
        actor.compile_request(0, 0, 1, 0)
    assert session.compile_request(0, 0, 1, 0).controller_source == "session"
    session.release()


def test_releasing_twice_or_with_a_foreign_token_is_refused():
    bus = A.KeyOwnership()
    actor = A.ActionCompiler("actor", ownership=bus)
    token = actor.token
    actor.release()
    with pytest.raises(A.OwnershipError):
        bus.release(token)  # already released
    other = A.KeyOwnership()
    session = A.ActionCompiler("session", ownership=other)
    with pytest.raises(A.OwnershipError):
        bus.release(session.token)  # a token from another bus
    session.release()


def test_a_stale_compiler_cannot_compile_after_another_owner_took_over():
    bus = A.KeyOwnership()
    actor = A.ActionCompiler("actor", ownership=bus)
    actor.release()
    with pytest.raises(A.OwnerConflictError):
        actor.compile_request(0, 0, 1, 0)


def test_unknown_controller_source_is_refused():
    with pytest.raises(A.OwnershipError):
        A.ActionCompiler("judge")
    with pytest.raises(A.OwnershipError):
        A.ActionCompiler(None)


# ---------------------------------------------------------------------------
# session vs actor policy loss
# ---------------------------------------------------------------------------
def test_session_decisions_carry_no_actor_loss():
    session = A.ActionCompiler("session")
    try:
        compiled = session.compile_request(2, 1, 1, 0)
        assert compiled.controller_source == "session"
        assert compiled.actor_loss_mask == C.SESSION_ACTOR_LOSS_MASK == 0
        assert session.actor_loss_mask == 0
    finally:
        session.release()


def test_actor_decisions_carry_actor_loss():
    compiled = compile_one((2, 1, 1, 0))
    assert compiled.controller_source == "actor"
    assert compiled.actor_loss_mask == 1


# ---------------------------------------------------------------------------
# mask range validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [-1, 0x10000, 0x1FFFF, True, 1.5, "0x10", None])
def test_masks_outside_sixteen_bits_are_refused(bad):
    with pytest.raises(A.MaskRangeError):
        A.check_mask(bad)


def test_zero_and_the_all_ones_word_are_in_range():
    assert A.check_mask(0) == 0
    assert A.check_mask(0xFFFF) == 0xFFFF
    # ...but 0xFFFF carries menu/skip/enter bits, which the policy may not press.
    with pytest.raises(A.MaskRangeError):
        A.check_agent_mask(0xFFFF)
    assert A.check_agent_mask(A.mask_for(8, 1, 1, 1)) == A.mask_for(8, 1, 1, 1)


def test_a_compiled_action_with_an_out_of_range_mask_is_refused():
    compiled = compile_one((0, 0, 1, 0))
    with pytest.raises(A.MaskRangeError):
        dataclasses.replace(compiled, executed_masks=(0x10000, 0x10000))
    with pytest.raises(A.MaskRangeError):
        dataclasses.replace(compiled, executed_masks=(0x8000 | 0x0400 | 0x0200,
                                                      0x8000 | 0x0400 | 0x0200))


def test_the_compiler_rejects_impossible_requests():
    compiler = A.ActionCompiler("actor")
    try:
        with pytest.raises(C.ContractError):
            compiler.compile_request(9, 0, 0, 0)
        with pytest.raises(C.ContractError):
            compiler.compile_request(0, 2, 0, 0)
        with pytest.raises(C.ContractError):
            compiler.compile_request(0, 0, True, 0)
    finally:
        compiler.release()


def test_a_hold_other_than_the_contract_windows_is_refused():
    with pytest.raises(A.ActionError):
        A.ActionCompiler("actor", hold_frames=4)
    with pytest.raises(A.ActionError):
        A.compile_masks((0, 0, 1, 0), frames=1)


# ---------------------------------------------------------------------------
# determinism: same state + same request => bit-identical masks
# ---------------------------------------------------------------------------
REQUESTS = [
    (0, 0, 1, 0),
    (2, 1, 0, 1),
    (2, 1, 0, 1),
    (2, 0, 0, 0),
    (3, 1, 1, 0),
    (3, 1, 1, 1),
    (8, 0, 0, 0),
    (5, 1, 1, 1),
]


def test_zero_update_replay_is_bitwise_identical():
    first = A.ActionCompiler("actor")
    base = first.snapshot()
    first_run = [(c.executed_masks, c.bomb_edge_frames, c.bomb_edges_total)
                 for c in first.compile_sequence(REQUESTS)]

    second = A.ActionCompiler("actor")
    second.restore(base)
    second_run = [(c.executed_masks, c.bomb_edge_frames, c.bomb_edges_total)
                  for c in second.compile_sequence(REQUESTS)]

    # Masks, bomb edges and the running edge count are all reproduced.  Each
    # 0 -> 1 bomb transition in REQUESTS produces exactly one edge frame.
    assert first_run == second_run
    assert [edges for _masks, edges, _total in first_run] == [
        (), (0,), (), (), (), (0,), (), (0,)]
    assert [total for _masks, _edges, total in first_run] == [0, 1, 1, 1, 1, 2, 2, 3]
    first.release()
    second.release()


def test_the_same_request_from_the_same_snapshot_yields_the_same_masks():
    compiler = A.ActionCompiler("actor")
    try:
        compiler.compile_request(0, 0, 1, 0)  # move away from the cold start
        snapshot = compiler.snapshot()
        compiler.restore(snapshot)
        once = compiler.compile_request(4, 1, 1, 1)
        compiler.restore(snapshot)
        twice = compiler.compile_request(4, 1, 1, 1)
        assert once.executed_masks == twice.executed_masks
        assert once.bomb_edge_frames == twice.bomb_edge_frames == (0,)
        assert once.bomb_edges_total == twice.bomb_edges_total
        assert once.frames == twice.frames
    finally:
        compiler.release()


def test_restoring_a_snapshot_reproduces_the_edge_bookkeeping():
    compiler = A.ActionCompiler("actor")
    try:
        compiler.compile_request(0, 0, 1, 1)  # one edge consumed
        base = compiler.snapshot()
        assert base.bomb_edges_total == 1
        held = compiler.compile_request(0, 0, 1, 1)
        assert held.bomb_edge_consumed is False
        compiler.restore(base)
        again = compiler.compile_request(0, 0, 1, 1)
        assert again.bomb_edge_consumed is False
        assert again.bomb_edges_total == 1
    finally:
        compiler.release()


def test_the_compiler_has_no_hidden_hit_state():
    # 100 decisions without a bomb must not change how a bomb request compiles:
    # the only history is the previously emitted key level.
    fresh = A.ActionCompiler("actor")
    cold = fresh.compile_request(0, 0, 1, 1)
    warm = A.ActionCompiler("actor")
    for _ in range(100):
        warm.compile_request(0, 0, 1, 0)
    warm_bomb = warm.compile_request(0, 0, 1, 1)
    assert warm_bomb.executed_masks == cold.executed_masks
    assert warm_bomb.bomb_edge_frames == cold.bomb_edge_frames == (0,)
    assert warm_bomb.bomb_edge_consumed is True
    fresh.release()
    warm.release()


def test_sampling_and_evaluation_share_one_entry_point():
    # compile_action (sampling) and compile_request (evaluation) must land on the
    # same state machine and produce the same masks for the same request.
    sampler = A.ActionCompiler("actor")
    evaluator = A.ActionCompiler("actor")
    try:
        for request in REQUESTS:
            action = C.JointAction(*request)
            assert sampler.compile_action(action).executed_masks == (
                evaluator.compile_request(*request).executed_masks)
    finally:
        sampler.release()
        evaluator.release()


def test_state_snapshot_is_validated():
    with pytest.raises(A.MaskRangeError):
        A.CompilerState(last_mask=0x10000)
    with pytest.raises(A.ActionError):
        A.CompilerState(last_bomb_level=2)
    with pytest.raises(A.ActionError):
        A.CompilerState(decision_index=1, bomb_edges_total=2)
    with pytest.raises(A.ActionError):
        A.ActionCompiler("actor").restore({"last_mask": 0})
