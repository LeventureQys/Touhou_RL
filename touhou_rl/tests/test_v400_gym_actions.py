"""Unit tests for the v4.0.0 action encoding (03 §3.2). Pure CPU, no engine."""

from __future__ import annotations

import pytest

from touhou_rl.spec import BTN_BOMB, BTN_FOCUS, BTN_SHOOT, MOVES
from touhou_rl.v400.gym import actions as A


def test_agent_mask_matches_the_documented_whitelist():
    """10-P0 §4.2 pins the agent's bits at 0x00F7."""
    assert A.AGENT_MASK == 0x00F7


def test_direction_table_has_nine_entries_and_distinct_bits():
    assert A.MOVES_PER_CAND == 9
    assert len(A._MOVE_BITS) == 9
    assert A._MOVE_BITS[0] == 0
    assert len(set(A._MOVE_BITS)) == 9


def test_every_candidate_round_trips():
    for cand in range(A.N_CAND):
        for bomb in (0, 1):
            mask = A.cand_to_mask(cand, bomb)
            assert A.mask_to_cand(mask) == (cand, bomb)


def test_candidates_produce_eighteen_distinct_masks():
    """A divmod slip once collapsed these onto four; guard it explicitly."""
    assert len(set(A.CAND_MASKS)) == A.N_CAND


def test_candidate_separates_direction_from_focus():
    # cand = focus * 9 + move, so 0..8 are unfocused and 9..17 are focused.
    for move in range(9):
        plain = A.cand_to_mask(move)
        focused = A.cand_to_mask(9 + move)
        assert focused == (plain | BTN_FOCUS)
        assert not plain & BTN_FOCUS


def test_shooting_is_always_on():
    for cand in range(A.N_CAND):
        assert A.cand_to_mask(cand) & BTN_SHOOT


def test_bomb_bit_only_appears_when_requested():
    for cand in range(A.N_CAND):
        assert not A.cand_to_mask(cand, 0) & BTN_BOMB
        assert A.cand_to_mask(cand, 1) & BTN_BOMB


def test_diagonal_is_not_decoded_as_a_subset_direction():
    """NE is UP|RIGHT; a subset match would wrongly return N."""
    ne = 2  # MOVES order: 0 idle, 1 N, 2 NE, ...
    mask = A.cand_to_mask(ne)
    assert A.mask_to_cand(mask)[0] == ne
    north = A.cand_to_mask(1)
    assert north != mask


def test_non_agent_bits_are_rejected():
    with pytest.raises(ValueError):
        A.mask_to_cand(BTN_SHOOT | 0x100)  # SKIP is system-only


def test_missing_shoot_is_rejected():
    with pytest.raises(ValueError):
        A.mask_to_cand(0)


@pytest.mark.parametrize("cand", [-1, A.N_CAND, 100])
def test_out_of_range_candidate_rejected(cand):
    with pytest.raises(ValueError):
        A.cand_to_mask(cand)


def test_out_of_range_bomb_rejected():
    with pytest.raises(ValueError):
        A.cand_to_mask(0, 2)


def test_student_action_space_is_thirty_six():
    assert len(A.STUDENT_ACTIONS) == 36
    assert len(set(A.STUDENT_ACTIONS)) == 36


def test_moves_table_is_shared_with_spec():
    """The encoder must not re-declare directions; it must reuse spec.MOVES."""
    for index, moves in enumerate(MOVES):
        expected = 0
        for bit in moves:
            expected |= bit
        assert A._MOVE_BITS[index] == expected
