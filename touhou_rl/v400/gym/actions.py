"""Action encoding for the v4.0.0 training ground (03 §3.2).

The engine sees a 16-bit button mask. Two encodings sit on top of it:

* the **candidate** action the teacher searches over: ``cand = focus * 9 + move``
  with shooting always on and bombing off (18 candidates);
* the **student** action: ``(cand, bomb)``, i.e. 36 combinations.

The mask is built from :data:`touhou_rl.spec.MOVES`, so there is exactly one table
of directions in the repository.
"""

from __future__ import annotations

from itertools import product

from touhou_rl.spec import BTN_BOMB, BTN_FOCUS, BTN_SHOOT, MOVES

#: Direction index: 0 idle, 1 N, 2 NE, 3 E, 4 SE, 5 S, 6 SW, 7 W, 8 NW (+y down).
MOVES_PER_CAND = len(MOVES)
#: 9 directions x 2 focus states.
N_CAND = MOVES_PER_CAND * 2


def _fold(moves: tuple[int, ...]) -> int:
    """OR a tuple of button bits into one mask."""
    bits = 0
    for bit in moves:
        bits |= bit
    return bits


#: Each direction's bits as a single OR, so exact set comparison is possible.
_MOVE_BITS: tuple[int, ...] = tuple(_fold(moves) for moves in MOVES)

#: The agent may only ever emit these bits; everything else is system input
#: (10-P0 §4.2) and must never come from the policy.
DIRECTION_MASK = _fold(tuple(bit for moves in MOVES for bit in moves))
AGENT_MASK = DIRECTION_MASK | BTN_SHOOT | BTN_BOMB | BTN_FOCUS

#: The bit layout of 03 §3.1, spelled out for the tests and for the teacher.
BUTTON_BITS = {"shoot": BTN_SHOOT, "bomb": BTN_BOMB, "focus": BTN_FOCUS}


def _split(cand: int) -> tuple[int, int]:
    """``cand = focus * 9 + move`` -> ``(move, focus)``.

    Order matters: divmod's remainder is the move, not the focus.  Getting this
    backwards makes every diagonal decode as N and collapses 18 candidates onto 4
    distinct masks, which is exactly what happened the first time.
    """
    focus, move = divmod(cand, MOVES_PER_CAND)
    if focus > 1:
        raise ValueError(f"candidate out of range: {cand}")
    return move, focus


def cand_to_mask(cand: int, bomb: int = 0) -> int:
    """Translate ``(cand, bomb)`` into the engine's button mask.

    Shooting is always on: the candidate space deliberately excludes it, because
    "not shooting" is never useful here and it would double the teacher's search
    space for nothing.
    """
    if not 0 <= cand < N_CAND:
        raise ValueError(f"candidate out of range: {cand}")
    if bomb not in (0, 1):
        raise ValueError(f"bomb must be 0 or 1, got {bomb}")
    move, focus = _split(cand)
    mask = _MOVE_BITS[move] | BTN_SHOOT
    if focus:
        mask |= BTN_FOCUS
    if bomb:
        mask |= BTN_BOMB
    return mask


def mask_to_cand(mask: int) -> tuple[int, int]:
    """Inverse of :func:`cand_to_mask`.

    Raises ``ValueError`` for masks the encoder cannot produce, so a client bug
    cannot silently inject system input or an ambiguous direction.
    """
    if mask & ~AGENT_MASK:
        raise ValueError(f"mask has non-agent bits: 0x{mask & ~AGENT_MASK:04x}")
    if not mask & BTN_SHOOT:
        raise ValueError("mask does not shoot; not representable as a candidate")
    directions = mask & DIRECTION_MASK
    move = None
    for index, bits in enumerate(_MOVE_BITS):
        if bits == directions:
            move = index
            break
    if move is None:
        raise ValueError(f"direction bits 0x{directions:04x} match no entry in MOVES")
    focus = 1 if mask & BTN_FOCUS else 0
    bomb = 1 if mask & BTN_BOMB else 0
    return focus * MOVES_PER_CAND + move, bomb


#: All 18 candidate masks, index == candidate id. What the teacher rolls out.
CAND_MASKS: tuple[int, ...] = tuple(cand_to_mask(cand) for cand in range(N_CAND))

#: All 36 student actions as ``(cand, bomb)``.
STUDENT_ACTIONS: tuple[tuple[int, int], ...] = tuple(product(range(N_CAND), (0, 1)))
