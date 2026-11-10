"""How one decision becomes ``STEP`` commands: SHOOT always on, plus the pulse.

Two facts decide this module (EXP-P1-009):

1. **SHOOT is always on.**  ``cand = focus * 9 + move`` (03 §3.2) deliberately
   excludes the shoot bit, and :func:`touhou_rl.v400.gym.actions.cand_to_mask`
   ORs ``BTN_SHOOT`` into every mask it builds — so the teacher has always shot,
   and ``13-P3 §2.2`` ("``shoot``：固定开火") is satisfied by construction.  A
   "force the shoot bit" option is therefore a no-op, which is what the control
   group of EXP-P1-009 measured.

2. **The stage transition needs a *rising edge* of ``SHOOT|ENTER``.**  The game
   only leaves the end-of-stage state when
   ``(g_CurFrameInput & 0x1001)`` is non-zero **and differs from the previous
   frame's** (``GameManager.cpp:381-389``, cited in 10-P0 §4.2: "按住 SHOOT 不算
   上升沿").  A mask that holds SHOOT every frame therefore never provides the
   edge: the run survives, the playfield stays empty, and ``stage_clear`` stays
   ``false`` forever.  The harness does **not** inject the ENTER pulse that
   10-P0 §4.2 describes (``rlharness.cpp`` has no ``0x1000`` path), so the
   client has to send it.

The fix is one frame of "SHOOT released" every ``shoot_pulse_period`` decisions:
the next frame carries SHOOT again, which *is* the rising edge.  Cost is
``1 / (period * frames_per_decision)`` of the firing time (1/96 at the default),
and it is spent in a frame that is otherwise identical, so the plan itself is
untouched.
"""

from __future__ import annotations

from touhou_rl.spec import BTN_BOMB, BTN_SHOOT
from touhou_rl.v400.gym.actions import cand_to_mask

#: 13-P3 §2.2 / D-08: the shoot button is a constant of the action space, not a
#: decision.  Every committed frame sets it unless it is the release frame.
SHOOT_ALWAYS = True

#: ``GameManager.cpp:381-389`` tests ``g_CurFrameInput & 0x1001``: SHOOT (0x1)
#: or ENTER (0x1000).  The agent mask must not contain ENTER (10-P0 §4.2 lists it
#: as system input), so the edge can only come from SHOOT.
STAGE_ADVANCE_BITS = BTN_SHOOT

#: Decisions between two release frames.  24 decisions == 96 frames, i.e. 1.04 %
#: of the frames fire one frame late.  Any period works: the transition state
#: waits indefinitely for the edge, so the pulse does not have to be aligned
#: with the stage end.
SHOOT_PULSE_PERIOD = 24


def release_due(decision_index: int, period: int) -> bool:
    """Is this decision the last one before a SHOOT release frame?"""

    if period is None or int(period) <= 0:
        return False
    return (int(decision_index) + 1) % int(period) == 0


def commit_frames(
    cand: int,
    *,
    bomb: int = 0,
    decision_frames: int = 2,
    release_shoot: bool = False,
) -> list[tuple[int, int]]:
    """``(mask, frames)`` pairs that advance one candidate for ``decision_frames``.

    ``release_shoot`` drops ``BTN_SHOOT`` (and only that bit) from the **last**
    frame, so the following frame's SHOOT bit is a rising edge.  The bomb bit is
    dropped with it: the release frame must not be the frame that spends a bomb.
    """

    if decision_frames <= 0:
        raise ValueError(f"decision_frames must be positive, got {decision_frames}")
    mask = cand_to_mask(int(cand), int(bomb))
    if not release_shoot:
        return [(mask, int(decision_frames))]
    held = mask & ~BTN_SHOOT & ~BTN_BOMB
    if decision_frames == 1:
        return [(held, 1)]
    return [(mask, int(decision_frames) - 1), (held, 1)]


def commit_block(
    cands,
    *,
    bomb: int = 0,
    decision_frames: int = 2,
    release_shoot: bool = False,
) -> list[tuple[int, int]]:
    """The whole decision's ``(mask, frames)`` list.

    ``bomb`` rides the first sub-decision only (11-P1 §4: "the next decision
    presses bomb", not "hold bomb for the whole plan").
    """

    cands = [int(cand) for cand in cands]
    if not cands:
        raise ValueError("a decision needs at least one committed candidate")
    out: list[tuple[int, int]] = []
    for index, cand in enumerate(cands):
        out.extend(
            commit_frames(
                cand,
                bomb=int(bomb) if index == 0 else 0,
                decision_frames=decision_frames,
                release_shoot=bool(release_shoot) and index == len(cands) - 1,
            )
        )
    return out


def commands(frames) -> list[str]:
    """``STEP`` command lines for a ``(mask, frames)`` list."""

    return [f"STEP {int(mask)} {int(count)}" for mask, count in frames]


def mask_summary(frames) -> dict:
    """How many frames of a decision carry SHOOT / BOMB (for labels and tests)."""

    total = sum(int(count) for _, count in frames)
    shoot = sum(int(count) for mask, count in frames if int(mask) & BTN_SHOOT)
    bomb = sum(int(count) for mask, count in frames if int(mask) & BTN_BOMB)
    return {
        "frames": total,
        "shoot_frames": shoot,
        "bomb_frames": bomb,
        "release_frames": total - shoot,
    }
