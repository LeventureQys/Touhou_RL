"""P1 teacher parameters (11-P1 §3 weights, §5 modes).

Every number here is an *initial* value from the specification.  Changing one is
allowed (P1-WP5 tunes them) but the change has to be written into an experiment
card, so the weights live in exactly one place and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from touhou_rl.v400.teacher.shoot import SHOOT_ALWAYS, SHOOT_PULSE_PERIOD

#: One decision == two logic frames (03 §3.3); the engine also produces the +1
#: frame input latency itself, see :mod:`touhou_rl.v400.teacher.geometry`.
DECISION_FRAMES = 2

#: ``cand = focus * 9 + move`` (03 §3.2): nine directions times two focus states.
N_CAND = 18

#: Playfield bounds, 11-P1 §2.2.
PLAYFIELD_X = (8.0, 376.0)
PLAYFIELD_Y = (16.0, 432.0)

#: The margin term saturates at this many pixels (11-P1 §3).
MARGIN_CAP = 16.0


@dataclass(frozen=True)
class Weights:
    """11-P1 §3, term by term.

    ``score(p) = base + min(margin, margin_cap) + align_weight * align
                 + items_weight * items - edge_penalty * edge - high_penalty * high``
    and ``score(dead) = death_frame / horizon * death_scale``.

    EXP-P1-015 adds two *living-plan* terms on top of 11-P1 §3, both zero by
    default so the shipped v1 weights are bit-identical:

    ``bottom``
        a safety band at the playfield bottom.  ``terminal y > bottom_y`` loses
        ``bottom_penalty * (y - bottom_y) / bottom_span`` and, past
        ``bottom_extra_y``, another
        ``bottom_extra_penalty * (y - bottom_extra_y) / bottom_extra_span``.
        Rationale (EXP-P1-015 §1): the margin term saturates at
        :data:`MARGIN_CAP`, so a ship with 16 px of clearance above a wall of
        bullets scores the same as one with 100 px, and 11-P1 §3's only vertical
        term is ``high`` (``y < 200``) — nothing charges for hugging the edge.
    ``escape``
        the **upward** escape room at the terminal position, used only where the
        downward exit is gone (``y > bottom_y``): ``escape_weight`` times the
        clearance to the nearest lethal AABB above the ship's own column,
        saturating at ``escape_span`` px.
    ``boss``
        EXP-P1-017 §2 (v2e): the *Boss body* proxemity cost.  A plan whose
        terminal position is closer than ``boss_radius`` px to any live Boss
        body AABB loses ``boss_penalty * (boss_radius - d) / boss_radius``
        (``d`` = AABB gap from the ship's centre, so ``d == 0`` inside the body
        costs the full weight).  Rationale (EXP-P1-016 §6): 5/6 deaths happened
        inside a Boss fight and Stage6B died 102 px from the body — the body is
        already a lethal AABB, but the margin term saturates at 16 px, so the
        ranking never charged for standing next to a 100 px object.
    ``top`` / ``corner``
        EXP-P1-020 §2 (v2f): the *continuous* top band and its corner
        surcharge, built exactly like ``bottom``.  A plan whose terminal
        ``y < top_y`` loses ``top_penalty * (top_y - y) / top_span`` and, when
        it also ends inside a top corner (``x < corner_x[0]`` or
        ``x > corner_x[1]`` **and** ``y < high_y``), another flat
        ``corner_penalty``.  Rationale (EXP-P1-017 §7): 3/5 of the v2e deaths
        ended at ``py`` 21.7 / 51.8 / 27.3 — the top corner is now the leading
        death mode, while the shipped ``high`` term is a *binary* 3.0 for every
        ``y < 200``: inside the band it has no gradient at all, so a plan that
        parks at ``y = 20`` and one that ends at ``y = 190`` score the same and
        the (capped, 16-point) margin term decides.  ``bottom`` fixed the same
        defect at the other edge (36-67 % -> 11-26 % bottom hugging).
    """

    base: float = 100.0
    margin_cap: float = MARGIN_CAP
    align_weight: float = 4.0
    items_weight: float = 2.0
    items_cap: float = 1.0
    edge_penalty: float = 1.0
    edge_band: float = 24.0
    high_penalty: float = 3.0
    high_y: float = 200.0
    death_scale: float = 10.0
    #: 11-P1 §3: an unverified first action keeps its geometric score minus this.
    unverified_discount: float = 5.0
    #: ``align`` is ``1 - |end_x - target_x| / align_range`` clipped to [0, 1].
    align_range: float = 192.0
    #: EXP-P1-015 §2: bottom safety band (0.0 = off, the v1 behaviour).
    bottom_y: float = 400.0
    bottom_span: float = 32.0
    bottom_penalty: float = 0.0
    bottom_extra_y: float = 420.0
    bottom_extra_span: float = 12.0
    bottom_extra_penalty: float = 0.0
    #: EXP-P1-015 §2 (optional): upward escape room, only inside the bottom band.
    escape_weight: float = 0.0
    escape_span: float = 32.0
    #: EXP-P1-017 §2 (v2e): the Boss-body proxemity cost.  ``0.0`` = off (v1/v2a).
    boss_penalty: float = 0.0
    #: Distance (px) from the Boss body at which the cost reaches zero.
    boss_radius: float = 48.0
    #: EXP-P1-020 §2 (v2f): the continuous top band.  ``0.0`` = off (v1/v2a).
    top_penalty: float = 0.0
    top_y: float = 200.0
    top_span: float = 180.0
    #: EXP-P1-020 §2 (v2f, optional): the flat top-corner surcharge, charged only
    #: while the plan is also inside the ``high`` band (``y < high_y``).
    corner_penalty: float = 0.0
    #: ``(left, right)`` terminal ``x`` thresholds of "inside a top corner".
    corner_x: tuple[float, float] = (64.0, 312.0)
    #: EXP-P1-021 §2 (v2h): the **laser** cost.  Charged from the gap between the
    #: plan's terminal position and the nearest laser the engine marked
    #: ``damageable`` on that frame; ``0.0`` = off (every profile before v2h), and
    #: a frame whose lasers are still fading in costs nothing because those rows
    #: carry ``damageable = 0``.
    laser_penalty: float = 0.0
    #: Distance (px) at which the laser cost reaches zero.
    laser_band: float = 8.0

    def top_penalty_at(self, y: float) -> float:
        """Points lost for ending at ``y`` (>= 0; 0 below the band).

        Exactly the shape of :meth:`bottom_penalty_at`: linear in the depth
        inside the band, zero at ``top_y``.  ``top_y``/``top_span`` are the
        published 200/180, so ``y = 20`` costs the full ``top_penalty`` and the
        playfield top (16 px) costs ``top_penalty * 184 / 180`` — the bottom band
        is not clipped either.
        """

        if self.top_penalty <= 0.0 or float(y) >= self.top_y:
            return 0.0
        span = max(float(self.top_span), 1e-9)
        return float(self.top_penalty) * (float(self.top_y) - float(y)) / span

    def corner_penalty_at(self, x: float, y: float) -> float:
        """The flat top-corner surcharge (0 unless inside a corner)."""

        if self.corner_penalty <= 0.0 or float(y) >= float(self.high_y):
            return 0.0
        if float(x) < self.corner_x[0] or float(x) > self.corner_x[1]:
            return float(self.corner_penalty)
        return 0.0

    def laser_penalty_at(self, gap: float | None) -> float:
        """Points lost for ending ``gap`` px from the nearest damageable laser.

        EXP-P1-021.  ``None`` or ``inf`` (no laser with a live hitbox on the
        terminal frame) costs nothing; ``gap <= 0`` (the hurtbox touches the
        beam's rectangle) costs the full ``laser_penalty``.  The gate that
        decides which lasers count is the **engine's** ``damageable`` field —
        this helper only grades the resulting distance.
        """

        if self.laser_penalty <= 0.0 or gap is None:
            return 0.0
        span = max(float(self.laser_band), 1e-9)
        clipped = min(max(span - float(gap), 0.0), span)
        return float(self.laser_penalty) * clipped / span

    def boss_penalty_at(self, distance: float | None) -> float:
        """Points lost for ending ``distance`` px from the nearest Boss body.

        ``None`` or ``inf`` (no Boss body on the terminal frame) costs nothing;
        ``distance <= 0`` (inside the body) costs the full ``boss_penalty``.
        """

        if self.boss_penalty <= 0.0 or distance is None:
            return 0.0
        span = max(float(self.boss_radius), 1e-9)
        clipped = min(max(span - float(distance), 0.0), span)
        return float(self.boss_penalty) * clipped / span

    def bottom_penalty_at(self, y: float) -> float:
        """Points lost for ending at ``y`` (>= 0; 0 above the band)."""

        penalty = 0.0
        if self.bottom_penalty > 0.0 and float(y) > self.bottom_y:
            span = max(float(self.bottom_span), 1e-9)
            penalty += self.bottom_penalty * (float(y) - self.bottom_y) / span
        if self.bottom_extra_penalty > 0.0 and float(y) > self.bottom_extra_y:
            span = max(float(self.bottom_extra_span), 1e-9)
            penalty += self.bottom_extra_penalty * (float(y) - self.bottom_extra_y) / span
        return penalty


@dataclass(frozen=True)
class TeacherConfig:
    """One operating mode of the teacher (11-P1 §5)."""

    mode: str
    horizon: int
    beam: int
    verify_k: int
    max_iterations: int
    replan_decisions: int = 2
    decision_frames: int = DECISION_FRAMES
    #: How long the first action is held, in decision slots (11-P1 §2: "18 个首
    #: 动作 × 持续时间").  Every (action, duration) pair is one beam-search root.
    durations: tuple[int, ...] = (1, 2)
    border_x: tuple[float, float] = PLAYFIELD_X
    border_y: tuple[float, float] = PLAYFIELD_Y
    weights: Weights = field(default_factory=Weights)
    #: Items need ``ITEMS`` paging on every recorded frame; off by default so the
    #: tape stays two round trips per frame.  When off, ``items(p) = 0`` for every
    #: plan, which leaves the ranking unchanged (a constant term).
    record_items: bool = False
    #: 11-P1 §4: the two bomb rules.  ``bomb_death`` additionally needs
    #: ``frameEventFlags`` from ``STATE``; the harness does not export it, so the
    #: label carries ``bomb_death_available=False`` instead of guessing.
    bomb_active: bool = True
    bomb_death: bool = True
    #: EXP-P1-015 §3: rule 1 ("every verified plan dies") additionally requires
    #: the **shortest** verified death to be at most this many frames away.
    #: ``None`` keeps 11-P1 §4 verbatim (bomb as soon as no plan survives).
    #: The v1 logs show rule 1 firing 27-47 frames before the predicted death on
    #: every bomb of Stage4A/Stage6A — i.e. bombs spent on a 48-frame forecast
    #: that replanning would very likely have avoided.
    bomb_rule1_max_death_frame: int | None = None
    #: EXP-P1-015 §3b (v2c): rule 1 also needs ``bombs > bomb_rule1_min_bombs_left``
    #: so the **last** bomb is never spent on a forecast — it is kept for the
    #: death-bomb of 11-P1 §4 rule 2 (a real ``PLAYER_HIT``).  ``0`` is the v1
    #: behaviour ("any bomb left is spendable"); v2c uses ``2``.
    bomb_rule1_min_bombs_left: int = 0
    #: EXP-P1-015 §3b (v2c): rule 1 only fires when the plan's own margin is below
    #: this many pixels (``None`` = no margin condition).  The v1 death analysis
    #: (EXP-P1-016) measured 0.9-11.2 px of margin at the six deaths, while every
    #: early bomb fired at 6.0-16.0 px with a 13-47 frame forecast.
    bomb_rule1_max_margin: float | None = None
    #: EXP-P1-015 §2b (v2c): the **hard corridor**.  When set, a plan whose
    #: terminal ``y`` is outside ``(top, bottom)`` cannot win its root unless no
    #: in-corridor survivor exists (then the soft band decides, as in v2a/v2b).
    #: ``None`` = no clip (v1/v2a/v2b).
    clip_y: tuple[float, float] | None = None
    #: 13-P3 §2.2 / D-08: SHOOT is a constant of the action space, never a
    #: decision.  ``cand_to_mask`` sets it on every frame, so this flag documents
    #: the fact instead of switching it (EXP-P1-009 §2).
    shoot_always: bool = SHOOT_ALWAYS
    #: 10-P0 §4.2 / ``GameManager.cpp:381-389``: leaving the end-of-stage state
    #: needs a **rising edge** of ``SHOOT|ENTER``, so a mask that holds SHOOT
    #: every frame can never complete a stage.  Every ``shoot_pulse_period``
    #: decisions the last committed frame drops SHOOT; the next frame's SHOOT is
    #: the edge.  ``0`` disables the pulse (the pre-EXP-P1-009 behaviour).
    shoot_pulse_period: int = SHOOT_PULSE_PERIOD

    @property
    def slots(self) -> int:
        """Decision slots inside one horizon (``horizon / decision_frames``)."""
        return self.horizon // self.decision_frames

    def __post_init__(self) -> None:
        if self.horizon % self.decision_frames:
            raise ValueError("horizon must be a whole number of decisions")
        if self.replan_decisions < 1:
            raise ValueError("replan_decisions must be >= 1")
        if self.beam < 1 or self.verify_k < 1 or self.max_iterations < 1:
            raise ValueError("beam / verify_k / max_iterations must be >= 1")
        if not self.durations:
            raise ValueError("at least one first-action duration is required")
        if self.shoot_pulse_period < 0:
            raise ValueError(f"shoot_pulse_period must be >= 0, got {self.shoot_pulse_period}")
        if self.clip_y is not None:
            top, bottom = (float(value) for value in self.clip_y)
            if not (0.0 <= top < bottom <= PLAYFIELD_Y[1]):
                raise ValueError(f"clip_y must touch the playfield, got {self.clip_y}")
        if self.bomb_rule1_min_bombs_left < 0:
            raise ValueError("bomb_rule1_min_bombs_left must be >= 0")


#: 11-P1 §5: full mode — G1 evaluation and data production.
FULL = TeacherConfig(mode="full", horizon=48, beam=64, verify_k=3, max_iterations=3)

#: 11-P1 §5: fast mode — DAgger labelling.
FAST = TeacherConfig(mode="fast", horizon=32, beam=24, verify_k=1, max_iterations=1)

#: EXP-P3-004 §3, profile **fast3**: ``FAST`` **verbatim** except ``verify_k``
#: 1 -> 3.  It exists so the LABEL-SIGNAL 2x2 can hold everything else fixed:
#:
#: * ``verify_k=1`` verifies exactly one action per decision, so 17 of 18
#:   candidates are ``-5.0`` (``Weights.unverified_discount``, 11-P1 §3) and the
#:   argmax is 100 % "which action happened to be verified" whenever the true
#:   geometric top-1-top-2 gap is below 5.0 — measured on the whole 76,912-row
#:   D0 + DAgger aggregate, that is 82.19 % / 55.55 % of the rows and the real
#:   geometric gap has p90 = 0.059 (EXP-P3-003 §8.3);
#: * with three verified actions their scores can actually be compared, and
#:   ``verified_count`` stops being the label's only degree of freedom.
#:
#: The search budget is deliberately **not** widened (``horizon=32``,
#: ``beam=24``, ``max_iterations=1``, same ``Weights``): ``v2a``/``v3a`` would
#: change four things at once (horizon, beam, weights, bomb window), which cannot
#: answer "does verify_k provide the signal".  ``FAST`` and ``V2A`` are untouched
#: — ``FAST3`` is a third, separately registered profile.
FAST3 = TeacherConfig(mode="fast3", horizon=32, beam=24, verify_k=3, max_iterations=1)

#: EXP-P1-015 §2, profile **v2a**: the bottom safety band only, plus the §3 bomb
#: window.  At ``y = 432`` the band costs ``1.5 + 2.0 = 3.5`` points, which is
#: 22 % of the whole margin term and larger than the measured margin advantage of
#: the bottom band (14.46 px at 420-432 vs 15.47 px at 350-400, EXP-P1-015 §1).
_V2A_WEIGHTS = Weights(bottom_penalty=1.5, bottom_extra_penalty=2.0)
V2A = TeacherConfig(
    mode="v2a",
    horizon=48,
    beam=64,
    verify_k=3,
    max_iterations=3,
    weights=_V2A_WEIGHTS,
    bomb_rule1_max_death_frame=12,
)

#: EXP-P1-015 §2, profile **v2b**: a stronger band (``2.5 + 3.5 = 6.0`` points at
#: ``y = 432``) *plus* the upward-escape term, so a bottom position that still has
#: an open upward exit is not punished as hard as a dead corner.
_V2B_WEIGHTS = Weights(
    bottom_penalty=2.5,
    bottom_extra_penalty=3.5,
    escape_weight=1.0,
    escape_span=32.0,
)
V2B = TeacherConfig(
    mode="v2b",
    horizon=48,
    beam=64,
    verify_k=3,
    max_iterations=3,
    weights=_V2B_WEIGHTS,
    bomb_rule1_max_death_frame=12,
)

#: EXP-P1-015 §2b, profile **v2c**: v2a's band kept the ship out of the bottom but
#: round 1 showed it then dying at ``py`` 38-52 (the *top* corner).  v2c therefore
#: makes the corridor between ``clip_y`` a **hard constraint** on the terminal
#: position, keeps the last bomb (§3b), and makes rule 1 margin-driven.
_V2C_WEIGHTS = Weights(bottom_penalty=1.5, bottom_extra_penalty=2.0)
V2C = TeacherConfig(
    mode="v2c",
    horizon=48,
    beam=64,
    verify_k=3,
    max_iterations=3,
    weights=_V2C_WEIGHTS,
    bomb_rule1_max_death_frame=24,
    bomb_rule1_min_bombs_left=1,
    bomb_rule1_max_margin=12.0,
    clip_y=(200.0, 400.0),
)

#: EXP-P1-015 §2c, profile **v2d**: v2c's *hard* corridor was measured worse on
#: 3/3 faces in round 2 (Stage5 21,504 -> 4,456 frames, Stage6A 13,820 -> 6,488,
#: Stage6B 17,564 -> 6,536), i.e. clipping the terminal position removes the
#: escape options instead of adding safety.  v2d keeps the whole field but makes
#: the **top** band as expensive as the bottom one (``high_penalty`` 3.0 -> 8.0),
#: so the ship cannot park in the top corner either; it also carries the §3b bomb
#: guards (keep the last bomb, margin-driven rule 1).
_V2D_WEIGHTS = Weights(bottom_penalty=1.5, bottom_extra_penalty=2.0, high_penalty=8.0)
V2D = TeacherConfig(
    mode="v2d",
    horizon=48,
    beam=64,
    verify_k=3,
    max_iterations=3,
    weights=_V2D_WEIGHTS,
    bomb_rule1_max_death_frame=24,
    bomb_rule1_min_bombs_left=1,
    bomb_rule1_max_margin=12.0,
)

#: EXP-P1-017 §2, profile **v2e**: v2a verbatim (the 1.5 / 2.0 bottom band and
#: the 12-frame rule-1 window) **minus** v2c/v2d's "keep the last bomb" and
#: margin gate, **plus** a Boss-body proxemity cost.  Why each piece (EXP-P1-015
#: §7, EXP-P1-016 §6):
#:
#: * the bottom band is kept: it moved the end-of-run ``py`` from 36-67 % to
#:   11-26 % bottom-hugging and pushed Stage3/5/6B to their best frames;
#: * "keep the last bomb" is dropped: keeping a bomb measured **harmful** — the
#:   same Stage5 face survived 21,504 frames with 3 bombs spendable and 2,428
#:   with 0, and 5/5 of the v2a deaths happened *after* the bombs were gone;
#: * the 12-frame rule-1 window is kept (it is what stopped the 27-47 frame
#:   forecast bombs that cost 4/6 v1 runs a bomb for nothing);
#: * the Boss-body cost is the only *new* term: 5/6 deaths were inside a Boss
#:   fight and Stage6B died 102 px from the body, which no living-plan term
#:   charged for (``margin`` saturates at 16 px).
#:
#: Implementation choice (EXP-P1-017 §2): **candidate-endpoint cost** (option b)
#: instead of removing the Boss from ``align``'s target (option a), because the
#: Boss is where the bullets come from — penalising "standing next to the body"
#: is the hazard we measured, while changing ``align``'s target would also change
#: every stage's *midboss* behaviour (the align term would silently retarget), and
#: a stage-local change must not leak into the six faces that already pass.
_V2E_WEIGHTS = Weights(bottom_penalty=1.5, bottom_extra_penalty=2.0, boss_penalty=3.0)
V2E = TeacherConfig(
    mode="v2e",
    horizon=48,
    beam=64,
    verify_k=3,
    max_iterations=3,
    weights=_V2E_WEIGHTS,
    bomb_rule1_max_death_frame=12,
)

#: EXP-P1-020 §2, profile **v2f**: v2a verbatim (bottom band ``1.5 / 2.0``,
#: rule-1 window 12, no keep-last-bomb, no margin gate, no hard clip) with the
#: **continuous top band** built like the bottom one, and a small flat top-corner
#: surcharge:
#:
#: * ``top_penalty = 1.5`` over ``top_y = 200`` / ``top_span = 180``: a terminal
#:   ``y = 20`` loses 1.5 points, ``y = 110`` loses 0.75, ``y = 200`` loses 0;
#: * ``corner_penalty = 1.0`` while the terminal ``x`` is inside
#:   ``(64, 312)``'s complement **and** ``y < high_y``;
#: * the shipped binary ``high_penalty = 3.0`` is **kept**, not replaced: it is
#:   the band-entry toll that keeps every v2a face's mid-field ranking intact (a
#:   plan at ``y = 190`` is still charged 3.0 exactly as in v2a), while the new
#:   linear term supplies the gradient the binary term cannot have.  At the
#:   extreme top corner the two plus ``edge`` cost ``3.0 + 1.5 + 1.0 (+1.0)``,
#:   i.e. 43-53 % of the whole 16-point margin term, against the 4.0 a v2a
#:   corner already paid for a 16-point margin edge (EXP-P1-017 §7 point 2).
#:
#: Why 1.5 and not 8.0 (v2d): v2d raised the *binary* ``high_penalty`` to 8.0,
#: which charges a plan at ``y = 199`` the same 8.0 as one at ``y = 20`` and was
#: measured on top of the harmful keep-last-bomb/margin gates, so its effect is
#: confounded (EXP-P1-015 §7.1); 1.5 is the same strength the bottom band proved
#: (36-67 % -> 11-26 % hugging) and leaves the six already-passing faces' options
#: intact.
_V2F_WEIGHTS = Weights(
    bottom_penalty=1.5,
    bottom_extra_penalty=2.0,
    top_penalty=1.5,
    corner_penalty=1.0,
)
V2F = TeacherConfig(
    mode="v2f",
    horizon=48,
    beam=64,
    verify_k=3,
    max_iterations=3,
    weights=_V2F_WEIGHTS,
    bomb_rule1_max_death_frame=12,
)

#: EXP-P1-020 §2, profile **v2g**: v2f with the top band at ``3.0`` (the second
#: hint of the task, "若首轮不够再试 2.5/3.0 一档").  Only run if v2f still leaves
#: more than four deaths and the budget allows a second full 8-face round.
_V2G_WEIGHTS = Weights(
    bottom_penalty=1.5,
    bottom_extra_penalty=2.0,
    top_penalty=3.0,
    corner_penalty=1.0,
)
V2G = TeacherConfig(
    mode="v2g",
    horizon=48,
    beam=64,
    verify_k=3,
    max_iterations=3,
    weights=_V2G_WEIGHTS,
    bomb_rule1_max_death_frame=12,
)

#: EXP-P1-019 §2, profile **v3a**: **v2a's weights verbatim** — not a single
#: scoring term changes — with a larger **search budget**: ``horizon`` 48 -> 64
#: and ``beam`` 64 -> 96 (``verify_k`` stays 3).  EXP-P1-015 §10.4 lists "give
#: the search more budget" as the fourth thing to try before touching the
#: weights again, because the full profile already survived 56 % of the Stage5
#: cap and 32 % of the Stage6B cap: the weights were steering, the search was the
#: thing running out of runway.  ``_V2A_WEIGHTS`` is shared (not copied) so the
#: two profiles provably have the same numbers.
V3A = TeacherConfig(
    mode="v3a",
    horizon=64,
    beam=96,
    verify_k=3,
    max_iterations=3,
    weights=_V2A_WEIGHTS,
    bomb_rule1_max_death_frame=12,
)

#: EXP-P1-019 §2, profile **v3b**: v3a's horizon plus a wider beam **and** one
#: more engine-verified plan per decision: ``beam`` 96 -> 128, ``verify_k`` 3 -> 4.
#: The extra verification slot matters because 11-P1 §4 rule 1 fires on
#: "every *verified* plan dies": with three forks a 4th live plan is invisible to
#: the rule, so widening the beam without widening ``k`` would keep spending
#: bombs on forecasts the 4th plan would have refuted.
V3B = TeacherConfig(
    mode="v3b",
    horizon=64,
    beam=128,
    verify_k=4,
    max_iterations=3,
    weights=_V2A_WEIGHTS,
    bomb_rule1_max_death_frame=12,
)

#: EXP-P1-021 §2, profile **v2h**: **v2a's weights verbatim** with the one thing
#: EXP-P1-016 measured as missing from the planner — a laser term.
#:
#: Why this profile exists at all: the P1 laser row exported ``state`` and
#: ``timer`` but not the judgement window, so a ``LASER_STATE_STARTING`` beam
#: that had already passed ``hitboxStartTime`` and a ``LASER_STATE_DESPAWNING``
#: tail still inside ``hitboxEndDelay`` were indistinguishable from harmless
#: ones, and the search — which only ever consumed bullet and enemy-body AABBs —
#: saw no laser hazard whatsoever.  Stage 6A died **to lasers** (EXP-P1-016:
#: 51 live slots, gap 0.0 on the death frame), so "no laser in the score" is a
#: structural blind spot, not a tuning question.
#:
#: The fix is two-sided and both sides are the engine's own numbers:
#:
#: * the row now carries ``startTime``/``hitboxStartTime``/``duration``/
#:   ``despawnDuration``/``hitboxEndDelay`` **and** the engine-computed
#:   ``damageable`` flag, so "does this beam have a hitbox on this frame" is a
#:   fact, not a client-side guess;
#: * those rows become one extra terminal-frame cost, built exactly like the
#:   proven v2e Boss cost (``clip(band - gap, 0, band) / band``), so a plan that
#:   ends inside a live beam pays ``laser_penalty`` and one in clear air pays
#:   nothing.  A fading-in beam is *not* charged because its ``damageable`` is 0
#:   — that is the "do not be over-conservative" half of the task.
#:
#: ``laser_penalty = 12.0`` and ``laser_band = 8.0``: inside the beam the cost
#: is 75 % of the whole 16-point margin term, i.e. it dominates a 12 px margin
#: advantage, while a plan 8 px away pays exactly 0.  The two bottom-band numbers
#: are copied from ``_V2A_WEIGHTS`` verbatim (``1.5 / 2.0``) and every other
#: weight keeps its v2a value, so
#: ``dataclasses.replace(V2H.weights, laser_penalty=0.0) == V2A.weights`` — the
#: unit test pins that equality, i.e. the laser term is the only difference.
_V2H_WEIGHTS = Weights(
    bottom_penalty=1.5,
    bottom_extra_penalty=2.0,
    laser_penalty=12.0,
    laser_band=8.0,
)
V2H = TeacherConfig(
    mode="v2h",
    horizon=48,
    beam=64,
    verify_k=3,
    max_iterations=3,
    weights=_V2H_WEIGHTS,
    bomb_rule1_max_death_frame=12,
)

#: EXP-P3-005 §2, profile **v2i**: the **margin-resolution** profile.  ``v2h``
#: verbatim except :attr:`Weights.margin_cap` 16 -> 64 — the single change the
#: label-signal post-mortem (EXP-P3-004 §7.3, P0) asked for, because 82 % of the
#: decisions have an 18-way *geometric* top1-top2 gap below 0.06 px while the
#: shipped term saturates at 16 px (EXP-P1-003 already recorded "margin 全顶 16").
#:
#: Why a **linear** cap and not ``log1p(margin)`` (the two options in the task):
#:
#: * every other term of 11-P1 §3 is *affine in pixels or in a 0..1 ratio*
#:   (``align`` 4.0, ``bottom`` 1.5/2.0, ``high`` 3.0, ``laser`` 12.0).  A linear
#:   margin term keeps the whole score an affine function of clearance, so the
#:   published weights keep their published meaning ("1 px of clearance = 1
#:   point") and a 16 -> 64 px cap is exactly "the same term with 4x the range";
#: * ``log1p`` compresses where the data actually is: ``log1p(16) = 2.83`` and
#:   ``log1p(64) = 4.17`` — a 1.3-point total dynamic range, i.e. *smaller* than
#:   ``edge`` (1.0) + ``high`` (3.0) and 9x smaller than the laser term (12.0).
#:   Its "equivalent dimension" would be ``1/lu`` points per pixel at margin
#:   ``mu`` (``d/dm log1p(m) = 1/(1+m)``: 0.059 pt/px at 16 px, 0.015 at 64), so
#:   the same clearance difference would be worth different amounts depending on
#:   where it sits — that is a *new* nonlinearity in a weight table that was
#:   measured under a linear 16-point term (EXP-P1-015/017/020/021 all calibrated
#:   bands against "the whole 16-point margin term"), and it cannot be expressed
#:   by any existing ``Weights`` field, i.e. it would not be a label experiment
#:   that differs in one number.
#:
#: **Structural caveat (measured, EXP-P3-005 §3).**  ``Weights.margin_cap`` is the
#: *last* clamp on an already-clamped number: ``search._margin_rows`` /
#: ``_frame_margin`` cap the per-frame clearance at the module constant
#: :data:`MARGIN_CAP` (16 px), the raster table caps it again at
#: ``raster.MARGIN_Q_CAP = 64`` (16 px in 1/4-px fixed point, stored in the low
#: byte of a ``uint16``), and the beam keys are built from ``1000 + MARGIN_CAP -
#: margin``.  So ``margin_cap = 64.0`` alone cannot move a single score; the
#: profile is registered exactly as the task specified and EXP-P3-005 measures
#: what its labels are instead of assuming the clamp moved.  Raising it for real
#: needs ``MARGIN_CAP`` itself raised (and the raster byte budget respected:
#: ``MARGIN_CAP * 4 <= 255``, i.e. **<= 63.75 px**, or the margin overflows into
#: the flag byte).
_V2I_WEIGHTS = replace(_V2H_WEIGHTS, margin_cap=64.0)
V2I = TeacherConfig(
    mode="v2i",
    horizon=48,
    beam=64,
    verify_k=3,
    max_iterations=3,
    weights=_V2I_WEIGHTS,
    bomb_rule1_max_death_frame=12,
)

#: EXP-P3-007 §1, profile **v2j**: the **action-duration** profile.  ``v2h``
#: **verbatim** -- the same ``Weights`` *instance* (so a later edit cannot drift
#: one copy), the same ``horizon``/``beam``/``verify_k``/``max_iterations``, the
#: same rule-1 window -- with exactly one field moved:
#:
#:     ``durations = (1, 2)``  ->  ``(1, 2, 3, 4)``
#:
#: Why this is the minimal change EXP-P3-005 §8 asks for (P0): the 18 candidates
#: of 11-P1 §3 are *almost the same action*.  At ``decision_frames = 2`` a
#: 1-2 slot first action only displaces the ship 2-8 px, while the margin term
#: saturates at :data:`MARGIN_CAP` (16 px) and ``align`` is worth
#: ``4.0 / 192 = 0.021`` points per pixel.  Holding the first action for up to
#: four slots (8 frames, ~16 px) doubles-to-quadruples the endpoint spread, so
#: (a) ``align``/``bottom``/``top`` get a real gradient across the 18 candidates
#: and (b) some plans end with **less** than 16 px of clearance, i.e. inside the
#: region where the margin term is still alive.  No engine, protocol, search or
#: weight change is involved: only the root grid widens (18 x 2 = 36 roots ->
#: 18 x 4 = 72), which is the ~2x search cost EXP-P3-005 §8 budgeted.
#:
#: Nothing about the *executed* plan changes shape: ``decide``/the CLI commit
#: ``replan_decisions = 2`` slots either way (11-P1 §2 replans every R slots), so
#: durations 2/3/4 commit the same two masks and differ only in how far the plan
#: assumed the hold would carry -- which is the information the score needs.
_V2J_DURATIONS = (1, 2, 3, 4)
V2J = TeacherConfig(
    mode="v2j",
    horizon=48,
    beam=64,
    verify_k=3,
    max_iterations=3,
    weights=_V2H_WEIGHTS,
    bomb_rule1_max_death_frame=12,
    durations=_V2J_DURATIONS,
)

CONFIGS = {
    FULL.mode: FULL,
    FAST.mode: FAST,
    FAST3.mode: FAST3,
    V2A.mode: V2A,
    V2B.mode: V2B,
    V2C.mode: V2C,
    V2D.mode: V2D,
    V2E.mode: V2E,
    V2F.mode: V2F,
    V2G.mode: V2G,
    V2H.mode: V2H,
    V2I.mode: V2I,
    V2J.mode: V2J,
    V3A.mode: V3A,
    V3B.mode: V3B,
}

#: EXP-P1-015: the profiles that are *candidates*, in the order they were tested.
#: ``full`` stays the v1 reference and is never redefined.  ``v2b`` is kept for the
#: record (it has unit tests) but was superseded by v2c before its round: round 1
#: falsified its premise — the band was strong enough, it was pushing the ship into
#: the *top* corner, so the fix had to be a corridor, not a stronger band.
#: ``v2i`` (EXP-P3-005) is the margin-resolution profile, a candidate exactly like
#: the others: it is ``v2h`` with one weight changed.  ``v2j`` (EXP-P3-007) is
#: ``v2h`` with one *duration* changed: the narrow action set is the second source
#: of the 18-way score spread measured in EXP-P3-005 §7.2.
PROFILES = (V2A, V2B, V2C, V2D, V2E, V2F, V2G, V2H, V2I, V2J, V3A, V3B)

#: 10-P0 §5, 11-P1 §2.1: TH08 player states.  ``ALIVE`` is the only one that can
#: die; ``INVULNERABLE`` still gets planned for (the position is part of the
#: label) but is never scored as a death.
PLAYER_STATE_ALIVE = 0
PLAYER_STATE_DYING = 2
PLAYER_STATE_INVULNERABLE = 3


def get_config(mode: str) -> TeacherConfig:
    try:
        return CONFIGS[mode]
    except KeyError:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(CONFIGS)}") from None


