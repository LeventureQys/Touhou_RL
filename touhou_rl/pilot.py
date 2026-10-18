"""Scripted avoidance teacher for TH08 ("the pilot").

Why this module exists
----------------------
v1.0.2 taught a network to copy one human Lunatic trajectory and measured the
result on the live game: 1,330 frames of survival versus 2,418 for a uniformly
random policy (55%).  The diagnosis was not "not enough data" but "the policy
never learned to avoid bullets": every life lasted about seven seconds, i.e.
about the time it takes a stationary target to be hit.

Before spending hours of wall-clock game time collecting more demonstrations,
this module answers a cheaper and more basic question:

    is the observation channel (bullet positions + velocities + radii) and the
    control channel (8-way move, shoot, focus, bomb) *information-theoretically*
    sufficient to dodge Lunatic patterns?

It answers it by being a hand-written policy that only does what the network is
also allowed to do: look at the bullets, predict where they will be in a few
frames, and pick a movement that keeps the hurtbox away from them.  If this
policy survives minutes instead of seconds, the channel is sufficient and the
remaining problem is learning; if it also dies in seconds, the observation is
missing something (enemy positions, spawn sources, lasers) and no amount of
imitation data will fix it.

How it works
------------
Bullet motion in TH08 is ``position += velocity`` per frame for the majority of
bullet types, so a linear extrapolation over the planning horizon (default
``horizon_steps * frameskip`` = 20 frames = 0.33 s) is a good approximation.
The pilot runs a beam search over the 9 movement directions times the two focus
states, scores each partial path by the *minimum* clearance (distance from the
player hurtbox to the nearest bullet surface) reached anywhere along it, and
executes the first action of the best path.

The score is deliberately a worst-case rather than an average: in a bullet hell
game the average is irrelevant, a single collision ends the life.

Bombs
-----
If every candidate path collides (best clearance < ``bomb_min_clearance``) the
pilot bombs, which clears bullets and grants invincibility.  Survival is worth
more than score here, so the threshold is a hard "unavoidable" test rather than
an optimisation.

Units and normalisation
-----------------------
The pilot can be driven either by the privileged observation dict (``bullets``
is 64 nearest bullets, normalised as in :mod:`touhou_rl.obs`) or by the raw
bullet snapshot the environment exposes as ``info["bullets_raw"]`` (all active
bullets, absolute playfield coordinates).  Working from the raw snapshot is
strictly better (in dense patterns the 64 nearest bullets can be a small ring
around the player and hide the column it is about to move into), so the live
evaluation path uses that; the observation path exists so the same code can be
replayed offline against a recorded dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import spec

# Normalisation factors used by :mod:`touhou_rl.obs`; kept here so the pilot can
# decode an observation dict without importing the encoder (and its coupling).
OBS_DX_SCALE = 192.0
OBS_DY_SCALE = 224.0
OBS_VEL_SCALE = 6.0
OBS_HALF_SCALE = 8.0

# Direction unit vectors for movement classes 0..8 (screen: +x right, +y down).
_MOVE_VECTORS = np.array(
    [
        [0.0, 0.0],  # 0 idle
        [0.0, -1.0],  # 1 north
        [1.0, -1.0],  # 2 north-east
        [1.0, 0.0],  # 3 east
        [1.0, 1.0],  # 4 south-east
        [0.0, 1.0],  # 5 south
        [-1.0, 1.0],  # 6 south-west
        [-1.0, 0.0],  # 7 west
        [-1.0, -1.0],  # 8 north-west
    ],
    dtype=np.float32,
)


@dataclass
class PilotConfig:
    """Tunables of the avoidance teacher (all distances in pixels, all times in game frames)."""

    frameskip: int = 2
    horizon_steps: int = 10  # * frameskip = 20 frames = 0.33 s at 60 fps
    beam: int = 24  # kept for reference; selection is per-action quota below
    quota_per_action: int = 2  # continuations kept for each of the 18 first actions
    # Player speeds per frame, measured on this build for Border Team (Lunatic)
    # by tools/probes/calibrate_player.py:
    #   unfocused axis 4.000 px/frame, diagonal 2.8284 (= axis / sqrt(2))
    #   focused   axis 2.000 px/frame, diagonal 1.4142
    # Diagonal components are per-axis, matching spec.MOVES (which presses both
    # arrow keys at once).
    speed: float = 4.0
    speed_diag: float = 2.8284
    focus_speed: float = 2.0
    focus_speed_diag: float = 1.4142
    # Player::size measured as 0.825 px half-extent in both focus states (the
    # hitbox does *not* shrink when focusing in TH08 - only the speed does).
    player_half: float = 0.825
    # Bullets are approximated as circles of radius ``max(half_x, half_y)``.
    # The pad absorbs the box-vs-circle mismatch of the real hitbox.
    bullet_pad: float = 0.5
    near_radius: float = 240.0
    max_bullets: int = 320
    # Tie-breakers, deliberately tiny and *bounded* so they never override
    # safety.  Both are normalised to the playfield and then added to the
    # clearance score, so together they can shift the ranking by at most
    # ``bottom_bias + aim_weight`` pixels of clearance - a policy that would
    # trade a 1 px clearance for a better shooting position is not a dodger.
    bottom_bias: float = 0.30  # prefer staying low (more reaction time)
    # Aiming is implemented (boss x pulls the path horizontally) but OFF by
    # default: measured 2026-10-18, 2 live Lunatic episodes with aim_weight=0.6
    # survived 1,719 / 4,067 frames versus 7,319 / 6,748 with it off, and the
    # stage still did not advance.  "Stay under the boss" trades safety for DPS
    # at a bad exchange rate; a real aiming term has to be a constraint applied
    # only when nothing is close, not a constant bias.
    aim_weight: float = 0.0
    switch_penalty: float = 0.05  # score -= switch_penalty when direction changes
    bomb_min_clearance: float = -1.5  # bomb when even the best path is this bad
    bomb_cooldown: int = 40  # decisions to wait after a bomb
    always_shoot: bool = True

    def validate(self) -> None:
        if self.frameskip < 1:
            raise ValueError("frameskip must be >= 1")
        if self.horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")
        if self.beam < 1:
            raise ValueError("beam must be >= 1")
        if self.speed <= 0.0 or self.focus_speed <= 0.0:
            raise ValueError("player speeds must be positive")


@dataclass
class PilotDebug:
    """Diagnostics of the last decision (used by the probe/report, not by the env)."""

    best_clearance: float = 0.0
    bullets_used: int = 0
    bombs: int = 0


@dataclass
class AvoidPilot:
    """Beam-search avoidance policy; see the module docstring."""

    config: PilotConfig = field(default_factory=PilotConfig)
    _prev_move: int = 0
    _bomb_cooldown: int = 0
    debug: PilotDebug = field(default_factory=PilotDebug)

    def __post_init__(self) -> None:
        self.config.validate()
        # (2, 9) per-frame displacement for [focus=0, focus=1] x directions.
        cfg = self.config
        axis = np.array([cfg.speed, cfg.focus_speed], dtype=np.float32)
        diag = np.array([cfg.speed_diag, cfg.focus_speed_diag], dtype=np.float32)
        disp = np.empty((2, 9, 2), dtype=np.float32)
        for focus in (0, 1):
            for move in range(9):
                ux, uy = _MOVE_VECTORS[move]
                if ux != 0.0 and uy != 0.0:
                    comp = diag[focus]
                else:
                    comp = axis[focus]
                disp[focus, move, 0] = ux * comp
                disp[focus, move, 1] = uy * comp
        # Per decision the same action is held for ``frameskip`` frames.
        self._step_disp = disp * float(cfg.frameskip)

    # -- input decoding ---------------------------------------------------
    def _bullets_from_raw(self, raw: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = np.asarray(raw["position"], dtype=np.float32).reshape(-1, 2)
        vel = np.asarray(raw["velocity"], dtype=np.float32).reshape(-1, 2)
        half = np.asarray(raw["half"], dtype=np.float32).reshape(-1)
        return pos, vel, half

    def _bullets_from_obs(self, obs: dict, player: tuple[float, float]):
        feat = np.asarray(obs["bullets"], dtype=np.float32).reshape(-1, 6)
        valid = feat[:, 5] > 0.5
        feat = feat[valid]
        dx = feat[:, 0] * OBS_DX_SCALE
        dy = feat[:, 1] * OBS_DY_SCALE
        pos = np.stack([dx + player[0], dy + player[1]], axis=1)
        vel = feat[:, 2:4] * OBS_VEL_SCALE
        half = np.maximum(feat[:, 4] * OBS_HALF_SCALE, 0.5)
        return pos, vel, half

    # -- geometry ---------------------------------------------------------
    @staticmethod
    def _player_position(obs: dict) -> tuple[float, float]:
        feat = np.asarray(obs["self_feat"], dtype=np.float32)
        px = spec.PLAYFIELD_X + float(feat[0]) * spec.PLAYFIELD_W
        py = spec.PLAYFIELD_Y + float(feat[1]) * spec.PLAYFIELD_H
        # Clamp the decoded position into the playfield: rel_x is normalised by
        # the *field* size, so tiny float error at the edges would otherwise
        # push the planning origin outside the real movement area.
        px = float(np.clip(px, spec.MOVE_AREA_X, spec.MOVE_AREA_X + spec.MOVE_AREA_W))
        py = float(np.clip(py, spec.MOVE_AREA_Y, spec.MOVE_AREA_Y + spec.MOVE_AREA_H))
        return px, py

    def _prepare(self, obs: dict, bullets_raw: dict | None):
        """Collect the bullets that can matter within the planning horizon."""
        cfg = self.config
        player = self._player_position(obs)
        if bullets_raw is not None and len(bullets_raw.get("position", [])) > 0:
            pos, vel, half = self._bullets_from_raw(bullets_raw)
        else:
            pos, vel, half = self._bullets_from_obs(obs, player)

        if pos.shape[0] > cfg.max_bullets:
            order = np.argsort(
                (pos[:, 0] - player[0]) ** 2 + (pos[:, 1] - player[1]) ** 2
            )[: cfg.max_bullets]
            pos, vel, half = pos[order], vel[order], half[order]
        # Only bullets that can matter within the horizon.
        dx = pos[:, 0] - player[0]
        dy = pos[:, 1] - player[1]
        ahead = vel[:, 0] * cfg.horizon_steps * cfg.frameskip
        ahead_y = vel[:, 1] * cfg.horizon_steps * cfg.frameskip
        keep = (
            (np.abs(dx) < cfg.near_radius + np.abs(ahead))
            & (np.abs(dy) < cfg.near_radius + np.abs(ahead_y))
        )
        return player, pos[keep], vel[keep], half[keep]

    def _rollout_scores(
        self,
        origin: np.ndarray,
        pos: np.ndarray,
        vel: np.ndarray,
        half: np.ndarray,
        boss_x: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-first-action value and raw clearance, both shape ``(18,)``.

        Returns ``(value, clearance)`` where

        * ``value`` is ``min clearance over the horizon + the bounded preference
          terms`` - i.e. exactly the quantity the teacher ranks actions by, so
          that ties are broken the same way the teacher breaks them;
        * ``clearance`` is the raw safety number, used for the bomb threshold.

        This is the teacher's *value*, not its choice: distilling the value
        instead of the argmax is what makes tied options harmless - two equally
        safe moves get two equal targets instead of one arbitrary label each.
        """
        cfg = self.config
        n_bullets = pos.shape[0]
        x_lo = spec.MOVE_AREA_X + cfg.player_half
        x_hi = spec.MOVE_AREA_X + spec.MOVE_AREA_W - cfg.player_half
        y_lo = spec.MOVE_AREA_Y + cfg.player_half
        y_hi = spec.MOVE_AREA_Y + spec.MOVE_AREA_H - cfg.player_half
        radius = half + cfg.player_half + cfg.bullet_pad

        # Beam state: (x, y) absolute, path score, first action's (move, focus).
        bx = np.array([origin[0]], dtype=np.float32)
        by = np.array([origin[1]], dtype=np.float32)
        bscore = np.array([np.inf], dtype=np.float32)  # min clearance so far
        bmean = np.zeros(1, dtype=np.float32)
        bpref = np.zeros(1, dtype=np.float32)  # preference at the path's last step
        bmove = np.array([self._prev_move], dtype=np.int8)
        bfocus = np.array([0], dtype=np.int8)

        disp = self._step_disp.reshape(18, 2)  # (focus*9+move, xy)
        focus_idx = np.repeat(np.arange(2, dtype=np.int8), 9)
        move_idx = np.tile(np.arange(9, dtype=np.int8), 2)

        for step in range(1, cfg.horizon_steps + 1):
            t = float(step * cfg.frameskip)
            if n_bullets:
                # Bullet centres at time t, relative to the planning origin.
                bpx = pos[None, :, 0] + vel[None, :, 0] * t
                bpy = pos[None, :, 1] + vel[None, :, 1] * t
            nx = np.clip(bx[:, None] + disp[None, :, 0], x_lo, x_hi)
            ny = np.clip(by[:, None] + disp[None, :, 1], y_lo, y_hi)
            shape = (bx.size, 18)
            if n_bullets:
                dx = nx.reshape(-1, 1) - bpx
                dy = ny.reshape(-1, 1) - bpy
                dist = np.sqrt(dx * dx + dy * dy) - radius[None, :]
                clear = dist.min(axis=1).reshape(shape)
            else:
                clear = np.full(shape, 60.0, dtype=np.float32)

            repeated_score = np.repeat(bscore, 18)
            flat_clear = clear.reshape(-1)
            new_min = np.minimum(repeated_score, flat_clear)
            new_mean = np.repeat(bmean, 18) + flat_clear
            new_x = nx.reshape(-1)
            new_y = ny.reshape(-1)
            # The recorded action of a path is its *first* action, so it is only
            # set on the first expansion and propagated afterwards.
            first_step = repeated_score == np.inf
            new_move = np.where(first_step, np.tile(move_idx, bx.size), np.repeat(bmove, 18))
            new_focus = np.where(first_step, np.tile(focus_idx, bx.size), np.repeat(bfocus, 18))

            # Preference terms enter only the ranking, never the safety floor.
            pref = cfg.bottom_bias * (new_y - spec.MOVE_AREA_Y) / max(1.0, spec.MOVE_AREA_H)
            if boss_x is not None:
                pref = pref - cfg.aim_weight * np.abs(new_x - boss_x) / max(1.0, spec.MOVE_AREA_W)
            # The switching penalty applies to the *first* action only.  Applying
            # it to every expansion charged a whole path for its first move (a
            # constant), which made "idle" strictly dominate: measured effect was
            # a pilot that stood still with a boss on its right.
            switched = np.logical_and(first_step, new_move != self._prev_move)
            flat_rank = (
                new_min
                + pref
                - cfg.switch_penalty * switched.astype(np.float32)
                + 1e-4 * new_mean
            )
            flat_pref = pref.reshape(-1)

            # Keep a fixed quota *per first action* instead of a global top-N.
            # A global beam lets one action crowd out the rest, and then the
            # per-action score for the crowded-out actions would be missing.
            first_action = new_focus.astype(np.int32) * 9 + new_move.astype(np.int32)
            keep: list[int] = []
            for action in range(18):
                indices = np.nonzero(first_action == action)[0]
                if indices.size == 0:
                    continue
                order = indices[np.argsort(-flat_rank[indices], kind="stable")][
                    : cfg.quota_per_action
                ]
                keep.extend(int(i) for i in order)

            idx = np.asarray(keep, dtype=np.int64)
            bx, by = new_x[idx], new_y[idx]
            bscore = new_min[idx]
            bmean = new_mean[idx]
            bpref = flat_pref[idx]
            bmove, bfocus = new_move[idx], new_focus[idx]
            if bscore.max() > 40.0 and step * cfg.frameskip >= 8:
                # Nothing relevant nearby for a while: stop early and save CPU.
                break

        action_of_beam = bfocus.astype(np.int32) * 9 + bmove.astype(np.int32)
        value = np.full(18, -1e9, dtype=np.float32)
        clearance = np.full(18, -1e9, dtype=np.float32)
        np.maximum.at(value, action_of_beam, bscore + bpref)
        np.maximum.at(clearance, action_of_beam, bscore)
        return value, clearance

    # -- public API -------------------------------------------------------
    def candidate_scores(
        self,
        obs: dict,
        bullets_raw: dict | None = None,
        bosses: np.ndarray | None = None,
    ) -> np.ndarray:
        """Teacher decision value for all 18 candidate first actions (18,) float32."""
        player, pos, vel, half = self._prepare(obs, bullets_raw)
        value, _clearance = self._rollout_scores(player, pos, vel, half, boss_x=_boss_x(bosses))
        return value

    def act(
        self,
        obs: dict,
        bullets_raw: dict | None = None,
        bombs: int = 0,
        bosses: np.ndarray | None = None,
    ) -> np.ndarray:
        """Choose one action ``[move, shoot, focus, bomb]`` for the current frame."""
        cfg = self.config
        player, pos, vel, half = self._prepare(obs, bullets_raw)
        value, clearance_by_action = self._rollout_scores(
            player, pos, vel, half, boss_x=_boss_x(bosses)
        )
        best = int(np.argmax(value))
        move, focus = best % 9, best // 9
        clearance = float(clearance_by_action[best])

        bomb = 0
        if self._bomb_cooldown > 0:
            self._bomb_cooldown -= 1
        elif clearance < cfg.bomb_min_clearance and bombs > 0:
            bomb = 1
            self._bomb_cooldown = cfg.bomb_cooldown
        if bomb:
            focus = 1

        self._prev_move = move
        self.debug = PilotDebug(
            best_clearance=clearance, bullets_used=int(pos.shape[0]), bombs=bombs
        )
        return np.array([move, int(cfg.always_shoot), int(focus), int(bomb)], dtype=np.int64)


def _boss_x(bosses: np.ndarray | None) -> float | None:
    """Horizontal aim target: the x of the boss nearest the player's column.

    Only the x coordinate matters for damage - every shot travels straight up -
    so the pilot steers horizontally while the vertical term keeps it low.
    """
    if bosses is None:
        return None
    array = np.asarray(bosses, dtype=np.float32).reshape(-1, 2)
    if array.shape[0] == 0:
        return None
    return float(array[0, 0])


def observation_agreement(pilot: AvoidPilot, observation: dict, expert_move: int) -> bool:
    """True when the pilot picks the same movement class as the expert.

    Used as a zero-game-time sanity check: if the pilot never agrees with the
    humans on the recorded Lunatic dataset, it is a bad teacher no matter what
    its survival numbers are.
    """
    action = pilot.act(observation)
    return int(action[0]) == int(expert_move)
