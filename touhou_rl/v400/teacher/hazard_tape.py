"""``HazardTape``: one forward recording of the live engine, frame by frame.

11-P1 搂2 step 1 asks for a *branch* of the current state that walks ``H`` frames
and stores every lethal's geometry.  The harness cannot do that (EXP-P1-003 搂2):
``BRANCH`` replies with ``hash``/``trace`` only and ``CHILD <handle> HAZARDS`` is
refused with ``ERR child-command``.  So this module does the only thing the
engine actually supports 鈥?it walks the authoritative shadow forward and records
each frame 鈥?and :mod:`touhou_rl.v400.teacher.search` gets a real tape either
way.  The caller restores the shadow afterwards (the CLI does it with
``RESET`` + replay), and the recording is what makes that restore expensive.

All geometry is kept exactly as the engine reported it (float bits decoded to
float, no re-scaling).  ``collisionSize`` therefore stays a **full** width here;
the halving happens once, in :mod:`touhou_rl.v400.teacher.geometry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from touhou_rl.v400.gym.actions import cand_to_mask
from touhou_rl.v400.gym.protocol import BulletRow, EnemyRow, HazardFrame, LaserRow, PlayerSnapshot
from touhou_rl.v400.teacher.geometry import SpeedTable, bullet_boxes, enemy_boxes

#: "涓嶅姩(鑱氱劍)": no direction, focus held.  ``cand = focus * 9 + move = 9``.
IDLE_FOCUSED = 9


class HazardExportUnavailable(RuntimeError):
    """The engine refused ``HAZARDS``; the teacher must fail closed, not guess."""


class ChildHazardsUnavailable(RuntimeError):
    """``record(source="child")`` was asked for, but the engine cannot serve it.

    A child recording is the only way to stop moving the authoritative shadow
    (the fork that records is exactly what 11-P1 §2 step 1 asks for).  Until the
    engine can answer ``CHILD <h> HAZARDS`` the teacher must **fail closed**:
    silently re-recording on the main process would advance the shadow 48 frames
    and turn a cheap decision into a full ``RESET`` + replay, i.e. it would hide
    a 1.6 s cost behind a "successful" decision.
    """


@dataclass
class HazardTape:
    """A recorded horizon: player rows, lethal rows, hashes, and the masks used."""

    horizon: int
    decision_frames: int
    frames: np.ndarray
    hashes: tuple[str, ...]
    player: dict[str, np.ndarray]
    speeds: SpeedTable
    bullets: tuple[tuple[BulletRow, ...], ...]
    lasers: tuple[tuple[LaserRow, ...], ...]
    enemies: tuple[tuple[EnemyRow, ...], ...]
    masks: tuple[int, ...]
    actions: tuple[int, ...]
    meta: dict[str, object] = field(default_factory=dict)
    _boxes: list[tuple[np.ndarray, np.ndarray]] | None = field(default=None, repr=False)

    # -- shape ------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.frames.shape[0])

    @property
    def start_position(self) -> tuple[float, float]:
        return (float(self.player["px"][0]), float(self.player["py"][0]))

    @property
    def hurtbox_half(self) -> tuple[float, float]:
        half_x = (self.player["max_x"][0] - self.player["min_x"][0]) / 2.0
        half_y = (self.player["max_y"][0] - self.player["min_y"][0]) / 2.0
        return (float(half_x), float(half_y))

    @property
    def hurtbox_center(self) -> np.ndarray:
        return np.array(
            [
                (self.player["min_x"] + self.player["max_x"]) / 2.0,
                (self.player["min_y"] + self.player["max_y"]) / 2.0,
            ],
            dtype=np.float64,
        )

    def lethal_counts(self) -> dict[str, int]:
        """Row counts per family, plus how many laser rows carry a live hitbox.

        EXP-P1-021: ``lasers_damageable`` is the engine's own count (rows with
        ``damageable == 1``), so a recording can be audited for "how many beams
        were actually judging this horizon" without re-deriving the windows.
        A pre-EXP-P1-021 row has no flag and is not counted here; the total
        ``lasers`` still counts it, so the two numbers answer different questions.
        """

        return {
            "bullets": int(sum(1 for rows in self.bullets for row in rows)),
            "lasers": int(sum(1 for rows in self.lasers for row in rows)),
            "lasers_damageable": int(
                sum(1 for rows in self.lasers for row in rows
                    if getattr(row, "damageable", None) == 1)
            ),
            "enemies": int(sum(1 for rows in self.enemies for row in rows)),
        }

    # -- geometry ---------------------------------------------------------
    def boxes(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Lethal bullet + enemy-body AABBs of frame ``index`` as ``(mins, maxs)``.

        Lasers are excluded because their hit region is a rotated rectangle, not
        an AABB; :mod:`touhou_rl.v400.teacher.verify` and
        :func:`touhou_rl.v400.teacher.geometry.frame_verdict` handle them
        separately.  Results are cached: the beam search asks for the same frame
        thousands of times.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"frame {index} outside tape of {len(self)}")
        if self._boxes is None:
            self._boxes = [None] * len(self)  # type: ignore[list-item]
        cached = self._boxes[index]
        if cached is not None:
            return cached
        bullet_boxes_ = bullet_boxes(self.bullets[index])
        enemy_boxes_ = enemy_boxes(self.enemies[index])
        mins = np.concatenate(
            [bullet_boxes_.mins, enemy_boxes_.mins], axis=0
        ).astype(np.float64)
        maxs = np.concatenate(
            [bullet_boxes_.maxs, enemy_boxes_.maxs], axis=0
        ).astype(np.float64)
        if mins.shape[0] == 0:
            mins = np.zeros((0, 2), dtype=np.float64)
            maxs = np.zeros((0, 2), dtype=np.float64)
        self._boxes[index] = (mins, maxs)
        return mins, maxs

    def box_array(self, index: int) -> np.ndarray:
        mins, maxs = self.boxes(index)
        if mins.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float64)
        return np.concatenate([mins, maxs], axis=1)

    def frame_hazards(self, index: int) -> HazardFrame:
        """Re-assemble the frame in the protocol's own type, for ``frame_verdict``."""
        return HazardFrame(
            frame=int(self.frames[index]),
            bullet_count=len(self.bullets[index]),
            laser_count=len(self.lasers[index]),
            enemy_count=len(self.enemies[index]),
            bullets=self.bullets[index],
            lasers=self.lasers[index],
            enemies=self.enemies[index],
        )

    def laser_gaps(self, index: int, center: np.ndarray, half: np.ndarray) -> list[float]:
        """Gaps to every **damageable** laser hit region of frame ``index``.

        Reuses the exact per-frame implementation, once per laser, because the
        laser frame rotation is cheap for a handful of slots.  EXP-P1-021: the
        gate is the engine's ``damageable`` flag (``geometry.laser_hits``), so a
        beam that is still fading in contributes no gap and a state-0 beam that
        already reached ``hitboxStartTime`` does — EXP-P1-016 measured 51 live
        rows on the Stage 6A death frame, so "there are no live lasers" is not
        an assumption this path may make.
        """
        from touhou_rl.v400.teacher.geometry import laser_hits

        if not self.lasers[index]:
            return []
        gaps: list[float] = []
        for laser in self.lasers[index]:
            hurt_min = (float(center[0] - half[0]), float(center[1] - half[1]))
            hurt_max = (float(center[0] + half[0]), float(center[1] + half[1]))
            verdict = laser_hits([laser], hurt_min, hurt_max)
            if verdict.considered == 0:
                continue
            if verdict.nearest is not None:
                gaps.append(float(verdict.nearest[2]))
        return gaps

    def summary(self) -> dict[str, object]:
        return {
            "horizon": self.horizon,
            "decision_frames": self.decision_frames,
            "frames": [int(self.frames[0]), int(self.frames[-1])] if len(self) else [],
            "speeds": list(self.speeds.as_tuple()),
            "lethal_counts": self.lethal_counts(),
            "start_position": self.start_position,
            "hurtbox_half": self.hurtbox_half,
            "meta": dict(self.meta),
        }


def expand_actions(
    planned_actions, horizon: int, decision_frames: int = 2
) -> tuple[int, ...]:
    """Decision actions -> one candidate id per frame, padded with idle+focus.

    ``planned_actions`` items may be ``int`` (a candidate) or ``(cand, bomb)``.
    The bomb bit is ignored for the tape's *geometry* because a bomb does not
    move the ship; it is tracked by the caller in the label.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    per_frame: list[int] = []
    for item in planned_actions:
        cand = item[0] if isinstance(item, (tuple, list)) else int(item)
        if not 0 <= int(cand) < 18:
            raise ValueError(f"candidate out of range: {cand!r}")
        per_frame.extend([int(cand)] * decision_frames)
        if len(per_frame) >= horizon:
            break
    while len(per_frame) < horizon:
        per_frame.extend([IDLE_FOCUSED] * decision_frames)
    return tuple(per_frame[:horizon])


def record(
    process,
    horizon: int = 48,
    planned_actions=(),
    *,
    decision_frames: int = 2,
    fail_closed: bool = True,
    source: str = "main",
    child_callable=None,
) -> HazardTape:
    """Walk ``horizon`` frames and record every lethal.

    ``source="main"`` (the default) keeps the measured behaviour of EXP-P1-003:
    one ``STEP`` + one ``HAZARDS`` + one ``PLAYER`` per frame on the
    authoritative shadow, which therefore moves ``horizon`` frames forward.

    ``source="child"`` is the interface for recording **without** moving the
    shadow: ``child_callable(frame)`` returns the frame's hazard rows (a
    ``HazardFrame``-shaped object or a dict with ``bullets``/``lasers``/
    ``enemies``, optionally ``player``/``frame``/``hash``) and is free to drive a
    fork.  It is fail-closed by contract:

    * a callable is mandatory — no silent fall back to the main process;
    * if the process exposes :meth:`probe_child_hazards` and the probe says the
      build refuses ``CHILD HAZARDS``, :class:`ChildHazardsUnavailable` is raised
      **before the first frame**;
    * a callable that raises is reported as :class:`ChildHazardsUnavailable`, not
      as a tape with missing geometry.

    Row 0 must describe the decision point, so it is taken from the callable's
    frame 0.  When the callable carries a ``player`` per frame that snapshot is
    used for that row; otherwise every row reuses the row-0 ``PLAYER`` (the beam
    search only reads row 0's position, hurtbox and bombs, and a fabricated
    per-frame position would be worse than an honest constant).
    """
    if source not in ("main", "child"):
        raise ValueError(f"source must be 'main' or 'child', got {source!r}")
    if source == "child":
        return _record_child(
            process,
            horizon,
            planned_actions,
            decision_frames=decision_frames,
            child_callable=child_callable,
        )
    slots = expand_actions(planned_actions, horizon, decision_frames)
    frames: list[int] = []
    hashes: list[str] = []
    bullets: list[tuple[BulletRow, ...]] = []
    lasers: list[tuple[LaserRow, ...]] = []
    enemies: list[tuple[EnemyRow, ...]] = []
    player_rows: dict[str, list[float]] = {
        key: []
        for key in (
            "px",
            "py",
            "min_x",
            "min_y",
            "max_x",
            "max_y",
            "state",
            "invuln",
            "focus",
            "bombs",
            "lives",
            "power",
        )
    }
    speeds: SpeedTable | None = None
    # Row 0 must describe the decision point itself, so the sample is taken
    # *before* the first step.  Reading ``HAZARDS``/``PLAYER`` after ``STEP``
    # labels frame d+1 as frame d, which shifts every plan by one frame — the
    # verifier's position check caught exactly that (EXP-P1-003 §5).
    current_hash = str(process.hash())
    for index, cand in enumerate(slots):
        mask = cand_to_mask(cand, 0)
        try:
            hazard = process.hazards()
        except Exception as error:  # noqa: BLE001 - re-raised with context below
            if fail_closed:
                raise HazardExportUnavailable(
                    f"HAZARDS refused at row {index}: {error}"
                ) from error
            raise
        player = process.player()
        if speeds is None:
            speeds = SpeedTable(
                normal_axial=float(player.speed_normal_axial),
                normal_diagonal=float(player.speed_normal_diagonal),
                focused_axial=float(player.speed_focused_axial),
                focused_diagonal=float(player.speed_focused_diagonal),
            )
            speeds.check()
        if player.hurtbox_min is None or player.hurtbox_max is None:
            raise HazardExportUnavailable("PLAYER reply carries no hurtbox bounds")
        frames.append(int(player.frame))
        hashes.append(current_hash)
        bullets.append(hazard.bullets)
        lasers.append(hazard.lasers)
        enemies.append(hazard.enemies)
        _append_player(player_rows, player)
        if index + 1 < len(slots):
            step = process.step(mask, 1)
            current_hash = str(step.get("hash", ""))
    return HazardTape(
        horizon=horizon,
        decision_frames=decision_frames,
        frames=np.asarray(frames, dtype=np.int64),
        hashes=tuple(hashes),
        player={key: np.asarray(values, dtype=np.float64) for key, values in player_rows.items()},
        speeds=speeds or SpeedTable(0.0, 0.0, 0.0, 0.0),
        bullets=tuple(bullets),
        lasers=tuple(lasers),
        enemies=tuple(enemies),
        masks=tuple(cand_to_mask(cand, 0) for cand in slots),
        actions=slots,
        meta={
            "recorded_by": "forward walk of the shadow process (BRANCH cannot carry hazards)",
            "nominal_fill": "idle+focused",
            "decision_frames": decision_frames,
        },
    )


def _child_hazard(payload, index: int):
    """Coerce one ``child_callable`` result into ``(bullets, lasers, enemies)``."""
    if isinstance(payload, dict):
        missing = [key for key in ("bullets", "lasers", "enemies") if key not in payload]
        if missing:
            raise ChildHazardsUnavailable(
                f"child_callable({index}) returned a dict without {missing}; "
                "a partial frame is not a frame"
            )
        return (
            tuple(payload["bullets"]),
            tuple(payload["lasers"]),
            tuple(payload["enemies"]),
        )
    try:
        return (tuple(payload.bullets), tuple(payload.lasers), tuple(payload.enemies))
    except AttributeError as error:
        raise ChildHazardsUnavailable(
            f"child_callable({index}) is neither a HazardFrame nor a hazard dict: {error}"
        ) from error


def _record_child(
    process,
    horizon: int,
    planned_actions,
    *,
    decision_frames: int,
    child_callable,
) -> HazardTape:
    """The ``source="child"`` half of :func:`record`; see that docstring."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if child_callable is None:
        raise ChildHazardsUnavailable(
            "record(source='child') needs an injected child_callable(frame)->hazard_dict; "
            "the shipped engine build refuses CHILD <h> HAZARDS, and falling back to a "
            "main-process recording would move the shadow"
        )
    probe = getattr(process, "probe_child_hazards", None)
    if callable(probe):
        verdict = probe()
        if not verdict.get("available"):
            raise ChildHazardsUnavailable(
                "engine reports child hazards unavailable: "
                f"{verdict.get('reason')!r}"
            )
    slots = expand_actions(planned_actions, horizon, decision_frames)
    frames: list[int] = []
    hashes: list[str] = []
    bullets: list[tuple[BulletRow, ...]] = []
    lasers: list[tuple[LaserRow, ...]] = []
    enemies: list[tuple[EnemyRow, ...]] = []
    player_rows: dict[str, list[float]] = {
        key: []
        for key in (
            "px",
            "py",
            "min_x",
            "min_y",
            "max_x",
            "max_y",
            "state",
            "invuln",
            "focus",
            "bombs",
            "lives",
            "power",
        )
    }
    speeds: SpeedTable | None = None
    fallback_player = None
    for index in range(horizon):
        try:
            payload = child_callable(index)
        except Exception as error:  # noqa: BLE001 - fail closed with context
            raise ChildHazardsUnavailable(
                f"child_callable({index}) failed: {error}"
            ) from error
        if payload is None:
            raise ChildHazardsUnavailable(f"child_callable({index}) returned nothing")
        frame_bullets, frame_lasers, frame_enemies = _child_hazard(payload, index)
        row_player = (
            payload.get("player") if isinstance(payload, dict) else getattr(payload, "player", None)
        )
        if index == 0:
            fallback_player = row_player or process.player()
            if fallback_player.hurtbox_min is None or fallback_player.hurtbox_max is None:
                raise HazardExportUnavailable("PLAYER reply carries no hurtbox bounds")
            speeds = SpeedTable(
                normal_axial=float(fallback_player.speed_normal_axial),
                normal_diagonal=float(fallback_player.speed_normal_diagonal),
                focused_axial=float(fallback_player.speed_focused_axial),
                focused_diagonal=float(fallback_player.speed_focused_diagonal),
            )
            speeds.check()
        player = row_player or fallback_player
        if player is None or player.hurtbox_min is None or player.hurtbox_max is None:
            raise HazardExportUnavailable(f"row {index} PLAYER carries no hurtbox bounds")
        frames.append(int(payload.get("frame", index) if isinstance(payload, dict) else index))
        hashes.append(str(payload.get("hash", "") if isinstance(payload, dict) else ""))
        bullets.append(frame_bullets)
        lasers.append(frame_lasers)
        enemies.append(frame_enemies)
        _append_player(player_rows, player)
    return HazardTape(
        horizon=horizon,
        decision_frames=decision_frames,
        frames=np.asarray(frames, dtype=np.int64),
        hashes=tuple(hashes),
        player={key: np.asarray(values, dtype=np.float64) for key, values in player_rows.items()},
        speeds=speeds or SpeedTable(0.0, 0.0, 0.0, 0.0),
        bullets=tuple(bullets),
        lasers=tuple(lasers),
        enemies=tuple(enemies),
        masks=tuple(cand_to_mask(cand, 0) for cand in slots),
        actions=slots,
        meta={
            "recorded_by": "injected child callable (CHILD HAZARDS path)",
            "source": "child",
            "nominal_fill": "idle+focused",
            "decision_frames": decision_frames,
        },
    )


def _append_player(target: dict[str, list[float]], player: PlayerSnapshot) -> None:
    target["px"].append(float(player.position[0]))
    target["py"].append(float(player.position[1]))
    min_x, min_y = player.hurtbox_min
    max_x, max_y = player.hurtbox_max
    target["min_x"].append(float(min_x))
    target["min_y"].append(float(min_y))
    target["max_x"].append(float(max_x))
    target["max_y"].append(float(max_y))
    target["state"].append(float(player.player_state))
    target["invuln"].append(float(player.invuln_state))
    target["focus"].append(float(player.focus))
    target["bombs"].append(float(player.bombs))
    target["lives"].append(float(player.lives))
    target["power"].append(float(player.power))
