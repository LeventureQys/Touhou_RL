"""A deterministic TH08 stand-in for the v3.7.1 clock/gate tests.

No game process is involved: ``gameplay_frame`` is a pure function of a shared
fake wall clock.  A test can therefore make planning "take 10 frames" by
advancing that clock, or make the counter stall/roll back/jump, and the harness
must classify the outcome exactly as it would on the real game.

The fake is deliberately a *port*, not a mock: it implements the same three calls
(``read_state``/``read_bullets``/``set_keys``) the live session exposes, plus a
player integrator so that "same input -> same trajectory" is a real assertion
rather than a tautology.

It lives in the package (not in ``tests/``) because ``escape experiment --fake``
runs the *same* harness end to end without a game process: that is how C-01 stays
executable on a machine with no TH08 running.

Timing model
------------
Every observable action costs fake time (``read_cost_s``), and ``sleep`` advances
fake time by at least 0.1 ms so the harness's polling loops always make progress.
At 60 fps a 2 ms poll therefore advances ~0.12 frames, i.e. about 8 polls per
frame, which is also what the real loop does at ``poll_s=0.002``.
"""

from __future__ import annotations

import numpy as np

from .. import spec
from ..state import BulletField, GameState

#: Per-frame displacement of the calibrated movement model (px/frame).
_AXIS = 4.0
_DIAGONAL = 4.0 / np.sqrt(2.0)
_FOCUS_AXIS = 2.0
_FOCUS_DIAGONAL = 2.0 / np.sqrt(2.0)


class FakeTime:
    """A monotonic clock the test controls, shared by the harness and the game."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = float(start)

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(float(seconds), 1e-4)

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)


def _displacement(buttons: int = 0, *, focus: bool = False) -> tuple[float, float]:
    """Per-frame displacement for a held button set (screen coords, +y down)."""

    ux = (-1 if buttons & spec.BTN_LEFT else 0) + (1 if buttons & spec.BTN_RIGHT else 0)
    uy = (-1 if buttons & spec.BTN_UP else 0) + (1 if buttons & spec.BTN_DOWN else 0)
    if ux == 0 and uy == 0:
        return (0.0, 0.0)
    if ux != 0 and uy != 0:
        component = _FOCUS_DIAGONAL if focus else _DIAGONAL
    else:
        component = _FOCUS_AXIS if focus else _AXIS
    return (ux * component, uy * component)


class FakeGame:
    """A frame-counter-accurate fake game exposing the ``harness.GamePort`` API."""

    def __init__(
        self,
        time: FakeTime,
        *,
        fps: float = 60.0,
        start_frame: int = 100,
        deaths_at: tuple[int, ...] = (),
        bombs_at: tuple[int, ...] = (),
        input_latency_frames: int = 1,
        read_cost_s: float = 0.00012,
        bullet_set: tuple[tuple[float, float, float, float, float], ...] = (),
        supervisor_state: int = 2,
        stall_from: int | None = None,
        rollback_at_read: int | None = None,
        read_fail_at_read: tuple[int, ...] = (),
        jump_at_frame: tuple[int, int] | None = None,
        player_start: tuple[float, float] = (192.0, 400.0),
        field_origin: tuple[float, float] = (spec.PLAYFIELD_X, spec.PLAYFIELD_Y),
        field_size: tuple[float, float] = (spec.PLAYFIELD_W, spec.PLAYFIELD_H),
        enemy_row: bool = False,
    ) -> None:
        self.time = time
        self.fps = float(fps)
        self.t0 = time.t
        self.base_frame = int(start_frame)
        self.frame = int(start_frame)
        self.deaths_at = tuple(int(v) for v in deaths_at)
        self.bombs_at = tuple(int(v) for v in bombs_at)
        self.input_latency_frames = int(input_latency_frames)
        self.read_cost_s = float(read_cost_s)
        self.supervisor_state = int(supervisor_state)
        self.stall_from = None if stall_from is None else int(stall_from)
        self.rollback_at_read = None if rollback_at_read is None else int(rollback_at_read)
        self.read_fail_at_read = set(int(v) for v in read_fail_at_read)
        self.jump_at_frame = None if jump_at_frame is None else (int(jump_at_frame[0]), int(jump_at_frame[1]))
        self.field_origin = (float(field_origin[0]), float(field_origin[1]))
        self.field_size = (float(field_size[0]), float(field_size[1]))
        self.player_start = (float(player_start[0]), float(player_start[1]))
        self.x, self.y = self.player_start
        self.deaths = 0
        self.bombs_used = 0
        self.bombs = 2.0
        self.lives = 2.0
        self.score = 0
        self.graze = 0
        self.player_state = 0
        self.reads = 0
        self.keys: set[str] = set()
        self.input_schedule: list[tuple[int, int]] = []
        self.key_log: list[dict] = []
        self.trace: list[dict] = []
        self._applied_mask = 0
        #: Active bullets: (x, y, vx, vy, half) at ``start_frame``.
        self.bullet_set = tuple(tuple(float(v) for v in bullet) for bullet in bullet_set)
        self.enemy_row = bool(enemy_row)

    # -- clock ------------------------------------------------------------
    def _target_frame(self) -> int:
        elapsed = self.time.t - self.t0
        target = self.base_frame + int(elapsed * self.fps)
        if self.stall_from is not None:
            target = min(target, self.stall_from)
        return max(target, self.frame)

    def _tick(self, frame: int) -> None:
        mask = self._mask_at(frame)
        self._applied_mask = mask
        focus = bool(mask & spec.BTN_FOCUS)
        dx, dy = _displacement(mask, focus=focus)
        self.x = float(np.clip(self.x + dx, self.field_origin[0], self.field_origin[0] + self.field_size[0]))
        self.y = float(np.clip(self.y + dy, self.field_origin[1], self.field_origin[1] + self.field_size[1]))
        if frame in self.deaths_at:
            self.deaths += 1
            self.lives -= 1.0
            self.x, self.y = self.player_start
        if frame in self.bombs_at:
            self.bombs_used += 1
            self.bombs = max(0.0, self.bombs - 1.0)
        self.trace.append(
            {
                "frame": int(frame),
                "x": round(self.x, 4),
                "y": round(self.y, 4),
                "deaths": int(self.deaths),
                "input_mask": int(mask),
            }
        )

    def _advance_to_now(self) -> None:
        target = self._target_frame()
        while self.frame < target:
            self.frame += 1
            self._tick(self.frame)
            if self.jump_at_frame is not None and self.frame >= self.jump_at_frame[0]:
                # A descheduled process: wall time kept running, the counter
                # skipped.  Applied here so the next read observes the gap.
                skipped = self.jump_at_frame[1]
                self.base_frame += skipped
                self.frame += skipped
                self.jump_at_frame = None

    def _mask_at(self, frame: int) -> int:
        """Buttons the engine consumes on ``frame`` (last schedule entry ≤ frame)."""

        mask = 0
        for effective_frame, buttons in self.input_schedule:
            if effective_frame <= frame:
                mask = buttons
            else:
                break
        return mask

    # -- port -------------------------------------------------------------
    def read_state(self, with_bullets: bool = False, with_title: bool = False) -> GameState:
        self.reads += 1
        if self.reads in self.read_fail_at_read:
            raise OSError("fake read failure")
        if self.rollback_at_read is not None and self.reads == self.rollback_at_read:
            # A restart: the counter restarts at the stage entry.
            self.base_frame = max(1, self.base_frame - 40)
            self.t0 = self.time.t
            self.frame = self.base_frame
            self.trace.clear()
        self._advance_to_now()
        self.time.advance(self.read_cost_s)
        state = GameState()
        state.supervisor_state = int(self.supervisor_state)
        state.stage = 0
        state.gameplay_frame = int(self.frame)
        state.run_frames = int(self.frame)
        state.stage_frames = int(self.frame)
        state.deaths = int(self.deaths)
        state.bombs_used = int(self.bombs_used)
        state.lives = float(self.lives)
        state.bombs = float(self.bombs)
        state.score = int(self.score)
        state.graze = int(self.graze)
        state.player_state = int(self.player_state)
        state.player_x = float(self.x)
        state.player_y = float(self.y)
        state.player_vx = 0.0
        state.player_vy = 0.0
        state.focus_mode = int(bool(self._applied_mask & spec.BTN_FOCUS))
        state.hurtbox_half_x = 0.825
        state.hurtbox_half_y = 0.825
        state.field_x, state.field_y = self.field_origin
        state.field_w, state.field_h = self.field_size
        state.input_mask = int(self._applied_mask)
        state.last_input_mask = int(self._mask_at(self.frame - 1))
        state.gui_input_current = int(self._applied_mask)
        state.active_bullets = len(self._active_bullets())
        return state

    def read_bullets(self) -> BulletField:
        self.time.advance(self.read_cost_s)
        positions, velocities = self._active_bullets()
        count = len(positions)
        half = np.full((count, 2), 2.5, dtype=np.float32)
        raw = np.zeros((max(1, count), 16), dtype=np.uint8)
        return BulletField(
            raw=raw,
            position=np.asarray(positions, dtype=np.float32).reshape(count, 2),
            velocity=np.asarray(velocities, dtype=np.float32).reshape(count, 2),
            state=np.full((count,), 1, dtype=np.uint16),
            disabled=np.zeros((count,), dtype=np.int8),
            half_size=half,
        )

    def _active_bullets(self):
        if not self.bullet_set:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
        elapsed = self.frame - self.base_frame
        positions = []
        velocities = []
        for x, y, vx, vy, _half in self.bullet_set:
            positions.append((x + vx * elapsed, y + vy * elapsed))
            velocities.append((vx, vy))
        return np.asarray(positions, dtype=np.float32), np.asarray(velocities, dtype=np.float32)

    def set_keys(self, keys: set[str]) -> None:
        self.reads += 0
        new = set(keys)
        if new == self.keys:
            return
        self.keys = new
        buttons = 0
        for name, bit in (
            ("up", spec.BTN_UP),
            ("down", spec.BTN_DOWN),
            ("left", spec.BTN_LEFT),
            ("right", spec.BTN_RIGHT),
            ("z", spec.BTN_SHOOT),
            ("shift", spec.BTN_FOCUS),
            ("x", spec.BTN_BOMB),
        ):
            if name in new:
                buttons |= bit
        self.time.advance(self.read_cost_s)
        self._advance_to_now()
        effective = int(self.frame) + int(self.input_latency_frames)
        self.input_schedule.append((effective, buttons))
        self.input_schedule.sort(key=lambda item: item[0])
        self.key_log.append({"frame": int(self.frame), "keys": sorted(new), "buttons": int(buttons)})

    def release_all(self) -> None:
        self.set_keys(set())

    # -- inspection -------------------------------------------------------
    def trace_document(self) -> dict:
        return {
            "schema": "fake-trace/1",
            "start_frame": int(self.base_frame),
            "frames": list(self.trace),
            "keys": list(self.key_log),
        }


def fingerprint(trace: list[dict]) -> list[tuple[int, float, float, int]]:
    """The comparable part of a trajectory: (frame, x, y, deaths)."""

    return [(int(row["frame"]), float(row["x"]), float(row["y"]), int(row["deaths"])) for row in trace]
