"""Experiment clock (v3.7.1 P0): the only time base an experiment may terminate on.

Why this module exists
----------------------
The v3.7.0 confirmation run terminated after ``frames // 2 = 900`` decisions, not
after ``gameplay_frame >= 1800``.  Two consequences made the comparison
uninterpretable:

* the subject spent 14 ms per decision planning while the game kept running, so
  its actions were held for an *irregular* number of logical frames and the run
  finished at frames 1845/1854 instead of 1802;
* the baseline recorded no frame span at all, so "the same 1800 frames" was an
  assumption, never a measurement.

This module is the fix, and it is deliberately small: one reader of the real
``gameplay_frame`` counter, one window definition, one event ledger.  Nothing
here terminates on wall-clock time.  Wall-clock is used only to *abort* a wait
that has stopped making progress, and such an abort is recorded as an anomaly,
never silently converted into a successful termination.

Testability
-----------
Every side effect is injected (``read_state``, ``monotonic``, ``sleep``), so the
whole file runs against a fake clock (``tests/test_experiment_clock.py``).  That
is C-01: a fake clock that advances 0/1/3/10 frames during planning, rolls the
counter back, fails a read, or jumps must produce the documented classification
instead of a negative exposure or an unbounded run.

Nothing here imports the game.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

CLOCK_SCHEMA = "v371-experiment-clock/1"

# -- anomaly kinds ----------------------------------------------------------
#: ``gameplay_frame`` moved backwards: a restart, a reload, or a straddled read
#: across two sessions.  This is *not* clamped away; the caller must treat the
#: episode as a new one (C-01) and may never emit a negative exposure.
ANOMALY_ROLLBACK = "rollback"
#: ``read_state`` raised (process gone, memory unmapped).
ANOMALY_READ_FAIL = "read_fail"
#: The frame counter did not move for longer than the stall budget.
ANOMALY_STALL = "stall"
#: The frame counter moved further than ``jump_threshold`` between two reads.
#: Legitimate under scheduler pressure, but it means per-frame observations have
#: a gap, so it must be visible in the record.
ANOMALY_JUMP = "jump"
#: The process left ``STATE_GAME_MANAGER`` (game over screen, menu, crash).
ANOMALY_LEFT_GAME = "left_game"
#: A ``wait_until`` deadline elapsed before the target frame was reached.
ANOMALY_TIMEOUT = "timeout"

#: ``Player::playerState`` values (Player.hpp).
PLAYER_ALIVE = 0
PLAYER_SPAWNING = 1
PLAYER_DYING = 2
PLAYER_INVULNERABLE = 3

#: Value of ``StateGameManager`` for the in-game supervisor state.
GAME_SUPERVISOR_STATE = 2


class ClockError(RuntimeError):
    """The clock cannot produce a trustworthy frame number."""


@dataclass(frozen=True)
class FrameReading:
    """One sample of the game clock plus the fields an auditor needs with it."""

    frame: int
    wall_s: float
    supervisor_state: int = -1
    stage: int = -1
    deaths: int = 0
    bombs_used: int = 0
    lives: float = 0.0
    bombs: float = 0.0
    player_state: int = -1
    x: float = 0.0
    y: float = 0.0
    input_mask: int = 0
    last_input_mask: int = 0
    read_ok: bool = True

    @property
    def alive(self) -> bool:
        """Player state 0 and 3 are both "not dead"; 2 is dying."""
        return self.player_state in (PLAYER_ALIVE, PLAYER_INVULNERABLE)

    @property
    def in_game(self) -> bool:
        return self.supervisor_state == GAME_SUPERVISOR_STATE

    def as_dict(self) -> dict:
        return {
            "frame": int(self.frame),
            "wall_s": round(float(self.wall_s), 6),
            "supervisor_state": int(self.supervisor_state),
            "stage": int(self.stage),
            "deaths": int(self.deaths),
            "bombs_used": int(self.bombs_used),
            "lives": round(float(self.lives), 3),
            "bombs": round(float(self.bombs), 3),
            "player_state": int(self.player_state),
            "x": round(float(self.x), 4),
            "y": round(float(self.y), 4),
            "input_mask": int(self.input_mask),
            "last_input_mask": int(self.last_input_mask),
            "alive": bool(self.alive),
        }


def reading_from_state(state: Any, wall_s: float, *, read_ok: bool = True) -> FrameReading:
    """Adapt a ``state.GameState``-shaped object (or a test double) to a reading.

    The harness never inspects the game state directly: it goes through this
    function so that a fake state object only has to provide the same attributes.
    """

    def number(name: str, default: float = 0.0) -> float:
        return float(getattr(state, name, default) or default)

    return FrameReading(
        frame=int(getattr(state, "gameplay_frame", -1)),
        wall_s=float(wall_s),
        supervisor_state=int(getattr(state, "supervisor_state", -1)),
        stage=int(getattr(state, "stage", -1)),
        deaths=int(number("deaths")),
        bombs_used=int(number("bombs_used")),
        lives=number("lives"),
        bombs=number("bombs"),
        player_state=int(getattr(state, "player_state", -1)),
        x=number("player_x"),
        y=number("player_y"),
        input_mask=int(getattr(state, "input_mask", 0) or 0),
        last_input_mask=int(getattr(state, "last_input_mask", 0) or 0),
        read_ok=bool(read_ok),
    )


@dataclass(frozen=True)
class ClockConfig:
    """Bounds under which the clock reports an anomaly instead of guessing."""

    #: Frames between two reads beyond which the per-frame record has a gap.
    jump_threshold: int = 12
    #: Consecutive failed reads tolerated before ``read`` raises ``ClockError``.
    max_read_failures: int = 8
    #: Seconds without a frame change before a wait reports ``stall``.
    stall_s: float = 3.0
    #: Treat a rollback as the start of a new episode (C-01) rather than an error.
    rollback_starts_new_episode: bool = True


@dataclass(frozen=True)
class WaitOutcome:
    """Result of waiting for a frame, plus every frame observed on the way."""

    reading: FrameReading
    reached: bool
    reason: str
    polls: int
    samples: tuple[FrameReading, ...] = ()

    @property
    def ok(self) -> bool:
        return self.reached or self.reason in ("reached", "left_game")


class ExperimentClock:
    """Monotonic ``gameplay_frame`` reader with explicit anomaly bookkeeping."""

    def __init__(
        self,
        read_state: Callable[..., Any],
        *,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        config: ClockConfig | None = None,
    ) -> None:
        self._read_state = read_state
        self._monotonic = monotonic or time.perf_counter
        self._sleep = sleep or time.sleep
        self.config = config or ClockConfig()
        self._last: FrameReading | None = None
        self._epoch = 0
        self._failures = 0
        self._reads = 0
        self._anomalies: list[dict] = []
        self._jumps = 0
        self._rollbacks = 0

    # -- properties -------------------------------------------------------
    @property
    def last(self) -> FrameReading | None:
        return self._last

    @property
    def last_frame(self) -> int | None:
        return None if self._last is None else int(self._last.frame)

    @property
    def epoch(self) -> int:
        """Incremented on every rollback: a new episode, not a negative exposure."""
        return self._epoch

    @property
    def anomalies(self) -> tuple[dict, ...]:
        return tuple(self._anomalies)

    @property
    def reads(self) -> int:
        return self._reads

    # -- reading ----------------------------------------------------------
    def read(self) -> FrameReading:
        """Read once, classify against the previous reading, and record anomalies."""

        wall = float(self._monotonic())
        try:
            state = self._read_state(with_bullets=False, with_title=False)
        except OSError as error:  # memory not committed / process gone
            self._failures += 1
            self._note(ANOMALY_READ_FAIL, None, str(error))
            if self._failures > self.config.max_read_failures:
                raise ClockError(f"read_state failed {self._failures} times: {error}") from error
            raise
        self._failures = 0
        self._reads += 1
        reading = reading_from_state(state, wall)
        self._classify(reading)
        self._last = reading
        return reading

    def _classify(self, reading: FrameReading) -> None:
        previous = self._last
        if previous is None:
            return
        delta = reading.frame - previous.frame
        if delta < 0:
            self._rollbacks += 1
            self._note(
                ANOMALY_ROLLBACK,
                reading.frame,
                f"gameplay_frame {previous.frame} -> {reading.frame}",
                previous_frame=int(previous.frame),
                epoch=int(self._epoch) + 1,
            )
            if self.config.rollback_starts_new_episode:
                self._epoch += 1
            return
        if delta > self.config.jump_threshold:
            self._jumps += 1
            self._note(
                ANOMALY_JUMP,
                reading.frame,
                f"gameplay_frame advanced {delta} frames between reads",
                previous_frame=int(previous.frame),
                delta=int(delta),
            )
        if reading.supervisor_state != GAME_SUPERVISOR_STATE:
            self._note(
                ANOMALY_LEFT_GAME,
                reading.frame,
                f"supervisor_state={reading.supervisor_state}",
            )

    def _note(self, kind: str, frame: int | None, detail: str, **extra: Any) -> None:
        self._anomalies.append(
            {
                "kind": kind,
                "frame": None if frame is None else int(frame),
                "wall_s": round(float(self._monotonic()), 6),
                "detail": detail,
                **extra,
            }
        )

    # -- waiting ----------------------------------------------------------
    def wait_until(
        self,
        target_frame: int,
        *,
        timeout_s: float = 2.0,
        poll_s: float = 0.0005,
        collect: bool = True,
    ) -> WaitOutcome:
        """Poll until ``gameplay_frame >= target_frame``.

        Returns the reached reading, or a ``WaitOutcome`` whose ``reason`` says
        why the wait stopped (``timeout``, ``left_game``, ``rollback``,
        ``stall``).  No deadline is ever reported as success.
        """

        target = int(target_frame)
        started = self._monotonic()
        polls = 0
        samples: list[FrameReading] = []
        seen: set[int] = set()
        last_change = started
        previous_frame = self.last_frame
        while True:
            polls += 1
            try:
                reading = self.read()
            except OSError:
                reading = None
            if reading is not None:
                if collect and reading.frame not in seen:
                    seen.add(reading.frame)
                    samples.append(reading)
                if reading.frame >= target:
                    return WaitOutcome(reading, True, "reached", polls, tuple(samples))
                if not reading.in_game:
                    return WaitOutcome(reading, False, "left_game", polls, tuple(samples))
                if previous_frame is not None and reading.frame < previous_frame:
                    return WaitOutcome(reading, False, "rollback", polls, tuple(samples))
                if reading.frame != previous_frame:
                    previous_frame = reading.frame
                    last_change = self._monotonic()
            now = self._monotonic()
            if now - started >= timeout_s:
                self._note(
                    ANOMALY_TIMEOUT,
                    self.last_frame,
                    f"wait_until({target}) timed out after {timeout_s:.3f}s",
                )
                return WaitOutcome(self._last or _null_reading(now), False, "timeout", polls, tuple(samples))
            if now - last_change >= self.config.stall_s:
                self._note(
                    ANOMALY_STALL,
                    self.last_frame,
                    f"gameplay_frame stuck at {self.last_frame} for {self.config.stall_s:.3f}s",
                )
                return WaitOutcome(self._last or _null_reading(now), False, "stall", polls, tuple(samples))
            self._sleep(poll_s)

    def wait_advance(
        self,
        frames: int,
        *,
        timeout_s: float = 2.0,
        poll_s: float = 0.0005,
        collect: bool = True,
    ) -> WaitOutcome:
        """Poll until the counter has advanced by ``frames`` from *now*."""

        base = self.read()
        return self.wait_until(
            int(base.frame) + int(frames),
            timeout_s=timeout_s,
            poll_s=poll_s,
            collect=collect,
        )

    def summary(self) -> dict:
        return {
            "schema": CLOCK_SCHEMA,
            "reads": int(self._reads),
            "last_frame": self.last_frame,
            "epoch": int(self._epoch),
            "rollbacks": int(self._rollbacks),
            "jumps": int(self._jumps),
            "anomalies": list(self._anomalies),
            "anomaly_counts": _count(kind["kind"] for kind in self._anomalies),
            "jump_threshold": int(self.config.jump_threshold),
            "stall_s": float(self.config.stall_s),
        }


def _null_reading(wall: float) -> FrameReading:
    return FrameReading(frame=-1, wall_s=wall, read_ok=False)


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# Windows and events
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameWindow:
    """The preregistered logical-frame window ``[start_frame, start_frame + frames)``."""

    start_frame: int
    frames: int

    def __post_init__(self) -> None:
        if self.frames < 1:
            raise ValueError("window needs at least one frame")
        object.__setattr__(self, "start_frame", int(self.start_frame))
        object.__setattr__(self, "frames", int(self.frames))

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.frames

    def contains(self, frame: int) -> bool:
        return self.start_frame <= int(frame) < self.end_frame

    def classify(self, frame: int) -> str:
        value = int(frame)
        if value < self.start_frame:
            return "before"
        if value >= self.end_frame:
            return "at_or_after_end"
        return "inside"

    def exposure(self, end_frame: int) -> int:
        """Frames actually observed inside the window.  Never negative, never over.

        A rollback can make ``end_frame < start_frame`` and a boundary read can
        make it exceed ``start_frame + frames``; exposure is a denominator in
        every later rate, so neither may inflate or invert it.
        """

        observed = int(end_frame) - self.start_frame
        return max(0, min(int(self.frames), observed))

    def straddles(self, lower: int, upper: int) -> bool:
        """True when an attribution interval is not wholly inside or outside."""

        inside = self.contains(lower) or self.contains(upper)
        outside = (upper >= self.end_frame) or (lower < self.start_frame)
        return inside and outside

    def as_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frames": self.frames,
        }


@dataclass(frozen=True)
class FrameInterval:
    """The frames an event could have happened on.

    A single-frame poll gives ``lower == upper``; a gap (a jump, a slow read, a
    planning stall) gives a real interval, and the interval is what decides
    whether the event belongs to the window.  ``boundary_unknown`` is the honest
    answer when the interval touches both sides -- C-02 forbids guessing.
    """

    lower: int
    upper: int

    def __post_init__(self) -> None:
        if self.upper < self.lower:
            raise ValueError("interval upper < lower")

    @property
    def exact(self) -> bool:
        return self.upper == self.lower

    def classify(self, window: FrameWindow) -> str:
        if self.exact:
            return window.classify(self.lower)
        if window.contains(self.lower) and window.contains(self.upper):
            return "inside"
        if self.upper < window.start_frame:
            return "before"
        if self.lower >= window.end_frame:
            return "at_or_after_end"
        return "boundary_unknown"

    def as_dict(self) -> dict:
        return {"lower": int(self.lower), "upper": int(self.upper), "exact": bool(self.exact)}


@dataclass
class EventLedger:
    """Counts deaths/bombs with frame attribution from a per-frame reading series.

    The game exposes cumulative counters (``deaths``, ``bombs_used``) and a
    player state.  A counter that increases between two observed frames bounds
    the event to ``(upper_prev, upper_now]``; that interval -- not the read time
    -- is what the window accounting uses.
    """

    window: FrameWindow
    deaths: list[dict] = field(default_factory=list)
    bombs: list[dict] = field(default_factory=list)
    _last: FrameReading | None = None
    _last_deaths: int | None = None
    _last_bombs_used: int | None = None

    def __post_init__(self) -> None:
        self._last_deaths = None
        self._last_bombs_used = None

    def observe(self, reading: FrameReading) -> None:
        """Feed every observed frame; counters are cumulative within one episode."""

        if self._last is None:
            self._last = reading
            self._last_deaths = int(reading.deaths)
            self._last_bombs_used = int(reading.bombs_used)
            return
        previous = self._last
        if int(reading.frame) == int(previous.frame):
            # Re-feeding the frame we already saw (a poll that landed on the same
            # frame, or the harness seeding its first sample).  Update the counters
            # in place; there is no interval and no event.
            self._last = reading
            self._last_deaths = int(reading.deaths)
            self._last_bombs_used = int(reading.bombs_used)
            return
        if int(reading.frame) < int(previous.frame):
            # Non-monotonic feed: a rollback, a re-read of the same frame, or a
            # restart.  There is no interval to attribute, and inventing one is
            # how a "death in the window" gets counted twice.
            self._last = reading
            self.deaths.append(
                {
                    "kind": "frame_regression",
                    "frame_lower": int(reading.frame),
                    "frame_upper": int(reading.frame),
                    "attribution": "boundary_unknown",
                    "reason": f"observed frame {reading.frame} after {previous.frame}",
                }
            )
            self._last_deaths = int(reading.deaths)
            self._last_bombs_used = int(reading.bombs_used)
            return
        self._last = reading
        interval = FrameInterval(int(previous.frame) + 1, int(reading.frame))

        if self._last_deaths is not None:
            delta = int(reading.deaths) - self._last_deaths
            if delta > 0:
                for _ in range(min(delta, 8)):
                    self.deaths.append(
                        self._event(reading, interval, window_y_note=None)
                    )
                    interval = FrameInterval(int(reading.frame), int(reading.frame))
            elif delta < 0:
                # Death counter reset: a new run (continue / restart).  That is a
                # different episode, i.e. a protocol problem, not a negative count.
                self.deaths.append(
                    {
                        "kind": "counter_reset",
                        "frame_lower": int(reading.frame),
                        "frame_upper": int(reading.frame),
                        "attribution": "boundary_unknown",
                        "reason": "deaths counter decreased (new run inside the window)",
                    }
                )
            self._last_deaths = int(reading.deaths)

        if self._last_bombs_used is not None:
            delta = int(reading.bombs_used) - self._last_bombs_used
            if delta > 0:
                for _ in range(min(delta, 8)):
                    self.bombs.append(
                        {
                            "kind": "bomb",
                            "frame_lower": int(interval.lower),
                            "frame_upper": int(interval.upper),
                            "attribution": interval.classify(self.window),
                            "bombs_left": round(float(reading.bombs), 3),
                        }
                    )
                    interval = FrameInterval(int(reading.frame), int(reading.frame))
            self._last_bombs_used = int(reading.bombs_used)

    def _event(self, reading: FrameReading, interval: FrameInterval, *, window_y_note) -> dict:
        return {
            "kind": "death",
            "frame_lower": int(interval.lower),
            "frame_upper": int(interval.upper),
            "attribution": interval.classify(self.window),
            "player_state": int(reading.player_state),
            "x": round(float(reading.x), 4),
            "y": round(float(reading.y), 4),
            "lives": round(float(reading.lives), 3),
            "bombs": round(float(reading.bombs), 3),
            "deaths_total": int(reading.deaths),
        }

    def summary(self) -> dict:
        death_counts = _attribute(self.deaths)
        bomb_counts = _attribute(self.bombs)
        return {
            "window": self.window.as_dict(),
            "deaths": self.deaths,
            "bombs": self.bombs,
            "deaths_in_window": death_counts["in_window"],
            "deaths_before_window": death_counts["before_window"],
            "deaths_after_window": death_counts["after_window"],
            "boundary_unknown": death_counts["boundary_unknown"],
            "death_events_total": death_counts["total"],
            "bomb_count": bomb_counts["in_window"],
            "bombs_boundary_unknown": bomb_counts["boundary_unknown"],
        }


def _attribute(events: list[dict]) -> dict:
    """Turn attribution labels into the counts a report may quote."""

    inside = sum(1 for event in events if event.get("attribution") == "inside")
    before = sum(1 for event in events if event.get("attribution") == "before")
    after = sum(1 for event in events if event.get("attribution") == "at_or_after_end")
    unknown = sum(1 for event in events if event.get("attribution") == "boundary_unknown")
    return {
        "in_window": int(inside),
        "before_window": int(before),
        "after_window": int(after),
        "boundary_unknown": int(unknown),
        "total": len(events),
    }


# ---------------------------------------------------------------------------
# Terminating a window
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TerminationRule:
    """The preregistered stop rule: real frames only.

    ``frames`` is the window length in logical frames; ``slots`` is how many
    action slots of ``frameskip`` frames fit inside it.  Both are derived from
    the frame window, never from a decision count, so "900 decisions" can no
    longer stand in for "1800 frames".
    """

    window: FrameWindow
    frameskip: int = 2

    def __post_init__(self) -> None:
        if self.frameskip < 1:
            raise ValueError("frameskip must be >= 1")
        object.__setattr__(self, "frameskip", int(self.frameskip))

    @property
    def slots(self) -> int:
        return self.window.frames // self.frameskip

    def slot_start(self, slot: int) -> int:
        return self.window.start_frame + int(slot) * self.frameskip

    def slot_end(self, slot: int) -> int:
        return self.slot_start(slot) + self.frameskip

    def fits(self, slot: int, act_lag: int = 0) -> bool:
        """Whether a full ``frameskip`` action for ``slot`` still lands in the window."""

        return self.slot_end(slot) + int(act_lag) <= self.window.end_frame

    def stop_reason(self, frame: int, slot: int, act_lag: int = 0) -> str | None:
        if not self.fits(slot, act_lag):
            return "window_complete"
        if int(frame) >= self.window.end_frame:
            return "window_complete"
        return None
