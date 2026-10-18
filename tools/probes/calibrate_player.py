"""Measure the TH08 player movement model on the live game.

Why this exists
---------------
The scripted avoidance teacher (``touhou_rl/pilot.py``) plans by predicting where
the player *would* be after holding a direction for a few frames.  That model
needs three numbers it cannot get from the observation:

* the per-axis speed when not focused,
* the per-axis speed when focused (holding shift),
* the diagonal speed (which is *not* ``speed / sqrt(2)`` in TH08: the shot files
  carry separate ``axisSpeed`` and ``diagonalSpeed`` values, and they differ per
  team).

Guessing them costs survival: a 20% speed error over a 20-frame plan is a 16 px
position error, which is the size of a dense bullet cluster.

Method (no writes, no injection)
--------------------------------
Centre the player away from the movement-area edges, then for each of the nine
movement classes and both focus states hold the key set for a fixed number of
logic frames and measure ``delta_position / frames``.  Each test is undone by
holding the opposite direction for the same number of frames, so the player
returns to the centre without drifting.

The probe also reads ``Player::size`` (the hurtbox half extents) in both focus
states, because the planner's collision margin depends on it.

Output: ``third_party/th08/build/rl/probe-v103/player-calibration.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from touhou_rl import spec, winapi
from touhou_rl.env import EnvConfig, Th08Env

ROOT = Path(__file__).resolve().parents[2]

KEY_FOR_MOVE = {
    0: set(),
    1: {"up"},
    2: {"up", "right"},
    3: {"right"},
    4: {"down", "right"},
    5: {"down"},
    6: {"down", "left"},
    7: {"left"},
    8: {"up", "left"},
}
OPPOSITE = {0: 0, 1: 5, 2: 6, 3: 7, 4: 8, 5: 1, 6: 2, 7: 3, 8: 4}


def _focus_guard(session) -> None:
    hwnd = session.refresh_window()
    if hwnd and winapi.user32.GetForegroundWindow() != hwnd:
        winapi.force_foreground(hwnd)


def _read(session):
    return session.read_state(with_bullets=False, with_title=False)


def _wait_frames(session, frames: int, timeout: float = 2.0):
    """Block until ``gameplay_frame`` advanced by at least ``frames``."""
    start = _read(session)
    base = start.gameplay_frame
    deadline = time.perf_counter() + timeout
    state = start
    while time.perf_counter() < deadline:
        state = _read(session)
        if state.supervisor_state != spec.STATE_GAME_MANAGER:
            break
        if ((state.gameplay_frame - base) & 0xFFFFFFFF) >= frames:
            break
        time.sleep(0)
    return start, state


def _hold(session, keys: set[str], frames: int, settle: int = 2):
    """Hold ``keys`` for ``frames`` logic frames, measured after a settle delay.

    The settle matters: the game samples the keyboard once per logic frame, so
    the first frame after ``set_keys`` still runs with the previous key state.
    Measuring from that first frame contaminates every sample with one frame of
    the *previous* direction (measured error: 0.35 px/frame on a 3.5 px/frame
    axis speed).
    """
    if keys:
        session.keyboard.set_keys(keys)
    else:
        session.keyboard.release_all()
    if settle:
        _wait_frames(session, settle)
    start, end = _wait_frames(session, frames)
    # Let the movement stop before the next measurement starts.
    session.keyboard.release_all()
    if settle:
        _wait_frames(session, settle)
    return start, end


def measure(env: Th08Env, out: Path) -> dict:
    session = env.session
    assert session is not None
    _focus_guard(session)
    # --- 1) movement-area clamp: hold each direction long enough to stop ------
    # The planner clips candidate positions to this rectangle, so a wrong bound
    # silently makes it believe it can dodge into a wall.
    _hold(session, {"right"}, 200)
    x_max = _read(session).player_x
    _hold(session, {"left"}, 260)
    x_min = _read(session).player_x
    _hold(session, {"down"}, 200)
    y_max = _read(session).player_y
    _hold(session, {"up"}, 260)
    y_min = _read(session).player_y

    def centre() -> None:
        # Walk to the middle of the measured rectangle, one axis at a time.
        target_x = 0.5 * (x_min + x_max)
        target_y = 0.5 * (y_min + y_max)
        cur = _read(session)
        steps_x = int(abs(target_x - cur.player_x) / 4.0)
        if steps_x:
            _hold(session, {"right"} if target_x > cur.player_x else {"left"}, steps_x)
        cur = _read(session)
        steps_y = int(abs(target_y - cur.player_y) / 4.0)
        if steps_y:
            _hold(session, {"down"} if target_y > cur.player_y else {"up"}, steps_y)

    centre()
    centred = _read(session)

    results: dict[str, dict] = {}
    deaths_before = centred.deaths
    for focus in (0, 1):
        for move in range(9):
            keys = set(KEY_FOR_MOVE[move])
            if focus:
                keys.add("shift")
            _focus_guard(session)
            start, end = _hold(session, keys, 12)
            frames = ((end.gameplay_frame - start.gameplay_frame) & 0xFFFFFFFF) or 1
            dx = end.player_x - start.player_x
            dy = end.player_y - start.player_y
            died = end.deaths != start.deaths
            name = f"{'focus' if focus else 'normal'}-m{move}"
            results[name] = {
                "move": move,
                "focus": focus,
                "frames": frames,
                "dx_per_frame": round(dx / frames, 4),
                "dy_per_frame": round(dy / frames, 4),
                "velocity_field": [round(start.player_vx, 4), round(start.player_vy, 4)],
                "hurtbox_half": [round(start.hurtbox_half_x, 3), round(start.hurtbox_half_y, 3)],
                "focus_field": int(start.focus_mode),
                # A death respawns the player mid-sample, so the delta of that
                # sample is meaningless; it is reported but flagged.
                "valid": not died,
            }
            if died:
                centre()
                continue
            # Undo the displacement so the next measurement starts from the same
            # place (diagonals undo exactly because both axes are mirrored).
            if move != 0:
                undo = set(KEY_FOR_MOVE[OPPOSITE[move]])
                if focus:
                    undo.add("shift")
                _hold(session, undo, 12)
    deaths_after = _read(session).deaths

    final = _read(session)
    report = {
        "captured": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "difficulty": env.config.difficulty,
        "team": env.config.team,
        "clamp": {
            "x_min": round(x_min, 3),
            "x_max": round(x_max, 3),
            "y_min": round(y_min, 3),
            "y_max": round(y_max, 3),
        },
        "centred_position": [round(centred.player_x, 2), round(centred.player_y, 2)],
        "final_position": [round(final.player_x, 2), round(final.player_y, 2)],
        "samples": results,
        "deaths_during_probe": deaths_after - deaths_before,
        "move_area": [spec.MOVE_AREA_X, spec.MOVE_AREA_Y, spec.MOVE_AREA_W, spec.MOVE_AREA_H],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure TH08 player movement model")
    parser.add_argument("--difficulty", default="lunatic", choices=["easy", "normal", "hard", "lunatic"])
    parser.add_argument("--team", type=int, default=0)
    parser.add_argument(
        "--out",
        default=str(ROOT / "third_party/th08/build/rl/probe-v103/player-calibration.json"),
    )
    args = parser.parse_args(argv)
    difficulty = {"easy": spec.EASY, "normal": spec.NORMAL, "hard": spec.HARD, "lunatic": spec.LUNATIC}[
        args.difficulty
    ]
    env = Th08Env(
        EnvConfig(
            difficulty=difficulty,
            team=args.team,
            frameskip=1,
            max_steps=200000,
            reset_mode="relaunch",
            run_dir=ROOT / "third_party/th08/build/rl/calib-session",
        )
    )
    try:
        env.reset()
        report = measure(env, Path(args.out))
    finally:
        env.close()
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, ensure_ascii=False, indent=2))
    for name, row in report["samples"].items():
        print(
            f"{name:12s} d=({row['dx_per_frame']:+.3f},{row['dy_per_frame']:+.3f}) "
            f"vel={row['velocity_field']} hurtbox={row['hurtbox_half']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
