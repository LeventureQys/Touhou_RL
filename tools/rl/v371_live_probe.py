"""Bounded live-clock probe: can the game boot and advance frames?

This is the smallest possible check before any input is sent.  It does not press
anything (``menu.enter_game`` navigates the menus, which is unavoidable), then
reads the frame counter for ~1 s and reports the advance pattern.

    python tools/rl/v371_live_probe.py --out <path>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from touhou_rl.env import EnvConfig, Th08Env  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--out", default=".tmp/v371-live-probe.json")
    parser.add_argument("--run-dir", default=".tmp/v371-probe-session")
    args = parser.parse_args(argv)

    report: dict = {"schema": "v371-live-probe/1", "steps": []}
    env = Th08Env(
        EnvConfig(
            run_dir=Path(args.run_dir),
            practice_stage=0,
            difficulty=3,
            frameskip=2,
            windowed=True,
            use_postmessage_input=False,
        )
    )
    started = time.perf_counter()
    try:
        report["boot_start"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        observation, info = env.reset()
        report["reset_s"] = round(time.perf_counter() - started, 3)
        state = env.last_state
        report["reset_state"] = {
            "gameplay_frame": int(state.gameplay_frame),
            "stage": int(state.stage),
            "supervisor_state": int(state.supervisor_state),
            "difficulty_hint": int(getattr(state, "difficulty_hint", -1)),
            "is_practice": bool(getattr(state, "is_practice", False)),
            "lives": float(state.lives),
            "bombs": float(state.bombs),
            "player_state": int(getattr(state, "player_state", -1)),
            "field": [float(state.field_x), float(state.field_y), float(state.field_w), float(state.field_h)],
        }
        reader = env.session.reader
        deadline = time.perf_counter() + args.seconds
        last = None
        while time.perf_counter() < deadline:
            current = reader.read_state(with_bullets=False, with_title=False)
            if last is None or current.gameplay_frame != last.gameplay_frame:
                report["steps"].append(
                    {
                        "frame": int(current.gameplay_frame),
                        "input_mask": int(current.input_mask),
                        "player": [round(float(current.player_x), 2), round(float(current.player_y), 2)],
                        "deaths": int(current.deaths),
                        "bombs": float(current.bombs),
                    }
                )
            last = current
            time.sleep(0.002)
        if report["steps"]:
            frames = [row["frame"] for row in report["steps"]]
            report["distinct_frames"] = len(frames)
            report["frame_span"] = [frames[0], frames[-1]]
            report["observed_rate_hz"] = round(len(frames) / max(1e-6, args.seconds), 2)
            report["max_gap"] = max((b - a) for a, b in zip(frames, frames[1:])) if len(frames) > 1 else 0
        report["ok"] = True
    except Exception as error:  # noqa: BLE001 - the probe reports, it does not raise
        report["ok"] = False
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report["elapsed_s"] = round(time.perf_counter() - started, 3)
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "steps"}, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
