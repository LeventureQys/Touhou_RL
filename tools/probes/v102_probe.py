"""v1.0.2 feasibility probe: single-instance throughput, read costs, instance lock.

The v1.0.1 design assumed that ``opts.dontUseDirectInput`` + ``PostMessage``
would allow background/parallel instances.  This probe tests that assumption
and, when it fails, measures what the runtime can actually do:

1. PostMessage input reaches the game or not (GetKeyboardState limitation).
2. Reading scalars / the bullet pool / a frame: measured cost.
3. Closed-loop decision rate at frame skips 1/2/4.
4. Whether the game keeps simulating while its window is not foreground.
5. Whether a second ``th08.exe`` can run (the ``Touhou 08 App`` mutex).

Run from the repository root:
    python tools/probes/v102_probe.py --frame-seconds 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from touhou_rl import menu, spec, winapi  # noqa: E402
from touhou_rl.session import GameSession, SessionOptions, prepare_run_dir  # noqa: E402


def timed(fn, n=20) -> dict:
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    return {
        "n": n,
        "mean_ms": round(statistics.mean(times), 3),
        "p50_ms": round(statistics.median(times), 3),
        "p95_ms": round(times[max(0, int(len(times) * 0.95) - 1)], 3),
        "max_ms": round(times[-1], 3),
    }


def measure_fps(session: GameSession, seconds: float) -> dict:
    reader = session.reader
    assert reader is not None
    first = reader.read_state(with_bullets=False, with_title=False)
    samples = 0
    deltas: list[int] = []
    last = first.gameplay_frame
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        state = reader.read_state(with_bullets=False, with_title=False)
        samples += 1
        if state.gameplay_frame != last:
            deltas.append((state.gameplay_frame - last) & 0xFFFFFFFF)
            last = state.gameplay_frame
    elapsed = time.perf_counter() - (deadline - seconds)
    frames = (last - first.gameplay_frame) & 0xFFFFFFFF
    return {
        "samples": samples,
        "gameplay_frames": frames,
        "logic_fps": round(frames / seconds, 2),
        "samples_per_s": round(samples / seconds, 1),
        "frame_step_histogram": {str(k): deltas.count(k) for k in sorted(set(deltas))},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="third_party/th08/build/play")
    parser.add_argument("--work-dir", default="third_party/th08/build/rl")
    parser.add_argument("--evidence", default="")
    parser.add_argument("--frame-seconds", type=float, default=5.0)
    args = parser.parse_args()

    template = ROOT / args.template
    work = ROOT / args.work_dir
    evidence = ROOT / args.evidence if args.evidence else work / "probe-v102"
    evidence.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "template": str(template),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    session: GameSession | None = None
    try:
        run_dir = prepare_run_dir(work / "probe-1", template, windowed=True)
        session = GameSession(SessionOptions(run_dir=run_dir))
        session.start()
        st = session.read_state()
        report["boot"] = {
            "boot_seconds": round(session.boot_seconds, 2),
            "client_size": list(session.client_size),
            "supervisor_state": st.supervisor_state,
            "title_screen_current": st.title_screen_current,
            "title_screen_state": st.title_screen_state,
            "module_base": f"0x{session.mem.module_base:08X}" if session.mem else None,
        }

        # --- 1. PostMessage input (expected to fail) -------------------------
        sw = winapi.PostMessageKeyboard(session.hwnd)
        before = session.read_state(with_bullets=False)
        sw.tap("down", 0.08)
        time.sleep(0.3)
        after = session.read_state(with_bullets=False)
        report["postmessage_input"] = {
            "cursor_before": before.title_cursor,
            "cursor_after": after.title_cursor,
            "works": before.title_cursor != after.title_cursor,
        }
        menu.tap(session, "down", settle_s=0.25)
        after2 = session.read_state(with_bullets=False)
        report["sendinput_input"] = {
            "cursor_before": after.title_cursor,
            "cursor_after": after2.title_cursor,
            "works": after2.title_cursor != after.title_cursor,
        }
        menu.tap(session, "up", settle_s=0.2)

        # --- 2. read costs ---------------------------------------------------
        assert session.reader is not None
        report["read_state_scalars"] = timed(
            lambda: session.reader.read_state(with_bullets=False, with_title=False), 30
        )
        report["read_state_full"] = timed(lambda: session.reader.read_state(), 20)
        report["read_bullets"] = timed(lambda: session.reader.read_bullets(), 20)
        report["read_bullets_bytes"] = spec.BULLET_BLOCK_SIZE
        report["capture"] = timed(lambda: session.grab(), 10)

        # --- 3. enter Lunatic and measure the decision loop -------------------
        menu.enter_game(session, difficulty=spec.LUNATIC, team=0)
        time.sleep(1.0)
        st = session.read_state()
        active = session.reader.read_bullets().active_mask()
        report["in_game"] = {
            "stage": st.stage,
            "lives": st.lives,
            "bombs": st.bombs,
            "power": st.power,
            "gameplay_frame": st.gameplay_frame,
            "player": [round(st.player_x, 2), round(st.player_y, 2)],
            "field": [st.field_x, st.field_y, st.field_w, st.field_h],
            "active_bullets_scalar": st.active_bullets,
            "active_bullets_mask": int(active.sum()),
            "difficulty_hint": st.difficulty_hint,
        }
        report["frame_rate_foreground"] = measure_fps(session, args.frame_seconds)

        loop = {}
        for skip in (1, 2, 4):
            decisions = 0
            inject_cost = []
            began = time.perf_counter()
            last_frame = None
            while decisions < 25 and time.perf_counter() - began < 20:
                state = session.reader.read_state()
                if last_frame is not None and state.gameplay_frame == last_frame:
                    continue
                t0 = time.perf_counter()
                session.keyboard.set_keys({"right"} if decisions % 2 == 0 else {"left"})
                inject_cost.append((time.perf_counter() - t0) * 1000)
                # Hold until the frame counter advanced by `skip`.
                target = (state.gameplay_frame + skip) & 0xFFFFFFFF
                while (session.reader.read_state(with_bullets=False, with_title=False).gameplay_frame & 0xFFFFFFFF) < target:
                    if session.reader.read_state(with_bullets=False, with_title=False).supervisor_state != 2:
                        break
                last_frame = target
                decisions += 1
            session.keyboard.release_all()
            elapsed = time.perf_counter() - began
            loop[str(skip)] = {
                "decisions": decisions,
                "seconds": round(elapsed, 2),
                "decisions_per_s": round(decisions / elapsed, 2),
                "inject_ms_mean": round(statistics.mean(inject_cost), 3) if inject_cost else None,
            }
        report["decision_loop"] = loop

        # --- 4. background simulation ---------------------------------------
        import subprocess

        before_fg = winapi.user32.GetForegroundWindow()
        subprocess.Popen(["notepad.exe"])
        time.sleep(1.5)
        report["background"] = {
            "foreground_changed": winapi.user32.GetForegroundWindow() != before_fg,
            **measure_fps(session, 3.0),
        }
        subprocess.run(["taskkill", "/IM", "notepad.exe", "/F"], capture_output=True)

        from PIL import Image

        Image.fromarray(session.grab()).save(evidence / "probe-stage1-lunatic.png")

        # --- 5. second instance (mutex) --------------------------------------
        second_dir = prepare_run_dir(work / "probe-2", template, windowed=True)
        second = GameSession(SessionOptions(run_dir=second_dir, log_to_file=False))
        try:
            second.start()
            report["second_instance"] = {"started": True}
        except Exception as exc:
            report["second_instance"] = {"started": False, "error": f"{type(exc).__name__}: {exc}"}
            if second.process is not None:
                second.process.kill()
        report["running_th08_pids"] = winapi.list_process_ids("th08.exe")
    except Exception as exc:  # report, do not hide
        report["error"] = f"{type(exc).__name__}: {exc}"
        import traceback

        report["traceback"] = traceback.format_exc()
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as exc:  # pragma: no cover
                report.setdefault("close_errors", []).append(repr(exc))
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    out = evidence / "probe-v102.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
