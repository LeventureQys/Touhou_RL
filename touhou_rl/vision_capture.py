"""Capture vision shards from a replay: frames plus privileged labels.

Why replay playback is still the cheapest vision data source
------------------------------------------------------------
A replay makes the game feed an expert input stream, so a capture run yields
``(frame, privileged state, expert action)`` triples with no human at the
keyboard and no input injection.  The privileged arrays are *labels* - they are
what the vision student is asked to reproduce - which is exactly the role the
memory-reading stack plays in the new architecture.

Sampling rate
-------------
Captured at the *decision* rate (``every`` frames, default 2 = 30 Hz) rather than
every logic frame: a frame tensor is ~43 KB, so 60 Hz would be 2.6 MB/s of disk
and twice the labels for a policy that can only act every other frame.  The
stride is explicit so a future 60 Hz study can change one argument.

Labels produced here
--------------------
``action``   what the human pressed (``g_GuiMessageInputCurrent``)
``self_feat``/``bullets``/``danger``  privileged observation
Teacher *values* are deliberately not written: they are a pure function of the
privileged observation, so they are computed offline in
:mod:`touhou_rl.vision_train` (and recomputed for free if the teacher changes).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import menu, obs, spec, vision, winapi
from .session import GameSession, SessionOptions, prepare_run_dir
from .vision_data import VisionShardWriter

ROOT = Path(__file__).resolve().parents[1]


def capture_vision(
    replay: Path,
    out_dir: Path,
    *,
    tag: str = "vision",
    every: int = 2,
    max_seconds: float = 600.0,
    stage: int | None = None,
    box: vision.PlayfieldBox | None = None,
    template_dir: Path | None = None,
    run_dir: Path | None = None,
    shard_size: int = 512,
    allow_append: bool = False,
    verbose: bool = True,
) -> dict:
    template = template_dir or (ROOT / "third_party/th08/build/play")
    work = run_dir or (ROOT / "third_party/th08/build/rl/vision-session")
    out_dir = Path(out_dir)
    box = box or vision.PlayfieldBox.for_th08()
    if every < 1:
        raise ValueError("every must be >= 1")

    if stage is None:
        from .capture_dataset import replay_first_stage

        stage = replay_first_stage(Path(replay))

    prepare_run_dir(work, template, windowed=True)
    menu.install_replay(work, Path(replay))
    writer = VisionShardWriter(
        out_dir,
        shard_size=shard_size,
        tag=tag,
        provenance={
            "replay": str(replay),
            "tag": tag,
            "stage_start": int(stage),
            "every": int(every),
            "box": [box.x, box.y, box.w, box.h],
            "label": "human_expert",
        },
        allow_append=allow_append,
    )
    session = GameSession(SessionOptions(run_dir=work))
    started = time.time()
    report = {
        "replay": str(replay),
        "frames": 0,
        "stages_seen": [],
        "ended_reason": "",
        "elapsed_seconds": 0.0,
        "box": [box.x, box.y, box.w, box.h],
        "frame_shape": [vision.VISION_H, vision.VISION_W, vision.VISION_CHANNELS],
        "every": int(every),
    }
    transient = {
        spec.STATE_GAME_MANAGER_REINIT,
        spec.STATE_RESTART_FROM_BEGINNING,
        12,
    }
    try:
        session.start()
        menu.enter_replay(session, stage=int(stage), timeout=60.0)
        reader = session.reader
        assert reader is not None
        session.keyboard.release_all()
        last_key: tuple[int, int] | None = None
        last_guard = 0.0
        last_progress = time.time()
        while time.time() - started < max_seconds:
            now = time.time()
            if now - last_guard > 2.0:
                last_guard = now
                if winapi.user32.GetForegroundWindow() != session.hwnd:
                    winapi.force_foreground(session.hwnd)
            state = reader.read_state(with_bullets=True, with_title=False)
            if state.supervisor_state != spec.STATE_GAME_MANAGER:
                if state.supervisor_state in transient:
                    time.sleep(0.02)
                    if time.time() - last_progress > 90.0:
                        report["ended_reason"] = "stalled during transition"
                        break
                    continue
                report["ended_reason"] = f"left game (supervisor_state={state.supervisor_state})"
                break
            if not state.is_replay and state.gameplay_frame > 60:
                report["ended_reason"] = "replay flag cleared"
                break
            key = (state.stage, state.gameplay_frame)
            if last_key is not None and key == last_key:
                continue
            if last_key is not None and key[0] != last_key[0]:
                if state.stage not in report["stages_seen"]:
                    report["stages_seen"].append(state.stage)
            # Only every ``every``-th logic frame is recorded (decision rate).
            if last_key is not None and key[0] == last_key[0]:
                if (key[1] - last_key[1]) % every != 0:
                    continue
            last_key = key
            last_progress = time.time()

            field = reader.read_bullets()
            observation = obs.encode(state, field)
            try:
                frame = vision.encode_frame(session.grab(), box)
            except Exception as exc:  # a transient BitBlt failure must not kill a long capture
                if verbose:
                    print(f"[vision-capture] grab failed: {exc!r}", flush=True)
                continue
            action = spec.buttons_to_action(state.gui_input_current)
            writer.add(
                frame,
                observation,
                action,
                {
                    "frame": state.gameplay_frame,
                    "stage": state.stage,
                    "score": state.score,
                    "graze": state.graze,
                    "lives": state.lives,
                    "bombs": state.bombs,
                    "power": state.power,
                    "deaths": state.deaths,
                },
            )
            report["frames"] += 1
            if verbose and report["frames"] % 1000 == 0:
                elapsed = max(1e-6, time.time() - started)
                print(
                    f"[vision-capture] {report['frames']} frames "
                    f"({report['frames'] / elapsed:.1f}/s) stage={state.stage} "
                    f"frame={state.gameplay_frame} lives={state.lives:.0f}",
                    flush=True,
                )
        else:
            report["ended_reason"] = "max_seconds reached"
    finally:
        summary = writer.close()
        try:
            session.close()
        except Exception:
            pass
        report["elapsed_seconds"] = round(time.time() - started, 1)
        report["shards"] = len(summary["shards"])
        report["bytes"] = sum((out_dir / name).stat().st_size for name in summary["shards"]) if summary["shards"] else 0
        (out_dir / "capture-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture vision shards from a TH08 replay")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="vision")
    parser.add_argument("--every", type=int, default=2, help="record every N-th logic frame")
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--stage", type=int, default=-1)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = capture_vision(
        Path(args.replay),
        Path(args.out),
        tag=args.tag,
        every=args.every,
        max_seconds=args.max_seconds,
        stage=None if args.stage < 0 else args.stage,
        shard_size=args.shard_size,
        allow_append=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
