"""Capture expert (observation, action) pairs by playing a TH08 replay.

TH08 replays store one input word per game frame.  While a replay plays back,
``g_CurFrameInput`` carries *that* input, so a read-only sampler observes the
pair (game state at frame f, expert action at frame f) without touching the
recording.  This is the data source for behaviour cloning -- the reason the
project can imitate Lunatic play at all, since the committed corpus contains 24
full Lunatic clears (``third_party/th08/config/replay-fixtures.json``).

The sampler only records a frame when ``gameplay_frame`` changes, so one shard
row == one real game frame, even though the loop polls faster than 60 Hz.

Usage (from the repository root)::

    python -m touhou_rl.capture_dataset \
        --replay third_party/th08/tests/fixtures/replays/th08-1.00d/accepted/th8_ud2ce5.rpy \
        --out third_party/th08/build/rl/datasets/lunatic-bt \
        --tag lunatic-bt --max-minutes 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import menu, obs, spec, winapi
from .dataset import ShardWriter
from .session import GameSession, SessionOptions, prepare_run_dir

ROOT = Path(__file__).resolve().parents[1]
# The replay header parser lives with the reconstruction sources; reuse it rather
# than re-deriving the obfuscated/LZSS layout here.
_REPLAY_SCRIPTS = ROOT / "third_party/th08/scripts"


def replay_first_stage(replay: Path) -> int:
    """Stage index the replay starts on.

    This matters for Extra replays: they contain a *single* stage (index 8), so
    the replay stage-select screen offers one entry and asking for stage 0 can
    never be satisfied - the menu walk times out after its full timeout.  Returns
    0 when the header cannot be parsed, which keeps the normal case (full runs
    starting at Stage 1) working without the third-party parser.
    """
    try:
        if str(_REPLAY_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(_REPLAY_SCRIPTS))
        from replay_file import inspect as _inspect_replay

        return int(_inspect_replay(Path(replay))["stages"][0]["index"])
    except Exception:
        return 0


@dataclass
class CaptureReport:
    replay: str
    dataset_dir: str
    started: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    frames: int = 0
    stages_seen: list[int] = field(default_factory=list)
    nonzero_action_frames: int = 0
    distinct_actions: set[str] = field(default_factory=set)
    frame_jumps: int = 0
    max_gap: int = 0
    foreground_fixes: int = 0
    ended_reason: str = ""
    elapsed_seconds: float = 0.0
    final_state: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        data = {
            "replay": self.replay,
            "dataset_dir": self.dataset_dir,
            "started": self.started,
            "frames": self.frames,
            "stages_seen": self.stages_seen,
            "nonzero_action_frames": self.nonzero_action_frames,
            "distinct_actions": sorted(self.distinct_actions),
            "frame_jumps": self.frame_jumps,
            "max_gap": self.max_gap,
            "foreground_fixes": self.foreground_fixes,
            "ended_reason": self.ended_reason,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "final_state": self.final_state,
        }
        return data


def capture_replay(
    replay: Path,
    out_dir: Path,
    *,
    tag: str = "replay",
    template_dir: Path | None = None,
    run_dir: Path | None = None,
    shard_size: int = 2048,
    stage: int = 0,
    max_seconds: float = 3600.0,
    start_timeout: float = 60.0,
    overwrite: bool = False,
    verbose: bool = True,
) -> CaptureReport:
    template = template_dir or (ROOT / "third_party/th08/build/play")
    work = run_dir or (ROOT / "third_party/th08/build/rl/capture-session")
    out_dir = Path(out_dir)
    if stage < 0:
        stage = replay_first_stage(Path(replay))
        if verbose:
            print(f"[capture] starting playback at stage index {stage}", flush=True)
    if out_dir.exists() and any(out_dir.glob("*.npz")):
        if not overwrite:
            raise FileExistsError(
                f"{out_dir} already holds a dataset; pass overwrite=True (--overwrite) to replace it"
            )
        for stale in out_dir.glob("*"):
            if stale.is_file():
                stale.unlink()
    prepare_run_dir(work, template, windowed=True)
    menu.install_replay(work, Path(replay))
    report = CaptureReport(replay=str(replay), dataset_dir=str(out_dir))
    writer = ShardWriter(
        out_dir,
        shard_size=shard_size,
        tag=tag,
        provenance={"replay": str(replay), "tag": tag, "stage_start": stage},
    )
    session = GameSession(SessionOptions(run_dir=work))
    started = time.time()
    last_key: tuple[int, int] | None = None
    stages: list[int] = []
    foreground_fixes = 0
    try:
        session.start()
        menu.enter_replay(session, stage=stage, timeout=start_timeout)
        reader = session.reader
        assert reader is not None
        # The replay plays itself; make sure nothing else holds the keyboard.
        session.keyboard.release_all()
        # TH08 drops from 60 to ~14 logic fps when its window is not foreground
        # (measured, see 问题清单 F-07), so keep it focused while sampling.
        last_guard = 0.0
        foreground_fixes = 0
        # During a stage transition the supervisor leaves STATE_GAME_MANAGER for a
        # few frames (GameManagerReInit=3, RestartFromBeginning=10, NextStageWeird=12)
        # and then re-enters it for the next stage; those states are not an exit.
        transient = {
            spec.STATE_GAME_MANAGER_REINIT,  # 3
            spec.STATE_RESTART_FROM_BEGINNING,  # 10
            12,  # SupervisorState_GameManagerNextStageWeird
        }
        last_progress = time.time()
        while time.time() - started < max_seconds:
            now = time.time()
            if now - last_guard > 2.0:
                last_guard = now
                if winapi.user32.GetForegroundWindow() != session.hwnd:
                    if winapi.force_foreground(session.hwnd):
                        foreground_fixes += 1
            state = reader.read_state(with_bullets=True, with_title=False)
            if state.supervisor_state != spec.STATE_GAME_MANAGER:
                if state.supervisor_state in transient:
                    time.sleep(0.03)
                    if time.time() - last_progress > 90.0:
                        report.ended_reason = "stalled during transition"
                        break
                    continue
                report.ended_reason = f"left game (supervisor_state={state.supervisor_state})"
                break
            if not state.is_replay and state.gameplay_frame > 60:
                report.ended_reason = "replay flag cleared"
                break
            key = (state.stage, state.gameplay_frame)
            if last_key is not None:
                if key[0] == last_key[0]:
                    gap = key[1] - last_key[1]
                    if gap == 0:
                        continue
                    if gap > 1:
                        report.frame_jumps += 1
                        report.max_gap = max(report.max_gap, gap)
                else:
                    stages.append(state.stage)
                    if state.stage not in report.stages_seen:
                        report.stages_seen.append(state.stage)
            last_key = key
            last_progress = time.time()
            field = reader.read_bullets()
            observation = obs.encode(state, field)
            # The game consumes g_GuiMessageInputCurrent (Player.cpp), which is
            # replay-fed during playback; g_CurFrameInput stays zero there.
            action = spec.buttons_to_action(state.gui_input_current)
            if any(action):
                report.nonzero_action_frames += 1
            report.distinct_actions.add("".join(str(v) for v in action))
            writer.add(
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
            report.frames += 1
            if verbose and report.frames % 5000 == 0:
                elapsed = max(1e-6, time.time() - started)
                print(
                    f"[capture] {report.frames} frames ({report.frames / elapsed:.1f}/s) "
                    f"stage={state.stage} frame={state.gameplay_frame} lives={state.lives:.0f} "
                    f"bullets={state.active_bullets} fg_fixes={foreground_fixes}",
                    flush=True,
                )
        else:
            report.ended_reason = "max_seconds reached"
        state = reader.read_state(with_bullets=False, with_title=False)
        report.final_state = {            "stage": state.stage,
            "frame": state.gameplay_frame,
            "lives": state.lives,
            "bombs": state.bombs,
            "score": state.score,
            "graze": state.graze,
            "deaths": state.deaths,
            "supervisor_state": state.supervisor_state,
        }
    finally:
        summary = writer.close()
        report.elapsed_seconds = time.time() - started
        report.frames = int(summary["frames"])
        report.foreground_fixes = foreground_fixes
        session.close()
        (out_dir / "capture-report.json").write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture TH08 replay frames into a BC dataset")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tag", default="replay")
    parser.add_argument("--template", default=str(ROOT / "third_party/th08/build/play"))
    parser.add_argument("--run-dir", default=str(ROOT / "third_party/th08/build/rl/capture-session"))
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument(
        "--stage",
        type=int,
        default=-1,
        help="stage index to start playback from (-1 = auto: the replay's own first stage, "
        "which is required for Extra replays since they only offer stage 8)",
    )
    parser.add_argument("--max-minutes", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing dataset directory")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    report = capture_replay(
        Path(args.replay),
        Path(args.out),
        tag=args.tag,
        template_dir=Path(args.template),
        run_dir=Path(args.run_dir),
        shard_size=args.shard_size,
        stage=args.stage,
        max_seconds=args.max_minutes * 60.0,
        overwrite=args.overwrite,
        verbose=not args.quiet,
    )
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
