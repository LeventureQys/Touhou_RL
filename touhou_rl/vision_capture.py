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
import hashlib
import json
import time
from pathlib import Path

from . import menu, obs, spec, vision, winapi
from .session import GameSession, SessionOptions, prepare_run_dir
from .vision_data import VisionShardWriter

ROOT = Path(__file__).resolve().parents[1]


class FrozenCapture(RuntimeError):
    """The captured pixels stopped changing while the game kept running."""


class FrameStability:
    """Detect a stale window DC.

    A live shmup never renders ``limit`` identical frames in a row when sampled
    every ``every``-th logic frame; a window that stopped being composited hands
    back the same buffer forever.  Since the recorder only accepts a sample when
    the game frame number advanced, consecutive identical samples mean the pixels
    are stale - not that the scene happens to be still.

    ``limit=0`` disables the check (kept for debugging only).
    """

    def __init__(self, limit: int = 5):
        self.limit = max(0, int(limit))
        self.last: bytes | None = None
        self.identical_run = 0  # repeated frames *beyond* the first of the run
        self.longest = 0  # longest run of identical frames, counted inclusive
        self.frozen_frames = 0
        self.frozen = False

    @property
    def run_length(self) -> int:
        """Frames in the current run of identical pictures, counted inclusive."""
        return self.identical_run + 1 if self.last is not None else 0

    def update(self, digest: bytes) -> bool:
        """Feed one frame digest; returns True when it repeats the previous one."""
        same = self.last is not None and digest == self.last
        self.last = digest
        self.identical_run = self.identical_run + 1 if same else 0
        self.longest = max(self.longest, self.run_length)
        if self.limit and self.run_length >= self.limit:
            self.frozen = True
            self.frozen_frames += 1
        return same

    def check(self) -> None:
        """Raise once the run of identical frames reaches the limit."""
        if not self.limit or not self.frozen:
            return
        raise FrozenCapture(
            f"{self.run_length} identical frames in a row (limit {self.limit}) while the game "
            "frame counter kept advancing: the window DC stopped updating "
            "(locked session, display off, minimised, or occluded)"
        )

    def summary(self) -> dict:
        return {
            "freeze_limit": self.limit,
            "frozen_frames": self.frozen_frames,
            "longest_identical_run": self.longest,
        }


def capture_vision(
    replay: Path,
    out_dir: Path,
    *,
    tag: str = "vision",
    every: int = 2,
    max_seconds: float = 600.0,
    stage: int | None = None,
    box: vision.PlayfieldBox | None = None,
    regions: tuple[vision.RegionSpec, ...] = vision.TH08_REGIONS,
    template_dir: Path | None = None,
    run_dir: Path | None = None,
    shard_size: int = 128,
    compress_level: int = 6,
    allow_append: bool = False,
    freeze_limit: int = 5,
    verbose: bool = True,
) -> dict:
    template = template_dir or (ROOT / "third_party/th08/build/play")
    work = run_dir or (ROOT / "third_party/th08/build/rl/vision-session")
    out_dir = Path(out_dir)
    box = box or vision.PlayfieldBox(0, 0, vision.NATIVE_W, vision.NATIVE_H)
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
    playback_started: float | None = None
    last_game_frame: int | None = None
    last_stage_for_span: int | None = None
    game_frames_total = 0
    stability = FrameStability(freeze_limit)
    window_hidden_events = 0
    report = {
        "replay": str(replay),
        "frames": 0,
        "stages_seen": [],
        "ended_reason": "",
        "elapsed_seconds": 0.0,
        "box": [box.x, box.y, box.w, box.h],
        "frame_shape": [vision.NATIVE_H, vision.NATIVE_W, vision.NATIVE_CHANNELS],
        "every": int(every),
        "window_hidden_events": 0,
    }
    transient = {
        spec.STATE_GAME_MANAGER_REINIT,
        spec.STATE_RESTART_FROM_BEGINNING,
        12,
    }
    try:
        winapi.keep_awake(True)
        session.start()
        menu.enter_replay(session, stage=int(stage), timeout=60.0)
        reader = session.reader
        assert reader is not None
        session.keyboard.release_all()
        last_key: tuple[int, int] | None = None
        next_target: int | None = None
        last_guard = 0.0
        last_progress = time.time()
        while time.time() - started < max_seconds:
            # A minimised / cloaked window cannot be captured correctly, and the
            # failure mode is silent: record it and stop instead of writing
            # thousands of identical samples.
            if session.hwnd and not winapi.window_visible(session.hwnd):
                window_hidden_events += 1
                report["window_hidden_events"] = window_hidden_events
                if window_hidden_events >= 3:
                    report["ended_reason"] = "window hidden (minimised or cloaked)"
                    break
                time.sleep(0.2)
                continue
            window_hidden_events = 0
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
                next_target = None  # stage boundary resets the stride clock
            # Stride by absolute target rather than by ``delta % every``: with the
            # modulo test, a sample that is taken a few milliseconds late makes the
            # next delta a non-multiple of ``every``, so that frame is dropped and
            # the following one is taken - silently halving the rate.  Targeting the
            # next frame number keeps the cadence even under jitter.
            if next_target is not None and key[1] < next_target:
                continue
            next_target = key[1] + every
            last_key = key
            last_progress = time.time()

            field = reader.read_bullets()
            observation = obs.encode(state, field)
            try:
                # Discrete, per-region recording: playfield and each HUD block are
                # cropped at native resolution and encoded separately, so the HUD
                # digits stay legible instead of being lost in a global downscale.
                crops = vision.extract_regions(session.grab(), regions)
                blobs = {
                    name: vision.encode_png(image, compress_level=compress_level)
                    for name, image in crops.items()
                }
            except Exception as exc:  # a transient BitBlt/PNG failure must not kill a long capture
                if verbose:
                    print(f"[vision-capture] frame skipped: {exc!r}", flush=True)
                continue
            action = spec.buttons_to_action(state.gui_input_current)
            # Freeze guard: the recorder only reaches this line when the game frame
            # number advanced, so a repeated picture means stale pixels, not a still
            # scene.  Hash the playfield only - it is the region that carries the
            # information and hashing all three costs more for no extra signal.
            stability.update(hashlib.blake2b(blobs["playfield"], digest_size=16).digest())
            try:
                stability.check()
            except FrozenCapture as exc:
                report["ended_reason"] = f"capture frozen: {exc}"
                if verbose:
                    print(f"[vision-capture] ABORT {exc}", flush=True)
                break
            writer.add(
                blobs,
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
                native_hw={name: (image.shape[0], image.shape[1]) for name, image in crops.items()},
            )
            report["frames"] += 1
            if state.stage not in report["stages_seen"]:
                # Record the first stage too: v1.0.3 only appended on a stage *change*,
                # so the opening stage never appeared in the report.
                report["stages_seen"].append(state.stage)
            if playback_started is None:
                playback_started = time.time()
            if last_game_frame is not None and key[0] == last_stage_for_span:
                advanced = (key[1] - last_game_frame) & 0xFFFFFFFF
                if 0 < advanced < 600:
                    game_frames_total += advanced
            last_stage_for_span = key[0]
            last_game_frame = key[1]
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
        winapi.keep_awake(False)
        report.update(stability.summary())
        report["elapsed_seconds"] = round(time.time() - started, 1)
        # Rate metrics split two things the earlier report conflated: the wall
        # clock includes boot + menu walk (several seconds with zero samples), and
        # the *effective* rate is samples per second of game time actually played.
        if playback_started is not None:
            playback_seconds = max(1e-6, time.time() - playback_started)
            report["playback_seconds"] = round(playback_seconds, 1)
            report["samples_per_playback_second"] = round(report["frames"] / playback_seconds, 2)
        # ``gameplay_frame`` resets at every stage boundary, so the meaningful span
        # is the sum of within-stage deltas (``game_frames_total``), not
        # last-minus-first.  (v1.0.3 had a dead ``first_game_frame`` guard here that
        # meant the field was never written at all - see D7.)
        if game_frames_total > 0:
            report["game_frames_span"] = int(game_frames_total)
            report["samples_per_game_second"] = round(
                report["frames"] / (game_frames_total / 60.0), 2
            )
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
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--compress-level", type=int, default=6, help="PNG compress level (1=fast, 9=small)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--freeze-limit",
        type=int,
        default=5,
        help="abort after N identical frames in a row (0 disables; default 5)",
    )
    args = parser.parse_args(argv)
    report = capture_vision(
        Path(args.replay),
        Path(args.out),
        tag=args.tag,
        every=args.every,
        max_seconds=args.max_seconds,
        stage=None if args.stage < 0 else args.stage,
        shard_size=args.shard_size,
        compress_level=args.compress_level,
        allow_append=args.overwrite,
        freeze_limit=args.freeze_limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
