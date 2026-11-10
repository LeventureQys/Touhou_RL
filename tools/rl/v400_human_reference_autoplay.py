"""EXP-P0-046: measure per-stage human reference from *real replay playback*.

The numbers this tool reports (gameplay frames, deaths, bombs used) come from the
engine's own ``STATE``/``DEBUG`` counters while a retail ``.rpy`` file is played
back by the real TH08 engine through its developer-autoplay entry point
(``TH08_AUTOPLAY_REPLAY`` / ``TH08_AUTOPLAY_STAGE``).  Nothing is derived from the
replay file's ``inputRecords``: input-record counts are stream extents, not
gameplay frames, and they carry no death/bomb semantics.

Playback input path (verified in ``third_party/th08-rl`` sources):
``ReplayManager::PlaybackInputAndFps`` writes the recorded input into
``g_GuiMessageInputCurrent``, which is what ``Player`` reads, so the harness
``STEP`` mask is irrelevant here and ``STEP 0`` is the correct pump command.

Field indices (``touhou_rl.v400.gym.env``):
``f0`` engine frame, ``f1`` stage, ``f2`` replay playback frame counter,
``f3`` gameplay frame counter, ``f6`` deaths (f32 bits), ``f7`` lives remaining
(f32 bits), ``f8`` bombs remaining (f32 bits), ``f9`` power (f32 bits),
``f13`` player state (recorded in bomb events for auditing).

Reference semantics (all read from engine counters, none inferred from the
replay file):

* ``gameplay_frames`` — final ``gameplayFrameCounter`` (f3).  It increments once
  per gameplay logic frame and starts at 0, so its final value is the stage's
  logic-frame length.  Replay playback fast-forwards skippable dialogue, so this
  can exceed the number of *rendered* engine frames (reported separately).
* ``replay_input_frames`` — final ``ReplayManager::frameCounter`` (f2), i.e. the
  logic frames driven by recorded input; it stops when the stage-clear sequence
  starts, which is why it is smaller than ``gameplay_frames``.
* ``deaths`` — sum of increases of ``globals.deaths`` (f6).
* ``bombs_used`` — number of bomb activations: one per frame whose
  ``bombsRemaining`` (f8) drops, excluding the respawn frame's death penalty.
  This matches the engine's own ``bombsUsed`` counter: a normal bomb costs one
  stock point, a deathbomb costs two (or all remaining) but is still one use.
* ``bombs_stock_spent`` — stock points actually spent (1 per normal bomb, 2 per
  deathbomb), so ``bombs_stock_spent > bombs_used`` reveals deathbombs.
* ``bombs_reset_on_death`` — stock restored/removed by
  ``Player::UpdateDeathAndRespawn`` on a lost-life frame; never a player bomb.

Fail-closed rules: the tool refuses to report a reference unless the engine
really entered replay playback (``DEBUG isReplay=1``), kept ``curState=2``
gameplay, and the stage ended naturally (game over or stage transition) inside
the bounded frame budget.  Otherwise the JSON says ``unavailable`` and the exit
code is 3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from touhou_rl.v400.gym.env import (  # noqa: E402
    F_BOMBS, F_DEATHS, F_GRAZE, F_LIVES, F_POWER, F_SCORE, _f32_bits,
)

try:  # Windows-only launcher; importing it is harmless on POSIX but unused there.
    from tools.rl.v400_wsl_env import WslHarnessClient  # noqa: E402
except Exception:  # pragma: no cover - the module only needs it for the WSL path
    WslHarnessClient = None  # type: ignore[assignment]

#: ``ReplayManager::frameCounter``: logic frames the playback consumed this stage.
F_REPLAY_FRAMES = 2

#: ``g_Player.playerState`` (0 alive, 1 spawning, 2 dying, 3 invulnerable).
F_PLAYER_STATE = 13

#: Fields STATE reports as float bits instead of plain integers.
FLOAT_FIELDS = (F_DEATHS, F_LIVES, F_BOMBS, F_POWER, 10, 11, 12)

#: ``SupervisorState_GameManager``.
STATE_GAMEPLAY = 2

#: ``SupervisorState_FinishReplay``.
STATE_FINISH_REPLAY = 7


class AutoplayUnavailable(RuntimeError):
    """The real replay playback path could not be proven inside the bound."""


def decode_state(fields: dict[str, int]) -> dict[str, float | int]:
    """Decode the 25 tracked STATE fields (float fields reinterpreted).

    ``STATE`` prints ``f6``..``f12`` as 8-digit hex float bits and the other
    fields as decimal, so the raw reply has to be split before decoding.
    """
    decoded: dict[str, float | int] = {}
    for index in range(25):
        key = f"f{index}"
        if key not in fields:
            continue
        value = int(fields[key])
        decoded[key] = _f32_bits(value) if index in FLOAT_FIELDS else value
    return decoded


def parse_reply(line: str) -> dict[str, str]:
    """``OK a=1 b=2`` -> ``{"_verb": "OK", "a": "1", "b": "2"}``."""
    parts = line.split()
    fields = {"_verb": parts[0] if parts else ""}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    return fields


def read_state_fields(client) -> dict[str, int]:
    """Ask the engine for STATE and return ``f0``..``f24`` as integers.

    ``HarnessClient.state()`` cannot be used: the real engine prints the float
    fields as ``%08x`` bit patterns, which its ``int(value)`` rejects.
    """
    reply = client.request("STATE")
    fields = parse_reply(reply)
    if fields.get("state") == "undefined":
        return {}
    out: dict[str, int] = {}
    for key, value in fields.items():
        if not key.startswith("f") or not key[1:].isdigit():
            continue
        index = int(key[1:])
        out[key] = int(value, 16) if index in FLOAT_FIELDS else int(value, 10)
    return out


def _as_float(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise AutoplayUnavailable(f"expected numeric state value, got {value!r}")
    return float(value)


def _tiny(value: float) -> float | int:
    """Collapse float32 noise so JSON shows 2.0 as 2 rather than 1.9999999."""

    rounded = round(value)
    if abs(value - rounded) <= 1e-4:
        return int(rounded)
    return value


def aggregate(samples: list[dict], *, stage: int) -> dict:
    """Turn an ordered per-frame sample list into one stage's reference row."""
    in_stage = [s for s in samples if s["state"] and s["state"].get("f1") == stage]
    gameplay = [s for s in in_stage if s["debug"] and s["debug"].get("curState") == str(STATE_GAMEPLAY)]
    if not in_stage:
        raise AutoplayUnavailable(f"no STATE sample reported stage={stage}; cannot aggregate")
    if not gameplay:
        raise AutoplayUnavailable(
            f"stage={stage} never reached gameplay (curState={STATE_GAMEPLAY}); "
            f"observed states={sorted({s['debug'].get('curState') for s in in_stage if s['debug']})}"
        )

    replay_flags = {s["debug"].get("isReplay") for s in gameplay if s["debug"]}
    if replay_flags != {"1"}:
        raise AutoplayUnavailable(
            f"gameplay ran without replay playback (DEBUG isReplay={sorted(map(str, replay_flags))}); "
            "refusing to report it as a human reference"
        )

    frames = [int(s["state"]["f3"]) for s in gameplay]
    if min(frames) != 0:
        raise AutoplayUnavailable(
            f"no gameplay sample reported the stage's frame zero (min f3={min(frames)}); "
            "the frame count would be biased"
        )
    if sum(1 for a, b in zip(frames, frames[1:]) if b < a):
        raise AutoplayUnavailable(
            "gameplayFrameCounter went backwards inside one stage; the sample window crossed a reset"
        )
    # The stage's globals exist (zero-filled) before the replay's stage data is
    # loaded, so deaths/bombs read 0 until BeginPlaybackStage lands the real
    # starting resources.  The last ``f3 == 0`` sample is the last frame before
    # gameplay logic starts and already carries those resources: it is the
    # baseline, and only samples from it onwards describe the stage.
    baseline_index = max(index for index, frame in enumerate(frames) if frame == 0)
    series = gameplay[baseline_index:]
    start = series[0]
    positive = [s for s in series if int(s["state"]["f3"]) > 0]
    if not positive:
        raise AutoplayUnavailable(
            f"stage={stage} never advanced gameplayFrameCounter past 0 inside the measured window"
        )
    first = positive[0]
    last = series[-1]

    events: list[dict] = []
    deaths_used = 0.0
    bombs_used = 0.0
    bombs_stock_spent = 0.0
    bombs_gained = 0.0
    bombs_reset_on_death = 0.0
    for prev, cur in zip(series, series[1:]):
        before_deaths = _as_float(prev["state"][f"f{F_DEATHS}"])
        after_deaths = _as_float(cur["state"][f"f{F_DEATHS}"])
        if after_deaths > before_deaths:
            deaths_used += after_deaths - before_deaths
            events.append({
                "kind": "death",
                "engine_frame": cur["engine_frame"],
                "gameplay_frame": cur["state"]["f3"],
                "deaths_before": _tiny(before_deaths),
                "deaths_after": _tiny(after_deaths),
            })
        before_lives = _as_float(prev["state"][f"f{F_LIVES}"])
        after_lives = _as_float(cur["state"][f"f{F_LIVES}"])
        before_bombs = _as_float(prev["state"][f"f{F_BOMBS}"])
        after_bombs = _as_float(cur["state"][f"f{F_BOMBS}"])
        if after_bombs < before_bombs:
            if after_lives < before_lives:
                # Player::UpdateDeathAndRespawn(): the respawn frame does
                # AddLives(-1) and SetBombCount(primaryShtFile->initialBombCount)
                # in the same update, so a stock drop on a lost-life frame is
                # the death penalty, not a bomb the player pressed.
                bombs_reset_on_death += before_bombs - after_bombs
                events.append({
                    "kind": "death_bomb_reset",
                    "engine_frame": cur["engine_frame"],
                    "gameplay_frame": cur["state"]["f3"],
                    "bombs_before": _tiny(before_bombs),
                    "bombs_after": _tiny(after_bombs),
                    "lives_before": _tiny(before_lives),
                    "lives_after": _tiny(after_lives),
                })
            else:
                # One decrease event == one bomb activation (normal bomb -1,
                # deathbomb -2 or all remaining), which is exactly what the
                # engine's own bombsUsed counter increments per activation.
                bombs_used += 1
                bombs_stock_spent += before_bombs - after_bombs
                events.append({
                    "kind": "bomb",
                    "engine_frame": cur["engine_frame"],
                    "gameplay_frame": cur["state"]["f3"],
                    "bombs_before": _tiny(before_bombs),
                    "bombs_after": _tiny(after_bombs),
                    "player_state": cur["state"].get(f"f{F_PLAYER_STATE}"),
                })
        elif after_bombs > before_bombs:
            bombs_gained += after_bombs - before_bombs
            events.append({
                "kind": "bomb_gained",
                "engine_frame": cur["engine_frame"],
                "gameplay_frame": cur["state"]["f3"],
                "bombs_before": _tiny(before_bombs),
                "bombs_after": _tiny(after_bombs),
            })

    # ``gameplayFrameCounter`` is incremented once at the end of every gameplay
    # logic frame and starts at 0, so its final value *is* the stage's logic
    # frame count.  It can exceed the rendered engine frames because replay
    # playback fast-forwards skippable dialogue with chained calc restarts, so
    # the two counters are reported separately instead of being forced equal.
    gameplay_frames = int(last["state"]["f3"])
    if gameplay_frames < 1:
        raise AutoplayUnavailable(f"stage={stage} reports {gameplay_frames} gameplay frames")
    engine_start = int(first["engine_frame"]) - (int(first["state"]["f3"]) - 1)
    engine_frames = int(last["engine_frame"]) - engine_start + 1
    replay_frames = int(last["state"][f"f{F_REPLAY_FRAMES}"])
    # ``ReplayManager::frameCounter`` stops when the stage-clear sequence starts,
    # so it is the number of input-driven gameplay frames; the gameplay counter
    # keeps running through the clear sequence.  A large gap means the measured
    # window crossed something other than a clear sequence.
    clear_frames = gameplay_frames - replay_frames
    if clear_frames < 0 or clear_frames > 600:
        raise AutoplayUnavailable(
            f"stage={stage} replay playback counter ({replay_frames}) and gameplay counter "
            f"({gameplay_frames}) disagree by {clear_frames} frames; "
            "the stage window is inconsistent"
        )

    deaths_start = _as_float(start["state"][f"f{F_DEATHS}"])
    deaths_end = _as_float(last["state"][f"f{F_DEATHS}"])
    bombs_start = _as_float(start["state"][f"f{F_BOMBS}"])
    bombs_end = _as_float(last["state"][f"f{F_BOMBS}"])
    balance = bombs_start - bombs_end + bombs_gained
    if abs(balance - (bombs_stock_spent + bombs_reset_on_death)) > 1e-3:
        raise AutoplayUnavailable(
            f"bomb accounting disagrees: sum of decreases="
            f"{bombs_stock_spent + bombs_reset_on_death} vs "
            f"start-end+gained={balance}; a decrease was missed between samples"
        )
    if abs((deaths_end - deaths_start) - deaths_used) > 1e-3:
        raise AutoplayUnavailable(
            f"death accounting disagrees: sum of increases={deaths_used} vs "
            f"end-start={deaths_end - deaths_start}"
        )

    return {
        "stage_index": stage,
        "difficulty": int(start["debug"].get("difficulty", "-1")),
        "shot": int(start["debug"].get("shot", "-1")),
        "gameplay_frames": gameplay_frames,
        "replay_input_frames": replay_frames,
        "stage_clear_frames": clear_frames,
        "engine_frames": engine_frames,
        "dialogue_fast_forward_frames": gameplay_frames - engine_frames,
        "engine_frame_first_gameplay": engine_start,
        "engine_frame_last_gameplay": int(last["engine_frame"]),
        "baseline_engine_frame": int(start["engine_frame"]),
        "deaths": _tiny(deaths_used),
        "deaths_counter_start": _tiny(deaths_start),
        "deaths_counter_end": _tiny(deaths_end),
        "bombs_used": _tiny(bombs_used),
        "bombs_stock_spent": _tiny(bombs_stock_spent),
        "bombs_gained": _tiny(bombs_gained),
        "bombs_reset_on_death": _tiny(bombs_reset_on_death),
        "bombs_start": _tiny(bombs_start),
        "bombs_end": _tiny(bombs_end),
        "lives_start": _tiny(_as_float(start["state"][f"f{F_LIVES}"])),
        "lives_end": _tiny(_as_float(last["state"][f"f{F_LIVES}"])),
        "score_end": int(last["state"][f"f{F_SCORE}"]),
        "graze_end": int(last["state"][f"f{F_GRAZE}"]),
        "events": events,
        "samples_in_stage": len(in_stage),
        "gameplay_samples": len(gameplay),
        "measured_samples": len(series),
    }


def _sample(client, *, engine_frame: int, note: str = "") -> dict:
    state = decode_state(read_state_fields(client))
    debug = client.debug()
    return {
        "engine_frame": engine_frame,
        "state": state,
        "debug": {key: str(value) for key, value in debug.items()},
        "note": note,
    }


def play_stage(
    client,
    *,
    stage: int,
    poll_frames: int = 2,
    max_frames: int = 40000,
    deadline_seconds: float = 240.0,
    enter_gameplay_seconds: float = 30.0,
    clock=time.monotonic,
) -> dict:
    """Pump the real engine through one replay stage and return its measured row."""
    if poll_frames < 1:
        raise ValueError("poll_frames must be positive")
    if max_frames < 1:
        raise ValueError("max_frames must be positive")

    hello = {key: str(value) for key, value in client.hello().items()}
    if hello.get("harness") != "1":
        raise AutoplayUnavailable(f"HELLO did not report a harness: {hello}")
    begun = clock()
    samples: list[dict] = []
    end_reason = ""
    engine_frames = 0
    trace: list[dict] = []
    saw_gameplay = False

    while engine_frames < max_frames:
        elapsed = clock() - begun
        if elapsed > deadline_seconds:
            end_reason = "deadline"
            break
        budget = min(poll_frames, max_frames - engine_frames)
        reply = client.step(0, budget)
        advanced = int(reply.get("advanced", "0"))
        engine_frames += advanced
        sample = _sample(client, engine_frame=engine_frames)
        samples.append(sample)
        debug = sample["debug"]
        state = sample["state"]
        if not saw_gameplay and state and debug.get("curState") == str(STATE_GAMEPLAY):
            saw_gameplay = True
        if len(trace) < 40:
            trace.append({
                "engine_frame": engine_frames,
                "curState": debug.get("curState"),
                "setupState": debug.get("setupState"),
                "stage": debug.get("stage"),
                "isReplay": debug.get("isReplay"),
                "f1": state.get("f1"),
                "f3": state.get("f3"),
            })
        if reply.get("exit") == "1":
            # The chain asked to exit (game over in replay playback). One more
            # STATE gives the exact final gameplay counter; it may be refused
            # once the globals are gone, so the last full sample stays authoritative.
            samples.append(_sample(client, engine_frame=engine_frames, note="after_exit"))
            end_reason = "game_over" if reply.get("game_over") == "1" else "engine_exit"
            break
        if advanced < budget:
            end_reason = "short_step"
            break
        if not saw_gameplay:
            if elapsed > enter_gameplay_seconds:
                raise AutoplayUnavailable(
                    "replay playback did not reach gameplay within "
                    f"{enter_gameplay_seconds:.0f}s ({len(samples)} pumps, {engine_frames} frames): "
                    f"last DEBUG={debug}"
                )
            continue
        if state and state.get("f1") != stage:
            end_reason = "next_stage"
            break
        if debug.get("curState") != str(STATE_GAMEPLAY):
            end_reason = "finish_replay" if debug.get("curState") == str(STATE_FINISH_REPLAY) \
                else f"curState_{debug.get('curState')}"
            break

    try:
        row = aggregate(samples, stage=stage)
    except AutoplayUnavailable as exc:
        # Keep the raw per-frame evidence reachable for the failure report.
        exc.samples = samples  # type: ignore[attr-defined]
        exc.trace = trace  # type: ignore[attr-defined]
        raise
    row["end_reason"] = end_reason or "frame_budget"
    row["stage_cleared"] = end_reason in ("next_stage", "curState_3", "finish_replay")
    row["complete"] = end_reason in (
        "game_over", "engine_exit", "next_stage", "curState_3", "finish_replay",
    )
    engine_frames_pumped = engine_frames
    return {
        "status": "measured" if row["complete"] else "incomplete",
        "kind": "real-engine-replay-playback",
        "hello": hello,
        "stage": row,
        "samples": len(samples),
        "trace": trace,
        "wall_seconds": round(clock() - begun, 3),
        "frames_per_second": round(engine_frames_pumped / max(1e-9, clock() - begun), 3),
        "engine_frames_pumped": engine_frames_pumped,
        "raw_samples": samples,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_wsl_path(path: str | Path) -> str:
    """Map ``D:\\...`` to ``/mnt/d/...`` for the WSL side of the pipe."""
    text = str(path)
    drive, tail = os.path.splitdrive(text)
    if len(drive) == 2 and drive[1] == ":":
        return "/mnt/" + drive[0].lower() + tail.replace("\\", "/")
    return text.replace("\\", "/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True, help="Windows-side .rpy path")
    parser.add_argument("--stage", type=int, default=0, help="TH08 stage slot (0..8) to start playback at")
    parser.add_argument("--distro", default="Ubuntu")
    parser.add_argument("--engine", default="/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern")
    parser.add_argument("--data-dir", default="/root/v400-data/game")
    parser.add_argument("--role", default="shadow", choices=("shadow", "env"))
    parser.add_argument("--poll-frames", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=40000)
    parser.add_argument("--deadline", type=float, default=240.0)
    parser.add_argument("--enter-gameplay-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path, help="JSON evidence path")
    parser.add_argument("--samples-output", type=Path,
                        help="JSONL dump of the raw per-frame samples (failure evidence)")
    parser.add_argument("--stderr", type=Path, help="capture the engine's stderr here")
    args = parser.parse_args(argv)

    report: dict = {
        "status": "unavailable",
        "kind": "real-engine-replay-playback",
        "replay": {"path": str(args.replay)},
        "engine": {"binary": args.engine, "data_dir": args.data_dir, "role": args.role,
                   "process": f"wsl.exe -d {args.distro}"},
        "human_reference_created": False,
    }
    client = None
    raw_samples = None
    try:
        if not args.replay.is_file():
            raise AutoplayUnavailable(f"replay not found: {args.replay}")
        if os.name == "nt" and WslHarnessClient is None:
            raise AutoplayUnavailable("WSL launcher unavailable on this host")
        report["replay"].update({
            "sha256": _sha256(args.replay),
            "size_bytes": args.replay.stat().st_size,
            "requested_stage": args.stage,
            "wsl_path": to_wsl_path(args.replay),
        })
        if os.name == "nt":
            client = ReplayAutoplayClient(
                args.engine, args.data_dir,
                replay=report["replay"]["wsl_path"], stage=args.stage,
                role=args.role, distro=args.distro, stderr_path=args.stderr,
            )
            client.start()
        else:  # already inside WSL: run the ELF directly
            client = LocalAutoplayClient(
                args.engine, args.data_dir,
                replay=report["replay"]["wsl_path"], stage=args.stage, role=args.role,
                stderr_path=args.stderr,
            )
            client.start()
        measured = play_stage(
            client, stage=args.stage, poll_frames=args.poll_frames,
            max_frames=args.max_frames, deadline_seconds=args.deadline,
            enter_gameplay_seconds=args.enter_gameplay_seconds,
        )
        raw_samples = measured.pop("raw_samples", None)
        report.update(measured)
        report["human_reference_created"] = measured["status"] == "measured"
    except (AutoplayUnavailable, OSError, RuntimeError, ValueError) as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
        raw_samples = getattr(exc, "samples", None)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # pragma: no cover - close failures are reported, not raised
                report.setdefault("warnings", []).append(f"close: {exc}")
            if getattr(client, "stderr_file", None) is not None:
                client.stderr_file.close()
        if args.stderr and args.stderr.is_file():
            text = args.stderr.read_bytes().decode("utf-8", "replace")
            report.setdefault("engine_stderr", text[-4000:])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.samples_output and raw_samples:
        args.samples_output.parent.mkdir(parents=True, exist_ok=True)
        with args.samples_output.open("w", encoding="utf-8") as handle:
            for sample in raw_samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        report["samples_output"] = str(args.samples_output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("human_reference_created") else 3


class ReplayAutoplayClient:  # pragma: no cover - exercised only on a real Windows host
    """``WslHarnessClient`` that passes the autoplay environment into WSL."""

    def __init__(self, binary, data_dir, *, replay, stage, role="shadow", distro="Ubuntu",
                 timeout=300.0, stderr_path=None):
        self._base = WslHarnessClient(binary, data_dir, distro=distro, timeout=timeout)
        self.wsl_binary = binary
        self.wsl_data_dir = data_dir
        self.replay = replay
        self.stage = int(stage)
        self.role = role
        self.distro = distro
        self.stderr_path = Path(stderr_path) if stderr_path else None
        self.stderr_file = None

    def __getattr__(self, name):
        return getattr(self._base, name)

    def start(self) -> None:
        import subprocess

        if self._base._proc is not None:
            raise RuntimeError("client already started")
        argv = [
            "wsl.exe", "-d", self.distro, "--", "env", "-u", "DISPLAY",
            "SDL_AUDIODRIVER=dummy",
            f"TH08_AUTOPLAY_REPLAY={self.replay}",
            f"TH08_AUTOPLAY_STAGE={self.stage}",
            self.wsl_binary, "--data-dir", self.wsl_data_dir,
            "--rl-control", "0,1", "--rl-role", self.role,
        ]
        self.command = argv
        if self.stderr_path is not None:
            self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self.stderr_file = self.stderr_path.open("wb")
        self._base._proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr_file if self.stderr_file is not None else subprocess.DEVNULL,
            env=self._base._env, bufsize=0,
        )
        self._base._start_pipe_reader()

    def close(self) -> None:
        self._base.close()
        if self.stderr_file is not None:
            self.stderr_file.close()
            self.stderr_file = None


class LocalAutoplayClient:  # pragma: no cover - exercised inside WSL only
    """Same protocol, but the child is the ELF directly (already inside WSL)."""

    def __init__(self, binary, data_dir, *, replay, stage, role="shadow",
                 timeout=300.0, stderr_path=None):
        from touhou_rl.v400.gym.protocol import HarnessClient

        self._base = HarnessClient(binary, data_dir, timeout=timeout)
        self.binary = binary
        self.data_dir = data_dir
        self.replay = replay
        self.stage = int(stage)
        self.role = role
        self.stderr_path = Path(stderr_path) if stderr_path else None
        self.stderr_file = None

    def __getattr__(self, name):
        return getattr(self._base, name)

    def start(self) -> None:
        import subprocess

        if self._base._proc is not None:
            raise RuntimeError("client already started")
        argv = [
            str(self.binary), "--data-dir", str(self.data_dir),
            "--rl-control", "0,1", "--rl-role", self.role,
        ]
        env = dict(self._base._env)
        env.pop("DISPLAY", None)
        env.pop("WAYLAND_DISPLAY", None)
        env["TH08_AUTOPLAY_REPLAY"] = str(self.replay)
        env["TH08_AUTOPLAY_STAGE"] = str(self.stage)
        self.command = argv
        if self.stderr_path is not None:
            self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
            self.stderr_file = self.stderr_path.open("wb")
        self._base._proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr_file if self.stderr_file is not None else subprocess.DEVNULL,
            env=env, bufsize=0,
        )
        if os.name == "nt":
            self._base._start_pipe_reader()

    def close(self) -> None:
        self._base.close()
        if self.stderr_file is not None:
            self.stderr_file.close()
            self.stderr_file = None


if __name__ == "__main__":
    raise SystemExit(main())
