"""D0 dual channel (EXP-P1-014): shadow labels + env pixels, aligned by ``HASH``.

The P3 hard blocker (EXP-P1-013 §6.2) is a *role* split, not a format split:

* the teacher must run as ``--rl-role shadow`` (``PARK`` refuses anything else,
  ``fork.cpp:261-269``), and shadow is a CPU-only null device that never creates
  SDL/GL (``main.cpp:210-244``, ``d3d8_compat.cpp:634-648``), so it publishes no
  pixels at all;
* the ``env`` role renders (``READBACK 1`` + ``SHM`` RGBA) but cannot ``PARK``.

So this tool runs the two channels as **two independent processes over the same
``RESET`` and the same per-frame mask sequence**, and proves they are the same
trajectory with ``HASH`` at every decision boundary (before ``decide`` and after
the 4-frame commit).  EXP-P0-049 (5000 frames, zero differences) and EXP-P0-019
(167 checkpoints) already established that shadow and env are frame-identical
under identical input; this tool turns that into a per-decision gate.

Phases (all under ``--work-dir``, resumable one phase at a time)::

    labels   channel A: ``v400_teacher.py --labels-only`` in the shadow role,
             writing ``labels.jsonl`` (scores[18]/verified_mask/chosen_plan/
             step_frames + hash_pre/hash_post per decision).  No pixels.
    pixels   channel B: one env-role process, ``SHM``+``READBACK 1``, same
             ``RESET``, one ``STEP <mask> 1`` per frame, pixels for ``d-4..d-1``
             of every decision, and the *same* HASH checkpoints.
    assemble write the EXP-P1-013 shard from A's labels + B's pixels, but only
             after every checkpoint matched.  Then load it back through
             ``touhou_rl.v400.student.dataset.load_teacher_samples``.

**Fail-closed.**  A single checkpoint that differs (hash or frame) aborts with
``EXIT_ALIGN=3``, writes ``alignment.json`` with the first mismatch and both
sides, and never creates a shard directory.  A blank readback (all-zero frame)
is refused by the same ``require_rendered`` gate EXP-P1-013 uses, so an env
process that somehow did not render cannot be passed off as observations.

Scale-up (EXP-P1-018) needs three things the 60-decision smoke did not:

* **resume** — a finished shard (``<out-dir>/shards/<key>/shard.json`` that still
  loads) is skipped on rerun, so a long batch can be restarted with the *same
  command* without rewriting 135 MB shards; ``--force-rerun`` opts out;
* **parallel** — ``--engine-limit N`` (with an optional ``--engine-pattern`` that
  matches only this batch's engines) waits for ``N`` free engine slots instead of
  demanding an empty machine, so two seeds can be produced side by side while a
  third engine belongs to somebody else;
* **early stop** — ``--allow-early-stop`` forwards ``--max-frames``/
  ``--stop-on-death`` to channel A and then accepts *fewer* decisions than
  requested, recording the real count and the teacher's ``stop_reason`` (a stage
  clear or a death ends the segment honestly instead of pretending it was cut).

The assemble phase takes a cross-process lock on ``<out-dir>/.assemble.lock``
because ``manifest.json`` is a read-modify-write file: two channels finalising at
the same time must not lose each other's shard entry.

**Observation version (T2 closeout / D-14).**  ``--obs`` selects the planes the
shard stores and defaults to ``v1``, so every existing command and artifact is
unchanged:

* ``--obs v1`` (default) — ``obs/play64.npy`` (4,64,64) + ``obs/hud.npy`` (56,160),
  obs-v1 digest, exactly as before;
* ``--obs v3`` — ``obs/play96x112.npy`` (4,112,96) + ``obs/hud.npy`` (56,160) +
  ``obs/act_hist.npy`` (8,18), obs-v3 digest
  (``touhou_rl/v400/student/obs_spec_v3.py``).  The 96x112 plane is EXP-P3-010's
  input B byte for byte; ``act_hist`` is rebuilt during assemble from the same
  aligned label rows with obs-v2's frozen rule (``chosen_plan.first_action``,
  cleared at the shard start), and ``shard.json`` records
  ``obs_spec_version="obs-v3"``, its digest and ``obs_version="v3"``.

Run inside WSL (the binary is a Linux ELF)::

    env -u DISPLAY xvfb-run -a /root/v400-venv/bin/python \
        tools/rl/v400_d0_dual.py --decisions 60 --stage 0 --difficulty 1 --seed 1 \
        --work-dir .tmp_v400_d0_dual --out-dir .tmp_v400_d0_dual/d0

    # obs-v3 (T3's D1 format); one instance, its own data root so it never waits
    # on another batch's engines:
    env -u DISPLAY xvfb-run -a /root/v400-venv/bin/python tools/rl/v400_d0_dual.py \
        --phase all --decisions 200 --stage 0 --difficulty 1 --seed 1 --obs v3 \
        --work-dir .tmp_v400_obsv3/work --out-dir .tmp_v400_obsv3/d0 \
        --data-root /root/v400-data/obsv3 \
        --engine-limit 4 --engine-pattern 'th08-modern --data-dir /root/v400-data/obsv3'
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):  # allow ``python tools/rl/v400_d0_dual.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from touhou_rl.v400.student import dataset, obs_spec, obs_spec_v2  # noqa: E402
from touhou_rl.v400.teacher import config as teacher_config  # noqa: E402
from touhou_rl.v400.teacher.engine import EXPECTED_ENGINE_SHA256, sha256_file  # noqa: E402
from touhou_rl.v400.teacher.obs import (  # noqa: E402
    ALIGNMENT,
    OBS_V1,
    OBS_V3,
    OBS_VERSIONS,
    DecisionObservationBuffer,
    ObservationError,
    RenderFrame,
    TeacherObsShardWriter,
    check_frozen_spec,
    check_obs_version,
    obs_version_spec,
    require_rendered,
)

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_FAIL_CLOSED = 3
EXIT_ALIGN = 4
EXIT_ENGINE = 5
EXIT_BUSY = 9

#: 4 logic frames per teacher decision (``replan_decisions 2 x decision_frames 2``).
DEFAULT_FRAMES_PER_DECISION = 4
HASH_RE = re.compile(r"^[0-9a-f]{16}$")

#: Data files the engine writes; the two archives stay read-only symlinks.
LINKED_FILES = ("th08.dat", "thbgm.dat")
COPIED_FILES = ("th08.cfg", "score.dat")


class DualError(RuntimeError):
    """A precondition, a protocol reply or the channel contract failed."""


class AlignError(DualError):
    """A HASH checkpoint disagrees; the two channels are not the same trajectory."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _tokens(reply: str) -> dict[str, str]:
    return {part.split("=", 1)[0]: part.split("=", 1)[1]
            for part in reply.split()[1:] if "=" in part}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return _sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_data_dir(template: Path, target: Path) -> dict[str, Any]:
    """A fresh data dir: read-only archives symlinked, writable files copied."""

    if not template.is_dir():
        raise DualError(f"data template {template} is not a directory")
    if target.exists():
        return {"data_dir": str(target), "reused": True}
    target.mkdir(parents=True)
    for name in LINKED_FILES:
        source = template / name
        if not source.exists():
            raise DualError(f"data template lacks {source}")
        (target / name).symlink_to(source)
    for name in COPIED_FILES:
        source = template / name
        if source.exists():
            shutil.copy2(source, target / name)
    return {"data_dir": str(target), "reused": False,
            "linked": list(LINKED_FILES), "copied": [n for n in COPIED_FILES
                                                    if (target / n).exists()]}


def _is_engine_process(pid: int) -> bool:
    """``/proc/<pid>/comm`` == ``th08-modern`` (the only self-match-free test).

    ``pgrep -f <pattern>`` matches *any* process whose argv contains the pattern
    string — including this Python driver, its ``xvfb-run`` wrapper and the
    shadow teacher, because they carry ``--engine-pattern``/``--data-dir`` on
    their command lines.  Filtering on the executable name is what makes an
    engine slot count engines instead of our own bookkeeping.
    """

    try:
        with open(f"/proc/{int(pid)}/comm", "r", encoding="utf-8") as handle:
            return handle.read().strip() == "th08-modern"
    except (OSError, ValueError):
        return True  # no /proc (Windows test host): trust pgrep


def engine_processes(pattern: str | None = None) -> int:
    """How many ``th08-modern`` processes WSL currently runs (engine slot).

    ``pattern`` narrows the count to one batch: ``pgrep -f`` matches the whole
    command line, so ``"th08-modern --data-dir /root/v400-data/d0s"`` counts this
    run's engines (plus the shadow's ``PARK``/``SPAWN`` children, which inherit
    the argv) and never somebody else's ``g1`` instances.  ``None`` keeps the
    original whole-machine ``pgrep -x`` count.
    """

    argv = ["pgrep", "-f", pattern] if pattern else ["pgrep", "-x", "th08-modern"]
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return 0
    pids = [int(line) for line in out.stdout.split() if line.strip().isdigit()]
    return sum(1 for pid in pids if _is_engine_process(pid))


def wait_for_engine_slot(*, limit: int = 1, pattern: str | None = None,
                         timeout: float = 300.0, poll: float = 5.0,
                         log=print) -> dict[str, Any]:
    """Wait until fewer than ``limit`` matching engines run, then give up loudly.

    ``limit=1`` is the original single-channel contract (wait for an empty
    machine).  ``limit=N`` lets N-1 of this batch's engines keep running, which is
    what the two-channel scale-up needs; ``pattern`` is what makes "this batch"
    precise rather than "any engine on the host".
    """

    started = time.monotonic()
    seen = []
    while True:
        count = engine_processes(pattern)
        seen.append({"t": round(time.monotonic() - started, 3), "engines": count})
        if count < max(1, int(limit)):
            return {"waited_s": round(time.monotonic() - started, 3),
                    "limit": max(1, int(limit)), "pattern": pattern,
                    "samples": seen}
        if time.monotonic() - started >= timeout:
            raise SystemExit(json.dumps({
                "status": "ENGINE_BUSY_AFTER_%dS" % int(timeout),
                "engines": count, "limit": int(limit), "pattern": pattern,
                "samples": seen[-5:],
            }, ensure_ascii=False))
        log(f"[d0] engine slot busy ({count} engine(s) >= limit {limit}), "
            f"waiting {poll:.0f}s ...")
        time.sleep(poll)


# ---------------------------------------------------------------------------
# channel A: shadow labels (delegated to the teacher CLI, --labels-only)
# ---------------------------------------------------------------------------
def load_teacher_report(path: str | Path) -> dict[str, Any]:
    """The teacher CLI's final ``kind=report`` row (stop verdict, stage clear)."""

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("kind") == "report":
                return payload
    return {}


def run_labels(args) -> dict[str, Any]:
    """Run the teacher in the shadow role and leave ``labels.jsonl`` behind."""

    labels_path = Path(args.labels_jsonl)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    if labels_path.exists():
        labels_path.unlink()
    log_path = Path(args.work_dir) / "labels_teacher.log"
    argv = [
        sys.executable, str(Path(args.teacher_tool)),
        "--decisions", str(int(args.decisions)),
        "--stage", str(int(args.stage)),
        "--difficulty", str(int(args.difficulty)),
        "--team", str(int(args.team)),
        "--seed", str(int(args.seed)),
        "--mode", args.mode,
        "--labels-only",
        "--out", str(labels_path),
        "--data-dir", str(Path(args.data_root) / "shadow"),
        "--engine", str(Path(args.engine)),
        "--expected-sha", args.expected_sha,
        "--quiet",
    ]
    max_frames = int(getattr(args, "max_frames", 0) or 0)
    if max_frames:
        argv.extend(["--max-frames", str(max_frames)])
    if getattr(args, "stop_on_death", False):
        # A death ends the segment: the decisions after it are a corpse, not a
        # teacher trajectory, and both channels must stop on the same frame.
        argv.append("--stop-on-death")
    if args.binary:
        argv.extend(["--binary", str(Path(args.binary))])
    data = prepare_data_dir(Path(args.data_template), Path(args.data_root) / "shadow")
    started = time.monotonic()
    proc = subprocess.run(argv, cwd=str(Path(args.repo_root)), capture_output=True, text=True)
    wall = time.monotonic() - started
    log_path.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != EXIT_OK:
        raise DualError(
            f"channel A (shadow teacher) exited {proc.returncode}: "
            f"{(proc.stdout or proc.stderr).strip()[-600:]}")
    labels = load_labels(labels_path)
    requested = int(args.decisions)
    if len(labels) != requested:
        if not getattr(args, "allow_early_stop", False) or len(labels) > requested:
            raise DualError(
                f"channel A produced {len(labels)} decisions, requested {args.decisions}; "
                f"see {log_path}")
        if len(labels) <= 0:
            raise DualError(f"channel A produced no decisions; see {log_path}")
    teacher = load_teacher_report(labels_path)
    report = {
        "channel": "A",
        "role": "shadow",
        "exit_code": proc.returncode,
        "wall_s": round(wall, 3),
        "decisions": len(labels),
        "decisions_requested": requested,
        "stopped_early": len(labels) != requested,
        "stop_reason": teacher.get("stop_reason"),
        "stage_clear": teacher.get("stage_clear"),
        "frames_committed": teacher.get("frames_committed"),
        "deaths_observed": teacher.get("deaths_observed"),
        "teacher_status": teacher.get("status"),
        "max_frames": max_frames,
        "stop_on_death": bool(getattr(args, "stop_on_death", False)),
        "labels_jsonl": str(labels_path),
        "labels_sha256": sha256_file(labels_path),
        "log": str(log_path),
        "engine_binary": str(Path(args.engine) / "th08-modern"),
        "engine_sha256": args.expected_sha,
        "data_dir": data,
        "reset_command": reset_command(args),
        "hash_checkpoints": sum(2 for _ in labels),
        "frames": [labels[0]["frame_pre"], labels[-1]["frame_post"]],
    }
    _write_json(Path(args.work_dir) / "channel_a.json", report)
    return report


def load_labels(path: str | Path) -> list[dict[str, Any]]:
    """Read channel A's JSONL and prove it carries the alignment fields."""

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("kind") != "decision":
                continue
            where = f"{path}:{line_number}"
            index = len(rows)
            if int(payload.get("decision", -1)) != index:
                raise DualError(f"{where}: decision {payload.get('decision')!r} != row {index}")
            frames = [int(value) for value in payload.get("step_frames") or []]
            if len(frames) != DEFAULT_FRAMES_PER_DECISION:
                raise DualError(
                    f"{where}: {len(frames)} per-frame masks, need "
                    f"{DEFAULT_FRAMES_PER_DECISION} (--labels-only records them)")
            for side in ("hash_pre", "hash_post"):
                checkpoint = payload.get(side) or {}
                if not HASH_RE.match(str(checkpoint.get("hash", ""))):
                    raise DualError(f"{where}: {side} is not an engine HASH: {checkpoint!r}")
            pre = payload["hash_pre"]
            post = payload["hash_post"]
            if int(pre["frames"]) != index * DEFAULT_FRAMES_PER_DECISION:
                raise DualError(
                    f"{where}: hash_pre frames {pre['frames']} != "
                    f"{index * DEFAULT_FRAMES_PER_DECISION}")
            if int(post["frames"]) != (index + 1) * DEFAULT_FRAMES_PER_DECISION:
                raise DualError(
                    f"{where}: hash_post frames {post['frames']} != "
                    f"{(index + 1) * DEFAULT_FRAMES_PER_DECISION}")
            scores = payload.get("scores") or payload.get("geo_score")
            if scores is None or len(scores) != obs_spec.N_MOVE_CLASSES:
                raise DualError(f"{where}: no scores[18]")
            rows.append({
                "decision": index,
                "frame": int(payload["frame"]),
                "frame_pre": int(pre["frames"]),
                "hash_pre": str(pre["hash"]),
                "frame_post": int(post["frames"]),
                "hash_post": str(post["hash"]),
                "step_frames": frames,
                "scores": [float(value) for value in scores],
                "verified_mask": [bool(value) for value in payload["verified_mask"]],
                "chosen_plan": payload.get("chosen_plan"),
                "margin": payload.get("margin"),
                "bomb": int(payload["bomb"]),
                "stage": int(payload.get("stage", 0)),
                "difficulty": int(payload.get("difficulty", 0)),
                "frame_event_flags": payload.get("frame_event_flags"),
                "shoot_release": bool(payload.get("shoot_release", False)),
                "bomb_death_available": bool(payload.get("bomb_death_available", False)),
            })
    if not rows:
        raise DualError(f"{path}: no decision rows")
    for index, row in enumerate(rows):
        expected = index * DEFAULT_FRAMES_PER_DECISION
        if row["frame_pre"] != expected:
            raise DualError(f"{path}: decision {index} pre frame {row['frame_pre']} != {expected}")
        if row["frame"] != expected + DEFAULT_FRAMES_PER_DECISION:
            raise DualError(
                f"{path}: decision {index} JSONL frame {row['frame']} != post commit "
                f"{expected + DEFAULT_FRAMES_PER_DECISION} (the CLI writes the counter "
                "after the commit; the boundary arithmetic must agree)")
    return rows


# ---------------------------------------------------------------------------
# channel B: env-role replay (pixels + the same checkpoints)
# ---------------------------------------------------------------------------
def reset_command(args) -> str:
    return f"RESET {int(args.stage)} {int(args.difficulty)} {int(args.team)} {int(args.seed)} 0"


def _checkpoint(client) -> dict[str, Any]:
    fields = _tokens(client.request("HASH"))
    value = str(fields.get("hash", ""))
    if not HASH_RE.match(value):
        raise DualError(f"HASH reply is not an engine hash: {fields!r}")
    if "frames" not in fields:
        raise DualError(f"HASH reply has no frame counter: {fields!r}")
    return {"hash": value, "frames": int(fields["frames"])}


class _ShadowLabel:
    """Duck-typed ``TeacherLabel`` so the EXP-P1-013 writer is reused unchanged."""

    def __init__(self, row: dict[str, Any]) -> None:
        self.scores = np.asarray(row["scores"], dtype=np.float64)
        self.verified_mask = np.asarray(row["verified_mask"], dtype=bool)
        self.bomb = int(row["bomb"])
        self.margin = 0.0 if row["margin"] is None else float(row["margin"])
        self.stage = int(row["stage"])
        self.difficulty = int(row["difficulty"])
        self.frame_event_flags = row.get("frame_event_flags")
        self.shoot_release = bool(row.get("shoot_release", False))
        self.bomb_death_available = bool(row.get("bomb_death_available", False))
        plan = row.get("chosen_plan")
        self.chosen_plan = None if plan is None else _Plan(plan)


class _Plan:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def as_dict(self) -> dict[str, Any]:
        return dict(self._payload)


def _maybe_reexec_under_xvfb(args) -> None:
    """The env role needs a private Xvfb: WSLg's ``DISPLAY=:0`` hangs before HELLO."""

    if args.xvfb == "off" or os.environ.get("V400_D0_XVFB") == "1":
        return
    anchor = sys.argv[0] if sys.argv and sys.argv[0] else __file__
    argv = ["xvfb-run", "-a", sys.executable, str(Path(anchor).resolve()), *sys.argv[1:]]
    env = dict(os.environ, V400_D0_XVFB="1")
    os.execvpe("xvfb-run", argv, env)


def run_pixels(args, labels: list[dict[str, Any]]) -> dict[str, Any]:
    """Replay channel A's masks on an env-role engine and align every checkpoint."""

    _maybe_reexec_under_xvfb(args)
    from touhou_rl.v400.gym.shm import PIXELS_OFFSET, SharedBlock
    from touhou_rl.v400.labels.runner import ControlClient

    spec = obs_version_spec(getattr(args, "obs", OBS_V1))

    binary = args.binary or str(Path(args.engine) / "th08-modern")
    if sha256_file(binary) != args.expected_sha:
        raise DualError(
            f"{binary}: sha256 {sha256_file(binary)} != expected {args.expected_sha}")
    data = prepare_data_dir(Path(args.data_template), Path(args.data_root) / "env")
    shm_name = f"v400d0dual-{os.getpid()}-{int(time.time() * 1000) % 10_000_000}"
    log_path = Path(args.work_dir) / "env_engine.stderr.log"
    pixel_bytes = 640 * 480 * 4
    client = ControlClient(binary, data["data_dir"], role="env", timeout=args.timeout,
                           stderr_path=log_path)
    buffer = DecisionObservationBuffer(play_shape=spec.planes[0].shape_tail[1:])
    samples: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    renders: list[dict[str, float]] = []
    mismatch: dict[str, Any] | None = None
    pixels_ms = 0.0
    frames_read = 0
    exit_code = None
    stderr_tail = ""
    report: dict[str, Any] = {
        "channel": "B", "role": "env", "shm": shm_name, "engine_binary": binary,
        "engine_sha256": args.expected_sha, "reset_command": reset_command(args),
        "data_dir": data, "alignment": "hash-checkpoint",
        "obs_version": spec.short,
        "play_shape": list(spec.planes[0].shape_tail),
    }
    started = time.monotonic()
    try:
        client.start()
        hello = _tokens(client.request("HELLO"))
        if hello.get("role") != "env":
            raise DualError(f"HELLO did not confirm the env role: {hello!r}")
        client.request(f"SHM /{shm_name}")
        readback = _tokens(client.request("READBACK 1"))
        if readback.get("readback") != "1":
            raise DualError(f"READBACK 1 refused: {readback!r}")
        block = SharedBlock(shm_name)
        try:
            size = block.size
            if size != PIXELS_OFFSET + pixel_bytes:
                raise DualError(f"shared block is {size} B, expected {PIXELS_OFFSET + pixel_bytes}")
            reset = _tokens(client.request(reset_command(args), timeout=args.timeout))
            if reset.get("ready") != "1":
                raise DualError(f"RESET did not become ready: {reset!r}")
            reset_frame = int(client.request("FRAME").split("=", 1)[1])
            if reset_frame != 0:
                raise DualError(f"RESET left the frame counter at {reset_frame}, expected 0")
            # Episode-start padding: the last render the RESET left in the block.
            play, hud, header = _read_frame(block, f"post-RESET seed render", args.obs)
            seed = RenderFrame(frame=int(header["frame"]), play=play, hud=hud)
            renders.append(require_rendered(play, hud, where="post-RESET seed render"))
            buffer.reset(seed)
            report["seed"] = {"render_frame": int(header["frame"]),
                              "state_frame": reset_frame, "flags": int(header["flags"])}
            for index, label in enumerate(labels):
                decision_frame = index * DEFAULT_FRAMES_PER_DECISION
                frame_now = int(client.request("FRAME").split("=", 1)[1])
                if frame_now != decision_frame:
                    raise DualError(
                        f"decision {index}: engine frame {frame_now} != {decision_frame}")
                pre = _checkpoint(client)
                pairs.append(_pair(index, "pre", decision_frame, label["hash_pre"], pre, "A/B"))
                if pre["hash"] != label["hash_pre"]:
                    mismatch = pairs[-1]
                    break
                sample = buffer.stack_for(decision_frame, episode_start=(index == 0))
                for offset in range(DEFAULT_FRAMES_PER_DECISION):
                    mask = int(label["step_frames"][offset])
                    reply = _tokens(client.request(f"STEP {mask} 1"))
                    if int(reply.get("advanced", 0)) != 1:
                        raise DualError(
                            f"decision {index} frame {offset}: advanced="
                            f"{reply.get('advanced')} on STEP {mask} 1")
                    expected_render = decision_frame + offset
                    started_px = time.perf_counter()
                    play, hud, header = _read_frame(block, f"frame {expected_render}",
                                                    args.obs)
                    pixels_ms += (time.perf_counter() - started_px) * 1000.0
                    frames_read += 1
                    renders.append(require_rendered(play, hud, where=f"frame {expected_render}"))
                    buffer.push(RenderFrame(frame=int(header["frame"]), play=play, hud=hud))
                post = _checkpoint(client)
                pairs.append(_pair(index, "post", decision_frame + DEFAULT_FRAMES_PER_DECISION,
                                   label["hash_post"], post, "A/B"))
                if post["hash"] != label["hash_post"]:
                    mismatch = pairs[-1]
                    break
                samples.append(sample)
            report["exit_code"] = 0
        finally:
            block.close()
            try:
                client.request("QUIT", timeout=20.0)
            except Exception:  # pragma: no cover - QUIT may lose the race with close
                pass
            client.close()
            exit_code = client._proc.returncode if client._proc is not None else None
            try:
                if log_path.is_file():
                    stderr_tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            except OSError:
                pass
            try:
                from touhou_rl.v400.gym.shm import segment_path
                segment_path(shm_name).unlink()
            except (OSError, ValueError):
                pass
    except BaseException:
        try:
            client.close()
        except Exception:
            pass
        raise
    report.update({
        "wall_s": round(time.monotonic() - started, 3),
        "decisions": len(samples),
        "frames_read": frames_read,
        "pixel_ms": round(pixels_ms, 3),
        "pixel_ms_per_frame": round(pixels_ms / frames_read, 3) if frames_read else None,
        "render": _aggregate(renders),
        "checkpoints": len(pairs),
        "checkpoints_equal": sum(1 for row in pairs if row["equal"]),
        "checkpoints_mismatched": sum(1 for row in pairs if not row["equal"]),
        "stderr_tail": stderr_tail,
    })
    if not all(row["equal"] for row in pairs):
        first = next(row for row in pairs if not row["equal"])
        report["first_mismatch"] = first
        report["hash_checkpoint_pairs"] = pairs
        report["status"] = "ALIGNMENT_FAILED"
        _write_json(Path(args.work_dir) / "channel_b.json", report)
        _write_alignment(args, labels, pairs)
        # EXP-P1-018: the one boundary that can legitimately differ is the commit
        # that completes the stage (the post state is the next stage's setup).  Keep
        # the artifact -- channel A's decisions up to that boundary are aligned
        # frame for frame -- and let assemble drop exactly that last decision,
        # recording both hashes.  Every other mismatch stays fail-closed.
        trim = None
        if getattr(args, "trim_transition", False):
            teacher = {}
            channel_a_path = Path(args.work_dir) / "channel_a.json"
            if channel_a_path.is_file():
                teacher = _read_json(channel_a_path)
            trim = _transition_info_from_pairs(pairs, len(labels), teacher)
        if trim is None:
            raise AlignError(
                f"checkpoint mismatch at decision {first['decision']} {first['boundary']} "
                f"frame {first['frame']}: shadow={first['shadow']} env={first['env']}; "
                "D0 was not written")
        report["transition_trim"] = trim
        report["status"] = "ok_transition_trim"
    else:
        report["hash_checkpoint_pairs"] = pairs
        report["status"] = "ok"
    _write_json(Path(args.work_dir) / "channel_b.json", report)
    # Pixels are held in memory (60 decisions ~1.5 MB); a production shard streams
    # them through the writer instead (see the scale plan in the experiment card).
    play_planes = np.stack([row["play"] for row in samples]).astype(np.uint8)
    hud_planes = np.stack([row["hud"] for row in samples]).astype(np.uint8)
    np.savez_compressed(
        args.pixels_npz,
        play=play_planes,
        hud=hud_planes,
        render_frames=np.asarray([row["render_frames"] for row in samples], dtype=np.int64),
        pads=np.asarray([row["pad"] for row in samples], dtype=np.int64),
        decision_frames=np.asarray([row["window"]["decision_frame"] for row in samples],
                                   dtype=np.int64),
    )
    report["pixels_npz"] = str(args.pixels_npz)
    report["pixels_npz_sha256"] = sha256_file(args.pixels_npz)
    report["hash_checkpoint_pairs"] = pairs
    if report.get("transition_trim") is None:
        report["status"] = "ok"
    _write_alignment(args, labels, pairs)
    _write_json(Path(args.work_dir) / "channel_b.json", report)
    return report


def _write_alignment(args, labels, pairs) -> dict[str, Any]:
    """Write ``alignment.json``: every checkpoint, both sides, and the verdict."""

    payload = {
        "schema": "v400-d0-dual-alignment-1",
        "method": "hash-checkpoint",
        "label_channel": "shadow",
        "pixel_channel": "env",
        "decisions": len(labels),
        "boundaries": len(pairs),
        "checkpoints_equal": sum(1 for row in pairs if row["equal"]),
        "checkpoints_mismatched": sum(1 for row in pairs if not row["equal"]),
        "first_mismatch": next((row for row in pairs if not row["equal"]), None),
        "pairs": pairs[: int(args.max_pairs)],
        "pairs_total": len(pairs),
        "all_equal": all(row["equal"] for row in pairs),
        "sha256": _canonical_sha256(pairs),
    }
    _write_json(Path(args.alignment_json), payload)
    return payload


def _read_frame(block, where: str, obs_version: str = OBS_V1):
    from touhou_rl.v400.gym.shm import FLAG_PIXELS_VALID

    header = block.header()
    if not int(header.get("flags", 0)) & FLAG_PIXELS_VALID:
        raise DualError(f"{where}: shared block has no published pixels")
    if int(header.get("width", 0)) != 640 or int(header.get("height", 0)) != 480:
        raise DualError(f"{where}: readback is {header.get('width')}x{header.get('height')}")
    if obs_version == OBS_V3:
        from touhou_rl.v400.student import obs_spec_v3

        play, hud = obs_spec_v3.encode_planes_v3(block.pixels())
    else:
        play, hud = obs_spec.encode_planes(block.pixels())
    return play, hud, header


def _pair(decision: int, boundary: str, frame: int, shadow: str, env: dict, source: str):
    return {
        "decision": int(decision),
        "boundary": boundary,
        "frame": int(frame),
        "shadow": str(shadow),
        "env": str(env["hash"]),
        "env_frames": int(env["frames"]),
        "equal": str(shadow) == str(env["hash"]) and int(env["frames"]) == int(frame),
        "source": source,
    }


def _aggregate(renders: list[dict[str, float]]) -> dict[str, Any]:
    if not renders:
        return {"frames_sampled": 0}
    keys = ("play_mean", "play_std", "play_max", "hud_mean", "hud_std", "hud_max")
    out: dict[str, Any] = {"frames_sampled": len(renders)}
    for key in keys:
        values = [row[key] for row in renders if key in row]
        if values:
            out[key] = round(float(np.mean(values)), 6)
            out[key + "_min"] = round(float(np.min(values)), 6)
    out["zero_frames"] = sum(1 for row in renders if float(row.get("play_max", 1)) <= 0.0
                            or float(row.get("hud_max", 1)) <= 0.0)
    return out


# ---------------------------------------------------------------------------
# the stage-clear boundary (EXP-P1-018)
# ---------------------------------------------------------------------------
def _transition_info_from_pairs(pairs: list[dict[str, Any]], decisions: int,
                                channel_a: dict[str, Any]) -> dict[str, Any] | None:
    """The stage-clear exclusion, proved from the checkpoint pairs themselves.

    Requires: the teacher's verdict is ``stage_changed``/``stage_clear``, exactly
    one checkpoint differs, and that checkpoint is the ``post`` of the final
    decision.  Anything else returns ``None`` (the caller stays fail-closed).
    """

    if decisions < 2:
        return None
    if channel_a.get("stop_reason") != "stage_changed" or not channel_a.get("stage_clear"):
        return None
    if len(pairs) != 2 * decisions:
        return None
    bad = [row for row in pairs if not row["equal"]]
    if len(bad) != 1:
        return None
    first = bad[0]
    if int(first["decision"]) != decisions - 1 or first.get("boundary") != "post":
        return None
    return {
        "decision": int(first["decision"]),
        "frame": int(first["frame"]),
        "shadow_post": first["shadow"],
        "env_post": first["env"],
        "env_frames": int(first.get("env_frames") or 0),
        "reason": "stage-clear commit crosses the stage boundary; the post-transition "
                  "state is the next stage's setup, not stage-0 gameplay",
        "checkpoints_before_trim": 2 * decisions,
        "checkpoints_mismatched_before_trim": 1,
        "checkpoints_after_trim": 2 * (decisions - 1),
        "decisions_before_trim": decisions,
        "decisions_after_trim": decisions - 1,
    }


def _trim_stage_transition(args, labels: list[dict[str, Any]],
                           pixels: dict[str, np.ndarray],
                           channel_a: dict[str, Any]) -> tuple[list, dict, dict | None]:
    """Drop the one decision whose commit crosses the stage boundary.

    Measured (EXP-P1-018, stage 0 / normal, seeds 1-2): the *only* HASH checkpoint
    that ever disagrees between the shadow label channel and the env pixel channel
    is the ``post`` of the decision whose 4-frame commit completes the stage.  The
    in-stage segment agrees everywhere (6,740/6,740 boundaries, frame 13,480), and
    both sides still report the same frame counter at the transition; what differs
    is the state *after* ``AdvanceToNextStage`` -- the next stage's setup, which the
    shadow reaches through its ``PARK``/``SPAWN`` child and the env replay reaches
    in the main loop.

    That state is not stage-0 gameplay, so the segment ends at the last decision
    that is still fully inside the stage.  This is a documented exclusion, not a
    faked comparison: it only triggers on that one boundary, and the raw
    ``alignment.json`` (all boundaries, including the excluded one) is untouched on
    disk -- the shard records both hashes and the reason.
    """

    if not getattr(args, "trim_transition", False):
        return labels, pixels, None
    pairs_doc = json.loads(Path(args.alignment_json).read_text(encoding="utf-8"))
    info = _transition_info_from_pairs(pairs_doc.get("pairs") or [],
                                       len(labels), channel_a)
    if info is None:
        return labels, pixels, None
    # Channel B only appends a sample *after* its post checkpoint matched, so the
    # pixel array is already one decision shorter than the labels (3,370 vs 3,371
    # on seed 1).  Either shape is fine as long as the dropped decision is exactly
    # the transition; anything else stays fail-closed.
    keep = len(labels) - 1
    pixel_rows = int(np.asarray(pixels["play"]).shape[0])
    if pixel_rows == len(labels):
        trimmed_pixels = {key: np.asarray(value)[:-1] for key, value in pixels.items()}
    elif pixel_rows == keep:
        trimmed_pixels = pixels
    else:
        return labels, pixels, None
    return labels[:keep], trimmed_pixels, info


# ---------------------------------------------------------------------------
# resume + cross-process assemble lock (EXP-P1-018 scale-up)
# ---------------------------------------------------------------------------
def shard_resume_report(shard_dir: str | Path) -> dict[str, Any] | None:
    """``(report)`` if a finished, loadable shard already sits at ``shard_dir``.

    "Finished" means the writer renamed its staging directory, and "loadable"
    means the loader of the shard's **own** obs version still accepts it
    (``dataset.load_teacher_shard`` for v1, ``obs_v3_dataset.load_teacher_shard_v3``
    for v3 — the version is read from ``shard.json``, not from the CLI flag, so a
    v3 shard is never skipped as "unloadable" and never accepted by the wrong
    loader).  A ``.partial`` staging directory is *not* a shard and never
    short-circuits a resume.
    """

    path = Path(shard_dir)
    if not (path / "shard.json").is_file():
        return None
    try:
        raw = json.loads((path / "shard.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        if raw.get("obs_spec_version") == obs_spec.OBS_SPEC_VERSION:
            shard = dataset.load_teacher_shard(path)
        else:
            from touhou_rl.v400.student import obs_v3_dataset

            shard = obs_v3_dataset.load_teacher_shard_v3(path)
        decisions = len(shard)
    except Exception:  # a corrupt shard must be re-made, not silently accepted
        return None
    if decisions <= 0:
        return None
    report = shard.report()
    report["stop_reason"] = raw.get("stop_reason")
    report["stage_clear"] = raw.get("stage_clear")
    report["obs_bytes_per_decision"] = (raw.get("obs") or {}).get("bytes_per_decision")
    report["resumed"] = True
    report["decisions"] = decisions
    return report


@contextlib.contextmanager
def out_dir_lock(out_dir: str | Path):
    """Serialise ``manifest.json`` read-modify-write across processes.

    ``TeacherObsShardWriter._update_manifest`` reads the manifest, merges its own
    shard entry and atomically replaces the file; two channels assembling in
    parallel would otherwise drop one entry.  ``flock`` is advisory and only used
    where it exists (the real runs are Linux; the Windows test host falls back to
    a no-op because the tests are single-process).
    """

    path = Path(out_dir) / ".assemble.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+", encoding="utf-8")
    locked = False
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            locked = True
        except (ImportError, OSError):  # pragma: no cover - Windows test host
            pass
        yield
    finally:
        if locked:  # pragma: no cover - exercised only on Linux
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
        handle.close()


# ---------------------------------------------------------------------------
# assemble: A's labels + B's pixels -> the EXP-P1-013 shard
# ---------------------------------------------------------------------------
def v3_action_history(labels: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    """Rebuild the obs-v3 ``act_hist`` (N,8,18) from the aligned label rows.

    The reconstruction is obs-v2's frozen rule: a D0/teacher-driven row executes
    ``chosen_plan.first_action`` at its decision, so ``build_action_history`` can
    rebuild ``a[t-8]..a[t-1]``.  The stage is constant inside one dual shard (the
    CLI's ``--stage``; a stage-clear commit is the single boundary the dual
    alignment either trims or refuses), so the history is cleared at the shard
    start and at every stage change that the shard really contains.

    Anything the rule cannot reconstruct is refused here, never padded silently.
    """

    try:
        cand, stage = obs_spec_v2.reconstruct_from_label_rows(labels)
        history = obs_spec_v2.build_action_history(cand, stage)
        obs_spec_v2.check_action_history(history)
    except obs_spec_v2.ObsSpecV2Error as error:
        raise DualError(
            f"obs-v3 act_hist cannot be reconstructed from the label rows: {error}"
        ) from error
    zeros = (history.sum(axis=-1) == 0).sum(axis=1)
    info = {
        "field": "act_hist",
        "shape": list(history.shape),
        "dtype": "uint8",
        "order": obs_spec_v2.ACTION_HISTORY_ORDER,
        "excludes_current_action": True,
        "source": ("chosen_plan.first_action (teacher-driven dual shard; the action "
                   "the teacher committed at that decision)"),
        "stage_values": sorted({int(value) for value in stage}),
        "stage_changes": int(np.sum(np.asarray(stage)[1:] != np.asarray(stage)[:-1]))
        if len(stage) > 1 else 0,
        "zero_rows_at_start": int(zeros[0]) if len(zeros) else 0,
        "rows_with_padding": int((zeros > 0).sum()),
        "first_unpadded_row": (int(np.argmax(zeros == 0))
                               if bool((zeros == 0).any()) else -1),
        "executed_cand_histogram": np.bincount(cand, minlength=obs_spec.N_MOVE_CLASSES)
        .tolist() if len(cand) else [],
        "sha256": obs_spec_v2.history_digest(history),
    }
    return history, info


def assemble(args, labels: list[dict[str, Any]], pixels: dict[str, np.ndarray],
             channel_a: dict[str, Any], channel_b: dict[str, Any]) -> dict[str, Any]:
    spec = obs_version_spec(getattr(args, "obs", OBS_V1))
    check_obs_version(spec.short)
    out_dir = Path(args.out_dir)
    shard_key = args.shard_key
    labels, pixels, trim_info = _trim_stage_transition(args, labels, pixels, channel_a)
    play = pixels["play"]
    hud = pixels["hud"]
    render_frames = pixels["render_frames"]
    pads = pixels["pads"]
    decision_frames = pixels["decision_frames"]
    expected_play = (len(labels), *spec.planes[0].shape_tail)
    if play.shape != expected_play:
        raise DualError(
            f"pixels play shape {play.shape} != {expected_play} for {spec.version}")
    act_hist = history_info = None
    if spec.short == OBS_V3:
        from touhou_rl.v400.student import obs_spec_v3

        act_hist, history_info = v3_action_history(labels)
        if act_hist.shape[0] != len(labels):
            raise DualError(
                f"act_hist rows {act_hist.shape[0]} != {len(labels)} decisions")
        expected_bytes_per_decision = obs_spec_v3.BYTES_PER_DECISION
    else:
        expected_bytes_per_decision = (
            obs_spec.FRAME_STACK * obs_spec.PLAY_SHAPE[0] * obs_spec.PLAY_SHAPE[1]
            + obs_spec.HUD_SHAPE[0] * obs_spec.HUD_SHAPE[1])
    pairs = json.loads(Path(args.alignment_json).read_text(encoding="utf-8"))
    pair_rows = pairs["pairs"]
    evidence = {"boundaries": int(pairs["boundaries"]),
                "checkpoints_equal": int(pairs["checkpoints_equal"]),
                "checkpoints_mismatched": int(pairs["checkpoints_mismatched"]),
                "sha256": pairs.get("sha256")}
    if trim_info is not None:
        # The excluded boundary is the raw file's single mismatch, so the shard's
        # own checkpoint set is the kept prefix and it must be all-equal on its own.
        keep = 2 * len(labels)
        if len(pair_rows) < keep:
            raise AlignError(
                f"alignment.json evidence holds {len(pair_rows)} pairs, the trimmed "
                f"segment needs {keep}; rerun --phase pixels with --max-pairs >= {keep}")
        pair_rows = pair_rows[:keep]
    equal = sum(1 for row in pair_rows if row["equal"])
    if equal != len(pair_rows):
        raise AlignError("alignment.json contains a mismatched checkpoint")
    if trim_info is None and not pairs.get("all_equal"):
        raise AlignError("alignment.json contains a mismatched checkpoint")
    kept_sha256 = _canonical_sha256(pair_rows)
    # The "before decide" of decision k+1 is the "after commit" of decision k: both
    # sides must agree across that duplicate too, otherwise one of the two runs
    # drifted between checkpoints even though each side matched itself.
    for index in range(len(labels) - 1):
        if labels[index]["hash_post"] != labels[index + 1]["hash_pre"]:
            raise AlignError(
                f"channel A moved between decision {index} and {index + 1}: "
                f"{labels[index]['hash_post']} != {labels[index + 1]['hash_pre']}")
    alignment_extra = {
        "pixel_source": "env_replay",
        "label_source": "shadow_teacher",
        "hash_alignment": "hash-checkpoint",
        "alignment_method": "hash-checkpoint",
        "hash_checkpoints": {
            "method": "hash-checkpoint",
            "boundaries": len(pair_rows),
            "decisions": len(labels),
            "checkpoints_equal": equal,
            "checkpoints_mismatched": len(pair_rows) - equal,
            "first_mismatch": next((row for row in pair_rows if not row["equal"]), None),
            "sha256": kept_sha256,
            "evidence": evidence,
            "trimmed_transition": trim_info,
        },
    }
    meta = {
        "stage": int(args.stage),
        "difficulty": int(args.difficulty),
        "team": int(args.team),
        "seed": int(args.seed),
        "mode": args.mode,
        "frames_per_decision": DEFAULT_FRAMES_PER_DECISION,
        "generated_by": "tools/rl/v400_d0_dual.py (EXP-P1-014)",
        "decisions": len(labels),
        "decisions_requested": int(getattr(args, "decisions_requested",
                                           channel_a.get("decisions_requested")
                                           or len(labels))),
        "stopped_early": bool(channel_a.get("stopped_early", False)),
        "stop_reason": channel_a.get("stop_reason"),
        "stage_clear": channel_a.get("stage_clear"),
        "frames_committed": channel_a.get("frames_committed"),
        "deaths_observed": channel_a.get("deaths_observed"),
        "transition_trim": trim_info,
        "label_channel": "shadow",
        "pixel_channel": "env",
        "alignment_method": "hash-checkpoint",
        # The requested observation version and the planes it implies.  ``shard.json``
        # already carries ``obs_spec_version``/``obs_spec_digest``; these keys make the
        # request auditable next to the planes that were actually written.
        "obs_requested": spec.short,
        "obs_planes": [plane.meta_key for plane in spec.planes],
        "obs_bytes_per_decision_expected": expected_bytes_per_decision,
        "alignment_extra": alignment_extra,
        "engine_shadow": {
            "binary": channel_a.get("engine_binary"),
            "sha256": channel_a.get("engine_sha256"),
            "role": "shadow",
        },
        "engine_env": {
            "binary": channel_b.get("engine_binary"),
            "sha256": channel_b.get("engine_sha256"),
            "role": "env",
        },
        "reset_command": reset_command(args),
        "label_channel_report": channel_a,
        "pixel_channel_report": channel_b,
        "hash_checkpoint_pairs": pair_rows,
    }
    if history_info is not None:
        meta["act_hist"] = history_info
    writer = TeacherObsShardWriter(out_dir, key=shard_key, meta=meta,
                                   obs_version=spec.short)
    try:
        for index, row in enumerate(labels):
            sample = {
                "play": play[index],
                "hud": hud[index],
                "render_frames": [int(value) for value in render_frames[index]],
                "pad": int(pads[index]),
                "window": {
                    "decision_frame": int(decision_frames[index]),
                    "newest_render_frame": int(decision_frames[index]) - 1,
                    "label_frames": [int(decision_frames[index]),
                                     int(decision_frames[index]) + 1],
                    "p3_window_last_row": int(decision_frames[index]) - 1,
                    "p3_label_frame": int(decision_frames[index]),
                },
            }
            if act_hist is not None:
                sample["act_hist"] = act_hist[index]
            writer.add(
                decision=index,
                decision_frame=int(decision_frames[index]),
                label=_ShadowLabel(row),
                sample=sample,
                extra={
                    "channel": "shadow-labels",
                    "pixel_channel": "env",
                    "hash_pre": row["hash_pre"],
                    "hash_post": row["hash_post"],
                    "step_frames": list(row["step_frames"]),
                    "frame_event_flags": row["frame_event_flags"],
                    "shoot_release": row["shoot_release"],
                    "bomb_death_available": row["bomb_death_available"],
                },
            )
        summary = writer.finalize(report={
            "dual_channel": "EXP-P1-014",
            "hash_alignment": alignment_extra["hash_checkpoints"],
            "channel_a": {key: channel_a.get(key) for key in
                          ("decisions", "labels_sha256", "wall_s", "exit_code")},
            "channel_b": {key: channel_b.get(key) for key in
                          ("decisions", "frames_read", "pixel_ms_per_frame", "wall_s")},
        })
    except BaseException as error:
        writer.abort(error)
        raise
    return summary


def verify_load(args, summary: dict[str, Any]) -> dict[str, Any]:
    from touhou_rl.v400.teacher.obs import (
        FROZEN_OBS_SPEC_DIGEST,
        FROZEN_OBS_SPEC_V3_DIGEST,
    )

    shard_dir = Path(summary["dir"])
    # With --allow-early-stop the shard holds the decisions the teacher really
    # produced (a stage clear / death ends the segment), so the load gate must ask
    # for that count, not the requested cap.
    expected = int(getattr(args, "decisions_actual", None) or args.decisions)
    if obs_version_spec(getattr(args, "obs", OBS_V1)).short == OBS_V3:
        from touhou_rl.v400.student import obs_v3_dataset

        loaded = obs_v3_dataset.load_teacher_samples_v3(
            [shard_dir / "labels.jsonl"],
            min_samples=expected,
            obs_root=Path(args.out_dir),
            require_observations=True,
        )
        shard = obs_v3_dataset.load_teacher_shard_v3(shard_dir)
        sample = shard[0]
        normalized = shard.normalized(1)
        return {
            "obs_version": shard.obs_version,
            "samples": int(loaded["count"]),
            "expected_samples": expected,
            "samples_match": int(loaded["count"]) == expected,
            "scores_shape": list(np.asarray(loaded["samples"][0]["scores"]).shape),
            "verified_mask_shape": list(
                np.asarray(loaded["samples"][0]["verified_mask"]).shape),
            "play_shape": list(np.asarray(sample["play"]).shape),
            "play_dtype": str(np.asarray(sample["play"]).dtype),
            "hud_shape": list(np.asarray(sample["hud"]).shape),
            "hud_dtype": str(np.asarray(sample["hud"]).dtype),
            "act_hist_shape": list(np.asarray(sample["act_hist"]).shape),
            "act_hist_dtype": str(np.asarray(sample["act_hist"]).dtype),
            "normalized_play_dtype": str(normalized["play"].dtype),
            "normalized_act_hist_shape": list(
                np.asarray(normalized["act_hist"]).shape),
            "verified_actions_first_row": list(sample["verified_actions"]),
            "obs_spec_version": loaded["obs_spec_version"],
            "obs_spec_digest": loaded["obs_spec_digest"],
            "frozen_digest_expected": FROZEN_OBS_SPEC_V3_DIGEST,
            "digest_matches_frozen": (loaded["obs_spec_digest"]
                                      == FROZEN_OBS_SPEC_V3_DIGEST),
            "alignment": loaded["alignment"],
            "obs_unresolved": loaded["obs_unresolved"],
            "bytes_per_decision": (shard.meta.get("obs") or {}).get(
                "bytes_per_decision"),
            "shard": shard.report(),
        }
    loaded = dataset.load_teacher_samples(
        [shard_dir / "labels.jsonl"],
        min_samples=expected,
        obs_root=Path(args.out_dir),
        require_observations=True,
    )
    shard = dataset.load_teacher_shard(shard_dir)
    sample = shard[0]
    normalized = shard.normalized(1)
    report = {
        "obs_version": OBS_V1,
        "samples": int(loaded["count"]),
        "expected_samples": expected,
        "samples_match": int(loaded["count"]) == expected,
        "scores_shape": list(np.asarray(loaded["samples"][0]["scores"]).shape),
        "verified_mask_shape": list(np.asarray(loaded["samples"][0]["verified_mask"]).shape),
        "play_shape": list(np.asarray(sample["play"]).shape),
        "hud_shape": list(np.asarray(sample["hud"]).shape),
        "play_dtype": str(np.asarray(sample["play"]).dtype),
        "hud_dtype": str(np.asarray(sample["hud"]).dtype),
        "normalized_play_dtype": str(normalized["play"].dtype),
        "obs_spec_version": loaded["obs_spec_version"],
        "obs_spec_digest": loaded["obs_spec_digest"],
        "frozen_digest_expected": FROZEN_OBS_SPEC_DIGEST,
        "digest_matches_frozen": loaded["obs_spec_digest"] == FROZEN_OBS_SPEC_DIGEST,
        "alignment": loaded["alignment"],
        "obs_unresolved": loaded["obs_unresolved"],
        "shard": shard.report(),
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("all", "labels", "pixels", "assemble"),
                        default="all")
    parser.add_argument("--decisions", type=int, default=60)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--team", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    # T2/D-14: obs-v1 stays the default (every existing shard); obs-v3 stores the
    # 96x112 play plane plus the 8x18 action history (T3's D1 format).
    parser.add_argument("--obs", choices=OBS_VERSIONS, default=OBS_V1,
                        help="observation version written by --phase assemble: "
                             "v1 (default, play64+hud) or v3 (play96x112+hud+act_hist)")
    # EXP-P3-004: the D0 pipeline must be able to collect with any *registered*
    # teacher profile, not just ``full``/``fast`` — the label-signal 2x2 re-runs
    # channel A with ``fast3`` (FAST with verify_k=3).  The choice list is the
    # teacher registry itself, so a profile can never be spelled two ways.
    parser.add_argument("--mode", choices=tuple(sorted(teacher_config.CONFIGS)),
                        default="fast")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--teacher-tool", default=None)
    parser.add_argument("--engine", default="/root/Touhou_RL/third_party/th08-rl/build/rl_hazard2")
    parser.add_argument("--binary", default=None)
    parser.add_argument("--expected-sha", default=EXPECTED_ENGINE_SHA256)
    parser.add_argument("--data-template", default="/root/v400-data/game")
    parser.add_argument("--data-root", default="/root/v400-data/d0dual")
    parser.add_argument("--work-dir", default=".tmp_v400_d0_dual")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--labels-jsonl", default=None)
    parser.add_argument("--pixels-npz", default=None)
    parser.add_argument("--alignment-json", default=None)
    parser.add_argument("--shard-key", default=None)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--xvfb", choices=("auto", "off"), default="auto")
    parser.add_argument("--engine-wait", type=float, default=300.0,
                        help="max seconds to wait for a free engine slot")
    parser.add_argument("--engine-limit", type=int, default=1,
                        help="how many engines of this batch may run at once "
                             "(1 = wait for an empty machine, 4 = 2 channels x "
                             "shadow parent+child)")
    parser.add_argument("--engine-pattern", default=None,
                        help="pgrep -f pattern that matches only this batch's "
                             "engines (default: count every th08-modern)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="channel A frame cap (0 = decisions x 4), forwarded "
                             "to the teacher; a stage clear/death also stops it")
    parser.add_argument("--stop-on-death", action="store_true",
                        help="channel A stops at the first committed death")
    parser.add_argument("--allow-early-stop", action="store_true",
                        help="accept fewer decisions than --decisions when the "
                             "teacher stopped on a stage clear/death (the real "
                             "count and stop_reason are recorded)")
    parser.add_argument("--trim-transition", action="store_true",
                        help="end the segment at the last decision fully inside the "
                             "stage when the single mismatching checkpoint is the "
                             "stage-clear commit's post state (EXP-P1-018); the "
                             "exclusion and both hashes are recorded in shard.json")
    parser.add_argument("--force-rerun", action="store_true",
                        help="assemble again even if a finished shard exists")
    parser.add_argument("--max-pairs", type=int, default=512)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    args.teacher_tool = args.teacher_tool or str(Path(args.repo_root) / "tools" / "rl" /
                                                 "v400_teacher.py")
    args.out_dir = args.out_dir or str(work / "d0")
    args.labels_jsonl = args.labels_jsonl or str(work / "labels.jsonl")
    args.pixels_npz = args.pixels_npz or str(work / "pixels.npz")
    args.alignment_json = args.alignment_json or str(work / "alignment.json")
    args.shard_key = args.shard_key or f"dual-s{args.stage}d{args.difficulty}seed{args.seed}"
    args.decisions_requested = int(args.decisions)
    args.decisions_actual = None
    shard_dir = Path(args.out_dir) / "shards" / args.shard_key
    if not args.force_rerun:
        resumed = shard_resume_report(shard_dir)
        if resumed is not None:
            print(json.dumps({"phase": "resume", "status": "SKIPPED_COMPLETED",
                              "shard_key": args.shard_key, "dir": str(shard_dir),
                              "decisions": resumed["decisions"],
                              "stop_reason": resumed.get("stop_reason"),
                              "obs_bytes_per_decision":
                                  resumed.get("obs_bytes_per_decision")},
                             ensure_ascii=False), flush=True)
            return EXIT_OK
    # A leftover staging directory from a killed run is evidence, not garbage: it
    # keeps the writer's "refuse to overwrite" contract honest, so the rerun stops
    # here with a pointer instead of quietly starting a second .partial.
    partial_dir = Path(args.out_dir) / "shards" / f".{args.shard_key}.partial"
    if partial_dir.exists():
        raise DualError(
            f"staging directory {partial_dir} exists from an earlier interrupted run; "
            f"inspect/keep it as evidence and rerun with a fresh --out-dir/--shard-key "
            f"or remove it deliberately (nothing was overwritten)")
    # The env role needs a private Xvfb (WSLg's DISPLAY=:0 hangs before HELLO).
    # Re-exec before any channel runs, so a resumed --phase all does not repeat A.
    if args.phase in ("pixels", "all"):
        _maybe_reexec_under_xvfb(args)
    if args.phase in ("labels", "all"):
        slot = wait_for_engine_slot(limit=args.engine_limit, pattern=args.engine_pattern,
                                    timeout=args.engine_wait)
        print(json.dumps({"phase": "engine_slot", "channel": "A", **slot},
                         ensure_ascii=False), flush=True)
        channel_a = run_labels(args)
        args.decisions_actual = int(channel_a["decisions"])
        print(json.dumps({"phase": "labels", **{k: channel_a[k] for k in
                            ("exit_code", "decisions", "decisions_requested", "stopped_early",
                             "stop_reason", "stage_clear", "labels_sha256", "wall_s")}},
                          ensure_ascii=False), flush=True)
    labels = load_labels(args.labels_jsonl) if args.phase in ("pixels", "assemble", "all") \
        else []
    if args.decisions_actual is None and labels:
        args.decisions_actual = len(labels)
    if args.phase in ("pixels", "all"):
        slot = wait_for_engine_slot(limit=args.engine_limit, pattern=args.engine_pattern,
                                    timeout=args.engine_wait)
        print(json.dumps({"phase": "engine_slot", "channel": "B", **slot},
                         ensure_ascii=False), flush=True)
        channel_b = run_pixels(args, labels)
        print(json.dumps({"phase": "pixels", **{k: channel_b[k] for k in
                            ("decisions", "frames_read", "pixel_ms_per_frame",
                             "checkpoints", "checkpoints_equal", "wall_s")}},
                          ensure_ascii=False), flush=True)
    if args.phase in ("assemble", "all"):
        if not Path(args.pixels_npz).is_file():
            raise SystemExit(f"{args.pixels_npz}: no pixel artifact; run --phase pixels")
        if not Path(args.alignment_json).is_file():
            raise SystemExit(f"{args.alignment_json}: no alignment artifact")
        channel_a = _read_json(Path(args.work_dir) / "channel_a.json")
        channel_b = _read_json(Path(args.work_dir) / "channel_b.json")
        pixels = dict(np.load(args.pixels_npz))
        with out_dir_lock(args.out_dir):
            summary = assemble(args, labels, pixels, channel_a, channel_b)
            # A trimmed segment is shorter than channel A's labels; the load gate
            # must ask for the decisions that were actually written.
            args.decisions_actual = int(summary["decisions"])
            verification = verify_load(args, summary)
            payload = {
                "schema": "v400-d0-dual-report-1",
                "shard_key": args.shard_key,
                "label_channel": "shadow",
                "pixel_channel": "env",
                "alignment": "hash-checkpoint",
                "shard": summary,
                "load": verification,
            }
            # Per-shard report: two channels assemble into the same out-dir, so the
            # shared ``dual_channel.json`` may only be seeded once (compat with the
            # 60-decision smoke artifact) and every shard keeps its own copy.
            _write_json(Path(args.out_dir) / f"dual_channel.{args.shard_key}.json", payload)
            legacy = Path(args.out_dir) / "dual_channel.json"
            if not legacy.exists():
                _write_json(legacy, payload)
        print(json.dumps({"phase": "assemble", "shard": summary["shard"],
                          "shard_key": args.shard_key,
                          "decisions": summary["decisions"],
                          "decisions_requested": args.decisions_requested,
                          "obs_bytes_per_decision": summary["obs_bytes_per_decision"],
                          "load": {key: verification[key] for key in
                                   ("samples", "expected_samples", "samples_match",
                                    "scores_shape", "play_shape", "hud_shape",
                                    "obs_spec_digest", "alignment")}},
                         ensure_ascii=False), flush=True)
    return EXIT_OK


def _load_pairs(args) -> list[dict[str, Any]]:
    channel_b = _read_json(Path(args.work_dir) / "channel_b.json")
    labels = load_labels(args.labels_jsonl)
    raw = channel_b.get("hash_checkpoint_pairs")
    if raw is None:
        raise SystemExit(
            f"{args.work_dir}/channel_b.json carries no checkpoint pairs; rerun --phase pixels")
    return raw


def _pair_row(row: dict[str, Any]) -> dict[str, Any]:
    return row


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AlignError as error:
        print(json.dumps({"status": "ALIGNMENT_FAILED", "reason": str(error)},
                         ensure_ascii=False), flush=True)
        sys.exit(EXIT_ALIGN)
    except (DualError, ObservationError) as error:
        print(json.dumps({"status": "FAIL_CLOSED", "reason": str(error)},
                         ensure_ascii=False), flush=True)
        sys.exit(EXIT_FAIL_CLOSED)
