"""P3-WP3 DAgger round-1 collector (EXP-P3-003).

One episode is a **lockstep pair of engine processes**:

* ``env`` role 鈥?owns the pixels (``obs-v1`` play64/hud planes) and executes the
  committed mask block;
* ``shadow`` role 鈥?runs the *real* teacher ``decide()`` at the same decision
  point, so ``scores[18]`` labels **the state the student actually visited**
  (a DAgger query, not the teacher's own trajectory).

Both processes receive the identical 4-frame mask block, so every decision
boundary is HASH-checkpointed on both sides and compared before the row is
accepted (``tools/rl/v400_d0_dual.py``'s alignment logic, reused verbatim:
``_checkpoint``/``_pair``/``_transition_info_from_pairs``/``--trim-transition``).
A single mismatched checkpoint is fail-closed: the shard is never written.

尾-mixing (pre-registered in EXP-P3-002 搂7): at every decision a seeded RNG
injects the **teacher's chosen action** with probability 尾 (default 0.2);
otherwise the BC checkpoint's greedy action is executed.  The injection draw,
the flag, both candidates and the executed candidate are recorded per decision,
so the realised injection rate and the student/teacher disagreement are
auditable from the shard itself.

Because the teacher labels every decision of the *executed* trajectory, this is
DAgger with ``尾``-mixing and the **labeling policy = 尾-mixed policy** (100 % of
the new states are labelled), which is the configuration registered in
EXP-P3-002 搂7.

Output schema is byte-compatible with D0 (``TeacherObsShardWriter``), so
``touhou_rl.v400.student.dataset.load_teacher_samples(require_observations=True)``
loads it unchanged.

Phases
------
``run``      orchestrate episodes end to end (default)
``verify``   load every finished shard through the frozen D0 loader and report
``plan``     print the episode plan (seeds, shard keys, budget) and exit

Usage (WSL, one pair of engines at a time)::

    wsl -e bash .tmp_v400_dagger/run_collect.sh \
        .tmp_v400_p3v1/checkpoint-v1.pt .tmp_v400_dagger/d0_extra 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_TOOLS_RL = str(REPO_ROOT / "tools" / "rl")
if _TOOLS_RL not in sys.path:
    sys.path.insert(0, _TOOLS_RL)

from touhou_rl.v400.student import obs_spec  # noqa: E402

#: D0's dual-channel alignment logic, reused verbatim (``_checkpoint``/
#: ``_pair``/``_transition_info_from_pairs``/``prepare_data_dir``/``sha256_file``).
import v400_d0_dual as dual  # noqa: E402

#: The D0/teacher decision period is 4 logic frames (``replan_decisions`` 2 x
#: ``decision_frames`` 2).  The student commits one action per decision; holding
#: it for the whole block keeps the *episode structure* identical to D0 so the
#: two datasets can be concatenated without a schema translation.
FRAMES_PER_DECISION = 4
SHARD_PREFIX = "dagger"
EPISODE_SCHEMA = "v400-dagger-episode-1"
PROGRESS_SCHEMA = "v400-dagger-progress-1"
EXIT_OK = 0
EXIT_FAIL_CLOSED = 4


class DaggerError(RuntimeError):
    """A precondition, a protocol reply or the shard contract failed."""


class AlignError(DaggerError):
    """A HASH checkpoint disagrees; the pair is not the same trajectory."""


# ---------------------------------------------------------------------------
# pure logic (unit-testable without an engine)
# ---------------------------------------------------------------------------
def check_beta(beta: float) -> float:
    """Validate 尾 (teacher-injection probability)."""

    value = float(beta)
    if not np.isfinite(value):
        raise DaggerError(f"beta {beta!r} is not finite")
    if value < 0.0 or value > 1.0:
        raise DaggerError(f"beta {value} is outside [0, 1]")
    return value


def inject_decision(draw: float, beta: float, teacher_cand: int | None) -> bool:
    """Should this decision execute the teacher's action?

    ``draw`` is the seeded uniform draw in ``[0, 1)``.  A decision with no
    teacher plan (``teacher_cand is None``) is **never** injected: there is no
    teacher action to inject, and silently substituting the student's would
    report a 尾 that the data does not have.
    """

    beta = check_beta(beta)
    if not 0.0 <= float(draw) < 1.0:
        raise DaggerError(f"injection draw {draw!r} is not in [0, 1)")
    if teacher_cand is None:
        return False
    return float(draw) < beta


def align_pair(decision: int, boundary: str, frame: int, shadow: dict[str, Any],
               env: dict[str, Any]) -> dict[str, Any]:
    """Compare one checkpoint; ``equal`` also requires the frame counters to agree."""

    for side, value in (("shadow", shadow), ("env", env)):
        if not isinstance(value, dict) or "hash" not in value or "frames" not in value:
            raise AlignError(f"decision {decision} {boundary}: {side} checkpoint is {value!r}")
    return {
        "decision": int(decision),
        "boundary": str(boundary),
        "frame": int(frame),
        "shadow": str(shadow["hash"]),
        "env": str(env["hash"]),
        "shadow_frames": int(shadow["frames"]),
        "env_frames": int(env["frames"]),
        "equal": str(shadow["hash"]) == str(env["hash"]) and int(env["frames"]) == int(frame),
        "source": "dagger A/B",
    }


def require_aligned(pairs: Iterable[dict[str, Any]], *, decisions: int) -> dict[str, Any]:
    """Count the checkpoints and raise on the first disagreement (fail-closed)."""

    rows = list(pairs)
    bad = [row for row in rows if not row["equal"]]
    report = {
        "method": "hash-checkpoint",
        "decisions": int(decisions),
        "boundaries": len(rows),
        "checkpoints_equal": len(rows) - len(bad),
        "checkpoints_mismatched": len(bad),
        "first_mismatch": bad[0] if bad else None,
        "all_equal": not bad,
    }
    if bad:
        first = bad[0]
        raise AlignError(
            f"checkpoint mismatch at decision {first['decision']} {first['boundary']} "
            f"frame {first['frame']}: shadow={first['shadow']} env={first['env']} "
            f"(env frames {first.get('env_frames')})")
    return report


def shard_key(seed: int, *, stage: int = 0, difficulty: int = 1) -> str:
    """``dagger-s<stage>d<difficulty>seed<seed>`` 鈥?same shape as D0's keys."""

    return f"{SHARD_PREFIX}-s{int(stage)}d{int(difficulty)}seed{int(seed)}"


def episode_plan(args) -> list[dict[str, Any]]:
    """The seeds this run may collect, and the shard key each maps to."""

    seeds = []
    seed = int(args.seed_start)
    for offset in range(int(args.episodes)):
        seeds.append(seed + offset)
    return [{"seed": value,
             "key": shard_key(value, stage=int(args.stage), difficulty=int(args.difficulty)),
             "eval_overlap": 1000 <= value <= 1004} for value in seeds]


def split_seed_collisions(entries: Iterable[dict[str, Any]]) -> list[int]:
    """Seeds that appear in more than one shard (would break episode isolation)."""

    seen: dict[int, str] = {}
    bad: list[int] = []
    for entry in entries:
        seed = int(entry.get("seed", -1))
        key = str(entry.get("key") or entry.get("shard"))
        if seed in seen and seen[seed] != key:
            bad.append(seed)
        seen[seed] = key
    return sorted(set(bad))


def count_roots(roots: Iterable[str | Path]) -> dict[str, Any]:
    """Decision/shard totals per D0-shaped root (aggregation accounting)."""

    per_root = []
    total = 0
    keys: list[str] = []
    for root in roots:
        path = Path(root)
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise DaggerError(f"{path}: no manifest.json (not a D0-shaped root)")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        decisions = 0
        shards = 0
        for row in manifest.get("shards") or []:
            key = str(row.get("shard"))
            shard_json = path / "shards" / key / "shard.json"
            if not shard_json.is_file():
                continue
            payload = json.loads(shard_json.read_text(encoding="utf-8"))
            if payload.get("complete") is not True:
                continue
            decisions += int(payload.get("decision_count") or 0)
            shards += 1
            keys.append(key)
        per_root.append({"root": str(path), "shards": shards, "decisions": decisions,
                         "manifest_updated_at": (manifest.get("totals") or {}).get("updated_at")})
        total += decisions
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise DaggerError(f"shard key(s) appear in more than one root: {duplicates}")
    return {"roots": per_root, "shards": len(keys), "decisions": int(total),
            "shard_keys": sorted(keys)}


def campaign_summary(*, decisions: int, target: int, checkpoints: int,
                     checkpoints_equal: int, elapsed_s: float, episodes: int,
                     failures: int, beta: float, mode: str,
                     checkpoint: str) -> dict[str, Any]:
    """The campaign-level accounting written to ``<work>/summary.json``.

    ``partial`` is *derived*, never asserted by the caller: a run that stops on the
    wall budget short of ``target`` must report ``partial=true``.  The alignment
    rate is ``None`` (not 1.0) when no checkpoint was taken, so an empty campaign
    cannot look aligned.
    """

    if int(target) < 0:
        raise DaggerError(f"target decisions must be >= 0, got {target!r}")
    if int(checkpoints) < 0 or int(checkpoints_equal) < 0:
        raise DaggerError("checkpoint counters must be >= 0")
    if int(checkpoints_equal) > int(checkpoints):
        raise DaggerError(
            f"{checkpoints_equal} equal checkpoints out of {checkpoints} is impossible")
    rate = (round(int(checkpoints_equal) / int(checkpoints), 6)
            if int(checkpoints) else None)
    return {
        "schema": PROGRESS_SCHEMA,
        "checkpoint": str(checkpoint),
        "beta": check_beta(beta),
        "mode": str(mode),
        "decisions": int(decisions),
        "episodes": int(episodes),
        "episodes_failed": int(failures),
        "target_decisions": int(target),
        "target_met": bool(int(target) and int(decisions) >= int(target)),
        "partial": bool(not int(target) or int(decisions) < int(target)),
        "hash_checkpoints": int(checkpoints),
        "hash_checkpoints_equal": int(checkpoints_equal),
        "hash_alignment_rate": rate,
        "hash_alignment_complete": bool(rate == 1.0),
        "elapsed_s": round(float(elapsed_s), 3),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def default_engine() -> str:
    return os.environ.get(
        "V400_DAGGER_ENGINE",
        "/root/Touhou_RL/third_party/th08-rl/build/rl_hazard2/th08-modern")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("run", "verify", "plan"), default="run")
    parser.add_argument("--checkpoint", default=".tmp_v400_p3v1/checkpoint-v1.pt",
                        help="BC checkpoint that plays the student side of 尾-mixing")
    parser.add_argument("--out-dir", default=".tmp_v400_dagger/d0_extra",
                        help="D0-shaped root the shards are written to")
    parser.add_argument("--work-dir", default=".tmp_v400_dagger/work")
    parser.add_argument("--engine",
                        default=os.environ.get(
                            "V400_DAGGER_ENGINE",
                            "/root/Touhou_RL/third_party/th08-rl/build/rl_hazard2"))
    parser.add_argument("--expected-sha", default=None,
                        help="required sha256 of the engine binary (default: accept and "
                             "record; pass the D0 build hash to enforce it)")
    parser.add_argument("--data-template", default="/root/v400-data/game")
    parser.add_argument("--data-root", default="/root/v400-data/dagger")
    parser.add_argument("--stage", type=int, default=0, help="0-based; 0 = Stage 1 (S1N)")
    parser.add_argument("--difficulty", type=int, default=1, help="1 = normal")
    parser.add_argument("--team", type=int, default=0)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=92)
    parser.add_argument("--decisions", type=int, default=3500,
                        help="per-episode decision cap (stage 0 is ~3,300 decisions)")
    parser.add_argument("--max-frames", type=int, default=20000,
                        help="per-episode committed-frame cap (D0 used 20000)")
    parser.add_argument("--target-decisions", type=int, default=50000,
                        help="stop starting new episodes once this many decisions exist")
    parser.add_argument("--max-seconds", type=float, default=0.0,
                        help="wall budget for the run phase (0 = no budget)")
    parser.add_argument("--beta", type=float, default=0.2,
                        help="teacher-injection probability per decision")
    parser.add_argument("--mode", default="fast",
                        help="teacher profile used for the labels (D0 used 'fast')")
    parser.add_argument("--rng-seed", type=int, default=0,
                        help="seed of the 尾-injection RNG (per episode: rng_seed+seed)")
    parser.add_argument("--stop-on-death", action="store_true", default=True)
    parser.add_argument("--allow-play-on-death", dest="stop_on_death", action="store_false",
                        help="keep playing after the first death (the post-death states "
                             "are then part of the shard; the default stops, as D0 did)")
    parser.add_argument("--trim-transition", action="store_true", default=True,
                        help="drop the single decision whose commit crosses the stage "
                             "boundary (D0 semantics; the only legitimate mismatch)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--xvfb", choices=("auto", "off"), default="auto")
    parser.add_argument("--progress-every", type=float, default=30.0)
    parser.add_argument("--continue-on-error", action="store_true",
                        help="EXP-P3-006: record a failed episode and move to the next seed "
                             "instead of ending the run.  The shard of a failed episode is "
                             "still never written (fail-closed); only the campaign survives.")
    return parser


# ---------------------------------------------------------------------------
# engine helpers
# ---------------------------------------------------------------------------
def _tokens(reply: str) -> dict[str, str]:
    return {part.split("=", 1)[0]: part.split("=", 1)[1]
            for part in reply.split()[1:] if "=" in part}


def _maybe_reexec_under_xvfb(args) -> None:
    """The env role needs a private Xvfb (WSLg's ``DISPLAY=:0`` hangs before HELLO)."""

    if args.xvfb == "off" or os.environ.get("V400_DAGGER_XVFB") == "1":
        return
    anchor = sys.argv[0] if sys.argv and sys.argv[0] else __file__
    argv = ["xvfb-run", "-a", sys.executable, str(Path(anchor).resolve()), *sys.argv[1:]]
    env = dict(os.environ, V400_DAGGER_XVFB="1")
    os.execvpe("xvfb-run", argv, env)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonable(payload: Any) -> Any:
    """Coerce engine snapshots (numpy scalars, tuples) into JSON-safe values."""

    if isinstance(payload, dict):
        return {str(key): _jsonable(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_jsonable(value) for value in payload]
    if isinstance(payload, (np.integer,)):
        return int(payload)
    if isinstance(payload, (np.floating,)):
        return float(payload)
    if isinstance(payload, (np.bool_,)):
        return bool(payload)
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        return payload
    return str(payload)


def _shadow_checkpoint(engine) -> dict[str, Any]:
    """``HASH`` reply of the shadow engine (same shape as ``d0_dual._checkpoint``)."""

    import re

    fields = _tokens(engine.ask("HASH"))
    value = str(fields.get("hash", ""))
    if not re.match(r"^[0-9a-f]{16}$", value):
        raise DaggerError(f"shadow HASH reply is not an engine hash: {fields!r}")
    if "frames" not in fields:
        raise DaggerError(f"shadow HASH reply has no frame counter: {fields!r}")
    return {"hash": value, "frames": int(fields["frames"])}


def _env_frame(client) -> int:
    return int(client.request("FRAME").split("=", 1)[1])


def reset_command(args, seed: int) -> str:
    """``RESET <stage> <difficulty> <team> <seed> 0`` (D0's exact form)."""

    return (f"RESET {int(args.stage)} {int(args.difficulty)} {int(args.team)} "
            f"{int(seed)} 0")


def load_student(checkpoint: str, device: str):
    """The BC policy that supplies 80 % of the executed actions."""

    import torch

    from touhou_rl.v400.student.model import load_checkpoint

    model, payload = load_checkpoint(checkpoint, device=device)
    if device == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return model, payload


def student_action(model, sample: dict[str, Any]) -> tuple[int, int]:
    """Greedy ``(cand, bomb)`` from one obs-v1 sample (D0 window, no state)."""

    import torch

    play = torch.from_numpy(obs_spec.normalize_play(sample["play"])[None]).float()
    hud = torch.from_numpy(obs_spec.normalize_hud(sample["hud"])[None]).float()
    with torch.no_grad():
        cand, bomb = model.act(play, hud)
    return int(cand), int(bomb)


# ---------------------------------------------------------------------------
# one lockstep episode
# ---------------------------------------------------------------------------
class _TeacherSide:
    """The shadow engine plus the teacher's per-decision decision function."""

    def __init__(self, binary: str, data_dir: str, *, mode: str, timeout: float) -> None:
        from touhou_rl.v400.teacher import get_config
        from touhou_rl.v400.teacher.engine import TeacherEngine
        from touhou_rl.v400.teacher.verify import Consistency

        self.config = get_config(mode)
        if (int(self.config.replan_decisions) * int(self.config.decision_frames)) != FRAMES_PER_DECISION:
            raise DaggerError(
                f"teacher mode {mode!r} commits {self.config.replan_decisions} x "
                f"{self.config.decision_frames} frames per decision, but D0's schema is "
                f"{FRAMES_PER_DECISION}; the two datasets would not be comparable")
        self.engine = TeacherEngine(binary, data_dir, timeout=timeout)
        self.consistency = Consistency()

    def start(self, *, stage: int, difficulty: int, team: int, seed: int) -> dict[str, Any]:
        self.engine.start()
        self.engine.hello()
        info = self.engine.reset(stage=stage, difficulty=difficulty, team=team, seed=seed)
        if str(info.get("ready")) != "1":
            raise DaggerError(f"shadow RESET did not become ready: {info!r}")
        return info

    def capabilities(self) -> dict[str, Any]:
        caps = self.engine.probe_capabilities()
        caps["child_hazards"] = self.engine.probe_child_hazards()
        if not caps["child_hazards"].get("available"):
            raise DaggerError(
                "engine cannot serve CHILD <h> HAZARDS; the teacher cannot label on the "
                f"shadow process: {caps['child_hazards']!r}")
        return caps

    def decide(self, *, committed, frame: int, stage: int, difficulty: int,
               shoot_release: bool):
        from touhou_rl.v400.teacher.decide import decide

        return decide(
            self.engine,
            self.config,
            committed=committed,
            frame=int(frame),
            stage=int(stage),
            difficulty=int(difficulty),
            consistency=self.consistency,
            record_source="child",
            shoot_release=bool(shoot_release),
        )

    def close(self) -> None:
        try:
            self.engine.close()
        except Exception:  # noqa: BLE001 - closing must never mask the real error
            pass


def run_episode(args, *, seed: int, key: str, model) -> dict[str, Any]:
    """One 尾-mixed lockstep episode; returns an episode record (never a shard)."""

    from touhou_rl.v400.gym.shm import PIXELS_OFFSET, SharedBlock
    from touhou_rl.v400.labels.runner import ControlClient
    from touhou_rl.v400.teacher.hazard_tape import (  # noqa: F401 - capability probe
        HazardExportUnavailable,  # noqa: F401
    )
    from touhou_rl.v400.teacher.obs import (
        DecisionObservationBuffer,
        RenderFrame,
        TeacherObsShardWriter,
        require_rendered,
    )
    from touhou_rl.v400.teacher.shoot import commit_block, commands as step_commands, release_due


    started = time.monotonic()
    engine_path = Path(str(args.engine))
    binary = str(engine_path / "th08-modern") if engine_path.is_dir() else str(args.engine)
    if not Path(binary).is_file():
        raise DaggerError(
            f"engine binary not found: {binary} (pass --engine as the build directory or "
            "the binary itself)")
    sha = dual.sha256_file(binary)
    if args.expected_sha and sha.lower() != str(args.expected_sha).lower():
        raise DaggerError(f"{binary}: sha256 {sha} != expected {args.expected_sha}")

    env_data = dual.prepare_data_dir(Path(args.data_template), Path(args.data_root) / f"env-{seed}")
    shadow_data = dual.prepare_data_dir(Path(args.data_template),
                                        Path(args.data_root) / f"shadow-{seed}")
    work_dir = Path(args.work_dir) / key
    work_dir.mkdir(parents=True, exist_ok=True)
    epilogue_path = work_dir / "episode.json"
    if epilogue_path.is_file():
        record = _read_json(epilogue_path)
        if record.get("status") == "ok":
            record["resumed"] = True
            return record

    shm_name = f"v400dagger-{os.getpid()}-{seed}"
    log_path = work_dir / "env_engine.stderr.log"
    rng = np.random.default_rng(int(args.rng_seed) + int(seed))
    beta = check_beta(args.beta)

    client = None
    block = None
    writer = None
    teacher = _TeacherSide(binary, shadow_data["data_dir"], mode=args.mode,
                           timeout=args.timeout)
    buffer = DecisionObservationBuffer()
    pairs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    frames = 0
    decisions = 0
    injections = 0
    teacher_available = 0
    agree = 0
    stage_start = 0
    stage_changed = False
    deaths = 0.0
    deaths_start = 0.0
    stop_reason = "decisions"
    exit_code = None
    error: str | None = None
    trim_info: dict[str, Any] | None = None
    render_stats: list[dict[str, float]] = []
    draw_sum = 0.0
    build_ms = 0.0
    commit_ms = 0.0
    align_ms = 0.0
    caps: dict[str, Any] = {}
    try:
        client = ControlClient(binary, env_data["data_dir"], role="env",
                              timeout=args.timeout, stderr_path=log_path)
        client.start()
        hello = _tokens(client.request("HELLO"))
        if hello.get("role") != "env":
            raise DaggerError(f"HELLO did not confirm the env role: {hello!r}")
        client.request(f"SHM /{shm_name}")
        if _tokens(client.request("READBACK 1")).get("readback") != "1":
            raise DaggerError("READBACK 1 was refused; the env side cannot produce obs-v1")
        block = SharedBlock(shm_name)
        expected_size = PIXELS_OFFSET + 640 * 480 * 4
        if block.size != expected_size:
            raise DaggerError(f"shared block is {block.size} B, expected {expected_size}")
        reset = _tokens(client.request(reset_command(args, seed), timeout=args.timeout))
        if reset.get("ready") != "1":
            raise DaggerError(f"env RESET did not become ready: {reset!r}")
        if _env_frame(client) != 0:
            raise DaggerError("env RESET left the frame counter non-zero")
        teacher.start(stage=int(args.stage), difficulty=int(args.difficulty),
                      team=int(args.team), seed=int(seed))
        caps = teacher.capabilities()
        if int(teacher.engine.frame()) != 0:
            raise DaggerError("shadow RESET left the frame counter non-zero")
        # Episode-start padding: the last render RESET left in the block.
        play, hud, header = dual._read_frame(block, f"post-RESET seed {seed}")
        buffer.reset(RenderFrame(frame=int(header["frame"]), play=play, hud=hud))
        render_stats.append(require_rendered(play, hud, where=f"seed {seed}"))
        state = _tokens(client.request("STATE"))
        stage_start = int(state.get("stage", args.stage))
        deaths_start = float(state.get("deaths", 0.0) or 0.0)
        deaths = deaths_start
        committed: list[int] = []
        while decisions < int(args.decisions) and frames < int(args.max_frames):
            decision_frame = decisions * FRAMES_PER_DECISION
            build_started = time.perf_counter()
            sample = buffer.stack_for(decision_frame, episode_start=(decisions == 0))
            student_cand, student_bomb = student_action(model, sample)
            build_ms += (time.perf_counter() - build_started) * 1000.0

            # --- checkpoint before the teacher moves anything -----------------
            align_started = time.perf_counter()
            env_pre = dual._checkpoint(client)
            shadow_pre = _shadow_checkpoint(teacher.engine)
            if int(env_pre["frames"]) != decision_frame:
                raise DaggerError(
                    f"decision {decisions}: env frame {env_pre['frames']} != {decision_frame}")
            if int(teacher.engine.frame()) != decision_frame:
                raise DaggerError(
                    f"decision {decisions}: shadow frame {teacher.engine.frame()} != "
                    f"{decision_frame}")
            pairs.append(align_pair(decisions, "pre", decision_frame, shadow_pre, env_pre))
            if not pairs[-1]["equal"]:
                raise AlignError(
                    f"decision {decisions} pre: shadow={shadow_pre['hash']} "
                    f"env={env_pre['hash']} 鈥?the pair diverged before the decision")
            align_ms += (time.perf_counter() - align_started) * 1000.0

            # --- the DAgger query: teacher labels *this* state ----------------
            release = release_due(decisions, teacher.config.shoot_pulse_period)
            try:
                decision = teacher.decide(committed=list(committed), frame=decision_frame,
                                          stage=int(args.stage),
                                          difficulty=int(args.difficulty),
                                          shoot_release=release)
            except Exception as exc:  # noqa: BLE001 - a state the teacher cannot label
                # The student visits states the teacher's own trajectory never does.
                # On some of them the child hazard recorder refuses (measured:
                # ``ERR player-speed-unavailable``); that decision point cannot be
                # labelled, so the segment ends here and keeps every row collected
                # so far (all of them HASH-aligned and teacher-labelled).
                unlabelable = f"{type(exc).__name__}: {exc}"
                stop_reason = "teacher_unavailable"
                print(f"[dagger] {key}: decision {decisions} unlabelable ({unlabelable}); "
                      f"ending the segment with {decisions} rows", flush=True)
                break
            label = decision.label
            replan = int(teacher.config.replan_decisions)
            teacher_cand = None
            teacher_slots: list[int] = []
            if decision.chosen is not None:
                teacher_slots = [int(value) for value in decision.chosen.slots]
                if teacher_slots:
                    teacher_cand = teacher_slots[0]
            if teacher_cand is not None:
                teacher_available += 1
                if teacher_cand == student_cand:
                    agree += 1
            draw = float(rng.random())
            draw_sum += draw
            injected = inject_decision(draw, beta, teacher_cand)
            if injected:
                injections += 1
                # The teacher's own plan for this decision: slot 0 first, then the
                # rest of its committed slots (the same prefix the teacher CLI uses).
                executed = teacher_slots[:replan]
                while len(executed) < replan:
                    executed.append(executed[-1])
                cand, bomb = int(executed[0]), int(round(float(label.bomb)))
            else:
                executed = [int(student_cand)] * replan
                cand, bomb = int(student_cand), int(student_bomb)
            # The executed block is what the teacher's record walk should assume for
            # the next decision's "remaining plan" (a prefix heuristic only; the
            # recording always forks from the live state).
            committed = list(executed)

            # --- commit the identical block on both engines -------------------
            block_cmds = commit_block(
                list(executed),
                bomb=int(bomb),
                decision_frames=int(teacher.config.decision_frames),
                release_shoot=release,
            )
            expect_frames = sum(int(count) for _, count in block_cmds)
            if expect_frames != FRAMES_PER_DECISION:
                raise DaggerError(
                    f"decision {decisions}: block commits {expect_frames} frames, schema is "
                    f"{FRAMES_PER_DECISION}")
            step_frames: list[int] = []
            commit_started = time.perf_counter()
            for mask, count in block_cmds:
                # One STEP per *frame*: four distinct renders are the only way to
                # build the 4-consecutive-frame obs stack (a STEP of 2 leaves only
                # the last one in the shared block).
                for _ in range(int(count)):
                    reply = _tokens(client.request(f"STEP {int(mask)} 1"))
                    if int(reply.get("advanced", 0)) != 1:
                        raise DaggerError(
                            f"decision {decisions}: env STEP advanced="
                            f"{reply.get('advanced')} exit={reply.get('exit')}")
                    play, hud, header = dual._read_frame(block, f"frame {frames}")
                    render_stats.append(require_rendered(play, hud, where=f"frame {frames}"))
                    buffer.push(RenderFrame(frame=int(header["frame"]), play=play, hud=hud))
                    step_frames.append(int(mask))
                    frames += 1
            if len(step_frames) != expect_frames:
                raise DaggerError(
                    f"decision {decisions}: env stepped {len(step_frames)} frames, the block "
                    f"declares {expect_frames}")
            shadow_steps = teacher.engine.ask_many(step_commands(block_cmds))
            shadow_advanced = sum(int(_tokens(reply).get("advanced", 0)) for reply in shadow_steps)
            if shadow_advanced != expect_frames:
                raise DaggerError(
                    f"decision {decisions}: shadow advanced {shadow_advanced} of "
                    f"{expect_frames} frames")
            commit_ms += (time.perf_counter() - commit_started) * 1000.0

            # --- checkpoint after the commit ---------------------------------
            align_started = time.perf_counter()
            env_post = dual._checkpoint(client)
            shadow_post = _shadow_checkpoint(teacher.engine)
            pairs.append(align_pair(decisions, "post", decision_frame + FRAMES_PER_DECISION,
                                    shadow_post, env_post))
            align_ms += (time.perf_counter() - align_started) * 1000.0

            state = _tokens(client.request("STATE"))
            deaths = float(state.get("deaths", deaths) or deaths)
            want_stage = int(state.get("stage", args.stage))
            if want_stage != stage_start:
                stage_changed = True

            row = {
                "decision": decisions,
                "frame": decision_frame,
                "hash_pre": shadow_pre["hash"],
                "hash_post": shadow_post["hash"],
                "env_hash_pre": env_pre["hash"],
                "env_hash_post": env_post["hash"],
                "step_frames": step_frames,
                "scores": [float(value) for value in np.asarray(label.scores).tolist()],
                "verified_mask": [bool(value) for value in
                                  np.asarray(label.verified_mask).tolist()],
                "chosen_plan": None if decision.chosen is None else
                               decision.chosen.as_dict(),
                "margin": None if decision.chosen is None else float(label.margin),
                "bomb": int(round(float(label.bomb))),
                "stage": int(label.stage),
                "difficulty": int(label.difficulty),
                "frame_event_flags": _jsonable(decision.events_frame),
                "shoot_release": bool(release),
                "bomb_death_available": bool(decision.bomb_death_available),
                "dagger": {
                    "beta": beta,
                    "draw": round(draw, 6),
                    "injected": bool(injected),
                    "teacher_available": teacher_cand is not None,
                    "student_cand": int(student_cand),
                    "student_bomb": int(student_bomb),
                    "teacher_cand": None if teacher_cand is None else int(teacher_cand),
                    "teacher_slots": [int(value) for value in teacher_slots[:replan]],
                    "executed_slots": [int(value) for value in executed],
                    "executed_cand": int(cand),
                    "executed_bomb": int(bomb),
                    "agreement": (teacher_cand is not None and teacher_cand == student_cand),
                    "decisions_aligned_at_decision_point": True,
                },
            }
            rows.append(row)
            samples.append(sample)
            decisions += 1
            if not pairs[-1]["equal"]:
                # Only the stage-clear commit may disagree (D0's documented
                # exclusion): prove it from the pairs, drop that one decision.
                if args.trim_transition and stage_changed and decisions >= 2:
                    info = dual._transition_info_from_pairs(pairs, decisions, {
                        "stop_reason": "stage_changed", "stage_clear": True})
                    if info is not None:
                        trim_info = info
                        rows = rows[:-1]
                        samples = samples[:-1]
                        decisions -= 1
                        stop_reason = "stage_clear"
                        break
                raise AlignError(
                    f"decision {decisions - 1} post: shadow={shadow_post['hash']} "
                    f"env={env_post['hash']} at frame {decision_frame + FRAMES_PER_DECISION} "
                    "- the pair is not the same trajectory")
            if deaths > deaths_start and bool(args.stop_on_death):
                stop_reason = "death"
                break
            if stage_changed:
                stop_reason = "stage_changed"
                break
        else:
            stop_reason = "frame_cap" if frames >= int(args.max_frames) else "decisions"

        # --- write the D0-shaped shard -----------------------------------
        if not rows:
            raise DaggerError(f"{key}: the episode produced no decisions")
        meta = {
            "stage": int(args.stage),
            "difficulty": int(args.difficulty),
            "team": int(args.team),
            "seed": int(seed),
            "mode": args.mode,
            "frames_per_decision": FRAMES_PER_DECISION,
            "generated_by": "tools/rl/v400_dagger_collect.py (EXP-P3-003)",
            "decisions": len(rows),
            "decisions_requested": int(args.decisions),
            "stopped_early": stop_reason in ("stage_changed", "death"),
            "stop_reason": stop_reason,
            "stage_clear": bool(stage_changed),
            "frames_committed": int(frames),
            "deaths_observed": float(deaths - deaths_start),
            "transition_trim": trim_info,
            "label_channel": "shadow",
            "pixel_channel": "env",
            "alignment_method": "hash-checkpoint",
            "dagger": {
                "round": 1,
                "beta": beta,
                "rng_seed": int(args.rng_seed) + int(seed),
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": dual.sha256_file(args.checkpoint),
                "policy": "80% BC v1 greedy action + 20% teacher chosen action "
                          "(seeded per-decision draw); labels = 100% teacher scores[18] "
                          "on the executed state",
                "injections": int(injections),
                "teacher_available": int(teacher_available),
                "agreement": int(agree),
                "executed_teacher_rate": round(injections / max(1, len(rows)), 6),
                "mean_draw": round(draw_sum / max(1, len(rows)), 6),
            },
            "alignment_extra": {
                "pixel_source": "env_live",
                "label_source": "shadow_teacher_lockstep",
                "hash_alignment": "hash-checkpoint",
                "alignment_method": "hash-checkpoint",
                "hash_checkpoints": {
                    "method": "hash-checkpoint",
                    "boundaries": len(pairs),
                    "decisions": len(rows),
                    "checkpoints_equal": sum(1 for row in pairs if row["equal"]),
                    "checkpoints_mismatched": sum(1 for row in pairs if not row["equal"]),
                    "trimmed_transition": trim_info,
                },
            },
            "engine_env": {"binary": binary, "sha256": sha, "role": "env"},
            "engine_shadow": {"binary": binary, "sha256": sha, "role": "shadow"},
            "reset_command": reset_command(args, seed),
            "capabilities": {key: caps.get(key) for key in ("child_hazards",)},
        }
        writer = TeacherObsShardWriter(Path(args.out_dir), key=key, meta=meta)
        writer.render_stats = render_stats
        for index, row in enumerate(rows):
            writer.add(
                decision=index,
                decision_frame=int(row["frame"]),
                label=dual._ShadowLabel(row),
                sample=samples[index],
                extra={
                    "channel": "dagger-lockstep",
                    "pixel_channel": "env_live",
                    "hash_pre": row["hash_pre"],
                    "hash_post": row["hash_post"],
                    "env_hash_pre": row["env_hash_pre"],
                    "env_hash_post": row["env_hash_post"],
                    "step_frames": list(row["step_frames"]),
                    "frame_event_flags": row["frame_event_flags"],
                    "shoot_release": row["shoot_release"],
                    "bomb_death_available": row["bomb_death_available"],
                    "dagger": row["dagger"],
                },
            )
        summary = writer.finalize(report={
            "dagger_round": 1,
            "hash_alignment": meta["alignment_extra"]["hash_checkpoints"],
            "injections": int(injections),
            "teacher_available": int(teacher_available),
            "agreement": int(agree),
            "wall_s": round(time.monotonic() - started, 3),
        })
        exit_code = 0
        record = {
            "schema": EPISODE_SCHEMA,
            "shard": key,
            "seed": int(seed),
            "decisions": int(len(rows)),
            "frames": int(frames),
            "injections": int(injections),
            "teacher_available": int(teacher_available),
            "agreement": int(agree),
            "executed_teacher_rate": round(injections / max(1, len(rows)), 6),
            "stop_reason": stop_reason,
            "stage_clear": bool(stage_changed),
            "deaths": float(deaths - deaths_start),
            "transition_trim": trim_info,
            "checkpoints": len(pairs),
            "checkpoints_equal": sum(1 for row in pairs if row["equal"]),
            "checkpoint_mismatched": sum(1 for row in pairs if not row["equal"]),
            "beta": beta,
            "rng_seed": int(args.rng_seed) + int(seed),
            "wall_s": round(time.monotonic() - started, 3),
            "decisions_per_s": round(len(rows) / max(1e-9, time.monotonic() - started), 3),
            "build_ms": round(build_ms, 1),
            "commit_ms": round(commit_ms, 1),
            "align_ms": round(align_ms, 1),
            "shard_summary": summary,
        }
    except BaseException as exc:  # noqa: BLE001 - a failed episode is a recorded episode
        error = f"{type(exc).__name__}: {exc}"
        if writer is not None:
            try:
                writer.abort(exc)
            except Exception:  # noqa: BLE001
                pass
        record = {
            "schema": EPISODE_SCHEMA,
            "shard": key,
            "seed": int(seed),
            "decisions": int(len(rows)),
            "frames": int(frames),
            "error": error,
            "stop_reason": stop_reason,
            "wall_s": round(time.monotonic() - started, 3),
            "status": "failed",
        }
    finally:
        if block is not None:
            try:
                block.close()
            except Exception:  # noqa: BLE001
                pass
        if client is not None:
            try:
                client.request("QUIT", timeout=20.0)
            except Exception:  # noqa: BLE001
                pass
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
            exit_code = client._proc.returncode if client._proc is not None else None
        teacher.close()
        try:
            from touhou_rl.v400.gym.shm import segment_path
            segment_path(shm_name).unlink()
        except (OSError, ValueError):
            pass
    record["exit_code"] = exit_code
    record["status"] = "failed" if error else "ok"
    _write_json(epilogue_path, record)
    return record


# ---------------------------------------------------------------------------
# verify phase
# ---------------------------------------------------------------------------
def verify(args) -> dict[str, Any]:
    """Load every finished shard through the frozen D0 loader (fail-closed)."""

    from touhou_rl.v400.student.dataset import load_teacher_samples, load_teacher_shard

    root = Path(args.out_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise DaggerError(f"{root}: no manifest.json yet")
    manifest = _read_json(manifest_path)
    label_paths = []
    shards = []
    for row in manifest.get("shards") or []:
        key = str(row.get("shard"))
        shard_dir = root / "shards" / key
        if not (shard_dir / "shard.json").is_file():
            continue
        shard = load_teacher_shard(shard_dir)
        shards.append({"shard": key, "decisions": len(shard),
                       "stage": shard.meta.get("stage"), "seed": shard.meta.get("seed"),
                       "stop_reason": shard.meta.get("stop_reason"),
                       "dagger": shard.meta.get("dagger")})
        label_paths.append(shard_dir / "labels.jsonl")
    if not label_paths:
        raise DaggerError(f"{root}: no loadable shard")
    loaded = load_teacher_samples(label_paths, min_samples=1, obs_root=root,
                                  require_observations=True)
    scores = np.asarray(loaded["samples"][0]["scores"])
    report = {
        "schema": "v400-dagger-verify-1",
        "root": str(root),
        "shards": len(shards),
        "decisions": int(loaded["count"]),
        "obs_spec_version": loaded["obs_spec_version"],
        "obs_spec_digest": loaded["obs_spec_digest"],
        "alignment": loaded.get("alignment"),
        "obs_unresolved": len(loaded.get("obs_unresolved") or []),
        "scores_shape": list(scores.shape),
        "verified_mask_shape": list(np.asarray(loaded["samples"][0]["verified_mask"]).shape),
        "per_shard": shards,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "plan":
        print(json.dumps({"schema": "v400-dagger-plan-1",
                          "episodes": episode_plan(args),
                          "beta": check_beta(args.beta),
                          "out_dir": args.out_dir}, indent=2, ensure_ascii=False))
        return EXIT_OK
    if args.phase == "verify":
        report = verify(args)
        _write_json(Path(args.work_dir) / "verify.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return EXIT_OK

    _maybe_reexec_under_xvfb(args)
    model, payload = load_student(args.checkpoint, args.device)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    plan = episode_plan(args)
    if any(row["eval_overlap"] for row in plan):
        raise SystemExit("DAgger seeds must not overlap the 1000-1004 evaluation seeds")
    total = 0
    started = time.monotonic()
    progress_path = Path(args.work_dir) / "progress.json"

    episodes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    checkpoints = 0
    checkpoints_equal = 0
    for index, entry in enumerate(plan):
        if args.target_decisions and total >= int(args.target_decisions):
            break
        if args.max_seconds and (time.monotonic() - started) >= float(args.max_seconds):
            print(f"[dagger] wall budget {args.max_seconds:.0f}s reached; stopping", flush=True)
            break
        shard_dir = out_root / "shards" / entry["key"]
        done = dual.shard_resume_report(shard_dir)
        if done is not None:
            total += int(done["decisions"])
            episodes.append({"shard": entry["key"], "seed": entry["seed"],
                             "decisions": int(done["decisions"]), "resumed": True})
            print(f"[dagger] {entry['key']}: resumed ({done['decisions']} decisions)", flush=True)
            continue
        record = run_episode(args, seed=int(entry["seed"]), key=entry["key"], model=model)
        checkpoints += int(record.get("checkpoints") or 0)
        checkpoints_equal += int(record.get("checkpoints_equal") or 0)
        episodes.append({"shard": entry["key"], "seed": entry["seed"],
                         "decisions": int(record.get("decisions") or 0),
                         "status": record.get("status"),
                         "stop_reason": record.get("stop_reason"),
                         "injections": record.get("injections"),
                         "executed_teacher_rate": record.get("executed_teacher_rate"),
                         "wall_s": record.get("wall_s"),
                         "error": record.get("error")})
        total += int(record.get("decisions") or 0)
        elapsed = time.monotonic() - started
        progress = campaign_summary(
            decisions=int(total), target=int(args.target_decisions),
            checkpoints=int(checkpoints), checkpoints_equal=int(checkpoints_equal),
            elapsed_s=elapsed, episodes=index + 1, failures=len(failures),
            beta=args.beta, mode=str(args.mode), checkpoint=str(args.checkpoint))
        progress.update({
            "checkpoint_sha256": dual.sha256_file(args.checkpoint),
            "episodes_done": index + 1,
            "episodes_total": len(plan),
            "decisions_per_s": round(total / max(1e-9, elapsed), 4),
            "failed": failures,
            "episodes": episodes,
        })
        _write_json(progress_path, progress)
        print(f"[dagger] {entry['key']}: {record.get('decisions')} decisions, "
              f"{record.get('injections')} injections, stop={record.get('stop_reason')}, "
              f"wall={record.get('wall_s')}s | total {total}/{args.target_decisions} "
              f"({progress['decisions_per_s']} dec/s)", flush=True)
        if record.get("status") != "ok":
            failures.append({"shard": entry["key"], "seed": entry["seed"],
                             "error": record.get("error"),
                             "decisions": int(record.get("decisions") or 0)})
            if args.continue_on_error:
                # The failed shard was never written (Writer.abort).  Skip the seed and
                # keep the campaign alive; the failure stays in progress.json.
                print(f"[dagger] episode failed (kept running): {record.get('error')}",
                      flush=True)
                continue
            print(f"[dagger] stopping on failure: {record.get('error')}", flush=True)
            return EXIT_FAIL_CLOSED
    summary = campaign_summary(
        decisions=int(total), target=int(args.target_decisions),
        checkpoints=int(checkpoints), checkpoints_equal=int(checkpoints_equal),
        elapsed_s=time.monotonic() - started, episodes=len(episodes),
        failures=len(failures), beta=args.beta, mode=str(args.mode),
        checkpoint=str(args.checkpoint))
    summary["failed"] = failures
    _write_json(Path(args.work_dir) / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return EXIT_OK if total > 0 else EXIT_FAIL_CLOSED


if __name__ == "__main__":
    raise SystemExit(main())
