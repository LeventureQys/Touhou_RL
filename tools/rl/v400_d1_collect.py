#!/usr/bin/env python3
"""D1 teacher dataset collector (task book T3 / EXP-P3-012): obs-v3, stages 0-7.

What D1 is
----------
The spec-size teacher dataset of T3: ``stage_index 0..7``, Normal, team 0, **>= 2,000,000
committed frames**, **>= 150,000 frames per stage**, every decision stored as an obs-v3
sample (4x112x96 play stack + HUD + 8x18 action history) next to the teacher label
(``scores[18]``, ``verified_mask[18]``, ``bomb``, ``chosen_plan``, ``margin``) and the
per-frame masks, with ``hash_pre``/``hash_post`` at every decision boundary.
``shard.json`` carries ``obs_spec_version=obs-v3`` + the frozen digest, ``stage_index``,
``start_mode``, ``frames`` and ``decisions``.

Nothing here fabricates a frame.  A stage whose teacher dies at frame D can only sustain
``D`` frames per episode; if a plan cannot reach the per-stage target inside its seed
budget the shortfall is written down as ``coverage_shortfall`` instead of being padded.

Phases
------
``probe``   one episode per stage, **to death or to the frame cap**, labels only: the
            reachability measurement the plan is built on (frames sustained, death
            frame, cleared or not).
``plan``    reachability + frame targets -> a list of immutable shard specs.  A shard
            has ``min(decisions_per_shard, reachable/4)`` decisions; the shard count per
            stage is ``ceil(per_stage_target / frames_per_shard)``.
            ``--plan-limit-per-stage`` caps the list (the small-scale validation); the
            uncapped plan is the full D1 plan.
``shard``   (internal, one shard) two channels over the same ``RESET`` and the same
            per-frame mask tape.  A = ``v400_teacher.py --labels-only`` in the shadow
            role (deterministic, CPU-only, no GL).  B = this module's own replay loop on
            an ``env``-role engine (``READBACK 1`` + shared pixels): it replays the same
            masks, reads the 4 renders ``d-4..d-1`` of every decision and compares the
            engine's 25-field ``HASH`` on both sides of every decision with channel A.
            One differing checkpoint aborts the shard (``EXIT_ALIGN``), leaves
            ``shards/.<key>.partial`` as evidence and never creates a shard.
``run``     ``--workers K`` (hard-capped at 4: this task's WSL engine budget) runs the
            plan's pending shards, one subprocess per shard = at most one engine at a
            time (A then B are sequential *inside* a shard).  Finished shards are
            skipped (resume); failed ones keep their ``.partial``.
``gate``    per-shard report + the HASH alignment rate (must be 100 %) + the
            ``obs_v3_dataset.load_teacher_samples_v3(require_observations=True)`` gate
            over *every* label row of the plan.
``report``  per-stage table + the full-size plan (shards, wall clock at N engines from
            the measured frames/s, disk against D-05).

Start modes
-----------
``shard``       enter the stage by seed (plain ``RESET``).
``trajectory``  ``TrajectoryStart``: ``RESET`` the recorded run's own seed, feed the
                recorded per-frame tape up to ``target_frame`` and hand control over only
                after the engine hash equals the recorded hash
                (``touhou_rl/v400/gym/trajectory.py``).  Channel B fast-forwards with
                :func:`replay_to_frame` and applies the last 4 recorded masks frame by
                frame to obtain the ``T-4..T-1`` renders; channel A must reproduce the
                recorded tape decision by decision (mask *and* hash) up to the take-over
                and its prefix is discarded.  The shard's decision frames are re-based to
                0 (the loader's ``frame == index * frames_per_decision`` arithmetic); the
                absolute take-over frame and hash are in ``shard.json`` as
                ``start_frame`` / ``trajectory``.

Run inside WSL (the engine is a Linux ELF)::

    env -u DISPLAY /root/v400-venv/bin/python tools/rl/v400_d1_collect.py \\
        --phase probe --stages 0,1,3,2,4,5,6,7 --workers 4
    ... --phase plan --plan-name smoke --plan-limit-per-stage 1 --decisions-per-shard 500
    ... --phase run --plan-name smoke
    ... --phase gate --plan-name smoke
    ... --phase report --plan-name smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

if __package__ in (None, ""):  # allow ``python tools/rl/v400_d1_collect.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from touhou_rl.v400.gym.env import DIFFICULTIES, StageStart, Th08Env  # noqa: E402
from touhou_rl.v400.gym.trajectory import (  # noqa: E402
    TrajectoryStartError,
    load_d0_shard,
    replay_to_frame,
    sample_target_frame,
)
from touhou_rl.v400.student import obs_spec_v3  # noqa: E402
from touhou_rl.v400.teacher import config as teacher_config  # noqa: E402
from touhou_rl.v400.teacher.engine import sha256_file  # noqa: E402
from touhou_rl.v400.teacher.obs import (  # noqa: E402
    FROZEN_OBS_SPEC_V3_DIGEST,
    DecisionObservationBuffer,
    ObservationError,
    RenderFrame,
    TeacherObsShardWriter,
    check_obs_version,
    decision_window,
    obs_version_spec,
    require_rendered,
)

EXIT_OK = 0
EXIT_FAIL_CLOSED = 3
EXIT_ALIGN = 4

#: 4 logic frames per teacher decision (``replan_decisions 2 x decision_frames 2``).
FRAMES_PER_DECISION = 4

#: T3's stage list, cheapest first: ``0,1,3`` are the stages the teacher clears,
#: ``2,4,5,6,7`` the ones it has measured deaths on.
DEFAULT_STAGES = "0,1,3,2,4,5,6,7"

#: The unified build (T7.2), frozen for this task.
DEFAULT_ENGINE = "/root/Touhou_RL/third_party/th08-rl/build/rl_v3"
DEFAULT_ENGINE_SHA = "691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6"

#: D0 corpus (Stage 1 only).  Stages 1-7 use D1's own finished shard-start shards as
#: their trajectory corpus; :func:`index_corpora` picks those up automatically.
DEFAULT_CORPUS_ROOT = ".tmp_v400_d0/d0/shards"

#: Stage 1's boss section in the D0 corpus (EXP-P3-012a); used only as the
#: ``boss_start`` of the late/boss sampling bias.
DEFAULT_BOSS_START = 2936

PLAN_SCHEMA = "v400-d1-plan-1"
REACHABILITY_SCHEMA = "v400-d1-reachability-1"
SHARD_SCHEMA_D1 = "v400-d1-shard-1"
GATE_SCHEMA = "v400-d1-gate-1"
REPORT_SCHEMA = "v400-d1-report-1"

#: D-05: disk-increase budget of the whole v4.0.0 programme.
DISK_BUDGET_BYTES = 300 * 1000 ** 3
#: Ask the负责人 at 80 % of D-05 (task book §2 rule 10).
DISK_ALERT_FRACTION = 0.80

HEX = frozenset("0123456789abcdef")


class D1Error(RuntimeError):
    """A precondition, a plan invariant or the channel contract failed."""


class AlignmentError(D1Error):
    """A HASH checkpoint disagrees between channel A and channel B."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: str | Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def parse_stage_list(text: str) -> list[int]:
    """``"0,1,3,2,4,5,6,7"`` -> ``[0, 1, 3, 2, 4, 5, 6, 7]`` (order kept)."""

    stages: list[int] = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        stage = int(part)
        if not 0 <= stage <= 8:
            raise D1Error(f"stage index out of range: {stage} (engine stages are 0..8)")
        if stage not in stages:
            stages.append(stage)
    if not stages:
        raise D1Error("no stages given")
    return stages


def parse_seed_list(text: str) -> list[int]:
    """``"1-8,20"`` -> ``[1..8, 20]``."""

    seeds: list[int] = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part[1:]:
            low, high = part.split("-", 1)
            if int(high) < int(low):
                raise D1Error(f"seed range is reversed: {part}")
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise D1Error("no seeds given")
    return seeds


def parse_start_mix(text: str) -> list[str]:
    """``"shard,trajectory"`` -> the start-mode cycle a plan assigns shard by shard."""

    modes: list[str] = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if part not in ("shard", "trajectory"):
            raise D1Error(f"unknown start mode {part!r} (known: shard, trajectory)")
        if part not in modes:
            modes.append(part)
    if not modes:
        raise D1Error("empty --start-mix")
    return modes


def difficulty_name(value: int | str) -> str:
    text = str(value).strip().lower()
    if text in DIFFICULTIES:
        return text
    number = int(text)
    if not 0 <= number < len(DIFFICULTIES):
        raise D1Error(f"difficulty out of range: {value}")
    return DIFFICULTIES[number]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


_DUAL = None


def dual():
    """Import ``tools/rl/v400_d0_dual.py`` (reuse its pieces, never modify it)."""

    global _DUAL
    if _DUAL is None:
        path = Path(__file__).resolve().parent / "v400_d0_dual.py"
        spec = importlib.util.spec_from_file_location("v400_d0_dual", path)
        if spec is None or spec.loader is None:  # pragma: no cover - import machinery
            raise D1Error(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _DUAL = module
    return _DUAL


def work_dir_of(args) -> Path:
    return Path(getattr(args, "work_dir", None) or repo_root() / ".tmp_v400_d1")


def out_dir_of(args) -> Path:
    value = getattr(args, "out_dir", None)
    return Path(value) if value else work_dir_of(args) / "d1"


def data_root_of(args, key: str) -> str:
    prefix = getattr(args, "data_root_prefix", None) or "/root/v400-data/d1"
    return f"{prefix}/{key}"


# ---------------------------------------------------------------------------
# channel A: shadow teacher labels
# ---------------------------------------------------------------------------
def _channel_a_argv(spec: dict[str, Any], *, decisions: int, max_frames: int,
                    work_dir: Path, data_root: str) -> SimpleNamespace:
    return SimpleNamespace(
        decisions=int(decisions),
        stage=int(spec["stage"]),
        difficulty=int(spec["difficulty"]),
        team=int(spec.get("team", 0)),
        seed=int(spec["seed"]),
        mode=str(spec.get("mode", "fast")),
        labels_jsonl=str(Path(work_dir) / "labels.jsonl"),
        work_dir=str(work_dir),
        teacher_tool=str(spec.get("teacher_tool")
                         or repo_root() / "tools" / "rl" / "v400_teacher.py"),
        data_root=data_root,
        data_template=str(spec.get("data_template") or "/root/v400-data/game"),
        engine=str(spec["engine"]),
        expected_sha=str(spec["expected_sha"]),
        binary=spec.get("binary"),
        repo_root=str(spec.get("repo_root") or repo_root()),
        max_frames=int(max_frames),
        stop_on_death=True,
        allow_early_stop=True,
        quiet=True,
    )


def run_channel_a(spec: dict[str, Any], *, decisions: int, max_frames: int,
                  work_dir: Path, data_root: str) -> dict[str, Any]:
    """Run the teacher in the shadow role; return its report plus the label rows."""

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    namespace = _channel_a_argv(spec, decisions=decisions, max_frames=max_frames,
                                work_dir=work_dir, data_root=data_root)
    report = dual().run_labels(namespace)
    rows = dual().load_labels(Path(namespace.labels_jsonl))
    if len(rows) != int(report["decisions"]):
        raise D1Error(
            f"channel A: {len(rows)} usable label rows vs report {report['decisions']}")
    return {"report": report, "labels": rows, "labels_jsonl": str(namespace.labels_jsonl)}


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def probe_stage(spec: dict[str, Any], *, cap_frames: int, work_dir: Path,
                data_root: str) -> dict[str, Any]:
    """One episode: to death or to ``cap_frames``.  Records what the teacher sustains."""

    decisions_cap = max(1, int(cap_frames) // FRAMES_PER_DECISION)
    began = time.monotonic()
    base = {"schema": REACHABILITY_SCHEMA, "stage": int(spec["stage"]),
            "seed": int(spec["seed"]), "mode": str(spec["mode"]),
            "difficulty": int(spec["difficulty"]), "cap_frames": int(cap_frames)}
    try:
        out = run_channel_a(spec, decisions=decisions_cap, max_frames=int(cap_frames),
                            work_dir=work_dir, data_root=data_root)
    except Exception as error:  # a probe failure is evidence, not a crash
        return {**base, "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "wall_s": round(time.monotonic() - began, 3)}
    report = out["report"]
    decisions = int(report["decisions"])
    frames = decisions * FRAMES_PER_DECISION
    wall = float(report.get("wall_s") or 0.0)
    stop = report.get("stop_reason")
    cleared = stop == "stage_changed" and bool(report.get("stage_clear"))
    died = stop in ("death", "engine_exit_game_over")
    capped = frames + FRAMES_PER_DECISION >= int(cap_frames) and not cleared and not died
    return {
        **base,
        "status": "ok",
        "decision_cap": decisions_cap,
        "decisions": decisions,
        "frames": frames,
        "frames_committed": report.get("frames_committed"),
        "stop_reason": stop,
        "stage_clear": cleared,
        "deaths_observed": report.get("deaths_observed"),
        "death_frame": (int(report.get("frames_committed") or frames) if died else None),
        # The reachable length is what one episode really sustains.  A cap is a lower
        # bound and is flagged as such, never extrapolated.
        "reachable_frames": int(frames),
        "reachable_is_lower_bound": bool(capped),
        "capped": bool(capped),
        "teacher_fps": (round(frames / wall, 3) if wall > 0 else None),
        "wall_s": round(wall, 3),
        "labels_sha256": report.get("labels_sha256"),
    }


def phase_probe(args) -> dict[str, Any]:
    work = work_dir_of(args)
    work.mkdir(parents=True, exist_ok=True)
    stages = parse_stage_list(args.stages)
    seeds = parse_seed_list(args.seeds)
    if args.mode not in teacher_config.CONFIGS:
        raise D1Error(f"unknown teacher mode {args.mode!r}")
    results: dict[int, dict[str, Any]] = {}
    lock = threading.Lock()
    queue = list(stages)

    def worker() -> None:
        while True:
            with lock:
                if not queue:
                    return
                stage = queue.pop(0)
            spec = {
                "stage": stage, "seed": seeds[0], "difficulty": int(args.difficulty),
                "team": int(args.team), "mode": args.mode, "engine": args.engine,
                "expected_sha": args.expected_sha, "binary": None,
                "data_template": args.data_template, "repo_root": str(repo_root()),
            }
            record = probe_stage(spec, cap_frames=int(args.probe_cap_frames),
                                  work_dir=work / "probe" / f"stage{stage}",
                                  data_root=data_root_of(args, f"probe-s{stage}"))
            with lock:
                results[stage] = record
                print(json.dumps({"phase": "probe", **{key: record.get(key) for key in
                                  ("stage", "status", "decisions", "frames", "capped",
                                   "stop_reason", "stage_clear", "death_frame",
                                   "teacher_fps")}}, ensure_ascii=False), flush=True)

    threads = [threading.Thread(target=worker, daemon=False)
               for _ in range(max(1, min(int(args.workers), len(stages))))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # A probe rerun of *some* stages must not throw away the measurements of the
    # others: merge with whatever sits on disk (this run wins for its own stages).
    existing = _read_json(work / "reachability.json", default={}) or {}
    previous = {int(row["stage"]): row for row in existing.get("stages") or []}
    merged: dict[int, dict[str, Any]] = dict(previous)
    merged.update(results)
    ordered = [merged[stage] for stage in stages if stage in merged]
    ordered += [row for stage, row in sorted(merged.items()) if stage not in stages]
    payload = {
        "schema": REACHABILITY_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": args.mode,
        "difficulty": int(args.difficulty),
        "cap_frames": int(args.probe_cap_frames),
        "stages_run_now": stages,
        "stages_reused": sorted(set(previous) - set(results)),
        "stages": ordered,
        "stages_missing": [stage for stage in stages if stage not in results],
    }
    _write_json(work / "reachability.json", payload)
    return payload


# ---------------------------------------------------------------------------
# corpora + take-over points
# ---------------------------------------------------------------------------
def index_corpora(args) -> dict[int, list[dict[str, Any]]]:
    """``{stage: [corpus, ...]}`` from the D0 root and D1's own finished shards.

    A corpus is a *shard-start* shard: a recorded ``RESET`` line, a per-frame mask tape
    and a hash checkpoint table.  A trajectory shard is re-based to its own take-over
    point and is therefore never used as another shard's corpus.
    """

    roots: list[Path] = []
    for root in [Path(args.corpus_root)] + [Path(path) for path in args.corpus_roots]:
        # A corpus root is the directory that *contains* the shard directories
        # (``.tmp_v400_d0/d0/shards``); accepting its parent as well costs nothing and
        # removes a foot-gun from the command line.
        if (root / "shards").is_dir() and not (root / "shard.json").is_file():
            root = root / "shards"
        if root.is_dir() and root not in roots:
            roots.append(root)
    out_shards = out_dir_of(args) / "shards"
    if out_shards.is_dir() and out_shards not in roots:
        roots.append(out_shards)
    index: dict[int, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for root in roots:
        for shard_dir in sorted(root.iterdir()):
            if not (shard_dir / "shard.json").is_file():
                continue  # a .partial staging directory is not a corpus
            if str(shard_dir) in seen:
                continue
            seen.add(str(shard_dir))
            raw = _read_json(shard_dir / "shard.json", default={}) or {}
            if raw.get("start_mode", "shard") != "shard":
                continue
            try:
                trajectory = load_d0_shard(shard_dir)
            except TrajectoryStartError as error:
                print(json.dumps({"phase": "plan", "corpus": str(shard_dir),
                                  "skipped": f"{type(error).__name__}: {error}"},
                                 ensure_ascii=False), flush=True)
                continue
            index.setdefault(int(trajectory.stage_index), []).append({
                "traj_id": trajectory.traj_id,
                "dir": str(shard_dir),
                "seed": int(trajectory.seed),
                "stage": int(trajectory.stage_index),
                "difficulty": trajectory.difficulty,
                "frames": int(trajectory.frames),
                "checkpoints": len(trajectory.hashes),
                "source": "d1" if str(shard_dir).startswith(str(out_shards)) else "d0",
            })
    for rows in index.values():
        rows.sort(key=lambda row: (row["source"], row["dir"]))
    return index


def choose_corpus(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Deterministic pick: rotate by seed so consecutive shards use different runs."""

    return rows[int(seed) % len(rows)]


def choose_takeover_frame(traj, *, decisions: int, rng: random.Random, bias: float,
                          boss_start: int | None, boss_weight: float) -> int:
    """A recorded checkpoint that still leaves ``decisions`` x 4 frames on the tape."""

    room = int(decisions) * FRAMES_PER_DECISION
    candidates = [frame for frame in traj.checkpoint_frames
                  if frame in traj.hashes and frame + room <= traj.frames]
    if not candidates:
        raise D1Error(
            f"{traj.traj_id}: no checkpoint leaves room for {decisions} decisions "
            f"({room} frames) on a {traj.frames}-frame tape")
    return int(sample_target_frame(rng, candidates, bias=bias, boss_start=boss_start,
                                   boss_weight=boss_weight))


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def _resolve_takeovers(args, shards: list[dict[str, Any]]) -> list[str]:
    """Turn each trajectory spec's corpus into an immutable take-over point."""

    errors: list[str] = []
    for spec in shards:
        if spec["start_mode"] != "trajectory":
            continue
        corpus = spec.get("corpus")
        if corpus is None:
            errors.append(f"{spec['key']}: trajectory start without a corpus")
            continue
        try:
            traj = load_d0_shard(corpus["dir"])
        except TrajectoryStartError as error:
            errors.append(f"{spec['key']}: corpus unreadable: {error}")
            continue
        rng = random.Random(f"{spec['key']}|{corpus['traj_id']}")
        try:
            target = choose_takeover_frame(
                traj, decisions=int(spec["decisions"]), rng=rng,
                bias=float(spec["bias"]), boss_start=spec.get("boss_start"),
                boss_weight=float(spec["boss_weight"]))
        except D1Error as error:
            errors.append(f"{spec['key']}: {error}")
            continue
        spec["target_frame"] = int(target)
        spec["prefix_decisions"] = int(target) // FRAMES_PER_DECISION
        spec["teacher_decisions"] = int(spec["prefix_decisions"]) + int(spec["decisions"])
        spec["expected_start_hash"] = str(traj.hashes[target])
        corpus["target_frame"] = int(target)
        corpus["expected_start_hash"] = spec["expected_start_hash"]
    return errors


def build_plan(args, reachability: dict[str, Any],
               corpora: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    stages = parse_stage_list(args.stages)
    seeds = parse_seed_list(args.seeds)
    modes = parse_start_mix(args.start_mix)
    by_stage = {int(row["stage"]): row for row in reachability.get("stages") or []}
    per_stage_min = int(args.frames_target_per_stage)
    total_target = int(args.frames_target_total)
    per_stage_target = max(per_stage_min,
                           int(math.ceil(total_target / max(1, len(stages)))))
    limit = int(args.plan_limit_per_stage)
    decisions_cap = int(args.decisions_per_shard)
    shards: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    trajectory_cap = int(getattr(args, "trajectory_decisions_per_shard", 0) or 0)
    for position, stage in enumerate(stages):
        reach = by_stage.get(stage)
        if reach is None or reach.get("status") != "ok":
            errors.append(f"stage {stage}: no reachability record (run --phase probe)")
            continue
        reachable = int(reach["reachable_frames"])
        if reachable <= 0:
            errors.append(f"stage {stage}: reachable_frames is 0")
            continue
        # One shard never exceeds what the teacher sustains, and never exceeds the
        # requested decisions; frames are always a whole number of decisions.
        decisions = max(1, min(decisions_cap, reachable // FRAMES_PER_DECISION))
        frames_per_shard = decisions * FRAMES_PER_DECISION
        # Plan until the per-stage target is met.  Shards may differ in size when a
        # start mode has its own cap (a take-over shard must leave room for its whole
        # segment inside the recorded corpus), so the loop -- not a division -- is what
        # guarantees ``frames_planned >= per_stage_target``.
        stage_shards: list[dict[str, Any]] = []
        frames_planned = 0
        index = 0
        used_keys: set[str] = set()
        while frames_planned < per_stage_target:
            start_mode = modes[index % len(modes)]
            shard_decisions = decisions
            if start_mode == "trajectory" and trajectory_cap > 0:
                shard_decisions = max(1, min(decisions, trajectory_cap))
            seed = seeds[(position + index) % len(seeds)]
            corpus = None
            if start_mode == "trajectory":
                rows = corpora.get(stage) or []
                if not rows:
                    errors.append(
                        f"stage {stage}: a trajectory start was requested but no recorded "
                        "corpus of this stage exists yet (run the shard-start phase first)")
                    break
                corpus = choose_corpus(rows, seed)
                # The RESET seed must be the recorded run's seed: the tape is only
                # valid for the trajectory it was recorded from.
                seed = int(corpus["seed"])
            # Key: one shard per (stage, seed, size, start mode).  A key is only
            # disambiguated by the plan index when the seed cycle would otherwise name
            # two different shards, so a plan regenerated with the same inputs keeps
            # the keys of the shards a batch has already finished.
            base_key = (f"d1-s{stage}d{int(args.difficulty)}-n{shard_decisions}"
                        f"-seed{seed}-{start_mode}")
            key = base_key if base_key not in used_keys else f"{base_key}-i{index:03d}"
            used_keys.add(key)
            spec: dict[str, Any] = {
                "schema": SHARD_SCHEMA_D1,
                "key": key,
                "index": index,
                "stage": stage,
                "seed": seed,
                "difficulty": int(args.difficulty),
                "team": int(args.team),
                "mode": args.mode,
                "decisions": shard_decisions,
                "teacher_decisions": shard_decisions,
                "start_mode": start_mode,
                "target_frame": 0,
                "frames_expected": shard_decisions * FRAMES_PER_DECISION,
                "obs": "v3",
                "engine": str(args.engine),
                "expected_sha": str(args.expected_sha),
                "binary": None,
                "data_template": str(args.data_template),
                "repo_root": str(repo_root()),
                "out_dir": str(out_dir_of(args)),
                "work_dir": str(work_dir_of(args) / "work" / key),
                "work_root": str(work_dir_of(args)),
                "data_root": data_root_of(args, key),
                "teacher_tool": str(repo_root() / "tools" / "rl" / "v400_teacher.py"),
                "timeout": float(args.timeout),
                "bias": float(args.trajectory_bias),
                "boss_weight": float(args.trajectory_boss_weight),
                "boss_start": (int(args.boss_start) if int(args.boss_start) >= 0 else None),
                "plan": str(args.plan_name),
            }
            if corpus is not None:
                spec["corpus"] = dict(corpus)
                spec["seed"] = int(corpus["seed"])
                spec["key"] = key
                spec["work_dir"] = str(work_dir_of(args) / "work" / key)
                spec["data_root"] = data_root_of(args, key)
            stage_shards.append(spec)
            frames_planned += int(spec["frames_expected"])
            index += 1
            if index > 5000:  # pragma: no cover - a target of >10 M frames
                raise D1Error(f"stage {stage}: the shard loop did not terminate")
        needed = len(stage_shards)
        planned_shards = stage_shards[:limit] if limit > 0 else stage_shards
        shards.extend(planned_shards)
        planned_frames = sum(int(spec["frames_expected"]) for spec in planned_shards)
        stage_rows.append({
            "stage": stage,
            "reachable_frames": reachable,
            "reachable_is_lower_bound": bool(reach.get("reachable_is_lower_bound")),
            "teacher_cleared": bool(reach.get("stage_clear")),
            "death_frame": reach.get("death_frame"),
            "per_stage_target": per_stage_target,
            "decisions_per_shard": decisions,
            "frames_per_shard": frames_per_shard,
            "trajectory_decisions_cap": trajectory_cap or decisions,
            "shards_needed": needed,
            "shards_planned": len(planned_shards),
            "frames_planned": planned_frames,
            "start_modes": [spec["start_mode"] for spec in planned_shards],
            "coverage_shortfall": max(0, per_stage_target - planned_frames),
            "coverage_shortfall_reason": ("plan_limit_per_stage" if limit > 0 else None),
        })

    errors.extend(_resolve_takeovers(args, shards))
    total_frames = sum(int(spec["frames_expected"]) for spec in shards)
    total_prefix = sum(int(spec.get("target_frame") or 0) for spec in shards)
    payload = {
        "schema": PLAN_SCHEMA,
        "name": str(args.plan_name),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stages": stages,
        "seeds": seeds,
        "start_mix": modes,
        "mode": args.mode,
        "difficulty": int(args.difficulty),
        "obs": "v3",
        "obs_spec_version": obs_spec_v3.OBS_SPEC_VERSION,
        "obs_spec_digest": obs_spec_v3.obs_spec_v3_digest(),
        "targets": {"total": total_target, "per_stage": per_stage_target,
                    "per_stage_floor": per_stage_min},
        "decisions_per_shard_cap": decisions_cap,
        "plan_limit_per_stage": limit,
        "engine": str(args.engine),
        "expected_sha": str(args.expected_sha),
        "probe_cap_frames": int(getattr(args, "probe_cap_frames", 0)),
        "stages_detail": stage_rows,
        "shards_planned": len(shards),
        "frames_planned": total_frames,
        "prefix_frames_paid": total_prefix,
        "coverage": {
            "shortfall_frames": sum(int(row["coverage_shortfall"]) for row in stage_rows),
            "stages_short": [row["stage"] for row in stage_rows
                             if row["coverage_shortfall"] > 0],
            "note": "a shortfall is recorded, never padded with synthetic frames",
        },
        "errors": errors,
        "shards": shards,
    }
    payload["status"] = "blocked" if errors else "ok"
    return payload


def phase_plan(args) -> dict[str, Any]:
    work = work_dir_of(args)
    reachability = _read_json(work / "reachability.json")
    if not reachability:
        raise SystemExit(f"{work / 'reachability.json'}: run --phase probe first")
    corpora = index_corpora(args)
    plan = build_plan(args, reachability, corpora)
    _write_json(work / f"plan.{args.plan_name}.json", plan)
    print(json.dumps({"phase": "plan", "name": args.plan_name,
                      "status": plan["status"], "shards": plan["shards_planned"],
                      "frames": plan["frames_planned"],
                      "prefix_frames": plan["prefix_frames_paid"],
                      "shortfall": plan["coverage"]["shortfall_frames"],
                      "corpora": {str(stage): len(rows) for stage, rows in corpora.items()},
                      "errors": plan["errors"]}, ensure_ascii=False), flush=True)
    if plan["status"] == "blocked":
        raise SystemExit(EXIT_FAIL_CLOSED)
    return plan


# ---------------------------------------------------------------------------
# channel B: env-role replay
# ---------------------------------------------------------------------------
def _read_pixels(env: Th08Env, where: str) -> tuple[np.ndarray, np.ndarray, dict[str, int],
                                                    dict[str, float]]:
    from touhou_rl.v400.gym.shm import FLAG_PIXELS_VALID

    block = env.shm
    if block is None:
        raise D1Error(f"{where}: no shared pixel block")
    header = block.header()
    if not int(header.get("flags", 0)) & FLAG_PIXELS_VALID:
        raise D1Error(f"{where}: shared block has no published pixels")
    if int(header.get("width", 0)) != 640 or int(header.get("height", 0)) != 480:
        raise D1Error(f"{where}: readback is {header.get('width')}x{header.get('height')}")
    play, hud = obs_spec_v3.encode_planes_v3(block.pixels())
    stats = require_rendered(play, hud, where=where)
    return play, hud, header, stats


def _pair(decision: int, boundary: str, frame: int, shadow: str, engine: str) -> dict[str, Any]:
    return {"decision": int(decision), "boundary": boundary, "frame": int(frame),
            "shadow": str(shadow), "env": str(engine),
            "equal": str(shadow) == str(engine), "source": "A/B"}


def _hash_of(reply_hash: str | None, where: str) -> str:
    value = str(reply_hash or "").strip().lower()
    if len(value) != 16 or any(ch not in HEX for ch in value):
        raise D1Error(f"{where}: no usable 16-hex state hash: {reply_hash!r}")
    return value


def _step_exact(env: Th08Env, mask: int, expected_frame: int,
                where: str) -> tuple[int, str]:
    """One single-frame ``STEP``; returns ``(render_frame, hash of the new state)``."""

    reply = env.advance(int(mask), 1)
    if reply.exit or reply.game_over:
        raise D1Error(
            f"{where}: the engine left the stage (exit={reply.exit} "
            f"game_over={reply.game_over}) at frame {reply.frame}; channel A committed "
            "this frame, so the two channels are no longer the same trajectory")
    if reply.advanced != 1:
        raise D1Error(f"{where}: STEP advanced {reply.advanced} of 1 frame")
    if int(reply.frame) != int(expected_frame) + 1:
        raise D1Error(
            f"{where}: engine frame {reply.frame} != {int(expected_frame) + 1} after a "
            "single-frame step")
    # The engine publishes the render of the frame it just played, i.e. the frame
    # before the counter it now reports (main.cpp: draw -> publish -> ++g_frameCount).
    return int(reply.frame) - 1, _hash_of(reply.hash, where)


def _rebase_sample(sample: dict[str, Any], offset: int) -> dict[str, Any]:
    """Re-base a take-over shard's decision clock to 0 (loader arithmetic)."""

    absolute = int(sample["window"]["decision_frame"])
    relative = absolute - int(offset)
    if relative < 0:
        raise D1Error(f"cannot re-base decision frame {absolute} by {offset}")
    window = decision_window(relative)
    return {"play": sample["play"], "hud": sample["hud"],
            "render_frames": [int(value) for value in window["render_frames"]],
            "window": window, "pad": 0, "episode_start": False, "seed_frame": None}


def _trajectory_warmup(env: Th08Env, spec: dict[str, Any], traj,
                       buffer: DecisionObservationBuffer) -> dict[str, Any]:
    """``TrajectoryStart`` + 4 real renders, so decision 0 has ``T-4..T-1``.

    ``replay_to_frame`` stops at the recorded checkpoint ``T-4`` (every decision
    boundary is one).  Four single-frame steps then apply the recorded masks
    ``T-4 .. T-1``: each render is kept and the hash of the last step must be the
    recorded hash at ``T``.  This is the take-over EXP-P3-012a verified on this engine.
    """

    target = int(spec["target_frame"])
    if target < FRAMES_PER_DECISION:
        raise D1Error(f"trajectory target frame {target} is below one decision")
    start = traj.start(target - FRAMES_PER_DECISION)
    takeover = replay_to_frame(env, start)
    if not takeover.matches:
        raise D1Error(f"trajectory take-over did not match: {takeover.as_record()}")
    expected = str(traj.hashes[target])
    buffer.reset(None)
    last_hash = ""
    for offset in range(FRAMES_PER_DECISION):
        frame = target - FRAMES_PER_DECISION + offset
        render_frame, last_hash = _step_exact(
            env, int(traj.masks[frame]), frame, f"take-over frame {frame}")
        if render_frame != frame:
            raise D1Error(
                f"take-over: render frame {render_frame} != recorded frame {frame}")
        play, hud, _header, _stats = _read_pixels(env, f"take-over render {render_frame}")
        buffer.push(RenderFrame(frame=render_frame, play=play, hud=hud))
    if last_hash != expected:
        raise D1Error(
            f"trajectory take-over at frame {target}: engine hash {last_hash} != "
            f"recorded {expected}")
    return {"traj_id": traj.traj_id, "seed": int(traj.seed),
            "stage": int(traj.stage_index), "target_frame": target,
            "expected_start_hash": expected, "start_hash": last_hash, "matches": True,
            "fast_forward": takeover.as_record()}


def _device(spec: dict[str, Any], *, decisions: int) -> tuple[Th08Env, Path, str]:
    engine = Path(spec["engine"]) / "th08-modern"
    binary = Path(spec["binary"]) if spec.get("binary") else engine
    actual = sha256_file(binary)
    if actual != spec["expected_sha"]:
        raise D1Error(f"{binary}: sha256 {actual} != expected {spec['expected_sha']}")
    data = dual().prepare_data_dir(Path(spec["data_template"]),
                                   Path(spec["data_root"]) / "env")
    shm = f"v400d1-{os.getpid()}-{int(time.time() * 1000) % 10_000_000}"
    env = Th08Env(binary, data["data_dir"], shm_name=shm, role="env",
                  max_frames=max(decisions * FRAMES_PER_DECISION + 1000, 10_000))
    return env, Path(data["data_dir"]), shm


def replay_shard(spec: dict[str, Any], labels: list[dict[str, Any]],
                 channel_a_report: dict[str, Any]) -> dict[str, Any]:
    """Channel B: replay the tape on an env-role engine and write the shard.

    Streaming: a decision goes into the writer as soon as its ``post`` checkpoint
    matched, so a 4,000-decision shard never sits in RAM.  A mismatch aborts the
    writer, which leaves ``shards/.<key>.partial`` as evidence.
    """

    check_obs_version(str(spec.get("obs", "v3")))
    out_dir = Path(spec["out_dir"])
    key = str(spec["key"])
    start_mode = str(spec["start_mode"])
    if not labels:
        raise D1Error(f"{key}: channel A produced no decision")
    start_frame = int(spec.get("target_frame") or 0) if start_mode == "trajectory" else 0
    difficulty = difficulty_name(int(spec["difficulty"]))
    traj = None
    if start_mode == "trajectory":
        corpus = spec.get("corpus") or {}
        if not corpus.get("dir"):
            raise D1Error(f"{key}: trajectory start without a corpus")
        traj = load_d0_shard(corpus["dir"])
        if (int(traj.seed) != int(spec["seed"])
                or int(traj.stage_index) != int(spec["stage"])):
            raise D1Error(
                f"{key}: corpus {traj.traj_id} is seed {traj.seed}/stage "
                f"{traj.stage_index}, the spec asks seed {spec['seed']}/stage {spec['stage']}")
    history, history_info = dual().v3_action_history(labels)
    if history.shape[0] != len(labels):
        raise D1Error(f"{key}: act_hist rows {history.shape[0]} != {len(labels)} decisions")

    env, data_dir, shm_name = _device(spec, decisions=len(labels))
    buffer = DecisionObservationBuffer(play_shape=obs_spec_v3.PLAY_SHAPE)
    pairs: list[dict[str, Any]] = []
    render_stats: list[dict[str, float]] = []
    expect_transition = str(channel_a_report.get("stop_reason")) == "stage_changed"
    transition_trim: dict[str, Any] | None = None
    written = 0
    frames_read = 0
    pixels_ms = 0.0
    meta = {
        "stage": int(spec["stage"]),
        "difficulty": int(spec["difficulty"]),
        "team": int(spec.get("team", 0)),
        "seed": int(spec["seed"]),
        "mode": str(spec["mode"]),
        "frames_per_decision": FRAMES_PER_DECISION,
        "generated_by": "tools/rl/v400_d1_collect.py (EXP-P3-012)",
        "decisions_requested": int(spec["decisions"]),
        # Channel A's verdict, on the shard itself (the D0 schema records the same
        # keys; the per-shard report keeps the full channel-A block).
        "stop_reason": channel_a_report.get("stop_reason"),
        "stage_clear": channel_a_report.get("stage_clear"),
        "deaths_observed": channel_a_report.get("deaths_observed"),
        "frames_committed": channel_a_report.get("frames_committed"),
        "stopped_early": bool(channel_a_report.get("stopped_early", False)),
        "channel_a_decisions": channel_a_report.get("decisions"),
        "start_mode": start_mode,
        "start_frame": int(start_frame),
        "stage_index": int(spec["stage"]),
        "label_channel": "shadow",
        "pixel_channel": "env",
        "alignment_method": "hash-checkpoint",
        "obs_requested": "v3",
        "obs_planes": [plane.meta_key for plane in obs_version_spec("v3").planes],
        "obs_bytes_per_decision_expected": obs_spec_v3.BYTES_PER_DECISION,
        "reset_command": (f"RESET {int(spec['stage'])} {int(spec['difficulty'])} "
                          f"{int(spec.get('team', 0))} {int(spec['seed'])} "
                          f"{1 if traj is not None and traj.resources == 'practice' else 0}"),
        "engine_shadow": {"binary": str(Path(spec["engine"]) / "th08-modern"),
                          "sha256": spec["expected_sha"], "role": "shadow"},
        "engine_env": {"binary": str(Path(spec["engine"]) / "th08-modern"),
                       "sha256": spec["expected_sha"], "role": "env",
                       "data_dir": str(data_dir)},
        "act_hist": history_info,
        "d1": {"schema": SHARD_SCHEMA_D1, "plan": spec.get("plan"), "key": key,
               "data_root": spec.get("data_root")},
    }
    if traj is not None:
        meta["trajectory"] = {
            "traj_id": traj.traj_id,
            "source": str(traj.source),
            "corpus_seed": int(traj.seed),
            "corpus_frames": int(traj.frames),
            "target_frame": int(start_frame),
            "expected_start_hash": str(spec.get("expected_start_hash")
                                       or traj.hashes.get(start_frame)),
            "prefix_decisions_discarded": int(start_frame) // FRAMES_PER_DECISION,
            "note": "decision frames are re-based to the take-over point; start_frame is "
                    "the absolute engine frame of decision 0",
        }
    writer = TeacherObsShardWriter(out_dir, key=key, meta=meta, obs_version="v3")
    began = time.monotonic()
    try:
        with env:
            if traj is not None:
                takeover = _trajectory_warmup(env, spec, traj, buffer)
                meta["trajectory"]["takeover"] = takeover
                current_hash = takeover["start_hash"]
            else:
                env.reset(StageStart(stage=int(spec["stage"]), difficulty=difficulty,
                                     team=int(spec.get("team", 0)),
                                     seed=int(spec["seed"])))
                current_hash = _hash_of(env.last_reset_hash, "RESET reply")
                play, hud, header, stats = _read_pixels(env, "post-RESET seed render")
                buffer.reset(RenderFrame(frame=int(header["frame"]), play=play, hud=hud))
                render_stats.append(stats)
                meta["seed_render"] = {"frame": int(header["frame"]),
                                       "flags": int(header.get("flags", 0))}
            last = len(labels) - 1
            pending: dict[str, Any] | None = None
            for index, row in enumerate(labels):
                absolute = start_frame + index * FRAMES_PER_DECISION
                relative = index * FRAMES_PER_DECISION
                pairs.append(_pair(index, "pre", absolute, str(row["hash_pre"]),
                                   current_hash))
                if current_hash != str(row["hash_pre"]):
                    raise AlignmentError(
                        f"{key}: checkpoint mismatch at decision {index} pre frame "
                        f"{absolute}: shadow={row['hash_pre']} env={current_hash}")
                sample = buffer.stack_for(
                    absolute, episode_start=(start_mode == "shard" and index == 0))
                if start_mode == "trajectory":
                    sample = _rebase_sample(sample, start_frame)
                pending = None
                if index == last and expect_transition:
                    # Do not commit the final decision before its post checkpoint is
                    # checked: the stage-clear commit is the one documented exclusion.
                    pending = {"index": index, "relative": relative, "sample": sample,
                               "row": row, "history": history[index]}
                else:
                    _add_sample(writer, index, relative, row, sample, history[index],
                                start_frame)
                    written += 1
                for offset in range(FRAMES_PER_DECISION):
                    frame = absolute + offset
                    render_frame, current_hash = _step_exact(
                        env, int(row["step_frames"][offset]), frame,
                        f"{key}: decision {index} frame {offset}")
                    started_px = time.perf_counter()
                    play, hud, _header, stats = _read_pixels(env, f"render {render_frame}")
                    pixels_ms += (time.perf_counter() - started_px) * 1000.0
                    frames_read += 1
                    render_stats.append(stats)
                    if render_frame != frame:
                        raise D1Error(
                            f"{key}: decision {index} offset {offset}: render frame "
                            f"{render_frame} != {frame}")
                    buffer.push(RenderFrame(frame=render_frame, play=play, hud=hud))
                pairs.append(_pair(index, "post", absolute + FRAMES_PER_DECISION,
                                   str(row["hash_post"]), current_hash))
                if current_hash != str(row["hash_post"]):
                    if index == last and expect_transition:
                        # EXP-P1-018's single documented exclusion: the commit that
                        # completes the stage leaves the *next* stage's setup as its
                        # post state, so it is not this stage's gameplay.  Drop that
                        # decision (and its two checkpoints), keep both hashes and the
                        # reason.
                        trimmed = pairs[-2:]
                        del pairs[-2:]
                        transition_trim = {
                            "decision": index,
                            "frame": absolute + FRAMES_PER_DECISION,
                            "shadow_post": str(row["hash_post"]),
                            "env_post": current_hash,
                            "reason": "stage-clear commit crosses the stage boundary; "
                                      "the post state is the next stage's setup",
                            "decisions_before_trim": len(labels),
                            "decisions_after_trim": index,
                            "checkpoints_removed": trimmed,
                        }
                        meta["transition_trim"] = transition_trim
                        pending = None
                        break
                    raise AlignmentError(
                        f"{key}: checkpoint mismatch at decision {index} post frame "
                        f"{absolute + FRAMES_PER_DECISION}: shadow={row['hash_post']} "
                        f"env={current_hash}")
                if pending is not None:
                    _add_sample(writer, pending["index"], pending["relative"],
                                pending["row"], pending["sample"], pending["history"],
                                start_frame)
                    written += 1
                    pending = None
            if pending is not None:  # pragma: no cover - defensive
                raise D1Error(f"{key}: the final decision was never committed")
        equal = sum(1 for row in pairs if row["equal"])
        if equal != len(pairs):
            raise AlignmentError(
                f"{key}: {len(pairs) - equal} of {len(pairs)} checkpoints disagree")
        alignment = {
            "method": "hash-checkpoint",
            "boundaries": len(pairs),
            "decisions": written,
            "checkpoints_equal": equal,
            "checkpoints_mismatched": len(pairs) - equal,
            "first_mismatch": None,
            "trimmed_transition": transition_trim,
        }
        writer.meta["alignment_extra"] = {
            "pixel_source": "env_replay",
            "label_source": "shadow_teacher",
            "start_mode": start_mode,
            "start_frame": int(start_frame),
            "hash_alignment": "hash-checkpoint",
            "alignment_method": "hash-checkpoint",
            "hash_checkpoints": alignment,
        }
        writer.meta["hash_checkpoint_pairs"] = pairs
        writer.meta["decisions"] = int(written)
        writer.meta["frames"] = int(written * FRAMES_PER_DECISION)
        summary = writer.finalize(report={
            "d1": {"schema": SHARD_SCHEMA_D1, "plan": spec.get("plan"),
                   "start_mode": start_mode, "start_frame": int(start_frame)},
            "hash_alignment": alignment,
            "channel_a": {name: channel_a_report.get(name) for name in
                          ("decisions", "labels_sha256", "wall_s", "exit_code",
                           "stop_reason", "stage_clear", "deaths_observed")},
            "channel_b": {"decisions": written, "frames_read": frames_read,
                          "pixel_ms_per_frame": (round(pixels_ms / frames_read, 3)
                                                 if frames_read else None)},
            "trajectory": meta.get("trajectory"),
        })
    except BaseException as error:
        try:
            writer.abort(error)
        except Exception:  # pragma: no cover - evidence is best effort
            pass
        raise
    finally:
        try:
            from touhou_rl.v400.gym.shm import segment_path
            segment_path(shm_name).unlink()
        except (OSError, ValueError):
            pass
    elapsed = time.monotonic() - began
    teacher_wall = float(channel_a_report.get("wall_s") or 0.0)
    summary.update({
        "start_mode": start_mode,
        "start_frame": int(start_frame),
        "alignment": alignment,
        "transition_trim": transition_trim,
        "env_wall_s": round(elapsed, 3),
        "frames_read": frames_read,
        "env_fps": (round(frames_read / elapsed, 3) if elapsed > 0 else None),
        "pixel_ms_per_frame": round(pixels_ms / frames_read, 3) if frames_read else None,
        "render": _aggregate_render(render_stats),
        "channel_a_wall_s": teacher_wall,
        "teacher_fps": (round(written * FRAMES_PER_DECISION / teacher_wall, 3)
                        if teacher_wall > 0 else None),
        "trajectory": meta.get("trajectory"),
    })
    return summary


def _aggregate_render(renders: list[dict[str, float]]) -> dict[str, Any]:
    if not renders:
        return {"frames_sampled": 0}
    keys = ("play_mean", "play_std", "play_max", "hud_mean", "hud_std", "hud_max")
    out: dict[str, Any] = {"frames_sampled": len(renders)}
    for key in keys:
        values = [row[key] for row in renders if key in row]
        if values:
            out[key] = round(float(np.mean(values)), 6)
            out[key + "_min"] = round(float(np.min(values)), 6)
    out["zero_frames"] = sum(1 for row in renders
                             if float(row.get("play_max", 1)) <= 0.0
                             or float(row.get("hud_max", 1)) <= 0.0)
    return out


def _add_sample(writer: TeacherObsShardWriter, index: int, relative: int,
                row: dict[str, Any], sample: dict[str, Any],
                act_hist: np.ndarray, start_frame: int) -> None:
    writer.add(
        decision=int(index),
        decision_frame=int(relative),
        label=dual()._ShadowLabel(row),
        sample={"play": sample["play"], "hud": sample["hud"],
                "render_frames": [int(value) for value in sample["render_frames"]],
                "pad": int(sample["pad"]), "window": sample["window"],
                "act_hist": np.asarray(act_hist, dtype=np.uint8)},
        extra={
            "channel": "shadow-labels",
            "pixel_channel": "env",
            "start_mode": writer.meta.get("start_mode"),
            "start_frame": int(start_frame),
            "hash_pre": row["hash_pre"],
            "hash_post": row["hash_post"],
            "step_frames": list(row["step_frames"]),
            "frame_event_flags": row.get("frame_event_flags"),
            "shoot_release": bool(row.get("shoot_release", False)),
            "bomb_death_available": bool(row.get("bomb_death_available", False)),
        },
    )


def _verify_trajectory_prefix(spec: dict[str, Any], labels: list[dict[str, Any]]
                              ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prove channel A reproduced the recorded tape up to the take-over point.

    The shadow teacher cannot start mid-stage by itself, so the take-over is performed
    the only way that preserves the recorded state: run it from the same ``RESET`` and
    require mask-by-mask and hash-by-hash equality with the recorded prefix, then
    discard the prefix labels.  Any difference is fail-closed -- the labels after it
    would describe states that never existed on the recording.
    """

    target = int(spec["target_frame"])
    prefix_decisions = target // FRAMES_PER_DECISION
    traj = load_d0_shard((spec.get("corpus") or {})["dir"])
    if prefix_decisions > len(labels):
        raise D1Error(
            f"{spec['key']}: {len(labels)} decisions from channel A are fewer than the "
            f"{prefix_decisions}-decision prefix of the take-over frame {target}")
    for index in range(prefix_decisions):
        row = labels[index]
        frame = index * FRAMES_PER_DECISION
        expected_masks = list(traj.masks[frame:frame + FRAMES_PER_DECISION])
        if list(row["step_frames"]) != expected_masks:
            raise AlignmentError(
                f"{spec['key']}: channel A diverged from the recorded tape at decision "
                f"{index} (frame {frame}): masks {list(row['step_frames'])} != recorded "
                f"{expected_masks}")
        for side, at in (("hash_pre", frame), ("hash_post", frame + FRAMES_PER_DECISION)):
            recorded = traj.hashes.get(at)
            if recorded is not None and str(row[side]) != str(recorded):
                raise AlignmentError(
                    f"{spec['key']}: channel A diverged from the recorded tape at "
                    f"decision {index} {side} (frame {at}): {row[side]} != {recorded}")
    return labels[prefix_decisions:], {
        "decisions": prefix_decisions, "frames": target, "verified": True,
        "masks_equal": True, "checkpoints": 2 * prefix_decisions,
        "traj_id": traj.traj_id, "expected_start_hash": str(traj.hashes[target]),
    }


def execute_shard(spec: dict[str, Any]) -> dict[str, Any]:
    """One shard: resume check, channel A, channel B, load gate, per-shard report."""

    out_dir = Path(spec["out_dir"])
    key = str(spec["key"])
    shard_dir = out_dir / "shards" / key
    resumed = dual().shard_resume_report(shard_dir)
    if resumed is not None:
        print(json.dumps({"phase": "shard", "shard": key, "status": "SKIPPED_COMPLETED",
                          "decisions": resumed.get("decisions")}, ensure_ascii=False),
              flush=True)
        return {"key": key, "status": "skipped", "shard": resumed}
    partial = out_dir / "shards" / f".{key}.partial"
    quarantined = None
    if partial.exists():
        if not spec.get("quarantine_partial"):
            raise D1Error(
                f"{partial}: a staging directory from an interrupted run exists; it is kept "
                "as evidence and never overwritten (move it aside deliberately, or run "
                "with --retry-failed to quarantine it under a timestamped name)")
        # Keep the failed staging directory as evidence but out of the writer's way: it
        # is *renamed*, never deleted.
        quarantined = out_dir / "shards" / (
            f".{key}.partial.failed-{time.strftime('%Y%m%dT%H%M%S')}")
        if quarantined.exists():  # pragma: no cover - one retry per second
            raise D1Error(f"{quarantined}: quarantine name already taken")
        partial.rename(quarantined)
        print(json.dumps({"phase": "shard", "shard": key,
                          "quarantined": str(quarantined)}, ensure_ascii=False), flush=True)
    start_mode = str(spec["start_mode"])
    teacher_decisions = int(spec.get("teacher_decisions") or spec["decisions"])
    began = time.monotonic()
    channel_a = run_channel_a(spec, decisions=teacher_decisions,
                              max_frames=teacher_decisions * FRAMES_PER_DECISION,
                              work_dir=Path(spec["work_dir"]), data_root=spec["data_root"])
    report = channel_a["report"]
    labels = channel_a["labels"]
    prefix = None
    if start_mode == "trajectory":
        prefix_decisions = int(spec["target_frame"]) // FRAMES_PER_DECISION
        if len(labels) < prefix_decisions:
            raise D1Error(
                f"{key}: the teacher stopped after {len(labels)} of {prefix_decisions} "
                f"prefix decisions (stop_reason={report.get('stop_reason')}); the "
                f"take-over frame {spec['target_frame']} is unreachable")
        labels, prefix = _verify_trajectory_prefix(spec, labels)
    summary = replay_shard(spec, labels, report)
    summary.update({
        "key": key,
        "quarantined_partial": str(quarantined) if quarantined else None,
        "stage": int(spec["stage"]),
        "seed": int(spec["seed"]),
        "mode": str(spec["mode"]),
        "obs": "v3",
        "channel_a": {name: report.get(name) for name in
                      ("decisions", "decisions_requested", "stopped_early", "stop_reason",
                       "stage_clear", "deaths_observed", "frames_committed",
                       "labels_sha256", "exit_code")},
        "prefix": prefix,
        "wall_s": round(time.monotonic() - began, 3),
    })
    # The D1 gate: every stored decision must resolve to a readable obs-v3 sample.
    from touhou_rl.v400.student import obs_v3_dataset

    loaded = obs_v3_dataset.load_teacher_samples_v3(
        [Path(summary["dir"]) / "labels.jsonl"], min_samples=int(summary["decisions"]),
        obs_root=out_dir, require_observations=True)
    shard = obs_v3_dataset.load_teacher_shard_v3(Path(summary["dir"]))
    summary["load"] = {
        "samples": int(loaded["count"]),
        "expected_samples": int(summary["decisions"]),
        "samples_match": int(loaded["count"]) == int(summary["decisions"]),
        "obs_spec_version": loaded["obs_spec_version"],
        "obs_spec_digest": loaded["obs_spec_digest"],
        "digest_matches_frozen": loaded["obs_spec_digest"] == FROZEN_OBS_SPEC_V3_DIGEST,
        "unresolved": len(loaded["obs_unresolved"]),
        "play_shape": list(shard.play.shape),
        "hud_shape": list(shard.hud.shape),
        "act_hist_shape": list(shard.act_hist.shape),
        "frames": shard.meta.get("stage_frames"),
        "decisions": len(shard),
    }
    summary["status"] = "ok"
    _write_json(Path(spec["work_root"]) / "reports" / f"{key}.json", summary)
    print(json.dumps({"phase": "shard", "shard": key, "status": "ok",
                      "stage": spec["stage"], "start_mode": start_mode,
                      "decisions": summary["decisions"],
                      "frames": summary["load"]["frames"],
                      "hash": f"{summary['alignment']['checkpoints_equal']}/"
                              f"{summary['alignment']['boundaries']}",
                      "load": f"{summary['load']['samples']}/"
                              f"{summary['load']['expected_samples']}",
                      "digest": summary["load"]["obs_spec_digest"][:8],
                      "trim": bool(summary["transition_trim"]),
                      "teacher_fps": summary["teacher_fps"],
                      "env_fps": summary["env_fps"],
                      "wall_s": summary["wall_s"]}, ensure_ascii=False), flush=True)
    return summary


def phase_shard(args) -> int:
    spec = _read_json(args.shard_spec)
    if not spec:
        raise SystemExit(f"{args.shard_spec}: no shard spec")
    _maybe_reexec_under_xvfb(args)
    try:
        execute_shard(spec)
    except AlignmentError as error:
        print(json.dumps({"phase": "shard", "shard": spec.get("key"),
                          "status": "ALIGNMENT_FAILED", "reason": str(error)},
                         ensure_ascii=False), flush=True)
        return EXIT_ALIGN
    except (D1Error, ObservationError, TrajectoryStartError) as error:
        print(json.dumps({"phase": "shard", "shard": spec.get("key"),
                          "status": "FAIL_CLOSED",
                          "reason": f"{type(error).__name__}: {error}"},
                         ensure_ascii=False), flush=True)
        return EXIT_FAIL_CLOSED
    return EXIT_OK


def _maybe_reexec_under_xvfb(args) -> None:
    if getattr(args, "xvfb", "auto") == "off" or os.environ.get("V400_D1_XVFB") == "1":
        return
    anchor = sys.argv[0] if sys.argv and sys.argv[0] else __file__
    argv = ["xvfb-run", "-a", sys.executable, str(Path(anchor).resolve()), *sys.argv[1:]]
    os.execvpe("xvfb-run", argv, dict(os.environ, V400_D1_XVFB="1"))


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def phase_run(args) -> dict[str, Any]:
    work = work_dir_of(args)
    plan = _read_json(work / f"plan.{args.plan_name}.json")
    if not plan:
        raise SystemExit(f"{work}/plan.{args.plan_name}.json: run --phase plan first")
    workers = int(args.workers)
    if workers > 4 and not args.allow_over_engine_budget:
        raise SystemExit(
            f"--workers {workers} exceeds this task's WSL engine budget (4); pass "
            "--allow-over-engine-budget only if the负责人 raised it")
    specs = list(plan.get("shards") or [])
    pending: list[tuple[dict[str, Any], Path]] = []
    for spec in specs:
        report = _read_json(Path(spec["work_root"]) / "reports" / f"{spec['key']}.json")
        if report and report.get("status") == "ok":
            continue
        if dual().shard_resume_report(Path(spec["out_dir"]) / "shards" / spec["key"]):
            continue
        spec = dict(spec)
        spec["quarantine_partial"] = bool(args.retry_failed)
        path = work / "specs" / f"{spec['key']}.json"
        _write_json(path, spec)
        pending.append((spec, path))
    print(json.dumps({"phase": "run", "plan": args.plan_name, "shards_total": len(specs),
                      "pending": len(pending), "workers": workers,
                      "out_dir": str(out_dir_of(args))}, ensure_ascii=False), flush=True)
    if not pending:
        payload = {"schema": GATE_SCHEMA, "plan": args.plan_name, "pending": 0,
                   "ran": 0, "failed": [], "records": []}
        _write_json(work / f"run.{args.plan_name}.json", payload)
        return payload
    results: list[dict[str, Any]] = []
    lock = threading.Lock()
    queue = list(pending)

    def worker() -> None:
        while True:
            with lock:
                if not queue:
                    return
                spec, path = queue.pop(0)
            log_path = work / "logs" / f"{spec['key']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--phase", "shard", "--shard-spec", str(path), "--xvfb", "off"]
            began = time.monotonic()
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("COMMAND xvfb-run -a " + " ".join(command) + "\n")
                handle.flush()
                try:
                    proc = subprocess.run(
                        ["xvfb-run", "-a", *command], cwd=str(repo_root()),
                        stdout=handle, stderr=subprocess.STDOUT,
                        timeout=float(args.shard_timeout),
                        env=dict(os.environ, V400_D1_XVFB="1", V400_D0_XVFB="1"))
                    code = proc.returncode
                except subprocess.TimeoutExpired:
                    code = 124
                except Exception as error:  # pragma: no cover - infra failure
                    handle.write(f"\nHARNESS_ERROR {error!r}\n")
                    code = 125
            record = {"key": spec["key"], "stage": spec["stage"],
                      "start_mode": spec["start_mode"], "exit_code": code,
                      "wall_s": round(time.monotonic() - began, 3),
                      "log": str(log_path),
                      "report": str(Path(spec["work_root"]) / "reports"
                                    / f"{spec['key']}.json")}
            with lock:
                results.append(record)
                print(json.dumps({"phase": "run-shard", **record}, ensure_ascii=False),
                      flush=True)

    threads = [threading.Thread(target=worker, daemon=False)
               for _ in range(max(1, min(workers, len(pending))))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    payload = {"schema": GATE_SCHEMA, "plan": args.plan_name, "ran": len(results),
               "failed": [row for row in results if row["exit_code"] != 0],
               "records": sorted(results, key=lambda row: row["key"])}
    _write_json(work / f"run.{args.plan_name}.json", payload)
    if payload["failed"]:
        print(json.dumps({"phase": "run", "status": "FAILED",
                          "failed": len(payload["failed"])}, ensure_ascii=False), flush=True)
    return payload


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------
def _shard_report(shard_dir: Path) -> dict[str, Any]:
    raw = _read_json(shard_dir / "shard.json", default={}) or {}
    obs = raw.get("obs") or {}
    alignment = ((raw.get("alignment") or {}).get("hash_checkpoints") or {})
    frames = raw.get("stage_frames") or [0, 0]
    # Shards written before the verdict keys were lifted to the top level carry them
    # only in the channel-A report; read both, never invent a value.
    channel_a = (raw.get("report") or {}).get("channel_a") or {}
    return {
        "shard": raw.get("shard") or shard_dir.name,
        "dir": str(shard_dir),
        "stage": int(raw.get("stage", raw.get("stage_index", -1))),
        "seed": raw.get("seed"),
        "mode": raw.get("mode"),
        "start_mode": raw.get("start_mode", "shard"),
        "start_frame": int(raw.get("start_frame") or 0),
        "decisions": int(raw.get("decision_count") or raw.get("rows") or 0),
        "frames": int(frames[1]) - int(frames[0]),
        "frames_per_decision": int(raw.get("frames_per_decision") or 0),
        "obs_spec_version": raw.get("obs_spec_version"),
        "obs_spec_digest": raw.get("obs_spec_digest"),
        "obs_bytes": int(obs.get("bytes_total") or 0),
        "obs_bytes_per_decision": obs.get("bytes_per_decision"),
        "labels_bytes": int((raw.get("labels") or {}).get("bytes") or 0),
        "labels_sha256": (raw.get("labels") or {}).get("sha256"),
        "hash_boundaries": int(alignment.get("boundaries") or 0),
        "hash_equal": int(alignment.get("checkpoints_equal") or 0),
        "hash_mismatched": int(alignment.get("checkpoints_mismatched") or 0),
        "stage_clear": raw.get("stage_clear", channel_a.get("stage_clear")),
        "stop_reason": raw.get("stop_reason") or channel_a.get("stop_reason"),
        "deaths_observed": raw.get("deaths_observed",
                                   channel_a.get("deaths_observed")),
        "frames_committed": raw.get("frames_committed",
                                    channel_a.get("frames_committed")),
        "stopped_early": raw.get("stopped_early", channel_a.get("stopped_early")),
        "transition_trim": raw.get("transition_trim"),
        "trajectory": raw.get("trajectory"),
        "shard_json_bytes": (shard_dir / "shard.json").stat().st_size,
    }


def phase_gate(args) -> dict[str, Any]:
    work = work_dir_of(args)
    out_dir = out_dir_of(args)
    shards_root = out_dir / "shards"
    if not shards_root.is_dir():
        raise SystemExit(f"{shards_root}: no shards; run --phase run first")
    plan = _read_json(work / f"plan.{args.plan_name}.json", default={}) or {}
    planned_keys = [spec["key"] for spec in plan.get("shards") or []]
    reports: list[dict[str, Any]] = []
    labels_paths: list[Path] = []
    missing: list[str] = []
    for key in planned_keys:
        shard_dir = shards_root / key
        if not (shard_dir / "shard.json").is_file():
            missing.append(key)
            continue
        entry = _shard_report(shard_dir)
        detail = _read_json(Path(plan["shards"][planned_keys.index(key)]["work_root"])
                            / "reports" / f"{key}.json", default={}) or {}
        entry["detail"] = None if not detail else {
            "status": detail.get("status"),
            "load": detail.get("load"),
            "teacher_fps": detail.get("teacher_fps"),
            "env_fps": detail.get("env_fps"),
            "channel_a_wall_s": detail.get("channel_a_wall_s"),
            "env_wall_s": detail.get("env_wall_s"),
            "wall_s": detail.get("wall_s"),
            "prefix": detail.get("prefix"),
        }
        labels_paths.append(shard_dir / "labels.jsonl")
        reports.append(entry)
    if not reports:
        raise SystemExit(f"{shards_root}: no shard of plan {args.plan_name}")
    from touhou_rl.v400.student import obs_v3_dataset

    total_decisions = sum(int(row["decisions"]) for row in reports)
    total_frames = sum(int(row["frames"]) for row in reports)
    loaded = obs_v3_dataset.load_teacher_samples_v3(
        labels_paths, min_samples=total_decisions, obs_root=out_dir,
        require_observations=True)
    loaded_shards = {entry["shard"]: obs_v3_dataset.load_teacher_shard_v3(
        Path(entry["dir"])).report() for entry in reports}
    actions = np.zeros(len(obs_spec_v3.ACTION_HISTORY_ORDER), dtype=np.int64)
    bombs = 0
    for path in labels_paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("kind") != "decision":
                    continue
                plan_row = payload.get("chosen_plan") or {}
                if "first_action" in plan_row:
                    actions[int(plan_row["first_action"])] += 1
                bombs += int(payload.get("bomb") or 0)
    boundaries = sum(int(row["hash_boundaries"]) for row in reports)
    equal = sum(int(row["hash_equal"]) for row in reports)
    per_stage: dict[int, dict[str, Any]] = {}
    for entry in reports:
        stage = int(entry["stage"])
        bucket = per_stage.setdefault(stage, {
            "stage": stage, "shards": 0, "decisions": 0, "frames": 0,
            "hash_boundaries": 0, "hash_equal": 0, "obs_bytes": 0,
            "start_modes": {}, "deaths_observed": 0, "cleared": 0, "stop_reasons": {},
            "shard_keys": [],
        })
        bucket["shards"] += 1
        bucket["decisions"] += int(entry["decisions"])
        bucket["frames"] += int(entry["frames"])
        bucket["hash_boundaries"] += int(entry["hash_boundaries"])
        bucket["hash_equal"] += int(entry["hash_equal"])
        bucket["obs_bytes"] += int(entry["obs_bytes"])
        bucket["deaths_observed"] += int(entry["deaths_observed"] or 0)
        bucket["start_modes"][entry["start_mode"]] = \
            bucket["start_modes"].get(entry["start_mode"], 0) + 1
        bucket["stop_reasons"][str(entry["stop_reason"])] = \
            bucket["stop_reasons"].get(str(entry["stop_reason"]), 0) + 1
        bucket["shard_keys"].append(entry["shard"])
        if entry["stage_clear"]:
            bucket["cleared"] += 1
    gate = {
        "schema": GATE_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plan": args.plan_name,
        "out_dir": str(out_dir),
        "shards": len(reports),
        "planned_shards": len(planned_keys),
        "planned_missing": missing,
        "decisions": total_decisions,
        "frames": total_frames,
        "hash": {"boundaries": boundaries, "equal": equal,
                 "mismatched": boundaries - equal,
                 "alignment_rate": (round(equal / boundaries, 9) if boundaries else None),
                 "all_equal": bool(boundaries) and equal == boundaries},
        "load_gate": {
            "require_observations": True,
            "samples": int(loaded["count"]),
            "min_samples": total_decisions,
            "passed": int(loaded["count"]) == total_decisions,
            "obs_spec_version": loaded["obs_spec_version"],
            "obs_spec_digest": loaded["obs_spec_digest"],
            "digest_matches_frozen": loaded["obs_spec_digest"] == FROZEN_OBS_SPEC_V3_DIGEST,
            "unresolved": loaded["obs_unresolved"],
            "shards_loaded": len(loaded_shards),
            "play_shape": loaded["obs_shapes"]["play"] if loaded["obs_shapes"] else None,
            "hud_shape": loaded["obs_shapes"]["hud"] if loaded["obs_shapes"] else None,
            "act_hist_shape": (loaded["obs_shapes"]["act_hist"]
                               if loaded["obs_shapes"] else None),
        },
        "actions": {"order": obs_spec_v3.ACTION_HISTORY_ORDER,
                    "counts": actions.tolist(), "total": int(actions.sum()),
                    "top1_share": (round(float(actions.max() / actions.sum()), 6)
                                   if actions.sum() else None),
                    "bombs": int(bombs)},
        "per_stage": [per_stage[stage] for stage in sorted(per_stage)],
        "shard_reports": reports,
    }
    _write_json(work / f"gate.{args.plan_name}.json", gate)
    print(json.dumps({"phase": "gate", "plan": args.plan_name, "shards": gate["shards"],
                      "planned_missing": missing, "decisions": gate["decisions"],
                      "frames": gate["frames"],
                      "hash_alignment": gate["hash"]["alignment_rate"],
                      "load_gate": gate["load_gate"]["passed"],
                      "digest": gate["load_gate"]["obs_spec_digest"][:12],
                      "unresolved": len(gate["load_gate"]["unresolved"])},
                     ensure_ascii=False), flush=True)
    return gate


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def estimate_full_plan(args, gate: dict[str, Any], reachability: dict[str, Any],
                       plan: dict[str, Any]) -> dict[str, Any]:
    """Full D1 plan: shards, wall clock at N engines, disk, D-05 ratio.

    The rates are the ones *this* run measured (channel A and channel B wall clocks of
    the validation shards); nothing is extrapolated from a nominal fps.
    """

    details = list(plan.get("stages_detail") or [])
    report_details = [entry.get("detail") or {} for entry in gate.get("shard_reports") or []]
    teacher_fps = [float(row["teacher_fps"]) for row in report_details
                   if row.get("teacher_fps")]
    env_fps = [float(row["env_fps"]) for row in report_details if row.get("env_fps")]
    walls = [float(row["wall_s"]) for row in report_details if row.get("wall_s")]
    shadow_rate = float(np.median(teacher_fps)) if teacher_fps else None
    env_rate = float(np.median(env_fps)) if env_fps else None
    measured_prefix = [int(row.get("prefix", {}).get("frames") or 0)
                       for row in report_details if row.get("prefix")]
    fixed_overhead = None
    if walls and shadow_rate and env_rate and gate.get("frames"):
        # wall = overhead + frames/shadow + frames/env on the validation shards
        per_frame = 1.0 / shadow_rate + 1.0 / env_rate
        overheads = [wall - gate["frames"] / len(walls) * per_frame for wall in walls]
        fixed_overhead = float(np.median(overheads))
    per_shard_wall: dict[int, float] = {}
    total_wall = 0.0
    trajectory_share = (1.0 / len(plan["start_mix"])
                        if plan.get("start_mix") and "trajectory" in plan["start_mix"] else 0.0)
    prefix = (int(np.median(measured_prefix))
              if (measured_prefix and trajectory_share) else 0)
    for row in details:
        stage = int(row["stage"])
        frames = int(row["frames_per_shard"])
        if shadow_rate and env_rate:
            # channel A pays every shipped frame plus the take-over prefix; channel B
            # renders every one of them again (the fast-forward renders too).
            wall = ((frames + prefix) / shadow_rate + (frames + prefix) / env_rate
                    + (fixed_overhead or 0))
        else:  # pragma: no cover - only before a run
            wall = 0.0
        per_shard_wall[stage] = round(wall, 1)
        total_wall += wall * int(row["shards_needed"])
    engines = (4, 8, 12, 16)
    wall_by_engines = {str(count): round(total_wall / count, 1) for count in engines}
    # Disk: measured bytes/decision (planes + labels + shard.json) x decisions.
    obs_bytes = sum(int(row["obs_bytes"]) for row in gate.get("shard_reports") or [])
    label_bytes = sum(int(row["labels_bytes"]) for row in gate.get("shard_reports") or [])
    shard_json_bytes = sum(int(row["shard_json_bytes"]) for row in gate.get("shard_reports")
                           or [])
    decisions_measured = max(1, int(gate.get("decisions") or 0))
    per_decision = (obs_bytes + label_bytes + shard_json_bytes) / decisions_measured
    frames_total = sum(int(row["shards_needed"]) * int(row["frames_per_shard"])
                       for row in details)
    decisions_total = frames_total // FRAMES_PER_DECISION
    disk_bytes = int(per_decision * decisions_total * (1 + trajectory_share))
    return {
        "stages": [int(row["stage"]) for row in details],
        "per_stage_target_frames": int(plan["targets"]["per_stage"]),
        "shards_needed": sum(int(row["shards_needed"]) for row in details),
        "frames_by_stage": {int(row["stage"]): int(row["shards_needed"])
                            * int(row["frames_per_shard"]) for row in details},
        "frames_total": frames_total,
        "decisions_total": decisions_total,
        "prefix_frames_paid_note": "trajectory shards pay their prefix again; the "
                                   "estimate multiplies disk/wall by (1 + trajectory share)",
        "trajectory_share": round(trajectory_share, 4),
        "measured": {
            "teacher_fps_median": (round(shadow_rate, 3) if shadow_rate else None),
            "env_fps_median": (round(env_rate, 3) if env_rate else None),
            "wall_s_per_shard_median": round(float(np.median(walls)), 2) if walls else None,
            "fixed_overhead_s": (round(fixed_overhead, 2)
                                 if fixed_overhead is not None else None),
            "bytes_per_decision_measured": round(per_decision, 1),
            "decisions_measured": int(gate.get("decisions") or 0),
            "shards_measured": len(gate.get("shard_reports") or []),
            "prefix_frames_measured_median": (int(np.median(measured_prefix))
                                              if measured_prefix else None),
        },
        "wall_s_per_shard": per_shard_wall,
        "wall_s": wall_by_engines,
        "disk": {
            "bytes": disk_bytes,
            "gb": round(disk_bytes / 1e9, 3),
            "d05_budget_gb": round(DISK_BUDGET_BYTES / 1e9, 1),
            "d05_fraction": round(disk_bytes / DISK_BUDGET_BYTES, 6),
            "d05_alert_fraction": DISK_ALERT_FRACTION,
            "triggers_d05_alert": disk_bytes >= DISK_BUDGET_BYTES * DISK_ALERT_FRACTION,
        },
        "coverage": plan.get("coverage"),
    }


def phase_report(args) -> dict[str, Any]:
    work = work_dir_of(args)
    reachability = _read_json(work / "reachability.json")
    plan = _read_json(work / f"plan.{args.plan_name}.json")
    gate = _read_json(work / f"gate.{args.plan_name}.json")
    # A full-size plan has no run of its own: its rates come from the verification
    # run's gate (``--rates-plan``), and the payload says which run that was.
    rates_plan = str(getattr(args, "rates_plan", "") or args.plan_name)
    rates_gate = gate or _read_json(work / f"gate.{rates_plan}.json")
    if not rates_gate or not plan or not reachability:
        raise SystemExit("run --phase probe, plan, run, gate first")
    estimate = estimate_full_plan(args, rates_gate, reachability, plan)
    payload = {
        "schema": REPORT_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plan": args.plan_name,
        "rates_plan": rates_plan,
        "gate": {key: rates_gate[key] for key in
                 ("shards", "planned_shards", "planned_missing", "decisions", "frames",
                  "hash", "load_gate", "actions", "per_stage")},
        "reachability": reachability.get("stages"),
        "full_plan": estimate,
    }
    _write_json(work / f"report.{args.plan_name}.json", payload)
    _write_report_markdown(work / f"report.{args.plan_name}.md", payload, rates_gate)
    print(json.dumps({"phase": "report", "plan": args.plan_name,
                      "rates_plan": rates_plan,
                      "frames": rates_gate["frames"],
                      "decisions": rates_gate["decisions"],
                      "shards_needed_full": estimate["shards_needed"],
                      "wall_s_full": estimate["wall_s"],
                      "disk_gb": estimate["disk"]["gb"],
                      "d05_fraction": estimate["disk"]["d05_fraction"]},
                     ensure_ascii=False), flush=True)
    return payload


def _write_report_markdown(path: Path, payload: dict[str, Any],
                           gate: dict[str, Any]) -> None:
    estimate = payload["full_plan"]
    lines = ["# D1 collector report", "",
             f"- plan: `{payload['plan']}`  shards: {gate['shards']}/"
             f"{gate['planned_shards']}  decisions: {gate['decisions']}  "
             f"frames: {gate['frames']}",
             f"- HASH alignment: {gate['hash']['equal']}/{gate['hash']['boundaries']} = "
             f"{gate['hash']['alignment_rate']}",
             f"- load gate (require_observations=True): {gate['load_gate']['samples']} "
             f"samples, digest `{gate['load_gate']['obs_spec_digest']}` "
             f"(frozen={gate['load_gate']['digest_matches_frozen']}, "
             f"unresolved={len(gate['load_gate']['unresolved'])})",
             f"- actions top-1 share: {gate['actions']['top1_share']}  bombs: "
             f"{gate['actions']['bombs']}", "",
             "## Per stage", "",
             "| stage | shards | decisions | frames | start modes | stop reasons | "
             "HASH equal |", "|---|---|---|---|---|---|---|"]
    reach = {int(row["stage"]): row for row in payload.get("reachability") or []}
    for row in gate["per_stage"]:
        stage = int(row["stage"])
        lines.append(
            f"| {stage} | {row['shards']} | {row['decisions']} | {row['frames']} | "
            f"{row['start_modes']} | {row['stop_reasons']} | "
            f"{row['hash_equal']}/{row['hash_boundaries']} |")
    for stage, row in sorted(reach.items()):
        lines.append(f"\nstage {stage}: reachable {row.get('reachable_frames')} frames "
                     f"(lower_bound={row.get('reachable_is_lower_bound')}), "
                     f"clear={row.get('stage_clear')}, death_frame={row.get('death_frame')}")
    lines += ["", "## Full plan", "",
              f"- shards needed: {estimate['shards_needed']}",
              f"- frames: {estimate['frames_total']}  decisions: "
              f"{estimate['decisions_total']}",
              f"- measured: teacher {estimate['measured']['teacher_fps_median']} fps, "
              f"env {estimate['measured']['env_fps_median']} fps, "
              f"{estimate['measured']['wall_s_per_shard_median']} s/shard",
              f"- wall clock by engine count: {estimate['wall_s']}",
              f"- disk: {estimate['disk']['gb']} GB = {estimate['disk']['d05_fraction']} "
              f"of D-05 ({estimate['disk']['d05_budget_gb']} GB); "
              f"alert={estimate['disk']['triggers_d05_alert']}", ""]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True,
                        choices=("probe", "plan", "shard", "run", "gate", "report"))
    parser.add_argument("--stages", default=DEFAULT_STAGES)
    parser.add_argument("--seeds", default="1-8")
    parser.add_argument("--workers", type=int, default=4,
                        help="concurrent shard workers (= concurrent engines); this "
                             "task's WSL budget is 4")
    parser.add_argument("--allow-over-engine-budget", action="store_true")
    parser.add_argument("--frames-target-total", type=int, default=2_000_000)
    parser.add_argument("--frames-target-per-stage", type=int, default=150_000)
    parser.add_argument("--decisions-per-shard", type=int, default=4000)
    parser.add_argument("--trajectory-decisions-per-shard", type=int, default=1000,
                        help="a take-over shard must leave room for its whole "
                             "segment inside the recorded corpus, so it is "
                             "capped separately (0 = same as --decisions-per-shard)")
    parser.add_argument("--plan-limit-per-stage", type=int, default=0,
                        help="cap the shards per stage in the plan (0 = the full plan)")
    parser.add_argument("--plan-name", default="full")
    parser.add_argument("--start-mix", default="shard")
    parser.add_argument("--obs", default="v3", choices=("v3",))
    parser.add_argument("--mode", default="fast")
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--team", type=int, default=0)
    parser.add_argument("--engine", default=DEFAULT_ENGINE)
    parser.add_argument("--expected-sha", default=DEFAULT_ENGINE_SHA)
    parser.add_argument("--data-template", default="/root/v400-data/game")
    parser.add_argument("--data-root-prefix", default="/root/v400-data/d1")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--corpus-root", default=str(repo_root() / DEFAULT_CORPUS_ROOT))
    parser.add_argument("--corpus-roots", nargs="*", default=[],
                        help="extra directories with recorded shard-start shards")
    parser.add_argument("--probe-cap-frames", type=int, default=16000,
                        help="frame cap of the reachability episode (to death or cap)")
    parser.add_argument("--trajectory-bias", type=float, default=0.65)
    parser.add_argument("--trajectory-boss-weight", type=float, default=0.25)
    parser.add_argument("--boss-start", type=int, default=DEFAULT_BOSS_START,
                        help="-1 disables the boss-section weight")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--shard-timeout", type=float, default=3600.0)
    parser.add_argument("--shard-spec", default=None)
    parser.add_argument("--retry-failed", action="store_true",
                        help="run: rename an existing shards/.<key>.partial to a "
                             "timestamped .failed-<ts> name (evidence kept, never "
                             "deleted) so the shard can be retried")
    parser.add_argument("--rates-plan", default="",
                        help="report: take the measured frames/s + bytes/decision from "
                             "this plan's gate (a full-size plan has no run of its own)")
    parser.add_argument("--python", default="/root/v400-venv/bin/python")
    parser.add_argument("--xvfb", choices=("auto", "off"), default="auto")
    args = parser.parse_args(argv)
    args.difficulty = int(args.difficulty)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.phase == "shard":
        return phase_shard(args)
    if args.phase == "probe":
        phase_probe(args)
        return EXIT_OK
    if args.phase == "plan":
        phase_plan(args)
        return EXIT_OK
    if args.phase == "run":
        payload = phase_run(args)
        return EXIT_OK if not payload.get("failed") else EXIT_FAIL_CLOSED
    if args.phase == "gate":
        gate = phase_gate(args)
        ok = (gate["hash"]["all_equal"] and gate["load_gate"]["passed"]
              and gate["load_gate"]["digest_matches_frozen"]
              and not gate["planned_missing"])
        return EXIT_OK if ok else EXIT_FAIL_CLOSED
    if args.phase == "report":
        phase_report(args)
        return EXIT_OK
    raise SystemExit(f"unknown phase {args.phase!r}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AlignmentError as error:
        print(json.dumps({"status": "ALIGNMENT_FAILED", "reason": str(error)},
                         ensure_ascii=False), flush=True)
        sys.exit(EXIT_ALIGN)
    except (D1Error, ObservationError, TrajectoryStartError) as error:
        print(json.dumps({"status": "FAIL_CLOSED",
                          "reason": f"{type(error).__name__}: {error}"},
                         ensure_ascii=False), flush=True)
        sys.exit(EXIT_FAIL_CLOSED)
