"""``teacher-v1`` command line: run N real decisions and report every phase.

    python tools/rl/v400_teacher.py --decisions 100 --stage 1 --difficulty 1 \
        --seed 1 --mode full --engine /root/Touhou_RL/third_party/th08-rl/build/rl_hazard2

``--stage`` is the engine's **0-based** stage index (``RESET <stage>`` and
``STATE.f1``; ``third_party/th08/src/ScoreDat.hpp``: ``STAGE1 = 0``), so
``--stage 0`` is the human-visible Stage 1 and ``--stage 1`` in the example above
starts the real Stage 2.  A suite that means Stage 1 (``S1N``) must pass
``--stage 0``; EXP-P0-077 audited this label.

One JSON object per decision goes to ``--out`` (JSONL) and a one-line summary to
stdout, followed by a final report with the survival length, the four phase
timings, the decision rate, the real-time multiple and the geometry-versus-engine
consistency counts.

Two recording paths exist, and only one of them is the shipped one:

``child`` (default)
    ``PARK`` once per decision, ``SPAWN`` one copy, let the copy ``STEP`` frame by
    frame and answer ``CHILD HAZARDS``/``CHILD PLAYER`` (:mod:`.child_record`),
    drop the copy, then advance the authoritative process ``R`` sub-decisions
    (4 frames) and **never reset it**.  If ``probe_child_hazards()`` says the
    engine cannot serve ``CHILD HAZARDS`` the CLI exits non-zero (fail-closed)
    instead of silently recording on the main process.

``legacy_restore`` (``--allow-legacy-restore`` only)
    The EXP-P1-003/006 forward walk of the authoritative shadow + new process +
    ``RESET`` + replay per decision.  Kept for A/B equivalence measurements; it
    is *not* a fallback that runs automatically.

The engine binary is hashed at start-up: a build whose SHA-256 is not the one the
child-recording and ``EVENTS`` behaviour was verified on is refused (exit 4).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # allow ``python tools/rl/v400_teacher.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from touhou_rl.v400.gym.actions import cand_to_mask
from touhou_rl.v400.teacher.config import CONFIGS, get_config
from touhou_rl.v400.teacher.decide import decide
from touhou_rl.v400.teacher.engine import (
    ENEMY_FLAG_BOSS,
    ENGINE_BINARY_NAME,
    EXPECTED_ENGINE_SHA256,
    EngineError,
    EngineShaMismatch,
    TeacherEngine,
    check_engine_binary,
    parse_fields,
)
from touhou_rl.v400.teacher.hazard_tape import (
    ChildHazardsUnavailable,
    HazardExportUnavailable,
)
from touhou_rl.v400.teacher.obs import (
    ObservationError,
    ShmPixelSource,
    TeacherObservationRun,
    default_shm_name,
)
from touhou_rl.v400.teacher.shoot import (
    SHOOT_ALWAYS,
    commit_block,
    commands as step_commands,
    mask_summary,
    release_due,
)
from touhou_rl.v400.teacher.verify import Consistency

DIFFICULTIES = {"easy": 0, "normal": 1, "hard": 2, "lunatic": 3}

#: 60 fps is the real-time reference the engine's logic frames are defined at.
REFERENCE_FPS = 60.0

#: Default engine build: the one with ``CHILD <h> HAZARDS/PLAYER/EVENTS``.
DEFAULT_ENGINE = "/root/Touhou_RL/third_party/th08-rl/build/rl_hazard2"

EXIT_OK = 0
EXIT_STOPPED_EARLY = 2
EXIT_FAIL_CLOSED = 3
EXIT_ENGINE_MISMATCH = 4
EXIT_ENGINE_ERROR = 5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--decisions", type=int, default=10, help="how many decisions to run")
    parser.add_argument("--stage", type=int, default=1,
                        help="0-based engine stage index, 0..8 (0 = Stage 1; 1 = the real "
                             "Stage 2, which is this option's default)")
    parser.add_argument("--difficulty", default="1", help="0..3 or easy/normal/hard/lunatic")
    parser.add_argument("--team", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=tuple(sorted(CONFIGS)),
        default="full",
        help="teacher profile: full/fast (v1), v2a/v2b/v2c/v2d/v2e "
        "(EXP-P1-015 bottom band, EXP-P1-017 v2e Boss-body cost), v2f/v2g "
        "(EXP-P1-020 top band), v2h (EXP-P1-021 damageable-laser cost), "
        "v2i (EXP-P3-005 margin_cap 64), v2j (EXP-P3-007 first-action "
        "durations 1/2/3/4) or v3a/v3b (EXP-P1-019: v2a weights with a "
        "larger search budget)",
    )
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        help="build directory containing th08-modern (default: build/rl_hazard2)",
    )
    parser.add_argument(
        "--binary",
        default=None,
        help="explicit engine binary (overrides --engine; the SHA gate still applies)",
    )
    parser.add_argument(
        "--expected-sha",
        default=EXPECTED_ENGINE_SHA256,
        help="required SHA-256 of the engine binary (default: the rl_hazard2 build)",
    )
    parser.add_argument("--data-dir", default="/root/v400-data/p1s1/game")
    parser.add_argument("--out", type=Path, default=None, help="JSONL path for the labels")
    parser.add_argument(
        "--save-observations",
        type=Path,
        default=None,
        metavar="DIR",
        help="also write the obs-v1 observation of every decision as a P3 teacher "
        "shard under DIR (<DIR>/shards/<key>/{labels.jsonl,obs/play64.npy,"
        "obs/hud.npy,shard.json} + <DIR>/manifest.json).  The observation is taken "
        "at the decision point, before the commit, and scores[18] labels the action "
        "that is committed next (see touhou_rl/v400/teacher/obs.py)",
    )
    parser.add_argument(
        "--obs-shard-key",
        default=None,
        help="shard key under DIR/shards (default: "
        "teacher-<mode>-s<stage>d<difficulty>t<team>-seed<seed>)",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="label channel of the D0 dual-channel pipeline (EXP-P1-014): record "
        "every decision's 4 per-frame masks and the HASH at both sides of the "
        "decision boundary (before decide, after the 4-frame commit) so an "
        "independent env-role replay can be aligned checkpoint by checkpoint. "
        "Never touches pixels (the shadow role renders nothing).",
    )
    parser.add_argument(
        "--allow-legacy-restore",
        action="store_true",
        help="use the EXP-P1-003/006 main-process recording + RESET + replay path",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="stop after this many committed game frames (0 = no frame cap)",
    )
    parser.add_argument(
        "--stop-on-death",
        action="store_true",
        help="stop at the first committed death (STATE f6 increase)",
    )
    parser.add_argument("--probe-only", action="store_true", help="print capabilities and stop")
    parser.add_argument(
        "--shoot-pulse-period",
        type=int,
        default=None,
        help="EXP-P1-009: decisions between two SHOOT release frames "
        "(default: config.shoot_pulse_period; 0 disables the pulse)",
    )
    parser.add_argument(
        "--follow-stage",
        action="store_true",
        help="EXP-P1-011: keep playing after a stage change and record every "
        "transition (default: stop at the first one, as the stage-clear gate did)",
    )
    parser.add_argument("--quiet", action="store_true", help="only print the final report")
    return parser.parse_args(argv)


def _config_with_pulse(config, period):
    """``config`` with EXP-P1-009's shoot-pulse period overridden (``None`` keeps it)."""

    if period is None:
        return config
    from dataclasses import replace

    return replace(config, shoot_pulse_period=int(period))


def _difficulty(value) -> int:
    text = str(value).strip().lower()
    if text in DIFFICULTIES:
        return DIFFICULTIES[text]
    number = int(text)
    if not 0 <= number <= 3:
        raise SystemExit(f"difficulty out of range: {value}")
    return number


def _resolve_engine(args) -> tuple[str, str]:
    """``(engine_dir, binary)`` from ``--engine``/``--binary``."""

    if args.binary:
        binary = str(args.binary)
        return str(Path(binary).parent), binary
    engine_dir = str(args.engine)
    return engine_dir, str(Path(engine_dir) / ENGINE_BINARY_NAME)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = _config_with_pulse(get_config(args.mode), args.shoot_pulse_period)
    difficulty = _difficulty(args.difficulty)
    engine_dir, binary = _resolve_engine(args)
    if args.labels_only:
        # The label channel must be one contiguous single-stage segment on the
        # authoritative process: a replay on another process can only be aligned
        # to it frame by frame if no decision moves the shadow itself.
        blocked = None
        if args.save_observations is not None:
            blocked = "--labels-only and --save-observations are different channels"
        elif args.allow_legacy_restore:
            blocked = "--allow-legacy-restore replays the shadow per decision"
        elif args.follow_stage:
            blocked = "--follow-stage inserts stage-load frames"
        if blocked:
            print(json.dumps({"status": "fail_closed",
                              "reason": f"--labels-only: {blocked}"},
                             ensure_ascii=False), flush=True)
            return EXIT_FAIL_CLOSED
    if args.save_observations is not None:
        # The observation contract is one contiguous, single-stage segment starting
        # at decision 0: the legacy path replays the shadow H frames per decision
        # (the pixels would be a different state than the label) and --follow-stage
        # inserts unpixelised stage-load frames (the frame/decision arithmetic
        # would break).  Refusing is cheaper than a shard nobody can align.
        blocked = None
        if args.allow_legacy_restore:
            blocked = "--allow-legacy-restore replays the shadow per decision; its "
            "frames are not the frames the teacher was queried on"
        elif args.follow_stage:
            blocked = "--follow-stage inserts stage-load frames that are not "
            "observed; one shard must be a contiguous single-stage segment"
        if blocked:
            print(json.dumps({"status": "fail_closed",
                              "reason": f"--save-observations: {blocked}"},
                             ensure_ascii=False), flush=True)
            return EXIT_FAIL_CLOSED
    try:
        gate = check_engine_binary(binary, args.expected_sha)
    except EngineShaMismatch as error:
        print(
            json.dumps(
                {
                    "status": "engine_sha_mismatch",
                    "reason": str(error),
                    "engine_dir": engine_dir,
                    "binary": binary,
                    "expected_sha256": args.expected_sha,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return EXIT_ENGINE_MISMATCH
    gate["engine_dir"] = engine_dir
    try:
        if args.allow_legacy_restore:
            return _run_legacy(args, config, difficulty, gate)
        return _run_child(args, config, difficulty, gate)
    except EngineError as error:
        print(
            json.dumps({"status": "engine_error", "reason": str(error)[:400]}, ensure_ascii=False),
            flush=True,
        )
        return EXIT_ENGINE_ERROR


# ---------------------------------------------------------------------------
# child recording: one process, one RESET, per-decision PARK + SPAWN
# ---------------------------------------------------------------------------
def _start_observations(args, config, difficulty: int, gate: dict, engine) -> TeacherObservationRun:
    """Create the obs run and attach the shared block (must precede ``RESET``).

    Readback is enabled before ``RESET`` so the setup frames publish: the render
    left in the block at reset completion is the episode-start padding row of
    decision 0 (``obs-spec-v1.md`` §2).  It is disabled again immediately after,
    so the recording/verification fork children never pay glReadPixels and never
    overwrite the block.
    """

    key = args.obs_shard_key or (
        f"teacher-{config.mode}-s{int(args.stage)}d{int(difficulty)}t{int(args.team)}"
        f"-seed{int(args.seed)}")
    meta = {
        "stage": int(args.stage),
        "difficulty": int(difficulty),
        "team": int(args.team),
        "seed": int(args.seed),
        "mode": config.mode,
        "engine": gate,
        "generated_by": "tools/rl/v400_teacher.py --save-observations",
        "config": {
            "horizon": int(config.horizon),
            "beam": int(config.beam),
            "verify_k": int(config.verify_k),
            "replan_decisions": int(config.replan_decisions),
            "decision_frames": int(config.decision_frames),
            "durations": list(config.durations),
            "shoot_pulse_period": int(config.shoot_pulse_period),
        },
    }
    source = ShmPixelSource(engine.ask, name=default_shm_name())
    run = TeacherObservationRun(
        out_root=args.save_observations,
        key=key,
        meta=meta,
        source=source,
        frames_per_decision=int(config.replan_decisions * config.decision_frames),
    )
    run.attach()
    return run


def _run_child(args, config, difficulty: int, gate: dict) -> int:
    consistency = Consistency()
    decisions: list[dict] = []
    events: list[str] = []
    engine = None
    capabilities: dict = {}
    record_ms = search_ms = verify_ms = commit_ms = events_ms = 0.0
    wall_ms = 0.0
    frames_committed = 0
    bomb_decisions = 0
    probe_result: dict | None = None
    state_start: dict | None = None
    state_last: dict | None = None
    stop_reason = "decisions"
    stage_clear = False
    shoot_releases = 0
    shoot_mask_summary = {"frames": 0, "shoot_frames": 0, "bomb_frames": 0}
    empty_tape_decisions = 0
    empty_tape_streak = 0
    empty_tape_streak_max = 0
    stage_frames_seen: list[int] = []
    observe_ms = 0.0
    stage_transitions: list[dict] = []
    stage_stats: list[dict] = []
    stage_index = int(args.stage)
    stage_entry_frame = 0
    stage_entry_deaths = 0.0
    stage_entry_bombs = None
    boss_decisions = 0
    boss_first: dict | None = None
    boss_life_max = 0
    enemy_count_max = 0
    enemy_life_sum_max = 0
    enemies_observed = 0
    enemies_available = None
    obs_run: TeacherObservationRun | None = None
    obs_summary: dict | None = None
    try:
        engine = TeacherEngine(gate["path"], args.data_dir)
        started = time.perf_counter()
        engine.start()
        engine.hello()
        start_ms = (time.perf_counter() - started) * 1000.0
        if args.save_observations is not None:
            obs_run = _start_observations(args, config, difficulty, gate, engine)
        started = time.perf_counter()
        info = engine.reset(
            stage=args.stage, difficulty=difficulty, team=args.team, seed=args.seed
        )
        reset_ms = (time.perf_counter() - started) * 1000.0
        if info.get("ready") != "1":
            raise SystemExit(f"RESET did not reach a playable state: {info}")
        if obs_run is not None:
            obs_run.seed_after_reset(state_frame=int(engine.frame()))
        capabilities = engine.probe_capabilities()
        capabilities["child_hazards"] = engine.probe_child_hazards()
        if args.probe_only:
            print(
                json.dumps(
                    {
                        "status": "probe",
                        "engine": gate,
                        "capabilities": capabilities,
                        "start_hello_ms": round(start_ms, 3),
                        "reset_ms": round(reset_ms, 3),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
            return EXIT_OK
        if not capabilities["child_hazards"].get("available"):
            print(
                json.dumps(
                    {
                        "status": "fail_closed",
                        "reason": "engine cannot serve CHILD <h> HAZARDS; refusing to "
                        "record on the main process (use --allow-legacy-restore to "
                        "ask for the old path explicitly)",
                        "child_hazards": capabilities["child_hazards"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return EXIT_FAIL_CLOSED

        history: list[tuple[int, int]] = []
        commits: list[int] = []
        baseline_state = engine.state()
        state_start = _state_snapshot(baseline_state)
        state_last = state_start
        deaths_baseline = _float_or_none(baseline_state.get("deaths")) or 0.0
        bombs_initial = _float_or_none(baseline_state.get("bombs")) or 0.0
        frame_start = int(state_start["frame"] or 0)
        stage_entry_deaths = deaths_baseline
        stage_entry_bombs = bombs_initial

        for index in range(args.decisions):
            started = time.perf_counter()
            # EXP-P1-009: every `shoot_pulse_period` decisions the last committed
            # frame drops SHOOT, so the next frame is the rising edge that
            # GameManager.cpp:381-389 needs before a stage can change.
            release = release_due(index, config.shoot_pulse_period)
            if obs_run is not None:
                # Frame gate + the stack the teacher is about to be queried on:
                # the 4 renders ending at ``frames_committed - 1``.
                obs_run.begin_decision(engine, frames_committed)
            # EXP-P1-014: the decision boundary's state hash, before the teacher is
            # queried.  The env-role replay of the same mask sequence must produce
            # the same hash at the same frame or the two channels are not aligned.
            boundary_pre = _hash_checkpoint(engine) if args.labels_only else None
            decision = decide(
                engine,
                config,
                committed=commits,
                frame=frames_committed,
                stage=args.stage,
                difficulty=difficulty,
                consistency=consistency,
                record_source="child",
                shoot_release=release,
            )
            if obs_run is not None:
                # Recording/verification fork children; the shadow must not move.
                obs_run.verify_frame_unchanged(engine)
            chosen = decision.chosen
            if chosen is None:
                if obs_run is not None:
                    obs_run.drop_pending("no_plan")
                events.append(f"no_plan_at_{index}")
                stop_reason = "no_plan"
                break
            bomb = int(decision.label.bomb)
            actions = list(chosen.slots[: config.replan_decisions])
            commits = actions + commits[len(actions):]

            # R sub-decisions (4 frames) on the same process, never a reset.  The
            # bomb bit rides the first sub-decision only: 11-P1 §4 says "the next
            # decision presses bomb", not "hold bomb for the whole plan".
            block = commit_block(
                actions,
                bomb=bomb,
                decision_frames=config.decision_frames,
                release_shoot=release,
            )
            expected_advanced = sum(int(count) for _, count in block)
            commit_started = time.perf_counter()
            if obs_run is not None:
                # One STEP per frame + one pixel read per frame: the only way to get
                # 4 *distinct* renders (a STEP of 2 only leaves the last one in shm).
                commit = obs_run.commit(engine, block)
                steps = commit["steps"]
                advanced = int(commit["advanced"])
                exited = bool(commit["exited"])
                commit_complete = bool(commit["complete"])
            else:
                commands = step_commands(block)
                steps = engine.ask_many(commands)
                advanced = sum(int(parse_fields(reply).get("advanced", 0)) for reply in steps)
                exited = any(parse_fields(reply).get("exit") == "1" for reply in steps)
                commit_complete = advanced == expected_advanced and not exited
            # EXP-P1-014: the other side of the same decision boundary (frame d+4).
            boundary_post = _hash_checkpoint(engine) if args.labels_only else None
            committed_state = engine.state()
            commit_ms += (time.perf_counter() - commit_started) * 1000.0
            state_now = _state_snapshot(committed_state)
            # EXP-P1-011 §2: ``ENEMIES`` is read-only (fork.cpp:345) and is the
            # only boss / enemy-life observation the protocol exports; ``STATE``'s
            # 25 fields carry neither.  Sampled after the commit so it describes
            # the frame the decision ends on.
            observe_started = time.perf_counter()
            enemy_now = engine.enemies_snapshot()
            debug_now = _debug_snapshot(engine)
            observe_ms += (time.perf_counter() - observe_started) * 1000.0
            for slot_index, cand in enumerate(actions):
                history.append((int(cand), bomb if slot_index == 0 else 0))
            frames_committed += advanced
            if bomb:
                bomb_decisions += 1
            spent = (time.perf_counter() - started) * 1000.0
            wall_ms += spent
            record_ms += decision.timing_ms["record"]
            search_ms += decision.timing_ms["search"]
            verify_ms += decision.timing_ms["verify"]
            events_ms += decision.timing_ms["events"]

            payload = _decision_payload(index, frames_committed, decision, state_now)
            payload["timing_ms"]["commit"] = round(
                (time.perf_counter() - commit_started) * 1000.0, 3
            )
            payload["timing_ms"]["total"] = round(spent, 3)
            payload["step_replies"] = [reply[:120] for reply in steps]
            payload["step_masks"] = [f"0x{int(mask):04x}x{int(count)}" for mask, count in block]
            payload["shoot_release"] = bool(release)
            payload["shoot_frames"] = mask_summary(block)
            if args.labels_only:
                # The dual-channel label channel's own record: the exact per-frame
                # masks the env replay must step, and both boundary hashes.
                payload["channel"] = "shadow-labels"
                payload["step_frames"] = [
                    int(mask) for mask, count in block for _ in range(int(count))
                ]
                payload["hash_pre"] = boundary_pre
                payload["hash_post"] = boundary_post
                # ``geo_score`` is the 4-decimal CLI view; the D0 label row uses
                # ``TeacherLabel.scores`` at 6 decimals (EXP-P1-013 §2).
                payload["scores"] = [round(float(value), 6)
                                     for value in decision.label.scores]
            payload["enemies"] = _enemy_summary(enemy_now)
            payload["debug"] = debug_now
            if enemy_now.get("available"):
                enemies_observed += 1
                enemies_available = True
                enemy_count_max = max(enemy_count_max, int(enemy_now.get("count") or 0))
                enemy_life_sum_max = max(
                    enemy_life_sum_max, int(enemy_now.get("life_sum") or 0)
                )
                if int(enemy_now.get("boss_count") or 0) > 0:
                    boss_decisions += 1
                    boss_life_max = max(boss_life_max, int(enemy_now.get("boss_life") or 0))
                    if boss_first is None:
                        boss_first = {
                            "decision": index,
                            "frame": frames_committed,
                            "life": int(enemy_now.get("boss_life") or 0),
                            "slots": list(enemy_now.get("boss_slots") or ()),
                        }
            elif enemies_available is None:
                enemies_available = False
            if release:
                shoot_releases += 1
            shoot_mask_summary["frames"] += payload["shoot_frames"]["frames"]
            shoot_mask_summary["shoot_frames"] += payload["shoot_frames"]["shoot_frames"]
            shoot_mask_summary["bomb_frames"] += payload["shoot_frames"]["bomb_frames"]
            decisions.append(payload)
            if obs_run is not None:
                if commit_complete:
                    row = obs_run.add_sample(index=index, label=decision.label)
                    payload["obs"] = row["obs"]
                else:
                    # A commit the engine cut short has no next 4 frames, so this
                    # decision's sample is dropped and the shard ends at the last
                    # complete one (never a half row).
                    obs_run.drop_pending(
                        f"commit_incomplete:advanced={advanced},exited={exited}")
            state_last = state_now
            boxes = payload["tape_boxes"]
            if boxes["frames"] and boxes["empty_frames"] == boxes["frames"]:
                empty_tape_decisions += 1
                empty_tape_streak += 1
                empty_tape_streak_max = max(empty_tape_streak_max, empty_tape_streak)
            else:
                empty_tape_streak = 0
            if state_now["stage_frames"] is not None:
                stage_frames_seen.append(int(state_now["stage_frames"]))
            if not args.quiet:
                print(
                    json.dumps(
                        {
                            "decision": index,
                            "frame": frames_committed,
                            "chosen": payload["chosen_plan"]["first_action"]
                            if payload["chosen_plan"]
                            else None,
                            "bomb": payload["bomb"],
                            "event_flags": payload["frame_event_flags"],
                            "deaths": state_now["deaths"],
                            "timing_ms": payload["timing_ms"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            # ---- stop conditions, in the order the task states them ------
            if exited:
                stop_reason = "engine_exit_game_over"
                events.append(f"engine_exit_at_{index}")
                break
            if advanced != config.replan_decisions * config.decision_frames:
                stop_reason = "engine_stopped_short"
                events.append(f"advanced_{advanced}_at_{index}")
                break
            deaths_now = _float_or_none(committed_state.get("deaths")) or 0.0
            if deaths_now > deaths_baseline:
                events.append(f"death_committed_at_{index}")
                if args.stop_on_death:
                    stop_reason = "death"
                    break
            stage_now = state_now["stage"]
            if stage_now is not None and int(stage_now) != int(stage_index):
                # EXP-P1-009 §4 / EXP-P1-011 §1: the game itself advanced the
                # stage index, so this is a completed stage, not merely "still
                # alive" — the two verdicts are kept apart on purpose.
                stage_clear = True
                bombs_now = _float_or_none(committed_state.get("bombs"))
                transition = _stage_transition(
                    decision=index,
                    frame=frames_committed,
                    from_stage=int(stage_index),
                    to_stage=int(stage_now),
                    entry_frame=stage_entry_frame,
                    entry_deaths=stage_entry_deaths,
                    entry_bombs=stage_entry_bombs,
                    deaths_now=deaths_now,
                    bombs_now=bombs_now,
                )
                stage_transitions.append(transition)
                stage_stats.append(dict(transition))
                events.append(
                    f"stage_changed_{transition['from_stage']}_to_"
                    f"{transition['to_stage']}_at_{index}"
                )
                stage_entry_frame = int(frames_committed)
                stage_entry_deaths = deaths_now
                stage_entry_bombs = bombs_now
                stage_index = int(stage_now)
                if not args.follow_stage:
                    stop_reason = "stage_changed"
                    break
                # EXP-P1-011 §6.2: the next stage runs its setup thread, and the
                # harness refuses PARK while it does (measured:
                # `ERR park-unsafe ... threads=2 setup=1`).  Walk the load out
                # before asking for the next decision, otherwise the run dies
                # with exit 5 right after the first completed stage.
                waited_started = time.perf_counter()
                wait_frames, playable = _advance_until_playable(engine)
                frames_committed += wait_frames
                observe_ms += (time.perf_counter() - waited_started) * 1000.0
                events.append(f"stage_load_wait_{wait_frames}_frames")
                if not playable:
                    events.append("stage_load_not_ready")
                    stop_reason = "stage_load_not_ready"
                    break
                stage_entry_frame = int(frames_committed)
                ready_state = engine.state()
                stage_entry_deaths = _float_or_none(ready_state.get("deaths")) or stage_entry_deaths
                stage_entry_bombs = _float_or_none(ready_state.get("bombs"))
                state_last = _state_snapshot(ready_state)
            if (
                args.max_frames
                and frames_committed >= args.max_frames
            ):
                stop_reason = "max_frames"
                break
        else:
            stop_reason = "decisions"
        if obs_run is not None:
            # Only a shard whose every stored row has its 4 render frames is
            # finalised; anything else is left in the .partial staging directory.
            obs_summary = obs_run.finalize(
                report={
                    "stop_reason": stop_reason,
                    "decisions": len(decisions),
                    "frames_committed": int(frames_committed),
                    "stage_clear": bool(stage_clear),
                    "seed": int(args.seed),
                    "stage": int(args.stage),
                    "difficulty": int(difficulty),
                    "engine_sha256": gate.get("sha256"),
                }
            )
        # The stage the run stopped in has no closing transition, so it is
        # reported as an open row: its frame/death/bomb deltas are what a G1
        # "deaths per stage" table needs for the last stage as well.
        deaths_final_open = _float_or_none((state_last or {}).get("deaths"))
        bombs_final_open = _float_or_none((state_last or {}).get("bombs"))
        stage_stats.append(
            dict(
                _stage_transition(
                    decision=len(decisions) - 1,
                    frame=frames_committed,
                    from_stage=int(stage_index),
                    to_stage=None,
                    entry_frame=stage_entry_frame,
                    entry_deaths=stage_entry_deaths,
                    entry_bombs=stage_entry_bombs,
                    deaths_now=deaths_final_open,
                    bombs_now=bombs_final_open,
                ),
                open=True,
            )
        )
        return _finish(
            args=args,
            config=config,
            difficulty=difficulty,
            gate=gate,
            record_source="child",
            decisions=decisions,
            consistency=consistency,
            capabilities=capabilities,
            events=events,
            phases={
                "record": record_ms,
                "search": search_ms,
                "verify": verify_ms,
                "commit": commit_ms,
                "observe": observe_ms,
                "events": events_ms,
                "restore": 0.0,
            },
            wall_ms=wall_ms,
            frames_committed=frames_committed,
            state_start=state_start,
            state_last=state_last,
            deaths_baseline=deaths_baseline,
            bombs_initial=bombs_initial,
            bomb_decisions=bomb_decisions,
            stop_reason=stop_reason,
            stage_clear=stage_clear,
            start_ms=start_ms,
            reset_ms=reset_ms,
            processes=1,
            engine=engine,
            shoot_releases=shoot_releases,
            shoot_mask_summary=shoot_mask_summary,
            empty_tape_decisions=empty_tape_decisions,
            empty_tape_streak_max=empty_tape_streak_max,
            stage_frames_seen=stage_frames_seen,
            stage_transitions=stage_transitions,
            stage_stats=stage_stats,
            enemies_available=enemies_available,
            enemies_observed=enemies_observed,
            enemy_count_max=enemy_count_max,
            enemy_life_sum_max=enemy_life_sum_max,
            boss_decisions=boss_decisions,
            boss_first=boss_first,
            boss_life_max=boss_life_max,
            observation=obs_summary,
        )
    except ObservationError as error:
        detail = obs_run.abort(error) if obs_run is not None else None
        print(
            json.dumps(
                {
                    "status": "fail_closed",
                    "reason": f"observations: {error}",
                    "incomplete_shard": detail,
                    "decisions": len(decisions),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return EXIT_FAIL_CLOSED
    except ChildHazardsUnavailable as error:
        print(
            json.dumps(
                {
                    "status": "fail_closed",
                    "reason": str(error),
                    "engine": gate,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return EXIT_FAIL_CLOSED
    except HazardExportUnavailable as error:
        print(
            json.dumps({"status": "fail_closed", "reason": str(error)}, ensure_ascii=False),
            flush=True,
        )
        return EXIT_FAIL_CLOSED
    finally:
        if obs_run is not None:
            try:
                obs_run.close()
            except Exception:  # pragma: no cover - closing must not mask the verdict
                pass
        if engine is not None:
            engine.close()


# ---------------------------------------------------------------------------
# legacy restore: new process + RESET + replay per decision (A/B only)
# ---------------------------------------------------------------------------
def _run_legacy(args, config, difficulty: int, gate: dict) -> int:
    consistency = Consistency()
    decisions: list[dict] = []
    events: list[str] = []
    engine = None
    capabilities: dict = {}
    record_ms = search_ms = verify_ms = commit_ms = events_ms = restore_ms = 0.0
    wall_ms = 0.0
    frames_committed = 0
    bomb_decisions = 0
    state_start: dict | None = None
    state_last: dict | None = None
    stop_reason = "decisions"
    stage_clear = False
    deaths_baseline = 0.0
    bombs_initial = 0.0
    start_ms = reset_ms = 0.0
    try:
        history: list[tuple[int, int]] = []
        commits: list[int] = []
        for index in range(args.decisions):
            started = time.perf_counter()
            engine = TeacherEngine(gate["path"], args.data_dir)
            process_started = time.perf_counter()
            engine.start()
            engine.hello()
            start_ms += (time.perf_counter() - process_started) * 1000.0
            reset_started = time.perf_counter()
            info = engine.reset(
                stage=args.stage, difficulty=difficulty, team=args.team, seed=args.seed
            )
            reset_ms += (time.perf_counter() - reset_started) * 1000.0
            if info.get("ready") != "1":
                raise SystemExit(f"RESET did not reach a playable state: {info}")
            for cand, bomb in history:
                engine.step(cand_to_mask(int(cand), int(bomb)), config.decision_frames)
            restore_now = (time.perf_counter() - started) * 1000.0
            restore_ms += restore_now
            if index == 0:
                capabilities = engine.probe_capabilities()
                capabilities["child_hazards"] = engine.probe_child_hazards()
                baseline_state = engine.state()
                state_start = _state_snapshot(baseline_state)
                deaths_baseline = _float_or_none(baseline_state.get("deaths")) or 0.0
                bombs_initial = _float_or_none(baseline_state.get("bombs")) or 0.0
                if args.probe_only:
                    print(
                        json.dumps(
                            {"status": "probe", "engine": gate, "capabilities": capabilities},
                            ensure_ascii=False,
                            indent=2,
                        ),
                        flush=True,
                    )
                    return EXIT_OK
            decision = decide(
                engine,
                config,
                committed=commits,
                frame=frames_committed,
                stage=args.stage,
                difficulty=difficulty,
                consistency=consistency,
                record_source="main",
            )
            chosen = decision.chosen
            if chosen is None:
                events.append(f"no_plan_at_{index}")
                stop_reason = "no_plan"
                break
            bomb = int(decision.label.bomb)
            actions = list(chosen.slots[: config.replan_decisions])
            commits = actions + commits[len(actions):]
            commands: list[str] = []
            for slot_index, cand in enumerate(actions):
                use = bomb if slot_index == 0 else 0
                commands.append(
                    f"STEP {cand_to_mask(int(cand), use)} {config.decision_frames}"
                )
            commit_started = time.perf_counter()
            steps = engine.ask_many(commands)
            advanced = sum(int(parse_fields(reply).get("advanced", 0)) for reply in steps)
            exited = any(parse_fields(reply).get("exit") == "1" for reply in steps)
            committed_state = engine.state()
            commit_ms += (time.perf_counter() - commit_started) * 1000.0
            state_now = _state_snapshot(committed_state)
            for slot_index, cand in enumerate(actions):
                history.append((int(cand), bomb if slot_index == 0 else 0))
            frames_committed += advanced
            if bomb:
                bomb_decisions += 1
            spent = (time.perf_counter() - started) * 1000.0
            wall_ms += spent
            record_ms += decision.timing_ms["record"]
            search_ms += decision.timing_ms["search"]
            verify_ms += decision.timing_ms["verify"]
            events_ms += decision.timing_ms["events"]
            payload = _decision_payload(index, frames_committed, decision, state_now)
            payload["timing_ms"]["commit"] = round(
                (time.perf_counter() - commit_started) * 1000.0, 3
            )
            payload["timing_ms"]["restore"] = round(restore_now, 3)
            payload["timing_ms"]["total"] = round(spent, 3)
            payload["step_replies"] = [reply[:120] for reply in steps]
            decisions.append(payload)
            state_last = state_now
            if not args.quiet:
                print(
                    json.dumps(
                        {
                            "decision": index,
                            "frame": frames_committed,
                            "chosen": payload["chosen_plan"]["first_action"]
                            if payload["chosen_plan"]
                            else None,
                            "bomb": payload["bomb"],
                            "event_flags": payload["frame_event_flags"],
                            "timing_ms": payload["timing_ms"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            engine.close()
            engine = None
            if exited:
                stop_reason = "engine_exit_game_over"
                events.append(f"engine_exit_at_{index}")
                break
            deaths_now = _float_or_none(committed_state.get("deaths")) or 0.0
            if deaths_now > deaths_baseline:
                events.append(f"death_committed_at_{index}")
                if args.stop_on_death:
                    stop_reason = "death"
                    break
            stage_now = state_now["stage"]
            if stage_now is not None and int(stage_now) != int(args.stage):
                stage_clear = True
                stop_reason = "stage_changed"
                break
            if args.max_frames and frames_committed >= args.max_frames:
                stop_reason = "max_frames"
                break
        return _finish(
            args=args,
            config=config,
            difficulty=difficulty,
            gate=gate,
            record_source="legacy_restore",
            decisions=decisions,
            consistency=consistency,
            capabilities=capabilities,
            events=events,
            phases={
                "record": record_ms,
                "search": search_ms,
                "verify": verify_ms,
                "commit": commit_ms,
                "events": events_ms,
                "restore": restore_ms,
            },
            wall_ms=wall_ms,
            frames_committed=frames_committed,
            state_start=state_start,
            state_last=state_last,
            deaths_baseline=deaths_baseline,
            bombs_initial=bombs_initial,
            bomb_decisions=bomb_decisions,
            stop_reason=stop_reason,
            stage_clear=stage_clear,
            start_ms=start_ms,
            reset_ms=reset_ms,
            processes=len(decisions),
            engine=None,
            shoot_releases=0,
            shoot_mask_summary={"frames": 0, "shoot_frames": 0, "bomb_frames": 0},
            empty_tape_decisions=0,
            empty_tape_streak_max=0,
            stage_frames_seen=[],
        )
    except ChildHazardsUnavailable as error:
        print(json.dumps({"status": "fail_closed", "reason": str(error)}, ensure_ascii=False))
        return EXIT_FAIL_CLOSED
    except HazardExportUnavailable as error:
        print(json.dumps({"status": "fail_closed", "reason": str(error)}, ensure_ascii=False))
        return EXIT_FAIL_CLOSED
    finally:
        if engine is not None:
            engine.close()


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _finish(
    *,
    args,
    config,
    difficulty,
    gate,
    record_source,
    decisions,
    consistency,
    capabilities,
    events,
    phases,
    wall_ms,
    frames_committed,
    state_start,
    state_last,
    deaths_baseline,
    bombs_initial,
    bomb_decisions,
    stop_reason,
    stage_clear,
    start_ms,
    reset_ms,
    processes,
    engine,
    shoot_releases=0,
    shoot_mask_summary=None,
    empty_tape_decisions=0,
    empty_tape_streak_max=0,
    stage_frames_seen=None,
    stage_transitions=None,
    stage_stats=None,
    enemies_available=None,
    enemies_observed=0,
    enemy_count_max=0,
    enemy_life_sum_max=0,
    boss_decisions=0,
    boss_first=None,
    boss_life_max=0,
    observation=None,
) -> int:
    steps = len(decisions)
    frames_per_decision = config.replan_decisions * config.decision_frames
    rate = (steps / (wall_ms / 1000.0)) if wall_ms else 0.0
    game_seconds = frames_committed / REFERENCE_FPS
    realtime = (game_seconds / (wall_ms / 1000.0)) if wall_ms else 0.0
    mean = {key: (value / steps if steps else None) for key, value in phases.items()}
    deaths_final = _float_or_none((state_last or {}).get("deaths"))
    bombs_final = _float_or_none((state_last or {}).get("bombs"))
    frame_start = int((state_start or {}).get("frame") or 0)
    frame_last = int((state_last or {}).get("frame") or 0)
    # ``frames_committed`` is the authoritative length of the played segment.  In
    # the legacy path the last ``STATE`` is read *after* the recording moved the
    # shadow H frames, so its f0 delta includes the recording and must not be
    # reported as survival; the child path's f0 delta is clean and equals it.
    survival_frames = int(frames_committed)
    state_frame_delta = max(0, frame_last - frame_start) if record_source == "child" else None
    observed_flags = [
        payload["frame_event_flags"]
        for payload in decisions
        if payload.get("frame_event_flags") is not None
    ]
    # A run that ended on purpose (decision cap, frame cap, a death, a stage
    # change) is complete; only an engine-side stop is "stopped_early".
    intended = {"decisions", "max_frames", "death", "stage_changed"}
    report = {
        "status": "ok" if stop_reason in intended else "stopped_early",
        "stop_reason": stop_reason,
        "record_source": record_source,
        "mode": config.mode,
        "stage": args.stage,
        "difficulty": difficulty,
        "team": args.team,
        "seed": args.seed,
        "engine": gate,
        "decisions": steps,
        "requested_decisions": args.decisions,
        "decisions_per_decision_frames": config.replan_decisions,
        "committed_frames_per_decision": frames_per_decision,
        "survival_frames": survival_frames,
        "survival_seconds": round(survival_frames / REFERENCE_FPS, 3),
        "state_frame_delta": state_frame_delta,
        "frames_committed": frames_committed,
        "stage_clear": bool(stage_clear),
        "stage_at_stop": (state_last or {}).get("stage"),
        "stage_frames_at_stop": (
            (state_last or {}).get("stage_frames") if record_source == "child" else None
        ),
        "deaths_observed": None
        if deaths_final is None
        else int(max(0.0, deaths_final - deaths_baseline)),
        # ---- EXP-P1-009: SHOOT and the stage-advance rising edge -------------
        # The candidate space has no shoot dimension; every committed frame sets
        # BTN_SHOOT except the release frame of a pulse decision.  A run whose
        # `frames_without_shoot` is 0 can never leave the end-of-stage state
        # (GameManager.cpp:381-389), whatever else it does.
        "shoot_always": bool(config.shoot_always),
        "shoot_pulse_period": int(config.shoot_pulse_period),
        "shoot_releases": int(shoot_releases),
        "frames_with_shoot": int((shoot_mask_summary or {}).get("shoot_frames", 0)),
        "frames_without_shoot": int(
            (shoot_mask_summary or {}).get("frames", 0)
            - (shoot_mask_summary or {}).get("shoot_frames", 0)
        ),
        "shoot_fraction": (
            round(
                (shoot_mask_summary or {}).get("shoot_frames", 0)
                / (shoot_mask_summary or {})["frames"],
                6,
            )
            if (shoot_mask_summary or {}).get("frames")
            else None
        ),
        # "alive but nothing happening": decisions whose whole 48-frame tape had
        # no lethal box at all.  A long streak is the signature of the stuck
        # end-of-stage state (EXP-P1-009 §3), not of a quiet stage.
        "empty_tape_decisions": int(empty_tape_decisions),
        "empty_tape_streak_max": int(empty_tape_streak_max),
        "stage_frames_seen": None
        if not stage_frames_seen
        else {
            "first": int(stage_frames_seen[0]),
            "last": int(stage_frames_seen[-1]),
            "max": int(max(stage_frames_seen)),
            "resets": int(
                sum(
                    1
                    for a, b in zip(stage_frames_seen, stage_frames_seen[1:])
                    if b < a
                )
            ),
        },
        # ---- EXP-P1-011: multi-stage progress + ENEMIES observation ---------
        # `stage_clear` is a *game* verdict (the engine's own stage index moved).
        # Survival is a different claim and is reported separately as
        # `survival_frames`/`deaths_observed`; a run can be alive and never clear.
        "follow_stage": bool(getattr(args, "follow_stage", False)),
        "stage_transitions": list(stage_transitions or []),
        "stage_stats": list(stage_stats or []),
        "stages_visited": [int(args.stage)]
        + [int(item["to_stage"]) for item in (stage_transitions or [])],
        "stages_completed": len(stage_transitions or []),
        "enemies_available": enemies_available,
        "enemies_observed_decisions": int(enemies_observed),
        "enemy_count_max": int(enemy_count_max),
        "enemy_life_sum_max": int(enemy_life_sum_max),
        "boss_decisions": int(boss_decisions),
        "boss_first": boss_first,
        "boss_life_max": int(boss_life_max),
        "deaths_final": deaths_final,
        "bombs_initial": bombs_initial,
        "bombs_final": bombs_final,
        "bombs_spent": None
        if bombs_final is None
        else round(float(max(0.0, bombs_initial - bombs_final)), 3),
        "bomb_decisions": int(bomb_decisions),
        "bomb_death_available": next(
            (payload["bomb_death_available"] for payload in decisions), None
        ),
        "frame_event_flags_sampled": len(observed_flags),
        "frame_event_flags_player_hit": sum(
            1 for flags in observed_flags if int(flags) & 0x2
        ),
        "probe": capabilities,
        "timing_ms": {key: (round(value, 3) if value is not None else None) for key, value in mean.items()},
        "timing_ms_sum": {key: round(value, 3) for key, value in phases.items()},
        "process_start_ms": round(start_ms, 3),
        "reset_ms": round(reset_ms, 3),
        "decisions_per_second": round(rate, 4),
        "game_frames_per_second": round(frames_committed / (wall_ms / 1000.0), 4) if wall_ms else 0.0,
        "game_seconds_per_wall_second": round(realtime, 4),
        "realtime_multiple": round(realtime, 4),
        "realtime_formula": "frames_committed / wall_seconds / 60",
        "at_least_realtime": bool(realtime >= 1.0),
        "consistency": consistency.as_dict(),
        "events": events,
        "config": {
            "horizon": config.horizon,
            "beam": config.beam,
            "verify_k": config.verify_k,
            "max_iterations": config.max_iterations,
            "durations": list(config.durations),
            "shoot_always": bool(config.shoot_always),
            "shoot_pulse_period": int(config.shoot_pulse_period),
        },
        "engine_stats": {
            "processes_launched": int(processes),
            "calls": int(getattr(getattr(engine, "stats", None), "calls", 0)) if engine else None,
        },
        # ``None`` unless --save-observations ran: the P3/D0 shard this run wrote,
        # its per-decision byte cost and the frozen-spec digests that were checked.
        "observation": observation,
    }
    out_path = args.out
    if out_path is None:
        out_path = Path(
            f".tmp_v400_teacher_{record_source}_{config.mode}_s{args.stage}"
            f"_d{difficulty}_seed{args.seed}.jsonl"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "report", **report}, ensure_ascii=False) + "\n")
        for payload in decisions:
            handle.write(json.dumps({"kind": "decision", **payload}, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return EXIT_OK if report["status"] == "ok" else EXIT_STOPPED_EARLY


def _decision_payload(index: int, frame: int, decision, state_now: dict) -> dict:
    return {
        "decision": index,
        "frame": frame,
        "state": state_now,
        "geo_score": [round(float(value), 4) for value in decision.label.scores],
        "verified_mask": [bool(value) for value in decision.label.verified_mask],
        "verified": [item.as_dict() for item in decision.verified],
        "chosen_plan": decision.label.chosen_plan.as_dict()
        if decision.label.chosen_plan is not None
        else None,
        "margin": decision.label.margin,
        "bomb": decision.label.bomb,
        "bomb_death_available": decision.label.bomb_death_available,
        "frame_event_flags": decision.label.frame_event_flags,
        "bombs": decision.bombs,
        "timing_ms": {
            key: round(float(value), 3) for key, value in decision.timing_ms.items()
        },
        "events": list(decision.label.events),
        "consistency": decision.consistency.as_dict(),
        "tape_boxes": _box_summary(decision.tape),
        "child_record": None
        if decision.child_record is None
        else decision.child_record.as_dict(),
        "events_frame": dict(decision.events_frame),
        # EXP-P1-015 §3: observation only — the inputs of bomb rule 1, so the
        # bomb timing can be audited from the JSONL without a replay.
        "bomb_rule1": {
            "max_death_frame": decision.label.extra.get("bomb_rule1_max_death_frame"),
            "shortest_verified_death_frame": decision.label.extra.get(
                "shortest_verified_death_frame"
            ),
            "imminent": decision.label.extra.get("bomb_rule1_imminent"),
            "suppressed": decision.label.extra.get("bomb_rule1_suppressed"),
            "death_bomb_window": decision.label.extra.get("death_bomb_window"),
            "chosen_bottom": None
            if decision.label.chosen_plan is None
            else round(float(decision.label.chosen_plan.bottom), 4),
            "chosen_escape": None
            if decision.label.chosen_plan is None
            else round(float(decision.label.chosen_plan.escape), 4),
            # EXP-P1-020 §2: observation only — the two new v2f cost terms of the
            # plan that was actually committed, so "was the top band charged" is
            # answerable from the JSONL without a replay.
            "chosen_top": None
            if decision.label.chosen_plan is None
            else round(float(decision.label.chosen_plan.top), 4),
            "chosen_corner": None
            if decision.label.chosen_plan is None
            else round(float(decision.label.chosen_plan.corner), 4),
            # EXP-P1-021 §2: observation only — the v2h laser cost of the plan that
            # was committed, so "did the damageable gate actually charge a beam"
            # is answerable from the JSONL without replaying the frame.
            "chosen_laser": None
            if decision.label.chosen_plan is None
            else round(float(decision.label.chosen_plan.laser), 4),
        },
        # EXP-P1-015 §1: observation only — how much room the ship had above and
        # below the decision point, and which way the chosen plan ends.  Computed
        # from the *already recorded* tape, so it cannot move the policy.
        "bottom_diag": _bottom_diag(decision),
    }


#: Playfield bottom edge (11-P1 §2.2), repeated here because this module only
#: observes; the teacher itself reads it from :mod:`touhou_rl.v400.teacher.config`.
_BOTTOM_Y = 432.0


def _bottom_diag(decision) -> dict | None:
    """Bottom-band observables of one decision (EXP-P1-015 §1).

    ``room_up`` is the vertical gap from the ship's hurtbox top to the nearest
    lethal AABB that sits above it **and overlaps its column** — i.e. what the
    ship would hit if it flew straight up.  ``room_down`` is the literal distance
    to the playfield bottom.  ``closing`` is the same ``room_up`` measured two
    frames later, so ``closing / 2`` is the measured closing speed of that
    hazard and ``frames_to_close`` is how long the upward exit stays open.
    """

    import numpy as np

    tape = getattr(decision, "tape", None)
    if tape is None or len(tape) < 3:
        return None
    try:
        half = np.asarray(tape.hurtbox_half, dtype=np.float64)
        x, y = (float(value) for value in tape.start_position)
        mins, maxs = tape.boxes(0)
        if mins.shape[0]:
            boxes = np.hstack([mins, maxs])
        else:
            boxes = np.zeros((0, 4), dtype=np.float64)
        mins2, maxs2 = tape.boxes(2)
        boxes2 = np.hstack([mins2, maxs2]) if mins2.shape[0] else np.zeros((0, 4), dtype=np.float64)
    except Exception:  # pragma: no cover - observation must never break a run
        return None

    def room_up(frame_boxes):
        if frame_boxes.shape[0] == 0:
            return None
        cols = (frame_boxes[:, 2] >= x - half[0]) & (frame_boxes[:, 0] <= x + half[0])
        above = frame_boxes[cols & (frame_boxes[:, 3] < y - half[1])]
        if above.shape[0] == 0:
            return None
        return float(np.min(y - half[1] - above[:, 3]))

    room_up_now = room_up(boxes)
    room_up_later = room_up(boxes2)
    closing = None
    frames_to_close = None
    if room_up_now is not None and room_up_later is not None:
        closing = (room_up_now - room_up_later) / 2.0
        if closing > 1e-6:
            frames_to_close = room_up_now / closing
    chosen = getattr(decision, "chosen", None)
    plan = None if chosen is None else chosen
    first = None if plan is None else int(plan.first_action)
    move = None if first is None else first % 9
    return {
        "x": round(x, 3),
        "y": round(y, 3),
        "room_down": round(_BOTTOM_Y - y, 3),
        "room_up": None if room_up_now is None else round(room_up_now, 3),
        "room_up_2f": None if room_up_later is None else round(room_up_later, 3),
        "closing_per_frame": None if closing is None else round(closing, 3),
        "frames_to_close": None if frames_to_close is None else round(frames_to_close, 1),
        "boxes": int(boxes.shape[0]),
        "chosen_first_action": first,
        "chosen_move": move,
        "chosen_dir": None
        if move is None
        else ("up" if move in (1, 2, 8) else "down" if move in (4, 5, 6) else "horiz" if move in (3, 7) else "idle"),
        "chosen_end_y": None if plan is None else round(float(plan.end_xy[1]), 3),
        "chosen_end_x": None if plan is None else round(float(plan.end_xy[0]), 3),
        "chosen_margin": None if plan is None else round(float(plan.margin), 3),
        "chosen_death_frame": None
        if plan is None or plan.death_frame is None
        else int(plan.death_frame),
    }


def _hash_checkpoint(engine) -> dict:
    """``HASH`` as ``{hash, frames}``: the alignment primitive of EXP-P1-014.

    ``HASH`` is a read-only query over the 25 tracked fields (``rlharness.cpp``),
    so asking for it on both sides of a decision cannot move the shadow.
    """

    fields = parse_fields(engine.ask("HASH"))
    return {"hash": str(fields["hash"]), "frames": int(fields.get("frames", -1))}


def _state_snapshot(state: dict) -> dict:
    raw = state.get("raw") or {}
    return {
        "frame": int(state.get("frame") or 0),
        "deaths": _float_or_none(state.get("deaths")),
        "lives": _float_or_none(state.get("lives")),
        "bombs": _float_or_none(state.get("bombs")),
        "px": _float_or_none(state.get("px")),
        "py": _float_or_none(state.get("py")),
        "stage": int(raw["f1"]) if "f1" in raw else None,
        "stage_frames": int(raw["f3"]) if "f3" in raw else None,
        "player_state": int(raw["f13"]) if "f13" in raw else None,
    }


def _advance_until_playable(engine, *, limit_frames: int = 600, poll: int = 4) -> tuple[int, bool]:
    """Step the new stage until ``PARK`` is safe again (``--follow-stage`` only).

    ``AddedCallback`` sets ``stageTransitionState = 1`` when it inherits a
    ``>= 2`` state, and ``GameplaySetupThread`` blocks in
    ``while (stageTransitionState != 0) Sleep(17)`` (``GameManagerSetup.cpp:426``)
    — the only thing that clears the state is the ``0x1001`` rising edge of
    ``GameManager.cpp:381-389``.  So right after a stage change the harness has
    **2 threads and ``setupState=1``**, and ``PARK`` fails closed with
    ``ERR park-unsafe`` (measured: ``role=shadow-required threads=2 setup=1``).

    This helper therefore alternates SHOOT held for 7 frames / released for 1
    (a guaranteed rising edge every 8 frames, no direction bits so the ship does
    not drift during the load) until ``DEBUG`` says the setup finished, then
    returns ``(frames_advanced, True)``.  It never invents a state: the verdict
    is the engine's own ``setupState``/``threads``.
    """

    advanced_total = 0
    while advanced_total < limit_frames:
        fields = engine.debug()
        try:
            setup = int(fields.get("setupState", 0))
            threads = int(fields.get("threads", 1))
        except (TypeError, ValueError):  # pragma: no cover - protocol drift
            return advanced_total, False
        if setup == 0 and threads <= 1:
            return advanced_total, True
        for index in range(poll):
            mask = 0 if index == poll - 1 else 1  # 1 = BTN_SHOOT
            reply = parse_fields(engine.ask(f"STEP {mask} 1"))
            advanced = int(reply.get("advanced", 0))
            advanced_total += advanced
            if advanced <= 0 or reply.get("exit") == "1":
                return advanced_total, False
    return advanced_total, False


def _stage_transition(
    *,
    decision: int,
    frame: int,
    from_stage: int,
    to_stage: int,
    entry_frame: int,
    entry_deaths: float,
    entry_bombs,
    deaths_now: float,
    bombs_now,
) -> dict:
    """One completed stage, as a row a G1 "deaths per stage" table can use.

    EXP-P1-009 §4 asks for the *frame* the stage finished; EXP-P1-011 §1 asks for
    deaths and bombs **inside that stage**, which is why the counters are deltas
    against the entry snapshot instead of run totals.
    """

    return {
        "decision": int(decision),
        "frame": int(frame),
        "from_stage": int(from_stage),
        "to_stage": None if to_stage is None else int(to_stage),
        "frames_in_stage": int(frame) - int(entry_frame),
        "deaths_in_stage": None
        if deaths_now is None
        else int(max(0.0, float(deaths_now) - float(entry_deaths))),
        "bombs_spent_in_stage": None
        if bombs_now is None or entry_bombs is None
        else round(float(max(0.0, float(entry_bombs) - float(bombs_now))), 3),
    }


def _debug_snapshot(engine) -> dict:
    """``DEBUG`` as loggable data: the supervisor/setup verdict per decision.

    EXP-P1-009 §6.3 asked for ``curState``/``setupState``/``stageTransitionState``
    per frame.  ``HandleDebug`` (``rlharness.cpp:696-723``) exports
    ``curState``/``setupState``/``subthread``/``globals``/``stage``/``threads``
    but **not** ``stageTransitionState`` — so that one stays unobservable and is
    reported absent rather than guessed.
    """

    try:
        fields = engine.debug()
    except EngineError as error:
        return {"available": False, "reason": str(error)[:160]}
    out = {"available": True, "reason": "DEBUG accepted"}
    for key in (
        "curState",
        "setupState",
        "subthread",
        "globals",
        "stage",
        "difficulty",
        "shot",
        "isReplay",
        "isDemo",
        "threads",
        "frames",
    ):
        if key in fields:
            out[key] = int(fields[key])
    out["stage_transition_state"] = None  # not exported by DEBUG (EXP-P1-009 §6.3)
    return out


def _float_or_none(value):
    return None if value is None else float(value)


def _enemy_summary(snapshot) -> dict:
    """One decision's ``ENEMIES`` page as loggable data (no policy input).

    EXP-P1-009 §6.2 said the Boss was "unobservable in the CLI": ``STATE`` has no
    enemy field, but ``ENEMIES`` has ``count``, ``flags1 & BOSS`` and ``life``.
    This only records them — nothing in :mod:`.teacher.decide` reads this dict,
    so the plan is bit-identical with or without the observation.
    """

    snapshot = snapshot or {}
    rows = []
    for row in snapshot.get("rows") or ():
        rows.append(
            {
                "slot": int(row.slot),
                "flags1": f"0x{int(row.flags1):08x}",
                "boss": bool(int(row.flags1) & ENEMY_FLAG_BOSS),
                "life": int(row.life),
                "timer": int(row.timer),
                "x": round(float(row.position[0]), 3),
                "y": round(float(row.position[1]), 3),
            }
        )
    return {
        "available": bool(snapshot.get("available")),
        "reason": snapshot.get("reason"),
        "frame": snapshot.get("frame"),
        "count": snapshot.get("count"),
        "pages": snapshot.get("pages", 1),
        "truncated": snapshot.get("truncated"),
        "boss_count": int(snapshot.get("boss_count") or 0),
        "boss_life": snapshot.get("boss_life"),
        "boss_slots": list(snapshot.get("boss_slots") or ()),
        "life_sum": snapshot.get("life_sum"),
        "rows": rows,
    }


def _box_summary(tape) -> dict:
    """Per-frame lethal AABB counts of the tape the search actually ran on."""
    counts = [int(tape.box_array(index).shape[0]) for index in range(len(tape))]
    if not counts:
        return {"min": 0, "max": 0, "mean": 0.0, "frames": 0}
    return {
        "min": min(counts),
        "max": max(counts),
        "mean": round(sum(counts) / len(counts), 3),
        "frames": len(counts),
        "empty_frames": sum(1 for value in counts if value == 0),
    }


if __name__ == "__main__":
    sys.exit(main())
