"""G1 gate evaluation: the ``S1N`` 20-episode suite and the ``PSN`` 8-stage suite.

This tool only *drives* :mod:`tools.rl.v400_teacher` as a subprocess, one engine at
a time (the WSL engine slot is single-occupancy), and turns each run's report into
one JSONL record.  It never re-implements, tunes or overrides the teacher policy:
the policy flags are exactly the ones the G1 task states, and the frame caps are
``ceil(1.5 * human gameplay frames)`` from ``Document/Update/Alpha/v4.0.0/bench/
human_reference.json`` (normal difficulty).

    python tools/rl/v400_g1_eval.py s1n                 # seeds 1000..1019, stage 0
    python tools/rl/v400_g1_eval.py psn                 # stages 0..7, seed 1
    python tools/rl/v400_g1_eval.py summary             # tables from the JSONL

Every run appends one line to ``.tmp_v400_g1/<suite>_runs.jsonl`` *immediately*
after it finishes, so a batch can be resumed (finished cases are skipped) and a
partial batch still has every number it earned.  The per-run teacher JSONL and
the subprocess log are kept next to it; nothing is deleted.

Verdict vocabulary is kept strict on purpose:

``stage_clear``
    the engine's own ``STATE.f1`` moved away from the started stage; a run that is
    merely alive at the frame cap is ``max_frames`` and is **not** a clear.
``deaths``
    ``deaths_final - deaths_baseline`` from ``STATE``; a run stopped by
    ``--stop-on-death`` has at least one.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The G1 task's human reference, at the path the task states (relative to the
#: v4.0.0 package root).
HUMAN_REFERENCE = REPO_ROOT / "Document" / "Update" / "Alpha" / "v4.0.0" / "bench" / "human_reference.json"

#: Where every artefact of this experiment goes.
DEFAULT_OUT_DIR = REPO_ROOT / ".tmp_v400_g1"

S1N_SEEDS = tuple(range(1000, 1020))
S1N_STAGE = 0
S1N_DIFFICULTY = "normal"
CAP_MULTIPLIER = 1.5

#: PSN stages are ``STATE.f1`` values 0..7 (0-based): Stage1..Stage6B.
PSN_STAGES = tuple(range(0, 8))
PSN_SEED = 1

#: Stage names by the game's own ``stage_index`` (``human_reference.json`` uses
#: the same numbering: 0=Stage1 .. 3=Stage4A/4B, 4=Stage4B, 5=Stage5,
#: 6=Stage6A, 7=Stage6B).
STAGE_NAMES = {
    0: "Stage1",
    1: "Stage2",
    2: "Stage3",
    3: "Stage4A",
    4: "Stage4B",
    5: "Stage5",
    6: "Stage6A",
    7: "Stage6B",
}

#: The task sanctions a fixed cap when no human reference exists for a stage;
#: the G1 ruling (2026-11-09) fixed that number at 45,000 and asked for the
#: ``cap_source`` to say ``fallback``.  It is used *only* for those stages.
MISSING_STAGE_CAP = 45000

#: POSIX path on purpose: the teacher CLI is always launched from the repo root
#: inside WSL, and a backslash path would be a literal filename there.
TEACHER = "tools/rl/v400_teacher.py"

#: Same-class proxy for a stage the normal human corpus does not cover.
PAIR_PROXY = {4: 3, 6: 7}


# ---------------------------------------------------------------------------
# human reference + caps
# ---------------------------------------------------------------------------
def load_human_reference(path: Path = HUMAN_REFERENCE) -> dict:
    """``{stage_index: {"stage_name", "gameplay_frames"}}`` for normal difficulty."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    out: dict[int, dict] = {}
    for item in payload.get("sets") or ():
        if str(item.get("difficulty", "")).lower() != "normal":
            continue
        for stage in item.get("stages") or ():
            out[int(stage["stage_index"])] = {
                "stage_name": str(stage.get("stage_name") or STAGE_NAMES.get(int(stage["stage_index"]))),
                "gameplay_frames": int(stage["gameplay_frames"]),
                "deaths": int(stage.get("deaths") or 0),
                "source_replay": item.get("source_replay_file"),
                "difficulty_index": int(item.get("difficulty_index") or 1),
            }
    return out


def cap_from_human(frames: int, multiplier: float = CAP_MULTIPLIER) -> int:
    """``ceil(multiplier * frames)`` 鈥?the cap rule the reference itself states."""

    return int(math.ceil(float(multiplier) * int(frames)))


def s1n_cap(human: dict) -> tuple[int, str]:
    """S1N cap: ``ceil(1.5 * Stage1 human frames)``; refuses to invent a number."""

    entry = human.get(S1N_STAGE)
    if not entry:
        raise SystemExit("human reference has no normal Stage1 entry; cannot derive the S1N cap")
    return cap_from_human(entry["gameplay_frames"]), (
        f"ceil(1.5*{entry['stage_name']}_normal_human_frames={entry['gameplay_frames']})"
    )


def psn_caps(human: dict) -> list[dict]:
    """One row per PSN stage: measured human cap, or the task's fixed fallback.

    ``cap_source`` is never blank: a stage with human data names the replay and
    the human frame count, a stage without it records ``missing_human_reference``
    plus what the same-class proxy *would* have been, so the reader can see that
    no threshold was bought by shrinking a cap.
    """

    rows = []
    for stage in PSN_STAGES:
        entry = human.get(stage)
        if entry:
            rows.append(
                {
                    "stage": stage,
                    "stage_name": STAGE_NAMES.get(stage, f"stage{stage}"),
                    "cap": cap_from_human(entry["gameplay_frames"]),
                    "cap_source": (
                        f"ceil(1.5*{entry['stage_name']}_normal_human_frames="
                        f"{entry['gameplay_frames']}) from {entry['source_replay']}"
                    ),
                    "human_frames": entry["gameplay_frames"],
                    "human_deaths": entry["deaths"],
                    "missing_human_reference": False,
                    "proxy_note": None,
                }
            )
        else:
            # Same-class proxy, reported but *not* used (the task sanctions 40,000).
            # Stage4B's pair is Stage4A (index 3); Stage6A's pair is Stage6B (7).
            proxy = human.get(PAIR_PROXY.get(stage, stage - 1)) or human.get(stage - 1) or human.get(stage + 1)
            proxy_cap = cap_from_human(proxy["gameplay_frames"]) if proxy else None
            rows.append(
                {
                    "stage": stage,
                    "stage_name": STAGE_NAMES.get(stage, f"stage{stage}"),
                    "cap": MISSING_STAGE_CAP,
                    "cap_source": (
                        "fallback: missing_human_reference -> task-sanctioned fixed "
                        f"--max-frames {MISSING_STAGE_CAP} (not a measured cap)"
                    ),
                    "human_frames": None,
                    "human_deaths": None,
                    "missing_human_reference": True,
                    "proxy_note": (
                        f"same-class proxy {proxy['stage_name']} normal {proxy['gameplay_frames']} "
                        f"frames would give cap {proxy_cap}"
                        if proxy
                        else "no same-class proxy available"
                    ),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# command construction
# ---------------------------------------------------------------------------
def teacher_command(
    *,
    stage: int,
    seed: int,
    cap: int,
    out_path: Path,
    python: str,
    difficulty: str = S1N_DIFFICULTY,
    mode: str = "full",
    data_dir: str | None = None,
) -> list[str]:
    """The exact CLI the G1 task states, with ``--decisions`` sized past the cap."""

    decisions = int(cap) // 4 + 8
    command = [
        python,
        TEACHER,
        "--decisions",
        str(decisions),
        "--max-frames",
        str(int(cap)),
        "--stop-on-death",
        "--stage",
        str(int(stage)),
        "--difficulty",
        str(difficulty),
        "--seed",
        str(int(seed)),
        "--mode",
        str(mode),
        "--quiet",
        "--out",
        str(out_path),
    ]
    if data_dir:
        # Per-worker game data dir: parallel workers must never share score.dat /
        # th08.cfg writes (the engine rewrites both on every run).
        command.extend(["--data-dir", str(data_dir)])
    return command


# ---------------------------------------------------------------------------
# report parsing
# ---------------------------------------------------------------------------
def read_report(jsonl_path: Path) -> dict:
    """The ``kind=report`` line of a teacher JSONL (first line by construction)."""

    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("kind") == "report":
                return payload
    raise ValueError(f"no kind=report line in {jsonl_path}")


def boss_analysis(jsonl_path: Path, max_lines: int | None = None) -> dict:
    """Stream the decision rows and report Boss visibility honestly.

    Only fields the engine exports are used (``ENEMIES`` rows: ``boss`` flag and
    ``life``).  A "kill" is *not* inferred from survival: the report carries the
    observed life curve, the zero-life row count, and whether the arena went
    empty after the Boss was present.
    """

    present = 0
    first_decision = None
    last_decision = None
    life_first = None
    life_last = None
    life_min = None
    life_max = None
    zero_life_rows = 0
    last_index = -1
    empty_after_last_boss = 0
    seen_boss = False
    unavailable = 0
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if max_lines is not None and line_no >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("kind") != "decision":
                continue
            index = int(payload.get("decision") or 0)
            last_index = max(last_index, index)
            enemies = payload.get("enemies") or {}
            if not enemies.get("available"):
                unavailable += 1
                continue
            rows = [row for row in (enemies.get("rows") or ()) if row.get("boss")]
            if rows:
                present += 1
                seen_boss = True
                if first_decision is None:
                    first_decision = index
                life_last = int(rows[-1].get("life") or 0)
                if life_first is None:
                    life_first = life_last
                life_min = life_last if life_min is None else min(life_min, life_last)
                life_max = life_last if life_max is None else max(life_max, life_last)
                zero_life_rows += sum(1 for row in rows if int(row.get("life") or 0) <= 0)
                last_decision = index
                empty_after_last_boss = 0
            elif seen_boss and int(enemies.get("count") or 0) == 0:
                empty_after_last_boss += 1
    return {
        "boss_present_decisions": present,
        "boss_first_decision": first_decision,
        "boss_last_decision": last_decision,
        "boss_life_first": life_first,
        "boss_life_last": life_last,
        "boss_life_min": life_min,
        "boss_life_max": life_max,
        "boss_zero_life_rows": zero_life_rows,
        "decisions_observed": last_index + 1,
        "empty_arena_decisions_after_last_boss": empty_after_last_boss,
        "enemies_unavailable_decisions": unavailable,
        "boss_kill_evidence": _kill_evidence(
            zero_life_rows, empty_after_last_boss, present, last_index, last_decision
        ),
    }


def _kill_evidence(zero_life_rows, empty_after, present, last_index, last_boss_decision) -> str:
    """A label, not a guess: what the record actually shows about the Boss."""

    if not present:
        return "not_observed"
    if zero_life_rows:
        return "life_reached_zero"
    if empty_after >= 3 and last_boss_decision is not None and last_index > last_boss_decision:
        return "arena_empty_after_boss"
    return "present_at_stop"


def parse_run(jsonl_path: Path, *, with_boss: bool = True) -> dict:
    """The fields a G1 table needs, from one teacher JSONL."""

    report = read_report(jsonl_path)
    out = {
        "status": report.get("status"),
        "stop_reason": report.get("stop_reason"),
        "stage": report.get("stage"),
        "stage_clear": bool(report.get("stage_clear")),
        "frames": int(report.get("frames_committed") or 0),
        "survival_frames": int(report.get("survival_frames") or 0),
        "deaths": report.get("deaths_observed"),
        "decisions": int(report.get("decisions") or 0),
        "bombs_spent": report.get("bombs_spent"),
        "bomb_decisions": report.get("bomb_decisions"),
        "bombs_initial": report.get("bombs_initial"),
        "bombs_final": report.get("bombs_final"),
        "realtime_multiple": report.get("realtime_multiple"),
        "stage_at_stop": report.get("stage_at_stop"),
        "engine_sha256": (report.get("engine") or {}).get("sha256"),
        "engine_sha_match": (report.get("engine") or {}).get("match"),
        "enemies_available": report.get("enemies_available"),
        "boss_decisions": report.get("boss_decisions"),
        "boss_life_max_report": report.get("boss_life_max"),
        "consistency": report.get("consistency"),
    }
    if with_boss:
        out["boss"] = boss_analysis(jsonl_path)
    return out


# ---------------------------------------------------------------------------
# one case
# ---------------------------------------------------------------------------
def run_case(
    *,
    case_id: str,
    suite: str,
    stage: int,
    seed: int,
    cap: int,
    cap_source: str,
    out_dir: Path,
    python: str,
    timeout_s: float,
    with_boss: bool,
    data_dir: str | None = None,
    jobs: int = 1,
    worker: int = 0,
) -> dict:
    run_dir = out_dir / suite
    log_dir = out_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / f"{case_id}.jsonl"
    log_path = log_dir / f"{case_id}.log"
    command = teacher_command(
        stage=stage, seed=seed, cap=cap, out_path=jsonl_path, python=python, data_dir=data_dir
    )
    record = {
        "case_id": case_id,
        "suite": suite,
        "stage": stage,
        "stage_name": STAGE_NAMES.get(stage, f"stage{stage}"),
        "difficulty": S1N_DIFFICULTY,
        "seed": seed,
        "cap": cap,
        "cap_source": cap_source,
        "command": command,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_s": None,
        "exit_code": None,
        "completed": False,
        "error": None,
        "jsonl": str(jsonl_path.relative_to(REPO_ROOT)),
        "log": str(log_path.relative_to(REPO_ROOT)),
        "data_dir": data_dir or "(teacher default)",
        "parallel_jobs": int(jobs),
        "worker": int(worker),
        # The harness strips DISPLAY/WAYLAND_DISPLAY (engine.py:209-214) and
        # HandlePark() refuses when a display is set, so no X server / xvfb is
        # involved at any parallelism.
        "display": "unset (null device; PARK refuses a live DISPLAY)",
    }
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        record["wall_s"] = round(time.perf_counter() - started, 3)
        record["exit_code"] = int(proc.returncode)
        log_text = (proc.stdout or "") + (proc.stderr or "")
        if proc.stderr:
            log_text = (proc.stdout or "") + "\n[stderr]\n" + proc.stderr
        log_path.write_text(log_text, encoding="utf-8")
        if jsonl_path.exists():
            try:
                record.update(parse_run(jsonl_path, with_boss=with_boss))
                record["completed"] = True
            except Exception as error:  # a truncated JSONL must not look like a run
                record["error"] = f"{type(error).__name__}: {error}"
        else:
            record["error"] = "teacher wrote no JSONL"
    except subprocess.TimeoutExpired as error:
        record["wall_s"] = round(time.perf_counter() - started, 3)
        record["exit_code"] = -9
        record["error"] = f"timeout after {timeout_s}s"
        log_text = ""
        if error.stdout:
            log_text += error.stdout if isinstance(error.stdout, str) else error.stdout.decode("utf-8", "replace")
        if error.stderr:
            log_text += "\n[stderr]\n" + (
                error.stderr if isinstance(error.stderr, str) else error.stderr.decode("utf-8", "replace")
            )
        log_path.write_text(log_text, encoding="utf-8")
    except Exception as error:  # pragma: no cover - subprocess/env failure
        record["wall_s"] = round(time.perf_counter() - started, 3)
        record["exit_code"] = -1
        record["error"] = f"{type(error).__name__}: {error}"
        log_path.write_text(record["error"], encoding="utf-8")
    return record


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    """Two-sided 95% Wilson score interval for a binomial proportion."""

    if trials <= 0:
        return None
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize_s1n(records: list[dict], declared: int) -> dict:
    completed = [row for row in records if row.get("completed")]
    clears = [row for row in completed if row.get("stage_clear")]
    deaths = [int(row.get("deaths") or 0) for row in completed]
    frames = [int(row.get("frames") or 0) for row in completed]
    walls = [float(row.get("wall_s") or 0.0) for row in completed]
    ci_completed = wilson_interval(len(clears), len(completed))
    ci_declared = wilson_interval(len(clears), declared)
    return {
        "suite": "s1n",
        "declared": declared,
        "completed": len(completed),
        "failed": len(records) - len(completed),
        "not_run": declared - len(records),
        "clears": len(clears),
        "clear_rate_completed": (len(clears) / len(completed)) if completed else None,
        "wilson95_completed": None if ci_completed is None else [round(ci_completed[0], 6), round(ci_completed[1], 6)],
        "clear_rate_declared": len(clears) / declared if declared else None,
        "wilson95_declared": None if ci_declared is None else [round(ci_declared[0], 6), round(ci_declared[1], 6)],
        "deaths_total": sum(deaths),
        "deaths_mean_per_episode": (sum(deaths) / len(deaths)) if deaths else None,
        "deaths_max": max(deaths) if deaths else None,
        "episodes_with_death": sum(1 for value in deaths if value > 0),
        "frames_mean": (sum(frames) / len(frames)) if frames else None,
        "frames_max": max(frames) if frames else None,
        "wall_total_s": round(sum(walls), 3),
        "wall_mean_s": round(sum(walls) / len(walls), 3) if walls else None,
        "exit_codes": _histogram([row.get("exit_code") for row in completed]),
        "stop_reasons": _histogram([row.get("stop_reason") for row in completed]),
        "partial": len(records) < declared or len(completed) < len(records),
    }


def summarize_psn(records: list[dict], caps: list[dict]) -> dict:
    by_stage = {int(row["stage"]): row for row in records if row.get("completed")}
    rows = []
    for cap in caps:
        stage = int(cap["stage"])
        row = by_stage.get(stage)
        entry = {
            "stage": stage,
            "stage_name": cap["stage_name"],
            "cap": cap["cap"],
            "cap_source": cap["cap_source"],
            "missing_human_reference": cap["missing_human_reference"],
            "proxy_note": cap["proxy_note"],
            "ran": row is not None,
        }
        if row:
            entry.update(
                {
                    "stage_clear": bool(row.get("stage_clear")),
                    "stop_reason": row.get("stop_reason"),
                    "frames": int(row.get("frames") or 0),
                    "deaths": int(row.get("deaths") or 0),
                    "bombs_spent": row.get("bombs_spent"),
                    "bomb_decisions": row.get("bomb_decisions"),
                    "wall_s": row.get("wall_s"),
                    "exit_code": row.get("exit_code"),
                    "boss": row.get("boss"),
                }
            )
        rows.append(entry)
    ran = [row for row in rows if row.get("ran")]
    deaths = [int(row.get("deaths") or 0) for row in ran]
    return {
        "suite": "psn",
        "stages": rows,
        "stages_ran": len(ran),
        "stages_declared": len(caps),
        "deaths_total": sum(deaths),
        "deaths_mean_per_stage": (sum(deaths) / len(deaths)) if deaths else None,
        "stages_with_death": sum(1 for value in deaths if value > 0),
        "stages_at_or_below_half": sum(1 for value in deaths if value <= 0.5),
        "clears": sum(1 for row in ran if row.get("stage_clear")),
        "wall_total_s": round(sum(float(row.get("wall_s") or 0.0) for row in ran), 3),
        "partial": len(ran) < len(caps),
    }


def _histogram(values) -> dict:
    out: dict = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return out


# ---------------------------------------------------------------------------
# markdown tables (the experiment card reuses these verbatim)
# ---------------------------------------------------------------------------
def markdown_s1n(records: list[dict], summary: dict) -> str:
    lines = [
        "| # | seed | stage_clear | frames | deaths | bombs | wall_s | exit | stop_reason |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for index, row in enumerate(sorted(records, key=lambda item: int(item.get("seed") or 0)), start=1):
        lines.append(
            "| {i} | {seed} | {clear} | {frames} | {deaths} | {bombs} | {wall} | {code} | {stop} |".format(
                i=index,
                seed=row.get("seed"),
                clear=("**true**" if row.get("stage_clear") else "false") if row.get("completed") else "n/a",
                frames=row.get("frames") if row.get("completed") else "n/a",
                deaths=row.get("deaths") if row.get("completed") else "n/a",
                bombs=row.get("bombs_spent") if row.get("completed") else "n/a",
                wall=row.get("wall_s"),
                code=row.get("exit_code"),
                stop=row.get("stop_reason") if row.get("completed") else row.get("error"),
            )
        )
    lines.append("")
    lines.append(
        f"clears {summary['clears']}/{summary['completed']} (completed) = "
        f"{_pct(summary['clear_rate_completed'])}  路  "
        f"declared denominator {summary['clears']}/{summary['declared']} = "
        f"{_pct(summary['clear_rate_declared'])}"
    )
    return "\n".join(lines)


def markdown_psn(summary: dict) -> str:
    lines = [
        "| stage | name | cap | cap source | clear | frames | deaths | bombs | boss | exit | wall_s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary["stages"]:
        boss = row.get("boss") or {}
        lines.append(
            "| {stage} | {name} | {cap} | {source} | {clear} | {frames} | {deaths} | {bombs} | {boss} | {code} | {wall} |".format(
                stage=row["stage"],
                name=row["stage_name"],
                cap=row["cap"],
                source=_short_cap_source(row["cap_source"]),
                clear=("**true**" if row.get("stage_clear") else "false") if row.get("ran") else "n/a",
                frames=row.get("frames", "n/a"),
                deaths=row.get("deaths", "n/a"),
                bombs=row.get("bombs_spent", "n/a"),
                boss=_boss_label(boss) if row.get("ran") else "n/a",
                code=row.get("exit_code", "n/a"),
                wall=row.get("wall_s", "n/a"),
            )
        )
    lines.append("")
    lines.append(
        f"deaths/stage = {_num(summary['deaths_mean_per_stage'])} over {summary['stages_ran']}/{summary['stages_declared']} stages"
    )
    return "\n".join(lines)


def _short_cap_source(text: str) -> str:
    text = str(text)
    if text.startswith("ceil(1.5*"):
        return text.split(") from ")[0] + ")"
    return text


def _boss_label(boss: dict) -> str:
    if not boss or not boss.get("boss_present_decisions"):
        return "absent"
    return f"{boss.get('boss_kill_evidence')} (life {boss.get('boss_life_first')}->{boss.get('boss_life_min')})"


def _pct(value) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _num(value) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


# ---------------------------------------------------------------------------
# batch drivers (serial or N-way parallel, each worker on its own data-dir)
# ---------------------------------------------------------------------------
def _already_done(records: list[dict]) -> set[str]:
    return {str(row.get("case_id")) for row in records if row.get("completed")}


def _data_dirs(args, jobs: int) -> list[str]:
    """One game data dir per worker; refuse to run parallel workers on one dir."""

    if args.data_dirs:
        dirs = [item.strip() for item in str(args.data_dirs).split(",") if item.strip()]
    else:
        dirs = []
    if not dirs:
        if jobs > 1:
            raise SystemExit(
                f"--jobs {jobs} needs --data-dirs with at least {jobs} entries: parallel "
                "workers must not share score.dat / th08.cfg writes"
            )
        return [str(args.data_dir)]
    if len(dirs) < jobs:
        raise SystemExit(f"--jobs {jobs} but only {len(dirs)} data dirs given")
    return dirs[:jobs]


def _case_record(case: dict, args, out_dir: Path, data_dir: str, workers: int, slot: int) -> dict:
    return run_case(
        case_id=case["case_id"],
        suite=case["suite"],
        stage=case["stage"],
        seed=case["seed"],
        cap=case["cap"],
        cap_source=case["cap_source"],
        out_dir=out_dir,
        python=args.python,
        timeout_s=args.timeout,
        with_boss=not args.no_boss,
        data_dir=data_dir,
        jobs=workers,
        worker=slot,
    )


def _progress_line(suite: str, record: dict, done: int, total: int) -> str:
    return (
        "[{suite}] {cid} clear={clear} frames={frames} deaths={deaths} stop={stop} "
        "wall={wall}s exit={code} ({done}/{total})"
    ).format(
        suite=suite,
        cid=record["case_id"],
        clear=record.get("stage_clear"),
        frames=record.get("frames"),
        deaths=record.get("deaths"),
        stop=record.get("stop_reason") if record.get("completed") else record.get("error"),
        wall=record.get("wall_s"),
        code=record.get("exit_code"),
        done=done,
        total=total,
    )


def _execute_cases(cases: list[dict], args, runs_path: Path, out_dir: Path, suite: str, total: int) -> list[dict]:
    """Run every pending case, appending each record as soon as it lands.

    ``--jobs 1`` is a plain loop.  ``--jobs N`` keeps N subprocesses in flight
    (one per worker slot, each pinned to its own data dir) and appends results in
    completion order; the teacher never sees a shared mutable game directory.
    """

    records = load_records(runs_path)
    done = _already_done(records)
    jobs = max(1, int(getattr(args, "jobs", 1) or 1))
    dirs = _data_dirs(args, jobs)
    pending = [case for case in cases if args.force or case["case_id"] not in done]
    for case in cases:
        if case["case_id"] in done and not args.force:
            print(f"[{suite}] {case['case_id']} already recorded; skipping", flush=True)
    deadline = time.time() + float(args.budget_seconds) if args.budget_seconds else None
    completed = len([row for row in records if row.get("completed")])

    def budget_open() -> bool:
        return deadline is None or time.time() < deadline

    if jobs == 1:
        for case in pending:
            if not budget_open():
                print(f"[{suite}] budget exhausted before {case['case_id']}; stopping", flush=True)
                break
            record = _case_record(case, args, out_dir, dirs[0], 1, 0)
            append_record(runs_path, record)
            records.append(record)
            if record.get("completed"):
                completed += 1
            print(_progress_line(suite, record, completed, total), flush=True)
        return records

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    free = list(range(jobs))
    queue = list(pending)
    running: dict = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        def submit_one() -> bool:
            if not queue or not free or not budget_open():
                return False
            case = queue.pop(0)
            slot = free.pop(0)
            future = pool.submit(_case_record, case, args, out_dir, dirs[slot], jobs, slot)
            running[future] = (case, slot)
            return True

        while len(running) < jobs and submit_one():
            pass
        while running:
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for future in finished:
                case, slot = running.pop(future)
                free.append(slot)
                free.sort()
                record = future.result()
                append_record(runs_path, record)
                records.append(record)
                if record.get("completed"):
                    completed += 1
                print(_progress_line(suite, record, completed, total), flush=True)
                submit_one()
    if queue:
        print(f"[{suite}] {len(queue)} case(s) not started (budget or time); partial", flush=True)
    return records


def run_s1n(args) -> int:
    human = load_human_reference(Path(args.reference))
    cap, cap_source = s1n_cap(human)
    out_dir = Path(args.out_dir).resolve()
    runs_path = out_dir / "s1n_runs.jsonl"
    seeds = [int(value) for value in (args.seeds or S1N_SEEDS)]
    cases = [
        {
            "case_id": f"s1n_seed{seed}",
            "suite": "s1n",
            "stage": S1N_STAGE,
            "seed": seed,
            "cap": cap,
            "cap_source": cap_source,
        }
        for seed in seeds
    ]
    _execute_cases(cases, args, runs_path, out_dir, "s1n", len(seeds))
    summary = summarize_s1n(load_records(runs_path), len(seeds))
    write_summary(out_dir, "s1n_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(markdown_s1n(load_records(runs_path), summary), flush=True)
    return 0 if summary["completed"] == summary["declared"] else 1


def run_psn(args) -> int:
    human = load_human_reference(Path(args.reference))
    caps = psn_caps(human)
    out_dir = Path(args.out_dir).resolve()
    runs_path = out_dir / "psn_runs.jsonl"
    selected = set(int(value) for value in args.stages) if args.stages else set(PSN_STAGES)
    cases = [
        {
            "case_id": f"psn_stage{int(row['stage'])}",
            "suite": "psn",
            "stage": int(row["stage"]),
            "seed": PSN_SEED,
            "cap": int(row["cap"]),
            "cap_source": row["cap_source"],
        }
        for row in caps
        if int(row["stage"]) in selected
    ]
    _execute_cases(cases, args, runs_path, out_dir, "psn", len(cases))
    summary = summarize_psn(load_records(runs_path), caps)
    write_summary(out_dir, "psn_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(markdown_psn(summary), flush=True)
    return 0 if summary["stages_ran"] == len(selected) else 1


def write_summary(out_dir: Path, name: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run_summary(args) -> int:
    out_dir = Path(args.out_dir).resolve()
    human = load_human_reference(Path(args.reference))
    cap, cap_source = s1n_cap(human)
    caps = psn_caps(human)
    s1n_records = load_records(out_dir / "s1n_runs.jsonl")
    psn_records = load_records(out_dir / "psn_runs.jsonl")
    s1n = summarize_s1n(s1n_records, len(S1N_SEEDS))
    psn = summarize_psn(psn_records, caps)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reference": str(HUMAN_REFERENCE.relative_to(REPO_ROOT)),
        "s1n_cap": cap,
        "s1n_cap_source": cap_source,
        "s1n": s1n,
        "psn": psn,
    }
    write_summary(out_dir, "g1_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    print(markdown_s1n(s1n_records, s1n), flush=True)
    print(markdown_psn(psn), flush=True)
    return 0


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------
def _default_python() -> str:
    return sys.executable or "python3"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--reference", default=str(HUMAN_REFERENCE))
    parser.add_argument("--python", default=_default_python(), help="interpreter for the teacher CLI")
    parser.add_argument("--timeout", type=float, default=5400.0, help="per-run hard timeout (s)")
    parser.add_argument("--budget-seconds", type=float, default=0.0, help="stop starting new runs after N s")
    parser.add_argument("--no-boss", action="store_true", help="skip the Boss stream analysis")
    parser.add_argument("--force", action="store_true", help="re-run cases already in the JSONL")
    parser.add_argument("--jobs", type=int, default=1, help="parallel teacher processes (each needs its own data dir)")
    parser.add_argument("--data-dir", default="/root/v400-data/p1s1/game", help="game data dir for a single worker")
    parser.add_argument(
        "--data-dirs",
        default=None,
        help="comma-separated data dirs, one per parallel worker (required when --jobs > 1)",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    s1n = subs.add_parser("s1n", help="S1N: 20 episodes, stage 0, seeds 1000..1019")
    s1n.add_argument("--seeds", nargs="*", type=int, default=None)
    s1n.set_defaults(func=run_s1n)

    psn = subs.add_parser("psn", help="PSN: stages 0..7, seed 1")
    psn.add_argument("--stages", nargs="*", type=int, default=None)
    psn.set_defaults(func=run_psn)

    summary = subs.add_parser("summary", help="recompute g1_summary.json from the JSONL")
    summary.set_defaults(func=run_summary)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
