#!/usr/bin/env python
"""S3-8 A-04 evidence consolidation (no game launch, read-only over the attempts).

Attempt 1 of the startup smoke and attempt 1 of the formal A-04 command both
failed at ``menu.select_difficulty`` because the game window cannot be made
foreground (see ``foreground-diagnostic.json``: the user's elevated Task Manager
owns the foreground, ``SPI_GETFOREGROUNDLOCKTIMEOUT = 0x7FFFFFFF``, and every
foreground path a medium-integrity process has is denied).

This script turns the raw attempts into two contracted artifacts:

``startup-evidence.json``
    The startup evidence index's artifact.  Its ``observations`` are the facts as
    measured: the game process really started and rendered (two raw PNGs from the
    focus probe), the checkpoint really loaded, but no action was ever produced, so
    the smoke cycle did not complete and A-04's startup gate is blocked rather than
    passed.

``a04-attempt.json``
    The formal ``python -m touhou_rl.evaluate`` attempt: exact argv, exit code,
    stderr excerpt, the partial ``eval-report.json`` (zero episodes) and its sha.

Both files carry the sha256 of everything they cite, so a reviewer can rebuild the
story from bytes instead of prose.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


def _repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "touhou_rl" / "vision_data.py").exists():
            return candidate
    raise SystemExit("cannot locate the repository root")


ROOT = _repo_root(Path(__file__).resolve().parent)
EVIDENCE = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "S3-8" / "evidence"
FROZEN = ROOT / "Document" / "Update" / "Alpha" / "v1.0.4" / "Stage3" / "final-artifacts.json"


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def dig(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_block(path: Path, **extra) -> dict:
    node = {"path": rel(path), "sha256": dig(path) if path.is_file() else None, "exists": path.is_file()}
    node.update(extra)
    return node


def main() -> int:
    frozen = read(FROZEN)
    smoke = read(EVIDENCE / "startup-smoke.json")
    diag = read(EVIDENCE / "foreground-diagnostic.json")
    probe = read(EVIDENCE / "focus-probe.json")
    startup_probe = read(EVIDENCE / "startup-probe.json")
    evaluate_err = EVIDENCE / "logs" / "a04-evaluate.err"
    evaluate_log = EVIDENCE / "logs" / "a04-evaluate.log"
    partial_report = ROOT / "third_party" / "th08" / "build" / "rl" / "eval" / "v104-acceptance" / "eval-report.json"
    raw_diag = ROOT / "third_party" / "th08" / "build" / "rl" / "eval" / "v104-acceptance" / "vision-diagnostics-raw.json"
    error_text = evaluate_err.read_text(encoding="utf-8", errors="replace") if evaluate_err.is_file() else ""

    screenshots = [ROOT / path for path in startup_probe.get("screenshots") or []]
    screenshots = [path for path in screenshots if path.is_file()]
    window = startup_probe.get("window") or {}
    capture = startup_probe.get("capture") or {}
    ui_checked = bool(
        window.get("visible")
        and not window.get("iconic")
        and not window.get("cloaked")
        and len(screenshots) == 2
        and capture.get("frames_differ")
        and float(capture.get("std") or 0.0) > 5.0
        and float(capture.get("second_std") or 0.0) > 5.0
    )
    live = startup_probe.get("live_process") or {}
    # The game really ran: a live pid, its queried executable path, a stable 640x480
    # TH08 window and two rendered frames 1.6 s apart.  No action was ever produced,
    # so the smoke cycle's act/step half never happened.
    game_ran = bool(startup_probe.get("pid") and startup_probe.get("hwnd") and live.get("path") and len(screenshots) == 2)
    clean_close = bool(startup_probe.get("clean_close"))
    session_markers = startup_probe.get("exception_markers") or []
    # "no_exception_log" is kept false on purpose: the game's own log is clean, but the
    # smoke and the formal A-04 attempt both ended in a client-side TimeoutError, which
    # is recorded here instead of being smoothed away.
    client_timeout = "TimeoutError" in error_text or "TimeoutError" in "\n".join(smoke.get("errors") or [])

    startup = {
        "schema": "v104-s3-8-startup/1",
        "kind": "startup",
        "created_at": now_iso(),
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/make_a04_evidence.py",
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "status": "blocked",
        "checkpoint": frozen["policy_checkpoint"],
        "checkpoint_sha256": frozen["policy_checkpoint_sha256"],
        "observations": {
            "real_game_started": game_ran,
            "pixels_only": False,
            "no_crash": bool(clean_close and not session_markers),
            "no_exception_log": False,
            "ui_checked": ui_checked,
            "screenshots": [rel(path) for path in screenshots],
            "exe_path": live.get("path") or (probe.get("exe") or {}).get("path"),
            "exe_sha256": live.get("sha256") or (probe.get("exe") or {}).get("sha256"),
            "exe_bytes": live.get("bytes") or (probe.get("exe") or {}).get("bytes"),
            "original_exe_sha256": (startup_probe.get("binaries") or {}).get("original_resources", {}).get("sha256"),
            "original_exe_bytes": (startup_probe.get("binaries") or {}).get("original_resources", {}).get("bytes"),
            "window_title": window.get("title_unicode"),
            "window_class": window.get("class"),
            "window_client_rect": window.get("client_rect"),
            "window_foreground": window.get("foreground"),
            "boot_seconds": startup_probe.get("boot_seconds"),
            "capture_std": capture.get("std"),
            "capture_second_std": capture.get("second_std"),
            "frames_differ": capture.get("frames_differ"),
            "checkpoint_loaded": True,
            "no_action_produced": True,
            "client_timeout_error": bool(client_timeout),
            "failure_stage": "env.reset -> menu.select_difficulty (screen 4 did not become ready)",
        },
        "reason": (
            "the game process starts, the frozen v3 policy.pt loads, and the 640x480 client area really"
            " renders (two raw screenshots 1.6 s apart, different pixels, 58-60 fps on screen), so the"
            " exe/window/capture path is proven; but no key can reach the game: the foreground is held by"
            " the user's elevated Task Manager while this agent runs at medium integrity, so"
            " SetForegroundWindow / SwitchToThisWindow / synthetic input are all denied (UIPI +"
            " ForegroundLockTimeout=0x7FFFFFFF). reset therefore never reaches gameplay, no action is"
            " produced, and the startup smoke cannot complete its act/step half. no_exception_log is false"
            " because the client side ended in a TimeoutError (the game's own session log is clean). This"
            " is an environment/security boundary, not a model or code failure; reported as blocked,"
            " never as pass."
        ),
        "artifacts_cited": [
            file_block(EVIDENCE / "startup-probe.json"),
            file_block(EVIDENCE / "startup-smoke.json"),
            file_block(EVIDENCE / "foreground-diagnostic.json"),
            file_block(EVIDENCE / "focus-probe.json"),
        ]
        + [file_block(path) for path in screenshots],
        "foreground_diagnostic": {
            "foreground_before": diag.get("foreground_before"),
            "foreground_lock_timeout_ms": diag.get("foreground_lock_timeout_ms"),
            "attempts": diag.get("attempts"),
            "alternative": diag.get("alternative"),
        },
        "probe_methods": probe.get("methods"),
        "binaries": startup_probe.get("binaries"),
        "session_log_tail": startup_probe.get("session_log_tail"),
    }
    (EVIDENCE / "startup-evidence.json").write_text(
        json.dumps(startup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    argv = [
        "python",
        "-m",
        "touhou_rl.evaluate",
        "--vision-checkpoint",
        frozen["policy_checkpoint"],
        "--out",
        "third_party/th08/build/rl/eval/v104-acceptance",
        "--episodes",
        "3",
        "--difficulty",
        "lunatic",
        "--team",
        "0",
        "--frameskip",
        "2",
        "--max-steps",
        "12000",
        "--device",
        "cuda",
    ]
    partial = read(partial_report) if partial_report.is_file() else {}
    attempt = {
        "schema": "v104-s3-8-a04-attempt/1",
        "kind": "eval_report",
        "created_at": now_iso(),
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/make_a04_evidence.py",
        "command": argv,
        "cwd": str(ROOT),
        "exit_code": 1,
        "status": "blocked",
        "checkpoint": frozen["policy_checkpoint"],
        "checkpoint_sha256": frozen["policy_checkpoint_sha256"],
        "deployed_checkpoint": (partial.get("vision") or {}).get("checkpoint"),
        "arch": (partial.get("vision") or {}).get("arch"),
        "depth": (partial.get("vision") or {}).get("depth"),
        "difficulty": partial.get("difficulty"),
        "team": partial.get("team"),
        "frameskip": partial.get("frameskip"),
        "practice_stage": partial.get("practice_stage"),
        "max_steps_configured": (partial.get("vision") or {}).get("max_steps"),
        "episodes": len(partial.get("episodes") or []),
        "frames_mean": (partial.get("summary") or {}).get("frames_mean"),
        "hit_step_cap": 0,
        "episode_fields_complete": False,
        "observations": {
            "checkpoint_sha256": frozen["policy_checkpoint_sha256"],
            "episodes": len(partial.get("episodes") or []),
            "frames_mean": float((partial.get("summary") or {}).get("frames_mean") or 0.0),
            "hit_step_cap": 0,
            "episode_fields_complete": False,
            "started": partial.get("started"),
            "finished": partial.get("finished"),
            "max_steps": (partial.get("vision") or {}).get("max_steps"),
        },
        "reason": (
            "the exact A-04 command ran and the deployment identity was correct (vision.checkpoint = the"
            " frozen policy.pt, arch v3-st depth 3, lunatic/team0/frameskip2/practice -1), but env.reset"
            " raised TimeoutError('screen 4 did not become ready') because the game window cannot be"
            " foregrounded (elevated Task Manager holds the foreground; medium-integrity agent). Zero"
            " episodes were produced; the partial eval-report is attached with its sha. Reported as"
            " blocked; a retry needs the user to release the foreground (close/minimise Task Manager) or"
            " an approved input-path change."
        ),
        "stderr_excerpt": error_text.strip().splitlines()[-6:] if error_text else [],
        "artifacts_cited": [
            file_block(partial_report),
            file_block(raw_diag),
            file_block(evaluate_err),
            file_block(evaluate_log),
        ],
    }
    (EVIDENCE / "a04-attempt.json").write_text(
        json.dumps(attempt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "startup_observations": startup["observations"],
                "ui_checked": ui_checked,
                "screenshots": startup["observations"]["screenshots"],
                "a04_attempt": {"exit_code": 1, "episodes": attempt["episodes"], "status": attempt["status"]},
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
