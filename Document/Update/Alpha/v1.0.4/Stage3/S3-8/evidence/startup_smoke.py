#!/usr/bin/env python
"""S3-8 startup smoke: real game, frozen checkpoint, reset -> screenshot -> act -> step -> screenshot.

This is the A-04 predecessor that V-16 asks for, in its own directory so the
formal evaluation is untouched.  It launches the actual ``th08.exe`` from a fresh
run directory, loads the frozen ``policy.pt`` (sha verified against
``Stage3/final-artifacts.json``), does exactly one ``reset`` -> ``render`` ->
pixels-only ``act`` -> ``env.step`` -> ``render`` cycle, and writes:

* ``startup-reset.png`` / ``startup-step.png`` (raw frames, not thumbnails),
* ``startup-smoke.json`` with the contracted observations
  (``real_game_started`` / ``pixels_only`` / ``no_crash`` / ``no_exception_log`` /
  ``ui_checked`` / ``screenshots``) plus the running exe path and sha256, the
  window state, the input shape the policy saw, the action it produced, the
  render->action latency and the observed frame delta.

Nothing privileged reaches the policy: the only tensors handed to ``act`` are
``regions_to_batch`` of ``env.render()`` crops, and the environment bookkeeping
(``obs``/``info``) is never used to build the input or the action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path


def _repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "touhou_rl" / "vision_data.py").exists():
            return candidate
    raise SystemExit("cannot locate the repository root")


ROOT = _repo_root(Path(__file__).resolve().parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from touhou_rl import spec, vision, winapi  # noqa: E402
from touhou_rl.env import EnvConfig, Th08Env  # noqa: E402
from touhou_rl.vmodels import load_vision_checkpoint, vision_arch  # noqa: E402


def rel(path) -> str:
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def running_exe(pid: int, fallback: Path) -> dict:
    """The executable the live process is actually running, plus its sha256."""
    import subprocess

    info = {"pid": int(pid), "path": None, "sha256": None, "bytes": None, "source": None}
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\").ExecutablePath",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        text = (completed.stdout or "").strip().splitlines()
        if text and text[0].strip():
            info["path"] = text[0].strip()
            info["source"] = "Win32_Process.ExecutablePath"
    except Exception as exc:  # diagnostics only
        info["query_error"] = repr(exc)
    if info["path"] is None:
        info["path"] = str(fallback)
        info["source"] = "run_dir/th08.exe (path query unavailable)"
    path = Path(info["path"])
    if path.is_file():
        info["sha256"] = digest(path)
        info["bytes"] = path.stat().st_size
    return info


def regions_from_screen(screen, region_sizes) -> dict:
    native = vision.extract_regions(screen)
    out = {}
    for name, (height, width) in region_sizes.items():
        crop = native.get(name)
        if crop is None:
            out[name] = np.zeros((height, width, vision.VISION_CHANNELS), dtype=np.uint8)
        else:
            out[name] = vision.resize_area(crop, height, width)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="Document/Update/Alpha/v1.0.4/Stage3/final-artifacts.json")
    parser.add_argument("--out", required=True, help="output directory (own directory, never the eval dir)")
    parser.add_argument("--run-dir", default="third_party/th08/build/rl/eval/v104-s38-smoke-session")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((ROOT / args.artifacts).read_text(encoding="utf-8"))
    checkpoint = ROOT / frozen["policy_checkpoint"]
    expected_sha = frozen["policy_checkpoint_sha256"]
    started = now_iso()
    payload = {
        "schema": "v104-s3-8-startup-smoke/1",
        "created_at": started,
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/startup_smoke.py",
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "frozen_artifacts": args.artifacts,
        "checkpoint": rel(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "checkpoint_sha256_expected": expected_sha,
        "checkpoint_matches_frozen": digest(checkpoint) == expected_sha,
        "run_dir": None,
        "exe": {},
        "window": {},
        "screenshots": [],
        "observations": {},
        "steps": [],
        "errors": [],
    }
    if not payload["checkpoint_matches_frozen"]:
        payload["reason"] = "frozen checkpoint sha256 mismatch; refusing to start the game"
        (out_dir / "startup-smoke.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(payload["reason"], file=sys.stderr)
        return 2

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, extra = load_vision_checkpoint(checkpoint, map_location=device)
    model.to(device).eval()
    depth = int(model.config.depth)
    region_sizes = {name: tuple(int(v) for v in size) for name, size in model.config.region_sizes.items()}
    payload["arch"] = vision_arch(model)
    payload["depth"] = depth
    payload["device"] = str(device)
    payload["region_sizes"] = {name: list(size) for name, size in region_sizes.items()}

    run_dir = ROOT / args.run_dir
    payload["run_dir"] = rel(run_dir)
    env = Th08Env(
        EnvConfig(
            difficulty=spec.LUNATIC,
            team=0,
            practice_stage=None,
            frameskip=2,
            max_steps=12000,
            reset_mode="relaunch",
            run_dir=run_dir,
        )
    )
    env_started = time.perf_counter()
    exit_code = 0
    try:
        winapi.keep_awake(True)  # no screensaver/lock during the run; reverted in finally
        obs, info = env.reset()
        payload["boot_seconds"] = round(time.perf_counter() - env_started, 2)
        session = env.session
        payload["exe"] = running_exe(session.pid, run_dir / "th08.exe")
        hwnd = int(session.hwnd)
        payload["window"] = {
            "hwnd": f"0x{hwnd:X}",
            "title": winapi.window_title(hwnd),
            "class": winapi.class_name(hwnd),
            "client_rect": list(winapi.client_rect(hwnd)),
            "visible": bool(winapi.window_visible(hwnd)),
            "iconic": bool(winapi.is_iconic(hwnd)),
            "cloaked": bool(winapi.is_cloaked(hwnd)),
            "foreground": bool(winapi.user32.GetForegroundWindow() == hwnd),
            "input_backend": type(session.keyboard).__name__,
        }
        screen = np.asarray(env.render())
        payload["screen_shape"] = list(screen.shape)
        first_png = out_dir / "startup-reset.png"
        from PIL import Image

        Image.fromarray(screen).save(first_png)
        payload["screenshots"].append(rel(first_png))

        regions = regions_from_screen(screen, region_sizes)
        stacks = {}
        for name, size in region_sizes.items():
            stack = vision.FrameStack(depth=depth, shape=(*size, vision.VISION_CHANNELS))
            stack.reset()
            for _ in range(depth):
                stack.push(regions[name])
            stacks[name] = stack
        batch = vision.regions_to_batch(stacks, device)
        payload["input"] = {
            "keys": sorted(batch.keys()) if isinstance(batch, dict) else None,
            "shapes": {key: list(value.shape) for key, value in batch.items()}
            if isinstance(batch, dict)
            else [list(value.shape) for value in batch],
            "dtype": {key: str(value.dtype) for key, value in batch.items()}
            if isinstance(batch, dict)
            else None,
            "source": "env.render() crops only; obs/info never used",
            "isolation": "pixels-only",
        }
        state_before = {
            "stage": info.get("state", {}).get("stage"),
            "frame": info.get("state", {}).get("frame"),
        }
        decision_started = time.perf_counter()
        with torch.no_grad():
            action, logp, value, extra_out = model.act(batch)
        latency_ms = (time.perf_counter() - decision_started) * 1000.0
        action_numpy = action[0].cpu().numpy()
        payload["action"] = [int(v) for v in action_numpy]
        payload["action_dims"] = list(np.asarray(action_numpy).shape)
        payload["bomb_requested"] = int(action_numpy[3])
        payload["shoot_requested"] = int(action_numpy[1])
        obs2, reward, terminated, truncated, info2 = env.step(action_numpy)
        screen2 = np.asarray(env.render())
        second_png = out_dir / "startup-step.png"
        Image.fromarray(screen2).save(second_png)
        payload["screenshots"].append(rel(second_png))
        payload["steps"].append(
            {
                "decision_latency_ms": round(latency_ms, 3),
                "state_before": state_before,
                "state_after": {
                    "stage": info2.get("stage"),
                    "frame": info2.get("frame"),
                    "lives": info2.get("lives"),
                    "bombs": info2.get("bombs"),
                },
                "frames_advanced": info2.get("frames_advanced"),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        payload["screen_nonblank"] = {
            "reset_mean": round(float(screen.mean()), 3),
            "reset_std": round(float(screen.std()), 3),
            "step_mean": round(float(screen2.mean()), 3),
            "step_std": round(float(screen2.std()), 3),
            "frames_differ": bool(screen.shape != screen2.shape or not np.array_equal(screen, screen2)),
        }
    except Exception as exc:  # noqa: BLE001 - the whole point is to record the failure
        import traceback

        payload["errors"].append(traceback.format_exc())
        payload["reason"] = f"startup smoke failed: {exc!r}"
        exit_code = 1
    finally:
        try:
            env.close()
        except Exception as exc:  # noqa: BLE001
            payload["errors"].append(f"close: {exc!r}")
        try:
            winapi.keep_awake(False)
        except Exception:
            pass
        payload["finished"] = now_iso()

    process_exit = None
    if env.session is None and env.config is not None:
        process_exit = "closed"
    log_path = run_dir / "rl-session.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    payload["session_log"] = rel(log_path) if log_path.is_file() else None
    payload["session_log_lines"] = len(log_text.splitlines())
    payload["session_log_tail"] = log_text.splitlines()[-8:]
    lowered = (log_text + "\n" + "\n".join(payload["errors"])).lower()
    exception_markers = ("traceback", "exception", "error:", "fatal")
    payload["exception_markers"] = [marker for marker in exception_markers if marker in lowered]
    for name in ("log.txt",):
        side = run_dir / name
        if side.is_file():
            text = side.read_text(encoding="utf-8", errors="replace")
            payload[f"{name}_lines"] = len(text.splitlines())
            payload[f"{name}_tail"] = text.splitlines()[-5:]
            for marker in exception_markers:
                if marker in text.lower():
                    payload["exception_markers"].append(f"{name}:{marker}")

    ui_ok = bool(
        payload["window"].get("visible")
        and not payload["window"].get("iconic")
        and not payload["window"].get("cloaked")
        and payload.get("screen_nonblank", {}).get("reset_std", 0.0) > 5.0
        and payload.get("screen_nonblank", {}).get("step_std", 0.0) > 5.0
        and len(payload["screenshots"]) == 2
    )
    steps = payload["steps"]
    advanced = steps[0]["frames_advanced"] if steps else None
    observations = {
        "real_game_started": bool(exit_code == 0 and steps and payload["window"].get("hwnd")),
        "pixels_only": bool(payload.get("input", {}).get("source", "").startswith("env.render()")),
        "no_crash": bool(exit_code == 0 and not payload["errors"]),
        "no_exception_log": bool(not payload["exception_markers"]),
        "ui_checked": ui_ok,
        "screenshots": payload["screenshots"],
        "exe_path": payload["exe"].get("path"),
        "exe_sha256": payload["exe"].get("sha256"),
        "exe_bytes": payload["exe"].get("bytes"),
        "window_title": payload["window"].get("title"),
        "window_foreground": payload["window"].get("foreground"),
        "action": payload.get("action"),
        "decision_latency_ms": steps[0]["decision_latency_ms"] if steps else None,
        "frames_advanced": advanced,
        "note": (
            "one reset -> screenshot -> pixels-only act -> env.step -> screenshot cycle;"
            " this is a startup smoke, not A-04"
        ),
    }
    payload["observations"] = observations
    payload["status"] = (
        "pass"
        if all(observations[key] for key in ("real_game_started", "pixels_only", "no_crash", "no_exception_log", "ui_checked"))
        else "fail"
    )
    (out_dir / "startup-smoke.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(observations, ensure_ascii=False), flush=True)
    print(f"written: {out_dir / 'startup-smoke.json'}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
