#!/usr/bin/env python
"""S3-8 startup evidence probe: which exe runs, and is the captured frame live?

The startup smoke could not finish (no foreground for DirectInput), so this probe
collects the parts that *do* work and that A-04's evidence needs anyway:

* the **executable the live process is actually running** (queried from
  ``Win32_Process.ExecutablePath``) with its size and sha256, while also recording
  the template exe and the original ``resources/th08.exe`` - the rebuilt 898048-byte
  build and the 840704-byte original are named separately, never conflated;
* the window identity with a Unicode-safe title (``GetWindowTextW``), class, client
  rect and the visible / iconic / cloaked / foreground flags;
* two raw client-area captures 1.6 s apart, so a stale DC would show up as two
  identical frames (that is the v1.0.3 failure mode the repo warns about);
* the session log tail and a clean-close check.

No key is injected: the game boots to the title screen, is captured twice, and is
closed.  One instance at a time (the session enforces the game mutex).
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import hashlib
import json
import subprocess
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
from PIL import Image  # noqa: E402

from touhou_rl import winapi  # noqa: E402
from touhou_rl.session import GameSession, SessionOptions, prepare_run_dir  # noqa: E402

user32 = ctypes.WinDLL("user32", use_last_error=True)


def rel(path) -> str:
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def dig(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def title_unicode(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(max(length + 1, 256))
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value


def exe_of(pid: int) -> dict:
    node = {"pid": int(pid), "path": None, "sha256": None, "bytes": None, "source": "Win32_Process.ExecutablePath"}
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').ExecutablePath",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    if lines:
        node["path"] = lines[0]
    if node["path"] and Path(node["path"]).is_file():
        node["bytes"] = Path(node["path"]).stat().st_size
        node["sha256"] = dig(Path(node["path"]))
    return node


def main(argv: list[str] | None = None) -> int:
    out_dir = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")).resolve()
    run_dir = ROOT / "third_party/th08/build/rl/eval/v104-s38-startup-evidence"
    prepare_run_dir(run_dir, ROOT / "third_party/th08/build/play", windowed=True)
    session = GameSession(SessionOptions(run_dir=run_dir, log_to_file=True))
    report: dict = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "producer": "Document/Update/Alpha/v1.0.4/Stage3/S3-8/evidence/startup_evidence_probe.py",
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "run_dir": rel(run_dir),
        "screenshots": [],
        "binaries": {
            "original_resources": {
                "path": "third_party/th08/resources/th08.exe",
                "bytes": (ROOT / "third_party/th08/resources/th08.exe").stat().st_size
                if (ROOT / "third_party/th08/resources/th08.exe").is_file()
                else None,
                "sha256": dig(ROOT / "third_party/th08/resources/th08.exe")
                if (ROOT / "third_party/th08/resources/th08.exe").is_file()
                else None,
                "note": "pristine 840704-byte original, never used as the run directory exe",
            },
            "template": {
                "path": "third_party/th08/build/play/th08.exe",
                "bytes": (ROOT / "third_party/th08/build/play/th08.exe").stat().st_size,
                "sha256": dig(ROOT / "third_party/th08/build/play/th08.exe"),
                "note": "rebuilt 898048-byte build that prepare_run_dir links into the run directory",
            },
        },
    }
    exit_code = 0
    started = time.perf_counter()
    try:
        session.start()
        report["boot_seconds"] = round(time.perf_counter() - started, 2)
        report["pid"] = session.pid
        report["hwnd"] = f"0x{session.hwnd:X}"
        report["input_backend"] = type(session.keyboard).__name__
        report["live_process"] = exe_of(session.pid)
        target = session.hwnd
        report["window"] = {
            "title_unicode": title_unicode(target),
            "class": winapi.class_name(target),
            "client_rect": list(winapi.client_rect(target)),
            "visible": bool(user32.IsWindowVisible(target)),
            "iconic": bool(winapi.is_iconic(target)),
            "cloaked": bool(winapi.is_cloaked(target)),
            "foreground": bool(user32.GetForegroundWindow() == target),
        }
        first = np.asarray(session.grab())
        first_path = out_dir / "startup-title-a.png"
        Image.fromarray(first).save(first_path)
        report["screenshots"].append(rel(first_path))
        time.sleep(1.6)
        second = np.asarray(session.grab())
        second_path = out_dir / "startup-title-b.png"
        Image.fromarray(second).save(second_path)
        report["screenshots"].append(rel(second_path))
        report["capture"] = {
            "shape": list(first.shape),
            "mean": round(float(first.mean()), 3),
            "std": round(float(first.std()), 3),
            "frames_differ": bool(not np.array_equal(first, second)),
            "second_mean": round(float(second.mean()), 3),
            "second_std": round(float(second.std()), 3),
        }
        report["state"] = {
            "supervisor_state": int(session.read_state(with_bullets=False).supervisor_state),
            "title_screen_current": int(session.read_state(with_bullets=False).title_screen_current),
        }
    except Exception as exc:  # noqa: BLE001
        import traceback

        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        exit_code = 1
    finally:
        try:
            session.close()
        except Exception as exc:  # noqa: BLE001
            report["close_error"] = repr(exc)
        report["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
        log_path = run_dir / "rl-session.log"
        report["session_log"] = rel(log_path) if log_path.is_file() else None
        if log_path.is_file():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            report["session_log_tail"] = text.splitlines()[-10:]
            report["clean_close"] = any("closed exit=0" in line for line in text.splitlines())
            report["exception_markers"] = [
                marker
                for marker in ("traceback", "exception", "fatal")
                if marker in text.lower()
            ]
        exe_in_run = run_dir / "th08.exe"
        report["run_dir_exe"] = (
            {"path": rel(exe_in_run), "bytes": exe_in_run.stat().st_size, "sha256": dig(exe_in_run)}
            if exe_in_run.is_file()
            else None
        )
    (out_dir / "startup-probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "boot_seconds": report.get("boot_seconds"),
                "live_exe": report.get("live_process"),
                "run_dir_exe": report.get("run_dir_exe"),
                "window": report.get("window"),
                "capture": report.get("capture"),
                "clean_close": report.get("clean_close"),
            },
            ensure_ascii=False,
        )
    )
    print(f"written: {out_dir / 'startup-probe.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
