"""Isolate which th08.cfg field makes the rebuilt TH08 fail to start."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from ctypes import wintypes as wt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from touhou_rl import config_file  # noqa: E402

BASE = ROOT / "third_party/th08/build/play"


def dialog_text(pid: int) -> list[str]:
    from touhou_rl import winapi

    found: list[str] = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _lp):
        owner = wt.DWORD()
        winapi.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value != pid:
            return True
        cls = ctypes.create_unicode_buffer(256)
        winapi.user32.GetClassNameW(hwnd, cls, 256)
        if cls.value != "#32770":
            return True
        found.append("TITLE:" + winapi.window_title(hwnd))

        @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        def child(ch, _lp2):
            buf = ctypes.create_unicode_buffer(8192)
            winapi.user32.GetWindowTextW(ch, buf, 8192)
            if buf.value.strip():
                found.append(buf.value)
            return True

        winapi.user32.EnumChildWindows(hwnd, child, 0)
        return True

    winapi.user32.EnumWindows(cb, 0)
    return found


def trial(name: str, mutate) -> dict:
    run = ROOT / f"third_party/th08/build/rl/cfgtest-{name}"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    for asset in ("th08.exe", "th08.dat", "thbgm.dat"):
        os.link(BASE / asset, run / asset)
    shutil.copy2(BASE / "score.dat", run / "score.dat")
    cfg = config_file.GameConfig.load(BASE / "th08.cfg")
    mutate(cfg)
    cfg.save(run / "th08.cfg")
    env = dict(os.environ)
    env["__COMPAT_LAYER"] = "HIGHDPIAWARE"
    proc = subprocess.Popen([str(run / "th08.exe")], cwd=str(run), env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(6.0)
    alive = proc.poll() is None
    texts = dialog_text(proc.pid)
    result = {
        "name": name,
        "alive": alive,
        "windowed": cfg.windowed,
        "opts": hex(cfg.opts),
        "dialog": texts,
        "config_on_disk": hex(config_file.GameConfig.load(run / "th08.cfg").opts),
    }
    proc.kill()
    proc.wait(timeout=5)
    time.sleep(0.5)
    return result


def main() -> int:
    results = []
    results.append(trial("baseline", lambda cfg: None))
    results.append(trial("windowed", lambda cfg: setattr(cfg, "windowed", 1)))
    results.append(
        trial(
            "windowed_dinput",
            lambda cfg: (setattr(cfg, "windowed", 1), cfg.set_opt(config_file.OPT_DONT_USE_DIRECTINPUT, True)),
        )
    )
    results.append(
        trial(
            "windowed_novsync",
            lambda cfg: (setattr(cfg, "windowed", 1), cfg.set_opt(config_file.OPT_DISABLE_VSYNC, True)),
        )
    )
    results.append(
        trial(
            "windowed_full",
            lambda cfg: (
                setattr(cfg, "windowed", 1),
                cfg.set_opt(config_file.OPT_DONT_USE_DIRECTINPUT, True),
                cfg.set_opt(config_file.OPT_DISABLE_VSYNC, True),
            ),
        )
    )
    out = ROOT / "third_party/th08/build/rl/cfgtest.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in results:
        print(item["name"], "alive=", item["alive"], "dialog=", bool(item["dialog"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
