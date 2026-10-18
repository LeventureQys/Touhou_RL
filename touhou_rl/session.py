"""Game session management: run directory, process launch, window and input.

A :class:`GameSession` owns exactly one ``th08.exe`` process and its run
directory.  Two properties matter for RL:

* **Isolation** -- every session gets its own directory (``th08.cfg``,
  ``score.dat``, ``log.txt``, ``replay/``) so that several instances can run at
  the same time without touching each other or the reference ``build/play``.
* **Input path** -- sessions launched with ``dontUseDirectInput`` accept keys
  posted straight to the window, so they do not need foreground focus.  That is
  what makes parallel data collection possible; the default DirectInput path
  would allow exactly one focused instance.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config_file, spec, winapi
from .memory import ProcessMemory
from .state import GameState, GameStateReader

REQUIRED_FILES = ("th08.exe", "th08.dat", "thbgm.dat")


def prepare_run_dir(
    run_dir: Path,
    template_dir: Path,
    *,
    windowed: bool = True,
    default_difficulty: int = spec.LUNATIC,
    use_postmessage_input: bool = False,
    force: bool = False,
) -> Path:
    """Materialise ``template_dir`` (``build/play``) into ``run_dir``.

    The three large assets are hard-linked when possible (same volume), which
    makes N parallel sessions cost almost no disk.  ``thbgm.dat`` is skipped when
    ``copy_bgm`` is false and music is muted: TH08 starts fine without it
    (``Supervisor::LoadConfig`` only logs the missing BGM file and switches
    ``musicMode`` to MIDI, which we then disable again on the next write).
    """
    run_dir = Path(run_dir).resolve()
    template_dir = Path(template_dir).resolve()
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "replay").mkdir(exist_ok=True)
    (run_dir / "evidence").mkdir(exist_ok=True)

    for name in REQUIRED_FILES:
        src = template_dir / name
        dst = run_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing game asset {src}")
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)  # same volume: N sessions cost ~no extra disk
        except OSError:
            shutil.copy2(src, dst)

    for name in ("score.dat",):
        src = template_dir / name
        dst = run_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    cfg_path = run_dir / "th08.cfg"
    if not cfg_path.exists():
        src_cfg = template_dir / "th08.cfg"
        if src_cfg.exists():
            shutil.copy2(src_cfg, cfg_path)
    config_file.prepare_config(
        cfg_path,
        windowed=windowed,
        default_difficulty=default_difficulty,
        use_postmessage_input=use_postmessage_input,
    )
    return run_dir


@dataclass
class SessionOptions:
    run_dir: Path
    windowed: bool = True
    highdpi_shim: bool = True
    boot_timeout_s: float = 40.0
    log_to_file: bool = True


class GameSession:
    """One launched TH08 process plus its readers and input backend."""

    def __init__(self, options: SessionOptions):
        self.options = options
        self.run_dir = Path(options.run_dir).resolve()
        self.process: subprocess.Popen | None = None
        self.pid = 0
        self.hwnd = 0
        self.mem: ProcessMemory | None = None
        self.reader: GameStateReader | None = None
        self.keyboard: winapi.SendInputKeyboard | winapi.PostMessageKeyboard | None = None
        self.capture: winapi.GdiCapture | None = None
        self.client_size = (0, 0)
        self.boot_seconds = 0.0
        self.log: list[str] = []
        self._log_path = self.run_dir / "rl-session.log"
        self._log_file = None

    # -- logging ----------------------------------------------------------
    def log_line(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message}"
        self.log.append(line)
        if self._log_file is not None:
            self._log_file.write(line + "\n")
            self._log_file.flush()

    # -- lifecycle --------------------------------------------------------
    def start(self) -> "GameSession":
        exe = self.run_dir / "th08.exe"
        if not exe.exists():
            raise FileNotFoundError(f"missing {exe}; call prepare_run_dir first")
        # Our own process must be DPI aware, otherwise GetClientRect reports the
        # virtualised size (640x480 / 1.25 = 512x384 on this 125% host).
        winapi.dpi_aware()
        last_error: Exception | None = None
        for attempt in range(1, 4):
            self._wait_no_other_instance()
            if self.options.log_to_file and self._log_file is None:
                self._log_file = open(self._log_path, "w", encoding="utf-8")
            try:
                self._launch_and_boot(exe)
                return self
            except Exception as exc:
                last_error = exc
                self.log_line(f"boot attempt {attempt} failed: {exc}")
                self._kill_process()
                time.sleep(2.0)
        raise RuntimeError(f"could not boot th08 after 3 attempts: {last_error}")

    def _wait_no_other_instance(self) -> None:
        """TH08 holds the session-wide mutex ``Touhou 08 App``; wait for a clear slate."""
        deadline = time.perf_counter() + 25.0
        others: list[int] = []
        while time.perf_counter() < deadline:
            others = list(winapi.list_process_ids("th08.exe"))
            if not others:
                return
            time.sleep(0.25)
        raise RuntimeError(f"another th08.exe is still running: {others}")

    def _kill_process(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
        if self.mem is not None:
            self.mem.close()
            self.mem = None
        self.process = None
        self.reader = None
        self.hwnd = 0

    def _launch_and_boot(self, exe) -> None:
        env = dict(os.environ)
        if self.options.highdpi_shim:
            # The game is DPI-unaware; without the shim a 125% desktop makes it
            # render into a virtualised 800x600 surface and every ROI shifts.
            env["__COMPAT_LAYER"] = "HIGHDPIAWARE"
        env.pop("PYTHONPATH", None)
        started = time.perf_counter()
        self.process = subprocess.Popen(
            [str(exe)],
            cwd=str(self.run_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.pid = self.process.pid
        self.log_line(f"launched pid={self.pid} dir={self.run_dir}")

        self.mem = ProcessMemory(self.pid)
        self.reader = GameStateReader(self.mem)
        try:
            self.wait_supervisor_state(spec.STATE_TITLE_SCREEN, timeout=self.options.boot_timeout_s)
        except TimeoutError:
            dialog = winapi.find_dialog(self.pid)
            if dialog:
                text = winapi.dialog_text(dialog).replace("\n", " | ")
                raise RuntimeError(f"th08 start-up dialog: {text[:300]}") from None
            raise
        self.boot_seconds = time.perf_counter() - started
        self.hwnd = self.wait_stable_window(timeout=20.0)
        x, y, w, h = winapi.client_rect(self.hwnd)
        self.client_size = (w, h)
        self.capture = winapi.GdiCapture(w, h)
        cfg = config_file.GameConfig.load(self.run_dir / "th08.cfg")
        if cfg.get_opt(config_file.OPT_DONT_USE_DIRECTINPUT):
            self.keyboard = winapi.PostMessageKeyboard(self.hwnd)
            self.log_line("input backend: PostMessage (dontUseDirectInput)")
        else:
            self.keyboard = winapi.SendInputKeyboard()
            self.log_line("input backend: SendInput (requires foreground)")
        self.log_line(f"window hwnd=0x{self.hwnd:X} client={w}x{h} boot={self.boot_seconds:.2f}s")
        return self

    def wait_stable_window(self, timeout: float = 20.0) -> int:
        """TH08 recreates its window during D3D setup; wait for a stable client rect."""
        deadline = time.perf_counter() + timeout
        last = 0
        last_rect = None
        seen: list[str] = []
        while time.perf_counter() < deadline:
            hwnd = winapi.find_window(self.pid)
            if hwnd:
                rect = winapi.client_rect(hwnd)
                if hwnd == last and rect == last_rect and rect[2] > 0 and rect[3] > 0:
                    return hwnd
                last, last_rect = hwnd, rect
            else:
                seen.append(f"no-window(proc_alive={self.process.poll() is None})")
            time.sleep(0.1)
        detail = "; ".join(seen[-5:]) or f"last hwnd=0x{last:X} rect={last_rect}"
        raise TimeoutError(f"no stable game window for pid {self.pid}: {detail}")

    def refresh_window(self) -> int:
        """Re-validate the cached HWND (it can be recreated across state changes)."""
        if not self.hwnd or not winapi.user32.IsWindow(self.hwnd):
            self.hwnd = winapi.find_window(self.pid)
            if self.hwnd and isinstance(self.keyboard, winapi.PostMessageKeyboard):
                self.keyboard.hwnd = self.hwnd
        return self.hwnd

    def read_state(self, with_bullets: bool = True, with_title: bool = True) -> GameState:
        assert self.reader is not None
        return self.reader.read_state(with_bullets=with_bullets, with_title=with_title)

    def wait_supervisor_state(self, state: int, timeout: float = 40.0, interval: float = 0.02) -> GameState:
        deadline = time.perf_counter() + timeout
        last_error: Exception | None = None
        while time.perf_counter() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"th08.exe exited during boot (code {self.process.returncode})")
            try:
                current = self.read_state(with_bullets=False, with_title=False)
                if current.supervisor_state == state:
                    return current
            except OSError as exc:  # memory not committed yet during boot
                last_error = exc
            time.sleep(interval)
        raise TimeoutError(f"timeout waiting for supervisor_state={state}: {last_error!r}")

    def wait_for(self, predicate, timeout: float = 20.0, interval: float = 0.01, description: str = "") -> GameState:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            state = self.read_state(with_bullets=False)
            if predicate(state):
                return state
            time.sleep(interval)
        raise TimeoutError(f"timeout waiting for {description or predicate}")

    # -- framing ----------------------------------------------------------
    def grab(self):
        assert self.capture is not None
        self.refresh_window()
        return self.capture.grab(self.hwnd)

    # -- shutdown ---------------------------------------------------------
    def close(self, timeout: float = 10.0) -> int | None:
        if self.keyboard is not None:
            try:
                self.keyboard.release_all()
            except OSError:
                pass
        if self.process is None:
            return None
        if self.process.poll() is None:
            target = self.hwnd
            if not target or not winapi.user32.IsWindow(target):
                target = winapi.find_window(self.pid)
            if target:
                try:
                    winapi.close_window(target)
                except OSError:
                    pass
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.log_line("WM_CLOSE timed out; killing process")
                self.process.kill()
                self.process.wait(timeout=5.0)
        if self.capture is not None:
            self.capture.close()
            self.capture = None
        if self.mem is not None:
            self.mem.close()
            self.mem = None
        code = self.process.returncode
        self.log_line(f"closed exit={code}")
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        return code

    def __enter__(self) -> "GameSession":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()
