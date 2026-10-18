"""Menu macros: bring the game from the title screen into a playable stage.

All navigation is closed-loop: each step reads the reported cursor/screen from
memory and corrects, rather than sleeping a fixed time and hoping.  That is what
makes it safe to run several instances whose frame pacing differs.

Relevant flow (``TitleScreen.cpp``):

* StartMenu ``currentScreen == 0``: items 0 Start, 3 Practice Start, 4 Replay.
* ``Z`` on Start/Practice -> DifficultySelect (``currentScreen`` 4 / 8),
  ``cursor`` = difficulty index (0 Easy .. 3 Lunatic).
* ``Z`` -> CharacterSelect (5 / 9), ``cursor`` = ``GameManager.shotType``;
  entries 0..3 are the four teams, 4..11 solo characters.
* Practice adds PracticeStageSelect (11) before the stage starts.
* Replay menu keeps ``currentScreen == 7`` and stores its own sub-state in
  ``currentScreenState``: 0 enumerate, 1 select replay, 2 select stage,
  3 select playback mode, 4 exit.  Replay files are enumerated from
  ``./replay/th8_NN.rpy`` (NN = 01..15) then ``./replay/th8_ud????.rpy``.
"""

from __future__ import annotations

import time
from pathlib import Path

from . import spec, winapi
from .session import GameSession

DEFAULT_TAP_HOLD_S = 0.06
DEFAULT_SETTLE_S = 0.12


def tap(session: GameSession, key: str, hold_s: float = DEFAULT_TAP_HOLD_S, settle_s: float = DEFAULT_SETTLE_S) -> None:
    assert session.keyboard is not None
    if isinstance(session.keyboard, winapi.SendInputKeyboard):
        if not winapi.force_foreground(session.refresh_window()):
            session.log_line("warning: could not take foreground for SendInput")
    session.keyboard.tap(key, hold_s)
    time.sleep(settle_s)


def _ensure_foreground_if_needed(session: GameSession) -> None:
    if isinstance(session.keyboard, winapi.SendInputKeyboard):
        winapi.force_foreground(session.refresh_window())


def ensure_title(session: GameSession, timeout: float = 25.0) -> None:
    """Return to the title screen, aborting attract demos and sub-menus."""
    deadline = time.perf_counter() + timeout
    session.log_line("ensure_title: begin")
    stable_reads = 0
    while time.perf_counter() < deadline:
        state = session.read_state(with_bullets=False)
        if state.supervisor_state == spec.STATE_TITLE_SCREEN:
            at_start_menu = state.title_screen_current == spec.SCREEN_START_MENU
            ready = state.title_screen_state == 1  # TitleCurrentScreenState_Ready
            # TitleScreen::OnUpdateStartMenu ignores input while stateTimer2 < 10
            # (TitleScreen.cpp:489), so "Ready" alone is not enough to be usable.
            receptive = state.title_timer2 >= 12
            if at_start_menu and ready and receptive:
                stable_reads += 1
                if stable_reads >= 2:
                    session.log_line("ensure_title: at start menu")
                    return
                time.sleep(0.05)
                continue
            stable_reads = 0
            if at_start_menu:
                time.sleep(0.05)
                continue
            # Inside another screen: back out one level.
            tap(session, "x", settle_s=0.35)
            continue
        stable_reads = 0
        if state.supervisor_state == spec.STATE_GAME_MANAGER:
            if state.is_demo:
                tap(session, "z", settle_s=0.2)
            else:
                # A real run is active: pause and quit to title.
                tap(session, "esc", settle_s=0.25)
                tap(session, "up", settle_s=0.1)
                tap(session, "up", settle_s=0.1)
                tap(session, "up", settle_s=0.1)
                tap(session, "z", settle_s=0.6)
            continue
        time.sleep(0.2)
    raise TimeoutError("could not reach the title start menu")


def goto_start_menu_item(session: GameSession, item: int, timeout: float = 5.0) -> None:
    """Move the StartMenu cursor to ``item`` using the reported cursor value."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        state = session.read_state(with_bullets=False)
        if state.title_screen_current != spec.SCREEN_START_MENU:
            raise RuntimeError(f"not on the start menu (current={state.title_screen_current})")
        delta = (item - state.title_cursor) % 9
        if delta == 0:
            return
        key = "down" if delta <= 4 else "up"
        tap(session, key, settle_s=0.1)
    raise TimeoutError(f"could not move start menu cursor to item {item}")


def _move_cursor(session: GameSession, target: int, getter, steps: int, timeout: float = 6.0) -> None:
    deadline = time.perf_counter() + timeout
    time.sleep(0.1)  # let one logic frame settle after the screen change
    while time.perf_counter() < deadline:
        current = getter()
        if current == target:
            return
        delta = (target - current) % steps
        key = "down" if delta <= steps // 2 else "up"
        tap(session, key, settle_s=0.08)
    raise TimeoutError(f"could not move cursor to {target}")


def _wait_screen(session: GameSession, screen: int, timeout: float = 8.0) -> None:
    """Wait for a title sub-screen to be ready *and* past its input gate.

    ``ChangeCurrentScreen`` resets ``stateTimer2`` and every screen ignores input
    for its first 10 frames (e.g. ``TitleScreen.cpp:489``), so "Ready" alone is
    not enough -- tapping too early silently does nothing.
    """
    deadline = time.perf_counter() + timeout
    last = None
    while time.perf_counter() < deadline:
        state = session.read_state(with_bullets=False)
        last = (state.title_screen_current, state.title_screen_state, state.title_timer2)
        if (
            state.title_screen_current == screen
            and state.title_screen_state == 1  # Ready
            and state.title_timer2 >= 12
        ):
            return
        time.sleep(0.02)
    raise TimeoutError(f"screen {screen} did not become ready (last={last})")


def select_difficulty(session: GameSession, difficulty: int, timeout: float = 8.0) -> None:
    expected = spec.SCREEN_DIFFICULTY_SELECT if difficulty != spec.EXTRA else spec.SCREEN_DIFFICULTY_SELECT_EXTRA
    if difficulty == spec.EXTRA:
        _wait_screen(session, expected, timeout)
        tap(session, "z", settle_s=0.25)
        return
    _wait_screen(session, spec.SCREEN_DIFFICULTY_SELECT, timeout)
    _move_cursor(
        session,
        difficulty,
        lambda: session.read_state(with_bullets=False).title_cursor,
        4,
    )
    tap(session, "z", settle_s=0.25)


def select_character(session: GameSession, team: int, practice: bool, timeout: float = 8.0) -> None:
    target_screen = spec.SCREEN_CHARACTER_SELECT_PRACTICE if practice else spec.SCREEN_CHARACTER_SELECT
    _wait_screen(session, target_screen, timeout)
    # The character grid is 4 teams (or 12 entries once Extra is unlocked);
    # only teams 0..3 are required by this project.
    _move_cursor(
        session,
        team,
        lambda: session.read_state(with_bullets=False).title_cursor,
        12,
    )
    tap(session, "z", settle_s=0.25)


def wait_human_game(session: GameSession, timeout: float = 30.0) -> None:
    """Wait until a non-demo, non-replay stage is actually running."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        state = session.read_state(with_bullets=False)
        if state.supervisor_state == spec.STATE_GAME_MANAGER and not state.is_demo and not state.is_replay:
            return
        if state.supervisor_state == spec.STATE_TITLE_SCREEN and state.is_demo:
            tap(session, "z", settle_s=0.2)
        time.sleep(0.05)
    raise TimeoutError("stage did not start")


def enter_game(
    session: GameSession,
    difficulty: int = spec.LUNATIC,
    team: int = 0,
    practice_stage: int | None = None,
) -> None:
    """Title -> (practice) stage start with the requested difficulty/team."""
    ensure_title(session)
    item = spec.MENU_ITEM_PRACTICE_START if practice_stage is not None else spec.MENU_ITEM_START
    goto_start_menu_item(session, item)
    tap(session, "z", settle_s=0.25)
    select_difficulty(session, difficulty)
    select_character(session, team, practice=practice_stage is not None)
    if practice_stage is not None:
        _wait_screen(session, spec.SCREEN_PRACTICE_STAGE_SELECT)
        _move_cursor(
            session,
            practice_stage,
            lambda: session.read_state(with_bullets=False).title_cursor,
            7,
        )
        tap(session, "z", settle_s=0.25)
    wait_human_game(session)
    session.log_line(f"entered game difficulty={difficulty} team={team} practice={practice_stage}")


def install_replay(run_dir: Path, replay_path: Path, index: int = 1) -> Path:
    """Expose ``replay_path`` as ``replay/th8_XX.rpy`` so it sorts first in the menu."""
    replay_dir = Path(run_dir) / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    for existing in replay_dir.glob("th8_*.rpy"):
        existing.unlink()
    target = replay_dir / f"th8_{index:02d}.rpy"
    target.write_bytes(Path(replay_path).read_bytes())
    return target


def enter_replay(session: GameSession, replay_index: int = 0, stage: int = 0, timeout: float = 30.0) -> None:
    """Title -> replay playback of the installed ``replay/th8_01.rpy``."""
    ensure_title(session)
    goto_start_menu_item(session, spec.MENU_ITEM_REPLAY)
    tap(session, "z", settle_s=0.3)

    def replay_state() -> tuple[int, int, int]:
        state = session.read_state(with_bullets=False)
        return state.title_screen_current, state.title_screen_state, state.title_replay_count

    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        screen, sub, count = replay_state()
        if screen == spec.SCREEN_REPLAY and count > 0 and sub == 1:
            break
        time.sleep(0.05)
    else:
        raise TimeoutError("replay list did not appear")

    # Choose the replay entry, then the stage, then normal playback.
    _move_cursor(
        session,
        replay_index,
        lambda: session.read_state(with_bullets=False).title_cursor,
        45,
    )
    tap(session, "z", settle_s=0.25)

    deadline = time.perf_counter() + 10.0
    while time.perf_counter() < deadline:
        screen, sub, _count = replay_state()
        if screen == spec.SCREEN_REPLAY and sub == 2:
            break
        time.sleep(0.05)
    else:
        raise TimeoutError("replay stage select did not appear")

    _move_cursor(
        session,
        stage,
        lambda: session.read_state(with_bullets=False).title_cursor,
        7,
    )
    tap(session, "z", settle_s=0.25)

    deadline = time.perf_counter() + 10.0
    while time.perf_counter() < deadline:
        screen, sub, _count = replay_state()
        if screen == spec.SCREEN_REPLAY and sub == 3:
            break
        time.sleep(0.05)
    else:
        raise TimeoutError("replay playback selection did not appear")

    tap(session, "z", settle_s=0.3)

    deadline = time.perf_counter() + 30.0
    while time.perf_counter() < deadline:
        state = session.read_state(with_bullets=False)
        if state.supervisor_state == spec.STATE_GAME_MANAGER and state.is_replay:
            session.log_line(f"replay playback started stage={state.stage} replay_mode={state.replay_mode}")
            return
        time.sleep(0.05)
    raise TimeoutError("replay playback did not start")


def exit_run_to_title(session: GameSession, timeout: float = 15.0) -> None:
    """Quit the current run through the pause menu (used between episodes)."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        state = session.read_state(with_bullets=False)
        if state.supervisor_state == spec.STATE_TITLE_SCREEN:
            return
        if state.in_game_menu:
            tap(session, "down", settle_s=0.12)
            tap(session, "down", settle_s=0.12)
            tap(session, "down", settle_s=0.12)
            tap(session, "z", settle_s=0.5)
        else:
            tap(session, "esc", settle_s=0.3)
    raise TimeoutError("could not exit the run to the title screen")
