"""Gymnasium environment over a live TH08 process (contract §6).

The environment owns one game session and steps it in *game frames*: each
``step`` writes the action's key set, waits until ``gameplayFrameCounter`` has
advanced by ``frameskip`` frames, then samples state and bullet pool again.  A
decision therefore costs one action plus ``frameskip`` simulated frames, which
is the only supported notion of time (wall-clock pacing is not used anywhere).

Measured cost per step on this host (RTX 4070 SUPER machine, 2026-10-18):
scalar read 0.02 ms, bullet pool 0.54 ms, frame capture 0.68 ms, so the loop is
bounded by the game's 60 Hz simulation, not by the sampler.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:  # gymnasium is optional at import time (dataset tools must work without it)
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = object  # type: ignore
    spaces = None  # type: ignore

from . import menu, obs, spec, winapi
from .obs import Observation
from .reward import RewardEngine, RewardWeights
from .session import GameSession, SessionOptions, prepare_run_dir
from .state import GameState

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EnvConfig:
    template_dir: Path = ROOT / "third_party/th08/build/play"
    run_dir: Path = ROOT / "third_party/th08/build/rl/env-session"
    team: int = 0
    difficulty: int = spec.LUNATIC
    practice_stage: int | None = None
    frameskip: int = 2
    max_steps: int = 9000
    reset_mode: str = "menu"  # "menu" keeps the process, "relaunch" restarts it
    death_penalty_lives: float = 0.0
    reward: RewardWeights = field(default_factory=RewardWeights)
    windowed: bool = True
    # Expose the full bullet snapshot (absolute coordinates) in ``info`` so
    # scripted policies / teachers can plan from every active bullet instead of
    # the 64 nearest ones the network observation keeps.
    expose_raw_bullets: bool = False
    # Expose live boss positions in ``info["bosses"]`` (shape (n, 2)); the pilot
    # uses them to stay under the boss so that stages actually progress.
    expose_bosses: bool = False

    def validate(self) -> None:
        if self.frameskip < 1:
            raise ValueError("frameskip must be >= 1")
        if self.reset_mode not in ("menu", "relaunch"):
            raise ValueError(f"unknown reset_mode {self.reset_mode}")
        if self.practice_stage is not None and not 0 <= self.practice_stage <= 6:
            raise ValueError("practice_stage must be in 0..6")


class Th08Env(gym.Env):  # type: ignore[misc]
    """One live TH08 instance as a Gymnasium environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, config: EnvConfig | None = None):
        if spaces is None:  # pragma: no cover
            raise ImportError("gymnasium is required to use Th08Env")
        self.config = config or EnvConfig()
        self.config.validate()
        self.action_space = spaces.MultiDiscrete(list(spec.ACTION_DIMS))
        self.observation_space = spaces.Dict(
            {
                "self_feat": spaces.Box(-10.0, 10.0, shape=(obs.N_SELF,), dtype=np.float32),
                "bullets": spaces.Box(
                    -10.0, 10.0, shape=(obs.N_BULLETS, obs.N_BULLET_FEATURES), dtype=np.float32
                ),
                "danger": spaces.Box(
                    0.0, 255.0, shape=(obs.DANGER_ROWS, obs.DANGER_COLS), dtype=np.float32
                ),
            }
        )
        self.session: GameSession | None = None
        self.reward_engine = RewardEngine(weights=self.config.reward)
        self.last_state: GameState | None = None
        self._step_index = 0
        self._episode_reward = 0.0
        self.episode_stats: dict = {}
        self._last_observation: Observation = obs.empty_observation()
        self._last_bullets = None
        self._last_frame = 0
        self._last_stage = 0
        self._deaths = 0
        self._start_score = 0
        self._start_graze = 0

    # -- session management ----------------------------------------------
    def _ensure_session(self) -> GameSession:
        if self.session is not None:
            alive = self.session.process is not None and self.session.process.poll() is None
            if alive and self.session.mem is not None and self.session.mem.is_alive():
                return self.session
            try:
                self.session.close(timeout=3.0)
            except Exception:
                pass
            self.session = None
        prepare_run_dir(self.config.run_dir, self.config.template_dir, windowed=self.config.windowed)
        self.session = GameSession(SessionOptions(run_dir=self.config.run_dir, log_to_file=True))
        self.session.start()
        return self.session

    def _capture_observation(self) -> tuple[Observation, GameState]:
        assert self.session is not None and self.session.reader is not None
        state = self.session.reader.read_state(with_bullets=True, with_title=False)
        field_snapshot = self.session.reader.read_bullets()
        self._last_bullets = field_snapshot
        return obs.encode(state, field_snapshot), state

    def _raw_bullet_dict(self) -> dict[str, np.ndarray]:
        """All active bullets in absolute playfield coordinates."""
        assert self._last_bullets is not None
        field = self._last_bullets
        mask = field.active_mask()
        half = field.half_size[mask]
        return {
            "position": np.array(field.position[mask], dtype=np.float32),
            "velocity": np.array(field.velocity[mask], dtype=np.float32),
            "half": np.maximum(np.asarray(half, dtype=np.float32).max(axis=1, initial=0.5), 0.5),
        }

    # -- gym API ----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        session = self._ensure_session()
        if self.config.reset_mode == "relaunch" or self._step_index == 0:
            # Guarantee a clean stage entry: menu -> title -> start.
            if self._step_index > 0:
                session.close(timeout=6.0)
                self.session = None
                session = self._ensure_session()
        try:
            menu.ensure_title(session)
        except Exception:
            session.close(timeout=3.0)
            self.session = None
            session = self._ensure_session()
            menu.ensure_title(session)
        menu.enter_game(
            session,
            difficulty=self.config.difficulty,
            team=self.config.team,
            practice_stage=self.config.practice_stage,
        )
        # Wait until the stage is actually running before the first sample.
        session.wait_for(
            lambda s: s.is_playing() and s.gameplay_frame > 1,
            timeout=30.0,
            description="first gameplay frame",
        )
        observation, state = self._capture_observation()
        self._last_observation = observation
        self.last_state = state
        self.reward_engine.reset(state)
        self._step_index = 0
        self._episode_reward = 0.0
        self._last_frame = state.gameplay_frame
        self._last_stage = state.stage
        self._deaths = state.deaths
        self._start_score = state.score
        self._start_graze = state.graze
        self.episode_stats = {
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "difficulty": self.config.difficulty,
            "team": self.config.team,
            "practice_stage": self.config.practice_stage,
        }
        return self._observation_dict(observation), {"state": self._state_dict(state)}

    def _observation_dict(self, observation: Observation) -> dict[str, np.ndarray]:
        return {
            "self_feat": observation.self_feat.astype(np.float32),
            "bullets": observation.bullets.astype(np.float32),
            "danger": observation.danger_map.astype(np.float32),
        }

    @staticmethod
    def _state_dict(state: GameState) -> dict:
        return {
            "stage": state.stage,
            "frame": state.gameplay_frame,
            "lives": state.lives,
            "bombs": state.bombs,
            "power": state.power,
            "score": state.score,
            "graze": state.graze,
            "deaths": state.deaths,
            "bullets": state.active_bullets,
        }

    def _wait_frames(self, frames: int, timeout: float = 0.5) -> GameState:
        assert self.session is not None and self.session.reader is not None
        reader = self.session.reader
        deadline = time.perf_counter() + timeout
        state = reader.read_state(with_bullets=False, with_title=False)
        while time.perf_counter() < deadline:
            state = reader.read_state(with_bullets=False, with_title=False)
            if state.supervisor_state != spec.STATE_GAME_MANAGER:
                return state
            if state.gameplay_frame != self._last_frame:
                if ((state.gameplay_frame - self._last_frame) & 0xFFFFFFFF) >= frames:
                    return state
            time.sleep(0)
        return state

    def _focus_guard(self) -> None:
        """Keep the game focused: TH08 only reads DirectInput while foreground."""
        assert self.session is not None
        if self._step_index % 15 == 0:
            hwnd = self.session.refresh_window()
            if hwnd and winapi.user32.GetForegroundWindow() != hwnd:
                winapi.force_foreground(hwnd)

    def step(self, action):
        assert self.session is not None and self.session.keyboard is not None
        self._focus_guard()
        move, shoot, focus, bomb = (int(v) for v in action)
        buttons = spec.action_to_buttons((move, shoot, focus, bomb))
        keys = set()
        for bit, name in (
            (spec.BTN_UP, "up"),
            (spec.BTN_DOWN, "down"),
            (spec.BTN_LEFT, "left"),
            (spec.BTN_RIGHT, "right"),
            (spec.BTN_SHOOT, "z"),
            (spec.BTN_FOCUS, "shift"),
            (spec.BTN_BOMB, "x"),
        ):
            if buttons & bit:
                keys.add(name)
        self.session.keyboard.set_keys(keys)

        state = self._wait_frames(self.config.frameskip)
        terminated = False
        truncated = False
        reason = ""
        advanced = (state.gameplay_frame - self._last_frame) & 0xFFFFFFFF
        if state.supervisor_state != spec.STATE_GAME_MANAGER:
            terminated = True
            reason = f"left game (supervisor_state={state.supervisor_state})"
        elif state.is_demo:
            terminated = True
            reason = "attract demo"
        elif state.lives < 0:
            terminated = True
            reason = "game over"
        elif advanced == 0:
            # A game-over / continue screen keeps supervisor_state == 2 but stops
            # advancing gameplayFrameCounter; without this check the loop spins on
            # a timeout per step (measured 0.3-1.0 s/step at the end of a run).
            terminated = True
            reason = "frame stalled"
        observation, full_state = self._capture_observation()
        if full_state.stage != self._last_stage and full_state.stage > self._last_stage:
            self._last_stage = full_state.stage
        vector, info = self.reward_engine.compute(full_state, terminal=terminated)
        reward = self.reward_engine.scalar(vector)
        self._episode_reward += reward
        self._step_index += 1
        self._last_frame = full_state.gameplay_frame
        self.last_state = full_state
        if self._step_index >= self.config.max_steps:
            truncated = True
        info.update(
            {
                "reward_vector": vector,
                "scalar_reward": reward,
                "episode_reward": self._episode_reward,
                "step": self._step_index,
                "frames_advanced": advanced,
                **self._state_dict(full_state),
            }
        )
        if self.config.expose_raw_bullets and self._last_bullets is not None:
            info["bullets_raw"] = self._raw_bullet_dict()
        if self.config.expose_bosses:
            assert self.session is not None and self.session.reader is not None
            try:
                info["bosses"] = self.session.reader.read_boss_positions()
            except OSError:
                info["bosses"] = np.zeros((0, 2), dtype=np.float32)
        if terminated or truncated:
            info["termination_reason"] = reason or ("max_steps" if truncated else "")
            info["episode"] = {
                **self.episode_stats,
                "steps": self._step_index,
                "frames": full_state.run_frames,
                "stages_reached": full_state.stage,
                "score": full_state.score - self._start_score,
                "graze": full_state.graze - self._start_graze,
                "deaths": full_state.deaths - self._deaths,
                "lives": full_state.lives,
                "episode_reward": self._episode_reward,
                "terminated": terminated,
                "truncated": truncated,
                "termination_reason": info["termination_reason"],
            }
        return self._observation_dict(observation), reward, terminated, truncated, info

    def render(self):
        assert self.session is not None
        return self.session.grab()

    def close(self):
        if self.session is not None:
            try:
                self.session.keyboard.release_all()
            except Exception:
                pass
            self.session.close()
            self.session = None
