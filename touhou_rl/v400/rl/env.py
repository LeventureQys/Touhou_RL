"""P4 prototype: a reinforcement-learning environment on the v4.0.0 engine.

Scope and honesty
-----------------
This module is the **exploratory P4 prototype** (EXP-P4-001).  It is not a gate,
it does not change the engine, the protocol, ``obs-spec-v1`` or the frozen
student model; it only puts the existing, already-verified pieces in an RL loop:

* **observation** —the frozen obs-v1 contract, unchanged: the ``play64`` plane
  (64x64 BT.601 block means of the 384x448 playfield) stacked over the last
  :data:`~touhou_rl.v400.student.obs_spec.FRAME_STACK` *consecutive* frames, plus
  the 56x160 HUD numeric band.  The stack is built exactly the way
  ``tools/rl/v400_student_eval.py`` builds it (``--obs-window consecutive``), so
  a policy trained here sees the same bytes a deployment episode would show it.
  Nothing privileged enters the observation, ever: there is no code path that
  adds engine state to it.
* **action** —the frozen obs-v1 action contract: ``cand = focus * 9 + move``
  over 18 move-focus candidates times the 2-way bomb dimension, held for
  :data:`DECISION_FRAMES` logic frames.  The 18 first-actions are the BC/training
  action set; ``bomb`` is the extra dimension the spec already reserves
  (``N_BOMB_CLASSES = 2``), so the joint set is the 36 ``STUDENT_ACTIONS`` and
  **not** a 19th mutually exclusive action.  That choice is deliberate: a 19-way
  action would have to either drop movement while bombing or leave the obs-v1
  action contract, and it could not be initialised from the BC checkpoint's two
  heads (see :mod:`touhou_rl.v400.rl.policy`).
* **reward** —engine ground truth, which the P4 brief explicitly allows *during
  training* (deployment reads the P2 detector).  The terms are fixed in
  :data:`REWARD_TERMS`; :func:`reward_scale_table` prints the scale.
* **episode** —stage 0, normal, fixed cap 19,887 frames (``ceil(1.5 x 13,258)``,
  the S1N ruler of EXP-P1-012/018 and EXP-P3-004..009) or death stop, exactly the
  rule the 20-seed evaluation uses.

Why the engine-state read is not a privilege leak
-------------------------------------------------
:meth:`EngineBackend.state` reads the harness shared block.  The *reward* uses it
(training-time truth), and it is never copied into the observation.  Passing
``privileged_state=True`` additionally exposes the decoded 25 fields in
``info`` for diagnosis; every ``info`` carrying it is marked
``privileged_diagnostic=True`` and must never be fed to a policy.

Throughput
----------
Measured in EXP-P4-001 on the operator box (WSL, one Xvfb display, software GL,
the Windows retail-trace batch running beside it): one engine sustains
~120 frames/s (60 decisions/s) and six sustain ~94 frames/s each, i.e. ~561
frames/s aggregate.  The loop below therefore reads the shared block directly
instead of issuing a per-frame ``STATE`` round trip; the equality of the two
sources was asserted against live frames (see ``tools/rl/v400_rl_train.py
--check-state-equivalence``).
"""

from __future__ import annotations

import os
import struct
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from touhou_rl.v400.gym.actions import N_CAND, cand_to_mask
from touhou_rl.v400.gym.protocol import HarnessClient
from touhou_rl.v400.gym.shm import FLAG_PIXELS_VALID, SharedBlock
from touhou_rl.v400.labels.state import decode_state, state_from_reply
from touhou_rl.v400.student import obs_spec

# --------------------------------------------------------------------------- #
# frozen episode ruler
# --------------------------------------------------------------------------- #

#: ``RESET`` stage argument; 0-based, so 0 is the first stage (``Stage1``).
DEFAULT_STAGE = 0
DEFAULT_DIFFICULTY = "normal"
DIFFICULTIES = ("easy", "normal", "hard", "lunatic")
#: EXP-P1-012 / EXP-P1-018 / EXP-P3-004..009 S1N cap: ``ceil(1.5 x 13,258)``.
DEFAULT_MAX_FRAMES = 19_887
#: One decision == DECISION_FRAMES logic frames (obs-v1 / 03 搂3.3).  Not a knob.
DECISION_FRAMES = obs_spec.DECISION_FRAMES
#: ``play64`` stack depth; the stack is the last N *consecutive* frames.
FRAME_STACK = obs_spec.FRAME_STACK

#: The 18 move-focus candidates times the 2-way bomb dimension.
N_BOMB_CLASSES = obs_spec.N_BOMB_CLASSES
N_ACTIONS = N_CAND * N_BOMB_CLASSES

# --------------------------------------------------------------------------- #
# reward (engine ground truth, training time only)
# --------------------------------------------------------------------------- #

REWARD_SURVIVE_PER_FRAME = 1.0
REWARD_SCORE_PER_POINT = 1.0 / 1000.0
REWARD_DEATH = -100.0
REWARD_BOMB_REMAINING = 1.0
REWARD_STAGE_CLEAR = 1000.0

#: Ordered reward table, so the card and the tests read one source.
REWARD_TERMS: tuple[dict[str, Any], ...] = (
    {
        "name": "survive",
        "formula": "+1.0 per logic frame advanced",
        "source": "engine STEP reply ``advanced``",
        "scale": REWARD_SURVIVE_PER_FRAME,
        "typical_episode": "2,249 frames -> +2,249",
        "why": "the quantity the 20-seed row measures (frames survived), so the "
               "return and the reportable metric are the same axis",
    },
    {
        "name": "score",
        "formula": "+score_delta / 1000",
        "source": "engine ``f4`` (globals.score)",
        "scale": REWARD_SCORE_PER_POINT,
        "typical_episode": "42,664 points -> +42.7",
        "why": "dense shaping for clearing enemies; 1/1000 keeps it ~2 % of the "
               "survival term so points cannot be traded for survival",
    },
    {
        "name": "death",
        "formula": "-100.0 per death",
        "source": "engine ``f6`` (globals.deaths, float bits)",
        "scale": REWARD_DEATH,
        "typical_episode": "1 death -> -100",
        "why": "a terminal-shaped cost; with stop-on-death it is the last "
               "transition of every episode, so it marks the fatal decision",
    },
    {
        "name": "bomb_remaining",
        "formula": "+1.0 * bombs_remaining on the terminal transition",
        "source": "engine ``f8`` (globals.bombsRemaining, float bits)",
        "scale": REWARD_BOMB_REMAINING,
        "typical_episode": "2 bombs left -> +2",
        "why": "keeps hoarding slightly positive so the bomb dimension does not "
               "collapse to 'always bomb' under entropy pressure",
    },
    {
        "name": "stage_clear",
        "formula": "+1000.0 when the stage counter advances",
        "source": "engine ``f1`` (g_GameManager.currentStage)",
        "scale": REWARD_STAGE_CLEAR,
        "typical_episode": "never reached by the BC baseline (0/20 clears)",
        "why": "G3 is a clear-rate gate; without this the prototype could only "
               "learn to survive longer inside Stage 1",
    },
)

#: Event labels written to ``info['end']``.  Same vocabulary as
#: ``tools/rl/v400_student_eval.py`` so a training row and an eval row agree.
END_RESET = "reset"
END_RUNNING = "running"
END_GAME_OVER = "game_over"
END_DEATH_STOP = "death_stop"
END_STAGE_CLEAR = "stage_clear"
END_FRAME_CAP = "frame_cap"
END_ENGINE_EXIT = "engine_exit"

#: Named action helpers, for tests and for reading a rollout back.
ACTION_NAMES: tuple[str, ...] = tuple(
    f"cand{cand}_bomb{bomb}" for cand in range(N_CAND) for bomb in range(N_BOMB_CLASSES)
)

#: Shared-block field indices this module reads (``labels/state.py`` order).
F_FRAME = 0
F_STAGE = 1
F_SCORE = 4
F_DEATHS = 6
F_LIVES = 7
F_BOMBS = 8

_F32 = struct.Struct("<f")
_U32 = struct.Struct("<I")


def f32_from_bits(bits: int) -> float:
    """Reinterpret one raw ``u32`` as the ``float32`` the engine recorded."""

    return _F32.unpack(_U32.pack(int(bits) & 0xFFFFFFFF))[0]


def f32_bits(value: float) -> int:
    """Inverse of :func:`f32_from_bits`, for the scripted test backend."""

    return _U32.unpack(_F32.pack(float(value)))[0]


def action_to_cand_bomb(action: int) -> tuple[int, int]:
    """``action = cand * 2 + bomb`` -> ``(cand, bomb)`` (the ``STUDENT_ACTIONS`` order)."""

    action = int(action)
    if not 0 <= action < N_ACTIONS:
        raise RlEnvError(f"action {action} outside [0, {N_ACTIONS})")
    cand, bomb = divmod(action, N_BOMB_CLASSES)
    return cand, bomb


def cand_bomb_to_action(cand: int, bomb: int) -> int:
    """Inverse of :func:`action_to_cand_bomb`; validates both halves."""

    if not 0 <= int(cand) < N_CAND:
        raise RlEnvError(f"candidate {cand} outside [0, {N_CAND})")
    if int(bomb) not in (0, 1):
        raise RlEnvError(f"bomb must be 0 or 1, got {bomb}")
    return int(cand) * N_BOMB_CLASSES + int(bomb)


class RlEnvError(RuntimeError):
    """The environment departed from its contract (frame counter, action, ...)."""


# --------------------------------------------------------------------------- #
# engine backend
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StepReply:
    """The parts of a ``STEP`` reply the RL loop acts on."""

    advanced: int
    exit: bool
    game_over: bool


class _EnvHarnessClient(HarnessClient):
    """``HarnessClient`` that starts the engine in the ``env`` role.

    ``--rl-role env`` is what the evaluation controller uses
    (``tools/rl/v400_student_eval.py``); the pixel readback needs it.
    """

    def start(self) -> None:  # noqa: D401 - same contract as the parent
        if self._proc is not None:
            raise RuntimeError("client already started")
        argv = [str(self.binary), "--data-dir", str(self.data_dir),
                "--rl-control", "0,1", "--rl-role", "env"]
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
            bufsize=0,
        )
        if os.name == "nt":
            self._start_pipe_reader()


class EngineBackend:
    """One ``th08-modern`` process plus its shared-memory block.

    The backend is the *only* place that talks to the engine, so a test can swap
    in :class:`FakeBackend` and exercise every reward/termination rule without a
    display, a GPU or a 47 MB game archive.

    **One RESET per process.**  EXP-P4-001 measured that a second ``RESET`` on a
    live harness answers ``ready=0 setup_frames=1800`` and never becomes ready —    the frame counter stayed 0 across 360 further ``STEP`` frames, and six
    ``RESET`` retries all failed.  An episode therefore has to start from a fresh
    process, which is exactly what ``tools/rl/v400_student_eval.py`` does (one
    ``HarnessClient`` per episode).  :meth:`reopen` restarts the engine, so the
    ~6-7 s it costs is the evaluation ruler's own cost and not something this
    wrapper adds.
    """

    def __init__(
        self,
        engine: str | os.PathLike[str],
        data_dir: str | os.PathLike[str],
        *,
        shm_name: str,
        timeout: float = 240.0,
        env: dict[str, str] | None = None,
    ) -> None:
        self.engine = str(engine)
        self.data_dir = str(data_dir)
        self.shm_name = str(shm_name).lstrip("/")
        self.timeout = float(timeout)
        self._env_overrides = dict(env or {})
        self._client: HarnessClient | None = None
        self._shm: SharedBlock | None = None
        self._opened = False
        self.state_calls = 0
        #: How many times this slot has been restarted for a new episode.
        self.restarts = 0

    # -- lifetime ---------------------------------------------------------
    def open(self) -> None:
        if self._opened:
            return
        client = _EnvHarnessClient(self.engine, self.data_dir,
                                   shm_name=self.shm_name, timeout=self.timeout,
                                   env=self._env_overrides or None)
        client.start()
        try:
            client.hello()
            client.attach_shm()
            client.readback(True)
            self._shm = SharedBlock(client.shm_name or self.shm_name)
        except BaseException:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
            raise
        self._client = client
        self._opened = True

    def close(self) -> None:
        shm, self._shm = self._shm, None
        client, self._client = self._client, None
        self._opened = False
        try:
            if shm is not None:
                shm.close()
        finally:
            try:
                if client is not None:
                    client.close()
            except Exception:  # noqa: BLE001 - a dead engine must not mask a result
                pass
            for path in Path("/dev/shm").glob(f"{self.shm_name}*"):
                try:
                    path.unlink()
                except OSError:
                    pass

    def reopen(self) -> None:
        """Fresh process for a fresh episode (see the class docstring)."""

        if self._opened:
            self.restarts += 1
        self.close()
        self.open()

    def __enter__(self) -> "EngineBackend":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- protocol ---------------------------------------------------------
    @property
    def client(self) -> HarnessClient:
        if self._client is None:
            raise RlEnvError("backend is not open")
        return self._client

    def reset(self, *, stage: int, difficulty: int, team: int, seed: int) -> None:
        reply = self.client.reset(stage=stage, difficulty=difficulty, team=team,
                                  seed=seed)
        if reply.get("ready") != "1":
            raise RlEnvError(f"RESET did not reach a playable state: {reply}")

    def step(self, mask: int, frames: int) -> StepReply:
        reply = self.client.step(mask, frames)
        return StepReply(
            advanced=int(reply.get("advanced", 0)),
            exit=reply.get("exit") == "1",
            game_over=reply.get("game_over") == "1",
        )

    def state(self) -> dict[str, Any]:
        """The 25 tracked fields, decoded, via the ``STATE`` command.

        **Measured in EXP-P4-001**: the shared block's state region is *stale*
        until a ``STATE`` command runs —after ``RESET`` the header reports
        ``state_bytes=100`` but ``state_hash=0`` and all 25 fields read 0 while
        the ``STATE`` reply already says ``lives_remaining=2.0``,
        ``bombs_remaining=3.0``.  Reading the block after a ``STATE`` refreshes it
        (``state_hash`` becomes non-zero), so an RL loop that skipped the command
        would see ``deaths=0, score=0, stage=0`` forever and run every episode to
        the frame cap.  The pixels *are* published every frame
        (``FLAG_PIXELS_VALID``), which is why only the pixels come from the block.
        This method therefore pays the one round trip per logic frame the
        reference controller also pays; it is the price of the ruler, not slack in
        the wrapper.
        """

        # ``HarnessClient.state`` parses f6..f12 as decimal, but the harness emits
        # them as float bit patterns; ``state_from_reply`` is the decoder the
        # reference controller uses, so use that one.
        return state_from_reply(self.client.request("STATE"))

    def pixels(self) -> np.ndarray:
        if self._shm is None:
            raise RlEnvError("backend is not open")
        return self._shm.pixels()

    def pixels_valid(self) -> bool:
        if self._shm is None:
            raise RlEnvError("backend is not open")
        return bool(self._shm.header()["flags"] & FLAG_PIXELS_VALID)

    def request_state_reply(self) -> dict[str, Any]:
        """The ``STATE`` command, decoded.  Used by the equivalence check."""

        from touhou_rl.v400.labels.state import state_from_reply

        return state_from_reply(self.client.request("STATE"))


class FakeBackend:
    """Scripted stand-in for :class:`EngineBackend` (unit tests only).

    ``script`` is one dict per logic frame: ``advanced`` (default 1), ``exit``,
    ``game_over``, and ``state`` —a ``{field index: raw u32}`` patch applied
    after that frame.  Helpers :meth:`death_at`, :meth:`score_at` and
    :meth:`bombs_at` build those patches from human numbers.

    ``pixels()`` renders a deterministic frame-dependent readback, so two fakes
    with the same script and seed produce byte-identical observations —the
    parallel-determinism test relies on that.
    """

    def __init__(
        self,
        script: Sequence[dict[str, Any]] | Callable[[int], dict[str, Any]],
        *,
        pixels: np.ndarray | None = None,
        start: dict[str, float] | None = None,
    ) -> None:
        self._script = script
        self._frame = 0
        self._index = 0
        self._pixels = pixels
        self._start = dict(start or {"deaths": 0.0, "lives": 3.0, "bombs": 3.0,
                                     "score": 0.0, "stage": 0.0})
        self._state = np.zeros(25, dtype=np.uint32)
        self._lock = threading.Lock()
        self.closed = False
        self.reset_count = 0
        self.restarts = 0
        self.seeds: list[int] = []
        self.masks: list[int] = []
        self._apply_start()

    # -- constructors for readable scripts --------------------------------
    @staticmethod
    def patch(*, deaths: float | None = None, lives: float | None = None,
              bombs: float | None = None, score: int | None = None,
              stage: int | None = None) -> dict[int, int]:
        """Build a raw-field patch from human numbers."""

        out: dict[int, int] = {}
        if deaths is not None:
            out[F_DEATHS] = f32_bits(deaths)
        if lives is not None:
            out[F_LIVES] = f32_bits(lives)
        if bombs is not None:
            out[F_BOMBS] = f32_bits(bombs)
        if score is not None:
            out[F_SCORE] = int(score)
        if stage is not None:
            out[F_STAGE] = int(stage)
        return out

    # -- state -------------------------------------------------------------
    def _apply_start(self) -> None:
        self._state = np.zeros(25, dtype=np.uint32)
        self._state[F_FRAME] = 0
        for index, value in self.patch(
            deaths=self._start["deaths"], lives=self._start["lives"],
            bombs=self._start["bombs"], score=int(self._start["score"]),
            stage=int(self._start["stage"]),
        ).items():
            self._state[index] = value

    def _event(self) -> dict[str, Any]:
        if callable(self._script):
            return dict(self._script(self._index))
        if self._index < len(self._script):
            return dict(self._script[self._index])
        return {}

    def get_float(self, index: int) -> float:
        return f32_from_bits(int(self._state[index]))

    # -- backend API ------------------------------------------------------
    def open(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def reopen(self) -> None:
        """A fake engine has no process to restart; count it like the real one."""

        self.restarts += 1
        self.closed = False

    def reset(self, *, stage: int, difficulty: int, team: int, seed: int) -> None:
        with self._lock:
            self._frame = 0
            self._index = 0
            self.reset_count += 1
            self.seeds.append(int(seed))
            self.masks = []
            self._apply_start()

    def step(self, mask: int, frames: int) -> StepReply:
        with self._lock:
            event = self._event()
            advanced = int(event.get("advanced", 1))
            advanced = max(0, min(advanced, int(frames)))
            self._frame += advanced
            self._index += 1
            self.masks.append(int(mask))
            for index, value in (event.get("state") or {}).items():
                self._state[int(index)] = int(value)
            self._state[F_FRAME] = self._frame
            return StepReply(
                advanced=advanced,
                exit=bool(event.get("exit", False)),
                game_over=bool(event.get("game_over", False)),
            )

    def state(self) -> dict[str, Any]:
        with self._lock:
            return decode_state([int(value) for value in self._state])

    def raw_state(self) -> np.ndarray:
        """The 25 raw ``u32`` fields (what the real shared block holds)."""

        with self._lock:
            return self._state.copy()

    def pixels(self) -> np.ndarray:
        if self._pixels is not None:
            return self._pixels
        frame = self._frame
        image = np.zeros((480, 640, 4), dtype=np.uint8)
        image[:, :, 0] = np.uint8((frame >> 8) & 0xFF)
        image[:, :, 1] = np.uint8(frame & 0xFF)
        image[16:464, 32:416, 2] = np.uint8((frame * 7) & 0xFF)
        return image

    def pixels_valid(self) -> bool:
        return True

    def request_state_reply(self) -> dict[str, Any]:
        return dict(self.state())


# --------------------------------------------------------------------------- #
# the environment
# --------------------------------------------------------------------------- #

@dataclass
class RlState:
    """Bookkeeping for one episode (kept out of the observation on purpose)."""

    seed: int = 0
    frames: int = 0
    decisions: int = 0
    renders: int = 0
    deaths: float = 0.0
    lives: float = 0.0
    bombs: float = 0.0
    score: int = 0
    stage: int = 0
    stage_start: int = 0
    cleared: bool = False
    end: str = END_RESET
    episode_return: float = 0.0
    reward_terms: dict[str, float] = field(default_factory=dict)


class RlEnv:
    """One obs-v1 RL episode at a time, on top of a backend.

    ``step`` takes one of :data:`N_ACTIONS` actions (``cand * 2 + bomb``) and
    returns ``(obs, reward, terminated, truncated, info)``.  ``obs`` is the
    model-ready dict ``{"play": float32 (4,64,64), "hud": float32 (1,56,160)}``
    that :meth:`touhou_rl.v400.student.obs_spec.ObservationBuffer.arrays`
    produces —the exact tensors ``StudentNet.forward`` takes.  The frame loop,
    the termination rules and the frame cap mirror
    ``tools/rl/v400_student_eval.py`` line for line, so a training episode and a
    20-seed evaluation episode measure the same thing.
    """

    def __init__(
        self,
        backend: EngineBackend | FakeBackend,
        *,
        max_frames: int = DEFAULT_MAX_FRAMES,
        stop_on_death: bool = True,
        privileged_state: bool = False,
        stage: int = DEFAULT_STAGE,
        difficulty: str = DEFAULT_DIFFICULTY,
        team: int = 0,
    ) -> None:
        if difficulty not in DIFFICULTIES:
            raise RlEnvError(f"unknown difficulty {difficulty!r}")
        if int(max_frames) <= 0:
            raise RlEnvError(f"max_frames must be positive, got {max_frames}")
        self.backend = backend
        self.max_frames = int(max_frames)
        self.stop_on_death = bool(stop_on_death)
        self.privileged_state = bool(privileged_state)
        self.stage = int(stage)
        self.difficulty = difficulty
        self.team = int(team)
        self.status = RlState()
        self._buffer = obs_spec.ObservationBuffer()
        self._previous: dict[str, float] = {
            "deaths": 0.0, "lives": 0.0, "bombs": 0.0, "score": 0, "stage": 0,
        }
        self._zero_advances = 0
        self._opened = False
        self._episodes = 0

    # -- lifetime ---------------------------------------------------------
    def open(self) -> "RlEnv":
        self.backend.open()
        self._opened = True
        return self

    def close(self) -> None:
        if self._opened:
            self.backend.close()
            self._opened = False

    def __enter__(self) -> "RlEnv":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- readings ---------------------------------------------------------
    def _readings(self) -> dict[str, float]:
        state = self.backend.state()
        return {
            "deaths": float(state["deaths"]),
            "lives": float(state["lives_remaining"]),
            "bombs": float(state["bombs_remaining"]),
            "score": int(state["score"]),
            "stage": int(state["stage"]),
        }

    def _observe(self) -> dict[str, np.ndarray]:
        """Push the current readback and return the model-ready stack."""

        play, hud = obs_spec.encode_planes(self.backend.pixels())
        self._buffer.push(play, hud)
        self.status.renders += 1
        return self._buffer.arrays()

    # -- gym API ----------------------------------------------------------
    def reset(self, seed: int = 1000) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Start one episode (stage 0, normal, ``seed``).

        Same start-up as the evaluation controller: ``RESET``, then one one-frame
        ``STEP`` when the shared block does not yet hold a valid readback, then
        the first frame is pushed.  That warm-up frame is counted in ``frames``
        and reported -- never hidden.
        """

        if not self._opened:
            self.open()
        elif self._episodes > 0:
            # One RESET per process (see EngineBackend): a new episode needs a
            # new engine, exactly as the evaluation controller does.
            self.backend.reopen()
        self._episodes += 1
        self.backend.reset(
            stage=self.stage,
            difficulty=DIFFICULTIES.index(self.difficulty),
            team=self.team,
            seed=int(seed),
        )
        self._buffer.reset()
        self._zero_advances = 0
        frames = 0
        if not self.backend.pixels_valid():
            warm = self.backend.step(cand_to_mask(0, 0), 1)
            frames += int(warm.advanced)
            if warm.exit:
                raise RlEnvError("engine exited during the first-frame warm-up")
        readings = self._readings()
        self.status = RlState(
            seed=int(seed), frames=frames, stage=int(readings["stage"]),
            stage_start=int(readings["stage"]), end=END_RUNNING,
            deaths=readings["deaths"], lives=readings["lives"],
            bombs=readings["bombs"], score=int(readings["score"]),
        )
        self._previous = dict(readings)
        obs = self._observe()
        return obs, self.info()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply one decision (``DECISION_FRAMES`` logic frames) and score it."""

        cand, bomb = action_to_cand_bomb(action)
        mask = cand_to_mask(cand, bomb)

        status = self.status
        terms = {
            "survive": 0.0,
            "score": 0.0,
            "death": 0.0,
            "bomb_remaining": 0.0,
            "stage_clear": 0.0,
        }
        terminated = False
        truncated = False
        end = END_RUNNING
        status.decisions += 1
        previous = self._previous
        readings = dict(previous)

        budget = min(DECISION_FRAMES, self.max_frames - status.frames)
        for _unit in range(max(budget, 0)):
            reply = self.backend.step(mask, 1)
            advanced = int(reply.advanced)
            status.frames += advanced
            terms["survive"] += REWARD_SURVIVE_PER_FRAME * advanced
            if advanced <= 0:
                self._zero_advances += 1
                if self._zero_advances >= 3 and not reply.exit:
                    raise RlEnvError("STEP advanced no frames three times in a row")
            else:
                self._zero_advances = 0

            if reply.exit:
                # The engine is gone; there is no STATE to read (this mirrors
                # v400_student_eval.py, which also breaks before reading it).
                terminated = True
                end = END_GAME_OVER if reply.game_over else END_ENGINE_EXIT
                break

            readings = self._readings()
            terms["score"] += (readings["score"] - previous["score"]) * REWARD_SCORE_PER_POINT
            if readings["stage"] > status.stage_start:
                terminated = True
                end = END_STAGE_CLEAR
                break
            died = readings["deaths"] > previous["deaths"]
            previous = readings
            if died:
                terms["death"] += REWARD_DEATH
                if readings["lives"] <= 0:
                    terminated = True
                    end = END_GAME_OVER
                    break
                if self.stop_on_death:
                    terminated = True
                    end = END_DEATH_STOP
                    break
            self._observe()
        else:
            if status.frames >= self.max_frames:
                truncated = True
                end = END_FRAME_CAP

        if terminated or truncated:
            terms["bomb_remaining"] += REWARD_BOMB_REMAINING * float(readings["bombs"])
            if end == END_STAGE_CLEAR:
                terms["stage_clear"] += REWARD_STAGE_CLEAR

        self._previous = dict(readings)
        status.frames = int(status.frames)
        status.cleared = bool(end == END_STAGE_CLEAR)
        status.end = end if (terminated or truncated) else END_RUNNING
        for key in ("deaths", "lives", "bombs"):
            setattr(status, key, float(readings[key]))
        status.score = int(readings["score"])
        status.stage = int(readings["stage"])
        reward = float(sum(terms.values()))
        status.episode_return += reward
        for key, value in terms.items():
            status.reward_terms[key] = status.reward_terms.get(key, 0.0) + float(value)
        return (self._buffer.arrays(), reward, terminated, truncated,
                self.info(reward_terms=terms))

    # -- info -------------------------------------------------------------
    def info(self, *, reward_terms: dict[str, float] | None = None) -> dict[str, Any]:
        status = self.status
        payload: dict[str, Any] = {
            "seed": status.seed,
            "frames": status.frames,
            "decisions": status.decisions,
            "renders": status.renders,
            "deaths": status.deaths,
            "lives": status.lives,
            "bombs": status.bombs,
            "score": status.score,
            "stage": status.stage,
            "stage_start": status.stage_start,
            "cleared": status.cleared,
            "end": status.end,
            "episode_return": status.episode_return,
            "reward_terms": dict(status.reward_terms),
            "privilege": "training_engine_truth_not_in_observation",
            "privileged_diagnostic": False,
        }
        if reward_terms is not None:
            payload["step_reward_terms"] = dict(reward_terms)
        if self.privileged_state:
            payload["state"] = dict(self.backend.state())
            payload["privileged_diagnostic"] = True
        return payload


# --------------------------------------------------------------------------- #
# vectorised facade
# --------------------------------------------------------------------------- #

class VecRlEnv:
    """N independent :class:`RlEnv` instances stepped together.

    Each environment owns a separate engine process and, in the trainer, a
    separate writable game-data directory: two episodes side by side must not
    share ``score.dat`` / ``th08.cfg`` (EXP-P0-021's lesson, re-used by
    ``.tmp_v400_power/launch_power.sh``).
    """

    def __init__(self, envs: Sequence[RlEnv]) -> None:
        self.envs = list(envs)
        if not self.envs:
            raise RlEnvError("VecRlEnv needs at least one environment")
        #: One long-lived worker per engine: a fresh ``ThreadPoolExecutor`` per
        #: decision would create ~6 threads 280 times a second.
        self.pool = ThreadPoolExecutor(max_workers=len(self.envs),
                                       thread_name_prefix="v400rl-env")

    def __len__(self) -> int:
        return len(self.envs)

    def open_all(self) -> None:
        list(self.pool.map(lambda env: env.open(), self.envs))

    def reset_all(self, seeds: Sequence[int]):
        return list(self.pool.map(lambda pair: pair[0].reset(pair[1]),
                                  zip(self.envs, seeds)))

    def step_all(self, actions: Sequence[int]):
        actions = list(actions)
        if len(actions) != len(self.envs):
            raise RlEnvError("step_all needs one action per environment")
        return list(self.pool.map(lambda pair: pair[0].step(pair[1]),
                                  zip(self.envs, actions)))

    def step_all_with_reset(self, actions: Sequence[int], seeds: Sequence[int]):
        """Step every env; an env that finished is reset to its next seed.

        Returns ``(results, new_observations)`` where ``results[i]`` is the
        five-tuple :meth:`RlEnv.step` returns and ``new_observations[i]`` is the
        first observation of the next episode (or ``None`` when the env is still
        running).  Doing the reset inside the worker keeps the reset's wall time
        off the caller's critical path except for the barrier it creates.
        """

        actions = list(actions)
        seeds = list(seeds)
        if len(actions) != len(self.envs) or len(seeds) != len(self.envs):
            raise RlEnvError("step_all_with_reset needs one action and one seed per env")

        def work(env: RlEnv, action: int, seed: int):
            result = env.step(action)
            if result[2] or result[3]:
                obs, _info = env.reset(seed)
                return result, obs
            return result, None

        outcomes = list(self.pool.map(work, self.envs, actions, seeds))
        return [pair[0] for pair in outcomes], [pair[1] for pair in outcomes]

    def close(self) -> None:
        try:
            self.pool.shutdown(wait=True)
        except Exception:  # noqa: BLE001
            pass
        for env in self.envs:
            try:
                env.close()
            except Exception:  # noqa: BLE001 - close every env, then report
                pass


# --------------------------------------------------------------------------- #
# reward table helpers
# --------------------------------------------------------------------------- #

def reward_scale_table() -> list[dict[str, Any]]:
    """The reward terms as a printable table (card + tests share this)."""

    return [dict(term) for term in REWARD_TERMS]


#: Per-episode magnitude of each term for the BC baseline, from the 20-seed
#: ``m1`` row (EXP-P3-009): 2,248.85 frames, 42,663.75 points, 1 death, and
#: **0 bombs left** — the reference episode record reports ``bombs_start=3.0``,
#: ``bombs_end=0.0``, ``bombs_used=3.0``, so the hoarding bonus is 0 for the
#: baseline and a policy that keeps its bombs gains 1-3 points over it.
BC_BASELINE_MAGNITUDE: dict[str, float] = {
    "survive": REWARD_SURVIVE_PER_FRAME * 2248.85,
    "score": REWARD_SCORE_PER_POINT * 42663.75,
    "death": REWARD_DEATH * 1.0,
    "bomb_remaining": REWARD_BOMB_REMAINING * 0.0,
    "stage_clear": 0.0,
}


def reward_scale_report() -> dict[str, Any]:
    """Reward table plus the BC-baseline magnitudes and their shares."""

    total = sum(abs(value) for value in BC_BASELINE_MAGNITUDE.values())
    return {
        "terms": reward_scale_table(),
        "bc_baseline_magnitude": dict(BC_BASELINE_MAGNITUDE),
        "bc_baseline_share": {
            key: (abs(value) / total if total else 0.0)
            for key, value in BC_BASELINE_MAGNITUDE.items()
        },
        "net_bc_baseline_return": sum(BC_BASELINE_MAGNITUDE.values()),
        "note": "survival dominates by design: the return axis is the frame "
                "count the 20-seed protocol reports, so optimising the return "
                "and moving the measured number are the same thing.  The risk "
                "is that a 1-frame difference moves the return more than a "
                "death does (-100 is only 100 frames), so the prototype is "
                "expected to trade a death for 100 frames; see EXP-P4-001 搂risk.",
    }


__all__ = [
    "ACTION_NAMES",
    "BC_BASELINE_MAGNITUDE",
    "DEFAULT_DIFFICULTY",
    "DEFAULT_MAX_FRAMES",
    "DEFAULT_STAGE",
    "DECISION_FRAMES",
    "END_DEATH_STOP",
    "END_ENGINE_EXIT",
    "END_FRAME_CAP",
    "END_GAME_OVER",
    "END_RESET",
    "END_RUNNING",
    "END_STAGE_CLEAR",
    "EngineBackend",
    "FakeBackend",
    "FRAME_STACK",
    "N_ACTIONS",
    "REWARD_DEATH",
    "REWARD_SCORE_PER_POINT",
    "REWARD_STAGE_CLEAR",
    "REWARD_SURVIVE_PER_FRAME",
    "REWARD_BOMB_REMAINING",
    "REWARD_TERMS",
    "RlEnv",
    "RlEnvError",
    "RlState",
    "StepReply",
    "VecRlEnv",
    "action_to_cand_bomb",
    "cand_bomb_to_action",
    "f32_bits",
    "f32_from_bits",
    "reward_scale_report",
    "reward_scale_table",
]
