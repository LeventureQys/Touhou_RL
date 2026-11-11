"""``Th08Env``: the gym-style face of the training ground (03 §5).

One environment == one harness process == one xvfb-free software-rendered engine.
The decision period is two logic frames (03 §3.3); the +1 frame input latency is
produced by the engine itself, not simulated here.

Privilege: with ``privileged=False`` the state export and the shadow handle are
refused, which is the mode the benchmark uses for students (03 §5, 20 §7).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from touhou_rl.v400.gym.actions import cand_to_mask
from touhou_rl.v400.gym.protocol import HarnessClient
from touhou_rl.v400.gym.shm import SharedBlock

#: Decision period in logic frames (03 §3.3).
DECISION_FRAMES = 2

DIFFICULTIES = ("easy", "normal", "hard", "lunatic")

#: Field indices into the 25 tracked values (same order as the replay trace).
F_FRAME = 0
F_STAGE = 1
F_SCORE = 4
F_GRAZE = 5
F_DEATHS = 6
F_LIVES = 7
F_BOMBS = 8
F_POWER = 9


class PrivilegeError(RuntimeError):
    """Raised when a zero-privilege environment is asked for privileged data."""


class StaleStateError(RuntimeError):
    """The shared state block does not belong to the frame the engine just reported.

    T7.3: the engine now publishes the 25 tracked fields itself at the end of
    ``RESET`` and ``STEP``, so "send ``STATE`` first" is no longer a prerequisite.
    This error is the fail-closed half of that change: if the block's frame
    counter disagrees with the frame the reply describes, the caller would read a
    previous episode's fields, and no value is returned instead.
    """


@dataclass(frozen=True)
class StageStart:
    """Deterministic entry into a stage (03 §5)."""

    stage: int
    difficulty: str
    team: int
    seed: int
    resources: str = "standard"

    def __post_init__(self) -> None:
        if not 0 <= self.stage <= 8:
            raise ValueError(f"stage out of range: {self.stage}")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"unknown difficulty: {self.difficulty!r}")
        if not 0 <= self.team <= 11:
            raise ValueError(f"team out of range: {self.team}")
        if not 0 <= self.seed <= 0xFFFF:
            raise ValueError(f"seed out of range: {self.seed}")
        if self.resources not in ("standard", "practice"):
            raise ValueError(f"unknown resources: {self.resources!r}")


@dataclass
class Info:
    """Everything the caller may want beyond the observation."""

    frame: int = 0
    events: list[str] = field(default_factory=list)
    state: dict[int, int] | None = None
    game_over: bool = False

    @property
    def privileged(self) -> bool:
        return self.state is not None


@dataclass(frozen=True)
class FrameReply:
    """One raw ``STEP`` reply, as the tracker needs it.

    ``hash`` is the engine's own 25-field digest (absent on the ``exit=1`` reply,
    which is exactly the case a trajectory fast-forward must refuse); ``frame``
    is the engine's ``FrameCount`` after the advance, i.e. the stage frame the
    recorded corpora number their hashes with.
    """

    frame: int
    advanced: int
    hash: str | None
    exit: bool = False
    game_over: bool = False


def _f32_bits(value: int) -> float:
    """Reinterpret a tracked field as the float it was recorded from."""
    import struct

    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


class Th08Env:
    """A single environment backed by one engine process."""

    def __init__(
        self,
        binary: str | os.PathLike[str],
        data_dir: str | os.PathLike[str],
        *,
        shm_name: str,
        privileged: bool = True,
        decision_frames: int = DECISION_FRAMES,
        max_frames: int = 100_000,
        role: str = "env",
    ) -> None:
        self.privileged = privileged
        self.decision_frames = decision_frames
        self.max_frames = max_frames
        #: ``env`` renders (needs a display); ``shadow`` is the null device and
        #: exports the same 25 fields and hash without GL.
        self.role = role
        self._binary = binary
        self._data_dir = data_dir
        self._shm_name = shm_name
        self._client = HarnessClient(binary, data_dir, shm_name=shm_name, role=role)
        self._shm: SharedBlock | None = None
        self._frames = 0
        self._start: StageStart | None = None
        self._prev_deaths: float | None = None
        self._prev_score: int | None = None
        self._restart_count = 0
        self._recovery_log: list[dict[str, object]] = []
        #: True once the engine has shown it publishes the state block itself
        #: (``RESET`` carries ``state_published=1``, T7.3).  Older builds do not,
        #: and for them the freshness check stays off instead of guessing.
        self._state_published = False
        #: The hash the last ``RESET`` reply carried, lowercase hex, or ``None``.
        #: ``TrajectoryStart`` compares it with the recorded frame-0 hash.
        self.last_reset_hash: str | None = None

    # -- lifetime ---------------------------------------------------------
    def close(self) -> None:
        shm = self._shm
        self._shm = None
        try:
            if shm is not None:
                shm.close()
        finally:
            self._client.close()

    def __enter__(self) -> "Th08Env":
        self._start_client()
        return self

    def recover(self) -> None:
        """Explicitly tear down and recreate the harness connection."""
        old_shm = self._shm
        self._shm = None
        close_error: BaseException | None = None
        try:
            if old_shm is not None:
                old_shm.close()
        except BaseException as exc:
            close_error = exc
        try:
            self._client.close()
        except BaseException as exc:
            if close_error is None:
                close_error = exc
        self._client = HarnessClient(self._binary, self._data_dir, shm_name=self._shm_name,
                                     role=self.role)
        try:
            self._start_client()
        except BaseException:
            self._shm = None
            raise
        if close_error is not None:
            raise close_error

    def _start_client(self) -> None:
        self._client.start()
        self._client.hello()
        self._client.attach_shm()
        self._client.readback(True)  # publish a frame every step
        self._shm = SharedBlock(self._client.shm_name or "")

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- gym API ----------------------------------------------------------
    def reset(self, start: StageStart) -> tuple[np.ndarray, Info]:
        if self._shm is None:
            raise RuntimeError("use the environment as a context manager")
        reply = self._client.reset(
            stage=start.stage,
            difficulty=DIFFICULTIES.index(start.difficulty),
            team=start.team,
            seed=start.seed,
            practice=start.resources == "practice",
        )
        if reply.get("ready") != "1":
            raise RuntimeError(f"RESET did not reach a playable state: {reply}")
        self._state_published = reply.get("state_published") == "1"
        self._verify_state_block(reply)
        reported = reply.get("hash")
        self.last_reset_hash = reported.strip().lower() if isinstance(reported, str) else None
        self._start = start
        self._frames = 0
        self._prev_deaths = None
        self._prev_score = None
        obs = self._observation()
        info = self._info(events=["reset"])
        self._prev_deaths = _f32_bits(info.state[F_DEATHS]) if info.state else None
        self._prev_score = info.state[F_SCORE] if info.state else None
        return obs, info

    def step(self, cand: int, bomb: int = 0) -> tuple[np.ndarray, float, bool, bool, Info]:
        mask = cand_to_mask(cand, bomb)
        try:
            reply = self._client.step(mask, self.decision_frames)
        except Exception as exc:
            # A dead harness is recoverable: recreate the process and restore the
            # deterministic stage before surfacing a clear event to the caller.
            if self._start is None:
                raise
            self.recover()
            self.reset(self._start)
            self._restart_count += 1
            self._recovery_log.append({
                "episode_aborted": True,
                "restart_index": self._restart_count,
                "reason": type(exc).__name__,
            })
            raise RuntimeError("harness crashed; episode aborted and automatically restarted") from exc
        self._frames += int(reply.get("advanced", 0))
        self._verify_state_block(reply)

        events: list[str] = []
        game_over = reply.get("game_over") == "1"
        if game_over:
            events.append("game_over")
        if reply.get("exit") == "1" and not game_over:
            events.append("exit")

        info = self._info(events=events, game_over=game_over)
        reward = self._privileged_reward(info)
        terminated = game_over
        truncated = (not terminated) and self._frames >= self.max_frames
        return self._observation(), reward, terminated, truncated, info

    def advance(self, mask: int, frames: int = 1) -> FrameReply:
        """Advance ``frames`` engine frames with one raw ``mask`` (per frame).

        Additive to the gym surface: :meth:`step` stays candidate/bomb based and
        keeps moving ``decision_frames`` frames per call.  A recorded trajectory
        needs the other granularity — the mask that was committed on *each* frame
        — and the engine's own hash of the frame it landed on, which is what a
        :class:`~touhou_rl.v400.gym.trajectory.TrajectoryStart` verifies against
        the recorded value.

        Unlike :meth:`step` this is **fail-closed on a dead harness**: it never
        recreates the process and replays the stage start, because a restarted
        episode is no longer the recorded trajectory and resuming from it would
        be a silent divergence.  The caller gets a ``RuntimeError`` instead.

        Bookkeeping matches :meth:`step`: ``self._frames`` counts what the engine
        advanced and the privileged reward baselines are re-read from the state
        block, so the first :meth:`step` after a fast-forward does not credit
        deaths or score that happened while the caller was not in control.
        """
        if not 0 <= int(mask) <= 0xFFFF:
            raise ValueError(f"mask out of range: {mask}")
        if not 1 <= int(frames) <= 100_000:
            raise ValueError(f"frames out of range: {frames}")
        if self._shm is None:
            raise RuntimeError("use the environment as a context manager")
        try:
            reply = self._client.step(int(mask), int(frames))
        except Exception as exc:
            raise RuntimeError(
                "harness died during a raw frame advance; a recorded trajectory cannot be "
                "resumed from a restarted episode") from exc
        advanced = int(reply.get("advanced", 0))
        exited = reply.get("exit") == "1"
        self._frames += advanced
        self._verify_state_block(reply)
        if not exited:
            # On an exit reply the engine has not published this frame's fields,
            # so the baselines are left alone rather than taken from the last
            # published frame.
            state = self._shm.state()
            self._prev_deaths = _f32_bits(int(state[F_DEATHS]))
            self._prev_score = int(state[F_SCORE])
        return FrameReply(
            frame=int(reply.get("frame", self._frames)),
            advanced=advanced,
            hash=reply.get("hash"),
            exit=exited,
            game_over=reply.get("game_over") == "1",
        )

    # -- internals --------------------------------------------------------
    def _verify_state_block(self, reply: dict[str, str]) -> None:
        """T7.3: refuse a state block that is older than the reply's own frame.

        ``RESET`` and ``STEP`` publish the 25 tracked fields into the shared
        block themselves (``rlharness.cpp`` calls ``PublishState`` in both
        handlers), so a caller no longer has to send ``STATE`` before reading
        them.  The ordering is not trusted blindly: the block's header carries the
        frame it was written for, and it has to be the frame this reply reports.
        A mismatch means the block still holds a previous episode's fields.

        Capability gate: an engine that predates T7.3 answers ``RESET`` without
        ``state_published=1``; for those the check stays off and the old
        "``STATE`` first" rule still applies.  Nothing is fabricated either way.
        """
        if not self._state_published or self._shm is None:
            return
        reported = reply.get("frame", reply.get("frames"))
        if reported is None:
            raise StaleStateError(f"reply carries no frame number: {reply}")
        header = self._shm.header()
        if int(header["frame"]) != int(reported):
            raise StaleStateError(
                f"state block frame {int(header['frame'])} != engine frame {int(reported)}; "
                "the block is not the state this reply describes")

    def _observation(self) -> np.ndarray:
        assert self._shm is not None
        return self._shm.pixels()

    def _info(self, *, events: list[str], game_over: bool = False) -> Info:
        assert self._shm is not None
        header = self._shm.header()
        state = None
        if self.privileged:
            fields = self._shm.state()
            state = {index: int(value) for index, value in enumerate(fields)}
        return Info(frame=int(header["frame"]), events=list(events), state=state, game_over=game_over)

    def _privileged_reward(self, info: Info) -> float:
        """Privileged shaping reward, for logging and the B2 arm only (03 §5).

        Zero-privilege environments get 0.0 here; their reward has to come from
        the pixel reward detector in P2, which is the whole point of that stage.
        """
        if not self.privileged or info.state is None:
            return 0.0
        deaths = _f32_bits(info.state[F_DEATHS])
        score = info.state[F_SCORE]
        reward = 0.0
        if self._prev_deaths is not None and deaths > self._prev_deaths:
            reward -= 1.0 * (deaths - self._prev_deaths)
        if self._prev_score is not None and score > self._prev_score:
            reward += 1e-6 * (score - self._prev_score)
        self._prev_deaths = deaths
        self._prev_score = score
        return reward

    @property
    def obs_spec(self) -> dict[str, object]:
        """Describe the RGBA8 observation returned by ``reset`` and ``step``."""
        if self._shm is None:
            raise RuntimeError("use the environment as a context manager")
        pixels = self._shm.pixels()
        return {
            "shape": pixels.shape,
            "dtype": np.uint8,
            "low": 0,
            "high": 255,
        }

    # -- privileged extras ------------------------------------------------
    def state(self) -> dict[int, int]:
        if not self.privileged:
            raise PrivilegeError("state export is unavailable in zero-privilege mode")
        assert self._shm is not None
        return {index: int(value) for index, value in enumerate(self._shm.state())}

    def debug(self) -> dict[str, str]:
        if not self.privileged:
            raise PrivilegeError("debug counters are unavailable in zero-privilege mode")
        return self._client.debug()

    @property
    def restart_count(self) -> int:
        return self._restart_count

    @property
    def recovery_log(self) -> tuple[dict[str, object], ...]:
        return tuple(self._recovery_log)

    @property
    def shm(self) -> SharedBlock | None:
        """The attached shared-memory block, when the environment is active."""
        return self._shm


class VecTh08Env:
    """Synchronous wrapper for a non-empty sequence of environments.

    Results stay as per-environment sequences, so this wrapper does not impose
    any observation or info type on the wrapped :class:`Th08Env` instances.
    """

    def __init__(self, envs) -> None:
        self.envs = list(envs)
        if not self.envs:
            raise ValueError("VecTh08Env requires at least one environment")

    def reset(self):
        """Reset every environment concurrently and return one result per environment."""
        with ThreadPoolExecutor(max_workers=len(self.envs)) as pool:
            return list(pool.map(lambda environment: environment.reset(), self.envs))

    def step(self, actions):
        """Step every environment concurrently, preserving input order."""
        actions = list(actions)
        if len(actions) != len(self.envs):
            raise ValueError("actions must contain one item per environment")
        with ThreadPoolExecutor(max_workers=len(self.envs)) as pool:
            return list(pool.map(lambda pair: pair[0].step(pair[1]), zip(self.envs, actions)))

    def close(self) -> None:
        """Close every environment, then re-raise the first close failure."""
        first_error = None
        for environment in self.envs:
            try:
                environment.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
