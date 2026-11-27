#!/usr/bin/env python3
"""Offline tests for ``tools/rl/v8_32_takeover.py`` -- no engine is started.

Every test drives the real runner against a **scripted HarnessClient** (the same
method surface ``touhou_rl.v400.gym.protocol.HarnessClient`` has: ``start``,
``reset``, ``step``, ``state_hash``, ``request("STATE")``, ``player``, ``hud``,
``enemies``, ``events``, ``attach_shm``, ``readback``, ``pixels``, ``close``).  The
tape is synthesized with the geometry of the real T04 shard (first slot at frame
16, the 8 lead-in decisions, a 16-hex hash chain with zero breaks), so the
frame/slot arithmetic under test is the real one without depending on an asset.

What is covered (the twelve items the ticket names are marked 01..12):

01  entry-frame mapping: 16 -> 0 slots (+8 lead-in), 2632 -> 1308, 2704 -> 1344
02  a matching slot-end hash continues to the hand-over
03  a mismatching hash fails closed and hands nothing over
04  an odd / below-16 / uncovered entry frame is refused
05  --cap-frames is spent in whole 2-frame decisions
06  the tape arm replays the recorded masks after the entry
07  the constant arm emits one cand plus the SHOOT pulse
08  --decision-log starts with a header and has one row per decision
09  without --decision-log no file is created
10  result.json carries every required field
11  an engine sha256 mismatch is refused before the start
12  --arm checkpoint without --checkpoint is refused (the arm used to be an
    interface that raised; it now runs, so this item pins the required argument)
plus 13..20: tape contract (chain break, grid break), seed mismatch, audible cfg,
a death stopping the window, --constant-cand validation, an unplayable RESET and an
engine exit.

The checkpoint arm (the ticket's second half, items 21..36):

21  (1) a --checkpoint-sha256 that is not the file's sha256 is refused
22  (2) another obs spec version/digest is refused (scripted and real loader)
23  (3) no pixels -> fail closed before any hand-over and no log
24  (3b) a shared block that never publishes pixels is refused at frame 1
25  (4) the 8x18 history is seeded from the tape's executed_cand
26  (4b) and continues with the policy's own candidates after the entry
27  (5) two frames per decision, two renders per decision, four-frame stack
28  (6) the tensors model.act receives are bit-identical to v400_student_eval.py
29  (7) result.json carries the arm identity
30  (8) the decision-log header and rows carry the job's fields
31  (9) an entry-hash failure starts no policy decision and writes no log
32  (10) the SHOOT pulse still applies to the checkpoint arm
33  (11) the tape/constant arms send no SHM/READBACK/PIXELS and claim no arm
34  (12) the pixels()-fallback source produces the same tensors as SHM
35  the real load_checkpoint_v3 path is used and a wrong file is refused
36  a policy outside the 18-candidate contract stops the window

The referee-side ``enemies`` reading and ``boss_summary`` (items 37..46):

37  the tape arm's rows carry the per-decision enemies reading
38  the constant arm's rows carry it too
39  the checkpoint arm's rows carry it too
40  a refused (and an undecodable) page is written as ``available: false`` with the
    engine's own text; the row is still written and the window still runs
41  no ``--decision-log`` means no extra ENEMIES round trip (counted, all arms)
42  ``boss_summary``: no boss listed -> ``first_boss_frame`` stays null
43  ``boss_summary``: a boss that keeps its life -> first/last life, no zero frame
44  ``boss_summary``: a boss that reaches 0 -> ``boss_zero_life_frame`` is pinned
45  a page smaller than the engine's census is marked ``truncated``; an exact page
    and a full page with no census are honest too
46  the reading never reaches the policy: identical tensors with and without a log

``--save-obs`` (the WP05/DAgger prerequisite, items 47..58):

47  the checkpoint arm writes slots.jsonl / the three .npy / meta.json / README,
    one row and one plane per decision
48  the planes are raw uint8, and normalize_play/normalize_hud of them are
    ``torch.equal`` to the tensors ``model.act`` received (bit for bit)
49  without --save-obs: not one new file, not one extra protocol command, and the
    policy's tensors are unchanged
50  a non-checkpoint arm with --save-obs is refused before any client exists (both
    through the API and through the CLI)
51  a non-empty directory is refused and its content is left untouched
52  an existing *empty* directory is accepted
53  a plane write failure fails closed: non-zero exit, rows kept, meta.json says why
54  a row write failure stops the window (fail-closed) and keeps what was written
55  history / executed_cand / mask agree with the decision-log rows
56  the frame windows are exactly the frozen T04 ones (decision_span/obs_window/
    history_window) and the runner's own labels are kept beside them
57  meta.json carries the identity, and README.md names every null field
58  each row's planes are the images of its own recorded frames (no re-render, no
    re-normalization, no off-by-one in the frame ruler)
59  a window that fails at the entry still leaves an empty, honest corpus (rows 0,
    ``complete: false``, the entry failure in meta.json), never a fake row
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for entry in (str(REPO), str(REPO / "tools" / "rl")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import v8_32_takeover as tk  # noqa: E402
import v400_make_cfg  # noqa: E402
from touhou_rl.v400.gym.actions import cand_to_mask  # noqa: E402
from touhou_rl.v400.gym.protocol import (  # noqa: E402
    EnemyRow,
    EventFrame,
    HudSnapshot,
    PlayerSnapshot,
)
from touhou_rl.v400.gym.shm import FLAG_PIXELS_VALID  # noqa: E402
from touhou_rl.v400.labels.state import (  # noqa: E402
    FIELD_NAMES,
    FLOAT_BIT_FIELDS,
    f32_bits,
)
from touhou_rl.v400.teacher.shoot import release_due  # noqa: E402

#: The real shard's slot count, so 2632/2704 map exactly as they do on the asset.
SLOT_COUNT = 6588
LEAD_IN = 8
CAND_CYCLE = (7, 0, 3, 12, 16, 1, 5, 9)
#: The eval tool the checkpoint arm must stay aligned with (read, never modified).
EVAL_PATH = REPO / "tools" / "rl" / "v400_student_eval.py"


# ---------------------------------------------------------------------------
# Fixtures: a tape with the real geometry, a muted data dir, an engine file
# ---------------------------------------------------------------------------
def write_tape(path: Path, *, slots: int = SLOT_COUNT, seed: int = 12,
               lead_in: int = LEAD_IN,
               history_cands: "list[int] | None" = None) -> Path:
    """One synthetic ``slots.jsonl`` on the T04 grid (frame 16, 2 frames/slot)."""

    cands = list(history_cands) if history_cands is not None else [7] * lead_in
    assert len(cands) == lead_in, (len(cands), lead_in)
    lines: list[str] = []
    for index in range(slots):
        frame = tk.TAPE_FIRST_FRAME + tk.DECISION_FRAMES * index
        cand = CAND_CYCLE[index % len(CAND_CYCLE)]
        row = {
            "schema": "v420-dataset-2f/1/slot",
            "episode_seed": int(seed),
            "slot_id": index,
            "decision_frame": frame,
            "commit_frames": [frame, frame + 2],
            "duration_frames": 2,
            "mask": int(cand_to_mask(cand, 0)),
            "executed_cand": cand,
            "bomb": 0,
            "hash_pre": f"{index:016x}",
            "hash_post": f"{index + 1:016x}",
            "history_frames": [2 * step for step in range(lead_in)],
            "history": list(cands),
            "chosen": cand,
        }
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def tape_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("tape")
    return write_tape(base / "slots.jsonl")


@pytest.fixture(scope="session")
def tape(tape_path: Path) -> dict:
    return tk.load_tape(tape_path)


def muted_cfg_bytes() -> bytes:
    """The harness template from ``tools/rl/v400_make_cfg.py`` (all audio 0)."""

    return v400_make_cfg.build()


def make_data_dir(path: Path, *, audible: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    data = bytearray(muted_cfg_bytes())
    if audible:
        data[0x1F] = 1          # musicMode
        data[0x20] = 1          # playSounds
        struct.pack_into("<b", data, 0x27, 39)   # musicVolume
        struct.pack_into("<b", data, 0x28, 36)   # sfxVolume
    (path / "th08.cfg").write_bytes(bytes(data))
    return path


def make_engine(path: Path, payload: bytes = b"fake th08-modern binary\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def make_config(tmp_path: Path, tape_path: Path, **overrides) -> tk.TakeoverConfig:
    engine = overrides.pop("engine", None) or make_engine(tmp_path / "build" / "th08-modern")
    data_dir = overrides.pop("data_dir", None) or make_data_dir(tmp_path / "data")
    expected_sha = overrides.pop("expected_sha", tk.sha256_file(engine))
    fields = {
        "engine": engine,
        "expected_sha": expected_sha,
        "data_dir": data_dir,
        "tape_slots": tape_path,
        "seed": 12,
        "entry_frame": 64,
        "arm": "tape",
        "cap_frames": 8,
        "out": tmp_path / "out",
        "argv": ("v8_32_takeover.py", "run"),
    }
    fields.update(overrides)
    return tk.TakeoverConfig(**fields)


# ---------------------------------------------------------------------------
# The scripted client
# ---------------------------------------------------------------------------
class FakeHarness:
    """Scripted ``HarnessClient``: exact protocol surface, no process.

    ``hash_at(frame)`` is the tape's own rule for the synthesized shard
    (``(frame - 16) // 2`` as 16 hex digits), so a correct replay matches and any
    entry in ``hash_overrides`` makes one slot disagree.

    The pixel surface is scripted too: ``pixels_mode="shm"`` (the default) lets
    ``attach_shm`` succeed and the test's ``block_factory`` hand back a
    :class:`FakeBlock`; ``"checksum"`` refuses the attach and makes ``pixels()``
    return a ``PIXELS``-style checksum (no bytes); ``"array"`` refuses the attach
    and makes ``pixels()`` return the RGBA frame itself.

    The enemy surface is scripted the same way: ``enemies_script(frame)`` returns
    the page for the engine's current frame (the default is the tape's single
    boss), ``enemy_total_script(frame)`` the engine's active-enemy census
    (``enemy_count``), ``enemies_error`` makes the page refused and
    ``enemy_count_missing`` models a client whose census command does not exist.
    """

    def __init__(self, *, stage: int = 0, deaths: float = 0.0, lives: float = 3.0,
                 bombs: float = 3.0, score: int = 0,
                 death_frame: int | None = None,
                 exit_at_frame: int | None = None,
                 stage_after_frame: int | None = None,
                 ready: str = "1",
                 pixels_mode: str = "shm",
                 attach_error: BaseException | None = None,
                 enemies_script=None,
                 enemy_total_script=None,
                 enemies_error: BaseException | None = None,
                 enemy_total_error: BaseException | None = None,
                 enemy_count_missing: bool = False) -> None:
        self.started = False
        self.closed = False
        self.frame = 0
        self.stage = int(stage)
        self.deaths = float(deaths)
        self.lives = float(lives)
        self.bombs = float(bombs)
        self.score = int(score)
        self.death_frame = death_frame
        self.exit_at_frame = exit_at_frame
        self.stage_after_frame = stage_after_frame
        self.ready = ready
        self.steps: list[int] = []
        self.reset_calls: list[dict] = []
        self.commands: list[str] = []
        self.hash_overrides: dict[int, str] = {}
        self.hash_overrides_by_call: dict[int, str] = {}
        self._hash_calls = 0
        self.state_reads = 0
        self.events_reads = 0
        self.player_reads = 0
        self.hud_reads = 0
        self.enemy_reads = 0
        self.enemy_count_reads = 0
        self.enemies_script = enemies_script
        self.enemy_total_script = enemy_total_script
        self.enemies_error = enemies_error
        self.enemy_total_error = enemy_total_error
        self.enemy_count_missing = bool(enemy_count_missing)
        if self.enemy_count_missing:
            # getattr(client, "enemy_count", None) then finds a non-callable, the
            # same shape an older client has: the runner must fall back, not crash
            self.enemy_count = None
        self.pixels_mode = str(pixels_mode)
        self.attach_error = attach_error
        self.shm_attaches = 0
        self.readback_calls = 0
        self.readback_enabled: bool | None = None
        self.pixel_reads = 0
        self.shm_name: str | None = None
        self.block: "FakeBlock | None" = None

    # -- lifetime -------------------------------------------------------
    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    # -- tape identity --------------------------------------------------
    @staticmethod
    def hash_at(frame: int) -> str:
        if frame < tk.TAPE_FIRST_FRAME:
            return "reset-reset-reset"
        return f"{(frame - tk.TAPE_FIRST_FRAME) // tk.DECISION_FRAMES:016x}"

    @property
    def hash_calls(self) -> int:
        return self._hash_calls

    # -- protocol -------------------------------------------------------
    def reset(self, *, stage: int, difficulty: int, team: int, seed: int):
        self.reset_calls.append({"stage": int(stage), "difficulty": int(difficulty),
                                 "team": int(team), "seed": int(seed)})
        self.frame = 0
        self._hash_calls = 0
        return {"_verb": "OK", "ready": self.ready, "stage": str(stage),
                "difficulty": str(difficulty), "team": str(team), "seed": str(seed)}

    def step(self, mask: int, frames: int):
        assert int(frames) == 1, "the takeover replay sends one frame per STEP"
        self.steps.append(int(mask))
        if self.exit_at_frame is not None and self.frame + 1 > self.exit_at_frame:
            return {"_verb": "OK", "advanced": "0", "exit": "1"}
        self.frame += int(frames)
        return {"_verb": "OK", "advanced": "1"}

    def state_hash(self):
        self._hash_calls = self._hash_calls + 1
        if self._hash_calls in self.hash_overrides_by_call:
            return str(self.hash_overrides_by_call[self._hash_calls]), self.frame
        override = self.hash_overrides.get(self.frame)
        return (str(override) if override is not None else self.hash_at(self.frame),
                self.frame)

    def request(self, line: str) -> str:
        self.commands.append(line)
        verb = line.split()[0] if line.split() else ""
        if verb == "STATE":
            return self.state_reply()
        raise AssertionError(f"the runner sent an unexpected command: {line!r}")

    def state_reply(self) -> str:
        self.state_reads += 1
        deaths, lives = self.deaths, self.lives
        if self.death_frame is not None and self.frame >= self.death_frame:
            deaths, lives = 1.0, 2.0
        stage = self.stage
        if self.stage_after_frame is not None and self.frame >= self.stage_after_frame:
            stage = self.stage + 1
        values = {name: 0 for name in FIELD_NAMES}
        values.update({"frame": self.frame, "stage": stage, "score": self.score,
                       "deaths": deaths, "lives_remaining": lives,
                       "bombs_remaining": self.bombs})
        parts = ["OK", f"frames={self.frame}"]
        for index, name in enumerate(FIELD_NAMES):
            value = values[name]
            if name in FLOAT_BIT_FIELDS:
                parts.append(f"f{index}={f32_bits(value):08x}")
            else:
                parts.append(f"f{index}={int(value)}")
        return " ".join(parts)

    def player(self):
        self.player_reads += 1
        return PlayerSnapshot(frame=self.frame, hp=float(self.lives),
                              lives=int(self.lives), bombs=float(self.bombs),
                              power=0.0, focus=0, is_youkai=0, player_state=0,
                              invuln_state=0, position=(120.0, 384.0),
                              hurtbox_min=(118.0, 382.0),
                              hurtbox_max=(122.0, 386.0))

    def hud(self):
        self.hud_reads += 1
        return HudSnapshot(frame=self.frame, display_score=self.score,
                           globals_score=self.score, lives=int(self.lives),
                           bombs=int(self.bombs), power=0, graze=0, point=0,
                           time=0, game_over=0, display_age=0)

    def enemy_page(self, frame: int) -> list:
        """The ``ENEMIES`` page of one engine frame (default: the tape's boss)."""

        if self.enemies_script is not None:
            return list(self.enemies_script(int(frame)))
        return [EnemyRow(slot=0, flags1=0x3, position=(120.0, 100.0, 0.0),
                         life=4878, timer=9)]

    def enemies(self, offset: int = 0, limit: int = 16):
        self.enemy_reads += 1
        if self.enemies_error is not None:
            raise self.enemies_error
        return self.enemy_page(self.frame)

    def enemy_count(self) -> int:
        self.enemy_count_reads += 1
        if self.enemy_total_error is not None:
            raise self.enemy_total_error
        if self.enemy_total_script is not None:
            return int(self.enemy_total_script(self.frame))
        return len(self.enemy_page(self.frame))

    def events(self):
        self.events_reads += 1
        return EventFrame(frame=self.frame, frame_event_flags=0x1000)

    # -- pixels ---------------------------------------------------------
    def attach_shm(self, name: str | None = None):
        self.shm_attaches += 1
        if self.attach_error is not None:
            raise self.attach_error
        if self.pixels_mode != "shm":
            return {"_verb": "OK", "active": "0", "reason": f"{self.pixels_mode} mode"}
        self.shm_name = str(name or "fake-shm").lstrip("/")
        return {"_verb": "OK", "active": "1", "name": self.shm_name,
                "bytes": str(tk_pixels_offset() + 640 * 480 * 4)}

    def readback(self, enabled: bool | None = None):
        self.readback_calls += 1
        self.readback_enabled = None if enabled is None else bool(enabled)
        if enabled is None:
            return {"_verb": "OK", "readback": "1", "frames": str(self.frame),
                    "micros": "0"}
        return {"_verb": "OK", "readback": "1" if enabled else "0",
                "frames": str(self.frame), "micros": "0"}

    def pixels(self, path: str | None = None):
        self.pixel_reads += 1
        if self.pixels_mode == "array":
            return fake_pixels(self.frame)
        if self.pixels_mode == "checksum":
            # What the frozen client's PIXELS command really returns: a digest.
            return {"_verb": "OK", "frame": str(self.frame),
                    "width": "640", "height": "480",
                    "fnv1a64": f"{(self.frame * 2654435761) & ((1 << 64) - 1):016x}"}
        raise AssertionError(f"PIXELS is not scripted in {self.pixels_mode} mode")


def tk_pixels_offset() -> int:
    """``touhou_rl.v400.gym.shm.PIXELS_OFFSET`` (192) without importing the reader."""

    from touhou_rl.v400.gym.shm import PIXELS_OFFSET  # noqa: PLC0415

    return int(PIXELS_OFFSET)


#: One deterministic RGBA frame per game frame: both code paths must read the same
#: bytes for the same frame, or the tensor comparison would be meaningless.
_PIXEL_BASE = (np.arange(480 * 640 * 4, dtype=np.uint32) % 251).astype(np.uint8)


def fake_pixels(frame: int) -> np.ndarray:
    """The 640x480 RGBA readback of game frame ``frame`` (pure function)."""

    shifted = (_PIXEL_BASE.astype(np.uint16) + (int(frame) * 37) % 251).astype(np.uint8)
    return shifted.reshape(480, 640, 4)


class FakeBlock:
    """Just enough ``SharedBlock``: a header with a frame counter and the RGBA.

    ``header()`` reports ``engine.frame`` -- which advances with every ``STEP`` --
    so the stale-frame gate passes on a correct run; ``flags=0`` models a block the
    engine never published and ``stale=True`` a block that stopped advancing.
    """

    def __init__(self, engine: FakeHarness, *, flags: int | None = None,
                 stale: bool = False) -> None:
        self.engine = engine
        self.flags = FLAG_PIXELS_VALID if flags is None else int(flags)
        self.stale = bool(stale)
        self.closed = False
        self.header_reads = 0
        self.pixel_reads = 0

    def header(self) -> dict:
        self.header_reads += 1
        return {"version": 1, "frame": (0 if self.stale else int(self.engine.frame)),
                "state_hash": 0, "width": 640, "height": 480,
                "state_bytes": 100, "flags": self.flags}

    def pixels(self) -> np.ndarray:
        self.pixel_reads += 1
        return fake_pixels(0 if self.stale else self.engine.frame)

    def close(self) -> None:
        self.closed = True


class RecordingModel:
    """A scripted obs-v3 actor that records every tensor ``act`` receives."""

    def __init__(self, cands: "list[int] | None" = None, *, cand: int = 5,
                 bomb: int = 0, offset: int = 0,
                 raises: BaseException | None = None) -> None:
        self.cands = list(cands or [])
        self.cand = int(cand)
        self.bomb = int(bomb)
        self.offset = int(offset)
        self.raises = raises
        self.calls = 0
        self.inputs: "list[tuple]" = []

    def act(self, play, hud, hist):
        index = self.calls
        self.calls += 1
        self.inputs.append((play.clone(), hud.clone(), hist.clone()))
        if self.raises is not None:
            raise self.raises
        if self.cands:
            position = self.offset + index
            assert position < len(self.cands), (
                f"RecordingModel ran out of candidates at call {position}")
            return int(self.cands[position]), self.bomb
        return self.cand, self.bomb


def policy_loader(model: object, *, obs_spec_version: str = tk.OBS_SPEC_V3_VERSION,
                  obs_spec_digest: str = tk.OBS_SPEC_V3_DIGEST,
                  arch: str = "v3", parameters: int | None = 1234):
    """A ``policy_loader`` that returns a scripted actor and a scripted payload.

    The sha gate and the obs-spec gate still run in front of it (they live in
    ``load_policy``), so a test cannot bypass them by injecting a loader.
    """

    payload = {
        "schema": "v400-student-v3-checkpoint/1",
        "arch": arch,
        "obs_spec_version": obs_spec_version,
        "obs_spec_digest": obs_spec_digest,
        "model": {"class": "StudentV3", "arch": arch},
        "parameters": parameters,
    }

    def loader(config):  # noqa: ANN001 - the runner's loader protocol
        return model, dict(payload)

    return loader


def make_checkpoint(path: Path, payload: bytes = b"fake obs-v3 checkpoint\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def checkpoint_block_factory(fake: FakeHarness, **kwargs):
    """A ``block_factory`` that maps a :class:`FakeBlock` onto ``fake``."""

    def factory(name: str) -> FakeBlock:
        fake.block = FakeBlock(fake, **kwargs)
        return fake.block

    return factory


def run_checkpoint(tmp_path: Path, tape_path: Path, *,
                   model: RecordingModel | None = None,
                   entry_frame: int = 64, cap_frames: int = 8,
                   decision_log: Path | None = None,
                   checkpoint_sha256: str | None = None,
                   fake_kwargs: "dict | None" = None,
                   block_kwargs: "dict | None" = None,
                   loader=None, out_name: str = "out_ckpt",
                   **config_overrides) -> tuple[dict, FakeHarness, RecordingModel]:
    """One scripted checkpoint-arm run through the real runner."""

    model = model if model is not None else RecordingModel(cand=5)
    checkpoint = make_checkpoint(tmp_path / "ckpt_v3.pt")
    config = make_config(
        tmp_path, tape_path, arm="checkpoint", entry_frame=entry_frame,
        cap_frames=cap_frames, decision_log=decision_log, checkpoint=checkpoint,
        checkpoint_sha256=(checkpoint_sha256 if checkpoint_sha256 is not None
                           else tk.sha256_file(checkpoint)),
        out=tmp_path / out_name, **config_overrides)
    fake = FakeHarness(**(fake_kwargs or {}))
    result = tk.run_takeover(
        config, client_factory=lambda _config: fake,
        block_factory=checkpoint_block_factory(fake, **(block_kwargs or {})),
        policy_loader=loader if loader is not None else policy_loader(model),
        clock=iter([0.0, 0.25, 0.5, 1.0, 2.0, 3.0]).__next__)
    return result, fake, model


class Factory:
    """``client_factory`` that records whether it was ever called."""

    def __init__(self, fake: FakeHarness | None = None) -> None:
        self.fake = fake
        self.calls = 0

    def __call__(self, config):  # noqa: ANN001 - the runner's factory protocol
        self.calls += 1
        if self.fake is None:
            raise AssertionError("the runner created a client it should never have")
        return self.fake


def run(config: tk.TakeoverConfig, fake: FakeHarness | None = None,
        factory: Factory | None = None, *,
        block_factory=None, policy_loader=None) -> tuple[dict, FakeHarness | None, Factory]:
    factory = factory or Factory(fake)
    result = tk.run_takeover(config, client_factory=factory,
                             block_factory=block_factory,
                             policy_loader=policy_loader,
                             clock=iter([0.0, 0.25, 0.5, 1.0, 2.0, 3.0]).__next__)
    return result, (fake if fake is not None else factory.fake), factory


def read_result(config: tk.TakeoverConfig) -> dict:
    return json.loads((config.out / "result.json").read_text(encoding="utf-8"))


def rows_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# ---------------------------------------------------------------------------
# 01 entry-frame mapping
# ---------------------------------------------------------------------------
def test_01_entry_frame_maps_to_the_recorded_number_of_slots(tape: dict) -> None:
    for frame, slots, decisions in ((16, 0, 8), (2632, 1308, 1316), (2704, 1344, 1352)):
        plan = tk.entry_plan(tape, frame)
        assert len(plan["lead_in"]) == LEAD_IN, frame
        assert len(plan["prefix_slots"]) == slots, frame
        assert plan["decision_count"] == decisions, frame
        assert plan["entry_index"] == frame // 2
        assert plan["entry_slot"].decision_frame == frame
        assert plan["anchor_hash"] == f"{slots:016x}"
        assert len(plan["takeover_slots"]) == SLOT_COUNT - slots
    # the last prefix slot's hash_post is the entry slot's hash_pre (0 chain breaks)
    plan = tk.entry_plan(tape, 2632)
    assert plan["prefix_slots"][-1].hash_post == plan["anchor_hash"]


# ---------------------------------------------------------------------------
# 02 a matching hash continues to the hand-over
# ---------------------------------------------------------------------------
def test_02_matching_slot_hashes_reach_the_entry_and_hand_over(
        tmp_path: Path, tape_path: Path) -> None:
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8)
    result, fake, factory = run(config, FakeHarness())
    assert factory.calls == 1
    assert result["entry_verified"] is True
    assert result["entry_ready"] is True
    assert result["entry_frame"] == 64
    assert result["entry_hash"] == f"{24:016x}"
    assert result["entry_hash_matches_tape_anchor"] is True
    assert result["frame_relation"] == {"expected": 64, "state_frame": 64,
                                        "hash_frames": 64, "ok": True}
    assert result["prefix"]["slots"] == 24
    assert result["prefix"]["lead_in_decisions"] == LEAD_IN
    assert result["prefix"]["verified_slots"] == 24
    assert result["prefix"]["frames"] == 64
    assert result["frames_prefix"] == 64
    assert result["frames_takeover"] == 8
    assert result["frames"] == 72
    assert result["decisions"] == 4
    assert result["stop"] == "frame_cap"
    assert result["exit_code"] == tk.EXIT_OK
    assert result["reset"] == {"_verb": "OK", "ready": "1", "stage": "0",
                               "difficulty": "1", "team": "0", "seed": "12"}
    assert fake.reset_calls == [{"stage": 0, "difficulty": 1, "team": 0, "seed": 12}]
    assert fake.started and fake.closed
    assert len(fake.steps) == 64 + 8
    assert fake.player_reads == 1 and fake.hud_reads == 1 and fake.enemy_reads == 1
    assert result["entry_state"]["state"]["frame"] == 64
    assert result["entry_state"]["player"]["lives"] == 3
    assert result["entry_state"]["enemies"]["boss_count"] == 1
    assert set(result["entry_state"]["state"]) == set(FIELD_NAMES)


# ---------------------------------------------------------------------------
# 03 a mismatching hash fails closed
# ---------------------------------------------------------------------------
def test_03_hash_mismatch_fails_closed_and_hands_nothing_over(
        tmp_path: Path, tape_path: Path) -> None:
    log = tmp_path / "decisions.jsonl"
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8,
                         decision_log=log)
    fake = FakeHarness()
    fake.hash_overrides[28] = "deadbeefdeadbeef"   # slot 5's hash_post
    result, _, _ = run(config, fake)
    assert result["entry_verified"] is False
    assert result["entry_ready"] is False
    assert result["mismatch_slot"] == 5
    assert result["mismatch"]["kind"] == "slot"
    assert result["mismatch"]["frame"] == 28
    assert result["mismatch"]["expected"] == f"{6:016x}"
    assert result["mismatch"]["actual"] == "deadbeefdeadbeef"
    assert result["stop"] == "entry_hash_mismatch"
    assert result["exit_code"] == tk.EXIT_ENTRY_FAILED
    assert result["decisions"] == 0
    assert result["frames_takeover"] == 0
    assert result["entry_state"] is None
    # frames 1..28 only: the run stopped at the bad slot and never took over
    assert len(fake.steps) == 28
    assert fake.state_reads == 0          # no STATE read for a window that never ran
    assert not log.exists()
    assert read_result(config)["mismatch_slot"] == 5


def test_03b_entry_hash_that_is_not_the_tape_anchor_fails_closed(
        tmp_path: Path, tape_path: Path) -> None:
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8)
    fake = FakeHarness()
    # The prefix reads HASH once after the lead-in (frame 16) and once per slot, so
    # 1 + 24 = 25 reads end on the entry frame; the entry read is the 26th.  The
    # same frame is read twice, so the override has to be per call, not per frame.
    entry_hash_call = 1 + 24 + 1
    fake.hash_overrides_by_call[entry_hash_call] = "1111111111111111"
    result, _, _ = run(config, fake)
    assert fake.hash_calls == entry_hash_call
    assert result["entry_verified"] is False
    assert result["entry_hash"] == "1111111111111111"
    assert result["entry_anchor_hash"] == f"{24:016x}"
    assert result["mismatch_slot"] == 24
    assert result["mismatch"]["kind"] == "entry"
    assert result["stop"] == "entry_hash_mismatch"
    assert result["exit_code"] == tk.EXIT_ENTRY_FAILED
    assert result["decisions"] == 0
    assert result["prefix"]["verified_slots"] == 24


# ---------------------------------------------------------------------------
# 04 entry-frame refusals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("frame, stop", [
    (2633, "entry_frame_invalid"),      # odd
    (8, "entry_frame_invalid"),         # before the first slot
    (13192, "entry_frame_uncovered"),   # past the last slot (16 + 2 * 6588)
])
def test_04_bad_entry_frames_are_refused_before_any_start(
        tmp_path: Path, tape_path: Path, frame: int, stop: str) -> None:
    config = make_config(tmp_path, tape_path, entry_frame=frame)
    factory = Factory(None)             # a client here is a failure
    result = tk.run_takeover(config, client_factory=factory)
    assert result["entry_verified"] is False
    assert result["stop"] == stop
    assert result["exit_code"] == tk.EXIT_REFUSED
    assert result["entry_frame"] is None
    assert factory.calls == 0
    assert read_result(config)["stop"] == stop


def test_04b_entry_plan_refuses_the_same_frames(tape: dict) -> None:
    for frame, stop in ((2633, "entry_frame_invalid"), (8, "entry_frame_invalid"),
                        (13192, "entry_frame_uncovered"), (13194, "entry_frame_uncovered")):
        with pytest.raises(tk.PreflightRefused) as info:
            tk.entry_plan(tape, frame)
        assert info.value.stop == stop, frame


# ---------------------------------------------------------------------------
# 05 --cap-frames semantics
# ---------------------------------------------------------------------------
def test_05_cap_frames_is_spent_in_whole_decisions(
        tmp_path: Path, tape_path: Path) -> None:
    # an odd cap cannot buy half a decision: 5 frames -> 2 decisions (4 frames)
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=5)
    result, fake, _ = run(config, FakeHarness())
    assert result["decisions"] == 2
    assert result["frames_takeover"] == 4
    assert result["frames"] == 68
    assert result["stop"] == "frame_cap"
    assert result["stop_detail"]["cap_frames"] == 5
    assert len(fake.steps) == 68

    # a cap of 0 verifies the entry and hands over nothing
    zero = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=0,
                       out=tmp_path / "out_zero")
    result_zero, fake_zero, _ = run(zero, FakeHarness())
    assert result_zero["entry_verified"] is True
    assert result_zero["decisions"] == 0
    assert result_zero["frames_takeover"] == 0
    assert result_zero["stop"] == "frame_cap"
    assert result_zero["exit_code"] == tk.EXIT_OK
    assert len(fake_zero.steps) == 64

    # the ticket's cap: 1868 frames -> 934 decisions from entry frame 16
    full = make_config(tmp_path, tape_path, entry_frame=16, cap_frames=1868,
                       out=tmp_path / "out_full")
    result_full, _, _ = run(full, FakeHarness())
    assert result_full["entry_frame"] == 16
    assert result_full["prefix"]["slots"] == 0
    assert result_full["prefix"]["lead_in_decisions"] == LEAD_IN
    assert result_full["frames_prefix"] == 16
    assert result_full["decisions"] == 934
    assert result_full["frames_takeover"] == 1868
    assert result_full["frames"] == 16 + 1868
    assert result_full["stop"] == "frame_cap"


def test_05b_negative_cap_is_refused(tmp_path: Path, tape_path: Path) -> None:
    config = make_config(tmp_path, tape_path, cap_frames=-2)
    factory = Factory(None)
    result = tk.run_takeover(config, client_factory=factory)
    assert result["stop"] == "cap_frames_invalid"
    assert result["exit_code"] == tk.EXIT_REFUSED
    assert factory.calls == 0


def test_05c_a_cap_beyond_the_tape_is_refused(tmp_path: Path,
                                              tape_path: Path) -> None:
    # entry 16 leaves 6588 slots = 13176 frames on the tape, so 13178 cannot fit
    config = make_config(tmp_path, tape_path, entry_frame=16, cap_frames=13178)
    factory = Factory(None)
    result = tk.run_takeover(config, client_factory=factory)
    assert result["stop"] == "cap_frames_invalid"
    assert result["exit_code"] == tk.EXIT_REFUSED
    assert "exceeds the tape" in result["failure"]["detail"]
    assert result["tape"]["cap_slots"] == SLOT_COUNT
    assert factory.calls == 0


# ---------------------------------------------------------------------------
# 06 the tape arm replays the recorded masks
# ---------------------------------------------------------------------------
def test_06_tape_arm_continues_the_recorded_masks(
        tmp_path: Path, tape_path: Path, tape: dict) -> None:
    from touhou_rl.v400.teacher.shoot import commit_frames  # noqa: PLC0415

    log = tmp_path / "tape.jsonl"
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=12,
                         decision_log=log)
    result, fake, _ = run(config, FakeHarness())
    start = (64 - tk.TAPE_FIRST_FRAME) // 2      # slot 24
    expected: list[int] = []
    for slot in tape["slots"][start:start + 6]:
        # the frozen executor rule, expanded here independently of the tool
        parts = commit_frames(slot.cand, decision_frames=2,
                              release_shoot=release_due(slot.decision_index, 48))
        for mask, frames in parts:
            expected.extend([int(mask)] * int(frames))
    first_parts = commit_frames(tape["slots"][start].cand, decision_frames=2)
    first_expanded = [int(mask) for mask, frames in first_parts
                      for _ in range(int(frames))]
    assert first_expanded == [tape["slots"][start].mask] * 2
    assert fake.steps[64:] == expected
    rows = rows_of(log)[1:]
    assert [row["cand"] for row in rows] == [
        tape["slots"][start + offset].cand for offset in range(6)]
    assert [row["masks"] for row in rows] == [
        expected[2 * offset:2 * offset + 2] for offset in range(6)]
    assert [row["mask"] for row in rows] == [row["masks"][0] for row in rows]
    assert [row["decision"] for row in rows] == list(range(32, 38))
    assert [row["frame"] for row in rows] == list(range(64, 76, 2))
    assert result["decisions"] == 6


# ---------------------------------------------------------------------------
# 07 the constant arm: one candidate plus the SHOOT pulse
# ---------------------------------------------------------------------------
def test_07_constant_arm_emits_one_cand_with_the_pulse(
        tmp_path: Path, tape_path: Path) -> None:
    log = tmp_path / "constant.jsonl"
    config = make_config(tmp_path, tape_path, entry_frame=64, arm="constant",
                         constant_cand=5, cap_frames=192, decision_log=log)
    result, fake, _ = run(config, FakeHarness())
    assert result["decisions"] == 96
    assert result["frames_takeover"] == 192
    rows = rows_of(log)[1:]
    assert len(rows) == 96
    assert {row["cand"] for row in rows} == {5}
    full = int(cand_to_mask(5, 0))
    held = full & ~1
    pulse_decisions = [row["decision"] for row in rows
                       if row["masks"][0] != row["masks"][1]]
    assert pulse_decisions == [47, 95]
    for row in rows:
        if row["decision"] in (47, 95):
            assert release_due(row["decision"], 48) is True
            assert row["masks"] == [full, held]
        else:
            assert row["masks"] == [full, full]
    assert fake.steps[64:66] == [full, full]


# ---------------------------------------------------------------------------
# 08 / 09 the decision log
# ---------------------------------------------------------------------------
def test_08_decision_log_header_then_one_row_per_decision(
        tmp_path: Path, tape_path: Path) -> None:
    log = tmp_path / "decisions.jsonl"
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8,
                         decision_log=log)
    result, _, _ = run(config, FakeHarness())
    lines = rows_of(log)
    assert len(lines) == result["decisions"] + 1 == 5
    header = lines[0]
    assert header["schema"] == tk.SCHEMA_LOG_HEADER
    for key in ("arm", "seed", "entry_frame", "entry_hash", "tape", "reset",
                "frames_per_decision", "cap_frames", "engine", "shoot_pulse_period",
                "checkpoint", "obs_spec_version"):
        assert key in header, key
    assert header["arm"] == "tape"
    assert header["seed"] == 12
    assert header["entry_frame"] == 64
    assert header["entry_hash"] == f"{24:016x}"
    assert header["tape"]["slots"] == SLOT_COUNT
    assert header["reset"]["ready"] == "1"
    assert header["constant_cand"] is None
    for row in lines[1:]:
        assert row["schema"] == tk.SCHEMA_LOG_ROW
        for key in ("decision", "frame", "cand", "mask", "deaths", "lives",
                    "bombs", "score", "stage", "events", "end"):
            assert key in row, key
        assert row["frames_per_decision"] == 2
        assert row["events"] == ["ENEMY_SPAWNED"]
        assert row["end"] == ""
    assert result["decision_log"]["rows"] == 5
    assert result["decision_log"]["header_rows"] == 1


def test_09_no_decision_log_means_no_file(tmp_path: Path, tape_path: Path) -> None:
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8)
    result, _, _ = run(config, FakeHarness())
    assert result["decision_log"] is None
    assert not (tmp_path / "decisions.jsonl").exists()
    assert [path.name for path in tmp_path.glob("*.jsonl")] == []


def test_09b_decision_log_parent_must_exist(tmp_path: Path, tape_path: Path) -> None:
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8,
                         decision_log=tmp_path / "missing" / "decisions.jsonl")
    factory = Factory(None)
    result = tk.run_takeover(config, client_factory=factory)
    assert result["stop"] == "decision_log_parent_missing"
    assert result["exit_code"] == tk.EXIT_REFUSED
    assert factory.calls == 0


# ---------------------------------------------------------------------------
# 10 result.json fields
# ---------------------------------------------------------------------------
def test_10_result_json_carries_every_required_field(
        tmp_path: Path, tape_path: Path) -> None:
    log = tmp_path / "decisions.jsonl"
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8,
                         decision_log=log)
    run(config, FakeHarness())
    result = read_result(config)
    for key in tk.RESULT_REQUIRED_KEYS:
        assert key in result, key
    assert result["schema"] == tk.SCHEMA_RESULT
    assert result["engine"]["sha256"] == tk.sha256_file(config.engine)
    assert result["engine"]["sha_matches"] is True
    assert result["mute"]["before"]["muted"] is True
    assert result["mute"]["before"]["bytes"] == {
        "musicMode": 0, "playSounds": 0, "musicVolume": 0, "sfxVolume": 0}
    assert result["mute"]["after"]["muted"] is True
    assert result["mute"]["ok"] is True
    assert result["mute"]["problems"] == []
    assert isinstance(result["wall_s"], float)
    assert result["frames"] == 72
    assert result["stop"] == "frame_cap"
    assert result["deaths"] == 0.0
    assert result["entry_verified"] is True
    assert result["mismatch_slot"] is None
    assert result["failure"] is None
    assert result["exit_code"] == tk.EXIT_OK
    assert result["tape"]["sha256"] == tk.sha256_file(tape_path)


def test_10b_a_refused_run_also_writes_a_complete_result(
        tmp_path: Path, tape_path: Path) -> None:
    config = make_config(tmp_path, tape_path, expected_sha="0" * 64)
    result = tk.run_takeover(config, client_factory=Factory(None))
    for key in tk.RESULT_REQUIRED_KEYS:
        assert key in result, key
    assert result["failure"]["stop"] == "engine_sha_mismatch"
    assert result["failure"]["detail"]
    assert read_result(config)["failure"]["stop"] == "engine_sha_mismatch"


# ---------------------------------------------------------------------------
# 11 engine sha256
# ---------------------------------------------------------------------------
def test_11_engine_sha_mismatch_is_refused_before_the_start(
        tmp_path: Path, tape_path: Path) -> None:
    engine = make_engine(tmp_path / "th08-modern", b"the real binary\n")
    config = make_config(tmp_path, tape_path, engine=engine,
                         expected_sha="f" * 64)
    factory = Factory(None)
    result = tk.run_takeover(config, client_factory=factory)
    assert factory.calls == 0
    assert result["engine"]["sha_matches"] is False
    assert result["engine"]["sha256"] == tk.sha256_file(engine)
    assert result["engine"]["expected_sha"] == "f" * 64
    assert result["stop"] == "engine_sha_mismatch"
    assert result["exit_code"] == tk.EXIT_REFUSED
    assert result["client"]["started"] is False


def test_11b_the_expected_sha_is_case_insensitive(tmp_path: Path,
                                                  tape_path: Path) -> None:
    engine = make_engine(tmp_path / "th08-modern", b"upper case\n")
    config = make_config(tmp_path, tape_path, engine=engine, entry_frame=64,
                         cap_frames=4, expected_sha=tk.sha256_file(engine).upper())
    result, _, _ = run(config, FakeHarness())
    assert result["engine"]["sha_matches"] is True
    assert result["exit_code"] == tk.EXIT_OK


# ---------------------------------------------------------------------------
# 12 the checkpoint arm requires its checkpoint
# ---------------------------------------------------------------------------
def test_12_checkpoint_arm_without_a_checkpoint_is_refused(
        tmp_path: Path, tape_path: Path) -> None:
    """Item 12: the arm now runs, so its one required argument is the gate.

    ``--arm checkpoint`` used to raise ``CheckpointArmNotImplemented``; it is
    implemented, and the refusal this item pins is the missing ``--checkpoint``
    (nothing loaded, nothing started, no policy identity claimed).
    """

    config = make_config(tmp_path, tape_path, arm="checkpoint")
    factory = Factory(None)
    result = tk.run_takeover(config, client_factory=factory)
    assert result["stop"] == "checkpoint_missing"
    assert result["exit_code"] == tk.EXIT_REFUSED
    assert "checkpoint" in result["failure"]["detail"]
    assert factory.calls == 0
    assert result["client"]["started"] is False
    assert result["checkpoint_arm"] is None
    assert result["checkpoint_sha256"] == ""
    assert result["obs_spec_version"] == tk.OBS_SPEC_NONE
    assert read_result(config)["stop"] == "checkpoint_missing"


def test_12b_the_cli_refuses_a_checkpoint_arm_without_a_checkpoint(
        tmp_path: Path, tape_path: Path) -> None:
    engine = make_engine(tmp_path / "th08-modern")
    data_dir = make_data_dir(tmp_path / "data")
    argv = ["run", "--engine", str(engine), "--expected-sha", tk.sha256_file(engine),
            "--data-dir", str(data_dir), "--tape-slots", str(tape_path),
            "--seed", "12", "--entry-frame", "64", "--arm", "checkpoint",
            "--cap-frames", "8", "--out", str(tmp_path / "out")]
    assert tk.main(argv) == tk.EXIT_REFUSED
    parsed = tk.build_parser().parse_args(argv)
    assert tk.TakeoverConfig.from_args(parsed).checkpoint is None
    assert read_result(tk.TakeoverConfig.from_args(parsed))["stop"] == "checkpoint_missing"
    # a checkpoint on a non-checkpoint arm is refused too (the arms must not mix)
    mixed = make_config(tmp_path, tape_path, arm="tape",
                        checkpoint=make_checkpoint(tmp_path / "mix.pt"))
    outcome = tk.run_takeover(mixed, client_factory=Factory(None))
    assert outcome["stop"] == "arm_invalid"
    assert outcome["exit_code"] == tk.EXIT_REFUSED


# ---------------------------------------------------------------------------
# 13..18 the tape contract, the seed, the mute bytes, a death
# ---------------------------------------------------------------------------
def test_13_a_broken_hash_chain_is_refused(tmp_path: Path) -> None:
    path = write_tape(tmp_path / "broken.jsonl", slots=8)
    rows = rows_of(path)
    rows[3]["hash_pre"] = "0" * 16
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8")
    with pytest.raises(tk.TapeError) as info:
        tk.load_tape(path)
    assert "hash chain is broken" in str(info.value)
    assert info.value.stop == "tape_invalid"


def test_14_an_off_grid_tape_is_refused(tmp_path: Path) -> None:
    path = write_tape(tmp_path / "offgrid.jsonl", slots=4)
    rows = rows_of(path)
    rows[2]["decision_frame"] = 24          # skips frame 22
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8")
    with pytest.raises(tk.TapeError) as info:
        tk.load_tape(path)
    assert "2-frame grid is broken" in str(info.value)


def test_15_a_foreign_seed_is_refused(tmp_path: Path, tape_path: Path) -> None:
    config = make_config(tmp_path, tape_path, seed=10)
    factory = Factory(None)
    result = tk.run_takeover(config, client_factory=factory)
    assert result["stop"] == "seed_invalid"
    assert result["exit_code"] == tk.EXIT_REFUSED
    assert factory.calls == 0


def test_16_an_audible_cfg_is_refused(tmp_path: Path, tape_path: Path) -> None:
    data_dir = make_data_dir(tmp_path / "audible", audible=True)
    config = make_config(tmp_path, tape_path, data_dir=data_dir)
    factory = Factory(None)
    result = tk.run_takeover(config, client_factory=factory)
    assert result["stop"] == "mute_refused"
    assert result["exit_code"] == tk.EXIT_REFUSED
    assert result["mute"]["before"]["muted"] is False
    assert result["mute"]["before"]["bytes"]["playSounds"] == 1
    assert factory.calls == 0


def test_17_a_death_ends_the_window(tmp_path: Path, tape_path: Path) -> None:
    log = tmp_path / "death.jsonl"
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=40,
                         decision_log=log)
    result, fake, _ = run(config, FakeHarness(death_frame=70))
    assert result["deaths_at_entry"] == 0.0
    assert result["deaths"] == 1.0
    assert result["deaths_delta"] == 1.0
    assert result["stop"] == "legal_death"
    assert result["lives_final"] == 2.0
    # decisions at frames 64/66/68; the death shows up at frame 70, so three
    # decisions (6 frames) were taken and the third one is the one that ends it
    assert result["frames_takeover"] == 6
    assert result["decisions"] == 3
    assert result["exit_code"] == tk.EXIT_OK
    rows = rows_of(log)[1:]
    assert rows[-1]["end"] == "legal_death"
    assert [row["end"] for row in rows[:-1]] == [""] * (len(rows) - 1)


def test_18_constant_arm_requires_a_legal_candidate(tmp_path: Path,
                                                    tape_path: Path) -> None:
    for cand, stop in ((None, "constant_cand_invalid"),
                       (18, "constant_cand_invalid"),
                       (-1, "constant_cand_invalid")):
        config = make_config(tmp_path, tape_path, arm="constant",
                             constant_cand=cand,
                             out=tmp_path / f"out_{cand}")
        factory = Factory(None)
        result = tk.run_takeover(config, client_factory=factory)
        assert result["stop"] == stop, cand
        assert result["exit_code"] == tk.EXIT_REFUSED
        assert factory.calls == 0
    # --constant-cand on the tape arm is refused too (the arms must not be mixed)
    config = make_config(tmp_path, tape_path, arm="tape", constant_cand=3,
                         out=tmp_path / "out_mixed")
    result = tk.run_takeover(config, client_factory=Factory(None))
    assert result["stop"] == "constant_cand_invalid"


def test_19_a_foreign_reset_is_not_playable(tmp_path: Path, tape_path: Path) -> None:
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=4)
    result, fake, _ = run(config, FakeHarness(ready="0"))
    assert result["stop"] == "protocol_error"
    assert result["exit_code"] == tk.EXIT_TECHNICAL
    assert "did not reach a playable state" in result["failure"]["detail"]
    assert fake.steps == []


def test_20_an_engine_exit_ends_the_window(tmp_path: Path, tape_path: Path) -> None:
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=40)
    result, fake, _ = run(config, FakeHarness(exit_at_frame=68))
    assert result["stop"] == "engine_exit"
    assert result["exit_code"] == tk.EXIT_TECHNICAL
    assert result["frames_takeover"] == 4
    assert result["decisions"] == 2
    # 64 prefix frames + 4 window frames + the refused STEP that reported the exit
    assert len(fake.steps) == 64 + 4 + 1


# ===========================================================================
# The checkpoint arm: an obs-v3 actor loaded from a checkpoint file
# ===========================================================================
def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def history_cands_from_rows(rows) -> list:
    """The candidate each 8x18 history row is one-hot for (``None`` for padding)."""

    out: list = []
    for row in np.asarray(rows):
        positions = np.flatnonzero(row)
        out.append(None if positions.size == 0 else int(positions[0]))
    return out


# ---------------------------------------------------------------------------
# 21 (1) the checkpoint sha256 gate
# ---------------------------------------------------------------------------
def test_21_a_checkpoint_sha_mismatch_is_refused(tmp_path: Path,
                                                 tape_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path / "ckpt.pt")
    config = make_config(tmp_path, tape_path, arm="checkpoint",
                         checkpoint=checkpoint, checkpoint_sha256="0" * 64)
    factory = Factory(None)             # a client here is a failure
    result = tk.run_takeover(config, client_factory=factory)
    assert factory.calls == 0
    assert result["stop"] == "checkpoint_sha_mismatch"
    assert result["exit_code"] == tk.EXIT_REFUSED
    assert tk.sha256_file(checkpoint) in result["failure"]["detail"]
    assert result["checkpoint_arm"] is None
    assert result["checkpoint_sha256"] == ""       # no identity was established
    assert result["client"]["started"] is False
    assert read_result(config)["stop"] == "checkpoint_sha_mismatch"


def test_21b_a_correct_checkpoint_sha256_is_accepted(tmp_path: Path,
                                                     tape_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path / "ckpt.pt")
    sha = tk.sha256_file(checkpoint)
    config = make_config(tmp_path, tape_path, arm="checkpoint", checkpoint=checkpoint,
                         checkpoint_sha256=sha.upper())
    policy = tk.load_policy(config, loader=policy_loader(RecordingModel(cand=5)))
    assert policy.sha256 == sha
    assert policy.sha_matches is True
    assert policy.sha_expected == sha           # normalised to lower case
    # and an absent --checkpoint-sha256 is not a mismatch (it is simply not asked)
    unasked = make_config(tmp_path, tape_path, arm="checkpoint",
                          checkpoint=checkpoint, out=tmp_path / "out_unasked")
    assert tk.load_policy(unasked, loader=policy_loader(RecordingModel())).sha_matches is True


# ---------------------------------------------------------------------------
# 22 (2) the obs-spec gate
# ---------------------------------------------------------------------------
def test_22_another_obs_spec_is_refused(tmp_path: Path, tape_path: Path) -> None:
    model = RecordingModel(cand=5)
    result, fake, _ = run_checkpoint(
        tmp_path, tape_path, model=model,
        loader=policy_loader(model, obs_spec_digest="0" * 64))
    assert result["stop"] == "obs_spec_mismatch"
    assert result["exit_code"] == tk.EXIT_REFUSED
    assert model.calls == 0
    assert result["checkpoint_arm"] is None
    assert "obs spec digest" in result["failure"]["detail"]

    other = RecordingModel(cand=5)
    result_v1, _, _ = run_checkpoint(
        tmp_path, tape_path, model=other, out_name="out_v1",
        loader=policy_loader(other, obs_spec_version="obs-v1"))
    assert result_v1["stop"] == "obs_spec_mismatch"
    assert other.calls == 0

    # the real loader refuses a real checkpoint whose digest was edited, and the
    # refusal is mapped to the same stop (not to a generic load failure)
    import torch

    from touhou_rl.v400.student.model_v3 import StudentV3, save_checkpoint_v3

    mutated = save_checkpoint_v3(tmp_path / "mutated.pt", StudentV3())
    payload = torch.load(mutated, map_location="cpu", weights_only=False)
    payload["obs_spec_digest"] = "0" * 64
    torch.save(payload, mutated)
    config = make_config(tmp_path, tape_path, arm="checkpoint", checkpoint=mutated,
                         out=tmp_path / "out_mutated")
    outcome = tk.run_takeover(config, client_factory=Factory(None))
    assert outcome["stop"] == "obs_spec_mismatch"
    assert outcome["exit_code"] == tk.EXIT_REFUSED


# ---------------------------------------------------------------------------
# 23 (3) no pixels at all: fail closed, before anything is replayed
# ---------------------------------------------------------------------------
def test_23_no_pixels_fails_closed_before_any_handover(tmp_path: Path,
                                                       tape_path: Path) -> None:
    log = tmp_path / "nopixels.jsonl"
    result, fake, model = run_checkpoint(
        tmp_path, tape_path, entry_frame=64, cap_frames=8, decision_log=log,
        fake_kwargs={"pixels_mode": "checksum"})
    assert result["stop"] == "pixels_unavailable"
    assert result["exit_code"] == tk.EXIT_TECHNICAL
    assert result["decisions"] == 0
    assert result["frames_takeover"] == 0
    assert result["entry_verified"] is False
    assert model.calls == 0
    assert fake.steps == []                     # not even the prefix was replayed
    assert fake.started is True and fake.closed is True
    assert result["pixels"]["source"] is None
    assert "checksum" in result["failure"]["detail"]
    assert result["reset"] is None
    assert not log.exists()                     # no log for a window that never ran


def test_23b_a_client_without_any_pixel_route_is_refused(tmp_path: Path,
                                                        tape_path: Path) -> None:
    result, fake, model = run_checkpoint(
        tmp_path, tape_path, fake_kwargs={"attach_error": RuntimeError("no shm")})
    assert result["stop"] == "pixels_unavailable"
    assert model.calls == 0
    assert "no shm" in result["failure"]["detail"]      # both routes named
    assert "pixels()" in result["failure"]["detail"] or "SHM" in result["failure"]["detail"]


# ---------------------------------------------------------------------------
# 24 (3b) a block that never publishes (or stops advancing) is refused
# ---------------------------------------------------------------------------
def test_24_a_shared_block_without_published_pixels_is_refused(
        tmp_path: Path, tape_path: Path) -> None:
    result, fake, model = run_checkpoint(tmp_path, tape_path, entry_frame=64,
                                         cap_frames=8, block_kwargs={"flags": 0})
    assert result["stop"] == "pixels_unavailable"
    assert result["exit_code"] == tk.EXIT_TECHNICAL
    assert result["decisions"] == 0 and model.calls == 0
    assert len(fake.steps) == 1                 # the probe fires after the first STEP
    assert "FLAG_PIXELS_VALID" in result["failure"]["detail"]
    assert result["pixels"]["probe_frames"] == []

    stale, stale_fake, stale_model = run_checkpoint(
        tmp_path, tape_path, out_name="out_stale", block_kwargs={"stale": True})
    assert stale["stop"] == "pixels_unavailable"
    assert "already seen" in stale["failure"]["detail"]
    assert stale["decisions"] == 0 and stale_model.calls == 0


# ---------------------------------------------------------------------------
# 25 (4) the history is seeded from the tape's executed_cand
# ---------------------------------------------------------------------------
def test_25_the_action_history_is_seeded_from_the_tape(tape: dict) -> None:
    plan16 = tk.entry_plan(tape, 16)
    history16, cands16 = tk.seed_action_history(plan16)
    assert cands16 == [7] * LEAD_IN
    assert history16.rows().shape == (8, 18)
    assert history_cands_from_rows(history16.rows()) == [7] * LEAD_IN

    plan64 = tk.entry_plan(tape, 64)
    history64, cands64 = tk.seed_action_history(plan64)
    expected = ([7] * LEAD_IN
                + [CAND_CYCLE[index % len(CAND_CYCLE)] for index in range(24)])
    assert cands64 == expected == tk.prefix_action_cands(plan64)
    assert len(cands64) == plan64["entry_index"] == 32
    tail = expected[-8:]
    assert history_cands_from_rows(history64.rows()) == tail
    rows = history64.rows()
    assert rows.dtype == np.uint8
    assert [int(row.sum()) for row in rows] == [1] * 8
    # the seed is exactly the tape's own record, not a re-derivation
    assert tail == [slot.cand for slot in plan64["prefix_slots"][-8:]]


# ---------------------------------------------------------------------------
# 26 (4b) and continues with the policy's own candidates after the entry
# ---------------------------------------------------------------------------
def test_26_the_history_continues_with_the_policy_candidates(
        tmp_path: Path, tape_path: Path) -> None:
    model = RecordingModel(cand=5)
    result, fake, _ = run_checkpoint(tmp_path, tape_path, model=model,
                                     entry_frame=64, cap_frames=8)
    assert model.calls == result["decisions"] == 4
    tape_tail = [slot.cand for slot in
                 tk.entry_plan(tk.load_tape(tape_path), 64)["prefix_slots"][-8:]]
    seen = [history_cands_from_rows(hist[0].numpy()) for _play, _hud, hist in model.inputs]
    assert seen[0] == tape_tail
    assert seen[1] == tape_tail[1:] + [5]
    assert seen[2] == tape_tail[2:] + [5, 5]
    assert seen[3] == tape_tail[3:] + [5, 5, 5]
    assert result["history_pushes"] == 4
    assert result["checkpoint_arm"]["history_seed"]["decisions"] == 32
    assert result["checkpoint_arm"]["history_seed"]["buffer_shape"] == [8, 18]


# ---------------------------------------------------------------------------
# 27 (5) two frames per decision, one render per frame, four-frame stack
# ---------------------------------------------------------------------------
def test_27_two_frames_per_decision_and_one_render_per_frame(
        tmp_path: Path, tape_path: Path) -> None:
    log = tmp_path / "cadence.jsonl"
    result, fake, model = run_checkpoint(tmp_path, tape_path, model=RecordingModel(cand=5),
                                         entry_frame=64, cap_frames=8,
                                         decision_log=log)
    assert result["frames_prefix"] == 64
    assert result["decisions"] == 4
    assert result["frames_takeover"] == 8
    assert result["frames"] == 72
    assert len(fake.steps) == 64 + 8
    assert result["renders"] == 4 + 8          # 4-frame stack window + 2 per decision
    assert result["policy_calls"] == 4
    assert result["pixels"]["reads"] == 4 + 8 + 1     # + the discarded frame-1 probe
    assert result["pixels"]["probe_frames"] == [1]
    assert fake.block.pixel_reads == 4 + 8 + 1
    assert (result["checkpoint_arm"]["window"]["rendered_frames"]
            == [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72])
    for play, hud, hist in model.inputs:
        assert tuple(play.shape) == (1, 4, 112, 96)
        assert tuple(hud.shape) == (1, 1, 56, 160)
        assert tuple(hist.shape) == (1, 8, 18)
    # the stack holds four *different* consecutive frames (not four copies)
    planes = [model.inputs[1][0][0, index].numpy() for index in range(4)]
    assert all(not np.array_equal(planes[left], planes[right])
               for left in range(4) for right in range(left + 1, 4))
    rows = rows_of(log)[1:]
    assert [row["frames_per_decision"] for row in rows] == [2, 2, 2, 2]
    assert [len(row["masks"]) for row in rows] == [2, 2, 2, 2]
    assert [row["decision"] for row in rows] == [32, 33, 34, 35]
    assert [row["frame"] for row in rows] == [64, 66, 68, 70]


# ---------------------------------------------------------------------------
# 28 (6) the tensors are bit-identical to tools/rl/v400_student_eval.py
# ---------------------------------------------------------------------------
def _load_eval_module():
    """``tools/rl/v400_student_eval.py`` by path (read, never modified)."""

    spec = importlib.util.spec_from_file_location("v400_student_eval_v832_align",
                                                  EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _EvalEngine:
    """The eval's scripted engine: a frame counter, one STEP per frame, a STATE."""

    def __init__(self, shm_name: str = "v832-align") -> None:
        self.frame = 0
        self.shm_name = shm_name
        self.masks: list[int] = []
        self.steps = 0
        self.closed = False

    def start(self) -> None:
        return None

    def hello(self) -> dict:
        return {"role": "env"}

    def attach_shm(self) -> dict:
        return {"active": "1"}

    def readback(self, enabled: bool) -> dict:
        return {"readback": "1" if enabled else "0"}

    def reset(self, **kwargs) -> dict:
        self.frame = 0
        return {"ready": "1"}

    def request(self, line: str, timeout: float | None = None) -> str:
        if str(line) == "STATE":
            return f"OK frame={self.frame}"
        raise AssertionError(f"the eval sent an unexpected command: {line!r}")

    def step(self, mask: int, want: int) -> dict:
        self.steps += 1
        self.masks.append(int(mask))
        self.frame += int(want)
        return {"advanced": str(int(want)), "exit": "0", "game_over": "0"}

    def close(self) -> None:
        self.closed = True


def _eval_state(reply: str) -> dict:
    frame = int(str(reply).rsplit("=", 1)[-1])
    return {"deaths": 0.0, "lives_remaining": 3.0, "bombs_remaining": 2.0,
            "score": 0, "stage": 0, "frame": frame, "gameplay_frame": frame}


def test_28_the_tensors_match_v400_student_eval_bit_for_bit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Item 6: same pixels, same frame, both real code paths, compare the tensors.

    The eval's own ``run_episode`` (``obs_v3=True``, ``obs_window="consecutive"``)
    and this runner's checkpoint arm are driven on the *same* deterministic RGBA
    frames.  The tape's 8 lead-in decisions carry the eval's first 8 emitted
    candidates, so the action history agrees at the entry, and the takeover starts
    at frame 16 where the eval's decision 8 sits.  Every tensor ``model.act``
    receives is then compared with ``torch.equal``.
    """

    import torch

    from touhou_rl.v400.student import obs_spec_v3

    ev = _load_eval_module()
    cands = [7, 0, 3, 12, 16, 1, 5, 9, 2, 11, 4, 6, 13, 14, 15, 10, 8, 17,
             3, 16, 0, 9, 7, 12]
    max_frames = 12 * ev.DECISION_FRAMES

    # -- the eval's episode -------------------------------------------------
    eval_engine = _EvalEngine()

    class EvalBlock:
        """The eval maps ``SharedBlock(client.shm_name or shm_name)``."""

        def __init__(self, _name: str) -> None:
            self.engine = eval_engine

        def header(self) -> dict:
            return {"version": 1, "frame": self.engine.frame, "state_hash": 0,
                    "width": 640, "height": 480, "state_bytes": 100,
                    "flags": FLAG_PIXELS_VALID}

        def pixels(self) -> np.ndarray:
            return fake_pixels(self.engine.frame)

        def close(self) -> None:
            pass

    eval_model = RecordingModel(cands)
    monkeypatch.setattr(ev, "EnvHarnessClient", lambda *a, **k: eval_engine)
    monkeypatch.setattr(ev, "SharedBlock", EvalBlock)
    monkeypatch.setattr(ev, "state_from_reply", _eval_state)
    record = ev.run_episode(
        engine=str(make_engine(tmp_path / "eval_engine.bin", b"fake engine\n")),
        data_dir=str(tmp_path),
        checkpoint_path=str(make_checkpoint(tmp_path / "eval_ckpt.pt",
                                            b"fake checkpoint\n")),
        model=eval_model, seed=12, stage=0, difficulty="normal",
        max_frames=max_frames, shm_name="v832-align", obs_v3=True,
        obs_window="consecutive", arch="v3")
    assert record["frames"] == max_frames
    assert record["obs_spec_digest"] == tk.OBS_SPEC_V3_DIGEST
    assert eval_engine.steps == max_frames
    assert eval_model.calls == 12

    # -- the takeover's window, at the eval's decision 8 (frame 16) ----------
    tape_path = write_tape(tmp_path / "align_tape.jsonl", slots=64,
                           history_cands=cands[:LEAD_IN])
    takeover_model = RecordingModel(cands, offset=LEAD_IN)
    result, fake, _ = run_checkpoint(tmp_path, tape_path, model=takeover_model,
                                     entry_frame=16, cap_frames=8,
                                     out_name="out_align")
    assert result["entry_verified"] is True
    assert result["decisions"] == 4
    assert takeover_model.calls == 4

    for offset in range(4):
        play_eval, hud_eval, hist_eval = eval_model.inputs[LEAD_IN + offset]
        play_arm, hud_arm, hist_arm = takeover_model.inputs[offset]
        assert tuple(play_arm.shape) == tuple(play_eval.shape) == (1, 4, 112, 96)
        assert tuple(hud_arm.shape) == tuple(hud_eval.shape) == (1, 1, 56, 160)
        assert tuple(hist_arm.shape) == tuple(hist_eval.shape) == (1, 8, 18)
        assert play_arm.dtype == play_eval.dtype == torch.float32
        assert torch.equal(play_arm, play_eval), f"play @ decision {8 + offset}"
        assert torch.equal(hud_arm, hud_eval), f"hud @ decision {8 + offset}"
        assert torch.equal(hist_arm, hist_eval), f"hist @ decision {8 + offset}"
    # the same masks went to the engine, so the frames fed to both were the same
    assert fake.steps[LEAD_IN * 2:] == eval_engine.masks[LEAD_IN * 2:]

    # -- and the stack/normalisation is independently re-derived ------------
    first_play = takeover_model.inputs[0][0][0].numpy()
    expected = np.stack([obs_spec_v3.encode_planes_v3(fake_pixels(frame))[0]
                         for frame in (13, 14, 15, 16)]).astype(np.float32) / 255.0
    assert np.array_equal(first_play, expected)
    assert float(first_play.min()) >= 0.0 and float(first_play.max()) <= 1.0
    assert float(first_play.sum()) > 0.0          # real pixels, not a zero image
    first_hud = takeover_model.inputs[0][1][0].numpy()
    hud_expected = obs_spec_v3.normalize_hud(
        obs_spec_v3.encode_planes_v3(fake_pixels(16))[1])
    assert np.array_equal(first_hud, hud_expected)


# ---------------------------------------------------------------------------
# 29 (7) result.json carries the arm identity
# ---------------------------------------------------------------------------
def test_29_result_json_carries_the_checkpoint_arm_identity(
        tmp_path: Path, tape_path: Path) -> None:
    result, fake, model = run_checkpoint(tmp_path, tape_path, entry_frame=64,
                                         cap_frames=8)
    on_disk = read_json(tmp_path / "out_ckpt" / "result.json")
    for key in tk.RESULT_REQUIRED_KEYS:
        assert key in on_disk, key
    assert on_disk["arm"] == "checkpoint"
    assert on_disk["checkpoint"] == (tmp_path / "ckpt_v3.pt").as_posix()
    assert on_disk["checkpoint_sha256"] == tk.sha256_file(tmp_path / "ckpt_v3.pt")
    assert on_disk["obs_spec_version"] == "obs-v3"
    assert on_disk["obs_spec_digest"] == tk.OBS_SPEC_V3_DIGEST
    assert on_disk["frames_per_decision"] == 2
    decision_frames = on_disk["policy_input"]
    assert decision_frames.startswith("play 4x112x96")
    assert "pixels" in decision_frames and "action history" in decision_frames
    assert "no privileged state" in decision_frames
    arm = on_disk["checkpoint_arm"]
    for key in ("arm", "checkpoint", "checkpoint_sha256", "obs_spec_version",
                "obs_spec_digest", "policy_input", "frames_per_decision",
                "parameters", "history_seed", "window", "action_contract",
                "loader", "device"):
        assert key in arm, key
    assert arm["arm"] == "checkpoint"
    assert arm["obs_spec_digest"] == tk.OBS_SPEC_V3_DIGEST
    assert arm["frames_per_decision"] == 2
    assert arm["history_seed"]["decisions"] == 32
    assert arm["history_seed"]["buffer_shape"] == [8, 18]
    assert arm["window"]["decisions"] == 4
    assert arm["window"]["renders"] == 12
    assert arm["window"]["policy_calls"] == 4
    assert arm["window"]["history_pushes"] == 4
    assert on_disk["pixels"]["source"] == "shm"
    assert on_disk["pixels"]["readback"]["readback"] == "1"
    assert on_disk["renders"] == 12
    assert on_disk["policy_calls"] == 4
    assert on_disk["exit_code"] == tk.EXIT_OK


# ---------------------------------------------------------------------------
# 30 (8) the decision-log header and rows
# ---------------------------------------------------------------------------
def test_30_the_log_header_and_rows_carry_the_checkpoint_fields(
        tmp_path: Path, tape_path: Path) -> None:
    log = tmp_path / "ckpt.jsonl"
    result, fake, model = run_checkpoint(tmp_path, tape_path, entry_frame=64,
                                         cap_frames=8, decision_log=log)
    lines = rows_of(log)
    assert len(lines) == result["decisions"] + 1 == 5
    header = lines[0]
    assert header["schema"] == tk.SCHEMA_LOG_HEADER
    assert header["arm"] == "checkpoint"
    assert header["checkpoint"] == (tmp_path / "ckpt_v3.pt").as_posix()
    assert header["checkpoint_sha256"] == tk.sha256_file(tmp_path / "ckpt_v3.pt")
    assert header["obs_spec_version"] == "obs-v3"
    assert header["obs_spec_digest"] == tk.OBS_SPEC_V3_DIGEST
    assert header["frames_per_decision"] == 2
    assert header["policy_input"] == tk.POLICY_INPUT_CHECKPOINT
    assert header["entry_frame"] == 64
    assert header["entry_hash"] == f"{24:016x}"
    for row in lines[1:]:
        assert row["schema"] == tk.SCHEMA_LOG_ROW
        assert row["cand"] == 5                       # the policy's own output
        assert row["mask"] == int(cand_to_mask(5, 0))
        assert row["masks"] == [int(cand_to_mask(5, 0))] * 2
        assert row["frames_per_decision"] == 2
        assert row["events"] == ["ENEMY_SPAWNED"]
        assert row["end"] == ""


# ---------------------------------------------------------------------------
# 31 (9) an entry-hash failure starts no policy decision and writes no log
# ---------------------------------------------------------------------------
def test_31_an_entry_hash_failure_starts_no_policy_decision(
        tmp_path: Path, tape_path: Path) -> None:
    log = tmp_path / "entry_fail.jsonl"
    checkpoint = make_checkpoint(tmp_path / "ckpt.pt")
    config = make_config(tmp_path, tape_path, arm="checkpoint", entry_frame=64,
                         cap_frames=8, decision_log=log, checkpoint=checkpoint,
                         checkpoint_sha256=tk.sha256_file(checkpoint))
    model = RecordingModel(cand=5)
    fake = FakeHarness()
    fake.hash_overrides_by_call[1 + 24 + 1] = "1111111111111111"   # the entry read
    result = tk.run_takeover(
        config, client_factory=lambda _config: fake,
        block_factory=checkpoint_block_factory(fake),
        policy_loader=policy_loader(model),
        clock=iter([0.0, 0.5, 1.0]).__next__)
    assert result["entry_verified"] is False
    assert result["stop"] == "entry_hash_mismatch"
    assert result["exit_code"] == tk.EXIT_ENTRY_FAILED
    assert result["mismatch"]["kind"] == "entry"
    assert model.calls == 0
    assert result["decisions"] == 0
    assert result["frames_takeover"] == 0
    assert result["policy_calls"] is None
    assert len(fake.steps) == 64                  # the prefix, then the entry failed
    assert not log.exists()
    assert read_json(tmp_path / "out" / "result.json")["stop"] == "entry_hash_mismatch"


# ---------------------------------------------------------------------------
# 32 (10) the SHOOT pulse still applies to the checkpoint arm
# ---------------------------------------------------------------------------
def test_32_the_shoot_pulse_still_applies_to_the_checkpoint_arm(
        tmp_path: Path, tape_path: Path) -> None:
    log = tmp_path / "pulse_ckpt.jsonl"
    result, fake, model = run_checkpoint(tmp_path, tape_path,
                                         model=RecordingModel(cand=5),
                                         entry_frame=64, cap_frames=192,
                                         decision_log=log)
    assert result["decisions"] == 96
    assert result["frames_takeover"] == 192
    assert result["exit_code"] == tk.EXIT_OK
    rows = rows_of(log)[1:]
    assert len(rows) == 96
    full = int(cand_to_mask(5, 0))
    held = full & ~1
    pulses = [row["decision"] for row in rows if row["masks"][0] != row["masks"][1]]
    assert pulses == [47, 95]
    for row in rows:
        assert row["cand"] == 5
        assert row["masks"] == ([full, held] if row["decision"] in pulses
                                else [full, full])
    assert release_due(47, 48) is True and release_due(46, 48) is False
    # the engine saw the release too (the log is not the only witness)
    offset = 2 * (47 - 32)
    assert fake.steps[64 + offset:64 + offset + 2] == [full, held]
    assert fake.steps[64 + offset - 2:64 + offset] == [full, full]


# ---------------------------------------------------------------------------
# 33 (11) the tape/constant arms are unchanged
# ---------------------------------------------------------------------------
#: The frozen row shape at the ticket's contract: every field of
#: ``v400-student-eval-decision-1`` plus ``masks`` (one mask per frame) and
#: ``enemies`` (the referee-side reading; never a policy input).
REGRESSION_ROW_KEYS = {
    "schema", "seed", "repeat", "decision", "frame", "frames_per_decision",
    "cand", "mask", "masks", "deaths", "lives", "bombs", "score", "stage",
    "events", "end", "enemies",
}


def test_33_the_tape_and_constant_arms_are_unchanged(tmp_path: Path,
                                                     tape_path: Path) -> None:
    log = tmp_path / "tape_regression.jsonl"
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8,
                         decision_log=log)
    result, fake, _ = run(config, FakeHarness())
    assert result["exit_code"] == tk.EXIT_OK and result["decisions"] == 4
    assert fake.shm_attaches == 0 and fake.readback_calls == 0
    assert fake.pixel_reads == 0
    assert result["pixels"] is None
    assert result["renders"] is None and result["policy_calls"] is None
    assert result["checkpoint"] == "" and result["checkpoint_sha256"] == ""
    assert result["checkpoint_arm"] is None
    assert result["obs_spec_version"] == tk.OBS_SPEC_NONE
    assert result["obs_spec_digest"] == ""
    assert result["policy_input"] == tk.POLICY_INPUT_NONE
    assert result["frames_per_decision"] == 2
    lines = rows_of(log)
    assert lines[0]["checkpoint"] == "" and lines[0]["obs_spec_digest"] == ""
    assert lines[0]["obs_spec_version"] == \
        "none (a tape/constant takeover builds no observation)"
    assert lines[0]["policy_input"] == tk.POLICY_INPUT_NONE
    for row in lines[1:]:
        assert set(row) == REGRESSION_ROW_KEYS           # the row shape is frozen

    constant = make_config(tmp_path, tape_path, arm="constant", constant_cand=5,
                           entry_frame=64, cap_frames=8, out=tmp_path / "out_const")
    result_c, fake_c, _ = run(constant, FakeHarness())
    assert result_c["exit_code"] == tk.EXIT_OK and result_c["decisions"] == 4
    assert fake_c.shm_attaches == 0 and fake_c.pixel_reads == 0
    assert result_c["checkpoint_arm"] is None
    assert result_c["pixels"] is None


# ---------------------------------------------------------------------------
# 34 (12) the pixels()-fallback source produces the same tensors as SHM
# ---------------------------------------------------------------------------
def test_34_the_pixels_fallback_yields_the_same_tensors(tmp_path: Path,
                                                       tape_path: Path) -> None:
    import torch

    shm_result, shm_fake, shm_model = run_checkpoint(
        tmp_path, tape_path, entry_frame=64, cap_frames=4, out_name="out_shm")
    fallback, fb_fake, fb_model = run_checkpoint(
        tmp_path, tape_path, entry_frame=64, cap_frames=4, out_name="out_fallback",
        fake_kwargs={"pixels_mode": "array"})
    assert shm_result["pixels"]["source"] == "shm"
    assert fallback["pixels"]["source"] == "pixels()"
    assert shm_fake.shm_attaches == 1 and fb_fake.shm_attaches == 1
    assert shm_fake.readback_calls == 1 and fb_fake.readback_calls == 0
    assert fallback["decisions"] == 2 and shm_result["decisions"] == 2
    # 1 discarded frame-1 probe + 4 prefix renders + 4 hand-over renders
    assert fallback["pixels"]["reads"] == 4 + 4 + 1
    # the pixels() source costs one extra capability read at open (the checksum
    # probe) and then one PIXELS command per read
    assert fb_fake.pixel_reads == 4 + 4 + 1 + 1
    assert shm_fake.block.pixel_reads == 4 + 4 + 1
    for offset in range(2):
        for left, right in zip(shm_model.inputs[offset], fb_model.inputs[offset]):
            assert torch.equal(left, right)


# ---------------------------------------------------------------------------
# 35 the real load_checkpoint_v3 path
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def real_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real (canonical) obs-v3 checkpoint, written by the training-side writer."""

    from touhou_rl.v400.student.model_v3 import StudentV3, save_checkpoint_v3

    return save_checkpoint_v3(tmp_path_factory.mktemp("real_ckpt") / "ckpt_v3.pt",
                              StudentV3(), train={"tag": "v832-takeover-test"})


def test_35_the_real_loader_is_used_and_a_wrong_file_is_refused(
        tmp_path: Path, tape_path: Path, real_checkpoint: Path) -> None:
    from touhou_rl.v400.student.model_v3 import load_checkpoint_v3

    sha = tk.sha256_file(real_checkpoint)
    config = make_config(tmp_path, tape_path, arm="checkpoint",
                         checkpoint=real_checkpoint, checkpoint_sha256=sha)
    policy = tk.load_policy(config)
    model, payload = load_checkpoint_v3(real_checkpoint)
    assert policy.sha256 == sha and policy.sha_matches is True
    assert policy.obs_spec_version == tk.OBS_SPEC_V3_VERSION
    assert policy.obs_spec_digest == tk.OBS_SPEC_V3_DIGEST
    assert policy.parameters == payload["parameters"]
    assert policy.arch == "v3"
    assert callable(policy.model.act)
    assert policy.identity()["frames_per_decision"] == 2

    bad = make_checkpoint(tmp_path / "bad.pt", b"this is not a torch checkpoint\n")
    bad_config = make_config(tmp_path, tape_path, arm="checkpoint", checkpoint=bad,
                             out=tmp_path / "out_bad")
    outcome = tk.run_takeover(bad_config, client_factory=Factory(None))
    assert outcome["stop"] == "policy_load_failed"
    assert outcome["exit_code"] == tk.EXIT_REFUSED
    assert outcome["checkpoint_arm"] is None


# ---------------------------------------------------------------------------
# 36 a policy outside the 18-candidate contract stops the window
# ---------------------------------------------------------------------------
def test_36_a_policy_outside_the_action_contract_stops_the_window(
        tmp_path: Path, tape_path: Path) -> None:
    log = tmp_path / "illegal.jsonl"
    result, fake, model = run_checkpoint(tmp_path, tape_path,
                                         model=RecordingModel(cand=18),
                                         entry_frame=64, cap_frames=8,
                                         decision_log=log)
    assert model.calls == 1
    assert result["decisions"] == 0
    assert result["frames_takeover"] == 0
    assert result["stop"] == "policy_output_invalid"
    assert result["exit_code"] == tk.EXIT_TECHNICAL
    assert result["failure"]["detail"]["cand"] == 18
    assert len(fake.steps) == 64             # nothing was committed
    assert len(rows_of(log)) == 1            # the header, no decision row


# ===========================================================================
# The per-decision ``enemies`` reading and ``boss_summary`` (items 37..46)
#
# The reading is referee-side telemetry: it goes into the log and into
# ``result.json``'s aggregate, and it must never become an input of a policy or a
# scoring source (judging uses ``HAZARDS`` row identity; ``ENEMIES`` is paginated).
# ===========================================================================
#: The exact field set of an available reading (a refused one is
#: ``{"available": False, "reason": <the engine's own text>}`` and nothing else).
ENEMIES_KEYS = {"available", "count", "boss_count", "boss_life", "boss_slots",
                "truncated"}
#: The engine's refusal text the refused-probe test replays verbatim.
ENEMIES_REFUSAL = "ERR enemies-unavailable: ENEMIES is not exported by this build"


def enemy_row(*, slot: int = 0, flags1: int = 0x3, life: int = 4878,
              timer: int = 9) -> EnemyRow:
    """One scripted ``ENEMIES`` row (``flags1 & 2`` is the BOSS bit)."""

    return EnemyRow(slot=int(slot), flags1=int(flags1),
                    position=(120.0, 100.0, 0.0), life=int(life), timer=int(timer))


def test_37_the_tape_arm_rows_carry_the_enemies_reading(tmp_path: Path,
                                                        tape_path: Path) -> None:
    """Item 37: the tape arm logs one measured ``enemies`` reading per decision."""

    log = tmp_path / "tape_enemies.jsonl"
    # the reading is taken after each decision's two frames, so the script sees
    # frame 66 for the row at frame 64 -- a per-decision value, not the entry's copy
    fake = FakeHarness(enemies_script=lambda frame: [
        enemy_row(slot=0, life=6000 - 10 * int(frame))])
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8,
                         decision_log=log)
    result, fake, _ = run(config, fake)
    assert result["exit_code"] == tk.EXIT_OK and result["decisions"] == 4
    rows = rows_of(log)[1:]
    assert [row["frame"] for row in rows] == [64, 66, 68, 70]
    for row in rows:
        enemies = row["enemies"]
        assert set(enemies) == ENEMIES_KEYS
        assert enemies["available"] is True
        assert enemies["count"] == 1
        assert enemies["boss_count"] == 1
        assert enemies["boss_slots"] == [0]
        assert enemies["truncated"] is False
    assert [row["enemies"]["boss_life"] for row in rows] == [5340, 5320, 5300, 5280]
    # one page + one census per logged decision, plus the entry's own reading
    assert fake.enemy_reads == 1 + 4 and fake.enemy_count_reads == 4
    assert result["entry_state"]["enemies"]["boss_count"] == 1
    assert result["boss_summary"]["samples"] == 4


def test_38_the_constant_arm_rows_carry_the_enemies_reading(
        tmp_path: Path, tape_path: Path) -> None:
    """Item 38: the constant arm's log carries the same reading."""

    log = tmp_path / "constant_enemies.jsonl"
    config = make_config(tmp_path, tape_path, arm="constant", constant_cand=5,
                         entry_frame=64, cap_frames=8, decision_log=log)
    result, fake, _ = run(config, FakeHarness())
    rows = rows_of(log)[1:]
    assert len(rows) == result["decisions"] == 4
    for row in rows:
        assert set(row["enemies"]) == ENEMIES_KEYS
        assert row["enemies"]["available"] is True
        assert row["enemies"]["boss_life"] == 4878
        assert row["enemies"]["boss_slots"] == [0]
        assert row["enemies"]["truncated"] is False
    assert fake.enemy_reads == 1 + 4


def test_39_the_checkpoint_arm_rows_carry_the_enemies_reading(
        tmp_path: Path, tape_path: Path) -> None:
    """Item 39: the checkpoint arm logs it too, and its policy identity is untouched."""

    log = tmp_path / "ckpt_enemies.jsonl"
    result, fake, _ = run_checkpoint(tmp_path, tape_path, entry_frame=64,
                                     cap_frames=8, decision_log=log)
    rows = rows_of(log)[1:]
    assert len(rows) == result["decisions"] == 4
    for row in rows:
        assert set(row["enemies"]) == ENEMIES_KEYS
        assert row["enemies"]["available"] is True
        assert row["enemies"]["count"] == 1
        assert row["enemies"]["boss_count"] == 1
    assert fake.enemy_reads == 1 + 4
    assert result["policy_input"] == tk.POLICY_INPUT_CHECKPOINT
    assert "enem" not in result["policy_input"].lower()
    assert "enemies" not in rows_of(log)[0]      # the header states no reading


def test_40_a_refused_or_undecodable_enemies_page_is_written_not_raised(
        tmp_path: Path, tape_path: Path) -> None:
    """Item 40: ``available: false`` with the engine's text; no exception, no zero."""

    log = tmp_path / "refused_enemies.jsonl"
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8,
                         decision_log=log)

    def refuse_after_the_entry(frame: int):
        """The entry's own reading succeeds; every decision-time page is refused."""

        if int(frame) > 64:
            raise RuntimeError(ENEMIES_REFUSAL)
        return [enemy_row()]

    result, fake, _ = run(config, FakeHarness(enemies_script=refuse_after_the_entry))
    assert result["exit_code"] == tk.EXIT_OK          # the window still ran
    assert result["stop"] == "frame_cap"
    rows = rows_of(log)[1:]
    assert len(rows) == result["decisions"] == 4
    for row in rows:
        assert row["enemies"] == {"available": False, "reason": ENEMIES_REFUSAL}
    # the census is never asked for a page that was refused
    assert fake.enemy_count_reads == 0
    assert result["boss_summary"] == {"available": False, "first_boss_frame": None,
                                      "boss_life_first": None,
                                      "boss_zero_life_frame": None,
                                      "boss_life_last": None, "samples": 0}
    assert read_json(tmp_path / "out" / "result.json")["boss_summary"]["samples"] == 0
    # the entry's own reading was taken and is untouched by the refusals
    assert result["entry_state"]["enemies"]["boss_count"] == 1

    # a page that cannot be decoded is a refusal as well, not a fabricated zero
    undecodable = tmp_path / "undecodable_enemies.jsonl"
    other = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8,
                        decision_log=undecodable, out=tmp_path / "out_undecodable")
    result_b, _, _ = run(other, FakeHarness(
        enemies_script=lambda frame: ([enemy_row()] if int(frame) <= 64
                                      else ["not-a-row"])))
    assert result_b["exit_code"] == tk.EXIT_OK
    for row in rows_of(undecodable)[1:]:
        assert set(row["enemies"]) == {"available", "reason"}
        assert row["enemies"]["available"] is False
        assert row["enemies"]["reason"].startswith("EngineProtocolError:")
    assert result_b["boss_summary"]["available"] is False


def test_41_no_decision_log_means_no_enemies_probe(tmp_path: Path,
                                                   tape_path: Path) -> None:
    """Item 41: the probe is decided by ``--decision-log`` alone (counted)."""

    plain = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8)
    result_plain, fake_plain, _ = run(plain, FakeHarness())
    assert result_plain["decision_log"] is None
    assert fake_plain.enemy_reads == 1            # the entry's own reading only
    assert fake_plain.enemy_count_reads == 0
    assert result_plain["boss_summary"] == {"available": False,
                                           "first_boss_frame": None,
                                           "boss_life_first": None,
                                           "boss_zero_life_frame": None,
                                           "boss_life_last": None, "samples": 0}

    logged = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8,
                         decision_log=tmp_path / "logged.jsonl",
                         out=tmp_path / "out_logged")
    result_logged, fake_logged, _ = run(logged, FakeHarness())
    assert fake_logged.enemy_reads == 1 + result_logged["decisions"] == 5
    assert fake_logged.enemy_count_reads == result_logged["decisions"] == 4
    assert result_logged["boss_summary"]["samples"] == 4

    # the checkpoint arm is no exception: no log, no extra round trip
    ckpt_result, ckpt_fake, _ = run_checkpoint(tmp_path, tape_path, entry_frame=64,
                                              cap_frames=8, out_name="out_no_log")
    assert ckpt_result["decision_log"] is None
    assert ckpt_fake.enemy_reads == 1 and ckpt_fake.enemy_count_reads == 0


def test_42_boss_summary_without_a_boss(tmp_path: Path, tape_path: Path) -> None:
    """Item 42: no boss listed -> the boss fields stay null, not 0."""

    log = tmp_path / "no_boss.jsonl"
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8,
                         decision_log=log)
    fake = FakeHarness(enemies_script=lambda frame: [
        enemy_row(slot=3, flags1=0x1, life=120)])     # 0x1 = active, no BOSS bit
    result, _, _ = run(config, fake)
    summary = result["boss_summary"]
    assert summary["available"] is True
    assert summary["samples"] == result["decisions"] == 4
    assert summary["first_boss_frame"] is None
    assert summary["boss_life_first"] is None
    assert summary["boss_zero_life_frame"] is None
    # ``boss_life`` over an empty boss list is 0 by the frozen convention; the
    # null ``first_boss_frame`` is what says "no boss was ever listed"
    assert summary["boss_life_last"] == 0
    assert all(row["enemies"]["boss_count"] == 0 for row in rows_of(log)[1:])
    assert read_json(tmp_path / "out" / "result.json")["boss_summary"] == summary


def test_43_boss_summary_with_a_boss_that_keeps_its_life(
        tmp_path: Path, tape_path: Path) -> None:
    """Item 43: boss present, never zero -> first/last life, no zero frame."""

    log = tmp_path / "boss_alive.jsonl"
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8,
                         decision_log=log)
    fake = FakeHarness(enemies_script=lambda frame: [
        enemy_row(slot=0, flags1=tk.ENEMY_FLAG_BOSS, life=6000)])
    result, _, _ = run(config, fake)
    summary = result["boss_summary"]
    assert summary["available"] is True
    assert summary["samples"] == 4
    assert summary["first_boss_frame"] == 64        # the first logged decision
    assert summary["boss_life_first"] == 6000
    assert summary["boss_zero_life_frame"] is None
    assert summary["boss_life_last"] == 6000
    assert [row["enemies"]["boss_life"] for row in rows_of(log)[1:]] == [6000] * 4


def test_44_boss_summary_with_a_boss_that_reaches_zero_life(
        tmp_path: Path, tape_path: Path) -> None:
    """Item 44: the zero-life frame is the first reading that lists a dead boss."""

    log = tmp_path / "boss_dead.jsonl"
    config = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8,
                         decision_log=log)
    # the probe of the row at frame 66 happens at frame 68: 6000, then 0 forever
    fake = FakeHarness(enemies_script=lambda frame: [
        enemy_row(slot=0, flags1=tk.ENEMY_FLAG_BOSS,
                  life=6000 if int(frame) < 68 else 0)])
    result, _, _ = run(config, fake)
    summary = result["boss_summary"]
    assert summary["available"] is True
    assert summary["samples"] == 4
    assert summary["first_boss_frame"] == 64
    assert summary["boss_life_first"] == 6000
    assert summary["boss_zero_life_frame"] == 66    # a listed boss reading 0
    assert summary["boss_life_last"] == 0
    assert [row["enemies"]["boss_life"] for row in rows_of(log)[1:]] == [
        6000, 0, 0, 0]
    assert [row["enemies"]["boss_count"] for row in rows_of(log)[1:]] == [1] * 4


def test_45_a_truncated_page_is_marked_truncated(tmp_path: Path,
                                                 tape_path: Path) -> None:
    """Item 45: a page smaller than the census is ``truncated``; exact/unknown alike."""

    page = [enemy_row(slot=slot, flags1=tk.ENEMY_FLAG_BOSS, life=6000)
            for slot in range(16)]

    truncated = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8,
                            decision_log=tmp_path / "truncated.jsonl")
    fake = FakeHarness(enemies_script=lambda frame: list(page),
                       enemy_total_script=lambda frame: 40)
    result, fake, _ = run(truncated, fake)
    rows = rows_of(truncated.decision_log)
    for row in rows[1:]:
        assert row["enemies"]["available"] is True
        assert row["enemies"]["count"] == 16
        assert row["enemies"]["truncated"] is True
        assert row["enemies"]["boss_count"] == 16
        assert row["enemies"]["boss_slots"] == list(range(16))
    assert fake.enemy_count_reads == result["decisions"] == 4

    # the same page, an exact census: not truncated
    exact = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8,
                        decision_log=tmp_path / "exact.jsonl",
                        out=tmp_path / "out_exact")
    run(exact, FakeHarness(enemies_script=lambda frame: list(page),
                           enemy_total_script=lambda frame: 16))
    assert all(row["enemies"]["truncated"] is False
               for row in rows_of(exact.decision_log)[1:])

    # a client with no census and a full page: a page that might hide a row is
    # never called a census
    blind = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8,
                        decision_log=tmp_path / "blind.jsonl",
                        out=tmp_path / "out_blind")
    run(blind, FakeHarness(enemies_script=lambda frame: list(page),
                           enemy_count_missing=True))
    for row in rows_of(blind.decision_log)[1:]:
        assert row["enemies"]["available"] is True
        assert row["enemies"]["truncated"] is True
    # a refused census is the same conservative answer (never a fake "false")
    refused = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8,
                          decision_log=tmp_path / "census_refused.jsonl",
                          out=tmp_path / "out_census_refused")
    run(refused, FakeHarness(enemies_script=lambda frame: list(page),
                             enemy_total_error=RuntimeError("ERR count-refused")))
    for row in rows_of(refused.decision_log)[1:]:
        assert row["enemies"]["available"] is True
        assert row["enemies"]["truncated"] is True

    # a short page with no census is not truncated (nothing can be hidden)
    short = make_config(tmp_path, tape_path, entry_frame=64, cap_frames=8,
                        decision_log=tmp_path / "short.jsonl",
                        out=tmp_path / "out_short")
    run(short, FakeHarness(enemy_count_missing=True))
    for row in rows_of(short.decision_log)[1:]:
        assert row["enemies"]["count"] == 1
        assert row["enemies"]["truncated"] is False


def test_46_the_enemies_reading_never_reaches_the_policy(tmp_path: Path,
                                                        tape_path: Path) -> None:
    """Item 46: with and without a log, the actor's tensors are bit-identical."""

    import torch

    logged, fake_logged, model_logged = run_checkpoint(
        tmp_path, tape_path, model=RecordingModel(cand=5), entry_frame=64,
        cap_frames=8, decision_log=tmp_path / "discipline.jsonl",
        out_name="out_log")
    plain, fake_plain, model_plain = run_checkpoint(
        tmp_path, tape_path, model=RecordingModel(cand=5), entry_frame=64,
        cap_frames=8, out_name="out_plain")
    # the only difference in protocol traffic is the referee-side probe
    assert fake_logged.enemy_reads == 1 + logged["decisions"] == 5
    assert fake_plain.enemy_reads == 1 and fake_plain.enemy_count_reads == 0
    # identical observations, identical actions: the reading cannot be an input
    assert len(model_logged.inputs) == len(model_plain.inputs) == 4
    for with_log, without in zip(model_logged.inputs, model_plain.inputs):
        for left, right in zip(with_log, without):
            assert torch.equal(left, right)
    assert logged["frames_takeover"] == plain["frames_takeover"] == 8
    assert logged["stop"] == plain["stop"] == "frame_cap"
    assert (logged["policy_input"] == plain["policy_input"]
            == tk.POLICY_INPUT_CHECKPOINT)
    assert "enem" not in logged["policy_input"].lower()
    # the reading is a log/result field, never part of the checkpoint arm identity
    assert "enemies" not in logged["checkpoint_arm"]
    assert "enemies" not in logged["checkpoint_arm"]["window"]
    # and the aggregate says exactly what was sampled
    assert logged["boss_summary"]["samples"] == 4
    assert plain["boss_summary"]["samples"] == 0
    assert plain["boss_summary"]["available"] is False


# ===========================================================================
# --save-obs: the student's own observation planes (WP05/DAgger prerequisite)
# ===========================================================================
#: Every top-level key the measured T04 shard row carries.  A ``--save-obs`` row
#: must carry each one (teacher-label keys as ``null``, see OBS_SLOT_NULL_FIELDS),
#: so a reader that knows the T04 corpus does not have to learn a second shape.
T04_SLOT_KEYS = (
    "schema", "dataset_version", "contract", "obs_contract", "action_contract",
    "shard_key", "episode_seed", "slot_id", "decision_frame", "commit_frames",
    "duration_frames", "obs_render_frames", "history_frames", "history",
    "executed_cand", "bomb", "mask", "executed", "scores", "verified_mask",
    "verified_count", "chosen", "chosen_verified", "chosen_score", "teacher_bomb",
    "margin", "label_source", "label_verified", "hash_pre", "hash_post", "obs",
    "provenance",
)


def run_save_obs(tmp_path: Path, tape_path: Path, *,
                 save_dir: Path | None = None,
                 out_name: str = "out_obs", **kwargs):
    """One scripted checkpoint-arm run with ``--save-obs DIR``."""

    root = Path(save_dir) if save_dir is not None else tmp_path / "corpus"
    result, fake, model = run_checkpoint(tmp_path, tape_path, save_obs=root,
                                         out_name=out_name, **kwargs)
    return result, fake, model, root


# ---------------------------------------------------------------------------
# 47 (1) the corpus exists: one row and one plane per decision
# ---------------------------------------------------------------------------
def test_47_save_obs_writes_slots_planes_meta_and_readme(tmp_path: Path,
                                                         tape_path: Path) -> None:
    result, fake, model, root = run_save_obs(tmp_path, tape_path)
    assert result["exit_code"] == tk.EXIT_OK and result["failure"] is None
    assert result["decisions"] == 4 and result["frames"] == 72
    assert sorted(path.name for path in root.iterdir()) == [
        "README.md", "meta.json", "obs", "slots.jsonl"]
    assert sorted(path.name for path in (root / "obs").iterdir()) == [
        "act_hist.npy", "hud.npy", "play96x112.npy"]
    slots = rows_of(root / "slots.jsonl")
    assert len(slots) == result["decisions"] == 4
    for name, file_name, shape, dtype in tk.OBS_PLANES:
        array = np.load(root / "obs" / file_name)
        assert array.dtype == np.dtype(dtype), name
        assert array.shape == (len(slots), *shape), name
    meta = read_json(root / "meta.json")
    assert meta["schema"] == tk.SCHEMA_OBS_META
    saved = result["save_obs"]
    assert saved["ok"] is True and saved["complete"] is True
    assert saved["error"] is None and saved["problems"] == []
    assert saved["rows"] == 4 and saved["decisions"] == 4
    # the row set is the window's decisions, in order, with the T04 identity keys
    assert [slot["obs"]["row"] for slot in slots] == [0, 1, 2, 3]
    assert [slot["decision"] for slot in slots] == [32, 33, 34, 35]
    assert [slot["window_decision"] for slot in slots] == [0, 1, 2, 3]
    assert [slot["decision_frame"] for slot in slots] == [64, 66, 68, 70]
    for slot in slots:
        assert slot["schema"] == tk.SCHEMA_OBS_SLOT
        assert slot["obs_source"] == tk.OBS_SOURCE == "takeover-runner"
        assert slot["label_source"] == tk.OBS_LABEL_SOURCE
        assert slot["episode_seed"] == 12
        assert slot["classification"] == "measured"
        missing = [key for key in T04_SLOT_KEYS if key not in slot]
        assert missing == [], missing


# ---------------------------------------------------------------------------
# 48 (2) the planes are bit-identical to the tensors the actor received
# ---------------------------------------------------------------------------
def test_48_the_saved_planes_are_the_tensors_the_actor_saw(
        tmp_path: Path, tape_path: Path) -> None:
    import torch

    from touhou_rl.v400.student import obs_spec_v3

    result, fake, model, root = run_save_obs(tmp_path, tape_path)
    play = np.load(root / "obs" / "play96x112.npy")
    hud = np.load(root / "obs" / "hud.npy")
    hist = np.load(root / "obs" / "act_hist.npy")
    assert play.shape == (4, 4, 112, 96) and play.dtype == np.uint8
    assert hud.shape == (4, 56, 160) and hud.dtype == np.uint8
    assert hist.shape == (4, 8, 18) and hist.dtype == np.uint8
    # raw 0..255: a /255 plane could not exceed 1
    assert int(play.max()) > 1 and int(hud.max()) > 1
    assert int(hist.max()) == 1 and int(hist.min()) == 0
    assert len(model.inputs) == len(play) == 4
    for row, (play_t, hud_t, hist_t) in enumerate(model.inputs):
        # the frozen normalizers applied to the stored bytes are the tensors
        # ``model.act`` was called with -- bit for bit, no re-render involved
        assert torch.equal(
            torch.from_numpy(obs_spec_v3.normalize_play(play[row])[None]).float(),
            play_t), row
        assert torch.equal(
            torch.from_numpy(obs_spec_v3.normalize_hud(hud[row])[None]).float(),
            hud_t), row
        assert torch.equal(
            torch.from_numpy(hist[row][None]).float(), hist_t), row


# ---------------------------------------------------------------------------
# 49 (3) without --save-obs: no file, no extra command, same tensors
# ---------------------------------------------------------------------------
def test_49_without_save_obs_nothing_is_written_and_nothing_extra_is_sent(
        tmp_path: Path, tape_path: Path) -> None:
    import torch

    plain, fake_plain, model_plain = run_checkpoint(
        tmp_path, tape_path, model=RecordingModel(cand=5), entry_frame=64,
        cap_frames=8, out_name="out_noobs")
    # zero new files: the corpus path of the next run is not created yet
    assert plain["save_obs"] is None
    assert sorted(path.name for path in (tmp_path / "out_noobs").iterdir()) == [
        "result.json"]
    assert not (tmp_path / "corpus").exists()
    saved, fake_saved, model_saved, root = run_save_obs(
        tmp_path, tape_path, model=RecordingModel(cand=5), entry_frame=64,
        cap_frames=8, out_name="out_withobs")
    # not one extra protocol command (the corpus costs one HASH per decision)
    assert fake_saved.hash_calls == fake_plain.hash_calls + saved["decisions"]
    assert fake_plain.hash_calls == fake_saved.hash_calls - 4 == 26
    assert fake_plain.block.pixel_reads == fake_saved.block.pixel_reads
    assert fake_plain.enemy_reads == fake_saved.enemy_reads
    assert fake_plain.state_reads == fake_saved.state_reads
    assert fake_plain.steps == fake_saved.steps
    # and the policy saw exactly the same observation
    assert len(model_plain.inputs) == len(model_saved.inputs) == 4
    for left, right in zip(model_plain.inputs, model_saved.inputs):
        for a, b in zip(left, right):
            assert torch.equal(a, b)
    assert plain["frames"] == saved["frames"] == 72
    assert plain["stop"] == saved["stop"] == "frame_cap"


# ---------------------------------------------------------------------------
# 50 (4) only the checkpoint arm may save an observation
# ---------------------------------------------------------------------------
def test_50_a_non_checkpoint_arm_cannot_save_obs(tmp_path: Path,
                                                 tape_path: Path) -> None:
    for arm, extra in (("tape", {}), ("constant", {"constant_cand": 5})):
        save_dir = tmp_path / f"corpus_{arm}"
        config = make_config(tmp_path, tape_path, arm=arm, save_obs=save_dir,
                             out=tmp_path / f"out_{arm}", **extra)
        factory = Factory(None)                 # a client here is a failure
        result = tk.run_takeover(config, client_factory=factory)
        assert factory.calls == 0, arm
        assert result["stop"] == "save_obs_invalid", arm
        assert result["exit_code"] == tk.EXIT_REFUSED, arm
        assert result["save_obs"] is None, arm
        assert not save_dir.exists(), arm
        assert read_result(config)["stop"] == "save_obs_invalid"

    # the CLI carries the flag and refuses exactly the same way, before any engine
    argv = ["run", "--engine", str(make_engine(tmp_path / "cli_engine")),
            "--expected-sha", "0" * 64, "--data-dir", str(make_data_dir(tmp_path)),
            "--tape-slots", str(tape_path), "--seed", "12", "--entry-frame", "64",
            "--arm", "tape", "--cap-frames", "8", "--out", str(tmp_path / "cli_out"),
            "--save-obs", str(tmp_path / "cli_corpus")]
    parsed = tk.TakeoverConfig.from_args(tk.build_parser().parse_args(argv))
    assert parsed.save_obs == tmp_path / "cli_corpus"
    assert tk.main(argv) == tk.EXIT_REFUSED
    assert read_json(tmp_path / "cli_out" / "result.json")["stop"] == "save_obs_invalid"
    assert not (tmp_path / "cli_corpus").exists()


# ---------------------------------------------------------------------------
# 51 (5) a non-empty directory is refused and left untouched
# ---------------------------------------------------------------------------
def test_51_a_non_empty_directory_is_refused(tmp_path: Path,
                                             tape_path: Path) -> None:
    save_dir = tmp_path / "corpus_nonempty"
    save_dir.mkdir()
    keep = save_dir / "keep.txt"
    keep.write_text("do not touch\n", encoding="utf-8")
    checkpoint = make_checkpoint(tmp_path / "ckpt_v3.pt")
    config = make_config(tmp_path, tape_path, arm="checkpoint",
                         checkpoint=checkpoint,
                         checkpoint_sha256=tk.sha256_file(checkpoint),
                         save_obs=save_dir)
    factory = Factory(None)
    result = tk.run_takeover(config, client_factory=factory)
    assert factory.calls == 0
    assert result["stop"] == "save_obs_dir_not_empty"
    assert result["exit_code"] == tk.EXIT_REFUSED
    assert result["save_obs"] is None
    # nothing was created, nothing was overwritten
    assert keep.read_text(encoding="utf-8") == "do not touch\n"
    assert sorted(path.name for path in save_dir.iterdir()) == ["keep.txt"]

    # a *file* where the directory should be is refused too
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    second = make_config(tmp_path, tape_path, arm="checkpoint",
                         checkpoint=checkpoint,
                         checkpoint_sha256=tk.sha256_file(checkpoint),
                         save_obs=blocker, out=tmp_path / "out_blocked")
    outcome = tk.run_takeover(second, client_factory=Factory(None))
    assert outcome["stop"] == "save_obs_invalid"
    assert outcome["exit_code"] == tk.EXIT_REFUSED
    assert blocker.read_text(encoding="utf-8") == "x"


# ---------------------------------------------------------------------------
# 52 an existing *empty* directory is accepted
# ---------------------------------------------------------------------------
def test_52_an_existing_empty_directory_is_accepted(tmp_path: Path,
                                                    tape_path: Path) -> None:
    save_dir = tmp_path / "corpus_empty"
    save_dir.mkdir()
    result, fake, model, root = run_save_obs(tmp_path, tape_path, save_dir=save_dir)
    assert result["exit_code"] == tk.EXIT_OK
    assert result["save_obs"]["root"] == save_dir.as_posix()
    assert len(rows_of(save_dir / "slots.jsonl")) == 4
    assert read_json(save_dir / "meta.json")["complete"] is True


# ---------------------------------------------------------------------------
# 53 (6a) a plane write failure fails closed and keeps the rows
# ---------------------------------------------------------------------------
def test_53_a_plane_write_failure_fails_closed(tmp_path: Path, tape_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(path, array):  # noqa: ANN001 - the np.save seam
        raise OSError(28, "No space left on device (scripted)")

    monkeypatch.setattr(tk, "_save_npy", boom)
    result, fake, model, root = run_save_obs(tmp_path, tape_path)
    assert result["exit_code"] == tk.EXIT_TECHNICAL
    assert result["failure"]["kind"] == "save_obs"
    assert result["failure"]["stop"] == "save_obs_error"
    assert "No space left" in result["failure"]["detail"]
    # the rows written before the failure are kept
    assert len(rows_of(root / "slots.jsonl")) == result["decisions"] == 4
    meta = read_json(root / "meta.json")
    assert meta["complete"] is False
    assert meta["decisions"] == 4
    assert any("No space left" in problem for problem in meta["problems"])
    assert meta["planes"] == {}
    saved = result["save_obs"]
    assert saved["ok"] is False and saved["complete"] is False
    assert "No space left" in (saved["error"] or "")
    # and result.json on disk says the same (non-zero), never a clean run
    on_disk = read_json(tmp_path / "out_obs" / "result.json")
    assert on_disk["exit_code"] == tk.EXIT_TECHNICAL
    assert on_disk["failure"]["kind"] == "save_obs"
    assert on_disk["save_obs"]["ok"] is False


# ---------------------------------------------------------------------------
# 54 (6b) a row write failure stops the window (fail-closed, no hole)
# ---------------------------------------------------------------------------
def test_54_a_row_write_failure_stops_the_window(tmp_path: Path, tape_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    original = tk.ObsRecorder.record
    calls = {"n": 0}

    def flaky(self, **kwargs):  # noqa: ANN001 - the recorder's own signature
        calls["n"] += 1
        if calls["n"] == 2:
            raise tk.ObsSaveError("the disk went away (scripted)")
        return original(self, **kwargs)

    monkeypatch.setattr(tk.ObsRecorder, "record", flaky)
    result, fake, model, root = run_save_obs(tmp_path, tape_path)
    assert result["stop"] == "save_obs_error"
    assert result["exit_code"] == tk.EXIT_TECHNICAL
    assert result["failure"]["stop"] == "save_obs_error"
    detail = result["failure"]["detail"]
    assert detail["decision"] == 33 and detail["corpus_rows"] == 1
    assert "the disk went away" in detail["reason"]
    # the second decision's frames were stepped, but it was never recorded as a
    # decision: the corpus has no hole, it simply ends where the disk failed
    assert result["decisions"] == 1 and result["frames_takeover"] == 4
    assert model.calls == 2                     # the second decision was measured
    slots = rows_of(root / "slots.jsonl")
    assert len(slots) == 1 and slots[0]["decision"] == 32
    for name, file_name, shape, dtype in tk.OBS_PLANES:
        array = np.load(root / "obs" / file_name)
        assert array.shape == (1, *shape), name
    meta = read_json(root / "meta.json")
    assert meta["decisions"] == 1 and meta["complete"] is False
    assert meta["stop"] == "save_obs_error"
    assert result["save_obs"]["ok"] is True       # nothing was lost: 1 row, 1 decision
    assert result["save_obs"]["complete"] is False
    assert read_json(tmp_path / "out_obs" / "result.json")["stop"] == "save_obs_error"


# ---------------------------------------------------------------------------
# 55 (7) history / executed_cand / mask agree with the decision log
# ---------------------------------------------------------------------------
def test_55_history_and_executed_action_agree_with_the_decision_log(
        tmp_path: Path, tape_path: Path) -> None:
    log = tmp_path / "obs_decisions.jsonl"
    result, fake, model, root = run_save_obs(tmp_path, tape_path, decision_log=log)
    slots = rows_of(root / "slots.jsonl")
    rows = rows_of(log)[1:]
    assert len(slots) == len(rows) == 4
    tape = tk.load_tape(tape_path)
    plan = tk.entry_plan(tape, 64)
    seed = tk.prefix_action_cands(plan)
    assert len(seed) == 32                     # 8 lead-in + 24 prefix slots
    emitted = [row["cand"] for row in rows]
    for index, (slot, row) in enumerate(zip(slots, rows)):
        assert slot["decision_frame"] == row["frame"]
        assert slot["executed_cand"] == row["cand"]
        assert slot["cand"] == row["cand"]
        assert slot["mask"] == row["mask"]
        assert slot["masks"] == row["masks"]
        assert slot["executed"] == {"cand": row["cand"], "bomb": 0,
                                    "mask": row["mask"], "source": "policy_act"}
        assert slot["duration_frames"] == row["frames_per_decision"]
        assert slot["state"]["deaths"] == row["deaths"]
        assert slot["state"]["lives"] == row["lives"]
        assert slot["state"]["bombs"] == row["bombs"]
        assert slot["state"]["score"] == row["score"]
        assert slot["state"]["stage"] == row["stage"]
        assert slot["events"] == row["events"]
        assert slot["end"] == row["end"]
        expected = (seed + emitted)[:len(seed) + index][-8:]
        assert slot["history"] == expected, index
    assert slots[0]["history"] == seed[-8:]
    assert slots[-1]["history"][-1] == rows[-2]["cand"]
    # the student's own hash chain: row 0's pre is the entry hash, and each row's
    # post is the next row's pre
    assert slots[0]["hash_pre"] == result["entry_hash"] == f"{24:016x}"
    for left, right in zip(slots, slots[1:]):
        assert left["hash_post"] == right["hash_pre"]
    assert [slot["hash_post"] for slot in slots] == [
        f"{value:016x}" for value in (25, 26, 27, 28)]
    # a --save-obs run still writes the ordinary decision log, unchanged
    assert result["decision_log"]["rows"] == result["decisions"] + 1 == 5


# ---------------------------------------------------------------------------
# 56 (7b) the frame windows are exactly the frozen T04 ones
# ---------------------------------------------------------------------------
def test_56_the_frame_windows_are_the_frozen_t04_ones(tmp_path: Path,
                                                      tape_path: Path) -> None:
    from touhou_rl.v420 import contracts as v420_contracts
    from touhou_rl.v420 import dataset_2f

    result, fake, model, root = run_save_obs(tmp_path, tape_path)
    slots = rows_of(root / "slots.jsonl")
    for slot in slots:
        frame = slot["decision_frame"]
        assert slot["commit_frames"] == list(v420_contracts.decision_span(frame))
        assert slot["duration_frames"] == v420_contracts.DECISION_FRAMES == 2
        assert slot["obs_render_frames"] == list(dataset_2f.obs_window(frame))
        assert slot["history_frames"] == list(dataset_2f.history_window(frame))
        assert slot["runner_frames"]["obs_render"] == [
            value + 1 for value in dataset_2f.obs_window(frame)]
        assert slot["runner_frames"]["commit"] == [frame + 1, frame + 2]
        assert slot["obs"]["hud_frame"] == slot["obs_render_frames"][-1]
        assert slot["decision"] == frame // tk.DECISION_FRAMES
        assert len(slot["history"]) == len(slot["history_frames"]) == 8
        assert slot["obs"]["row"] == slot["decision"] - 32
    first = slots[0]
    # spelled out, so the numbers are readable without importing the contracts
    assert first["obs_render_frames"] == [60, 61, 62, 63]
    assert first["runner_frames"]["obs_render"] == [61, 62, 63, 64]
    assert first["history_frames"] == [48, 50, 52, 54, 56, 58, 60, 62]
    assert first["commit_frames"] == [64, 66]


# ---------------------------------------------------------------------------
# 57 meta.json identity + README's null table
# ---------------------------------------------------------------------------
def test_57_meta_json_carries_the_identity_and_the_readme_names_the_nulls(
        tmp_path: Path, tape_path: Path) -> None:
    result, fake, model, root = run_save_obs(tmp_path, tape_path)
    meta = read_json(root / "meta.json")
    assert meta["schema"] == tk.SCHEMA_OBS_META
    assert meta["runner"]["path"].endswith("v8_32_takeover.py")
    assert meta["runner"]["sha256"] == tk.sha256_file(tk.TOOL_PATH)
    assert meta["checkpoint"]["path"].endswith("ckpt_v3.pt")
    assert meta["checkpoint"]["sha256"] == tk.sha256_file(tmp_path / "ckpt_v3.pt")
    assert meta["checkpoint"]["sha_matches"] is True
    assert meta["obs_spec_version"] == tk.OBS_SPEC_V3_VERSION
    assert meta["obs_spec_digest"] == tk.OBS_SPEC_V3_DIGEST
    assert meta["policy_input"] == tk.POLICY_INPUT_CHECKPOINT
    assert meta["entry_frame"] == 64
    assert meta["seed"] == 12
    assert meta["frames_per_decision"] == 2
    assert meta["decisions"] == meta["slots"]["rows"] == result["decisions"] == 4
    assert meta["frames"] == result["frames"] == 72
    assert meta["frames_prefix"] == 64 and meta["frames_takeover"] == 8
    assert meta["stop"] == "frame_cap"
    assert meta["complete"] is True and meta["problems"] == []
    assert meta["tape"]["sha256"] == tk.sha256_file(tape_path)
    assert meta["tape"]["slots"] == SLOT_COUNT
    assert isinstance(meta["created_at"], str) and meta["created_at"]
    for name, file_name, shape, dtype in tk.OBS_PLANES:
        record = meta["planes"][name]
        path = root / record["file"]
        assert record["file"] == f"obs/{file_name}"
        assert record["shape"] == [4, *shape]
        assert record["dtype"] == dtype
        assert record["bytes"] == path.stat().st_size
        assert record["sha256"] == tk.sha256_file(path)
    assert meta["slots"]["sha256"] == tk.sha256_file(root / "slots.jsonl")
    assert meta["null_fields"] == dict(tk.OBS_SLOT_NULL_FIELDS)
    readme = (root / "README.md").read_text(encoding="utf-8")
    for name, why in tk.OBS_SLOT_NULL_FIELDS:
        assert f"`{name}`" in readme, name
        assert why[:40] in readme, name
    for text in (tk.SCHEMA_OBS_META, tk.OBS_LABEL_SOURCE, tk.OBS_SOURCE,
                 "obs_spec_v3", "normalize_play"):
        assert text in readme, text
    # every null field of every row is declared, and no row invents a teacher value
    slots = rows_of(root / "slots.jsonl")
    for slot in slots:
        for name, _why in tk.OBS_SLOT_NULL_FIELDS:
            assert slot[name] is None, name
        assert slot["label_verified"] is None
        assert slot["chosen"] is None and slot["scores"] is None
        assert slot["margin"] is None and slot["teacher_bomb"] is None


# ---------------------------------------------------------------------------
# 58 the planes are the images of the frames the row records (no re-render)
# ---------------------------------------------------------------------------
def test_58_each_row_holds_the_images_of_its_own_frames(tmp_path: Path,
                                                       tape_path: Path) -> None:
    from touhou_rl.v400.student import obs_spec_v3

    result, fake, model, root = run_save_obs(tmp_path, tape_path)
    play = np.load(root / "obs" / "play96x112.npy")
    hud = np.load(root / "obs" / "hud.npy")
    for index, slot in enumerate(rows_of(root / "slots.jsonl")):
        frames = slot["runner_frames"]["obs_render"]
        expected_play = np.stack([
            obs_spec_v3.encode_planes_v3(fake_pixels(frame))[0] for frame in frames])
        expected_hud = obs_spec_v3.encode_planes_v3(fake_pixels(frames[-1]))[1]
        assert np.array_equal(play[index], expected_play), index
        assert np.array_equal(hud[index], expected_hud), index
    # the first stack is frames 61..64 (four consecutive renders), the last 67..70
    assert rows_of(root / "slots.jsonl")[0]["runner_frames"]["obs_render"] == [
        61, 62, 63, 64]
    assert rows_of(root / "slots.jsonl")[-1]["runner_frames"]["obs_render"] == [
        67, 68, 69, 70]


# ---------------------------------------------------------------------------
# 59 an entry failure leaves an empty corpus that says so
# ---------------------------------------------------------------------------
def test_59_an_entry_failure_leaves_an_empty_honest_corpus(
        tmp_path: Path, tape_path: Path) -> None:
    save_dir = tmp_path / "corpus_entry_fail"
    checkpoint = make_checkpoint(tmp_path / "ckpt_v3.pt")
    config = make_config(tmp_path, tape_path, arm="checkpoint", entry_frame=64,
                         cap_frames=8, checkpoint=checkpoint,
                         checkpoint_sha256=tk.sha256_file(checkpoint),
                         save_obs=save_dir)
    fake = FakeHarness()
    # the entry HASH is the 26th call: 1 lead-in anchor + 24 prefix slots, then the
    # entry itself; every prefix slot still verifies, only the entry disagrees
    fake.hash_overrides_by_call = {26: "deadbeef"}
    model = RecordingModel(cand=5)
    result = tk.run_takeover(
        config, client_factory=lambda _config: fake,
        block_factory=checkpoint_block_factory(fake),
        policy_loader=policy_loader(model),
        clock=iter([0.0, 0.25, 0.5, 1.0, 2.0, 3.0]).__next__)
    root = save_dir
    assert result["stop"] == "entry_hash_mismatch"
    assert result["mismatch"]["kind"] == "entry"
    assert result["mismatch"]["frame"] == 64
    assert result["entry_verified"] is False
    assert result["exit_code"] == tk.EXIT_ENTRY_FAILED
    assert result["decisions"] == 0 and model.calls == 0
    # the corpus really exists (the directory was prepared before the engine), and
    # it claims nothing it did not measure
    assert rows_of(root / "slots.jsonl") == []
    for name, file_name, shape, dtype in tk.OBS_PLANES:
        array = np.load(root / "obs" / file_name)
        assert array.shape == (0, *shape) and array.dtype == np.dtype(dtype), name
    meta = read_json(root / "meta.json")
    assert meta["decisions"] == 0 and meta["complete"] is False
    assert meta["stop"] == "entry_hash_mismatch"
    assert meta["failure"]["kind"] == "entry_hash_mismatch"
    assert result["save_obs"]["ok"] is True        # nothing was lost: 0 rows, 0
    assert result["save_obs"]["complete"] is False
    # result.json keeps the entry failure; the corpus did not overwrite it
    on_disk = read_result(config)
    assert on_disk["stop"] == "entry_hash_mismatch"
    assert on_disk["exit_code"] == tk.EXIT_ENTRY_FAILED

