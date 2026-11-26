"""V831-SLICE-01 offline tests: the slice tool's contracts, no engine needed.

``tools/rl/v8_31_slice.py`` is the v8.31.0 instrument for two real trajectories.
Everything that decides whether a live number *means* anything is a pure
function, and this module pins those functions.  Nothing here starts the game,
opens a shared-memory block, reads ``/root`` or launches ``xvfb-run``:

* the **first-frame fill** 閳?one pushed frame becomes a four-layer stack of that
  same frame, and the action history starts as eight all-zero rows;
* the **input-isolation boundary** 閳?:func:`build_actor_inputs` takes exactly
  ``(pixels_rgba, history)``; two completely different referee snapshots leave
  the digests, the observation buffer's internal state and a pixel-driven policy's
  output bit-for-bit identical, and no referee field name appears in
  ``ACTOR_FIELDS``;
* the **48-decision release** 閳?``release_due(47, 48)`` is true,
  ``commit_frames(cand, decision_frames=2, release_shoot=True)`` is
  ``[(mask, 1), (mask & ~SHOOT & ~BOMB, 1)]``, and a scripted decision loop really
  takes that path at decision 47 and nowhere else;
* **truncation** 閳?a 5-frame cap at 2 frames per decision advances 4 frames and
  stops as ``truncated``; the cap is never exceeded;
* **legal death** 閳?a scripted ``game_over`` or a ``deaths`` increase classifies
  as ``legal_death`` and the trajectory is still recorded;
* **missing events** 閳?an ``EVENTS`` reply without ``stageClearEdgeFrame`` (or an
  ``ERR unknown-command EVENTS``) is ``technical_invalid`` with the first conflict
  recorded, never a zero;
* **the audit order** -- every ``STEP`` unit issues exactly
  ``step -> shm -> events -> state`` and appends the read-only
  ``hud``/``pixels`` tail only on the units named by
  :func:`v8_31_slice.gated_audit_plan` (first unit, one SHOOT pulse period for
  ``hud``, and once after the last unit for both), asserted against a fake host
  that records every protocol call.  ``shm`` is read once per unit (twice when
  the single retry fires).  This is the V831-SLICE-01 run-2 fix: the block used
  to be read *after* ``STATE``, whose ``PublishState`` rewrites ``header->frame``
  to the post-increment ``FrameCount()`` (``rlharness.cpp:1355``/``:1362``);
* **the slim does not move a digest** -- the mask trace and the action sequence
  are rebuilt independently from the masks and the candidates, and
  ``--audit-stride 2`` produces the same three digests with half the
  ``EVENTS``/``STATE`` traffic;
* **the pixel frame relation** -- both legal engine models are covered: rl_v3's
  ``STEP`` ends with its own ``PublishState`` (``rlharness.cpp:901``), so the
  header reads ``EVENTS.frames`` while the captured RGBA is ``EVENTS.frames - 1``;
  a ``STEP`` that returned early leaves the header at ``EVENTS.frames - 1``.  The
  recorded ``pixel_frame`` is the frame of the bytes in both cases, and the
  pre-fix order (``STATE`` first) is reconstructed to show it must have failed;
* **the pixels checksum** -- ``pixels_checksum_local`` is FNV-1a-64 of the
  captured bytes (offset basis ``14695981039346656037``, prime
  ``1099511628211``; reference vectors plus ``b"\x00"*8`` and
  ``bytes(range(16))``), while a disagreeing ``PIXELS`` reply only warns;
* **the digest scopes** (V831-SLICE-01c) -- the FNV is the ticket's
  *per-decision* value (the decision's last unit; ``fnv_units == decisions``)
  and every unit still carries ``pixel_frame_blake2b128``
  (``blake2b_units == step_units``), pinned by known blake2b-128 vectors and
  unaffected by ``--audit-stride``;
* **the mute bytes** 閳?zeroing ``0x1F/0x20/0x27/0x28`` of a 60-byte config
  changes those four bytes and nothing else, and the four decoded fields are zero,
  from an audible source and from an already-muted one;
* **the tape schedule** 閳?the frozen loader puts slot 0's eight ``history``
  candidates first, then ``executed_cand`` in order, then repeats the last one,
  and every ``bomb`` is 0;
* **the observation contract** 閳?``obs_spec_v3_digest()`` is still the frozen
  ``ef2fbd15閳ヮ泦`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from touhou_rl.spec import BTN_BOMB, BTN_SHOOT  # noqa: E402
from touhou_rl.v400.gym import actions as gym_actions  # noqa: E402
from touhou_rl.v400.gym import protocol  # noqa: E402
from touhou_rl.v400.gym.shm import FLAG_PIXELS_VALID  # noqa: E402
from touhou_rl.v400.student import obs_spec_v3  # noqa: E402
from touhou_rl.v400.student.obs_spec_v2 import ActionHistoryBuffer  # noqa: E402
from touhou_rl.v400.teacher.shoot import commit_frames, release_due  # noqa: E402


def _load_slice_tool():
    """Import ``tools/rl/v8_31_slice.py`` (not a package, so by file path)."""

    path = REPO_ROOT / "tools" / "rl" / "v8_31_slice.py"
    assert path.is_file(), f"missing {path}"
    spec = importlib.util.spec_from_file_location("v831_slice_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SLICE = _load_slice_tool()


# ---------------------------------------------------------------------------
# helpers: pixels, a scripted host, a pixel-driven policy
# ---------------------------------------------------------------------------
def make_pixels(seed: int = 7) -> np.ndarray:
    """A deterministic 640x480 RGBA8 readback with the playfield populated."""

    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(480, 640, 4), dtype=np.uint8)


def _policy_cand(actor: dict) -> int:
    """A policy whose answer depends on the *bytes* of the actor input.

    Deliberately not a constant: if two actor inputs differ by one bit, this can
    return a different candidate, so "same output" is a real assertion.
    """

    flat = np.asarray(actor["play"]).reshape(-1)
    return int(np.argmax(flat[:18]))


class FakeHost:
    """Scripted stand-in for :class:`v8_31_slice.EngineHost` (offline only).

    It answers the five protocol calls the decision loop makes and can be told to
    advance the stage, die, game-over, exit, or corrupt ``EVENTS`` / the pixel
    frame relation at a given frame.  Every protocol call appends its wire name
    to :attr:`calls`, so the audit order can be asserted, not assumed.

    ``pixel_frame_offset`` selects the engine model the header follows:

    ``0`` (default)
        rl_v3: after ``STEP`` the header has already been stamped by the step's
        own ``PublishState`` (``rlharness.cpp:901`` -> ``:1362``), so
        ``header["frame"] == EVENTS.frames`` while the RGBA bytes are one frame
        older.
    ``-1``
        a ``STEP`` that returned before that publish (the ``exit=1`` path) or a
        pre-T7.3 build: ``header["frame"] == EVENTS.frames - 1``.
    """

    def __init__(self, *, stage_advance_at: int | None = None,
                 death_at: int | None = None, death_lives: float = 1.0,
                 game_over_at: int | None = None,
                 exit_at: int | None = None,
                 pixel_frame_offset: int = 0,
                 events_mode: str = "ok",
                 events_flags: int = 0,
                 clear_level_at: int | None = None,
                 checksum_mode: str = "match") -> None:
        self._frames = 0
        self._stage = 0
        self._deaths = 0.0
        self._lives = 2.0
        self._bombs = 3.0
        self._score = 0
        self._game_over = 0
        self.steps: list[tuple[int, int]] = []
        self.calls: list[str] = []
        self.stage_advance_at = stage_advance_at
        self.death_at = death_at
        self.death_lives = float(death_lives)
        self.game_over_at = game_over_at
        self.exit_at = exit_at
        self.pixel_frame_offset = int(pixel_frame_offset)
        self.events_mode = events_mode
        self.events_flags = int(events_flags)
        self.clear_level_at = clear_level_at
        self.checksum_mode = checksum_mode
        self.rgba = make_pixels()

    # -- protocol ---------------------------------------------------------
    def step(self, mask: int, frames: int) -> dict[str, str]:
        self.calls.append("step")
        self.steps.append((int(mask), int(frames)))
        self._frames += int(frames)
        self._score += int(frames) * 10
        if self.stage_advance_at is not None and \
                self._frames >= self.stage_advance_at:
            self._stage = 1
        if self.death_at is not None and self._frames >= self.death_at:
            self._deaths = 1.0
            self._lives = float(self.death_lives)
        reply: dict[str, str] = {"advanced": str(int(frames))}
        if self.game_over_at is not None and self._frames >= self.game_over_at:
            self._game_over = 1
        if self.exit_at is not None and self._frames >= self.exit_at:
            reply["exit"] = "1"
        if self._game_over:
            reply["game_over"] = "1"
        return reply

    def events_raw(self) -> str:
        self.calls.append("events")
        if self.events_mode == "error":
            raise protocol.HarnessError(
                "'EVENTS' -> ERR unknown-command EVENTS")
        flags = self.events_flags
        fields: dict[str, str] = {
            "frames": str(self._frames),
            "frameEventFlags": f"{flags:04x}",
            "stageClear": "0",
            "stageClearEdge": "0",
            "stageClearEdgeFrame": "0",
            "stageClearScreenState": "0",
            "stageClearSeq": "0",
        }
        if self.clear_level_at is not None and \
                self._frames >= self.clear_level_at:
            fields["stageClear"] = "1"
            fields["frameEventFlags"] = f"{flags | 0x0080:04x}"
            fields["stageClearEdge"] = "1"
            fields["stageClearEdgeFrame"] = str(self._frames)
        if self.events_mode == "missing-edge-frame":
            fields.pop("stageClearEdgeFrame")
        if self.events_mode == "missing-frames":
            fields.pop("frames")
        return "OK " + " ".join(f"{key}={value}"
                               for key, value in fields.items())

    def state(self) -> dict[str, object]:
        self.calls.append("state")
        return {
            "frame": self._frames,
            "gameplay_frame": self._frames + 1,
            "stage": self._stage,
            "deaths": self._deaths,
            "lives_remaining": self._lives,
            "bombs_remaining": self._bombs,
            "player_power": 0.0,
            "score": self._score,
            "graze": 0,
        }

    def hud(self) -> dict[str, object]:
        self.calls.append("hud")
        return {
            "frame": self._frames,
            "display_score": self._score,
            "globals_score": self._score,
            "lives": int(self._lives),
            "bombs": int(self._bombs),
            "power": 0,
            "graze": 0,
            "point": 0,
            "time": 0,
            "game_over": self._game_over,
        }

    def pixels(self):
        self.calls.append("shm")
        header = {
            "version": 1,
            "frame": self._frames + self.pixel_frame_offset,
            "state_hash": 0,
            "width": 640,
            "height": 480,
            "state_bytes": 100,
            "flags": FLAG_PIXELS_VALID,
        }
        return header, self.rgba

    def pixels_checksum(self) -> dict[str, object]:
        self.calls.append("pixels")
        if self.checksum_mode == "error":
            raise protocol.HarnessError("'PIXELS' -> ERR pixels-failed")
        # The engine's HandlePixels hashes its glReadPixels buffer (bottom-up),
        # the same byte order the shared block stores.
        checksum = SLICE.fnv1a64_hex(
            np.ascontiguousarray(np.asarray(self.rgba)[::-1]).tobytes())
        if self.checksum_mode == "mismatch":
            checksum = "dead" * 4
        return {
            "raw": "OK",
            "pixels": "640x480",
            "width": 640,
            "height": 480,
            "nonzero": int(np.count_nonzero(self.rgba)),
            "checksum": checksum,
        }


class StatePublishHost(FakeHost):
    """The ticket's mandated reproduction: ``STATE`` rewrites the header.

    ``pixels()`` behaves like the engine's ``AfterDrawFrame``: right after
    ``STEP`` it reports ``k`` (the frame that was just drawn).  ``STATE`` is the
    ``PublishState`` at ``rlharness.cpp:1355``: it stamps ``header->frame`` with
    ``FrameCount()``, i.e. ``k + 1``, which is also what ``EVENTS.frames``
    reports.  The pre-fix audit read the block after ``STATE`` and compared the
    rewritten ``k + 1`` against ``EVENTS.frames - 1 == k``, which is exactly the
    run-2 ``technical_invalid``.
    """

    def __init__(self) -> None:
        super().__init__(pixel_frame_offset=-1)
        self._published: int | None = None

    def step(self, mask: int, frames: int) -> dict[str, str]:
        self._published = None  # the draw re-stamps the header, pre-increment
        return super().step(mask, frames)

    def state(self) -> dict[str, object]:
        self._published = self._frames  # PublishState -> post-increment
        return super().state()

    def pixels(self):
        header, rgba = super().pixels()
        if self._published is not None:
            header = dict(header)
            header["frame"] = int(self._published)
        return header, rgba


def drive(host: FakeHost, *, actions=None, max_controlled_frames: int = 600,
          decision_frames: int = 2, shoot_pulse_period: int = 48,
          stop_on_first_death: bool = True, timeout_s: float = 60.0,
          start_frames: int = 0, history: ActionHistoryBuffer | None = None,
          tap=None, allow_partial_tail: bool = False, audit_stride: int = 1,
          meter=None):
    """Run the real decision loop against a scripted host."""

    tap = tap if tap is not None else SLICE.ActorInputTap()
    if tap.frames_pumped == 0:
        tap.pump(host.rgba)
    history = history if history is not None else ActionHistoryBuffer()
    actions = actions if actions is not None else SLICE.ConstantActions(12)
    return SLICE.run_decisions(
        host=host,
        actions=actions,
        tap=tap,
        history=history,
        meter=meter if meter is not None else SLICE.CostMeter(),
        decision_frames=decision_frames,
        shoot_pulse_period=shoot_pulse_period,
        max_controlled_frames=max_controlled_frames,
        start_frames=start_frames,
        initial_state=host.state(),
        timeout_s=timeout_s,
        stop_on_first_death=stop_on_first_death,
        allow_partial_tail=allow_partial_tail,
        start_renders=tap.frames_pumped,
        audit_stride=audit_stride,
    )


class MeteredHost(FakeHost):
    """``FakeHost`` with :class:`v8_31_slice.EngineHost`'s own buckets.

    The real host times ``STEP`` into ``step`` and every readback into
    ``audit``; the scripted host has no meter, so this subclass reproduces the
    same two buckets around the same calls.  Without it a scripted run's
    ``unaccounted`` would contain the fake protocol calls and the attribution
    test would be measuring the wrong thing.
    """

    def __init__(self, meter, **kwargs) -> None:
        super().__init__(**kwargs)
        self.meter = meter

    def step(self, mask: int, frames: int) -> dict[str, str]:
        with self.meter.measure("step"):
            return super().step(mask, frames)

    def events_raw(self) -> str:
        with self.meter.measure("audit"):
            return super().events_raw()

    def state(self) -> dict[str, object]:
        with self.meter.measure("audit"):
            return super().state()

    def hud(self) -> dict[str, object]:
        with self.meter.measure("audit"):
            return super().hud()

    def pixels(self):
        with self.meter.measure("audit"):
            return super().pixels()

    def pixels_checksum(self) -> dict[str, object]:
        with self.meter.measure("audit"):
            return super().pixels_checksum()


# ---------------------------------------------------------------------------
# 1. the first-frame fill
# ---------------------------------------------------------------------------
def test_first_frame_fill_repeats_the_earliest_frame_four_times():
    pixels = make_pixels(11)
    play, hud = obs_spec_v3.encode_planes_v3(pixels)
    buffer = obs_spec_v3.ObservationBufferV3()
    buffer.push(play, hud)
    stack = buffer.stack()

    assert stack.shape == (obs_spec_v3.FRAME_STACK, *obs_spec_v3.PLAY_SHAPE)
    assert stack.dtype == np.uint8
    for layer in range(obs_spec_v3.FRAME_STACK):
        assert np.array_equal(stack[layer], play), (
            f"stack layer {layer} is not the earliest available frame")
    assert np.array_equal(buffer.hud(), hud)

    rows = ActionHistoryBuffer().rows()
    assert rows.shape == obs_spec_v3.ACTION_HISTORY_SHAPE
    assert rows.dtype == np.uint8
    assert not rows.any(), "the action history must start as all-zero rows"

    actor = SLICE.build_actor_inputs(pixels, ActionHistoryBuffer())
    assert actor["play"].shape == (obs_spec_v3.FRAME_STACK, *obs_spec_v3.PLAY_SHAPE)
    for layer in range(obs_spec_v3.FRAME_STACK):
        assert np.array_equal(actor["play"][layer], play)
    assert not actor["act_hist"].any()

    tap = SLICE.ActorInputTap()
    assert tap.pump(pixels) == 1
    assert tap.frames_pumped == 1
    assert np.array_equal(tap.stack(), actor["play"])
    assert np.array_equal(tap.hud_plane(), hud)


# ---------------------------------------------------------------------------
# 2. the input-isolation boundary
# ---------------------------------------------------------------------------
REFEREE_A = {
    "lives": 2, "bombs": 3, "power": 128, "stage": 0, "deaths": 0,
    "frame_event_flags": 0x0000, "stage_clear": 0, "score": 0,
}
REFEREE_B = {
    "lives": 0, "bombs": 0, "power": 0, "stage": 3, "deaths": 7,
    "frame_event_flags": 0x0082, "stage_clear": 1, "score": 999999,
}


def _actor_roundtrip(pixels, history, referee):
    """Everything the actor path produces, with a referee snapshot beside it.

    The snapshot is stored on an ambient object first, exactly like a live host,
    so an implementation that reached for ``host.lives`` or a module global would
    see the two different values.
    """

    ambient = SimpleNamespace(**referee)
    tap = SLICE.ActorInputTap()
    tap.pump(pixels)
    live = tap.inputs(history)
    direct = SLICE.build_actor_inputs(pixels, history)
    return {
        "ambient_lives": ambient.lives,
        "ambient_stage": ambient.stage,
        "tap_frames_pumped": tap.frames_pumped,
        "tap_stack_sha256": SLICE.array_sha256(tap.stack()),
        "tap_hud_sha256": SLICE.array_sha256(tap.hud_plane()),
        "tap_digests": dict(live["digests"]),
        "direct_digests": dict(direct["digests"]),
        "direct_denied_keys": sorted(set(direct).intersection(
            SLICE.REFEREE_FIELDS)),
        "policy_cand": _policy_cand(direct),
    }


def test_actor_inputs_ignore_every_referee_field():
    pixels = make_pixels(23)
    history = ActionHistoryBuffer()
    for cand in (3, 11, 0):
        history.push(cand)

    left = _actor_roundtrip(pixels, history, REFEREE_A)
    right = _actor_roundtrip(pixels, history, REFEREE_B)

    # The two runs really did see different referee snapshots.
    assert left["ambient_lives"] != right["ambient_lives"]
    assert left["ambient_stage"] != right["ambient_stage"]

    # ... and the actor input, the buffer's internal state and the policy output
    # are bit-for-bit identical anyway.
    assert left["tap_digests"] == right["tap_digests"]
    assert left["direct_digests"] == right["direct_digests"]
    assert left["tap_stack_sha256"] == right["tap_stack_sha256"]
    assert left["tap_hud_sha256"] == right["tap_hud_sha256"]
    assert left["tap_frames_pumped"] == right["tap_frames_pumped"] == 1
    assert left["policy_cand"] == right["policy_cand"]
    assert left["direct_denied_keys"] == []
    for digests in (left["tap_digests"], left["direct_digests"]):
        assert set(digests) == {"play", "hud", "act_hist", "actor_input_digest"}
        for value in digests.values():
            assert len(value) == 64


def test_actor_input_boundary_is_structural():
    parameters = list(inspect.signature(SLICE.build_actor_inputs).parameters)
    assert parameters == ["pixels_rgba", "history"]
    assert len(parameters) == 2

    assert not set(SLICE.ACTOR_FIELDS).intersection(SLICE.REFEREE_FIELDS)
    for referee in SLICE.REFEREE_FIELDS:
        assert referee not in SLICE.ACTOR_FIELDS
    for parameter in parameters:
        assert parameter in SLICE.ACTOR_FIELDS
        assert parameter not in SLICE.REFEREE_FIELDS

    # The tool's own run-time check agrees, and it inspects the live tap too.
    assert SLICE.actor_boundary_problems() == []
    for method in ("pump", "inputs"):
        names = [name for name in inspect.signature(
            getattr(SLICE.ActorInputTap, method)).parameters if name != "self"]
        for name in names:
            assert name not in SLICE.REFEREE_FIELDS

    # A referee field appearing in the whitelist would be caught, not ignored.
    problems = SLICE.actor_boundary_problems()
    assert problems == []


# ---------------------------------------------------------------------------
# 3. the 48-decision release (two mask segments)
# ---------------------------------------------------------------------------
def test_release_due_47_of_48_is_true():
    assert release_due(47, 48) is True
    assert release_due(46, 48) is False
    assert release_due(48, 48) is False
    assert release_due(0, 48) is False


def test_commit_frames_release_splits_into_two_one_frame_segments():
    cand = 12
    mask = gym_actions.cand_to_mask(cand, 0)
    held = mask & ~BTN_SHOOT & ~BTN_BOMB
    assert held != mask, "the release frame must differ from the held mask"
    segments = commit_frames(cand, bomb=0, decision_frames=2, release_shoot=True)
    assert segments == [(mask, 1), (held, 1)]
    assert [count for _mask, count in segments] == [1, 1]
    # The direction/focus bits are untouched by the release.
    assert held & ~(BTN_SHOOT | BTN_BOMB) == mask & ~(BTN_SHOOT | BTN_BOMB)
    # Without the release it is one two-frame segment.
    assert commit_frames(cand, bomb=0, decision_frames=2, release_shoot=False) == [
        (mask, 2)]


def test_decision_47_really_takes_the_two_segment_path():
    host = FakeHost()
    # 49 decisions: decision 47 is the release, 48 shows the rising edge back.
    summary = drive(host, max_controlled_frames=98)
    assert summary["terminal"]["kind"] == "truncated"
    assert summary["steps"] == 49
    assert summary["frames"] == 98

    rows = summary["rows"]
    cand = 12
    mask = gym_actions.cand_to_mask(cand, 0)
    held = mask & ~BTN_SHOOT & ~BTN_BOMB
    row47 = rows[47]
    assert row47["decision_index"] == 47
    assert row47["release"] is True
    assert row47["segments"] == [[mask, 1, 1], [held, 1, 1]]
    assert row47["requested_mask"] == mask
    for index in (0, 45, 46):
        assert rows[index]["release"] is False
        # One STEP per frame: a non-release decision is TWO one-frame STEPs,
        # never one two-frame STEP (that is what the frozen loop sends).
        assert rows[index]["segments"] == [[mask, 1, 1], [mask, 1, 1]]

    # The freed action history row is the candidate, pushed only after it ran,
    # once per decision (never once per STEP unit).
    assert summary["history_pushes"] == 49
    assert summary["releases_planned"] == 1
    assert summary["releases_applied"] == 1
    assert summary["shoot_frames"] == 97
    assert summary["no_shoot_frames"] == 1
    # One rising edge at the very first frame, one after the release.
    assert summary["shoot_rising_edges"] == 2
    assert summary["step_units"] == 98
    assert summary["renders"] == 1 + 98

    # Every STEP is one frame; decision 47 is the only released pair.
    sent = list(host.steps)
    assert len(sent) == 98
    assert all(frames == 1 for _mask, frames in sent)
    assert sent[94] == (mask, 1)
    assert sent[95] == (held, 1)
    assert sent[96] == (mask, 1)
    assert [step for step in sent if step[0] == held] == [(held, 1)]


def test_mask_trace_hashes_every_step_as_two_bytes():
    host = FakeHost()
    summary = drive(host, max_controlled_frames=8)
    masks = [mask for mask, _frames in host.steps]
    assert len(masks) == summary["step_units"] == 8
    expected = hashlib.sha256(b"".join(
        int(mask).to_bytes(2, "little") for mask in masks)).hexdigest()
    assert summary["mask_trace_sha256"] == expected
    # Four bytes per decision, the two one-frame STEPs of each decision.
    assert 2 * 2 * summary["steps"] == 2 * len(masks)

    cands = [row["cand"] for row in summary["rows"]]
    action_expected = hashlib.sha256(b"".join(
        np.asarray([cand], dtype=np.uint8).tobytes() for cand in cands)).hexdigest()
    assert summary["action_sequence_sha256"] == action_expected


def test_single_frame_decision_releases_only_the_held_mask():
    # decision_frames=1: the release decision has one STEP, and it is the held
    # mask (the frozen ``units > 1`` rule cannot release inside one unit).
    mask = gym_actions.cand_to_mask(12, 0)
    held = mask & ~BTN_SHOOT & ~BTN_BOMB
    assert SLICE.commit_frames(12, bomb=0, decision_frames=1,
                               release_shoot=True) == [(held, 1)]
    assert SLICE.commit_frames(12, bomb=0, decision_frames=1,
                               release_shoot=False) == [(mask, 1)]
    host = FakeHost()
    summary = drive(host, max_controlled_frames=96, decision_frames=1)
    assert summary["steps"] == 96
    assert summary["history_pushes"] == 96
    assert summary["terminal"]["kind"] == "truncated"
    assert all(frames == 1 for _mask, frames in host.steps)
    assert all(len(row["segments"]) == 1 for row in summary["rows"])
    # release_due(47, 48) is still true, but a one-unit decision never releases.
    assert summary["releases_planned"] == 0
    assert summary["releases_applied"] == 0


def test_partial_tail_is_opt_in_and_never_exceeds_the_cap():
    # Default: the cap rounds down to whole decisions (4 frames at cap 5).
    host = FakeHost()
    summary = drive(host, max_controlled_frames=5, decision_frames=2)
    assert summary["controlled_frames"] == 4
    assert summary["terminal"]["reason"] == "frame_cap"

    # Opt-in: the remaining unit is issued as the frozen v4.2.0 tail does, and
    # the cap is still not exceeded.
    host = FakeHost()
    summary = drive(host, max_controlled_frames=5, decision_frames=2,
                    allow_partial_tail=True)
    assert summary["controlled_frames"] == 5
    assert summary["frames"] == 5
    assert summary["steps"] == 3
    assert summary["terminal"]["kind"] == "truncated"
    assert summary["terminal"]["reason"] == "frame_cap"
    assert host.steps[-1] == (gym_actions.cand_to_mask(12, 0), 1)
    assert len(host.steps) == 5


# ---------------------------------------------------------------------------
# 4. truncation
# ---------------------------------------------------------------------------
def test_truncation_never_exceeds_the_controlled_frame_cap():
    host = FakeHost()
    summary = drive(host, max_controlled_frames=5, decision_frames=2)
    assert summary["terminal"]["kind"] == "truncated"
    assert summary["terminal"]["reason"] == "frame_cap"
    assert summary["controlled_frames"] == 4
    assert summary["frames"] == 4
    assert summary["frames"] <= 5
    assert summary["terminal"]["unspent_frames"] == 1
    assert summary["steps"] == 2
    assert sum(int(frames) for _mask, frames in host.steps) == 4
    assert host.steps == [(gym_actions.cand_to_mask(12, 0), 1),
                          (gym_actions.cand_to_mask(12, 0), 1)] * 2
    assert SLICE._accepted("pixel-smoke", summary["terminal"],
                           controlled_frames=4, max_controlled_frames=5,
                           decision_frames=2) is True
    assert SLICE._accepted("teacher-calibration", summary["terminal"],
                           controlled_frames=4, max_controlled_frames=5,
                           decision_frames=2) is False
    # A cap so small that not even one decision fits is not "the full budget".
    assert SLICE._accepted("pixel-smoke", summary["terminal"],
                           controlled_frames=0, max_controlled_frames=1,
                           decision_frames=2) is False


# ---------------------------------------------------------------------------
# 5. legal death
# ---------------------------------------------------------------------------
def test_game_over_is_classified_legal_death_and_the_trajectory_is_kept():
    host = FakeHost(game_over_at=4)
    summary = drive(host, max_controlled_frames=600)
    assert summary["terminal"]["kind"] == "legal_death"
    assert summary["terminal"]["reason"] == "game_over"
    assert summary["frames"] == 4
    # The trajectory is recorded, not discarded: every issued decision has a row.
    assert summary["steps"] == len(summary["rows"]) == 2
    assert summary["rows"][-1]["ended_after"] is True
    assert summary["rows"][-1]["segments"] == [
        [gym_actions.cand_to_mask(12, 0), 1, 1],
        [gym_actions.cand_to_mask(12, 0), 1, 1]]
    # A game over is a legal death but not this subcommand's acceptance: the
    # ticket's 0 is "the full budget or a legal first death".
    assert SLICE._accepted("pixel-smoke", summary["terminal"],
                           controlled_frames=4, max_controlled_frames=600,
                           decision_frames=2) is False


def test_deaths_increase_is_a_legal_death_and_still_effective():
    host = FakeHost(death_at=3, death_lives=1.0)
    summary = drive(host, max_controlled_frames=600, stop_on_first_death=True)
    assert summary["terminal"]["kind"] == "legal_death"
    assert summary["terminal"]["reason"] == "first_death"
    assert summary["frames"] == 3
    assert summary["steps"] == len(summary["rows"]) == 2
    assert any("death recorded" in warning for warning in summary["warnings"])
    assert SLICE._accepted("pixel-smoke", summary["terminal"],
                           controlled_frames=3, max_controlled_frames=600,
                           decision_frames=2) is True

    # With no lives left the same event is a game over, which is not accepted.
    host = FakeHost(death_at=3, death_lives=0.0)
    summary = drive(host, max_controlled_frames=600, stop_on_first_death=True)
    assert summary["terminal"]["kind"] == "legal_death"
    assert summary["terminal"]["reason"] == "game_over_no_lives"
    assert SLICE._accepted("pixel-smoke", summary["terminal"],
                           controlled_frames=3, max_controlled_frames=600,
                           decision_frames=2) is False


# ---------------------------------------------------------------------------
# 6. missing / broken EVENTS
# ---------------------------------------------------------------------------
def test_events_without_stage_clear_edge_frame_is_technical_invalid():
    host = FakeHost(events_mode="missing-edge-frame")
    summary = drive(host, max_controlled_frames=600)
    assert summary["terminal"]["kind"] == "technical_invalid"
    assert "stageClearEdgeFrame" in summary["terminal"]["reason"]
    conflict = summary["first_conflict"]
    assert conflict is not None
    assert conflict["decision_index"] == 0
    assert conflict["frame"] == 1
    assert conflict["mask"] == gym_actions.cand_to_mask(12, 0)
    assert "stageClearEdgeFrame" in conflict["reason"]


def test_unknown_events_command_is_technical_invalid():
    host = FakeHost(events_mode="error")
    summary = drive(host, max_controlled_frames=600)
    assert summary["terminal"]["kind"] == "technical_invalid"
    assert "unknown-command EVENTS" in summary["terminal"]["reason"]
    assert summary["first_conflict"]["decision_index"] == 0
    assert summary["first_conflict"]["frame"] == 1
    assert summary["first_conflict"]["mask"] == gym_actions.cand_to_mask(12, 0)
    # A missing signal is never read as "not cleared, keep going".
    assert SLICE._accepted("pixel-smoke", summary["terminal"],
                           controlled_frames=1, max_controlled_frames=600,
                           decision_frames=2) is False


def test_pixel_frame_relation_is_checked_and_fails_closed():
    # Both legal engine models pass: rl_v3's own STEP publish (header ==
    # EVENTS.frames) and the early-return/exit path (header == EVENTS.frames-1).
    for offset, relation in ((0, "state_publish"), (-1, "draw_publish")):
        good = FakeHost(pixel_frame_offset=offset)
        summary = drive(good, max_controlled_frames=4)
        assert summary["terminal"]["kind"] == "truncated"
        for row in summary["frame_rows"]:
            assert row["pixel_frame_ok"] is True
            assert row["header_frame_relation"] == relation
            assert row["pixel_frame"] == row["events_frames"] - 1

    bad = FakeHost(pixel_frame_offset=-2)
    summary = drive(bad, max_controlled_frames=600)
    assert summary["terminal"]["kind"] == "technical_invalid"
    assert "pixel frame" in summary["terminal"]["reason"]
    assert summary["first_conflict"]["frame"] == 1
    # The evidence, not a story: two reads, both numbers, the command order.
    conflict = summary["first_conflict"]
    assert conflict["capture_reads"] == 2
    assert [item["header_frame"] for item in conflict["captures"]] == [-1, -1]
    assert conflict["events_frames"] == 1
    assert conflict["legal_header_frames"] == [0, 1]
    assert conflict["audit_sequence"] == ["step", "shm", "events", "shm"]
    assert conflict["pixel_frame_rule"] == SLICE.PIXEL_FRAME_RULE


def test_step_end_state_publish_is_what_the_pre_fix_order_misread():
    """The real-machine failure, reconstructed from the engine's own pump.

    ``rl_v3``'s ``STEP`` ends with ``PublishState`` (``rlharness.cpp:901`` ->
    ``:1362``), so the header already reads ``FrameCount()`` when ``STEP``
    returns, one frame above the RGBA bytes.  Reading the block after ``STATE``
    yields exactly the run-2 numbers (pixel frame 1 against ``EVENTS.frames - 1``
    = 0); reading it first and classifying the pair yields the frame of the
    bytes, 0.

    ``StatePublishHost`` covers the other half of the same story (the header is
    still ``k`` after ``STEP`` and ``STATE`` is what rewrites it to ``k + 1``).
    """

    for host_class in (FakeHost, StatePublishHost):
        host = host_class()
        # The pre-fix order: step, then STATE, then read the block.
        host.step(gym_actions.cand_to_mask(12, 0), 1)
        host.state()
        header_after_state, _rgba = host.pixels()
        events_frames = int(
            host.events_raw().split("frames=")[1].split()[0])
        assert events_frames == 1
        # This is why the old check had to fire: 1 != 1 - 1.
        assert header_after_state["frame"] == events_frames
        assert header_after_state["frame"] != events_frames - 1

        # The fixed loop on the same host model: no conflict, and the recorded
        # pixel_frame is the frame of the captured bytes.
        summary = drive(host_class(), max_controlled_frames=4)
        assert summary["terminal"]["kind"] == "truncated"
        assert summary["first_conflict"] is None
        for row in summary["frame_rows"]:
            assert row["pixel_frame"] == row["events_frames"] - 1
            # The fake host's own model: k, or k + 1 after the STATE rewrite.
            if host_class is FakeHost:
                assert row["header_frame"] == row["events_frames"]
                assert row["header_frame_relation"] == "state_publish"
            else:
                assert row["header_frame"] == row["events_frames"] - 1
                assert row["header_frame_relation"] == "draw_publish"
        assert [row["pixel_frame"] for row in summary["rows"]] == [1, 3]
        assert all(row["pixel_frame_relation"] in (
            "state_publish", "draw_publish") for row in summary["rows"])


def test_audit_command_order_is_step_shm_events_state_hud_pixels():
    """The per-unit order is fixed; the read-only tail is gated.

    V831-SLICE-01b issues ``step -> shm -> events -> state`` on **every** unit
    and appends the read-only ``hud``/``pixels`` tail only on the units named by
    :func:`v8_31_slice.gated_audit_plan` (here: unit 0; four units never reach
    the 96-unit pulse period).  The order that matters for the run-2 bug is
    unchanged: the shared block is read before any command that can publish.
    """

    host = FakeHost()
    summary = drive(host, max_controlled_frames=4)
    assert summary["steps"] == 2

    groups: list[list[str]] = []
    current: list[str] | None = None
    for call in host.calls:
        if call == "step":
            current = [call]
            groups.append(current)
        elif current is not None:
            current.append(call)
    bare = list(SLICE.AUDIT_COMMAND_SEQUENCE)
    gated = bare + ["hud", "pixels"]
    assert bare == ["step", "shm", "events", "state"]
    assert groups == [gated] + [bare] * (len(groups) - 1)
    assert len(groups) == summary["step_units"] == 4
    # Exactly one HUD and one PIXELS per run so far; every unit was sampled.
    assert summary["audit_commands_sent"] == {
        "step": 4, "shm": 4, "events": 4, "state": 4, "hud": 1, "pixels": 1}
    assert summary["audit_sampled_units"] == 4
    assert summary["audit_stride"] == 1
    assert summary["audit_hud_period_units"] == SLICE.AUDIT_HUD_PERIOD_UNITS
    assert summary["audit_command_sequence"] == bare
    assert summary["hud_measured_units"] == 1
    assert summary["hud_mismatches"] == 0
    assert summary["pixels_checksum_mismatches"] == 0
    for row in summary["rows"]:
        assert row["capture_reads"] == row["advanced_total"]
        assert row["audit_sequence"] == bare
        assert row["audit_gated"] == []
        assert row["state_sampled"] is True
    for index, row in enumerate(summary["frame_rows"]):
        assert row["audit_sequence"] == (gated if index == 0 else bare)
        assert row["capture_reads"] == 1
        # The block is read once per unit: no retry fired.
        assert row["header_frame_relation"] == "state_publish"
        assert row["hud_measured"] is (index == 0)
        assert row["hud_source"] == SLICE.HUD_SOURCE_STATE


def test_gated_audit_constants_and_plan_state_the_conditions():
    """``AUDIT_COMMAND_GATED`` is data, and the plan is a pure function."""

    assert SLICE.AUDIT_COMMAND_SEQUENCE == ("step", "shm", "events", "state")
    assert set(SLICE.AUDIT_COMMAND_GATED) == {"hud", "pixels"}
    assert SLICE.AUDIT_HUD_PERIOD_UNITS == 96
    assert SLICE.AUDIT_HUD_PERIOD_UNITS == (
        SLICE.SHOOT_PULSE_PERIOD * SLICE.DECISION_FRAMES)
    assert SLICE.HUD_SOURCE_STATE == "state_derived"
    plan = SLICE.gated_audit_plan
    assert plan(unit_number=0, hud_period=96) == {"hud": True, "pixels": True}
    assert plan(unit_number=1, hud_period=96) == {"hud": False, "pixels": False}
    assert plan(unit_number=95, hud_period=96) == {
        "hud": False, "pixels": False}
    assert plan(unit_number=96, hud_period=96) == {
        "hud": True, "pixels": False}
    assert plan(unit_number=192, hud_period=96) == {
        "hud": True, "pixels": False}
    # The period follows the effective pulse: period=2 units here.
    assert plan(unit_number=2, hud_period=2) == {"hud": True, "pixels": False}
    for name, rule in SLICE.AUDIT_COMMAND_GATED.items():
        assert rule["first_unit"] is True
        assert isinstance(rule["when"], str) and rule["when"]
        assert "after the loop" in rule["last_unit"]
        assert rule["replaced_by"]
        assert "gate" in rule["difference_policy"]


def test_hud_is_read_once_per_pulse_period_not_every_unit():
    """A short pulse makes the cadence observable on a small drive."""

    host = FakeHost()
    summary = drive(host, max_controlled_frames=8, shoot_pulse_period=1)
    # 8 units, one decision per 2 units -> hud_period = 1 decision x 2 = 2 units.
    assert summary["audit_hud_period_units"] == 2
    groups: list[list[str]] = []
    current: list[str] | None = None
    for call in host.calls:
        if call == "step":
            current = [call]
            groups.append(current)
        elif current is not None:
            current.append(call)
    with_hud = [index for index, group in enumerate(groups) if "hud" in group]
    assert with_hud == [0, 2, 4, 6]
    assert [index for index, group in enumerate(groups) if "pixels" in group] == [0]
    assert summary["audit_commands_sent"]["hud"] == 4
    assert summary["audit_commands_sent"]["pixels"] == 1
    assert summary["hud_measured_units"] == 4
    assert summary["hud_crosschecks"] == 4
    assert summary["hud_mismatches"] == 0


#: Digests of the ``FakeHost``/``ConstantActions(12)``/8-unit script measured
#: with the **pre-slim** instrument (six commands per unit) and re-asserted here
#: after the slim: the mask trace and the action sequence are independently
#: reconstructable from the masks and the candidates, the observation digest is
#: the same script's fixed value.
PRE_SLIM_MASK_TRACE_SHA256 = (
    "57712907a2f30022a93ba397f0207af1bdaca4c134120f2c829ee7c8c205b2f1")
PRE_SLIM_ACTION_SEQUENCE_SHA256 = (
    "009391559b2777d7003b4a8f2ac076416b66eee634361c208c85dae6d174cda0")
PRE_SLIM_OBSERVED_INPUT_SHA256 = (
    "4d3654e7bc817f2e667147a790e400aee4deb7dbdc7d9bc75ef09e5a42fe81c1")


def test_the_slim_does_not_change_one_digest_bit():
    summary = drive(FakeHost(), max_controlled_frames=8)
    assert summary["mask_trace_sha256"] == PRE_SLIM_MASK_TRACE_SHA256
    assert summary["action_sequence_sha256"] == PRE_SLIM_ACTION_SEQUENCE_SHA256
    assert summary["observed_input_sha256"] == PRE_SLIM_OBSERVED_INPUT_SHA256
    # Independently reconstructed: one 2-byte little-endian mask per STEP unit,
    # one uint8 candidate per decision.  The gated HUD/PIXELS never touch them.
    assert summary["audit_command_sequence"] == [
        "step", "shm", "events", "state"]
    mask = gym_actions.cand_to_mask(12, 0)
    rebuilt_mask = hashlib.sha256(
        b"".join(int(mask).to_bytes(2, "little")
                 for _ in range(summary["step_units"]))).hexdigest()
    rebuilt_action = hashlib.sha256(
        bytes([12] * summary["steps"])).hexdigest()
    assert summary["mask_trace_sha256"] == rebuilt_mask
    assert summary["action_sequence_sha256"] == rebuilt_action
    assert summary["distinct_masks"] == 1
    assert summary["mask_counts"] == {str(int(mask)): 8}


def test_audit_stride_thins_the_observations_and_keeps_the_digests():
    host = FakeHost()
    summary = drive(host, max_controlled_frames=8, audit_stride=2)
    assert summary["audit_stride"] == 2
    assert summary["audit_sampled_units"] == 4
    # STEP and the shared-block capture are still one per unit.
    assert summary["audit_commands_sent"]["step"] == 8
    assert summary["audit_commands_sent"]["shm"] == 8
    # EVENTS/STATE are halved.
    assert summary["audit_commands_sent"]["events"] == 4
    assert summary["audit_commands_sent"]["state"] == 4
    groups: list[list[str]] = []
    current: list[str] | None = None
    for call in host.calls:
        if call == "step":
            current = [call]
            groups.append(current)
        elif current is not None:
            current.append(call)
    assert groups[1] == ["step", "shm"]
    assert groups[3] == ["step", "shm"]
    assert groups[0] == ["step", "shm", "events", "state", "hud", "pixels"]
    # The three digests are the stride-1 values, bit for bit.
    assert summary["mask_trace_sha256"] == PRE_SLIM_MASK_TRACE_SHA256
    assert summary["action_sequence_sha256"] == PRE_SLIM_ACTION_SEQUENCE_SHA256
    assert summary["observed_input_sha256"] == PRE_SLIM_OBSERVED_INPUT_SHA256
    # A skipped unit says so instead of inventing a frame or a reading.
    skipped = [row for row in summary["frame_rows"] if not row["state_sampled"]]
    assert len(skipped) == 4
    for row in skipped:
        assert row["pixel_frame"] is None
        assert row["pixel_frame_ok"] is None
        assert row["events_frames"] is None
        assert row["audit_sequence"] == ["step", "shm"]
        # The captured bytes are still hashed on a skipped unit: FNV when it is
        # the decision's last unit (these are), and blake2b always.
        assert len(row["pixels_checksum_local"]) == 16
        assert len(row["pixel_frame_blake2b128"]) == 32
    # The per-frame digest is never thinned by the stride; the FNV keeps its
    # per-decision scope.
    assert summary["blake2b_units"] == summary["step_units"] == 8
    assert summary["fnv_units"] == summary["steps"] == 4
    assert summary["fnv_crosscheck_units"] == 1
    # The decision rows fall back to the last sampled unit for the frame.
    assert [row["pixel_frame"] for row in summary["rows"]] == [0, 2, 4, 6]
    assert [row["pixel_frame_unit"] for row in summary["rows"]] == [0, 0, 0, 0]
    assert all(row["pixel_frames"] == [row["pixel_frame"], None]
               for row in summary["rows"])
    assert any("audit_stride=2" in warning for warning in summary["warnings"])


def test_pixel_frame_blake2b128_matches_known_vectors():
    """The per-frame C-level identity is blake2b-128, pinned by vectors."""

    assert SLICE.PIXEL_FRAME_DIGEST_ALGORITHM == "blake2b-128"
    assert SLICE.PIXEL_FRAME_DIGEST_FIELD == "pixel_frame_blake2b128"
    # Independently computed with hashlib.blake2b(..., digest_size=16).
    assert SLICE.pixel_frame_blake2b128(b"") == (
        "cae66941d9efbd404e4d88758ea67670")
    assert SLICE.pixel_frame_blake2b128(b"v831") == (
        "48775f021ea45f213ac402907eed32ee")
    assert SLICE.pixel_frame_blake2b128(b"\x00" * 64) == (
        "532fe82fc5db1ef54fd5be9ba44b69b4")
    assert SLICE.pixel_frame_blake2b128(bytes(range(16))) == (
        "f9477b91722fe719f946834577e0e8a2")
    # It is neither the FNV nor sha256: 32 hex digits over the same bytes.
    pixels = make_pixels()
    raw = np.ascontiguousarray(np.asarray(pixels)[::-1]).tobytes()
    assert SLICE.pixel_frame_blake2b128(raw) == hashlib.blake2b(
        raw, digest_size=16).hexdigest()
    assert SLICE.pixel_frame_blake2b128(raw) != SLICE.fnv1a64_hex(raw)


def test_digest_scopes_are_per_decision_fnv_and_per_unit_blake2b():
    """The FNV is the ticket's per-decision value; the blake2b is per unit."""

    host = FakeHost()
    summary = drive(host, max_controlled_frames=8)
    assert summary["checksum_scope"] == "per_decision"
    assert summary["frame_digest_scope"] == "per_unit_blake2b128"
    assert "每次决策" in summary["checksum_scope_rationale"]
    assert summary["pixel_frame_digest_algorithm"] == "blake2b-128"
    assert summary["steps"] == 4
    # One FNV per decision, plus exactly one extra for the gated first PIXELS.
    assert summary["fnv_units"] == summary["steps"] == 4
    assert summary["fnv_crosscheck_units"] == 1
    # Every STEP unit has a per-frame identity, stride or not.
    assert summary["blake2b_units"] == summary["step_units"] == 8
    unit0, unit1, unit2 = (summary["frame_rows"][index] for index in (0, 1, 2))
    # The fake host serves the same RGBA every unit, so the identity repeats --
    # what matters is that it exists on every frame row and is 32 hex digits.
    assert len(unit2["pixel_frame_blake2b128"]) == 32
    assert unit0["pixel_frame_blake2b128"] == unit2["pixel_frame_blake2b128"]
    # FNV: first unit (cross-check) and last unit of the decision only.  Unit 2
    # is the first unit of decision 1 -- neither -- so it is not computed.
    assert len(unit0["pixels_checksum_local"]) == 16
    assert len(unit1["pixels_checksum_local"]) == 16
    assert unit2["pixels_checksum_local"] is None
    assert unit2["pixels_checksum_matches"] is None
    # The decision row records its last unit's values.
    assert summary["rows"][0]["pixels_checksum_local"] == unit1[
        "pixels_checksum_local"]
    assert summary["rows"][0]["pixel_frame_blake2b128"] == unit1[
        "pixel_frame_blake2b128"]
    assert all(len(row["pixel_frame_blake2b128"]) == 32
               for row in summary["rows"])


def test_timeout_defaults_are_unchanged_and_any_positive_value_is_accepted():
    """``--timeout-s`` keeps 120/600; the CLI accepts any positive number."""

    assert SLICE.DEFAULT_TIMEOUT_S == {
        "pixel-smoke": 120.0, "teacher-calibration": 600.0}
    parser = SLICE.build_parser()
    common = ["--engine", "/fake/e", "--data-source", "/fake/d",
              "--out", "/fake/o", "--seed", "1", "--max-controlled-frames", "4"]
    teacher = ["teacher-calibration", *common, "--tape", "/fake/t"]
    # No --timeout-s: None here, i.e. main() applies the unchanged default.
    assert parser.parse_args(teacher).timeout_s is None
    assert parser.parse_args(
        [*teacher, "--timeout-s", "2400"]).timeout_s == 2400.0
    assert parser.parse_args(
        [*teacher, "--timeout-s", "900"]).timeout_s == 900.0


def test_engine_hud_disagreement_only_warns():
    """A HUD snapshot that disagrees with STATE is a warning, never a gate."""

    class LoudHudHost(FakeHost):
        def hud(self):
            snapshot = dict(super().hud())
            snapshot["lives"] = int(snapshot["lives"]) + 1
            return snapshot

    host = LoudHudHost()
    summary = drive(host, max_controlled_frames=4)
    assert summary["terminal"]["kind"] == "truncated"
    assert summary["first_conflict"] is None
    assert summary["hud_crosschecks"] == 1
    assert summary["hud_mismatches"] == 1
    crosscheck = summary["rows"][0]["hud_crosscheck"]
    assert crosscheck["matches"]["lives"] is False
    assert crosscheck["lives_bombs_match"] is False
    assert crosscheck["engine"]["lives"] == 3
    assert crosscheck["state"]["lives"] == 2
    # The recorded column is the STATE value, never the engine snapshot.
    assert summary["frame_rows"][0]["hud_lives"] == 2
    assert any("HUD disagrees with STATE" in warning
               for warning in summary["warnings"])
    assert any("never a gate" in warning for warning in summary["warnings"])


def test_local_fnv1a64_matches_the_engine_and_known_vectors():
    # The engine's HandlePixels constants, spelled out.
    assert SLICE.FNV1A64_OFFSET_BASIS == 14695981039346656037
    assert SLICE.FNV1A64_PRIME == 1099511628211
    # Reference FNV-1a-64 vectors.
    assert SLICE.fnv1a64(b"") == 0xCBF29CE484222325
    assert SLICE.fnv1a64_hex(b"") == "cbf29ce484222325"
    assert SLICE.fnv1a64(b"a") == 0xAF63DC4C8601EC8C
    assert SLICE.fnv1a64(b"foobar") == 0x85944171F73967E8
    # The two vectors this ticket asks for, computed independently.
    assert SLICE.fnv1a64_hex(b"\x00" * 8) == "a8c7f832281a39c5"
    assert SLICE.fnv1a64_hex(bytes(range(16))) == "7c84dc9477851775"
    # The hex form is exactly the 16 digits a PIXELS reply carries.
    assert len(SLICE.fnv1a64_hex(make_pixels().tobytes())) == 16


def test_pixels_checksum_local_is_the_captured_bytes_and_engine_differences_warn():
    # A matching engine: the local FNV is the captured bytes, and the one gated
    # PIXELS reply (unit 0) agrees with it.
    host = FakeHost()
    summary = drive(host, max_controlled_frames=4)
    assert summary["pixels_checksum_mismatches"] == 0
    assert not any("PIXELS checksum" in warning
                   for warning in summary["warnings"])
    captured = host.rgba
    expected = SLICE.fnv1a64_hex(
        np.ascontiguousarray(np.asarray(captured)[::-1]).tobytes())
    for row in summary["rows"]:
        # Every decision keeps the local value; only the unit that issued the
        # gated PIXELS has an engine value to compare with.
        assert row["pixels_checksum_local"] == expected
        assert row["pixels_checksum_engine"] is None
        assert row["pixels_checksum_matches"] is None
        assert row["pixels_nonzero"] == int(np.count_nonzero(captured))
        assert row["pixels_size"] == "640x480"
    assert summary["rows"][0]["pixels_crosscheck"] == {
        "unit": 0, "local": expected, "engine": expected, "matches": True,
        "error": None}
    assert summary["rows"][0]["pixels_checksum_engine_units"] == [0]
    assert summary["rows"][1]["pixels_crosscheck"] is None

    # A PIXELS reply that disagrees is recorded and warns, but is never a gate.
    host = FakeHost(checksum_mode="mismatch")
    summary = drive(host, max_controlled_frames=4)
    assert summary["terminal"]["kind"] == "truncated"
    assert summary["first_conflict"] is None
    # Only the gated unit is compared, so exactly one mismatch is counted.
    assert summary["pixels_checksum_mismatches"] == 1
    row = summary["rows"][0]
    assert row["pixels_checksum_local"] == expected
    assert row["pixels_crosscheck"]["engine"] == "dead" * 4
    assert row["pixels_crosscheck"]["matches"] is False
    assert row["pixels_checksum_matches"] is None
    assert any("PIXELS checksum" in warning
               for warning in summary["warnings"])
    assert any("not a gate" in warning for warning in summary["warnings"])


def test_a_failed_pixels_reply_is_recorded_and_never_a_gate():
    host = FakeHost(checksum_mode="error")
    summary = drive(host, max_controlled_frames=4)
    assert summary["terminal"]["kind"] == "truncated"
    assert summary["first_conflict"] is None
    row = summary["rows"][0]
    assert row["pixels_checksum_engine"] is None
    assert row["pixels_crosscheck"]["engine"] is None
    assert "pixels-failed" in row["pixels_crosscheck"]["error"]
    assert row["pixels_checksum_matches"] is None
    # The local value is still there: the capture, not the reply, is the record.
    assert len(row["pixels_checksum_local"]) == 16
    assert row["pixels_nonzero"] == int(np.count_nonzero(host.rgba))
    assert summary["pixels_checksum_mismatches"] == 0


def test_events_level_bit_is_not_the_edge():
    host = FakeHost(clear_level_at=2)
    summary = drive(host, max_controlled_frames=8)
    signals = summary["signals"]
    assert signals["first_clear_edge_frame"] == 2
    assert signals["first_clear_edge_frame_reported"] == 2
    assert signals["first_level_bit_frame"] == 2
    assert signals["first_stage_advance_frame"] is None
    assert signals["missing_second_signal"] is True
    assert signals["interval_frames"] is None
    # A real EVENTS clear without a STATE advance is still a clear, and it says
    # which signal is missing instead of inventing the other one.
    assert summary["terminal"]["kind"] == "stage_clear"
    assert summary["terminal"]["reason"] == "events_clear_without_state_advance"


def test_stage_advance_is_the_stage_clear_and_records_the_interval():
    host = FakeHost(clear_level_at=2, stage_advance_at=4)
    summary = drive(host, max_controlled_frames=600)
    signals = summary["signals"]
    assert signals["events_clear_seen"] is True
    assert signals["first_clear_event_frame"] == 2
    assert signals["first_stage_advance_frame"] == 4
    assert signals["stage_advanced"] is True
    assert signals["order"][:2] == ["events_clear",
                                   "state_stage_advance"]
    assert signals["interval_frames"] == 2
    assert signals["missing_second_signal"] is False
    assert summary["terminal"]["kind"] == "stage_clear"
    assert summary["frames"] == 4
    assert SLICE._accepted("teacher-calibration", summary["terminal"],
                           controlled_frames=4, max_controlled_frames=600,
                           decision_frames=2) is True


# ---------------------------------------------------------------------------
# 7. the mute bytes
# ---------------------------------------------------------------------------
def synth_cfg(*, music_mode: int, play_sounds: int, music_volume: int,
              sfx_volume: int, marker: int = 0xA5) -> bytes:
    """A 60-byte ``th08.cfg``-shaped buffer with the four audio bytes set."""

    raw = bytearray(SLICE.CFG_SIZE)
    for index in range(SLICE.CFG_SIZE):
        raw[index] = (index * 7 + marker) & 0xFF
    # A legal version, so the buffer looks like a real config, not filler.
    raw[0x14:0x18] = (0x00080001).to_bytes(4, "little")
    raw[0x1F] = music_mode
    raw[0x20] = play_sounds
    raw[0x27] = music_volume & 0xFF
    raw[0x28] = sfx_volume & 0xFF
    return bytes(raw)


@pytest.mark.parametrize(
    "source",
    [
        synth_cfg(music_mode=1, play_sounds=1, music_volume=39, sfx_volume=36),
        synth_cfg(music_mode=0, play_sounds=0, music_volume=0, sfx_volume=0),
    ],
    ids=["audible-source", "already-muted-source"],
)
def test_muting_changes_only_the_four_offsets(source):
    assert len(source) == SLICE.CFG_SIZE
    muted = SLICE.zero_mute_offsets(source)
    assert len(muted) == SLICE.CFG_SIZE
    assert SLICE.MUTE_OFFSETS == (0x1F, 0x20, 0x27, 0x28)
    for offset in SLICE.MUTE_OFFSETS:
        assert muted[offset] == 0
    others = [index for index in range(SLICE.CFG_SIZE)
              if index not in SLICE.MUTE_OFFSETS]
    assert len(others) == 56
    for index in others:
        assert muted[index] == source[index], (
            f"byte 0x{index:02X} outside the mute offsets changed")

    decoded = SLICE.decode_cfg_mute_fields(muted, path="copy")
    assert decoded["fields"] == {
        "musicMode": 0, "playSounds": 0, "musicVolume": 0, "sfxVolume": 0}
    assert decoded["muted_strict_four_field"] is True
    assert decoded["muted_legacy_three_field"] is True

    diff = SLICE.cfg_copy_diff(source, muted)
    assert diff["comparable"] is True
    assert diff["other_bytes_identical"] is True
    assert diff["unexpected_offsets"] == []
    assert diff["other_byte_count"] == 60 - len(diff["diff_byte_offsets"])
    expected_offsets = [offset for offset in SLICE.MUTE_OFFSETS
                        if source[offset] != 0]
    assert diff["diff_byte_offsets"] == expected_offsets
    if expected_offsets:
        assert diff["diff_offsets_match_expected"] is True
    else:
        assert diff["diff_offsets_match_expected"] is False


def test_audible_and_muted_decodes_are_distinguished():
    audible = synth_cfg(music_mode=1, play_sounds=1, music_volume=39,
                        sfx_volume=36)
    decoded = SLICE.decode_cfg_mute_fields(audible, path="source")
    assert decoded["fields"] == {
        "musicMode": 1, "playSounds": 1, "musicVolume": 39, "sfxVolume": 36}
    assert decoded["muted_strict_four_field"] is False
    assert decoded["muted_legacy_three_field"] is False
    assert decoded["sha256"] == SLICE.sha256_bytes(audible)

    # The legacy rule never looked at musicMode: one with MIDI on but the two
    # volumes and playSounds off passes it and fails the strict rule.
    legacy_only = synth_cfg(music_mode=1, play_sounds=0, music_volume=0,
                            sfx_volume=0)
    decoded = SLICE.decode_cfg_mute_fields(legacy_only)
    assert decoded["muted_legacy_three_field"] is True
    assert decoded["muted_strict_four_field"] is False


def test_wrong_size_cfg_is_an_error_not_a_muted_config():
    decoded = SLICE.decode_cfg_mute_fields(b"\x00" * 59, path="short")
    assert decoded["size_ok"] is False
    assert decoded["fields"] is None
    assert "59" in decoded["error"]
    assert decoded["muted_strict_four_field"] is False
    with pytest.raises(SLICE.Refusal):
        SLICE.zero_mute_offsets(b"\x00" * 59)


def test_mute_report_records_the_byte_diff_and_both_decodes():
    source = synth_cfg(music_mode=1, play_sounds=1, music_volume=39,
                       sfx_volume=36)
    copy_before = SLICE.zero_mute_offsets(source)
    copy_after = copy_before
    report = SLICE.build_mute_report(
        source_path=Path("/nonexistent/source/th08.cfg"),
        copy_path=Path("/nonexistent/copy/th08.cfg"),
        source_raw=source, copy_before_raw=copy_before,
        copy_after_raw=copy_after, engine_env=SLICE.engine_env_record(),
        checked_at=[SLICE.now_iso()], problems=[])
    assert report["schema"] == SLICE.MUTE_SCHEMA
    assert report["source"]["fields"]["playSounds"] == 1
    assert report["copy_before_start"]["fields"] == {
        "musicMode": 0, "playSounds": 0, "musicVolume": 0, "sfxVolume": 0}
    assert report["copy_before_start"]["diff_byte_offsets"] == [0x1F, 0x20, 0x27, 0x28]
    assert report["copy_after_exit"] is not None
    assert report["engine_env"]["SDL_AUDIODRIVER"] == "dummy"
    assert "problems" in report
    # The synthetic source is not the registered file, so the report says so.
    assert report["source"]["matches_registered_source_sha256"] is False
    problems = SLICE.mute_report_problems(report)
    assert any("registered" in message for message in problems)
    assert not any("outside the four mute offsets" in message
                   for message in problems)


def test_an_unmuted_copy_after_exit_is_reported_not_patched(tmp_path):
    source = synth_cfg(music_mode=1, play_sounds=1, music_volume=39,
                       sfx_volume=36)
    copy_before = SLICE.zero_mute_offsets(source)
    audible_again = synth_cfg(music_mode=1, play_sounds=1, music_volume=39,
                              sfx_volume=36, marker=0xA5)
    report = SLICE.build_mute_report(
        source_path=tmp_path / "source.cfg", copy_path=tmp_path / "copy.cfg",
        source_raw=source, copy_before_raw=copy_before,
        copy_after_raw=audible_again, engine_env=SLICE.engine_env_record(),
        checked_at=[SLICE.now_iso()], problems=[])
    assert report["copy_before_start"]["muted_strict_four_field"] is True
    assert report["copy_after_exit"]["muted_strict_four_field"] is False
    problems = SLICE.mute_report_problems(report)
    assert any("NOT strictly muted after exit" in message for message in problems)


# ---------------------------------------------------------------------------
# 8. the teacher tape schedule
# ---------------------------------------------------------------------------
def _write_tape(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8")
    return path


def test_tape_schedule_is_leading_then_executed_then_tail(tmp_path):
    from touhou_rl.v420.rl import load_clear_schedule

    tape = _write_tape(tmp_path / "slots.jsonl", [
        {"decision_frame": 16, "executed_cand": 7, "bomb": 0,
         "episode_seed": 12,
         "history_frames": [0, 2, 4, 6, 8, 10, 12, 14],
         "history": [7, 7, 7, 7, 7, 7, 7, 7]},
        {"decision_frame": 18, "executed_cand": 5, "bomb": 0},
        {"decision_frame": 20, "executed_cand": 3, "bomb": 0},
    ])
    info = load_clear_schedule(tape, tail_decisions=2)
    candidates = [cand for _frame, cand in info["schedule"]]
    assert info["leading_decisions"] == 8
    assert candidates[:8] == [7] * 8
    assert candidates[8:11] == [7, 5, 3]
    assert candidates[11:] == [3, 3]
    assert len(candidates) == 8 + 3 + 2
    frames = [frame for frame, _cand in info["schedule"]]
    assert frames[:8] == [0, 2, 4, 6, 8, 10, 12, 14]
    assert frames[8:11] == [16, 18, 20]
    assert frames[11:] == [22, 24]
    assert info["last_recorded_decision_frame"] == 20
    assert info["slots"] == 3
    assert info["seed"] == 12

    # The tool's own bomb check: every row of a real tape carries bomb = 0.
    assert SLICE.tape_bomb_values(tape) == [0, 0, 0]
    assert SLICE.tape_is_bomb_free(tape) is True

    dirty = _write_tape(tmp_path / "dirty.jsonl", [
        {"decision_frame": 16, "executed_cand": 7, "bomb": 0},
        {"decision_frame": 18, "executed_cand": 5, "bomb": 1},
    ])
    assert SLICE.tape_bomb_values(dirty) == [0, 1]
    assert SLICE.tape_is_bomb_free(dirty) is False


def test_tape_actions_walk_leading_then_recorded_then_tail():
    tape = {
        "candidates": [7] * 8 + [7, 5, 3] + [3, 3],
        "leading_decisions": 8,
        "slots": 3,
    }
    actions = SLICE.TapeActions(tape)
    observed = [actions.decision(index) for index in range(13)]
    assert [cand for cand, _label in observed[:8]] == [7] * 8
    assert [label for _cand, label in observed[:8]] == ["leading"] * 8
    assert [cand for cand, _label in observed[8:11]] == [7, 5, 3]
    assert [label for _cand, label in observed[8:11]] == ["tape"] * 3
    assert [cand for cand, _label in observed[11:]] == [3, 3]
    assert [label for _cand, label in observed[11:]] == ["tail"] * 2
    # Past the schedule the last candidate repeats, and the row says so.
    cand, label = actions.decision(99)
    assert (cand, label) == (3, "tail_past_schedule")


def test_constant_actions_refuse_an_out_of_range_candidate():
    for bad in (-1, gym_actions.N_CAND, 99):
        with pytest.raises(SLICE.Refusal):
            SLICE.ConstantActions(bad)
    assert SLICE.ConstantActions(17).cand == 17


def test_private_game_dir_is_copied_whole_and_only_the_copy_is_muted(tmp_path):
    """The source data directory is read, never written; the copy carries the mute."""

    source = tmp_path / "source-game"
    source.mkdir()
    audible = synth_cfg(music_mode=1, play_sounds=1, music_volume=39,
                        sfx_volume=36)
    (source / "th08.cfg").write_bytes(audible)
    (source / "th08.dat").write_bytes(b"dat" * 100)
    (source / "thbgm.dat").write_bytes(b"bgm" * 50)
    (source / "sub").mkdir()
    (source / "sub" / "extra.bin").write_bytes(b"\x00\x01")
    source_before = {path.name: SLICE.sha256_bytes(path.read_bytes())
                     for path in source.rglob("*") if path.is_file()}

    out = tmp_path / "run"
    out.mkdir()
    meter = SLICE.CostMeter()
    info = SLICE.prepare_private_game_dir(
        out_dir=out, data_source=source, meter=meter)
    copy = out / "game"
    assert copy.is_dir()
    assert (copy / "th08.dat").read_bytes() == b"dat" * 100
    assert (copy / "thbgm.dat").read_bytes() == b"bgm" * 50
    assert (copy / "sub" / "extra.bin").read_bytes() == b"\x00\x01"
    assert Path(info["source_dir"]) == source
    assert Path(info["copy_dir"]) == copy

    # The source is untouched, byte for byte.
    source_after = {path.name: SLICE.sha256_bytes(path.read_bytes())
                    for path in source.rglob("*") if path.is_file()}
    assert source_after == source_before
    assert (source / "th08.cfg").read_bytes() == audible

    # Only the copy's cfg was muted, and only at the four offsets.
    muted = (copy / "th08.cfg").read_bytes()
    assert SLICE.decode_cfg_mute_fields(muted)["muted_strict_four_field"] is True
    diff = SLICE.cfg_copy_diff(audible, muted)
    assert diff["diff_byte_offsets"] == [0x1F, 0x20, 0x27, 0x28]
    assert diff["other_bytes_identical"] is True
    assert info["copied_cfg_sha256_before_mute"] == SLICE.sha256_bytes(audible)
    assert info["copied_cfg_sha256_after_mute"] == SLICE.sha256_bytes(muted)

    # A second run into the same directory is refused instead of clobbering it.
    with pytest.raises(SLICE.Refusal):
        SLICE.prepare_private_game_dir(
            out_dir=out, data_source=source, meter=meter)


def test_engine_environment_record_forces_the_dummy_audio_driver():
    record = SLICE.engine_env_record()
    assert record["SDL_AUDIODRIVER"] == "dummy"
    assert record["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert record["GALLIUM_DRIVER"] == "llvmpipe"
    assert "DISPLAY" in record
    assert SLICE.XVFB_GUARD_ENV in record
    assert SLICE.FORCED_ENGINE_ENV == {"SDL_AUDIODRIVER": "dummy"}


def test_cost_meter_accounts_every_bucket():
    meter = SLICE.CostMeter()
    meter.begin()
    with meter.measure("step"):
        pass
    with meter.measure("audit"):
        pass
    meter.add("io", 0.0)
    meter.add("session_close", 0.0)
    meter.add("copytree", 0.0)
    meter.add("preflight", 0.0)
    meter.add("capture", 0.0)
    meter.finish()
    payload = meter.payload(step_units=3)
    assert set(payload["seconds"]) == set(SLICE.CostMeter.BUCKETS)
    assert payload["seconds"]["resume"] == 0.0
    assert payload["calls"]["step"] == 1
    assert payload["calls"]["audit"] == 1
    assert payload["total_seconds"] is not None
    assert payload["unaccounted_seconds"] is not None
    assert payload["unaccounted_by_unit"] is not None
    assert payload["unaccounted_share"] is not None
    assert payload["bucket_definitions"]["resume"].startswith("always 0")
    # Task A: the three new leaf buckets exist and mean what their definitions say.
    assert set(SLICE.CostMeter.LEAF_BUCKETS) == {
        "startup", "resume", "readback", "actor", "step", "audit", "io",
        "capture", "session_close", "copytree", "preflight"}
    assert SLICE.CostMeter.PARENT_BUCKETS == ("unit_total",)
    assert set(payload["bucket_definitions"]) == set(SLICE.CostMeter.BUCKETS)


def test_cost_meter_parent_contains_children_without_double_counting():
    """``unit_total`` is a parent: accounted sums leaves only, exactly once."""

    meter = SLICE.CostMeter()
    meter.begin()
    with meter.measure("unit_total"):
        with meter.measure("step"):
            time.sleep(0.01)
        with meter.measure("capture"):
            time.sleep(0.01)
        time.sleep(0.01)  # in-unit work no leaf covers
    with meter.measure("copytree"):
        time.sleep(0.005)
    meter.finish()
    payload = meter.payload(step_units=1)

    leaves = sum(payload["seconds"][name]
                 for name in SLICE.CostMeter.LEAF_BUCKETS)
    # Each reported value is rounded to 1e-6, so a sum of reported values is
    # compared with a tolerance that covers that rounding.
    assert payload["accounted_seconds"] == pytest.approx(leaves, abs=5e-6)
    # The parent's total is *not* added to the leaves it contains.
    assert payload["parent_seconds"]["unit_total"] > (
        payload["seconds"]["step"] + payload["seconds"]["capture"])
    assert payload["accounted_inside_parent_seconds"] == pytest.approx(
        payload["seconds"]["step"] + payload["seconds"]["capture"], abs=5e-6)
    # unit_total minus its children is the in-unit work no leaf measured.
    assert payload["unit_internal_seconds"] >= 0.008
    assert payload["unit_internal_by_unit"] == pytest.approx(
        payload["unit_internal_seconds"], abs=1e-6)
    assert payload["accounted_seconds"] < payload["total_seconds"]
    assert payload["unaccounted_seconds"] == pytest.approx(
        payload["total_seconds"] - payload["accounted_seconds"], abs=5e-6)
    assert payload["unaccounted_by_unit"] == pytest.approx(
        payload["unaccounted_seconds"] / 1, abs=1e-6)
    assert 0.0 <= payload["unaccounted_share"] <= 1.0
    assert "CONTAINS" in payload["nested_measure_note"]
    assert "accounted_seconds" in payload["nested_measure_note"]


def test_cost_attribution_leaves_almost_nothing_unaccounted():
    """Task A's goal: with ``capture`` and ``unit_total`` in place, almost all
    of the wall clock is inside a named bucket instead of ``unaccounted``."""

    meter = SLICE.CostMeter()
    meter.begin()
    host = MeteredHost(meter)
    summary = drive(host, max_controlled_frames=16, meter=meter)
    meter.finish()
    payload = meter.payload(step_units=summary["step_units"])
    assert summary["step_units"] == 16
    assert payload["unaccounted_share"] < 0.05, payload["unaccounted_share"]
    # The local digests still dominate -- that is the per-decision FNV plus the
    # per-unit blake2b, not protocol traffic -- but they are now ~55 ms per
    # decision instead of ~55 ms per unit.
    assert payload["seconds"]["capture"] > 0.3
    assert payload["seconds"]["capture"] > 4 * payload["seconds"]["step"]
    assert payload["seconds"]["capture"] > 4 * payload["seconds"]["audit"]
    # The parent covers the children, and its residual (row building, digest
    # bookkeeping) is a small per-unit number rather than minutes.
    assert payload["parent_seconds"]["unit_total"] >= payload["seconds"]["capture"]
    assert payload["unit_internal_by_unit"] < 0.01
    # Self-consistent: the parent is not added to the leaves it contains.
    leaves = sum(payload["seconds"][name]
                 for name in SLICE.CostMeter.LEAF_BUCKETS)
    assert payload["accounted_seconds"] == pytest.approx(leaves, abs=5e-6)
    assert payload["accounted_seconds"] < payload["parent_seconds"]["unit_total"]


def test_capture_is_measured_so_it_cannot_be_unaccounted_again():
    """The local RGBA digest is the 60 ms/unit that hid in ``unaccounted``."""

    meter = SLICE.CostMeter()
    meter.begin()
    summary = drive(FakeHost(), max_controlled_frames=4, meter=meter)
    meter.finish()
    payload = meter.payload(step_units=summary["step_units"])
    assert payload["calls"]["capture"] == summary["step_units"] == 4
    # The digest of a 1.2 MB RGBA block is real work, not noise.
    assert payload["seconds"]["capture"] > 0.05
    assert payload["parent_calls"]["unit_total"] == summary["step_units"]
    assert payload["parent_seconds"]["unit_total"] >= (
        payload["seconds"]["capture"])
    assert payload["accounted_inside_parent_seconds"] >= (
        payload["seconds"]["capture"])


# ---------------------------------------------------------------------------
# 9. the frozen observation contract
# ---------------------------------------------------------------------------
def test_obs_spec_v3_digest_is_unchanged():
    assert obs_spec_v3.obs_spec_v3_digest() == (
        "ef2fbd15385f26f425db2bfe855d105ee009023acb08c26916e811963553a105")
    assert SLICE.REGISTERED_OBS_V3_DIGEST == (
        "ef2fbd15385f26f425db2bfe855d105ee009023acb08c26916e811963553a105")
    assert SLICE.EXPECTED_TEACHER_CALIBRATION["shoot"]["mask_trace_sha256"] == (
        "66c81344b00f8766f7d9e860d1874b1141c699122f491ed1a8ed0aea9f179924")
    flat = SLICE._flatten(SLICE.EXPECTED_TEACHER_CALIBRATION)
    for key in ("frames", "score", "decisions", "state_frame", "gameplay_frame",
                "lives_end", "deaths", "renders", "pixels_warmup_frames",
                "shoot.shoot_frames", "shoot.no_shoot_frames",
                "shoot.releases_planned", "shoot.releases_applied",
                "shoot.shoot_rising_edges", "shoot.distinct_masks",
                "shoot.mask_trace_sha256", "action_sequence_sha256",
                "observed_input_sha256", "harness_role", "obs_window"):
        assert key in flat, f"frozen expectation is missing {key}"
    assert flat["frames"] == 13195
    assert flat["decisions"] == 6598
    assert flat["state_frame"] == 13195
    assert flat["gameplay_frame"] == 13196


# ---------------------------------------------------------------------------
# extra: the observations the loop records per decision / per frame
# ---------------------------------------------------------------------------
def test_decisions_carry_the_required_audit_fields():
    host = FakeHost()
    summary = drive(host, max_controlled_frames=8)
    assert summary["steps"] == 4
    required = {
        "decision_index", "logical_frame", "pixel_frame", "pixel_frames",
        "pixel_frame_header", "pixel_frame_relation", "pixel_frame_unit",
        "pixels_checksum_local", "pixel_frame_blake2b128",
        "pixels_checksum_engine",
        "pixels_checksum_matches", "pixels_nonzero", "pixels_size",
        "capture_reads", "audit_sequence", "audit_gated", "actor_input_digest",
        "requested_mask", "segments", "advanced_total",
        # the slim's own fields: where the resource columns came from, which
        # units were really audited, and the gated cross-checks
        "hud_source", "hud_measured_units", "hud_crosscheck",
        "pixels_crosscheck", "pixels_checksum_engine_units", "state_sampled",
    }
    for row in summary["rows"]:
        assert required <= set(row)
        assert row["pixels_checksum_error"] is None
        assert len(row["actor_input_digest"]) == 64
        assert row["actor_hist_pushes"] == row["decision_index"]
        assert row["advanced_total"] == 2
        assert len(row["pixel_frames"]) == 2
        # The local FNV is the decision's recorded value (its last unit); the
        # engine reply only exists where the gated PIXELS was issued.  Every
        # decision also carries the per-frame C-level identity.
        assert len(row["pixels_checksum_local"]) == 16
        assert len(row["pixel_frame_blake2b128"]) == 32
        assert row["pixels_checksum_engine"] is None
        assert row["pixels_checksum_matches"] is None
        assert row["hud_source"] == "state_derived"
        assert row["state_sampled"] is True
    # Only decision 0 contains the gated unit 0, so it is the only row that can
    # carry an engine-side cross-check.
    assert summary["rows"][0]["hud_measured_units"] == [0]
    assert summary["rows"][0]["hud_crosscheck"]["lives_bombs_match"] is True
    assert summary["rows"][0]["pixels_checksum_engine_units"] == [0]
    assert summary["rows"][0]["pixels_crosscheck"]["matches"] is True
    assert all(row["hud_crosscheck"] is None for row in summary["rows"][1:])
    assert [row["logical_frame"] for row in summary["rows"]] == [0, 2, 4, 6]
    # The recorded pixel_frame is the frame of the captured bytes (EVENTS.frames
    # - 1 of the last audited unit), never the overloaded header field: on the
    # rl_v3 model the header reads 1, 2, ... while the bytes are 0, 1, ...
    assert [row["pixel_frame"] for row in summary["rows"]] == [1, 3, 5, 7]
    assert [row["pixel_frame_header"] for row in summary["rows"]] == [2, 4, 6, 8]
    assert all(row["pixel_frame_relation"] == "state_publish"
               for row in summary["rows"])
    frame_required = {
        "frame", "stage_clear", "stage_clear_edge", "stage_clear_edge_frame",
        "level_stage_clear_bit", "state_stage", "pixel_frame", "pixel_frame_ok",
        "pixel_flags", "pixel_flags_valid", "header_frame",
        "header_frame_relation", "events_frames", "pixels_checksum_local",
        "pixel_frame_blake2b128",
        "pixels_checksum_engine", "pixels_checksum_matches", "capture_reads",
        "audit_sequence", "hud_source", "hud_measured", "hud_crosscheck",
        "state_sampled", "lives_remaining", "bombs_remaining", "player_power",
        "score", "graze", "deaths",
    }
    for row in summary["frame_rows"]:
        assert frame_required <= set(row)
        assert row["pixel_frame"] == row["events_frames"] - 1
        assert row["header_frame"] == row["events_frames"]
        assert row["pixel_flags_valid"] is True
        # Every frame row carries the C-level per-frame identity, whatever the
        # FNV scope says about pixels_checksum_local.
        assert len(row["pixel_frame_blake2b128"]) == 32
        # The resource columns are STATE-derived, never the HUD snapshot.
        assert row["hud_source"] == "state_derived"
        assert row["hud_lives"] == int(row["lives_remaining"])
        assert row["hud_bombs"] == int(row["bombs_remaining"])
    assert summary["renders"] == 1 + summary["step_units"]


def test_partial_advance_of_a_decision_still_pushes_once():
    """``advanced_total > 0`` is enough: the candidate did take effect."""

    class LateHost(FakeHost):
        """The second unit of the first decision behaves like a normal frame."""

        def step(self, mask: int, frames: int) -> dict[str, str]:
            reply = super().step(mask, frames)
            if len(self.steps) == 1:
                reply["advanced"] = str(int(self.steps[-1][1]))
            return reply

    host = LateHost()
    history = ActionHistoryBuffer()
    summary = drive(host, max_controlled_frames=8, history=history)
    first = summary["rows"][0]
    assert first["advanced_total"] == 2
    assert first["history_pushed"] is True
    assert first["not_executed"] is False
    assert history.pushes == summary["history_pushes"] == 4
    # One push per decision, never one per STEP unit.
    assert summary["step_units"] == 8
    assert summary["history_pushes"] == 4


def test_stalled_decision_is_not_pushed_into_the_history():
    class DeadHost(FakeHost):
        def step(self, mask: int, frames: int) -> dict[str, str]:
            self.steps.append((int(mask), int(frames)))
            return {"advanced": "0"}

    host = DeadHost()
    history = ActionHistoryBuffer()
    summary = drive(host, max_controlled_frames=2, history=history)
    assert summary["steps"] == 1
    assert summary["rows"][0]["advanced_total"] == 0
    assert summary["rows"][0]["not_executed"] is True
    assert summary["rows"][0]["history_pushed"] is False
    assert history.pushes == 0
    assert summary["decisions_without_effect"] == 1
    assert any("NOT pushed" in warning for warning in summary["warnings"])
    assert summary["terminal"]["kind"] == "environment_exit"
    assert summary["terminal"]["reason"] == "short_advance"


def test_prime_first_frame_uses_the_flag_and_counts_its_frames():
    class BlockedHost(FakeHost):
        def __init__(self) -> None:
            super().__init__()
            self._warm = 0

        def pixels(self):
            header, rgba = super().pixels()
            if self._warm < 1:
                header = dict(header)
                header["flags"] = 0
            return header, rgba

        def step(self, mask: int, frames: int) -> dict[str, str]:
            self._warm += 1
            return super().step(mask, frames)

    host = BlockedHost()
    tap = SLICE.ActorInputTap()
    prime = SLICE.prime_first_frame(host=host, tap=tap,
                                    meter=SLICE.CostMeter())
    assert prime["warmup_frames"] == 1
    assert prime["warmup_attempts"] == 1
    assert prime["pixels_valid"] is True
    assert tap.frames_pumped == 1
    # The warmup STEP carries cand_to_mask(0, 0) (SHOOT, no direction), which is
    # the frozen v4.2.0 warmup mask, never a bare STEP 0.
    assert host.steps == [(gym_actions.cand_to_mask(0, 0), 1)]
    assert len(prime["rgba_sha256"]) == 64


# ---------------------------------------------------------------------------
# extra: the whole subcommand's record, assembled without an engine
# ---------------------------------------------------------------------------
def test_run_subcommand_assembles_the_record_without_an_engine(tmp_path,
                                                              monkeypatch):
    """Exercise the ``run_subcommand`` wiring around the loop (result.json).

    The loop itself is covered above; this covers the assembly the next real run
    will be read for, so a missing key cannot hide until the engine budget is
    spent.  ``EngineHost`` and ``preflight`` are replaced; nothing is started,
    nothing outside ``tmp_path`` is written.
    """

    data_source = tmp_path / "source-game"
    data_source.mkdir()
    audible = synth_cfg(music_mode=1, play_sounds=1, music_volume=39,
                        sfx_volume=36)
    (data_source / "th08.cfg").write_bytes(audible)
    (data_source / "th08.dat").write_bytes(b"dat")
    (data_source / "thbgm.dat").write_bytes(b"bgm")
    out = tmp_path / "run"

    class ProbeHost(FakeHost):
        """The protocol the loop needs plus ``open``/``close``, nothing else."""

        def __init__(self, **_kwargs) -> None:
            super().__init__()
            self.closed = False

        def open(self, **_kwargs):
            return {"hello": {"role": "env"}, "shm": {"active": "1"},
                    "readback": {"readback": "1"}, "reset": {"ready": "1"}}

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(SLICE, "EngineHost", ProbeHost)
    monkeypatch.setattr(SLICE, "preflight", lambda args, out_dir: {
        "engine": "/fake/th08-modern",
        "engine_sha256": "0" * 64,
        "engine_sha256_registered": SLICE.REGISTERED_ENGINE_SHA256,
        "engine_sha256_matches": True,
        "obs_spec_digest": SLICE.REGISTERED_OBS_V3_DIGEST,
        "obs_spec_digest_matches": True,
        "data_source": str(data_source),
        "data_source_cfg": SLICE.decode_cfg_file(data_source / "th08.cfg"),
        "source_cfg_registered_sha256": SLICE.REGISTERED_SOURCE_CFG_SHA256,
        "source_cfg_matches_registered": False,
        "stage": 0, "difficulty_index": 0, "shot_type": 0, "seed": 1000,
    })
    args = SimpleNamespace(
        engine="/fake/th08-modern", data_source=str(data_source),
        out=str(out), stage=0, difficulty="normal", shot_type=0, seed=1000,
        decision_frames=2, shoot_pulse_period=48, max_controlled_frames=4,
        timeout_s=60.0, no_xvfb=True, allow_partial_tail=False,
        constant_cand=12)

    exit_code = SLICE.run_subcommand(args, mode="pixel-smoke")
    assert exit_code == 0

    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["terminal_kind"] == "truncated"
    assert result["terminal_reason"] == "frame_cap"
    assert result["frames"] == 4
    assert result["decisions"] == 2
    assert result["audit"]["command_sequence"] == [
        "step", "shm", "events", "state"]
    assert "hud" in result["audit"]["command_gated"]
    assert "pixels" in result["audit"]["command_gated"]
    assert result["audit"]["stride"] == 1
    assert result["audit_stride"] == 1
    assert result["audit"]["sampled_units"] == 4
    assert result["audit"]["commands_sent"]["step"] == 4
    assert result["audit"]["commands_sent"]["hud"] == 1
    assert result["audit"]["commands_sent"]["pixels"] == 1
    assert result["audit"]["hud_source"] == "state_derived"
    assert result["audit"]["hud_measured_units"] == 1
    assert result["audit"]["pixel_frame_rule"] == SLICE.PIXEL_FRAME_RULE
    assert result["audit"]["pixels_checksum_mismatches"] == 0
    # Digest scopes (V831-SLICE-01c), recorded with the ticket quote.
    assert result["audit"]["checksum_scope"] == "per_decision"
    assert result["audit"]["frame_digest_scope"] == "per_unit_blake2b128"
    assert "每次决策" in result["audit"]["checksum_scope_rationale"]
    assert result["audit"]["pixel_frame_digest_algorithm"] == "blake2b-128"
    assert result["audit"]["fnv_units"] == result["decisions"] == 2
    assert result["audit"]["fnv_crosscheck_units"] == 1
    assert result["audit"]["blake2b_units"] == result["step_units"] == 4
    # --timeout-s is recorded verbatim; its defaults are untouched.
    assert result["timeout_s"] == 60.0
    assert SLICE.DEFAULT_TIMEOUT_S["teacher-calibration"] == 600.0
    assert SLICE.DEFAULT_TIMEOUT_S["pixel-smoke"] == 120.0
    # The last-unit half of the gated tail: one HUD and one PIXELS after the
    # loop, compared with the last captured bytes / the final STATE.
    crosscheck = result["audit"]["final_crosscheck"]
    assert crosscheck["pixels"]["matches"] is True
    assert crosscheck["pixels"]["engine"] == crosscheck["pixels"]["local"]
    assert crosscheck["hud"]["matches_lives_bombs"] is True
    assert crosscheck["hud"]["state"]["lives"] == 2
    assert len(result["first_pixels_checksum_local"]) == 16
    assert result["decisions_head"][0]["pixel_frame"] == 1
    assert result["last_pixels_checksum_kind"].startswith("engine PIXELS reply")

    rows = [json.loads(line) for line in
            (out / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    assert len(rows) == 2
    assert [row["pixel_frame"] for row in rows] == [1, 3]
    assert all(row["audit_sequence"] == [
        "step", "shm", "events", "state"] for row in rows)
    assert all(row["hud_source"] == "state_derived" for row in rows)
    assert all(row["pixels_checksum_matches"] is None for row in rows)
    assert rows[0]["pixels_crosscheck"]["matches"] is True

    # The private copy and both mute documents were still written; no engine ran.
    assert (out / "game" / "th08.cfg").is_file()
    assert (out / "mute.json").is_file()
    assert result["counts"]["started"] == 1
