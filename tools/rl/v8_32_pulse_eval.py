#!/usr/bin/env python3
"""V836-PULSE: a **short forced-action pulse** inside a checkpoint takeover window.

The question this tool answers
------------------------------
The previous round measured 18 *sustained* fixed actions from entry 64: they lived
598-1,166 frames, while the policy at the same entry lives 1,270-2,286 frames
(``.tmp_v836_rwr/run/``).  So a *constant* exploration policy is a strong
out-of-distribution, high-mortality regime -- but it is 100 % of the exploration
budget for the whole run.  The question that decides whether *safe* online
exploration is possible is therefore narrower:

    if the action is perturbed for only 50 decisions (100 frames) and control is
    then handed **back** to the policy, does the policy recover?

That is what this tool measures.  It is the same checkpoint take-over the RWR round
ran, with **one** change: inside a decision-number window every emitted candidate is
replaced by ``--pulse-cand``; outside the window the policy's own ``(cand, bomb)``
goes through untouched.

Why this is built on ``v8_32_takeover.py`` and not on ``v8_32_leftrecover_eval.py``
----------------------------------------------------------------------------------
The equivalence gate of this round is *bit-identity with an existing artifact*:
``--pulse-len 0`` re-run at seed 1000 must reproduce
``.tmp_v836_rwr/run/s1000/result.json:frames = 1270`` **and** the same decision rows
as ``.tmp_v836_rwr/logs/s1000/decisions.jsonl``.  Those artifacts were produced by
``tools/rl/v8_32_takeover.py run --tape-slots ... --entry-frame 64 --arm
checkpoint``: the entry is a *tape replay* (RESET, the 8 lead-in decisions, 24
recorded slots, one ``HASH`` per slot, the tape's ``executed_cand`` seeding the
8x18 action history, frames ``F-3..F`` rendered into the 4-frame stack).  The
left-recovery ruler (``v8_32_leftrecover_eval.py``, itself the pinned
``v400_student_eval.py`` with one hook) starts the policy at ``RESET`` and has no
tape, no ``--tape-slots``, no ``--entry-frame`` and therefore **cannot** reproduce
that trajectory -- a bit-identity gate against the RWR artifact is unreachable from
it.  So the pulse is inserted where the RWR trajectory is actually produced: the
take-over runner's policy call.  ``v8_32_takeover.py`` is imported **by path and
never modified** (it is 163 kB of pinned runner code; its sha256 is recorded in
every ``pulse.json`` and in every result row); this file only adds the wrapper, the
window CLI and the pulse telemetry -- the same shape ``v8_32_leftrecover_eval.py``
uses (a ``PulseConfig`` dataclass, a ``PulseModel`` wrapper around the actor, a
per-decision JSONL, a run summary), applied one layer down so the gate can hold.

Where the pulse sits, exactly
-----------------------------
``TakeoverConfig``/``run_takeover`` are the pinned runner's own; ``policy_loader``
is the one injection point it offers (``run_takeover(..., policy_loader=...)`` ->
``load_policy(config, loader=...)``; the sha256 gate and the obs-v3 digest gate run
**in front of** the loader and cannot be skipped).  The loader loads the real actor
with the pinned ``default_policy_loader`` and wraps it::

    class PulseModel:
        def act(self, play, hud, hist):
            cand, bomb = self.model.act(play, hud, hist)   # the policy is always run
            if in window: cand = --pulse-cand
            return cand, bomb

Consequences, in order of importance:

1. **The policy is still called on every decision**, inside the window included.  A
   pulse therefore measures "force this action here", not "skip the policy": the
   action history the policy sees afterwards is the *forced* one, exactly as it
   would be in an online-learning loop that executed a probe.
2. **No extra engine command.**  The decision number is the whole trigger, so
   unlike a position-triggered guard there is nothing to read before acting: the
   pulse adds zero protocol traffic, and a disabled pulse sends the pinned
   runner's exact command sequence.
3. **``--pulse-len 0`` is the equivalence switch.**  With the window empty the
   wrapper returns the actor's own values **without** converting, copying or
   substituting anything, so the run is the pinned runner's run (the wrapper only
   appends to its own log).

Window definition (frozen, written to ``pulse.json`` and to the pre-registration)
--------------------------------------------------------------------------------
``decision`` is the number the decision log already carries: ``frame // 2``
(``frame`` = frames elapsed before the decision), so the first hand-over decision
of entry 64 is ``decision 32`` and the window

    --pulse-start S --pulse-len L   covers   S <= decision <= S+L-1
                                              frames 2S .. 2(S+L-1)+1

is the *whole* rule: no state, no position, no RNG.  The default run of this round
is ``S=300, L=50, cand in {0, 3, 14}`` = decisions 300..349 = frames 600..699.

Telemetry (``--pulse-log``)
---------------------------
One JSONL row per hand-over decision::

    {"schema": "v836-pulse-decision-1", "seed": 1000, "call": 268,
     "decision": 300, "frame": 600, "in_window": true, "window_index": 0,
     "pulse_start": 300, "pulse_len": 50, "pulse_cand": 0,
     "cand_in": 12, "cand_out": 0, "bomb": 0, "changed": true}

``cand_in`` is what the policy chose (``null`` if it is not an integer -- recorded,
never forged), ``cand_out`` what was emitted, ``changed`` whether they differ.
``applied`` counts *window* decisions the substitution was applied to; ``changed``
counts those where the policy's own choice differed, so a window where the policy
happens to agree with ``--pulse-cand`` is visible as ``applied=50, changed<50``
instead of being silently counted as 50 substitutions.

``pulse.json`` (next to the runner's ``result.json``) carries the summary: the
window, ``applied``/``changed``, the 50 window rows, the histograms and the
alignment check against the decision log (row count and the ``decision``/``frame``
sequence must agree row for row; a misalignment is a technical failure, exit 2).

The player coordinate (``--log-player``, on by default)
-------------------------------------------------------
A decision row says *what was emitted*, never *where the player stood*, so a
counterfactual that reproduces the baseline exactly cannot be explained from the
row: "forced down for two frames changed nothing" is either a clamp at the bottom
of the play area or something else, and the row cannot tell the two apart.  This
round therefore reads the player's own coordinate **once per decision, before
that decision's action**, and writes it into the row::

    "player": {"schema": "v836-pulse-player-1", "read_phase": "before_action",
               "command": "PLAYER", "decoder": "touhou_rl.v400.gym.protocol.
               parse_player_reply", "read_index": 268, "decision": 300,
               "decision_frame": 600, "frame": 600, "frame_matches": true,
               "available": true, "position": [192.4, 395.3], "focus": 0,
               "invuln_state": 1, "player_state": 3, "is_youkai": 1, "hp": 2.0,
               "lives": 2, "bombs": 3.0, "power": 0.0,
               "speeds": {"normal_axial": 4.0, "normal_diagonal": 2.828427,
                          "focused_axial": 2.0, "focused_diagonal": 1.414214},
               "raw": {"frames": "600", "px": "%08x", "py": "%08x"}}

Three properties make the number usable:

1. **The decoder is the factory's.**  The reply is decoded by
   ``touhou_rl.v400.gym.protocol.parse_player_reply`` -- the same decoder
   ``tools/rl/v8_32_relabel.py`` and ``TeacherEngine.player`` use -- and the raw
   ``px=``/``py=`` tokens are kept next to it.  Nothing splits the string here.
2. **The frame must be the decision's.**  A ``PLAYER`` reply carries the engine's
   own ``frames=`` counter; a reply whose frame is not the decision's frame is
   refused (``PlayerReadError``, exit 2) because that coordinate belongs to
   another state.  This is fail-closed **always**, not only under
   ``--require-player``.
3. **The read happens before the action.**  ``PulseModel.act`` sends ``PLAYER``
   and only then calls the actor; the two ``STEP``s of that decision happen after
   ``act`` returns, in the pinned runner.  ``read_phase="before_action"`` and
   ``read_index`` (the pulse wrapper's call counter) travel with the number.

``--log-player`` is on by default and ``--no-log-player`` turns it off: the
wrapper then sends no ``PLAYER`` command at all and the decision log is the
pinned runner's byte for byte.  ``--require-player`` additionally refuses a run
in which no coordinate could be read (it needs ``--arm checkpoint``: only that
arm calls the policy this wrapper sits in front of).  ``PLAYER`` is read-only
(``HandlePlayer`` advances nothing), which is what makes the trajectory gate
(1,270 frames, the same 603 rows) reachable with the block added.

Exit codes are the pinned runner's (0 the window ran, 1 the entry/hash failed, 2 a
technical failure, 4 refused before the start, 5 the mute bytes changed), plus 2
when the pulse log does not align with the decision log.

Safety
------
No audio: the wrapper starts no engine itself; ``run_takeover`` refuses a
``<data-dir>/th08.cfg`` whose four audio bytes are not all 0 and re-reads them after
the exit (``AGENTS.md`` section 9).  ``SDL_AUDIODRIVER=dummy`` is set by the
launcher (and by ``HarnessClient``).  Nothing outside ``--out``/``--pulse-log`` is
written.

Run inside WSL::

    xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \\
      tools/rl/v8_32_pulse_eval.py run \\
      --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \\
      --expected-sha 691bfd77... --data-dir .tmp_v836_pulse/game-muted \\
      --tape-slots .tmp_v836_rwr/tape/s1000/slots.jsonl \\
      --seed 1000 --entry-frame 64 --arm checkpoint \\
      --checkpoint .tmp_v832_s2m/ckpt_s1.pt --checkpoint-sha256 1414822f... \\
      --cap-frames 2500 --out .tmp_v836_pulse/p0_s1000 \\
      --decision-log .tmp_v836_pulse/p0_s1000/decisions.jsonl \\
      --pulse-start 300 --pulse-len 50 --pulse-cand 0 \\
      --pulse-log .tmp_v836_pulse/p0_s1000/pulse.jsonl
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONTROLLER = "tools/rl/v8_32_pulse_eval.py"
#: The pinned runner this tool drives (imported by path, never modified).
PINNED_RUNNER = "tools/rl/v8_32_takeover.py"
PULSE_SCHEMA = "v836-pulse-decision-1"
PULSE_SUMMARY_SCHEMA = "v836-pulse-run-1"
#: ``cand = focus * 9 + move``; ``move 0`` = idle, ``move 3`` = E, ``move 5`` = S.
#: Read from the frozen action table, never re-typed here.
from touhou_rl.v400.gym import actions as gym_actions  # noqa: E402

MOVES_PER_CAND = int(gym_actions.MOVES_PER_CAND)
N_CAND = int(gym_actions.N_CAND)
#: The window this round pre-registered: decisions 300..349 (frames 600..699).
DEFAULT_PULSE_START = 300
DEFAULT_PULSE_LEN = 0  # 0 == disabled == the equivalence switch
DEFAULT_PULSE_CAND = 0
EXIT_TECHNICAL = 2
#: The player coordinate read (``HandlePlayer``; read-only, advances nothing).
PLAYER_SCHEMA = "v836-pulse-player-1"
PLAYER_COMMAND = "PLAYER"
PLAYER_DECODER = "touhou_rl.v400.gym.protocol.parse_player_reply"
PLAYER_READ_PHASE = "before_action"
PLAYER_PRECEDES = ("act", "step")
#: The names a ``player`` block must carry (the round's contract, checked on disk).
PLAYER_REQUIRED_FIELDS = ("position", "focus", "invuln_state", "player_state",
                          "frame", "frame_matches", "read_phase")


# ---------------------------------------------------------------------------
# The pinned runner, imported by path
# ---------------------------------------------------------------------------
def pinned_path(path: str | Path | None = None) -> Path:
    """The pinned take-over runner this tool extends (never written here)."""

    if path is not None:
        return Path(path)
    return Path(__file__).resolve().parent / PINNED_RUNNER.split("/")[-1]


def load_pinned(path: str | Path | None = None) -> Any:
    """Import ``v8_32_takeover.py`` as a module, once.

    Importing it defines functions and dataclasses only -- the engine is started
    exclusively by its ``run`` subcommand -- so this cannot start a game.
    """

    target = pinned_path(path)
    name = "v8_32_takeover_pinned"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, target)
    if spec is None or spec.loader is None:  # pragma: no cover - broken install
        raise RuntimeError(f"cannot import the pinned runner from {target}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:  # pragma: no cover - broken install
        sys.modules.pop(name, None)
        raise
    return module


PINNED = load_pinned()


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class PulseConfig:
    """The three numbers the window is made of, validated once."""

    start: int
    length: int
    cand: int

    @property
    def enabled(self) -> bool:
        return int(self.length) > 0

    @property
    def end(self) -> int:
        """First decision **outside** the window (``start + length``)."""

        return int(self.start) + int(self.length)

    @property
    def first_frame(self) -> int:
        return int(self.start) * int(PINNED.DECISION_FRAMES)

    @property
    def last_frame(self) -> int:
        """First frame of the window's last decision."""

        return (int(self.end) - 1) * int(PINNED.DECISION_FRAMES)

    @property
    def frames(self) -> int:
        return int(self.length) * int(PINNED.DECISION_FRAMES)

    @property
    def move(self) -> int:
        return int(self.cand) % MOVES_PER_CAND

    @property
    def focus(self) -> int:
        return int(self.cand) // MOVES_PER_CAND

    def covers(self, decision: int) -> bool:
        """The whole rule: ``start <= decision < start + length``."""

        if not self.enabled:
            return False
        return int(self.start) <= int(decision) < self.end

    def as_dict(self) -> dict:
        return {
            "start": int(self.start),
            "length": int(self.length),
            "end": int(self.end),
            "cand": int(self.cand),
            "move": int(self.move),
            "focus": int(self.focus),
            "enabled": bool(self.enabled),
            "decision_definition": (
                "decision = frame // 2: the number the decision log carries (frames "
                "elapsed before the decision); the first hand-over decision of entry "
                "64 is decision 32"),
            "window": ("%d <= decision <= %d" % (self.start, self.end - 1)
                       if self.enabled else "empty (no decision is forced)"),
            "window_frames": ([int(self.first_frame), int(self.last_frame) + 1]
                              if self.enabled else None),
            "frames": int(self.frames),
            "disable_length": 0,
        }


def resolve_pulse(*, start: Any, length: Any, cand: Any) -> PulseConfig:
    """Validate the window before any file or engine exists (fail closed)."""

    try:
        first = int(start)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--pulse-start {start!r} is not an integer") from exc
    try:
        span = int(length)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--pulse-len {length!r} is not an integer") from exc
    try:
        forced = int(cand)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--pulse-cand {cand!r} is not an integer") from exc
    if first < 0:
        raise ValueError(f"--pulse-start {first} is negative")
    if span < 0:
        raise ValueError(f"--pulse-len {span} is negative (0 disables the pulse)")
    if not 0 <= forced < N_CAND:
        raise ValueError(f"--pulse-cand {forced} is outside 0..{N_CAND - 1}")
    return PulseConfig(start=first, length=span, cand=forced)


# ---------------------------------------------------------------------------
# The wrapper: the one change this round makes
# ---------------------------------------------------------------------------
def _as_int(value: Any) -> int | None:
    """``int(value)`` or ``None`` -- never a fabricated number."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The player coordinate: one ``PLAYER`` read per decision, before the action
# ---------------------------------------------------------------------------
class PlayerReadError(RuntimeError):
    """A ``PLAYER`` read that cannot be used: no coordinate, or another frame."""


def _round_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), 6)


def player_transport(client: Any) -> Any:
    """The one callable that can send ``PLAYER`` to this client.

    ``HarnessClient`` answers on ``request`` (the frozen wire method); the
    ``ask`` name is accepted too because ``tools/rl/v8_32_relabel.py``'s teacher
    engine exposes that one.  An object with neither is refused instead of being
    asked for a coordinate it cannot produce.
    """

    for name in ("request", "ask"):
        method = getattr(client, name, None)
        if callable(method):
            return method
    raise PlayerReadError(
        f"the engine client {type(client).__name__} exposes neither request() nor "
        f"ask(), so no {PLAYER_COMMAND} coordinate can be read from it")


def _reply_token(reply: str, name: str) -> str | None:
    """The raw ``name=<token>`` of one reply -- **provenance only**.

    The value written to a row is the one
    :func:`touhou_rl.v400.gym.protocol.parse_player_reply` decoded; this helper
    only keeps the wire token next to it (``px=%08x`` is the float32 bits).
    """

    for token in str(reply).split():
        if token.startswith(f"{name}="):
            return token[len(name) + 1:]
    return None


def read_player(client: Any) -> tuple[Any, str]:
    """Send one ``PLAYER`` and decode it with the **shipped** decoder.

    ``request``/``ask`` + ``parse_player_reply`` is the pair
    ``tools/rl/v8_32_relabel.py::default_player_reader`` and
    ``TeacherEngine.player`` use, so the names in the block are the wire names.
    ``PLAYER`` is a read-only command: it advances no frame and changes no state.
    """

    send = player_transport(client)
    reply = str(send(PLAYER_COMMAND))
    from touhou_rl.v400.gym import protocol  # noqa: PLC0415

    return protocol.parse_player_reply(reply), reply


def player_block(snapshot: Any, *, decision: int, frame: int, read_index: int,
                 reply: str = "") -> dict:
    """The per-row ``player`` block, built from one decoded ``PLAYER`` reply."""

    px, py = (float(snapshot.position[0]), float(snapshot.position[1]))
    return {
        "schema": PLAYER_SCHEMA,
        "read_phase": PLAYER_READ_PHASE,
        "precedes": list(PLAYER_PRECEDES),
        "command": PLAYER_COMMAND,
        "decoder": PLAYER_DECODER,
        "read_index": int(read_index),
        "decision": int(decision),
        "decision_frame": int(frame),
        "frame": int(snapshot.frame),
        "frame_matches": int(snapshot.frame) == int(frame),
        "available": True,
        "position": [round(px, 6), round(py, 6)],
        "focus": int(snapshot.focus),
        "invuln_state": int(snapshot.invuln_state),
        "player_state": int(snapshot.player_state),
        "is_youkai": int(snapshot.is_youkai),
        "hp": float(snapshot.hp),
        "lives": int(snapshot.lives),
        "bombs": float(snapshot.bombs),
        "power": float(snapshot.power),
        "speeds": {
            "normal_axial": _round_or_none(snapshot.speed_normal_axial),
            "normal_diagonal": _round_or_none(snapshot.speed_normal_diagonal),
            "focused_axial": _round_or_none(snapshot.speed_focused_axial),
            "focused_diagonal": _round_or_none(snapshot.speed_focused_diagonal),
        },
        "raw": {name: _reply_token(reply, name)
                for name in ("frames", "px", "py")},
    }


def unavailable_player_block(*, decision: int, frame: int, read_index: int,
                             reason: str, kind: str = "read_failed") -> dict:
    """A block for a read that did not happen: ``available=False``, never a 0."""

    return {
        "schema": PLAYER_SCHEMA,
        "read_phase": PLAYER_READ_PHASE,
        "precedes": list(PLAYER_PRECEDES),
        "command": PLAYER_COMMAND,
        "decoder": PLAYER_DECODER,
        "read_index": int(read_index),
        "decision": int(decision),
        "decision_frame": int(frame),
        "frame": None,
        "frame_matches": None,
        "available": False,
        "position": None,
        "focus": None,
        "invuln_state": None,
        "player_state": None,
        "is_youkai": None,
        "hp": None,
        "lives": None,
        "bombs": None,
        "power": None,
        "speeds": None,
        "raw": None,
        "kind": str(kind),
        "reason": str(reason),
    }


class PlayerProbe:
    """One ``PLAYER`` read per decision, kept until that decision's row is written.

    The block is stored under the decision's own frame, so the row hook can only
    attach the coordinate of **that** decision: a block read for another frame is
    never found, and a reply whose ``frames=`` disagrees with the decision raises
    :class:`PlayerReadError` immediately (fail-closed, exit 2).
    """

    def __init__(self, *, client_provider: Any, require: bool = False) -> None:
        self.client_provider = client_provider
        self.require = bool(require)
        self.blocks: dict[int, dict] = {}
        self.taken: set[int] = set()
        self.reads = 0
        self.failures: list[dict] = []
        self.mismatches: list[dict] = []
        self.last_reply: str | None = None

    # -- one read --------------------------------------------------------
    def read(self, *, decision: int, frame: int, read_index: int) -> dict:
        try:
            client = self.client_provider()
            if client is None:
                raise PlayerReadError(
                    "the engine client is not available to the pulse wrapper")
            snapshot, reply = read_player(client)
        except Exception as exc:  # noqa: BLE001 - a refused read is recorded
            block = unavailable_player_block(
                decision=decision, frame=frame, read_index=read_index,
                kind="read_failed", reason=f"{type(exc).__name__}: {exc}")
            self.failures.append({"decision": int(decision), "frame": int(frame),
                                  "reason": block["reason"]})
        else:
            self.last_reply = reply
            block = player_block(snapshot, decision=decision, frame=frame,
                                 read_index=read_index, reply=reply)
            self.reads += 1
            if block["frame_matches"] is False:
                self.mismatches.append({
                    "decision": int(decision), "frame": int(frame),
                    "reply_frame": block["frame"]})
        self.blocks[int(frame)] = block
        return block

    # -- the gate --------------------------------------------------------
    def gate(self, block: Mapping[str, Any], *, where: str) -> None:
        """Refuse a coordinate from another frame **always**; refuse a missing one
        only under ``--require-player``.  Both are non-zero exits."""

        if block.get("frame_matches") is False:
            raise PlayerReadError(
                f"{where}: the {PLAYER_COMMAND} reply reports frame "
                f"{block.get('frame')}, the decision sits on frame "
                f"{block.get('decision_frame')}; the coordinate belongs to another "
                "state and a position-conditioned reading of this row would be wrong")
        if self.require and not block.get("available"):
            raise PlayerReadError(
                f"{where}: --require-player is set but the {PLAYER_COMMAND} "
                f"coordinate could not be read ({block.get('reason')!r}); a row "
                "without the player's position does not answer this round's question")

    # -- consume ---------------------------------------------------------
    def take(self, frame: int) -> dict | None:
        """The block read before this decision's action, once."""

        block = self.blocks.get(int(frame))
        if block is None:
            return None
        self.taken.add(int(frame))
        return block

    # -- report ----------------------------------------------------------
    def summary(self) -> dict:
        return {
            "schema": PLAYER_SCHEMA,
            "enabled": True,
            "require": bool(self.require),
            "command": PLAYER_COMMAND,
            "decoder": PLAYER_DECODER,
            "read_phase": PLAYER_READ_PHASE,
            "precedes": list(PLAYER_PRECEDES),
            "decisions_read": int(self.reads),
            "blocks": len(self.blocks),
            "blocks_attached": len(self.taken),
            "blocks_unused": sorted(set(self.blocks) - self.taken),
            "frame_mismatches": [dict(row) for row in self.mismatches],
            "failures": [dict(row) for row in self.failures],
        }


class PlayerRowHook:
    """Attach the block read before the action to the row the runner writes.

    ``run_takeover`` builds every decision row with the module-level
    :func:`v8_32_takeover.decision_row` (keyword-only arguments), and the row is
    written **after** that decision's two frames.  Wrapping that one function is
    therefore the only seam needed: the row keeps every field the pinned runner
    produced and gains exactly one key, ``player``.  Nothing is written twice and
    no file is rewritten after the run.  With ``--no-log-player`` the hook is
    never installed, so the module is untouched and the log is the pinned
    runner's byte for byte.
    """

    def __init__(self, module: Any, probe: PlayerProbe) -> None:
        self.module = module
        self.probe = probe
        self._original = module.decision_row
        self._installed = False
        self.rows = 0
        self.attached = 0
        self.missing: list[int] = []

    def install(self) -> "PlayerRowHook":
        if self._installed:  # pragma: no cover - install is called once
            return self
        original = self._original
        hook = self

        def decision_row_with_player(*args: Any, **kwargs: Any) -> dict:
            row = original(*args, **kwargs)
            frame = kwargs.get("frame", args[2] if len(args) > 2 else None)
            block = None if frame is None else hook.probe.take(int(frame))
            hook.rows += 1
            if block is None:
                hook.missing.append(_as_int(kwargs.get("decision")) or -1)
                if hook.probe.require:
                    raise PlayerReadError(
                        f"decision {kwargs.get('decision')!r} (frame {frame!r}): no "
                        f"{PLAYER_COMMAND} coordinate was read before this decision's "
                        "action, and --require-player forbids a row without one")
            else:
                row["player"] = block
                hook.attached += 1
            return row

        decision_row_with_player.__name__ = "decision_row_with_player"
        decision_row_with_player.__doc__ = (
            "The pinned runner's decision_row plus the pre-action PLAYER block.")
        self.module.decision_row = decision_row_with_player
        self._installed = True
        return self

    def remove(self) -> None:
        if self._installed:
            self.module.decision_row = self._original
            self._installed = False

    def summary(self) -> dict:
        return {
            "rows_seen": int(self.rows),
            "rows_with_player": int(self.attached),
            "rows_missing_player": len(self.missing),
            "missing_decisions": sorted(set(self.missing))[:20],
        }


def player_audit(*, probe: PlayerProbe | None, hook: PlayerRowHook | None,
                 log_rows: Sequence[Mapping[str, Any]], require: bool) -> dict:
    """What the decision log on disk actually carries (recomputed, not claimed)."""

    blocks = [row.get("player") for row in log_rows]
    present = [block for block in blocks if isinstance(block, Mapping)]
    missing = [int(row.get("decision", -1)) for row, block in zip(log_rows, blocks)
               if not isinstance(block, Mapping)]
    bad_phase = [int(row.get("decision", -1))
                 for row, block in zip(log_rows, blocks)
                 if isinstance(block, Mapping)
                 and block.get("read_phase") != PLAYER_READ_PHASE]
    mismatched = sorted(set(
        [int(row.get("decision", -1))
         for row, block in zip(log_rows, blocks)
         if isinstance(block, Mapping) and block.get("frame_matches") is False]
        + [int(row.get("decision", -1))
           for row in (probe.mismatches if probe is not None else [])]))
    frame_mismatch = [int(row.get("decision", -1))
                      for row, block in zip(log_rows, blocks)
                      if isinstance(block, Mapping)
                      and isinstance(block.get("frame"), int)
                      and int(block["frame"]) != int(row.get("frame", -1))]
    incomplete = [int(row.get("decision", -1))
                  for row, block in zip(log_rows, blocks)
                  if isinstance(block, Mapping)
                  and any(name not in block for name in PLAYER_REQUIRED_FIELDS)]
    unavailable = [int(row.get("decision", -1))
                   for row, block in zip(log_rows, blocks)
                   if isinstance(block, Mapping) and block.get("available") is not True]
    ok = not (mismatched or frame_mismatch or incomplete or bad_phase)
    if require:
        ok = ok and not missing and not unavailable and bool(present)
    report = {
        "enabled": probe is not None,
        "require": bool(require),
        "command": PLAYER_COMMAND,
        "decoder": PLAYER_DECODER,
        "read_phase": PLAYER_READ_PHASE,
        "decision_rows": len(log_rows),
        "rows_with_player": len(present),
        "rows_missing_player": len(missing),
        "missing_decisions": missing[:20],
        "rows_unavailable": len(unavailable),
        "frame_mismatches": mismatched,
        "frame_disagreements": frame_mismatch,
        "phase_violations": bad_phase,
        "incomplete_blocks": incomplete,
        "reads": (probe.reads if probe is not None else 0),
        "blocks_attached": (hook.attached if hook is not None else 0),
        "ok": bool(ok),
    }
    if probe is not None:
        report["probe"] = probe.summary()
    if hook is not None:
        report["hook"] = hook.summary()
    return report


def player_failure_reasons(report: Mapping[str, Any]) -> list[str]:
    """The human-readable reasons a player audit is not ``ok`` (empty = fine)."""

    reasons: list[str] = []
    if report.get("frame_mismatches"):
        reasons.append(f"{PLAYER_COMMAND} replies on another frame for decisions "
                       f"{report['frame_mismatches'][:5]}")
    if report.get("frame_disagreements"):
        reasons.append(f"player.frame is not the row's frame for decisions "
                       f"{report['frame_disagreements'][:5]}")
    if report.get("phase_violations"):
        reasons.append("player.read_phase is not 'before_action' for decisions "
                       f"{report['phase_violations'][:5]}")
    if report.get("incomplete_blocks"):
        reasons.append("a player block is missing a required field for decisions "
                       f"{report['incomplete_blocks'][:5]}")
    if report.get("require"):
        if not report.get("rows_with_player"):
            reasons.append(f"--require-player is set but no decision row carries a "
                           f"{PLAYER_COMMAND} coordinate "
                           f"(reads={report.get('reads')})")
        if report.get("rows_missing_player"):
            reasons.append(f"--require-player is set but {report['rows_missing_player']} "
                           f"of {report.get('decision_rows')} rows carry no coordinate "
                           f"(decisions {report.get('missing_decisions')})")
        if report.get("rows_unavailable"):
            reasons.append(f"--require-player is set but {report['rows_unavailable']} "
                           "rows record an unavailable coordinate")
    return reasons


class PulseModel:
    """Wrap the obs-v3 actor so only the emitted candidate may be replaced.

    ``act`` is called exactly as the pinned runner calls the actor, so the policy's
    inputs (play stack, HUD strip, its own 8x18 action history) are untouched.  The
    policy is always run: inside the window its ``bomb`` is kept and only ``cand``
    is replaced, so the probe is "force this action at this decision", not "skip the
    policy".  With ``--pulse-len 0`` nothing is converted or replaced.

    ``probe`` (``--log-player``) is the one addition: ``act`` sends ``PLAYER`` and
    decodes the reply **before** it calls the actor, so the coordinate is the state
    at this decision's frame and both of the decision's ``STEP``s happen after it.
    A reply on another frame raises :class:`PlayerReadError` -- the pinned runner
    then stops the window instead of writing a coordinate that belongs elsewhere.

    ``decision`` is ``entry_frame // 2 + call``: the take-over hands over at
    ``entry_frame``, so call ``0`` is the decision the engine is at when control is
    handed over, which is exactly the first ``decision`` the decision log carries.
    The alignment with the decision log is re-checked from raw bytes after the run
    (``pulse.py:align``), so a wrong base cannot masquerade as a pulse.
    """

    def __init__(self, model: Any, *, config: PulseConfig, decision_base: int,
                 seed: int, log: Any = None, probe: PlayerProbe | None = None
                 ) -> None:
        self.model = model
        self.config = config
        self.decision_base = int(decision_base)
        self.seed = int(seed)
        self.log = log
        self.probe = probe
        self.calls = 0
        self.applied = 0
        self.changed = 0
        self.cand_kept = 0
        self.cand_in_hist: dict[str, int] = {}
        self.cand_out_hist: dict[str, int] = {}
        self.window_rows: list[dict] = []
        self.first_decision: int | None = None
        self.last_decision: int | None = None
        self.rows: list[dict] = []

    # -- act -------------------------------------------------------------
    def act(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        call = self.calls
        decision = self.decision_base + call
        frame = decision * int(PINNED.DECISION_FRAMES)
        if self.probe is not None:
            # The coordinate, read on the authoritative process **before** this
            # decision's action: the state is the one the observation above was
            # rendered from, and neither of the two STEPs has happened yet.
            block = self.probe.read(decision=decision, frame=frame,
                                    read_index=call)
            self.probe.gate(block, where=f"decision {decision} (frame {frame})")
        raw_cand, raw_bomb = self.model.act(*args, **kwargs)
        cand = _as_int(raw_cand)
        bomb = _as_int(raw_bomb)
        self.calls += 1
        if self.first_decision is None:
            self.first_decision = decision
        self.last_decision = decision
        if not self.config.enabled:
            # Equivalence mode: no conversion, no substitution, no copy -- the
            # actor's own values go back, exactly as the pinned runner sent them.
            row = {
                "schema": PULSE_SCHEMA, "seed": self.seed, "call": call,
                "decision": decision, "frame": frame, "in_window": False,
                "window_index": None, "pulse_start": int(self.config.start),
                "pulse_len": int(self.config.length),
                "pulse_cand": int(self.config.cand),
                "cand_in": cand, "cand_out": cand, "bomb": bomb,
                "changed": False,
            }
            self.rows.append(row)
            if self.log is not None:
                self.log.write(row)
            return raw_cand, raw_bomb

        in_window = self.config.covers(decision)
        out = int(self.config.cand) if in_window else cand
        if cand is not None:
            self.cand_in_hist[str(cand)] = self.cand_in_hist.get(str(cand), 0) + 1
        if out is not None:
            self.cand_out_hist[str(out)] = self.cand_out_hist.get(str(out), 0) + 1
        changed = bool(in_window and cand is not None and int(cand) != int(out))
        if in_window:
            self.applied += 1
            if changed:
                self.changed += 1
            else:
                self.cand_kept += 1
            entry = {"decision": decision, "frame": frame,
                     "window_index": decision - int(self.config.start),
                     "cand_in": cand, "cand_out": out, "changed": changed}
            self.window_rows.append(entry)
        row = {
            "schema": PULSE_SCHEMA, "seed": self.seed, "call": call,
            "decision": decision, "frame": frame, "in_window": bool(in_window),
            "window_index": (decision - int(self.config.start)
                             if in_window else None),
            "pulse_start": int(self.config.start),
            "pulse_len": int(self.config.length),
            "pulse_cand": int(self.config.cand),
            "cand_in": cand, "cand_out": out, "bomb": bomb,
            "changed": changed,
        }
        self.rows.append(row)
        if self.log is not None:
            self.log.write(row)
        return out, raw_bomb

    # -- report ----------------------------------------------------------
    def summary(self) -> dict:
        return {
            "schema": PULSE_SUMMARY_SCHEMA,
            "controller": CONTROLLER,
            "pinned_runner": str(pinned_path()),
            "pinned_runner_sha256": PINNED.sha256_file(pinned_path()),
            "config": self.config.as_dict(),
            "decision_base": int(self.decision_base),
            "seed": int(self.seed),
            "calls": int(self.calls),
            "first_decision": self.first_decision,
            "last_decision": self.last_decision,
            "in_window": int(self.applied),
            "applied": int(self.applied),
            "changed": int(self.changed),
            "cand_kept": int(self.cand_kept),
            "cand_in_hist": dict(sorted(self.cand_in_hist.items(),
                                        key=lambda kv: int(kv[0]))),
            "cand_out_hist": dict(sorted(self.cand_out_hist.items(),
                                         key=lambda kv: int(kv[0]))),
            "window_rows": [dict(row) for row in self.window_rows],
            "model_calls_forwarded": int(self.calls),
        }


# ---------------------------------------------------------------------------
# Alignment: the pulse log must describe the same decisions the decision log does
# ---------------------------------------------------------------------------
def decision_log_rows(path: str | Path | None) -> list[dict]:
    """The data rows of a ``--decision-log`` file (the header is not one)."""

    rows: list[dict] = []
    if not path:
        return rows
    target = Path(path)
    if not target.is_file():
        return rows
    for line in target.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        doc = json.loads(text)
        if doc.get("schema") == PINNED.SCHEMA_LOG_ROW:
            rows.append(doc)
    return rows


def align(*, rows: Sequence[Mapping[str, Any]], decisions: int | None,
          log_rows: Sequence[Mapping[str, Any]]) -> dict:
    """Compare the pulse log with the decision log, row for row.

    ``ok`` is ``None`` when there is no decision log to compare (the row count of
    the pulse log must still equal ``result.json:decisions`` when that is known).
    A mismatch is never repaired: the run is reported as technically failed.
    """

    problems: list[str] = []
    if decisions is not None and int(decisions) != len(rows):
        problems.append(f"pulse rows {len(rows)} != result decisions {decisions}")
    if log_rows:
        if len(log_rows) != len(rows):
            problems.append(f"pulse rows {len(rows)} != decision rows {len(log_rows)}")
        for index, (left, right) in enumerate(zip(rows, log_rows)):
            if int(left["decision"]) != int(right["decision"]):
                problems.append(
                    f"row {index}: pulse decision {left['decision']} != decision-log "
                    f"decision {right['decision']}")
                break
            if int(left["frame"]) != int(right["frame"]):
                problems.append(
                    f"row {index}: pulse frame {left['frame']} != decision-log frame "
                    f"{right['frame']}")
                break
        matched = min(len(rows), len(log_rows))
    else:
        matched = 0
    return {
        "ok": (not problems) if (decisions is not None or log_rows) else None,
        "decision_rows": len(log_rows),
        "pulse_rows": len(rows),
        "matched_rows": int(matched),
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The pinned runner's own ``run`` subparser (so the CLI cannot drift)."""

    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, Mapping) and "run" in choices:
            return choices["run"]
    raise RuntimeError(  # pragma: no cover - argparse layout changed
        f"{PINNED_RUNNER} has no `run` subparser")


def build_parser() -> argparse.ArgumentParser:
    parser = PINNED.build_parser()
    parser.prog = "v8_32_pulse_eval.py"
    parser.description = (
        "V836-PULSE: the pinned take-over runner (RESET, tape replay to "
        "--entry-frame, hash-verified entry, obs-v3 checkpoint arm) with a forced "
        "--pulse-cand for the decisions of one window.  --pulse-len 0 is the "
        "equivalence switch: the run is then the pinned runner's run.")
    run = run_parser(parser)
    run.add_argument("--pulse-start", dest="pulse_start", type=int,
                     default=DEFAULT_PULSE_START,
                     help="first decision of the window, on the decision log's own "
                          "numbering (decision = frame // 2; the entry-64 hand-over "
                          "starts at decision 32).  Default %d."
                          % DEFAULT_PULSE_START)
    run.add_argument("--pulse-len", dest="pulse_len", type=int,
                     default=DEFAULT_PULSE_LEN,
                     help="how many decisions the window covers (0 disables the "
                          "pulse and is the equivalence switch).  Default %d."
                          % DEFAULT_PULSE_LEN)
    run.add_argument("--pulse-cand", dest="pulse_cand", type=int,
                     default=DEFAULT_PULSE_CAND,
                     help="the candidate emitted for every decision of the window "
                          "(cand = focus*9 + move, 0..17).  Default %d."
                          % DEFAULT_PULSE_CAND)
    run.add_argument("--pulse-log", dest="pulse_log", default=None,
                     help="one JSONL row per hand-over decision (schema %s); "
                          "mandatory while the window is not empty" % PULSE_SCHEMA)
    run.add_argument("--log-player", dest="log_player",
                     action=argparse.BooleanOptionalAction, default=True,
                     help="read the player's own coordinate once per decision "
                          "(PLAYER, before that decision's action) and write it into "
                          "the decision row as a `player` block.  On by default; "
                          "--no-log-player sends no PLAYER command at all and leaves "
                          "the decision log byte for byte the pinned runner's.")
    run.add_argument("--require-player", dest="require_player",
                     action="store_true", default=False,
                     help="refuse a run whose decision rows would carry no player "
                          "coordinate (needs --log-player and --arm checkpoint)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    PINNED.maybe_reexec_under_xvfb(enabled=not bool(args.no_xvfb))
    args.argv = list(sys.argv if argv is None else ["v8_32_pulse_eval.py", *argv])

    try:
        pulse = resolve_pulse(start=args.pulse_start, length=args.pulse_len,
                              cand=args.pulse_cand)
    except ValueError as exc:
        parser.error(str(exc))
    config = PINNED.TakeoverConfig.from_args(args)
    log_player = bool(getattr(args, "log_player", True))
    require_player = bool(getattr(args, "require_player", False))
    if require_player and not log_player:
        parser.error("--require-player contradicts --no-log-player: a run that reads "
                     "no coordinate cannot require one")
    if require_player and str(config.arm) != "checkpoint":
        parser.error("--require-player needs --arm checkpoint: only that arm calls the "
                     "policy this wrapper sits in front of, so only that arm reads a "
                     f"{PLAYER_COMMAND} coordinate")
    if pulse.enabled:
        if str(config.arm) != "checkpoint":
            parser.error("--pulse-len > 0 needs --arm checkpoint: the tape and "
                         "constant arms never call the policy this wrapper replaces")
        base = int(config.entry_frame) // int(PINNED.DECISION_FRAMES)
        if int(pulse.start) < base:
            parser.error(
                f"--pulse-start {pulse.start} is before the first hand-over decision "
                f"of entry {config.entry_frame} (decision {base}); no decision of "
                "the pulse would ever be reached")
        if not str(args.pulse_log or "").strip():
            parser.error("--pulse-log is required while the window is not empty: a "
                         "run whose pulse telemetry is missing is not a measured run")

    pulse_log = None
    if str(args.pulse_log or "").strip():
        target = Path(args.pulse_log)
        if target.exists() and target.stat().st_size > 0:
            parser.error(f"--pulse-log {target.as_posix()} already holds rows; a "
                         "window is never appended to a previous one")
        try:
            pulse_log = PINNED.DecisionLog(target)
        except PINNED.DecisionLogError as exc:
            parser.error(str(exc))

    holder: dict = {}
    base = int(config.entry_frame) // int(PINNED.DECISION_FRAMES)
    probe = (PlayerProbe(client_provider=lambda: holder.get("client"),
                         require=require_player) if log_player else None)
    hook: PlayerRowHook | None = None

    def client_factory(cfg: Any) -> Any:
        """The pinned client factory, keeping the object the probe reads from.

        ``run_takeover`` creates the client after the policy loader has run, so the
        wrapper cannot hold the client directly; this seam is the one the runner
        offers (``client_factory=``) and nothing about the protocol changes.
        """

        client = PINNED.default_client_factory(cfg)
        holder["client"] = client
        return client

    def pulse_loader(cfg: Any) -> tuple[Any, Mapping[str, Any]]:
        """The pinned loader, plus the window wrapper (the sha/spec gates ran)."""

        model, payload = PINNED.default_policy_loader(cfg)
        wrapped = PulseModel(model, config=pulse,
                             decision_base=int(cfg.entry_frame)
                             // int(PINNED.DECISION_FRAMES),
                             seed=int(cfg.seed), log=pulse_log, probe=probe)
        holder["model"] = wrapped
        return wrapped, payload

    result: dict = {}
    align_report: dict = {"ok": None, "problems": ["the run did not start"]}
    try:
        if probe is not None:
            hook = PlayerRowHook(PINNED, probe).install()
        result = PINNED.run_takeover(
            config, policy_loader=pulse_loader,
            client_factory=(client_factory if probe is not None else None))
    except OSError as exc:
        print(f"run: cannot write under --out {config.out.as_posix()}: {exc}")
        return EXIT_TECHNICAL
    except Exception as exc:  # noqa: BLE001 - a technical failure is evidence
        print("run: technical failure before the runner produced a result")
        traceback.print_exc()
        return EXIT_TECHNICAL
    finally:
        if hook is not None:
            hook.remove()
            holder["hook"] = hook
        if pulse_log is not None:
            try:
                pulse_log.close()
            except PINNED.DecisionLogError as exc:  # pragma: no cover - disk error
                print(f"pulse log close failed: {exc}")

    wrapped: PulseModel | None = holder.get("model")
    log_rows = decision_log_rows(config.decision_log)
    align_report = align(rows=(wrapped.rows if wrapped is not None else []),
                         decisions=(result.get("decisions")
                                    if wrapped is not None else None),
                         log_rows=log_rows)
    summary = (wrapped.summary() if wrapped is not None else {
        "schema": PULSE_SUMMARY_SCHEMA, "controller": CONTROLLER,
        "pinned_runner": str(pinned_path()),
        "pinned_runner_sha256": PINNED.sha256_file(pinned_path()),
        "config": pulse.as_dict(), "decision_base": base,
        "seed": int(config.seed), "calls": 0, "first_decision": None,
        "last_decision": None, "in_window": 0, "applied": 0, "changed": 0,
        "cand_kept": 0, "cand_in_hist": {}, "cand_out_hist": {},
        "window_rows": [], "model_calls_forwarded": 0,
        "note": "the policy was never loaded, so no decision was ever pulsed",
    })
    summary["alignment"] = align_report
    summary["run"] = {
        "seed": int(config.seed), "arm": str(config.arm),
        "entry_frame": int(config.entry_frame),
        "cap_frames": int(config.cap_frames),
        "out": config.out.as_posix(),
        "decision_log": (config.decision_log.as_posix()
                         if config.decision_log is not None else None),
        "pulse_log": (Path(args.pulse_log).as_posix()
                      if args.pulse_log else None),
        "frames": result.get("frames"), "decisions": result.get("decisions"),
        "stop": result.get("stop"), "entry_verified": result.get("entry_verified"),
        "result_json": (config.out / "result.json").as_posix(),
        "exit_code": result.get("exit_code"),
    }
    player_report = player_audit(
        probe=probe, hook=holder.get("hook"), log_rows=log_rows,
        require=require_player)
    if probe is None:
        player_report["enabled"] = False
        player_report["note"] = ("--no-log-player: no PLAYER command was sent and the "
                                 "decision log carries only the pinned runner's fields")
        player_report["ok"] = True
    summary["player"] = player_report
    PINNED.write_json(config.out / "pulse.json", summary)

    player_problems = player_failure_reasons(player_report)
    if align_report.get("ok") is False:
        rc = EXIT_TECHNICAL
        print("run: FAILED the pulse log does not align with the decision log")
        for problem in align_report.get("problems") or []:
            print(f"  {problem}")
    elif player_problems:
        rc = EXIT_TECHNICAL
        print("run: FAILED the player coordinate contract")
        for problem in player_problems:
            print(f"  {problem}")
    else:
        rc = int(result.get("exit_code", EXIT_TECHNICAL))
        print(f"run: arm={result.get('arm')} frames={result.get('frames')} "
              f"decisions={result.get('decisions')} stop={result.get('stop')} "
              f"rc={rc} pulse_applied={summary['applied']} "
              f"pulse_changed={summary['changed']} "
              f"window={summary['config']['window']} "
              f"cand={summary['config']['cand']} align={align_report.get('ok')} "
              f"player_rows={player_report['rows_with_player']}/"
              f"{player_report['decision_rows']} "
              f"player_reads={player_report['reads']} "
              f"player_ok={player_report['ok']}")
    print(f"wrote: {(config.out / 'pulse.json').as_posix()}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
