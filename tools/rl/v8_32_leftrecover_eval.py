#!/usr/bin/env python3
"""V832-S2Z: the *left-recovery* guard -- force ``cand 3`` (E) from the left side.

Why this run exists (the premise, taken from the previous round's measurement)
------------------------------------------------------------------------------
The natural-opening corpus shows the T04 student living on the **left** of the
movement area, not on the right wall both earlier guards aimed at:

* ``x < 100`` holds in 79.5 % of 1,840 student states; ``x = 8.0`` -- the left
  clamp ``x = MOVE_AREA_X`` (``touhou_rl/spec.py:366-369``) -- appears 86 times
  and ``x >= 300`` never occurs;
* the teacher's ``cand3`` (E, ``+x``) share over those same states is 100 % at
  ``x <= 8.001``, 48.56 % at ``x < 100``, 28.64 % at 100-200 and 14.56 % at
  ``x >= 200`` -- strictly falling with ``x``;
* the student does **not** execute that advice: over the 1,462 rows with
  ``x < 100`` where the teacher says right, the student emits right 9.58 % of the
  time and idle 28.80 %.

So the two earlier guards (idle substitution V832-S2S, hysteresis bounce
V832-S2T) both policed the *right* wall while the student is pinned on the left.
This round applies the teacher's own label as a recovery rule on the left side.

The rule, written before the run (``.tmp_v832_s2z/results.json:prior``)
----------------------------------------------------------------------
::

    if px <= clamp_x (8.5):                             # left clamp: teacher 100 % right
        emit force_cand (3)
    elif px <= left_x (100) and argmax move in {0, 7}:  # idle or left while on the left
        emit force_cand (3)
    else:
        no intervention

``move = cand % MOVES_PER_CAND``; ``move 0`` = idle, ``move 7`` = W (``-x``); the
substitute ``cand 3`` = ``focus 0`` + ``move 3`` = E (``+x``).  Both branches are
evaluated on the ``px`` measured *before* the action, so the decision is a
function of the state it is made at.  ``--left-x -1`` disables the whole guard
(both branches) and every extra engine command: that is the equivalence switch.

Falsifiable prediction (pre-registered)
--------------------------------------
P1: the 15-seed mean of this guard is ``> 1,898.3`` frames (the same-seed mean of
pure T04 = ``.tmp_v832_s2m/ckpt_s1.pt``) with a paired difference ``>= +200``
frames.  P2: the share of decisions with ``px <= 16`` is *lower* than the pure
T04 baseline.  P3: the mean is *not worse* than 900 frames (the DAgger 910.6
floor).  P1 and P2 together = "the left side can be opened"; P1 false = "forcing
right cannot open it", with P3 separating "useless" from "harmful".

What this controller is
-----------------------
It is the pinned ruler ``tools/rl/v400_student_eval.py`` with **one** change: the
action-selection point.  Everything else -- checkpoint loading, the obs-v3
planes, the 2-frame decision budget, the 4-frame stack, the frame cap, the
stop-on-death rule, the ``--decision-log`` schema -- is imported from that file
(by path, not copied), so a record produced here is the same record class the
pinned tool produces.  The pinned file itself is never modified; its sha256 is
recorded in every episode record.  This file was copied from
``v8_32_wallguard_eval.py`` (V832-S2S), which is left byte-for-byte untouched;
its right-wall state machine is replaced by the left-recovery rule above.

Two hard invariants (unchanged from the tool this was copied from):

1. **Guard off == pinned tool, bit for bit.**  ``--left-x -1`` disables the guard
   *and* every extra engine command: the wrapper then returns the model's own
   ``(cand, bomb)`` untouched, reads no ``PLAYER`` reply and writes no guard row.
   The run's ``decisions.jsonl`` must hash-match the pinned run's file.
2. **Guard on adds exactly one read-only command per decision.**  The player x is
   read with ``PLAYER`` (``touhou_rl/v400/gym/protocol.py::parse_player_reply``,
   field ``px``, the same movement-area coordinate ``MOVE_AREA_*`` describe -- the
   field name and unit are the protocol's, not assumed here) *before* the action is
   emitted, so ``x`` is the position the decision is actually made at.  Every
   decision's ``x``, whether a branch fired, which branch, and the overridden
   argmax candidate are written to ``--guard-log`` (JSONL), so the trigger count
   can be recomputed from raw bytes.  A failed read is recorded, never forged: the
   last known ``x`` is carried forward and marked ``x_source="carry"``; with no
   prior reading the decision is marked ``x_source="unavailable"`` and cannot fire.

Run inside WSL::

    xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
      tools/rl/v8_32_leftrecover_eval.py \
      --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
      --data-dir .tmp_v832_s2z/game-muted \
      --checkpoint .tmp_v832_s2m/ckpt_s1.pt \
      --seeds 1000 --stage 0 --difficulty normal --max-frames 4500 \
      --stop-on-death --left-x 100 --clamp-x 8.5 --force-cand 3 \
      --out-dir .tmp_v832_s2z/G/1000 \
      --decision-log .tmp_v832_s2z/G/1000/decisions.jsonl \
      --guard-log .tmp_v832_s2z/G/1000/guard.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from touhou_rl.v400.gym import actions as gym_actions  # noqa: E402

CONTROLLER = "tools/rl/v8_32_leftrecover_eval.py"
PINNED_CONTROLLER = "tools/rl/v400_student_eval.py"
GUARD_SCHEMA = "v832-s2z-leftrecover-decision-1"
GUARD_SUMMARY_SCHEMA = "v832-s2z-leftrecover-run-1"
#: A negative ``--left-x`` disables the guard (both branches) *and* every extra
#: engine command.  It is the equivalence switch: a disabled run must be
#: bit-identical to the pinned ruler.
LEFT_DISABLE_X = -1.0
DEFAULT_LEFT_X = 100.0
DEFAULT_CLAMP_X = 8.5
DEFAULT_LEFT_MOVES = "0,7"
DEFAULT_FORCE_CAND = 3
#: The two branch names a guard row may carry, in precedence order.
BRANCH_CLAMP = "clamp"
BRANCH_BAND = "band"
BRANCH_NONE = "none"


# ---------------------------------------------------------------------------
# Pinned ruler, imported by path
# ---------------------------------------------------------------------------
def pinned_path() -> Path:
    """The pinned evaluation controller this tool extends (never modified)."""

    return Path(__file__).resolve().parent / PINNED_CONTROLLER.split("/")[-1]


def load_pinned(path: str | Path | None = None):
    """Import ``v400_student_eval.py`` as a module, once.

    Importing it only defines functions and constants -- its ``main`` is behind an
    ``if __name__ == "__main__"`` guard -- so this cannot start an engine.
    """

    target = Path(path) if path is not None else pinned_path()
    name = "v400_student_eval_pinned"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, target)
    if spec is None or spec.loader is None:  # pragma: no cover - broken install
        raise RuntimeError(f"cannot import the pinned ruler from {target}")
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
# Guard configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class GuardConfig:
    """The three parameters the prediction names, plus the derived switch."""

    left_x: float
    clamp_x: float
    force_cand: int
    left_moves: tuple[int, ...]
    enabled: bool

    @property
    def force_move(self) -> int:
        return int(self.force_cand) % gym_actions.MOVES_PER_CAND

    def as_dict(self) -> dict:
        return {
            "left_x": float(self.left_x),
            "clamp_x": float(self.clamp_x),
            "force_cand": int(self.force_cand),
            "force_move": self.force_move,
            "left_moves": [int(move) for move in self.left_moves],
            "enabled": bool(self.enabled),
            "disable_left_x": float(LEFT_DISABLE_X),
            "rule": ("if px <= clamp_x emit force_cand; elif px <= left_x and "
                     "(cand % 9) in left_moves emit force_cand; else leave the "
                     "argmax alone.  px is read with PLAYER.px before the action; "
                     "a negative left_x disables both branches and every extra "
                     "engine command"),
        }


def parse_left_moves(text: str) -> tuple[int, ...]:
    """``"0,7"`` -> ``(0, 7)``; fail closed on anything else."""

    moves: list[int] = []
    for piece in str(text or "").split(","):
        token = piece.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"left move {token!r} is not an integer") from exc
        if not 0 <= value < gym_actions.MOVES_PER_CAND:
            raise ValueError(f"left move {value} is outside 0.."
                             f"{gym_actions.MOVES_PER_CAND - 1}")
        moves.append(value)
    if not moves:
        raise ValueError("--left-moves is empty")
    return tuple(sorted(set(moves)))


def resolve_guard(*, left_x: float, clamp_x: float, force_cand: int,
                  left_moves: str = DEFAULT_LEFT_MOVES) -> GuardConfig:
    """Validate the guard parameters before any file or engine exists."""

    try:
        x = float(left_x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--left-x {left_x!r} is not a number") from exc
    try:
        clamp = float(clamp_x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--clamp-x {clamp_x!r} is not a number") from exc
    moves = parse_left_moves(left_moves)
    try:
        cand = int(force_cand)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--force-cand {force_cand!r} is not an integer") from exc
    if not 0 <= cand < gym_actions.N_CAND:
        raise ValueError(f"--force-cand {cand} is outside 0..{gym_actions.N_CAND - 1}")
    if cand % gym_actions.MOVES_PER_CAND in moves:
        raise ValueError(
            f"--force-cand {cand} itself carries a left move "
            f"({cand % gym_actions.MOVES_PER_CAND}); it would not push the ship "
            "away from the left clamp")
    enabled = x >= 0.0
    if enabled:
        if clamp < 0.0:
            raise ValueError(
                f"--clamp-x {clamp} is negative while the guard is enabled "
                f"(--left-x {x} >= 0): a negative clamp would silently disable the "
                "left-clamp branch; use --left-x -1 to disable the whole guard")
        if clamp > x:
            raise ValueError(
                f"--clamp-x {clamp} must not exceed --left-x {x}: the unconditional "
                "branch would then swallow part of the move-filtered band")
    return GuardConfig(left_x=x, clamp_x=clamp, force_cand=cand,
                       left_moves=moves, enabled=enabled)


# ---------------------------------------------------------------------------
# Guarded action selection
# ---------------------------------------------------------------------------
def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * float(q)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    frac = position - low
    return float(ordered[low] * (1.0 - frac) + ordered[high] * frac)


class LeftRecoverModel:
    """Wrap the student so only the emitted candidate can be changed.

    The wrapped model is called exactly as :func:`run_episode` would call it, so
    the policy's inputs (planes, HUD strip, action history) are untouched; only the
    candidate that reaches ``cand_to_mask`` may be replaced.  ``bomb`` is never
    changed.

    The rule is two nested branches on the pre-action ``px`` of the decision it
    acts on::

        if px <= clamp_x:                                # left clamp, unconditional
            branch = "clamp"; out = force_cand
        elif px <= left_x and move in left_moves:        # idle/left while on the left
            branch = "band";  out = force_cand
        else:
            branch = "none";  out = cand

    ``fired`` means a branch matched; ``changed`` means the emitted candidate
    actually differs from the argmax (a clamp decision whose argmax is already
    ``force_cand`` fires and changes nothing).  Both are recorded per decision, so
    the verifier can recompute either count from the raw rows.
    """

    def __init__(self, model, *, seed: int, config: GuardConfig,
                 client_ref, log=None) -> None:
        self.model = model
        self.seed = int(seed)
        self.config = config
        self.client_ref = client_ref
        self.log = log
        self.calls = 0
        self.fires = 0
        self.fires_clamp = 0
        self.fires_band = 0
        self.changed = 0
        self.player_reads = 0
        self.read_errors = 0
        self.x_unavailable = 0
        self.last_x: float | None = None
        self.x_values: list[float] = []
        self.fired_x_values: list[float] = []
        self.x_hist: dict[str, int] = {}
        self.fired_x_hist: dict[str, int] = {}
        self.replaced_cand_hist: dict[str, int] = {}
        self.replaced_move_hist: dict[str, int] = {}
        self.first_fire: dict | None = None
        self.last_fire: dict | None = None
        self.moves_hist: dict[str, int] = {}
        self.cand_in_hist: dict[str, int] = {}
        self.cand_out_hist: dict[str, int] = {}

    # -- x ---------------------------------------------------------------
    def _read_x(self) -> tuple[float | None, str, str | None, int | None]:
        """``(x, source, error, engine_frame)`` -- never a fabricated number."""

        client = self.client_ref()
        if client is None:
            return self.last_x, ("carry" if self.last_x is not None else "unavailable"), \
                "no client attached", None
        try:
            snapshot = client.player()
        except Exception as exc:  # noqa: BLE001 - a refused read is recorded, not raised
            self.read_errors += 1
            source = "carry" if self.last_x is not None else "unavailable"
            if self.last_x is None:
                self.x_unavailable += 1
            return self.last_x, source, f"{type(exc).__name__}: {exc}", None
        self.player_reads += 1
        x = float(snapshot.position[0])
        self.last_x = x
        return x, "player", None, int(snapshot.frame)

    # -- decision --------------------------------------------------------
    def decide(self, *, cand: int, x: float | None) -> tuple[int, str]:
        """The pure rule: ``(cand_out, branch)``.  No I/O, no state."""

        if not self.config.enabled or x is None:
            return int(cand), BRANCH_NONE
        if x <= self.config.clamp_x:
            return int(self.config.force_cand), BRANCH_CLAMP
        move = int(cand) % gym_actions.MOVES_PER_CAND
        if x <= self.config.left_x and move in self.config.left_moves:
            return int(self.config.force_cand), BRANCH_BAND
        return int(cand), BRANCH_NONE

    # -- act -------------------------------------------------------------
    def act(self, *args, **kwargs):
        cand, bomb = self.model.act(*args, **kwargs)
        cand = int(cand)
        bomb = int(bomb)
        if not self.config.enabled:
            # Equivalence mode: no read, no row, no substitution -- the pinned
            # ruler's own output goes through unchanged.
            return cand, bomb
        decision = self.calls
        self.calls += 1
        x, source, error, engine_frame = self._read_x()
        move = cand % gym_actions.MOVES_PER_CAND
        focus = cand // gym_actions.MOVES_PER_CAND
        out, branch = self.decide(cand=cand, x=x)
        fired = branch != BRANCH_NONE
        changed = out != cand
        mask_in = gym_actions.cand_to_mask(cand, bomb)
        mask_out = gym_actions.cand_to_mask(out, bomb)

        if x is not None:
            self.x_values.append(x)
            bucket = str(int(round(x)))
            self.x_hist[bucket] = self.x_hist.get(bucket, 0) + 1
        self.moves_hist[str(move)] = self.moves_hist.get(str(move), 0) + 1
        self.cand_in_hist[str(cand)] = self.cand_in_hist.get(str(cand), 0) + 1
        self.cand_out_hist[str(out)] = self.cand_out_hist.get(str(out), 0) + 1
        if fired:
            self.fires += 1
            if branch == BRANCH_CLAMP:
                self.fires_clamp += 1
            else:
                self.fires_band += 1
            if changed:
                self.changed += 1
                self.replaced_cand_hist[str(cand)] = \
                    self.replaced_cand_hist.get(str(cand), 0) + 1
                self.replaced_move_hist[str(move)] = \
                    self.replaced_move_hist.get(str(move), 0) + 1
            self.fired_x_values.append(float(x))
            bucket = str(int(round(x)))
            self.fired_x_hist[bucket] = self.fired_x_hist.get(bucket, 0) + 1
            entry = {
                "decision": int(decision),
                "engine_frame": engine_frame,
                "x": float(x),
                "branch": branch,
                "cand_in": int(cand),
                "cand_out": int(out),
                "changed": bool(changed),
            }
            if self.first_fire is None:
                self.first_fire = dict(entry)
            self.last_fire = dict(entry)
        if self.log is not None:
            self.log.write({
                "schema": GUARD_SCHEMA,
                "seed": self.seed,
                "decision": int(decision),
                "engine_frame": engine_frame,
                "x": x,
                "x_source": source,
                "x_error": error,
                "move": int(move),
                "focus": int(focus),
                "bomb": int(bomb),
                "cand_in": int(cand),
                "cand_out": int(out),
                "mask_in": int(mask_in),
                "mask_out": int(mask_out),
                "branch": branch,
                "fired": bool(fired),
                "changed": bool(changed),
                "left_x": float(self.config.left_x),
                "clamp_x": float(self.config.clamp_x),
                "force_cand": int(self.config.force_cand),
                "left_moves": [int(m) for m in self.config.left_moves],
            })
        return out, bomb

    # -- report ----------------------------------------------------------
    def summary(self) -> dict:
        return {
            "schema": GUARD_SUMMARY_SCHEMA,
            "controller": CONTROLLER,
            "config": self.config.as_dict(),
            "decisions": int(self.calls),
            "fires": int(self.fires),
            "fires_clamp": int(self.fires_clamp),
            "fires_band": int(self.fires_band),
            "changed": int(self.changed),
            "fire_share": (float(self.fires) / self.calls) if self.calls else None,
            "player_reads": int(self.player_reads),
            "read_errors": int(self.read_errors),
            "x_unavailable": int(self.x_unavailable),
            "x_min": min(self.x_values) if self.x_values else None,
            "x_max": max(self.x_values) if self.x_values else None,
            "x_p50": _percentile(self.x_values, 0.5),
            "x_p90": _percentile(self.x_values, 0.9),
            "x_le_16": int(sum(1 for x in self.x_values if x <= 16.0)),
            "x_le_16_share": (float(sum(1 for x in self.x_values if x <= 16.0))
                              / len(self.x_values)) if self.x_values else None,
            "x_at_or_below_left_x": int(sum(1 for x in self.x_values
                                            if x <= self.config.left_x)),
            "x_at_or_below_clamp_x": int(sum(1 for x in self.x_values
                                             if x <= self.config.clamp_x)),
            "x_hist": dict(sorted(self.x_hist.items(), key=lambda kv: int(kv[0]))),
            "fired_x_min": min(self.fired_x_values) if self.fired_x_values else None,
            "fired_x_max": max(self.fired_x_values) if self.fired_x_values else None,
            "fired_x_hist": dict(sorted(self.fired_x_hist.items(),
                                        key=lambda kv: int(kv[0]))),
            "replaced_cand_hist": dict(sorted(self.replaced_cand_hist.items(),
                                              key=lambda kv: int(kv[0]))),
            "replaced_move_hist": dict(sorted(self.replaced_move_hist.items(),
                                              key=lambda kv: int(kv[0]))),
            "first_fire": self.first_fire,
            "last_fire": self.last_fire,
            "moves_hist": dict(sorted(self.moves_hist.items(),
                                      key=lambda kv: int(kv[0]))),
            "cand_in_hist": dict(sorted(self.cand_in_hist.items(),
                                        key=lambda kv: int(kv[0]))),
            "cand_out_hist": dict(sorted(self.cand_out_hist.items(),
                                         key=lambda kv: int(kv[0]))),
        }


@contextlib.contextmanager
def captured_client(holder: dict):
    """Expose the episode's client to the guard for the duration of one episode.

    ``run_episode`` builds its own ``EnvHarnessClient`` internally; the subclass
    only records the instance.  It adds no command, no thread and no behaviour.
    """

    original = PINNED.EnvHarnessClient

    class _Captured(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            holder["client"] = self
            holder["instances"] = int(holder.get("instances", 0)) + 1

    PINNED.EnvHarnessClient = _Captured
    try:
        yield holder
    finally:
        PINNED.EnvHarnessClient = original
        holder["client"] = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V832-S2Z left-recovery ruler: the pinned v400_student_eval run "
                    "loop with a left-side force-right guard inserted at action "
                    "selection.")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", default="1000")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=4500)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--difficulty", default="normal",
                        choices=list(PINNED.DIFFICULTIES))
    parser.add_argument("--stop-on-death", action="store_true")
    parser.add_argument("--obs-window", choices=list(PINNED.OBS_WINDOWS),
                        default="consecutive")
    parser.add_argument("--obs", choices=("v3",), default="v3")
    parser.add_argument("--arch", choices=("v3",), default="v3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-iters", type=int, default=PINNED.BOOTSTRAP_ITERS)
    parser.add_argument("--ci-level", type=float, default=PINNED.CI_LEVEL)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--decision-log", default=None,
                        help="pinned per-decision JSONL (schema "
                             "v400-student-eval-decision-1); byte-comparable with "
                             "a pinned run when the guard is disabled")
    parser.add_argument("--guard-log", default=None,
                        help="one JSONL row per decision with the pre-action px, the "
                             "fired branch and the overridden argmax candidate "
                             "(schema %s); mandatory while the guard is enabled"
                             % GUARD_SCHEMA)
    parser.add_argument("--left-x", type=float, default=DEFAULT_LEFT_X,
                        help="fire when px <= this and the argmax move is a left "
                             "move (default 100).  A negative value disables the "
                             "whole guard and every extra engine command (%g)"
                             % LEFT_DISABLE_X)
    parser.add_argument("--clamp-x", type=float, default=DEFAULT_CLAMP_X,
                        help="fire unconditionally when px <= this (left clamp, "
                             "default 8.5); must be <= --left-x")
    parser.add_argument("--force-cand", type=int, default=DEFAULT_FORCE_CAND,
                        help="candidate emitted when a branch fires; 3 = E (+x, "
                             "default)")
    parser.add_argument("--left-moves", default=DEFAULT_LEFT_MOVES,
                        help="comma list of `move` ids that count as left side: "
                             "default 0,7 = idle / W")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = resolve_guard(left_x=args.left_x, clamp_x=args.clamp_x,
                               force_cand=args.force_cand,
                               left_moves=args.left_moves)
    except ValueError as exc:
        parser.error(str(exc))
    if config.enabled and not (args.guard_log or "").strip():
        parser.error("--guard-log is required while the guard is enabled: a run "
                     "whose guard telemetry is missing is not a measured run")

    decision_log = None
    guard_log = None
    try:
        decision_log = PINNED.open_decision_log(args.decision_log)
        guard_log = PINNED.open_decision_log(args.guard_log)
    except PINNED.EvalError as exc:
        parser.error(str(exc))

    try:
        out_dir = Path(args.out_dir)
        if PINNED.route_obs(args) != "v3":
            raise SystemExit("this ruler only evaluates the obs-v3 / arch-v3 route "
                             "(the checkpoint under test is an obs-v3 student)")
        out_dir.mkdir(parents=True, exist_ok=True)
        import torch  # noqa: F401 - only the policy route needs it

        model, payload = PINNED.load_checkpoint_v3(args.checkpoint,
                                                   device=args.device)
        model.eval()
        disclosure, policy_note = PINNED._policy_identity(payload)
        seeds = PINNED.parse_seeds(args.seeds)
        repeats = int(args.repeats)
        if repeats < 1:
            raise SystemExit(f"--repeats must be >= 1, got {repeats!r}")

        records: list[dict] = []
        for repeat in range(repeats):
            for seed in seeds:
                shm_name = (f"s2z-{os.getpid()}-{seed}-"
                            f"{int(time.time() * 1000) % 100000}")
                holder: dict = {}
                guarded = LeftRecoverModel(model, seed=seed, config=config,
                                           client_ref=lambda: holder.get("client"),
                                           log=guard_log)
                with captured_client(holder):
                    record = PINNED.run_episode(
                        engine=args.engine,
                        data_dir=args.data_dir,
                        checkpoint_path=args.checkpoint,
                        model=guarded,
                        seed=seed,
                        stage=args.stage,
                        difficulty=args.difficulty,
                        max_frames=args.max_frames,
                        shm_name=shm_name,
                        stop_on_death=bool(args.stop_on_death),
                        obs_window=str(args.obs_window),
                        obs_v2=False,
                        obs_v3=True,
                        arch="v3",
                        repeat=repeat,
                        arm="policy",
                        constant_cand=0,
                        decision_log=decision_log,
                    )
                # Traceability only; the pinned validator already ran inside
                # run_episode, so these keys cannot alter the measured record.
                record["controller"] = CONTROLLER
                record["pinned_controller"] = str(pinned_path())
                record["pinned_controller_sha256"] = PINNED._sha256(pinned_path())
                record["leftrecover"] = guarded.summary()
                record["leftrecover"]["client_instances"] = int(holder.get("instances", 0))
                records.append(record)
                print(json.dumps({
                    "seed": record["seed"],
                    "frames": record["frames"],
                    "deaths": record["deaths"],
                    "end": record["end"],
                    "score": record["score"],
                    "fires": record["leftrecover"]["fires"],
                    "fires_clamp": record["leftrecover"]["fires_clamp"],
                    "fires_band": record["leftrecover"]["fires_band"],
                    "changed": record["leftrecover"]["changed"],
                    "decisions": record["leftrecover"]["decisions"],
                    "x_min": record["leftrecover"]["x_min"],
                    "x_max": record["leftrecover"]["x_max"],
                    "x_le_16": record["leftrecover"]["x_le_16"],
                    "wall_s": record["wall_s"],
                }, ensure_ascii=False), flush=True)
                (out_dir / f"leftrecover-{seed}.json").write_text(
                    json.dumps(record["leftrecover"], indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")

        summary = PINNED.summarise(records, disclosure=disclosure,
                                   policy_note=policy_note,
                                   bootstrap_iters=int(args.bootstrap_iters),
                                   ci_level=float(args.ci_level))
    finally:
        if decision_log is not None:
            decision_log.close()
        if guard_log is not None:
            guard_log.close()

    summary["controller"] = CONTROLLER
    summary["controller_note"] = (
        "the pinned ruler's run loop with a left-side force-right guard at action "
        "selection; with --left-x < 0 no command and no action differs from %s"
        % PINNED_CONTROLLER)
    summary["leftrecover"] = {
        "config": config.as_dict(),
        "fires_total": int(sum(r["leftrecover"]["fires"] for r in records)),
        "fires_clamp_total": int(sum(r["leftrecover"]["fires_clamp"]
                                     for r in records)),
        "fires_band_total": int(sum(r["leftrecover"]["fires_band"]
                                    for r in records)),
        "changed_total": int(sum(r["leftrecover"]["changed"] for r in records)),
        "per_seed": [
            {
                "seed": record["seed"],
                "frames": record["frames"],
                "deaths": record["deaths"],
                "end": record["end"],
                "decisions": record["leftrecover"]["decisions"],
                "fires": record["leftrecover"]["fires"],
                "fires_clamp": record["leftrecover"]["fires_clamp"],
                "fires_band": record["leftrecover"]["fires_band"],
                "changed": record["leftrecover"]["changed"],
                "x_min": record["leftrecover"]["x_min"],
                "x_max": record["leftrecover"]["x_max"],
                "x_le_16": record["leftrecover"]["x_le_16"],
                "x_le_16_share": record["leftrecover"]["x_le_16_share"],
                "x_hist": record["leftrecover"]["x_hist"],
                "fired_x_hist": record["leftrecover"]["fired_x_hist"],
                "replaced_cand_hist": record["leftrecover"]["replaced_cand_hist"],
                "replaced_move_hist": record["leftrecover"]["replaced_move_hist"],
            }
            for record in records
        ],
    }

    (out_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "leftrecover": summary["leftrecover"]["config"],
        "fires_total": summary["leftrecover"]["fires_total"],
        "fires_clamp_total": summary["leftrecover"]["fires_clamp_total"],
        "fires_band_total": summary["leftrecover"]["fires_band_total"],
        "changed_total": summary["leftrecover"]["changed_total"],
        "frames": [record["frames"] for record in records],
    }, ensure_ascii=False), flush=True)
    failed = [record for record in records if record["end"] == "error"]
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
