#!/usr/bin/env python3
"""V832-S2T: the hysteresis *bounce* guard for the DAgger student D2.

Why this run exists (the premise, taken from the previous round's measurement)
-------------------------------------------------------------------------------
V832-S2S replaced a +x action with ``cand 0`` (idle) whenever ``x >= 370``.  It
fired 2,781 times, but 2,193 of those fires were at exactly ``x = 376.0`` -- the
right wall ``x = MOVE_AREA_X + MOVE_AREA_W`` (``touhou_rl/spec.py:366-369``),
where *both* E and idle move the ship by 0 px.  The trajectory therefore did not
change (854-1,015 frames, 0/15 above 1,200) and the "frozen in the corner"
mechanism was falsified as an *idle-substitution* claim.

What changed here
-----------------
Idle cannot leave the wall; W (``cand 7``, ``move 7``, ``-x``) can.  So the
substitution is no longer instantaneous -- it is a **state with hysteresis**:

* entry (arm the state): ``x >= --guard-x`` (370) **and** the argmax action
  carries a +x component (``move in --guard-moves``, default ``{2,3,4}``);
* while the state is armed **every** decision emits ``--excursion-cand``
  (default ``7`` = W), whatever the argmax is;
* the state disarms only when ``x <= --release-x`` (340), i.e. after the ship has
  actually been pushed 30+ px away from the wall, not merely one decision later.

The state is evaluated on the ``x`` measured *before* the action, so for every
decision ``in_bounce`` and ``cand_out == excursion_cand`` are the same fact; the
argmax candidate that was overridden is still recorded (``cand_in``), so the whole
run is recomputable from raw bytes.

Falsifiable prediction
----------------------
If "frozen in the right corner" is the repairable cause of D2's early death, the
natural opening must last ``> 1,200`` frames on ``>= 10/15`` seeds.  ``<= 5/15``
means the corner as a *repairable* cause is falsified too.

What this controller is
-----------------------
It is the pinned ruler ``tools/rl/v400_student_eval.py`` with **one** change: the
action-selection point.  Everything else -- checkpoint loading, the obs-v3 planes,
the 2-frame decision budget, the 4-frame stack, the frame cap, the stop-on-death
rule, the ``--decision-log`` schema -- is imported from that file (by path, not
copied), so a record produced here is the same record class the pinned tool
produces.  The pinned file itself is never modified; its sha256 is recorded in
every episode record.  This file was copied from ``v8_32_wallguard_eval.py``
(V832-S2S), which is left byte-for-byte untouched.

Two hard invariants (unchanged from the tool this was copied from):

1. **Guard off == pinned tool, bit for bit.**  ``--guard-x 99999`` disables the
   guard *and* every extra engine command: the wrapper then returns the model's
   own ``(cand, bomb)`` untouched, reads no ``PLAYER`` reply and writes no guard
   row.  The run's ``decisions.jsonl`` must hash-match the pinned run's file.
2. **Guard on adds exactly one read-only command per decision.**  The player x is
   read with ``PLAYER`` (``touhou_rl/v400/gym/protocol.py::parse_player_reply``)
   before the action is emitted, so ``x`` is the position the decision is actually
   made at.  Every decision's ``x``, ``in_bounce`` and overridden ``cand_in`` are
   written to ``--guard-log`` (JSONL) so every published count can be recomputed
   from raw bytes.  A failed read is recorded, never forged: the last known ``x``
   is carried forward and marked ``x_source="carry"``; with no prior reading the
   decision is marked ``x_source="unavailable"`` and cannot arm the state.

Run inside WSL::

    xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
      tools/rl/v8_32_bounce_eval.py \
      --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
      --data-dir .tmp_v832_s2t/game-muted \
      --checkpoint .tmp_v832_s2h/ckpt_d2.pt \
      --seeds 1000 --stage 0 --difficulty normal --max-frames 4500 \
      --stop-on-death --guard-x 370 --release-x 340 --excursion-cand 7 \
      --guard-moves 2,3,4 \
      --out-dir .tmp_v832_s2t/G/1000 \
      --decision-log .tmp_v832_s2t/G/1000/decisions.jsonl \
      --guard-log .tmp_v832_s2t/G/1000/guard.jsonl
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

CONTROLLER = "tools/rl/v8_32_bounce_eval.py"
PINNED_CONTROLLER = "tools/rl/v400_student_eval.py"
GUARD_SCHEMA = "v832-s2t-bounce-decision-1"
GUARD_SUMMARY_SCHEMA = "v832-s2t-bounce-run-1"
#: ``--guard-x`` at or above this value disables the guard (and every extra
#: engine command).  It is the equivalence switch: a disabled run must be
#: bit-identical to the pinned ruler.
GUARD_DISABLE_X = 99999.0
DEFAULT_GUARD_X = 370.0
DEFAULT_RELEASE_X = 340.0
DEFAULT_GUARD_MOVES = "2,3,4"
DEFAULT_EXCURSION_CAND = 7


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
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


PINNED = load_pinned()


# ---------------------------------------------------------------------------
# Guard configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class GuardConfig:
    """The four parameters the prediction names, plus the derived switch."""

    guard_x: float
    release_x: float
    guard_moves: tuple[int, ...]
    excursion_cand: int
    enabled: bool

    @property
    def excursion_move(self) -> int:
        return int(self.excursion_cand) % gym_actions.MOVES_PER_CAND

    @property
    def hysteresis_px(self) -> float:
        return float(self.guard_x) - float(self.release_x)

    def as_dict(self) -> dict:
        return {
            "guard_x": float(self.guard_x),
            "release_x": float(self.release_x),
            "hysteresis_px": self.hysteresis_px,
            "guard_moves": [int(move) for move in self.guard_moves],
            "excursion_cand": int(self.excursion_cand),
            "excursion_move": self.excursion_move,
            "enabled": bool(self.enabled),
            "disable_x": float(GUARD_DISABLE_X),
            "rule": ("arm when x >= guard_x and (cand % 9) in guard_moves; while "
                     "armed emit excursion_cand on every decision; disarm when "
                     "x <= release_x; x is read with PLAYER.px before the action"),
        }


def parse_guard_moves(text: str) -> tuple[int, ...]:
    """``"2,3,4"`` -> ``(2, 3, 4)``; fail closed on anything else."""

    moves: list[int] = []
    for piece in str(text or "").split(","):
        token = piece.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"guard move {token!r} is not an integer") from exc
        if not 0 <= value < gym_actions.MOVES_PER_CAND:
            raise ValueError(f"guard move {value} is outside 0.."
                             f"{gym_actions.MOVES_PER_CAND - 1}")
        moves.append(value)
    if not moves:
        raise ValueError("--guard-moves is empty")
    return tuple(sorted(set(moves)))


def resolve_guard(*, guard_x: float, release_x: float, guard_moves: str,
                  excursion_cand: int) -> GuardConfig:
    """Validate the guard parameters before any file or engine exists."""

    try:
        x = float(guard_x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--guard-x {guard_x!r} is not a number") from exc
    try:
        release = float(release_x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--release-x {release_x!r} is not a number") from exc
    moves = parse_guard_moves(guard_moves)
    try:
        cand = int(excursion_cand)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--excursion-cand {excursion_cand!r} is not an "
                         "integer") from exc
    if not 0 <= cand < gym_actions.N_CAND:
        raise ValueError(f"--excursion-cand {cand} is outside "
                         f"0..{gym_actions.N_CAND - 1}")
    if release >= x:
        raise ValueError(
            f"--release-x {release} must be strictly below --guard-x {x}: without "
            "a positive dead band the state could re-arm on the very next decision "
            "and the hysteresis would be a no-op")
    if cand % gym_actions.MOVES_PER_CAND in moves:
        raise ValueError(
            f"--excursion-cand {cand} itself carries a guarded +x move "
            f"({cand % gym_actions.MOVES_PER_CAND}); it would not push the ship "
            "away from the wall")
    return GuardConfig(guard_x=x, release_x=release, guard_moves=moves,
                       excursion_cand=cand, enabled=x < GUARD_DISABLE_X)


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


class BounceGuardModel:
    """Wrap the student so only the emitted candidate can be changed.

    The wrapped model is called exactly as :func:`run_episode` would call it, so
    the policy's inputs (planes, HUD strip, action history) are untouched; only the
    candidate that reaches ``cand_to_mask`` may be replaced.  ``bomb`` is never
    changed.

    The substitution is a two-branch state machine with hysteresis, evaluated on the
    pre-action ``x`` of the decision it acts on::

        if bouncing and x <= release_x:                 # off the wall again
            bouncing = False
        if (not bouncing) and x >= guard_x and move in guard_moves:
            bouncing = True                             # a +x action at the wall
        in_bounce = bouncing
        out = excursion_cand if in_bounce else cand

    The two arms cannot both fire on one decision (``release_x < guard_x``), so the
    branch order is not observable.  ``in_bounce`` is therefore identical to
    ``out == excursion_cand`` on every decision -- which is exactly what the
    independent verifier recomputes from the raw ``guard.jsonl`` rows.
    """

    def __init__(self, model, *, seed: int, config: GuardConfig,
                 client_ref, log=None) -> None:
        self.model = model
        self.seed = int(seed)
        self.config = config
        self.client_ref = client_ref
        self.log = log
        self.calls = 0
        self.bouncing = False
        self.entries = 0
        self.exits = 0
        self.excursion_decisions = 0
        self.player_reads = 0
        self.read_errors = 0
        self.x_unavailable = 0
        self.last_x: float | None = None
        self.x_values: list[float] = []
        self.bounce_x_values: list[float] = []
        self.x_hist: dict[str, int] = {}
        self.bounce_x_hist: dict[str, int] = {}
        self.replaced_cand_hist: dict[str, int] = {}
        self.replaced_move_hist: dict[str, int] = {}
        self.runs: list[dict] = []
        self.run_lengths: list[int] = []
        self.current_run: dict | None = None
        self.first_entry: dict | None = None
        self.last_entry: dict | None = None
        self.first_exit: dict | None = None
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

        # -- hysteresis state machine, on the pre-action x ----------------
        exited = False
        entered = False
        if self.bouncing:
            if x is not None and x <= self.config.release_x:
                self.bouncing = False
                exited = True
                self.exits += 1
                run = self.current_run
                if run is not None:
                    run["exit_decision"] = int(decision)
                    run["exit_x"] = float(x)
                    self.run_lengths.append(int(run["decisions"]))
                    self.runs.append(run)
                    self.current_run = None
                if self.first_exit is None:
                    self.first_exit = {"decision": int(decision),
                                       "engine_frame": engine_frame,
                                       "x": float(x)}
        elif x is not None and x >= self.config.guard_x \
                and move in self.config.guard_moves:
            self.bouncing = True
            entered = True
            self.entries += 1
            self.current_run = {
                "run": int(self.entries - 1),
                "enter_decision": int(decision),
                "enter_frame": engine_frame,
                "enter_x": float(x),
                "enter_cand_in": int(cand),
                "decisions": 0,
                "x_min": float(x),
                "x_max": float(x),
                "exit_decision": None,
                "exit_x": None,
            }
            entry = {"decision": int(decision), "engine_frame": engine_frame,
                     "x": float(x), "cand_in": int(cand), "move": int(move)}
            if self.first_entry is None:
                self.first_entry = dict(entry)
            self.last_entry = dict(entry)

        in_bounce = bool(self.bouncing)
        out = self.config.excursion_cand if in_bounce else cand
        mask_in = gym_actions.cand_to_mask(cand, bomb)
        mask_out = gym_actions.cand_to_mask(out, bomb)

        if x is not None:
            self.x_values.append(x)
            bucket = str(int(round(x)))
            self.x_hist[bucket] = self.x_hist.get(bucket, 0) + 1
        self.moves_hist[str(move)] = self.moves_hist.get(str(move), 0) + 1
        self.cand_in_hist[str(cand)] = self.cand_in_hist.get(str(cand), 0) + 1
        self.cand_out_hist[str(out)] = self.cand_out_hist.get(str(out), 0) + 1
        if in_bounce:
            self.excursion_decisions += 1
            self.bounce_x_values.append(x)
            bucket = str(int(round(x)))
            self.bounce_x_hist[bucket] = self.bounce_x_hist.get(bucket, 0) + 1
            self.replaced_cand_hist[str(cand)] = \
                self.replaced_cand_hist.get(str(cand), 0) + 1
            self.replaced_move_hist[str(move)] = \
                self.replaced_move_hist.get(str(move), 0) + 1
            run = self.current_run
            if run is not None:
                run["decisions"] = int(run["decisions"]) + 1
                run["x_min"] = min(float(run["x_min"]), x)
                run["x_max"] = max(float(run["x_max"]), x)

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
                "in_bounce": in_bounce,
                "entered": entered,
                "exited": exited,
                "bounce_run": (int(self.current_run["run"])
                               if (in_bounce and self.current_run is not None) else None),
                "guard_x": float(self.config.guard_x),
                "release_x": float(self.config.release_x),
                "guard_moves": [int(m) for m in self.config.guard_moves],
                "excursion_cand": int(self.config.excursion_cand),
            })
        return out, bomb

    # -- report ----------------------------------------------------------
    def runs_snapshot(self) -> list[dict]:
        """Every bounce run, including one still open when the episode ended."""

        runs = [dict(run) for run in self.runs]
        if self.current_run is not None:
            runs.append(dict(self.current_run))
        return runs

    def summary(self) -> dict:
        lengths = list(self.run_lengths)
        if self.current_run is not None:
            lengths.append(int(self.current_run["decisions"]))
        return {
            "schema": GUARD_SUMMARY_SCHEMA,
            "controller": CONTROLLER,
            "config": self.config.as_dict(),
            "decisions": int(self.calls),
            "excursion_decisions": int(self.excursion_decisions),
            "bounce_entries": int(self.entries),
            "bounce_exits": int(self.exits),
            "bounce_open_at_end": bool(self.bouncing),
            "bounce_runs": len(lengths),
            "bounce_run_lengths": lengths,
            "bounce_dwell_mean": ((sum(lengths) / len(lengths)) if lengths else None),
            "bounce_dwell_min": min(lengths) if lengths else None,
            "bounce_dwell_max": max(lengths) if lengths else None,
            "bounce_dwell_total": sum(lengths),
            "bounce_run_records": self.runs_snapshot(),
            "player_reads": int(self.player_reads),
            "read_errors": int(self.read_errors),
            "x_unavailable": int(self.x_unavailable),
            "x_min": min(self.x_values) if self.x_values else None,
            "x_max": max(self.x_values) if self.x_values else None,
            "x_p50": _percentile(self.x_values, 0.5),
            "x_p90": _percentile(self.x_values, 0.9),
            "x_at_or_above_guard": int(sum(1 for x in self.x_values
                                           if x >= self.config.guard_x)),
            "x_at_or_below_release": int(sum(1 for x in self.x_values
                                             if x <= self.config.release_x)),
            "x_hist": dict(sorted(self.x_hist.items(), key=lambda kv: int(kv[0]))),
            "bounce_x_min": (min(self.bounce_x_values)
                             if self.bounce_x_values else None),
            "bounce_x_max": (max(self.bounce_x_values)
                             if self.bounce_x_values else None),
            "bounce_x_hist": dict(sorted(self.bounce_x_hist.items(),
                                         key=lambda kv: int(kv[0]))),
            "replaced_cand_hist": dict(sorted(self.replaced_cand_hist.items(),
                                              key=lambda kv: int(kv[0]))),
            "replaced_move_hist": dict(sorted(self.replaced_move_hist.items(),
                                              key=lambda kv: int(kv[0]))),
            "first_entry": self.first_entry,
            "last_entry": self.last_entry,
            "first_exit": self.first_exit,
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
        description="V832-S2T bounce ruler: the pinned v400_student_eval run loop "
                    "with a hysteresis right-wall bounce guard inserted at action "
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
                        help="one JSONL row per decision with the pre-action x, the "
                             "bounce state and the overridden argmax candidate "
                             "(schema %s); mandatory while the guard is enabled"
                             % GUARD_SCHEMA)
    parser.add_argument("--guard-x", type=float, default=DEFAULT_GUARD_X,
                        help="arm the bounce state when x >= this (PLAYER.px) and the "
                             "argmax move is guarded.  %g or more disables the guard "
                             "and every extra engine command" % GUARD_DISABLE_X)
    parser.add_argument("--release-x", type=float, default=DEFAULT_RELEASE_X,
                        help="disarm the bounce state when x <= this; must be "
                             "strictly below --guard-x (default 340 vs 370)")
    parser.add_argument("--guard-moves", default=DEFAULT_GUARD_MOVES,
                        help="comma list of `move` ids that carry +x: default 2,3,4 "
                             "= NE/E/SE")
    parser.add_argument("--excursion-cand", type=int, default=DEFAULT_EXCURSION_CAND,
                        help="candidate emitted on every decision while the bounce "
                             "state is armed; 7 = W (default)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = resolve_guard(guard_x=args.guard_x, release_x=args.release_x,
                               guard_moves=args.guard_moves,
                               excursion_cand=args.excursion_cand)
    except ValueError as exc:
        parser.error(str(exc))
    if config.enabled and not (args.guard_log or "").strip():
        parser.error("--guard-log is required while the guard is enabled: a run "
                     "whose bounce telemetry is missing is not a measured run")

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
                shm_name = (f"s2t-{os.getpid()}-{seed}-"
                            f"{int(time.time() * 1000) % 100000}")
                holder: dict = {}
                guarded = BounceGuardModel(model, seed=seed, config=config,
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
                record["bounce"] = guarded.summary()
                record["bounce"]["client_instances"] = int(holder.get("instances", 0))
                records.append(record)
                print(json.dumps({
                    "seed": record["seed"],
                    "frames": record["frames"],
                    "deaths": record["deaths"],
                    "end": record["end"],
                    "score": record["score"],
                    "bounce_entries": record["bounce"]["bounce_entries"],
                    "bounce_exits": record["bounce"]["bounce_exits"],
                    "excursion_decisions": record["bounce"]["excursion_decisions"],
                    "bounce_runs": record["bounce"]["bounce_runs"],
                    "bounce_dwell_mean": record["bounce"]["bounce_dwell_mean"],
                    "x_min": record["bounce"]["x_min"],
                    "x_max": record["bounce"]["x_max"],
                    "wall_s": record["wall_s"],
                }, ensure_ascii=False), flush=True)
                (out_dir / f"bounce-{seed}.json").write_text(
                    json.dumps(record["bounce"], indent=2, ensure_ascii=False) + "\n",
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
        "the pinned ruler's run loop with a hysteresis right-wall bounce guard at "
        "action selection; with --guard-x >= %g no command and no action differs "
        "from %s" % (GUARD_DISABLE_X, PINNED_CONTROLLER))
    _lengths = [int(n) for record in records
                for n in record["bounce"]["bounce_run_lengths"]]
    summary["bounce"] = {
        "config": config.as_dict(),
        "bounce_entries_total": int(sum(r["bounce"]["bounce_entries"]
                                        for r in records)),
        "bounce_exits_total": int(sum(r["bounce"]["bounce_exits"]
                                      for r in records)),
        "excursion_decisions_total": int(sum(r["bounce"]["excursion_decisions"]
                                             for r in records)),
        "bounce_runs_total": len(_lengths),
        "bounce_dwell_mean": ((sum(_lengths) / len(_lengths)) if _lengths else None),
        "bounce_dwell_min": min(_lengths) if _lengths else None,
        "bounce_dwell_max": max(_lengths) if _lengths else None,
        "per_seed": [
            {
                "seed": record["seed"],
                "frames": record["frames"],
                "deaths": record["deaths"],
                "end": record["end"],
                "decisions": record["bounce"]["decisions"],
                "bounce_entries": record["bounce"]["bounce_entries"],
                "bounce_exits": record["bounce"]["bounce_exits"],
                "bounce_open_at_end": record["bounce"]["bounce_open_at_end"],
                "excursion_decisions": record["bounce"]["excursion_decisions"],
                "bounce_runs": record["bounce"]["bounce_runs"],
                "bounce_dwell_mean": record["bounce"]["bounce_dwell_mean"],
                "bounce_dwell_max": record["bounce"]["bounce_dwell_max"],
                "x_min": record["bounce"]["x_min"],
                "x_max": record["bounce"]["x_max"],
                "x_hist": record["bounce"]["x_hist"],
                "bounce_x_hist": record["bounce"]["bounce_x_hist"],
                "replaced_cand_hist": record["bounce"]["replaced_cand_hist"],
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
        "bounce": summary["bounce"]["config"],
        "bounce_entries_total": summary["bounce"]["bounce_entries_total"],
        "bounce_exits_total": summary["bounce"]["bounce_exits_total"],
        "excursion_decisions_total":
            summary["bounce"]["excursion_decisions_total"],
        "frames": [record["frames"] for record in records],
    }, ensure_ascii=False), flush=True)
    failed = [record for record in records if record["end"] == "error"]
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
