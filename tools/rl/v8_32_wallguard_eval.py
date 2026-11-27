#!/usr/bin/env python3
"""V832-S2S: the wall-guard falsification run for the DAgger student D2.

The mechanism under test (established offline, taken as the premise)
---------------------------------------------------------------------
``ckpt_d2.pt`` locks candidate 3 (``focus 0`` + ``move 3`` = E / ``BTN_RIGHT``)
after frame 600 on 15/15 seeds, pushes itself into the right wall
``x = MOVE_AREA_X + MOVE_AREA_W = 376.0`` (``touhou_rl/spec.py:366-369``) where a
rightward input moves the ship by exactly 0 px, and then dies 206-376 frames
after the frame-662 bomb invulnerability ends.

Falsifiable prediction
----------------------
At inference time, if ``x >= --guard-x`` (default 370) **and** the argmax action
carries a +x component (``move in --guard-moves``, default ``{2,3,4}`` = NE/E/SE),
replace it with ``--substitute`` (default candidate 0 = idle).  Prediction: the
natural opening lasts ``> 1,200`` frames on ``>= 10/15`` seeds.  If it still lands
near ~880, the mechanism is falsified.

What this controller is
-----------------------
It is the pinned ruler ``tools/rl/v400_student_eval.py`` with **one** change: the
action-selection point.  Everything else -- checkpoint loading, the obs-v3 planes,
the 2-frame decision budget, the 4-frame stack, the frame cap, the stop-on-death
rule, the ``--decision-log`` schema -- is imported from that file (by path, not
copied), so a record produced here is the same record class the pinned tool
produces.  The pinned file itself is never modified; its sha256 is recorded in
every episode record.

Two hard invariants:

1. **Guard off == pinned tool, bit for bit.**  ``--guard-x 99999`` disables the
   guard *and* every extra engine command: the wrapper then returns the model's
   own ``(cand, bomb)`` untouched, reads no ``PLAYER`` reply and writes no guard
   row.  The run's ``decisions.jsonl`` must hash-match the pinned run's file.
2. **Guard on adds exactly one read-only command per decision.**  The player x is
   read with ``PLAYER`` (``touhou_rl/v400/gym/protocol.py::parse_player_reply``,
   field ``px``, the same movement-area coordinate ``MOVE_AREA_*`` describe -- the
   field name and unit are the protocol's, not assumed here) *before* the action is
   emitted, so ``x`` is the position the decision is actually made at.  Every
   decision's ``x`` is written to ``--guard-log`` (JSONL) so the trigger count can
   be recomputed from raw bytes.  A failed read is recorded, never forged: the last
   known ``x`` is carried forward and marked ``x_source="carry"``; with no prior
   reading the decision is marked ``x_source="unavailable"`` and does not fire.

Run inside WSL::

    xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
      tools/rl/v8_32_wallguard_eval.py \
      --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
      --data-dir .tmp_v832_s2s/game-muted \
      --checkpoint .tmp_v832_s2h/ckpt_d2.pt \
      --seeds 1000 --stage 0 --difficulty normal --max-frames 4500 \
      --stop-on-death --guard-x 99999 \
      --out-dir .tmp_v832_s2s/EQ/1000 \
      --decision-log .tmp_v832_s2s/EQ/1000/decisions.jsonl \
      --guard-log .tmp_v832_s2s/EQ/1000/guard.jsonl
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

CONTROLLER = "tools/rl/v8_32_wallguard_eval.py"
PINNED_CONTROLLER = "tools/rl/v400_student_eval.py"
GUARD_SCHEMA = "v832-s2s-wallguard-decision-1"
GUARD_SUMMARY_SCHEMA = "v832-s2s-wallguard-run-1"
#: ``--guard-x`` at or above this value disables the guard (and every extra
#: engine command).  It is the equivalence switch: a disabled run must be
#: bit-identical to the pinned ruler.
GUARD_DISABLE_X = 99999.0
DEFAULT_GUARD_X = 370.0
DEFAULT_GUARD_MOVES = "2,3,4"
DEFAULT_SUBSTITUTE = 0


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
    """The three parameters the prediction names, plus the derived switch."""

    guard_x: float
    guard_moves: tuple[int, ...]
    substitute: int
    enabled: bool

    def as_dict(self) -> dict:
        return {
            "guard_x": float(self.guard_x),
            "guard_moves": [int(move) for move in self.guard_moves],
            "substitute": int(self.substitute),
            "enabled": bool(self.enabled),
            "disable_x": float(GUARD_DISABLE_X),
            "rule": ("if x >= guard_x and (cand % 9) in guard_moves then emit "
                     "substitute; x is read with PLAYER.px before the action"),
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


def resolve_guard(*, guard_x: float, guard_moves: str,
                  substitute: int) -> GuardConfig:
    """Validate the guard parameters before any file or engine exists."""

    try:
        x = float(guard_x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--guard-x {guard_x!r} is not a number") from exc
    moves = parse_guard_moves(guard_moves)
    try:
        cand = int(substitute)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--substitute {substitute!r} is not an integer") from exc
    if not 0 <= cand < gym_actions.N_CAND:
        raise ValueError(f"--substitute {cand} is outside 0..{gym_actions.N_CAND - 1}")
    if cand % gym_actions.MOVES_PER_CAND in moves:
        raise ValueError(
            f"--substitute {cand} itself carries a guarded +x move "
            f"({cand % gym_actions.MOVES_PER_CAND}); it would not change anything")
    return GuardConfig(guard_x=x, guard_moves=moves, substitute=cand,
                       enabled=x < GUARD_DISABLE_X)


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


class WallGuardModel:
    """Wrap the student so only the emitted candidate can be changed.

    The wrapped model is called exactly as :func:`run_episode` would call it, so
    the policy's inputs (planes, HUD strip, action history) are untouched; only the
    candidate that reaches ``cand_to_mask`` may be replaced.  ``bomb`` is never
    changed.
    """

    def __init__(self, model, *, seed: int, config: GuardConfig,
                 client_ref, log=None) -> None:
        self.model = model
        self.seed = int(seed)
        self.config = config
        self.client_ref = client_ref
        self.log = log
        self.calls = 0
        self.guard_fires = 0
        self.player_reads = 0
        self.read_errors = 0
        self.x_unavailable = 0
        self.last_x: float | None = None
        self.x_values: list[float] = []
        self.fired_x_values: list[float] = []
        self.x_hist: dict[str, int] = {}
        self.fired_x_hist: dict[str, int] = {}
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
        fired = bool(x is not None and x >= self.config.guard_x
                     and move in self.config.guard_moves)
        out = self.config.substitute if fired else cand
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
            self.guard_fires += 1
            self.fired_x_values.append(x)
            bucket = str(int(round(x)))
            self.fired_x_hist[bucket] = self.fired_x_hist.get(bucket, 0) + 1
            entry = {
                "decision": int(decision),
                "engine_frame": engine_frame,
                "x": x,
                "cand_in": int(cand),
                "cand_out": int(out),
            }
            if self.first_fire is None:
                self.first_fire = entry
            self.last_fire = entry
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
                "guard_fired": fired,
                "guard_x": float(self.config.guard_x),
                "guard_moves": [int(m) for m in self.config.guard_moves],
                "substitute": int(self.config.substitute),
            })
        return out, bomb

    # -- report ----------------------------------------------------------
    def summary(self) -> dict:
        return {
            "schema": GUARD_SUMMARY_SCHEMA,
            "controller": CONTROLLER,
            "config": self.config.as_dict(),
            "decisions": int(self.calls),
            "guard_fires": int(self.guard_fires),
            "player_reads": int(self.player_reads),
            "read_errors": int(self.read_errors),
            "x_unavailable": int(self.x_unavailable),
            "x_min": min(self.x_values) if self.x_values else None,
            "x_max": max(self.x_values) if self.x_values else None,
            "x_p50": _percentile(self.x_values, 0.5),
            "x_p90": _percentile(self.x_values, 0.9),
            "x_at_or_above_guard": int(sum(1 for x in self.x_values
                                           if x >= self.config.guard_x)),
            "x_hist": dict(sorted(self.x_hist.items(), key=lambda kv: int(kv[0]))),
            "fired_x_min": min(self.fired_x_values) if self.fired_x_values else None,
            "fired_x_max": max(self.fired_x_values) if self.fired_x_values else None,
            "fired_x_hist": dict(sorted(self.fired_x_hist.items(),
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
        description="V832-S2S wall-guard ruler: the pinned v400_student_eval run "
                    "loop with a right-wall guard inserted at action selection.")
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
                        help="one JSONL row per decision with the pre-action x "
                             "(schema %s); mandatory while the guard is enabled"
                             % GUARD_SCHEMA)
    parser.add_argument("--guard-x", type=float, default=DEFAULT_GUARD_X,
                        help="fire when x >= this (PLAYER.px).  %g or more disables "
                             "the guard and every extra engine command"
                             % GUARD_DISABLE_X)
    parser.add_argument("--guard-moves", default=DEFAULT_GUARD_MOVES,
                        help="comma list of `move` ids that carry +x: default 2,3,4 "
                             "= NE/E/SE")
    parser.add_argument("--substitute", type=int, default=DEFAULT_SUBSTITUTE,
                        help="candidate emitted instead of a guarded one; 0 = idle")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = resolve_guard(guard_x=args.guard_x, guard_moves=args.guard_moves,
                               substitute=args.substitute)
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
                shm_name = (f"s2s-{os.getpid()}-{seed}-"
                            f"{int(time.time() * 1000) % 100000}")
                holder: dict = {}
                guarded = WallGuardModel(model, seed=seed, config=config,
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
                record["wallguard"] = guarded.summary()
                record["wallguard"]["client_instances"] = int(holder.get("instances", 0))
                records.append(record)
                print(json.dumps({
                    "seed": record["seed"],
                    "frames": record["frames"],
                    "deaths": record["deaths"],
                    "end": record["end"],
                    "score": record["score"],
                    "guard_fires": record["wallguard"]["guard_fires"],
                    "guard_decisions": record["wallguard"]["decisions"],
                    "x_max": record["wallguard"]["x_max"],
                    "wall_s": record["wall_s"],
                }, ensure_ascii=False), flush=True)
                (out_dir / f"wallguard-{seed}.json").write_text(
                    json.dumps(record["wallguard"], indent=2, ensure_ascii=False) + "\n",
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
        "the pinned ruler's run loop with a right-wall guard at action selection; "
        "with --guard-x >= %g no command and no action differs from %s"
        % (GUARD_DISABLE_X, PINNED_CONTROLLER))
    summary["wallguard"] = {
        "config": config.as_dict(),
        "guard_fires_total": int(sum(r["wallguard"]["guard_fires"] for r in records)),
        "per_seed": [
            {
                "seed": record["seed"],
                "frames": record["frames"],
                "deaths": record["deaths"],
                "end": record["end"],
                "guard_fires": record["wallguard"]["guard_fires"],
                "decisions": record["wallguard"]["decisions"],
                "x_min": record["wallguard"]["x_min"],
                "x_max": record["wallguard"]["x_max"],
                "x_hist": record["wallguard"]["x_hist"],
                "fired_x_hist": record["wallguard"]["fired_x_hist"],
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
        "wallguard": summary["wallguard"]["config"],
        "guard_fires_total": summary["wallguard"]["guard_fires_total"],
        "frames": [record["frames"] for record in records],
    }, ensure_ascii=False), flush=True)
    failed = [record for record in records if record["end"] == "error"]
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
