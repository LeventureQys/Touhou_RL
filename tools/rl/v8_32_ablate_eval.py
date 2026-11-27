#!/usr/bin/env python3
"""V832-S2U: the observation-channel ablation ruler for the pure-T04 student.

Pinned tool, one changed input
-----------------------------
``tools/rl/v400_student_eval.py`` is the pinned ruler and is **never modified**.
This file imports it by path (its ``main`` is behind an ``if __name__`` guard, so
importing cannot start an engine) and calls its :func:`run_episode`, which keeps
the checkpoint loading, the obs-v3 planes, the 4-frame stack, the 2-frame decision
budget, the frame cap, the stop-on-death rule and the ``--decision-log`` schema
exactly where they were.  The only change is at the point where the observation is
handed to the policy:

``--ablate``
    ``none``     pass every tensor through untouched (the equivalence switch),
    ``play``     replace the ``(1,4,112,96)`` play stack with zeros,
    ``hud``      replace the ``(1,56,160)`` HUD strip with zeros,
    ``act_hist`` replace the ``(1,8,18)`` one-hot action history with zeros.

Zeroing is done with ``torch.zeros_like`` on the tensor that reaches
``model.act`` -- no buffer, no history row and no engine command is touched, so
all 16 launches see the identical simulation and differ only in the policy input.
``act_hist`` all-zeros is not an out-of-distribution input: it is the obs-v3
contract's own "no recorded action yet" padding (``obs_spec_v3.arrays()`` without
a history), which every episode's first decisions carry.

Two hard invariants:

1. **``--ablate none`` == the pinned ruler.**  In that mode the wrapper calls
   ``model.act`` with the arguments it received, unmodified, and reads no tensor;
   a record produced here is then byte-comparable with a pinned record.
2. **Every ablation is evidenced, never assumed.**  With ``--ablate-log`` one row
   per decision records the switch state (``ablate``, which plane was replaced,
   ``applied``) and the plane's absolute sums before/after, so
   ``sum_after == 0`` and ``other planes unchanged`` can be recomputed from raw
   bytes.

Run inside WSL::

    xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
      tools/rl/v8_32_ablate_eval.py \
      --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
      --data-dir .tmp_v832_s2u/game-muted \
      --checkpoint .tmp_v832_s2m/ckpt_s1.pt \
      --seeds 1000 --stage 0 --difficulty normal --max-frames 4500 \
      --stop-on-death --ablate act_hist \
      --out-dir .tmp_v832_s2u/A/1000 \
      --decision-log .tmp_v832_s2u/A/1000/decisions.jsonl \
      --ablate-log .tmp_v832_s2u/A/1000/ablate.jsonl
"""

from __future__ import annotations

import argparse
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
from touhou_rl.v400.student import obs_spec  # noqa: E402
from touhou_rl.v400.student import obs_spec_v3  # noqa: E402

CONTROLLER = "tools/rl/v8_32_ablate_eval.py"
PINNED_CONTROLLER = "tools/rl/v400_student_eval.py"
ABLATE_SCHEMA = "v832-s2u-ablate-decision-1"
ABLATE_SUMMARY_SCHEMA = "v832-s2u-ablate-run-1"
#: The four switch positions.  ``none`` is the equivalence control.
ABLATIONS = ("none", "act_hist", "play", "hud")
#: Position of each plane in ``model.act(play, hud, act_hist)``.
PLANE_ARGUMENT = {"play": 0, "hud": 1, "act_hist": 2}
#: The shape ``run_episode`` must hand the policy for the obs-v3 route, taken from
#: the spec itself rather than restated here.  ``hud`` keeps both leading axes:
#: ``ObservationBufferV3.arrays()`` returns ``hud (1,56,160)`` (one HUD *frame*)
#: and ``run_episode`` adds the batch axis, so the tensor is ``(1,1,56,160)``.
EXPECTED_SHAPES = {
    "play": (1, obs_spec.FRAME_STACK, *obs_spec_v3.PLAY_SHAPE),
    "hud": (1, obs_spec.HUD_FRAMES, *obs_spec_v3.HUD_SHAPE),
    "act_hist": (1, *obs_spec_v3.ACTION_HISTORY_SHAPE),
}
#: The same three shapes written out independently of the derivation above, so a
#: wrong derivation cannot hide behind itself; ``--selftest`` compares the two.
DECLARED_SHAPES = {
    "play": (1, 4, 112, 96),
    "hud": (1, 1, 56, 160),
    "act_hist": (1, 8, 18),
}


# ---------------------------------------------------------------------------
# Pinned ruler, imported by path
# ---------------------------------------------------------------------------
def pinned_path() -> Path:
    """The pinned evaluation controller this tool extends (never modified)."""

    return Path(__file__).resolve().parent / PINNED_CONTROLLER.split("/")[-1]


def load_pinned(path: str | Path | None = None):
    """Import ``v400_student_eval.py`` as a module, once."""

    target = Path(path) if path is not None else pinned_path()
    name = "v400_student_eval_pinned_s2u"
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
# The ablation switch
# ---------------------------------------------------------------------------
def resolve_ablate(text: str | None) -> str:
    """``None`` -> ``"none"``; anything outside :data:`ABLATIONS` is refused."""

    name = str("none" if text is None else text)
    if name not in ABLATIONS:
        raise ValueError(f"--ablate {name!r} not in {list(ABLATIONS)}")
    return name


def _abs_sum(tensor) -> float:
    return float(tensor.abs().sum().item())


class AblationModel:
    """Wrap the student so only the observation handed to it can change.

    ``act`` is called exactly as :func:`v400_student_eval.run_episode` calls the
    model (``(play, hud, act_hist)`` for the obs-v3 route), so the buffers, the
    action history and the engine traffic are untouched.  With ``ablate="none"``
    the received arguments are forwarded as they are and no tensor is read.
    """

    def __init__(self, model, *, seed: int, ablate: str, log=None) -> None:
        self.model = model
        self.seed = int(seed)
        self.ablate = ablate
        self.log = log
        self.calls = 0
        self.ablated_calls = 0
        self.shape_checked = False
        self.plane_in_abs: dict[str, float] = {}
        self.plane_out_abs: dict[str, float] = {}
        self.plane_out_max: dict[str, float] = {}
        self.plane_in_max: dict[str, float] = {}
        self.other_changed = 0
        self.target_nonzero_in = 0
        self.first_row: dict | None = None
        self.last_row: dict | None = None

    # -- shape contract ---------------------------------------------------
    def _check_shapes(self, args: tuple) -> None:
        """Refuse an observation the ablation does not describe (fail closed)."""

        if len(args) < 3:
            raise RuntimeError(
                f"the obs-v3 route must call act(play, hud, act_hist); got "
                f"{len(args)} positional argument(s)")
        for name, index in PLANE_ARGUMENT.items():
            shape = tuple(int(value) for value in args[index].shape)
            if shape != EXPECTED_SHAPES[name]:
                raise RuntimeError(
                    f"{name} tensor shape {shape} != {EXPECTED_SHAPES[name]}: the "
                    "ablation switch does not describe this observation")
        self.shape_checked = True

    # -- act --------------------------------------------------------------
    def act(self, *args, **kwargs):
        if self.ablate == "none":
            # Equivalence mode: no copy, no zero, no tensor read.  The pinned
            # ruler's own call goes through unchanged.
            result = self.model.act(*args, **kwargs)
            if self.log is not None:
                self._write_row(decision=self.calls, applied=False,
                                sums_in={}, sums_out={}, maxima_in={}, maxima_out={})
            self.calls += 1
            return result

        if not self.shape_checked:
            self._check_shapes(args)
        index = PLANE_ARGUMENT[self.ablate]
        sums_in = {name: _abs_sum(args[i]) for name, i in PLANE_ARGUMENT.items()}
        maxima_in = {name: float(args[i].abs().max().item())
                     for name, i in PLANE_ARGUMENT.items()}

        import torch  # only the zeroing needs it; ``none`` never imports it here

        replaced = list(args)
        replaced[index] = torch.zeros_like(args[index])
        sums_out = {name: _abs_sum(replaced[i]) for name, i in PLANE_ARGUMENT.items()}
        maxima_out = {name: float(replaced[i].abs().max().item())
                      for name, i in PLANE_ARGUMENT.items()}

        unchanged = all(sums_in[name] == sums_out[name] for name in PLANE_ARGUMENT
                        if name != self.ablate)
        if not unchanged:
            self.other_changed += 1
        if sums_in[self.ablate] != 0.0:
            self.target_nonzero_in += 1
        for name in PLANE_ARGUMENT:
            self.plane_in_abs[name] = self.plane_in_abs.get(name, 0.0) + sums_in[name]
            self.plane_out_abs[name] = self.plane_out_abs.get(name, 0.0) + sums_out[name]
            self.plane_in_max[name] = max(self.plane_in_max.get(name, 0.0),
                                          maxima_in[name])
            self.plane_out_max[name] = max(self.plane_out_max.get(name, 0.0),
                                           maxima_out[name])
        if sums_out[self.ablate] != 0.0:
            raise RuntimeError(f"{self.ablate} plane is not zero after ablation: "
                               f"abs sum {sums_out[self.ablate]}")
        self.ablated_calls += 1

        result = self.model.act(*replaced, **kwargs)
        if self.log is not None:
            self._write_row(decision=self.calls, applied=True, sums_in=sums_in,
                            sums_out=sums_out, maxima_in=maxima_in,
                            maxima_out=maxima_out, unchanged_others=unchanged)
        self.calls += 1
        return result

    # -- telemetry --------------------------------------------------------
    def _write_row(self, *, decision: int, applied: bool, sums_in: dict,
                   sums_out: dict, maxima_in: dict, maxima_out: dict,
                   unchanged_others: bool | None = None) -> None:
        row = {
            "schema": ABLATE_SCHEMA,
            "seed": self.seed,
            "decision": int(decision),
            "ablate": self.ablate,
            "applied": bool(applied),
            "zeroed_plane": self.ablate if applied else None,
            "zeroed_argument_index": (PLANE_ARGUMENT[self.ablate] if applied else None),
            "expected_shapes": {name: list(shape)
                                for name, shape in EXPECTED_SHAPES.items()},
            "abs_sum_in": {name: sums_in.get(name) for name in PLANE_ARGUMENT},
            "abs_sum_out": {name: sums_out.get(name) for name in PLANE_ARGUMENT},
            "abs_max_in": {name: maxima_in.get(name) for name in PLANE_ARGUMENT},
            "abs_max_out": {name: maxima_out.get(name) for name in PLANE_ARGUMENT},
            "others_unchanged": unchanged_others,
        }
        if self.first_row is None:
            self.first_row = row
        self.last_row = row
        self.log.write(row)

    # -- report -----------------------------------------------------------
    def summary(self) -> dict:
        return {
            "schema": ABLATE_SUMMARY_SCHEMA,
            "controller": CONTROLLER,
            "ablate": self.ablate,
            "enabled": self.ablate != "none",
            "decisions": int(self.calls),
            "ablated_decisions": int(self.ablated_calls),
            "shape_checked": bool(self.shape_checked),
            "expected_shapes": {name: list(shape)
                                for name, shape in EXPECTED_SHAPES.items()},
            "target_nonzero_before": int(self.target_nonzero_in),
            "others_changed": int(self.other_changed),
            "plane_abs_sum_in": dict(self.plane_in_abs),
            "plane_abs_sum_out": dict(self.plane_out_abs),
            "plane_abs_max_in": dict(self.plane_in_max),
            "plane_abs_max_out": dict(self.plane_out_max),
            "first_decision": self.first_row,
            "last_decision": self.last_row,
        }


# ---------------------------------------------------------------------------
# Self-test: no engine, no checkpoint, no launch budget
# ---------------------------------------------------------------------------
def selftest() -> int:
    """Prove on synthetic tensors that exactly the named plane is zeroed."""

    import torch

    class _Stub:
        def __init__(self) -> None:
            self.seen: list[tuple] = []

        def act(self, play, hud, act_hist):
            self.seen.append((play.clone(), hud.clone(), act_hist.clone()))
            return 3, 0

    play = torch.full((1, *EXPECTED_SHAPES["play"][1:]), 0.5, dtype=torch.float32)
    hud = torch.full((1, *EXPECTED_SHAPES["hud"][1:]), 0.25, dtype=torch.float32)
    hist = torch.zeros((1, *EXPECTED_SHAPES["act_hist"][1:]), dtype=torch.float32)
    hist[0, 0, 2] = 1.0
    payload = {"declared_shapes": {
        name: {"derived": list(EXPECTED_SHAPES[name]),
               "declared": list(DECLARED_SHAPES[name]),
               "ok": tuple(EXPECTED_SHAPES[name]) == tuple(DECLARED_SHAPES[name])}
        for name in PLANE_ARGUMENT}}
    receipt = {"play": tuple(play.shape), "hud": tuple(hud.shape),
               "hist": tuple(hist.shape)}
    payload["declared_shapes_ok"] = {
        "ok": all(block["ok"] for block in payload["declared_shapes"].values()),
        "observed_arguments": {name: list(shape) for name, shape in receipt.items()},
    }
    for name in ABLATIONS:
        stub = _Stub()
        wrapper = AblationModel(stub, seed=1000, ablate=name, log=None)
        wrapper.act(play, hud, hist)
        seen_play, seen_hud, seen_hist = stub.seen[-1]
        record = {
            "play_abs": float(seen_play.abs().sum().item()),
            "hud_abs": float(seen_hud.abs().sum().item()),
            "hist_abs": float(seen_hist.abs().sum().item()),
            "play_identical": bool(torch.equal(seen_play, play)),
            "hud_identical": bool(torch.equal(seen_hud, hud)),
            "hist_identical": bool(torch.equal(seen_hist, hist)),
        }
        record["shapes_ok"] = (tuple(seen_play.shape) == DECLARED_SHAPES["play"]
                               and tuple(seen_hud.shape) == DECLARED_SHAPES["hud"]
                               and tuple(seen_hist.shape) == DECLARED_SHAPES["act_hist"])
        record["ok"] = (
            record["shapes_ok"]
            and (record["play_identical"] if name != "play" else record["play_abs"] == 0.0)
            and (record["hud_identical"] if name != "hud" else record["hud_abs"] == 0.0)
            and (record["hist_identical"] if name != "act_hist"
                 else record["hist_abs"] == 0.0)
            and record["play_abs"] in (0.0, float(play.abs().sum()))
            and record["hud_abs"] in (0.0, float(hud.abs().sum()))
            and record["hist_abs"] in (0.0, float(hist.abs().sum())))
        payload[name] = record
    # The input tensors themselves must never be mutated by the wrapper.
    payload["inputs_untouched"] = {
        "play": float(play.abs().sum().item()),
        "hud": float(hud.abs().sum().item()),
        "hist": float(hist.abs().sum().item()),
    }
    payload["inputs_untouched"]["ok"] = (
        payload["inputs_untouched"]["play"] == float(play.numel()) * 0.5
        and payload["inputs_untouched"]["hud"] == float(hud.numel()) * 0.25
        and payload["inputs_untouched"]["hist"] == 1.0)
    payload["all_ok"] = (
        all(payload[name]["ok"] for name in ABLATIONS)
        and payload["declared_shapes_ok"]["ok"]
        and payload["inputs_untouched"]["ok"])
    print(json.dumps({"schema": "v832-s2u-selftest-1", "planes": payload},
                     indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if payload["all_ok"] else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V832-S2U observation-channel ablation ruler: the pinned "
                    "v400_student_eval run loop with one input plane zeroed.")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
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
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--decision-log", default=None,
                        help="pinned per-decision JSONL (schema "
                             "v400-student-eval-decision-1); with --ablate none it is "
                             "byte-comparable with a pinned run")
    parser.add_argument("--ablate", choices=list(ABLATIONS), default="none",
                        help="which observation plane is replaced by zeros before the "
                             "policy sees it; 'none' forwards the tensors untouched")
    parser.add_argument("--ablate-log", default=None,
                        help="one JSONL row per decision with the switch state and the "
                             "plane absolute sums before/after (schema %s); mandatory "
                             "while --ablate is not 'none'" % ABLATE_SCHEMA)
    parser.add_argument("--selftest", action="store_true",
                        help="prove the switch on synthetic tensors; starts no engine")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    for required in ("engine", "data_dir", "checkpoint", "out_dir"):
        if not getattr(args, required):
            parser.error(f"--{required.replace('_', '-')} is required")
    try:
        ablate = resolve_ablate(args.ablate)
    except ValueError as exc:
        parser.error(str(exc))
    if ablate != "none" and not (args.ablate_log or "").strip():
        parser.error("--ablate-log is required while an ablation is active: a run "
                     "whose switch telemetry is missing is not a measured run")
    if PINNED.route_obs(args) != "v3":
        raise SystemExit("this ruler only evaluates the obs-v3 / arch-v3 route "
                         "(the checkpoint under test is an obs-v3 student)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decision_log = None
    ablate_log = None
    try:
        decision_log = PINNED.open_decision_log(args.decision_log)
        ablate_log = PINNED.open_decision_log(args.ablate_log)
    except PINNED.EvalError as exc:
        parser.error(str(exc))

    try:
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
                shm_name = (f"s2u-{os.getpid()}-{seed}-"
                            f"{int(time.time() * 1000) % 100000}")
                ablated = AblationModel(model, seed=seed, ablate=ablate,
                                        log=ablate_log)
                record = PINNED.run_episode(
                    engine=args.engine,
                    data_dir=args.data_dir,
                    checkpoint_path=args.checkpoint,
                    model=ablated,
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
                record["ablation"] = ablated.summary()
                records.append(record)
                print(json.dumps({
                    "seed": record["seed"],
                    "frames": record["frames"],
                    "deaths": record["deaths"],
                    "end": record["end"],
                    "score": record["score"],
                    "ablate": ablate,
                    "ablated_decisions": record["ablation"]["ablated_decisions"],
                    "wall_s": record["wall_s"],
                }, ensure_ascii=False), flush=True)
                (out_dir / f"ablate-{seed}.json").write_text(
                    json.dumps(record["ablation"], indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")

        summary = PINNED.summarise(records, disclosure=disclosure,
                                   policy_note=policy_note,
                                   bootstrap_iters=int(args.bootstrap_iters),
                                   ci_level=float(args.ci_level))
    finally:
        if decision_log is not None:
            decision_log.close()
        if ablate_log is not None:
            ablate_log.close()

    summary["controller"] = CONTROLLER
    summary["controller_note"] = (
        "the pinned ruler's run loop with one observation plane zeroed before "
        "model.act; with --ablate none no tensor and no action differs from %s"
        % PINNED_CONTROLLER)
    summary["ablation"] = {
        "ablate": ablate,
        "enabled": ablate != "none",
        "zeroed_plane": None if ablate == "none" else ablate,
        "zeroed_argument_index": (None if ablate == "none"
                                  else PLANE_ARGUMENT[ablate]),
        "expected_shapes": {name: list(shape)
                            for name, shape in EXPECTED_SHAPES.items()},
        "note": ("the tensor handed to model.act is replaced by torch.zeros_like; the "
                 "observation buffers, the action history and the engine commands are "
                 "untouched, so every launch sees the same simulation"),
        "per_seed": [
            {
                "seed": record["seed"],
                "frames": record["frames"],
                "deaths": record["deaths"],
                "end": record["end"],
                "ablated_decisions": record["ablation"]["ablated_decisions"],
                "decisions": record["ablation"]["decisions"],
                "target_nonzero_before": record["ablation"]["target_nonzero_before"],
                "others_changed": record["ablation"]["others_changed"],
                "plane_abs_sum_in": record["ablation"]["plane_abs_sum_in"],
                "plane_abs_sum_out": record["ablation"]["plane_abs_sum_out"],
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
        "ablate": ablate,
        "frames": [record["frames"] for record in records],
        "ablated_decisions": [record["ablation"]["ablated_decisions"]
                              for record in records],
    }, ensure_ascii=False), flush=True)
    failed = [record for record in records if record["end"] == "error"]
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
