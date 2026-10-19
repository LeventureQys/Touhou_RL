"""Evaluate a checkpoint on the live game and write machine-readable evidence.

Each episode is a fresh menu start at the requested difficulty/team.  The report
records survival frames (the primary metric), stages reached, deaths, score and
graze, plus a screenshot and the full per-episode state trail so a reviewer can
reproduce the numbers without trusting prose.

The vision branch (``--vision-checkpoint``) is the deployment path: the policy is
fed nothing but per-region crops of ``env.render()``.  ``env.step()`` still reads
game memory to compute rewards and terminal flags, but no privileged value ever
reaches the model - see the isolation note in :data:`VISION_ISOLATION_NOTE`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import spec, vision
from .env import EnvConfig, Th08Env
from .models import load_checkpoint, observation_batch
from .pilot import AvoidPilot, PilotConfig
from .vmodels import VISION_ARCH_V3, load_vision_checkpoint, vision_arch, vision_contract_node

ROOT = Path(__file__).resolve().parents[1]

#: Codes of the shared ``v104:<CODE>: <detail>`` error format (C6).  The CLI turns
#: any of them into an stderr line with the same prefix and exit code 2; anything
#: else keeps its traceback and its own non-zero exit.
V104_CODES = frozenset(
    {"CONFIG", "DATA", "CACHE", "TEACHER", "CHECKPOINT", "COVERAGE", "BUDGET", "EVIDENCE"}
)
_V104_PREFIX = re.compile(r"^v104:([A-Z]+): ")


def v104_error_code(message: object) -> str | None:
    """``"CHECKPOINT"`` for ``"v104:CHECKPOINT: ..."``, else ``None``."""
    match = _V104_PREFIX.match(str(message))
    if match and match.group(1) in V104_CODES:
        return match.group(1)
    return None


def v104_error(code: str, detail: str) -> ValueError:
    """A contract violation in the shared ``v104:<CODE>: <detail>`` format."""
    return ValueError(f"v104:{code}: {detail}")


# Recorded in ``eval-report.json`` under ``vision.isolation`` so the deployment
# claim travels with the numbers instead of living only in a design document.
VISION_ISOLATION_NOTE = (
    "I-1: policy input is regions_to_batch(env.render crops) only; "
    "env bookkeeping still reads memory"
)


@dataclass
class EvalConfig:
    checkpoint: Path
    out_dir: Path
    episodes: int = 3
    difficulty: int = spec.LUNATIC
    team: int = 0
    practice_stage: int | None = None
    frameskip: int = 2
    max_steps: int = 12000
    deterministic: bool = False
    device: str = "cuda"
    screenshot: bool = True
    random_policy: bool = False
    reset_mode: str = "relaunch"
    run_dir: Path = ROOT / "third_party/th08/build/rl/eval-session"
    # Scripted avoidance teacher (no checkpoint needed); see touhou_rl/pilot.py.
    pilot: bool = False
    pilot_config: PilotConfig | None = None
    # Vision policy: the deployment path - frames only, no privileged reads.
    vision_checkpoint: Path | None = None
    # 0 = follow the checkpoint's ``config.depth``; a non-zero value must agree
    # with it (the checkpoint's first conv takes ``3 * depth`` channels).
    vision_depth: int = 0
    vision_box: vision.PlayfieldBox | None = None
    # Environment factory: ``None`` builds the real ``Th08Env`` (production).
    # Tests inject a stub so the deployment path can be exercised with no game.
    env_factory: Callable[[EnvConfig], object] | None = None


def resolve_vision_depth(model, requested: int) -> int:
    """Frame-stack depth to use, taken from the checkpoint (C6).

    ``0`` means "follow the checkpoint".  Any other value is only accepted when it
    agrees with ``model.config.depth``: the model's first convolution expects
    ``3 * depth`` input channels, so a mismatched stack cannot be fed to it at all
    (before v1.0.3 the default was 3, which crashed on the depth=1 pilot model
    with "expected input[1,9,224,192] to have 3 channels").  A negative value or a
    conflicting positive value is ``v104:CHECKPOINT``: the CLI prints the same
    prefix and exits 2 instead of the old exit 1, and this runs *before* the
    environment is built so a bad checkpoint never reaches the game.
    """
    depth = int(getattr(model.config, "depth", 0) or 0)
    if depth < 1:
        raise v104_error("CHECKPOINT", f"checkpoint depth {depth!r} is not a positive integer")
    requested = int(requested)
    if requested < 0:
        raise v104_error(
            "CHECKPOINT",
            f"--vision-depth {requested} is negative; use 0 to follow the checkpoint",
        )
    if requested not in (0, depth):
        raise v104_error("CHECKPOINT", f"--vision-depth {requested} != checkpoint depth {depth}")
    return depth


def vision_report_node(
    checkpoint, model, depth: int, max_steps: int | None = None, extra: dict | None = None
) -> dict:
    """The ``vision`` node of ``eval-report.json`` (contract C6).

    Everything a reviewer needs to audit a deployment run: which checkpoint, which
    architecture, which stack depth, the exact tensor shape each region becomes,
    the isolation statement the run is claiming, and the A-04 step cap the run was
    configured with (the rest of the report cannot express it, and A-04 must not
    infer ``hit_step_cap`` from anything but the per-episode reasons).  Only the v3
    branch adds a ``contract`` node: a v3 deployment has to record the
    normalisation/candidate-order/history contract it was trained under, while the
    v1/v2 node keeps its v1.0.3 key set.
    """
    arch = vision_arch(model)
    region_sizes = model.config.region_sizes
    node = {
        "checkpoint": str(checkpoint),
        "arch": arch,
        "depth": int(depth),
        "region_sizes": {name: [int(h), int(w)] for name, (h, w) in region_sizes.items()},
        "frame_shape": {
            name: [int(h), int(w), vision.VISION_CHANNELS] for name, (h, w) in region_sizes.items()
        },
        "isolation": VISION_ISOLATION_NOTE,
    }
    if max_steps is not None:
        node["max_steps"] = int(max_steps)
    if extra is not None and arch == VISION_ARCH_V3:
        node["contract"] = vision_contract_node(model, extra)
    return node


def build_pilot(cfg: EvalConfig) -> AvoidPilot | None:
    """Construct the scripted teacher, honouring the frameskip of the environment."""
    if not cfg.pilot:
        return None
    base = cfg.pilot_config or PilotConfig()
    base.frameskip = cfg.frameskip
    return AvoidPilot(base)


def _regions_from_screen(
    screen, region_sizes: dict[str, tuple[int, int]], box=None
) -> dict:
    """Native crops of every region, resized to the sizes the model expects."""
    native = vision.extract_regions(screen)
    out = {}
    for name, (height, width) in region_sizes.items():
        crop = native.get(name)
        if crop is None:  # region missing on this screen: feed zeros rather than fail
            out[name] = np.zeros((height, width, vision.VISION_CHANNELS), dtype=np.uint8)
        else:
            out[name] = vision.resize_area(crop, height, width)
    return out


# --------------------------------------------------------------------------- #
# deployment diagnostics (C6): latency + observed frame delta, evidence only
# --------------------------------------------------------------------------- #
#: Contract keys of ``vision_diagnostics.screenshot_to_action_ms`` /
#: ``.observed_frame_delta``.  ``count == 0`` means "no usable sample" and every
#: other statistic is ``None`` - never 0.0, which would read like a measurement.
TIMING_KEYS = ("count", "mean", "p50", "p95", "max")
DELTA_KEYS = ("count", "min", "p50", "p95", "max", "nonpositive")


def _finite_number(value) -> float | None:
    """``value`` as a finite float, or ``None`` (``None``/NaN/inf/bool/non-numeric)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def timing_summary(samples: list[float]) -> dict:
    """``{count,mean,p50,p95,max}`` of one episode's render->action samples (ms)."""
    values = [float(value) for value in samples if np.isfinite(value)]
    if not values:
        return {key: None for key in TIMING_KEYS} | {"count": 0}
    return {
        "count": len(values),
        "mean": round(float(np.mean(values)), 3),
        "p50": round(_percentile(values, 50), 3),
        "p95": round(_percentile(values, 95), 3),
        "max": round(float(max(values)), 3),
    }


def delta_summary(deltas: list[int], nonpositive: int) -> dict:
    """``{count,min,p50,p95,max,nonpositive}`` of same-stage frame deltas."""
    values = [int(value) for value in deltas]
    if not values:
        node = {key: None for key in DELTA_KEYS}
        node["count"] = 0
        node["nonpositive"] = int(nonpositive)
        return node
    return {
        "count": len(values),
        "min": int(min(values)),
        "p50": round(_percentile([float(value) for value in values], 50), 3),
        "p95": round(_percentile([float(value) for value in values], 95), 3),
        "max": int(max(values)),
        "nonpositive": int(nonpositive),
    }


def new_vision_diagnostics() -> dict:
    """Per-episode accumulator; raw samples are kept for the evidence file."""
    return {"timing_ms": [], "frame_delta": [], "nonpositive": 0, "bomb_actions_requested": 0}


def vision_diagnostics_node(accumulator: dict) -> dict:
    """The contracted per-episode ``vision_diagnostics`` node (exactly 3 keys)."""
    return {
        "bomb_actions_requested": int(accumulator.get("bomb_actions_requested", 0)),
        "screenshot_to_action_ms": timing_summary(accumulator.get("timing_ms", [])),
        "observed_frame_delta": delta_summary(
            accumulator.get("frame_delta", []), accumulator.get("nonpositive", 0)
        ),
    }


RAW_DIAGNOSTICS_NOTE = (
    "raw per-decision samples behind vision_diagnostics; latency is measured from the "
    "start of this decision's env.render() to the action handed to env.step(), and the "
    "frame delta only counts consecutive same-stage decisions with a positive delta "
    "(nonpositive ones are counted separately, never used as a statistic). Neither value "
    "is fed to the policy or used to reset history."
)


def _write_raw_diagnostics(out_dir: Path, episodes: list[dict]) -> None:
    """Save the unsmoothed samples so every p95 above can be recomputed."""
    payload = {
        "schema": "v104-vision-diagnostics-raw/1",
        "note": RAW_DIAGNOSTICS_NOTE,
        "episodes": episodes,
    }
    (out_dir / "vision-diagnostics-raw.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_evaluation(cfg: EvalConfig) -> dict:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    pilot = build_pilot(cfg)
    vision_model = None
    vision_extra: dict = {}
    vision_depth = int(cfg.vision_depth)
    if cfg.random_policy:
        model = None
        extra: dict = {"random_policy": True}
    elif cfg.vision_checkpoint is not None:
        model = None
        vision_model, vision_extra = load_vision_checkpoint(cfg.vision_checkpoint, map_location=device)
        vision_model.to(device).eval()
        # The stack depth has to come from the checkpoint, not from a CLI default:
        # resolve it before the environment starts so a mismatch fails fast.
        vision_depth = resolve_vision_depth(vision_model, cfg.vision_depth)
        extra = {"vision_checkpoint": True, **vision_extra}
    elif pilot is not None:
        model = None
        extra = {"pilot_config": pilot.config.__dict__ if hasattr(pilot.config, "__dict__") else {
            name: getattr(pilot.config, name) for name in pilot.config.__dataclass_fields__
        }}
    else:
        model, extra = load_checkpoint(cfg.checkpoint, map_location=device)
        model.to(device).eval()
    env = (cfg.env_factory or Th08Env)(
        EnvConfig(
            difficulty=cfg.difficulty,
            team=cfg.team,
            practice_stage=cfg.practice_stage,
            frameskip=cfg.frameskip,
            max_steps=cfg.max_steps,
            reset_mode=cfg.reset_mode,
            run_dir=cfg.run_dir,
            expose_raw_bullets=pilot is not None,
            expose_bosses=pilot is not None,
        )
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.random_policy:
        policy = "random"
    elif vision_model is not None:
        policy = "vision"
    elif pilot is not None:
        policy = "pilot"
    elif model is not None and model.__class__.__name__ == "ScorePolicy":
        policy = "scorer"
    else:
        policy = "network"
    report = {
        "checkpoint": str(cfg.checkpoint),
        "checkpoint_extra": extra,
        "policy": policy,
        "difficulty": cfg.difficulty,
        "team": cfg.team,
        "practice_stage": cfg.practice_stage,
        "frameskip": cfg.frameskip,
        "deterministic": cfg.deterministic,
        "device": str(device),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "episodes": [],
    }
    diagnostics_raw: list[dict] = []
    if vision_model is not None:
        # Deployment evidence (contract C6): only the vision branch gets this node,
        # every other policy's report keeps its existing shape.
        report["vision"] = vision_report_node(
            cfg.vision_checkpoint,
            vision_model,
            vision_depth,
            max_steps=cfg.max_steps,
            extra=vision_extra,
        )
    try:
        for episode in range(1, cfg.episodes + 1):
            obs, info = env.reset()
            started = time.perf_counter()
            steps = 0
            episode_info = {}
            clearances: list[float] = []
            frame_stack = None
            diagnostics = new_vision_diagnostics()
            previous_frame: tuple[int, int] | None = None
            if vision_model is not None:
                # Deployment path: the policy sees only pixels, as per-region
                # streams (playfield + HUD blocks).  Each stream is stacked so the
                # model gets motion without a recurrent state; the depth comes from
                # the checkpoint because the first conv expects 3 * depth channels.
                #
                # Isolation (I-1/I-2): ``vision_model.act`` below is handed nothing
                # but ``vision.regions_to_batch(stacks, device)``, and every stack
                # is built from ``env.render()``.  ``env.step`` itself reads game
                # memory to compute reward and terminal flags - that is the
                # evaluation harness, not the policy, and none of it is forwarded.
                region_sizes = vision_model.config.region_sizes
                frame_stack = {
                    name: vision.FrameStack(
                        depth=vision_depth, shape=(*size, vision.VISION_CHANNELS)
                    )
                    for name, size in region_sizes.items()
                }
                first = _regions_from_screen(env.render(), region_sizes, cfg.vision_box)
                for name, stack in frame_stack.items():
                    stack.reset()
                    for _ in range(vision_depth):
                        stack.push(first[name])
            # ``reset`` returns {"state": {...}} whereas ``step`` returns a flat
            # dict, so the bomb counter has to be read from both shapes.
            reset_state = info.get("state", {}) if isinstance(info, dict) else {}
            bombs_start = int(round(float(reset_state.get("bombs", info.get("bombs", 0)) or 0)))
            while True:
                if vision_model is not None:
                    # Diagnostics (C6) are measured, never fed back: the latency is
                    # timed from this decision's render to the action handed to
                    # ``env.step``, and neither it nor the frame delta can reach the
                    # policy or the history reset below.
                    decision_started = time.perf_counter()
                    regions = _regions_from_screen(env.render(), region_sizes, cfg.vision_box)
                    stacks = {name: frame_stack[name].push(regions[name]) for name in frame_stack}
                    with torch.no_grad():
                        action, _logp, value, _extra = vision_model.act(vision.regions_to_batch(stacks, device))
                    action_numpy = action[0].cpu().numpy()
                    diagnostics["timing_ms"].append(
                        (time.perf_counter() - decision_started) * 1000.0
                    )
                    # Bomb requests are counted from the action that actually reaches
                    # the environment; the current contract is zero (never a rule).
                    diagnostics["bomb_actions_requested"] += int(action_numpy[3])
                    obs, reward, terminated, truncated, info = env.step(action_numpy)
                    stage_now = _finite_number(info.get("stage"))
                    frame_now = _finite_number(info.get("frame"))
                    if stage_now is None or frame_now is None:
                        previous_frame = None  # untrusted bookkeeping breaks the interval
                    else:
                        if previous_frame is not None and previous_frame[0] == int(stage_now):
                            delta = int(frame_now) - previous_frame[1]
                            if delta > 0:
                                diagnostics["frame_delta"].append(delta)
                            else:
                                diagnostics["nonpositive"] += 1
                        previous_frame = (int(stage_now), int(frame_now))
                elif model is None:
                    if pilot is None:
                        action = env.action_space.sample()
                    else:
                        action = pilot.act(
                            obs,
                            info.get("bullets_raw"),
                            bombs=int(round(float(info.get("bombs", 0)))),
                            bosses=info.get("bosses"),
                        )
                        clearances.append(pilot.debug.best_clearance)
                    obs, reward, terminated, truncated, info = env.step(action)
                else:
                    batch = observation_batch(obs, device)
                    with torch.no_grad():
                        action, logp, value, _ = model.act(batch, deterministic=cfg.deterministic)
                    obs, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
                steps += 1
                if cfg.screenshot and steps == 60:
                    try:
                        from PIL import Image

                        Image.fromarray(env.render()).save(cfg.out_dir / f"episode-{episode:02d}-mid.png")
                    except Exception:
                        pass
                if terminated or truncated:
                    episode_info = info.get("episode", {})
                    break
            frames = int(episode_info.get("frames", 0))
            elapsed = time.perf_counter() - started
            row = {
                "episode": episode,
                "steps": steps,
                "frames_reported": frames,
                "stages_reached": int(info.get("stages_reached", 0)),
                "deaths": int(episode_info.get("deaths", 0)),
                "lives_left": float(episode_info.get("lives", 0.0)),
                "score": int(episode_info.get("score", 0)),
                "graze": int(episode_info.get("graze", 0)),
                "episode_reward": round(float(episode_info.get("episode_reward", 0.0)), 3),
                "terminated": bool(episode_info.get("terminated", False)),
                "truncated": bool(episode_info.get("truncated", False)),
                "termination_reason": episode_info.get("termination_reason", ""),
                "wall_seconds": round(elapsed, 1),
                "decisions_per_s": round(steps / max(1e-6, elapsed), 2),
                "last_state": {
                    "stage": info.get("stage"),
                    "frame": info.get("frame"),
                    "bullets": info.get("bullets"),
                },
            }
            if vision_model is not None:
                row["vision_diagnostics"] = vision_diagnostics_node(diagnostics)
                diagnostics_raw.append(
                    {
                        "episode": episode,
                        "decisions": steps,
                        "timing_ms": [round(float(value), 6) for value in diagnostics["timing_ms"]],
                        "frame_delta": list(diagnostics["frame_delta"]),
                        "nonpositive": int(diagnostics["nonpositive"]),
                    }
                )
            if pilot is not None:
                row["bombs_used"] = bombs_start - int(round(float(info.get("bombs", 0))))
                if clearances:
                    row["pilot_clearance"] = {
                        "mean": round(float(np.mean(clearances)), 2),
                        "min": round(float(np.min(clearances)), 2),
                    }
            report["episodes"].append(row)
            if cfg.screenshot:
                try:
                    from PIL import Image

                    frame = env.render()
                    path = cfg.out_dir / f"episode-{episode:02d}.png"
                    Image.fromarray(frame).save(path)
                    row["screenshot"] = str(path)
                except Exception as exc:  # diagnostics only
                    row["screenshot_error"] = repr(exc)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        env.close()
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        frames = [row["frames_reported"] for row in report["episodes"]]
        report["summary"] = {
            "episodes": len(frames),
            "frames_mean": round(float(np.mean(frames)), 1) if frames else 0.0,
            "frames_max": int(max(frames)) if frames else 0,
            "stages_max": max((row["stages_reached"] for row in report["episodes"]), default=0),
            "deaths_mean": round(float(np.mean([r["deaths"] for r in report["episodes"]])), 2)
            if report["episodes"]
            else 0.0,
        }
        (cfg.out_dir / "eval-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if vision_model is not None:
            _write_raw_diagnostics(cfg.out_dir, diagnostics_raw)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a TH08 policy on the live game")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--difficulty", default="lunatic", choices=["easy", "normal", "hard", "lunatic"])
    parser.add_argument("--team", type=int, default=0)
    parser.add_argument("--practice-stage", type=int, default=-1)
    parser.add_argument("--frameskip", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--random-policy", action="store_true", help="uniform-random baseline")
    parser.add_argument(
        "--vision-checkpoint",
        default="",
        help="vision scorer checkpoint: the policy plays from a cropped+resized frame stack only",
    )
    parser.add_argument(
        "--vision-depth",
        type=int,
        default=0,
        help="frame-stack depth for the vision policy (0 = follow the checkpoint's config.depth)",
    )
    parser.add_argument(
        "--pilot", action="store_true", help="scripted avoidance teacher (no checkpoint needed)"
    )
    parser.add_argument("--pilot-horizon", type=int, default=0, help="planning steps (0 = default)")
    parser.add_argument("--pilot-beam", type=int, default=0, help="beam width (0 = default)")
    parser.add_argument("--pilot-speed", type=float, default=0.0, help="normal axis speed px/frame")
    parser.add_argument("--pilot-speed-diag", type=float, default=0.0, help="normal diagonal px/frame")
    parser.add_argument("--pilot-focus-speed", type=float, default=0.0, help="focused axis speed px/frame")
    parser.add_argument(
        "--pilot-focus-speed-diag", type=float, default=0.0, help="focused diagonal px/frame"
    )
    parser.add_argument("--pilot-player-half", type=float, default=0.0, help="player hurtbox half size px")
    parser.add_argument(
        "--pilot-max-bullets",
        type=int,
        default=0,
        help="cap the bullets the pilot may plan with (0 = default); set 64 to emulate the network observation",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if not args.random_policy and not args.pilot and not args.checkpoint and not args.vision_checkpoint:
        parser.error("one of --checkpoint / --random-policy / --pilot / --vision-checkpoint is required")
    difficulty = {"easy": spec.EASY, "normal": spec.NORMAL, "hard": spec.HARD, "lunatic": spec.LUNATIC}[
        args.difficulty
    ]
    pilot_config = None
    if args.pilot:
        overrides = {
            "horizon_steps": args.pilot_horizon,
            "beam": args.pilot_beam,
            "speed": args.pilot_speed,
            "speed_diag": args.pilot_speed_diag,
            "focus_speed": args.pilot_focus_speed,
            "focus_speed_diag": args.pilot_focus_speed_diag,
            "player_half": args.pilot_player_half,
            "max_bullets": args.pilot_max_bullets,
        }
        pilot_config = PilotConfig(
            **{name: value for name, value in overrides.items() if value not in (0, 0.0)}
        )
    cfg = EvalConfig(
        checkpoint=Path(args.checkpoint) if args.checkpoint else Path("."),
        out_dir=Path(args.out),
        episodes=args.episodes,
        difficulty=difficulty,
        team=args.team,
        practice_stage=None if args.practice_stage < 0 else args.practice_stage,
        frameskip=args.frameskip,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        device=args.device,
        random_policy=args.random_policy,
        pilot=args.pilot,
        pilot_config=pilot_config,
        vision_checkpoint=Path(args.vision_checkpoint) if args.vision_checkpoint else None,
        vision_depth=args.vision_depth,
    )
    try:
        report = run_evaluation(cfg)
    except ValueError as exc:
        # Known contract violations (v104:CHECKPOINT / CONFIG / ... ) are printed
        # with the same prefix and exit 2.  This replaces the old ``SystemExit``
        # (exit 1) for a ``--vision-depth`` mismatch: an unusable checkpoint must
        # not look like a run-time failure of the game.  Anything else keeps its
        # traceback and its own non-zero exit.
        code = v104_error_code(exc)
        if code is None:
            raise
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
