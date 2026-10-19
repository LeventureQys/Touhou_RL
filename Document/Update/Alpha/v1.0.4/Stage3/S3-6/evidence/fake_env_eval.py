"""Fake-environment deployment check for ``touhou_rl/evaluate.py`` (S3-6).

This is **not** a real-game run and says nothing about A-04: ``EvalConfig.env_factory``
injects a deterministic stub, so the whole vision deployment path can run on the
CPU with no game, no window and no memory reads.  What it proves is the plumbing:

* ``run_evaluation`` loads a depth-3 v2 checkpoint, composes the three-region
  frame stack from ``env.render()`` crops only, and writes ``eval-report.json``;
* the ``vision`` node records the actual arch (``vision_arch``), the depth that
  came from the checkpoint when ``--vision-depth`` was 0, the three ROI shapes,
  the A-04 step cap and the isolation statement (a v3 checkpoint would additionally
  carry the ``contract`` node - not exercised while G-v3 is unapproved);
* every episode carries ``vision_diagnostics`` with exactly the three contracted
  keys, and the raw per-decision samples are saved for recomputation;
* the same pixels with a *poisoned* privileged ``info`` produce byte-identical
  actions and never touch ``env.state`` (isolation I-1/I-2/I-3).

Exit code 0 when every check passes, 1 otherwise (the checks are printed and
written to ``fake-env-checks.json``).

Usage::

    python fake_env_eval.py --out <evidence>/fake-env --checkpoint <ckpt.pt>
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from touhou_rl import vision
from touhou_rl.evaluate import EvalConfig, run_evaluation, vision_arch
from touhou_rl.vmodels import VISION_ARCH_V2, load_vision_checkpoint, save_vision_checkpoint
from touhou_rl.vmodels2 import VisionNetConfigV2, VisionScorerV2

CLIENT_H, CLIENT_W = 480, 640  # TH08 client area the native regions are cut from
EPISODES = 3
STEPS_PER_EPISODE = 4
MAX_STEPS = 12000


class Poison:
    """A stand-in for any privileged field: touching it fails the run."""

    def __getattr__(self, name):
        raise AssertionError(f"privileged field accessed: {name}")


def screen() -> np.ndarray:
    """Deterministic 32 px block pattern covering every native region."""
    image = np.zeros((CLIENT_H, CLIENT_W, 3), dtype=np.uint8)
    cells = itertools.product(range(0, CLIENT_H, 32), range(0, CLIENT_W, 32))
    for index, (row, column) in enumerate(cells):
        image[row : row + 32, column : column + 32] = (
            (index * 37) % 256,
            (index * 91) % 256,
            (index * 53) % 256,
        )
    return image


class StubEnv:
    """Minimal ``Th08Env``: fixed pixels out, deterministic bookkeeping in."""

    action_space = SimpleNamespace(sample=lambda: np.zeros(4, dtype=np.int64))

    def __init__(self, *, poison: bool):
        self.poison = bool(poison)
        self.actions: list[np.ndarray] = []
        self.renders = 0
        self.state_reads = 0
        self.closed = False
        self._step = 0
        self._screen = screen()

    @property
    def state(self):
        self.state_reads += 1
        raise AssertionError("vision branch read env.state")

    def reset(self):
        self._step = 0
        info = {"state": {"bombs": 2}, "stages_reached": 0, "stage": 0, "frame": 0, "bullets": 0}
        return {"state": {"bombs": 2}}, info

    def render(self):
        self.renders += 1
        return self._screen

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.int64).copy())
        self._step += 1
        done = self._step >= STEPS_PER_EPISODE
        privileged = Poison() if self.poison else np.zeros((0, 4), dtype=np.float32)
        info = {
            "episode": {
                "frames": 2 * self._step,
                "deaths": 0,
                "lives": 2.0,
                "terminated": done,
                "truncated": False,
                "termination_reason": "game over" if done else "",
                "stages_reached": 0,
                "score": 0,
                "graze": 0,
                "episode_reward": 0.0,
            },
            "stages_reached": 0,
            "stage": 0,
            "frame": 2 * self._step,
            "bullets": 0,
            "bullets_raw": privileged,
            "bosses": privileged,
            "bombs": 2,
        }
        return {"state": {"bombs": 2}}, 0.0, done, False, info

    def close(self):
        self.closed = True


class StubFactory:
    def __init__(self, *, poison: bool):
        self.poison = poison
        self.envs: list[StubEnv] = []
        self.configs: list = []

    def __call__(self, env_config):
        self.configs.append(env_config)
        env = StubEnv(poison=self.poison)
        self.envs.append(env)
        return env


def _checkpoint(path: Path) -> Path:
    torch.manual_seed(7)
    config = VisionNetConfigV2(
        depth=3, pf_width=8, hud_width=4, trunk_hidden=32, state_hidden=32, candidate_hidden=16
    )
    save_vision_checkpoint(
        path,
        VisionScorerV2(config),
        {"stage": "policy", "epoch": 1, "history_contract": "causal-segment-v1"},
    )
    return path


def _v3_checkpoint(path: Path) -> Path:
    """A v3-st checkpoint (G-v3 approved, S3-5 model) with the C6 identity."""
    from touhou_rl.vmodels3 import VisionSpatialTemporalConfig, VisionSpatialTemporalPolicy

    torch.manual_seed(11)
    save_vision_checkpoint(
        path,
        VisionSpatialTemporalPolicy(VisionSpatialTemporalConfig()),
        {
            "source_state_sha": "d" * 64,
            "teacher_fingerprint": "a" * 64,
            "manifest_sha": "b" * 64,
            "val_index_sha256": "c" * 64,
            "loss_config": {"objective": "rank", "rank_delta_px": 0.5},
            "stage": "policy",
            "epoch": 1,
            "depth": 3,
            "history_contract": "causal-segment-v1",
        },
    )
    return path


def _deploy(out_dir: Path, checkpoint: Path, *, poison: bool):
    """One full vision deployment against the stub environment."""
    factory = StubFactory(poison=poison)
    report = run_evaluation(
        EvalConfig(
            checkpoint=Path("."),
            out_dir=out_dir,
            episodes=EPISODES,
            device="cpu",
            screenshot=False,
            max_steps=MAX_STEPS,
            vision_checkpoint=checkpoint,
            vision_depth=0,  # 0 = follow the checkpoint's config.depth
            env_factory=factory,
        )
    )
    return report, factory


def run(out: Path, checkpoint: Path, v3_checkpoint: Path | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    plain, plain_factory = _deploy(out / "plain", checkpoint, poison=False)
    poisoned, poison_factory = _deploy(out / "poison", checkpoint, poison=True)

    model, _extra = load_vision_checkpoint(checkpoint, map_location="cpu")
    node = plain["vision"]
    actions_plain = np.stack(plain_factory.envs[0].actions)
    actions_poison = np.stack(poison_factory.envs[0].actions)
    raw_path = out / "plain" / "vision-diagnostics-raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    checks: list[dict] = []

    def check(name: str, ok: bool, detail="") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    check("policy_is_vision", plain["policy"] == "vision", plain["policy"])
    check("vision_arch_is_v2", node["arch"] == vision_arch(model) == VISION_ARCH_V2, node["arch"])
    check("depth_follows_the_checkpoint", node["depth"] == 3, node["depth"])
    check(
        "region_sizes_are_the_three_roi",
        node["region_sizes"]
        == {"playfield": [224, 192], "hud_score": [34, 216], "hud_status": [126, 216]},
        node["region_sizes"],
    )
    check(
        "frame_shape_includes_rgb3",
        node["frame_shape"]["playfield"] == [224, 192, vision.VISION_CHANNELS],
        node["frame_shape"],
    )
    check("isolation_statement_is_recorded", str(node["isolation"]).startswith("I-1"), node["isolation"])
    check("step_cap_is_recorded", node.get("max_steps") == MAX_STEPS, node.get("max_steps"))
    check("no_v3_contract_node_without_g_v3", "contract" not in node, sorted(node))
    check(
        "eval_report_written",
        (out / "plain" / "eval-report.json").is_file(),
        str(out / "plain" / "eval-report.json"),
    )
    check("episode_count", len(plain["episodes"]) == EPISODES, len(plain["episodes"]))
    check(
        "episode_fields_complete",
        all(
            {"terminated", "truncated", "termination_reason"} <= set(row)
            for row in plain["episodes"]
        ),
        "terminated/truncated/termination_reason",
    )
    diagnostics_ok = True
    for row in plain["episodes"]:
        node_diag = row.get("vision_diagnostics", {})
        if set(node_diag) != {
            "bomb_actions_requested",
            "screenshot_to_action_ms",
            "observed_frame_delta",
        }:
            diagnostics_ok = False
            continue
        if node_diag["bomb_actions_requested"] != 0:
            diagnostics_ok = False
        timing = node_diag["screenshot_to_action_ms"]
        if not (
            timing["count"] == STEPS_PER_EPISODE
            and timing["mean"] > 0
            and timing["p50"] <= timing["p95"] <= timing["max"]
        ):
            diagnostics_ok = False
        delta = node_diag["observed_frame_delta"]
        if not (delta["count"] == STEPS_PER_EPISODE - 1 and delta["nonpositive"] == 0):
            diagnostics_ok = False
    check("vision_diagnostics_contract", diagnostics_ok, "3 keys, counts, p50<=p95<=max")
    check(
        "raw_diagnostics_saved",
        raw["episodes"][0]["decisions"] == STEPS_PER_EPISODE
        and len(raw["episodes"][0]["timing_ms"]) == STEPS_PER_EPISODE,
        str(raw_path),
    )
    check(
        "poisoned_privileged_info_does_not_change_actions",
        np.array_equal(actions_plain, actions_poison),
        f"{actions_plain.shape} identical={np.array_equal(actions_plain, actions_poison)}",
    )
    check(
        "env_state_never_read",
        plain_factory.envs[0].state_reads == 0 and poison_factory.envs[0].state_reads == 0,
        f"plain={plain_factory.envs[0].state_reads} poison={poison_factory.envs[0].state_reads}",
    )
    check(
        "env_closed",
        plain_factory.envs[0].closed and poison_factory.envs[0].closed,
        "close() called",
    )
    decisions = EPISODES * STEPS_PER_EPISODE
    check(
        "action_shape_is_the_four_tuple",
        actions_plain.shape == (decisions, 4)
        and actions_plain[:, 1].tolist() == [1] * decisions
        and actions_plain[:, 3].tolist() == [0] * decisions,
        f"shape={actions_plain.shape} actions={actions_plain[:2].tolist()}",
    )

    # --- v3-st dispatch (G-v3 approved; S3-5 model) -------------------------
    v3_node = None
    if v3_checkpoint is not None:
        v3_plain, v3_factory = _deploy(out / "v3-plain", v3_checkpoint, poison=False)
        v3_poisoned, v3_poison_factory = _deploy(out / "v3-poison", v3_checkpoint, poison=True)
        v3_model, _v3_extra = load_vision_checkpoint(v3_checkpoint, map_location="cpu")
        v3_node = v3_plain["vision"]
        v3_actions_plain = np.stack(v3_factory.envs[0].actions)
        v3_actions_poison = np.stack(v3_poison_factory.envs[0].actions)
        contract = v3_node.get("contract", {})
        check("v3_policy_is_vision", v3_plain["policy"] == "vision", v3_plain["policy"])
        check(
            "v3_arch_is_v3st",
            v3_node["arch"] == vision_arch(v3_model) == "v3-st",
            v3_node["arch"],
        )
        check("v3_depth_follows_the_checkpoint", v3_node["depth"] == 3, v3_node["depth"])
        check(
            "v3_contract_node",
            contract.get("schema") == "vision-checkpoint/3"
            and contract.get("arch") == "v3-st"
            and contract.get("candidate_order_version") == "focus9-move-v1"
            and contract.get("score_semantics") == "preference_logits"
            and contract.get("history_contract") == "causal-segment-v1",
            sorted(contract),
        )
        check(
            "v3_contract_identity",
            isinstance(contract.get("identity"), dict)
            and len(contract["identity"]) >= 5
            and contract["identity"].get("manifest_sha") == "b" * 64,
            contract.get("identity"),
        )
        check(
            "v3_diagnostics_contract",
            all(
                set(row.get("vision_diagnostics", {}))
                == {"bomb_actions_requested", "screenshot_to_action_ms", "observed_frame_delta"}
                for row in v3_plain["episodes"]
            ),
            "3 keys on every v3 episode",
        )
        check(
            "v3_poisoned_privileged_info_does_not_change_actions",
            np.array_equal(v3_actions_plain, v3_actions_poison),
            f"{v3_actions_plain.shape} identical={np.array_equal(v3_actions_plain, v3_actions_poison)}",
        )
        check(
            "v3_env_state_never_read",
            v3_factory.envs[0].state_reads == 0 and v3_poison_factory.envs[0].state_reads == 0,
            "0 reads",
        )
        check(
            "v3_action_shape_is_the_four_tuple",
            v3_actions_plain.shape == (decisions, 4)
            and v3_actions_plain[:, 1].tolist() == [1] * decisions
            and v3_actions_plain[:, 3].tolist() == [0] * decisions,
            f"shape={v3_actions_plain.shape} actions={v3_actions_plain[:2].tolist()}",
        )

    payload = {
        "schema": "s3-6-fake-env-check/1",
        "note": (
            "fake environment (EvalConfig.env_factory stub): proves the deployment plumbing and "
            "isolation only. It is not a real-game run and cannot stand in for A-04."
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
        "vision_node": node,
        "episode_0": plain["episodes"][0],
        "raw_diagnostics_note": raw["note"],
        "v3": {
            "checkpoint": str(v3_checkpoint) if v3_checkpoint is not None else None,
            "checkpoint_sha256": (
                __import__("hashlib").sha256(v3_checkpoint.read_bytes()).hexdigest()
                if v3_checkpoint is not None
                else None
            ),
            "vision_node": v3_node,
            "eval_report": str(out / "v3-plain" / "eval-report.json"),
        },
        "checks": checks,
        "passed": all(item["ok"] for item in checks),
        "eval_report": str(out / "plain" / "eval-report.json"),
    }
    (out / "fake-env-checks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for item in checks:
        print(f"[{'PASS' if item['ok'] else 'FAIL'}] {item['check']}: {item['detail']}", flush=True)
    print(f"passed={payload['passed']}", flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fake-env vision deployment check (S3-6)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--v3-checkpoint", default="")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else out / "fake-v2-depth3.pt"
    if not checkpoint.is_file():
        _checkpoint(checkpoint)
    v3_checkpoint = Path(args.v3_checkpoint) if args.v3_checkpoint else out / "fake-v3-depth3.pt"
    if not v3_checkpoint.is_file():
        _v3_checkpoint(v3_checkpoint)
    payload = run(out, checkpoint, v3_checkpoint)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
