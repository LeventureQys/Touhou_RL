"""Exercise the actual evaluate entry with the opt-in macro sidecar, no game."""
from pathlib import Path

import pytest
import torch

from touhou_rl.evaluate import EvalConfig, run_evaluation
from touhou_rl.macro_policy import MacroPolicy, save_macro_checkpoint
from touhou_rl.vmodels import load_vision_checkpoint
from touhou_rl.tests.test_evaluate_vision import _write_v3, StubFactory, _run


def test_macro_actions_reach_pixel_deployment(tmp_path):
    checkpoint = _write_v3(tmp_path / "vision.pt")
    base, _ = load_vision_checkpoint(checkpoint)
    policy = MacroPolicy(base)
    with torch.no_grad():
        # Forced action checks the wiring, not learned gameplay skill.
        policy.heads["residual"][-1].bias[7] = 1000
    sidecar = tmp_path / "macro.pt"
    save_macro_checkpoint(sidecar, policy, checkpoint, metadata={"engineering_test":True})
    factory = StubFactory(steps_to_end=2, poison=True)
    report = _run(tmp_path, factory, vision_checkpoint=checkpoint,
                  macro_checkpoint=sidecar, vision_score_ema=0, vision_switch_penalty=0)
    assert report["vision"]["macro"]["checkpoint"] == str(sidecar)
    assert report["vision"]["arch"] == "v3-st"
    assert factory.envs[0].state_reads == 0
    assert factory.envs[0].actions
    assert all(action.tolist() == [7,1,0,0] for action in factory.envs[0].actions)


def test_macro_requires_exclusive_pixel_policy_before_env(tmp_path):
    def forbidden(*args):
        raise AssertionError("must reject before opening a game")
    with pytest.raises(ValueError, match="exclusive"):
        run_evaluation(EvalConfig(checkpoint=Path("."), out_dir=tmp_path,
            macro_checkpoint=tmp_path / "macro.pt", env_factory=forbidden))
