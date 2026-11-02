"""Stage 2 human-action supervision contract tests."""

import math
from pathlib import Path

import pytest

from touhou_rl.vision_train import VisionTrainConfig, validate_config


def _cfg(**overrides):
    values = dict(dataset=Path("."), out_dir=Path("."), arch="v3-st", selection_contract="v104", depth=3)
    values.update(overrides)
    return VisionTrainConfig(**values)


def test_human_action_weight_zero_is_baseline_and_positive_is_v3_only():
    plan = validate_config(_cfg(human_action_weight=0.0))
    assert plan.objective == "legacy-reg"
    validate_config(_cfg(human_action_weight=0.05))
    with pytest.raises(ValueError, match="human_action_weight"):
        validate_config(_cfg(arch="v2", depth=1, human_action_weight=0.05))


@pytest.mark.parametrize("value", [-0.01, math.nan, math.inf, True])
def test_human_action_weight_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        validate_config(_cfg(human_action_weight=value))
