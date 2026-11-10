"""P4 prototype package: an RL environment and an actor-critic for v4.0.0.

Everything here is exploratory (EXP-P4-001).  Nothing in this package is a gate,
and nothing here changes the engine, the protocol, ``obs-spec-v1`` or the frozen
student model -- it consumes them.
"""

from touhou_rl.v400.rl.env import (
    ACTION_NAMES,
    DECISION_FRAMES,
    DEFAULT_MAX_FRAMES,
    EngineBackend,
    FakeBackend,
    N_ACTIONS,
    REWARD_TERMS,
    RlEnv,
    RlEnvError,
    VecRlEnv,
    action_to_cand_bomb,
    cand_bomb_to_action,
    reward_scale_report,
    reward_scale_table,
)
from touhou_rl.v400.rl.policy import ActorCritic, build_actor_critic, save_policy_checkpoint

__all__ = [
    "ACTION_NAMES",
    "ActorCritic",
    "DECISION_FRAMES",
    "DEFAULT_MAX_FRAMES",
    "EngineBackend",
    "FakeBackend",
    "N_ACTIONS",
    "REWARD_TERMS",
    "RlEnv",
    "RlEnvError",
    "VecRlEnv",
    "action_to_cand_bomb",
    "build_actor_critic",
    "cand_bomb_to_action",
    "reward_scale_report",
    "reward_scale_table",
    "save_policy_checkpoint",
]
