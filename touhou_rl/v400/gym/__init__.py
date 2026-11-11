"""v4.0.0 training-ground client (P0-WP4).

Layout follows 03 §9::

    gym/  actions.py   candidate/student encoding -> button mask
          protocol.py  the engine's line protocol
          shm.py       reader for the shared state/pixel block
          env.py       Th08Env: gym-style reset/step
"""

from touhou_rl.v400.gym.actions import (
    AGENT_MASK,
    CAND_MASKS,
    N_CAND,
    STUDENT_ACTIONS,
    cand_to_mask,
    mask_to_cand,
)
from touhou_rl.v400.gym.env import (
    DECISION_FRAMES,
    DIFFICULTIES,
    FrameReply,
    Info,
    PrivilegeError,
    StageStart,
    Th08Env,
    VecTh08Env,
)
from touhou_rl.v400.gym.protocol import HarnessClient, HarnessError, ProtocolStats
from touhou_rl.v400.gym.shm import MAGIC, SharedBlock, segment_path
from touhou_rl.v400.gym.trajectory import (
    RecordedTrajectory,
    TakeoverReport,
    TrajectoryStart,
    TrajectoryStartError,
    load_d0_shard,
    load_d0_shards,
    replay_to_frame,
    sample_target_frame,
)

__all__ = [
    "AGENT_MASK",
    "CAND_MASKS",
    "DECISION_FRAMES",
    "DIFFICULTIES",
    "FrameReply",
    "HarnessClient",
    "HarnessError",
    "Info",
    "MAGIC",
    "N_CAND",
    "PrivilegeError",
    "ProtocolStats",
    "RecordedTrajectory",
    "STUDENT_ACTIONS",
    "SharedBlock",
    "StageStart",
    "TakeoverReport",
    "Th08Env",
    "TrajectoryStart",
    "TrajectoryStartError",
    "VecTh08Env",
    "cand_to_mask",
    "load_d0_shard",
    "load_d0_shards",
    "mask_to_cand",
    "replay_to_frame",
    "sample_target_frame",
    "segment_path",
]
