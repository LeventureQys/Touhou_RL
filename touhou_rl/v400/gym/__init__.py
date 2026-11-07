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
    Info,
    PrivilegeError,
    StageStart,
    Th08Env,
)
from touhou_rl.v400.gym.protocol import HarnessClient, HarnessError, ProtocolStats
from touhou_rl.v400.gym.shm import MAGIC, SharedBlock, segment_path

__all__ = [
    "AGENT_MASK",
    "CAND_MASKS",
    "DECISION_FRAMES",
    "DIFFICULTIES",
    "HarnessClient",
    "HarnessError",
    "Info",
    "MAGIC",
    "N_CAND",
    "PrivilegeError",
    "ProtocolStats",
    "STUDENT_ACTIONS",
    "SharedBlock",
    "StageStart",
    "Th08Env",
    "cand_to_mask",
    "mask_to_cand",
    "segment_path",
]
