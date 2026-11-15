"""v4.3.0 model package (work package W03, slice D: the actor candidate).

Only one model class exists in this slice: :class:`touhou_rl.v430.models.actor.ActorPolicy`
— the "first implementable network candidate" of ``DEVELOPMENT_GUIDE.md`` §4.3
(three spatial CNN streams + a 65-dim history projection, fused to 256, one GRU
layer of 256, four independent linear action heads).

The critic and the six auxiliary heads of ``PLAN_SPEC.model.auxiliary_heads`` are
**not** implemented here, and the actor's parameter report states that explicitly
(``includes_critic = False``, ``includes_auxiliary_heads = False``) so an actor
number can never absorb a critic number.

The single public factory is :func:`build_actor`; import the class directly when a
caller needs ``joint_log_prob``/``joint_entropy``/``sample``.
"""

from __future__ import annotations

from touhou_rl.v430.models.actor import (
    HEAD_ORDER,
    MODEL_SPEC,
    ActorPolicy,
    Evaluation,
    HeadLogits,
    HiddenStateError,
    ModelError,
    PrivilegedModelInputError,
    ShapeMismatchError,
    build_actor,
    joint_entropy,
    joint_log_prob,
    parameter_report,
    sample,
)

__all__ = [
    "HEAD_ORDER",
    "MODEL_SPEC",
    "ActorPolicy",
    "Evaluation",
    "HeadLogits",
    "ModelError",
    "PrivilegedModelInputError",
    "ShapeMismatchError",
    "HiddenStateError",
    "build_actor",
    "joint_log_prob",
    "joint_entropy",
    "parameter_report",
    "sample",
]
