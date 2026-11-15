"""BC training orchestration on **synthetic batches** (v4.3.0 work package W03E).

Specification: ``Document/Update/Alpha/v4.3.0/DEVELOPMENT_GUIDE.md`` §7.4 (独立监督项、
各自按有效标签数归一、缺标签的 batch 不产生该 loss) and §8 (战术/BC 训练语义),
``PLAN_SPEC.json`` ``bc`` (``optimizer=AdamW``, ``learning_rate=1e-4``,
``weight_decay=1e-4``, ``missing_action_labels=mask``) and ``micro`` (arms
``D-state`` / ``D-pixel`` / ``D-pixel-memory``).

Scope of this slice — deliberately small:

* only the **orchestration**: a four-head cross-entropy BC loss, a deterministic
  training loop, a checkpoint writer/reader, and the fail-closed entry point that
  refuses to train before the manifest and the frozen split registry pass;
* **no game data, no engine, no rollout.**  ``batches`` are synthetic tensors the
  caller builds; nothing here reads ``.tmp_*`` corpora, launches TH08, or claims
  any game capability.  The accompanying test file is a unit test of this module,
  not evidence of learned play.

Reused, never re-implemented (one rule per fact):

=================================== ==================================================
fact                                owner (imported, unchanged)
=================================== ==================================================
four heads 9/2/2/2, ``joint=72``    :mod:`touhou_rl.v430.contracts`
``/255`` normalisation point        :mod:`touhou_rl.v430.observation` /
                                    ``contracts.O2_NORMALIZE_DIVISOR`` (applied inside
                                    the actor encoder, **never here**)
``ActorPolicy`` / ``build_actor``   :mod:`touhou_rl.v430.models.actor`
``o2_digest`` / ``SPEC_REVISION``   :mod:`touhou_rl.v430.contracts`
manifest gate                       :mod:`touhou_rl.v430.data.manifest`
split/lineage gate                  :mod:`touhou_rl.v430.data.lineage`
result schema                       :mod:`touhou_rl.v430.contracts`
=================================== ==================================================

Semantics fixed here (each one is a refusal, never a silent default):

1. :func:`bc_loss` — the four heads' cross entropies are **added**, not averaged:
   ``total = mean_ce(move) + mean_ce(focus) + mean_ce(shoot) + mean_ce(bomb)``.
   Each head is averaged over its own samples (so one label per head per sample is
   counted once), but the total is **not** divided by four: a silent ``/4`` would
   shrink every gradient and make the reported number impossible to hand-check.
2. :func:`train_bc` — deterministic given ``seed`` + the same batches and the same
   starting parameters: no global RNG, no implicit shuffle, only an explicit
   :class:`torch.Generator` (used when ``shuffle_batches=True``).  The loop adds no
   randomness of its own; the actor has no dropout and no mutable batch statistics
   (GroupNorm/LayerNorm), so a forward pass is a pure function of its inputs.
3. :func:`save_checkpoint` / :func:`load_checkpoint` — a checkpoint records the
   model config (enough to rebuild the same structure), the ``torch`` version,
   ``o2_digest`` and ``spec_revision``.  A differing ``o2_digest`` (or
   ``spec_revision``, or an unrebuildable ``model_config``) is refused at load, so
   weights trained on another observation encoding cannot masquerade as these; an
   existing file is never overwritten (``--out`` semantics of the CLI).
4. :func:`train_from_manifest` — ``validate_manifest`` first, then the O2 guard
   (:func:`~touhou_rl.v430.data.manifest.assert_o2_compatible`, which refuses a
   corpus recorded under a legacy ``obs_contract`` even when the manifest is
   ``valid``), then ``validate_splits``, training last.  A ``valid=False`` /
   ``passed=False`` report raises, and an O2-incompatible manifest raises the
   manifest layer's own exception; there is no "train first, validate later" path,
   and ``run=False`` validates (guard included) without training — the returned
   result then says ``not_run``.  ``batch_source`` is the *caller's* declaration
   (``"synthetic"`` or ``"captured_o2"``), recorded but never verifiable here.
5. ``memory_mode`` — ``"none"`` (arm ``D-pixel``) never passes ``hidden`` to the
   actor (a fresh zero state per forward); ``"gru"`` (arm ``D-pixel-memory``)
   carries the GRU state across the steps of one sequence.  The mode actually used
   is recorded in every step's report **and** in the result dict.

Purity: torch + stdlib + the modules above.  No engine import, no ``gym``, no
legacy ``v400``/``v420`` package, no file I/O outside the checkpoint path, no
global mutable state.

TODO(W03E): full PPO / value-warmup / burn-in prefix recompute (guide §7.1) and the
real D-state / D-pixel / D-pixel-memory arms on real corpora are **out of this
slice**; only the BC orchestration on synthetic batches is implemented here.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from touhou_rl.v430 import contracts
from touhou_rl.v430.data import lineage, manifest as manifest_module
from touhou_rl.v430.models import actor

__all__ = [
    # errors
    "LearnError",
    # constants
    "MEMORY_MODES",
    "BATCH_SOURCES",
    "BC_HEADS",
    "CHECKPOINT_SCHEMA",
    "MODEL_CONFIG_SCHEMA",
    "BATCH_KEYS",
    "ADAMW_WEIGHT_DECAY",
    "BCLoss",
    "TrainingReport",
    "Checkpoint",
    # API
    "bc_loss",
    "train_bc",
    "checkpoint_model_config",
    "save_checkpoint",
    "load_checkpoint",
    "train_from_manifest",
]


class LearnError(RuntimeError):
    """Every fail-closed refusal of this module (bad data, bad order, bad file)."""


#: ``memory_mode`` values: ``"none"`` = D-pixel, ``"gru"`` = D-pixel-memory.
MEMORY_MODES: tuple[str, ...] = ("none", "gru")

#: ``batch_source`` values a caller may declare for ``train_from_manifest``.
#: ``"synthetic"`` = generated in code, ``"captured_o2"`` = real O2 capture.
#: This is a **declaration**: nothing in this module can inspect a batch's
#: provenance, so the value is recorded, never verified.
BATCH_SOURCES: tuple[str, ...] = ("synthetic", "captured_o2")

#: The four heads, in contract order.  ``contracts.HEAD_ORDER`` is the owner.
BC_HEADS: tuple[str, ...] = contracts.HEAD_ORDER

#: Checkpoint schema tag; a checkpoint with another tag is refused at load.
CHECKPOINT_SCHEMA = "v430-checkpoint/1"
#: Model-config schema tag inside a checkpoint.
MODEL_CONFIG_SCHEMA = "v430-model-config/1"

#: The required keys of one training batch: ``{"observations": ..., "targets": ...}``.
BATCH_KEYS: tuple[str, ...] = ("observations", "targets")

#: ``PLAN_SPEC.bc.weight_decay`` — the default of the AdamW optimizer built here.
ADAMW_WEIGHT_DECAY: float = 1e-4


# ---------------------------------------------------------------------------
# small guards: every refusal is a LearnError, never a silent default
# ---------------------------------------------------------------------------
def _require_mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LearnError(f"{what} must be a mapping, got {type(value).__name__}")
    return value


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LearnError(f"{name} must be an int, got {type(value).__name__}={value!r}")
    if minimum is not None and value < minimum:
        raise LearnError(f"{name}={value} is below the minimum {minimum}")
    return value


def _require_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LearnError(f"{name} must be a number, got {type(value).__name__}={value!r}")
    number = float(value)
    if not math.isfinite(number) or not number > 0.0:
        raise LearnError(f"{name}={value!r} must be finite and > 0")
    return number


def _require_memory_mode(value: Any) -> str:
    if value not in MEMORY_MODES:
        raise LearnError(
            f"memory_mode {value!r} not in {list(MEMORY_MODES)}; 'none' is D-pixel "
            f"(no hidden state) and 'gru' is D-pixel-memory (per-sequence hidden)")
    return str(value)


def _require_batch_source(value: Any) -> str:
    """``batch_source`` is a caller declaration; an unregistered one is refused."""

    if value not in BATCH_SOURCES:
        raise LearnError(
            f"batch_source {value!r} not in {list(BATCH_SOURCES)}; declare "
            f"'synthetic' for batches generated in code or 'captured_o2' for "
            f"real O2 captures (the module records the declaration, it cannot "
            f"verify it)")
    return str(value)


def _field_mapping(observation_like: Any) -> Mapping[str, Any]:
    """One observation record -> ``{field: value}`` (a mapping or a dataclass)."""

    if isinstance(observation_like, Mapping):
        return observation_like
    if dataclasses.is_dataclass(observation_like) and not isinstance(
            observation_like, type):
        return {field.name: getattr(observation_like, field.name)
                for field in dataclasses.fields(observation_like)}
    raise LearnError(
        f"an observation must be a mapping or a dataclass (O2Observation / "
        f"ActorObservation), got {type(observation_like).__name__}")


def _observation_count(observations: Any) -> int:
    """The number of decisions in one (batched or unbatched) observation."""

    fields = _field_mapping(observations)
    if "global_rgb" not in fields:
        raise LearnError(
            "the observation has no 'global_rgb' field, so its sample count is "
            "unknown; the O2 fields are required as a whole")
    shape = getattr(fields["global_rgb"], "shape", None)
    if shape is None:
        raise LearnError(
            f"global_rgb is {type(fields['global_rgb']).__name__}, expected an array "
            f"or a tensor with a shape")
    dimensions = tuple(int(dim) for dim in shape)
    if not dimensions:
        raise LearnError("global_rgb has no shape; an empty observation batch is not "
                         "a training sample")
    return int(dimensions[0]) if len(dimensions) == 4 else 1


def _is_step_sequence(value: Any) -> bool:
    """True for a sequence of per-step items (never a tensor, mapping or string)."""

    return (isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray, torch.Tensor, Mapping)))


def _target_dtype_ok(tensor: torch.Tensor) -> bool:
    return bool(tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64,
                                 torch.uint8))


def _as_label_tensor(labels: Any, *, what: str) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        tensor = labels.detach()
    else:
        array = labels
        try:
            tensor = torch.as_tensor(array)
        except Exception as exc:  # noqa: BLE001 - an unreadable label is a caller error
            raise LearnError(
                f"{what}: labels of type {type(labels).__name__} cannot become a "
                f"tensor: {exc}") from exc
    if not _target_dtype_ok(tensor):
        raise LearnError(
            f"{what}: label dtype {tensor.dtype} is not integral; a label is an "
            f"action index, so a float/bool label is refused instead of cast")
    return tensor


def _as_head_logits(head_logits: Any) -> actor.HeadLogits:
    """``HeadLogits`` or a four-key mapping -> a validated :class:`actor.HeadLogits`."""

    if isinstance(head_logits, actor.HeadLogits):
        return head_logits
    mapping = _require_mapping(head_logits, "head_logits")
    missing = [head for head in BC_HEADS if head not in mapping]
    unknown = sorted(str(name) for name in mapping if name not in BC_HEADS)
    if missing or unknown:
        raise LearnError(
            f"head_logits must hold exactly the four heads {list(BC_HEADS)}; "
            f"missing {missing}, unknown {unknown}")
    converted: dict[str, torch.Tensor] = {}
    for head in BC_HEADS:
        value = mapping[head]
        if not isinstance(value, torch.Tensor):
            raise LearnError(
                f"head_logits[{head!r}] is {type(value).__name__}, expected a "
                f"torch.Tensor with raw logits")
        # Never detached: bc_loss must stay differentiable through the heads.
        converted[head] = value
    try:
        return actor.HeadLogits(**converted)
    except (actor.ModelError, contracts.ContractError) as exc:
        raise LearnError(f"head_logits are not a valid four-head logit bundle: {exc}") \
            from exc


def _as_targets(targets: Any, *, batch_size: int, what: str = "targets"
                ) -> dict[str, torch.Tensor]:
    """``(move, focus, shoot, bomb)`` labels -> ``{head: int64 (B,)}``, bounds checked.

    Accepted forms: a mapping of the four head names to 1-D label columns, or a
    ``(B, 4)`` / ``(4, B)``-free integer array whose columns are in
    ``contracts.HEAD_ORDER`` (i.e. ``(B, 4)``).  A wrong count, a wrong number of
    columns or a label outside ``contracts.HEAD_SIZES`` raises :class:`LearnError`.
    """

    if isinstance(targets, Mapping):
        mapping = targets
        missing = [head for head in BC_HEADS if head not in mapping]
        unknown = sorted(str(name) for name in mapping if name not in BC_HEADS)
        if missing or unknown:
            raise LearnError(
                f"{what} must hold exactly the four heads {list(BC_HEADS)}; missing "
                f"{missing}, unknown {unknown}")
        columns = {head: _as_label_tensor(mapping[head], what=f"{what}[{head!r}]")
                   for head in BC_HEADS}
    else:
        tensor = _as_label_tensor(targets, what=what)
        if tensor.ndim != 2 or int(tensor.shape[1]) != len(BC_HEADS):
            raise LearnError(
                f"{what} has shape {tuple(int(d) for d in tensor.shape)}; expected "
                f"(N, {len(BC_HEADS)}) columns in the contract order {list(BC_HEADS)}")
        columns = {head: tensor[:, index]
                   for index, head in enumerate(BC_HEADS)}

    label_map: dict[str, torch.Tensor] = {}
    for head in BC_HEADS:
        column = columns[head].reshape(-1).to(torch.int64)
        if int(column.numel()) != int(batch_size):
            raise LearnError(
                f"{what}[{head!r}] holds {int(column.numel())} label(s) but the "
                f"logits carry {batch_size} sample(s); a short label column cannot "
                f"be broadcast or zero-filled")
        if int(column.numel()) == 0:
            raise LearnError(f"{what} is empty; an empty batch is not a training step")
        size = int(contracts.HEAD_SIZES[head])
        smallest = int(column.min())
        largest = int(column.max())
        if smallest < 0 or largest >= size:
            raise LearnError(
                f"{what}[{head!r}] holds label(s) in [{smallest}, {largest}] but head "
                f"{head!r} has only {size} classes (valid 0..{size - 1})")
        label_map[head] = column
    return label_map


# ---------------------------------------------------------------------------
# 1. the BC loss: four cross entropies, added
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class BCLoss:
    """One BC loss value plus its per-head decomposition (still differentiable)."""

    total: torch.Tensor
    per_head: dict[str, torch.Tensor]
    n_samples: int

    def item(self) -> float:
        """The total loss as a Python float (detached)."""

        return float(self.total.detach())

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready record: ``{loss, per_head_loss, n_samples}``."""

        return {
            "loss": self.item(),
            "per_head_loss": {head: float(value.detach())
                              for head, value in self.per_head.items()},
            "n_samples": int(self.n_samples),
        }


def bc_loss(head_logits: Any, targets: Any) -> BCLoss:
    """The four-head behaviour-cloning loss: ``sum(mean CE per head)``.

    ``mean_ce(head) = -mean_i log softmax(logits[head])[i, target_i]``.  The total
    is the plain sum of the four numbers — **not** divided by four — so the value
    is exactly what a hand calculation gives and the gradient is not silently
    scaled down.

    Parameters
    ----------
    head_logits:
        An :class:`~touhou_rl.v430.models.actor.HeadLogits` (or a mapping of the
        four head names to raw-logit tensors), batched ``(B, size)`` or unbatched
        ``(size,)``.
    targets:
        ``(move, focus, shoot, bomb)`` integer labels: a mapping of the four head
        names to 1-D label columns, or an ``(B, 4)`` integer array in
        ``contracts.HEAD_ORDER`` order.

    Returns
    -------
    BCLoss
        ``total`` (a differentiable 0-dim tensor), ``per_head`` and ``n_samples``.

    Raises
    ------
    LearnError
        A missing/unknown head, a non-tensor or wrong-width logit, a label count
        that disagrees with the batch, a non-integer label, an out-of-range label,
        or an empty batch.
    """

    logits = _as_head_logits(head_logits)
    labels = _as_targets(targets, batch_size=logits.batch_size)

    per_head: dict[str, torch.Tensor] = {}
    for head in BC_HEADS:
        size = int(contracts.HEAD_SIZES[head])
        head_logits_tensor = logits[head]
        if int(head_logits_tensor.shape[-1]) != size:  # pragma: no cover - HeadLogits
            raise LearnError(
                f"head {head!r} has {int(head_logits_tensor.shape[-1])} logit(s), "
                f"expected {size} (contracts.HEAD_SIZES)")
        flat = head_logits_tensor.reshape(-1, size)
        per_head[head] = torch.nn.functional.cross_entropy(
            flat, labels[head], reduction="mean")

    # A fixed addition order keeps the floating-point sum reproducible bit for bit.
    total = per_head[BC_HEADS[0]]
    for head in BC_HEADS[1:]:
        total = total + per_head[head]
    return BCLoss(total=total, per_head=per_head, n_samples=int(logits.batch_size))


# ---------------------------------------------------------------------------
# 2. the deterministic training loop
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class TrainingReport:
    """The record of one :func:`train_bc` call (synthetic-batch scope only)."""

    steps: tuple[dict[str, Any], ...]
    memory_mode: str
    seed: int
    epochs: int
    lr: float
    optimizer: str
    max_steps: int | None
    shuffle_batches: bool

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def losses(self) -> tuple[float, ...]:
        """The per-step total losses, in execution order."""

        return tuple(float(step["loss"]) for step in self.steps)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready record of the run (includes the memory mode actually used)."""

        return {
            "status": "ok" if self.steps else "not_run",
            "memory_mode": self.memory_mode,
            "seed": int(self.seed),
            "epochs": int(self.epochs),
            "lr": float(self.lr),
            "optimizer": self.optimizer,
            "max_steps": self.max_steps,
            "shuffle_batches": bool(self.shuffle_batches),
            "n_steps": self.n_steps,
            "steps": [dict(step) for step in self.steps],
            "first_loss": self.losses[0] if self.steps else None,
            "last_loss": self.losses[-1] if self.steps else None,
        }


def _batch_step(model: torch.nn.Module, batch: Any, memory_mode: str
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    """One optimizer step's loss for one batch: ``(total, per_head, n_samples)``."""

    mapping = _require_mapping(batch, "a training batch")
    unknown = sorted(str(name) for name in mapping if name not in BATCH_KEYS)
    missing = [name for name in BATCH_KEYS if name not in mapping]
    if unknown or missing:
        raise LearnError(
            f"a training batch must hold exactly {list(BATCH_KEYS)}; missing "
            f"{missing}, unknown {unknown}")
    observations = mapping["observations"]
    targets = mapping["targets"]

    if memory_mode == "none":
        if _is_step_sequence(observations):
            raise LearnError(
                "memory_mode='none' (D-pixel) got a sequence of per-step "
                "observations; only the last step would be trained. Use "
                "memory_mode='gru' for a sequence, or pass one observation")
        # D-pixel: no hidden state is passed at all (the actor starts from zeros).
        logits, _hidden = model(observations)
        loss = bc_loss(logits, targets)
        return loss.total, loss.per_head, loss.n_samples

    if not _is_step_sequence(observations):
        raise LearnError(
            "memory_mode='gru' (D-pixel-memory) requires a sequence of per-step "
            "observations; a single observation has no sequence to carry a hidden "
            "state over")
    steps = list(observations)
    if not steps:
        raise LearnError("memory_mode='gru' got an empty observation sequence")
    if not _is_step_sequence(targets):
        raise LearnError(
            "memory_mode='gru' requires one label block per step (a sequence of the "
            "same length as the observation sequence), got "
            f"{type(targets).__name__}")
    target_steps = list(targets)
    if len(target_steps) != len(steps):
        raise LearnError(
            f"memory_mode='gru' got {len(steps)} observation step(s) but "
            f"{len(target_steps)} label block(s); a missing step cannot be reused")

    hidden: Any = None  # (1, B, 256) float32, carried across this sequence only
    totals: list[torch.Tensor] = []
    per_head_steps: dict[str, list[torch.Tensor]] = {head: [] for head in BC_HEADS}
    n_samples = 0
    for step_index, (step_observations, step_targets) in enumerate(
            zip(steps, target_steps)):
        logits, hidden = model(step_observations, hidden)
        loss = bc_loss(logits, step_targets)
        n_samples += int(loss.n_samples)
        totals.append(loss.total)
        for head in BC_HEADS:
            per_head_steps[head].append(loss.per_head[head])
    total = torch.stack(totals).sum() / len(totals)
    per_head = {head: torch.stack(per_head_steps[head]).sum() / len(totals)
                for head in BC_HEADS}
    return total, per_head, n_samples


def train_bc(model: Any, batches: Any, *, epochs: int, lr: float, seed: int,
             optimizer: Any = None, memory_mode: str = "none",
             max_steps: int | None = None,
             weight_decay: float = ADAMW_WEIGHT_DECAY,
             shuffle_batches: bool = False) -> TrainingReport:
    """Train ``model`` on ``batches`` with the four-head BC loss.

    Determinism: the same ``seed``, the same batches **and the same starting
    parameters** give bit-for-bit identical per-step losses and final weights.
    Nothing here touches the global RNG — an explicit :class:`torch.Generator` is
    the only source of randomness (and it is only consulted when
    ``shuffle_batches=True``).  The actor has no dropout and no mutable batch
    statistics, so a forward pass adds no hidden randomness either.

    Parameters
    ----------
    model:
        A :class:`torch.nn.Module` whose ``forward(obs[, hidden])`` returns
        ``(HeadLogits, hidden)`` — normally
        :class:`~touhou_rl.v430.models.actor.ActorPolicy`.
    batches:
        A non-empty sequence of ``{"observations": ..., "targets": ...}`` mappings.
        With ``memory_mode="none"`` ``observations`` is one (possibly batched)
        observation; with ``"gru"`` it is a sequence of per-step observations and
        ``targets`` a sequence of label blocks of the same length.
    epochs, lr, seed:
        Required.  ``epochs >= 1``, ``lr > 0`` and a finite seed.
    optimizer:
        An existing :class:`torch.optim.Optimizer` (used as is) or ``None`` for a
        fresh AdamW over the model's trainable parameters.
    memory_mode:
        ``"none"`` (D-pixel, no hidden) or ``"gru"`` (D-pixel-memory; the GRU state
        is carried over the steps of each sequence).  Recorded in the report.
    max_steps:
        Stop after this many optimizer steps (across epochs).  ``None`` runs every
        epoch; a value ``<= 0`` is refused.
    weight_decay:
        AdamW's weight decay when this call builds the optimizer.
    shuffle_batches:
        Permute the batch order each epoch with the seeded generator.  ``False``
        (default) keeps the caller's order, so the loss sequence is reproducible
        even without noting the permutation.

    Returns
    -------
    TrainingReport
        Every step's ``{"step", "epoch", "loss", "per_head_loss", "n_samples"}``
        plus the run identity (seed, epochs, lr, optimizer, memory_mode).

    Raises
    ------
    LearnError
        Empty ``batches``, ``epochs <= 0``, ``lr <= 0``, ``max_steps <= 0``, an
        invalid ``memory_mode``, a malformed batch, out-of-range labels, a label /
        batch-size mismatch, or a model/optimizer that is not a torch object.
    """

    if not isinstance(model, torch.nn.Module):
        raise LearnError(
            f"model must be a torch.nn.Module, got {type(model).__name__}")
    if isinstance(batches, (str, bytes, Mapping)) or not isinstance(batches, Sequence):
        raise LearnError(
            f"batches must be a non-empty sequence of batch mappings, got "
            f"{type(batches).__name__}")
    batch_list = list(batches)
    if not batch_list:
        raise LearnError("batches is empty; an empty dataset is never a training run")
    epochs = _require_int(epochs, "epochs", minimum=1)
    lr = _require_positive_float(lr, "lr")
    seed = _require_int(seed, "seed")
    memory_mode = _require_memory_mode(memory_mode)
    if max_steps is not None:
        max_steps = _require_int(max_steps, "max_steps", minimum=1)
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)):
        raise LearnError(
            f"weight_decay must be a number, got {type(weight_decay).__name__}")
    weight_decay = float(weight_decay)
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise LearnError(f"weight_decay={weight_decay!r} must be finite and >= 0")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if optimizer is None:
        parameters = [parameter for parameter in model.parameters()
                      if parameter.requires_grad]
        if not parameters:
            raise LearnError(
                "model has no trainable parameter; refusing to report an ok run "
                "that could not change a single weight")
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        optimizer_name = "AdamW"
    else:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise LearnError(
                f"optimizer must be a torch.optim.Optimizer or None, got "
                f"{type(optimizer).__name__}")
        trained = {id(parameter) for group in optimizer.param_groups
                   for parameter in group["params"]}
        if not any(id(parameter) in trained for parameter in model.parameters()):
            raise LearnError(
                "the supplied optimizer does not own any parameter of this model; "
                "it would silently train nothing")
        optimizer_name = type(optimizer).__name__

    steps: list[dict[str, Any]] = []
    step = 0
    was_training = model.training
    model.train()
    try:
        for epoch in range(epochs):
            order = list(range(len(batch_list)))
            if shuffle_batches:
                shuffled = torch.randperm(len(batch_list), generator=generator)
                order = [int(index) for index in shuffled.tolist()]
            for index in order:
                total, per_head, n_samples = _batch_step(
                    model, batch_list[index], memory_mode)
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()
                step += 1
                steps.append({
                    "step": step,
                    "epoch": epoch,
                    "loss": float(total.detach()),
                    "per_head_loss": {head: float(value.detach())
                                      for head, value in per_head.items()},
                    "n_samples": int(n_samples),
                })
                if max_steps is not None and step >= max_steps:
                    break
            if max_steps is not None and step >= max_steps:
                break
    finally:
        model.train(was_training)

    return TrainingReport(steps=tuple(steps), memory_mode=memory_mode, seed=int(seed),
                          epochs=int(epochs), lr=float(lr),
                          optimizer=optimizer_name, max_steps=max_steps,
                          shuffle_batches=bool(shuffle_batches))


# ---------------------------------------------------------------------------
# 3. checkpoints: rebuildable config, honest identity, no overwrite
# ---------------------------------------------------------------------------
def checkpoint_model_config() -> dict[str, Any]:
    """The model config stored in a checkpoint — enough to rebuild the same actor.

    ``actor.MODEL_SPEC`` is the single architecture table (``build_actor()`` takes
    no arguments), so the recorded block is that table plus the contract geometry
    it is checked against.  A checkpoint whose recorded block differs from this
    one cannot be rebuilt by this code and is refused at load.
    """

    return {
        "schema": MODEL_CONFIG_SCHEMA,
        "builder": "touhou_rl.v430.models.actor.build_actor",
        "model_spec": copy.deepcopy(actor.MODEL_SPEC),
        "head_order": list(actor.HEAD_ORDER),
        "head_sizes": {head: int(actor.HEAD_SIZES[head]) for head in actor.HEAD_ORDER},
        "joint_count": int(actor.JOINT_COUNT),
        "o2_shapes": {name: [int(dim) for dim in shape]
                      for name, shape in actor.O2_SHAPES.items()},
        "normalize_divisor": float(contracts.O2_NORMALIZE_DIVISOR),
        "hidden_size": int(actor.MODEL_SPEC["hidden_size"]),
    }


@dataclasses.dataclass(frozen=True)
class Checkpoint:
    """A loaded checkpoint: the rebuilt model, its weights and its identity."""

    path: str
    model: actor.ActorPolicy
    state_dict: dict[str, torch.Tensor]
    model_config: dict[str, Any]
    torch_version: str
    o2_digest: str
    spec_revision: str
    extra: dict[str, Any]

    @property
    def current_torch_version(self) -> str:
        """The running ``torch`` version (bitwise round-trip needs the same build)."""

        return str(torch.__version__)


def _json_ready_extra(extra: Any) -> dict[str, Any]:
    if extra is None:
        return {}
    mapping = _require_mapping(extra, "extra")
    for key in mapping:
        if not isinstance(key, str):
            raise LearnError(
                f"extra keys must be strings, got {type(key).__name__}={key!r}")
    try:
        json.dumps(dict(mapping))
    except (TypeError, ValueError) as exc:
        raise LearnError(
            f"extra must be JSON-serialisable (it is written into the checkpoint and "
            f"read back with weights_only=True): {exc}") from exc
    return copy.deepcopy(dict(mapping))


def save_checkpoint(model: Any, path: Any, *, extra: Any = None) -> str:
    """Write ``model`` + identity to ``path``; **never** overwrite an existing file.

    The payload holds the ``state_dict`` (CPU clones), the rebuildable
    :func:`checkpoint_model_config`, ``torch.__version__``, ``contracts.o2_digest()``
    and ``contracts.SPEC_REVISION``.  The file is opened with mode ``"xb"`` so the
    existence check and the write are one operation: a partly written or silently
    overwritten checkpoint cannot happen.

    Raises
    ------
    LearnError
        ``model`` is not an :class:`ActorPolicy`, the target already exists, the
        parent directory is missing, or ``extra`` is not JSON-serialisable.
    """

    if not isinstance(model, actor.ActorPolicy):
        raise LearnError(
            f"save_checkpoint needs an ActorPolicy (to record a rebuildable "
            f"model_config), got {type(model).__name__}")
    if not isinstance(path, (str, Path)):
        raise LearnError(
            f"path must be a str or a Path, got {type(path).__name__}")
    target = Path(path)
    if target.exists():
        raise LearnError(
            f"refusing to overwrite the existing checkpoint {str(target)!r}; the "
            f"CLI's --out semantics never replace an existing artefact")
    if not target.parent.is_dir():
        raise LearnError(
            f"the parent directory {str(target.parent)!r} does not exist; it is not "
            f"created silently")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model_config": checkpoint_model_config(),
        "state_dict": {name: tensor.detach().cpu().clone()
                       for name, tensor in model.state_dict().items()},
        "torch_version": str(torch.__version__),
        "o2_digest": contracts.o2_digest(),
        "spec_revision": contracts.SPEC_REVISION,
        "extra": _json_ready_extra(extra),
    }
    try:
        with open(target, "xb") as handle:
            torch.save(payload, handle)
    except FileExistsError as exc:  # pragma: no cover - the exists() check raced
        raise LearnError(
            f"refusing to overwrite the existing checkpoint {str(target)!r}") from exc
    except OSError as exc:
        raise LearnError(f"cannot write the checkpoint {str(target)!r}: {exc}") from exc
    return str(target)


def load_checkpoint(path: Any, *, map_location: Any = "cpu",
                    expected_o2_digest: str | None = None) -> Checkpoint:
    """Read a checkpoint, refusing one that belongs to a different observation/.

    Verification order (each step is a refusal, not a warning):

    1. the file exists and is a readable checkpoint of :data:`CHECKPOINT_SCHEMA`;
    2. ``o2_digest`` equals :func:`touhou_rl.v430.contracts.o2_digest` — weights
       trained under another O2 encoding cannot masquerade as these;
    3. ``spec_revision`` equals :func:`contracts.SPEC_REVISION`;
    4. ``model_config`` is exactly :func:`checkpoint_model_config`'s architecture
       table, so :func:`build_actor` really rebuilds the same structure;
    5. the state dict loads with ``strict=True`` (no missing/unexpected key).

    ``expected_o2_digest`` is an **additional** constraint, never a switch: the
    comparison against the running :func:`contracts.o2_digest` always happens
    first, so leaving the argument out never skips it and naming a foreign digest
    cannot whitelist the file.

    Bitwise identity of a saved-then-loaded forward pass holds on the same ``torch``
    build; the recorded :attr:`Checkpoint.torch_version` is the value to compare
    before claiming it across environments.
    """

    if not isinstance(path, (str, Path)):
        raise LearnError(f"path must be a str or a Path, got {type(path).__name__}")
    target = Path(path)
    if not target.is_file():
        raise LearnError(f"checkpoint {str(target)!r} does not exist")
    try:
        payload = torch.load(target, map_location=map_location, weights_only=True)
    except Exception as exc:  # noqa: BLE001 - an unreadable checkpoint is a refusal
        raise LearnError(f"cannot read the checkpoint {str(target)!r}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise LearnError(
            f"checkpoint {str(target)!r} holds {type(payload).__name__}, expected a "
            f"mapping")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise LearnError(
            f"checkpoint {str(target)!r} has schema {payload.get('schema')!r}, "
            f"expected {CHECKPOINT_SCHEMA!r}")
    for key in ("model_config", "state_dict", "torch_version", "o2_digest",
                "spec_revision"):
        if key not in payload:
            raise LearnError(
                f"checkpoint {str(target)!r} is missing the {key!r} field; an "
                f"unidentified checkpoint is never loaded")

    recorded_digest = payload["o2_digest"]
    current_digest = contracts.o2_digest()
    if recorded_digest != current_digest:
        raise LearnError(
            f"checkpoint {str(target)!r} was trained on o2_digest "
            f"{recorded_digest!r} but the current observation encoder is "
            f"{current_digest!r}; weights from another observation encoding cannot "
            f"be loaded as these")
    if expected_o2_digest is not None and recorded_digest != expected_o2_digest:
        raise LearnError(
            f"checkpoint {str(target)!r} carries o2_digest {recorded_digest!r}, not "
            f"the expected {expected_o2_digest!r}")
    if payload["spec_revision"] != contracts.SPEC_REVISION:
        raise LearnError(
            f"checkpoint {str(target)!r} was written under spec_revision "
            f"{payload['spec_revision']!r}, current is {contracts.SPEC_REVISION!r}")

    recorded_config = payload["model_config"]
    if not isinstance(recorded_config, Mapping):
        raise LearnError(
            f"checkpoint {str(target)!r} holds model_config "
            f"{type(recorded_config).__name__}, expected a mapping")
    expected_config = checkpoint_model_config()
    if recorded_config.get("schema") != expected_config["schema"]:
        raise LearnError(
            f"checkpoint {str(target)!r} has model_config schema "
            f"{recorded_config.get('schema')!r}, expected {expected_config['schema']!r}")
    if recorded_config.get("model_spec") != expected_config["model_spec"]:
        raise LearnError(
            f"checkpoint {str(target)!r} records a different MODEL_SPEC, so "
            f"build_actor() cannot rebuild the same structure; load it in the code "
            f"revision that wrote it instead of silently mismatch-loading")
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, Mapping):
        raise LearnError(
            f"checkpoint {str(target)!r} holds state_dict "
            f"{type(state_dict).__name__}, expected a mapping")
    if not isinstance(payload["extra"], Mapping):
        raise LearnError(
            f"checkpoint {str(target)!r} holds extra "
            f"{type(payload['extra']).__name__}, expected a mapping")

    model = actor.build_actor()
    try:
        model.load_state_dict(dict(state_dict), strict=True)
    except Exception as exc:  # noqa: BLE001 - a mismatching state dict is a refusal
        raise LearnError(
            f"checkpoint {str(target)!r} does not load into build_actor() with "
            f"strict=True: {exc}") from exc
    model.eval()
    return Checkpoint(
        path=str(target),
        model=model,
        state_dict={name: tensor for name, tensor in model.state_dict().items()},
        model_config=copy.deepcopy(dict(recorded_config)),
        torch_version=str(payload["torch_version"]),
        o2_digest=str(recorded_digest),
        spec_revision=str(payload["spec_revision"]),
        extra=copy.deepcopy(dict(payload["extra"])),
    )


# ---------------------------------------------------------------------------
# 4. the fail-closed entry point: validate manifest -> validate splits -> train
# ---------------------------------------------------------------------------
def _problem_codes(report: Mapping[str, Any]) -> list[str]:
    codes: list[str] = []
    for problem in report.get("problems", ()):
        code = problem.get("code") if isinstance(problem, Mapping) else None
        codes.append(str(code))
    return codes


def _report_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """The fields of a validation report that matter to a training decision."""

    summary = {
        "valid": report.get("valid"),
        "passed": report.get("passed"),
        "complete": report.get("complete"),
        "problems": _problem_codes(report),
        "checks_performed": list(report.get("checks_performed", ())),
        "exit_code": report.get("exit_code"),
    }
    return {key: value for key, value in summary.items() if value is not None}


def train_from_manifest(
    manifest: Any,
    *,
    split_config: Any,
    model: Any = None,
    batches: Any = (),
    epochs: int = 1,
    lr: float = 1e-4,
    seed: int = 0,
    memory_mode: str = "none",
    max_steps: int | None = None,
    optimizer: Any = None,
    weight_decay: float = ADAMW_WEIGHT_DECAY,
    shuffle_batches: bool = False,
    batch_source: str = "synthetic",
    run: bool = True,
    base_dir: Any = None,
    checkpoint_path: Any = None,
) -> dict[str, Any]:
    """Validate the dataset and the frozen splits, then (optionally) train.

    The order is the contract: ``validate_manifest`` first, the O2 guard
    (:func:`touhou_rl.v430.data.manifest.assert_o2_compatible`) second,
    ``validate_splits`` third, training fourth.  A ``valid=False`` manifest (a
    missing shard, a sha256 that disagrees with the bytes, a partial dataset, a
    crossing ancestor…) or a ``passed=False`` split registry (train/dev overlap, an
    unregistered source…) raises :class:`LearnError` **before** an optimizer
    exists, so no report can say "trained" for a dataset that never passed its
    gate.

    The O2 guard is what refuses a corpus recorded under another observation
    contract (this repository's older corpora are ``obs-v420-2f``): a manifest
    that merely *validates* can still not be O2 training data, and
    ``validate_manifest`` accepts any *registered* contract by name.  The guard
    runs on the ``run=False`` dry path as well, because that path also reports
    ``protocol_validity="valid"`` and would otherwise vouch for a legacy corpus.

    Parameters
    ----------
    manifest:
        A :class:`~touhou_rl.v430.data.manifest.Manifest`, its raw mapping, or a
        path to ``manifest.json`` (loaded with :func:`load_manifest`).
    split_config:
        The frozen split config: a path to ``configs/v430/splits.json`` or a
        mapping (passed straight to :func:`lineage.validate_splits`).
    run:
        ``True`` (default) trains.  ``False`` validates only: the returned result
        says ``research_result.status = "not_run"`` and ``closing / training`` is
        ``not_run`` — validation is never dressed up as a trained model.
    batches, epochs, lr, seed, memory_mode, max_steps, optimizer, weight_decay,
    shuffle_batches:
        Forwarded to :func:`train_bc` (``batches`` must be non-empty when
        ``run=True``; this slice never reads a real corpus).
    batch_source:
        The caller's declaration of where ``batches`` came from, one of
        :data:`BATCH_SOURCES` (``"synthetic"``, the default, or
        ``"captured_o2"``).  Any other value raises :class:`LearnError`.
        **This is a declaration, not a verified fact**: nothing in this module
        inspects a batch's provenance (a torch tensor carries none), so a caller
        can label captured data "synthetic" or code-generated data "captured_o2"
        and this function cannot tell.  It is recorded — in ``batch_source``,
        ``synthetic_data_only``, and ``research_result.batch_source`` /
        ``research_result.scope`` — so that a reader sees *who claimed what*, and
        ``batch_source_verified``/``batch_source_note`` say so in the result.
        ``synthetic_data_only`` is derived as
        ``batch_source == "synthetic"``.
    base_dir:
        Where relative shard paths resolve (defaults to the manifest's directory).
    checkpoint_path:
        Optional: save the trained weights there with :func:`save_checkpoint`.

    Returns
    -------
    dict
        A ``contracts.new_result`` result validated by
        ``contracts.assert_result_schema``.  It carries both validation reports,
        the training report (including the **actual** ``memory_mode``),
        ``o2_digest``, ``spec_revision``, and the declared ``batch_source`` /
        derived ``synthetic_data_only`` (with ``batch_source_verified=False`` and a
        note that the value is an unverified caller declaration).
        ``capability_gate`` stays ``not_run``: a training run is not a capability.

    Raises
    ------
    LearnError
        Either gate failed, ``batch_source`` is unregistered, or anything
        :func:`train_bc` / :func:`save_checkpoint` refuses.
    touhou_rl.v430.data.manifest.ManifestError
        Propagated **unchanged** from
        :func:`~touhou_rl.v430.data.manifest.assert_o2_compatible` when the
        manifest is not O2-compatible (a legacy ``obs_contract``, a missing
        contract, or O2 declared with disagreeing channel geometry).  It is a
        ``RuntimeError`` but deliberately **not** a :class:`LearnError`: the
        refusal belongs to the manifest layer, whose message names the actual
        contract, and re-wrapping it would lose that type for callers that catch
        it.  Both the ``run=True`` and the ``run=False`` paths raise it.
    """

    memory_mode = _require_memory_mode(memory_mode)
    # The declaration is checked before any gate: an unregistered ``batch_source``
    # is a configuration error, not something to discover after training.
    batch_source = _require_batch_source(batch_source)

    # --- gate 1: the manifest -------------------------------------------------
    manifest_input = manifest
    if isinstance(manifest_input, (str, Path)):
        try:
            manifest_input = manifest_module.load_manifest(manifest_input)
        except Exception as exc:  # noqa: BLE001 - an unreadable manifest is a refusal
            raise LearnError(
                f"cannot load the manifest {str(manifest)!r}: {exc}") from exc
    try:
        manifest_report = manifest_module.validate_manifest(
            manifest_input, base_dir=base_dir)
    except Exception as exc:  # noqa: BLE001 - never let a validator crash into a train
        raise LearnError(f"validate_manifest raised: {exc}") from exc
    if manifest_report.get("valid") is not True:
        raise LearnError(
            f"the manifest is not valid, so no training is started: "
            f"{_problem_codes(manifest_report)}")

    # --- gate 1b: the observation contract ------------------------------------
    # ``validate_manifest`` accepts any *registered* ``obs_contract``, and
    # ``obs-v420-2f`` is registered (this repository's older corpora), so a legacy
    # manifest passes gate 1.  Only this guard refuses to hand such a corpus to an
    # O2 model.  It runs before the split gate and before any training step, and
    # on the ``run=False`` path too: the dry run also reports
    # ``protocol_validity="valid"``, which would otherwise vouch for a legacy
    # corpus.  ``ManifestError`` is **not** wrapped: the refusal belongs to the
    # manifest layer and its type/message are part of that layer's contract.
    manifest_module.assert_o2_compatible(manifest_input)

    # --- gate 2: the frozen splits -------------------------------------------
    try:
        splits_report = lineage.validate_splits(split_config)
    except Exception as exc:  # noqa: BLE001
        raise LearnError(f"validate_splits raised: {exc}") from exc
    if splits_report.get("passed") is not True:
        raise LearnError(
            f"the split registry did not pass, so no training is started: "
            f"{_problem_codes(splits_report)}")

    # --- training -------------------------------------------------------------
    model_supplied = model is not None
    if run:
        if model is None:
            # ``build_actor()`` initialises from torch's default generator; a run
            # that must be reproducible has to pass a model built with an explicit
            # seed.  That fact is recorded as ``model_supplied`` in the result.
            model = actor.build_actor()
        if isinstance(batches, (str, bytes, Mapping)) or not isinstance(
                batches, Sequence) or not batches:
            raise LearnError(
                "run=True needs a non-empty sequence of synthetic batches; this "
                "slice never reads a real corpus, so an empty batch list is a "
                "configuration error, not a zero-step success")
        training = train_bc(
            model, batches, epochs=epochs, lr=lr, seed=seed, optimizer=optimizer,
            memory_mode=memory_mode, max_steps=max_steps,
            weight_decay=weight_decay, shuffle_batches=shuffle_batches)
        training_block = training.as_dict()
        saved_path = None
        if checkpoint_path is not None:
            saved_path = save_checkpoint(
                model, checkpoint_path,
                extra={"seed": int(seed), "memory_mode": memory_mode,
                       "lr": float(lr), "epochs": int(epochs),
                       "batch_source": batch_source,
                       "batch_source_verified": False})
        if saved_path is not None:
            training_block["checkpoint_path"] = saved_path
        # A completed training step is **not** a research pass: nothing was
        # measured, so the honest status is "inconclusive" (the contract's own set
        # has no "trained" value, and capability_gate stays not_run).
        research_status = "inconclusive"
    else:
        training_block = {
            "status": "not_run",
            "memory_mode": memory_mode,
            "n_steps": 0,
            "steps": [],
            "note": "run=False: the gates were evaluated, no optimizer was created "
                    "and no weight was updated",
        }
        research_status = "not_run"

    # ``batch_source`` is the caller's unverified declaration: it drives both the
    # derived flag and the scope string, so a reader never sees
    # ``synthetic_data_only=True`` (or a "synthetic-batches" scope) next to
    # ``batch_source="captured_o2"``.
    synthetic_data_only = batch_source == "synthetic"
    scope = ("bc-orchestration/synthetic-batches" if synthetic_data_only
             else "bc-orchestration/captured-o2-batches")
    batch_source_note = (
        f"declared by the caller as {batch_source!r}; this is NOT verified: "
        f"train_from_manifest cannot inspect a batch's provenance, so a caller "
        f"could label captured data synthetic or vice versa")

    research: dict[str, Any] = {
        "status": research_status,
        "scope": scope,
        "capability_claim": "none",
        "batch_source": batch_source,
        "batch_source_verified": False,
        "batch_source_note": batch_source_note,
        "synthetic_data_only": synthetic_data_only,
        "memory_mode": memory_mode,
        "model_supplied": bool(model_supplied),
        "training": training_block,
        "manifest_validation": _report_summary(manifest_report),
        "splits_validation": _report_summary(splits_report),
        "o2_digest": contracts.o2_digest(),
        "spec_revision": contracts.SPEC_REVISION,
    }
    kwargs: dict[str, Any] = {
        "execution_status": "ok",
        "protocol_validity": "valid",
        "research_result": research,
        "o2_digest": contracts.o2_digest(),
        "spec_revision": contracts.SPEC_REVISION,
        "memory_mode": memory_mode,
        "batch_source": batch_source,
        "batch_source_verified": False,
        "batch_source_note": batch_source_note,
        "synthetic_data_only": synthetic_data_only,
    }
    # ``command_class`` is a contracts-owned result field: only pass it as an extra
    # when the contract does not already treat it as a core field.
    if "command_class" not in contracts.RESULT_FIELDS:
        kwargs["command_class"] = "training"
    result = contracts.new_result(**kwargs)
    return contracts.assert_result_schema(result)
