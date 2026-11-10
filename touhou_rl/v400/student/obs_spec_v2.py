"""Frozen observation spec **obs-v2** for the v4.0.0 pixel student (P3-WP7 / EXP-P3-008).

obs-v2 is **additive**: every byte of obs-v1 (:mod:`touhou_rl.v400.student.obs_spec`)
stays exactly as frozen — ``play64`` 4-frame stack, HUD strip, normalisation, action
table, alignment.  obs-v2 only *appends* inputs, so an obs-v1 checkpoint still loads
and an obs-v1 sample is still a valid obs-v2 sample with the new fields defaulted.

The single addition is the **8-decision action history** (``act_hist``), 8 rows of
18 one-hot move-focus candidates, oldest first::

    row 0 = action executed at decision t-8  (oldest)
    row 7 = action executed at decision t-1  (the decision immediately before the
            observation instant, i.e. what the policy did *last*)

Why 8 *past* decisions and never the current one: the label at decision ``t`` is the
action applied at ``t`` (obs-v1 §2: ``frames[i-3..i] -> mask[i+1]``).  A history that
contained ``a_t`` would leak the target into its own input, so the window is strictly
``a_{t-8} .. a_{t-1}``.  That is also the only window available at deployment, where
the policy has emitted exactly those past actions.

Two inputs from the task card are **deliberately not included**, and the reason is a
property of the data, not a preference:

``self_state`` (player ``px/py``)
    The D0/DAgger label rows carry no player coordinates — the teacher label is
    ``scores[18] + verified_mask + bomb + chosen_plan`` and ``chosen_plan`` stores the
    plan's *end* xy, not the player's.  Recovering the player position from ``play64``
    is a perception problem P2 has not solved (EXP-P2-003/004) and would be a new
    estimator smuggled into the observation.  So obs-v2 does not contain self state;
    a future ``obs-v3`` that has a *verified* position source can add it.

``danger_summary`` (lives/bombs deltas)
    The obs-v1 HUD strip is a picture of the counters, not the counters.  There is no
    HUD digit reader in the tree (``touhou_rl/v400/labels/hud.py`` deliberately labels
    the *state* the HUD draws, not the render), so a lives/bombs delta cannot be
    produced from pixels on the deployment side.  Reading it from the engine's ``HUD``
    protocol reply instead would silently change the eval's ``deploy_pixel`` privilege
    contract (``v400_student_eval.py``), which is a bigger change than obs-v2 is
    allowed to make.  The field is therefore *reserved and not populated*; the digest
    records that it is absent so a checkpoint cannot claim to have seen it.

:func:`obs_spec_v2_digest` is the sha256 of the canonical JSON of :func:`obs_spec_v2`;
it is stored in every obs-v2 checkpoint and every obs-v2 shard, and
:func:`check_obs_spec_v2` fails closed on a different digest.  obs-v1's digest
(``24f0ddac…``) is untouched, and :data:`OBS_V1_DIGEST` is asserted here so a future
edit to obs-v1 that changes its meaning breaks this import chain, not a training run.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

import numpy as np

from touhou_rl.v400.student import obs_spec

OBS_SPEC_VERSION: Final = "obs-v2"
#: The frozen v1 digest obs-v2 supersets.  Literal on purpose: obs-v2 must name the
#: exact obs-v1 bytes it extends, so a silent v1 change is a hard error here.
OBS_V1_DIGEST: Final = "24f0ddace59104938dbe991c0d8efbb6a8cd21dfd5a29ebcb6d023b5b364a381"

#: Number of past decisions kept in the action history (task card: "最近 8 个决策").
ACTION_HISTORY_DECISIONS: Final = 8
#: One entry per decision — no frame-level history, so the window is unambiguous.
ACTION_HISTORY_STRIDE_DECISIONS: Final = 1
#: ``cand = focus*9 + move`` (18 classes); the bomb bit is *not* in the history.
ACTION_HISTORY_DIM: Final = ACTION_HISTORY_DECISIONS * obs_spec.N_MOVE_CLASSES
ACTION_HISTORY_ORDER: Final = "oldest-first a[t-8],...,a[t-1]"
#: A row of all zeros is "no recorded action" (episode start / stage-clear boundary
#: pad).  It is not a valid candidate, so it cannot alias action 0 (which is one-hot).
ACTION_HISTORY_PAD: Final = "all-zero row"
ACTION_HISTORY_DTYPE: Final = "uint8"
#: Deployment-side zero padding repeats the *first* action row backwards, same idea as
#: obs-v1's frame padding: deterministic, recorded, never invented motion.
ACTION_HISTORY_PADDING_RULE: Final = (
    "before the first action of an episode the window is all-zero rows; after the "
    "shard/stage boundary the earlier rows are zero, never the previous episode's"
)

#: The two reserved inputs (see the module docstring).  ``present`` is ``False`` in
#: the frozen digest, so a checkpoint trained on a future obs-v3 cannot load here.
SELF_STATE_PRESENT: Final = False
DANGER_SUMMARY_PRESENT: Final = False
SELF_STATE_REASON: Final = (
    "no player px/py in the D0/DAgger labels (teacher label = scores/verified_mask/"
    "bomb/chosen_plan; chosen_plan.end_xy is the plan endpoint, not the player); "
    "recovering it from play64 is unsolved P2 perception, so obs-v2 omits self state"
)
DANGER_SUMMARY_REASON: Final = (
    "lives/bombs exist only as HUD pixels in the deployment observation; there is no "
    "HUD digit reader (labels/hud.py labels the drawn state, not the render) and "
    "reading the engine's HUD reply would change the deploy_pixel privilege contract, "
    "so the field is reserved and not populated"
)


class ObsSpecV2Error(ValueError):
    """Raised when an observation does not match obs-v2."""


class ObsSpecV2Mismatch(ObsSpecV2Error):
    """Raised when a checkpoint/shard was produced under a different obs spec."""


def _base_spec() -> dict:
    """The frozen obs-v1 payload, copied so obs-v2 cannot mutate it."""

    return json.loads(json.dumps(obs_spec.obs_spec()))


def obs_spec_v2() -> dict[str, object]:
    """The frozen obs-v2 spec as JSON-serialisable data (the digest covers this)."""

    payload = _base_spec()
    base_digest = obs_spec.obs_spec_digest()
    if base_digest != OBS_V1_DIGEST:  # pragma: no cover - guard, never expected
        raise ObsSpecV2Mismatch(
            f"obs-v1 digest changed to {base_digest!r}; obs-v2 supersets "
            f"{OBS_V1_DIGEST!r} and cannot silently follow a new v1"
        )
    payload["version"] = OBS_SPEC_VERSION
    payload["supersedes"] = {"version": obs_spec.OBS_SPEC_VERSION, "digest": base_digest}
    payload["action_history"] = {
        "field": "act_hist",
        "decisions": ACTION_HISTORY_DECISIONS,
        "stride_decisions": ACTION_HISTORY_STRIDE_DECISIONS,
        "order": ACTION_HISTORY_ORDER,
        "encoding": "one-hot over the 18 move-focus candidates (cand = focus*9 + move)",
        "dims": ACTION_HISTORY_DIM,
        "dtype": ACTION_HISTORY_DTYPE,
        "pad": ACTION_HISTORY_PAD,
        "padding_rule": ACTION_HISTORY_PADDING_RULE,
        "excludes_current_action": True,
        "bomb_bit_in_history": False,
        "reconstruction": {
            "D0": "chosen_plan.first_action (the action the teacher committed at that "
                  "decision; D0 is a teacher-driven shard)",
            "DAgger": "row.dagger.executed_cand (the action the env actually applied, "
                      "which is the student's greedy action unless the teacher injected)",
            "boundary": "cleared at the shard start and at any decision whose stage "
                        "differs from the previous row (stage clear = episode change)",
        },
        "deployment": "ActionHistoryBuffer.push(cand) after each emitted decision; "
                      "all-zero rows before the first action",
    }
    payload["self_state"] = {
        "present": SELF_STATE_PRESENT,
        "reason": SELF_STATE_REASON,
    }
    payload["danger_summary"] = {
        "present": DANGER_SUMMARY_PRESENT,
        "reason": DANGER_SUMMARY_REASON,
        "reserved_fields": ["lives_delta[4]", "bombs_delta[4]"],
    }
    payload["excluded_inputs"] = [
        item for item in obs_spec.EXCLUDED_INPUTS
        if "action history" not in item
    ] + ["recent action history (moved into obs-v2 as act_hist)"]
    return payload


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def obs_spec_v2_digest() -> str:
    """sha256 of the canonical obs-v2 spec; stored in every obs-v2 artefact."""

    return hashlib.sha256(_canonical(obs_spec_v2())).hexdigest()


def check_obs_spec_v2(version: str | None, digest: str | None) -> None:
    """Fail closed if a checkpoint/shard was produced under another spec."""

    if version != OBS_SPEC_VERSION:
        raise ObsSpecV2Mismatch(
            f"obs spec version {version!r} is not {OBS_SPEC_VERSION!r}; "
            "an obs-v1 checkpoint cannot be evaluated as obs-v2 and vice versa"
        )
    expected = obs_spec_v2_digest()
    if digest != expected:
        raise ObsSpecV2Mismatch(
            f"obs spec digest {digest!r} does not match {expected!r}")


# ---------------------------------------------------------------------------
# action history: encoding, reconstruction, deployment buffer
# ---------------------------------------------------------------------------
def action_one_hot(cand: int) -> np.ndarray:
    """``cand (0..17)`` -> one-hot ``uint8[18]``; rejects out-of-range actions."""

    value = int(cand)
    if not 0 <= value < obs_spec.N_MOVE_CLASSES:
        raise ObsSpecV2Error(
            f"action history candidate {cand!r} is not in [0,"
            f"{obs_spec.N_MOVE_CLASSES})")
    row = np.zeros(obs_spec.N_MOVE_CLASSES, dtype=np.uint8)
    row[value] = 1
    return row


def build_action_history(
    executed_cand: np.ndarray,
    stage: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct ``(N,8,18)`` uint8 history from one shard's executed actions.

    Row ``t`` holds the one-hot actions of decisions ``t-8..t-1`` (oldest first).
    The window is cleared at:

    * the shard start (``t < 8``: earlier rows are zero), and
    * any decision ``t`` whose ``stage`` differs from ``stage[t-1]`` — a stage
      clear ends the episode, so the previous stage's actions are a different
      trajectory and must not be carried across (task card: "换面/分片边界要清零").

    ``executed_cand`` is the action actually applied by the env, not the teacher's
    preferred action; :func:`reconstruct_from_label_rows` extracts it.
    """

    cand = np.asarray(executed_cand)
    if cand.ndim != 1:
        raise ObsSpecV2Error(f"executed_cand must be 1-D, got {cand.shape}")
    count = cand.shape[0]
    if count and (cand.min() < 0 or cand.max() >= obs_spec.N_MOVE_CLASSES):
        raise ObsSpecV2Error(
            f"executed_cand must be in [0,{obs_spec.N_MOVE_CLASSES}), got "
            f"[{cand.min()},{cand.max()}]")
    history = np.zeros((count, ACTION_HISTORY_DECISIONS, obs_spec.N_MOVE_CLASSES),
                       dtype=np.uint8)
    if count == 0:
        return history
    if stage is not None:
        stages = np.asarray(stage)
        if stages.shape != cand.shape:
            raise ObsSpecV2Error(
                f"stage shape {stages.shape} != executed_cand shape {cand.shape}")
        boundaries = np.zeros(count, dtype=bool)
        boundaries[0] = True
        boundaries[1:] = stages[1:] != stages[:-1]
    else:
        boundaries = np.zeros(count, dtype=bool)
        boundaries[0] = True
    segment_start = 0
    for t in range(count):
        if boundaries[t]:
            segment_start = t
        first = max(segment_start, t - ACTION_HISTORY_DECISIONS)
        # rows are filled oldest-first ending at t-1
        for offset, source in enumerate(range(t - 1, first - 1, -1)):
            history[t, ACTION_HISTORY_DECISIONS - 1 - offset] = action_one_hot(
                int(cand[source]))
    return history


def reconstruct_from_label_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """``[(label row, ...)]`` -> ``(executed_cand[N], stage[N])`` for one shard.

    The executed action is read from the shard kind, because the two shard kinds
    record different things:

    * a DAgger row carries ``dagger.executed_cand`` — the action the env applied
      (student greedy unless the teacher injected), which is what the trajectory
      actually did;
    * a D0 row carries no ``dagger`` block, and the shard is teacher-driven, so the
      executed action is ``chosen_plan.first_action`` (the plan's first slot, the
      action committed at that decision).

    A row with neither is refused rather than guessed.
    """

    cand = np.full(len(rows), -1, dtype=np.int16)
    stage = np.zeros(len(rows), dtype=np.int32)
    for index, row in enumerate(rows):
        dagger = row.get("dagger") or {}
        if "executed_cand" in dagger:
            cand[index] = int(dagger["executed_cand"])
            if "executed_bomb" in dagger:
                pass  # bomb is a separate head target, not part of act_hist
        else:
            plan = row.get("chosen_plan") or {}
            if "first_action" not in plan:
                raise ObsSpecV2Error(
                    f"decision {index}: neither dagger.executed_cand nor "
                    "chosen_plan.first_action exists; the executed action cannot be "
                    "reconstructed without guessing")
            cand[index] = int(plan["first_action"])
        stage[index] = int(row.get("stage", 0))
    if len(cand) and cand.min() < 0:
        raise ObsSpecV2Error("some decisions have no reconstructible executed action")
    return cand, stage


def check_action_history(history: np.ndarray) -> np.ndarray:
    """Validate an ``(N,8,18)`` history array as uint8 rows that are one-hot or zero."""

    array = np.asarray(history)
    if array.ndim != 3 or array.shape[1:] != (ACTION_HISTORY_DECISIONS,
                                              obs_spec.N_MOVE_CLASSES):
        raise ObsSpecV2Error(
            f"act_hist shape {array.shape} != (N,{ACTION_HISTORY_DECISIONS},"
            f"{obs_spec.N_MOVE_CLASSES})")
    if array.dtype != np.uint8:
        raise ObsSpecV2Error(f"act_hist dtype {array.dtype} is not uint8")
    sums = array.sum(axis=-1)
    if int(sums.max(initial=0)) > 1 or int(array.max(initial=0)) > 1:
        raise ObsSpecV2Error("act_hist rows must be one-hot or all-zero")
    if np.any((sums == 0) & (array.any(axis=-1))):  # pragma: no cover - defensive
        raise ObsSpecV2Error("act_hist has a row with several ones")
    return array


def normalize_action_history(history: np.ndarray) -> np.ndarray:
    """``uint8 (8,18)`` -> ``float32 (144,)`` in [0,1]; no mean/std subtractions."""

    array = check_action_history(np.asarray(history)[None, ...])[0]
    return array.reshape(ACTION_HISTORY_DIM).astype(np.float32)


class ActionHistoryBuffer:
    """Deployment-side rolling history of the actions the policy itself emitted.

    Mirrors :class:`touhou_rl.v400.student.obs_spec.ObservationBuffer`: ``reset()``
    at episode start, ``push(cand)`` once per decision *after* the action is applied,
    and :meth:`rows` returns the ``(8,18)`` uint8 window that obs-v2 feeds the model.
    A stage clear also calls :meth:`reset` because it starts a new episode.
    """

    def __init__(self) -> None:
        self._actions: list[int] = []
        self.pushes = 0

    def reset(self) -> None:
        self._actions = []
        self.pushes = 0

    def push(self, cand: int) -> None:
        value = int(cand)
        if not 0 <= value < obs_spec.N_MOVE_CLASSES:
            raise ObsSpecV2Error(f"cannot push candidate {cand!r}")
        self._actions.append(value)
        if len(self._actions) > ACTION_HISTORY_DECISIONS:
            del self._actions[0]
        self.pushes += 1

    def rows(self) -> np.ndarray:
        window = np.zeros((ACTION_HISTORY_DECISIONS, obs_spec.N_MOVE_CLASSES),
                          dtype=np.uint8)
        actions = self._actions[-ACTION_HISTORY_DECISIONS:]
        for offset, action in enumerate(actions):
            window[ACTION_HISTORY_DECISIONS - len(actions) + offset] = action_one_hot(action)
        return window

    def normalized(self) -> np.ndarray:
        return normalize_action_history(self.rows())


def history_digest(history: np.ndarray) -> str:
    """sha256 of one history array's bytes (per-shard provenance in the index)."""

    array = np.ascontiguousarray(np.asarray(history, dtype=np.uint8))
    return hashlib.sha256(array.tobytes()).hexdigest()


__all__ = [
    "ACTION_HISTORY_DECISIONS",
    "ACTION_HISTORY_DIM",
    "ACTION_HISTORY_DTYPE",
    "ACTION_HISTORY_ORDER",
    "ACTION_HISTORY_PAD",
    "ACTION_HISTORY_STRIDE_DECISIONS",
    "DANGER_SUMMARY_PRESENT",
    "OBS_SPEC_VERSION",
    "OBS_V1_DIGEST",
    "SELF_STATE_PRESENT",
    "ActionHistoryBuffer",
    "ObsSpecV2Error",
    "ObsSpecV2Mismatch",
    "action_one_hot",
    "build_action_history",
    "check_action_history",
    "check_obs_spec_v2",
    "history_digest",
    "normalize_action_history",
    "obs_spec_v2",
    "obs_spec_v2_digest",
    "reconstruct_from_label_rows",
]
