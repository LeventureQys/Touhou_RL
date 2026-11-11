"""Frozen observation spec **obs-v3** for the v4.0.0 pixel student (T2 closeout / T3).

obs-v3 is the combination the 2026-11-10 ruling (D-14) selected after the T2a
perception probe (EXP-P3-010) failed its pre-registered gate:

* the **play/HUD conventions of obs-v1** stay exactly as frozen — integer BT.601
  luminance ``(77R + 151G + 28B) >> 8``, the same 640x480 readback, the same
  ``(32,16)`` 384x448 playfield viewport, the same 56x160 green HUD strip, the same
  4-frame stack ``t-3..t`` oldest-first, the same ``float32 / 255.0`` normalisation,
  the same 18 move-focus x 2 bomb action table;
* the **playfield plane is 96x112 instead of 64x64** — a 4x4 *integer* block mean of
  the native 384x448 field (``384/96 = 4`` columns, ``448/112 = 4`` rows), which is
  byte-for-byte the T2a input **B**
  (:func:`touhou_rl.v400.student.probe.native.encode_native` key ``play96x112``);
* the **``act_hist`` of obs-v2** is appended: 8 rows x 18 one-hot columns holding
  ``a[t-8] .. a[t-1]`` (oldest first, strictly past actions, cleared at the shard
  start and at every stage change).

Why 96x112 and not 64x64 (the obs-v1 choice): EXP-P3-010 measured the occupancy F1
inside 48 px of the ship on held-out seeds 15-16.  Every candidate stayed below the
pre-registered ``0.90`` gate, so the T2 stop-loss fired and the负责人 ruled on
2026-11-10 (D-14): adopt **B**, keep the occupancy F1 only as a recorded proxy, and
move the acceptance criterion to the **closed loop** (G3 ``S1N`` clear rate).  B's
F1 (0.8502 on the pre-registered metric, 0.8661 at the best post-hoc threshold) is a
**known limitation**, not a passed gate; it is written into the digest so a reader
cannot mistake it for one.

The digest (:func:`obs_spec_v3_digest`) is the sha256 of the canonical JSON of
:func:`obs_spec_v3`; it is stored in every obs-v3 shard and checkpoint and
:func:`check_obs_spec_v3` fails closed on a different digest.  obs-v1 and obs-v2 are
**not touched**: their digests (``24f0ddac…`` / ``95110743…``) and their file bytes
are asserted as literals here, so an edit to either frozen file breaks this import
chain instead of a training run.

Deliberately **not** in obs-v3 (identical to obs-v2, reasons unchanged):
``self_state`` (no verified pixel player position; the T2a locator's median error is
11.59 px) and ``danger_summary`` (lives/bombs exist only as HUD pixels at deployment
time).  The one *added* policy-facing label is not an input at all: the teacher's
``verified_mask[18]`` is the "verified surviving action set" used from T3/T4 on as
the auxiliary label of the closed-loop criterion.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

import numpy as np

from touhou_rl.v400.student import obs_spec, obs_spec_v2

OBS_SPEC_VERSION: Final = "obs-v3"

#: Frozen obs-v1 spec digest obs-v3 keeps the play/HUD conventions of.
OBS_V1_DIGEST: Final = "24f0ddace59104938dbe991c0d8efbb6a8cd21dfd5a29ebcb6d023b5b364a381"
#: Frozen obs-v2 spec digest obs-v3 takes ``act_hist`` from.
OBS_V2_DIGEST: Final = "95110743993c090e2b30fa40ac349a350e74021b35464e3204c784a69ae4dba3"
#: sha256 of the *file bytes* of the two frozen modules.  Literals on purpose: the
#: ruling is "obs-v1/obs-v2 do not move", and a byte is stricter than a digest.
OBS_V1_FILE_SHA256: Final = (
    "0d47daf1bddc2b6fb8a33eb54a558f2d40f4a8c3b3d878155035d914d455578b")
OBS_V2_FILE_SHA256: Final = (
    "388b9bd86625200cda8355e4f2c3ddff8deaa993f5fab3a71946dd762a7b6370")

#: A student decision is 2 logic frames (``obs_spec.DECISION_FRAMES``); a D0 CLI
#: decision commits ``2 x 2 = 4`` logic frames and labels only its first slot.
DECISION_FRAMES: Final = obs_spec.DECISION_FRAMES
#: Stacked playfield frames ``t-3 .. t`` (oldest first) — unchanged from obs-v1.
FRAME_STACK: Final = obs_spec.FRAME_STACK
STACK_ORDER: Final = obs_spec.STACK_ORDER

#: Play plane name/shape/dtype.  The name is the T2a input name, the shape is the
#: *array* layout ``(rows, cols) = (height, width)``: the plane is 96 px wide and
#: 112 px tall.  ``PLAY_EXTENT_WH`` spells the same plane in (width, height) order
#: because that is how every card and the ruling name it ("96x112").
PLAY_PLANE: Final = "play96x112"
PLAY_SHAPE: Final[tuple[int, int]] = (112, 96)
PLAY_EXTENT_WH: Final[tuple[int, int]] = (96, 112)
#: Integer block mean geometry: 4 rows x 4 columns of native pixels per plane pixel.
PLAY_BLOCK: Final[tuple[int, int]] = (4, 4)
PLAY_PIXELS: Final = PLAY_SHAPE[0] * PLAY_SHAPE[1]  # 10,752 per frame

#: HUD is byte-identical to obs-v1 (same bands, same native resolution, same frame).
HUD_PLANE: Final = obs_spec.HUD_PLANE
HUD_SHAPE: Final[tuple[int, int]] = obs_spec.HUD_SHAPE
HUD_FRAMES: Final = obs_spec.HUD_FRAMES
HUD_BANDS: Final[tuple[tuple[int, int], ...]] = obs_spec.HUD_BANDS
HUD_X0: Final = obs_spec.HUD_X0
HUD_X1: Final = obs_spec.HUD_X1
HUD_PIXELS: Final = HUD_SHAPE[0] * HUD_SHAPE[1]  # 8,960 per decision

#: Encoder constants — the frozen obs-v1 ones, restated so this module is readable
#: on its own (the values are asserted equal to ``obs_spec`` in the tests).
GRAY_WEIGHTS: Final[tuple[int, int, int]] = obs_spec.GRAY_WEIGHTS
GRAY_SHIFT: Final = obs_spec.GRAY_SHIFT
PLAYFIELD_ORIGIN: Final[tuple[int, int]] = obs_spec.PLAYFIELD_ORIGIN
PLAYFIELD_SHAPE: Final[tuple[int, int]] = obs_spec.PLAYFIELD_SHAPE
NORMALIZATION: Final = obs_spec.NORMALIZATION

#: Action history: obs-v2's exact contract (8 past decisions x 18 move-focus cands).
ACTION_HISTORY_DECISIONS: Final = obs_spec_v2.ACTION_HISTORY_DECISIONS
ACTION_HISTORY_DIM: Final = obs_spec_v2.ACTION_HISTORY_DIM
ACTION_HISTORY_ORDER: Final = obs_spec_v2.ACTION_HISTORY_ORDER
ACTION_HISTORY_DTYPE: Final = obs_spec_v2.ACTION_HISTORY_DTYPE
ACTION_HISTORY_PAD: Final = obs_spec_v2.ACTION_HISTORY_PAD
ACTION_HISTORY_SHAPE: Final[tuple[int, int]] = (ACTION_HISTORY_DECISIONS,
                                                obs_spec.N_MOVE_CLASSES)
ACTION_HISTORY_BYTES: Final = ACTION_HISTORY_DECISIONS * obs_spec.N_MOVE_CLASSES  # 144

#: Bytes a *v3 shard* stores per decision (planes only, labels excluded): the
#: number T3's disk estimate uses.
BYTES_PER_DECISION: Final = (FRAME_STACK * PLAY_PIXELS + HUD_PIXELS
                             + ACTION_HISTORY_BYTES)  # 52,112

#: T2a (EXP-P3-010) results as recorded, so the freeze document and the code agree.
#: ``f1_preregistered`` = micro-F1 at threshold 0.5 on seeds 15-16, mean over the 3
#: training seeds (the pre-registered metric).  ``f1_best_threshold`` = the post-hoc
#: max-aggregation / threshold sweep in ``.tmp_v400_p3diag/aggregate.json`` (the
#: numbers the负责人 quoted); it does **not** replace the pre-registered metric.
PROBE_RESULTS: Final[dict[str, dict[str, object]]] = {
    "A": {"input": "64x64 whole field (obs-v1 plane)",
          "f1_preregistered": 0.563262, "f1_best_threshold": 0.628991,
          "selected": False},
    "B": {"input": "96x112 whole field (4x4 integer block mean)",
          "f1_preregistered": 0.850197, "f1_best_threshold": 0.866093,
          "selected": True},
    "C": {"input": "192x224 whole field (2x2 integer block mean)",
          "f1_preregistered": 0.839214, "f1_best_threshold": 0.853958,
          "selected": False},
    "D": {"input": "64x64 whole field + 128x128 native local window",
          "f1_preregistered": 0.739720, "f1_best_threshold": 0.745797,
          "locator_median_px": 11.591909, "locator_gate_px": 4.0,
          "selected": False,
          "rejected_because": "pixel player locator median error 11.59 px > 4 px gate"},
}
#: Every candidate's F1 stayed under the pre-registered gate.
PROBE_GATE_F1: Final = 0.90
PROBE_SELECTION_RULE: Final = (
    "smallest input with F1 >= 0.90 and F1 >= 0.95 * F_max; input D additionally "
    "needs the pixel locator median error <= 4 px")
PROBE_OUTCOME: Final = (
    "no candidate reached F1 >= 0.90, so the T2 stop-loss condition fired")
#: The ruling that replaced the F1 gate (31-决策记录 D-14).
RULING_ID: Final = "D-14"
RULING_DATE: Final = "2026-11-10"
RULING_DECISION: Final = (
    "adopt B (96x112) and replace the 'occupancy F1 >= 0.90' acceptance criterion "
    "with the closed-loop score; the occupancy F1 stays recorded as a proxy only")
RULING_CRITERION: Final = (
    "closed loop: G3 S1N clear rate (stage 1 normal, confirmation seeds), not the "
    "probe occupancy F1")
KNOWN_LIMITATION: Final = (
    "B's occupancy F1 is 0.8502 (pre-registered metric) / 0.8661 (best post-hoc "
    "threshold) and therefore < 0.90; this is a recorded known limitation of "
    "obs-v3, not a passed gate")

#: The policy-facing auxiliary label the ruling asked for (T3/T4).  It is not an
#: input: the teacher's own ``verified_mask`` already carries it, per decision.
POLICY_AUX_LABEL: Final[dict[str, object]] = {
    "name": "verified_surviving_action_set",
    "source": "TeacherLabel.verified_mask[18] (11-P1 §3), already in every "
              "labels.jsonl row; no extra plane is stored",
    "definition": "cand c belongs to the set iff verified_mask[c] is True, i.e. the "
                  "teacher verified that choosing c at this decision survives its "
                  "search horizon",
    "consumer": "T4/EXP-P3-011 danger-state agreement: on states with teacher margin "
                "< 8 px, the share of student argmax inside this set",
    "supports": "the D-14 closed-loop criterion (occupancy F1 is only a proxy)",
    "not_a_dense_target": "it is a mask, not a probability; training it as a BCE "
                          "target needs an explicitly registered loss in T4",
}
SELF_STATE_PRESENT: Final = obs_spec_v2.SELF_STATE_PRESENT
DANGER_SUMMARY_PRESENT: Final = obs_spec_v2.DANGER_SUMMARY_PRESENT


class ObsSpecV3Error(ValueError):
    """Raised when an observation does not match obs-v3."""


class ObsSpecV3Mismatch(ObsSpecV3Error):
    """Raised when a checkpoint/shard was produced under a different obs spec."""


def _frozen_v2() -> dict:
    """The frozen obs-v2 payload, copied so obs-v3 cannot mutate it."""

    return json.loads(json.dumps(obs_spec_v2.obs_spec_v2()))


def _play_block() -> dict[str, object]:
    return {
        "plane": PLAY_PLANE,
        "shape": list(PLAY_SHAPE),
        "shape_note": "array layout (rows, cols) = (height, width); the plane is "
                      "96 px wide x 112 px tall",
        "extent_wh": list(PLAY_EXTENT_WH),
        "dtype": "uint8",
        "channels": 1,
        "content": "BT.601 integer luminance of the 384x448 playfield, 4x4 integer "
                   "block mean (sum // 16) at native 96x112",
        "source_region": {
            "screen_origin": list(PLAYFIELD_ORIGIN),
            "playfield": list(PLAYFIELD_SHAPE),
        },
        "block": list(PLAY_BLOCK),
        "pixels_per_frame": PLAY_PIXELS,
        "stack": FRAME_STACK,
        "stack_order": STACK_ORDER,
        "probe_equivalence": (
            "byte-identical to EXP-P3-010 input B "
            "(probe.native.encode_native(...)['play96x112'], "
            "probe.inputs.build_input('B', ...) / 255.0)"),
        "chosen_because": "D-14: T2a (EXP-P3-010) selected input B and replaced the "
                          "F1 gate with the closed-loop criterion",
    }


def obs_spec_v3() -> dict[str, object]:
    """The frozen obs-v3 spec as JSON-serialisable data (the digest covers this)."""

    v2 = _frozen_v2()
    if obs_spec.obs_spec_digest() != OBS_V1_DIGEST:  # pragma: no cover - guard
        raise ObsSpecV3Mismatch(
            f"obs-v1 digest changed to {obs_spec.obs_spec_digest()!r}; obs-v3 keeps "
            f"the frozen {OBS_V1_DIGEST!r} play/HUD conventions")
    if obs_spec_v2.obs_spec_v2_digest() != OBS_V2_DIGEST:  # pragma: no cover - guard
        raise ObsSpecV3Mismatch(
            f"obs-v2 digest changed to {obs_spec_v2.obs_spec_v2_digest()!r}; obs-v3 "
            f"takes the frozen {OBS_V2_DIGEST!r} action history")
    payload = json.loads(json.dumps(obs_spec.obs_spec()))
    payload["version"] = OBS_SPEC_VERSION
    payload["supersedes"] = [
        {"version": obs_spec_v2.OBS_SPEC_VERSION, "digest": OBS_V2_DIGEST},
        {"version": obs_spec.OBS_SPEC_VERSION, "digest": OBS_V1_DIGEST},
    ]
    payload["play"] = _play_block()
    payload["action_history"] = v2["action_history"]
    payload["self_state"] = v2["self_state"]
    payload["danger_summary"] = v2["danger_summary"]
    payload["excluded_inputs"] = v2["excluded_inputs"]
    payload["resolution"] = {
        "probe": "EXP-P3-010 (T2a, 2026-11-10)",
        "decision": "B: 96x112 whole field, 4x4 integer block mean of the native "
                    "384x448 playfield",
        "candidates": PROBE_RESULTS,
        "gate_f1": PROBE_GATE_F1,
        "pre_registered_rule": PROBE_SELECTION_RULE,
        "outcome": PROBE_OUTCOME,
        "ruling": {
            "id": RULING_ID,
            "date": RULING_DATE,
            "decision": RULING_DECISION,
            "criterion": RULING_CRITERION,
            "known_limitation": KNOWN_LIMITATION,
        },
    }
    payload["policy_aux_label"] = POLICY_AUX_LABEL
    payload["bytes_per_decision"] = {
        "play": FRAME_STACK * PLAY_PIXELS,
        "hud": HUD_PIXELS,
        "act_hist": ACTION_HISTORY_BYTES,
        "total": BYTES_PER_DECISION,
        "note": "planes only; labels.jsonl is counted separately",
    }
    return payload


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def obs_spec_v3_digest() -> str:
    """sha256 of the canonical obs-v3 spec; stored in every obs-v3 artefact."""

    return hashlib.sha256(_canonical(obs_spec_v3())).hexdigest()


def check_obs_spec_v3(version: str | None, digest: str | None) -> None:
    """Fail closed if a checkpoint/shard was produced under another spec."""

    if version != OBS_SPEC_VERSION:
        raise ObsSpecV3Mismatch(
            f"obs spec version {version!r} is not {OBS_SPEC_VERSION!r}; an obs-v1/"
            "obs-v2 artefact cannot be read as obs-v3 and vice versa")
    expected = obs_spec_v3_digest()
    if digest != expected:
        raise ObsSpecV3Mismatch(
            f"obs spec digest {digest!r} does not match {expected!r}")


# ---------------------------------------------------------------------------
# encoder: 640x480 RGBA readback -> (play96x112, hud)
# ---------------------------------------------------------------------------
def _check_readback(pixels: np.ndarray) -> None:
    if not isinstance(pixels, np.ndarray):
        raise ObsSpecV3Error(f"pixels must be a numpy array, got {type(pixels)!r}")
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ObsSpecV3Error(
            f"pixels must be uint8 HxWx4, got {pixels.dtype} {pixels.shape}")


def encode_planes_v3(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """640x480 RGBA8 readback -> ``(play96x112, hud)`` uint8 planes.

    Byte-exact with EXP-P3-010's input B: integer BT.601 luminance
    ``(77R + 151G + 28B) >> 8`` over the ``(32,16)`` 384x448 viewport, then a
    ``sum // 16`` 4x4 block mean.  The HUD is the *same* green-channel strip
    obs-v1 stores, so a v3 shard and a v1 shard agree on their ``hud`` bytes.
    """

    _check_readback(pixels)
    x0, y0 = PLAYFIELD_ORIGIN
    width, height = PLAYFIELD_SHAPE
    crop = pixels[y0:y0 + height, x0:x0 + width]
    if crop.shape[0] != height or crop.shape[1] != width:
        raise ObsSpecV3Error(
            f"readback {pixels.shape} does not contain the {width}x{height} playfield")
    red = crop[:, :, 0].astype(np.uint16)
    green = crop[:, :, 1].astype(np.uint16)
    blue = crop[:, :, 2].astype(np.uint16)
    lum = ((red * GRAY_WEIGHTS[0] + green * GRAY_WEIGHTS[1] + blue * GRAY_WEIGHTS[2])
           >> GRAY_SHIFT)
    block_h, block_w = PLAY_BLOCK
    if (height % block_h or width % block_w) or (height // block_h != PLAY_SHAPE[0]
                                                or width // block_w != PLAY_SHAPE[1]):
        raise ObsSpecV3Error(  # pragma: no cover - constants are frozen
            f"{width}x{height} does not divide into {PLAY_SHAPE} by {PLAY_BLOCK}")
    play = (lum.reshape(PLAY_SHAPE[0], block_h, PLAY_SHAPE[1], block_w)
            .sum(axis=(1, 3)) // (block_h * block_w)).astype(np.uint8)
    if play.shape != PLAY_SHAPE:  # pragma: no cover - shapes are frozen
        raise ObsSpecV3Error(f"play plane shape {play.shape} != {PLAY_SHAPE}")

    hud_rows = [pixels[a:b, HUD_X0:HUD_X1, 1] for (a, b) in HUD_BANDS]
    hud = np.concatenate(hud_rows, axis=0)
    if hud.shape != HUD_SHAPE:  # pragma: no cover - shapes are frozen
        raise ObsSpecV3Error(f"hud plane shape {hud.shape} != {HUD_SHAPE}")
    return play, hud


# ---------------------------------------------------------------------------
# normalisation + validity checks
# ---------------------------------------------------------------------------
def check_play_stack(stack: np.ndarray) -> np.ndarray:
    """Validate a ``(4,112,96)`` uint8 play stack."""

    array = np.asarray(stack)
    if array.dtype != np.uint8:
        raise ObsSpecV3Error(f"play stack dtype {array.dtype} is not uint8")
    if array.shape != (FRAME_STACK, *PLAY_SHAPE):
        raise ObsSpecV3Error(
            f"play stack shape {array.shape} != {(FRAME_STACK, *PLAY_SHAPE)}")
    return array


def normalize_play(stack: np.ndarray) -> np.ndarray:
    """``uint8 (4,112,96)`` -> ``float32 (4,112,96)`` in [0,1] (``/ 255.0``)."""

    return check_play_stack(stack).astype(np.float32) / 255.0


def normalize_hud(hud: np.ndarray) -> np.ndarray:
    """``uint8 (56,160)`` -> ``float32 (1,56,160)`` in [0,1] (obs-v1 bytes)."""

    return obs_spec.normalize_hud(hud)


def normalize_action_history(history: np.ndarray) -> np.ndarray:
    """``uint8 (8,18)`` -> ``float32 (144,)`` in [0,1] (obs-v2 contract)."""

    return obs_spec_v2.normalize_action_history(history)


def check_action_history(history: np.ndarray) -> np.ndarray:
    """``(N,8,18)`` uint8 rows that are one-hot or all-zero (obs-v2 contract)."""

    return obs_spec_v2.check_action_history(history)


# ---------------------------------------------------------------------------
# deployment-side buffers
# ---------------------------------------------------------------------------
class ObservationBufferV3:
    """Rolling 4-frame 96x112 playfield stack plus the current HUD strip.

    Same padding rule as obs-v1: before the stack is full the earliest frame is
    repeated, so a live evaluation never invents motion it did not see.  The action
    history lives in :class:`touhou_rl.v400.student.obs_spec_v2.ActionHistoryBuffer`
    (the obs-v2 contract); :meth:`arrays` composes both.
    """

    def __init__(self) -> None:
        self._play: list[np.ndarray] = []
        self._hud: np.ndarray | None = None
        self.frames_seen = 0

    def reset(self) -> None:
        self._play = []
        self._hud = None
        self.frames_seen = 0

    def push(self, play: np.ndarray, hud: np.ndarray) -> None:
        play = np.asarray(play, dtype=np.uint8)
        hud = np.asarray(hud, dtype=np.uint8)
        if play.shape != PLAY_SHAPE:
            raise ObsSpecV3Error(f"play frame shape {play.shape} != {PLAY_SHAPE}")
        if hud.shape != HUD_SHAPE:
            raise ObsSpecV3Error(f"hud frame shape {hud.shape} != {HUD_SHAPE}")
        self._play.append(play.copy())
        if len(self._play) > FRAME_STACK:
            del self._play[0]
        self._hud = hud.copy()
        self.frames_seen += 1

    def stack(self) -> np.ndarray:
        if not self._play:
            raise ObsSpecV3Error("observation buffer is empty; push at least one frame")
        frames = list(self._play)
        while len(frames) < FRAME_STACK:
            frames.insert(0, frames[0])
        return np.stack(frames, axis=0).astype(np.uint8)

    def hud(self) -> np.ndarray:
        if self._hud is None:
            raise ObsSpecV3Error("observation buffer has no HUD frame yet")
        return self._hud

    def arrays(self, history=None) -> dict[str, np.ndarray]:
        """Model-ready arrays: ``play (4,112,96)``, ``hud (1,56,160)``, ``act_hist (144,)``.

        ``history`` is an ``ActionHistoryBuffer`` (or any object with ``rows()``);
        without one the action history is all-zero rows, which is exactly the
        documented "no recorded action yet" padding.
        """

        rows = (history.rows() if history is not None
                else np.zeros(ACTION_HISTORY_SHAPE, dtype=np.uint8))
        return {
            "play": normalize_play(self.stack()),
            "hud": normalize_hud(self.hud()),
            "act_hist": normalize_action_history(rows),
        }


__all__ = [
    "ACTION_HISTORY_BYTES",
    "ACTION_HISTORY_DECISIONS",
    "ACTION_HISTORY_DIM",
    "ACTION_HISTORY_ORDER",
    "ACTION_HISTORY_PAD",
    "ACTION_HISTORY_SHAPE",
    "BYTES_PER_DECISION",
    "DANGER_SUMMARY_PRESENT",
    "DECISION_FRAMES",
    "FRAME_STACK",
    "HUD_PLANE",
    "HUD_SHAPE",
    "KNOWN_LIMITATION",
    "NORMALIZATION",
    "OBS_SPEC_VERSION",
    "OBS_V1_DIGEST",
    "OBS_V1_FILE_SHA256",
    "OBS_V2_DIGEST",
    "OBS_V2_FILE_SHA256",
    "ObsSpecV3Error",
    "ObsSpecV3Mismatch",
    "ObservationBufferV3",
    "PLAY_BLOCK",
    "PLAY_EXTENT_WH",
    "PLAY_PIXELS",
    "PLAY_PLANE",
    "PLAY_SHAPE",
    "POLICY_AUX_LABEL",
    "PROBE_GATE_F1",
    "PROBE_OUTCOME",
    "PROBE_RESULTS",
    "PROBE_SELECTION_RULE",
    "RULING_CRITERION",
    "RULING_DATE",
    "RULING_DECISION",
    "RULING_ID",
    "SELF_STATE_PRESENT",
    "STACK_ORDER",
    "check_action_history",
    "check_obs_spec_v3",
    "check_play_stack",
    "encode_planes_v3",
    "normalize_action_history",
    "normalize_hud",
    "normalize_play",
    "obs_spec_v3",
    "obs_spec_v3_digest",
]
