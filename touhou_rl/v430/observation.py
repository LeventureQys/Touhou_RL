"""TH08 **O2** pixel encoder (v4.3.0 work package W02, slice A).

Specification: ``Document/Update/Alpha/v4.3.0/DEVELOPMENT_GUIDE.md`` §4.2 (the
initial TH08 O2 encoding) and ``PLAN_SPEC.json`` ``observation.O2``.  Every
parameter is **reused** from :mod:`touhou_rl.v430.contracts` (crops, luma weights,
downsample rule, storage dtype, normalisation point, key order); this module adds
the *encoder* — the actual pixel arithmetic and the fail-closed gates — not a
second parameter table.

What one decision carries (field names as in guide §4.1):

============================ ==================== ================================================
field                        shape / dtype        rule here
============================ ==================== ================================================
``global_rgb``               ``(3,240,320)`` u8   the *complete* 640x480 frame, 2x2 integer mean
``play_gray``                ``(4,224,192)`` u8   crop ``(32,16,384,448)`` -> BT.601 -> 2x2 mean,
                                                  four genuine consecutive frames, oldest first
``hud_gray``                 ``(1,448,160)`` u8   ``[16:464, 480:640]`` -> BT.601, native
                                                  resolution, **no** downsample
``executed_key_history``     ``(8,7)`` u8         actual executed key bits, ``KEY_ORDER`` columns,
                                                  oldest first
``history_valid``            ``(8,)`` bool        cold start is all ``False``; never an
                                                  "executed idle" claim
``elapsed_capture_s``        scalar f32           the capture clock's own interval, not a frame no.
============================ ==================== ================================================

Integer arithmetic, in the order that is hashed into the digest:

* ``global``/``play``: ``uint32`` sum of the four pixels of each 2x2 block, then
  floor division by 4.  Summing in ``uint8`` would wrap (255+255+255+255 = 1020 =
  252 mod 256) and silently darken bright bullets; :func:`downsample_2x2_int`
  therefore rejects non-``uint8`` input and never allocates a ``uint8``
  accumulator.
* luma: ``(77R + 151G + 28B) >> 8`` in ``uint32`` (:func:`gray_bt601_int`), so
  pure primaries land on 76 / 150 / 27.

Deliberate refusals (each one is a way a *broken* capture could otherwise become a
"successful" training sample):

1. a frame that is not exactly ``uint8 (480,640,4)`` — **no resize / no
   interpolation**, the guide's "禁止自动插值成另一形状";
2. fewer than four frames — a black-padded or repeated stack is not four genuine
   consecutive frames, so it is refused, not zero-filled;
3. a frame whose RGB content is a single value (an all-black placeholder) or that
   is byte-identical to another frame in the stack (a stale/repeated read);
4. an alpha plane holding intermediate values (a blended/premultiplied buffer:
   dropping alpha would silently change colours).  Constant 0 (unused DIB alpha),
   constant 255 (opaque) and binary 0/255 are dropped unblended.  **This refusal is
   the default gate only** — the explicit, audited opt-in ``alpha_mode="any"``
   disables the inspection (see below);
5. a ``history_valid`` window with a hole, or one that claims more valid slots
   than supplied key rows — a missing row can never be replaced by "executed
   idle";
6. an ``elapsed_capture_s`` that is an integer (a frame counter is not a capture
   interval), non-finite, or non-positive;
7. any privileged name in the inputs, or any unknown keyword — the actor may see
   pixels and its own executed keys only.

Alpha modes (``alpha_mode`` keyword argument of :func:`encode_o2`): the accepted
values are :data:`ALPHA_MODE_CHOICES` = ``("opaque_255", "unused_0",
"binary_mixed", "any")`` plus the default policy name ``"strict"``.  ``"strict"``
is the pre-existing gate — it classifies the plane and accepts the three lossless
verdicts, refusing intermediate alpha.  ``"any"`` is for a **real engine readback**:
``glReadPixels`` over the game window returns a *blended* alpha plane, measured
**202..219**, not 255, and no amount of re-reading changes that.  Under ``"any"``:

* the alpha channel is **still dropped and never blended** — ``global_rgb``,
  ``play_gray`` and ``hud_gray`` are computed from the RGB channels alone, so an
  accepted alpha plane cannot alter one encoded byte (the ``"any"`` and strict
  paths give bit-identical planes for identical RGB);
* therefore **any** alpha is accepted, including the 202..219 blended band and the
  premultiplied buffers the strict gate refuses;
* **the cost** is the weak signal "this frame is translucent", which strict mode
  used as an extra hint of a blended read.  The module provides **no** substitute:
  a caller using ``"any"`` **must** run its own placeholder/staleness detection
  (e.g. reject a frame whose ``global_rgb`` is all-zero or near-zero).  The
  short-stack, duplicate-frame and uniform-RGB gates stay active in every mode.

Every observation records the gate it was built with (``O2Observation.alpha_mode``,
``"any"`` or the strict verdict), and :func:`encoder_parameters` /
:func:`encoder_digest` carry ``alpha_mode_used`` so a lenient-alpha dataset can
never be mistaken for strictly gated data.  :func:`o2_digest` — the contract's
parameter block — is identical in every mode.

Normalisation is **not** here: arrays stay ``uint8`` on disk and in the rollout,
and :func:`to_float_minibatch` is the single place that divides by 255.

Purity: numpy + stdlib only.  No torch, no engine, no game data, no I/O, no
global mutable state.  Nothing in this module launches or talks to the game, so
its tests are **unit tests of the encoder** and are not evidence of game
capability.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from touhou_rl.v430 import contracts

__all__ = [
    # identity
    "ENCODER_VERSION",
    "ENCODER_STEPS",
    "O2_DIGEST_SCOPE",
    "PIXEL_DIGEST_SCOPE",
    "DIGEST_SCOPES",
    # shapes / keys (derived from contracts, never restated)
    "O2_SHAPES",
    "O2_DTYPES",
    "O2_ENCODED_KEYS",
    "O2_FIELD_KEYS",
    "KEY_ORDER",
    "KEY_SLOTS",
    "KEY_COLUMNS",
    "PLAY_STACK",
    "SOURCE_WH",
    "SOURCE_SHAPE_RGBA",
    "PLAY_CROP_XYWH",
    "HUD_CROP_XYWH",
    "ALPHA_CHANNEL_INDEX",
    "RGB_CHANNEL_INDEX",
    "ALPHA_MODES",
    "ALPHA_STRICT_MODES",
    "ALPHA_MODE_DEFAULT",
    "ALPHA_MODE_CHOICES",
    "ALPHA_POLICY",
    "check_alpha_mode",
    "PLAY_STACK_ORDER",
    # errors
    "ObservationError",
    "FrameFormatError",
    "FrameAlphaError",
    "ShortFrameStackError",
    "PlaceholderFrameError",
    "DuplicateFrameError",
    "KeyHistoryError",
    "PrivilegedObservationError",
    # privileged sentinel
    "EXTRA_FORBIDDEN_NAMES",
    "FORBIDDEN_INPUT_NAMES",
    "assert_actor_inputs_clean",
    # pixels
    "alpha_mode",
    "rgba_to_rgb",
    "gray_bt601_int",
    "downsample_2x2_int",
    "play_gray_frame",
    "hud_gray_frame",
    # encoding
    "O2Observation",
    "encode_o2",
    "check_o2_observation",
    "attach_memory",
    "to_float_minibatch",
    # digest
    "o2_digest",
    "encoder_parameters",
    "encoder_digest",
    "o2_identity",
    "canonical_text_bytes",
    "sha256_text",
    "sha256_pixels",
    "check_identity_record",
]

#: Version of the *encoder* (the pixel steps and their order).  The O2 parameter
#: block itself is versioned by ``contracts.CONTRACT_VERSION``.
ENCODER_VERSION = "o2-th08-encoder/1"

#: Digest scopes from ``evidence/W00/SHA_LINE_ENDING_FINDING.md`` §6: text
#: artefacts hash CRLF->LF normalised bytes, binary artefacts hash raw bytes.
#: No identity record in this module may carry a sha256 without one of these.
O2_DIGEST_SCOPE = "text/lf-normalized"
PIXEL_DIGEST_SCOPE = "binary/raw"
DIGEST_SCOPES: tuple[str, ...] = (O2_DIGEST_SCOPE, PIXEL_DIGEST_SCOPE)


# ---------------------------------------------------------------------------
# errors: every failure path is an exception, there is no sentinel return
# ---------------------------------------------------------------------------
class ObservationError(contracts.ContractError):
    """Base class of the O2 encoding contract violations (a ``ContractError``)."""


class FrameFormatError(ObservationError):
    """A source frame is not ``uint8 (480,640,4)`` (no resize, no dtype guess)."""


class FrameAlphaError(FrameFormatError):
    """The alpha plane is blended/premultiplied, so dropping it is not lossless."""


class ShortFrameStackError(ObservationError):
    """Fewer than :data:`PLAY_STACK` genuine frames were supplied."""


class PlaceholderFrameError(ObservationError):
    """A frame carries no image content (a black placeholder), not a real read."""


class DuplicateFrameError(PlaceholderFrameError):
    """Two stack frames are byte-identical: a repeated read, not four frames."""


class KeyHistoryError(ObservationError):
    """``executed_key_history`` / ``history_valid`` do not describe real keys."""


class PrivilegedObservationError(ObservationError):
    """Truth / teacher / judge state was handed to the encoder."""


# ---------------------------------------------------------------------------
# dimensions, crops and key order: reused from contracts, never restated
# ---------------------------------------------------------------------------
#: ``(width, height)`` of the source capture, from ``contracts.SOURCE_RGB_WH``.
SOURCE_WH: tuple[int, int] = contracts.SOURCE_RGB_WH
SOURCE_WIDTH, SOURCE_HEIGHT = SOURCE_WH
#: RGBA channel count and the index of the channel that is dropped.
N_RGBA_CHANNELS = len(contracts.RGBA_INPUT_ORDER)
N_RGB_CHANNELS = len(contracts.RGB_CHANNEL_ORDER)
ALPHA_CHANNEL_INDEX = contracts.RGBA_INPUT_ORDER.index("A")
RGB_CHANNEL_INDEX: tuple[int, ...] = tuple(
    contracts.RGBA_INPUT_ORDER.index(name) for name in contracts.RGB_CHANNEL_ORDER)
#: ``(height, width, channels)`` of one accepted source frame.
SOURCE_SHAPE_RGBA: tuple[int, int, int] = (
    SOURCE_HEIGHT, SOURCE_WIDTH, N_RGBA_CHANNELS)

PLAY_CROP_XYWH: tuple[int, int, int, int] = contracts.O2_PLAY_CROP_XYWH
HUD_CROP_XYWH: tuple[int, int, int, int] = contracts.O2_HUD_CROP_XYWH
KEY_ORDER: tuple[str, ...] = contracts.KEY_ORDER
KEY_SLOTS: int = contracts.FIELD_SPECS["executed_key_history"].shape[0]
KEY_COLUMNS: int = contracts.FIELD_SPECS["executed_key_history"].shape[1]
PLAY_STACK: int = contracts.FIELD_SPECS["play_gray"].shape[0]
PLAY_STACK_ORDER = "oldest_first:t-3..t"

#: ``global_rgb``/``play_gray``/... shape and dtype, derived from the contract's
#: ``o2_parameters()`` so this module cannot drift from the plan.
def _plan_shapes() -> dict[str, tuple[int, ...]]:
    params = contracts.o2_parameters()
    rank = int(params["elapsed_capture_s_rank"])
    if rank != 0:  # pragma: no cover - PLAN_SPEC pins a scalar
        raise RuntimeError(
            f"PLAN_SPEC elapsed_capture_s_rank is {rank}; the O2 encoder encodes "
            f"one scalar interval per decision")
    return {
        "global_rgb": tuple(params["global_rgb_chw"]),
        "play_gray": tuple(params["play_gray_chw"]),
        "hud_gray": tuple(params["hud_gray_chw"]),
        "executed_key_history": tuple(params["executed_key_history_shape"]),
        "history_valid": tuple(params["history_valid_shape"]),
        "elapsed_capture_s": (),
        "memory": tuple(params["memory_shape"]),
    }


#: Every field of guide §4.1 (``memory`` included), in contract order.
O2_SHAPES: dict[str, tuple[int, ...]] = _plan_shapes()
O2_DTYPES: dict[str, str] = {
    name: spec.dtype for name, spec in contracts.FIELD_SPECS.items()}
#: The fields the *pixel encoder* produces; ``memory`` is the model's own state
#: and cannot be observed here (see :meth:`O2Observation.to_actor_observation`).
O2_ENCODED_KEYS: tuple[str, ...] = tuple(
    name for name in contracts.ACTOR_OBSERVATION_KEYS if name != "memory")
O2_FIELD_KEYS: tuple[str, ...] = contracts.ACTOR_OBSERVATION_KEYS

if tuple(O2_SHAPES) != O2_FIELD_KEYS:  # pragma: no cover - frozen table
    raise RuntimeError(
        f"O2_SHAPES keys {tuple(O2_SHAPES)} != contract keys {O2_FIELD_KEYS}")
if O2_SHAPES != {name: spec.shape for name, spec in contracts.FIELD_SPECS.items()}:
    raise RuntimeError(  # pragma: no cover - frozen table
        "O2_SHAPES disagrees with contracts.FIELD_SPECS; the encoder and the "
        "contract must be the same table")
if "memory" not in O2_FIELD_KEYS or "memory" in O2_ENCODED_KEYS:  # pragma: no cover
    raise RuntimeError("memory must stay the one field the pixel encoder omits")

#: Alpha modes **classifiable by inspection** (``alpha_mode(frame)``); every one of
#: them can be dropped losslessly because alpha carries no colour.
#: ``opaque_255`` = a normal opaque RGBA capture; ``unused_0`` = a 32bpp GDI/BitBlt
#: DIB whose alpha byte was never written; ``binary_mixed`` = only 0 and 255, i.e.
#: a mask that we deliberately do *not* apply; ``any`` = **do not inspect alpha at
#: all** (see the "any" semantics below).
ALPHA_MODES: tuple[str, ...] = ("opaque_255", "unused_0", "binary_mixed", "any")
#: The three **strict** verdicts: they are established by looking at the plane, and
#: in every one of them dropping alpha is provably lossless.
ALPHA_STRICT_MODES: tuple[str, ...] = ("opaque_255", "unused_0", "binary_mixed")
#: Default of the ``alpha_mode`` keyword argument.  ``"strict"`` is *not* a fifth
#: verdict and deliberately **not** a member of :data:`ALPHA_MODES`: it is the name
#: of the pre-existing **policy** "whatever the strict classifier says, refuse an
#: undecidable plane".  Keeping it out of ``ALPHA_MODES`` means no code can mistake
#: a policy name for an observed plane; keeping it as the default means the default
#: gate is exactly the v4.3.0 W02 gate (255/0/binary accepted, anything else
#: refused) — the four strict verdicts stay distinguishable in the record.
ALPHA_MODE_DEFAULT = "strict"
#: Every spelling the ``alpha_mode`` keyword argument accepts: the four concrete
#: modes plus the default policy name.  No other alias exists.
ALPHA_MODE_CHOICES: tuple[str, ...] = ALPHA_MODES + (ALPHA_MODE_DEFAULT,)

ALPHA_POLICY = (
    "the alpha channel is dropped, never blended; constant 0 (unused), constant "
    "255 (opaque) and binary 0/255 are accepted, any intermediate alpha is refused "
    "under the default alpha_mode='strict'; alpha_mode='any' drops alpha without "
    "inspecting it (real glReadPixels buffers are blended, 202..219), which loses "
    "the translucent-frame signal and makes placeholder rejection the caller's job")


# ---------------------------------------------------------------------------
# privileged-input sentinel
# ---------------------------------------------------------------------------
#: Names this module refuses **in addition to** ``contracts.FORBIDDEN_ACTOR_FIELDS``
#: and ``contracts.PRIVILEGED_FIELD_NAMES``.  Same class of leak (engine/teacher/
#: judge truth a hand-built dict could smuggle in), listed here rather than in the
#: contract so ``FORBIDDEN_ACTOR_FIELDS`` stays exactly the plan's list.
EXTRA_FORBIDDEN_NAMES: tuple[str, ...] = (
    "stage_id", "stage_number", "player_center", "player_position", "player_pos",
    "player_xy", "hitbox", "lives", "bombs", "power", "score", "graze",
    "resource", "resources", "boss_hp", "enemy_hp", "enemy_hp_count",
    "frame_index", "game_frame", "game_frames", "rng", "label", "labels",
    "teacher_label", "judge_label", "hidden_state", "true_state",
)
#: The whole refused set: the contract's names first, then this module's extras.
FORBIDDEN_INPUT_NAMES: tuple[str, ...] = (
    tuple(contracts.PRIVILEGED_FIELD_NAMES) + EXTRA_FORBIDDEN_NAMES)

if not set(contracts.PRIVILEGED_FIELD_NAMES) <= set(FORBIDDEN_INPUT_NAMES):
    raise RuntimeError(  # pragma: no cover - frozen union
        "the encoder's forbidden-name set must be a superset of the contract's")
if set(EXTRA_FORBIDDEN_NAMES) & set(contracts.PRIVILEGED_FIELD_NAMES):
    raise RuntimeError(  # pragma: no cover - frozen union
        "EXTRA_FORBIDDEN_NAMES duplicates a contract name")

#: Recursion depth of the local name scan, matching ``contracts``' own limit.
_SCAN_DEPTH = 6


def _scan_names(obj: Any, forbidden: frozenset[str], path: str = "",
                depth: int = 0) -> tuple[str, ...]:
    """Dotted paths of ``forbidden`` names reachable in ``obj`` (like contracts')."""

    if depth > _SCAN_DEPTH:
        return ()
    found: list[str] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            name = key if isinstance(key, str) else str(key)
            child = f"{path}.{name}" if path else name
            if name in forbidden:
                found.append(child)
                continue
            found.extend(_scan_names(value, forbidden, child, depth + 1))
        return tuple(found)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            child = f"{path}.{field.name}" if path else field.name
            if field.name in forbidden:
                found.append(child)
                continue
            found.extend(
                _scan_names(getattr(obj, field.name), forbidden, child, depth + 1))
        return tuple(found)
    if isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            child = f"{path}[{index}]" if path else f"[{index}]"
            found.extend(_scan_names(item, forbidden, child, depth + 1))
        return tuple(found)
    attributes = getattr(obj, "__dict__", None)
    if isinstance(attributes, dict):
        for name, value in attributes.items():
            child = f"{path}.{name}" if path else name
            if name in forbidden:
                found.append(child)
                continue
            found.extend(_scan_names(value, forbidden, child, depth + 1))
    return tuple(found)


def assert_actor_inputs_clean(obj: Any, *, what: str = "O2 encoder input") -> Any:
    """Refuse ``obj`` if it carries truth/teacher/judge state anywhere inside.

    Runs the contract sentinel first (``contracts.assert_not_privileged``, whose
    name list is the plan's ``forbidden_actor_fields``) and then the same scan
    with this module's extra names (``stage_id``, ``player_center``, ...).  Returns
    ``obj`` unchanged on success; the failure is always the exception, never a
    falsy return.
    """

    try:
        contracts.assert_not_privileged(obj, what=what)
    except contracts.ContractError as exc:
        raise PrivilegedObservationError(str(exc)) from exc
    found = _scan_names(obj, frozenset(FORBIDDEN_INPUT_NAMES))
    if found:
        raise PrivilegedObservationError(
            f"{what} carries privileged fields {list(found)}; the actor may see "
            f"pixels and its own executed keys only "
            f"(contract forbids {list(contracts.FORBIDDEN_ACTOR_FIELDS)})")
    return obj


# ---------------------------------------------------------------------------
# pixels
# ---------------------------------------------------------------------------
def _check_frame_array(frame: Any, *, name: str = "frame") -> np.ndarray:
    """Accept only ``uint8 (480,640,4)``; refuse every other shape and dtype."""

    try:
        array = np.asarray(frame)
    except Exception as exc:  # noqa: BLE001 - a ragged/odd input is a format error
        raise FrameFormatError(f"{name} cannot be read as an array: {exc}") from exc
    if array.dtype != np.dtype(np.uint8):
        raise FrameFormatError(
            f"{name} dtype is {array.dtype}, required uint8; a float/16-bit frame "
            f"would need a scale convention this encoder refuses to guess")
    if array.shape != SOURCE_SHAPE_RGBA:
        raise FrameFormatError(
            f"{name} shape {array.shape} != {SOURCE_SHAPE_RGBA}; the source is "
            f"{SOURCE_WIDTH}x{SOURCE_HEIGHT} RGBA and this encoder never resizes or "
            f"interpolates into another shape (guide §4.2: '禁止自动插值成另一形状')")
    return array


def alpha_mode(frame: Any) -> str:
    """Classify the alpha plane of one source frame, or raise.

    Returns one of :data:`ALPHA_MODES`.  A plane with intermediate values is
    refused with :class:`FrameAlphaError`: such a buffer is blended or
    premultiplied, so dropping alpha would silently alter every colour.
    """

    array = _check_frame_array(frame)
    alpha = array[..., ALPHA_CHANNEL_INDEX]
    low = int(alpha.min())
    high = int(alpha.max())
    if low == high:
        if low == 255:
            return "opaque_255"
        if low == 0:
            return "unused_0"
        raise FrameAlphaError(
            f"alpha plane is constant {low}; only fully opaque (255) or unused (0) "
            f"alpha can be dropped without changing colours")
    if bool(np.isin(alpha, (0, 255)).all()):
        return "binary_mixed"
    raise FrameAlphaError(
        f"alpha plane holds intermediate values ({low}..{high}); the buffer is "
        f"blended/premultiplied and dropping alpha is not lossless")


def check_alpha_mode(mode: Any = ALPHA_MODE_DEFAULT) -> str:
    """Validate the ``alpha_mode`` **argument**; return it unchanged.

    Accepted values are exactly :data:`ALPHA_MODE_CHOICES`: the four concrete modes
    ``("opaque_255", "unused_0", "binary_mixed", "any")`` plus the default policy
    name ``"strict"``.  Any other value raises :class:`ObservationError` — the same
    exception class the encoder already uses for an unknown keyword — rather than
    silently falling back to a stricter gate (which would reject real data) or to
    the lenient gate (which would silently accept an unreviewed buffer).
    """

    if not isinstance(mode, str):
        raise ObservationError(
            f"alpha_mode {mode!r} of type {type(mode).__name__} is not a string; "
            f"accepted values are {list(ALPHA_MODE_CHOICES)}")
    if mode not in ALPHA_MODE_CHOICES:
        raise ObservationError(
            f"alpha_mode {mode!r} is not one of {list(ALPHA_MODE_CHOICES)}; "
            f"'strict' is the default gate (refuses intermediate alpha), 'any' "
            f"drops alpha without inspecting it")
    return mode


def _resolve_frame_alpha(frame: np.ndarray, requested: Any) -> str:
    """Apply ``requested`` to one frame; return the mode **actually applied**.

    * ``"any"`` -> alpha is not inspected at all; returns ``"any"``.
    * ``"strict"`` (default) -> the strict classifier runs and its verdict
      (``opaque_255`` / ``unused_0`` / ``binary_mixed``) is returned; every other
      plane raises :class:`FrameAlphaError`, exactly as before this parameter
      existed.
    * an explicit concrete mode -> that exact verdict is **required**; a frame that
      classifies as another (still lossless) flavour is refused, because the caller
      asserted which buffer it was reading and a mismatch means the assertion is
      stale.
    """

    mode = check_alpha_mode(requested)
    if mode == "any":
        return "any"
    observed = alpha_mode(frame)
    if mode == ALPHA_MODE_DEFAULT:
        return observed
    if observed != mode:
        raise FrameAlphaError(
            f"alpha_mode={mode!r} was requested but this frame classifies as "
            f"{observed!r}; the caller's declared buffer flavour does not match the "
            f"pixels, so it is refused instead of reclassified")
    return observed


def rgba_to_rgb(frame: Any, *, alpha_mode: str = ALPHA_MODE_DEFAULT) -> np.ndarray:
    """``uint8 (480,640,4)`` RGBA -> ``uint8 (480,640,3)`` RGB, alpha dropped.

    The channel order is explicit: ``contracts.RGBA_INPUT_ORDER`` is
    ``("R","G","B","A")`` and the dropped index is
    :data:`ALPHA_CHANNEL_INDEX` == 3.  Alpha is **never** blended into the colour
    channels; the plane is gated by :func:`alpha_mode` under the default
    ``alpha_mode="strict"``, while ``alpha_mode="any"`` drops the plane without
    inspecting it (same RGB arithmetic, no alpha term anywhere).
    """

    array = _check_frame_array(frame)
    _resolve_frame_alpha(array, alpha_mode)
    return np.ascontiguousarray(np.take(array, RGB_CHANNEL_INDEX, axis=-1))


def _check_channel(channel: Any, *, name: str) -> np.ndarray:
    array = np.asarray(channel)
    if array.dtype != np.dtype(np.uint8):
        raise FrameFormatError(
            f"{name} dtype is {array.dtype}, required uint8; the integer rules of "
            f"guide §4.2 are defined on uint8 samples only")
    if array.ndim != 2:
        raise FrameFormatError(f"{name} must be a 2-D (rows, cols) plane, got {array.shape}")
    return array


def gray_bt601_int(rgb: Any) -> np.ndarray:
    """BT.601 integer luma ``(77R + 151G + 28B) >> 8`` on ``uint8`` RGB.

    ``(...,3)`` -> ``(...)``.  The accumulation is ``uint32`` (the maximum is
    ``255 * 256 = 65280``), and the weights/shift come from
    ``contracts.O2_GRAY_WEIGHTS`` / ``contracts.O2_GRAY_SHIFT``.
    """

    array = np.asarray(rgb)
    if array.dtype != np.dtype(np.uint8):
        raise FrameFormatError(
            f"gray input dtype is {array.dtype}, required uint8")
    if array.ndim < 1 or array.shape[-1] != N_RGB_CHANNELS:
        raise FrameFormatError(
            f"gray input last axis is {array.shape}, required {N_RGB_CHANNELS} (RGB)")
    weights = contracts.O2_GRAY_WEIGHTS
    acc = np.zeros(array.shape[:-1], dtype=np.uint32)
    for index, weight in enumerate(weights):
        acc += array[..., index].astype(np.uint32) * np.uint32(weight)
    return (acc >> np.uint32(contracts.O2_GRAY_SHIFT)).astype(np.uint8)


def downsample_2x2_int(channel: Any) -> np.ndarray:
    """2x2 integer block mean: ``uint32`` sum, then floor division by 4.

    ``uint8 (2m,2n)`` -> ``uint8 (m,n)``.  The sum is accumulated in ``uint32``
    because ``255+255+255+255 = 1020`` wraps to 252 in ``uint8``; a wrapped
    accumulator would turn a saturated bullet block into 63 instead of 255.
    """

    array = _check_channel(channel, name="downsample input")
    rows, columns = array.shape
    factor = contracts.O2_DOWNSAMPLE_FACTOR
    if factor != 2:  # pragma: no cover - frozen parameter
        raise ObservationError(f"PLAN_SPEC downsample factor is {factor}, not 2")
    if rows % factor or columns % factor:
        raise ObservationError(
            f"downsample input {array.shape} is not divisible by {factor}; the "
            f"encoder refuses to drop or pad a row/column")
    acc = array.astype(np.uint32)
    summed = (acc[0::factor, 0::factor] + acc[1::factor, 0::factor]
              + acc[0::factor, 1::factor] + acc[1::factor, 1::factor])
    return (summed // np.uint32(factor * factor)).astype(np.uint8)


def _crop(rgb: np.ndarray, crop_xywh: Sequence[int], *, name: str) -> np.ndarray:
    x, y, width, height = (int(value) for value in crop_xywh)
    rows, columns = rgb.shape[:2]
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > columns or y + height > rows:
        raise ObservationError(
            f"{name} crop {list(crop_xywh)} does not fit the {columns}x{rows} source; "
            f"the encoder never pads a short crop")
    window = rgb[y:y + height, x:x + width]
    if window.shape[:2] != (height, width):  # pragma: no cover - checked above
        raise ObservationError(
            f"{name} crop produced {window.shape} for {list(crop_xywh)}")
    return window


def play_gray_frame(rgb: Any) -> np.ndarray:
    """One frame's play plane: crop -> BT.601 -> 2x2 mean -> ``uint8 (224,192)``."""

    array = np.asarray(rgb)
    if array.dtype != np.dtype(np.uint8) or array.ndim != 3:
        raise FrameFormatError(
            f"play plane input must be uint8 HxWx3, got {array.dtype} {array.shape}")
    window = _crop(array, PLAY_CROP_XYWH, name="play")
    plane = downsample_2x2_int(gray_bt601_int(window))
    expected = O2_SHAPES["play_gray"][1:]
    if plane.shape != expected:  # pragma: no cover - crops are frozen
        raise ObservationError(f"play plane shape {plane.shape} != {expected}")
    return plane


def hud_gray_frame(rgb: Any) -> np.ndarray:
    """One frame's HUD strip: crop -> BT.601 at native resolution (no downsample)."""

    array = np.asarray(rgb)
    if array.dtype != np.dtype(np.uint8) or array.ndim != 3:
        raise FrameFormatError(
            f"HUD plane input must be uint8 HxWx3, got {array.dtype} {array.shape}")
    window = _crop(array, HUD_CROP_XYWH, name="HUD")
    plane = gray_bt601_int(window)
    expected = O2_SHAPES["hud_gray"][1:]
    if plane.shape != expected:  # pragma: no cover - crops are frozen
        raise ObservationError(f"HUD plane shape {plane.shape} != {expected}")
    return plane


# ---------------------------------------------------------------------------
# one encoded observation
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class O2Observation:
    """The six fields the pixel encoder can actually observe (guide §4.1 minus
    ``memory``).

    ``memory`` is the model's recurrent state: it comes from the GRU, not from a
    pixel, and filling it here (with zeros or with a teacher's vector) would be a
    fabrication.  :meth:`to_actor_observation` / :func:`attach_memory` add a real
    ``memory`` and return a contract-validated
    :class:`~touhou_rl.v430.contracts.ActorObservation`.

    ``alpha_mode`` is **provenance, not an observation**: it records which alpha
    gate actually produced these pixels (``"any"`` = alpha was never inspected, so
    the translucent-frame signal is gone and placeholder detection was the caller's
    job; ``"opaque_255"``/``"unused_0"``/``"binary_mixed"`` = the strict verdict the
    default gate established).  It is deliberately **not** one of the six encoded
    fields: :meth:`as_dict` and ``O2_ENCODED_KEYS`` stay at the six contract fields,
    so the actor's input surface and every ``to_float_minibatch`` key are unchanged.
    """

    global_rgb: Any
    play_gray: Any
    hud_gray: Any
    executed_key_history: Any
    history_valid: Any
    elapsed_capture_s: Any
    #: The alpha gate actually applied; see the class docstring and
    #: :data:`ALPHA_MODE_CHOICES`.  Default = the strict policy, so a record built
    #: without the keyword describes exactly what the pre-existing encoder did.
    alpha_mode: str = ALPHA_MODE_DEFAULT

    def __post_init__(self) -> None:
        check_o2_observation(self)
        check_alpha_mode(self.alpha_mode)

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in O2_ENCODED_KEYS}

    @property
    def history_valid_count(self) -> int:
        """How many of the eight key slots are backed by a real executed row."""

        return int(np.count_nonzero(np.asarray(self.history_valid)))

    def to_actor_observation(self, memory: Any) -> contracts.ActorObservation:
        """Add the model's own ``memory`` and return the full actor record."""

        return attach_memory(self, memory)


# ---------------------------------------------------------------------------
# fail-closed checks on the input side
# ---------------------------------------------------------------------------
def _as_frame_sequence(frames: Any) -> list[np.ndarray]:
    """Normalise ``frames`` to a list of arrays; refuse anything else."""

    if isinstance(frames, np.ndarray):
        if frames.ndim == 4:
            return [frames[index] for index in range(frames.shape[0])]
        if frames.ndim == 3:
            # A single frame is a legal object, just not a four-frame stack.
            _check_frame_array(frames, name="frames")
            raise ShortFrameStackError(
                f"frames is one {SOURCE_WIDTH}x{SOURCE_HEIGHT} frame; the play "
                f"branch needs the current frame and the {PLAY_STACK - 1} previous "
                f"genuine consecutive frames")
        raise FrameFormatError(
            f"frames array has ndim {frames.ndim}, expected 3 (one frame) or 4 "
            f"(a stack); shape {frames.shape}")
    if isinstance(frames, (str, bytes)) or not isinstance(frames, (list, tuple)):
        raise FrameFormatError(
            f"frames must be a list/tuple of frames or a 4-D array, got "
            f"{type(frames).__name__}")
    return [frames[index] for index in range(len(frames))]


def _check_stack(frames: Any, *, alpha_mode: str = ALPHA_MODE_DEFAULT
                 ) -> tuple[list[np.ndarray], str]:
    """Validate every frame, then refuse placeholders and repeated frames.

    Returns the newest :data:`PLAY_STACK` frames **and** the alpha mode actually
    applied to the newest one (that is the mode recorded on the observation).
    ``alpha_mode`` gates the alpha plane only; the placeholder, duplicate and
    short-stack checks below are unchanged in every mode.
    """

    items = _as_frame_sequence(frames)
    if not items:
        raise ShortFrameStackError(
            f"frames is empty; {PLAY_STACK} genuine consecutive frames are required")
    # Format of *every* supplied frame (cheap, catches a broken tail).
    for index, item in enumerate(items):
        _check_frame_array(item, name=f"frames[{index}]")
    if len(items) < PLAY_STACK:
        raise ShortFrameStackError(
            f"{len(items)} frame(s) supplied, {PLAY_STACK} required; a short stack "
            f"is refused rather than padded with a black or repeated frame "
            f"(guide §4.2)")
    window = items[-PLAY_STACK:]
    applied: str | None = None
    for index, frame in enumerate(window):
        applied = _resolve_frame_alpha(np.asarray(frame), alpha_mode)
        rgb = np.take(np.asarray(frame), RGB_CHANNEL_INDEX, axis=-1)
        if int(rgb.min()) == int(rgb.max()):
            raise PlaceholderFrameError(
                f"frames[-{PLAY_STACK - index}] has a constant RGB value "
                f"({int(rgb.min())}); a uniform frame is a placeholder, not a "
                f"captured game frame")
    for left in range(PLAY_STACK):
        for right in range(left + 1, PLAY_STACK):
            if np.array_equal(window[left], window[right]):
                raise DuplicateFrameError(
                    f"frames[-{PLAY_STACK - left}] and frames[-{PLAY_STACK - right}] "
                    f"are byte-identical; a repeated frame is not two consecutive "
                    f"captured frames (guide §4.2: '两帧动作不代表可用重复一帧伪造"
                    f"四帧栈'). Capture again or replay the real prefix")
    assert applied is not None  # PLAY_STACK >= 1, so the loop assigned it
    return window, applied


def _check_history_flags(history_valid: Any) -> np.ndarray:
    """``history_valid``: a real bool array of eight flags with a False prefix."""

    if history_valid is None:
        raise KeyHistoryError(
            "history_valid is required: a cold start must be declared as eight "
            "False flags, not left out (a missing flag could be read as executed "
            "idle)")
    array = np.asarray(history_valid)
    if array.dtype != np.dtype(np.bool_):
        raise KeyHistoryError(
            f"history_valid dtype is {array.dtype}, required bool; 0/1 integers are "
            f"refused because a validity flag is not a value")
    if array.shape != O2_SHAPES["history_valid"]:
        raise KeyHistoryError(
            f"history_valid shape {array.shape} != {O2_SHAPES['history_valid']}")
    try:
        contracts.check_history_valid_prefix(array)
    except contracts.ContractError as exc:
        raise KeyHistoryError(str(exc)) from exc
    return np.array(array, dtype=np.bool_, copy=True)


def _check_key_rows(key_history: Any) -> np.ndarray:
    """``key_history``: ``(n,7)`` of 0/1 actual executed states, ``n <= 8``."""

    if key_history is None:
        return np.zeros((0, KEY_COLUMNS), dtype=np.uint8)
    array = np.asarray(key_history)
    if array.dtype == np.dtype(np.bool_):
        array = array.astype(np.uint8)
    if array.dtype != np.dtype(np.uint8):
        raise KeyHistoryError(
            f"executed key rows dtype is {array.dtype}, required uint8 (or bool); a "
            f"float would claim a press strength the key state does not have")
    if array.ndim != 2 or array.shape[1] != KEY_COLUMNS:
        raise KeyHistoryError(
            f"executed key rows shape {array.shape} != (n, {KEY_COLUMNS}) with "
            f"columns {list(KEY_ORDER)}")
    if array.shape[0] > KEY_SLOTS:
        raise KeyHistoryError(
            f"{array.shape[0]} executed key rows supplied but only {KEY_SLOTS} "
            f"decision slots exist; pass the most recent {KEY_SLOTS}")
    if array.size and not bool(np.isin(array, (0, 1)).all()):
        raise KeyHistoryError(
            f"executed key rows hold values outside {{0,1}}: "
            f"{sorted({int(value) for value in np.unique(array)})}")
    return np.array(array, dtype=np.uint8, copy=True)


def _build_key_history(key_history: Any, history_valid: Any
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Align the supplied executed rows with the validity flags, oldest first.

    The flags are the caller's statement of *which* slots were really executed.  A
    mismatch in either direction is refused: claiming more valid slots than
    supplied rows would read as "we executed idle", and supplying more rows than
    valid slots means the older rows do not belong to this episode and their
    alignment is undefined.
    """

    flags = _check_history_flags(history_valid)
    rows_in = _check_key_rows(key_history)
    supplied = int(rows_in.shape[0])
    claimed = int(np.count_nonzero(flags))
    if claimed != supplied:
        raise KeyHistoryError(
            f"history_valid claims {claimed} valid slot(s) but {supplied} executed "
            f"key row(s) were supplied; the newest {supplied} slot(s) are the only "
            f"ones that can be filled, and a missing row must stay invalid instead "
            f"of becoming 'executed idle'")
    rows = np.zeros((KEY_SLOTS, KEY_COLUMNS), dtype=np.uint8)
    if supplied:
        rows[KEY_SLOTS - supplied:] = rows_in
    return rows, flags


def _check_capture_interval(elapsed_capture_s: Any) -> np.float32:
    """The capture clock's own interval, in positive finite float32 seconds."""

    value = elapsed_capture_s
    if value is None:
        raise ObservationError(
            "elapsed_capture_s is required: the decision's interval on the capture "
            "clock (guide §5.3); it cannot default to zero or to a frame count")
    if isinstance(value, (bool, np.bool_)):
        raise ObservationError(
            f"elapsed_capture_s {value!r} is a bool, not a number of seconds")
    if isinstance(value, (int, np.integer)):
        raise ObservationError(
            f"elapsed_capture_s {value!r} is an integer; an integer here is a "
            f"frame/step counter, and the engine frame number is not an actor "
            f"input — pass the capture clock's interval as float seconds")
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise ObservationError(
                f"elapsed_capture_s shape {value.shape} is not a scalar; a batch of "
                f"intervals is not one observation")
        if value.dtype.kind != "f":
            raise ObservationError(
                f"elapsed_capture_s dtype is {value.dtype}, required a float scalar")
        seconds = float(value)
    elif isinstance(value, (float, np.floating)):
        seconds = float(value)
    else:
        raise ObservationError(
            f"elapsed_capture_s {value!r} of type {type(value).__name__} is not a "
            f"float number of seconds")
    if not math.isfinite(seconds):
        raise ObservationError(f"elapsed_capture_s {seconds!r} is not finite")
    if seconds <= 0.0:
        raise ObservationError(
            f"elapsed_capture_s {seconds} <= 0; a decision that consumed the clock "
            f"cannot have a non-positive interval, and a zero would masquerade as a "
            f"normal step")
    return np.float32(seconds)


# ---------------------------------------------------------------------------
# the encoder
# ---------------------------------------------------------------------------
def encode_o2(frames: Any, key_history: Any = None, history_valid: Any = None,
              elapsed_capture_s: Any = None, *,
              alpha_mode: str = ALPHA_MODE_DEFAULT,
              **unexpected: Any) -> O2Observation:
    """Encode one TH08 O2 decision.  Fail-closed on every malformed input.

    Parameters
    ----------
    frames:
        The current frame and at least the three previous **genuine consecutive**
        frames, ``uint8 (480,640,4)`` RGBA, oldest first; a 4-D array is accepted
        too.  Only the newest :data:`PLAY_STACK` are encoded.  A short stack, a
        non-480x640 frame, an intermediate-alpha frame, a uniform (placeholder)
        frame or a repeated frame raises — the alpha case only under the default
        ``alpha_mode="strict"`` (see below).
    key_history:
        ``uint8 (n,7)`` (or bool) of the **actual executed** key states, oldest
        first, columns :data:`KEY_ORDER`; ``n <= 8``.  ``None`` means "no history
        at all" (cold start) and is only legal together with all-``False`` flags.
    history_valid:
        ``bool (8,)``; a leading-``False`` prefix is required.  Mandatory.
    elapsed_capture_s:
        The decision's interval on the capture clock, float seconds, ``> 0``.
    alpha_mode:
        Which **alpha gate** to apply, one of :data:`ALPHA_MODE_CHOICES`.
        ``"strict"`` (the default) is the pre-existing gate: the plane is
        classified by :func:`alpha_mode` and ``opaque_255`` / ``unused_0`` /
        ``binary_mixed`` are dropped unblended while any intermediate alpha raises
        :class:`FrameAlphaError`.  ``"any"`` **skips the inspection entirely** and
        therefore accepts a real ``glReadPixels`` buffer, whose alpha plane is
        blended between **202 and 219** rather than 255.

        Why ``"any"`` is sound *and* what it costs: alpha is **dropped, never
        blended** in every mode — ``global_rgb``, ``play_gray`` and ``hud_gray``
        are computed from the RGB channels alone, so an accepted-dropped alpha
        plane cannot change a single encoded byte.  What is lost is the *weak
        signal* "this frame is translucent", which the strict gate used as an
        extra hint that the read was a blended/overlay buffer.  In ``"any"`` mode
        the module offers **no** substitute, so the caller **must** do its own
        placeholder/staleness detection (e.g. test whether ``global_rgb`` is
        all-zero or near-zero, plus the duplicate/short-stack gates that stay
        active here).  ``"any"`` is an explicit, audited opt-in: it never becomes
        the default and the returned observation records it.

    Returns
    -------
    O2Observation
        The six observed fields, still ``uint8``/``bool``/``float32``; nothing is
        normalised and no ``memory`` is invented.  ``observation.alpha_mode``
        carries the gate actually applied (``"any"``, or the strict verdict the
        default gate established), which is provenance and not a seventh encoded
        field.
    """

    check_alpha_mode(alpha_mode)
    if unexpected:
        privileged = [name for name in unexpected if name in FORBIDDEN_INPUT_NAMES]
        if privileged:
            raise PrivilegedObservationError(
                f"O2 encoder was handed privileged field(s) {sorted(privileged)}; the "
                f"actor may see pixels and its own executed keys only "
                f"(contract forbids {list(contracts.FORBIDDEN_ACTOR_FIELDS)})")
        raise ObservationError(
            f"O2 encoder was handed unknown keyword(s) {sorted(unexpected)}; the "
            f"input surface is closed and an unknown keyword is refused rather than "
            f"ignored")
    assert_actor_inputs_clean(
        {"frames": frames, "key_history": key_history,
         "history_valid": history_valid,
         "elapsed_capture_s": elapsed_capture_s},
        what="O2 encoder input")

    window, applied_alpha_mode = _check_stack(frames, alpha_mode=alpha_mode)
    seconds = _check_capture_interval(elapsed_capture_s)
    key_rows, flags = _build_key_history(key_history, history_valid)

    # The *requested* mode is passed on, not the per-frame verdict: under the
    # default "strict" policy a stack may legitimately mix opaque_255 and
    # unused_0 frames, exactly as before this parameter existed.
    rgb_frames = [rgba_to_rgb(frame, alpha_mode=alpha_mode) for frame in window]
    current = rgb_frames[-1]

    global_channels = [
        downsample_2x2_int(current[..., channel])
        for channel in range(N_RGB_CHANNELS)]
    global_rgb = np.stack(global_channels, axis=0)

    play_planes = [play_gray_frame(rgb) for rgb in rgb_frames]
    play_gray = np.stack(play_planes, axis=0)

    hud_gray = hud_gray_frame(current)[np.newaxis, :, :]

    observation = O2Observation(
        global_rgb=global_rgb,
        play_gray=play_gray,
        hud_gray=hud_gray,
        executed_key_history=key_rows,
        history_valid=flags,
        elapsed_capture_s=seconds,
        alpha_mode=applied_alpha_mode,
    )
    # The encoder validates its own output: a shape that drifted from the frozen
    # contract cannot escape as a "successful" sample.
    check_o2_observation(observation)
    return observation


# ---------------------------------------------------------------------------
# validation / composition
# ---------------------------------------------------------------------------
def _shape_of(value: Any, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ObservationError(
            f"{name}: value of type {type(value).__name__} exposes no .shape; pass a "
            f"typed array")
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError as exc:
        raise ObservationError(f"{name}: shape {shape!r} is not iterable") from exc


def _as_field_mapping(observation: Any) -> dict[str, Any]:
    if isinstance(observation, O2Observation):
        return observation.as_dict()
    if isinstance(observation, Mapping):
        return dict(observation)
    if dataclasses.is_dataclass(observation) and not isinstance(observation, type):
        raise ObservationError(
            f"pass an O2Observation or a mapping of the six O2 fields; a "
            f"{type(observation).__name__} must be checked by "
            f"contracts.check_actor_observation instead")
    raise ObservationError(
        f"an O2 observation must be an O2Observation or a mapping, got "
        f"{type(observation).__name__}")


def check_o2_observation(observation: Any) -> dict[str, tuple[tuple[int, ...], str]]:
    """Validate the six encoded fields; return ``{field: (shape, dtype)}``.

    Refused (all with :class:`ObservationError`): a missing/extra field, a
    privileged name, a shape that is not the frozen one, a non-``uint8`` image (a
    float image means it was already normalised — that must happen once, at
    minibatch time), a key row outside 0/1, a ``history_valid`` window with a hole,
    and an ``elapsed_capture_s`` that is not a finite non-negative float32 scalar.
    """

    data = _as_field_mapping(observation)
    assert_actor_inputs_clean(data, what="O2 observation")
    missing = [name for name in O2_ENCODED_KEYS if name not in data]
    if missing:
        raise ObservationError(
            f"O2 observation is missing fields {missing}; a missing field cannot be "
            f"zero-filled")
    extra = [name for name in data if name not in O2_ENCODED_KEYS]
    if extra:
        raise ObservationError(
            f"O2 observation carries extra fields {extra}; the encoded surface is "
            f"exactly {list(O2_ENCODED_KEYS)} (memory is added by attach_memory)")
    verified: dict[str, tuple[tuple[int, ...], str]] = {}
    for name in O2_ENCODED_KEYS:
        value = data[name]
        expected_shape = O2_SHAPES[name]
        expected_dtype = O2_DTYPES[name]
        shape = _shape_of(value, name)
        if shape != expected_shape:
            raise ObservationError(
                f"{name}: shape {shape} != required {expected_shape}; resizing or "
                f"interpolating into another shape is refused, not adapted")
        dtype = str(getattr(value, "dtype", "") or "")
        if dtype != expected_dtype:
            raise ObservationError(
                f"{name}: dtype {dtype or 'none'} != required {expected_dtype}" +
                ("; a float image has already been normalised — /255 belongs to "
                 "to_float_minibatch only" if expected_dtype == "uint8" else ""))
        verified[name] = (shape, dtype)
    try:
        contracts.check_executed_key_history(data["executed_key_history"])
    except contracts.ContractError as exc:
        raise KeyHistoryError(str(exc)) from exc
    try:
        contracts.check_history_valid_prefix(data["history_valid"])
    except contracts.ContractError as exc:
        raise KeyHistoryError(str(exc)) from exc
    seconds = float(data["elapsed_capture_s"])
    if not math.isfinite(seconds):
        raise ObservationError(f"elapsed_capture_s {seconds!r} is not finite")
    if seconds < 0.0:
        raise ObservationError(f"elapsed_capture_s {seconds} < 0")
    return verified


def attach_memory(observation: Any, memory: Any) -> contracts.ActorObservation:
    """Add a real ``memory`` vector and return the contract-validated record.

    ``memory`` is the model's own recurrent state: it must be a finite
    ``float32 (256,)`` array.  Truth, a teacher vector or a hand-built dict is
    refused by the same sentinel that guards the pixel inputs, and the composed
    record is re-validated by :func:`touhou_rl.v430.contracts.check_actor_observation`
    so the seven fields cannot silently disagree with the contract.
    """

    data = _as_field_mapping(observation)
    check_o2_observation(data)
    assert_actor_inputs_clean({"memory": memory}, what="actor memory")
    array = np.asarray(memory)
    expected_shape = O2_SHAPES["memory"]
    if array.dtype != np.dtype(np.float32):
        raise ObservationError(
            f"memory dtype is {array.dtype}, required float32")
    if array.shape != expected_shape:
        raise ObservationError(
            f"memory shape {array.shape} != {expected_shape}")
    if not bool(np.isfinite(array).all()):
        raise ObservationError("memory holds a non-finite value")
    record = contracts.ActorObservation(
        global_rgb=data["global_rgb"],
        play_gray=data["play_gray"],
        hud_gray=data["hud_gray"],
        executed_key_history=data["executed_key_history"],
        history_valid=data["history_valid"],
        elapsed_capture_s=data["elapsed_capture_s"],
        memory=array,
    )
    try:
        contracts.check_actor_observation(record)
    except contracts.ContractError as exc:
        raise ObservationError(str(exc)) from exc
    return record


def _to_float_one(observation: Any) -> dict[str, np.ndarray]:
    check_o2_observation(observation)
    data = _as_field_mapping(observation)
    scale = np.float32(contracts.O2_NORMALIZE_DIVISOR)
    if float(scale) != 255.0:  # pragma: no cover - frozen parameter
        raise ObservationError(
            f"PLAN_SPEC normalize_divisor is {scale}, not 255.0")
    out: dict[str, np.ndarray] = {}
    for name in ("global_rgb", "play_gray", "hud_gray"):
        out[name] = np.ascontiguousarray(
            np.asarray(data[name], dtype=np.uint8).astype(np.float32) / scale)
    # Key states and validity flags are already 0/1: dividing them by 255 would
    # make them 0/0.0039 and destroy the pressed/released meaning.
    out["executed_key_history"] = np.asarray(
        data["executed_key_history"], dtype=np.uint8).astype(np.float32)
    out["history_valid"] = np.asarray(
        data["history_valid"], dtype=np.bool_).astype(np.float32)
    out["elapsed_capture_s"] = np.float32(data["elapsed_capture_s"])
    return out


def to_float_minibatch(observations: Any) -> dict[str, np.ndarray]:
    """The **single** place ``/255.0`` is applied (one observation or a batch).

    Storage and the rollout keep ``uint8``; only a minibatch is converted.  The
    input is never mutated, and an already-float image is refused (double
    normalisation is a silent 1/65025 error).
    """

    if isinstance(observations, (O2Observation, Mapping)):
        return _to_float_one(observations)
    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise ObservationError(
            f"to_float_minibatch expects one observation or a sequence of them, got "
            f"{type(observations).__name__}")
    items = list(observations)
    if not items:
        raise ObservationError("to_float_minibatch received an empty batch")
    parts = [_to_float_one(item) for item in items]
    keys = tuple(parts[0])
    if any(tuple(part) != keys for part in parts):  # pragma: no cover - same code
        raise ObservationError("batch observations disagree on their field set")
    return {key: np.stack([part[key] for part in parts]) for key in keys}


# ---------------------------------------------------------------------------
# digest: one hashing path, explicit scope, no identity without a scope
# ---------------------------------------------------------------------------
#: The encoding steps, in the order they are applied.  They name parameters
#: symbolically (``play_crop_xywh`` etc.) so the numbers live once, in
#: :func:`touhou_rl.v430.contracts.o2_parameters`.
ENCODER_STEPS: tuple[str, ...] = (
    "1 validate_frame: uint8 (source_h, source_w, 4); the alpha plane is dropped, "
    "never blended (alpha_mode='strict' refuses an undecidable plane, 'any' drops "
    "the plane without inspecting it)",
    "2 rgba_to_rgb: channels in rgb_channel_order (rgba_input_order minus A)",
    "3 global_rgb: downsample_2x2_int(downsample_rule) over the complete source "
    "frame, one plane per RGB channel",
    "4 play_gray: crop(play_crop_xywh) -> gray_bt601_int(gray_rule) -> "
    "downsample_2x2_int(downsample_rule), one plane per frame, play_stack_order",
    "5 hud_gray: crop(hud_crop_xywh) of the current frame -> gray_bt601_int at "
    "native resolution, no downsample",
    "6 executed_key_history: actual executed key states, key_order columns, "
    "oldest first",
    "7 history_valid: bool flags; cold start is all false and never 'executed idle'",
    "8 elapsed_capture_s: the capture clock's own interval, float32 seconds",
    "9 minibatch: divide by normalize_divisor at training time only; storage_dtype "
    "stays uint8",
)


def encoder_parameters(overrides: Mapping[str, Any] | None = None, *,
                       alpha_mode: str = ALPHA_MODE_DEFAULT) -> dict[str, Any]:
    """The encoder parameter block: the contract block **plus** the encoder steps.

    The contract block comes from :func:`touhou_rl.v430.contracts.o2_parameters`
    unchanged, so there is exactly one table of crops/weights/rules in the
    repository.  ``overrides`` exists so a test (or a run-identity check) can prove
    the digest is sensitive to a single changed parameter; an unknown key raises
    instead of silently changing nothing.

    ``alpha_mode`` is the gate a **run** used, and it enters the block as
    ``alpha_mode_used``: a dataset captured through ``alpha_mode="any"`` must not be
    able to share an ``encoder_digest`` with strict-gated data, because a reader
    must be able to tell that the translucent-frame signal was never checked.  The
    default ``"strict"`` reproduces the parameter block of a strict run.
    """

    block: dict[str, Any] = contracts.o2_parameters()
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ObservationError(
                f"overrides must be a mapping, got {type(overrides).__name__}")
        unknown = sorted(name for name in overrides if name not in block)
        if unknown:
            raise ObservationError(
                f"overrides name unknown parameter(s) {unknown}; known keys are "
                f"{sorted(block)} (a typo must not silently change nothing)")
        block.update(overrides)
    block["encoder_version"] = ENCODER_VERSION
    block["encoder_steps"] = list(ENCODER_STEPS)
    block["play_stack_size"] = PLAY_STACK
    block["play_stack_order"] = PLAY_STACK_ORDER
    block["alpha_policy"] = ALPHA_POLICY
    block["alpha_modes_accepted"] = list(ALPHA_MODES)
    block["alpha_mode_default"] = ALPHA_MODE_DEFAULT
    block["alpha_mode_choices"] = list(ALPHA_MODE_CHOICES)
    block["alpha_mode_used"] = check_alpha_mode(alpha_mode)
    block["alpha_channel_index"] = ALPHA_CHANNEL_INDEX
    block["rgb_channel_index"] = list(RGB_CHANNEL_INDEX)
    block["source_shape_rgba"] = list(SOURCE_SHAPE_RGBA)
    block["placeholder_policy"] = (
        "a uniform (constant RGB) frame and a byte-identical repeated frame are "
        "refused, never padded over")
    block["key_history_policy"] = (
        "actual executed key states only; the newest n<=8 rows are the only ones "
        "that can fill slots, a missing row stays invalid")
    block["frame_source"] = "real 640x480 RGBA capture; no generated frame"
    block["digest_scope"] = O2_DIGEST_SCOPE
    return block


def o2_digest(parameters: Mapping[str, Any] | None = None) -> str:
    """Pass-through to :func:`touhou_rl.v430.contracts.o2_digest`.

    ``parameters=None`` hashes ``contracts.o2_parameters()``; an explicit mapping
    is hashed as given.  The canonical JSON has no line breaks, so its raw bytes
    already equal the ``CRLF->LF`` normalised bytes — :func:`sha256_text` proves
    that on the same payload.

    The **applied alpha mode is deliberately not here**: this function is the
    contract's parameter block and must stay byte-identical to
    ``contracts.o2_digest()`` (the four crops/weights/rules are unaffected by which
    alpha gate a run chose).  Provenance lives one level up, in
    :func:`encoder_parameters` / :func:`encoder_digest` (``alpha_mode_used``) and in
    each observation's ``alpha_mode`` field.
    """

    return contracts.o2_digest(parameters)


def encoder_digest(overrides: Mapping[str, Any] | None = None, *,
                   alpha_mode: str = ALPHA_MODE_DEFAULT) -> str:
    """sha256 of :func:`encoder_parameters` — one hashing path, via ``o2_digest``.

    ``alpha_mode`` selects the ``alpha_mode_used`` entry (default ``"strict"``), so
    ``encoder_digest(alpha_mode="any")`` is the identity of a lenient-alpha run and
    differs from the strict one.
    """

    return o2_digest(encoder_parameters(overrides, alpha_mode=alpha_mode))


def canonical_text_bytes(payload: Any) -> bytes:
    """Canonical JSON (sorted keys, no spaces, ASCII) as UTF-8 bytes."""

    try:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ObservationError(
            f"payload is not stably serialisable as JSON: {exc}") from exc
    return text.encode("utf-8")


def sha256_text(payload: Any) -> dict[str, Any]:
    """Text identity: sha256 over the ``CRLF->LF`` normalised canonical JSON.

    Carries ``digest_scope = "text/lf-normalized"`` (W00
    ``SHA_LINE_ENDING_FINDING.md`` §6): a sha256 without a scope cannot be
    reproduced on a machine with different line endings.
    """

    raw = canonical_text_bytes(payload)
    normalised = raw.replace(b"\r\n", b"\n")
    return {
        "sha256": hashlib.sha256(normalised).hexdigest(),
        "digest_scope": O2_DIGEST_SCOPE,
        "digest_algorithm": "sha256",
        "byte_length": len(raw),
        "lf_normalised_byte_length": len(normalised),
    }


def sha256_pixels(arrays: Mapping[str, Any]) -> dict[str, Any]:
    """Binary identity of encoded arrays: sha256 over the **raw** C-order bytes.

    Pixel data must never be line-ending normalised (a ``\\r\\n`` byte pair inside
    an image is image content).  Every digest here carries
    ``digest_scope = "binary/raw"`` and the exact shape/dtype it was computed over,
    so a reader can tell "different bytes" from "different scope".
    """

    if not isinstance(arrays, Mapping):
        raise ObservationError(
            f"sha256_pixels expects a mapping of name -> array, got "
            f"{type(arrays).__name__}")
    if not arrays:
        raise ObservationError("sha256_pixels received no arrays")
    assert_actor_inputs_clean(arrays, what="pixel identity input")
    digest = hashlib.sha256()
    fields: dict[str, Any] = {}
    total = 0
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        raw = array.tobytes(order="C")
        fields[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "digest_scope": PIXEL_DIGEST_SCOPE,
            "digest_algorithm": "sha256",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "byte_length": len(raw),
        }
        for piece in (name.encode("utf-8"), str(array.dtype).encode("utf-8"),
                      str(list(array.shape)).encode("utf-8"), raw):
            digest.update(piece)
            digest.update(b"\x00")
        total += len(raw)
    record = {
        "sha256": digest.hexdigest(),
        "digest_scope": PIXEL_DIGEST_SCOPE,
        "digest_algorithm": "sha256",
        "byte_length": total,
        "fields": fields,
    }
    return check_identity_record(record)


def check_identity_record(record: Any, *, what: str = "identity record") -> dict[str, Any]:
    """Refuse an identity record that carries a sha256 without a scope.

    ``X_digest`` requires ``X_digest_scope``; a bare ``sha256`` requires
    ``digest_scope``.  The value must be 64 lowercase hex characters and the scope
    one of :data:`DIGEST_SCOPES`, so "different content" and "different scope"
    stay distinguishable (W00 finding §6 item 3/4).
    """

    if not isinstance(record, Mapping):
        raise ObservationError(
            f"{what} must be a mapping, got {type(record).__name__}")
    digest_keys = [
        key for key in record
        if isinstance(key, str) and (key == "sha256" or key.endswith("_digest"))]
    if not digest_keys:
        raise ObservationError(
            f"{what} carries no digest field; an identity record without a digest "
            f"identifies nothing")
    for key in digest_keys:
        scope_key = "digest_scope" if key == "sha256" else (
            key[:-len("_digest")] + "_digest_scope")
        scope = record.get(scope_key)
        if scope not in DIGEST_SCOPES:
            raise ObservationError(
                f"{what}: field {key!r} has no digest_scope ({scope_key!r} is "
                f"{scope!r}); allowed scopes are {list(DIGEST_SCOPES)}")
        value = record[key]
        if not isinstance(value, str) or len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value):
            raise ObservationError(
                f"{what}: field {key!r} is not a lowercase sha256 hex digest: "
                f"{value!r}")
        algorithm_key = ("digest_algorithm" if key == "sha256"
                         else key[:-len("_digest")] + "_digest_algorithm")
        algorithm = record.get(algorithm_key)
        if algorithm is not None and algorithm != "sha256":
            raise ObservationError(
                f"{what}: field {key!r} declares algorithm {algorithm!r}, not sha256")
    return dict(record)


def o2_identity(*, alpha_mode: str = ALPHA_MODE_DEFAULT) -> dict[str, Any]:
    """The O2 encoding identity a run/config/checkpoint binds to.

    Both digests are sha256 with an explicit ``text/lf-normalized`` scope:
    ``o2_digest`` covers the contract's parameter block (identical to
    ``contracts.o2_digest()``), ``encoder_digest`` covers that block plus the
    encoder steps and policies.  Pixel identity is a different scope
    (``binary/raw``, see :func:`sha256_pixels`) and is deliberately *not* part of a
    parameter identity.

    ``alpha_mode`` is the gate the run used: it is recorded as both
    ``alpha_mode_used`` and inside ``encoder_digest``, so a checkpoint that binds
    this identity distinguishes "strict-gated data" from ``"any"``-gated data.
    ``o2_digest`` itself stays identical in every mode (see :func:`o2_digest`).
    """

    record = {
        "encoder_version": ENCODER_VERSION,
        "contract_version": contracts.CONTRACT_VERSION,
        "spec_revision": contracts.SPEC_REVISION,
        "o2_digest": o2_digest(),
        "o2_digest_scope": O2_DIGEST_SCOPE,
        "o2_digest_algorithm": "sha256",
        "encoder_digest": encoder_digest(alpha_mode=alpha_mode),
        "encoder_digest_scope": O2_DIGEST_SCOPE,
        "encoder_digest_algorithm": "sha256",
        "alpha_mode_used": check_alpha_mode(alpha_mode),
        "pixel_digest_scope": PIXEL_DIGEST_SCOPE,
        "storage_dtype": contracts.O2_STORAGE_DTYPE,
        "normalize_divisor": contracts.O2_NORMALIZE_DIVISOR,
        "image_bytes_per_decision": contracts.O2_IMAGE_BYTES_PER_DECISION,
        "observation_keys": list(O2_ENCODED_KEYS),
        "shapes": {name: list(shape) for name, shape in O2_SHAPES.items()},
        "parameters": encoder_parameters(alpha_mode=alpha_mode),
    }
    return check_identity_record(record)
