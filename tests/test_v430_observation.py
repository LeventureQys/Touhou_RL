"""Behavioural tests for the v4.3.0 O2 pixel encoder (W02 slice A).

Every assertion here pins a *property the encoder promises*, not the existence of
a function.  The properties, and where they are asserted:

* the 2x2 mean is a hand-computable integer rule with a ``uint32`` accumulator
  (``test_downsample_2x2_*``);
* BT.601 luma uses ``(77R + 151G + 28B) >> 8`` and not the float coefficients
  (``test_gray_bt601_*``);
* RGBA -> RGB drops **alpha**, never a colour channel, and anomalous alpha is
  refused (``test_rgba_to_rgb_*``, ``test_alpha_*``);
* the three image planes really are the frozen shapes/dtypes and the play/HUD
  crops cover exactly the documented rectangles
  (``test_encode_*shapes*``, ``test_hud_*``, ``test_play_*``);
* a non-480x640 frame raises instead of being resized
  (``test_encode_refuses_*resize*``);
* a short stack, a black placeholder and a repeated frame are refused instead of
  being padded (``test_encode_refuses_*``);
* truth/teacher/judge state cannot enter the encoder
  (``test_privileged_*``, ``test_observation_carries_no_truth``);
* cold start is ``history_valid = False`` and never "executed idle"
  (``test_cold_start_*``, ``test_key_window_*``);
* storage stays ``uint8``; ``/255`` happens only in ``to_float_minibatch``
  (``test_storage_stays_uint8_*``);
* the digest is stable, sensitive to one changed parameter, carries an explicit
  ``digest_scope``, and pixels are hashed raw (``test_digest_*``,
  ``test_identity_*``, ``test_pixel_digest_*``).

Synthetic: the frames here are constructed numpy arrays.  They prove the encoder's
arithmetic and gates; they are **not** evidence that the game, a capture pipeline
or a session works.  No engine is started, no asset is read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from touhou_rl.v430 import contracts as C
from touhou_rl.v430 import observation as O

#: Source geometry, taken from the module under test (which takes it from the
#: contract) so a silent change fails here too.
HEIGHT, WIDTH = O.SOURCE_HEIGHT, O.SOURCE_WIDTH
#: A pixel outside both the play crop (x >= 32, y >= 16) and the HUD strip
#: (x >= 480), so a per-frame uniqueness marker cannot disturb a crop assertion.
MARKER_XY = (8, 8)

FLAGS_NONE = np.zeros(8, dtype=bool)
FLAGS_FULL = np.ones(8, dtype=bool)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def frame(rgb=0, alpha=255):
    """A 640x480 RGBA frame with a constant colour and constant alpha."""

    if isinstance(rgb, int):
        rgb = (rgb, rgb, rgb)
    out = np.empty((HEIGHT, WIDTH, 4), dtype=np.uint8)
    for index, value in enumerate(rgb):
        out[..., index] = value
    out[..., 3] = alpha
    return out


def stack(base, marker_values=(3, 40, 77, 114), marker_xy=MARKER_XY):
    """Four copies of ``base`` that differ only at one marker pixel.

    The marker sits outside the play crop and outside the HUD strip, so it makes
    the stack four genuinely different frames without touching what a test
    asserts about a plane.
    """

    frames = []
    for value in marker_values:
        item = base.copy()
        y, x = marker_xy
        item[y, x, 0] = value
        frames.append(item)
    return frames


def pattern_frame():
    """A deterministic non-uniform RGB pattern (used for recomputation tests)."""

    rows, columns = np.indices((HEIGHT, WIDTH))
    out = np.empty((HEIGHT, WIDTH, 4), dtype=np.uint8)
    out[..., 0] = (columns % 251).astype(np.uint8)
    out[..., 1] = (rows % 241).astype(np.uint8)
    out[..., 2] = ((columns + rows) % 239).astype(np.uint8)
    out[..., 3] = 255
    return out


def play_frame(inside, outside=200):
    """A frame whose play crop is ``inside`` and whose rest is ``outside``."""

    x, y, width, height = O.PLAY_CROP_XYWH
    out = frame((outside, outside, outside))
    out[y:y + height, x:x + width, :3] = inside
    return out


def hud_frame(inside, outside):
    """A frame whose HUD strip is ``inside`` and whose rest is ``outside``."""

    x, y, width, height = O.HUD_CROP_XYWH
    out = frame((outside, outside, outside))
    out[y:y + height, x:x + width, :3] = inside
    return out


def good_frames():
    return stack(play_frame(90))


def encode(frames=None, key_history=None, history_valid=FLAGS_NONE, seconds=1 / 60):
    return O.encode_o2(good_frames() if frames is None else frames,
                       key_history, history_valid, seconds)


# ---------------------------------------------------------------------------
# 2x2 integer mean
# ---------------------------------------------------------------------------
def test_downsample_2x2_is_hand_computable():
    block = np.array([[0, 1], [2, 255]], dtype=np.uint8)

    out = O.downsample_2x2_int(block)

    assert out.shape == (1, 1)
    assert out.dtype == np.uint8
    # (0 + 1 + 2 + 255) // 4 == 258 // 4 == 64, the hand-computed answer.
    assert int(out[0, 0]) == (0 + 1 + 2 + 255) // 4 == 64


def test_downsample_2x2_uses_a_uint32_accumulator():
    saturated = np.full((2, 2), 255, dtype=np.uint8)

    out = O.downsample_2x2_int(saturated)

    # 255 * 4 == 1020, which in uint8 wraps to 252 and would give 63.
    assert int(out[0, 0]) == 255
    assert (255 * 4) % 256 // 4 == 63  # the value a uint8 accumulator would give
    near = np.array([[255, 255], [255, 254]], dtype=np.uint8)
    assert int(O.downsample_2x2_int(near)[0, 0]) == (255 * 3 + 254) // 4 == 254


def test_downsample_2x2_refuses_odd_shapes_and_non_uint8():
    with pytest.raises(O.ObservationError):
        O.downsample_2x2_int(np.zeros((3, 4), dtype=np.uint8))
    with pytest.raises(O.FrameFormatError):
        O.downsample_2x2_int(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(O.FrameFormatError):
        O.downsample_2x2_int(np.zeros((2, 2, 2), dtype=np.uint8))


# ---------------------------------------------------------------------------
# BT.601 integer luma
# ---------------------------------------------------------------------------
def test_gray_bt601_pure_primaries():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255],
                     [255, 255, 255], [0, 0, 0]]], dtype=np.uint8)

    out = O.gray_bt601_int(rgb)

    assert out.dtype == np.uint8
    assert out.shape == (1, 5)
    red, green, blue, white, black = (int(value) for value in out[0])
    assert red == (77 * 255) >> 8 == 76
    assert green == (151 * 255) >> 8 == 150
    assert blue == (28 * 255) >> 8 == 27
    assert white == 255 and black == 0
    # 76/150/27 are the integer-coefficient answers; the float BT.601
    # coefficients would give floor(0.114 * 255) == 29 for blue.
    assert blue != int(0.114 * 255)


def test_gray_bt601_matches_the_integer_formula_on_a_random_block():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(9, 7, 3), dtype=np.uint8)

    out = O.gray_bt601_int(rgb)

    weights = C.O2_GRAY_WEIGHTS
    assert weights == (77, 151, 28) and C.O2_GRAY_SHIFT == 8
    red = rgb[..., 0].astype(np.uint32)
    green = rgb[..., 1].astype(np.uint32)
    blue = rgb[..., 2].astype(np.uint32)
    expected = ((red * 77 + green * 151 + blue * 28) >> 8).astype(np.uint8)
    assert np.array_equal(out, expected)


def test_gray_bt601_refuses_non_uint8_and_non_rgb():
    with pytest.raises(O.FrameFormatError):
        O.gray_bt601_int(np.zeros((4, 4, 3), dtype=np.float32))
    with pytest.raises(O.FrameFormatError):
        O.gray_bt601_int(np.zeros((4, 4, 4), dtype=np.uint8))


# ---------------------------------------------------------------------------
# RGBA -> RGB: alpha is what gets dropped
# ---------------------------------------------------------------------------
def test_rgba_to_rgb_drops_alpha_not_a_colour_channel():
    source = frame((10, 20, 30), alpha=255)

    rgb = O.rgba_to_rgb(source)

    assert rgb.shape == (HEIGHT, WIDTH, 3)
    assert rgb.dtype == np.uint8
    assert int(rgb[0, 0, 0]) == 10
    assert int(rgb[0, 0, 1]) == 20
    assert int(rgb[0, 0, 2]) == 30
    # Changing *only* alpha cannot change the RGB result: alpha is the dropped one.
    assert np.array_equal(O.rgba_to_rgb(frame((10, 20, 30), alpha=0)), rgb)
    # And the channel order is R,G,B (not B,G,R).
    assert O.RGB_CHANNEL_INDEX == (0, 1, 2)
    assert O.ALPHA_CHANNEL_INDEX == 3
    assert C.RGBA_INPUT_ORDER[O.ALPHA_CHANNEL_INDEX] == "A"


def test_rgba_to_rgb_refuses_wrong_format():
    with pytest.raises(O.FrameFormatError):
        O.rgba_to_rgb(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))  # RGB, no alpha
    with pytest.raises(O.FrameFormatError):
        O.rgba_to_rgb(np.zeros((HEIGHT, WIDTH, 5), dtype=np.uint8))
    with pytest.raises(O.FrameFormatError):
        O.rgba_to_rgb(np.zeros((HEIGHT, WIDTH, 4), dtype=np.float32))
    with pytest.raises(O.FrameFormatError):
        O.rgba_to_rgb(np.zeros((HEIGHT // 2, WIDTH // 2, 4), dtype=np.uint8))


def test_alpha_anomaly_is_refused():
    abnormal = frame((10, 20, 30))
    abnormal[0, 0, 3] = 128

    with pytest.raises(O.FrameAlphaError) as info:
        O.rgba_to_rgb(abnormal)
    assert "alpha" in str(info.value).lower()

    # A constant but neither-0-nor-255 alpha is a blended buffer: refused too.
    with pytest.raises(O.FrameAlphaError):
        O.rgba_to_rgb(frame((10, 20, 30), alpha=200))

    # Binary 0/255 alpha is a mask we deliberately do not apply: accepted, and
    # the RGB bytes are identical to the opaque frame.
    masked = frame((10, 20, 30))
    masked[::2, ::2, 3] = 0
    assert O.alpha_mode(masked) == "binary_mixed"
    assert O.alpha_mode(frame((0, 0, 0), alpha=255)) == "opaque_255"
    assert O.alpha_mode(frame((0, 0, 0), alpha=0)) == "unused_0"
    assert np.array_equal(O.rgba_to_rgb(masked), O.rgba_to_rgb(frame((10, 20, 30))))


# ---------------------------------------------------------------------------
# encode_o2: shapes, dtypes, planes
# ---------------------------------------------------------------------------
def test_encode_o2_shapes_and_dtypes():
    observation = encode()

    params = C.o2_parameters()
    assert list(observation.global_rgb.shape) == params["global_rgb_chw"] == [3, 240, 320]
    assert list(observation.play_gray.shape) == params["play_gray_chw"] == [4, 224, 192]
    assert list(observation.hud_gray.shape) == params["hud_gray_chw"] == [1, 448, 160]
    assert list(observation.executed_key_history.shape) == params["executed_key_history_shape"] == [8, 7]
    assert list(observation.history_valid.shape) == params["history_valid_shape"] == [8]

    for name in ("global_rgb", "play_gray", "hud_gray"):
        assert getattr(observation, name).dtype == np.uint8
    assert observation.executed_key_history.dtype == np.uint8
    assert observation.history_valid.dtype == np.bool_
    assert observation.elapsed_capture_s.shape == ()
    assert isinstance(observation.elapsed_capture_s, np.floating)
    assert observation.elapsed_capture_s.dtype == np.float32

    assert set(observation.as_dict()) == set(O.O2_ENCODED_KEYS)
    assert "memory" not in observation.as_dict()


def test_o2_shapes_are_the_contract_table():
    params = C.o2_parameters()

    assert O.O2_SHAPES["global_rgb"] == tuple(params["global_rgb_chw"])
    assert O.O2_SHAPES["play_gray"] == tuple(params["play_gray_chw"])
    assert O.O2_SHAPES["hud_gray"] == tuple(params["hud_gray_chw"])
    assert O.O2_SHAPES["executed_key_history"] == tuple(params["executed_key_history_shape"])
    assert O.O2_SHAPES["history_valid"] == tuple(params["history_valid_shape"])
    assert O.O2_SHAPES["memory"] == tuple(params["memory_shape"])
    assert O.O2_SHAPES["elapsed_capture_s"] == ()
    # Bidirectional: the module's table *is* the contract's table, not a copy.
    assert O.O2_SHAPES == {name: spec.shape for name, spec in C.FIELD_SPECS.items()}
    assert tuple(O.O2_SHAPES) == C.ACTOR_OBSERVATION_KEYS
    assert O.KEY_ORDER == tuple(params["key_order"]) == C.KEY_ORDER
    assert O.PLAY_STACK == params["play_gray_chw"][0] == 4
    assert O.KEY_SLOTS == 8 and O.KEY_COLUMNS == len(O.KEY_ORDER) == 7
    assert C.O2_IMAGE_BYTES_PER_DECISION == 474112


def test_global_rgb_is_the_whole_frame_2x2_mean():
    # R rises by 10 per frame, G and B are constant; one marker pixel at (0,0).
    frames = []
    for index in range(4):
        item = frame((10 * index, 255, 128))
        item[0, 0, :3] = (255, 255, 128)
        frames.append(item)

    observation = O.encode_o2(frames, None, FLAGS_NONE, 1 / 60)
    plane = observation.global_rgb

    # The *newest* frame is the one encoded: its R is 30, and the marker pixel
    # contributes 255 to a 2x2 block of four (255,30,30,30) -> 345 // 4 = 86.
    assert int(plane[0, 0, 0]) == (255 + 30 + 30 + 30) // 4 == 86
    assert int(plane[0].sum()) == 30 * (240 * 320 - 1) + 86
    assert int(plane[1].min()) == int(plane[1].max()) == 255
    assert int(plane[2].min()) == int(plane[2].max()) == 128
    # Channel order: 30 is R, 128 is B; a BGR swap would show 128 where 30 is.
    assert int(plane[0, 5, 5]) == 30
    assert int(plane[2, 5, 5]) == 128


def test_global_plane_matches_an_independent_recomposition():
    base = pattern_frame()
    frames = stack(base, marker_values=(0, 1, 2, 3))
    newest = frames[-1]

    observation = O.encode_o2(frames, None, FLAGS_NONE, 1 / 60)

    rgb = newest[..., :3].astype(np.uint32)
    channels = []
    for channel in range(3):
        blocks = (rgb[..., channel]
                  .reshape(240, 2, 320, 2)
                  .sum(axis=(1, 3)) // 4)
        channels.append(blocks.astype(np.uint8))
    assert np.array_equal(observation.global_rgb, np.stack(channels, axis=0))


def test_play_plane_matches_an_independent_recomposition():
    base = pattern_frame()
    frames = stack(base, marker_values=(0, 1, 2, 3))

    observation = O.encode_o2(frames, None, FLAGS_NONE, 1 / 60)

    x, y, width, height = O.PLAY_CROP_XYWH
    assert (x, y, width, height) == (32, 16, 384, 448)
    planes = []
    for item in frames:
        rgb = item[..., :3].astype(np.uint32)
        crop = rgb[y:y + height, x:x + width]
        luma = (crop[..., 0] * 77 + crop[..., 1] * 151 + crop[..., 2] * 28) >> 8
        planes.append((luma.reshape(224, 2, 192, 2).sum(axis=(1, 3)) // 4).astype(np.uint8))
    assert np.array_equal(observation.play_gray, np.stack(planes, axis=0))


def test_play_stack_is_oldest_first():
    values = (10, 60, 110, 160)
    frames = [play_frame(value) for value in values]

    observation = O.encode_o2(frames, None, FLAGS_NONE, 1 / 60)

    for index, value in enumerate(values):
        # A grey pixel (v,v,v) has luma exactly v, so the plane is flat.
        assert int(observation.play_gray[index].min()) == value
        assert int(observation.play_gray[index].max()) == value
    assert O.PLAY_STACK_ORDER.startswith("oldest_first")


def test_play_crop_origin_is_exact():
    # The rectangle itself is pinned to PLAN_SPEC, so a changed constant cannot
    # make the marks below self-consistent and pass anyway.
    assert O.PLAY_CROP_XYWH == (32, 16, 384, 448)
    x, y, width, height = O.PLAY_CROP_XYWH

    # 1) Marks *outside* the rectangle never enter the plane: this is the part
    #    that proves the crop does not eat neighbouring pixels.
    outside = frame((100, 100, 100))
    outside[:, x - 1, :3] = 255      # column just left of the crop
    outside[y - 1, :, :3] = 255      # row just above the crop
    outside[:, x + width, :3] = 255  # column just right of the crop
    outside[y + height, :, :3] = 255  # row just below the crop
    observation = O.encode_o2(stack(outside), None, FLAGS_NONE, 1 / 60)
    assert int(observation.play_gray.min()) == 100
    assert int(observation.play_gray.max()) == 100

    # 2) Marks on the first crop row/column, with the 2x2 means computed by hand.
    #    A source row/column pair is what one plane pixel averages, so the four
    #    neighbours of plane[0,0] are (16,32)=(16,33)=(17,32)=255 and (17,33)=100.
    edge = frame((100, 100, 100))
    edge[y, :, :3] = 255   # crop row 0 (source row 16) white
    edge[:, x, :3] = 255   # crop column 0 (source column 32) white
    plane = O.encode_o2(stack(edge), None, FLAGS_NONE, 1 / 60).play_gray[-1]
    assert int(plane[0, 0]) == (255 + 255 + 255 + 100) // 4 == 216
    assert int(plane[0, 1]) == (255 + 255 + 100 + 100) // 4 == 177
    assert int(plane[1, 0]) == (255 + 100 + 255 + 100) // 4 == 177
    assert int(plane[10, 10]) == 100

    # 3) The whole rectangle, in both directions: a crop shifted by one source
    #    pixel would pull a wrong value into the blocks along the moving edge.
    white_rect = frame((0, 0, 0))
    white_rect[y:y + height, x:x + width, :3] = 255
    observation = O.encode_o2(stack(white_rect), None, FLAGS_NONE, 1 / 60)
    assert bool((observation.play_gray == 255).all())

    black_rect = frame((255, 255, 255))
    black_rect[y:y + height, x:x + width, :3] = 0
    observation = O.encode_o2(stack(black_rect), None, FLAGS_NONE, 1 / 60)
    assert bool((observation.play_gray == 0).all())


def test_hud_covers_exactly_the_right_strip():
    x, y, width, height = O.HUD_CROP_XYWH
    assert (x, y, width, height) == (480, 16, 160, 448)

    white_strip = hud_frame(255, 0)
    observation = O.encode_o2(stack(white_strip), None, FLAGS_NONE, 1 / 60)
    assert observation.hud_gray.shape == (1, 448, 160)
    assert bool((observation.hud_gray == 255).all())

    # The reverse direction: everything *outside* the strip is white and the strip
    # is black, so a crop that is off by even one pixel/row shows a 255.
    black_strip = hud_frame(0, 255)
    observation = O.encode_o2(stack(black_strip), None, FLAGS_NONE, 1 / 60)
    assert bool((observation.hud_gray == 0).all())


def test_hud_keeps_native_resolution():
    x, y, _width, _height = O.HUD_CROP_XYWH
    base = frame((0, 0, 0))
    base[y, x, :3] = (255, 0, 0)
    base[y + 1, x + 1, :3] = (0, 0, 255)

    observation = O.encode_o2(stack(base), None, FLAGS_NONE, 1 / 60)
    hud = observation.hud_gray[0]

    # A 2x2 mean would have blended the two neighbours into ~19/~13.
    assert int(hud[0, 0]) == 76
    assert int(hud[1, 1]) == 27
    assert int((hud == 0).sum()) == 448 * 160 - 2


# ---------------------------------------------------------------------------
# encode_o2: refusals
# ---------------------------------------------------------------------------
def test_encode_refuses_a_frame_that_is_not_the_source_size():
    wrong = [np.zeros((240, 320, 4), dtype=np.uint8) for _ in range(4)]

    with pytest.raises(O.FrameFormatError) as info:
        O.encode_o2(wrong, None, FLAGS_NONE, 1 / 60)

    message = str(info.value).lower()
    assert "resize" in message and "interpolat" in message
    # Nothing was silently resized: the size in the message is the source size.
    assert "640" in str(info.value) and "480" in str(info.value)


def test_encode_refuses_a_short_stack():
    short = stack(play_frame(90))[:3]

    with pytest.raises(O.ShortFrameStackError) as info:
        O.encode_o2(short, None, FLAGS_NONE, 1 / 60)

    assert "4" in str(info.value)
    with pytest.raises(O.ShortFrameStackError):
        O.encode_o2([], None, FLAGS_NONE, 1 / 60)
    with pytest.raises(O.ShortFrameStackError):
        O.encode_o2(play_frame(90), None, FLAGS_NONE, 1 / 60)  # one 3-D frame


def test_encode_refuses_a_black_placeholder():
    black = [np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8) for _ in range(4)]

    with pytest.raises(O.PlaceholderFrameError):
        O.encode_o2(black, None, FLAGS_NONE, 1 / 60)

    # A single black frame inside an otherwise good stack is refused too: it is
    # never replaced by a neighbour or by black padding as a "successful" sample.
    mixed = stack(play_frame(90))
    mixed[1] = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    with pytest.raises(O.PlaceholderFrameError):
        O.encode_o2(mixed, None, FLAGS_NONE, 1 / 60)


def test_encode_refuses_a_repeated_frame():
    one = play_frame(90)

    with pytest.raises(O.DuplicateFrameError):
        O.encode_o2([one.copy() for _ in range(4)], None, FLAGS_NONE, 1 / 60)

    # Only two of the four may coincide, and that is still not four frames.
    repeated = [play_frame(20), play_frame(40), play_frame(60), play_frame(60)]
    with pytest.raises(O.DuplicateFrameError):
        O.encode_o2(repeated, None, FLAGS_NONE, 1 / 60)


def test_encode_refuses_frames_with_an_alpha_anomaly():
    frames = stack(play_frame(90))
    frames[2][5, 5, 3] = 128

    with pytest.raises(O.FrameAlphaError):
        O.encode_o2(frames, None, FLAGS_NONE, 1 / 60)


# ---------------------------------------------------------------------------
# truth must not enter the actor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["stage_id", "seed", "engine_frame",
                                  "player_center", "resource_truth", "enemy_table"])
def test_privileged_keyword_is_refused(name):
    with pytest.raises(O.PrivilegedObservationError) as info:
        O.encode_o2(good_frames(), None, FLAGS_NONE, 1 / 60, **{name: 1})

    assert name in str(info.value)


def test_privileged_field_inside_the_frame_container_is_refused():
    frames = [{"frame": item, "engine_frame": 12} for item in good_frames()]

    with pytest.raises(O.PrivilegedObservationError) as info:
        O.encode_o2(frames, None, FLAGS_NONE, 1 / 60)

    assert "engine_frame" in str(info.value)


def test_unknown_keyword_is_refused():
    with pytest.raises(O.ObservationError) as info:
        O.encode_o2(good_frames(), None, FLAGS_NONE, 1 / 60, frames_list=[])

    message = str(info.value).lower()
    assert "unknown keyword" in message and "frames_list" in message


def test_local_name_scan_covers_names_the_contract_does_not_list():
    assert "stage_id" not in C.FORBIDDEN_ACTOR_FIELDS
    assert "player_center" not in C.FORBIDDEN_ACTOR_FIELDS
    assert set(C.FORBIDDEN_ACTOR_FIELDS) <= set(O.FORBIDDEN_INPUT_NAMES)

    for payload in ({"stage_id": 3}, {"player_center": (120, 400)},
                    {"nested": {"player_position": [1, 2]}},
                    [{"live": {"lives": 2}}]):
        with pytest.raises(O.PrivilegedObservationError):
            O.assert_actor_inputs_clean(payload)
    assert O.assert_actor_inputs_clean({"frames": good_frames()}) is not None


def test_observation_carries_no_truth():
    observation = encode()

    assert tuple(observation.as_dict()) == O.O2_ENCODED_KEYS
    assert C.find_privileged_fields(observation) == ()
    assert C.find_privileged_fields(observation.as_dict()) == ()

    record = observation.to_actor_observation(np.zeros(256, dtype=np.float32))
    C.check_actor_observation(record)  # raises if any field is off-contract
    assert set(record.as_dict()) == set(C.ACTOR_OBSERVATION_KEYS)
    assert C.find_privileged_fields(record) == ()


# ---------------------------------------------------------------------------
# key history / history_valid
# ---------------------------------------------------------------------------
def test_cold_start_is_invalid_and_not_idle():
    observation = O.encode_o2(good_frames(), None, FLAGS_NONE, 1 / 60)

    assert observation.history_valid.tolist() == [False] * 8
    assert observation.history_valid_count == 0
    # The rows are zero bytes, but they are *flagged* invalid: a consumer that
    # respects history_valid cannot read them as "we executed idle".
    assert not bool(observation.executed_key_history.any())
    assert O.check_o2_observation(observation)["history_valid"] == ((8,), "bool")

    # Claiming valid history without any executed row would fabricate idle keys.
    with pytest.raises(O.KeyHistoryError):
        O.encode_o2(good_frames(), None, FLAGS_FULL, 1 / 60)
    # A cold start must be declared, not omitted.
    with pytest.raises(O.KeyHistoryError):
        O.encode_o2(good_frames(), None, None, 1 / 60)


def test_key_window_alignment_is_oldest_first_and_trailing():
    rows = np.zeros((3, 7), dtype=np.uint8)
    rows[0, 0] = 1   # oldest: left
    rows[1, 1] = 1   # right
    rows[2, 2] = 1   # newest: up
    flags = np.array([False] * 5 + [True] * 3)

    observation = O.encode_o2(good_frames(), rows, flags, 1 / 60)

    assert observation.history_valid.tolist() == flags.tolist()
    assert observation.executed_key_history[:5].tolist() == [[0] * 7] * 5
    assert observation.executed_key_history[5:].tolist() == rows.tolist()
    assert observation.history_valid_count == 3
    assert O.KEY_ORDER == ("left", "right", "up", "down", "focus", "shoot", "bomb")


def test_history_valid_cannot_claim_more_than_was_supplied():
    rows = np.zeros((3, 7), dtype=np.uint8)

    with pytest.raises(O.KeyHistoryError) as info:
        O.encode_o2(good_frames(), rows, np.array([False] * 4 + [True] * 4), 1 / 60)
    assert "claims 4" in str(info.value)

    # ... and the reverse (more rows than declared-valid slots) is ambiguous.
    with pytest.raises(O.KeyHistoryError):
        O.encode_o2(good_frames(), np.zeros((8, 7), dtype=np.uint8),
                    np.array([False] * 5 + [True] * 3), 1 / 60)


def test_history_valid_hole_is_refused():
    holed = np.array([False, False, True, False, True, True, True, True])

    with pytest.raises(O.KeyHistoryError) as info:
        O.encode_o2(good_frames(), np.ones((5, 7), dtype=np.uint8), holed, 1 / 60)

    assert "False after a valid entry" in str(info.value)


def test_key_rows_must_be_binary_uint8():
    rows = np.zeros((1, 7), dtype=np.uint8)
    rows[0, 0] = 2
    flags = np.array([False] * 7 + [True])

    with pytest.raises(O.KeyHistoryError) as info:
        O.encode_o2(good_frames(), rows, flags, 1 / 60)
    assert "{0,1}" in str(info.value)

    with pytest.raises(O.KeyHistoryError):
        O.encode_o2(good_frames(), np.zeros((1, 7), dtype=np.float32), flags, 1 / 60)
    with pytest.raises(O.KeyHistoryError):
        O.encode_o2(good_frames(), np.zeros((1, 6), dtype=np.uint8), flags, 1 / 60)


# ---------------------------------------------------------------------------
# capture clock
# ---------------------------------------------------------------------------
def test_elapsed_capture_is_a_float32_scalar_on_the_capture_clock():
    observation = O.encode_o2(good_frames(), None, FLAGS_NONE, 1 / 60)

    assert observation.elapsed_capture_s.shape == ()
    assert isinstance(observation.elapsed_capture_s, np.floating)
    assert observation.elapsed_capture_s.dtype == np.float32
    assert abs(float(observation.elapsed_capture_s) - 1 / 60) < 1e-6

    # An integer here is a frame/step counter, not a capture interval.
    with pytest.raises(O.ObservationError) as info:
        O.encode_o2(good_frames(), None, FLAGS_NONE, 2)
    assert "frame" in str(info.value).lower()

    for bad in (0, 0.0, -1 / 60, float("nan"), float("inf"), True, "0.016",
                np.array([1 / 60], dtype=np.float32)):
        with pytest.raises(O.ObservationError):
            O.encode_o2(good_frames(), None, FLAGS_NONE, bad)


# ---------------------------------------------------------------------------
# storage dtype vs minibatch normalisation
# ---------------------------------------------------------------------------
def test_storage_stays_uint8_and_normalisation_is_a_minibatch_step():
    observation = encode()

    for name in ("global_rgb", "play_gray", "hud_gray"):
        assert getattr(observation, name).dtype == np.uint8
    assert observation.executed_key_history.dtype == np.uint8

    batch = O.to_float_minibatch(observation)
    scale = np.float32(255.0)
    for name in ("global_rgb", "play_gray", "hud_gray"):
        assert batch[name].dtype == np.float32
        assert np.array_equal(batch[name], getattr(observation, name).astype(np.float32) / scale)
        assert float(batch[name].min()) >= 0.0
        assert float(batch[name].max()) <= 1.0
    # 0/1 key states must not be divided by 255.
    assert np.array_equal(batch["executed_key_history"],
                          observation.executed_key_history.astype(np.float32))
    assert batch["history_valid"].dtype == np.float32
    # The stored observation is untouched.
    assert observation.global_rgb.dtype == np.uint8


def test_to_float_minibatch_refuses_double_normalisation_and_accepts_a_batch():
    observation = encode()
    already_float = dict(observation.as_dict())
    already_float["global_rgb"] = observation.global_rgb.astype(np.float32)

    with pytest.raises(O.ObservationError) as info:
        O.to_float_minibatch(already_float)
    assert "normalis" in str(info.value).lower()

    with pytest.raises(O.ObservationError):
        O.to_float_minibatch([])

    batched = O.to_float_minibatch([observation, observation])
    assert batched["global_rgb"].shape == (2, 3, 240, 320)
    assert batched["play_gray"].shape == (2, 4, 224, 192)
    assert batched["hud_gray"].shape == (2, 1, 448, 160)
    assert batched["executed_key_history"].shape == (2, 8, 7)
    assert batched["elapsed_capture_s"].shape == (2,)


def test_attach_memory_completes_a_contract_valid_actor_observation():
    observation = encode()
    memory = (np.arange(256, dtype=np.float32) / 256.0).astype(np.float32)

    record = O.attach_memory(observation, memory)

    C.check_actor_observation(record)
    assert record.memory.dtype == np.float32 and record.memory.shape == (256,)

    with pytest.raises(O.ObservationError):
        O.attach_memory(observation, np.zeros(256, dtype=np.float64))
    with pytest.raises(O.ObservationError):
        O.attach_memory(observation, np.zeros(255, dtype=np.float32))
    with pytest.raises(O.ObservationError):
        O.attach_memory(observation, np.full(256, np.nan, dtype=np.float32))
    with pytest.raises(O.PrivilegedObservationError):
        O.attach_memory(observation, {"player_center": (120, 400)})


# ---------------------------------------------------------------------------
# digest / identity
# ---------------------------------------------------------------------------
def test_encoder_parameters_reuse_the_contract_block():
    block = O.encoder_parameters()
    contract_block = C.o2_parameters()

    for key, value in contract_block.items():
        assert block[key] == value          # unchanged, not re-typed here
    assert set(contract_block) < set(block)  # a strict superset, never a copy
    assert block["encoder_version"] == O.ENCODER_VERSION
    assert block["encoder_steps"] == list(O.ENCODER_STEPS)
    assert block["digest_scope"] == O.O2_DIGEST_SCOPE
    assert block["storage_dtype"] == "uint8"
    assert block["normalization"] == "divide by 255 in minibatch"
    assert block["image_bytes_per_decision"] == 474112
    json.dumps(block)  # JSON-serialisable identity

    with pytest.raises(O.ObservationError) as info:
        O.encoder_parameters({"play_crop_x": 33})
    assert "play_crop_x" in str(info.value)

    # A returned block is a fresh object: mutating it cannot poison the contract.
    block["play_crop_xywh"][0] = 0
    assert C.o2_parameters()["play_crop_xywh"] == [32, 16, 384, 448]
    assert O.encoder_parameters()["play_crop_xywh"] == [32, 16, 384, 448]


def test_digest_is_stable_and_sensitive_to_one_changed_parameter():
    params = C.o2_parameters()

    assert O.o2_digest() == O.o2_digest() == C.o2_digest()
    assert O.o2_digest(params) == O.o2_digest()

    # PLAN_SPEC play_crop_xywh = [32, 16, 384, 448]; moving x to 33 must change
    # the identity, because the encoded bytes change.
    moved = dict(params)
    moved["play_crop_xywh"] = [33, 16, 384, 448]
    assert O.o2_digest(moved) != O.o2_digest()

    assert O.encoder_digest() == O.encoder_digest()
    assert O.encoder_digest() != O.o2_digest()  # the steps are covered too
    assert O.encoder_digest({"play_crop_xywh": [33, 16, 384, 448]}) != O.encoder_digest()
    assert O.encoder_digest({"play_crop_xywh": [33, 16, 384, 448]}) == \
        O.encoder_digest({"play_crop_xywh": [33, 16, 384, 448]})

    # The step *order* is part of the identity, not just the parameter values.
    reordered = O.encoder_parameters()
    reordered["encoder_steps"] = list(reversed(reordered["encoder_steps"]))
    assert O.o2_digest(reordered) != O.encoder_digest()


def test_identity_records_carry_an_explicit_digest_scope():
    identity = O.o2_identity()

    assert identity["o2_digest"] == C.o2_digest() == O.o2_digest()
    assert identity["o2_digest_scope"] == "text/lf-normalized"
    assert identity["encoder_digest"] == O.encoder_digest()
    assert identity["encoder_digest_scope"] == "text/lf-normalized"
    assert identity["pixel_digest_scope"] == "binary/raw"
    assert identity["storage_dtype"] == "uint8"
    O.check_identity_record(identity)

    # The declared text scope is the one actually used: the canonical JSON has no
    # CR byte at all (json escapes them), so raw bytes == LF-normalised bytes.
    raw = O.canonical_text_bytes(O.encoder_parameters())
    assert b"\r" not in raw
    assert json.loads(raw.decode("utf-8")) == O.encoder_parameters()
    assert identity["encoder_digest"] == hashlib.sha256(raw).hexdigest()
    assert O.sha256_text(O.encoder_parameters())["sha256"] == O.encoder_digest()
    assert O.sha256_text(O.encoder_parameters())["digest_scope"] == "text/lf-normalized"
    # Even a payload containing CRLF cannot produce a CR in canonical JSON.
    assert b"\r" not in O.canonical_text_bytes({"note": "a\r\nb"})

    # No identity field may exist without a scope (W00 SHA_LINE_ENDING_FINDING §6).
    with pytest.raises(O.ObservationError) as info:
        O.check_identity_record({"o2_digest": "0" * 64})
    assert "digest_scope" in str(info.value)
    with pytest.raises(O.ObservationError):
        O.check_identity_record({})
    with pytest.raises(O.ObservationError):
        O.check_identity_record({"o2_digest": "0" * 64,
                                 "o2_digest_scope": "text/crlf-normalized"})


def test_pixel_digest_is_raw_binary_not_line_ending_normalised():
    probe = np.frombuffer(b"\r\n\x00\xff\x01", dtype=np.uint8)

    record = O.sha256_pixels({"probe": probe})

    assert record["digest_scope"] == "binary/raw"
    field = record["fields"]["probe"]
    assert field["sha256"] == hashlib.sha256(probe.tobytes()).hexdigest()
    lf_normalised = probe.tobytes().replace(b"\r\n", b"\n")
    assert field["sha256"] != hashlib.sha256(lf_normalised).hexdigest()
    assert field["digest_scope"] == "binary/raw"
    assert field["dtype"] == "uint8" and field["shape"] == [5]
    O.check_identity_record(record)

    zeros = np.zeros((2, 2), dtype=np.uint8)
    changed = zeros.copy()
    changed[0, 0] = 1
    assert O.sha256_pixels({"x": zeros})["sha256"] == \
        O.sha256_pixels({"x": zeros.copy()})["sha256"]
    assert O.sha256_pixels({"x": zeros})["sha256"] != \
        O.sha256_pixels({"x": changed})["sha256"]
    with pytest.raises(O.PrivilegedObservationError):
        O.sha256_pixels({"engine_frame": zeros})
    with pytest.raises(O.ObservationError):
        O.sha256_pixels({})


# ---------------------------------------------------------------------------
# purity guard
# ---------------------------------------------------------------------------
def test_encoder_imports_nothing_from_torch_or_the_engine():
    source = Path(O.__file__).read_text(encoding="utf-8")
    imports = [line.strip() for line in source.splitlines()
               if line.strip().startswith(("import ", "from "))]

    assert any(line.startswith("import numpy") for line in imports)
    assert any("touhou_rl.v430" in line for line in imports)
    for line in imports:
        for forbidden in ("torch", "touhou_rl.v400", "touhou_rl.v420", "gym",
                          "teacher", "subprocess", "socket"):
            assert forbidden not in line, f"forbidden import in observation.py: {line}"


# ---------------------------------------------------------------------------
# W03I: the explicit ``alpha_mode`` keyword argument of ``encode_o2``
# ---------------------------------------------------------------------------
# Why this exists: a real engine readback (``glReadPixels`` over the game window)
# returns a *blended* alpha plane, measured **202..219**, not a constant 255.  The
# strict gate refused that buffer as "not lossless" even though alpha is dropped
# and never blended, so the only real data source was locked out by an over-strict
# rule.  ``alpha_mode="any"`` is the explicit opt-in; ``"strict"`` stays the
# default.  The frames below are still synthetic numpy arrays: these tests prove
# the gate, **not** game capability.
def alpha_stack(alpha, base_factory=None):
    """Four marker-distinct copies of ``play_frame(90)`` with a constant alpha."""

    frames = stack(play_frame(90) if base_factory is None else base_factory())
    for item in frames:
        item[..., 3] = alpha
    return frames


def encode_with(frames, alpha_mode, key_history=None, history_valid=FLAGS_NONE,
                seconds=1 / 60):
    """``encode_o2`` with the alpha gate passed explicitly (keyword-only)."""

    return O.encode_o2(frames, key_history, history_valid, seconds,
                       alpha_mode=alpha_mode)


def blended_alpha_stack(low=202, high=219):
    """Four frames whose alpha plane really spans ``low..high`` (a blended read)."""

    frames = stack(play_frame(90))
    ramp = (low + (np.arange(WIDTH, dtype=np.uint16) % (high - low + 1))).astype(
        np.uint8)
    for item in frames:
        item[..., 3] = ramp
    plane = frames[-1][..., 3]
    assert int(plane.min()) == low and int(plane.max()) == high
    assert set(int(value) for value in np.unique(plane)) == set(
        range(low, high + 1))
    return frames


def test_alpha_mode_values_and_default_stay_strict():
    # Four concrete modes, and the default policy name is *not* a fifth verdict.
    assert O.ALPHA_MODES == ("opaque_255", "unused_0", "binary_mixed", "any")
    assert O.ALPHA_STRICT_MODES == ("opaque_255", "unused_0", "binary_mixed")
    assert O.ALPHA_MODE_DEFAULT == "strict"
    assert "strict" not in O.ALPHA_MODES
    assert O.ALPHA_MODE_CHOICES == O.ALPHA_MODES + ("strict",)
    assert set(O.ALPHA_STRICT_MODES) | {"any"} == set(O.ALPHA_MODES)
    assert O.check_alpha_mode("any") == "any"
    assert O.check_alpha_mode() == "strict"


def test_alpha_mode_default_still_refuses_intermediate_alpha():
    # 1. Omitting ``alpha_mode`` keeps the pre-existing gate verbatim.
    frames = alpha_stack(128)

    with pytest.raises(O.FrameAlphaError) as info:
        encode(frames)
    assert "alpha" in str(info.value).lower()

    # The explicit spelling of the default behaves identically.
    with pytest.raises(O.FrameAlphaError):
        encode_with(frames, "strict")

    # The pixel helper keeps the same default too.
    with pytest.raises(O.FrameAlphaError):
        O.rgba_to_rgb(frame((10, 20, 30), alpha=128))
    # ... and alpha=200 (a constant blended plane) is still refused under strict.
    with pytest.raises(O.FrameAlphaError):
        O.rgba_to_rgb(frame((10, 20, 30), alpha=200))


def test_alpha_mode_any_accepts_intermediate_alpha():
    # 2. The one-line opt-in unlocks an undecidable (but dropped) plane.
    frames = alpha_stack(128)

    observation = encode_with(frames, "any")

    assert observation.alpha_mode == "any"
    assert observation.global_rgb.shape == O.O2_SHAPES["global_rgb"]
    assert observation.play_gray.shape == O.O2_SHAPES["play_gray"]
    assert observation.hud_gray.shape == O.O2_SHAPES["hud_gray"]
    # A uniform-alpha plane is not an image: alpha is dropped, so the RGB planes
    # are the same ones the strict gate would produce for an opaque frame.
    assert np.array_equal(observation.global_rgb, encode().global_rgb)


def test_alpha_mode_any_accepts_the_real_engine_blended_alpha_band():
    # 3. Real engine: glReadPixels returns blended alpha, measured 202..219.
    frames = blended_alpha_stack(202, 219)

    with pytest.raises(O.FrameAlphaError) as info:
        encode(frames)
    message = str(info.value)
    assert "202" in message and "219" in message

    observation = encode_with(frames, "any")
    assert observation.alpha_mode == "any"
    assert observation.play_gray.shape == O.O2_SHAPES["play_gray"]

    # Every constant value inside the measured band is accepted, and every one of
    # them produces exactly the same RGB planes as the others (alpha is dropped).
    reference = encode_with(alpha_stack(255), "any")
    for value in range(202, 220):
        observation = encode_with(alpha_stack(value), "any")
        assert observation.alpha_mode == "any"
        assert np.array_equal(observation.global_rgb, reference.global_rgb)
        with pytest.raises(O.FrameAlphaError):  # the strict gate still says no
            encode_with(alpha_stack(value), "strict")


def test_alpha_mode_any_drops_alpha_bit_for_bit():
    # 4. The decisive property: identical RGB with different alpha -> identical
    #    encoded planes.  If alpha leaked into any arithmetic, these would differ.
    opaque = stack(pattern_frame())
    blended = stack(pattern_frame())
    for item in opaque:
        item[..., 3] = 255
    for item in blended:
        item[..., 3] = 210
    assert not np.array_equal(opaque[-1], blended[-1])      # the inputs differ
    assert np.array_equal(opaque[-1][..., :3], blended[-1][..., :3])  # RGB equal

    # The strict gate cannot encode the 210 plane at all; "any" can, twice.
    with pytest.raises(O.FrameAlphaError):
        encode(blended)
    left = encode_with(opaque, "any")
    right = encode_with(blended, "any")

    for name in ("global_rgb", "play_gray", "hud_gray"):
        left_plane = getattr(left, name)
        right_plane = getattr(right, name)
        assert left_plane.shape == right_plane.shape
        assert left_plane.dtype == right_plane.dtype == np.uint8
        assert np.array_equal(left_plane, right_plane), f"{name} differs by alpha"
        # Byte-for-byte, not just numerically equal.
        assert left_plane.tobytes() == right_plane.tobytes(), f"{name} bytes differ"

    # The same holds one level down, in the RGBA -> RGB drop itself.
    assert O.rgba_to_rgb(blended[-1], alpha_mode="any").tobytes() == \
        O.rgba_to_rgb(opaque[-1], alpha_mode="any").tobytes()
    assert O.sha256_pixels({"global_rgb": left.global_rgb})["sha256"] == \
        O.sha256_pixels({"global_rgb": right.global_rgb})["sha256"]


def test_alpha_mode_refuses_an_unknown_value():
    # 5. An unknown mode raises the module's existing exception type; it never
    #    silently falls back to either gate.
    frames = alpha_stack(128)

    for bad in ("whatever", "ANY", "any ", "", "lenient"):
        with pytest.raises(O.ObservationError) as info:
            encode_with(frames, bad)
        assert "alpha_mode" in str(info.value)
    with pytest.raises(O.ObservationError):
        encode_with(frames, None)          # not a string
    with pytest.raises(O.ObservationError):
        encode_with(frames, 255)
    with pytest.raises(O.ObservationError):
        O.rgba_to_rgb(frame((10, 20, 30), alpha=128), alpha_mode="whatever")
    with pytest.raises(O.ObservationError):
        O.encoder_parameters(alpha_mode="whatever")
    with pytest.raises(O.ObservationError):
        O.encoder_digest(alpha_mode="whatever")


def test_observation_records_the_alpha_mode_actually_used():
    # 6. Provenance is readable on the record, and it is not a seventh field.
    opaque = encode()
    unused = encode(alpha_stack(0))
    masked_frames = stack(play_frame(90))
    for item in masked_frames:
        item[::2, ::2, 3] = 0
    masked = encode(masked_frames)
    lenient = encode_with(alpha_stack(210), "any")

    assert opaque.alpha_mode == "opaque_255"
    assert unused.alpha_mode == "unused_0"
    assert masked.alpha_mode == "binary_mixed"
    assert lenient.alpha_mode == "any"
    for observation in (opaque, unused, masked, lenient):
        assert observation.alpha_mode in O.ALPHA_MODE_CHOICES

    # The encoded surface is unchanged: six fields, in contract order, no alpha.
    assert tuple(lenient.as_dict()) == O.O2_ENCODED_KEYS
    assert set(lenient.as_dict()) == set(O.O2_ENCODED_KEYS)
    assert "alpha_mode" not in lenient.as_dict()
    assert set(O.to_float_minibatch(lenient)) == set(O.O2_ENCODED_KEYS)
    # ... and the composed actor record still validates.
    record = O.attach_memory(lenient, np.zeros(256, dtype=np.float32))
    assert set(record.as_dict()) == set(C.ACTOR_OBSERVATION_KEYS)


def test_alpha_mode_enters_the_encoder_identity_but_not_o2_digest():
    # 7. Digest decision: the *contract* block (_o2_digest_) is mode-independent
    #    and byte-identical; the encoder identity records which gate a run used.
    assert O.o2_digest() == C.o2_digest()
    assert O.encoder_parameters()["alpha_mode_used"] == "strict"
    assert O.encoder_parameters(alpha_mode="any")["alpha_mode_used"] == "any"
    assert O.encoder_parameters()["alpha_modes_accepted"] == list(O.ALPHA_MODES)
    assert "any" in O.encoder_parameters()["alpha_modes_accepted"]

    strict_digest = O.encoder_digest()
    lenient_digest = O.encoder_digest(alpha_mode="any")
    assert strict_digest == O.encoder_digest() == O.encoder_digest(alpha_mode="strict")
    assert lenient_digest == O.encoder_digest(alpha_mode="any")
    assert lenient_digest != strict_digest, "any-gated data must not share an identity"

    identity = O.o2_identity(alpha_mode="any")
    assert identity["alpha_mode_used"] == "any"
    assert identity["encoder_digest"] == lenient_digest
    # The contract parameter identity is untouched by the gate.
    assert identity["o2_digest"] == C.o2_digest()
    assert O.o2_identity()["alpha_mode_used"] == "strict"
    O.check_identity_record(identity)


def test_alpha_mode_any_does_not_relax_the_other_gates():
    # 8. "any" touches the alpha plane only.  Short stack, placeholder and
    #    duplicate frames stay refused (no pre-multiply, no padding, no fill).
    blank = [frame((0, 0, 0), alpha=128).copy() for _ in range(4)]
    with pytest.raises(O.PlaceholderFrameError):
        encode_with(blank, "any")

    duplicated = alpha_stack(128)
    duplicated[1] = duplicated[0].copy()
    with pytest.raises(O.DuplicateFrameError):
        encode_with(duplicated, "any")

    with pytest.raises(O.ShortFrameStackError):
        encode_with(alpha_stack(128)[:2], "any")

    with pytest.raises(O.FrameFormatError):
        encode_with([item[:240] for item in alpha_stack(128)], "any")
