"""Unit tests for the LP_NUM_THREADS pixel-consistency comparison.

These tests cover the *comparison arithmetic and fail-closed logic* only.  A
synthetic framebuffer is never evidence about the engine's pixels; that comes
from the real 100-frame captures recorded in EXP-P0-069.
"""

import importlib.util
import struct
import sys
from pathlib import Path

import pytest

TOOL = (Path(__file__).resolve().parents[1] / "tools" / "rl"
        / "v400_render_pixel_consistency.py")


def _load():
    spec = importlib.util.spec_from_file_location("v400_render_pixel_consistency", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pix = _load()

FRAME = pix.PIXEL_BYTES


def _blank(value: int = 0) -> bytes:
    return bytes([value]) * FRAME


def _with_byte(raw: bytes, offset: int, value: int) -> bytes:
    out = bytearray(raw)
    out[offset] = value
    return bytes(out)


def _shm(frame: int = 1, *, magic: int = pix.SHM_MAGIC,
         version: int = pix.SHM_VERSION, width: int = 640, height: int = 480,
         flags: int = 3, payload: bytes | None = None) -> bytes:
    # The engine's header *record* is 40 bytes; the pixel payload still starts
    # at offset 192, so the segment is header + reserved padding + RGBA.
    record = pix.HEADER.pack(magic, version, frame, 0, width, height, width * 4, flags)
    header = record + bytes(pix.SHM_HEADER_SIZE - len(record))
    assert len(header) == pix.SHM_HEADER_SIZE
    return header + (payload if payload is not None else _blank())


def _capture(tag: str, lp: int, hashes: list, checksums: list | None = None) -> dict:
    checksums = checksums or ["abc"] * len(hashes)
    return {"tag": tag, "lp_threads": lp,
            "frames": [{"frame": index + 1, "rgba_sha256": digest,
                        "pixels_checksum": checksums[index]}
                       for index, digest in enumerate(hashes)]}


# ------------------------------------------------------------------ protocol
def test_parse_reply_splits_verb_and_fields():
    parsed = pix.parse_reply("OK pixels=640x480 nonzero=1228800 checksum=a1eff2690fe0f694")
    assert parsed["_verb"] == "OK"
    assert parsed["pixels"] == "640x480"
    assert parsed["nonzero"] == "1228800"
    assert parsed["checksum"] == "a1eff2690fe0f694"


def test_parse_reply_of_an_empty_line_has_no_verb():
    assert pix.parse_reply("")["_verb"] == ""


# ---------------------------------------------------------------------- arms
def test_arm_specs_keep_order_and_thread_counts():
    arms = pix.parse_arm_specs(["lp1:1", "lp4:4"])
    assert arms == [{"name": "lp1", "lp_threads": 1},
                    {"name": "lp4", "lp_threads": 4}]


@pytest.mark.parametrize("spec", ["lp4", "lp4:", ":4", "lp4:x", "lp4:0", "lp4:-2"])
def test_arm_specs_reject_malformed_specs(spec):
    with pytest.raises(pix.PixelCheckUnavailable):
        pix.parse_arm_specs([spec])


def test_arm_specs_reject_duplicate_names():
    with pytest.raises(pix.PixelCheckUnavailable, match="duplicate arm"):
        pix.parse_arm_specs(["lp1:1", "lp1:2"])


def test_arm_specs_require_at_least_one_arm():
    with pytest.raises(pix.PixelCheckUnavailable, match="at least one"):
        pix.parse_arm_specs([])
    with pytest.raises(pix.PixelCheckUnavailable):
        pix.parse_arm_specs(None)


def test_masks_are_one_per_frame_and_constant():
    assert pix.masks_for(3, 0) == [0, 0, 0]
    assert pix.masks_for(2, 5) == [5, 5]
    with pytest.raises(ValueError):
        pix.masks_for(0)


# ------------------------------------------------------------------ SHM header
def test_shm_header_is_accepted_when_it_matches_the_expected_frame():
    fields = pix.validate_shm_payload(_shm(frame=7), expected_frame=7)
    assert fields["frame"] == 7
    assert fields["width"] == 640 and fields["height"] == 480


@pytest.mark.parametrize("kwargs,expected", [
    ({"magic": 0xDEADBEEF}, "magic"),
    ({"version": 2}, "version"),
    ({"width": 320}, "320x480"),
    ({"height": 240}, "640x240"),
    ({"flags": 1}, "pixel-valid"),
])
def test_shm_header_mismatches_fail_closed(kwargs, expected):
    with pytest.raises(pix.PixelCheckUnavailable, match=expected):
        pix.validate_shm_payload(_shm(**kwargs), expected_frame=1)


def test_shm_header_frame_mismatch_fails_closed():
    """A stale segment would silently compare the wrong picture."""
    with pytest.raises(pix.PixelCheckUnavailable, match="header frame is 3"):
        pix.validate_shm_payload(_shm(frame=3), expected_frame=4)


def test_shm_payload_size_mismatch_fails_closed():
    with pytest.raises(pix.PixelCheckUnavailable, match="size is"):
        pix.validate_shm_payload(b"short", expected_frame=1)


# ------------------------------------------------------------------ byte store
def test_byte_store_deduplicates_identical_frames():
    store = pix.ByteStore()
    first = store.put(_blank(0))
    second = store.put(_blank(0))
    third = store.put(_blank(9))
    assert first == second != third
    assert len(store) == 2
    assert store.puts == 3
    assert store.get(first) == _blank(0)


def test_byte_store_reports_an_unknown_digest_instead_of_inventing_bytes():
    with pytest.raises(pix.PixelCheckUnavailable, match="never captured"):
        pix.ByteStore().get("0" * 64)


# ------------------------------------------------------------------ bbox / diff
def test_diff_bbox_of_an_all_zero_mask_is_none():
    assert pix.diff_bbox(_blank(0)) is None


def test_diff_bbox_maps_bytes_to_pixels():
    # byte offset  (row 2, col 3) -> 2*2560 + 3*4
    assert pix.diff_bbox(_with_byte(_blank(0), 2 * 2560 + 3 * 4, 1)) == (3, 2, 3, 2)


def test_diff_bbox_covers_two_corners():
    mask = bytearray(_blank(0))
    mask[0] = 1                                  # (0,0)
    mask[(479 * 640 + 639) * 4 + 3] = 1          # (639,479) last channel
    assert pix.diff_bbox(bytes(mask)) == (0, 0, 639, 479)


def test_diff_rgba_identical_frames_take_the_fast_path():
    result = pix.diff_rgba(_blank(1), _blank(1))
    assert result["identical"] is True
    assert result["differing_bytes"] == 0
    assert result["differing_pixels"] == 0
    assert result["bbox"] is None


def test_diff_rgba_reports_exact_size_and_location_of_one_changed_pixel():
    left = _blank(0)
    right = _with_byte(left, (10 * 640 + 20) * 4, 0xFF)
    result = pix.diff_rgba(left, right)
    assert result["identical"] is False
    assert result["differing_bytes"] == 1
    assert result["differing_pixels"] == 1
    assert result["bbox"] == (20, 10, 20, 10)


def test_diff_rgba_counts_pixels_not_bytes_for_a_changed_rgba_quad():
    left = _blank(0)
    right = _with_byte(_with_byte(_with_byte(_with_byte(left, 8, 1), 9, 2), 10, 3), 11, 4)
    result = pix.diff_rgba(left, right)
    assert result["differing_bytes"] == 4
    assert result["differing_pixels"] == 1
    assert result["bbox"] == (2, 0, 2, 0)


def test_diff_rgba_of_two_completely_different_frames_covers_the_screen():
    result = pix.diff_rgba(_blank(0), _blank(1))
    assert result["differing_bytes"] == FRAME
    assert result["differing_pixels"] == 640 * 480
    assert result["bbox"] == (0, 0, 639, 479)


def test_diff_rgba_rejects_frames_of_the_wrong_size():
    with pytest.raises(ValueError):
        pix.diff_rgba(b"a", b"b")
    with pytest.raises(ValueError):
        pix.diff_rgba(_blank(0), _blank(0) + b"x")


# ------------------------------------------------------------------ comparison
def test_compare_identical_captures_reports_no_difference():
    store = pix.ByteStore()
    digest = store.put(_blank(2))
    left = _capture("lp1.i0", 1, [digest, digest])
    right = _capture("lp4.i0", 4, [digest, digest])
    result = pix.compare_captures(left, right, store)
    assert result["identical"] is True
    assert result["frames_compared"] == 2
    assert result["different_frame_count"] == 0
    assert result["first_different_frame"] is None
    assert result["checksum_mismatch_frames"] == []


def test_compare_reports_the_first_differing_frame_and_its_size():
    store = pix.ByteStore()
    base = store.put(_blank(0))
    changed = store.put(_with_byte(_blank(0), (5 * 640 + 6) * 4, 0x7F))
    left = _capture("lp1.i0", 1, [base, base, base])
    right = _capture("lp4.i0", 4, [base, changed, changed])
    result = pix.compare_captures(left, right, store)
    assert result["identical"] is False
    assert result["frames_compared"] == 3
    assert result["different_frame_count"] == 2
    assert result["first_different_frame"] == 2
    assert result["first_different_bytes"] == 1
    assert result["first_different_pixels"] == 1
    assert result["first_bbox"] == [6, 5, 6, 5]
    assert result["union_bbox"] == [6, 5, 6, 5]


def test_compare_flags_a_checksum_mismatch_even_when_bytes_match():
    store = pix.ByteStore()
    digest = store.put(_blank(3))
    left = _capture("lp1.i0", 1, [digest], ["aaa"])
    right = _capture("lp4.i0", 4, [digest], ["bbb"])
    result = pix.compare_captures(left, right, store)
    assert result["identical"] is True
    assert result["checksum_mismatch_frames"] == [1]


def test_compare_records_incomplete_captures_rather_than_padding_them():
    store = pix.ByteStore()
    digest = store.put(_blank(0))
    left = _capture("lp1.i0", 1, [digest, digest])
    right = _capture("lp4.i0", 4, [digest])
    result = pix.compare_captures(left, right, store)
    assert result["frames_compared"] == 1
    assert result["frames_missing"] == 1


def test_compare_truncates_the_detail_list_but_keeps_the_count():
    store = pix.ByteStore()
    base = store.put(_blank(0))
    changed = store.put(_blank(1))
    left = _capture("lp1.i0", 1, [base] * 12)
    right = _capture("lp4.i0", 4, [changed] * 12)
    result = pix.compare_captures(left, right, store, max_detail=5)
    assert result["different_frame_count"] == 12
    assert len(result["differing_frames"]) == 5
    assert result["differing_frames_truncated"] == 7


# -------------------------------------------------------------------- verdict
def _comparison(left_lp: int, right_lp: int, identical: bool = True,
                different: int = 0, missing: int = 0) -> dict:
    return {"left": "a", "right": "b", "left_lp_threads": left_lp,
            "right_lp_threads": right_lp, "frames_compared": 100,
            "frames_missing": missing, "identical": identical,
            "different_frame_count": different, "first_different_frame": None,
            "first_different_bytes": None, "first_different_pixels": None,
            "first_bbox": None, "union_bbox": None,
            "checksum_mismatch_frames": []}


def test_verdict_is_invariant_when_every_cross_arm_pair_matches():
    verdict = pix.summarise([_comparison(1, 1), _comparison(4, 4), _comparison(1, 4)])
    assert verdict["lp_invariant_bytes"] is True
    assert verdict["within_lp_identical"] is True
    assert verdict["cross_lp_identical"] is True
    assert verdict["noise_floor_different_frames"] == 0


def test_verdict_is_not_invariant_when_one_cross_arm_pair_differs():
    verdict = pix.summarise([_comparison(1, 4, identical=False, different=3)])
    assert verdict["lp_invariant_bytes"] is False
    assert verdict["cross_lp_identical"] is False


def test_verdict_is_not_invariant_when_a_comparison_was_incomplete():
    verdict = pix.summarise([_comparison(1, 4, missing=4)])
    assert verdict["incomplete_comparisons"] == 1
    assert verdict["lp_invariant_bytes"] is False


def test_within_arm_noise_is_reported_separately_from_the_lp_comparison():
    verdict = pix.summarise([_comparison(1, 1, identical=False, different=2),
                             _comparison(1, 4)])
    assert verdict["within_lp_identical"] is False
    assert verdict["noise_floor_different_frames"] == 2
    assert verdict["cross_lp_identical"] is True
    assert verdict["lp_invariant_bytes"] is True


# ------------------------------------------------------------------ exit codes
@pytest.mark.parametrize("status,verdict,expected", [
    ("measured", {"lp_invariant_bytes": True}, 0),
    ("measured", {"lp_invariant_bytes": False}, 4),
    ("measured", None, 4),
    ("invalid", {"lp_invariant_bytes": True}, 3),
    ("unavailable", None, 3),
])
def test_exit_codes_distinguish_identical_divergent_and_invalid(status, verdict, expected):
    assert pix.exit_code_for(status, verdict) == expected


# ------------------------------------------------------------------ environment
def test_software_arm_env_forces_the_llvmpipe_knobs():
    env = pix.build_arm_env("software", display=":98", lp_threads=4,
                            base_env={"PATH": "/usr/bin", "LP_NUM_THREADS": "1"})
    assert env["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert env["GALLIUM_DRIVER"] == "llvmpipe"
    assert env["LP_NUM_THREADS"] == "4"
    assert env["DISPLAY"] == ":98"
    assert env["SDL_AUDIODRIVER"] == "dummy"


def test_hardware_arm_env_strips_and_then_refuses_a_software_knob():
    env = pix.build_arm_env("hardware", display=":0", lp_threads=1,
                            base_env={"LIBGL_ALWAYS_SOFTWARE": "1", "PATH": "/x"})
    assert "LIBGL_ALWAYS_SOFTWARE" not in env
    assert pix.backend_conflict("hardware", "LIBGL_ALWAYS_SOFTWARE", "1")
    assert not pix.backend_conflict("hardware", "LP_NUM_THREADS", "4")
    assert not pix.backend_conflict("software", "LIBGL_ALWAYS_SOFTWARE", "1")


def test_unknown_backend_and_zero_threads_are_rejected():
    with pytest.raises(ValueError):
        pix.build_arm_env("vulkan", display=None, lp_threads=1)
    with pytest.raises(ValueError):
        pix.build_arm_env("software", display=None, lp_threads=0)


def test_gl_fingerprint_is_per_pid_and_matched_on_renderer():
    text = ("pid=11\nGL_RENDERER=llvmpipe (LLVM 15.0.7, 256 bits)\n"
            "pid=22\nGL_RENDERER=D3D12 (NVIDIA)\n")
    assert pix.renderer_matches("llvmpipe", pix.parse_gl_fingerprint(text, 11))
    assert not pix.renderer_matches("llvmpipe", pix.parse_gl_fingerprint(text, 22))
    assert not pix.renderer_matches("llvmpipe", {})  # unproven is never a match
    assert pix.renderer_matches(None, {})


# ------------------------------------------------------------------ data dirs
def _template(root: Path) -> Path:
    template = root / "template"
    template.mkdir()
    for name in ("th08.dat", "thbgm.dat", "th08.cfg", "score.dat"):
        (template / name).write_bytes(b"x")
    return template


def test_prepare_data_dirs_creates_one_dir_per_label(tmp_path):
    template = _template(tmp_path)
    result = pix.prepare_data_dirs(template, tmp_path / "run", ["lp1-i0", "lp4-i0"])
    assert [Path(p).name for p in result["dirs"]] == ["lp1-i0", "lp4-i0"]
    for path in result["dirs"]:
        assert (Path(path) / "th08.dat").is_file()
        assert not (Path(path) / "score.dat").is_symlink()


def test_prepare_data_dirs_refuses_to_reuse_an_existing_dir(tmp_path):
    template = _template(tmp_path)
    pix.prepare_data_dirs(template, tmp_path / "run", ["lp1-i0"])
    with pytest.raises(pix.PixelCheckUnavailable, match="refusing to reuse"):
        pix.prepare_data_dirs(template, tmp_path / "run", ["lp1-i0"])


def test_prepare_data_dirs_requires_a_th08_dat_template(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(pix.PixelCheckUnavailable, match="no th08.dat"):
        pix.prepare_data_dirs(empty, tmp_path / "run", ["lp1-i0"])


# ---------------------------------------------------------------------- CLI
def test_cli_rejects_nonpositive_instances_and_frames():
    parser = pix.build_parser()

    def parse(*extra):
        return parser.parse_args(["--binary", __file__, "--data-template", ".",
                                  "--run-root", "r", "--out", "o.json", *extra])

    with pytest.raises(SystemExit):
        pix.main(["--binary", __file__, "--data-template", ".", "--run-root", "r",
                  "--out", "o.json", "--instances", "0"])
    with pytest.raises(SystemExit):
        pix.main(["--binary", __file__, "--data-template", ".", "--run-root", "r",
                  "--out", "o.json", "--frames", "0"])
    assert parse().instances == 2  # two concurrent instances per arm by default
    assert parse().frames == 100


def test_cli_without_arms_writes_an_invalid_report_and_exits_three(tmp_path):
    out = tmp_path / "report.json"
    code = pix.main(["--binary", __file__, "--data-template", str(tmp_path),
                     "--run-root", str(tmp_path / "run"), "--out", str(out)])
    assert code == 3
    report = __import__("json").loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "invalid"
    assert "at least one --arm" in report["invalid_reason"]
