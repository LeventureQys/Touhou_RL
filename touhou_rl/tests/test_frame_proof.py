"""Unit tests for the frame-proof records (no game, no capture)."""

from __future__ import annotations

import pytest

from touhou_rl import frame_proof as fp

FAKE_BUILD = {
    "exeSha256": "A" * 64,
    "textSha256": "B" * 64,
    "addressTableSha256": "C" * 64,
}
FAKE_ENV = {
    "platform": "Windows-test",
    "gpu": ["GPU, 1.2.3"],
    "python": "3.12",
    "gitHead": "deadbeef",
}


def _checks(passing: bool = True) -> dict[str, bool]:
    return {name: passing for name in fp.REQUIRED_CHECKS}


def _exact(**overrides) -> fp.FrameProofRecord:
    payload = {
        "proof": fp.PROOF_EXACT,
        "logic_frame": 1000,
        "ticket": 5,
        "checks": _checks(),
        "image_sha256": "D" * 64,
        "image_path": "samples/0000-ticket-000005.png",
        "build": dict(FAKE_BUILD),
        "environment": dict(FAKE_ENV),
        "hold_us": 8000,
    }
    payload.update(overrides)
    return fp.FrameProofRecord(**payload)


def test_check_list_matches_the_sampler():
    from touhou_rl.framefence import FenceFrameSampler
    import inspect

    source = inspect.getsource(FenceFrameSampler._attempt)
    for name in fp.REQUIRED_CHECKS:
        assert name in source, name
    assert fp.CHECK_COUNT == 13


def test_valid_exact_record_has_no_problems():
    assert fp.validate_exact(_exact()) == []


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"proof": fp.PROOF_UNKNOWN}, "not 'exact'"),
        ({"ticket": None}, "missing ticket"),
        ({"image_sha256": "not-a-hash"}, "imageSha256"),
        ({"image_path": ""}, "missing imagePath"),
        ({"build": {}}, "build.exeSha256"),
        ({"environment": {}}, "environment.platform"),
        ({"logic_frame": -1}, "invalid logicFrame"),
    ],
)
def test_invalid_exact_records_are_rejected(overrides, fragment):
    problems = fp.validate_exact(_exact(**overrides))
    assert any(fragment in problem for problem in problems), problems


def test_a_single_failed_check_blocks_exact():
    checks = _checks()
    checks["pixels_match_backbuffer"] = False
    problems = fp.validate_exact(_exact(checks=checks))
    assert any("pixels_match_backbuffer" in problem for problem in problems)


def test_missing_check_blocks_exact():
    checks = _checks()
    checks.pop("enemy_digest")
    problems = fp.validate_exact(_exact(checks=checks))
    assert any("missing checks" in problem for problem in problems)


def test_unknown_record_factory_is_not_exact():
    record = fp.unknown_record(42, image_path="x.png", image_sha256="E" * 64)
    assert record.proof == fp.PROOF_UNKNOWN
    assert fp.validate_exact(record)


def test_require_exact_raises_on_the_first_bad_record():
    records = [_exact(), _exact(ticket=None)]
    with pytest.raises(ValueError) as info:
        fp.require_exact(records)
    assert "record 1" in str(info.value)


def test_require_homogeneous_refuses_mixed_datasets():
    mixed = [_exact(), fp.unknown_record(7)]
    with pytest.raises(ValueError) as info:
        fp.require_homogeneous(mixed)
    assert "mixed proof levels" in str(info.value)
    assert fp.require_homogeneous([fp.unknown_record(7), fp.unknown_record(8)]) == (
        fp.PROOF_UNKNOWN
    )
    assert fp.require_homogeneous([_exact(), _exact(ticket=6)]) == fp.PROOF_EXACT


def test_summary_counts_exact_valid_frames():
    summary = fp.summarize([_exact(), _exact(ticket=6)])
    assert summary["frameProof"] == fp.PROOF_EXACT
    assert summary["frames"] == 2
    assert summary["exactValidFrames"] == 2
    assert summary["checkCount"] == 13
    assert summary["imageHashes"] == 1  # both records reuse the same hash


def test_environment_fingerprint_carries_the_given_fields():
    fingerprint = fp.environment_fingerprint({"gpu": ["GPU, 1.0"]})
    assert fingerprint["gpu"] == ["GPU, 1.0"]
    assert "platform" in fingerprint
