"""Frame-provenance records for captured samples.

The fence build can prove that a captured image and a state read describe the
same presented frame (``DATA-ACQUISITION.md`` §4.1, round26).  Everything else —
the canonical image, plain GDI capture — cannot, and must stay
``alignment_unknown``.  Downstream code must never have to guess which one it has:

* every captured frame carries a :class:`FrameProofRecord`;
* ``exact`` records are rejected unless all 13 checks passed and the build /
  address-table / environment identity is present;
* a dataset that mixes proof levels is refused outright, because "most frames are
  exact" is not a property anyone can use.

The 13 check names mirror ``touhou_rl.framefence.FenceFrameSampler``; the
canonical list lives here so a rename cannot silently drop a check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

PROOF_EXACT = "exact"
PROOF_UNKNOWN = "alignment_unknown"
PROOF_LEVELS = (PROOF_EXACT, PROOF_UNKNOWN)

REQUIRED_CHECKS = (
    "hold_stable",
    "player_position",
    "player_velocity",
    "player_state",
    "focus_mode",
    "gameplay_frame",
    "stage_frames",
    "input_mask",
    "active_bullets",
    "bullet_digest",
    "enemy_digest",
    "pixels_frozen",
    "pixels_match_backbuffer",
)
CHECK_COUNT = len(REQUIRED_CHECKS)

BUILD_IDENTITY_KEYS = ("exeSha256", "textSha256", "addressTableSha256")
ENVIRONMENT_KEYS = ("platform", "gpu", "python", "gitHead")


@dataclass(frozen=True)
class FrameProofRecord:
    """Provenance of one captured frame."""

    proof: str
    logic_frame: int
    ticket: int | None = None
    checks: Mapping[str, bool] = field(default_factory=dict)
    image_sha256: str = ""
    image_path: str = ""
    build: Mapping[str, str] = field(default_factory=dict)
    environment: Mapping[str, object] = field(default_factory=dict)
    hold_us: int | None = None
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "proof": self.proof,
            "logicFrame": int(self.logic_frame),
            "ticket": None if self.ticket is None else int(self.ticket),
            "checks": dict(self.checks),
            "imageSha256": self.image_sha256,
            "imagePath": self.image_path,
            "build": dict(self.build),
            "environment": dict(self.environment),
            "holdUs": None if self.hold_us is None else int(self.hold_us),
            "note": self.note,
        }


def unknown_record(logic_frame: int, **kwargs) -> FrameProofRecord:
    """A frame the capture path cannot prove; the honest default."""
    return FrameProofRecord(proof=PROOF_UNKNOWN, logic_frame=int(logic_frame), **kwargs)


def validate_exact(record: FrameProofRecord) -> list[str]:
    """Reasons why ``record`` cannot be labelled ``exact`` (empty == it can)."""
    problems: list[str] = []
    if record.proof != PROOF_EXACT:
        problems.append(f"proof is {record.proof!r}, not {PROOF_EXACT!r}")
    if record.ticket is None or int(record.ticket) <= 0:
        problems.append("missing ticket")
    if not isinstance(record.logic_frame, int) or record.logic_frame < 0:
        problems.append("invalid logicFrame")
    missing = [name for name in REQUIRED_CHECKS if name not in record.checks]
    if missing:
        problems.append(f"missing checks: {','.join(missing)}")
    failed = [name for name in REQUIRED_CHECKS if record.checks.get(name) is not True]
    if failed:
        problems.append(f"failed checks: {','.join(failed)}")
    extra = [name for name in record.checks if name not in REQUIRED_CHECKS]
    if extra:
        # Extra checks are allowed, but they must not substitute for the 13.
        pass
    if len(record.image_sha256) != 64 or any(
        character not in "0123456789ABCDEF" for character in record.image_sha256
    ):
        problems.append("imageSha256 must be 64 upper-case hex characters")
    if not record.image_path:
        problems.append("missing imagePath")
    for key in BUILD_IDENTITY_KEYS:
        if not record.build.get(key):
            problems.append(f"missing build.{key}")
    for key in ENVIRONMENT_KEYS:
        if key not in record.environment:
            problems.append(f"missing environment.{key}")
    return problems


def require_exact(records: Iterable[FrameProofRecord]) -> None:
    """Raise unless every record is a valid ``exact`` record."""
    reasons: list[str] = []
    for index, record in enumerate(records):
        problems = validate_exact(record)
        if problems:
            reasons.append(f"record {index}: {'; '.join(problems)}")
    if reasons:
        raise ValueError("dataset is not exact:\n" + "\n".join(reasons))


def proof_levels(records: Iterable[FrameProofRecord]) -> set[str]:
    return {record.proof for record in records}


def require_homogeneous(records: Iterable[FrameProofRecord]) -> str:
    """Return the single proof level of a dataset, refusing a mixed one."""
    records = list(records)
    levels = proof_levels(records)
    if not records:
        raise ValueError("empty dataset")
    if len(levels) != 1:
        raise ValueError(f"mixed proof levels: {sorted(levels)}")
    level = levels.pop()
    if level not in PROOF_LEVELS:
        raise ValueError(f"unknown proof level {level!r}")
    return level


def summarize(records: Iterable[FrameProofRecord]) -> dict:
    """Manifest-friendly summary; refuses mixed levels like ``require_homogeneous``."""
    records = list(records)
    level = require_homogeneous(records)
    exact_valid = sum(1 for record in records if not validate_exact(record))
    return {
        "frameProof": level,
        "frames": len(records),
        "exactValidFrames": exact_valid,
        "checkNames": list(REQUIRED_CHECKS),
        "checkCount": CHECK_COUNT,
        "imageHashes": len({record.image_sha256 for record in records}),
    }


def environment_fingerprint(extra: Mapping[str, object] | None = None) -> dict:
    """Stable identity of the capture environment.

    Deliberately small: the fields that change *whether* an exact claim holds
    (OS, GPU driver, window mode) rather than everything about the machine.
    """
    import platform
    import sys

    fingerprint: dict[str, object] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
    }
    if extra:
        fingerprint.update(dict(extra))
    return fingerprint


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()
