"""Build / inspect the TH08 human reference used by the WP7 frame caps.

Two modes, deliberately different:

``--measured``
    Build ``bench/human_reference.json`` from engine-measured evidence
    (EXP-P0-046: ``STATE`` counters collected while the real engine played the
    retail replays).  Exit 0 only when a validated document was written.

``REPLAY`` (positional)
    Inspect a replay file's verified stream extents.  The codec certifies no
    gameplay frames, deaths or bombs, so this mode never writes a reference and
    exits 3: evidence is not a measurement.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from touhou_rl.v400.bench.human_reference import (  # noqa: E402
    HumanReferenceError, MissingHumanMeasurements, build_human_reference,
    build_reference_from_measurements, caps_for, inspect_human_replay, write_reference,
)

DEFAULT_OUTPUT = ROOT / "Document/Update/Alpha/v4.0.0/bench/human_reference.json"


def _measured_at(path: Path, explicit: str | None) -> tuple[str, str]:
    if explicit:
        return datetime.fromisoformat(explicit).isoformat(timespec="seconds"), "operator supplied (--measured-at)"
    stamp = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    return stamp, "file mtime of the measured evidence JSON (EXP-P0-046 recorded no in-band clock)"


def _build(measured_path: Path, output: Path | None, *, measured_at: str | None,
           engine_sha256: str | None, engine_sha256_source: str | None,
           dry_run: bool) -> int:
    try:
        measured = json.loads(measured_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "reason": f"cannot read measured evidence: {exc}"},
                         ensure_ascii=False))
        return 3
    try:
        stamp, stamp_source = _measured_at(measured_path, measured_at)
        document = build_reference_from_measurements(
            measured, measured_at=stamp, measured_at_source=stamp_source,
            evidence_path=str(measured_path.resolve()),
            engine_sha256=engine_sha256, engine_sha256_source=engine_sha256_source,
        )
    except (OSError, ValueError, HumanReferenceError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "human_reference_created": False},
                         ensure_ascii=False))
        return 3
    caps = caps_for(document)
    if dry_run:
        print(json.dumps({"status": "dry_run", "document": document, "caps": caps},
                         ensure_ascii=False, indent=2))
        return 0
    if output is None:
        print(json.dumps({"status": "blocked", "reason": "--output is required to write the reference"},
                         ensure_ascii=False))
        return 3
    report = write_reference(document, output)
    print(json.dumps({"status": "ok", **report, "caps": caps}, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay", nargs="?", type=Path,
                        help="retail .rpy to inspect (never produces a reference)")
    parser.add_argument("--measured", type=Path,
                        help="engine-measured stages JSON to build the reference from")
    parser.add_argument("--output", type=Path, default=None,
                        help=f"reference path (default for --measured: {DEFAULT_OUTPUT})")
    parser.add_argument("--measured-at", help="ISO-8601 measurement timestamp override")
    parser.add_argument("--engine-sha256",
                        help="sha256 of the exact harness binary that produced the measured numbers "
                             "(measured with sha256sum inside WSL)")
    parser.add_argument("--engine-sha256-source",
                        help="how --engine-sha256 was obtained (recorded beside the digest)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and print the document without writing it")
    parser.add_argument("--evidence-output", type=Path,
                        help="Write inspect-only evidence JSON (never benchmark reference)")
    args = parser.parse_args(argv)

    if args.measured is not None:
        output = args.output
        if output is None and not args.dry_run:
            output = DEFAULT_OUTPUT
        return _build(args.measured, output, measured_at=args.measured_at,
                      engine_sha256=args.engine_sha256,
                      engine_sha256_source=args.engine_sha256_source,
                      dry_run=args.dry_run)

    if args.replay is None:
        parser.error("either --measured EVIDENCE or a replay file is required")

    evidence = None
    try:
        evidence = inspect_human_replay(args.replay)
        if args.evidence_output:
            if args.evidence_output.name == "human_reference.json":
                parser.error("evidence cannot be written as human_reference.json")
            args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
            args.evidence_output.write_text(
                json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.output:
            build_human_reference(args.replay, args.output)
    except (ValueError, OSError, MissingHumanMeasurements) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "evidence": evidence},
                         ensure_ascii=False))
        return 3
    print(json.dumps(evidence, indent=2, ensure_ascii=False))
    return 3  # Inspection is useful but never a successfully built human reference.


if __name__ == "__main__":
    raise SystemExit(main())
