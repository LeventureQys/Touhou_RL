"""Round 1: prove same-frame pixel/state pairing with the TH08 frame fence.

Runs the research-only ``fence`` build of the reconstruction (an extra
publication hook after ``D3D8 Present``), navigates into practice stage 1, and
then captures frames two ways:

* ``pinned``   -- the game holds inside the fence while the reader does a GDI
  capture, a full state read (including the 6.3 MB bullet pool) and an optional
  10 MB enemy read, then acknowledges the ticket.  Every read is verified against
  the digests the *game* computed for that same ticket.
* ``legacy``   -- the same reads with no hold and no ticket pinning, i.e. what
  the existing pipeline does.  Reported so the difference is measured, not
  asserted.

    python tools/rl/frame_fence_round1.py --frames 120 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FENCE_EXE = ROOT / "third_party/th08/build_fence/th08.exe"
FENCE_ADDRESSES = ROOT / "third_party/th08/build_fence/frame-fence-addresses.json"
PLAY_TEMPLATE = ROOT / "third_party/th08/build/play"
FENCE_TEMPLATE = ROOT / "third_party/th08/build/play_fence"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def text_section_sha256(path: Path) -> str:
    """Same definition as tools/th08/Th08Pipeline.psm1 Get-Th08CodeHash."""
    import struct

    data = path.read_bytes()
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    count = struct.unpack_from("<H", data, e_lfanew + 6)[0]
    optional_size = struct.unpack_from("<H", data, e_lfanew + 20)[0]
    section_offset = e_lfanew + 24 + optional_size
    for index in range(count):
        offset = section_offset + index * 40
        name = data[offset : offset + 8].split(b"\x00")[0].decode("ascii", "replace")
        if name == ".text":
            raw_size, raw_pointer = struct.unpack_from("<II", data, offset + 16)
            return hashlib.sha256(data[raw_pointer : raw_pointer + raw_size]).hexdigest().upper()
    raise ValueError(f"no .text section in {path}")


def prepare_fence_template() -> Path:
    """A ``build/play``-shaped directory whose exe is the fence build."""
    if not FENCE_EXE.is_file():
        raise SystemExit(
            f"missing fence build {FENCE_EXE}; run: python third_party/th08/scripts/build.py --build-type fence"
        )
    if not FENCE_ADDRESSES.is_file():
        raise SystemExit(
            f"missing address table {FENCE_ADDRESSES}; run: python third_party/th08/scripts/fence-addresses.py "
            "third_party/th08/build_fence/th08.map third_party/th08/build_fence/frame-fence-addresses.json"
        )
    # A stale table silently reads the wrong globals (the variant moves every
    # global), so refuse to run when it does not belong to the current link.
    map_path = ROOT / "third_party/th08/build_fence/th08.map"
    table = json.loads(FENCE_ADDRESSES.read_text(encoding="utf-8"))
    if map_path.is_file() and table.get("mapSha256") != _sha256(map_path):
        raise SystemExit(
            "address table is stale: rerun python third_party/th08/scripts/fence-addresses.py "
            "third_party/th08/build_fence/th08.map third_party/th08/build_fence/frame-fence-addresses.json"
        )
    FENCE_TEMPLATE.mkdir(parents=True, exist_ok=True)
    for name in ("th08.dat", "thbgm.dat", "score.dat", "th08.cfg"):
        source = PLAY_TEMPLATE / name
        target = FENCE_TEMPLATE / name
        if not source.is_file():
            raise SystemExit(f"missing reference asset {source}")
        if target.exists() and target.stat().st_size == source.stat().st_size:
            continue
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    target_exe = FENCE_TEMPLATE / "th08.exe"
    if not target_exe.exists() or _sha256(target_exe) != _sha256(FENCE_EXE):
        shutil.copy2(FENCE_EXE, target_exe)
    return FENCE_TEMPLATE


def bootstrap_ack(fence, timeout: float = 5.0) -> int:
    """Ack one ordinary (non-held) ticket so the game starts honouring holds."""
    start = fence.ticket
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        current = fence.ticket
        if current != start and current:
            fence.ack(current)
            return current
        time.sleep(0.0005)
    raise TimeoutError("the fence never published a ticket: is the game running?")


def summarise(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 3),
        "p50": round(statistics.median(ordered), 3),
        "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 3),
        "max": round(ordered[-1], 3),
        "mean": round(sum(ordered) / len(ordered), 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("frame-fence-round1")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--legacy-frames", type=int, default=60)
    parser.add_argument("--run-dir", default=".tmp/frame-fence-round1/run")
    parser.add_argument(
        "--out-dir", default="Document/Update/Alpha/v3.7.4/evidence/frame-fence-round1"
    )
    parser.add_argument("--hold-max-us", type=int, default=90_000)
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--warmup-seconds", type=float, default=8.0)
    parser.add_argument(
        "--mode",
        choices=["demo", "practice"],
        default="demo",
        help="demo: wait for the title-screen attract demo (no input at all); "
        "practice: navigate into practice stage 1 with posted keys",
    )
    args = parser.parse_args(argv)

    template = prepare_fence_template()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # The variant image moved every global by 0x3000, so the reader must use the
    # fence build's own address table (this must happen before touhou_rl.spec is
    # imported anywhere).
    os.environ["TH08_ADDRESS_FILE"] = str(FENCE_ADDRESSES)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from touhou_rl import menu, spec
    from touhou_rl.framefence import (
        FenceFrameSampler,
        FenceStateDigest,
        FrameFence,
        legacy_sample,
    )
    from touhou_rl.session import GameSession, SessionOptions, prepare_run_dir

    if not spec.ADDRESS_SOURCE.get("addressFile"):
        raise SystemExit("TH08_ADDRESS_FILE was not honoured by touhou_rl.spec")

    run_dir = (ROOT / args.run_dir).resolve()
    prepare_run_dir(
        run_dir,
        template,
        windowed=True,
        default_difficulty=spec.LUNATIC,
        use_postmessage_input=True,
        force=True,
    )

    report: dict[str, object] = {
        "schema": "th08-frame-fence-round1/1",
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "build": {
            "exe": str(FENCE_EXE.relative_to(ROOT)),
            "exeSha256": _sha256(FENCE_EXE),
            "textSha256": text_section_sha256(FENCE_EXE),
            "addressFile": str(FENCE_ADDRESSES.relative_to(ROOT)),
            "addressFileSha256": _sha256(FENCE_ADDRESSES),
            "imageBase": f"0x{spec.IMAGE_BASE:08X}",
        },
        "canonicalBaseline": {
            "exeSha256": "CDD23BCA0C724941866EFDA5DA280BE67EB6CDD935F83548045C2A8E037F1DDF",
            "textSha256": "7B5B3C8AB5023BFA7960DA309338A01982F651DC6B1E7734BEE92437202ED6DD",
        },
        "requestedFrames": args.frames,
        "pinned": {"samples": [], "aggregate": {}},
        "legacy": {"samples": [], "aggregate": {}},
    }

    session = GameSession(
        SessionOptions(run_dir=run_dir, windowed=True, boot_timeout_s=45.0)
    )
    fence = None
    try:
        session.start()
        report["bootSeconds"] = round(session.boot_seconds, 3)
        report["session"] = {
            "pid": session.pid,
            "client": list(session.client_size),
            "inputBackend": type(session.keyboard).__name__,
        }
        fence = FrameFence.open(timeout=20.0)
        header = fence.header()
        report["fenceHeader"] = header
        if header["pid"] != session.pid:
            raise SystemExit(
                f"fence belongs to pid {header['pid']}, session is {session.pid}: stale mapping"
            )
        report["fenceControl"] = fence.control()

        # Publishing only, no holds: neither the attract demo nor the menu
        # navigation may be perturbed while we settle the session.
        fence.configure(enable=True, hold_every=0, hold_max_us=args.hold_max_us)

        if args.mode == "practice":
            menu.enter_game(session, difficulty=spec.LUNATIC, practice_stage=0)
            session.log_line("entered practice stage 1")
        else:
            # TH08 plays an attract demo on the title screen after a short idle
            # period.  The demo runs the real stage loop (bullets, enemies,
            # camera), so the fence can be measured with no input path at all.
            deadline = time.perf_counter() + 30.0
            while time.perf_counter() < deadline:
                state = session.read_state(with_bullets=False)
                if state.supervisor_state == spec.STATE_GAME_MANAGER:
                    break
                time.sleep(0.05)
            else:
                raise SystemExit("title-screen attract demo never started")
            session.log_line("attract demo running")
        state = session.read_state(with_bullets=False)
        report["stage"] = {
            "stage": int(state.stage),
            "isPractice": bool(state.is_practice),
            "isDemo": bool(state.is_demo),
            "isReplay": bool(state.is_replay),
            "difficulty": int(state.difficulty_hint),
            "gameplayFrame": int(state.gameplay_frame),
            "supervisorState": int(state.supervisor_state),
        }

        # Let the stage build up enemies and bullets: the digests are far
        # stronger evidence when the pool actually moves.
        deadline = time.perf_counter() + args.warmup_seconds
        best_bullets = 0
        while time.perf_counter() < deadline:
            current = session.read_state(with_bullets=True)
            best_bullets = max(best_bullets, int(current.active_bullets))
            if best_bullets >= 40:
                break
            time.sleep(0.05)
        state = session.read_state(with_bullets=True)
        report["warmup"] = {
            "gameplayFrame": int(state.gameplay_frame),
            "activeBullets": int(state.active_bullets),
            "maxActiveBullets": best_bullets,
            "player": [round(float(state.player_x), 2), round(float(state.player_y), 2)],
        }

        digest = FenceStateDigest(session.reader)
        ack = bootstrap_ack(fence)
        report["bootstrapAck"] = ack

        # ---- pinned round -------------------------------------------------
        fence.configure(
            enable=True,
            pixels=True,
            hold_every=1,
            hold_min_us=0,
            hold_max_us=args.hold_max_us,
        )
        sampler = FenceFrameSampler(session, fence, with_pixels=True, with_enemies=True)
        pinned = report["pinned"]
        started = time.perf_counter()
        logic_first = None
        logic_last = None
        pixel_dumps: list[tuple[int, object, object]] = []
        pixel_matches: list[tuple[int, object, object]] = []
        for _ in range(args.frames):
            sample = sampler.next(timeout=2.0, attempts=12)
            if sample is None:
                break
            telemetry = sample.telemetry
            assert telemetry is not None and sample.state is not None
            if logic_first is None:
                logic_first = telemetry.logic_frame
            logic_last = telemetry.logic_frame
            record = {
                "ticket": sample.ticket,
                "logicFrame": telemetry.logic_frame,
                "stageFrames": telemetry.stage_frames,
                "player": [round(telemetry.player_x, 4), round(telemetry.player_y, 4)],
                "activeBullets": telemetry.active_bullets,
                "holdUs": telemetry.hold_last_us,
                "holdBudgetSkips": telemetry.hold_budget_skips,
                "digestTicket": telemetry.digest_ticket,
                "pixelTicket": telemetry.pixel_ticket,
                "checks": sample.checks,
                "ok": sample.ok,
                "timingsUs": {k: round(v, 1) for k, v in sample.timings_us.items()},
                "gdiSha256": hashlib.sha256(sample.pixels.tobytes()).hexdigest().upper()
                if sample.pixels is not None
                else None,
                "fenceSha256": hashlib.sha256(sample.fence_pixels.tobytes()).hexdigest().upper()
                if sample.fence_pixels is not None
                else None,
            }
            pinned["samples"].append(record)
            if sample.pixels is not None and sample.fence_pixels is not None:
                if not sample.checks.get("pixels_match_backbuffer", True) and len(pixel_dumps) < 3:
                    pixel_dumps.append((sample.ticket, sample.pixels, sample.fence_pixels))
                elif len(pixel_matches) < 2:
                    pixel_matches.append((sample.ticket, sample.pixels, sample.fence_pixels))
        pinned_elapsed = time.perf_counter() - started
        fence.configure(enable=True, pixels=False, hold_every=0)
        session.keyboard.release_all()

        # ---- legacy round -------------------------------------------------
        if not args.skip_legacy:
            fence.configure(enable=True, hold_every=0, digest_always=True)
            legacy = report["legacy"]
            started = time.perf_counter()
            for _ in range(args.legacy_frames):
                legacy["samples"].append(legacy_sample(session, fence, digest))
                time.sleep(0.004)
            legacy_elapsed = time.perf_counter() - started
            fence.configure(enable=True, hold_every=0, digest_always=False)
            legacy["wallSeconds"] = round(legacy_elapsed, 3)
        else:
            legacy_elapsed = 0.0

        final_telemetry = fence.telemetry()
        report["fenceFinal"] = {
            "ticket": final_telemetry.ticket,
            "holdCount": final_telemetry.hold_count,
            "holdMaxUs": final_telemetry.hold_max_us,
            "holdBudgetSkips": final_telemetry.hold_budget_skips,
        }

        # ---- aggregates ---------------------------------------------------
        records = pinned["samples"]
        if records:
            check_names = sorted({name for row in records for name in row["checks"]})
            pinned["aggregate"] = {
                "frames": len(records),
                "wallSeconds": round(pinned_elapsed, 3),
                "presentRateHz": round(len(records) / max(1e-9, pinned_elapsed), 2),
                "logicFramesPerPresent": round(
                    (logic_last - logic_first) / max(1, len(records) - 1), 4
                )
                if logic_first is not None and logic_last is not None
                else None,
                "ticketDeltaAllOne": all(
                    b["ticket"] - a["ticket"] == 1 for a, b in zip(records, records[1:])
                ),
                "logicFrameDeltaAllOne": all(
                    b["logicFrame"] - a["logicFrame"] == 1 for a, b in zip(records, records[1:])
                ),
                "checkPassRate": {
                    name: round(sum(1 for row in records if row["checks"].get(name)) / len(records), 4)
                    for name in check_names
                },
                "allChecksPass": all(row["ok"] for row in records),
                "pixelDumpsPresent": round(
                    sum(1 for row in records if row["fenceSha256"] is not None) / len(records), 4
                ),
                "gdiEqualsFencePixels": round(
                    sum(
                        1
                        for row in records
                        if row["fenceSha256"] is not None and row["gdiSha256"] == row["fenceSha256"]
                    )
                    / max(1, sum(1 for row in records if row["fenceSha256"] is not None)),
                    4,
                ),
                "holdUs": summarise([float(row["holdUs"]) for row in records]),
                "pipelineUs": summarise([float(row["timingsUs"]["total"]) for row in records]),
                "blitUs": summarise([float(row["timingsUs"]["blit"]) for row in records]),
                "stateUs": summarise([float(row["timingsUs"]["state"]) for row in records]),
                "digestUs": summarise([float(row["timingsUs"]["digest"]) for row in records]),
                "retries": sampler.retries,
                "distinctGdiFrames": len({row["gdiSha256"] for row in records if row["gdiSha256"]}),
                "distinctFenceFrames": len(
                    {row["fenceSha256"] for row in records if row["fenceSha256"]}
                ),
            }
        legacy_records = report["legacy"]["samples"]
        if legacy_records:
            report["legacy"]["aggregate"] = {
                "frames": len(legacy_records),
                "wallSeconds": round(legacy_elapsed, 3),
                "ticketChangedDuringReadRate": round(
                    sum(1 for row in legacy_records if not row["ticket_unchanged"])
                    / len(legacy_records),
                    4,
                ),
                "bulletDigestMatchRate": round(
                    sum(1 for row in legacy_records if row["bullet_digest_match"])
                    / len(legacy_records),
                    4,
                ),
                "enemyDigestMatchRate": round(
                    sum(1 for row in legacy_records if row["enemy_digest_match"])
                    / len(legacy_records),
                    4,
                ),
                "playerMatchRate": round(
                    sum(1 for row in legacy_records if row["player_match"]) / len(legacy_records),
                    4,
                ),
                "frameMatchRate": round(
                    sum(1 for row in legacy_records if row["frame_match"]) / len(legacy_records),
                    4,
                ),
            }

        # Save the first mismatching pixel pair (if any) for the report, and one
        # matching pair as positive evidence that the pixels really are the same
        # bytes (the diff is then all zeros by construction).
        if pixel_matches:
            try:
                from PIL import Image

                ticket, gdi, fence_pixels = pixel_matches[0]
                Image.fromarray(gdi).save(out_dir / f"frame-{ticket}-gdi.png")
                Image.fromarray(fence_pixels).save(out_dir / f"frame-{ticket}-backbuffer.png")
                report["pixelMatchDump"] = ticket
            except Exception as exc:  # pragma: no cover - diagnostics only
                report["pixelMatchDumpError"] = repr(exc)
        if pixel_dumps:
            try:
                from PIL import Image

                ticket, gdi, fence_pixels = pixel_dumps[0]
                Image.fromarray(gdi).save(out_dir / f"pixel-mismatch-{ticket}-gdi.png")
                Image.fromarray(fence_pixels).save(out_dir / f"pixel-mismatch-{ticket}-fence.png")
                report["pixelMismatchDump"] = ticket
            except Exception as exc:  # pragma: no cover - diagnostics only
                report["pixelMismatchDumpError"] = repr(exc)
        elif records and pinned["aggregate"].get("gdiEqualsFencePixels") == 1.0:
            report["pixelEvidence"] = (
                "every pinned GDI capture is byte-identical to the back buffer the game "
                "published for that ticket"
            )

        report["finishedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (out_dir / "frames.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary = {
            "schema": report["schema"],
            "build": report["build"],
            "fenceHeader": report["fenceHeader"],
            "session": report["session"],
            "stage": report["stage"],
            "warmup": report["warmup"],
            "pinned": report["pinned"]["aggregate"],
            "legacy": report["legacy"]["aggregate"],
            "fenceFinal": report["fenceFinal"],
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        if fence is not None:
            try:
                fence.disable()
            except Exception:
                pass
            fence.close()
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
