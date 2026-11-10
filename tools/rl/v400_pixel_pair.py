"""Strict WP8 frame pairing; synthetic data is never G0 evidence.

Manifest: ``{"source":"windows-fence"|"linux-harness", "replay_sha256":hex64,
"frames":[{"logicFrame":int,"image":"relative.png","state_sha256":hex64,
"state_evidence":{...}}]}``.  ``state_evidence`` is optional and is passed
through verbatim; the two sides do not share a state digest algorithm
(``windows-fence`` publishes ``bulletDigest``/``enemyDigest``, ``linux-harness``
publishes a 25-field FNV-1a), so a shared ``state_sha256`` only means "the same
side-specific evidence record".

Two pairing modes:

``strict`` (default)
    The default and the only mode that may claim state equality: both sides must
    report the same ``state_sha256`` for every paired ``logicFrame``.

``same-logic-frame``
    Explicit opt-in for the case where the two sides legitimately carry
    different state digests.  It still enforces the same replay hash and an
    identical ``logicFrame`` set, keeps both sides' evidence in the report, and
    writes an explicit warning that the pixel difference cannot be attributed to
    rendering alone.  It never claims the states agree.

``common-projection``
    For the retail original (``source="windows-retail"``) against the Linux
    harness.  The two sides publish different digests *and* different evidence
    shapes, but both publish the seven fields ``score, lives, bombs, deaths,
    power, graze, stage``.  This mode re-projects both sides to that common
    projection, compares the exact 32-bit patterns frame by frame, and only when
    every paired frame agrees does it report
    ``state_consistency="asserted-on-projection"``.  One unequal field is
    fail-closed: no report is written and the exit code is 3, naming the first
    unequal frame and fields.  The projection is not the whole game state
    (RNG, bullets, enemies are outside it), and the report says so.

``--global-phase`` (EXP-P0-065)
    Adds a ``global_phase`` block: one *single* offset ``s`` per stream, scored by
    the sum of ``MAD(retail[n], linux[n+s])`` over the playfield for
    ``s in {-1, 0, +1}``.  The winner and the runner-up are both reported, so a
    stream-level constant can be told apart from a per-frame coin flip; the
    per-frame best-of-``s`` figure is kept only as an oracle bound and is labelled
    as one.  It changes no verdict and writes no new claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HEX = re.compile(r"[a-fA-F0-9]{64}\Z")
REGIONS = ("playfield", "hud")
# The HUD's frame-rate readout.  It was located by measurement, not assumption:
# across every paired frame of the first WP8 run it is the only rectangle whose
# channels differ by more than 16/255, because the Windows fence build prints the
# real (held) frame rate while the Linux harness pins the HUD at 60.00fps.  It is
# a HUD *value* difference, so the required playfield/HUD numbers stay untouched
# and this box only feeds the auxiliary "excluding the fps readout" breakdown.
FPS_READOUT_BOX = {"y": [463, 480], "x": [512, 581]}
FPS_READOUT_NOTE = (
    "auxiliary only: the fps readout is a HUD value difference (windows prints the "
    "real held frame rate, linux pins 60.00fps), not a renderer difference"
)
PAIRING_MODES = ("strict", "same-logic-frame", "common-projection")
STATE_WARNING = (
    "Pixel differences cannot be fully attributed to rendering: the two sides "
    "publish different state evidence (windows: bulletDigest/enemyDigest; linux: "
    "25-field FNV-1a), so state equality across the paired frames is NOT "
    "established by this report."
)
STATE_CLAIM = "not-asserted"
STATE_PROJECTION_CLAIM = "asserted-on-projection"

#: The seven fields both sides publish that can be compared without sharing a
#: digest algorithm: three integers and four float32 values read from the same
#: ``ZunGlobals`` the engine uses.
PROJECTION_FIELDS = ("score", "lives", "bombs", "deaths", "power", "graze", "stage")
PROJECTION_FLOAT_FIELDS = frozenset({"lives", "bombs", "deaths", "power"})
#: ``rlharness.cpp TrackedFields`` order: f4 score, f5 graze, f6 deaths, f7 lives,
#: f8 bombs, f9 power, f1 stage.  f6..f9 are published as float bit patterns.
LINUX_PROJECTION_INDEX = {
    "score": 4,
    "graze": 5,
    "deaths": 6,
    "lives": 7,
    "bombs": 8,
    "power": 9,
    "stage": 1,
}
#: Which evidence source each pairing mode accepts.  ``strict`` and
#: ``same-logic-frame`` keep requiring the fence build (unchanged behaviour); the
#: retail original is only admissible where its projection is actually checked.
WINDOWS_SOURCES = {
    "strict": ("windows-fence",),
    "same-logic-frame": ("windows-fence",),
    "common-projection": ("windows-retail",),
}
PROJECTION_NOTE = (
    "exact equality of the seven-field common projection (score, lives, bombs, "
    "deaths, power, graze, stage) on every paired frame; float fields are "
    "compared as 32-bit patterns, not with a tolerance.  RNG, bullets, enemies "
    "and every other piece of state are OUTSIDE the projection."
)


class EvidenceError(ValueError):
    pass


def load(path: Path, source):
    """Load a manifest; ``source`` is the accepted source string or a tuple of them."""
    allowed = (source,) if isinstance(source, str) else tuple(source)
    data = json.loads(path.read_text(encoding="utf-8"))
    found_source = data.get("source") if isinstance(data, dict) else None
    if (
        not isinstance(data, dict)
        or found_source not in allowed
        or not HEX.fullmatch(str(data.get("replay_sha256", "")))
    ):
        raise EvidenceError(
            f"{path}: source {found_source!r} / replay SHA256 invalid; expected one of {list(allowed)}"
        )
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise EvidenceError(f"{path}: nonempty frames required")
    found = {}
    for row in frames:
        if not isinstance(row, dict) or type(row.get("logicFrame")) is not int or row["logicFrame"] < 0 or not HEX.fullmatch(str(row.get("state_sha256", ""))):
            raise EvidenceError(f"{path}: invalid logicFrame/state provenance")
        n = row["logicFrame"]
        if n in found:
            raise EvidenceError(f"{path}: duplicate frame {n}")
        image = (path.resolve().parent / str(row.get("image", ""))).resolve()
        if not image.is_relative_to(path.resolve().parent) or image.suffix.lower() != ".png" or not image.is_file():
            raise EvidenceError(f"{path}: frame {n} needs local lossless PNG")
        evidence = row.get("state_evidence")
        if evidence is not None and not isinstance(evidence, dict):
            raise EvidenceError(f"{path}: frame {n} state_evidence must be an object")
        found[n] = row, image
    return data, found


def project_retail(evidence: dict, frame: int) -> dict[str, int]:
    """Project a ``windows-retail`` frame onto the seven common fields.

    The retail capture publishes every field as an exact 32-bit pattern in
    ``field_bits`` (``score``/``graze``/``stage`` are integers widened to 32 bits,
    the other four are the game's float32 values), so this is a pure relabelling
    with no rounding step.
    """
    if not isinstance(evidence, dict):
        raise EvidenceError(f"frame {frame}: windows-retail frame has no state_evidence")
    bits = evidence.get("field_bits")
    if not isinstance(bits, dict):
        raise EvidenceError(f"frame {frame}: windows-retail evidence lacks field_bits")
    projection: dict[str, int] = {}
    for name in PROJECTION_FIELDS:
        raw = bits.get(name)
        if raw is None:
            raise EvidenceError(f"frame {frame}: retail projection lacks {name}")
        try:
            projection[name] = int(str(raw), 16)
        except ValueError as exc:
            raise EvidenceError(f"frame {frame}: retail projection {name}={raw!r} is not 32-bit hex") from exc
    return projection


def project_linux(evidence: dict, frame: int) -> dict[str, int]:
    """Project a ``linux-harness`` frame onto the same seven fields.

    ``rlharness.cpp`` publishes ``f6..f9`` as ``%08x`` float bit patterns and
    ``f1/f4/f5`` as decimals, so the parsed integers already are the exact
    patterns this projection wants.
    """
    if not isinstance(evidence, dict):
        raise EvidenceError(f"frame {frame}: linux-harness frame has no state_evidence")
    fields = evidence.get("fields")
    if not isinstance(fields, dict):
        raise EvidenceError(f"frame {frame}: linux-harness evidence lacks fields")
    projection: dict[str, int] = {}
    for name, index in LINUX_PROJECTION_INDEX.items():
        key = f"f{index}"
        if key not in fields:
            raise EvidenceError(f"frame {frame}: linux projection lacks {key} ({name})")
        try:
            projection[name] = int(fields[key])
        except (TypeError, ValueError) as exc:
            raise EvidenceError(f"frame {frame}: linux projection {key}={fields[key]!r} is not an integer") from exc
    return projection


def bits_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(bits) & 0xFFFFFFFF))[0]


def projection_display(projection: dict[str, int]) -> dict:
    """Human-readable view of a projection; equality is still decided on the bits."""
    return {
        name: bits_float(value) if name in PROJECTION_FLOAT_FIELDS else int(value)
        for name, value in projection.items()
    }


def region_stats(delta: np.ndarray) -> dict[str, dict[str, float]]:
    regions = {}
    for name, pixels in (
        ("playfield", delta[:, 32:416]),
        ("hud", np.concatenate((delta[:, :32], delta[:, 416:]), axis=1)),
    ):
        regions[name] = {
            "mean_absolute_difference": float(pixels.mean()),
            "different_pixel_ratio": float(np.any(pixels, axis=2).mean()),
            "max_channel_difference": int(pixels.max()),
        }
    return regions


def aux_stats(delta: np.ndarray) -> dict[str, dict[str, float]]:
    """Same metrics with the fps readout excluded, plus the outside-box maximum."""
    height, width = delta.shape[:2]
    outside = np.ones((height, width), dtype=bool)
    outside[FPS_READOUT_BOX["y"][0]:FPS_READOUT_BOX["y"][1],
            FPS_READOUT_BOX["x"][0]:FPS_READOUT_BOX["x"][1]] = False
    hud = outside.copy()
    hud[:, 32:416] = False
    stats = {}
    for name, mask in (
        ("whole_frame_excluding_fps_readout", outside),
        ("hud_excluding_fps_readout", hud),
    ):
        pixels = delta[mask]
        stats[name] = {
            "mean_absolute_difference": float(pixels.mean()),
            "different_pixel_ratio": float(np.count_nonzero(pixels) / pixels.size),
            "max_channel_difference": int(pixels.max()),
        }
    return stats


def aggregate(rows: list[dict]) -> dict[str, dict[str, float]]:
    summary = {}
    for name in REGIONS:
        means = [row["regions"][name]["mean_absolute_difference"] for row in rows]
        ratios = [row["regions"][name]["different_pixel_ratio"] for row in rows]
        summary[name] = {
            "mean_of_frame_mean_absolute_difference": float(np.mean(means)),
            "min_frame_mean_absolute_difference": float(np.min(means)),
            "max_frame_mean_absolute_difference": float(np.max(means)),
            "mean_different_pixel_ratio": float(np.mean(ratios)),
            "max_frame_different_pixel_ratio": float(np.max(ratios)),
            "max_channel_difference": int(
                max(row["regions"][name]["max_channel_difference"] for row in rows)
            ),
        }
    for name in rows[0]["aux_regions"]:
        means = [row["aux_regions"][name]["mean_absolute_difference"] for row in rows]
        ratios = [row["aux_regions"][name]["different_pixel_ratio"] for row in rows]
        summary[name] = {
            "mean_of_frame_mean_absolute_difference": float(np.mean(means)),
            "min_frame_mean_absolute_difference": float(np.min(means)),
            "max_frame_mean_absolute_difference": float(np.max(means)),
            "mean_different_pixel_ratio": float(np.mean(ratios)),
            "max_frame_different_pixel_ratio": float(np.max(ratios)),
            "max_channel_difference": int(
                max(row["aux_regions"][name]["max_channel_difference"] for row in rows)
            ),
        }
    return summary


# ---------------------------------------------------------------------------
# global presentation phase (EXP-P0-065)
# ---------------------------------------------------------------------------
#: Candidate stream-level offsets.  ``s`` in this module always means
#: ``MAD(retail[n], linux[n + s])``, i.e. ``s = -1`` reads "the retail grab under
#: logicFrame n shows what the Linux harness renders at logicFrame n-1".
GLOBAL_PHASE_SHIFTS = (-1, 0, 1)
#: A frame whose playfield MAD is at or below this is "the same picture" for the
#: same-sign comparison (EXP-P0-063/065 use the same 0.2).  Recorded as a *count*
#: only: this tool still has no G0.4 gate and its exit code does not depend on it.
PLAYFIELD_CLEAN_THRESHOLD = 0.2
GLOBAL_PHASE_NOTE = (
    "one constant offset per stream, chosen by the sum of the playfield MAD over "
    "the whole stream; the per-frame best-of-{s} numbers are an oracle bound and "
    "must NOT be used as the stream verdict, because picking a different s per "
    "frame is frame-by-frame cherry-picking, not calibration"
)


def load_rgb(path: Path) -> np.ndarray:
    """Lossless PNG as an HxWx3 uint8 array (alpha dropped: GDI has none)."""
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != (640, 480):
            raise EvidenceError(f"{path}: expected PNG 640x480")
        return np.asarray(image.convert("RGB")).copy()


def playfield_mad(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference over the playfield columns ``x in [32, 416)``."""
    return float(
        np.abs(
            a[:, 32:416].astype(np.int16) - b[:, 32:416].astype(np.int16)
        ).mean()
    )


def phase_scan(windows: dict, linux: dict, shifts=GLOBAL_PHASE_SHIFTS) -> dict:
    """Score every *global* offset ``s`` over a whole paired stream.

    ``windows``/``linux`` are the ``{logicFrame: (row, path)}`` maps returned by
    :func:`load`.  For each ``s`` the module computes the sum, mean, min and max of
    ``MAD(retail[n], linux[n+s])`` over every frame whose partner exists, plus the
    same aggregate restricted to the frames that are comparable for *all* shifts
    (so the winner cannot be an artefact of one edge frame disappearing).  It also
    records the per-frame best shift as an oracle bound, clearly labelled as such.
    """
    ordered = sorted(windows)
    shifts = tuple(int(s) for s in shifts)
    cache: dict[int, np.ndarray] = {}

    def linux_rgb(frame: int) -> np.ndarray:
        if frame not in cache:
            cache[frame] = load_rgb(linux[frame][1])
            for stale in [key for key in cache if key < frame - 1]:
                del cache[stale]
        return cache[frame]

    per_frame: dict[int, dict[int, float]] = {}
    for n in ordered:
        retail = load_rgb(windows[n][1])
        for s in shifts:
            partner = n + s
            if partner not in linux:
                continue
            per_frame.setdefault(n, {})[s] = playfield_mad(retail, linux_rgb(partner))

    per_shift: dict[str, dict] = {}
    for s in shifts:
        values = [row[s] for row in per_frame.values() if s in row]
        clean = [value for value in values if value <= PLAYFIELD_CLEAN_THRESHOLD]
        per_shift[str(s)] = {
            "frames": len(values),
            "missing_pairs": len(ordered) - len(values),
            "playfield_mad_sum": float(np.sum(values)) if values else None,
            "playfield_mad_mean": float(np.mean(values)) if values else None,
            "playfield_mad_min": float(np.min(values)) if values else None,
            "playfield_mad_max": float(np.max(values)) if values else None,
            "playfield_mad_clean_frames": len(clean),
            "playfield_mad_clean_fraction": round(len(clean) / len(values), 4) if values else None,
        }

    comparable = [n for n in ordered if all(s in per_frame.get(n, {}) for s in shifts)]
    on_comparable = {
        str(s): {
            "frames": len(comparable),
            "playfield_mad_sum": float(np.sum([per_frame[n][s] for n in comparable])) if comparable else None,
            "playfield_mad_mean": float(np.mean([per_frame[n][s] for n in comparable])) if comparable else None,
            "playfield_mad_clean_frames": sum(
                1 for n in comparable if per_frame[n][s] <= PLAYFIELD_CLEAN_THRESHOLD
            ),
            "playfield_mad_clean_fraction": (
                round(
                    sum(1 for n in comparable if per_frame[n][s] <= PLAYFIELD_CLEAN_THRESHOLD)
                    / len(comparable),
                    4,
                )
                if comparable
                else None
            ),
        }
        for s in shifts
    }
    # Rank by the whole-stream sum; ties fall back to s=0 (the unshifted reading)
    # because a set of offsets that score identically carries no phase information.
    ranked = sorted(
        shifts,
        key=lambda s: (
            on_comparable[str(s)]["playfield_mad_sum"] is None,
            on_comparable[str(s)]["playfield_mad_sum"] or 0.0,
            0 if s == 0 else abs(s),
            s,
        ),
    )
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    best_sum = on_comparable[str(best)]["playfield_mad_sum"]
    runner_sum = None if runner_up is None else on_comparable[str(runner_up)]["playfield_mad_sum"]

    per_frame_best: dict[int, int] = {}
    counts = {str(s): 0 for s in shifts}
    oracle = []
    for n in comparable:
        row = per_frame[n]
        winner = min(shifts, key=lambda s: row[s])
        per_frame_best[n] = winner
        counts[str(winner)] += 1
        oracle.append(row[winner])
    return {
        "shifts": list(shifts),
        "definition": "s means MAD(retail[n], linux[n+s]) on the playfield; s=-1 "
        "means the retail grab filed under logicFrame n depicts Linux frame n-1",
        "per_shift": per_shift,
        "comparable_frames": len(comparable),
        "per_shift_on_comparable_frames": on_comparable,
        "best_shift": best,
        "best_shift_sum": best_sum,
        "best_shift_mean": None if best_sum is None else best_sum / max(1, len(comparable)),
        "runner_up_shift": runner_up,
        "runner_up_sum": runner_sum,
        "runner_up_mean": None if runner_sum is None else runner_sum / max(1, len(comparable)),
        "gap_sum": None if (best_sum is None or runner_sum is None) else runner_sum - best_sum,
        "per_frame_best_shift_counts": counts,
        "per_frame_best_shift": {str(n): per_frame_best[n] for n in comparable},
        "per_frame_playfield_mad": {
            str(n): {str(s): per_frame[n][s] for s in shifts} for n in comparable
        },
        "per_frame_best_oracle": {
            "playfield_mad_sum": float(np.sum(oracle)) if oracle else None,
            "playfield_mad_mean": float(np.mean(oracle)) if oracle else None,
            "playfield_mad_max": float(np.max(oracle)) if oracle else None,
            "frames": len(oracle),
            "playfield_mad_clean_frames": sum(
                1 for value in oracle if value <= PLAYFIELD_CLEAN_THRESHOLD
            ),
            "playfield_mad_clean_fraction": (
                round(
                    sum(1 for value in oracle if value <= PLAYFIELD_CLEAN_THRESHOLD) / len(oracle),
                    4,
                )
                if oracle
                else None
            ),
        },
        "clean": {
            "threshold": PLAYFIELD_CLEAN_THRESHOLD,
            "definition": "a frame is clean when MAD(retail[n], linux[n+s]) <= threshold "
            "for that single stream-level s; the per-frame best-of-shifts clean count is "
            "an oracle bound and is NOT a verdict",
            "per_shift_on_comparable_frames": {
                str(s): {
                    "clean_frames": on_comparable[str(s)]["playfield_mad_clean_frames"],
                    "frames": on_comparable[str(s)]["frames"],
                    "clean_fraction": on_comparable[str(s)]["playfield_mad_clean_fraction"],
                }
                for s in shifts
            },
        },
        "note": GLOBAL_PHASE_NOTE,
    }


def compare(
    windows: Path,
    linux: Path,
    frames: int = 200,
    *,
    state_pairing: str = "strict",
    global_phase: bool = False,
):
    if state_pairing not in PAIRING_MODES:
        raise EvidenceError(f"unknown state pairing {state_pairing!r}")
    wd, w = load(windows, WINDOWS_SOURCES[state_pairing])
    ld, l = load(linux, "linux-harness")
    if wd["replay_sha256"].lower() != ld["replay_sha256"].lower():
        raise EvidenceError("replay hash differs")
    if len(w) != frames or set(w) != set(l):
        raise EvidenceError(f"require {frames} exact logicFrame pairs: windows={len(w)} linux={len(l)} missing_windows={sorted(set(l)-set(w))[:10]} missing_linux={sorted(set(w)-set(l))[:10]}")
    rows = []
    projection_mismatches: list[dict] = []
    projection_bits: dict[str, list[int]] = {name: [] for name in PROJECTION_FIELDS}
    for n in sorted(w):
        wr, wp = w[n]
        lr, lp = l[n]
        same_state = wr["state_sha256"].lower() == lr["state_sha256"].lower()
        if state_pairing == "strict" and not same_state:
            raise EvidenceError(f"frame {n}: state mismatch; cannot attribute pixel difference")
        with Image.open(wp) as wi, Image.open(lp) as li:
            if wi.format != "PNG" or li.format != "PNG" or wi.mode != "RGBA" or li.mode != "RGBA" or wi.size != (640, 480) or li.size != wi.size:
                raise EvidenceError(f"frame {n}: expected PNG RGBA 640x480")
            a, b = np.asarray(wi).copy(), np.asarray(li).copy()
        delta = np.abs(a[:, :, :3].astype(np.int16) - b[:, :, :3].astype(np.int16))
        row = {
            "logicFrame": n,
            "windows_png_sha256": hashlib.sha256(wp.read_bytes()).hexdigest(),
            "linux_png_sha256": hashlib.sha256(lp.read_bytes()).hexdigest(),
            "windows_image": str(wp),
            "linux_image": str(lp),
            "regions": region_stats(delta),
            "aux_regions": aux_stats(delta),
            "windows_state_sha256": wr["state_sha256"].lower(),
            "linux_state_sha256": lr["state_sha256"].lower(),
            "state_sha256_equal": same_state,
            "windows_state_evidence": wr.get("state_evidence"),
            "linux_state_evidence": lr.get("state_evidence"),
        }
        if state_pairing == "common-projection":
            wproj = project_retail(wr.get("state_evidence"), n)
            lproj = project_linux(lr.get("state_evidence"), n)
            for name in PROJECTION_FIELDS:
                projection_bits[name].append(wproj[name])
            failed = [name for name in PROJECTION_FIELDS if wproj[name] != lproj[name]]
            row["state_projection"] = {
                "fields": list(PROJECTION_FIELDS),
                "windows": projection_display(wproj),
                "linux": projection_display(lproj),
                "equal": not failed,
                "mismatched_fields": failed,
            }
            if failed:
                projection_mismatches.append(
                    {
                        "logicFrame": n,
                        "mismatched_fields": failed,
                        "windows": projection_display(wproj),
                        "linux": projection_display(lproj),
                    }
                )
        rows.append(row)
    if projection_mismatches:
        first = projection_mismatches[0]
        detail = "; ".join(
            f"{name}: retail={first['windows'][name]!r} linux={first['linux'][name]!r}"
            for name in first["mismatched_fields"]
        )
        raise EvidenceError(
            f"frame {first['logicFrame']}: common projection mismatch on "
            f"{first['mismatched_fields']} ({len(projection_mismatches)}/{len(rows)} frames differ); {detail}"
        )
    report = {
        "schema": "v400-wp8-paired-pixel-audit/2",
        "status": "measured-not-G0-pass",
        "state_pairing": state_pairing,
        "state_consistency": {
            "strict": "enforced-equal",
            "same-logic-frame": STATE_CLAIM,
            "common-projection": STATE_PROJECTION_CLAIM,
        }[state_pairing],
        "replay_sha256": wd["replay_sha256"].lower(),
        "pairs": len(rows),
        "region_note": "RGB channels; playfield x=[32,416); HUD remaining columns; FPS not masked",
        "fps_readout_box": {**FPS_READOUT_BOX, "note": FPS_READOUT_NOTE},
        "aggregate": aggregate(rows),
        "worst_10": sorted(rows, key=lambda r: max(r["regions"][n]["mean_absolute_difference"] for n in REGIONS), reverse=True)[:10],
        "frames": rows,
    }
    if global_phase:
        report["global_phase"] = phase_scan(w, l)
        report["global_phase"]["retail_manifest_present_lag_frames"] = (
            wd.get("capture", {}) or {}
        ).get("presentLagFrames", 0)
        report["global_phase"]["windows_manifest_capture"] = {
            key: (wd.get("capture", {}) or {}).get(key)
            for key in ("presentLagFrames", "presentLagNote", "sameFrameProof", "frameFence")
        }
    if state_pairing == "common-projection":
        field_variation = {}
        for name in PROJECTION_FIELDS:
            patterns = projection_bits[name]
            display = [bits_float(value) if name in PROJECTION_FLOAT_FIELDS else int(value) for value in patterns]
            field_variation[name] = {
                "distinct_patterns": len(set(patterns)),
                "distinct_values": sorted(set(display)),
                "changes": sum(1 for a, b in zip(display, display[1:]) if a != b),
                "first": display[0],
                "last": display[-1],
            }
        report["state_projection"] = {
            "fields": list(PROJECTION_FIELDS),
            "float_fields": sorted(PROJECTION_FLOAT_FIELDS),
            "frames_equal": len(rows),
            "frames_total": len(rows),
            "field_equality": {
                name: {"equal": len(rows), "total": len(rows)} for name in PROJECTION_FIELDS
            },
            "field_variation": field_variation,
            "note": PROJECTION_NOTE,
        }
        report["warnings"] = [
            "state equality is asserted for the seven-field projection only; it is "
            "NOT a proof that the two implementations hold the same full game state "
            "(RNG, bullets, enemies and every field outside the projection are unchecked).",
            "the retail image has no frame fence: 'same frame' means the "
            "gameplayFrameCounter read before and after the BitBlt was identical, not "
            "that the logic frame was held still while the pixels were copied.",
            "the two sides' state_sha256 values are not comparable and are kept for "
            "provenance only.",
        ]
    elif state_pairing == "strict":
        report["warnings"] = [
            "state_sha256 is comparable only because both sides used the same evidence "
            "algorithm; it is not a cross-implementation proof of game-state equality."
        ]
    else:
        report["warnings"] = [
            STATE_WARNING,
            "same-logic-frame is opt-in evidence, not a weaker form of a state check: "
            "the report must not be read as 'states confirmed identical'.",
            "Only the replay SHA-256 and the exact logicFrame sets are enforced in this mode.",
        ]
    return report


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--windows", type=Path, required=True)
    p.add_argument("--linux", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument(
        "--state-pairing",
        choices=PAIRING_MODES,
        default="strict",
        help="strict (default) requires identical state_sha256; same-logic-frame only "
        "requires the same replay and logicFrame set and keeps both evidence records; "
        "common-projection compares the seven-field projection of a windows-retail "
        "manifest against the linux harness and fails closed on any unequal field",
    )
    p.add_argument(
        "--global-phase",
        action="store_true",
        help="also score a single stream-level offset s in {-1,0,+1} over the whole "
        "stream (MAD(retail[n], linux[n+s])): this is the calibrated number; the "
        "per-frame best-of-shifts figure is reported as an oracle bound only",
    )
    args = p.parse_args(argv)
    try:
        if args.frames < 1:
            raise EvidenceError("frames must be positive")
        result = compare(
            args.windows,
            args.linux,
            args.frames,
            state_pairing=args.state_pairing,
            global_phase=args.global_phase,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"MEASURED {result['pairs']} frames ({result['state_pairing']}); G0.4 still requires three real replays and visual review")
        if result["state_pairing"] == "common-projection":
            projection = result["state_projection"]
            print(
                f"PROJECTION: {', '.join(projection['fields'])} equal on "
                f"{projection['frames_equal']}/{projection['frames_total']} frames "
                f"({result['state_consistency']})"
            )
        elif result["state_pairing"] != "strict":
            print(f"WARNING: {STATE_WARNING}")
        phase = result.get("global_phase")
        if phase:
            print(
                "GLOBAL PHASE: best s={best} (sum {bsum} mean {bmean}) "
                "runner-up s={runner} (sum {rsum} mean {rmean}, gap {gap}); "
                "per-frame oracle mean {oracle} over {frames} comparable frames; "
                "shift counts {counts}".format(
                    best=phase["best_shift"],
                    bsum=_fmt(phase["best_shift_sum"]),
                    bmean=_fmt(phase["best_shift_mean"]),
                    runner=phase["runner_up_shift"],
                    rsum=_fmt(phase["runner_up_sum"]),
                    rmean=_fmt(phase["runner_up_mean"]),
                    gap=_fmt(phase["gap_sum"]),
                    oracle=_fmt(phase["per_frame_best_oracle"]["playfield_mad_mean"]),
                    frames=phase["comparable_frames"],
                    counts=phase["per_frame_best_shift_counts"],
                )
            )
            clean = phase["clean"]["per_shift_on_comparable_frames"]
            print(
                f"CLEAN (playfield MAD <= {phase['clean']['threshold']}): "
                + "; ".join(
                    f"s={shift} {row['clean_frames']}/{row['frames']} "
                    f"({_fmt(row['clean_fraction'])})"
                    for shift, row in clean.items()
                )
                + "; oracle bound "
                + f"{phase['per_frame_best_oracle']['playfield_mad_clean_frames']}"
                f"/{phase['per_frame_best_oracle']['frames']}"
            )
        return 0
    except (OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
