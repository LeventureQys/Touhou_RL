"""Three-replay WP8 paired-pixel suite for v4.0.0 (G0.4 scale-out).

The first paired frame (EXP-P0-056) proved the mechanism on one replay and 48
frames.  G0.4 asks for **three real replays x 200 consecutive logicFrames** with
a per-replay ``pixel-report.json``, a combined summary, and a human visual
review.  This tool is the orchestration layer for exactly that; every step is a
subprocess whose command line, exit code and wall time are recorded in
``suite-steps.json`` so a reader can replay it.

Subcommands (each writes its own evidence; ``all`` runs them in order):

``plan``      resolve the replay/stage/start-frame specs and print the chosen
              windows (decode-only; does not touch the game)
``windows``   run ``tools/rl/v400_win_fence_capture.py`` per spec
``linux``     stage the replay in WSL, capture the *actual* Windows logicFrame
              list (so the two sides cannot drift), ``convert`` to RGBA PNG
``compare``   run ``tools/rl/v400_pixel_pair.py --state-pairing same-logic-frame``
``summary``   per-replay + combined statistics, worst-frame lists, cross-side
              field checks, and worst-frame montages
``all``       windows -> linux -> compare -> summary

Deliberate limits, all reported instead of papered over:

* the Windows side is the *research fence build*, not the retail original;
* the two sides publish different state digests, so the pixel pairing mode is
  ``same-logic-frame`` and the report says ``state_consistency=not-asserted``;
  the strict default in ``v400_pixel_pair.py`` is untouched and still returns
  exit 3 on this data;
* this module adds a *field-level* cross-check (player bits, score, lives,
  power, stage, difficulty) that is real evidence about state agreement for
  those fields only -- it is not a proof of full state equality, and it never
  recomputes a shared digest.

    python tools/rl/v400_wp8_pair_suite.py all --out .tmp_v400_wp8_pair3_suite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
LINUX_CAPTURE = "tools/rl/v400_linux_frame_capture.py"
PIXEL_PAIR = "tools/rl/v400_pixel_pair.py"
WIN_CAPTURE = "tools/rl/v400_win_fence_capture.py"
WSL_ROOT = "/mnt/d/WorkShop/Touhou_RL"
WSL_REPLAY_DIR = "/root/v400-data/pair3"
FRAMES_PER_REPLAY = 200

#: The shared, effectively read-only retail data directory.  The game writes
#: ``score.dat`` and ``th08.cfg`` back on exit, so pointing ``--data-dir`` here
#: silently mutates shared state (this happened once; see EXP-P0-058).  Each case
#: therefore gets a **private runtime directory**: pristine ``.orig`` copies of
#: the two writable files plus read-only links to the two immovable data blobs.
WSL_SHARED_DATA = "/root/v400-data/game"
WSL_RUNTIME_ROOT = "/root/v400-data/runtime3"
SHARED_TEMPLATE_FILES = ("score.dat", "th08.cfg")
SHARED_TEMPLATE_LINKS = ("th08.dat", "thbgm.dat")
#: SHA-256 of the pristine ``*.orig`` copies.  The suite refuses to call a run
#: clean if the shared files differ from these before or after a capture, and it
#: never repairs or deletes anything on its own.
EXPECTED_SHARED_TEMPLATE = {
    "score.dat": "60e49abc8f0a955aa8fa883769a30c66913417be453632c9a8a99b0b8a5c5191",
    "th08.cfg": "48bf7621da5879ef5ec1392fcac2cc0a85c313912a24fb33b4cddccf0d28f07a",
}

#: The three replays.  Chosen for content spread, not convenience: a Normal
#: Stage 4A window, a Lunatic Stage 5 window (densest bullet load in the corpus)
#: and a Hard Stage 4B window.  None of them is the Stage 1 title card that
#: EXP-P0-056 had to use.
#:
#: ``startLogicFrame`` is a *minimum*; the window actually starts at the first
#: frame at or after it where the engine reports at least
#: ``bulletThreshold`` active bullets (a pre-registered content rule applied
#: before any pixel comparison, capped at ``maxSeekFrame``).  The per-replay
#: bullet profile recorded in the Windows manifest makes the rule auditable.
DEFAULT_SPECS: list[dict] = [
    {
        "name": "r1-normal-stage4a",
        "replay": "th08_replay/Th8_ud2970.rpy",
        "stage": 3,
        "stageName": "Stage 4A",
        "startLogicFrame": 1800,
        "bulletThreshold": 40,
        "maxSeekFrame": 4000,
        "note": "Normal / Marisa (solo); the replay EXP-P0-056 used, at a different stage",
    },
    {
        "name": "r2-lunatic-stage5",
        "replay": "th08_replay/Th8_ud0059.rpy",
        "stage": 5,
        "stageName": "Stage 5",
        "startLogicFrame": 1800,
        "bulletThreshold": 40,
        "maxSeekFrame": 4000,
        "note": "Lunatic / Youmu; heaviest bullet load in the corpus",
    },
    {
        "name": "r3-hard-stage4b",
        "replay": "th08_replay/Th8_ud2563.rpy",
        "stage": 4,
        "stageName": "Stage 4B",
        "startLogicFrame": 1800,
        "bulletThreshold": 40,
        "maxSeekFrame": 4000,
        "note": "Hard / Reimu (solo); 4B route, different boss than r1",
    },
]

#: Linux tracked-field index -> the Windows evidence key it must equal.
#: ``rlharness.cpp TrackedFields`` order: f4 score, f5 graze, f6 deaths,
#: f7 livesRemaining, f8 bombsRemaining, f9 playerPower, f10/f11 player x/y,
#: f13 playerState, f24 difficulty.  f6..f12 are published as %08x float bits.
FLOAT_FIELD_MAP = {
    "lives": 7,
    "power": 9,
    "player_x": 10,
    "player_y": 11,
}
INT_FIELD_MAP = {
    "score": 4,
    "player_state": 13,
    "stage": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def float_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def bits_float(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(value) & 0xFFFFFFFF))[0]


# ---------------------------------------------------------------------------
# Subprocess recording
# ---------------------------------------------------------------------------
class StepRunner:
    """Run subprocesses, record the exact command/exit code/wall time."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.steps: list[dict] = []

    def run(self, name: str, argv: list[str], *, timeout: float = 600.0, cwd: Path | None = None) -> dict:
        started = time.perf_counter()
        printable = " ".join(argv)
        try:
            completed = subprocess.run(
                argv,
                cwd=str(cwd or ROOT),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            exit_code = completed.returncode
            stdout, stderr = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = f"TIMEOUT after {timeout:.0f}s"
        wall = round(time.perf_counter() - started, 3)
        record = {
            "name": name,
            "command": printable,
            "exitCode": exit_code,
            "wallSeconds": wall,
            "stdoutTail": stdout[-4000:],
            "stderrTail": stderr[-4000:],
        }
        self.steps.append(record)
        self.flush()
        return record

    def flush(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(
            json.dumps({"steps": self.steps}, indent=2) + "\n", encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def windows_logic_frames(manifest_path: Path) -> list[int]:
    data = load_json(manifest_path)
    if data.get("source") != "windows-fence":
        raise ValueError(f"{manifest_path}: not a windows-fence manifest")
    frames = [int(row["logicFrame"]) for row in data["frames"]]
    if frames != sorted(frames) or len(set(frames)) != len(frames):
        raise ValueError(f"{manifest_path}: logicFrame list is not strictly increasing")
    if len(frames) > 1:
        expected = list(range(frames[0], frames[0] + len(frames)))
        if frames != expected:
            raise ValueError(
                f"{manifest_path}: logicFrames are not consecutive "
                f"({frames[0]}..{frames[-1]} with {len(frames)} rows)"
            )
    return frames


def require_frame_count(frames: list[int], expected: int) -> None:
    if len(frames) != expected:
        raise ValueError(f"require {expected} logicFrames, got {len(frames)}")


# ---------------------------------------------------------------------------
# Cross-side field check (never a shared digest)
# ---------------------------------------------------------------------------
def state_cross_check(
    windows_manifest_path: Path,
    linux_manifest_path: Path,
    *,
    expected_difficulty: int | None = None,
    expected_shot: int | None = None,
) -> dict:
    """Compare the fields both sides genuinely publish, bit for bit.

    Windows publishes the player position as float *bit patterns*
    (``player_bits``) and the scalar globals as Python floats read from the same
    ``ZunGlobals`` fields the Linux harness hashes, so the comparison is exact
    integer equality, not a tolerance.  Nothing here recomputes a state digest:
    the two algorithms stay different and the pixel report keeps saying
    ``state_consistency=not-asserted``.
    """
    win = load_json(windows_manifest_path)
    lin = load_json(linux_manifest_path)
    w = {int(row["logicFrame"]): row for row in win["frames"]}
    l = {int(row["logicFrame"]): row for row in lin["frames"]}
    if set(w) != set(l):
        raise ValueError(
            f"logicFrame sets differ: windows-only={sorted(set(w) - set(l))[:10]} "
            f"linux-only={sorted(set(l) - set(w))[:10]}"
        )
    counters = {name: 0 for name in (*FLOAT_FIELD_MAP, *INT_FIELD_MAP, "logicFrame")}
    mismatches: list[dict] = []
    for frame in sorted(w):
        wev = w[frame]["state_evidence"]
        lfields = l[frame]["state_evidence"]["fields"]
        checks = {"logicFrame": frame == int(l[frame]["logicFrame"])}
        for name, index in FLOAT_FIELD_MAP.items():
            checks[name] = float_bits(wev[name]) == int(lfields[f"f{index}"])
        for name, index in INT_FIELD_MAP.items():
            # Windows player_state is an int, score/stage are ints; compare the
            # raw published integer to the tracked field.
            value = wev.get(name)
            checks[name] = value is not None and int(value) == int(lfields[f"f{index}"])
        failed = [name for name, ok in checks.items() if not ok]
        for name, ok in checks.items():
            counters[name] += int(ok)
        if failed:
            mismatches.append(
                {
                    "logicFrame": frame,
                    "failed": failed,
                    "windows": {name: wev.get(name) for name in failed},
                    "linux": {
                        name: lfields.get(f"f{FLOAT_FIELD_MAP.get(name, INT_FIELD_MAP.get(name, 0))}")
                        for name in failed
                    },
                }
            )
    total = len(w)
    result = {
        "frames": total,
        "fields": {
            name: {"equal": counters[name], "total": total}
            for name in (*FLOAT_FIELD_MAP, *INT_FIELD_MAP, "logicFrame")
        },
        "mismatchCount": len(mismatches),
        "mismatches": mismatches[:20],
        "note": "field-level equality on the fields both sides publish verbatim; "
        "NOT a shared state digest and NOT a full state-equality proof",
    }
    if expected_difficulty is not None:
        lin_diff = {int(row["state_evidence"]["fields"]["f24"]) for row in lin["frames"]}
        result["difficulty"] = {
            "expected": expected_difficulty,
            "linuxFramesDiffering": sorted(value for value in lin_diff if value != expected_difficulty),
            "linuxValues": sorted(lin_diff),
        }
    if expected_shot is not None:
        lin_shot = {int(row["state_evidence"]["fields"]["f23"]) for row in lin["frames"]}
        result["shot"] = {
            "expected": expected_shot,
            "linuxValues": sorted(lin_shot),
            "equal": sorted(lin_shot) == [expected_shot],
        }
    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
REGIONS = ("playfield", "hud")
AUX_KEY = "whole_frame_excluding_fps_readout"


def select_review_frames(rows: list[dict], *, global_top: int = 3) -> list[dict]:
    """Pick the frames a human should actually look at.

    Two rules, both stated in the output so the choice is auditable:

    1. the worst frame of **each replay** by whole-frame MAD *excluding* the fps
       readout box -- the spec metric includes that box, which is a HUD value
       difference and cannot answer "is content missing?";
    2. the global ``global_top`` worst frames by playfield MAD that rule 1 did
       not already pick.

    The selection uses only measurements from the same reports; it never looks at
    the images themselves, so it cannot cherry-pick a flattering frame.
    """
    chosen: list[dict] = []
    seen: set[tuple[str, int]] = set()
    worst_per_replay: dict[str, dict] = {}
    for row in sorted(rows, key=lambda item: item["wholeFrameExcludingFpsMad"], reverse=True):
        worst_per_replay.setdefault(row["replay"], row)
    for rows_for_replay in worst_per_replay.values():
        chosen.append(
            {
                **rows_for_replay,
                "reason": "worst whole-frame MAD excluding the fps readout for this replay",
            }
        )
        seen.add((rows_for_replay["replay"], rows_for_replay["logicFrame"]))
    added = 0
    for row in sorted(rows, key=lambda item: item["playfieldMad"], reverse=True):
        if added >= global_top:
            break
        if (row["replay"], row["logicFrame"]) in seen:
            continue
        chosen.append({**row, "reason": "global top playfield MAD"})
        seen.add((row["replay"], row["logicFrame"]))
        added += 1
    return chosen


def frame_severity(row: dict) -> float:
    """The spec metric: the larger of the two required region MADs."""
    return max(row["regions"][name]["mean_absolute_difference"] for name in REGIONS)


def summarise_report(path: Path) -> dict:
    report = load_json(path)
    if report.get("schema") != "v400-wp8-paired-pixel-audit/2":
        raise ValueError(f"{path}: unexpected schema {report.get('schema')!r}")
    frames = report["frames"]
    aggregate = report["aggregate"]
    summary: dict = {
        "report": str(path),
        "replaySha256": report["replay_sha256"],
        "pairs": report["pairs"],
        "statePairing": report["state_pairing"],
        "stateConsistency": report["state_consistency"],
        "logicFrameRange": [frames[0]["logicFrame"], frames[-1]["logicFrame"]],
        "regions": aggregate,
        "aux": {name: aggregate[name] for name in aggregate if name not in REGIONS},
    }
    summary["worstBySpecMetric"] = [
        {
            "logicFrame": row["logicFrame"],
            "playfieldMad": row["regions"]["playfield"]["mean_absolute_difference"],
            "hudMad": row["regions"]["hud"]["mean_absolute_difference"],
            "differentPixelRatio": max(
                row["regions"][name]["different_pixel_ratio"] for name in REGIONS
            ),
            "maxChannelDifference": max(
                row["regions"][name]["max_channel_difference"] for name in REGIONS
            ),
        }
        for row in sorted(frames, key=frame_severity, reverse=True)[:10]
    ]
    summary["worstByPlayfieldMad"] = [
        {
            "logicFrame": row["logicFrame"],
            "playfieldMad": row["regions"]["playfield"]["mean_absolute_difference"],
            "hudMad": row["regions"]["hud"]["mean_absolute_difference"],
            "wholeFrameExcludingFpsMad": row["aux_regions"][AUX_KEY]["mean_absolute_difference"],
            "maxChannelDifference": row["regions"]["playfield"]["max_channel_difference"],
        }
        for row in sorted(
            frames,
            key=lambda row: row["regions"]["playfield"]["mean_absolute_difference"],
            reverse=True,
        )[:10]
    ]
    return summary


def combine_summaries(summaries: list[dict]) -> dict:
    """Pool per-frame means across replays; keep min/max as observed extremes."""
    combined: dict = {"replays": len(summaries), "frames": sum(s["pairs"] for s in summaries)}
    for name in REGIONS:
        per_replay = [s["regions"][name] for s in summaries]
        means = [entry["mean_of_frame_mean_absolute_difference"] for entry in per_replay]
        ratios = [entry["mean_different_pixel_ratio"] for entry in per_replay]
        counts = [s["pairs"] for s in summaries]
        combined[name] = {
            # Weighted by frame count so a replay with more pairs cannot be
            # outvoted by a replay with fewer.
            "pooled_mean_of_frame_mean_absolute_difference": float(
                np.average(means, weights=counts)
            ),
            "pooled_mean_different_pixel_ratio": float(np.average(ratios, weights=counts)),
            "max_frame_mean_absolute_difference": max(
                entry["max_frame_mean_absolute_difference"] for entry in per_replay
            ),
            "min_frame_mean_absolute_difference": min(
                entry["min_frame_mean_absolute_difference"] for entry in per_replay
            ),
            "max_frame_different_pixel_ratio": max(
                entry["max_frame_different_pixel_ratio"] for entry in per_replay
            ),
            "max_channel_difference": max(entry["max_channel_difference"] for entry in per_replay),
            "per_replay_mean": [round(value, 6) for value in means],
        }
    aux_names = [name for name in summaries[0]["aux"]]
    for name in aux_names:
        entries = [s["aux"][name] for s in summaries]
        counts = [s["pairs"] for s in summaries]
        combined[name] = {
            "pooled_mean_of_frame_mean_absolute_difference": float(
                np.average(
                    [entry["mean_of_frame_mean_absolute_difference"] for entry in entries],
                    weights=counts,
                )
            ),
            "max_channel_difference": max(entry["max_channel_difference"] for entry in entries),
            "max_frame_mean_absolute_difference": max(
                entry["max_frame_mean_absolute_difference"] for entry in entries
            ),
        }
    return combined


# ---------------------------------------------------------------------------
# Montages for the human review
# ---------------------------------------------------------------------------
def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGBA" or image.size != (640, 480):
            raise ValueError(f"{path}: expected PNG RGBA 640x480, got {image.format} {image.mode} {image.size}")
        return np.asarray(image.convert("RGB")).copy()


def difference_image(windows: np.ndarray, linux: np.ndarray, gain: int = 8) -> np.ndarray:
    delta = np.abs(windows.astype(np.int16) - linux.astype(np.int16)).astype(np.uint16)
    return np.clip(delta * gain, 0, 255).astype(np.uint8)


def worst_region(windows: np.ndarray, linux: np.ndarray, block: int = 40) -> tuple[int, int]:
    """Top-left of the ``block`` x ``block`` window with the largest summed delta."""
    delta = np.abs(windows.astype(np.int16) - linux.astype(np.int16)).sum(axis=2)
    integral = np.zeros((delta.shape[0] + 1, delta.shape[1] + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(delta, axis=0), axis=1)
    best = (-1, 0, 0)
    for y in range(0, delta.shape[0] - block + 1, block // 2):
        for x in range(0, delta.shape[1] - block + 1, block // 2):
            total = int(
                integral[y + block, x + block]
                - integral[y, x + block]
                - integral[y + block, x]
                + integral[y, x]
            )
            if total > best[0]:
                best = (total, y, x)
    return best[1], best[2]


def build_review_image(
    windows_path: Path,
    linux_path: Path,
    out_path: Path,
    *,
    label: str,
    zoom: int = 4,
    block: int = 40,
) -> dict:
    """Left: Windows frame. Middle: Linux frame. Right: amplified |diff| + zoom crop."""
    win = load_rgb(windows_path)
    lin = load_rgb(linux_path)
    diff = difference_image(win, lin)
    top, left = worst_region(win, lin, block)
    win_zoom = np.asarray(
        Image.fromarray(win[top : top + block, left : left + block]).resize(
            (block * zoom, block * zoom), Image.NEAREST
        )
    )
    lin_zoom = np.asarray(
        Image.fromarray(lin[top : top + block, left : left + block]).resize(
            (block * zoom, block * zoom), Image.NEAREST
        )
    )
    gap = np.full((480, 8, 3), 32, dtype=np.uint8)
    zoom_row = np.concatenate([win_zoom, np.full((block * zoom, 8, 3), 32, dtype=np.uint8), lin_zoom], axis=1)
    canvas = np.concatenate([win, gap, lin, gap, diff], axis=1)
    canvas = np.concatenate([canvas, np.full((8, canvas.shape[1], 3), 32, dtype=np.uint8)], axis=0)
    if zoom_row.shape[1] < canvas.shape[1]:
        pad = np.full((zoom_row.shape[0], canvas.shape[1] - zoom_row.shape[1], 3), 32, dtype=np.uint8)
        zoom_row = np.concatenate([zoom_row, pad], axis=1)
    canvas = np.concatenate([canvas, zoom_row], axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)
    return {
        "label": label,
        "image": str(out_path),
        "windows": str(windows_path),
        "linux": str(linux_path),
        "zoomBox": {"y": [top, top + block], "x": [left, left + block], "scale": zoom},
        "layout": "windows | linux | amplified x8 absolute difference (RGB); bottom: 4x zoom of the "
        "worst 40x40 block (windows left, linux right)",
        "diffGain": 8,
    }


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------
def wsl_command(inner: str) -> list[str]:
    return ["wsl", "-e", "bash", "-lc", f"cd {WSL_ROOT} && {inner}"]


def stage_replay_in_wsl(local: Path, runner: StepRunner, *, name: str) -> tuple[str, str]:
    """Copy the replay into WSL and verify the copy by SHA-256."""
    target = f"{WSL_REPLAY_DIR}/{local.name}"
    rel = local.resolve().relative_to(ROOT).as_posix()
    runner.run(
        f"{name}-stage",
        wsl_command(f"mkdir -p {WSL_REPLAY_DIR} && cp -f '{WSL_ROOT}/{rel}' '{target}'"),
        timeout=120.0,
    )
    local_sha = sha256_file(local)
    probe = runner.run(
        f"{name}-stage-sha", wsl_command(f"sha256sum '{target}'"), timeout=60.0
    )
    wsl_sha = probe["stdoutTail"].split()[0] if probe["exitCode"] == 0 and probe["stdoutTail"].split() else ""
    if wsl_sha.lower() != local_sha.lower():
        raise ValueError(f"staged replay sha mismatch for {target}: {wsl_sha!r} != {local_sha}")
    return target, local_sha


def parse_sha256sum(text: str) -> dict[str, str]:
    """Parse ``sha256sum`` output into ``{basename: hex}``."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            result[parts[1].rsplit("/", 1)[-1]] = parts[0].lower()
    return result


def shared_template_status(observed: dict[str, str]) -> dict:
    """Compare the shared files against the pristine ``.orig`` digests."""
    ok = all(observed.get(name) == digest for name, digest in EXPECTED_SHARED_TEMPLATE.items())
    return {
        "expected": dict(EXPECTED_SHARED_TEMPLATE),
        "observed": dict(observed),
        "clean": ok,
        "differing": sorted(
            name for name, digest in EXPECTED_SHARED_TEMPLATE.items() if observed.get(name) != digest
        ),
    }


def probe_shared_template(runner: StepRunner, tag: str) -> dict:
    paths = " ".join(f"'{WSL_SHARED_DATA}/{name}'" for name in SHARED_TEMPLATE_FILES)
    record = runner.run(f"shared-template-{tag}", wsl_command(f"sha256sum {paths}"), timeout=120.0)
    observed = parse_sha256sum(record["stdoutTail"]) if record["exitCode"] == 0 else {}
    return {"tag": tag, "exitCode": record["exitCode"], **shared_template_status(observed)}


def prepare_wsl_runtime(spec_name: str, runner: StepRunner) -> dict:
    """Build the private ``--data-dir`` for one case and prove its provenance.

    ``score.dat`` / ``th08.cfg`` are byte copies of the pristine ``.orig`` files
    (the game rewrites both), and ``th08.dat`` / ``thbgm.dat`` are read-only
    symlinks so the 500 MB of retail data is never duplicated or modified.
    """
    runtime = f"{WSL_RUNTIME_ROOT}/{spec_name}/game"
    copies = " ".join(
        f"cp -f '{WSL_SHARED_DATA}/{name}.orig' '{runtime}/{name}';" for name in SHARED_TEMPLATE_FILES
    )
    links = " ".join(
        f"ln -sfn '{WSL_SHARED_DATA}/{name}' '{runtime}/{name}';" for name in SHARED_TEMPLATE_LINKS
    )
    checks = " ".join(f"sha256sum '{runtime}/{name}';" for name in SHARED_TEMPLATE_FILES)
    resolved = " ".join(f"readlink -f '{runtime}/{name}';" for name in SHARED_TEMPLATE_LINKS)
    script = f"set -e; mkdir -p '{runtime}'; {copies} {links} {checks} {resolved} true"
    record = runner.run(f"{spec_name}-runtime", wsl_command(script), timeout=180.0)
    observed = parse_sha256sum(record["stdoutTail"])
    status = shared_template_status(observed)
    links = [line.strip() for line in record["stdoutTail"].splitlines() if line.strip().startswith("/")]
    return {
        "runtimeDataDir": runtime,
        "exitCode": record["exitCode"],
        "copiedSha256": observed,
        "pristineEqual": status["clean"],
        "dataLinks": links,
        "note": "private --data-dir: score.dat/th08.cfg copied from .orig, "
        "th08.dat/thbgm.dat read-only symlinks",
    }


def linux_capture(
    spec: dict, win_manifest: Path, out_dir: Path, runner: StepRunner
) -> dict:
    local = ROOT / spec["replay"]
    wsl_replay, replay_sha = stage_replay_in_wsl(local, runner, name=spec["name"])
    runtime = prepare_wsl_runtime(spec["name"], runner)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = windows_logic_frames(win_manifest)
    targets_path = out_dir / "targets.json"
    targets_path.write_text(json.dumps({"logicFrames": frames}) + "\n", encoding="utf-8")
    capture_dir = out_dir / "linux"
    rel_targets = targets_path.relative_to(ROOT).as_posix()
    rel_capture = capture_dir.relative_to(ROOT).as_posix()
    record = runner.run(
        f"{spec['name']}-linux-capture",
        wsl_command(
            f"python3 {LINUX_CAPTURE} capture --out {rel_capture} "
            f"--targets-file {rel_targets} --replay '{wsl_replay}' --stage {spec['stage']} "
            f"--data-dir '{runtime['runtimeDataDir']}'"
        ),
        timeout=900.0,
    )
    convert = runner.run(
        f"{spec['name']}-linux-convert",
        [sys.executable, str(ROOT / LINUX_CAPTURE), "convert", "--capture", str(capture_dir / "capture.json")],
        timeout=600.0,
    )
    # Per-case proof that the private data dir did the writing, not the shared one.
    shared_after = probe_shared_template(runner, f"after-{spec['name']}")
    runtime_files = runner.run(
        f"{spec['name']}-runtime-after",
        wsl_command(
            f"sha256sum '{runtime['runtimeDataDir']}/score.dat' '{runtime['runtimeDataDir']}/th08.cfg'"
        ),
        timeout=120.0,
    )
    runtime_after = parse_sha256sum(runtime_files["stdoutTail"])
    return {
        "captureExitCode": record["exitCode"],
        "convertExitCode": convert["exitCode"],
        "replayStagedInWsl": wsl_replay,
        "replaySha256": replay_sha,
        "runtime": runtime,
        "sharedTemplateAfter": shared_after,
        "runtimeAfter": {
            "sha256": runtime_after,
            "rewritten": sorted(
                name
                for name in SHARED_TEMPLATE_FILES
                if runtime_after.get(name) != runtime["copiedSha256"].get(name)
            ),
            "note": "files the game rewrote in the private runtime copy; "
            "a rewrite here is expected and is the proof the shared template was spared",
        },
    }


def strict_control(spec: dict, out_dir: Path, runner: StepRunner) -> dict:
    """Same real data under the untouched strict default: it must refuse."""
    target = out_dir / "pixel-report-strict.json"
    if target.exists():
        target.unlink()
    record = runner.run(
        f"{spec['name']}-compare-strict",
        [
            sys.executable,
            str(ROOT / PIXEL_PAIR),
            "--windows", str(out_dir / "windows" / "windows-manifest.json"),
            "--linux", str(out_dir / "linux" / "linux-manifest.json"),
            "--frames", str(FRAMES_PER_REPLAY),
            "--out", str(target),
        ],
        timeout=600.0,
    )
    return {"exitCode": record["exitCode"], "reportWritten": target.exists()}


def load_specs(path: Path | None) -> list[dict]:
    specs = json.loads(path.read_text(encoding="utf-8")) if path else DEFAULT_SPECS
    if not isinstance(specs, list) or not specs:
        raise ValueError("specs must be a nonempty list")
    for spec in specs:
        for key in ("name", "replay", "stage", "startLogicFrame"):
            if key not in spec:
                raise ValueError(f"spec missing {key}: {spec}")
        if not (ROOT / spec["replay"]).is_file():
            raise ValueError(f"spec replay missing: {spec['replay']}")
    names = [spec["name"] for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("spec names must be unique")
    return specs


def replay_metadata(path: Path) -> dict:
    sys.path.insert(0, str(ROOT))
    from touhou_rl.v400.gym.replay_starts import decode_replay

    starts = decode_replay(path)
    return {
        "sha256": sha256_file(path),
        "difficulty": starts[0].difficulty,
        "difficultyName": starts[0].difficulty_name,
        "shot": starts[0].shot,
        "shotName": starts[0].shot_name,
        "stages": [
            {"index": s.stage_index, "name": s.stage_name, "inputRecords": s.input_records}
            for s in starts
        ],
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_plan(args: argparse.Namespace) -> int:
    specs = load_specs(Path(args.specs) if args.specs else None)
    out = {
        "framesPerReplay": FRAMES_PER_REPLAY,
        "specs": [],
    }
    for spec in specs:
        meta = replay_metadata(ROOT / spec["replay"])
        stages = [entry["index"] for entry in meta["stages"]]
        if spec["stage"] not in stages:
            raise ValueError(f"{spec['name']}: stage {spec['stage']} not in replay stages {stages}")
        out["specs"].append({**spec, "replayMetadata": meta})
    text = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def cmd_windows(args: argparse.Namespace) -> int:
    specs = load_specs(Path(args.specs) if args.specs else None)
    base = Path(args.out).resolve()
    runner = StepRunner(base / "suite-steps-windows.json")
    failures = []
    for spec in specs:
        out_dir = base / spec["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        record = runner.run(
            f"{spec['name']}-windows-capture",
            [
                sys.executable,
                str(ROOT / WIN_CAPTURE),
                "--out", str(out_dir / "windows"),
                "--run-dir", str(out_dir / "windows-run"),
                "--replay-file", str(ROOT / spec["replay"]),
                "--replay-stage", str(spec["stage"]),
                "--start-logic-frame", str(spec["startLogicFrame"]),
                "--seek-bullet-threshold", str(spec.get("bulletThreshold", 0)),
                "--max-seek-frame", str(spec.get("maxSeekFrame", 0)),
                "--frames", str(FRAMES_PER_REPLAY),
                "--label", spec["name"],
            ],
            timeout=300.0,
        )
        if record["exitCode"] != 0:
            failures.append(spec["name"])
    print(json.dumps({"windowsFailures": failures}, indent=2))
    return 0 if not failures else 1

def cmd_linux(args: argparse.Namespace) -> int:
    specs = load_specs(Path(args.specs) if args.specs else None)
    base = Path(args.out).resolve()
    runner = StepRunner(base / "suite-steps-linux.json")
    results, failures = {}, []
    before = probe_shared_template(runner, "before")
    for spec in specs:
        out_dir = base / spec["name"]
        info = linux_capture(spec, out_dir / "windows" / "windows-manifest.json", out_dir, runner)
        results[spec["name"]] = info
        if info["captureExitCode"] != 0 or info["convertExitCode"] != 0:
            failures.append(spec["name"])
        if not info["runtime"]["pristineEqual"]:
            failures.append(f"{spec['name']}:runtime-not-pristine")
    after = probe_shared_template(runner, "after")
    # The template must be untouched by a private-data-dir run.  Nothing is
    # repaired or deleted here: a dirty template is reported and stops the run.
    if not before["clean"] or not after["clean"]:
        failures.append("shared-template-dirty")
    report = {
        "linux": results,
        "sharedTemplate": {"before": before, "after": after},
        "failures": failures,
    }
    (base / "linux-runtime.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


def cmd_compare(args: argparse.Namespace) -> int:
    specs = load_specs(Path(args.specs) if args.specs else None)
    base = Path(args.out).resolve()
    runner = StepRunner(base / "suite-steps-compare.json")
    results, failures = {}, []
    for spec in specs:
        out_dir = base / spec["name"]
        record = runner.run(
            f"{spec['name']}-compare",
            [
                sys.executable,
                str(ROOT / PIXEL_PAIR),
                "--windows", str(out_dir / "windows" / "windows-manifest.json"),
                "--linux", str(out_dir / "linux" / "linux-manifest.json"),
                "--frames", str(FRAMES_PER_REPLAY),
                "--state-pairing", "same-logic-frame",
                "--out", str(out_dir / "pixel-report.json"),
            ],
            timeout=600.0,
        )
        strict = strict_control(spec, out_dir, runner)
        results[spec["name"]] = {"exitCode": record["exitCode"], "strict": strict}
        if record["exitCode"] != 0:
            failures.append(spec["name"])
    print(json.dumps({"compare": results, "failures": failures}, indent=2))
    return 0 if not failures else 1


def cmd_summary(args: argparse.Namespace) -> int:
    specs = load_specs(Path(args.specs) if args.specs else None)
    base = Path(args.out).resolve()
    runner = StepRunner(base / "suite-steps-summary.json")
    reviews_dir = base / "review"
    per_replay: list[dict] = []
    all_rows: list[dict] = []
    runtime_report_path = base / "linux-runtime.json"
    runtime_report = load_json(runtime_report_path) if runtime_report_path.is_file() else None
    for spec in specs:
        out_dir = base / spec["name"]
        report_path = out_dir / "pixel-report.json"
        summary = summarise_report(report_path)
        report = load_json(report_path)
        for row in report["frames"]:
            all_rows.append(
                {
                    "replay": spec["name"],
                    "logicFrame": row["logicFrame"],
                    "playfieldMad": row["regions"]["playfield"]["mean_absolute_difference"],
                    "hudMad": row["regions"]["hud"]["mean_absolute_difference"],
                    "wholeFrameExcludingFpsMad": row["aux_regions"][AUX_KEY]["mean_absolute_difference"],
                    "maxChannelDifference": row["regions"]["playfield"]["max_channel_difference"],
                    "windowsImage": row["windows_image"],
                    "linuxImage": row["linux_image"],
                }
            )
        meta = replay_metadata(ROOT / spec["replay"])
        cross = state_cross_check(
            out_dir / "windows" / "windows-manifest.json",
            out_dir / "linux" / "linux-manifest.json",
            expected_difficulty=meta["difficulty"],
            expected_shot=meta["shot"],
        )
        win_manifest = load_json(out_dir / "windows" / "windows-manifest.json")
        bullets = [
            int(row["state_evidence"]["active_bullets"]) for row in win_manifest["frames"]
        ]
        seek = (win_manifest.get("capture") or {}).get("seek") or {}
        summary.update(
            {
                "name": spec["name"],
                "replay": spec["replay"],
                "replaySha256": meta["sha256"],
                "stage": spec["stage"],
                "stageName": spec["stageName"],
                "difficulty": meta["difficultyName"],
                "shot": meta["shotName"],
                "startLogicFrameRequested": spec["startLogicFrame"],
                "windowRule": {
                    "kind": "first logicFrame >= startLogicFrame with activeBullets >= threshold",
                    "bulletThreshold": spec.get("bulletThreshold", 0),
                    "maxSeekFrame": spec.get("maxSeekFrame", 0),
                    "stoppedBy": seek.get("stoppedBy"),
                    "bulletThresholdReachedAt": seek.get("bulletThresholdReachedAt"),
                    "maxActiveBulletsDuringSeek": seek.get("maxActiveBulletsDuringSeek"),
                    "note": seek.get("bulletThresholdNote"),
                    "profileEveryFrames": seek.get("profileEveryFrames"),
                    "profile": seek.get("profile"),
                },
                "replayMetadata": meta,
                "crossCheck": cross,
                "windowsActiveBullets": {
                    "min": min(bullets),
                    "mean": round(float(np.mean(bullets)), 3),
                    "max": max(bullets),
                    "framesWithBullets": int(sum(1 for value in bullets if value > 0)),
                    "frames": len(bullets),
                },
                "windowsAllChecksPass": all(
                    bool(row["checks_all_pass"]) for row in win_manifest["frames"]
                ),
            }
        )
        per_replay.append(summary)

    combined = combine_summaries(per_replay)
    worst_overall = sorted(all_rows, key=lambda row: max(row["playfieldMad"], row["hudMad"]), reverse=True)[:10]
    worst_content = sorted(all_rows, key=lambda row: row["wholeFrameExcludingFpsMad"], reverse=True)[:10]
    worst_playfield = sorted(all_rows, key=lambda row: row["playfieldMad"], reverse=True)[:10]

    # Review images: one per replay plus the global playfield worst, because the
    # spec metric (which includes the HUD fps readout) is a value difference and
    # cannot answer "is content missing, or is it filtering/colour/font?".
    review_entries = []
    for index, row in enumerate(select_review_frames(all_rows)):
        if row["reason"].startswith("worst whole-frame"):
            key = f"per-replay-{row['replay']}-f{row['logicFrame']:05d}"
        else:
            key = f"global-{index:02d}-{row['replay']}-f{row['logicFrame']:05d}"
        out_png = reviews_dir / f"{key}.png"
        info = build_review_image(
            Path(row["windowsImage"]),
            Path(row["linuxImage"]),
            out_png,
            label=f"{row['replay']} logicFrame {row['logicFrame']}",
        )
        info["metrics"] = row
        info["selectionReason"] = row["reason"]
        review_entries.append(info)

    payload = {
        "schema": "v400-wp8-pair-suite-summary/1",
        "status": "measured-not-G0-pass",
        "framesPerReplay": FRAMES_PER_REPLAY,
        "replays": len(per_replay),
        "perReplay": per_replay,
        "combined": combined,
        "worst10BySpecMetric": worst_overall,
        "worst10ByPlayfieldMad": worst_playfield,
        "worst10ByWholeFrameExcludingFps": worst_content,
        "reviewImages": review_entries,
        "sharedTemplate": (
            {
                "before": runtime_report["sharedTemplate"]["before"],
                "after": runtime_report["sharedTemplate"]["after"],
                "privateDataDirs": {
                    name: info["runtime"]["runtimeDataDir"] for name, info in runtime_report["linux"].items()
                },
                "note": "the linux capture runs on a private --data-dir; the shared template's "
                "score.dat/th08.cfg must still equal their pristine .orig digests",
            }
            if runtime_report
            else None
        ),
        "warnings": [
            "the Windows side is the research fence build, not the retail original",
            "state consistency is not asserted: the two sides publish different digests",
            "the HUD fps readout is a value difference; HUD numbers include it",
        ],
    }
    summary_path = base / "suite-summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "replays": len(per_replay),
                "frames": payload["combined"]["frames"],
                "playfieldPooledMean": combined["playfield"][
                    "pooled_mean_of_frame_mean_absolute_difference"
                ],
                "hudPooledMean": combined["hud"]["pooled_mean_of_frame_mean_absolute_difference"],
                "reviewImages": [entry["image"] for entry in review_entries],
            },
            indent=2,
        )
    )
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    for command in (cmd_plan, cmd_windows, cmd_linux, cmd_compare, cmd_summary):
        code = command(args)
        if code != 0:
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("v400-wp8-pair-suite", description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--specs", default=None, help="JSON list of {name,replay,stage,startLogicFrame}")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (
        ("plan", cmd_plan),
        ("windows", cmd_windows),
        ("linux", cmd_linux),
        ("compare", cmd_compare),
        ("summary", cmd_summary),
        ("all", cmd_all),
    ):
        parser_sub = sub.add_parser(name)
        parser_sub.set_defaults(func=func)
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

