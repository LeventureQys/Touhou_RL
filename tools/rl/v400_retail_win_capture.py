"""EXP-P0-060 (v4.0.0 WP8): can the ORIGINAL retail TH08 be run and observed on Windows?

Three bounded feasibility phases.  Every phase is one launch of the retail image
(``third_party/th08/resources/th08.exe``, 840,704 B, SHA256 ``330fbdbf...d924``)
and is expected to finish well inside 180 s:

``launch``
    Assemble a *private* data dir (retail exe + ``th08.dat``/``thbgm.dat``/
    ``th08.cfg``/``score.dat`` copied out of ``third_party/th08/build/play_fence``),
    boot it, and record process / window class+title / client size / boot time.
    Then navigate to Replay and play back ``th08_replay/Th8_ud2970.rpy``, recording
    how long each step took and a GDI screenshot of the title screen, the replay
    list and the running playback.
``memory``
    Same navigation, then sample retail memory (``gameplayFrameCounter``, score,
    lives, deaths, bombs, power, graze, player position, replay flag) and prove the
    reads behave like the game's own state: frame counter strictly monotonic with a
    ~60 Hz slope, score never decreasing, resource fields in range.  An independent
    
    scan for the ``u32`` that advances by one per frame validates the GameManager
    offset instead of trusting it.
``capture``
    Same navigation, then N *consecutive* logic frames, each with one 640x480 PNG
    whose pixels are proven same-frame (the frame counter read before and after the
    ``BitBlt`` is identical) plus the memory values of that frame.  It also writes
    ``windows-retail-manifest.json`` (``source="windows-retail"``, one
    ``state_evidence`` per frame) so the retail frames can be paired with the
    Linux harness through ``v400_pixel_pair.py --state-pairing common-projection``.
    ``--present-lag n`` publishes the pixels grabbed while the counter reads ``C``
    as the presentation of logic frame ``C - n`` (the state of that logic frame is
    the probe read at ``C - n``, so the projection stays honest); ``n=0`` is the
    unmodified behaviour.

Hard rules encoded here (all of them are checked, not documented):

* the retail image is identified by size+SHA256 before anything runs (exit 2 otherwise);
* exactly one ``th08.exe`` may exist -- the phase aborts otherwise (exit 5);
* nothing under ``third_party/th08`` is written: the exe is copied, the assets are
  copied (not hard-linked), the ``.rpy`` is read and copied, and the hash of every
  source file is recorded before and after the run;
* the game process is only ever opened with ``PROCESS_VM_READ|PROCESS_QUERY_INFORMATION``
  (``touhou_rl.memory.ProcessMemory``), never written;
* the whole phase is bounded by ``--budget-seconds`` (default 170) and closes the
  process on every exit path.

Usage::

    python tools/rl/v400_retail_win_capture.py --phase launch  --out .tmp_v400_retail_win_launch
    python tools/rl/v400_retail_win_capture.py --phase memory  --out .tmp_v400_retail_win_memory
    python tools/rl/v400_retail_win_capture.py --phase capture --out .tmp_v400_retail_win_capture
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RETAIL_EXE = ROOT / "third_party/th08/resources/th08.exe"
ASSET_DIR = ROOT / "third_party/th08/build/play_fence"
DEFAULT_REPLAY = ROOT / "th08_replay/Th8_ud2970.rpy"
DEFAULT_ADDRESSES = ROOT / ".tmp_v400_retail_win_addresses.json"
DEFAULT_RUN_DIR = ROOT / ".tmp_v400_retail_win_rundir"

RETAIL_SIZE = 840704
RETAIL_SHA256 = "330fbdbf58a710829d65277b4f312cfbb38d5448b3df523e79350b879213d924"

#: Files the retail 1.00d engine needs next to the exe.  ``th08.dat``/``thbgm.dat``
#: are read-only assets; ``th08.cfg``/``score.dat`` are rewritten by the game, which
#: is exactly why they are copies in a private directory and never the originals.
ASSET_FILES = ("th08.dat", "thbgm.dat", "th08.cfg", "score.dat")
MUTABLE_ASSETS = ("th08.cfg", "score.dat")

#: Right-hand HUD strip of the 640x480 client area (Score / HiScore / lives / bombs
#: / power / graze).  Cropped out so a human can compare the numbers with the memory
#: values; no OCR is performed by this tool.
HUD_CROP = (440, 0, 640, 480)

#: Common state projection shared with the Linux harness manifest (EXP-P0-062).
#: The names are the *canonical* side-neutral names; the probe keys are the retail
#: memory fields they come from.  ``lives``/``bombs``/``deaths``/``power`` are
#: float32 in the game's ``ZunGlobals`` while ``score``/``graze``/``stage`` are
#: integers, so every field is published as its exact 32-bit pattern instead of a
#: rounded decimal.
PROJECTION_FIELDS = ("score", "lives", "bombs", "deaths", "power", "graze", "stage")
PROJECTION_FLOAT_FIELDS = ("lives", "bombs", "deaths", "power")
PROJECTION_PROBE_KEYS = {
    "score": "score",
    "lives": "livesF32",
    "bombs": "bombsF32",
    "deaths": "deathsF32",
    "power": "powerF32",
    "graze": "graze",
    "stage": "stage",
}

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_BLOCKED = 3
EXIT_PROCESS_DIED = 4
EXIT_INSTANCE = 5
EXIT_STATE = 6
EXIT_TIMEOUT = 7


# ---------------------------------------------------------------------------
# identity helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pe_summary(path: Path) -> dict:
    """Machine / image base / subsystem / DllCharacteristics of a PE32 image.

    The retail image must be a 32-bit GUI executable with a *fixed* image base:
    ``DllCharacteristics & 0x0040`` (``DYNAMIC_BASE``) has to be clear, otherwise
    the linker-map addresses in the address table are not the runtime addresses.
    """
    raw = Path(path).read_bytes()[:0x400]
    offset = struct.unpack_from("<I", raw, 0x3C)[0]
    if raw[offset : offset + 4] != b"PE\x00\x00":
        raise ValueError(f"{path}: missing PE signature")
    machine = struct.unpack_from("<H", raw, offset + 4)[0]
    sections = struct.unpack_from("<H", raw, offset + 6)[0]
    optional = offset + 24
    magic = struct.unpack_from("<H", raw, optional)[0]
    image_base = struct.unpack_from("<I", raw, optional + 28)[0]
    subsystem = struct.unpack_from("<H", raw, optional + 68)[0]
    dll_characteristics = struct.unpack_from("<H", raw, optional + 70)[0]
    return {
        "machine": hex(machine),
        "machineIsI386": machine == 0x014C,
        "sections": sections,
        "optionalMagic": hex(magic),
        "imageBase": hex(image_base),
        "subsystem": subsystem,
        "subsystemIsGui": subsystem == 2,
        "dllCharacteristics": hex(dll_characteristics),
        "dynamicBase": bool(dll_characteristics & 0x0040),
    }


def retail_identity(path: Path = RETAIL_EXE) -> dict:
    """Size + SHA256 + PE facts of the retail image (no launch)."""
    path = Path(path)
    info: dict = {"path": str(path), "present": path.is_file()}
    if not info["present"]:
        info["verified"] = False
        info["reason"] = "retail image missing"
        return info
    info["size"] = path.stat().st_size
    info["sha256"] = sha256_file(path)
    info["verified"] = info["size"] == RETAIL_SIZE and info["sha256"] == RETAIL_SHA256
    if not info["verified"]:
        info["reason"] = "size/SHA256 mismatch: refusing to run a reconstruction as retail"
    try:
        info["pe"] = pe_summary(path)
    except (ValueError, struct.error) as exc:  # pragma: no cover - defensive
        info["pe"] = {"error": f"{type(exc).__name__}: {exc}"}
    return info


def plan_assets(asset_dir: Path = ASSET_DIR) -> list[dict]:
    """The copy plan for the private data dir (pure: no filesystem writes).

    ``kind`` is ``"exe"`` for the retail image (always copied from ``resources/``)
    and ``"asset"`` for everything taken from the template.  ``mutable`` marks the
    two files the engine rewrites at runtime, which is why they are never linked.
    """
    plan = [{"name": "th08.exe", "source": str(RETAIL_EXE), "kind": "exe", "mutable": False}]
    for name in ASSET_FILES:
        plan.append(
            {
                "name": name,
                "source": str(Path(asset_dir) / name),
                "kind": "asset",
                "mutable": name in MUTABLE_ASSETS,
            }
        )
    return plan


# ---------------------------------------------------------------------------
# pure reducers (unit-testable without a game)
# ---------------------------------------------------------------------------
def monotonic_problems(frames: list[int], scores: list[int], deaths: list[int]) -> list[str]:
    """Return the list of contract violations in a sample sequence."""
    problems: list[str] = []
    for index in range(1, len(frames)):
        if frames[index] < frames[index - 1]:
            problems.append(f"gameplayFrame decreased at sample {index}: {frames[index-1]} -> {frames[index]}")
    for index in range(1, len(scores)):
        if scores[index] < scores[index - 1]:
            problems.append(f"score decreased at sample {index}: {scores[index-1]} -> {scores[index]}")
    for index in range(1, len(deaths)):
        if deaths[index] < deaths[index - 1]:
            problems.append(f"deaths decreased at sample {index}: {deaths[index-1]} -> {deaths[index]}")
    return problems


def consecutive_run(frames: list[int]) -> tuple[int, int]:
    """Longest run of strictly +1 frame numbers: ``(start_index, length)``."""
    best = (0, 0)
    start = 0
    for index in range(1, len(frames) + 1):
        if index < len(frames) and frames[index] == frames[index - 1] + 1:
            continue
        if index - start > best[1]:
            best = (start, index - start)
        start = index
    return best


def frame_deltas(frames: list[int]) -> list[int]:
    return [frames[i] - frames[i - 1] for i in range(1, len(frames))]


def seek_counter_target(start_logic_frame: int, present_lag: int) -> int:
    """Counter value to seek to so that every published label can be assembled.

    A published label ``L`` needs the grab at counter ``L + present_lag`` *and* the
    memory probe at counter ``L``, so the run has to start at
    ``min(L, L + present_lag)`` -- i.e. ``start + min(0, present_lag)``.  A positive
    lag therefore seeks to the requested frame (its probe is still needed) and only
    the grabs move forward.  The target is clamped at 0: the counter never goes
    below it.
    """
    return max(0, int(start_logic_frame) + min(0, int(present_lag)))


def present_lag_label(counter: int, present_lag: int) -> int:
    """Logic frame whose *pixels* a grab taken at ``counter`` is filed under."""
    return int(counter) - int(present_lag)


def align_present_lag(
    samples: list[dict], needed: int, present_lag: int
) -> tuple[list[dict], list[int], list[int]]:
    """Re-file consecutive counter samples into ``needed`` presentation labels.

    ``samples`` is an increasing run of accepted grabs, each
    ``{"counter", "probe", "image", "t"}``.  A published row for logic frame ``F``
    takes its pixels from the sample at counter ``F + present_lag`` and its memory
    state from the sample at counter ``F`` -- the state therefore still belongs to
    the frame the row is labelled with, which is what makes the seven-field
    projection comparable.  The first label is the earliest one both halves of which
    exist (``first_counter + max(0, -present_lag)``); missing counters are reported,
    never filled in: the caller fails closed on a non-empty ``missing`` list.
    """
    by_counter = {int(row["counter"]): row for row in samples}
    if not samples:
        return [], [], []
    first_label = samples[0]["counter"] + max(0, -int(present_lag))
    rows: list[dict] = []
    missing: list[int] = []
    for label in range(first_label, first_label + needed):
        image_row = by_counter.get(label + present_lag)
        state_row = by_counter.get(label)
        if image_row is None or state_row is None:
            missing.append(label)
            continue
        rows.append(
            {
                "frame": label,
                "stateFrame": label,
                "frameBefore": image_row.get("frameBefore", image_row["counter"]),
                "frameAfter": image_row.get("frameAfter", image_row["counter"]),
                "probe": state_row["probe"],
                "image": image_row["image"],
                "t": image_row["t"],
            }
        )
    return rows, missing, sorted(by_counter)


#: Which grab of a frame's series is published as that frame's pixels.
DOUBLE_GRAB_PICKS = ("first", "last")
DOUBLE_GRAB_DEFAULT_COUNT = 2
DOUBLE_GRAB_DEFAULT_INTERVAL_MS = 4.0
DOUBLE_GRAB_MAX_PER_FRAME = 12


def grab_series_summary(grabs: list[dict]) -> dict:
    """Summarise every grab taken while ``gameplayFrameCounter`` stayed on one value.

    A grab only enters a series when the two counter reads that bracket the BitBlt
    are *equal*, so the counter did not move while the pixels were copied.  If the
    pixels of two grabs in the same series differ, something outside the logic
    frame changed the client area -- a present.  That is the only present evidence
    the retail build offers without a frame fence, and it is what makes a frame's
    "1 frame presentation ambiguity" decidable instead of a coin flip:

    * ``phaseDecidable`` -- at least two grabs differ, so a present happened inside
      the frame window and the *last* grab is provably the post-present content;
    * ``identical`` -- every grab is byte-identical, so no present was observed;
      the series says nothing about whether the pixels are this frame's content or
      the previous one's.
    """
    if not grabs:
        return {
            "count": 0,
            "distinctSha256": 0,
            "identical": False,
            "phaseDecidable": False,
            "firstSha256": None,
            "lastSha256": None,
            "changeIndices": [],
            "spanMs": 0.0,
            "maxGapMs": 0.0,
            "durationMs": 0.0,
            "note": "no grab was accepted for this frame",
        }
    shas = [str(row["sha256"]) for row in grabs]
    changes = [index for index in range(1, len(shas)) if shas[index] != shas[index - 1]]
    distinct = len(dict.fromkeys(shas))
    starts = [float(row.get("startMs", 0.0)) for row in grabs]
    gaps = [starts[index] - starts[index - 1] for index in range(1, len(starts))]
    return {
        "count": len(grabs),
        "distinctSha256": distinct,
        "identical": distinct == 1,
        "phaseDecidable": distinct > 1,
        "firstSha256": shas[0],
        "lastSha256": shas[-1],
        "changeIndices": changes,
        "spanMs": round(starts[-1] - starts[0], 4) if len(starts) > 1 else 0.0,
        "maxGapMs": round(max(gaps), 4) if gaps else 0.0,
        "durationMs": round(sum(float(row.get("durationMs", 0.0)) for row in grabs), 4),
        "note": (
            "a present was observed inside this logic frame, so the last grab is "
            "the post-present content of the frame"
            if distinct > 1
            else "no present was observed inside this logic frame: every grab shows "
            "the same presentation, so the phase is not decidable from pixels alone"
        ),
    }


def pick_grab_index(grabs: list[dict], pick: str) -> int:
    """Index of the grab published for a frame: the earliest or the latest one."""
    if pick not in DOUBLE_GRAB_PICKS:
        raise ValueError(f"unknown double-grab pick {pick!r}; expected one of {DOUBLE_GRAB_PICKS}")
    if not grabs:
        raise ValueError("cannot pick from an empty grab series")
    return 0 if pick == "first" else len(grabs) - 1


def other_grab_pick(pick: str) -> str:
    """The pick that is *not* published as the primary manifest."""
    return "last" if pick == "first" else "first"


def float_bits(value: float) -> int:
    """The exact 32-bit pattern of a value read as the game's ``float``."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def retail_state_projection(probe: dict) -> dict:
    """Seven-field projection of one retail memory probe: exact, side-neutral.

    Returns ``{field: {"value": <decimal>, "bits": "0x........"}}``.  Equality
    downstream is decided on ``bits`` only, so ``-0.0``/``0.0`` and NaN payloads
    cannot compare equal by accident and no tolerance is involved.
    """
    projection: dict = {}
    missing: list[str] = []
    for name in PROJECTION_FIELDS:
        key = PROJECTION_PROBE_KEYS[name]
        value = probe.get(key)
        if value is None:
            missing.append(key)
            continue
        if name in PROJECTION_FLOAT_FIELDS:
            projection[name] = {"value": float(value), "bits": f"0x{float_bits(value):08x}"}
        else:
            projection[name] = {"value": int(value), "bits": f"0x{int(value) & 0xFFFFFFFF:08x}"}
    if missing:
        raise ValueError(
            f"retail memory probe lacks {missing}; refusing to publish a partial projection"
        )
    return projection


def retail_state_digest(logic_frame: int, projection: dict) -> str:
    """SHA-256 over the canonical projection of one frame (retail-side only).

    This digest is *not* comparable with the Linux ``state_sha256``: it covers
    the seven common fields, while the Linux digest folds its own 25-field
    record.  The pairing that matters is the projection itself
    (``--state-pairing common-projection``), not this hash.
    """
    canonical = json.dumps(
        {
            "algorithm": "retail-memory-projection/1",
            "logicFrame": int(logic_frame),
            "projection": {name: projection[name]["bits"] for name in PROJECTION_FIELDS},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_bytes(canonical.encode("utf-8"))


def to_rgba(image):
    """GDI ``BitBlt`` frames are RGB; append an opaque alpha so every PNG is RGBA."""
    import numpy as np

    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 4:
        return np.ascontiguousarray(array.astype(np.uint8, copy=False))
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 or HxWx4 frame, got {array.shape}")
    alpha = np.full(array.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.ascontiguousarray(
        np.concatenate([array.astype(np.uint8, copy=False), alpha], axis=2)
    )


def build_retail_manifest(
    frames: list[dict],
    *,
    replay_path: Path,
    replay_sha256: str,
    stage: int,
    retail_image: dict,
    out_dir: Path,
    present_lag: int = 0,
    double_grab: dict | None = None,
) -> dict:
    """Assemble the ``source="windows-retail"`` manifest for ``v400_pixel_pair.py``.

    ``frames`` rows are ``{"logicFrame", "image" (absolute PNG path), "state"
    (the memory probe), "frameBefore", "frameAfter"}``.  The function refuses to
    publish anything that is not a proven consecutive run with a complete
    projection, so a downstream pairing can never silently skip a frame.

    With ``present_lag=n`` the grab happens at counter ``logicFrame + n`` (the
    pixels are published as the presentation of ``logicFrame``) while the memory
    probe still describes ``logicFrame``; both facts are asserted per row and
    recorded in the evidence so the shift can never be mistaken for a state match.
    """
    if not frames:
        raise ValueError("refusing to build a manifest with no frames")
    logic_frames = [int(row["logicFrame"]) for row in frames]
    if len(set(logic_frames)) != len(logic_frames):
        raise ValueError(f"duplicate logicFrame values: {logic_frames}")
    if logic_frames != sorted(logic_frames):
        raise ValueError(f"logicFrame values are not increasing: {logic_frames}")
    if logic_frames != list(range(logic_frames[0], logic_frames[0] + len(logic_frames))):
        raise ValueError(
            f"logicFrames are not consecutive: {logic_frames[0]}..{logic_frames[-1]} "
            f"with {len(logic_frames)} rows"
        )
    lag = int(present_lag)
    root = Path(out_dir).resolve()
    series: list[dict] | None = None
    if double_grab is not None:
        series = list(double_grab.get("series") or [])
        if len(series) != len(frames):
            raise ValueError(
                f"double-grab evidence covers {len(series)} frames but {len(frames)} "
                "were captured"
            )
        if lag != 0:
            raise ValueError(
                "double-grab publishing cannot be combined with a non-zero present "
                "lag: the series is filed under the counter it was grabbed at"
            )
    rows: list[dict] = []
    for index, row in enumerate(frames):
        frame = int(row["logicFrame"])
        before = int(row.get("frameBefore", frame))
        after = int(row.get("frameAfter", frame))
        if before != after:
            raise ValueError(f"frame {frame}: the BitBlt bracket did not prove the same frame")
        if before != frame + lag:
            raise ValueError(
                f"frame {frame}: grab counter {before} is not logicFrame+presentLag "
                f"({frame}+{lag})"
            )
        if int(row.get("stateFrame", frame)) != frame:
            raise ValueError(
                f"frame {frame}: the memory probe describes counter "
                f"{row.get('stateFrame')} instead of the labelled frame"
            )
        projection = retail_state_projection(row["state"])
        png = Path(row["image"]).resolve()
        try:
            relative = png.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"frame {frame}: image {png} is outside {root}") from exc
        if not png.is_file():
            raise ValueError(f"frame {frame}: missing PNG {png}")
        memory_keys = (
            "gameplayFrame",
            "stage",
            "flags",
            "replayMode",
            "currentSpell",
            "playerState",
            "playerX",
            "playerY",
            "stageFrames",
            "runFrames",
        )
        evidence = {
            "algorithm": "retail-memory-projection/1: absolute-address "
            "ReadProcessMemory of g_GameManager and its globals pointer "
            "(no write access is ever requested)",
            "fields": {name: projection[name]["value"] for name in PROJECTION_FIELDS},
            "field_bits": {name: projection[name]["bits"] for name in PROJECTION_FIELDS},
            "frameCounterBefore": before,
            "frameCounterAfter": after,
            "stateFrameCounter": int(row.get("stateFrame", frame)),
            "presentLagFrames": lag,
            "sameFrameProven": True,
            "memory": {key: row["state"].get(key) for key in memory_keys},
        }
        if series is not None:
            frame_series = series[index]
            published = str(frame_series.get("publishedSha256") or "")
            grabbed = {str(item["sha256"]) for item in frame_series.get("grabs", [])}
            if published not in grabbed:
                raise ValueError(
                    f"frame {frame}: published pixel digest {published} is not one of "
                    f"the {len(grabbed)} grabbed digests; the PNG does not come from "
                    "the recorded series"
                )
            evidence["doubleGrab"] = {
                "publishedPick": frame_series.get("publishedPick"),
                "publishedIndex": frame_series.get("publishedIndex"),
                "publishedSha256": published,
                "grabCount": len(frame_series.get("grabs", [])),
                "distinctSha256": frame_series.get("distinctSha256"),
                "phaseDecidable": bool(frame_series.get("phaseDecidable")),
                "grabs": [
                    {
                        key: item.get(key)
                        for key in (
                            "index",
                            "sha256",
                            "startMs",
                            "durationMs",
                            "counterBefore",
                            "counterAfter",
                            "sameFrameProven",
                        )
                    }
                    for item in frame_series.get("grabs", [])
                ],
            }
        rows.append(
            {
                "logicFrame": frame,
                "image": relative,
                "image_sha256": sha256_file(png),
                "state_sha256": retail_state_digest(frame, projection),
                "state_evidence": evidence,
            }
        )
    capture_block = {
        "sameFrameProof": "gameplayFrameCounter (g_GameManager+0x3DDC0) is read "
        "before and after the BitBlt and has to be identical",
        "frameFence": "absent: the retail image has no frame fence, so the "
        "counter bracket is the only same-frame proof available",
        "pngMode": "RGBA 640x480; alpha is synthesized opaque because GDI "
        "BitBlt has no alpha channel",
        "stateDigestAlgorithm": "sha256 of canonical JSON "
        "{algorithm, logicFrame, projection bits} (retail-side only, not "
        "comparable with the Linux state_sha256)",
        "projectionFields": list(PROJECTION_FIELDS),
        "projectionFloatFields": list(PROJECTION_FLOAT_FIELDS),
        "presentLagFrames": lag,
        "presentLagNote": (
            "logicFrame = frameCounter - presentLag; the pixels were grabbed at "
            "frameCounterBefore and the memory probe was read at logicFrame, so "
            "the projection always describes the labelled frame"
            if lag
            else "no present lag: pixels and memory both taken at the labelled frame"
        ),
    }
    if double_grab is not None and series is not None:
        decidable = sum(1 for item in series if item.get("phaseDecidable"))
        capture_block["doubleGrab"] = {
            key: double_grab.get(key)
            for key in ("count", "intervalMs", "untilChange", "pick", "maxPerFrame")
        } | {
            "frames": len(series),
            "framesPhaseDecidable": decidable,
            "framesIdentical": len(series) - decidable,
            "phaseDecidableFraction": round(decidable / len(series), 4) if series else 0.0,
            "pickRule": "the published grab is chosen by a single rule applied to every "
            "frame (first or last of that frame's series); no frame is chosen by "
            "looking at the Linux reference",
            "phaseEvidence": "two or more grabs taken while gameplayFrameCounter read "
            "the same value differ in pixels => a present happened inside the frame "
            "window => the last grab is the post-present content",
        }
    return {
        "schema": "v400-wp8-retail-frame-manifest/1",
        "source": "windows-retail",
        "role": "ORIGINAL retail th08.exe 1.00d playing the replay through its own "
        "menu; frames are GDI BitBlt grabs bracketed by two gameplayFrameCounter reads",
        "replay_sha256": replay_sha256,
        "replay_file": str(replay_path),
        "replay_stage": int(stage),
        "retail_image": {
            key: retail_image.get(key)
            for key in ("path", "size", "sha256", "verified")
        },
        "capture": capture_block,
        "frames": rows,
    }


def write_retail_manifest(path: Path, manifest: dict) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "frames": len(manifest["frames"]),
        "source": manifest["source"],
    }


# ---------------------------------------------------------------------------
# run directory assembly
# ---------------------------------------------------------------------------
def assemble_run_dir(
    run_dir: Path, asset_dir: Path = ASSET_DIR, force: bool = False, skip: tuple[str, ...] = ()
) -> dict:
    """Copy the retail image plus its assets into a private directory.

    Deliberately ``shutil.copy2`` and never ``os.link``: a hard link would make a
    stray write from the game visible in ``third_party/th08``, which this task
    forbids.  The two read-only assets are the only large files (496 MB total).
    """
    from touhou_rl import config_file, spec

    run_dir = Path(run_dir).resolve()
    asset_dir = Path(asset_dir).resolve()
    if run_dir.exists() and force:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "replay").mkdir(exist_ok=True)

    copied: list[dict] = []
    for entry in plan_assets(asset_dir):
        if entry["name"] in skip:
            copied.append({**entry, "action": "skipped"})
            continue
        source = Path(entry["source"])
        target = run_dir / entry["name"]
        if not source.is_file():
            raise FileNotFoundError(f"missing source asset {source}")
        if target.exists() and target.stat().st_size == source.stat().st_size:
            copied.append({**entry, "target": str(target), "action": "reused"})
            continue
        if target.exists():
            target.unlink()
        shutil.copy2(source, target)
        copied.append(
            {
                **entry,
                "target": str(target),
                "action": "copied",
                "bytes": target.stat().st_size,
            }
        )

    cfg_path = run_dir / "th08.cfg"
    # SendInput, not PostMessage: the menu only reacts to a foreground window, and
    # the retail build reads DirectInput/GetKeyboardState (EXP-P0-053).
    config_file.prepare_config(
        cfg_path,
        windowed=True,
        default_difficulty=spec.NORMAL,
        use_postmessage_input=False,
    )
    return {
        "runDir": str(run_dir),
        "files": copied,
        "cfg": config_file.GameConfig.load(cfg_path).raw.hex(),
    }


# ---------------------------------------------------------------------------
# retail memory probe
# ---------------------------------------------------------------------------
def read_retail_probe(mem, spec) -> dict:  # noqa: ANN001 - duck-typed ProcessMemory
    """One read-only snapshot of the retail globals this experiment needs.

    Every field is read by *absolute* address from the address table, never by an
    assumed relative distance: the retail input globals sit 4 bytes apart while the
    reconstruction declares ``last+4 == cur`` (8 bytes), so the relative read in
    ``GameStateReader`` is invalid for retail and is not used here.
    """
    out: dict = {}
    gm = spec.ADDR_GAME_MANAGER
    sup = spec.ADDR_SUPERVISOR

    def guard(key: str, fn) -> None:
        try:
            out[key] = fn()
        except OSError as exc:
            out[f"{key}Error"] = str(exc)

    guard("supervisorWanted", lambda: mem.i32(sup + spec.OFF_SUP_WANTED_STATE))
    guard("supervisorState", lambda: mem.i32(sup + spec.OFF_SUP_CUR_STATE))
    guard("gameplayFrame", lambda: mem.u32(gm + spec.OFF_GM_GAMEPLAY_FRAME))
    guard("stage", lambda: mem.i32(gm + spec.OFF_GM_CURRENT_STAGE))
    guard("flags", lambda: mem.u32(gm + spec.OFF_GM_FLAGS))
    guard("replayMode", lambda: mem.u8(gm + spec.OFF_GM_REPLAY_MODE))
    guard("inGameMenu", lambda: mem.u8(gm + spec.OFF_GM_IS_IN_GAME_MENU))
    guard(
        "currentSpell",
        lambda: struct.unpack("<h", mem.read(gm + spec.OFF_GM_CURRENT_SPELL, 2))[0],
    )
    guard("stageFrames", lambda: mem.u32(gm + spec.OFF_GM_STAGE_ACTIVE_FRAMES))
    guard("runFrames", lambda: mem.u32(gm + spec.OFF_GM_RUN_ACTIVE_FRAMES))

    def globals_fields() -> dict:
        pointer = mem.u32(gm + spec.OFF_GM_GLOBALS)
        fields = {"globalsPointer": pointer}
        if pointer:
            raw = mem.read(pointer, spec.GLOBALS_READ_SIZE)
            fields.update(
                score=struct.unpack_from("<I", raw, spec.OFF_GLOBALS_SCORE)[0],
                graze=struct.unpack_from("<i", raw, spec.OFF_GLOBALS_GRAZE)[0],
                deathsF32=struct.unpack_from("<f", raw, spec.OFF_GLOBALS_DEATHS)[0],
                livesF32=struct.unpack_from("<f", raw, spec.OFF_GLOBALS_LIVES)[0],
                bombsF32=struct.unpack_from("<f", raw, spec.OFF_GLOBALS_BOMBS)[0],
                bombsUsedF32=struct.unpack_from("<f", raw, spec.OFF_GLOBALS_BOMBS_USED)[0],
                powerF32=struct.unpack_from("<f", raw, spec.OFF_GLOBALS_POWER)[0],
                clockTime=struct.unpack_from("<b", raw, spec.OFF_GLOBALS_CLOCK_TIME)[0],
                timeOrbs=struct.unpack_from("<i", raw, spec.OFF_GLOBALS_TIME_ORBS)[0],
            )
        return fields

    try:
        out.update(globals_fields())
    except OSError as exc:
        out["globalsError"] = str(exc)

    guard("playerState", lambda: mem.i8(spec.ADDR_PLAYER))

    def player_pos() -> dict:
        x, y, _z = struct.unpack("<fff", mem.read(spec.ADDR_PLAYER + spec.OFF_PLAYER_POSITION, 12))
        return {"playerX": x, "playerY": y}

    try:
        out.update(player_pos())
    except OSError as exc:
        out["playerError"] = str(exc)

    # Title screen: g_TitleScreen is a *pointer* global and the object is allocated
    # and freed on every screen change, so a torn read is expected, not fatal.
    def title_fields() -> dict:
        pointer = mem.u32(spec.ADDR_TITLE_SCREEN)
        if not pointer:
            return {"titlePointer": 0}
        tail = mem.read(pointer + spec.OFF_TITLE_CURRENT_SCREEN, 0x10)
        head = mem.read(pointer, spec.OFF_TITLE_PRACTICE_STATE + 4)
        replay = mem.read(pointer + spec.OFF_TITLE_REPLAY_COUNT - 4, 0x10)
        return {
            "titlePointer": pointer,
            "titleScreenCurrent": struct.unpack_from("<i", tail, 0)[0],
            "titleTimer2": struct.unpack_from("<i", tail, 4)[0],
            "titleCursor": struct.unpack_from("<i", head, spec.OFF_TITLE_CURSOR)[0],
            "titleScreenState": struct.unpack_from("<i", head, spec.OFF_TITLE_SCREEN_STATE)[0],
            "titlePracticeState": struct.unpack_from("<i", head, spec.OFF_TITLE_PRACTICE_STATE)[0],
            "titleReplayCount": struct.unpack_from("<i", replay, 4)[0],
            "titleSelectedReplay": struct.unpack_from("<i", replay, 8)[0],
            "titleSelectedReplayStage": struct.unpack_from("<i", replay, 0xC)[0],
        }

    try:
        out.update(title_fields())
    except OSError as exc:
        out["titleError"] = str(exc)

    # Input globals, each by its own absolute retail address.
    for key, address in (
        ("curFrameInput", spec.ADDR_CUR_FRAME_INPUT),
        ("lastFrameInput", spec.ADDR_LAST_FRAME_INPUT),
        ("guiInputCurrent", spec.ADDR_GUI_INPUT_CURRENT),
        ("guiInputPrevious", spec.ADDR_GUI_INPUT_PREVIOUS),
    ):
        guard(key, lambda address=address: mem.u16(address))

    return out


def scan_advancing_u32(mem, spec, wait_s: float = 0.5, size: int = 0x3DE30) -> dict:
    """Find the ``u32`` inside ``g_GameManager`` that advances once per logic frame.

    This is the independent check of ``OFF_GM_GAMEPLAY_FRAME``: instead of trusting
    the reconstruction's ``C_ASSERT`` offset on a retail image, take two snapshots
    ``wait_s`` apart and report every offset whose ``u32`` grew by
    ``wait_s * 60 +/- 12``.  ``OFF_GM_GAMEPLAY_FRAME`` must be among them.
    """
    gm = spec.ADDR_GAME_MANAGER
    first = mem.read(gm, size)
    time.sleep(wait_s)
    second = mem.read(gm, size)
    expected = wait_s * 60.0
    hits: list[dict] = []
    for offset in range(0, size - 3, 4):
        a = struct.unpack_from("<I", first, offset)[0]
        b = struct.unpack_from("<I", second, offset)[0]
        delta = b - a
        if abs(delta - expected) <= 12:
            hits.append({"offset": hex(offset), "before": a, "after": b, "delta": delta})
    return {
        "waitSeconds": wait_s,
        "expectedDelta": expected,
        "bucketBytes": size,
        "hits": hits,
        "gameplayFrameOffset": hex(spec.OFF_GM_GAMEPLAY_FRAME),
        "gameplayFrameOffsetHit": any(
            hit["offset"] == hex(spec.OFF_GM_GAMEPLAY_FRAME) for hit in hits
        ),
    }


def scan_supervisor_states(mem, spec, size: int = 0x400) -> list[dict]:
    """Candidate ``(offset, i32)`` pairs inside ``g_Supervisor`` in state range.

    Fallback diagnostic for the case where the retail ``Supervisor`` layout differs
    from the reconstruction: a title-screen state is ``1`` and the wanted state is
    ``1`` too, so a correct offset shows up as two adjacent in-range dwords.
    """
    raw = mem.read(spec.ADDR_SUPERVISOR, size)
    values = struct.unpack_from(f"<{size // 4}i", raw, 0)
    candidates: list[dict] = []
    for index, value in enumerate(values):
        if -1 <= value <= 13:
            candidates.append({"offset": hex(index * 4), "value": value})
    return candidates


class RetailStateReader:
    """``GameState``-compatible reader that only trusts retail absolute addresses.

    ``touhou_rl.state.GameStateReader`` derives the second half of several global
    pairs from a *relative* distance (``g_GuiMessageInputPrevious - g_GuiMessageInputCurrent``,
    ``g_CurFrameInput - g_LastFrameInput``).  In the retail image those pairs are
    8 bytes apart while the reconstruction declares 4, so the relative read runs off
    the 8-byte buffer (``struct.error``) and no state can be read at all.

    This shim reads every global by its own address from the retail table and fills
    the same :class:`touhou_rl.state.GameState` dataclass, which is all ``menu``
    needs.  It is deliberately read-only and has no other behaviour.
    """

    def __init__(self, mem, spec_module):  # noqa: ANN001 - duck-typed
        self.mem = mem
        self.spec = spec_module
        self.reads = 0

    def read_state(self, with_bullets: bool = True, with_title: bool = True):  # noqa: ANN201
        from touhou_rl.state import GameState

        spec = self.spec
        probe = read_retail_probe(self.mem, spec)
        self.reads += 1
        state = GameState()
        state.bullet_capacity = spec.BULLET_COUNT
        state.supervisor_state = probe.get("supervisorState", 0)
        flags = probe.get("flags") or 0
        state.flags = flags
        state.is_replay = bool(flags & spec.FLAG_REPLAY)
        state.is_demo = bool(flags & spec.FLAG_DEMO)
        state.is_practice = bool(flags & spec.FLAG_PRACTICE)
        state.is_spell_practice = bool(flags & spec.FLAG_SPELL_PRACTICE)
        state.replay_mode = probe.get("replayMode") or 0
        state.in_game_menu = bool(probe.get("inGameMenu"))
        state.gameplay_frame = probe.get("gameplayFrame") or 0
        state.stage = probe.get("stage") or 0
        state.stage_frames = probe.get("stageFrames") or 0
        state.run_frames = probe.get("runFrames") or 0
        state.current_spell = probe.get("currentSpell", -1)
        state.score = probe.get("score") or 0
        state.graze = probe.get("graze") or 0
        state.deaths = int(round(probe.get("deathsF32") or 0.0))
        state.lives = probe.get("livesF32") or 0.0
        state.bombs = probe.get("bombsF32") or 0.0
        state.bombs_used = int(round(probe.get("bombsUsedF32") or 0.0))
        state.power = probe.get("powerF32") or 0.0
        state.player_state = probe.get("playerState", -1)
        state.player_x = probe.get("playerX") or 0.0
        state.player_y = probe.get("playerY") or 0.0
        state.input_mask = probe.get("curFrameInput") or 0
        state.last_input_mask = probe.get("lastFrameInput") or 0
        state.gui_input_current = probe.get("guiInputCurrent") or 0
        state.gui_input_previous = probe.get("guiInputPrevious") or 0
        state.title_cursor = probe.get("titleCursor", -1)
        state.title_screen_state = probe.get("titleScreenState", -1)
        state.title_screen_current = probe.get("titleScreenCurrent", -1)
        state.title_timer2 = probe.get("titleTimer2", -1)
        state.title_practice_state = probe.get("titlePracticeState", -1)
        state.title_replay_count = probe.get("titleReplayCount") or 0
        return state


def query_crash_events(max_events: int = 3) -> dict:
    """Last ``Application Error`` (id 1000) rows, for the crash post-mortem.

    The retail image crashes with an access violation whose *offset* is the only
    usable stack information we have; Windows records it here.  Read-only, bounded,
    and its failure never affects the experiment.
    """
    script = (
        "Get-WinEvent -FilterHashtable @{LogName='Application'; Id=1000} "
        f"-MaxEvents {int(max_events)} -ErrorAction SilentlyContinue | "
        "Select-Object @{n='time';e={$_.TimeCreated.ToString('o')}},Message | ConvertTo-Json -Depth 3"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            timeout=25,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - host dependent
        return {"error": f"{type(exc).__name__}: {exc}"}
    text = (completed.stdout or b"").decode("utf-8", "replace").strip()
    if not text:
        return {
            "error": "no Application Error events",
            "stderr": (completed.stderr or b"").decode("utf-8", "replace")[:400],
        }
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text[:2000]}
    if isinstance(rows, dict):
        rows = [rows]
    return {"events": rows[:max_events]}



# ---------------------------------------------------------------------------
# launch helper (mirrors GameSession._launch_and_boot with diagnostics)
# ---------------------------------------------------------------------------
class LaunchError(RuntimeError):
    def __init__(self, message: str, exit_code: int):
        super().__init__(message)
        self.exit_code = exit_code


def _wait_stable_window(session, deadline: float) -> tuple[int, tuple[int, int, int, int]]:
    from touhou_rl import winapi

    hwnd = 0
    rect: tuple[int, int, int, int] | None = None
    stable = 0
    while time.perf_counter() < deadline:
        if session.process is not None and session.process.poll() is not None:
            raise LaunchError(
                f"th08.exe exited before a window appeared (code {session.process.returncode})",
                EXIT_PROCESS_DIED,
            )
        candidate = winapi.find_window(session.pid)
        if candidate:
            current = winapi.client_rect(candidate)
            if candidate == hwnd and current == rect and current[2] > 0 and current[3] > 0:
                stable += 1
                if stable >= 2:
                    return hwnd, current
            else:
                stable = 0
            hwnd, rect = candidate, current
        time.sleep(0.1)
    raise LaunchError(
        f"no stable window for pid {session.pid} (last hwnd=0x{hwnd:X} rect={rect})", EXIT_TIMEOUT
    )


def launch_retail(session, report: dict, deadline: float) -> dict:
    """Start the retail exe in ``session.run_dir`` and attach all readers."""
    from touhou_rl import spec as _spec
    from touhou_rl import winapi
    from touhou_rl.memory import ProcessMemory

    exe = Path(session.run_dir) / "th08.exe"
    if not exe.is_file():
        raise LaunchError(f"missing {exe}", EXIT_USAGE)
    winapi.dpi_aware()
    env = dict(os.environ)
    env["__COMPAT_LAYER"] = "HIGHDPIAWARE"  # the engine is DPI-unaware
    env.pop("PYTHONPATH", None)
    started = time.perf_counter()
    session.process = subprocess.Popen(
        [str(exe)],
        cwd=str(session.run_dir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    session.pid = session.process.pid
    report["events"].append({"t": round(time.perf_counter() - started, 3), "event": "process_started", "pid": session.pid})

    hwnd, rect = _wait_stable_window(session, min(deadline, time.perf_counter() + 30.0))
    session.hwnd = hwnd
    session.client_size = (rect[2], rect[3])
    title = winapi.window_title(hwnd)
    report["window"] = {
        "hwnd": hex(hwnd),
        "class": winapi.class_name(hwnd),
        "title": title,
        "titleUtf8Hex": title.encode("utf-8", "backslashreplace").hex(),
        "client": [rect[2], rect[3]],
        "screenOrigin": [rect[0], rect[1]],
        "visible": bool(winapi.user32.IsWindowVisible(hwnd)),
        "titleSeconds": round(time.perf_counter() - started, 3),
    }

    session.mem = ProcessMemory(session.pid)
    report["moduleBase"] = hex(session.mem.module_base)
    report["moduleBaseIsImageBase"] = session.mem.module_base == 0x00400000
    session.reader = RetailStateReader(session.mem, _spec)
    session.capture = winapi.GdiCapture(rect[2], rect[3])
    session.keyboard = winapi.SendInputKeyboard()
    report["events"].append({"t": round(time.perf_counter() - started, 3), "event": "readers_attached"})
    return {"started": started, "exe": str(exe)}


def wait_title(session, report: dict, deadline: float, spec, timeout: float) -> bool:
    """Poll the retail supervisor state until the title screen is up."""
    from touhou_rl import winapi

    end = min(deadline, time.perf_counter() + timeout)
    observed: list[dict] = []
    ok = False
    while time.perf_counter() < end:
        if session.process.poll() is not None:
            report["result"]["processExitedDuringBoot"] = session.process.returncode
            break
        probe = read_retail_probe(session.mem, spec)
        observed.append(
            {
                "t": round(time.perf_counter() - report["_t0"], 3),
                "supervisorState": probe.get("supervisorState"),
                "supervisorWanted": probe.get("supervisorWanted"),
                "gameplayFrame": probe.get("gameplayFrame"),
            }
        )
        if probe.get("supervisorState") == spec.STATE_TITLE_SCREEN:
            ok = True
            break
        dialog = winapi.find_dialog(session.pid)
        if dialog:
            session.dialog = dialog
            report["result"]["startupDialog"] = winapi.dialog_text(dialog)
            break
        time.sleep(0.05)
    # Keep the evidence small but honest: the value sequence is what proves the read.
    report["supervisorPoll"] = {
        "reads": len(observed),
        "first": observed[:6],
        "last": observed[-6:] if len(observed) > 6 else [],
        "distinctStates": sorted({row["supervisorState"] for row in observed if row["supervisorState"] is not None}),
        "titleScreenSeen": ok,
    }
    if not ok and "startupDialog" not in report["result"]:
        try:
            report["supervisorCandidates"] = scan_supervisor_states(session.mem, spec)
            report["supervisorVerdict"] = (
                "expected offsets 0x158/0x15C did not report STATE_TITLE_SCREEN; "
                "the candidate table lists every in-range i32 inside the first 0x400 bytes of g_Supervisor"
            )
        except OSError as exc:  # pragma: no cover - defensive
            report["supervisorCandidatesError"] = str(exc)
    return ok


# ---------------------------------------------------------------------------
# navigation
# ---------------------------------------------------------------------------
def navigate_to_replay(session, report: dict, spec, replay_path: Path, stage: int, mode: str) -> dict:
    """Title -> Replay playback, closed-loop first and blind taps as a fallback."""
    from touhou_rl import menu, winapi

    menu.install_replay(session.run_dir, replay_path, index=1)
    result: dict = {"mode": mode, "replay": str(replay_path), "stage": stage, "attempts": []}

    if mode in ("auto", "closed"):
        try:
            menu.enter_replay(session, replay_index=0, stage=stage, timeout=20.0)
            result["path"] = "closed_loop"
            result["attempts"].append({"path": "closed_loop", "ok": True})
            return result
        except Exception as exc:  # noqa: BLE001 - the fallback is the point
            result["attempts"].append(
                {"path": "closed_loop", "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
            if mode == "closed":
                raise
            try:
                session.keyboard.release_all()
            except Exception:  # noqa: BLE001
                pass

    # Blind fallback: 4 x Down reaches StartMenu item 4 (Replay); the replay list has
    # one entry (the installed th8_01.rpy) and stage/mode default to the first entry,
    # so four Z presses walk list -> stage -> playback mode -> play.
    sequence = [("down", 0.22), ("down", 0.22), ("down", 0.22), ("down", 0.22),
                ("z", 1.0), ("z", 1.0), ("z", 1.0), ("z", 1.2)]
    for key, settle in sequence:
        menu.tap(session, key, settle_s=settle)
    result["path"] = "blind"
    result["attempts"].append({"path": "blind", "ok": True, "sequence": sequence})
    return result


def wait_playback(session, report: dict, spec, deadline: float, timeout: float) -> bool:
    """Wait until a replay is actually running (supervisor == GameManager + FLAG_REPLAY)."""
    end = min(deadline, time.perf_counter() + timeout)
    observed: list[dict] = []
    ok = False
    while time.perf_counter() < end:
        probe = read_retail_probe(session.mem, spec)
        flags = probe.get("flags") or 0
        observed.append(
            {
                "t": round(time.perf_counter() - report["_t0"], 3),
                "supervisorState": probe.get("supervisorState"),
                "flags": flags,
                "isReplay": bool(flags & spec.FLAG_REPLAY),
                "gameplayFrame": probe.get("gameplayFrame"),
                "titleScreenCurrent": probe.get("titleScreenCurrent"),
                "stage": probe.get("stage"),
            }
        )
        if probe.get("supervisorState") == spec.STATE_GAME_MANAGER and (flags & spec.FLAG_REPLAY):
            ok = True
            break
        time.sleep(0.05)
    report["playbackPoll"] = {
        "reads": len(observed),
        "first": observed[:5],
        "last": observed[-5:] if len(observed) > 5 else [],
        "replaySeen": ok,
    }
    return ok


# ---------------------------------------------------------------------------
# screenshots
# ---------------------------------------------------------------------------
def save_png(session, path: Path, hud_path: Path | None = None) -> dict:
    """Grab the 640x480 client area, write PNG + optional HUD crop, return facts."""
    from PIL import Image

    import numpy as np

    session.refresh_window()
    image = session.capture.grab(session.hwnd)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    facts = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "mean": round(float(np.asarray(image, dtype=np.float32).mean()), 3),
        "distinctColors": int(len(np.unique(np.asarray(image).reshape(-1, 3), axis=0))),
    }
    if hud_path is not None:
        left, top, right, bottom = HUD_CROP
        hud = image[top:bottom, left:right]
        hud_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(hud).save(hud_path)
        facts["hudPath"] = str(hud_path)
        facts["hudBytes"] = hud_path.stat().st_size
    return facts


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------
def phase_idle(session, args, report: dict, deadline: float, spec) -> int:
    """Control experiment: boot, reach the title screen, then touch nothing.

    This is the ablation that tells an intrinsic crash (image/data/environment)
    apart from one caused by observation.  ``--idle-read`` adds the memory probe and
    ``--idle-grab`` adds the GDI ``BitBlt`` on the same cadence, so three runs of
    this phase separate "the retail build dies on its own" from "our read kills it"
    from "our capture kills it".
    """
    from touhou_rl import winapi

    started = launch_retail(session, report, deadline)
    report["_t0"] = started["started"]
    # The pure control arm must not even *read* the target, so the title handshake
    # only runs when memory observation is part of the arm.
    reached: bool | None = None
    if args.idle_read:
        reached = wait_title(session, report, deadline, spec, args.title_timeout)
    report["result"]["titleReached"] = reached
    if reached:
        report["title"] = save_png(session, args.out / "00-title.png", None)

    arm = {"readMemory": bool(args.idle_read), "gdiGrab": bool(args.idle_grab)}
    report["idleArm"] = arm
    timeline: list[dict] = []
    end = min(deadline, time.perf_counter() + args.idle_seconds)
    alive_at_end = False
    while time.perf_counter() < end:
        row = {"t": round(time.perf_counter() - started["started"], 3)}
        exited = session.process.poll()
        row["exitCode"] = exited
        row["window"] = winapi.find_window(session.pid)
        row["hwndIsWindow"] = bool(winapi.user32.IsWindow(session.hwnd)) if session.hwnd else False
        if arm["readMemory"]:
            probe = read_retail_probe(session.mem, spec)
            row["supervisorState"] = probe.get("supervisorState")
            row["gameplayFrame"] = probe.get("gameplayFrame")
            row["readError"] = probe.get("supervisorStateError")
        if arm["gdiGrab"]:
            try:
                image = session.capture.grab(session.hwnd)
                row["grabShape"] = list(image.shape)
                row["grabMean"] = round(float(image.mean()), 2)
            except OSError as exc:
                row["grabError"] = str(exc)
        timeline.append(row)
        if exited is not None:
            report["result"]["crashDetectedAtSeconds"] = row["t"]
            break
        alive_at_end = True
        time.sleep(0.2)
    report["idleTimeline"] = timeline
    report["result"]["idleSeconds"] = round(timeline[-1]["t"], 3) if timeline else 0.0
    report["result"]["survivedIdle"] = alive_at_end and session.process.poll() is None
    if report["result"]["survivedIdle"]:
        report["survivorScreenshot"] = save_png(session, args.out / "01-survived.png", None)
        try:
            report["finalProbe"] = read_retail_probe(session.mem, spec)
        except OSError as exc:
            report["finalProbeError"] = str(exc)
    report["status"] = "ok" if report["result"]["survivedIdle"] else "failed"
    report["result"]["exitCode"] = EXIT_OK if report["result"]["survivedIdle"] else EXIT_PROCESS_DIED
    return report["result"]["exitCode"]


def phase_launch(session, args, report: dict, deadline: float, spec) -> int:
    started = launch_retail(session, report, deadline)
    report["_t0"] = started["started"]

    if not wait_title(session, report, deadline, spec, args.title_timeout):
        report["status"] = "failed"
        report["result"]["reason"] = "title screen never reached"
        report["result"]["exitCode"] = EXIT_STATE
        return EXIT_STATE
    report["events"].append(
        {"t": round(time.perf_counter() - started["started"], 3), "event": "title_screen"}
    )
    report["title"] = save_png(session, args.out / "01-title.png", args.out / "01-title-hud.png")

    # Replay playback: the point of this phase.
    nav = navigate_to_replay(session, report, spec, args.replay, args.replay_stage, args.nav)
    report["navigation"] = nav
    report["events"].append(
        {"t": round(time.perf_counter() - started["started"], 3), "event": "replay_keys_sent", "path": nav["path"]}
    )
    playing = wait_playback(session, report, spec, deadline, args.playback_timeout)
    report["result"]["replayPlaying"] = playing
    report["events"].append(
        {"t": round(time.perf_counter() - started["started"], 3), "event": "replay_playback", "ok": playing}
    )
    if not playing:
        report["screenshotMenu"] = save_png(session, args.out / "02-menu.png", args.out / "02-menu-hud.png")
        report["status"] = "partial"
        report["result"]["reason"] = "booted to the title screen but replay playback did not start"
        report["result"]["exitCode"] = EXIT_STATE
        return EXIT_STATE

    if args.settle_seconds > 0:
        time.sleep(args.settle_seconds)
    report["result"]["playbackProbe"] = read_retail_probe(session.mem, spec)
    report["playback"] = save_png(session, args.out / "03-playback.png", args.out / "03-playback-hud.png")
    report["result"]["exitCode"] = EXIT_OK
    report["status"] = "ok"
    return EXIT_OK


def phase_memory(session, args, report: dict, deadline: float, spec) -> int:
    started = launch_retail(session, report, deadline)
    report["_t0"] = started["started"]
    if not wait_title(session, report, deadline, spec, args.title_timeout):
        report["status"] = "failed"
        report["result"]["reason"] = "title screen never reached"
        report["result"]["exitCode"] = EXIT_STATE
        return EXIT_STATE
    report["navigation"] = navigate_to_replay(session, report, spec, args.replay, args.replay_stage, args.nav)
    if not wait_playback(session, report, spec, deadline, args.playback_timeout):
        report["status"] = "partial"
        report["result"]["reason"] = "replay playback did not start"
        report["result"]["exitCode"] = EXIT_STATE
        return EXIT_STATE
    report["events"].append({"t": round(time.perf_counter() - started["started"], 3), "event": "replay_playback"})

    if args.settle_seconds > 0:
        time.sleep(args.settle_seconds)

    # Independent validation of the GameManager offset before relying on it.
    report["frameCounterScan"] = scan_advancing_u32(session.mem, spec, wait_s=0.5)

    samples: list[dict] = []
    end = min(deadline, time.perf_counter() + args.sample_seconds)
    while time.perf_counter() < end and len(samples) < args.samples:
        probe = read_retail_probe(session.mem, spec)
        probe["t"] = round(time.perf_counter() - started["started"], 4)
        samples.append(probe)
        time.sleep(args.sample_interval)
    report["samples"] = samples
    report["sampleCount"] = len(samples)

    frames = [row.get("gameplayFrame") for row in samples if isinstance(row.get("gameplayFrame"), int)]
    scores = [
        row.get("score") for row in samples
        if isinstance(row.get("score"), int)
    ]
    deaths = [
        int(round(row["deathsF32"])) for row in samples
        if isinstance(row.get("deathsF32"), float)
    ]
    problems = monotonic_problems(frames, scores, deaths)
    deltas = frame_deltas(frames)
    report["validation"] = {
        "framesRead": len(frames),
        "framesDistinct": len(set(frames)),
        "frameFirst": frames[0] if frames else None,
        "frameLast": frames[-1] if frames else None,
        "frameSpan": (frames[-1] - frames[0]) if frames else None,
        "frameDeltaMin": min(deltas) if deltas else None,
        "frameDeltaMax": max(deltas) if deltas else None,
        "scoreFirst": scores[0] if scores else None,
        "scoreLast": scores[-1] if scores else None,
        "deathsDistinct": sorted(set(deaths)),
        "livesDistinct": sorted({round(row.get("livesF32", -1.0), 3) for row in samples}),
        "powerDistinct": sorted({round(row.get("powerF32", -1.0), 2) for row in samples}),
        "monotonicProblems": problems,
        "stageSeen": sorted({row.get("stage") for row in samples if isinstance(row.get("stage"), int)}),
        "replayModeSeen": sorted({row.get("replayMode") for row in samples if isinstance(row.get("replayMode"), int)}),
    }

    ok = bool(frames) and not problems and len(set(frames)) > 1 and report["frameCounterScan"]["gameplayFrameOffsetHit"]
    report["result"]["memoryReadable"] = ok
    report["result"]["exitCode"] = EXIT_OK if ok else EXIT_STATE
    report["status"] = "ok" if ok else "failed"
    if not ok:
        if not frames:
            report["result"]["reason"] = "no gameplayFrameCounter value could be read"
        elif problems:
            report["result"]["reason"] = "; ".join(problems[:3])
        elif not report["frameCounterScan"]["gameplayFrameOffsetHit"]:
            report["result"]["reason"] = (
                "the offset scan did not find OFF_GM_GAMEPLAY_FRAME among the per-frame advancing u32s"
            )
        else:
            report["result"]["reason"] = "the frame counter did not advance"
    return report["result"]["exitCode"]


def _read_gameplay_counter(session, spec) -> int:  # noqa: ANN001 - duck-typed session
    """One absolute-address read of ``g_GameManager+0x3DDC0`` (no write access)."""
    return int(session.mem.u32(spec.ADDR_GAME_MANAGER + spec.OFF_GM_GAMEPLAY_FRAME))


def collect_grab_series(
    session,  # noqa: ANN001 - duck-typed session
    spec,
    counter: int,
    count: int,
    interval_s: float,
    until_change: bool,
    max_per_frame: int,
    deadline: float,
    probe_at=None,
) -> dict:
    """BitBlt the client area repeatedly while the counter still reads ``counter``.

    Each grab is bracketed by two counter reads.  A grab whose bracket disagrees is
    dropped: the BitBlt straddled the counter update, so the pixels cannot be filed
    under a single logic frame.  The first ``count`` grabs start ``interval_s``
    apart; with ``until_change`` the loop then keeps grabbing back-to-back until the
    counter moves (hypothesis 4: the last grab is as close to the end of the logic
    frame as the retail build allows, i.e. as late as possible before the counter
    increments).  Only the first and the last image are kept -- the middle ones are
    hashed and released so a long series cannot accumulate 100 MB of pixels.

    ``probe_at`` is called once, right after the first accepted grab: the earliest
    grab is the one furthest from the present (maximum contrast), but the state
    probe has to be late enough that the logic frame's own memory writes (score
    included) have landed, so it is taken a grab or two into the series instead of
    before it.
    """
    import numpy as np

    grabs: list[dict] = []
    images: dict[str, object] = {"first": None, "last": None}
    probe = None
    first_start: float | None = None
    stop_reason = "count"
    while len(grabs) < max(1, int(max_per_frame)) and time.perf_counter() < deadline:
        start = time.perf_counter()
        try:
            before = _read_gameplay_counter(session, spec)
            image = session.capture.grab(session.hwnd)
            after = _read_gameplay_counter(session, spec)
        except OSError as exc:
            return {
                "grabs": grabs,
                "images": images,
                "probe": probe,
                "stopReason": "grab-failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        duration = time.perf_counter() - start
        if first_start is None:
            first_start = start
        if before != after:
            # Straddled the counter update: the pixels belong to neither frame.
            stop_reason = "straddled"
            break
        if before != counter:
            stop_reason = "advanced"
            break
        array = np.asarray(image)
        is_first = not grabs
        grabs.append(
            {
                "index": len(grabs),
                "sha256": sha256_bytes(array.tobytes()),
                "startMs": round((start - first_start) * 1000.0, 4),
                "durationMs": round(duration * 1000.0, 4),
                "counterBefore": before,
                "counterAfter": after,
                "sameFrameProven": True,
            }
        )
        if is_first:
            images["first"] = image
        images["last"] = image
        if is_first and probe_at is not None:
            probe = probe_at(counter)
            if probe is None:
                return {
                    "grabs": grabs,
                    "images": images,
                    "probe": None,
                    "stopReason": "state-race",
                }
        if not until_change and len(grabs) >= max(2, int(count)):
            stop_reason = "count"
            break
        if interval_s > 0 and not until_change:
            # Pace the series to ``interval_s`` between grab *starts*.
            remaining = interval_s - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)
    else:
        stop_reason = "max-per-frame" if len(grabs) >= max(1, int(max_per_frame)) else "deadline"
    return {"grabs": grabs, "images": images, "probe": probe, "stopReason": stop_reason}


def probe_state_at_counter(
    session,  # noqa: ANN001 - duck-typed session
    spec,
    counter: int,
    attempts: int = 8,
) -> dict | None:
    """Read the full memory probe while the counter provably stays on ``counter``.

    The seven-field projection has to describe the labelled frame, so the probe is
    bracketed by two counter reads exactly like the BitBlt.  A probe that straddles
    the update is retried inside the same frame; if the frame is already gone the
    caller skips it instead of publishing a projection from another frame.
    """
    for _ in range(max(1, int(attempts))):
        try:
            before = _read_gameplay_counter(session, spec)
            probe = read_retail_probe(session.mem, spec)
            after = _read_gameplay_counter(session, spec)
        except OSError:
            return None
        if before == after == counter:
            return probe
        if before > counter or after > counter:
            return None
    return None


def phase_capture(session, args, report: dict, deadline: float, spec) -> int:
    started = launch_retail(session, report, deadline)
    report["_t0"] = started["started"]
    if not wait_title(session, report, deadline, spec, args.title_timeout):
        report["status"] = "failed"
        report["result"]["reason"] = "title screen never reached"
        report["result"]["exitCode"] = EXIT_STATE
        return EXIT_STATE
    report["navigation"] = navigate_to_replay(session, report, spec, args.replay, args.replay_stage, args.nav)
    if not wait_playback(session, report, spec, deadline, args.playback_timeout):
        report["status"] = "partial"
        report["result"]["reason"] = "replay playback did not start"
        report["result"]["exitCode"] = EXIT_STATE
        return EXIT_STATE
    report["events"].append({"t": round(time.perf_counter() - started["started"], 3), "event": "replay_playback"})
    if args.settle_seconds > 0:
        time.sleep(args.settle_seconds)

    needed = args.consecutive_frames
    lag = int(getattr(args, "present_lag", 0) or 0)
    # A published label needs two samples: the grab at ``label + lag`` and the
    # memory probe at ``label``.  Both come from the same consecutive run, so the
    # run has to be ``needed + |lag|`` samples long.
    sample_target = needed + abs(lag)
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    attempts = 0
    accepted = 0
    duplicates = 0
    skipped = 0
    buffer: list[dict] = []
    session.refresh_window()
    end = min(deadline, time.perf_counter() + args.capture_seconds)

    # Optional seek (EXP-P0-063): skip the running replay until the gameplay frame
    # counter reaches ``--start-logic-frame``, then start the consecutive run there.
    # The first captured frame is the first sampled frame >= the request and is
    # reported verbatim, so a small overshoot cannot silently shift the window: the
    # recorded range is what both sides are then asked to reproduce.
    # With ``--present-lag n`` the grab happens n frames later than the label, so the
    # seek target moves by n and the first *published* label stays the request.
    seek = None
    start = args.start_logic_frame
    counter_target = None if start is None else seek_counter_target(start, lag)
    if start is not None:
        seek_started = time.perf_counter()
        probes = 0
        first_probe = None
        last_frame = None
        while time.perf_counter() < end:
            probe = read_retail_probe(session.mem, spec)
            probes += 1
            frame = probe.get("gameplayFrame")
            if first_probe is None:
                first_probe = frame
            last_frame = frame
            if isinstance(frame, int) and frame >= counter_target:
                break
            time.sleep(0.002)
        seek = {
            "requestedStart": start,
            "presentLagFrames": lag,
            "counterTarget": counter_target,
            "expectedFirstLabel": (
                None if counter_target is None else counter_target + max(0, -lag)
            ),
            "probes": probes,
            "firstProbeFrame": first_probe,
            "lastProbeFrame": last_frame,
            "reached": isinstance(last_frame, int) and last_frame >= counter_target,
            "seconds": round(time.perf_counter() - seek_started, 3),
        }
        report["seek"] = seek
        if not seek["reached"]:
            report["status"] = "partial"
            report["result"]["reason"] = (
                f"the replay never reached frame counter {counter_target} inside the "
                f"capture window (last observed frame {last_frame}); probes={probes}"
            )
            report["result"]["exitCode"] = EXIT_STATE
            return EXIT_STATE

    double_grab = {
        "enabled": bool(getattr(args, "double_grab", False)),
        "count": int(getattr(args, "double_grab_count", DOUBLE_GRAB_DEFAULT_COUNT) or 0),
        "intervalMs": float(
            getattr(args, "double_grab_interval_ms", DOUBLE_GRAB_DEFAULT_INTERVAL_MS) or 0.0
        ),
        "untilChange": bool(getattr(args, "double_grab_until_change", False)),
        "pick": str(getattr(args, "double_grab_pick", "first") or "first"),
        "maxPerFrame": int(
            getattr(args, "double_grab_max_per_frame", DOUBLE_GRAB_MAX_PER_FRAME) or 0
        ),
    }
    interval_s = max(0.0, double_grab["intervalMs"] / 1000.0)

    while time.perf_counter() < end and len(buffer) < sample_target:
        if not double_grab["enabled"]:
            try:
                before = read_retail_probe(session.mem, spec)
                frame_before = before.get("gameplayFrame")
                image = session.capture.grab(session.hwnd)
                after = read_retail_probe(session.mem, spec)
                frame_after = after.get("gameplayFrame")
            except OSError:
                skipped += 1
                continue
            attempts += 1
            if not isinstance(frame_before, int) or frame_before != frame_after:
                skipped += 1
                continue
            accepted += 1
            if buffer and frame_before == buffer[-1]["counter"]:
                duplicates += 1
                continue
            if buffer and frame_before != buffer[-1]["counter"] + 1:
                buffer = []  # discontinuity: the run of consecutive frames restarts here
            buffer.append(
                {
                    "counter": frame_before,
                    "frameBefore": frame_before,
                    "frameAfter": frame_after,
                    "probe": after,
                    "image": image,
                    "t": round(time.perf_counter() - started["started"], 4),
                }
            )
            continue

        # -- double-grab path -------------------------------------------------
        try:
            counter = _read_gameplay_counter(session, spec)
        except OSError:
            skipped += 1
            continue
        if buffer and counter <= buffer[-1]["counter"]:
            # This frame's series is already filed; wait for the counter to move
            # instead of sampling the same logic frame twice.
            time.sleep(0.0005)
            continue
        series = collect_grab_series(
            session,
            spec,
            counter,
            double_grab["count"],
            interval_s,
            double_grab["untilChange"],
            double_grab["maxPerFrame"],
            end,
            probe_at=lambda frame: probe_state_at_counter(session, spec, frame),
        )
        attempts += 1
        # The state probe is taken inside the series (a grab or two after the
        # counter increments) because the logic frame's own memory writes land
        # after the increment: a probe at the very start raced one frame in 200
        # during the first EXP-P0-066 run and published the previous frame's score.
        probe = series.get("probe")
        grabs = series["grabs"]
        report.setdefault("doubleGrabStopReasons", {})
        reason = series["stopReason"]
        report["doubleGrabStopReasons"][reason] = report["doubleGrabStopReasons"].get(reason, 0) + 1
        if not grabs or probe is None:
            skipped += 1
            continue
        picked = pick_grab_index(grabs, double_grab["pick"])
        alt_pick = other_grab_pick(double_grab["pick"])
        alt_index = pick_grab_index(grabs, alt_pick)
        images = series["images"]
        accepted += 1
        if buffer and counter == buffer[-1]["counter"]:
            duplicates += 1
            continue
        if buffer and counter != buffer[-1]["counter"] + 1:
            buffer = []  # discontinuity: the run of consecutive frames restarts here
        summary = grab_series_summary(grabs)
        buffer.append(
            {
                "counter": counter,
                "frameBefore": counter,
                "frameAfter": counter,
                "probe": probe,
                "image": images["first"] if picked == 0 else images["last"],
                "altImage": images["last"] if alt_index == len(grabs) - 1 else images["first"],
                "grabs": grabs,
                "grabSummary": summary,
                "pickIndex": picked,
                "altPickIndex": alt_index,
                "t": round(time.perf_counter() - started["started"], 4),
            }
        )
    report["attempts"] = attempts
    report["accepted"] = accepted
    report["duplicateFrames"] = duplicates
    report["skippedAttempts"] = skipped
    report["presentLag"] = {
        "frames": lag,
        "labelRule": "logicFrame = grabFrameCounter - presentLag",
        "stateRule": "the memory probe is read while the counter equals logicFrame, "
        "so the projection describes the labelled frame, not the grab frame",
        "samplesCaptured": len(buffer),
        "sampleCounters": [row["counter"] for row in buffer],
        "note": (
            "unmodified capture: pixels and memory are taken at the same frame"
            if lag == 0
            else f"pixels grabbed {abs(lag)} frame(s) later than the frame they are "
            f"published as ({'delayed' if lag > 0 else 'advanced'} presentation)"
        ),
    }

    published, missing_labels, counter_run = align_present_lag(buffer, needed, lag)
    report["presentLag"]["publishedLabels"] = [row["frame"] for row in published]
    report["presentLag"]["missingLabels"] = missing_labels
    by_counter = {row["counter"]: row for row in buffer}
    if double_grab["enabled"]:
        for row in published:
            source = by_counter[row["frame"]]
            for key in ("grabs", "grabSummary", "pickIndex", "altPickIndex", "altImage"):
                row[key] = source[key]

    from PIL import Image

    records: list[dict] = []
    for index, row in enumerate(published):
        png = frames_dir / f"frame-{row['frame']:06d}.png"
        # RGBA is what v400_pixel_pair.py consumes on both hosts; GDI has no alpha,
        # so the channel is synthesized opaque and said so in the manifest below.
        Image.fromarray(to_rgba(row["image"]), "RGBA").save(png)
        hud = frames_dir / f"frame-{row['frame']:06d}-hud.png"
        left, top, right, bottom = HUD_CROP
        Image.fromarray(row["image"][top:bottom, left:right]).save(hud)
        record = {
            "index": index,
            "frame": row["frame"],
            "stateFrame": row["stateFrame"],
            "grabFrameCounter": row["frameBefore"],
            "frameBefore": row["frameBefore"],
            "frameAfter": row["frameAfter"],
            "sameFrameProven": row["frameBefore"] == row["frameAfter"],
            "tSeconds": row["t"],
            "png": str(png),
            "pngBytes": png.stat().st_size,
            "pngSha256": sha256_file(png),
            "hudPng": str(hud),
            "state": row["probe"],
        }
        if double_grab["enabled"]:
            alt_png = frames_dir / f"frame-{row['frame']:06d}-alt.png"
            Image.fromarray(to_rgba(row["altImage"]), "RGBA").save(alt_png)
            record["altPng"] = str(alt_png)
            record["altPngBytes"] = alt_png.stat().st_size
            record["altPngSha256"] = sha256_file(alt_png)
            record["grabSeries"] = row["grabs"]
            record["grabSummary"] = row["grabSummary"]
            record["pickIndex"] = row["pickIndex"]
            record["altPickIndex"] = row["altPickIndex"]
        records.append(record)
    report["frames"] = records
    report["result"]["consecutiveFrames"] = [row["frame"] for row in records]
    report["result"]["pngCount"] = len(records)
    report["result"]["distinctPngSha256"] = len({row["pngSha256"] for row in records})
    report["result"]["sameFrameProvenAll"] = all(row["sameFrameProven"] for row in records)
    if double_grab["enabled"]:
        summaries = [row["grabSummary"] for row in records]
        decidable = [row["frame"] for row, item in zip(records, summaries) if item["phaseDecidable"]]
        report["doubleGrab"] = {
            "enabled": True,
            "count": double_grab["count"],
            "intervalMs": double_grab["intervalMs"],
            "untilChange": double_grab["untilChange"],
            "pick": double_grab["pick"],
            "maxPerFrame": double_grab["maxPerFrame"],
            "rule": "every logic frame is grabbed ``count`` times while the counter "
            "reads it; --double-grab-until-change keeps grabbing back-to-back until "
            "the counter moves, so the last grab is as late in the frame as this "
            "build allows",
            "framesCaptured": len(records),
            "framesPhaseDecidable": len(decidable),
            "phaseDecidableFraction": round(len(decidable) / len(records), 4) if records else 0.0,
            "framesIdentical": len(records) - len(decidable),
            "grabCountHistogram": {
                str(n): sum(1 for item in summaries if item["count"] == n)
                for n in sorted({item["count"] for item in summaries})
            },
            "spanMs": {
                "min": min((item["spanMs"] for item in summaries), default=0.0),
                "mean": (
                    round(sum(item["spanMs"] for item in summaries) / len(summaries), 4)
                    if summaries
                    else 0.0
                ),
                "max": max((item["spanMs"] for item in summaries), default=0.0),
            },
            # The complete list, not a sample: a truncated list silently made the
            # cross-check in .tmp_v400_wp8_dgrab_analyze.py look like 32 misses.
            "decidableFrames": decidable,
        }
    frames = [row["frame"] for row in records]
    scores = [row["state"].get("score", 0) for row in records]
    deaths = [int(round(row["state"].get("deathsF32", 0.0))) for row in records]
    problems = monotonic_problems(frames, scores, deaths)

    # The manifest is what a pairing tool reads, so it is written even for a
    # short run (an honest 60-frame fallback beats a fabricated 200).
    double_evidence = None
    if double_grab["enabled"] and records:
        double_evidence = {
            **double_grab,
            "series": [
                {
                    **row["grabSummary"],
                    "grabs": row["grabSeries"],
                    "publishedPick": double_grab["pick"],
                    "publishedIndex": row["pickIndex"],
                    "publishedSha256": row["grabSeries"][row["pickIndex"]]["sha256"],
                }
                for row in records
            ],
        }
    if records:
        try:
            manifest = build_retail_manifest(
                [
                    {
                        "logicFrame": row["frame"],
                        "image": row["png"],
                        "state": row["state"],
                        "stateFrame": row["stateFrame"],
                        "frameBefore": row["frameBefore"],
                        "frameAfter": row["frameAfter"],
                    }
                    for row in records
                ],
                replay_path=args.replay,
                replay_sha256=sha256_file(args.replay),
                stage=args.replay_stage,
                retail_image=report["retailImage"],
                out_dir=args.out,
                present_lag=lag,
                double_grab=double_evidence,
            )
            report["retailManifest"] = write_retail_manifest(
                args.out / "windows-retail-manifest.json", manifest
            )
            if double_evidence is not None:
                alt_pick = other_grab_pick(double_grab["pick"])
                alt_evidence = {
                    **double_evidence,
                    "pick": alt_pick,
                    "series": [
                        {
                            **row["grabSummary"],
                            "grabs": row["grabSeries"],
                            "publishedPick": alt_pick,
                            "publishedIndex": row["altPickIndex"],
                            "publishedSha256": row["grabSeries"][row["altPickIndex"]]["sha256"],
                        }
                        for row in records
                    ],
                }
                alt_manifest = build_retail_manifest(
                    [
                        {
                            "logicFrame": row["frame"],
                            "image": row["altPng"],
                            "state": row["state"],
                            "stateFrame": row["stateFrame"],
                            "frameBefore": row["frameBefore"],
                            "frameAfter": row["frameAfter"],
                        }
                        for row in records
                    ],
                    replay_path=args.replay,
                    replay_sha256=manifest["replay_sha256"],
                    stage=args.replay_stage,
                    retail_image=report["retailImage"],
                    out_dir=args.out,
                    present_lag=lag,
                    double_grab=alt_evidence,
                )
                report["retailManifestAlt"] = write_retail_manifest(
                    args.out / "windows-retail-manifest-alt.json", alt_manifest
                )
        except (OSError, ValueError) as exc:
            report["retailManifestError"] = f"{type(exc).__name__}: {exc}"

    report["validation"] = {
        "consecutive": frames == list(range(frames[0], frames[0] + len(frames))) if frames else False,
        "frameDelta": frame_deltas(frames),
        "monotonicProblems": problems,
        "scoreRange": [min(scores), max(scores)] if scores else None,
        "startLogicFrameRequested": args.start_logic_frame,
        "firstCapturedFrame": frames[0] if frames else None,
        "startRequestHonoured": (
            args.start_logic_frame is None
            or (bool(frames) and frames[0] >= args.start_logic_frame)
        ),
        "presentLagFrames": lag,
        "presentLagLabelRule": "logicFrame = grabFrameCounter - presentLag",
        "firstGrabFrameCounter": records[0]["grabFrameCounter"] if records else None,
        "lastGrabFrameCounter": records[-1]["grabFrameCounter"] if records else None,
        "grabCounterSpan": [counter_run[0], counter_run[-1]] if counter_run else None,
        "missingPresentLabels": missing_labels,
        "doubleGrab": (
            {
                "pick": double_grab["pick"],
                "altPick": other_grab_pick(double_grab["pick"]),
                "framesPhaseDecidable": report["doubleGrab"]["framesPhaseDecidable"],
                "phaseDecidableFraction": report["doubleGrab"]["phaseDecidableFraction"],
                "framesIdentical": report["doubleGrab"]["framesIdentical"],
                "altManifest": report.get("retailManifestAlt"),
            }
            if double_grab["enabled"]
            else None
        ),
    }
    if len(records) < needed:
        report["status"] = "partial"
        report["result"]["reason"] = (
            f"only {len(records)} consecutive logic frames captured (wanted {needed}); "
            f"attempts={attempts} accepted={accepted} skipped={skipped} "
            f"missingLabels={len(missing_labels)}"
        )
        report["result"]["exitCode"] = EXIT_STATE
        return EXIT_STATE
    ok = (
        report["result"]["sameFrameProvenAll"]
        and not problems
        and report["validation"]["consecutive"]
        and report["validation"]["startRequestHonoured"]
        and not missing_labels
    )
    report["result"]["exitCode"] = EXIT_OK if ok else EXIT_STATE
    report["status"] = "ok" if ok else "failed"
    if not ok:
        report["result"]["reason"] = "the captured frames are not a proven consecutive run"
    return report["result"]["exitCode"]


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def _check_single_instance(report: dict) -> int | None:
    """Refuse to run while any th08.exe exists (the engine takes a global mutex)."""
    from touhou_rl import winapi

    others = winapi.list_process_ids("th08.exe")
    report["otherInstances"] = others
    if others:
        report["status"] = "blocked"
        report["result"]["reason"] = f"another th08.exe is running: {others}"
        report["result"]["exitCode"] = EXIT_INSTANCE
        return EXIT_INSTANCE
    return None


def _hash_dir_snapshot(paths: list[Path]) -> dict:
    return {str(path): sha256_file(path) for path in paths if path.is_file()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("v400-retail-win-capture", description=__doc__)
    parser.add_argument("--phase", choices=["launch", "memory", "capture", "idle"], required=True)
    parser.add_argument("--out", type=Path, required=True, help="evidence directory (must be new or empty)")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR)
    parser.add_argument("--addresses", type=Path, default=DEFAULT_ADDRESSES)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--replay-stage", type=int, default=0)
    parser.add_argument("--nav", choices=["auto", "closed", "blind"], default="auto")
    parser.add_argument("--budget-seconds", type=float, default=170.0)
    parser.add_argument("--title-timeout", type=float, default=40.0)
    parser.add_argument("--playback-timeout", type=float, default=45.0)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--sample-seconds", type=float, default=4.0)
    parser.add_argument("--sample-interval", type=float, default=0.02)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--capture-seconds", type=float, default=40.0)
    parser.add_argument("--consecutive-frames", type=int, default=20)
    parser.add_argument(
        "--start-logic-frame",
        type=int,
        default=None,
        help="skip the running replay until gameplayFrameCounter reaches this value, "
        "then capture the consecutive run from there (default: first playable frame)",
    )
    parser.add_argument(
        "--present-lag",
        type=int,
        default=0,
        choices=range(-3, 4),
        metavar="{n}",
        help="publish the pixels grabbed while the frame counter reads C as logic "
        "frame C-n (n in [-3,3]; default 0). The memory probe for that logic frame "
        "is taken at C-n, so the seven-field projection stays honest. Negative n "
        "files a grab ahead of the frame it is compared with.",
    )
    parser.add_argument(
        "--double-grab",
        action="store_true",
        help="capture phase: grab every logic frame several times (see "
        "--double-grab-count/--double-grab-interval-ms) while the counter still reads "
        "that frame, hash each grab, and publish the first or the last one. Two "
        "grabs whose pixels differ prove a present happened inside the frame, which "
        "is the phase evidence the counter bracket alone cannot give",
    )
    parser.add_argument(
        "--double-grab-count",
        type=int,
        default=DOUBLE_GRAB_DEFAULT_COUNT,
        help="grabs per logic frame (>=2; default 2)",
    )
    parser.add_argument(
        "--double-grab-interval-ms",
        type=float,
        default=DOUBLE_GRAB_DEFAULT_INTERVAL_MS,
        help="milliseconds between the starts of the first --double-grab-count grabs "
        "(default 4.0; 0 grabs back to back)",
    )
    parser.add_argument(
        "--double-grab-until-change",
        action="store_true",
        help="after the counted grabs keep BitBlt-ing back to back until the counter "
        "moves, so the last grab of the frame is as late before the counter "
        "increment as this build allows (hypothesis 4)",
    )
    parser.add_argument(
        "--double-grab-pick",
        choices=DOUBLE_GRAB_PICKS,
        default="first",
        help="which grab of the series is published as the frame's pixels (default "
        "first); the other one is published as a second manifest so both can be "
        "paired against the same Linux frames",
    )
    parser.add_argument(
        "--double-grab-max-per-frame",
        type=int,
        default=DOUBLE_GRAB_MAX_PER_FRAME,
        help=f"hard cap on grabs per frame (default {DOUBLE_GRAB_MAX_PER_FRAME})",
    )
    parser.add_argument("--idle-seconds", type=float, default=15.0)
    parser.add_argument("--idle-read", action="store_true", help="idle phase: also probe memory each tick")
    parser.add_argument("--idle-grab", action="store_true", help="idle phase: also BitBlt the client area each tick")
    parser.add_argument("--no-score-dat", action="store_true", help="do not seed the run dir with the template score.dat")
    parser.add_argument("--force-run-dir", action="store_true")
    args = parser.parse_args(argv)

    if args.double_grab:
        if args.present_lag:
            print(
                "refusing --double-grab with --present-lag: the series is filed under "
                "the counter it was grabbed at, so a relabelling would publish pixels "
                "under a frame they were not proven to belong to",
                file=sys.stderr,
            )
            return EXIT_USAGE
        if args.double_grab_count < 2:
            print("refusing --double-grab-count below 2: one grab is not a series", file=sys.stderr)
            return EXIT_USAGE
        if args.double_grab_interval_ms < 0:
            print("refusing a negative --double-grab-interval-ms", file=sys.stderr)
            return EXIT_USAGE
        if args.double_grab_max_per_frame < args.double_grab_count:
            print(
                "refusing --double-grab-max-per-frame below --double-grab-count",
                file=sys.stderr,
            )
            return EXIT_USAGE

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"refusing to reuse non-empty {out}; choose a new --out", file=sys.stderr)
        return EXIT_USAGE
    out.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "schema": "v400-retail-win-capture/1",
        "phase": args.phase,
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "running",
        "arguments": {key: str(value) for key, value in vars(args).items()},
        "events": [],
        "result": {},
        "limitations": [
            "no OCR: the HUD crop is saved for a human to compare with the memory values",
            "one launch per phase; the engine takes a session-wide mutex so phases can never overlap",
            "the retail build has no frame-fence translation unit, so 'same frame' is proven by "
            "gameplayFrameCounter read before and after the BitBlt, not by holding the logic frame",
        ],
    }
    if args.present_lag:
        report["limitations"].append(
            f"--present-lag {args.present_lag}: the PNG filed under logicFrame F was grabbed "
            f"while the counter read F{args.present_lag:+d}; this is a relabelling of the grab, "
            "not a change to the game, and it is only meaningful if the retail present lags the "
            "logic frame by that constant"
        )
    if args.double_grab:
        report["limitations"].append(
            "--double-grab: every grab of a series is bracketed by the same counter value, so "
            "the series is filed under that logic frame; a frame whose grabs are all identical "
            "has no present evidence and keeps the 1-frame presentation ambiguity of EXP-P0-065"
        )
    t0 = time.perf_counter()
    report["_t0"] = t0
    deadline = t0 + args.budget_seconds
    report["budgetSeconds"] = args.budget_seconds

    identity = retail_identity(RETAIL_EXE)
    report["retailImage"] = identity
    if not identity["verified"]:
        report["status"] = "blocked"
        report["result"].update(
            reason=identity.get("reason", "retail identity mismatch"), exitCode=EXIT_USAGE
        )
        _write_report(out, report)
        return EXIT_USAGE
    if not args.addresses.is_file():
        report["status"] = "blocked"
        report["result"].update(reason=f"missing address table {args.addresses}", exitCode=EXIT_USAGE)
        _write_report(out, report)
        return EXIT_USAGE
    report["addressTable"] = {
        "path": str(args.addresses),
        "sha256": sha256_file(args.addresses),
    }
    if not args.replay.is_file():
        report["status"] = "blocked"
        report["result"].update(reason=f"missing replay {args.replay}", exitCode=EXIT_USAGE)
        _write_report(out, report)
        return EXIT_USAGE

    # The address table has to be in the environment *before* touhou_rl.spec is
    # imported: spec reads TH08_ADDRESS_FILE at import time.
    os.environ["TH08_ADDRESS_FILE"] = str(Path(args.addresses).resolve())
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from touhou_rl import spec
    from touhou_rl.session import GameSession, SessionOptions

    report["addressSource"] = dict(spec.ADDRESS_SOURCE)
    report["addressesUsed"] = {
        name: hex(value)
        for name, value in (
            ("g_Supervisor", spec.ADDR_SUPERVISOR),
            ("g_GameManager", spec.ADDR_GAME_MANAGER),
            ("g_Player", spec.ADDR_PLAYER),
            ("g_TitleScreen", spec.ADDR_TITLE_SCREEN),
            ("g_Gui", spec.ADDR_GUI),
            ("g_CurFrameInput", spec.ADDR_CUR_FRAME_INPUT),
            ("g_LastFrameInput", spec.ADDR_LAST_FRAME_INPUT),
            ("g_GuiMessageInputCurrent", spec.ADDR_GUI_INPUT_CURRENT),
            ("g_GuiMessageInputPrevious", spec.ADDR_GUI_INPUT_PREVIOUS),
        )
    }
    report["structOffsetsUsed"] = {
        "OFF_SUP_CUR_STATE": hex(spec.OFF_SUP_CUR_STATE),
        "OFF_GM_GAMEPLAY_FRAME": hex(spec.OFF_GM_GAMEPLAY_FRAME),
        "OFF_GM_CURRENT_STAGE": hex(spec.OFF_GM_CURRENT_STAGE),
        "OFF_GM_FLAGS": hex(spec.OFF_GM_FLAGS),
        "OFF_GM_GLOBALS": hex(spec.OFF_GM_GLOBALS),
        "OFF_GLOBALS_SCORE": hex(spec.OFF_GLOBALS_SCORE),
        "OFF_GLOBALS_LIVES": hex(spec.OFF_GLOBALS_LIVES),
        "OFF_GLOBALS_DEATHS": hex(spec.OFF_GLOBALS_DEATHS),
        "OFF_TITLE_CURRENT_SCREEN": hex(spec.OFF_TITLE_CURRENT_SCREEN),
    }

    blocked = _check_single_instance(report)
    if blocked is not None:
        _write_report(out, report)
        return blocked

    if time.perf_counter() > deadline:
        report["status"] = "timeout"
        report["result"].update(reason="budget exhausted before launch", exitCode=EXIT_TIMEOUT)
        _write_report(out, report)
        return EXIT_TIMEOUT

    # Hash the third_party sources before and after to prove nothing was written.
    watched = [RETAIL_EXE, args.asset_dir / "th08.dat", args.replay]
    before_hashes = _hash_dir_snapshot(watched)
    skip = ("score.dat",) if args.no_score_dat else ()
    report["runDirAssembly"] = assemble_run_dir(
        args.run_dir, args.asset_dir, force=args.force_run_dir, skip=skip
    )

    session = GameSession(SessionOptions(run_dir=Path(args.run_dir), windowed=True, boot_timeout_s=45.0))
    exit_code = EXIT_TIMEOUT
    try:
        if args.phase == "launch":
            exit_code = phase_launch(session, args, report, deadline, spec)
        elif args.phase == "idle":
            exit_code = phase_idle(session, args, report, deadline, spec)
        elif args.phase == "memory":
            exit_code = phase_memory(session, args, report, deadline, spec)
        else:
            exit_code = phase_capture(session, args, report, deadline, spec)
    except LaunchError as exc:
        report["status"] = "failed"
        report["result"].update(reason=str(exc), exitCode=exc.exit_code)
        exit_code = exc.exit_code
    except TimeoutError as exc:
        report["status"] = "timeout"
        report["result"].update(reason=f"TimeoutError: {exc}", exitCode=EXIT_TIMEOUT)
        exit_code = EXIT_TIMEOUT
    except Exception as exc:  # noqa: BLE001 - every failure must be recorded, not raised
        import traceback

        report["status"] = "failed"
        report["result"].update(
            reason=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            exitCode=EXIT_STATE,
        )
        exit_code = EXIT_STATE
    finally:
        dialog = getattr(session, "dialog", 0)
        if dialog:
            from touhou_rl import winapi

            report["result"]["startupDialog"] = winapi.dialog_text(dialog)
            report["result"]["startupDialogExitCode"] = EXIT_BLOCKED
            if exit_code == EXIT_OK:
                exit_code = EXIT_BLOCKED
        try:
            close_code = session.close()
            report["closeExitCode"] = close_code
        except Exception as exc:  # noqa: BLE001
            report["closeError"] = f"{type(exc).__name__}: {exc}"
        # A non-zero close code is the retail image dying (0xC0000005 = access
        # violation); Windows records the faulting offset, which is the only stack
        # information available for an image without a matching PDB.
        code_for_postmortem = report.get("closeExitCode")
        if code_for_postmortem not in (0, None):
            report["postMortem"] = query_crash_events()
        report["sessionLog"] = list(getattr(session, "log", []))
        report["elapsedSeconds"] = round(time.perf_counter() - t0, 3)
        report["thirdPartyUnchanged"] = _hash_dir_snapshot(watched) == before_hashes
        report["result"]["exitCode"] = exit_code
        report["_t0"] = None
        _write_report(out, report)

    print(json.dumps({"phase": args.phase, "status": report.get("status"), "exitCode": exit_code,
                      "elapsedSeconds": report.get("elapsedSeconds"),
                      "report": str(out / "report.json")}, ensure_ascii=False))
    return exit_code


def _write_report(out: Path, report: dict) -> None:
    payload = {key: value for key, value in report.items() if not key.startswith("_")}
    (Path(out) / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
