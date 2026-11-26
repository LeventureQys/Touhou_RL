#!/usr/bin/env python3
"""V831-SLICE-02 follow-up: the offline source/adapter gates of ticket 02-A.

This is a **new instrument**.  It imports the frozen modules, it never edits
them, and step 1 (this file's current state) starts no engine.

What is implemented here and what is not
----------------------------------------
Implemented, fully offline, stdlib only at import time:

``sources``   the four registered corpora of ``TRAINING_HOST_REQUEST_02`` §02-A.1
              are read as four separate identities.  Each gets sha256/bytes/rows,
              the frame grid, the 2-frame decision count, leading frames, mask
              statistics, hash coverage (``hash_pre``/``hash_post`` frame
              interval + step) and its recorded ``obs`` shape.  The two
              registered sha256 values are checked verbatim.  Three (four) frame
              level alignments are computed **only over the intervals the
              evidence actually covers**.
``adapter``   the real import path of the frozen two-frame rebuild is bound by
              ``inspect.getsourcefile`` + source sha256, and :func:`decode_pair`
              turns a recorded two-frame pair back into ``(cand, bomb)`` with one
              expected-mask rule plus one narrowly named tolerance.  Every
              rejection is named and counted.
``prefix``    §02-A.3: **one engine start**.  RESET 0 1 0 12, then 8,664 frames
              through the D1 masks one ``STEP mask 1`` at a time, checking every
              D1 anchor (2,165 ``hash_pre`` after the reset + 2,166 ``hash_post``;
              the reset hash is a third, separate check).  Seven consecutive
              frames (STEP 8,659..8,665 → pixel frames 8,658..8,664) are captured
              with the shared block, stored as deterministic PNGs and given a
              blake2b-128, a sha256 and the engine-comparable FNV.  The obs-v3
              stack (pixel frames 8,660..8,663, oldest first) and the 4,332-entry
              action history are rebuilt, three actor-input digests are taken, and
              ``STATE``+``PLAYER``+``HUD`` are recorded at frames 8,663 and 8,664.
              Four more frames continue the same executor and frame 8,668's hash
              is checked against D1 row 2,166's ``hash_post``.
``reference`` §02-A.4: **a second, independent engine start** on the same masks
              with a *different STEP granularity* (one ``STEP mask 2`` when a
              decision's two masks agree, two ``STEP mask 1`` otherwise), its own
              private game directory, a ``PIXELS`` reply cross-check on every
              captured frame and an independent ``mask_to_cand`` decode.
``compare``   offline: the seven frames (blake2b-128, sha256, PNG bytes), the mask
              trace, the three actor-input digests, HUD, the last eight actions,
              the 8,663/8,664 states and the reset hash, each with an equal/首差
              verdict and the first difference verbatim.
``verify``    re-reads the raw artefacts of all four packages, recomputes the
              source sha256 values and the counters, re-applies the decode gate,
              decodes the stored PNGs again, re-derives both engine packages'
              anchors from their ``hashes.jsonl`` and re-runs ``compare``.
``run``       skeleton: runs the two offline packages in-process, writes
              ``<out>/run.json`` and ``<out>/ledger.jsonl``, and reports the two
              engine packages as ``not-run`` (each needs its own ledger start).
``recon``     §02-B: **one engine start**.  RESET 0 1 0 12, then the recorded
              T04 tape with one ``STEP mask 1`` per frame, so *every* frame's
              ``EVENTS`` word is read; after each decision's two frames
              ``HAZARDS`` (every enemy slot, no limit) + ``STATE`` + ``HUD`` +
              ``PLAYER`` are recorded, and every T04/D1 hash anchor is checked.
              Stops on a technical conflict, on ``STATE.stage`` leaving 0, on
              ``game_over``/a death, or at 13,195 frames (the 2-frame grid stops
              at 13,194).  Writes ``02b/recon/decisions.jsonl`` + ``result.json``.
``nominate``  §02-B offline: segments every enemy slot into generations, applies
              ``enemy_semantics.md`` §4 to every ending (kill M0/M1/M2, boss phase
              end M3, scripted retreat, out-of-screen despawn, clear wipe, no event,
              slot reuse), and freezes the earliest BOSS kill whose entry is at
              most 600 frames before its completion into ``02b/task_card.{json,md}``
              with a ``content_sha256``.  When nothing is freezable it writes the
              near-miss table and keeps the gate failed.
``intervene`` §02-C: **two engine starts** (teacher continuation vs constant
              candidate 12) from the frozen entry.  Each arm walks the frozen
              prefix from its own RESET, must reproduce the card's entry hash, and
              then runs its controller for at most 300 decisions / 600 frames while
              the frozen generation is tracked decision by decision.
``pilot``     §03: one frozen pixel-BC fit (init seed 0, exactly 200 updates), the
              open-loop table frozen on the independent donor root **before** the
              first development root, then the three arms on every development
              root; ``--only-seed 1000`` is the technical smoke.
``strength``  §03 strength check (ticket L51): **two engine starts** on the fit
              root (seed 13) only — the frozen open-loop table vs the frozen #12
              control — on the card's same task, prefix and window, plus the
              action-frequency/dwell statistics and the real-executability facts.
              ``pilot --strength-seed 13`` is the same code path.

Nothing here writes outside ``--out``, touches an existing evidence file, or
performs any git operation.  The two engine packages copy the whole game
directory, mute the **copy**, set ``SDL_AUDIODRIVER=dummy``, re-exec under
``xvfb-run``, re-check the copy's four mute bytes after the exit and re-hash the
read-only source before and after; any failure refuses the start or marks the run
``technical_invalid``.

The decode rule (why there is exactly one tolerance)
----------------------------------------------------
``cand = focus * 9 + move`` cannot express "SHOOT off", so the frozen executor
(:func:`touhou_rl.v400.teacher.shoot.commit_frames`) emits, for one decision of
two frames, either ``[m, m]`` or — on a pulse decision — ``[m, m & ~SHOOT]``.
:func:`decode_pair` therefore:

1. computes ``release = release_due(decision_index, period)``;
2. expands ``commit_frames(cand, decision_frames=2, release_shoot=release)``
   into the two expected masks and requires **bit-for-bit** equality with the
   recorded pair.  This is the main criterion;
3. accepts the recorded pair as a pulse release when it is exactly
   ``[m, m & ~SHOOT]``, ``mask_to_cand(m)`` yields ``bomb == 0`` and the release
   is due.  The STEP masks are never changed and never rebuilt;
4. names everything else: ``incomplete_pair``, ``unknown_bits``,
   ``no_shoot_both``, ``reverse_release``, ``bomb_pressed``,
   ``release_not_due``, ``release_missing``, ``split_non_shoot``.

``release_missing`` is the one class this file adds to the ticket's list: it is
the only hole left by "release due but the second frame kept SHOOT" (equal
masks), which is *not* a split and must not be reported as one.  It is a named
class with its own counter, never a catch-all, and it does not occur in any of
the four corpora.

There is **no** blanket ``mask | SHOOT`` anywhere: the tolerance is an exact
equality against ``m & ~SHOOT``, so a pair that changes a direction bit *and*
drops SHOOT falls through to ``split_non_shoot``.

Classification vocabulary used in every JSON this tool writes
-------------------------------------------------------------
``measured``   computed by this invocation from its inputs
``offline``    recomputed without an engine (all of ``sources``/``adapter``)
``historical`` quoted from a registered artefact (``R22/dense_blocker.json``)
``unverified`` asserted by a source but not re-derivable here

Exit codes: ``0`` gate passed, ``1`` gate failed with evidence written,
``2`` technical failure, ``3`` not implemented / refused by budget.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
import traceback
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOOL_PATH = Path(__file__).resolve()

#: Host code name of the training machine (ticket header field).
HOST_CODE_NAME = "轩神开播了"

#: ``(decision_index + 1) % 48 == 0``: the ticket's pulse phase, 2 frames per
#: decision, period 48.  Measured on the registered corpora (see ``sources``).
DEFAULT_PULSE_PERIOD = 48

SCHEMA_SOURCES = "v831-02a-sources/1"
SCHEMA_ADAPTER = "v831-02a-adapter/1"
SCHEMA_VERIFY = "v831-verify/1"
SCHEMA_RUN = "v831-run/1"
SCHEMA_LEDGER = "v831-ledger/1"
SCHEMA_ALIGNMENT = "v831-alignment/1"
SCHEMA_PREFIX = "v831-02a-prefix/1"
SCHEMA_REFERENCE = "v831-02a-reference/1"
SCHEMA_COMPARE = "v831-02a-compare/1"
SCHEMA_FRAMES = "v831-02a-frames/1"
SCHEMA_TAKEOVER = "v831-02a-takeover/1"

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_TECHNICAL = 2
EXIT_NOT_IMPLEMENTED = 3

#: Ticket budget (``TRAINING_HOST_REQUEST_02`` "全单" row).
BOOKED_AT = "2026-11-26T11:07:00+08:00"

#: Registered sha256 expectations, verbatim from the ticket.
REGISTERED_SHA256 = {
    "d1-canonical-seed12": "457728a988717ae2e5deae425196a43a6d0de4cbb21f18b4c41630f9d3aaccbd",
    "t04-2f-seed12": "f759de7dacc78343dec8825da5de14942e749b8a5a2c5319304e513d1618b7b9",
}

#: R22 (``Document/Update/Alpha/v4.3.1/evidence/R22/dense_blocker.json``) numbers
#: the ticket asks to re-check.  Quoted, then recomputed.
R22_EVIDENCE = "Document/Update/Alpha/v4.3.1/evidence/R22/dense_blocker.json"
R22_REGISTERED = {
    "slots": 6588,
    "first_decision_frame": 16,
    "last_decision_frame": 13190,
    "split_mask_slots": 137,
    "hash_frames": 4327,
}

#: The four registered sources of §02-A.1, each kept as its own identity.
SOURCES: tuple[dict[str, Any], ...] = (
    {
        "id": "d1-canonical-seed12",
        "kind": "labels",
        "label": "D1 canonical seed12 (teacher obs shard)",
        "rel": ".tmp_v400_d1/d1/shards/d1-s0d1-n3371-seed12-shard/labels.jsonl",
        "meta_rel": ".tmp_v400_d1/d1/shards/d1-s0d1-n3371-seed12-shard/shard.json",
        "obs_rel": ".tmp_v400_d1/d1/shards/d1-s0d1-n3371-seed12-shard",
        "seed": 12,
        "stage": 0,
        "difficulty": "normal",
        "registered_sha256": REGISTERED_SHA256["d1-canonical-seed12"],
        "registered_by": "TRAINING_HOST_REQUEST_02 §02-A.1",
    },
    {
        "id": "s1c-seed12",
        "kind": "labels",
        "label": "S1c seed12 (re-recorded teacher obs shard)",
        "rel": ".tmp_v410_s1c/d1/shards/d1-s0d1-n3300-seed12-shard/labels.jsonl",
        "meta_rel": ".tmp_v410_s1c/d1/shards/d1-s0d1-n3300-seed12-shard/shard.json",
        "obs_rel": ".tmp_v410_s1c/d1/shards/d1-s0d1-n3300-seed12-shard",
        "seed": 12,
        "stage": 0,
        "difficulty": "normal",
        "registered_sha256": None,
        "registered_by": None,
    },
    {
        "id": "t02c-2f-seed12",
        "kind": "slots",
        "label": "T02c 2-frame rebuild seed12 (per-frame committed masks)",
        "rel": ".tmp_v420_data/T02c/rebuild/d1-s0d1-n3300-seed12-shard/2f/slots.jsonl",
        "meta_rel": ".tmp_v420_data/T02c/rebuild/d1-s0d1-n3300-seed12-shard/2f/meta.json",
        "obs_rel": ".tmp_v420_data/T02c/rebuild/d1-s0d1-n3300-seed12-shard/2f",
        "seed": 12,
        "stage": 0,
        "difficulty": "normal",
        "registered_sha256": None,
        "registered_by": None,
        # Registered *inside* T04's meta.json (source.corpus.slot_file_sha256).
        "registered_sha256_alt": {
            "value": "ef5fb00f943ae77c3276849bad26fa7d301d984657964ebdaef72075c8bd7a69",
            "registered_by": "T04 meta.json source.corpus.slot_file_sha256",
        },
    },
    {
        "id": "t04-2f-seed12",
        "kind": "slots",
        "label": "T04 2-frame dataset seed12 (single mask per slot + hashes)",
        "rel": ".tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed12-shard/slots.jsonl",
        "meta_rel": ".tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed12-shard/meta.json",
        "obs_rel": ".tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed12-shard",
        "seed": 12,
        "stage": 0,
        "difficulty": "normal",
        "registered_sha256": REGISTERED_SHA256["t04-2f-seed12"],
        "registered_by": "TRAINING_HOST_REQUEST_02 §02-A.1",
    },
)

SOURCE_BY_ID = {s["id"]: s for s in SOURCES}

#: Modules whose *imported source* every number in this file is bound to.
IMPORT_BINDINGS: tuple[dict[str, Any], ...] = (
    {
        "module": "touhou_rl.v400.gym.actions",
        "role": "the one action table: cand = focus * 9 + move, 18 candidates",
        "attrs": ("cand_to_mask", "mask_to_cand", "N_CAND", "AGENT_MASK",
                  "DIRECTION_MASK", "MOVES_PER_CAND"),
        "calls": (),
    },
    {
        "module": "touhou_rl.v400.teacher.shoot",
        "role": "SHOOT pulse rule and the two-frame commit",
        "attrs": ("release_due", "commit_frames", "SHOOT_PULSE_PERIOD",
                  "commit_block", "mask_summary"),
        "calls": (),
    },
    {
        "module": "touhou_rl.v420.contracts",
        "role": "obs-v420-2f / action-v420-2f contract (decision span, history)",
        "attrs": ("DECISION_FRAMES", "FRAME_STACK", "ACTION_HISTORY",
                  "CONTRACT_VERSION", "OBS_CONTRACT", "ACTION_CONTRACT",
                  "decision_span", "history_frames_for"),
        "calls": (),
    },
    {
        "module": "touhou_rl.v420.counterfactual",
        "role": "the frozen history rebuild path (_pair_cand / replay)",
        "attrs": ("_StudentSession._pair_cand", "_StudentSession.replay"),
        "calls": (),
    },
    {
        "module": "touhou_rl.v400.student.obs_spec_v3",
        "role": "frozen obs-v3 contract digest",
        "attrs": ("obs_spec_v3_digest", "FRAME_STACK"),
        "calls": (("obs_spec_v3_digest", ()),),
    },
    {
        "module": "touhou_rl.spec",
        "role": "button bits and the direction table",
        "attrs": ("BTN_SHOOT", "BTN_BOMB", "BTN_FOCUS", "BTN_UP", "BTN_DOWN",
                  "BTN_LEFT", "BTN_RIGHT", "MOVES"),
        "calls": (),
    },
)

#: Support files whose identity this tool records but does not import.
SUPPORT_FILES: tuple[dict[str, str], ...] = (
    {"path": "tools/rl/v8_31_slice.py",
     "role": "mute / private-game-dir helpers reused by the later engine stages"},
    {"path": "tests/test_v8_31_followup.py", "role": "the offline tests of this tool"},
    {"path": R22_EVIDENCE, "role": "R22 registered numbers quoted by `sources`"},
)

# ---------------------------------------------------------------------------
# 02-A.3 / 02-A.4: the constants the two engine packages are built on
# ---------------------------------------------------------------------------
#: The ticket's takeover boundary: 8,664 prefix frames = 4,332 complete 2-frame
#: decisions (``TRAINING_HOST_REQUEST_02`` line 31).
PREFIX_FRAME = 8664
PREFIX_DECISIONS = PREFIX_FRAME // 2
#: Four more frames through the same executor, ending on a recorded ``hash_post``.
CONTINUATION_FRAMES = 4
CONTINUATION_LAST_FRAME = PREFIX_FRAME + CONTINUATION_FRAMES
#: The seven-step window: STEP F leaves frame ``F-1`` in the shared block, so
#: STEP 8,659..8,665 publishes pixel frames 8,658..8,664 and the last four
#: pictures before the takeover boundary are pixel frames 8,660..8,663.
WINDOW_STEPS: tuple[int, ...] = tuple(range(PREFIX_FRAME - 5, PREFIX_FRAME + 2))
WINDOW_PIXEL_FRAMES: tuple[int, ...] = tuple(step - 1 for step in WINDOW_STEPS)
#: The four frames pushed onto the obs-v3 stack, oldest first.
STACK_PIXEL_FRAMES: tuple[int, ...] = (8660, 8661, 8662, 8663)
#: Where the two resource records are taken (R23's position and the boundary).
RESOURCE_FRAMES: tuple[int, ...] = (8663, 8664)
#: ``RESET 0 1 0 12``: stage 0, difficulty 1 = normal, team 0, seed 12.
RESET_STAGE = 0
RESET_DIFFICULTY = "normal"
RESET_TEAM = 0
RESET_SEED = 12
#: One decision is two game frames; the SHOOT pulse repeats every 48 decisions.
DECISION_FRAMES = 2

#: The frame relation, quoted from ``tools/rl/v8_31_slice.py`` and re-measured on
#: every capture instead of trusted (``classify_captured_frame`` does the check).
PIXEL_FRAME_RELATION = (
    "STEP F returns header.frame == EVENTS.frames == F and the RGBA in the shared "
    "block belongs to frame F-1 (AfterDrawFrame publishes before ++g_frameCount); "
    "captured_at_step_F and pixel_frame = F-1 are recorded for every capture")


# ---------------------------------------------------------------------------
# small helpers (the same shapes as tools/rl/v8_31_slice.py, on purpose)
# ---------------------------------------------------------------------------
def now_iso() -> str:
    """Local wall clock with its timezone, the project's report format."""

    return datetime.now().astimezone().isoformat()


def sha256_bytes(data: bytes) -> str:
    """sha256 of a byte string."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """sha256 of a file's bytes, streamed so a 283 MB ``.npy`` is cheap."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON atomically (``<path>.tmp`` then replace)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(target)
    return target


def write_jsonl_row(handle: Any, row: Mapping[str, Any]) -> None:
    """Append one compact JSON row; the caller owns the handle's lifetime."""

    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def _parse_kv(reply: str) -> dict[str, str]:
    """``OK a=1 b=2`` -> ``{"_verb": "OK", "a": "1", "b": "2"}``.

    The engine-speaking stages use it; kept here so the follow-up tool does not
    depend on a private helper of the frozen client.
    """

    parts = str(reply).split()
    out: dict[str, str] = {"_verb": parts[0] if parts else ""}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream one JSON object per non-empty line (never loads the file whole)."""

    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # fail closed, never skip
                raise ValueError(f"{path}:{number}: not a JSON object: {exc}") from exc


def repo_rel(path: str | Path, repo: str | Path = REPO_ROOT) -> str:
    """A stable, POSIX-style path relative to the repo root when possible."""

    target = Path(path).expanduser().resolve()
    try:
        return target.relative_to(Path(repo).expanduser().resolve()).as_posix()
    except ValueError:
        return target.as_posix()


def npy_header(path: str | Path) -> dict[str, Any]:
    """Read only a ``.npy`` header: shape/dtype without touching the payload.

    A ``.npy`` inside these corpora can be a **WSL symlink** (reparse tag
    ``0xa000001d``).  Windows cannot ``stat`` such a link, so the failure is
    reported as data (``npy: False`` + ``error``) instead of raising; the caller
    then resolves the shape through the link target the corpus itself registers.
    """

    target = Path(path)
    info: dict[str, Any] = {"npy": False, "path": target.as_posix()}
    try:
        size = target.stat().st_size
    except OSError as exc:
        info["error"] = f"{type(exc).__name__}: {exc.strerror or exc}"
        info["link_target"] = None
        try:
            info["link_target"] = os.readlink(target)
        except (OSError, ValueError):
            pass
        return info
    info["bytes"] = size
    with target.open("rb") as handle:
        magic = handle.read(8)
        if magic[:6] != b"\x93NUMPY":
            info["error"] = "not a .npy"
            return info
        if magic[6] == 1:
            (length,) = struct.unpack("<H", handle.read(2))
        else:
            (length,) = struct.unpack("<I", handle.read(4))
        raw = handle.read(length).decode("latin-1").strip()
    try:
        header = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        info["error"] = f"unreadable header: {exc}"
        return info
    info.update({
        "npy": True,
        "shape": [int(value) for value in header.get("shape", ())],
        "descr": header.get("descr"),
        "fortran_order": bool(header.get("fortran_order", False)),
    })
    return info


def _wsl_to_windows(path: str) -> str | None:
    """``/mnt/d/WorkShop/x`` -> ``D:/WorkShop/x``; anything else -> None."""

    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", str(path))
    if not match:
        return None
    return f"{match.group(1).upper()}:/{match.group(2)}"


def resolve_broken_obs_links(sources: Sequence[dict[str, Any]],
                             repo: str | Path) -> list[dict[str, Any]]:
    """Resolve an unreadable ``obs`` file through the link its meta registers.

    Nothing is invented: the replacement path comes from the corpus's own
    ``meta.json`` (``obs.files[].target`` relative to the slot directory, or the
    absolute WSL ``source`` mapped through ``/mnt/<drive>/``).  Everything that
    was resolved is recorded with its provenance.
    """

    root = Path(repo).expanduser().resolve()
    resolved: list[dict[str, Any]] = []
    for source in sources:
        obs = source.get("obs") or {}
        registered = {entry.get("name"): entry
                      for entry in (obs.get("registered_files") or [])}
        for name, header in (obs.get("files") or {}).items():
            if header.get("npy"):
                continue
            entry = None
            for registered_name, candidate in registered.items():
                if registered_name and str(registered_name).endswith(
                        Path(header["path"]).name):
                    entry = candidate
                    break
            if entry is None:
                continue
            attempts: list[tuple[str, Path]] = []
            relative = entry.get("target")
            if relative:
                attempts.append(("meta.obs.target",
                                 (root / obs["dir"] / str(relative)).resolve()))
            absolute = entry.get("source")
            if absolute:
                mapped = _wsl_to_windows(str(absolute))
                if mapped:
                    attempts.append(("meta.obs.source (/mnt/<drive> mapped)",
                                     Path(mapped)))
            for how, candidate in attempts:
                if not candidate.is_file():
                    continue
                fresh = npy_header(candidate)
                if not fresh.get("npy"):
                    continue
                header.update({
                    "npy": True,
                    "shape": fresh["shape"],
                    "descr": fresh["descr"],
                    "bytes": fresh["bytes"],
                    "resolved_from": candidate.as_posix(),
                    "resolution": how,
                    "registered_sha256": entry.get("sha256"),
                    "registered_size": entry.get("size"),
                })
                resolved.append({"source": source["id"], "file": name,
                                 "resolved_from": candidate.as_posix(),
                                 "resolution": how})
                break
    return resolved


# ---------------------------------------------------------------------------
# identities: host, git, imported code
# ---------------------------------------------------------------------------
def host_identity() -> dict[str, Any]:
    """The training host, by its ticket code name plus the measured facts."""

    return {
        "code_name": HOST_CODE_NAME,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "tzname": time.tzname[0] if time.tzname else None,
        "utc_offset_s": -(time.altzone if time.daylight else time.timezone),
        "measured_at": now_iso(),
        "classification": "measured",
    }


def _git(repo: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} exited {completed.returncode}: "
            f"{completed.stderr.strip()}")
    return completed.stdout


def git_identity(repo: str | Path = REPO_ROOT) -> dict[str, Any]:
    """HEAD, branch and the dirty count the receipt must quote."""

    root = Path(repo).expanduser().resolve()
    try:
        head = _git(root, ["rev-parse", "HEAD"]).strip()
        branch = _git(root, ["rev-parse", "--abbrev-ref", "HEAD"]).strip()
        status = _git(root, ["status", "--short", "--untracked-files=no"])
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return {"available": False, "repo": root.as_posix(),
                "error": f"{type(exc).__name__}: {exc}", "classification": "measured"}
    dirty = [line for line in status.splitlines() if line.strip()]
    return {
        "available": True,
        "repo": root.as_posix(),
        "branch": branch,
        "head": head,
        "dirty_count": len(dirty),
        "dirty": dirty,
        "classification": "measured",
    }


def _resolve_dotted(module: Any, dotted: str) -> Any:
    obj = module
    for part in str(dotted).split("."):
        obj = getattr(obj, part)
    return obj


def import_identity(bindings: Sequence[Mapping[str, Any]] = IMPORT_BINDINGS,
                    repo: str | Path = REPO_ROOT) -> list[dict[str, Any]]:
    """Bind every reused attribute to its real file and that file's sha256.

    ``inspect.getsourcefile`` is the authority: it reports the file Python
    actually loaded, so a stale ``.pyc``/shadowed package cannot pass.  A module
    whose import fails is reported with ``import_error`` instead of a fabricated
    digest (fail closed: the gate then fails).
    """

    out: list[dict[str, Any]] = []
    for binding in bindings:
        entry: dict[str, Any] = {
            "module": binding["module"],
            "role": binding.get("role"),
            "classification": "measured",
        }
        try:
            module = importlib.import_module(binding["module"])
        except BaseException as exc:  # noqa: BLE001 - reported, never smoothed
            entry["import_error"] = f"{type(exc).__name__}: {exc}"
            entry["attributes"] = {}
            out.append(entry)
            continue
        module_file = inspect.getsourcefile(module) or getattr(module, "__file__", "")
        entry["file"] = repo_rel(module_file, repo) if module_file else None
        entry["sha256"] = sha256_file(module_file) if module_file else None
        entry["is_package"] = hasattr(module, "__path__")
        attributes: dict[str, Any] = {}
        for dotted in binding.get("attrs", ()):
            try:
                obj = _resolve_dotted(module, dotted)
            except AttributeError:
                attributes[dotted] = {"present": False}
                continue
            try:
                source_file = inspect.getsourcefile(obj)
            except TypeError:
                source_file = module_file
            try:
                lineno = inspect.getsourcelines(obj)[1]
            except (OSError, TypeError):
                lineno = None
            attributes[dotted] = {
                "present": True,
                "qualname": getattr(obj, "__qualname__", getattr(obj, "__name__", dotted)),
                "file": repo_rel(source_file, repo) if source_file else None,
                "file_sha256": sha256_file(source_file) if source_file else None,
                "lineno": lineno,
            }
        entry["attributes"] = attributes
        calls: dict[str, Any] = {}
        for name, args in binding.get("calls", ()):
            try:
                calls[name] = {"ok": True, "value": _resolve_dotted(module, name)(*args)}
            except BaseException as exc:  # noqa: BLE001
                calls[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        entry["calls"] = calls
        out.append(entry)
    return out


def support_file_identity(repo: str | Path = REPO_ROOT) -> list[dict[str, Any]]:
    """Identity of files this tool reads or reuses but does not import."""

    root = Path(repo).expanduser().resolve()
    out: list[dict[str, Any]] = []
    for spec in SUPPORT_FILES:
        path = root / spec["path"]
        entry = {"path": spec["path"], "role": spec["role"],
                 "present": path.is_file(), "bytes": None, "sha256": None}
        if path.is_file():
            entry["bytes"] = path.stat().st_size
            entry["sha256"] = sha256_file(path)
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# budget: limits, two-phase ledger, refusal
# ---------------------------------------------------------------------------
class BudgetRefused(Exception):
    """A reservation would exceed the ticket budget; nothing was started."""

    def __init__(self, reason: str, payload: Mapping[str, Any]) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.payload = dict(payload)


class LedgerError(Exception):
    """The ledger was asked to settle something it never reserved."""


#: The four budget dimensions, in the ticket's own units.
BUDGET_DIMENSIONS = ("starts", "engine_frames", "wall_s", "write_gib")


@dataclass(frozen=True)
class BudgetLimits:
    """The ticket's whole-ticket budget, taken from the CLI or the defaults."""

    max_starts: int = 28
    max_engine_frames: int = 360_000
    max_wall_s: float = 8 * 3600.0
    max_write_gib: float = 18.0
    booked_at: str = BOOKED_AT

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_starts": int(self.max_starts),
            "max_engine_frames": int(self.max_engine_frames),
            "max_wall_s": float(self.max_wall_s),
            "max_write_gib": float(self.max_write_gib),
            "booked_at": self.booked_at,
            "deadline": self.deadline(),
            "classification": "historical",
            "source": "TRAINING_HOST_REQUEST_02 budget table (全单 row)",
        }

    def deadline(self) -> str | None:
        """``booked_at + max_wall_s``; the wall clock is never restarted."""

        try:
            start = datetime.fromisoformat(self.booked_at)
        except (TypeError, ValueError):
            return None
        return (start + timedelta(seconds=float(self.max_wall_s))).isoformat()

    def limit_for(self, dimension: str) -> float:
        mapping = {
            "starts": self.max_starts,
            "engine_frames": self.max_engine_frames,
            "wall_s": self.max_wall_s,
            "write_gib": self.max_write_gib,
        }
        return mapping[str(dimension)]


def _zero_totals() -> dict[str, float]:
    return {name: 0 for name in BUDGET_DIMENSIONS}


def _add_totals(target: dict[str, float], addend: Mapping[str, Any]) -> None:
    for name in BUDGET_DIMENSIONS:
        target[name] = target.get(name, 0) + float(addend.get(name, 0) or 0)


class Ledger:
    """Two-phase attempt ledger: ``reserve`` before a start, ``settle`` after.

    One JSON object per line in ``<out>/ledger.jsonl``.  The file is re-read on
    construction, so a resumed run continues the same totals (the ticket's
    "中断后从同一台账续行, 疑似已启动先计已耗").  A reservation is refused —
    :class:`BudgetRefused`, with a ``refuse`` row written — as soon as any
    dimension would exceed its limit; nothing is ever started after a refusal.
    """

    def __init__(self, path: str | Path, limits: BudgetLimits,
                 *, clock=now_iso) -> None:
        self.path = Path(path)
        self.limits = limits
        self.clock = clock
        self.rows: list[dict[str, Any]] = []
        if self.path.is_file():
            for row in iter_jsonl(self.path):
                self.rows.append(dict(row))

    # -- persistence -------------------------------------------------------
    def _append(self, row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        payload.setdefault("schema", SCHEMA_LEDGER)
        payload["seq"] = len(self.rows) + 1
        payload.setdefault("at", self.clock())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            write_jsonl_row(handle, payload)
        self.rows.append(payload)
        return payload

    # -- derived state -----------------------------------------------------
    def reservations(self) -> dict[str, dict[str, Any]]:
        """Open reservations by attempt id (reserved, not yet settled)."""

        open_by_id: dict[str, dict[str, Any]] = {}
        for row in self.rows:
            phase = row.get("phase")
            attempt = row.get("attempt_id")
            if phase == "reserve" and attempt is not None:
                open_by_id[str(attempt)] = row
            elif phase == "settle" and attempt is not None:
                open_by_id.pop(str(attempt), None)
        return open_by_id

    def totals(self) -> dict[str, Any]:
        """Reserved, actual and committed (= actual + still open) per dimension."""

        reserved = _zero_totals()
        actual = _zero_totals()
        for row in self.rows:
            phase = row.get("phase")
            if phase == "reserve":
                _add_totals(reserved, row.get("requested") or {})
            elif phase == "settle":
                _add_totals(actual, row.get("actual") or {})
        committed = dict(actual)
        for reservation in self.reservations().values():
            _add_totals(committed, reservation.get("requested") or {})
        remaining = {name: float(self.limits.limit_for(name)) - committed.get(name, 0)
                     for name in BUDGET_DIMENSIONS}
        return {
            "limits": self.limits.as_dict(),
            "reserved": reserved,
            "actual": actual,
            "committed": committed,
            "remaining": remaining,
            "open_reservations": len(self.reservations()),
            "rows": len(self.rows),
            "path": self.path.as_posix(),
            "classification": "measured",
        }

    # -- two phases --------------------------------------------------------
    def reserve(self, attempt_id: str, *, starts: int = 1, engine_frames: int = 0,
                wall_s: float = 0.0, write_gib: float = 0.0,
                inputs: Mapping[str, Any] | None = None,
                note: str | None = None) -> dict[str, Any]:
        """Reserve one attempt; raise :class:`BudgetRefused` before any start."""

        requested = {
            "starts": float(starts),
            "engine_frames": float(engine_frames),
            "wall_s": float(wall_s),
            "write_gib": float(write_gib),
        }
        totals = self.totals()
        over: list[dict[str, Any]] = []
        for name in BUDGET_DIMENSIONS:
            would = float(totals["committed"].get(name, 0)) + requested[name]
            limit = float(self.limits.limit_for(name))
            if would > limit:
                over.append({"dimension": name, "limit": limit,
                             "already_committed": float(totals["committed"].get(name, 0)),
                             "requested": requested[name], "would_be": would})
        if over:
            payload = {
                "attempt_id": str(attempt_id),
                "phase": "refuse",
                "requested": requested,
                "exceeded": over,
                "totals": totals,
                "note": note,
                "classification": "measured",
            }
            self._append(payload)
            raise BudgetRefused(
                f"budget refused for {attempt_id}: "
                + "; ".join(f"{o['dimension']} {o['would_be']} > {o['limit']}"
                            for o in over),
                payload)
        if str(attempt_id) in self.reservations():
            raise LedgerError(f"{attempt_id} is already reserved and not settled")
        return self._append({
            "attempt_id": str(attempt_id),
            "phase": "reserve",
            "requested": requested,
            "committed_before": totals["committed"],
            "limits": self.limits.as_dict(),
            "inputs": dict(inputs or {}),
            "note": note,
            "classification": "measured",
        })

    def settle(self, attempt_id: str, *, starts: int | None = None,
               engine_frames: int = 0, wall_s: float = 0.0,
               write_gib: float = 0.0, result: Mapping[str, Any] | None = None,
               note: str | None = None) -> dict[str, Any]:
        """Record the actuals of a reserved attempt (never the reservation)."""

        reservation = self.reservations().get(str(attempt_id))
        if reservation is None:
            raise LedgerError(f"{attempt_id} was never reserved; refusing to settle")
        reserved = dict(reservation.get("requested") or {})
        actual = {
            "starts": float(reserved.get("starts", 0) if starts is None else starts),
            "engine_frames": float(engine_frames),
            "wall_s": float(wall_s),
            "write_gib": float(write_gib),
        }
        return self._append({
            "attempt_id": str(attempt_id),
            "phase": "settle",
            "reserved": reserved,
            "actual": actual,
            "delta": {name: actual[name] - float(reserved.get(name, 0))
                      for name in BUDGET_DIMENSIONS},
            "result": dict(result or {}),
            "note": note,
            "classification": "measured",
        })


# ---------------------------------------------------------------------------
# gate result structure
# ---------------------------------------------------------------------------
@dataclass
class Check:
    """One named predicate of a gate."""

    name: str
    ok: bool
    detail: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": bool(self.ok), "detail": self.detail}


@dataclass
class Gate:
    """A gate is a named list of checks; it passes only if every check passes."""

    name: str
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: Any = None) -> Check:
        check = Check(name=name, ok=bool(ok), detail=detail)
        self.checks.append(check)
        return check

    def extend(self, other: "Gate") -> None:
        for check in other.checks:
            self.checks.append(Check(f"{other.name}.{check.name}", check.ok, check.detail))

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.ok for check in self.checks)

    @property
    def problems(self) -> list[str]:
        return [f"{check.name}: {check.detail}" for check in self.checks if not check.ok]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
            "failed": len(self.problems),
            "problems": self.problems,
        }


# ---------------------------------------------------------------------------
# the action adapter
# ---------------------------------------------------------------------------
class AdapterUnavailable(RuntimeError):
    """The frozen action table could not be imported (technical failure)."""


_ADAPTER: dict[str, Any] = {}


def adapter() -> dict[str, Any]:
    """Import the frozen action/pulse functions once and cache them.

    Every number this tool reports comes from these exact objects; the tool
    never re-implements ``cand = focus * 9 + move``, ``commit_frames`` or
    ``release_due``.
    """

    if _ADAPTER:
        return _ADAPTER
    try:
        from touhou_rl.spec import (BTN_BOMB, BTN_DOWN, BTN_FOCUS, BTN_SHOOT,
                                    BTN_UP, MOVES)
        from touhou_rl.v400.gym.actions import (AGENT_MASK, DIRECTION_MASK,
                                                N_CAND, cand_to_mask,
                                                mask_to_cand)
        from touhou_rl.v400.gym.actions import MOVES_PER_CAND
        from touhou_rl.v400.teacher.shoot import (commit_frames, mask_summary,
                                                  release_due)
        from touhou_rl.v420 import contracts as contracts
    except BaseException as exc:  # noqa: BLE001 - reported as a technical failure
        raise AdapterUnavailable(f"cannot import the frozen action table: "
                                 f"{type(exc).__name__}: {exc}") from exc
    _ADAPTER.update({
        "BTN_SHOOT": BTN_SHOOT,
        "BTN_BOMB": BTN_BOMB,
        "BTN_FOCUS": BTN_FOCUS,
        "BTN_UP": BTN_UP,
        "BTN_DOWN": BTN_DOWN,
        "MOVES": MOVES,
        "AGENT_MASK": AGENT_MASK,
        "DIRECTION_MASK": DIRECTION_MASK,
        "N_CAND": N_CAND,
        "MOVES_PER_CAND": MOVES_PER_CAND,
        "cand_to_mask": cand_to_mask,
        "mask_to_cand": mask_to_cand,
        "commit_frames": commit_frames,
        "release_due": release_due,
        "mask_summary": mask_summary,
        "contracts": contracts,
    })
    return _ADAPTER


#: Accepted classifications.
ACCEPTED = ("accepted", "accepted_via_release")

#: Every named rejection class, in the order they are tested.
REJECTION_CLASSES: tuple[str, ...] = (
    "incomplete_pair",
    "unknown_bits",
    "no_shoot_both",
    "reverse_release",
    "bomb_pressed",
    "release_not_due",
    "release_missing",
    "split_non_shoot",
)

CLASSIFICATIONS: tuple[str, ...] = ACCEPTED + REJECTION_CLASSES


def expand_commit(frames: Iterable[tuple[int, int]]) -> list[int]:
    """``[(mask, count), ...]`` -> one mask per frame."""

    out: list[int] = []
    for mask, count in frames:
        out.extend([int(mask)] * int(count))
    return out


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return None
    return None


def decode_pair(pair: Any, decision_index: int, period: int = DEFAULT_PULSE_PERIOD,
                *, decision_frames: int = 2) -> tuple[int | None, int | None, str, dict[str, Any]]:
    """Decode one recorded two-frame decision into ``(cand, bomb)``.

    Returns ``(cand, bomb, classification, detail)``.  ``cand`` is ``None`` only
    for ``incomplete_pair``/``unknown_bits`` (nothing was decodable); for every
    other class it is the candidate the *first* mask decodes to.

    The main criterion is bit-for-bit equality with
    ``commit_frames(cand, decision_frames=2, release_shoot=release_due(i, period))``.
    The single tolerance is the exact pair ``[m, m & ~SHOOT]`` on a decision whose
    release is due with ``bomb == 0``: accepted as a pulse release, and the STEP
    masks are *not* modified.  Everything else is a named rejection.
    """

    table = adapter()
    shoot = int(table["BTN_SHOOT"])
    bomb_bit = int(table["BTN_BOMB"])
    agent_mask = int(table["AGENT_MASK"])
    index = int(decision_index)
    period = int(period)
    detail: dict[str, Any] = {
        "decision_index": index,
        "period": period,
        "decision_frames": int(decision_frames),
        "release_due": False,
        "expected": None,
        "expected_without_release": None,
        "rule": None,
    }

    masks: list[Any] | None
    if isinstance(pair, (list, tuple)):
        masks = list(pair)
    elif isinstance(pair, (str, bytes)):
        masks = None
    else:
        masks = None
    if masks is None or len(masks) != decision_frames:
        detail.update(
            rule="pair must carry exactly %d masks" % int(decision_frames),
            got=None if masks is None else len(masks),
        )
        return (None, None, "incomplete_pair", detail)

    values = [_maybe_int(mask) for mask in masks]
    if any(value is None for value in values):
        detail.update(rule="a mask is not an integer", pair=list(masks))
        return (None, None, "incomplete_pair", detail)
    m0, m1 = int(values[0]), int(values[1])  # type: ignore[arg-type]
    detail["pair"] = [m0, m1]

    outside = (m0 | m1) & ~agent_mask
    if outside:
        detail.update(rule=f"mask has bits outside the agent mask: 0x{outside:04x}",
                      offending_bits=int(outside))
        return (None, None, "unknown_bits", detail)

    if not (m0 & shoot) and not (m1 & shoot):
        detail.update(rule="neither frame carries SHOOT")
        return (None, None, "no_shoot_both", detail)
    if not m0 & shoot:
        detail.update(rule="SHOOT off on the first frame and on on the second")
        return (None, None, "reverse_release", detail)

    try:
        cand, bomb = table["mask_to_cand"](m0)
    except ValueError as exc:
        detail.update(rule=f"the first mask is not a candidate mask: {exc}")
        return (None, None, "unknown_bits", detail)
    cand, bomb = int(cand), int(bomb)
    detail["cand"] = cand
    detail["bomb"] = bomb

    if bomb != 0 or (m1 & bomb_bit):
        detail.update(rule="a bomb bit is pressed; the frozen student never bombs",
                      bomb_first=bomb, bomb_second=int(m1 & bomb_bit))
        return (cand, bomb, "bomb_pressed", detail)

    release = bool(table["release_due"](index, period))
    expected = expand_commit(table["commit_frames"](
        cand, bomb=0, decision_frames=decision_frames, release_shoot=release))
    plain = expand_commit(table["commit_frames"](
        cand, bomb=0, decision_frames=decision_frames, release_shoot=False))
    detail.update(
        release_due=release,
        expected=expected,
        expected_without_release=plain,
    )

    if [m0, m1] == expected:
        if expected != plain:
            detail["rule"] = "expected_with_release"
            return (cand, 0, "accepted_via_release", detail)
        detail["rule"] = "expected"
        return (cand, 0, "accepted", detail)

    released = [m0, m0 & ~shoot]
    if [m0, m1] == released:
        if release:
            detail["rule"] = "release_tolerance"
            return (cand, 0, "accepted_via_release", detail)
        detail["rule"] = "release_shape_but_phase_not_due"
        return (cand, 0, "release_not_due", detail)

    if m1 == m0:
        detail["rule"] = "release_due_but_the_second_frame_kept_SHOOT"
        return (cand, 0, "release_missing", detail)

    detail["rule"] = "the masks differ by more than the SHOOT bit"
    detail["xor"] = int(m0 ^ m1)
    return (cand, 0, "split_non_shoot", detail)


def rejection_counter() -> dict[str, int]:
    """A zeroed counter for every classification this tool can emit."""

    return {name: 0 for name in CLASSIFICATIONS}


# ---------------------------------------------------------------------------
# corpora: loading, per-source statistics
# ---------------------------------------------------------------------------
class SourceError(RuntimeError):
    """A registered source is missing or contradicts its own registration."""


def _grid_summary(frames: Sequence[int]) -> dict[str, Any] | None:
    """``{count, first, last, step, regular}`` for one hash frame list."""

    ordered = sorted(int(frame) for frame in frames)
    if not ordered:
        return None
    if len(ordered) == 1:
        return {"count": 1, "first": ordered[0], "last": ordered[0],
                "step": None, "regular": True, "frames": ordered}
    diffs = {b - a for a, b in zip(ordered, ordered[1:])}
    step = None
    if len(diffs) == 1:
        step = diffs.pop()
    return {
        "count": len(ordered),
        "first": ordered[0],
        "last": ordered[-1],
        "step": step,
        "regular": step is not None,
        "span": ordered[-1] - ordered[0],
    }


def _hash_chain(pre: Mapping[int, str], post: Mapping[int, str]) -> dict[str, Any]:
    """``hash_pre`` and ``hash_post`` must agree wherever they name one frame.

    Both corpora place the row's ``hash_pre`` at the frame its predecessor's
    ``hash_post`` already stamped, so the two dicts overlap on every interior
    anchor: an overlap that disagrees is a chain break.
    """

    shared = sorted(set(pre) & set(post))
    breaks = [frame for frame in shared if pre[frame] != post[frame]]
    return {
        "hash_chain_compared": len(shared),
        "hash_chain_breaks": len(breaks),
        "hash_chain_first_break": breaks[0] if breaks else None,
    }


def load_labels_source(spec: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    """Read one ``labels.jsonl`` shard: per-frame masks, per-row hash anchors."""

    path = repo / spec["rel"]
    if not path.is_file():
        raise SourceError(f"missing registered source {spec['id']}: {path}")
    frame_masks: dict[int, int] = {}
    hash_pre: dict[int, str] = {}
    hash_post: dict[int, str] = {}
    pairs: list[tuple[int, list[int]]] = []
    distinct: set[int] = set()
    rows = mask_entries = split_rows = split_pairs = release_rows = bomb_rows = 0
    first_frame: int | None = None
    last_frame = -1
    first_decision_frame: int | None = None
    last_decision_frame: int | None = None
    frames_per_decision = None
    decisions_per_row = None
    cands: set[int] = set()
    bombs: set[int] = set()
    problems: list[str] = []
    for row in iter_jsonl(path):
        rows += 1
        decision_frame = int(row["decision_frame"])
        per_row = int(row.get("frames_per_decision") or 4)
        steps = [int(value) for value in row["step_frames"]]
        if frames_per_decision is None:
            frames_per_decision = per_row
            decisions_per_row = per_row // 2
        if per_row != frames_per_decision:
            problems.append(f"row {rows - 1}: frames_per_decision {per_row} "
                            f"!= {frames_per_decision}")
        if len(steps) != per_row:
            problems.append(f"row {rows - 1}: {len(steps)} step_frames, "
                            f"expected {per_row}")
        if first_decision_frame is None:
            first_decision_frame = decision_frame
        last_decision_frame = decision_frame
        if first_frame is None:
            first_frame = decision_frame
        row_split = 0
        row_release = 0
        row_bomb = 0
        for offset, mask in enumerate(steps):
            frame = decision_frame + offset
            frame_masks[frame] = mask
            last_frame = max(last_frame, frame)
            mask_entries += 1
            distinct.add(mask)
            if mask & int(adapter()["BTN_BOMB"]):
                row_bomb += 1
        for offset in range(0, per_row, 2):
            pair = steps[offset:offset + 2]
            if len(pair) != 2:
                problems.append(f"row {rows - 1}: incomplete pair at offset {offset}")
                continue
            pairs.append(((decision_frame + offset) // 2, [int(pair[0]), int(pair[1])]))
            if pair[0] != pair[1]:
                split_pairs += 1
                row_split += 1
                if (int(pair[0]) & int(adapter()["BTN_SHOOT"])) and \
                        int(pair[1]) == int(pair[0]) & ~int(adapter()["BTN_SHOOT"]):
                    row_release += 1
            try:
                cand, bomb = adapter()["mask_to_cand"](int(pair[0]))
                cands.add(int(cand))
                bombs.add(int(bomb))
            except ValueError:
                pass
        if row_split:
            split_rows += 1
        if row_release or row.get("shoot_release"):
            release_rows += 1
        if row_bomb:
            bomb_rows += 1
        if row.get("hash_pre"):
            hash_pre[decision_frame] = str(row["hash_pre"]).strip().lower()
        if row.get("hash_post"):
            hash_post[decision_frame + per_row] = str(row["hash_post"]).strip().lower()
    return {
        "kind": "labels",
        "rows": rows,
        "pairs": pairs,
        "frame_masks": frame_masks,
        "hash_pre": hash_pre,
        "hash_post": hash_post,
        "distinct_masks": distinct,
        "mask_entries": mask_entries,
        "split_rows": split_rows,
        "split_pairs": split_pairs,
        "shoot_release_rows": release_rows,
        "bomb_rows": bomb_rows,
        "first_frame": first_frame,
        "last_frame": (int(first_frame) + len(pairs) * 2 - 1) if first_frame is not None else -1,
        "last_recorded_mask_frame": last_frame,
        "frames_covered": len(pairs) * 2,
        "first_decision_frame": first_decision_frame,
        "last_decision_frame": last_decision_frame,
        "frames_per_decision": frames_per_decision,
        "decisions_per_row": decisions_per_row,
        "cands": cands,
        "bomb_values": bombs,
        "problems": problems,
        "pair_source": "recorded",
        "independent": True,
    }


def pair_from_single_mask(lead: int, decision_index: int) -> list[int]:
    """The executor's pair for a record that stores only the lead mask.

    Used **only** for the t04 slot list, whose records carry one ``mask`` with
    ``duration_frames = 2``; the result is flagged ``independent: false`` in every
    report because the tail is derived from the same frozen pulse rule the decoder
    then checks.  The literal alternative is reported beside it.
    """

    table = adapter()
    try:
        cand, _bomb = table["mask_to_cand"](int(lead))
    except ValueError:
        return [int(lead), int(lead)]
    release = bool(table["release_due"](int(decision_index), DEFAULT_PULSE_PERIOD))
    return expand_commit(table["commit_frames"](int(cand), bomb=0,
                                                decision_frames=2,
                                                release_shoot=release))


def load_slots_source(spec: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    """Read one 2-frame ``slots.jsonl``: one recorded decision per row.

    ``t02c`` records both committed masks (``corpus.committed_masks``) and is a
    fully recorded pair source.  ``t04`` records a single ``mask`` with
    ``duration_frames = 2``; its tail mask is therefore *derived* from the frozen
    pulse phase and marked ``independent: false`` (the literal single-mask
    reading is reported separately by :func:`run_adapter`, never hidden).
    """

    path = repo / spec["rel"]
    if not path.is_file():
        raise SourceError(f"missing registered source {spec['id']}: {path}")
    table = adapter()
    shoot = int(table["BTN_SHOOT"])
    frame_masks: dict[int, int] = {}
    hash_pre: dict[int, str] = {}
    hash_post: dict[int, str] = {}
    pairs: list[tuple[int, list[int]]] = []
    literal_pairs: list[tuple[int, list[int]]] = []
    distinct: set[int] = set()
    rows = mask_entries = split_rows = split_pairs = release_rows = bomb_rows = 0
    first_frame: int | None = None
    last_frame = -1
    first_decision_frame: int | None = None
    last_decision_frame: int | None = None
    durations: set[int] = set()
    cands: set[int] = set()
    bombs: set[int] = set()
    problems: list[str] = []
    derived_tails = 0
    for row in iter_jsonl(path):
        rows += 1
        decision_frame = int(row["decision_frame"])
        duration = int(row.get("duration_frames") or 2)
        durations.add(duration)
        committed = (row.get("corpus") or {}).get("committed_masks")
        recorded = [int(value) for value in committed] if committed else []
        if not recorded:
            recorded = [int(row["mask"])]
        if first_decision_frame is None:
            first_decision_frame = decision_frame
        last_decision_frame = decision_frame
        if first_frame is None:
            first_frame = decision_frame
        for offset, mask in enumerate(recorded):
            frame = decision_frame + offset
            frame_masks[frame] = mask
            last_frame = max(last_frame, frame)
            mask_entries += 1
            distinct.add(mask)
        row_bomb = sum(1 for mask in recorded if mask & int(table["BTN_BOMB"]))
        index = decision_frame // 2
        if len(recorded) == 2:
            pair = [recorded[0], recorded[1]]
        else:
            derived_tails += 1
            pair = pair_from_single_mask(recorded[0], index)
        pairs.append((index, pair))
        literal_pairs.append((index, [recorded[0], recorded[-1]]))
        if pair[0] != pair[1]:
            split_pairs += 1
            split_rows += 1
            if (pair[0] & shoot) and pair[1] == pair[0] & ~shoot:
                release_rows += 1
        if row_bomb:
            bomb_rows += 1
        if row.get("hash_pre"):
            hash_pre[decision_frame] = str(row["hash_pre"]).strip().lower()
        if row.get("hash_post"):
            hash_post[decision_frame + duration] = str(row["hash_post"]).strip().lower()
        for field in ("cand", "executed_cand"):
            if row.get(field) is not None:
                cands.add(int(row[field]))
        if row.get("bomb") is not None:
            bombs.add(int(row["bomb"]))
        if decision_frame % duration != 0:
            problems.append(f"row {rows - 1}: decision_frame {decision_frame} "
                            f"is off the {duration}-frame grid")
    if len(durations) > 1:
        problems.append(f"duration_frames is not constant: {sorted(durations)}")
    duration = max(durations) if durations else 2
    frames_covered = len(pairs) * duration
    return {
        "kind": "slots",
        "rows": rows,
        "pairs": pairs,
        "literal_pairs": literal_pairs,
        "frame_masks": frame_masks,
        "hash_pre": hash_pre,
        "hash_post": hash_post,
        "distinct_masks": distinct,
        "mask_entries": mask_entries,
        "split_rows": split_rows,
        "split_pairs": split_pairs,
        "shoot_release_rows": release_rows,
        "bomb_rows": bomb_rows,
        "first_frame": first_frame,
        "last_frame": (int(first_frame) + frames_covered - 1) if first_frame is not None else -1,
        "last_recorded_mask_frame": last_frame,
        "frames_covered": frames_covered,
        "first_decision_frame": first_decision_frame,
        "last_decision_frame": last_decision_frame,
        "frames_per_decision": max(durations) if durations else None,
        "decisions_per_row": 1,
        "cands": cands,
        "bomb_values": bombs,
        "problems": problems,
        "derived_tail_slots": derived_tails,
        "pair_source": "recorded" if not derived_tails else "derived",
        "independent": not derived_tails,
    }


def load_source(spec: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    """Load one registered source and describe it as its own identity."""

    path = repo / spec["rel"]
    meta_path = repo / spec["meta_rel"]
    loader = load_labels_source if spec["kind"] == "labels" else load_slots_source
    model = loader(spec, repo)
    meta_sha = sha256_file(meta_path) if meta_path.is_file() else None
    meta_payload: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            model["problems"].append(f"meta unreadable: {type(exc).__name__}: {exc}")
    sha = sha256_file(path)
    registered = spec.get("registered_sha256")
    alt = spec.get("registered_sha256_alt") or {}
    obs_dir = repo / spec["obs_rel"] / "obs"
    obs: dict[str, Any] = {"dir": repo_rel(obs_dir, repo), "files": {}}
    meta_obs = meta_payload.get("obs") or {}
    if isinstance(meta_obs, Mapping) and meta_obs.get("files"):
        obs["registered_files"] = list(meta_obs["files"])
        obs["registered_mode"] = meta_obs.get("mode")
    for name, filename in (("play", "play96x112.npy"),
                           ("hud", "hud.npy"),
                           ("act_hist", "act_hist.npy")):
        header = npy_header(obs_dir / filename)
        header["repo_relative"] = repo_rel(obs_dir / filename, repo)
        header["link"] = (obs_dir / filename).is_symlink()
        obs["files"][name] = header
    span = int(model["frames_covered"])
    pre = _grid_summary(list(model["hash_pre"]))
    post = _grid_summary(list(model["hash_post"]))
    anchors = sorted(set(model["hash_pre"]) | set(model["hash_post"]))
    record = {
        "id": spec["id"],
        "label": spec["label"],
        "kind": spec["kind"],
        "seed": spec["seed"],
        "stage": spec["stage"],
        "difficulty": spec["difficulty"],
        "path": repo_rel(path, repo),
        "bytes": path.stat().st_size,
        "sha256": sha,
        "sha256_registered": registered,
        "sha256_registered_by": spec.get("registered_by"),
        "sha256_matches_registration": (None if registered is None else sha == registered),
        "sha256_registered_alt": alt or None,
        "sha256_matches_alt": (None if not alt.get("value") else sha == alt["value"]),
        "meta": {
            "path": repo_rel(meta_path, repo) if meta_path.is_file() else None,
            "sha256": meta_sha,
            "bytes": meta_path.stat().st_size if meta_path.is_file() else None,
            "schema": meta_payload.get("schema"),
            "registered": _meta_expectations(spec["kind"], meta_payload),
            "classification": "measured",
        },
        "obs": obs,
        "counts": {
            "rows": int(model["rows"]),
            "decision_frames": len(model["pairs"]),
            "frames": span,
            "frames_per_decision": int(model["frames_per_decision"] or 2),
            "decisions_per_row": int(model["decisions_per_row"] or 1),
            "first_frame": int(model["first_frame"]),
            "last_frame": int(model["last_frame"]),
            "leading_frames": int(model["first_frame"]),
            "first_decision_frame": model["first_decision_frame"],
            "last_decision_frame": model["last_decision_frame"],
            "first_pair_frame": 2 * int(model["pairs"][0][0]),
            "last_pair_frame": 2 * int(model["pairs"][-1][0]),
            "classification": "measured",
        },
        "masks": {
            "mask_entries": int(model["mask_entries"]),
            "distinct_masks": len(model["distinct_masks"]),
            "distinct_mask_values": sorted(int(v) for v in model["distinct_masks"]),
            "rows_with_split_pair": int(model["split_rows"]),
            "split_pairs": int(model["split_pairs"]),
            "shoot_release_rows": int(model["shoot_release_rows"]),
            "bomb_rows": int(model["bomb_rows"]),
            "candidate_values": sorted(int(v) for v in model["cands"]),
            "bomb_values": sorted(int(v) for v in model["bomb_values"]),
            "derived_tail_slots": int(model.get("derived_tail_slots") or 0),
            "classification": "measured",
        },
        "hash_coverage": {
            "hash_pre_frames": _strip(pre),
            "hash_post_frames": _strip(post),
            "hash_frames_total": (0 if pre is None else pre["count"])
                                 + (0 if post is None else post["count"]),
            "hash_anchor_frames_distinct": len(anchors),
            "hash_anchor_first": anchors[0] if anchors else None,
            "hash_anchor_last": anchors[-1] if anchors else None,
            **_hash_chain(model["hash_pre"], model["hash_post"]),
            "classification": "measured",
        },
        "pair_source": model["pair_source"],
        "independent": bool(model["independent"]),
        "problems": list(model["problems"]),
        "_model": model,
    }
    record["meta"]["agreement"] = meta_agreement(spec, record)
    return record


def _public(source: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the in-memory ``_model`` (13k mask entries) before serialisation."""

    return {key: value for key, value in source.items() if not key.startswith("_")}


def load_all_sources(repo: str | Path) -> tuple[dict[str, dict[str, Any]],
                                                list[dict[str, Any]],
                                                list[str]]:
    """Load the four registered sources, resolve their ``obs`` links, report gaps."""

    sources: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for spec in SOURCES:
        try:
            sources[spec["id"]] = load_source(spec, repo)
        except SourceError as exc:
            problems.append(f"{spec['id']}: {exc}")
    resolutions = resolve_broken_obs_links(list(sources.values()), repo)
    return sources, resolutions, problems


def _strip(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {key: summary[key] for key in
            ("count", "first", "last", "step", "regular", "span")
            if key in summary}


def _meta_expectations(kind: str, meta: Mapping[str, Any]) -> dict[str, Any]:
    """The fields a registered ``shard.json``/``meta.json`` asserts, quoted."""

    if not meta:
        return {}
    if kind == "labels":
        pairs = meta.get("hash_checkpoint_pairs") or []
        return {
            "shard": meta.get("shard"),
            "rows": meta.get("rows"),
            "frames": meta.get("frames"),
            "frames_committed": meta.get("frames_committed"),
            "decisions": meta.get("decisions"),
            "decision_frames": meta.get("decision_frames"),
            "frames_per_decision": meta.get("frames_per_decision"),
            "first_decision_frame": meta.get("first_decision_frame"),
            "last_decision_frame": meta.get("last_decision_frame"),
            "deaths_observed": meta.get("deaths_observed"),
            "hash_checkpoint_pairs": len(pairs),
            "hash_checkpoint_pairs_all_equal":
                (all(bool(entry.get("equal")) for entry in pairs) if pairs else None),
            "obs_spec_digest": meta.get("obs_spec_digest"),
            "obs_spec_version": meta.get("obs_spec_version"),
            "reset_command": meta.get("reset_command"),
            "complete": meta.get("complete"),
        }
    counts = meta.get("counts") or {}
    return {
        "schema": meta.get("schema"),
        "slots": counts.get("slots", counts.get("slots_written")),
        "slots_expected": counts.get("slots_expected"),
        "first_decision_frame": counts.get("first_decision_frame"),
        "last_decision_frame": counts.get("last_decision_frame"),
        "cold_start_dropped": counts.get("slots_cold_start_dropped",
                                         counts.get("label_rows_cold_start")),
        "hash": meta.get("hash"),
        "contract": (meta.get("contract") or {}).get("contract_version"),
    }


def meta_agreement(spec: Mapping[str, Any], source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compare the fields a meta file registers with what was measured."""

    registered = (source.get("meta") or {}).get("registered") or {}
    counts = source["counts"]
    checks: list[dict[str, Any]] = []

    def add(field: str, expected: Any, measured: Any) -> None:
        if expected is None:
            return
        checks.append({"field": field, "registered": expected, "measured": measured,
                       "ok": expected == measured})

    if spec["kind"] == "labels":
        add("rows", registered.get("rows"), counts["rows"])
        add("frames", registered.get("frames"), counts["frames"])
        add("frames_per_decision", registered.get("frames_per_decision"),
            counts["frames_per_decision"])
        add("decision_frames", registered.get("decision_frames"),
            counts["frames_per_decision"] // 2)
        add("first_decision_frame", registered.get("first_decision_frame"),
            counts["first_decision_frame"])
        add("last_decision_frame", registered.get("last_decision_frame"),
            counts["last_decision_frame"])
        add("hash_checkpoint_pairs", registered.get("hash_checkpoint_pairs"),
            source["hash_coverage"]["hash_frames_total"])
    else:
        add("slots", registered.get("slots"), counts["rows"])
        add("first_decision_frame", registered.get("first_decision_frame"),
            counts["first_decision_frame"])
        add("last_decision_frame", registered.get("last_decision_frame"),
            counts["last_decision_frame"])
        add("decision_frames", registered.get("slots"), counts["decision_frames"])
    return checks


# ---------------------------------------------------------------------------
# alignments
# ---------------------------------------------------------------------------
def _compare_maps(left: Mapping[int, Any], right: Mapping[int, Any],
                  first: int, last: int) -> dict[str, Any]:
    """Frame-by-frame comparison inside ``[first, last]``; never outside it."""

    compared = missing_left = missing_right = mismatches = 0
    first_diff: dict[str, Any] | None = None
    for frame in range(int(first), int(last) + 1):
        has_left = frame in left
        has_right = frame in right
        if has_left and has_right:
            compared += 1
            if left[frame] != right[frame]:
                mismatches += 1
                if first_diff is None:
                    first_diff = {"frame": frame, "left": left[frame],
                                  "right": right[frame]}
        elif has_left:
            missing_right += 1
        elif has_right:
            missing_left += 1
    return {
        "compared": compared,
        "mismatches": mismatches,
        "missing_left": missing_left,
        "missing_right": missing_right,
        "first_diff": first_diff,
    }


def build_alignment(left: Mapping[str, Any], right: Mapping[str, Any],
                    *, alignment_id: str, mapping: str,
                    extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One alignment row: masks, hashes, coverage, first difference, verdict."""

    left_model = left["_model"]
    right_model = right["_model"]
    first = max(int(left_model["first_frame"]), int(right_model["first_frame"]))
    last = min(int(left_model["last_frame"]), int(right_model["last_frame"]))
    if last < first:
        raise SourceError(f"{alignment_id}: the two sources do not overlap")
    masks = _compare_maps(left_model["frame_masks"], right_model["frame_masks"],
                          first, last)
    hash_left: dict[int, str] = {}
    hash_left.update(left_model["hash_pre"])
    hash_left.update(left_model["hash_post"])
    hash_right: dict[int, str] = {}
    hash_right.update(right_model["hash_pre"])
    hash_right.update(right_model["hash_post"])
    hashes = _compare_maps(hash_left, hash_right, first, last)
    if hashes["missing_left"] and hashes["missing_right"]:
        hash_state = "both_partial"
    elif hashes["compared"] == 0 and not hash_left:
        hash_state = "left_unavailable"
    elif hashes["compared"] == 0 and not hash_right:
        hash_state = "right_unavailable"
    elif hashes["compared"] == 0:
        hash_state = "no_common_anchor"
    else:
        hash_state = "compared"
    first_diff = None
    if masks["first_diff"] is not None:
        first_diff = {"kind": "mask", **masks["first_diff"]}
    if hashes["first_diff"] is not None and (
            first_diff is None or hashes["first_diff"]["frame"] < first_diff["frame"]):
        first_diff = {"kind": "hash", **hashes["first_diff"]}
    identical = masks["mismatches"] == 0 and hashes["mismatches"] == 0
    if masks["compared"] == 0:
        verdict = "no_mask_overlap"
    elif identical:
        verdict = "same_run_compatible"
    elif first_diff is not None and first_diff["frame"] <= first:
        verdict = "different"
    else:
        verdict = f"compatible_until_frame_{first_diff['frame']}" if first_diff else "different"
    return {
        "schema": SCHEMA_ALIGNMENT,
        "id": alignment_id,
        "left": left["id"],
        "right": right["id"],
        "mapping": mapping,
        "interval": {"first_frame": first, "last_frame": last,
                     "frames": last - first + 1},
        "coverage": {
            "compared": masks["compared"],
            "frames_compared": masks["compared"],
            "missing_left": masks["missing_left"],
            "missing_right": masks["missing_right"],
            "hash_compared": hashes["compared"],
            "hash_missing_left": hashes["missing_left"],
            "hash_missing_right": hashes["missing_right"],
            "classification": "measured",
        },
        "masks": {
            "compared": masks["compared"],
            "mismatches": masks["mismatches"],
            "first_diff": masks["first_diff"],
        },
        "hashes": {
            "compared": hashes["compared"],
            "matched": hashes["compared"] - hashes["mismatches"],
            "mismatched": hashes["mismatches"],
            "first_mismatch": hashes["first_diff"],
            "state": hash_state,
            "left_anchors_in_interval": hashes["compared"] + hashes["missing_right"],
            "right_anchors_in_interval": hashes["compared"] + hashes["missing_left"],
            "right_anchors_off_grid": hashes["missing_right"],
            "left_anchors_off_grid": hashes["missing_left"],
        },
        "extra": dict(extra or {}),
        "first_diff": first_diff,
        "identical": bool(identical),
        "verdict": verdict,
        "classification": "offline",
    }


def build_alignments(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The four alignment rows of §02-A.1, each inside its own interval."""

    d1 = sources["d1-canonical-seed12"]
    s1c = sources["s1c-seed12"]
    t02c = sources["t02c-2f-seed12"]
    t04 = sources["t04-2f-seed12"]
    rows = [
        build_alignment(
            d1, s1c,
            alignment_id="d1-vs-s1c",
            mapping="same 4-frame label grid: frame f -> step_frames[f % 4]; "
                    "raw masks are compared frame by frame, hashes at the "
                    "4-frame anchors both files record",
        ),
        build_alignment(
            d1, t04,
            alignment_id="d1-vs-t04",
            mapping="slot k (decision_frame 16+2k) -> D1 0-based frames "
                    "(16+2k, 17+2k) = row (16+2k)//4 element (16+2k)%4; the "
                    "recorded lead mask is compared, the tail mask is not "
                    "recorded by t04 (missing_right); hash_pre at 16+2k and "
                    "hash_post at 18+2k are compared only where D1 records an "
                    "anchor (multiples of 4)",
            extra={"t04_tail_mask_recorded": False,
                   "note": "t04 stores one mask per slot plus duration_frames=2; "
                           "its tail mask is only derivable from the pulse phase"},
        ),
        build_alignment(
            d1, t02c,
            alignment_id="d1-vs-t02c",
            mapping="slot k (decision_frame 16+2k) -> D1 0-based frames "
                    "(16+2k, 17+2k); both committed masks are recorded by t02c, "
                    "so both are compared; t02c records no hash",
            extra={"t02c_hash_recorded": False,
                   "note": "t02c is a per-frame mask record without hashes"},
        ),
        build_alignment(
            t02c, t04,
            alignment_id="t02c-vs-t04",
            mapping="same 2-frame slot grid (decision_frame 16+2k): the lead "
                    "masks are compared one-to-one; t02c records the tail mask, "
                    "t04 does not",
            extra={"same_grid": t02c["counts"]["decision_frames"]
                               == t04["counts"]["decision_frames"],
                   "candidate_values_equal": (t02c["masks"]["candidate_values"]
                                              == t04["masks"]["candidate_values"])},
        ),
    ]
    return rows


# ---------------------------------------------------------------------------
# 02-A.1 `sources`
# ---------------------------------------------------------------------------
def cmd_sources(args: argparse.Namespace) -> int:
    """§02-A.1: the four registered sources, their statistics, the alignments."""

    started = time.perf_counter()
    repo = Path(args.repo).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() / "02a"
    try:
        adapter()  # bind the frozen action table before reading anything
    except AdapterUnavailable as exc:
        print(f"technical failure: {exc}", file=sys.stderr)
        return EXIT_TECHNICAL

    gate = Gate("02a_sources")
    sources, obs_resolutions, load_problems = load_all_sources(repo)
    for problem in load_problems:
        print(f"missing source: {problem}", file=sys.stderr)
    if load_problems:
        payload = {
            "schema": SCHEMA_SOURCES,
            "generated_by": "tools/rl/v8_31_followup.py sources",
            "created_at": now_iso(),
            "classification": "offline",
            "host": host_identity(),
            "heads": git_identity(repo),
            "repo": repo.as_posix(),
            "sources": [],
            "alignments": [],
            "gate": Gate("02a_sources", [Check("sources_present", False,
                                               load_problems)]).as_dict(),
            "problems": load_problems,
        }
        write_json(out_dir / "sources.json", payload)
        print(f"refused: {len(load_problems)} of {len(SOURCES)} sources are missing",
              file=sys.stderr)
        return EXIT_GATE_FAILED

    alignments = build_alignments(sources)

    # -- gate -------------------------------------------------------------
    gate.add("four_identities_present", len(sources) == 4,
             sorted(sources))
    for source_id, expected in REGISTERED_SHA256.items():
        measured = sources[source_id]["sha256"]
        gate.add(f"registered_sha256:{source_id}", measured == expected,
                 {"registered": expected, "measured": measured})
    for row in alignments:
        coverage = row["coverage"]
        numeric = all(isinstance(coverage[key], int) for key in
                      ("compared", "missing_left", "missing_right"))
        gate.add(f"coverage_is_numeric:{row['id']}",
                 numeric and coverage["compared"] > 0, coverage)
        interval = row["interval"]
        inside_left = (interval["first_frame"] >= sources[row["left"]]["counts"]["first_frame"]
                       and interval["last_frame"] <= sources[row["left"]]["counts"]["last_frame"])
        inside_right = (interval["first_frame"] >= sources[row["right"]]["counts"]["first_frame"]
                        and interval["last_frame"] <= sources[row["right"]]["counts"]["last_frame"])
        gate.add(f"interval_inside_both_sources:{row['id']}", inside_left and inside_right,
                 {"interval": interval,
                  "left_frames": [sources[row["left"]]["counts"]["first_frame"],
                                  sources[row["left"]]["counts"]["last_frame"]],
                  "right_frames": [sources[row["right"]]["counts"]["first_frame"],
                                   sources[row["right"]]["counts"]["last_frame"]]})
        gate.add(f"comparisons_inside_interval:{row['id']}",
                 coverage["compared"] + coverage["missing_left"] + coverage["missing_right"]
                 == interval["frames"], {"interval_frames": interval["frames"],
                                         "coverage": coverage})
    for source_id, source in sources.items():
        gate.add(f"no_source_problems:{source_id}", not source["problems"],
                 source["problems"])
        gate.add(f"bomb_free:{source_id}", source["masks"]["bomb_rows"] == 0
                 and source["masks"]["bomb_values"] in ([], [0]),
                 {"bomb_rows": source["masks"]["bomb_rows"],
                  "bomb_values": source["masks"]["bomb_values"]})
        agree = source["meta"]["agreement"]
        bad = [entry for entry in agree if not entry["ok"]]
        gate.add(f"meta_agrees:{source_id}", not bad, bad or
                 f"{len(agree)} registered fields agree")
    obs_missing = [f"{source['id']}.{name}"
                   for source in sources.values()
                   for name, header in source["obs"]["files"].items()
                   if not header.get("npy")]
    gate.add("obs_shapes_read", not obs_missing, obs_missing or
             f"{3 * len(sources)} obs files have a shape")

    # -- R22 re-check ------------------------------------------------------
    r22_path = repo / R22_EVIDENCE
    r22_file = None
    if r22_path.is_file():
        try:
            r22_file = json.loads(r22_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            r22_file = None
    t04_hash = sources["t04-2f-seed12"]["hash_coverage"]
    t02c = sources["t02c-2f-seed12"]
    t04 = sources["t04-2f-seed12"]
    r22_measured = {
        "slots": t04["counts"]["rows"],
        "first_decision_frame": t04["counts"]["first_decision_frame"],
        "last_decision_frame": t04["counts"]["last_decision_frame"],
        "split_mask_slots": t02c["masks"]["rows_with_split_pair"],
        "hash_frames": t04_hash["hash_frames_total"],
    }
    # R22's 4,327 is quoted for a tape that ends at frame 8,666; recompute the
    # candidate denominators together with the prefix window it belongs to.
    hash_grid = sorted(set(t04["_model"]["hash_pre"]) | set(t04["_model"]["hash_post"]))
    r22_hash_candidates = {
        "anchors_16_8666": sum(1 for frame in hash_grid if 16 <= frame <= 8666),
        "anchors_16_8668": sum(1 for frame in hash_grid if 16 <= frame <= 8668),
        "slots_decision_frame_le_8666": sum(1 for frame in
                                            sorted(t04["_model"]["hash_pre"])
                                            if frame <= 8666),
        "slots_decision_frame_le_8668": sum(1 for frame in
                                            sorted(t04["_model"]["hash_pre"])
                                            if frame <= 8668),
    }
    r22_registered = dict(R22_REGISTERED)
    if r22_file:
        tape = ((r22_file.get("teacher_tapes") or {}).get("12") or {})
        r22_registered.update({
            "file": R22_EVIDENCE,
            "file_sha256": sha256_file(r22_path),
            "ticket_slots": tape.get("slots"),
            "ticket_first_decision_frame": tape.get("first_decision_frame"),
            "ticket_last_decision_frame": tape.get("last_decision_frame"),
            "ticket_split_mask_slots": tape.get("split_mask_slots"),
            "hash_frames": ((r22_file.get("attempt") or {}).get("record") or {})
                           .get("tape", {}).get("hash_frames", R22_REGISTERED["hash_frames"]),
            "offline_agreement": r22_file.get("offline_prefix_check", {}).get("offline_agreement"),
        })
    r22_checks = {
        "slots": r22_registered["slots"] == r22_measured["slots"],
        "first_decision_frame": (r22_registered["first_decision_frame"]
                                 == r22_measured["first_decision_frame"]),
        "last_decision_frame": (r22_registered["last_decision_frame"]
                                == r22_measured["last_decision_frame"]),
        "split_mask_slots": (r22_registered["split_mask_slots"]
                             == r22_measured["split_mask_slots"]),
    }
    r22_hash_reproduced = [name for name, value in r22_hash_candidates.items()
                           if value == r22_registered["hash_frames"]]
    gate.add("r22_registered_numbers_reproduced", all(r22_checks.values()),
             {"registered": r22_registered, "measured": r22_measured,
              "checks": r22_checks,
              "hash_frames_registered": r22_registered["hash_frames"],
              "hash_frames_candidates": r22_hash_candidates,
              "hash_frames_reproduced_by": r22_hash_reproduced or None})
    gate.add("t02c_sha256_matches_t04_meta_registration",
             sources["t02c-2f-seed12"]["sha256_matches_alt"] is True,
             sources["t02c-2f-seed12"]["sha256_registered_alt"])

    imports = import_identity(repo=repo)
    imports_ok = all("import_error" not in entry for entry in imports)
    gate.add("imported_code_bound", imports_ok,
             [entry["module"] for entry in imports if "import_error" in entry] or
             f"{len(imports)} modules bound by inspect.getsourcefile + sha256")

    elapsed = time.perf_counter() - started
    payload = {
        "schema": SCHEMA_SOURCES,
        "generated_by": "tools/rl/v8_31_followup.py sources",
        "created_at": now_iso(),
        "classification": "offline",
        "repo": repo.as_posix(),
        "argv": list(getattr(args, "argv", [])) or None,
        "host": host_identity(),
        "heads": git_identity(repo),
        "tool": {"path": repo_rel(TOOL_PATH, repo), "sha256": sha256_file(TOOL_PATH),
                 "bytes": TOOL_PATH.stat().st_size},
        "imports": imports,
        "support_files": support_file_identity(repo),
        "pulse": {
            "period_decisions": DEFAULT_PULSE_PERIOD,
            "rule": "release_due(i, period) = (i + 1) % period == 0",
            "decision_frames": 2,
            "measured_from": "the 137 split pairs of every corpus land on "
                             "decisions 47 mod 48 (2-frame global index)",
            "classification": "measured",
        },
        "registered_expectations": {
            "sha256": dict(REGISTERED_SHA256),
            "r22": r22_registered,
        },
        "r22_recheck": {
            "registered": r22_registered,
            "measured": r22_measured,
            "checks": r22_checks,
            "hash_frames_candidates": r22_hash_candidates,
            "hash_frames_reproduced_by": r22_hash_reproduced or None,
            "classification": "offline",
        },
        "sources": [_public(sources[spec["id"]]) for spec in SOURCES],
        "obs_link_resolutions": obs_resolutions,
        "alignments": alignments,
        "cost": {"seconds": round(elapsed, 3)},
        "gate": gate.as_dict(),
    }
    path = write_json(out_dir / "sources.json", payload)
    _print_sources_summary(payload)
    print(f"wrote: {path}  ({elapsed:.2f}s)  gate={'passed' if gate.passed else 'FAILED'}")
    return EXIT_OK if gate.passed else EXIT_GATE_FAILED


def _print_sources_summary(payload: Mapping[str, Any]) -> None:
    print(f"{'source':24s} {'sha256':>12s} {'rows':>6s} {'frames':>7s} "
          f"{'decisions':>9s} {'split':>5s} {'bomb':>5s} {'hashes':>7s}")
    print("-" * 92)
    for source in payload["sources"]:
        counts = source["counts"]
        masks = source["masks"]
        print(f"{source['id']:24s} {source['sha256'][:12]:>12s} "
              f"{counts['rows']:6d} {counts['frames']:7d} "
              f"{counts['decision_frames']:9d} {masks['rows_with_split_pair']:5d} "
              f"{masks['bomb_rows']:5d} "
              f"{source['hash_coverage']['hash_frames_total']:7d}")
    print()
    for row in payload["alignments"]:
        print(f"{row['id']:14s} masks {row['masks']['compared']:6d}/"
              f"{row['masks']['mismatches']:d} hash {row['hashes']['compared']:6d}/"
              f"{row['hashes']['mismatched']:d} "
              f"missing {row['coverage']['missing_left']:5d}/"
              f"{row['coverage']['missing_right']:5d} -> {row['verdict']}")


# ---------------------------------------------------------------------------
# 02-A.2 `adapter`
# ---------------------------------------------------------------------------
def describe_plan() -> list[dict[str, Any]]:
    """The exact decode runs §02-A.2 asks for (prefix, full, slot lists)."""

    return [
        {"id": "d1-prefix-8664", "source": "d1-canonical-seed12",
         "frame_cap": 8664,
         "description": "D1 first 8,664 frames = 4,332 complete 2-frame decisions"},
        {"id": "d1-full-13192", "source": "d1-canonical-seed12", "frame_cap": None,
         "description": "D1 full recording = 6,596 decisions"},
        {"id": "s1c-full-13192", "source": "s1c-seed12", "frame_cap": None,
         "description": "S1c full recording"},
        {"id": "t02c-slots-6588", "source": "t02c-2f-seed12", "frame_cap": None,
         "description": "T02c 2-frame slot list (both committed masks recorded)"},
        {"id": "t04-slots-6588", "source": "t04-2f-seed12", "frame_cap": None,
         "description": "T04 2-frame slot list (one recorded mask per slot; the "
                        "tail is derived from the frozen pulse phase)"},
    ]


def run_adapter(pairs: Sequence[tuple[int, list[int]]], *, frame_cap: int | None,
                period: int = DEFAULT_PULSE_PERIOD,
                samples: int = 3) -> dict[str, Any]:
    """Decode every pair of one run and count every classification."""

    counter = rejection_counter()
    accepted = accepted_via_release = tolerance_uses = mismatches = 0
    first_mismatch: dict[str, Any] | None = None
    bomb_nonzero = 0
    pair_cand_would_refuse = 0
    decisions = 0
    examples: dict[str, dict[str, Any]] = {}
    release_decisions = 0
    for decision_index, pair in pairs:
        if frame_cap is not None and 2 * int(decision_index) + 2 > int(frame_cap):
            continue
        decisions += 1
        cand, bomb, classification, detail = decode_pair(pair, decision_index, period)
        counter[classification] = counter.get(classification, 0) + 1
        if classification in ACCEPTED:
            accepted += 1
            if classification == "accepted_via_release":
                accepted_via_release += 1
                if detail.get("rule") == "release_tolerance":
                    tolerance_uses += 1
        if bomb not in (0, None):
            bomb_nonzero += 1
        if len(pair) == 2 and int(pair[0]) != int(pair[1]):
            pair_cand_would_refuse += 1
        if detail.get("release_due"):
            release_decisions += 1
        expected = detail.get("expected")
        if expected is not None and list(pair) != list(expected):
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = {"decision_index": int(decision_index),
                                  "frame": 2 * int(decision_index),
                                  "pair": list(pair), "expected": list(expected),
                                  "classification": classification}
        if classification in ACCEPTED:
            example = {
                "decision_index": int(decision_index),
                "frame": 2 * int(decision_index),
                "pair": [int(pair[0]), int(pair[1])],
                "cand": cand,
                "bomb": bomb,
                "classification": classification,
                "rule": detail.get("rule"),
                "release_due": detail.get("release_due"),
                "expected": expected,
            }
            examples.setdefault("first_accepted", example)
            examples.setdefault(classification, example)
            examples["last_accepted"] = example
    ordered = [examples[key] for key in
               ("first_accepted", "accepted_via_release", "last_accepted")
               if key in examples]
    if len(ordered) < samples:
        for key, example in examples.items():
            if example not in ordered:
                ordered.append(example)
            if len(ordered) >= samples:
                break
    return {
        "decisions": decisions,
        "accepted": accepted,
        "accepted_via_release": accepted_via_release,
        "tolerance_uses": tolerance_uses,
        "release_due_decisions": release_decisions,
        "expected_mask_mismatches": mismatches,
        "first_mismatch": first_mismatch,
        "pair_cand_would_refuse": pair_cand_would_refuse,
        "bomb_nonzero": bomb_nonzero,
        "rejections": {name: counter[name] for name in REJECTION_CLASSES},
        "classifications": counter,
        "samples": ordered[:samples],
        "period": int(period),
    }


def synthetic_rejection_samples() -> dict[str, Any]:
    """One concrete pair per classification, decoded by the same function.

    Every value here is computed by :func:`decode_pair` during this invocation;
    nothing is transcribed.  The two negatives that matter most:
    ``split_non_shoot_with_shoot_change`` changes a direction bit *and* drops
    SHOOT (it must not be swallowed by the release tolerance), and
    ``reverse_release`` has SHOOT off first.
    """

    table = adapter()
    cand_to_mask = table["cand_to_mask"]
    shoot = int(table["BTN_SHOOT"])
    m12 = int(cand_to_mask(12))
    m3 = int(cand_to_mask(3))
    m5 = int(cand_to_mask(5))
    bad_direction = int(table["BTN_UP"]) | int(table["BTN_DOWN"])
    cases: dict[str, tuple[Any, int, str]] = {
        "incomplete_pair_len1": ([m12], 0, "incomplete_pair"),
        "incomplete_pair_len3": ([m12, m12, m12], 0, "incomplete_pair"),
        "unknown_bits_agent": ([m12 | (1 << 8), m12 | (1 << 8)], 0, "unknown_bits"),
        "unknown_bits_direction": ([m12 | bad_direction, m12 | bad_direction], 0,
                                   "unknown_bits"),
        "no_shoot_both": ([m12 & ~shoot, m12 & ~shoot], 0, "no_shoot_both"),
        "reverse_release": ([m12 & ~shoot, m12], 0, "reverse_release"),
        "bomb_pressed": ([int(cand_to_mask(12, 1)), int(cand_to_mask(12, 1))], 0,
                         "bomb_pressed"),
        "release_not_due": ([m12, m12 & ~shoot], 0, "release_not_due"),
        "release_missing": ([m12, m12], 47, "release_missing"),
        "split_non_shoot": ([m12, m3], 0, "split_non_shoot"),
        "split_non_shoot_with_shoot_change": ([m12, m5 & ~shoot], 0,
                                              "split_non_shoot"),
        "accepted": ([m12, m12], 0, "accepted"),
        "accepted_via_release": ([m12, m12 & ~shoot], 47, "accepted_via_release"),
    }
    out: dict[str, Any] = {}
    for name, (pair, index, expected_class) in cases.items():
        cand, bomb, classification, detail = decode_pair(pair, index)
        out[name] = {
            "pair": [int(mask) for mask in pair],
            "decision_index": int(index),
            "release_due": detail.get("release_due"),
            "cand": cand,
            "bomb": bomb,
            "classification": classification,
            "expected_class": expected_class,
            "rule": detail.get("rule"),
            "expected": detail.get("expected"),
            "as_expected_class": classification == expected_class,
        }
    return out


def test_coverage(repo: Path) -> dict[str, Any]:
    """Does ``tests/test_v8_31_followup.py`` name every rejection class?"""

    path = repo / "tests" / "test_v8_31_followup.py"
    entry: dict[str, Any] = {
        "path": repo_rel(path, repo),
        "present": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "classes_found": [],
        "classes_missing": list(REJECTION_CLASSES),
    }
    if not path.is_file():
        return entry
    text = path.read_text(encoding="utf-8")
    found = [name for name in REJECTION_CLASSES if name in text]
    entry["classes_found"] = found
    entry["classes_missing"] = [name for name in REJECTION_CLASSES if name not in found]
    entry["accepted_classes_found"] = [name for name in ACCEPTED if name in text]
    return entry


def cmd_adapter(args: argparse.Namespace) -> int:
    """§02-A.2: bind the real rebuild path and decode every recorded pair."""

    started = time.perf_counter()
    repo = Path(args.repo).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() / "02a"
    try:
        adapter()
    except AdapterUnavailable as exc:
        print(f"technical failure: {exc}", file=sys.stderr)
        return EXIT_TECHNICAL

    gate = Gate("02a_adapter")
    imports = import_identity(repo=repo)
    gate.add("imported_code_bound",
             all("import_error" not in entry for entry in imports),
             [entry["module"] for entry in imports if "import_error" in entry] or
             f"{len(imports)} modules bound by inspect.getsourcefile + sha256")
    for entry in imports:
        for dotted, meta in (entry.get("attributes") or {}).items():
            if not meta.get("present"):
                gate.add(f"attribute_present:{entry['module']}.{dotted}", False,
                         "missing attribute")
    for entry in imports:
        for name, call in (entry.get("calls") or {}).items():
            gate.add(f"call_ok:{entry['module']}.{name}", bool(call.get("ok")),
                     {k: v for k, v in call.items() if k != "ok"})

    loaded, obs_resolutions, load_problems = load_all_sources(repo)
    if load_problems:
        payload = {
            "schema": SCHEMA_ADAPTER,
            "generated_by": "tools/rl/v8_31_followup.py adapter",
            "created_at": now_iso(),
            "classification": "offline",
            "repo": repo.as_posix(),
            "host": host_identity(),
            "heads": git_identity(repo),
            "imports": imports,
            "runs": [],
            "problems": load_problems,
            "gate": Gate("02a_adapter", [Check("sources_present", False,
                                               load_problems)]).as_dict(),
        }
        write_json(out_dir / "adapter.json", payload)
        print(f"refused: {len(load_problems)} registered sources are missing",
              file=sys.stderr)
        return EXIT_GATE_FAILED

    runs: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for plan in describe_plan():
        source = loaded[plan["source"]]
        result = run_adapter(source["_model"]["pairs"], frame_cap=plan["frame_cap"])
        result.update({
            "id": plan["id"],
            "source": plan["source"],
            "description": plan["description"],
            "frame_cap": plan["frame_cap"],
            "frames_covered": result["decisions"] * 2,
            "pair_source": source["pair_source"],
            "independent": bool(source["independent"]),
            "classification": "offline",
        })
        runs.append(result)
        current = summary.get(plan["source"])
        if current is None or result["decisions"] > current["decisions"]:
            summary[plan["source"]] = {
                "decisions": result["decisions"],
                "accepted": result["accepted"],
                "accepted_via_release": result["accepted_via_release"],
                "expected_mask_mismatches": result["expected_mask_mismatches"],
                "run": result["id"],
                "pair_source": result["pair_source"],
                "independent": result["independent"],
            }

    # The literal single-mask reading of t04, reported beside the derived one.
    t04_literal = run_adapter(loaded["t04-2f-seed12"]["_model"]["literal_pairs"],
                              frame_cap=None)
    t04_literal.update({
        "id": "t04-literal-single-mask",
        "source": "t04-2f-seed12",
        "description": "t04 read literally: one mask for both frames "
                       "(duration_frames=2), no pulse tail",
        "classification": "offline",
    })

    # -- gate -------------------------------------------------------------
    for run in runs:
        gate.add(f"expected_mask_reproduction:{run['id']}",
                 run["expected_mask_mismatches"] == 0 and run["accepted"] == run["decisions"],
                 {"decisions": run["decisions"], "accepted": run["accepted"],
                  "mismatches": run["expected_mask_mismatches"],
                  "first_mismatch": run["first_mismatch"]})
        gate.add(f"no_rejections:{run['id']}",
                 all(count == 0 for count in run["rejections"].values()),
                 run["rejections"])
        gate.add(f"bomb_always_zero:{run['id']}", run["bomb_nonzero"] == 0,
                 {"bomb_nonzero": run["bomb_nonzero"]})
    coverage = test_coverage(repo)
    gate.add("every_rejection_class_covered_by_a_unit_test",
             coverage["present"] and not coverage["classes_missing"],
             {"file": coverage["path"], "sha256": coverage["sha256"],
              "found": coverage["classes_found"],
              "missing": coverage["classes_missing"]})
    samples = synthetic_rejection_samples()
    gate.add("synthetic_samples_land_in_their_class",
             all(entry["as_expected_class"] for entry in samples.values()),
             {name: entry["classification"] for name, entry in samples.items()})
    gate.add("shoot_tolerance_is_narrow",
             samples["split_non_shoot_with_shoot_change"]["classification"]
             == "split_non_shoot"
             and samples["reverse_release"]["classification"] == "reverse_release"
             and samples["release_not_due"]["classification"] == "release_not_due",
             {"direction_plus_shoot_change":
                  samples["split_non_shoot_with_shoot_change"]["classification"],
              "reverse_release": samples["reverse_release"]["classification"],
              "release_not_due": samples["release_not_due"]["classification"],
              "note": "the only tolerance is an exact equality against m & ~SHOOT; "
                      "a direction change plus a SHOOT change stays a rejection"})

    elapsed = time.perf_counter() - started
    payload = {
        "schema": SCHEMA_ADAPTER,
        "generated_by": "tools/rl/v8_31_followup.py adapter",
        "created_at": now_iso(),
        "classification": "offline",
        "repo": repo.as_posix(),
        "argv": list(getattr(args, "argv", [])) or None,
        "host": host_identity(),
        "heads": git_identity(repo),
        "tool": {"path": repo_rel(TOOL_PATH, repo), "sha256": sha256_file(TOOL_PATH),
                 "bytes": TOOL_PATH.stat().st_size},
        "imports": imports,
        "support_files": support_file_identity(repo),
        "action_table": {
            "n_cand": int(adapter()["N_CAND"]),
            "moves_per_cand": int(adapter()["MOVES_PER_CAND"]),
            "agent_mask": int(adapter()["AGENT_MASK"]),
            "direction_mask": int(adapter()["DIRECTION_MASK"]),
            "cand_masks": [int(adapter()["cand_to_mask"](cand))
                           for cand in range(int(adapter()["N_CAND"]))],
            "classification": "measured",
        },
        "decode_rule": {
            "pulse_period_decisions": DEFAULT_PULSE_PERIOD,
            "decision_frames": 2,
            "main_criterion": "bit-for-bit equality with "
                              "expand(commit_frames(cand, decision_frames=2, "
                              "release_shoot=release_due(decision_index, 48)))",
            "tolerance": "pair == [m, m & ~SHOOT] with mask_to_cand(m).bomb == 0 "
                         "and release_due true -> the pulse release; the STEP "
                         "masks are not modified",
            "accepted_via_release_definition":
                "an accepted decision whose legality requires the SHOOT pulse "
                "release, i.e. its pair differs from the release-free commit; "
                "`tolerance_uses` counts the fallback branch separately",
            "rejection_classes": list(REJECTION_CLASSES),
            "extra_class_note":
                "release_missing is added to the ticket's list because the only "
                "hole left (release due, equal masks, SHOOT held) is not a split "
                "and must not be reported as split_non_shoot; it is a named "
                "class with its own counter and 0 occurrences on the four corpora",
            "classification": "offline",
        },
        "runs": runs,
        "summary": summary,
        "t04_literal_reading": t04_literal,
        "samples": {
            "accepted": [entry for run in runs for entry in run["samples"]][:3],
            "per_run": {run["id"]: run["samples"] for run in runs},
            "rejections": samples,
        },
        "test_coverage": coverage,
        "cost": {"seconds": round(elapsed, 3)},
        "gate": gate.as_dict(),
    }
    path = write_json(out_dir / "adapter.json", payload)
    _print_adapter_summary(payload)
    print(f"wrote: {path}  ({elapsed:.2f}s)  gate={'passed' if gate.passed else 'FAILED'}")
    return EXIT_OK if gate.passed else EXIT_GATE_FAILED


def _print_adapter_summary(payload: Mapping[str, Any]) -> None:
    print(f"{'run':22s} {'decisions':>9s} {'accepted':>8s} {'release':>7s} "
          f"{'mismatch':>8s} {'refuse':>6s} {'rejections':>10s}")
    print("-" * 84)
    for run in payload["runs"]:
        rejected = sum(run["rejections"].values())
        print(f"{run['id']:22s} {run['decisions']:9d} {run['accepted']:8d} "
              f"{run['accepted_via_release']:7d} "
              f"{run['expected_mask_mismatches']:8d} "
              f"{run['pair_cand_would_refuse']:6d} {rejected:10d}")
    literal = payload["t04_literal_reading"]
    print(f"{literal['id']:22s} {literal['decisions']:9d} {literal['accepted']:8d} "
          f"{literal['accepted_via_release']:7d} "
          f"{literal['expected_mask_mismatches']:8d} "
          f"{literal['pair_cand_would_refuse']:6d} "
          f"{sum(literal['rejections'].values()):10d}")
    print()
    for name, entry in sorted(payload["samples"]["rejections"].items()):
        print(f"  sample {name:38s} -> {entry['classification']}"
              f" (index {entry['decision_index']})")


# ---------------------------------------------------------------------------
# 02-A.3 / 02-A.4: the two engine runs, their artefacts and the comparison
# ---------------------------------------------------------------------------
class EngineRefused(RuntimeError):
    """A pre-start check failed; the engine was never spawned."""


class FrameRelationConflict(RuntimeError):
    """A capture cannot be tied to a frame the ``STEP`` actually drew."""


#: Guard against an ``xvfb-run`` re-exec loop.  A separate variable from the
#: frozen slice tool's ``V831_SLICE_XVFB``: this tool re-executes *itself*, and
#: ``v8_31_slice.maybe_reexec_under_xvfb`` would re-execute ``v8_31_slice.py``
#: (its own ``__file__``) instead, which drops the subcommand and exits 2.
XVFB_GUARD_ENV = "V831_FOLLOWUP_XVFB"


def maybe_reexec_under_xvfb(*, enabled: bool) -> None:
    """One-shot ``os.execvpe`` into ``xvfb-run -a <python> <this file> <argv>``.

    The ``env`` role needs a private X server (WSLg's ``DISPLAY=:0`` hangs before
    ``HELLO``), so every engine package re-executes itself under ``xvfb-run``
    unless a non-default display is already in use, the guard variable says this
    is the child, or ``--no-xvfb`` was given.  The same rule and the same env
    additions as ``tools/rl/v8_31_slice.py`` — only the script path differs.
    """

    if not enabled:
        return
    if os.environ.get(XVFB_GUARD_ENV) == "1":
        return
    if os.environ.get("V831_SLICE_XVFB") == "1":
        return
    display = os.environ.get("DISPLAY", "")
    if display and display != ":0":
        return
    executable = shutil.which("xvfb-run")
    if not executable:
        raise EngineRefused(
            "xvfb-run is not installed, and the env role needs a private X "
            "server (WSLg's DISPLAY=:0 hangs before HELLO); pass --no-xvfb only "
            "if the current display is known to be usable")
    argv = [executable, "-a", sys.executable, str(TOOL_PATH), *sys.argv[1:]]
    env = dict(os.environ)
    env[XVFB_GUARD_ENV] = "1"
    env["SDL_AUDIODRIVER"] = "dummy"
    os.execvpe(executable, argv, env)


_SLICE: dict[str, Any] = {}


def slice_tool() -> Any:
    """Import ``tools/rl/v8_31_slice.py`` once and record its identity.

    The frozen helper module is imported, never copied: the mute helpers, the
    shared-block capture, ``ActorInputTap`` and ``CostMeter`` are the *same
    objects* the V831-SLICE-01/01b/01c runs used, and the sha256 of that file is
    written into every artefact so a later reader can tell whether the helper
    moved between two artifacts.
    """

    if _SLICE:
        return _SLICE["module"]
    tools_dir = str(Path(__file__).resolve().parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    module = importlib.import_module("v8_31_slice")
    path = Path(module.__file__).resolve()
    _SLICE.update({"module": module, "file": path, "sha256": sha256_file(path)})
    return module


def slice_identity() -> dict[str, Any]:
    """``{path, sha256}`` of the imported helper module (imports it if needed)."""

    slice_tool()
    return {"path": _SLICE["file"].as_posix(), "sha256": _SLICE["sha256"],
            "import": "import v8_31_slice as S31 (sys.path += tools/rl)",
            "classification": "measured"}


# -- the D1 replay plan ------------------------------------------------------
def step_masks_from_d1(flat: Mapping[int, int], last_frame: int) -> dict[int, int]:
    """Map **STEP frame number** -> the D1 mask that step applies.

    D1 stores one flat ``step_frames`` list per row, indexed from frame 0, and the
    mask at flat index ``j`` is the one the executor sent for the transition into
    frame ``j + 1``: row 0 carries masks for frames 0..3 and its ``hash_post``
    sits at frame 4, and row 2,165 (``decision_frame`` 8,660) puts ``hash_post``
    at 8,664.  So the ``STEP`` that ends on frame ``f`` sends flat mask ``f - 1``,
    and the seven-step window 8,659..8,665 expands to flat masks 8,658..8,664.
    """

    out: dict[int, int] = {}
    for frame in range(1, int(last_frame) + 1):
        mask = flat.get(frame - 1)
        if mask is None:
            raise SourceError(f"D1 has no recorded mask for STEP frame {frame}")
        out[frame] = int(mask)
    return out


def mask_trace_digest(step_masks: Mapping[int, int], first_frame: int,
                      last_frame: int) -> str:
    """sha256 of the per-frame mask trace ``"<frame>:<mask>\\n"``.

    This is the digest that must survive a change of ``STEP`` *granularity*: one
    ``STEP mask 2`` and two ``STEP mask 1`` units put exactly the same mask on the
    same two frames, so both runs expand to the same frame -> mask sequence and
    the same digest.
    """

    payload = "".join(f"{frame}:{int(step_masks[frame])}\n"
                      for frame in range(int(first_frame), int(last_frame) + 1))
    return sha256_bytes(payload.encode("ascii"))


def plan_prefix_steps(step_masks: Mapping[int, int],
                      last_frame: int) -> list[dict[str, Any]]:
    """One ``STEP <mask> 1`` per frame, so the recorded frame numbering is exact."""

    return [{"start": frame, "frames": 1, "mask": int(step_masks[frame]),
             "granularity": "mask1", "forced_split": False,
             "decision_index": (frame - 1) // DECISION_FRAMES}
            for frame in range(1, int(last_frame) + 1)]


def plan_reference_steps(step_masks: Mapping[int, int], last_frame: int,
                         forced_frames: Iterable[int]) -> list[dict[str, Any]]:
    """The independent granularity: ``STEP mask 2`` when a decision's masks agree.

    Decision ``k`` is the STEP-frame pair ``(2k+1, 2k+2)`` -- its two masks are
    D1 flat ``2k``/``2k+1`` -- so a unit never spans two decisions.  When both
    masks are equal the pair is sent as **one** ``STEP mask 2``, otherwise as two
    ``STEP mask 1``.  A frame in ``forced_frames`` has to be the end of its own
    ``STEP`` (the capture window reads the shared block once per frame), so a
    would-be two-frame unit that touches one is split and flagged
    ``forced_split``.  A trailing half decision becomes a one-frame unit with
    ``partial_decision`` set, which is what makes the 7th window frame (pixel
    frame 8,664, readable only after ``STEP`` 8,665) reachable.
    """

    forced = {int(frame) for frame in forced_frames}
    units: list[dict[str, Any]] = []
    decisions = (int(last_frame) + DECISION_FRAMES - 1) // DECISION_FRAMES
    for decision_index in range(decisions + 1):
        first = decision_index * DECISION_FRAMES + 1
        second = first + 1
        if first > int(last_frame):
            break
        if second > int(last_frame):
            units.append({"start": first, "frames": 1,
                          "mask": int(step_masks[first]), "granularity": "mask1",
                          "forced_split": False, "partial_decision": True,
                          "decision_index": decision_index})
            break
        equal = int(step_masks[first]) == int(step_masks[second])
        blocked = first in forced or second in forced
        if equal and not blocked:
            units.append({"start": first, "frames": 2,
                          "mask": int(step_masks[first]),
                          "granularity": "mask2_equal_pair",
                          "forced_split": False,
                          "decision_index": decision_index})
        else:
            for frame in (first, second):
                units.append({"start": frame, "frames": 1,
                              "mask": int(step_masks[frame]),
                              "granularity": "mask1",
                              "forced_split": bool(equal),
                              "decision_index": decision_index})
    return units


def unit_end_frames(units: Sequence[Mapping[str, Any]]) -> list[int]:
    """The frames each unit ends on, in order (one entry per ``STEP``)."""

    return [int(unit["start"]) + int(unit["frames"]) - 1 for unit in units]


def plan_summary(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count the granularity mix of a step plan (never a guess about it)."""

    frames = unit_end_frames(units)
    mask2 = sum(1 for unit in units if int(unit["frames"]) == 2)
    mask1 = len(units) - mask2
    return {
        "steps": len(units),
        "frames": sum(int(unit["frames"]) for unit in units),
        "first_frame": int(units[0]["start"]) if units else None,
        "last_frame": frames[-1] if frames else None,
        "mask2_steps": mask2,
        "mask1_steps": mask1,
        "forced_splits": sum(1 for unit in units if unit.get("forced_split")),
        "partial_decision_units": sum(1 for unit in units
                                      if unit.get("partial_decision")),
        "distinct_masks": len({int(unit["mask"]) for unit in units}),
        "only_units_of_1_or_2_frames": all(
            int(unit["frames"]) in (1, 2) for unit in units),
        "classification": "measured",
    }


# -- PNG: deterministic bytes, so two runs can be compared byte for byte -----
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


def png_bytes(rgba_top_down: bytes, width: int, height: int) -> bytes:
    """Encode RGBA8 rows as a deterministic PNG (filter 0, no timestamp).

    Deterministic on purpose: the two runs must write the *same bytes* for the
    same picture, so no ancillary chunk, no filter heuristic and one fixed
    compression level are used.
    """

    stride = int(width) * 4
    data = bytes(rgba_top_down)
    if len(data) != stride * int(height):
        raise ValueError(f"RGBA payload is {len(data)} bytes, expected "
                         f"{stride * int(height)} for {width}x{height}")
    raw = bytearray()
    for row in range(int(height)):
        raw.append(0)
        raw += data[row * stride:(row + 1) * stride]
    header = struct.pack(">IIBBBBB", int(width), int(height), 8, 6, 0, 0, 0)
    return (PNG_SIGNATURE + _png_chunk(b"IHDR", header)
            + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + _png_chunk(b"IEND", b""))


def read_png(data: bytes) -> tuple[int, int, bytes]:
    """Decode the PNG this tool writes: ``(width, height, RGBA top-down rows)``.

    Used by ``compare`` and ``verify`` to re-derive a frame's digest **from the
    file on disk**, so the comparison never trusts a recorded number.
    """

    if not bytes(data).startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG (bad signature)")
    offset = len(PNG_SIGNATURE)
    width = height = 0
    idat = bytearray()
    seen_end = False
    while offset + 8 <= len(data):
        (length,) = struct.unpack(">I", data[offset:offset + 4])
        tag = bytes(data[offset + 4:offset + 8])
        payload = bytes(data[offset + 8:offset + 8 + length])
        (crc_expected,) = struct.unpack(
            ">I", data[offset + 8 + length:offset + 12 + length])
        if zlib.crc32(tag + payload) & 0xFFFFFFFF != crc_expected:
            raise ValueError(f"PNG chunk {tag!r} has a bad CRC")
        offset += 12 + length
        if tag == b"IHDR":
            width, height, depth, color, comp, filt, inter = struct.unpack(
                ">IIBBBBB", payload)
            if (depth, color, comp, filt, inter) != (8, 6, 0, 0, 0):
                raise ValueError(
                    f"unsupported PNG: depth={depth} color={color} "
                    f"compression={comp} filter={filt} interlace={inter}")
        elif tag == b"IDAT":
            idat += payload
        elif tag == b"IEND":
            seen_end = True
            break
    if not width or not height or not idat or not seen_end:
        raise ValueError("PNG is missing IHDR/IDAT/IEND")
    raw = zlib.decompress(bytes(idat))
    stride = int(width) * 4
    if len(raw) != (stride + 1) * int(height):
        raise ValueError(f"PNG IDAT decodes to {len(raw)} bytes, expected "
                         f"{(stride + 1) * int(height)}")
    out = bytearray()
    previous = bytearray(stride)
    for row in range(int(height)):
        start = row * (stride + 1)
        method = raw[start]
        line = bytearray(raw[start + 1:start + 1 + stride])
        if method == 1:
            for index in range(4, stride):
                line[index] = (line[index] + line[index - 4]) & 0xFF
        elif method == 2:
            for index in range(stride):
                line[index] = (line[index] + previous[index]) & 0xFF
        elif method == 3:
            for index in range(stride):
                left = line[index - 4] if index >= 4 else 0
                line[index] = (line[index] + ((left + previous[index]) >> 1)) & 0xFF
        elif method == 4:
            for index in range(stride):
                left = line[index - 4] if index >= 4 else 0
                up = previous[index]
                upper_left = previous[index - 4] if index >= 4 else 0
                estimate = left + up - upper_left
                pa = abs(estimate - left)
                pb = abs(estimate - up)
                pc = abs(estimate - upper_left)
                if pa <= pb and pa <= pc:
                    predictor = left
                elif pb <= pc:
                    predictor = up
                else:
                    predictor = upper_left
                line[index] = (line[index] + predictor) & 0xFF
        elif method != 0:
            raise ValueError(f"PNG row {row} uses filter {method}")
        out += line
        previous = line
    return int(width), int(height), bytes(out)


def raw_order_bytes(rgba_top_down: bytes, width: int, height: int) -> bytes:
    """Flip top-down rows back to the engine's bottom-up byte order."""

    stride = int(width) * 4
    rows = [bytes(rgba_top_down[row * stride:(row + 1) * stride])
            for row in range(int(height))]
    return b"".join(reversed(rows))


def frame_digests_from_png(path: str | Path) -> dict[str, Any]:
    """Re-derive a stored frame's identities from the PNG file alone."""

    data = Path(path).read_bytes()
    width, height, rgba = read_png(data)
    raw = raw_order_bytes(rgba, width, height)
    return {
        "path": Path(path).as_posix(),
        "bytes": len(data),
        "png_sha256": sha256_bytes(data),
        "width": width,
        "height": height,
        "rgba_top_down_sha256": sha256_bytes(rgba),
        "sha256_raw": sha256_bytes(raw),
        "blake2b128": hashlib.blake2b(raw, digest_size=16).hexdigest(),
        "classification": "offline",
    }


# -- PLAYER: the only command that carries px / py ---------------------------
PLAYER_FLOAT_FIELDS: tuple[str, ...] = ("hp", "bombs", "power", "px", "py")
PLAYER_BOX_FIELDS: tuple[str, ...] = (
    "hurtboxMinX", "hurtboxMinY", "hurtboxMaxX", "hurtboxMaxY")


def parse_player_reply(reply: str) -> dict[str, Any]:
    """Decode one ``PLAYER`` reply: position, hitbox, resources, speed files.

    ``px``/``py``/``hp``/``bombs``/``power`` and the four speed constants arrive
    as ``%08x`` float bit patterns; both the float and the raw bits are kept so a
    comparison never depends on a re-rounding.  ``STATE`` also carries
    ``player_x``/``player_y`` (fields f10/f11) on this build, so the caller
    cross-checks the two and reports the pair instead of choosing silently.
    """

    from touhou_rl.v400.labels.state import f32_from_bits  # noqa: PLC0415

    fields = _parse_kv(reply)
    if fields.get("_verb") != "OK":
        raise EngineRefused(f"PLAYER reply is not OK: {reply!r}")

    def bits(name: str) -> int:
        if name not in fields:
            raise EngineRefused(f"PLAYER reply has no {name}: {reply!r}")
        return int(fields[name], 16)

    def integer(name: str) -> int:
        if name not in fields:
            raise EngineRefused(f"PLAYER reply has no {name}: {reply!r}")
        return int(fields[name])

    decoded = {name: f32_from_bits(bits(name)) for name in PLAYER_FLOAT_FIELDS}
    box = {name: f32_from_bits(bits(name)) for name in PLAYER_BOX_FIELDS}
    speeds = {name: f32_from_bits(bits(name)) for name in (
        "speedNormalAxial", "speedNormalDiagonal",
        "speedFocusedAxial", "speedFocusedDiagonal")}
    from touhou_rl.v400.labels.state import player_state_name  # noqa: PLC0415

    return {
        "raw": reply,
        "frames": integer("frames"),
        "px": decoded["px"],
        "py": decoded["py"],
        "hp": decoded["hp"],
        "lives": integer("lives"),
        "bombs": decoded["bombs"],
        "power": decoded["power"],
        "focus": integer("focus"),
        "is_youkai": integer("isYoukai"),
        "player_state": integer("playerState"),
        "player_state_name": player_state_name(integer("playerState")),
        "invuln_state": integer("invulnState"),
        "raw_bits": {name: bits(name) for name in
                     PLAYER_FLOAT_FIELDS + PLAYER_BOX_FIELDS},
        "hurtbox": box,
        "speed": speeds,
        "position_source": "PLAYER px/py",
    }


def resource_record(*, frame: int, state: Mapping[str, Any],
                    player: Mapping[str, Any], hud: Mapping[str, Any]) -> dict[str, Any]:
    """One resource record, keyed by frame, with both position sources.

    ``STATE`` has no ``x``/``y`` *columns* in the older dumps this ticket
    describes, so the position is taken from ``PLAYER``; on this build ``STATE``
    also carries f10/f11, and the two are recorded side by side rather than
    merged so a disagreement stays visible.
    """

    state_position = {"x": state.get("player_x"), "y": state.get("player_y")}
    same = (
        isinstance(state_position["x"], float) and isinstance(state_position["y"], float)
        and abs(float(state_position["x"]) - float(player["px"])) < 1e-6
        and abs(float(state_position["y"]) - float(player["py"])) < 1e-6)
    return {
        "frame": int(frame),
        "position": {"x": float(player["px"]), "y": float(player["py"]),
                     "source": "PLAYER px/py"},
        "state_position": state_position,
        "state_position_source": "STATE f10/f11 (present on rl_v3)",
        "position_sources_agree": bool(same),
        "resources": {
            "hp": float(player["hp"]),
            "lives": int(player["lives"]),
            "bombs": float(player["bombs"]),
            "power": float(player["power"]),
            "focus": int(player["focus"]),
            "is_youkai": int(player["is_youkai"]),
            "player_state": int(player["player_state"]),
            "player_state_name": player["player_state_name"],
            "score": int(state["score"]),
            "graze": int(state["graze"]),
            "deaths": float(state["deaths"]),
            "stage": int(state["stage"]),
            "gameplay_frame": int(state["gameplay_frame"]),
        },
        "hud": {
            "frame": int(hud["frame"]),
            "display_score": int(hud["display_score"]),
            "globals_score": int(hud["globals_score"]),
            "lives": int(hud["lives"]),
            "bombs": int(hud["bombs"]),
            "power": int(hud["power"]),
            "graze": int(hud["graze"]),
            "time": int(hud["time"]),
            "game_over": int(hud["game_over"]),
        },
        "state": {key: state[key] for key in sorted(state)},
        "player": {key: value for key, value in player.items() if key != "raw"},
        "classification": "measured",
    }


# -- anchor comparison -------------------------------------------------------
def anchor_report(*, expected: Mapping[int, str], actual: Mapping[int, str],
                  first_frame: int, last_frame: int, label: str,
                  rows_kept: int = 24) -> dict[str, Any]:
    """Compare every recorded D1 hash anchor inside ``[first_frame, last_frame]``.

    ``expected`` is the D1 anchor map (frame -> 16 hex digits), ``actual`` the
    hash each ``STEP`` returned.  The function never drops a frame it was asked
    about: a frame missing from ``actual`` is a mismatch with ``actual: null``,
    not a smaller denominator.
    """

    frames = sorted(int(frame) for frame in expected
                    if int(first_frame) <= int(frame) <= int(last_frame))
    rows: list[dict[str, Any]] = []
    for frame in frames:
        want = str(expected[frame]).strip().lower()
        got = actual.get(frame)
        got = None if got is None else str(got).strip().lower()
        rows.append({"frame": frame, "expected": want, "actual": got,
                     "ok": got is not None and got == want})
    mismatches = [row for row in rows if not row["ok"]]
    return {
        "label": label,
        "expected": len(frames),
        "compared": len(rows),
        "matched": len(rows) - len(mismatches),
        "mismatches": len(mismatches),
        "missing_actual": sum(1 for row in mismatches if row["actual"] is None),
        "first_mismatch_frame": mismatches[0]["frame"] if mismatches else None,
        "first_mismatch": mismatches[0] if mismatches else None,
        "coverage": {
            "requested_interval": [int(first_frame), int(last_frame)],
            "compared_first_frame": frames[0] if frames else None,
            "compared_last_frame": frames[-1] if frames else None,
            "anchors_in_source": len(expected),
            "all_source_anchors_inside_interval": all(
                int(first_frame) <= int(frame) <= int(last_frame)
                for frame in expected),
        },
        "rows": rows[:int(rows_kept)],
        "rows_kept": min(len(rows), int(rows_kept)),
        "rows_truncated": len(rows) > int(rows_kept),
        "classification": "measured",
    }


# -- filesystem helpers ------------------------------------------------------
SOURCE_HASH_FILES: tuple[str, ...] = ("th08.cfg", "th08.dat", "thbgm.dat")


def tree_fingerprint(root: str | Path) -> dict[str, Any]:
    """``(relative path, size, mtime_ns)`` for every file under ``root``.

    The read-only source game directory is fingerprinted before and after the run
    and the three data files are re-hashed on both sides, so "the source was not
    written" is a measurement, not an assumption.
    """

    root_path = Path(root)
    entries: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames.sort()
        for name in sorted(filenames):
            full = Path(dirpath) / name
            rel = full.relative_to(root_path).as_posix()
            try:
                info = full.stat()
            except OSError as exc:  # a broken symlink is data, not a crash
                entries.append({"rel": rel, "error": f"{type(exc).__name__}: {exc}"})
                continue
            entries.append({"rel": rel, "bytes": int(info.st_size),
                            "mtime_ns": int(info.st_mtime_ns)})
    hashes: dict[str, str | None] = {}
    for name in SOURCE_HASH_FILES:
        target = root_path / name
        hashes[name] = sha256_file(target) if target.is_file() else None
    return {
        "root": root_path.as_posix(),
        "files": len(entries),
        "entries": entries,
        "digest": sha256_bytes(json.dumps(entries, sort_keys=True).encode()),
        "file_sha256": hashes,
        "classification": "measured",
    }


def du_bytes(root: str | Path) -> int:
    """Sum of the file sizes under ``root`` (the ledger's ``write_gib`` input)."""

    total = 0
    for dirpath, _dirnames, filenames in os.walk(Path(root)):
        for name in filenames:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                continue
    return int(total)


# -- preflight, mute, private game dir --------------------------------------
def engine_preflight(args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    """Every identity check that must pass before a single frame is stepped."""

    module = slice_tool()
    engine = Path(args.engine).expanduser()
    if not engine.is_file():
        raise EngineRefused(f"engine binary not found: {engine}")
    engine_sha = sha256_file(engine)
    registered_engine = str(module.REGISTERED_ENGINE_SHA256)
    if engine_sha != registered_engine:
        raise EngineRefused(
            f"engine sha256 {engine_sha} != registered {registered_engine}; this "
            "ticket runs the registered rl_v3 build only")
    obs_digest = str(module.obs_spec_v3.obs_spec_v3_digest())
    registered_obs = str(module.REGISTERED_OBS_V3_DIGEST)
    if obs_digest != registered_obs:
        raise EngineRefused(
            f"obs-v3 digest {obs_digest} != registered {registered_obs}")

    data_source = Path(args.data_source).expanduser().resolve()
    if not data_source.is_dir():
        raise EngineRefused(f"--data-source is not a directory: {data_source}")
    if data_source == out_root or out_root in data_source.parents:
        raise EngineRefused(
            f"--data-source {data_source} points inside the run directory; the "
            "source must be a separate read-only input")
    for required in SOURCE_HASH_FILES:
        if not (data_source / required).is_file():
            raise EngineRefused(
                f"--data-source {data_source} lacks {required}; the private copy "
                "must reproduce the whole game directory")
    source_cfg = module.decode_cfg_file(data_source / "th08.cfg")
    if source_cfg.get("error"):
        raise EngineRefused(
            f"source th08.cfg cannot be decoded: {source_cfg['error']}")
    difficulty = str(RESET_DIFFICULTY)
    difficulty_table = list(module.DIFFICULTIES)
    if difficulty not in difficulty_table:
        raise EngineRefused(f"{difficulty!r} not in {difficulty_table}")
    space = shutil.disk_usage(str(out_root.parent if out_root.parent.exists()
                                  else REPO_ROOT))
    return {
        "engine": engine.as_posix(),
        "engine_sha256": engine_sha,
        "engine_sha256_registered": registered_engine,
        "engine_sha256_matches": True,
        "obs_spec_version": str(module.obs_spec_v3.OBS_SPEC_VERSION),
        "obs_spec_digest": obs_digest,
        "obs_spec_digest_registered": registered_obs,
        "obs_spec_digest_matches": True,
        "data_source": data_source.as_posix(),
        "data_source_cfg": source_cfg,
        "source_cfg_registered_sha256": str(module.REGISTERED_SOURCE_CFG_SHA256),
        "source_cfg_matches_registered": bool(
            source_cfg.get("sha256") == str(module.REGISTERED_SOURCE_CFG_SHA256)),
        "reset": {"stage": RESET_STAGE, "difficulty": difficulty,
                  "difficulty_index": difficulty_table.index(difficulty),
                  "team": RESET_TEAM, "seed": RESET_SEED,
                  "command": f"RESET {RESET_STAGE} "
                             f"{difficulty_table.index(difficulty)} "
                             f"{RESET_TEAM} {RESET_SEED}"},
        "decision_frames": DECISION_FRAMES,
        "pulse_period": DEFAULT_PULSE_PERIOD,
        "disk": {"total_bytes": int(space.total), "used_bytes": int(space.used),
                 "free_bytes": int(space.free), "checked_at": now_iso()},
        "python": {"executable": sys.executable,
                   "version": sys.version.split()[0]},
        "classification": "measured",
    }


def prepare_private_game_dir(*, out_dir: Path, data_source: Path,
                             meter: Any) -> dict[str, Any]:
    """Copy the whole tree and mute the **copy** through the frozen helper.

    The source is only read.  The four mute bytes are then re-checked on the raw
    bytes of the copy (not only through the decoded struct), which is what the
    hard rule in ``AGENTS.md`` §9 asks for.
    """

    module = _SLICE["module"]
    if (out_dir / "game").exists():
        raise EngineRefused(
            f"private game directory already exists: {out_dir / 'game'}; every "
            "run needs its own fresh .tmp_v831_followup_02a_* directory")
    info = module.prepare_private_game_dir(
        out_dir=out_dir, data_source=data_source, meter=meter)
    copy_raw = bytes(info["copy_raw"])
    offsets = [int(offset) for offset in module.MUTE_OFFSETS]
    zeros = {f"0x{offset:02X}": int(copy_raw[offset]) for offset in offsets}
    info["mute_byte_check"] = {
        "offsets": [f"0x{offset:02X}" for offset in offsets],
        "values": zeros,
        "all_zero": all(value == 0 for value in zeros.values()),
        "copied_cfg_bytes": len(copy_raw),
        "classification": "measured",
    }
    info.pop("source_raw", None)
    info.pop("copy_raw", None)
    return info


def _mute_gate(report: Mapping[str, Any]) -> list[str]:
    """The pre-start mute predicates, as problems (empty == safe to start)."""

    module = _SLICE["module"]
    problems: list[str] = []
    for message in module.mute_report_problems(report):
        if "after the process exited" in message:
            continue  # not yet knowable before the first start
        problems.append(message)
    before = report.get("copy_before_start") or {}
    if not before.get("muted_strict_four_field"):
        problems.append(
            "copy cfg is not strictly muted before start: "
            f"{before.get('fields')}")
    for name in ("musicMode", "playSounds", "musicVolume", "sfxVolume"):
        value = (before.get("fields") or {}).get(name)
        if value != 0:
            problems.append(f"copy cfg {name}={value!r} before start, expected 0")
    return problems


# -- the engine packages -----------------------------------------------------
def _package_paths(out_root: Path, mode: str) -> dict[str, Path]:
    base = out_root / "02a" / mode
    return {
        "base": base,
        "result": base / "result.json",
        "frames": base / "frames.jsonl",
        "history": base / "history.jsonl",
        "hashes": base / "hashes.jsonl",
        "takeover": base / "takeover.json",
        "mute": base / "mute.json",
        "frames_dir": base / "frames",
        "game": base / "game",
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write one JSON object per line; return the row count."""

    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            write_jsonl_row(handle, dict(row))
            count += 1
    return count


def _classify_capture(step_frame: int, reply: Mapping[str, Any],
                      capture: Mapping[str, Any], events: Mapping[str, Any]
                      ) -> dict[str, Any]:
    """Tie one shared-block read to the frame the ``STEP`` actually drew."""

    module = _SLICE["module"]
    events_frames = int(events["frames"])
    step_frame_reply = int(reply.get("frame", -1))
    relation = module.classify_captured_frame(int(capture["header_frame"]),
                                             events_frames)
    if relation is None:
        raise FrameRelationConflict(module.pixel_relation_reason(
            int(capture["header_frame"]), events_frames))
    if step_frame_reply != events_frames:
        raise FrameRelationConflict(
            f"STEP frame={step_frame_reply} but EVENTS frames={events_frames}")
    if step_frame_reply != int(step_frame):
        raise FrameRelationConflict(
            f"STEP ended on frame {step_frame_reply}, planned {step_frame}")
    if int(relation["pixel_frame"]) != int(step_frame) - 1:
        raise FrameRelationConflict(
            f"EVENTS frames={events_frames} implies pixel frame "
            f"{relation['pixel_frame']}, planned {int(step_frame) - 1}")
    return dict(relation)


def _decode_decision(pair: Sequence[int], decision_index: int) -> dict[str, Any]:
    """The adapter's verdict on one decision (the *only* pulse-phase decoder)."""

    cand, bomb, classification, detail = decode_pair(
        list(pair), int(decision_index), DEFAULT_PULSE_PERIOD)
    return {
        "decision_index": int(decision_index),
        "pair": [int(pair[0]), int(pair[1])],
        "cand": None if cand is None else int(cand),
        "bomb": None if bomb is None else int(bomb),
        "classification": classification,
        "accepted": classification in ACCEPTED,
        "release_due": bool(detail.get("release_due")),
        "expected": detail.get("expected"),
        "rule": detail.get("rule"),
    }


def _decode_decision_direct(pair: Sequence[int],
                           decision_index: int | None = None) -> dict[str, Any]:
    """An *independent* decode: ``mask_to_cand`` on the first mask, nothing else.

    The reference run must not reuse :func:`decode_pair`: the ticket's own rule is
    "each two frames decodes its legal candidate and bomb from the **first**
    frame", so the reference applies exactly that and lets ``compare`` find any
    difference against the adapter's verdict.  ``decision_index`` is recorded only
    for traceability and never changes the decode.
    """

    table = adapter()
    try:
        cand, bomb = table["mask_to_cand"](int(pair[0]))
    except ValueError as exc:
        return {"pair": [int(pair[0]), int(pair[1])], "cand": None, "bomb": None,
                "classification": f"mask_to_cand_failed: {exc}",
                "accepted": False, "release_due": None, "expected": None,
                "rule": "mask_to_cand(first mask)",
                "decision_index": decision_index}
    return {
        "pair": [int(pair[0]), int(pair[1])],
        "cand": int(cand),
        "bomb": int(bomb),
        "classification": "decoded_from_first_mask",
        "accepted": True,
        "release_due": None,
        "expected": None,
        "rule": "mask_to_cand(first mask); no expected-mask expansion",
        "decision_index": decision_index,
    }


# ---------------------------------------------------------------------------
# `verify`, `run`, and the not-implemented packages
# ---------------------------------------------------------------------------
ENGINE_PACKAGES: tuple[str, ...] = ("prefix", "reference")


def verify_offline_pair(sources: Mapping[str, Any], adapter_json: Mapping[str, Any],
                        repo: Path, gate: Gate) -> dict[str, Any]:
    """Re-derive the offline 02-A.1/02-A.2 gates from the raw corpora."""

    tool = sources.get("tool") or {}
    # Recorded, **not gated**: this instrument is still being extended for the
    # later packages of the same ticket, and every number 02-A reports is
    # re-derived below from the raw corpora (sha256, counters, decode, imports),
    # so a later revision of this file cannot weaken the gate.  The engine
    # packages have reported the same value under `tool_revision` for the same
    # reason since the first revision.  A different value here means the artefacts
    # were written by an earlier revision, nothing more.
    gate.add("tool_revision_recorded", bool(tool.get("sha256")),
             {"recorded": tool.get("sha256"),
              "current": sha256_file(TOOL_PATH),
              "same": tool.get("sha256") == sha256_file(TOOL_PATH),
              "note": "recorded only; the raw predicates below are recomputed"})

    re_hash: list[dict[str, Any]] = []
    for entry in sources.get("sources", []):
        path = repo / entry["path"]
        measured = sha256_file(path) if path.is_file() else None
        re_hash.append({"id": entry["id"], "path": entry["path"],
                        "recorded": entry["sha256"], "recomputed": measured,
                        "ok": measured == entry["sha256"]})
    gate.add("source_sha256_recomputed",
             bool(re_hash) and all(item["ok"] for item in re_hash), re_hash)

    recount: list[dict[str, Any]] = []
    for entry in sources.get("sources", []):
        spec = SOURCE_BY_ID.get(entry["id"])
        if spec is None:
            recount.append({"id": entry["id"], "ok": False,
                            "error": "not a registered source id"})
            continue
        model = ((load_labels_source if spec["kind"] == "labels"
                  else load_slots_source)(spec, repo))
        pairs = len(model["pairs"])
        frames = int(model["last_frame"]) - int(model["first_frame"]) + 1
        hashes = len(model["hash_pre"]) + len(model["hash_post"])
        recorded = entry["counts"]
        ok = (recorded["rows"] == model["rows"]
              and recorded["decision_frames"] == pairs
              and recorded["frames"] == frames
              and entry["hash_coverage"]["hash_frames_total"] == hashes
              and entry["masks"]["distinct_masks"] == len(model["distinct_masks"])
              and entry["masks"]["rows_with_split_pair"] == model["split_rows"])
        recount.append({"id": entry["id"], "ok": ok,
                        "recorded": {"rows": recorded["rows"],
                                     "decision_frames": recorded["decision_frames"],
                                     "frames": recorded["frames"],
                                     "hash_frames_total":
                                         entry["hash_coverage"]["hash_frames_total"],
                                     "distinct_masks": entry["masks"]["distinct_masks"],
                                     "rows_with_split_pair":
                                         entry["masks"]["rows_with_split_pair"]},
                        "recomputed": {"rows": model["rows"],
                                       "decision_frames": pairs,
                                       "frames": frames,
                                       "hash_frames_total": hashes,
                                       "distinct_masks": len(model["distinct_masks"]),
                                       "rows_with_split_pair": model["split_rows"]}})
    gate.add("counters_recomputed", all(item["ok"] for item in recount), recount)

    import_recheck = []
    for entry in adapter_json.get("imports", []):
        if not entry.get("file"):
            import_recheck.append({"module": entry["module"], "ok": False,
                                   "error": "no file bound"})
            continue
        path = repo / entry["file"]
        measured = sha256_file(path) if path.is_file() else None
        import_recheck.append({"module": entry["module"], "file": entry["file"],
                               "recorded": entry.get("sha256"),
                               "recomputed": measured,
                               "ok": measured == entry.get("sha256")})
    gate.add("imports_unchanged",
             bool(import_recheck) and all(item["ok"] for item in import_recheck),
             import_recheck)

    decode_recheck: list[dict[str, Any]] = []
    for run in adapter_json.get("runs", []):
        spec = SOURCE_BY_ID.get(run["source"])
        if spec is None:
            decode_recheck.append({"id": run["id"], "ok": False,
                                   "error": "unknown source"})
            continue
        model = ((load_labels_source if spec["kind"] == "labels"
                  else load_slots_source)(spec, repo))
        cap = run.get("frame_cap")
        fresh = run_adapter(model["pairs"], frame_cap=cap)
        ok = (fresh["decisions"] == run["decisions"]
              and fresh["accepted"] == run["accepted"]
              and fresh["expected_mask_mismatches"] == 0
              and all(count == 0 for count in fresh["rejections"].values()))
        decode_recheck.append({"id": run["id"], "ok": ok,
                               "recorded": {"decisions": run["decisions"],
                                            "accepted": run["accepted"],
                                            "mismatches": run["expected_mask_mismatches"]},
                               "recomputed": {
                                   "decisions": fresh["decisions"],
                                   "accepted": fresh["accepted"],
                                   "mismatches": fresh["expected_mask_mismatches"],
                                   "rejections": fresh["rejections"]}})
    gate.add("decode_recomputed",
             bool(decode_recheck) and all(item["ok"] for item in decode_recheck),
             decode_recheck)

    gate.add("sources_gate_claimed_passed",
             bool((sources.get("gate") or {}).get("passed")), sources.get("gate"))
    gate.add("adapter_gate_claimed_passed",
             bool((adapter_json.get("gate") or {}).get("passed")),
             adapter_json.get("gate"))
    return {"re_hash": re_hash, "decode_recheck": decode_recheck}


def _history_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """sha256 of ``"<decision_index>:<cand>\\n"``: the decoded action sequence."""

    return sha256_bytes("".join(
        f"{int(row['decision_index'])}:{row.get('cand')}\n" for row in rows
    ).encode())


def verify_engine_package(out_root: Path, mode: str, gate: Gate,
                          plan: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive one engine package's gate from the raw files under ``out_root``."""

    base = out_root / "02a" / mode
    record = _load_json(base / "result.json")
    report: dict[str, Any] = {"mode": mode, "base": base.as_posix()}
    artifacts = {}
    for name in ("result.json", "frames.jsonl", "history.jsonl", "hashes.jsonl",
                 "takeover.json", "mute.json", "frames"):
        path = base / name
        artifacts[name] = {
            "present": path.exists(),
            "bytes": (path.stat().st_size if path.is_file() else
                      (len(list(path.iterdir())) if path.is_dir() else None)),
        }
    report["artifacts"] = artifacts
    gate.add(f"{mode}_artifacts_present",
             all(entry["present"] for entry in artifacts.values()), artifacts)
    gate.add(f"{mode}_gate_claimed_passed",
             bool((record.get("gate") or {}).get("passed")),
             (record.get("gate") or {}).get("passed"))
    gate.add(f"{mode}_terminal_completed",
             (record.get("terminal") or {}).get("kind") == "completed",
             record.get("terminal"))

    takeover = _load_json(base / "takeover.json")
    captures = list((takeover.get("window") or {}).get("captures") or [])
    recheck = _png_recheck(out_root, captures)
    gate.add(f"{mode}_seven_frames",
             len(captures) == 7
             and [int(row["pixel_frame"]) for row in captures]
             == list(WINDOW_PIXEL_FRAMES),
             [int(row["pixel_frame"]) for row in captures])
    gate.add(f"{mode}_pngs_match_their_records",
             bool(recheck) and all(row["ok"] for row in recheck), recheck)
    gate.add(f"{mode}_frame_relation",
             all(int(row["header_frame"]) == int(row["events_frames"])
                 and int(row["pixel_frame"]) == int(row["captured_at_step_F"]) - 1
                 for row in captures),
             [{"captured_at_step_F": row["captured_at_step_F"],
               "pixel_frame": row["pixel_frame"],
               "header_frame": row["header_frame"],
               "events_frames": row["events_frames"]} for row in captures])

    hashes = {int(row["frame"]): str(row["hash"]).strip().lower()
              for row in iter_jsonl(base / "hashes.jsonl")}
    report["hash_rows"] = len(hashes)
    pre = anchor_report(expected=plan["expected_pre"], actual=hashes, first_frame=1,
                        last_frame=PREFIX_FRAME - DECISION_FRAMES, label="verify-pre")
    post = anchor_report(expected=plan["expected_post"], actual=hashes,
                         first_frame=1, last_frame=PREFIX_FRAME, label="verify-post")
    gate.add(f"{mode}_anchors_recomputed",
             pre["mismatches"] == 0 and post["mismatches"] == 0
             and pre["expected"] == 2165 and post["expected"] == 2166,
             {"pre": {"expected": pre["expected"], "compared": pre["compared"],
                      "mismatches": pre["mismatches"],
                      "first_mismatch_frame": pre["first_mismatch_frame"]},
              "post": {"expected": post["expected"], "compared": post["compared"],
                       "mismatches": post["mismatches"],
                       "first_mismatch_frame": post["first_mismatch_frame"]}})
    report["anchors"] = {"pre": pre, "post": post}

    recorded_trace = (record.get("mask_trace") or {})
    fresh_trace = mask_trace_digest(plan["step_masks"], 1, PREFIX_FRAME)
    gate.add(f"{mode}_mask_trace_recomputed",
             fresh_trace == recorded_trace.get("digest"),
             {"recorded": recorded_trace.get("digest"), "recomputed": fresh_trace})

    history_rows = [dict(row) for row in iter_jsonl(base / "history.jsonl")]
    report["history_rows"] = len(history_rows)
    decode = _decode_decision if mode == "prefix" else _decode_decision_direct
    redecoded = [decode(row["pair"], int(row["decision_index"]))
                 for row in history_rows]
    same_cands = all(int(left.get("cand", -1)) == int(right.get("cand", -2))
                     for left, right in zip(redecoded, history_rows))
    gate.add(f"{mode}_history_recount",
             len(history_rows) == PREFIX_DECISIONS and same_cands
             and [int(row["decision_index"]) for row in history_rows]
             == list(range(PREFIX_DECISIONS)),
             {"rows": len(history_rows), "expected": PREFIX_DECISIONS,
              "cands_reproduced": same_cands})
    fresh_digest = _history_digest(history_rows)
    gate.add(f"{mode}_history_digest",
             fresh_digest == record.get("history_digest"),
             {"recomputed": fresh_digest, "recorded": record.get("history_digest")})
    actor = record.get("actor_inputs") or {}
    gate.add(f"{mode}_actor_inputs_recorded",
             bool(actor) and all(actor.get(key) for key in
                                 ("play", "hud", "act_hist", "actor_input_digest")),
             {key: actor.get(key) for key in
              ("play", "hud", "act_hist", "actor_input_digest",
               "history_entries", "next_decision_index", "frames_pumped")})
    gate.add(f"{mode}_history_entries_4332",
             int(actor.get("history_entries", -1)) == PREFIX_DECISIONS
             and int(actor.get("next_decision_index", -1)) == PREFIX_DECISIONS,
             {"history_entries": actor.get("history_entries"),
              "next_decision_index": actor.get("next_decision_index")})
    resources = record.get("resources") or {}
    gate.add(f"{mode}_resources_recorded_twice",
             sorted(resources) == sorted(str(frame) for frame in RESOURCE_FRAMES),
             sorted(resources))
    if mode == "prefix":
        expected = plan["continuation_expected"]
        actual = hashes.get(CONTINUATION_LAST_FRAME)
        gate.add("prefix_continuation_hash_recomputed",
                 actual is not None and actual == expected,
                 {"frame": CONTINUATION_LAST_FRAME, "expected": expected,
                  "actual": actual,
                  "expected_source": "D1 row 2,166 hash_post"})
    else:
        gate.add("reference_pixels_reply_recomputed",
                 bool(captures) and all(row.get("engine_equals_local_fnv")
                                        for row in captures),
                 [row.get("engine_equals_local_fnv") for row in captures])
        adapter_rows = [_decode_decision(row["pair"], int(row["decision_index"]))
                        for row in history_rows]
        gate.add("reference_direct_decode_agrees_with_adapter",
                 all(int(row.get("cand", -1)) == int(right.get("cand", -2))
                     for row, right in zip(adapter_rows, history_rows)),
                 {"compared": len(adapter_rows),
                  "rejections": sorted({row["classification"]
                                        for row in adapter_rows
                                        if not row["accepted"]})})
    mute = _load_json(base / "mute.json")
    post_fields = ((mute.get("copy_after_exit") or {}).get("fields")) or {}
    gate.add(f"{mode}_mute_recomputed",
             all(int(post_fields.get(name, -1)) == 0 for name in
                 ("musicMode", "playSounds", "musicVolume", "sfxVolume")),
             {"copy_after_exit_fields": post_fields,
              "problems": mute.get("problems")})
    recorded_tool = (record.get("tool") or {})
    report["tool_revision"] = {
        "recorded": recorded_tool.get("sha256"),
        "current": sha256_file(TOOL_PATH),
        "same": recorded_tool.get("sha256") == sha256_file(TOOL_PATH),
        "note": ("recorded, not gated: every number above is re-derived from the "
                 "raw files (hashes.jsonl vs D1, the PNGs, history.jsonl), so a "
                 "later revision of the offline code cannot weaken this gate; a "
                 "different value here means the artefacts were written by an "
                 "earlier revision of this file"),
    }
    gate.add(f"{mode}_tool_revision_recorded",
             bool(recorded_tool.get("sha256")),
             report["tool_revision"])
    return report

def recompute_arm_outcome(rows: Sequence[Mapping[str, Any]],
                          card: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive one arm's outcome from its raw decision rows.

    The same terminal priority the arm itself used: the frozen generation's own
    ending first, then ``game_over``, then a death, and only then "still there at
    the cap".  A control arm that dies with the boss alive is a ``legal_death``,
    not a ``not_removed_within_cap``.
    """

    instance = card["event"]["seed12_instance"]
    tracker = FrozenTracker(
        slot=int(instance["slot"]),
        spawn_decision=int(instance["spawn_decision"]),
        life_at_spawn=int(instance["life_at_spawn"]),
        entry_decision=int(card["entry"]["prefix_decisions"]))
    for row in rows:
        tracker.observe(row)
    summary = tracker.summary()
    entry_deaths = float(
        ((card.get("resources_at_entry") or {}).get("state") or {}).get("deaths", 0.0))
    last = rows[-1] if rows else None
    outcome = summary["outcome"]
    how = "tracker"
    if outcome is None and last is not None:
        if int((last.get("hud") or {}).get("game_over", 0) or 0):
            outcome, how = "game_over", "last row's HUD gameOverFlag"
        elif float((last.get("state") or {}).get("deaths", 0.0)) > entry_deaths:
            outcome, how = "legal_death", "the last row's deaths exceeds the entry's"
        else:
            outcome, how = "not_removed_within_cap", "the frozen generation is still alive"
    elif outcome is None and last is None:
        outcome, how = "not_removed_within_cap", "no control rows"
    if not summary["seen"] and outcome == "not_removed_within_cap":
        outcome, how = "generation_not_observed", "the slot never appears in the log"
    return dict(summary, recomputed_outcome=outcome, recomputed_how=how,
                entry_deaths=entry_deaths,
                last_deaths=(None if last is None
                             else float(last["state"]["deaths"])),
                classification="offline")


def recon_anchor_recheck(rows: Sequence[Mapping[str, Any]],
                         repo: str | Path) -> dict[str, Any]:
    """Re-derive the T04 and D1 anchors from a row log's own state hashes.

    Used by ``nominate`` (does its input really replay the teacher tape?) and by
    ``verify`` (does the recorded gate match the raw numbers?).  Nothing is taken
    from the recorded summary: the expected maps are re-read from the two corpora
    and the actual values from the rows' ``hash`` fields.
    """

    repo_path = Path(repo)
    hashes: dict[int, str] = {}
    for row in rows:
        pair = row.get("hash") or {}
        for key in ("frame_start", "frame_end"):
            if pair.get(key):
                hashes[int(row[key])] = str(pair[key]).strip().lower()
    t04_hash: dict[int, str] = {}
    for row in iter_jsonl(repo_path / SOURCE_BY_ID["t04-2f-seed12"]["rel"]):
        value = str(row.get("hash_pre") or "").strip().lower()
        if value:
            t04_hash[int(row["decision_frame"])] = value
    d1 = d1_replay_plan(repo_path)
    # the window is the rows' **own** frame range: an arm's control log starts at
    # its entry, so an anchor outside the log is not a missing comparison
    first_frame = min((int(row["frame_start"]) for row in rows), default=1)
    last_frame = max((int(row["frame_end"]) for row in rows),
                     default=max(hashes) if hashes else 0)
    return {
        "t04_pre": anchor_report(expected=t04_hash, actual=hashes,
                                 first_frame=first_frame, last_frame=last_frame,
                                 label="T04 hash_pre anchors (recomputed)"),
        "d1_pre": anchor_report(expected=d1["hash_pre"], actual=hashes,
                                first_frame=first_frame, last_frame=last_frame,
                                label="D1 hash_pre anchors (recomputed)"),
        "d1_post": anchor_report(expected=d1["hash_post"], actual=hashes,
                                 first_frame=first_frame, last_frame=last_frame,
                                 label="D1 hash_post anchors (recomputed)"),
        "hash_rows": len(hashes),
        "first_frame": first_frame,
        "last_frame": last_frame,
        "classification": "offline",
    }


def _recon_anchor_summary(reports: Mapping[str, Any]) -> dict[str, Any]:
    """The compact form of a :func:`recon_anchor_recheck` result for a gate."""

    return {name: {"compared": reports[name]["compared"],
                   "expected": reports[name]["expected"],
                   "mismatches": reports[name]["mismatches"],
                   "first_mismatch_frame": reports[name]["first_mismatch_frame"]}
            for name in ("t04_pre", "d1_pre", "d1_post")}


def _verify_recon_rows(rows: Sequence[Mapping[str, Any]], result: Mapping[str, Any],
                       repo: Path, gate: Gate, tag: str, *,
                       check_anchors: bool = True) -> dict[str, Any]:
    """Recount the recon/arm rows and (optionally) re-derive their anchors.

    ``check_anchors`` is false for a *control* arm's log: after the entry its masks
    are no longer the recorded tape's, so its state hashes are **supposed** to
    diverge from D1/T04 at its first own decision.  The prefix log of the same arm
    is the one that must reproduce every anchor, and it is checked separately.
    """

    decisions = len(rows)
    frames = decisions * DECISION_FRAMES
    first = int(rows[0]["decision_index"]) if rows else 0
    contiguous = all(
        int(row["decision_index"]) == first + position
        and int(row["frame_start"]) == int(row["decision_index"]) * DECISION_FRAMES + 1
        and int(row["frame_end"]) == int(row["frame_start"]) + 1
        for position, row in enumerate(rows))
    two_words = all(len(row.get("ev_flags") or []) == 2
                    and len(row.get("ev_flags_dec") or []) == 2
                    for row in rows)
    two_masks = all(len(row.get("masks") or []) == 2 for row in rows)
    checks = recon_anchor_recheck(rows, repo)
    reports = {name: checks[name] for name in ("t04_pre", "d1_pre", "d1_post")}
    gate.add(f"{tag}_rows_contiguous", contiguous and two_words and two_masks
             and bool(rows),
             {"decisions": decisions, "frames": frames,
              "first_decision": first, "contiguous": contiguous,
              "two_events_per_decision": two_words,
              "two_masks_per_decision": two_masks})
    gate.add(f"{tag}_frames_recounted",
             int((result.get("counts") or {}).get("frames", -1)) == frames
             and int((result.get("counts") or {}).get("decisions", -1)) == decisions,
             {"recorded": (result.get("counts") or {}), "recomputed_frames": frames,
              "recomputed_decisions": decisions})
    if check_anchors:
        gate.add(f"{tag}_anchors_recomputed",
                 all(item["mismatches"] == 0 for item in reports.values())
                 and reports["t04_pre"]["compared"] > 0,
                 _recon_anchor_summary(reports))
    return {"decisions": decisions, "frames": frames, "anchors": reports,
            "hash_rows": checks["hash_rows"], "first_decision": first,
            "anchors_checked": bool(check_anchors)}


def verify_02b(out_root: Path, repo: Path, gate: Gate) -> dict[str, Any]:
    """Re-derive 02-B from the recon rows and the frozen card (no engine)."""

    report: dict[str, Any] = {"status": "not-run",
                              "recon": (out_root / "02b" / "recon").as_posix(),
                              "nomination": (out_root / "02b" / "nomination.json"
                                             ).as_posix()}
    recon_dir = out_root / "02b" / "recon"
    result_path = recon_dir / "result.json"
    rows_path = recon_dir / "decisions.jsonl"
    if not result_path.is_file() or not rows_path.is_file():
        gate.add("02b_recon_present", False,
                 {"result": result_path.is_file(), "decisions": rows_path.is_file()})
        return report
    result = _load_json(result_path)
    rows = [dict(row) for row in iter_jsonl(rows_path)]
    report["status"] = "verified"
    report["recon_terminal"] = (result.get("terminal") or {}).get("kind")
    report["recon_gate_claimed"] = bool((result.get("gate") or {}).get("passed"))
    report["recon_gate_claim_recorded"] = {
        "claimed": report["recon_gate_claimed"],
        "failed_checks": [check["name"] for check in
                          ((result.get("gate") or {}).get("checks") or [])
                          if not check.get("ok")],
        "note": ("the claim is recorded, not trusted: the raw predicates below "
                 "are re-derived from decisions.jsonl and both corpora.  The "
                 "recon of 02-B was written by an earlier revision of this tool "
                 "whose tape check compared D1's 6,596-decision coverage with the "
                 "tape's full 6,660 decisions (the declared 64-decision tail) and "
                 "therefore failed on its own denominator; a second recon would "
                 "need another 13,194 engine frames, which this stage's allowance "
                 "does not hold"),
    }
    gate.add("02b_recon_rows_present", bool(rows), {"rows": len(rows)})
    gate.add("02b_recon_mute_clean",
             not ((result.get("mute") or {}).get("before_start_problems") or [])
             and not ((result.get("mute") or {}).get("after_exit_problems") or []),
             {"before": (result.get("mute") or {}).get("before_start_problems"),
              "after": (result.get("mute") or {}).get("after_exit_problems")})
    audit = _verify_recon_rows(rows, result, repo, gate, "02b_recon")
    report["recon"] = audit
    # the tape the recon claims to have replayed
    with_tool_digest = (result.get("tape") or {}).get("digest")
    tape = tape_schedule(repo)
    gate.add("02b_recon_tape_digest_recomputed",
             with_tool_digest == tape_digest(tape),
             {"recorded": with_tool_digest, "recomputed": tape_digest(tape)})
    segments = generation_segments(rows)
    report["generations"] = segments["counts"]
    # -- the nomination ---------------------------------------------------
    nomination_path = out_root / "02b" / "nomination.json"
    if not nomination_path.is_file():
        gate.add("02b_nomination_present", False, {"path": nomination_path.as_posix()})
        report["status"] = "failed"
        return report
    nomination = _load_json(nomination_path)
    report["nomination"] = {
        "verdict": nomination.get("verdict"),
        "chosen": (nomination.get("chosen") or {}).get("candidate_id"),
        "counts": nomination.get("counts"),
        "gate_claimed": bool((nomination.get("gate") or {}).get("passed")),
    }
    card_path = out_root / "02b" / "task_card.json"
    md_path = out_root / "02b" / "task_card.md"
    frozen = bool(nomination.get("chosen")) and card_path.is_file()
    report["frozen"] = frozen
    if not frozen:
        rows_missing = nomination.get("near_miss") or []
        boss_generations = [item for item in segments["generations"]
                            if item["boss"]]
        covered = {str(item.get("slot")) for item in rows_missing}
        gate.add("02b_no_candidate_is_documented",
                 bool(rows_missing) and bool(boss_generations)
                 and all(str(item["slot"]) in covered for item in boss_generations),
                 {"near_miss_rows": len(rows_missing),
                  "boss_generations": len(boss_generations),
                  "covered_slots": sorted(covered),
                  "note": ("a failed nomination must still name, for every BOSS "
                           "generation, the condition it missed")})
        gate.add("02b_frozen_card_exists", not True,
                 {"verdict": nomination.get("verdict"),
                  "reason": "the nomination did not freeze a candidate"})
        report["status"] = "not-applicable"
        return report
    frozen_card = read_task_card(out_root / "02b")
    card = frozen_card["card"]
    report["task_card"] = {
        "content_sha256": card["content_sha256"],
        "content_sha256_recomputed": frozen_card["content_sha256_recomputed"],
        "json_sha256": sha256_file(card_path),
        "md_present": md_path.is_file(),
        "md_sha256": sha256_file(md_path) if md_path.is_file() else None,
        "md_carries_digest": (md_path.is_file()
                              and card["content_sha256"]
                              in md_path.read_text(encoding="utf-8")),
    }
    gate.add("02b_card_content_sha256_recomputed", True, report["task_card"])
    gate.add("02b_card_markdown_carries_the_digest",
             bool(report["task_card"]["md_carries_digest"]), report["task_card"])
    gate.add("02b_card_recorded_sha_matches_nomination",
             (nomination.get("task_card") or {}).get("content_sha256")
             == card["content_sha256"],
             {"nomination": (nomination.get("task_card") or {}).get("content_sha256"),
              "card": card["content_sha256"]})
    both_gate = Gate("02b_frozen_event")
    chosen = nomination.get("chosen") or {}
    instance = card["event"]["seed12_instance"]
    match = next((item for item in segments["generations"]
                  if int(item["slot"]) == int(instance["slot"])
                  and int(item["spawn_decision"]) == int(instance["spawn_decision"])),
                 None)
    both_gate.add("chosen_generation_present_in_the_raw_rows", match is not None,
                  {"slot": instance["slot"],
                   "spawn_decision": instance["spawn_decision"]})
    if match is not None:
        both_gate.add("outcome_reproduced",
                      match["outcome"] == card["event"]["classification"]
                      == chosen.get("outcome"),
                      {"raw": match["outcome"],
                       "card": card["event"]["classification"],
                       "nomination": chosen.get("outcome")})
        both_gate.add("completion_frame_reproduced",
                      match.get("completion_frame")
                      == (card["event"].get("completion") or {}).get("frame"),
                      {"raw": match.get("completion_frame"),
                       "card": (card["event"].get("completion") or {}).get("frame"),
                       "kind": match.get("completion_kind")})
        both_gate.add("removal_frame_reproduced",
                      match["removal_frame"] == (card["event"]["removal"] or {}).get(
                          "frame"),
                      {"raw": match["removal_frame"],
                       "card": (card["event"]["removal"] or {}).get("frame")})
        both_gate.add("life_figures_reproduced",
                      int(match["life_first"]) == int(instance["life_first"])
                      and int(match["life_last"]) == int(instance["life_last"])
                      and int(match["life_min"]) == int(instance["life_min"]),
                      {"raw": {key: match[key] for key in
                               ("life_first", "life_last", "life_min")},
                       "card": {key: instance[key] for key in
                                ("life_first", "life_last", "life_min")}})
        both_gate.add("kill_outcome", match["outcome"] in KILL_OUTCOMES,
                      {"outcome": match["outcome"], "why": match["outcome_why"]})
        span = (None if match.get("completion_frame") is None
                else int(match["completion_frame"]) - int(card["entry"]["frame"]))
        both_gate.add("span_within_cap",
                      span is not None and span <= int(card["frame_cap"]["frames"]),
                      {"span": span, "cap": card["frame_cap"]["frames"],
                       "entry_frame": card["entry"]["frame"],
                       "completion_frame": match.get("completion_frame")})
        both_gate.add("entry_before_the_consequence",
                      int(card["entry"]["decision_index"]) >= 1
                      and int(card["entry"]["frame"])
                      < int(match.get("completion_frame") or 0),
                      {"entry_decision": card["entry"]["decision_index"],
                       "entry_frame": card["entry"]["frame"],
                       "completion_frame": match.get("completion_frame")})
        both_gate.add("target_alive_at_entry",
                      any(int(item["life"]) > 0
                          for item in (rows[int(card["entry"]["decision_index"]) - 1]
                                       .get("enemies") or [])
                          if int(item["slot"]) == int(instance["slot"]))
                      if int(card["entry"]["decision_index"]) >= 1 else False,
                      {"entry_decision": card["entry"]["decision_index"],
                       "slot": instance["slot"],
                       "card_target_life": card["entry"].get(
                           "target_life_at_entry")})
    both_gate.add("entry_hash_is_the_raw_row_hash",
                  card["entry"]["hash"] == (rows[int(
                      card["entry"]["decision_index"]) - 1].get("hash") or {}).get(
                          "frame_end") if int(card["entry"]["decision_index"]) >= 1
                  else False,
                  {"card": card["entry"]["hash"]})
    both_gate.add("prefix_digest_matches_the_tape",
                  card["entry"]["prefix_masks_sha256"]
                  == tape_digest(tape, 0, int(card["entry"]["prefix_decisions"])),
                  {"card": card["entry"]["prefix_masks_sha256"],
                   "recomputed": tape_digest(
                       tape, 0, int(card["entry"]["prefix_decisions"]))})
    both_gate.add("frame_cap_is_the_ticket_cap",
                  int(card["frame_cap"]["frames"]) == CARD_FRAME_CAP
                  and int(card["frame_cap"]["decisions"]) == CARD_DECISION_CAP,
                  card["frame_cap"])
    both_gate.add("control_is_constant_cand_12",
                  (card.get("control") or {}).get("cand") == CONTROL_CAND,
                  card.get("control"))
    both_gate.add("not_a_policy_result_flag", card.get("not_a_policy_result") is True,
                  card.get("not_a_policy_result"))
    gate.extend(both_gate)
    if not gate.passed:
        report["status"] = "failed"
    return report


def verify_02c(out_root: Path, repo: Path, gate: Gate) -> dict[str, Any]:
    """Re-derive 02-C from the two arms' raw decision rows (no engine)."""

    report: dict[str, Any] = {"status": "not-run",
                              "path": (out_root / "02c" / "intervene.json").as_posix()}
    path = out_root / "02c" / "intervene.json"
    if not path.is_file():
        gate.add("02c_intervene_present", False, {"path": path.as_posix()})
        return report
    record = _load_json(path)
    report["status"] = "verified"
    report["recorded_gate"] = bool((record.get("gate") or {}).get("passed"))
    report["recorded_comparison"] = record.get("comparison")
    gate.add("02c_intervene_gate_claimed", report["recorded_gate"],
             (record.get("gate") or {}).get("passed"))
    frozen = read_task_card(out_root / "02b")
    card = frozen["card"]
    if not (out_root / "02b" / "task_card.json").is_file():
        raise SourceError("02c exists without a frozen task card")
    # the arms must have run against **this** card: a card frozen afterwards would
    # make their entry and their consequence a different contract
    recorded_card = (record.get("task_card") or {}).get("content_sha256")
    gate.add("02c_arms_used_this_frozen_card",
             recorded_card == card["content_sha256"],
             {"intervene_recorded": recorded_card,
              "card_on_disk": card["content_sha256"],
              "note": "a frozen card is never rewritten; this binds both arms to it"})
    tape = tape_schedule(repo)
    report["task_card_content_sha256"] = card["content_sha256"]
    recomputed_arms: dict[str, Any] = {}
    for arm in ("T", "C"):
        arm_record = (record.get("arms") or {}).get(arm) or {}
        base = out_root / "02c" / ("arm_T" if arm == "T" else "arm_C")
        control_path = base / "decisions.jsonl"
        prefix_path = base / "prefix.jsonl"
        rows = ([dict(row) for row in iter_jsonl(control_path)]
                if control_path.is_file() else [])
        prefix_rows = ([dict(row) for row in iter_jsonl(prefix_path)]
                       if prefix_path.is_file() else [])
        audit = _verify_recon_rows(rows, {"counts": {
            "decisions": (arm_record.get("counts") or {}).get("control_decisions"),
            "frames": (arm_record.get("counts") or {}).get("control_frames")}},
            repo, gate, f"02c_arm{arm}_control",
            check_anchors=(arm == "T"))
        entry = int(card["entry"]["prefix_decisions"])
        entry_frame = int(card["entry"]["frame"])
        mask_ok = True
        mask_first: dict[str, Any] | None = None
        for row in rows:
            index = int(row["decision_index"])
            if arm == "T":
                want = [int(tape["masks"][index][0]), int(tape["masks"][index][1])]
            else:
                want = decision_masks([CONTROL_CAND] * (index + 1), index)
            if [int(value) for value in row["masks"]] != want:
                mask_ok = False
                mask_first = mask_first or {"decision": index,
                                            "recorded": row["masks"], "want": want}
        gate.add(f"02c_arm{arm}_masks_recomputed", mask_ok,
                 {"rows": len(rows), "first_mismatch": mask_first,
                  "controller": arm_record.get("controller")})
        prefix_ok = (len(prefix_rows) == entry
                     and all(int(row["decision_index"]) == position
                             and [int(value) for value in row["masks"]]
                             == [int(tape["masks"][position][0]),
                                 int(tape["masks"][position][1])]
                             for position, row in enumerate(prefix_rows)))
        gate.add(f"02c_arm{arm}_prefix_rows", prefix_ok,
                 {"rows": len(prefix_rows), "entry_decisions": entry})
        prefix_hash = (prefix_rows[-1].get("hash") or {}).get("frame_end") \
            if prefix_rows else None
        gate.add(f"02c_arm{arm}_entry_hash_recomputed",
                 prefix_hash is not None and str(prefix_hash) == str(
                     card["entry"]["hash"]),
                 {"recomputed": prefix_hash, "card": card["entry"]["hash"],
                  "entry_frame": entry_frame})
        anchors: dict[str, Any] = {}
        for row in prefix_rows:
            for check in ((row.get("anchor") or {}).get("checks") or []):
                bucket = anchors.setdefault(check["kind"], {"compared": 0,
                                                            "mismatches": 0,
                                                            "first": None})
                bucket["compared"] += 1
                if not check["ok"]:
                    bucket["mismatches"] += 1
                    if bucket["first"] is None:
                        bucket["first"] = check
        gate.add(f"02c_arm{arm}_prefix_anchors",
                 (anchors.get("t04_hash_pre") or {}).get("mismatches") == 0
                 and (anchors.get("t04_hash_pre") or {}).get("compared", 0) > 0
                 and (anchors.get("d1_hash_pre") or {}).get("mismatches") == 0
                 and (anchors.get("d1_hash_post") or {}).get("mismatches") == 0,
                 anchors)
        outcome = recompute_arm_outcome(rows, card)
        recorded_outcome = arm_record.get("outcome")
        gate.add(f"02c_arm{arm}_outcome_recomputed",
                 outcome["recomputed_outcome"] == recorded_outcome,
                 {"recomputed": outcome["recomputed_outcome"],
                  "recorded": recorded_outcome,
                  "life_first": outcome["life_first"],
                  "life_last": outcome["life_last"],
                  "life_min": outcome["life_min"]})
        mute = arm_record.get("mute") or {}
        gate.add(f"02c_arm{arm}_mute_clean",
                 not (mute.get("after_exit_problems") or [])
                 and not (mute.get("before_start_problems") or []),
                 {"after": mute.get("after_exit_problems"),
                  "before": mute.get("before_start_problems")})
        gate.add(f"02c_arm{arm}_ledger_settled",
                 bool((arm_record.get("ledger") or {}).get("settle_seq")),
                 (arm_record.get("ledger") or {}).get("actual"))
        recomputed_arms[arm] = {"outcome": outcome["recomputed_outcome"],
                                "recorded": recorded_outcome,
                                "frames": audit["frames"],
                                "decisions": audit["decisions"],
                                "prefix_rows": len(prefix_rows),
                                "entry_hash": prefix_hash,
                                "anchors": anchors,
                                "life_series": outcome["life_series"]}
    report["arms"] = recomputed_arms
    outcome_t = recomputed_arms["T"]["outcome"]
    outcome_c = recomputed_arms["C"]["outcome"]
    recorded = record.get("comparison") or {}
    gate.add("02c_recorded_comparison_recomputes",
             recorded.get("arm_T_outcome") == outcome_t
             and recorded.get("arm_C_outcome") == outcome_c,
             {"recorded": {"T": recorded.get("arm_T_outcome"),
                           "C": recorded.get("arm_C_outcome")},
              "recomputed": {"T": outcome_t, "C": outcome_c}})
    gate.add("02c_teacher_succeeded", outcome_t in KILL_OUTCOMES,
             {"outcome": outcome_t})
    gate.add("02c_control_distinguishable",
             bool(recorded.get("distinguishable")) and (outcome_t != outcome_c
                                                       or bool(recorded.get("resource_pairs_T_vs_C"))),
             {"distinguishable": recorded.get("distinguishable"),
              "T": outcome_t, "C": outcome_c,
              "resource_pairs": recorded.get("resource_pairs_T_vs_C")})
    if not (outcome_t in KILL_OUTCOMES and recorded.get("distinguishable")):
        report["status"] = "failed"
    return report


def cmd_verify(args: argparse.Namespace) -> int:
    """Re-derive every implemented package from its raw files; never starts an engine."""

    out_root = Path(args.out).expanduser().resolve()
    package = getattr(args, "package", "02a") or "02a"
    allowed = ("02a", "02a-prefix", "02a-reference", "02b", "02c", "03", "all")
    if package not in allowed:
        print(f"not-implemented: verify --package {package}")
        return EXIT_NOT_IMPLEMENTED
    want_pair = package in ("02a", "all")
    wanted_modes = [mode for mode in ENGINE_PACKAGES
                    if package in ("all", f"02a-{mode}")]
    want_compare = package == "all"
    want_02b = package in ("02b", "all")
    want_02c = package in ("02c", "all")
    want_03 = package in ("03", "all")
    # `all` follows the artefacts that exist: 02-A needs its own inputs, 02-B
    # needs the recon, 02-C needs the frozen card *and* the intervening evidence.
    if not (out_root / "02a" / "sources.json").is_file():
        if package == "all":
            want_pair = False
    if not (out_root / "02b" / "recon" / "result.json").is_file():
        if package == "all":
            want_02b = False
            want_02c = False
    if not (out_root / "02b" / "task_card.json").is_file():
        if package == "all":
            want_02c = False
            want_03 = False
    if not ((out_root / "03" / "pilot.json").is_file()
            or (out_root / "03" / "smoke" / "pilot.json").is_file()):
        if package == "all":
            want_03 = False
    repo = Path(getattr(args, "repo", REPO_ROOT) or REPO_ROOT).expanduser().resolve()
    sources_path = out_root / "02a" / "sources.json"
    adapter_path = out_root / "02a" / "adapter.json"
    problems: list[str] = []
    if want_pair:
        if not sources_path.is_file():
            problems.append(f"missing {sources_path}")
        if not adapter_path.is_file():
            problems.append(f"missing {adapter_path}")
    for mode in wanted_modes:
        if not (out_root / "02a" / mode / "result.json").is_file():
            problems.append(f"missing {out_root / '02a' / mode / 'result.json'}")
    if want_compare and not (out_root / "02a" / "compare.json").is_file():
        problems.append(f"missing {out_root / '02a' / 'compare.json'}")
    if want_02b:
        for name in ("result.json", "decisions.jsonl"):
            path = out_root / "02b" / "recon" / name
            if not path.is_file():
                problems.append(f"missing {path}")
        if not (out_root / "02b" / "nomination.json").is_file():
            problems.append(f"missing {out_root / '02b' / 'nomination.json'}")
    if want_02c and not (out_root / "02c" / "intervene.json").is_file():
        problems.append(f"missing {out_root / '02c' / 'intervene.json'}")
    if want_03 and package == "03" and not (
            (out_root / "03" / "pilot.json").is_file()
            or (out_root / "03" / "smoke" / "pilot.json").is_file()):
        problems.append(f"missing {out_root / '03' / 'pilot.json'}")
    if problems:
        payload = {
            "schema": SCHEMA_VERIFY, "created_at": now_iso(),
            "classification": "offline", "out": out_root.as_posix(),
            "packages": {}, "gate": Gate("verify", [Check("inputs_present", False,
                                                          problems)]).as_dict(),
            "problems": problems,
        }
        write_json(out_root / "verify.json", payload)
        print("verify refused: " + "; ".join(problems), file=sys.stderr)
        return EXIT_TECHNICAL

    sources = json.loads(sources_path.read_text(encoding="utf-8")) if want_pair else {}
    adapter_json = (json.loads(adapter_path.read_text(encoding="utf-8"))
                    if want_pair else {})
    repo = Path(sources.get("repo") or repo).expanduser().resolve()
    gate = Gate("verify")
    engine_report: dict[str, Any] = {}
    compare_report: dict[str, Any] = {}

    # 1-6. the offline 02-A pair (only when that package was asked for)
    re_hash: list[dict[str, Any]] = []
    decode_recheck: list[dict[str, Any]] = []
    if want_pair:
        pair_report = verify_offline_pair(sources, adapter_json, repo, gate)
        re_hash = pair_report["re_hash"]
        decode_recheck = pair_report["decode_recheck"]

    # 7. the ledger (when present) stays inside the ticket budget
    ledger_path = out_root / "ledger.jsonl"
    ledger_block: dict[str, Any] = {"present": ledger_path.is_file()}
    ledger_rows: list[dict[str, Any]] = []
    if ledger_path.is_file():
        ledger_rows = [dict(row) for row in iter_jsonl(ledger_path)]
        ledger = Ledger(ledger_path, BudgetLimits())
        totals = ledger.totals()
        over = {name: {"committed": totals["committed"][name],
                       "limit": totals["limits"][f"max_{name}"]}
                for name in BUDGET_DIMENSIONS
                if totals["committed"][name] > float(totals["limits"][f"max_{name}"])}
        ledger_block.update({"totals": totals, "over_budget": over})
        gate.add("ledger_inside_budget", not over, over or totals["committed"])

    # 8. the two engine packages, re-derived from their own raw artefacts
    engine_report: dict[str, Any] = {}
    mode_status: dict[str, bool] = {}
    plan: dict[str, Any] | None = None
    if wanted_modes:
        plan = d1_replay_plan(repo)
        gate.add("verify_d1_source_current",
                 plan["sha256"] == plan["registered_sha256"],
                 {"sha256": plan["sha256"],
                  "registered": plan["registered_sha256"]})
        for mode in wanted_modes:
            before = len(gate.checks)
            engine_report[mode] = verify_engine_package(out_root, mode, gate, plan)
            mode_status[mode] = all(check.ok for check in gate.checks[before:])

    # 9. `compare` recomputed from the same files it read
    if want_compare:
        claimed = _load_json(out_root / "02a" / "compare.json")
        fresh = compare_runs(out_root)
        claimed_flags = {row["item"]: bool(row["equal"])
                         for row in claimed.get("items") or []}
        fresh_flags = {row["item"]: bool(row["equal"])
                       for row in fresh.get("items") or []}
        difference = first_difference(claimed_flags, fresh_flags, "compare")
        compare_report = {
            "claimed_verdict": claimed.get("verdict"),
            "recomputed_verdict": fresh.get("verdict"),
            "claimed_items": len(claimed_flags),
            "recomputed_items": len(fresh_flags),
            "first_difference_between_runs": difference,
            "different_items": fresh.get("different_items"),
            "first_difference": fresh.get("first_difference"),
        }
        gate.add("compare_recomputed",
                 difference is None
                 and claimed.get("verdict") == fresh.get("verdict")
                 and bool((claimed.get("gate") or {}).get("passed"))
                 == bool((fresh.get("gate") or {}).get("passed")),
                 compare_report)
        gate.add("compare_verdict_identical",
                 fresh.get("verdict") == "identical"
                 and int(fresh.get("different_items", -1)) == 0, compare_report)

    run_json_path = out_root / "run.json"
    packages: dict[str, Any] = {}
    if want_pair:
        packages["02a"] = {"status": "verified" if gate.passed else "failed",
                           "sources": sources_path.as_posix(),
                           "adapter": adapter_path.as_posix()}
    for mode in ENGINE_PACKAGES:
        if mode in wanted_modes:
            packages[f"02a-{mode}"] = {
                "status": "verified" if mode_status.get(mode) else "failed",
                "result": (out_root / "02a" / mode / "result.json").as_posix(),
                "name": mode,
                "report": engine_report.get(mode),
            }
    if want_compare:
        packages["02a-compare"] = {"status": "verified", "report": compare_report}
    report_02b: dict[str, Any] = {}
    report_02c: dict[str, Any] = {}
    report_03: dict[str, Any] = {}
    if want_02b:
        before = len(gate.checks)
        try:
            report_02b = verify_02b(out_root, repo, gate)
        except (SourceError, EngineRefused) as exc:
            report_02b = {"status": "technical_invalid",
                          "error": f"{type(exc).__name__}: {exc}"}
            gate.add("02b_verified", False, report_02b)
        b_ok = all(check.ok for check in gate.checks[before:])
        packages["02b"] = {
            "status": ("verified" if b_ok and report_02b.get("status") == "verified"
                       else ("not-applicable"
                             if report_02b.get("status") == "not-applicable"
                             else "failed")),
            "recon": (out_root / "02b" / "recon" / "result.json").as_posix(),
            "nomination": (out_root / "02b" / "nomination.json").as_posix(),
            "report": report_02b,
        }
    else:
        packages["02b"] = {"status": "not-run",
                           "reason": "not asked for (--package 02b or all)"}
    if want_02c:
        before = len(gate.checks)
        try:
            report_02c = verify_02c(out_root, repo, gate)
        except (SourceError, EngineRefused) as exc:
            report_02c = {"status": "technical_invalid",
                          "error": f"{type(exc).__name__}: {exc}"}
            gate.add("02c_verified", False, report_02c)
        c_ok = all(check.ok for check in gate.checks[before:])
        packages["02c"] = {
            "status": "verified" if c_ok and report_02c.get("status") == "verified"
            else "failed",
            "intervene": (out_root / "02c" / "intervene.json").as_posix(),
            "report": report_02c,
        }
    else:
        packages["02c"] = {"status": "not-run",
                           "reason": ("not asked for (--package 02c or all)"
                                      if package != "all" else
                                      "no frozen card / no intervening evidence")}
    if want_03:
        before = len(gate.checks)
        try:
            report_03 = verify_03(out_root, repo, gate)
        except (SourceError, EngineRefused, AdapterUnavailable) as exc:
            report_03 = {"status": "technical_invalid",
                         "error": f"{type(exc).__name__}: {exc}"}
            gate.add("03_verified", False, report_03)
        o_ok = all(check.ok for check in gate.checks[before:])
        packages["03"] = {
            "status": "verified" if o_ok and report_03.get("status") == "verified"
            else "failed",
            "smoke": report_03.get("smoke"),
            "report": report_03,
        }
    else:
        packages["03"] = {"status": "not-run",
                          "reason": ("not asked for (--package 03 or all)"
                                     if package != "all" else
                                     "no 03 pilot evidence under this out dir")}
    # 9b. the 03 strength check on the fit root, re-derived from its own raw rows
    report_03_strength: dict[str, Any] = {}
    strength_paths = strength_layout(out_root)
    want_strength = (package in ("03", "all")) and Path(
        strength_paths["result"]).is_file()
    if want_strength:
        before = len(gate.checks)
        try:
            report_03_strength = verify_03_strength(out_root, repo, gate)
        except (SourceError, EngineRefused, AdapterUnavailable) as exc:
            report_03_strength = {"status": "technical_invalid",
                                  "error": f"{type(exc).__name__}: {exc}"}
            gate.add("03_strength_verified", False, report_03_strength)
        strength_ok = all(check.ok for check in gate.checks[before:])
        packages["03-strength"] = {
            "status": ("verified" if strength_ok
                       and report_03_strength.get("status") == "verified"
                       else "failed"),
            "result": Path(strength_paths["result"]).as_posix(),
            "report": report_03_strength,
        }
    else:
        packages["03-strength"] = {
            "status": "not-run",
            "reason": "no 03/strength/result.json under this out dir"}
    # 9c. the teacher-arm diagnostic, re-derived from its own raw rows
    report_diag_teacher: dict[str, Any] = {}
    want_diag_teacher = (package in ("03", "all")
                         and Path(diag_teacher_layout(out_root)["result"]).is_file())
    if want_diag_teacher:
        before = len(gate.checks)
        try:
            report_diag_teacher = verify_03_diag_teacher(out_root, repo, gate)
        except (SourceError, EngineRefused, AdapterUnavailable) as exc:
            report_diag_teacher = {"status": "technical_invalid",
                                   "error": f"{type(exc).__name__}: {exc}"}
            gate.add("03_diag_teacher_verified", False, report_diag_teacher)
        diag_ok = all(check.ok for check in gate.checks[before:])
        packages["03-diag-teacher"] = {
            "status": ("verified" if diag_ok
                       and report_diag_teacher.get("status") == "verified"
                       else "failed"),
            "result": Path(diag_teacher_layout(out_root)["result"]).as_posix(),
            "report": report_diag_teacher,
        }
    else:
        packages["03-diag-teacher"] = {
            "status": "not-run",
            "reason": "no 03/diag/teacher_arm/result.json under this out dir"}
    # 10. run.json consistency: every gate it claims is re-derived here
    run_report: dict[str, Any] = {"present": run_json_path.is_file()}
    if run_json_path.is_file():
        claimed_run = json.loads(run_json_path.read_text(encoding="utf-8"))
        claimed_gates = claimed_run.get("gates") or {}
        mapping = {
            "02a-sources": "02a", "02a-adapter": "02a",
            "02a-prefix": "02a-prefix", "02a-reference": "02a-reference",
            "02a-compare": "02a-compare", "02b": "02b", "02c": "02c",
            "03": "03",
        }
        disagreements: list[dict[str, Any]] = []
        compared = 0
        ok_status = ("passed", "verified", "reused")
        bad_status = ("failed", "technical", "budget_refused")
        for name, target in mapping.items():
            claimed = claimed_gates.get(name)
            if claimed is None or target not in packages:
                continue
            status = str((claimed or {}).get("status"))
            actual = str((packages.get(target) or {}).get("status"))
            # only the packages whose evidence verify actually re-derived here
            # can be compared; a `not-run`/`skipped` claim states absence
            if status not in ok_status + bad_status:
                continue
            if actual in ("not-run",) or actual.startswith("not-"):
                continue
            compared += 1
            agreed = ((status in ok_status and actual == "verified")
                      or (status in bad_status and actual in ("failed",)))
            if not agreed:
                disagreements.append({"package": name, "run_json": status,
                                      "verify": actual})
        # a package verify *proved* to be broken can never be claimed healthy
        for name, target in mapping.items():
            status = str(((claimed_gates.get(name) or {}).get("status")))
            actual = str((packages.get(target) or {}).get("status"))
            if actual == "failed" and status in ok_status:
                row = {"package": name, "run_json": status, "verify": actual}
                if row not in disagreements:
                    disagreements.append(row)
        run_report.update({
            "claimed_gates": sorted(claimed_gates),
            "compared": compared,
            "disagreements": disagreements,
            "schema": claimed_run.get("schema"),
            "engine_starts": claimed_run.get("engine_starts"),
            "started_at": claimed_run.get("started_at"),
            "finished_at": claimed_run.get("finished_at"),
            "deadline": claimed_run.get("deadline"),
            "next_step": claimed_run.get("next_step"),
        })
        gate.add("run_json_consistent_with_the_recomputed_gates",
                 not disagreements, run_report)
        if ledger_block.get("present"):
            reconciliation = run_json_ledger_reconciliation(
                claimed_totals=claimed_run.get("ledger_totals") or {},
                ledger_rows=ledger_rows,
                ledger_actual=((ledger_block.get("totals") or {}).get("actual")
                               or {}),
                accounted_attempt_ids=(strength_attempt_ids(out_root)
                                       | diag_teacher_attempt_ids(out_root)))
            run_report["ledger_reconciliation"] = reconciliation
            gate.add("run_json_ledger_totals_match", reconciliation["ok"],
                     reconciliation)
        else:
            gate.add("run_json_ledger_totals_match", True,
                     {"present": False,
                      "note": "no ledger under this out dir to compare against"})
    payload = {
        "schema": SCHEMA_VERIFY,
        "generated_by": "tools/rl/v8_31_followup.py verify",
        "created_at": now_iso(),
        "classification": "offline",
        "out": out_root.as_posix(),
        "repo": repo.as_posix(),
        "host": host_identity(),
        "heads": git_identity(repo),
        "package": package,
        "packages": packages,
        "ledger": ledger_block,
        "gate": gate.as_dict(),
        "engine_packages": engine_report,
        "compare": compare_report,
        "verified": {
            "sources": [item["id"] for item in re_hash if item["ok"]],
            "runs": [item["id"] for item in decode_recheck if item["ok"]],
            "engine_packages": list(engine_report),
            "compare": bool(compare_report),
            "02b": report_02b.get("status"),
            "02c": report_02c.get("status"),
            "03": report_03.get("status"),
            "03_strength": report_03_strength.get("status"),
            "03_diag_teacher": report_diag_teacher.get("status"),
        },
        "02b": report_02b,
        "02c": report_02c,
        "03": report_03,
        "03_strength": report_03_strength,
        "03_diag_teacher": report_diag_teacher,
        "run_json": run_report,
    }
    path = write_json(out_root / "verify.json", payload)
    for check in gate.checks:
        print(f"  {'ok  ' if check.ok else 'FAIL'} {check.name}")
    print(f"wrote: {path}  gate={'passed' if gate.passed else 'FAILED'}")
    if not gate.passed:
        return EXIT_GATE_FAILED
    return EXIT_OK


# ---------------------------------------------------------------------------
# 02-A.3 `prefix` / 02-A.4 `reference`: one engine start each
# ---------------------------------------------------------------------------
#: The reference walks one frame past the boundary: pixel frame 8,664 is only
#: readable after ``STEP`` 8,665, and it is one of the seven frames ``compare``
#: has to find equal (`PIXEL_FRAME_RELATION`).
REFERENCE_LAST_FRAME = PREFIX_FRAME + 1


def d1_replay_plan(repo: str | Path) -> dict[str, Any]:
    """Read D1 once: the flat masks, the two anchor maps and their denominators.

    Nothing is quoted from memory: the file is re-hashed, the anchor frames and
    their values are re-derived, and the two denominators the ticket names
    (2,165 ``hash_pre`` comparisons after the reset + 2,166 ``hash_post``) fall
    out of the recording itself.
    """

    repo_path = Path(repo)
    spec = SOURCE_BY_ID[D1_SOURCE_ID]
    model = load_labels_source(spec, repo_path)
    path = repo_path / spec["rel"]
    digest = sha256_file(path)
    flat = {int(frame): int(mask) for frame, mask in model["frame_masks"].items()}
    hash_pre = {int(frame): str(value).strip().lower()
                for frame, value in model["hash_pre"].items()}
    hash_post = {int(frame): str(value).strip().lower()
                 for frame, value in model["hash_post"].items()}
    if not hash_pre or not hash_post:
        raise SourceError("D1 carries no hash anchors")
    reset_frame = min(hash_pre)
    pre_frames = sorted(frame for frame in hash_pre
                        if 1 <= frame <= PREFIX_FRAME - DECISION_FRAMES)
    post_frames = sorted(frame for frame in hash_post if 1 <= frame <= PREFIX_FRAME)
    step_masks = step_masks_from_d1(flat, CONTINUATION_LAST_FRAME + 8)
    shared = sorted(set(pre_frames) & set(post_frames))
    chain_breaks = [frame for frame in shared if hash_pre[frame] != hash_post[frame]]
    continuation_frame = CONTINUATION_LAST_FRAME
    return {
        "id": D1_SOURCE_ID,
        "path": path.as_posix(),
        "rel": spec["rel"],
        "sha256": digest,
        "registered_sha256": REGISTERED_SHA256[D1_SOURCE_ID],
        "rows": int(model["rows"]),
        "frames_covered": int(model["frames_covered"]),
        "frames_per_decision": model["frames_per_decision"],
        "step_masks": step_masks,
        "flat_masks": flat,
        "hash_pre": hash_pre,
        "hash_post": hash_post,
        "reset_frame": reset_frame,
        "reset_hash": hash_pre[reset_frame],
        "pre_frames": pre_frames,
        "post_frames": post_frames,
        "expected_pre": {frame: hash_pre[frame] for frame in pre_frames},
        "expected_post": {frame: hash_post[frame] for frame in post_frames},
        "pre_is_subset_of_post": set(pre_frames) <= set(post_frames),
        "shared_anchors": len(shared),
        "chain_breaks": len(chain_breaks),
        "first_chain_break": chain_breaks[0] if chain_breaks else None,
        "continuation_frame": continuation_frame,
        "continuation_expected": hash_post.get(continuation_frame),
        "continuation_pre_also": hash_pre.get(continuation_frame),
        "continuation_pre_matches_post": bool(
            hash_post.get(continuation_frame) == hash_pre.get(continuation_frame)),
        "classification": "offline",
    }


D1_SOURCE_ID = "d1-canonical-seed12"


def _plan_for_mode(mode: str, step_masks: Mapping[int, int]) -> tuple[list[dict[str, Any]], int]:
    """``(units, last_frame)`` for one package, from the D1 mask trace."""

    if mode == "prefix":
        last_frame = CONTINUATION_LAST_FRAME
        return plan_prefix_steps(step_masks, last_frame), last_frame
    last_frame = REFERENCE_LAST_FRAME
    return (plan_reference_steps(step_masks, last_frame, WINDOW_STEPS), last_frame)


def _check_plan(units: Sequence[Mapping[str, Any]], step_masks: Mapping[int, int],
                last_frame: int) -> dict[str, Any]:
    """The plan must expand to D1's frame -> mask trace, with no frame missing."""

    expanded: list[tuple[int, int]] = []
    for unit in units:
        for offset in range(int(unit["frames"])):
            expanded.append((int(unit["start"]) + offset, int(unit["mask"])))
    expected = [(frame, int(step_masks[frame])) for frame in range(1, int(last_frame) + 1)]
    ends = set(unit_end_frames(units))
    missing_window = [frame for frame in WINDOW_STEPS if frame not in ends]
    return {
        "units_expand_to_the_d1_trace": expanded == expected,
        "expanded_frames": len(expanded),
        "expected_frames": len(expected),
        "first_expansion_difference": next(
            ({"index": index, "planned": expanded[index], "expected": expected[index]}
             for index in range(min(len(expanded), len(expected)))
             if expanded[index] != expected[index]),
            None),
        "window_frames_are_step_ends": not missing_window,
        "missing_window_ends": missing_window,
        "last_frame": int(last_frame),
        "classification": "offline",
    }


def run_engine_package(args: argparse.Namespace, mode: str) -> int:
    """Prefix (02-A.3) or reference (02-A.4): one RESET, one continuous walk."""

    if mode not in ("prefix", "reference"):
        raise EngineRefused(f"unknown engine package {mode!r}")
    module = slice_tool()
    maybe_reexec_under_xvfb(enabled=not bool(getattr(args, "no_xvfb", False)))

    started_at = now_iso()
    started = time.perf_counter()
    out_root = Path(args.out).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve()
    paths = _package_paths(out_root, mode)
    schema = SCHEMA_PREFIX if mode == "prefix" else SCHEMA_REFERENCE
    gate = Gate(f"02a_{mode}")
    limits = BudgetLimits(max_starts=args.max_starts,
                          max_engine_frames=args.max_engine_frames,
                          max_wall_s=args.max_wall_s,
                          max_write_gib=args.max_write_gib)
    record: dict[str, Any] = {
        "schema": schema,
        "generated_by": f"tools/rl/v8_31_followup.py {mode}",
        "created_at": started_at,
        "classification": "measured",
        "repo": repo.as_posix(),
        "out": out_root.as_posix(),
        "argv": list(getattr(args, "argv", []) or []),
        "host": host_identity(),
        "heads": git_identity(repo),
        "tool": {"path": TOOL_PATH.as_posix(), "sha256": sha256_file(TOOL_PATH)},
        "slice_helper": slice_identity(),
        "limits": limits.as_dict(),
        "engine_started": False,
        "counts": {"planned": 1, "started": 0, "technical_invalid": 0,
                   "not_started": 1, "effective": 0},
    }
    exit_code = EXIT_TECHNICAL
    attempt: str | None = None
    ledger: Ledger | None = None
    measured = {"engine_frames": 0, "wall_s": 0.0, "write_gib": 0.0}
    # every list the artefact writer touches exists before the walk, so a refusal
    # at any point still writes a readable (if empty) evidence set
    hashes: dict[int, str] = {}
    captures: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    resources: dict[str, Any] = {}
    actor_inputs: dict[str, Any] | None = None
    executed_units: list[dict[str, Any]] = []
    continuation_decisions: list[dict[str, Any]] = []
    extra_decisions: list[dict[str, Any]] = []
    frames_pumped = 0
    sent_frames = 0
    stop: dict[str, Any] | None = None
    meter = module.CostMeter()
    meter.begin()
    try:
        with meter.measure("preflight"):
            preflight = engine_preflight(args, out_root)
            plan = d1_replay_plan(repo)
        record["preflight"] = preflight
        record["d1"] = {key: value for key, value in plan.items()
                        if key not in ("step_masks", "flat_masks", "hash_pre",
                                       "hash_post", "expected_pre", "expected_post")}
        gate.add("d1_source_registered",
                 plan["sha256"] == plan["registered_sha256"],
                 {"path": plan["path"], "sha256": plan["sha256"],
                  "registered": plan["registered_sha256"]})
        gate.add("d1_anchor_chain_unbroken", plan["chain_breaks"] == 0,
                 {"shared_anchors": plan["shared_anchors"],
                  "breaks": plan["chain_breaks"],
                  "first_break": plan["first_chain_break"]})

        units, last_frame = _plan_for_mode(mode, plan["step_masks"])
        plan_check = _check_plan(units, plan["step_masks"], last_frame)
        record["plan"] = {"summary": plan_summary(units), "check": plan_check,
                          "steps": units if len(units) <= 32 else units[:8],
                          "steps_kept": min(len(units), 32)}
        gate.add("step_masks_unmodified", plan_check["units_expand_to_the_d1_trace"],
                 plan_check)
        gate.add("capture_window_reachable", plan_check["window_frames_are_step_ends"],
                 plan_check["missing_window_ends"])
        if not plan_check["units_expand_to_the_d1_trace"]:
            raise EngineRefused("the STEP plan does not expand to the D1 mask trace")
        if not plan_check["window_frames_are_step_ends"]:
            raise EngineRefused(
                f"the STEP plan does not end on {plan_check['missing_window_ends']}")

        # -- the private game directory and the muted copy ------------------
        paths["base"].mkdir(parents=True, exist_ok=True)
        paths["frames_dir"].mkdir(parents=True, exist_ok=True)
        source_dir = Path(preflight["data_source"])
        with meter.measure("preflight"):
            source_before = tree_fingerprint(source_dir)
        private = prepare_private_game_dir(out_dir=paths["base"],
                                           data_source=source_dir, meter=meter)
        record["private_game_dir"] = private
        source_cfg = source_dir / "th08.cfg"
        copy_cfg = Path(private["copy_dir"]) / "th08.cfg"
        source_raw = source_cfg.read_bytes()
        copy_before_raw = copy_cfg.read_bytes()
        checked_at = [now_iso()]
        _, mute = module.write_mute_json(
            out_dir=paths["base"],
            paths={"source": source_cfg, "copy": copy_cfg},
            source_raw=source_raw, copy_before_raw=copy_before_raw,
            copy_after_raw=None, checked_at=checked_at, problems=[], meter=meter)
        mute_problems = _mute_gate(mute)
        record["mute"] = {
            "path": paths["mute"].as_posix(),
            "before_start_fields": mute["copy_before_start"].get("fields"),
            "before_start_problems": mute_problems,
            "mute_byte_check": private["mute_byte_check"],
            "engine_env": mute["engine_env"],
        }
        gate.add("mute_before_start", not mute_problems, mute_problems)
        gate.add("mute_bytes_all_zero",
                 bool(private["mute_byte_check"]["all_zero"]),
                 private["mute_byte_check"])
        if mute_problems or not private["mute_byte_check"]["all_zero"]:
            raise EngineRefused(
                "refusing to start the engine with a non-muted private copy: "
                + "; ".join(mute_problems or ["mute bytes are not all zero"]))

        # -- reserve, then start -------------------------------------------
        ledger = Ledger(out_root / "ledger.jsonl", limits)
        attempt = f"02a-{mode}"
        reservation = ledger.reserve(
            attempt, starts=1, engine_frames=int(last_frame), wall_s=900.0,
            write_gib=1.0,
            inputs={"engine": preflight["engine"],
                    "engine_sha256": preflight["engine_sha256"],
                    "data_source": preflight["data_source"],
                    "d1_sha256": plan["sha256"],
                    "planned_frames": int(last_frame),
                    "steps": len(units),
                    "reset": preflight["reset"]},
            note=f"{mode}: one RESET, {last_frame} frames, {len(units)} STEP units")
        record["ledger"] = {"attempt_id": attempt, "reserve_seq": reservation["seq"],
                            "limits": limits.as_dict()}

        shm_name = module.make_shm_name(f"followup02a-{mode}")
        host = module.EngineHost(
            engine=preflight["engine"],
            data_dir=Path(private["copy_dir"]),
            shm_name=shm_name,
            timeout=max(60.0, float(getattr(args, "timeout_s", 120.0) or 120.0)),
            env=dict(module.FORCED_ENGINE_ENV),
            meter=meter,
        )
        session = host.open(stage=RESET_STAGE,
                            difficulty_index=int(
                                preflight["reset"]["difficulty_index"]),
                            shot_type=RESET_TEAM, seed=RESET_SEED)
        record["engine_started"] = True
        record["counts"].update({"started": 1, "not_started": 0})
        record["engine_session"] = session
        record["engine_env"] = module.engine_env_record()
        reset_reply = dict(session["reset"])
        reset_hash = str(reset_reply.get("hash", "")).strip().lower()
        record["reset"] = {
            "reply": reset_reply,
            "hash": reset_hash,
            "frame": int(reset_reply.get("frames", -1)),
            "setup_frames": int(reset_reply.get("setup_frames", -1)),
            "expected_d1_hash_pre_at_0": plan["reset_hash"],
            "matches": reset_hash == plan["reset_hash"],
            "role": "separate reset check; never counted as a hash_post",
            "d1_anchor_frame": plan["reset_frame"],
        }
        gate.add("reset_hash_matches_d1_pre0", reset_hash == plan["reset_hash"],
                 record["reset"])

        history = module.ActionHistoryBuffer()
        tap = module.ActorInputTap()
        capture_set = set(WINDOW_STEPS)
        try:
            for unit in units:
                last = int(unit["start"]) + int(unit["frames"]) - 1
                reply = host.step(int(unit["mask"]), int(unit["frames"]))
                advanced = int(reply.get("advanced", 0) or 0)
                sent_frames += advanced
                executed_units.append({"start": int(unit["start"]),
                                       "frames": int(unit["frames"]),
                                       "mask": int(unit["mask"]),
                                       "granularity": unit["granularity"],
                                       "forced_split": bool(unit.get("forced_split")),
                                       "advanced": advanced})
                hash_value = reply.get("hash")
                if hash_value is not None:
                    hashes[last] = str(hash_value).strip().lower()
                capture_row: dict[str, Any] | None = None
                if advanced == int(unit["frames"]) and "exit" not in reply:
                    if last in capture_set:
                        capture = module.capture_shared_block(
                            host, meter=meter, checksum=True)
                        events = module.parse_events_reply_full(host.events_raw())
                        relation = _classify_capture(last, reply, capture, events)
                        top_down = bytes(capture["rgba"].tobytes())
                        raw_order = bytes(capture["rgba"][::-1].tobytes())
                        if (module.pixel_frame_blake2b128(raw_order)
                                != capture[module.PIXEL_FRAME_DIGEST_FIELD]):
                            raise FrameRelationConflict(
                                "the recomputed blake2b-128 disagrees with the "
                                "capture helper")
                        pixel_frame = last - 1
                        png_path = paths["frames_dir"] / f"frame_pix{pixel_frame}.png"
                        png_data = png_bytes(top_down, int(capture["width"]),
                                             int(capture["height"]))
                        png_path.write_bytes(png_data)
                        capture_row = {
                            "captured_at_step_F": int(last),
                            "pixel_frame": int(pixel_frame),
                            "header_frame": int(capture["header_frame"]),
                            "events_frames": int(events["frames"]),
                            "step_reply_frame": int(reply.get("frame", -1)),
                            "relation": relation["relation"],
                            "relation_why": relation["why"],
                            "legal_header_frames": relation["legal_header_frames"],
                            "width": int(capture["width"]),
                            "height": int(capture["height"]),
                            "rgba_bytes": int(capture["rgba_bytes"]),
                            "pixels_nonzero": int(capture["pixels_nonzero"]),
                            "blake2b128": capture[module.PIXEL_FRAME_DIGEST_FIELD],
                            "sha256_raw": sha256_bytes(raw_order),
                            "rgba_sha256": capture["rgba_sha256"],
                            "pixels_checksum_local": capture["pixels_checksum_local"],
                            "png": repo_rel(png_path, out_root),
                            "png_sha256": sha256_bytes(png_data),
                            "png_bytes": len(png_data),
                            "stage_clear": int(events["stage_clear"]),
                            "player_hit": bool(events["player_hit"]),
                            "classification": "measured",
                        }
                        if mode == "reference":
                            engine_pixels = host.pixels_checksum()
                            capture_row["pixels_reply"] = {
                                key: engine_pixels.get(key)
                                for key in ("pixels", "width", "height", "nonzero",
                                            "checksum")}
                            capture_row["engine_equals_local_fnv"] = bool(
                                str(engine_pixels.get("checksum", "")).lower()
                                == str(capture["pixels_checksum_local"]).lower())
                            capture_row["engine_nonzero_equals_local"] = bool(
                                engine_pixels.get("nonzero")
                                == int(capture["pixels_nonzero"]))
                        captures.append(capture_row)
                        if pixel_frame in STACK_PIXEL_FRAMES:
                            tap.pump(capture["rgba"])
                            frames_pumped += 1
                    if last in RESOURCE_FRAMES:
                        state = host.state()
                        with host.meter.measure("audit"):
                            player = parse_player_reply(
                                host.client.request("PLAYER"))
                        hud = host.hud()
                        resources[str(last)] = resource_record(
                            frame=last, state=state, player=player, hud=hud)
                    if last % DECISION_FRAMES == 0 and last >= DECISION_FRAMES:
                        decision_index = last // DECISION_FRAMES - 1
                        first_pair_frame = DECISION_FRAMES * decision_index + 1
                        pair = [int(plan["step_masks"][first_pair_frame]),
                                int(plan["step_masks"][first_pair_frame + 1])]
                        row = (_decode_decision(pair, decision_index)
                               if mode == "prefix"
                               else _decode_decision_direct(pair, decision_index))
                        row["decision_index"] = decision_index
                        row["pair_frames"] = [first_pair_frame, first_pair_frame + 1]
                        row["hash_at_pair_end"] = hashes.get(last)
                        if row["cand"] is not None:
                            history.push(int(row["cand"]))
                        if last <= PREFIX_FRAME:
                            history_rows.append(row)
                        elif last <= CONTINUATION_LAST_FRAME:
                            continuation_decisions.append(row)
                        else:
                            extra_decisions.append(row)
                    if last == PREFIX_FRAME and actor_inputs is None:
                        inputs = tap.inputs(history)
                        digests = inputs["digests"]
                        actor_inputs = {
                            "frame": PREFIX_FRAME,
                            "stack_pixel_frames": list(STACK_PIXEL_FRAMES),
                            "stack_order": "oldest first",
                            "frames_pumped": int(frames_pumped),
                            "history_entries": int(history.pushes),
                            "next_decision_index": int(len(history_rows)),
                            "play": digests["play"],
                            "hud": digests["hud"],
                            "act_hist": digests["act_hist"],
                            "actor_input_digest": digests["actor_input_digest"],
                            "play_shape": [int(value) for value in inputs["play"].shape],
                            "hud_shape": [int(value) for value in inputs["hud"].shape],
                            "act_hist_shape": [int(value)
                                               for value in inputs["act_hist"].shape],
                            "policy_control": False,
                            "used_for_control": False,
                            "digest_basis": ("uint8 planes: play = stack(), hud = hud(), "
                                             "act_hist = history.rows(); "
                                             "digest_actor_arrays"),
                            "classification": "measured",
                        }
                        record["actor_inputs"] = actor_inputs
                        record["history_at_boundary"] = {
                            "entries": int(history.pushes),
                            "next_decision_index": int(len(history_rows)),
                            "last_eight": [dict(row) for row in history_rows[-8:]],
                        }
                if last >= WINDOW_STEPS[0]:
                    frame_rows.append({
                        "schema": SCHEMA_FRAMES,
                        "step_frame": int(last),
                        "unit_start": int(unit["start"]),
                        "unit_frames": int(unit["frames"]),
                        "mask": int(unit["mask"]),
                        "granularity": unit["granularity"],
                        "forced_split": bool(unit.get("forced_split")),
                        "advanced": advanced,
                        "reply_frame": int(reply.get("frame", -1)),
                        "hash": hashes.get(last),
                        "exit": bool("exit" in reply),
                        "game_over": int(reply.get("game_over", 0) or 0),
                        "phase": ("prefix" if last <= PREFIX_FRAME
                                  else "continuation"),
                        "captured": (None if capture_row is None else {
                            "pixel_frame": capture_row["pixel_frame"],
                            "blake2b128": capture_row["blake2b128"],
                            "sha256_raw": capture_row["sha256_raw"],
                            "png": capture_row["png"],
                            "png_sha256": capture_row["png_sha256"],
                        }),
                        "resources_recorded": last in RESOURCE_FRAMES,
                        "classification": "measured",
                    })
                if advanced != int(unit["frames"]) or "exit" in reply:
                    stop = {"kind": ("engine_exit" if "exit" in reply
                                     else "frame_shortfall"),
                            "frame": int(last),
                            "requested_frames": int(unit["frames"]),
                            "advanced": advanced,
                            "exit": bool("exit" in reply),
                            "game_over": int(reply.get("game_over", 0) or 0),
                            "reply": dict(reply)}
                    break
        finally:
            with meter.measure("session_close"):
                host.close()

        executed_trace = [(frame, mask) for unit in executed_units
                          for frame, mask in [
                              (int(unit["start"]) + offset, int(unit["mask"]))
                              for offset in range(int(unit["frames"]))]]
        expected_trace = [(frame, int(plan["step_masks"][frame]))
                          for frame in range(1, sent_frames + 1)]
        record["mask_trace"] = {
            "first_frame": 1,
            "last_frame": PREFIX_FRAME,
            "digest": mask_trace_digest(plan["step_masks"], 1, PREFIX_FRAME),
            "digest_definition": ('sha256 of "<frame>:<mask>\\n" for every frame '
                                  "1..8664, expanded from the STEP units"),
            "executed_digest": mask_trace_digest(
                {frame: mask for frame, mask in executed_trace}, 1,
                min(PREFIX_FRAME, sent_frames)),
            "executed_units": len(executed_units),
            "executed_frames": sent_frames,
            "executed_masks_match_d1": executed_trace == expected_trace,
            "plan": plan_summary(units),
            "classification": "measured",
        }
        record["engine_frame"] = {
            "prefix_boundary": PREFIX_FRAME,
            "planned_last_frame": int(last_frame),
            "reached_last_frame": max(hashes) if hashes else 0,
            "prefix_boundary_reached": PREFIX_FRAME in hashes,
            "advanced_total": int(sent_frames),
            "hash_count": len(hashes),
            "classification": "measured",
        }
        record["stop"] = stop
        record["captures"] = captures
        record["resources"] = resources
        record["history_rows"] = len(history_rows)
        record["continuation_decisions"] = continuation_decisions
        record["extra_decisions"] = extra_decisions
        if mode == "prefix":
            actual = hashes.get(CONTINUATION_LAST_FRAME)
            expected = plan["continuation_expected"]
            record["continuation"] = {
                "frames": CONTINUATION_FRAMES,
                "start_frame": PREFIX_FRAME + 1,
                "end_frame": CONTINUATION_LAST_FRAME,
                "expected_hash": expected,
                "actual_hash": actual,
                "expected_source": "D1 row 2,166 hash_post (frame 8,668)",
                "match": bool(expected is not None and actual == expected),
                "d1_hash_pre_also": plan["continuation_pre_also"],
                "d1_pre_matches_post": plan["continuation_pre_matches_post"],
                "decisions": continuation_decisions,
                "hash_chain_continuous": bool(
                    actual is not None and expected is not None
                    and actual == expected),
            }
        if stop is not None:
            raise EngineRefused(
                f"the walk stopped early at frame {stop['frame']}: {stop['kind']}")

        # -- anchors -------------------------------------------------------
        pre_report = anchor_report(
            expected=plan["expected_pre"], actual=hashes, first_frame=1,
            last_frame=PREFIX_FRAME - DECISION_FRAMES,
            label="hash_pre anchors (D1 frames 4,8,...,8,660)")
        post_report = anchor_report(
            expected=plan["expected_post"], actual=hashes, first_frame=1,
            last_frame=PREFIX_FRAME,
            label="hash_post anchors (D1 frames 4,8,...,8,664)")
        union: dict[int, str] = dict(plan["expected_pre"])
        union.update(plan["expected_post"])
        combined = anchor_report(expected=union, actual=hashes, first_frame=1,
                                 last_frame=PREFIX_FRAME,
                                 label="combined anchors (union of pre and post)")
        record["anchors"] = {"pre": pre_report, "post": post_report,
                             "combined": combined}
        gate.add("pre_anchors", pre_report["mismatches"] == 0, pre_report)
        gate.add("post_anchors", post_report["mismatches"] == 0, post_report)
        gate.add("combined_anchors", combined["mismatches"] == 0,
                 {"compared": combined["compared"],
                  "mismatches": combined["mismatches"]})
        gate.add("pre_denominator", pre_report["expected"] == 2165,
                 {"expected": pre_report["expected"], "ticket": 2165})
        gate.add("post_denominator", post_report["expected"] == 2166,
                 {"expected": post_report["expected"], "ticket": 2166})

        # -- the gate ------------------------------------------------------
        frame_pixels = {row["pixel_frame"] for row in captures}
        gate.add("seven_frames_captured",
                 len(captures) == 7 and frame_pixels == set(WINDOW_PIXEL_FRAMES),
                 {"captured": len(captures),
                  "pixel_frames": sorted(frame_pixels),
                  "expected": list(WINDOW_PIXEL_FRAMES)})
        gate.add("frame_relation_clean",
                 all(row["header_frame"] == row["events_frames"]
                     for row in captures)
                 and all(row["pixel_frame"] == row["captured_at_step_F"] - 1
                         for row in captures),
                 [{"captured_at_step_F": row["captured_at_step_F"],
                   "pixel_frame": row["pixel_frame"],
                   "header_frame": row["header_frame"],
                   "events_frames": row["events_frames"],
                   "relation": row["relation"]} for row in captures])
        gate.add("four_stack_frames_pumped", int(frames_pumped) == 4,
                 {"frames_pumped": int(frames_pumped),
                  "stack_pixel_frames": list(STACK_PIXEL_FRAMES)})
        gate.add("actor_inputs_recorded", bool(actor_inputs),
                 None if actor_inputs is None else {
                     key: actor_inputs[key] for key in
                     ("play", "hud", "act_hist", "actor_input_digest",
                      "history_entries", "next_decision_index", "frames_pumped")})
        gate.add("resources_recorded_twice",
                 set(resources) == {str(frame) for frame in RESOURCE_FRAMES},
                 sorted(resources))
        gate.add("engine_frame_prefix_boundary",
                 bool(record["engine_frame"]["prefix_boundary_reached"]),
                 record["engine_frame"])
        if mode == "prefix":
            history_at = record.get("history_at_boundary") or {}
            gate.add("history_entries_4332",
                     int(history_at.get("entries", -1)) == PREFIX_DECISIONS,
                     {"entries": history_at.get("entries"),
                      "expected": PREFIX_DECISIONS})
            gate.add("next_decision_index_4332",
                     int(history_at.get("next_decision_index", -1)) == PREFIX_DECISIONS,
                     {"next_decision_index": history_at.get("next_decision_index"),
                      "expected": PREFIX_DECISIONS})
            gate.add("last_eight_actions_recorded",
                     len(history_at.get("last_eight") or []) == 8,
                     [row["decision_index"]
                      for row in (history_at.get("last_eight") or [])])
            gate.add("continuation_hash_matches",
                     bool(record["continuation"]["match"]),
                     {key: record["continuation"][key] for key in
                      ("expected_hash", "actual_hash", "match",
                       "end_frame", "d1_pre_matches_post")})
        else:
            matches = [row.get("engine_equals_local_fnv") for row in captures]
            gate.add("pixels_reply_matches_local_fnv",
                     bool(matches) and all(matches),
                     {"per_frame": matches,
                      "note": "ENGINE-side FNV-1a-64 (PIXELS) vs the local FNV of "
                              "the captured block bytes"})
            gate.add("pixels_reply_nonzero_matches_local",
                     all(row.get("engine_nonzero_equals_local") for row in captures),
                     [row.get("engine_nonzero_equals_local") for row in captures])

        # -- post-exit mute, source immutability, accounting ---------------
        copy_after_raw = copy_cfg.read_bytes()
        checked_at.append(now_iso())
        _, mute_after = module.write_mute_json(
            out_dir=paths["base"],
            paths={"source": source_cfg, "copy": copy_cfg},
            source_raw=source_raw, copy_before_raw=copy_before_raw,
            copy_after_raw=copy_after_raw,
            checked_at=checked_at, problems=[], meter=meter)
        post_problems = module.mute_report_problems(mute_after)
        record["mute"].update({
            "after_exit_fields": (mute_after.get("copy_after_exit") or {}).get("fields"),
            "after_exit_problems": post_problems,
            "after_exit_sha256": (mute_after.get("copy_after_exit") or {}).get("sha256"),
            "copied_entries": private.get("copied_entries"),
        })
        gate.add("mute_after_exit", not post_problems, post_problems)
        with meter.measure("preflight"):
            source_after = tree_fingerprint(source_dir)
        source_unchanged = (source_after["digest"] == source_before["digest"]
                            and source_after["file_sha256"] == source_before["file_sha256"])
        record["source_dir"] = {
            "before": source_before, "after": source_after,
            "unchanged": bool(source_unchanged),
        }
        gate.add("source_dir_unchanged", source_unchanged, {
            "digest_before": source_before["digest"],
            "digest_after": source_after["digest"],
            "file_sha256_before": source_before["file_sha256"],
            "file_sha256_after": source_after["file_sha256"]})

        measured["engine_frames"] = int(sent_frames)
        measured["wall_s"] = round(time.perf_counter() - started, 3)
        measured["write_gib"] = round(du_bytes(paths["base"]) / 1024 ** 3, 3)
        record["cost"] = meter.payload(step_units=len(executed_units))
        record["counts"]["effective"] = 1
        record["terminal"] = {"kind": "completed", "reached_frame": int(sent_frames),
                              "stop": stop, "classification": "measured"}
        exit_code = EXIT_OK if gate.passed else EXIT_GATE_FAILED
    except BudgetRefused:
        raise
    except (EngineRefused, FrameRelationConflict, SourceError) as exc:
        record["counts"]["technical_invalid"] = 1
        record["terminal"] = {"kind": "technical_invalid",
                              "reason": type(exc).__name__,
                              "detail": str(exc),
                              "engine_started": bool(record.get("engine_started"))}
        exit_code = EXIT_TECHNICAL
    except Exception as exc:  # noqa: BLE001 - evidence before any traceback
        record["counts"]["technical_invalid"] = 1
        record["terminal"] = {"kind": "technical_invalid",
                              "reason": type(exc).__name__,
                              "detail": str(exc),
                              "traceback": traceback.format_exc(),
                              "engine_started": bool(record.get("engine_started"))}
        exit_code = EXIT_TECHNICAL
    finally:
        meter.finish()

    record.setdefault("terminal", {"kind": "unknown"})
    record.setdefault("cost", meter.payload(step_units=0))
    record["finished_at"] = now_iso()
    measured["wall_s"] = measured["wall_s"] or round(time.perf_counter() - started, 3)
    if ledger is not None and attempt is not None and attempt in ledger.reservations():
        settlement = ledger.settle(
            attempt, starts=1, engine_frames=measured["engine_frames"],
            wall_s=measured["wall_s"], write_gib=max(0.0, measured["write_gib"]),
            result={"command": f"02a-{mode}", "exit_code": exit_code,
                    "terminal": record["terminal"].get("kind"),
                    "reached_frame": record.get("engine_frame", {}).get(
                        "reached_last_frame")},
            note=f"{mode}: settled after the engine process exited")
        record["ledger"]["settle_seq"] = settlement["seq"]
        record["ledger"]["actual"] = settlement["actual"]
    elif ledger is not None:
        record["ledger"]["settled"] = False
    if ledger is not None:
        record["ledger"]["totals"] = ledger.totals()

    # the per-frame and per-decision evidence is written before the gate, so a
    # failed gate still leaves the rows that show why
    written: dict[str, Any] = {}
    frame_rows_path = paths["frames"]
    history_rows_path = paths["history"]
    hashes_path = paths["hashes"]
    written["frames_rows"] = _write_jsonl(frame_rows_path, frame_rows)
    written["history_rows"] = _write_jsonl(history_rows_path, history_rows)
    written["hash_rows"] = _write_jsonl(
        hashes_path, ({"frame": frame, "hash": hashes[frame]}
                      for frame in sorted(hashes)))
    record["history_digest"] = _history_digest(history_rows)
    takeover = {
        "schema": SCHEMA_TAKEOVER,
        "generated_by": f"tools/rl/v8_31_followup.py {mode}",
        "created_at": now_iso(),
        "package": mode,
        "classification": "measured",
        "frame_relation": PIXEL_FRAME_RELATION,
        "boundary_frame": PREFIX_FRAME,
        "decision_count_at_boundary": PREFIX_DECISIONS,
        "window": {"steps": list(WINDOW_STEPS),
                   "pixel_frames": list(WINDOW_PIXEL_FRAMES),
                   "captures": captures},
        "stack": {"pixel_frames": list(STACK_PIXEL_FRAMES),
                  "order": "oldest first",
                  "frames_pumped": int(frames_pumped)},
        "history": record.get("history_at_boundary"),
        "actor_inputs": actor_inputs,
        "resources": resources,
        "policy_control": False,
        "notes": [
            "no policy was consulted; the STEP masks are D1's recorded masks",
            "the obs-v3 stack and the action history are rebuilt only so the "
            "three actor-input digests can be compared",
        ],
    }
    written["takeover_sha256"] = sha256_file(write_json(paths["takeover"], takeover))
    record["artifacts"] = written
    record["gate"] = gate.as_dict()
    record["classification"] = "measured"
    result_path = write_json(paths["result"], record)
    for check in gate.checks:
        print(f"  {'ok  ' if check.ok else 'FAIL'} {check.name}")
    engine_frame = record.get("engine_frame") or {}
    print(f"{mode}: terminal={record['terminal'].get('kind')} "
          f"frames={engine_frame.get('advanced_total')} "
          f"steps={len(executed_units)} "
          f"captures={(record.get('captures') or []) and len(record['captures'])} "
          f"rc={exit_code}")
    print(f"wrote: {paths['base'].as_posix()} (result.json sha256="
          f"{sha256_file(result_path)[:16]}...)")
    if exit_code == EXIT_OK:
        print(f"gate: passed ({len(gate.checks)} checks)")
    else:
        print(f"gate: {'FAILED' if exit_code == EXIT_GATE_FAILED else 'technical'} "
              f"({len(gate.problems)} failed checks)")
        for problem in gate.problems[:12]:
            print(f"  - {problem}")
    return exit_code


def cmd_prefix(args: argparse.Namespace) -> int:
    """02-A.3: replay the D1 prefix to frame 8,664, take over, continue 4 frames."""

    return run_engine_package(args, "prefix")


def cmd_reference(args: argparse.Namespace) -> int:
    """02-A.4: an independent second start with a different STEP granularity."""

    return run_engine_package(args, "reference")


# ---------------------------------------------------------------------------
# 02-A.4 `compare`: the offline equality of the two runs
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def first_difference(left: Any, right: Any, path: str = "root") -> dict[str, Any] | None:
    """The first place two JSON values differ, with both sides verbatim."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = list(left.keys()) + [key for key in right.keys() if key not in left]
        for key in keys:
            if key not in left or key not in right:
                return {"path": f"{path}.{key}",
                        "reason": "present on one side only",
                        "left": left.get(key), "right": right.get(key)}
            found = first_difference(left[key], right[key], f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return {"path": f"{path}[len]", "reason": "length differs",
                    "left_length": len(left), "right_length": len(right),
                    "left": list(left), "right": list(right)}
        for index in range(len(left)):
            found = first_difference(left[index], right[index], f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if isinstance(left, bool) or isinstance(right, bool):
        same = left is right
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        same = float(left) == float(right)
    else:
        same = left == right
    if same:
        return None
    return {"path": path, "reason": "value differs", "left": left, "right": right}


def _frame_keyed(captures: Sequence[Mapping[str, Any]],
                 key: str) -> list[dict[str, Any]]:
    """``[{"pixel_frame": .., key: ..}, ...]`` in capture order."""

    return [{"pixel_frame": int(row["pixel_frame"]), key: row.get(key)}
            for row in captures]


def _png_recheck(root: Path, captures: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Re-derive each stored frame's digests from the PNG file itself."""

    rows = []
    for capture in captures:
        rel = capture.get("png")
        path = (root / rel) if rel else None
        entry: dict[str, Any] = {"pixel_frame": int(capture["pixel_frame"]),
                                 "path": rel}
        if path is None or not path.is_file():
            entry.update({"error": "missing PNG", "ok": False})
            rows.append(entry)
            continue
        try:
            fresh = frame_digests_from_png(path)
        except (OSError, ValueError) as exc:
            entry.update({"error": f"{type(exc).__name__}: {exc}", "ok": False})
            rows.append(entry)
            continue
        entry.update({
            "ok": bool(fresh["blake2b128"] == capture.get("blake2b128")
                       and fresh["sha256_raw"] == capture.get("sha256_raw")
                       and fresh["png_sha256"] == capture.get("png_sha256")),
            "recorded": {"blake2b128": capture.get("blake2b128"),
                         "sha256_raw": capture.get("sha256_raw"),
                         "png_sha256": capture.get("png_sha256")},
            "recomputed": {"blake2b128": fresh["blake2b128"],
                           "sha256_raw": fresh["sha256_raw"],
                           "png_sha256": fresh["png_sha256"],
                           "width": fresh["width"], "height": fresh["height"],
                           "bytes": fresh["bytes"]},
            "first_difference": first_difference(
                {"blake2b128": capture.get("blake2b128"),
                 "sha256_raw": capture.get("sha256_raw"),
                 "png_sha256": capture.get("png_sha256")},
                {"blake2b128": fresh["blake2b128"],
                 "sha256_raw": fresh["sha256_raw"],
                 "png_sha256": fresh["png_sha256"]}, rel or "png"),
        })
        rows.append(entry)
    return rows


def compare_runs(out_root: str | Path) -> dict[str, Any]:
    """Re-read both runs and report every equality item, with its first difference.

    Pure function of the files under ``out_root``: it starts nothing, trusts no
    recorded digest (the PNGs are decoded again), and reports the first
    difference verbatim instead of smoothing it.
    """

    root = Path(out_root).expanduser().resolve()
    prefix_base = root / "02a" / "prefix"
    reference_base = root / "02a" / "reference"
    gate = Gate("02a_compare")
    items: list[dict[str, Any]] = []
    missing = [path.as_posix() for path in (
        prefix_base / "result.json", prefix_base / "takeover.json",
        prefix_base / "history.jsonl", prefix_base / "frames.jsonl",
        reference_base / "result.json", reference_base / "takeover.json",
        reference_base / "history.jsonl", reference_base / "frames.jsonl")
        if not path.is_file()]
    payload: dict[str, Any] = {
        "schema": SCHEMA_COMPARE,
        "generated_by": "tools/rl/v8_31_followup.py compare",
        "created_at": now_iso(),
        "classification": "offline",
        "out": root.as_posix(),
        "inputs": {"prefix": prefix_base.as_posix(),
                   "reference": reference_base.as_posix()},
        "missing": missing,
    }
    if missing:
        gate.add("inputs_present", False, missing)
        payload["items"] = items
        payload["gate"] = gate.as_dict()
        payload["first_difference"] = {"path": "inputs", "reason": "missing",
                                       "detail": missing}
        payload["equal_items"] = 0
        payload["different_items"] = 0
        payload["verdict"] = "differs"
        return payload
    gate.add("inputs_present", True, None)

    prefix = _load_json(prefix_base / "result.json")
    reference = _load_json(reference_base / "result.json")
    prefix_take = _load_json(prefix_base / "takeover.json")
    reference_take = _load_json(reference_base / "takeover.json")
    prefix_caps = list((prefix_take.get("window") or {}).get("captures") or [])
    reference_caps = list((reference_take.get("window") or {}).get("captures") or [])

    def record(name: str, left: Any, right: Any, *, extra: Mapping[str, Any] | None = None
               ) -> dict[str, Any]:
        difference = first_difference(left, right, name)
        row = {"item": name, "equal": difference is None,
               "first_difference": difference}
        if extra:
            row.update(dict(extra))
        items.append(row)
        gate.add(name, difference is None, row)
        return row

    # 1. the two runs claim the same frames, in the same order
    record("window_frame_order",
           [int(row["pixel_frame"]) for row in prefix_caps],
           [int(row["pixel_frame"]) for row in reference_caps],
           extra={"prefix_pixel_frames": [int(row["pixel_frame"]) for row in prefix_caps],
                  "reference_pixel_frames": [int(row["pixel_frame"])
                                             for row in reference_caps]})

    # 2. the seven per-frame digests
    for key in ("blake2b128", "sha256_raw", "png_sha256"):
        record(f"frames_7_of_7_{key}", _frame_keyed(prefix_caps, key),
               _frame_keyed(reference_caps, key),
               extra={"equal_frames": sum(
                   1 for left, right in zip(prefix_caps, reference_caps)
                   if left.get(key) == right.get(key)),
                   "frames": len(prefix_caps)})

    # 3. each run's PNG must still hash to its own recorded digest
    prefix_recheck = _png_recheck(root, prefix_caps)
    reference_recheck = _png_recheck(root, reference_caps)
    record("prefix_pngs_match_their_records",
           [row["ok"] for row in prefix_recheck], [True] * len(prefix_recheck),
           extra={"rows": prefix_recheck})
    record("reference_pngs_match_their_records",
           [row["ok"] for row in reference_recheck], [True] * len(reference_recheck),
           extra={"rows": reference_recheck})

    # 4. the frame relation and the granularity evidence
    record("frame_relation",
           [{"pixel_frame": row["pixel_frame"], "header_frame": row["header_frame"],
             "events_frames": row["events_frames"]} for row in prefix_caps],
           [{"pixel_frame": row["pixel_frame"], "header_frame": row["header_frame"],
             "events_frames": row["events_frames"]} for row in reference_caps])

    # 5. the mask trace (D1's masks, expanded to one mask per frame)
    record("mask_trace_sha256",
           (prefix.get("mask_trace") or {}).get("digest"),
           (reference.get("mask_trace") or {}).get("digest"),
           extra={"definition": (prefix.get("mask_trace") or {}).get("digest_definition"),
                  "prefix_plan": (prefix.get("mask_trace") or {}).get("plan"),
                  "reference_plan": (reference.get("mask_trace") or {}).get("plan")})

    # 6. the three actor-input digests plus the combined one
    prefix_actor = prefix.get("actor_inputs") or {}
    reference_actor = reference.get("actor_inputs") or {}
    record("obs_stack_and_history_digests",
           {key: prefix_actor.get(key) for key in
            ("play", "hud", "act_hist", "actor_input_digest",
             "history_entries", "next_decision_index", "frames_pumped")},
           {key: reference_actor.get(key) for key in
            ("play", "hud", "act_hist", "actor_input_digest",
             "history_entries", "next_decision_index", "frames_pumped")})

    # 7. the resource anchors at 8,663 and 8,664
    prefix_resources = prefix.get("resources") or {}
    reference_resources = reference.get("resources") or {}
    record("resource_frames_recorded", sorted(prefix_resources),
           sorted(reference_resources))
    for frame in RESOURCE_FRAMES:
        key = str(frame)
        record(f"resources_at_{frame}", prefix_resources.get(key),
               reference_resources.get(key))
        record(f"hud_at_{frame}",
               (prefix_resources.get(key) or {}).get("hud"),
               (reference_resources.get(key) or {}).get("hud"))
        record(f"position_at_{frame}",
               (prefix_resources.get(key) or {}).get("position"),
               (reference_resources.get(key) or {}).get("position"))
        record(f"state_at_{frame}",
               (prefix_resources.get(key) or {}).get("state"),
               (reference_resources.get(key) or {}).get("state"))

    # 8. the last eight complete actions
    prefix_history = [json.loads(line) for line in
                      (prefix_base / "history.jsonl").read_text(
                          encoding="utf-8").splitlines() if line.strip()]
    reference_history = [json.loads(line) for line in
                         (reference_base / "history.jsonl").read_text(
                             encoding="utf-8").splitlines() if line.strip()]
    record("history_row_count", len(prefix_history), len(reference_history),
           extra={"prefix": len(prefix_history), "reference": len(reference_history)})
    last_eight_prefix = [
        {"decision_index": row["decision_index"], "cand": row.get("cand"),
         "pair": row.get("pair")} for row in prefix_history[-8:]]
    last_eight_reference = [
        {"decision_index": row["decision_index"], "cand": row.get("cand"),
         "pair": row.get("pair")} for row in reference_history[-8:]]
    record("last_eight_actions", last_eight_prefix, last_eight_reference)
    record("action_sequence_sha256",
           sha256_bytes("".join(f"{row['decision_index']}:{row.get('cand')}\n"
                                for row in prefix_history).encode()),
           sha256_bytes("".join(f"{row['decision_index']}:{row.get('cand')}\n"
                                for row in reference_history).encode()))

    # 9. the reset hash and the anchor coverage
    record("reset_hash", (prefix.get("reset") or {}).get("hash"),
           (reference.get("reset") or {}).get("hash"))
    record("anchor_reports",
           {group: {"expected": (prefix.get("anchors") or {}).get(group, {}).get("expected"),
                    "compared": (prefix.get("anchors") or {}).get(group, {}).get("compared"),
                    "mismatches": (prefix.get("anchors") or {}).get(group, {}).get(
                        "mismatches")}
            for group in ("pre", "post", "combined")},
           {group: {"expected": (reference.get("anchors") or {}).get(group, {}).get(
               "expected"),
                    "compared": (reference.get("anchors") or {}).get(group, {}).get(
                        "compared"),
                    "mismatches": (reference.get("anchors") or {}).get(group, {}).get(
                        "mismatches")}
            for group in ("pre", "post", "combined")})

    # 10. the frames each step unit covered, in both runs
    prefix_frames = [json.loads(line) for line in
                     (prefix_base / "frames.jsonl").read_text(
                         encoding="utf-8").splitlines() if line.strip()]
    reference_frames = [json.loads(line) for line in
                        (reference_base / "frames.jsonl").read_text(
                            encoding="utf-8").splitlines() if line.strip()]
    shared_steps = sorted({int(row["step_frame"]) for row in prefix_frames}
                          & {int(row["step_frame"]) for row in reference_frames})
    record("shared_window_hashes",
           [{"step_frame": int(row["step_frame"]), "hash": row.get("hash")}
            for row in prefix_frames if int(row["step_frame"]) in shared_steps],
           [{"step_frame": int(row["step_frame"]), "hash": row.get("hash")}
            for row in reference_frames if int(row["step_frame"]) in shared_steps],
           extra={"step_frames": shared_steps})

    differences = [row for row in items if not row["equal"]]
    payload.update({
        "items": items,
        "equal_items": sum(1 for row in items if row["equal"]),
        "different_items": len(differences),
        "first_difference": (differences[0]["first_difference"]
                             if differences else None),
        "engine_frame": {"prefix": prefix.get("engine_frame"),
                         "reference": reference.get("engine_frame")},
        "step_granularity": {"prefix": (prefix.get("mask_trace") or {}).get("plan"),
                             "reference": (reference.get("mask_trace") or {}).get("plan")},
        "gate": gate.as_dict(),
        "verdict": ("identical" if not differences else "differs"),
    })
    return payload


def cmd_compare(args: argparse.Namespace) -> int:
    """02-A.4: offline equality of the prefix and reference runs."""

    out_root = Path(args.out).expanduser().resolve()
    payload = compare_runs(out_root)
    path = write_json(out_root / "02a" / "compare.json", payload)
    for row in payload["items"]:
        mark = "ok  " if row["equal"] else "FAIL"
        print(f"  {mark} {row['item']}")
        if not row["equal"]:
            print(f"       first difference: {json.dumps(row['first_difference'])[:400]}")
    print(f"compare: {payload['equal_items']}/{len(payload['items'])} items equal, "
          f"verdict={payload['verdict']}")
    print(f"wrote: {path}")
    if payload["missing"]:
        return EXIT_TECHNICAL
    return EXIT_OK if (payload["gate"] or {}).get("passed") else EXIT_GATE_FAILED


# ---------------------------------------------------------------------------
# 02-B / 02-C: the enemy-identity channel, the tape schedule, the shared budget
# ---------------------------------------------------------------------------
#: Everything below is built on exactly two engine channels and nothing else:
#:
#: ``HAZARDS``  with **no limit** returns every enemy slot (``slot < 481``,
#:              ``rlharness.cpp:679-681``) as
#:              ``slot:flags1:posXyz:life:timer3318:worldXyz:hitboxXy``; the whole
#:              reply is capped at 256 KiB and 481 worst-case rows are ~51 KB, so
#:              one call per decision always carries the boss row.
#: ``EVENTS``   ``frameEventFlags`` is a **u16 shared by the whole frame** and is
#:              cleared at the start of every frame (``ReplayManager.cpp:297``), so
#:              it is read once per *frame*, not once per decision.
#:
#: ``ENEMIES`` paging is never used: its ``offset``/``limit`` index the ACTIVE
#: slots by ordinal (``rlharness.cpp:537-542``), so one slot disappearing shifts
#: every later ordinal and two frames cannot be aligned.
#:
#: The flag bits and the damage cap are quoted from
#: ``.tmp_v831_followup_probe/enemy_semantics.md`` §1-§3, which in turn quotes the
#: RL-side engine tree (``/root/Touhou_RL/third_party/th08-rl/src``).  No bit
#: below is re-derived here.
EVENT_BITS: tuple[tuple[int, str], ...] = (
    (0x0001, "BOMB_STARTED"),
    (0x0002, "PLAYER_HIT"),
    (0x0004, "PLAYER_DEATH_COMMITTED"),
    (0x0020, "ENEMY_REMOVED"),
    (0x0040, "ITEM_COLLECTED"),
    (0x0080, "STAGE_CLEAR_LEVEL"),
    (0x0100, "PAUSE_RECORDED"),
    (0x0200, "PLAYER_DYING_STARTED"),
    (0x0400, "EFFECT_SPAWNED"),
    (0x0800, "BULLET_PATTERN_SPAWNED"),
    (0x1000, "ENEMY_SPAWNED"),
    (0x2000, "PLAYER_GRAZED"),
)
EVENT_BOMB_STARTED = 0x0001
EVENT_PLAYER_HIT = 0x0002
EVENT_PLAYER_DEATH_COMMITTED = 0x0004
EVENT_ENEMY_REMOVED = 0x0020
EVENT_ITEM_COLLECTED = 0x0040
EVENT_STAGE_CLEAR_LEVEL = 0x0080
EVENT_PLAYER_DYING_STARTED = 0x0200
EVENT_EFFECT_SPAWNED = 0x0400
EVENT_BULLET_PATTERN_SPAWNED = 0x0800
EVENT_ENEMY_SPAWNED = 0x1000
EVENT_PLAYER_GRAZED = 0x2000

#: ``Enemy::flags1`` (``EnemyManager.hpp:67-100``): 24 single bits + 3 masks.
ENEMY_FLAGS: tuple[tuple[int, str], ...] = (
    (0x00000001, "ACTIVE"),
    (0x00000002, "BOSS"),
    (0x00000004, "COLLISION"),
    (0x00000008, "DAMAGEABLE"),
    (0x00000010, "NO_SPRITE"),
    (0x00000020, "HIDE_PRIMARY_ANM"),
    (0x00000040, "ACCEPTS_DAMAGE"),
    (0x00000080, "SPECIAL_INTERACTION"),
    (0x00000100, "LINKED_CHILD"),
    (0x00000200, "INHERIT_PARENT_POSITION"),
    (0x00000400, "SUPPRESS_DEATH_EFFECTS"),
    (0x00000800, "YOUKAI_ALIGNED"),
    (0x00020000, "DEFER_BULLET_PATTERN"),
    (0x00040000, "MIRROR_MOVEMENT_X"),
    (0x00080000, "CLAMP_POSITION"),
    (0x00800000, "PERSIST_AFTER_DEATH"),
    (0x01000000, "HAS_BEEN_IN_BOUNDS"),
    (0x02000000, "ROTATE_ANM_WITH_MOVEMENT"),
    (0x04000000, "DISABLE_ECL_CALL_STACK"),
    (0x08000000, "TIMEOUT_SPELL"),
    (0x10000000, "ALLOW_OFFSCREEN"),
    (0x20000000, "SKIP_MOVEMENT"),
    (0x40000000, "PAUSE_TIMER"),
    (0x80000000, "NO_DAMAGE_DURING_STOP"),
)
ENEMY_FLAG_BOSS = 0x00000002
ENEMY_FLAG_DAMAGEABLE = 0x00000008
ENEMY_FLAG_ACCEPTS_DAMAGE = 0x00000040
ENEMY_FLAG_PERSIST_AFTER_DEATH = 0x00800000
ENEMY_FLAG_HAS_BEEN_IN_BOUNDS = 0x01000000
ENEMY_FLAG_ALLOW_OFFSCREEN = 0x10000000
ENEMY_DEATH_MODE_MASK = 0x00700000
ENEMY_DEATH_MODE_SHIFT = 20

#: ``EnemyManagerUpdate.cpp:366-367``: the per-frame damage on one enemy is
#: clamped to 70 *after* both hitboxes are summed, so no frame can take more.
ENEMY_MAX_DAMAGE_PER_FRAME = 70

#: The ticket's 02-B cap: 13,195 frames.  One decision is two frames, so the
#: decision grid stops at 13,194 and the remaining single frame is reported as
#: unreachable by the grid rather than silently rounded away.
RECON_FRAME_CAP = 13195
RECON_GRID_LAST_FRAME = (RECON_FRAME_CAP // DECISION_FRAMES) * DECISION_FRAMES
RECON_GRID_REMAINDER = RECON_FRAME_CAP - RECON_GRID_LAST_FRAME

#: The tape: T04's slots plus the recorded leading decisions and a declared tail.
T04_SOURCE_ID = "t04-2f-seed12"
RECON_TAIL_DECISIONS = 64
#: The recorded mask sequence must reproduce these two independent corpora.
RECORDED_MASK_CROSSCHECK = ("t04-2f-seed12", D1_SOURCE_ID)

#: 02-C's frozen frame cap (a decision is two frames: 300 decisions == 600 frames).
CARD_DECISION_CAP = 300
CARD_FRAME_CAP = 600
#: The frozen control arm: constant candidate 12, same pulse phase, same executor.
CONTROL_CAND = 12

#: This stage's allowance (02-B/C), set by the parent for this slice.
STAGE_CAP_STARTS = 3
STAGE_CAP_FRAMES = 20000
#: Ledger rows whose ``attempt_id`` starts with one of these belong to this stage.
STAGE_ATTEMPT_PREFIXES: tuple[str, ...] = ("02b-", "02c-")
#: The ticket's whole-ticket budget (``TRAINING_HOST_REQUEST_02`` 全单 row).
GRAND_TOTAL_STARTS = 28
GRAND_TOTAL_FRAMES = 360_000
GRAND_TOTAL_WALL_S = 8 * 3600.0
GRAND_TOTAL_WRITE_GIB = 18.0

SCHEMA_RECON = "v831-02b-recon/1"
SCHEMA_RECON_ROW = "v831-02b-decision/1"
SCHEMA_NOMINATION = "v831-02b-nomination/1"
SCHEMA_TASK_CARD = "v831-02b-task-card/1"
SCHEMA_INTERVENE = "v831-02c-intervene/1"
SCHEMA_INTERVENE_ROW = "v831-02c-decision/1"

#: The task card's ticket id, verbatim.
TICKET_ID = "V831-SLICE-02 / 02-B + 02-C"


def _bits_float32(token: str) -> float:
    """Decode one ``%08x`` ``FloatBits()`` field (the engine's own encoding)."""

    return struct.unpack("<f", struct.pack("<I", int(token, 16) & 0xFFFFFFFF))[0]


def signed_life(raw: int) -> int:
    """``life`` is an ``i32`` printed with ``%08x`` and read back unsigned.

    ``protocol.py:282`` decodes the field with ``int(parts[5], 16)``, so a negative
    life (only ``DEATH_MODE 2`` can produce one) arrives as ``0xFFFFFFxx``.  The
    sign is restored here, once, for every row this tool writes.
    """

    value = int(raw) & 0xFFFFFFFF
    return value - 2 ** 32 if value >= 2 ** 31 else value


def flags1_names(flags: int) -> list[str]:
    """The single-bit names set in ``flags1`` (masks are reported separately)."""

    return [name for bit, name in ENEMY_FLAGS if int(flags) & bit]


def enemy_row_record(text: str) -> dict[str, Any]:
    """One ``HAZARDS`` enemy row, decoded with the sign of ``life`` restored."""

    parts = str(text).split(":")
    if len(parts) < 7:
        raise EngineRefused(
            f"HAZARDS enemy row has {len(parts)} fields, needs 7: {text!r}")
    flags = int(parts[1], 16)
    life_raw = int(parts[5], 16)
    record: dict[str, Any] = {
        "slot": int(parts[0]),
        "flags1": f"0x{flags:08x}",
        "flags1_dec": flags,
        "flags1_names": flags1_names(flags),
        "life": signed_life(life_raw),
        "life_raw_unsigned": int(life_raw),
        "timer": int(parts[6]),
        "x": _bits_float32(parts[2]),
        "y": _bits_float32(parts[3]),
        "z": _bits_float32(parts[4]),
        "boss": bool(flags & ENEMY_FLAG_BOSS),
        "damageable": bool(flags & ENEMY_FLAG_DAMAGEABLE),
        "accepts_damage": bool(flags & ENEMY_FLAG_ACCEPTS_DAMAGE),
        "persist_after_death": bool(flags & ENEMY_FLAG_PERSIST_AFTER_DEATH),
        "has_been_in_bounds": bool(flags & ENEMY_FLAG_HAS_BEEN_IN_BOUNDS),
        "allow_offscreen": bool(flags & ENEMY_FLAG_ALLOW_OFFSCREEN),
        "death_mode": (flags & ENEMY_DEATH_MODE_MASK) >> ENEMY_DEATH_MODE_SHIFT,
        "world_x": None,
        "world_y": None,
        "world_z": None,
        "hitbox_w": None,
        "hitbox_h": None,
    }
    if len(parts) > 9:
        record["world_x"] = _bits_float32(parts[7])
        record["world_y"] = _bits_float32(parts[8])
        record["world_z"] = _bits_float32(parts[9])
    if len(parts) > 11:
        record["hitbox_w"] = _bits_float32(parts[10])
        record["hitbox_h"] = _bits_float32(parts[11])
    return record


def parse_hazards_enemies(reply: str) -> dict[str, Any]:
    """Decode a ``HAZARDS`` reply's header and its **enemy rows only**.

    ``nb``/``nl`` are the engine's own bullet/laser counts and are kept as such;
    the bullet and laser *rows* are deliberately not decoded.  This package needs
    enemy identity, and decoding 183 bullet rows per decision for numbers ``nb``
    already carries would be pure cost.  The reply is one round trip and one
    whole-frame snapshot, which is what makes a 6,600-decision recon affordable.
    """

    fields = _parse_kv(reply)
    if fields.get("_verb") != "OK":
        raise EngineRefused(f"HAZARDS reply is not OK: {reply[:200]!r}")
    for key in ("frames", "nb", "nl", "ne"):
        if key not in fields:
            raise EngineRefused(f"HAZARDS reply has no {key}: {reply[:200]!r}")
    raw = fields.get("enemies", "")
    rows = [enemy_row_record(text) for text in raw.split(",") if text]
    if len(rows) != int(fields["ne"]):
        raise EngineRefused(
            f"HAZARDS ne={fields['ne']} but {len(rows)} enemy rows were decoded")
    return {
        "frame": int(fields["frames"]),
        "bullets": int(fields["nb"]),
        "lasers": int(fields["nl"]),
        "enemies_count": int(fields["ne"]),
        "enemies": rows,
        "reply_bytes": len(reply),
    }


def _word_int(value: Any) -> int:
    """One ``EVENTS`` word that may arrive as ``0x...`` text or as an int."""

    if isinstance(value, str):
        return int(value, 16)
    return int(value or 0)


def event_names(flags: int) -> list[str]:
    """The names of every event bit set in one ``frameEventFlags`` word."""

    return [name for bit, name in EVENT_BITS if int(flags) & bit]


# -- the teacher tape -------------------------------------------------------
def decision_masks(cands: Sequence[int], index: int, *,
                   period: int = DEFAULT_PULSE_PERIOD,
                   decision_frames: int = DECISION_FRAMES) -> list[int]:
    """The two STEP masks the frozen executor emits for decision ``index``.

    ``commit_frames(cand, decision_frames=2, release_shoot=release_due(i, 48))``
    expands to ``[m, m]`` or, on a pulse decision, ``[m, m & ~SHOOT]``.  The
    expansion is the frozen function's output, never a local rule.
    """

    table = adapter()
    release = bool(table["release_due"](int(index), int(period)))
    parts = table["commit_frames"](int(cands[int(index)]),
                                   decision_frames=int(decision_frames),
                                   release_shoot=release)
    masks = expand_commit(parts)
    if len(masks) != int(decision_frames):
        raise SourceError(
            f"decision {index}: commit_frames expanded to {len(masks)} masks, "
            f"expected {decision_frames}")
    return [int(mask) for mask in masks]


def tape_schedule(repo: str | Path,
                  *, tail_decisions: int = RECON_TAIL_DECISIONS) -> dict[str, Any]:
    """Rebuild the seed-12 teacher tape: one candidate and two masks per decision.

    The ticket names ``load_clear_schedule(T04 slots, tail_decisions=64)``: the
    recorded shard's eight leading decisions (cand 7, mask 65), the 6,588 executed
    slots and a declared 64-decision tail that repeats the last action, i.e.
    6,660 decisions covering STEP frames 1..13,320.

    Decision ``i`` owns STEP frames ``2i+1`` and ``2i+2`` and the boundary frame
    ``2i``; that is what makes T04's ``hash_pre`` at frame ``16+2k`` the boundary
    of decision ``k+8``.
    """

    from touhou_rl.v420.rl import load_clear_schedule  # noqa: PLC0415

    table = adapter()
    spec = SOURCE_BY_ID[T04_SOURCE_ID]
    path = Path(repo) / spec["rel"]
    if not path.is_file():
        raise SourceError(f"the tape shard is missing: {path}")
    try:
        model = load_clear_schedule(path, tail_decisions=int(tail_decisions))
    except Exception as exc:  # noqa: BLE001 - reported as a source failure
        raise SourceError(
            f"cannot rebuild the tape schedule from {path}: "
            f"{type(exc).__name__}: {exc}") from exc
    schedule = [(int(frame), int(cand)) for frame, cand in model["schedule"]]
    cands = [cand for _frame, cand in schedule]
    masks = [decision_masks(cands, index) for index in range(len(cands))]
    offgrid = [index for index, (frame, _cand) in enumerate(schedule)
               if frame != index * DECISION_FRAMES]
    unknown = [{"decision": index, "cand": cand,
                "mask": int(table["cand_to_mask"](cand))}
               for index, cand in enumerate(cands)
               if int(table["cand_to_mask"](cand)) & int(table["AGENT_MASK"]) == 0]
    bombed = [{"decision": index, "cand": cand}
              for index, cand in enumerate(cands)
              if any(int(mask) & int(table["BTN_BOMB"]) for mask in masks[index])]
    leading = [cand for _frame, cand in schedule[:int(model["leading_decisions"])]]
    return {
        "shard": path.as_posix(),
        "shard_rel": repo_rel(path, repo),
        "sha256": sha256_file(path),
        "registered_sha256": REGISTERED_SHA256.get(T04_SOURCE_ID),
        "slots": int(model["slots"]),
        "leading_decisions": int(model["leading_decisions"]),
        "leading_candidates": list(model["leading_candidates"]),
        "leading_first_eight": leading,
        "tail_decisions": int(model["tail_decisions"]),
        "last_recorded_decision_frame": int(model["last_recorded_decision_frame"]),
        "decisions": len(schedule),
        "frames": len(schedule) * DECISION_FRAMES,
        "cands": cands,
        "masks": masks,
        "decision_frames": [frame for frame, _cand in schedule],
        "release_decisions": sum(1 for index in range(len(cands))
                                 if bool(table["release_due"](index,
                                                              DEFAULT_PULSE_PERIOD))),
        "offgrid_decisions": offgrid,
        "grid_is_two_frames_per_decision": not offgrid,
        "decisions_without_agent_bits": unknown,
        "decisions_carrying_bomb": bombed,
        "note": ("decision i owns STEP frames 2i+1 and 2i+2; the recorded shard "
                 "ends at frame 13,190 and the declared tail repeats its last "
                 "action so the tape reaches frame 13,320"),
        "classification": "offline",
    }


def tape_cross_check(repo: str | Path, tape: Mapping[str, Any]) -> dict[str, Any]:
    """Check the rebuilt mask sequence against the two corpora that recorded it.

    ``T04`` stores, per slot, the single mask the executor sent for the decision's
    **first** frame; ``D1`` stores the flat per-frame masks (``step_frames``, one
    mask per transition, ``flat[j]`` is the STEP into frame ``j+1``).  Both are
    compared frame for frame with no interpolation: the tape's two masks for
    decision ``i`` must equal ``[flat[2i], flat[2i+1]]`` and ``masks[i][0]`` must
    equal T04's recorded ``mask``.
    """

    repo_path = Path(repo)
    masks = tape["masks"]
    t04_spec = SOURCE_BY_ID["t04-2f-seed12"]
    t04_rows = [dict(row) for row in iter_jsonl(repo_path / t04_spec["rel"])]
    t04_rows.sort(key=lambda row: int(row["decision_frame"]))
    t04_mismatch: list[dict[str, Any]] = []
    t04_compared = 0
    for slot, row in enumerate(t04_rows):
        index = int(row["decision_frame"]) // DECISION_FRAMES
        if index >= len(masks):
            break
        t04_compared += 1
        recorded = int(row["mask"])
        if int(masks[index][0]) != recorded:
            t04_mismatch.append({"slot": slot, "decision": index,
                                 "decision_frame": int(row["decision_frame"]),
                                 "recorded": recorded, "tape": int(masks[index][0])})
    d1 = d1_replay_plan(repo_path)
    flat = d1["flat_masks"]
    d1_covered = len(flat) // DECISION_FRAMES
    d1_mismatch: list[dict[str, Any]] = []
    d1_compared = 0
    for index in range(len(masks)):
        want = [flat.get(2 * index), flat.get(2 * index + 1)]
        if want[0] is None or want[1] is None:
            break
        d1_compared += 1
        if [int(masks[index][0]), int(masks[index][1])] != [int(want[0]), int(want[1])]:
            d1_mismatch.append({"decision": index, "d1": [int(want[0]), int(want[1])],
                                "tape": [int(masks[index][0]), int(masks[index][1])]})
    # the same state hash recorded by two independent corpora at multiples of 4
    t04_hash: dict[int, str] = {}
    for slot, row in enumerate(t04_rows):
        frame = int(row["decision_frame"])
        value = str(row.get("hash_pre") or "").strip().lower()
        if value:
            t04_hash[frame] = value
    shared = sorted(frame for frame in t04_hash
                    if frame in d1["hash_post"] or frame in d1["hash_pre"])
    hash_mismatch = [{"frame": frame, "t04": t04_hash[frame],
                      "d1_pre": d1["hash_pre"].get(frame),
                      "d1_post": d1["hash_post"].get(frame)}
                     for frame in shared
                     if d1["hash_post"].get(frame, d1["hash_pre"].get(frame))
                     != t04_hash[frame]]
    return {
        "t04": {"compared": t04_compared,
                "expected": min(len(masks), len(t04_rows)),
                "corpus_decisions": len(t04_rows),
                "tape_decisions": len(masks),
                "uncovered_tail_decisions": max(0, len(masks) - len(t04_rows)),
                "mismatches": len(t04_mismatch), "first": t04_mismatch[:3]},
        "d1": {"compared": d1_compared,
               "expected": min(len(masks), d1_covered),
               "corpus_decisions": d1_covered,
               "tape_decisions": len(masks),
               "uncovered_tail_decisions": max(0, len(masks) - d1_covered),
               "mismatches": len(d1_mismatch), "first": d1_mismatch[:3]},
        "hash_identity": {"compared": len(shared), "mismatches": len(hash_mismatch),
                          "first": hash_mismatch[:3],
                          "note": ("T04 hash_pre at frame 16+2k vs D1's recorded "
                                   "hash at the same frame (pre or post)")},
        "note": ("`expected` is what the corpus can cover, never the tape's full "
                 "length: the tape carries a declared "
                 f"{int(tape['tail_decisions'])}-decision tail the recordings do "
                 "not contain"),
        "d1_sha256": d1["sha256"],
        "d1_registered": d1["registered_sha256"],
        "classification": "offline",
    }


# -- the shared budget ------------------------------------------------------
def stage_budget(out_root: str | Path, *, need_starts: int, need_frames: int,
                 label: str, stage_cap: Mapping[str, Any] | None = None,
                 stage_prefixes: Sequence[str] | None = None
                 ) -> dict[str, Any]:
    """Check one package against the stage allowance **and** the grand total.

    The ledger is one file for the whole ticket and is read, never rewritten: its
    settled rows are the ``02-A`` actuals (2 starts / 17,333 frames / 150.6 s /
    0.932 GiB).  Two different ceilings are checked before any engine is spawned:

    * this stage's own allowance for 02-B/C — ``3`` starts and ``20,000`` frames
      (the CLI may restate them; they are never larger than the defaults),
      counted over the rows whose ``attempt_id`` starts with ``02b-``/``02c-``;
    * the ticket's whole-ticket budget — ``28`` starts, ``360,000`` frames, ``8 h``
      from the booking time and ``18 GiB`` of new bytes, counted over **every**
      ledger row.

    The :class:`BudgetLimits` handed back makes the ledger's own ``reserve``
    refuse an over-booking as well (``committed_so_far + stage_remaining`` capped
    by the grand total), so both ceilings are enforced twice.

    ``stage_prefixes`` is which attempt ids count against *this stage*: the 02-B/C
    packages use the default ``02b-``/``02c-``, the 03 pilot passes ``03-`` so the
    two stages never charge each other.
    """

    prefixes = (tuple(STAGE_ATTEMPT_PREFIXES) if stage_prefixes is None
                else tuple(str(value) for value in stage_prefixes))

    cap = {"starts": float(STAGE_CAP_STARTS),
           "engine_frames": float(STAGE_CAP_FRAMES)}
    if stage_cap:
        # the CLI may restate this stage's allowance in either direction; both the
        # default and the effective value are reported so a raised cap is visible
        for name in cap:
            if stage_cap.get(name) is not None:
                cap[name] = float(stage_cap[name])
    ledger_path = Path(out_root) / "ledger.jsonl"
    rows: list[dict[str, Any]] = []
    if ledger_path.is_file():
        rows = [dict(row) for row in iter_jsonl(ledger_path)]
    # the same arithmetic as `Ledger.totals`: actual = every settle row, committed
    # = actual + the reservations that were never settled ("疑似已启动先计已耗")
    open_by_id: dict[str, dict[str, Any]] = {}
    actual = _zero_totals()
    stage_actual = _zero_totals()
    for row in rows:
        phase = row.get("phase")
        attempt = str(row.get("attempt_id", ""))
        if phase == "reserve":
            open_by_id[attempt] = row
        elif phase == "settle":
            open_by_id.pop(attempt, None)
            payload = row.get("actual") or {}
            _add_totals(actual, payload)
            if attempt.startswith(prefixes):
                _add_totals(stage_actual, payload)
    reserved = _zero_totals()
    stage_reserved = _zero_totals()
    for attempt, row in open_by_id.items():
        payload = row.get("requested") or {}
        _add_totals(reserved, payload)
        if attempt.startswith(prefixes):
            _add_totals(stage_reserved, payload)
    cumulative = dict(actual)
    _add_totals(cumulative, reserved)
    stage_used = dict(stage_actual)
    _add_totals(stage_used, stage_reserved)
    stage_remaining = {
        "starts": cap["starts"] - stage_used["starts"],
        "engine_frames": cap["engine_frames"] - stage_used["engine_frames"],
        "wall_s": float(GRAND_TOTAL_WALL_S) - cumulative["wall_s"],
        "write_gib": float(GRAND_TOTAL_WRITE_GIB) - cumulative["write_gib"],
    }
    grand_remaining = {
        "starts": float(GRAND_TOTAL_STARTS) - cumulative["starts"],
        "engine_frames": float(GRAND_TOTAL_FRAMES) - cumulative["engine_frames"],
        "wall_s": float(GRAND_TOTAL_WALL_S) - cumulative["wall_s"],
        "write_gib": float(GRAND_TOTAL_WRITE_GIB) - cumulative["write_gib"],
    }
    need = {"starts": float(need_starts), "engine_frames": float(need_frames)}
    exceeded: list[dict[str, Any]] = []
    for name in ("starts", "engine_frames"):
        if need[name] > stage_remaining[name]:
            exceeded.append({"ceiling": "stage_02bc", "dimension": name,
                             "needed": need[name],
                             "remaining": stage_remaining[name],
                             "cap": cap[name],
                             "used_by_this_stage": stage_used[name]})
        if need[name] > grand_remaining[name]:
            exceeded.append({"ceiling": "grand_total", "dimension": name,
                             "needed": need[name],
                             "remaining": grand_remaining[name],
                             "cap": (GRAND_TOTAL_STARTS if name == "starts"
                                     else GRAND_TOTAL_FRAMES),
                             "used_by_the_ticket": cumulative[name]})
    payload = {
        "label": label,
        "ledger_path": ledger_path.as_posix(),
        "ledger_rows_read": len(rows),
        "cumulative": cumulative,
        "stage_used": stage_used,
        "stage_remaining": stage_remaining,
        "grand_remaining": grand_remaining,
        "need": need,
        "stage_cap": {"starts": cap["starts"],
                      "engine_frames": cap["engine_frames"],
                      "attempt_prefixes": list(prefixes)},
        "stage_cap_default": {"starts": STAGE_CAP_STARTS,
                              "engine_frames": STAGE_CAP_FRAMES,
                              "note": ("the planning allowance this stage started "
                                       "with; `stage_cap` above is the effective "
                                       "value (the CLI may restate it)")},
        "grand_total": {"starts": GRAND_TOTAL_STARTS,
                        "engine_frames": GRAND_TOTAL_FRAMES,
                        "wall_s": GRAND_TOTAL_WALL_S,
                        "write_gib": GRAND_TOTAL_WRITE_GIB},
        "exceeded": exceeded,
        "classification": "measured",
    }
    if exceeded:
        raise BudgetRefused(
            f"{label}: {len(exceeded)} budget check(s) failed: "
            + "; ".join(f"{item['ceiling']}.{item['dimension']} "
                        f"{item['needed']} > {item['remaining']}"
                        for item in exceeded), payload)
    limits = BudgetLimits(
        max_starts=min(int(cumulative["starts"] + stage_remaining["starts"]),
                       GRAND_TOTAL_STARTS),
        max_engine_frames=min(
            int(cumulative["engine_frames"] + stage_remaining["engine_frames"]),
            GRAND_TOTAL_FRAMES),
        max_wall_s=GRAND_TOTAL_WALL_S,
        max_write_gib=GRAND_TOTAL_WRITE_GIB,
    )
    payload["ledger_limits"] = limits.as_dict()
    payload["ledger_limits_basis"] = (
        "min(committed_so_far + stage_remaining, grand total) per dimension, so "
        "the ledger refuses an over-booking before the engine is spawned")
    return {"report": payload, "limits": limits}


# ---------------------------------------------------------------------------
# 02-B / 02-C: one private, muted engine session with its own ledger row
# ---------------------------------------------------------------------------
class EngineAttempt:
    """One engine start: preflight, private copy, mute, reserve, RESET, settle.

    The same five checks ``run_engine_package`` performs for 02-A are re-used here
    (``engine_preflight``, ``prepare_private_game_dir``, ``write_mute_json``,
    ``_mute_gate``, ``mute_report_problems``, ``tree_fingerprint``); nothing about
    that path is re-implemented or changed.  What differs is the walk on top of
    it, which is the caller's.
    """

    def __init__(self, *, args: argparse.Namespace, out_root: Path, base_dir: Path,
                 attempt_id: str, planned_frames: int, note: str,
                 limits: BudgetLimits, meter: Any,
                 reset_seed: int = RESET_SEED) -> None:
        self.args = args
        self.out_root = Path(out_root)
        self.base_dir = Path(base_dir)
        self.attempt_id = str(attempt_id)
        self.planned_frames = int(planned_frames)
        self.note = str(note)
        self.limits = limits
        self.meter = meter
        #: 02-A/02-B/02-C all reset seed 12; each 03 development root is its own
        #: seed, so the seed is a parameter whose default is the frozen value.
        self.reset_seed = int(reset_seed)
        self.module = slice_tool()
        self.started_at = now_iso()
        self.started = time.perf_counter()
        self.preflight: dict[str, Any] = {}
        self.private: dict[str, Any] = {}
        self.mute: dict[str, Any] = {}
        self.mute_after: dict[str, Any] = {}
        self.post_problems: list[str] = []
        self.ledger: Ledger | None = None
        self.reservation: dict[str, Any] | None = None
        self.settlement: dict[str, Any] | None = None
        self.host: Any = None
        self.session: dict[str, Any] = {}
        self.reset: dict[str, Any] = {}
        self.reset_hash = ""
        self.source_before: dict[str, Any] = {}
        self.source_after: dict[str, Any] = {}
        self.source_cfg: Path | None = None
        self.copy_cfg: Path | None = None
        self.source_raw = b""
        self.copy_before_raw = b""
        self.checked_at: list[str] = []
        self.engine_frames = 0
        self.engine_started = False

    # -- before the start -------------------------------------------------
    def open(self) -> dict[str, Any]:
        """Every pre-start check, then one RESET; raises before any spawn."""

        module = self.module
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.preflight = engine_preflight(self.args, self.out_root)
        # `engine_preflight` reports the frozen stage-0/normal/team-0 template with
        # the default seed 12; the attempt records the seed it actually resets.
        reset = dict(self.preflight.get("reset") or {})
        reset["seed"] = int(self.reset_seed)
        reset["command"] = (f"RESET {reset.get('stage')} "
                            f"{reset.get('difficulty_index')} "
                            f"{reset.get('team')} {int(self.reset_seed)}")
        self.preflight["reset"] = reset
        source_dir = Path(self.preflight["data_source"])
        with self.meter.measure("preflight"):
            self.source_before = tree_fingerprint(source_dir)
        self.private = prepare_private_game_dir(out_dir=self.base_dir,
                                               data_source=source_dir,
                                               meter=self.meter)
        self.source_cfg = source_dir / "th08.cfg"
        self.copy_cfg = Path(self.private["copy_dir"]) / "th08.cfg"
        self.source_raw = self.source_cfg.read_bytes()
        self.copy_before_raw = self.copy_cfg.read_bytes()
        self.checked_at = [now_iso()]
        _, self.mute = module.write_mute_json(
            out_dir=self.base_dir,
            paths={"source": self.source_cfg, "copy": self.copy_cfg},
            source_raw=self.source_raw, copy_before_raw=self.copy_before_raw,
            copy_after_raw=None, checked_at=self.checked_at, problems=[],
            meter=self.meter)
        problems = _mute_gate(self.mute)
        if problems or not self.private["mute_byte_check"]["all_zero"]:
            raise EngineRefused(
                "refusing to start the engine with a non-muted private copy: "
                + "; ".join(problems or ["mute bytes are not all zero"]))
        self.ledger = Ledger(self.out_root / "ledger.jsonl", self.limits)
        self.reservation = self.ledger.reserve(
            self.attempt_id, starts=1, engine_frames=self.planned_frames,
            wall_s=900.0, write_gib=1.0,
            inputs={"engine": self.preflight["engine"],
                    "engine_sha256": self.preflight["engine_sha256"],
                    "data_source": self.preflight["data_source"],
                    "data_source_cfg_sha256": (self.preflight["data_source_cfg"]
                                               or {}).get("sha256"),
                    "planned_frames": self.planned_frames,
                    "reset": self.preflight["reset"]},
            note=self.note)
        shm_name = module.make_shm_name(f"followup-{self.attempt_id}")
        self.host = module.EngineHost(
            engine=self.preflight["engine"],
            data_dir=Path(self.private["copy_dir"]),
            shm_name=shm_name,
            timeout=max(60.0, float(getattr(self.args, "timeout_s", 120.0) or 120.0)),
            env=dict(module.FORCED_ENGINE_ENV),
            meter=self.meter,
        )
        self.session = self.host.open(
            stage=RESET_STAGE,
            difficulty_index=int(self.preflight["reset"]["difficulty_index"]),
            shot_type=RESET_TEAM, seed=self.reset_seed)
        self.engine_started = True
        reset_reply = dict(self.session["reset"])
        self.reset_hash = str(reset_reply.get("hash", "")).strip().lower()
        self.reset = {
            "reply": reset_reply,
            "hash": self.reset_hash,
            "frame": int(reset_reply.get("frames", -1)),
            "setup_frames": int(reset_reply.get("setup_frames", -1)),
            "seed": int(self.reset_seed),
        }
        return {"preflight": self.preflight, "private": self.private,
                "mute": self.mute, "reset": self.reset,
                "session": self.session, "reservation": self.reservation}

    # -- after the walk ---------------------------------------------------
    def close(self, *, result: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Close the process, re-check the mute bytes and settle the ledger."""

        module = self.module
        try:
            if self.host is not None:
                with self.meter.measure("session_close"):
                    self.host.close()
        finally:
            self.host = None
        copy_after_raw = b""
        have_copy = (self.copy_cfg is not None and self.copy_cfg.is_file()
                     and self.source_cfg is not None
                     and self.source_cfg.is_file())
        if have_copy:
            copy_after_raw = self.copy_cfg.read_bytes()
        self.checked_at.append(now_iso())
        if have_copy:
            _, self.mute_after = module.write_mute_json(
                out_dir=self.base_dir,
                paths={"source": self.source_cfg, "copy": self.copy_cfg},
                source_raw=self.source_raw, copy_before_raw=self.copy_before_raw,
                copy_after_raw=copy_after_raw, checked_at=self.checked_at,
                problems=[], meter=self.meter)
            self.post_problems = module.mute_report_problems(self.mute_after)
        else:
            self.mute_after = {}
            self.post_problems = [
                "no private copy existed, so the post-exit mute bytes cannot be "
                "re-checked"]
        if self.preflight.get("data_source") and self.source_before:
            with self.meter.measure("preflight"):
                self.source_after = tree_fingerprint(
                    Path(self.preflight["data_source"]))
        unchanged = bool(
            self.source_after
            and self.source_after["digest"] == self.source_before["digest"]
            and self.source_after["file_sha256"] == self.source_before["file_sha256"])
        wall_s = round(time.perf_counter() - self.started, 3)
        write_gib = round(du_bytes(self.base_dir) / 1024 ** 3, 3)
        if self.ledger is not None and self.attempt_id in self.ledger.reservations():
            self.settlement = self.ledger.settle(
                self.attempt_id, starts=1, engine_frames=self.engine_frames,
                wall_s=wall_s, write_gib=max(0.0, write_gib),
                result=dict(result or {}),
                note=f"{self.attempt_id}: settled after the engine process exited")
        return {
            "mute": {
                "path": (self.base_dir / "mute.json").as_posix(),
                "before_start_fields": (self.mute.get("copy_before_start")
                                        or {}).get("fields"),
                "before_start_problems": _mute_gate(self.mute),
                "mute_byte_check": self.private.get("mute_byte_check"),
                "engine_env": self.mute.get("engine_env"),
                "after_exit_fields": (self.mute_after.get("copy_after_exit")
                                      or {}).get("fields"),
                "after_exit_problems": self.post_problems,
                "after_exit_sha256": (self.mute_after.get("copy_after_exit")
                                      or {}).get("sha256"),
                "copied_entries": self.private.get("copied_entries"),
            },
            "source_dir": {"before": self.source_before, "after": self.source_after,
                           "unchanged": bool(unchanged)},
            "measured": {"starts": 1.0, "engine_frames": float(self.engine_frames),
                         "wall_s": wall_s, "write_gib": max(0.0, write_gib)},
            "ledger": {
                "attempt_id": self.attempt_id,
                "reserve_seq": (self.reservation or {}).get("seq"),
                "settle_seq": (self.settlement or {}).get("seq"),
                "actual": (self.settlement or {}).get("actual"),
                "totals": (self.ledger.totals() if self.ledger is not None else None),
                "limits": self.limits.as_dict(),
            },
        }

    def record_block(self) -> dict[str, Any]:
        """The identity half of a package record (preflight/private/mute/reset)."""

        return {
            "preflight": self.preflight,
            "private_game_dir": self.private,
            "reset": self.reset,
            "engine_session": self.session,
            "engine_env": self.module.engine_env_record(),
        }


# -- one decision's reads ---------------------------------------------------
def read_decision(host: Any) -> dict[str, Any]:
    """``HAZARDS`` + ``STATE`` + ``HUD`` + ``PLAYER`` after a decision's two frames.

    All four are read-only engine commands: none of them advances a frame, which
    is why the hash chain stays intact while they are interleaved (02-A.3 already
    interleaved ``STATE``/``HUD``/``PLAYER`` and kept 2,166/2,166 anchors).
    """

    with host.meter.measure("audit"):
        hazards = parse_hazards_enemies(host.client.request("HAZARDS"))
    state = host.state()
    hud = host.hud()
    with host.meter.measure("audit"):
        player = parse_player_reply(host.client.request("PLAYER"))
    return {"hazards": hazards, "state": state, "hud": hud, "player": player}


def engine_state_record(state: Mapping[str, Any]) -> dict[str, Any]:
    """The eight ``STATE`` fields the 02-B/C rows carry."""

    return {
        "frame": int(state["frame"]),
        "stage": int(state["stage"]),
        "lives_remaining": float(state["lives_remaining"]),
        "bombs_remaining": float(state["bombs_remaining"]),
        "player_power": float(state["player_power"]),
        "score": int(state["score"]),
        "graze": int(state["graze"]),
        "deaths": float(state["deaths"]),
    }


def hud_record(hud: Mapping[str, Any]) -> dict[str, Any]:
    """The ten ``HUD`` fields the 02-B/C rows carry."""

    return {name: int(hud[name]) for name in
            ("frame", "display_score", "globals_score", "lives", "bombs",
             "power", "graze", "time", "game_over")}


def player_record(player: Mapping[str, Any]) -> dict[str, Any]:
    """The seven ``PLAYER`` fields the 02-B/C rows carry."""

    return {name: player[name] for name in
            ("px", "py", "hp", "focus", "player_state", "player_state_name",
             "invuln_state")}


def anchor_at_boundary(*, decision_index: int, frame: int,
                       hashes: Mapping[int, str], t04_hash: Mapping[int, str],
                       d1_pre: Mapping[int, str],
                       d1_post: Mapping[int, str]) -> dict[str, Any]:
    """The anchor of one decision boundary, exactly as the ticket asks for it.

    The primary anchor is T04's ``hash_pre`` for slot ``i-8`` (its
    ``decision_frame`` is ``16+2k``, i.e. ``2i`` for ``i == k+8``).  When the
    boundary frame is also a multiple of four, the two D1 anchors at that frame
    are checked beside it.  A frame with no recorded anchor reports
    ``kind: "none"`` rather than a silent pass.
    """

    index = int(decision_index)
    boundary = int(frame)
    actual = hashes.get(boundary)
    actual = None if actual is None else str(actual).strip().lower()
    t04_slot = index - 8
    checks: list[dict[str, Any]] = []
    if t04_slot >= 0 and boundary in t04_hash:
        checks.append({"kind": "t04_hash_pre", "slot": t04_slot,
                       "frame": boundary, "expected": str(t04_hash[boundary]),
                       "actual": actual,
                       "ok": actual is not None and actual == str(t04_hash[boundary])})
    if boundary >= 4 and boundary % 4 == 0:
        for kind, table in (("d1_hash_pre", d1_pre), ("d1_hash_post", d1_post)):
            if boundary in table:
                want = str(table[boundary]).strip().lower()
                checks.append({"kind": kind, "frame": boundary, "expected": want,
                               "actual": actual,
                               "ok": actual is not None and actual == want})
    primary = checks[0] if checks else {"kind": "none", "frame": boundary,
                                        "expected": None, "actual": actual,
                                        "ok": True}
    return {
        "decision_index": index,
        "frame": boundary,
        "kind": primary["kind"],
        "expected": primary["expected"],
        "actual": actual,
        "ok": bool(primary["ok"]),
        "checks": checks,
    }


def tape_digest(tape: Mapping[str, Any], first: int = 0,
                last: int | None = None) -> str:
    """sha256 of ``"<decision>:<cand>:<mask1>:<mask2>\\n"`` over an interval.

    This is the digest a frozen card records for its prefix and a verifier
    recomputes from the T04 shard: it covers the candidate *and* both STEP masks,
    so a change of pulse phase or of the executor cannot hide behind an equal
    candidate list.
    """

    end = len(tape["cands"]) if last is None else int(last)
    payload = "".join(
        f"{index}:{int(tape['cands'][index])}:{int(tape['masks'][index][0])}:"
        f"{int(tape['masks'][index][1])}\n"
        for index in range(int(first), end))
    return sha256_bytes(payload.encode())


# ---------------------------------------------------------------------------
# 02-B `recon`: one engine start, the whole teacher tape, frame by frame
# ---------------------------------------------------------------------------
def recon_target_frames(tape: Mapping[str, Any]) -> int:
    """The last STEP frame the recon walks: the ticket's cap on the 2-frame grid."""

    return min(int(RECON_GRID_LAST_FRAME), int(tape["frames"]))


def cmd_recon(args: argparse.Namespace) -> int:
    """02-B recon: replay the seed-12 teacher tape and record every decision.

    One engine start, ``RESET 0 1 0 12``, then one ``STEP mask 1`` per frame so
    that **every** frame's ``EVENTS`` word is read (a 2-frame ``STEP`` would only
    publish the last frame's word and lose the other one).  After each decision's
    two frames the four read-only commands are taken: ``HAZARDS`` (all enemy
    slots), ``STATE``, ``HUD`` and ``PLAYER``.

    Stop conditions, in the ticket's priority order: a technical conflict, then
    ``STATE.stage`` leaving 0, then ``game_over``/a death, then the 13,195-frame
    cap.  The 2-frame decision grid stops at frame 13,194 and the single remaining
    frame is reported as ``grid_remainder_frames`` — never rounded away.

    No student policy is imported, read or consulted: the masks come from the
    recorded tape alone.
    """

    module = slice_tool()
    maybe_reexec_under_xvfb(enabled=not bool(getattr(args, "no_xvfb", False)))
    started_at = now_iso()
    started = time.perf_counter()
    out_root = Path(args.out).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve()
    base = out_root / "02b" / "recon"
    base.mkdir(parents=True, exist_ok=True)
    rows_path = base / "decisions.jsonl"
    result_path = base / "result.json"
    gate = Gate("02b_recon")
    meter = module.CostMeter()
    meter.begin()
    record: dict[str, Any] = {
        "schema": SCHEMA_RECON,
        "generated_by": "tools/rl/v8_31_followup.py recon",
        "created_at": started_at,
        "classification": "measured",
        "ticket": TICKET_ID,
        "repo": repo.as_posix(),
        "out": out_root.as_posix(),
        "base": base.as_posix(),
        "argv": list(getattr(args, "argv", []) or []),
        "host": host_identity(),
        "heads": git_identity(repo),
        "tool": {"path": TOOL_PATH.as_posix(), "sha256": sha256_file(TOOL_PATH)},
        "slice_helper": slice_identity(),
        "frame_cap": {"ticket_frames": RECON_FRAME_CAP,
                      "decision_grid_last_frame": RECON_GRID_LAST_FRAME,
                      "grid_remainder_frames": RECON_GRID_REMAINDER,
                      "decision_frames": DECISION_FRAMES},
        "read_method": {
            "events": "EVENTS after every single-frame STEP (frameEventFlags is a "
                      "per-frame shared u16, cleared at the start of each frame)",
            "enemies": "HAZARDS with no limit: every ACTIVE slot < 481 in one reply",
            "resources": "STATE + HUD + PLAYER after each decision's two frames",
            "hash": "the 16-hex state hash every STEP reply carries",
            "life": "signed: int(x,16) then v-2**32 if v>=2**31 (protocol.py reads "
                    "the %08x field unsigned)",
            "not_used": "ENEMIES paging (its offset/limit index the ACTIVE slots by "
                        "ordinal, so two frames cannot be aligned)",
        },
        "no_policy": True,
        "engine_started": False,
        "counts": {"decisions_planned": 0, "decisions": 0, "frames": 0,
                   "step_units": 0, "started": 0, "technical_invalid": 0},
        "classification_vocabulary": {
            "measured": "read from this run's engine process",
            "offline": "recomputed from a registered corpus without an engine",
            "historical": "quoted from a registered artefact",
            "unverified": "asserted but not re-derivable here",
        },
    }
    exit_code = EXIT_TECHNICAL
    attempt: EngineAttempt | None = None
    decisions = 0
    frames_done = 0
    step_units = 0
    hashes: dict[int, str] = {}
    terminal: dict[str, Any] = {"kind": "unknown"}
    budget_report: dict[str, Any] = {}
    closed: dict[str, Any] = {}
    enemies_seen: dict[int, dict[str, Any]] = {}
    death_frames: list[int] = []
    last_state: dict[str, Any] = {}
    target_frames = 0
    tape: dict[str, Any] = {}
    cross: dict[str, Any] = {}
    try:
        with meter.measure("io"):
            tape = tape_schedule(repo)
            cross = tape_cross_check(repo, tape)
        target_frames = recon_target_frames(tape)
        record["tape"] = {key: value for key, value in tape.items()
                          if key not in ("cands", "masks", "decision_frames")}
        record["tape"]["digest"] = tape_digest(tape)
        record["tape"]["digest_definition"] = (
            'sha256 of "<decision>:<cand>:<mask1>:<mask2>\\n" for every decision')
        record["tape_cross_check"] = cross
        record["frame_cap"]["walk_frames"] = int(target_frames)
        record["counts"]["decisions_planned"] = min(
            len(tape["cands"]), target_frames // DECISION_FRAMES)
        gate.add("tape_schedule_rebuilt",
                 int(tape["decisions"]) == 6660 and int(tape["slots"]) == 6588
                 and bool(tape["grid_is_two_frames_per_decision"]),
                 {"decisions": tape["decisions"], "slots": tape["slots"],
                  "leading_decisions": tape["leading_decisions"],
                  "tail_decisions": tape["tail_decisions"],
                  "offgrid": tape["offgrid_decisions"][:3]})
        gate.add("tape_masks_reproduce_t04_and_d1",
                 cross["t04"]["mismatches"] == 0 and cross["d1"]["mismatches"] == 0
                 and cross["t04"]["compared"] == cross["t04"]["expected"]
                 and cross["d1"]["compared"] == cross["d1"]["expected"],
                 {"t04": cross["t04"], "d1": cross["d1"]})
        gate.add("tape_hash_identity_between_corpora",
                 cross["hash_identity"]["mismatches"] == 0
                 and cross["hash_identity"]["compared"] > 0,
                 cross["hash_identity"])
        gate.add("tape_is_bomb_free",
                 not tape["decisions_carrying_bomb"]
                 and not tape["decisions_without_agent_bits"],
                 {"carrying_bomb": tape["decisions_carrying_bomb"][:3],
                  "without_agent_bits": tape["decisions_without_agent_bits"][:3]})

        with meter.measure("preflight"):
            budget = stage_budget(out_root, need_starts=1,
                                  need_frames=int(target_frames),
                                  label="02b-recon",
                                  stage_cap={
                                      "starts": getattr(args, "max_starts", None),
                                      "engine_frames": getattr(
                                          args, "max_engine_frames", None)})
        budget_report = budget["report"]
        record["budget"] = budget_report
        gate.add("budget_checked_from_the_existing_ledger",
                 not budget_report["exceeded"], {
                     "ledger_rows_read": budget_report["ledger_rows_read"],
                     "cumulative": budget_report["cumulative"],
                     "stage_used": budget_report["stage_used"],
                     "stage_remaining": budget_report["stage_remaining"],
                     "need": budget_report["need"],
                     "grand_remaining": budget_report["grand_remaining"],
                 })

        attempt = EngineAttempt(
            args=args, out_root=out_root, base_dir=base,
            attempt_id="02b-recon", planned_frames=int(target_frames),
            note=(f"recon: one RESET, teacher tape to frame {target_frames} "
                  f"of {tape['frames']} available ({tape['decisions']} decisions)"),
            limits=budget["limits"], meter=meter)
        opened = attempt.open()
        record.update(attempt.record_block())
        record["engine_started"] = True
        record["counts"]["started"] = 1
        gate.add("mute_before_start", not opened["mute"] or
                 not _mute_gate(opened["mute"]), _mute_gate(opened["mute"]))
        gate.add("mute_bytes_all_zero",
                 bool(attempt.private["mute_byte_check"]["all_zero"]),
                 attempt.private["mute_byte_check"])
        d1 = d1_replay_plan(repo)
        t04_hash: dict[int, str] = {}
        for row in iter_jsonl(repo / SOURCE_BY_ID["t04-2f-seed12"]["rel"]):
            frame = int(row["decision_frame"])
            value = str(row.get("hash_pre") or "").strip().lower()
            if value:
                t04_hash[frame] = value
        record["anchors_expected"] = {
            "t04_hash_pre": {"count": len(t04_hash),
                             "first_frame": min(t04_hash) if t04_hash else None,
                             "last_frame": max(t04_hash) if t04_hash else None},
            "d1_hash_pre": {"count": len(d1["hash_pre"]),
                            "first_frame": min(d1["hash_pre"]) if d1["hash_pre"] else None,
                            "last_frame": max(d1["hash_pre"]) if d1["hash_pre"] else None},
            "d1_hash_post": {"count": len(d1["hash_post"]),
                             "first_frame": min(d1["hash_post"]) if d1["hash_post"] else None,
                             "last_frame": max(d1["hash_post"]) if d1["hash_post"] else None},
        }
        gate.add("reset_hash_matches_d1_pre0",
                 attempt.reset_hash == d1["hash_pre"].get(0),
                 {"actual": attempt.reset_hash,
                  "expected": d1["hash_pre"].get(0),
                  "note": "the RESET state is D1's frame-0 hash_pos/pre, checked "
                          "separately and never counted as a post"})
        anchor_counts = {"t04_hash_pre": {"compared": 0, "mismatches": 0,
                                          "first": None},
                         "d1_hash_pre": {"compared": 0, "mismatches": 0, "first": None},
                         "d1_hash_post": {"compared": 0, "mismatches": 0, "first": None}}
        with rows_path.open("w", encoding="utf-8") as handle:
            for index in range(len(tape["cands"])):
                frame_start = index * DECISION_FRAMES + 1
                frame_end = frame_start + DECISION_FRAMES - 1
                if frame_end > target_frames:
                    terminal = {"kind": "frame_cap", "reached_frame": frames_done,
                                "cap_frames": RECON_FRAME_CAP,
                                "grid_last_frame": RECON_GRID_LAST_FRAME,
                                "grid_remainder_frames": RECON_GRID_REMAINDER,
                                "decision_index": index,
                                "why": ("the 2-frame decision grid cannot spend the "
                                        "last odd frame without breaking a decision "
                                        "in half")}
                    break
                masks = tape["masks"][index]
                ev_flags: list[int] = []
                reply: dict[str, Any] = {}
                shortfall: dict[str, Any] | None = None
                for offset, mask in enumerate(masks):
                    reply = attempt.host.step(int(mask), 1)
                    step_units += 1
                    advanced = int(reply.get("advanced", 0) or 0)
                    frames_done += advanced
                    if advanced != 1 or "exit" in reply:
                        shortfall = {"frame": frame_start + offset,
                                     "requested_frames": 1, "advanced": advanced,
                                     "exit": bool("exit" in reply),
                                     "reply": dict(reply)}
                        break
                    frame = frame_start + offset
                    value = reply.get("hash")
                    if value is not None:
                        hashes[frame] = str(value).strip().lower()
                    events = _parse_kv(attempt.host.events_raw())
                    if events.get("_verb") != "OK":
                        raise EngineRefused(
                            f"EVENTS reply is not OK at frame {frame}: "
                            f"{str(events)[:200]}")
                    if "frameEventFlags" not in events:
                        raise EngineRefused(
                            f"EVENTS reply has no frameEventFlags at frame {frame}")
                    if int(events.get("frames", -1)) != frame:
                        raise FrameRelationConflict(
                            f"EVENTS frames={events.get('frames')} but the STEP "
                            f"ended on frame {frame}")
                    ev_flags.append(int(events["frameEventFlags"], 16))
                if shortfall is not None:
                    terminal = {"kind": ("engine_exit" if shortfall["exit"]
                                         else "frame_shortfall"),
                                "reached_frame": frames_done, **shortfall}
                    break
                reads = read_decision(attempt.host)
                anchors = anchor_at_boundary(
                    decision_index=index, frame=frame_start - 1,
                    hashes=hashes, t04_hash=t04_hash,
                    d1_pre=d1["hash_pre"], d1_post=d1["hash_post"])
                for check in anchors["checks"]:
                    bucket = anchor_counts[check["kind"]]
                    bucket["compared"] += 1
                    if not check["ok"]:
                        bucket["mismatches"] += 1
                        if bucket["first"] is None:
                            bucket["first"] = check
                state = reads["state"]
                hazards = reads["hazards"]
                for enemy in hazards["enemies"]:
                    slot = int(enemy["slot"])
                    entry = enemies_seen.setdefault(slot, {
                        "slot": slot, "first_decision": index,
                        "first_frame": frame_end, "decisions": 0,
                        "boss_decisions": 0, "last_decision": index,
                        "last_life": enemy["life"], "min_life": enemy["life"],
                        "max_life": enemy["life"]})
                    entry["decisions"] += 1
                    entry["last_decision"] = index
                    entry["last_life"] = enemy["life"]
                    entry["min_life"] = min(entry["min_life"], enemy["life"])
                    entry["max_life"] = max(entry["max_life"], enemy["life"])
                    if enemy["boss"]:
                        entry["boss_decisions"] += 1
                        entry["boss"] = True
                        entry["boss_first_decision"] = min(
                            entry.get("boss_first_decision", index), index)
                        entry["boss_first_frame"] = min(
                            entry.get("boss_first_frame", frame_end), frame_end)
                last_state = state
                row = {
                    "schema": SCHEMA_RECON_ROW,
                    "decision_index": index,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "boundary_frame": frame_start - 1,
                    "cand": int(tape["cands"][index]),
                    "masks": [int(masks[0]), int(masks[1])],
                    "release_due": bool(adapter()["release_due"](
                        index, DEFAULT_PULSE_PERIOD)),
                    "ev_flags": [f"0x{ev_flags[0]:04x}", f"0x{ev_flags[1]:04x}"],
                    "ev_flags_dec": [int(ev_flags[0]), int(ev_flags[1])],
                    "ev_names": [event_names(ev_flags[0]), event_names(ev_flags[1])],
                    "state": engine_state_record(state),
                    "hud": hud_record(reads["hud"]),
                    "player": player_record(reads["player"]),
                    "enemies": hazards["enemies"],
                    "counts": {"bullets": hazards["bullets"],
                               "lasers": hazards["lasers"],
                               "enemies": hazards["enemies_count"]},
                    "hash": {"frame_start": hashes.get(frame_start),
                             "frame_end": hashes.get(frame_end)},
                    "anchor": {"kind": anchors["kind"],
                               "frame": anchors["frame"],
                               "expected": anchors["expected"],
                               "actual": anchors["actual"], "ok": anchors["ok"],
                               "checks": anchors["checks"]},
                    "classification": "measured",
                }
                write_jsonl_row(handle, row)
                decisions += 1
                game_over = int(reads["hud"].get("game_over", 0) or 0) \
                    or int(reply.get("game_over", 0) or 0)
                deaths = float(state["deaths"])
                if int(state["stage"]) != RESET_STAGE:
                    terminal = {"kind": "stage_advance", "reached_frame": frame_end,
                                "decision_index": index,
                                "stage": int(state["stage"]),
                                "stage_from": RESET_STAGE,
                                "ev_flags": row["ev_flags"],
                                "stage_clear_level_bit": bool(
                                    (ev_flags[0] | ev_flags[1])
                                    & EVENT_STAGE_CLEAR_LEVEL)}
                    break
                if game_over:
                    terminal = {"kind": "game_over", "reached_frame": frame_end,
                                "decision_index": index,
                                "hud_game_over": int(reads["hud"]["game_over"]),
                                "step_game_over": int(reply.get("game_over", 0) or 0)}
                    break
                if deaths > 0.0:
                    death_frames.append(int(state["frame"]))
                    terminal = {"kind": "legal_death", "reached_frame": frame_end,
                                "decision_index": index, "deaths": deaths,
                                "player_state": row["player"]["player_state"],
                                "note": ("a death stops the recon: the ticket's "
                                         "stop order puts the stage advance first "
                                         "and a death second")}
                    break
                if decisions % 500 == 0:
                    print(f"  recon {decisions} decisions / {frames_done} frames "
                          f"stage={int(state['stage'])} enemies={hazards['enemies_count']} "
                          f"boss={sum(1 for e in hazards['enemies'] if e['boss'])}",
                          flush=True)
            else:
                terminal = {"kind": "tape_exhausted", "reached_frame": frames_done,
                            "decision_index": decisions}
        attempt.engine_frames = frames_done
        closed = attempt.close(result={
            "command": "02b-recon", "terminal": terminal.get("kind"),
            "reached_frame": frames_done})
        record["mute"] = closed["mute"]
        record["source_dir"] = closed["source_dir"]
        record["ledger"] = closed["ledger"]
        record["measured"] = closed["measured"]
        gate.add("mute_after_exit", not closed["mute"]["after_exit_problems"],
                 closed["mute"]["after_exit_problems"])
        gate.add("source_dir_unchanged", closed["source_dir"]["unchanged"],
                 {"digest_before": closed["source_dir"]["before"]["digest"],
                  "digest_after": closed["source_dir"]["after"]["digest"]})
        gate.add("ledger_settled", bool(record["ledger"].get("settle_seq")),
                 {"attempt_id": record["ledger"].get("attempt_id"),
                  "actual": record["ledger"].get("actual")})

        t04_report = anchor_report(
            expected=t04_hash, actual=hashes, first_frame=1,
            last_frame=int(target_frames),
            label="T04 hash_pre anchors (frame 16+2k: the boundary of decision k+8)")
        pre_report = anchor_report(
            expected=d1["hash_pre"], actual=hashes, first_frame=1,
            last_frame=int(target_frames),
            label="D1 hash_pre anchors (every multiple of 4)")
        post_report = anchor_report(
            expected=d1["hash_post"], actual=hashes, first_frame=1,
            last_frame=int(target_frames),
            label="D1 hash_post anchors (every multiple of 4)")
        record["anchors"] = {
            "t04_pre": t04_report, "d1_pre": pre_report, "d1_post": post_report,
            "streamed": anchor_counts,
            "denominators": {
                "t04_pre": {"compared": t04_report["compared"],
                            "expected": t04_report["expected"]},
                "d1_pre": {"compared": pre_report["compared"],
                           "expected": pre_report["expected"]},
                "d1_post": {"compared": post_report["compared"],
                            "expected": post_report["expected"]},
            },
        }
        gate.add("t04_pre_anchors", t04_report["mismatches"] == 0
                 and t04_report["compared"] > 0, {
                     "compared": t04_report["compared"],
                     "expected": t04_report["expected"],
                     "mismatches": t04_report["mismatches"],
                     "first_mismatch_frame": t04_report["first_mismatch_frame"]})
        gate.add("d1_pre_anchors", pre_report["mismatches"] == 0
                 and pre_report["compared"] > 0, {
                     "compared": pre_report["compared"],
                     "expected": pre_report["expected"],
                     "mismatches": pre_report["mismatches"],
                     "first_mismatch_frame": pre_report["first_mismatch_frame"]})
        gate.add("d1_post_anchors", post_report["mismatches"] == 0
                 and post_report["compared"] > 0, {
                     "compared": post_report["compared"],
                     "expected": post_report["expected"],
                     "mismatches": post_report["mismatches"],
                     "first_mismatch_frame": post_report["first_mismatch_frame"]})
        gate.add("every_decision_has_two_frames_and_two_events",
                 decisions > 0 and frames_done == decisions * DECISION_FRAMES,
                 {"decisions": decisions, "frames": frames_done,
                  "step_units": step_units})
        record["counts"].update({"decisions": decisions, "frames": frames_done,
                                 "step_units": step_units})
        record["terminal"] = dict(terminal, classification="measured")
        record["last_state"] = engine_state_record(last_state) if last_state else None
        record["death_frames"] = death_frames
        boss_slots = sorted(slot for slot, entry in enemies_seen.items()
                            if entry.get("boss"))
        record["enemies"] = {
            "distinct_slots": len(enemies_seen),
            "boss_slots": boss_slots,
            "slots": [enemies_seen[slot] for slot in sorted(enemies_seen)],
            "note": ("presence only: the generation segmentation and the removal "
                     "classification are `nominate`'s job (it re-reads every row "
                     "of decisions.jsonl)"),
            "classification": "measured",
        }
        record["counts"]["effective"] = 1
        exit_code = EXIT_OK if gate.passed else EXIT_GATE_FAILED
    except BudgetRefused as exc:
        record["budget"] = dict(exc.payload, label="02b-recon")
        record["terminal"] = {"kind": "budget_refused", "reason": exc.reason,
                              "detail": exc.payload.get("exceeded"),
                              "engine_started": bool(record.get("engine_started"))}
        exit_code = EXIT_NOT_IMPLEMENTED
    except (EngineRefused, FrameRelationConflict, SourceError) as exc:
        record["counts"]["technical_invalid"] = 1
        record["terminal"] = {"kind": "technical_invalid",
                              "reason": type(exc).__name__, "detail": str(exc),
                              "engine_started": bool(record.get("engine_started"))}
        exit_code = EXIT_TECHNICAL
    except Exception as exc:  # noqa: BLE001 - evidence before any traceback
        record["counts"]["technical_invalid"] = 1
        record["terminal"] = {"kind": "technical_invalid",
                              "reason": type(exc).__name__, "detail": str(exc),
                              "traceback": traceback.format_exc(),
                              "engine_started": bool(record.get("engine_started"))}
        exit_code = EXIT_TECHNICAL
    finally:
        meter.finish()
        if attempt is not None and not closed and (
                attempt.engine_started
                or (attempt.reservation is not None and attempt.settlement is None)):
            # a failure inside the walk still has to close the process, re-check
            # the mute bytes and settle the ledger row
            try:
                attempt.engine_frames = frames_done
                closed = attempt.close(result={"command": "02b-recon",
                                               "terminal": "aborted",
                                               "reached_frame": frames_done})
                record["mute"] = closed["mute"]
                record["source_dir"] = closed["source_dir"]
                record["ledger"] = closed["ledger"]
                record["measured"] = closed["measured"]
            except Exception as exc:  # noqa: BLE001
                record["close_error"] = f"{type(exc).__name__}: {exc}"

    record.setdefault("terminal", {"kind": "unknown"})
    record["finished_at"] = now_iso()
    record["counts"].setdefault("decisions", decisions)
    record["counts"].setdefault("frames", frames_done)
    record["cost"] = meter.payload(step_units=step_units)
    record["gate"] = gate.as_dict()
    record.setdefault("measured", {"starts": float(bool(record.get("engine_started"))),
                                   "engine_frames": float(frames_done),
                                   "wall_s": round(time.perf_counter() - started, 3),
                                   "write_gib": 0.0})
    record["artifacts"] = {"decisions": rows_path.as_posix(),
                           "decisions_rows": decisions,
                           "decisions_bytes": (rows_path.stat().st_size
                                               if rows_path.is_file() else 0),
                           "result": result_path.as_posix()}
    write_json(result_path, record)
    for check in gate.checks:
        print(f"  {'ok  ' if check.ok else 'FAIL'} {check.name}")
    print(f"recon: terminal={record['terminal'].get('kind')} "
          f"frames={frames_done} decisions={decisions} "
          f"t04_pre={record.get('anchors', {}).get('t04_pre', {}).get('compared')}"
          f"/{record.get('anchors', {}).get('t04_pre', {}).get('expected')} "
          f"d1_pre={record.get('anchors', {}).get('d1_pre', {}).get('compared')}"
          f"/{record.get('anchors', {}).get('d1_pre', {}).get('expected')} "
          f"d1_post={record.get('anchors', {}).get('d1_post', {}).get('compared')}"
          f"/{record.get('anchors', {}).get('d1_post', {}).get('expected')} "
          f"rc={exit_code}")
    print(f"wrote: {base.as_posix()} rows={decisions}")
    if exit_code == EXIT_OK:
        print(f"gate: passed ({len(gate.checks)} checks)")
    else:
        print(f"gate: {'FAILED' if exit_code == EXIT_GATE_FAILED else 'technical'} "
              f"({len(gate.problems)} failed checks)")
        for problem in gate.problems[:12]:
            print(f"  - {problem}")
    return exit_code


# ---------------------------------------------------------------------------
# 02-B `nominate`: per-slot generations, the ending rule table, one frozen card
# ---------------------------------------------------------------------------
def _flag_word(row: Mapping[str, Any]) -> int:
    """The decision's two ``EVENTS`` words OR-ed together.

    ``frameEventFlags`` is cleared at the start of every frame, so a decision has
    exactly two words.  Sampling at the decision point is what hides one frame;
    OR-ing the two **recorded** frames never hides one of *them*.
    """

    values = row.get("ev_flags_dec") or [0, 0]
    return int(values[0]) | int(values[1])


def _frame_with_bit(row: Mapping[str, Any], bit: int) -> int | None:
    """The first frame of this decision whose ``EVENTS`` word carries ``bit``."""

    values = row.get("ev_flags_dec") or [0, 0]
    if int(values[0]) & int(bit):
        return int(row["frame_start"])
    if int(values[1]) & int(bit):
        return int(row["frame_end"])
    return None


def _out_of_playfield(enemy: Mapping[str, Any]) -> bool:
    """``worldPosition`` outside ``[0,384]x[0,448]`` (``GameManager.cpp:137-152``)."""

    x = enemy.get("world_x")
    y = enemy.get("world_y")
    if x is None or y is None:
        return False
    return not (0.0 <= float(x) <= 384.0 and 0.0 <= float(y) <= 448.0)


def generation_segments(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Segment every enemy slot into generations and describe every ending.

    A **generation** is the longest run of consecutive decisions in which a slot
    is observed, broken by exactly the three conditions the ticket names:

    1. the same decision carries ``ENEMY_SPAWNED`` *and* ``ENEMY_REMOVED``;
    2. the ``life`` jumps back to the generation's spawn value while
       ``ENEMY_SPAWNED`` is set (a ``DEATH_MODE 3`` phase end also writes
       ``life = 1``, which is why the spawn bit is required here);
    3. the slot disappears and reappears (the break is at the reappearance).

    Detection samples the row at the end of every decision's two frames, so a
    removal is reported with the frame the ``EVENTS`` word names, not with the
    sample it was noticed at.
    """

    index_of = {int(row["decision_index"]): row for row in rows}
    last_decision = max(index_of) if index_of else -1
    by_slot: dict[int, dict[int, dict[str, Any]]] = {}
    for row in rows:
        index = int(row["decision_index"])
        for enemy in row.get("enemies") or []:
            by_slot.setdefault(int(enemy["slot"]), {})[index] = enemy
    generations: list[dict[str, Any]] = []
    for slot in sorted(by_slot):
        seen = by_slot[slot]
        current: dict[str, Any] | None = None
        for index in sorted(seen):
            row = index_of[index]
            flags = _flag_word(row)
            enemy = seen[index]
            life = int(enemy["life"])
            spawn_bit = bool(flags & EVENT_ENEMY_SPAWNED)
            removed_bit = bool(flags & EVENT_ENEMY_REMOVED)
            fresh = False
            if current is None:
                fresh = True
            elif index != int(current["last_decision"]) + 1:
                fresh = True
            elif spawn_bit and removed_bit:
                fresh = True
            elif spawn_bit and life == int(current["life_at_spawn"]) \
                    and life != int(current["life"][-1]):
                fresh = True
            if fresh:
                current = {"slot": slot, "generation": len(generations),
                           "spawn_decision": index, "decisions": [], "life": [],
                           "rows": [], "last_decision": index,
                           "life_at_spawn": life,
                           "flags_at_spawn": int(enemy["flags1_dec"]),
                           "boss": False}
                generations.append(current)
            current["decisions"].append(index)
            current["life"].append(life)
            current["rows"].append(enemy)
            current["last_decision"] = index
            current["flags_last"] = int(enemy["flags1_dec"])
            current["boss"] = bool(current["boss"] or enemy["boss"])
            spawn_frame = _frame_with_bit(row, EVENT_ENEMY_SPAWNED)
            if "spawn_frame" not in current:
                current["spawn_frame"] = int(spawn_frame or row["frame_end"])
            if enemy["boss"] and "boss_from_decision" not in current:
                current["boss_from_decision"] = index
                current["boss_from_frame"] = int(row["frame_end"])
    out: list[dict[str, Any]] = []
    for generation in generations:
        decisions = generation["decisions"]
        last = int(generation["last_decision"])
        last_row = index_of[last]
        last_enemy = generation["rows"][-1]
        alive_at_end = last == last_decision
        next_index = None if alive_at_end else last + 1
        next_row = index_of.get(next_index) if next_index is not None else None
        next_has_slot = bool(
            next_row is not None
            and any(int(enemy["slot"]) == int(generation["slot"])
                    for enemy in next_row.get("enemies") or []))
        removal_frame = None
        removal_word = None
        if next_row is not None and not next_has_slot:
            removal_word = next_row.get("ev_flags")
            removal_frame = _frame_with_bit(next_row, EVENT_ENEMY_REMOVED)
        phase_ends: list[dict[str, Any]] = []
        for position in range(1, len(decisions)):
            before = generation["rows"][position - 1]
            after = generation["rows"][position]
            word = _flag_word(index_of[decisions[position]])
            if (after["boss"] and before["damageable"]
                    and not after["damageable"] and int(after["life"]) == 1
                    and word & EVENT_ENEMY_REMOVED):
                phase_ends.append({
                    "decision": decisions[position],
                    "life_before": int(before["life"]),
                    "life_after": int(after["life"]),
                    "ev_flags": index_of[decisions[position]]["ev_flags"],
                    "kind": "boss_phase_end_M3"})
        life_zero_inside = [{"decision": decisions[position],
                             "ev_flags": index_of[decisions[position]]["ev_flags"]}
                            for position, value in enumerate(generation["life"])
                            if int(value) == 0]
        # A corpse mode (DEATH_MODE 1/2) never makes the row disappear, so its
        # completion is the frame the first ``life == 0`` signature was observed
        # — that is where ``ENEMY_REMOVED`` names the death dispatcher.
        corpse_decision = None
        corpse_frame = None
        if life_zero_inside:
            first_zero = life_zero_inside[0]
            zero_row = index_of[first_zero["decision"]]
            corpse_decision = int(first_zero["decision"])
            corpse_frame = _frame_with_bit(zero_row, EVENT_ENEMY_REMOVED) \
                or int(zero_row["frame_end"])
        previous_row = index_of.get(decisions[-2]) if len(decisions) > 1 else None
        previous_life = (int(generation["life"][-2]) if len(generation["life"]) > 1
                         else None)
        outcome, why = classify_generation_ending(
            last_enemy=last_enemy, previous_life=previous_life,
            alive_at_end=alive_at_end, replaced=next_has_slot,
            removal_frame=removal_frame,
            removal_word=removal_word, last_word=_flag_word(last_row))
        completion_frame = (int(removal_frame) if removal_frame is not None
                            else corpse_frame)
        completion_decision = (next_index if removal_frame is not None
                               else corpse_decision)
        completion_ev_flags = (removal_word if removal_frame is not None
                               else (life_zero_inside[0]["ev_flags"]
                                     if life_zero_inside else None))
        score_at_last = int((last_row.get("hud") or {}).get("globals_score", 0))
        score_at_next = (int((next_row.get("hud") or {}).get("globals_score", 0))
                         if next_row is not None else None)
        frames_between = (2 if next_row is not None else 0)
        score_delta = (None if score_at_next is None else score_at_next - score_at_last)
        out.append({
            "slot": int(generation["slot"]),
            "generation_index": int(generation["generation"]),
            "boss": bool(generation["boss"]),
            "boss_from_decision": generation.get("boss_from_decision"),
            "boss_from_frame": generation.get("boss_from_frame"),
            "spawn_decision": int(generation["spawn_decision"]),
            "spawn_frame": int(generation["spawn_frame"]),
            "life_at_spawn": int(generation["life_at_spawn"]),
            "flags_at_spawn": f"0x{int(generation['flags_at_spawn']):08x}",
            "flags_at_spawn_names": flags1_names(int(generation["flags_at_spawn"])),
            "first_decision": int(decisions[0]),
            "last_decision": last,
            "last_frame": int(last_row["frame_end"]),
            "decisions_observed": len(decisions),
            "frames_observed": len(decisions) * DECISION_FRAMES,
            "life_first": int(generation["life"][0]),
            "life_last": int(generation["life"][-1]),
            "life_min": int(min(generation["life"])),
            "life_max": int(max(generation["life"])),
            "life_monotonic_to_zero": all(
                int(right) <= int(left) for left, right
                in zip(generation["life"], generation["life"][1:]))
                and int(generation["life"][-1]) == 0,
            "life_series_head": [int(value) for value in generation["life"][:6]],
            "life_series_tail": [int(value) for value in generation["life"][-6:]],
            "life_series": [int(value) for value in generation["life"]],
            "last_life": int(last_enemy["life"]),
            "last_flags1": f"0x{int(last_enemy['flags1_dec']):08x}",
            "last_damageable": bool(last_enemy["damageable"]),
            "last_persist": bool(last_enemy["persist_after_death"]),
            "alive_at_recon_end": bool(alive_at_end),
            "removal_decision": next_index,
            "removal_frame": removal_frame,
            "removal_ev_flags": removal_word,
            "completion_decision": completion_decision,
            "completion_frame": completion_frame,
            "completion_ev_flags": completion_ev_flags,
            "completion_kind": ("row_removed" if removal_frame is not None
                                else ("life_zero_signature" if corpse_frame is not None
                                      else None)),
            "removal_ev_names": ([event_names(_word_int(removal_word[0]))
                                  + event_names(_word_int(removal_word[1]))]
                                 if removal_word else None),
            "replaced_by_a_new_generation": bool(next_has_slot),
            "post_phase_end_removal": bool(
                phase_ends and not alive_at_end and not next_has_slot
                and phase_ends[-1]["decision"] >= last - 1),
            "removal_effect_bit": bool(
                removal_word is not None
                and (_frame_with_bit(next_row, EVENT_EFFECT_SPAWNED) is not None)),
            "phase_ends": phase_ends,
            "life_zero_inside": life_zero_inside,
            "out_of_playfield_at_last": _out_of_playfield(last_enemy),
            "score_delta_to_removal": score_delta,
            "score_delta_above_damage_bound": (
                None if score_delta is None else
                score_delta > ENEMY_MAX_DAMAGE_PER_FRAME * 2 * max(frames_between, 1)),
            "outcome": outcome,
            "outcome_why": why,
            "terminal": outcome not in ("boss_phase_end_M3", "still_active_at_cap",
                                        "unknown"),
            "previous_life": previous_life,
            "previous_decision": (decisions[-2] if len(decisions) > 1 else None),
            "classification": "offline",
        })
    return {
        "generations": out,
        "slots": {str(slot): {"decisions": len(seen),
                              "first_decision": min(seen),
                              "last_decision": max(seen)}
                  for slot, seen in sorted(by_slot.items())},
        "last_decision": last_decision,
        "counts": {
            "slots": len(by_slot),
            "generations": len(out),
            "boss_generations": sum(1 for item in out if item["boss"]),
            "generations_by_outcome": _count_by(out, "outcome"),
            "generations_by_slot": {str(slot): sum(1 for item in out
                                                   if item["slot"] == slot)
                                    for slot in sorted(by_slot)},
        },
        "classification": "offline",
    }


def _count_by(items: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    """Count the distinct values of one key (never a guess about the set)."""

    out: dict[str, int] = {}
    for item in items:
        value = str(item.get(key))
        out[value] = out.get(value, 0) + 1
    return out


def classify_generation_ending(*, last_enemy: Mapping[str, Any],
                               previous_life: int | None, alive_at_end: bool,
                               replaced: bool, removal_frame: int | None,
                               removal_word: Any,
                               last_word: int) -> tuple[str, str]:
    """Apply ``enemy_semantics.md`` §4's rule table to one generation's ending.

    The table, in the order it is applied:

    ``kill_M1``   the row is still there with ``life == 0`` **and**
                  ``PERSIST_AFTER_DEATH`` (``DEATH_MODE 1`` leaves the corpse);
    ``kill_M2``   the row is still there with ``life == 0``, still ``DAMAGEABLE``
                  and the same decision carries ``ENEMY_REMOVED``
                  (``DEATH_MODE 2``; only this mode can go negative later);
    ``boss_phase_end_M3``  ``BOSS`` row, ``DAMAGEABLE`` 1 -> 0, ``life == 1``,
                  ``ENEMY_REMOVED`` present: a **phase** ended, the enemy did not
                  leave (``EnemyManagerUpdate.cpp:493-517``);
    ``clear_wipe`` the row is there with ``life == 0``, no ``PERSIST`` and no
                  ``ENEMY_REMOVED`` in that decision: a script / life-callback /
                  timeout wrote ``life = 0`` without the death dispatcher
                  (``EnemyManager.cpp:490/657/1495``);
    ``kill_M0``   the row is gone, the removal frame carries ``ENEMY_REMOVED`` and
                  the last observed ``life <= 70``: one frame of damage could have
                  crossed it;
    ``scripted_retreat``      gone, ``ENEMY_REMOVED``, ``life > 70``: ``Despawn``
                  never touches ``life`` and one frame cannot take 70+;
    ``offscreen_despawn``     the same, with the last ``worldPosition`` outside the
                  playfield and ``HAS_BEEN_IN_BOUNDS`` set / ``ALLOW_OFFSCREEN``
                  clear;
    ``not_removed_no_event``  gone without ``ENEMY_REMOVED``: not a removal at all
                  (chain teardown / a recording gap);
    ``still_active_at_cap``   still there at the last recorded decision.
    """

    life = int(last_enemy["life"])
    if life == 0 and last_enemy["persist_after_death"]:
        return ("kill_M1", "life==0 and PERSIST_AFTER_DEATH: DEATH_MODE 1 corpse")
    if life == 0 and last_enemy["damageable"] and last_word & EVENT_ENEMY_REMOVED:
        return ("kill_M2",
                "life==0, still DAMAGEABLE and ENEMY_REMOVED in the same decision: "
                "DEATH_MODE 2")
    if (last_enemy["boss"] and not last_enemy["damageable"] and life == 1
            and last_word & EVENT_ENEMY_REMOVED):
        return ("boss_phase_end_M3",
                "BOSS row with DAMAGEABLE cleared and life forced to 1 plus "
                "ENEMY_REMOVED: a phase ended, the row stays (not a kill)")
    if life == 0 and not last_enemy["persist_after_death"] \
            and not last_word & EVENT_ENEMY_REMOVED:
        return ("clear_wipe",
                "life==0 without ENEMY_REMOVED in that decision: a life write "
                "(KillAll / life callback / timeout), not the death dispatcher")
    if alive_at_end:
        return ("still_active_at_cap",
                f"still ACTIVE at the last recorded decision (life={life})")
    if replaced:
        return ("slot_reused",
                "the slot still holds an enemy but the generation broke on this "
                "decision (ENEMY_SPAWNED + ENEMY_REMOVED, or a life reset): the "
                "slot was reused, it was not removed")
    if removal_frame is None:
        return ("not_removed_no_event",
                "the row vanished but no ENEMY_REMOVED word names that frame: "
                "chain teardown or a recording gap, not a removal")
    if life <= ENEMY_MAX_DAMAGE_PER_FRAME:
        return ("kill_M0",
                f"row gone, ENEMY_REMOVED at frame {removal_frame} and the last "
                f"life {life} <= {ENEMY_MAX_DAMAGE_PER_FRAME}: one frame of damage "
                "can cross it (DEATH_MODE 0)")
    if (_out_of_playfield(last_enemy) and last_enemy["has_been_in_bounds"]
            and not last_enemy["allow_offscreen"]):
        return ("offscreen_despawn",
                f"row gone with life {life} > {ENEMY_MAX_DAMAGE_PER_FRAME} and the "
                "last world position outside the playfield "
                "(HAS_BEEN_IN_BOUNDS set, ALLOW_OFFSCREEN clear)")
    return ("scripted_retreat",
            f"row gone, ENEMY_REMOVED at frame {removal_frame} and the last life "
            f"{life} > {ENEMY_MAX_DAMAGE_PER_FRAME}: Despawn does not touch life, "
            "so the script took it away")


#: The outcomes a frozen card may complete with (the ticket's ``kill_*`` set).
KILL_OUTCOMES: tuple[str, ...] = ("kill_M0", "kill_M1", "kill_M2")
#: Every outcome an arm can report, in the ticket's vocabulary.
OUTCOME_VOCABULARY: tuple[str, ...] = (
    "kill_M0", "kill_M1", "kill_M2", "boss_phase_end_M3", "scripted_retreat",
    "offscreen_despawn", "clear_wipe", "not_removed_no_event", "slot_reused",
    "still_active_at_cap", "not_removed_within_cap", "generation_not_observed",
    "legal_death", "game_over", "frame_cap", "technical_conflict",
)


def candidate_windows(generations: Sequence[Mapping[str, Any]], *,
                      frame_cap: int = CARD_FRAME_CAP,
                      decision_cap: int = CARD_DECISION_CAP,
                      lead_decisions: int = 2,
                      stage_frames_remaining: int | None = None
                      ) -> list[dict[str, Any]]:
    """Every BOSS generation scored against the freeze conditions.

    The **entry** is the frozen teacher prefix.  Two rules are applied, in order:

    1. ``spawn_minus_lead``: two controllable decisions before the generation
       appears (the ticket's "spawn decision or its previous 1-5 decisions");
    2. ``latest_entry_within_the_frame_cap``: when that entry puts the consequence
       further than ``frame_cap`` frames away, the entry moves **later** — to the
       earliest decision that satisfies ``completion_frame - 2*entry <= frame_cap``.
       The ticket's hard rules are "the entry precedes the consequence", "at least
       one controllable decision", "span <= 600 frames" and "a frozen prefix no
       policy touches"; all four still hold, because the prefix is the recorded
       tape.  A candidate whose consequence is too close to the entry, or whose two
       02-C arms cannot be paid for inside ``stage_frames_remaining``, stays
       ineligible and says which condition it missed.
    """

    windows: list[dict[str, Any]] = []
    for generation in generations:
        if not generation["boss"]:
            continue
        completion_frame = generation.get("completion_frame")
        completion_decision = generation.get("completion_decision")
        preferred_entry = max(1, int(generation["spawn_decision"])
                              - int(lead_decisions))
        entry_decision = preferred_entry
        entry_rule = "spawn_minus_lead"
        if completion_frame is not None:
            latest = -(-(int(completion_frame) - int(frame_cap)) // DECISION_FRAMES)
            if int(entry_decision) < latest:
                entry_decision = int(latest)
                entry_rule = "latest_entry_within_the_frame_cap"
        entry_decision = max(1, int(entry_decision))
        entry_frame = entry_decision * DECISION_FRAMES
        span = (None if completion_frame is None
                else int(completion_frame) - entry_frame)
        arm_frames = entry_frame + int(frame_cap)
        target_life_at_entry = None
        spawn = int(generation["spawn_decision"])
        life_series = generation.get("life_series") or generation.get("life") or []
        target_spawns_inside_window = bool(entry_decision < spawn)
        if (not target_spawns_inside_window
                and 0 <= entry_decision - spawn < len(life_series)):
            target_life_at_entry = int(life_series[entry_decision - spawn])
        missing: list[str] = []
        if generation["outcome"] not in KILL_OUTCOMES:
            missing.append(f"outcome is {generation['outcome']}, not kill_*")
        if completion_frame is None:
            missing.append("no completion frame (neither a removal carrying "
                           "ENEMY_REMOVED nor a life==0 signature)")
        if span is not None and span > int(frame_cap):
            missing.append(f"span {span} frames > the {frame_cap}-frame cap")
        if span is not None and span < DECISION_FRAMES:
            missing.append("no controllable decision between the entry and the "
                           "consequence")
        if completion_decision is not None and entry_decision >= int(
                completion_decision):
            missing.append("the entry is not before the consequence")
        if entry_decision < 1:
            missing.append("no decision before the entry")
        if target_spawns_inside_window:
            pass
        elif target_life_at_entry is None:
            missing.append("the frozen generation is not observed at the entry "
                           "decision, so the target is not controllable there")
        elif target_life_at_entry <= 0:
            missing.append(f"the frozen generation's life is {target_life_at_entry} "
                           "at the entry: the target is already dead there")
        if generation["decisions_observed"] <= 1:
            missing.append("the generation is observed at one decision only, so its "
                           "life trajectory cannot be described")
        if stage_frames_remaining is not None and (
                2 * arm_frames > int(stage_frames_remaining)):
            missing.append(
                f"the two 02-C arms need {2 * arm_frames} frames "
                f"(2 x ({entry_frame} prefix + {frame_cap} control)) but only "
                f"{int(stage_frames_remaining)} are left in this stage's allowance")
        windows.append({
            "candidate_id": f"slot{generation['slot']}-gen{generation['generation_index']}",
            "slot": int(generation["slot"]),
            "generation_index": int(generation["generation_index"]),
            "boss": True,
            "spawn_decision": spawn,
            "spawn_frame": int(generation["spawn_frame"]),
            "life_at_spawn": int(generation["life_at_spawn"]),
            "life_first": int(generation["life_first"]),
            "life_last": int(generation["life_last"]),
            "life_min": int(generation["life_min"]),
            "monotonic_to_zero": bool(generation["life_monotonic_to_zero"]),
            "decisions_observed": int(generation["decisions_observed"]),
            "removal_decision": generation["removal_decision"],
            "removal_frame": generation["removal_frame"],
            "removal_ev_flags": generation["removal_ev_flags"],
            "completion_decision": completion_decision,
            "completion_frame": completion_frame,
            "completion_kind": generation.get("completion_kind"),
            "completion_ev_flags": generation.get("completion_ev_flags"),
            "outcome": generation["outcome"],
            "outcome_why": generation["outcome_why"],
            "phase_ends": generation["phase_ends"],
            "preferred_entry_decision": preferred_entry,
            "entry_rule": entry_rule,
            "entry_decision": entry_decision,
            "entry_frame": entry_frame,
            "lead_decisions": int(lead_decisions),
            "span_frames": span,
            "target_life_at_entry": target_life_at_entry,
            "target_spawns_inside_window": target_spawns_inside_window,
            "arm_frames": arm_frames,
            "pair_frames": 2 * arm_frames,
            "teacher_span_decisions": (None if span is None else
                                       (span + DECISION_FRAMES - 1) // DECISION_FRAMES),
            "evidence": {
                "removal_effect_bit": generation["removal_effect_bit"],
                "score_delta_to_removal": generation["score_delta_to_removal"],
                "score_delta_above_damage_bound":
                    generation["score_delta_above_damage_bound"],
                "last_persist": generation["last_persist"],
                "last_damageable": generation["last_damageable"],
                "out_of_playfield_at_last": generation["out_of_playfield_at_last"],
                "life_zero_inside": generation["life_zero_inside"],
                "post_phase_end_removal": generation["post_phase_end_removal"],
                "removal_evidence_full": generation["removal_ev_names"],
                "last_observed_life": generation["life_last"],
                "last_observed_frame": generation["last_frame"],
                "frames_from_last_observation_to_removal":
                    (None if generation["removal_frame"] is None else
                     int(generation["removal_frame"])
                     - int(generation["last_frame"])),
                "max_damage_since_last_observation":
                    (None if generation["removal_frame"] is None else
                     ENEMY_MAX_DAMAGE_PER_FRAME
                     * max(1, int(generation["removal_frame"])
                           - int(generation["last_frame"]))),
            },
            "eligible": not missing,
            "missing": missing,
            "classification": "offline",
        })
    windows.sort(key=lambda item: (0 if item["eligible"] else 1,
                                   item["completion_frame"] or 10 ** 9,
                                   item["slot"]))
    return windows


def load_recon(recon_dir: str | Path) -> dict[str, Any]:
    """Read a recon directory's ``result.json`` and every ``decisions.jsonl`` row."""

    base = Path(recon_dir)
    result_path = base / "result.json"
    rows_path = base / "decisions.jsonl"
    if not result_path.is_file():
        raise SourceError(f"the recon result is missing: {result_path}")
    if not rows_path.is_file():
        raise SourceError(f"the recon decisions are missing: {rows_path}")
    result = _load_json(result_path)
    rows = [dict(row) for row in iter_jsonl(rows_path)]
    return {"base": base, "result": result, "rows": rows,
            "result_sha256": sha256_file(result_path),
            "rows_sha256": sha256_file(rows_path)}


def canonical_json(payload: Mapping[str, Any]) -> str:
    """The canonical form a content digest is computed over."""

    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def content_digest(payload: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of ``payload`` alone."""

    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def build_task_card(*, candidate: Mapping[str, Any], tape: Mapping[str, Any],
                    rows: Sequence[Mapping[str, Any]], recon: Mapping[str, Any],
                    repo: str | Path, budget: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze one attack window into the card 02-C must obey.

    Every field is derived from the recon rows and the T04 shard: the entry hash is
    the hash the recon actually saw at the entry frame (so both arms must reproduce
    it), the prefix digest covers the candidate *and* both masks of every prefix
    decision, and the teacher continuation digest covers the 300 decisions the tape
    arm may execute.
    """

    entry = int(candidate["entry_decision"])
    entry_frame = int(candidate["entry_frame"])
    if entry < 1:
        raise SourceError(f"entry decision {entry} is not after a controllable one")
    index_of = {int(row["decision_index"]): row for row in rows}
    if entry_frame == 0:
        raise SourceError("entry frame 0 has no preceding decision")
    boundary_row = index_of.get(entry - 1)
    if boundary_row is None:
        raise SourceError(f"the recon has no decision {entry - 1} for the entry")
    entry_hash = (boundary_row.get("hash") or {}).get("frame_end")
    if not entry_hash:
        raise SourceError(
            f"the recon recorded no hash at frame {entry_frame} (the entry)")
    recon_result = recon.get("result") or {}
    recon_heads = recon_result.get("heads") or {}
    entry_row = index_of.get(entry)
    resources = {
        "state": dict((boundary_row or {}).get("state") or {}),
        "hud": dict((boundary_row or {}).get("hud") or {}),
        "player": dict((boundary_row or {}).get("player") or {}),
        "counts": dict((boundary_row or {}).get("counts") or {}),
        "frame": int(entry_frame),
        "source": (f"02b/recon/decisions.jsonl decision {entry - 1} "
                   f"(the record taken at the entry frame)"),
        "classification": "measured",
    }
    control_first = entry
    control_last = entry + CARD_DECISION_CAP - 1
    card: dict[str, Any] = {
        "schema": SCHEMA_TASK_CARD,
        "task_id": f"v831-02bc-seed12-{candidate['candidate_id']}",
        "created_at": now_iso(),
        "ticket": TICKET_ID,
        "repo_head": recon_heads.get("head"),
        "repo_dirty_entries": recon_heads.get("dirty_count"),
        "tool_sha256": sha256_file(TOOL_PATH),
        "recon": {
            "dir": str(recon.get("base")),
            "result": Path(str(recon.get("base"))) .joinpath("result.json").as_posix(),
            "decisions": Path(str(recon.get("base"))).joinpath(
                "decisions.jsonl").as_posix(),
            "terminal": (recon.get("result") or {}).get("terminal", {}).get("kind"),
            "frames": (recon.get("result") or {}).get("counts", {}).get("frames"),
            "decisions": (recon.get("result") or {}).get("counts", {}).get("decisions"),
            "classification": "measured",
        },
        "event": {
            "kind": "boss_generation_kill",
            "classification": candidate["outcome"],
            "why": candidate["outcome_why"],
            "enemy_identity_rule": (
                "slot + generation: `HAZARDS` rows carry `slot` and `flags1`; a new "
                "generation starts when the slot is absent then present, when one "
                "decision carries ENEMY_SPAWNED and ENEMY_REMOVED, or when `life` "
                "jumps back to the spawn value with ENEMY_SPAWNED set. `ENEMIES` "
                "paging is never used (its ordinals shift between frames)."),
            "seed12_instance": {
                "slot": int(candidate["slot"]),
                "generation_index": int(candidate["generation_index"]),
                "spawn_decision": int(candidate["spawn_decision"]),
                "spawn_frame": int(candidate["spawn_frame"]),
                "life_at_spawn": int(candidate["life_at_spawn"]),
                "life_first": int(candidate["life_first"]),
                "life_last": int(candidate["life_last"]),
                "life_min": int(candidate["life_min"]),
                "monotonic_to_zero": bool(candidate["monotonic_to_zero"]),
                "decisions_observed": int(candidate["decisions_observed"]),
            },
            "completion_rule": (
                "the frozen generation's row disappears (or stays as a life==0 "
                "corpse) inside the 600-frame window AND the removal frame carries "
                "ENEMY_REMOVED (0x0020) AND the last observed life is <= 70, i.e. "
                "one frame of damage can cross it -> kill_M0; the DEATH_MODE 1/2 "
                "signatures (`life==0` + PERSIST_AFTER_DEATH, or `life==0` + still "
                "DAMAGEABLE + ENEMY_REMOVED) -> kill_M1/kill_M2. A row that vanishes "
                "with life > 70, a Boss row whose DAMAGEABLE flips while life is "
                "forced to 1, a life==0 row without ENEMY_REMOVED, and a row that is "
                "still there at the cap are the named alternatives below."),
            "completion": {
                "decision": candidate.get("completion_decision"),
                "frame": candidate.get("completion_frame"),
                "kind": candidate.get("completion_kind"),
                "span_from_entry": candidate.get("span_frames"),
                "observed_life": candidate.get("life_last"),
                "ev_flags": candidate.get("completion_ev_flags"),
                "signature": candidate.get("outcome"),
            },
            "removal": ({
                "decision": candidate["removal_decision"],
                "frame": candidate["removal_frame"],
                "ev_flags": candidate["removal_ev_flags"],
            } if candidate["removal_frame"] is not None else None),
            "removal_note": (
                "null when the death mode leaves the corpse in place "
                "(DEATH_MODE 1/2): the completion above is then the frame the "
                "life==0 signature first appears, and the row's later disappearance "
                "is not a removal (no ENEMY_REMOVED word names it)"),
            "evidence": candidate["evidence"],
            "phase_ends": candidate["phase_ends"],
            "alternative_outcomes": [
                {"outcome": "scripted_retreat",
                 "rule": "the row vanishes with life > 70 (Despawn never touches "
                         "life and one frame takes at most 70)"},
                {"outcome": "offscreen_despawn",
                 "rule": "the row vanishes with life > 70 while the last world "
                         "position is outside [0,384]x[0,448], HAS_BEEN_IN_BOUNDS "
                         "set and ALLOW_OFFSCREEN clear"},
                {"outcome": "boss_phase_end_M3",
                 "rule": "the BOSS row's DAMAGEABLE flips 1->0 while life is forced "
                         "to 1 and ENEMY_REMOVED is set: a phase ended, the row stays"},
                {"outcome": "clear_wipe",
                 "rule": "life==0 without ENEMY_REMOVED: KillAll / life callback / "
                         "timeout wrote the life, not the death dispatcher"},
                {"outcome": "not_removed_no_event",
                 "rule": "the row vanished without ENEMY_REMOVED: chain teardown or "
                         "a recording gap"},
                {"outcome": "slot_reused",
                 "rule": "the slot broke its generation on one decision "
                         "(ENEMY_SPAWNED + ENEMY_REMOVED, or life back at the spawn "
                         "value): the slot was reused, not removed"},
                {"outcome": "not_removed_within_cap",
                 "rule": "the frozen generation is still there when the card's frame "
                         "cap is reached"},
                {"outcome": "generation_not_observed",
                 "rule": "the slot never carries the frozen identity inside the arm"},
                {"outcome": "legal_death",
                 "rule": "the player loses a life before the consequence"},
                {"outcome": "game_over",
                 "rule": "the run ends with gameOverFlag set"},
                {"outcome": "frame_cap",
                 "rule": "600 frames / 300 decisions elapse without a consequence"},
            ],
        },
        "entry": {
            "decision_index": entry,
            "frame": int(entry_frame),
            "prefix_decisions": entry,
            "prefix_frames": int(entry_frame),
            "prefix_masks_sha256": tape_digest(tape, 0, entry),
            "prefix_masks_digest_definition": (
                'sha256 of "<decision>:<cand>:<mask1>:<mask2>\\n" for decisions '
                "0..entry-1, from T04's recorded schedule"),
            "pulse_period": DEFAULT_PULSE_PERIOD,
            "controller": "recorded_tape_masks",
            "reads_state": False,
            "hash": str(entry_hash),
            "hash_source": (f"02b/recon decision {entry - 1} hash.frame_end, i.e. "
                            f"the state hash the recon's STEP returned at frame "
                            f"{entry_frame}"),
            "rule": candidate.get("entry_rule"),
            "preferred_entry_decision": candidate.get("preferred_entry_decision"),
            "rule_note": (
                "`spawn_minus_lead` is the ticket's suggestion (the spawn decision "
                "or its previous 1-5 decisions).  `latest_entry_within_the_frame_cap` "
                "is used only when that entry would put the frozen completion "
                "further than the 600-frame cap away: the entry then moves later, "
                "and the hard rules (entry before the consequence, at least one "
                "controllable decision, span <= 600, a frozen tape prefix no policy "
                "touches) still hold."),
            "target_life_at_entry": candidate.get("target_life_at_entry"),
            "target_spawns_inside_window": candidate.get(
                "target_spawns_inside_window"),
            "classification": "measured",
        },
        "resources_at_entry": resources,
        "success": (
            f"the frozen generation (slot {candidate['slot']}, spawn decision "
            f"{candidate['spawn_decision']}) is removed with a kill_* outcome "
            f"(kill_M0/M1/M2) within {CARD_FRAME_CAP} frames of the entry"),
        "legal_failure": (
            "the frozen generation ends as any named alternative outcome, or the "
            "player loses a life / the run ends before the consequence; both are "
            "recorded, never re-interpreted"),
        "frame_cap": {"decisions": CARD_DECISION_CAP, "frames": CARD_FRAME_CAP},
        "termination_priority": [
            "technical_conflict (a hash/EVENTS/HAZARDS inconsistency: no number is "
            "used from that arm)",
            "game_over (HUD gameOverFlag or STEP game_over)",
            "legal_death (STATE.deaths increases)",
            "frozen_consequence (the frozen generation ends: kill_* or a named "
            "alternative)",
            "decision_cap (300 decisions)",
            "frame_cap (600 frames)",
        ],
        "teacher_continuation": {
            "rule": ("arm T executes the recorded T04 schedule's decisions "
                     "entry..entry+299 through the same two-frame executor and the "
                     "same SHOOT release phase (release_due(i, 48)); the tape's "
                     "declared 64-decision tail covers the end of the recording"),
            "source": "T04 executed_cand schedule",
            "first_decision": control_first,
            "last_decision": control_last,
            "masks_sha256": tape_digest(tape, control_first, control_last + 1),
        },
        "control": {"name": "constant_cand_12", "cand": CONTROL_CAND,
                    "release_rule": "release_due(i, 48) applied to cand 12",
                    "pulse_period": DEFAULT_PULSE_PERIOD},
        "read_method": {
            "per_frame": "EVENTS (frameEventFlags, a per-frame shared u16)",
            "per_decision": "HAZARDS (no limit, every enemy slot) + STATE + HUD + "
                            "PLAYER + the STEP reply's state hash",
            "identity": "HAZARDS slot + flags1 (bit1 BOSS, bit3 DAMAGEABLE, "
                        "bit6 ACCEPTS_DAMAGE, bit23 PERSIST_AFTER_DEATH, "
                        "bit24 HAS_BEEN_IN_BOUNDS, bit28 ALLOW_OFFSCREEN)",
            "life": "signed: int(x,16) then v-2**32 if v>=2**31",
            "never": "ENEMIES paging, score alone, a single ENEMY_REMOVED bit",
        },
        "not_a_policy_result": True,
        "not_a_policy_result_note": (
            "the entry controller is the frozen recorded tape, not a policy: no "
            "student or teacher model is consulted, the masks are the recorded "
            "candidate and the global pulse phase only"),
        "budget_at_freeze": dict(budget),
        "classification": "measured",
    }
    card["content_sha256"] = content_digest(card)
    return card


def task_card_markdown(card: Mapping[str, Any]) -> str:
    """The human-readable card: the digest sits under the title."""

    event = card["event"]
    instance = event["seed12_instance"]
    entry = card["entry"]
    removal = event["removal"]
    lines = [
        f"# Task card — {card['task_id']}",
        "",
        f"`content_sha256 = {card['content_sha256']}`",
        "",
        f"- ticket: {card['ticket']}",
        f"- created: {card['created_at']}",
        f"- repo HEAD: `{card['repo_head']}`",
        f"- tool sha256: `{card['tool_sha256']}`",
        f"- frozen from: `{card['recon']['dir']}` "
        f"({card['recon']['decisions']} decisions / {card['recon']['frames']} frames, "
        f"terminal `{card['recon']['terminal']}`)",
        "",
        "## The event",
        "",
        f"- kind: **{event['kind']}** ({event['classification']})",
        f"- identity rule: {event['enemy_identity_rule']}",
        f"- seed-12 instance: slot **{instance['slot']}**, generation "
        f"{instance['generation_index']}, spawn decision "
        f"**{instance['spawn_decision']}** (frame {instance['spawn_frame']}), "
        f"life at spawn **{instance['life_at_spawn']}**",
        f"- life: first {instance['life_first']}, last {instance['life_last']}, "
        f"min {instance['life_min']}, monotonic to zero "
        f"{instance['monotonic_to_zero']}, observed at "
        f"{instance['decisions_observed']} decisions",
        f"- completion: {event['completion_rule']}",
        (f"- observed completion: decision {event['completion']['decision']}, "
         f"frame {event['completion']['frame']} "
         f"({event['completion']['kind']}), EVENTS "
         f"{event['completion']['ev_flags']}, life "
         f"{event['completion']['observed_life']}"),
        (f"- observed row removal: decision {removal['decision']}, frame "
         f"{removal['frame']}, EVENTS {removal['ev_flags']}"
         if removal else "- observed row removal: none (the corpse stays)"),
        f"- why: {event['why']}",
        "",
        "## The entry",
        "",
        f"- decision **{entry['decision_index']}**, frame **{entry['frame']}**",
        f"- prefix: {entry['prefix_decisions']} decisions, digest "
        f"`{entry['prefix_masks_sha256']}`",
        f"- controller: `{entry['controller']}` (reads state: "
        f"{entry['reads_state']}), pulse period {entry['pulse_period']}",
        f"- entry hash: `{entry['hash']}`",
        f"- resources at the entry: lives "
        f"{card['resources_at_entry']['state'].get('lives_remaining')}, bombs "
        f"{card['resources_at_entry']['state'].get('bombs_remaining')}, power "
        f"{card['resources_at_entry']['state'].get('player_power')}, score "
        f"{card['resources_at_entry']['state'].get('score')}",
        "",
        "## The rules",
        "",
        f"- success: {card['success']}",
        f"- legal failure: {card['legal_failure']}",
        f"- frame cap: {card['frame_cap']['decisions']} decisions / "
        f"{card['frame_cap']['frames']} frames",
        f"- termination priority: {' > '.join(card['termination_priority'])}",
        f"- teacher continuation: {card['teacher_continuation']['rule']}",
        f"- control: `{card['control']['name']}` (cand {card['control']['cand']}, "
        f"{card['control']['release_rule']})",
        f"- not a policy result: {card['not_a_policy_result']}",
        "",
        "## Alternative outcomes",
        "",
    ]
    for alternative in event["alternative_outcomes"]:
        lines.append(f"- `{alternative['outcome']}`: {alternative['rule']}")
    lines.append("")
    return "\n".join(lines)


def write_task_card(out_dir: Path, card: Mapping[str, Any]) -> dict[str, Any]:
    """Write ``task_card.json`` and ``task_card.md``; return their identities."""

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "task_card.json"
    md_path = out_dir / "task_card.md"
    write_json(json_path, dict(card))
    md_path.write_text(task_card_markdown(card), encoding="utf-8")
    return {"json": json_path.as_posix(), "json_sha256": sha256_file(json_path),
            "md": md_path.as_posix(), "md_sha256": sha256_file(md_path),
            "content_sha256": card["content_sha256"]}


def read_task_card(out_dir: str | Path) -> dict[str, Any]:
    """Read and re-verify a frozen card before an arm is allowed to start."""

    path = Path(out_dir) / "task_card.json"
    if not path.is_file():
        raise EngineRefused(f"the frozen task card is missing: {path}")
    card = _load_json(path)
    recorded = str(card.get("content_sha256") or "")
    body = {key: value for key, value in card.items() if key != "content_sha256"}
    recomputed = content_digest(body)
    if not recorded or recorded != recomputed:
        raise EngineRefused(
            f"the task card's content_sha256 does not match its own content: "
            f"recorded {recorded!r}, recomputed {recomputed!r}")
    return {"card": card, "path": path, "sha256": sha256_file(path),
            "content_sha256": recorded,
            "content_sha256_recomputed": recomputed,
            "content_sha256_matches": True}


def cmd_nominate(args: argparse.Namespace) -> int:
    """02-B nominate: segment the recon, freeze one card or report the near misses.

    Offline only: no engine is started and nothing outside ``--out`` is written.
    The nomination reads ``decisions.jsonl`` row by row, rebuilds every slot's
    generations, applies ``enemy_semantics.md`` §4 to each ending, and freezes the
    **earliest** BOSS generation that (a) ends in ``kill_*``, (b) completes within
    600 frames of its entry, (c) has a real controllable lead, and (d) can be paid
    for by the two 02-C arms inside this stage's remaining frame allowance.
    """

    started = time.perf_counter()
    out_root = Path(args.out).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve()
    base = out_root / "02b"
    base.mkdir(parents=True, exist_ok=True)
    gate = Gate("02b_nominate")
    record: dict[str, Any] = {
        "schema": SCHEMA_NOMINATION,
        "generated_by": "tools/rl/v8_31_followup.py nominate",
        "created_at": now_iso(),
        "classification": "offline",
        "ticket": TICKET_ID,
        "repo": repo.as_posix(),
        "out": out_root.as_posix(),
        "argv": list(getattr(args, "argv", []) or []),
        "host": host_identity(),
        "heads": git_identity(repo),
        "tool": {"path": TOOL_PATH.as_posix(), "sha256": sha256_file(TOOL_PATH)},
        "recon": {"dir": Path(args.recon).expanduser().as_posix()},
    }
    exit_code = EXIT_TECHNICAL
    try:
        recon = load_recon(args.recon)
        record["recon"].update({
            "dir": recon["base"].as_posix(),
            "result_sha256": recon["result_sha256"],
            "decisions_sha256": recon["rows_sha256"],
            "terminal": (recon["result"].get("terminal") or {}).get("kind"),
            "frames": (recon["result"].get("counts") or {}).get("frames"),
            "decisions": (recon["result"].get("counts") or {}).get("decisions"),
            "gate_passed": (recon["result"].get("gate") or {}).get("passed"),
        })
        rows = recon["rows"]
        if not rows:
            raise SourceError("the recon has no decision rows")
        tape = tape_schedule(repo)
        record["tape"] = {key: value for key, value in tape.items()
                          if key not in ("cands", "masks", "decision_frames")}
        record["tape"]["prefix_digest_full"] = tape_digest(tape)
        # the recorded gate is *reported*, not trusted: this reading re-derives
        # every predicate it depends on from the raw rows and the two corpora
        gate.add("recon_gate_claimed_recorded",
                 (recon["result"].get("gate") or {}).get("passed") is not None,
                 {"gate": (recon["result"].get("gate") or {}).get("passed"),
                  "terminal": record["recon"]["terminal"],
                  "note": ("recorded only; the three checks below re-derive the "
                           "same predicates from decisions.jsonl itself")})
        gate.add("recon_rows_present", len(rows) > 0, {"rows": len(rows)})
        anchors = recon_anchor_recheck(rows, repo)
        record["recon"]["anchors"] = _recon_anchor_summary(anchors)
        gate.add("recon_anchors_recomputed",
                 all(anchors[name]["mismatches"] == 0
                     for name in ("t04_pre", "d1_pre", "d1_post"))
                 and anchors["t04_pre"]["compared"] > 0,
                 _recon_anchor_summary(anchors))
        cross = tape_cross_check(repo, tape)
        record["tape_cross_check"] = cross
        gate.add("recon_tape_cross_check_recomputed",
                 cross["t04"]["mismatches"] == 0 and cross["d1"]["mismatches"] == 0
                 and cross["t04"]["compared"] == cross["t04"]["expected"]
                 and cross["d1"]["compared"] == cross["d1"]["expected"],
                 {"t04": cross["t04"], "d1": cross["d1"]})
        gate.add("recon_tape_digest_recomputed",
                 (recon["result"].get("tape") or {}).get("digest")
                 == tape_digest(tape),
                 {"recorded": (recon["result"].get("tape") or {}).get("digest"),
                  "recomputed": tape_digest(tape)})
        budget = stage_budget(out_root, need_starts=0, need_frames=0,
                              label="02b-nominate",
                              stage_cap={
                                  "starts": getattr(args, "max_starts", None),
                                  "engine_frames": getattr(
                                      args, "max_engine_frames", None)})
        stage_remaining_frames = int(
            budget["report"]["stage_remaining"]["engine_frames"])
        record["budget"] = budget["report"]
        record["stage_frames_remaining_at_nomination"] = stage_remaining_frames
        segments = generation_segments(rows)
        windows = candidate_windows(
            segments["generations"],
            stage_frames_remaining=stage_remaining_frames)
        record["generations"] = {
            "counts": segments["counts"],
            "slots": segments["slots"],
            "boss_generations": [item for item in segments["generations"]
                                 if item["boss"]],
            "last_decision": segments["last_decision"],
            "classification": "offline",
        }
        record["candidates"] = windows
        eligible = [item for item in windows if item["eligible"]]
        record["counts"] = {
            "decisions": len(rows),
            "frames": len(rows) * DECISION_FRAMES,
            "slots": segments["counts"]["slots"],
            "generations": segments["counts"]["generations"],
            "boss_generations": segments["counts"]["boss_generations"],
            "candidates": len(windows),
            "eligible": len(eligible),
            "outcomes": segments["counts"]["generations_by_outcome"],
        }
        gate.add("generations_segmented",
                 segments["counts"]["generations"] > 0
                 and segments["counts"]["boss_generations"] > 0,
                 {"slots": segments["counts"]["slots"],
                  "generations": segments["counts"]["generations"],
                  "boss_generations": segments["counts"]["boss_generations"],
                  "outcomes": segments["counts"]["generations_by_outcome"]})
        if not eligible:
            record["verdict"] = "no_freezeable_candidate"
            record["near_miss"] = [
                {"candidate_id": item["candidate_id"], "slot": item["slot"],
                 "spawn_decision": item["spawn_decision"],
                 "removal_frame": item["removal_frame"],
                 "outcome": item["outcome"], "span_frames": item["span_frames"],
                 "missing": item["missing"]}
                for item in windows]
            gate.add("one_candidate_frozen", False,
                     {"reason": "no BOSS generation satisfies every condition",
                      "near_miss_rows": len(windows)})
            gate.add("entry_before_consequence", False,
                     {"reason": "no candidate"})
            record["task_card"] = None
            exit_code = EXIT_GATE_FAILED
        else:
            chosen = eligible[0]
            record["chosen"] = chosen
            record["near_miss"] = [
                {"candidate_id": item["candidate_id"], "slot": item["slot"],
                 "outcome": item["outcome"], "span_frames": item["span_frames"],
                 "missing": item["missing"]}
                for item in windows if not item["eligible"]]
            card = build_task_card(candidate=chosen, tape=tape, rows=rows,
                                   recon=recon, repo=repo,
                                   budget=budget["report"])
            written = write_task_card(base, card)
            record["task_card"] = {
                "json": written["json"], "json_sha256": written["json_sha256"],
                "md": written["md"], "md_sha256": written["md_sha256"],
                "content_sha256": written["content_sha256"],
                "task_id": card["task_id"],
            }
            gate.add("one_candidate_frozen", True,
                     {"task_id": card["task_id"],
                      "candidate_id": chosen["candidate_id"],
                      "content_sha256": written["content_sha256"],
                      "near_miss_rows": len(windows) - len(eligible)})
            gate.add("completion_rule_is_a_kill",
                     chosen["outcome"] in KILL_OUTCOMES,
                     {"outcome": chosen["outcome"], "why": chosen["outcome_why"]})
            gate.add("completion_frame_carries_enemy_removed",
                     chosen.get("completion_ev_flags") is not None
                     and any(int(value, 16) & EVENT_ENEMY_REMOVED
                             for value in chosen["completion_ev_flags"]),
                     {"completion_ev_flags": chosen.get("completion_ev_flags"),
                      "completion_kind": chosen.get("completion_kind"),
                      "completion_frame": chosen.get("completion_frame"),
                      "removal_frame": chosen.get("removal_frame")})
            gate.add("span_within_frame_cap",
                     chosen["span_frames"] is not None
                     and int(chosen["span_frames"]) <= CARD_FRAME_CAP,
                     {"span_frames": chosen["span_frames"],
                      "cap": CARD_FRAME_CAP})
            gate.add("entry_after_a_controllable_decision",
                     int(chosen["entry_decision"]) >= 1
                     and int(chosen["entry_frame"]) >= DECISION_FRAMES,
                     {"entry_decision": chosen["entry_decision"],
                      "entry_frame": chosen["entry_frame"],
                      "entry_rule": chosen.get("entry_rule")})
            gate.add("target_alive_and_damageable_at_entry",
                     bool(chosen.get("target_spawns_inside_window"))
                     or (chosen.get("target_life_at_entry") is not None
                         and int(chosen["target_life_at_entry"]) > 0),
                     {"target_life_at_entry": chosen.get("target_life_at_entry"),
                      "target_spawns_inside_window": chosen.get(
                          "target_spawns_inside_window"),
                      "entry_frame": chosen["entry_frame"]})
            gate.add("two_arms_fit_the_stage_allowance",
                     int(chosen["pair_frames"]) <= stage_remaining_frames,
                     {"pair_frames": chosen["pair_frames"],
                      "stage_remaining_frames": stage_remaining_frames})
            gate.add("card_content_sha256_recomputed",
                     content_digest({key: value for key, value in card.items()
                                     if key != "content_sha256"})
                     == card["content_sha256"],
                     {"content_sha256": card["content_sha256"]})
            gate.add("card_prefix_digest_matches_the_tape",
                     card["entry"]["prefix_masks_sha256"]
                     == tape_digest(tape, 0, int(chosen["entry_decision"])),
                     {"prefix_masks_sha256": card["entry"]["prefix_masks_sha256"]})
            gate.add("alternative_outcomes_named",
                     len(card["event"]["alternative_outcomes"]) >= 6,
                     [row["outcome"]
                      for row in card["event"]["alternative_outcomes"]])
            record["verdict"] = "frozen"
            exit_code = EXIT_OK if gate.passed else EXIT_GATE_FAILED
    except BudgetRefused as exc:
        record["budget"] = dict(exc.payload)
        record["verdict"] = "budget_refused"
        record["near_miss"] = []
        gate.add("budget_checked", False, exc.payload.get("exceeded"))
        exit_code = EXIT_NOT_IMPLEMENTED
    except (SourceError, EngineRefused) as exc:
        record["verdict"] = "technical_invalid"
        record["error"] = {"reason": type(exc).__name__, "detail": str(exc)}
        gate.add("inputs_readable", False, {"error": str(exc)})
        exit_code = EXIT_TECHNICAL
    except Exception as exc:  # noqa: BLE001
        record["verdict"] = "technical_invalid"
        record["error"] = {"reason": type(exc).__name__, "detail": str(exc),
                           "traceback": traceback.format_exc()}
        gate.add("nomination_completed", False, {"error": str(exc)})
        exit_code = EXIT_TECHNICAL

    record["finished_at"] = now_iso()
    record["cost"] = {"seconds": round(time.perf_counter() - started, 3)}
    record["gate"] = gate.as_dict()
    path = write_json(base / "nomination.json", record)
    for check in gate.checks:
        print(f"  {'ok  ' if check.ok else 'FAIL'} {check.name}")
    counts = record.get("counts") or {}
    print(f"nominate: verdict={record.get('verdict')} "
          f"generations={counts.get('generations')} "
          f"boss={counts.get('boss_generations')} "
          f"candidates={counts.get('candidates')} "
          f"eligible={counts.get('eligible')} rc={exit_code}")
    if record.get("chosen"):
        chosen = record["chosen"]
        print(f"  chosen: {chosen['candidate_id']} slot={chosen['slot']} "
              f"spawn_decision={chosen['spawn_decision']} "
              f"removal_frame={chosen['removal_frame']} "
              f"outcome={chosen['outcome']} span={chosen['span_frames']} "
              f"entry={chosen['entry_decision']}/{chosen['entry_frame']}")
        print(f"  card: {record['task_card']['json']} "
              f"content_sha256={record['task_card']['content_sha256']}")
    else:
        for row in (record.get("near_miss") or [])[:12]:
            print(f"  near miss: {row['candidate_id']} "
                  f"{row.get('outcome')} {row.get('missing')}")
    print(f"wrote: {path}")
    return exit_code


# ---------------------------------------------------------------------------
# 02-C `intervene`: two arms from the same frozen entry
# ---------------------------------------------------------------------------
class FrozenTracker:
    """Follow one frozen enemy slot decision by decision and name its ending.

    The tracker applies the same rule table as :func:`classify_generation_ending`,
    but incrementally, because an arm must be able to *stop* on the consequence:
    the first decision whose row shows a terminal signature ends the arm.  A
    ``boss_phase_end_M3`` is **not** terminal (the row stays), so it is recorded
    and the arm keeps going.
    """

    def __init__(self, *, slot: int, spawn_decision: int, life_at_spawn: int,
                 entry_decision: int) -> None:
        self.slot = int(slot)
        self.spawn_decision = int(spawn_decision)
        self.life_at_spawn = int(life_at_spawn)
        self.entry_decision = int(entry_decision)
        self.seen = False
        self.first_decision: int | None = None
        self.first_frame: int | None = None
        self.last_decision: int | None = None
        self.last_frame: int | None = None
        self.last_life: int | None = None
        self.last_flags1: str | None = None
        self.life_series: list[int] = []
        self.life_decisions: list[int] = []
        self.phase_ends: list[dict[str, Any]] = []
        self.observations = 0
        self.identity_warnings: list[str] = []
        self.outcome: str | None = None
        self.outcome_decision: int | None = None
        self.outcome_frame: int | None = None
        self.outcome_why: str | None = None
        self.removal_ev_flags: Any = None
        self.vanished = False
        self.previous_life: int | None = None

    def observe(self, row: Mapping[str, Any]) -> str | None:
        """Feed one control row; return the terminal outcome the first time."""

        if self.outcome is not None:
            return self.outcome
        index = int(row["decision_index"])
        enemy = next((item for item in row.get("enemies") or []
                      if int(item["slot"]) == self.slot), None)
        word = _flag_word(row)
        if enemy is None:
            if self.seen and not self.vanished:
                self.vanished = True
                self.removal_ev_flags = row.get("ev_flags")
                removal_frame = _frame_with_bit(row, EVENT_ENEMY_REMOVED)
                life = int(self.last_life if self.last_life is not None else 0)
                if removal_frame is None:
                    outcome, why = ("not_removed_no_event",
                                    "the row vanished without ENEMY_REMOVED")
                elif life <= ENEMY_MAX_DAMAGE_PER_FRAME:
                    outcome, why = ("kill_M0",
                                    f"row gone, ENEMY_REMOVED at frame "
                                    f"{removal_frame}, last life {life} <= "
                                    f"{ENEMY_MAX_DAMAGE_PER_FRAME}")
                elif self._last_out_of_playfield:
                    outcome, why = ("offscreen_despawn",
                                    "row gone with life > 70 and the last world "
                                    "position outside the playfield")
                else:
                    outcome, why = ("scripted_retreat",
                                    f"row gone, ENEMY_REMOVED at frame "
                                    f"{removal_frame}, last life {life} > "
                                    f"{ENEMY_MAX_DAMAGE_PER_FRAME}")
                self.outcome = outcome
                self.outcome_why = why
                self.outcome_decision = index
                self.outcome_frame = int(removal_frame or row["frame_end"])
            return self.outcome
        life = int(enemy["life"])
        flags = int(enemy["flags1_dec"])
        if not self.seen:
            self.seen = True
            self.first_decision = index
            self.first_frame = int(row["frame_end"])
            if not enemy["boss"]:
                self.identity_warnings.append(
                    f"slot {self.slot} is not BOSS at decision {index} "
                    f"(flags1={enemy['flags1']})")
            if life != self.life_at_spawn and index > self.spawn_decision:
                self.identity_warnings.append(
                    f"slot {self.slot} carries life {life} at decision {index}, "
                    f"the card's spawn value is {self.life_at_spawn}")
        self.observations += 1
        self.last_decision = index
        self.last_frame = int(row["frame_end"])
        self.last_life = life
        self.last_flags1 = str(enemy["flags1"])
        self._last_out_of_playfield = _out_of_playfield(enemy)
        self.life_series.append(life)
        self.life_decisions.append(index)
        if (enemy["boss"] and self.previous_life is not None
                and not enemy["damageable"] and life == 1
                and word & EVENT_ENEMY_REMOVED
                and not (len(self.phase_ends) and
                         self.phase_ends[-1]["decision"] == index)):
            self.phase_ends.append({
                "decision": index,
                "frame": _frame_with_bit(row, EVENT_ENEMY_REMOVED) or row["frame_end"],
                "ev_flags": row.get("ev_flags"),
                "life_before": int(self.previous_life),
                "life_after": life,
                "kind": "boss_phase_end_M3"})
        self.previous_life = life
        if life == 0 and enemy["persist_after_death"]:
            self.outcome, self.outcome_why = (
                "kill_M1", "life==0 with PERSIST_AFTER_DEATH (DEATH_MODE 1)")
        elif life == 0 and enemy["damageable"] and word & EVENT_ENEMY_REMOVED:
            self.outcome, self.outcome_why = (
                "kill_M2",
                "life==0, still DAMAGEABLE, ENEMY_REMOVED in the same decision "
                "(DEATH_MODE 2)")
        elif life == 0 and not enemy["persist_after_death"] \
                and not word & EVENT_ENEMY_REMOVED:
            self.outcome, self.outcome_why = (
                "clear_wipe",
                "life==0 without ENEMY_REMOVED: a life write, not the death "
                "dispatcher")
        if self.outcome is not None:
            self.outcome_decision = index
            self.outcome_frame = _frame_with_bit(row, EVENT_ENEMY_REMOVED) \
                or int(row["frame_end"])
        return self.outcome

    def summary(self) -> dict[str, Any]:
        """The arm-facing summary: where it was seen, how its life moved, how it ended."""

        return {
            "slot": self.slot,
            "spawn_decision_from_card": self.spawn_decision,
            "life_at_spawn_from_card": self.life_at_spawn,
            "seen": bool(self.seen),
            "first_decision": self.first_decision,
            "first_frame": self.first_frame,
            "last_decision": self.last_decision,
            "last_frame": self.last_frame,
            "observations": int(self.observations),
            "life_first": (int(self.life_series[0]) if self.life_series else None),
            "life_last": (int(self.life_series[-1]) if self.life_series else None),
            "life_min": (int(min(self.life_series)) if self.life_series else None),
            "life_max": (int(max(self.life_series)) if self.life_series else None),
            "life_series": [int(value) for value in self.life_series],
            "life_series_decisions": [int(value) for value in self.life_decisions],
            "phase_ends": self.phase_ends,
            "life_jumps_up": sum(
                1 for left, right in zip(self.life_series, self.life_series[1:])
                if int(right) > int(left)),
            "identity_warnings": self.identity_warnings,
            "outcome": self.outcome,
            "outcome_why": self.outcome_why,
            "outcome_decision": self.outcome_decision,
            "outcome_frame": self.outcome_frame,
            "removal_ev_flags": self.removal_ev_flags,
            "classification": "measured",
        }


def _prefix_walk(*, attempt: EngineAttempt, tape: Mapping[str, Any],
                 entry_decision: int, t04_hash: Mapping[int, str],
                 d1: Mapping[str, Any], rows_path: Path) -> dict[str, Any]:
    """Walk the frozen prefix frame by frame and check every T04 anchor.

    One ``STEP mask 1`` per frame (so the recorded frame numbering is exact), the
    ``EVENTS`` word read after every frame, and the anchor compared at every
    decision boundary from decision 8 (the first one T04 records) to the entry.
    """

    hashes: dict[int, str] = {}
    anchors: list[dict[str, Any]] = []
    frames = 0
    with rows_path.open("w", encoding="utf-8") as handle:
        for index in range(int(entry_decision)):
            masks = tape["masks"][index]
            frame_start = index * DECISION_FRAMES + 1
            ev_flags: list[int] = []
            for offset, mask in enumerate(masks):
                reply = attempt.host.step(int(mask), 1)
                advanced = int(reply.get("advanced", 0) or 0)
                if advanced != 1 or "exit" in reply:
                    raise EngineRefused(
                        f"the prefix stopped at frame {frame_start + offset} "
                        f"(advanced={advanced}, exit={bool('exit' in reply)})")
                frames += advanced
                frame = frame_start + offset
                value = reply.get("hash")
                if value is not None:
                    hashes[frame] = str(value).strip().lower()
                events = _parse_kv(attempt.host.events_raw())
                if events.get("_verb") != "OK" or "frameEventFlags" not in events:
                    raise EngineRefused(
                        f"EVENTS reply is unusable at frame {frame}: "
                        f"{str(events)[:160]}")
                if int(events.get("frames", -1)) != frame:
                    raise FrameRelationConflict(
                        f"EVENTS frames={events.get('frames')} but the STEP ended "
                        f"on frame {frame}")
                ev_flags.append(int(events["frameEventFlags"], 16))
            boundary = frame_start - 1
            anchor = anchor_at_boundary(
                decision_index=index, frame=boundary, hashes=hashes,
                t04_hash=t04_hash, d1_pre=d1["hash_pre"], d1_post=d1["hash_post"])
            for check in anchor["checks"]:
                if check["kind"] in ("t04_hash_pre", "d1_hash_pre", "d1_hash_post"):
                    anchors.append(check)
            write_jsonl_row(handle, {
                "schema": SCHEMA_INTERVENE_ROW,
                "phase": "prefix",
                "decision_index": index,
                "frame_start": frame_start,
                "frame_end": frame_start + 1,
                "boundary_frame": boundary,
                "cand": int(tape["cands"][index]),
                "masks": [int(masks[0]), int(masks[1])],
                "ev_flags": [f"0x{ev_flags[0]:04x}", f"0x{ev_flags[1]:04x}"],
                "hash": {"frame_start": hashes.get(frame_start),
                         "frame_end": hashes.get(frame_start + 1)},
                "anchor": {"kind": anchor["kind"], "frame": anchor["frame"],
                           "expected": anchor["expected"],
                           "actual": anchor["actual"], "ok": anchor["ok"],
                           "checks": anchor["checks"]},
                "state_read": False,
                "classification": "measured",
            })
    classes: dict[str, dict[str, Any]] = {}
    for check in anchors:
        bucket = classes.setdefault(check["kind"], {"compared": 0, "mismatches": 0,
                                                    "first": None})
        bucket["compared"] += 1
        if not check["ok"]:
            bucket["mismatches"] += 1
            if bucket["first"] is None:
                bucket["first"] = check
    return {"decisions": int(entry_decision), "frames": frames,
            "hash_at_entry": hashes.get(int(entry_decision) * DECISION_FRAMES),
            "anchors": classes, "anchor_rows": len(anchors)}


def run_intervene_arm(*, args: argparse.Namespace, out_root: Path, repo: Path,
                      card: Mapping[str, Any], tape: Mapping[str, Any],
                      t04_hash: Mapping[int, str], d1: Mapping[str, Any],
                      arm: str, limits: BudgetLimits, meter: Any) -> dict[str, Any]:
    """One arm: the shared frozen prefix, then its own controller for the window."""

    entry = int(card["entry"]["prefix_decisions"])
    entry_frame = int(card["entry"]["frame"])
    frame_cap = int(card["frame_cap"]["frames"])
    decision_cap = int(card["frame_cap"]["decisions"])
    base = out_root / "02c" / ("arm_T" if arm == "T" else "arm_C")
    base.mkdir(parents=True, exist_ok=True)
    prefix_path = base / "prefix.jsonl"
    control_path = base / "decisions.jsonl"
    if arm == "T":
        controller = "recorded_tape_masks"
        attempt_id = "02c-armT"
        cands_control = list(tape["cands"])
        description = ("arm T: the recorded T04 continuation (same executor, same "
                       "SHOOT pulse phase, same release rule)")
    else:
        controller = "constant_cand_12"
        attempt_id = "02c-armC"
        cands_control = [CONTROL_CAND] * (entry + decision_cap + 1)
        description = ("arm C: constant candidate 12 (mask "
                       f"{int(adapter()['cand_to_mask'](CONTROL_CAND))}), the same "
                       "pulse phase and the same release rule")
    tracker = FrozenTracker(
        slot=int(card["event"]["seed12_instance"]["slot"]),
        spawn_decision=int(card["event"]["seed12_instance"]["spawn_decision"]),
        life_at_spawn=int(card["event"]["seed12_instance"]["life_at_spawn"]),
        entry_decision=entry)
    planned = entry_frame + frame_cap
    attempt = EngineAttempt(
        args=args, out_root=out_root, base_dir=base, attempt_id=attempt_id,
        planned_frames=int(planned),
        note=f"{description}; entry decision {entry} (frame {entry_frame}), "
             f"window <= {decision_cap} decisions / {frame_cap} frames",
        limits=limits, meter=meter)
    arm_record: dict[str, Any] = {
        "arm": arm,
        "attempt_id": attempt_id,
        "controller": controller,
        "description": description,
        "base": base.as_posix(),
        "engine_started": False,
        "entry": {"card_decision": entry, "card_frame": entry_frame,
                  "card_hash": card["entry"]["hash"], "reached": False,
                  "hash": None, "hash_matches": False},
        "frozen_identity": {
            "slot": int(card["event"]["seed12_instance"]["slot"]),
            "spawn_decision": int(card["event"]["seed12_instance"]["spawn_decision"]),
            "spawn_frame": int(card["event"]["seed12_instance"]["spawn_frame"]),
            "life_at_spawn": int(card["event"]["seed12_instance"]["life_at_spawn"]),
        },
        "counts": {"prefix_decisions": 0, "prefix_frames": 0,
                   "control_decisions": 0, "control_frames": 0,
                   "engine_frames": 0},
        "terminal": {"kind": "unknown"},
        "classification": "measured",
    }
    exit_code = EXIT_TECHNICAL
    closed: dict[str, Any] = {}
    try:
        opened = attempt.open()
        arm_record["engine_started"] = True
        arm_record.update({
            "preflight": opened["preflight"],
            "private_game_dir": opened["private"],
            "reset": opened["reset"],
            "engine_session": opened["session"],
            "engine_env": opened["engine_env"] if "engine_env" in opened
            else attempt.module.engine_env_record(),
        })
        prefix = _prefix_walk(attempt=attempt, tape=tape, entry_decision=entry,
                              t04_hash=t04_hash, d1=d1, rows_path=prefix_path)
        arm_record["counts"].update({"prefix_decisions": prefix["decisions"],
                                     "prefix_frames": prefix["frames"]})
        arm_record["prefix"] = {"decisions": prefix["decisions"],
                                "frames": prefix["frames"],
                                "rows": prefix_path.as_posix(),
                                "anchors": prefix["anchors"],
                                "anchor_rows": prefix["anchor_rows"],
                                "classification": "measured"}
        reached = (prefix["hash_at_entry"] is not None
                   and str(prefix["hash_at_entry"]) == str(card["entry"]["hash"]))
        arm_record["entry"].update({"reached": prefix["hash_at_entry"] is not None,
                                    "hash": prefix["hash_at_entry"],
                                    "hash_matches": bool(reached)})
        arm_record["resources_at_entry"] = {
            "source": "the frozen card (measured by the recon at the entry frame)",
            "state": dict(card["resources_at_entry"]["state"]),
            "hud": dict(card["resources_at_entry"]["hud"]),
            "player": dict(card["resources_at_entry"]["player"]),
            "hash": card["entry"]["hash"],
            "classification": "measured"}
        if not reached:
            arm_record["terminal"] = {
                "kind": "entry_not_reached",
                "detail": (f"the prefix ended on hash "
                           f"{prefix['hash_at_entry']} instead of the card's "
                           f"{card['entry']['hash']}"),
                "engine_started": True}
            exit_code = EXIT_GATE_FAILED
            raise EngineRefused(
                f"arm {arm}: the entry hash does not match the frozen card "
                f"({prefix['hash_at_entry']} != {card['entry']['hash']})")
        # -- the controlled window --------------------------------------
        deaths_at_start = float(card["resources_at_entry"]["state"]["deaths"])
        lives_at_start = float(card["resources_at_entry"]["state"]["lives_remaining"])
        last_row: dict[str, Any] | None = None
        consequence: str | None = None
        stop_reason = "decision_cap"
        frames_in_window = 0
        with control_path.open("w", encoding="utf-8") as handle:
            for index in range(entry, min(entry + decision_cap,
                                          len(cands_control))):
                masks = decision_masks(cands_control, index)
                frame_start = index * DECISION_FRAMES + 1
                frame_end = frame_start + DECISION_FRAMES - 1
                if frames_in_window + DECISION_FRAMES > frame_cap:
                    stop_reason = "frame_cap"
                    break
                ev_flags: list[int] = []
                reply: dict[str, Any] = {}
                hashes: dict[int, str] = {}
                shortfall = None
                for offset, mask in enumerate(masks):
                    reply = attempt.host.step(int(mask), 1)
                    advanced = int(reply.get("advanced", 0) or 0)
                    if advanced != 1 or "exit" in reply:
                        shortfall = {"frame": frame_start + offset,
                                     "advanced": advanced,
                                     "exit": bool("exit" in reply)}
                        break
                    frames_in_window += advanced
                    frame = frame_start + offset
                    value = reply.get("hash")
                    if value is not None:
                        hashes[frame] = str(value).strip().lower()
                    events = _parse_kv(attempt.host.events_raw())
                    if events.get("_verb") != "OK" or "frameEventFlags" not in events:
                        raise EngineRefused(
                            f"EVENTS reply is unusable at frame {frame}")
                    if int(events.get("frames", -1)) != frame:
                        raise FrameRelationConflict(
                            f"EVENTS frames={events.get('frames')} but the STEP "
                            f"ended on frame {frame}")
                    ev_flags.append(int(events["frameEventFlags"], 16))
                if shortfall is not None:
                    stop_reason = ("engine_exit" if shortfall["exit"]
                                   else "frame_shortfall")
                    break
                reads = read_decision(attempt.host)
                row = {
                    "schema": SCHEMA_INTERVENE_ROW,
                    "phase": "control",
                    "arm": arm,
                    "decision_index": index,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "cand": int(cands_control[index]),
                    "masks": [int(masks[0]), int(masks[1])],
                    "release_due": bool(adapter()["release_due"](
                        index, DEFAULT_PULSE_PERIOD)),
                    "ev_flags": [f"0x{ev_flags[0]:04x}", f"0x{ev_flags[1]:04x}"],
                    "ev_flags_dec": [int(ev_flags[0]), int(ev_flags[1])],
                    "ev_names": [event_names(ev_flags[0]), event_names(ev_flags[1])],
                    "state": engine_state_record(reads["state"]),
                    "hud": hud_record(reads["hud"]),
                    "player": player_record(reads["player"]),
                    "enemies": reads["hazards"]["enemies"],
                    "counts": {"bullets": reads["hazards"]["bullets"],
                               "lasers": reads["hazards"]["lasers"],
                               "enemies": reads["hazards"]["enemies_count"]},
                    "hash": {"frame_start": hashes.get(frame_start),
                             "frame_end": hashes.get(frame_end)},
                    "classification": "measured",
                }
                observed = tracker.observe(row)
                row["frozen"] = {
                    "slot": tracker.slot,
                    "present": any(int(item["slot"]) == tracker.slot
                                   for item in reads["hazards"]["enemies"]),
                    "life": tracker.last_life if tracker.last_life is not None
                    else None,
                    "outcome_so_far": observed,
                    "phase_ends_so_far": len(tracker.phase_ends),
                }
                write_jsonl_row(handle, row)
                last_row = row
                arm_record["counts"]["control_decisions"] += 1
                arm_record["counts"]["control_frames"] += DECISION_FRAMES
                if observed is not None:
                    consequence = observed
                    stop_reason = "frozen_consequence"
                    break
                game_over = int(reads["hud"].get("game_over", 0) or 0) \
                    or int(reply.get("game_over", 0) or 0)
                deaths = float(reads["state"]["deaths"])
                if game_over:
                    consequence = "game_over"
                    stop_reason = "game_over"
                    break
                if deaths > deaths_at_start:
                    consequence = "legal_death"
                    stop_reason = "legal_death"
                    break
            else:
                stop_reason = "decision_cap"
        arm_record["counts"]["engine_frames"] = (
            prefix["frames"] + frames_in_window)
        attempt.engine_frames = arm_record["counts"]["engine_frames"]
        arm_record["engine_frame"] = {
            "prefix_last_frame": entry_frame,
            "window_frames": frames_in_window,
            "last_decision": (None if last_row is None
                              else int(last_row["decision_index"])),
            "last_frame": (None if last_row is None else int(last_row["frame_end"])),
            "classification": "measured"}
        arm_record["resources_at_end"] = (
            None if last_row is None else {
                "frame": int(last_row["frame_end"]),
                "state": dict(last_row["state"]),
                "hud": dict(last_row["hud"]),
                "player": dict(last_row["player"]),
                "counts": dict(last_row["counts"]),
                "classification": "measured"})
        arm_record["deaths_observed"] = (
            0.0 if last_row is None else
            float(last_row["state"]["deaths"]) - deaths_at_start)
        arm_record["lives_lost"] = (
            0.0 if last_row is None else
            lives_at_start - float(last_row["state"]["lives_remaining"]))
        if consequence is None:
            consequence = "not_removed_within_cap"
        arm_record["frozen_generation"] = tracker.summary()
        arm_record["outcome"] = consequence
        arm_record["outcome_decision"] = tracker.outcome_decision if (
            consequence in OUTCOME_VOCABULARY) else None
        arm_record["outcome_frame"] = tracker.outcome_frame
        arm_record["stop_reason"] = stop_reason
        arm_record["terminal"] = {
            "kind": ("frozen_consequence" if consequence == "kill_M0" or
                     consequence in OUTCOME_VOCABULARY and consequence not in
                     ("legal_death", "game_over") else consequence),
            "stop_reason": stop_reason,
            "outcome": consequence,
            "window_frames": frames_in_window,
            "last_decision": (None if last_row is None
                              else int(last_row["decision_index"])),
            "game_over": (0 if last_row is None
                          else int(last_row["hud"]["game_over"])),
            "stage": (None if last_row is None else int(last_row["state"]["stage"])),
            "classification": "measured"}
        exit_code = EXIT_OK
    except (EngineRefused, FrameRelationConflict, SourceError) as exc:
        arm_record["terminal"] = {"kind": "technical_invalid",
                                  "reason": type(exc).__name__,
                                  "detail": str(exc),
                                  "engine_started": bool(
                                      arm_record.get("engine_started"))}
        exit_code = EXIT_TECHNICAL
    except Exception as exc:  # noqa: BLE001
        arm_record["terminal"] = {"kind": "technical_invalid",
                                  "reason": type(exc).__name__,
                                  "detail": str(exc),
                                  "traceback": traceback.format_exc(),
                                  "engine_started": bool(
                                      arm_record.get("engine_started"))}
        exit_code = EXIT_TECHNICAL
    finally:
        if attempt.engine_started or (attempt.reservation is not None
                                      and attempt.settlement is None):
            try:
                attempt.engine_frames = int(
                    arm_record["counts"].get("engine_frames")
                    or arm_record["counts"].get("prefix_frames", 0))
                closed = attempt.close(result={
                    "command": f"02c-arm{arm}",
                    "terminal": arm_record["terminal"].get("kind"),
                    "outcome": arm_record.get("outcome"),
                    "reached_frame": attempt.engine_frames})
                arm_record["mute"] = closed["mute"]
                arm_record["source_dir"] = closed["source_dir"]
                arm_record["ledger"] = closed["ledger"]
                arm_record["measured"] = closed["measured"]
            except Exception as exc:  # noqa: BLE001
                arm_record["close_error"] = f"{type(exc).__name__}: {exc}"
    arm_record["arm_exit_code"] = int(exit_code)
    return arm_record


def cmd_intervene(args: argparse.Namespace) -> int:
    """02-C intervene: the teacher continuation against constant candidate 12.

    Both arms start from their own ``RESET 0 1 0 12``, walk the **frozen** prefix
    (the recorded masks of decisions ``0..entry-1``) frame by frame, prove they
    arrived by reproducing the card's entry hash, and then execute their own
    controller for at most 300 decisions / 600 frames.  The frozen generation is
    tracked on every control decision.
    """

    module = slice_tool()
    maybe_reexec_under_xvfb(enabled=not bool(getattr(args, "no_xvfb", False)))
    started = time.perf_counter()
    out_root = Path(args.out).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve()
    base = out_root / "02c"
    base.mkdir(parents=True, exist_ok=True)
    gate = Gate("02c_intervene")
    meter = module.CostMeter()
    meter.begin()
    record: dict[str, Any] = {
        "schema": SCHEMA_INTERVENE,
        "generated_by": "tools/rl/v8_31_followup.py intervene",
        "created_at": now_iso(),
        "classification": "measured",
        "ticket": TICKET_ID,
        "repo": repo.as_posix(),
        "out": out_root.as_posix(),
        "argv": list(getattr(args, "argv", []) or []),
        "host": host_identity(),
        "heads": git_identity(repo),
        "tool": {"path": TOOL_PATH.as_posix(), "sha256": sha256_file(TOOL_PATH)},
        "slice_helper": slice_identity(),
        "arms": {},
        "engine_starts": 0,
    }
    exit_code = EXIT_TECHNICAL
    try:
        frozen = read_task_card(out_root / "02b")
        card = frozen["card"]
        record["task_card"] = {
            "path": frozen["path"].as_posix(),
            "sha256": frozen["sha256"],
            "content_sha256": frozen["content_sha256"],
            "content_sha256_recomputed": frozen["content_sha256_recomputed"],
            "task_id": card.get("task_id"),
        }
        with meter.measure("io"):
            tape = tape_schedule(repo)
            d1 = d1_replay_plan(repo)
            t04_hash: dict[int, str] = {}
            for row in iter_jsonl(repo / SOURCE_BY_ID["t04-2f-seed12"]["rel"]):
                value = str(row.get("hash_pre") or "").strip().lower()
                if value:
                    t04_hash[int(row["decision_frame"])] = value
        entry = int(card["entry"]["prefix_decisions"])
        entry_frame = int(card["entry"]["frame"])
        frame_cap = int(card["frame_cap"]["frames"])
        decision_cap = int(card["frame_cap"]["decisions"])
        gate.add("task_card_read_back", True, record["task_card"])
        gate.add("card_prefix_digest_matches_the_tape",
                 card["entry"]["prefix_masks_sha256"] == tape_digest(tape, 0, entry),
                 {"card": card["entry"]["prefix_masks_sha256"],
                  "recomputed": tape_digest(tape, 0, entry)})
        gate.add("card_teacher_continuation_matches_the_tape",
                 card["teacher_continuation"]["masks_sha256"]
                 == tape_digest(tape, entry, entry + decision_cap),
                 {"card": card["teacher_continuation"]["masks_sha256"],
                  "recomputed": tape_digest(tape, entry, entry + decision_cap)})
        per_arm_frames = entry_frame + frame_cap
        budget = stage_budget(out_root, need_starts=2,
                              need_frames=2 * per_arm_frames,
                              label="02c-intervene",
                              stage_cap={
                                  "starts": getattr(args, "max_starts", None),
                                  "engine_frames": getattr(
                                      args, "max_engine_frames", None)})
        record["budget"] = budget["report"]
        gate.add("budget_checked_from_the_existing_ledger",
                 not budget["report"]["exceeded"], {
                     "ledger_rows_read": budget["report"]["ledger_rows_read"],
                     "cumulative": budget["report"]["cumulative"],
                     "stage_used": budget["report"]["stage_used"],
                     "stage_remaining": budget["report"]["stage_remaining"],
                     "need": budget["report"]["need"]})
        for arm in ("T", "C"):
            arm_record = run_intervene_arm(
                args=args, out_root=out_root, repo=repo, card=card, tape=tape,
                t04_hash=t04_hash, d1=d1, arm=arm, limits=budget["limits"],
                meter=meter)
            record["arms"][arm] = arm_record
            record["engine_starts"] = int(record["engine_starts"]) + int(
                bool(arm_record.get("engine_started")))
            if not arm_record.get("engine_started"):
                print(f"  arm {arm}: refused before the start "
                      f"({arm_record['terminal'].get('kind')})", flush=True)
                break
        # -- the gate ---------------------------------------------------
        arms = record["arms"]
        arm_t = arms.get("T") or {}
        arm_c = arms.get("C") or {}
        for arm, arm_record in (("T", arm_t), ("C", arm_c)):
            mute = arm_record.get("mute") or {}
            gate.add(f"arm_{arm}_engine_started",
                     bool(arm_record.get("engine_started")),
                     arm_record.get("terminal"))
            gate.add(f"arm_{arm}_entry_reached",
                     bool((arm_record.get("entry") or {}).get("hash_matches")),
                     arm_record.get("entry"))
            anchors = (arm_record.get("prefix") or {}).get("anchors") or {}
            t04 = anchors.get("t04_hash_pre") or {"compared": 0, "mismatches": None}
            d1_pre = anchors.get("d1_hash_pre") or {"compared": 0, "mismatches": None}
            d1_post = anchors.get("d1_hash_post") or {"compared": 0,
                                                     "mismatches": None}
            gate.add(f"arm_{arm}_prefix_anchors",
                     t04.get("mismatches") == 0 and d1_pre.get("mismatches") == 0
                     and d1_post.get("mismatches") == 0 and t04.get("compared", 0) > 0,
                     {"t04_hash_pre": t04, "d1_hash_pre": d1_pre,
                      "d1_hash_post": d1_post})
            gate.add(f"arm_{arm}_mute_after_exit",
                     not (mute.get("after_exit_problems") or []),
                     mute.get("after_exit_problems"))
            gate.add(f"arm_{arm}_source_unchanged",
                     bool((arm_record.get("source_dir") or {}).get("unchanged")),
                     {"before": ((arm_record.get("source_dir") or {}).get(
                         "before") or {}).get("digest"),
                      "after": ((arm_record.get("source_dir") or {}).get(
                          "after") or {}).get("digest")})
            gate.add(f"arm_{arm}_ledger_settled",
                     bool((arm_record.get("ledger") or {}).get("settle_seq")),
                     (arm_record.get("ledger") or {}).get("actual"))
            gate.add(f"arm_{arm}_frames_recorded",
                     int((arm_record.get("counts") or {}).get(
                         "engine_frames", 0)) > 0,
                     arm_record.get("counts"))
        same_entry = (arm_t.get("entry", {}).get("hash") is not None
                      and arm_t.get("entry", {}).get("hash")
                      == arm_c.get("entry", {}).get("hash")
                      == card["entry"]["hash"])
        outcome_t = arm_t.get("outcome")
        outcome_c = arm_c.get("outcome")
        delta = {}
        end_t = arm_t.get("resources_at_end") or {}
        end_c = arm_c.get("resources_at_end") or {}
        if end_t and end_c:
            state_t = end_t.get("state") or {}
            state_c = end_c.get("state") or {}
            for name in ("lives_remaining", "bombs_remaining", "player_power",
                         "score", "graze", "deaths"):
                delta[name] = (state_t.get(name), state_c.get(name))
            delta["hud_globals_score"] = ((end_t.get("hud") or {}).get("globals_score"),
                                          (end_c.get("hud") or {}).get("globals_score"))
            delta["frame"] = (end_t.get("frame"), end_c.get("frame"))
        distinguishable = False
        if outcome_t is not None and outcome_c is not None:
            if outcome_t != outcome_c:
                distinguishable = True
            elif delta:
                distinguishable = any(
                    isinstance(pair, tuple) and len(pair) == 2
                    and pair[0] != pair[1] for pair in delta.values())
        comparison = {
            "same_entry_hash": bool(same_entry),
            "arm_T_outcome": outcome_t,
            "arm_C_outcome": outcome_c,
            "arm_T_terminal": (arm_t.get("terminal") or {}).get("kind"),
            "arm_C_terminal": (arm_c.get("terminal") or {}).get("kind"),
            "distinguishable": bool(distinguishable),
            "resource_pairs_T_vs_C": delta,
            "frozen_life_series": {
                "T": ((arm_t.get("frozen_generation") or {}).get("life_series")),
                "C": ((arm_c.get("frozen_generation") or {}).get("life_series")),
            },
            "frozen_life_tail": {
                "T": ((arm_t.get("frozen_generation") or {}).get("life_series") or [])[-8:],
                "C": ((arm_c.get("frozen_generation") or {}).get("life_series") or [])[-8:],
            },
            "verdict": ("teacher_kill_control_different" if
                        (outcome_t in KILL_OUTCOMES and distinguishable)
                        else ("no_discriminating_power" if outcome_t is not None
                              and outcome_c is not None else "incomplete")),
            "classification": "measured",
        }
        record["comparison"] = comparison
        gate.add("same_entry_hash_both_arms", bool(same_entry),
                 {"card": card["entry"]["hash"],
                  "T": arm_t.get("entry", {}).get("hash"),
                  "C": arm_c.get("entry", {}).get("hash")})
        gate.add("teacher_arm_completed_the_frozen_event",
                 outcome_t in KILL_OUTCOMES,
                 {"outcome": outcome_t,
                  "why": (arm_t.get("frozen_generation") or {}).get("outcome_why"),
                  "outcome_frame": arm_t.get("outcome_frame")})
        gate.add("control_arm_is_distinguishable", bool(distinguishable),
                 {"T": outcome_t, "C": outcome_c,
                  "resource_pairs": delta,
                  "note": ("different outcomes, or the same outcome with a "
                           "reportable resource difference; two identical arms "
                           "mean the control has no discriminating power")})
        gate.add("no_lesson_from_a_policy_result",
                 bool(card.get("not_a_policy_result")) is True,
                 card.get("not_a_policy_result"))
        record["engine_starts"] = int(record["engine_starts"])
        record["arms_summary"] = {
            arm: {
                "controller": (arms.get(arm) or {}).get("controller"),
                "entry_reached": ((arms.get(arm) or {}).get("entry") or {}
                                  ).get("hash_matches"),
                "engine_frames": ((arms.get(arm) or {}).get("counts") or {}
                                  ).get("engine_frames"),
                "control_decisions": ((arms.get(arm) or {}).get("counts") or {}
                                      ).get("control_decisions"),
                "outcome": (arms.get(arm) or {}).get("outcome"),
                "stop_reason": (arms.get(arm) or {}).get("stop_reason"),
                "deaths_observed": (arms.get(arm) or {}).get("deaths_observed"),
                "lives_lost": (arms.get(arm) or {}).get("lives_lost"),
            } for arm in ("T", "C")}
        exit_code = EXIT_OK if gate.passed else EXIT_GATE_FAILED
    except BudgetRefused as exc:
        record["budget"] = dict(exc.payload)
        record["terminal"] = {"kind": "budget_refused", "reason": exc.reason,
                              "detail": exc.payload.get("exceeded")}
        gate.add("budget_checked", False, exc.payload.get("exceeded"))
        exit_code = EXIT_NOT_IMPLEMENTED
    except (EngineRefused, SourceError) as exc:
        record["terminal"] = {"kind": "technical_invalid",
                              "reason": type(exc).__name__, "detail": str(exc),
                              "engine_started": bool(record.get("engine_starts"))}
        gate.add("inputs_readable", False, {"error": str(exc)})
        exit_code = EXIT_TECHNICAL
    except Exception as exc:  # noqa: BLE001
        record["terminal"] = {"kind": "technical_invalid",
                              "reason": type(exc).__name__, "detail": str(exc),
                              "traceback": traceback.format_exc(),
                              "engine_started": bool(record.get("engine_starts"))}
        gate.add("intervene_completed", False, {"error": str(exc)})
        exit_code = EXIT_TECHNICAL
    finally:
        meter.finish()
    record.setdefault("terminal", {"kind": "unknown"})
    record["finished_at"] = now_iso()
    record["cost"] = meter.payload(step_units=0)
    record["gate"] = gate.as_dict()
    record["classification"] = "measured"
    path = write_json(base / "intervene.json", record)
    for check in gate.checks:
        print(f"  {'ok  ' if check.ok else 'FAIL'} {check.name}")
    comparison = record.get("comparison") or {}
    print(f"intervene: starts={record.get('engine_starts')} "
          f"T={comparison.get('arm_T_outcome')} C={comparison.get('arm_C_outcome')} "
          f"distinguishable={comparison.get('distinguishable')} "
          f"verdict={comparison.get('verdict')} rc={exit_code}")
    print(f"wrote: {path}")
    if exit_code == EXIT_OK:
        print(f"gate: passed ({len(gate.checks)} checks)")
    else:
        print(f"gate: {'FAILED' if exit_code == EXIT_GATE_FAILED else 'technical'} "
              f"({len(gate.problems)} failed checks)")
        for problem in gate.problems[:12]:
            print(f"  - {problem}")
    return exit_code


# ---------------------------------------------------------------------------
# 03 `pilot`: one frozen pixel-BC fit, the open-loop table and the three arms
# ---------------------------------------------------------------------------
#: ``TRAINING_HOST_REQUEST_02.md`` L42-L52.  The pilot is **one** supervised fit
#: (init seed 0), an open-loop action table derived from that fit on an
#: independent **donor** root, and three arms on every development root: the
#: initial weights, the final weights and the frozen table.  The table is written
#: and hashed **before** the first development root is started, and no arm ever
#: reads the fit root's pixels.
SCHEMA_PILOT = "v831-03-pilot/1"
SCHEMA_PILOT_ROOTS = "v831-pilot03-roots/1"
SCHEMA_PILOT_TRAIN = "v831-03-train/1"
SCHEMA_PILOT_TABLE = "v831-03-open-loop/1"
SCHEMA_PILOT_ARM = "v831-03-arm/1"
SCHEMA_PILOT_ROW = "v831-03-decision/1"
SCHEMA_PILOT_PREFIX_ROW = "v831-03-prefix/1"

PILOT_TICKET = "V831-PILOT-03 / TRAINING_HOST_REQUEST_02.md#L42-L52"
#: The one new registration file this ticket allows (``configs/v8_31/``).
#: Nothing is written beside it: the file-bytes sha256 goes into the pilot
#: evidence (``03/.../pilot.json`` → ``roots_config.sha256``), not into a
#: companion checksum file outside the allowed write scope.
PILOT_CONFIG_REL = "configs/v8_31/pilot_03_roots.json"
#: The four development roots this pilot allocates (v4.3.0's reserved 1020-1039).
PILOT_DEV_SEEDS: tuple[int, ...] = (1020, 1021, 1022, 1023)
#: The technical smoke seed: inside the legacy dev panel, **never** a pilot root.
PILOT_SMOKE_SEED = 1000
PILOT_RESERVED_SEEDS = {"seed_min": 1020, "seed_max": 1039}
PILOT_INIT_SEED = 0
PILOT_LR = 0.0003
PILOT_BATCH = 32
PILOT_UPDATES = 200
PILOT_GRAD_CLIP = 5.0
PILOT_WALL_CAP_S = 1800.0
PILOT_WINDOW_DECISIONS = 300
PILOT_WINDOW_FRAMES = 600
PILOT_PARAMETERS = 2_602_490
PILOT_INITIAL_NAME = "weights_initial.pt"
PILOT_FINAL_NAME = "weights_final.pt"
PILOT_ARMS: tuple[str, ...] = ("init", "final", "open_loop")
PILOT_FIT_SEED = 13
PILOT_DONOR_SEED = 14
PILOT_FIT_ROOT_ID = "d1-n3371-seed13"
PILOT_DONOR_ROOT_ID = "d1-n3371-seed14"
PILOT_D1_REL = ".tmp_v400_d1/d1"
PILOT_SMOKE_CLASSIFICATION = "technical_smoke_not_a_pilot_result"
#: This stage's allowance (the ticket's 02/03 stage cap for this run): 3 starts
#: and 11,200 engine frames.  The smoke is 3 arms x 3,686 frames = 11,058.
STAGE_CAP_03_STARTS = 3
STAGE_CAP_03_FRAMES = 11_200
PILOT_ATTEMPT_PREFIXES: tuple[str, ...] = ("03-",)
PILOT_WINDOW_KEY = "03"


def pilot_layout(out_root: str | Path, *, smoke: bool) -> dict[str, Any]:
    """Where one ``pilot`` invocation writes: everything under ``03/``.

    A smoke run (``--only-seed 1000``) writes **every** artefact under
    ``03/smoke/`` and is marked ``technical_smoke_not_a_pilot_result``: its
    numbers may never be quoted as a pilot result.  A formal run writes
    ``03/train``, ``03/open_loop`` and ``03/dev/<seed>/<arm>``.
    """

    root = Path(out_root) / PILOT_WINDOW_KEY
    base = root / "smoke" if smoke else root
    return {
        "out": Path(out_root),
        "root": root,
        "base": base,
        "smoke": bool(smoke),
        "train": base / "train",
        "open_loop": base / "open_loop",
        "arms": (base if smoke else base / "dev"),
        "summary": base / "pilot.json",
        "classification": (PILOT_SMOKE_CLASSIFICATION if smoke else "measured"),
    }


# -- the frozen development-root registration -------------------------------
def _shard_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def d1_root_row(repo: str | Path, seed: int, stage: int = 0) -> dict[str, Any]:
    """The registered D1 shard row of one ``(stage, seed)`` pair.

    ``train_v3.complete_shard_rows`` is the frozen reader: only shards whose
    ``shard.json`` says ``complete: true`` are returned, so a half-written shard
    can never become a training root.
    """

    from touhou_rl.v400.student import train_v3  # noqa: PLC0415

    root = Path(repo) / PILOT_D1_REL
    if not root.is_dir():
        raise SourceError(f"the D1 corpus is missing: {root}")
    rows = train_v3.complete_shard_rows(root)
    matches = [row for row in rows
               if _shard_int(row.get("seed")) == int(seed)
               and _shard_int(row.get("stage")) == int(stage)]
    if not matches:
        raise SourceError(
            f"no complete D1 shard for stage {stage} seed {seed} under {root}")
    matches.sort(key=lambda row: str(row["shard"]))
    return matches[0]


def d1_root_identity(repo: str | Path, seed: int, stage: int = 0) -> dict[str, Any]:
    """One training root's identity: shard, labels sha, the three obs shas."""

    row = d1_root_row(repo, seed, stage)
    shard_dir = Path(row["dir"])
    meta = _load_json(shard_dir / "shard.json")
    obs = meta.get("obs") or {}
    ancestor = (f"teacher_query|d1-teacher-fast|{row['shard']}|"
                "th08-stage1-normal-team0")
    return {
        "id": f"d1-n3371-seed{seed}",
        "registered_ancestor_id": ancestor,
        "shard": str(row["shard"]),
        "path": repo_rel(shard_dir, repo),
        "seed": int(seed),
        "stage": int(stage),
        "difficulty": meta.get("difficulty"),
        "decisions": int(row["decisions"]),
        "frames": meta.get("frames"),
        "shard_json_sha256": sha256_file(shard_dir / "shard.json"),
        "labels_sha256": (meta.get("labels") or {}).get("sha256"),
        "obs_sha256": {
            "play96x112": (obs.get("play96x112") or {}).get("sha256"),
            "hud": (obs.get("hud") or {}).get("sha256"),
            "act_hist": (obs.get("act_hist") or {}).get("sha256"),
        },
        "obs_dtype_shapes": {
            "play96x112": (obs.get("play96x112") or {}).get("shape"),
            "hud": (obs.get("hud") or {}).get("shape"),
            "act_hist": (obs.get("act_hist") or {}).get("shape"),
        },
        "obs_spec_version": meta.get("obs_spec_version"),
        "obs_spec_digest": meta.get("obs_spec_digest"),
        "alignment": (meta.get("alignment") or {}).get("observation_time"),
        "complete": meta.get("complete"),
        "classification": "measured",
    }


def pilot_exposure_block(repo: str | Path) -> dict[str, Any]:
    """Prove 1020-1023 are unexposed *from the registered evidence*.

    Three authoritative checks are recomputed here: no split ancestor names the
    seed, the ``ancestors`` table has no entry for it, and the runs registry is
    empty.  The host-asset sweep is quoted from ``03_roots_analysis.md`` §2 (it is
    a ``os.walk`` over the whitelist roots, not a registration) and the
    prospective policy mentions are reported verbatim with their line numbers.
    """

    repo = Path(repo)
    splits_rel = "configs/v430/splits.json"
    splits_path = repo / splits_rel
    splits = _load_json(splits_path) if splits_path.is_file() else {}
    splits_sha = sha256_file(splits_path) if splits_path.is_file() else None
    ancestors: list[str] = []
    for _name, block in (splits.get("splits") or {}).items():
        for ancestor in (block.get("ancestor_ids") or []):
            ancestors.append(str(ancestor))
    table: list[str] = []
    for entry in (splits.get("ancestors") or []):
        table.append(canonical_json(entry) if isinstance(entry, dict) else str(entry))
    runs_rel = "Document/Update/Alpha/v4.3.0/registries/runs.json"
    runs_path = repo / runs_rel
    runs_entries = None
    if runs_path.is_file():
        try:
            runs_entries = len(_load_json(runs_path).get("entries") or [])
        except (OSError, ValueError):
            runs_entries = None
    holdout_rel = "Document/Update/Alpha/v4.3.0/registries/holdout_families.json"
    holdout_path = repo / holdout_rel
    holdout_mentions: list[dict[str, Any]] = []
    if holdout_path.is_file():
        for number, line in enumerate(
                holdout_path.read_text(encoding="utf-8", errors="replace").splitlines(),
                1):
            if re.search(r"(?<![0-9])102[0-3](?![0-9])", line):
                holdout_mentions.append({"line": number, "text": line.strip()[:400]})
    per_seed: dict[str, Any] = {}
    for seed in PILOT_DEV_SEEDS:
        token = re.compile(rf"(?<![0-9]){seed}(?![0-9])")
        in_ancestors = [value for value in ancestors if token.search(value)]
        in_table = [value for value in table if token.search(value)]
        per_seed[str(seed)] = {
            "seed": int(seed),
            "status": ("exposed" if (in_ancestors or in_table or runs_entries)
                       else "not_exposed"),
            "split_ancestors_naming_the_seed": in_ancestors,
            "ancestors_table_naming_the_seed": len(in_table),
            "runs_registry_entries": runs_entries,
            "evidence": [
                f"{splits_rel}#splits.*.ancestor_ids: {len(ancestors)} ids, "
                f"{len(in_ancestors)} name seed {seed}",
                f"{splits_rel}#ancestors: {len(table)} rows, {len(in_table)} name "
                f"seed {seed}",
                f"{runs_rel}#entries: {runs_entries}",
                "03_roots_analysis.md#2: os.walk over the whitelist assets found no "
                f"1020-1023 entry (host sweep, not repeated here)",
            ],
            "classification": "measured",
        }
    verdict = ("not_exposed"
               if all(item["status"] == "not_exposed" for item in per_seed.values())
               else "exposed")
    return {
        "verdict": verdict,
        "checked": [
            f"{splits_rel}#splits.*.ancestor_ids",
            f"{splits_rel}#ancestors",
            runs_rel,
            holdout_rel,
            "ASSET_SYNC.md (quoted from 03_roots_analysis.md §2)",
            "host asset whitelist (quoted from 03_roots_analysis.md §2)",
        ],
        "split": {"path": splits_rel, "sha256": splits_sha,
                  "frozen_at": splits.get("frozen_at"), "rows": None},
        "splits_registry_digests": pilot_registry_digests(repo, splits),
        "reserved_seed_range": dict(PILOT_RESERVED_SEEDS),
        "per_seed": per_seed,
        "prospective_mentions": holdout_mentions,
        "prospective_mentions_note": (
            "policy text about the reserved 1020-1039 window, not a run reference; "
            "recorded verbatim so the reader can tell the two apart"),
        "classification": "measured",
    }


def pilot_registry_digests(repo: str | Path, splits: Mapping[str, Any] | None = None
                           ) -> dict[str, Any]:
    """``path -> sha256`` of the split file and every registry it binds."""

    repo = Path(repo)
    if splits is None:
        path = repo / "configs/v430/splits.json"
        splits = _load_json(path) if path.is_file() else {}
    digests: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    split_rel = "configs/v430/splits.json"
    split_path = repo / split_rel
    digests[split_rel] = sha256_file(split_path) if split_path.is_file() else None
    for entry in (splits.get("registries") or []):
        rel = str(entry.get("path"))
        path = repo / rel
        actual = sha256_file(path) if path.is_file() else None
        digests[rel] = actual
        checks.append({"role": entry.get("role"), "path": rel,
                       "registered_sha256": entry.get("sha256"),
                       "actual_sha256": actual,
                       "matches": bool(actual is not None
                                       and actual == entry.get("sha256")),
                       "classification": "measured"})
    return {"digests": digests, "checks": checks,
            "classification": "measured"}


def pilot_roots_payload(*, repo: str | Path, card: Mapping[str, Any],
                        created_at: str) -> dict[str, Any]:
    """The frozen ``configs/v8_31/pilot_03_roots.json`` body (no digest yet)."""

    repo = Path(repo)
    exposure = pilot_exposure_block(repo)
    registries = exposure["splits_registry_digests"]
    split = exposure["split"]
    fit = d1_root_identity(repo, PILOT_FIT_SEED, 0)
    donor = d1_root_identity(repo, PILOT_DONOR_SEED, 0)
    split_sha = split.get("sha256")
    heads = git_identity(repo)
    roots = []
    for seed in PILOT_DEV_SEEDS:
        state = exposure["per_seed"][str(seed)]
        roots.append({
            "seed": int(seed),
            "role": "dev",
            "split_binding": {
                "split_sha256": split_sha,
                "reserved_range": (f"{PILOT_RESERVED_SEEDS['seed_min']}-"
                                   f"{PILOT_RESERVED_SEEDS['seed_max']}"),
                "ancestor_id": None,
                "note": ("seeds 1020-1023 are reserved, never registered: this "
                         "manifest is their first and only registration"),
            },
            "exposure": {"status": state["status"], "pretraining": "none",
                         "candidate_selection": "none",
                         "evidence": state["evidence"]},
            "source": (f"fresh RESET 0 1 0 {seed} on the training host; no stored "
                       "asset"),
            "entry": {
                "controller": ("frozen prefix v1: the recorded T04 schedule's "
                               "decisions 0..1542 through the two-frame executor"),
                "prefix_frames": int(card["entry"]["frame"]),
                "prefix_decisions": int(card["entry"]["prefix_decisions"]),
                "period": DEFAULT_PULSE_PERIOD,
                "prefix_masks_sha256": card["entry"]["prefix_masks_sha256"],
                "entry_frame": int(card["entry"]["frame"]),
                "arrived": None,
                "note": ("filled by the arm evidence; the arrival rate is reported, "
                         "never forced"),
            },
            "task_card_sha256": card["content_sha256"],
            "status": "planned",
            "classification": "measured",
        })
    return {
        "schema": SCHEMA_PILOT_ROOTS,
        "manifest_id": "v831-pilot-03-roots/1",
        "created_at": str(created_at),
        "ticket": PILOT_TICKET,
        "repo": repo.as_posix(),
        "repo_head": heads.get("head"),
        "repo_dirty_tracked": int(heads.get("dirty_count") or 0),
        "split": {"path": split["path"], "sha256": split_sha,
                  "frozen_at": split.get("frozen_at"),
                  "digest_scope": "file bytes (git LF checkout)",
                  "note": ("the split's own registries block is checked below, "
                           "entry by entry")},
        "splits_registry_digests": registries["digests"],
        "splits_registry_checks": registries["checks"],
        "reserved_seed_range": dict(PILOT_RESERVED_SEEDS),
        "purpose": ("this pilot's same-family development roots; never training "
                    "data; never the confirm/adapt family"),
        "roots": roots,
        "training_roots": {
            "fit": {**fit, "role": "fit: the pilot's only supervised fit"},
            "donor": {**donor, "role": ("donor: the open-loop table's observation "
                                        "source; never fitted")},
            "separation": (
                "the fit root and the donor root are different seeds of the same "
                "first stage, so they are different runs; S1c seeds 13/14 are "
                "truncation/relabel copies of D1 and therefore cannot supply a "
                "second independent root"),
        },
        "recipe": {
            "obs": "obs-v3",
            "model": "model_v3",
            "arch": "v3",
            "init_seed": PILOT_INIT_SEED,
            "initial_checkpoint_basename": PILOT_INITIAL_NAME,
            "final_checkpoint_basename": PILOT_FINAL_NAME,
            "optimizer": "Adam",
            "lr": PILOT_LR,
            "betas": [0.9, 0.999],
            "batch": PILOT_BATCH,
            "updates": PILOT_UPDATES,
            "grad_clip": PILOT_GRAD_CLIP,
            "loss": "ce18",
            "loss_definition": (
                "cross entropy of the 18-way move head against "
                "one_hot(chosen_plan.first_action), i.e. the 18 executed candidates"),
            "loss_implementation": "touhou_rl.v410.student.bc.bc_loss (bomb weight 0)",
            "readout": "argmax",
            "bomb": ("disabled: the bomb head's argmax is discarded and "
                     "cand_to_mask(cand, 0) is sent"),
            "frames_per_decision": DECISION_FRAMES,
            "pulse_period": DEFAULT_PULSE_PERIOD,
            "wall_cap_s": PILOT_WALL_CAP_S,
            "no_best_point_rollback": True,
            "seed_order": ("torch.manual_seed(0); np.random.seed(0); "
                           "cudnn.deterministic=True; cudnn.benchmark=False"),
            "open_loop_window_decisions": PILOT_WINDOW_DECISIONS,
            "open_loop_history_mode": "self_consistent",
        },
        "exposure": {key: value for key, value in exposure.items()
                     if key not in ("split", "splits_registry_digests")},
        "exclusions": {
            "legacy_dev_1000_1019": "registered dev panel; not allocatable here",
            "reserved_1040_1099": "pre-registered ablations; not allocatable",
            "banned_60000_60199": "sealed confirm family; never allocated",
            "seed_12": "the 02-B/02-C frozen evidence seed; never a dev root",
        },
        "task_card": {
            "path": repo_rel(Path(card.get("_card_path") or ""), repo)
            if card.get("_card_path") else None,
            "content_sha256": card["content_sha256"],
            "task_id": card.get("task_id"),
            "entry_decision": int(card["entry"]["prefix_decisions"]),
            "entry_frame": int(card["entry"]["frame"]),
        },
        "status": "frozen",
        "pending": [
            "the open-loop table's strength check on the fit root (needs its own "
            "engine start; deferred out of the technical smoke)",
            "the per-root arrival witnesses (each arm is its own witness: it replays "
            "the frozen prefix from RESET)",
        ],
        "classification": "measured",
    }


def write_pilot_roots_config(path: str | Path, payload: Mapping[str, Any]
                             ) -> dict[str, Any]:
    """Freeze the registration: **one** JSON file, no companion file.

    The ticket's write scope names exactly one new file,
    ``configs/v8_31/pilot_03_roots.json``, so nothing else is written beside it:
    ``content_sha256`` inside the file is the canonical digest of every other
    field, and the sha256 of the file's bytes is recorded in the pilot evidence
    (``03/.../pilot.json`` → ``roots_config.sha256``).  An existing file is
    **never** overwritten with a different body: that is a refusal, not an update.
    """

    path = Path(path)
    body = {key: value for key, value in dict(payload).items()
            if key != "content_sha256"}
    digest = content_digest(body)
    document = dict(body)
    document["content_sha256"] = digest
    text = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    if path.is_file():
        existing = load_pilot_roots_config(path)
        if existing["content_sha256"] != digest:
            raise EngineRefused(
                f"{path} is already frozen with content_sha256 "
                f"{existing['content_sha256']}, this invocation would write "
                f"{digest}; refusing to overwrite a frozen registration")
        return {**existing, "written": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {"path": path, "sha256": sha256_file(path), "content_sha256": digest,
            "content_sha256_recomputed": digest, "sidecar": None,
            "sidecar_sha256": None, "payload": document, "written": True}


def load_pilot_roots_config(path: str | Path) -> dict[str, Any]:
    """Read a frozen registration and re-verify its two digests.

    Only the file itself is read: the canonical ``content_sha256`` makes a
    companion checksum file unnecessary, and the ticket's write scope allows no
    second file next to the registration.
    """

    path = Path(path)
    if not path.is_file():
        raise SourceError(f"the frozen pilot registration is missing: {path}")
    payload = _load_json(path)
    if payload.get("schema") != SCHEMA_PILOT_ROOTS:
        raise EngineRefused(
            f"{path}: schema {payload.get('schema')!r} is not "
            f"{SCHEMA_PILOT_ROOTS!r}")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    recomputed = content_digest(body)
    recorded = str(payload.get("content_sha256") or "")
    if not recorded or recorded != recomputed:
        raise EngineRefused(
            f"{path}: content_sha256 does not match its own content (recorded "
            f"{recorded!r}, recomputed {recomputed!r})")
    if payload.get("status") != "frozen":
        raise EngineRefused(
            f"{path}: status {payload.get('status')!r} is not 'frozen'")
    return {"path": path, "sha256": sha256_file(path), "sidecar": None,
            "sidecar_sha256": None, "content_sha256": recorded,
            "content_sha256_recomputed": recomputed, "payload": payload}


# -- the one supervised fit -------------------------------------------------
def _array_digest(array: Any) -> str:
    import numpy as np  # noqa: PLC0415

    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def pilot_train(*, repo: str | Path, layout: Mapping[str, Any],
                card: Mapping[str, Any], config_identity: Mapping[str, Any],
                meter: Any) -> dict[str, Any]:
    """One BC fit: seed 0, Adam lr 3e-4, batch 32, **exactly** 200 updates.

    The initial checkpoint is written **before the first forward pass**: a
    ``train()``-mode forward updates the BatchNorm running statistics and would
    change the "initial" sha (03_recipe.md §10.4).  ``train_v3.train_run`` is not
    used: it optimises with AdamW, runs on a time budget and reloads
    ``best_state`` at the end, which contradicts all four frozen requirements.
    """

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from touhou_rl.v400.student import model_v3, obs_spec_v3, train_v3  # noqa: PLC0415
    from touhou_rl.v410.student.bc import bc_loss  # noqa: PLC0415

    started = time.perf_counter()
    base = Path(layout["train"])
    base.mkdir(parents=True, exist_ok=True)
    initial_path = base / PILOT_INITIAL_NAME
    final_path = base / PILOT_FINAL_NAME
    cuda = bool(torch.cuda.is_available())
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if cuda else "cpu")

    torch.manual_seed(PILOT_INIT_SEED)
    np.random.seed(PILOT_INIT_SEED)
    model = model_v3.build_model_v3({"arch": "v3"})
    parameters = int(model.n_parameters())
    provenance = {
        "kind": "bc-v3-pilot-03",
        "controller": "tools/rl/v8_31_followup.py pilot",
        "ticket": PILOT_TICKET,
        "task_card_content_sha256": card["content_sha256"],
        "pilot_config_content_sha256": config_identity["content_sha256"],
    }
    with meter.measure("io"):
        model_v3.save_checkpoint_v3(
            initial_path, model,
            train={"init_seed": PILOT_INIT_SEED, "staging": "initial", "updates": 0,
                   "parameters": parameters},
            provenance=provenance)
    initial_sha = model_v3.checkpoint_digest(initial_path)

    # "先存 initial" is only meaningful if a second seed-0 build is bit-identical.
    torch.manual_seed(PILOT_INIT_SEED)
    np.random.seed(PILOT_INIT_SEED)
    rebuilt = model_v3.build_model_v3({"arch": "v3"})
    reference = model.state_dict()
    max_abs = 0.0
    for key, value in rebuilt.state_dict().items():
        other = reference[key]
        if tuple(value.shape) != tuple(other.shape):
            raise EngineRefused(
                f"the rebuilt model's tensor {key} is {tuple(value.shape)}, the "
                f"saved one {tuple(other.shape)}")
        max_abs = max(max_abs, float((value - other).abs().max()))
    del rebuilt

    row = d1_root_row(repo, PILOT_FIT_SEED, 0)
    shard_meta = _load_json(Path(row["dir"]) / "shard.json")
    with meter.measure("io"):
        block = train_v3.load_shard_block(row, value_target="none")
    rows = int(len(block))
    if rows < PILOT_BATCH:
        raise SourceError(
            f"the fit root has {rows} decisions, a batch of {PILOT_BATCH} needs at "
            "least that many")

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=PILOT_LR,
                                 betas=(0.9, 0.999))
    order = np.random.default_rng(PILOT_INIT_SEED).permutation(rows)
    cursor = 0
    passes = 0
    losses: list[float] = []
    seen: set[int] = set()
    grad_norms: list[float] = []
    with meter.measure("io"):
        for step in range(PILOT_UPDATES):
            if cursor + PILOT_BATCH > len(order):
                order = np.random.default_rng(step).permutation(rows)
                cursor = 0
                passes += 1
            index = order[cursor:cursor + PILOT_BATCH]
            cursor += PILOT_BATCH
            seen.update(int(value) for value in index)
            play = torch.from_numpy(np.ascontiguousarray(block.play[index])
                                    ).to(device).float().div_(255.0)
            hud = torch.from_numpy(np.ascontiguousarray(block.hud[index])
                                   ).to(device).float().div_(255.0).unsqueeze(1)
            hist = torch.from_numpy(np.ascontiguousarray(block.act_hist[index])
                                    ).to(device).float()
            chosen = torch.from_numpy(
                np.asarray(block.chosen[index], dtype=np.int64)).to(device)
            out = model(play, hud, hist)
            onehot = torch.nn.functional.one_hot(
                chosen, 18).float()
            loss, _parts = bc_loss(out.move, onehot)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), PILOT_GRAD_CLIP))
            optimizer.step()
            losses.append(float(loss.detach()))
            grad_norms.append(norm)
    model.eval()
    with meter.measure("io"):
        model_v3.save_checkpoint_v3(
            final_path, model,
            train={"init_seed": PILOT_INIT_SEED, "updates": PILOT_UPDATES,
                   "optimizer": f"Adam(lr={PILOT_LR})",
                   "batch": PILOT_BATCH, "loss": "ce18",
                   "loss_first": losses[0], "loss_last": losses[-1],
                   "samples_seen": PILOT_UPDATES * PILOT_BATCH,
                   "distinct_decisions": len(seen)},
            provenance=provenance)
    final_sha = model_v3.checkpoint_digest(final_path)
    seconds = round(time.perf_counter() - started, 3)
    classes = np.bincount(np.asarray(block.chosen, dtype=np.int64), minlength=18)
    record = {
        "schema": SCHEMA_PILOT_TRAIN,
        "created_at": now_iso(),
        "classification": "measured",
        "ticket": PILOT_TICKET,
        "repo": Path(repo).as_posix(),
        "out": Path(layout["out"]).as_posix(),
        "mode": ("smoke" if layout["smoke"] else "formal"),
        "host": host_identity(),
        "heads": git_identity(repo),
        "tool": {"path": TOOL_PATH.as_posix(), "sha256": sha256_file(TOOL_PATH)},
        "device": str(device),
        "torch": torch.__version__,
        "cuda": cuda,
        "cudnn": {"deterministic": bool(cuda and torch.backends.cudnn.deterministic),
                  "benchmark": bool(cuda and torch.backends.cudnn.benchmark),
                  "note": ("set only when CUDA is available; the CPU path is "
                           "deterministic by construction")},
        "seed_order": ["torch.manual_seed(0)", "np.random.seed(0)"],
        "init_seed": PILOT_INIT_SEED,
        "arch": "v3",
        "model_config": {"arch": "v3"},
        "obs_spec_version": obs_spec_v3.OBS_SPEC_VERSION,
        "obs_spec_digest": obs_spec_v3.obs_spec_v3_digest(),
        "parameters": parameters,
        "parameter_budget": [2_000_000, 6_000_000],
        "parameter_budget_ok": 2_000_000 <= parameters <= 6_000_000,
        "optimizer": {"kind": "Adam", "lr": PILOT_LR, "betas": [0.9, 0.999]},
        "updates": PILOT_UPDATES,
        "batch": PILOT_BATCH,
        "grad_clip": PILOT_GRAD_CLIP,
        "samples_seen": PILOT_UPDATES * PILOT_BATCH,
        "distinct_decisions": len(seen),
        "shard_passes": passes,
        "loss": {
            "name": "ce18",
            "implementation": "touhou_rl.v410.student.bc.bc_loss",
            "first": losses[0], "last": losses[-1],
            "mean": float(sum(losses) / len(losses)),
            "min": float(min(losses)), "max": float(max(losses)),
            "series": losses,
        },
        "grad_norm": {"first": grad_norms[0], "last": grad_norms[-1],
                      "max": float(max(grad_norms))},
        "fit_shard": {
            "id": PILOT_FIT_ROOT_ID,
            "shard": str(row["shard"]),
            "path": repo_rel(Path(row["dir"]), repo),
            "seed": int(row["seed"]),
            "stage": _shard_int(row.get("stage")),
            "rows": rows,
            "shard_json_sha256": sha256_file(block.dir / "shard.json"),
            "labels_sha256": (shard_meta.get("labels") or {}).get("sha256"),
            "obs_sha256": {
                "play96x112": ((shard_meta.get("obs") or {}).get(
                    "play96x112") or {}).get("sha256"),
                "hud": ((shard_meta.get("obs") or {}).get("hud") or {}).get(
                    "sha256"),
                "act_hist": ((shard_meta.get("obs") or {}).get("act_hist")
                             or {}).get("sha256"),
            },
            "obs_spec_digest": block.obs_spec_digest,
            "chosen_histogram": {str(index): int(count)
                                 for index, count in enumerate(classes)},
            "distinct_classes": int((classes > 0).sum()),
            "bomb_positives": int(np.asarray(block.bomb).sum()),
        },
        "weights_initial": {"path": initial_path.as_posix(),
                            "sha256": initial_sha,
                            "bytes": int(initial_path.stat().st_size),
                            "basename": PILOT_INITIAL_NAME},
        "weights_final": {"path": final_path.as_posix(),
                          "sha256": final_sha,
                          "bytes": int(final_path.stat().st_size),
                          "basename": PILOT_FINAL_NAME},
        "initial_rebuild_max_abs_diff": max_abs,
        "initial_rebuild_bitwise_identical": max_abs == 0.0,
        "bn_note": (
            "the initial checkpoint is written before the first forward pass; the "
            "200 train-mode forwards update the BatchNorm running statistics, so "
            "the final file differs in those buffers as well as in the weights"),
        "no_best_point_rollback": True,
        "best_point_note": ("no validation split and no reload: the file on disk "
                           "is the state after the 200th optimizer step"),
        "wall_cap_s": PILOT_WALL_CAP_S,
        "wall_cap_exceeded": bool(seconds > PILOT_WALL_CAP_S),
        "seconds": seconds,
        "pilot_config": {"path": Path(config_identity["path"]).as_posix(),
                         "sha256": config_identity["sha256"],
                         "content_sha256": config_identity["content_sha256"]},
    }
    path = write_json(base / "train.json", record)
    record["_path"] = path.as_posix()
    return record


# -- the frozen open-loop action table --------------------------------------
def pilot_table_masks(cands: Sequence[int], index_base: int, *,
                      period: int = DEFAULT_PULSE_PERIOD) -> list[list[int]]:
    """Two STEP masks per decision from the frozen executor.

    ``index_base + k`` is the **global** decision index, because the executor's
    SHOUT pulse phase is global and the arms continue it after the frozen prefix.
    """

    table = adapter()
    masks: list[list[int]] = []
    for offset, cand in enumerate(cands):
        release = bool(table["release_due"](int(index_base) + offset, int(period)))
        parts = table["commit_frames"](int(cand), decision_frames=DECISION_FRAMES,
                                       release_shoot=release)
        values = expand_commit(parts)
        if len(values) != DECISION_FRAMES:
            raise SourceError(
                f"candidate {cand} at decision {int(index_base) + offset} expanded "
                f"to {len(values)} masks")
        masks.append([int(values[0]), int(values[1])])
    return masks


def table_sha256(cands: Sequence[int]) -> str:
    """``sha256`` of ``"<i>:<cand>\\n"`` over the whole table (03_recipe.md §7)."""

    payload = "".join(f"{index}:{int(cand)}\n" for index, cand in enumerate(cands))
    return sha256_bytes(payload.encode("ascii"))


def flat_masks_of(masks: Sequence[Sequence[int]], first_frame: int) -> dict[int, int]:
    """Frame -> mask for a table: decision ``k`` owns ``first_frame + 2k`` (+1)."""

    flat: dict[int, int] = {}
    for offset, pair in enumerate(masks):
        flat[int(first_frame) + 2 * offset] = int(pair[0])
        flat[int(first_frame) + 2 * offset + 1] = int(pair[1])
    return flat


def build_open_loop_table(*, repo: str | Path, layout: Mapping[str, Any],
                          card: Mapping[str, Any], final_model: Any,
                          final_sha: str, meter: Any) -> dict[str, Any]:
    """Arm three's controller: the final actor's argmax on the donor's pixels.

    Nothing here reads the target episode: the observations come from the donor
    root's recorded ``obs-v3`` planes, one donor row (4 recorded frames) feeding
    **two** consecutive student decisions (zero-order hold), exactly as
    ``03_recipe.md`` §7 maps them.  Two history modes are evaluated; the frozen
    table is the self-consistent one and the difference is recorded.
    """

    import torch  # noqa: PLC0415
    from touhou_rl.v400.student import obs_v3_dataset  # noqa: PLC0415
    from touhou_rl.v400.student.obs_spec_v2 import ActionHistoryBuffer  # noqa: PLC0415

    started = time.perf_counter()
    base = Path(layout["open_loop"])
    base.mkdir(parents=True, exist_ok=True)
    donor_row = d1_root_row(repo, PILOT_DONOR_SEED, 0)
    with meter.measure("io"):
        donor = obs_v3_dataset.load_teacher_shard_v3(donor_row["dir"],
                                                    verify_sha=True,
                                                    verify_history=True)
    entry = int(card["entry"]["prefix_decisions"])
    entry_frame = int(card["entry"]["frame"])
    r0 = entry_frame // 4
    total_rows = int(len(donor))
    needs = r0 + (PILOT_WINDOW_DECISIONS - 1) // 2
    tail_rule = None
    tail_repeats = 0
    if needs > total_rows - 1:
        tail_rule = ("repeat_last_candidate: the donor has no row for the tail, so "
                     "the table repeats its last candidate (TapePolicy semantics)")
        tail_repeats = int(needs - (total_rows - 1))
    history = ActionHistoryBuffer()
    device = next(final_model.parameters()).device
    cands: list[int] = []
    teacher_cands: list[int] = []
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(PILOT_WINDOW_DECISIONS):
            row_index = min(r0 + index // 2, total_rows - 1)
            sample = donor.normalized(row_index)
            play = torch.from_numpy(sample["play"][None]).float().to(device)
            hud = torch.from_numpy(sample["hud"][None]).float().to(device)
            own = torch.from_numpy(history.rows()[None]).float().to(device)
            cand, bomb = final_model.act(play, hud, own)
            teacher_hist = torch.from_numpy(
                sample["act_hist"][None]).float().to(device)
            teacher_cand, teacher_bomb = final_model.act(play, hud, teacher_hist)
            cands.append(int(cand))
            teacher_cands.append(int(teacher_cand))
            history.push(int(cand))
            rows.append({
                "schema": SCHEMA_PILOT_ROW,
                "index": index,
                "donor_row": row_index,
                "donor_decision_frame": row_index * 4,
                "cand": int(cand),
                "bomb_argmax_discarded": int(bomb),
                "cand_teacher_history": int(teacher_cand),
                "bomb_argmax_teacher_history_discarded": int(teacher_bomb),
                "history_agree": bool(int(cand) == int(teacher_cand)),
                "play_sha256": _array_digest(sample["play"]),
                "hud_sha256": _array_digest(sample["hud"]),
                "act_hist_sha256": _array_digest(history.rows()),
                "classification": "measured",
            })
    masks = pilot_table_masks(cands, entry)
    masks_local = pilot_table_masks(cands, 0)
    first = entry_frame + 1
    last = entry_frame + PILOT_WINDOW_FRAMES
    flat = flat_masks_of(masks, first)
    flat_local = flat_masks_of(masks_local, first)
    trace = mask_trace_digest(flat, first, last)
    trace_local = mask_trace_digest(flat_local, first, last)
    phase_diff = sum(1 for left, right in zip(masks, masks_local)
                     if [int(left[0]), int(left[1])] != [int(right[0]), int(right[1])])
    differing = [index for index, (left, right) in enumerate(zip(cands, teacher_cands))
                 if int(left) != int(right)]
    histogram: dict[str, int] = {}
    for value in cands:
        histogram[str(int(value))] = histogram.get(str(int(value)), 0) + 1
    table = {
        "schema": SCHEMA_PILOT_TABLE,
        "created_at": now_iso(),
        "classification": "measured",
        "ticket": PILOT_TICKET,
        "mode": ("smoke" if layout["smoke"] else "formal"),
        "repo": Path(repo).as_posix(),
        "out": Path(layout["out"]).as_posix(),
        "host": host_identity(),
        "heads": git_identity(repo),
        "tool": {"path": TOOL_PATH.as_posix(), "sha256": sha256_file(TOOL_PATH)},
        "final_checkpoint_sha256": final_sha,
        "entry": {
            "decision_index": entry,
            "frame": entry_frame,
            "pulse_period": DEFAULT_PULSE_PERIOD,
            "release_index_base": entry,
            "release_rule": ("release_due(entry_decision + k, pulse_period): the "
                             "global phase the arms continue after the prefix"),
            "classification": "measured",
        },
        "donor": {
            "id": PILOT_DONOR_ROOT_ID,
            "shard": str(donor_row["shard"]),
            "path": repo_rel(Path(donor_row["dir"]), repo),
            "seed": int(donor_row["seed"]),
            "stage": _shard_int(donor_row.get("stage")),
            "decisions": total_rows,
            "labels_sha256": (donor.meta.get("labels") or {}).get("sha256"),
            "play_sha256": (donor.meta.get("obs") or {}).get("play96x112", {}
                                                            ).get("sha256"),
            "hud_sha256": (donor.meta.get("obs") or {}).get("hud", {}
                                                           ).get("sha256"),
            "act_hist_sha256": (donor.meta.get("obs") or {}).get("act_hist", {}
                                                                 ).get("sha256"),
            "obs_spec_digest": donor.meta.get("obs_spec_digest"),
            "classification": "measured",
        },
        "r0": r0,
        "r0_rule": "r0 = F0 // 4 with F0 = the frozen entry frame",
        "length": PILOT_WINDOW_DECISIONS,
        "row_map": ("r(k) = r0 + k // 2: one donor row (four recorded frames) "
                    "feeds two student decisions, because the student decision is "
                    "two frames and the donor grid is four"),
        "obs_hold_decisions": 2,
        "frames": [first, last],
        "boundary_frames": [entry_frame, entry_frame + PILOT_WINDOW_FRAMES - 1],
        "decision_frames": DECISION_FRAMES,
        "pulse_period": DEFAULT_PULSE_PERIOD,
        "history_mode": "self_consistent",
        "history_note": (
            "the frozen table's act_hist is the table's own previous eight "
            "candidates (zero padded at the start), because arm three's history in "
            "the target episode is exactly what it emitted itself"),
        "history_sensitivity": {
            "mode": "donor_teacher_history",
            "note": ("the same pixels with the donor row's recorded act_hist; the "
                     "argmax difference is the free 'fixed picture, swapped "
                     "history' check"),
            "differing_decisions": len(differing),
            "differing_list": differing,
            "cands_teacher_history": teacher_cands,
        },
        "cands": cands,
        "masks": masks,
        "table_sha256": table_sha256(cands),
        "table_sha_definition": 'sha256 of "<i>:<cand>\\n" over i = 0..299 (ascii)',
        "mask_trace_sha256": trace,
        "mask_trace_definition": (
            'sha256 of "<step_frame>:<mask>\\n" over the 600 STEP frames the arm '
            f"executes ({first}..{last}), from commit_frames(cand, bomb=0, "
            "decision_frames=2, release_shoot=release_due(entry_decision + k, 48))"),
        "mask_trace_sha256_local_index": trace_local,
        "mask_phase_differing_decisions": phase_diff,
        "mask_phase_note": (
            "`mask_trace_sha256` is the executable trace (global pulse phase); the "
            "local-index variant is reported so the phase choice is visible, never "
            "silently mixed"),
        "histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        "distinct_candidates": len(histogram),
        "dominant_share": (max(histogram.values()) / len(cands) if cands else 0.0),
        "tail_rule": tail_rule,
        "tail_repeats": tail_repeats,
        "seconds": round(time.perf_counter() - started, 3),
        "rows": rows,
    }
    table_path = base / "table.json"
    write_json(table_path, table)
    with (base / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            write_jsonl_row(handle, row)
    out = dict(table)
    out.pop("rows", None)
    out["path"] = table_path.as_posix()
    out["path_sha256"] = sha256_file(table_path)
    out["rows_path"] = (base / "rows.jsonl").as_posix()
    out["rows_sha256"] = sha256_file(base / "rows.jsonl")
    return out


# -- one arm of one development root ---------------------------------------
def _step_one(attempt: Any, mask: int, frame: int) -> dict[str, Any]:
    """One ``STEP mask 1`` plus the frame's ``EVENTS`` word (frame-exact)."""

    reply = attempt.host.step(int(mask), 1)
    advanced = int(reply.get("advanced", 0) or 0)
    if advanced != 1 or "exit" in reply:
        return {"ok": False, "advanced": advanced,
                "exit": bool("exit" in reply),
                "game_over": int(reply.get("game_over", 0) or 0),
                "frame": int(frame)}
    events = _parse_kv(attempt.host.events_raw())
    if events.get("_verb") != "OK" or "frameEventFlags" not in events:
        raise EngineRefused(
            f"EVENTS reply is unusable at frame {frame}: {str(events)[:160]}")
    if int(events.get("frames", -1)) != int(frame):
        raise FrameRelationConflict(
            f"EVENTS frames={events.get('frames')} but the STEP ended on frame "
            f"{frame}")
    value = reply.get("hash")
    if value is None or not str(value).strip():
        raise FrameRelationConflict(
            f"the STEP into frame {frame} carried no state hash: "
            f"{str(reply)[:160]}")
    return {"ok": True, "advanced": advanced, "frame": int(frame),
            "flags": int(events["frameEventFlags"], 16),
            "hash": str(value).strip().lower(),
            "game_over": int(reply.get("game_over", 0) or 0)}


def _slot_probe(attempt: Any, slot: int) -> dict[str, Any]:
    """One ``HAZARDS`` read for the frozen slot (the prefix identity probe)."""

    hazards = parse_hazards_enemies(attempt.host.client.request("HAZARDS"))
    enemy = next((item for item in hazards["enemies"]
                  if int(item["slot"]) == int(slot)), None)
    return {"enemies": hazards["enemies"], "enemy": enemy,
            "count": hazards["enemies_count"]}


def _pilot_prefix_walk(*, attempt: Any, tape: Mapping[str, Any],
                       entry_decision: int, entry_frame: int, slot: int,
                       buffer: Any, rows_path: Path) -> dict[str, Any]:
    """Replay the frozen prefix from this arm's own RESET, frame by frame.

    One ``STEP mask 1`` per frame (so the frame numbering is exact), ``EVENTS``
    after every frame, and one ``HAZARDS`` probe per decision so the frozen
    slot's generation boundaries are **observed** on this seed instead of assumed.
    Pixels are read only for the last four frames, which is all the obs-v3 stack
    needs at the entry (a model arm never needs the earlier pictures).
    """

    from touhou_rl.v400.student import obs_spec_v3  # noqa: PLC0415

    digest_parts: list[str] = []
    anchors: list[dict[str, Any]] = []
    spawns: list[dict[str, Any]] = []
    previous_present = False
    previous_life: int | None = None
    last_hash: dict[int, str] = {}
    frames = 0
    rows = 0
    pixel_frames = 0
    stack_start = int(entry_frame) - 3
    with Path(rows_path).open("w", encoding="utf-8") as handle:
        for index in range(int(entry_decision)):
            masks = tape["masks"][index]
            frame_start = index * DECISION_FRAMES + 1
            flags: list[int] = []
            hashes: dict[int, str] = {}
            for offset, mask in enumerate(masks):
                frame = frame_start + offset
                stepped = _step_one(attempt, int(mask), frame)
                if not stepped["ok"]:
                    return {"decisions": index, "frames": frames,
                            "hash_at_entry": None, "rows": rows,
                            "shortfall": {"frame": frame,
                                          "advanced": stepped["advanced"],
                                          "exit": stepped["exit"],
                                          "game_over": stepped["game_over"],
                                          "decision_index": index},
                            "anchors": [], "spawns": spawns,
                            "pixel_frames": pixel_frames,
                            "executed_masks": digest_parts}
                frames += 1
                flags.append(int(stepped["flags"]))
                if stepped["hash"] is not None:
                    hashes[frame] = str(stepped["hash"])
                    last_hash[frame] = str(stepped["hash"])
                if buffer is not None and frame >= stack_start:
                    _header, rgba = attempt.host.pixels()
                    play, hud = obs_spec_v3.encode_planes_v3(rgba)
                    buffer.push(play, hud)
                    pixel_frames += 1
            probe = _slot_probe(attempt, slot)
            enemy = probe["enemy"]
            present = enemy is not None
            life = None if enemy is None else int(enemy["life"])
            if present and not previous_present:
                spawns.append({"decision_index": index,
                               "frame": frame_start + DECISION_FRAMES - 1,
                               "life": life,
                               "flags1": str(enemy["flags1"]),
                               "boss": bool(enemy["boss"])})
            elif present and previous_life is not None and life is not None \
                    and int(life) > int(previous_life):
                spawns.append({"decision_index": index,
                               "frame": frame_start + DECISION_FRAMES - 1,
                               "life": life, "flags1": str(enemy["flags1"]),
                               "boss": bool(enemy["boss"]),
                               "kind": "life_jump_up"})
            previous_present = present
            previous_life = life
            digest_parts.append(f"{index}:{int(tape['cands'][index])}:"
                                f"{int(masks[0])}:{int(masks[1])}")
            rows += 1
            write_jsonl_row(handle, {
                "schema": SCHEMA_PILOT_PREFIX_ROW,
                "phase": "prefix",
                "decision_index": index,
                "frame_start": frame_start,
                "frame_end": frame_start + DECISION_FRAMES - 1,
                "cand": int(tape["cands"][index]),
                "masks": [int(masks[0]), int(masks[1])],
                "ev_flags": [f"0x{value:04x}" for value in flags],
                "hash": {"frame_start": hashes.get(frame_start),
                         "frame_end": hashes.get(frame_start + DECISION_FRAMES - 1)},
                "frozen_slot": {"slot": int(slot), "present": present,
                                "life": life,
                                "flags1": None if enemy is None
                                else str(enemy["flags1"])},
                "state_read": False,
                "classification": "measured",
            })
    prefix_digest = sha256_bytes(("\n".join(digest_parts) + "\n").encode("ascii"))
    return {"decisions": int(entry_decision), "frames": frames, "rows": rows,
            "hash_at_entry": last_hash.get(int(entry_frame)),
            "shortfall": None, "anchors": anchors, "spawns": spawns,
            "pixel_frames": pixel_frames,
            "prefix_digest_recomputed": prefix_digest,
            "executed_masks": None}


class _PilotStop(RuntimeError):
    """An arm stopped on a named, non-technical outcome (entry not reached)."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__(str((payload or {}).get("kind")))
        self.payload = dict(payload or {})


def run_pilot_arm(*, args: argparse.Namespace, out_root: Path, repo: Path,
                  card: Mapping[str, Any], tape: Mapping[str, Any],
                  layout: Mapping[str, Any], seed: int, arm: str,
                  model: Any, table: Mapping[str, Any] | None,
                  limits: BudgetLimits, meter: Any) -> dict[str, Any]:
    """One engine start: the frozen prefix from this root's RESET, then 300 decisions."""

    entry = int(card["entry"]["prefix_decisions"])
    entry_frame = int(card["entry"]["frame"])
    decision_cap = int(card["frame_cap"]["decisions"])
    frame_cap = int(card["frame_cap"]["frames"])
    slot = int(card["event"]["seed12_instance"]["slot"])
    card_spawn = int(card["event"]["seed12_instance"]["spawn_decision"])
    card_life = int(card["event"]["seed12_instance"]["life_at_spawn"])
    base = Path(layout["arms"]) / str(seed) / arm
    base.mkdir(parents=True, exist_ok=True)
    pixel_arm = arm in ("init", "final")
    attempt_id = f"03-{'smoke' if layout['smoke'] else 'dev'}-{seed}-{arm}"
    planned = entry_frame + frame_cap
    attempt = EngineAttempt(
        args=args, out_root=out_root, base_dir=base, attempt_id=attempt_id,
        planned_frames=int(planned),
        note=(f"03 arm {arm} on dev root seed {seed}: frozen prefix decisions "
              f"0..{entry - 1} (frames 1..{entry_frame}), then {decision_cap} "
              f"decisions / {frame_cap} frames"),
        limits=limits, meter=meter, reset_seed=int(seed))
    record: dict[str, Any] = {
        "schema": SCHEMA_PILOT_ARM,
        "created_at": now_iso(),
        "classification": layout["classification"],
        "smoke": bool(layout["smoke"]),
        "ticket": PILOT_TICKET,
        "seed": int(seed),
        "root_seed": int(seed),
        "arm": arm,
        "attempt_id": attempt_id,
        "repo": Path(repo).as_posix(),
        "out": Path(out_root).as_posix(),
        "base": base.as_posix(),
        "host": host_identity(),
        "heads": git_identity(repo),
        "tool": {"path": TOOL_PATH.as_posix(), "sha256": sha256_file(TOOL_PATH)},
        "controller": ({"init": "weights_initial.pt",
                        "final": "weights_final.pt",
                        "open_loop": "frozen open-loop table"}[arm]),
        "engine_started": False,
        "entry": {"card_decision": entry, "card_frame": entry_frame,
                  "card_hash": card["entry"]["hash"], "reached": False,
                  "hash": None, "frame": None, "stage": None,
                  "dead": None, "identity_mismatch": None},
        "counts": {"prefix_decisions": 0, "prefix_frames": 0,
                   "control_decisions": 0, "control_frames": 0,
                   "engine_frames": 0, "pixel_frames": 0},
        "terminal": {"kind": "unknown"},
    }
    exit_code = EXIT_TECHNICAL
    closed: dict[str, Any] = {}
    try:
        opened = attempt.open()
        record["engine_started"] = True
        record.update({"preflight": opened["preflight"],
                       "private_game_dir": opened["private"],
                       "reset": opened["reset"],
                       "engine_session": opened["session"],
                       "engine_env": attempt.module.engine_env_record()})
        buffer = None
        history = None
        if pixel_arm:
            from touhou_rl.v400.student import obs_spec_v3  # noqa: PLC0415
            from touhou_rl.v400.student.obs_spec_v2 import (  # noqa: PLC0415
                ActionHistoryBuffer)
            buffer = obs_spec_v3.ObservationBufferV3()
            history = ActionHistoryBuffer()
        prefix = _pilot_prefix_walk(
            attempt=attempt, tape=tape, entry_decision=entry,
            entry_frame=entry_frame, slot=slot, buffer=buffer,
            rows_path=base / "prefix.jsonl")
        record["counts"].update({"prefix_decisions": prefix["decisions"],
                                 "prefix_frames": prefix["frames"],
                                 "pixel_frames": prefix["pixel_frames"]})
        record["prefix"] = {
            "decisions": prefix["decisions"], "frames": prefix["frames"],
            "rows": (base / "prefix.jsonl").as_posix(),
            "prefix_masks_sha256_recomputed": prefix.get(
                "prefix_digest_recomputed"),
            "prefix_masks_sha256_card": card["entry"]["prefix_masks_sha256"],
            "prefix_masks_sha256_matches": bool(
                prefix.get("prefix_digest_recomputed")
                == card["entry"]["prefix_masks_sha256"]),
            "spawn_transitions": prefix["spawns"],
            "shortfall": prefix["shortfall"],
            "classification": "measured",
        }
        if prefix["shortfall"] is not None:
            short = prefix["shortfall"]
            record["terminal"] = {
                "kind": "entry_not_reached",
                "detail": (f"the STEP into frame {short['frame']} returned "
                           f"advanced={short['advanced']} exit={short['exit']} "
                           f"game_over={short['game_over']}"),
                "shortfall": short, "engine_started": True}
            exit_code = EXIT_GATE_FAILED
            raise _PilotStop(record["terminal"])
        entry_reads = read_decision(attempt.host)
        state = engine_state_record(entry_reads["state"])
        hud = hud_record(entry_reads["hud"])
        player = player_record(entry_reads["player"])
        entry_rows = entry_reads["hazards"]["enemies"]
        frozen_rows = [item for item in entry_rows if int(item["slot"]) == slot]
        frozen_row = (max(frozen_rows, key=lambda item: int(item["life"]))
                      if frozen_rows else None)
        spawns = [item for item in prefix["spawns"]
                  if item.get("kind") != "life_jump_up"]
        observed_spawn = int(spawns[-1]["decision_index"]) if spawns else None
        identity_mismatch = (observed_spawn is None
                             or abs(observed_spawn - card_spawn) > 4)
        # "not dead" is a resource predicate, not a playerState one: SPAWNING and
        # INVULNERABLE are alive states and a legitimately live entry may be in
        # either; only a death in progress (or a lost life / game over) fails it.
        dead = bool(state["lives_remaining"] <= 0 or hud["game_over"]
                    or player["player_state_name"] == "DYING")
        record["entry"].update({
            "hash": prefix["hash_at_entry"],
            "frame": int(state["frame"]),
            "stage": int(state["stage"]),
            "dead": dead,
            "reached": bool(state["frame"] == entry_frame and not dead
                            and int(state["stage"]) == 0),
            "frozen_slot_present": frozen_row is not None,
            "frozen_slot_boss": None if frozen_row is None else bool(
                frozen_row["boss"]),
            "frozen_slot_life": None if frozen_row is None else int(
                frozen_row["life"]),
            "frozen_slot_flags1": None if frozen_row is None else str(
                frozen_row["flags1"]),
            "observed_spawn_decision": observed_spawn,
            "card_spawn_decision": card_spawn,
            "spawn_delta": (None if observed_spawn is None
                            else int(observed_spawn) - card_spawn),
            "identity_mismatch": bool(identity_mismatch),
            "classification": "measured"})
        record["resources_at_entry"] = {
            "source": ("this arm's own RESET, at the entry frame "
                       f"{entry_frame}"),
            "state": state, "hud": hud, "player": player,
            "counts": {"bullets": entry_reads["hazards"]["bullets"],
                       "lasers": entry_reads["hazards"]["lasers"],
                       "enemies": entry_reads["hazards"]["enemies_count"]},
            "frame": entry_frame,
            "classification": "measured"}
        if not record["entry"]["reached"]:
            record["terminal"] = {
                "kind": "entry_not_reached",
                "detail": (f"frame {state['frame']} stage {state['stage']} dead "
                           f"{dead}"),
                "engine_started": True}
            exit_code = EXIT_GATE_FAILED
            raise _PilotStop(record["terminal"])
        tracker = FrozenTracker(
            slot=slot,
            spawn_decision=(card_spawn if observed_spawn is None
                            else int(observed_spawn)),
            life_at_spawn=(card_life if frozen_row is None
                           else int(frozen_row["life"])),
            entry_decision=entry)
        deaths_at_entry = float(state["deaths"])
        lives_at_entry = float(state["lives_remaining"])
        event_bits: dict[str, int] = {}
        nonzero_frames: list[list[Any]] = []
        event_frames = 0
        last_row: dict[str, Any] | None = None
        frames_in_window = 0
        outcome: str | None = None
        stop_reason = "decision_cap"
        tracker_outcome: str | None = None
        cand_histogram: dict[str, int] = {}
        with (base / "decisions.jsonl").open("w", encoding="utf-8") as handle:
            for offset in range(decision_cap):
                index = entry + offset
                if arm == "open_loop":
                    if table is None:
                        raise EngineRefused("the open-loop arm needs the frozen table")
                    cand = int(table["cands"][offset])
                    masks = [int(value) for value in table["masks"][offset]]
                    expected = pilot_table_masks([cand], index)[0]
                    if masks != expected:
                        raise EngineRefused(
                            f"the frozen table's masks for decision {index} are "
                            f"{masks}, the executor expands {expected}")
                    inputs: dict[str, Any] = {"source": "frozen table"}
                    move_top: list[list[Any]] = []
                    bomb_argmax = None
                elif arm in ("init", "final"):
                    arrays = buffer.arrays(history)
                    import torch  # noqa: PLC0415
                    device = next(model.parameters()).device
                    play_t = torch.from_numpy(
                        arrays["play"][None]).float().to(device)
                    hud_t = torch.from_numpy(
                        arrays["hud"][None]).float().to(device)
                    hist_t = torch.from_numpy(
                        history.rows()[None]).float().to(device)
                    cand, bomb_argmax = model.act(play_t, hud_t, hist_t)
                    cand = int(cand)
                    masks = pilot_table_masks([cand], index)[0]
                    with torch.no_grad():
                        logits = model(play_t, hud_t, hist_t).move[0]
                        probabilities = torch.softmax(logits, dim=-1)
                        top = torch.topk(probabilities, k=3)
                    move_top = [[int(rank), round(float(prob), 6)]
                                for rank, prob in zip(top.indices.tolist(),
                                                      top.values.tolist())]
                    inputs = {"play_sha256": _array_digest(arrays["play"]),
                              "hud_sha256": _array_digest(arrays["hud"]),
                              "act_hist_sha256": _array_digest(history.rows()),
                              "history_pushes": int(history.pushes),
                              "buffer_frames_seen": int(buffer.frames_seen)}
                else:  # pragma: no cover - the arm names are frozen
                    raise EngineRefused(f"unknown arm {arm!r}")
                frame_start = index * DECISION_FRAMES + 1
                frame_end = frame_start + DECISION_FRAMES - 1
                if frames_in_window + DECISION_FRAMES > frame_cap:
                    stop_reason = "frame_cap"
                    break
                flags: list[int] = []
                hashes: dict[int, str] = {}
                reply: dict[str, Any] = {}
                shortfall = None
                for step_offset, mask in enumerate(masks):
                    frame = frame_start + step_offset
                    stepped = _step_one(attempt, int(mask), frame)
                    if not stepped["ok"]:
                        shortfall = {"frame": frame,
                                     "advanced": stepped["advanced"],
                                     "exit": stepped["exit"],
                                     "game_over": stepped["game_over"]}
                        break
                    frames_in_window += 1
                    flags.append(int(stepped["flags"]))
                    if stepped["hash"] is not None:
                        hashes[frame] = str(stepped["hash"])
                    reply = stepped
                    if pixel_arm:
                        from touhou_rl.v400.student import obs_spec_v3  # noqa
                        _header, rgba = attempt.host.pixels()
                        play_plane, hud_plane = obs_spec_v3.encode_planes_v3(rgba)
                        buffer.push(play_plane, hud_plane)
                        record["counts"]["pixel_frames"] += 1
                    event_frames += 1
                    for name in event_names(int(stepped["flags"])):
                        event_bits[name] = event_bits.get(name, 0) + 1
                    if int(stepped["flags"]):
                        if len(nonzero_frames) < 400:
                            nonzero_frames.append([frame,
                                                   f"0x{int(stepped['flags']):04x}"])
                if shortfall is not None:
                    stop_reason = ("engine_exit" if shortfall["exit"]
                                   else "frame_shortfall")
                    record["terminal"] = {
                        "kind": "technical_conflict" if not shortfall["exit"]
                        else "engine_exit",
                        "detail": shortfall, "engine_started": True}
                    exit_code = (EXIT_TECHNICAL if not shortfall["exit"]
                                 else EXIT_GATE_FAILED)
                    break
                reads = read_decision(attempt.host)
                row: dict[str, Any] = {
                    "schema": SCHEMA_PILOT_ROW,
                    "phase": "control",
                    "arm": arm,
                    "seed": int(seed),
                    "decision_index": index,
                    "window_index": offset,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "cand": int(cand),
                    "action_source": ("frozen_table" if arm == "open_loop"
                                      else f"model:{arm}"),
                    "bomb_argmax_discarded": bomb_argmax,
                    "move_top": move_top,
                    "inputs": inputs,
                    "masks": [int(masks[0]), int(masks[1])],
                    "release_due": bool(adapter()["release_due"](
                        index, DEFAULT_PULSE_PERIOD)),
                    "ev_flags": [f"0x{value:04x}" for value in flags],
                    "ev_flags_dec": [int(value) for value in flags],
                    "ev_names": [event_names(value) for value in flags],
                    "state": engine_state_record(reads["state"]),
                    "hud": hud_record(reads["hud"]),
                    "player": player_record(reads["player"]),
                    "enemies": reads["hazards"]["enemies"],
                    "hazards_counts": {"bullets": reads["hazards"]["bullets"],
                                       "lasers": reads["hazards"]["lasers"],
                                       "enemies": reads["hazards"]["enemies_count"]},
                    "hash": {"frame_start": hashes.get(frame_start),
                             "frame_end": hashes.get(frame_end)},
                    "classification": "measured",
                }
                observed = tracker.observe(row)
                if observed is not None and tracker_outcome is None:
                    tracker_outcome = observed
                row["frozen"] = {
                    "slot": tracker.slot,
                    "present": any(int(item["slot"]) == tracker.slot
                                   for item in reads["hazards"]["enemies"]),
                    "life": tracker.last_life,
                    "outcome_so_far": tracker_outcome,
                    "phase_ends_so_far": len(tracker.phase_ends),
                }
                write_jsonl_row(handle, row)
                last_row = row
                cand_histogram[str(int(cand))] = cand_histogram.get(
                    str(int(cand)), 0) + 1
                record["counts"]["control_decisions"] += 1
                record["counts"]["control_frames"] += DECISION_FRAMES
                game_over = int(reads["hud"].get("game_over", 0) or 0) \
                    or int(reply.get("game_over", 0) or 0)
                deaths = float(reads["state"]["deaths"])
                # the card's termination priority, highest first
                if game_over:
                    outcome, stop_reason = "game_over", "game_over"
                    break
                if deaths > deaths_at_entry:
                    outcome, stop_reason = "legal_death", "legal_death"
                    break
                if tracker_outcome is not None:
                    outcome, stop_reason = tracker_outcome, "frozen_consequence"
                    break
            else:
                stop_reason = "decision_cap"
        if history is not None:
            record["history_at_end"] = {"pushes": int(history.pushes),
                                        "rows": history.rows().tolist()}
        record["counts"]["engine_frames"] = (
            prefix["frames"] + frames_in_window)
        attempt.engine_frames = record["counts"]["engine_frames"]
        record["window"] = {
            "decisions": int(record["counts"]["control_decisions"]),
            "frames": int(frames_in_window),
            "frame_range": [entry_frame + 1, entry_frame + frames_in_window],
            "decision_range": [entry, entry + max(
                0, record["counts"]["control_decisions"]) - 1],
            "decision_cap": decision_cap, "frame_cap": frame_cap,
            "last_decision": (None if last_row is None
                              else int(last_row["decision_index"])),
            "last_frame": (None if last_row is None else int(last_row["frame_end"])),
            "classification": "measured"}
        record["decisions_path"] = (base / "decisions.jsonl").as_posix()
        record["decisions_sha256"] = sha256_file(base / "decisions.jsonl")
        record["events_summary"] = {
            "frames": int(event_frames),
            "frames_with_any_bit": len(nonzero_frames),
            "by_name": dict(sorted(event_bits.items())),
            "nonzero_frames_sample": nonzero_frames[:400],
            "classification": "measured"}
        record["resources_at_end"] = (
            None if last_row is None else {
                "frame": int(last_row["frame_end"]),
                "state": dict(last_row["state"]), "hud": dict(last_row["hud"]),
                "player": dict(last_row["player"]),
                "counts": dict(last_row["hazards_counts"]),
                "classification": "measured"})
        record["deaths_observed"] = (0.0 if last_row is None else
                                     float(last_row["state"]["deaths"])
                                     - deaths_at_entry)
        record["lives_lost"] = (0.0 if last_row is None else
                                lives_at_entry
                                - float(last_row["state"]["lives_remaining"]))
        record["life_lost"] = record["lives_lost"]
        record["frozen_generation"] = tracker.summary()
        record["frozen_generation"]["expected_identity"] = {
            "slot": slot, "card_spawn_decision": card_spawn,
            "card_life_at_spawn": card_life,
            "observed_spawn_decision": observed_spawn,
            "identity_mismatch": bool(identity_mismatch)}
        if outcome is None:
            outcome = ("not_removed_within_cap" if tracker.seen
                       else "generation_not_observed")
        record["outcome"] = outcome
        record["tracker_outcome"] = tracker_outcome
        record["outcome_decision"] = tracker.outcome_decision
        record["outcome_frame"] = tracker.outcome_frame
        record["stop_reason"] = stop_reason
        record["candidate_histogram"] = dict(
            sorted(cand_histogram.items(), key=lambda item: int(item[0])))
        record["candidate_dominant_share"] = (
            max(cand_histogram.values()) / sum(cand_histogram.values())
            if cand_histogram else 0.0)
        record["terminal"] = {
            "kind": ("frozen_consequence" if outcome in OUTCOME_VOCABULARY
                     and outcome not in ("legal_death", "game_over")
                     else outcome),
            "outcome": outcome, "stop_reason": stop_reason,
            "window_frames": int(frames_in_window),
            "last_decision": (None if last_row is None
                              else int(last_row["decision_index"])),
            "game_over": (0 if last_row is None
                          else int(last_row["hud"]["game_over"])),
            "stage": (None if last_row is None
                      else int(last_row["state"]["stage"])),
            "classification": "measured"}
        exit_code = EXIT_OK
    except _PilotStop as stop:
        record["terminal"] = dict(stop.payload)
    except (EngineRefused, FrameRelationConflict, SourceError) as exc:
        record["terminal"] = {"kind": "technical_conflict",
                              "reason": type(exc).__name__,
                              "detail": str(exc),
                              "engine_started": bool(record.get("engine_started"))}
        exit_code = EXIT_TECHNICAL
    except Exception as exc:  # noqa: BLE001
        record["terminal"] = {"kind": "technical_conflict",
                              "reason": type(exc).__name__,
                              "detail": str(exc),
                              "traceback": traceback.format_exc(),
                              "engine_started": bool(record.get("engine_started"))}
        exit_code = EXIT_TECHNICAL
    finally:
        if attempt.engine_started or (attempt.reservation is not None
                                      and attempt.settlement is None):
            try:
                attempt.engine_frames = int(
                    record["counts"].get("engine_frames")
                    or record["counts"].get("prefix_frames", 0))
                closed = attempt.close(result={
                    "command": f"03-arm-{arm}",
                    "seed": int(seed),
                    "terminal": (record.get("terminal") or {}).get("kind"),
                    "outcome": record.get("outcome"),
                    "reached_frame": attempt.engine_frames})
                record["mute"] = closed["mute"]
                record["source_dir"] = closed["source_dir"]
                record["ledger"] = closed["ledger"]
                record["measured"] = closed["measured"]
            except Exception as exc:  # noqa: BLE001
                record["close_error"] = f"{type(exc).__name__}: {exc}"
    record["arm_exit_code"] = int(exit_code)
    record["finished_at"] = now_iso()
    path = write_json(base / "result.json", record)
    record["result_path"] = path.as_posix()
    record["result_sha256"] = sha256_file(path)
    return record


# -- the pilot driver -------------------------------------------------------
def _arm_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """The per-arm numbers the comparison and the receipt quote."""

    entry = record.get("entry") or {}
    counts = record.get("counts") or {}
    end = record.get("resources_at_end") or {}
    frozen = record.get("frozen_generation") or {}
    return {
        "arm": record.get("arm"),
        "seed": record.get("seed"),
        "engine_started": bool(record.get("engine_started")),
        "entry_reached": bool(entry.get("reached")),
        "entry_frame": entry.get("frame"),
        "entry_hash": entry.get("hash"),
        "entry_stage": entry.get("stage"),
        "entry_dead": entry.get("dead"),
        "identity_mismatch": entry.get("identity_mismatch"),
        "observed_spawn_decision": entry.get("observed_spawn_decision"),
        "frozen_slot_present": entry.get("frozen_slot_present"),
        "frozen_slot_life_at_entry": entry.get("frozen_slot_life"),
        "prefix_frames": counts.get("prefix_frames"),
        "control_decisions": counts.get("control_decisions"),
        "control_frames": counts.get("control_frames"),
        "engine_frames": counts.get("engine_frames"),
        "outcome": record.get("outcome"),
        "stop_reason": record.get("stop_reason"),
        "outcome_frame": record.get("outcome_frame"),
        "deaths_observed": record.get("deaths_observed"),
        "lives_lost": record.get("lives_lost"),
        "lives_at_entry": ((record.get("resources_at_entry") or {}).get(
            "state") or {}).get("lives_remaining"),
        "lives_at_end": (end.get("state") or {}).get("lives_remaining"),
        "score_at_end": (end.get("state") or {}).get("score"),
        "frozen_life_first": frozen.get("life_first"),
        "frozen_life_last": frozen.get("life_last"),
        "frozen_life_min": frozen.get("life_min"),
        "frozen_observations": frozen.get("observations"),
        "terminal": (record.get("terminal") or {}).get("kind"),
        "candidate_dominant_share": record.get("candidate_dominant_share"),
        "candidate_histogram": record.get("candidate_histogram"),
        "exit_code": record.get("arm_exit_code"),
        "result": record.get("result_path"),
        "result_sha256": record.get("result_sha256"),
        "mute_problems": ((record.get("mute") or {}).get("after_exit_problems")),
        "ledger": record.get("ledger"),
        "classification": record.get("classification"),
    }


def pilot_comparison(*, arms: Mapping[str, Mapping[str, Mapping[str, Any]]],
                     smoke: bool) -> dict[str, Any]:
    """The per-root table and the ticket's nomination rule.

    Completion is "the frozen generation was removed with a ``kill_*`` outcome
    inside the frozen window".  Two contrasts are formed per root — final minus
    initial ("did the fit learn anything") and final minus the strong open-loop
    table ("is the closed loop better than the frozen table") — and both are
    averaged with **equal weight per root**.
    """

    roots = sorted(arms)
    per_root: dict[str, Any] = {}
    for seed in roots:
        bucket = arms[seed]
        scores: dict[str, Any] = {}
        for arm in PILOT_ARMS:
            summary = bucket.get(arm) or {}
            outcome = summary.get("outcome")
            completed = outcome in KILL_OUTCOMES
            scores[arm] = {
                "completed": bool(completed),
                "outcome": outcome,
                "stop_reason": summary.get("stop_reason"),
                "technical": (summary.get("terminal") == "technical_conflict"
                              or summary.get("exit_code") not in (0, 1)),
                "entry_reached": summary.get("entry_reached"),
                "control_decisions": summary.get("control_decisions"),
                "lives_lost": summary.get("lives_lost"),
                "frozen_life_first": summary.get("frozen_life_first"),
                "frozen_life_last": summary.get("frozen_life_last"),
                "frozen_life_min": summary.get("frozen_life_min"),
                "frozen_observations": summary.get("frozen_observations"),
            }
        present = [arm for arm in PILOT_ARMS if arm in bucket]
        complete = len(present) == len(PILOT_ARMS) and all(
            scores[arm]["entry_reached"] and not scores[arm]["technical"]
            for arm in PILOT_ARMS)
        learned_before = (int(scores["final"]["completed"])
                          - int(scores["init"]["completed"]))
        learned_strong = (int(scores["final"]["completed"])
                          - int(scores["open_loop"]["completed"]))
        per_root[str(seed)] = {
            "seed": int(seed),
            "arms_present": present,
            "complete": bool(complete),
            "arms": scores,
            "learned_minus_before": learned_before,
            "learned_minus_strong": learned_strong,
            "completion_rate": {arm: float(int(scores[arm]["completed"]))
                                for arm in PILOT_ARMS},
            "classification": "measured",
        }
    means = {}
    positives = {}
    for contrast, key in (("learned_minus_before", "learned_minus_before"),
                          ("learned_minus_strong", "learned_minus_strong")):
        values = [per_root[seed][key] for seed in per_root]
        means[contrast] = (sum(values) / len(values)) if values else None
        positives[contrast] = [int(seed) for seed in per_root
                               if per_root[seed][key] > 0]
    complete_roots = [seed for seed in per_root if per_root[seed]["complete"]]
    nominate = bool(
        not smoke and len(per_root) == len(PILOT_DEV_SEEDS)
        and len(complete_roots) == len(PILOT_DEV_SEEDS)
        and means["learned_minus_before"] is not None
        and means["learned_minus_before"] > 0
        and means["learned_minus_strong"] > 0
        and len(positives["learned_minus_before"]) >= 2
        and len(positives["learned_minus_strong"]) >= 2)
    dominant = {
        str(seed): {arm: (arms[seed].get(arm) or {}).get("candidate_dominant_share")
                    for arm in PILOT_ARMS if arm in arms[seed]}
        for seed in roots}
    return {
        "per_root": per_root,
        "equal_weight_means": means,
        "positive_roots": positives,
        "complete_roots": complete_roots,
        "expected_roots": [int(seed) for seed in PILOT_DEV_SEEDS],
        "nominate_expansion": nominate,
        "rule": ("nomination needs both equal-weight per-root differences > 0 and "
                 "at least two roots positive for each; fewer roots or any "
                 "technical loss forbids nomination"),
        "smoke": bool(smoke),
        "candidate_dominant_share": dominant,
        "classification": ("technical_smoke_not_a_pilot_result" if smoke
                           else "measured"),
    }


def pilot_next_diagnostic(*, comparison: Mapping[str, Any],
                          train: Mapping[str, Any],
                          table: Mapping[str, Any]) -> dict[str, Any]:
    """The ticket's three first checks when the contrasts do not separate.

    "先查动作是否近恒定、真实可见先兆和标签覆盖" (``TRAINING_HOST_REQUEST_02``
    line 52): the action histogram of every arm, whether a visible precursor is
    available (not evaluated here: it needs a per-decision frame review) and the
    label coverage of the fit root.
    """

    shares = comparison.get("candidate_dominant_share") or {}
    worst = None
    for seed, arms in shares.items():
        for arm, share in arms.items():
            if share is None:
                continue
            if worst is None or float(share) > float(worst["share"]):
                worst = {"seed": seed, "arm": arm, "share": float(share)}
    return {
        "action_near_constant": {
            "evaluated": bool(shares),
            "worst_dominant_share": worst,
            "threshold_note": ("a share above ~0.9 over 300 decisions is the "
                               "'near constant action' signature; the raw "
                               "histograms are in every arm's result.json"),
            "histograms": shares,
        },
        "visible_precursor": {
            "evaluated": False,
            "reason": ("needs a per-decision frame review of the arm logs (the "
                       "decisions.jsonl rows carry the masks and the frame range, "
                       "the pictures do not); not attempted in this stage"),
        },
        "label_coverage": {
            "fit_root_classes": (train.get("fit_shard") or {}).get(
                "distinct_classes"),
            "fit_root_histogram": (train.get("fit_shard") or {}).get(
                "chosen_histogram"),
            "open_loop_distinct": table.get("distinct_candidates"),
            "note": ("the fit root covers 17 of the 18 classes; a class with no "
                     "label can only be learned as a negative"),
        },
    }


def cmd_pilot(args: argparse.Namespace) -> int:
    """03 pilot: freeze the registration, fit once, freeze the table, run the arms.

    Order matters and is enforced: the registration file is written (or verified)
    first, the fit second, the open-loop table third — and only then may an engine
    start.  ``--only-seed 1000`` is the **technical smoke**: everything lands under
    ``03/smoke/`` and every artefact is marked
    ``technical_smoke_not_a_pilot_result``.
    """

    if getattr(args, "strength_seed", None):
        # One additive entry point: `pilot --strength-seed 13` is exactly
        # `strength --seed 13`.  Nothing below this line runs when the flag is
        # absent, so the pilot's own path (fit, table, four roots) is untouched.
        forwarded = argparse.Namespace(
            **{**vars(args), "seed": int(args.strength_seed)})
        return cmd_strength(forwarded)
    module = slice_tool()
    maybe_reexec_under_xvfb(enabled=not bool(getattr(args, "no_xvfb", False)))
    started = time.perf_counter()
    out_root = Path(args.out).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    only = [int(value) for value in (getattr(args, "only_seed", None) or [])]
    smoke = bool(only == [PILOT_SMOKE_SEED] or getattr(args, "smoke", False))
    if smoke and only and only != [PILOT_SMOKE_SEED]:
        print("pilot: --smoke only pairs with --only-seed 1000")
        return EXIT_NOT_IMPLEMENTED
    seeds = only if only else list(PILOT_DEV_SEEDS)
    if smoke and not only:
        seeds = [PILOT_SMOKE_SEED]
    layout = pilot_layout(out_root, smoke=smoke)
    for path in (layout["root"], layout["base"], layout["train"],
                 layout["open_loop"], layout["arms"]):
        Path(path).mkdir(parents=True, exist_ok=True)
    gate = Gate("03_pilot")
    meter = module.CostMeter()
    meter.begin()
    record: dict[str, Any] = {
        "schema": SCHEMA_PILOT,
        "generated_by": "tools/rl/v8_31_followup.py pilot",
        "created_at": now_iso(),
        "classification": layout["classification"],
        "smoke": bool(smoke),
        "ticket": PILOT_TICKET,
        "repo": repo.as_posix(),
        "out": out_root.as_posix(),
        "argv": list(getattr(args, "argv", []) or []),
        "host": host_identity(),
        "heads": git_identity(repo),
        "tool": {"path": TOOL_PATH.as_posix(), "sha256": sha256_file(TOOL_PATH)},
        "slice_helper": slice_identity(),
        "layout": {key: Path(value).as_posix() if isinstance(value, Path) else value
                   for key, value in layout.items()},
        "seeds": [int(seed) for seed in seeds],
        "arms": {},
        "engine_starts": 0,
        "started_at": now_iso(),
    }
    exit_code = EXIT_TECHNICAL
    try:
        frozen = read_task_card(out_root / "02b")
        card = dict(frozen["card"])
        card["_card_path"] = frozen["path"]
        record["task_card"] = {
            "path": frozen["path"].as_posix(), "sha256": frozen["sha256"],
            "content_sha256": frozen["content_sha256"],
            "content_sha256_recomputed": frozen["content_sha256_recomputed"],
            "task_id": card.get("task_id"),
            "entry_decision": int(card["entry"]["prefix_decisions"]),
            "entry_frame": int(card["entry"]["frame"]),
        }
        gate.add("task_card_read_back", frozen["content_sha256_matches"],
                 record["task_card"])
        # 1. the frozen registration, before anything else runs
        config_path = Path(getattr(args, "config", None)
                           or (repo / PILOT_CONFIG_REL))
        if config_path.is_file():
            config_identity = load_pilot_roots_config(config_path)
            config_identity["written"] = False
        else:
            payload = pilot_roots_payload(repo=repo, card=card,
                                          created_at=now_iso())
            config_identity = write_pilot_roots_config(config_path, payload)
        roots_config = {
            "path": Path(config_identity["path"]).as_posix(),
            "sha256": config_identity["sha256"],
            "bytes": Path(config_identity["path"]).stat().st_size,
            "content_sha256": config_identity["content_sha256"],
            "content_sha256_recomputed": config_identity[
                "content_sha256_recomputed"],
            "written_this_invocation": bool(config_identity.get("written")),
            "status": (config_identity.get("payload") or {}).get("status"),
            "manifest_id": (config_identity.get("payload") or {}).get(
                "manifest_id"),
            "read_only_after_freeze": True,
            "note": ("the registration is one JSON file; its byte sha256 lives "
                     "here in the evidence, not in a companion file"),
            "classification": "measured",
        }
        # `roots_config` is the field the receipt quotes; `config` is kept as an
        # alias for readers of the first smoke evidence.
        record["roots_config"] = roots_config
        record["config"] = dict(roots_config)
        gate.add("pilot_registration_frozen",
                 record["roots_config"]["status"] == "frozen"
                 and bool(record["roots_config"]["sha256"]),
                 {key: record["roots_config"][key] for key in
                  ("path", "sha256", "content_sha256", "written_this_invocation")})
        gate.add("pilot_registration_binds_the_dev_seeds",
                 len((config_identity.get("payload") or {}).get("roots") or [])
                 == len(PILOT_DEV_SEEDS),
                 {"roots": [entry.get("seed") for entry in
                            (config_identity.get("payload") or {}).get("roots") or []]})
        config_payload = config_identity.get("payload") or {}
        record["roots_config"]["training_roots"] = config_payload.get(
            "training_roots")
        record["config"] = dict(record["roots_config"])
        # 2. one fit
        with meter.measure("io"):
            tape = tape_schedule(repo)
        record["tape"] = {
            "shard": tape["shard"], "sha256": tape["sha256"],
            "decisions": tape["decisions"],
            "prefix_masks_sha256": tape_digest(
                tape, 0, int(card["entry"]["prefix_decisions"])),
            "prefix_masks_sha256_card": card["entry"]["prefix_masks_sha256"],
            "classification": "offline"}
        gate.add("frozen_prefix_matches_the_tape",
                 record["tape"]["prefix_masks_sha256"]
                 == card["entry"]["prefix_masks_sha256"],
                 {"recomputed": record["tape"]["prefix_masks_sha256"],
                  "card": card["entry"]["prefix_masks_sha256"]})
        stage = stage_budget(
            out_root, need_starts=len(seeds) * len(PILOT_ARMS),
            need_frames=len(seeds) * len(PILOT_ARMS)
            * (int(card["entry"]["frame"]) + int(card["frame_cap"]["frames"])),
            label=("03-pilot-smoke" if smoke else "03-pilot"),
            stage_cap={"starts": getattr(args, "max_starts", None),
                       "engine_frames": getattr(args, "max_engine_frames", None)},
            stage_prefixes=PILOT_ATTEMPT_PREFIXES)
        record["budget"] = stage["report"]
        record["budget"]["ledger_cumulative_at_start"] = stage["report"][
            "cumulative"]
        gate.add("budget_checked_from_the_existing_ledger",
                 not stage["report"]["exceeded"],
                 {"ledger_rows_read": stage["report"]["ledger_rows_read"],
                  "cumulative": stage["report"]["cumulative"],
                  "stage_used": stage["report"]["stage_used"],
                  "stage_remaining": stage["report"]["stage_remaining"],
                  "need": stage["report"]["need"]})
        train = pilot_train(repo=repo, layout=layout, card=card,
                            config_identity=config_identity, meter=meter)
        record["train"] = {key: value for key, value in train.items()
                           if key not in ("loss",) and not key.startswith("_")}
        record["train"]["loss"] = {key: value for key, value in train["loss"].items()
                                   if key != "series"}
        record["train"]["loss_series"] = train["loss"]["series"]
        gate.add("one_fit_ran",
                 int(train["updates"]) == PILOT_UPDATES
                 and int(train["samples_seen"]) == PILOT_UPDATES * PILOT_BATCH,
                 {"updates": train["updates"], "batch": train["batch"],
                  "samples_seen": train["samples_seen"],
                  "distinct_decisions": train["distinct_decisions"]})
        gate.add("initial_weights_are_the_seed0_build",
                 bool(train["initial_rebuild_bitwise_identical"]),
                 {"max_abs_diff": train["initial_rebuild_max_abs_diff"]})
        gate.add("two_weight_digests_recorded",
                 bool(train["weights_initial"]["sha256"])
                 and bool(train["weights_final"]["sha256"])
                 and train["weights_initial"]["sha256"]
                 != train["weights_final"]["sha256"],
                 {"initial": train["weights_initial"]["sha256"],
                  "final": train["weights_final"]["sha256"]})
        # 3. the frozen open-loop table, before the first development root
        import torch  # noqa: PLC0415
        from touhou_rl.v400.student import model_v3  # noqa: PLC0415
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models: dict[str, Any] = {}
        if getattr(args, "reuse", False) and (layout["open_loop"] / "table.json"
                                              ).is_file():
            existing = _load_json(layout["open_loop"] / "table.json")
            if table_sha256(existing.get("cands") or []) == existing.get(
                    "table_sha256"):
                table = {**existing,
                         "path": (layout["open_loop"] / "table.json").as_posix(),
                         "rows_path": (layout["open_loop"] / "rows.jsonl"
                                       ).as_posix()}
                record["table_reused"] = True
            else:
                table = None
        else:
            table = None
        if table is None:
            final_model, _payload = model_v3.load_checkpoint_v3(
                train["weights_final"]["path"], device=device)
            models["final"] = final_model
            table = build_open_loop_table(
                repo=repo, layout=layout, card=card, final_model=final_model,
                final_sha=train["weights_final"]["sha256"], meter=meter)
            record["table_reused"] = False
        record["open_loop"] = {key: value for key, value in table.items()
                               if key not in ("cands", "masks",
                                              "history_sensitivity", "rows")}
        record["open_loop"]["cands"] = list(table["cands"])
        record["open_loop"]["masks"] = [list(pair) for pair in table["masks"]]
        record["open_loop"]["history_sensitivity"] = {
            key: value for key, value in table["history_sensitivity"].items()
            if key != "cands_teacher_history"}
        gate.add("open_loop_table_frozen_before_any_dev_root",
                 bool(table["table_sha256"]) and bool(table["mask_trace_sha256"]),
                 {"table_sha256": table["table_sha256"],
                  "mask_trace_sha256": table["mask_trace_sha256"],
                  "length": table["length"], "r0": table["r0"],
                  "history_mode": table["history_mode"],
                  "differing_history_decisions": table["history_sensitivity"][
                      "differing_decisions"]})
        # 4. the arms, one engine start each
        for seed in seeds:
            per_seed = record["arms"].setdefault(str(int(seed)), {})
            for arm in PILOT_ARMS:
                if arm in ("init", "final") and arm not in models:
                    models[arm], _payload = model_v3.load_checkpoint_v3(
                        train[f"weights_{'initial' if arm == 'init' else 'final'}"
                             ]["path"], device=device)
                arm_record = run_pilot_arm(
                    args=args, out_root=out_root, repo=repo, card=card,
                    tape=tape, layout=layout, seed=int(seed), arm=arm,
                    model=models.get(arm), table=(table if arm == "open_loop"
                                                  else None),
                    limits=stage["limits"], meter=meter)
                per_seed[arm] = _arm_summary(arm_record)
                record["engine_starts"] = int(record["engine_starts"]) + int(
                    bool(arm_record.get("engine_started")))
                print(f"  03 seed {seed} arm {arm}: "
                      f"entry={per_seed[arm]['entry_reached']} "
                      f"frames={per_seed[arm]['engine_frames']} "
                      f"outcome={per_seed[arm]['outcome']} "
                      f"stop={per_seed[arm]['stop_reason']} "
                      f"rc={per_seed[arm]['exit_code']}", flush=True)
                if not arm_record.get("engine_started"):
                    break
        # 5. the result
        comparison = pilot_comparison(arms=record["arms"], smoke=smoke)
        record["comparison"] = comparison
        record["next_diagnostic"] = pilot_next_diagnostic(
            comparison=comparison, train=train, table=table)
        record["smoke_ok"] = None
        if smoke:
            arms_done = [arm for seed in record["arms"]
                         for arm in record["arms"][seed]]
            entries = [record["arms"][seed][arm]["entry_reached"]
                       for seed in record["arms"] for arm in record["arms"][seed]]
            technical = [f"{seed}:{arm}"
                         for seed in record["arms"]
                         for arm, value in record["arms"][seed].items()
                         if value["terminal"] == "technical_conflict"
                         or value["exit_code"] not in (0, 1)]
            hashes = {record["arms"][seed][arm]["entry_hash"]
                      for seed in record["arms"] for arm in record["arms"][seed]}
            smoke_ok = bool(len(arms_done) == len(PILOT_ARMS)
                            and all(entries) and not technical
                            and len(hashes) == 1 and None not in hashes)
            record["smoke_ok"] = smoke_ok
            gate.add("smoke_three_arms_completed",
                     len(arms_done) == len(PILOT_ARMS), {"arms": arms_done})
            gate.add("smoke_entry_reached_by_every_arm", all(entries),
                     {"entry_reached": entries})
            gate.add("smoke_no_technical_conflict", not technical,
                     {"technical": technical})
            gate.add("smoke_entry_hashes_agree", len(hashes) == 1
                     and None not in hashes, {"entry_hashes": sorted(
                         str(value) for value in hashes)})
            gate.add("smoke_is_not_a_pilot_result", True,
                     {"classification": PILOT_SMOKE_CLASSIFICATION,
                      "smoke_ok": smoke_ok,
                      "note": ("the pilot gate (nomination) is not evaluated in a "
                               "technical smoke")})
            exit_code = EXIT_OK if smoke_ok else EXIT_GATE_FAILED
        else:
            gate.add("all_expected_roots_complete",
                     len(comparison["complete_roots"]) == len(PILOT_DEV_SEEDS),
                     {"complete_roots": comparison["complete_roots"],
                      "expected": comparison["expected_roots"]})
            gate.add("both_contrasts_positive_per_root_weighted",
                     (comparison["equal_weight_means"]["learned_minus_before"]
                      or 0) > 0
                     and (comparison["equal_weight_means"]["learned_minus_strong"]
                          or 0) > 0,
                     comparison["equal_weight_means"])
            gate.add("at_least_two_roots_positive_each",
                     len(comparison["positive_roots"]["learned_minus_before"]) >= 2
                     and len(comparison["positive_roots"]["learned_minus_strong"]) >= 2,
                     comparison["positive_roots"])
            gate.add("nomination_rule_applied",
                     comparison["nominate_expansion"] is not None, comparison[
                         "nominate_expansion"])
            record["nominate_expansion"] = comparison["nominate_expansion"]
            exit_code = EXIT_OK if comparison["nominate_expansion"] else (
                EXIT_GATE_FAILED)
        record["verdict"] = ("smoke_ok" if smoke and record["smoke_ok"]
                             else ("smoke_failed" if smoke
                                   else ("nominate" if comparison[
                                       "nominate_expansion"] else "no_nomination")))
    except BudgetRefused as exc:
        record["budget"] = dict(exc.payload)
        record["verdict"] = "budget_refused"
        record["terminal"] = {"kind": "budget_refused", "reason": exc.reason,
                              "detail": exc.payload.get("exceeded")}
        gate.add("budget_checked", False, exc.payload.get("exceeded"))
        exit_code = EXIT_NOT_IMPLEMENTED
    except (EngineRefused, SourceError) as exc:
        record["verdict"] = "technical_invalid"
        record["terminal"] = {"kind": "technical_invalid",
                              "reason": type(exc).__name__, "detail": str(exc),
                              "engine_started": bool(record.get("engine_starts"))}
        gate.add("pilot_inputs_readable", False, {"error": str(exc)})
        exit_code = EXIT_TECHNICAL
    except AdapterUnavailable as exc:
        record["verdict"] = "technical_invalid"
        record["terminal"] = {"kind": "technical_invalid",
                              "reason": type(exc).__name__, "detail": str(exc)}
        gate.add("adapter_available", False, {"error": str(exc)})
        exit_code = EXIT_TECHNICAL
    except Exception as exc:  # noqa: BLE001
        record["verdict"] = "technical_invalid"
        record["terminal"] = {"kind": "technical_invalid",
                              "reason": type(exc).__name__, "detail": str(exc),
                              "traceback": traceback.format_exc(),
                              "engine_started": bool(record.get("engine_starts"))}
        gate.add("pilot_completed", False, {"error": str(exc)})
        exit_code = EXIT_TECHNICAL
    finally:
        meter.finish()
    record.setdefault("terminal", {"kind": "unknown"})
    record["finished_at"] = now_iso()
    record["cost"] = meter.payload(step_units=0)
    record["seconds"] = round(time.perf_counter() - started, 3)
    record["gate"] = gate.as_dict()
    target = Path(layout["summary"])
    if target.is_file() and int(exit_code) != EXIT_OK:
        # A failing invocation must never replace a passing summary: a re-run of
        # this one-shot smoke is refused by the 3-start/11,200-frame cap, and that
        # refusal belongs in its own file, not on top of the evidence.
        try:
            previous = _load_json(target)
        except (OSError, ValueError):  # pragma: no cover - corrupt previous file
            previous = {}
        if bool((previous.get("gate") or {}).get("passed")):
            alternate = target.with_name(
                f"{target.stem}.{record.get('verdict') or 'failed'}-"
                f"{datetime.now().strftime('%Y%m%dT%H%M%S')}{target.suffix}")
            record["preserved_evidence"] = {
                "kept": target.as_posix(),
                "previous_gate_passed": True,
                "this_run_written_to": alternate.as_posix(),
                "reason": "a failing invocation never overwrites a passing summary",
                "classification": "measured",
            }
            target = alternate
    path = write_json(target, record)
    for check in gate.checks:
        print(f"  {'ok  ' if check.ok else 'FAIL'} {check.name}")
    print(f"pilot: mode={'smoke' if smoke else 'formal'} "
          f"starts={record.get('engine_starts')} verdict={record.get('verdict')} "
          f"rc={exit_code}")
    print(f"wrote: {path}")
    return exit_code


def verify_03(out_root: str | Path, repo: str | Path, gate: Gate) -> dict[str, Any]:
    """Re-derive 03 from its raw files: cfg, weights, train, table, arms, gate."""

    out_root = Path(out_root)
    report: dict[str, Any] = {"status": "technical_invalid", "checks": 0}
    layouts = [pilot_layout(out_root, smoke=False), pilot_layout(out_root, smoke=True)]
    chosen = None
    for candidate in layouts:
        if Path(candidate["summary"]).is_file():
            chosen = candidate
            break
    if chosen is None:
        raise SourceError(
            f"no 03 pilot evidence under {out_root / PILOT_WINDOW_KEY}")
    summary = _load_json(chosen["summary"])
    layout = {key: value for key, value in (summary.get("layout") or {}).items()}
    smoke = bool(summary.get("smoke"))
    report.update({"smoke": smoke, "summary": Path(chosen["summary"]).as_posix(),
                   "summary_sha256": sha256_file(chosen["summary"]),
                   "verdict_claimed": summary.get("verdict"),
                   "classification": summary.get("classification")})
    gate.add("03_summary_present", bool(summary.get("schema") == SCHEMA_PILOT),
             {"schema": summary.get("schema"),
              "classification": summary.get("classification")})
    # 1. the frozen registration
    config = summary.get("roots_config") or summary.get("config") or {}
    config_path = Path(config.get("path") or "")
    if not config_path.is_file():
        gate.add("03_registration_present", False, {"path": str(config_path)})
        report["status"] = "failed"
        return report
    identity = load_pilot_roots_config(config_path)
    gate.add("03_registration_sha_matches",
             identity["sha256"] == config.get("sha256")
             and identity["content_sha256"] == config.get("content_sha256"),
             {"file_sha256": identity["sha256"],
              "recorded": config.get("sha256"),
              "content_sha256": identity["content_sha256"]})
    gate.add("03_registration_is_frozen",
             (identity.get("payload") or {}).get("status") == "frozen",
             (identity.get("payload") or {}).get("status"))
    companions = sorted(path.name for path in
                        config_path.parent.glob(f"{config_path.stem}.*")
                        if path.name != config_path.name)
    gate.add("03_registration_has_no_companion_file", not companions,
             {"dir": config_path.parent.as_posix(),
              "companions": companions,
              "note": ("the ticket's write scope allows exactly one new config "
                       "file; its sha256 is recorded in this evidence instead")})
    payload_config = identity.get("payload") or {}
    gate.add("03_registration_binds_four_dev_roots",
             [entry.get("seed") for entry in payload_config.get("roots") or []]
             == [int(seed) for seed in PILOT_DEV_SEEDS],
             {"roots": [entry.get("seed") for entry in
                        payload_config.get("roots") or []]})
    # 2. the two weights
    train_claimed = summary.get("train") or {}
    weights: dict[str, Any] = {}
    for name in ("initial", "final"):
        block = train_claimed.get(f"weights_{name}") or {}
        path = Path(block.get("path") or "")
        actual = sha256_file(path) if path.is_file() else None
        weights[name] = {"path": path.as_posix(), "claimed": block.get("sha256"),
                         "actual": actual, "exists": path.is_file()}
        gate.add(f"03_weights_{name}_sha_matches",
                 bool(actual) and actual == block.get("sha256"), weights[name])
    gate.add("03_two_weights_differ",
             bool(weights["initial"]["actual"])
             and weights["initial"]["actual"] != weights["final"]["actual"],
             {"initial": weights["initial"]["actual"],
              "final": weights["final"]["actual"]})
    report["weights"] = weights
    # 3. train.json recomputed
    train_path = Path(chosen["train"]) / "train.json"
    train_ok = False
    if train_path.is_file():
        train = _load_json(train_path)
        # the summary keeps the series beside the checked statistics
        series = [float(value) for value in
                  (summary.get("train") or {}).get("loss_series")
                  or (train.get("loss") or {}).get("series") or []]
        first = series[0] if series else None
        last = series[-1] if series else None
        mean = (sum(series) / len(series)) if series else None
        checks = {
            "updates": int(train.get("updates", -1)) == PILOT_UPDATES,
            "batch": int(train.get("batch", -1)) == PILOT_BATCH,
            "lr": float(((train.get("optimizer") or {}).get("lr", -1))) == PILOT_LR,
            "init_seed": int(train.get("init_seed", -1)) == PILOT_INIT_SEED,
            "parameters": int(train.get("parameters", -1)) == PILOT_PARAMETERS,
            "samples_seen": int(train.get("samples_seen", -1))
            == PILOT_UPDATES * PILOT_BATCH,
            "loss_series_length": len(series) == PILOT_UPDATES,
            "loss_first": first is not None
            and abs(float(train["loss"]["first"]) - first) < 1e-12,
            "loss_last": last is not None
            and abs(float(train["loss"]["last"]) - last) < 1e-12,
            "loss_mean": mean is not None
            and abs(float(train["loss"]["mean"]) - mean) < 1e-9,
            "weights_initial_path": Path(
                (train.get("weights_initial") or {}).get("path") or "").as_posix()
            == weights["initial"]["path"],
            "weights_final_path": Path(
                (train.get("weights_final") or {}).get("path") or "").as_posix()
            == weights["final"]["path"],
            "fit_root_seed": int((train.get("fit_shard") or {}).get("seed", -1))
            == PILOT_FIT_SEED,
        }
        train_ok = all(checks.values())
        gate.add("03_train_json_recomputed", train_ok,
                 {"checks": checks, "parameters": train.get("parameters"),
                  "loss_first": first, "loss_last": last})
        report["train"] = {"path": train_path.as_posix(),
                           "sha256": sha256_file(train_path),
                           "checks": checks,
                           "parameters": train.get("parameters"),
                           "device": train.get("device")}
    else:
        gate.add("03_train_json_recomputed", False, {"missing": train_path.as_posix()})
        report["train"] = {"missing": train_path.as_posix()}
    # 4. the open-loop table, recomputed from its own candidate list
    table_claimed = summary.get("open_loop") or {}
    cands = [int(value) for value in table_claimed.get("cands") or []]
    recomputed_table = table_sha256(cands)
    entry_decision = int((summary.get("task_card") or {}).get("entry_decision", -1))
    entry_frame = int((summary.get("task_card") or {}).get("entry_frame", -1))
    masks = pilot_table_masks(cands, entry_decision)
    flat = flat_masks_of(masks, entry_frame + 1)
    trace = mask_trace_digest(flat, entry_frame + 1,
                              entry_frame + PILOT_WINDOW_FRAMES)
    claimed_masks = [[int(pair[0]), int(pair[1])] for pair in
                     table_claimed.get("masks") or []]
    checks = {
        "length": len(cands) == PILOT_WINDOW_DECISIONS,
        "candidates_in_range": all(0 <= value < 18 for value in cands),
        "table_sha256": recomputed_table == table_claimed.get("table_sha256"),
        "mask_trace_sha256": trace == table_claimed.get("mask_trace_sha256"),
        "masks_recomputed": claimed_masks == masks,
        "r0": int(table_claimed.get("r0", -1)) == entry_frame // 4,
        "history_mode": table_claimed.get("history_mode") == "self_consistent",
    }
    gate.add("03_open_loop_table_recomputed", all(checks.values()),
             {"checks": checks, "table_sha256": recomputed_table,
              "mask_trace_sha256": trace,
              "claimed_table_sha256": table_claimed.get("table_sha256"),
              "claimed_mask_trace_sha256": table_claimed.get("mask_trace_sha256")})
    report["open_loop"] = {"table_sha256": recomputed_table,
                           "mask_trace_sha256": trace, "checks": checks,
                           "history_differing": (table_claimed.get(
                               "history_sensitivity") or {}).get(
                                   "differing_decisions")}
    # 5. every arm, recomputed from its decisions.jsonl
    arms_summary: dict[str, Any] = {}
    arm_ok = True
    for seed, arms in sorted((summary.get("arms") or {}).items()):
        for arm, claimed in sorted(arms.items()):
            result = Path(claimed.get("result") or "")
            detail: dict[str, Any] = {
                "result": result.as_posix(), "exists": result.is_file(),
                "claimed_outcome": claimed.get("outcome")}
            ok = result.is_file()
            if ok:
                arm_record = _load_json(result)
                decisions_path = Path(arm_record.get("decisions_path") or "")
                rows = ([dict(row) for row in iter_jsonl(decisions_path)]
                        if decisions_path.is_file() else [])
                detail.update({
                    "decisions_sha256_matches": (
                        sha256_file(decisions_path)
                        == arm_record.get("decisions_sha256")
                        if decisions_path.is_file() else False),
                    "row_count": len(rows),
                    "claimed_row_count": (arm_record.get("counts") or {}).get(
                        "control_decisions"),
                    "result_sha256": sha256_file(result),
                    "result_sha256_claimed": claimed.get("result_sha256")})
                ok = (ok and len(rows) == int((arm_record.get("counts") or {}).get(
                    "control_decisions", -1)))
                # re-apply the frozen tracker to the recorded rows
                tracker = FrozenTracker(
                    slot=int((arm_record.get("frozen_generation") or {}).get(
                        "slot", -1)),
                    spawn_decision=int((arm_record.get("frozen_generation") or {}
                                        ).get("spawn_decision_from_card", -1)),
                    life_at_spawn=int((arm_record.get("frozen_generation") or {}
                                       ).get("life_at_spawn_from_card", -1)),
                    entry_decision=int((arm_record.get("entry") or {}).get(
                        "card_decision", -1)))
                for row in rows:
                    tracker.observe(row)
                recomputed = (tracker.outcome if tracker.outcome is not None
                              else ("not_removed_within_cap" if tracker.seen
                                    else "generation_not_observed"))
                death_hit = any(float(row["state"]["deaths"])
                                > float(((arm_record.get("resources_at_entry")
                                          or {}).get("state") or {}).get(
                                              "deaths", 0.0))
                                for row in rows)
                game_over_hit = any(int(row["hud"]["game_over"])
                                    for row in rows)
                if game_over_hit:
                    recomputed = "game_over"
                elif death_hit:
                    recomputed = "legal_death"
                detail["recomputed_outcome"] = recomputed
                detail["outcome_matches"] = recomputed == arm_record.get(
                    "outcome")
                detail["frames"] = sum(2 for _row in rows)
                ok = ok and detail["outcome_matches"]
                if arm == "open_loop" and rows:
                    trace_ok = all(
                        [int(row["masks"][0]), int(row["masks"][1])]
                        == masks[int(row["window_index"])]
                        for row in rows)
                    detail["matches_the_frozen_table"] = trace_ok
                    ok = ok and trace_ok
            arms_summary[f"{seed}:{arm}"] = detail
            gate.add(f"03_arm_{seed}_{arm}_recomputed", ok, detail)
            arm_ok = arm_ok and ok
    report["arms"] = arms_summary
    # 6. the gate itself
    if smoke:
        entries = [detail for key, detail in arms_summary.items()
                   if key.endswith(":init") or key.endswith(":final")
                   or key.endswith(":open_loop")]
        recomputed_smoke = bool(len(arms_summary) == len(PILOT_ARMS)
                                and all(detail.get("exists") for detail in
                                        entries))
        gate.add("03_smoke_ok_recomputed",
                 bool(summary.get("smoke_ok")) == recomputed_smoke,
                 {"claimed": summary.get("smoke_ok"),
                  "recomputed": recomputed_smoke,
                  "classification": summary.get("classification")})
        gate.add("03_smoke_is_marked_not_a_pilot_result",
                 summary.get("classification") == PILOT_SMOKE_CLASSIFICATION,
                 summary.get("classification"))
        report["status"] = "verified" if (arm_ok and train_ok
                                          and all(checks.values())) else "failed"
    else:
        comparison = pilot_comparison(
            arms=summary.get("arms") or {}, smoke=False)
        claimed = summary.get("comparison") or {}
        gate.add("03_nomination_recomputed",
                 bool(claimed.get("nominate_expansion"))
                 == bool(comparison.get("nominate_expansion"))
                 == bool(summary.get("nominate_expansion")),
                 {"claimed": summary.get("nominate_expansion"),
                  "comparison": claimed.get("nominate_expansion"),
                  "recomputed": comparison.get("nominate_expansion"),
                  "means": comparison.get("equal_weight_means"),
                  "positive_roots": comparison.get("positive_roots")})
        report["nomination"] = comparison
        report["status"] = "verified" if (arm_ok and train_ok
                                          and comparison.get(
                                              "nominate_expansion")
                                          == summary.get(
                                              "nominate_expansion")) else "failed"
    report["checks"] = len(gate.checks)
    report["classification"] = "offline"
    return report


# ---------------------------------------------------------------------------
# 03 strength check on the fit root (``TRAINING_HOST_REQUEST_02.md`` L51)
# ---------------------------------------------------------------------------
#: L51 asks for one thing beyond the four-root comparison: "开环只用训练材料查强度：
#: 在拟合根的同一任务上至少完成一次、成绩不低于已冻结 #12，并记录动作频率/持续长度
#: 及真实执行可行性".  The pilot measured the three arms on the four *development*
#: roots; this block adds the missing strength check and nothing else.  It re-reads
#: the frozen card, the frozen prefix, the frozen #12 definition and the frozen
#: open-loop table, never edits any of them, and writes only ``03/strength/``.
#: Every number it produces is classified
#: :data:`STRENGTH_CLASSIFICATION`, because seed 13 is the root the single fit
#: consumed: a success here can never be quoted as a development result.
SCHEMA_STRENGTH = "v831-03-strength/1"
SCHEMA_STRENGTH_ROW = "v831-03-strength-decision/1"
STRENGTH_CLASSIFICATION = "strength_check_on_fit_root_not_a_dev_result"
STRENGTH_TICKET = ("V831-PILOT-03 / TRAINING_HOST_REQUEST_02.md#L51 "
                   "(strength on the fit root)")
#: The one open-loop table this check may play.  A different digest refuses the
#: run **before** the first start, so the check can never measure a new policy.
STRENGTH_REQUIRED_TABLE_SHA256 = (
    "77adfa10c02a5667f56482c59ef83f20666a86a54c54ec430535c870fa340bca")
STRENGTH_ARM_OPEN_LOOP = "open_loop"
STRENGTH_ARM_CONSTANT_12 = "constant_12"
STRENGTH_ARMS: tuple[str, ...] = (STRENGTH_ARM_OPEN_LOOP, STRENGTH_ARM_CONSTANT_12)
#: Exactly two starts, one per arm (the work order's own allowance for this check).
STRENGTH_STAGE_STARTS = 2
STRENGTH_ATTEMPT_PREFIXES: tuple[str, ...] = ("03-strength-",)
#: The ticket's outcome order for ``not_worse_than_12``: ``kill_*`` > alive at the
#: cap > ``legal_death``/``game_over``.  A named alternative outside those three
#: tiers ranks 0 (below all of them) and is quoted by name, never folded into a tier.
STRENGTH_OUTCOME_TIERS: dict[str, int] = {
    "kill_M0": 3, "kill_M1": 3, "kill_M2": 3,
    "not_removed_within_cap": 2,
    "legal_death": 1, "game_over": 1,
}
STRENGTH_TIER_LABELS: dict[int, str] = {
    3: "kill_* (the frozen event completed)",
    2: "not_removed_within_cap (alive when the window closed)",
    1: "legal_death/game_over (a life was lost or the run ended)",
    0: ("a named alternative outside the ticket's three tiers "
        "(scripted_retreat/offscreen_despawn/clear_wipe/not_removed_no_event/"
        "generation_not_observed)"),
}


def strength_layout(out_root: str | Path) -> dict[str, Any]:
    """Where the strength check writes: ``03/strength/`` inside the same OUT."""

    root = Path(out_root) / PILOT_WINDOW_KEY
    base = root / "strength"
    return {
        "out": Path(out_root),
        "root": root,
        "base": base,
        "result": base / "result.json",
        "arms": {arm: base / f"arm_{arm}" for arm in STRENGTH_ARMS},
        "table": root / "open_loop" / "table.json",
        "pilot": root / "pilot.json",
        "card": Path(out_root) / "02b",
        "intervene": Path(out_root) / "02c" / "intervene.json",
    }


def strength_table_block(out_root: str | Path, *,
                         required: str = STRENGTH_REQUIRED_TABLE_SHA256
                         ) -> dict[str, Any]:
    """Re-read the frozen open-loop table and recompute its digest from ``cands``.

    Three independent statements must agree before a frame is spent: the table's
    own ``table_sha256``, the digest recomputed from ``"<i>:<cand>\\n"``, and the
    digest the pilot summary recorded when the table was frozen.  The last one is
    what makes this "the table that was frozen before the first development root".
    """

    paths = strength_layout(out_root)
    table_path = Path(paths["table"])
    if not table_path.is_file():
        raise SourceError(f"the frozen open-loop table is missing: {table_path}")
    table = _load_json(table_path)
    cands = [int(value) for value in table.get("cands") or []]
    masks = [[int(pair[0]), int(pair[1])] for pair in table.get("masks") or []]
    recomputed = table_sha256(cands)
    pilot_sha = None
    if Path(paths["pilot"]).is_file():
        pilot_sha = ((_load_json(paths["pilot"]).get("open_loop") or {})
                     .get("table_sha256"))
    checks = {
        "rows_cover_the_window": len(cands) == PILOT_WINDOW_DECISIONS,
        "candidates_in_range": all(0 <= value < 18 for value in cands),
        "recomputed_matches_the_file": recomputed == table.get("table_sha256"),
        "file_matches_the_pilot_summary": (pilot_sha is None
                                          or pilot_sha == recomputed),
        "matches_the_required_digest": recomputed == str(required),
        "masks_cover_the_window": len(masks) == len(cands),
    }
    block = {
        "path": table_path.as_posix(),
        "file_sha256": sha256_file(table_path),
        "bytes": table_path.stat().st_size,
        "table_sha256": recomputed,
        "claimed_table_sha256": table.get("table_sha256"),
        "pilot_summary_table_sha256": pilot_sha,
        "required_table_sha256": str(required),
        "table_sha_definition": table.get("table_sha_definition"),
        "length": len(cands),
        "checks": checks,
        "cands": cands,
        "masks": masks,
        "classification": "measured",
    }
    failed = {name: value for name, value in checks.items() if not value}
    if failed:
        raise SourceError(
            "refusing the open-loop table before any engine start: "
            + "; ".join(f"{name}={value}" for name, value in failed.items()))
    return block


def strength_arm_candidates(arm: str, table_cands: Sequence[int]) -> list[int]:
    """The frozen candidate list of one arm (no policy is consulted here).

    * ``open_loop`` replays the frozen table verbatim;
    * ``constant_12`` is the card's frozen control ``#12``: the same constant
      candidate 12 the 02-C arm C executed, with the same pulse phase.
    """

    if arm == STRENGTH_ARM_OPEN_LOOP:
        return [int(value) for value in table_cands]
    if arm == STRENGTH_ARM_CONSTANT_12:
        return [int(CONTROL_CAND)] * len(table_cands)
    raise EngineRefused(f"unknown strength arm {arm!r}")  # pragma: no cover


def strength_action_stats(*, cands: Sequence[int], masks: Sequence[Sequence[int]],
                          first_decision: int,
                          period: int = DEFAULT_PULSE_PERIOD) -> dict[str, Any]:
    """Action frequency, dwell length and real-executability facts of one arm.

    Every field is derived from the **executed** candidate list and the
    **executed** two-frame mask pairs, so ``verify`` recomputes it from
    ``decisions.jsonl`` alone.  The release rule is the frozen
    ``release_due(decision_index, 48)``: the arm never emits a release of its own,
    and a release frame is the only frame allowed to carry no SHOOT.
    """

    table = adapter()
    shoot = int(table["BTN_SHOOT"])
    bomb_bit = int(table["BTN_BOMB"])
    agent = int(table["AGENT_MASK"])
    values = [int(value) for value in cands]
    pairs = [[int(pair[0]), int(pair[1])] for pair in masks]
    count = min(len(values), len(pairs))
    values = values[:count]
    pairs = pairs[:count]
    histogram: dict[str, int] = {}
    for value in values:
        histogram[str(value)] = histogram.get(str(value), 0) + 1
    longest: dict[str, Any] = {"cand": None, "decisions": 0,
                               "first_window_index": None, "last_window_index": None,
                               "first_decision": None, "last_decision": None}
    runs = 0
    start = 0
    for offset in range(1, count + 1):
        if offset == count or values[offset] != values[start]:
            length = offset - start
            if length > 1:
                runs += 1
            if length > int(longest.get("decisions") or 0):
                longest = {"cand": int(values[start]), "decisions": int(length),
                           "first_window_index": int(start),
                           "last_window_index": int(offset - 1),
                           "first_decision": int(first_decision) + int(start),
                           "last_decision": int(first_decision) + int(offset - 1)}
            start = offset
    release_due = table["release_due"]
    release_offsets: list[int] = []
    per_block: dict[str, int] = {}
    cleared: dict[int, int] = {}
    for offset in range(count):
        if bool(release_due(int(first_decision) + offset, int(period))):
            release_offsets.append(int(offset))
            block = str(int(offset) // int(period))
            per_block[block] = per_block.get(block, 0) + 1
            cleared[int(offset)] = int(pairs[offset][0]) & ~shoot
    release_pairs_ok = sum(
        1 for offset in release_offsets
        if (pairs[offset][0] & shoot) and pairs[offset][1] == cleared[offset])
    anomalies: list[str] = []
    decoded_ok = 0
    frames_without_shoot = 0
    frames_with_bomb = 0
    frames_with_unknown = 0
    for offset, pair in enumerate(pairs):
        for which, mask in enumerate(pair):
            frame = int(first_decision) + offset
            if not mask & shoot:
                frames_without_shoot += 1
            if mask & bomb_bit:
                frames_with_bomb += 1
            if mask & ~agent:
                frames_with_unknown += 1
            if mask & bomb_bit:
                anomalies.append(f"{frame}:bomb_bit")
                continue
            if mask & ~agent:
                anomalies.append(f"{frame}:non_agent_bits")
                continue
            is_release_frame = (which == 1 and offset in cleared
                                and mask == cleared[offset]
                                and bool(pair[0] & shoot))
            if not mask & shoot:
                if not is_release_frame:
                    anomalies.append(f"{frame}:no_shoot")
                continue
            try:
                cand, bomb = table["mask_to_cand"](int(mask))
            except ValueError as exc:
                anomalies.append(f"{frame}:undecodable:{exc}")
                continue
            if int(cand) != int(values[offset]) or int(bomb) != 0:
                anomalies.append(f"{frame}:cand={int(cand)},bomb={int(bomb)}")
                continue
            decoded_ok += 1
    blocks = sorted({offset // int(period) for offset in range(count)})
    sizes: dict[int, int] = {}
    for offset in range(count):
        block = offset // int(period)
        sizes[block] = sizes.get(block, 0) + 1
    complete = [block for block in blocks if sizes.get(block, 0) == int(period)]
    exactly_one = [block for block in complete
                   if per_block.get(str(block), 0) == 1]
    return {
        "decisions": int(count),
        "frames": int(count * DECISION_FRAMES),
        "first_decision": int(first_decision),
        "last_decision": (int(first_decision) + count - 1) if count else None,
        "candidate_histogram": dict(sorted(histogram.items(),
                                           key=lambda item: int(item[0]))),
        "distinct_candidates": len(histogram),
        "dominant_share": (max(histogram.values()) / count) if count else 0.0,
        "longest_same_candidate_run": longest,
        "same_candidate_runs_longer_than_one": int(runs),
        "releases": {
            "period": int(period),
            "rule": "release_due(decision_index, 48) of the frozen executor",
            "count": len(release_offsets),
            "window_indices": release_offsets,
            "decisions": [int(first_decision) + offset
                          for offset in release_offsets],
            "per_48_decision_block": dict(sorted(per_block.items(),
                                                 key=lambda item: int(item[0]))),
            "blocks": len(blocks),
            "complete_blocks": len(complete),
            "partial_blocks": [block for block in blocks
                               if block not in complete],
            "blocks_with_exactly_one_release": len(exactly_one),
            "expected_per_block": 1,
            "pairs_that_clear_shoot_on_the_second_frame": int(release_pairs_ok),
            "all_complete_blocks_have_exactly_one_release": bool(
                len(exactly_one) == len(complete)
                and int(release_pairs_ok) == len(release_offsets)),
        },
        "execution_feasibility": {
            "agent_mask": f"0x{agent:04x}",
            "frames_without_shoot": int(frames_without_shoot),
            "frames_with_bomb": int(frames_with_bomb),
            "frames_with_unknown_bits": int(frames_with_unknown),
            "frames_whose_mask_decodes_to_the_executed_candidate": int(decoded_ok),
            "mask_anomalies": len(anomalies),
            "mask_anomaly_sample": anomalies[:8],
            "all_frames_executable": not anomalies,
            "per_decision_two_frames": bool(count == len(pairs)),
            "note": ("a release frame is the only frame allowed to carry no SHOOT: "
                     "it must be exactly pair[0] & ~SHOOT, the frozen executor's "
                     f"pulse at release_due(i, {int(period)})"),
        },
        "classification": "measured",
    }


def strength_judgment(*, arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The two required verdicts, recomputed from the two arm summaries.

    ``completed_at_least_once``: the open-loop arm's own outcome is a ``kill_*``.
    ``not_worse_than_12``: the ticket's outcome order first (``kill_*`` >
    ``not_removed_within_cap`` > ``legal_death``/``game_over``); on the **same**
    tier both the score and the frozen target's life drop must be at least the
    frozen #12 arm's (a tie-break where only one of the two holds is reported as
    *not* better).  A technical arm makes the comparison inadmissible.
    """

    open_loop = dict(arms.get(STRENGTH_ARM_OPEN_LOOP) or {})
    control = dict(arms.get(STRENGTH_ARM_CONSTANT_12) or {})
    ol_outcome = open_loop.get("outcome")
    ctl_outcome = control.get("outcome")
    ol_tier = int(STRENGTH_OUTCOME_TIERS.get(str(ol_outcome), 0))
    ctl_tier = int(STRENGTH_OUTCOME_TIERS.get(str(ctl_outcome), 0))
    technical = [
        {"arm": arm, "terminal": (arms.get(arm) or {}).get("terminal")}
        for arm in STRENGTH_ARMS
        if str((arms.get(arm) or {}).get("terminal")) == "technical_conflict"
        or (arms.get(arm) or {}).get("exit_code") not in (0, 1)]
    completed = bool(ol_outcome in KILL_OUTCOMES)
    tie_break: dict[str, Any] | None = None
    if technical:
        not_worse = False
        decided_by = "technical_arm"
    elif ol_tier > ctl_tier:
        not_worse = True
        decided_by = "outcome_tier"
    elif ol_tier < ctl_tier:
        not_worse = False
        decided_by = "outcome_tier"
    else:
        ol_score = int(open_loop.get("score_at_end") or 0)
        ctl_score = int(control.get("score_at_end") or 0)
        ol_drop = float(open_loop.get("frozen_life_drop") or 0.0)
        ctl_drop = float(control.get("frozen_life_drop") or 0.0)
        score_ok = bool(ol_score >= ctl_score)
        life_ok = bool(ol_drop >= ctl_drop)
        tie_break = {
            "same_outcome": str(ol_outcome),
            "score_at_end": {"open_loop": ol_score, "constant_12": ctl_score,
                             "not_lower": score_ok},
            "frozen_target_life_drop": {"open_loop": ol_drop,
                                        "constant_12": ctl_drop,
                                        "not_smaller": life_ok,
                                        "definition": ("life_first - life_min of the "
                                                       "frozen slot inside the window")},
            "conflict": bool(score_ok != life_ok),
        }
        not_worse = bool(score_ok and life_ok)
        decided_by = "same_tier_tie_break"
    passed = bool(completed and not_worse and not technical)
    return {
        "rule": (
            "outcome order fixed by the ticket: kill_M0/M1/M2 (3) > "
            "not_removed_within_cap, i.e. alive at the cap (2) > legal_death/"
            "game_over (1); any other named outcome ranks 0 and is quoted by name. "
            "A strictly higher tier is not worse. On the SAME tier the open-loop "
            "arm is not worse only if BOTH its score at the end and its frozen "
            "target's life drop (life_first - life_min) are at least the frozen #12 "
            "arm's; a tie-break where only one of the two holds is recorded as "
            "not worse = false. A technical arm makes the comparison inadmissible "
            "(not worse = false)."),
        "outcome_tiers": dict(STRENGTH_OUTCOME_TIERS),
        "tier_labels": {str(key): value for key, value in
                        STRENGTH_TIER_LABELS.items()},
        "arms": {
            STRENGTH_ARM_OPEN_LOOP: {
                "outcome": ol_outcome, "tier": ol_tier,
                "score_at_end": open_loop.get("score_at_end"),
                "frozen_life_first": open_loop.get("frozen_life_first"),
                "frozen_life_min": open_loop.get("frozen_life_min"),
                "frozen_life_drop": open_loop.get("frozen_life_drop"),
                "terminal": open_loop.get("terminal"),
                "stop_reason": open_loop.get("stop_reason"),
            },
            STRENGTH_ARM_CONSTANT_12: {
                "outcome": ctl_outcome, "tier": ctl_tier,
                "score_at_end": control.get("score_at_end"),
                "frozen_life_first": control.get("frozen_life_first"),
                "frozen_life_min": control.get("frozen_life_min"),
                "frozen_life_drop": control.get("frozen_life_drop"),
                "terminal": control.get("terminal"),
                "stop_reason": control.get("stop_reason"),
            },
        },
        "completed_at_least_once": completed,
        "completed_evidence": {
            "arm": STRENGTH_ARM_OPEN_LOOP,
            "outcome": ol_outcome,
            "outcome_decision": open_loop.get("outcome_decision"),
            "outcome_frame": open_loop.get("outcome_frame"),
            "stop_reason": open_loop.get("stop_reason"),
            "kill_outcomes": list(KILL_OUTCOMES),
        },
        "not_worse_than_12": bool(not_worse),
        "not_worse_evidence": {"decided_by": decided_by,
                               "tie_break": tie_break,
                               "technical_arms": technical,
                               "comparator": (
                                   "the same-seed frozen control arm #12 measured in "
                                   "this check (constant cand 12, release_due(i,48), "
                                   "same prefix, same window)"),
                               },
        "strength_passed": passed,
        "classification": STRENGTH_CLASSIFICATION,
        "ticket_requirement": (
            "TRAINING_HOST_REQUEST_02.md#L51: at least one completion on the fit "
            "root's same task and a result not worse than the frozen #12"),
    }


def strength_arm_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """The per-arm row ``result.json`` and ``verify`` both read."""

    entry = record.get("entry") or {}
    counts = record.get("counts") or {}
    end = record.get("resources_at_end") or {}
    start = record.get("resources_at_entry") or {}
    frozen = record.get("frozen_generation") or {}
    life_first = frozen.get("life_first")
    life_min = frozen.get("life_min")
    drop = (None if life_first is None or life_min is None
            else float(int(life_first) - int(life_min)))
    stats = record.get("action_stats") or {}
    return {
        "arm": record.get("arm"),
        "seed": record.get("seed"),
        "attempt_id": record.get("attempt_id"),
        "controller": record.get("controller"),
        "engine_started": bool(record.get("engine_started")),
        "entry_reached": bool(entry.get("reached")),
        "entry_frame": entry.get("frame"),
        "entry_hash": entry.get("hash"),
        "entry_stage": entry.get("stage"),
        "entry_dead": entry.get("dead"),
        "card_entry_hash": entry.get("card_hash"),
        "card_entry_hash_matches": entry.get("card_hash_matches"),
        "identity_mismatch": entry.get("identity_mismatch"),
        "observed_spawn_decision": entry.get("observed_spawn_decision"),
        "card_spawn_decision": entry.get("card_spawn_decision"),
        "frozen_slot_present": entry.get("frozen_slot_present"),
        "frozen_slot_life_at_entry": entry.get("frozen_slot_life"),
        "prefix_decisions": counts.get("prefix_decisions"),
        "prefix_frames": counts.get("prefix_frames"),
        "control_decisions": counts.get("control_decisions"),
        "control_frames": counts.get("control_frames"),
        "engine_frames": counts.get("engine_frames"),
        "outcome": record.get("outcome"),
        "stop_reason": record.get("stop_reason"),
        "outcome_decision": record.get("outcome_decision"),
        "outcome_frame": record.get("outcome_frame"),
        "deaths_observed": record.get("deaths_observed"),
        "lives_lost": record.get("lives_lost"),
        "lives_at_entry": (start.get("state") or {}).get("lives_remaining"),
        "lives_at_end": (end.get("state") or {}).get("lives_remaining"),
        "score_at_entry": (start.get("state") or {}).get("score"),
        "score_at_end": (end.get("state") or {}).get("score"),
        "power_at_end": (end.get("state") or {}).get("player_power"),
        "graze_at_end": (end.get("state") or {}).get("graze"),
        "frozen_life_first": life_first,
        "frozen_life_last": frozen.get("life_last"),
        "frozen_life_min": life_min,
        "frozen_life_drop": drop,
        "frozen_observations": frozen.get("observations"),
        "terminal": (record.get("terminal") or {}).get("kind"),
        "candidate_histogram": record.get("candidate_histogram"),
        "candidate_dominant_share": record.get("candidate_dominant_share"),
        "longest_same_candidate_run": stats.get("longest_same_candidate_run"),
        "releases": (stats.get("releases") or {}).get("count"),
        "releases_per_48_decision_block": (
            (stats.get("releases") or {}).get("per_48_decision_block")),
        "action_stats": stats,
        "exit_code": record.get("arm_exit_code"),
        "result": record.get("result_path"),
        "result_sha256": record.get("result_sha256"),
        "mute_problems": ((record.get("mute") or {}).get("after_exit_problems")),
        "ledger": record.get("ledger"),
        "classification": record.get("classification"),
    }


def run_strength_arm(*, args: argparse.Namespace, out_root: Path, repo: Path,
                     card: Mapping[str, Any], tape: Mapping[str, Any],
                     table: Mapping[str, Any], paths: Mapping[str, Any],
                     seed: int, arm: str, limits: BudgetLimits,
                     meter: Any) -> dict[str, Any]:
    """One engine start: the frozen prefix from this seed's RESET, then one arm.

    Nothing about the executor is new here: the prefix is
    :func:`_pilot_prefix_walk` (the card's decisions ``0..1542`` of the frozen
    tape, one ``STEP mask 1`` per frame with ``EVENTS`` after each frame), the
    window is :func:`_step_one` + :func:`read_decision` +
    :class:`FrozenTracker`, and the terminal priority is the card's own.  What is
    new is only which controller supplies the window's candidates.
    """

    entry = int(card["entry"]["prefix_decisions"])
    entry_frame = int(card["entry"]["frame"])
    decision_cap = int(card["frame_cap"]["decisions"])
    frame_cap = int(card["frame_cap"]["frames"])
    slot = int(card["event"]["seed12_instance"]["slot"])
    card_spawn = int(card["event"]["seed12_instance"]["spawn_decision"])
    card_life = int(card["event"]["seed12_instance"]["life_at_spawn"])
    base = Path(paths["arms"][arm])
    base.mkdir(parents=True, exist_ok=True)
    attempt_id = f"03-strength-{arm}"
    cands = strength_arm_candidates(arm, table["cands"])
    if len(cands) != decision_cap:
        raise EngineRefused(
            f"strength arm {arm}: the controller supplies {len(cands)} decisions, "
            f"the frozen card's window is {decision_cap}; the window must be "
            "covered")
    masks = pilot_table_masks(cands, entry)
    if arm == STRENGTH_ARM_OPEN_LOOP:
        recorded = [[int(pair[0]), int(pair[1])] for pair in table["masks"]]
        if recorded != masks:
            raise EngineRefused(
                "the frozen table's masks are not the frozen executor's expansion "
                f"for decisions {entry}..{entry + decision_cap - 1}")
    controller = {
        STRENGTH_ARM_OPEN_LOOP: "frozen open-loop table (03/open_loop/table.json)",
        STRENGTH_ARM_CONSTANT_12: ("frozen control #12: constant candidate 12, the "
                                   "same pulse phase"),
    }[arm]
    planned = entry_frame + frame_cap
    attempt = EngineAttempt(
        args=args, out_root=out_root, base_dir=base, attempt_id=attempt_id,
        planned_frames=int(planned),
        note=(f"03 strength arm {arm} on the fit root seed {seed}: frozen prefix "
              f"decisions 0..{entry - 1} (frames 1..{entry_frame}), then "
              f"{decision_cap} decisions / {frame_cap} frames"),
        limits=limits, meter=meter, reset_seed=int(seed))
    record: dict[str, Any] = {
        "schema": SCHEMA_STRENGTH,
        "created_at": now_iso(),
        "classification": STRENGTH_CLASSIFICATION,
        "ticket": STRENGTH_TICKET,
        "seed": int(seed),
        "fit_root_seed": True,
        "arm": arm,
        "attempt_id": attempt_id,
        "controller": controller,
        "repo": Path(repo).as_posix(),
        "out": Path(out_root).as_posix(),
        "base": base.as_posix(),
        "host": host_identity(),
        "heads": git_identity(repo),
        "tool": {"path": TOOL_PATH.as_posix(), "sha256": sha256_file(TOOL_PATH)},
        "table": {"row_table_sha256": table.get("table_sha256"),
                  "table_path": table.get("path"),
                  "table_file_sha256": table.get("file_sha256")},
        "engine_started": False,
        "entry": {"card_decision": entry, "card_frame": entry_frame,
                  "card_hash": card["entry"]["hash"], "reached": False,
                  "hash": None, "frame": None, "stage": None, "dead": None,
                  "card_hash_matches": None,
                  "card_hash_note": (
                      "the card's entry hash was measured on seed 12; this check "
                      "runs seed 13, so the hash is recorded and compared but it "
                      "is not an arrival predicate here"),
                  "identity_mismatch": None},
        "counts": {"prefix_decisions": 0, "prefix_frames": 0,
                   "control_decisions": 0, "control_frames": 0,
                   "engine_frames": 0},
        "terminal": {"kind": "unknown"},
    }
    exit_code = EXIT_TECHNICAL
    closed: dict[str, Any] = {}
    try:
        opened = attempt.open()
        record["engine_started"] = True
        record.update({"preflight": opened["preflight"],
                       "private_game_dir": opened["private"],
                       "reset": opened["reset"],
                       "engine_session": opened["session"],
                       "engine_env": attempt.module.engine_env_record()})
        prefix = _pilot_prefix_walk(
            attempt=attempt, tape=tape, entry_decision=entry,
            entry_frame=entry_frame, slot=slot, buffer=None,
            rows_path=base / "prefix.jsonl")
        record["counts"].update({"prefix_decisions": prefix["decisions"],
                                 "prefix_frames": prefix["frames"]})
        record["prefix"] = {
            "decisions": prefix["decisions"], "frames": prefix["frames"],
            "rows": (base / "prefix.jsonl").as_posix(),
            "prefix_masks_sha256_recomputed": prefix.get(
                "prefix_digest_recomputed"),
            "prefix_masks_sha256_card": card["entry"]["prefix_masks_sha256"],
            "prefix_masks_sha256_matches": bool(
                prefix.get("prefix_digest_recomputed")
                == card["entry"]["prefix_masks_sha256"]),
            "spawn_transitions": prefix["spawns"],
            "shortfall": prefix["shortfall"],
            "classification": "measured"}
        if prefix["shortfall"] is not None:
            short = prefix["shortfall"]
            record["terminal"] = {
                "kind": "entry_not_reached",
                "detail": (f"the STEP into frame {short['frame']} returned "
                           f"advanced={short['advanced']} exit={short['exit']} "
                           f"game_over={short['game_over']}"),
                "shortfall": short, "engine_started": True}
            exit_code = EXIT_GATE_FAILED
            raise _PilotStop(record["terminal"])
        entry_reads = read_decision(attempt.host)
        state = engine_state_record(entry_reads["state"])
        hud = hud_record(entry_reads["hud"])
        player = player_record(entry_reads["player"])
        entry_rows = entry_reads["hazards"]["enemies"]
        frozen_rows = [item for item in entry_rows if int(item["slot"]) == slot]
        frozen_row = (max(frozen_rows, key=lambda item: int(item["life"]))
                      if frozen_rows else None)
        spawns = [item for item in prefix["spawns"]
                  if item.get("kind") != "life_jump_up"]
        observed_spawn = int(spawns[-1]["decision_index"]) if spawns else None
        identity_mismatch = (observed_spawn is None
                             or abs(observed_spawn - card_spawn) > 4)
        dead = bool(state["lives_remaining"] <= 0 or hud["game_over"]
                    or player["player_state_name"] == "DYING")
        record["entry"].update({
            "hash": prefix["hash_at_entry"],
            "frame": int(state["frame"]),
            "stage": int(state["stage"]),
            "dead": dead,
            "reached": bool(state["frame"] == entry_frame and not dead
                            and int(state["stage"]) == 0),
            "card_hash_matches": bool(str(prefix["hash_at_entry"])
                                      == str(card["entry"]["hash"])),
            "frozen_slot_present": frozen_row is not None,
            "frozen_slot_boss": None if frozen_row is None else bool(
                frozen_row["boss"]),
            "frozen_slot_life": None if frozen_row is None else int(
                frozen_row["life"]),
            "frozen_slot_flags1": None if frozen_row is None else str(
                frozen_row["flags1"]),
            "observed_spawn_decision": observed_spawn,
            "card_spawn_decision": card_spawn,
            "spawn_delta": (None if observed_spawn is None
                            else int(observed_spawn) - card_spawn),
            "identity_mismatch": bool(identity_mismatch),
            "classification": "measured"})
        record["resources_at_entry"] = {
            "source": (f"this arm's own RESET 0 1 0 {int(seed)}, at the entry "
                       f"frame {entry_frame}"),
            "state": state, "hud": hud, "player": player,
            "counts": {"bullets": entry_reads["hazards"]["bullets"],
                       "lasers": entry_reads["hazards"]["lasers"],
                       "enemies": entry_reads["hazards"]["enemies_count"]},
            "frame": entry_frame,
            "classification": "measured"}
        if not record["entry"]["reached"]:
            record["terminal"] = {
                "kind": "entry_not_reached",
                "detail": (f"frame {state['frame']} stage {state['stage']} dead "
                           f"{dead}"),
                "engine_started": True}
            exit_code = EXIT_GATE_FAILED
            raise _PilotStop(record["terminal"])
        tracker = FrozenTracker(
            slot=slot,
            spawn_decision=(card_spawn if observed_spawn is None
                            else int(observed_spawn)),
            life_at_spawn=(card_life if frozen_row is None
                           else int(frozen_row["life"])),
            entry_decision=entry)
        deaths_at_entry = float(state["deaths"])
        lives_at_entry = float(state["lives_remaining"])
        event_bits: dict[str, int] = {}
        nonzero_frames: list[list[Any]] = []
        event_frames = 0
        hash_frames = 0
        last_row: dict[str, Any] | None = None
        frames_in_window = 0
        outcome: str | None = None
        stop_reason = "decision_cap"
        tracker_outcome: str | None = None
        cand_histogram: dict[str, int] = {}
        with (base / "decisions.jsonl").open("w", encoding="utf-8") as handle:
            for offset in range(decision_cap):
                index = entry + offset
                cand = int(cands[offset])
                pair = [int(masks[offset][0]), int(masks[offset][1])]
                frame_start = index * DECISION_FRAMES + 1
                frame_end = frame_start + DECISION_FRAMES - 1
                if frames_in_window + DECISION_FRAMES > frame_cap:
                    stop_reason = "frame_cap"
                    break
                flags: list[int] = []
                hashes: dict[int, str] = {}
                reply: dict[str, Any] = {}
                shortfall = None
                for step_offset, mask in enumerate(pair):
                    frame = frame_start + step_offset
                    stepped = _step_one(attempt, int(mask), frame)
                    if not stepped["ok"]:
                        shortfall = {"frame": frame,
                                     "advanced": stepped["advanced"],
                                     "exit": stepped["exit"],
                                     "game_over": stepped["game_over"]}
                        break
                    frames_in_window += 1
                    flags.append(int(stepped["flags"]))
                    if stepped["hash"] is not None:
                        hashes[frame] = str(stepped["hash"])
                        hash_frames += 1
                    reply = stepped
                    event_frames += 1
                    for name in event_names(int(stepped["flags"])):
                        event_bits[name] = event_bits.get(name, 0) + 1
                    if int(stepped["flags"]):
                        if len(nonzero_frames) < 400:
                            nonzero_frames.append([frame,
                                                   f"0x{int(stepped['flags']):04x}"])
                if shortfall is not None:
                    stop_reason = ("engine_exit" if shortfall["exit"]
                                   else "frame_shortfall")
                    record["terminal"] = {
                        "kind": "technical_conflict" if not shortfall["exit"]
                        else "engine_exit",
                        "detail": shortfall, "engine_started": True}
                    exit_code = (EXIT_TECHNICAL if not shortfall["exit"]
                                 else EXIT_GATE_FAILED)
                    break
                reads = read_decision(attempt.host)
                row: dict[str, Any] = {
                    "schema": SCHEMA_STRENGTH_ROW,
                    "phase": "control",
                    "arm": arm,
                    "seed": int(seed),
                    "classification_note": STRENGTH_CLASSIFICATION,
                    "decision_index": index,
                    "window_index": offset,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "cand": int(cand),
                    "action_source": ("frozen_table" if arm == STRENGTH_ARM_OPEN_LOOP
                                      else "constant_cand_12"),
                    "masks": pair,
                    "release_due": bool(adapter()["release_due"](
                        index, DEFAULT_PULSE_PERIOD)),
                    "ev_flags": [f"0x{value:04x}" for value in flags],
                    "ev_flags_dec": [int(value) for value in flags],
                    "ev_names": [event_names(value) for value in flags],
                    "state": engine_state_record(reads["state"]),
                    "hud": hud_record(reads["hud"]),
                    "player": player_record(reads["player"]),
                    "enemies": reads["hazards"]["enemies"],
                    "hazards_counts": {"bullets": reads["hazards"]["bullets"],
                                       "lasers": reads["hazards"]["lasers"],
                                       "enemies": reads["hazards"]["enemies_count"]},
                    "hash": {"frame_start": hashes.get(frame_start),
                             "frame_end": hashes.get(frame_end)},
                    "classification": "measured",
                }
                observed = tracker.observe(row)
                if observed is not None and tracker_outcome is None:
                    tracker_outcome = observed
                row["frozen"] = {
                    "slot": tracker.slot,
                    "present": any(int(item["slot"]) == tracker.slot
                                   for item in reads["hazards"]["enemies"]),
                    "life": tracker.last_life,
                    "outcome_so_far": tracker_outcome,
                    "phase_ends_so_far": len(tracker.phase_ends),
                }
                write_jsonl_row(handle, row)
                last_row = row
                cand_histogram[str(int(cand))] = cand_histogram.get(
                    str(int(cand)), 0) + 1
                record["counts"]["control_decisions"] += 1
                record["counts"]["control_frames"] += DECISION_FRAMES
                game_over = int(reads["hud"].get("game_over", 0) or 0) \
                    or int(reply.get("game_over", 0) or 0)
                deaths = float(reads["state"]["deaths"])
                # the frozen card's termination priority, highest first
                if game_over:
                    outcome, stop_reason = "game_over", "game_over"
                    break
                if deaths > deaths_at_entry:
                    outcome, stop_reason = "legal_death", "legal_death"
                    break
                if tracker_outcome is not None:
                    outcome, stop_reason = tracker_outcome, "frozen_consequence"
                    break
            else:
                stop_reason = "decision_cap"
        record["counts"]["engine_frames"] = prefix["frames"] + frames_in_window
        attempt.engine_frames = record["counts"]["engine_frames"]
        executed = int(record["counts"]["control_decisions"])
        record["window"] = {
            "decisions": executed,
            "frames": int(frames_in_window),
            "frame_range": [entry_frame + 1, entry_frame + frames_in_window],
            "decision_range": [entry, entry + max(0, executed) - 1],
            "decision_cap": decision_cap, "frame_cap": frame_cap,
            "last_decision": (None if last_row is None
                              else int(last_row["decision_index"])),
            "last_frame": (None if last_row is None else int(last_row["frame_end"])),
            "classification": "measured"}
        record["decisions_path"] = (base / "decisions.jsonl").as_posix()
        record["decisions_sha256"] = sha256_file(base / "decisions.jsonl")
        record["execution"] = {
            "frames_stepped": int(frames_in_window),
            "events_frames_read": int(event_frames),
            "frames_with_a_state_hash": int(hash_frames),
            "decisions_with_hazards_state_hud_player": int(executed),
            "shortfall": None,
            "one_step_mask_1_per_frame": True,
            "note": ("one STEP mask 1 per frame with EVENTS after each frame, then "
                     "HAZARDS+STATE+HUD+PLAYER once per decision: the same read "
                     "method the card froze"),
            "classification": "measured"}
        record["events_summary"] = {
            "frames": int(event_frames),
            "frames_with_any_bit": len(nonzero_frames),
            "by_name": dict(sorted(event_bits.items())),
            "nonzero_frames_sample": nonzero_frames[:400],
            "classification": "measured"}
        record["resources_at_end"] = (
            None if last_row is None else {
                "frame": int(last_row["frame_end"]),
                "state": dict(last_row["state"]), "hud": dict(last_row["hud"]),
                "player": dict(last_row["player"]),
                "counts": dict(last_row["hazards_counts"]),
                "classification": "measured"})
        record["deaths_observed"] = (0.0 if last_row is None else
                                     float(last_row["state"]["deaths"])
                                     - deaths_at_entry)
        record["lives_lost"] = (0.0 if last_row is None else
                                lives_at_entry
                                - float(last_row["state"]["lives_remaining"]))
        record["frozen_generation"] = tracker.summary()
        record["frozen_generation"]["expected_identity"] = {
            "slot": slot, "card_spawn_decision": card_spawn,
            "card_life_at_spawn": card_life,
            "observed_spawn_decision": observed_spawn,
            "identity_mismatch": bool(identity_mismatch)}
        if outcome is None:
            outcome = ("not_removed_within_cap" if tracker.seen
                       else "generation_not_observed")
        record["outcome"] = outcome
        record["tracker_outcome"] = tracker_outcome
        record["outcome_decision"] = tracker.outcome_decision
        record["outcome_frame"] = tracker.outcome_frame
        record["stop_reason"] = stop_reason
        record["candidate_histogram"] = dict(
            sorted(cand_histogram.items(), key=lambda item: int(item[0])))
        record["candidate_dominant_share"] = (
            max(cand_histogram.values()) / sum(cand_histogram.values())
            if cand_histogram else 0.0)
        record["action_stats"] = strength_action_stats(
            cands=cands[:executed], masks=masks[:executed],
            first_decision=entry)
        record["terminal"] = {
            "kind": ("frozen_consequence" if outcome in OUTCOME_VOCABULARY
                     and outcome not in ("legal_death", "game_over")
                     else outcome),
            "outcome": outcome, "stop_reason": stop_reason,
            "window_frames": int(frames_in_window),
            "last_decision": (None if last_row is None
                              else int(last_row["decision_index"])),
            "game_over": (0 if last_row is None
                          else int(last_row["hud"]["game_over"])),
            "stage": (None if last_row is None
                      else int(last_row["state"]["stage"])),
            "classification": "measured"}
        exit_code = EXIT_OK
    except _PilotStop as stop:
        record["terminal"] = dict(stop.payload)
    except (EngineRefused, FrameRelationConflict, SourceError) as exc:
        record["terminal"] = {"kind": "technical_conflict",
                              "reason": type(exc).__name__,
                              "detail": str(exc),
                              "engine_started": bool(record.get("engine_started"))}
        exit_code = EXIT_TECHNICAL
    except Exception as exc:  # noqa: BLE001
        record["terminal"] = {"kind": "technical_conflict",
                              "reason": type(exc).__name__,
                              "detail": str(exc),
                              "traceback": traceback.format_exc(),
                              "engine_started": bool(record.get("engine_started"))}
        exit_code = EXIT_TECHNICAL
    finally:
        if attempt.engine_started or (attempt.reservation is not None
                                      and attempt.settlement is None):
            try:
                attempt.engine_frames = int(
                    record["counts"].get("engine_frames")
                    or record["counts"].get("prefix_frames", 0))
                closed = attempt.close(result={
                    "command": f"03-strength-{arm}",
                    "seed": int(seed),
                    "terminal": (record.get("terminal") or {}).get("kind"),
                    "outcome": record.get("outcome"),
                    "reached_frame": attempt.engine_frames})
                record["mute"] = closed["mute"]
                record["source_dir"] = closed["source_dir"]
                record["ledger"] = closed["ledger"]
                record["measured"] = closed["measured"]
            except Exception as exc:  # noqa: BLE001
                record["close_error"] = f"{type(exc).__name__}: {exc}"
    record["arm_exit_code"] = int(exit_code)
    record["finished_at"] = now_iso()
    path = write_json(base / "result.json", record)
    record["result_path"] = path.as_posix()
    record["result_sha256"] = sha256_file(path)
    return record


def strength_control_reference(out_root: str | Path) -> dict[str, Any]:
    """The 02-C frozen #12 arm, quoted as **context only** (seed 12, not seed 13).

    The comparator of this check is the same-seed ``constant_12`` arm measured
    beside the table; this block exists so the reader can see the frozen control's
    own seed-12 numbers without them ever entering the verdict.
    """

    path = Path(strength_layout(out_root)["intervene"])
    if not path.is_file():
        return {"present": False, "path": path.as_posix(),
                "note": "no 02-C evidence; the comparator is the same-seed arm",
                "classification": "unverified"}
    try:
        record = _load_json(path)
    except (OSError, ValueError) as exc:
        return {"present": False, "path": path.as_posix(),
                "error": f"{type(exc).__name__}: {exc}",
                "classification": "unverified"}
    arm = ((record.get("arms") or {}).get("C") or {})
    generation = arm.get("frozen_generation") or {}
    return {
        "present": True,
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "seed": 12,
        "controller": arm.get("controller"),
        "outcome": arm.get("outcome"),
        "stop_reason": arm.get("stop_reason"),
        "control_decisions": (arm.get("counts") or {}).get("control_decisions"),
        "control_frames": (arm.get("counts") or {}).get("control_frames"),
        "score_at_end": ((arm.get("resources_at_end") or {}).get("state")
                         or {}).get("score"),
        "deaths_observed": arm.get("deaths_observed"),
        "lives_lost": arm.get("lives_lost"),
        "frozen_life_first": generation.get("life_first"),
        "frozen_life_last": generation.get("life_last"),
        "frozen_life_min": generation.get("life_min"),
        "used_by_the_verdict": False,
        "note": ("seed 12 is not the fit root, so these numbers are quoted for "
                 "context and are explicitly excluded from not_worse_than_12"),
        "classification": "historical",
    }


def strength_fit_root_shard(repo: str | Path, seed: int) -> str | None:
    """The fit root's registered shard name, or ``None`` if it is not readable.

    A label only: nothing in the check is derived from it, so a missing corpus
    must not stop a start that the card/table gates would allow.  It is recorded
    as ``None`` and the reader can see it.
    """

    if int(seed) != int(PILOT_FIT_SEED):
        return None
    try:
        return str(d1_root_row(repo, seed, 0).get("shard"))
    except (SourceError, OSError, ValueError, KeyError):
        return None


def strength_attempt_ids(out_root: str | Path) -> set[str]:
    """The ``03-strength-*`` attempt ids a later package's evidence accounts for."""

    path = Path(strength_layout(out_root)["result"])
    if not path.is_file():
        return set()
    try:
        record = _load_json(path)
    except (OSError, ValueError):
        return set()
    return {str(summary.get("attempt_id")) for summary in
            (record.get("arms") or {}).values() if summary.get("attempt_id")}


def run_json_ledger_reconciliation(*, claimed_totals: Mapping[str, Any],
                                   ledger_rows: Sequence[Mapping[str, Any]],
                                   ledger_actual: Mapping[str, Any],
                                   accounted_attempt_ids: Iterable[str]
                                   ) -> dict[str, Any]:
    """Does the ledger's growth after ``run.json`` reconcile exactly?

    ``TRAINING_HOST_REQUEST_02`` line 60 allows a package to be driven on its own
    in the same directory ("实现复杂处可在同一目录用等价的显式命令分步执行"), so the
    ledger legitimately grows after ``run`` wrote its totals: the 03 pilot and the
    03 strength check are such steps.  The strict equality is tried first; when it
    does not hold, every row past ``ledger_totals.rows`` must belong to an attempt
    that a later package's own evidence accounts for **and** the recorded totals
    plus those rows' settle actuals must equal the ledger's.  Nothing is
    "tolerated": both sets of numbers and the extra attempt ids are reported.
    """

    recorded_rows = int(claimed_totals.get("rows") or 0)
    recorded_actual = dict(claimed_totals.get("actual") or {})
    strict = bool(recorded_actual.get("starts") == ledger_actual.get("starts"))
    extra = ([dict(row) for row in ledger_rows
              if int(row.get("seq") or 0) > recorded_rows] if recorded_rows else [])
    extra_ids = sorted({str(row.get("attempt_id")) for row in extra})
    accounted = {str(value) for value in accounted_attempt_ids}
    extra_actual = _zero_totals()
    for row in extra:
        if row.get("phase") == "settle":
            _add_totals(extra_actual, row.get("actual") or {})
    reconciled = {name: float(recorded_actual.get(name, 0) or 0)
                  + float(extra_actual.get(name, 0) or 0)
                  for name in BUDGET_DIMENSIONS}
    unaccounted = sorted(set(extra_ids) - accounted)
    reconciled_ok = bool(
        extra_ids and not unaccounted
        and all(abs(reconciled[name] - float(ledger_actual.get(name, 0) or 0)) < 1e-6
                for name in BUDGET_DIMENSIONS))
    return {
        "strict_starts_equal": strict,
        "recorded_rows": recorded_rows,
        "recorded_actual": recorded_actual,
        "ledger_rows": len(ledger_rows),
        "ledger_actual": dict(ledger_actual),
        "extra_rows": len(extra),
        "extra_attempt_ids": extra_ids,
        "unaccounted_attempt_ids": unaccounted,
        "extra_actual": dict(extra_actual),
        "reconciled_actual": reconciled,
        "reconciled": reconciled_ok,
        "ok": bool(strict or reconciled_ok),
        "rule": ("a ledger that grew after run.json is consistent only if every "
                 "later row belongs to an attempt a later package's evidence "
                 "accounts for and recorded + later settle actuals == the ledger"),
        "classification": "measured",
    }


def cmd_strength(args: argparse.Namespace) -> int:
    """03 strength check on the fit root: two starts, two arms, one verdict.

    Nothing here is a new executor: the frozen card, the frozen tape prefix, the
    frozen #12 definition and the frozen table are read, re-checked and replayed.
    Only ``03/strength/`` is written.
    """

    module = slice_tool()
    maybe_reexec_under_xvfb(enabled=not bool(getattr(args, "no_xvfb", False)))
    started = time.perf_counter()
    out_root = Path(args.out).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve()
    paths = strength_layout(out_root)
    base = Path(paths["base"])
    base.mkdir(parents=True, exist_ok=True)
    seed = int(getattr(args, "seed", None) or PILOT_FIT_SEED)
    gate = Gate("03_strength")
    meter = module.CostMeter()
    meter.begin()
    record: dict[str, Any] = {
        "schema": SCHEMA_STRENGTH,
        "generated_by": "tools/rl/v8_31_followup.py strength",
        "created_at": now_iso(),
        "classification": STRENGTH_CLASSIFICATION,
        "ticket": STRENGTH_TICKET,
        "requirement": (
            "TRAINING_HOST_REQUEST_02.md#L51: '开环只用训练材料查强度：在拟合根的同一"
            "任务上至少完成一次、成绩不低于已冻结 #12，并记录动作频率/持续长度及真实"
            "执行可行性'"),
        "seed": int(seed),
        "fit_root": {"seed": int(seed), "id": PILOT_FIT_ROOT_ID,
                     "shard": strength_fit_root_shard(repo, seed),
                     "only_the_fit_root_is_accepted": int(PILOT_FIT_SEED)},
        "repo": repo.as_posix(),
        "out": out_root.as_posix(),
        "argv": list(getattr(args, "argv", []) or []),
        "host": host_identity(),
        "heads": git_identity(repo),
        "tool": {"path": TOOL_PATH.as_posix(), "sha256": sha256_file(TOOL_PATH)},
        "slice_helper": slice_identity(),
        "layout": {"root": Path(paths["root"]).as_posix(),
                   "base": base.as_posix(),
                   "result": Path(paths["result"]).as_posix(),
                   "arms": {arm: Path(paths["arms"][arm]).as_posix()
                            for arm in STRENGTH_ARMS}},
        "arms": {},
        "engine_starts": 0,
        "judgment": {"strength_passed": False,
                     "completed_at_least_once": False,
                     "not_worse_than_12": False},
        "cli_budget_declared": {
            "max_starts": getattr(args, "max_starts", None),
            "max_engine_frames": getattr(args, "max_engine_frames", None),
            "max_wall_s": getattr(args, "max_wall_s", None),
            "max_write_gib": getattr(args, "max_write_gib", None),
            "note": ("the CLI restates the whole-ticket limits; this stage is fixed "
                     f"at {STRENGTH_STAGE_STARTS} starts / its two arms' frames and "
                     "the CLI can never widen it"),
            "classification": "measured"},
        "started_at": now_iso(),
    }
    exit_code = EXIT_TECHNICAL
    try:
        if seed != int(PILOT_FIT_SEED):
            raise EngineRefused(
                f"the strength check runs the fit root only (seed {PILOT_FIT_SEED}); "
                f"seed {seed} was asked for")
        gate.add("strength_runs_the_fit_root_only", True,
                 {"seed": int(seed), "fit_root": PILOT_FIT_ROOT_ID})
        frozen = read_task_card(paths["card"])
        card = dict(frozen["card"])
        record["task_card"] = {
            "path": Path(frozen["path"]).as_posix(),
            "sha256": frozen["sha256"],
            "content_sha256": frozen["content_sha256"],
            "content_sha256_recomputed": frozen["content_sha256_recomputed"],
            "task_id": card.get("task_id"),
            "entry_decision": int(card["entry"]["prefix_decisions"]),
            "entry_frame": int(card["entry"]["frame"]),
            "frame_cap": dict(card["frame_cap"]),
            "termination_priority": list(card["termination_priority"]),
            "control": dict(card["control"]),
            "success": card["success"],
            "legal_failure": card["legal_failure"],
            "classification": "historical"}
        gate.add("strength_uses_the_frozen_card", frozen["content_sha256_matches"],
                 {"path": record["task_card"]["path"],
                  "sha256": record["task_card"]["sha256"],
                  "content_sha256": record["task_card"]["content_sha256"]})
        entry = int(card["entry"]["prefix_decisions"])
        entry_frame = int(card["entry"]["frame"])
        decision_cap = int(card["frame_cap"]["decisions"])
        frame_cap = int(card["frame_cap"]["frames"])
        table = strength_table_block(out_root)
        record["table"] = {key: value for key, value in table.items()
                           if key not in ("cands", "masks")}
        record["table"]["cands_sha256"] = sha256_bytes(
            ("".join(f"{index}:{int(cand)}\n"
                     for index, cand in enumerate(table["cands"]))).encode("ascii"))
        gate.add("open_loop_table_matches_the_frozen_digest",
                 table["table_sha256"] == STRENGTH_REQUIRED_TABLE_SHA256
                 and all(table["checks"].values()),
                 {"table_sha256": table["table_sha256"],
                  "required": STRENGTH_REQUIRED_TABLE_SHA256,
                  "file_sha256": table["file_sha256"],
                  "checks": table["checks"]})
        tape = tape_schedule(repo)
        record["frozen_prefix"] = {
            "source": "the frozen tape's decisions 0..entry-1",
            "tape_shard": tape["shard"],
            "tape_sha256": tape["sha256"],
            "tape_registered_sha256": tape["registered_sha256"],
            "prefix_masks_sha256_recomputed": tape_digest(tape, 0, entry),
            "prefix_masks_sha256_card": card["entry"]["prefix_masks_sha256"],
            "entry_decision": entry, "entry_frame": entry_frame,
            "classification": "offline"}
        gate.add("frozen_prefix_matches_the_tape",
                 record["frozen_prefix"]["prefix_masks_sha256_recomputed"]
                 == record["frozen_prefix"]["prefix_masks_sha256_card"],
                 {"recomputed":
                  record["frozen_prefix"]["prefix_masks_sha256_recomputed"],
                  "card": record["frozen_prefix"]["prefix_masks_sha256_card"]})
        need_starts = int(STRENGTH_STAGE_STARTS)
        need_frames = int(len(STRENGTH_ARMS) * (entry_frame + frame_cap))
        stage = stage_budget(
            out_root, need_starts=need_starts, need_frames=need_frames,
            label=f"03-strength-on-the-fit-root-seed{seed}",
            stage_cap={"starts": need_starts, "engine_frames": need_frames},
            stage_prefixes=STRENGTH_ATTEMPT_PREFIXES)
        record["budget"] = stage["report"]
        gate.add("budget_checked_from_the_existing_ledger",
                 not stage["report"]["exceeded"],
                 {"ledger_rows_read": stage["report"]["ledger_rows_read"],
                  "cumulative": stage["report"]["cumulative"],
                  "stage_used": stage["report"]["stage_used"],
                  "stage_remaining": stage["report"]["stage_remaining"],
                  "need": stage["report"]["need"]})
        for arm in STRENGTH_ARMS:
            arm_record = run_strength_arm(
                args=args, out_root=out_root, repo=repo, card=card, tape=tape,
                table=table, paths=paths, seed=seed, arm=arm,
                limits=stage["limits"], meter=meter)
            summary = strength_arm_summary(arm_record)
            record["arms"][arm] = summary
            record["engine_starts"] = int(record["engine_starts"]) + int(
                bool(arm_record.get("engine_started")))
            print(f"  03-strength seed {seed} arm {arm}: "
                  f"entry={summary['entry_reached']} "
                  f"frames={summary['engine_frames']} "
                  f"outcome={summary['outcome']} "
                  f"stop={summary['stop_reason']} "
                  f"score={summary['score_at_end']} "
                  f"life={summary['frozen_life_first']}->"
                  f"{summary['frozen_life_min']} "
                  f"rc={summary['exit_code']}", flush=True)
            if not arm_record.get("engine_started"):
                break
        judgment = strength_judgment(arms=record["arms"])
        record["judgment"] = judgment
        record["control_reference_02c"] = strength_control_reference(out_root)
        started_arms = [arm for arm in STRENGTH_ARMS if arm in record["arms"]]
        entries = [record["arms"][arm]["entry_reached"] for arm in started_arms]
        technical = [arm for arm in started_arms
                     if record["arms"][arm]["terminal"] == "technical_conflict"
                     or record["arms"][arm]["exit_code"] not in (0, 1)]
        gate.add("both_arms_started", int(record["engine_starts"]) == len(STRENGTH_ARMS)
                 and len(started_arms) == len(STRENGTH_ARMS),
                 {"engine_starts": record["engine_starts"], "arms": started_arms})
        gate.add("both_arms_reached_the_entry", bool(entries) and all(entries),
                 {arm: record["arms"][arm]["entry_reached"] for arm in started_arms})
        gate.add("no_technical_conflict", not technical, {"technical_arms": technical})
        gate.add("open_loop_completed_at_least_once",
                 bool(judgment["completed_at_least_once"]),
                 judgment["completed_evidence"])
        gate.add("open_loop_not_worse_than_the_frozen_12",
                 bool(judgment["not_worse_than_12"]),
                 judgment["not_worse_evidence"])
        gate.add("strength_passed", bool(judgment["strength_passed"]),
                 {"completed_at_least_once":
                  judgment["completed_at_least_once"],
                  "not_worse_than_12": judgment["not_worse_than_12"]})
        gate.add("classification_is_the_fit_root_one",
                 record["classification"] == STRENGTH_CLASSIFICATION,
                 {"classification": record["classification"]})
        exit_code = EXIT_OK if judgment["strength_passed"] else EXIT_GATE_FAILED
        record["verdict"] = ("strength_passed" if judgment["strength_passed"]
                             else "strength_failed")
    except BudgetRefused as exc:
        record["budget"] = dict(exc.payload)
        record["verdict"] = "budget_refused"
        record["terminal"] = {"kind": "budget_refused", "reason": exc.reason,
                              "detail": exc.payload.get("exceeded")}
        gate.add("budget_checked", False, exc.payload.get("exceeded"))
        exit_code = EXIT_NOT_IMPLEMENTED
    except (EngineRefused, SourceError, FrameRelationConflict) as exc:
        record["verdict"] = "technical_invalid"
        record["terminal"] = {"kind": "technical_invalid",
                              "reason": type(exc).__name__, "detail": str(exc),
                              "engine_started": bool(record.get("engine_starts"))}
        gate.add("strength_inputs_readable", False, {"error": str(exc)})
        exit_code = EXIT_TECHNICAL
    except AdapterUnavailable as exc:
        record["verdict"] = "technical_invalid"
        record["terminal"] = {"kind": "technical_invalid",
                              "reason": type(exc).__name__, "detail": str(exc)}
        gate.add("adapter_available", False, {"error": str(exc)})
        exit_code = EXIT_TECHNICAL
    except Exception as exc:  # noqa: BLE001
        record["verdict"] = "technical_invalid"
        record["terminal"] = {"kind": "technical_invalid",
                              "reason": type(exc).__name__, "detail": str(exc),
                              "traceback": traceback.format_exc(),
                              "engine_started": bool(record.get("engine_starts"))}
        gate.add("strength_completed", False, {"error": str(exc)})
        exit_code = EXIT_TECHNICAL
    finally:
        meter.finish()
    record.setdefault("terminal", {"kind": "unknown"})
    record.setdefault("verdict", "strength_failed")
    record["finished_at"] = now_iso()
    record["cost"] = meter.payload(step_units=0)
    record["seconds"] = round(time.perf_counter() - started, 3)
    record["gate"] = gate.as_dict()
    target = Path(paths["result"])
    previous: dict[str, Any] = {}
    if target.is_file():
        try:
            previous = _load_json(target)
        except (OSError, ValueError):  # pragma: no cover - corrupt previous file
            previous = {}
    previous_arms = [arm for arm in STRENGTH_ARMS
                     if arm in (previous.get("arms") or {})]
    if previous_arms and not record.get("arms"):
        # An invocation that started nothing (a budget refusal, a missing input)
        # must never replace evidence that already carries the two arms: the stage
        # has exactly two starts and a re-run is refused by the budget.
        alternate = target.with_name(
            f"{target.stem}.{record.get('verdict') or 'failed'}-"
            f"{datetime.now().strftime('%Y%m%dT%H%M%S')}{target.suffix}")
        record["preserved_evidence"] = {
            "kept": target.as_posix(),
            "previous_arms": previous_arms,
            "previous_strength_passed": bool(
                (previous.get("judgment") or {}).get("strength_passed")),
            "this_run_written_to": alternate.as_posix(),
            "reason": ("an invocation that started nothing never replaces evidence "
                       "that already carries both arms"),
            "classification": "measured"}
        target = alternate
    path = write_json(target, record)
    for check in gate.checks:
        print(f"  {'ok  ' if check.ok else 'FAIL'} {check.name}")
    print(f"strength: seed={seed} starts={record.get('engine_starts')} "
          f"strength_passed={bool((record.get('judgment') or {}).get('strength_passed'))} "
          f"rc={exit_code}")
    print(f"wrote: {path}")
    return exit_code


def verify_03_strength(out_root: str | Path, repo: str | Path,
                       gate: Gate) -> dict[str, Any]:
    """Re-derive the 03 strength check from its own raw rows; never starts an engine.

    Recomputes the table digest, the two arms' masks from the frozen executor, the
    two outcomes from ``decisions.jsonl``, the action statistics, the mute reports,
    the source-fingerprint equality and the ledger rows — then re-applies the
    judgment rule to the recomputed numbers and compares it with the claim.
    """

    out_root = Path(out_root)
    repo = Path(repo)
    paths = strength_layout(out_root)
    if not Path(paths["result"]).is_file():
        return {"status": "not-run",
                "reason": "no 03/strength/result.json under this out dir"}
    claimed = _load_json(paths["result"])
    report: dict[str, Any] = {
        "status": "failed",
        "path": Path(paths["result"]).as_posix(),
        "result_sha256": sha256_file(paths["result"]),
        "claimed_verdict": claimed.get("verdict"),
        "claimed_judgment": {
            "completed_at_least_once": (claimed.get("judgment") or {}).get(
                "completed_at_least_once"),
            "not_worse_than_12": (claimed.get("judgment") or {}).get(
                "not_worse_than_12"),
            "strength_passed": (claimed.get("judgment") or {}).get(
                "strength_passed"),
        },
        "classification": claimed.get("classification"),
    }
    gate.add("03_strength_present",
             claimed.get("schema") == SCHEMA_STRENGTH
             and claimed.get("classification") == STRENGTH_CLASSIFICATION,
             {"schema": claimed.get("schema"),
              "classification": claimed.get("classification"),
              "expected": STRENGTH_CLASSIFICATION})
    frozen = read_task_card(Path(paths["card"]))
    card = dict(frozen["card"])
    entry = int(card["entry"]["prefix_decisions"])
    entry_frame = int(card["entry"]["frame"])
    frame_cap = int(card["frame_cap"]["frames"])
    decision_cap = int(card["frame_cap"]["decisions"])
    gate.add("03_strength_used_this_frozen_card",
             (claimed.get("task_card") or {}).get("sha256") == frozen["sha256"],
             {"card": frozen["sha256"],
              "claimed": (claimed.get("task_card") or {}).get("sha256")})
    try:
        table = strength_table_block(out_root)
        table_ok = True
        table_error = None
    except SourceError as exc:
        table = {}
        table_ok = False
        table_error = str(exc)
    claimed_table = claimed.get("table") or {}
    gate.add("03_strength_table_recomputed",
             table_ok
             and table.get("table_sha256") == claimed_table.get("table_sha256")
             and table.get("file_sha256") == claimed_table.get("file_sha256")
             and table.get("table_sha256") == STRENGTH_REQUIRED_TABLE_SHA256,
             {"recomputed_table_sha256": table.get("table_sha256"),
              "claimed_table_sha256": claimed_table.get("table_sha256"),
              "recomputed_file_sha256": table.get("file_sha256"),
              "claimed_file_sha256": claimed_table.get("file_sha256"),
              "required": STRENGTH_REQUIRED_TABLE_SHA256,
              "error": table_error})
    cands_by_arm = {arm: strength_arm_candidates(arm, table.get("cands") or [])
                    for arm in STRENGTH_ARMS} if table_ok else {}
    arms_report: dict[str, Any] = {}
    recomputed_arms: dict[str, Any] = {}
    arms_ok = True
    ledger_rows = ([dict(row) for row in iter_jsonl(Path(out_root) / "ledger.jsonl")]
                   if (Path(out_root) / "ledger.jsonl").is_file() else [])
    accounted: dict[str, Any] = {}
    for row in ledger_rows:
        attempt = str(row.get("attempt_id") or "")
        if not attempt.startswith(STRENGTH_ATTEMPT_PREFIXES):
            continue
        block = accounted.setdefault(attempt, {"reserve": 0, "settle": 0,
                                               "actual": {}, "requested": {}})
        if row.get("phase") == "reserve":
            block["reserve"] += 1
            block["requested"] = dict(row.get("requested") or {})
        elif row.get("phase") == "settle":
            block["settle"] += 1
            block["actual"] = dict(row.get("actual") or {})
    for arm in STRENGTH_ARMS:
        claim = dict((claimed.get("arms") or {}).get(arm) or {})
        detail: dict[str, Any] = {"arm": arm, "claimed_outcome": claim.get("outcome")}
        ok = bool(claim)
        result_path = Path(claim.get("result") or "")
        detail["result"] = result_path.as_posix()
        detail["result_exists"] = result_path.is_file()
        ok = ok and result_path.is_file()
        if result_path.is_file():
            arm_record = _load_json(result_path)
            detail["result_sha256"] = sha256_file(result_path)
            detail["result_sha256_matches"] = (
                detail["result_sha256"] == claim.get("result_sha256"))
            ok = ok and detail["result_sha256_matches"]
            decisions_path = Path(arm_record.get("decisions_path") or "")
            rows = ([dict(row) for row in iter_jsonl(decisions_path)]
                    if decisions_path.is_file() else [])
            detail["decisions"] = decisions_path.as_posix()
            detail["row_count"] = len(rows)
            detail["decisions_sha256_matches"] = bool(
                decisions_path.is_file()
                and sha256_file(decisions_path) == arm_record.get(
                    "decisions_sha256"))
            claimed_rows = int((arm_record.get("counts") or {}).get(
                "control_decisions", -1))
            detail["claimed_row_count"] = claimed_rows
            ok = (ok and detail["decisions_sha256_matches"]
                  and len(rows) == claimed_rows)
            # 1. the masks, from the frozen executor and this arm's own candidates
            want = (pilot_table_masks(cands_by_arm[arm], entry)
                    if arm in cands_by_arm else [])
            masks_ok = bool(want) and all(
                [int(row["masks"][0]), int(row["masks"][1])] == want[
                    int(row["window_index"])]
                for row in rows
                if int(row["window_index"]) < len(want))
            detail["masks_recomputed"] = masks_ok
            if arm == STRENGTH_ARM_OPEN_LOOP and table_ok:
                detail["matches_the_frozen_table"] = (
                    [[int(pair[0]), int(pair[1])] for pair in table["masks"]]
                    == want)
                masks_ok = masks_ok and detail["matches_the_frozen_table"]
            ok = ok and masks_ok
            # 2. the outcome, from the raw rows and the card's priority
            generation = arm_record.get("frozen_generation") or {}
            tracker = FrozenTracker(
                slot=int(generation.get("slot", -1)),
                spawn_decision=int(generation.get("spawn_decision_from_card", -1)),
                life_at_spawn=int(generation.get("life_at_spawn_from_card", -1)),
                entry_decision=int((arm_record.get("entry") or {}).get(
                    "card_decision", -1)))
            deaths_at_entry = float(((arm_record.get("resources_at_entry") or {}
                                      ).get("state") or {}).get("deaths", 0.0))
            recomputed_outcome: str | None = None
            for row in rows:
                observed = tracker.observe(row)
                if int(row["hud"]["game_over"]):
                    recomputed_outcome = "game_over"
                    break
                if float(row["state"]["deaths"]) > deaths_at_entry:
                    recomputed_outcome = "legal_death"
                    break
                if observed is not None:
                    recomputed_outcome = observed
                    break
            if recomputed_outcome is None:
                recomputed_outcome = ("not_removed_within_cap" if tracker.seen
                                      else "generation_not_observed")
            detail["recomputed_outcome"] = recomputed_outcome
            detail["outcome_matches"] = recomputed_outcome == arm_record.get(
                "outcome")
            ok = ok and detail["outcome_matches"]
            # 3. the action statistics, from the same rows
            fresh_stats = strength_action_stats(
                cands=[int(row["cand"]) for row in rows],
                masks=[[int(row["masks"][0]), int(row["masks"][1])] for row in rows],
                first_decision=entry)
            detail["action_stats_match"] = bool(
                fresh_stats == (arm_record.get("action_stats") or {}))
            detail["action_stats_match_claim"] = bool(
                fresh_stats == (claim.get("action_stats") or {}))
            ok = ok and detail["action_stats_match"] and detail[
                "action_stats_match_claim"]
            # 4. frames, resources, the mute report and the source copy
            counts = arm_record.get("counts") or {}
            detail["engine_frames"] = counts.get("engine_frames")
            detail["frames_match"] = bool(
                int(counts.get("engine_frames", -1))
                == int(counts.get("prefix_frames", -2))
                + int(counts.get("control_frames", -3))
                and int(counts.get("control_decisions", -1)) == len(rows)
                and int(counts.get("prefix_frames", -1)) == entry_frame)
            ok = ok and detail["frames_match"]
            mute_path = result_path.parent / "mute.json"
            mute = _load_json(mute_path) if mute_path.is_file() else {}
            post = ((mute.get("copy_after_exit") or {}).get("fields")) or {}
            detail["mute_recomputed"] = bool(
                mute_path.is_file()
                and all(int(post.get(name, -1)) == 0 for name in
                        ("musicMode", "playSounds", "musicVolume", "sfxVolume"))
                and not (mute.get("problems") or []))
            detail["mute_path"] = mute_path.as_posix()
            source = arm_record.get("source_dir") or {}
            detail["source_dir_unchanged"] = bool(
                (source.get("unchanged") is True)
                and ((source.get("before") or {}).get("digest")
                     == (source.get("after") or {}).get("digest")))
            ok = ok and detail["mute_recomputed"] and detail[
                "source_dir_unchanged"]
            attempt = str(arm_record.get("attempt_id") or "")
            block = accounted.get(attempt) or {}
            detail["ledger"] = {"attempt_id": attempt,
                                "reserve_rows": block.get("reserve"),
                                "settle_rows": block.get("settle"),
                                "actual": block.get("actual")}
            detail["ledger_settled"] = bool(
                block.get("reserve") == 1 and block.get("settle") == 1
                and float((block.get("actual") or {}).get("starts", 0) or 0) == 1.0
                and float((block.get("actual") or {}).get("engine_frames", -1))
                == float(counts.get("engine_frames", -2)))
            ok = ok and detail["ledger_settled"]
            recomputed_arms[arm] = {
                "arm": arm,
                "outcome": recomputed_outcome,
                "score_at_end": ((arm_record.get("resources_at_end") or {}).get(
                    "state") or {}).get("score"),
                "frozen_life_drop": (
                    None if not tracker.life_series
                    else float(int(tracker.life_series[0])
                               - int(min(tracker.life_series)))),
                "frozen_life_first": (tracker.life_series[0]
                                      if tracker.life_series else None),
                "frozen_life_min": (min(tracker.life_series)
                                    if tracker.life_series else None),
                "terminal": (arm_record.get("terminal") or {}).get("kind"),
                "stop_reason": arm_record.get("stop_reason"),
                "outcome_decision": tracker.outcome_decision,
                "outcome_frame": tracker.outcome_frame,
                "exit_code": int(arm_record.get("arm_exit_code", -1)),
            }
        arms_report[arm] = detail
        gate.add(f"03_strength_arm_{arm}_recomputed", ok, detail)
        arms_ok = arms_ok and ok
    # 5. the judgment, re-applied to the recomputed numbers
    fresh = strength_judgment(arms=recomputed_arms) if recomputed_arms else {}
    claim_j = claimed.get("judgment") or {}
    judgment_ok = bool(fresh) and all(
        bool(fresh.get(key)) == bool(claim_j.get(key)) for key in
        ("completed_at_least_once", "not_worse_than_12", "strength_passed"))
    gate.add("03_strength_judgment_recomputed", judgment_ok,
             {"recomputed": {key: fresh.get(key) for key in
                             ("completed_at_least_once", "not_worse_than_12",
                              "strength_passed")},
              "claimed": {key: claim_j.get(key) for key in
                          ("completed_at_least_once", "not_worse_than_12",
                           "strength_passed")},
              "recomputed_arms": {arm: {"outcome": value.get("outcome"),
                                        "score_at_end": value.get("score_at_end"),
                                        "frozen_life_drop": value.get(
                                            "frozen_life_drop")}
                                  for arm, value in recomputed_arms.items()},
              "rule": fresh.get("rule")})
    gate.add("03_strength_verdict_matches_the_gate",
             bool((claimed.get("gate") or {}).get("passed"))
             == bool(claim_j.get("strength_passed")),
             {"gate_passed": (claimed.get("gate") or {}).get("passed"),
              "strength_passed": claim_j.get("strength_passed")})
    report.update({"arms": arms_report,
                   "table": {"recomputed_table_sha256": table.get("table_sha256"),
                             "checks": table.get("checks"),
                             "error": table_error},
                   "judgment": {"recomputed": fresh, "claimed": claim_j},
                   "ledger": {name: {key: value for key, value in block.items()}
                              for name, block in accounted.items()},
                   "status": ("verified" if (arms_ok and table_ok and judgment_ok)
                              else "failed")})
    return report


# ---------------------------------------------------------------------------
# `diag-teacher`: is the frozen task seed-12-only, or is the policy the problem?
# ---------------------------------------------------------------------------
#: 02-C / 03 left one question open.  The four unexposed development roots
#: (1020-1023) all reached the frozen entry, but **none** of the three pilot arms
#: completed the frozen event: they lost a life or were still shooting a living
#: boss when the window closed, and their candidate histogram was (near-)constant.
#: A degraded arm cannot separate the two explanations:
#:
#: (a) the task/entry is not reachable on these seeds, so the card is seed-12
#:     specific — the teacher would fail there too;
#: (b) the three (near-constant) arms are degenerate — the teacher would succeed.
#:
#: This block adds **one** measurement and nothing else: 02-C's teacher
#: continuation (``recorded_tape_masks``, T04's recorded decisions
#: ``1543..1842``, the card's own ``teacher_continuation``) replayed from the same
#: frozen entry, once per root.  It writes ``03/diag/teacher_arm/`` only.
SCHEMA_DIAG_TEACHER = "v831-03-diag-teacher/1"
SCHEMA_DIAG_TEACHER_ARM = "v831-03-diag-teacher-arm/1"
SCHEMA_DIAG_TEACHER_ROW = "v831-03-diag-teacher-decision/1"
DIAG_TEACHER_CLASSIFICATION = "teacher_arm_diagnostic_not_a_pilot_result"
DIAG_TEACHER_TICKET = ("V831-SLICE-02 follow-up on the V831-PILOT-03 evidence "
                       "(TRAINING_HOST_REQUEST_02.md#L42-L52)")
DIAG_TEACHER_ARM = "teacher"
DIAG_TEACHER_CONTROLLER = "recorded_tape_masks"
#: The pilot's own four development roots: the diagnostic reuses that
#: registration and adds none.
DIAG_TEACHER_SEEDS: tuple[int, ...] = tuple(int(seed) for seed in PILOT_DEV_SEEDS)
#: This diagnostic's own allowance, fixed: one start per root (4), each
#: ``entry_frame + frame_cap = 3,686`` frames, so 14,744 under the 15,000-frame
#: ceiling.  The CLI may narrow it and can never widen it.
DIAG_TEACHER_STAGE_STARTS = 4
DIAG_TEACHER_STAGE_FRAMES = 15_000
DIAG_TEACHER_ATTEMPT_PREFIX = "03-diag-teacher-"
DIAG_TEACHER_RULE = (
    "one engine start per unexposed root; from the frozen entry (frame 3,086, the "
    "card's decisions 0..1542) the arm plays the card's own teacher continuation "
    "(decisions 1,543..1,842, 600 frames) with the frozen two-frame executor and "
    "release_due(i, 48); the frozen generation (slot 1, BOSS, spawn decision "
    "1,466) is tracked on every decision and the card's termination priority is "
    "applied unchanged.  The card's entry hash was measured on seed 12, so it is "
    "recorded and compared but it is **not** an arrival predicate here: the "
    "arrival predicate is the pilot's own (frame 3,086, stage 0, alive, slot "
    "present).")


def diag_teacher_layout(out_root: str | Path) -> dict[str, Any]:
    """Where the diagnostic writes: ``03/diag/teacher_arm/`` and nothing else."""

    root = Path(out_root) / PILOT_WINDOW_KEY
    base = root / "diag" / "teacher_arm"
    return {
        "out": Path(out_root),
        "root": root,
        "base": base,
        "result": base / "result.json",
        "seeds": {int(seed): base / str(int(seed))
                  for seed in DIAG_TEACHER_SEEDS},
        "card": Path(out_root) / "02b",
        "card_json": Path(out_root) / "02b" / "task_card.json",
        "pilot": root / "pilot.json",
        "intervene": Path(out_root) / "02c" / "intervene.json",
        "ledger": Path(out_root) / "ledger.jsonl",
    }


def diag_teacher_stage_cap(*, max_starts: Any = None,
                           max_engine_frames: Any = None) -> dict[str, Any]:
    """The diagnostic's fixed allowance (4 starts / 15,000 frames), never widened.

    ``--max-starts``/``--max-engine-frames`` restate the **whole-ticket** limits
    everywhere else in this tool; here they may only narrow this diagnostic's own
    allowance, so a run can never spend more than the four roots' frames on it.
    """

    def _effective(default: int, value: Any) -> float:
        if value is None:
            return float(default)
        return float(min(float(default), float(value)))

    starts = _effective(DIAG_TEACHER_STAGE_STARTS, max_starts)
    frames = _effective(DIAG_TEACHER_STAGE_FRAMES, max_engine_frames)
    narrowed = (starts < float(DIAG_TEACHER_STAGE_STARTS)
                or frames < float(DIAG_TEACHER_STAGE_FRAMES))
    return {
        "starts": starts,
        "engine_frames": frames,
        "fixed_cap": {"starts": DIAG_TEACHER_STAGE_STARTS,
                      "engine_frames": DIAG_TEACHER_STAGE_FRAMES},
        "cli_asked": {"max_starts": max_starts,
                      "max_engine_frames": max_engine_frames},
        "narrowed_by_the_cli": bool(narrowed),
        "note": ("the diagnostic is fixed at 4 starts / 15,000 frames; the CLI "
                 "may only narrow it"),
        "classification": "measured",
    }


def diag_teacher_window(*, card: Mapping[str, Any],
                        tape: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen teacher continuation: the card's decisions ``entry..entry+299``.

    Three independent statements must agree before a frame is spent: the digest of
    ``"<decision>:<cand>:<mask1>:<mask2>\\n"`` over the window (the digest the card
    froze as ``teacher_continuation.masks_sha256``), the masks the frozen executor
    expands from the same candidates (``commit_frames`` + ``release_due(i, 48)``,
    i.e. :func:`pilot_table_masks`), and the T04 masks themselves.
    """

    entry = int(card["entry"]["prefix_decisions"])
    decision_cap = int(card["frame_cap"]["decisions"])
    last = entry + decision_cap
    if last > len(tape["cands"]) or last > len(tape["masks"]):
        raise SourceError(
            f"the teacher tape has {len(tape['cands'])} decisions; the card's "
            f"window {entry}..{last - 1} does not fit")
    cands = [int(value) for value in tape["cands"][entry:last]]
    masks = [[int(pair[0]), int(pair[1])] for pair in tape["masks"][entry:last]]
    executor = pilot_table_masks(cands, entry)
    digest = tape_digest(tape, entry, last)
    card_digest = str((card.get("teacher_continuation") or {}).get(
        "masks_sha256") or "")
    stats = strength_action_stats(cands=cands, masks=masks,
                                  first_decision=entry)
    return {
        "first_decision": int(entry),
        "last_decision": int(last - 1),
        "decisions": len(cands),
        "frames": DECISION_FRAMES * len(cands),
        "controller": DIAG_TEACHER_CONTROLLER,
        "source": ("T04's recorded schedule, decisions entry..entry+299: the "
                   "card's own teacher_continuation (02-C arm T's controller)"),
        "masks_sha256": digest,
        "masks_sha256_card": card_digest,
        "masks_sha256_matches_the_card": bool(card_digest) and digest == card_digest,
        "masks_are_the_frozen_executor_expansion": masks == executor,
        "executor_masks_sha256": sha256_bytes(canonical_json(
            [[int(pair[0]), int(pair[1])] for pair in executor]).encode()),
        "cands_sha256": table_sha256(cands),
        "cands": cands,
        "masks": masks,
        "action_stats": stats,
        "classification": "measured",
    }


def diag_teacher_pilot_reference(out_root: str | Path) -> dict[str, Any]:
    """Read the pilot's per-root three-arm outcome column; **never** recompute it.

    The contrast this diagnostic exists for is "the teacher on root X vs the three
    arms the pilot already ran on root X", so the pilot's numbers are quoted from
    ``03/pilot.json`` exactly as written (with its file sha256), not re-derived
    from the arms' raw rows.
    """

    paths = diag_teacher_layout(out_root)
    path = Path(paths["pilot"])
    if not path.is_file():
        raise SourceError(
            f"the 03 pilot summary is missing: {path}; the diagnostic quotes its "
            "per-root three-arm outcome column")
    summary = _load_json(path)
    per_root = (summary.get("comparison") or {}).get("per_root") or {}
    arms_summary = summary.get("arms") or {}
    roots: dict[str, Any] = {}
    completed: list[int] = []
    for seed in DIAG_TEACHER_SEEDS:
        key = str(int(seed))
        block = per_root.get(key) or {}
        arms = block.get("arms") or {}
        seed_arms = arms_summary.get(key) or {}
        outcomes = {arm: (arms.get(arm) or {}).get("outcome")
                    for arm in PILOT_ARMS}
        present = [arm for arm in PILOT_ARMS if arm in arms]
        kill_arms = [arm for arm in PILOT_ARMS
                     if outcomes.get(arm) in KILL_OUTCOMES]
        if kill_arms:
            completed.append(int(seed))
        roots[key] = {
            "seed": int(seed),
            "present": bool(block),
            "arms_present": present,
            "outcome_by_arm": outcomes,
            "outcome_column": "; ".join(
                f"{arm}={outcomes.get(arm)}" for arm in PILOT_ARMS),
            "completed_arms": kill_arms,
            "control_decisions": {arm: (arms.get(arm) or {}).get(
                "control_decisions") for arm in PILOT_ARMS},
            "deaths_observed": {arm: (arms.get(arm) or {}).get(
                "deaths_observed") for arm in PILOT_ARMS},
            "frozen_life_first": {arm: (arms.get(arm) or {}).get(
                "frozen_life_first") for arm in PILOT_ARMS},
            "frozen_life_min": {arm: (arms.get(arm) or {}).get(
                "frozen_life_min") for arm in PILOT_ARMS},
            "candidate_dominant_share": {
                arm: (seed_arms.get(arm) or {}).get("candidate_dominant_share")
                for arm in PILOT_ARMS},
            "candidate_histogram": {
                arm: (seed_arms.get(arm) or {}).get("candidate_histogram")
                for arm in PILOT_ARMS},
            "source": ("03/pilot.json comparison.per_root[...].arms (read here, "
                       "never recomputed)"),
            "classification": "historical",
        }
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "schema": summary.get("schema"),
        "verdict": summary.get("verdict"),
        "gate_passed": bool((summary.get("gate") or {}).get("passed")),
        "seeds": [int(seed) for seed in summary.get("seeds") or []],
        "roots": roots,
        "completed_roots": completed,
        "read_only": True,
        "note": ("the pilot's three arms are quoted, not re-run and not "
                 "re-derived: this diagnostic starts four engines and no more"),
        "classification": "historical",
    }


def diag_teacher_interpretation(*, teacher_roots: Mapping[str, Mapping[str, Any]],
                                pilot_reference: Mapping[str, Any]
                                ) -> dict[str, Any]:
    """Which of the two explanations the four measurements support.

    The rule is the ticket's own: at least one ``kill_*`` root for the teacher
    means the task is completable there and the pilot's arms were the problem
    (``policy_quality``); zero ``kill_*`` roots means the teacher cannot complete
    it either, so the frozen task/entry is seed-12 specific
    (``entry_or_task_seed_specific``).
    """

    teacher_completed = sorted(
        int(key) for key, value in teacher_roots.items()
        if (value or {}).get("outcome") in KILL_OUTCOMES)
    concluded = sorted(
        int(key) for key, value in teacher_roots.items()
        if (value or {}).get("conclusion") == "measured")
    pilot_completed = sorted(int(seed) for seed in
                             (pilot_reference.get("completed_roots") or []))
    if teacher_completed:
        interpretation = "policy_quality"
        reading = ("the teacher completed the frozen event on at least one of the "
                   "unexposed roots, so the entry and the task are reachable "
                   "there; the pilot's three arms failing is a policy-quality "
                   "result, not a task-card defect")
    else:
        interpretation = "entry_or_task_seed_specific"
        reading = ("the teacher completed the frozen event on none of the "
                   "unexposed roots, so the frozen task card (the entry, the "
                   "window and the kill it asks for) is specific to seed 12; the "
                   "pilot's three arms never had a reachable target there")
    return {
        "teacher_completed_roots": teacher_completed,
        "teacher_concluded_roots": concluded,
        "pilot_completed_roots": pilot_completed,
        "interpretation": interpretation,
        "reading": reading,
        "explanation_a": ("(a) the task/entry is not reachable on these seeds: "
                          "the teacher fails too, so the card is seed-12 "
                          "specific — `entry_or_task_seed_specific`"),
        "explanation_b": ("(b) the three (near-constant) arms are degenerate: the "
                          "teacher succeeds — `policy_quality`"),
        "rule": ("teacher_completed_roots is non-empty -> policy_quality; empty "
                 "-> entry_or_task_seed_specific.  A `kill_*` outcome is the "
                 "card's own completion definition (kill_M0/M1/M2 inside the "
                 "600-frame window)."),
        "classification": DIAG_TEACHER_CLASSIFICATION,
    }


def diag_teacher_row(record: Mapping[str, Any], *,
                     pilot: Mapping[str, Any] | None = None
                     ) -> dict[str, Any]:
    """The per-root row the summary quotes (and ``verify`` re-derives)."""

    entry = record.get("entry") or {}
    counts = record.get("counts") or {}
    start = record.get("resources_at_entry") or {}
    end = record.get("resources_at_end") or {}
    frozen = record.get("frozen_generation") or {}
    window = record.get("window") or {}
    stats = record.get("action_stats") or {}
    terminal = (record.get("terminal") or {}).get("kind")
    outcome = record.get("outcome")
    if record.get("engine_started") and outcome:
        conclusion = "measured"
    elif terminal == "not_started_budget":
        conclusion = "not_started_budget"
    else:
        conclusion = "incomplete"
    row = {
        "seed": int(record.get("seed") or 0),
        "attempt_id": record.get("attempt_id"),
        "engine_started": bool(record.get("engine_started")),
        "controller": record.get("controller"),
        "entry_reached": bool(entry.get("reached")),
        "entry_ok": bool(entry.get("reached")) and bool(
            entry.get("frozen_slot_present")),
        "entry_frame": entry.get("frame"),
        "entry_stage": entry.get("stage"),
        "entry_dead": entry.get("dead"),
        "entry_hash": entry.get("hash"),
        "card_entry_hash": entry.get("card_hash"),
        "card_entry_hash_matches": entry.get("card_hash_matches"),
        "entry_note": entry.get("card_hash_note"),
        "frozen_slot_present": entry.get("frozen_slot_present"),
        "frozen_slot_boss": entry.get("frozen_slot_boss"),
        "frozen_slot_life_at_entry": entry.get("frozen_slot_life"),
        "observed_spawn_decision": entry.get("observed_spawn_decision"),
        "card_spawn_decision": entry.get("card_spawn_decision"),
        "spawn_delta": entry.get("spawn_delta"),
        "identity_mismatch": entry.get("identity_mismatch"),
        "prefix_decisions": counts.get("prefix_decisions"),
        "prefix_frames": counts.get("prefix_frames"),
        "control_decisions": counts.get("control_decisions"),
        "control_frames": counts.get("control_frames"),
        "engine_frames": counts.get("engine_frames"),
        "window_decision_range": window.get("decision_range"),
        "window_frame_range": window.get("frame_range"),
        "stop_reason": record.get("stop_reason"),
        "outcome": outcome,
        "completed": outcome in KILL_OUTCOMES,
        "outcome_decision": record.get("outcome_decision"),
        "outcome_frame": record.get("outcome_frame"),
        "terminal": terminal,
        "deaths_observed": record.get("deaths_observed"),
        "lives_lost": record.get("lives_lost"),
        "lives_at_entry": (start.get("state") or {}).get("lives_remaining"),
        "lives_at_end": (end.get("state") or {}).get("lives_remaining"),
        "score_at_entry": (start.get("state") or {}).get("score"),
        "score_at_end": (end.get("state") or {}).get("score"),
        "power_at_entry": (start.get("state") or {}).get("player_power"),
        "power_at_end": (end.get("state") or {}).get("player_power"),
        "graze_at_end": (end.get("state") or {}).get("graze"),
        "frozen_life_first": frozen.get("life_first"),
        "frozen_life_last": frozen.get("life_last"),
        "frozen_life_min": frozen.get("life_min"),
        "frozen_life_drop": (
            None if frozen.get("life_first") is None
            or frozen.get("life_min") is None
            else float(int(frozen["life_first"]) - int(frozen["life_min"]))),
        "frozen_observations": frozen.get("observations"),
        "frozen_phase_ends": len(frozen.get("phase_ends") or []),
        "candidate_histogram": record.get("candidate_histogram"),
        "candidate_dominant_share": record.get("candidate_dominant_share"),
        "distinct_candidates": stats.get("distinct_candidates"),
        "longest_same_candidate_run": stats.get("longest_same_candidate_run"),
        "releases": (stats.get("releases") or {}).get("count"),
        "exit_code": record.get("arm_exit_code"),
        "result": record.get("result_path"),
        "result_sha256": record.get("result_sha256"),
        "mute_problems": ((record.get("mute") or {}).get("after_exit_problems")),
        "ledger_actual": ((record.get("ledger") or {}).get("actual")),
        "conclusion": conclusion,
        "classification": record.get("classification"),
    }
    if pilot is not None:
        row["pilot_same_root"] = dict(pilot)
        row["pilot_same_root_outcome_column"] = pilot.get("outcome_column")
    return row


def diag_teacher_attempt_ids(out_root: str | Path) -> set[str]:
    """The ``03-diag-teacher-*`` attempt ids the diagnostic's own evidence accounts for."""

    path = Path(diag_teacher_layout(out_root)["result"])
    if not path.is_file():
        return set()
    try:
        record = _load_json(path)
    except (OSError, ValueError):
        return set()
    return {str(row.get("attempt_id")) for row in (record.get("roots") or {}).values()
            if (row or {}).get("engine_started") and row.get("attempt_id")}


def diag_teacher_unstarted_record(*, seed: int, frame_plan: Mapping[str, Any],
                                 budget: Mapping[str, Any]) -> dict[str, Any]:
    """One root the budget did not allow to start: a named conclusion, not a run."""

    return {
        "schema": SCHEMA_DIAG_TEACHER_ARM,
        "created_at": now_iso(),
        "classification": DIAG_TEACHER_CLASSIFICATION,
        "seed": int(seed),
        "root_seed": int(seed),
        "arm": DIAG_TEACHER_ARM,
        "attempt_id": None,
        "controller": DIAG_TEACHER_CONTROLLER,
        "engine_started": False,
        "frame_plan": dict(frame_plan),
        "budget": dict(budget),
        "counts": {"prefix_decisions": 0, "prefix_frames": 0,
                   "control_decisions": 0, "control_frames": 0,
                   "engine_frames": 0},
        "terminal": {
            "kind": "not_started_budget",
            "detail": ("the shared ledger has no room for this root's start; the "
                       "roots are taken in seed order and this one was not started"),
            "engine_started": False},
        "arm_exit_code": EXIT_NOT_IMPLEMENTED,
    }


def run_diag_teacher_arm(*, args: argparse.Namespace, out_root: Path, repo: Path,
                         card: Mapping[str, Any], tape: Mapping[str, Any],
                         window: Mapping[str, Any], paths: Mapping[str, Any],
                         seed: int, limits: BudgetLimits,
                         meter: Any) -> dict[str, Any]:
    """One engine start: the frozen prefix, then the teacher continuation.

    Nothing about the executor is new.  The prefix is :func:`_pilot_prefix_walk`
    (the card's decisions ``0..1542``, one ``STEP mask 1`` per frame with
    ``EVENTS`` after every frame — the same function the pilot's three arms and
    the 03 strength check walk); the window is :func:`_step_one` +
    :func:`read_decision` + :class:`FrozenTracker` (the same 02-B/02-C reader and
    the same frozen-identity rule table); the masks are the card's own
    ``teacher_continuation`` — 02-C arm T's ``recorded_tape_masks``; and the
    terminal priority is the card's own.  What is new is only the root each start
    resets and the fact that the card's entry hash is quoted, not required.
    """

    entry = int(card["entry"]["prefix_decisions"])
    entry_frame = int(card["entry"]["frame"])
    decision_cap = int(card["frame_cap"]["decisions"])
    frame_cap = int(card["frame_cap"]["frames"])
    slot = int(card["event"]["seed12_instance"]["slot"])
    card_spawn = int(card["event"]["seed12_instance"]["spawn_decision"])
    card_life = int(card["event"]["seed12_instance"]["life_at_spawn"])
    cands = [int(value) for value in window["cands"]]
    masks = [[int(pair[0]), int(pair[1])] for pair in window["masks"]]
    if len(cands) != decision_cap:
        raise EngineRefused(
            f"the teacher continuation supplies {len(cands)} decisions, the frozen "
            f"card's window is {decision_cap}; the window must be covered")
    base = Path(paths["seeds"][int(seed)])
    base.mkdir(parents=True, exist_ok=True)
    attempt_id = f"{DIAG_TEACHER_ATTEMPT_PREFIX}{int(seed)}"
    planned = entry_frame + frame_cap
    attempt = EngineAttempt(
        args=args, out_root=out_root, base_dir=base, attempt_id=attempt_id,
        planned_frames=int(planned),
        note=(f"03 diag-teacher on dev root seed {seed}: frozen prefix decisions "
              f"0..{entry - 1} (frames 1..{entry_frame}), then the recorded teacher "
              f"continuation decisions {entry}..{entry + decision_cap - 1} "
              f"({decision_cap} decisions / {frame_cap} frames)"),
        limits=limits, meter=meter, reset_seed=int(seed))
    record: dict[str, Any] = {
        "schema": SCHEMA_DIAG_TEACHER_ARM,
        "created_at": now_iso(),
        "classification": DIAG_TEACHER_CLASSIFICATION,
        "ticket": DIAG_TEACHER_TICKET,
        "diagnostic": {"arm": DIAG_TEACHER_ARM,
                       "question": ("can the frozen entry be completed on this "
                                    "unexposed root by the teacher continuation "
                                    "the pilot's arms replaced?"),
                       "not_a_pilot_result": True},
        "seed": int(seed),
        "root_seed": int(seed),
        "arm": DIAG_TEACHER_ARM,
        "attempt_id": attempt_id,
        "controller": DIAG_TEACHER_CONTROLLER,
        "repo": Path(repo).as_posix(),
        "out": Path(out_root).as_posix(),
        "base": base.as_posix(),
        "host": host_identity(),
        "heads": git_identity(repo),
        "tool": {"path": TOOL_PATH.as_posix(), "sha256": sha256_file(TOOL_PATH)},
        "frame_plan": {"entry_frame": entry_frame, "frame_cap": frame_cap,
                       "decision_cap": decision_cap,
                       "planned_frames": int(planned)},
        "engine_started": False,
        "entry": {"card_decision": entry, "card_frame": entry_frame,
                  "card_hash": card["entry"]["hash"], "reached": False,
                  "hash": None, "frame": None, "stage": None, "dead": None,
                  "card_hash_matches": None,
                  "card_hash_note": (
                      "the card's entry hash was measured on seed 12; this root is "
                      f"seed {int(seed)}, so the hash is recorded and compared but "
                      "it is not an arrival predicate here"),
                  "frozen_slot_present": None, "identity_mismatch": None},
        "counts": {"prefix_decisions": 0, "prefix_frames": 0,
                   "control_decisions": 0, "control_frames": 0,
                   "engine_frames": 0},
        "terminal": {"kind": "unknown"},
    }
    exit_code = EXIT_TECHNICAL
    closed: dict[str, Any] = {}
    try:
        opened = attempt.open()
        record["engine_started"] = True
        record.update({"preflight": opened["preflight"],
                       "private_game_dir": opened["private"],
                       "reset": opened["reset"],
                       "engine_session": opened["session"],
                       "engine_env": attempt.module.engine_env_record()})
        prefix = _pilot_prefix_walk(
            attempt=attempt, tape=tape, entry_decision=entry,
            entry_frame=entry_frame, slot=slot, buffer=None,
            rows_path=base / "prefix.jsonl")
        record["counts"].update({"prefix_decisions": prefix["decisions"],
                                 "prefix_frames": prefix["frames"]})
        record["prefix"] = {
            "decisions": prefix["decisions"], "frames": prefix["frames"],
            "rows": (base / "prefix.jsonl").as_posix(),
            "prefix_masks_sha256_recomputed": prefix.get(
                "prefix_digest_recomputed"),
            "prefix_masks_sha256_card": card["entry"]["prefix_masks_sha256"],
            "prefix_masks_sha256_matches": bool(
                prefix.get("prefix_digest_recomputed")
                == card["entry"]["prefix_masks_sha256"]),
            "spawn_transitions": prefix["spawns"],
            "shortfall": prefix["shortfall"],
            "classification": "measured"}
        record["window_masks"] = {
            "first_decision": int(window["first_decision"]),
            "last_decision": int(window["last_decision"]),
            "controller": DIAG_TEACHER_CONTROLLER,
            "masks_sha256_recomputed": window["masks_sha256"],
            "masks_sha256_card": window["masks_sha256_card"],
            "masks_sha256_matches_the_card": bool(
                window["masks_sha256_matches_the_card"]),
            "masks_are_the_frozen_executor_expansion": bool(
                window["masks_are_the_frozen_executor_expansion"]),
            "cands_sha256": window["cands_sha256"],
            "executor_masks_sha256": window["executor_masks_sha256"],
            "classification": "measured"}
        if prefix["shortfall"] is not None:
            short = prefix["shortfall"]
            record["terminal"] = {
                "kind": "entry_not_reached",
                "detail": (f"the STEP into frame {short['frame']} returned "
                           f"advanced={short['advanced']} exit={short['exit']} "
                           f"game_over={short['game_over']}"),
                "shortfall": short, "engine_started": True}
            exit_code = EXIT_GATE_FAILED
            raise _PilotStop(record["terminal"])
        entry_reads = read_decision(attempt.host)
        state = engine_state_record(entry_reads["state"])
        hud = hud_record(entry_reads["hud"])
        player = player_record(entry_reads["player"])
        entry_rows = entry_reads["hazards"]["enemies"]
        frozen_rows = [item for item in entry_rows if int(item["slot"]) == slot]
        frozen_row = (max(frozen_rows, key=lambda item: int(item["life"]))
                      if frozen_rows else None)
        spawns = [item for item in prefix["spawns"]
                  if item.get("kind") != "life_jump_up"]
        observed_spawn = int(spawns[-1]["decision_index"]) if spawns else None
        identity_mismatch = (observed_spawn is None
                             or abs(observed_spawn - card_spawn) > 4)
        dead = bool(state["lives_remaining"] <= 0 or hud["game_over"]
                    or player["player_state_name"] == "DYING")
        record["entry"].update({
            "hash": prefix["hash_at_entry"],
            "frame": int(state["frame"]),
            "stage": int(state["stage"]),
            "dead": dead,
            "reached": bool(state["frame"] == entry_frame and not dead
                            and int(state["stage"]) == 0),
            "card_hash_matches": bool(str(prefix["hash_at_entry"])
                                      == str(card["entry"]["hash"])),
            "frozen_slot_present": frozen_row is not None,
            "frozen_slot_boss": None if frozen_row is None else bool(
                frozen_row["boss"]),
            "frozen_slot_life": None if frozen_row is None else int(
                frozen_row["life"]),
            "frozen_slot_flags1": None if frozen_row is None else str(
                frozen_row["flags1"]),
            "observed_spawn_decision": observed_spawn,
            "card_spawn_decision": card_spawn,
            "spawn_delta": (None if observed_spawn is None
                            else int(observed_spawn) - card_spawn),
            "identity_mismatch": bool(identity_mismatch),
            "classification": "measured"})
        record["resources_at_entry"] = {
            "source": (f"this arm's own RESET 0 1 0 {int(seed)}, at the entry "
                       f"frame {entry_frame}"),
            "state": state, "hud": hud, "player": player,
            "counts": {"bullets": entry_reads["hazards"]["bullets"],
                       "lasers": entry_reads["hazards"]["lasers"],
                       "enemies": entry_reads["hazards"]["enemies_count"]},
            "frame": entry_frame,
            "classification": "measured"}
        if not record["entry"]["reached"]:
            record["terminal"] = {
                "kind": "entry_not_reached",
                "detail": (f"frame {state['frame']} stage {state['stage']} dead "
                           f"{dead}"),
                "engine_started": True}
            exit_code = EXIT_GATE_FAILED
            raise _PilotStop(record["terminal"])
        tracker = FrozenTracker(
            slot=slot,
            spawn_decision=(card_spawn if observed_spawn is None
                            else int(observed_spawn)),
            life_at_spawn=(card_life if frozen_row is None
                           else int(frozen_row["life"])),
            entry_decision=entry)
        deaths_at_entry = float(state["deaths"])
        lives_at_entry = float(state["lives_remaining"])
        event_bits: dict[str, int] = {}
        nonzero_frames: list[list[Any]] = []
        event_frames = 0
        hash_frames = 0
        last_row: dict[str, Any] | None = None
        frames_in_window = 0
        outcome: str | None = None
        stop_reason = "decision_cap"
        tracker_outcome: str | None = None
        cand_histogram: dict[str, int] = {}
        with (base / "decisions.jsonl").open("w", encoding="utf-8") as handle:
            for offset in range(decision_cap):
                index = entry + offset
                cand = int(cands[offset])
                pair = [int(masks[offset][0]), int(masks[offset][1])]
                frame_start = index * DECISION_FRAMES + 1
                frame_end = frame_start + DECISION_FRAMES - 1
                if frames_in_window + DECISION_FRAMES > frame_cap:
                    stop_reason = "frame_cap"
                    break
                flags: list[int] = []
                hashes: dict[int, str] = {}
                reply: dict[str, Any] = {}
                shortfall = None
                for step_offset, mask in enumerate(pair):
                    frame = frame_start + step_offset
                    stepped = _step_one(attempt, int(mask), frame)
                    if not stepped["ok"]:
                        shortfall = {"frame": frame,
                                     "advanced": stepped["advanced"],
                                     "exit": stepped["exit"],
                                     "game_over": stepped["game_over"]}
                        break
                    frames_in_window += 1
                    flags.append(int(stepped["flags"]))
                    if stepped["hash"] is not None:
                        hashes[frame] = str(stepped["hash"])
                        hash_frames += 1
                    reply = stepped
                    event_frames += 1
                    for name in event_names(int(stepped["flags"])):
                        event_bits[name] = event_bits.get(name, 0) + 1
                    if int(stepped["flags"]):
                        if len(nonzero_frames) < 400:
                            nonzero_frames.append(
                                [frame, f"0x{int(stepped['flags']):04x}"])
                if shortfall is not None:
                    stop_reason = ("engine_exit" if shortfall["exit"]
                                   else "frame_shortfall")
                    record["terminal"] = {
                        "kind": "technical_conflict" if not shortfall["exit"]
                        else "engine_exit",
                        "detail": shortfall, "engine_started": True}
                    exit_code = (EXIT_TECHNICAL if not shortfall["exit"]
                                 else EXIT_GATE_FAILED)
                    break
                reads = read_decision(attempt.host)
                row: dict[str, Any] = {
                    "schema": SCHEMA_DIAG_TEACHER_ROW,
                    "phase": "control",
                    "arm": DIAG_TEACHER_ARM,
                    "seed": int(seed),
                    "classification_note": DIAG_TEACHER_CLASSIFICATION,
                    "decision_index": index,
                    "window_index": offset,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "cand": int(cand),
                    "action_source": "recorded_tape_masks",
                    "masks": pair,
                    "release_due": bool(adapter()["release_due"](
                        index, DEFAULT_PULSE_PERIOD)),
                    "ev_flags": [f"0x{value:04x}" for value in flags],
                    "ev_flags_dec": [int(value) for value in flags],
                    "ev_names": [event_names(value) for value in flags],
                    "state": engine_state_record(reads["state"]),
                    "hud": hud_record(reads["hud"]),
                    "player": player_record(reads["player"]),
                    "enemies": reads["hazards"]["enemies"],
                    "hazards_counts": {"bullets": reads["hazards"]["bullets"],
                                       "lasers": reads["hazards"]["lasers"],
                                       "enemies": reads["hazards"]["enemies_count"]},
                    "hash": {"frame_start": hashes.get(frame_start),
                             "frame_end": hashes.get(frame_end)},
                    "classification": "measured",
                }
                observed = tracker.observe(row)
                if observed is not None and tracker_outcome is None:
                    tracker_outcome = observed
                row["frozen"] = {
                    "slot": tracker.slot,
                    "present": any(int(item["slot"]) == tracker.slot
                                   for item in reads["hazards"]["enemies"]),
                    "life": tracker.last_life,
                    "outcome_so_far": tracker_outcome,
                    "phase_ends_so_far": len(tracker.phase_ends),
                }
                write_jsonl_row(handle, row)
                last_row = row
                cand_histogram[str(int(cand))] = cand_histogram.get(
                    str(int(cand)), 0) + 1
                record["counts"]["control_decisions"] += 1
                record["counts"]["control_frames"] += DECISION_FRAMES
                game_over = int(reads["hud"].get("game_over", 0) or 0) \
                    or int(reply.get("game_over", 0) or 0)
                deaths = float(reads["state"]["deaths"])
                # the frozen card's termination priority, unchanged
                if game_over:
                    outcome, stop_reason = "game_over", "game_over"
                    break
                if deaths > deaths_at_entry:
                    outcome, stop_reason = "legal_death", "legal_death"
                    break
                if tracker_outcome is not None:
                    outcome, stop_reason = tracker_outcome, "frozen_consequence"
                    break
            else:
                stop_reason = "decision_cap"
        record["counts"]["engine_frames"] = prefix["frames"] + frames_in_window
        attempt.engine_frames = record["counts"]["engine_frames"]
        executed = int(record["counts"]["control_decisions"])
        record["window"] = {
            "decisions": executed,
            "frames": int(frames_in_window),
            "frame_range": [entry_frame + 1, entry_frame + frames_in_window],
            "decision_range": [entry, entry + max(0, executed) - 1],
            "decision_cap": decision_cap, "frame_cap": frame_cap,
            "last_decision": (None if last_row is None
                              else int(last_row["decision_index"])),
            "last_frame": (None if last_row is None else int(last_row["frame_end"])),
            "classification": "measured"}
        record["decisions_path"] = (base / "decisions.jsonl").as_posix()
        record["decisions_sha256"] = sha256_file(base / "decisions.jsonl")
        record["execution"] = {
            "frames_stepped": int(frames_in_window),
            "events_frames_read": int(event_frames),
            "frames_with_a_state_hash": int(hash_frames),
            "decisions_with_hazards_state_hud_player": int(executed),
            "shortfall": None,
            "one_step_mask_1_per_frame": True,
            "note": ("one STEP mask 1 per frame with EVENTS after each frame, then "
                     "HAZARDS+STATE+HUD+PLAYER once per decision: the same read "
                     "method the card froze"),
            "classification": "measured"}
        record["events_summary"] = {
            "frames": int(event_frames),
            "frames_with_any_bit": len(nonzero_frames),
            "by_name": dict(sorted(event_bits.items())),
            "nonzero_frames_sample": nonzero_frames[:400],
            "classification": "measured"}
        record["resources_at_end"] = (
            None if last_row is None else {
                "frame": int(last_row["frame_end"]),
                "state": dict(last_row["state"]), "hud": dict(last_row["hud"]),
                "player": dict(last_row["player"]),
                "counts": dict(last_row["hazards_counts"]),
                "classification": "measured"})
        record["deaths_observed"] = (0.0 if last_row is None else
                                     float(last_row["state"]["deaths"])
                                     - deaths_at_entry)
        record["lives_lost"] = (0.0 if last_row is None else
                                lives_at_entry
                                - float(last_row["state"]["lives_remaining"]))
        record["frozen_generation"] = tracker.summary()
        record["frozen_generation"]["expected_identity"] = {
            "slot": slot, "card_spawn_decision": card_spawn,
            "card_life_at_spawn": card_life,
            "observed_spawn_decision": observed_spawn,
            "identity_mismatch": bool(identity_mismatch)}
        if outcome is None:
            outcome = ("not_removed_within_cap" if tracker.seen
                       else "generation_not_observed")
        record["outcome"] = outcome
        record["tracker_outcome"] = tracker_outcome
        record["outcome_decision"] = tracker.outcome_decision
        record["outcome_frame"] = tracker.outcome_frame
        record["stop_reason"] = stop_reason
        record["candidate_histogram"] = dict(
            sorted(cand_histogram.items(), key=lambda item: int(item[0])))
        record["candidate_dominant_share"] = (
            max(cand_histogram.values()) / sum(cand_histogram.values())
            if cand_histogram else 0.0)
        record["action_stats"] = strength_action_stats(
            cands=cands[:executed], masks=masks[:executed],
            first_decision=entry)
        record["terminal"] = {
            "kind": ("frozen_consequence" if outcome in OUTCOME_VOCABULARY
                     and outcome not in ("legal_death", "game_over")
                     else outcome),
            "outcome": outcome, "stop_reason": stop_reason,
            "window_frames": int(frames_in_window),
            "last_decision": (None if last_row is None
                              else int(last_row["decision_index"])),
            "game_over": (0 if last_row is None
                          else int(last_row["hud"]["game_over"])),
            "stage": (None if last_row is None
                      else int(last_row["state"]["stage"])),
            "classification": "measured"}
        exit_code = EXIT_OK
    except _PilotStop as stop:
        record["terminal"] = dict(stop.payload)
    except (EngineRefused, FrameRelationConflict, SourceError) as exc:
        record["terminal"] = {"kind": "technical_conflict",
                              "reason": type(exc).__name__,
                              "detail": str(exc),
                              "engine_started": bool(record.get("engine_started"))}
        exit_code = EXIT_TECHNICAL
    except Exception as exc:  # noqa: BLE001
        record["terminal"] = {"kind": "technical_conflict",
                              "reason": type(exc).__name__,
                              "detail": str(exc),
                              "traceback": traceback.format_exc(),
                              "engine_started": bool(record.get("engine_started"))}
        exit_code = EXIT_TECHNICAL
    finally:
        if attempt.engine_started or (attempt.reservation is not None
                                      and attempt.settlement is None):
            try:
                attempt.engine_frames = int(
                    record["counts"].get("engine_frames")
                    or record["counts"].get("prefix_frames", 0))
                closed = attempt.close(result={
                    "command": f"03-diag-teacher-{int(seed)}",
                    "seed": int(seed),
                    "terminal": (record.get("terminal") or {}).get("kind"),
                    "outcome": record.get("outcome"),
                    "reached_frame": attempt.engine_frames})
                record["mute"] = closed["mute"]
                record["source_dir"] = closed["source_dir"]
                record["ledger"] = closed["ledger"]
                record["measured"] = closed["measured"]
            except Exception as exc:  # noqa: BLE001
                record["close_error"] = f"{type(exc).__name__}: {exc}"
    record["arm_exit_code"] = int(exit_code)
    record["finished_at"] = now_iso()
    path = write_json(base / "result.json", record)
    record["result_path"] = path.as_posix()
    record["result_sha256"] = sha256_file(path)
    return record


def diag_teacher_settled_attempts(out_root: str | Path) -> dict[str, dict[str, Any]]:
    """The diagnostic's own settled ledger attempts, by attempt id.

    A later invocation of the same command must never spend a second start on a
    root whose attempt is already settled: the per-seed evidence and its ledger
    row exist, so the root is reused as it stands.
    """

    path = Path(diag_teacher_layout(out_root)["ledger"])
    settled: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return settled
    for row in iter_jsonl(path):
        attempt = str(row.get("attempt_id") or "")
        if not attempt.startswith(DIAG_TEACHER_ATTEMPT_PREFIX):
            continue
        if row.get("phase") == "settle":
            settled[attempt] = dict(row)
    return settled


def cmd_diag_teacher(args: argparse.Namespace) -> int:
    """The teacher continuation on the four unexposed roots: 4 starts, one verdict.

    Order matters.  The frozen card, the frozen registration, the tape prefix and
    the card's own teacher continuation are re-checked **before** the ledger is
    touched; the shared ledger is then read to prove the diagnostic fits both its
    own allowance (4 starts / 15,000 frames) and the whole-ticket totals; and only
    then does one start per root happen, in seed order, each one marked
    ``not_started_budget`` instead of started if the remaining room is too small.
    Only ``03/diag/teacher_arm/`` is written.
    """

    module = slice_tool()
    maybe_reexec_under_xvfb(enabled=not bool(getattr(args, "no_xvfb", False)))
    started = time.perf_counter()
    out_root = Path(args.out).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve()
    paths = diag_teacher_layout(out_root)
    base = Path(paths["base"])
    base.mkdir(parents=True, exist_ok=True)
    gate = Gate("03_diag_teacher")
    meter = module.CostMeter()
    meter.begin()
    cap = diag_teacher_stage_cap(
        max_starts=getattr(args, "max_starts", None),
        max_engine_frames=getattr(args, "max_engine_frames", None))
    only = [int(value) for value in (getattr(args, "only_seed", None) or [])]
    for seed in only:
        if seed not in DIAG_TEACHER_SEEDS:
            print(f"diag-teacher: --only-seed {seed} is not one of "
                  f"{list(DIAG_TEACHER_SEEDS)}")
            return EXIT_NOT_IMPLEMENTED
    run_seeds = [int(seed) for seed in (only or DIAG_TEACHER_SEEDS)]
    record: dict[str, Any] = {
        "schema": SCHEMA_DIAG_TEACHER,
        "generated_by": "tools/rl/v8_31_followup.py diag-teacher",
        "created_at": now_iso(),
        "classification": DIAG_TEACHER_CLASSIFICATION,
        "ticket": DIAG_TEACHER_TICKET,
        "question": ("does the frozen task card survive on the four unexposed dev "
                     "roots when the teacher continuation the pilot's arms "
                     "replaced plays it?"),
        "rule": DIAG_TEACHER_RULE,
        "repo": repo.as_posix(),
        "out": out_root.as_posix(),
        "argv": list(getattr(args, "argv", []) or []),
        "host": host_identity(),
        "heads": git_identity(repo),
        "tool": {"path": TOOL_PATH.as_posix(), "sha256": sha256_file(TOOL_PATH)},
        "slice_helper": slice_identity(),
        "layout": {"root": Path(paths["root"]).as_posix(),
                   "base": base.as_posix(),
                   "result": Path(paths["result"]).as_posix(),
                   "seeds": {str(int(seed)): Path(paths["seeds"][int(seed)]).as_posix()
                             for seed in DIAG_TEACHER_SEEDS}},
        "seeds": [int(seed) for seed in DIAG_TEACHER_SEEDS],
        "seeds_asked": [int(seed) for seed in run_seeds],
        "cap": cap,
        "budget": {},
        "roots": {},
        "evidence": {},
        "engine_starts": 0,
        "started_at": now_iso(),
    }
    exit_code = EXIT_TECHNICAL
    try:
        frozen = read_task_card(paths["card"])
        card = dict(frozen["card"])
        entry = int(card["entry"]["prefix_decisions"])
        entry_frame = int(card["entry"]["frame"])
        decision_cap = int(card["frame_cap"]["decisions"])
        frame_cap = int(card["frame_cap"]["frames"])
        record["task_card"] = {
            "path": Path(frozen["path"]).as_posix(),
            "sha256": frozen["sha256"],
            "content_sha256": frozen["content_sha256"],
            "content_sha256_recomputed": frozen["content_sha256_recomputed"],
            "task_id": card.get("task_id"),
            "entry_decision": entry, "entry_frame": entry_frame,
            "frame_cap": dict(card["frame_cap"]),
            "teacher_continuation": dict(card["teacher_continuation"]),
            "termination_priority": list(card["termination_priority"]),
            "not_a_policy_result": bool(card.get("not_a_policy_result")),
            "classification": "historical"}
        gate.add("task_card_read_back", frozen["content_sha256_matches"],
                 {"path": record["task_card"]["path"],
                  "sha256": record["task_card"]["sha256"],
                  "content_sha256": record["task_card"]["content_sha256"]})
        # the frozen registration, read-only: never rewritten by this command
        config_path = Path(getattr(args, "config", None)
                           or (repo / PILOT_CONFIG_REL))
        identity = load_pilot_roots_config(config_path)
        config_payload = identity.get("payload") or {}
        root_seeds = [int(item.get("seed")) for item in
                      config_payload.get("roots") or []]
        record["roots_config"] = {
            "path": Path(identity["path"]).as_posix(),
            "sha256": identity["sha256"],
            "bytes": Path(identity["path"]).stat().st_size,
            "content_sha256": identity["content_sha256"],
            "content_sha256_recomputed": identity["content_sha256_recomputed"],
            "status": config_payload.get("status"),
            "manifest_id": config_payload.get("manifest_id"),
            "seeds": root_seeds,
            "read_only": True,
            "written_this_invocation": False,
            "note": ("the registration is read and re-verified; this diagnostic "
                     "never rewrites it"),
            "classification": "measured"}
        gate.add("roots_config_frozen_and_read_only",
                 record["roots_config"]["status"] == "frozen"
                 and bool(record["roots_config"]["sha256"])
                 and record["roots_config"]["content_sha256"]
                 == record["roots_config"]["content_sha256_recomputed"],
                 {key: record["roots_config"][key] for key in
                  ("path", "sha256", "content_sha256", "status")})
        gate.add("roots_config_binds_the_four_dev_roots",
                 root_seeds == [int(seed) for seed in DIAG_TEACHER_SEEDS],
                 {"roots": root_seeds, "expected": list(DIAG_TEACHER_SEEDS)})
        with meter.measure("io"):
            tape = tape_schedule(repo)
        prefix_digest = tape_digest(tape, 0, entry)
        record["frozen_entry"] = {
            "source": "the frozen tape's decisions 0..entry-1 (the card's prefix)",
            "tape_shard": tape["shard"],
            "tape_sha256": tape["sha256"],
            "tape_registered_sha256": tape["registered_sha256"],
            "prefix_masks_sha256_recomputed": prefix_digest,
            "prefix_masks_sha256_card": card["entry"]["prefix_masks_sha256"],
            "entry_decision": entry, "entry_frame": entry_frame,
            "classification": "offline"}
        gate.add("frozen_prefix_matches_the_tape",
                 prefix_digest == card["entry"]["prefix_masks_sha256"],
                 {"recomputed": prefix_digest,
                  "card": card["entry"]["prefix_masks_sha256"]})
        window = diag_teacher_window(card=card, tape=tape)
        record["frozen_window"] = {key: value for key, value in window.items()
                                   if key not in ("cands", "masks")}
        gate.add("teacher_window_matches_the_card_and_the_frozen_executor",
                 window["masks_sha256_matches_the_card"]
                 and window["masks_are_the_frozen_executor_expansion"],
                 {"masks_sha256_recomputed": window["masks_sha256"],
                  "masks_sha256_card": window["masks_sha256_card"],
                  "executor_expansion": window[
                      "masks_are_the_frozen_executor_expansion"],
                  "first_decision": window["first_decision"],
                  "last_decision": window["last_decision"]})
        pilot = diag_teacher_pilot_reference(out_root)
        record["pilot_reference"] = pilot
        gate.add("pilot_reference_read_from_03_pilot_json",
                 bool(pilot["sha256"])
                 and all((pilot["roots"][str(int(seed))] or {}).get("present")
                         for seed in DIAG_TEACHER_SEEDS)
                 and all(len((pilot["roots"][str(int(seed))] or {}).get(
                     "outcome_by_arm") or {}) == len(PILOT_ARMS)
                     for seed in DIAG_TEACHER_SEEDS),
                 {"path": pilot["path"], "sha256": pilot["sha256"],
                  "verdict": pilot["verdict"],
                  "outcome_columns": {key: value.get("outcome_column")
                                      for key, value in pilot["roots"].items()},
                  "note": "read from 03/pilot.json, never recomputed"})
        arm_frames = entry_frame + frame_cap
        frame_plan = {"entry_frame": entry_frame, "frame_cap": frame_cap,
                      "decision_cap": decision_cap,
                      "per_root_frames": int(arm_frames),
                      "roots_asked": len(run_seeds),
                      "planned_frames": int(arm_frames * len(run_seeds)),
                      "cap": cap}
        record["frame_plan"] = frame_plan
        # -- the budget: read the existing ledger first -----------------------
        budget: dict[str, Any] = {
            "cap": cap, "frame_plan": frame_plan,
            "per_root": {}, "per_root_limits": {},
            "skipped_for_budget": [],
            "root_order": [int(seed) for seed in run_seeds],
            "classification": "measured"}
        planned_starts = len(run_seeds)
        planned_frames = int(arm_frames * planned_starts)
        budget["planned"] = {"starts": planned_starts,
                             "engine_frames": planned_frames}
        try:
            preflight = stage_budget(
                out_root, need_starts=planned_starts,
                need_frames=planned_frames, label="03-diag-teacher",
                stage_cap={"starts": cap["starts"],
                           "engine_frames": cap["engine_frames"]},
                stage_prefixes=(DIAG_TEACHER_ATTEMPT_PREFIX,))
            budget["preflight"] = dict(preflight["report"])
            budget["preflight"]["admissible"] = True
        except BudgetRefused as exc:
            budget["preflight"] = dict(exc.payload)
            budget["preflight"]["admissible"] = False
        record["budget"] = budget
        record["ledger_cumulative_at_start"] = dict(
            budget["preflight"].get("cumulative") or {})
        gate.add("budget_read_from_the_existing_ledger",
                 bool(budget["preflight"].get("ledger_rows_read")),
                 {"ledger_rows_read": budget["preflight"].get("ledger_rows_read"),
                  "cumulative": budget["preflight"].get("cumulative"),
                  "stage_used": budget["preflight"].get("stage_used"),
                  "need": budget["preflight"].get("need"),
                  "admissible": budget["preflight"].get("admissible"),
                  "note": ("the whole diagnostic is priced before the first start; "
                           "the per-root check below decides root by root")})
        # -- one start per root, in seed order --------------------------------
        settled = diag_teacher_settled_attempts(out_root)
        budget["reused"] = []
        for position, seed in enumerate(run_seeds):
            seed_dir = Path(paths["seeds"][int(seed)])
            settled_row = settled.get(f"{DIAG_TEACHER_ATTEMPT_PREFIX}{int(seed)}")
            if settled_row is not None and (seed_dir / "result.json").is_file():
                try:
                    existing = _load_json(seed_dir / "result.json")
                except (OSError, ValueError):
                    existing = {}
                if existing.get("engine_started") and existing.get("outcome"):
                    budget["reused"].append(int(seed))
                    print(f"  03-diag-teacher seed {seed}: reused "
                          f"(settled as {settled_row.get('attempt_id')}, no new "
                          f"start)", flush=True)
                    continue
            if (seed_dir / "game").exists():
                raise EngineRefused(
                    f"seed {int(seed)}: a private game directory already exists "
                    f"({seed_dir / 'game'}) but the ledger has no settled attempt "
                    "for it; every start needs its own fresh directory")
            try:
                step = stage_budget(
                    out_root, need_starts=1, need_frames=int(arm_frames),
                    label=f"03-diag-teacher-seed{int(seed)}",
                    stage_cap={"starts": cap["starts"],
                               "engine_frames": cap["engine_frames"]},
                    stage_prefixes=(DIAG_TEACHER_ATTEMPT_PREFIX,))
                budget["per_root"][str(int(seed))] = {
                    "admissible": True,
                    "cumulative": dict(step["report"]["cumulative"]),
                    "stage_used": dict(step["report"]["stage_used"]),
                    "stage_remaining": dict(step["report"]["stage_remaining"]),
                    "grand_remaining": dict(step["report"]["grand_remaining"]),
                    "need": dict(step["report"]["need"])}
                budget["per_root_limits"][str(int(seed))] = step[
                    "limits"].as_dict()
                limits = step["limits"]
            except BudgetRefused as exc:
                budget["per_root"][str(int(seed))] = {
                    "admissible": False, "payload": dict(exc.payload)}
                remaining = [int(value) for value in run_seeds[position:]]
                budget["skipped_for_budget"] = remaining
                record["budget"] = budget
                for later in remaining:
                    unstarted = diag_teacher_unstarted_record(
                        seed=later, frame_plan=frame_plan,
                        budget=budget["per_root"][str(int(seed))])
                    path = Path(paths["seeds"][int(later)]) / "result.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    write_json(path, unstarted)
                    print(f"  03-diag-teacher seed {later}: not_started_budget",
                          flush=True)
                break
            arm_record = run_diag_teacher_arm(
                args=args, out_root=out_root, repo=repo, card=card, tape=tape,
                window=window, paths=paths, seed=int(seed), limits=limits,
                meter=meter)
            record["engine_starts"] = int(record["engine_starts"]) + int(
                bool(arm_record.get("engine_started")))
            row = diag_teacher_row(arm_record)
            print(f"  03-diag-teacher seed {seed}: "
                  f"entry={row['entry_reached']} "
                  f"slot_present={row['frozen_slot_present']} "
                  f"frames={row['engine_frames']} "
                  f"outcome={row['outcome']} "
                  f"stop={row['stop_reason']} "
                  f"deaths={row['deaths_observed']} "
                  f"life={row['frozen_life_first']}->{row['frozen_life_min']} "
                  f"score={row['score_at_entry']}->{row['score_at_end']} "
                  f"rc={row['exit_code']}", flush=True)
            if not arm_record.get("engine_started"):
                break
        # -- the per-root table, re-read from the files it just wrote ---------
        table: dict[str, Any] = {}
        evidence: dict[str, Any] = {}
        for seed in DIAG_TEACHER_SEEDS:
            path = Path(paths["seeds"][int(seed)]) / "result.json"
            evidence[str(int(seed))] = {"path": path.as_posix(),
                                        "exists": path.is_file()}
            if path.is_file():
                evidence[str(int(seed))]["sha256"] = sha256_file(path)
                reloaded = _load_json(path)
                table[str(int(seed))] = diag_teacher_row(
                    reloaded, pilot=pilot["roots"].get(str(int(seed))) or {})
                table[str(int(seed))]["result"] = path.as_posix()
                table[str(int(seed))]["result_sha256"] = evidence[
                    str(int(seed))]["sha256"]
            else:
                table[str(int(seed))] = {
                    "seed": int(seed), "engine_started": False,
                    "outcome": None, "conclusion": "missing",
                    "pilot_same_root": pilot["roots"].get(str(int(seed))) or {},
                    "pilot_same_root_outcome_column": (
                        pilot["roots"].get(str(int(seed))) or {}).get(
                            "outcome_column")}
        record["roots"] = table
        record["evidence"] = evidence
        verdict = diag_teacher_interpretation(teacher_roots=table,
                                              pilot_reference=pilot)
        record["teacher_completed_roots"] = verdict["teacher_completed_roots"]
        record["pilot_completed_roots"] = verdict["pilot_completed_roots"]
        record["interpretation"] = verdict["interpretation"]
        record["interpretation_block"] = verdict
        record["classification"] = DIAG_TEACHER_CLASSIFICATION
        # -- the technical gate (and nothing about policy quality) ------------
        technical = [key for key, row in table.items()
                     if (row or {}).get("terminal") in (
                         "technical_conflict", "technical_invalid")]
        started_rows = [row for row in table.values() if (row or {}).get(
            "engine_started")]
        gate.add("all_four_roots_have_a_conclusion",
                 len(verdict["teacher_concluded_roots"]) == len(DIAG_TEACHER_SEEDS)
                 and sorted(table) == sorted(str(int(seed)) for seed in
                                             DIAG_TEACHER_SEEDS),
                 {"concluded": verdict["teacher_concluded_roots"],
                  "outcomes": {key: row.get("outcome")
                               for key, row in table.items()},
                  "not_started_budget": budget["skipped_for_budget"]})
        gate.add("no_technical_invalid_root", not technical,
                 {"technical": technical})
        gate.add("all_four_roots_reached_the_frozen_entry",
                 len(started_rows) == len(DIAG_TEACHER_SEEDS)
                 and all(row.get("entry_reached") and row.get(
                     "frozen_slot_present") and not row.get("identity_mismatch")
                     for row in started_rows),
                 {key: {"entry_reached": row.get("entry_reached"),
                        "frame": row.get("entry_frame"),
                        "stage": row.get("entry_stage"),
                        "slot_present": row.get("frozen_slot_present"),
                        "spawn_delta": row.get("spawn_delta")}
                  for key, row in table.items()})
        gate.add("card_entry_hash_is_recorded_for_every_root",
                 all(row.get("card_entry_hash") is not None
                     and row.get("card_entry_hash_matches") is not None
                     for row in started_rows),
                 {key: {"entry_hash": row.get("entry_hash"),
                        "card_hash": row.get("card_entry_hash"),
                        "matches": row.get("card_entry_hash_matches")}
                  for key, row in table.items()})
        gate.add("engine_starts_within_the_cap",
                 int(record["engine_starts"]) <= int(cap["starts"])
                 and sum(int(row.get("engine_frames") or 0)
                         for row in started_rows)
                 <= int(cap["engine_frames"]),
                 {"engine_starts": record["engine_starts"],
                  "cap_starts": cap["starts"],
                  "engine_frames": sum(int(row.get("engine_frames") or 0)
                                       for row in started_rows),
                  "cap_frames": cap["engine_frames"]})
        final_budget = stage_budget(
            out_root, need_starts=0, need_frames=0, label="03-diag-teacher-final",
            stage_cap={"starts": cap["starts"],
                       "engine_frames": cap["engine_frames"]},
            stage_prefixes=(DIAG_TEACHER_ATTEMPT_PREFIX,))
        final_report = final_budget["report"]
        record["budget_final"] = final_report
        record["ledger_cumulative_at_end"] = dict(final_report["cumulative"])
        gate.add("budget_not_exceeded",
                 float(final_report["stage_used"]["starts"])
                 <= float(cap["starts"])
                 and float(final_report["stage_used"]["engine_frames"])
                 <= float(cap["engine_frames"])
                 and float(final_report["cumulative"]["starts"])
                 <= float(GRAND_TOTAL_STARTS)
                 and float(final_report["cumulative"]["engine_frames"])
                 <= float(GRAND_TOTAL_FRAMES),
                 {"stage_used": final_report["stage_used"],
                  "stage_remaining": final_report["stage_remaining"],
                  "cumulative": final_report["cumulative"],
                  "cap": cap,
                  "grand_total": final_report["grand_total"]})
        gate.add("every_start_is_settled_in_the_ledger",
                 all(((row.get("ledger_actual") or {}).get("starts") == 1.0)
                     for row in started_rows),
                 {key: row.get("ledger_actual")
                  for key, row in table.items() if row.get("engine_started")})
        gate.add("mute_after_exit_clean",
                 all(not (row.get("mute_problems") or [])
                     for row in started_rows),
                 {key: row.get("mute_problems")
                  for key, row in table.items() if row.get("engine_started")})
        gate.add("per_root_evidence_written",
                 all(item["exists"] for item in evidence.values()),
                 evidence)
        gate.add("classification_is_the_diagnostic_one",
                 record["classification"] == DIAG_TEACHER_CLASSIFICATION,
                 {"classification": record["classification"]})
        record["verdict"] = ("diagnostic_complete" if gate.passed
                             else "diagnostic_incomplete")
        exit_code = EXIT_OK if gate.passed else EXIT_GATE_FAILED
    except BudgetRefused as exc:
        record["budget"] = dict(exc.payload)
        record["verdict"] = "budget_refused"
        record["terminal"] = {"kind": "budget_refused", "reason": exc.reason,
                              "detail": exc.payload.get("exceeded")}
        gate.add("budget_checked", False, exc.payload.get("exceeded"))
        exit_code = EXIT_NOT_IMPLEMENTED
    except (EngineRefused, SourceError, FrameRelationConflict) as exc:
        record["verdict"] = "technical_invalid"
        record["terminal"] = {"kind": "technical_invalid",
                              "reason": type(exc).__name__, "detail": str(exc),
                              "engine_started": bool(record.get("engine_starts"))}
        gate.add("diagnostic_inputs_readable", False, {"error": str(exc)})
        exit_code = EXIT_TECHNICAL
    except AdapterUnavailable as exc:
        record["verdict"] = "technical_invalid"
        record["terminal"] = {"kind": "technical_invalid",
                              "reason": type(exc).__name__, "detail": str(exc)}
        gate.add("adapter_available", False, {"error": str(exc)})
        exit_code = EXIT_TECHNICAL
    except Exception as exc:  # noqa: BLE001
        record["verdict"] = "technical_invalid"
        record["terminal"] = {"kind": "technical_invalid",
                              "reason": type(exc).__name__, "detail": str(exc),
                              "traceback": traceback.format_exc(),
                              "engine_started": bool(record.get("engine_starts"))}
        gate.add("diagnostic_completed", False, {"error": str(exc)})
        exit_code = EXIT_TECHNICAL
    finally:
        meter.finish()
    record.setdefault("terminal", {"kind": "unknown"})
    record.setdefault("verdict", "diagnostic_incomplete")
    record["finished_at"] = now_iso()
    record["cost"] = meter.payload(step_units=0)
    record["seconds"] = round(time.perf_counter() - started, 3)
    record["gate"] = gate.as_dict()
    target = Path(paths["result"])
    previous: dict[str, Any] = {}
    if target.is_file():
        try:
            previous = _load_json(target)
        except (OSError, ValueError):  # pragma: no cover - corrupt previous file
            previous = {}
    previous_conclusions = [key for key, row in (previous.get("roots") or {}).items()
                            if (row or {}).get("conclusion") == "measured"]
    fresh_conclusions = [key for key, row in (record.get("roots") or {}).items()
                         if (row or {}).get("conclusion") == "measured"]
    if previous_conclusions and not fresh_conclusions:
        # An invocation that started nothing (a budget refusal, a missing input)
        # never replaces evidence that already carries a conclusion for a root.
        alternate = target.with_name(
            f"{target.stem}.{record.get('verdict') or 'failed'}-"
            f"{datetime.now().strftime('%Y%m%dT%H%M%S')}{target.suffix}")
        record["preserved_evidence"] = {
            "kept": target.as_posix(),
            "previous_conclusions": previous_conclusions,
            "this_run_written_to": alternate.as_posix(),
            "reason": ("an invocation that concluded nothing never replaces "
                       "evidence that already carries a per-root conclusion"),
            "classification": "measured"}
        target = alternate
    path = write_json(target, record)
    for check in gate.checks:
        print(f"  {'ok  ' if check.ok else 'FAIL'} {check.name}")
    print(f"diag-teacher: roots={len(record.get('seeds') or [])} "
          f"starts={record.get('engine_starts')} "
          f"interpretation={record.get('interpretation')} "
          f"teacher_completed={record.get('teacher_completed_roots')} "
          f"rc={exit_code}")
    print(f"wrote: {path}")
    return exit_code


def verify_03_diag_teacher(out_root: str | Path, repo: str | Path,
                           gate: Gate) -> dict[str, Any]:
    """Re-derive the teacher-arm diagnostic from its own raw rows; never starts an engine.

    Recomputes the frozen window from the tape, the card and the registration,
    re-reads every root's ``result.json`` (its sha256, its frozen digests, its
    outcome), re-applies the interpretation rule to the recomputed outcomes and
    compares it with the claim.  Nothing about the arms is re-run.
    """

    out_root = Path(out_root)
    repo = Path(repo)
    paths = diag_teacher_layout(out_root)
    result_path = Path(paths["result"])
    if not result_path.is_file():
        return {"status": "not-run",
                "reason": "no 03/diag/teacher_arm/result.json under this out dir"}
    claimed = _load_json(result_path)
    report: dict[str, Any] = {
        "status": "failed",
        "path": result_path.as_posix(),
        "result_sha256": sha256_file(result_path),
        "claimed_interpretation": claimed.get("interpretation"),
        "claimed_teacher_completed_roots": claimed.get("teacher_completed_roots"),
        "claimed_pilot_completed_roots": claimed.get("pilot_completed_roots"),
        "classification": claimed.get("classification"),
    }
    gate.add("03_diag_teacher_present",
             claimed.get("schema") == SCHEMA_DIAG_TEACHER
             and claimed.get("classification") == DIAG_TEACHER_CLASSIFICATION,
             {"schema": claimed.get("schema"),
              "classification": claimed.get("classification"),
              "expected": DIAG_TEACHER_CLASSIFICATION})
    frozen = read_task_card(Path(paths["card"]))
    card = dict(frozen["card"])
    gate.add("03_diag_teacher_used_this_card",
             (claimed.get("task_card") or {}).get("sha256") == frozen["sha256"],
             {"card": frozen["sha256"],
              "claimed": (claimed.get("task_card") or {}).get("sha256")})
    config_path = Path((claimed.get("roots_config") or {}).get("path") or "")
    identity = load_pilot_roots_config(config_path)
    gate.add("03_diag_teacher_roots_config_unchanged",
             identity["content_sha256"]
             == (claimed.get("roots_config") or {}).get("content_sha256")
             and identity["sha256"]
             == (claimed.get("roots_config") or {}).get("sha256")
             and (identity.get("payload") or {}).get("status") == "frozen",
             {"file_sha256": identity["sha256"],
              "content_sha256": identity["content_sha256"],
              "claimed_sha256": (claimed.get("roots_config") or {}).get("sha256"),
              "status": (identity.get("payload") or {}).get("status")})
    tape = tape_schedule(repo)
    entry = int(card["entry"]["prefix_decisions"])
    prefix_digest = tape_digest(tape, 0, entry)
    gate.add("03_diag_teacher_prefix_recomputed",
             prefix_digest == card["entry"]["prefix_masks_sha256"]
             == (claimed.get("frozen_entry") or {}).get(
                 "prefix_masks_sha256_recomputed"),
             {"recomputed": prefix_digest,
              "card": card["entry"]["prefix_masks_sha256"]})
    window = diag_teacher_window(card=card, tape=tape)
    claimed_window = claimed.get("frozen_window") or {}
    gate.add("03_diag_teacher_window_recomputed",
             window["masks_sha256"] == claimed_window.get("masks_sha256")
             and window["masks_sha256"] == window["masks_sha256_card"]
             and bool(window["masks_are_the_frozen_executor_expansion"]),
             {"recomputed": window["masks_sha256"],
              "claimed": claimed_window.get("masks_sha256"),
              "card": window["masks_sha256_card"],
              "executor_expansion": window[
                  "masks_are_the_frozen_executor_expansion"]})
    pilot = diag_teacher_pilot_reference(out_root)
    gate.add("03_diag_teacher_pilot_reference_recomputed",
             pilot["sha256"]
             == (claimed.get("pilot_reference") or {}).get("sha256")
             and pilot["completed_roots"] == list(
                 (claimed.get("pilot_reference") or {}).get("completed_roots")
                 or [])
             and bool(pilot["path"]),
             {"pilot_sha256": pilot["sha256"],
              "claimed_pilot_sha256": (claimed.get("pilot_reference") or {}).get(
                  "sha256"),
              "completed_roots": pilot["completed_roots"],
              "claimed_completed_roots": (claimed.get("pilot_reference") or {}).get(
                  "completed_roots")})
    # 1. every root's raw file, re-read
    per_root: dict[str, Any] = {}
    recomputed_outcomes: dict[str, Any] = {}
    roots_ok = True
    for seed in DIAG_TEACHER_SEEDS:
        key = str(int(seed))
        claim = (claimed.get("roots") or {}).get(key) or {}
        evidence = (claimed.get("evidence") or {}).get(key) or {}
        path = Path(evidence.get("path") or
                    (Path(paths["seeds"][int(seed)]) / "result.json"))
        detail: dict[str, Any] = {"seed": int(seed), "path": path.as_posix(),
                                  "exists": path.is_file(),
                                  "claimed_outcome": claim.get("outcome")}
        ok = bool(claim) and path.is_file()
        if path.is_file():
            raw = _load_json(path)
            detail["sha256"] = sha256_file(path)
            detail["sha256_matches"] = bool(
                evidence.get("sha256")
                and detail["sha256"] == evidence.get("sha256"))
            row = diag_teacher_row(raw)
            recomputed_outcomes[key] = {"outcome": row.get("outcome"),
                                        "conclusion": row.get("conclusion")}
            detail.update({
                "recomputed_outcome": row.get("outcome"),
                "recomputed_conclusion": row.get("conclusion"),
                "entry_frame": row.get("entry_frame"),
                "entry_reached": row.get("entry_reached"),
                "prefix_frames": row.get("prefix_frames"),
                "control_decisions": row.get("control_decisions"),
                "engine_frames": row.get("engine_frames"),
                "deaths_observed": row.get("deaths_observed"),
                "frozen_life_first": row.get("frozen_life_first"),
                "frozen_life_min": row.get("frozen_life_min"),
                "score_at_entry": row.get("score_at_entry"),
                "score_at_end": row.get("score_at_end"),
            })
            digest_ok = (
                (raw.get("prefix") or {}).get("prefix_masks_sha256_recomputed")
                == card["entry"]["prefix_masks_sha256"]
                if raw.get("engine_started") else True)
            window_ok = (
                (raw.get("window_masks") or {}).get("masks_sha256_recomputed")
                == window["masks_sha256"]
                if raw.get("engine_started") else True)
            detail["prefix_digest_matches_the_card"] = bool(digest_ok)
            detail["window_digest_matches_the_card"] = bool(window_ok)
            frames_ok = int(row.get("engine_frames") or 0) <= int(
                (claimed.get("frame_plan") or {}).get("per_root_frames") or 0)
            detail["frames_within_the_plan"] = bool(frames_ok)
            ok = (ok and detail["sha256_matches"]
                  and detail["recomputed_outcome"] == claim.get("outcome")
                  and digest_ok and window_ok and frames_ok)
        roots_ok = roots_ok and ok
        per_root[key] = detail
        gate.add(f"03_diag_teacher_root_{int(seed)}_recomputed", ok, detail)
    # 2. the interpretation, re-applied to the recomputed outcomes
    fresh = diag_teacher_interpretation(
        teacher_roots=dict(recomputed_outcomes),
        pilot_reference=pilot)
    gate.add("03_diag_teacher_interpretation_recomputed",
             fresh["interpretation"] == claimed.get("interpretation")
             and list(fresh["teacher_completed_roots"])
             == list(claimed.get("teacher_completed_roots") or [])
             and list(fresh["pilot_completed_roots"])
             == list(claimed.get("pilot_completed_roots") or []),
             {"recomputed": fresh["interpretation"],
              "claimed": claimed.get("interpretation"),
              "recomputed_teacher_completed_roots":
                  fresh["teacher_completed_roots"],
              "claimed_teacher_completed_roots":
                  claimed.get("teacher_completed_roots"),
              "recomputed_pilot_completed_roots": fresh["pilot_completed_roots"],
              "claimed_pilot_completed_roots": claimed.get("pilot_completed_roots")})
    # 3. the ledger's four starts, as the diagnostic claims them
    starts = 0
    settled = 0
    ledger_rows = ([dict(row) for row in iter_jsonl(Path(out_root) / "ledger.jsonl")]
                   if (Path(out_root) / "ledger.jsonl").is_file() else [])
    for row in ledger_rows:
        attempt = str(row.get("attempt_id") or "")
        if not attempt.startswith(DIAG_TEACHER_ATTEMPT_PREFIX):
            continue
        if row.get("phase") == "reserve":
            starts += 1
        elif row.get("phase") == "settle":
            settled += 1
    cap = diag_teacher_stage_cap()
    gate.add("03_diag_teacher_ledger_settled",
             settled == starts == int(claimed.get("engine_starts") or 0)
             and starts <= DIAG_TEACHER_STAGE_STARTS,
             {"reserve_rows": starts, "settle_rows": settled,
              "claimed_engine_starts": claimed.get("engine_starts"),
              "cap_starts": DIAG_TEACHER_STAGE_STARTS})
    total_frames = 0.0
    for row in ledger_rows:
        if not str(row.get("attempt_id") or "").startswith(
                DIAG_TEACHER_ATTEMPT_PREFIX):
            continue
        if row.get("phase") == "settle":
            total_frames += float((row.get("actual") or {}).get(
                "engine_frames", 0) or 0)
    gate.add("03_diag_teacher_frames_within_the_cap",
             total_frames <= float(DIAG_TEACHER_STAGE_FRAMES),
             {"settled_engine_frames": total_frames,
              "cap_engine_frames": DIAG_TEACHER_STAGE_FRAMES})
    report.update({
        "roots": per_root,
        "interpretation": {"recomputed": fresh, "claimed_interpretation":
                           claimed.get("interpretation")},
        "ledger": {"reserve_rows": starts, "settle_rows": settled,
                   "settled_engine_frames": total_frames},
        "cap": cap,
        "status": ("verified" if (roots_ok and gate.passed) else "failed")})
    return report


NOT_IMPLEMENTED: dict[str, str] = {}


def cmd_not_implemented(args: argparse.Namespace) -> int:
    """Print ``not-implemented`` and return 3 (nothing was started)."""

    reason = NOT_IMPLEMENTED.get(args.command, "not implemented in this step")
    print(f"not-implemented: {args.command} ({reason})")
    return EXIT_NOT_IMPLEMENTED


# ---------------------------------------------------------------------------
# `run`: the whole ticket in one ledger, reusing anything that re-verifies
# ---------------------------------------------------------------------------
#: The package sequence of ``TRAINING_HOST_REQUEST_02`` line 60, in order.  Each
#: entry names the package (the key ``run.json`` and ``verify`` share), the
#: command(s) that produce it and whether it may start an engine.
RUN_PACKAGES: tuple[dict[str, Any], ...] = (
    {"name": "02a-sources", "commands": ("sources",), "engine": False},
    {"name": "02a-adapter", "commands": ("adapter",), "engine": False},
    {"name": "02a-prefix", "commands": ("prefix",), "engine": True},
    {"name": "02a-reference", "commands": ("reference",), "engine": True},
    {"name": "02a-compare", "commands": ("compare",), "engine": False},
    {"name": "02b", "commands": ("recon", "nominate"), "engine": True},
    {"name": "02c", "commands": ("intervene",), "engine": True},
    {"name": "03", "commands": ("pilot",), "engine": True},
    {"name": "verify", "commands": ("verify",), "engine": False},
)
RUN_PACKAGE_BY_NAME = {entry["name"]: entry for entry in RUN_PACKAGES}
#: ``--only``/``--skip`` accept the package name or any of its command names.
RUN_SELECTION_ALIASES = {
    "sources": "02a-sources", "adapter": "02a-adapter", "prefix": "02a-prefix",
    "reference": "02a-reference", "compare": "02a-compare",
    "recon": "02b", "nominate": "02b", "intervene": "02c", "pilot": "03",
    "verify": "verify",
}


def run_selection(args: argparse.Namespace) -> list[str]:
    """Which packages this ``run`` invocation may touch, in the frozen order."""

    def resolve(text: str) -> list[str]:
        out: list[str] = []
        for token in str(text or "").replace(" ", "").split(","):
            if not token:
                continue
            name = RUN_SELECTION_ALIASES.get(token, token)
            if name not in RUN_PACKAGE_BY_NAME:
                raise SystemExit(
                    f"--only/--skip: {token!r} is not one of "
                    f"{sorted(RUN_PACKAGE_BY_NAME)} or their command names "
                    f"{sorted(RUN_SELECTION_ALIASES)}")
            out.append(name)
        return out

    only = resolve(getattr(args, "only", "") or "")
    skip = set(resolve(getattr(args, "skip", "") or ""))
    names = [entry["name"] for entry in RUN_PACKAGES]
    if only:
        names = [name for name in names if name in only]
    return [name for name in names if name not in skip]


def run_verify_selection(out_root: Path) -> str:
    """The broadest ``verify --package`` whose inputs actually exist.

    ``all`` demands the two 02-A engine packages whenever ``sources.json`` exists,
    so a partial run (``--only sources,adapter``) must not ask for it: it would
    report a missing input as a technical failure of a package that was never due.
    """

    modes = all((out_root / "02a" / mode / "result.json").is_file()
                for mode in ENGINE_PACKAGES)
    if modes and (out_root / "02a" / "compare.json").is_file():
        return "all"
    if (out_root / "02a" / "sources.json").is_file() and (
            out_root / "02a" / "adapter.json").is_file():
        return "02a"
    return "02a"


def run_package_namespace(args: argparse.Namespace, command: str,
                          out_root: Path, repo: Path) -> argparse.Namespace:
    """The namespace one packaged command expects, with the run's own budget.

    ``prefix``/``reference`` price themselves against the **whole-ticket** budget
    (``--max-starts``/``--max-engine-frames``), so they get the run's numbers.
    The stage-capped packages (``recon``/``nominate``/``intervene``/``pilot``)
    get ``None`` instead: their own stage caps (3 starts / 20,000 or 11,200
    frames) must never be silently raised by a run-level number.
    """

    stage_capped = command in ("recon", "nominate", "intervene", "pilot")
    values: dict[str, Any] = {
        "repo": repo.as_posix(), "out": out_root.as_posix(), "command": command,
        "argv": list(getattr(args, "argv", []) or []),
    }
    if command in ("prefix", "reference"):
        values.update({
            "engine": args.engine, "data_source": args.data_source,
            "max_starts": args.max_starts,
            "max_engine_frames": args.max_engine_frames,
            "max_wall_s": args.max_wall_s, "max_write_gib": args.max_write_gib,
            "timeout_s": getattr(args, "timeout_s", 120.0),
            "no_xvfb": bool(getattr(args, "no_xvfb", False))})
    elif command in ("recon", "intervene", "pilot"):
        values.update({
            "engine": args.engine, "data_source": args.data_source,
            "max_starts": None, "max_engine_frames": None,
            "max_wall_s": args.max_wall_s, "max_write_gib": args.max_write_gib,
            "timeout_s": getattr(args, "timeout_s", 120.0),
            "no_xvfb": bool(getattr(args, "no_xvfb", False))})
    elif command == "nominate":
        values.update({"recon": (out_root / "02b" / "recon").as_posix(),
                       "max_starts": None, "max_engine_frames": None})
    if command == "pilot":
        values.update({"config": getattr(args, "config", None),
                       "only_seed": list(getattr(args, "only_seed", []) or []),
                       "smoke": bool(getattr(args, "smoke", False)),
                       "reuse": True})
    if command == "verify":
        values.update({"package": run_verify_selection(out_root)})
    return argparse.Namespace(**values)


#: The record each package rewrites wholesale.  ``run`` snapshots these before a
#: package is re-run and restores them when the re-run fails, so a wrong input
#: (a bad ``--repo``, a moved corpus) can never destroy evidence that already
#: passed: the failure is reported, the passing record survives.
RUN_EVIDENCE_FILES: dict[str, tuple[str, ...]] = {
    "02a-sources": ("02a/sources.json",),
    "02a-adapter": ("02a/adapter.json",),
    "02a-compare": ("02a/compare.json",),
    "02b": ("02b/nomination.json", "02b/task_card.json", "02b/task_card.md"),
    "02c": ("02c/intervene.json",),
    "03": ("03/pilot.json", "03/smoke/pilot.json"),
}


def run_evidence_snapshot(out_root: Path, name: str) -> dict[str, bytes]:
    """The bytes of one package's record files, before it is re-run."""

    snapshot: dict[str, bytes] = {}
    for relative in RUN_EVIDENCE_FILES.get(name) or ():
        path = out_root / relative
        if path.is_file():
            snapshot[relative] = path.read_bytes()
    return snapshot


def run_evidence_restore(out_root: Path, snapshot: Mapping[str, bytes]) -> list[str]:
    """Put back any snapshot file the failed re-run changed; return what moved."""

    restored: list[str] = []
    for relative, data in snapshot.items():
        path = out_root / relative
        if path.is_file() and path.read_bytes() == data:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        restored.append(relative)
    return restored


def run_prerequisites(name: str, out_root: Path,
                      args: argparse.Namespace) -> list[str]:
    """The missing inputs that would make a package meaningless right now."""
    missing: list[str] = []
    if name in ("02a-prefix", "02a-reference", "02b", "02c", "03"):
        if not getattr(args, "engine", None):
            missing.append("--engine is not set")
        if not getattr(args, "data_source", None):
            missing.append("--data-source is not set")
    if name == "02a-compare":
        for mode in ENGINE_PACKAGES:
            path = out_root / "02a" / mode / "result.json"
            if not path.is_file():
                missing.append(f"{path} is missing")
    if name == "02b":
        if not (out_root / "02b" / "recon" / "result.json").is_file():
            missing.append("no recon evidence (nominate needs it)")
    if name in ("02c", "03"):
        if not (out_root / "02b" / "task_card.json").is_file():
            missing.append("no frozen task card")
    if name == "03":
        intervene = out_root / "02c" / "intervene.json"
        if intervene.is_file():
            claimed = _load_json(intervene).get("gate") or {}
            if not claimed.get("passed"):
                missing.append("the 02-C gate did not pass")
    return missing


def run_reuse_check(name: str, out_root: Path,
                    repo: Path) -> dict[str, Any]:
    """Can this package's evidence be re-derived from its raw files?

    "Evidence present" is never enough: the same verifier the ticket runs
    independently is applied to the raw artefacts, and only a clean pass allows
    the package to be skipped (``reused: true``, no engine start).
    """

    probe = Gate(f"reuse:{name}")
    detail: dict[str, Any] = {"package": name, "checked": False,
                              "reusable": False, "reason": None}
    try:
        if name in ("02a-sources", "02a-adapter"):
            sources_path = out_root / "02a" / "sources.json"
            adapter_path = out_root / "02a" / "adapter.json"
            if not (sources_path.is_file() and adapter_path.is_file()):
                detail["reason"] = "sources.json/adapter.json not present"
                return detail
            verify_offline_pair(json.loads(sources_path.read_text(encoding="utf-8")),
                                json.loads(adapter_path.read_text(encoding="utf-8")),
                                repo, probe)
        elif name in ("02a-prefix", "02a-reference"):
            mode = name.split("-", 1)[1]
            if not (out_root / "02a" / mode / "result.json").is_file():
                detail["reason"] = f"02a/{mode}/result.json not present"
                return detail
            verify_engine_package(out_root, mode, probe, d1_replay_plan(repo))
        elif name == "02a-compare":
            if not (out_root / "02a" / "compare.json").is_file():
                detail["reason"] = "02a/compare.json not present"
                return detail
            fresh = compare_runs(out_root)
            probe.add("compare_verdict_identical",
                      fresh.get("verdict") == "identical"
                      and int(fresh.get("different_items", -1)) == 0,
                      {"verdict": fresh.get("verdict")})
        elif name == "02b":
            if not (out_root / "02b" / "nomination.json").is_file():
                detail["reason"] = "02b/nomination.json not present"
                return detail
            verify_02b(out_root, repo, probe)
        elif name == "02c":
            if not (out_root / "02c" / "intervene.json").is_file():
                detail["reason"] = "02c/intervene.json not present"
                return detail
            verify_02c(out_root, repo, probe)
        elif name == "03":
            if not ((out_root / "03" / "pilot.json").is_file()
                    or (out_root / "03" / "smoke" / "pilot.json").is_file()):
                detail["reason"] = "no 03 pilot.json"
                return detail
            verify_03(out_root, repo, probe)
        else:
            detail["reason"] = "this package is never reused"
            return detail
    except (SourceError, EngineRefused, AdapterUnavailable, OSError,
            ValueError, KeyError) as exc:
        detail.update({"reason": f"{type(exc).__name__}: {exc}",
                       "checks": len(probe.checks)})
        return detail
    detail.update({"checked": True, "reusable": probe.passed,
                   "checks": len(probe.checks),
                   "problems": probe.problems[:8],
                   "reason": (None if probe.passed
                              else "the recomputed gate did not pass")})
    return detail


def cmd_run(args: argparse.Namespace) -> int:
    """Drive ``sources -> ... -> pilot -> verify`` with one ledger.

    A package whose evidence already re-verifies from its raw files is
    **reused** (``reused: true``) and no engine is started for it.  A package
    whose inputs are missing is ``not-run`` with the reason; a package whose gate
    fails stops only itself.  ``run.json`` is rewritten before and after every
    package, and ``verify`` is invoked last.
    """

    started = time.perf_counter()
    out_root = Path(args.out).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    limits = BudgetLimits(max_starts=args.max_starts,
                          max_engine_frames=args.max_engine_frames,
                          max_wall_s=args.max_wall_s,
                          max_write_gib=args.max_write_gib)
    ledger = Ledger(out_root / "ledger.jsonl", limits)
    baseline = dict(ledger.totals()["actual"])
    wanted = run_selection(args)
    gates: dict[str, Any] = {}
    codes: dict[str, int] = {}
    reused: dict[str, Any] = {}
    begun = now_iso()

    def fresh_totals() -> dict[str, Any]:
        """Re-read the ledger: a packaged command settles into the same file."""

        try:
            return Ledger(out_root / "ledger.jsonl", limits).totals()
        except (OSError, ValueError):  # pragma: no cover - defensive
            return ledger.totals()

    def payload(status: str) -> dict[str, Any]:
        failed = [name for name, value in gates.items()
                  if value.get("status") in ("failed", "technical")]
        totals = fresh_totals()
        return {
            "schema": SCHEMA_RUN,
            "generated_by": "tools/rl/v8_31_followup.py run",
            "created_at": now_iso(),
            "classification": ("technical_smoke_not_a_pilot_result"
                               if getattr(args, "smoke", False) else "measured"),
            "repo": repo.as_posix(),
            "out": out_root.as_posix(),
            "argv": list(getattr(args, "argv", [])) or None,
            "host": host_identity(),
            "heads": git_identity(repo),
            "engine": args.engine,
            "data_source": args.data_source,
            "started_at": begun,
            "finished_at": (None if status == "running" else now_iso()),
            "deadline": limits.deadline(),
            "limits": limits.as_dict(),
            "status": status,
            "order": [entry["name"] for entry in RUN_PACKAGES],
            "selection": wanted,
            "gates": gates,
            "exit_codes": codes,
            "reused": reused,
            "ledger_totals": totals,
            "failed": failed,
            "next_step": next_step_for(gates, wanted),
            "engine_starts": int(round(
                float(totals["actual"]["starts"])
                - float(baseline.get("starts", 0.0)))),
            "engine_frames": int(round(
                float(totals["actual"]["engine_frames"])
                - float(baseline.get("engine_frames", 0.0)))),
            "cost": {"seconds": round(time.perf_counter() - started, 3)},
        }

    def flush(status: str = "running") -> Path:
        return write_json(out_root / "run.json", payload(status))

    for entry in RUN_PACKAGES:
        name = entry["name"]
        gate_payload = {"package": name, "commands": list(entry["commands"]),
                        "engine": bool(entry["engine"])}
        if name == "verify":
            gate_payload["verify_package"] = run_verify_selection(out_root)
        if name not in wanted:
            gates[name] = {**gate_payload, "status": "skipped",
                           "reason": "--only/--skip excluded this package"}
            continue
        attempt_id = f"run-{name}"
        try:
            ledger.reserve(attempt_id, starts=0, engine_frames=0, wall_s=0.0,
                           write_gib=0.0,
                           inputs={"package": name, "engine": bool(entry["engine"])},
                           note="run bookkeeping row; a packaged command keeps its "
                                "own engine attempt rows")
        except BudgetRefused as exc:
            gates[name] = {**gate_payload, "status": "budget_refused",
                           "exceeded": exc.payload.get("exceeded")}
            codes[name] = EXIT_NOT_IMPLEMENTED
            continue
        if name != "verify":
            reuse = run_reuse_check(name, out_root, repo)
            reuse["package"] = name
            reused[name] = reuse
            if reuse["reusable"]:
                gates[name] = {**gate_payload, "status": "reused",
                               "reused": True,
                               "checks": reuse.get("checks"),
                               "note": ("the package's raw evidence re-derives its "
                                        "own gate, so it was not re-run and no "
                                        "engine was started")}
                codes[name] = EXIT_OK
                ledger.settle(attempt_id, starts=0, engine_frames=0, wall_s=0.0,
                              write_gib=0.0,
                              result={"package": name, "reused": True})
                flush()
                continue
        missing = run_prerequisites(name, out_root, args)
        if missing:
            gates[name] = {**gate_payload, "status": "not-run",
                           "reused": False, "reason": "; ".join(missing)}
            ledger.settle(attempt_id, starts=0, engine_frames=0, wall_s=0.0,
                          write_gib=0.0,
                          result={"package": name, "not_run": missing})
            flush()
            continue
        gates[name] = {**gate_payload, "status": "running", "reused": False}
        flush()
        snapshot = run_evidence_snapshot(out_root, name)
        package_codes = []
        error = None
        for command in entry["commands"]:
            handler = globals().get(f"cmd_{command}")
            if handler is None:  # pragma: no cover - the table is frozen
                error = f"no handler for {command}"
                break
            try:
                code = int(handler(run_package_namespace(args, command, out_root,
                                                         repo)))
            except SystemExit as exc:  # argparse/budget exit inside a package
                code = int(exc.code or 0)
            except Exception as exc:  # noqa: BLE001
                code = EXIT_TECHNICAL
                error = f"{type(exc).__name__}: {exc}"
            package_codes.append({"command": command, "exit_code": code})
            if code not in (EXIT_OK,):
                break
        codes[name] = (package_codes[-1]["exit_code"] if package_codes
                       else EXIT_TECHNICAL)
        restored = (run_evidence_restore(out_root, snapshot)
                    if codes[name] != EXIT_OK else [])
        status = ("passed" if package_codes
                  and all(item["exit_code"] == EXIT_OK for item in package_codes)
                  else ("technical" if codes[name] == EXIT_TECHNICAL
                        else "failed"))
        gates[name] = {**gate_payload, "status": status, "reused": False,
                       "exit_code": codes[name],
                       "commands_run": package_codes,
                       "restored_evidence": restored,
                       **({"evidence_note": (
                           "the failed re-run's record was replaced by the "
                           "snapshot of the previously passing record; nothing "
                           "that passed was lost")} if restored else {}),
                       **({"error": error} if error else {})}
        if restored:
            print(f"run: {name}: the re-run failed, restored "
                  f"{', '.join(restored)} from the pre-run snapshot")
        ledger.settle(attempt_id, starts=0, engine_frames=0, wall_s=0.0,
                      write_gib=0.0,
                      result={"package": name, "status": status,
                              "commands": package_codes,
                              "restored": restored})
        flush()
    final = payload("finished")
    path = write_json(out_root / "run.json", final)
    for name in [entry["name"] for entry in RUN_PACKAGES]:
        value = gates.get(name) or {}
        print(f"run: {name:14s} {value.get('status')}"
              f"{'' if not value.get('reused') else ' (reused)'} "
              f"rc={codes.get(name)}")
    print(f"wrote: {path}")
    worst = [code for code in codes.values() if code != EXIT_OK]
    if not worst:
        return EXIT_OK
    for code in (EXIT_TECHNICAL, EXIT_NOT_IMPLEMENTED, EXIT_GATE_FAILED):
        if code in worst:
            return code
    return EXIT_GATE_FAILED


def next_step_for(gates: Mapping[str, Any], wanted: Sequence[str]) -> str:
    """One sentence: what the next invocation should do."""

    pending = [name for name in wanted
               if (gates.get(name) or {}).get("status") in
               ("not-run", "failed", "technical", "budget_refused")]
    if not pending:
        return ("nothing is pending: every selected package passed or was reused; "
                "read 03/pilot.json for the pilot gate")
    hints = {
        "02a-prefix": "run `prefix --engine ... --data-source ...` (one start)",
        "02a-reference": "run `reference --engine ... --data-source ...` (one start)",
        "02a-compare": "run `compare --out ...` once both 02-A runs exist",
        "02b": "run `recon` then `nominate` (one start + offline)",
        "02c": "run `intervene --engine ... --data-source ...` (two starts)",
        "03": "run `pilot --engine ... --data-source ...` (one fit + the arms)",
        "verify": "run `verify --package all`",
    }
    return "; ".join(f"{name}: {hints.get(name, 'inspect the gate')}"
                     for name in pending)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
PACKAGE_HELP = {
    "prefix": "02-A.3 replay the D1 prefix (one engine start)",
    "reference": "02-A.4 an independent continuous reference run (one engine start)",
    "compare": "02-A.4 compare the two recorded runs (offline)",
    "recon": "02-B teacher tape recon for the attack nomination (one engine start)",
    "nominate": "02-B nominate and freeze one attack card (offline)",
    "intervene": "02-C teacher continuation vs constant #12 (two engine starts)",
    "pilot": "03 one frozen pixel-BC fit and the three arms",
    "strength": ("03 strength check on the fit root: the frozen open-loop table vs "
                 "the frozen #12 control (two engine starts)"),
    "diag-teacher": ("teacher-arm diagnostic on the four unexposed dev roots: the "
                     "frozen teacher continuation from the frozen entry (four "
                     "engine starts)"),
}


def _add_offline_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--repo", default=str(REPO_ROOT),
                     help="repository root to bind the sources to (default: this "
                          "checkout)")
    sub.add_argument("--out", required=True,
                     help="run directory; 02a/<sources|adapter>.json is written "
                          "under it")


def _add_engine_args(sub: argparse.ArgumentParser) -> None:
    _add_offline_args(sub)
    sub.add_argument("--engine", required=True,
                     help="path to the rl_v3 th08-modern harness binary")
    sub.add_argument("--data-source", required=True, dest="data_source",
                     help="read-only game data directory (copied, never written)")


def _add_budget_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--max-starts", type=int, default=28, dest="max_starts",
                     help="engine starts allowed for this run (default 28)")
    sub.add_argument("--max-engine-frames", type=int, default=360_000,
                     dest="max_engine_frames",
                     help="engine frames allowed for this run (default 360000)")
    sub.add_argument("--max-wall-s", type=float, default=8 * 3600.0,
                     dest="max_wall_s",
                     help="wall-clock seconds allowed from the booking time "
                          "(default 28800)")
    sub.add_argument("--max-write-gib", type=float, default=18.0,
                     dest="max_write_gib",
                     help="new bytes allowed on disk, GiB (default 18)")


def _add_stage_budget_args(sub: argparse.ArgumentParser) -> None:
    """The 02-B/C allowance: 3 starts / 20,000 frames **for this stage**.

    These are not the ticket's ceilings: :func:`stage_budget` reads the shared
    ``ledger.jsonl`` first, then checks ``this stage's rows + this package``
    against these numbers *and* the whole-ticket totals (28 starts / 360,000
    frames / 8 h / 18 GiB) that ``--max-wall-s`` and ``--max-write-gib`` restate.
    """

    sub.add_argument("--max-starts", type=int, default=STAGE_CAP_STARTS,
                     dest="max_starts",
                     help=f"02-B/C stage starts allowed (default {STAGE_CAP_STARTS})")
    sub.add_argument("--max-engine-frames", type=int, default=STAGE_CAP_FRAMES,
                     dest="max_engine_frames",
                     help=("02-B/C stage engine frames allowed "
                           f"(default {STAGE_CAP_FRAMES})"))
    sub.add_argument("--max-wall-s", type=float, default=GRAND_TOTAL_WALL_S,
                     dest="max_wall_s",
                     help=("whole-ticket wall seconds from the booking time "
                           f"(default {int(GRAND_TOTAL_WALL_S)})"))
    sub.add_argument("--max-write-gib", type=float, default=GRAND_TOTAL_WRITE_GIB,
                     dest="max_write_gib",
                     help=("whole-ticket new bytes, GiB "
                           f"(default {GRAND_TOTAL_WRITE_GIB})"))


def build_parser() -> argparse.ArgumentParser:
    """The CLI of SPEC §1: every subcommand has ``--help`` without an engine."""

    parser = argparse.ArgumentParser(
        prog="v8_31_followup.py",
        description=(
            "V831-SLICE-02 follow-up.  `sources` (02-A.1) and `adapter` (02-A.2) "
            "are offline; `prefix` (02-A.3) and `reference` (02-A.4) each start "
            "the engine once through the shared budget ledger; `compare`, "
            "`verify` and `run` never start one."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes: 0 gate passed, 1 gate failed with evidence written, "
            "2 technical failure, 3 not implemented / refused by the budget.\n"
            "Nothing in this step starts the engine."))
    subparsers = parser.add_subparsers(dest="command", required=True)

    sources = subparsers.add_parser(
        "sources", help="02-A.1: the four registered sources and the alignments",
        description=("Read the four registered corpora as four identities; check "
                     "the registered sha256 values; report frame grid, mask "
                     "statistics and hash coverage; align d1/s1c, d1/t04, "
                     "d1/t02c and t02c/t04 only inside the recorded intervals. "
                     "Writes <out>/02a/sources.json."))
    _add_offline_args(sources)

    adapter_cmd = subparsers.add_parser(
        "adapter", help="02-A.2: bind the frozen rebuild path and decode every pair",
        description=("Bind _StudentSession._pair_cand/.replay, release_due, "
                     "commit_frames, mask_to_cand/cand_to_mask, contracts and "
                     "obs_spec_v3_digest by inspect.getsourcefile + sha256; decode "
                     "every recorded two-frame pair with one expected-mask rule and "
                     "one named tolerance.  Writes <out>/02a/adapter.json."))
    _add_offline_args(adapter_cmd)

    for name in ("prefix", "reference"):
        sub = subparsers.add_parser(
            name, help=PACKAGE_HELP[name],
            description=(f"{PACKAGE_HELP[name]}.  One engine start, one RESET, one "
                         "continuous walk over D1's recorded masks; xvfb-run is "
                         "applied automatically unless --no-xvfb.  "
                         + ("Writes <out>/02a/prefix/{result,takeover,frames,"
                            "history,hashes}.json[l], mute.json and frames/*.png."
                            if name == "prefix" else
                            "Writes <out>/02a/reference/{result,takeover,frames,"
                            "history,hashes}.json[l], mute.json and frames/*.png.")))
        _add_engine_args(sub)
        _add_budget_args(sub)
        sub.add_argument("--timeout-s", type=float, default=120.0, dest="timeout_s",
                         help="per-command protocol timeout (default 120)")
        sub.add_argument("--no-xvfb", action="store_true", dest="no_xvfb",
                         help="do not re-exec under xvfb-run (only when the current "
                              "DISPLAY is known to be usable)")

    recon = subparsers.add_parser(
        "recon", help=PACKAGE_HELP["recon"],
        description=(f"{PACKAGE_HELP['recon']}.  RESET 0 1 0 12, then one STEP "
                     "mask 1 per frame with the recorded T04 masks: every frame's "
                     "EVENTS word is read, and after each decision's two frames "
                     "HAZARDS (all enemy slots) + STATE + HUD + PLAYER are taken.  "
                     "Stops on a technical conflict, STATE.stage leaving 0, "
                     "game_over/a death, or 13,195 frames (the decision grid stops "
                     "at 13,194).  Writes 02b/recon/{decisions.jsonl,result.json}. "
                     "No policy is consulted."))
    _add_engine_args(recon)
    _add_stage_budget_args(recon)
    recon.add_argument("--timeout-s", type=float, default=120.0, dest="timeout_s",
                       help="per-command protocol timeout (default 120)")
    recon.add_argument("--no-xvfb", action="store_true", dest="no_xvfb",
                       help="do not re-exec under xvfb-run (only when the current "
                            "DISPLAY is known to be usable)")

    intervene = subparsers.add_parser(
        "intervene", help=PACKAGE_HELP["intervene"],
        description=(f"{PACKAGE_HELP['intervene']}.  Both arms walk the frozen "
                     "prefix from their own RESET, must reproduce the card's entry "
                     "hash, then execute their controller for at most 300 "
                     "decisions / 600 frames while the frozen generation is "
                     "tracked on every decision.  Writes "
                     "02c/{intervene.json,arm_T/*,arm_C/*}."))
    _add_engine_args(intervene)
    _add_stage_budget_args(intervene)
    intervene.add_argument("--timeout-s", type=float, default=120.0,
                           dest="timeout_s",
                           help="per-command protocol timeout (default 120)")
    intervene.add_argument("--no-xvfb", action="store_true", dest="no_xvfb",
                           help="do not re-exec under xvfb-run (only when the "
                                "current DISPLAY is known to be usable)")

    compare = subparsers.add_parser(
        "compare", help=PACKAGE_HELP["compare"],
        description=(f"{PACKAGE_HELP['compare']}.  Re-reads both runs' raw files, "
                     "re-decodes the stored PNGs and reports every equality item "
                     "with its first difference.  Writes <out>/02a/compare.json."))
    compare.add_argument("--out", required=True, help="run directory to compare")

    nominate = subparsers.add_parser(
        "nominate", help=PACKAGE_HELP["nominate"],
        description=(f"{PACKAGE_HELP['nominate']}.  Segments every enemy slot's "
                     "generations, applies the kill-vs-retreat rule table to each "
                     "ending, and freezes the earliest BOSS kill that completes "
                     "within 600 frames of its entry into 02b/task_card.{json,md}; "
                     "when nothing is freezable it writes the near-miss table and "
                     "keeps the gate failed.  Never starts an engine."))
    nominate.add_argument("--repo", default=str(REPO_ROOT))
    nominate.add_argument("--out", required=True)
    nominate.add_argument("--recon", required=True, help="the 02b recon directory")
    nominate.add_argument("--max-starts", type=int, default=STAGE_CAP_STARTS,
                          dest="max_starts",
                          help=("02-B/C stage starts to price the 02-C pair "
                                f"against (default {STAGE_CAP_STARTS})"))
    nominate.add_argument("--max-engine-frames", type=int,
                          default=STAGE_CAP_FRAMES, dest="max_engine_frames",
                          help=("02-B/C stage frames to price the 02-C pair against "
                                f"(default {STAGE_CAP_FRAMES})"))

    pilot = subparsers.add_parser(
        "pilot", help=PACKAGE_HELP["pilot"],
        description=(f"{PACKAGE_HELP['pilot']}.  Writes (freezes) the "
                     "registration `configs/v8_31/pilot_03_roots.json` first, "
                     "fits once (init seed 0, Adam lr 3e-4, batch 32, exactly 200 "
                     "updates, cross entropy over the 18 executed candidates), "
                     "freezes the open-loop table on the independent donor root, "
                     "and only then starts the arms (initial weights, final "
                     "weights, frozen table) on every development root.  "
                     "`--only-seed 1000` is the technical smoke: every artefact "
                     "lands under 03/smoke/ and is marked "
                     "technical_smoke_not_a_pilot_result."))
    _add_engine_args(pilot)
    _add_budget_args(pilot)
    pilot.add_argument("--config", default=None,
                       help=("path of the frozen registration (default: "
                             f"<repo>/{PILOT_CONFIG_REL}); an existing file is "
                             "verified and reused, never overwritten"))
    pilot.add_argument("--only-seed", type=int, action="append",
                       dest="only_seed", default=None,
                       help=("restrict the development roots (repeatable); "
                             f"{PILOT_SMOKE_SEED} is the technical smoke"))
    pilot.add_argument("--smoke", action="store_true", dest="smoke",
                       help=(f"technical smoke: everything under 03/smoke/, "
                             f"classification {PILOT_SMOKE_CLASSIFICATION}"))
    pilot.add_argument("--reuse", action="store_true", dest="reuse",
                       help=("reuse a frozen open-loop table whose table_sha256 "
                             "recomputes; the fit is still performed"))
    pilot.add_argument("--timeout-s", type=float, default=120.0,
                       dest="timeout_s",
                       help="per-command protocol timeout (default 120)")
    pilot.add_argument("--no-xvfb", action="store_true", dest="no_xvfb",
                       help="do not re-exec under xvfb-run (only when the "
                            "current display is known to be usable)")
    pilot.add_argument("--strength-seed", type=int, default=None,
                       dest="strength_seed",
                       help=("run the 03 strength check on this seed instead of the "
                             f"pilot; only the fit root ({PILOT_FIT_SEED}) is "
                             "accepted and it writes 03/strength/ only — the same "
                             "code path as the `strength` subcommand"))

    strength = subparsers.add_parser(
        "strength", help=PACKAGE_HELP["strength"],
        description=(f"{PACKAGE_HELP['strength']}.  Reads the frozen card, the "
                     "frozen tape prefix (decisions 0..1542), the frozen #12 "
                     "definition and the frozen open-loop table "
                     f"({STRENGTH_REQUIRED_TABLE_SHA256[:8]}…), re-checks every "
                     "digest, and runs exactly two engine starts on the fit root "
                     f"(seed {PILOT_FIT_SEED}): `open_loop` plays the table and "
                     "`constant_12` runs the frozen control #12, both with "
                     "release_due(i, 48).  Writes 03/strength/{result.json,"
                     "arm_<arm>/*} classified "
                     f"{STRENGTH_CLASSIFICATION}.  Never edits the pilot, the card, "
                     "the config or the table."))
    _add_engine_args(strength)
    _add_budget_args(strength)
    strength.add_argument("--seed", type=int, default=PILOT_FIT_SEED, dest="seed",
                          help=(f"the fit root to check (only {PILOT_FIT_SEED} is "
                                "accepted)"))
    strength.add_argument("--timeout-s", type=float, default=120.0,
                          dest="timeout_s",
                          help="per-command protocol timeout (default 120)")
    strength.add_argument("--no-xvfb", action="store_true", dest="no_xvfb",
                          help="do not re-exec under xvfb-run (only when the "
                               "current display is known to be usable)")

    diag_teacher = subparsers.add_parser(
        "diag-teacher", help=PACKAGE_HELP["diag-teacher"],
        description=(f"{PACKAGE_HELP['diag-teacher']}.  Re-reads the frozen card, "
                     f"the frozen registration ({PILOT_CONFIG_REL}), the frozen "
                     "tape prefix (decisions 0..1542) and the card's own teacher "
                     "continuation (decisions 1543..1842), proves three digests "
                     "agree, prices the four starts against the shared ledger and "
                     "then starts one engine per unexposed root (1020-1023) in seed "
                     "order.  Each root records the entry (frame 3,086, stage 0, "
                     "alive, frozen slot present), the card's entry hash (recorded, "
                     "**not** an arrival predicate: it was measured on seed 12), the "
                     "600-frame teacher window and the card's own termination "
                     "priority.  Writes 03/diag/teacher_arm/{<seed>/result.json,"
                     "result.json} classified "
                     f"{DIAG_TEACHER_CLASSIFICATION}; `--only-seed` restricts the "
                     "roots and the CLI can never widen the 4-start/15,000-frame "
                     "allowance."))
    _add_engine_args(diag_teacher)
    _add_budget_args(diag_teacher)
    diag_teacher.add_argument("--config", default=None,
                              help=("path of the frozen registration (default: "
                                    f"<repo>/{PILOT_CONFIG_REL}); it is read and "
                                    "re-verified, never written"))
    diag_teacher.add_argument("--only-seed", type=int, action="append",
                              dest="only_seed", default=None,
                              help=("restrict the roots (repeatable); only "
                                    f"{list(DIAG_TEACHER_SEEDS)} are accepted"))
    diag_teacher.add_argument("--timeout-s", type=float, default=120.0,
                              dest="timeout_s",
                              help="per-command protocol timeout (default 120)")
    diag_teacher.add_argument("--no-xvfb", action="store_true", dest="no_xvfb",
                              help="do not re-exec under xvfb-run (only when the "
                                   "current display is known to be usable)")

    run = subparsers.add_parser(
        "run", help="run the package sequence with one ledger",
        description=("Drive sources -> adapter -> prefix -> reference -> compare -> "
                     "recon -> nominate -> intervene -> pilot -> verify in that "
                     "order with one ledger.  A package whose raw evidence "
                     "re-verifies is reused (`reused: true`) and starts no engine; a "
                     "package with missing inputs is `not-run` with the reason.  "
                     "<out>/run.json is rewritten before and after every package and "
                     "verify runs last."))
    run.add_argument("--repo", default=str(REPO_ROOT))
    run.add_argument("--out", required=True)
    run.add_argument("--engine", default=None, help="engine binary")
    run.add_argument("--data-source", default=None, dest="data_source",
                     help="read-only game data directory")
    _add_budget_args(run)
    run.add_argument("--only", default="",
                     help=("comma separated packages/commands to run "
                           "(sources,adapter,prefix,reference,compare,recon,"
                           "nominate,intervene,pilot,verify or the package names)"))
    run.add_argument("--skip", default="",
                     help="comma separated packages/commands to skip")
    run.add_argument("--config", default=None,
                     help="frozen pilot registration forwarded to `pilot`")
    run.add_argument("--only-seed", type=int, action="append",
                     dest="only_seed", default=None,
                     help="forwarded to `pilot` (repeatable)")
    run.add_argument("--smoke", action="store_true", dest="smoke",
                     help="forwarded to `pilot`")
    run.add_argument("--timeout-s", type=float, default=120.0,
                     dest="timeout_s",
                     help="per-command protocol timeout (default 120)")
    run.add_argument("--no-xvfb", action="store_true", dest="no_xvfb",
                     help="do not re-exec under xvfb-run")

    verify = subparsers.add_parser(
        "verify", help="re-derive every implemented gate from the raw outputs",
        description=("Recompute the source sha256 values and the counters, re-apply "
                     "the decode gate, re-check the imported code and the ledger, "
                     "and re-derive the two 02-A engine packages, `compare`, 02-B, "
                     "02-C, the 03 pilot (registration sha, both weight shas, "
                     "train.json, the open-loop table digest, every arm) and the "
                     "run.json consistency from their raw artefacts.  Never starts "
                     "an engine.  Writes <out>/verify.json."))
    verify.add_argument("--out", required=True)
    verify.add_argument("--repo", default=str(REPO_ROOT),
                        help="repository root to re-read D1 from (default: this "
                             "checkout)")
    verify.add_argument("--package", default="02a",
                        choices=("02a", "02a-prefix", "02a-reference", "02b", "02c",
                                 "03", "all"),
                        help="which packages to re-derive: 02a (sources+adapter), "
                             "02a-prefix, 02a-reference, 02b, 02c, 03 (the pilot), "
                             "or all (every package whose evidence exists)")
    return parser


DISPATCH = {
    "sources": cmd_sources,
    "adapter": cmd_adapter,
    "verify": cmd_verify,
    "run": cmd_run,
    "compare": cmd_compare,
    "prefix": cmd_prefix,
    "reference": cmd_reference,
    "recon": cmd_recon,
    "nominate": cmd_nominate,
    "intervene": cmd_intervene,
    "pilot": cmd_pilot,
    "strength": cmd_strength,
    "diag-teacher": cmd_diag_teacher,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Parse and dispatch; only ``prefix``/``reference`` can start an engine."""

    parser = build_parser()
    args = parser.parse_args(argv)
    args.argv = list(sys.argv if argv is None else argv)
    handler = DISPATCH.get(args.command, cmd_not_implemented)
    try:
        return int(handler(args))
    except AdapterUnavailable as exc:
        print(f"technical failure: {exc}", file=sys.stderr)
        return EXIT_TECHNICAL
    except (SourceError, EngineRefused, FrameRelationConflict) as exc:
        print(f"technical failure: {exc}", file=sys.stderr)
        return EXIT_TECHNICAL
    except BudgetRefused as exc:
        print(f"budget_refused: {exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED


if __name__ == "__main__":
    raise SystemExit(main())
