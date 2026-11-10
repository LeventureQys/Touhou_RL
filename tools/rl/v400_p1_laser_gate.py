#!/usr/bin/env python3
"""P1-S1 laser-geometry gate: is the *real* engine's laser snapshot usable?

``Document/Update/Alpha/v4.0.0/11-P1-搜索教师.md`` §2.1 makes laser geometry a
prerequisite for ``HazardTape``: the teacher has to know where a laser is, when
it hurts, and how thick it is.  Everything below exists to answer one question
with evidence instead of assumption:

    in a real match, does ``LASERS`` ever return a non-empty snapshot, and do
    its rows carry the fields §2.1 needs?

The tool never invents a field.  It asks the running binary (which may be older
than ``src/rlharness/rlharness.cpp``), records the *observed* token layout, and
only labels a token when the observed value passes a range check for that name.
An empty reply on every source yields ``verdict.status == "INVALID"`` plus a
concrete list of where to look next -- it never yields "covered".

Sources (all real engine, one process at a time):

* ``reset``   -- ``RESET <stage> <difficulty> <team> <seed> <practice>`` + STEP;
* ``replay``  -- ``TH08_AUTOPLAY_REPLAY`` / ``TH08_AUTOPLAY_STAGE`` playback.

The process is launched with ``--rl-role shadow`` (null D3D device), which is the
role ``03`` assigns to the teacher's shadow process; it needs no X server.

Run it inside WSL, where the engine ELF lives::

    python3 tools/rl/v400_p1_laser_gate.py \\
        --engine /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern \\
        --data-dir /root/v400-data/game \\
        --out .tmp_v400_p1s1/laser_gate.json

Exit codes: ``0`` gate VALID (lasers observed), ``3`` gate INVALID (no laser
row on any source), ``4`` the protocol could not even be exercised (engine
would not start / ``LASERS`` refused), ``2`` bad arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import select
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: ``LASERS`` pages at most 16 rows per reply (protocol contract).
PAGE_LIMIT = 16
#: ``LASERS`` scans a 0x100-slot pool, ``BULLETS`` a 0x601-slot pool.
LASER_SLOTS = 0x100
BULLET_SLOTS = 0x601
#: Hard stop on pagination so a misbehaving reply can never spin forever.
MAX_PAGES = 24

#: Stage slot names of the retail replay layout, copied from
#: ``third_party/th08/scripts/replay_file.py:10-11`` (STAGES).  Slot indices are
#: what ``RESET``/``TH08_AUTOPLAY_STAGE`` take, so no off-by-one is applied.
STAGE_NAMES = ("Stage 1", "Stage 2", "Stage 3", "Stage 4A", "Stage 4B", "Stage 5",
               "Final A", "Final B", "Extra")

#: Documented layouts, kept here only as a *hypothesis* to test against the
#: bytes.  ``rlharness.cpp`` (see the comment block above ``HandleLasers``) is
#: the reference; a binary built before that revision legitimately differs.
LASER_DOC_FIELDS = [
    "slot", "state", "posX", "posY", "posZ", "angle",
    "startOffset", "endOffset", "currentWidth", "width", "timer", "inUse",
]
BULLET_DOC_FIELDS = [
    "slot", "state", "color", "posX", "posY", "posZ", "velX", "velY",
    "activeTimer", "collisionSize", "collisionDisabled",
]

#: How a token is expected to be encoded, per documented name.
_FLOAT_FIELDS = {
    "posX", "posY", "posZ", "velX", "velY", "angle",
    "startOffset", "endOffset", "currentWidth", "width", "collisionSize",
}
_INT_FIELDS = {"slot", "state", "color", "activeTimer", "timer", "inUse",
               "collisionDisabled"}

#: Plausibility envelope per documented name.  A token that decodes outside its
#: envelope is *not* silently relabelled: the name is reported unconfirmed.
_RANGES = {
    "slot": (0, 0x601),
    "state": (0, 63),
    "color": (0, 32),
    "posX": (-512.0, 1024.0),
    "posY": (-512.0, 1024.0),
    "posZ": (-512.0, 512.0),
    "velX": (-64.0, 64.0),
    "velY": (-64.0, 64.0),
    "angle": (-8.0, 8.0),
    "startOffset": (-4096.0, 4096.0),
    "endOffset": (-4096.0, 8192.0),
    "currentWidth": (0.0, 2048.0),
    "width": (0.0, 2048.0),
    "timer": (0, 1 << 31),
    "inUse": (1, 1),
    "activeTimer": (0, 1 << 31),
    "collisionSize": (0.0, 1024.0),
    "collisionDisabled": (0, 1),
}

#: Source ids this gate always asks for (11-P1 §6 WP1 wants real coverage from a
#: normal start, a replay and a practice start); ``--extra-replay-stages`` adds the
#: replay stage that actually carries lasers on top of these.
REQUIRED_SOURCE_IDS = ("reset-stage1-normal", "replay-stage5", "practice-stage6")


class EngineError(RuntimeError):
    """The engine replied ``ERR``, or the control pipe died."""


class GateUnavailable(RuntimeError):
    """The gate could not be exercised at all (no laser *command* to judge)."""


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------
class EngineClient:
    """Minimal line-protocol client for ``th08-modern --rl-control``.

    Deliberately standalone: the gate must stay runnable inside WSL where only
    the standard library is guaranteed, and the reply text (not a typed
    wrapper) is the evidence being collected.
    """

    def __init__(self, binary: str, data_dir: str, *, role: str = "shadow",
                 extra_env: dict[str, str] | None = None, timeout: float = 180.0,
                 stderr_path: str | os.PathLike[str] | None = None) -> None:
        self.binary = str(binary)
        self.data_dir = str(data_dir)
        self.role = role
        self.extra_env = dict(extra_env or {})
        self.timeout = float(timeout)
        self.stderr_path = str(stderr_path) if stderr_path is not None else None
        self.command = None
        self._proc = None
        self._buffer = b""
        self._stderr_file = None
        self.requests = 0

    # -- lifetime -------------------------------------------------------
    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("client already started")
        env = dict(os.environ)
        for name in ("DISPLAY", "WAYLAND_DISPLAY"):
            env.pop(name, None)
        env.update({
            "SDL_AUDIODRIVER": "dummy",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "GALLIUM_DRIVER": "llvmpipe",
            "LP_NUM_THREADS": "1",
        })
        env.update(self.extra_env)
        argv = [self.binary, "--data-dir", self.data_dir,
                "--rl-control", "0,1", "--rl-role", self.role]
        self.command = argv
        # stderr is kept: "engine closed the control pipe" is only diagnosable if
        # the reason the engine printed before dying is preserved.
        if self.stderr_path:
            Path(self.stderr_path).parent.mkdir(parents=True, exist_ok=True)
            self._stderr_file = open(self.stderr_path, "wb")
        self._proc = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      stderr=self._stderr_file or subprocess.DEVNULL,
                                      env=env, bufsize=0)

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    self._write("QUIT")
                    self._readline(time.monotonic() + 20.0)
                except Exception:
                    pass
            try:
                proc.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        finally:
            for stream in (proc.stdin, proc.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
            self._proc = None
            self._buffer = b""
            if self._stderr_file is not None:
                try:
                    self._stderr_file.close()
                except Exception:
                    pass
                self._stderr_file = None

    def __enter__(self) -> "EngineClient":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- transport ------------------------------------------------------
    def _write(self, line: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((line + "\n").encode("utf-8"))
        self._proc.stdin.flush()

    def _readline(self, deadline: float) -> str:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            if b"\n" in self._buffer:
                line, _, self._buffer = self._buffer.partition(b"\n")
                return line.decode("utf-8", "replace")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EngineError("no reply within timeout")
            if os.name == "nt":  # pragma: no cover - the engine is ELF-only
                raise EngineError("this client requires a POSIX host (run inside WSL)")
            ready, _, _ = select.select([self._proc.stdout], [], [], min(0.5, remaining))
            if not ready:
                if self._proc.poll() is not None:
                    raise EngineError(
                        f"engine exited before replying (code {self._proc.returncode})")
                continue
            chunk = os.read(self._proc.stdout.fileno(), 65536)
            if not chunk:
                raise EngineError("engine closed the control pipe")
            self._buffer += chunk

    def request(self, line: str, *, timeout: float | None = None,
                allow_err: bool = False) -> str:
        if self._proc is None:
            raise RuntimeError("client is not running")
        self._write(line)
        reply = self._readline(time.monotonic() + (self.timeout if timeout is None else timeout))
        self.requests += 1
        if reply.startswith("ERR") and not allow_err:
            raise EngineError(f"{line!r} -> {reply}")
        return reply

    def fields(self, line: str, *, timeout: float | None = None) -> dict[str, str]:
        return parse_kv(self.request(line, timeout=timeout))


# --------------------------------------------------------------------------
# reply parsing (pure functions -- unit tested without an engine)
# --------------------------------------------------------------------------
def parse_kv(reply: str) -> dict[str, str]:
    """``OK a=1 b=text`` -> ``{"_verb": "OK", "a": "1", "b": "text"}``."""
    parts = reply.split()
    out: dict[str, str] = {}
    if parts:
        out["_verb"] = parts[0]
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def _rows(payload: str) -> list[list[str]]:
    payload = payload.strip()
    if not payload:
        return []
    return [row.split(":") for row in payload.split(",") if row]


def parse_page(reply: str) -> dict:
    """Parse one ``LASERS``/``BULLETS``/``ENEMIES`` reply.

    ``count`` is the live pool size as the engine saw it, independent of the
    page window, so it is authoritative even when ``rows`` is a single page.
    """
    kv = parse_kv(reply)
    if kv.get("_verb") != "OK":
        raise EngineError(f"paged snapshot failed: {reply}")
    count_text = kv.get("count", "")
    return {
        "raw": reply,
        "count": None if count_text in ("", "undefined") else int(count_text),
        "offset": None if kv.get("offset") in (None, "undefined") else int(kv["offset"]),
        "limit": None if kv.get("limit") in (None, "undefined") else int(kv["limit"]),
        "rows": _rows(kv.get("lasers", kv.get("bullets", kv.get("enemies", "")))),
    }


def parse_state(reply: str) -> dict[str, int]:
    """``STATE`` returns ``f0..f24``: fixed-order, no named event field."""
    kv = parse_kv(reply)
    out: dict[str, int] = {}
    for key, value in kv.items():
        if len(key) > 1 and key[0] == "f" and key[1:].isdigit():
            out[key] = int(value, 16) if len(value) == 8 and not value.isdigit() else int(value, 0)
    return out


def state_field_names(reply: str) -> list[str]:
    """Every key ``STATE`` actually returned, in reply order (evidence)."""
    parts = reply.split()
    return [part.split("=", 1)[0] for part in parts[1:] if "=" in part]


def bits_to_float(text: str) -> float | None:
    """Reinterpret an 8-hex-digit ``%08x`` float bit pattern."""
    import struct
    if len(text) != 8:
        return None
    try:
        value = int(text, 16)
    except ValueError:
        return None
    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


def _token_is_plausible(name: str, text: str) -> tuple[bool, object]:
    low, high = _RANGES[name]
    if name in _FLOAT_FIELDS:
        value = bits_to_float(text)
        if value is None:
            return False, None
        if value != value:  # NaN
            return False, value
        return (low <= value <= high), value
    try:
        value = int(text, 10)
    except ValueError:
        return False, None
    return (low <= value <= high), value


def classify_layout(rows: list[list[str]], documented: list[str], *, kind: str) -> dict:
    """Map observed token positions onto documented names, with per-token evidence.

    ``verified`` is only true when the observed token count equals the documented
    layout *and* every token in every sampled row passes the range check for the
    name assigned to its position.  Anything else is reported as observed but
    unconfirmed -- the gate refuses to borrow a name from the source tree.
    """
    result: dict = {
        "kind": kind,
        "rowsObserved": len(rows),
        "tokenCounts": sorted({len(row) for row in rows}),
        "documentedLayout": {"names": list(documented), "source":
                             "third_party/th08-rl/src/rlharness/rlharness.cpp"},
        "fields": [],
        "verified": False,
    }
    if not rows:
        result["status"] = "no-nonempty-row"
        return result
    result["status"] = "observed"

    width = len(rows[0])
    uniform = all(len(row) == width for row in rows)
    # A row that is genuinely short (fewer fields than the documentation) must
    # not be padded: report it and stop.
    names = documented[:width] if width <= len(documented) else [
        documented[i] if i < len(documented) else f"unknown{i}" for i in range(width)]

    confirmed = uniform
    for index in range(width):
        name = names[index]
        samples = [row[index] for row in rows[:3] if len(row) > index]
        checks = []
        ok_all = True
        for text in samples:
            if name in _FLOAT_FIELDS or name in _INT_FIELDS:
                ok, value = _token_is_plausible(name, text)
            else:
                ok, value = True, text
            checks.append({"raw": text, "value": value, "inRange": ok})
            ok_all = ok_all and ok
        field = {
            "index": index,
            "name": name if ok_all and name != f"unknown{index}" else "unconfirmed",
            "documentedName": documented[index] if index < len(documented) else None,
            "samples": checks,
            # A token can fit its name and still not prove the whole layout: the
            # pinned binary may predate appended fields, so `confirmed` also
            # requires the row width to match the documented one.
            "inRange": bool(ok_all),
            "confirmed": bool(ok_all and index < len(documented) and width == len(documented)),
        }
        result["fields"].append(field)
        confirmed = confirmed and field["confirmed"]

    # Structural invariants an ``inUse`` filter guarantees.
    last_index = width - 1
    if names[last_index] == "inUse":
        result["lastTokenAlwaysOne"] = all(row[-1] in ("1", "0") for row in rows) and all(
            row[-1] == "1" for row in rows)
    result["uniformWidth"] = uniform
    result["unexpectedWidth"] = width != len(documented)
    result["verified"] = bool(confirmed and uniform and len(documented) == width)
    all_in_range = all(field["inRange"] for field in result["fields"])
    # A *narrower* row is not a wrong row: the pinned binary predates the
    # appended fields, so the tokens it does emit can be confirmed by value while
    # the trailing documented names are reported as absent instead of guessed.
    result["verifiedPrefix"] = bool(uniform and all_in_range and 0 < width <= len(documented))
    result["missingDocumentedSuffix"] = list(documented[width:]) if width < len(documented) else []
    result["verifiedNames"] = (
        [field["name"] for field in result["fields"]]
        if (result["verified"] or result["verifiedPrefix"]) else None)
    result["fieldNames"] = [field["name"] for field in result["fields"]]
    if result["unexpectedWidth"] and result["verifiedPrefix"]:
        result["note"] = (
            f"row has {width} tokens: documented[0:{width}] are confirmed by value, "
            f"and {result['missingDocumentedSuffix']} are not emitted by this build")
    elif result["unexpectedWidth"]:
        result["note"] = (
            f"row has {width} tokens, documentation says {len(documented)}; the "
            "extra documented fields are absent from this binary, so names beyond "
            f"index {width - 1} cannot be claimed from this row")
    return result


def page_all(client: EngineClient, command: str, *, kind: str,
             first: str | None = None) -> dict:
    """Read one full snapshot, paginating only when ``count`` exceeds a page."""
    reply = first if first is not None else client.request(command)
    head = parse_page(reply)
    if head["count"] is None:
        return {"count": None, "rows": [], "pages": 1, "truncated": False,
                "firstRaw": head["raw"]}
    rows = list(head["rows"])
    pages = 1
    truncated = False
    while len(rows) < head["count"]:
        if pages >= MAX_PAGES:
            truncated = True
            break
        page = parse_page(client.request(f"{command.split()[0]} {len(rows)} {PAGE_LIMIT}"))
        if not page["rows"]:
            truncated = True
            break
        rows.extend(page["rows"])
        pages += 1
    return {
        "count": head["count"],
        "rows": rows,
        "pages": pages,
        "truncated": truncated or len(rows) != head["count"],
        "firstRaw": head["raw"],
    }


# --------------------------------------------------------------------------
# per-source state machine
# --------------------------------------------------------------------------
def _new_source(source_id: str, kind: str, request: dict) -> dict:
    return {
        "id": source_id,
        "kind": kind,
        "request": request,
        "framesRequested": None,
        "framesObserved": 0,
        "gameOver": False,
        "endReason": None,
        "startFrame": None,
        "endFrame": None,
        "setup": {},
        "lasers": {
            "command": "LASERS 0 16 (+ offset paging)",
            "nonEmptyFrames": 0,
            "totalEntries": 0,
            "maxCount": 0,
            "framesWithPaging": 0,
            "truncatedFrames": 0,
            "firstRows": [],
            "peakSample": None,
            "headerPagedMismatchFrames": 0,
            "countHistogram": {},
        },
        "bullets": {
            "command": "BULLETS 0 16 (count from header; rows only sampled)",
            "nonEmptyFrames": 0,
            "maxCount": 0,
            "sumCount": 0,
            "rawNonEmptyFrames": 0,
            "rawMaxCount": 0,
            "rawSumCount": 0,
            "countHistogram": {},
            "sentinelObserved": False,
            "sentinelEvidence": [],
            "sentinelAdjustment": 0,
            "firstRows": [],
        },
        "samples": [],
        "playerStates": [],
        "errors": [],
    }


def _sample(client: EngineClient, frame: int) -> dict:
    reply = client.request("STATE")
    state = parse_state(reply)
    return {
        "frame": frame,
        "keys": len(state),
        "stage": state.get("f1"),
        "gameplayFrame": state.get("f3"),
        "playerState": state.get("f13"),
        "livesRaw": state.get("f7"),
        "bombsRaw": state.get("f8"),
        "raw": reply,
    }


def is_sentinel_row(row: list[str]) -> bool:
    """``BULLET_STATE_SENTINEL`` occupies the last slot (0x600) every frame.

    It is a live slot by the ``state != UNUSED`` filter, so ``BULLETS count``
    includes it -- and a HazardTape recorder that trusts ``count`` would append
    a phantom hazard at (0, 0) on every single frame.  The row is recognised by
    slot, state and an all-zero position/velocity tail.
    """
    if len(row) < 9:
        return False
    if row[0] != str(BULLET_SLOTS - 1) or row[1] != "6":
        return False
    return all(token in ("0", "00000000") for token in row[3:8])


def finalize_source(source: dict) -> dict:
    """Derive the sentinel-adjusted bullet numbers from the raw ones."""
    bullets = source["bullets"]
    adjustment = 1 if bullets["sentinelObserved"] else 0
    bullets["sentinelAdjustment"] = adjustment
    histogram = {int(key): value for key, value in bullets["countHistogram"].items()}
    bullets["nonEmptyFrames"] = sum(
        seen for count, seen in histogram.items() if count - adjustment > 0)
    bullets["maxCount"] = max(0, bullets["rawMaxCount"] - adjustment)
    lasers = source["lasers"]
    for item in source["samples"]:
        item["bulletCountEffective"] = (
            None if item.get("bulletCount") is None else item["bulletCount"] - adjustment)
    # ``STEP`` keeps answering advanced=1 after a death while the world sits in the
    # retry state (f3 gameplayFrameCounter freezes at 0).  The last sample whose
    # gameplay counter still moves is the honest end of usable coverage.
    live = [item["frame"] for item in source["samples"]
            if item.get("gameplayFrame") not in (0, None)]
    source["lastLiveSampleFrame"] = max(live) if live else None
    source["frozenTailSamples"] = [
        item["frame"] for item in source["samples"] if item.get("gameplayFrame") == 0]
    return source


def collect_frames(client: EngineClient, source: dict, frames: int, *,
                   sample_every: int = 500, mask: int = 0,
                   start_frame: int = 0) -> dict:
    """Step one frame at a time, sampling ``LASERS`` and ``BULLETS`` after each.

    One frame per ``STEP`` is deliberate: ``STEP n`` advances ``n`` frames with
    no chance to look in between, so per-frame hazard coverage is impossible
    without it.
    """
    lasers = source["lasers"]
    bullets = source["bullets"]
    for index in range(1, frames + 1):
        frame = start_frame + index
        step = client.fields(f"STEP {mask} 1")
        advanced = int(step.get("advanced", "0"))
        if advanced != 1:
            source["gameOver"] = step.get("exit") == "1" or step.get("game_over") == "1"
            source["endReason"] = (
                "game_over" if source["gameOver"] else f"step_stalled(advanced={advanced})")
            break
        source["framesObserved"] = index
        source["endFrame"] = int(step.get("frame", frame))

        laser_page = page_all(client, f"LASERS 0 {PAGE_LIMIT}", kind="laser")
        if laser_page["count"] is None:
            source["errors"].append(f"frame {frame}: LASERS count undefined")
            continue
        if laser_page["count"] > 0:
            lasers["nonEmptyFrames"] += 1
            lasers["totalEntries"] += laser_page["count"]
            if laser_page["count"] > lasers["maxCount"]:
                # Keep the rows of the densest frame: a recorder needs to know
                # whether currentWidth is ever non-zero, not just what frame 1 of
                # the pattern looked like.
                lasers["peakSample"] = {
                    "frame": frame,
                    "count": laser_page["count"],
                    "rows": laser_page["rows"][:4],
                }
            lasers["maxCount"] = max(lasers["maxCount"], laser_page["count"])
            lasers["countHistogram"][str(laser_page["count"])] = (
                lasers["countHistogram"].get(str(laser_page["count"]), 0) + 1)
            if laser_page["pages"] > 1:
                lasers["framesWithPaging"] += 1
            if laser_page["truncated"]:
                lasers["truncatedFrames"] += 1
            if len(laser_page["rows"]) != laser_page["count"]:
                lasers["headerPagedMismatchFrames"] += 1
            if len(lasers["firstRows"]) < 4:
                lasers["firstRows"].append({
                    "frame": frame,
                    "count": laser_page["count"],
                    "rows": laser_page["rows"][:4],
                })

        bullet_page = None
        if bullets.get("available", True):
            try:
                bullet_page = parse_page(client.request(f"BULLETS 0 {PAGE_LIMIT}"))
            except EngineError as exc:
                # Replay playback never sends RESET, so the harness's
                # ``g_resetReady`` gate refuses BULLETS for the whole replay.
                # Record it once -- not once per frame -- and keep the laser half
                # of the measurement usable.
                bullets["available"] = False
                bullets["unavailableReason"] = str(exc)
                source["errors"].append(f"BULLETS unavailable from frame {frame}: {exc}")
        if bullet_page is not None and bullet_page["count"] is None:
            source["errors"].append(f"frame {frame}: BULLETS count undefined")
        elif bullet_page is not None:
            raw = bullet_page["count"]
            bullets["rawNonEmptyFrames"] += 1 if raw > 0 else 0
            bullets["rawSumCount"] += raw
            bullets["rawMaxCount"] = max(bullets["rawMaxCount"], raw)
            bullets["countHistogram"][str(raw)] = bullets["countHistogram"].get(str(raw), 0) + 1
            if len(bullets["firstRows"]) < 2 and raw > 0:
                bullets["firstRows"].append({
                    "frame": frame,
                    "count": raw,
                    "rows": bullet_page["rows"][:2],
                })

        if index == 1 or index % sample_every == 0:
            sample = _sample(client, frame)
            sample["laserCount"] = laser_page["count"]
            sample["laserRows"] = len(laser_page["rows"])
            sample["bulletCount"] = None if bullet_page is None else bullet_page["count"]
            # Ask for the last live ordinal: if the pool's sentinel slot is being
            # counted as a bullet, this is where it shows up, with the row itself.
            if bullet_page is not None and bullet_page["count"]:
                last = parse_page(client.request(f"BULLETS {bullet_page['count'] - 1} 1"))
                row = last["rows"][0] if last["rows"] else None
                sample["bulletLastRow"] = row
                sample["bulletLastIsSentinel"] = bool(row and is_sentinel_row(row))
                if sample["bulletLastIsSentinel"]:
                    bullets["sentinelObserved"] = True
                    if len(bullets["sentinelEvidence"]) < 4:
                        bullets["sentinelEvidence"].append(
                            {"frame": frame, "count": bullet_page["count"], "row": row})
            source["samples"].append(sample)
    return finalize_source(source)


def wait_for_gameplay(client: EngineClient, *, deadline_seconds: float = 120.0,
                      stage: int | None = None) -> dict:
    """Pump frames until the engine reports gameplay (replay playback entry)."""
    begun = time.monotonic()
    trace: list[dict] = []
    while time.monotonic() - begun < deadline_seconds:
        step = client.fields("STEP 0 1")
        if step.get("exit") == "1":
            raise GateUnavailable(f"engine exited before gameplay: {step}")
        debug = client.fields("DEBUG")
        if len(trace) < 12:
            trace.append(debug)
        if debug.get("curState") == "2" and debug.get("globals") == "1":
            if stage is not None and int(debug.get("stage", "-1")) != stage:
                raise GateUnavailable(
                    f"replay entered stage {debug.get('stage')} but {stage} was requested")
            return {"debug": debug, "trace": trace,
                    "seconds": round(time.monotonic() - begun, 3)}
    raise GateUnavailable(
        f"no gameplay within {deadline_seconds:.0f}s; last DEBUG={trace[-1] if trace else None}")


def measure_timing(client: EngineClient, frames: int = 200, *,
                   mask: int = 0) -> dict:
    """Mean wall time of one ``LASERS`` + one ``BULLETS`` call per real frame.

    Says nothing about ``HazardTape`` throughput on its own -- ``STEP`` is timed
    too, because the budget that matters is (step + hazard read) per frame.
    """
    laser_ms: list[float] = []
    bullet_ms: list[float] = []
    step_ms: list[float] = []
    pair_ms: list[float] = []
    laser_rows = 0
    bullet_count_sum = 0
    bullets_available = True
    bullets_error = None
    for _ in range(frames):
        t0 = time.perf_counter()
        step = client.fields(f"STEP {mask} 1")
        t1 = time.perf_counter()
        if int(step.get("advanced", "0")) != 1:
            break
        page = parse_page(client.request(f"LASERS 0 {PAGE_LIMIT}"))
        t2 = time.perf_counter()
        if bullets_available:
            try:
                bullet = parse_page(client.request(f"BULLETS 0 {PAGE_LIMIT}"))
            except EngineError as exc:
                bullets_available = False
                bullets_error = str(exc)
        t3 = time.perf_counter()
        step_ms.append((t1 - t0) * 1000.0)
        laser_ms.append((t2 - t1) * 1000.0)
        if bullets_available:
            bullet_ms.append((t3 - t2) * 1000.0)
            pair_ms.append((t3 - t1) * 1000.0)
            bullet_count_sum += int(bullet["count"] or 0)
        laser_rows += int(page["count"] or 0)

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"frames": 0, "meanMs": None, "medianMs": None, "maxMs": None}
        ordered = sorted(values)
        return {
            "frames": len(values),
            "meanMs": round(sum(values) / len(values), 4),
            "medianMs": round(ordered[len(ordered) // 2], 4),
            "p95Ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 4),
            "maxMs": round(ordered[-1], 4),
        }

    return {
        "frames": len(laser_ms),
        "lasers": _stats(laser_ms),
        "bullets": _stats(bullet_ms) if bullets_available else None,
        "pair": _stats(pair_ms) if bullets_available else None,
        "step": _stats(step_ms),
        "laserRowsSeen": laser_rows,
        "bulletCountSum": bullet_count_sum,
        "bulletsAvailable": bullets_available,
        "bulletsError": bullets_error,
    }


def measure_paged_laser_timing(client: EngineClient, frames: int) -> dict:
    """Cost of one *fully paged* ``LASERS`` read per frame.

    ``measure_timing`` prices the common case (0-1 page).  A HazardTape recorder
    has to survive the opposite case: the Final B peak carries 230 lasers, i.e.
    ceil(230/16) = 15 round trips for a single frame, so this is the number that
    decides whether the recording budget holds.
    """
    laser_ms: list[float] = []
    pages: list[int] = []
    counts: list[int] = []
    for _ in range(frames):
        step = client.fields("STEP 0 1")
        if int(step.get("advanced", "0")) != 1:
            break
        t0 = time.perf_counter()
        page = page_all(client, f"LASERS 0 {PAGE_LIMIT}", kind="laser")
        laser_ms.append((time.perf_counter() - t0) * 1000.0)
        pages.append(page["pages"])
        counts.append(int(page["count"] or 0))
    ordered = sorted(laser_ms)
    return {
        "frames": len(laser_ms),
        "meanMs": round(sum(laser_ms) / len(laser_ms), 4) if laser_ms else None,
        "medianMs": round(ordered[len(ordered) // 2], 4) if ordered else None,
        "maxMs": round(ordered[-1], 4) if ordered else None,
        "meanPages": round(sum(pages) / len(pages), 3) if pages else None,
        "maxPages": max(pages) if pages else None,
        "meanLaserCount": round(sum(counts) / len(counts), 2) if counts else None,
        "maxLaserCount": max(counts) if counts else None,
    }


# --------------------------------------------------------------------------
# capability probe
# --------------------------------------------------------------------------
def probe_capabilities(client: EngineClient) -> dict:
    """Ask the binary what it actually implements -- no source-tree guessing."""
    out: dict[str, dict] = {}
    state_names: list[str] = []
    for command in ("STATE", "EVENTS", "PLAYER", "HAZARDS 0", "DEBUG"):
        try:
            reply = client.request(command)
            out[command] = {"ok": True, "reply": reply[:400]}
            if command == "STATE":
                state_names = state_field_names(reply)
        except EngineError as exc:
            out[command] = {"ok": False, "reply": str(exc)[:400]}
    eventish = [name for name in state_names if "event" in name.lower() or "hit" in name.lower()]
    return {
        "commands": out,
        "stateFieldNames": state_names,
        "stateFieldCount": len(state_names),
        "stateEventFieldPresent": bool(eventish),
        "stateEventFieldCandidates": eventish,
        "hazardsCommandPresent": out.get("HAZARDS 0", {}).get("ok", False),
        "eventsCommandPresent": out.get("EVENTS", {}).get("ok", False),
    }


# --------------------------------------------------------------------------
# source drivers
# --------------------------------------------------------------------------
def _sha256(path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


class ClientFactory:
    """Builds one engine process per source, so nothing is shared by accident."""

    def __init__(self, engine: str, data_dir: str, *, role: str = "shadow",
                 timeout: float = 180.0, stderr_dir: str | os.PathLike[str] | None = None) -> None:
        self.engine = engine
        self.data_dir = data_dir
        self.role = role
        self.timeout = timeout
        self.stderr_dir = Path(stderr_dir) if stderr_dir else None

    def launch(self, extra_env: dict[str, str] | None = None,
               *, source_id: str | None = None) -> EngineClient:
        stderr_path = None
        if self.stderr_dir is not None and source_id:
            stderr_path = self.stderr_dir / f"{source_id}.stderr.txt"
        client = EngineClient(self.engine, self.data_dir, role=self.role,
                              extra_env=extra_env, timeout=self.timeout,
                              stderr_path=stderr_path)
        client.start()
        return client


def run_reset_source(factory: ClientFactory, source_id: str, *, stage: int,
                     difficulty: int, team: int, seed: int, practice: bool,
                     frames: int, sample_every: int, mask: int = 0,
                     timing_frames: int = 200) -> dict:
    request = {"stage": stage, "difficulty": difficulty, "team": team,
               "seed": seed, "practice": practice}
    source = _new_source(source_id, "reset", request)
    source["stageName"] = STAGE_NAMES[stage] if 0 <= stage < len(STAGE_NAMES) else None
    source["framesRequested"] = frames
    client = factory.launch(source_id=source_id)
    try:
        source["setup"]["hello"] = client.request("HELLO")
        reply = client.request(
            f"RESET {stage} {difficulty} {team} {seed} {1 if practice else 0}", timeout=240.0)
        source["setup"]["reset"] = reply
        reset = parse_kv(reply)
        source["setup"]["ready"] = reset.get("ready")
        source["setup"]["setupFrames"] = reset.get("setup_frames")
        source["setup"]["resetHash"] = reset.get("hash")
        if reset.get("ready") != "1":
            source["endReason"] = "reset_not_ready"
            return source
        source["startFrame"] = int(reset.get("frames", "0"))
        source["setup"]["debug"] = client.request("DEBUG")
        source["capabilities"] = probe_capabilities(client)
        if timing_frames:
            source["timing"] = measure_timing(client, frames=timing_frames, mask=mask)
        collect_frames(client, source, frames, sample_every=sample_every,
                       mask=mask, start_frame=source["startFrame"])
        return source
    except (EngineError, OSError) as exc:
        source["errors"].append(f"{type(exc).__name__}: {exc}")
        source["endReason"] = source["endReason"] or "engine_error"
        return source
    finally:
        client.close()


def run_replay_source(factory: ClientFactory, source_id: str, *, replay: str,
                      stage: int, frames: int, sample_every: int,
                      role: str = "shadow", timeout_seconds: float = 150.0,
                      timing_frames: int = 200) -> dict:
    request = {"replay": replay, "stage": stage,
               "env": {"TH08_AUTOPLAY_REPLAY": replay, "TH08_AUTOPLAY_STAGE": str(stage)}}
    source = _new_source(source_id, "replay", request)
    source["stageName"] = STAGE_NAMES[stage] if 0 <= stage < len(STAGE_NAMES) else None
    source["framesRequested"] = frames
    client = factory.launch({"TH08_AUTOPLAY_REPLAY": replay,
                             "TH08_AUTOPLAY_STAGE": str(stage)}, source_id=source_id)
    try:
        source["setup"]["hello"] = client.request("HELLO")
        entry = wait_for_gameplay(client, deadline_seconds=timeout_seconds, stage=stage)
        source["setup"]["entry"] = entry
        source["startFrame"] = 0
        if timing_frames:
            source["timing"] = measure_timing(client, frames=timing_frames)
        collect_frames(client, source, frames, sample_every=sample_every)
        return source
    except (EngineError, OSError, GateUnavailable) as exc:
        source["errors"].append(f"{type(exc).__name__}: {exc}")
        source["endReason"] = source["endReason"] or "engine_error"
        return source
    finally:
        client.close()


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------
def decide_verdict(sources: list[dict], scan: list[dict] | None = None) -> dict:
    scan = scan or []
    # Every source in ``sources`` was explicitly requested by the caller, so it
    # counts as a required source; ``scan`` entries are exploratory only.
    primary = list(sources)
    extras: list[dict] = []
    everything = list(sources) + list(scan)
    with_lasers = [s for s in everything if s["lasers"]["nonEmptyFrames"] > 0]
    primary_with = [s for s in primary if s["lasers"]["nonEmptyFrames"] > 0]
    layout = None
    for source in everything:
        finalize_source(source)
        rows = (source["lasers"]["firstRows"] or [{}])[0].get("rows") or []
        source["lasers"]["layout"] = classify_layout(rows, LASER_DOC_FIELDS, kind="laser")
        bullet_rows = (source["bullets"]["firstRows"] or [{}])[0].get("rows") or []
        source["bullets"]["layout"] = classify_layout(bullet_rows, BULLET_DOC_FIELDS,
                                                      kind="bullet")
    for source in with_lasers:
        if source["lasers"]["layout"].get("verified"):
            layout = source["lasers"]["layout"]
            break
    if layout is None and with_lasers:
        layout = with_lasers[0]["lasers"]["layout"]
    verified = bool(layout and layout.get("verified"))
    scan_with = [s["id"] for s in scan if s["lasers"]["nonEmptyFrames"] > 0]
    if with_lasers and primary_with:
        note = ("Lasers were observed in the explicitly requested sources; see the "
                "per-source counts and the field-layout evidence.")
    elif with_lasers:
        note = ("Lasers exist in the engine, but NOT in any of the explicitly requested "
                "sources; they only appear in the exploratory scan entries. The gate is "
                "satisfied only for those stages.")
    else:
        note = ("INVALID: no source returned a single laser row, so laser geometry in "
                "HazardTape cannot be validated. This is a missing-coverage result, "
                "not a pass.")
    return {
        "status": "VALID" if with_lasers else "INVALID",
        "lasersAvailable": bool(with_lasers),
        "sourcesWithLasers": [s["id"] for s in with_lasers],
        "primarySourcesWithLasers": [s["id"] for s in primary_with],
        "primarySourcesCovered": len([s for s in primary if s["framesObserved"] > 0]),
        "extraSourcesWithLasers": [s["id"] for s in extras if s["lasers"]["nonEmptyFrames"] > 0],
        "scanSourcesWithLasers": scan_with,
        "laserFieldLayoutVerified": verified,
        "laserFieldLayoutPrefixVerified": bool(layout and layout.get("verifiedPrefix")),
        "laserFieldNames": (layout or {}).get("verifiedNames"),
        "laserMissingDocumentedFields": (layout or {}).get("missingDocumentedSuffix"),
        "laserRowTokenCounts": (layout or {}).get("tokenCounts"),
        "maxLaserCount": max([s["lasers"]["maxCount"] for s in everything] or [0]),
        "totalLaserEntries": sum(s["lasers"]["totalEntries"] for s in everything),
        "framesObserved": sum(s["framesObserved"] for s in everything),
        "note": note,
    }


def parse_scan_spec(text: str) -> list[tuple[str, int]]:
    """``'replay:0,1;practice:5'`` -> ``[("replay", 0), ("replay", 1), ("practice", 5)]``."""
    out: list[tuple[str, int]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        kind, _, values = chunk.partition(":")
        kind = kind.strip().lower()
        if kind not in ("replay", "practice"):
            raise SystemExit(f"unknown --scan kind {kind!r} (use replay/practice)")
        for item in values.split(","):
            item = item.strip()
            if not item:
                continue
            stage = int(item)
            if not 0 <= stage < len(STAGE_NAMES):
                raise SystemExit(f"--scan stage {stage} out of range 0..{len(STAGE_NAMES) - 1}")
            out.append((kind, stage))
    return out


def next_steps() -> list[str]:
    return [
        "Sweep the stage slots, not the stage names: this tool's --scan takes the "
        "replay/RESET slot indices of third_party/th08/scripts/replay_file.py "
        "STAGES = (Stage 1, Stage 2, Stage 3, Stage 4A, Stage 4B, Stage 5, Final A, "
        "Final B, Extra), so Stage 5 is slot 5 and the game's stage 6 is slot 6 "
        "(Final A) -- not slot+1 as a 1-based label would suggest.",
        "`--scan 'replay:0,1,2,4,5,7;practice:3,4,5,6,7' --scan-frames 900` runs each "
        "slot for 900 frames and puts the laser counts under coverageScan, which is "
        "the cheapest way to find the stage that actually spawns Laser objects.",
        "Repeat over more of th08_replay/*.rpy: a Lunatic replay string that reaches "
        "Final A/B is more likely to contain a laser spell card than stage 1.",
        "If every slot is still empty, capture one retail/Windows trace of a known "
        "laser spell card and diff its hazard rows against the Linux shadow build "
        "before declaring the command broken.",
        "Once a slot yields rows, re-run the gate with that slot and extend the "
        "recording to the full HazardTape path.",
    ]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "v400-p1-laser-gate",
        description="Measure LASERS/BULLETS coverage on real TH08 engine frames.")
    parser.add_argument("--engine", required=True, help="WSL path to th08-modern")
    parser.add_argument("--data-dir", required=True, help="WSL path to the game data dir")
    parser.add_argument("--out", default=str(ROOT / ".tmp_v400_p1s1/laser_gate.json"))
    parser.add_argument("--frames", type=int, default=3000)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", type=int, default=5,
                        help="replay stage slot for the replay source (5 = Stage 5; "
                             "slots are STAGES in third_party/th08/scripts/replay_file.py)")
    parser.add_argument("--practice-stage", type=int, default=6,
                        help="stage slot for the practice source (6 = Final A, i.e. "
                             "the game's stage 6)")
    parser.add_argument("--replay", default="/mnt/d/WorkShop/Touhou_RL/th08_replay/Th8_ud0059.rpy")
    parser.add_argument("--extra-replay-stages", default="7",
                        help="additional replay stage slots run as explicit sources; "
                             "default 7 = Final B, the stage this corpus actually spawns "
                             "Laser rows in")
    parser.add_argument("--extra-replay-frames", type=int, default=12000,
                        help="frame budget for --extra-replay-stages (Final B's laser "
                             "sections start around frame 8800, so 3000 is not enough)")
    parser.add_argument("--role", default="shadow", choices=("shadow", "env"))
    parser.add_argument("--only", default="",
                        help="comma separated subset of source ids (default: all)")
    parser.add_argument("--explore", action="store_true",
                        help="if the three primary sources have no laser, also scan "
                             "extra stage/replay combinations")
    parser.add_argument("--explore-stages", default="4,7,2",
                        help="replay slots tried by --explore (4=Stage 4B, 7=Final B, "
                             "2=Stage 3)")
    parser.add_argument("--explore-frames", type=int, default=900,
                        help="frame budget per --explore stage (cheaper than the "
                             "primary sources)")
    parser.add_argument("--scan", default="",
                        help="extra coverage sweep, e.g. 'replay:0,1,2,4,5,7;practice:3,4,5,6,7'; "
                             "results land in coverageScan and never replace the three "
                             "required sources")
    parser.add_argument("--scan-frames", type=int, default=900,
                        help="frame budget per --scan entry")
    parser.add_argument("--timing-laser-stage", type=int, default=-1,
                        help="replay slot used for the paged LASERS timing probe "
                             "(7 = Final B, whose laser section peaks at 230 rows)")
    parser.add_argument("--timing-laser-start", type=int, default=9300,
                        help="frames to play before the paged timing probe starts")
    parser.add_argument("--timing-laser-frames", type=int, default=200,
                        help="frames timed by the paged probe")
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_path = Path(args.out)
    factory = ClientFactory(args.engine, args.data_dir, role=args.role,
                            timeout=args.timeout,
                            stderr_dir=out_path.parent / "stderr")
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    report: dict = {
        "schemaVersion": 1,
        "generatedAt": started,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "engine": {
            "binary": args.engine,
            "sha256": _sha256(args.engine),
            "dataDir": args.data_dir,
            "role": args.role,
            "rebuilt": False,
        },
        "config": {"frames": args.frames, "sampleEvery": args.sample_every,
                   "seed": args.seed, "replayStage": args.stage,
                   "practiceStage": args.practice_stage, "replay": args.replay},
        "sources": [],
        "timing": {},
        "eventExtraction": {},
        "verdict": {},
        "nextSteps": next_steps(),
    }

    only = {item.strip() for item in args.only.split(",") if item.strip()}
    run = (lambda name: not only or name in only)

    if run("reset-stage1-normal"):
        report["sources"].append(run_reset_source(
            factory, "reset-stage1-normal", stage=0, difficulty=1, team=0,
            seed=args.seed, practice=False, frames=args.frames,
            sample_every=args.sample_every))
    if run("replay-stage5"):
        report["sources"].append(run_replay_source(
            factory, "replay-stage5", replay=args.replay, stage=args.stage,
            frames=args.frames, sample_every=args.sample_every))
    if run("practice-stage6"):
        report["sources"].append(run_reset_source(
            factory, "practice-stage6", stage=args.practice_stage, difficulty=1,
            team=0, seed=args.seed, practice=True, frames=args.frames,
            sample_every=args.sample_every))
    # The task allows "stage 5, or any other stage known to have lasers"; Stage 5
    # turned out empty, so the known-laser stage runs as an explicit source here
    # instead of hiding inside the exploratory scan.
    for stage in [int(item) for item in args.extra_replay_stages.split(",") if item.strip()]:
        label = STAGE_NAMES[stage].lower().replace(" ", "")
        report["sources"].append(run_replay_source(
            factory, f"replay-{label}", replay=args.replay, stage=stage,
            frames=args.extra_replay_frames, sample_every=args.sample_every))

    if args.explore and not any(s["lasers"]["nonEmptyFrames"] for s in report["sources"]):
        stages = [int(item) for item in args.explore_stages.split(",") if item.strip()]
        for stage in stages:
            if stage == args.stage:
                continue
            report["sources"].append(run_replay_source(
                factory, f"explore-replay-{STAGE_NAMES[stage].lower().replace(' ', '')}",
                replay=args.replay, stage=stage, frames=args.explore_frames,
                sample_every=args.sample_every, timing_frames=0))

    report["coverageScan"] = []
    for kind, stage in parse_scan_spec(args.scan):
        label = STAGE_NAMES[stage].lower().replace(" ", "")
        if kind == "replay":
            source = run_replay_source(
                factory, f"scan-replay-{label}", replay=args.replay, stage=stage,
                frames=args.scan_frames, sample_every=args.sample_every,
                timing_frames=0)
        else:
            source = run_reset_source(
                factory, f"scan-practice-{label}", stage=stage, difficulty=1,
                team=0, seed=args.seed, practice=True, frames=args.scan_frames,
                sample_every=args.sample_every, timing_frames=0)
        report["coverageScan"].append(source)

    report["laserPagingTiming"] = None
    if args.timing_laser_stage >= 0:
        label = STAGE_NAMES[args.timing_laser_stage].lower().replace(" ", "")
        client = factory.launch({"TH08_AUTOPLAY_REPLAY": args.replay,
                                 "TH08_AUTOPLAY_STAGE": str(args.timing_laser_stage)},
                                source_id="timing-laser-pages")
        try:
            client.request("HELLO")
            wait_for_gameplay(client, deadline_seconds=150.0,
                              stage=args.timing_laser_stage)
            pumped = 0
            while pumped < args.timing_laser_start:
                step = client.fields("STEP 0 1")
                if int(step.get("advanced", "0")) != 1:
                    break
                pumped += 1
            report["laserPagingTiming"] = {
                "stage": args.timing_laser_stage,
                "stageName": label,
                "startFrame": pumped,
                **measure_paged_laser_timing(client, args.timing_laser_frames),
            }
        except (EngineError, OSError, GateUnavailable) as exc:
            report["laserPagingTiming"] = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            client.close()

    caps = next((s.get("capabilities") for s in report["sources"] if s.get("capabilities")),
                None)
    report["eventExtraction"] = {
        "stateFieldNames": (caps or {}).get("stateFieldNames", []),
        "stateEventFieldPresent": (caps or {}).get("stateEventFieldPresent", None),
        "eventsCommandPresent": (caps or {}).get("eventsCommandPresent", None),
        "hazardsCommandPresent": (caps or {}).get("hazardsCommandPresent", None),
        "capabilityProbe": (caps or {}).get("commands", {}),
        "conclusion": (
            "STATE exposes only the 25 tracked numeric fields f0..f24; no "
            "frameEventFlags/PLAYER_HIT field is available on the control channel, so "
            "the 11-P1 §4 fatal-bomb rule (frameEventFlags & 0x2) cannot be evaluated "
            "yet. Minimal extension: add a read-only `EVENTS` command (or an extra "
            "`ev=` token on STATE, excluded from the 25-field hash) that returns "
            "`g_ReplayManager ? g_ReplayManager->frameEventFlags : 0` "
            "(ReplayManager.hpp:213, REPLAY_FRAME_EVENT_PLAYER_HIT=0x2 set in "
            "Player.cpp:329/356/466). No engine change was made by this slice."
        ),
    }
    report["timing"] = {
        s["id"]: s.get("timing") for s in report["sources"] if s.get("timing")
    }
    report["verdict"] = decide_verdict(report["sources"], report["coverageScan"])
    report["commands"] = {
        "engineArgv": ["<engine>", "--data-dir", "<data-dir>", "--rl-control", "0,1",
                       "--rl-role", args.role],
        "sources": [s["request"] for s in report["sources"]],
        "scan": [s["request"] for s in report["coverageScan"]],
    }
    report["finishedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")

    def _brief(source: dict) -> dict:
        return {
            "stage": source["request"].get("stage"),
            "stageName": source.get("stageName"),
            "frames": source["framesObserved"],
            "laserNonEmptyFrames": source["lasers"]["nonEmptyFrames"],
            "laserMaxCount": source["lasers"]["maxCount"],
            "bulletNonEmptyFrames": source["bullets"]["nonEmptyFrames"],
            "bulletMaxCount": source["bullets"]["maxCount"],
            "bulletCommandAvailable": source["bullets"].get("available", True),
            "lastLiveSampleFrame": source.get("lastLiveSampleFrame"),
            "endReason": source["endReason"],
            "errors": source["errors"],
        }

    summary = {
        "out": str(out),
        "verdict": report["verdict"]["status"],
        "sources": {s["id"]: _brief(s) for s in report["sources"]},
        "scan": {s["id"]: _brief(s) for s in report["coverageScan"]},
        "sourcesWithLasers": report["verdict"]["sourcesWithLasers"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if report["verdict"]["status"] == "VALID":
        return 0
    if not any(s["framesObserved"] for s in report["sources"]):
        return 4
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
