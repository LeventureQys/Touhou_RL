"""Thin, timable wrapper around one ``th08-modern --rl-role shadow`` process.

Why a wrapper rather than :class:`touhou_rl.v400.gym.protocol.HarnessClient`:
the teacher needs three commands the gym client does not expose (``PARK``,
``SPAWN``, ``CHILD``), a capability probe that is recorded rather than assumed,
and per-call timing so the three phases of one decision can be reported
separately (11-P1 §5).

Capability facts this wrapper measures instead of guessing (EXP-P1-003):
``BRANCH`` replies with hashes only — it cannot carry hazard geometry — and
``CHILD <handle> HAZARDS`` is refused.  Both are reported by
:meth:`TeacherEngine.probe_capabilities`.
"""

from __future__ import annotations

import hashlib
import os
import select
import subprocess
import time
from dataclasses import dataclass, field

from touhou_rl.v400.gym.protocol import (
    EventFrame,
    HazardFrame,
    PlayerSnapshot,
    parse_enemy_row,
    parse_events_reply,
    parse_hazards_reply,
    parse_player_reply,
)

#: The engine build the teacher is written against: ``build/rl_hazard2``
#: (``CHILD <h> HAZARDS/PLAYER/EVENTS`` and ``EVENTS`` exist, EXP-P1-007 §2).
EXPECTED_ENGINE_SHA256 = "f172399f61b75baa8e62be46041ef756a710759f439e7fb07111f4747c78264e"

#: The binary name inside a build directory (``build/rl_hazard2/th08-modern``).
ENGINE_BINARY_NAME = "th08-modern"

#: ``EnemyManager.hpp:69-71``: the two flags ``ENEMIES`` needs.  ``ACTIVE``
#: decides whether the row is listed at all; ``BOSS`` is the only boss signal
#: any command exports, and ``STATE``'s 25 fields do not carry it (EXP-P1-009 §6
#: item 2 — "Boss 不可观测" was a client gap, not an engine one).
ENEMY_FLAG_ACTIVE = 1 << 0
ENEMY_FLAG_BOSS = 1 << 1

#: ``HandleEnemies`` caps one reply at 16 rows; the caller pages by ``offset``.
ENEMIES_PAGE_LIMIT = 16


class EngineError(RuntimeError):
    """The engine refused a command or the control pipe died."""


class EngineShaMismatch(RuntimeError):
    """The binary is missing or is not the build this code was measured against.

    A different binary can still speak the protocol, so without this gate the
    teacher would happily record with a build whose ``CHILD HAZARDS`` semantics
    or ``EVENTS`` bits were never verified — a silent correctness change.
    """


def sha256_file(path: str | os.PathLike[str], *, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in bounded chunks."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def check_engine_binary(
    binary: str | os.PathLike[str], expected: str | None = EXPECTED_ENGINE_SHA256
) -> dict[str, object]:
    """Hash the engine binary and refuse anything but the verified build.

    ``expected=None`` records the hash without enforcing it (used only by the
    legacy comparison, which must run the *same* engine as the child path).
    """

    path = os.fspath(binary)
    if not os.path.isfile(path):
        raise EngineShaMismatch(f"engine binary not found: {path}")
    actual = sha256_file(path)
    if expected and actual.lower() != str(expected).lower():
        raise EngineShaMismatch(
            f"engine SHA mismatch for {path}: sha256={actual} expected={expected}"
        )
    return {
        "path": os.path.abspath(path),
        "sha256": actual,
        "expected": expected,
        "match": bool(expected) and actual.lower() == str(expected).lower(),
    }


@dataclass
class CallStats:
    """Per-command accounting, so the report can price each phase."""

    calls: int = 0
    seconds: float = 0.0
    per_command: dict[str, list[float]] = field(default_factory=dict)

    def add(self, verb: str, seconds: float) -> None:
        self.calls += 1
        self.seconds += seconds
        self.per_command.setdefault(verb, []).append(seconds)


def parse_fields(reply: str) -> dict[str, str]:
    parts = reply.split()
    out: dict[str, str] = {"_verb": parts[0] if parts else ""}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            out[key] = value
    return out


def _bits_float(token: str) -> float:
    import struct

    return struct.unpack("<f", struct.pack("<I", int(token, 16) & 0xFFFFFFFF))[0]


def parse_enemies_reply(reply: str) -> dict[str, object]:
    """One ``ENEMIES`` page, decoded — the only boss/life observation there is.

    ``HandleEnemies`` (``rlharness.cpp:387-421``) emits
    ``slot:flags1:posX:posY:posZ:life:timer:worldX:worldY:worldZ:hitboxW:hitboxH``
    per row and, in the header, the authoritative ``count=`` of active slots.
    ``flags1`` and ``life`` are hex; ``life`` is an integer HP (not float bits).

    A row is *listed* only when ``flags1 & ENEMY_FLAG_ACTIVE`` is set, so the row
    list is already "live enemies"; the boss verdict is ``flags1 & BOSS``.  No
    field is invented: an engine that answers ``ERR`` yields ``available=False``
    with the engine's own text.
    """

    fields = parse_fields(reply)
    if fields.get("_verb") != "OK":
        return {
            "available": False,
            "reason": reply[:160],
            "frame": None,
            "count": None,
            "rows": (),
            "boss_count": 0,
            "boss_life": None,
            "life_sum": None,
        }
    rows = tuple(
        parse_enemy_row(row) for row in fields.get("enemies", "").split(",") if row
    )
    bosses = tuple(row for row in rows if row.flags1 & ENEMY_FLAG_BOSS)
    return {
        "available": True,
        "reason": "ENEMIES accepted",
        "frame": int(fields.get("frames", 0)),
        "count": int(fields["count"]) if "count" in fields else len(rows),
        "offset": int(fields.get("offset", 0)),
        "limit": int(fields.get("limit", 0)),
        "rows": rows,
        "boss_count": len(bosses),
        "boss_life": max((row.life for row in bosses), default=0),
        "boss_slots": tuple(row.slot for row in bosses),
        "life_sum": int(sum(row.life for row in rows)),
    }


def decode_state_fields(fields: dict[str, str]) -> dict[str, object]:
    """The six ``STATE`` numbers a plan rollout needs, from parsed fields.

    ``STATE`` carries ``f0..f24`` plus ``frames``; ``f6`` is the death counter
    and ``f10``/``f11`` the ship position.  A missing field stays ``None``
    instead of defaulting to 0 — "no counter" and "counter is zero" are
    different claims.
    """
    return {
        "frame": int(fields.get("frames", 0)),
        "deaths": _bits_float(fields["f6"]) if "f6" in fields else None,
        "lives": _bits_float(fields["f7"]) if "f7" in fields else None,
        "bombs": _bits_float(fields["f8"]) if "f8" in fields else None,
        "px": _bits_float(fields["f10"]) if "f10" in fields else None,
        "py": _bits_float(fields["f11"]) if "f11" in fields else None,
        "raw": fields,
    }


class TeacherEngine:
    """One shadow process: the authoritative state plus its forks."""

    def __init__(
        self,
        binary: str | os.PathLike[str],
        data_dir: str | os.PathLike[str],
        *,
        role: str = "shadow",
        timeout: float = 180.0,
    ) -> None:
        self.binary = str(binary)
        self.data_dir = str(data_dir)
        self.role = role
        self.timeout = timeout
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("DISPLAY", "WAYLAND_DISPLAY")
        }
        env.update(SDL_AUDIODRIVER="dummy", LIBGL_ALWAYS_SOFTWARE="1", LP_NUM_THREADS="1")
        argv = [
            self.binary,
            "--data-dir",
            self.data_dir,
            "--rl-control",
            "0,1",
            "--rl-role",
            role,
        ]
        self.argv = argv
        self._env = env
        self._proc: subprocess.Popen[bytes] | None = None
        self._buffer = b""
        self.stats = CallStats()
        self.log: list[str] = []

    # -- lifetime ---------------------------------------------------------
    def start(self) -> "TeacherEngine":
        if self._proc is not None:
            raise EngineError("already started")
        if not os.path.isfile(self.binary):
            raise EngineError(f"harness binary not found: {self.binary}")
        if not os.path.isdir(self.data_dir):
            raise EngineError(f"game data dir not found: {self.data_dir}")
        self._proc = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
            bufsize=0,
        )
        return self

    def close(self, *, quit_timeout: float = 20.0) -> int:
        proc = self._proc
        if proc is None:
            return -1
        code = -1
        try:
            if proc.poll() is None:
                try:
                    self.ask("QUIT", timeout=quit_timeout)
                except (EngineError, OSError):
                    pass
            try:
                code = proc.wait(timeout=quit_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait(timeout=5)
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
            self._proc = None
        return code

    def __enter__(self) -> "TeacherEngine":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- transport --------------------------------------------------------
    def ask(self, command: str, timeout: float | None = None) -> str:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise EngineError("engine is not running")
        limit = self.timeout if timeout is None else timeout
        started = time.monotonic()
        try:
            proc.stdin.write((command + "\n").encode("ascii"))
            proc.stdin.flush()
        except OSError as error:
            raise EngineError(f"write failed on {command!r}: {error}") from error
        reply = self._readline(command, limit)
        self.stats.add(command.split(" ", 1)[0], time.monotonic() - started)
        self.log.append(reply[:120])
        if reply.startswith("ERR"):
            raise EngineError(f"{command!r} -> {reply[:200]}")
        return reply

    def ask_many(self, commands, timeout: float | None = None) -> list[str]:
        """Send every command, then read every reply — in order, one per command.

        The harness reads its control channel line by line and answers each one
        before looking at the next, so pipelining cannot reorder anything; it
        only removes the caller-side wait between a reply and the next write.
        Verification is 144 such commands per decision and recording 145, which
        is 0.1-0.15 ms of pure latency each when sent one at a time.
        """
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise EngineError("engine is not running")
        limit = self.timeout if timeout is None else timeout
        commands = list(commands)
        if not commands:
            return []
        try:
            proc.stdin.write("".join(command + "\n" for command in commands).encode("ascii"))
            proc.stdin.flush()
        except OSError as error:
            raise EngineError(f"batched write failed: {error}") from error
        replies: list[str] = []
        for command in commands:
            started = time.monotonic()
            reply = self._readline(command, limit)
            self.stats.add(command.split(" ", 1)[0], time.monotonic() - started)
            self.log.append(reply[:120])
            if reply.startswith("ERR"):
                raise EngineError(f"{command!r} -> {reply[:200]}")
            replies.append(reply)
        return replies


    def _readline(self, command: str, timeout: float) -> str:
        assert self._proc is not None and self._proc.stdout is not None
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buffer:
            left = deadline - time.monotonic()
            if left <= 0:
                raise EngineError(f"no reply within {timeout:.1f}s for {command!r}")
            ready, _, _ = select.select([self._proc.stdout], [], [], min(left, 0.5))
            if ready:
                chunk = os.read(self._proc.stdout.fileno(), 1 << 20)
                if not chunk:
                    raise EngineError(
                        f"engine closed the control pipe during {command!r} "
                        f"(exit={self._proc.poll()})"
                    )
                self._buffer += chunk
            elif self._proc.poll() is not None:
                raise EngineError(
                    f"engine exited ({self._proc.returncode}) during {command!r}"
                )
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line.decode("utf-8", "replace").strip()

    # -- typed commands ---------------------------------------------------
    def hello(self) -> dict[str, str]:
        return parse_fields(self.ask("HELLO"))

    def debug(self) -> dict[str, str]:
        return parse_fields(self.ask("DEBUG"))

    def frame(self) -> int:
        return int(parse_fields(self.ask("FRAME"))["frame"])

    def hash(self) -> str:
        return parse_fields(self.ask("HASH"))["hash"]

    def reset(self, *, stage: int, difficulty: int, team: int, seed: int, practice: bool = False):
        return parse_fields(
            self.ask(f"RESET {stage} {difficulty} {team} {seed} {1 if practice else 0}")
        )

    def step(self, mask: int, frames: int = 1) -> dict[str, str]:
        if not 0 <= mask <= 0xFFFF:
            raise EngineError(f"mask out of range: {mask}")
        if frames <= 0:
            raise EngineError(f"frames must be positive: {frames}")
        return parse_fields(self.ask(f"STEP {mask} {frames}"))

    def hazards(self) -> HazardFrame:
        return parse_hazards_reply(self.ask("HAZARDS"))

    def player(self) -> PlayerSnapshot:
        return parse_player_reply(self.ask("PLAYER"))

    def state(self) -> dict[str, object]:
        """The 25 tracked fields, plus whether an event-flags field exists.

        ``STATE`` is the only per-frame counter export.  It carries ``f0..f24``
        and nothing else, so ``frameEventFlags`` (10-P0 §5, 11-P1 §4 rule 2) is
        **not** available; callers must not invent it.
        """
        fields = parse_fields(self.ask("STATE"))
        float_keys = {f"f{i}" for i in (6, 7, 8, 9, 10, 11, 12)}
        values = {
            key: int(value, 16) if key in float_keys else int(value)
            for key, value in fields.items()
            if key.startswith("f")
        }
        return {
            "frame": int(fields.get("frames", 0)),
            "raw": fields,
            "fields": values,
            "deaths": _bits_float(fields["f6"]) if "f6" in fields else None,
            "lives": _bits_float(fields["f7"]) if "f7" in fields else None,
            "bombs": _bits_float(fields["f8"]) if "f8" in fields else None,
            "px": _bits_float(fields["f10"]) if "f10" in fields else None,
            "py": _bits_float(fields["f11"]) if "f11" in fields else None,
            "has_frame_event_flags": any(
                key in fields for key in ("frameEventFlags", "eventFlags", "f25")
            ),
            "float_keys": sorted(float_keys),
        }

    # -- enemy / boss observation (EXP-P1-011 §2) --------------------------
    def enemies(self, *, limit: int = ENEMIES_PAGE_LIMIT, max_pages: int = 4):
        """Every active enemy row of the current frame, plus the boss verdict.

        ``STATE``'s 25 fields contain no boss and no enemy HP, so before this the
        CLI could not tell "no boss yet" from "boss not implemented".  ``ENEMIES``
        has both; it is read-only (``fork.cpp:345``) so observing it cannot move
        the shadow.

        The reply caps one page at 16 rows, so a crowded wave is paged until
        ``count`` is covered (``max_pages`` bounds the cost; a truncated answer
        says so via ``truncated`` instead of pretending the page is the frame).
        """

        limit = max(1, min(int(limit), ENEMIES_PAGE_LIMIT))
        pages = max(1, int(max_pages))
        first = parse_enemies_reply(self.ask(f"ENEMIES 0 {limit}"))
        if not first.get("available"):
            return first
        rows = list(first["rows"])
        total = int(first["count"])
        page_count = 1
        offset = limit
        while len(rows) < total and page_count < pages:
            page = parse_enemies_reply(self.ask(f"ENEMIES {offset} {limit}"))
            if not page.get("available"):
                break
            rows.extend(page["rows"])
            page_count += 1
            offset += limit
            if not page["rows"]:
                break
        bosses = tuple(row for row in rows if row.flags1 & ENEMY_FLAG_BOSS)
        return {
            "available": True,
            "reason": first["reason"],
            "frame": first["frame"],
            "count": total,
            "offset": 0,
            "limit": limit,
            "pages": page_count,
            "rows": tuple(rows),
            "truncated": len(rows) < total,
            "boss_count": len(bosses),
            "boss_life": max((row.life for row in bosses), default=0),
            "boss_slots": tuple(row.slot for row in bosses),
            "life_sum": int(sum(row.life for row in rows)),
        }

    def enemies_snapshot(self) -> dict[str, object]:
        """``enemies()`` as a recordable verdict instead of an exception.

        Same contract as :meth:`events_snapshot`: an engine that refuses
        ``ENEMIES`` keeps the refusal verbatim, so a log can say *why* the boss
        could not be observed rather than showing ``boss=0``.
        """

        try:
            return self.enemies()
        except EngineError as error:
            return {
                "available": False,
                "reason": str(error)[:200],
                "frame": None,
                "count": None,
                "rows": (),
                "truncated": None,
                "boss_count": 0,
                "boss_life": None,
                "boss_slots": (),
                "life_sum": None,
            }

    # -- replay frame events (11-P1 §4 rule 2) -----------------------------
    def events(self) -> EventFrame:
        """The last advanced frame's replay bitmask (``EVENTS``).

        ``ERR events-unavailable`` (no live ``ReplayManager``) and a pre-4.0.0
        ``ERR unknown-command`` both raise :class:`EngineError`; the caller must
        report the word as unavailable rather than inventing ``0``, which would
        read as "no player hit".
        """

        return parse_events_reply(self.ask("EVENTS"))

    def events_snapshot(self) -> dict[str, object]:
        """``EVENTS`` as a recordable verdict instead of an exception.

        ``available=False`` keeps the engine's own refusal text in ``reason`` so a
        label can say *why* the death bomb could not be judged.
        """

        try:
            event = self.events()
        except EngineError as error:
            return {
                "available": False,
                "reason": str(error)[:200],
                "frame": None,
                "frame_event_flags": None,
                "names": [],
                "player_hit": False,
            }
        return {
            "available": True,
            "reason": "EVENTS accepted",
            "frame": int(event.frame),
            "frame_event_flags": int(event.frame_event_flags),
            "names": list(event.names),
            "player_hit": bool(event.player_hit),
        }

    def child_events(self, handle: int) -> EventFrame:
        """``CHILD <handle> EVENTS`` — the fork's own last-frame event flags."""

        return parse_events_reply(self.child(handle, "EVENTS"))

    # -- fork mechanism ---------------------------------------------------
    def park(self) -> int:
        fields = parse_fields(self.ask("PARK"))
        return int(fields["template"])

    def spawn(self, template: int) -> int:
        fields = parse_fields(self.ask(f"SPAWN {template}"))
        if fields.get("spawned") != "1":
            raise EngineError(f"SPAWN did not report spawned=1: {fields}")
        return int(fields["handle"])

    def child(self, handle: int, command: str, timeout: float | None = None) -> str:
        return self.ask(f"CHILD {handle} {command}", timeout=timeout)

    def child_state(self, handle: int) -> dict[str, object]:
        return self._decode_state_fields(parse_fields(self.child(handle, "STATE")))

    def child_step(self, handle: int, mask: int, frames: int = 1) -> dict[str, str]:
        return parse_fields(self.child(handle, f"STEP {mask} {frames}"))

    def child_quit(self, handle: int) -> None:
        try:
            self.child(handle, "QUIT")
        except EngineError:
            pass

    def _decode_state_fields(self, fields: dict[str, str]) -> dict[str, object]:
        return decode_state_fields(fields)

    # -- capability probe -------------------------------------------------
    def probe_child_hazards(self, *, refresh: bool = False) -> dict[str, object]:
        """Can a forked child answer ``HAZARDS``?  Measured, never assumed.

        A real ``PARK`` + ``SPAWN`` is taken and a real ``CHILD <h> HAZARDS``
        is sent; the engine's own refusal text is returned verbatim as
        ``reason``.  The answer is cached per process because the probe costs a
        fork pair (~1.5 ms) and the capability of a build cannot change while
        the process lives.  ``available=False`` is the honest current answer for
        ``build/rl_hazard``; callers must fail closed on it instead of falling
        back to a main-process recording that would move the shadow.
        """
        cached = getattr(self, "_child_hazards", None)
        if cached is not None and not refresh:
            return dict(cached)
        verdict: dict[str, object]
        if self._proc is None:
            verdict = {"available": False, "reason": "engine is not running"}
            self._child_hazards = verdict
            return dict(verdict)
        try:
            template = self.park()
        except EngineError as error:
            verdict = {"available": False, "reason": f"PARK refused: {error}"[:200]}
            self._child_hazards = verdict
            return dict(verdict)
        handle = None
        try:
            handle = self.spawn(template)
        except EngineError as error:
            verdict = {"available": False, "reason": f"SPAWN refused: {error}"[:200]}
            self._child_hazards = verdict
            return dict(verdict)
        try:
            reply = self.child(handle, "HAZARDS 16")
            verdict = {
                "available": True,
                "reason": "CHILD <h> HAZARDS accepted",
                "reply": reply[:160],
            }
        except EngineError as error:
            verdict = {"available": False, "reason": str(error)[:200]}
        finally:
            self.child_quit(handle)
        self._child_hazards = verdict
        return dict(verdict)

    def probe_capabilities(self) -> dict[str, object]:
        """Ask the engine what it can actually do, and record the answers.

        Everything here is evidence for EXP-P1-003: the reply text of each
        probe is kept verbatim so a reader can check the conclusion.
        """
        out: dict[str, object] = {}
        hello = parse_fields(self.ask("HELLO"))
        out["role"] = hello.get("role")
        out["protocol"] = hello.get("proto")
        try:
            branch = parse_fields(self.ask("BRANCH 0 1"))
            out["branch_fields"] = sorted(branch.keys())
            out["branch_carries_hazards"] = any(
                key in branch for key in ("bullets", "lasers", "enemies", "nb", "nl", "ne")
            )
        except EngineError as error:
            # BRANCH is only Ready() after a stage is running; record the engine's
            # own words instead of hiding the branch behind a guess.
            out["branch_fields"] = None
            out["branch_error"] = str(error)[:160]
            out["branch_carries_hazards"] = False
        state = parse_fields(self.ask("STATE"))
        out["state_fields"] = sorted(state.keys())
        out["state_has_frame_event_flags"] = any(
            key in state for key in ("frameEventFlags", "eventFlags", "f25")
        )
        hazards = parse_fields(self.ask("HAZARDS 1024"))
        out["hazards_limit_1024_ok"] = hazards.get("_verb") == "OK"
        # A fork cannot answer HAZARDS: prove it on a real handle, then drop it.
        template = self.park()
        handle = self.spawn(template)
        try:
            try:
                self.child(handle, "HAZARDS")
                out["child_hazards"] = "OK"
            except EngineError as error:
                out["child_hazards"] = str(error)[:160]
            out["child_state_ok"] = self.child_state(handle).get("frame") is not None
        finally:
            self.child_quit(handle)
        return out
