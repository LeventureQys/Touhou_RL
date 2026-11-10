"""Stable contract tests for the v400 line protocol transport."""

from __future__ import annotations

import subprocess

import pytest

from touhou_rl.v400.gym import protocol


class _Pipe:
    def __init__(self, owner=None, payload=b""):
        self.owner = owner
        self.payload = bytearray(payload)
        self.closed = False
        self.writes: list[bytes] = []

    def fileno(self):
        return id(self)

    def write(self, data):
        self.writes.append(data)
        return len(data)

    def flush(self):
        return None

    def close(self):
        self.closed = True

    def read(self, size=-1):
        if not self.payload:
            return b""
        data = bytes(self.payload[:size])
        del self.payload[:size]
        return data


class _Proc:
    def __init__(self, payload=b"", *, poll_value=None, wait_raises=False):
        self.stdin = _Pipe(self)
        self.stdout = _Pipe(self, payload)
        self.stderr = _Pipe(self)
        self.returncode = poll_value
        self._poll_value = poll_value
        self.wait_raises = wait_raises
        self.wait_calls: list[float] = []
        self.kill_calls = 0

    def poll(self):
        return self._poll_value

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.wait_raises and len(self.wait_calls) == 1:
            raise subprocess.TimeoutExpired("fake", timeout)
        self._poll_value = 0
        self.returncode = 0
        return 0

    def kill(self):
        self.kill_calls += 1
        self._poll_value = -9
        self.returncode = -9


def _client(tmp_path, proc):
    work = tmp_path / f"client-{id(proc)}"
    work.mkdir()
    binary = work / "harness"
    data = work / "data"
    binary.write_bytes(b"")
    data.mkdir()
    client = protocol.HarnessClient(binary, data, timeout=1.0)
    client._proc = proc
    return client


def _ready_select(monkeypatch):
    monkeypatch.setattr(protocol.select, "select", lambda r, w, x, timeout: (r, [], []))
    monkeypatch.setattr(protocol.os, "read", lambda fd, size: _FD_PAYLOADS[fd].read(size))


_FD_PAYLOADS = {}


def _install_read(monkeypatch, proc):
    _FD_PAYLOADS[proc.stdout.fileno()] = proc.stdout
    _ready_select(monkeypatch)


def test_request_replies_and_updates_stats(tmp_path, monkeypatch):
    proc = _Proc(b"OK value=7\n")
    client = _client(tmp_path, proc)
    _install_read(monkeypatch, proc)

    assert client.request("PING") == "OK value=7"
    assert proc.stdin.writes == [b"PING\n"]
    assert client.stats.commands == 1
    assert client.stats.replies == {"OK": 1}
    assert client.stats.wall_seconds >= 0


def test_err_reply_raises_after_accounting_reply(tmp_path, monkeypatch):
    proc = _Proc(b"ERR bad thing\n")
    client = _client(tmp_path, proc)
    _install_read(monkeypatch, proc)

    with pytest.raises(protocol.HarnessError, match=r"'PING' -> ERR bad thing"):
        client.request("PING")
    assert client.stats.commands == 1
    assert client.stats.replies == {"ERR": 1}


def test_readline_reports_eof_and_early_exit(tmp_path, monkeypatch):
    eof = _Proc(b"")
    client = _client(tmp_path, eof)
    _install_read(monkeypatch, eof)
    with pytest.raises(protocol.HarnessError, match="closed the control pipe"):
        client.request("PING")

    exited = _Proc(b"", poll_value=23)
    client = _client(tmp_path, exited)
    monkeypatch.setattr(protocol.select, "select", lambda r, w, x, timeout: ([], [], []))
    with pytest.raises(protocol.HarnessError, match=r"exited before replying \(code 23\)"):
        client.request("PING")


def test_readline_timeout_is_reported(tmp_path, monkeypatch):
    proc = _Proc()
    client = _client(tmp_path, proc)
    clock = iter((10.0, 10.0, 10.6))
    monkeypatch.setattr(protocol.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(protocol.select, "select", lambda r, w, x, timeout: ([], [], []))

    with pytest.raises(protocol.HarnessError, match=r"no reply within 0.5s"):
        client.request("PING", timeout=0.5)


def test_close_sends_quit_waits_and_closes_streams(tmp_path, monkeypatch):
    proc = _Proc(b"BYE\n")
    client = _client(tmp_path, proc)
    _install_read(monkeypatch, proc)

    client.close(quit_timeout=0.25)
    assert proc.stdin.writes == [b"QUIT\n"]
    assert proc.wait_calls == [0.25]
    assert proc.kill_calls == 0
    assert all(stream.closed for stream in (proc.stdin, proc.stdout, proc.stderr))
    assert client._proc is None


def test_close_kills_when_wait_times_out_and_still_closes_streams(tmp_path, monkeypatch):
    proc = _Proc(b"", wait_raises=True)
    client = _client(tmp_path, proc)
    _install_read(monkeypatch, proc)

    client.close(quit_timeout=0.125)
    assert proc.stdin.writes == [b"QUIT\n"]
    assert proc.wait_calls == [0.125, 5]
    assert proc.kill_calls == 1
    assert all(stream.closed for stream in (proc.stdin, proc.stdout, proc.stderr))


def test_parse_blank_fields_and_duplicate_keys_follow_existing_behavior():
    assert protocol._parse("") == {"_verb": ""}
    assert protocol._parse("  ") == {"_verb": ""}
    assert protocol._parse("OK bare a=1 ignored=two a=3") == {
        "_verb": "OK",
        "a": "3",
        "ignored": "two",
    }
    assert protocol._parse("ERR reason") == {"_verb": "ERR"}


# ---------------------------------------------------------------------------
# HUD (EXP-P2-005): displayScore is what the picture contains
# ---------------------------------------------------------------------------
HUD_REPLY = (
    "OK frames=1234 displayScore=4100 globalsScore=4200 lives=1 bombs=2 power=64 "
    "graze=37 point=12 time=3 gameOverFlag=0 displayAge=100 displayStale=7 "
    "scoreDisplayStep=3"
)


def test_hud_reply_parses_every_field():
    hud = protocol.parse_hud_reply(HUD_REPLY)
    assert hud.frame == 1234
    assert hud.display_score == 4100
    assert hud.globals_score == 4200
    assert hud.lives == 1
    assert hud.bombs == 2
    assert hud.power == 64
    assert hud.graze == 37
    assert hud.point == 12
    assert hud.time == 3
    assert hud.game_over == 0
    assert hud.display_age == 100
    assert hud.display_stale == 7
    assert hud.score_display_step == 3
    assert hud.settled is False


def test_hud_reply_marks_settled_counter():
    settled = protocol.parse_hud_reply(
        "OK frames=1 displayScore=4100 globalsScore=4100 lives=1 bombs=2 power=64 "
        "graze=0 point=0 time=0 gameOverFlag=0 displayAge=0 displayStale=33 "
        "scoreDisplayStep=0"
    )
    assert settled.settled is True
    assert settled.display_age == 0
    assert settled.score_display_step == 0


def test_hud_reply_keeps_appended_fields_optional():
    """A pre-EXP-P2-005 engine cannot exist, but the append-only contract must
    still parse a reply without the extra fields instead of raising."""

    hud = protocol.parse_hud_reply(
        "OK frames=9 displayScore=0 globalsScore=0 lives=2 bombs=3 power=0 "
        "graze=0 point=0 time=0 gameOverFlag=0 displayAge=0"
    )
    assert hud.display_stale is None
    assert hud.score_display_step is None


def test_hud_reply_fails_closed_on_missing_field():
    with pytest.raises(protocol.HarnessError, match="displayScore"):
        protocol.parse_hud_reply(
            "OK frames=1 globalsScore=0 lives=2 bombs=3 power=0 graze=0 point=0 "
            "time=0 gameOverFlag=0 displayAge=0"
        )


def test_hud_reply_fails_closed_on_non_ok_verb():
    with pytest.raises(protocol.HarnessError, match="not OK"):
        protocol.parse_hud_reply("ERR hud-not-ready reset-required")


def test_child_forwarding_decodes_hud_like_the_parent_command():
    decoded = protocol.parse_child_reply("HUD", HUD_REPLY)
    assert isinstance(decoded, protocol.HudSnapshot)
    assert decoded.display_score == 4100
    assert "HUD" in protocol.CHILD_SNAPSHOT_VERBS


def test_hud_method_sends_the_bare_command(tmp_path, monkeypatch):
    proc = _Proc(HUD_REPLY.encode() + b"\n")
    client = _client(tmp_path, proc)
    _install_read(monkeypatch, proc)
    hud = client.hud()
    assert proc.stdin.writes == [b"HUD\n"]
    assert hud.globals_score - hud.display_score == hud.display_age
