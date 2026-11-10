"""Live WP3 item export acceptance against the real WSL TH08 executable.

Requires the existing WSL RL build, xvfb-run and authentic calibration game
assets; no synthetic item lists or mocked protocol responses are accepted.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

BINARY = "/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern"
DATA = "/mnt/d/WorkShop/Touhou_RL/.tmp/calibration"


def _run(role: str, commands: list[str]) -> list[str]:
    if shutil.which("wsl") is None:
        pytest.skip("WSL executable unavailable")
    shell = (
        "test -x /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern "
        "&& test -f /mnt/d/WorkShop/Touhou_RL/.tmp/calibration/th08.dat"
    )
    available = subprocess.run(["wsl", "--exec", "sh", "-lc", shell], timeout=15)
    if available.returncode:
        pytest.skip("existing RL build or authentic TH08 game data unavailable")
    argv = ["wsl", "--exec"]
    if role == "env":
        if subprocess.run(["wsl", "--exec", "sh", "-lc", "command -v xvfb-run"],
                          capture_output=True, timeout=15).returncode:
            pytest.skip("xvfb-run unavailable")
        argv += ["xvfb-run", "-a", "env"]
    else:
        argv += ["env", "-u", "DISPLAY"]
    argv += [
        "SDL_AUDIODRIVER=dummy", "LIBGL_ALWAYS_SOFTWARE=1",
        "GALLIUM_DRIVER=llvmpipe", "LP_NUM_THREADS=1", BINARY,
        "--data-dir", DATA, "--rl-control", "0,1", "--rl-role", role,
    ]
    proc = subprocess.run(
        argv, input="\n".join(commands) + "\n", text=True,
        capture_output=True, timeout=120,
    )
    assert proc.returncode == 0, (role, proc.returncode, proc.stdout, proc.stderr)
    lines = proc.stdout.splitlines()
    assert len(lines) == len(commands), (role, lines, proc.stderr)
    return lines


def _items(reply: str, *, offset: int, limit: int, frame: int) -> tuple[int, list[str]]:
    match = re.fullmatch(
        r"OK frames=(\d+) count=(\d+) offset=(\d+) limit=(\d+) items=(.*)", reply
    )
    assert match is not None, reply
    assert (int(match[1]), int(match[3]), int(match[4])) == (frame, offset, limit)
    count = int(match[2])
    records = match[5].split(",") if match[5] else []
    assert len(records) == min(limit, max(0, count - offset))
    indices = []
    for record in records:
        assert re.fullmatch(r"\d+:\d+:\d+:[0-9a-f]{8}:[0-9a-f]{8}:[0-9a-f]{8}:-?\d+", record)
        indices.append(int(record.split(":", 1)[0]))
    assert indices == sorted(set(indices))
    return count, records


@pytest.mark.parametrize("role", ["shadow", "env"])
def test_live_items_are_counted_serialized_and_paginated(role):
    lines = _run(role, [
        "HELLO", "ITEMS", "RESET 0 1 0 1234", "STEP 0 1200",
        "ITEMS 0 16", "ITEMS 1 2", "HASH", "STATE", "QUIT",
    ])
    assert f"role={role}" in lines[0] and "proto=1" in lines[0]
    assert lines[1] == "OK count=undefined items=undefined frames=0"
    assert "ready=1" in lines[2]
    assert "advanced=1200" in lines[3]
    total, records = _items(lines[4], offset=0, limit=16, frame=1200)
    assert total > 2, "the real stage produced too few live items; pagination is not validated"
    assert total <= 16, "increase the full-page limit before claiming full-list pagination"
    assert len(records) == total
    page_total, page = _items(lines[5], offset=1, limit=2, frame=1200)
    assert page_total == total
    assert page == records[1:3]
    assert lines[6].startswith("OK hash=")
    assert lines[7].startswith("OK frames=1200 f0=1200 ")
    assert lines[8] == "OK bye"
