"""Multiprocessing smoke checks for the device-free WP5 shadow worker.

These tests exercise Python process and IPC lifecycle only; they do not claim a
real game fork, rendering, or pixel-level consistency.
"""

import multiprocessing
import os
import time

from touhou_rl.v400.wp5 import shadow_worker


SEED = 2026
ACTIONS = [None, "left", {"fire": 1}, ["right", 2]]


def _crash_worker():
    os._exit(17)


def _run_worker(ctx):
    parent, child = ctx.Pipe(duplex=True)
    process = ctx.Process(target=shadow_worker, args=(child,))
    process.start()
    child.close()
    parent.send({"seed": SEED, "actions": ACTIONS})
    assert parent.poll(5.0)
    result = parent.recv()
    process.join(5.0)
    assert not process.is_alive()
    parent.close()
    assert process.exitcode == 0
    assert result["ok"] is True
    return result["state_digests"]


def test_two_independent_workers_match_state_digests():
    ctx = multiprocessing.get_context("spawn")
    assert _run_worker(ctx) == _run_worker(ctx)


def test_abnormal_worker_can_be_reaped_without_deadlock():
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe(duplex=True)
    process = ctx.Process(target=_crash_worker)
    process.start()
    child.close()
    process.join(5.0)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
    parent.close()
    assert not process.is_alive()
    assert process.exitcode != 0
