"""v4.3.1 recheck R32: the fork-server already exists — PARK / SPAWN / CHILD.

R30b closed the last client-side optimisation: 7.7 ms/frame is the engine simulating a frame,
so the 70.4 s episode cost cannot be reduced by batching. The conclusion recorded in rev L was
"only a save-state can help, and that needs approval to touch third_party".

That conclusion was WRONG, and this probe is why. `third_party/th08-rl/src/rlharness/fork.cpp`
already implements a fork-server:

    PARK                  -> OK park=1 template=<id> pid=<pid> frame=<f>
    SPAWN <template_id>   -> OK spawned=1 pid=<pid> handle=<h> template=<id>
    CHILD <handle> <cmd>  -> forwards the child's own reply line

The child accepts read-only commands (HELLO/HASH/FRAME/PING/STATE/PLAYER/EVENTS/HAZARDS/
BULLETS/LASERS/ENEMIES) and `STEP <mask> <frames<=100000>`. The only gate is `Ready()`, which
requires the shadow role, one thread, and no DISPLAY.

What this probe establishes, on the real engine:

  1. a shadow-role client (role="shadow", DISPLAY unset) can start, RESET and step;
  2. PARK succeeds at a mid-stage frame and returns a template id;
  3. a child spawned from that template, driven with the SAME mask list, produces a hash
     trace IDENTICAL to the parent's -- i.e. the fork is an exact state copy;
  4. a SECOND child from the same template reproduces the same trace again -- i.e. the
     prefix has to be replayed only once;
  5. the child answers STATE and HAZARDS, so arm A's full observation is available on it.

If (3) and (4) hold, the 8,663-frame prefix (96% of every episode) becomes a one-time cost and
every later experiment gets cheap without touching third_party at all.

Writes only Document/Update/Alpha/v4.3.1/evidence/R32/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import re
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from touhou_rl.v400.teacher.engine import check_engine_binary      # noqa: E402
from touhou_rl.v420 import contracts as C                          # noqa: E402
from touhou_rl.v420 import counterfactual as cf                    # noqa: E402
from touhou_rl.v420.rl import check_data_dir                       # noqa: E402

OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R32"
WARMUP_FRAMES = 400
TRACE_FRAMES = 300
OVERRIDE_KEYS = ("DISPLAY", "WAYLAND_DISPLAY")


def log(message: str) -> None:
    print(f"[park] {message}", flush=True)


def mask_list(n: int) -> list[int]:
    """A reproducible pseudo-random mask sequence -- the probe tests exactness, not policy."""
    out = []
    for i in range(n):
        cand = (i * 7 + (i // 5) * 3) % int(C.N_MOVE_FOCUS)
        out.append(int(C.cand_to_mask(cand)))
    return out


def head(reply: str, n: int = 90) -> str:
    return reply if len(reply) <= n else reply[:n] + "..."


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=12)
    args = ap.parse_args(argv)

    cfg = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    work = pathlib.Path(str(cfg["work_dir"]))
    template = work / "data" / "template-env"
    data_dir = work / "run" / f"park-{os.getpid()}"
    check_data_dir(str(template), role="engine data template")
    check_data_dir(str(data_dir), role="episode data dir")
    gate = dict(check_engine_binary(str(cfg["engine420"]),
                                    expected=cfg.get("engine_sha256") or None))

    bundle = cf.load_student_policy(cfg)
    evaluator = bundle["evaluator"]

    import shutil
    if data_dir.exists():
        shutil.rmtree(data_dir)
    shutil.copytree(template, data_dir, symlinks=True)

    env = {k: v for k, v in os.environ.items() if k not in OVERRIDE_KEYS}
    env["SDL_AUDIODRIVER"] = "dummy"
    shm_name = f"v431park{os.getpid() % 100000}"

    client = evaluator.EnvHarnessClient(str(cfg["engine420"]), str(data_dir),
                                        shm_name=shm_name, timeout=240.0, role="shadow",
                                        env=env)
    record: dict = {
        "schema": "v431-recheck-r32/1",
        "role": "shadow", "display_unset": "DISPLAY" not in env,
        "engine_gate": gate, "seed": int(args.seed),
        "protocol": {"park": "OK park=1 template=<id> pid=<pid> frame=<f>",
                     "spawn": "OK spawned=1 pid=<pid> handle=<h> template=<id>",
                     "child": "CHILD <handle> <command> -> forwards the child reply"},
        "steps": {}, "traces_match": None, "second_spawn_traces_match": None,
        "error": None,
    }
    try:
        client.start()
        client.attach_shm()
        client.readback(True)
        reset = client.reset(stage=int(cfg.get("stage", 0)),
                             difficulty=int(cfg.get("difficulty", 1)),
                             team=0, seed=int(args.seed))
        record["steps"]["reset"] = head(reset)
        # let the stage reach a live mid-stage state
        for i in range(WARMUP_FRAMES):
            client.step(mask_list(WARMUP_FRAMES)[i], 1)
        record["steps"]["warmup_frame"] = int(client.frame())
        record["steps"]["warmup_state_hash"] = str(client.request("HASH"))

        t0 = time.perf_counter()
        park_reply = client.request("PARK")
        record["steps"]["park_s"] = round(time.perf_counter() - t0, 3)
        record["steps"]["park"] = head(park_reply, 160)
        m = re.search(r"template=(\d+)", park_reply)
        if not park_reply.startswith("OK") or not m:
            raise RuntimeError(f"PARK refused: {park_reply}")
        template_id = int(m.group(1))
        record["steps"]["template_id"] = template_id

        masks = mask_list(TRACE_FRAMES)
        t0 = time.perf_counter()
        parent_trace = [str(client.step(mk, 1).get("hash", "")) for mk in masks]
        record["steps"]["parent_trace_s"] = round(time.perf_counter() - t0, 3)
        record["steps"]["parent_frame_after"] = int(client.frame())

        traces = []
        spawn_times = []
        for attempt in range(2):
            t0 = time.perf_counter()
            spawn_reply = client.request(f"SPAWN {template_id}")
            spawn_times.append(round(time.perf_counter() - t0, 3))
            h = re.search(r"handle=(\d+)", spawn_reply)
            if not spawn_reply.startswith("OK") or not h:
                raise RuntimeError(f"SPAWN refused: {spawn_reply}")
            handle = int(h.group(1))
            t0 = time.perf_counter()
            child_trace = []
            for mk in masks:
                reply = client.request(f"CHILD {handle} STEP {mk} 1")
                mm = re.search(r"hash=([0-9a-f]{16})", reply)
                child_trace.append(mm.group(1) if mm else f"BAD:{head(reply, 40)}")
            child_s = round(time.perf_counter() - t0, 3)
            traces.append({"attempt": attempt, "handle": handle,
                           "spawn_s": spawn_times[-1], "trace_s": child_s,
                           "matches_parent": child_trace == parent_trace,
                           "first_diff": next((i for i, (a, b) in
                                               enumerate(zip(child_trace, parent_trace)) if a != b),
                                              None)})
            if attempt == 0:
                record["steps"]["child_state"] = head(
                    client.request(f"CHILD {handle} STATE"), 120)
                record["steps"]["child_hazards"] = head(
                    client.request(f"CHILD {handle} HAZARDS"), 120)
                client.request(f"CHILD {handle} QUIT")

        record["steps"]["spawns"] = traces
        record["steps"]["spawn_s"] = spawn_times
        record["traces_match"] = traces[0]["matches_parent"]
        record["second_spawn_traces_match"] = traces[1]["matches_parent"]
        record["trace_frames"] = TRACE_FRAMES
        record["child_ms_per_frame"] = round(1000.0 * traces[0]["trace_s"] / TRACE_FRAMES, 3)
        record["parent_ms_per_frame"] = round(
            1000.0 * record["steps"]["parent_trace_s"] / TRACE_FRAMES, 3)
    except Exception as exc:                                      # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            client.close()
        except Exception:                                         # noqa: BLE001
            pass

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "park_spawn.json").write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print(f"role={record['role']} DISPLAY_unset={record['display_unset']}")
    for k in ("reset", "warmup_frame", "park", "park_s", "template_id",
              "parent_trace_s", "child_ms_per_frame", "parent_ms_per_frame",
              "child_state", "child_hazards"):
        if k in record["steps"]:
            print(f"  {k:22s} {record['steps'][k]}")
    print()
    for t in record.get("steps", {}).get("spawns", []):
        print(f"  spawn {t['attempt']}: handle={t['handle']} spawn={t['spawn_s']}s "
              f"trace={t['trace_s']}s match={t['matches_parent']} first_diff={t['first_diff']}")
    print()
    print(f"traces match parent   : {record['traces_match']}")
    print(f"2nd spawn also matches: {record['second_spawn_traces_match']}")
    if record["error"]:
        print(f"ERROR: {record['error']}")
    print(f"wrote: {OUT / 'park_spawn.json'}")
    return 0 if record["error"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
