"""v4.3.1 recheck R33: is PARK merely too early, or structurally shut?

R32 parked at frame 400 and got:
    ERR park-unsafe role=shadow-required threads=5 setup=0 null_device=0

Three numbers, three unknowns. `setup=0` and `null_device=0` could simply mean "the stage has
not finished setting up yet at frame 400" -- in which case a later frame would satisfy them and
only the thread count would remain. That changes the repair from "three judgement calls" to
"one", so it is worth 75 seconds to find out instead of guessing.

This walks the real seed-12 tape (hash-checked, same masks as R23) and retries PARK at several
later frames, recording the exact error each time. PARK does not disturb the engine when it
refuses, so retrying in one session is safe.

Writes only Document/Update/Alpha/v4.3.1/evidence/R33/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import re
import shutil
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from touhou_rl.v400.teacher.engine import check_engine_binary      # noqa: E402
from touhou_rl.v420 import contracts as C                          # noqa: E402
from touhou_rl.v420 import counterfactual as cf                    # noqa: E402
from touhou_rl.v420.rl import check_data_dir                       # noqa: E402

OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R33"
TARGETS = (400, 1200, 2400, 4800, 8664)
OVERRIDE_KEYS = ("DISPLAY", "WAYLAND_DISPLAY")


def log(message: str) -> None:
    print(f"[frames] {message}", flush=True)


def load_dense_driver():
    path = REPO / "tools" / "rl" / "v431_dense_driver.py"
    spec = importlib.util.spec_from_file_location("v431_dense_driver", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_unsafe(reply: str) -> dict:
    out = {"raw": reply}
    for key in ("threads", "setup", "null_device"):
        m = re.search(rf"{key}=(\d+)", reply)
        out[key] = int(m.group(1)) if m else None
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=12)
    args = ap.parse_args(argv)

    cfg = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    drv = load_dense_driver()
    masks, hashes, meta = drv.build_masks(int(args.seed), max(TARGETS))

    work = pathlib.Path(str(cfg["work_dir"]))
    template = work / "data" / "template-env"
    data_dir = work / "run" / f"parkframes-{os.getpid()}"
    check_data_dir(str(template), role="engine data template")
    check_data_dir(str(data_dir), role="episode data dir")
    gate = dict(check_engine_binary(str(cfg["engine420"]),
                                    expected=cfg.get("engine_sha256") or None))
    bundle = cf.load_student_policy(cfg)
    evaluator = bundle["evaluator"]

    if data_dir.exists():
        shutil.rmtree(data_dir)
    shutil.copytree(template, data_dir, symlinks=True)

    env = {k: v for k, v in os.environ.items() if k not in OVERRIDE_KEYS}
    env["SDL_AUDIODRIVER"] = "dummy"
    shm_name = f"v431pf{os.getpid() % 100000}"

    client = evaluator.EnvHarnessClient(str(cfg["engine420"]), str(data_dir),
                                        shm_name=shm_name, timeout=240.0, role="shadow",
                                        env=env)
    rec: dict = {
        "schema": "v431-recheck-r33/1",
        "question": ("is PARK refused because the stage has not finished setting up, or is it "
                     "structurally unreachable in this build?"),
        "role": "shadow", "display_unset": "DISPLAY" not in env,
        "targets": list(TARGETS), "source": {k: meta[k] for k in
                                            ("frames_built", "recorded_hash_frames")},
        "attempts": [], "any_success": None, "error": None,
    }
    try:
        client.start()
        client.attach_shm()
        client.readback(True)
        reset = client.reset(stage=int(cfg.get("stage", 0)),
                             difficulty=int(cfg.get("difficulty", 1)),
                             team=0, seed=int(args.seed))
        rec["reset"] = {"ready": reset.get("ready"), "hash": reset.get("hash"),
                        "setup_frames": reset.get("setup_frames")}
        frame = 0
        checked = mismatches = 0
        for target in TARGETS:
            while frame < target:
                reply = client.step(int(masks[frame]), 1)
                frame += 1
                expected = hashes.get(frame)
                if expected is not None:
                    checked += 1
                    if str(reply.get("hash", "")) != expected:
                        mismatches += 1
            t0 = time.perf_counter()
            # request() RAISES on an ERR reply, so a refusal must be caught per attempt:
            # letting it propagate ends the whole walk after the first frame.
            try:
                park = client.request("PARK")
            except Exception as exc:                            # noqa: BLE001
                park = str(exc)
                m = re.search(r"(ERR [^']+)", park)
                if m:
                    park = m.group(1)
            entry = {"frame": target, "park_s": round(time.perf_counter() - t0, 3),
                     "reply_ok": park.startswith("OK"),
                     "hash_checks_so_far": checked,
                     "hash_mismatches_so_far": mismatches}
            entry.update(parse_unsafe(park) if not park.startswith("OK") else {"raw": park})
            if park.startswith("OK"):
                entry["template_id"] = int(re.search(r"template=(\d+)", park).group(1))
                spawn = client.request(f"SPAWN {entry['template_id']}")
                entry["spawn"] = spawn[:120]
                m = re.search(r"handle=(\d+)", spawn)
                if spawn.startswith("OK") and m:
                    entry["child_state"] = client.request(
                        f"CHILD {int(m.group(1))} STATE")[:120]
                    entry["child_step"] = client.request(
                        f"CHILD {int(m.group(1))} STEP {int(masks[frame])} 1")[:120]
                    client.request(f"CHILD {int(m.group(1))} QUIT")
            rec["attempts"].append(entry)
            log(f"frame {target}: {'OK' if entry['reply_ok'] else 'REFUSED'} "
                f"threads={entry.get('threads')} setup={entry.get('setup')} "
                f"null={entry.get('null_device')}")
            if entry["reply_ok"]:
                break
        rec["any_success"] = any(a["reply_ok"] for a in rec["attempts"])
    except Exception as exc:                                    # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            client.close()
        except Exception:                                       # noqa: BLE001
            pass

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "park_frames.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print(f"{'frame':>6s} {'ok':>3s} {'threads':>8s} {'setup':>6s} {'null':>5s} "
          f"{'hash_ok':>8s}  reply")
    print("-" * 78)
    for a in rec["attempts"]:
        print(f"{a['frame']:6d} {str(a['reply_ok']):>3s} {str(a.get('threads')):>8s} "
              f"{str(a.get('setup')):>6s} {str(a.get('null_device')):>5s} "
              f"{a['hash_checks_so_far'] - a['hash_mismatches_so_far']:8d}  "
              f"{a['raw'][:38]}")
    print()
    print(f"any success: {rec['any_success']}")
    if rec["error"]:
        print(f"ERROR: {rec['error']}")
    print(f"wrote: {OUT / 'park_frames.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
