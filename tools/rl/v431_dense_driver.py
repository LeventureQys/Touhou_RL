"""v4.3.1: reach a dense late-stage frame WITHOUT `_StudentSession.replay`.

The blocker (R22)
-----------------
`_StudentSession.replay` refuses a two-frame decision whose two frames used different
masks (`_pair_cand`, `counterfactual.py:3924-3929`). Every teacher recording in this repo
violates that invariant: the first offending block starts at frame 92 on all 15 tapes
across two independent corpora (T04/T02c 2-frame slots, and the v4.1.0 S1c 4-frame shards),
firing the refusal at frame 96. So `replay()` can restore a state at frame <= 94 only.

What this file does differently
-------------------------------
`step_frame(mask)` is a plain per-frame STEP + STATE + hash; it never calls `_pair_cand`.
The action-history rebuild that needs one-mask-per-decision exists only for the obs-v3
pixel stack, which the D-state arm does not use. So the D-state arm can walk to ANY frame:

    RESET(seed) -> step_frame(mask) x (target-1) -> policy segment

Each recorded frame hash (every 2 frames in S1c, since `step_frames` is per-frame and
`hash_pre`/`hash_post` sit on the 4-frame block boundaries) is compared as it is reached,
so the restore gate is preserved rather than skipped.

Scope, stated honestly
----------------------
* This unblocks ARM A (D-state) and the constant-action / random baselines. It does NOT
  unblock the pixel-memory arms (B/C), which genuinely need the obs-v3 history buffer and
  therefore the one-mask invariant. Partial unblock, not a full one.
* No file in `touhou_rl/` is modified. This is a new script that uses the public-ish
  session primitives the way `a1_expand.py` does.

Source of masks: `.tmp_v410_s1c/d1/shards/d1-s0d1-n3300-seed<N>-shard/labels.jsonl`,
whose `step_frames` is per-frame (4 entries) and whose `decision_frame` spans 0..13,196.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time
from typing import Any, Mapping

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from touhou_rl.v400.teacher.engine import check_engine_binary      # noqa: E402
from touhou_rl.v420 import contracts as C                          # noqa: E402
from touhou_rl.v420 import counterfactual as cf                    # noqa: E402
from touhou_rl.v420.rl import check_data_dir                       # noqa: E402

SCHEMA = "v431-dense-episode/1"
N_CAND = int(C.N_MOVE_FOCUS)
S1C = pathlib.Path(".tmp_v410_s1c") / "d1" / "shards"


def log(message: str) -> None:
    print(f"[dense] {message}", flush=True)


def build_masks(seed: int, target: int) -> tuple[list[int], dict[int, str], dict]:
    p = REPO / S1C / f"d1-s0d1-n3300-seed{seed}-shard" / "labels.jsonl"
    if not p.is_file():
        raise SystemExit(f"no S1c shard at {p}")
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    masks: list[int] = []
    hashes: dict[int, str] = {}
    for r in rows:
        f = int(r["decision_frame"])
        steps = [int(x) for x in (r.get("step_frames") or [])]
        if len(steps) != 4:
            raise SystemExit(f"frame {f}: expected 4 step_frames, got {len(steps)}")
        if len(masks) > f:
            raise SystemExit(f"frame {f}: masks already at {len(masks)}; tape overlaps")
        if len(masks) < f:
            raise SystemExit(f"frame {f}: gap, masks at {len(masks)}")
        masks.extend(steps)
        hashes[f] = str(r["hash_pre"])
        hashes[f + 4] = str(r["hash_post"])
    if len(masks) < target:
        raise SystemExit(f"tape covers {len(masks)} frames, need {target}")
    # NOTE: the mask list is deliberately NOT truncated to `target`. The walk uses
    # masks[0..target-1]; a `recorded` policy then keeps following the tape from
    # masks[target] onward, which is the whole point of that baseline.
    if target not in hashes:
        raise SystemExit(f"no recorded hash at frame {target}")
    split_blocks = sum(1 for r in rows
                       if int(r["decision_frame"]) < len(masks)
                       and len(r.get("step_frames") or []) == 4
                       and ([int(x) for x in r["step_frames"]][0::2]
                            != [int(x) for x in r["step_frames"]][1::2]))
    meta = {"shard": str(p), "rows_in_shard": len(rows), "frames_built": len(masks),
            "recorded_hash_frames": len(hashes), "target": target,
            "split_mask_blocks_in_tape": split_blocks,
            "note": ("split-mask blocks are fine here: this path does not call _pair_cand")}
    return masks, hashes, meta


def make_policy(spec: str, masks: list[int], rng: random.Random):
    if spec == "recorded":
        def _tape(frame: int) -> int:
            if frame >= len(masks):
                raise SystemExit(
                    f"recorded policy ran past the tape at frame {frame} "

                    f"(tape has {len(masks)} frames); ask for fewer decisions")
            return masks[frame]
        return _tape, {"kind": "recorded", "detail": "the tape's own per-frame mask"}
    if spec.startswith("const:"):
        cand = int(spec.split(":", 1)[1])
        mask = int(C.cand_to_mask(cand))
        return (lambda frame: mask), {"kind": "const", "cand": cand, "mask": mask}
    if spec == "random":
        return (lambda frame: int(C.cand_to_mask(rng.randrange(N_CAND)))), {"kind": "random"}
    raise SystemExit(f"unknown policy {spec!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--policy", default="recorded")
    ap.add_argument("--decisions", type=int, default=90)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    C.assert_seed_allowed(int(args.seed), "train")
    masks, hashes, meta = build_masks(int(args.seed), int(args.frame))

    work = pathlib.Path(str(config["work_dir"]))
    template = work / "data" / "template-env"
    data_dir = work / "run" / f"dense-seed{args.seed}-f{args.frame}-{os.getpid()}"
    check_data_dir(str(template), role="engine data template")
    check_data_dir(str(data_dir), role="episode data dir")

    gate = dict(check_engine_binary(str(config["engine420"]),
                                   expected=config.get("engine_sha256") or None))
    bundle = cf.load_student_policy(config)
    session = cf._StudentSession(
        engine_path=str(config["engine420"]), data_dir=data_dir,
        source_env_dir=template, stage=int(config.get("stage", 0)),
        difficulty=int(config.get("difficulty", 1)), bundle=bundle, gate=gate, log=log)

    frames_per = int(config["decision_frames"])
    rng = random.Random(args.rng_seed)
    pick, policy_meta = make_policy(args.policy, masks, rng)
    target = int(args.frame)

    rec: dict[str, Any] = {
        "schema": SCHEMA, "seed": int(args.seed), "target_frame": target,
        "policy_spec": args.policy, "policy": policy_meta,
        "decisions": int(args.decisions), "frames_per_decision": frames_per,
        "source": meta, "restore": {}, "alive_120": None, "alive_180": None,
        "first_death_frame": None, "first_death_offset": None, "terminal": None,
        "frames_executed": 0, "error": None, "wall_s": None,
    }
    started = time.perf_counter()
    try:
        session.start()
        reset_frame, reset_hash = session.reset_with_retry(int(args.seed))
        checked = mismatches = 0
        first_bad = None
        for i in range(target - 1):
            h, reading, exited, game_over = session.step_frame(int(masks[i]))
            frame = i + 1
            expected = hashes.get(frame)
            if expected is not None:
                checked += 1
                if h != expected:
                    mismatches += 1
                    if first_bad is None:
                        first_bad = frame
                    break
        rec["restore"] = {"reset_frame": int(reset_frame), "reset_hash": reset_hash,
                          "frames_walked": target - 1, "hash_checks": checked,
                          "hash_mismatches": mismatches,
                          "first_mismatch_frame": first_bad,
                          "engine_frame": int(session.client.frame())}
        if mismatches:
            raise cf.CounterfactualError(
                f"manual prefix diverged at frame {first_bad} ({mismatches} mismatches)")

        baseline = session.read_state()
        deaths0 = float(baseline.get("deaths") or 0.0)
        stage0 = int(baseline.get("stage") or session.stage)
        rec["deaths_start"] = deaths0
        rec["lives_start"] = float(baseline.get("lives_remaining") or 0.0)
        rec["score_start"] = int(baseline.get("score") or 0)

        frame_now = target - 1
        died_at = None
        terminal = None
        for _ in range(int(args.decisions)):
            mask = int(pick(frame_now))
            for _ in range(frames_per):
                h, reading, exited, game_over = session.step_frame(mask)
                frame_now += 1
                rec["frames_executed"] += 1
                term = cf._terminal_of(reading, deaths_start=deaths0, stage0=stage0,
                                      hash_hex=h, frame=frame_now, exited=exited,
                                      game_over=game_over)
                if term is not None:
                    terminal = str(term.get("terminal"))
                    if term.get("engine_dead"):
                        died_at = frame_now
                    break
            if terminal is not None:
                break
        rec["terminal"] = terminal
        rec["first_death_frame"] = died_at
        off = None if died_at is None else int(died_at) - target
        rec["first_death_offset"] = off
        rec["alive_120"] = bool(off is None or off > 120)
        rec["alive_180"] = bool(off is None or off > 180)
    except Exception as exc:                                   # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            session.close()
        except Exception:                                       # noqa: BLE001
            pass
        rec["wall_s"] = round(time.perf_counter() - started, 3)
        rec["session_stats"] = dict(getattr(session, "stats", {}) or {})

    text = json.dumps(rec, indent=1, ensure_ascii=False) + "\n"
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    log(f"seed {args.seed} frame {target} policy {args.policy}: "
        f"terminal={rec['terminal']} death_off={rec['first_death_offset']} "
        f"alive120={rec['alive_120']} alive180={rec['alive_180']} "
        f"checks={rec['restore'].get('hash_checks')} "
        f"mismatch={rec['restore'].get('hash_mismatches')} wall={rec['wall_s']}s "
        f"error={rec['error']}")
    return 0 if rec["error"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
