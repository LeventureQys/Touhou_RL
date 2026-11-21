"""v4.3.1 SEG-001 driver: a policy-driven episode from a frozen A1 state.

Why this exists
---------------
The A1 pass measured **one decision** per state and then let the frozen student drive
(`cf._run_student_arm`). That is an 18-way bandit, not a 3-10 s control segment. SEG-001
needs a policy that acts every 2 frames for the whole segment, so this driver drives
`cf._StudentSession.step_frame` directly instead of calling the student.

It is deliberately staged: this file implements the *harness* with scripted policies
only (`recorded`, `const:<cand>`, `random`). No learning. The point is to prove, on the
real engine, that

  * the restore + per-frame hash gate still lands on the frozen `expected_hash`,
  * a policy can act every `decision_frames` frames for N decisions,
  * death is detected exactly (`_terminal_of`, i.e. the same classifier A1 used),
  * `alive@120` and `alive@180` are measurable for the same episode.

Only after that is green does online learning mean anything.

Reused read-only: `touhou_rl.v420.counterfactual` (`Prefix`, `_StudentSession`,
`_terminal_of`, `load_states_file`, `trace_prefix`, `load_student_policy`) and
`touhou_rl.v420.contracts.cand_to_mask`. Nothing in this repo is modified.

Writes one JSON record per episode plus a JSONL line.
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

SCHEMA = "v431-seg001-episode/1"
N_CAND = int(C.N_MOVE_FOCUS)


def log(message: str) -> None:
    print(f"[seg001] {message}", flush=True)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(config: Mapping[str, Any], key: str) -> pathlib.Path:
    value = config.get(key)
    if not value:
        raise SystemExit(f"config has no {key!r}")
    p = pathlib.Path(str(value))
    if p.is_absolute():
        return p
    return pathlib.Path(str(config.get("root420") or ".")) / p


def make_policy(spec: str, *, prefix: cf.Prefix, state, rng: random.Random):
    """Return `pick(frame) -> (cand, mask)` for a scripted policy."""
    if spec == "recorded":
        def pick(frame: int) -> tuple[int, int]:
            mask = int(prefix.masks[frame])
            return -1, mask
        return pick, {"kind": "recorded", "detail": "the tape's own mask at this frame"}

    if spec.startswith("const:"):
        cand = int(spec.split(":", 1)[1])
        if not 0 <= cand < N_CAND:
            raise SystemExit(f"const cand {cand} outside [0,{N_CAND})")
        mask = int(C.cand_to_mask(cand))
        return (lambda frame: (cand, mask)), {"kind": "const", "cand": cand,
                                              "mask": mask}

    if spec == "random":
        def pick(frame: int) -> tuple[int, int]:
            cand = rng.randrange(N_CAND)
            return cand, int(C.cand_to_mask(cand))
        return pick, {"kind": "random", "seed": rng.random()}

    raise SystemExit(f"unknown policy {spec!r}")


def tape_state_and_prefix(config: Mapping[str, Any], seed: int, frame: int):
    """Synthesise a replayable state + prefix from a teacher tape's `slots.jsonl`.

    Why this exists: A1's states all sit before frame 3,051 because the A0 student dies
    there (R21). A state in the dense late-stage regime therefore cannot come from a
    student trace; it has to come from a teacher tape, whose `slots.jsonl` carries a
    `hash_pre`/`hash_post` pair per two-frame slot and whose hash chain links with zero
    breaks over 6,588 slots.

    Alignment, from `cf.Prefix`: ``masks[i]`` is the mask applied to REACH frame ``i+1``.
    A slot at ``decision_frame f`` commits frames ``[f, f+2)`` with its own mask, so
    ``masks[f] == masks[f+1] == slot.mask``. Frames 1..16 are the eight leading decisions
    recorded as cand 7 (`.tmp_v430_o2/*/labels_provenance.json`), so they replay as
    ``cand_to_mask(7)``.

    The frame-0 RESET hash is not recorded for a teacher tape, so ``hashes[0]`` is a
    placeholder and the restore gate here is "every recorded frame 16..frame matches",
    which is what `_StudentSession.replay` actually checks. That is stated in the record.
    """

    shard = (pathlib.Path(str(config.get("root420") or "."))
             / ".tmp_v420_data" / "T04" / "dataset" / f"d1-s0d1-n3300-seed{seed}-shard"
             / "slots.jsonl")
    if not shard.is_file():
        raise SystemExit(f"no teacher tape at {shard}")
    # Per-frame masks live in the T02c rebuild (`corpus.committed_masks`), the hashes only in
    # the T04 dataset. 137 of 6,588 slots commit TWO DIFFERENT masks over their two frames,
    # which is why replaying `mask` twice diverges (measured: first divergence at frame 96 =
    # slot 39, whose committed masks are [133, 132]).
    frame_masks_path = (pathlib.Path(str(config.get("root420") or "."))
                        / ".tmp_v420_data" / "T02c" / "rebuild"
                        / f"d1-s0d1-n3300-seed{seed}-shard" / "2f" / "slots.jsonl")
    slots = [json.loads(ln) for ln in shard.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not frame_masks_path.is_file():
        raise SystemExit(f"no per-frame mask record at {frame_masks_path}")
    fm_rows = [json.loads(ln) for ln in
               frame_masks_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(fm_rows) != len(slots):
        raise SystemExit(f"{len(slots)} slots but {len(fm_rows)} per-frame rows")
    for k, (a, b) in enumerate(zip(slots, fm_rows)):
        if int(a["decision_frame"]) != int(b["decision_frame"]):
            raise SystemExit(f"slot {k}: decision_frame {a['decision_frame']} vs {b['decision_frame']}")
    split_slots = 0
    first = int(slots[0]["decision_frame"])
    if frame % 2 or frame < first:
        raise SystemExit(f"frame {frame} is not on the two-frame grid at or after {first}")

    lead = first                           # frames 1..first are the eight leading decisions
    lead_mask = int(C.cand_to_mask(7))
    masks: list[int] = [lead_mask] * lead
    hashes: dict[int, str] = {}
    for slot, fm in zip(slots, fm_rows):
        f = int(slot["decision_frame"])
        if f >= frame:
            break
        committed = (fm.get("corpus") or {}).get("committed_masks")
        if not committed or len(committed) != 2:
            committed = [int(slot["mask"]), int(slot["mask"])]
        committed = [int(x) for x in committed]
        if committed != [int(slot["mask"]), int(slot["mask"])]:
            split_slots += 1
        masks.extend(committed)
        hashes[f] = str(slot["hash_pre"])
        hashes[f + 2] = str(slot["hash_post"])
    if len(masks) != frame:
        raise SystemExit(f"built {len(masks)} masks for frame {frame}; the tape does not cover it")
    if frame not in hashes:
        raise SystemExit(f"no recorded hash at frame {frame}")
    reset_placeholder = str(slots[0]["hash_pre"])   # frame 0 is not recorded for a tape
    hashes[0] = reset_placeholder
    # Prefix is built directly, not through `cf.trace_prefix`: a teacher tape records a
    # hash every 2 frames, and `trace_prefix` insists on a contiguous per-frame trace with
    # a hash on every frame. Sparse hashes are what `Prefix` is designed for ("a four-frame
    # corpus only has every fourth frame"), and `_StudentSession.replay` skips frames whose
    # hash is absent - so the gate checks every frame that was recorded (16..frame).
    prefix = cf.Prefix(label=f"teacher-tape:seed{int(seed)}", kind=cf.KIND_TEACHER,
                       seed=int(seed), masks=tuple(masks), hashes=hashes,
                       reset_hash=reset_placeholder,
                       frames_per_decision=int(config.get("decision_frames", 2)),
                       source=str(shard))
    if len(prefix.masks) != len(masks):
        raise SystemExit(f"prefix has {len(prefix.masks)} masks, expected {len(masks)}")
    state = _TapeState(seed=int(seed), decision_frame=int(frame),
                       expected_hash=str(hashes[frame]))
    return state, prefix, {"shard": str(shard),
                           "per_frame_masks": str(frame_masks_path),
                           "frame": frame,
                           "lead_frames": lead, "lead_mask": lead_mask,
                           "hash_frames": len(hashes),
                           "slots_with_two_different_masks": split_slots,
                           "frame0_hash_available": False,
                           "gate": ("every recorded frame 16..frame is hash-checked; frame 0 "
                                    "is not recorded for a teacher tape, so the RESET check "
                                    "is unavailable for tape-derived states")}


class _TapeState:
    """Minimal stand-in for `cf.StateRef` carrying only what the driver reads."""

    __slots__ = ("seed", "decision_frame", "expected_hash")

    def __init__(self, *, seed: int, decision_frame: int, expected_hash: str) -> None:
        self.seed = seed
        self.decision_frame = decision_frame
        self.expected_hash = expected_hash


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--state", type=int, default=None,
                    help="index into states_frozen.json (A1 pool)")
    ap.add_argument("--tape", type=int, default=None,
                    help="teacher tape seed; use with --frame to freeze a dense state")
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--policy", default="recorded",
                    help="recorded | const:<cand> | random")
    ap.add_argument("--decisions", type=int, default=90,
                    help="90 decisions x 2 frames = 180 frames = 3.00 s at 60 fps")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--records", default="")
    args = ap.parse_args(argv)

    config = read_json(pathlib.Path(args.config))
    work = pathlib.Path(str(config["work_dir"]))
    tape_meta = None
    if args.tape is not None:
        if args.frame is None:
            raise SystemExit("--tape needs --frame")
        C.assert_seed_allowed(int(args.tape), "train")
        state, prefix, tape_meta = tape_state_and_prefix(config, int(args.tape), int(args.frame))
        state_index = -1
    else:
        if args.state is None:
            raise SystemExit("give --state (A1 pool) or --tape/--frame (dense regime)")
        states = cf.load_states_file(work / "states_frozen.json")
        if not 0 <= args.state < len(states):
            raise SystemExit(f"state {args.state} outside 0:{len(states)}")
        state = states[args.state]
        if int(state.seed) not in {int(s) for s in config["seed_source"]["seeds"]}:
            raise SystemExit(f"seed {state.seed} is not one of this pass's TRAIN seeds")
        C.assert_seed_allowed(int(state.seed), "train")
        state_index = int(args.state)

    template = work / "data" / "template-env"
    if not template.is_dir():
        raise SystemExit(f"no engine data template at {template}")
    data_dir = work / "run" / f"seg001-state{state_index:03d}-{os.getpid()}"
    check_data_dir(str(template), role="engine data template")
    check_data_dir(str(data_dir), role="episode data dir")

    gate = dict(check_engine_binary(str(config["engine420"]),
                                   expected=config.get("engine_sha256") or None))
    bundle = cf.load_student_policy(config)
    if str(bundle["checkpoint"].sha256) != str(config["student_checkpoint_sha256"]):
        raise SystemExit("the student checkpoint is not the configured A0")

    if tape_meta is None:
        trace = read_json(work / "student_traces" / f"seed{int(state.seed)}.json")
        prefix = cf.trace_prefix(trace)
    if prefix.hash_at(state.decision_frame) != state.expected_hash:
        raise SystemExit("the recorded trace no longer hashes to the frozen expected_hash")

    frames_per = int(config["decision_frames"])
    total_frames = args.decisions * frames_per
    rng = random.Random(args.rng_seed)
    pick, policy_meta = make_policy(args.policy, prefix=prefix, state=state, rng=rng)

    session = cf._StudentSession(
        engine_path=str(config["engine420"]), data_dir=data_dir,
        source_env_dir=template, stage=int(config.get("stage", 0)),
        difficulty=int(config.get("difficulty", 1)), bundle=bundle,
        gate=gate, log=log)

    record: dict[str, Any] = {
        "schema": SCHEMA,
        "state_index": int(state_index),
        "seed": int(state.seed),
        "decision_frame": int(state.decision_frame),
        "expected_hash": str(state.expected_hash),
        "policy": policy_meta,
        "tape": tape_meta,
        "policy_spec": args.policy,
        "decisions": int(args.decisions),
        "frames_per_decision": frames_per,
        "total_frames": int(total_frames),
        "alive_120": None,
        "alive_180": None,
        "first_death_frame": None,
        "terminal": None,
        "deaths_start": None,
        "deaths_end": None,
        "frames_executed": 0,
        "restore": {},
        "wall_s": None,
        "error": None,
    }

    started = time.perf_counter()
    try:
        session.start()
        reset_frame, reset_hash = session.reset_with_retry(int(state.seed))
        replay = session.replay(prefix, int(state.decision_frame))
        record["restore"] = {
            "reset_frame": int(reset_frame),
            "reset_hash": reset_hash,
            "recorded_reset_hash": prefix.hash_at(0),
            "final_frame": int(replay["final_frame"]),
            "final_hash": replay["final_hash"],
            "hash_mismatches": int(replay["hash_mismatches"]),
            "first_mismatch_frame": replay.get("first_mismatch_frame"),
            "matches_frozen": replay["final_hash"] == state.expected_hash,
        }
        if replay["hash_mismatches"]:
            raise cf.CounterfactualError(
                f"prefix diverged at frame {replay['first_mismatch_frame']}")
        if replay["final_hash"] != state.expected_hash:
            raise cf.CounterfactualError(
                f"restored hash {replay['final_hash']} != frozen {state.expected_hash}")

        baseline = session.read_state()
        deaths_start = float(baseline.get("deaths") or 0.0)
        stage0 = int(baseline.get("stage") or session.stage)
        record["deaths_start"] = deaths_start
        record["lives_start"] = float(baseline.get("lives_remaining") or 0.0)
        record["bombs_start"] = float(baseline.get("bombs_remaining") or 0.0)
        record["score_start"] = int(baseline.get("score") or 0)

        frame_now = int(replay["final_frame"])
        died_at: int | None = None
        terminal: str | None = None

        for decision in range(args.decisions):
            cand, mask = pick(frame_now)
            for _ in range(frames_per):
                hash_hex, reading, exited, game_over = session.step_frame(int(mask))
                frame_now += 1
                record["frames_executed"] += 1
                term = cf._terminal_of(reading, deaths_start=deaths_start,
                                      stage0=stage0, hash_hex=hash_hex,
                                      frame=frame_now, exited=exited,
                                      game_over=game_over)
                if term is not None:
                    terminal = str(term.get("terminal"))
                    if term.get("engine_dead"):
                        died_at = int(frame_now)
                    break
            if terminal is not None:
                break

        offset_at_stop = record["frames_executed"]
        record["first_death_frame"] = died_at
        record["terminal"] = terminal
        # NOTE: `died_at` is an ABSOLUTE engine frame, so it must be offset by the
        # state's own frame before being compared with the 120/180-frame horizons.
        death_offset = None if died_at is None else int(died_at) - int(state.decision_frame)
        record["first_death_offset"] = death_offset
        record["alive_120"] = bool(death_offset is None or death_offset > 120)
        record["alive_180"] = bool(death_offset is None or death_offset > 180)
        record["stopped_early_at_frame"] = offset_at_stop
        end = session.read_state()
        record["deaths_end"] = float(end.get("deaths") or 0.0)
        record["final_frame"] = int(frame_now)
    except Exception as exc:                                   # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            session.close()
        except Exception:                                       # noqa: BLE001
            pass
        record["wall_s"] = round(time.perf_counter() - started, 3)
        record["session_stats"] = dict(getattr(session, "stats", {}) or {})

    text = json.dumps(record, indent=1, ensure_ascii=False) + "\n"
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
        log(f"wrote {args.out}")
    else:
        print(text)
    if args.records:
        pathlib.Path(args.records).parent.mkdir(parents=True, exist_ok=True)
        with open(args.records, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    log(f"state {state_index} policy {args.policy}: terminal={record['terminal']} "
        f"death={record['first_death_frame']} alive120={record['alive_120']} "
        f"alive180={record['alive_180']} frames={record['frames_executed']} "
        f"wall={record['wall_s']}s error={record['error']}")
    return 0 if record["error"] is None else 1



def main(argv: list[str] | None = None) -> int:
    """Run one episode and ALWAYS leave a record.

    A missing input (for example a `student_traces/seed<N>.json` that was never copied
    into the isolated work dir) used to raise before any record was written, so the
    episode left no artifact at all and the failure was invisible except in the log.
    Failures are evidence too, so this wrapper writes a stub record on any exception.
    """

    try:
        return _main(argv)
    except SystemExit:
        raise
    except Exception as exc:                                   # noqa: BLE001
        stub = {"schema": SCHEMA, "error": f"{type(exc).__name__}: {exc}",
                "alive_120": None, "alive_180": None, "terminal": None,
                "first_death_frame": None, "frames_executed": 0}
        argv_list = list(sys.argv[1:] if argv is None else argv)
        try:
            ap = argparse.ArgumentParser(add_help=False)
            ap.add_argument("--state", type=int, default=-1)
            ap.add_argument("--policy", default="")
            ap.add_argument("--decisions", type=int, default=0)
            ap.add_argument("--out", default="")
            ap.add_argument("--records", default="")
            known, _ = ap.parse_known_args(argv_list)
        except Exception:                                      # noqa: BLE001
            known = None
        if known is not None:
            stub.update({"state_index": known.state, "policy_spec": known.policy,
                         "decisions": known.decisions})
            text = json.dumps(stub, indent=1, ensure_ascii=False) + "\n"
            if known.out:
                pathlib.Path(known.out).parent.mkdir(parents=True, exist_ok=True)
                pathlib.Path(known.out).write_text(text, encoding="utf-8")
            if known.records:
                pathlib.Path(known.records).parent.mkdir(parents=True, exist_ok=True)
                with open(known.records, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(stub, ensure_ascii=False) + "\n")
        log(f"FAILED: {stub['error']}")
        return 1

if __name__ == "__main__":
    sys.exit(main())