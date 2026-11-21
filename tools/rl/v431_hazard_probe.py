"""v4.3.1 recheck R26: can an env-role session see bullets?

R25 showed `STATE` carries no bullet or boss field, so the prereg's arm A ("player
position, power, boss HP, nearest-bullet geometry") cannot be built from STATE. The only
bullet channel is `HAZARDS`, which appears in `CHILD_SNAPSHOT_VERBS` - i.e. it is a
*child* snapshot verb, used by the parent to talk to forked children.

This probe asks the question directly instead of inferring it: start one env-role session,
walk to a DENSE frame (where there are certainly bullets), then issue each snapshot verb on
the env session itself and report the raw outcome.

  HAZARDS / PLAYER / EVENTS / HUD  -> reply, or the exact error

The answer decides how arm A is built:
  * HAZARDS answers  -> A-HAZARDS: the state arm can see bullets
  * HAZARDS refuses  -> A-STATE:  the arm is limited to the 25 STATE fields, and any
                                  "state arm failed" conclusion must be qualified by
                                  "blind to bullets"

Writes only Document/Update/Alpha/v4.3.1/evidence/R26/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from touhou_rl.v400.teacher.engine import check_engine_binary      # noqa: E402
from touhou_rl.v420 import contracts as C                          # noqa: E402
from touhou_rl.v420 import counterfactual as cf                    # noqa: E402
from touhou_rl.v420.rl import check_data_dir                       # noqa: E402

OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R26"
VERBS = ("HAZARDS", "PLAYER", "EVENTS", "HUD", "STATE")


def log(message: str) -> None:
    print(f"[hazprobe] {message}", flush=True)


def load_dense_driver():
    path = REPO / "tools" / "rl" / "v431_dense_driver.py"
    spec = importlib.util.spec_from_file_location("v431_dense_driver", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--frame", type=int, default=8664)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    C.assert_seed_allowed(int(args.seed), "train")
    drv = load_dense_driver()
    masks, hashes, meta = drv.build_masks(int(args.seed), int(args.frame))

    work = pathlib.Path(str(config["work_dir"]))
    template = work / "data" / "template-env"
    data_dir = work / "run" / f"hazprobe-{os.getpid()}"
    check_data_dir(str(template), role="engine data template")
    check_data_dir(str(data_dir), role="episode data dir")

    gate = dict(check_engine_binary(str(config["engine420"]),
                                   expected=config.get("engine_sha256") or None))
    bundle = cf.load_student_policy(config)
    session = cf._StudentSession(
        engine_path=str(config["engine420"]), data_dir=data_dir,
        source_env_dir=template, stage=int(config.get("stage", 0)),
        difficulty=int(config.get("difficulty", 1)), bundle=bundle, gate=gate, log=log)

    rec: dict = {
        "schema": "v431-recheck-r26/1",
        "role": "env (the role every arm must run in: obs-v3 needs pixels)",
        "seed": int(args.seed), "frame": int(args.frame), "source": meta,
        "walk": {}, "verbs": {}, "error": None,
    }
    try:
        session.start()
        reset_frame, reset_hash = session.reset_with_retry(int(args.seed))
        checked = mismatches = 0
        for i in range(int(args.frame) - 1):
            h, _reading, _exited, _game_over = session.step_frame(int(masks[i]))
            expected = hashes.get(i + 1)
            if expected is not None:
                checked += 1
                if h != expected:
                    mismatches += 1
                    break
        rec["walk"] = {"frames": int(args.frame) - 1, "hash_checks": checked,
                       "hash_mismatches": mismatches,
                       "engine_frame": int(session.client.frame())}
        if mismatches:
            raise cf.CounterfactualError(f"walk diverged ({mismatches} mismatches)")

        for verb in VERBS:
            entry: dict = {"verb": verb}
            try:
                reply = session.client.request(verb)
                entry["ok"] = True
                entry["reply_len"] = len(reply)
                entry["reply_head"] = reply[:240]
                entry["is_err"] = reply.strip().upper().startswith("ERR")
            except Exception as exc:                            # noqa: BLE001
                entry["ok"] = False
                entry["error"] = f"{type(exc).__name__}: {exc}"
            rec["verbs"][verb] = entry
            log(f"{verb}: ok={entry['ok']} "
                f"{entry.get('reply_len', entry.get('error', ''))}")
    except Exception as exc:                                    # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            session.close()
        except Exception:                                       # noqa: BLE001
            pass

    def answered(verb: str) -> bool:
        e = rec["verbs"].get(verb) or {}
        return bool(e.get("ok")) and not e.get("is_err")

    rec["verdict"] = {
        "hazards_answered_on_env": answered("HAZARDS"),
        "player_answered_on_env": answered("PLAYER"),
        "events_answered_on_env": answered("EVENTS"),
        "hud_answered_on_env": answered("HUD"),
        "state_answered_on_env": answered("STATE"),
        "arm_a_choice": ("A-HAZARDS: bullets are visible on the env session"
                         if answered("HAZARDS") else
                         "A-STATE: bullets are NOT visible; the arm is limited to the 25 "
                         "STATE fields and any negative result must be qualified"),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    target = pathlib.Path(args.out) if args.out else OUT / "hazard_probe.json"
    target.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print(f"walk: {rec['walk']}")
    for verb in VERBS:
        e = rec["verbs"].get(verb) or {}
        if e.get("ok"):
            print(f"  {verb:8s} OK   len={e['reply_len']:6d}  "
                  f"is_err={e['is_err']}  head={e['reply_head'][:80]!r}")
        else:
            print(f"  {verb:8s} FAIL {e.get('error')}")
    print()
    print(f"arm A choice: {rec['verdict']['arm_a_choice']}")
    if rec["error"]:
        print(f"ERROR: {rec['error']}")
    print(f"wrote: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
