"""v4.3.1 recheck R18: did the A1 restore path reproduce one frozen state today?

SEG-001 rev C is built on the assumption that a frozen A1 state can still be restored:
`RESET(seed)` -> replay the recorded masks to `decision_frame` -> the per-frame hash chain
must still land on the frozen `expected_hash` -> then the 18 first actions can be run.

This script consumes the shard produced by the unchanged `a1_expand.py arms --state 0`
run (isolated work dir, muted template cfg, SDL_AUDIODRIVER=dummy) and compares its 18
verdicts against the frozen A1 table. Agreement is the precondition; any disagreement is
a blocker for SEG-001 and must be shown, not smoothed.

Read-only over both inputs. Writes only evidence/R18/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
A1 = REPO / "Document" / "Update" / "Alpha" / "v4.2.0" / "evidence" / "T08" / "a1_safe_set_full.json"
SHARD = REPO / ".tmp_v431_seg001" / "full" / "raw" / "state_000.json"
LOG = REPO / ".tmp_v431_seg001" / "full" / "logs" / "probe-state0.log"
CFG = REPO / ".tmp_v431_seg001" / "full" / "data" / "template-env" / "th08.cfg"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R18"

MUTED_SHA = "cc062cfbf9fd2c14a1225376e182beffc52f964c342c187b409045f495d5f0fa"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not SHARD.exists():
        print(f"shard not produced: {SHARD}", file=sys.stderr)
        return 2

    shard = json.loads(SHARD.read_text(encoding="utf-8"))
    a1 = json.loads(A1.read_text(encoding="utf-8"))
    state0 = a1["states"][0]

    a1_outcomes = {int(k): v for k, v in state0["outcomes"].items()}
    new_outcomes = {int(k): v for k, v in (shard.get("outcomes") or {}).items()}

    cands = sorted(set(a1_outcomes) | set(new_outcomes))
    per_cand = []
    agree = 0
    for c in cands:
        a = a1_outcomes.get(c)
        b = new_outcomes.get(c)
        same = a == b
        agree += 1 if same else 0
        # first death frame, if both report a death
        a_arms = {int(x["cand"]): x for x in (state0.get("arms") or [])}
        b_arms = {int(x["cand"]): x for x in (shard.get("arms") or [])}
        per_cand.append({
            "cand": c,
            "a1": a,
            "probe": b,
            "agree": same,
            "a1_first_death_frame": (a_arms.get(c) or {}).get("first_death_frame"),
            "probe_first_death_frame": (b_arms.get(c) or {}).get("first_death_frame"),
            "a1_executed_frames": (a_arms.get(c) or {}).get("executed_frames"),
            "probe_executed_frames": (b_arms.get(c) or {}).get("executed_frames"),
        })

    a1_safe = sorted(int(x) for x in state0["safe_set"])
    new_safe = sorted(int(x) for x in (shard.get("safe_set") or []))

    probe_cfg_sha = None
    if CFG.exists():
        import hashlib
        probe_cfg_sha = hashlib.sha256(CFG.read_bytes()).hexdigest()

    log_tail = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-12:] \
        if LOG.exists() else []

    out = {
        "schema": "v431-recheck-r18/1",
        "source_shard": str(SHARD.relative_to(REPO)),
        "source_a1": str(A1.relative_to(REPO)),
        "state": {
            "index": int(state0["index"]),
            "seed": int(state0["seed"]),
            "decision_frame": int(state0["decision_frame"]),
            "expected_hash": state0["expected_hash"],
            "prefix_frames": int(state0["prefix_frames"]),
        },
        "shard": {
            "status": shard.get("status"),
            "exit_code": shard.get("exit_code"),
            "error": shard.get("error"),
            "complete": shard.get("complete"),
            "arms": len(shard.get("arms") or []),
            "cost": shard.get("cost"),
            "repeat_agreement": shard.get("repeat_agreement"),
            "prefix_checks": shard.get("prefix_hash_checks"),
            "prefix_mismatches": shard.get("prefix_hash_mismatches"),
        },
        "audio": {
            "template_cfg_sha256": probe_cfg_sha,
            "expected_muted_sha256": MUTED_SHA,
            "muted": probe_cfg_sha == MUTED_SHA,
            "log_tail": log_tail,
        },
        "comparison": {
            "cands": len(cands),
            "outcome_agreement": agree,
            "outcome_agreement_rate": (agree / len(cands)) if cands else None,
            "safe_set_a1": a1_safe,
            "safe_set_probe": new_safe,
            "safe_set_equal": a1_safe == new_safe,
            "per_cand": per_cand,
        },
    }

    (OUT / "probe_reproduction.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    c = out["comparison"]
    print(f"state 0: seed {state0['seed']} frame {state0['decision_frame']} "
          f"expected_hash {state0['expected_hash']}")
    print(f"shard status={out['shard']['status']} exit={out['shard']['exit_code']} "
          f"arms={out['shard']['arms']} complete={out['shard']['complete']}")
    print(f"prefix checks={out['shard']['prefix_checks']} "
          f"mismatches={out['shard']['prefix_mismatches']}")
    print()
    print(f"audio muted: {out['audio']['muted']}  (cfg sha {probe_cfg_sha})")
    print()
    print(f"outcome agreement: {agree}/{len(cands)} = {c['outcome_agreement_rate']}")
    print(f"safe set  A1={a1_safe}")
    print(f"safe set new={new_safe}   equal={c['safe_set_equal']}")
    print()
    print(f"{'cand':>4s} {'A1':>14s} {'probe':>14s} {'agree':>6s} "
          f"{'a1_death':>9s} {'pr_death':>9s}")
    print("-" * 62)
    for r in per_cand:
        print(f"{r['cand']:4d} {str(r['a1']):>14s} {str(r['probe']):>14s} "
              f"{str(r['agree']):>6s} {str(r['a1_first_death_frame']):>9s} "
              f"{str(r['probe_first_death_frame']):>9s}")
    print()
    if out["shard"]["cost"]:
        print("cost:", json.dumps(out["shard"]["cost"], ensure_ascii=False))
    print()
    print(f"wrote: {OUT / 'probe_reproduction.json'}")
    return 0 if (agree == len(cands) and c["safe_set_equal"]) else 1


if __name__ == "__main__":
    sys.exit(main())
