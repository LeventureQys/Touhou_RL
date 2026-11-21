"""v4.3.1 SEG-001 gate arithmetic: turn episode records into the S1-S7 verdicts.

Why this file exists
--------------------
SEG-001 rev C is frozen but not executed (no engine budget authorised yet). The part of
the future run that is easiest to get silently wrong is not the training loop -- it is
the arithmetic that decides whether the run passed or must stop. This module implements
only that, as pure functions over records, so it can be tested without an engine.

Record shape (one JSON object per line, produced by the future run)::

    {"arm": "A|B|C|C-fair|C-priv",
     "split": "train|dev",
     "state_index": 0,
     "attempt": 0,
     "alive120": true,
     "alive180": false,
     "aborted": false}

Rules implemented (from PREREGISTRATION_SEG_001.md):

  S1  hash-gate mismatch rate > 5%                       -> stop
  S2  arm A dev alive@120 <= C-fair (Wilson intervals overlap) -> stop
  S4  arms B and C both <= C-fair                        -> stop
  S6  alive@120 and alive@180 disagree in direction       -> stop
  S7  arm A train 100% but dev <= C-fair                  -> "memorisation", not a pass
  aborted attempts are excluded from both numerator and denominator

Wilson intervals are used (no continuity correction), matching the convention already in
`touhou_rl/v420/contracts.py`.

Self-test: `python tools/rl/v431_seg001_gate.py --selftest`
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any, Iterable, Mapping, Sequence

FPS_GATE = 0.05
ARMS = ("A", "B", "C", "C-fair", "C-const-best")
MIN_ADMISSIBLE = 4


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def wilson(successes: int, trials: int, z: float = 1.96) -> dict[str, float] | None:
    """Wilson score interval, no continuity correction (repo convention)."""
    if trials <= 0:
        return None
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return {"successes": successes, "trials": trials, "p": p,
            "lower": max(0.0, centre - half), "upper": min(1.0, centre + half),
            "method": "Wilson score interval, no continuity correction"}


def intervals_overlap(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
    return not (a["upper"] < b["lower"] or b["upper"] < a["lower"])


def strictly_better(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
    """`a` beats `b` only if its point estimate is higher AND the intervals do not overlap."""
    return a["p"] > b["p"] and not intervals_overlap(a, b)


# --------------------------------------------------------------------------- #
# accounting
# --------------------------------------------------------------------------- #
def load_records(paths: Iterable[pathlib.Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def rate(records: Sequence[Mapping[str, Any]], *, arm: str, split: str,
         field: str) -> dict[str, float] | None:
    """Success rate for one arm/split/field, with aborted attempts excluded."""
    sel = [r for r in records
           if r.get("arm") == arm and r.get("split") == split and not r.get("aborted")]
    if not sel:
        return None
    n = len(sel)
    k = sum(1 for r in sel if bool(r.get(field)))
    return wilson(k, n)


def aborted_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        if r.get("aborted"):
            key = str(r.get("arm"))
            out[key] = out.get(key, 0) + 1
    return out


def hash_gate_mismatch_rate(records: Sequence[Mapping[str, Any]]) -> float:
    """Share of attempts that failed the restoration hash gate."""
    sel = [r for r in records if "hash_gate_ok" in r]
    if not sel:
        return 0.0
    return sum(1 for r in sel if not r["hash_gate_ok"]) / len(sel)


# --------------------------------------------------------------------------- #
# gate decisions
# --------------------------------------------------------------------------- #
def decide(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    a120 = rate(records, arm="A", split="dev", field="alive120")
    a180 = rate(records, arm="A", split="dev", field="alive180")
    b180 = rate(records, arm="B", split="dev", field="alive180")
    c180 = rate(records, arm="C", split="dev", field="alive180")
    fair180 = rate(records, arm="C-fair", split="dev", field="alive180")
    const180 = rate(records, arm="C-const-best", split="dev", field="alive180")
    a_train = rate(records, arm="A", split="train", field="alive180")

    verdicts: dict[str, Any] = {
        "rates": {"A": a180, "B": b180, "C": c180,
                  "C-fair": fair180, "C-const-best": const180,
                  "A@120": a120, "A_train": a_train},
        "aborted": aborted_counts(records),
        "hash_gate_mismatch_rate": hash_gate_mismatch_rate(records),
        "triggered": [],
        "notes": [],
    }

    if verdicts["hash_gate_mismatch_rate"] > FPS_GATE:
        verdicts["triggered"].append({
            "rule": "S1",
            "why": f"hash gate mismatch rate {verdicts['hash_gate_mismatch_rate']:.3f} > {FPS_GATE}",
            "action": "stop: restoration protocol untrustworthy, fix before any arm"})

    # S1b: too few C4-admissible states to say anything
    admissible = {str(r.get("cluster")) for r in records
                  if r.get("admissible") is True and r.get("cluster") is not None}
    verdicts["admissible_states"] = len(admissible)
    if records and len(admissible) and len(admissible) < MIN_ADMISSIBLE:
        verdicts["triggered"].append({
            "rule": "S1b",
            "why": f"only {len(admissible)} C4-admissible state(s) < {MIN_ADMISSIBLE}",
            "action": "stop: widen the C4 sweep or change the state source; do not pad "
                      "with states that failed C4"})

    def _must_beat(arm_rate: Mapping[str, float] | None, name: str) -> list[str]:
        """Arm must be strictly better than BOTH baselines."""
        beaten = []
        for base_name, base in (("C-fair", fair180), ("C-const-best", const180)):
            if base is None:
                continue
            if arm_rate is None or not strictly_better(arm_rate, base):
                beaten.append(f"{base_name} ({arm_rate['p'] if arm_rate else float('nan'):.3f} "
                              f"vs {base['p']:.3f})")
        return beaten

    if a180 is not None and (fair180 is not None or const180 is not None):
        lost = _must_beat(a180, "A")
        if lost:
            verdicts["triggered"].append({
                "rule": "S2",
                "why": "arm A does not beat " + "; ".join(lost),
                "action": "stop: do not start the pixel arms; inspect reward/exploration/optimisation"})

    if b180 is not None and c180 is not None:
        b_lost = _must_beat(b180, "B")
        c_lost = _must_beat(c180, "C")
        if b_lost and c_lost:
            verdicts["triggered"].append({
                "rule": "S4",
                "why": "neither B nor C beats the baselines",
                "action": "stop: do not scale to more segments or seeds"})

    if a120 is not None and a180 is not None and fair180 is not None:
        fair120 = rate(records, arm="C-fair", split="dev", field="alive120")
        if fair120 is not None:
            d120 = a120["p"] - fair120["p"]
            d180 = a180["p"] - fair180["p"]
            if (d120 > 0) != (d180 > 0):
                verdicts["triggered"].append({
                    "rule": "S6",
                    "why": f"delta@120 {d120:+.3f} vs delta@180 {d180:+.3f} have opposite signs",
                    "action": "stop: unify the segment length before judging"})

    if a_train is not None and a180 is not None:
        lost = _must_beat(a180, "A")
        if a_train["p"] >= 1.0 and lost:
            verdicts["notes"].append({
                "rule": "S7",
                "why": ("arm A fits its training states perfectly but does not beat the "
                        "baselines on dev: memorisation, not generalisation"),
                "action": "not a stop, but must not be reported as 'learned'"})

    if const180 is not None and fair180 is not None:
        verdicts["notes"].append({
            "rule": "reference",
            "why": (f"best constant action survives {const180['p']:.3f} vs random "
                    f"{fair180['p']:.3f}: the harder of the two baselines "
                    f"{'is the constant one' if const180['p'] > fair180['p'] else 'is random'}"),
            "action": "report only",
        })

    verdicts["passed_all_gates"] = not verdicts["triggered"]
    return verdicts


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
def _rec(arm, split, alive120, alive180=None, aborted=False, hash_ok=True, i=0):
    return {"arm": arm, "split": split, "state_index": i, "attempt": 0,
            "alive120": bool(alive120),
            "alive180": bool(alive120 if alive180 is None else alive180),
            "aborted": bool(aborted), "hash_gate_ok": bool(hash_ok)}


def selftest() -> int:
    checks = 0

    def eq(name, got, want):
        nonlocal checks
        checks += 1
        if got != want:
            raise AssertionError(f"{name}: got {got!r} want {want!r}")

    # ---- Wilson matches a hand-computed value ----
    w = wilson(8, 10)
    eq("wilson p", round(w["p"], 6), 0.8)
    eq("wilson lower", round(w["lower"], 4), 0.4902)
    eq("wilson upper", round(w["upper"], 4), 0.9433)
    eq("wilson zero trials", wilson(0, 0), None)
    eq("wilson 0/10 lower", round(wilson(0, 10)["lower"], 6), 0.0)
    eq("wilson 10/10 upper", round(wilson(10, 10)["upper"], 6), 1.0)

    # ---- overlap / better ----
    eq("overlap true", intervals_overlap(wilson(8, 10), wilson(6, 10)), True)
    eq("overlap false", intervals_overlap(wilson(99, 100), wilson(1, 100)), False)
    eq("strictly better", strictly_better(wilson(99, 100), wilson(1, 100)), True)
    eq("not better equal", strictly_better(wilson(5, 10), wilson(5, 10)), False)

    # ---- aborted excluded ----
    recs = [_rec("A", "dev", True), _rec("A", "dev", False),
            _rec("A", "dev", False, aborted=True)]
    eq("aborted excluded", rate(recs, arm="A", split="dev", field="alive180")["trials"], 2)
    eq("aborted counted", aborted_counts(recs), {"A": 1})

    # ---- S1: hash gate ----
    recs = [_rec("A", "dev", True, hash_ok=(i < 4), i=i) for i in range(10)]
    d = decide(recs)
    eq("S1 fires", any(t["rule"] == "S1" for t in d["triggered"]), True)

    # ---- S2: arm A must beat BOTH baselines ----
    # A beats C-fair but NOT C-const-best -> S2 must still fire (this is the rev D point)
    recs = ([_rec("A", "dev", i < 80, i=i) for i in range(100)]
            + [_rec("C-fair", "dev", i < 20, i=i) for i in range(100)]
            + [_rec("C-const-best", "dev", i < 85, i=i) for i in range(100)])
    d = decide(recs)
    eq("S2 fires when a constant beats A", any(t["rule"] == "S2" for t in d["triggered"]), True)

    # A beats both -> S2 quiet
    recs = ([_rec("A", "dev", i < 95, i=i) for i in range(100)]
            + [_rec("C-fair", "dev", i < 20, i=i) for i in range(100)]
            + [_rec("C-const-best", "dev", i < 40, i=i) for i in range(100)])
    d = decide(recs)
    eq("S2 quiet when A beats both", any(t["rule"] == "S2" for t in d["triggered"]), False)
    eq("passes", d["passed_all_gates"], True)

    # ---- S1b: too few admissible states ----
    recs = []
    for arm, k in (("A", 90), ("C-fair", 20), ("C-const-best", 30)):
        recs += [dict(_rec(arm, "dev", i < k, i=i), admissible=True, cluster=0)
                 for i in range(100)]
    d = decide(recs)
    eq("S1b fires with 1 admissible cluster",
       any(t["rule"] == "S1b" for t in d["triggered"]), True)

    # enough clusters -> S1b quiet
    recs = []
    for cluster in range(5):
        for arm, k in (("A", 90), ("C-fair", 20), ("C-const-best", 30)):
            recs += [dict(_rec(arm, "dev", i < k, i=(cluster * 100 + i)),
                          admissible=True, cluster=cluster) for i in range(100)]
    d = decide(recs)
    eq("S1b quiet with 5 clusters",
       any(t["rule"] == "S1b" for t in d["triggered"]), False)

    # ---- S4: neither B nor C beats both baselines ----
    recs = ([_rec("A", "dev", True, i=i) for i in range(40)]
            + [_rec("B", "dev", False, i=i) for i in range(40)]
            + [_rec("C", "dev", False, i=i) for i in range(40)]
            + [_rec("C-fair", "dev", False, i=i) for i in range(40)]
            + [_rec("C-const-best", "dev", False, i=i) for i in range(40)])
    d = decide(recs)
    eq("S4 fires", any(t["rule"] == "S4" for t in d["triggered"]), True)

    # ---- S6: direction flip between horizons ----
    recs = ([_rec("A", "dev", True, False, i=i) for i in range(40)]
            + [_rec("C-fair", "dev", False, True, i=i) for i in range(40)])
    d = decide(recs)
    eq("S6 fires", any(t["rule"] == "S6" for t in d["triggered"]), True)

    # ---- S7: train perfect, dev not better than both ----
    recs = ([_rec("A", "train", True, i=i) for i in range(20)]
            + [_rec("A", "dev", i < 25, i=i) for i in range(100)]
            + [_rec("C-fair", "dev", i < 20, i=i) for i in range(100)]
            + [_rec("C-const-best", "dev", i < 40, i=i) for i in range(100)])
    d = decide(recs)
    eq("S7 note", any(n["rule"] == "S7" for n in d["notes"]), True)
    eq("S7 is not a stop", any(t["rule"] == "S7" for t in d["triggered"]), False)

    # ---- C-priv is gone from the rev D gate ----
    eq("no C-priv rule anywhere",
       any("C-priv" in json.dumps(t) for t in d["triggered"]) or
       any("C-priv" in json.dumps(n) for n in d["notes"]), False)

    # ---- empty input must not crash ----
    d = decide([])
    eq("empty passes", d["passed_all_gates"], True)

    print(f"selftest: {checks} checks passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--records", nargs="*", default=[],
                    help="JSONL files of episode records")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.records:
        print("nothing to do: pass --selftest or --records <jsonl>...", file=sys.stderr)
        return 2

    records = load_records(pathlib.Path(p) for p in args.records)
    verdict = decide(records)
    text = json.dumps(verdict, indent=1, ensure_ascii=False) + "\n"
    if args.out:
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote: {args.out}")
    else:
        print(text)
    print(f"records={len(records)}  passed_all_gates={verdict['passed_all_gates']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
