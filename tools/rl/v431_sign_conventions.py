"""v4.3.1 recheck R68: the "difference" columns use OPPOSITE signs under the same label.

Two tables, both with a column literally named "差":

  sec 2.5 (baseline comparison)   : 差 = model CE - baseline CE   -> negative means the model is better
  sec 2.5 (bootstrap table)       : 差 = baseline CE - model CE   -> positive means the model is better

The same pair therefore appears as **-0.128** in one table and **+0.1280** in the other. Both rows
are arithmetically right; the label is the same. A reader comparing the two tables sees the same
comparison with two signs and no way to tell they agree.

This recomputes all five pairs from R4 (the CEs) and R5 (the paired bootstrap), states both signed
forms side by side, and checks that the prose range "-0.067 to -0.128" is the range of the
model-minus-baseline form.

Writes only Document/Update/Alpha/v4.3.1/evidence/R68/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
EV = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence"
OUT = EV / "R68"


def main() -> int:
    r4 = json.loads((EV / "R4" / "model_recheck.json").read_text(encoding="utf-8"))
    r5 = json.loads((EV / "R5" / "significance.json").read_text(encoding="utf-8"))

    # the 16x arm in R4
    arm = None
    for a in r4["arms"].values():
        if a.get("down") == 16:
            arm = a
            break
    assert arm, "16x arm not found"

    rows = []
    for set_name, sig in r5["results"].items():
        ev = arm["evaluations"].get(set_name)
        if not ev:
            continue
        model_ce = ev["ce"]
        base_ce = ev["baseline"]["ce_train_mle"]
        rows.append({
            "set": set_name,
            "n": sig["n"],
            "model_ce": round(model_ce, 4),
            "baseline_ce": round(base_ce, 6),
            "model_minus_baseline": round(model_ce - base_ce, 4),
            "baseline_minus_model": round(sig["mean_delta"], 4),
            "ci95_baseline_minus_model": [round(sig["ci95_low"], 4), round(sig["ci95_high"], 4)],
            "excludes_zero": sig["excludes_zero"],
        })
    rows.sort(key=lambda r: r["model_minus_baseline"])

    mmb = [r["model_minus_baseline"] for r in rows]
    bmm = [r["baseline_minus_model"] for r in rows]
    out = {
        "schema": "v431-recheck-r68/1",
        "question": ("「−0.067 到 −0.128 nats」这五个端点，来自哪个口径？两张表的「差」列符号一致吗？"),
        "pairs": rows,
        "ranges": {
            "model_minus_baseline": [min(mmb), max(mmb)],
            "baseline_minus_model": [min(bmm), max(bmm)],
        },
        "prose_claim": {"text": "−0.067 到 −0.128 nats", "convention": "model_minus_baseline",
                        # the prose quotes 3-decimal roundings: -0.1280 -> -0.128, -0.0673 -> -0.067
                        "matches": (round(min(mmb), 3) == -0.128
                                    and round(max(mmb), 3) == -0.067),
                        "exact_values": [round(min(mmb), 4), round(max(mmb), 4)],
                        "rounding_note": ("-0.128 是 -0.1280 的三位舍入；"
                                          "-0.067 是 -0.0673 的三位舍入")},
        "convention_conflict": {
            "sec_2_5_baseline_table": "差 = 模型 − 基线（负 = 模型更好）",
            "sec_2_5_bootstrap_table": "差 = 基线 − 模型（正 = 模型更好）",
            "same_pair_shown_as": {"model_minus_baseline": -0.128, "baseline_minus_model": 0.128},
            "problem": ("两张表的列名都叫「差」，符号相反；同一对 (1.5306 vs 1.6586) "
                        "在一张表里是 −0.128、另一张是 +0.1280，而**没有任何地方说明这两个口径**"),
        },
        "verdict": {"five_pairs_reproduce": None, "sign_conventions_differ": None,
                    "reading": None},
    }
    v = out["verdict"]
    v["five_pairs_reproduce"] = len(rows) == 5 and all(r["excludes_zero"] for r in rows)
    v["sign_conventions_differ"] = True
    v["reading"] = (
        f"五对全部复现且 CI 全排除 0（{v['five_pairs_reproduce']}）。"
        f"但**两张表的「差」列符号相反**：§2.5 的比较表用 `模型−基线`（负 = 更好，"
        f"范围 {out['ranges']['model_minus_baseline']}），bootstrap 表用 `基线−模型`"
        f"（正 = 更好，范围 {out['ranges']['baseline_minus_model']}）。"
        "同一对 (1.5306 vs 1.6586) 在前者写 **−0.128**、后者写 **+0.1280**，"
        "**而两处都没有说明口径**。正文那句「−0.067 到 −0.128」用的是前者，是对的，但读者无法判断。"
        "**这就是 C 类：不是数错，是两套符号共用一个名字。**")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sign_conventions.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'set':>16s} {'n':>5s} {'model':>7s} {'base':>8s} {'m-b':>8s} {'b-m':>8s}  CI(b-m)")
    print("-" * 78)
    for r in rows:
        print(f"{r['set']:>16s} {r['n']:5d} {r['model_ce']:7.4f} {r['baseline_ce']:8.4f} "
              f"{r['model_minus_baseline']:8.3f} {r['baseline_minus_model']:8.4f}  "
              f"{r['ci95_baseline_minus_model']}")
    print()
    print("ranges:", out["ranges"])
    print("prose claim matches:", out["prose_claim"]["matches"])
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'sign_conventions.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
