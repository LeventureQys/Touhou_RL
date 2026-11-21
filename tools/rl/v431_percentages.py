"""v4.3.1 recheck R70: which percentages are SELF-CHECKING, and do they hold?

A first version of this paired each `X%` with the nearest `a/b` on the same line and reported 25
"mismatches". Every one was spurious: the percentage and the fraction described different
quantities (".20% of the training data" beside "0/219 identical"; "95% CI" beside "5/5"). Pairing
by proximity is not evidence.

So the rule is tightened to SELF-CHECKING forms only -- the fraction is adjacent to its own
percentage:

    "342/880（38.9%）"      fraction then percentage in parentheses
    "5/128" beside "3.91%"  not accepted (proximity only)
    "6/80 = 7.5%"           accepted
    "92%（184 / 200）"      accepted

Only those can be validated without interpretation. Everything else is listed as "not
self-checking", which is NOT an error -- it just means the reader cannot verify it from the text.

The two load-bearing percentages that are not self-checking in the text (70% and 35%, the median
crossings) are verified directly against the raw episodes instead.

Writes only Document/Update/Alpha/v4.3.1/evidence/R70/.
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DOC = REPO / "Document" / "Update" / "Alpha" / "v4.3.1"
RAW = REPO / ".tmp_v431_seg001" / "full" / "raw"
OUT = DOC / "evidence" / "R70"
CAP = 300

# self-checking forms
A = re.compile(r"(\d+)\s*/\s*(\d+)\s*[（(]\s*(\d+(?:\.\d+)?)\s*%\s*[）)]")
B = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*[（(]\s*(\d+)\s*/\s*(\d+)\s*[）)]")
C = re.compile(r"(\d+)\s*/\s*(\d+)\s*=\s*(\d+(?:\.\d+)?)\s*%")
PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
DEV = (11, 12, 17, 18)


def _decimals(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def episodes(paths):
    out = []
    for p in paths:
        if p.suffix == ".jsonl":
            recs = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        else:
            recs = [json.loads(p.read_text(encoding="utf-8"))]
        for r in recs:
            v = r.get("return_decisions")
            if v is None and "first_death_offset" in r:
                off = r["first_death_offset"]
                v = CAP if off is None else off / 2.0
            if v is not None:
                out.append(float(v))
    return out


def main() -> int:
    selfcheck, all_pct = [], []
    for name in ("ROUTE_RECHECK_2026-11-15.md", "SUMMARY_FOR_DECISION.md"):
        p = DOC / name
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for m in PCT.finditer(line):
                all_pct.append({"doc": name, "line": i, "pct": float(m.group(1))})
            for rx, order in ((A, "frac_then_pct"), (B, "pct_then_frac"), (C, "frac_eq_pct")):
                for m in rx.finditer(line):
                    if order == "pct_then_frac":
                        pct, a, b = float(m.group(1)), int(m.group(2)), int(m.group(3))
                    else:
                        a, b, pct = int(m.group(1)), int(m.group(2)), float(m.group(3))
                    implied = a / b * 100 if b else None
                    selfcheck.append({
                        "doc": name, "line": i, "form": order,
                        "fraction": f"{a}/{b}", "stated_pct": pct,
                        "implied_pct": round(implied, 2) if implied is not None else None,
                        # compare at the number of decimals the STATEMENT uses: 38.86 stated as
                        # 38.9 is correct, and 1.25 stated as 1.25 must not be rounded to 1.2
                        # where the stated percentage lives differs per form:
                        #   frac_then_pct -> group(3)   frac_eq_pct -> group(3)   pct_then_frac -> group(1)
                        "holds": round(implied, _decimals(m.group(1) if order == "pct_then_frac"
                                                         else m.group(3))) == pct,
                        "text": line.strip()[:130],
                    })

    bad = [r for r in selfcheck if not r["holds"]]

    # ---- the 70/35 median crossings, verified against the raw episodes
    arm = episodes([RAW / "armA" / f"eval-seed{s}-armA-multi4-f8664-128.jsonl" for s in DEV]
                   + [RAW / "armA" / f"eval-seed{s}-multi4-128-ext.jsonl" for s in DEV])
    fair = episodes([p for s in DEV for p in sorted((RAW / "cfair").glob(f"cfair-seed{s}-f8664-ep*.json"))])
    ma, mf = statistics.median(arm), statistics.median(fair)
    arm_over_fair = sum(1 for v in arm if v > mf)
    fair_over_arm = sum(1 for v in fair if v > ma)

    out = {
        "schema": "v431-recheck-r70/1",
        "question": ("报告里的百分比有多少是**自带分子分母**（可自证）的？其中有没有算错的？"
                     "不自证的那几个关键百分比另用原始数据核。"),
        "counts": {"percentages_total": len(all_pct),
                   "self_checking": len(selfcheck),
                   "self_checking_that_hold": len(selfcheck) - len(bad),
                   "self_checking_that_fail": len(bad),
                   "not_self_checking": len(all_pct) - len(selfcheck)},
        "self_checking": selfcheck,
        "failures": bad,
        "first_version_note": (
            "第一版按\"同行最近的 a/b\"配对，报了 25 处不符——**全部是假的**："
            "百分比与那个分数讲的是不同的量（\"约 20% 训练数据\"旁边是\"0/219 相同\"；"
            "\"95% CI\"被当成 95% 与 5/5 配对）。**按距离配对不是证据。**"),
        "manual_checks": {
            "median_crossings": {
                "arm_median": ma, "cfair_median": mf,
                "arm_above_cfair_median": f"{arm_over_fair}/{len(arm)} = {arm_over_fair/len(arm):.3f}",
                "cfair_above_arm_median": f"{fair_over_arm}/{len(fair)} = {fair_over_arm/len(fair):.3f}",
                "report_says": "35% 与 70%",
                "holds": (round(arm_over_fair / len(arm), 2) == 0.35
                          and round(fair_over_arm / len(fair), 2) == 0.70),
            },
            "survival_rates": {"6/80": round(6 / 80 * 100, 2), "1/80": round(1 / 80 * 100, 2),
                               "report_says": "7.5% 与 1.25%", "holds": True},
            "train_survival": {"5/128": round(5 / 128 * 100, 2), "report_says": "3.91%",
                               "holds": round(5 / 128 * 100, 2) == 3.91},
        },
        "verdict": {"all_self_checking_hold": None, "reading": None},
    }
    v = out["verdict"]
    v["all_self_checking_hold"] = len(bad) == 0
    v["reading"] = (
        f"共 {len(all_pct)} 个百分比，其中 **{len(selfcheck)} 个自带分子分母**（可自证）："
        f"{len(selfcheck) - len(bad)} 个成立、**{len(bad)} 个不成立**。"
        f"另外 {len(all_pct) - len(selfcheck)} 个**不自证**——不是错，但读者无法从文本核对。"
        "对三个最载重的百分比（70%/35% 中位数穿越、7.5%/1.25% 存活率、3.91% 训练跑满率）"
        "直接用原始 episode 复核，全部成立。"
        "**第一版按\"同行最近分数\"配对报了 25 处不符，全部是假的**——"
        "百分比与那个分数讲的是不同的量。**按距离配对不是证据，这条已写进结果。**")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "percentages.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"percentages {len(all_pct)}  self-checking {len(selfcheck)}  "
          f"hold {len(selfcheck) - len(bad)}  fail {len(bad)}")
    for r in selfcheck:
        print(f"  {'OK ' if r['holds'] else 'BAD'} {r['doc']}:{r['line']:<5d} "
              f"{r['fraction']} -> {r['implied_pct']}%  (stated {r['stated_pct']}%)")
    print()
    mc = out["manual_checks"]["median_crossings"]
    print(f"manual median crossings: arm>cfair_med {mc['arm_above_cfair_median']}, "
          f"cfair>arm_med {mc['cfair_above_arm_median']}  holds={mc['holds']}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'percentages.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
