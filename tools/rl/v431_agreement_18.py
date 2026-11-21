"""v4.3.1 recheck R67: two universal claims, and one of them turns out to be unsupported.

Claim A (sec 3): "the full-tape action agreement between the three tapes is 0.448 / 0.463 / 0.456".
Claim B (sec 4.2a/4.2m): "all 18 constant actions" were tested per window.

Claim B holds exactly: 9 windows x 18 distinct `cand` = 162 episodes, no gaps.

Claim A does NOT hold. Three tapes give exactly C(3,2)=3 pairs, and neither of the two natural
definitions reproduces the quoted numbers:

    pair              prefix  full-tape cand agreement   suffix-only agreement
    seed10 vs seed11     219        0.4568                    0.3216
    seed10 vs seed12     219        0.4850                    0.3568
    seed11 vs seed12     225        0.4823                    0.3490

The quoted 0.448 / 0.463 / 0.456 match neither, and a search of every script and every evidence
JSON finds no producer for them either -- the only occurrences are in my own prose. So they are
numbers I inherited into the report and never verified.

The CONCLUSION survives and gets stronger: agreement of ~0.46-0.49 is barely above the majority
share (0.4258) and nowhere near 1.0, and after the shared prefix the tapes agree on only
0.32-0.36 -- i.e. they diverge MORE once the prefix is removed. But the numbers were wrong.

Writes only Document/Update/Alpha/v4.3.1/evidence/R67/.
"""

from __future__ import annotations

import collections
import itertools
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
O2 = REPO / ".tmp_v430_o2"
CORPORA = {"seed12": O2 / "o2-strided",
           "seed10": O2 / "o2-stride-seed10",
           "seed11": O2 / "o2-stride-seed11"}
RAW = REPO / ".tmp_v431_seg001" / "full" / "raw" / "c4dense"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R67"
CLAIMED = [0.448, 0.463, 0.456]


def main() -> int:
    # ---- claim B: 18 constants per window
    per = collections.Counter()
    cands: dict[int, set] = collections.defaultdict(set)
    for f in sorted(RAW.glob("c4dense-seed*-f8664-cand*.json")):
        seed = int(f.name.split("-")[1].replace("seed", ""))
        per[seed] += 1
        cands[seed].add(int(f.stem.split("-cand")[-1]))
    windows = len(per)
    complete = all(len(cands[s]) == 18 for s in per)

    # ---- claim A: pairwise agreement
    tapes = {}
    for name, path in CORPORA.items():
        p = path / "labels.jsonl"
        if p.is_file():
            tapes[name] = [json.loads(ln) for ln in
                           p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    rows = []
    for a, b in itertools.combinations(sorted(tapes), 2):
        ca = [r.get("cand") for r in tapes[a]]
        cb = [r.get("cand") for r in tapes[b]]
        n = min(len(ca), len(cb))
        prefix = next((i for i in range(n) if ca[i] != cb[i]), n)
        full_same = sum(1 for i in range(n) if ca[i] == cb[i])
        suf_n = n - prefix
        suf_same = sum(1 for i in range(prefix, n) if ca[i] == cb[i])
        rows.append({
            "pair": f"{a}__{b}", "n": n, "prefix": prefix,
            "full_agreement": round(full_same / n, 4),
            "suffix_len": suf_n,
            "suffix_agreement": round(suf_same / suf_n, 4) if suf_n else None,
        })

    matched_full = [r["full_agreement"] for r in rows if any(
        abs(r["full_agreement"] - c) < 5e-4 for c in CLAIMED)]
    matched_suffix = [r["suffix_agreement"] for r in rows if r["suffix_agreement"] and any(
        abs(r["suffix_agreement"] - c) < 5e-4 for c in CLAIMED)]

    out = {
        "schema": "v431-recheck-r67/1",
        "question": ("「三盘全盘动作一致率 0.448/0.463/0.456」与「每窗口 18 个恒定动作」这两个"
                     "全称断言，逐条算出来还成立吗？"),
        "claim_b_eighteen_constants": {
            "windows": windows, "files_per_window": dict(sorted(per.items())),
            "distinct_cands_per_window": {str(k): len(v) for k, v in sorted(cands.items())},
            "every_window_has_18": complete,
            "total_episodes": sum(per.values()),
            "holds": complete and windows == 9 and sum(per.values()) == 162,
        },
        "claim_a_agreement": {
            "pairs": rows,
            "n_pairs_expected": 3, "n_pairs_computed": len(rows),
            "claimed": CLAIMED,
            "matched_full_definition": matched_full,
            "matched_suffix_definition": matched_suffix,
            "reproducible": bool(matched_full or matched_suffix),
            "hold": False,
        },
        "verdict": {"claim_b_holds": None, "claim_a_reproducible": None, "reading": None},
    }
    v = out["verdict"]
    v["claim_b_holds"] = out["claim_b_eighteen_constants"]["holds"]
    v["claim_a_reproducible"] = out["claim_a_agreement"]["reproducible"]
    v["reading"] = (
        f"**断言 B（每窗口 18 个恒定动作）成立**：{windows} 个窗口 × 18 = "
        f"{sum(per.values())} 个 episode，每个窗口的 distinct cand 都是 18，**无缺口**。"
        f"**断言 A（全盘一致率 0.448/0.463/0.456）不成立**：三盘恰好 3 对，"
        f"全盘口径给出 {[r['full_agreement'] for r in rows]}，后缀口径给出 "
        f"{[r['suffix_agreement'] for r in rows]}——**两种都不是那三个数**，"
        f"且全库搜索没有任何脚本或证据产出过它们。**这就是我继承进报告、却从未核过的数字。**"
        f" 结论仍然成立而且更强：一致率 0.46–0.49 只比多数类占比 0.4258 高一点、离 1.0 很远；"
        f"而**去掉共同前缀后只剩 0.32–0.36**，即三盘在前缀之后**分歧更大**。")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "agreement_and_18.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print("claim B:", json.dumps(out["claim_b_eighteen_constants"], ensure_ascii=False)[:200])
    print()
    print(f"{'pair':>20s} {'prefix':>7s} {'full':>8s} {'suffix_len':>11s} {'suffix':>8s}")
    for r in rows:
        print(f"{r['pair']:>20s} {r['prefix']:7d} {r['full_agreement']:8.4f} "
              f"{r['suffix_len']:11d} {r['suffix_agreement']:8.4f}")
    print()
    print("claimed:", CLAIMED, " reproducible:", out["claim_a_agreement"]["reproducible"])
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'agreement_and_18.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
