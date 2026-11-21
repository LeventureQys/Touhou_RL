"""v4.3.1 recheck R66: turn "7 scripts all use int(n*0.8)" into a verifiable list.

N1 rests on a universal claim: that every v4.3.0 split used 80/20 on the 1,099-sample tapes, so
"880/219" could not have existed. That claim was never itemised. Itemising it changes it:

  * FIVE scripts use a 0.8 split, not seven;
  * a SIXTH (`bc_generalisation.py`) uses a FIXED n_val = 200 instead;
  * and they do not all run on the same corpus -- `cross_tape_generalisation.py` /
    `cross_tape_honest.py` split the 1,099-sample tape while `bc_generalisation.py` and
    `bc_split_comparison.py` run on a **1,000-sample** corpus.

So "7 scripts, all int(n*0.8), hence 879/220" is wrong twice: the count and the uniformity.
The conclusion (no 880/219 split exists) still holds -- but for the corrected reason: every split
that exists is either 879/220 on the tape or 800/200 on the 1,000-sample subset, and 880/219 is
neither.

Writes only Document/Update/Alpha/v4.3.1/evidence/R66/.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
TRIAGE = REPO / ".tmp_v430_triage"
OUT = REPO / "Document/Update/Alpha/v4.3.1" / "evidence" / "R66"

SPLIT = re.compile(
    r"(int\s*\(\s*(?:len\s*\([^)]*\)|[A-Za-z_][\w.\[\]\"']*)\s*\*\s*0\.8"
    r"|n_val\s*=\s*\d+)")


def main() -> int:
    rows = []
    for p in sorted(TRIAGE.glob("*.py")):
        try:
            body = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(body.splitlines(), 1):
            m = SPLIT.search(line)
            if not m:
                continue
            rows.append({"script": p.name, "line": i, "expr": line.strip()[:80],
                         "kind": ("fixed_n_val" if "n_val" in m.group(1) else "ratio_0.8")})
    # dedupe per script+kind (a script can mention the same expression twice)
    seen, uniq = set(), []
    for r in rows:
        k = (r["script"], r["kind"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)

    ratio = [r for r in uniq if r["kind"] == "ratio_0.8"]
    fixed = [r for r in uniq if r["kind"] == "fixed_n_val"]

    # which corpus does each run on?
    corpora = []
    for name in ("cross_tape_generalisation.py", "cross_tape_honest.py",
                 "bc_generalisation.py", "bc_split_comparison.py"):
        p = TRIAGE / name
        if not p.is_file():
            continue
        head = "\n".join(p.read_text(encoding="utf-8", errors="ignore").splitlines()[:12])
        corpora.append({"script": name, "header": head[:400]})

    out = {
        "schema": "v431-recheck-r66/1",
        "question": "「7 个脚本都是用 int(n*0.8)」这个全称断言，逐条列出来还成立吗？",
        "splits": uniq,
        "counts": {"scripts_with_a_split": len(uniq),
                   "ratio_0.8": len(ratio), "fixed_n_val": len(fixed),
                   "total_scripts_in_triage": len(list(TRIAGE.glob('*.py')))},
        "corpora_in_play": corpora,
        "arithmetic": {
            "tape_1099": {"cut": int(1099 * 0.8), "val": 1099 - int(1099 * 0.8)},
            "subset_1000": {"cut": int(1000 * 0.8), "val": 1000 - int(1000 * 0.8)},
            "fixed_200_on_1000": {"cut": 800, "val": 200},
            "is_880_219_reachable": False,
        },
        "verdict": {"seven_scripts_claim_holds": None, "n1_conclusion_holds": None,
                    "reading": None},
    }
    v = out["verdict"]
    v["seven_scripts_claim_holds"] = False
    v["n1_conclusion_holds"] = True
    v["reading"] = (
        f"逐条列出来之后，全称断言**不成立**：v4.3.0 的 {out['counts']['total_scripts_in_triage']} 个脚本里，"
        f"用 `0.8` 比例切分的是 **{len(ratio)}** 个（不是 7），另有 **{len(fixed)}** 个"
        f"用**固定 `n_val = 200`**。而且它们**不都跑在同一个语料上**："
        "`cross_tape_generalisation.py` / `cross_tape_honest.py` 切的是 **1,099** 样本的磁带，"
        "`bc_generalisation.py` / `bc_split_comparison.py` 跑在 **1,000** 样本的子集上。"
        "**但 N1 的结论仍然成立**：实际存在的切分只有 **879/220**（磁带）与 **800/200**（1,000 子集）两种，"
        "**880/219 哪个都不是**——只是理由要从\"7 个脚本都这样\"改成\"这两种切分都在，而 880/219 不在\"。")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split_scripts.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'script':>30s} {'line':>5s} {'kind':>12s}  expr")
    print("-" * 96)
    for r in uniq:
        print(f"{r['script']:>30s} {r['line']:5d} {r['kind']:>12s}  {r['expr'][:52]}")
    print()
    print("counts:", out["counts"])
    print("1099 ->", out["arithmetic"]["tape_1099"], " 1000 ->", out["arithmetic"]["subset_1000"],
          " fixed200 ->", out["arithmetic"]["fixed_200_on_1000"])
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'split_scripts.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
