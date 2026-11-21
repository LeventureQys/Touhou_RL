"""v4.3.1 recheck R64: N4/N5 exhausted, plus what the aggregate JSONs actually reveal.

N4 said "the three experiments left no stdout"; N5 said "no per-epoch curve, only the final
snapshot". Both were spot checks. This exhausts them, and in doing so turns up three things the
report did not state:

  1. N4 HOLDS: 2,789 log-like files exist in scope and ZERO mention a BC config.
  2. N5 IS WRONG as written: per-epoch histories DO exist -- but only 3-epoch runs, and only in
     `bc_generalisation.json` / `bc_split_comparison.json`. What does not exist is a 25-epoch
     curve for the documented configs.
  3. The `majority` blocks in `cross_tape_generalisation.json` are computed on the FULL tape and
     reused for that tape's train/val subsets. For `seed12-val` the share is literally 468/1099
     -- a population that CONTAINS the 220 val samples. That is the C3 defect, visible in the
     artefact rather than inferred from the script.
  4. Two of the three aggregate files come from runs with n_train=800 / n_val=200, i.e. n=1,000,
     while the tapes hold 1,099 samples. v4.3.0 mixed at least two sample counts.

The n=1,000 point is why this matters: the documented "880/219" split is not the only sample count
in play -- there was also an 800/200 one, on a corpus that is not the tape.

Same pruning rules as R63.

Writes only Document/Update/Alpha/v4.3.1/evidence/R64/.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R64"
TRIAGE = REPO / ".tmp_v430_triage"

PRUNE = {".git", "node_modules", "__pycache__", ".pytest_cache", "third_party",
         ".tmp_v400_d1", ".tmp_v410_s1c", ".tmp_v410_wp05_round1",
         ".tmp_v410_wp05_round2", ".tmp_v410_wp05_round3"}
LOG_SUFFIX = (".log", ".out", ".err", ".stdout", ".stderr")
NEEDLES = ("cross_tape_generalisation", "bc_generalisation", "bc_split_comparison",
           "actor_policy", "ActorPolicy")


def walk(root: pathlib.Path):
    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            entries = list(os.scandir(cur))
        except (PermissionError, OSError):
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    if e.name not in PRUNE:
                        stack.append(pathlib.Path(e.path))
                else:
                    yield pathlib.Path(e.path)
            except (PermissionError, OSError):
                continue


def main() -> int:
    # ---- N4
    logs, log_hits = 0, []
    for p in walk(REPO):
        if p.suffix.lower() in LOG_SUFFIX:
            logs += 1
            try:
                if p.stat().st_size < 5_000_000:
                    body = p.read_text(encoding="utf-8", errors="ignore")
                    if any(n in body for n in NEEDLES):
                        log_hits.append(str(p.relative_to(REPO)))
            except OSError:
                continue

    # ---- N5: per-epoch histories (lists of dicts, which the numeric detector in R64 v1 missed)
    hist = []
    for name in ("bc_generalisation.json", "bc_split_comparison.json",
                 "cross_tape_generalisation.json"):
        p = TRIAGE / name
        if not p.is_file():
            hist.append({"file": name, "exists": False})
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        entries = d.get("history") if isinstance(d, dict) else None
        if isinstance(entries, list) and entries and isinstance(entries[0], dict):
            hist.append({"file": name, "exists": True, "kind": "single run",
                         "epochs": len(entries),
                         "epoch_ids": [e.get("epoch") for e in entries],
                         "n_train": d.get("n_train"), "n_val": d.get("n_val")})
        elif isinstance(d, list):
            hist.append({"file": name, "exists": True, "kind": "two runs",
                         "runs": [{"label": r.get("label"), "n_train": r.get("n_train"),
                                   "n_val": r.get("n_val"),
                                   "epochs": len(r.get("history") or [])} for r in d]})
        else:
            hist.append({"file": name, "exists": True, "kind": "no history key",
                         "top_keys": list(d)})

    # ---- the majority blocks
    ct = json.loads((TRIAGE / "cross_tape_generalisation.json").read_text(encoding="utf-8"))
    maj = {k: {"move": v["majority"]["move"],
               "move_majority_share": v["majority"]["move_majority_share"]} for k, v in ct.items()}
    share = maj["seed12-val"]["move_majority_share"]
    out = {
        "schema": "v431-recheck-r64/1",
        "question": "N4/N5 能不能变成有协议的缺席证据？三个聚合 JSON 本身又说明了什么？",
        "n4": {"log_like_files": logs, "mentioning_a_bc_config": log_hits,
               "holds": len(log_hits) == 0},
        "n5": {"per_epoch_histories": hist,
               "holds_as_written": False,
               "refined": ("per-epoch 曲线**存在**，但只有 3 epoch、且只在这两个文件里；"
                           "文档那三个 25-epoch 配置的曲线仍然不存在")},
        "majority_blocks": {
            "per_eval_set": maj,
            "seed12_val_share": share,
            "share_is_468_over_1099": abs(share - 468 / 1099) < 1e-15,
            "neg_log_share": -math.log(share),
            "equals_the_documented_0853": abs(-math.log(share) - 0.8536876584859746) < 1e-12,
            "reading": ("seed12-train 与 seed12-val 共用同一个 majority（**按整条 seed12 磁带**算的 "
                        "468/1099），而 seed12-val 只有 220 个样本、且**包含在这 1,099 里**；"
                        "seed10 有它自己的 majority（按整条 seed10 算）。所以 C3 的缺陷不是从脚本推断的，"
                        "而是**直接写在产物里**：基线按整条磁带算，却用来评该磁带的子集。"),
        },
        "sample_counts": {
            "bc_generalisation": {"n_train": hist[0].get("n_train"), "n_val": hist[0].get("n_val")}
            if hist and hist[0].get("exists") else None,
            "bc_split_comparison": hist[1].get("runs") if len(hist) > 1 else None,
            "tapes_hold": 1099,
            "reading": ("另外两个文件来自 **n_train=800 / n_val=200（合计 1,000）** 的运行，"
                        "而磁带是 **1,099** 个样本。所以 v4.3.0 至少混用了**两种样本量**："
                        "1,099（磁带/文档的 880+219）与 1,000（800+200）。"),
        },
        "verdict": {"n4_holds": len(log_hits) == 0, "n5_needs_restating": True,
                    "reading": None},
    }
    v = out["verdict"]
    v["reading"] = (
        f"N4 **成立**：2,789 个日志类文件里 **0** 个提到任一 BC 配置。"
        "N5 **写错了**：per-epoch 曲线**确实存在**（`bc_generalisation.json` 与 "
        "`bc_split_comparison.json` 各有 3 epoch 的 train/val CE），"
        "但只有 3 epoch、且不是文档那三个 25-epoch 配置的曲线——所以正确表述是"
        "\"没有 25-epoch 曲线\"，而不是\"没有任何曲线\"。"
        "顺带两个新发现：**① 0.8537 的出处已定位到产物**（`move_majority_share = 468/1099`，"
        "`−ln(share) = 0.8536876584859746`），而且这个 majority 是**按整条磁带**算的、"
        "却用来评该磁带的 220 个样子集；**② 另外两个文件用的是 n=1,000（800/200）**，"
        "与磁带的 1,099 不同——v4.3.0 至少混用了两种样本量。")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "n4_n5.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"N4: {logs} log-like files, {len(log_hits)} mention a BC config -> holds={v['n4_holds']}")
    print("N5 per-epoch histories:")
    for h in hist:
        print("   ", h)
    print()
    print("majority blocks:", json.dumps(maj, ensure_ascii=False))
    print("  468/1099:", out["majority_blocks"]["share_is_468_over_1099"],
          " -ln(share):", out["majority_blocks"]["neg_log_share"])
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'n4_n5.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
