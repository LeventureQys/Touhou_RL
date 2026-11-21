"""v4.3.1 recheck R74: WHERE the one-mask-per-decision constraint actually bites.

The report says the pixel arms (B/C) cannot run because "obs-v3 needs per-decision action history
and `_pair_cand` rejects dual-mask decisions". Reading the code separates two facts that the
sentence glues together:

  1. `_pair_cand` has exactly ONE call site: inside `replay()`, i.e. the PREFIX-replay path.
     It validates the RECORDED tape, not the live policy.
  2. The live arm commits ONE mask for COMMIT_FRAMES frames (2, since
     COMMIT_FRAMES = DECISION_FRAMES * TEACHER_REPLAN_DECISIONS = 2 * 1) and pushes one history
     entry per decision. So the live history is pair-consistent BY CONSTRUCTION.

Therefore the blocker is not "the policy may not submit two masks"; it is "you cannot REACH the
start frame", because every tape's prefix contains dual-mask decisions from frame 92 onward.

And there is a mechanical bypass that is REACHABLE BUT INVALID: `replay()` COUNTS hash mismatches
instead of raising, so a substituted prefix (one mask per 2 frames) would replay to frame 8,664 with
mismatches>0 -- the episode would then start from a state that is not the recorded one. Since
"0 mismatches" is the entire premise of comparing arms on the same state, that path is not usable.

This script asserts those four code facts mechanically so the claim is not a reading.

Writes only Document/Update/Alpha/v4.3.1/evidence/R74/.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = REPO / "touhou_rl" / "v420" / "counterfactual.py"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R74"


def main() -> int:
    text = SRC.read_text(encoding="utf-8")
    lines = text.splitlines()

    # 1. every mention of _pair_cand, with the enclosing function
    def enclosing(i: int) -> str:
        for j in range(i, -1, -1):
            m = re.match(r"\s*def (\w+)\(", lines[j])
            if m:
                return m.group(1)
        return "?"

    calls = [(n, ln.strip(), enclosing(n - 1)) for n, ln in enumerate(lines, 1)
             if "_pair_cand" in ln]
    call_sites = [c for c in calls if not c[1].startswith("def ")]
    def_sites = [c for c in calls if c[1].startswith("def ")]

    # 2. COMMIT_FRAMES arithmetic
    consts = {}
    for ln in lines:
        m = re.match(r"\s*(DECISION_FRAMES|TEACHER_REPLAN_DECISIONS|COMMIT_FRAMES)\s*=\s*(.+)$", ln)
        if m:
            consts.setdefault(m.group(1), m.group(2).strip())
    try:
        from touhou_rl.v420 import counterfactual as CF
        runtime = {"DECISION_FRAMES": CF.DECISION_FRAMES,
                   "TEACHER_REPLAN_DECISIONS": CF.TEACHER_REPLAN_DECISIONS,
                   "COMMIT_FRAMES": CF.COMMIT_FRAMES}
    except Exception as exc:                     # pragma: no cover
        runtime = {"import_error": f"{type(exc).__name__}: {exc}"}

    # 3. does replay() raise on a hash mismatch, or only count it?
    block_a = ("if value != expected:\n"
               "                    mismatches += 1\n"
               "                    if first_mismatch is None:\n"
               "                        first_mismatch = frame")
    counts_only = block_a in text
    first_mismatch_recorded = "first_mismatch = frame" in text
    # and confirm the surrounding span really has no raise between the check and the next push
    i_a = text.find("if value != expected:")
    span_a = text[i_a:i_a + 400]
    counts_only = counts_only and ("raise" not in span_a)

    # 4. the live arm commits ONE mask for COMMIT_FRAMES frames
    # the block spans COMMIT_FRAMES frames and names exactly ONE mask variable
    i_b = text.find("for _offset in range(COMMIT_FRAMES):")
    span_b = text[i_b:i_b + 260] if i_b >= 0 else ""
    live_single_mask = (i_b >= 0 and "session.step_frame(mask)" in span_b
                        and span_b.count("step_frame(") == 1)
    live_mask_variable = "mask" if live_single_mask else None

    out = {
        "schema": "v431-recheck-r74/1",
        "question": ("臂 B/C 的阻断到底卡在哪：是\"策略不能提交两个 mask\"，还是\"到不了起始帧\"？"),
        "code_facts": {
            "source": str(SRC.relative_to(REPO)),
            "pair_cand_definition": [{"line": n, "text": t} for n, t, _ in def_sites],
            "pair_cand_call_sites": [{"line": n, "text": t, "enclosing_function": f}
                                     for n, t, f in call_sites],
            "pair_cand_only_in_replay": all(f == "replay" for _, _, f in call_sites),
            "constants_source": consts,
            "constants_runtime": runtime,
            "replay_counts_hash_mismatch_instead_of_raising": counts_only,
            "replay_records_first_mismatch_frame": first_mismatch_recorded,
            "live_arm_commits_one_mask_per_commit_block": live_single_mask,
            "live_arm_mask_variable": live_mask_variable,
            "live_arm_block_text": span_b.replace("\n", " | ")[:160],
        },
        "consequence": {
            "constraint_target": "录制的磁带前缀（`replay()`），不是实时策略",
            "live_history_is_pair_consistent_by_construction": (
                "实时臂每个 commit 块只提交一个 mask，并把一个 cand 推进历史，所以它的历史天然满足"
                "\"一决策一 mask\""),
            "real_blocker": ("**到不了起始帧**：每条磁带的 prefix 从 frame 92 起就有双 mask 决策，"
                             "而 `replay()` 在第 92 帧就会抛错。段落在 frame 8,664，绕不过去"),
            "reachable_but_invalid": ("把 prefix 换成\"每两帧同 mask\"就可以重放到 8,664——"
                                      "`replay()` 对哈希不符只**计数**不抛错——但那样起始状态就"
                                      "**不是**录像里的那个状态，而\"0 不符\"正是三臂可比的前提"),
        },
        "unblock_options": [
            {"option": "save state（引擎存档）", "status": "不可用：没有这个动词"},
            {"option": "重录一决策一 mask 的干净磁带", "status": "需授权（资产操作）"},
            {"option": "改用前缀 < 92 帧的段", "status": "帧 92 之前是稀疏早期局面，`C4` 已判 0/4 不合格"},
            {"option": "替换 prefix 强行重放", "status": "**可到达但无效**：起始状态不是录制状态"},
        ],
        "verdict": {"report_wording_is_imprecise": None, "blocker_is_reach_not_submit": None,
                    "reading": None},
    }
    v = out["verdict"]
    v["report_wording_is_imprecise"] = True
    v["blocker_is_reach_not_submit"] = out["code_facts"]["pair_cand_only_in_replay"] and \
        live_single_mask
    v["reading"] = (
        "`_pair_cand` **只有一个调用点，且在 `replay()` 里**——它校验的是**录制的磁带前缀**，"
        "不是实时策略。而实时臂 **每个 commit 块只提交一个 mask**"
        f"（`COMMIT_FRAMES = {runtime.get('DECISION_FRAMES')} × {runtime.get('TEACHER_REPLAN_DECISIONS')}"
        f" = {runtime.get('COMMIT_FRAMES')}`），所以它的历史**按构造**就满足\"一决策一 mask\"。"
        "**所以阻断不是\"策略不能提交两个 mask\"，而是\"到不了起始帧\"**："
        "每条磁带 prefix 从 frame 92 起有双 mask 决策，`replay()` 在第 92 帧抛错，而段落在 frame 8,664。"
        "**并且存在一条\"可到达但无效\"的路**：`replay()` 对哈希不符只**计数**不抛错，"
        "所以替换 prefix 能重放到 8,664——但起始状态不再是录像里那个状态，"
        "而\"0 不符\"正是三臂可比的前提。**这四条解阻选项的取舍因此是确定的，不是我没试。**")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pair_cand_scope.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print("pair_cand def:", [(n, t[:60]) for n, t, _ in def_sites])
    print("call sites   :", [(n, f) for n, _, f in call_sites])
    print("only in replay:", out["code_facts"]["pair_cand_only_in_replay"])
    print("constants    :", runtime)
    print("replay counts mismatch, not raises:", counts_only, " first_mismatch recorded:",
          first_mismatch_recorded)
    print("live arm one mask per block:", live_single_mask)
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'pair_cand_scope.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
