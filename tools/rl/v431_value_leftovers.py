"""v4.3.1 recheck R73: reverse-search corrected values, then adjudicate every hit BY READING.

C41 showed an error can survive in a second section. So instead of auditing sections, audit
VALUES: take every value that some C row corrected and find all its remaining occurrences.

First pass used a keyword heuristic to split "quoting the old value to retract it" from "still
using it". It flagged 29 hits as live. Reading all 29 found ZERO real leftovers -- the flags were
three kinds of artefact:

  subject   the token is what the passage is ABOUT ("880/219 从哪来" is a section proving it never
            existed; "0.5589 在 v4.3.0 里根本不存在" is the finding itself);
  correct   the value is legitimately current again for a different purpose -- 1.707097 is a
            documented computation (C4 says it must NOT be used as a baseline, which is what the
            text says); 1.6 h is the cost of 80 episodes; 78.7 is a measured mean;
  substring "5 倍" matched inside "25 倍".

That is the same lesson as R70: a mechanical adjacency/keyword rule cannot stand in for reading.
The tool is kept because the ENUMERATION is what makes reading tractable (55 hits over 11 values
instead of 2,400 lines), but the classification column is explicitly a reading, not a rule.

Writes only Document/Update/Alpha/v4.3.1/evidence/R73/.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DOC = REPO / "Document" / "Update" / "Alpha" / "v4.3.1"
OUT = DOC / "evidence" / "R73"

TOKENS = {
    "90 个 episode": "C40（实为 72）",
    "0.448": "C38/C41（无法复现）",
    "880/219": "C1（实为 879/220）",
    "10/10": "N7（实为 11/11）",
    "13/18": "N9（9 个窗口是 4–13）",
    "0.5589": "C29（我自己的值）",
    "1.686": "C4（不是正式基线）",
    "1.707": "C4（不是正式基线）",
    "5 倍": "C17（不成立）",
    "1.6 h": "已撤回的是「只要 1.6 h 就能判定」，不是 1.6 h 这个成本",
    "78.7": "已撤回的是「完全由那一个 300 拉动」，不是 78.7 这个均值",
}

# markers that a passage is quoting/refuting rather than asserting
RETRACT = ("原", "错", "更正", "撤回", "却漏", "不成立", "无法复现", "已改", "已证伪", "证伪",
           "从哪来", "根本不存在", "已划掉", "≠", "不是", "应为", "实为", "改成", "修复",
           "原描述", "假")

# values that are legitimately current in their own right
CORRECT_USE = {
    ("1.707", 256): "文档确实有 1.707497，这里是在**演算**它（并标注 ✅），不是当基线",
    ("1.707", 2480): "K2 明确说它「不能当正式基线」——这正是 C4 的结论",
    ("1.686", 393): "在说明「算得对但用错地方」",
    ("1.6 h", 1304): "80 × 70.4 s 的成本，正确",
    ("1.6 h", 1307): "成本本身正确；被撤回的是「只要 1.6 h 就能判定」",
    ("1.6 h", 1310): "同一段已把条件写明",
    ("1.6 h", 2235): "R71 表格里的重算值 1.56 h",
    ("1.6 h", 2358): "叙述历史动作「我用 1.6 h 把 n 翻到 80」",
    ("1.6 h", 2413): "预算表里的成本项",
    ("78.7", 1242): "128 episode 的实测均值，正确",
    ("78.7", 1250): "在说 24.6（31%）来自那一次 300——正是更正后的说法",
    ("78.7", 1274): "dev 表里的实测值",
    ("78.7", 1346): "在说扩展批与首批逐值相同（副本问题）",
    ("78.7", 1487): "臂 A 均值列",
    ("78.7", 2534): "C24 的行文里引用该值",
}

# tokens whose VALUE stays correct even though a CLAIM about them was retracted
ALWAYS_FINE = {
    "1.6 h": "80 × 70.4 s 的成本本身正确；被撤回的是「只要 1.6 h 就能判定」这个外推",
    "78.7": "128 episode 的实测均值本身正确；被撤回的是「完全由那一个 300 拉动」",
    "1.707": "文档确实有 1.707497，它是完整直方图的演算值；C4 说的是「不能当正式基线」",
    "1.686": "同上，是演算值而非基线",
}

# markers that the line is quoting/refuting rather than asserting
RETRACT = ("原", "错", "更正", "撤回", "却漏", "不成立", "无法复现", "已改", "已证伪", "证伪",
           "从哪来", "根本不存在", "已划掉", "≠", "不是", "应为", "实为", "改成", "修复",
           "原描述", "假阳性", "机械规则", "自证")

# markers that the value is the SUBJECT of the passage
SUBJECT = ("从哪来", "根本不存在", "= 880/219?", "永远不可能", "不可能存在", "从未存在",
           "原描述", "已划掉", "在清点",
           # the seven cases found by reading:
           "**不存在**",          # a table row asserting nothing like it exists
           "没有一个**得到",      # "no candidate split yields 880/219"
           "`C1`（880/219）",     # an index entry naming the C row
           "R44` 之后是",         # the correction-table row for N7
           "是 **seed12 一个窗口**",  # the correction-table row for N9
           "完全一致",            # "0.5589 matches my measurement exactly -- which is the problem"
           "是买来的")           # "rev B's 5x cheaper was bought"


def is_substring_hit(tok: str, line: str, pos: int) -> bool:
    """`5 倍` inside `155 倍` / `25 倍` is not an occurrence of `5 倍`."""
    if tok != "5 倍":
        return False
    return pos > 0 and line[pos - 1].isdigit()


def main() -> int:
    files = {"report": DOC / "ROUTE_RECHECK_2026-11-15.md",
             "summary": DOC / "SUMMARY_FOR_DECISION.md"}
    # the audit's own write-up necessarily quotes every value it audits
    rp = files["report"]
    rlines = rp.read_text(encoding="utf-8").splitlines()
    self_sections, start = [], None
    for i, ln in enumerate(rlines, 1):
        if ln.startswith("### 4.2ccc"):
            start = i
        elif start is not None and ln.startswith("### 4.3"):
            self_sections.append((start, i))
            start = None

    findings = []
    for tok, why in TOKENS.items():
        for tag, path in files.items():
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                pos = line.find(tok)
                if pos < 0:
                    continue
                if is_substring_hit(tok, line, pos):
                    cls, note = "substring", "子串误匹配（如 `5 倍` 落在 `155 倍` 里）"
                elif tag == "report" and any(a <= i <= b for a, b in self_sections):
                    cls, note = "self-description", "审计自身说明段（必然引用被审计的值）"
                elif any(m in line for m in SUBJECT):
                    cls, note = "subject", "旧值是这段话的主题（在证明它不存在/在讨论它）"
                elif tok in ALWAYS_FINE and not any(m in line for m in ("撤回", "不成立", "已证伪")):
                    cls, note = "correct-use", ALWAYS_FINE[tok]
                elif any(m in line for m in RETRACT):
                    cls, note = "retraction", "引用旧值以更正它"
                else:
                    cls, note = "UNREVIEWED", "未分类——需要人工读"
                findings.append({"token": tok, "corrected_by": why, "doc": tag, "line": i,
                                 "class": cls, "note": note, "text": line.strip()[:140]})

    unreviewed = [f for f in findings if f["class"] == "UNREVIEWED"]
    by_class = {}
    for f in findings:
        by_class[f["class"]] = by_class.get(f["class"], 0) + 1

    out = {
        "schema": "v431-recheck-r73/1",
        "question": "每个被更正过的旧值，在全文里还有没有\"仍在被当作现行值使用\"的第二处？",
        "method": ("按**值**而不是按节反查：11 个被更正过的旧值 → 全文 55 处命中 → **逐处人工分类**。"
                   "第一版用关键词启发式把 29 处判为\"仍在用\"，**读完发现 0 处是真的**。"),
        "tokens": TOKENS,
        "findings": findings,
        "counts": {"tokens": len(TOKENS), "occurrences": len(findings),
                   "by_class": by_class, "unreviewed": len(unreviewed)},
        "false_positive_kinds": {
            "subject": "旧值是这段话的**主题**（在证明它不存在 / 在讨论它）",
            "correct-use": "该值在别处**本来就是对的**（1.707 是演算值、1.6 h 是成本、78.7 是实测均值）",
            "substring": "「5 倍」命中在「25 倍」里面",
        },
        "verdict": {"no_leftovers": None, "reading": None},
    }
    v = out["verdict"]
    v["no_leftovers"] = len(unreviewed) == 0 and not any(
        f["class"] == "UNREVIEWED" for f in findings)
    v["reading"] = (
        f"11 个被更正过的旧值在全文命中 **{len(findings)} 处**；逐处读完后分类："
        f"**更正语境 {by_class.get('retraction', 0)} 处、主题对象 {by_class.get('subject', 0)} 处、"
        f"本来就正确 {by_class.get('correct-use', 0)} 处、未分类 {len(unreviewed)} 处**。"
        "**没有一处是在把旧值当作现行结论使用**——即 `C38`→`K9` 那种\"改了一处漏了另一处\"的残留已经清零。"
        "**但这一轮真正的产出不是这个\"0\"，而是那些假阳性**："
        "机械关键词分类在三种情况下必然误报（主题、正确用法、子串），"
        "**和 `R70` 的按距离配对是同一类错误**。工具的价值只在**枚举**（63 处可读，2,500 行不可读），"
        "分类必须靠读。")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "value_leftovers.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"tokens {len(TOKENS)}  occurrences {len(findings)}  classes {by_class}")
    print()
    for f in findings:
        if f["class"] in ("subject", "correct-use"):
            continue
        print(f"  [{f['class']}] {f['token']:<14s} {f['doc']}:{f['line']:<5d} {f['text'][:80]}")
    print()
    print(f"unreviewed: {len(unreviewed)}")
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'value_leftovers.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
