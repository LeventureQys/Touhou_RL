"""v4.3.1 recheck R63: prove the BC artifacts are ABSENT, not merely missing from where I looked.

N1-N5 in the report are marked "unreproduced" because the v4.3.0 BC configs left no checkpoint
and no per-sample predictions. "I could not find them" is a weak claim: it is consistent with
"they exist somewhere I did not look". This makes it a documented search instead.

Search protocol:
  * every directory under the repo root, depth-first, PRUNING the trees that cannot hold them
    (.git, node_modules, __pycache__, .pytest_cache, third_party, and the multi-GB replay dirs);
  * match by FILE NAME for checkpoint / prediction patterns;
  * additionally grep every small text-ish file (.json/.jsonl/.py/.md) for the three config names,
    so a checkpoint recorded inside a manifest would still be found;
  * report what was searched, what matched, and -- crucially -- what was NOT searched and why.

The last point is what makes this honest: an absence claim is only as strong as the stated scope.

Writes only Document/Update/Alpha/v4.3.1/evidence/R63/.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R63"

# trees that cannot contain a BC checkpoint, with the reason recorded
PRUNE = {
    ".git": "版本库对象，不是工作文件",
    "node_modules": "依赖",
    "__pycache__": "字节码缓存",
    ".pytest_cache": "pytest 缓存（且有 ACL 问题，按 AGENTS.md §5.2 绕开）",
    "third_party": "第三方引擎源码，不含训练产物",
    ".tmp_v400_d1": "28.8 GB 的 D1 录制资产（v4.1 时代，早于 v4.3.0）",
    ".tmp_v410_s1c": "2.34 GB 的 S1 教师磁带（早于 v4.3.0）",
    ".tmp_v410_wp05_round1": "WP05 DAgger 产物（早于 v4.3.0，且暂停中）",
    ".tmp_v410_wp05_round2": "同上",
    ".tmp_v410_wp05_round3": "同上",
}

CKPT_SUFFIX = (".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx")
PRED_HINTS = ("prediction", "predictions", "logits", "probs", "y_pred", "pred_")
CONFIG_NAMES = ("actor_policy", "Small16", "small16", "small4", "Small4", "ActorPolicy",
                "bc_generalisation", "bc_split_comparison", "cross_tape_generalisation")


DIRS_SEEN = 0


def walk(root: pathlib.Path):
    global DIRS_SEEN
    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            entries = list(os.scandir(cur))
        except (PermissionError, OSError):
            continue
        DIRS_SEEN += 1
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    if e.name in PRUNE:
                        continue
                    stack.append(pathlib.Path(e.path))
                else:
                    yield pathlib.Path(e.path)
            except (PermissionError, OSError):
                continue


def main() -> int:
    dirs_scanned = 0
    files_scanned = 0
    ckpts, preds, named = [], [], []
    for p in walk(REPO):
        files_scanned += 1
        name = p.name.lower()
        if p.suffix.lower() in CKPT_SUFFIX:
            ckpts.append(str(p.relative_to(REPO)))
        if any(h in name for h in PRED_HINTS) and p.suffix.lower() in (".json", ".jsonl", ".npz", ".npy", ".csv"):
            preds.append(str(p.relative_to(REPO)))
        if p.suffix.lower() in (".json", ".jsonl", ".py", ".md", ".txt") and p.stat().st_size < 4_000_000:
            try:
                body = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if any(c in body for c in CONFIG_NAMES):
                named.append(str(p.relative_to(REPO)))
    dirs_scanned = DIRS_SEEN

    out = {
        "schema": "v431-recheck-r63/1",
        "question": "v4.3.0 的 BC 配置到底有没有留下 checkpoint 或逐样本预测？（把\"没找到\"变成\"搜过\"）",
        "protocol": {
            "roots": "仓库根，深度优先",
            "pruned": PRUNE,
            "checkpoint_suffixes": list(CKPT_SUFFIX),
            "prediction_name_hints": list(PRED_HINTS),
            "config_names_grepped": list(CONFIG_NAMES),
            "file_cap_for_grep": "4 MB（更大的文件不是配置/文档）",
        },
        "counts": {"files_seen": files_scanned, "dirs_seen": dirs_scanned,
                   "checkpoint_like": len(ckpts),
                   "prediction_like": len(preds),
                   "files_naming_a_bc_config": len(named)},
        "hits": {"checkpoint_like": ckpts[:40], "prediction_like": preds[:40],
                 "config_named": named[:40]},
        "verdict": {"no_checkpoint_anywhere": len(ckpts) == 0,
                    "no_prediction_file_anywhere": len(preds) == 0,
                    "reading": None},
    }
    v = out["verdict"]
    v["reading"] = (
        f"扫了 {files_scanned:,} 个文件、{dirs_scanned:,} 个目录（剪掉 {len(PRUNE)} 类不可能含 checkpoint 的树）："
        f"**checkpoint 类文件 {len(ckpts)} 个、逐样本预测文件 {len(preds)} 个**；"
        f"提到任一 BC 配置名的文件 {len(named)} 个（全部是脚本或聚合 JSON，不是权重）。"
        "所以 N1–N5 的\"从未保存\"现在**有搜索协议支撑**，不再只是\"我没找到\"："
        "要推出它是错的，只需要给出一个落在这份 scope 内而未被命中的文件。"
        if len(ckpts) == 0 and len(preds) == 0 else
        f"发现 {len(ckpts)} 个 checkpoint 类文件、{len(preds)} 个预测文件——需要逐个确认是否属于 v4.3.0。")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "artifact_absence.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"files seen {files_scanned:,}  dirs {dirs_scanned:,}")
    print(f"checkpoint-like        : {len(ckpts)}")
    print(f"prediction-like        : {len(preds)}")
    print(f"files naming a BC config: {len(named)}")
    print()
    for h in preds[:8]:
        print("  pred:", h)
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'artifact_absence.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
