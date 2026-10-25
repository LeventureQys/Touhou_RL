"""Build the consolidated verdict index for v3.6.0/v3.6.1.

One machine-readable table so a reviewer can see, per strategy, what was tested,
where the evidence lives, and why a verdict was reached — without re-reading eight
reports.  Pure read of the committed JSON artefacts; no numbers are recomputed.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

ROOT = pathlib.Path("Document/Update/Alpha")
V360 = ROOT / "v3.6.0"
V361 = ROOT / "v3.6.1"

STRATEGIES = ("LocalDodge", "REGION", "LURE", "ENEMY", "POWER", "BOSS", "EMERGENCY", "HISTORY")
NAMES = {
    "LocalDodge": "LocalDodge（局部避弹，对照）",
    "REGION": "LeaveRegion（大范围危险区域）",
    "LURE": "Lure / Reposition（诱导换边）",
    "ENEMY": "ClearEnemy（主动清怪）",
    "POWER": "CollectPower（条件收集 P）",
    "BOSS": "BossDamage（输出窗口）",
    "EMERGENCY": "Emergency / Bomb",
    "HISTORY": "Temporal encoder（历史块，§3.1）",
}


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    d1 = load(V360 / "experiments/T1-ablation/ablation.json")
    d2 = load(V360 / "experiments/T1-ablation-D2/ablation.json")
    d3 = load(V360 / "experiments/T1-ablation-D3/ablation.json")
    enemy_t2 = load(V361 / "experiments/t2-enemy/paired.json")
    power_t2 = load(V361 / "experiments/t2-power/paired.json")

    t1 = {"D1": d1, "D2": d2, "D3": d3}
    rows = []
    for strategy in STRATEGIES:
        entry = {"strategy": strategy, "name": NAMES[strategy], "t1": {}, "t2": None}
        for source, report in t1.items():
            verdict = report["verdicts"]["per_strategy"].get(strategy, {})
            cell = next(
                (
                    item
                    for item in report["cells"]
                    if item["owner"] == strategy and item["role"] == "primary"
                ),
                None,
            )
            entry["t1"][source] = {
                "verdict": verdict.get("verdict"),
                "gate_label": verdict.get("primary_label"),
                "absolute_value": verdict.get("absolute_value"),
                "increments": [
                    {
                        "label": branch["label"],
                        "delta": branch["delta"],
                        "p": branch["p"],
                    }
                    for branch in verdict.get("increments", [])
                ],
                "cells": int(report["inventory"]["rows"]),
                "episodes": int(report["inventory"]["episodes"]),
                "primary_delta": None if cell is None else cell.get("delta"),
            }
        rows.append(entry)

    # T2 (v3.6.1): only the strategies that passed their own T1 gate were admitted.
    for entry in rows:
        if entry["strategy"] == "ENEMY":
            summary = enemy_t2["summary"]
            entry["t2"] = {
                "arm": "clear_enemy",
                "pairs": enemy_t2["design"]["pairs"],
                "primary": enemy_t2["primary"],
                "effect": summary["primary_reduction"]["mean"],
                "effect_ci": [summary["primary_reduction"]["ci_low"], summary["primary_reduction"]["ci_high"]],
                "deaths_delta": summary["death_delta"]["mean"],
                "gates": summary["gates"],
                "verdict": summary["verdict"],
            }
        elif entry["strategy"] == "POWER":
            summary = power_t2["summary"]
            entry["t2"] = {
                "arm": "collect_power",
                "pairs": power_t2["design"]["pairs"],
                "primary": power_t2["primary"],
                "effect": summary["primary_reduction"]["mean"],
                "effect_ci": [summary["primary_reduction"]["ci_low"], summary["primary_reduction"]["ci_high"]],
                "deaths_delta": summary["death_delta"]["mean"],
                "gates": summary["gates"],
                "verdict": summary["verdict"],
            }

    admission = {
        "rule": "v3.6.1 §1 范围修正：策略在自己的 T1 门上通过即可进入 T2；无阈值改动",
        "admitted": [entry["strategy"] for entry in rows if entry["t1"]["D3"]["verdict"] == "pass" and entry["strategy"] != "LocalDodge"],
        "rejected": {
            entry["strategy"]: entry["t1"]["D3"]["verdict"]
            for entry in rows
            if entry["t1"]["D3"]["verdict"] != "pass" and entry["strategy"] != "LocalDodge"
        },
    }

    payload = {
        "schema": "v360-verdict-index/1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scope": {
            "T0": "v3.6.0 reports/T0-首步快照核查.md（5 项门禁全部闭合）",
            "T1": "v3.6.0 experiments/T1-ablation{,-D2,-D3}（7 策略 × 3 语料）",
            "T2": "v3.6.1 experiments/t2-{enemy,power}（2 条准入策略 × 20 对真机）",
        },
        "admission": admission,
        "rows": rows,
    }
    out = V360 / "teacher/index.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out), "strategies": len(rows), "admitted": admission["admitted"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
