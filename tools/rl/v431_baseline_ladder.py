"""v4.3.1 recheck R51: is C-const-best an achievable baseline, or an oracle?

`R27` defined `C-const-best` as, per window, the best of the 18 constant actions. Selecting the
best on the SAME window it is scored on is an in-sample maximum -- an oracle that knows both the
window identity and the outcome of all 18 actions in it. No policy can know that in advance.

That matters because §5.4's stop condition S2 fires when "arm A <= C-const-best", and the report
used that trigger (alongside the technical blocker) to justify not starting the pixel arms. If the
baseline is an oracle, S2 fires for ANY real method and is therefore unfalsifiable.

The honest, deployable version is computable from the same raw data:

    C-const-train : pick the ONE constant with the best mean over the 4 TRAIN windows,
                    then score that same constant on the 4 DEV windows.

That is a baseline a policy could actually commit to. This computes it, plus the oracle and the
pooled mean, so the three are side by side.

Writes only Document/Update/Alpha/v4.3.1/evidence/R51/.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RAW = REPO / ".tmp_v431_seg001" / "full" / "raw"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R51"
CAP = 300
TRAIN = (13, 14, 15, 16)
DEV = (11, 12, 17, 18)


def window_table(seed: int) -> dict[int, float]:
    """candidate index -> decisions survived in this window (300 if it never died)."""
    out = {}
    for p in sorted((RAW / "c4dense").glob(f"c4dense-seed{seed}-f8664-cand*.json")):
        cand = int(p.stem.split("-cand")[-1])
        rec = json.loads(p.read_text(encoding="utf-8"))
        off = rec.get("first_death_offset")
        out[cand] = CAP if off is None else off / 2.0
    return out


def main() -> int:
    train = {s: window_table(s) for s in TRAIN}
    dev = {s: window_table(s) for s in DEV}
    cands = sorted(set.intersection(*[set(t) for t in train.values()],
                                    *[set(d) for d in dev.values()]))

    # ---- oracle: per-window best, averaged ----
    oracle_per_window = {s: max(dev[s].values()) for s in DEV}
    oracle_mean = statistics.mean(oracle_per_window.values())

    # ---- deployable: one constant chosen on TRAIN, scored on DEV ----
    train_mean = {c: statistics.mean([train[s][c] for s in TRAIN]) for c in cands}
    best_cand = max(cands, key=lambda c: train_mean[c])
    chosen = {s: dev[s][best_cand] for s in DEV}
    chosen_mean = statistics.mean(chosen.values())

    # ---- pooled mean of all constants on DEV ----
    pool_mean = statistics.mean([v for s in DEV for v in dev[s].values()])

    # ---- how much the oracle inflates things ----
    inflation = oracle_mean - chosen_mean

    out = {
        "schema": "v431-recheck-r51/1",
        "question": ("C-const-best 是\"在同一批数据上挑最强\"的神谕，还是可部署的基线？"),
        "windows": {"train": list(TRAIN), "dev": list(DEV)},
        "candidates": len(cands),
        "baselines": {
            "oracle_per_window_best": {
                "mean": round(oracle_mean, 1),
                "per_window": {str(s): round(v, 1) for s, v in oracle_per_window.items()},
                "kind": "ORACLE",
                "why": ("每窗口在 18 个里挑最大的，且在同一窗口上评分——需要知道窗口身份"
                        "与该窗口全部 18 个动作的结果，任何真实策略都不可能预先知道"),
            },
            "const_chosen_on_train": {
                "mean": round(chosen_mean, 1),
                "constant_index": best_cand,
                "train_means_ranked": sorted(
                    ({"cand": c, "train_mean": round(m, 1)} for c, m in train_mean.items()),
                    key=lambda r: -r["train_mean"])[:5],
                "per_window": {str(s): round(v, 1) for s, v in chosen.items()},
                "kind": "DEPLOYABLE",
                "why": ("只在 4 个 train 窗口上挑一个动作，然后在 4 个 dev 窗口上评分——"
                        "这是策略能真正承诺的东西"),
            },
            "const_pool_mean_dev": {
                "mean": round(pool_mean, 1),
                "kind": "DEPLOYABLE",
                "why": "18 个恒定动作在 dev 上的池均值，等价于\"均匀随机挑一个动作一直用\"",
            },
        },
        "oracle_inflation": {
            "oracle_minus_deployable": round(inflation, 1),
            "factor": round(oracle_mean / chosen_mean, 2) if chosen_mean else None,
        },
        "verdict": {"s2_right_half_is_falsifiable": None, "reading": None},
    }

    # arm A's number, from R38
    ev = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence"
    r38 = json.loads((ev / "R38" / "n80.json").read_text(encoding="utf-8"))
    arm = r38["pooled"]["arm_a"]["mean"]
    out["arm_a_mean"] = arm
    out["comparisons"] = {
        "vs_oracle": round(arm - oracle_mean, 1),
        "vs_train_chosen": round(arm - chosen_mean, 1),
        "vs_pool": round(arm - pool_mean, 1),
    }

    v = out["verdict"]
    beats_deployable = arm > chosen_mean and arm > pool_mean
    # can a real (non-oracle) method ever PASS the condition? With the oracle: no, by
    # construction. With the deployable baseline: yes in principle.
    v["s2_right_half_is_falsifiable_with_oracle"] = False
    v["s2_right_half_is_falsifiable_with_deployable"] = True
    v["arm_beats_deployable_constant"] = bool(arm > chosen_mean)
    v["reading"] = (
        f"`C-const-best` 是神谕：每窗口在 18 个里挑最大并在同一窗口评分，均值 "
        f"{oracle_mean:.1f}，比**可部署**的\"train 上选、dev 上评\"（{chosen_mean:.1f}，"
        f"动作 #{best_cand}）高 **{inflation:.1f}**（{out['oracle_inflation']['factor']}×）。"
        f"臂 A 的 {arm} **低于神谕**（{arm - oracle_mean:+.1f}）但**高于所有可部署恒定动作基线**"
        f"（train-chosen {arm - chosen_mean:+.1f}、池均值 {arm - pool_mean:+.1f}）。"
        if beats_deployable else
        f"`C-const-best` 是神谕（{oracle_mean:.1f} vs 可部署 {chosen_mean:.1f}）；"
        f"臂 A 的 {arm} 既低于神谕（{arm - oracle_mean:+.1f}）也低于可部署版本"
        f"（{arm - chosen_mean:+.1f}）。")
    v["reading"] += (
        " 所以 `S2` 的右半（臂 A ≤ `C-const-best`）**对任何非神谕方法都会触发**，"
        "它作为停止条件是不可证伪的——报告用它来论证\"不启动像素臂\"是**误用**，"
        "真正挡住像素臂的是 `_pair_cand` 与 `PARK`，不是这个触发。"
        if arm < oracle_mean else "")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baseline_ladder.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"candidates           : {len(cands)}")
    print(f"oracle per-window    : {oracle_mean:7.1f}  (ORACLE)")
    print(f"const chosen on train: {chosen_mean:7.1f}  (cand #{best_cand}, DEPLOYABLE)")
    print(f"const pool mean (dev): {pool_mean:7.1f}  (DEPLOYABLE)")
    print(f"arm A                : {arm:7.1f}")
    print()
    print(f"oracle inflation     : +{inflation:.1f} over the deployable version "
          f"({out['oracle_inflation']['factor']}x)")
    print(f"arm A vs oracle      : {out['comparisons']['vs_oracle']:+.1f}")
    print(f"arm A vs train-chosen: {out['comparisons']['vs_train_chosen']:+.1f}")
    print(f"arm A vs pool        : {out['comparisons']['vs_pool']:+.1f}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'baseline_ladder.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
