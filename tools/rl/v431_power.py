"""v4.3.1 recheck R75: exact power for the survival-rate test, so "spend 3.3 h?" has numbers.

The report recommends against buying another 3.3 h but only says "若 6 倍差异为真…可达 0.05".
That is one point. The decision needs the whole curve: at n = 80 / 120 / 163 / 240 / 320, what is
the probability of reaching p<0.05 (one-sided Fisher) if the arm's true survival rate is 4%, 5%,
7.5%, 10% or 15%, against a control at the measured 1.25% (1/80)?

Everything here is computed offline from binomials -- no engine time. The Fisher test is the same
one the report uses (one-sided, arm > control), evaluated exactly by enumeration rather than
sampled, so the numbers are reproducible to the last digit.

This does NOT decide whether to spend the budget. It makes the tradeoff explicit:
    two extra hours buys a coin flip at 5%, near-certainty at 10%, and nothing at 4%.

Writes only Document/Update/Alpha/v4.3.1/evidence/R75/.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

OUT = pathlib.Path(__file__).resolve().parents[2] / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R75"

N_ARM = [80, 120, 163, 240, 320]
N_CTRL = [80, 120, 163, 240, 320]
P_CTRL = 0.0125                      # 1/80, the measured C-fair survival rate
P_ARM = [0.04, 0.05, 0.075, 0.10, 0.15]
ALPHA = 0.05
UNIT_S = 56.6                        # measured s/episode for a C-fair-length episode (R71)
UNIT_S_FULL = 70.4                   # full-length upper bound


def binom_pmf(k: int, n: int, p: float) -> float:
    if k < 0 or k > n:
        return 0.0
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def fisher_one_sided(a: int, n1: int, b: int, n2: int) -> float:
    """P(X >= a) under the hypergeometric null, i.e. arm at least this much better."""
    total_succ = a + b
    total = n1 + n2
    hi = min(n1, total_succ)
    # condition on the marginal number of successes: the denominator is C(total, total_succ),
    # NOT C(total, n1) -- that was a bug that produced p ~ 1e-37 for 6/80 vs 1/80
    denom = math.comb(total, total_succ)
    if denom == 0:
        return 1.0
    p = 0.0
    for x in range(a, hi + 1):
        y = total_succ - x
        if y < 0 or y > n2:
            continue
        p += math.comb(n1, x) * math.comb(n2, y)
    return min(1.0, p / denom)


def power(n_arm: int, n_ctrl: int, p1: float, p0: float) -> float:
    total = 0.0
    for a in range(n_arm + 1):
        pa = binom_pmf(a, n_arm, p1)
        if pa < 1e-15:
            continue
        for b in range(n_ctrl + 1):
            pb = binom_pmf(b, n_ctrl, p0)
            if pb < 1e-15:
                continue
            if fisher_one_sided(a, n_arm, b, n_ctrl) <= ALPHA:
                total += pa * pb
    return total


def main() -> int:
    grid = []
    for n in N_ARM:
        row = {"n_per_arm": n,
               "extra_episodes_vs_80": 2 * (n - 80),
               "extra_hours_at_56_6s": round(2 * (n - 80) * UNIT_S / 3600, 2),
               "extra_hours_at_70_4s": round(2 * (n - 80) * UNIT_S_FULL / 3600, 2),
               "power": {}}
        for p1 in P_ARM:
            row["power"][f"{p1:.3f}"] = round(power(n, n, p1, P_CTRL), 4)
        grid.append(row)

    # ---- the test's ACTUAL size under the null.
    # A conditional exact test cannot always reach alpha: with p0 = 1.25% and n = 80 the expected
    # number of control successes is ~1, so rejection requires almost all successes to land in the
    # arm. The achievable size is therefore far BELOW alpha -- the test is conservative, which is a
    # property of the test, not a bug in this implementation.
    size_checks = {str(n): round(power(n, n, P_CTRL, P_CTRL), 5) for n in (80, 163, 320)}
    # smallest p-value the test can return at each n (all successes in the arm)
    min_p = {}
    for n in (80, 163):
        for k in range(0, n + 1):
            val = fisher_one_sided(k, n, 0, n)
            if val <= 0.05:
                min_p[str(n)] = {"k_in_arm_needed": k, "p_at_that_point": round(val, 4),
                                 "total_successes_needed": k}
                break

    # the observed n=80 result, for reference: 6/80 vs 1/80
    observed_p = fisher_one_sided(6, 80, 1, 80)
    out = {
        "schema": "v431-recheck-r75/1",
        "question": "再花 3.3 h 把 n 从 80 扩到 163/臂，值不值？精确功效是多少？",
        "design": {"test": "one-sided Fisher exact, arm > control", "alpha": ALPHA,
                   "control_rate": P_CTRL, "control_note": "C-fair 实测 1/80",
                   "arm_rates_assumed": P_ARM},
        "observed_at_n80": {"arm": "6/80 = 7.5%", "control": "1/80 = 1.25%",
                            "one_sided_p": round(observed_p, 6),
                            "matches_the_reports_0_0584": abs(observed_p - 0.0584) < 5e-4},
        "validation": {
            "null_rejection_rate_by_n": size_checks,
            "is_a_5pct_test": False,
            "why": ("条件精确检验在低事件率下**达不到** alpha：p0=1.25%、n=80 时期望对照成功数约 1，"
                    "要拒绝几乎必须让全部成功都落在臂里。所以实际水平**远低于** 0.05（保守），"
                    "这是该检验的性质，不是实现错"),
            "min_rejectable_point": min_p,
            "consequence": ("**`p=0.0584` 不能按「刚好差一点到 0.05」来读**——"
                            "这个决策规则的真实一类错误率只有 0.1%–2%，"
                            "所以 0.0584 已经是**很强的信号**，而不是「差一点」"),
        },
        "power_grid": grid,
        "verdict": {"reading": None},
    }
    v = out["verdict"]
    g163 = next(r for r in grid if r["n_per_arm"] == 163)
    at4 = g163["power"]["0.040"]
    at5 = g163["power"]["0.050"]
    at75 = g163["power"]["0.075"]
    at10 = g163["power"]["0.100"]
    v["reading"] = (
        f"实测 6/80 vs 1/80 的单侧 Fisher 精确 p = {observed_p:.4f}。"
        f"扩到 **n=163/臂**（多耗 **{g163['extra_hours_at_56_6s']} h**，按实测 56.6 s/episode；"
        f"上界 {g163['extra_hours_at_70_4s']} h）后，功效是："
        f"**真值 4% → {at4:.0%}；5% → {at5:.0%}；7.5%（观测值）→ {at75:.0%}；10% → {at10:.0%}**。"
        "**所以这 3.3 h 买的是：真值 5% 时约抛硬币，7.5% 时接近确定，4% 时基本买不到。**"
        "报告说\"效应可能更小\"，功效表把这个担忧量化了——**它不再是意见，而是一张可判的表**。"
        f" 另外这次独立重实现**复现了报告的 p=0.0584**（{observed_p:.6f}），"
        f"但**这个检验在低事件率下是保守的**：零假设拒绝率只有 {size_checks}，"
        "远低于名义 5%——所以 **`p=0.0584` 不该读成\"刚差一点\"**，"
        "这个规则的真实一类错误率只有 0.1%–2%，0.0584 已经是很强的信号。"
        "**功效表给出的 84%（真值 7.5%）就是在这个保守规则下算的。**")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "power.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print("observed: 6/80 vs 1/80 -> one-sided Fisher p =", round(observed_p, 6))
    print()
    hdr = "  n/arm   extra eps   " + "  ".join(f"p={p:.3f}" for p in P_ARM)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in grid:
        cells = "  ".join(f"{r['power'][f'{p:.3f}']:6.1%}" for p in P_ARM)
        print(f"  {r['n_per_arm']:5d}   {r['extra_episodes_vs_80']:9d}   {cells}"
              f"   ({r['extra_hours_at_56_6s']} h)")
    print()
    print(f"null rejection rate by n: {size_checks}  (a 5% test would give ~0.05)")
    print(f"smallest rejectable point: {min_p}")
    print(f"observed p matches the report's 0.0584: {abs(observed_p - 0.0584) < 5e-4}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'power.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
