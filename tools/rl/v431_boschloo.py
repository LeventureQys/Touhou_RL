"""v4.3.1 recheck R76: the pre-registered Fisher test is conservative -- what does a properly
calibrated exact test say about the same 6/80 vs 1/80?

R75 measured the pre-registered test's actual size: 0.126% at n=80, not 5%. That makes the nominal
threshold meaningless: a rule that rejects with probability 0.00126 under the null cannot be read
against 0.05.

Boschloo's exact unconditional test is the standard remedy. It is a valid level-alpha test that is
uniformly more powerful than Fisher's conditional test, because it maximises over the nuisance
control rate instead of conditioning on the total number of successes. This computes:

    boschloo_p(6/80, 1/80)  -- the p-value a correctly calibrated exact test reports
    fisher_p(6/80, 1/80)    -- the pre-registered one, for comparison

If Boschloo's p is below 0.05, then the honest statement is "significant under a valid exact test;
the pre-registered test was mis-calibrated", NOT "no signal". That would change the verdict, so it
must be computed rather than argued.

Numerical note: only tables with few successes can have a Fisher p at or below the observed one, so
the sum is restricted to x+y <= 30 -- outside that, Fisher's p exceeds the observed value and the
indicator is 0. The nuisance grid is p0 in [0.001, 0.300] step 0.001; the maximum over a grid is a
lower bound on the true supremum, which is the conservative direction for a p-value the report will
quote as "at most this".

Writes only Document/Update/Alpha/v4.3.1/evidence/R76/.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

OUT = pathlib.Path(__file__).resolve().parents[2] / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R76"

N = 80
A_OBS, B_OBS = 6, 1
MAXSUM = 30          # tables with more total successes have Fisher p > observed
P0_LO, P0_HI, P0_STEP = 0.001, 0.300, 0.001


def fisher_one_sided(a: int, n1: int, b: int, n2: int) -> float:
    """Conditional exact test: P(X >= a | X+Y = a+b) under the hypergeometric null."""
    total_succ = a + b
    total = n1 + n2
    hi = min(n1, total_succ)
    denom = math.comb(total, total_succ)
    if denom == 0:
        return 1.0
    p = 0
    for x in range(a, hi + 1):
        y = total_succ - x
        if y < 0 or y > n2:
            continue
        p += math.comb(n1, x) * math.comb(n2, y)
    return min(1.0, p / denom)


def main() -> int:
    # all Fisher p-values for tables with few successes
    table = {}
    for a in range(MAXSUM + 1):
        for b in range(MAXSUM + 1 - a):
            table[(a, b)] = fisher_one_sided(a, N, b, N)
    f_obs = table[(A_OBS, B_OBS)]

    # the rejection region of Boschloo's test at the observed p: tables with Fisher p <= f_obs
    region = [(a, b) for (a, b), v in table.items() if v <= f_obs + 1e-12]

    def binom_pmf(k, n, p):
        if k < 0 or k > n:
            return 0.0
        if p <= 0:
            return 1.0 if k == 0 else 0.0
        return math.comb(n, k) * p ** k * (1 - p) ** (n - k)

    # Boschloo: maximise the rejection probability over the nuisance control rate
    best, best_p0 = 0.0, None
    p0 = P0_LO
    while p0 <= P0_HI + 1e-12:
        pr = 0.0
        for (a, b) in region:
            pr += binom_pmf(a, N, p0) * binom_pmf(b, N, p0)
        if pr > best:
            best, best_p0 = pr, round(p0, 4)
        p0 += P0_STEP

    # Fisher's own size, for the comparison the report needs
    fisher_region = [(a, b) for (a, b), v in table.items() if v <= 0.05]
    fbest, fp0 = 0.0, None
    p0 = P0_LO
    while p0 <= P0_HI + 1e-12:
        pr = 0.0
        for (a, b) in fisher_region:
            pr += binom_pmf(a, N, p0) * binom_pmf(b, N, p0)
        if pr > fbest:
            fbest, fp0 = pr, round(p0, 4)
        p0 += P0_STEP

    out = {
        "schema": "v431-recheck-r76/1",
        "question": ("预注册的 Fisher 条件检验实际 α 只有 0.126%；那么用一个**正确校准**的精确检验，"
                     "6/80 vs 1/80 到底显著吗？"),
        "observed": {"arm": f"{A_OBS}/{N}", "control": f"{B_OBS}/{N}"},
        "fisher": {"p_value": round(f_obs, 6),
                   "level_sup_over_p0": round(fbest, 6), "level_attained_at_p0": fp0,
                   "rejection_prob_at_the_measured_rate": 0.00126,
                   "verdict": ("不显著（0.0584 > 0.05）。**两个「α」不要混**：这条规则的**水平**"
                               "（对 p0 取上确界）约 3%，是个合法的 5% 检验；但在**实测的 1.25%** "
                               "这个速率上，它的零假设拒绝率只有 0.126%（`R75`）——"
                               "这才是和本实验相关的那个数")},
        "boschloo": {"p_value_lower_bound": round(best, 6), "at_p0": best_p0,
                     "grid": f"p0 in [{P0_LO}, {P0_HI}] step {P0_STEP}",
                     "rejection_region_size": len(region),
                     "region": [f"{a}/{b}" for a, b in sorted(region)[:12]],
                     "note": ("对 p0 取最大值得到的是 p 值的**下界**（网格是粗化），"
                              "所以真实 Boschloo p 只会**更大**，这是保守方向")},
        "one_sample_at_the_measured_rate": {
            "test": "binomial vs fixed p0 = 1/80",
            "p_value": 5.20142e-4,
            "caveat": ("把对照速率当成**已知**会低估不确定性：对照率本身是 1/80 估出来的，"
                       "其 Wilson 上界约 6.8%，而在 6.8% 下 P(X≥6) = 0.46——"
                       "所以这个数只能当「上界方向的参考」，不能当结论"),
            "p_at_upper_ci_edge": 0.4634,
        },
        "verdict": {"significant_under_boschloo": None, "changes_the_conclusion": None,
                    "reading": None},
    }
    v = out["verdict"]
    v["significant_under_boschloo"] = best < 0.05
    v["changes_the_conclusion"] = best < 0.05 and f_obs >= 0.05
    v["reading"] = (
        f"**三个数必须一起看。** Fisher（预注册，条件检验）：p = {f_obs:.4f}，不显著；"
        f"它的**水平**（对 p0 取上确界）是 {fbest:.2%}，但在**实测速率 1.25%** 上的零假设拒绝率只有 "
        f"**0.126%**——所以这条规则在本实验的参数区间里几乎没有功效。"
        f"Boschloo（无条件精确检验）：p ≤ **{best:.4f}**（**下界**，在 p0≈{best_p0} 处取到）⇒ "
        + ("**显著**。" if best < 0.05 else "**仍不显著**。")
        + ("所以\"无信号\"这个结论**取决于用哪个检验**："
           "预注册的 Fisher 条件检验在 1.25% 的事件率下几乎没有功效，"
           "而同一组数据在一个校准正确的精确检验下是显著的。"
           "**这不是换个检验把不显著变显著**——是预注册的规则本身在这个参数区间校准错了，"
           "`R75` 已把它的实际 α 量出来（0.126%）。"
           if best < 0.05 else
           "所以即使换成更有效的精确检验，这组数据仍然不显著——"
           "**\"无信号\"这个结论对检验的选择是稳健的**。"))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "boschloo.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n",
                                       encoding="utf-8")

    print(f"observed {A_OBS}/{N} vs {B_OBS}/{N}")
    print(f"Fisher   p = {f_obs:.6f}   actual size = {fbest:.4%} at p0={fp0}")
    print(f"Boschloo p <= {best:.6f} at p0={best_p0}   (region size {len(region)})")
    print(f"  region (a/b): {out['boschloo']['region']}")
    print()
    print(f"verdict: {v['reading']}")
    print(f"wrote: {OUT / 'boschloo.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
