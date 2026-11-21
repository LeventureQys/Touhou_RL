"""v4.3.1 recheck R7e: why do seeds 10 and 11 have no qualifying segment?

R7d applied the seed12-derived rule verbatim to the other two ancestors and found zero
windows. Before re-tuning anything, this script reports, per ancestor, the per-criterion
pass rate and the achievable joint maxima, so the reason is a measurement rather than a
guess.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
ROOT = REPO / ".tmp_v420_data"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R7"
SEEDS = {"seed12": 12, "seed10": 10, "seed11": 11}


def main() -> None:
    report = {}
    for name, seed in SEEDS.items():
        shard = ROOT / "T04" / "dataset" / f"d1-s0d1-n3300-seed{seed}-shard"
        obs_p = (ROOT / "T02c" / "rebuild" / f"d1-s0d1-n3300-seed{seed}-shard"
                 / "2f" / "obs" / "play96x112.npy")
        slots = [json.loads(ln) for ln in
                 (shard / "slots.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        n = len(slots)
        cand = np.array([int(s["executed_cand"]) for s in slots], dtype=np.int64)
        margin = np.array([float(s["margin"]) for s in slots], dtype=np.float64)
        obs = np.load(obs_p, mmap_mode="r")
        load = np.empty(obs.shape[0], dtype=np.float32)
        for s in range(0, obs.shape[0], 512):
            b = np.asarray(obs[s:s + 512], dtype=np.uint8)
            load[s:s + b.shape[0]] = (b.reshape(b.shape[0], -1) > 96).mean(axis=1)
        pct = np.empty_like(load)
        pct[np.argsort(load)] = np.linspace(0.0, 100.0, len(load))

        # how concentrated is the ink? fraction of slots above 20% ink
        above = float((load > 0.20).mean())
        # best joint: for each width, the window maximising min(normalised criteria)
        best = {}
        for width in (90, 120, 150, 200, 300):
            rows = []
            for start in range(0, n - width + 1, 15):
                mp = float(pct[start:start + width].mean())
                dc = int(len(set(cand[start:start + width].tolist())))
                tm = int((margin[start:start + width] < 8.0).sum())
                rows.append((mp, dc, tm, start))
            rows.sort(key=lambda r: -r[0])
            top = rows[0]
            # best window by the tight-margin criterion alone
            tight = max(rows, key=lambda r: r[2])
            best[width] = {
                "max_mean_bullet_pct": round(top[0], 2),
                "cands_at_that_window": top[1],
                "tight_at_that_window": top[2],
                "start_of_best_bullet": top[3],
                "max_tight_slots": tight[2],
                "bullet_pct_at_max_tight": round(tight[0], 2),
                "start_of_max_tight": tight[3],
                "n_windows_meeting_C1_alone": int(sum(1 for r in rows if r[0] >= 60.0)),
                "n_windows_meeting_C1_and_C2cand": int(
                    sum(1 for r in rows if r[0] >= 60.0 and r[1] >= 4)),
                "n_windows_meeting_all": int(
                    sum(1 for r in rows if r[0] >= 60.0 and r[1] >= 4 and r[2] >= 1)),
            }
        report[name] = {
            "slots": n,
            "fraction_slots_with_ink_above_20pct": round(above, 4),
            "mean_ink": round(float(load.mean()), 5),
            "median_ink": round(float(np.median(load)), 5),
            "p95_ink": round(float(np.percentile(load, 95)), 5),
            "max_ink": round(float(load.max()), 5),
            "by_width": best,
        }
        print(f"== {name}  ink>20%: {above:.4f}  mean {load.mean():.5f} "
              f"p95 {np.percentile(load,95):.5f}  max {load.max():.5f}")
        for width, r in best.items():
            print(f"   w={width:3d}: C1alo={r['n_windows_meeting_C1_alone']:4d} "
                  f"C1+C2cand={r['n_windows_meeting_C1_and_C2cand']:4d} "
                  f"all={r['n_windows_meeting_all']:4d} | max bullet% "
                  f"{r['max_mean_bullet_pct']:6.2f} @{r['start_of_best_bullet']:5d} "
                  f"(cands {r['cands_at_that_window']}, tight {r['tight_at_that_window']}) "
                  f"| max tight {r['max_tight_slots']:3d} @{r['start_of_max_tight']:5d}")

    (OUT / "segment_rule_diagnosis.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print()
    print(f"wrote: {OUT / 'segment_rule_diagnosis.json'}")


if __name__ == "__main__":
    main()
