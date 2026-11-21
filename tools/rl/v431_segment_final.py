"""v4.3.1 recheck R8: final segment selection under corrected, comparable criteria.

R7f established that the `margin < 8` criterion I froze in R7 is not a usable
"decision required" test: `margin` is quantised with median and maximum both 16.0 on
all three tapes, it falls below 8 four times in 6,588 slots on seed12 and never on
seed10/seed11, and it is not the top-1/top-2 score gap (that gap is below 8 in 6,588 of
6,588 slots). Using it would have selected hold-out segments by an instrument outlier.

This script therefore fixes the criteria to quantities that are behavioural and
comparable across tapes:

  C1 combat, not cutscene : mean play-field ink percentile >= 60 within the tape
  C2 the teacher decided  : >= 4 distinct executed cands inside the window
  C3 cheap enough         : prefix replay <= 120 s at the measured 104.8 frames/s
  C4 (engine, later)      : a frozen constant-action control must FAIL the window in
                            >= 4 of 5 attempts, otherwise the window is rejected

C4 cannot be evaluated offline; it is named here as a hard pre-registered gate so the
segment cannot be certified "needs a combat decision" from data alone.

Writes only new files under Document/Update/Alpha/v4.3.1/evidence/R8/.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
ROOT = REPO / ".tmp_v420_data"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R8"
SEEDS = {"seed12": 12, "seed10": 10, "seed11": 11}
FPS_MEASURED = 13195 / 125.9
DECISION_FRAMES = 2
WIDTHS = (90, 150, 300)          # 3 s, 5 s, 10 s
STRIDE = 5
PREFIX_BUDGET_S = 120.0


def analyse(name: str, seed: int) -> dict:
    shard = ROOT / "T04" / "dataset" / f"d1-s0d1-n3300-seed{seed}-shard"
    obs_p = (ROOT / "T02c" / "rebuild" / f"d1-s0d1-n3300-seed{seed}-shard"
             / "2f" / "obs" / "play96x112.npy")
    slots = [json.loads(ln) for ln in
             (shard / "slots.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    n = len(slots)
    frame = np.array([int(s["decision_frame"]) for s in slots], dtype=np.int64)
    cand = np.array([int(s["executed_cand"]) for s in slots], dtype=np.int64)
    hashpre = [s["hash_pre"] for s in slots]
    hashpost = [s["hash_post"] for s in slots]
    masks = [int(s["mask"]) for s in slots]
    chain_ok = sum(1 for i in range(n - 1) if hashpost[i] == hashpre[i + 1])

    obs = np.load(obs_p, mmap_mode="r")
    ink = np.empty(obs.shape[0], dtype=np.float32)
    for s in range(0, obs.shape[0], 512):
        b = np.asarray(obs[s:s + 512], dtype=np.uint8)
        ink[s:s + b.shape[0]] = (b.reshape(b.shape[0], -1) > 96).mean(axis=1)
    pct = np.empty_like(ink)
    pct[np.argsort(ink)] = np.linspace(0.0, 100.0, len(ink))

    # precompute cand entropy per slot is unnecessary; windows use distinct count
    best = {}
    for width in WIDTHS:
        rows = []
        for start in range(0, n - width + 1, STRIDE):
            mp = float(pct[start:start + width].mean())
            dc = int(len(set(cand[start:start + width].tolist())))
            prefix_s = start * DECISION_FRAMES / FPS_MEASURED
            rows.append((start, width, mp, dc, prefix_s))
        elig = [r for r in rows if r[2] >= 60.0 and r[3] >= 4 and r[4] <= PREFIX_BUDGET_S]
        elig.sort(key=lambda r: (-r[2], -r[3], r[0]))
        if elig:
            s, w, mp, dc, ps = elig[0]
            best[width] = {
                "start_slot": int(s),
                "end_slot": int(s + w),
                "width_slots": int(w),
                "duration_s": round(w * DECISION_FRAMES / 60.0, 2),
                "start_frame": int(frame[s]),
                "end_frame": int(frame[s + w - 1]),
                "mean_ink_percentile": round(mp, 2),
                "distinct_executed_cand": int(dc),
                "prefix_replay_s": round(ps, 1),
                "segment_play_s": round(w * DECISION_FRAMES / FPS_MEASURED, 1),
                "restore_gate_hash_pre": hashpre[s],
                "recorded_masks_prefix_first5": masks[:5],
                "n_eligible_ranked_1st": len(elig),
            }
        else:
            best[width] = None
    return {
        "episode_seed": seed,
        "slots": n,
        "first_frame": int(frame[0]),
        "last_frame": int(frame[-1]),
        "minutes": round((int(frame[-1]) - int(frame[0])) / 60.0 / 60.0, 2),
        "hash_chain_ok": int(chain_ok),
        "hash_chain_total": n - 1,
        "distinct_cand_in_tape": len({int(x) for x in cand}),
        "windows": best,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    per = {}
    for name, seed in SEEDS.items():
        print(f"analysing {name} ...", flush=True)
        per[name] = analyse(name, seed)
        for w, seg in per[name]["windows"].items():
            if seg is None:
                print(f"   {w:4d} slots : none eligible")
            else:
                print(f"   {w:4d} slots : start {seg['start_slot']:5d} "
                      f"({seg['duration_s']:5.2f}s) ink% {seg['mean_ink_percentile']:6.2f} "
                      f"cands {seg['distinct_executed_cand']:3d} "
                      f"prefix {seg['prefix_replay_s']:6.1f}s")

    def pick(tape: str, width: int):
        w = per[tape]["windows"].get(width)
        if w is None:
            return None
        return {**w, "tape": tape, "episode_seed": per[tape]["episode_seed"],
                "restore_gate_hash_pre": w["restore_gate_hash_pre"]}

    spec = {
        "schema": "v431-recheck-r8/1",
        "corrected_criteria": {
            "C1_mean_ink_percentile_min": 60.0,
            "C2_distinct_executed_cand_min": 4,
            "C3_prefix_replay_s_max": PREFIX_BUDGET_S,
            "C4_control_must_fail": {
                "gate": "frozen constant-action policy must lose a life or fail to "
                        "complete the window in >= 4 of 5 attempts",
                "status": "NOT EVALUATED OFFLINE — engine required",
            },
            "dropped": {
                "C2_teacher_margin_below_8": (
                    "dropped after R7f: margin is quantised (median = max = 16.0 on all "
                    "three tapes), fires 4/6588 on seed12 and 0/6592 on both others, and "
                    "is not the top-1/top-2 score gap (that gap is < 8 on 6588/6588)."),
            },
        },
        "measured_replay_fps": round(FPS_MEASURED, 2),
        "per_ancestor": per,
        "selection": {
            "train_5s": pick("seed12", 150),
            "train_3s": pick("seed12", 90),
            "train_10s": pick("seed12", 300),
            "dev_seed10_5s": pick("seed10", 150),
            "dev_seed11_5s": pick("seed11", 150),
        },
    }
    (OUT / "segment_spec_final.json").write_text(
        json.dumps(spec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print("== frozen selection ==")
    for role, seg in spec["selection"].items():
        if seg is None:
            print(f"  {role:16s}: none")
            continue
        print(f"  {role:16s}: {seg['tape']} slot {seg['start_slot']}..{seg['end_slot']} "
              f"({seg['duration_s']}s, frames {seg['start_frame']}..{seg['end_frame']}) "
              f"ink% {seg['mean_ink_percentile']} cands {seg['distinct_executed_cand']} "
              f"prefix {seg['prefix_replay_s']}s")
    print()
    print(f"wrote: {OUT / 'segment_spec_final.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
