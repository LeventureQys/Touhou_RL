"""v4.3.1 recheck R7d: the same frozen selection rule applied to all three ancestors.

R7 picked a combat window on the seed12 tape. A hold-out start point must come from a
different trajectory ancestor, so this script applies the *identical* rule to the
seed10 and seed11 tapes and assigns train/dev by ancestry:

    train : seed12
    dev   : seed10, seed11   (independent tapes, distinct file sha256, ~46% action
                              agreement with seed12 => different playthroughs)

Criteria are the ones frozen in R7 and are not re-tuned per tape.

Writes only new files under Document/Update/Alpha/v4.3.1/evidence/R7/.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
ROOT = REPO / ".tmp_v420_data"
OUT = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R7"
SEEDS = {"seed12": 12, "seed10": 10, "seed11": 11}
FPS_MEASURED = 13195 / 125.9
DECISION_FRAMES = 2
MIN_SLOTS, MAX_SLOTS = 90, 300
WIDTHS = (90, 120, 150, 200, 300)
STRIDE = 15


def shard(seed: int) -> pathlib.Path:
    return ROOT / "T04" / "dataset" / f"d1-s0d1-n3300-seed{seed}-shard"


def obs_path(seed: int) -> pathlib.Path:
    return ROOT / "T02c" / "rebuild" / f"d1-s0d1-n3300-seed{seed}-shard" / "2f" / "obs" / "play96x112.npy"


def analyse(name: str, seed: int) -> dict:
    slots = [json.loads(ln) for ln in
             (shard(seed) / "slots.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    n = len(slots)
    frame = np.array([int(s["decision_frame"]) for s in slots], dtype=np.int64)
    cand = np.array([int(s["executed_cand"]) for s in slots], dtype=np.int64)
    margin = np.array([float(s["margin"]) for s in slots], dtype=np.float64)
    scores = np.array([[float(x) for x in s["scores"]] for s in slots], dtype=np.float32)
    hashpre = [s["hash_pre"] for s in slots]
    hashpost = [s["hash_post"] for s in slots]
    chain_ok = sum(1 for i in range(n - 1) if hashpost[i] == hashpre[i + 1])

    obs = np.load(obs_path(seed), mmap_mode="r")
    load = np.empty(obs.shape[0], dtype=np.float32)
    for s in range(0, obs.shape[0], 512):
        blk = np.asarray(obs[s:s + 512], dtype=np.uint8)
        load[s:s + blk.shape[0]] = (blk.reshape(blk.shape[0], -1) > 96).mean(axis=1)
    pct = np.empty_like(load)
    pct[np.argsort(load)] = np.linspace(0.0, 100.0, len(load))
    spread = scores.max(axis=1) - scores.min(axis=1)

    cands = []
    for width in WIDTHS:
        for start in range(0, n - width + 1, STRIDE):
            cands.append({
                "start_slot": start,
                "end_slot": start + width,
                "width_slots": width,
                "duration_s": round(width * DECISION_FRAMES / 60.0, 2),
                "start_frame": int(frame[start]),
                "end_frame": int(frame[start + width - 1]),
                "mean_bullet_load_percentile": round(float(pct[start:start + width].mean()), 2),
                "distinct_executed_cand": int(len(set(cand[start:start + width].tolist()))),
                "slots_with_margin_below_8": int((margin[start:start + width] < 8.0).sum()),
                "mean_teacher_score_spread": round(float(spread[start:start + width].mean()), 3),
                "prefix_replay_s": round(start * DECISION_FRAMES / FPS_MEASURED, 1),
                "restore_hash_pre": hashpre[start],
                "restore_hash_pre_next": hashpre[start + 1],
            })

    def ok(c):
        return (c["mean_bullet_load_percentile"] >= 60.0
                and c["distinct_executed_cand"] >= 4
                and c["slots_with_margin_below_8"] >= 1)

    passing = [c for c in cands if ok(c)]
    passing.sort(key=lambda c: (-c["mean_bullet_load_percentile"], c["start_slot"]))
    return {
        "name": name,
        "episode_seed": seed,
        "slots": n,
        "first_frame": int(frame[0]),
        "last_frame": int(frame[-1]),
        "minutes": round((int(frame[-1]) - int(frame[0])) / 60.0 / 60.0, 2),
        "hash_chain_ok": int(chain_ok),
        "hash_chain_total": n - 1,
        "distinct_cand": len({int(x) for x in cand}),
        "n_windows": len(cands),
        "n_passing": len(passing),
        "top": passing[:6],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, seed in SEEDS.items():
        print(f"analysing {name} (seed {seed}) ...", flush=True)
        res[name] = analyse(name, seed)
        r = res[name]
        print(f"  slots={r['slots']} chain={r['hash_chain_ok']}/{r['hash_chain_total']} "
              f"passing={r['n_passing']}/{r['n_windows']}")

    # primary train segment: best 5 s window on seed12
    train5 = next((c for c in res["seed12"]["top"] if c["width_slots"] == 150), None)
    dev = {}
    for name in ("seed10", "seed11"):
        dev[name] = next((c for c in res[name]["top"] if c["width_slots"] == 150), None)

    spec = {
        "schema": "v431-recheck-r7d/1",
        "rule": {
            "seconds": [round(MIN_SLOTS * DECISION_FRAMES / 60.0, 2),
                        round(MAX_SLOTS * DECISION_FRAMES / 60.0, 2)],
            "C1_mean_bullet_load_percentile_min": 60.0,
            "C2_distinct_executed_cand_min": 4,
            "C2_slots_with_margin_below_8_min": 1,
            "preferred_width_slots": 150,
            "stride_slots": STRIDE,
            "note": ("rule fixed once in R7 and re-used unchanged for all three tapes; "
                     "no per-tape tuning"),
        },
        "per_ancestor": res,
        "train_segment": train5,
        "dev_segments": dev,
        "fps_measured": round(FPS_MEASURED, 2),
    }
    (OUT / "segment_spec_all_ancestors.json").write_text(
        json.dumps(spec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print(f"{'role':6s} {'tape':8s} {'slots':>12s} {'sec':>5s} {'frames':>13s} "
          f"{'bullets%':>9s} {'cands':>6s} {'prefix s':>9s}")
    print("-" * 78)
    for role, seg, tape in (("train", train5, "seed12"),
                            ("dev", dev["seed10"], "seed10"),
                            ("dev", dev["seed11"], "seed11")):
        if seg is None:
            print(f"{role:6s} {tape:8s}  no qualifying 5 s window")
            continue
        print(f"{role:6s} {tape:8s} {seg['start_slot']:5d}+{seg['width_slots']:<6d} "
              f"{seg['duration_s']:5.2f} {seg['start_frame']:6d}..{seg['end_frame']:<6d} "
              f"{seg['mean_bullet_load_percentile']:9.2f} {seg['distinct_executed_cand']:6d} "
              f"{seg['prefix_replay_s']:9.1f}")
    print()
    print(f"wrote: {OUT / 'segment_spec_all_ancestors.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
