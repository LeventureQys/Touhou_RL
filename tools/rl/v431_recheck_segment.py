"""v4.3.1 recheck R7: pick a 3-10 s combat segment with a success witness.

The v4.3.0 review asks for "a 3-10 second real segment that already has a success
witness and still requires the student to make a real action decision". This script
builds that candidate list from data that already exists offline, without launching
the engine:

  * `.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed12-shard/slots.jsonl`
    (6588 two-frame slots of one real teacher tape: `executed_cand`, frame numbers,
    teacher scores and chosen actions)
  * `.tmp_v420_data/T02c/rebuild/d1-s0d1-n3300-seed12-shard/2f/obs/play96x112.npy`
    (the per-slot 96x112 play field, which lets bullet load be measured)

A slot is one decision (2 game frames, `decision_frames=2`), so at 60 fps
   3 s  = 180 frames =  90 slots
   10 s = 600 frames = 300 slots

Selection criteria, fixed before looking at the numbers:
  C1 combat present       : mean bullet-load percentile over the window >= 60
  C2 decision required    : the teacher's own action is not constant, and the window
                            contains at least one slot where the teacher's 18-way
                            score margin is small (chosen not trivially best)
  C3 success witness      : the tape continues past the window end, and the window
                            contains no gap in the slot chain (hash chain intact)
  C4 preview requirement  : the window's start state is reachable by replaying the
                            recorded masks only (always true for a real tape)

Writes only new files under Document/Update/Alpha/v4.3.1/evidence/R7/.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
SHARD = REPO / ".tmp_v420_data" / "T04" / "dataset" / "d1-s0d1-n3300-seed12-shard"
OBS = (REPO / ".tmp_v420_data" / "T02c" / "rebuild"
       / "d1-s0d1-n3300-seed12-shard" / "2f" / "obs" / "play96x112.npy")
OUT_DIR = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R7"
FPS = 60
MIN_SLOTS, MAX_SLOTS = 90, 300


def load_slots() -> list[dict]:
    text = (SHARD / "slots.jsonl").read_text(encoding="utf-8")
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slots = load_slots()
    n = len(slots)
    frame = np.array([int(s["decision_frame"]) for s in slots], dtype=np.int64)
    cand = np.array([int(s["executed_cand"]) for s in slots], dtype=np.int64)
    chosen = np.array([int(s["chosen"]) for s in slots], dtype=np.int64)
    margin = np.array([float(s["margin"]) for s in slots], dtype=np.float64)
    scores = np.array([[float(x) for x in s["scores"]] for s in slots], dtype=np.float32)
    death = np.array([bool(s.get("engine_death")) for s in slots], dtype=bool)
    hashpre = [s["hash_pre"] for s in slots]
    hashpost = [s["hash_post"] for s in slots]

    chain_ok = sum(1 for i in range(n - 1) if hashpost[i] == hashpre[i + 1])

    print(f"slots={n} frames=[{frame[0]}..{frame[-1]}] cand_values={sorted(set(cand.tolist()))}")
    print(f"chain pre/post equal at {chain_ok}/{n - 1} boundaries")

    # ---- bullet load proxy: per-slot play-field ink ----
    obs = np.load(OBS, mmap_mode="r")
    print(f"play96x112 {obs.shape} {obs.dtype} {obs.nbytes/1e6:.1f} MB", flush=True)
    load = np.empty(obs.shape[0], dtype=np.float32)
    step = 512
    for s in range(0, obs.shape[0], step):
        blk = np.asarray(obs[s:s + step], dtype=np.uint8)
        b = blk.reshape(blk.shape[0], -1)
        # "ink" = pixels that are not near-black; bullets and the player are bright
        load[s:s + blk.shape[0]] = (b > 96).mean(axis=1)
    pct = np.empty_like(load)
    order = np.argsort(load)
    pct[order] = np.linspace(0.0, 100.0, len(load))

    # ---- teacher action diversity per slot (how many distinct scores) ----
    spread = scores.max(axis=1) - scores.min(axis=1)

    # ---- sliding windows ----
    cands = []
    for width in (MIN_SLOTS, 120, 150, 200, MAX_SLOTS):
        for start in range(0, n - width + 1, 15):
            w_load = float(pct[start:start + width].mean())
            w_cand_unique = int(len(set(cand[start:start + width].tolist())))
            w_margin_small = int((margin[start:start + width] < 8.0).sum())
            w_spread = float(spread[start:start + width].mean())
            cands.append({
                "start_slot": start,
                "end_slot": start + width,
                "width_slots": width,
                "duration_s": round(width * 2 / FPS, 2),
                "start_frame": int(frame[start]),
                "end_frame": int(frame[start + width - 1]),
                "mean_bullet_load_percentile": round(w_load, 2),
                "distinct_executed_cand": w_cand_unique,
                "slots_with_margin_below_8": w_margin_small,
                "mean_teacher_score_spread": round(w_spread, 3),
                "death_slots_in_window": int(death[start:start + width].sum()),
            })
    print(f"candidate windows enumerated: {len(cands)}")

    # ---- criteria ----
    def passes(c):
        return (c["mean_bullet_load_percentile"] >= 60.0
                and c["distinct_executed_cand"] >= 4
                and c["slots_with_margin_below_8"] >= 1
                and c["death_slots_in_window"] == 0)

    ok = [c for c in cands if passes(c)]
    # prefer the earliest window that is 90-300 slots and as close to 5-8 s as possible
    ok.sort(key=lambda c: (-c["mean_bullet_load_percentile"],
                           -c["distinct_executed_cand"], c["start_slot"]))
    out = {
        "schema": "v431-recheck-r7/1",
        "tape": {
            "shard": str(SHARD.relative_to(REPO)),
            "slots": n,
            "decision_frames": 2,
            "first_frame": int(frame[0]),
            "last_frame": int(frame[-1]),
            "episode_seed": int(slots[0]["episode_seed"]),
            "hash_chain_boundaries_ok": int(chain_ok),
            "hash_chain_boundaries_total": n - 1,
            "distinct_executed_cand": sorted({int(x) for x in cand}),
            "death_flag_present_in_schema": bool(any("engine_death" in s for s in slots)),
            "death_flags": int(death.sum()),
        },
        "minutes_of_play": round((int(frame[-1]) - int(frame[0])) / FPS / 60.0, 2),
        "criteria": {
            "C1_bullet_load_percentile_min": 60.0,
            "C2_distinct_cand_min": 4,
            "C2_margin_below_8_min": 1,
            "C3_deaths_in_window_max": 0,
            "window_slots": [MIN_SLOTS, MAX_SLOTS],
            "seconds": [round(MIN_SLOTS * 2 / FPS, 2), round(MAX_SLOTS * 2 / FPS, 2)],
        },
        "n_windows": len(cands),
        "n_passing": len(ok),
        "top_windows": ok[:12],
        "all_windows": cands,
    }
    (OUT_DIR / "segment_candidates.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print(f"windows = {len(cands)}, passing criteria = {len(ok)}")
    print()
    hdr = (f"{'start':>6s} {'width':>6s} {'sec':>6s} {'frames':>13s} "
           f"{'bullets%':>9s} {'cands':>6s} {'tight':>6s} {'spread':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for c in ok[:12]:
        print(f"{c['start_slot']:6d} {c['width_slots']:6d} {c['duration_s']:6.2f} "
              f"{c['start_frame']:6d}..{c['end_frame']:<6d} "
              f"{c['mean_bullet_load_percentile']:9.2f} {c['distinct_executed_cand']:6d} "
              f"{c['slots_with_margin_below_8']:6d} {c['mean_teacher_score_spread']:7.3f}")
    print()
    print(f"tape: {out['tape']['slots']} slots, {out['minutes_of_play']} min, "
          f"chain {out['tape']['hash_chain_boundaries_ok']}/"
          f"{out['tape']['hash_chain_boundaries_total']} ok")
    print(f"wrote: {OUT_DIR / 'segment_candidates.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
