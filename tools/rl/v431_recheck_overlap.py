"""v4.3.1 recheck R2: is the cross-tape "common prefix" real input duplication?

The v4.3.0 documents claim seed10 and seed12 share their first 219 *decisions* and
conclude that validating on seed10 includes "~20% literal training data". That
conclusion was derived from the `cand` (action) sequence alone.

`cand` equality is necessary but NOT sufficient for input duplication. This script
separates three questions that were previously collapsed into one:

  Q1  Are the `cand` sequences identical over the prefix?          (action level)
  Q2  Are the stored observations identical over the prefix?       (pixel / history level)
  Q3  Do the tapes share a trajectory ancestor?                    (provenance level)

Q2 is the one that actually decides whether a hold-out sample duplicates a
training sample. If the pixels differ, a model cannot "memorise" the sample; if
they are byte-identical, it can.

Also measures how much *within-tape* near-duplication exists, because stride-6
sampling puts every adjacent pair 6 game frames apart.

Writes only new files under Document/Update/Alpha/v4.3.1/evidence/R2/.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
O2 = REPO / ".tmp_v430_o2"
CORPORA = {
    "seed12": O2 / "o2-strided",
    "seed10": O2 / "o2-stride-seed10",
    "seed11": O2 / "o2-stride-seed11",
}
OUT_DIR = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R2"
OBS_ARRAYS = ("play_gray", "hud_gray", "global_rgb", "executed_key_history", "history_valid")


def load_labels(p: pathlib.Path) -> list[dict]:
    return [json.loads(ln) for ln in
            (p / "labels.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]


def row_hashes(arr: np.ndarray) -> list[str]:
    """SHA-256 of each sample's bytes, so duplicates can be found by set logic."""
    a = np.asarray(arr)
    a = a.reshape(a.shape[0], -1)
    return [hashlib.sha256(np.ascontiguousarray(r).tobytes()).hexdigest() for r in a]


def common_prefix(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def compare_prefix(arr_a, arr_b, cp: int, limit: int = 0) -> dict:
    """Per-sample agreement over [0, cp)."""
    if cp == 0:
        return {"n_compared": 0}
    idx = range(cp) if limit <= 0 else range(min(cp, limit))
    ha = row_hashes(np.asarray(arr_a[:cp]))
    hb = row_hashes(np.asarray(arr_b[:cp]))
    identical = sum(1 for x, y in zip(ha, hb) if x == y)
    a = np.asarray(arr_a[:cp]).astype(np.int16)
    b = np.asarray(arr_b[:cp]).astype(np.int16)
    diff = np.abs(a - b)
    return {
        "n_compared": int(cp),
        "n_byte_identical": int(identical),
        "identical_fraction": float(identical / cp),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": int(diff.max()),
        "sample_shape": list(np.asarray(arr_a).shape[1:]),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"schema": "v431-recheck-r2/1", "question": (
        "does an identical `cand` prefix imply identical model inputs?"), "pairs": {},
        "ancestry": {}, "within_tape_near_duplication": {}}

    data: dict[str, dict] = {}
    for name, path in CORPORA.items():
        labels = load_labels(path)
        data[name] = {
            "labels": labels,
            "cand": np.array([int(r["cand"]) for r in labels], dtype=np.int64),
            "move": np.array([int(r["move"]) for r in labels], dtype=np.int64),
            "decision": np.array([int(r["decision"]) for r in labels], dtype=np.int64),
            "decision_frame": np.array([int(r["decision_frame"]) for r in labels], dtype=np.int64),
            "path": path,
        }
        for arr in OBS_ARRAYS:
            data[name][arr] = np.load(path / f"{arr}.npy", mmap_mode="r")

    # ---------------- Q3: provenance / ancestry ---------------- #
    for name, path in CORPORA.items():
        prov = json.loads((path / "labels_provenance.json").read_text(encoding="utf-8"))
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        out["ancestry"][name] = {
            "tape_path": prov.get("tape", {}).get("path"),
            "tape_sha256": prov.get("tape", {}).get("sha256"),
            "tape_slots": prov.get("tape", {}).get("slots"),
            "labels_total": prov.get("labels_total"),
            "supervised_heads": prov.get("supervised_heads"),
            "unsupervised_heads": prov.get("unsupervised_heads"),
            "shoot_distinct": prov.get("head_distribution", {}).get("shoot", {}).get("distinct"),
            "bomb_distinct": prov.get("head_distribution", {}).get("bomb", {}).get("distinct"),
            "meta_tape_keys": {k: meta.get(k) for k in meta if "tape" in str(k).lower()},
            "first_decision_frame": int(data[name]["decision_frame"][0]),
            "last_decision_frame": int(data[name]["decision_frame"][-1]),
            "decision_gap_values": sorted(
                {int(b - a) for a, b in zip(data[name]["decision"], data[name]["decision"][1:])}),
            "n_samples": int(len(data[name]["cand"])),
        }

    # ---------------- Q1 + Q2: per-pair comparison ---------------- #
    for a, b in itertools.combinations(CORPORA, 2):
        cp = common_prefix(data[a]["cand"], data[b]["cand"])
        pair = {
            "cand_common_prefix": int(cp),
            "fraction_of_shorter_tape": float(cp / min(len(data[a]["cand"]),
                                                      len(data[b]["cand"]))),
            "observations_over_prefix": {},
            "observations_over_suffix": {},
        }
        for arr in OBS_ARRAYS:
            pair["observations_over_prefix"][arr] = compare_prefix(
                data[a][arr], data[b][arr], cp)

        # suffix: same index alignment beyond the prefix, plus any-value duplication
        n = min(len(data[a]["cand"]), len(data[b]["cand"]))
        suf = range(cp, n)
        if len(suf) > 0:
            for arr in OBS_ARRAYS:
                ha = row_hashes(np.asarray(data[a][arr][cp:n]))
                hb = row_hashes(np.asarray(data[b][arr][cp:n]))
                same_pos = sum(1 for x, y in zip(ha, hb) if x == y)
                # cross-position duplication: any sample of A equal to ANY sample of B
                set_a, set_b = set(ha), set(hb)
                shared = set_a & set_b
                pair["observations_over_suffix"][arr] = {
                    "n_compared": int(len(suf)),
                    "n_identical_at_same_index": int(same_pos),
                    "n_distinct_values_A": len(set_a),
                    "n_distinct_values_B": len(set_b),
                    "n_shared_values_cross_position": len(shared),
                }
        out["pairs"][f"{a}__{b}"] = pair

    # ---------------- within-tape near duplication ---------------- #
    for name in CORPORA:
        p = data[name]["path"]
        pg = np.load(p / "play_gray.npy", mmap_mode="r")
        h = row_hashes(np.asarray(pg))
        out["within_tape_near_duplication"][name] = {
            "n_samples": len(h),
            "n_distinct_play_gray": len(set(h)),
            "n_exact_duplicate_rows": len(h) - len(set(h)),
            "decision_gap": "stride-6 sampling of one tape",
        }
        # adjacent-frame correlation: mean abs diff between sample i and i+1
        step = 64
        diffs = []
        for i in range(0, len(pg) - 1, step):
            diffs.append(float(np.abs(pg[i].astype(np.int16) - pg[i + 1].astype(np.int16)).mean()))
        out["within_tape_near_duplication"][name]["mean_abs_diff_adjacent_samples"] = (
            float(np.mean(diffs)) if diffs else None)

    # ---------------- random-pair control ---------------- #
    # mean abs diff between two random samples of the same tape, as a scale reference
    rng = np.random.default_rng(0)
    pg12 = data["seed12"]["play_gray"]
    vals = []
    for _ in range(64):
        i, j = rng.integers(0, len(pg12), 2)
        vals.append(float(np.abs(pg12[i].astype(np.int16) - pg12[j].astype(np.int16)).mean()))
    out["control_mean_abs_diff_random_pairs_seed12_play_gray"] = float(np.mean(vals))

    (OUT_DIR / "overlap_audit.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---------------- console ---------------- #
    print("== Q1/Q2 per pair ==")
    for key, pair in out["pairs"].items():
        print(f"{key}: cand common prefix = {pair['cand_common_prefix']} "
              f"({pair['fraction_of_shorter_tape']:.3%} of shorter tape)")
        for arr, st in pair["observations_over_prefix"].items():
            if st.get("n_compared"):
                print(f"    prefix {arr:22s} identical {st['n_byte_identical']:4d}/{st['n_compared']:<4d} "
                      f"({st['identical_fraction']:6.1%})  meanAbsDiff {st['mean_abs_diff']:7.3f}  "
                      f"max {st['max_abs_diff']}")
        for arr, st in pair["observations_over_suffix"].items():
            print(f"    suffix {arr:22s} same-index identical {st['n_identical_at_same_index']:4d}/"
                  f"{st['n_compared']:<4d}  shared-values-anywhere {st['n_shared_values_cross_position']}")
        print()
    print("== within-tape ==")
    for k, v in out["within_tape_near_duplication"].items():
        print(f"  {k}: distinct play_gray {v['n_distinct_play_gray']}/{v['n_samples']}  "
              f"exact dup rows {v['n_exact_duplicate_rows']}  "
              f"adjacent meanAbsDiff {v['mean_abs_diff_adjacent_samples']:.3f}")
    print(f"  control (random pairs, seed12): "
          f"{out['control_mean_abs_diff_random_pairs_seed12_play_gray']:.3f}")
    print()
    print("== ancestry ==")
    for k, v in out["ancestry"].items():
        print(f"  {k}: tape={v['tape_path']}")
        print(f"      sha={v['tape_sha256']}  slots={v['tape_slots']}  "
              f"frames [{v['first_decision_frame']}..{v['last_decision_frame']}]  "
              f"gap {v['decision_gap_values']}")
    print()
    print(f"wrote: {OUT_DIR / 'overlap_audit.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
