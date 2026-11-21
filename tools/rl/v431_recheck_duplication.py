"""v4.3.1 recheck R3: quantify near-duplication vs literal duplication.

R2 showed that the shared `cand` prefix is NOT literal duplication (0/219
byte-identical frames) but IS near-duplication (mean abs pixel diff ~0.6/255
against a random-pair control of ~39/255). This script measures the structure of
that gap and the random-split optimism, so the documentation can state what was
actually true instead of "20% literal training data".

Measurements
------------
A. Label identity over the prefix (cand/move/focus compared sample by sample).
B. Pixel difference structure: what fraction of pixels differ at all, and at
   thresholds 0/1/8/32; which image regions carry the difference.
C. Random-split optimism: for each seed12 validation sample, the distance to its
   temporally nearest training sample (stride-6 grid => +/-6 frames) compared with
   the distance to a random training sample.
D. Cross-tape: for each seed10 sample in the prefix, distance to its seed12
   same-index twin; and the same at a random offset.

Writes only new files under Document/Update/Alpha/v4.3.1/evidence/R3/.
"""

from __future__ import annotations

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
OUT_DIR = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R3"


def load_labels(p: pathlib.Path) -> list[dict]:
    return [json.loads(ln) for ln in
            (p / "labels.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]


def common_prefix(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def pixel_structure(x: np.ndarray, y: np.ndarray) -> dict:
    a, b = x.astype(np.int16), y.astype(np.int16)
    diff = np.abs(a - b)
    total = diff.size
    out = {
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": int(diff.max()),
        "pixels_differing_at_all": float((diff > 0).mean()),
        "pixels_off_by_ge_1": float((diff >= 1).mean()),
        "pixels_off_by_ge_8": float((diff >= 8).mean()),
        "pixels_off_by_ge_32": float((diff >= 32).mean()),
    }
    # region breakdown on the last (most recent) frame channel set: 3x3 grid
    fr = diff.reshape(diff.shape[0], -1)
    return out


def grid_regions(x: np.ndarray, y: np.ndarray, grid: int = 3) -> list[float]:
    a, b = x.astype(np.int16), y.astype(np.int16)
    diff = np.abs(a - b)
    if diff.ndim == 3:      # (C,H,W)
        h, w = diff.shape[1], diff.shape[2]
        vals = []
        for i in range(grid):
            for j in range(grid):
                blk = diff[:, i * h // grid:(i + 1) * h // grid,
                           j * w // grid:(j + 1) * w // grid]
                vals.append(float(blk.mean()))
        return vals
    return []


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"schema": "v431-recheck-r3/1"}

    d: dict[str, dict] = {}
    for name, path in CORPORA.items():
        labels = load_labels(path)
        d[name] = {
            "labels": labels,
            "cand": np.array([int(r["cand"]) for r in labels], dtype=np.int64),
            "move": np.array([int(r["move"]) for r in labels], dtype=np.int64),
            "focus": np.array([int(r["focus"]) for r in labels], dtype=np.int64),
            "pg": np.load(path / "play_gray.npy", mmap_mode="r"),
            "rgb": np.load(path / "global_rgb.npy", mmap_mode="r"),
        }

    # ---- A. label identity over the prefix ---- #
    label_identity = {}
    for a, b in itertools.combinations(CORPORA, 2):
        cp = common_prefix(d[a]["cand"], d[b]["cand"])
        label_identity[f"{a}__{b}"] = {
            "prefix": int(cp),
            "move_identical": int(sum(d[a]["move"][:cp] == d[b]["move"][:cp])),
            "focus_identical": int(sum(d[a]["focus"][:cp] == d[b]["focus"][:cp])),
            "cand_identical": int(cp),
            "suffix_move_identical": int(sum(
                d[a]["move"][cp:] == d[b]["move"][cp:min(cp + 0, 0) or None][:0].shape[0]
                if False else d[a]["move"][cp:len(d[b]["move"])] == d[b]["move"][cp:len(d[a]["move"])]
            )) if False else None,
        }
        # explicit suffix agreement, aligned by index
        n = min(len(d[a]["move"]), len(d[b]["move"]))
        label_identity[f"{a}__{b}"]["suffix_move_identical"] = int(
            sum(d[a]["move"][cp:n] == d[b]["move"][cp:n]))
        label_identity[f"{a}__{b}"]["suffix_n"] = int(n - cp)
    out["label_identity"] = label_identity

    # ---- B. pixel difference structure over the prefix ---- #
    pix = {}
    for a, b in itertools.combinations(CORPORA, 2):
        cp = common_prefix(d[a]["cand"], d[b]["cand"])
        per_sample = [pixel_structure(d[a]["pg"][i], d[b]["pg"][i]) for i in range(cp)]
        agg = {k: float(np.mean([s[k] for s in per_sample])) for k in per_sample[0]}
        agg["n_samples"] = cp
        agg["regions_3x3_last_channel"] = grid_regions(
            d[a]["pg"][0], d[b]["pg"][0])
        agg["samples_with_zero_pixels_differing"] = (
            float(np.mean([s["pixels_differing_at_all"] == 0.0 for s in per_sample])))
        pix[f"{a}__{b}"] = agg
    out["prefix_pixel_structure_play_gray"] = pix

    # ---- C. random-split optimism inside seed12 ---- #
    n = len(d["seed12"]["move"])
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    cut = int(n * 0.8)
    tr_i, va_i = np.sort(perm[:cut]), np.sort(perm[cut:])
    tr_set = set(int(x) for x in tr_i)

    def nearest_train(idx: int) -> int | None:
        for off in range(1, 400):
            for cand_i in (idx - off, idx + off):
                if cand_i in tr_set:
                    return cand_i
        return None

    sample_va = list(va_i)[::5]
    twin_d, rand_d = [], []
    for iv in sample_va:
        jt = nearest_train(int(iv))
        if jt is None:
            continue
        twin_d.append(float(np.abs(d["seed12"]["pg"][iv].astype(np.int16) -
                                   d["seed12"]["pg"][jt].astype(np.int16)).mean()))
        jr = int(rng.integers(0, n))
        rand_d.append(float(np.abs(d["seed12"]["pg"][iv].astype(np.int16) -
                                   d["seed12"]["pg"][jr].astype(np.int16)).mean()))
    out["random_split_optimism"] = {
        "n_val": int(len(va_i)),
        "n_val_sampled": int(len(sample_va)),
        "val_to_nearest_train_sample_mean_abs_diff": float(np.mean(twin_d)),
        "val_to_random_train_sample_mean_abs_diff": float(np.mean(rand_d)),
        "ratio": float(np.mean(twin_d) / np.mean(rand_d)) if np.mean(rand_d) else None,
        "note": ("stride-6 grid: the temporally nearest sample is 6 frames away, so a "
                 "random split puts every validation sample next to a near twin"),
    }

    # ---- D. cross-tape prefix twins vs suffix ---- #
    cp = common_prefix(d["seed12"]["cand"], d["seed10"]["cand"])
    twin = [float(np.abs(d["seed12"]["pg"][i].astype(np.int16) -
                         d["seed10"]["pg"][i].astype(np.int16)).mean()) for i in range(cp)]
    rng2 = np.random.default_rng(1)
    ctrl = [float(np.abs(d["seed12"]["pg"][i].astype(np.int16) -
                         d["seed10"]["pg"][int(rng2.integers(0, len(d["seed10"]["cand"])))]
                         .astype(np.int16)).mean()) for i in range(cp)]
    out["cross_tape_distance"] = {
        "pair": "seed12__seed10",
        "prefix_n": int(cp),
        "prefix_same_index_mean_abs_diff": float(np.mean(twin)),
        "control_random_pair_mean_abs_diff": float(np.mean(ctrl)),
        "min_over_prefix": float(np.min(twin)),
        "max_over_prefix": float(np.max(twin)),
    }

    (OUT_DIR / "duplication_structure.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- console ---- #
    print("== A. label identity over the cand prefix ==")
    for k, v in out["label_identity"].items():
        print(f"  {k}: prefix {v['prefix']}  move identical {v['move_identical']}/{v['prefix']}  "
              f"focus identical {v['focus_identical']}/{v['prefix']}")
        print(f"      suffix: move identical {v['suffix_move_identical']}/{v['suffix_n']}")
    print()
    print("== B. pixel structure over the prefix (play_gray) ==")
    for k, v in out["prefix_pixel_structure_play_gray"].items():
        print(f"  {k}: n={v['n_samples']}  meanAbsDiff={v['mean_abs_diff']:.3f}  "
              f"pixels differing at all={v['pixels_differing_at_all']:.2%}  "
              f">=8: {v['pixels_off_by_ge_8']:.2%}  >=32: {v['pixels_off_by_ge_32']:.2%}")
        print(f"      samples with zero differing pixels: {v['samples_with_zero_pixels_differing']:.1%}")
    print()
    print("== C. random-split optimism inside seed12 ==")
    r = out["random_split_optimism"]
    print(f"  n_val={r['n_val']} sampled={r['n_val_sampled']}")
    print(f"  val -> temporally nearest train : {r['val_to_nearest_train_sample_mean_abs_diff']:.3f}")
    print(f"  val -> random train             : {r['val_to_random_train_sample_mean_abs_diff']:.3f}")
    print(f"  ratio (near twin / random)      : {r['ratio']:.3f}")
    print()
    print("== D. cross-tape distance ==")
    c = out["cross_tape_distance"]
    print(f"  prefix same-index {c['prefix_same_index_mean_abs_diff']:.3f}  "
          f"(min {c['min_over_prefix']:.3f} max {c['max_over_prefix']:.3f})  "
          f"control {c['control_random_pair_mean_abs_diff']:.3f}")
    print()
    print(f"wrote: {OUT_DIR / 'duplication_structure.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
