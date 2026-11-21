"""v4.3.1 recheck R1: recover the original BC split and recompute baselines properly.

Why this script exists
----------------------
`Document/Update/Alpha/v4.3.0/ROUTE_REVIEW_2026-11-15.md` states that the original
model used an "880 train / 219 validation" split and that the documented "baseline
CE" was `-log(468/1099) = 0.8537`. This script does two things, and nothing else:

  1. Recovers the split that the original triage scripts actually used, by
     re-executing their exact split expression against the exact corpora.
  2. Recomputes the "baseline" under several *explicitly stated* definitions, all
     of them cross-entropy of a fixed proposal distribution q over the classes,
     fitted on the TRAINING subset only and evaluated on the held-out subset.

Definitions used (all pre-registered here, before running):

  train_mle      q_c = n_train_c / n_train          (unseen class -> +inf loss)
  train_laplace  q_c = (n_train_c + 1) / (n_train + K)
  uniform        q_c = 1 / K
  train_entropy  H(q) on the training subset, i.e. CE of q on itself
  majority_acc   accuracy of the deterministic argmax(q) predictor
                 (its CE is +inf on every non-majority sample, so accuracy is the
                  only meaningful read-out)
  old_formula    -log(max_c q_c)  -- reproduced ONLY to show it is not a CE

K = number of classes in the head (move=9, focus/shoot/bomb=2).

Read-only with respect to every existing asset: loads the O2 corpora via mmap and
writes new files only under Document/Update/Alpha/v4.3.1/evidence/R1/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
HEADS = {"move": 9, "focus": 2, "shoot": 2, "bomb": 2}
OUT_DIR = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R1"


def load_labels(corpus: pathlib.Path) -> list[dict]:
    text = (corpus / "labels.jsonl").read_text(encoding="utf-8")
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# The three split expressions found in .tmp_v430_triage/*.py, reproduced verbatim
# --------------------------------------------------------------------------- #
def split_int_0p8(n: int) -> tuple[np.ndarray, np.ndarray]:
    """`small_model_test.py`:62-65, `small_model_test_d4.py`:62-65,
    `cross_tape_honest.py`:67-70, `cross_tape_generalisation.py`:95-98.

    rng = np.random.default_rng(0); perm = rng.permutation(n)
    cut = int(n * 0.8);  tr = perm[:cut];  va = perm[cut:]
    """
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    cut = int(n * 0.8)
    return perm[:cut], perm[cut:]


def split_fixed_200(n: int) -> tuple[np.ndarray, np.ndarray]:
    """`bc_generalisation.py`:31-34 -- n_val = 200 hard-coded."""
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    return order[200:], order[:200]


def split_random_200(n: int) -> tuple[np.ndarray, np.ndarray]:
    """`bc_split_comparison.py`:95-96 -- random arm, val = first 200 of perm."""
    perm = np.random.default_rng(0).permutation(n)
    return perm[200:], perm[:200]


# --------------------------------------------------------------------------- #
# Baseline arithmetic
# --------------------------------------------------------------------------- #
def counts(y: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(y, minlength=k).astype(np.int64)


def ce_of_q(q: np.ndarray, y: np.ndarray) -> float:
    """-mean(log q[y]); returns inf if any evaluated sample lands on q==0."""
    qy = q[y]
    if np.any(qy <= 0.0):
        return math.inf
    return float(-np.mean(np.log(qy)))


def baseline_block(y_train: np.ndarray, y_eval: np.ndarray, k: int) -> dict:
    c_tr = counts(y_train, k)
    c_ev = counts(y_eval, k)
    n_tr = int(c_tr.sum())
    n_ev = int(c_ev.sum())

    q_mle = c_tr / n_tr
    q_lap = (c_tr + 1.0) / (n_tr + k)
    q_uni = np.full(k, 1.0 / k)

    unseen = [int(y) for y in np.flatnonzero(c_tr == 0)]
    unseen_hits = int(sum(c_ev[y] for y in unseen))
    top = int(np.argmax(c_tr))

    return {
        "k": k,
        "n_train": n_tr,
        "n_eval": n_ev,
        "train_counts": c_tr.tolist(),
        "eval_counts": c_ev.tolist(),
        "train_share": (c_tr / n_tr).tolist(),
        "eval_share": (c_ev / n_ev).tolist(),
        "classes_absent_in_train": unseen,
        "eval_samples_in_absent_classes": unseen_hits,
        "train_majority_class": top,
        "train_majority_share": float(q_mle[top]),
        "ce_train_mle": ce_of_q(q_mle, y_eval),
        "ce_train_laplace": ce_of_q(q_lap, y_eval),
        "ce_uniform": ce_of_q(q_uni, y_eval),
        "ce_train_entropy": float(-np.sum(q_mle[q_mle > 0] * np.log(q_mle[q_mle > 0]))),
        "majority_accuracy": float(np.mean(y_eval == top)),
        "old_formula_neg_log_train_majority": float(-math.log(q_mle[top])),
        "old_formula_neg_log_full_majority": (
            float(-math.log(counts(np.concatenate([y_train, y_eval]), k).max() /
                            (n_tr + n_ev)))
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "baseline_recheck.json"))
    args = ap.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "schema": "v431-recheck-r1/1",
        "generated_at_utc": None,
        "note": (
            "R1 recovers the original split and recomputes baselines. It trains no "
            "model and makes no capability claim."
        ),
        "corpora": {},
        "split_recovery": {},
        "documented_vs_recomputed": {},
    }

    import datetime as _dt
    report["generated_at_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()

    # ---- 1. corpus identity + histograms ---------------------------------- #
    for name, path in CORPORA.items():
        if not (path / "labels.jsonl").exists():
            report["corpora"][name] = {"present": False, "path": str(path)}
            continue
        labels = load_labels(path)
        y_move = np.array([int(r["move"]) for r in labels], dtype=np.int64)
        entry = {
            "present": True,
            "path": str(path.relative_to(REPO)),
            "n_samples": len(labels),
            "labels_sha256": sha256_file(path / "labels.jsonl"),
            "move_histogram": counts(y_move, 9).tolist(),
            "focus_histogram": counts(np.array([int(r["focus"]) for r in labels]), 2).tolist(),
            "shoot_histogram": counts(np.array([int(r["shoot"]) for r in labels]), 2).tolist(),
            "bomb_histogram": counts(np.array([int(r["bomb"]) for r in labels]), 2).tolist(),
            "tape_provenance_values": sorted({str(r.get("tape_provenance")) for r in labels}),
            "is_blank_count": int(sum(1 for r in labels if r.get("is_blank"))),
            "decision_index_min": int(min(r["decision"] for r in labels)),
            "decision_index_max": int(max(r["decision"] for r in labels)),
            "decision_index_unique": len({int(r["decision"]) for r in labels}),
            "decision_frame_min": int(min(r["decision_frame"] for r in labels)),
            "decision_frame_max": int(max(r["decision_frame"] for r in labels)),
            "decision_gap_unique": sorted(
                {int(b["decision"] - a["decision"]) for a, b in zip(labels, labels[1:])}
            ),
            "label_keys": sorted(labels[0].keys()),
        }
        report["corpora"][name] = entry

    seed12 = CORPORA["seed12"]
    labels12 = load_labels(seed12)
    n = len(labels12)
    y12 = np.array([int(r["move"]) for r in labels12], dtype=np.int64)

    # ---- 2. split recovery ------------------------------------------------ #
    tr_a, va_a = split_int_0p8(n)
    tr_b, va_b = split_fixed_200(n)
    tr_c, va_c = split_random_200(n)

    def digest(idx: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(idx, dtype=np.int64).tobytes()).hexdigest()

    report["split_recovery"] = {
        "n_total": n,
        "claimed_in_route_review": "880 train / 219 validation",
        "evaluated": {
            "int_0p8": {
                "source": "small_model_test.py:64, small_model_test_d4.py:64, "
                          "cross_tape_honest.py:69, cross_tape_generalisation.py:97",
                "n_train": int(len(tr_a)),
                "n_val": int(len(va_a)),
                "cut_value": int(n * 0.8),
                "train_index_sha256": digest(tr_a),
                "val_index_sha256": digest(va_a),
                "val_class_counts": counts(y12[va_a], 9).tolist(),
                "train_class_counts": counts(y12[tr_a], 9).tolist(),
                "matches_claimed_880_219": bool(len(tr_a) == 880 and len(va_a) == 219),
            },
            "fixed_200": {
                "source": "bc_generalisation.py:33",
                "n_train": int(len(tr_b)),
                "n_val": int(len(va_b)),
                "matches_claimed_880_219": bool(len(tr_b) == 880 and len(va_b) == 219),
            },
            "random_200": {
                "source": "bc_split_comparison.py:95-96 (random arm)",
                "n_train": int(len(tr_c)),
                "n_val": int(len(va_c)),
                "matches_claimed_880_219": bool(len(tr_c) == 880 and len(va_c) == 219),
            },
        },
        "arithmetic": {
            "int(1099*0.8)": int(1099 * 0.8),
            "1099 - int(1099*0.8)": 1099 - int(1099 * 0.8),
            "1099 - 219": 1099 - 219,
            "note": "no expression in the triage scripts yields 880/219",
        },
    }

    # ---- 3. baselines on the recovered split ------------------------------ #
    blocks = {}
    for head, k in HEADS.items():
        yh = np.array([int(r[head]) for r in labels12], dtype=np.int64)
        blocks[f"seed12_val__{head}"] = baseline_block(yh[tr_a], yh[va_a], k)

    # cross-tape: fit on seed12 train, evaluate on the whole seed10 tape
    seed10 = CORPORA["seed10"]
    suffix_info = {}
    if (seed10 / "labels.jsonl").exists():
        labels10 = load_labels(seed10)
        y10 = np.array([int(r["move"]) for r in labels10], dtype=np.int64)
        cand12 = np.array([int(r["cand"]) for r in labels12], dtype=np.int64)
        cand10 = np.array([int(r["cand"]) for r in labels10], dtype=np.int64)
        cp = 0
        while cp < min(len(cand12), len(cand10)) and cand12[cp] == cand10[cp]:
            cp += 1
        suffix_info = {
            "common_cand_prefix_len": int(cp),
            "seed10_n": int(len(y10)),
            "seed10_suffix_n": int(len(y10) - cp),
            "seed12_n_minus_prefix": int(len(y12) - cp),
            "fraction_of_tape": float(cp / len(y10)),
        }
        blocks["seed10_full__move"] = baseline_block(y12[tr_a], y10, 9)
        blocks["seed10_suffix__move"] = baseline_block(y12[tr_a], y10[cp:], 9)

    report["baselines"] = blocks
    report["cross_tape_prefix"] = suffix_info

    # ---- 4. documented vs recomputed -------------------------------------- #
    full_hist = counts(y12, 9)
    p_full = full_hist / full_hist.sum()
    p10_full = (
        counts(np.array([int(r["move"]) for r in load_labels(seed10)]), 9)
        if (seed10 / "labels.jsonl").exists() else None
    )
    recomputed = {
        "seed12_full_histogram": full_hist.tolist(),
        "seed12_full_n": int(full_hist.sum()),
        "neg_log_full_majority": float(-math.log(p_full.max())),
        "full_seed12_entropy": float(-np.sum(p_full[p_full > 0] * np.log(p_full[p_full > 0]))),
    }
    if p10_full is not None:
        r10 = p10_full / p10_full.sum()
        p12 = p_full[p_full > 0]
        h = 0.0
        for c in range(9):
            if r10[c] <= 0:
                continue
            if p_full[c] <= 0:
                h = math.inf
                break
            h += r10[c] * math.log(p_full[c])
        recomputed["seed12_full_prior_on_seed10_full"] = float(-h)

    report["documented_vs_recomputed"] = {
        "documented": {
            "old_formula_neg_log_468_over_1099": 0.8537,
            "route_review_full_seed12_prior_on_seed12": 1.685989,
            "route_review_full_seed12_prior_on_seed10": 1.707497,
            "uniform_9_class_ce": float(math.log(9)),
            "split": "880 / 219",
        },
        "recomputed": recomputed,
        "verdict": {
            "old_formula_is_not_a_cross_entropy": True,
            "full_corpus_histogram_includes_validation_labels": True,
        },
    }

    out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- console summary -------------------------------------------------- #
    print(f"corpora loaded: {[k for k, v in report['corpora'].items() if v['present']]}")
    print(f"seed12 n = {n}")
    print()
    print("== split recovery ==")
    for key, blk in report["split_recovery"]["evaluated"].items():
        print(f"  {key:12s} n_train={blk['n_train']:5d} n_val={blk['n_val']:4d} "
              f"matches 880/219: {blk['matches_claimed_880_219']}")
    print(f"  claimed in ROUTE_REVIEW: {report['split_recovery']['claimed_in_route_review']}")
    print()
    print("== move-head baselines on the recovered hold-out (train-only fitting) ==")
    hdr = f"{'evaluation set':22s} {'n':>5s} {'train_mle':>10s} {'laplace':>9s} " \
          f"{'uniform':>8s} {'H(train)':>9s} {'maj_acc':>8s} {'old -log':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for key, blk in report["baselines"].items():
        if not key.endswith("__move"):
            continue
        print(f"{key[:22]:22s} {blk['n_eval']:5d} {blk['ce_train_mle']:10.4f} "
              f"{blk['ce_train_laplace']:9.4f} {blk['ce_uniform']:8.4f} "
              f"{blk['ce_train_entropy']:9.4f} {blk['majority_accuracy']:8.4f} "
              f"{blk['old_formula_neg_log_train_majority']:9.4f}")
    print()
    print("== cross-tape common prefix (cand level) ==")
    print(f"  {suffix_info}")
    print()
    print("== documented numbers re-derived ==")
    for k, v in recomputed.items():
        print(f"  {k} = {v}")
    print()
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
