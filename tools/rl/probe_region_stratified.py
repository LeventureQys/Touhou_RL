"""T1c: does "inside a large danger region" survive conditioning on clearance?

The three previous REGION failures all compared two correlated features directly,
which lets the stronger one swamp the weaker.  This probe asks the conditional
question instead: *within* a clearance decile, can the region score still tell a
60-frame death from a non-death?

Pre-registered in 执行计划-T1c补遗.md.  Uses the cached D1 primitives, so nothing is
re-extracted and the numbers are tied to the committed feature cache.

    python tools/rl/probe_region_stratified.py --features <features-D1.npz> --out <json>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from touhou_rl.ablate36 import binary_auc, spearman


def _bin_edges(bin_width: float, lo: float = -3.0, hi: float = 60.0) -> np.ndarray:
    """Fixed-width bin edges covering the clearance range (inclusive of the cap)."""
    count = int(round((hi - lo) / bin_width))
    return lo + bin_width * np.arange(count + 1)


def _merge_small_bins(baseline: np.ndarray, edges: np.ndarray, min_samples: int) -> np.ndarray:
    """Widen edges until every bin holds at least ``min_samples`` rows.

    Merging keeps the conditioning honest on the low tail, where a 3 px bin can
    hold only a handful of decisions; a bin with too few samples cannot be
    checked for being clearance-flat, so it must not be reported as a stratum.
    """
    edges = list(np.asarray(edges, dtype=np.float64))
    while len(edges) > 2:
        counts = np.histogram(baseline, bins=np.asarray(edges))[0]
        smallest = int(np.argmin(counts))
        if counts[smallest] >= min_samples:
            break
        if smallest == 0:
            del edges[1]
        elif smallest == len(counts) - 1:
            del edges[-2]
        elif edges[smallest] - edges[smallest - 1] <= edges[smallest + 2] - edges[smallest + 1]:
            del edges[smallest]
        else:
            del edges[smallest + 1]
    return np.asarray(edges, dtype=np.float64)


def _stratified_auc(
    score: np.ndarray,
    baseline: np.ndarray,
    label: np.ndarray,
    groups: np.ndarray,
    deciles: int = 10,
    train_mask: np.ndarray | None = None,
) -> dict:
    """Weighted mean of within-decile AUCs, with decile cut points from train.

    ``score`` is the region score, ``baseline`` the clearance (the stratifier).
    Returns the weighted AUC, the per-decile rows, and the within-stratum
    baseline AUC (which must be ~0.5 or the stratification failed).
    """
    score = np.asarray(score, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    train_mask = np.ones(label.shape[0], dtype=bool) if train_mask is None else train_mask

    valid = np.isfinite(score) & np.isfinite(baseline) & np.isfinite(label)
    train_valid = valid & train_mask
    if train_valid.sum() < deciles * 10:
        return {"auc": None, "rows": [], "reason": "too few training samples"}
    cuts = np.quantile(baseline[train_valid], np.linspace(0.0, 1.0, deciles + 1)[1:-1])
    buckets = np.digitize(baseline, cuts)

    rows = []
    weighted = 0.0
    weight = 0.0
    base_weighted = 0.0
    positives_total = 0
    positives_used = 0
    for bucket in range(deciles):
        mask = valid & (buckets == bucket)
        n = int(mask.sum())
        positives = int((label[mask] > 0.5).sum())
        positives_total += positives
        if n == 0:
            continue
        auc = binary_auc(label[mask], score[mask])
        base_auc = binary_auc(label[mask], -baseline[mask])
        rows.append(
            {
                "decile": bucket,
                "n": n,
                "positives": positives,
                "auc": auc,
                "baseline_auc": base_auc,
            }
        )
        if auc is None:
            continue
        positives_used += positives
        weighted += auc * positives
        weight += positives
        if base_auc is not None:
            base_weighted += base_auc * positives
    return {
        "auc": (weighted / weight) if weight else None,
        "baseline_auc": (base_weighted / weight) if weight else None,
        "rows": rows,
        "positives_total": positives_total,
        "positives_used": positives_used,
        "weight": weight,
    }


def _stratified_auc_width(
    score: np.ndarray,
    baseline: np.ndarray,
    label: np.ndarray,
    bin_width: float,
    min_samples: int = 200,
    train_mask: np.ndarray | None = None,
) -> dict:
    """Within-fixed-width-bin AUC with a per-bin clearance-flatness check.

    A bin is *kept* only when its own baseline AUC is <= 0.55, i.e. when the
    conditioning actually removed clearance.  Dropped bins and their positives are
    reported so the caller can apply the pre-registered >20% invalidation rule.
    """
    score = np.asarray(score, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    train_mask = np.ones(label.shape[0], dtype=bool) if train_mask is None else train_mask
    valid = np.isfinite(score) & np.isfinite(baseline) & np.isfinite(label)

    edges = _bin_edges(bin_width)
    # Widths come from the training fold only; the test fold is only *assigned*.
    edges = _merge_small_bins(baseline[valid & train_mask], edges, min_samples)
    buckets = np.digitize(baseline, edges) - 1
    n_bins = len(edges) - 1

    rows = []
    weighted = 0.0
    weight = 0.0
    base_weighted = 0.0
    dropped_positives = 0
    positives_total = 0
    for bucket in range(n_bins):
        mask = valid & (buckets == bucket)
        n = int(mask.sum())
        positives = int((label[mask] > 0.5).sum())
        positives_total += positives
        if n == 0:
            continue
        auc = binary_auc(label[mask], score[mask])
        base_auc = binary_auc(label[mask], -baseline[mask])
        kept = base_auc is not None and base_auc <= 0.55 and auc is not None
        rows.append(
            {
                "bin": bucket,
                "lo": float(edges[bucket]),
                "hi": float(edges[bucket + 1]),
                "n": n,
                "positives": positives,
                "auc": auc,
                "baseline_auc": base_auc,
                "kept": bool(kept),
            }
        )
        if not kept:
            dropped_positives += positives
            continue
        weighted += auc * positives
        weight += positives
        base_weighted += base_auc * positives
    return {
        "auc": (weighted / weight) if weight else None,
        "baseline_auc": (base_weighted / weight) if weight else None,
        "rows": rows,
        "positives_total": positives_total,
        "dropped_positives": dropped_positives,
        "dropped_fraction": (dropped_positives / positives_total) if positives_total else None,
        "weight": weight,
        "edges": [float(edge) for edge in edges],
    }


def loeo_stratified_fixed_width(score, baseline, label, groups, bin_width: float, min_samples: int = 200) -> dict:
    """LOEO version of the fixed-width stratified AUC."""
    weighted_sum = 0.0
    weight_sum = 0.0
    base_sum = 0.0
    base_weight = 0.0
    dropped = 0
    positives_total = 0
    per_fold = []
    for group in np.unique(groups):
        result = _stratified_auc_width(
            score, baseline, label, bin_width, min_samples, train_mask=(groups != group)
        )
        per_fold.append(
            {
                "held_out": str(group),
                "auc": result["auc"],
                "baseline_auc": result["baseline_auc"],
                "kept_positives": result["weight"],
                "dropped_fraction": result["dropped_fraction"],
            }
        )
        if result["auc"] is not None and result["weight"]:
            weighted_sum += result["auc"] * result["weight"]
            weight_sum += result["weight"]
        if result["baseline_auc"] is not None and result["weight"]:
            base_sum += result["baseline_auc"] * result["weight"]
            base_weight += result["weight"]
        dropped += int(result["dropped_positives"])
        positives_total += int(result["positives_total"])
    return {
        "auc": weighted_sum / weight_sum if weight_sum else None,
        "baseline_auc": base_sum / base_weight if base_weight else None,
        "per_fold": per_fold,
        "dropped_positives": dropped,
        "positives_total": positives_total // max(1, len(np.unique(groups))),
        "dropped_fraction": (dropped / positives_total) if positives_total else None,
        "kept_weight": weight_sum,
    }


def loeo_stratified(score, baseline, label, groups, deciles: int = 10) -> dict:
    """Leave-one-episode-out stratified AUC (deciles from the training folds)."""
    preds = np.full(label.shape[0], np.nan)  # per-row within-decile rank score
    per_fold = []
    weighted_sum = 0.0
    weight_sum = 0.0
    base_sum = 0.0
    base_weight = 0.0
    for group in np.unique(groups):
        test = groups == group
        train = ~test
        result = _stratified_auc(score, baseline, label, groups, deciles, train_mask=train)
        per_fold.append({"held_out": str(group), **{k: result.get(k) for k in ("auc", "baseline_auc", "positives_used")}})
        if result["auc"] is not None and result["weight"]:
            weighted_sum += result["auc"] * result["weight"]
            weight_sum += result["weight"]
        if result["baseline_auc"] is not None and result["weight"]:
            base_sum += result["baseline_auc"] * result["weight"]
            base_weight += result["weight"]
    return {
        "auc": weighted_sum / weight_sum if weight_sum else None,
        "baseline_auc": base_sum / base_weight if base_weight else None,
        "per_fold": per_fold,
    }


def permutation_p(
    score,
    baseline,
    label,
    groups,
    observed,
    permutations: int,
    seed: int,
    stratify: str = "decile",
    bin_width: float = 3.0,
    min_samples: int = 200,
) -> dict:
    rng = np.random.default_rng(seed)
    negatives = []
    for _ in range(permutations):
        shuffled = score.copy()
        for group in np.unique(groups):
            rows = np.nonzero(groups == group)[0]
            shuffled[rows] = score[rows[rng.permutation(rows.size)]]
        if stratify == "width":
            value = loeo_stratified_fixed_width(shuffled, baseline, label, groups, bin_width, min_samples)["auc"]
        else:
            value = loeo_stratified(shuffled, baseline, label, groups)["auc"]
        if value is not None:
            negatives.append(value)
    if not negatives:
        return {"p": None, "null_mean": None, "permutations": 0}
    negatives = np.asarray(negatives)
    p = (int((negatives >= observed).sum()) + 1) / (negatives.size + 1)
    return {"p": float(p), "null_mean": float(negatives.mean()), "permutations": int(negatives.size)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/rl/probe_region_stratified.py")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stratify", choices=["decile", "width"], default="width",
                        help="'decile' is T1c (invalidated by the clearance cap atom); "
                             "'width' is T1d fixed-width bins with per-bin validation")
    parser.add_argument("--bin-width", type=float, default=3.0)
    parser.add_argument("--min-bin-samples", type=int, default=200)
    parser.add_argument("--deciles", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3610)
    args = parser.parse_args(argv)

    payload = np.load(args.features, allow_pickle=True)
    clearance = payload["prim_clearance"].astype(np.float64)
    region = payload["prim_region20"].astype(np.float64)
    label = payload["label_death_60"].astype(np.float64)
    groups = payload["groups"].astype(np.int64)

    started = time.perf_counter()
    if args.stratify == "width":
        result = loeo_stratified_fixed_width(
            region, clearance, label, groups, args.bin_width, args.min_bin_samples
        )
        perm = permutation_p(
            region, clearance, label, groups, result["auc"], args.permutations, args.seed,
            stratify="width", bin_width=args.bin_width, min_samples=args.min_bin_samples,
        )
    else:
        result = loeo_stratified(region, clearance, label, groups, args.deciles)
        perm = permutation_p(region, clearance, label, groups, result["auc"], args.permutations, args.seed)

    # Unstratified reference on the same rows: the direct comparison that failed
    # three times before, so the report can show both numbers side by side.
    valid = np.isfinite(region) & np.isfinite(label)
    direct = binary_auc(label[valid], region[valid])
    direct_baseline = binary_auc(label[valid], -clearance[valid])

    dropped_fraction = result.get("dropped_fraction") if args.stratify == "width" else None
    if result["baseline_auc"] is None or result["auc"] is None:
        verdict = "invalid"
        reason = "stratified AUC undefined"
    elif dropped_fraction is not None and dropped_fraction > 0.20:
        verdict = "invalid"
        reason = f"{dropped_fraction:.1%} of positives fell in bins that were not clearance-flat"
    elif result["baseline_auc"] > 0.55:
        verdict = "invalid"
        reason = f"stratification failed: within-stratum baseline AUC {result['baseline_auc']:.4f} > 0.55"
    elif result["auc"] >= 0.60 and perm.get("p") is not None and perm["p"] < 0.05:
        verdict = "rescued"
        reason = "within-stratum region AUC >= 0.60 with p < 0.05"
    elif result["auc"] >= 0.55:
        verdict = "weak"
        reason = "within-stratum region AUC in [0.55, 0.60) or not significant"
    else:
        verdict = "dead"
        reason = "within-stratum region AUC < 0.55"

    bin_detail = None
    if args.stratify == "width":
        # Diagnostic only: the same binning on all data, so a reader can see which
        # strata were dropped and how many deaths they held.  Never used for the
        # verdict, which is always the LOEO aggregate.
        detail = _stratified_auc_width(
            region, clearance, label, args.bin_width, args.min_bin_samples
        )
        bin_detail = {
            "rows": detail["rows"],
            "dropped_positives": detail["dropped_positives"],
            "dropped_fraction": detail["dropped_fraction"],
            "positives_total": detail["positives_total"],
        }

    report = {
        "schema": "v360-region-stratified/1",
        "stratifier": args.stratify,
        "bin_detail_full_data": bin_detail,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "features": str(args.features),
        "config": {
            "deciles": args.deciles,
            "bin_width_px": args.bin_width,
            "min_bin_samples": args.min_bin_samples,
            "dropped_positive_fraction": dropped_fraction,
            "permutations": args.permutations,
            "seed": args.seed,
            "stratifier": "prim_clearance (current local clearance)",
            "score": "prim_region20 (player-connected danger region area, 8 px grid, t=20)",
            "label": "death_60",
            "cv": "leave-one-episode-out",
        },
        "totals": {
            "rows": int(label.shape[0]),
            "valid_rows": int(valid.sum()),
            "positives": int((label[valid] > 0.5).sum()),
        },
        "direct_comparison": {"region_auc": direct, "clearance_auc": direct_baseline},
        "stratified": result,
        "permutation": perm,
        "verdict": verdict,
        "reason": reason,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("verdict", "reason")}, ensure_ascii=False, indent=1))
    print(
        json.dumps(
            {
                "direct_region_auc": direct,
                "direct_clearance_auc": direct_baseline,
                "stratified_region_auc": result["auc"],
                "stratified_clearance_auc": result["baseline_auc"],
                "permutation_p": perm.get("p"),
                "positives": report["totals"]["positives"],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"elapsed {time.perf_counter() - started:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
