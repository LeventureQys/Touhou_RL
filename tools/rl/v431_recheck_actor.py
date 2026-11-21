"""v4.3.1 recheck R6: re-execute the 2,726,671-parameter ActorPolicy BC arm.

This reproduces `python .tmp_v430_triage/cross_tape_generalisation.py 25` — the run
whose output survives in `.tmp_v430_triage/cross_tape_generalisation.json` and which
supplied the 0.0293 / 1.8807 / 1.8837 numbers quoted in the v4.3.0 documents.

New in this run:
  * both heads (move, focus) are scored against a baseline fitted on the TRAINING
    fold only, instead of `-log(global majority share)`;
  * the seed10 and seed11 suffixes (common `cand` prefix removed) are scored too;
  * per-sample cross-entropy is retained so a paired bootstrap can say whether any
    difference from the baseline is larger than sampling noise.

No checkpoint is saved (the original saved none either); the JSON is the artifact.

Writes only new files under Document/Update/Alpha/v4.3.1/evidence/R6/.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

from touhou_rl.v430.learn import bc_loss
from touhou_rl.v430.models import ActorPolicy

REPO = pathlib.Path(__file__).resolve().parents[2]
O2 = REPO / ".tmp_v430_o2"
SEED12 = O2 / "o2-strided"
SEED10 = O2 / "o2-stride-seed10"
SEED11 = O2 / "o2-stride-seed11"
OUT_DIR = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R6"
HEADS = {"move": 9, "focus": 2, "shoot": 2, "bomb": 2}
EPOCHS = 25
ARRAYS = ("global_rgb", "play_gray", "hud_gray", "executed_key_history",
          "history_valid", "elapsed_capture_s")


def load(corpus: pathlib.Path) -> dict:
    labels = [json.loads(ln) for ln in
              (corpus / "labels.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = {"labels": labels}
    for key in ARRAYS:
        out[key] = np.load(corpus / f"{key}.npy", mmap_mode="r")
    out["cand"] = np.array([int(r["cand"]) for r in labels], dtype=np.int64)
    return out


def batch_of(data: dict, idx: np.ndarray) -> dict:
    obs = {k: torch.from_numpy(np.ascontiguousarray(data[k][idx])) for k in ARRAYS}
    obs["history_valid"] = obs["history_valid"].bool()
    obs["elapsed_capture_s"] = obs["elapsed_capture_s"].float()
    tgt = {h: torch.tensor([int(data["labels"][i][h]) for i in idx], dtype=torch.long)
           for h in HEADS}
    return {"observations": obs, "targets": tgt}


def evaluate(model, data: dict, idx: np.ndarray, bs: int = 128) -> dict:
    """Original method (batch-mean) plus per-sample CE for paired tests."""
    model.eval()
    totals = {h: 0.0 for h in HEADS}
    nb = 0
    per_sample = {h: [] for h in HEADS}
    correct = {h: 0 for h in HEADS}
    n_seen = 0
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            chunk = idx[s:s + bs]
            b = batch_of(data, chunk)
            logits, _ = model(b["observations"])
            loss = bc_loss(logits, b["targets"])
            for h in HEADS:
                totals[h] += float(loss.per_head[h])
            nb += 1
            for h, k in HEADS.items():
                lg = logits[h].reshape(-1, k)
                y = b["targets"][h]
                per_sample[h].append(F.cross_entropy(lg, y, reduction="none").numpy())
                correct[h] += int((lg.argmax(1) == y).sum())
                if h == "move":
                    n_seen += int(len(y))
    return {
        "batch_mean_ce": {h: v / nb for h, v in totals.items()},
        "sample_mean_ce": {h: float(np.concatenate(per_sample[h]).mean()) for h in HEADS},
        "accuracy": {h: correct[h] / n_seen for h in HEADS},
        "per_sample_ce": {h: np.concatenate(per_sample[h]) for h in HEADS},
        "n": int(n_seen),
    }


def train_baseline(y_tr: np.ndarray, y_ev: np.ndarray, k: int) -> dict:
    c = np.bincount(y_tr, minlength=k).astype(np.float64)
    q = c / c.sum()
    q_lap = (c + 1.0) / (c.sum() + k)
    per_sample = np.where(q[y_ev] > 0, -np.log(np.maximum(q[y_ev], 1e-300)), np.inf)
    top = int(np.argmax(c))
    return {
        "train_counts": c.tolist(),
        "q": q.tolist(),
        "ce_train_mle": float(per_sample.mean()),
        "ce_train_laplace": float(-np.log(q_lap[y_ev]).mean()),
        "ce_uniform": math.log(k),
        "majority_class": top,
        "majority_accuracy": float(np.mean(y_ev == top)),
        "old_formula": float(-math.log(q[top])),
        "per_sample_ce": per_sample,
    }


def paired_bootstrap(model_ce: np.ndarray, base_ce: np.ndarray,
                     n_boot: int = 10000, seed: int = 0) -> dict:
    """Bootstrap CI for mean(base - model); positive means the model is better."""
    d = base_ce - model_ce
    finite = np.isfinite(d)
    if not finite.all():
        d = d[finite]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {
        "n": int(len(d)),
        "mean_delta": float(d.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "p_model_better": float(np.mean(means > 0)),
        "significant": bool(np.percentile(means, 2.5) > 0
                            or np.percentile(means, 97.5) < 0),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {
        "schema": "v431-recheck-r6/1",
        "identity": ("NEW experiment identity v431-R6: re-execution of "
                     "`cross_tape_generalisation.py 25`, which saved no checkpoint."),
        "epochs": EPOCHS,
        "batch_size": 50,
        "lr": 1e-4,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "documented": {"train_move": 0.0293, "val_move": 1.8807, "seed10_move": 1.8837,
                       "val_focus": 0.5589, "seed10_focus": 0.6461},
        "reported_baselines": {"seed12_old": 0.8537, "seed10_old": 0.8558},
    }

    train = load(SEED12)
    n = len(train["labels"])
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    cut = int(n * 0.8)
    tr_idx, va_idx = perm[:cut], perm[cut:]
    out["split"] = {"n_total": int(n), "n_train": int(len(tr_idx)),
                    "n_val": int(len(va_idx)),
                    "train_move_counts": np.bincount(
                        np.array([int(train["labels"][i]["move"]) for i in tr_idx],
                                 dtype=np.int64), minlength=9).tolist()}
    print(f"train {len(tr_idx)} val {len(va_idx)}", flush=True)

    torch.manual_seed(0)
    model = ActorPolicy()
    out["n_params"] = int(sum(p.numel() for p in model.parameters()))
    print(f"params {out['n_params']:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(len(tr_idx))
        for s in range(0, len(order), 50):
            chunk = tr_idx[order[s:s + 50]]
            if len(chunk) < 2:
                continue
            b = batch_of(train, chunk)
            logits, _ = model(b["observations"])
            loss = bc_loss(logits, b["targets"])
            opt.zero_grad()
            loss.total.backward()
            opt.step()
        print(f"  epoch {epoch} done", flush=True)

    sets: dict[str, tuple[dict, np.ndarray]] = {
        "seed12_train": (train, tr_idx),
        "seed12_val": (train, va_idx),
    }
    c12 = train["cand"]
    temp_loaded = []
    for name, path in (("seed10", SEED10), ("seed11", SEED11)):
        d = load(path)
        temp_loaded.append(d)
        cp = int(next((i for i, (a, b) in enumerate(zip(c12, d["cand"])) if a != b),
                      min(len(c12), len(d["cand"]))))
        out.setdefault("common_prefix", {})[name] = cp
        sets[f"{name}_full"] = (d, np.arange(len(d["labels"])))
        sets[f"{name}_suffix"] = (d, np.arange(cp, len(d["labels"])))

    out["evaluations"] = {}
    for set_name, (data, idx) in sets.items():
        got = evaluate(model, data, idx)
        entry = {
            "n": got["n"],
            "batch_mean_ce": got["batch_mean_ce"],
            "sample_mean_ce": got["sample_mean_ce"],
            "accuracy": got["accuracy"],
            "heads": {},
        }
        y_tr = np.array([int(train["labels"][i]["move"]) for i in tr_idx], dtype=np.int64)
        for head, k in HEADS.items():
            y_ev = np.array([int(data["labels"][i][head]) for i in idx], dtype=np.int64)
            y_trh = np.array([int(train["labels"][i][head]) for i in tr_idx], dtype=np.int64)
            base = train_baseline(y_trh, y_ev, k)
            mc = got["per_sample_ce"][head]
            entry["heads"][head] = {
                "model_ce": float(mc.mean()),
                "model_accuracy": got["accuracy"][head],
                "baseline_ce_train_mle": base["ce_train_mle"],
                "baseline_ce_laplace": base["ce_train_laplace"],
                "baseline_ce_uniform": base["ce_uniform"],
                "baseline_majority_class": base["majority_class"],
                "baseline_majority_accuracy": base["majority_accuracy"],
                "old_formula": base["old_formula"],
                "delta_vs_train_mle": float(base["ce_train_mle"] - mc.mean()),
                "bootstrap": paired_bootstrap(mc, base["per_sample_ce"]),
            }
        out["evaluations"][set_name] = entry
        print(f"  evaluated {set_name} (n={got['n']})", flush=True)

    del temp_loaded
    (OUT_DIR / "actor_recheck.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print("== move / focus vs TRAIN-FIT baseline ==")
    hdr = (f"{'set':15s} {'n':>5s} | {'move CE':>8s} {'base':>7s} {'d +CI':>18s} "
           f"{'acc':>6s} {'maj':>6s} | {'focus CE':>9s} {'base':>7s} {'d +CI':>18s}")
    print(hdr)
    print("-" * len(hdr))
    for set_name, ev in out["evaluations"].items():
        m, f = ev["heads"]["move"], ev["heads"]["focus"]
        print(f"{set_name:15s} {ev['n']:5d} | {m['model_ce']:8.4f} "
              f"{m['baseline_ce_train_mle']:7.4f} "
              f"{m['delta_vs_train_mle']:+7.4f}[{m['bootstrap']['ci95_low']:+6.4f},"
              f"{m['bootstrap']['ci95_high']:+6.4f}] "
              f"{m['model_accuracy']:6.4f} {m['baseline_majority_accuracy']:6.4f} | "
              f"{f['model_ce']:9.4f} {f['baseline_ce_train_mle']:7.4f} "
              f"{f['delta_vs_train_mle']:+7.4f}[{f['bootstrap']['ci95_low']:+6.4f},"
              f"{f['bootstrap']['ci95_high']:+6.4f}]")
    print()
    print(f"wrote: {OUT_DIR / 'actor_recheck.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
