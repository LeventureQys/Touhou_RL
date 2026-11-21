"""v4.3.1 recheck R5: is the 16x model's CE advantage over the prior real?

R4 showed the 30,793-parameter arm scoring 0.067-0.128 nats below a baseline fitted on
the training fold, on five held-out sets. A mean difference is not evidence, so this
script re-trains that arm exactly as R4 does, keeps the per-sample cross-entropy, and
runs a paired bootstrap on the per-sample difference (baseline CE - model CE).
Positive means the model is better.

Paired, not two-sample: both quantities are computed on the same sample, so the pairing
removes the label-mix variance that dominates the raw CE.

Writes only new files under Document/Update/Alpha/v4.3.1/evidence/R5/.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = pathlib.Path(__file__).resolve().parents[2]
O2 = REPO / ".tmp_v430_o2"
CORPORA = {
    "seed12": O2 / "o2-strided",
    "seed10": O2 / "o2-stride-seed10",
    "seed11": O2 / "o2-stride-seed11",
}
OUT_DIR = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R5"
DOWN = 16
K = 9


def load_corpus(path: pathlib.Path) -> dict:
    labels = [json.loads(ln) for ln in
              (path / "labels.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    pg = np.load(path / "play_gray.npy")
    hg = np.load(path / "hud_gray.npy")
    n, c, h, w = pg.shape
    pg_s = pg.reshape(n, c, h // DOWN, DOWN, w // DOWN, DOWN).mean(axis=(3, 5))
    kh, kw = hg.shape[2] // DOWN, hg.shape[3] // DOWN
    hg_s = hg[:, :, : kh * DOWN, : kw * DOWN].reshape(n, 1, kh, DOWN, kw, DOWN).mean(axis=(3, 5))
    x = np.concatenate([pg_s.reshape(n, -1), hg_s.reshape(n, -1)],
                       axis=1).astype(np.float32) / 255.0
    return {
        "x": x,
        "y": np.array([int(r["move"]) for r in labels], dtype=np.int64),
        "cand": np.array([int(r["cand"]) for r in labels], dtype=np.int64),
    }


class Small(nn.Module):
    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, 9))

    def forward(self, x):
        return self.net(x)


def per_sample_ce(model, x: torch.Tensor, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 256):
            chunk = idx[s:s + 256]
            out.append(torch.nn.functional.cross_entropy(
                model(x[torch.from_numpy(np.asarray(chunk, dtype=np.int64))]),
                torch.from_numpy(y[chunk]), reduction="none").numpy())
    return np.concatenate(out)


def bootstrap(d: np.ndarray, n_boot: int = 20000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n": int(len(d)),
        "mean_delta": float(d.mean()),
        "median_delta": float(np.median(d)),
        "se": float(d.std(ddof=1) / math.sqrt(len(d))),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "frac_bootstrap_gt_0": float(np.mean(means > 0)),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_samples_model_better": int((d > 0).sum()),
        "n_samples_baseline_better": int((d < 0).sum()),
        "n_ties": int((d == 0).sum()),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tr = load_corpus(CORPORA["seed12"])
    n = len(tr["y"])
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    cut = int(n * 0.8)
    tr_i, va_i = perm[:cut], perm[cut:]

    torch.manual_seed(0)
    model = Small(tr["x"].shape[1])
    print(f"n_train={len(tr_i)} n_val={len(va_i)} "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    x12 = torch.from_numpy(tr["x"])
    y12 = torch.from_numpy(tr["y"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss()
    for epoch in range(40):
        model.train()
        order = rng.permutation(len(tr_i))
        for s in range(0, len(order), 64):
            idx = tr_i[order[s:s + 64]]
            if len(idx) < 2:
                continue
            loss = lossf(model(x12[idx]), y12[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 9:
            print(f"  epoch {epoch} done", flush=True)

    # baseline proposal from the training fold only
    c_tr = np.bincount(tr["y"][tr_i], minlength=K).astype(np.float64)
    q = c_tr / c_tr.sum()
    lp = np.log(np.maximum(q, 1e-300))

    sets = {"seed12_val": (tr["x"], tr["y"], va_i)}
    c12 = tr["cand"]
    for name in ("seed10", "seed11"):
        d = load_corpus(CORPORA[name])
        cp = int(next((i for i, (a, b) in enumerate(zip(c12, d["cand"])) if a != b),
                      min(len(c12), len(d["cand"]))))
        sets[f"{name}_full"] = (d["x"], d["y"], np.arange(len(d["y"])))
        sets[f"{name}_suffix"] = (d["x"], d["y"], np.arange(cp, len(d["y"])))
        print(f"  {name} common prefix {cp}", flush=True)

    out = {"schema": "v431-recheck-r5/1", "arm": "down16 (30,793 params)",
           "n_boot": 20000, "split": {"n_train": int(len(tr_i)), "n_val": int(len(va_i))},
           "results": {}}

    for sname, (xs, ys, idx) in sets.items():
        mc = per_sample_ce(model, torch.from_numpy(np.ascontiguousarray(xs)), ys, idx)
        base = -lp[ys[idx]]
        d = base - mc
        res = bootstrap(d)
        res["model_ce"] = float(mc.mean())
        res["baseline_ce"] = float(base.mean())
        res["model_accuracy"] = None
        out["results"][sname] = res
        print(f"  {sname:14s} model {res['model_ce']:.4f}  base {res['baseline_ce']:.4f}  "
              f"delta {res['mean_delta']:+.4f}  "
              f"CI [{res['ci95_low']:+.4f}, {res['ci95_high']:+.4f}]  "
              f"excl0={res['excludes_zero']}", flush=True)

    # accuracy separately (needs argmax)
    for sname, (xs, ys, idx) in sets.items():
        with torch.no_grad():
            lg = model(torch.from_numpy(np.ascontiguousarray(xs))[
                torch.from_numpy(np.asarray(idx, dtype=np.int64))])
            acc = float((lg.argmax(1).numpy() == ys[idx]).mean())
        top = int(np.argmax(c_tr))
        mask = c_tr >= 0
        # conditional-accuracy baseline: how often the majority class alone is right
        out["results"][sname]["model_accuracy"] = acc
        out["results"][sname]["majority_accuracy"] = float(np.mean(ys[idx] == top))

    (OUT_DIR / "significance.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print(f"{'set':14s} {'n':>5s} {'model':>8s} {'baseline':>9s} {'delta':>8s} "
          f"{'95% CI':>22s} {'excl 0':>7s} {'acc':>7s} {'maj':>7s}")
    print("-" * 92)
    for sname, r in out["results"].items():
        print(f"{sname:14s} {r['n']:5d} {r['model_ce']:8.4f} {r['baseline_ce']:9.4f} "
              f"{r['mean_delta']:+8.4f} [{r['ci95_low']:+9.4f},{r['ci95_high']:+9.4f}] "
              f"{str(r['excludes_zero']):>7s} {r['model_accuracy']:7.4f} "
              f"{r['majority_accuracy']:7.4f}")
    print()
    print(f"wrote: {OUT_DIR / 'significance.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
