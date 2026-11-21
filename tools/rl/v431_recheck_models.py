"""v4.3.1 recheck R4: re-execute the two small-model BC arms under a NEW identity.

Provenance note (important)
---------------------------
No checkpoint and no per-sample prediction from the v4.3.0 BC experiments exists on
disk (verified independently by two separate audits; every 2,726,671-parameter file
found is a pytest fixture tagged `extra.note='synthetic'`, and `.tmp_v430_triage`
contains no `.log`, no `.pt`, no prediction array). The reported numbers
1.5306 / 1.6020 / 2.1299 / 2.1161 existed only in a terminal.

Therefore this script is a RE-EXECUTION under a new experiment identity, not a
replay of stored predictions. It re-implements `small_model_test.py` (DOWN=16) and
`small_model_test_d4.py` (DOWN=4) statement for statement, including their RNG use:

    rng = np.random.default_rng(0); perm = rng.permutation(n)   # n = 1099 -> cut = 879
    tr_i, va_i = perm[:cut], perm[cut:]
    torch.manual_seed(0)
    for epoch in range(40):
        order = rng.permutation(len(tr_i))                       # same rng continues

and evaluates against baselines fitted on the TRAINING subset only.

If the reproduced validation CE does not match the reported 1.5306 / 2.1299, the
difference is itself evidence that the reported numbers cannot be checked.

Writes only new files under Document/Update/Alpha/v4.3.1/evidence/R4/.
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
OUT_DIR = REPO / "Document" / "Update" / "Alpha" / "v4.3.1" / "evidence" / "R4"
ARMS = {
    "down16": {"down": 16, "reported_params": 30793,
               "reported": {"val_ce": 1.5306, "seed10_ce": 1.6020}},
    "down4": {"down": 4, "reported_params": 487753,
              "reported": {"val_ce": 2.1299, "seed10_ce": 2.1161}},
}
K = 9


def load_corpus(path: pathlib.Path, down: int) -> dict:
    """Reproduces `load()` from small_model_test.py verbatim."""
    labels = [json.loads(ln) for ln in
              (path / "labels.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    pg = np.load(path / "play_gray.npy")
    hg = np.load(path / "hud_gray.npy")
    n, c, h, w = pg.shape
    pg_s = pg.reshape(n, c, h // down, down, w // down, down).mean(axis=(3, 5))
    kh, kw = hg.shape[2] // down, hg.shape[3] // down
    hg_s = hg[:, :, : kh * down, : kw * down].reshape(n, 1, kh, down, kw, down).mean(axis=(3, 5))
    x = np.concatenate([pg_s.reshape(n, -1), hg_s.reshape(n, -1)],
                       axis=1).astype(np.float32) / 255.0
    y = np.array([int(r["move"]) for r in labels], dtype=np.int64)
    cand = np.array([int(r["cand"]) for r in labels], dtype=np.int64)
    return {"x": x, "y": y, "cand": cand}


class Small(nn.Module):
    """Verbatim from small_model_test.py."""

    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32), nn.ReLU(),
            nn.Linear(32, 9),
        )

    def forward(self, x):
        return self.net(x)


def baseline_block(y_train: np.ndarray, y_eval: np.ndarray) -> dict:
    c_tr = np.bincount(y_train, minlength=K).astype(np.float64)
    n_tr = c_tr.sum()
    q = c_tr / n_tr
    q_lap = (c_tr + 1.0) / (n_tr + K)

    def ce(dist):
        vals = dist[y_eval]
        return math.inf if np.any(vals <= 0) else float(-np.mean(np.log(vals)))

    top = int(np.argmax(c_tr))
    return {
        "n_train": int(n_tr),
        "n_eval": int(len(y_eval)),
        "ce_train_mle": ce(q),
        "ce_train_laplace": ce(q_lap),
        "ce_uniform": math.log(K),
        "majority_class": top,
        "majority_accuracy": float(np.mean(y_eval == top)),
        "old_formula": float(-math.log(q[top])),
    }


def score(model, x, y: np.ndarray, idx: np.ndarray) -> dict:
    lossf = nn.CrossEntropyLoss()
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.ascontiguousarray(x))
    model.eval()
    with torch.no_grad():
        logits = model(x[torch.from_numpy(np.asarray(idx, dtype=np.int64))])
        ce = float(lossf(logits, torch.from_numpy(y[idx])))
        pred = logits.argmax(1).numpy()
    yy = y[idx]
    per_class = {}
    for c in range(K):
        m = yy == c
        if m.any():
            per_class[str(c)] = {
                "n": int(m.sum()),
                "recall": float(np.mean(pred[m] == c)),
            }
    return {
        "ce": ce,
        "accuracy": float(np.mean(pred == yy)),
        "prediction_histogram": np.bincount(pred, minlength=K).tolist(),
        "per_class": per_class,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {
        "schema": "v431-recheck-r4/1",
        "identity": (
            "NEW experiment identity v431-R4. This is a re-execution of the v4.3.0 "
            "scripts, not a replay of stored predictions: no checkpoint and no "
            "prediction array from the original runs exists on disk."
        ),
        "split": {},
        "arms": {},
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "torch_threads": torch.get_num_threads(),
    }

    corpus_cache: dict[str, dict] = {}
    for name, path in CORPORA.items():
        corpus_cache[name] = {"path": path, "labels_loaded": False}

    for arm_name, arm in ARMS.items():
        down = arm["down"]
        print(f"===== arm {arm_name} (DOWN={down}) =====", flush=True)
        tr = load_corpus(CORPORA["seed12"], down)
        n = len(tr["y"])
        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        cut = int(n * 0.8)
        tr_i, va_i = perm[:cut], perm[cut:]
        out["split"] = {
            "n_total": int(n),
            "cut": int(cut),
            "n_train": int(len(tr_i)),
            "n_val": int(len(va_i)),
            "val_class_counts": np.bincount(tr["y"][va_i], minlength=K).tolist(),
            "train_class_counts": np.bincount(tr["y"][tr_i], minlength=K).tolist(),
        }

        torch.manual_seed(0)
        model = Small(tr["x"].shape[1])
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  n_train={len(tr_i)} n_val={len(va_i)} n_params={n_params:,} "
              f"(reported {arm['reported_params']:,})", flush=True)

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
                print(f"    epoch {epoch} done", flush=True)

        arm_out: dict = {
            "down": down,
            "n_params": int(n_params),
            "reported_params": arm["reported_params"],
            "reported": arm["reported"],
            "evaluations": {},
        }

        # ---- evaluation sets ---- #
        evalsets: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
        evalsets["seed12_train"] = (tr["x"], tr["y"], tr_i)
        evalsets["seed12_val"] = (tr["x"], tr["y"], va_i)

        cp10 = 0
        te10 = load_corpus(CORPORA["seed10"], down)
        c12 = tr["cand"]
        cp10 = int(next((i for i, (a, b) in enumerate(zip(c12, te10["cand"])) if a != b),
                        min(len(c12), len(te10["cand"]))))
        evalsets["seed10_full"] = (te10["x"], te10["y"], np.arange(len(te10["y"])))
        evalsets["seed10_suffix"] = (te10["x"], te10["y"], np.arange(cp10, len(te10["y"])))

        te11 = load_corpus(CORPORA["seed11"], down)
        cp11 = int(next((i for i, (a, b) in enumerate(zip(c12, te11["cand"])) if a != b),
                        min(len(c12), len(te11["cand"]))))
        evalsets["seed11_full"] = (te11["x"], te11["y"], np.arange(len(te11["y"])))
        evalsets["seed11_suffix"] = (te11["x"], te11["y"], np.arange(cp11, len(te11["y"])))

        arm_out["common_prefix"] = {"seed10": cp10, "seed11": cp11}

        for set_name, (xs, ys, idx) in evalsets.items():
            got = score(model, xs, ys, idx)
            base = baseline_block(tr["y"][tr_i], ys[idx])
            arm_out["evaluations"][set_name] = {
                **got,
                "baseline": base,
                "beats_train_mle_baseline": bool(got["ce"] < base["ce_train_mle"]),
                "margin_vs_train_mle": float(base["ce_train_mle"] - got["ce"]),
                "beats_majority_accuracy": bool(got["accuracy"] > base["majority_accuracy"]),
            }

        out["arms"][arm_name] = arm_out
        del te10, te11
        print("  evaluated", list(arm_out["evaluations"]), flush=True)

    (OUT_DIR / "model_recheck.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print("== split ==")
    print(f"  n={out['split']['n_total']} cut={out['split']['cut']} "
          f"train={out['split']['n_train']} val={out['split']['n_val']}")
    print()
    hdr = (f"{'arm':7s} {'evaluation set':16s} {'n':>5s} {'model CE':>9s} "
           f"{'base MLE':>9s} {'base lap':>9s} {'model acc':>10s} {'maj acc':>8s} "
           f"{'verdict':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for arm_name, aw in out["arms"].items():
        for set_name, ev in aw["evaluations"].items():
            print(f"{arm_name:7s} {set_name:16s} {ev['baseline']['n_eval']:5d} "
                  f"{ev['ce']:9.4f} {ev['baseline']['ce_train_mle']:9.4f} "
                  f"{ev['baseline']['ce_train_laplace']:9.4f} "
                  f"{ev['accuracy']:10.4f} {ev['baseline']['majority_accuracy']:8.4f} "
                  f"{'BEATS' if ev['beats_train_mle_baseline'] else 'no':>10s}")
    print()
    print(f"params: " + ", ".join(f"{k}={v['n_params']:,}" for k, v in out["arms"].items()))
    print(f"wrote: {OUT_DIR / 'model_recheck.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
