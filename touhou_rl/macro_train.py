"""v3.7.2 grouped, positive-witness macro-head training on frozen vision features.

Usage: python -m touhou_rl.macro_train --cache paired.npz --baseline vision.pt
       --out new-directory --epochs 30

No game access. Cache contract is deliberately strict: [N,D] features extracted
from the named baseline's pixels, [N,18] base_logits/witness_mask, [N,2] route_target
(normalised displacement), bool [N] valid/route_valid, unicode [N] groups and a
scalar JSON metadata with schema/source_sha256/input_kind/label_semantics.
Unknown routes are NOT negative examples. Template variants must share a group.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SCHEMA = "v372-macro-cache/1"


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(path, baseline):
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "base_logits", "witness_mask", "route_target", "valid", "route_valid", "groups", "metadata"}
        if not required.issubset(archive.files):
            raise ValueError(f"missing cache fields: {sorted(required - set(archive.files))}")
        data = {key: archive[key].copy() for key in required}
        if "local_valid" in archive.files:
            data["local_valid"] = archive["local_valid"].copy()
    meta = json.loads(str(data.pop("metadata").item()))
    if meta.get("schema") != SCHEMA or meta.get("source_sha256") != file_sha(baseline):
        raise ValueError("cache schema or baseline identity mismatch")
    if meta.get("input_kind") not in ("real_pixels", "synthetic_pixels", "synthetic_test_features"):
        raise ValueError("explicit input_kind required")
    if meta.get("label_semantics") != "positive_verified_route_witnesses":
        raise ValueError("beam failure must not be used as negative safety labels")
    features = data["features"]
    if features.ndim != 2 or not len(features) or features.dtype != np.float32:
        raise ValueError("features must be nonempty float32 [N,D]")
    n = len(features)
    for key, shape, dtype in (("base_logits", (n,18), np.float32), ("route_target", (n,2), np.float32),
                              ("witness_mask", (n,18), np.bool_), ("valid", (n,), np.bool_), ("route_valid", (n,), np.bool_)):
        if data[key].shape != shape or data[key].dtype != dtype:
            raise ValueError(f"invalid {key} shape/dtype")
    for key in ("features", "base_logits", "route_target"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"nonfinite {key}")
    if data["groups"].shape != (n,) or data["groups"].dtype.kind not in "US" or any(not str(v) for v in data["groups"]):
        raise ValueError("nonempty string groups required")
    if np.any(data["valid"] & ~data["witness_mask"].any(1)):
        raise ValueError("valid sample has no positive route witness")
    if np.any(data["route_valid"] & ~data["valid"]):
        raise ValueError("route_valid requires a valid witness sample")
    if "local_valid" not in data:
        data["local_valid"] = np.zeros(n, dtype=bool)
    if data["local_valid"].shape != (n,) or data["local_valid"].dtype != np.bool_:
        raise ValueError("local_valid must be bool [N]")
    if np.any(data["local_valid"] & data["valid"]):
        raise ValueError("local regression and macro correction rows must not overlap")
    return data, meta


def grouped_split(groups, seed=7, validation_fraction=.25):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("at least two independent episode/template groups required")
    ordered = np.random.default_rng(seed).permutation(unique)
    count = min(len(unique)-1, max(1, int(round(len(unique)*validation_fraction))))
    val = np.isin(groups, ordered[:count])
    return ~val, val


def metrics(logits, witnesses, valid):
    valid = valid.bool()
    if not valid.any():
        return {"n": 0, "witness_agreement": None}
    chosen = logits.argmax(-1)
    agreement = witnesses.gather(1, chosen[:,None]).squeeze(1)
    return {"n": int(valid.sum()), "witness_agreement": float(agreement[valid].float().mean())}


def train(cache, baseline, out, *, epochs=30, batch_size=128, lr=1e-3, seed=7, device="cpu"):
    from .macro_policy import MacroPolicy, save_macro_checkpoint, macro_loss
    from .vmodels import load_vision_checkpoint, vision_arch

    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("positive epochs, batch_size and learning rate required")
    out = Path(out)
    if out.exists():
        raise FileExistsError("use a new output directory; never overwrite a run")
    data, meta = load_cache(cache, baseline)
    train_mask, val_mask = grouped_split(data["groups"], seed)
    if not data["valid"][train_mask].any() or not data["valid"][val_mask].any():
        raise ValueError("both group splits require valid route witnesses")
    torch.manual_seed(seed)
    backbone, _ = load_vision_checkpoint(baseline, map_location=device)
    if vision_arch(backbone) != "v3-st":
        raise ValueError("macro training requires a v3-st vision backbone")
    policy = MacroPolicy(backbone).to(device)
    # Cached base_logits are checked against this backbone's policy head. A
    # matching SHA alone cannot prevent an accidentally mismatched feature cache.
    arrays = {key: torch.from_numpy(data[key]).to(device) for key in data if key != "groups"}
    if arrays["features"].shape[1] != backbone.config.trunk_hidden:
        raise ValueError("feature dimension differs from baseline")
    with torch.no_grad():
        expected = backbone.policy_head(arrays["features"])
        if not torch.allclose(expected, arrays["base_logits"], atol=1e-5, rtol=1e-5):
            raise ValueError("cached base logits do not match baseline features")
    optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=lr)
    indices = torch.from_numpy(np.flatnonzero(train_mask)).to(device)
    val_indices = torch.from_numpy(np.flatnonzero(val_mask)).to(device)
    history = []
    best = -float("inf")
    out.mkdir(parents=True)
    for epoch in range(epochs):
        policy.train()
        shuffled = indices[torch.randperm(len(indices), device=device)]
        losses = []
        for batch in shuffled.split(batch_size):
            supervised = arrays["valid"][batch] | arrays["route_valid"][batch] | arrays["local_valid"][batch]
            if not bool(supervised.any()):
                # Adam momentum can move weights even with an exactly zero
                # gradient; unknown-only batches must not call step at all.
                continue
            output = policy.forward_features(arrays["features"][batch], arrays["base_logits"][batch])
            loss = macro_loss(output, arrays["witness_mask"][batch], arrays["valid"][batch],
                              route_target=arrays["route_target"][batch], route_valid=arrays["route_valid"][batch],
                              base_logits=arrays["base_logits"][batch],
                              local_valid=arrays["local_valid"][batch], kl_weight=1.0)
            if isinstance(loss, dict):
                loss = loss["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        policy.eval()
        with torch.no_grad():
            pred = policy.forward_features(arrays["features"][val_indices], arrays["base_logits"][val_indices])
            current = metrics(pred["policy_logits"], arrays["witness_mask"][val_indices], arrays["valid"][val_indices])
        row = {"epoch":epoch+1, "train_loss":float(np.mean(losses)), "validation":current}
        history.append(row)
        if current["witness_agreement"] > best:
            best = current["witness_agreement"]
            # Keep immutable epoch files; best filename is reported, not rewritten.
            best_path = out / f"epoch-{epoch+1:03d}.pt"
            save_macro_checkpoint(best_path, policy, baseline, metadata={"cache_metadata":meta, "seed":seed,
                "epoch":epoch+1, "real_pixel_training":meta["input_kind"] == "real_pixels",
                "live_effect":"not_run", "capability":"unvalidated"})
    with torch.no_grad():
        base_metric = metrics(arrays["base_logits"][val_indices], arrays["witness_mask"][val_indices], arrays["valid"][val_indices])
    report = {"schema":"v372-macro-training/1", "cache":str(cache), "cache_sha256":file_sha(cache),
              "baseline":str(baseline), "baseline_sha256":file_sha(baseline), "metadata":meta,
              "train_groups":sorted(set(data["groups"][train_mask].tolist())),
              "validation_groups":sorted(set(data["groups"][val_mask].tolist())),
              "baseline_validation":base_metric, "best_validation_witness_agreement":best,
              "best_checkpoint":str(best_path), "history":history,
              "warning":"Witness agreement is not survival or proof of safety; no independent test or live evaluation performed."}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = train(args.cache, args.baseline, args.out, epochs=args.epochs, batch_size=args.batch_size,
                   lr=args.lr, seed=args.seed, device=args.device)
    print(json.dumps({k:v for k,v in result.items() if k != "history"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
