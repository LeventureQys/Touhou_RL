"""Reproducible engineering smoke; NOT pixels, NOT a gameplay benchmark."""
from pathlib import Path
import json
import numpy as np
import torch

from touhou_rl.macro_train import file_sha, train
from touhou_rl.vmodels import load_vision_checkpoint
from touhou_rl.macro_policy import MacroPolicy, load_macro_checkpoint


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "Document/Update/Alpha/v3.7.2/evidence"
    out.mkdir(parents=True, exist_ok=True)
    baseline = root / "third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt"
    torch.manual_seed(7)
    backbone, _ = load_vision_checkpoint(baseline, map_location="cpu")
    features = torch.randn(64, backbone.config.trunk_hidden)
    with torch.no_grad():
        logits = backbone.policy_head(features)
        wrapped = MacroPolicy(backbone).forward_features(features, logits)["policy_logits"]
    assert torch.equal(wrapped, logits)
    # A deliberately artificial target tests only gradients, persistence and CLI.
    target = (logits.argmax(-1).numpy() + 3) % 18
    positive = np.zeros((64,18), bool)
    positive[np.arange(64), target] = True
    path = out / "synthetic-feature-smoke.npz"
    if path.exists():
        raise FileExistsError(path)
    np.savez(path, features=features.numpy(), base_logits=logits.numpy(), witness_mask=positive,
             route_target=np.zeros((64,2),np.float32), valid=np.ones(64,bool),
             route_valid=np.zeros(64,bool), groups=np.array([f"artificial-{i//8}" for i in range(64)]),
             metadata=np.array(json.dumps({"schema":"v372-macro-cache/1", "source_sha256":file_sha(baseline),
               "input_kind":"synthetic_test_features", "label_semantics":"positive_verified_route_witnesses",
               "warning":"ARTIFICIAL labels; not produced by route teacher; engineering smoke ONLY"})))
    report = train(path, baseline, out / "synthetic-feature-training", epochs=3, batch_size=16, seed=7)
    restored, metadata = load_macro_checkpoint(report["best_checkpoint"], baseline_checkpoint=baseline)
    with torch.no_grad():
        predictions = restored.forward_features(features, logits)["policy_logits"]
    result = {"schema":"v372-engineering-smoke/1", "base_initialisation_exact":True,
              "sidecar_roundtrip_finite":bool(torch.isfinite(predictions).all()),
              "trained_on_real_pixels":False, "macro_gameplay_effect":"not_run",
              "warning":"This artifact must NOT be deployed as a trained gameplay policy.",
              "training_report":str(out / "synthetic-feature-training/report.json")}
    (out / "engineering-smoke.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
