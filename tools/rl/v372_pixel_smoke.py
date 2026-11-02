"""Real pixel forward smoke only: no labels, no training, no game input."""
from pathlib import Path
import json
import torch

from touhou_rl import vision
from touhou_rl.vision_data import VisionDataset
from touhou_rl.vmodels import load_vision_checkpoint
from touhou_rl.macro_policy import MacroPolicy
from touhou_rl.macro_train import file_sha


def main():
    baseline = Path("third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt")
    dataset = Path("third_party/th08/build/rl/datasets/pilot-ud0059-v2")
    model, _ = load_vision_checkpoint(baseline, map_location="cpu")
    indices = [0, 3, 7]
    with VisionDataset(dataset, max_cached_shards=1, history_contract=model.config.history_contract,
                       use_decode_cache=False, decode_threads=1) as data:
        rows = data.stack_indices(indices, model.config.depth)
        flat = rows.reshape(-1).tolist()
        regions = data.region_frames(flat, model.config.region_sizes)
        stacks = {key:value.reshape(len(indices),model.config.depth,*value.shape[1:]) for key,value in regions.items()}
        batch = vision.regions_to_batch(stacks)
        source = data.paths[0]
        with torch.no_grad():
            base = model.forward_core(batch)
            macro = MacroPolicy(model).forward_core(batch)
        assert torch.equal(base["policy_logits"],macro["policy_logits"])
    result = {"schema":"v372-real-pixel-forward-smoke/1", "source":str(source),
              "source_sha256":file_sha(source), "baseline_sha256":file_sha(baseline),
              "indices":indices, "history_indices":rows.tolist(), "features_shape":list(base["features"].shape),
              "initial_logits_exact":True, "finite":bool(torch.isfinite(macro["policy_logits"]).all()),
              "real_pixel_inference":True, "real_pixel_training":False, "gameplay_effect":"not_run"}
    path = Path("Document/Update/Alpha/v3.7.2/evidence/real-pixel-forward-smoke.json")
    with path.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
