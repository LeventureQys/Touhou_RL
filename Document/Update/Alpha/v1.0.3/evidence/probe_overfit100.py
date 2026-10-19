import numpy as np, torch
from touhou_rl.vmodels2 import VisionScorerV2, VisionNetConfigV2, decode_position, heatmap_target
from touhou_rl.vision_data import VisionDataset
from touhou_rl.vision import regions_to_batch
from touhou_rl.pipeline.manifest import split_indices

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ds = VisionDataset(r"third_party/th08/build/rl/datasets/pilot-ud0059")
tr, va = split_indices(len(ds), 7, 0.1)
idx = tr[:100]
cfg = VisionNetConfigV2(depth=1)
m, _ = None, None
for _ in idx:
    pass
m = VisionScorerV2(cfg).to(dev)
sizes = cfg.region_sizes
regions = ds.region_frames(idx, sizes)
stacked = {n: f.reshape(len(idx), 1, sizes[n][0], sizes[n][1], 3) for n, f in regions.items()}
batch = {k: v.to(dev) for k, v in regions_to_batch(stacked, dev).items()}
truth = torch.as_tensor(ds.labels(idx)["self_feat"][:, :2], device=dev)
H, W = cfg.heatmap_hw
tgt = torch.as_tensor(heatmap_target(ds.labels(idx)["self_feat"][:,0], ds.labels(idx)["self_feat"][:,1], H, W, 1.5), device=dev)

def pos_err(model):
    model.eval()
    with torch.no_grad():
        out = model.decode_state(batch)
    rx, ry = decode_position(out["heatmap_logits"])
    p = torch.stack([rx, ry], 1).cpu().numpy(); t = truth.cpu().numpy()
    return float((np.abs(p[:,0]-t[:,0])*384 + np.abs(p[:,1]-t[:,1])*448).mean()/2)

opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
print("device", dev, "| frames", len(idx), "| init pos_err = %.2f px" % pos_err(m))
for step in range(1, 401):
    m.train()
    out = m.decode_state(batch)
    logp = torch.log_softmax(out["heatmap_logits"].flatten(1), dim=-1)
    loss = -(tgt * logp).sum(-1).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 50 == 0:
        print(f"  step {step:3d}  loss {loss.item():.4f}  pos_err {pos_err(m):6.2f} px")
final = pos_err(m)
print("FINAL overfit pos_err on the same 100 frames = %.2f px" % final)
print("VERDICT:", "architecture CAN localize (capacity OK)" if final < 10 else "architecture CANNOT overfit -> structural problem")
