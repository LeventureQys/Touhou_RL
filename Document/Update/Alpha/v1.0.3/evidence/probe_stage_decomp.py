import numpy as np, torch
from touhou_rl.vmodels import load_vision_checkpoint
from touhou_rl.vision_data import VisionDataset
from touhou_rl.vision import regions_to_batch
from touhou_rl.pipeline.manifest import split_indices

ds = VisionDataset(r"third_party/th08/build/rl/datasets/pilot-ud0059")
tr, va = split_indices(len(ds), 7, 0.1)
va = np.array(va)
m, _ = load_vision_checkpoint(r"third_party/th08/build/rl/runs/v103-exp-05/state.pt", map_location="cpu")
m.eval(); sizes = m.config.region_sizes
H, W = sizes["playfield"]

errs = np.zeros(len(va)); preds = np.zeros((len(va),2)); truths = np.zeros((len(va),2))
stages = np.zeros(len(va)); locmax = np.zeros(len(va))
B = 128
for s in range(0, len(va), B):
    chunk = va[s:s+B].tolist()
    regions = ds.region_frames(chunk, sizes)
    stacked = {n: f.reshape(len(chunk), 1, H if n=="playfield" else sizes[n][0], sizes[n][1], 3) for n, f in regions.items()}
    with torch.no_grad():
        out = m.decode_state(regions_to_batch(stacked, "cpu"))
    lab = ds.labels(chunk)
    p = out["self_feat"][:, :2].numpy(); t = lab["self_feat"][:, :2]
    preds[s:s+len(chunk)] = p; truths[s:s+len(chunk)] = t
    errs[s:s+len(chunk)] = (np.abs(p[:,0]-t[:,0])*384 + np.abs(p[:,1]-t[:,1])*448)/2
    stages[s:s+len(chunk)] = lab["meta"][:, 1]
    pf = regions["playfield"]
    for k in range(len(chunk)):
        cy = int(np.clip(t[k,1]*H, 0, H-1)); cx = int(np.clip(t[k,0]*W, 0, W-1))
        y0, y1 = max(0,cy-10), min(H,cy+10); x0, x1 = max(0,cx-10), min(W,cx+10)
        locmax[s+k] = pf[k, y0:y1, x0:x1].max()

tot = errs.sum()
print(f"N={len(va)}  mean={errs.mean():.2f}  p50={np.percentile(errs,50):.1f}  p90={np.percentile(errs,90):.1f}  p99={np.percentile(errs,99):.1f}")
print("\n-- 按 stage 分解 --")
print(f"{'stage':>6} {'n':>6} {'mean':>8} {'p90':>8} {'误差占比':>10}")
for st in np.unique(stages):
    mk = stages == st
    print(f"{int(st):>6} {mk.sum():>6} {errs[mk].mean():>8.2f} {np.percentile(errs[mk],90):>8.1f} {errs[mk].sum()/tot:>9.1%}")
print("\n-- 按真值是否贴边（外 10%）分解 --")
edge = (truths[:,0] < 0.1) | (truths[:,0] > 0.9) | (truths[:,1] < 0.1) | (truths[:,1] > 0.9)
for name, mk in (("贴边", edge), ("非贴边", ~edge)):
    print(f"{name:>6} n={mk.sum():>5} mean={errs[mk].mean():>7.2f} 误差占比={errs[mk].sum()/tot:>6.1%}")
print("\n-- 按真值是否越出场地 [0,1] --")
out = (truths[:,0] < 0) | (truths[:,0] > 1) | (truths[:,1] < 0) | (truths[:,1] > 1)
print(f"越界 n={out.sum()} ({out.mean():.2%}) mean={errs[out].mean() if out.any() else 0:.2f} 误差占比={errs[out].sum()/tot:.1%}")
print("\n-- 按真值处局部最亮值（自机可见性代理）分解 --")
for lo, hi in ((0,150),(150,220),(220,256)):
    mk = (locmax >= lo) & (locmax < hi)
    if mk.sum():
        print(f"locmax [{lo},{hi}) n={mk.sum():>5} {mk.mean():>6.1%} mean={errs[mk].mean():>7.2f} 误差占比={errs[mk].sum()/tot:>6.1%}")
print(f"\ncorr(err, locmax) = {np.corrcoef(errs, locmax)[0,1]:.3f}")
# 若只算「自机清晰可见且不贴边」的帧
easy = (~edge) & (locmax >= 220)
print(f"\n>> 子集「非贴边 且 locmax>=220」: n={easy.sum()} ({easy.mean():.1%}) mean={errs[easy].mean():.2f} p90={np.percentile(errs[easy],90):.1f}")
