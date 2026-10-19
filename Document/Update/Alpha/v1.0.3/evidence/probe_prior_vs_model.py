import numpy as np, torch
from touhou_rl.vmodels import load_vision_checkpoint
from touhou_rl.vision_data import VisionDataset
from touhou_rl.vision import regions_to_batch
from touhou_rl.pipeline.manifest import split_indices

ds = VisionDataset(r"third_party/th08/build/rl/datasets/pilot-ud0059")
tr, va = split_indices(len(ds), 7, 0.1); va = np.array(va)
m, _ = load_vision_checkpoint(r"third_party/th08/build/rl/runs/v103-exp-05/state.pt", map_location="cpu")
m.eval(); sizes = m.config.region_sizes
N = len(va)
pred = np.zeros((N,2)); truth = np.zeros((N,2)); amax = np.zeros((N,2)); stages = np.zeros(N, int)
B = 128
for s in range(0, N, B):
    chunk = va[s:s+B].tolist()
    regions = ds.region_frames(chunk, sizes)
    stacked = {n: f.reshape(len(chunk),1,sizes[n][0],sizes[n][1],3) for n, f in regions.items()}
    with torch.no_grad():
        out = m.decode_state(regions_to_batch(stacked, "cpu"))
    lab = ds.labels(chunk)
    pred[s:s+len(chunk)] = out["self_feat"][:,:2].numpy()
    truth[s:s+len(chunk)] = lab["self_feat"][:,:2]
    stages[s:s+len(chunk)] = lab["meta"][:,1]
    lg = out["heatmap_logits"].flatten(1).argmax(1).numpy()
    amax[s:s+len(chunk),0] = (lg % 24 + 0.5)/24; amax[s:s+len(chunk),1] = (lg // 24 + 0.5)/28
np.savez(r"C:\Users\a5935\AppData\Local\Temp\vp.npz", pred=pred, truth=truth, amax=amax, stages=stages)

def mae(p,t): return float((np.abs(p[:,0]-t[:,0])*384 + np.abs(p[:,1]-t[:,1])*448).mean()/2)
print(f"{'stage':>6} {'n':>5} {'model':>7} {'argmax':>7} {'constant':>9} {'R2_vs_const':>12} {'corr_x':>7} {'corr_y':>7} {'std_p':>6} {'std_t':>6}")
for st in sorted(set(stages.tolist())):
    mk = stages == st; t = truth[mk]
    const = np.broadcast_to(t.mean(0), t.shape)
    cm = mae(pred[mk], t); cmx = mae(amax[mk], t); cc = mae(const, t)
    r2 = 1 - cm/cc
    print(f"{st:>6} {mk.sum():>5} {cm:>7.2f} {cmx:>7.2f} {cc:>9.2f} {r2:>11.3f} "
          f"{np.corrcoef(pred[mk,0],t[:,0])[0,1]:>7.3f} {np.corrcoef(pred[mk,1],t[:,1])[0,1]:>7.3f} "
          f"{pred[mk].std():>6.3f} {t.std():>6.3f}")
t = truth; const = np.broadcast_to(t.mean(0), t.shape)
print(f"{'ALL':>6} {N:>5} {mae(pred,t):>7.2f} {mae(amax,t):>7.2f} {mae(const,t):>9.2f} {1-mae(pred,t)/mae(const,t):>11.3f} "
      f"{np.corrcoef(pred[:,0],t[:,0])[0,1]:>7.3f} {np.corrcoef(pred[:,1],t[:,1])[0,1]:>7.3f} {pred.std():>6.3f} {t.std():>6.3f}")
