"""Decisive probe: train v2 state head ONLY on the uncorrupted frames.

Discovery that motivates this script
------------------------------------
In `datasets/pilot-ud0059`, shards 153..565 (global frames 19,584..72,471, i.e.
73% of the dataset) contain a **single frozen frame** per region: every sample in
those shards has the identical playfield PNG hash while `self_feat` and
`meta.frame` keep advancing.  The model therefore cannot learn from that 73%,
and the reported `val_player_pos_px = 68.07` is dominated by impossible frames.

This script asks the only question that matters now: **if we train on the clean
27% only, what does the state head actually achieve?**

Run:
    $py tools_probe_clean_train.py   (or from TEMP)
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from touhou_rl.pipeline.manifest import split_indices
from touhou_rl.vision import regions_to_batch
from touhou_rl.vision_data import VisionDataset
from touhou_rl.vision_metrics import all_state_metrics
from touhou_rl.vmodels2 import VisionNetConfigV2, VisionScorerV2, heatmap_target, decode_position
from touhou_rl.vision_train import _danger_target

CLEAN_MAX = 19584          # first frozen frame
DATASET = r"third_party/th08/build/rl/datasets/pilot-ud0059"
EPOCHS = 8
BATCH = 96
LR = 3e-4


def gather(ds, indices, sizes, depth=1):
    regions = ds.region_frames(indices, sizes)
    stacked = {
        name: frames.reshape(len(indices), depth, sizes[name][0], sizes[name][1], 3)
        for name, frames in regions.items()
    }
    labels = ds.labels(indices)
    return stacked, labels


def main() -> int:
    torch.manual_seed(7)
    np.random.seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = VisionDataset(DATASET)
    train_all, val_all = split_indices(len(ds), 7, 0.1)
    train = [i for i in train_all if i < CLEAN_MAX]
    val = [i for i in val_all if i < CLEAN_MAX]
    print(f"device={device} clean train={len(train)} clean val={len(val)} "
          f"(dropped {len(train_all)-len(train)} train / {len(val_all)-len(val)} val frozen frames)")

    cfg = VisionNetConfigV2(depth=1, heatmap_gauss_sigma=1.0)
    model = VisionScorerV2(cfg).to(device)
    sizes = cfg.region_sizes
    grid_h, grid_w = cfg.heatmap_hw
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(7)

    def epoch(training: bool) -> dict:
        model.train(training)
        order = list(train if training else val)
        if training:
            rng.shuffle(order)
        preds, truths, pdanger, tdanger, hit_src = [], [], [], [], []
        total = 0.0
        for start in range(0, len(order) - BATCH + 1, BATCH):
            chunk = order[start:start + BATCH]
            stacked, labels = gather(ds, chunk, sizes)
            batch = regions_to_batch(stacked, device)
            truth_self = labels["self_feat"].astype(np.float32)
            with torch.set_grad_enabled(training):
                out = model.decode_state(batch)
                tgt = torch.as_tensor(
                    heatmap_target(truth_self[:, 0], truth_self[:, 1], grid_h, grid_w, cfg.heatmap_gauss_sigma),
                    device=device,
                )
                logp = torch.log_softmax(out["heatmap_logits"].flatten(1), dim=-1)
                loss_heat = -(tgt * logp).sum(-1).mean()
                loss_self = torch.nn.functional.smooth_l1_loss(
                    out["self_feat"][:, 2:24], torch.as_tensor(truth_self[:, 2:24], device=device)
                )
                loss_danger = torch.nn.functional.smooth_l1_loss(
                    out["danger_log"].reshape(len(chunk), 16, 8),
                    torch.as_tensor(_danger_target(labels["danger"]), device=device),
                )
                # 关键改动：显式坐标损失（直接优化指标所读的量）
                rx, ry = decode_position(out["heatmap_logits"])
                tcoord = torch.as_tensor(truth_self[:, :2], device=device)
                loss_coord = ((torch.abs(rx - tcoord[:, 0]) * 384.0 + torch.abs(ry - tcoord[:, 1]) * 448.0) / 2.0).mean() / 100.0
                loss = loss_heat + loss_self + loss_danger + 1.0 * loss_coord
                if training:
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
            total += float(loss.detach()) * len(chunk)
            preds.append(out["self_feat"].detach().cpu().numpy())
            truths.append(truth_self)
            pdanger.append(out["danger_log"].detach().cpu().numpy())
            tdanger.append(labels["danger"])
            hit_src.append(out["heatmap_logits"].detach().cpu().numpy())
        metrics = all_state_metrics(
            np.concatenate(preds), np.concatenate(pdanger),
            np.concatenate(truths), np.concatenate(tdanger),
            heatmap_logits=np.concatenate(hit_src),
        )
        metrics["loss"] = round(total / max(1, len(preds) * BATCH), 5)
        return metrics

    history = []
    for ep in range(1, EPOCHS + 1):
        started = time.perf_counter()
        tr_row = epoch(True)
        with torch.no_grad():
            va_row = epoch(False)
        row = {"epoch": ep, "seconds": round(time.perf_counter() - started, 1),
               "train_loss": tr_row["loss"], **{f"val_{k}": v for k, v in va_row.items() if k != "loss"}}
        history.append(row)
        print(f"[coord] ep{ep:2d} {json.dumps({k: row[k] for k in ('val_player_pos_px','val_player_pos_p95_px','val_heatmap_hit_rate','val_danger_corr_perframe_median')})}", flush=True)
    best = min(history, key=lambda r: r["val_player_pos_px"])
    print("\nBEST", json.dumps({k: best[k] for k in ("epoch", "val_player_pos_px", "val_player_pos_p95_px",
                                                    "val_heatmap_hit_rate", "val_danger_corr_perframe_median")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

