# Stage 3 · train（训练）

读 `manifest.json`，训练一个模型，写出可审计的 `train-report.json`。
训练算法本身在 `bc.py` / `distill.py` / `vision_train.py`，本阶段负责**校验、选择、记账**。

## 三种模型

| `--model` | 需要的 modality | 输入 | 标签 | 实现 |
| --- | --- | --- | --- | --- |
| `privileged_bc` | `privileged` | 特权观测 | 人类专家动作（4 头分类） | `touhou_rl.bc` |
| `privileged_value` | `privileged` | 特权观测 | 教师 18 路价值（回归） | `touhou_rl.distill` |
| `vision` | `vision` | 帧张量 `[B, depth, 128, 112, 3]` | 先特权状态，再教师价值 | `touhou_rl.vision_train` |

## 用法

```powershell
# 视觉学生（先表征、后策略，两阶段）
python -m touhou_rl.pipeline train --model vision `
    --manifest third_party/th08/build/rl/datasets/vision-lunatic/manifest.json `
    --out third_party/th08/build/rl/runs/vision-v2 --epochs 12 --batch-size 128

# 只训策略阶段，并从表征阶段热启动
python -m touhou_rl.pipeline train --model vision --vision-stage policy `
    --init-from third_party/th08/build/rl/runs/vision-v1/state.pt ...

# 特权价值学生（对照基线）
python -m touhou_rl.pipeline train --model privileged_value --epochs 24 --batch-size 512 ...
```

## 开跑前的三道闸

1. **manifest 必须存在**（没有就让 `process` 跑一次）；
2. **模态必须匹配**：`--model vision` 用特权数据集会直接报错；
3. **数据集必须仍与 manifest 一致**：分片缺失、字节数变化、总帧数变化 → 一律拒绝。
   这一步用 `DatasetManifest.verify()`，默认核对存在性与帧数，`inspect --deep` 会重算哈希。

## 指标口径（这里最容易自欺）

每个模型只报**能换算成游戏意义的数**，不报裸 loss：

| 模型 | 主指标 | 辅助指标 |
| --- | --- | --- |
| `vision`（表征） | 自机位置误差 **px**、危险图相关、子弹数误差 **颗** | 验证 loss |
| `vision`（策略） | **后悔值 px**（教师最优动作价值 − 学生所选动作价值）、top1 | 验证 loss |
| `privileged_value` | 同上 | — |
| `privileged_bc` | 方向准确率 | 验证 loss |

**模型按什么选**：

- `vision` 策略阶段按 **后悔值** 选，不按 loss。实测过：loss 一路下降，而 top1 与后悔值
  从第 4 epoch 起持续变差——"越训越自信地选错"，按 loss 选会挑到明显更差的策略。
- `vision` 表征阶段按验证 loss 选，因为它的 loss 与像素误差同向。
- `privileged_bc` / `privileged_value` 沿用各自脚本的准则（loss）。

## 产出

```
<out>/
  best.pt | state.pt | policy.pt | last.pt    检查点（含 obs/frame 契约与 extra 元信息）
  bc-summary.json | distill-summary.json | vision-summary.json   训练器原始摘要
  train-report.json                          ← 本阶段产出：模型种类、超参、数据集身份、扁平指标
```

## 已知缺口（不要误读指标）

1. **视觉策略目前没有 DAgger**：视觉数据全部来自人类轨迹，学生自己会死的状态没有标签；
2. **自机位置误差 48 px**：主干末端 `AdaptiveAvgPool(4,4)` 抹平了空间精度，应改热图回归；
3. **小数据下游指标很脆**：4,524 帧训练的视觉策略后悔值 3.98 px（特权学生 2.34 px），
   低于随机策略的存活帧数——先看数据量，再谈调参。
