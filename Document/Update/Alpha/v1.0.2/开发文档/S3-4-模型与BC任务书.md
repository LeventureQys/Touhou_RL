# S3-4 模型与行为克隆任务书

- 所属 Stage：Stage 3（开发）
- 依赖前置：无游戏依赖；只依赖 `框架接口契约.md` §3/§4/§7/§8
- 并行状态：**可并行 B**（与 S3-1、S3-2 同时开工）
- 所属阶段：开发

---

## 1. 背景与目标（零上下文）

训练数据是「最近 64 发子弹 + 自机特征 + 危险图」与「人类当帧按了什么」。本任务实现网络与
监督训练，把数据变成可实时推理的策略检查点。

网络设计动机：

- **排列不变**：弹幕是集合，逐子弹 MLP + 掩码池化对顺序/数量不敏感（PIFE 思路）；
- **四个头**：方向（9）、射击（2）、低速（2）、炸弹（2）——直接对应人类输入，不做 72 类扁平 softmax；
- **附加头**：`value`（PPO 用）与 6 维 `reward_pred`（多目标辅助，v1.0.3 的 MOPPO 复用）。

## 2. 交付物

```
touhou_rl/models.py   # NetConfig / MaskedSetEncoder / DanmakuPolicy / save_checkpoint / load_checkpoint
touhou_rl/bc.py       # BCConfig / train_bc / split_indices / CLI
```

## 3. 契约要点

| 契约节 | 要求 |
| --- | --- |
| §3 | 输入三键形状必须与 `obs` 完全一致；`bullets` 第 5 列是有效位 |
| §4 | 四个头的类别数 `9/2/2/2` |
| §8 | 检查点含 `state_dict/config/obs/extra`；加载时校验 `obs` 维度 |
| §9 | `python -m touhou_rl.bc --dataset ... --out ...` |

## 4. 训练规则（必须按此实现）

- 损失：四头交叉熵之和；`bomb` 头用类权重（`clip(1,8)`），其它头 `clip(0.5,8)`；
- 优化器：AdamW（`lr=3e-4`、`wd=1e-4`）+ 余弦退火；梯度裁剪 1.0；
- 划分：按索引随机划分，`val_fraction=0.1`，固定种子；
- 每 epoch 保存 `last.pt`，验证损失更优时保存 `best.pt`；
- 日志：每 epoch 打印一行 JSON（训练/验证损失、四头准确率、耗时、lr）。

## 5. 输入 / 输出

- 输入：`--dataset <dir>`（S3-3 产出的分片目录），可选 `--epochs/--batch-size/--lr/--device/--max-steps-per-epoch`；
- 输出：`best.pt`、`last.pt`、`bc-summary.json`（含参数量、类权重、每 epoch 指标）。

## 6. 可观察的验收标准

1. `DanmakuPolicy()` 参数量 < 2 M，CPU 单样本前向 < 2 ms（实测环境 RTX 4070 SUPER 机器）；
2. 随机初始化时四头 logits 有限（不能出现 NaN——掩码池化必须正确处理「全部无效」）；
3. 掩码池化：把无效行的前 5 列改成任意值，输出特征不变；
4. 检查点保存→加载→同一输入输出一致（`atol=1e-5`）；
5. 在真实数据集上训练 ≥ 3 epoch：方向头验证准确率显著高于 1/9 基线（报告实际数字，不设人为门槛造假）；
6. `bc-summary.json` 中 `frames` 与数据集总帧数一致。

## 7. 禁止事项

- 禁止把观测改成像素/网格（本版只做对象级）；
- 禁止在 BC 里使用未来帧信息（标签只能是当前帧动作）；
- 禁止静默加载维度不符的检查点；
- 禁止用测试集调参后仍声称是验证集结果。

## 8. 出错行为

| 情况 | 期望 |
| --- | --- |
| 数据集为空/太小（< 2×batch） | `RuntimeError("dataset too small for batch N: M frames")` |
| 检查点 `obs` 维度不符 | `ValueError("checkpoint observation contract does not match this build")` |
| 指定 `cuda` 但不可用 | 自动回落到 CPU，并在 `bc-summary.json` 里记录 `device` |
