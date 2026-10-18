# Stage 2 · process（处理）

把「一堆分片」变成「一个被记录清楚的数据集」。这个阶段不改游戏、不训练模型，
它只做四件事：**校验、统计、打标签、钉死划分**，然后写 `manifest.json`。

## 为什么需要它

在此之前，每个训练脚本都自己重新打开数据集、自己推导"这里面有什么"。
第一次出问题就很具体：一次中断的采集让某片只有半截数据，而训练报告仍然声称用了某个划分。

`manifest.json` 就是把这个不确定性消掉：**train 阶段不认没有 manifest 的数据集**，
并且会拿 manifest 去核对磁盘上的分片。

## 用法

```powershell
# 默认：自动推断 modality，计算教师价值，写 manifest 到数据集目录
python -m touhou_rl.pipeline process --raw third_party/th08/build/rl/datasets/vision-lunatic

# 只校验不重算标签（教师没变时用，省时间）
python -m touhou_rl.pipeline process --raw ... --no-teacher

# 指定划分
python -m touhou_rl.pipeline process --raw ... --seed 11 --val-fraction 0.05
```

## 校验项（任一不满足直接失败）

| 检查 | 为什么 |
| --- | --- |
| 必需键齐全（`frame/self_feat/bullets/danger/action/meta`） | 缺键在训练时才炸，代价高 |
| `action` 形状 `(n,4)` 且各维不越界（9/2/2/2） | 越界动作会让 `action_to_buttons` 抛错或静默错位 |
| `self_feat` `(n,24)`、`bullets` `(n,64,6)`、`danger` `(n,16,8)` | 与观测契约对齐 |
| 分片非空 | 空分片会让 epoch 切分出零批次 |
| `self_feat` 数值不荒谬（\|x\| ≤ 1e6） | 读错版本的内存时，特征会变成天文数字——这是最危险的静默失败 |
| 总帧数 ≥ `--min-frames`（默认 64） | 防止误把空目录当数据 |

## 打标签（教师价值）

`/process` 会调用 `touhou_rl.pilot.AvoidPilot`，对每帧算出 **18 个候选动作的价值**
（9 方向 × 2 火力状态），存到 `<数据集目录同级>/<名字>-teacher-values.npz`。

两条设计原则：

1. **不落进分片**：价值是特权观测的纯函数，换教师重算只要几秒/千帧，不该写死进视频数据；
2. **标注时 `switch_penalty = 0`**：这个惩罚依赖教师**上一帧的动作**，学生观测不到，
   留在标签里等于往目标里灌教师历史。

## 划分钉死

`manifest.splits` 里存三样东西：

```json
{"seed": 7, "val_fraction": 0.1, "val_frames": 452, "val_index_sha256": "..."}
```

`train` 会用同样的 `(seed, val_fraction)` 重新推导，并**校验哈希**。
所以：数据集加了分片但没重跑 process → train 直接报错，而不是悄悄换了一套验证集。

## 产出

```
<out>/manifest.json      # 版本、模态、分片清单（含 sha256/字节数/帧数）、
                         # 教师配置、划分、统计、采集溯源
<dataset>-teacher-values.npz   # 教师价值（若启用了打标签）
```

`manifest.stats` 里会记录：总帧数、关卡直方图、方向动作直方图、火力键比例、
炸弹帧数、教师标注耗时——这些是判断"这份数据够不够、偏不偏"的第一手依据。

## 什么时候要重跑

- 采集加了新分片 → **必须**重跑（否则 manifest 与磁盘不一致，train 会拒绝）；
- 教师改了（`pilot.py` 的评分逻辑/参数）→ 重跑，并加 `--no-teacher` 之外的方式强制重算
  （删掉 `<dataset>-teacher-values.npz`）；
- 只想改划分比例/随机种子 → 重跑即可，不需要重新采集。
