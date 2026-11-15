# T04 证据：A0 / A0v 受控学习（WP420-03 第一轮）

> 机器可读镜像：[`train_controlled.json`](train_controlled.json)（本文件每个数字都在那里；
> 它由 `.tmp_v420_t04/write_evidence_json.py` 从 `train.json` 与两个 checkpoint 文件生成，
> 不手写，checkpoint 的 sha256 在写证据时**重新计算**过）。
>
> 层 1（256 槽过拟合）单列在 [`LAYER1.md`](LAYER1.md)；开发评测单列在
> [`DEV_EVAL.md`](DEV_EVAL.md)；门的据实判定在 [`G420_2.md`](G420_2.md)。
>
> **这不是能力声明。** 本文件只报告一次真实的训练运行与它的离线测量。没有游戏指标、
> 没有泛化主张、没有 G420-2 判定。

---

## 0 结论摘要

| 问题 | 回答 | 依据 |
|---|---|---|
| 2 帧语料是否被**真实消费**（进入训练） | **是**：两个臂各 5,000 个 Adam 步、每步 batch 128，累计 **637,880** 次样本呈现/臂 | §3 |
| A0 / A0v 是否训练到收敛 | 训练池 **CE 0.0510 / 0.0927**、top1 **0.9957 / 0.9875**、macroR **0.939 / 0.893** | §5 |
| 留出（20%，3,953 槽）表现 | top1 **0.5378 / 0.5621**、CE **2.802 / 2.642**（均匀基线 `ln18 = 2.8904`） | §5 |
| 两个 checkpoint 能否被**冻结** evaluator 原样加载 | **能**：`load_v410_checkpoint` 读回成功，参数 731,802/731,802，重载后 logits 与训练模型 **逐比特相等（max abs diff = 0.0）** | §6 |
| 这是否 G420-2 通过 | **不是**。见 [`G420_2.md`](G420_2.md)：`not passed` | §8 |

一句话：**A0 与 A0v 在同一协议、同一 split、同一 B0 模型上各真训了 5,000 步，训练池被拟合
到 top1 ≈ 0.99，留出 top1 ≈ 0.54–0.56；两个 checkpoint 都能被冻结 evaluator 读回且前向逐比特
一致。这证明「语料可被真实训练消费」，不证明任何能力，也不构成 G420-2。**

---

## 1 运行命令与退出码

```bash
cd /mnt/d/WorkShop/Touhou_RL
PYTHONPATH=$PWD /root/v400-venv/bin/python tools/rl/v420.py train \
  --config configs/v420/learn-controlled.json \
  --out $PWD/.tmp_v420_data/T04/train.json --execute      # 退出码 0，墙钟 386.67 s
```

- 入口脚本：`.tmp_v420_t04/train.sh`（记录开始/结束 UTC 时间与退出码，日志
  `.tmp_v420_data/T04/train.log`）。
- 事前 20 步 smoke（同一协议、独立 checkpoint 目录 `.tmp_v420_data/T04/checkpoints_smoke/`）：
  退出码 0、墙钟 22.47 s，用于先证明 GPU 路径、写盘与读回都成立；**它不是结果**，只在
  `train_controlled.json#smoke_run` 里留证。
- 训练日志的进度行（每 100 步）在 `train.log`，本文件的曲线表是其中每 500 步的截取，
  完整 50 个点（每 100 步）在 `train_controlled.json#training.<arm>.curve`。

---

## 2 协议（**事前**冻结，先写进 `configs/v420/learn-controlled.json` 再运行）

| 项 | 值 | 说明 |
|---|---|---|
| 臂 | `A0`、`A0v` | 两次独立运行，两个 checkpoint |
| 训练种子 | `seed = 0` | 计划的筛查 seed |
| 步数 | **5,000** | 计划的筛查预算；扩到 20,000 是另一次有条件决定，本轮**没有**做 |
| batch | **128** | |
| 优化器 | **Adam**（`torch.optim.Adam`） | 不是 AdamW |
| 学习率 | **1e-4** | |
| weight decay | **0.0** | |
| 梯度裁剪 | **0.5**（`clip_grad_norm_`） | 记录裁剪前/后范数 |
| 设备 | `device=auto` → **cuda**（RTX 4070 SUPER 12,282 MiB） | 见 §4 |
| 确定性 | `torch.manual_seed(0)`、`numpy.seed(0)`、`random.seed(0)`、`cudnn.deterministic=True`、`cudnn.benchmark=False`、`torch.use_deterministic_algorithms(True)`（**strict 生效，非 warn_only**）、`CUBLAS_WORKSPACE_CONFIG=:4096:8` | 见 §4 |
| 模型 | 冻结的 `touhou_rl.v410.student.diagnostics.ProbeNet`，**731,802** 参数 | `build_model("B0")`，参数漂移会直接报错 |
| 损失 | `touhou_rl.v410.student.bc.bc_loss` = 18 路 one-hot 交叉熵；无 bomb 头、无 soft target、无 mask 权重 | 复用，不重写 |
| epochs | 41（`ceil(5000 / 124)`，`steps_per_epoch = ceil(15819/128) = 124`） | checkpoint 的 `epochs`/`last_epoch` 字段 |

### 2.1 标签口径（必须与数字一起读）

版本计划把 A0 的目标称作「chosen 动作 CE」。**本语料里能不加发明的两列是**：

- `executed_cand` —— 教师**实际执行**的动作（`source=step_frames`），行为克隆目标；
- `chosen_verified_cand` —— 教师在**引擎验证过**的候选里选出的动作（每槽恰 1 个）。

两者在 **6,288 / 19,772 槽（31.8%）** 上不同。所以本轮：

- **A0 = `executed` CE**（行为目标）；
- **A0v = `chosen_verified` CE**（计划里「chosen」的那个量，单独命名、单独 checkpoint）。

两列从不相加、不相乘、不互相填充；`targets.jsonl` 分栏就是为了让下游必须显式点名。
报告 `train_controlled.json#label_semantics` 与每个 checkpoint 的 `v420_target` 都写明列名。

---

## 3 数据与 split（A0/A0v 都不在留出上训练）

| 项 | 实测 |
|---|---|
| 语料 | `.tmp_v420_data/T04/dataset/`，版本 `v420-2f-dataset/1`，**19,772 槽**（seed 10/11/12） |
| 语料身份 | `index.json` sha256 `dbb17ec201a63a0c91112739c80e44de2cbd55d3712a25271ed17770672b09a8` |
| 每槽自有 obs/HUD/已执行动作/8 条历史/逐槽 verified 标签 | 19,772 / 19,772（自检 0 违反，见 [DATASET_2F.md](DATASET_2F.md)） |
| 留出规则 | 每 shard 的 `row_index % 5 == 4` → 恰好 20% |
| train / holdout | **15,819 / 3,953**（留出比例 0.199929） |
| 每 shard | 5,274 / 5,274 / 5,271（train）与 1,318 / 1,318 / 1,317（holdout） |
| 训练是否碰过留出 | **没有**：`SlotStream` 只接收 train 行；留出只用于前向测量 |

留出是**三个训练 seed 的语料内切分**，不是干净泛化集，也不是 dev/confirm 划分。

---

## 4 设备与确定性（实测字段）

| 项 | A0 | A0v |
|---|---|---|
| device | `cuda`（NVIDIA GeForce RTX 4070 SUPER） | 同 |
| `deterministic_algorithms` | `true` | `true` |
| strict 生效 | `true`（`warn_only = false`） | `true` |
| `cudnn_deterministic` / `benchmark` | `true` / `false` | 同 |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | 同 |

---

## 5 离线结果

### 5.1 最终指标（训练池 15,819 槽 / 留出 3,953 槽，各自按自己臂的目标列）

| arm | train CE | train top1 | train macroR | train proper safe-set | holdout CE | holdout top1 | holdout macroR | holdout proper safe-set |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.050999 | 0.995701 | 0.939394 | 0.679752 | 2.802158 | 0.537819 | 0.286150 | 0.438654 |
| A0v | 0.092735 | 0.987547 | 0.893496 | 0.987547 | 2.642093 | 0.562105 | 0.239212 | 0.562105 |

- 参考标尺：均匀分布 `ln 18 = 2.890372`。A0/A0v 的**留出 CE 都略低于均匀基线**，
  留出 top1 在 0.54–0.56 之间 —— 比随机（1/18 = 0.0556）高很多，但离可用还远。
- A0 的 train safe-set 0.6798 ≈ `executed == chosen_verified` 的比例（13,484/19,772 = 0.6820）：
  一个把 `executed` 学到 top1 0.996 的模型，落在 verified 集合里的上限就是这个比例。
- A0v 的 train safe-set 与 top1 相同（0.9875），因为它的目标就是那个 1-of-18 的 verified 候选。

### 5.2 同一模型在**另一列**上的表现（交叉检查，不是第二个结果）

四个测量块全部来自 `train_controlled.json#training.<arm>.final`（每臂 6 个块：train/holdout ×
executed/chosen_verified，其中 train/holdout 就是自己目标列的那两个）。

| arm（模型） | 被测列 | train CE | train top1 | holdout CE | holdout top1 |
|---|---|---:|---:|---:|---:|
| A0 | `executed`（自己的目标） | 0.050999 | 0.995701 | 2.802158 | 0.537819 |
| A0 | `chosen_verified`（另一列） | 3.234825 | 0.679752 | **4.361222** | 0.438654 |
| A0v | `chosen_verified`（自己的目标） | 0.092735 | 0.987547 | 2.642093 | 0.562105 |
| A0v | `executed`（另一列） | 3.509210 | 0.675896 | **4.309964** | 0.458133 |

读法：**两个模型在留出上对「另一列」的 CE 分别是 4.36 / 4.31，都高于均匀基线 `ln18 = 2.89`**，
即高置信地答错。这不是新结果，而是「行为标签训练的模型不能直接当验证目标模型」的量化表现；
它也是 A1（安全集合监督）不能靠改标签/平移标签凑出来的原因之一。

### 5.3 训练曲线（每 500 步截取；完整曲线每 100 步，共 50 点）

| step | A0 loss | A0 top1 | A0v loss | A0v top1 |
|---:|---:|---:|---:|---:|
| 500 | 1.767122 | 0.438848 | 1.589585 | 0.500824 |
| 1000 | 1.394659 | 0.580999 | 1.289475 | 0.592453 |
| 1500 | 1.097596 | 0.673649 | 1.051386 | 0.682357 |
| 2000 | 0.868241 | 0.748098 | 0.858198 | 0.742528 |
| 2500 | 0.663013 | 0.810073 | 0.676534 | 0.800581 |
| 3000 | 0.504686 | 0.859496 | 0.534367 | 0.846709 |
| 3500 | 0.339718 | 0.914411 | 0.392862 | 0.892445 |
| 4000 | 0.227634 | 0.950812 | 0.289227 | 0.924688 |
| 4500 | 0.132002 | 0.978897 | 0.190319 | 0.959363 |
| 5000 | 0.067322 | 0.992547 | 0.112860 | 0.981094 |

曲线单调下降、无平台、无发散；梯度裁剪前范数 1.75 → 3.1（A0）/ 1.90 → 4.32（A0v），
裁剪后恒为 0.5（裁剪生效）。**曲线健康 = 值得扩到 20k 的前提之一，但本轮没有扩**。

### 5.4 吞吐（实测，不含 4 次评估前向）

| arm | train 秒 | steps/s | samples/s | samples_seen | steps/epoch |
|---|---:|---:|---:|---:|---:|
| A0 | 118.227 | 42.292 | 5,395.4 | 637,880 | 124 |
| A0v | 111.205 | 44.962 | 5,736.1 | 637,880 | 124 |

A0 是第一个臂，吞吐里含首次 CUDA 初始化与 mmap 预热；两个臂的稳态速率都在
5.4k–5.7k samples/s（≈ 42–45 steps/s），瓶颈在 Python 侧的 `.npy` 取行与归一化，不在 GPU。

---

## 6 checkpoint（可被**冻结** evaluator 读回）

| 项 | A0 | A0v |
|---|---|---|
| 路径 | `.tmp_v420_data/T04/checkpoints/a0_seed0_step5000.pt` | `.tmp_v420_data/T04/checkpoints/a0v_seed0_step5000.pt` |
| 大小 | 2,940,699 B | 2,940,741 B |
| **sha256** | `96cb219eac83391f42b18ea656b5a6c5ac81e97cd1fd8c1e2c0d404aee6b89ad` | `f7188c0d773e1085299e797327fd62a2bcec35bf50c115cb203d2ad2fa1f4fd9` |
| schema | `v410-wp04-bc-arm-checkpoint/1` | 同 |
| `arm` / `family` / `model_class` | `B0` / `probe` / `touhou_rl.v410.student.diagnostics.ProbeNet` | 同 |
| `parameters` | 731,802 | 731,802 |
| `model_config` | 冻结的 6 键 probe 配置 | 同 |
| `seed` / `epochs` / `last_epoch` | 0 / 41 / 41 | 0 / 41 / 41 |
| `metrics.train` | CE 0.050999、top1 0.9957、macroR 0.9394 | CE 0.092735、top1 0.9875、macroR 0.8935 |
| `metrics.val`（留出） | CE 2.802158、top1 0.537819 | CE 2.642093、top1 0.562105 |
| `v420_arm` / `v420_target` | `A0` / `executed` | `A0v` / `chosen_verified` |
| **读回证明** | `tools/rl/v410_student_eval.py::load_v410_checkpoint` 成功；`loaded=True`、`arm=B0`、`parameters=731802`、`parameters_pinned=731802`、`epochs=41`、`last_epoch=41`；8 个留出槽上训练模型与重载模型 logits **max abs diff = 0.0**（两侧都跑在 CPU） | 同（`loaded=True`、731,802/731,802、diff 0.0） |

`v420_arm` / `v420_target` 是**附加**字段：冻结 evaluator 只校验 `arm`/`family`/`model_class`/
`model_config`/`parameters`/`seed`/`epochs`/`state_dict`，不认识也不拒绝 v4.2.0 自己的键。

---

## 7 G420-0 L1 的「被消费」动作

- 本轮之前：语料存在、自检 0 违反，但**没有被任何训练运行消费**（[DATASET_2F.md](DATASET_2F.md) §6）。
- 本轮之后：两个臂各 5,000 步、每步 128 槽，共 **2 × 637,880 = 1,275,760** 次样本呈现；
  所有呈现的样本都来自 `v420-2f-dataset/1`（train 15,819 槽 + 留出 3,953 槽前向）。
- 因此 **L1 的「语料被消费」动作已执行**。语料本身的每槽时间/动作索引一致性由
  [DATASET_2F.md](DATASET_2F.md) 的 19,772 槽自检（0 违反、HASH 0 断点）承担。
- **L1 门判定本身不写在这里**：那是主代理按 §6 与本节一起裁定的事，本文件只提供数字。

---

## 8 边界（必须与上面的数字一起读）

1. **不是能力**：本文件没有任何游戏指标。标准 S1N 与首死即停的开发结果在
   [`DEV_EVAL.md`](DEV_EVAL.md)，且两者都是 0/20 过关。
2. **不是泛化**：留出只是三个训练 seed 的语料内切分；`holdout top1 0.54–0.56` 不能写成
   「学习有效」，也不能写成「无效」。
3. **不是 G420-2**：A1 未做（缺 T03 的引擎真值安全集合）、A2 未建、层 2 微段未跑。
   见 [`G420_2.md`](G420_2.md)。
4. **只跑了一个训练 seed**：`seed=0`。计划里「≥2/3 训练运行改善游戏指标」需要 seeds 0/1/2，
   本轮没有做，因此该继续条件**既未被满足也未被否定**。
5. **曲线健康不等于该扩 20k**：扩步数是另一次有条件的决定，本轮明确不做。
