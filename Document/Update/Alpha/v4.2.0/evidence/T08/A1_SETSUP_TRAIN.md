# T08：A1 集合监督训练 + 新仪器首次有效过关测量

日期：2026-11-14（Asia/Shanghai）。机器生成：[`a1_setsup_train.json`](a1_setsup_train.json) 由 `.tmp_v420_setsup/emit_evidence.py` 从训练报告与评测汇总生成；本文件的每个数字都在那份 JSON 里。

> **这不是 R1。** R1 是本项目从未运行的 v4.1.0 WP02 曲线（回溯投影）。本文件是 WP420-02 的 T08 续做：在 A1 安全集合上做集合监督，并用**唯一能报告 `stage_clear` 的仪器**做第一次有效过关测量。

> **旧仪器结构性无法报告过关。** `tools/rl/v410_student_eval.py`（以及 v400）的 `cand_to_mask` 在每一帧无条件带上 `BTN_SHOOT`，永远不产生上升沿，因此引擎的换面等待状态**永远离不开**，`stage_clear` 在结构上不可报告（T08 [`CLEAR_REACHABILITY.md`](CLEAR_REACHABILITY.md)）。所以 T08 之前的所有 `0/20`、`0/80` 都是**一台不能报告过关的仪器的测量**，与本文件的任何数字都不可比。

---

## 0 结论摘要

| 问题 | 回答 |
|---:|---:|
| A1 训练集多大 | **178 个 proper 状态**（源 200 个；49 条 TRAIN 祖先） |
| 排除了什么 | `|S|=0` 11 个、`|S|=1` 6 个、`|S|=18` 5 个、invalid 0 个 |
| 主损失 | `L_set = -log sum_{a in S} pi(a|s)`（唯一被优化的项） |
| 诊断项 | `L_unsafe = -log (1 - sum_{a not in S} pi(a|s))`（`no_grad`，不参与梯度） |
| 训练协议 | seed 0 / 5000 步 / batch 128 / Adam lr 0.0001 / wd 0.0 / clip 0.5 / cuda |
| 最终 L_set（训练/留出） | **0.000000** / **3.543519**（A0 起点 7.335824 / 5.766594） |
| 留出集合内命中率（训练后 / 随机水平） | **0.550000** / 0.5389 |
| 集合内命中率（训练/留出） | **1.000000** / **0.550000** |
| checkpoint | `a1set_seed0_step5000.pt` sha256 `7bacab5e2349…`、2,941,273 B |
| 读回差（冻结 loader） | logit max abs diff = **0.0** |
| diag 口径 20 局帧数均值 | **669.000**（R0 1650.400；门槛 1636.8） |
| s1n 口径 20 局帧数均值 | **1859.350**（R0 2824.350；门槛 3011.5） |
| 过关计数（40 局） | diag **0/20**、s1n **0/20**（R0：0/20、0/20） |
| 判定分支 | **`neither`** — set supervision is not enough at this scale |
| 策略塌缩（诊断） | diag 口径 20 局**长度完全相同**（669 帧、分数 600）；主导动作 `NE` 占 0.587（R0 的主导动作是 `idle`，占 0.293） |

一句话：**set supervision is not enough at this scale**

---

## 1 身份

| 项 | 值 |
|---:|---:|
| A1 语料 | `a1_safe_set_full.json` |
| A1 sha256 | `d6da8ea80eab2a2bc967c7802a6aa4d47a590fc500afa4f02b575351e7c482ef` |
| A0 起点 | `/mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T04/checkpoints/a0_seed0_step5000.pt` |
| A0 sha256 | `96cb219eac83391f42b18ea656b5a6c5ac81e97cd1fd8c1e2c0d404aee6b89ad` |
| 新 checkpoint | `/mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T08/setsup/checkpoints/a1set_seed0_step5000.pt` |
| 新 checkpoint sha256 | `7bacab5e23492f4bda7dc30cc8c256de8067ef25c6c08b559ed014219c2bcfba` |
| 评测仪器 | `tools/rl/v420_eval_shoot.py` |
| 仪器 sha256 | `7dbc2a3145562e54f4a7397fa63c8391938b6f6982f5d4b495a66b28fc6e8519` |
| 引擎 sha256 | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` |
| 训练配置 | `configs/v420/a1-setsup-train.json` sha256 `7c774788009425729286c85cfe59fca4e2cc530470f400fb73ea82d9d4522731` |
| 集合监督模块 | `touhou_rl/v420/setsup.py` sha256 `1e2f38e8a47c8b2f061f5312933ab1f357027328241da0d8fad7990578d7d479` |
| 既有训练器 | `touhou_rl/v420/train.py` sha256 `bf4c2c1bf99015536127e5bf4f28b804ebcb74645b23b187011a722a4849a7b9`（**未修改**） |

---

## 2 A1 训练集：200 → 178

从 A1 语料取 `2 ≤ |S| ≤ 17` 且 18 条臂全部有效的状态。三档排除与理由（跑前写死，不是事后挑选）：

| 桶 | 状态数 | 理由 |
|---:|---:|---:|
| `|S|=0` | 11 | |S| = 0: no candidate survived the 120-frame horizon, so there is no safe set to put probability mass on; filling it with a candidate is explicitly forbidden by the A1 config |
| `|S|=1` | 6 | |S| = 1: the set has no structure - '-log sum_{a in S} pi(a|s)' is exactly the single-action cross entropy A0/A0v already train on, so the state cannot test set supervision |
| `|S|=18` | 5 | |S| = 18: every candidate survived only because H=120 is a truncation window ending 30-120 frames before the recorded death (A1 section 0.10); the loss would be identically 0 for any policy |
| invalid（缺臂/HASH 不一致） | 0 | the record is incomplete: an unknown arm, a prefix HASH mismatch, a missing verdict, or a safe_set that does not match its own size |
| **保留** | **178** | 进入训练集 |

**proper 状态数 = 178 / 200**，祖先 **49 条**（全部是 TRAIN seed 3042–3091）。`|S|` 直方图（2..17）：`2`:6、`3`:15、`4`:10、`5`:18、`6`:17、`7`:10、`8`:8、`9`:11、`10`:13、`11`:9、`12`:12、`13`:14、`14`:11、`15`:13、`16`:8、`17`:3；平均 **9.017**、最小 2、最大 17。

A1 的状态源是 **50 条 TRAIN seed**；其中 3082 的 4 个状态全部被排除（`|S| = 0/0/0/18`），所以 178 个 proper 状态落在 **49 条** seed 上。

**观测从哪来。** A1 证据 JSON 存的是裁决，不是像素。训练所需的 obs-v3 张量（`play (4,112,96)`、`hud (56,160)`、`act_hist (8,18)`，`uint8`）由`.tmp_v420_setsup/capture.py` 在 env 角色引擎里重放 A0 自己录制的轨迹、在冻结状态帧上快照得到，并**要求冻结学生在捕获张量上重新给出的 argmax 等于 A1 记录里的 `student_cand_at_state`**，否则拒绝该状态。

| 项 | 值 |
|---:|---:|
| 捕获文件数 | 178 |
| 逐帧 HASH 校验一致 | 178 / 178 |
| 学生 argmax 复现 A1 记录 | **178 / 178** |
| 不一致 | 0 |

**划分（状态级、祖先不交叉）。** ancestor split: every state of seed s goes to holdout iff s % 5 == 4, else to train。训练 **138** 状态 / 39 祖先，留出 **40** 状态 / 10 祖先；`states_in_both = []`、`seeds_in_both = []`。状态不会被劈成两半（每个状态整体落在一边），祖先也不会两边都有，所以不存在同状态泄漏，也不存在同轨迹泄漏。

---

## 3 损失（跑前写死）

```text
主损失（唯一被优化）: L_set = -log sum_{a in S} pi(a|s)
诊断项（no_grad）    : L_unsafe = -log (1 - sum_{a not in S} pi(a|s))
梯度               : dL_set/dz_a = p_a - p_a/P * 1[a in S] with P = sum_{a in S} p_a: the target is the softmax renormalised inside S, so the model is pushed onto the set and stays indifferent inside it (|S| = 1 collapses to the ordinary CE)
```

**为什么和对单一 chosen 动作做 CE 本质不同**：A0/A0v minimise -log pi(a*|s) on one chosen action per slot; this minimises -log sum_{a in S} pi(a|s), which leaves every choice inside S at full probability

**诊断项的性质**：1 - sum_{a not in S} pi = sum_{a in S} pi, so L_unsafe must equal L_set up to float rounding; it is kept as a numerical cross-check through the complement and never enters backward()

---

## 4 训练

| 项 | 值 |
|---:|---:|
| 起点 | A0（同一 sha256，经冻结 v4.1.0 loader 载入并逐字核验） |
| 步数 / batch | 5000 / 128 |
| 优化器 | adam，lr 0.0001，wd 0.0，clip 0.5 |
| epochs / steps per epoch | 2500 / 2 |
| device | cuda（NVIDIA GeForce RTX 4070 SUPER） |
| 确定性 | `torch.use_deterministic_algorithms` = True，strict = True，warn_only = False |
| 吞吐 | 69.34 steps/s、4784.5 samples/s |

### 4.1 `L_set` 曲线（每 500 步截取；完整曲线每 100 步在证据 JSON 里）

| step | L_set | L_unsafe（诊断） | 集合内命中率 | grad pre-clip |
|---:|---:|---:|---:|---:|
| 500 | 0.000025 | 0.000025 | 1.000000 | 0.002 |
| 1000 | 0.000008 | 0.000008 | 1.000000 | 0.001 |
| 1500 | 0.000004 | 0.000004 | 1.000000 | 0.000 |
| 2000 | 0.000002 | 0.000002 | 1.000000 | 0.000 |
| 2500 | 0.000001 | 0.000001 | 1.000000 | 0.000 |
| 3000 | 0.000001 | 0.000001 | 1.000000 | 0.000 |
| 3500 | 0.000001 | 0.000001 | 1.000000 | 0.000 |
| 4000 | 0.000000 | 0.000000 | 1.000000 | 0.000 |
| 4500 | 0.000000 | 0.000000 | 1.000000 | 0.000 |
| 5000 | 0.000000 | 0.000000 | 1.000000 | 0.000 |

### 4.2 最终测量

| split | 状态数 | L_set | L_unsafe（诊断） | 集合内命中率 | p_set |
|---:|---:|---:|---:|---:|---:|
| A0 起点（**未训练**） | 138 | 7.335824 | 6.711018 | 0.108696 | 0.153422 |
| train（训练后） | 138 | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| A0 起点（**未训练**） | 40 | 5.766594 | 5.147290 | 0.200000 | 0.253331 |
| holdout（训练后） | 40 | 3.543519 | 3.105222 | 0.550000 | 0.561575 |

参考标尺：在 18 个候选上平均撒概率时，训练集 `L_set` = **0.713481**（命中率 0.4899）、留出集 `L_set` = **0.618246**（命中率 0.5389）。

### 4.3 怎么读这张表（拟合 vs 外推）

* **起点本身远差于均匀策略**：未训练的 A0 在这些近死状态上训练集 `L_set` = **7.335824**、集合内命中率 **0.108696**（均匀参考命中率 0.4899）——它就是在这些状态上把概率压在安全集合**外面**，这正是它在那里死掉的原因。
* **训练集被完全拟合**：138 个状态的 `L_set` = **0.000000**、集合内命中率 **1.000000**。这是欠约束的 138 个样本被反复看 2,500 个 epoch 的结果，**是拟合，不是能力**。
* **留出集只回到“随机水平”**：祖先不交叉的 40 个留出状态上，`L_set` 从 A0 的 **5.766594** 降到 **3.543519**、命中率从 **0.200000** 升到 **0.550000**；而均匀随机策略在同样集合大小下的命中率是 **0.5389**（`mean |S| / 18`）。训练把留出行为从“主动选错”拉回到了**随机水平**，没有拉出随机水平之上。
* **`L_unsafe` 与 `L_set` 的关系**：两者是同一量经补集计算，在训练窗口里直到小数点后 6 位都相等。只有在 `p_set` 掉到 float32 无法表示（< 1e-6）时才会分开：`L_unsafe` 饱和于 `ln(1e6)` = 13.8155，`L_set` 在对数空间不饱和。本表里 A0 起点行差 0.6248、留出行差 0.4383，这正是诊断项该暴露的东西，也说明它**没有**参与优化。

**跑前 smoke（不是结果）**：`.tmp_v420_data/T08/setsup/smoke.json`，20 步、独立 checkpoint 目录、读回差 **0.0**，用来先证明 GPU 路径、写盘与冻结 loader 读回都成立。

---

## 5 checkpoint 与读回

| 项 | 值 |
|---:|---:|
| 路径 | `/mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T08/setsup/checkpoints/a1set_seed0_step5000.pt` |
| 大小 | 2,941,273 B |
| sha256 | `7bacab5e23492f4bda7dc30cc8c256de8067ef25c6c08b559ed014219c2bcfba` |
| schema / arm / family | `v410-wp04-bc-arm-checkpoint/1` / `B0` / `probe` |
| 参数 | 731,802 |
| epochs / last_epoch | 2500 / 2500 |
| 冻结 loader 读回 | loaded=True、arm=B0、parameters=731802/731802、seed=0 |
| 读回差（CPU，8 行） | **logit max abs diff = 0.0** |

**权重可复现（机器核对）**：同一协议在两个进程上各跑一次 5,000 步，`state_dict` 的 sha256 = `0e45f6ae86a6403dadc0c1f80c4053e1177c42f671b69fe2d2d2b6db2f23d5fb`，逐张量相同 = **True**（不同张量 0 个）。checkpoint 文件 sha256 之所以会在两次运行间变化，只是因为 payload 里带 `created_at`；评估实际测的是与当前文件**逐比特相同**的权重，被评估的那份文件保留为 `a1set_seed0_step5000.evaluated-243b23.pt`（sha256 `243b23b3bdb1297d7cf6dd1ab64ebef0a1db647b2fa2c41f5ed7e14a0a13c1e9`）。

---

## 6 评测仪器

仪器：`tools/rl/v420_eval_shoot.py`（sha256 `7dbc2a3145562e54f4a7397fa63c8391938b6f6982f5d4b495a66b28fc6e8519`），`--shoot-pulse-period 48`、`--obs-window consecutive`、`--max-frames 19887`、stage 0 / normal / cpu。

**为什么必须换仪器。** tools/rl/v410_student_eval.py OR the v400 evaluator cannot report stage_clear at all: cand_to_mask ORs BTN_SHOOT into every frame, so no SHOOT rising edge ever reaches the engine and an end-of-stage wait can never be left (T08 CLEAR_REACHABILITY.md).  Therefore every pre-T08 '0/20' / '0/80' number is a measurement of an instrument that could not report a clear, and NO old number is comparable with the numbers below.

**新仪器正负对照（既有证据，逐位相同 mask）**：

| 对照 | period | stage_clear | frames | score | stop_reason |
|---:|---:|---:|---:|---:|---:|
| 正向（教师录制轨迹 + 脉冲） | 48 | true | 13195 | 2,631,430 | stage_clear |
| 负向（同轨迹，永不释放） | 1000000007 | false | 15564 | 163,945 | game_over |

两条对照只差 `shoot_pulse_period`，所以 `stage_clear` 的有无只能来自脉冲。

---

## 7 评测结果：两口径各 20 局

dev seeds **1000–1019**（20 个），每口径 20 局；`diag` = 带 `--stop-on-death`，`s1n` = 不带。

| arm | 口径 | n | 帧数均值 | 中位 | min | max | sd | 过关数 | 过关 seed | 结束原因 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0（A0 原样） | diag | 20 | 1650.400 | 1562.0 | 1397 | 2308 | 267.43 | 0 | — | {"death_stop": 20} |
| R0（A0 原样） | s1n | 20 | 2824.350 | 2863.0 | 2137 | 3478 | 360.67 | 0 | — | {"game_over": 20} |
| A1-set（本文件） | diag | 20 | 669.000 | 669.0 | 669 | 669 | 0.00 | 0 | — | {"death_stop": 20} |
| A1-set（本文件） | s1n | 20 | 1859.350 | 1768.0 | 1616 | 2634 | 256.23 | 0 | — | {"game_over": 20} |

**逐局 `stage_clear`（A1-set，两口径）**：

| seed | diag 帧数 | diag stage_clear | diag stop | s1n 帧数 | s1n stage_clear | s1n stop |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 669 | false | death_stop | 1795 | false | game_over |
| 1001 | 669 | false | death_stop | 1728 | false | game_over |
| 1002 | 669 | false | death_stop | 2634 | false | game_over |
| 1003 | 669 | false | death_stop | 2268 | false | game_over |
| 1004 | 669 | false | death_stop | 1769 | false | game_over |
| 1005 | 669 | false | death_stop | 1749 | false | game_over |
| 1006 | 669 | false | death_stop | 1750 | false | game_over |
| 1007 | 669 | false | death_stop | 1702 | false | game_over |
| 1008 | 669 | false | death_stop | 1845 | false | game_over |
| 1009 | 669 | false | death_stop | 1779 | false | game_over |
| 1010 | 669 | false | death_stop | 1767 | false | game_over |
| 1011 | 669 | false | death_stop | 1878 | false | game_over |
| 1012 | 669 | false | death_stop | 1616 | false | game_over |
| 1013 | 669 | false | death_stop | 1754 | false | game_over |
| 1014 | 669 | false | death_stop | 1906 | false | game_over |
| 1015 | 669 | false | death_stop | 1746 | false | game_over |
| 1016 | 669 | false | death_stop | 1616 | false | game_over |
| 1017 | 669 | false | death_stop | 1988 | false | game_over |
| 1018 | 669 | false | death_stop | 1625 | false | game_over |
| 1019 | 669 | false | death_stop | 2272 | false | game_over |

R0 同仪器的参考均值：diag **1650.400**、s1n **2824.350**（与判决门槛 1636.8 / 3011.5 分开记录：门槛是跑前写死的常量，不从这里的数据反推）。

**评测重跑一致性**：同一份权重（两个 checkpoint 文件只差 `created_at`）各跑 80 局，逐局比较 `frames` / `stage_clear` / `stop_reason` / `score` / `action_sequence_sha256`，全部一致 = **True**。

### 7.1 行为剖面：策略塌缩（这是帧数下降的机制，不是另一条证据线）

| arm | 口径 | 主导动作 | 主导动作占比（20 局均值） | 不同动作数均值 | action0（idle）占比 |
|---:|---:|---:|---:|---:|---:|
| r0 | diag | idle | 0.2926 | 9.90 | 0.2926 |
| r0 | s1n | idle | 0.3351 | 12.15 | 0.3351 |
| setsup | diag | NE | 0.5870 | 10.50 | 0.0630 |
| setsup | s1n | NE | 0.3745 | 11.50 | 0.0404 |

* **diag 口径 20 局只有 1 种长度**：全部 669 帧、分数全部 600、决策数全部 335。也就是说，训练后的策略在 20 个不同的 dev 种子上产生了**几乎同一条短轨迹**：它把 A0 那种四处闪避的行为换成了以 `NE` 为主的固定方向，在 669 帧处撞死。
* 这是集合监督在小样本上被拟合到极致的直接后果：138 个训练状态的 `L_set` 压到 0，代价是不同状态上的偏好被抹平成单一动作；留出 `L_set` = 3.543519 > 均匀参考 0.618246 已经预示了这一点。

### 7.2 两个门槛从哪来

`Document/Update/Alpha/v4.2.0/evidence/T04/dev_eval.json`（sha256 `577c3b2917c08b3efbd228a7d1d104d78a78d4f9e9ec61544b7d99d7968cb4fe`）里的 A0/R0 20 局均值就是门槛本身：diag **1636.800**、s1n **3011.500**（逐项核对：diag `matches_the_frozen_threshold` = True，s1n = True）。the T04 development evaluation ran the OLD instrument (tools/rl/v410_student_eval.py), which has no SHOOT release path and therefore cannot report stage_clear; the two thresholds are old-instrument R0 means

所以门槛与本次新仪器重测的 R0 基线并不完全相同（新仪器 R0：diag 1650.400、s1n 2824.350）——这是**旧仪器门槛 vs 新仪器基线**的差异，写在这里是为了不把它读成效应。结论对两者都成立：A1-set 的 669.000 / 1859.350 同时低于门槛和新仪器基线。

---

## 8 与 R0 的逐 seed 配对差

### diag（A1-set − R0，正数 = 帧数更长）

| seed | R0 帧数 | A1-set 帧数 | 差 | R0 clear | A1-set clear |
|---:|---:|---:|---:|---:|---:|
| 1000 | 1409 | 669 | -740 | false | false |
| 1001 | 1914 | 669 | -1245 | false | false |
| 1002 | 1499 | 669 | -830 | false | false |
| 1003 | 1557 | 669 | -888 | false | false |
| 1004 | 1410 | 669 | -741 | false | false |
| 1005 | 1605 | 669 | -936 | false | false |
| 1006 | 1397 | 669 | -728 | false | false |
| 1007 | 1408 | 669 | -739 | false | false |
| 1008 | 1431 | 669 | -762 | false | false |
| 1009 | 1492 | 669 | -823 | false | false |
| 1010 | 1588 | 669 | -919 | false | false |
| 1011 | 1413 | 669 | -744 | false | false |
| 1012 | 1912 | 669 | -1243 | false | false |
| 1013 | 1728 | 669 | -1059 | false | false |
| 1014 | 2138 | 669 | -1469 | false | false |
| 1015 | 2308 | 669 | -1639 | false | false |
| 1016 | 1567 | 669 | -898 | false | false |
| 1017 | 1494 | 669 | -825 | false | false |
| 1018 | 1945 | 669 | -1276 | false | false |
| 1019 | 1793 | 669 | -1124 | false | false |

配对差：均值 **-981.400**、中位 **-893.0**；改善 **0** / 变差 **20** / 持平 **0**。仅 A1-set 过关的 seed：—；仅 R0 过关的 seed：—。

### s1n（A1-set − R0，正数 = 帧数更长）

| seed | R0 帧数 | A1-set 帧数 | 差 | R0 clear | A1-set clear |
|---:|---:|---:|---:|---:|---:|
| 1000 | 2137 | 1795 | -342 | false | false |
| 1001 | 2654 | 1728 | -926 | false | false |
| 1002 | 2185 | 2634 | 449 | false | false |
| 1003 | 2741 | 2268 | -473 | false | false |
| 1004 | 2456 | 1769 | -687 | false | false |
| 1005 | 2579 | 1749 | -830 | false | false |
| 1006 | 2572 | 1750 | -822 | false | false |
| 1007 | 3280 | 1702 | -1578 | false | false |
| 1008 | 3181 | 1845 | -1336 | false | false |
| 1009 | 3000 | 1779 | -1221 | false | false |
| 1010 | 3091 | 1767 | -1324 | false | false |
| 1011 | 2540 | 1878 | -662 | false | false |
| 1012 | 2935 | 1616 | -1319 | false | false |
| 1013 | 2791 | 1754 | -1037 | false | false |
| 1014 | 3078 | 1906 | -1172 | false | false |
| 1015 | 3110 | 1746 | -1364 | false | false |
| 1016 | 2550 | 1616 | -934 | false | false |
| 1017 | 3478 | 1988 | -1490 | false | false |
| 1018 | 3064 | 1625 | -1439 | false | false |
| 1019 | 3065 | 2272 | -793 | false | false |

配对差：均值 **-965.000**、中位 **-985.5**；改善 **1** / 变差 **19** / 持平 **0**。仅 A1-set 过关的 seed：—；仅 R0 过关的 seed：—。

---

## 9 判定（跑前写死的双条）

| 判据 | 规则 | 实测 | 成立 |
|---:|---:|---:|---:|
| 能力判据 | stage_clear >= 1 over the 40 setsup episodes | 40 局里过关 **0** 局 | false |
| 主判据（两口径同时） | diag mean >= 1636.8 and s1n mean >= 3011.5 | diag **669.000**（≥ 1636.8 ? False）、s1n **1859.350**（≥ 3011.5 ? False） | false |

四条分支（跑前写死）：

| 分支 | 含义 | 本次 |
|---:|---:|---:|
| 都 hold | capability AND primary: first capability evidence (still single seed / n=20) |  |
| 只主判据 | frames improved but no clear |  |
| 只能力 | first clear, but the mean frames did not improve |  |
| 都不 hold | set supervision is not enough at this scale | **← 成立** |

**成立的分支：`neither`。** set supervision is not enough at this scale

---

## 10 测试与验收

| 侧 | 命令 | 汇总 | 退出码 |
|---:|---:|---:|---:|
| Windows | new_file | 24 passed in 3.63s | 0 |
| Windows | ten_files | 393 passed, 1 skipped in 16.78s | 0 |
| Windows | eval_shoot | 33 passed in 1.60s | 0 |
| Windows | all_files | 450 passed, 1 skipped in 17.72s | 0 |
| Windows | compileall |  | 0 |
| Windows | cli_help |   -h, --help            show this help message and exit | 0 |
| WSL | new_file | 24 passed in 2.36s | 0 |
| WSL | ten_files | 394 passed in 38.95s | 0 |
| WSL | eval_shoot | 33 passed in 2.22s | 0 |
| WSL | all_files | 451 passed in 38.36s | 0 |
| WSL | compileall |  | 0 |
| WSL | cli_help |   -h, --help            show this help message and exit | 0 |

**与派发基线的对账**：派发时写的是「全量 10 文件基线 419 passed」。本树实测 10 文件 = WSL **394 passed**、Windows **393 passed + 1 skipped**，与同一日 T08 [`A1_SAFE_SET_FULL.md`](A1_SAFE_SET_FULL.md) §10 记录的基线（394 / 393+1）逐字一致；本轮只**新增** `tests/test_v420_setsup.py`（24 个测试），既有的 10 个 v420 测试文件与 `tests/test_v420_eval_shoot.py` **一个字节都没有改**。因此 419 这个数在本树上没有复现出来：它不是本轮改动造成的差额（既有文件未被改动），差额只能来自派发时的另一份树或另一批并发追加。

---

## 11 不能主张什么

* this is not R1 and not a capability claim beyond the 40 episodes measured
* single training seed (0); n = 20 per ruler; no interval, no rate estimate
* no confirm seeds (61000-61499) and no legacy 60000-60199 were read
* the A1 states sit on A0's own TRAIN trajectories (3042-3091) at H=120; they are a diagnostic localisation, not a generalisation corpus
* the clear measurement is a development measurement: it is not G420-2 and it is not a pass rate
* the old instruments cannot report stage_clear, so no pre-T08 number is comparable with any number here

## 12 剩余阻塞

* the pre-registered decision rule is a neither branch: set supervision at this scale did not produce the both-hold outcome
* the mechanism behind the collapse is not separated: whether 138 states are simply too few, or set supervision itself pulls the policy off A0's behaviour, is not decided by this run (that would be a second factor, which this round was explicitly not allowed to add)
* one training seed and n = 20 dev episodes per ruler: no interval, so this is a description of 40 episodes and not an estimate of any rate

---

## 13 复现

```bash
ROOT420=/mnt/d/WorkShop/Touhou_RL; PY420=/root/v400-venv/bin/python
cd $ROOT420
bash .tmp_v420_setsup/capture_all.sh      # 178 个状态的 obs-v3 捕获（引擎）
bash .tmp_v420_setsup/train.sh smoke      # 20 步 GPU/写盘/读回 smoke
bash .tmp_v420_setsup/train.sh main       # 5,000 步集合监督训练
bash .tmp_v420_setsup/make_eval_data.sh   # 4 个私有 cp -a data 目录
bash .tmp_v420_setsup/eval_all.sh         # 2 臂 × 2 口径 × 20 局
$PY420 .tmp_v420_setsup/aggregate.py      # 配对 + 判定
$PY420 .tmp_v420_setsup/accept.py --side wsl
$PY420 .tmp_v420_setsup/emit_evidence.py
$PY420 .tmp_v420_setsup/emit_evidence.py --check   # 证据与产物一致
```

引擎：`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`；env 角色 `xvfb-run -a -s "-screen 0 1024x768x24"`；data 目录私有 `cp -a`；每条命令 ≤ 10 分钟。

## 14 产物

| 路径 | 内容 |
|---:|---:|
| `configs/v420/a1-setsup-train.json` | 预注册配置（跑前冻结） |
| `touhou_rl/v420/setsup.py` | 集合监督模块（损失、过滤、划分、训练） |
| `tools/rl/v420.py setsup` | 新增子命令（默认 dry run，`--execute` 才训练） |
| `tests/test_v420_setsup.py` | 24 个 engine-free 测试 |
| `Document/Update/Alpha/v4.2.0/evidence/T08/A1_SETSUP_TRAIN.md` | 本文件 |
| `Document/Update/Alpha/v4.2.0/evidence/T08/a1_setsup_train.json` | 机器生成的证据负载 |
| `.tmp_v420_setsup/` | 全部脚本（capture / train / eval / aggregate / emit / accept / 对账） |
| `.tmp_v420_data/T08/setsup/obs/` | 178 个状态的 obs-v3 捕获（.npz） |
| `.tmp_v420_data/T08/setsup/checkpoints/` | 训练 checkpoint（含被评估那份） |
| `.tmp_v420_data/T08/setsup/eval/` | 16 个评测分块汇总（80 局） |
| `.tmp_v420_data/T08/setsup/eval_rerun1_243b23/` | 第一次评测（用于重跑一致性） |
| `.tmp_v420_data/T08/setsup-eval/{arm}-{ruler}/` | 4 个私有 cp -a data 目录 |

训练报告 `.tmp_v420_data/T08/setsup/train.json`（sha256 `cfbd230eebf86d1bea2b3c7536f4612ed8337107e865384f55b8dea9d4dbc2e0`）、评测汇总 `.tmp_v420_data/T08/setsup/eval_aggregate.json`（sha256 `49098092f78b8f4b8a5e40c391c55be626cc0a98ba78eb080d0dbbbea68599a2`）。
