# EXP-P3-002 BC v1：教师软标签行为克隆 + S1N 5 局评测（P3 真实起步）

- 日期 / 切片：2026-11-09 / **P3-WP1-b**（首个用教师标签训练的像素学生 + 首次 5 局 S1N）
- 依赖：[EXP-P1-018](<EXP-P1-018-D0-scaleup.md>)（D0 采集，教师标签+观测）、
  [EXP-P1-012](<EXP-P1-012-g1-evaluation.md>)（S1N 口径与教师 20 局）、
  13-P3 §2.4（教师接口 `π_t=softmax(scores/τ)`，loss = `KL(π_t‖π_s)` + bomb BCE）、
  [EXP-P3-001](<EXP-P3-001-obs-spec-bc-smoke.md>)（obs-v1 冻结、V0 88k 与评测通路）
- 结论先说：**BC v1 训练成功、评测通路打通，但离教师仍差一个数量级**。S1N（`--stage 0`
  normal，cap 19,887，`--stop-on-death`）5 局**全部首死**（`death_stop`），均值
  **1,084.0 帧 / 1 死 / 0 过关**，是 V0 同口径（636 帧）的 **1.70×**，但只有教师
  （13,565.2 帧 / 0 死 / 20/20 过）的 **8.0%**。离线 val top-1 59.6%、KL 0.362，看似"及格"，
  但 **macro recall 只有 21.2%**、动作 0 预测占比 69%（教师 42%）、8 个 focus 类里有 5 个从不输出
  —— 这是**模式塌缩**，不是学会了教师。本卡**不是 G3 通过、不是 PSN、不是 confirm split**。

## 1. 范围与预注册

1. 本切片只做：D0 到 50k 后训练 BC v1（软标签 KL + bomb BCE，τ 在 val 上定标，按
   shard/seed 整片切分）、5 局 S1N 评测、诊断与 DAgger 首轮配置、测试与实验卡。
2. **不做**：DAgger 实跑、PSN、多 stage、obs-v2、模型结构搜索、教师提速、引擎/协议/依赖改动、
   重建 build、删除文件。评测**只用 1 个引擎**（v2g 6 引擎与零售 trace 批跑未受影响）。
3. **S1N 口径 = `--stage 0`**（0-based，`stage 0` 就是真 Stage 1）。证据见 §2.3；这也是
   EXP-P1-012（G1）与 EXP-P1-018（D0）实际用的口径。
4. 训练用当前 D0 快照（50,118 决策 / 15 片，manifest `updated_at=20:06:37`）；采集在训练开始后
   继续增长到 53,512 / 16 片（20:09:45），**本卡不把这 3,394 条算进训练集**，实际样本数如实记录。

## 2. 数据

### 2.1 D0 实况（训练用快照）

| 项 | 值 |
|---|---|
| 目录 | `.tmp_v400_d0/d0/`（`shards/<key>/{labels.jsonl,obs/play64.npy,obs/hud.npy,shard.json}` + `manifest.json`） |
| shards | **15**（`dual-s0d1seed{1..15}`，每片 = 一局完整 episode = 一个种子） |
| 决策数 | **50,118**（= 样本数） |
| stage frames | 200,472（每局 13,096–13,880，2 帧/决策） |
| 观测字节 | 1,270,206,252 B（play64 `(N,4,64,64)` + hud `(N,56,160)`，uint8） |
| stage / 难度 | 全部 `stage 0`（= 真 Stage 1）/ `difficulty 1`（normal） |
| 教师表现 | 20/20 过面、0 死亡（教师在此面上的标签来自"从不死"的轨迹） |
| 对齐 | HASH 对齐 107,024/107,024 全等（`hash_alignment_rate=1.0`，16 片统计），obs-v1 摘要 `24f0ddac…` |
| manifest sha256 | `eb3f80094f81976730df9130e355508526d2d9cd3f6701b3505b369cdad57a91` |

加载器：`load_teacher_samples(require_observations=True)`（逐片复核 obs-v1 版本/摘要、plane
形状与 `frames[i-3..i] -> mask[i+1]` 对齐、label 行与 `decision_frame` 一致）；实测 **8.0 s / 50,118 条**，
`gate_waived=false`（真达到 50k 契约，没有放宽门槛）。

### 2.2 整片切分（禁止同一局跨 split）

按 shard 排序 + `split_seed=0` 确定性置换，70/15/15 最大余数分配（每 split ≥1 片），
`check_episode_isolation` 再从未篡改的 shard→seed 映射反推一遍，**同一种子跨 split 直接报错**。

| split | shards | seeds | 决策数 | 占比 |
|---|---|---:|---:|---:|
| train | 11 | 1,2,3,4,5,6,10,11,12,13,14 | **36,662** | 73.2% |
| val | 2 | 7, 9 | **6,812** | 13.6% |
| test | 2 | 8, 15 | **6,644** | 13.3% |
| 合计 | 15 | 1–15 | 50,118 | 100% |

- 切分签名 `879153def90729424bcc50b7108f9b4f07d1942b7f279614d6cc55eb355a5bdc`
  （写进 checkpoint 的 `provenance.split_signature`，可反推）。
- **评测种子 1000–1004 不在 D0 的任何一片里**（D0 只用种子 1–15），所以 S1N 的 5 局是**完全没见过的种子**；
  test 片（8、15）只用于离线指标，绝不参与训练或选模。
- 一片 = 一局 = 一个种子，因此"按 shard 切"等价于"按 episode 切"，结构上不可能有轨迹泄漏。

### 2.3 为什么 S1N 是 `--stage 0`（顺手修正一处口径错误）

`RESET <stage>` 与 `STATE.f1` 都是 **0-based**：`bench/human_reference.json` 里
`stage_index 0 ↔ Stage1 (13,258 游戏帧)`；D0 用 `--stage 0` 打出的每局都是 **13.2k 帧换面**
（13,196/13,258 = 99.5% 吻合），若 `stage 0` 是 Stage2 应该 ~18.5k 帧。EXP-P1-012（G1）的 S1N
定义就是 `--stage 0`，`stage_clear` 由 `STATE.f1` 是否离开 0 判定。

**因此 `tools/rl/v400_student_eval.py` 原先的 `DEFAULT_STAGE = 1`（注释写 "S1N = stage 1"）是错的：
`--stage 1` 起的是真 Stage 2。** 本卡把它改成 `DEFAULT_STAGE = 0` 并在 docstring 记录证据。
同一 v0 checkpoint 的单局对照（seed 1000，`--stop-on-death`）：

| `--stage` | 首死帧 | score | `stage_start→stage_end` |
|---|---:|---:|---|
| 0（S1N，本次） | 636 | 750 | 0→0 |
| 1（真 Stage 2） | 924 | 29,197 | 1→1 |

**推论**：[EXP-P3-001](<EXP-P3-001-obs-spec-bc-smoke.md>) 的 V0 行（1,634 帧 / 3 死 / 0 过）
是在 **真 Stage 2** 上测的，**不是 S1N**，不能与本卡的 S1N 行逐局比较。为了让对比成立，本卡用
**同一个脚本、同一把尺（stage 0 / cap 19887 / stop-on-death / seeds 1000–1004）重测了 V0**，
见 §5.2。这条也提醒：`bench/scoreboard.md` 里所有用 `suite.stages[0]=1` 跑出来的 "S1N" 行可能
同样落在 Stage 2 上，**需要单独复核**（本卡未改 bench，只报事实）。

## 3. 模型、损失与 τ 定标

| 项 | BC v0（对照） | **BC v1（本卡）** |
|---|---|---|
| 结构 | play 3×Conv(32/64/64)+GAP、hud 3×Conv(16/32/32)+GAP、concat→Linear128 | 同构，**加宽**：width 48 / hidden 256（play Conv 48/96/96、hud 24/48/48、Linear 256） |
| 参数量 | 88,436 | **204,260** |
| 输入 | play64 4 帧堆叠 + hud 56×160（obs-v1） | 同 |
| 输出 | move_logits[18] + bomb_logits[2] | 同 |
| 标签 | 采集策略自己的按键掩码（硬标签 CE） | **教师 `scores[18]` 软标签** |
| 损失 | CE(move)+CE(bomb) | **`KL(softmax(scores/τ) ‖ π_s)` + 1.0 × BCE(bomb; pos_weight)** |

- **KL 方向**：`F.kl_div(log π_s, π_t, reduction="batchmean")` = `KL(π_t‖π_s)`，与 13-P3 §2.4 一致；
  numpy 实现与 torch 在测试里逐值对齐（`test_kl_matches_torch_kl_div`）。
- **`verified_mask` 只作诊断**：不参与 loss、不参与加权（教师写 `scores` 时已内部打过折扣，
  再乘一次会重复折扣）。`teacher_soft_labels(..., verified_mask=...)` 显式接受但**不改变目标**，测试锁住。
- **bomb BCE 与 pos_weight**：D0 50,118 条里 **bomb 正例 = 0**（教师 0 死亡、从不用 bomb）。
  `pos_weight = neg/pos` 在 pos=0 时不存在有限值，实现**退化为 1.0 并标记 `identifiable=false`**，
  同时在卡里写明 bomb accuracy=100% 是**空指标**（模型从头到尾只预测 0，`bomb_pred_positive_rate=0.0`）。
- **τ 定标**：τ ∈ {1.0, 2.0, 4.0}，每个 τ 用**等预算 90 s** 从头训练一个探针模型（同种子、同批序），
  在 val 的固定 2,048 子集上每 100 步评估，取各探针最佳点；排序规则**预注册**为
  "val 教师 top-1 一致率最大 → val 教师 argmax 动作的 NLL 最小 → τ 小者优先"。

| τ | val top-1 | val KL | val NLL(argmax) | H(π_t) on val |
|---:|---:|---:|---:|---:|
| 1.0 | 0.6108 | 1.2189 | 1.4671 | **0.574** nats（max p 0.906） |
| **2.0（选中）** | **0.6143** | 0.3201 | 1.7344 | **2.270** nats（max p 0.436） |
| 4.0 | 0.6079 | 0.0610 | 2.2545 | **2.790** nats（max p 0.177） |

**KL 不能用来给 τ 排名**（这就是排序规则不用 KL 的原因）：目标熵随 τ 从 0.574 涨到 2.790 nats
（ln18=2.89，τ=4 时目标已接近均匀），同一个学生自然在 τ=4 得到最小 KL（0.061）。表中三个 τ 的
top-1 差 0.6 个百分点，而 2,048 样本二项噪声约 ±1.1% ——**τ 在这个数据规模上不可辨识**，
"选中 τ=2"应读作"三者等价，取规则第一条的胜者"，不是调优结果。

## 4. 训练与离线指标

### 4.1 训练

| 项 | 值 |
|---|---|
| 设备 | RTX 4070 SUPER（Windows 原生 CUDA，与 v2g/零售批跑不冲突） |
| 预算 / 实测 | `--train-seconds 480`（探针 3×90 s 另计）；**wall 762.7 s = 12.7 min**（≤15 min） |
| 优化 | AdamW lr 2e-3, wd 1e-4, batch 256 |
| 步数 / 样本 | **27,341 step / 6,999,296 样本**（= 36,662 决策的 **191 个 epoch**），14.5k 样本/s |
| checkpoint | `.tmp_v400_p3v1/checkpoint-v1.pt`，843,607 B，sha256 `dd851b3182470b34bd4ddd1384c5f52ab65dc6f4de140a8ed21fde91e485338b` |
| 选模 | **best-by-val-top1**：第 2,200 步（2,048 子集 top-1 0.6084）；之后 train loss 继续降到 0.017 而 val 不再改善 → 过拟合 |

训练曲线摘要（2,048 子集）：首评（step 100）top-1 0.3999 / KL 0.450；最佳（step 2,200）
top-1 0.6084 / KL 0.350；末评（step 27,341）top-1 0.5767 / KL 0.362；train loss 0.473 → 0.017。

### 4.2 离线指标（最终 checkpoint，整 split 全量）

| split | 决策数 | top-1 | top-3 | KL(π_t‖π_s) | NLL(教师 argmax) | H(π_t) | macro recall | 多数类基线 | bomb acc | bomb 正例 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 36,662 | 0.7291 | 0.8750 | 0.2351 | 1.4495 | 2.2680 | 0.2992 | 0.4445 | 1.0000 | 0 |
| **val** | 6,812 | **0.5960** | 0.7538 | **0.3623** | 1.7811 | 2.2699 | **0.2121** | 0.4403 | 1.0000 | 0 |
| **test** | 6,644 | **0.5826** | 0.7337 | **0.3792** | 1.8098 | 2.2673 | **0.2085** | 0.4222 | 1.0000 | 0 |

（train 行是 2048 子集口径的训练集评估 0.7432；上表 train 列来自全量诊断脚本，两者差 1.4 个点，
说明**只看 top-1 会被多数类带偏**，见 §6.1。）

### 4.3 top-1 之外的三个事实（同一 checkpoint，`diagnostics.json`）

1. **多数类是"不动"**：教师 argmax 直方图里动作 0 = 42.2–44.5%，所以**瞎猜 0 就有 42–44% top-1**；
   BC v1 的 58.3%（test）只比多数类高 **16 个百分点**，而 macro recall 只有 **20.9%**。
2. **模式塌缩**：test 上模型把 **69.1%** 的决策判成动作 0（教师 42.2%）；8 个 focus 类（10–17）里
   **10、13、14、15、16、17 六类一次都没有被输出**（recall 0），11/12 只有 0.343/0.390。
3. **教师 margin 结构性贴地**：82.8%（val）的决策 top1−top2 差值 **恰好 5.0**（生存项地板价），
   教师其实在"近似平局"里用对齐/边缘等 tie-break 选动作。BC v1 在平局上一致率 63.5%，
   在**教师更有把握（gap>5）的状态上只有 40.7%** —— 越关键的少数状态越不会。

## 5. S1N 5 局评测（stage 0 = 真 Stage 1，normal，cap 19,887，`--stop-on-death`）

引擎：`/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern`（sha256 `591c521d…`），
`xvfb-run` + env role + 共享内存像素读回，**同一时刻只有 1 个引擎**，私有 data-dir
`/root/v400-data/p3v1/game`，seeds **1000–1004**（D0 未出现过），`privilege=deploy_pixel`。

### 5.1 逐局结果

| seed | frames | deaths | cleared | score | end | exit | stage | wall s |
|---:|---:|---:|---|---:|---|---|---|---:|
| 1000 | 740 | 1.0 | False | 1,900 | death_stop | controller | 0→0 | 26.7 |
| 1001 | 1,124 | 1.0 | False | 18,012 | death_stop | controller | 0→0 | 30.5 |
| 1002 | 1,330 | 1.0 | False | 21,782 | death_stop | controller | 0→0 | 35.1 |
| 1003 | 1,122 | 1.0 | False | 18,724 | death_stop | controller | 0→0 | 28.1 |
| 1004 | 1,104 | 1.0 | False | 24,955 | death_stop | controller | 0→0 | 24.5 |
| **均值** | **1,084.0** | **1.0** | **0/5** | **17,074.6** | death_stop ×5 | controller ×5 | 0→0 | 29.0 |

中位数 1,122 帧；5/5 首死即停（`--stop-on-death`），0/5 过关，`deaths` 恒为 1 是规则的产物，
不是"死得少"。`exit=controller` 表示由控制器停（不是引擎退出/游戏结束）。

### 5.2 与 V0 / 教师对比（**同一把尺**：stage 0 / cap 19887 / stop-on-death / seeds 1000–1004）

| 策略 | 种子数 | 均值 frames | 均值 deaths | 过关 | 均值 score | 备注 |
|---|---:|---:|---:|---:|---:|---|
| **BC v1（本卡）** | 5 | **1,084.0** | 1.0 | **0/5** | **17,074.6** | τ=2，204,260 参数，教师软标签 |
| BC v0（**同口径重测**） | 5 | **636.0** | 1.0 | 0/5 | 750.0 | 5 个种子帧数完全相同（策略近似常量） |
| BC v0（EXP-P3-001 历史行，`--stage 1`） | 3 | 1,634.0 | 3.0 | 0/3 | 62,989.7 | **真 Stage 2，不是 S1N，不可比** |
| 教师（EXP-P1-012，`--stage 0`） | 20 | **13,565.2** | **0.0** | **20/20** | — | min 12,812 / max 15,204 |

- BC v1 / BC v0 = **1.70×** 帧数（1,084.0 vs 636.0），score **22.8×**（17,074.6 vs 750.0）。
- BC v1 / 教师 = **8.0%** 帧数（1,084.0 / 13,565.2），**0/5 vs 20/20 过关**，死亡 1.0 vs 0.0。
- 结论：**BC v1 明显优于 V0，但离教师仍是"能不能活过前 8% 面"的差距** —— 这是 P3 的真实起点，
  不是 G3 通过。本行也**不是 PSN**（PSN = 8 面连打，本卡只打 stage 0 一面）。

## 6. 诊断：为什么离线 59.6% 却在闭环只有教师的 8%

### 6.1 数据量：过拟合早于"学会"

36,662 条训练决策跑 191 个 epoch，train KL 0.235 / NLL 1.45 对 val KL 0.362 / NLL 1.78，
**best 出现在第 2,200 步（8% 的训练进度）**。这条曲线说明瓶颈不是算力或 epoch：
再训只会更过拟合。真正的瓶颈是**决策的状态多样性**——15 局全部来自"教师从不死"的成功轨迹，
D0 里**没有任何死亡状态、也没有任何"已经偏离轨迹后怎么救"的状态**。

### 6.2 动作表示与类别不平衡：top-1 是假象

教师 42–44% 的动作是"不动"（动作 0），BC v1 预测出 69% 动作 0、8 个 focus 类里 6 个从不输出、
macro recall 0.209。**softmax + 不加权的 KL 会奖励"押多数类"**：把 42% 的动作 0 全对、
其余 17 类近乎放弃，就能拿到 ~58% top-1。闭环里"只会不动"= 不会躲，1,084 帧首死完全自洽。

### 6.3 教师标签本身是"近似平局 + tie-break"

82.8% 的决策 top1−top2 恰好 5.0（生存项地板价），教师是靠对齐/边缘/上界等小项在平局里选方向；
而这些小项来自教师的**几何搜索**（它知道弹幕的几何结构），学生只有 4 帧灰度。学生能学到
"分布的骨架"（KL 0.36），但**学不到 tie-break 的理由**：平局上 63.5%，教师有把握时反而只有 40.7%。

### 6.4 帧堆叠 / 观测

obs-v1 是 4 帧 64×64 灰度 + 1 帧 HUD。1 个决策 = 2 逻辑帧，所以视野只有 **8 个逻辑帧**；
没有动作历史（obs-v2 预留）、没有 focus/shoot 释放状态、没有敌人的运动矢量。对"平局里选方向"
这种需要速度信息的任务，**观测容量本身可能就是瓶颈**（本卡不能把这条与 6.3 分开量化）。

### 6.5 bomb 与"死亡状态"缺失

50,118 条里 bomb 正例 0 → bomb 头不可辨识（准确率 100% 是空指标）。更严重的是**没有死亡样本**：
教师 0 死亡意味着学生从没见过"必须用 bomb/必须反向"的状态。这不是加权能修的，是**数据分布**问题。

### 6.6 闭环偏差 → DAgger 是直接解法

BC 在教师分布上训练、在学生自己造出的分布上测试。v1 每局在 ~540 个决策后就离开了教师分布，
之后没有任何一条训练样本覆盖它看到的状态（这是分布偏移，不是容量不够）。DAgger 的
"在学生访问的状态上问教师"正好补这一块，且**不需要新 stage、不需要教师提速**（教师在本面
~16.8 决策/s、0 死亡，标注成本可控）。

## 7. DAgger 首轮最小配置（下一步，尚未执行）

| 项 | 取值 | 依据 |
|---|---|---|
| 轮数 K | **1** | 先验证"在学生状态上标注"是否有增量；无增量就不加轮次 |
| 新决策数 M | **50,000** | ≈ D0 的一倍；按 v1 实测 542 决策/局 ≈ **92 局**（β 注入后会更长，取 70–95 局） |
| 起点/执行策略 | **β = 0.2**：80% 执行学生动作、20% 执行教师动作 | 纯学生（β=0）每局 ~540 决策就死，出不了中段状态；β=0.2 让轨迹活到教师能标注的中后段 |
| 教师标注比例 | **100%** 的新状态（β=0 的标注策略），标签 = 教师 `scores[18]` 软标签 | DAgger 定义；标注与 D0 同一条 shadow 通路（`v400_d0_dual.py` 已验证 env+shadow 同帧 HASH 对齐 100%） |
| 聚合 | 50,118（D0）+ 50,000（DAgger）= **100,118**，1:1 | 保留原始分布，防止只学到"死亡附近"的偏置 |
| 切分 | 仍按**整局** 70/15/15；DAgger 用**种子 2000+**，评测仍用 **1000–1004**（尺子不变） | 禁止同一局跨 split |
| 重训 | 同一配方（τ 在 val 上定标、KL+BCE、≤15 min GPU） | 与 v1 可比 |
| 验收门（预注册） | S1N 5 局均值 **≥ 2× v1（≥2,170 帧）** 或 **≥ 教师 30%（≥4,070 帧）**；否则记为 null，转向 6.2/6.4（类平衡 KL 或 obs-v2） | 没有门槛的 DAgger 会自欺 |
| 成本 | ≈ 92 局 × ~28 s ≈ **43 min 单引擎** + 13 min GPU | 本轮评测实测 28–35 s/局 |
| 预防措施 | 同时记录学生/教师动作分歧率、分歧状态的教师 margin、每轮 macro recall | 否则"KL 降了但不会躲"会重演 |

## 8. 测试

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p3v1/pytest -q tests/test_v400_student.py
# → 26 passed in 1.96s, exit 0
```

新增 7 个用例（连同原有 19 个 = 26）：

1. `test_teacher_soft_labels_and_kl`：`softmax(scores/τ)` 归一/非负、argmax 与 τ 无关、τ→∞ 均匀、
   τ≤0 与非 18 维报错、`verified_mask` 不改变目标（且形状错要报错）、`KL(p‖p)=0`、KL 不对称、
   非 log-prob 输入报错。
2. `test_kl_matches_torch_kl_div`：numpy `KL(π_t‖π_s)` 与 `F.kl_div(log π_s, π_t)` **逐值相等**
   （方向不会悄悄反掉）。
3. `test_tau_selection_ranks_top1_not_kl`：合成指标里 top-1 最高者胜（即使 KL 更差）、
   tie-break 用 NLL(argmax) 而不是 KL、缺键与空候选报错、rationale 里带"entropy"解释。
4. `test_shard_split_is_episode_isolated`：15 片 → 11/2/2、可复现、换 `split_seed` 会变、
   每 split ≥1、重复 shard 键报错、片数不足报错、**同一种子跨 split 报错**、签名与顺序无关。
5. `test_bomb_pos_weight_and_unidentifiable_case`：0 正例 → `pos_weight=1.0` 且 `identifiable=False`、
   1/99 → 99、clamp 到 100、非 0/1 报错。
6. `test_checkpoint_summary_verifies_version_and_digest`：`bc_version`/`tau`/`d0`/`splits`/`metrics`
   五个键必须在、`parameters` 必须等于 state_dict 参数数（BN buffer 不计）、
   **obs 摘要被改 → `ObsSpecMismatch`**、schema 错/参数数错/bc_version 错全报错。
7. `test_eval_schema_death_stop_stage_clear_and_parser`：`DEFAULT_STAGE == 0`、parser 默认 5 个种子、
   `--stop-on-death` 生效、`death_stop`/`stage_clear` 是合法 end、`cleared` 必须与
   `stage_start/stage_end` 一致、非法 `exit`/错类型/非法 end 全拒绝。

`python -m compileall -q touhou_rl\v400\student tools\rl\v400_student_eval.py tools\rl\v400_student_train_teacher.py`
→ 退出码 **0**。

## 9. 命令与退出码（可复现）

```powershell
# Windows（GPU）：训练 BC v1（τ 探针 3×90 s + 终训 480 s，总 wall 762.7 s）
cd D:\WorkShop\Touhou_RL; $env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python tools\rl\v400_student_train_teacher.py --d0-root .tmp_v400_d0/d0 `
  --checkpoint .tmp_v400_p3v1/checkpoint-v1.pt --metrics .tmp_v400_p3v1/train-metrics.json `
  --split-json .tmp_v400_p3v1/split.json --taus 1.0,2.0,4.0 `
  --sweep-seconds 90 --train-seconds 480 --max-total-seconds 900 `
  --eval-every 100 --val-eval-samples 2048 --batch-size 256 `
  --width 48 --hidden 256 --seed 0
# → exit 0
```

```bash
# WSL：S1N 5 局（stage 0 = 真 Stage 1，cap 19887，--stop-on-death，单引擎）
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p3v1/run_eval.sh \
  /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p3v1/checkpoint-v1.pt \
  /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p3v1/eval
# → exit 0

# WSL：同口径重测 V0（历史行的 stage 口径修正，见 §2.3）
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p3v1/run_eval.sh \
  /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p3/checkpoint-v0.pt \
  /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p3v1/eval-v0-s1n
# → exit 0

# WSL：stage 语义对照（v0，seed 1000，--stage 1 = 真 Stage 2）
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p3v1/probe_stage.sh 1 \
  /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p3v1/probe-stage1 1000
# → exit 0
```

```powershell
# Windows：离线诊断（多数类基线 / macro recall / 逐动作 / margin 分层）
python .tmp_v400_p3v1\analyze.py
# → exit 0

# Windows：测试
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p3v1/pytest -q tests/test_v400_student.py
# → 26 passed, exit 0
```

产物（均在 `.tmp_v400_p3v1/`）：`checkpoint-v1.pt`（843,607 B，sha256 `dd851b31…`）、
`train-metrics.json`、`split.json`、`diagnostics.json`、`train.log`、`analyze.log`、
`eval/{episodes.jsonl,summary.json}`、`eval-v0-s1n/…`、`probe-stage1/…`、`run_eval.sh`、
`probe_stage.sh`、`probe_env.sh`、`analyze.py`。

## 10. 变更与产物

| 文件 | 变更 |
|---|---|
| `touhou_rl/v400/student/teacher_bc.py` | **新增**：整片切分 + 泄漏检查、`softmax(scores/τ)`、`KL(π_t‖π_s)`、bomb `pos_weight`（0 正例的显式退化）、τ 选择规则、checkpoint 摘要校验、`BC_VERSION`/`TEACHER_BC_DISCLOSURE`（torch-free，可单测） |
| `tools/rl/v400_student_train_teacher.py` | **新增**：D0 → 切分 → 软标签训练 → τ 探针 → 终训 → checkpoint/metrics/split 落盘 |
| `tools/rl/v400_student_eval.py` | `DEFAULT_STAGE=1→0`（S1N 口径修正，§2.3）；新增 `--stop-on-death`、`stage_clear` 终止、`stage_start/stage_end/cleared/exit` 记录与校验、V0/教师对照常量、按 checkpoint 的 `bc_version` 自动选披露文本、`build_parser()` 可测 |
| `tests/test_v400_student.py` | +7 用例（19 → 26 passed） |
| `Document/Update/Alpha/v4.0.0/experiments/EXP-P3-002-bc-v1-teacher.md` | 本卡 |

未改：引擎、协议、依赖、第三方、`build/`；未删除任何文件；未提交。

## 11. 限制（不要过度解读）

1. **不是 G3、不是 PSN、不是 confirm split**。5 局、1 个 stage、1 个难度；`n=5` 的均值不构成策略推断。
2. **`--stop-on-death` 改变了"死亡数"的含义**：本卡 5 局 deaths=1.0 是规则给的，不能与
   `game_over` 行的 3.0 直接比；能比的是 `frames` / `score` / `cleared`。
3. **V0 历史行（1,634 帧）在真 Stage 2 上**（§2.3），与 S1N 不可比；本卡的 V0 对照是重测的 636 帧。
4. **τ 不可辨识**（三个候选差 0.6 点、噪声 ±1.1%）；"τ=2"是规则选择的产物，不是调优结论。
5. **bomb 指标无效**：0 正例，accuracy=100% 只是"永远预测 0"。
6. **BC v1 的 val/test 数字是离线、单步、教师轨迹分布**上的；闭环 1,084 帧才是它的真实水平。
7. D0 在训练开始后继续增长（53,512/16 片）；本轮**没有**使用这 3,394 条，下一轮可以直接用。
8. 本卡未复核 `bench/scoreboard.md` 的 S1N 行是否也落在 Stage 2（§2.3 只给出了可得证据与推论）。
