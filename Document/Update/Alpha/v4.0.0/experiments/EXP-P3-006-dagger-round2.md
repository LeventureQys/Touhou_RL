# EXP-P3-006 DAgger 第二轮：用**已被证明有效的两个因子**（`geo_restore` × 数据量）把 P3 学生从 1,816 推向 ≥2,500 帧

- 日期 / 切片：2026-11-10 / **P3-WP6**（EXP-P3-004 §5.1 的 `geo_restore` + EXP-P3-004/005 的"数据量"两项，第一次**同时**上）
- 依赖：[EXP-P3-003](<EXP-P3-003-dagger-round1.md>)（DAgger R1 采集器、β=0.2、HASH 对齐语义）、
  [EXP-P3-004](<EXP-P3-004-label-signal-2x2.md>)（`geo_restore` +447.8 帧，5/5 同向，t=24.90；数据量 8→27 片 = +552.2 帧）、
  [EXP-P3-005](<EXP-P3-005-margin-resolution.md>)（新尺 `consecutive` 的 F 臂 1,816.0；`verify_k` / `margin_cap` 两个因子被判死；
  教师 30 % = 4,070）

> 本卡不是 G3 通过、不是 PSN、不是 confirm split；全部 S1N 数字都在**新尺** `--obs-window consecutive` 上。

## 0. 结论先说（五句话）

1. **两个"已被证明有效"的因子，在 127,304 条数据上只活下来一个**：数据量把 S1N 从 **1,816.0 → 2,078.8 帧**
   （+262.8，**新旧最好成绩**），而 `geo_restore` 从 +447.8 帧**反转成 −1,025.4 帧**（5/5 反同向，t=−3.75）。
   预注册判据 **FAIL**（均值 <2,500、macroR <0.25 两条腿都没过）。
2. **数据量的收益只有线性外推的 45 %**：实测 **5.22 帧 / 1k 决策**，而 EXP-P3-004/005 的 8→27 片斜率是
   **11.53 帧 / 1k**。按实测斜率，到 2,500 还需要 **80,765 决策 ≈ 2.55 h**（1 个锁步对），到"教师 30 %"（4,069.6）
   还需要 **381,729 决策 ≈ 12.0 h**。
3. **`geo_restore` 的机制这次被完整复现了，但机制与指标分家**：目标熵 2.2398 → **2.8677**（ln18 = 2.8904，几乎均匀）、
   动作 0 占比 0.583 → **0.283**、预测类数 8 → **14**、focus 类覆盖 2/8 → **5/8**。
   它确实"抑制了模式塌缩"，代价是闭环寿命腰斩 —— **不塌缩 ≠ 活得久**。
4. **学生仍然不会"基本生存"**：闭环探针显示对照臂（2,078.8 帧的那一臂）开局**连续 180 个决策（720 帧）站着不动**、
   首 250 决策 76.8 % 是动作 0、**0/8 的 focus 方向一次没用过**、整局只有 3–4 个不同动作。它活得久，是因为它**不动**。
5. **R2 采集是 partial**：目标 100,000 决策，**写入 50,392（22 片）/ 处理 60,121（25 局）**，
   3 局被 HASH 对齐门 fail-closed 丢弃（3,155 + 3,227 + 3,347 = 9,729 决策，均在 12.6k–13.4k 帧的自然终局处），
   写入分片的对齐率 **100 %（100,804/100,804 检查点相等）**，实测 **8.92 决策/s**（1 个锁步对 = 2 引擎）。

## 1. 执行策略（DAgger 的"学生侧"）选谁、为什么

| 候选 | S1N 5 局均值（新尺） | `geo_restore` | 数据量（决策） | 选它吗 |
|---|---:|---|---:|---|
| `checkpoint-v2`（27 片，k1/fast） | 1,718.8 | off | 76,912 | ✗ 比 F 低 97.2 帧 |
| **`armF_v2i27`（27 片，v2i 标签）** | **1,816.0** | off | 76,664 | **✓ 采用** |
| `armB_k1_geoon`（8 片 + geo_restore） | 1,614.4 | on | 26,648 | ✗ 数据量只有 1/3 |

**依据（三条，按权重排）**：

1. **帧数是本任务的唯一闭环目标量**，且两臂同一把新尺、同一组 5 个评测种子：1,816.0 > 1,718.8（配对 +97.2 帧，t=0.37，2/5 同向）。
   两者统计上不可区分，但 R2 的执行策略只影响**探索分布**，不需要显著性 —— 取上界更合理。
2. **只有 F 过了"≥教师 30 %"那条腿**（1,816.0 = 44.6 % vs 1,718.8 = 42.2 %），且 F 的 S1N 最大值（2,907）说明它对更长局有样本。
3. `armB`（1,614.4）虽然是**唯一带 `geo_restore` 的臂**，但它的数据量只有 26.6k —— 用它做执行策略会把 R2 的探索分布
   锚在一个 8 片学生上；R2 的目的恰恰是"在 27 片口径上补 on-policy 状态"，所以执行策略必须与目标口径同量级。

**执行策略带 `geo_restore` 吗？不。** `armF_v2i27.pt` 是不带 `--geo-restore` 训出来的（EXP-P3-005 §4 明确"不折叠第二个因子"）。
执行策略只决定"走哪条轨迹"，标签由教师给（100 % 教师软标签），所以执行策略的目标形状不影响数据；它只影响状态分布。
这一点的代价写在 §12：R2 的探索分布会带着"塌缩学生"的偏置（动作 0 占 91 %，EXP-P3-005 §4.1）。

**`--mode` 选 `fast`（`verify_k=1`），不选 `fast3`**：

- **语料一致性**：训练集前 76,912 条（D0 53,512 + R1 23,400）**全部**是 `verify_k=1` 写出的标签，`verified_mask` 每决策恰好 1 个 True。
  `geo_restore` 的定义是 `scores + 5·(1−verified_mask)` —— 若 R2 用 `fast3`，则新增 100k 行每行只有 15 个未验证候选被加 5（其余 3 个已验证），
  而旧 76,912 行是 17 个。**同一个 `geo_restore` 开关在语料的两半上会写出两种不同的目标形状**，两臂对照就不再是"只翻一个开关"。
- **信号已被判死**：EXP-P3-004 实测 `verify_k` 1→3 真偏好率 1.20 %→1.20 %（0 变化）、闭环 −147.5 帧（不显著）。
  既然它对信号无益，就没有理由为它付出语料不一致的代价。
- **吞吐**：`verify_k=3` 让教师每决策多 2 次引擎验证（`CHILD k HAZARDS`），采集更慢；本轮受 240 min 总预算约束（§2 实测 ~8.9 dec/s），
  `fast` 是能多拿数据的那个。

## 2. R2 采集（方法 + 实测）

- 采集器：`tools/rl/v400_dagger_collect.py --phase run`（EXP-P3-003 的锁步双引擎：**env 出像素、shadow 出标签**），
  `--beta 0.2`（每决策按 `rng_seed+seed` 的种子抽签，抽中则执行教师自己的 plan slots，否则执行学生贪心动作）、
  `--mode fast`、`--stage 0`（= 真 Stage 1）、`--difficulty 1`、`--team 0`、`--seed-start 3000`、
  `--decisions 3500 --max-frames 20000 --stop-on-death --trim-transition`、`--target-decisions 100000`。
- 标签：**100 % 教师软标签**（`scores[18]` 写在学生实际到达的状态上，DAgger query；不是教师自己的轨迹）。
- 引擎：`third_party/th08-rl/build/rl_hazard2/th08-modern`，sha256
  `f172399f61b75baa8e62be46041ef756a710759f439e7fb07111f4747c78264e`（`--expected-sha` 同值强校验；与 R1 同一把）。
  评测用的是 `build/rl/th08-modern`，sha256 `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb`（与 EXP-P3-002/003/004/005 同一把尺的同一引擎）。
- 并发：**1 个锁步对 = 2 个引擎**（env + shadow；教师侧的 child recorder 由 shadow 进程自己拉起），全程不超任务卡的
  "WSL 引擎并发 ≤2"。采集期机器上另有别的实验在跑（`loadavg` 实测 3–16），吞吐因此波动。
- 对齐：每决策 **2 个 HASH 检查点**（pre/post），`pre` 要求两侧帧号都等于 `4k`，`post` 要求两侧一致；
  任一处不齐 → `AlignError` → `writer.abort()` → **该分片永不落盘**。换面（stage clear）语义沿用 D0：
  只允许"最后一条 post 的那一次不一致"，且必须由 `_transition_info_from_pairs` 证明是换面，才 `--trim-transition` 掉那一条。
- **本轮新增的唯一采集器改动**：`--continue-on-error`（单局失败只记进 `progress.json`/`summary.json`，不终止整场；
  失败分片仍然 fail-closed 不写），以及 `campaign_summary()` 把"partial / 对齐率"变成**派生量**而不是调用方断言。

### 2.1 实测（rollup：`work/summary.json` + `work/progress.json` + `report.py agg`）

| 量 | 值 | 证据 |
|---|---:|---|
| 尝试局数 / 处理决策 | 25 局 / **60,121** | `logs/collect.log` 末行 + `work/summary.json` |
| **写入分片 / 决策** | **22 片 / 50,392** | `r2/manifest.json`、`report.py agg` |
| 目标 | 100,000 决策（200 局） | `--target-decisions` |
| **完成度** | **50.4 %（partial=true）** | `summary.json` 的 `target_met=false / partial=true` |
| HASH 检查点（写入分片） | **100,804 / 100,804 相等** | 逐片 `shard.json:alignment_extra.hash_checkpoints` |
| 对齐率 | **1.0000** | `summary.json:hash_alignment_rate` |
| 实测 β | **0.19954**（50,392 条逐决策记录） | 每行 `dagger.injected` 计数 |
| 教师可标注率 | **1.0000** | `dagger.teacher_available` |
| 学生/教师一致率 | 0.4220 | `dagger.agreement` |
| 吞吐 | **8.92 决策/s**（6,830 s） | `summary.json`；峰值 12.3 dec/s（R1 同机空载） |
| 观测体积 | 1.28 GB（22 片 obs） | `report.py agg` |
| 引擎 | `rl_hazard2` = `f172399f…` | `--expected-sha` 强校验 |

### 2.2 被对齐门丢弃的 3 局（fail-closed 的原始证据，不写入任何分片）

| seed | 决策 | 时间点 | 失败原因 |
|---|---:|---|---|
| 3005 | 3,155 | `decision 3154 post` frame **12,620** | `AlignError: shadow=48f9c79e… env=8e339a06…` |
| 3013 | 3,227 | `decision 3226 post` frame **12,908** | `AlignError: shadow=19501548… env=fa021ee3…` |
| 3023 | 3,347 | `decision 3346 post` frame **13,388** | `AlignError: shadow=84d7fa0c… env=bd020e3e…` |

三条都发生在 **12.6k–13.4k 帧**（stage-0 的自然终局/换面边界），**不是**随机噪声：seed 3013 的
`decision 3226 post / frame 12908` 与 **EXP-P3-003 记录的 seed 2011 完全同一位置**（同一 decision、同一帧号，hash 不同
—— 两条轨迹走到同一个引擎边界时都会分叉）。这正是 `--trim-transition` 只对"换面"放行、对其余一切 fail-closed 的理由：
这 3 局**没有**被当成换面放行，也没有一行进入训练集（`work/progress.json` 的 `failed` 数组逐条记录）。

### 2.3 与既有实现的一致性（"分片/续跑/trim"三件事）

- **分片**：D0 同构（`v400-teacher-obs-shard-1`、`frames_per_decision=4`、`label_channel=shadow`、`pixel_channel=env`），
  逐片通过 `dataset.load_teacher_samples(require_observations=True)`：写入分片 22/22、`obs_unresolved=0`、
  `obs_spec_digest` = `24f0ddac…`（与 D0/R1 同一个），`verify` 阶段 exit 0。
- **续跑**：`shard_resume_report` 跳过已完成分片，进度写在 `progress.json`；`--continue-on-error` 只让"整场"活下来，
  **不改**任何一局的落盘判定。
- **换面 trim**：语义未改（D0 的 `_transition_info_from_pairs`，只允许最后一个 post 不一致且必须由教师判定 stage_clear）。
  本轮 3 次不一致**都不是**换面（没有 `stage_changed`），所以一次 trim 也没发生 —— 记录在案。

## 3. 聚合 + 重训（两臂，只翻 `geo_restore` 一个开关）

### 3.1 语料（三个 D0 形状的 root，49 片 / 127,304 决策）

| root | 片 | 决策 | HASH 检查点 | 对齐率 | obs 摘要 |
|---|---:|---:|---:|---:|---|
| `.tmp_v400_d0/d0`（D0） | 16 | 53,512 | 107,024 | **1.0000** | `24f0ddac…` / unresolved 0 |
| `.tmp_v400_dagger/d0_extra`（R1） | 11 | 23,400 | 46,811 | **1.0000** | 同上 |
| `.tmp_v400_dagger2/r2`（**R2 本轮**） | 22 | 50,392 | 100,804 | **1.0000** | 同上 |
| **合计** | **49** | **127,304** | **254,639 全等** | **1.0000** | 无重复分片键 |

分片键跨 root 零重叠（`dual-s0d1seed1–16` / `dagger-s0d1seed2000–2010` / `dagger-s0d1seed3000+`），
所以"整片切分 = 整局切分 = 整 seed 切分"成立。

### 3.2 切分与训练（两臂同 split、同配方）

| 项 | 值 |
|---|---|
| split | **按整片 70/15/15，`split_seed=0`，签名 `29d3197…`（两臂逐字节相同）** |
| train / val / test | **34 片 / 91,384** · **8 片 / 19,604** · **7 片 / 16,316** |
| τ / 参数 / 损失 | 2.0 / **204,260** / `KL(π_t‖π_s) + BCE`（AdamW lr 2e-3, wd 1e-4, batch 256, seed 0） |
| 预算 | `--train-seconds 420`，两臂并发，`aborted_at_deadline=true`（跑满预算，不是提前停） |
| **geo_on** | `--geo-restore`：**8,598 步 / 2,201,088 样本 / wall 661.9 s** |
| **geo_off** | 无开关：**8,244 步 / 2,110,464 样本 / wall 661.9 s** |
| 门 | D0 = 53,512 ≥ 50,000 契约，**无 waiver**（`gate_waived=false`） |

**`geo_restore` 的逐值可复原性**（训练器自报，不是推断）：

| 量 | geo_off | geo_on |
|---|---:|---:|
| 参与恢复的未验证候选值 | 2,164,168 | 2,164,168 |
| `geo_restore_total_delta` | 0.0 | **10,820,840.0 = 5.0 × 2,164,168（精确）** |
| 盘上 `top1−top2` 均值 / 恰等于 5.0 | 5.04918 / 66.65 % | 同（**未改盘上标签**） |
| 训练用 `top1−top2` 均值 / 恰等于 5.0 | 5.04918 / 66.65 % | **0.04919 / 0.00 %**（几乎全平） |
| `verified_mask_mean` | 1.00 | 1.00（= `verify_k=1`，见 §1 的口径理由） |
| `scores` 范围 | −5.0 … 120.0 | 0.0 … 120.0 |

两臂的盘上标签统计**完全相同**（5.04918 / 66.65 % / 2,164,168）⇒ 数据集与 split 一致，差别只在训练目标。

## 4. 离线指标（val 19,604 / test 16,316，整 split 全量）

| 指标 | **geo_on** | **geo_off** | Δ(on−off) | 参照 armF（27 片） |
|---|---:|---:|---:|---:|
| val top-1 | 0.2428 | **0.5776** | −0.3348 | 0.5493 |
| val top-3 | 0.4241 | **0.7839** | −0.3598 | 0.7103 |
| **val macroR** | 0.0859 | **0.1363** | −0.0504 | 0.1471 |
| val KL | 0.0226 | 0.3516 | −0.3290 | 0.4129 |
| val 教师熵 | **2.8677** | 2.2398 | **+0.6279** | 2.2642 |
| val 预测动作 0 占比 | **0.2827** | 0.5827 | **−0.3000** | —（armF 的动作 0 recall 0.913，见 EXP-P3-005 §4.1） |
| val 预测类数 / focus 覆盖 | **14 / 5-of-8** | 8 / 2-of-8 | +6 / +3 | 12 / — |
| test top-1 / macroR | 0.2740 / 0.0671 | **0.5897 / 0.1187** | −0.3157 / −0.0516 | 0.5370 / 0.1426 |

**读法（三条，第 2 条是本轮实测的更正）**：

1. **机制被复现**：`geo_restore` 把目标熵从 2.2398 拉到 **2.8677**（ln18 = 2.8904），
   预测动作 0 从 0.583 压到 **0.283**、预测类数 8 → **14**、focus 覆盖 2/8 → **5/8**。
   这与 EXP-P3-004 记录的机制（2.261→2.865、动作 0 recall 0.861→0.207）**同一方向、同一量级**，而且是在 **4.8 倍数据量**上。
2. **`macroR` / top-1 的差距是"真的"（可比）**：本轮专门查了"恢复后标签的 argmax 是否还是同一个类"——
   **23,460 行里 23,454 行不变（99.974 %）**，且盘上 argmax **100 % 就是被验证的那个候选**（§4.2）。
   所以两臂比的是**同一个类的偏好**：`geo_on` 的 top-1 0.2428 / macroR 0.0859 **不是**"测了另一个量"，
   而是"目标被压平之后，学生在同一个类上再也给不出尖峰" —— 这是 `geo_restore` 的**代价本身**，不是测量噪声。
3. **离线更差 + 闭环更"活" 可以同时成立**：`geo_on` 的 top-1 只有对照臂的 42 %，但它的**动作分布**更宽
   （唯一动作 14 vs 8、focus 5/8 vs 2/8、动作 0 0.283 vs 0.583）。也就是说：**它学会了"多动"，但没学会"选对"**。

### 4.2 两个独立测量的交叉验证（`cross_eval.py` + `argcheck_multi.py`）

用同一批 ckpt，把两臂都重新在**盘上标签（未恢复）**上打分（`build_splits(geo_restore=False)`），
得到的结果与训练器自报的 val/test 指标**逐位相同**（top-1 0.2428 / 0.5776、macroR 0.0859 / 0.1363）。
原因查明了，不是"两处代码巧合"：

| 检查（9 片 / 23,460 行：R2×6 + D0×2 + R1×1） | 值 |
|---|---:|
| 恢复前后 `argmax(scores)` **相同**的行 | **23,454 / 23,460 = 99.974 %** |
| 盘上 `argmax` 就是**被验证候选**的行 | **23,460 / 23,460 = 100.000 %** |
| `top1−top2` **恰好等于**折价 5.0 的行 | 13,399 / 23,460 = 57.11 %（全语料 66.65 %） |

**机制**：`scores + 5·(1−mask)` 给 17 个未验证候选加同一个常数，**不改变它们之间的次序**；
被测候选（盘上 argmax，100 % 是被验证那个）只有在"领先第二名不到 5.0"时才会被追平，
而追平时 `argmax` 取首个最大值的下标（实测**全部**落在被验证候选上）。所以
**`geo_restore` 改的是目标的"形状"（熵、softmax 质量），不是"标签类"**——这两件事在 EXP-P3-004 里没有被分开测过。

分切片（`cross_eval.json`，盘上目标）：

| 切片（val） | geo_on top-1 / macroR | geo_off top-1 / macroR |
|---|---|---|
| D0（6,668 行） | 0.1914 / 0.0666 | **0.5361 / 0.1377** |
| R1（2,088 行） | 0.1557 / 0.0799 | **0.6159 / 0.1427** |
| **R2（10,848 行）** | **0.2912 / 0.0925** | **0.5958 / 0.1190** |

**R2（on-policy）状态上两臂的差距最小**（+0.30 vs D0 的 +0.34 top-1），且 `geo_on` 在 R2 上比在 D0 上**更好**
（0.2912 vs 0.1914）—— 说明 R2 的 on-policy 状态确实落在 `geo_on` 的分布里，DAgger 的"把标签放到学生轨迹上"这一步是有效的，
只是**它的收益不足以抵消目标被压平带来的损失**。

## 5. S1N 5 局（新尺 `consecutive`，stage 0，cap 19,887，`--stop-on-death`，1 引擎/进程 × 2 进程）

引擎 `build/rl/th08-modern`（sha256 `591c521d…`，与 EXP-P3-002/003/004/005 同一把尺的同一引擎），
所有记录的 `obs_window=consecutive`、`renders == frames`、`window_alignment.matches_training_obs=true`。

| seed | **geo_on** | **geo_off** | 参照 armF | 参照 v2 |
|---:|---:|---:|---:|---:|
| 1000 | 1,332 | 1,430 | 1,763 | 1,644 |
| 1001 | 1,445 | 2,180 | 2,907 | 1,852 |
| 1002 | 983 | **2,231** | 1,293 | 1,409 |
| 1003 | 780 | **2,260** | 1,324 | 1,885 |
| 1004 | 727 | **2,293** | 1,793 | 1,804 |
| **均值** | **1,053.4** | **2,078.8** | 1,816.0 | 1,718.8 |
| 中位数 | 983 | 2,231 | 1,763 | 1,804 |
| sd / se | 323.0 / 144.4 | 365.1 / 163.3 | 653.7 / 292.3 | — |
| 均值 score | 9,250.4 | **39,923.6** | 29,373.8 | 25,114.2 |
| 过关 / 首死停 | 0/5 · 5/5 | 0/5 · 5/5 | 0/5 · 5/5 | 0/5 |

### 5.1 配对（同一组 5 个评测种子）

| 对比 | 均值差（帧） | sd | 配对 t | 同向 |
|---|---:|---:|---:|---:|
| **geo_on − geo_off**（只翻 `geo_restore`） | **−1,025.4** | 610.9 | **−3.75** | **0/5** |
| geo_off − armF（数据量 76.9k → 127.3k） | **+262.8** | 758.3 | 0.77 | 3/5 |
| geo_off − v2（旧基线） | **+360.0** | 374.4 | **2.15** | 4/5 |
| geo_on − armF | −762.6 | 485.7 | −3.51 | 0/5 |

**读法**：数据量的 +262.8 帧**方向正确但统计上不显著**（t=0.77，3/5；±sd 758 很大）；
`geo_restore` 的 −1,025.4 帧**是显著的负效应**（5/5 反同向，t=−3.75），且比它当初的 +447.8 大一倍多。

## 6. 判据（预注册，新尺）——**FAIL**

| 判据 | 门槛 | geo_on | geo_off | 结果 |
|---|---|---:|---:|---|
| S1N 均值 | **≥ 2,500 帧** | 1,053.4（42.1 %）✗ | **2,078.8（83.2 %）** ✗ | **两臂都不过** |
| macro recall（val） | **≥ 0.25** | 0.0859 ✗ | 0.1363 ✗ | **两臂都不过** |
| ≥ 教师 30 % | **4,069.6** | 1,053.4（7.8 %）✗ | 2,078.8（15.3 %）✗ | **两臂都不过** |

**判定：FAIL（两条腿都没过）。** 三条如实标注：

1. **相对进步是真的**：`geo_off` = **2,078.8** 是新的最好成绩（旧最好 1,816.0，**+262.8 / 114.5 %**），
   也是第一次在"27 片口径 + 更多数据"上把 v2 基线拉开到 t=2.15（4/5 同向）。但它离 2,500 还差 421 帧。
2. **绝对水平仍在"不会玩"区间**：0/5 过关、5/5 首死即停、教师 30 % 门槛（4,069.6）差 1,991 帧。
3. **`geo_restore` 在本轮是负因子**（−1,025.4，t=−3.75），**与 EXP-P3-004 的 +447.8 方向相反**。这不是测量失败：
   两臂同数据、同 split、同配方、同种子，只有那一个开关不同（§3.2 的 delta 精确等于 5×未验证值数）。

## 7. 数据量-性能外推（本轮最该记住的一张表）

### 7.1 事前的双边斜率（EXP-P3-004/005 的两个 8→27 片对照）

| 对照 | Δ帧 | Δ决策 | **帧 / 1k 决策** |
|---|---:|---:|---:|
| v2 − armA | +552.2 | +50,264 | 10.99 |
| armF − armE | +606.4 | +50,264 | 12.06 |
| **均值** | — | — | **11.53** |

按这个斜率，R2 的 +50,392 决策应给 **+580.8 帧** → 预计 2,396.8；若再把 `geo_restore` 的 +447.8 **加性**叠上去 → 2,844.6。

### 7.2 实测（本轮）

| 量 | 值 |
|---|---:|
| geo_off − armF | **+262.8 帧**（+50,392 决策） |
| **实测斜率** | **5.22 帧 / 1k 决策** |
| 相对事前外推 | **45.2 %**（收益递减，不是线性） |
| 到 2,500 还需要的决策 | **80,765**（≈ **2.55 h** 采集，31,688 决策/h） |
| 到"教师 30 %"（4,069.6）还需要的决策 | **381,729**（≈ **12.0 h** 采集） |
| `geo_restore` 的加性预测（2,844.6）与实测（1,053.4） | **差 −1,791 帧**，加性假设被证伪 |

**结论**：**"数据量"这条路的边际收益在本轮已经掉到外推值的一半**，只靠加数据到 2,500 需要再来 8 万决策（2.5 h）；
要到"教师 30 %"需要 38 万决策（12 h）—— 这不是"再多采一点"的量级，必须换因子（见 §12.4）。

## 8. 诊断（动作分布 / 唯一动作 / 首死 / "会不会站着不动"）

### 8.1 R2 采集时执行策略（armF）在闭环里的行为（50,392 决策，逐行 `dagger` 记录）

| 量 | 值 |
|---|---:|
| 注入率（实测 β） | 0.19954 |
| 执行动作 0 占比 / 唯一动作数 / focus 方向覆盖 | 0.4861 / **17** / **8-of-8**（含 20 % 教师注入的动作） |
| **学生自己**（未注入时）的动作 0 占比 / 唯一动作 / focus 覆盖 | **0.5253** / 12 / **3-of-8** |
| 首 250 决策（前 1,000 帧）动作 0 占比 / focus 覆盖 | **0.4724** / **3-of-8** |
| 停止原因 | `teacher_unavailable` 20 局、`decisions` 2 局（**零** death 记录：教师恰好在死亡状态上无法标注，两者重合） |

### 8.2 两臂的闭环动作探针（2 seed × 2 臂，复用评测同一段 `run_episode`，只加动作记录）

| 量 | geo_on (seed 1000 / 1002) | geo_off (seed 1000 / 1002) |
|---|---|---|
| 帧数 | 1,332 / 983 | 1,430 / 2,231 |
| 唯一动作数 | **10 / 9** | **3 / 4** |
| 动作 0 占比 | 0.404 / 0.453 | 0.565 / 0.491 |
| idle（动作 0 或 9）占比 | 0.527 / 0.535 | 0.565 / 0.491 |
| **focus 方向覆盖** | **4-of-8 / 4-of-8** | **0-of-8 / 0-of-8** |
| 首 250 决策动作 0 占比 | 0.300 / 0.324 | **0.768 / 0.768** |
| 首 250 决策唯一动作 | 9 / 9 | **3 / 3** |
| **首段最长"站着不动"连击** | 35 / 42 决策（140 / 168 帧） | **180 / 180 决策（720 帧）** |
| 动作改变率 | 0.281 / 0.281 | 0.182 / 0.206 |
| 炸弹使用次数 | 2 / 0 | 0 / 0 |

### 8.3 首死帧分布（S1N，`--stop-on-death`，5/5 都是 `death_stop`）

| seed | 1000 | 1001 | 1002 | 1003 | 1004 | 均值 |
|---|---:|---:|---:|---:|---:|---:|
| **geo_on** | 1,332 | 1,445 | 983 | 780 | 727 | **1,053.4** |
| **geo_off** | 1,430 | 2,180 | 2,231 | 2,260 | 2,293 | **2,078.8** |

### 8.4 "学生学会 shooter 基本生存了吗？"—— **没有，而且答案分两层**

1. **活得更久的那一臂是靠"不动"活下来的**：`geo_off` 开局 **180 个连续决策（720 帧，占它 2,231 帧的三分之一）**
   完全不动，首 250 决策 76.8 % 是动作 0，整局只用 3–4 个动作，**focus 一次没用**（0/8）。
   它没有学会"躲"，它学会了"在出生点站住"。
2. **会动的那一臂立刻掉帧**：`geo_on` 的动作丰富度是另一臂的 3 倍（10 vs 3）、会用 focus（4/8）、
   开局只停 140–168 帧，但**帧数腰斩到 1,053.4**。也就是说：**目前的策略里"移动"与"存活"是反向的**
   —— 它还没有学会"朝有用的方向动"，移动只是把碰撞概率换成另一种分布。
3. **离线-闭环反向**：`geo_off` 的 val top-1 更高（0.578 vs 0.243）且动作 0 更多（0.583），闭环帧数更高；
   `geo_on` 的动作分布更接近教师（唯一类数 14、focus 5/8）却闭环更差。
   **"更像教师"不是"更能活"**——这是本轮最贵的教训。

## 9. 测试

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_dagger2\pytest1 -q tests\test_v400_dagger.py
# → 19 passed，exit 0
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_dagger2\pytest_reg2 -q `
  tests\test_v400_dagger.py tests\test_v400_student.py tests\test_v400_teacher.py `
  tests\test_v400_d0_dual.py tests\test_v400_margin_resolution.py
# → 196 passed, 2 skipped, 1 failed，exit 1（唯一的红是既有的激光字段，见下）
python -m compileall -q tools\rl\v400_dagger_collect.py tools\rl\v400_student_train_teacher.py
# → exit 0（语法门，不等于测试通过）
```

`tests/test_v400_dagger.py` 新增 5 个用例（+ 既有 14 个，共 19 passed）：

1. `test_continue_on_error_is_off_by_default_and_selectable`：新开关默认 **False**（不改变 R1 语义），
   `--mode fast3` 可选但默认仍是 `fast`。
2. `test_campaign_summary_derives_partial_from_the_target_not_from_the_caller`：`partial`/`target_met` 是**派生量**；
   对齐率在 0 检查点时为 `None`（不是空口 1.0）；`checkpoints_equal > checkpoints` 与非法 β 都抛错。
3. `test_geo_restore_is_a_value_exact_inverse_on_unverified_candidates_only`：`on − off == 5·(1−mask)`**逐值相等**、
   已验证候选**不动**（mask 不反推）、`restored_delta == 5×未验证数`、盘上标签不被改写、
   `UNVERIFIED_DISCOUNT == Weights().unverified_discount`（不硬编码）。
4. `test_two_arms_share_one_split_and_differ_only_in_the_restored_mass`：三个 root（D0/R1/R2 形状）聚合后
   **两臂 split 签名相同**、obs/bomb/`scores_raw` 逐字节相同、`on−off` 的总体差 = `5 × geo_restore_unverified_values`。
5. `test_r2_seeds_extend_the_corpus_without_colliding_with_d0_or_r1`：D0(1–16)/R1(2000–2010)/R2(3000–3059) 键与 seed 零重叠。

**唯一的红（既有的，不是本卡改动引起）**：

- `tests/test_v400_teacher.py::test_child_recording_is_one_batch_and_matches_the_main_process_tape`：
  `lethal_counts()` 多了 `lasers_damageable` 键（EXP-P1-021 的激光判定窗），测试里的字典字面量没跟上；
  EXP-P3-004 §11.7 / EXP-P3-005 §9 已记录的同一条红。

**并发观察（记录在案，不影响判据）**：本轮中途另有一个 campaign（EXP-P3-007）在同一仓库注册了 `v2j` 档，
一度让 `tests/test_v400_margin_resolution.py::test_fast_v2a_and_v2h_keep_their_published_numbers` 的
"未注册名报错"守卫生效失败（那次跑是 195 passed / 2 failed）。该 campaign 随后自行修好了同一文件，
最终态是 **8 passed**、总回归 **196 passed / 2 skipped / 1 failed**。本卡的四个改动文件没有碰过
`teacher/config.py`、`teacher/decide.py` 或 `v400_student_eval.py`。

## 10. 命令与退出码（可复现）

```bash
# WSL：R2 采集（1 个锁步对 = 2 引擎；β=0.2；fast；seeds 3000+）
BUDGET=6600 bash .tmp_v400_dagger2/launch.sh
# → exit 0；wall 6,830 s；处理 60,121 决策 / 写入 50,392（22 片）；HASH 100,804/100,804；
#   日志 .tmp_v400_dagger2/logs/collect.log，rollup .tmp_v400_dagger2/work/{progress,summary}.json
bash .tmp_v400_dagger2/smoke.sh            # 60 决策冒烟（β/标签/schema）
# → exit 0（SMOKE_RC=0 / VERIFY_RC=0，HASH 120/120）
```

```bash
# WSL：逐片加载门（frozen D0 loader）
/root/v400-venv/bin/python tools/rl/v400_dagger_collect.py --phase verify \
  --out-dir .tmp_v400_dagger2/r2 --work-dir .tmp_v400_dagger2/work
# → exit 0（obs-v1 / 24f0ddac… / unresolved 0 / alignment=decision_time_before_commit）

# WSL：两臂训练（同数据、同 split，只差 --geo-restore）
bash .tmp_v400_dagger2/run_train.sh
# → exit 0（TRAIN_EXITS armG_on=0 armG_off=0；wall 661.9 s / 661.9 s）

# WSL：S1N 5 局（新尺 consecutive；2 进程 = 2 引擎；每进程 2 线程，避免饿死引擎）
bash .tmp_v400_dagger2/launch_eval.sh
# → exit 0（EVALS DONE window=consecutive；EXIT armG_on rc=0 / armG_off rc=0）

# WSL：闭环动作探针（2 seed × 2 臂）
bash .tmp_v400_dagger2/launch_probe.sh
# → exit 0（diag/probe_geo_{on,off}.json）
```

```powershell
python .tmp_v400_dagger2\report.py agg       # → aggregate.json（49 片 / 127,304 / 对齐率 1.0）
python .tmp_v400_dagger2\report.py offline   # → offline.json（两臂离线指标 + label_stats）
python .tmp_v400_dagger2\report.py eval      # → s1n.json（逐局/均值/配对 t/判据）
python .tmp_v400_dagger2\report.py extrap    # → extrap.json（斜率 + 外推 + 需要多少数据/算力）
python .tmp_v400_dagger2\report.py diag      # → diag.json（R2 行为 + 探针 + 首死帧）
```

```bash
# WSL：两臂在"盘上标签"上的重新打分（§4.2 的交叉验证）
bash .tmp_v400_dagger2/launch_cross.sh       # → cross_eval.json
/root/v400-venv/bin/python .tmp_v400_dagger2/argcheck_multi.py
# → rows 23460 / argmax unchanged 23454 (0.999744) / argmax at verified 23460 (1.0)
```

## 11. 变更与产物

| 文件 | 变更 |
|---|---|
| `tools/rl/v400_dagger_collect.py` | **新增** `--continue-on-error`（单局失败只记录、不终止整场；失败分片**仍不落盘**）、`campaign_summary()`（`partial`/`target_met`/`hash_alignment_rate` 全部派生，空 campaign 的对齐率是 `None`）；`progress.json` 现在带 `mode`/`hash_checkpoints*`/`failed`，并额外落 `summary.json`。既有 `run/verify/plan` 语义、HASH fail-closed、trim 语义**未改** |
| `tools/rl/v400_student_train_teacher.py` | `evaluate()` **增补**动作分布诊断（`pred_histogram`、`pred_action0_share`、`pred_idle_share`、`pred_unique_classes`、`pred_focus_classes_covered`、`teacher_action0_share`、`teacher_argmax_histogram`）；`MOVES_PER_CAND` 由 `gym/actions.py` 导入并断言 `2×9==18`。**损失、优化器、split、geo_restore 数学未改** |
| `tests/test_v400_dagger.py` | **新增** 5 用例（§9） |
| `Document/Update/Alpha/v4.0.0/experiments/EXP-P3-006-dagger-round2.md` | 本卡 |
| `.tmp_v400_dagger2/*` | `r2/`（22 片 / 50,392 决策 / 1.28 GB）、`work/`（`progress.json`/`summary.json`/`verify.json`）、`arms/`（两臂 ckpt+metrics+split）、`eval/{on,off}/`、`diag/probe_geo_*.json`、`cross_eval.json`、`logs/`、`*.sh`、`report.py`、`probe_actions.py`、`cross_eval.py`、`argcheck_multi.py`、`show_*.py`、`smoke*.sh`、聚合 JSON（`aggregate/offline/s1n/extrap/diag.json`） |

**未改**：引擎、协议（`gym/protocol.py`）、依赖、第三方、`build/`（`rl`=`591c521d…`、`rl_hazard2`=`f172399f…` 只读校验）、
`teacher/{search,raster,label,decide}.py`、`tools/rl/v400_student_eval.py`。未删除任何文件，未提交。

## 12. 限制（不要过度解读）

1. **不是 G3、不是 PSN、不是 confirm split**；1 个 stage（stage 0 = 真 Stage 1）、1 个难度（normal）、n = 5 局，
   se = 144–163 帧 —— 小于 400 帧的效应在本轮**测不出来**。
2. **R2 是 partial（50.4 %）**：目标 100,000，写入 50,392。补满还差 49,608 决策 ≈ 1.57 h 采集；
   按实测 5.22 帧/1k，那 4.96 万决策**预计再给 +259 帧**（→ ~2,338），**仍不过 2,500**。
3. **`geo_restore` 的结论是"在 127k 决策上反转"，不是"它永远有害"**。它当初的 +447.8 测于 26.6k 决策、8 片、纯 D0 标签；
   本轮是 127k 决策、49 片、含 50k **on-policy** DAgger 状态。两者不同，本轮**没有**在 8 片口径上重跑它做对照。
   可检验的机制假说（未验证）：目标越平，数据越多模型越能拟合"平"，于是动作越接近均匀 → 移动更随机 → 更快撞弹。
4. **下一步该换的因子**（按本轮证据排序）：(a) **动作持续时间集** `durations` {1,2}→{1,2,3,4}（EXP-P3-005 §8 的 P0，
   仍未做）；(b) 把"存活"直接写进目标（用死亡前 N 帧的代价项），因为本轮证明"更像教师"与"更能活"是两条腿；
   (c) 教师侧把 12.6k–13.4k 帧的对齐分叉（§2.2）诊断清楚 —— 它每 ~1.5 h 会吃掉一整局的数据。
5. **窗口尺只有一把**：本卡所有 S1N 数字都是 `--obs-window consecutive`；1,718.8 / 1,816.0 也是同一把尺
   （EXP-P3-004 §6 / EXP-P3-005 §5）。历史 `strided` 数字（2,399.6 等）**不与本卡比较**。
6. **吞吐受同机负载影响**：采集期机器上有 2–3 个别的 campaign（`loadavg` 3–16、`th08-modern` 峰值 15 个），
   8.92 决策/s 不是空载上限（R1 空载 12.3 dec/s）。评测第一波因**每进程 torch 占 5 核**把引擎饿死（首局 >3 min），
   已改为每进程 2 线程后 5 局 3 min 跑完 —— 这条是复现他人实验时要注意的坑。
7. **"首死帧"= `--stop-on-death` 的帧数**，不是"能活多久"；且 `geo_off` 的 720 帧静止说明这个量**奖励不动**
   （站桩策略在出生点附近不会被弹幕扫到）。任何用 S1N 帧数当唯一目标的判据都会把"站桩"判成进步。
8. **评测并发**：本轮评测是 2 进程（2 引擎）并行、每进程 2 线程；记录里 `obs_window=consecutive`、
   `renders == frames` 全部通过（`s1n.json` 的 `window_alignment_ok=true`），**没有**混用旧 `strided` 尺。

> **更正注记（2026-11-10，EXP-P3-009）**：本卡的 `geo_restore` 在 127k 决策语料上 `−1,025` 与 EXP-P3-004 的 `+447.8` **符号相反**，两者都是单 checkpoint × 5 局；按新判据两条均 无法判定，`+463.1` 主效应同此。原始数字不改。详见 [EXP-P3-009](<EXP-P3-009-eval-power.md>)。
