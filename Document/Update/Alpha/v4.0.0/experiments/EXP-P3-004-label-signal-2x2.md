# EXP-P3-004 标签信号 2×2：`verify_k` × 几何分复原（D0 匹配 8 片 + fast3 重采 8 片）

- 日期 / 切片：2026-11-09 / **P3-WP4-a**（EXP-P3-003 §8.4 的 P0 + P0' 对照：4 臂同配方、同 seed、同 S1N 尺）
- 依赖：[EXP-P3-003](<EXP-P3-003-dagger-round1.md>) §8.3（`verify_k=1` 的 −5 折扣让 argmax 等于"哪个动作被抽到验证"）、
  §8.4（下一轮最小改动 = P0 `verify_k` 1→3 + P0' 几何分复原的 2×2）、
  [EXP-P3-002](<EXP-P3-002-bc-v1-teacher.md>)（BC v1 = 1,084.0 帧）、
  [EXP-P1-018](<EXP-P1-018-D0-scaleup.md>)（D0 16 片 / 53,512 决策）

## 0. 结论先说（三句话）

1. **`verify_k=3` 确实消灭了"折扣伪影"，但并没有给教师制造出偏好**：标签里 `top1−top2` 恰为 5.0 的比例
   **82.16 % → 0 %**、`verified_mask[第 2 名]` **0 % → 100 %**，可是**复原后的真实几何分差分布一模一样**
   （p90 0.0589 vs 0.0589，`>0.1` 的比例 1.20 % vs 1.20 %）。也就是说，被噪声淹没的是**几何本身**
   （82 % 的决策里 18 个动作的分差 < 0.06），不是只有那个 −5。
2. **闭环里唯一显著、可归因的因子是 `geo_restore`（方向与预注册预期相反）**：`B−A = +447.8 帧`
   （5/5 个种子同向，配对 t = 24.9）、`D−C = +478.4 帧`（4/5，t = 2.37），主效应 **+463.1 帧**；
   但它在**离线指标上是大负向**（val top-1 0.5656 → 0.1887、macroR 0.2514 → 0.1088）。
   `verify_k` 的主效应是 **−147.5 帧**（3 比 1 更差），即 **`verify_k=3` 不提供闭环信号**。
3. **没有任何一臂过预注册判据**：均值最高 1,614.4（B）< 2,399.6，`macroR ≥0.25` 只有 A 满足 val（0.2514）
   而 test 差 0.0013。而且 **2,399.6 这把尺本身被 §6 的窗口修正改了**：同一个 `checkpoint-v2`
   在旧（strided）尺上**逐种子复现 2,399.6**，在新（consecutive）尺上只有 **1,718.8（−28.4 %）**。
   **本卡不是 G3 通过、不是 PSN、不是 confirm split。**

## 1. 设计与预注册

| 项 | 值 | 说明 |
|---|---|---|
| 因子 1 | `verify_k ∈ {1, 3}` | 1 = `FAST`（现状）；3 = 新增 `FAST3`（`FAST` 逐字段复制，只改 `verify_k`） |
| 因子 2 | `geo_restore ∈ {off, on}` | on = 训练标签用 `scores + 5·(1−verified_mask)`，在训练器里做（`--geo-restore`） |
| 数据集 | **成对**：k=1 用 D0 的 seeds 1–8 八片；k=3 用同 seeds 1–8 的新采八片 | 两套 root 都是 **26,648 决策**，且**逐决策动作序列完全相同**（§2.2） |
| 切分 | `split_seed=0`，整局 70/15/15 → 6/1/1 片；两套 root 分片键相同 ⇒ val=seed8、test=seed7 逐臂一致 | §3.2 |
| 配方 | width 48 / hidden 256（**204,260 参数**）、τ=2.0、KL(π_t‖π_s)+BCE、AdamW lr 2e-3、batch 256、seed 0 | 4 臂完全相同 |
| 预算 | `--sweep-seconds 20 --train-seconds 420`（≤10 min/臂） | 实测每臂 wall **446.4–448.9 s** |
| 评测 | seeds 1000–1004，stage 0，cap 19,887，`--stop-on-death`，1 引擎（并发 ≤2），`--obs-window consecutive` | 同 EXP-P3-002/003 的尺，**除 §6 的窗口修正** |
| 预注册判据 | 胜出臂均值 **≥2,399.6 帧**（DAgger-1 基线）**且** macroR **≥0.25**；同时报是否 ≥ 教师 30 %（4,070） | §5.1 |
| 未预注册但记录 | `checkpoint-v2`（k=1 / geo off / **全量 76,912**）在同一条尺上；旧尺复现 | §5.2 / §6 |

**与任务卡的一处偏离（必须写明）**：任务卡写"`verify_k=1` 维直接用现成 D0（16 片）+ DAgger（11 片）快照"。
本轮**没有**这么做：新采的 k=3 只有 8 片（26,648 决策），若 k=1 臂用 27 片（76,912），`verify_k`
就会与"数据量 3 倍"混在一起，2×2 失去意义。因此 k=1 臂改用 **D0 的 seeds 1–8 八片**
（`.tmp_v400_p3sig/k1`，NTFS junction 指向冻结的 D0 分片，manifest 只保留 8 行、totals 重算），与 k=3 臂**成对**。
全量 76,912 的 k=1 行另用同一条尺复核（`checkpoint-v2`，§5.2），作为绝对门参照。

## 2. 数据

### 2.1 k=3 重采（`.tmp_v400_p3sig/k3/`，8 片，26,648 决策）

- 档：`config.py` 新增 `FAST3 = TeacherConfig(mode="fast3", horizon=32, beam=24, verify_k=3, max_iterations=1)`；
  **`FAST` / `V2A` 逐字段未改**（`tests/test_v400_labels_signal.py::test_fast3_is_fast_with_three_verified_plans_and_nothing_else`
  与 `test_fast_and_v2a_keep_their_meaning` 钉住）。不用现成的 `v2a`：它同时改 `horizon` 48、`beam` 64、
  `Weights`（底部带）与 bomb 窗口，一次动四个量，无法回答"`verify_k` 是否提供信号"。
- 采集：`tools/rl/v400_d0_dual.py --phase all --mode fast3`（channel A shadow 出标签、channel B env 出像素、
  HASH 逐检查点对齐），stage 0 / normal / seeds 1–8 / `--max-frames 20000 --stop-on-death --trim-transition`，
  2 引擎并发，wall **2,899.8 s（48.3 min）**。

| shard | seed | decisions | frames | stop_reason | HASH 检查点 | 对齐 | load |
|---|---:|---:|---:|---|---:|---:|---:|
| dual-s0d1seed1 | 1 | 3,370 | 13,480 | stage_changed | 6,740 | 6,740/6,740 | 3,370/3,370 |
| dual-s0d1seed2 | 2 | 3,322 | 13,288 | stage_changed | 6,644 | 6,644/6,644 | 3,322/3,322 |
| dual-s0d1seed3 | 3 | 3,346 | 13,384 | stage_changed | 6,692 | 6,692/6,692 | 3,346/3,346 |
| dual-s0d1seed4 | 4 | 3,274 | 13,096 | stage_changed | 6,548 | 6,548/6,548 | 3,274/3,274 |
| dual-s0d1seed5 | 5 | 3,394 | 13,576 | stage_changed | 6,788 | 6,788/6,788 | 3,394/3,394 |
| dual-s0d1seed6 | 6 | 3,274 | 13,096 | stage_changed | 6,548 | 6,548/6,548 | 3,274/3,274 |
| dual-s0d1seed7 | 7 | 3,346 | 13,384 | stage_changed | 6,692 | 6,692/6,692 | 3,346/3,346 |
| dual-s0d1seed8 | 8 | 3,322 | 13,288 | stage_changed | 6,644 | 6,644/6,644 | 3,322/3,322 |
| **合计** | 1–8 | **26,648** | **106,592** | 8× stage_changed | **53,296** | **53,296/53,296** | **26,648/26,648** |

（每片 3,274–3,394 决策，满足"每片 ≤3,500"；`pixels/play64+hud`、`obs_spec_digest 24f0ddac…`、25,344 B/决策；
`render_zero_frames=0`。`progress_k3.json`：`decisions=26,648 / shards_done=8 / shards_failed=0 /
hash_checkpoints=53,296 全等 / obs_bytes=675,375,264 / elapsed 2,899.8 s`。里面的 `complete=false,
partial=true` 只说明 `--target 30000` 这个**循环上界**没被填满（8 片自然结束在 26,648），
不是失败、也不是截断：`seeds_planned` 的 8 片全部完成、0 片失败。
每片 wall **580.4–791.0 s**，`fast3` 的 channel A 比 `fast` 慢约 **21 ms/决策**
（seed1：502.0 s / 3,371 决策 vs D0 `fast` 的 431.5 s）——EXP-P3-003 §8.4 估的是 +9 ms，
实测更大，因为 3 个 fork 都要跑完整 hazard 录制。）

### 2.2 两套 root 是**逐决策同一轨迹**

`compare_trajectories.py` 直接比对两套 root 每片的 `chosen_plan.first_action` 序列：

| seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| k=1 决策 | 3,370 | 3,322 | 3,346 | 3,274 | 3,394 | 3,274 | 3,346 | 3,322 |
| k=3 决策 | 3,370 | 3,322 | 3,346 | 3,274 | 3,394 | 3,274 | 3,346 | 3,322 |
| 相同动作前缀 | 3,370 | 3,322 | 3,346 | 3,274 | 3,394 | 3,274 | 3,346 | 3,322 |

**8/8 片完全相同**：`verify_k` 没有改变教师提交的动作，所以两套数据是**同一批状态、同一批动作**，
唯一的差别就是标签结构（谁被 verified、谁被 −5）。这是能拿到的最强配对控制。

### 2.3 标签结构：k=1 vs k=3（`analyze_labels.py`，全量 26,648 行 × 2）

| 指标 | k=1（D0 seeds 1–8） | k=3（fast3 seeds 1–8） |
|---|---:|---:|
| `verified_count` min / max / mean | 1 / 1 / 1.00 | **3 / 3 / 3.00** |
| `verified_mask[argmax]` | 100.00 % | 100.00 % |
| `verified_mask[第 2 名]` | **0.00 %** | **100.00 %** |
| `top1−top2` 恰为 5.0（存盘） | **82.16 %** | **0.00 %** |
| 存盘 `top1−top2` 均值 | 5.0145 | **0.0145** |
| 复原后几何 `top1−top2` 均值 / p90 / max | 0.0145 / **0.0589** / 5.573 | 0.0145 / **0.0589** / 5.573 |
| 复原后 `>0.1` / `>1.0` / `>5.0` | **1.20 %** / 0.10 % / 0.004 % | **1.20 %** / 0.10 % / 0.004 % |
| 教师 target 熵（τ=2，存盘） | 2.2687 | 2.1240 |
| argmax 直方图（18 类） | 见下 | **与 k=1 逐类相同** |
| bomb 正例 | 0 | 0 |

argmax 直方图（两套完全相同）：
`[11659, 616, 2279, 2646, 548, 335, 1799, 1877, 285, 0, 73, 1707, 1934, 194, 53, 309, 224, 110]`。

**读法（本轮最重要的诊断）**：

1. `verify_k=3` 让标签的 **top-1/top-2 比较第一次发生在两个 verified 动作之间**（第 2 名 verified 率 0 % → 100 %），
   折扣伪影彻底消失（82.16 % → 0 %）；
2. 但**几何分差没有变**：复原后 p90 都是 0.0589，`>0.1` 都是 1.20 %，argmax 直方图逐类相同
   ——`verify_k` **没有**提高"教师有真实偏好"的状态占比（这是任务卡的预期，被数据否定）；
3. 发生的变化是**标签的形状**：k=1 是"1 个动作拿走 42 % 概率、其余各 3 %"（单峰伪影），
   k=3 是"3 个动作各约 24 %、其余各 2 %"（**三峰平局**）。两者都不是"有信息的目标"，
   但两者的**塌缩方式不同**，这才是 §4/§5 里 4 臂行为差异的来源。

### 2.4 k=1 匹配子集（`.tmp_v400_p3sig/k1/`，8 片，26,648 决策）

`make_k1_subset.py`：D0 的 seeds 1–8 以 **NTFS junction** 挂进新 root（不复制字节、不改 D0），
manifest 只保留 8 行、totals 重算，`derived.source_manifest_sha256` 记源 manifest
（`964d3a78…`）。决策数与 §2.1 **完全相等**。

## 3. 训练 4 臂

### 3.1 命令（4 臂只差 `--d0-root` 与 `--geo-restore`）

见 §9。每臂 `--taus 2.0`（预注册值，不再扫 τ）、`--train-seconds 420`、`--eval-every 100`。

| 臂 | root | `--geo-restore` | τ | 参数 | step | 样本 | wall |
|---|---|---|---|---:|---:|---:|---:|
| A | `.tmp_v400_p3sig/k1` | off | 2.0 | 204,260 | 12,183 | 3,118,848 | 448.9 s |
| B | `.tmp_v400_p3sig/k1` | **on** | 2.0 | 204,260 | 11,995 | 3,070,720 | 448.9 s |
| C | `.tmp_v400_p3sig/k3` | off | 2.0 | 204,260 | 12,498 | 3,199,488 | 446.4 s |
| D | `.tmp_v400_p3sig/k3` | **on** | 2.0 | 204,260 | 12,566 | 3,216,896 | 446.9 s |

### 3.2 切分（两套 root 分片键相同 → 逐臂配对）

| split | 片数 | 种子 | 决策数 |
|---|---:|---|---:|
| train | 6 | 1,2,3,4,5,6 | 19,980 |
| val | 1 | **8** | 3,322 |
| test | 1 | **7** | 3,346 |

4 臂的 `split.json` 逐字节同一个映射（`split_seed=0` + 相同分片键）；`check_episode_isolation`
无跨 split 种子。`--accept-partial-d0` 因 26,648 < 50,000 契约而带 `gate_waived=true`（**设计使然**，见 §1 偏离说明）。

## 4. 离线指标（整 split 全量，不是 2,048 子集）

| 臂 | `verify_k` | `geo_restore` | val top-1 | val macroR | val KL | val bomb acc | test top-1 | test macroR | test KL | test bomb acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** | 1 | off | **0.5656** | **0.2514** | 0.4176 | 1.0000 | **0.5696** | **0.2487** | 0.3957 | 1.0000 |
| **B** | 1 | **on** | 0.1887 | 0.1088 | 0.0307 | 1.0000 | 0.2295 | 0.1516 | 0.0234 | 1.0000 |
| **C** | **3** | off | 0.4530 | 0.1105 | 0.5589 | 1.0000 | 0.4743 | 0.1097 | 0.5474 | 1.0000 |
| **D** | **3** | **on** | 0.3477 | 0.0658 | 0.0274 | 1.0000 | 0.3637 | 0.0714 | 0.0216 | 1.0000 |
| 参照 | 1 | off | 0.5912 | 0.1912 | 0.3473 | 0.9984 | 0.5766 | 0.1887 | 0.3542 | 0.9974 |

（参照行 = `checkpoint-v2`，k=1 / geo off / **全量 76,912 行**，τ=2.0、27,748 step。它的 val/test 是
**不同 split**（27 片 70/15/15 → 12,294 / 12,054），所以不是逐值可比，只作绝对锚点。）

补充诊断（同一 `evaluate()`）：预测类数 A 16 / B 18 / C 10 / D 12；动作 0 的逐类 recall
A **0.861** / B 0.207 / C **0.937** / D 0.804（教师动作 0 占比 43.98 %，多数类基线 0.4094）。
**A 与 C 的"只会不动"比 v1/v2 更严重**（v2 是 61.5 % 预测动作 0）。

### 4.1 两套 label 的分差分布（训练器自报，证明 `--geo-restore` 只动了该动的值）

| 臂 | 盘上 `eq5` / `gt5` | 训练用 `eq5` / `gt5` | 实际复原的格数 | ΣΔ | 教师目标熵（val） |
|---|---:|---:|---:|---:|---:|
| A | 0.8260 / 0.1741 | 0.8260 / 0.1741 | 0 | 0 | 2.2608 |
| B | 0.8260 / 0.1741 | **0.0000 / 0.0000** | **453,016** | **2,265,080.0** | 2.8649 |
| C | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0 | 0 | 2.1187 |
| D | 0.0000 / 0.0000 | 0.0000 / 0.0000 | **399,720** | **1,998,600.0** | 2.8649 |

`453,016 × 5.0 = 2,265,080.0` 与 `399,720 × 5.0 = 1,998,600.0` **逐值精确**
（`453,016 = 26,648 × 17`，`399,720 = 26,648 × 15`：k=1 每行 17 个未验证，k=3 每行 15 个）。
"实际复原的格数" = `ΣΔ / 5`：A/C 的 `--geo-restore` 关闭，所以是 0（它们 `metrics.json` 里
`d0.geo_restore_unverified_values` 记的是**数据中未验证的格数** 453,016 / 399,720 —— 那正是"若打开会复原多少"，
而 `d0.geo_restore_total_delta` 才是真正加回去的总量：A/C 为 0.0）。

### 4.2 主效应（4 臂均值）

| 因子 | 水平 | val top-1 | val macroR | val KL | S1N 帧均值 |
|---|---|---:|---:|---:|---:|
| `verify_k` | 1（A+B） | 0.3772 | 0.1801 | 0.2242 | **1,390.5** |
| | 3（C+D） | 0.4004 | 0.0882 | 0.2932 | **1,243.0** |
| `geo_restore` | off（A+C） | **0.5093** | **0.1810** | 0.4883 | **1,085.2** |
| | on（B+D） | **0.2682** | **0.0873** | **0.0291** | **1,548.3** |

**离线与闭环方向完全相反**：`geo_restore` 让 top-1 掉 0.24、macroR 掉 0.09（并让 KL 假性变好 0.46 →
0.03，因为目标变平），却让闭环多活 **+463 帧**。`verify_k` 在两边都没有正效应。

## 5. S1N 5 局（stage 0 = 真 Stage 1，normal，cap 19,887，`--stop-on-death`，aligned 窗口）

### 5.1 4 臂逐局（引擎 `build/rl/th08-modern` sha256 `591c521d…`，seeds 1000–1004）

| seed | **A** (k1,off) | **B** (k1,on) | **C** (k3,off) | **D** (k3,on) |
|---:|---:|---:|---:|---:|
| 1000 | 1,115 | **1,616** | 739 | 1,615 |
| 1001 | 1,433 | **1,823** | 1,084 | 1,305 |
| 1002 | 1,049 | 1,486 | 758 | **1,487** |
| 1003 | 1,044 | 1,494 | 817 | **1,574** |
| 1004 | 1,192 | **1,653** | 1,621 | 1,430 |
| **均值** | **1,166.6** | **1,614.4** | **1,003.8** | **1,482.2** |
| 中位数 | 1,115 | 1,616 | 817 | 1,487 |
| sd / se | 160.6 / 71.8 | 137.8 / 61.6 | 371.8 / 166.3 | 122.7 / 54.9 |
| 均值 score | 8,334.8 | 8,997.6 | 7,692.8 | **13,427.0** |
| 过关 | 0/5 | 0/5 | 0/5 | 0/5 |
| 首死停 | 5/5 | 5/5 | 5/5 | 5/5 |

### 5.2 参照：`checkpoint-v2`（全量 76,912）在同一条尺上

| 行 | 均值帧 | 中位数 | 均值 score | 过关 | 尺 |
|---|---:|---:|---:|---|---|
| `v2`（consecutive，本轮） | **1,718.8** | 1,804 | 25,114.2 | 0/5 | 与 4 臂同尺 |
| `v2_strided`（strided，复现） | **2,399.6** | 1,866 | 49,262.2 | 0/5 | EXP-P3-003 §6 那把尺 |

`v2_strided` 的逐种子 = **3,636 / 1,866 / 1,790 / 1,580 / 3,126**、均值 2,399.6、score 49,262.2
——与 EXP-P3-003 §6 表**逐值相同**：`strided` 模式是对旧尺的忠实复现（同时也证明评测确定性）。

### 5.3 配对对比（同一组 5 个评测种子，配对 t）

| 对比 | 均值差（帧） | sd | 配对 t | 同向种子 |
|---|---:|---:|---:|---:|
| **B−A**：`geo_restore` 在 `verify_k=1` 上 | **+447.8** | 40.2 | **24.90** | **5/5** |
| **D−C**：`geo_restore` 在 `verify_k=3` 上 | **+478.4** | 450.8 | 2.37 | 4/5 |
| C−A：`verify_k=3` 在 geo off 上 | −162.8 | 335.7 | −1.08 | 1/5 |
| D−B：`verify_k=3` 在 geo on 上 | −132.2 | 243.4 | −1.21 | 2/5 |
| B−C（最好 vs 最差臂） | +610.6 | 331.8 | 4.11 | 5/5 |
| A−D | −315.6 | 272.8 | −2.59 | 1/5 |
| `v2 − A`（数据量 27 片 vs 8 片） | +552.2 | 188.5 | 6.55 | 5/5 |
| `v2 − B`（数据量 vs 最好 2x2 臂） | +104.4 | 179.4 | 1.30 | 4/5 |

### 5.4 是否过预注册判据：**4 臂全部 FAIL**

| 判据 | 门槛 | A | B | C | D | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 均值帧 ≥ 2,399.6 | 2,399.6 | 1,166.6 | 1,614.4 | 1,003.8 | 1,482.2 | **全 FAIL**（最高 67.3 %） |
| macroR ≥ 0.25 | 0.25 | **0.2514**（test 0.2487） | 0.1088 | 0.1105 | 0.0658 | **仅 A 的 val 过，test 差 0.0013** |
| ≥ 教师 30 %（帧） | 4,070 | 28.7 % | 39.7 % | 24.7 % | 36.4 % | **全 FAIL** |
| 与 `v2` 同尺比 | 1,718.8 | 67.9 % | 93.9 % | 58.4 % | 86.2 % | 最好 B = 93.9 % |

**判定：按预注册规则两条腿都没过**（唯一接近的是 A 的 macroR，且它的帧数是四臂最低）。
三条如实标注：

1. **2,399.6 是旧尺的数字**：同一个 `checkpoint-v2` 在 corrected 尺上是 **1,718.8**（§6）。用旧尺门槛去卡新尺的臂
   是拿两把不同的尺比——本卡把两个数都列出来，不做换算；
2. **8 片 vs 27 片**：`v2 − A = +552.2 帧`（5/5，t = 6.55）说明数据量本身值 550 帧左右；
   4 臂用的是成对的 8 片，所以它们的绝对高度**结构性低于** v2，这与因子无关；
3. **n = 5**：臂间 se 55–166 帧，`B−A` 是同向 5/5 的强效应（t = 24.9），而 `D−C`（t = 2.37）、
   `C−A`（t = −1.08）都在噪声边缘；不构成策略推断。

## 6. 训练/评测观测窗口错配的修正（零成本臂）

### 6.1 修的是哪一边、为什么

**改评测侧（`tools/rl/v400_student_eval.py`），不改训练侧。** 理由：

1. 训练数据是**冻结的证据**：D0/DAgger 的 `obs/play64.npy` 每决策只存 4 帧（`frames[4k-4..4k-1]`，连续），
   `shard.json` 的 `frames_per_decision=4`、`observation_time=decision_time_before_commit` 都是 schema 的一部分。
   要让训练变成"隔帧 4 帧"，必须**重采** 76,912 × 25,344 B ≈ **1.95 GB** 的像素并重跑教师标签——
   不是零成本，而且会同时改掉数据，污染本轮唯一的对照。
2. 评测侧只多一次像素读回：实测 `pixel_ms_per_frame = 1.283`（D0 channel B），每决策多 1 次 ≈ +1.3 ms，
   整局（≈1,200–2,400 决策）+2–3 s（实测每局 14–46 s，无感）。
3. 修正后闭环输入 = 训练输入（`frames[t-3..t]` 连续 4 帧），这是"学生真实上限"的前提。

### 6.2 改法

`run_episode(obs_window=...)` + `--obs-window {consecutive,strided}` + 纯函数 `decision_steps()`：

- **`consecutive`（新默认）**：一决策仍花 `DECISION_FRAMES=2` 帧，但**逐帧 STEP（want=1）+ 逐帧读回/入栈**，
  于是 `ObservationBuffer` 的最后 4 个 push 是**连续 4 帧**（跨度 4 帧）；
- **`strided`（旧行为，保留）**：一决策 `STEP 2` 帧 + **1** 次读回，4 个 push 相隔 2 帧、跨度 **7 帧**
  ——EXP-P3-002 的 1,084.0 与 EXP-P3-003 的 2,399.6 就是这把尺量出来的；
- 两者**每决策的帧预算相同**（`units × frames == DECISION_FRAMES`，单测钉住），所以帧数可比；
  记录里的 `window_alignment` 写明 `renders_per_decision` / `stack_span_frames` / `matches_training_obs`，
  `validate_episode_record` 拒绝自相矛盾的行。判别证据在 `renders` 字段：consecutive 的 `renders ≈ frames`、
  strided 的 `renders ≈ frames/2`（如 `v2_strided` seed1000：3,636 帧 / 1,818 renders）。

### 6.3 效果：**旧尺系统性高估 26–28 %**

| checkpoint | strided（旧尺） | consecutive（修正后） | 差 | 相对 |
|---|---:|---:|---:|---:|
| **A**（8 片，k1/off） | **1,568.0** | **1,166.6** | **−401.4** | **−25.6 %** |
| **v2**（27 片，k1/off） | **2,399.6** | **1,718.8** | **−680.8** | **−28.4 %** |

配对：`armA_strided − armA = +401.4`（5/5 同向，t = 4.59）；`v2_strided − v2 = +680.8`（4/5）。
**结论**：EXP-P3-002/003 的 S1N 排行榜（1,084.0 / 2,399.6 / 以及所有更早的 v0 行）都是在观测错配下测的，
按本节的两种 checkpoint 校准，**修正后应整体下修约 26–28 %**；历史数字仍然自洽（同尺可比），
但**不能与修正后的数字直接比较**。这是本轮的一个独立于 2×2 的发现。

## 7. 结论：哪个因子真正提供了信号

### 7.1 三层的答案（不合并成一个数）

| 层 | `verify_k` 1→3 | `geo_restore` off→on |
|---|---|---|
| **标签** | 消灭折扣伪影（eq5 82.16 % → 0 %、第 2 名 verified 0 % → 100 %），但**真实偏好占比不变（1.20 % → 1.20 %）**；形状从"单峰伪影"变"三峰平局" | 不改 mask、不改几何；把 17/15 个 −5 加回，目标从"尖峰 + 地板"变成**近均匀**（熵 2.26 → 2.86） |
| **离线**（val top-1 / macroR / KL） | 0.377 → 0.400 / 0.180 → 0.088 / 0.224 → 0.293（macroR **降**，KL 变差） | 0.509 → 0.268 / 0.181 → 0.087 / 0.488 → 0.029（**大幅变差**，KL 假性变好） |
| **闭环**（S1N 帧） | 1,390.5 → 1,243.0（**−147.5**，配对 C−A 与 D−B 都不显著为负） | 1,085.2 → 1,548.3（**+463.1**，B−A 5/5、t = 24.9） |

**结论**：真正改变闭环行为的是 **`geo_restore`（把目标抹平）**，不是 `verify_k`。
机制是**目标熵 → 模式塌缩**，4 臂严格单调：

| 臂 | 教师目标熵（val, τ=2） | 动作 0 的逐类 recall | 闭环帧 |
|---|---:|---:|---:|
| C（三峰平局，H 最低） | 2.1187 | **0.937** | **1,003.8** |
| A（单峰伪影） | 2.2608 | 0.861 | 1,166.6 |
| D（近均匀 + 真梯度） | 2.8649 | 0.804 | 1,482.2 |
| B（近均匀） | 2.8649 | **0.207** | **1,614.4** |

目标越尖，学生越退化成"押动作 0"（C 的动作 0 recall 0.937 ≈ 完全塌缩），闭环越短；目标越平，
学生越接近"准随机"，闭环越长。**这解释了 EXP-P3-003 §8.1 的"离线降、闭环升"**，而且本轮把它变成了
单因子因果（`B−A` 只差 `--geo-restore`，5/5 同向）。

### 7.2 对任务卡两个预期的裁定

1. **"`verify_k=3` 提高'教师有真实偏好'的状态占比" → 被数据否定**：真实几何分差 `>0.1` 的比例
   **1.20 % → 1.20 %**，p90 **0.0589 → 0.0589**。根因不是"1 个动作被验证"，而是**几何本身在 82 % 的决策里没有分辨度**
   （18 个动作的分差 < 0.06，比 5.0 小 85 倍）。`verify_k=3` 只是把"哪个动作被抽到"这个伪影
   从 1 峰变 3 峰——**噪声换了个形状，没有变成信号**。
2. **"`geo_restore` 单独做可能反而更差" → 离线正确、闭环相反**：离线 top-1 −0.377、macroR −0.143（确实更差），
   但闭环 **+447.8 帧（5/5）**。它不是"提供了标签信号"，而是**移除了模式塌缩的驱动力**：
   目标变平后学生不再死守动作 0。**它有用，但不是因为它恢复了信号**。

### 7.3 下一步（按本轮的证据重排，不推荐立刻做）

| 优先级 | 改动 | 依据 |
|---|---|---|
| **P0** | **提高几何分辨度**：`verify_k` / `geo_restore` 都不是杠杆；需要让 18 个候选的真实分差拉开（更大的 `horizon`、`beam`、或给"近邻动作"做 tie-break 的显式项），或改目标形式（top-k 软标签 / 只在 `gap>0.1` 的 1.2 % 决策上学） | §2.3：p90 = 0.0589 与 5.0 无关，换 `verify_k` 不动它 |
| **P1** | **用 `geo_restore` 当正则**（不改标签语义，只在 loss 侧对"预测分布过度尖"加惩罚），而不是整份标签复原 | §7.1：`geo_restore` 的正效应来自"别塌缩"，不是来自"更准" |
| P2 | 修 §4.3 的 `ERR player-speed-unavailable`（教师可标注性边界） | EXP-P3-003 §4.3 |
| P3 | 类平衡 KL / obs-v2 | EXP-P3-003 §8.2；但要在 P0 之后 |

**不推荐**：再做 `verify_k` 或 `geo_restore` 的其它组合（4 个格子已经填满，两个因子的主效应都不支持）。

## 8. 测试

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m compileall -q touhou_rl tools\rl\v400_student_train_teacher.py tools\rl\v400_student_eval.py tools\rl\v400_d0_dual.py
# → exit 0
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p3sig/pytest -q tests/test_v400_labels_signal.py
# → 16 passed in 2.11s，exit 0
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p3sig/pytest_reg -q `
  tests/test_v400_student.py tests/test_v400_d0_dual.py tests/test_v400_labels_signal.py
# → 72 passed in 2.49s，exit 0
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p3sig/pytest_final -q `
  tests/test_v400_labels_signal.py tests/test_v400_student.py tests/test_v400_d0_dual.py tests/test_v400_teacher.py
# → 180 passed, 2 skipped, 1 failed in 4.21s，exit 1（唯一红是既有 v2h 激光字段，见 §11.1）
```

`tests/test_v400_labels_signal.py` 的 16 个用例：

1. `test_fast3_is_fast_with_three_verified_plans_and_nothing_else`：`fast3` 与 `fast` 除 `verify_k` 外
   **逐字段相等**（horizon/beam/max_iterations/durations/weights/bomb 窗口/clip_y/shoot 脉冲），
   `replan_decisions × decision_frames == 4`，注册在 `CONFIGS` 且不进 `PROFILES`（评分候选表）。
2. `test_fast_and_v2a_keep_their_meaning`：`fast=(32,24,1,1)`、`v2a=(48,64,3,3)`、底部带 1.5/2.0 不变，
   所有档 `unverified_discount == 5.0`。
3. `test_the_teacher_cli_and_the_d0_pipeline_accept_the_new_profile`：`v400_teacher --mode fast3` 可用、
   默认仍 `full`；`v400_d0_dual --mode fast3` 可用（选择集 = 教师注册表）、拼错即 `SystemExit`。
4. **`test_geo_restore_on_the_real_d0_first_row`**：冻结 D0 首行的**逐值**复原——index 7 是 verified / 117.125，
   复原是**恒等**；index 0 从 111.791667 精确变 116.791667；`chosen_plan.geo_score == scores[argmax]`。
5. `test_geo_restore_only_touches_unverified_values`：`restored[mask] == scores[mask]`（逐位相等）、
   `restored[~mask] == scores[~mask] + 5`，且 `restored − 5·(~mask) == scores`（回到盘上原值）。
6. `test_geo_restore_rejects_malformed_input`：mask 形状不符 / 末轴 ≠ 18 / `discount=nan` 全部报错。
7. **`test_geo_restore_flattens_a_one_of_eighteen_target`**：同一条 D0 首行，τ=2 下复原前 `max π_t > 0.40`、
   复原后 `< 0.08`（≈1/18）、熵 > ln 18 − 0.01、几何分差 < 折扣/20、argmax 不变
   ——机制被数值钉住（这正是 B/D 臂塌成 chance 的原因）。
8. `test_macro_recall_is_hand_checkable`：teacher `[0,0,0,1,1,2]` / pred `[0,0,1,1,2,2]` → 0.7222…，
   `classes_present=3`、逐类 recall、多数类基线 0.5。
9. `test_macro_recall_averages_only_over_present_classes`：缺席类不拉低均值；"只输出多数类"的退化模型
   macroR 0.5 而 accuracy 0.75。
10. `test_macro_recall_rejects_bad_arguments`：形状不符 / 空 / 越界全部报错。
11. `test_evaluate_reports_top1_macro_recall_kl_and_bomb`：真实 `StudentNet` 前向，`evaluate()` 的返回里
    `macro_recall` / `macro_recall_per_class` / `predicted_classes` / `top1` / `kl` / `bomb_accuracy` 都在且自洽。
12. `test_decision_steps_keep_the_frame_ruler_and_double_the_renders`：`("consecutive",2)→(2,1)`、
    `("strided",2)→(1,2)`、两者 `units×want==2`、末决策 budget=1 的退化、非法窗口/预算报错。
13. **`test_consecutive_window_is_the_training_window`**：用真实 `ObservationBuffer` 跑两种窗口，
    consecutive 的栈 = `[5,6,7,8]`（= 训练窗口 `frames[4k-4..4k-1]`），strided 的栈 = `[2,4,6,8]`
    （间隔 (FRAME_STACK−1)×DECISION_FRAMES = 6 帧）。
14. `test_the_eval_cli_defaults_to_the_aligned_window`：`OBS_WINDOWS` 两项、`run_episode` 默认 `consecutive`、
    CLI 默认 `consecutive` 且 `--obs-window dense` 被拒。
15. `test_eval_record_pins_the_window_it_was_measured_with`：记录的 `obs_window` 与
    `window_alignment.renders_per_decision` 必须一致，否则 `EvalError`；旧 `strided` 行仍可加载。
16. `test_summary_carries_the_window_forward`：`summary.json` 带 `obs_window` + `window_alignment` + 说明。

## 9. 命令与退出码（可复现）

```bash
# WSL：k=3 标签重采（8 片，2 引擎并发；.tmp_v400_p3sig/batch_k3.py 是 .tmp_v400_d0/batch.py 的
# 换 WORK/OUT 版本，--mode 默认改成 fast3、engine-limit 2、seeds 1-8）
wsl -- bash -lc "cd /mnt/d/WorkShop/Touhou_RL && PYTHONPATH=/mnt/d/WorkShop/Touhou_RL \
  /root/v400-venv/bin/python -u .tmp_v400_p3sig/batch_k3.py --seeds 1-8 --workers 2 \
  --target 30000 --decisions 5000 --max-frames 20000 --mode fast3 --engine-limit 2 --threads 1 \
  2>&1 | tee .tmp_v400_p3sig/logs/batch_k3.log"
# → exit 0（8 片 / 26,648 决策 / HASH 53,296 全等 / wall 2,899.8 s）
```

```powershell
# Windows：k=1 匹配子集（junction + 过滤 manifest）
python .tmp_v400_p3sig\make_k1_subset.py
# → exit 0（8 片 / 26,648 决策 / manifest sha256 964d3a78…）

# Windows：4 臂同配方训练（A 与 B 并发、C 与 D 并发；只有 --d0-root / --geo-restore / 输出路径不同）
python tools\rl\v400_student_train_teacher.py --d0-root .tmp_v400_p3sig/k1 \
  --checkpoint .tmp_v400_p3sig/arms/armA_k1_geooff.pt --metrics .tmp_v400_p3sig/arms/armA_metrics.json \
  --split-json .tmp_v400_p3sig/arms/armA_split.json --taus 2.0 --sweep-seconds 20 --train-seconds 420 \
  --max-total-seconds 700 --eval-every 100 --val-eval-samples 2048 --batch-size 256 --width 48 \
  --hidden 256 --seed 0 --accept-partial-d0                       # → exit 0，wall 448.9 s
#   … armB 同一行 + --geo-restore                                  → exit 0，wall 448.9 s
#   … armC/armD 同一行，--d0-root .tmp_v400_p3sig/k3（D 加 --geo-restore）→ exit 0，wall 446.4 / 446.9 s

# Windows：离线指标表
python .tmp_v400_p3sig\show_arms.py           # → exit 0（4 臂 val/test 全指标）
python .tmp_v400_p3sig\summarize_arms.py      # → exit 0（写 .tmp_v400_p3sig/arm_table.json）
python .tmp_v400_p3sig\analyze_labels.py .tmp_v400_p3sig/k1 .tmp_v400_p3sig/k3
# → exit 0（写 label_signal.json）
```

```bash
# WSL：S1N 5 局（每波 2 个进程 = 2 引擎；aligned 窗口）
wsl -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p3sig/run_evals_ab.sh consecutive
# → A/B/v2 各 exit 0；其中 armA_strided 那一行是**用错窗口的废数据**，已在下一波重跑覆盖
wsl -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p3sig/run_evals_wave2.sh
# → exit 0（armC / armD 走 consecutive，armA_strided / v2_strided 走 strided，四个都 rc=0）

# WSL：逐臂汇总 + 配对效应
python .tmp_v400_p3sig/summarize_eval.py      # → exit 0（写 eval_table.json）
python .tmp_v400_p3sig/s1n_effects.py         # → exit 0（写 s1n_effects.json）
python .tmp_v400_p3sig/show_effects.py        # → exit 0
```

## 10. 变更与产物

| 文件 | 变更 |
|---|---|
| `touhou_rl/v400/teacher/config.py` | **新增** `FAST3`（`fast3`：`FAST` 逐字段 + `verify_k=3`）并注册进 `CONFIGS`；`FAST` / `V2A` 未改，`PROFILES` 未改 |
| `tools/rl/v400_d0_dual.py` | `--mode` 的 `choices` 从 `("full","fast")` 改为**教师注册表** `tuple(sorted(CONFIGS))`（否则采不到 `fast3`；不改会让 `channel_a.json` 写出错误的档名） |
| `tools/rl/v400_student_train_teacher.py` | `--geo-restore`（`geo_restore_scores` / `UNVERIFIED_DISCOUNT`，discount 从教师 `Weights` 读取而非硬编码）、`macro_recall()` + `evaluate()` 增列 macroR / 逐类 recall / 预测类集合、`label_stats` 增加"盘上 vs 复原后"两套 `top1−top2` 与 `fraction_equal_to_discount`、`geo_restore` 证据块（复原值数 / ΣΔ / 规则 / 来源） |
| `tools/rl/v400_student_eval.py` | `--obs-window {consecutive,strided}`（默认 consecutive）+ `decision_steps()` + 逐帧 STEP/读回 + `window_alignment` 记录与校验 + `summary` 携带窗口 + `renders` 计数 |
| `tests/test_v400_labels_signal.py` | **新增** 16 用例（见 §8） |
| `Document/Update/Alpha/v4.0.0/experiments/EXP-P3-004-label-signal-2x2.md` | 本卡 |
| `.tmp_v400_p3sig/*` | `k3/`（8 片 fast3，675 MB）、`k1/`（8 片 junction + 过滤 manifest）、`arms/`（4 checkpoint + metrics + split）、`eval/`（7 个评测目录：4 臂 + v2 + 2 个旧尺复现）、`logs/`、`work/`、`batch_k3.py`、`make_k1_subset.py`、`analyze_labels.py`、`compare_trajectories.py`、`summarize_arms.py`、`show_arms.py`、`summarize_eval.py`、`s1n_effects.py`、`show_effects.py`、`run_eval.sh`、`run_evals*.sh`、`probe_d0_row0.py`、`label_signal.json`、`arm_table.json`、`eval_table.json`、`s1n_effects.json` |

未改：引擎、协议、依赖、第三方、`build/`；未删除任何文件；未提交；`touhou_rl/v400/student/*` 未改。

## 11. 限制（不要过度解读）

1. **不是 G3、不是 PSN、不是 confirm split**；1 个 stage、1 个难度、n=5 局。
2. **数据集只有 8 片**（成对设计的选择，§1）：绝对帧数不能直接与 2,399.6（27 片）比，所以额外用
   `checkpoint-v2` 在同一条尺上作参照（§5.2）。`v2 − A = +552.2` 这部分差距是**数据量**，不是因子。
3. **`geo_restore` 的"有用"是"别塌缩"的有用，不是"更准"的有用**：它的 4 臂离线指标全线下降（§4），
   正效应只在闭环出现；本卡不把 `B` 的 1,614.4 解释为"学生变强了"。
4. **`verify_k=3` 的负结论限于本 teacher 配置**：`fast3` 只改 `verify_k`，search 预算仍是
   `horizon=32 / beam=24 / max_iterations=1`。`v2a/v3a`（更大 beam/horizon）可能让 top-3 之间出现
   真实差异——但那已经是"换 search 预算"，不是"换 verify_k"，本卡没有测。
5. **`--accept-partial-d0`**：8 片 = 26,648 决策 < 50,000 契约，4 臂都带 `gate_waived=true` 记录在
   `metrics.json`；这是**设计**（成对），不是采集截断。
6. `bomb` 在两套标签里都是 0 正例（D0 全量也是 0），所以 4 臂的 bomb acc 都是空指标（`identifiable=false`）。
7. `test_v400_teacher.py::test_child_recording_is_one_batch_and_matches_the_main_process_tape` 的 1 个红是
   **既有的 v2h 激光字段**问题（差异项 `{'lasers_damageable': 0}`，来自 EXP-P1-021 的 `hazard_tape` 改动，
   本轮未触碰该文件），不是本轮引入。
8. **窗口修正后历史数字需整体下修**（§6.3）：本卡只在 2 个 checkpoint 上校准（−25.6 % / −28.4 %），
   别的 checkpoint 可能略有不同，但方向与量级应一致。
9. **`strided` 模式的逐种子复现（3,636 / 1,866 / 1,790 / 1,580 / 3,126）是确定性证据**，
   说明两个窗口之外的东西都没变（引擎、种子、动作选择、帧计数规则）。
10. **采集受机器负载影响**：本轮 wall 48.3 min（D0 同期 2.06 h 完成 16 片），并发 2 引擎时有别的实验在跑
    （`loadavg` 一度 24–26），所以每片 580–791 s 的绝对值不是 `fast3` 的吞吐上限。

> **更正注记（2026-11-10，EXP-P3-009）**：本卡 §7 的 `+552.2`（数据量 8→27 片）、`+447.8`（`geo_restore`，t=24.9）、`−147.5`（`verify_k` 1→3）都是**单 checkpoint × 5 局**的均值比较；按本卡新判据，`verify_k` = 无法判定（CI 重叠），另外两条 = 无法判定 / 无法判定（CI 不重叠，但 |Δ| 低于 checkpoint 级分辨极限 1,309.7 帧）。原始数字不改。详见 [EXP-P3-009](<EXP-P3-009-eval-power.md>)。
