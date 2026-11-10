# EXP-P3-005 margin 分辨率：`v2i`（`margin_cap` 16→64）+ v2h 预算重采 8 片 → 重训 → S1N 5 局

- 日期 / 切片：2026-11-09 / **P3-WP5-a**（EXP-P3-004 §7.3 的 P0：给教师侧几何分辨率做一次直接实验）
- 依赖：[EXP-P3-004](<EXP-P3-004-label-signal-2x2.md>) §2.3（82 % 决策的 18 路几何分差 < 0.06）、
  §7.3（P0 = 提高几何分辨度）、§6（窗口修正：新尺 `consecutive`）、
  [EXP-P1-003](<EXP-P1-003-teacher-core-first.md>)（"margin 全顶 16"）、
  [EXP-P1-021](<EXP-P1-021-laser-damageable.md>)（`v2h` 激光项 = `v2i` 的另一半）

## 0. 结论先说（四句话）

1. **`v2i` 按任务卡注册完成，但 `Weights.margin_cap` 16→64 在数学上不可能改变任何一条标签**：
   16 px 的钳位发生在**权重之前** —— `search._frame_margin`/`_margin_rows` 把逐帧净空截到模块常量
   `config.MARGIN_CAP = 16.0`，raster 表再把同一值存进 `uint16` 低字节（`MARGIN_Q_CAP = 64`，1/4 px 定点，
   最大可表示 **63.75 px**，64.0 会溢出到 flags 字节）。用真实教师代码实测：一条真实净空 41.175 px 的 tape，
   `simulate_plan` 返回 **16.0**，`v2h`/`v2i` 的 18 路分数**逐位相同**（§1.2）。
2. **因此本轮 v2i 数据集 = "v2h 预算 + 激光项" 的数据集**（horizon 48 / beam 64 / `verify_k=3` /
   `max_iterations=3` / `laser 12@8`），而不是"margin 分辨率更高"的数据集。这一点不靠推断：新标签里
   `chosen_plan.margin` 的最大值在 26,400 条上**恰好是 16.0**、96.41 % 恰等于 16.0（§3）。
   实验仍然有价值 —— 它是**第一次**用教师完整搜索预算（此前 D0/DAgger/fast3 全是 `fast` 的 32/24/1）产出标签，
   能回答"几何分辨率是不是瓶颈"的另一半：**搜索预算 / 连续几何项**能否把 `>0.1` 占比拉起来。
3. **标签分辨率**：`v2i` 的 `复原后 top1−top2 > 0.1` 占比 **2.2159 %**（旧 **1.20 %**，+84.5 % 相对）、
   p90 **0.0795 px**（旧 0.0589，+35 %）、`> 1.0` **0.0795 %**（旧 0.10 %）；argmax 直方图
   `[11146, 786, 1824, 2238, 652, 435, 1423, 2422, 556, 0, 151, 1846, 1470, 233, 141, 669, 226, 182]`
   （最大类占比 **0.4222**，旧 0.4375；归一化熵 0.7244 vs 0.6871）—— **比 `k=1` 更分散，但没有质变**：
   类数仍是 17、97.8 % 的决策分差仍 < 0.1 px、p90 只有 0.08 px。增量来自**连续项**（激光 12@8、底带 1.5/2.0）
   与**更深的搜索**，不是 margin（margin 的最大值在全部 26,400 条新标签里仍是 16.0）。
4. **闭环（新尺 `consecutive`，seeds 1000–1004，stage 0，cap 19,887，`--stop-on-death`，1 引擎）**：
   E（新 8 片）**1,209.6**、F（27 片口径）**1,816.0** 帧，v2 基线 **1,718.8**、教师 30 % 门 **4,070**。
   **预注册判据 FAIL**：F 过帧数腿（105.7 %）但两臂的 `macroR`（0.0879 / 0.1471）都远低于 0.25；
   E 两条腿都不过（70.4 % / 29.7 %）。配对看，**只换标签来源的两组都测不出差别**（F−v2 = +97 帧 t=0.37；
   E−armA = +43 帧 t=0.33）⇒ **几何分辨率不是闭环瓶颈**。**本卡不是 G3 通过、不是 PSN、不是 confirm split。**

## 1. `v2i` 档：改了什么、为什么这么改

`touhou_rl/v400/teacher/config.py` 新增 `V2I`（注册进 `CONFIGS` 与 `PROFILES`）：

```python
_V2I_WEIGHTS = replace(_V2H_WEIGHTS, margin_cap=64.0)
V2I = TeacherConfig(mode="v2i", horizon=48, beam=64, verify_k=3, max_iterations=3,
                    weights=_V2I_WEIGHTS, bomb_rule1_max_death_frame=12)
```

| 字段 | `v2h` | `v2i` |
|---|---|---|
| `weights.margin_cap` | 16.0 | **64.0** |
| `horizon` / `beam` / `verify_k` / `max_iterations` | 48 / 64 / 3 / 3 | 同（逐字段相等） |
| `bottom_penalty` / `bottom_extra_penalty` | 1.5 / 2.0 | 同 |
| `high_penalty`（`high_y` 200） | 3.0 | 同 |
| `laser_penalty` / `laser_band` | 12.0 / 8.0 | 同 |
| `bomb_rule1_max_death_frame` | 12 | 同 |
| `durations` / `decision_frames` | (1, 2) / 2 | 同 |
| `unverified_discount` | 5.0 | 同（**不改** `verify_k`：EXP-P3-004 已证它对信号无益） |

测试钉住 `replace(V2I, mode="v2h", weights=replace(V2I.weights, margin_cap=16.0)) == V2H`
（`tests/test_v400_margin_resolution.py::test_v2i_is_v2h_with_the_margin_cap_raised_and_nothing_else`）。

### 1.1 为什么选线性 `margin_cap=64`，不选 `log1p(margin)`

1. **量纲**：11-P1 §3 的其它项全是"点"且线性 —— `align_weight=4.0`（0..1 比率）、`bottom` 1.5/2.0、
   `high` 3.0、`laser` 12.0。线性 margin 保持"1 px 净空 = 1 点"，只是把量程从 16 px 扩到 64 px，
   已发表的权重仍然字面成立（"16 点是整个 margin 项" → "16 点是 1/4 个 margin 项"）。
2. **log 的等价量纲会让权重表失真**：`log1p(16)=2.83`、`log1p(64)=4.17`，总动态范围只有 1.34 点 ——
   比 `edge`(1.0)+`high`(3.0) 还小、比激光项(12.0)小 9 倍；且边际价值 `d/dm log1p(m)=1/(1+m)`
   在 16 px 处 0.059 点/px、64 px 处 0.015 点/px，同一段净空在不同位置值不同的分 —— 这是给一张
   "在 16 点线性项下标定过"的权重表**新增非线性**（EXP-P1-015/017/020 的带子都是按"整个 16 点 margin 项"的
   百分比标定的）。而且 log 变换无法用任何现有 `Weights` 字段表达，等于同时改"标签的语义形状"和"一个数字"。
3. 所以：**线性 cap=64**，二选一里它是唯一"只动一个量、且量纲不变"的选项。

### 1.2 结构性钳位：cap 在权重之前就已经被截断（本轮最重要的更正）

`Weights.margin_cap` 是**最后一道**截断，作用在一个已经被截断的数上：

| 层 | 位置 | 行为 |
|---|---|---|
| 逐帧净空 | `search._frame_margin` / `_margin_rows` | `np.minimum(gap, MARGIN_CAP)`，`MARGIN_CAP = 16.0` |
| raster 表 | `raster.build_table` / `MARGIN_Q_CAP` | `min(best, 16.0)` 存进 `uint16` 低字节，1/4 px 定点 ⇒ 最大 **63.75 px** |
| 束搜索键 | `search` 的 `new_margin = full(MARGIN_CAP)`、`keys = 1000 + MARGIN_CAP − margin` | 排序键同样以 16 为上限 |
| 打分 | `search.score_plan` / `_finish_roots` | `min(margin, weights.margin_cap)` ← **只有这里是 `Weights`** |

**实测（`.tmp_v400_margin/prove_clamp.py`，exit 0，写 `margin_clamp.json`）**：

| 量 | 值 |
|---|---|
| 测试 tape 的真实净空 | **41.175 px** |
| `simulate_plan` 返回的 margin | **16.0**（= `MARGIN_CAP`，钳位生效） |
| `score_plan(margin=41.175)`，`v2h` | **120.0** |
| `score_plan(margin=41.175)`，`v2i` | **145.175**（Δ = **25.175** = 41.175−16 ⇒ 权重本身是活的） |
| `beam_search` 18 路分数（`v2h` vs `v2i`） | **逐位相同**，`max_abs_delta = 0.0` |
| raster 字节预算 | `QUANT=4`、`MARGIN_Q_CAP=64`、可表示上限 **63.75 px**、`64×4 = 256 > 255` ⇒ **溢出** |

**结论**：要让 margin 分辨率真的提高，必须改 `config.MARGIN_CAP` **并**把它贯通到 `raster`/`search`
（且真实上限只能到 63.75 px），这不是"新增一个档"能做的改动。本轮**没有**动 `search.py`/`raster.py`
（任务卡的允许修改范围只到 `config.py` 的新增档 + `tools/rl/v400_teacher.py` 的 mode 支持 +
训练器数据源参数 + 新增测试），所以 `v2i` 的标签与 `v2h` **完全相同**，而采集数据仍然有独立价值（见 §0.2）。

## 2. 采集（`stage 0` / `normal` / seeds 1–8，引擎 `build/rl_laser`）

- 采集器：`tools/rl/v400_d0_dual.py --phase all`（channel A shadow 出标签、channel B env 出像素，
  HASH 逐检查点对齐），`--mode v2i`；**没有**改 `--mode` 的 choices（它早就是教师注册表）。
- 引擎：`/root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern`，
  sha256 `1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f`（`--expected-sha` 同值）
  —— EXP-P1-021 之后的二进制，**保留激光判定窗**（`laser_penalty=12` 需要它）。
- 并发：**2 个 worker**（2 teacher × shadow/child，WSL 引擎实测 `pgrep -c th08-modern` **5–6 ≤ 6**）；
  `--engine-limit 6`、`--threads 1`（numba/BLAS 单线程，避免和机器上的零售 trace 批跑抢核；
  3 worker 时实测 7–8 个 `th08-modern`，超预算，已回退到 2）。
- 参数：`--decisions 3300 --max-frames 20000 --stop-on-death --allow-early-stop --trim-transition`，
  `--target 26400`。

| shard | seed | decisions | frames | stop_reason | HASH 检查点 | 对齐 | channel A wall | channel B wall | wall | obs bytes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| dual-s0d1seed1 | 1 | 3,300 | 13,200 | decisions | 6,600 | 6,600/6,600 | 384.6 s | 112.6 s | 505.4 s | 83,636,244 |
| dual-s0d1seed2 | 2 | 3,300 | 13,200 | decisions | 6,600 | 6,600/6,600 | 386.2 s | 112.2 s | 510.5 s | 83,636,244 |
| dual-s0d1seed3 | 3 | 3,300 | 13,200 | decisions | 6,600 | 6,600/6,600 | 374.9 s | 112.5 s | 496.3 s | 83,636,244 |
| dual-s0d1seed4 | 4 | 3,300 | 13,200 | decisions | 6,600 | 6,600/6,600 | 378.5 s | 112.5 s | 499.7 s | 83,636,244 |
| dual-s0d1seed5 | 5 | 3,300 | 13,200 | decisions | 6,600 | 6,600/6,600 | 384.8 s | 112.3 s | 505.6 s | 83,636,244 |
| dual-s0d1seed6 | 6 | 3,300 | 13,200 | decisions | 6,600 | 6,600/6,600 | 379.8 s | 111.3 s | 502.2 s | 83,636,244 |
| dual-s0d1seed7 | 7 | 3,300 | 13,200 | decisions | 6,600 | 6,600/6,600 | 374.1 s | 111.0 s | 494.0 s | 83,636,244 |
| dual-s0d1seed8 | 8 | 3,300 | 13,200 | decisions | 6,600 | 6,600/6,600 | 378.8 s | 113.2 s | 500.1 s | 83,636,244 |
| **合计** | 1–8 | **26,400** | **105,600** | 8× decisions | **52,800** | **52,800/52,800 = 100 %** | ~3,040 s | ~898 s | **2,012.8 s（33.5 min）** | **669,089,952** |

`progress.json`：`decisions=26400 / shards_done=8 / shards_failed=0 / hash_alignment_rate=1.0 /
hash_checkpoints=52800 / hash_checkpoints_equal=52800 / load_verified_shards=8 / elapsed_s=2012.8 /
complete=true / partial=false`（`--target 26400` 恰好填满，不是截断标记）。

`load_gate.json`（逐片 `dataset.load_teacher_samples(require_observations=True)`）：8/8 片 `matches=true`、
`all_digest_ok=true`（`obs_spec_digest 24f0ddac…`）、`unresolved_observations=0`、
**`max_chosen_margin_px = 16.0`**、`margin_clamped_at_16 = true`。

**每决策成本**：channel A（teacher）= 374–386 s / 3,300 ≈ **114 ms/决策**；channel B（env 回放 + 像素）≈ **34 ms/决策**
（整片 496–510 s）。8 片串行 ~6.7 h，2 worker 下 **33.5 min**（13.1 决策/s），每片 3,300 决策满足"≤3,500"。
`stop_reason=decisions`（按决策数截断）：D0 的同种子在 3,274–3,394 决策自然换面，所以只切掉尾部几十条，
分片身份（seed、stage、obs spec、对齐）不变。

## 3. 标签分辨率：新旧逐项对比

读数口径与 EXP-P3-004 §2.3 **完全一致**（`analyze_labels.py`，同一 `restored = scores + 5·(1−mask)`），
旧列就是本脚本在旧 root 上的复算（p90 0.058925、`>0.1` 1.2008 %，与 EXP-P3-004 **逐值复现**）。

| 指标 | 旧 k=1（D0 seeds1–8） | 旧 k=3（fast3 seeds1–8） | **新 v2i（seeds1–8）** | v2i vs k=1 |
|---|---:|---:|---:|---:|
| 决策数 | 26,648 | 26,648 | **26,400** | — |
| `verified_count` mean | 1.00 | 3.00 | **3.00** | — |
| `verified_mask[第 2 名]` | 0.00 % | 100 % | **100 %** | — |
| 存盘 `top1−top2` 恰等于 5.0 | 82.16 % | 0.00 % | **0.00 %** | 折扣伪影仍不存在 |
| 复原后 `top1−top2` 均值 | 0.014488 | 0.014488 | **0.016801** | +16.0 % |
| **p50** | 0.0 | 0.0 | **0.0** | 不变 |
| **p90** | **0.058925** | 0.058925 | **0.079545** | **+35.0 %** |
| p99 | 0.117851 | 0.117851 | **0.166667** | +41.4 % |
| max | 5.573148 | 5.573148 | **4.827411** | −13.4 % |
| **`>0.1` 占比** | **1.2008 %** | 1.2008 % | **2.2159 %** | **+84.5 %（+1.02 pp）** |
| `>1.0` 占比 | 0.1013 % | 0.1013 % | **0.0795 %** | −21.5 % |
| `>5.0` 占比 | 0.0038 % | 0.0038 % | **0.0 %** | — |
| argmax 类数（共 18） | 17 | 17 | **17** | 不变（动作 9 从未被选中） |
| argmax 最大类占比 | 0.4375 | 0.4375 | **0.4222** | −0.015（更分散一点） |
| argmax 归一化熵 | 0.6871 | 0.6871 | **0.7244** | +5.4 % |
| 教师目标熵（τ=2） | 2.2687 | 2.1240 | **2.1200** | −6.6 % |
| 教师动作 0 占比 | 43.75 % | 43.75 % | **42.22 %** | −1.5 pp |
| **`chosen_plan.margin` p50 / p90 / max** | 16.0 / 16.0 / 16.0 | 16.0 / 16.0 / 16.0 | **16.0 / 16.0 / 16.0** | **完全没变（钳位）** |
| `chosen_plan.margin` 恰等于 16.0 | 96.06 % | 96.06 % | **96.41 %** | **+0.35 pp（钳位依旧）** |
| bomb 正例 | 0 | 0 | **0** | — |

argmax 直方图（18 类，顺序 = 动作 0..17）：

| root | 直方图 |
|---|---|
| 旧 k=1 / k=3（逐类相同） | `[11659, 616, 2279, 2646, 548, 335, 1799, 1877, 285, 0, 73, 1707, 1934, 194, 53, 309, 224, 110]` |
| **新 v2i** | `[11146, 786, 1824, 2238, 652, 435, 1423, 2422, 556, 0, 151, 1846, 1470, 233, 141, 669, 226, 182]` |

**读法（逐项对比的结论）**：

1. **`margin_cap` 没有留下任何痕迹**：`chosen_plan.margin` 的 p50/p90/**max 全是 16.0**，96.41 % 的决策恰好等于 16.0
   —— 和旧标签的 96.06 % 几乎一样。这不是"教师恰好都只有 16 px 净空"，而是 §1.2 的结构性钳位：
   `simulate_plan`/raster 在打分之前就把净空截到 16 px，所以 `margin_cap=64` 从来没有被用到。
2. **分辨率确实上升了一点，但来自连续项而不是 margin**：`>0.1` **1.2008 % → 2.2159 %**（+84.5 % 相对、+1.02 pp），
   p90 **0.0589 → 0.0795**（+35 %），p99 0.118 → 0.167。这些增量只可能来自 `v2h` 的连续项
   （`laser_penalty=12 @ 8 px`、`bottom 1.5/2.0`）与更深的搜索（32/24/1/1 → 48/64/3/3）。
3. **argmax 只是"稍微"更分散**：类数仍 17、最大类占比 0.4375 → 0.4222、归一化熵 0.687 → 0.724。
   和 k=1 比，**没有出现质变**（EXP-P3-004 §2.3 里 k=1→k=3 的 argmax 直方图是**逐类相同**的；本轮至少每一类都动了，
   但形状仍由动作 0/3/7/11 主导）。
4. **绝对水平仍然是"平"**：p90 = **0.08 px**，也就是 18 个候选里最强的那个只比第二名好 8 % 个 margin 点
   （1 点 = 1 px）。教师的真实偏好率（`>0.1`）从 1.2 % 变成 2.2 %，**98 % 的决策仍然没有可学的排序**。

## 4. 重训（两种口径，分开报告）

- 配方与 armA/armB/`checkpoint-v2` **完全相同**：width 48 / hidden 256（204,260 参数）、τ=2.0、
  `KL(π_t‖π_s)+BCE`、AdamW lr 2e-3、batch 256、seed 0、`--train-seconds 420`（≤10 min）；**不带** `--geo-restore`
  （问题是"教师写的标签本身分辨率够不够"，不能再动第二个因子；`geo_restore` 是 EXP-P3-004 的另一个因子）。
- **口径 E**：`--d0-root .tmp_v400_margin/v2i`（新 8 片），`--accept-partial-d0`（26.4k < 50k 契约，成对设计）。
- **口径 F**：`--d0-root .tmp_v400_margin/v2i --extra-root .tmp_v400_margin/d0_916 --extra-root .tmp_v400_dagger/d0_extra`
  = **27 片 / v2 同数据量**。`d0_916` 是 D0 的 seeds **9–16**（8 片，NTFS junction，26,864 决策，
  manifest sha256 `e9bdbae095606a1396158044bb8456c0dcf7b17fb7703638e4294d5523b39095`）。
  **为什么不是字面的 35 片（新 8 + v2 的 27）**：`dual-s0d1seed1..8` 这个**分片键**在新旧两套里完全相同
  （同一 stage/seed），训练器会 `SystemExit("duplicate shard key(s) across roots")`
  （`v400_student_train_teacher.py:688-689`）；就算改名，同一个 seed 的同一局会同时落进 train 和 val/test，
  数据泄漏。所以口径 F 用 **D0 seeds 9–16 替换 D0 seeds 1–8**（那 8 片被新的 v2i 同键分片取代），
  总数仍是 **27 片、76,664 决策**，与 v2 的 76,912 对齐 —— 这是"新 8 片 + v2 其余 19 片"，改动最小且不泄漏。

| 臂 | root（分片） | 决策 | split（train/val/test） | τ | 参数 | step / 样本 | wall | 跑满预算 |
|---|---|---:|---|---:|---:|---:|---:|---|
| **E** | v2i 8 片 | 26,400 | 19,800（seeds 1–6）/ 3,300（seed 8）/ 3,300（seed 7） | 2.0 | 204,260 | 14,948 / 3,826,688 | **458.3 s** | 是（`aborted_at_deadline=true`） |
| **F** | v2i 8 + D0 9–16 + DAgger 11 = **27 片** | **76,664** | 52,362（19 片）/ 12,294（4 片）/ 12,008（4 片） | 2.0 | 204,260 | 15,562 / 3,983,872 | **513.2 s** | 是 |

E 的 `split` 分片键与 armA/B/C/D **逐字节相同**（val=seed8、test=seed7），所以 §4.1 的离线对比是配对的。
E 带 `--accept-partial-d0`（26,400 < 50,000 契约，`gate_waived=true` 记录在案，设计使然）；F 不豁免。
两臂都 `--geo-restore off`、`unverified_discount=5.0` 未改。

### 4.1 离线指标（整 split 全量）

| 臂 | val top-1 | val top-3 | **val macroR** | val KL | val 教师熵 | test top-1 | **test macroR** | test KL | val 预测类数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **E**（8 片，v2i） | 0.3288 | 0.4721 | **0.0879** | 0.8019 | 2.1098 | 0.4724 | **0.0971** | 0.5910 | 12 |
| **F**（27 片，v2i） | 0.5493 | 0.7103 | **0.1471** | 0.4129 | 2.2642 | 0.5370 | **0.1426** | 0.4274 | 12 |
| 参照 `checkpoint-v2`（27 片，k1/fast） | 0.5912 | — | **0.1912** | 0.3473 | — | 0.5766 | **0.1887** | 0.3542 | — |
| 参照 armA（8 片，k1/fast） | 0.5656 | — | **0.2514** | 0.4176 | 2.2608 | 0.5696 | **0.2487** | 0.3957 | 16 |

（参照行来自 EXP-P3-004 §4，不同 split 不完全可比，只作绝对锚点。）

**offline 全线更差**：同样口径下 v2i 标签比 fast 标签**更难克隆** —— 8 片口径 val top-1 0.5656 → 0.3288、
macroR 0.2514 → 0.0879、KL 0.4176 → 0.8019；27 片口径 0.5912 → 0.5493、0.1912 → 0.1471、0.3473 → 0.4129。
逐类 recall 也印证塌缩方向：E 的动作 0 recall 0.889（其余 11 类加起来 0.11），F 的动作 0 0.913、
动作 7 0.659、动作 3 0.401 —— **教师更"公平"的标签没有让学生更准，只是让它更常押动作 0**。

### 4.2 两臂之间只有数据量

`F − E`：val top-1 **+0.2205**、val macroR **+0.0592**、test top-1 +0.0646、test macroR +0.0455、val KL −0.3890。

### 4.3 标签结构（训练器自报，证明 `--geo-restore` 关闭时只读原值）

| 臂 | 盘上 `top1−top2` 均值 / 恰等于 5.0 | 训练用同上 | `verified_mask_mean` | `scores` 范围 |
|---|---:|---:|---:|---|
| E | 0.01706 / 0.0 | 同（未复原） | 3.00 | −4.375 … 120.0 |
| F | 2.82164 / 0.357 | 同（未复原） | 1.88 | −5.0 … 120.0 |
| 旧 armA | 5.0141 / 0.826 | 同 | 1.00 | −4.06 … 120.0 |

## 5. S1N 5 局（新尺 `consecutive`）

- 尺：`tools/rl/v400_student_eval.py --seeds 1000,1001,1002,1003,1004 --max-frames 19887 --stage 0
  --difficulty normal --stop-on-death --obs-window consecutive`，引擎 `build/rl/th08-modern`
  （sha256 `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb`，与 EXP-P3-002/003/004 同一把尺的同一引擎），
  **1 引擎/进程、2 进程并发**（2 引擎）。
- 记录的 `window_alignment.renders_per_decision` 与 `consecutive` 一致（`validate_episode_record` 拒绝自相矛盾的行）。

### 5.1 逐局

| seed | **E**（8 片） | **F**（27 片） | v2 基线（EXP-P3-004 同尺） |
|---:|---:|---:|---:|
| 1000 | 943 | 1,763 | 1,644 |
| 1001 | 1,233 | **2,907** | 1,852 |
| 1002 | 945 | 1,293 | 1,409 |
| 1003 | 1,399 | 1,324 | 1,885 |
| 1004 | 1,528 | 1,793 | 1,804 |
| **均值** | **1,209.6** | **1,816.0** | **1,718.8** |
| 中位数 | 1,233 | 1,763 | 1,804 |
| sd / se | 264.0 / 118.1 | 653.7 / 292.3 | — |
| 均值 score | 17,767.6 | **29,373.8** | 25,114.2 |
| 过关 | 0/5 | 0/5 | 0/5 |
| 首死停 | 5/5 | 5/5 | 5/5 |
| `renders` | = frames（943…1,528） | = frames（1,293…2,907） | = frames |

`obs_window=consecutive`、`renders ≈ frames`（`consecutive` 每决策 2 次渲染）—— 两条记录都通过
`window_alignment` 校验，**没有混用旧 strided 尺**。

### 5.2 配对（同一组 5 个评测种子）

| 对比 | 均值差（帧） | sd | 配对 t | 同向种子 |
|---|---:|---:|---:|---|
| **F − v2**（27 片 v2i vs 27 片 fast） | **+97.2** | 593.4 | **0.37** | 2/5 |
| **E − v2**（8 片 v2i vs 27 片 fast，混杂数据量） | **−509.2** | 162.6 | **−7.00** | 0/5 |
| **F − E**（数据量 27 vs 8） | **+606.4** | 677.0 | **2.00** | 4/5 |
| E − armA（8 片 vs 8 片，k1/fast） | +43.0 | 264.1 | 0.33 | 2/5 |

**读法**：F 与 v2 在统计上**不可区分**（+97 帧、t=0.37、2/5）；E 比 v2 显著差 **−509 帧**，但把它和**同数据量的
armA（1,166.6）**比只差 +43 帧（t=0.33）—— 也就是说 **8 片口径下"重教师"与"fast 教师"没有差别**，
差的是数据量（F−E = +606，t=2.00，与 EXP-P3-004 的 `v2 − A = +552` 同量级）。

## 6. 判据（预注册）

| 判据 | 门槛 | E（8 片） | F（27 片） | 结果 |
|---|---|---|---|---|
| S1N 均值（新尺） | **≥ 1,718.8**（v2 基线） | 1,209.6（**70.4 %**）✗ | **1,816.0（105.7 %）** ✓ | **F 过，E 不过** |
| macro recall（val） | **≥ 0.25** | **0.0879** ✗ | **0.1471** ✗ | **两臂不过** |
| ≥ 教师 30 % | 4,070 | 1,209.6（29.7 %）✗ | **1,816.0（44.6 %）** ✓ | F 过，E 差 0.3 pp |

**判定：按预注册规则（两条腿都要过）→ FAIL**，失败的是 `macroR ≥ 0.25` 这条腿（两臂都差很远）。
三条如实标注：

1. **F 的帧数腿是唯一一次在 v2 数据量上追平基线**（1,816.0 vs 1,718.8），但配对 t = 0.37、2/5 同向
   ⇒ **不能说"重教师的标签更好"**，只能说"没有变差"；
2. **F 与 v2 的差别只有标签来源**（数据量 76,664 vs 76,912、同 seeds 结构、同配方），E 与 armA 同理；
   在这个配对下，v2i 的标签**没有带来闭环增益**，这正好回答了 §7 的问题；
3. **`macroR` 这条腿在 EXP-P3-004 里只有 armA 的 val（0.2514）过过**，而本轮两臂的 macroR（0.088/0.147）
   系统性更低 —— 因为 v2i 的教师目标更"公平"（动作 0 从 43.75 % 降到 42.22 %、熵 2.27→2.12），
   学生的多数类先验不再能虚高 macroR。**这不是测量失败，是 macroR 与 top-1 的方向不一致**（见 §7.3）。

## 7. 「几何分辨率是否真是瓶颈」的判定

分三层回答，不合并成一个数：

### 7.1 被判死的机制：`margin_cap` 不是可以拧的旋钮

| 证据 | 值 |
|---|---|
| 新标签 `chosen_plan.margin` 的最大值（26,400 条） | **16.0**（p50 = p90 = max = 16.0） |
| 新标签恰好顶在 16.0 的比例 | **96.41 %**（旧标签 96.06 %） |
| 真实净空 41.175 px 的 tape → `simulate_plan` | **16.0** |
| 同一 tape 上 `v2h` vs `v2i` 的 18 路分数（两种搜索路径） | **逐位相同**，`max_abs_delta = 0.0` |
| `score_plan(margin=41.175)` 的 `v2i−v2h` | **+25.175**（权重活着，只是拿不到 >16 的数） |
| raster 可表示上限 | **63.75 px**（`QUANT=4`、低字节 255），`64×4=256` 会溢出 |

⇒ **EXP-P3-004 把"直接原因"写成"评分用 `min(margin,16)` 封顶"是不完整的**：封顶发生在 `search._frame_margin` /
`raster` 层，`Weights.margin_cap` 只是最后一道、作用在已封顶值上的 `min`。**任何只改 `Weights` 的档都不会改变标签**
（本轮的 26,400 条新标签与 `v2h` 会产出的标签相同）。

### 7.2 教师侧能拿到的分辨率增量：有，但很小

| 量 | 旧 | 新 | 变化 |
|---|---:|---:|---|
| `>0.1` 占比（真实偏好率） | 1.2008 % | **2.2159 %** | **+84.5 % 相对 / +1.02 pp** |
| p90 分差 | 0.0589 px | **0.0795 px** | +35.0 % |
| p99 分差 | 0.1179 px | 0.1667 px | +41.4 % |
| argmax 归一化熵 | 0.6871 | 0.7244 | +5.4 % |
| 动作 0 占比 | 43.75 % | 42.22 % | −1.5 pp |

分辨率**确实上升**（证据链的方向是对的：`verify_k` 不动它，`laser`/`bottom`/更深搜索动它），
但**绝对水平仍然是平的**：97.8 % 的决策里 18 个候选的 top1−top2 仍 < 0.1 px，p90 = 0.08 px。

### 7.3 闭环：这点分辨率增量没有转化成行为

| 配对（同一组 5 个评测种子） | 均值差 | t | 同向 |
|---|---:|---:|---|
| F − v2（同 27 片数据量，只换标签来源） | **+97.2 帧** | 0.37 | 2/5 |
| E − armA（同 8 片数据量，只换标签来源） | **+43.0 帧** | 0.33 | 2/5 |

**在"只换标签"的配对控制下，v2i 与 fast 的闭环表现不可区分**（+97 / +43 帧，都远小于 se）。
对照 EXP-P3-004 里真正动过闭环的两个因子：

| 因子 | 改的是什么 | 闭环效应 |
|---|---|---|
| `geo_restore` | **目标形状**（把 15–17 个 −5 加回去、目标变平、熵 2.26→2.86） | **+463.1 帧**（5/5，t = 24.9） |
| 数据量（27 片 vs 8 片） | **样本数** | **+552 ~ +606 帧** |
| 教师预算 + 连续几何项（本轮） | **分辨率**（`>0.1` +84.5 %） | **+97 / +43 帧，不显著** |

### 7.4 判定

**「几何分辨率是 P3 标签的瓶颈」这个命题，按本轮能拿到的证据：不成立（至少不是闭环瓶颈）。**

- 被点名的机制（`margin_cap` 封顶）**在结构上就是惰性的**，改它不会产生任何新标签；
- 通过教师侧**唯一可动**的连续项（激光 12@8、底带 1.5/2.0）、更深的搜索（beam 24→64、horizon 32→48、
  max_iter 1→3），真实偏好率能翻近一倍（1.2 % → 2.2 %），但**闭环上测不出差别**；
- 真正动过闭环的是**目标形状（塌缩程度）**与**数据量**，不是分辨率。

**必须同时写明的边界**（不要把它读成"分辨率永远没用"）：

1. 本轮能到达的分辨率上限只有 **2.2 %**，离"教师能稳定排序 18 个动作"还很远；**没有**测过"把 cap 真正贯通到
   60 px、把 `>0.1` 拉到 30 %+"的世界。要测那个，必须改 `MARGIN_CAP` + raster 编码（超出本任务允许的修改范围）；
2. `macroR` 与 top-1 在 v2i 上**反向**（macroR 0.088/0.147、top-1 0.33/0.55），说明"更平的目标"会让
   macroR 变差而 top-1 反而上升 —— **macroR ≥ 0.25 这条腿与"目标更公平"是冲突的**，后续判据要分开处理；
3. n = 5 局、单 stage，se 118–292 帧，+97 帧的效应即使真实也小于本轮的分辨率上限带来的预期。

## 8. 下一条最小改动

按本轮的证据排序（推荐只做 P0，然后按结果决定）：

| 优先级 | 改动 | 为什么是这一条 | 代价 |
|---|---|---|---|
| **P0** | **动作持续时间集 `durations` {1,2} → {1,2,3,4}**（`TeacherConfig.durations`，**纯配置**，新增档 `v2j`） | 18 个候选之所以"分不开"，物理原因是**它们几乎是同一个动作**：1–2 个决策槽 = 2–8 px 位移，而 margin 项在 16 px 就饱和、`align` 只有 4.0/192 = 0.021 点/px。把承诺时长扩到 4 槽（16 px）能同时做两件事：(a) 端点散布 ×2–4，`align`/`bottom` 产生真实梯度；(b) 让部分计划的净空**落到 16 px 以下**，进入 margin 项**仍然活着**的区间 —— 这是本轮唯一"不碰引擎、不碰协议、不碰打分函数"却能让候选真正分开的旋钮。 | 根数 18×2=36 → 18×4=72，搜索 ~2×（实测 114 ms/决策 → 预计 ~200 ms/决策），8 片采集 ~65 min（2 worker）；训练与评测配方不变，仍可与 armA/E 配对 |
| P1 | `horizon` 48 → 64（`v3a` 已有先例） | 只加长计划，不改变首动作之间的位移，**不会**分开候选；只在 P0 之后用来验证"更长视野是否让 `align`/激光项更早起作用" | 搜索 +~40 %；EXP-P1-019 已测过 v3a 的代价 |
| P2 | 把 `items` 变成连续项（`record_items=True` + 权重 >0） | 需要一个每帧 `ITEMS` 分页的引擎字段（协议里已有，但会让 tape 每帧多一次往返）；是**新的连续信号**，方向对，但不是最小改动 | tape 成本 +1 往返/帧（EXP-P1-007 实测 ~5 %），采集吞吐下降 |
| P3 | 真·提高 margin 分辨率：`config.MARGIN_CAP` 16 → **60**，并把它贯通 `raster.MARGIN_Q_CAP` / `search` 的束键 | 只有这样才能让 26,400 条标签里 96 % 的"margin 恒等于 16"变成真正的梯度；上限 **63.75 px**（1/4 px × 1 字节），并且**必须同时改三处**，否则就是本轮的惰性档 | 改 `search.py`/`raster.py`（本任务**不允许**）；会让所有既有档的束排序键一起变，需要逐档回归 |
| 不推荐 | 继续调 `verify_k` / `geo_restore` / 更重的搜索预算 | EXP-P3-004 与本轮各自证伪了一次：`verify_k` 无信号、`geo_restore` 是"别塌缩"而不是"更准"、更重的搜索在两种数据量下都没有增益 | — |

**如果只能做一件事：做 P0，并且用本轮的 E/F 双口径 + 同一把 `consecutive` 尺** —— 这样"候选分不开"这个假设
第一次被直接检验（而不是靠改分数的量纲去间接影响它）。

## 9. 测试

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_margin/pytest1 -q tests\test_v400_margin_resolution.py
# → 8 passed，exit 0
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_margin/pytest_reg2 -q `
  tests\test_v400_margin_resolution.py tests\test_v400_teacher.py tests\test_v400_labels_signal.py `
  tests\test_v400_d0_dual.py tests\test_v400_student.py
# → 188 passed, 2 skipped, 1 failed，exit 1（唯一红是既有 v2h 激光字段，EXP-P3-004 §11.7 已记录）
```

`tests/test_v400_margin_resolution.py` 的 8 个用例：

1. `test_v2i_is_v2h_with_the_margin_cap_raised_and_nothing_else`：`v2i` 折回 `margin_cap=16` 后与 `v2h`
   **逐字段相等**；horizon/beam/verify_k/max_iterations/bottom/high/laser/rule1/durations/discount 全部钉住。
2. `test_v2i_is_registered_and_selectable_by_both_clis`：`CONFIGS`/`PROFILES` 注册、`v400_d0_dual` 与
   `v400_teacher` 的 `--mode` choices 都可选（不改 CLI 也支持，因为 choices 就是教师注册表）。
3. `test_fast_v2a_and_v2h_keep_their_published_numbers`：`FAST`(32/24/1/1)、`V2A`(48/64/3/3 + 1.5/2.0)、
   `V2H`(laser 12@8)、`replace(V2H.weights, laser_penalty=0) == V2A.weights`、所有档 discount 5.0 逐位不变。
4. `test_the_margin_term_is_monotone_and_saturating_under_each_cap`：两种 cap 下 margin 项单调、封顶前逐点
   增量 = min(Δ, 剩余量程)、封顶后恒为 0（**封顶前后对比**）。
5. `test_raising_the_cap_changes_a_score_that_sees_an_uncapped_margin`：给它 41.175 px 时 `v2i−v2h = 25.175`。
6. `test_simulate_plan_reports_a_capped_margin_for_a_real_41_px_clearance`：真实 41.175 px 净空 → 16.0。
7. `test_v2i_and_v2h_score_every_candidate_identically_on_a_real_tape`：`beam_search` 与
   `beam_search_reference` 两条路径、18 路分数**逐位相同**（非退化断言 `len(plans)==18`）。
8. `test_the_raster_byte_budget_caps_a_real_raise_at_63_75_px`：`QUANT=4`、`MARGIN_Q_CAP=64`、
   上限 63.75 px、`64×4=256>255` ⇒ 真正提高 cap 必须改编码而不是只改权重。

## 10. 命令与退出码（可复现）

```bash
# WSL：v2i 8 片采集（2 worker，2 teacher，引擎 rl_laser）
wsl -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_margin/launch_batch.sh
# → exit 0（后台批处理；日志 .tmp_v400_margin/logs/batch_v2i.log；wall 2,012.8 s / 26,400 决策 / HASH 100 %）
```

```powershell
# Windows：D0 seeds 9–16 的 v2-同量口径（junction + 过滤 manifest）
python .tmp_v400_margin\make_d0_subset.py            # → exit 0（8 片 / 26,864 决策 / e9bdbae0…）
# 标签分辨率（新旧逐项）
python .tmp_v400_margin\analyze_labels.py .tmp_v400_margin\v2i .tmp_v400_p3sig\k1 .tmp_v400_p3sig\k3
# → exit 0（写 label_resolution.json）
# 结构性钳位实测
python .tmp_v400_margin\prove_clamp.py               # → exit 0（写 margin_clamp.json）
# 逐片加载门
python .tmp_v400_margin\verify_load.py .tmp_v400_margin\v2i   # → exit 0（写 load_gate.json）
```

```bash
# WSL：两臂训练（同配方；E 8 片 / F 27 片）
wsl -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_margin/run_train.sh
# → exit 0（TRAIN_EXITS armE=0 armF=0；wall 458.3 s / 513.2 s）
# WSL：S1N 5 局（consecutive 新尺，2 进程 = 2 引擎）
wsl -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_margin/launch_evals.sh
# → exit 0（EVALS DONE window=consecutive；EXIT armE rc=0 / EXIT armF rc=0）
```

```powershell
python .tmp_v400_margin\summarize_arms.py    # → exit 0（写 arm_table.json）
python .tmp_v400_margin\summarize_eval.py    # → exit 0（写 eval_table.json）
python .tmp_v400_margin\s1n_effects.py       # → exit 0（写 s1n_effects.json）
```

## 11. 变更与产物

| 文件 | 变更 |
|---|---|
| `touhou_rl/v400/teacher/config.py` | **新增** `V2I`（`v2i` = `dataclasses.replace(_V2H_WEIGHTS, margin_cap=64.0)` + v2h 的 `horizon/beam/verify_k/max_iterations/bomb_rule1_max_death_frame`），注册进 `CONFIGS` 与 `PROFILES`；`FAST`/`V2A`/`V2H`/`FAST3` 未改（测试钉住逐字段相等） |
| `tools/rl/v400_teacher.py` | **未改**：`--mode` 的 `choices` 早已是教师注册表 `tuple(sorted(CONFIGS))`，"mode 支持"零改动即成立 |
| `tools/rl/v400_student_train_teacher.py` | **未改**：`--d0-root` / `--extra-root`（可重复）已能满足两种口径 |
| `tests/test_v400_margin_resolution.py` | **新增** 8 用例（§9） |
| `tests/test_v400_teacher.py` | 1 处**语义保持**更新：`test_v2f_profile_is_v2a_plus_the_top_band_and_keeps_high_penalty` 里"未注册名报错"的守卫由 `v2i` 移到 `v2j`（`v2i` 本轮注册了），并加断言 `get_config("v2i").mode == "v2i"`；**没有**放松任何既有断言 |
| `Document/Update/Alpha/v4.0.0/experiments/EXP-P3-005-margin-resolution.md` | 本卡 |
| `.tmp_v400_margin/*` | `v2i/`（8 片，669 MB）、`d0_916/`（D0 seeds 9–16 的 junction root，26,864 决策）、`arms/`（E/F checkpoint + metrics + split）、`eval/`（E/F 两个评测目录）、`logs/`（采集/训练/评测日志）、`batch.py`、`smoke_batch.py`、`make_d0_subset.py`、`analyze_labels.py`、`prove_clamp.py`、`verify_load.py`、`summarize_arms.py`、`summarize_eval.py`、`s1n_effects.py`、`launch_batch.sh`、`launch_evals.sh`、`run_train.sh`、`run_eval.sh`、`run_evals.sh`、`stop_batch.sh`、`engine_audit.sh`、`ps_inventory.sh`、`train_status.sh`、`label_resolution.json`、`margin_clamp.json`、`load_gate.json`、`arm_table.json`、`eval_table.json`、`s1n_effects.json`、`progress.json` |

**未改**：引擎、协议（`gym/protocol.py`）、依赖、第三方、`build/`（`rl`/`rl_laser` 的 sha256 只读校验：
`591c521d…` / `1462829b…`）、`touhou_rl/v400/teacher/{search,raster,raster_numba}.py`（**因此 `margin_cap` 是惰性的**，
见 §1.2）、`touhou_rl/v400/student/*`。未删除任何文件，未提交。

## 12. 限制（不要过度解读）

1. **不是 G3、不是 PSN、不是 confirm split**；1 个 stage（stage 0 = 真 Stage 1）、1 个难度（normal）、n = 5 局。
2. **`margin_cap=64` 是惰性的**（§1.2）：本轮 S1N 的差异**不能**归因给 margin 分辨率。要归因必须先把
   `MARGIN_CAP` 贯通（非本轮允许范围）。
3. 8 片 / 27 片两臂的绝对帧数结构性不同（EXP-P3-004 §5.3 实测"数据量"本身值 ~550 帧），两臂分开报告、
   不合并。
4. `bomb` 在全套 D0/DAgger/fast3 里都是 0 正例，本轮的 bomb acc 同样是空指标（`identifiable=false`；
   F 的 val 有 20 个正例，但那是 DAgger 分片贡献的，不是 v2i）。
5. 采集是**按决策数截断**的（`stop_reason=decisions`，每片 3,300），不是自然换面；D0 的同种子自然结束在
   3,274–3,394 决策，所以截断只切掉尾部几十条决策，不改变分片身份。
6. 窗口尺只有一把：本卡所有 S1N 数字都是 `--obs-window consecutive`；`1,718.8` 也是同一把尺（EXP-P3-004 §6）。
   历史 `strided` 数字（2,399.6 / 1,084.0 / …）**不与本卡比较**。
7. 机器上同时有零售 trace 批跑与其它 WSL 任务，采集 wall 受负载影响；每片 496–510 s、总墙 2,012.8 s（2 worker）
   不是吞吐上限（`loadavg` 全程 1.5–2.0，机器 20 核）。
8. **`.tmp_v400_margin/v2i` 这 8 片、26,400 条标签的 `margin` 与 `v2h` 逐位相同，不应用于任何"提高了分辨率"的
   结论**（`margin_cap=64` 惰性，§1.2/§7.1）：它的 `>0.1` 1.20 %→2.22 % 来自连续项（激光/底带）与更深的搜索，
   与 margin 无关；该 root 只可作为"v2h 预算 + 激光项"的标签集复用（E/F 两臂的可复现证据），
   且**不得**与 D0 的 `dual-s0d1seed1..8` 混进同一 root 或同一次训练（分片键相同 ⇒ `duplicate shard key`、
   同一局跨 split 泄漏）。标记文件：`.tmp_v400_margin/v2i/LABELS-STATUS.md` / `.json`（截至 2026-11-10 00:22
   尚未被新实验复用，仅本卡的两个诊断脚本引用）。

> **更正注记（2026-11-10，EXP-P3-009）**：本卡的 `armE 1,209.6` / `armF 1,816.0` 是**单 checkpoint** 读数——可作参照行，**不可作阈值行**（EXP-P3-008 的 `2,116` 绝对门就是这样来的）。该 `armF`（`.tmp_v400_margin/arms/armF_v2i27.pt`）本轮 20 局重测为 **1,468.3 帧**、95% CI [1,326.3, 1,659.1]，**1,816.0 不在该区间内**（高 347.7 帧 / +23.7 %）——即这条锚点行本身有相当部分是 seed 1000–1004 的有利抽样；注意不要与 EXP-P3-007 的 `v2j/armF`（20 局 1,470.7、CI [1,286.8, 1,664.9]）混为一谈，两者是不同的 checkpoint。原始数字不改。详见 [EXP-P3-009](<EXP-P3-009-eval-power.md>)。
