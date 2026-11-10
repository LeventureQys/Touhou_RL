# EXP-P3-003 DAgger 首轮：在学生访问的状态上标注 + 聚合重训 + S1N 5 局

- 日期 / 切片：2026-11-09 / **P3-WP3-a**（DAgger K=1 首轮：β=0.2 采集、D0 聚合、同配方重训、S1N 验收门）
- 依赖：[EXP-P3-002](<EXP-P3-002-bc-v1-teacher.md>)（BC v1 = 1,084.0 帧 / 1 死 / 0 过；预注册配置在 §7）、
  [EXP-P1-018](<EXP-P1-018-D0-scaleup.md>)（D0：53,512 决策 / 16 片 / HASH 107,024-全等）、
  [EXP-P1-014](<EXP-P1-014-D0-dual-channel.md>)（shadow 出标签 + env 出像素 + HASH 对齐；shadow **没有渲染器**）、
  [EXP-P1-012](<EXP-P1-012-g1-evaluation.md>)（S1N 口径与教师 13,565.2 帧 / 20-20 过）
- 结论先说：**预注册门 PASS**。S1N 5 局均值 **2,399.6 帧**（v1 = 1,084.0，**2.21×**），
  过的是"≥ 2× v1（≥2,168）"这条腿；**没过**"≥ 教师 30 %（≥4,070）"那条腿
  （教师 13,565.2 的 **17.7 %**），仍然 **0/5 过关、5/5 首死**。
  最有意思的一条：**闭环翻倍不是靠离线精度**——离线 top-1 / macro recall 反而略降
  （0.596/0.212 → 0.591/0.191），提升来自**状态覆盖**（DAgger 数据）与**更少的"只会不动"**
  （动作 0 预测占比 69.5 % → 61.5 %），以及**教师真正有把握的状态上顺带学会了**
  （gap>5 的一致率 40.7 % → 49.9 %）。数据侧正面证据：教师 `gap=5.0` 地板价占比
  82.2 % → **55.6 %**，bomb 正例 0 → **203**。
- **根因（本轮最重要的发现，§8.3）**：D0 用 `--mode fast`，而 `fast` 的 **`verify_k=1`**
  ——每个决策只验证 1 个动作；`11-P1 §3` 又对未验证动作统一 **−5**。于是
  **`verified_mask[argmax]` 在 76,912 行标签上 100.00 % 为真、第 2 名 100.00 % 为假**，
  而真实几何分差 p90 只有 **0.059**（比 5.0 小 85 倍）：教师的 argmax 实际上是
  "**哪个动作被抽到验证**"，对状态不可预测。学生 82 % 的决策在学噪声尖峰 →
  macro recall 0.19 / 动作 0 占 69 % / focus 只输出 2/8，是同一个根因。
  下一轮应按 **`verify_k` 1→3（或 9）+ 几何分复原** 做 2×2 对照（§8.4），而不是加数据。
  本卡**不是 G3 通过、不是 PSN、不是 confirm split**。

## 1. 范围与预注册（照 EXP-P3-002 §7 执行，不改配置）

| 项 | 预注册值 | 本轮实际 | 说明 |
|---|---|---|---|
| 轮数 K | 1 | **1** | — |
| 新决策数 M | 50,000（≈92 局） | **23,400 决策 / 11 局** | **未达 M**，见 §4.3 的真实原因（教师侧不可标注状态），不是超时截断 |
| 执行策略 | β = 0.2（80 % 学生 + 20 % 教师） | β = 0.2，实测注入率 **20.16 %**（4,717/23,400） | 每决策固定概率、种子化抽样，逐行记录 |
| 新状态标签 | 100 % 教师软标签 `scores[18]` | 100 %（每决策都查教师） | 未标注状态直接结束该段，不入库 |
| 聚合 | 50,118 + 50,000 = 100,118 | **53,512（D0 全量 16 片）+ 23,400 = 76,912** | D0 本轮用满 16 片（v1 只用了 15 片快照） |
| 切分 | 整局 70/15/15，同种子不跨 split | 同（`split_seed=0`）：19/4/4 片，52,564 / 12,294 / 12,054 | 见 §5 |
| 采集种子 | 2000+ | **2000–2010** | 评测种子 1000–1004 未出现在任何一片 |
| 评测 | seeds 1000–1004，stage 0，cap 19,887，`--stop-on-death`，1 引擎 | 同 | 与 v1 同一把尺 |
| 重训 | 同配方（width 48 / hidden 256、τ=2.0、KL+BCE、≤15 min） | 同配方，**wall 528.6 s**（≤15 min） | `--taus 2.0 --train-seconds 480` |
| 验收门 | 均值 ≥2×v1（≥2,168）或 ≥教师 30 %（≥4,070） | **2,399.6 帧 → PASS（2×v1 腿）** | 见 §7 |

**先记一条最重要的偏差**：M 没到 50,000，原因是**学生访问到的状态里有教师无法标注的点**
（§4.3，`ERR player-speed-unavailable`）。按预注册"100 % 教师软标签"的规则，这些状态
不能假装有标签：采集在**该决策点**结束该局，前面的行（全部 HASH 对齐、全部有真标签）入库。
这条不是超时，也不是降低标准——它是 DAgger 路线在第 1,848–2,400 决策处撞到的**真实边界**。

## 2. 采集器设计（`tools/rl/v400_dagger_collect.py`）

### 2.1 为什么必须是"双引擎锁步"

shadow 角色**没有渲染器**（`EXP-P1-014 §2`：`main.cpp:210-244`、`d3d8_compat.cpp:634-648`，
shadow 是 CPU-only null device，`ReadBuffer` 全 0），而 `PARK` **只允许 shadow**
（`fork.cpp:261-269`，env 会 `ERR park-unsafe`）。因此：

- **像素只能来自 env 角色**；
- **教师 `decide()` 只能在 shadow 角色**（它要 PARK/SPAWN/CHILD 去录 hazard）；
- 而 DAgger 要求**标签打在学生走到的状态上**，即"两个角色必须踩同一条轨迹"。

于是每局是一个**锁步对**：同一 4 帧 mask 块同时喂给 env 与 shadow，每次决策前后各打一个
`HASH` 检查点，两侧必须逐点相等（复用 `v400_d0_dual.py` 的 `_checkpoint`/`_pair`/
`_transition_info_from_pairs`/`--trim-transition` 语义）。**任何一处不等即 fail-closed**，
分片不落盘。

### 2.2 每个决策做什么

1. env 侧 `HASH`（frame=4k）与 shadow 侧 `HASH` → **pre 检查点**，必须相等；
2. shadow 取观测？**不能**——obs-v1 的 `play64/hud` 由 env 的共享内存读回，
   `DecisionObservationBuffer.stack_for(4k)` 给出**连续 4 帧** `frames[4k-4..4k-1]`（与 D0 逐字节同构，
   行 0 是 `pad=3` 的 episode-start 补帧）；
3. 学生动作：BC v1 checkpoint 对**该 obs** 取 greedy `(cand, bomb)`；
4. 教师查询：`decide(engine_shadow, config_fast, committed=<上一决策实际执行的 slots>,
   frame=4k, record_source="child")` → `scores[18]`、`verified_mask[18]`、`bomb`、`margin`、
   `chosen_plan`；**这就是"教师在学生状态上的软标签"**；
5. 注入：`draw = rng(beta_seed+seed).random()`，`injected = draw < 0.2 且 教师有 chosen plan`。
   注入时执行**教师自己的 plan slots**（与其 CLI 一致：`chosen.slots[:replan_decisions]`），
   否则执行 `[学生 cand] × replan_decisions`；
6. 同一 `commit_block(...)` 同时喂两侧：env 逐帧 `STEP mask 1` + 逐帧读回像素，
   shadow `STEP mask 1` × 4；**post 检查点**必须相等（唯一例外见 §2.3）；
7. 写一行：学生/教师动作、注入标志、`draw`、`scores[18]`、mask、两侧 HASH、`frame_event_flags`。

### 2.3 换面语义（与 D0 一致）

`stage_changed` 时（4 帧 commit 跨过换面边界），post 状态是**下一面的 setup**，不是 stage-0 玩法。
此时若**恰好只有这一个检查点不等**且它就是本决策的 post，则按 `--trim-transition`
丢掉**该决策**、结束该段（`dual._transition_info_from_pairs` 证明这一点，证明不了就照旧 fail-closed）。
本轮所有分片的 `transition_trim` 均为 `null`：**没有一局走到换面**（都在 §4.3 的不可标注点停下）。

### 2.4 续跑与断点

- 分片键 `dagger-s<stage>d<difficulty>seed<seed>`，一局=一片=一个种子；
- 已完成的片由 `dual.shard_resume_report`（**重新用 `dataset.load_teacher_shard` 验一遍**）认定，
  续跑跳过；失败的一局**不写分片**（`TeacherObsShardWriter` 只在循环结束后创建，
  中途失败连 staging 目录都不留）；
- `progress.json` 每局落盘（决策数、注入数、dec/s、stop_reason）。

## 3. 采集结果（`.tmp_v400_dagger/d0_extra/`，11 片）

### 3.1 逐局

| shard (seed) | decisions | frames | injections | 注入率 | teacher_available | agreement | stop_reason |
|---|---:|---:|---:|---:|---:|---:|---|
| 2000 | 1,872 | 7,488 | 380 | 20.30 % | 1,872 | 777 (41.5 %) | teacher_unavailable |
| 2001 | 2,064 | 8,256 | 393 | 19.04 % | 2,064 | 1,035 (50.1 %) | teacher_unavailable |
| 2002 | 2,160 | 8,640 | 441 | 20.42 % | 2,160 | 722 (33.4 %) | teacher_unavailable |
| 2003 | 2,400 | 9,600 | 482 | 20.08 % | 2,400 | 793 (33.0 %) | teacher_unavailable |
| 2004 | 1,848 | 7,392 | 396 | 21.43 % | 1,848 | 770 (41.7 %) | teacher_unavailable |
| 2005 | 1,896 | 7,584 | 372 | 19.62 % | 1,896 | 969 (51.1 %) | teacher_unavailable |
| 2006 | 2,136 | 8,544 | 409 | 19.15 % | 2,136 | 990 (46.3 %) | teacher_unavailable |
| 2007 | 2,448 | 9,792 | 513 | 20.96 % | 2,448 | 925 (37.8 %) | teacher_unavailable |
| 2008 | 2,088 | 8,352 | 408 | 19.54 % | 2,088 | 951 (45.5 %) | teacher_unavailable |
| 2009 | 2,376 | 9,504 | 477 | 20.08 % | 2,376 | 987 (41.5 %) | teacher_unavailable |
| 2010 | 2,112 | 8,448 | 446 | 21.12 % | 2,112 | 817 (38.7 %) | teacher_unavailable |
| **合计** | **23,400** | **93,600** | **4,717** | **20.16 %** | **23,400** | **9,736 (41.6 %)** | 11× teacher_unavailable |

- 总计 **23,400 决策 / 93,600 帧**；注入率逐局 19.0–21.4 %（合计 **20.16 %**），
  与 β=0.2 的 23,400 次二项噪声（sd 0.26 pp）一致；
- `agreement` = 教师 chosen 的 slot0 与学生 greedy 相同的比例，逐局 **33–51 %**；
  D0 上 BC v1 的离线 top-1 是 59.6 %——**在自己造的状态上一致率明显更低**，这正是 DAgger 要补的部分；
- **0 死亡、0 换面**：所有段都在"教师不可标注"处结束（§4.3），不是学生死亡；
- 采集墙钟：两轮共 ~2,970 s（第一轮 850 s / 10,344 决策，第二轮 3,000 s 预算，
  中途在 seed 2011 的换面边界 fail-closed 停下，见 §4.4），稳态 **12.16 决策/s**（≈ 0.082 s/决策，
  含教师 `decide()` ≈ 30 ms + env 4 帧像素读回 + 4 个 HASH/STATE 往返）。

### 3.2 HASH 对齐（fail-closed 的原始证据）

- 每局 `boundaries = 2 × decisions`，`checkpoints_equal == boundaries`，
  `checkpoints_mismatched == 0`；合计 **46,800 个检查点、0 个不等**；
- `render_zero_frames = 0`（每帧都真渲染过：事前 `require_rendered` 闸门 + 事后 `zero_frames` 双证据）；
- 分片 schema 与 D0 **完全一致**：`v400-teacher-obs-shard-1`、`frames_per_decision=4`、
  `obs/play64.npy (N,4,64,64) uint8` + `obs/hud.npy (N,56,160) uint8`、
  **25,344 B/决策**、`observation_time=decision_time_before_commit`、`p3_window_last_row=-1`、
  行 0 `pad=3`（episode-start 补帧）、obs-v1 摘要 `24f0ddac…`（与 D0 同一个 `24f0ddace591049…`）；
- 被冻结的 D0 加载器**原样加载**（`load_teacher_samples(require_observations=True)`）：
  **11 片 / 23,400 条 / 0 unresolved / `scores[18]` + `verified_mask[18]`**，`--phase verify` exit 0。

### 3.3 教师的"建议"在学生状态上确实变了（本轮最硬的正面证据）

同一把尺（`--mode fast`、stage 0 / normal）下比较**教师 argmax 的分布**：

| 指标 | D0（教师自己的轨迹，53,512） | **DAgger（学生的轨迹，23,400）** |
|---|---:|---:|
| 动作 0（不动）占比 | 43.98 % | **39.86 %** |
| focus 类（10–17）占比 | 16.69 % | **3.55 %** |
| 动作 3 占比 | 10.00 % | **17.44 %** |
| 动作 7 占比 | 7.04 % | **22.74 %** |
| `top1−top2` 恰好 = 5.0（生存项地板价） | **82.19 %** | **55.55 %** |
| `top1−top2` 均值 | 5.014 | 5.046 |
| `verified_count` 均值 | 1.00 | 1.00（0 个零验证） |
| **bomb 正例** | **0** | **203**（0.87 %） |

三点读法：
1. **教师不再"近似平局"**：在地板价的决策从 82 % 掉到 56 %，说明学生在自己造的状态上
   把教师逼进了"有真实偏好"的区域——这正是 DAgger 想采的样本；
2. **bomb 头第一次有正例**：D0 里 0 个 bomb 正例（bomb 指标是空指标），DAgger 里有 203 个
   （0.87 %），训练集 pos_weight = **100.0（clamp 到上限，`identifiable=true`）**，
   v1 那个"永远预测 0"的空指标第一次变成真指标；
3. **focus 类反而更少**：学生轨迹上教师更少要求 focus/shoot 变化（16.7 % → 3.6 %），
   这解释了为什么"离线 macro recall 提升"不必然等于"闭环会躲"（§8），
   也解释了为什么本轮 v2 的 focus 覆盖仍然是 **2/8**。

## 4. 采集的两个真实限制（不要过度解读）

### 4.1 shadow 与 env 必须同 build

两侧都用 D0 同一份 `build/rl_hazard2/th08-modern`（sha256 `f172399f…`），
所以"HASH 相等"证明的是**同一条轨迹**，不是同一份像素渲染器。训练数据的像素来自
`rl_hazard2`（与 D0 一致），而 S1N 评测用 `build/rl/th08-modern`（sha256 `591c521d…`，
与 EXP-P3-002 一致）——**这条跨 build 的事实与 v1 完全相同**，不是本轮引入的。

### 4.2 一局只有 1.8k–2.4k 决策（不是 ~3.3k）

D0 的一局是 3,370–3,440 决策（走到换面）。本轮全部在 **1,848–2,448 决策**处结束。

### 4.3 为什么结束：教师侧 `ERR player-speed-unavailable`

```
ChildHazardsUnavailable: child recording batch failed after 96 commands:
'CHILD 1 PLAYER' -> ERR player-speed-unavailable primary_sht=0 secondary_sht=0
```

**11 局全部**在 1,848–2,448 决策之间（frame 7,392–9,792）撞到同一个状态。读法：

- 教师自己的轨迹（D0）**从不**进入这个状态；学生（β=0.2 混合）会进入——
  这本身就是"分布偏移"的直接证据；
- 在这个状态上 `PLAYER` 的 player-speed 字段不可读（`primary_sht=secondary_sht=0`）→
  教师的 hazard 录制整批失败 → **该状态无法标注**；
- 按预注册"新状态 100 % 教师软标签"，本轮**不猜标签**：结束该段、保留已标注行。
  这条也解释了 M 为什么没到 50,000（每局少 ~1,000–1,400 决策，而 92 局的目标本身是按
  3,300 决策/局估的）。

### 4.4 seed 2011：换面边界的 fail-closed（本轮唯一一次）

第二轮跑到 seed 2011 时，这一局**没有**在 §4.3 的可标注点停下，而是活到 frame 12,908
（≈ stage-0 的 13,258 帧，几乎打到换面）。它在 `decision 3226 post` 出现**一个** HASH 不等：

```
AlignError: decision 3226 post: shadow=e1abc55136e02854 env=ef5e27e5533b099c at frame 12908
```

这正是 D0 记录的"换面 commit 的 post 状态是下一面的 setup"现象，但本工具的 `--trim-transition`
**要求 env 侧 `STATE.stage` 已推进**才认这条排除（`dual._transition_info_from_pairs` 的
`stop_reason=stage_changed` 前提）。这个状态的 stage 计数还没变，所以**排除条件不成立**：

- 按 fail-closed 处理：**该局不写分片**（连 staging 目录都不留），3,227 条决策被丢弃；
- 采集器因此停在 seed 2011，M 停在 23,400；
- 顺带一个正面信号：**β=0.2 的混合策略在 seed 2011 上活到了 12,908 帧**
  （= v1 的 11.9×、教师的 95 %）——说明"活不到中段"这个问题在混合策略上确实被缓解了，
  但这**不是**学生策略单独的成绩（见 §12）。

## 5. 聚合与重训（`checkpoint-v2.pt`）

聚合规则：**D0 全量 16 片 + DAgger 11 片**，按 `shard` 排序 + `split_seed=0` 确定性置换，
70/15/15 最大余数分配，`check_episode_isolation` 反查 seed→split 唯一。

| split | shards | seeds | 决策数 | 占比 | 其中 DAgger |
|---|---:|---|---:|---:|---:|
| train | 19 | 1,2,3,4,5,6,8,12,15,16 + 2000,2002,2003,2004,2005,2006,2007,2009,2010 | **52,564** | 68.3 % | 9 片 / 18,864 |
| val | 4 | 9, 13, 14 + **2008** | **12,294** | 16.0 % | 1 片 / 2,088 |
| test | 4 | 7, 10, 11 + **2001** | **12,054** | 15.7 % | 1 片 / 2,064 |
| 合计 | 27 | 1–16 + 2000–2010 | **76,912** | 100 % | 11 片 / 23,400 |

- 切分签名 `19005ff22256277e7bebb2ad835237d6ed4e339c85a3ca6a2989aeea01d90158`（写进 checkpoint 的
  `provenance.split_signature`）；`check_episode_isolation` 未报错：**没有任何 seed 跨 split**；
- DAgger 的 11 片被分到 9/1/1（train/val/test）。这是 `split_seed=0` 下的确定性结果，
  **没有为了把 DAgger 塞进 train 而改 split_seed**（改就是按结果挑切分）；
- 训练：同一配方（width 48 / hidden 256、**τ=2.0**（预注册值，不再扫 τ）、KL(π_t‖π_s) + bomb BCE、
  AdamW lr 2e-3 / wd 1e-4 / batch 256）；
- 离线指标（下面）用**整 split 全量**，不是 2,048 子集。

| 项 | 值 |
|---|---|
| 设备 | RTX 4070 SUPER（Windows 原生 CUDA） |
| 预算 / 实测 | `--train-seconds 480`（τ 探针 30 s）；**wall 528.6 s = 8.8 min**（≤15 min），`aborted_at_deadline=true`（跑满预算） |
| 步数 / 样本 | **27,748 step / 7,103,488 样本**（= 52,564 决策的 135 epoch），13.4 k 样本/s |
| checkpoint | `.tmp_v400_dagger/checkpoint-v2.pt`，842,391 B，204,260 参数（与 v1 同构） |
| 选模 | best-by-val-top1：**step 8,900**（v1 是 2,200）——收敛点晚了 4×，与"数据更多、也更难"一致 |

| split | 决策数 | top-1 | top-3 | KL(π_t‖π_s) | NLL(argmax) | H(π_t) | macro recall | 多数类基线 | bomb acc | bomb 正例 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 52,564 | 0.7066 | 0.8470 | 0.2529 | 1.5456 | 2.2546 | 0.2491 | 0.4470 | 0.9998 | 156 |
| **val** | 12,294 | **0.5912** | 0.7483 | **0.3473** | 1.8266 | 2.2642 | **0.1912** | 0.4390 | 0.9984 | 20 |
| **test** | 12,054 | **0.5766** | 0.7427 | **0.3542** | 1.8453 | 2.2653 | **0.1887** | 0.4306 | 0.9974 | 27 |

（对照 v1：val top-1 0.5960 / KL 0.3623 / macroR 0.2121；test 0.5826 / 0.3792 / 0.2085。
**v2 的 top-1 与 macro recall 都略降，KL 略好**——这正是 §8 要解释的"闭环涨、离线不涨"。）

## 6. S1N 5 局评测（stage 0 = 真 Stage 1，normal，cap 19,887，`--stop-on-death`）

引擎 `build/rl/th08-modern`（sha256 `591c521d…`），`xvfb-run`，**单引擎**，seeds 1000–1004
（D0 / DAgger 都未出现过）。

| seed | frames | deaths | cleared | score | end | exit | stage | wall s |
|---:|---:|---:|---|---:|---|---|---|---:|
| 1000 | **3,636** | 1.0 | False | 98,086 | death_stop | controller | 0→0 | 66.0 |
| 1001 | **1,866** | 1.0 | False | 32,623 | death_stop | controller | 0→0 | 36.5 |
| 1002 | **1,790** | 1.0 | False | 25,170 | death_stop | controller | 0→0 | 36.0 |
| 1003 | **1,580** | 1.0 | False | 40,705 | death_stop | controller | 0→0 | 37.7 |
| 1004 | **3,126** | 1.0 | False | 49,727 | death_stop | controller | 0→0 | 62.4 |
| **均值** | **2,399.6** | **1.0** | **0/5** | **49,262.2** | death_stop ×5 | controller ×5 | 0→0 | 47.7 |

- 中位数 1,866 帧，min 1,580 / max 3,636；5/5 首死即停，0/5 过关（`deaths` 恒为 1 是 `--stop-on-death` 的产物）；
- **均值 2,399.6 vs v1 的 1,084.0 = 2.21×；均值 score 49,262.2 vs 17,074.6 = 2.89×**；
- 与教师（13,565.2 / 0 死 / 20-20 过）= **17.7 %** 帧数，仍是"能不能活过前 18 % 面"的差距。

### 6.1 逐局首死帧（同一把尺）

| seed | v1 首死帧 | **v2 首死帧** | 变化 |
|---:|---:|---:|---:|
| 1000 | 740 | **3,636** | 4.91× |
| 1001 | 1,124 | **1,866** | 1.66× |
| 1002 | 1,330 | **1,790** | 1.35× |
| 1003 | 1,122 | **1,580** | 1.41× |
| 1004 | 1,104 | **3,126** | 2.83× |
| 均值 | 1,084.0 | **2,399.6** | **2.21×** |

比 v1 单调变好的是 4/5 个种子（1000、1004 提升最大），没有种子变差。

## 7. 是否过预注册门：**PASS（2×v1 那条腿）**

| 判据 | 门槛 | 实测 | 结果 |
|---|---:|---:|---|
| ≥ 2 × v1 | 2,168 帧 | **2,399.6** | ✅ **PASS**（+10.7 % 余量） |
| ≥ 教师 30 % | 4,069.6 帧 | 2,399.6 | ❌ 未达（17.7 %） |
| 与 v1 同口径（1,084.0） | — | 2,399.6 | **2.21×** |
| 与教师同口径（13,565.2） | — | 2,399.6 | 17.7 % |

**判定**：按预注册规则（"≥2×v1 **或** ≥30 % 教师"）本轮**过门**，不需要记 null。
但要如实标注三条：

1. **过的是较低的那条腿**：2,168 帧门槛只比 v1 高 2 倍，而教师是 13,565 帧；"过门"不等于
   "接近教师"，仍是 **0/5 过关 / 5/5 首死**；
2. **n=5**：2,399.6 的 5 局 sd ≈ 950 帧（se ≈ 425），门槛 2,168 与实测 2,399.6 的距离约 0.55 se，
   **这条 PASS 在统计上是脆的**——换 5 个种子可能翻转。本卡不把 PASS 写成"DAgger 已被证明有效"，
   只写"本轮 5 局过了预注册门"；
3. **不是 G3 / 不是 PSN / 不是 confirm split**：一个 stage、一个难度、评测种子 1000–1004。

## 8. 诊断：闭环翻倍，离线却没涨——为什么

### 8.1 三个指标，三种方向

| 指标（val / test） | v1 | **v2** | 方向 |
|---|---:|---:|---|
| top-1 | 0.5960 / 0.5826 | 0.5912 / 0.5766 | **略降**（−0.5 / −0.6 pp） |
| macro recall | 0.2121 / 0.2085 | 0.1912 / 0.1887 | **降**（−2.1 / −2.0 pp） |
| KL(π_t‖π_s) | 0.3623 / 0.3792 | 0.3473 / 0.3542 | **改善**（−0.015 / −0.025） |
| 动作 0 预测占比 | **69.5 %**（val） | **61.5 %**（val） | **塌缩缓解** |
| focus 类被输出 | 2/8（11、12） | 2/8（11、12） | 不变 |
| 教师有把握时（gap>5）一致率 | 0.4072 | **0.4988** | **+9.2 pp** |
| 教师平局时（gap≤5）一致率 | 0.6353 | 0.6140 | −2.1 pp |

**结论**：闭环 1,084 → 2,400 **不是**靠"离线更准"。真正动的是两件事：

1. **状态覆盖**：训练集第一次包含"学生自己会走到的状态"（23,400 条，30 % 的聚合量），
   其中教师**有真实偏好**（gap>5）的比例从 17.8 % 升到 44.4 %。学生在这类状态上的一致率
   从 40.7 % 提到 **49.9 %**——而这恰恰是闭环里"要不要变向"的关键少数；
2. **模式塌缩缓解**：动作 0 的预测占比 69.5 % → 61.5 %（教师 43.98 %），
   动作 3 的预测数 344 → 1,931、动作 7 的 307 → 1,211（val），与教师**在 DAgger 状态上**
   的偏好（动作 3：17.4 %、动作 7：22.7 %）对齐。闭环里"只会不动"的 1,084 帧因此变成
   "会往右/右上偏"，首死帧翻倍。

### 8.2 没解决的部分（下一轮的目标）

1. **focus 覆盖仍是 2/8**：8 个 focus 类里只有 11、12 被输出过；DAgger 状态下教师对 focus 的
   要求本来就少（3.55 %），所以**这一轮不可能学到**——无权重 KL 下 3.55 % 的类依旧是多数类的
   牺牲品（v1 的诊断 §6.2 就指出过）。这直接指向**类平衡 KL**；
2. **macro recall 反而降了**：因为 DAgger 数据的多数类更平（动作 0 只占 39.9 %），
   模型把概率质量重新分配给了 3/7，代价是稀有类（1、4、5、8、13–17）更少被碰；
3. **首死点仍在 1,580–3,636 帧**：还不到教师（13,565）的 27 %。§4.3 说明**学生自己会进入
   教师无法标注的状态**——这不是靠多采几局能解决的，是"教师的可观测性边界"；
4. **训练/评测的观测窗口不一致（本轮新发现，v1 也有）**：训练（D0 与 DAgger 一样）用的是
   **连续 4 帧** `frames[d-4..d-1]`（`DecisionObservationBuffer.stack_for`，一个决策 = 4 帧）；
   而 `tools/rl/v400_student_eval.py:330-374` 每个决策只 `STEP mask 2` 并 push **一次**渲染，
   `obs_spec.ObservationBuffer` 于是把这 4 个渲染按**决策间隔**堆起来 —— 实际喂给网络的是
   **相隔 2 帧的 4 帧、跨度 8 帧**。也就是说 v1 的 1,084 与 v2 的 2,399.6 **都是在观测错配下
   测的**，训练分布（连续 4 帧）和闭环输入（隔帧 4 帧）不是同一个东西。
   这条不改变本轮的对比（v1/v2 同尺），但它给下一轮多了一个**零成本**的候选：
   让 eval 的 buffer 也按"连续 4 帧"喂（或按 4 帧/决策跑闭环），再测一次同两个 checkpoint。

### 8.3 根因：`verify_k=1` + 5.0 折扣把标签变成了"猜哪个动作被抽到验证"

上面的"macro recall 不涨 / focus 学不出"不是容量问题，也不是数据量问题。在 D0 + DAgger 的
**全部 76,912 行标签**上直接统计（`.tmp_v400_dagger/verify-discount.json`、`geo-scores.json`）：

| 证据 | D0（53,512） | DAgger（23,400） |
|---|---:|---:|
| `verified_count`（每决策被引擎验证的动作数） | **恒为 1**（min=max=mean=1） | **恒为 1** |
| `verified_mask[argmax] == True` | **100.00 %** | **100.00 %** |
| `verified_mask[第 2 名] == True` | **0.00 %** | **0.00 %** |
| `top1−top2 == 5.0` 的比例 | 82.19 % | 55.55 % |
| 其中 argmax 已验证 / 第 2 名未验证 | **100 % / 100 %** | **100 % / 100 %** |
| 把 5.0 折扣加回去后 `top1−top2 == 0` | **82.19 %** | 55.55 % |
| 真实几何 `top1−top2` 均值 / p90 | **0.0143 / 0.0589** | 0.0459 / 0.0976 |
| 真实几何 `top1−top2 > 0.1` 的比例 | **1.19 %** | 3.16 % |
| 真实几何 `top1−top2 > 1.0` 的比例 | **0.10 %** | 0.35 % |

**读法**：`11-P1 §3` 的评分规则是"验证过的用验证后的分；**没验证的用几何分减 5**"。
D0 用的是 `--mode fast`，而 `touhou_rl/v400/teacher/config.py:277` 的
`FAST = TeacherConfig(..., verify_k=1)` —— **每个决策只验证 1 个动作**。于是：

1. 18 个动作里恰好 1 个拿到"未减 5"的分，其余 17 个统一 −5；
2. 18 个动作的**真实几何分差**（p90 = 0.059）比 5.0 的折扣小 **~85 倍**；
3. 所以 **argmax 100 % 就是"被抽到验证的那个动作"**——它由 beam 搜索的预算/顺序决定，
   对状态**不可预测**；
4. softmax(scores/τ=2) 于是给出"42 % 押那个动作 + 3.4 % × 17"的目标，学生只能在
   82 % 的决策上学一个**噪声尖峰**，最后退化成"押多数类"（动作 0 = 69 % 预测、
   macro recall 0.19、8 个 focus 类只输出 2 个）——**这三件事是同一个根因**。

这也解释了本轮唯一反常的正面数据：DAgger 状态下 `gap==5.0` 从 82 % 掉到 **56 %**，
即 44 % 的决策**真实几何分差超过了折扣**，教师第一次"有真偏好"；学生在这类状态上的
一致率因此从 40.7 % 涨到 **49.9 %**，闭环翻倍。**信号来自那 44 %，噪声来自那 56 %。**

### 8.4 下一轮最小改动（按这个根因重排）

| 优先级 | 改动 | 依据 | 成本 |
|---|---|---|---|
| **P0** | **`verify_k` 1 → 3（或 9）**：用 `--mode v2a` 起的配置（`verify_k=3`），或给 `fast` 加一个 `verify_k=3` 变体 | §8.3：只有 1/18 个动作无折扣，5.0 折扣是真实分差的 85 倍；`verify_k≥3` 才有"验证过 vs 验证过"的真实比较 | 教师 verify 阶段实测 4.5 ms（k=1）→ k=3 约 +9 ms/决策，30 ms → ~40 ms/决策，仍 ≥25 决策/s |
| **P0'** | **把折扣加回去（几何分复原）**：`scores_geo = scores + 5·(~verified_mask)`，softmax 用几何分 | 5.0 是常数，复原**精确**（示例：`119.958` vs `114.958+5 == 119.958`） | 零成本，一行；但它会把 82 % 的 D0 目标**抹平为均匀分布**（无信号），所以只能与 P0 一起做，做 A/B 对照 |
| P1 | **修 §4.3 的可标注性**（`PLAYER` 缺 speed → 教师退化到无速度录制） | 11/11 局卡在同一状态，M 卡在 23,400 | 引擎/协议侧，需单独立项 |
| P2 | **类平衡 KL**（`w(a) ∝ 1/p_teacher(a)`） | macroR 反降 2 pp、focus 2/8 | 一行 loss，但**在 P0 之前做会掩盖根因** |
| P3 | β 0.2 → 0.35 / K=2 / obs-v2 | seed 2011 混合策略活到 12,908 帧 | 都要在 P0/P1 之后才有意义 |

**本轮推荐**：下一轮（EXP-P3-004）做 **P0 + P0'** 的 2×2 对照（verify_k ∈ {1,3} × 几何复原 ∈ {否,是}，
共 4 个臂，同数据/同 seed/同 S1N 尺），这是唯一能把"标签结构"与"数据/容量"分开的设计；
类平衡 KL 与 obs-v2 都排在它后面——在标签本身是噪声尖峰的情况下，修 loss 只会更精细地拟合噪声。

**不建议**现在做方案 C（pairwise ranking loss）：真实几何分差 p50 = 0（82 % 的决策 18 个动作
几何分**完全相等**），ranking loss 在这些状态上没有可排的东西，只会把浮点噪声当信号。

## 9. 测试

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_dagger/pytest -q tests/test_v400_dagger.py
# → 14 passed in 0.30s，exit 0（+ tests/test_v400_student.py 26 = 40 passed）
```

`tests/test_v400_dagger.py` 的 14 个用例（连同 `tests/test_v400_student.py` 的 26 个 = 40）：

1. `test_injection_rate_matches_beta_and_is_seeded`：100k 次抽样的实测注入率与 β=0.2 相差 < 0.6 pp；
   同种子 RNG 逐值相同。
2. `test_injection_edges_and_no_teacher_action`：β=0/1 的边界、`teacher_cand=None` **永不相交**
   （没有教师动作就不注入，不假装有 β）、β∉[0,1] 与 draw∉[0,1) 报错。
3. `test_align_pair_requires_hash_and_frame_agreement`：同 hash 但帧号不同 → `equal=False`；
   缺字段 → `AlignError`。
4. `test_require_aligned_is_fail_closed_and_counts_boundaries`：一个不等即 `AlignError`，
   且证据行**不被改写**。
5. `test_stage_clear_trim_needs_the_documented_single_boundary`：只有"最后一个 post + 教师确认换面"
   能触发删除；其它单点不等、或缺教师裁决 → 不 trim（fail-closed）。
6. `test_collector_shard_loads_through_the_d0_loader`：分片被**冻结的 D0 加载器**加载，
   `scores[18]`/`verified_mask[18]`/obs 形状/`dagger.round` 全对。
7. `test_schema_rejects_a_row_with_a_broken_label`：`scores` 截成 17 维 → 加载器拒绝。
8. `test_shard_writer_refuses_a_second_shard_with_the_same_key`：同键第二片拒绝（不覆盖）。
9. `test_episode_plan_never_overlaps_the_evaluation_seeds`：2000+ 计划不碰 1000–1004；
   若把 `--seed-start` 设成 1000 会被标出 `eval_overlap`。
10. `test_count_roots_adds_up_and_refuses_duplicate_keys`：两 root 计数相加；同键跨 root → 报错；
    缺 manifest → 报错。
11. `test_split_assignment_is_episode_isolated_across_roots`：22 片（D0+DAgger）同一 seed 不跨 split、
    总数不变、与顺序无关、签名稳定。
12. `test_split_seed_collisions_detects_a_cross_root_duplicate`：跨 root 同种子检测。
13. `test_frames_per_decision_matches_the_d0_schema`：`fast` 档的
    `replan_decisions × decision_frames == 4`（schema 的 4 帧/决策）。
14. `test_shard_key_shape_and_eval_seed_guard`：分片键形状与默认 β。

## 10. 命令与退出码（可复现）

```powershell
# Windows：测试
cd D:\WorkShop\Touhou_RL; $env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_dagger/pytest -q tests/test_v400_dagger.py
# → 14 passed in 0.30s，exit 0（+ tests/test_v400_student.py 26 = 40 passed）
```

```bash
# WSL：DAgger 采集（锁步 env+shadow，最多 2 个引擎；两轮，第二轮续跑）
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_dagger/run_collect.sh \
  /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p3v1/checkpoint-v1.pt \
  /mnt/d/WorkShop/Touhou_RL/.tmp_v400_dagger/d0_extra 700
# → exit 0（5 局 / 10,344 决策，850 s；"wall budget 700s reached" 后正常退出）

wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_dagger/run_collect.sh \
  /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p3v1/checkpoint-v1.pt \
  /mnt/d/WorkShop/Touhou_RL/.tmp_v400_dagger/d0_extra 3000
# → exit 1（续跑 +6 局到 11 局 / 23,400 决策；在 seed 2011 的换面边界 fail-closed 停止，§4.4）
#   这个 1 是**预期内的 fail-closed**：那一局不写分片，已完成的 11 片不受影响

# WSL：分片自检（冻结加载器 + obs 摘要 + 逐片计数）
wsl -e bash -c "cd /mnt/d/WorkShop/Touhou_RL && PYTHONPATH=/mnt/d/WorkShop/Touhou_RL \
  /root/v400-venv/bin/python tools/rl/v400_dagger_collect.py --phase verify \
  --out-dir .tmp_v400_dagger/d0_extra --work-dir .tmp_v400_dagger/work"
# → exit 0（11 片 / 23,400 条 / 0 unresolved）
```

```powershell
# Windows（GPU）：聚合 + 同配方重训（τ=2.0 是预注册值，所以只跑一个 τ 候选）
python tools\rl\v400_student_train_teacher.py --d0-root .tmp_v400_d0/d0 `
  --extra-root .tmp_v400_dagger/d0_extra `
  --checkpoint .tmp_v400_dagger/checkpoint-v2.pt --metrics .tmp_v400_dagger/train-metrics.json `
  --split-json .tmp_v400_dagger/split.json --taus 2.0 `
  --sweep-seconds 30 --train-seconds 480 --max-total-seconds 900 `
  --eval-every 100 --val-eval-samples 2048 --batch-size 256 --width 48 --hidden 256 --seed 0
# → exit 0（wall 528.6 s）
```

```bash
# WSL：S1N 5 局
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_dagger/run_eval.sh \
  /mnt/d/WorkShop/Touhou_RL/.tmp_v400_dagger/checkpoint-v2.pt \
  /mnt/d/WorkShop/Touhou_RL/.tmp_v400_dagger/eval-v2
# → exit 0（5 局，47.7 s/局）
```

## 11. 变更与产物

| 文件 | 变更 |
|---|---|
| `tools/rl/v400_dagger_collect.py` | **新增**：DAgger 收集器（锁步 env+shadow 双引擎、β 注入、逐决策教师软标签、HASH 双检查点 fail-closed、D0 同构分片、续跑、`--phase run/verify/plan`、`--trim-transition` 换面语义） |
| `tools/rl/v400_student_train_teacher.py` | 聚合参数：`--extra-root`（可重复）+ 多 root 加载/计数/去重，切分仍按整局 |
| `tests/test_v400_dagger.py` | **新增** 14 用例（注入比例、HASH fail-closed、schema 兼容、切分无泄漏、聚合计数） |
| `Document/Update/Alpha/v4.0.0/experiments/EXP-P3-003-dagger-round1.md` | 本卡 |
| `.tmp_v400_dagger/*` | 采集产物（`d0_extra/` 分片 + `work/` 逐局记录 + `logs/`）、`checkpoint-v2.pt`、`split.json`、`train-metrics.json`、`diagnostics.json`、`label-stats.json`、**`verify-discount.json`**、**`geo-scores.json`**、`eval-v2/`、`eval-summary.json`、脚本（`analyze.py`/`label_stats.py`/`verify_discount.py`/`geo_gap.py`/`summarize_eval.py`/`show_shard.py`/`show_verify.py`/`run_collect.sh`/`run_eval.sh`） |

未改：引擎、协议、依赖、第三方、`build/`；未删除任何文件；未提交。
`touhou_rl/v400/student/*` 本轮**不需要改动**（DAgger 复用了既有的 obs-v1 编码、`TeacherObsShardWriter`
与 `decide()`）。

## 12. 限制（不要过度解读）

1. **不是 G3、不是 PSN、不是 confirm split**；5 局、1 个 stage、1 个难度、n=5 不构成策略推断。
2. **M 远小于预注册的 50,000**（23,400）：原因是 §4.3 的不可标注状态，不是调参；因此本轮的
   "聚合 1:1"也不再成立（D0 53,512 : DAgger 23,400）。
3. **PASS 是脆的**：2,399.6 vs 门槛 2,168，5 局 sd ≈ 950（se ≈ 425），差 0.55 se；
   换一组种子可能翻转。本卡只写"本轮 5 局过门"。
4. **§8.3 的根因是统计事实，不是推测**：它来自 76,912 行标签的全量统计（`verified_count`、
   `verified_mask[argmax]`、折扣复原后的几何差），但"改了 verify_k 就会变好"是**假设**，
   必须由下一轮的 2×2 对照验证。
5. **β 注入用的是教师的 slot 执行**，与"β 只对当前决策的一个动作"在语义上略有差别
   （执行 4 帧 = 2 个 slot）；两者都被记录（`dagger.teacher_slots` / `executed_slots`）。
6. **教师不可标注 ≠ 教师不会**：`ERR player-speed-unavailable` 是协议侧字段缺失，
   下一轮可以让教师在该状态退化到"无 speed 录制"的保守路径，而不是放弃该状态。
7. **闭环只有 5 局**，且 `--stop-on-death` 让 deaths 恒为 1；能比的是 frames / score / cleared。
8. **训练/评测观测窗口不一致**（§8.2-4）：v1 与 v2 都是在"训练连续 4 帧 / 闭环隔帧 4 帧"的
   错配下测的；这不影响 v1↔v2 的对比，但意味着两个绝对数字都不是"学生真实上限"。
9. **离线 top-1 与闭环行为脱节**（v1 59.6 % → 1,084；v2 59.1 % → 2,400）：本卡不把离线提升当结论。

> **更正注记（2026-11-10，EXP-P3-009）**：本卡 §6 的 `2,399.6` vs `1,084.0`（2.21×）是**已退役的 `strided` 尺**上的 5 局均值比较，且 +1,315.6 帧与"同配方重训"的 checkpoint 级漂移同阶（实测 sd_ckpt = 472.5 帧、单次训练分辨极限 1,309.7 帧）；原始数字不改，该结论按新判据降级为**无法判定（分辨率不足 + 尺已退役）**。详见 [EXP-P3-009](<EXP-P3-009-eval-power.md>)。
