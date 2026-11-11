# EXP-P3-012：D1 教师数据集——采集驱动与 8 面小规模验证

- 日期：2026-11-10 起，2026-11-12 本轮（驱动 + 小规模验证）。
- Provider=DeepSeek，模型=DeepSeek v4.1 flash（执行子代理）。
- 工作包：任务书 T3（[2026-11-10-P3 重启任务书 §4 T3](../tasks/2026-11-10-P3重启任务书.md)）的
  **驱动实现 + 小规模验证 + 全量规划**。本卡只写「设计与验证」；**全量 200 万帧的结果留空位**
  （由主代理用受管后台作业跑完后补 §7）。
- 结论（本卡范围）：**驱动成立，8 面小规模验证通过**；全量数据集尚未产出。

> 本卡与 [EXP-P3-012a](EXP-P3-012a-trajectory-start.md) 的分工：012a 验收 `TrajectoryStart`
> 本身（20 个接管点、同一进程连续 10 次、fail-closed 三例）；本卡把它接进采集驱动，做
> **起点混合**、**逐面可达性**、**逐决策 obs-v3 落盘**与**全量规划**。两卡不重复记数字。

## 预注册（运行之前写定）

本轮只跑小规模验证，**不跑全量**（全量 200 万帧由主代理用受管后台作业跑）。以下判据在
任何一次 `--phase run` 之前写定，运行后未改动：

| # | 预注册判据 | 判据来源 |
|---|---|---|
| P1 | 8 面各 **1 个 shard、≥500 决策**（共 ≥4,000 决策），Normal、team 0、`--obs v3`、`--mode fast` | 任务书 T3 §4 + 派发要求 |
| P2 | 每个 shard 的 HASH 对齐率 **100%**（`hash_checkpoint_pairs` 全部 `equal=true`）；出现 1 个不等即判失败 | T3 验收「HASH 对齐 100%」 |
| P3 | `obs_v3_dataset.load_teacher_samples_v3(require_observations=True)` 在**全部** 4,000 条决策上通过，digest == `ef2fbd15…`，`obs_unresolved == []` | T3 验收「加载闸门」 |
| P4 | 每个 `shard.json` 记录 `obs_spec_version=obs-v3` + digest + `stage_index` + `start_mode` + `frames` + `decisions` | 派发要求第 4 条 |
| P5 | 逐面可达性先实测（每面 1 局，到死亡或到 cap），plan 的每片帧数**不得超过**该面可达帧数；拿不到目标**不得**用合成数据补，只记 `coverage_shortfall` | 任务书 T3 §4 步骤 3 + 派发要求第 2 条 |
| P6 | `trajectory` 起点必须走 `TrajectoryStart`：channel A 逐决策复现语料（掩码 + 哈希）才允许接管，channel B 用 `replay_to_frame` 快进后逐帧核对；任何不一致 → `EXIT_ALIGN(4)`，不留 shard | 任务书 T3 步骤 1 + 派发要求第 3 条 |
| P7 | 并发：本任务 WSL 引擎 ≤4（`--workers` 硬上限 4） | 派发要求 |
| P8 | 全量规划给出：分片数、预计墙钟（用实测 frames/s 与并发）、每面可达帧数、磁盘占用（对照 D-05 300 GB 与 80% 告警线） | 派发要求 |

**不做**：不跑全量、不改引擎/协议/依赖/第三方、不重建或覆盖任何 `build/rl*`、不覆盖 D0、
不删除文件、不降低 HASH 闸门、不在 plan 里把短缺口写成已补齐。

## 1. 设计

### 1.1 驱动 CLI

新增 `tools/rl/v400_d1_collect.py`（六阶段，全部可 resume）：

```bash
# 1) 逐面可达性：每面 1 局，到死亡或到 cap（只跑 channel A，不落像素）
python tools/rl/v400_d1_collect.py --phase probe --stages 0,1,3,2,4,5,6,7 --seeds 1 \
    --workers 4 --probe-cap-frames 16000 --mode fast --difficulty 1

# 2) 规划：可达性 + 目标 -> 不可变 shard spec（--plan-limit-per-stage 给小规模用）
python tools/rl/v400_d1_collect.py --phase plan --plan-name smoke \
    --stages 0,1,3,2,4,5,6,7 --seeds 1-8 --workers 4 \
    --frames-target-total 2000000 --frames-target-per-stage 150000 \
    --decisions-per-shard 500 --plan-limit-per-stage 1 --start-mix shard --obs v3

# 3) 采集：K 个 worker，一个 shard 一个子进程（同一 shard 内 A/B 串行）
python tools/rl/v400_d1_collect.py --phase run --plan-name smoke --workers 4

# 4) 闸门：HASH 100% + load_teacher_samples_v3(require_observations=True)
python tools/rl/v400_d1_collect.py --phase gate --plan-name smoke

# 5) 报告：逐面表 + 全量规划（分片/墙钟/磁盘）
python tools/rl/v400_d1_collect.py --phase report --plan-name smoke
```

关键参数（默认值按任务书）：`--stages 0,1,3,2,4,5,6,7`、`--seeds 1-8`、`--workers 4`、
`--frames-target-total 2000000`、`--frames-target-per-stage 150000`、
`--decisions-per-shard 4000`（小规模用 500）、`--obs v3`、`--mode fast|v2h`、
`--difficulty 1`、`--start-mix shard,trajectory`（默认 `shard`）。

### 1.2 一个 shard 的两条通道

沿用 D0 的 A/B 结构（EXP-P1-014/018），但 channel B 由本驱动自己实现（见 1.4）：

| 通道 | 角色 | 做什么 | 失败行为 |
|---|---|---|---|
| A 标签 | `shadow`（null device，无 GL） | `tools/rl/v400_teacher.py --labels-only`：逐决策 `scores[18]` / `verified_mask` / `bomb` / `chosen_plan` / `margin` + 4 个逐帧掩码 + 决策前后 25 字段 `HASH` | 非零退出码 → `EXIT_FAIL_CLOSED(3)`，不留 shard |
| B 像素 | `env`（`READBACK 1` + 共享内存） | 同一 `RESET`、同一逐帧掩码带重放；每决策读 `d-4..d-1` 四帧渲染，并在决策两侧比对 `HASH` | 任一边界不等 → `EXIT_ALIGN(4)`，留 `.partial` 作证据 |

`shard.json` 由 `TeacherObsShardWriter(obs_version="v3")` 写出，除冻结字段外补：
`stage_index`、`start_mode`、`start_frame`、`decisions`、`frames`、`d1{schema,plan,key}`、
`trajectory{…}`（trajectory 起点）、`hash_checkpoint_pairs`（全部边界）、`alignment_extra`。

### 1.3 逐面可达性 → 分片规划（不用合成数据）

- `probe` 每面跑 **1 局**，`--max-frames` 给 cap、`--stop-on-death`：
  - 过关（`stage_changed`）→ 可达帧数 = 该面真实长度；
  - 死亡 → 可达帧数 = 死亡帧（`death_frame`）；
  - 到 cap 还没结束 → 记录 `capped=true`、`reachable_is_lower_bound=true`，**不外推**。
- `plan` 的每片决策数 = `min(--decisions-per-shard, 可达帧数/4)`，因此**单片帧数 ≤ 可达帧数**；
  每面片数 = `ceil(每面目标 / 单片帧数)`。跑不满时写 `coverage_shortfall` 与原因，不补数据。

### 1.4 起点混合与 `TrajectoryStart`

`--start-mix shard,trajectory` 按 shard 轮转分配起点；默认 `shard`。

- **shard 起点**：普通 `RESET stage diff team seed 0`，决策帧从 0 起，决策 0 用 obs-v1 的
  关首 padding（`pad=3`）。
- **trajectory 起点**：语料必须是**同一面的 shard 起点分片**（记录过 `RESET` 行、逐帧掩码带、
  每个决策边界的哈希表）。`RESET` 用**语料的 seed**（掩码带只对该轨迹有效），抽帧只在
  **带记录哈希的帧**里抽（`bias=0.65` 偏后半段、`boss_weight=0.25` 偏 Boss 段），且必须
  留够本片全部决策的帧数。
  - channel B：`replay_to_frame(traj.start(T-4))` 快进到 `T-4`（每个决策边界都是记录检查点，
    沿途逐点比对），再逐帧走 4 个记录掩码拿到 `T-4..T-1` 四帧渲染，第 4 步的 `HASH` 必须
    等于记录的 `T` 帧哈希；
  - channel A：shadow 教师不能从中间起，所以**从同一个 `RESET` 跑前缀并要求逐决策复现**
    （掩码列表 + 两侧哈希），复现不了 → `EXIT_ALIGN(4)`；前缀标签**丢弃**；
  - 落盘的决策帧**重基到 0**（loader 的 `frame == index*frames_per_decision` 算术），
    绝对接管帧、接管哈希、语料 id 记在 `shard.json` 的 `start_frame`/`trajectory` 里；
  - 代价：前缀要重放一遍（时间 +0~100%），**不产生数据**，规划里按
    `prefix_frames_paid` 单列。

### 1.5 resume 与失败留痕

- 已完成分片：`shard.json` 能被 obs-v3 loader 加载 → 跳过（`SKIPPED_COMPLETED`）；
- 失败分片：writer 把 `shards/.<key>.partial` 留在盘上（含 `incomplete.json`），
  重跑遇到它直接 `EXIT_FAIL_CLOSED`，**不覆盖**；
- 聚合：writer 维护 `d1/manifest.json`（跨进程 flock），另按分片写
  `.tmp_v400_d1/reports/<key>.json`，`gate` 再汇总成 `gate.<plan>.json`。

### 1.6 冻结物与闸门

- obs-v3 冻结：`obs_spec_version=obs-v3`、digest `ef2fbd15385f26f425db2bfe855d105ee009023acb08c26916e811963553a105`；
  `shard.json` 的 `obs_spec_digest` 与 `FROZEN_OBS_SPEC_V3_DIGEST` 不等即失败；
- HASH 对齐：`hash.checkpoints_equal == hash.boundaries`（100%），否则 `EXIT_ALIGN`；
- 加载闸门：`load_teacher_samples_v3(require_observations=True)` + 每片
  `load_teacher_shard_v3`（含 act_hist 重建比对）。

## 2. 环境

- 引擎：`build/rl_v3`，SHA-256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`（T7.2 统一构建，未重建）。
- 数据模板 `/root/v400-data/game`；每片独立 data-dir `/root/v400-data/d1/<key>/{shadow,env}`。
- 教师 `--mode fast`（0.57× 实时，本机实测 ≈30–46 帧/秒/实例，取决于同机负载）。
- 并发：本任务 WSL 引擎 ≤4；同机另有 stage_clear 量产批次（rl_events，12+ 实例在跑）。
- WSL 侧一律 `env -u DISPLAY /root/v400-venv/bin/python`；env 通道走 `xvfb-run -a`。
- 线程：每个教师/引擎 `*_NUM_THREADS=1`（EXP-P1-018：共享机上单线程更快且只占 1 核）。

## 3. 逐面可达性实测

`--phase probe`，每面 **1 局**，`--max-frames 16000`、`--stop-on-death`，只跑标签通道。
cap 是**下界**：`capped=True` 表示到 cap 时教师还没死也没过关，`reachable_frames` 是下界，不外推。

| stage | 教师结局 | 可达帧数 | 下界 | 死亡帧 | 帧/秒 |
|---|---|---|---|---|---|
| 0 | 过关 | 13,484 | 否 | — | 34.8 |
| 1 | 到 cap | 16,000 | 是 | — | 34.1 |
| 2 | 到 cap | 16,000 | 是 | — | 28.8 |
| 3 | 死亡 | 15,332 | 否 | 15,332 | 22.2 |
| 4 | 到 cap | 16,000 | 是 | — | 50.9 |
| 5 | 到 cap | 16,000 | 是 | — | 46.8 |
| 6 | 死亡 | 15,344 | 否 | 15,344 | 37.9 |
| 7 | 死亡 | 12,872 | 否 | 12,872 | 45.0 |

- 与 D0 的 Stage 1 一致：本轮 stage 0 过关 13,484 帧，D0（EXP-P1-018，seed 1）是 13,480 帧。
- 派发说明里「教师只在 0/1/3 面过关」：本轮 cap 16,000 内只观察到 **0 过关**，3/6/7 死亡，
  1/2/4/5 到 cap 未死（下界）。是否过关要更长的 cap 才能判，本卡不据此断言。
- 探测失败重跑：首轮 4 个晚面（4/5/6/7）在 4 并发、同机 loadavg≈20 时全部
  `SPAWN … Broken pipe`（shadow 子进程管道断），2 并发重跑全部成功。见 §10 限制 1。


## 4. 8 面小规模验证（每面 1 shard × 500 决策）

`--phase plan --plan-name smoke --plan-limit-per-stage 1 --decisions-per-shard 500
--start-mix shard`，然后 `--phase run --workers 4`、`--phase gate`。
产物：`.tmp_v400_d1/d1-round1/shards/*`（9 个分片）、`.tmp_v400_d1/gate.smoke.json`。

| stage | shards | 决策 | 帧 | 起点 | 教师停止 | HASH 全等 | 加载闸门 | 探测死亡帧 |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 500 | 2,000 | shard | max_frames | 1,000/1,000 | 500/500 | — |
| 1 | 1 | 500 | 2,000 | shard | max_frames | 1,000/1,000 | 500/500 | — |
| 2 | 1 | 500 | 2,000 | shard | max_frames | 1,000/1,000 | 500/500 | — |
| 3 | 1 | 500 | 2,000 | shard | max_frames | 1,000/1,000 | 500/500 | 15,332 |
| 4 | 1 | 500 | 2,000 | shard | max_frames | 1,000/1,000 | 500/500 | — |
| 5 | 1 | 500 | 2,000 | shard | max_frames | 1,000/1,000 | 500/500 | — |
| 6 | 1 | 500 | 2,000 | shard | max_frames | 1,000/1,000 | 500/500 | 15,344 |
| 7 | 1 | 500 | 2,000 | shard | max_frames | 1,000/1,000 | 500/500 | 12,872 |

**合计 8 shard / 4,000 决策 / 16,000 帧；HASH 检查点 8,000/8,000 = 1.0；加载闸门
4,000/4,000 通过；digest `ef2fbd15…a105`；`obs_unresolved=[]`；分片全部 `complete=true`。**

逐片：

| shard | stage | seed | 起点 | 决策 | 帧 | HASH | 加载 | 教师帧/秒 | 环境帧/秒 | 墙钟 s |
|---|---|---|---|---|---|---|---|---|---|---|
| `d1-s0d1-n500-seed1-shard` | 0 | 1 | shard | 500 | 2,000 | 1,000/1,000 | 500/500 | 90.3 | 86.2 | 45.5 |
| `d1-s1d1-n500-seed2-shard` | 1 | 2 | shard | 500 | 2,000 | 1,000/1,000 | 500/500 | 80.2 | 74.5 | 52.0 |
| `d1-s3d1-n500-seed3-shard` | 3 | 3 | shard | 500 | 2,000 | 1,000/1,000 | 500/500 | 70.4 | 69.7 | 57.3 |
| `d1-s2d1-n500-seed4-shard` | 2 | 4 | shard | 500 | 2,000 | 1,000/1,000 | 500/500 | 48.5 | 73.6 | 68.6 |
| `d1-s4d1-n500-seed5-shard` | 4 | 5 | shard | 500 | 2,000 | 1,000/1,000 | 500/500 | 68.9 | 72.1 | 57.2 |
| `d1-s5d1-n500-seed6-shard` | 5 | 6 | shard | 500 | 2,000 | 1,000/1,000 | 500/500 | 63.6 | 91.8 | 53.4 |
| `d1-s6d1-n500-seed7-shard` | 6 | 7 | shard | 500 | 2,000 | 1,000/1,000 | 500/500 | 30.4 | 79.3 | 91.2 |
| `d1-s7d1-n500-seed8-shard` | 7 | 8 | shard | 500 | 2,000 | 1,000/1,000 | 500/500 | 43.0 | 68.3 | 75.9 |

- 动作分布（教师 `chosen_plan.first_action`，18 类）：
  `[2161, 102, 274, 243, 75, 64, 205, 169, 55, 0, 13, 206, 292, 28, 19, 32, 38, 24]`，
  **顶类占比 0.540**（第 0 类 2,161/4,000），bomb 6 次。这是 D1 训练要处理的不平衡基线。
- 墙钟：8 片在 4 worker 下首轮约 3 分钟（其中 stage 0 首片 26 s 因引擎 `HELLO` 阶段控制管道
  被关而失败、隔离 `.partial` 后重试 46 s 成功，见 §10 限制 2）；中位 57.3 s/片。
- 教师停止原因全部是 `max_frames`（500 决策 × 4 帧封顶），没有触发死亡，因此
  **小规模验证没有覆盖「到死亡为止」的截断路径**；该路径的可达上界由 §3 的探测给出。


## 5. 起点混合验证（stage 0，`TrajectoryStart`）

`--phase plan --plan-name traj --stages 0 --start-mix shard,trajectory --decisions-per-shard 500`：
第 0 片是关首起点（与 §4 的 stage 0 分片同 key，直接 resume），第 1 片是接管起点。

| 项 | 值 |
|---|---|
| 分片 | `d1-s0d1-n500-seed10-trajectory` |
| 语料 | `d0/dual-s0d1seed10`（D0，Stage 1，13,288 帧，shard 起点） |
| 接管帧 `target_frame` | **11,176**（偏后半段 + Boss 段：`bias=0.65`、`boss_weight=0.25`、`boss_start=2936`） |
| 记录哈希 / 实测哈希 | `1b51ca9c2e4a4b1d` / `1b51ca9c2e4a4b1d`（相等） |
| 前缀 | 2,794 决策 / 11,176 帧，逐决策比对掩码 + 两侧哈希后**丢弃**；`fast_forward.frames_advanced=11,172`、检查点 2,794 |
| 落盘 | 500 决策 / 2,000 帧，决策帧**重基到 0**，`start_frame=11176` |
| HASH 对齐 | 1,000/1,000（含接管检查点） |
| 加载闸门 | 500/500，digest `ef2fbd15…a105` |
| 墙钟 | 280.6 s（教师 11.2 帧/秒「按落盘帧算」、环境 19.7 帧/秒；前缀重放占了大部分） |

`shard.json` 实际字段（节选）：

```json
{"shard": "d1-s0d1-n500-seed10-trajectory", "obs_spec_version": "obs-v3",
 "obs_spec_digest": "ef2fbd15385f26f425db2bfe855d105ee009023acb08c26916e811963553a105",
 "stage_index": 0, "start_mode": "trajectory", "start_frame": 11176,
 "decisions": 500, "frames": 2000, "frames_per_decision": 4, "stage_frames": [0, 2000],
 "reset_command": "RESET 0 1 0 10 0",
 "trajectory": {"traj_id": "d0/dual-s0d1seed10", "target_frame": 11176,
                "expected_start_hash": "1b51ca9c2e4a4b1d", "prefix_decisions_discarded": 2794},
 "alignment": {"hash_checkpoints": {"boundaries": 1000, "checkpoints_equal": 1000,
                                    "checkpoints_mismatched": 0}}}
```

- **代价**：接管片要重放前缀（本片 11,176 帧重放 vs 2,000 帧落盘，约 5.6×），时间成本高、
  磁盘成本不变（前缀不落盘）。全量若一半分片用接管起点，墙钟约 ×1.27（见 §6 注）。
- **限制**：语料只有 13–16k 帧，所以全量的接管片必须比关首片小
  （`--trajectory-decisions-per-shard 1000`，即 4,000 帧/片），片数随之上升。
- **fail-closed 证据**：本片跑通前，单测覆盖了「前缀掩码不等」「前缀哈希不等」「前缀长度不足」
  三条拒绝路径（`tests/test_v400_d1_collect.py`）。


## 6. 全量 200 万帧规划

`--phase plan --plan-name full --decisions-per-shard 4000`（不限片数），
`--phase report --plan-name full --rates-plan smoke`：rates 用 §4 的 8 个实测分片。

| stage | 可达帧数 | 帧/片 | 片数 | 帧数 |
|---|---|---|---|---|
| 0 | 13,484 | 13,484 | 19 | 256,196 |
| 1 | 16,000 | 16,000 | 16 | 256,000 |
| 2 | 16,000 | 16,000 | 16 | 256,000 |
| 3 | 15,332 | 15,332 | 17 | 260,644 |
| 4 | 16,000 | 16,000 | 16 | 256,000 |
| 5 | 16,000 | 16,000 | 16 | 256,000 |
| 6 | 15,344 | 15,344 | 17 | 260,848 |
| 7 | 12,872 | 12,872 | 20 | 257,440 |
| **合计** | | | **137** | **2,059,128** |

- **每面 25 万帧**（> 每面 15 万帧的门），总计 2,059,128 帧 ≥ 200 万；`coverage_shortfall=0`。
- **预计墙钟**（实测教师 66.2 帧/秒、环境 74.1 帧/秒、每片固定开销 ≈0.05 s）：
  **4 引擎 4.1 h、8 引擎 2.0 h、12 引擎 1.4 h、16 引擎 1.0 h**。
  任务书 §4 T3 的「17–30 帧/秒、12–16 实例、2–4 h」是保守估计；本机空闲时实测明显更快。
  **注意**：若同机还有别的大批量（例如 stage_clear 12+ 实例），实测降到 ≈11 min/片，
  137 片 / 4 worker ≈ 6.3 h —— 这就是本卡建议全量用 **12–16 worker** 的原因。
- **磁盘**：实测 **54,256.9 B/决策**（planes 52,112 + labels + `shard.json` 的哈希对表），
  514,782 决策 → **27.93 GB = D-05（300 GB）的 9.31%**，**不触发 80% 告警线**。
  其中 `shard.json` 的 `hash_checkpoint_pairs`（2 条/决策）约 0.4 GB，已计入。
- **起点混合的成本**：`--start-mix shard,trajectory` 时，接管片按 `--trajectory-decisions-per-shard
  1000` 降到 4,000 帧/片（语料只有 13–16k 帧），片数升到 ≈ 2×；每片再付一次前缀重放。
  按本卡实测的前缀/落盘比（11,176/2,000）估算，墙钟约 **×1.27**，磁盘**不变**。
  本轮全量规划按默认 `shard` 起点给出。
- **每实例帧速**：教师 66.2 帧/秒（中位，范围 30–90）、环境 74.1 帧/秒（范围 68–92）。


## 7. 全量结果（留空，主代理跑完后补）

- 全量观测：**本卡不写**。本节留给主代理用受管后台作业跑 200 万帧后的审计表
  （各面帧数、起点类型分布、死亡数、bomb 数、动作类别分布、实际墙钟、每实例帧速、
  `partial=false`、HASH 对齐率、加载闸门、`manifest.json` 总计）。
- 交接命令（数据目录 `.tmp_v400_d1/d1`，与验证产物 `.tmp_v400_d1/d1-round1` 分离）：

```bash
cd /mnt/d/WorkShop/Touhou_RL
bash .tmp_v400_d1/run.sh --phase plan --plan-name full \
    --out-dir .tmp_v400_d1/d1 --stages 0,1,3,2,4,5,6,7 --seeds 1-8 \
    --frames-target-total 2000000 --frames-target-per-stage 150000 \
    --decisions-per-shard 4000 --trajectory-decisions-per-shard 1000 \
    --start-mix shard --obs v3 --mode fast --difficulty 1
bash .tmp_v400_d1/run.sh --phase run --plan-name full --workers 12 --retry-failed
bash .tmp_v400_d1/run.sh --phase gate --plan-name full
bash .tmp_v400_d1/run.sh --phase report --plan-name full
```
- `--retry-failed` 是必需的：shadow 子进程在重负载下偶发 `SPAWN … Broken pipe`（§10 限制 1），
  它会把失败分片的 `.partial` 改名隔离后重试，不删文件、不覆盖已完成分片。

## 8. 有效性自检（V1–V9）

| # | 检查项 | 结论与证据 |
|---|---|---|
| V1 | 输入是否包含目标信息 | 不适用（本卡不训练）。数据集本身的输入是 obs-v3 平面，标签是教师在**同一帧**上的搜索结论；观测时刻（`frames[i-3..i] -> mask[i+1]`）由 loader 逐行复核，`pad` 只允许出现在决策 0。 |
| V2 | 标签/取值是否有方差、正例够不够 | 是。18 类动作全都有非零计数（最小类 13 次，最大 2,161 次），bomb 6 次；`scores` 每决策 18 维、`verified_mask` 每决策 18 位，没有常量列。 |
| V3 | 数值是否物理合理 | 是。每决策 4 帧、每片 500 决策 = 2,000 帧；stage 0 的 13,484 帧与 D0 的 13,480 帧、人类参考 13,258 帧一致；帧速 30–90 帧/秒落在 EXP-P1-018 的量级内。 |
| V4 | 统计单位是否独立 episode、样本量是否达标 | 小规模验证是 **8 个独立 episode**（各 1 片、各 1 seed），共 4,000 决策；任务书要求的 8 面 × ≥500 决策达标。全量规划 137 片 / 514,782 决策，远高于 50 万决策的规格。 |
| V5 | 平凡基线跑出来是多少 | 不适用（没有训练）。但动作分布本身就是基线：教师「第 0 类」占 54.0%，任何只预测第 0 类的学生在这里能拿到 54% 的 top-1 —— 这正是 T4 要用 macro recall / 危险状态一致率而不是 top-1 的理由。 |
| V6 | 训练/评估场景是否分开 | 本卡只产数据，未训练。数据侧的分割：8 面各自独立分片、种子互不相同（seed 1–8）；确认种子 60000–60199 完全没有使用。 |
| V7 | 换 seed 重跑是否仍成立 | 是。probe、8 面采集、trajectory 分片用了 seed 1–10；stage 0 的 HASH 全等在 seed 1 与 seed 10（接管）两种起点下都成立。 |
| V8 | 是否混入特权信息 | 数据里**有**教师的内存态哈希（`hash_pre/hash_post`、`checkpoint_pairs`），但它只用于 A/B 对齐与接管校验，**不是** obs-v3 的输入（obs-v3 的 planes 只有 play96x112 / hud / act_hist，见 `shard.json` 的 `obs` 块与 loader 的形状检查）；特权信息不进学生输入。 |
| V9 | 是否用了预注册指标 | 是。P1–P8 在第一次 `--phase run` 之前写定（§预注册），运行后未改动；本卡所有数字由 `.tmp_v400_d1/{reachability,plan.*,gate.*,report.*}.json` 渲染（`.tmp_v400_d1/card_tables.py`），没有手抄。 |


## 9. 成本

- 机器时间（WSL 引擎）：可用分片 9 个（8 面 + 1 接管）合计 4,500 决策 / 18,000 帧，
  实跑 4 worker 首轮 ≈3 min + 重试 46 s + 接管片 281 s；**可达性探测是长尾**：
  8 面各 1 局、cap 16,000 帧，首轮 4 并发 ≈13 min（stage 0 过关、3/6/7 死亡），
  4 个晚面在 4 并发下 `SPAWN` 断管后以 2 并发重跑 ≈20 min。合计约 **55 分钟引擎墙钟**。
- 磁盘（新增，全在 `.tmp_*`，不进版本库）：`.tmp_v400_d1` 共 **1.3 GB**，其中
  验证数据集 `d1-round1` 244 MB（9 片，实测 obs 234.5 MB，manifest 记 4,500 决策 /
  18,000 帧）、probe 标签 137 MB、第二遍验证的 staging 109 MB、分片工作目录 100 MB。
- 测试：`tests/test_v400_d1_collect.py` **25 passed**；全量 `-k v400` 回归
  **1542 passed, 6 skipped**（65 s）。
- 没有新增 `build/rl*`，没有改引擎/教师/协议/依赖/第三方，没有覆盖 D0。


## 10. 限制

1. **shadow 子进程在重负载下会断管。** 首轮 4 个晚面（4/5/6/7）在 4 并发、同机 loadavg≈20
   （另一批量 15–19 个引擎）时全部以 `SPAWN … Broken pipe` 失败；同机空闲后 1 并发、2 并发
   都成功。全量 12–16 worker 时预计仍会偶发，**必须用 `--phase run --retry-failed` 续跑**
   （它会先把 `.<key>.partial` 改名隔离成 `.partial.failed-<ts>`，不删文件）。
2. **并发首启偶发控制管道关闭。** 4 个 worker 同时 `xvfb-run` 起引擎时，有 1 个分片在
   `HELLO` 阶段拿到 `harness closed the control pipe`（26 s 失败）；单 worker 重试即成功。
3. **可达性是下界。** cap 16,000 帧对 1/2/4/5 面只证明「教师至少能活 16,000 帧」，
   不能据此判过关还是更晚死亡；要判请加大 `--probe-cap-frames`。
4. **小规模验证没有覆盖死亡截断与过关截断。** 8 片的教师停止原因都是 `max_frames`
   （2,000 帧远小于各面可达长度），所以「到死亡为止」的 `--stop-on-death` 路径和
   stage-clear 的 `transition_trim` 路径只有单测与 D0 语料证据，本轮没有真实样本。
5. **接管起点不增加覆盖。** shadow 教师不能从中间起，所以 `TrajectoryStart` 只能实现成
   「从同一 `RESET` 重放前缀 → 逐决策核对 → 丢弃前缀」。它改变**起点分布**，不产生
   shard 起点跑不出来的状态；前缀是纯成本（本片 11,176 帧重放换 2,000 帧落盘）。
6. **只有 stage 0 的接管起点做过真机验证。** 其余 7 面的语料要等全量的关首阶段产出后才能建；
   plan 在那之前会把「要求 trajectory 起点但无该面语料」判为 blocked（fail-closed），
   不会静默降级。
7. **墙钟是在空闲机器上测的。** 66 帧/秒（教师）只有在别的批量停下后才成立；同机有
   12+ 引擎时实测降到 ≈11 min/片。全量请按 §6 的 12–16 worker 估算，并按实测修正。
8. **接管分片的帧号是重基后的。** `labels.jsonl` / `frames` 从 0 起，绝对引擎帧在
   `shard.json.start_frame`；`act_hist` 在分片起点清零（这是段语义，不是 bug），
   需要绝对帧的消费者必须自己加 `start_frame`。
9. **快速教师模式的质量门不在本卡重跑。** T3 第 2 步要求 `--mode fast` 在 Stage 1 开发种子上
   ≥9/10 过关；D0 的 16 个 fast 分片全部 `stage_clear=true`（EXP-P1-018）是既有证据，
   本卡只引用，不重复实验。
10. **分片 key 必须唯一（已修）。** 早期版本的 key 只含 `(stage, seed, 每片决策数, 起点模式)`；
   每面需要 16–20 片而 seeds 只有 8 个，于是第 9 片起 key 与第 1 片重复，137 片会坍缩成
   ≈64 个不同分片。现在 key 只在**真正冲突时**才追加 `-i{index}`，所以同一份输入重新生成的
   key 与已在跑/已完成的批次一致（不会白跑）。全量 plan 已确认 **137 片 / 137 个唯一 key**。


## 证据

- 驱动：`tools/rl/v400_d1_collect.py`（六阶段：probe / plan / shard / run / gate / report，
  全部可 resume；`--retry-failed` 隔离而非删除失败 staging）。
- 测试：`tests/test_v400_d1_collect.py`（25 passed）：plan 数学与 `coverage_shortfall`、
  trajectory 接管点/前缀 fail-closed、重基分片的 loader 往返、probe 记录形状、并发上限、
  全量磁盘估算。
- 可达性：`.tmp_v400_d1/reachability.json`（8 面，`stages_run_now` / `stages_reused` 标明
  两轮探测的合并来源）、`.tmp_v400_d1/probe/stage*/labels_teacher.log`。
- 规划：`.tmp_v400_d1/plan.smoke.json`、`plan.traj.json`、`plan.full.json`（137 片，key 唯一）。
- 数据：`.tmp_v400_d1/d1-round1/shards/*`（9 片）与 `manifest.json`（4,500 决策 / 18,000 帧 /
  234,518,094 B）；每片含 `shard.json`（obs 版本+digest+stage_index+start_mode+frames+decisions）、
  `labels.jsonl`、`obs/{play96x112,hud,act_hist}.npy`。
- 闸门：`.tmp_v400_d1/gate.smoke.json`、`gate.traj.json`；报告：
  `.tmp_v400_d1/report.{full,traj}.json` 与同名 `.md`；表格渲染脚本 `.tmp_v400_d1/card_tables.py`。
- 运行日志：`.tmp_v400_d1/logs/*.log`（每片的 `COMMAND` + 单行结果/失败原因）、
  `.tmp_v400_d1/{probe,probe2,smoke,run_smoke}.log`。
- 与全量的关系：全量用同一驱动、同一 `plan.full.json`（`--out-dir .tmp_v400_d1/d1`，
  与验证产物分离）；全量结果补进 §7。

