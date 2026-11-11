# EXP-P3-012a：`TrajectoryStart`——按已记录轨迹快进并在校验过的帧上交出控制权

- 日期：2026-11-10 起，2026-11-11 完成验收。Provider=DeepSeek，模型=DeepSeek v4.1 flash。
- 工作包：T3 第 1 步（任务书 [2026-11-10-P3 重启](../tasks/2026-11-10-P3重启任务书.md) §4 T3）。对应规格 [03 §5](../03-系统架构.md) 的第三种进关方式。
- 结论：**成立**。20 个随机接管点（偏置采样）**20/20** 的引擎哈希等于语料记录的哈希，共比对 45,502 个检查点、快进 181,928 帧；另做一次均匀采样的 20 点补充运行，**20/20** 相等（31,420 检查点、125,600 帧）。同一进程内用 `rl_v3` 的进程内 `RESET` 连续做 **10/10** 次接管，全部相等、`engine_restarts=0`；每次 `RESET` 中位 **0.583 s**、含快进的单次接管中位 **3.707 s**，而每局新起一个进程的冷启动中位 **1.918 s**（建目录+启动+`RESET`），进程内 `RESET` 比它快 **3.29×**，比 EXP-P4-001 §7.1 记录的历史值 6–7 s/局快 **10.3–12.0×**。fail-closed 在真实引擎上被触发两次（目标帧、途中检查点各一次，均以退出码 1 结束并打印 frame/expected/observed），掩码长度不足与目标帧越界由单测覆盖。单测 55 passed / 1 skipped（引擎不可用时跳过的 live 用例），全量 `-k v400` **1517 passed, 6 skipped, 0 failed**（exit 0）。

> 卡片补写说明：本轮是**续跑**。前一个代理写完了 `trajectory.py`、`trajectory_verify.py`、单测与 2 点冒烟（`.tmp_v400_traj/smoke1.json`）后中断，没有跑验收、没有写卡片。本卡片正文由续跑代理在验收运行**之后**补写；判据来自任务书 §4 T3 第 1 步与驱动里的常量，运行前后**没有**改动任何阈值、采样参数或代码判据，全部数字由两份 JSON 报告用 `.tmp_v400_traj/card_tables.py` 渲染，没有手工转录。

## 预注册（判据的来源与冻结位置）

- 主指标 1：**20** 个随机接管点，每一个在接管帧上的引擎 25 字段哈希都等于语料记录的哈希；20/20 才算通过。
- 主指标 2：同一进程内连续 **10** 次 `TrajectoryStart`（走 `rl_v3` 的 T7.1 进程内 `RESET`），10/10 的 `RESET` 哈希与接管帧哈希都相等，且全程 `engine_restarts=0`。
- 主指标 3（fail-closed）：故意给错哈希必须抛 `TrajectoryHashMismatch` 并以非零退出码结束；掩码长度不足、目标帧越界必须抛错；语料自相矛盾必须拒绝加载。
- 冻结位置：`touhou_rl/v400/gym/trajectory_verify.py` 的 `REQUIRED_POINTS=20`、`REQUIRED_CONSECUTIVE=10`，`BOSS_START_FRAME=2936`；任何一项不满足，驱动写 `status="failed"` 并返回退出码 **3**。单测与驱动写于 2026-11-10（前一个代理），冒烟运行之后判据未改。
- 采样：`--sampling-seed 20261110`、`bias=0.65`、`boss_weight=0.25`、`boss_start=2936`（T3 要求"偏向后半段与 Boss 段"）；接管点只从**带记录哈希的帧**里抽，因此每个点按构造即可校验。补充运行另用 `--sampling-seed 20261111`、`--bias 0 --boss-weight 0`（均匀）。
- 失败即停：任一哈希不等、任一 `STEP` 少走帧、任一 `exit/game_over`、任一静默重启、批量与逐帧两种回放不一致，都会写入 `reasons` 并让运行 `failed`。

## 1. 接口

`TrajectoryStart` 是 **frozen dataclass**，用"同一个 `StageStart` + 逐帧掩码带 + 记录哈希表"描述一次接管：

```python
TrajectoryStart(seed, stage_index, difficulty, masks, target_frame,
                hashes, team=0, resources="standard", traj_id="")
```

- `masks`：逐帧掩码（u16，每帧一个），来自语料的 `step_frames` 展开；长度即"已记录帧数"。
- `hashes`：`{stage_frame: hash}`，`stage_frame` 是 `RESET` 归零后的引擎帧号；**`target_frame` 必须在表里**。
- `target_frame`：接管帧，取值 `0 ≤ target ≤ len(masks)`；等于 `len(masks)` 是合法的最后一帧。
- 构造时即 fail-closed：负目标/越界目标 → `TrajectoryTargetError`/`TrajectoryMaskError`；掩码不是 `0..0xFFFF` → `TrajectoryMaskError`；目标帧无记录哈希 → `TrajectoryHashUnavailable`；哈希不是 16 位十六进制（含 `"unknown"`、空串、整数、截断值）→ `TrajectoryHashUnavailable`。大小写会被归一化，但**不会**把不可验证的值当成可比较的值。
- `replay_to_frame(env, start, batch_frames=64)`：用 `env.reset(start.stage_start)` 进关（并核对 `RESET` 回复里的哈希 == `hashes[0]`），然后按掩码带发 `STEP <mask> <n>` 块；**每一块都不跨过记录哈希帧**，于是沿途每个检查点都被比对；到 `target_frame` 后交还控制权，并返回 `TakeoverReport`（reset/目标哈希、期望哈希、帧数、`STEP` 次数、检查点数、`wall_s`/`reset_s`/`fast_forward_s`）。
- 交出的状态由三处独立确认：`STEP` 回复自带的 `hash`、`step` 回复自带的 `frame`（必须等于累计帧数）、共享内存 header 的 `frame`（必须等于 `target_frame`）。
- 错误类型：`TrajectoryStartError` 为基类，`TrajectoryHashMismatch`（带 `frame`/`expected`/`observed`/`where` 与 `as_record()`）、`TrajectoryMaskError`、`TrajectoryTargetError`、`TrajectoryHashUnavailable`、`TrajectoryAborted`、`TrajectoryCorpusError`。`TrajectoryHashMismatch` 的 `where` 区分 `"RESET"`、`"fast-forward checkpoint"`。
- `sample_target_frame(rng, candidates, bias, boss_start, boss_weight)`：只从候选（带哈希的帧）中抽，一次 `random()` 定池、一次 `choice()` 取值，所以给定种子可复现；采样规则由 `tests/test_v400_trajectory_start.py` 的 6 个用例覆盖。
- `load_d0_shard` / `load_d0_shards`：D0 分片 → `RecordedTrajectory`（同上字段 + `start(target)`）。加载是 fail-closed 的：分片未 `complete`、`reset_command` 不是 `RESET stage diff team seed practice`、决策帧号不连续、`step_frames` 数量不等于 `frames_per_decision`、`hash_pre` 不接上一决策的 `hash_post`、`hash_checkpoint_pairs` 里任何一个 `equal != true`、两个独立哈希来源互相矛盾、或标签里有检查点缺失 A/B 对照，一律抛 `TrajectoryCorpusError`。

新增的 `gym` 侧入口（本次一并入库）：`Th08Env.advance(mask, frames)` 返回 `FrameReply(frame, advanced, hash, exit, game_over)`，`Th08Env.last_reset_hash` 暴露 `RESET` 回复的哈希，`HarnessClient(role="env"|"shadow")` 支持 `--rl-role shadow`（null device，无窗口/GL），`touhou_rl.v400.gym.__init__` 导出上述符号。

## 2. 语料：D0 有没有逐帧 STATE 哈希？

**没有"逐帧"，但有"逐决策边界"（每 4 帧）**，而且每个检查点有**两个独立来源**：

- `labels.jsonl`：每个 `kind=decision` 行有 `hash_pre` / `hash_post`（该决策 4 帧的前后状态哈希）与 `step_frames`（4 个逐帧掩码）。
- `shard.json` 的 `hash_checkpoint_pairs`：`{decision, boundary, frame, shadow, env, env_frames, equal, source="A/B"}`，即同一帧在 **shadow 进程**与 **env 进程**各算一次哈希并逐帧对齐比较；`equal=False` 的检查点就是语料自己承认不可用的点。

本卡用的语料是 `.tmp_v400_d0/d0/shards`（16 个分片，seeds 1–16，Stage 1 Normal，team 0，standard resources）：**214,048 帧 / 53,528 个检查点**，与 [EXP-P1-018](../experiments/EXP-P1-018-D0-scaleup.md) 记录的 D0 规模一致。分片还记录了一次 `ok_transition_trim`：stage-clear 提交跨过面边界时 `decision 3370` 的 post 状态已经是下一面的 setup，A/B 不等，于是语料把该检查点剪掉（`checkpoints_before_trim=6742 → after=6740`）。这是语料侧的已知处理，加载器只接受剪枝后的自洽表。

**既然语料自带哈希表，本卡不做"现采"，而是做"复现校验"**：把记录哈希表当作待验证对象，在真实引擎上按同一 `RESET` + 同一逐帧掩码重放，逐检查点比对。这是更强的证据：如果哈希表本身错了、或者引擎在 `rl_v3` 上没有复现原轨迹，比对就会在第一个分歧帧失败。

若换成一个**不带**哈希的语料（T3 的 D1 应当照此记录），程序里已固定的做法是：先按记录把 `RESET` + 掩码带重放一遍，把每个决策边界的 `STEP` 回复哈希写进 `shard.json` 的 `hash_checkpoint_pairs`（同时记 `env_frames`），再从**新进程**重放第二遍并逐点比对；两遍都相等才承认这张表。`TrajectoryStart` 对没有哈希的目标帧**直接拒绝**（`TrajectoryHashUnavailable`），不会凭空补一个哈希，也不会跳过校验——这是刻意的 fail-closed，避免"看起来跑通"的接管。

## 3. 设置（真实环境）

- 引擎：WSL Linux ELF `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，SHA-256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`，4,836,008 B。harness 源树 `/root/Touhou_RL/third_party/th08-rl`，`HEAD=1523df26ec643b7276d2cf487b299ad932a982b7`，`git status --porcelain` 为空（`dirty=false`）。
- 数据模板：`/root/v400-data/game`；每个接管点用**全新** data-dir（`create_data_dir` 对已存在路径直接拒绝，从不删除既有目录）：主运行 `/root/v400-traj/run2/dt-a-001…020`、`dt-b`、`dt-c`、`dt-d-00/01`、`dt-e-00/01/02`；补充运行 `/root/v400-traj/run3/…`；fail-closed `/root/v400-traj/failclosed/…`。
- 角色：主指标全部用 `--rl-role shadow`（null device，无 SDL/GL）；另用 `xvfb-run -a` 在 `env`（渲染）角色上复算 2 个点，验证生产像素路径与 shadow 的哈希一致。
- 并发：**同一时刻只跑 1 个引擎**（本任务上限 WSL ≤2；同机另有 T1 的 12 实例量产与别的任务在跑，`nproc=20`，主运行期间 `load1≈11.4`）。
- 采样与规模：`--points 20 --consecutive 10 --env-cross-check 2 --cold-restarts 3 --batch-frames 64`；主运行墙钟 **398.6 s**，补充运行 **97.3 s**。

## 4. 结果

### 4.1 20 个随机接管点（主运行）

**主运行（T3 规定的偏置采样，`--sampling-seed 20261110`；表格由 `card_tables.py` 从 JSON 渲染）**

status=pass；A 20/20，181,928 帧，45,502 个检查点；B 10/10，94,540 帧，23,645 个检查点；四元组 {'0': 0, '1': 2, '2': 12, '3': 6}；帧 5104–12612；Boss 20/20；RESET 中位 0.583 s；快进中位 3.021 s；吞吐中位 3088 帧/秒；墙钟 398.6 s

| # | 轨迹 | seed | 接管帧 | 分位 | 后半段 | Boss 段 | 期望哈希 | 实际哈希 | 相等 | 快进秒 | 单点总计秒 | 校验点 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `d0/dual-s0d1seed3` | 3 | 11328 | Q3 | 是 | 是 | `d40af4a0f7198cab` | `d40af4a0f7198cab` | ✅ | 3.73 | 5.69 | 2833 |
| 2 | `d0/dual-s0d1seed9` | 9 | 9988 | Q2 | 是 | 是 | `b432486e70d1752d` | `b432486e70d1752d` | ✅ | 3.22 | 5.07 | 2498 |
| 3 | `d0/dual-s0d1seed4` | 4 | 8296 | Q2 | 是 | 是 | `a7fc8fb3e4b75020` | `a7fc8fb3e4b75020` | ✅ | 2.75 | 4.64 | 2075 |
| 4 | `d0/dual-s0d1seed16` | 16 | 11140 | Q3 | 是 | 是 | `c72e31b088a42647` | `c72e31b088a42647` | ✅ | 3.69 | 5.56 | 2786 |
| 5 | `d0/dual-s0d1seed6` | 6 | 9428 | Q2 | 是 | 是 | `c3103dce2911a06b` | `c3103dce2911a06b` | ✅ | 3.09 | 5.01 | 2358 |
| 6 | `d0/dual-s0d1seed8` | 8 | 12612 | Q3 | 是 | 是 | `014e3d9ffee02c8c` | `014e3d9ffee02c8c` | ✅ | 4.08 | 5.98 | 3154 |
| 7 | `d0/dual-s0d1seed16` | 16 | 11076 | Q3 | 是 | 是 | `42bb158bed45a17c` | `42bb158bed45a17c` | ✅ | 3.60 | 5.48 | 2770 |
| 8 | `d0/dual-s0d1seed15` | 15 | 5592 | Q1 | 否 | 是 | `afd6b14b35af02f2` | `afd6b14b35af02f2` | ✅ | 1.73 | 3.62 | 1399 |
| 9 | `d0/dual-s0d1seed15` | 15 | 7184 | Q2 | 是 | 是 | `1da97e5693df91aa` | `1da97e5693df91aa` | ✅ | 2.23 | 4.10 | 1797 |
| 10 | `d0/dual-s0d1seed11` | 11 | 7896 | Q2 | 是 | 是 | `2f1963ec4dfa6d29` | `2f1963ec4dfa6d29` | ✅ | 2.54 | 4.40 | 1975 |
| 11 | `d0/dual-s0d1seed10` | 10 | 9284 | Q2 | 是 | 是 | `f434ebc26747d50e` | `f434ebc26747d50e` | ✅ | 3.04 | 4.91 | 2322 |
| 12 | `d0/dual-s0d1seed16` | 16 | 8272 | Q2 | 是 | 是 | `8ef47f7e57b6373b` | `8ef47f7e57b6373b` | ✅ | 2.65 | 4.53 | 2069 |
| 13 | `d0/dual-s0d1seed11` | 11 | 10608 | Q3 | 是 | 是 | `9ddbb53e2bde7469` | `9ddbb53e2bde7469` | ✅ | 3.45 | 5.34 | 2653 |
| 14 | `d0/dual-s0d1seed7` | 7 | 5104 | Q1 | 否 | 是 | `81aa016f48fd634b` | `81aa016f48fd634b` | ✅ | 1.56 | 3.44 | 1277 |
| 15 | `d0/dual-s0d1seed1` | 1 | 8732 | Q2 | 是 | 是 | `cff4a4a9e8c2d89b` | `cff4a4a9e8c2d89b` | ✅ | 2.72 | 4.60 | 2184 |
| 16 | `d0/dual-s0d1seed13` | 13 | 12604 | Q3 | 是 | 是 | `a5b3cc29cea18855` | `a5b3cc29cea18855` | ✅ | 4.08 | 5.97 | 3152 |
| 17 | `d0/dual-s0d1seed9` | 9 | 7516 | Q2 | 是 | 是 | `159fcdeb81c66ea9` | `159fcdeb81c66ea9` | ✅ | 2.36 | 4.24 | 1880 |
| 18 | `d0/dual-s0d1seed15` | 15 | 7608 | Q2 | 是 | 是 | `0e3764f233439350` | `0e3764f233439350` | ✅ | 2.48 | 4.37 | 1903 |
| 19 | `d0/dual-s0d1seed1` | 1 | 9312 | Q2 | 是 | 是 | `31f63cb96ef1cfb9` | `31f63cb96ef1cfb9` | ✅ | 3.02 | 4.91 | 2329 |
| 20 | `d0/dual-s0d1seed1` | 1 | 8348 | Q2 | 是 | 是 | `1648c70caa86ef45` | `1648c70caa86ef45` | ✅ | 2.74 | 4.61 | 2088 |

**分布**：stage 全部是 `0`（Stage 1，D0 只有这一面）；帧号 5,104–12,612；分位 Q0/Q1/Q2/Q3 = 0/2/12/6；后半段 18/20；Boss 段（≥2936）**20/20**；涉及 12 个不同 seed。

### 4.2 补充运行：均匀采样的 20 点（覆盖前段与 Q0）

T3 的偏置把 20 点全放进了 Boss 段，前段（Q0）没有样本。为了不让"HASH 对齐"只在后半段成立，另跑一次**均匀**采样：

**补充运行（均匀采样，`--sampling-seed 20261111`，`--bias 0 --boss-weight 0`）**

status=pass；A 20/20，125,600 帧，31,420 个检查点；B 1/1，4,388 帧，1,098 个检查点；四元组 {'0': 6, '1': 5, '2': 3, '3': 6}；帧 496–12352；Boss 15/20；RESET 中位 0.591 s；快进中位 1.624 s；吞吐中位 3126 帧/秒；墙钟 97.3 s

| # | 轨迹 | seed | 接管帧 | 分位 | 后半段 | Boss 段 | 期望哈希 | 实际哈希 | 相等 | 快进秒 | 单点总计秒 | 校验点 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `d0/dual-s0d1seed8` | 8 | 4388 | Q1 | 否 | 是 | `b0a61efb96303e5c` | `b0a61efb96303e5c` | ✅ | 1.41 | 3.33 | 1098 |
| 2 | `d0/dual-s0d1seed5` | 5 | 5140 | Q1 | 否 | 是 | `404b6bb79fb903db` | `404b6bb79fb903db` | ✅ | 1.62 | 3.55 | 1286 |
| 3 | `d0/dual-s0d1seed12` | 12 | 1208 | Q0 | 否 | 否 | `f08d07b7f2e0cadc` | `f08d07b7f2e0cadc` | ✅ | 0.33 | 2.31 | 303 |
| 4 | `d0/dual-s0d1seed10` | 10 | 4104 | Q1 | 否 | 是 | `12ca448fb5d2d976` | `12ca448fb5d2d976` | ✅ | 1.24 | 3.18 | 1027 |
| 5 | `d0/dual-s0d1seed13` | 13 | 5936 | Q1 | 否 | 是 | `2384336119156c13` | `2384336119156c13` | ✅ | 1.90 | 3.97 | 1485 |
| 6 | `d0/dual-s0d1seed12` | 12 | 4456 | Q1 | 否 | 是 | `66c392feca87f548` | `66c392feca87f548` | ✅ | 1.40 | 3.29 | 1115 |
| 7 | `d0/dual-s0d1seed15` | 15 | 3260 | Q0 | 否 | 是 | `8d33a31a9c3dbb8d` | `8d33a31a9c3dbb8d` | ✅ | 1.04 | 2.93 | 816 |
| 8 | `d0/dual-s0d1seed11` | 11 | 11128 | Q3 | 是 | 是 | `8e23365d145109b6` | `8e23365d145109b6` | ✅ | 3.66 | 5.54 | 2783 |
| 9 | `d0/dual-s0d1seed4` | 4 | 1800 | Q0 | 否 | 否 | `4b20c3a9b9d064fa` | `4b20c3a9b9d064fa` | ✅ | 0.52 | 2.42 | 451 |
| 10 | `d0/dual-s0d1seed5` | 5 | 10268 | Q3 | 是 | 是 | `4ba5fdad291d633d` | `4ba5fdad291d633d` | ✅ | 3.50 | 5.40 | 2568 |
| 11 | `d0/dual-s0d1seed16` | 16 | 12352 | Q3 | 是 | 是 | `a9265092e74f2f55` | `a9265092e74f2f55` | ✅ | 4.14 | 6.02 | 3089 |
| 12 | `d0/dual-s0d1seed9` | 9 | 7628 | Q2 | 是 | 是 | `fc438c9f838c87fe` | `fc438c9f838c87fe` | ✅ | 2.53 | 4.41 | 1908 |
| 13 | `d0/dual-s0d1seed13` | 13 | 11492 | Q3 | 是 | 是 | `0d80ba3c71c890b6` | `0d80ba3c71c890b6` | ✅ | 3.79 | 5.67 | 2874 |
| 14 | `d0/dual-s0d1seed15` | 15 | 6808 | Q2 | 是 | 是 | `6f3252d0e410fd5b` | `6f3252d0e410fd5b` | ✅ | 2.25 | 4.12 | 1703 |
| 15 | `d0/dual-s0d1seed5` | 5 | 8044 | Q2 | 是 | 是 | `9b31681106329428` | `9b31681106329428` | ✅ | 2.77 | 4.74 | 2012 |
| 16 | `d0/dual-s0d1seed15` | 15 | 10972 | Q3 | 是 | 是 | `a35ff93c4b875be4` | `a35ff93c4b875be4` | ✅ | 3.68 | 5.56 | 2744 |
| 17 | `d0/dual-s0d1seed3` | 3 | 2260 | Q0 | 否 | 否 | `88a7c5838b57e85d` | `88a7c5838b57e85d` | ✅ | 0.70 | 2.59 | 566 |
| 18 | `d0/dual-s0d1seed2` | 2 | 2368 | Q0 | 否 | 否 | `4b3504c7861eb6eb` | `4b3504c7861eb6eb` | ✅ | 0.66 | 2.50 | 593 |
| 19 | `d0/dual-s0d1seed12` | 12 | 496 | Q0 | 否 | 否 | `f2f724618805cd83` | `f2f724618805cd83` | ✅ | 0.11 | 1.96 | 125 |
| 20 | `d0/dual-s0d1seed6` | 6 | 11492 | Q3 | 是 | 是 | `5c1ec69f4880b865` | `5c1ec69f4880b865` | ✅ | 3.80 | 5.68 | 2874 |

两次运行合计 **40 个接管点、307,528 帧、113,944 个检查点比对全部相等**（含批量/逐帧等价性与 env 角色复算，见 §4.4）。

### 4.3 同一进程内 10 次连续接管

**10 次连续接管（主运行的 phase B）**

| # | 轨迹 | seed | 接管帧 | RESET 秒 | 快进秒 | 单次总计秒 | 校验点 | 期望哈希 | 实际哈希 | 相等 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `d0/dual-s0d1seed3` | 3 | 11328 | 0.603 | 3.687 | 4.295 | 2833 | `d40af4a0f7198cab` | `d40af4a0f7198cab` | ✅ |
| 2 | `d0/dual-s0d1seed9` | 9 | 9988 | 0.584 | 3.218 | 3.808 | 2498 | `b432486e70d1752d` | `b432486e70d1752d` | ✅ |
| 3 | `d0/dual-s0d1seed4` | 4 | 8296 | 0.587 | 2.667 | 3.259 | 2075 | `a7fc8fb3e4b75020` | `a7fc8fb3e4b75020` | ✅ |
| 4 | `d0/dual-s0d1seed16` | 16 | 11140 | 0.585 | 3.619 | 4.210 | 2786 | `c72e31b088a42647` | `c72e31b088a42647` | ✅ |
| 5 | `d0/dual-s0d1seed6` | 6 | 9428 | 0.580 | 3.020 | 3.606 | 2358 | `c3103dce2911a06b` | `c3103dce2911a06b` | ✅ |
| 6 | `d0/dual-s0d1seed8` | 8 | 12612 | 0.583 | 4.197 | 4.786 | 3154 | `014e3d9ffee02c8c` | `014e3d9ffee02c8c` | ✅ |
| 7 | `d0/dual-s0d1seed16` | 16 | 11076 | 0.577 | 3.606 | 4.190 | 2770 | `42bb158bed45a17c` | `42bb158bed45a17c` | ✅ |
| 8 | `d0/dual-s0d1seed15` | 15 | 5592 | 0.578 | 1.759 | 2.343 | 1399 | `afd6b14b35af02f2` | `afd6b14b35af02f2` | ✅ |
| 9 | `d0/dual-s0d1seed15` | 15 | 7184 | 0.591 | 2.257 | 2.854 | 1797 | `1da97e5693df91aa` | `1da97e5693df91aa` | ✅ |
| 10 | `d0/dual-s0d1seed11` | 11 | 7896 | 0.576 | 2.555 | 3.137 | 1975 | `2f1963ec4dfa6d29` | `2f1963ec4dfa6d29` | ✅ |

RESET 秒：中位 0.583，最小 0.576，最大 0.603；含快进的单次接管总计：中位 3.707，最小 2.343，最大 4.786

10/10 全部相等，进程全程 `engine_restarts=0`（没有任何静默重启），快进 94,540 帧、比对 23,645 个检查点。

### 与"每局重启进程"的对比

同一台机器、同一构建下测得的三种口径：

| 口径 | 中位 | 说明 |
|---|---|---|
| 进程内 `RESET`（本卡实测） | **0.583 s** | 10 次，0.576–0.603 |
| 每局新进程：建目录 + 启动 + `RESET`（phase E，3 次） | **1.918 s** | 1.903–1.935；建目录 0.001 s，启动 1.297 s，`RESET` 0.620 s |
| 每局重启进程（历史值，EXP-P4-001 §7.1） | **6–7 s** | 旧构建、评测器完整周期；**不是**本次测量 |

- 进程内 `RESET` **比同构建冷启动快 3.29×**（1.918 / 0.583）。
- 对照历史 6–7 s/局，进程内 `RESET` 快 **10.3–12.0×**（6/0.583、7/0.583）。
- 含快进的完整接管（`RESET` + 快进到目标帧）中位 **3.707 s**；这部分是 `TrajectoryStart` 的固有成本（要重放记录帧），与"重启"无关，DAgger 选起点时应按"目标帧越靠后越贵"估预算：本卡实测 **3,088 帧/秒**（shadow，`batch_frames=64`）。

### 4.4 其余两项自检

- **批量 vs 逐帧等价**：同一接管点（seed3 frame 11328）用 `batch_frames=64` 与 `batch_frames=1` 各回放一次，`STEP` 次数 4,343 vs 11,328，**2,833 个检查点哈希序列完全相同**（`identical=true`）。这一条证明引擎的 `STEP mask N` 等价于 N 次 `STEP mask 1`，否则"按块快进"会引入不可见的状态偏差。
- **env 角色复算**：在 `xvfb-run -a` 下用 `env`（渲染）角色复算主运行最后 2 个点（frame 9312 / 8348），哈希分别为 `31f63cb96ef1cfb9` / `1648c70caa86ef45`，与 shadow 完全相同（2/2）。软件 GL 下快进吞吐实测 79.8 / 80.1 帧/秒（shadow 是 3,088 帧/秒），每点总耗时 123.8 s / 111.4 s，所以只复算 2 点：生产像素路径与 shadow 的哈希一致，但慢约 39×，这也是 D1 量产应该用 shadow 复算、只在采样点用 env 的理由。

## 5. fail-closed 证据

### 5.1 真实引擎上的"故意给错哈希"（`EXIT=1`）

`/mnt/d/WorkShop/Touhou_RL/.tmp_v400_traj/failclosed_live.py` 读真实分片、在**不改引擎**的前提下把期望哈希改掉一位十六进制数字（flip 第 8 位），异常**不捕获**，让进程以非零码结束：

| 用例 | 目标帧 | 改动后期望哈希 | 引擎实际哈希 | 结果 |
|---|---|---|---|---|
| `control` | 13480 | `d026a2640b17462d`（未改） | `d026a2640b17462d` | 相等，`EXIT=0`，6.30 s |
| `tamper-target` | 13480 | `d026a2641b17462d`（`0`→`1`） | `d026a2640b17462d` | `TrajectoryHashMismatch`，**`EXIT=1`** |
| `tamper-mid` | 6740 | `29c0f5b406261ea9`（`a`→`0`） | `29c0f5b4a6261ea9` | `TrajectoryHashMismatch`，**`EXIT=1`** |

`tamper-target` 的原始输出（`.tmp_v400_traj/failclosed_tamper-target.log`，`wsl.exe` 只在外层包了 `NativeCommandError`，traceback 原样）：

```text
touhou_rl.v400.gym.trajectory.TrajectoryHashMismatch: trajectory take-over diverged at frame 13480
(fast-forward checkpoint): expected hash=d026a2641b17462d, observed hash=d026a2640b17462d;
refusing to hand over control from an unverified state
EXIT=1
```

`tamper-mid` 在**途中**检查点（frame 6740）就报错，说明偏差不是只在最后一帧才被发现——快进沿途每个检查点都在比。两个用例都以退出码 1 结束，没有任何一局被继续。

### 5.2 单测覆盖的 fail-closed（`tests/test_v400_trajectory_start.py`）

- 掩码长度不足：`test_start_rejects_target_beyond_the_tape`（`TrajectoryMaskError`）与 `..._with_a_counting_message`（消息里给出"2 frames"与"target_frame is 3"）。
- 目标帧越界：`test_start_rejects_negative_target`（`TrajectoryTargetError`）、`test_recorded_trajectory_start_rejects_a_frame_past_the_tape`。
- 掩码越界：`test_start_rejects_mask_out_of_range`（-1 与 0x1_0000）。
- 语料自相矛盾：`test_load_d0_shard_refuses_an_unequal_checkpoint`（A/B `equal=false`）、`..._refuses_a_broken_hash_chain`（`hash_pre` 不接上一 `hash_post`）、`..._refuses_a_label_checkpoint_without_an_ab_pair`、`..._refuses_a_decision_with_the_wrong_number_of_masks`（决策只有 3 个掩码）。
- 不可验证的哈希：`test_start_rejects_unusable_hash`（`"unknown"`、空串、`0x7f04`、15 位、非十六进制、整数）、`test_start_rejects_target_without_recorded_hash`、`test_replay_refuses_a_reset_reply_without_a_usable_hash`。
- 运行期分歧：`test_replay_reports_target_mismatch_with_frame_expected_and_observed`、`..._reports_the_first_differing_checkpoint_not_only_the_target`、`..._reports_a_reset_hash_mismatch_at_frame_zero`、`..._aborts_when_the_engine_exits_midway`、`..._aborts_on_game_over_midway`、`..._aborts_when_the_engine_comes_up_short`、`..._batch_frames_must_be_positive`。
- 参考同一帧的边界：`test_start_accepts_the_final_frame_of_the_tape`（`target == len(masks)` 合法）。

## 6. 命令与退出码

1. 主验收（WSL，`EXIT=0`，墙钟 398.6 s）：

```bash
/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/touhou_rl/v400/gym/trajectory_verify.py \
    --binary /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
    --data-template /root/v400-data/game \
    --shards-root /mnt/d/WorkShop/Touhou_RL/.tmp_v400_d0/d0/shards \
    --run-root /root/v400-traj/run2 \
    --out /mnt/d/WorkShop/Touhou_RL/.tmp_v400_traj/accept20.json \
    --points 20 --consecutive 10 --env-cross-check 2 --cold-restarts 3
```

单行汇总（节选）：`{"status": "pass", "points": {"matched": 20, "requested": 20, "frames": 181928, "checkpoints": 45502}, "consecutive": {"matched": 10, "requested": 10, "reset_s": [0.602599, 0.583808, …, 0.576094]}, "batch_equivalence": true, "env_cross_check": {"requested": 2, "matched": 2}, "cold_restart": {"median_s": 1.917656}, "reasons": []}`。

2. 补充均匀 20 点（WSL，`EXIT=0`，墙钟 97.3 s）：

```bash
/root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/touhou_rl/v400/gym/trajectory_verify.py \
    --binary /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
    --data-template /root/v400-data/game \
    --shards-root /mnt/d/WorkShop/Touhou_RL/.tmp_v400_d0/d0/shards \
    --run-root /root/v400-traj/run3 \
    --out /mnt/d/WorkShop/Touhou_RL/.tmp_v400_traj/accept20_uniform.json \
    --points 20 --consecutive 1 --bias 0 --boss-weight 0 \
    --env-cross-check 0 --cold-restarts 0 --sampling-seed 20261111
```

3. fail-closed 三个用例（WSL；`0` / `1` / `1`）：

```bash
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_traj/run_failclosed.sh control        # EXIT=0
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_traj/run_failclosed.sh tamper-target  # EXIT=1
bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_traj/run_failclosed.sh tamper-mid     # EXIT=1
```

4. 单测（Windows，`EXIT=0`）：`55 passed, 1 skipped in 0.45s`

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests/test_v400_trajectory_start.py
```

5. 全量回归（Windows，`EXIT=0`，在本次提交的树上重跑）：`1517 passed, 6 skipped, 2937 deselected in 59.06s`

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests touhou_rl/tests -k v400
```

6. 语法检查（Windows，`EXIT=0`）：`python -m compileall -q touhou_rl`。

## 7. 有效性自检（V1–V9）

| # | 检查项 | 结论与证据 |
|---|---|---|
| V1 | 输入是否包含目标信息 | 是。接管点由**语料本身的记录帧号 + 逐帧掩码 + 记录哈希**决定；验收要判的就是"引擎停在那一帧时的哈希"，三者同源。 |
| V2 | 标签/取值是否有方差、正例够不够 | 是。断开类样本（不一致）由 `tamper-*` 两个用例与单测构造；正常样本 40 个点、40/40 相等，帧覆盖 496–12,612（Q0–Q3 都有）。 |
| V3 | 数值是否物理合理 | 是。帧号 496–12,612 落在 D0 分片 13,096–13,864 帧内；`STEP` 次数 216–4,911；吞吐 3,088 帧/秒（shadow）与 T7 卡的 5,000 帧/秒同量级。 |
| V4 | 统计单位是否为独立 episode、样本量是否达标 | 是（在本任务的判据下）。20 个点分属 12 个独立分片/种子，`--consecutive 10` 是**同一进程**内的 10 次，属于"能力"而非"统计"证据；任务书要求的量（20 点 / 10 次）已满足。 |
| V5 | 平凡基线跑出来是多少 | 本卡的基线就是"每次新起一个进程"：冷启动 1.918 s/局（实测 3 次）、历史 6–7 s/局。进程内 0.583 s 明显优于此，且包含在对照表里。 |
| V6 | 训练/评估场景是否分开 | 不适用（没有训练）；接管点与验证用的进程互相独立（每个点一个全新 data-dir，phase B 用另一个 data-dir）。 |
| V7 | 换 seed 重跑是否仍成立 | 是。两次运行用不同采样种子（20261110 / 20261111）、不同采样规则，40/40 都相等。 |
| V8 | 是否混入特权信息 | 不影响本判据：`.tmp_v400_d0` 的哈希来自引擎导出的 25 字段，不是从像素推的；本卡只验证"引擎状态 == 记录状态"，不训练任何学生，因此不构成学生侧特权泄漏。D1 的学生输入仍是像素（见 T3 后续步骤）。 |
| V9 | 是否用了预注册指标 | 是。主指标 1/2/3 与阈值写在任务书 §4 T3 与驱动常量里，运行前固定，运行后未改；补充均匀 20 点是**追加**的覆盖证据，不替换主指标。 |

## 8. 限制

1. **只覆盖 Stage 1 Normal（`stage_index=0`，team 0）**。D0 只有这一面，所以 20 点的 stage 分布是 `{0: 20}`。D1 要覆盖 0–7 面，届时每个面都要重跑一次同样的验收；代码里没有面相关分支，但"没有测过"就只能写"没有测过"。
2. **哈希表粒度是每 4 帧**，不是每帧。块内（≤4 帧）的分歧只能在下一个检查点被发现，最多晚 3 帧；如果将来 D1 需要更细的定位，得在语料里加更密的检查点（代价是磁盘与比对时间）。
3. **分辨率与角色**：主指标全部在 shadow（null device）上测；生产像素路径只在 2 个点上复算过（env 角色），软件 GL 实测约 80 帧/秒，约为 shadow 的 1/39。所以"接管在 env 角色上也成立"的样本量是 2，不是 20。
4. **6–7 s/局是历史值**（EXP-P4-001 §7.1，旧构建 + 评测器完整周期），不是本次测量。本次同构建的冷启动实测是 1.918 s，因此"10.3–12.0×"这个倍数依赖历史口径；同口径比值应引用 **3.29×**（1.918 / 0.583）。
5. **批量 `STEP` 等价性只证了 1 个点**（seed3 frame 11328，2,833 检查点）。引擎实现是同一个 `PumpFrame` 循环，但"只证 1 点"就是 1 点。
6. **`game_over`/`exit` 中途终止只在假引擎单测里覆盖**：D0 的 16 条轨迹都是通关轨迹，真实引擎上没有构造出"快进途中死亡"的样本。
7. **本卡不构成 D1 本身**：D1 还需要 ≥200 万帧、8 面、起点一半为 `TrajectoryStart`、落盘 obs-v3，并按 T3 第 2 步先验收"快速教师模式"。
8. 时间数字是在**高负载**下测的（同机 20 核，主运行期间 `load1≈11.4`，另有 12 个引擎实例在跑），绝对值偏悲观；比值（3.29×）比绝对值更可靠。

## 9. 成本

- 墙钟：主验收 398.6 s + 补充 97.3 s + fail-closed 3 次约 20 s + 单测/回归约 66 s + 冒烟（前一个代理）约 50 s ≈ **11 分钟机器时间**（不含本代理读写与文档时间）。
- 引擎并发：**1**（同一时刻一个实例），远低于本任务的上限 2。
- 磁盘（新增，全部在 `.tmp_*`，不进版本库）：`/root/v400-traj/{run2,run3,failclosed}` 的 data-dir 全是符号链接 + 两个小文件拷贝，每个约 3 个目录项，**< 1 MB**；两份 JSON 报告约 **75 KB**（43,768 + 31,624 B）。
- 没有新增任何 `build/rl*`，没有改动引擎源码，没有改动教师，没有覆盖 D0。

## 10. 下一步

1. 把 `TrajectoryStart` 接进 `tools/rl/v400_d0_dual.py --start-mix`（T3 第 3 步：起点一半 `TrajectoryStart`，偏向后半段与 Boss 段）。
2. D1 生产时按 §2 的做法记录 `hash_pre`/`hash_post` + `hash_checkpoint_pairs`，并让采样只落在带哈希的帧上。
3. 8 个面各跑一次本卡的验收（`--points 20`，可在一次运行里跨分片采样）。
4. T5 的 DAgger 起点"学生自己死亡点前 5 秒"复用的是同一机制，届时需要学生轨迹的哈希表（同一套 loader + 同一套 fail-closed）。

## 证据

- `.tmp_v400_traj/accept20.json`：主运行（20 点 + 10 次连续 + 批量等价 + env 复算 + 冷启动）。
- `.tmp_v400_traj/accept20_uniform.json`：均匀补充 20 点。
- `.tmp_v400_traj/smoke1.json`：前一个代理的 2 点冒烟（SHA 与本次一致）。
- `.tmp_v400_traj/failclosed_live.py`、`run_failclosed.sh`、`failclosed_{control,tamper-target,tamper-mid}.log`。
- `.tmp_v400_traj/card_tables.py`、`card_tables.md`：本卡表格的生成脚本与产物（数字全部来自 JSON）。
- `.tmp_v400_traj/run2.log`、`run3.log`：两次运行的 stdout 与 `EXIT=`。
- 代码：`touhou_rl/v400/gym/trajectory.py`、`touhou_rl/v400/gym/trajectory_verify.py`、`tests/test_v400_trajectory_start.py`（以及 `gym` 侧新增的 `Th08Env.advance` / `FrameReply` / `last_reset_hash` / `role` 入口）。
