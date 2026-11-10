# EXP-P1-021 激光判定窗（`damageable`）导出与 teacher v2h

- **日期**：2026-11-09
- **状态**：**完成**（两道门都过）：v2h 8 面 **死亡 4/8、PSN 0.500 ≤ 0.5**；S1N **20/20 过面、每局 0 死**（≥18/20）。激光 `damageable` 导出与教师接入完成，`HASH` 前 100 帧与 `rl_hazard2` 逐字节一致，三个既有 build 未被重建。
- **上游**：[EXP-P1-016](EXP-P1-016-death-analysis.md)（死亡分析：Stage 6A 主因是 laser，死亡帧 51 条激光、gap 0.0）、[EXP-P1-020](EXP-P1-020-teacher-v2f-top-penalty.md)（v2f/v2g 均 5 死 / PSN 0.625）、[EXP-P1-012](EXP-P1-012-g1-evaluation.md)（帽表）
- **产物**：`third_party/th08-rl/src/rlharness/rlharness.cpp`（LASERS/HAZARDS 行尾追加 6 字段 + `damageable`）、`touhou_rl/v400/gym/protocol.py`、`touhou_rl/v400/teacher/{geometry,search,config,label,decide}.py`、`tools/rl/v400_teacher.py`、`tests/test_v400_teacher.py`、`build/rl_laser`、`.tmp_v400_laser*`
- **一句话**：把激光的判定时间窗（`startTime`/`hitboxStartTime`/`duration`/`despawnDuration`/`hitboxEndDelay`）和引擎自己算的 `damageable` 追加到 `LASERS`/`HAZARDS` 行尾（前缀不变、25 字段 HASH 不变、旧客户端兼容），教师据此把"有判定的激光"变成 v2a 之外**唯一**新增的终端代价项 `laser_penalty`，得到 v2h 档并重跑 8 面。

---

## 1. 目标与口径

| 项 | 口径 |
|---|---|
| 盲区 | 激光行只导出 `state`/`timer`，没有 `startTime`/`duration`/`hitboxStartTime`/`hitboxEndDelay`，客户端无法知道某条激光**这一帧是否已经有判定**、何时结束 |
| 规格依据 | 11-P1 §2.1「激光按 `state` 与 timer 处于有判定的区间计算几何」 |
| 源码依据 | `BulletManager.cpp:1096-1142`（三态判定窗）、`Player.cpp:416-440`（`CalcLaserHitbox`）、`BulletManager.hpp:313-341`（`Laser` 字段与偏移） |
| 判定来源 | `damageable` **由引擎算**，客户端不猜：`state==0 → timer>=hitboxStartTime`；`state==1 → true`；`state==2 → timer<hitboxEndDelay` |
| 8 面达标 | 8 面死亡合计 **≤4**（PSN ≤0.5） |
| 帽表 | EXP-P1-012 §2：Stage1 19,887 / Stage2 27,788 / Stage3 30,014 / Stage4A 41,583 / Stage4B 45,000 / Stage5 38,513 / Stage6A 45,000 / Stage6B 54,605 |
| 其它 | normal、seed 1、每面各自 `RESET`、无 `--follow-stage`、6 并发、每 worker 独立 data-dir |

## 2. 引擎扩展（append-only）

### 2.1 追加的 6 个字段

`LASERS` 行（G0 前缀 `slot:state:posXyz:angle:startOffset:endOffset:currentWidth:width:timer` 之后，`inUse` 之后）：

| # | 字段 | 类型 | 语义 | 源码 |
|---|---|---|---|---|
| 12 | `startTime` | i32 | `LASER_STATE_STARTING` 的淡入长度 | `BulletManager.hpp:350`（+0x570） |
| 13 | `hitboxStartTime` | i32 | 状态 0 的判定起点：`timer >= hitboxStartTime` 才调 `CalcLaserHitbox` | `BulletManager.cpp:1106-1108` |
| 14 | `duration` | i32 | 状态 1 的长度，超过即转状态 2 | `BulletManager.cpp:1126` |
| 15 | `despawnDuration` | i32 | 状态 2 的长度；为 0 时活动段结束当帧释放槽位 | `BulletManager.cpp:1129-1133` |
| 16 | `hitboxEndDelay` | i32 | 状态 2 的判定尾：`timer < hitboxEndDelay` 仍有判定 | `BulletManager.cpp:1140-1142` |
| 17 | `damageable` | 0/1 | **引擎算**：下一次 `OnUpdate` 是否会为该激光调用 `Player::CalcLaserHitbox` | 上三窗 + 导出的 `state`/`timer` |

**`damageable` 的相位**：行里导出的 `timer` 是**下一帧** `OnUpdate` 会读到的值（刚跑完的那一帧在判定之后才 `laser->timer++`，`BulletManager.cpp:1148`），所以 `damageable` 精确回答"我下一步动作将被模拟的那一帧，这条激光有没有判定"，而不是"上一帧有没有"。这条在源码里是可验证的：状态机只在状态切换时 `timer=0` 并顺序推进，因此 `(state, timer)` 唯一决定下一帧的判定分支。

### 2.2 编码与兼容

- 行尾追加，**前缀字段一字不改**；`HAZARDS` 的 laser 段与 `LASERS` 用同一个 `FormatLaserRow()` 渲染，两条通道不可能漂移（原先两处各写一份 `fprintf`/`reply.Append` 格式串，是同一个漂移风险）。
- `HASH`/25 字段完全不变：只改了激光行的**输出文本**，没有读状态、没有推进时间、没有画图、没有动 RNG。
- 旧客户端兼容：`protocol.py` 的 `LaserRow` 是长度自适应解码（12~18 字段都能解析），缺字段为 `None`；`geometry.laser_hits` 在 `damageable is None` 时回落到旧规则（只认 `LASER_STATE_ACTIVE`），并在 `skipped_inactive` 里计数。

## 3. 构建（独立目录 `build/rl_laser`）

`build/rl`、`build/rl_hazard`、`build/rl_hazard2` **只读不写**，构建前后各哈希一次。CMake 关键项照抄 `build/rl`：`CMAKE_BUILD_TYPE=Debug`、Ninja、`/usr/bin/c++`、`TH08_RL_HARNESS=ON`、`TH08_ENABLE_REPLAY_TRACE=OFF`、`TH08_DX8_SDK_ROOT=<repo>/scripts/prefix/mssdk`。

| 项 | 值 |
|---|---|
| `cmake_exit` | 0 |
| `BUILD_EXIT` | 0 |
| `build/rl/th08-modern` 前/后 | `591c521d…a25d3eb` / `591c521d…a25d3eb`（未变） |
| `build/rl_hazard/th08-modern` 前/后 | `8053dfaa…00ba7b32a` / 同（未变） |
| `build/rl_hazard2/th08-modern` 前/后 | `f172399f…47c78264e` / 同（未变） |
| `build/rl_laser/th08-modern` | `1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f` |

## 4. 引擎验证

### 4.1 HASH 前 100 帧与 `build/rl_hazard2` 一致

同一 `RESET 6 1 0 1 0`、同一 data-dir 模板、两进程逐帧 `STEP 0 1` 并逐帧比对 `HASH`：

| 项 | 值 |
|---|---|
| 比对单元 | 101（帧 0…100，含 RESET 后初值） |
| 不一致 | **0** |
| 首帧 | `OK hash=4160dcb8eb733738 frames=0`（两引擎逐字节相同） |
| 末帧（100） | `OK hash=5e878e5b1ecf1e80 frames=100`（两引擎逐字节相同） |

### 4.2 计数/字段与源码常量交叉核对

- 激光池 `0x100`（`BulletManager.hpp:456` `Laser lasers[0x100]`），`LASERS` 的 `count` 与 `HAZARDS` 的 `nl` 同一谓词 `inUse`。
- 追加字段直接取 `Laser` 结构体字段，`C_ASSERT(offsetof(Laser, startTime) == 0x570)` 等偏移见 `BulletManager.hpp:350-356`；无换算、无缩放。
- `HAZARDS` 的 laser 行与 `LASERS` 行**字段数、字段值逐字节相同**（同一渲染函数；§6 的序列里同时抓了两条通道）。

## 5. 教师接入与 v2h

| 文件 | 改动 | 为什么 |
|---|---|---|
| `gym/protocol.py` | `LaserRow` +6 字段、`windows_available`/`carries_hitbox` 两个属性、`LASER_STATE_*` | 解析追加字段且旧行仍可解析 |
| `teacher/geometry.py` | `laser_hits` 只统计 `damageable=true` 的激光；抽出 `_laser_hitbox`/`laser_rectangle`/`laser_half_thickness`；新增 `laser_gap_rows` | 判定窗由引擎给，客户端不重算；search 与 `frame_verdict` 共用一个门 |
| `teacher/search.py` | `Plan.laser`、`score_plan(laser_gap=)`、`_finish_roots(laser_rows=)`、`_laser_rows_gap`、`laser_clearance`、`beam_search`/`beam_search_reference` 传入终端帧激光 | v2a 的权重一字不改，新增的只有"终端帧离有判定激光多远"这一项 |
| `teacher/config.py` | `Weights.laser_penalty`(12.0)/`laser_band`(8.0)、`laser_penalty_at`、`V2H`、`CONFIGS`/`PROFILES` | v2h = v2a + 激光项 |
| `teacher/{label,decide}.py` | `Plan.laser` 往返/复制 | JSONL 可审计、旧 JSONL（缺 `laser`）仍可读 |
| `tools/rl/v400_teacher.py` | `bomb_rule1.chosen_laser` 观测 | 从 JSONL 就能看出"激光项是否真的计入了选中计划" |
| `tests/test_v400_teacher.py` | +6 项 | 见 §8 |

**v2h 的权重明细**：`bottom 1.5 / extra 2.0`（v2a 原样）+ `laser_penalty 12.0 / laser_band 8.0`；其余（`high 3.0`、`edge 1.0`、`align 4.0`、`margin_cap 16`、`rule1 12`、无 `clip_y`、无 keep-last-bomb）全部与 v2a 相同。终端帧落在有判定激光的矩形内（gap 0）付 12.0 分 = margin 上限 16 的 75 %；8 px 之外付 0。

**为什么是"终端位姿代价"而不是把激光塞进逐步碰撞检测**：`HazardTape.box_array()`（search 的热路径）只返回子弹/敌机 AABB，激光是旋转矩形；逐帧逐节点做旋转测试会把每次扩展的代价乘上激光数（Stage 6B 实测单帧最多 190 条）。终端帧一项与 EXP-P1-017 的 Boss 项同构，代价可忽略（§7 计时），且 `damageable=0` 的淡入激光不收费——正是任务要求的"不要过度保守"。

## 6. 真实引擎上的 `damageable` 序列（Stage 6B，帧 17586-17611）

因为 **Stage 6A 本次一条激光都没有**（§6.1），真正的激光证据取自本次 8 面里激光最密的 Stage 6B（engine stage index 7）。做法：把 v2h Stage 6B 的 JSONL 的 `step_masks` 逐帧回放到新引擎，逐帧抓 `HAZARDS`，并**逐决策边界比对引擎 `PLAYER` 的 px/py 与 JSONL 记录的 `state.px/py`**：4,411 个决策 / 17,644 帧，**0 个 mismatch**，即回放与实跑同轨。

回放汇总：`frames_with_lasers=933`、`frames_with_damageable=874`、`max_nl=190`、首个有判定激光帧 `14,080`。

**激光行状态分布**（全部 17,644 帧）：`state=1,damageable=1` **103,582** 行；`state=0,damageable=0` **1,947** 行；`state=0,damageable=1` **33** 行；`state=2` 0 行。注意中间那两行：淡入期（`timer < hitboxStartTime`）引擎自己说"没判定"，而 **33 行是"还在 state 0 但 timer 已到 hitboxStartTime"** ——旧规则"只看 `state==1`"会把它们判成无判定，这正是本轮要修的盲区。

连续 26 帧（slot 0，`startTime=60 hitboxStartTime=60 duration=120 despawnDuration=20 hitboxEndDelay=20`）：

| frame | state | timer | damageable |
|---|---|---|---|
| 17586 | 0 | 56 | **0** |
| 17587 | 0 | 57 | **0** |
| 17588 | 0 | 58 | **0** |
| 17589 | 0 | 59 | **0** |
| 17590 | 0 | **60** | **1** ← `timer == hitboxStartTime`，判定开启 |
| 17591 | 1 | 1 | **1** ← 状态切到 ACTIVE，timer 归零 |
| 17592 | 1 | 2 | **1** |
| 17593 | 1 | 3 | **1** |
| 17594 | 1 | 4 | **1** |
| 17595 | 1 | 5 | **1** |
| 17596 | 1 | 6 | **1** |
| 17597 | 1 | 7 | **1** |
| 17598 | 1 | 8 | **1** |
| 17599 | 1 | 9 | **1** |
| 17600 | 1 | 10 | **1** |
| 17601 | 1 | 11 | **1** |
| 17602 | 1 | 12 | **1** |
| 17603 | 1 | 13 | **1** |
| 17604 | 1 | 14 | **1** |
| 17605 | 1 | 15 | **1** |
| 17606 | 1 | 16 | **1** |
| 17607 | 1 | 17 | **1** |
| 17608 | 1 | 18 | **1** |
| 17609 | 1 | 19 | **1** |
| 17610 | 1 | 20 | **1** |
| 17611 | 1 | 21 | **1** |

### 6.1 Stage 6A 没有激光（对 EXP-P1-016 前提的纠正）

- v2h Stage 6A JSONL（3,455 决策 / 13,820 帧）逐帧回放，fidelity 0 mismatch：**每一帧 `nl=0`**，`frames_with_lasers=0`。
- 两次直走探针（`RESET 6 1 0 1 0`，STEP 30 帧一批）：idle 30,000 帧、按住 SHOOT 30,000 帧，`LASERS count` 始终 0，`HAZARDS` 的 `nl=0`。
- 解析器自检：同一帧 `HAZARDS` 的 `nl` 与 `lasers=` 段的解析结果一致（`nl=0` ↔ 空段），所以"没有激光"不是解析 bug。
- 源码可达性：`EclRunHigh.inl:392` 的 `ECL_OPCODE_CREATE_LASER(_AIMED)` 会调 `g_BulletManager.SpawnLaserPattern()`（`BulletManager.cpp:719`），激光是**能**生成的；Stage 6A 本次只是没走到那些 opcode。
- EXP-P1-016 自己的产物内部不一致：`.tmp_v400_death/psn_stage6/frames.jsonl` 的 179 帧里 `hazards.lasers` 全为 0、`verdict.lasers_considered=0`，但 `death.json` 的 evidence 写 `frames_with_laser=10 / nearest_gap_px=0.0`。**"51 条激光"在本引擎与本次语料上不可复现**，建议把 EXP-P1-016 的"Stage6A 主因 laser"降级为待复核。

## 7. v2h 8 面重测（全 8 面，各 1 局，6 并发）

基线取 **v2a**（不是 v2f）：v2h 的定义就是 "v2a 权重 + 激光项"，只有拿 v2a 比，差异才能归因到激光。`v2f` 的三列也一并列出，因为它决定 P1 的门以前卡在哪。

| stage | face | cap | clear | frames | deaths | stop | exit | wall s | 选中计划被收激光费 | 最大激光分 | v2a frames | v2a deaths | v2f frames | v2f deaths |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Stage1 | 19,887 | **true** | 13,484 | **0** | stage_changed | 0 | 704 | 0 / 3371 | 0.00 | 13,484 | 0 | 13,484 | 0 |
| 1 | Stage2 | 27,788 | **true** | 21,740 | **0** | stage_changed | 0 | 1165 | 0 / 5435 | 0.00 | 21,740 | 0 | 21,740 | 0 |
| 2 | Stage3 | 30,014 | **true** | 26,828 | **0** | stage_changed | 0 | 1913 | 4 / 6707 | 0.36 | **25,052** | **1** | 23,172 | 1 |
| 3 | Stage4A | 41,583 | **true** | 30,952 | **0** | stage_changed | 0 | 2692 | 0 / 7738 | 0.00 | 30,952 | 0 | 29,564 | 0 |
| 4 | Stage4B | 45,000 | false | 17,364 | **1** | death | 0 | 1314 | 0 / 4341 | 0.00 | 17,364 | 1 | 24,068 | 1 |
| 5 | Stage5 | 38,513 | false | 21,504 | **1** | death | 0 | 1903 | 0 / 5376 | 0.00 | 21,504 | 1 | 21,660 | 1 |
| 6 | Stage6A | 45,000 | false | 13,820 | **1** | death | 0 | 1134 | 0 / 3455 | 0.00 | 13,820 | 1 | 19,584 | 1 |
| 7 | Stage6B | 54,605 | false | 17,644 | **1** | death | 0 | 1473 | 17 / 4411 | 7.55 | 17,564 | 1 | 17,564 | 1 |

- **死亡合计 4 / 8 → PSN = 0.500（目标 ≤0.5，达标）**；clear 4 / 8；8/8 面都是真实 JSONL、`exit=0`、引擎 SHA 一致（`1462829b…`）；整轮 2,845 s。
- **归因**：与 v2a 相比，8 面里 **6 面的帧数逐帧相同**（Stage1/2/4A/4B/5/6A 完全相同），只有两面变了 —— 正是两面有激光的 Stage3 与 Stage6B。**Stage3 从 25,052 帧 / 1 死变成 26,828 帧 / clear**，Stage6B 换了走法（17,564 → 17,644）但仍死 1 次。也就是说：v2a 的 5 死里那 1 个 Stage3 死亡被激光项直接救回来了，PSN 0.625 → 0.500，`v2h` 恰好把 P1 的门压到达标线上。
- 与 v2f 的差别（Stage4B/5/6A/6B 帧数不同）来自 v2f 的 top/corner 权重，与激光无关；v2f 的 5 死里 Stage3 也是死 —— 说明"顶部带"没能救 Stage3，激光项救了。

### 7.1 瓶颈与下一步

1. **Stage4B / Stage5 / Stage6A / Stage6B 各死 1 次**，且都不在 cap 上（17k–21k / 45k–54k 帽）。其中 Stage6A **本次一条激光都没有**（§6.1），说明这面的死因不是激光；Stage4B/5 也没有激光行。激光项能救的是"终端帧落在光束里"的候选，Stage3 就是这类。
2. 要把激光变成**硬约束**（而不只是终端代价）得把它放进逐步碰撞检测：`HazardTape.box_array()` 只有 AABB，激光是旋转矩形，逐帧逐节点测试的代价乘以激光数（Stage6B 实测单帧最多 190 条），需要先做向量化预算（EXP-P1-010 的 raster 表是同一条路）。
3. 剩余 4 个死亡点需要新的**死因**证据（`tools/rl/v400_death_analysis.py` 的 `laser` 分支在本次语料上应该全部为 0，值得重跑一次确认标签不再误判）。


## 8. 测试

| 项 | 命令 | 结果 |
|---|---|---|
| 新增 6 项（damageable 过滤 / 字段与旧行兼容 / v2h 档 / 激光几何 gap 一致性 / 零权重逐位等于 v2a / label 往返） | `pytest tests/test_v400_teacher.py -k 'laser or v2h'` | **6 passed** |
| 教师全量 | `pytest tests/test_v400_teacher.py` | **111 项：110 passed + 1 failed**（`test_v3_search_stays_within_the_measured_performance_ceiling`，比值断言在 6 worker 并发下抖动；**单独重跑 1 passed in 3.26 s**） |
| 邻居（几何/labels/hazards/protocol） | `pytest tests/test_v400_geometry_agreement.py tests/test_v400_labels.py tests/test_v400_hazards.py tests/test_v400_protocol.py` | 98 passed，2 failed（`ModuleNotFoundError: No module named 'PIL'`，WSL venv 缺 Pillow，与本次改动无关） |
| 语法 | `python -m compileall -q touhou_rl tools/rl` | exit 0 |

## 9. 命令与退出码

| # | 命令 | 结果 |
|---|---|---|
| 1 | `python3 .tmp_v400_laser_patch.py` | rc 0；`[ok] doc-comment / helper / lasers-row / hazards-row` |
| 2 | `sh .tmp_v400_laser_build.sh` | `cmake_exit=0` / `BUILD_EXIT=0`；三个既有 build SHA 前后不变；新二进制 `1462829b…d2f` |
| 3 | `python .tmp_v400_laser_verify.py --parity --frames 100 --stage 6` | rc 0；**101 比对点 / 0 不一致** |
| 4 | `python .tmp_v400_laser_verify.py --stage6a --probe 24000/30000（idle 与 shoot）` | rc 0；两次都是 `first_laser_frame=null`（Stage6A 无激光） |
| 5 | `python .tmp_v400_laser_parser.py` | rc 0；`nl=0` 与解析段一致（解析器自检） |
| 6 | `bash .tmp_v400_laser_smoke.sh`（400 帧/15 决策，stage 6，v2h，新引擎） | `SMOKE_EXIT=0`，`sha256 match`，`mode=v2h` |
| 7 | `bash .tmp_v400_laser_round.sh`（8 面，v2h，6 worker，timeout 9000，21:07:28→21:54:53） | `ROUND_EXIT=0`，8/8 面 `exit=0`，整轮 2,845 s |
| 8 | `python .tmp_v400_laser_table.py round1-v2h round1-v2f` | rc 0；写 `.tmp_v400_laser/table-round1-v2h.md`（§7 表） |
| 9 | `python .tmp_v400_laser_replay.py --jsonl …psn_stage7.jsonl --stage 7` | rc 0；17,644 帧、933 帧有激光、fidelity 0 mismatch |
| 10 | `python .tmp_v400_laser_flip.py .tmp_v400_laser/stage6b-replay.json` | rc 0；状态分布 + §6 的 26 帧表 |
| 11 | `python .tmp_v400_laser_replay.py --jsonl …psn_stage6.jsonl --stage 6` | rc 0；13,820 帧、0 激光、fidelity 0 mismatch（§6.1） |
| 12 | `bash .tmp_v400_laser_s1n.sh`（20 局，stage 0，cap 19,887，6 并发，22:03:01 起） | 见 §9.1 |

### 9.1 S1N 闸门（seeds 1000-1019，stage 0，cap 19,887，6 并发，独立 data-dir）

**第一次启动（22:02:09，已作废）**用了一个尚未准备的 data-dir 根
（`--data-dir-root /root/v400-data/s1n-laser`），6 个槽里 3 个直接 `exit=5`
（`engine_error: game data dir not found: …/inst1|inst2|inst4`），单局 0.36 s。
这一批**没有进入任何统计**：`runs.jsonl` 在正式启动前被清空（§9 第 12 行），
`runs.jsonl` 只保留正式批的记录。修法有两条，都落地了：

1. `.tmp_v400_laser_datadirs.sh <root> <slots>` 先建好**每个槽位**（`score.dat`/`th08.cfg`
   从 `*.orig` 复制、`th08.dat`/`thbgm.dat` 只读符号链接），已对
   `/root/v400-data/s1n-laser/inst0..5` 执行（`prepared` ×6）；
2. 正式批改用已验证的 `/root/v400-data/g1/inst0..5`（8 面那一轮同款、当时 8/8 面都跑通）。

正式批 22:03:01 启动，`runs.jsonl` 逐局追加、resume 语义（`done_cases` 跳过已完成局），
日志里 `engine_error` 计数 **0**。**申报 20 局 / 实际跑完**两个分母在 §9.2 一并给出。
旧日志文件 `.tmp_v400_laser/s1n/round1-v2h/logs/s1n_seed10XX.log` 里若还能看到
`game data dir not found`，那是 22:02 那次作废批留下的；该 seed 正式跑完后会被覆盖。

正式批 22:03:01 → 22:56:45（3,224 s），**20/20 局全部跑完，全部 clear 且 0 死**。逐局：

| seed | exit | clear | frames | deaths | stop | wall s |
|---|---|---|---|---|---|---|
| 1000 | 0 | **true** | 13,484 | **0** | stage_changed | 874 |
| 1001 | 0 | **true** | 13,676 | **0** | stage_changed | 890 |
| 1002 | 0 | **true** | 13,292 | **0** | stage_changed | 868 |
| 1003 | 0 | **true** | 13,292 | **0** | stage_changed | 865 |
| 1004 | 0 | **true** | 13,292 | **0** | stage_changed | 864 |
| 1005 | 0 | **true** | 13,388 | **0** | stage_changed | 874 |
| 1006 | 0 | **true** | 13,484 | **0** | stage_changed | 831 |
| 1007 | 0 | **true** | 13,196 | **0** | stage_changed | 804 |
| 1008 | 0 | **true** | 13,484 | **0** | stage_changed | 827 |
| 1009 | 0 | **true** | 13,868 | **0** | stage_changed | 849 |
| 1010 | 0 | **true** | 13,676 | **0** | stage_changed | 840 |
| 1011 | 0 | **true** | 13,580 | **0** | stage_changed | 833 |
| 1012 | 0 | **true** | 13,580 | **0** | stage_changed | 810 |
| 1013 | 0 | **true** | 13,964 | **0** | stage_changed | 844 |
| 1014 | 0 | **true** | 13,676 | **0** | stage_changed | 830 |
| 1015 | 0 | **true** | 13,388 | **0** | stage_changed | 810 |
| 1016 | 0 | **true** | 13,580 | **0** | stage_changed | 818 |
| 1017 | 0 | **true** | 13,484 | **0** | stage_changed | 807 |
| 1018 | 0 | **true** | 13,868 | **0** | stage_changed | 540 |
| 1019 | 0 | **true** | 13,388 | **0** | stage_changed | 496 |

### 9.2 两个分母与区间（`S1N_EXIT=0`）

| 项 | 值 |
|---|---|
| 申报局数（分母 B） | **20**（seeds 1000-1019，无缺失） |
| 实际跑完（分母 A） | **20**（`completed=true` 且 `exit_code=0`，每局都有真实 JSONL） |
| 失败/超时/missing | **0**（`engine_error` 计数 0；缺 seed 列表 `none`） |
| 过面（`stage_clear` 且 `deaths==0`） | **20 / 20** |
| 过关率 | 分母 A **1.000**、分母 B **1.000**，95 % Wilson **0.839–1.000** |
| 帧数 | min 13,196 / mean 13,532.0 / max 13,964 |
| 闸门 | ≥18/20 过面 **且每局 0 死** → **过**（配套 PSN 4/8 = 0.500 ≤ 0.5 也过） |

### 9.3 仍开放的项

1. **Stage 6A 没有激光**（§6.1）：本轮的 v2h Stage6A 13,820 帧、两次 30,000 帧直走探针都没有一条激光行，且 EXP-P1-016 自己的 `frames.jsonl` 与 `death.json` 互相矛盾。要把"激光击杀"这条前提做实，需要一次真正打到 Stage 6A 激光阶段的 run（例如换 seed / 更高 cap / 更稳的走法），再跑本卡的 replay 取证。
2. **Stage4B / Stage5 / Stage6A / Stage6B 各 1 死**：需要新的死因证据；`tools/rl/v400_death_analysis.py` 的 `laser` 分支在本次语料上应当全为 0，重跑一次可确认标签不再误判。
3. `tools/rl/v400_geometry_agreement.py` 与 `tools/rl/v400_p1_laser_gate.py` 自己按索引解析 12 字段激光行，因而仍走旧的 `state==ACTIVE` 回退（不读 `damageable`）。它们不在本卡的允许修改范围内，此处只做记录。




