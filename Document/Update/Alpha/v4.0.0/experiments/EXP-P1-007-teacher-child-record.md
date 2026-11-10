# EXP-P1-007 教师切子进程录制：1.6 s 还原归零、EVENTS 决死 bomb、stage 1 关卡级首个真实结果

- 日期 / 工作包：2026-11-09 / P1 教师（[11-P1](<../../11-P1-搜索教师.md>) §2、§4、§5；承接 EXP-P1-003 的三段计时与 EXP-P1-006 的性能/还原实测）
- 目标：
  1. 引擎切到 `build/rl_hazard2`（SHA `f172399f…`），CLI 启动即校验 SHA，不符 fail-closed 退出；
  2. 录制改到 **fork 出的子进程**：`PARK` 一次 → 每决策 `SPAWN` 一个子进程 → 子在原地逐帧 `STEP` + `CHILD HAZARDS`/`CHILD PLAYER` 录 H 帧 → 子丢弃；主进程每决策只前进 R=2 个子决策（4 帧），**永不 RESET**；
  3. 接 `EVENTS`，实现 §4 两条 bomb 规则，label 记 `bomb_death_available` 与 `frameEventFlags`；`ERR events-unavailable` 时如实记 `false`；
  4. 同一批 20 个决策跑「旧路径（legacy restore）」与「新路径（child recording）」，逐项比 `geo_score[18]` / `chosen_plan` / 判死结论；
  5. 用新路径跑 stage 1 normal seed 1，直到「死亡 / 过面 / 3,000 决策（约 6,000 帧）」任一条件，报关卡级真实结果。
- 预注册（运行前固定，全部可证伪）：
  1. **还原归零**：新路径单决策无「进程启动 + RESET + 重放」，进程只启动 1 次；若仍出现每决策 1.6 s 量级的还原，则本卡失败；
  2. **录制等价**：同一决策点、同一世界下，子进程录制的 `HazardTape` 必须与主进程录制逐行相同（帧号、hash、每帧致死行、PLAYER 行、mask）；
  3. **决策等价**：20 个决策的 `geo_score[18]`、`chosen_plan`、判死结论必须逐项一致（位置/死亡允许 fork 晚 1 帧的已知偏差，判死结论不允许不一致）；
  4. **子命令批量化**：48 帧录制的 144 条子命令必须**一次** `ask_many` 往返，不得逐条等回包；
  5. **fail-closed**：`probe_child_hazards()` 为 false 时不得静默回退主进程录制，必须非零退出并说明，除非显式 `--allow-legacy-restore`；
  6. **bomb 不伪造**：`EVENTS` 不可用时 `bomb_death_available=false`、`frameEventFlags=null`，不得用任何相关量替代。
- 样本量：正确性 A/B = stage 1 normal seed 1 的 **20 决策**（80 帧），新旧两条路径各一次；关卡级 = 同一关卡 **1,500 决策 / 6,000 帧**（帧上限先到）；单测 52 项（本文件 44 + 导入回归 8）。
- 环境：`/root/Touhou_RL/third_party/th08-rl/build/rl_hazard2/th08-modern`（`sha256=f172399f61b75baa8e62be46041ef756a710759f439e7fb07111f4747c78264e`，未重建），数据目录 `/root/v400-data/p1s1/game`，WSL 内一次只跑一个引擎进程。
- **负载说明**：运行期间 Windows 侧的零售 trace 批跑同时在跑（按要求不干扰），所以本卡 §5 的绝对耗时高于隔离测量；§3 的 A/B 数值与 §5 的 1,500 决策数值都照实列出，不混用。

---

## 1. 新架构（文字图）

```text
进程启动 1 次（1179–1280 ms）+ RESET 1 次（403–420 ms）
  │
  ├─ 决策 k（主进程停在决策点，frame f；它从 RESET 后只前进过 4k 帧）
  │    ├─ EVENTS            1 次往返，读“主环境最后一帧”的 frameEventFlags（§4 规则 2 的输入）
  │    ├─ PARK             fork 出模板 = 决策点快照（~0.3 ms）；下一次 PARK 释放它
  │    ├─ SPAWN 1 个录制子进程
  │    │     └─ 子供进程原地执行：HAZARDS, PLAYER, (STEP 1, HAZARDS, PLAYER) × (H-1)
  │    │        共 1+2+3(H-1) = 144 条子命令，**一次 ask_many 写入、按序读回**
  │    ├─ CHILD <h> QUIT    录制子进程用完即丢弃（模板仍有效）
  │    ├─ 束搜索            tape → geo_score[18]
  │    ├─ 验证             从**同一个模板** SPAWN k 个 fork，逐个 CHILD STEP 2 + STATE，判死用引擎自己的 deaths 计数
  │    └─ commit           R=2 个子决策 = 4 帧，直接在主进程 STEP（bomb 位只挂在第 1 个子决策）
  └─ 决策 k+1 在 frame f+4，主进程**从不 RESET**
```

与旧路径（`--allow-legacy-restore`）的差别，只在一处：旧路径每个决策 `新进程(1180 ms) → RESET(415 ms) → 重放已提交动作 → 在主进程录制 H 帧 → 提交 4 帧 → 丢弃进程`。新路径把「录制」从权威进程搬到 fork，于是「还原」这一整段不再存在；主进程的 6,000 帧只被推进 1,500 次 × 4 帧，一次 RESET 都没做。

代码落点：

| 文件 | 改动 |
|---|---|
| `touhou_rl/v400/teacher/engine.py` | `EXPECTED_ENGINE_SHA256` / `sha256_file` / `check_engine_binary`（SHA 门）；`events()` / `events_snapshot()` / `child_events()`（`EVENTS` 取不到时返回 `available=false` + 引擎原文，不伪造 0） |
| `touhou_rl/v400/teacher/child_record.py` | **新增** `record_child_tape()`：探测能力 → `SPAWN` → 144 条 `CHILD` 命令一次批量 → 解析成 `HazardTape` → `CHILD QUIT`；返回 `ChildRecord`（命令数、回包字节、spawn/advance 耗时） |
| `touhou_rl/v400/teacher/hazard_tape.py` | `source="child"` 的 tape 组装改成**逐帧 PLAYER**（原来只取第 0 帧快照填满全表），行 0 仍是决策点（先 HAZARDS/PLAYER 再 STEP） |
| `touhou_rl/v400/teacher/decide.py` | `record_source="main"|"child"`；决策点先读一次 `EVENTS`；`record` 段含 `PARK`；`death_bomb_window()` / `death_bomb_feasible()`；§4 两条 bomb 规则；`restore()` 支持 `(cand, bomb)` 历史（bomb 位会改变世界，重放必须带上） |
| `touhou_rl/v400/teacher/label.py` | label 增 `frame_event_flags`（`None` = 引擎没答） |
| `tools/rl/v400_teacher.py` | `--engine`（默认 `rl_hazard2`）+ `--expected-sha` + SHA fail-closed；默认 child 路径、`--allow-legacy-restore` 才走旧路径；`--max-frames` / `--stop-on-death`；四段计时 + `EVENTS` 子项；按 `STATE` 的 `f1`（stage）与 `f3`（gameplayFrameCounter）判过面；实时倍率口径改成「已提交帧数 / 墙钟 / 60」 |

### 1.1 SHA 门（启动即 fail-closed）

| 输入 | 结果 |
|---|---|
| `--engine /root/Touhou_RL/third_party/th08-rl/build/rl_hazard2`（SHA `f172399f…`） | 通过，`engine.match=true`，报告里记 `path/sha256/expected` |
| `--engine .../build/rl_hazard`（SHA `8053dfaa…`） | **退出码 4**，stdout：`{"status":"engine_sha_mismatch","reason":"engine SHA mismatch … sha256=8053dfaa… expected=f172399f…"}`；**没有启动任何引擎进程** |
| 不存在的 build 目录 | 退出码 4（`engine binary not found`） |

`--probe-only` 在同一次启动里记录 `child_hazards: {available: true, reason: "CHILD <h> HAZARDS accepted", reply: "OK frames=0 nb=1 nl=0 ne=1 …"}`、`child_state_ok: true`、`state_has_frame_event_flags: false`（`STATE` 仍不导出事件字，事件只走 `EVENTS`）。

---

## 2. 子进程录制：144 条命令一次往返

单命令隔离成本（同一进程内实测，48 帧样本）：

| 命令 | 每次耗时 |
|---|---|
| `CHILD <h> HAZARDS` | **0.1248 ms** |
| `CHILD <h> STEP <mask> 1` | **0.2347 ms** |
| 主进程 `HAZARDS`（对照） | 0.0948 ms |
| `PARK` + `SPAWN`（模板） | ~0.6 ms |

一次 48 帧录制的批量调用：**144 条命令 / 28,040 字节回包 / 17.75 ms**（含子进程 47 次 `STEP 1` 的真实推进）。子进程在 48 帧里的最大单条回包 329 字节（stage 1 早期），远低于引擎 `HAZARDS_REPLY_CAPACITY=256 KiB`；引擎的 `CHILD` 转发走 `ForwardLine()` 流式逐块转发，>4 KiB 的 `HAZARDS` 不会被 4 KiB 缓冲截断（`fork.cpp` 实测源码 + 真机）。

等价性（真机探针，同一次 `PARK`）：`CHILD <h> HASH` 与主进程 `HASH` 在决策点相同（`fc599ed82436ae5e`）；双方各 `STEP 2`×2 后 `hash`（`357e22f59f661f1d`→`68ce53fa1dc991a4`）与 `STATE`（`f0..f24` 全 25 个字段）**逐字节相同**。子进程的 `EVENTS` 也可读（`frameEventFlags=1400`），本次没有用到（§4 的规则读的是主环境）。

---

## 3. 四段耗时与实时倍率

口径：**1 决策 = R=2 个子决策 = 4 帧**，所以 1× 实时的门槛是 60/4 = **66.7 ms/决策**；实时倍率 = `已提交帧数 ÷ 墙钟秒数 ÷ 60`。

### 3.1 A/B 那 20 个决策（隔离、同一引擎、同一配置）

| 段 | child recording | legacy restore | 说明 |
|---|---|---|---|
| record（含 `PARK`） | **18.32 ms** | 12.43 ms | 子进程多 145 条两跳命令；旧路径在主进程上一条跳 |
| search | **17.59 ms** | 18.59 ms | 同一 tape、同一搜索 |
| verify | **16.86 ms** | 18.34 ms | 同一模板、同一批量子命令 |
| commit（2×`STEP 2`+`STATE`） | **0.62 ms** | 0.47 ms | 主进程只前进 4 帧 |
| `EVENTS`（子项，另计） | **0.04 ms** | 0.05 ms | 每决策 1 次 |
| **合计（四段 + EVENTS）** | **53.44 ms** | **1652.30 ms** | 旧路径另含 restore 1615.19 ms |
| restore | **0 ms** | 1615.19 ms | 旧路径 = 启动+HELLO 1186.4 + RESET 425.7 + 重放 3.1 |
| 决策/秒 | 18.69 | 0.600 | |
| **实时倍率** | **1.246×** | **0.040×** | 另一轮同配置 child 为 51.13 ms → **1.302×** |

结论：**算法四段 53.4 ms < 66.7 ms → 1.25× 实时**，与 EXP-P1-006 的算法三段 51.53 ms / 1.294× 同量级（record 因两跳子命令 +6.3 ms，其余三段因负载波动 ±1 ms）；**还原从 1615 ms 降到 0**，这是本卡的主目标，已达成。

### 3.2 1,500 决策的关卡级运行（Windows 零售 trace 批跑同时在跑）

| 段（每决策） | 均值 | p50 | p90 | max |
|---|---|---|---|---|
| record | 45.62 | 41.47 | 69.48 | 180.80 |
| search | **430.17** | 395.68 | 810.65 | 2400.19 |
| verify | 46.25 | 47.13 | 71.27 | 142.93 |
| commit | 3.12 | 2.75 | 5.22 | 14.77 |
| `EVENTS` | 0.20 | 0.09 | 0.18 | 7.00 |
| **total** | **526.07** | 496.25 | 937.77 | 2566.59 |

- 决策/秒 **1.90**，已提交帧/秒 **7.60**，**实时倍率 0.127×**；墙钟 **789.1 s** 对应游戏内 **100.0 s**。
- 同一运行里按 tape 是否有致死 AABB 分层：**空 tape 206 个决策**（`total` 均值 116.6 ms，其中 record 31.5 / search 38.9 / verify 42.9 / commit 2.7），**有弹幕 1,294 个决策**（`total` 均值 591.3 ms，其中 search 492.5 ms）。也就是说 1.25× 的结论只对“屏幕上没有致死弹幕”的早期成立；真实弹幕一出现，搜索立刻成为唯一瓶颈（占单决策 82%）。
- 空 tape 的 116.6 ms 仍高于 §3.1 的 53.4 ms，原因是同时段 Windows 零售实例在跑（CPU 争用），不是代码回退；两套数字分开列，不合并。

---

## 4. 与 legacy 的一致性（20 决策 A/B）

同一引擎、同一数据目录、同一配置（`--mode full --stage 1 --difficulty normal --seed 1 --decisions 20`），只有录制路径不同；两次运行各自从 RESET 起，逐决策比较 JSONL。

| 比较项 | 结果 |
|---|---|
| `geo_score[18]`（逐元素，4 位小数序列化后） | **20/20 完全相同** |
| `chosen_plan`（first_action/duration/slots/margin/death_frame/end_xy/score/align/edge/high/verified/geo_score） | **20/20 相同** |
| 判死结论（每个已验证计划的 `first_action, died, death_frame`） | **20/20 相同**（共 60/60 个计划） |
| `bomb` / `bomb_death_available` | **20/20 相同**（全 0 / 全 true） |
| 首个分歧 | **无（0 处分歧）** |
| 已提交帧数 | 两侧 80 帧 |
| `engine_stats` | child：**1 个进程** / 3,572 次调用；legacy：**20 个进程** |
| 一致性计数（两侧相同） | `agree_alive 60`、`agree_dead 0`、`geo_dead_engine_alive 0`、`geo_alive_engine_dead 0`、`trajectory_mismatch 60/60`（= EXP-P1-006 的 fork 晚 1 帧、4.0 px 已知偏差，未修） |

诚实边界：这 20 个决策落在 stage 1 的前 80 帧，**没有一发致死弹幕，也没有一次死亡**，所以「判死一致」在这段样本上只证明了“两侧都判活”。真实死亡的判死一致性由两处覆盖：① 单测 `test_death_verdict_agrees_between_child_recording_and_legacy_restore`（fake 引擎在 `kills_at=3` 时真的把每个 rollout 杀死，两条录制路径给出相同的 `(first_action, died, death_frame)` 与相同的 `bomb=1`）；② §5 的 1,500 决策运行里有 **4 个待验计划被引擎判死**（`geo_alive_engine_dead=4`），说明验证路径在真实弹幕下确实在判死，只是它们没落在 A/B 窗口里。

---

## 5. EVENTS 与 §4 两条 bomb 规则

实现（`decide.py`）：

1. 决策点先读**一次** `EVENTS`（在任何推进之前；`EVENTS` 是“最后一帧”的字，引擎每帧开头清零）。label 记 `frame_event_flags`（原始 u16）、`bomb_death_available`、`extra.event_names`；
2. **规则 1（主动 bomb）**：第 5 步发现**所有已验证计划都会死**（引擎判死，不是几何判死）且 `bombs > 0` → `bomb=1`，事件 `active_bomb`；
3. **规则 2（决死 bomb）**：`frameEventFlags & 0x2`（`PLAYER_HIT`）且 `bombs > 0` 且窗口够 → `bomb=1`，事件 `death_bomb_player_hit`。窗口按 `Player.cpp:534-590` 的**下界**实现：`bombs×6` 上限 15；时间点加成、符卡翻倍、机体 0/4/5 的 ×9/5 都不从协议导出，缺省不加（只会把窗口算小，方向保守）；再减 1 帧输入延迟，`≤2` 帧（无 bomb 情形）判定为来不及，只能靠规则 1；
4. `EVENTS` 返回 `ERR events-unavailable`（或旧引擎 `ERR unknown-command`）→ `bomb_death_available=false`、`frame_event_flags=null`、事件记 `death_bomb_unavailable:<引擎原文>`，**不放 bomb、不伪造 0**。

真机证据：

| 项 | 结果 |
|---|---|
| A/B 20 决策 | `EVENTS` 20/20 可用，`PLAYER_HIT` **0** 次；`bomb=0` 两侧一致 |
| 关卡级 1,500 决策 | `EVENTS` **1,500/1,500** 可用；采样到的 `frameEventFlags` 集合 = {0, 0x20, 0x40, 0x400, 0x420, 0x440, 0x460, 0x800, 0x820…0x3400}，**含 0x2 的帧 = 0** → 全程没有触发决死 bomb |
| 4 个被引擎判死的待验计划 | 出现在决策 550（首动作 1、2，death_frame 43）与 564（首动作 16、6，death_frame 33）；每个决策 3 个验证计划里只有 2 个死 → 不属于“全部会死”，因此**没有**触发规则 1；当时 `PLAYER_HIT=0`，也没有触发规则 2 → `bomb=0`，与 label 一致 |

一处规则修正（本卡范围内）：规则 1 原来用 `chosen.death_frame is not None`（**几何**判死）作条件，于是“引擎判死但几何判活”的计划（正是 `geo_alive_engine_dead` 这一类）永远不会触发主动 bomb。现已改成“所有**已验证**计划都死”。改前改后都在单测里锁住（fake 引擎 `wall=True, kills_at=0` 时几何与引擎同时判死；`kills_at=3` 且无弹幕时只有引擎判死，改前 `bomb=0`、改后 `bomb=1`）。

---

## 6. stage 1 关卡级真实结果（首个）

命令（WSL 侧，一次一个引擎；Windows 零售实例未动）：

```powershell
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_teacher.py \
  --decisions 3000 --max-frames 6000 --stop-on-death \
  --stage 1 --difficulty normal --seed 1 --mode full \
  --out .tmp_v400_teacher_child_stage1.jsonl > .tmp_v400_teacher_child_stage1.log 2>&1"
```

| 指标 | 实测 | 判据/说明 |
|---|---|---|
| 停止原因 | **`max_frames`** | 帧上限 6,000 先到；决策上限 3,000 未到（跑了 1,500） |
| 决策数 | **1,500** | 每决策 4 帧 |
| **存活帧数** | **6,000 帧**（= 100.0 s 游戏内时间） | `frames_committed`；主进程 `STATE f0` 从 0 → 6,000，`state_frame_delta=6000` 与之一致 |
| **是否过面** | **否** | `stage_clear=false`；`STATE f1`（stage）在停止时仍 = **1**，`f3`（gameplayFrameCounter）= **6,002**（未重置、未换面）→ 6,000 帧内没有 stage 变化 |
| **死亡次数** | **0** | `STATE f6`（deaths）= 0.0 全程；`lives` 2.0 未变 |
| **bomb 使用** | **0 次**（3 → 3） | `bombs_spent=0.0`、`bomb_decisions=0`；`EVENTS` 1,500/1,500 可用但 `PLAYER_HIT=0` |
| 玩家终态 | `playerState=0`（ALIVE），`px=76.887, py=421.172` | 一直在画面内活动 |
| 平均单决策耗时 | **526.07 ms**（四段 + EVENTS，均值；p50 496.3，p90 937.8） | 见 §3.2 分层 |
| 墙钟 | **789.1 s**（13 min 09 s） | 含进程启动 1280.4 ms + RESET 419.5 ms（一次性） |
| **墙钟 / 游戏内时间** | 789.1 s 墙钟 : 100.0 s 游戏内 → **实时倍率 0.127×** | 已提交 6,000 帧 / 789.1 s / 60 |
| 协议调用 | 275,296 次（1 个进程） | 183.5 次/决策 |

样本形态（不是人为构造）：决策 **192** 起 tape 开始有致死 AABB；1,294/1,500 个决策的 tape 非空；单帧致死 AABB 峰值 **191** 个，非空决策均值 **46.1** 个/帧；`margin` 中位数 16.0（`MARGIN_CAP`，p90 也是 16.0），最小 5.94 → 搜索确实在“贴着弹幕走”，不是空场景空转。

一致性（同一运行）：`agree_alive 4,496`、`agree_dead 0`、`geo_dead_engine_alive 0`、**`geo_alive_engine_dead 4`**、`trajectory_mismatch 4,455/4,500`。前三个说明“几何判活 + 引擎判活”占绝大多数；那 4 个反向不一致是**引擎判死而几何判活**，已由 bomb 规则与 label 如实记录，没有为了让数字好看而改判据。

**必须说清的一句**：这次停止是“6,000 帧上限先到”，不是过面；6,000 帧 = 100 s 的游戏内时间，stage 1 在 normal 下没有在 100 s 内结束，所以本卡只能报“存活 6,000 帧、0 死亡、0 bomb、未过面”。

---

## 7. 测试

```text
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_teacher_child_pytest3 -q `
    tests/test_v400_teacher.py tests/test_v400_teacher_import.py
52 passed in 1.17s          # 44 (teacher) + 8 (import)
python -m compileall -q touhou_rl/v400/teacher tools/rl/v400_teacher.py   # exit 0
```

`tests/test_v400_teacher.py` 从 31 增到 **44**（原有 31 项**全部保持通过、无改动放宽**），新增 13 项：

1. `test_child_recording_is_one_batch_and_matches_the_main_process_tape` — 18 条命令（H=6 的样本；真实 H=48 是 144 条）一次批量、主进程 0 步、命令顺序（HASH/HAZARDS/PLAYER 再 STEP 三元组）、子进程被丢弃、**子路径 tape 与主路径 tape 逐行相同**（帧号/hash/每帧致死行/PLAYER 12 列/boxes）；
2. `test_child_recording_fails_closed_before_touching_anything` — 探针 false 时不 spawn、不发命令、主进程 0 步，异常里带引擎原文；
3. `test_child_recording_wraps_a_refused_child_reply_and_still_drops_the_child` — 批量中途被 `ERR child-command` 拒绝 → `ChildHazardsUnavailable` 且子进程被回收；
4. `test_death_bomb_window_and_feasibility_follow_the_spec` — 0/1/2/3 bomb 的窗口（2/6/12/15）、上限 15、符卡 30、机体 ×1.8、可行性；
5. `test_decision_death_bombs_on_a_real_player_hit` — `0x0602` + bombs 3 → `bomb=1`、`death_bomb_player_hit`、`frame_event_flags=0x0602`、`event_names` 含 `PLAYER_HIT`；
6. `test_decision_does_not_death_bomb_without_a_player_hit` — `0x0400` → `bomb=0`；
7. `test_decision_does_not_death_bomb_without_a_bomb` — 有 hit、bombs=0 → `bomb=0` + `death_bomb_player_hit_but_no_bombs`；
8. `test_decision_reports_events_unavailable_instead_of_fabricating_a_zero` — `ERR events-unavailable` → `bomb_death_available=False`、`frame_event_flags=None`、`death_bomb_unavailable:…`；
9. `test_decision_active_bomb_rule_is_unchanged` — 全道墙 → `all_verified_plans_die` + `active_bomb`；
10. `test_death_verdict_agrees_between_child_recording_and_legacy_restore` — 同一 `kills_at=3` 世界，child 录制（主进程 0 步）与 legacy 录制（主进程有步进）给出**相同的判死结论、相同的 chosen_plan、相同的 bomb=1**；
11. `test_legacy_restore_replays_the_bomb_bit` — 重放历史 `(3,1),(4,0)` 真的把 bomb 位还原（否则决策点会漂）；
12. `test_engine_sha_gate_refuses_a_foreign_build` — 正确 SHA 通过、错 SHA/缺文件抛 `EngineShaMismatch`、`expected=None` 只记录不强制；
13. `test_cli_defaults_to_the_verified_child_build_and_keeps_legacy_opt_in` — 默认 `--engine` 指向 `rl_hazard2`、默认 SHA = `f172399f…`、`--allow-legacy-restore` 默认关、缺 build 时 CLI 退出码 4。

---

## 8. 剩余瓶颈与 G1 差距

1. **搜索 430 ms/决策（真实弹幕下占 82%）是唯一的大头**。空 tape 38.9 ms、有弹幕 492.5 ms（p90 860、峰值 2,400）；几何搜索是 Python+numpy，191 个致死 AABB/帧 × 48 帧的全展开就是 1,800 万次 AABB 间距。G1 要 ≥1× 实时（66.7 ms/决策），$需再快约 7.9×$：按 `11 §5` 的降配顺序（先减 k、再减束宽、再缩 H）只能到 ~2×，真要达标得上 numba/C 扩展，或走 `13 §4` 的“只在学生动作不同/余量很小的状态做完整搜索”。**这是 P1 进 G1 的真正门槛，不是还原。**
2. **record 45.6 ms（隔离 18.3 ms）**：144 条两跳子命令 ≈ 0.25 ms/条 + 子进程 47 帧真实推进。还能省：`PLAYER` 其实只有第 0 行被搜索读取（`start_position`/`hurtbox`/`bombs`），逐帧 `PLAYER` 是 48 条命令；若改成只在第 0 帧读，命令数 144→96（约 −1/3），代价是 tape 的 `player` 行变成常量。需要协议级“一次多帧录制”命令才能再降一个量级。
3. **verify 46.3 ms（隔离 16.9 ms）**：3 个计划 × ~8 条子命令，仍是父→模板→子两跳 + 每槽 `STATE`。同上的协议级批量能压下来。
4. **决死 bomb 的采样粒度**：每决策一次 `EVENTS` 只采到“这 4 帧里最后一帧”的字；若命中发生在块内前 3 帧，本次规则 2 不会触发（要等 `deaths` 计数上升，而它滞后到死亡提交）。要真正覆盖窗口，应把 R 降到 1 个子决策或在每个 commit 帧后读一次 `EVENTS`（每决策多 3 次 ~0.1 ms 的往返，代价很小）。本卡按任务要求“每决策一次”实现，未擅自增加。
5. **几何漏判 4 次（`geo_alive_engine_dead`）**：`trajectory_mismatch 4,455/4,500` 仍是 fork 晚 1 帧的已知 4.0 px 偏差（EXP-P1-006 就有）；它会让 tape 与真实轨迹错一帧，是 4 次漏判的候选解释之一（另两个候选是 ECL 激光/未建模致死物）。修法仍是“从 d−1 帧 PARK”或验证前补 1 帧原语。
6. **“过面”没有测到**：6,000 帧（100 s）内 stage 1 未结束。要拿“过面”结论，得把帧上限提到 stage 1 的真实长度（或干脆不设上限、用 `--stop-on-death` 与 stage 变化停），按当前 0.127× 的实时倍率，再翻一倍帧数就是再 13 分钟。
7. **G1 差距**：`S1N` 开发集 20 局一局都还没跑。本卡 1 局（100 s 游戏内）用 13 min 墙钟；20 局 ≈ 4.4 h（还只是 6,000 帧/局，不是整面）。**先把搜索拉回 ≥1× 实时，再谈 G1 的 20 局。**

---

## 9. 复现命令与产物

```powershell
# 单测（Windows 侧）
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_teacher_child_pytest3 -q `
    tests/test_v400_teacher.py tests/test_v400_teacher_import.py
python -m compileall -q touhou_rl/v400/teacher tools/rl/v400_teacher.py

# 能力探针 + SHA 门（WSL 侧，一个引擎）
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_teacher.py \
  --probe-only --engine /root/Touhou_RL/third_party/th08-rl/build/rl_hazard2"
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_teacher.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_hazard"   # 退出码 4

# 正确性 A/B：同 20 个决策，两条录制路径
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_teacher.py \
  --decisions 20 --stage 1 --difficulty normal --seed 1 --mode full --quiet \
  --out .tmp_v400_teacher_child_ab_child.jsonl"
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_teacher.py \
  --decisions 20 --stage 1 --difficulty normal --seed 1 --mode full --quiet --allow-legacy-restore \
  --out .tmp_v400_teacher_child_ab_legacy.jsonl"
python .tmp_v400_teacher_child_compare.py .tmp_v400_teacher_child_ab_child.jsonl .tmp_v400_teacher_child_ab_legacy.jsonl

# 关卡级运行（§6）
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_teacher.py \
  --decisions 3000 --max-frames 6000 --stop-on-death --stage 1 --difficulty normal --seed 1 \
  --mode full --out .tmp_v400_teacher_child_stage1.jsonl > .tmp_v400_teacher_child_stage1.log 2>&1"
```

产物（均在仓库根）：

| 文件 | 内容 |
|---|---|
| `.tmp_v400_teacher_child_probe.json` | 真机探针：SHA、启动/RESET、子命令单次成本、48 帧批量 17.75 ms、父子 hash/STATE 逐字节相同 |
| `.tmp_v400_teacher_child_probe_cli.json` | CLI `--probe-only`：SHA 门通过 + `child_hazards.available=true` |
| `.tmp_v400_teacher_child_sha_gate.json` | 旧 build（`8053dfaa…`）被 SHA 门拒绝的原文 |
| `.tmp_v400_teacher_child_ab_child.jsonl` / `.log` | A/B 的 child 录制 20 决策 |
| `.tmp_v400_teacher_child_ab_legacy.jsonl` / `.log` | A/B 的 legacy restore 20 决策 |
| `.tmp_v400_teacher_child_compare.json` | 逐决策比对结果（0 处分歧） |
| `.tmp_v400_teacher_child_stage1.jsonl` / `.log` | 关卡级 1,500 决策 / 6,000 帧的每决策 JSONL + 进度日志 |
| `.tmp_v400_teacher_child_stage1_summary.json` | 关卡级运行的分层统计（空/有弹幕、percentile、判死决策清单、事件字集合） |

退出码：SHA 门（旧 build）`4`；A/B 两次 `0`；关卡级 `0`。

## 10. 修订记录

| 日期 | 修改 | 原因 |
|---|---|---|
| 2026-11-09 | 初版 | 切 rl_hazard2 + 子进程录制 + EVENTS 两条 bomb 规则 + 首个关卡级结果 |
