# EXP-P1-009 — stage_clear 诊断：教师活着但不过面

- **日期**：2026-11-09
- **起因**：EXP-P1-008 §5 的关卡级运行（4,000 决策 / 16,000 帧 / 0 死亡 / bomb 3 / `stage_clear=false` / 停止时 `stage=1`）与本轮 P1 的过面验收门冲突。
- **结论（先给）**：**根因是 ④「stage 推进判据不是 stageFrames」的一个具体形态**——TH08 结束当前面之后，`g_Supervisor`/`GameManager` 只有在 `g_CurFrameInput & 0x1001`（`SHOOT|ENTER`）出现**上升沿**时才会离开"等待换面"状态（`GameManager.cpp:381-389`，10-P0 §4.2 原文："按住 SHOOT 不算上升沿"）。教师的 18 动作空间把 SHOOT 恒开（`cand_to_mask` 无条件 OR `BTN_SHOOT`），因此**永远不产生上升沿**，游戏就停在"本面已结束、下一面未开始"的状态：无敌、无弹幕、无敌人、`stageFrames` 继续自增、`stage` 不变。**①「教师从不射击」被数据直接否证**（逐帧掩码 100 % 含 SHOOT 位，score 从 0 涨到 151,081）。

---

## 1. 先确诊（不改代码）

### 1.1 动作/掩码构造：18 个动作**不含** shoot 维，但**每一帧都带 shoot 位**

| 位置 | 事实 |
|---|---|
| `touhou_rl/v400/teacher/config.py` | `N_CAND = 18 = focus(2) × move(9)`，**没有 shoot 维** |
| `touhou_rl/v400/teacher/motion.py` | `split_cand(cand) -> (move, focus)`，`cand < 18`，无 shoot |
| `touhou_rl/v400/gym/actions.py:58-75` | `cand_to_mask(cand, bomb)` = `方向位 | BTN_SHOOT`（+ focus / + bomb），**SHOOT 无条件置位** |
| `tools/rl/v400_teacher.py`（原） | 提交 `STEP {cand_to_mask(cand, bomb)} {decision_frames}`，即每帧都带 SHOOT |
| `13-P3 §2.2` | 动作是因子化的，`shoot`：固定开火（D-08 主线设置）——与实现一致 |
| `10-P0 §5` / `spec.py:343` | `BTN_SHOOT = 1 << 0 = 0x0001`，`BTN_BOMB = 0x0002`，`BTN_FOCUS = 0x0004` |

**结论**：教师**不需要**"补上 shoot"，它一直是射击的。真正缺失的是**射击的释放沿**。

### 1.2 SHORT 真实对局：1,000 决策 / 4,000 帧（stage index 1 = 第 2 面，normal，seed 1）

命令（WSL，1 个引擎实例）：

```bash
/root/v400-venv/bin/python .tmp_v400_clear_teacher.py --label a_teacher \
  --decisions 1000 --pulse-period 0 --sample-every 200 --out .tmp_v400_clear_a.jsonl
```

| 决策 | frame | stage | stageFrames(f3) | score(f4) | enemies | boss | enemyLife | deaths | bombs | playerState | curState | setupState | 输入掩码 f16 | frameEventFlags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1 | 2 | 0 | 1 | 0 | 20 | 0 | 3 | 3（无敌） | 2 | 0 | 0x0000 | 0x1400(5120) |
| 200 | 800 | 1 | 802 | 1,938 | 13 | 0 | 140 | 0 | 3 | 0 | 2 | 0 | 0x0091 | 0x0420(1056) |
| 400 | 1,600 | 1 | 1,602 | 44,535 | 6 | 0 | 1,147 | 0 | 3 | 0 | 2 | 0 | 0x00a1 | 0x0400(1024) |
| 600 | 2,400 | 1 | 2,402 | 71,046 | 4 | 0 | 469 | 0 | 3 | 0 | 2 | 0 | 0x0041 | 0x0400(1024) |
| 800 | 3,200 | 1 | 3,202 | 107,240 | 8 | 0 | 1,017 | 0 | 2 | 0 | 2 | 0 | 0x0081 | 0x0440(1088) |
| 1000 | 4,000 | 1 | 4,002 | 151,081 | 5 | 0 | 1,082 | 0 | 2 | 0 | 2 | 0 | 0x0011 | 0x0000 |

- **敌人死、score 涨**（0 → 151,081，4,000 帧内 +151k）→ **①「从不射击 → 敌人/Boss 不死」被否证**；
- `playerState` 大部分帧是 0（存活），`curState=2`（`SupervisorState_GameManager`）、`setupState=0`（COMPLETE）、`subthread=0` → 游戏在**正常运行**，不是卡在加载/换面 setup；
- Boss 在 4,000 帧内**尚未出现**（`boss=0`，但 `enemyLife` 上千、弹幕密，属于中 Boss/道中）；
- 掩码统计：`frames_with_shoot / frames = 1.000000`（**每一帧都带 SHOOT**）。

### 1.3 停下来之后到底发生了什么（EXP-P1-008 的 4,000 决策 JSONL 逐 200 决策采样）

`.tmp_v400_raster_stage1.jsonl` 里 `tape_boxes.empty_frames == 48`（整个 48 帧 horizon **一个致死物都没有**）从决策 ≈ 2,160（frame ≈ 8,640）开始，一直到 4,000（frame 16,000）**再没恢复**；`px/py` 以 400 决策（1,600 帧）为周期**逐位重复**（27.2/409.9 → 36.6/408.4 → 28.7/418.3 循环）→ 世界已经**完全静止**，教师在做同一个无意义循环：

```text
dec=2000 f=8004  stage=1 sf=8006  deaths=0 lives=2 bombs=0 boxes_max=0   empty=48
dec=2400 f=9604  stage=1 sf=9606  deaths=0 lives=2 bombs=0 boxes_max=0   empty=48
dec=2800 f=11204 stage=1 sf=11206 deaths=0 lives=2 bombs=0 boxes_max=0   empty=48
dec=3200 f=12804 stage=1 sf=12806 deaths=0 lives=2 bombs=0 boxes_max=0   empty=48
dec=3600 f=14404 stage=1 sf=14406 deaths=0 lives=2 bombs=0 boxes_max=0   empty=48
dec=3999 f=16000 stage=1 sf=16002 deaths=0 lives=2 bombs=0
```

`f3`（`gameplayFrameCounter`）与 `f0` 严格 1:1 增长 ⇒ `GameManager::OnUpdate` 每帧都在跑（**不是**重试菜单那种冻结），但 `stage` 一直是 1 ⇒ 从来没有进入下一面的 `RegisterChain`（那会把 `gameplayFrameCounter` 清零）。

### 1.4 根因判定（源码 + 实测两边对齐）

| 代码 | 含义 |
|---|---|
| `GameManager.cpp:297-390` | `flags.stageTransitionState == 2` 时才 `AdvanceToNextStage()`；而把 `STAGE_TRANSITION_STATE_MASK` **清零**只在 `((g_CurFrameInput & 0x1001) != 0) && ((g_CurFrameInput & 0x1001) != (g_LastFrameInput & 0x1001))` 成立时发生 |
| `GameManager.cpp:1085`（setup 线程） | `while (gameManager->flags.stageTransitionState != 0) Sleep(17);` — 换面期间 `gameplaySetupState` 停在 IN_PROGRESS |
| `Gui.cpp:1124-1127` | 本面结束序列把 `stageTransitionState` 置 2（`stageClearSequenceActive` 之后） |
| `src/rlharness/rlharness.cpp` | **没有** `0x1000`/ENTER 注入路径（10-P0 §4.2 描述的"harness 每隔一帧注入 ENTER 脉冲"**没有实现**），`g_mask` 原样回给 `Controller::GetInput()`，不做任何位过滤 |

→ 教师恒开 SHOOT ⇒ `cur & 0x1001 == last & 0x1001 == 0x0001` 恒成立 ⇒ **换面永远不完成**。

**判定：④（stage 推进判据不是 stageFrames）**，而不是 ①（不射击，已否证）或 ②/③（Boss 战保守/未击破：教师在 16,000 帧里把能打的敌人都打了，弹幕密度峰值 219 个/帧，死亡 0；它不是"不敢打"，而是**打完之后无法翻篇**）。

修复后的对照把这条判定钉实了：同一个教师、同一个开局，**只多了 208 个"释放 SHOOT 一帧"**，EXP-P1-008 里那段**永久**空窗就变成**有界**空窗并且之后继续推进（§4.1）。恒开 SHOOT 的唯一区别就是"永远没有上升沿"，因此这个差别只能归因于上升沿。

### 1.5 对照实验（同一开局 stage=1, normal, seed=1）

| 组 | 策略 | 决策/帧 | 是否射击 | `stageFrames` 增速 | `score` 增速 | enemies / boss | 是否换面 | 停止原因 |
|---|---|---|---|---|---|---|---|---|
| **(a)** 教师原策略 | teacher-v1 full，child 录制 | 1,000 / 4,000 | **是**（100 % 帧带 SHOOT） | 1.000 帧/帧 | **+151,081 / 4,000 帧**（37.8/帧） | 1→13→6→4→8→5，`boss=0`（Boss 未出现） | **否**（`stage=1`） | `decisions` |
| **(a′)** 教师原策略·长程（EXP-P1-008 既有产物） | 同上，不射释放沿 | 4,000 / 16,000 | 是 | 1.000 帧/帧（**后 8,000 帧零致死物**） | 帧 8,000 后**不再增长**（世界静止） | 后 8,000 帧 `enemies` 恒为空、`boss=0` | **否**（`stage=1`） | `max_frames` |
| **(b)** 教师 + 强制 shoot 位恒开 | `--force-shoot`（把 `BTN_SHOOT` 再 OR 一遍） | 1,000 / 4,000 | 是（**与 (a) 逐位相同**） | 同 (a) | 同 (a) | 同 (a) | 否 | `decisions` |
| **(c)** `still` + 强制 shoot 位恒开 | 掩码恒 `0x0001`，不动 | 12,000 帧（脚本探针） | 是 | 0.999 帧/帧，直到 f0=1,698 后**冻结** | +65,960 然后就停 | 峰值 8 个；`deaths` 3 次、`lives` 2→0 | 否（**先死**） | 3 次死亡后停在 `showRetryMenu` |
| **(d)** `still` + **shoot 脉冲**（每 120 帧释放 1 帧） | 掩码 `0x0001`，每 120 帧 1 帧 `0x0000` | 12,000 帧 | 是（有释放沿） | 死亡后**能回到 title 并重新开局**（`stage=0`/`curState=1→2` 循环） | 每个循环重新从 0 起 | — | 由 `deaths` 决定 | `decisions` |

要点：

1. **(a) ≡ (b)**（掩码逐位相同）："强制 shoot 恒开"这一组证明**shoot 从来不是缺失项**，教师 100 % 的帧已经在射击。任务书假设的"教师缺 shoot"不成立。
2. **(c) `still` 不能作为过面对照**：不动就被打死（f0≈1,700 三次死亡、lives 0），之后游戏停在重试菜单（`gameplayFrameCounter` 冻结在 1,698）。这本身是一条证据：**光开火不躲，连 1,700 帧都活不过**，所以"不过面"不能归因于射击。
3. **(d) 的释放沿让卡死的流程动了**：同样会死光的 `still` 策略，加上每 120 帧 1 帧释放 SHOOT 之后，死亡后能离开结果画面、回到标题并重新开局（(c) 则永久冻结在同一帧）——**这是"上升沿缺失让流程停住"的直接可观测差异**，与 §1.4 的源码判据同源。
4. (a) 的 wall 132.1 s / 4,000 帧 = **0.5048× 实时**（EXP-P1-008 是 0.722×；本轮机器上 Windows 零售 trace 批跑同时在跑，且 `raster`/`search` 正被并行的 EXP-P1-010 改动）。

---

## 2. 修复（最小改动）

见 §3 文件清单。核心是**每 `shoot_pulse_period` 个决策，让最后一帧释放 SHOOT**（下一帧重新按下 = 上升沿）：

```python
# touhou_rl/v400/teacher/shoot.py（新增）
commit_frames(cand, bomb, decision_frames=2, release_shoot=True)
# -> [(mask, 1), (mask & ~SHOOT & ~BOMB, 1)]   帧数不变、bomb 位不受影响
```

代价：`1 / (period × 4 帧)` 的开火时间（period=24 → 1.04 %）。收益：换面等待是"无限期"的，脉冲**不需要和换面对齐**，任意周期都会在有限时间内补上那个沿。

---

## 3. 变更文件

| 文件 | 变更 |
|---|---|
| `touhou_rl/v400/teacher/shoot.py` | **新增**：`SHOOT_ALWAYS`、`STAGE_ADVANCE_BITS=BTN_SHOOT`、`SHOOT_PULSE_PERIOD=24`、`release_due`、`commit_frames`、`commit_block`、`commands`、`mask_summary`；文件头写清 `GameManager.cpp:381-389` 与 10-P0 §4.2 的判据 |
| `touhou_rl/v400/teacher/config.py` | `TeacherConfig.shoot_always=True`、`shoot_pulse_period=24`（+ 负数校验），注释直指 13-P3 §2.2 与上升沿判据 |
| `touhou_rl/v400/teacher/label.py` | `TeacherLabel.shoot`（默认 True）、`shoot_release`（默认 False），写进 `as_dict`/`from_dict`（旧 JSONL 缺字段时取默认，不报错） |
| `touhou_rl/v400/teacher/decide.py` | `decide(..., shoot_release=False)`；label 记录 `shoot/shoot_release`，`extra` 记录 `shoot_always/shoot_release/shoot_pulse_period` |
| `tools/rl/v400_teacher.py` | `--shoot-pulse-period`；提交改用 `commit_block`/`step_commands`；每决策记录 `step_masks`/`shoot_release`/`shoot_frames`；报告新增 `shoot_always/shoot_pulse_period/shoot_releases/frames_with_shoot/frames_without_shoot/shoot_fraction/empty_tape_decisions/empty_tape_streak_max/stage_frames_seen{first,last,max,resets}` |
| `tests/test_v400_teacher.py` | +5 项（见 §5） |
| `Document/Update/Alpha/v4.0.0/experiments/EXP-P1-009-stage-clear-diagnosis.md` | 本卡 |

没有改引擎/协议/依赖/第三方，没有重建 build，没有删除文件。

---

## 4. 过面重测（修复后）

```bash
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_teacher.py \
  --decisions 5000 --max-frames 20000 --stop-on-death --stage 1 --difficulty normal --seed 1 \
  --mode full --quiet --out .tmp_v400_clear_stage1.jsonl > .tmp_v400_clear_stage1.log 2>&1"
```

<!-- RESULT-SECTION -->

### 4.1 结果：**仍未过面，但已经不是"卡住"**

| 指标 | 实测（修复后，20,000 帧） | EXP-P1-008（修复前，16,000 帧） |
|---|---|---|
| 停止原因 | `max_frames` | `max_frames` |
| 决策数 | 5,000 | 4,000 |
| 存活帧数 | **20,000**（333.3 s 游戏内） | 16,000（266.7 s） |
| **是否过面** | **否**（`stage_clear=false`，`stage_at_stop=1`） | 否 |
| 死亡 | **0** | 0 |
| bomb | **0 次**（`bombs_spent=0`，`bombs 3→3`） | 3 次 |
| 空 tape 决策 | 827（`empty_tape_streak_max=611`） | 2,317 |
| 墙钟 | **694.7 s**（11 min 35 s，1 进程 922,234 次协议调用） | 369.27 s |
| **实时倍率** | **0.4797×**（20,000 帧 / 694.7 s / 60） | 0.7221× |
| 决策/秒 | 7.195 | 10.83 |
| 四段均值 | record **47.53** / search **62.89** / verify **26.01** / commit 1.55 / events 0.05 ms | 27.05 / 41.84 / 21.80 / 1.15 / 0.05 |
| SHOOT 帧占比 | **0.9896**（19,792 / 20,000 帧带 SHOOT，208 个释放帧 = 208 次上升沿） | 1.0000（无释放帧） |
| `stageFrames` | 6 → 20,002，`resets=0` | 6 → 16,002，`resets=0` |
| frameEventFlags | 5,000 帧全部可读，`PLAYER_HIT` 0 次 | 同 |

**关键差别（这一条才是修复的证据）**：

| | 世界在做什么 |
|---|---|
| **修复前**（EXP-P1-008，无释放沿） | frame ≈ 8,000 起 `tape_boxes.empty_frames == 48`，**一直空到 16,000 结束**（8,000 帧零致死物），`px/py` 以 1,600 帧为周期逐位重复 |
| **修复后**（本卡，208 个释放沿） | 空窗**有界**：frame ≈ 9,000–10,460 为空（611 决策），之后**恢复**并且一路打到 20,000 帧仍是重弹幕（单帧致死物 162 → 306） |

即：恒开 SHOOT 会把游戏永久停在一个"只认上升沿"的状态里；给释放沿之后同一个状态**会自己走出来**，只是本面还没打完。

### 4.2 没过的原因不是停滞，而是帧上限 + `--stage` 0-based

1. **帧上限**：20,000 帧（333 s）时教师还在打（1,000 决策采样点：frame 19,004 处 `boxes max=306, mean=273`，boss 级弹幕），没有任何"世界静止"的迹象。要过面必须把上限提到 `stageFrames` 真实长度之上。
2. **`--stage` 是 0-based**：`STATE.f1 = GameManager.currentStage`，`ScoreDat.hpp:71-83` 里 `STAGE1 = 0`。`--stage 1` 打的是**第 2 面**；人类参考里 "Stage1 = 13,258 gameplay_frames（Normal `Th8_ud2970.rpy`，0 死 0 bomb，EXP-P0-046）" 是**第 1 面**。拿 `--stage 1` 去对 13,258 帧是**错面对比**：同一份人类参考的第 2 面是 **18,525** 帧。

### 4.3 独立复核（EXP-P1-011 §3）：§1.4 的根因**被否证**

§4.1 的观测数据全部复现，但 §4.1 末尾「恒开 SHOOT 会把游戏永久停在一个只认上升沿的状态里；给释放沿之后同一个状态会自己走出来」这句**因果解释不成立**。证据三条（详见 [EXP-P1-011 §3](EXP-P1-011-multistage-progress.md)）：

1. **源码**：`GameManager.cpp:297-390` 里 `stageTransitionState == 2` 分支**直接** `AdvanceToNextStage()` + `curState = SupervisorState_GameManagerReInit`；`(g_CurFrameInput & 0x1001)` 的上升沿只负责**清 `STAGE_TRANSITION_STATE_MASK`** 和切 `nextSupervisorState`。所以上升沿只在**已经开始换面之后**起作用，不是换面的前置条件。`2` 的唯一来源是 `Gui.cpp:1121-1127`，其前提是本面结束序列（`GUI_MSG_SHOW_STAGE_RESULTS`，`Gui.cpp:672-687`）已经跑过。实测 `stage_at_stop=1`、`stageFrames.resets=0` ⇒ 状态机从未到 2。
2. **数据**：`ENEMIES` 接进 CLI 后（EXP-P1-011 §2）逐 50 决策可看到 Boss：frame ≈5,400 起 `boss=1`，`boss_life` 11,506 → **62**（≈2,600 帧打掉 11,444 HP），随后 frame 8,204 起**精确冻结在 18,200**、tape 全空，持续 **≈2,400 帧（40 s）**，再继续掉血（17,278 → 14,884）。`curState=2`、`setupState=0`、`subthread=0` 全程不变 ⇒ 不是加载、不是冻结、不是等换面，是 **Boss 阶段性的无敌/演出窗口**。
3. **那条空窗里有 25 个上升沿**（`2400 / 96 = 25`），窗口长度**一帧没变**（仍是 2,400 帧）。若它真由上升沿驱动，第一个脉冲（≤96 帧后）就会放行。⇒ 上升沿不是这个状态的门。

**订正后的判读**：本次重测的 `stage_clear=false` 不是"被上升沿卡住"，而是**帧预算不足**——20,000 帧 = 333 s 而人类第 2 面是 **18,525** 帧（308.8 s，`--stage 1` 就是 0-based Stage 2，见 §6.1 / EXP-P1-011 §5），上限只有人类的 1.08×，多阶段 Boss 还没打完，`GUI_MSG_SHOW_STAGE_RESULTS` 自然没跑到。

**§1.5(c)/(d) 的对照仍然有效**，但它证明的是「上升沿能让**死亡→标题→重开**这条流程动」，不是「上升沿能换面」。两个状态机被混为一谈了。

**§4.1 表格里的数字全部保留、未改**；被订正的只有它对因果的那句解释。

### 4.4 Stage 1（`--stage 0`，S1N cap 19,887 帧）过面重测：**过一次面**

```bash
wsl -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_teacher.py \
  --decisions 5000 --max-frames 19887 --stop-on-death --stage 0 --difficulty normal --seed 1 \
  --mode full --quiet --out .tmp_v400_clear_stage0.jsonl > .tmp_v400_clear_stage0.log 2>&1; echo EXIT=$?'
# EXIT=0
```

| 指标 | 实测 |
|---|---|
| `stage_clear` | **`true`** |
| `stop_reason` | **`stage_changed`** |
| **换面帧号 / 决策号** | **13,676 / 3,418** |
| **换面后 `STATE.f1`** | **1**（`stages_visited=[0,1]`、`stages_completed=1`） |
| `stage_frames` | 13,674 → **0**（`resets=1`，`stage_frames_at_stop=0`） |
| 存活 / 死亡 / bomb | 13,676 帧 / **0** / **0**（3→3） |
| 墙钟 / 实时倍率 | 399.1 s / **0.5711×** |
| 对人类 13,258 帧 | **1.032×**，用掉 1.5× cap 的 68.8% ⇒ **S1N 帧预算达标** |
| Boss | `boss_first={dec 733, frame 2,936, life 6,000}`、`boss_life_max=60,000`、`boss_decisions=2,100` ⇒ **Boss 被打死**，不是逃掉/超时 |

**结论**：本卡 §1.4 归因的「上升沿」既不是**这次** `--stage 1` 不过面的原因（§4.3），**也不再是**过面的障碍——同一套带释放沿的配置在真正的第 1 面上 13,676 帧过面。**过面的成立条件因此被定位为"帧预算 + Boss 击杀"，不是输入边沿。**

仍未回答：**脉冲是否必要**。本轮所有运行都带 `shoot_pulse_period=24`，没有 `--shoot-pulse-period 0` 的 A/B，所以"脉冲修好了过面"这个说法**不能从本卡得出**（见 [EXP-P1-011 §4.3](EXP-P1-011-multistage-progress.md)）。

<!-- TEST-SECTION -->

---

## 5. 测试

```text
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_clear_pytest -q \
    tests/test_v400_teacher.py -k "shoot or release or config_and_label"
5 passed, 55 deselected                       # 本卡新增 5 项
```

新增 5 项：

1. `test_shoot_bit_is_in_every_candidate_mask` — 18 个候选**每个**都带 `BTN_SHOOT`、不带 BOMB、focus 位等于 `cand // 9`，且 2 帧提交里两帧都射击；
2. `test_release_frame_is_last_exactly_once_and_only_drops_shoot` — 释放帧必须是**最后一帧**、只清 SHOOT（连 BOMB 一起清）、总帧数不变（4 帧）、下一决策首帧重新带 SHOOT（**上升沿**成立）、`mask_summary` 精确 3/2/1；
3. `test_release_due_is_periodic_and_zero_disables_it` — period 命中点、`period=0` 关闭、`period=1` 每决策释放；
4. `test_config_and_label_carry_the_shoot_decision` — config 两个字段、负数参数报错、label 往返 + **旧 JSONL 缺字段取默认**；
5. `test_cli_exposes_the_shoot_pulse_and_applies_it_to_the_config` — CLI 参数与 `_config_with_pulse` 三种取值（None/12/0）。

**全量（既有 50 项 + 并行 EXP-P1-010 新增项 + 本卡 5 项）**：

```text
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_clear_pytest2 -q \
    tests/test_v400_teacher.py tests/test_v400_teacher_import.py
2 failed, 64 passed, 2 skipped                      # 退出码 1
```

**这 2 个失败不是本卡引入的**，是并行工作包 EXP-P1-010 正在改 `search.py`/`raster.py`（在飞）留下的：

- `test_vectorised_search_matches_the_reference_on_dense_and_empty_tapes`
- `test_vectorised_search_finds_unknown_closed_form_on_a_wall`

失败断言都是 `detail` 字典多了/少了键（`{'finalist_extra': 0}` vs 期望），属于 `search.py` 新增的 finalist 阶段；本卡没有碰 `search.py`/`raster.py`。本卡新增的 5 项与 `shoot`/`label`/`config`/CLI 相关的既有项全绿（`-k "shoot or release or config_and_label"` → **5 passed**）。**合并时必须以同一时刻的树重跑一次，并确认这 2 项恢复。**

### 5.1 复核轮的全量统计（本轮实测，替换上面"在飞时"的 2 failed）

```text
# WSL（既有 68 项 + 本卡/EXP-P1-011 新增 7 项）
$ PYTHONPATH=/mnt/d/WorkShop/Touhou_RL /root/v400-venv/bin/python -m pytest \
    -p no:cacheprovider --basetemp .tmp_v400_clear_pytest_wsl2 -q \
    tests/test_v400_teacher.py tests/test_v400_teacher_import.py
75 passed, 0 skipped                       # 退出码 0

# Windows（numba 相关 2 项按设计 skip）
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider \
    --basetemp .tmp_v400_clear_pytest4 -q tests/test_v400_teacher.py
65 passed, 2 skipped                       # 退出码 0
```

即 EXP-P1-010 引入的那 2 项 `finalist_extra` 断言失败**已消失**（同一棵树、同一时刻重跑，0 failed）；本卡没有为了让测试变绿改任何断言。

新增 7 项（EXP-P1-011 §2/§3/§6.2，全部落在 `tests/test_v400_teacher.py`）：

1. `test_pulse_produces_exactly_one_rising_edge_per_period` — 把真实 `commit_block` 输出展开成**逐帧输入字**，数 SHOOT 的下降沿/上升沿：3 个周期必须有 3 对，且没有两个相邻的释放帧；
2. `test_pulse_period_cli_values_reach_the_committed_frames` — `period=1` 每个决策都释放、`period=0` 永远不释放；
3. `test_stage_change_is_bookkept_apart_from_survival` — 换面行是**面内增量**（`frames_in_stage/deaths_in_stage/bombs_spent_in_stage`）、未完成面必须是 `to_stage=None` 的 open 行、`--follow-stage` 默认关闭；
4. `test_enemies_reply_decodes_count_boss_and_life` — 按真实 `HandleEnemies` 报文解 `count=2`、第二个 `flags1` 的 `BOSS` 位、`life=0x310=784`、`ERR` 不当成 `boss=0`；
5. `test_engine_enemies_pages_a_crowded_frame_and_reports_truncation` — 20 个敌人分 2 页、`max_pages=1` 时 `truncated=True`、`ERR` 变成 `available=False`；
6. `test_cli_logs_enemy_and_debug_rows_without_touching_the_policy` — `_enemy_summary`/`_debug_snapshot` 的字段与"不可观测"口径（`stage_transition_state=null`）；
7. `test_follow_stage_walks_the_next_stage_load_out` — `--follow-stage` 换面后必须先用 SHOOT 边沿把下一面的 setup 线程走完（否则 `PARK` 返回 `ERR park-unsafe ... threads=2 setup=1`，实测 exit 5），并且加载永不结束时必须在帧预算处退出而不是死循环。

<!-- TEST-SECTION -->

---

## 6. 剩余问题

1. **过面仍未达标（唯一未过项）**：修复后 20,000 帧仍 `stage_clear=false`，但世界在推进（§4.1）。**下一步最小实验**（按优先级）：
   - `--stage 0 --max-frames 20000`：第 1 面，直接对 EXP-P0-046 的 13,258 帧；教师若在 13,258×1.5 = 19,887 帧内过面就是达标；
   - `--stage 1 --max-frames 40000`（或 `--decisions 11000`）：第 2 面的人类参考是 18,525 帧，教师现在 20,000 帧还在打，需要更大的上限才能给出"过面/不过面"的确定结论。本卡已起跑一次 36,000 帧的确认跑（`.tmp_v400_clear_stage1_long.*`），但 0.4797× 实时意味着约 21 min，超出本卡预算，**不能拿它当结论**。
   - 若 40,000 帧仍不过：才需要怀疑"教师在 Boss 战被保守策略卡住"（②/③），届时的判据是 `stageFrames` 在涨而 Boss `life` 不降（需要先按第 2 条把 `ENEMIES` 接进 CLI）。
2. **Boss 不可观测**（CLI 侧）：`STATE` 25 字段里没有 boss/敌人；`ENEMIES` 命令有（本卡的探针脚本用它测到 `boss` 位与 `life`），但 CLI 未接。建议把 `ENEMIES 0 16` 的 `count/boss/life` 接进 `_state_snapshot`，否则"Boss 是否存在/HP"在 CLI 产物里永远是"不可观测"。
3. **`stageTransitionState` / `dialogue_present` 不可观测**：`DEBUG` 只给 `curState/setupState/stage/difficulty/shot/isReplay/isDemo/threads/frames`，`STATE` 没有 GameManager flags 与 Gui 字段。本卡只能从"世界静止但 `gameplayFrameCounter` 照涨"推断，不能直接读出那个 2-bit 状态机。**建议**在 harness 的 `DEBUG` 里加 `stageTransition=0..3` 与 `dialogue=0/1`（只读，不改协议语义），这是把 §1.4 从"源码判据 + 行为推断"升级为"直接观测"的最小改动。
4. **`--stage` 0-based 的命名债**：CLI 的 `--stage 1` 与文档/人类参考的 "Stage 1" 差一面，EXP-P1-008 §5 的"266.7 s 还没结束"要按第 2 面重读。**建议**：要么把 CLI 改成 1-based 并在报告里打 `current_stage = args.stage - 1`，要么在所有卡里显式写 `--stage 1 (= currentStage 1 = 第 2 面)`。
5. **并行 EXP-P1-010 的干扰**：`raster.py`/`search.py`/`tests/test_v400_teacher.py` 在本卡测量期间被另一工作包**同时编辑**（首次运行直接报 `'LethalTable' object has no attribute 'backend'`，随后 `NameError: _refine_finalists`）。本卡因此把 `raster.build_table` 钉在 `backend="numpy"`（EXP-P1-008 的参考路径）测完所有对照，并且 §4 的合成搜索默认带上了 `finalist_extra=8`。所以：
   - EXP-P1-008 与本卡的**决策序列不可逐位比较**（不同的搜索配置）；
   - 本卡对"上升沿"的因果证据里，**唯一干净的同策略 A/B 是 §1.5 的 (c) vs (d)**（同一脚本策略，只差一个释放帧）；
   - **合并前必须用同一棵树重跑基线**。
6. **实时倍率回退**：0.7221× → 0.4797×，其中 search 41.84 → 62.89 ms 主要是并行工作包的 finalist 精确重算，record 27.05 → 47.53 ms 主要是机器负载（Windows 零售 trace 批跑同时在跑）。1× 实时门未过，且本卡没有做性能优化。

### 6.1 复核轮对上面第 1 条的订正（2026-11-09，EXP-P1-011）

- 第 1 条「过面仍未达标（唯一未过项）」**已过期**：`--stage 0`（真正的第 1 面）**13,676 帧 / 3,418 决策过面**，`stage_clear=true`、`stage` 0→1、死亡 0、bomb 0、0.5711× 实时，见 §4.4。
- 因此"下一步最小实验"里的 `--stage 0 --max-frames 20000` 已经跑完并**过面**；`--stage 1 --max-frames 40000` 仍然没跑（预算不允许），第 2 面用 20,000 帧（人类的 1.08×）不下结论。
- 第 2 条「Boss 不可观测」**已解决**：`ENEMIES` 的 `count/boss/life` 已接进 CLI 每决策日志（EXP-P1-011 §2），第 1 面的 `boss_life_max=60,000`、`boss_first={dec 733, frame 2,936}` 就是它产出的。
- 第 3 条「`stageTransitionState` 不可观测」**仍然成立**，本卡没有绕过它：`debug.stage_transition_state` 一律写 `null`，换面判定只用 `STATE.f1`。
- 第 4 条「`--stage` 0-based 的命名债」**仍然成立**，且已被本卡的 §4.4 / EXP-P1-011 §5 按 0-based 口径使用；EXP-P1-008 §5 的表述仍待订正（不在本卡白名单内）。

---

## 7. 产物（仓库根，`.tmp_v400_clear*`）

| 文件 | 内容 |
|---|---|
| `.tmp_v400_clear_probe.py` / `.json` | 脚本策略三组：R1 不射击、R2 still+SHOOT、R3 still+SHOOT 脉冲（每 120 帧释放 1 帧），12,000 帧 × 3，采样 STATE/ENEMIES/DEBUG/EVENTS |
| `.tmp_v400_clear_teacher.py` / `_a.jsonl` / `_a.log` | 教师策略探针（child 录制 + PARK/SPAWN），1,000 决策 / 4,000 帧，逐 200 决策采样，含逐帧掩码直方图 |
| `.tmp_v400_clear_stage1.jsonl` / `.log` | 修复后 CLI 关卡级运行（§4） |
| `.tmp_v400_clear_analyze.py` | 采样器（EXP-P1-008 与本卡 JSONL 通用） |

## 8. 修订记录

| 日期 | 修改 | 原因 |
|---|---|---|
| 2026-11-09 | 初版 | 定位"活着但不过面"：恒开 SHOOT 缺上升沿，阶段推进停在 `stageTransitionState` |
