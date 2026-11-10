# EXP-P1-011 — 多面推进 / Boss 可观测 / 释放沿参数扫描

- **日期**：2026-11-09
- **状态**：进行中（Stage 1 过面结论见 §4；多面推进见 §6）
- **上游**：[EXP-P1-009](EXP-P1-009-stage-clear-diagnosis.md)（过面诊断）、[EXP-P1-008](EXP-P1-008-search-rasterized.md)（关卡级运行）
- **一句话**：把「过面」从**不可观测**变成**可观测**（`ENEMIES` 的 `count/boss/life` + `DEBUG` 的 `curState/setupState` 进每决策日志 + `--follow-stage` 换面记录），再用它回答一个问题——教师在真实对局里到底卡在「等换面」「打不死 Boss」还是「就是帧不够」。

---

## 1. 为什么需要这张卡

EXP-P1-009 的结论是「恒开 SHOOT 缺上升沿 → 换面永不完成」，并据此加了 `shoot_pulse_period=24`（每 96 帧释放 1 帧 SHOOT）。修复后重测（5,000 决策 / 20,000 帧 / stage 1）**仍然 `stage_clear=false`**。本轮独立复核要回答三件事：

1. 重测结果到底是什么（**过面 / 活着 / 帧上限** 必须分开写）；
2. 那个「上升沿」假说是否成立（源码 + 数据两侧验证）；
3. 教师卡在哪个可观测状态上（Boss 在不在、HP 降不降、`curState/setupState` 是什么）。

---

## 2. 观测补齐（只加观测，不改策略）

`ENEMIES [offset [limit]]`（`rlharness.cpp:387-421`，`HandleEnemies`）是**唯一**导出 Boss 与敌人 HP 的命令；`STATE` 的 25 个 tracked 字段（`TrackedFields`，`rlharness.cpp:86-136`）里没有 Boss、没有敌人、也没有 `GameManager` flags。所以「Boss 不可观测」是**客户端缺口**，不是引擎缺口。

| 位置 | 变更 |
|---|---|
| `touhou_rl/v400/teacher/engine.py` | 新增 `ENEMY_FLAG_ACTIVE=1<<0`、`ENEMY_FLAG_BOSS=1<<1`（`EnemyManager.hpp:69-71`）、`ENEMIES_PAGE_LIMIT=16`、`parse_enemies_reply()`、`TeacherEngine.enemies()`（按 `offset` 分页，`max_pages=4`，溢出时报 `truncated=True`）、`TeacherEngine.enemies_snapshot()`（`ERR` 变成 `available=False` 的判定，不伪造 `boss=0`） |
| `tools/rl/v400_teacher.py` | 每决策 `payload["enemies"]`（`count/pages/truncated/boss_count/boss_life/boss_slots/life_sum/rows[slot,flags1,boss,life,timer,x,y]`）与 `payload["debug"]`（`curState/setupState/subthread/globals/stage/threads/frames`）；新增 `observe` 计时；报告新增 `enemies_available/enemies_observed_decisions/enemy_count_max/enemy_life_sum_max/boss_decisions/boss_first/boss_life_max`；新增 `--follow-stage` 与换面记录 `stage_transitions/stage_stats/stages_visited/stages_completed` |
| `tests/test_v400_teacher.py` | +7 项（上升沿在帧流里真的出现、period 1/0 的边界、换面行是「增量」不是「总量」、`ENEMIES` 解析、分页与截断、CLI 的 `_enemy_summary`/`_debug_snapshot`、`--follow-stage` 的下一面加载等待） |

`observe` 的代价实测 **0.135 ms/决策**（占单决策 ~90–140 ms 的 0.1%），且 `decide()` 完全不读这些字段，策略逐位不变。

### 2.1 `stageTransitionState` 仍然不可观测（明确记录，不推断）

`DEBUG`（`rlharness.cpp:696-723`）导出 `curState / setupState / subthread / globals / stage / difficulty / shot / isReplay / isDemo / threads / frames`，**没有** `stageTransitionState`、没有 `stageClearScreenState`、没有 `dialogue`。`STATE` 的 25 字段同样没有 `GameManagerFlags`。因此：

- 本卡把 `debug.stage_transition_state` **显式记为 `null`**（"不可观测"），不写 0；
- 换面判定只用引擎自己给的 `STATE.f1`（`GameManager.currentStage`）；
- 想把 §3.2 的状态机从「源码判据 + 行为推断」升级成直接观测，最小改动是在 `DEBUG` 里加 `stageTransition=0..3`（只读，不动协议语义）——**本卡不做**（禁止改引擎）。

---

## 3. 复核：上升沿假说被否证

### 3.1 源码：上升沿**不是**换面的前置条件

`GameManager.cpp:297-390` 的实际结构（逐行核对）：

```cpp
if (gameManager->flags.stageTransitionState != 0)
{
    if (gameManager->flags.stageTransitionState == 2)
    {
        ...                                   // 记分 / 通关数据
        g_GameManager.AdvanceToNextStage();   // ← 直接换面
        g_Supervisor.curState = SupervisorState_GameManagerReInit;
    }

    if ((((g_CurFrameInput & 0x1001) != 0) &&
         ((g_CurFrameInput & 0x1001) != (g_LastFrameInput & 0x1001))) ||
        g_GameManager.flags.isReplay || ...)
    {
        GM_FLAGS_WORD(gameManager) &= ~GameManagerFlags::STAGE_TRANSITION_STATE_MASK;
        if (g_GameManager.nextSupervisorState >= 0)
            g_Supervisor.curState = g_GameManager.nextSupervisorState;
    }
}
```

`AdvanceToNextStage()` 在 `stageTransitionState == 2` 时**无条件**执行；`0x1001` 的上升沿只负责**清状态位**并把 supervisor 切到 `nextSupervisorState`。所以「恒开 SHOOT 缺上升沿」只有在 `stageTransitionState` **已经等于 2** 时才可能成为卡点。

而 2 是**谁**置的：`Gui.cpp:1121-1127`，条件 `currentStage < STAGE6A && stageClear.clockDisplayCurrent != 0 && clockDisplayCurrent >= clockDisplayTarget && stageTransitionState == 0`；`stageClear` 本身由 `GUI_MSG_SHOW_STAGE_RESULTS`（`Gui.cpp:672-687`）初始化，即**本面结束序列**必须先把这条 msg 指令跑出来。实测 `stage_at_stop` 从头到尾是 1、`stageFrames.resets=0` ⇒ **`stageTransitionState` 从未到 2** ⇒ 上升沿假说在因果链的第一环就断了。

### 3.2 数据：空窗里有 ≈23–27 个上升沿，窗口一秒都没缩短

`.tmp_v400_clear_diag96.jsonl`（stage 1，`shoot_pulse_period=24` ⇒ 每 96 帧一个释放沿）逐 50 决策采样（节选）：

| dec | frame | stage | stageFrames | `DEBUG` stage | curState | setupState | enemies | **boss** | **boss life** | enemy life sum | tape 盒 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1300 | 5,204 | 1 | 5,206 | 1 | 2 | 0 | 4 | 0 | 0 | 209 | 82 |
| **1350** | **5,404** | 1 | 5,406 | 1 | 2 | 0 | 2 | **1** | **11,506** | 11,526 | 102 |
| 1500 | 6,004 | 1 | 6,006 | 1 | 2 | 0 | 2 | 1 | **6,754** | 6,774 | 144 |
| 1750 | 7,004 | 1 | 7,006 | 1 | 2 | 0 | 2 | 1 | **1,451** | 1,451 | 281 |
| 2000 | 8,004 | 1 | 8,006 | 1 | 2 | 0 | 2 | 1 | **62** | 62 | 208 |
| **2050** | **8,204** | 1 | 8,206 | 1 | 2 | 0 | 2 | 1 | **18,200** | 18,200 | **0** |
| 2100–2600 | 8,404–10,404 | 1 | +2/行 | 1 | 2 | 0 | 2 | 1 | **18,200（不动）** | 18,200 | **0** |
| 2700 | 10,804 | 1 | 10,806 | 1 | 2 | 0 | 2 | 1 | **17,278** | 17,278 | 199 |
| 2799 | 11,200 | 1 | 11,202 | 1 | 2 | 0 | 2 | 1 | 15,676→14,884 | 14,884 | 210 |

三件事同时成立：

1. **Boss 在，而且被打**：`boss=1` 从 frame ≈5,400 起一直在；`boss_life` 11,506 → 62（≈2,600 帧内打掉 11,444 HP），说明 SHOOT 一直是有效的、DPS 不差；
2. **空窗是 Boss 的下一阶段无敌/演出期**：frame 8,204 起 `boss_life` 精确冻结在 **18,200**，`tape_boxes.empty_frames == 48`（无敌 → 无致死盒）；最后一次观测到 18,200 是 frame 10,404，frame 10,804 已经是 17,278 ⇒ 冻结持续 **≈2,200–2,600 帧（37–43 s）**，然后 HP 继续往下掉。`curState` 全程 2（`SupervisorState_GameManager`）、`setupState` 全程 0（COMPLETE）、`subthread=0`、`gameplayFrameCounter` 逐帧 +1 ⇒ **不是加载、不是冻结、不是等换面**；
3. **这个窗口里发生了 ≈23–27 个 SHOOT 上升沿**（每 96 帧一个：`2,200/96≈23`、`2,600/96≈27`），窗口长度**一帧没变**。如果这个等待真由一个上升沿驱动，第一个脉冲（≤96 帧后）就会把它放走。⇒ **上升沿不是这个状态的门**。

### 3.3 两个假说并列（必须保留，因为它们是同一条因果链的两端）

| 假说 | 主张 | 支持证据 | 否决证据 | 裁决 |
|---|---|---|---|---|
| **H1（EXP-P1-009 §1.4 原结论）** | `stageTransitionState` 已经到「等换面」，但因为恒开 SHOOT 没有 `0x1001` 上升沿而**清不掉状态位** | `GameManager.cpp:381-389` 确实用 `0x1001` 上升沿清位；`still`+脉冲的对照（§1.5 (c)/(d)）里，脉冲让**死亡→标题→重开**这条流程动了 | ① `stageTransitionState==2` 分支本身已 `AdvanceToNextStage()`（§3.1）；② 实测 `stage_at_stop=1`、`stageFrames.resets=0`、`stage_clear=false` ⇒ 状态机从未到 2；③ 释放沿 208 个、`frames_without_shoot=208`，仍 0 次换面；④ 空窗内 ≈23–27 个沿没有缩短窗口 | **否决**（作为「不过面」的根因）。H1 描述的机制是真的，但它作用于**已经换面之后**的 supervisor 切换，不是换面本身 |
| **H2（本卡）** | 教师在 Stage 2 上是**帧预算不足**：多阶段 Boss（先 11,506→62，再 18,200）还在打，`GUI_MSG_SHOW_STAGE_RESULTS` 从未跑到，所以 `stageTransitionState` 根本没到 2 | §3.2 的 Boss HP 曲线；20,000 帧 = 333 s 而人类 Stage 2 是 18,525 帧（308.8 s）；`--stage 1` 就是 0-based Stage 2（§5） | **未决**：还没有一次跑到 Stage 2 的 Boss 死亡。若 40,000 帧仍无换面，H2 也要改（届时才轮到「保守策略打不死 Boss」） | **当前最简假说**，需要一次更长的 Stage 2 跑或 Stage 1 的过面来判定 |

**记录口径**：H1 不是"错在算错了某行代码"，而是"把**换面之后**的 supervisor 切换当成了**换面本身**的门；§1.5(c)/(d) 的对照确实证明"上升沿能让某个卡住的流程动"，但那条流程是**重试/标题**，不是 `AdvanceToNextStage`。

---

## 4. Stage 1（`--stage 0`）过面结果：**过一次面**

### 4.1 命令与结果

```bash
wsl -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_teacher.py \
  --decisions 5000 --max-frames 19887 --stop-on-death --stage 0 --difficulty normal --seed 1 \
  --mode full --quiet --out .tmp_v400_clear_stage0.jsonl > .tmp_v400_clear_stage0.log 2>&1; echo EXIT=$?'
# EXIT=0
```

| 指标 | 实测 | 说明 |
|---|---|---|
| **`stage_clear`** | **`true`** | 引擎自己把 `currentStage` 从 0 变成 1 |
| `stop_reason` | **`stage_changed`** | 不是 `max_frames`，不是 `death` |
| **换面帧号** | **13,676**（`frames_committed`） | 同时是 `survival_frames` |
| **换面决策号** | **3,418** | 4 帧/决策 |
| **换面后 `stage`** | **1**（`stage_at_stop=1`） | `stages_visited=[0,1]`、`stages_completed=1` |
| `stage_frames_at_stop` | **0** | `gameplayFrameCounter` 被 `RegisterChain` 清零 |
| `stage_frames_seen` | `{first 6, last 0, max 13,674, resets 1}` | **1 次 reset** = 真的重开了下一面的计数器 |
| 存活帧数 | 13,676（227.9 s 游戏内） | |
| 死亡 | **0** | `deaths_observed=0`、`deaths_final=0.0` |
| bomb | **0 次**（3→3） | `bomb_decisions=0`、`bombs_spent=0.0` |
| 墙钟 | 399 s（6 min 39 s） | |
| **实时倍率** | **0.5711×** | `13676 / (399.1×60)` |
| 决策/秒 | 8.566 | |
| 四段均值 (ms) | record **40.56** / search **45.88** / verify **27.83** / commit 1.54 / **observe 0.16** / events 0.05 | observe = 本卡新增的 `ENEMIES`+`DEBUG` 观测 |
| `empty_tape_streak_max` | 404 决策（≈1,616 帧） | 道中空窗，本轮**有界**且之后恢复 |
| 释放沿 | `shoot_releases=142`、`frames_without_shoot=142`、`shoot_fraction=0.9896` | |
| 一致性 | `agree_alive=10,257`、`agree_dead=0`、`geo_dead_engine_alive=0`、`geo_alive_engine_dead=0` | |

**对人类参考（Normal `Th8_ud2970.rpy` Stage 1 = 13,258 gameplay frames，EXP-P0-046）**：

- 教师 13,676 帧 = **1.032× 人类**；
- S1N 的 cap 是 `ceil(1.5 × 13,258) = 19,887`，教师用了 cap 的 **68.8%** ⇒ **帧预算达标**；
- 死亡 0 ⇒ G1「每面死亡 ≤0.5」的第 1 面 **0 ≤ 0.5 达标**。

### 4.2 Boss 是被打死的，不是逃掉/超时

| 字段 | 值 |
|---|---|
| `boss_first` | `{decision 733, frame 2,936, life 6,000, slots [1]}` |
| `boss_life_max` | **60,000** |
| `boss_decisions` | 2,100 / 3,419（61% 的决策有 Boss） |
| `enemy_life_sum_max` | 60,020 |
| `enemy_count_max` | 17 |

⇒ 第 1 面的 Boss（60,000 HP 量级）在 13,676 帧内被清空，`ENEMIES` 从 frame 2,936 起就能看到它。**这正是 EXP-P1-009 §6.2 说的"CLI 侧不可观测"被补上之后的直接收益**：没有 `boss/life` 这一列，就只能写"活着但不过面"，写不出"Boss 被打死了但帧不够"。

### 4.3 这一轮**没有**回答的问题（不许冒充结论）

1. **脉冲（释放沿）对过面是否必需？** 本跑带 `shoot_pulse_period=24`，所以只能说"带脉冲的配置能过"，**没有做 `--shoot-pulse-period 0` 的 A/B**，因此既不能说"脉冲修好了过面"，也不能说"脉冲无用"。§3.2 只说：上升沿不是那个 Boss 无敌空窗的门。
   - **但要记一笔**：`--follow-stage` 的实现过程中确实撞到 H1 机制的**真实**形态——换面后 `AddedCallback` 把 `stageTransitionState` 置 1，`GameplaySetupThread` 卡在 `while (stageTransitionState != 0) Sleep(17)`（`GameManagerSetup.cpp:426`），只有 `0x1001` 上升沿能放它走；不打这个沿，`PARK` 会一直返回 `ERR park-unsafe ... threads=2 setup=1`（实测 exit 5）。**所以上升沿对「换面之后能不能继续」是必需的，对「能不能换面」不是。** 这正是 H1 与 H2 的正确分工。
2. **第 2 面能不能过**：见 §6（多面推进跑）。
3. **1× 实时**：0.5711× 仍未达 1×（本卡不做性能优化；EXP-P1-006/010 负责）。

### 4.4 为什么**放弃**了「每 N 帧释放一帧」的参数扫描

任务书要求「若仍未过面」才做 N=30/60/120 帧扫描。实际执行顺序是：

1. 先按 `--stage 1`（=第 2 面）跑了 20,000 帧 → 未过面（§3.2 的 Boss 曲线显示它正打到第二阶段），于是准备了 `period=7/15/30`（28/60/120 帧）三个 arm；
2. 扫描脚本（`.tmp_v400_clear_scan.sh`）每 arm 只给 9,000 帧预算（0.57× 下约 4.4 min）；
3. **扫描起跑后才发现 Stage 1 的换面点在 13,676 帧 > 9,000 帧上限** ⇒ 三个 arm 都不可能看到换面，结论必然是"三个 arm 都没换面"，这是**上限截断**而不是周期无效。

因此按"不得把上限截断写成结论"的规矩**主动终止**了扫描（已 `pkill`，`.tmp_v400_clear_scan_p7.jsonl` 只写了一半并被删除），把预算换成 §6 的多面推进跑。**没有 N=30/60/120 的扫描结论**；要做得给每个 arm ≥19,887 帧（≈9.4 min × 3），本轮预算不允许。

---

## 5. `--stage` 是 0-based：口径订正（会影响历史卡片）

`ScoreDat.hpp:71-83` 的 `enum Stage { STAGE1, STAGE2, ... }` 从 **0** 开始，`STATE.f1 = GameManager.currentStage`。所以：

| CLI 参数 | `STATE.f1` | 人类（Normal `Th8_ud2970.rpy`，[bench/human_reference.json](../bench/human_reference.json) / EXP-P0-046） |
|---|---|---|
| `--stage 0` | 0 | Stage 1：**13,258** frames（0 死 0 bomb），S1N cap = `ceil(1.5×13258)` = **19,887** |
| `--stage 1` | 1 | Stage 2：**18,525** frames（0 死 0 bomb） |
| `--stage 2` | 2 | Stage 3：20,009 frames |

订正：

1. EXP-P1-009 §4 的 `--stage 1 ... --max-frames 20000` 打的是**第 2 面**，上限只有人类的 **1.08×**，不是 1.5×；「20,000 帧还没结束」这句子要按 Stage 2（人类 18,525）重读。
2. EXP-P1-008 §5 的「266.7 s 还没结束」同理，也不能和 13,258 帧的 Stage 1 对比。
3. **本卡起的所有运行都在报告里同时写 `args.stage` 与 `stage_at_stop`**，不写「stage 1」这种不带基数的说法。

---

## 6. 多面推进（G1「每面死亡 ≤0.5」的第一份真实数据）

**口径**：`--stage N` 是 **0-based**（`ScoreDat.hpp:71-83`，`STAGE1 = 0`），所以「第 1 面」= `--stage 0`、`stage=1` 表示**已经进到第 2 面**。下表同时写 1-based 与 `STATE.f1`，避免 §5 说的命名债。

`stage_transitions` / `stage_stats` 由 CLI 直接产出（EXP-P1-011 §2），每次换面写一行，**未完成的面是 `to_stage=None` 的 open 行**：

| 面（1-based） | CLI `--stage` | 进入帧 | 完成帧 | 面内帧数 | 死亡 | bomb | 是否换面成功 |
|---|---|---|---|---|---|---|---|
| 第 1 面 | 0 | 4 | **13,676** | **13,676** | **0** | **0** | **是**（`0 → 1`，`stageFrames` 13,674 → 0） |
| 第 2 面 | 1 | 13,736（含 60 帧加载） | **33,512** | **19,776** | **0** | **0** | **是**（`1 → 2`） |
| 第 3 面 | 2 | 33,592（含 80 帧加载） | — | 2,408（到 36,000 帧上限） | 0 | 0 | 未完成（帧上限） |

- 第 1 面：`stage_stats[0] = {decision 3418, frame 13676, from_stage 0, to_stage 1, frames_in_stage 13676, deaths_in_stage 0, bombs_spent_in_stage 0.0}`；
- 第 2 面：`stage_stats[1] = {decision 8362, frame 33512, from_stage 1, to_stage 2, frames_in_stage 19776, ...}`；
- 与之对照，**冷启动**在**第 2 面**（`--stage 1`）单跑 20,000 帧**没有换面**（`.tmp_v400_clear_stage1.jsonl`，`stage_at_stop=1`、`stage_clear=false`）。两者不矛盾：多面推进跑是从第 1 面带过来的 **power/item** 进入第 2 面的（DPS 更高），而冷启动跑是 base power 打第 2 面。所以"第 2 面需要多少帧"取决于进入该面时的资源，不能只用冷启动跑下结论。

### 6.1 单面读数（第 1 面；已被 §6.4 的 2 面表取代，保留为过程记录）

| 指标 | 第 1 面实测 | G1 门 | 判定 |
|---|---|---|---|
| 每面死亡 | **0** | ≤ 0.5 | **达标**（1 面样本） |
| bomb 使用 | **0**（3→3） | — | 记录 |
| 帧预算 | 13,676 = **1.032× 人类 13,258** | ≤ 1.5× (19,887) | **达标** |
| 是否过面 | **是**（`stage_clear=true`） | 必须过面 | **达标** |

**样本量声明**：这是 **1 面 × 1 seed（seed 1）× normal × 1 次**，不能当作 G1 的统计结论（G1 要 20 seeds / 多面），只能作为"该策略确实能过面、且能在人类帧预算内过面"的存在性证据。

### 6.2 多面推进跑（`--follow-stage`，36,000 帧）：**连续过 2 面**

```bash
wsl -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_teacher.py \
  --decisions 9000 --max-frames 36000 --stop-on-death --stage 0 --difficulty normal --seed 1 \
  --mode full --quiet --follow-stage --out .tmp_v400_clear_multi.jsonl > .tmp_v400_clear_multi.log 2>&1; echo EXIT=$?'
# EXIT=0
```

| 指标 | 实测 |
|---|---|
| `stage_clear` / `stop_reason` | **`true`** / `max_frames`（不是 `death`，全程 0 死） |
| `stages_visited` / `stages_completed` | **[0, 1, 2]** / **2** |
| `stage_at_stop` | **2**（0-based，即已进入第 3 面） |
| `stage_frames_seen.resets` | **2**（`gameplayFrameCounter` 被清零两次） |
| 帧 / 决策 | 36,000 帧 / 8,965 决策 |
| 死亡 / bomb | **0** / **0**（3→3） |
| 墙钟 / 实时倍率 | **1,301 s（21 min 41 s）** / **0.461×** |
| `stage_transitions` | 见 §6.3 |
| `events` | `stage_changed_0_to_1_at_3418`、**`stage_load_wait_60_frames`**、`stage_changed_1_to_2_at_8362`、**`stage_load_wait_80_frames`** |
| Boss 观测 | `boss_decisions=5,526`、`boss_life_max=60,000`、`enemy_count_max=32` |
| 释放沿 | `shoot_releases=373`、`shoot_fraction=0.9896` |

**`stage_load_wait_*` 是这轮的第二个修复**：换面后 `PARK` 会 fail-closed 为 `ERR park-unsafe role=shadow-required threads=2 setup=1`（实测第一次跑直接 `EXIT=5`，`stop_reason=engine_error`）。原因是 `AddedCallback` 把 `stageTransitionState` 置 1、`GameplaySetupThread` 卡在 `while (stageTransitionState != 0) Sleep(17)`，只有 `0x1001` 上升沿能放行。CLI 现在换面后用「SHOOT 保持 7 帧 + 释放 1 帧」走 60/80 帧把加载走完再继续决策（`_advance_until_playable`，单测 `test_follow_stage_walks_the_next_stage_load_out`）。

### 6.3 每面数据（G1 口径，`stage_transitions` + open 行）

| 面（1-based） | CLI `--stage` | 换面帧（完成帧） | 面内 gameplay 帧 | 死亡 | bomb | 是否换面成功 |
|---|---|---|---|---|---|---|
| **第 1 面** | 0 | **13,676** | **13,676** | **0** | **0** | **是**（`0 → 1`） |
| **第 2 面** | 1 | **33,512** | **19,776** | **0** | **0** | **是**（`1 → 2`） |
| 第 3 面（未完成） | 2 | — | 2,408（到 36,000 帧上限） | 0 | 0 | 未完成（**帧上限**，不是卡住） |

- 第 1 面换面后等了 **60 帧**加载；第 2 面换面后等了 **80 帧**加载。这两段不计入"面内 gameplay 帧"。
- 人类（Normal `Th8_ud2970.rpy`，EXP-P0-046）：Stage1 13,258 → 教师 **1.032×**；Stage2 18,525 → 教师 **1.0675×**。两面都在 1.5× cap 内。
- 第 3 面只跑了 2,408 帧（到上限），**没有任何关于第 3 面能否过面的结论**。

### 6.4 G1 口径的第一次真实读数（2 面样本）

| 指标 | 第 1 面 | 第 2 面 | G1 门 | 判定 |
|---|---|---|---|---|
| 每面死亡 | **0** | **0** | ≤ 0.5 | **两面均达标** |
| bomb 使用 | 0 | 0 | — | 记录 |
| 帧预算（对人类的倍数） | 1.032× | 1.0675× | ≤ 1.5× | **两面均达标** |
| 是否过面 | **是** | **是** | 必须过面 | **两面均达标** |

**样本量声明**：**2 面 × 1 seed（seed 1）× normal × 1 次**，只能证明「该策略确实能连续过两面、且在人类帧预算内、且 0 死 0 bomb」，**不能**当作 G1 的统计结论（G1 要 20 seeds / 多面 / 多难度）。

### 6.5 并发干扰（必须记录，影响实时倍率）

本轮 36,000 帧跑在 14:00 前后与**另一工作包的 G1 S1N 20-seed 扫描**（`--stage 0 --max-frames 19887 --seed 1003 ... .tmp_v400_g1/s1n/s1n_seed1003.jsonl`）在 WSL 里重叠运行，**同时有 2 个 `th08-modern` 实例**，违反「WSL 引擎并发 ≤1」。后果：

- 本跑的实时倍率 **0.461×**（无干扰时应 ≈0.57×），`record/search/verify` 全部变慢（53.4 / 58.1 / 30.5 ms，对照 §4.1 的 40.6 / 45.9 / 27.8）；
- 但**判定类结论（换面帧号、死亡、bomb）不受影响**：游戏逻辑是帧驱动的，与墙钟无关；`stage_frames`、`stage_transitions`、`deaths` 全部由引擎状态给出；
- 两个跑都用 `--out` 各自落盘，没有共享状态。

---

## 7. 产物（仓库根，`.tmp_v400_clear*`）

| 文件 | 内容 |
|---|---|
| `.tmp_v400_clear_stage1.jsonl` / `.log` | 修复后重测（stage 1 = 第 2 面，5,000 决策 / 20,000 帧） |
| `.tmp_v400_clear_diag96.jsonl` / `.log` | **本轮**带 `DEBUG`+`ENEMIES` 的诊断跑（stage 1，period 24 = 96 帧，2,800 决策 / 11,200 帧，`--follow-stage`） |
| `.tmp_v400_clear_stage0.jsonl` / `.log` | **过面跑**：`--stage 0`（第 1 面），13,676 帧过面，`EXIT=0` |
| `.tmp_v400_clear_multi.jsonl` / `.log` | **多面推进跑**：`--stage 0 --follow-stage --max-frames 36000 --stop-on-death`，**连续过 2 面**（13,676 / 33,512） |
| `.tmp_v400_clear_smoke.jsonl` / `.log` | 40 决策冒烟（验证 `ENEMIES`/`DEBUG` 已进日志） |
| `.tmp_v400_clear_diag50.py` | 每 50 决策的 STATE/DEBUG/ENEMIES 表（本卡 §3.2 的表就是它出的） |
| `.tmp_v400_clear_diag_analyze.py` / `.tmp_v400_clear_show.py` | 通用采样器 / 报告摘要器（`show.py` 打印 `stage_transitions`、`boss_*`、`enemies_*`） |
| `.tmp_v400_clear_scan.sh` / `.tmp_v400_clear_scan_wait.sh` / `.tmp_v400_clear_scan.log` | 释放沿参数扫描（N=30/60/120 帧 = period 7/15/30）驱动脚本；**本轮主动终止**，`.tmp_v400_clear_scan_p7.jsonl` 已删除（见 §4.4） |

## 8. 修订记录

| 日期 | 修改 | 原因 |
|---|---|---|
| 2026-11-09 | 初版 | `ENEMIES`/`DEBUG` 进日志 + 上升沿假说复核 + Stage 1（`--stage 0`）过面重测 |
| 2026-11-09 | 补 §6.2–§6.5 | `--follow-stage` 连续过 2 面；修 `stage_load_wait`（换面后 setup 线程未完成时 `PARK` fail-closed，实测 exit 5）；记录与 G1 20-seed 扫描的并发重叠 |
