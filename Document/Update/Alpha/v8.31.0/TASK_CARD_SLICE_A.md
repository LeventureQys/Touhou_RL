# V831-SLICE-01 A 节：短阶段任务卡候选

状态：**部分可冻结（缺项 8 条，见 §5）**

> 起草：研究端，只读。2026-11-25。本卡是本任务**唯一**写入的文件；未启动引擎、未训练、未采集、未改任何既有文件/证据/`state.json`/`journal.jsonl`/资产、未提交。
> 全卡统一来源标签：`研究端离线复算` = 本轮在本检出上的只读计算（哈希、JSON 键值、算术）；`历史记录` = 既有登记文档/产物 JSON/v8.30 回执里训练端上一轮的核实；`待核验` = 无可用证据。`训练端本次实测` / `训练端本次离线检查` 在本卡为 **0 条**（那两类标签属于 B 节回执，见 §7）。
> 引用格式：相对仓库根的路径 + 行号或 JSON 键路径。行号以本轮读到的文件为准。

## 0 结论一句话

现有登记材料里**唯一**可辨识的"攻击完成"是整段 stage 1 的 `stage_clear`（同约束教师磁带 seed 12，受控帧 **13,195**，`shoot_pulse_period=48`）；**不存在任何"短子阶段"的事件定义，也不存在可用的中局合法前缀**，而计划里的 600 帧窗口只覆盖到完成点的 **4.55%**。因此本卡可以冻结"阶段身份 / 起点类型 / 两套完成口径 / 合法动作 / 两类见证"，**不能**冻结"短"这一属性与"短阶段终态可辨"——按工单第 25 行的口径，A 的缺项随 B 的接线结果一起回报，不硬凑假短阶段。

## 1 起止和分母

### 1.1 逐项

| 子项 | 可冻结值 / 候选 | 证据（路径 : 行或键） | 来源 | 冻结 |
| --- | --- | --- | --- | --- |
| 攻击/阶段身份 | `th08-stage1-normal-team0` = stage 0 / difficulty normal / **shotType 0（Border Team）** / 非 practice；教师 seed 12，像素接线 seed 1000 | `configs/v430/splits.json` → `ancestors[0].task_family`、`ancestors[0].note`、`roots.engine_binary_relative`；`Document/Update/Alpha/v8.30.0/TRAINING_HOST_REPLY_01.md` §1 行「`team` 参数（**更正**）」（第 48 行）与行「机体（**本回合已定**）」（第 49 行） | 研究端离线复算（splits 键值）+ 历史记录（team=shotType 机理，源自 `rlharness.cpp:1142-1146,1163-1165`，本检出无该源码） | ✅ |
| 安全前缀身份 | 教师磁带 `slots.jsonl`，sha256 `f759de7dacc78343dec8825da5de14942e749b8a5a2c5319304e513d1618b7b9`，6,588 槽 | `.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed12-shard/slots.jsonl`（本轮 `Get-FileHash` 逐位相同）；同值登记在 `.tmp_v420_evalpulse/positive_control.json` → `tape.shard_sha256_frozen`；v8.31 工单第 82 行同一常量 | 研究端离线复算 | ✅ |
| 起点类型 | **只有自然开局**：`RESET <stage> <difficulty> <team(=shotType)> <seed> [practice]`，**无起始帧参数**；`ArmStart()` 由标题画面 Ready 状态自消费，外部不能强改 | v8.30 回执 §1 末行（第 50 行）、§9 第 1 项（第 210 行）；机理出处 `rlharness.cpp:981,1004,1142-1165`、`rlharness.hpp:43-51`（**本检出无此源码，未复读**） | 历史记录 | ✅（"只能自然开局"） |
| 首受控帧 | 自然开局：受控帧从 **RESET 后第 0 索引**开始（STEP 计数与 `STATE.frame` 均为 1 起算的 13,195，`gameplay_frame=13,196`） | v8.30 回执 §1 行「起止事件」（第 36 行，`rlharness.hpp:68-73`）；`.tmp_v420_evalpulse/positive_control.json` → `frames` / `state_frame` / `gameplay_frame` / `pixels_warmup_frames=0`；`tools/rl/v420_eval_shoot.py:792-793`（`state_frame=int(state["frame"])`） | 历史记录 + 研究端离线复算 | ⚠ 见缺项 A1 |
| 教师磁带的首个"录制"决策帧 | `decision_frame = 16`：磁带前有 **8 个前导决策 × 2 帧 = 16 帧**（`leading_decisions=8`），首个被录制的决策在帧 16；末个在帧 13,190 | `.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed12-shard/meta.json` → `counts.first_decision_frame` / `counts.last_decision_frame`；`.tmp_v420_evalpulse/positive_control.json` → `tape.leading_decisions=8`、`tape.last_recorded_decision_frame=13190` | 研究端离线复算（两数互证：8×2=16） | ✅ |
| 结束事件（主口径） | **(a) `STATE.stage` 前进**：`stage_start=0 → stage_end=1` + `stage_clear=true` + `stop_reason="stage_clear"` + `exit="controller"` | `.tmp_v420_evalpulse/positive_control.json` → `stage_start`/`stage_end`/`stage_clear`/`stop_reason`/`exit`/`referee_state_source`（= `engine STATE reply after each stepped frame`） | 研究端离线复算（原样字段） | ✅ |
| 结束事件（交叉口径） | **(b) `EVENTS` 0x0080 电平 + `stageClearEdgeFrame` 边沿** | 实现存在：`tools/rl/v400_stage_clear_replay.py:579-586,628`（记录 `stageClearEdgeFrame`）、`tools/rl/v400_p2_reward_detector.py:2486`（`STAGE_CLEAR_BIT = 0x0080`）；机理：v8.30 回执 §1 行「完成定义（可核）」（第 37 行，`rlharness.cpp:82-124`） | 研究端离线复算（代码存在性）+ 历史记录（机理） | ❌ 见缺项 A2 |
| 两套口径的关系 | 现有产物里**从未同时出现**：(a) 出现在全部评测行，0 条记录带 `stageClearEdgeFrame`；"应当一致"未验证 | v8.30 回执 §1 行「两套口径的差别」（第 39 行，自认缺口）、§9 第 6 项（第 215 行）；`positive_control.json` 顶层键表无 `stageClearEdgeFrame` | 研究端离线复算 | ❌ A2 |
| 失败定义 | ①首死：`stop_on_death=True` → `stop_reason="death_stop"`；②命尽：`stop_reason="game_over"`, `exit="game_over"`, `deaths=3.0`, `lives_end=0.0` | `Document/Update/Alpha/v4.2.0/evidence/T08/CLEAR_REMEASURE.md:51-60`（diag 全 `death_stop:20`，s1n 全 `game_over:20`）；`.tmp_v420_evalpulse/negative_control.json` → 上述 5 个字段 | 历史记录 + 研究端离线复算 | ✅ |
| 截断定义 | 达到 `max_frames` 上限即截断（登记上限 19,887）；技术无效 = `error != null` 或 `exit ∉ {controller, game_over}` | `CLEAR_REMEASURE.md:23`（`--max-frames 19887`）；`v420_eval_shoot.py:1027`（`frames > max_frames` 拒绝）；登记 160 局 errors=0/invalid=0（`CLEAR_REMEASURE.md:62`） | 历史记录 | ⚠ 定义可冻，**无见证**（A3） |
| 每局候选受控帧上限 | 登记过的三个值：**19,887**（T08 评测/教师）、**600**（像素切片）、**13,195**（完成点这一局的实际长度） | `CLEAR_REMEASURE.md:23,211`；`Document/Update/Alpha/v8.30.0/PIXEL_SLICE_PLAN.md:25,88`；v8.31 工单第 34、43 行；`positive_control.json` → `frames=13195`、`max_frames=19887` | 历史记录 + 研究端离线复算 | ❌ 见缺项 A4 |
| 分母 | 登记能力分母 = **4 臂 × 2 口径 × 20 dev seed = 160 局，`stage_clear=0`**；完成见证 = **1 局教师磁带**；另有 100 局教师确认批（61300–61399，100/100 过关）但该批已 spent | `CLEAR_REMEASURE.md:3-8,51-64`；`EVAL_SHOOT_PULSE.md:151-158`；`configs/v430/splits.json` → `confirm_batches[3]` | 历史记录 | ✅（作为分母口径） |

### 1.2 完成点的精确位置（三个数字，必须一起引）

| 量 | 值 | 键 | 说明 |
| --- | ---: | --- | --- |
| 受控帧 / STEP 计数 | 13,195 | `positive_control.json` → `frames` | 6,598 决策 × 2 帧 = 13,196，而 `frames = 13,195 = 6,598×2 − 1` ⇒ **至少有一个决策只执行了 1 帧**（最可能是末决策在完成事件处被终止；研究端离线复算，属算术恒等式 + 单点推断） |
| `STATE.frame` | 13,195 | 同上 → `state_frame` | 引擎自己的帧计数（1 起算） |
| `gameplay_frame` | 13,196 | 同上 → `gameplay_frame` | `= state_frame + 1` |
| 磁带自身最后一帧 | 13,190（末决策起始）/ 13,191（其第二帧） | `tape.last_recorded_decision_frame=13190`；`meta.json:counts.last_decision_frame=13190` | 完成点在磁带末尾之后 **3–4 帧**（差 1 取决于编号基，见 A1） |
| 这 3–4 帧由谁提供 | **`--tail 64` 的重复动作**，不是教师录制帧 | `v420_eval_shoot.py:1478-1479`（`--tail` = "tail decisions repeated after the tape ends"）；`positive_control.json` → `tape.tail_decisions=64`、`tail_generated_decisions=0`（完成落在 64 个 tail 决策之内） | ⚠ 见 §7 不主张第 4 条 |

> 这一条是 A 节最容易被漏掉的边界：**"同约束教师成功见证"的最后 3–4 帧来自运行器的 tail 重复，而不是教师录制的动作。** 它不使见证失效（父本动作在磁带里，tail 只是按住最后一个动作），但任何把它写成"教师录制到过关帧"的表述都是过头的。

### 1.3 为什么"短"冻不下来（硬算术）

| 量 | 值 | 来源 |
| --- | ---: | --- |
| 计划中的短切片上限 | 600 受控帧 | `PIXEL_SLICE_PLAN.md:25`（"至多 600 个受控游戏帧"）、`:88`；v8.31 工单第 33、43 行 |
| 完成点 | 13,195 帧 | 同 §1.2 |
| 覆盖率 | **600 / 13,195 = 4.55%** | 研究端离线复算 |
| 缺口 | 12,595 帧 | 同上 |
| 现有最短的"完成类"事件 | **仍只有 stage_clear**；登记里不存在更早的攻击/转场/击破事件 | `configs/v430/splits.json` 全部 360 条 `ancestors` 跨 **8 个 stage 族**（stage1 **242** / stage2 16 / stage3 16 / stage4a 17 / stage4b 16 / stage5 16 / stage6a 17 / stage6b 20），**没有任何段内（sub-stage）粒度**；`meta.json:contract.termination_kinds` 只登记五类终止（`first_death` / `game_over` / `stage_clear` / `cap` / `error`），没有任何段内事件 |

**结论**：A 若采用 600 帧上限，"阶段完成"在**构造上不可观测**；A 若要观测完成，上限必须 ≥13,195 + 余量（登记过的 19,887 就够），但那就**不是短阶段**，而是整段 stage 1（该局实测墙钟 121.573 s，`positive_control.json` → `wall_s`）。二者不能同时满足，这是缺项 A4，也是全卡最关键的风险。

## 2 任务信号（成功 / 失败或不同后果）

### 2.1 成功（唯一同约束见证）

| 项 | 值 | 证据 | 来源 |
| --- | --- | --- | --- |
| run / record | `.tmp_v420_evalpulse/positive_control.json`，sha256 `9cdd90335edcc5a5d2afcf3b1e18d7713e921099967dbf4c62b5b84ef3dbaf63`（本轮 `Get-FileHash` 与登记值逐位相同） | `Document/Update/Alpha/v4.2.0/evidence/T08/eval_shoot_pulse.json` → `positive_control.sha256` | 研究端离线复算 |
| 身份 | schema `v420-eval-shoot-tape/1`，`controller=tools/rl/v420_eval_shoot.py`，`kind=replayed-tape-control`，seed 12，stage 0，normal，`decision_frames=2`，`shoot_pulse_period=48`，引擎 `691bfd77…a568ef6` | 同上 → `positive_control`、`env.engine_sha256`；`EVAL_SHOOT_PULSE.md:166-173` | 历史记录 |
| 原始事件 | `stage_start=0`，`stage_end=1`，`stage_clear=true`，`stop_reason="stage_clear"`，`exit="controller"`，`deaths=0.0`，`lives_end=2.0` | `positive_control.json` 同名字段 | 研究端离线复算 |
| 帧号 | `frames=13,195`，`state_frame=13,195`，`gameplay_frame=13,196`；`score_at_frames={"13195": 2631430}` | 同上 | 研究端离线复算 |
| 动作机制 | 6,598 决策；`releases_planned=releases_applied=137`；`shoot_frames=13,058`；`no_shoot_frames=137`；`shoot_rising_edges=138`；`mask_trace_sha256=66c81344…f179924` | 同上 → `shoot.*`；`EVAL_SHOOT_PULSE.md:181-186` | 研究端离线复算 |
| 墙钟 | `wall_s=121.573`（`--timeout 240` 未触顶） | 同上 → `wall_s` | 研究端离线复算 |
| 来源性质 | 教师磁带重放，**不是策略**（`not_a_gate=true`、`not_a_policy_result=true`） | 同上；`EVAL_SHOOT_PULSE.md:190,277-279` | 历史记录 |

### 2.2 失败 / 不同后果（三条，按"同约束"强度排序）

| # | 项 | 值 | 证据 | 来源 |
| --- | --- | --- | --- | --- |
| 1 | **同磁带同通道、只改一个变量** | `period=1,000,000,007` → `stage_clear=false`，`stage_start=stage_end=0`，`frames=15,564`，`stop_reason="game_over"`，`exit="game_over"`，`deaths=3.0`，`lives_end=0.0`，`score=163,945`，`score_at_frames={"13195": 141,379}`，`releases_applied=0`，`shoot_rising_edges=1`，`mask_trace_sha256=92cea74d…feba10`，record sha256 `ecb58cb4…` | `.tmp_v420_evalpulse/negative_control.json`（本轮复算哈希与登记一致）；`T08/eval_shoot_pulse.json` → `negative_control.*`；`EVAL_SHOOT_PULSE.md:194-210` | 研究端离线复算 |
| 2 | 策略分母（160 局） | 4 臂 × 2 口径 × 20 dev seed，`stage_clear=0`，errors=0，invalid=0；最深 3,914 帧（D3，seed 1010，s1n），距 13,195 差 9,281 帧；无局达 12,000 帧 | `T08/CLEAR_REMEASURE.md:3-8,51-64`；`T08/clear_remeasure.json` → `arms.<臂>.<口径>.episodes` | 历史记录 |
| 3 | 首死口径 | diag 20/20 局 `death_stop`，均值 1,597.9–1,757.5 帧 | `T08/CLEAR_REMEASURE.md:53-59`；v8.30 回执 §2 行「失败的三类分离」（第 67 行） | 历史记录 |

### 2.3 "可观测完成"如何区别于自动推进/超时

| 区分手段 | 读数 | 证据 | 来源 |
| --- | --- | --- | --- |
| **不是自动推进** | no-pulse 局在 `stage_end=0` 下把分数**冻结在 141,379 共 8,187 帧**（第 5,058 → 13,245 帧），到 15,564 帧才 game_over ⇒ 换面等待态等的是输入沿，不是时间 | `T08/clear_reachability.json` → `q2_experiment.runs[eval].longest_constant_score = {score:141379, floor_frame:5058, until_frame:13245, frames:8187}`；`EVAL_SHOOT_PULSE.md:196-204` | 历史记录 |
| **不是超时** | 两次对照的终止来源是 `exit=controller`（positive）/`exit=game_over`（negative）；`--timeout 240` vs 实测 121.573 s / 138.9 s；`max_frames=19,887` 均未触顶 | `positive_control.json` → `exit`/`max_frames`/`wall_s`；`negative_control.json` 同；`EVAL_SHOOT_PULSE.md:164,198` | 研究端离线复算 |
| **不是进度累积** | 同一帧预算（13,195）下两条通道的分数是 2,631,430 vs 141,379（同磁带、同引擎、只差 `shoot_pulse_period`）；`stage_start→stage_end` 只在带脉冲那条发生 | 同 §2.1、§2.2 第 1 行；`T08/clear_reachability.json` → `three_numbers` | 研究端离线复算 |
| **可观测完成的正读数** | `stage_start=0 → stage_end=1` + `stop_reason="stage_clear"` + `exit="controller"`：这是引擎 STATE 每步回包给出的阶段计数前进 | `positive_control.json` → `referee_state_source`（= `engine STATE reply after each stepped frame`）、`stage_start`/`stage_end`；`T08/clear_reachability.json` → `q2_experiment.channel.state_after_every_frame=true` | 研究端离线复算 |
| 边界 | 该"完成"是**阶段计数前进**，不是 `EVENTS` 边沿；两者未在同一条记录里核对过（A2） | 同 §1.1 | — |

## 3 入口和资源

| 子项 | 可冻结值 / 候选 | 证据 | 来源 | 冻结 |
| --- | --- | --- | --- | --- |
| 入口（唯一可冻） | 自然开局：stage 0 / normal / shotType 0 / seed 12（教师）或 1000（像素冒烟）/ 非 practice | v8.31 工单第 33-34 行；`configs/v430/splits.json` → `ancestors[0]` | 历史记录 + 研究端离线复算 | ✅ |
| 入口（中局） | **不存在**：无起始帧入口；合法前缀需"一决策一 mask"磁带，R74 已定位首个冲突帧 92、目标段 8,664，四条解阻路已排除三条 | `Document/Update/Alpha/v4.3.1/evidence/R74/pair_cand_scope.json:39,42-58,63`；v8.30 回执 §3 第 79 行、§9 第 2 项（第 211 行） | 历史记录 | ❌ A5 |
| 已登记前缀 SHA | `f759de7d…18b7b9`（6,588 槽，16 帧前导，首/末录制决策帧 16 / 13,190） | 同 §1.1 | 研究端离线复算 | ✅（身份）／❌（作为"合法前缀"，见 A5） |
| 前缀完整性锚（可当恢复锚用） | `corpus_hash_post_checks=3,298` 且 `corpus_hash_post_equal=3,298`；`corpus_frame_hashes_observed=13,192`（=16+6,588×2 ✓）；`dataset_hash_chain_breaks=0`；`label_mismatches=0`；`dataset_chain_head_matches_cold_start=true`；`corpus_reset_hash_matches_first_pre=true` | `.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed12-shard/meta.json` → `hash.*` | 研究端离线复算 | ✅ |
| 中局恢复哈希 | **不存在**（无中局入口）；R74 的"0 不符"是三臂可比前提，不能替代 | `R74/pair_cand_scope.json:40,63` | 历史记录 | ❌ A5 |
| 实际姿态与资源（源码级） | Normal/非 practice/非 replay 普通开局：`livesRemaining = cfg->lifeCount = 2`（2 条备用残机 ⇒ 允许 3 次死亡）；`bombsRemaining = primaryShtFile->initialBombCount`（**不是** `cfg.bombCount=3`）；`playerPower = 0`；practice 会把 lifeCount 强制成 8（**禁用作自然起点**）；replay 路径不走资源初始化 | v8.30 回执 §1 行「初始资源（**源码已核，替换先验**）」（第 45 行）、行「资源语义的三个陷阱」（第 46 行）、§10.1 第 263 行；机理出处 `GameManagerSetup.cpp:105-130`、`Player.cpp:1393-1400`（**本检出无此源码，未复读**） | 历史记录 | ⚠ 部分 |
| 实际姿态与资源（帧 0 实测） | **无任何登记记录**（x, y, power, lives, bombs, graze 都没有帧 0 读数）；只有终局资源 `lives_end=2.0`（positive）/`0.0`（negative） | `positive_control.json` → `lives_end`；反向：无 `STATE` 帧 0 字段可用 | 研究端离线复算（缺失本身）+ 待核验 | ❌ A6 |
| 像素帧栈 | obs-v3：`FRAME_STACK=4`（`t-3..t`，oldest first）；`PLAY_SHAPE=(112,96)`；`HUD_SHAPE=(56,160)`，`HUD_FRAMES=1` | 直接 import `touhou_rl/v400/student/obs_spec_v3.py`（本轮运行）；与 `meta.json:contract.frame_stack=4 / play_shape=[112,96] / hud_shape=[56,160]` 一致 | 研究端离线复算 | ✅ |
| 动作历史 | `ACTION_HISTORY_DECISIONS=8`，`ACTION_HISTORY_DIM=144`，`ACTION_HISTORY_PAD` = **all-zero row**，`ORDER` = `a[t-8]…a[t-1]`（oldest first） | 同上；`meta.json:contract.action_history=8` | 研究端离线复算 | ✅ |
| 填充规则 | 自然开局：obs-v3 既有初始填充（重复最早可用画面，**不额外推进游戏帧**）；接管语义下"接管前动作历史用零行" | v8.31 工单第 40 行（§帧与判分验收 1）；`PIXEL_SLICE_PLAN.md:48-50`；`ACTION_HISTORY_PAD` 即零行（本轮核实） | 历史记录 + 研究端离线复算 | ✅ |
| **合法动作（可冻结）** | **18 候选**（`cand = focus*9 + move`；9 方向 = 0 idle / 1 N / …/ 8 NW；**射击恒开、bomb 恒关**），学生动作 `(cand,bomb)`=36；**SHOOT 脉冲是外生运行器约定**：每 48 决策在该决策**最后**一帧放掉 `SHOOT\|BOMB`（`decision_frames=2` ⇒ 周期 96 帧，占射击时间 ≈1.04%）；bomb 只挂该决策第一个子决策 | v8.30 回执 §1 行「动作库（**三套，别混用**）」（第 42 行）、§13.1–13.3（第 341–375 行）、§13.6（第 400–416 行）；`touhou_rl/v400/teacher/shoot.py:43-47,50-55,58-80`；`T08/EVAL_SHOOT_PULSE.md:60-81`；v8.31 工单第 29 行（诊断只用 18 类移动 + focus，`bomb=0`） | 历史记录 + 研究端离线复算（shoot.py 规则） | ✅（规则）／⚠ 脉冲是否属**任务定义**未裁定（§7 第 5 条） |
| **具体冲突 1（历史口径）** | 教师磁带首决策（`decision_frame=16`）自带 `history=[7,7,7,7,7,7,7,7]`（8 个 one-hot(7)），**不是** obs-v3 的零行 pad；同槽 `obs_render_frames=[12,13,14,15]`、`history_frames=[0,2,…,14]` | `.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed12-shard/slots.jsonl` 第 1 行（slot_id 0）；`meta.json:counts.label_rows_cold_start=8` | 研究端离线复算 | ⚠ 需在卡内写死"像素用零行 / 教师重放用磁带行"，工单第 40 行已给两条规则，无需新证据 |
| **具体冲突 2（双 mask）** | 契约要求同一 2 帧决策的两帧 mask 相同（`DECISION_FRAMES=2`）；R74 登记**首个冲突帧 = 92**（即帧 91/92 不同）、目标段 = **帧 8,664**；**具体掩码未登记** | `touhou_rl/v420/counterfactual.py:3924-3940`（`_pair_cand`，错误文本会带 `0x…. then 0x….`）、调用点 `:3988-3989`；`R74/pair_cand_scope.json:12-19,30-34,39` | 研究端离线复算（契约）+ 历史记录（帧号） | ❌ A7（具体掩码） |
| 双 mask 的歧义（本轮发现） | T04 seed12 磁带**每槽只存 1 个 mask**（slot 0：`mask=65`，`commit_frames=[16,18]`，`duration_frames=2`），但 `executed ≠ chosen` 的槽有 **2,105/6,588**；R74 未登记它用的 pair prefix 是哪个文件 ⇒ "双 mask"指哪种配对**待核验** | `slots.jsonl` 第 1 行 + 本轮扫前 60 行；`meta.json:counts.slots_executed_neq_chosen=2105`；`R74/pair_cand_scope.json` 全文无文件名 | 研究端离线复算 + 待核验 | ❌ A7 |

## 4 祖先用途（训练根 / 开发根 / 冲突）

| 类别 | 现状 | 证据 | 来源 |
| --- | --- | --- | --- |
| 祖先登记总数 | **360** | `configs/v430/splits.json` → `ancestors` | 研究端离线复算 |
| 训练根（可划） | **328** 条，跨 **8 个 stage 族**（stage1 **210** / stage2 16 / stage3 16 / stage4a 17 / stage4b 16 / stage5 16 / stage6a 17 / stage6b 20）；来源 `teacher_query` **225** / `dagger_round` **98** / `policy_rl` **5**；`ancestors[0]` = `teacher_query|d1-teacher-fast|d1-s0d1-n3371-seed1-shard`，`note="difficulty fixed at 1 by the corpus"` | 同上 → `splits.train.ancestor_ids`、`ancestors[0]`；族与来源直方图由主代理复核脚本按祖先 id 末段重算 | 研究端离线复算（**主代理已复核并更正族/来源计数**） |
| 训练根曝光 | 25 个登记来源中 **17 个 `used_for_training`** / 2 `used_for_selection` / 3 `evaluated_only` / 1 `rejected_not_used` / 1 `rule_based_no_training_data` / 1 `executed_instrument_only` | `Document/Update/Alpha/v4.3.0/registries/pretraining_sources.json`；v8.30 回执 §5 行「曝光/同源」（第 121 行） | 历史记录（分类）+ 研究端离线复算（登记文件存在且摘要 4/4 相符） |
| 开发根（"未参与拟合或选参"） | **登记里没有**：`splits.dev` 的 20 条（seeds 1000–1019，`model_selection|v420-dev-panel`）是**旧开发面板**，已在 T08/T05/T06 的臂比较与选参中反复使用（`CLEAR_REMEASURE` 的 160 局全部落在这 20 个种子上） | `configs/v430/splits.json` → `splits.dev.ancestor_ids`、`splits.dev.purpose`（"never training data"，但含 candidate-selection）；`T08/CLEAR_REMEASURE.md:25,51-60` | 研究端离线复算 + 历史记录 |
| 可用的候选干净根 | `splits.dev.reserved_seeds` = **1020–1039**（"reserved for pre-registered ablations"）；**尚未登记为 ancestors，也没有任何运行证据** | `configs/v430/splits.json` → `splits.dev.reserved_seeds` | 研究端离线复算 | ❌ A8 |
| 正式确认 | `splits.confirm` = **0**（v4.3.0 未冻结确认批）；历史确认批 **61300–61399 已 spent**（100 局教师，100/100 过关），**不可复用** | `splits.confirm` / `confirm_batches[3]`；`Document/Update/Alpha/v4.1.0/evidence/WP02/G4101_100_TEACHER.md`（本卡未读，帧级细节待核验） | 研究端离线复算 |
| 适应保留 | `splits.adapt` = **0**（仅声明 2 个目标族，无 episode） | `splits.adapt` | 研究端离线复算 |
| 隔离 | `splits.quarantine` = **12**（S1b，dev seeds 1000–1011，"种子来源被拒"；源 `teacher_query|d1-s1b-rejected`，episode `d1-s0d1-n3251-seed10xx-shard`） | `splits.quarantine` | 研究端离线复算 |
| **冲突：种子 1000–1011 双登记** | 同一种子出现**两条不同祖先**：`model_selection|v420-dev-panel|dev-seed-10xx`（dev 面板）与 `teacher_query|d1-s1b-rejected|…seed10xx-shard`（隔离）。id 不冲突，但按种子选开发根时必须点名用哪条；隔离那 12 条禁止进入任何拟合路径 | `splits.dev.ancestor_ids` × `splits.quarantine.ancestor_ids` | 研究端离线复算 |
| 永久禁训 | **60000–60199**；保护段 61000–61099 / 61100–61199 / 61400–61499 未用、61200–61299 留给 v430 adapt、61300–61399 spent | `splits.reserved_training_exclusion.seed_min/seed_max`、`protected_ranges` | 研究端离线复算 |
| 封存结果 | `sealed.present_on_host=false`、`sealed.read_sealed_results=false`（不读、不引用封存路径） | `splits.sealed` | 研究端离线复算 |
| 登记完整性 | 4 个 registry 的 LF-normalized sha256 在本检出 **4/4** 与 `splits.json` 记录相符（task_families / pretraining_sources / holdout_families / runs） | `splits.registries[*].sha256` + `digest_scope="text/lf-normalized"`（本轮重算） | 研究端离线复算 |
| 泄漏边界 | 20 个 dev 面板种子在 `used_for_training` 来源的祖先里出现 **0 次**（两种写法都查过） | v8.30 回执 §5 行「曝光/同源」（第 121 行） | 历史记录 |
| **对 A 的直接后果** | **dev 20 条与 quarantine 12 条全部是** `th08-stage1-normal-team0`；train 328 条中**同族 210 条**（其余 118 条分布 stage2–stage6b）。⇒ 若 A 退回"整段 stage 1"，评测族**必然**已出现在训练集里，"未见祖先 / 未见族"这一条**在 A 内无法满足**，只能留给后续版本 | `configs/v430/splits.json`（dev/quarantine 全 stage1；train 的 stage1 210 条）；v8.31 `README.md:21`（后续版本才要求未见祖先） | 研究端离线复算（**主代理已复核**） |

## 5 缺项与最小补证动作

每条四列：**缺什么 / 为什么缺 / 补它最少要做什么 / 谁授权**。前三条（A1、A2、A6）可由**本单 B 节已授权的两次引擎启动**顺带闭合，不必新开资产工单。

| # | 缺什么 | 为什么缺 | 补它最少要做什么 | 谁授权 |
| --- | --- | --- | --- | --- |
| **A1** | "首受控帧"的编号基与对齐：`frames`(13,195) / `STATE.frame`(13,195) / `gameplay_frame`(13,196) / 磁带末帧(13,190–13,191) 四种计数不同基，完成点落在磁带末尾之后 **3 或 4 帧**（差 1 取决于基） | 登记产物只给计数，不给逐帧索引；数据集 `decision_frame` 与评测 `frames` 的基未在任何文档里钉死 | B 教师校准的**同一条记录**里并列给出 4 个数：`frames`、`STATE.frame`、`gameplay_frame`、`EVENTS.stageClearEdgeFrame`（一次运行、四个字段） | 训练端执行 B（本单已授权）；研究端据此冻结口径 |
| **A2** | 双完成信号的一致性：无任何记录同时含 `STATE.stage` 前进帧与 `EVENTS` 0x0080 边沿 | 已登记成功走 (a) 口径，`EVENTS` 口径只在 `v400_stage_clear_replay.py` 里实现过，两条从未并测 | 同 A1 的那一条记录：报告两信号的**顺序与间隔**、当时动作与资源（工单第 42 行 §帧与判分验收 3 已写） | 同上 |
| **A3** | "截断（cap）"的真实见证：160 局最深 3,914、positive 13,195、negative 15,564，**没有一局触到 19,887** | 现有语料里没有任何 episode 跑到上限 | 二选一：(a) 研究端接受"截断只按定义、无见证"并把这一条写进卡；(b) 在 B 的 600 帧冒烟里把 `max_frames` 设成小于实际长度以取得 1 条 cap 记录（会占用 B 的进程预算） | 研究端定口径；(b) 还需训练端进程预算确认 |
| **A4** | **"短"与"完成可观测"不可同时成立**（600 vs 13,195，覆盖 4.55%） | 计划里的 600 上限来自接线冒烟，不是完成类任务 | 二选一：(a) A 采用**整段口径**，每局上限冻到已登记的 19,887（≈1.5×13,195）；(b) 先取得"一决策一 mask"干净磁带，再按"前缀重放 → PARK"切后期子窗 | 研究端决定路线；选 (b) 需**资产重录授权**（新工单） |
| **A5** | 中局合法前缀 + 接管哈希 | 引擎无"从第 N 帧开始"入口；现有磁带 prefix 从帧 92 起双 mask，R74 已排除"引擎存档 / <92 帧前缀 / 替换 prefix"三条 | 重录一决策一 mask 的干净磁带（R74 唯一未排除项），或放弃"短"、只用自然开局 | 研究端 + 资产操作授权 |
| **A6** | 帧 0 的实际姿态与资源（x, y, power, lives, bombs, graze） | 登记产物只有源码级初值与终局资源，无帧 0 的 `STATE` 读数 | B 的 pixel-smoke **首帧** `STATE` 各字段（1 条记录、6 个数），并注明是否被 obs-v3 初始填充影响 | 训练端执行 B（本单已授权） |
| **A7** | R74 首个冲突的**具体掩码**与 **pair prefix 源文件身份** | R74 只登记帧号（92 / 8,664）与结论，不登记 `0x….`-`0x….` 两值，也不点名磁带文件 | 该 prefix 的**2 行原始 mask**（或记录的 `CounterfactualError` 原文）+ 文件 sha256；若是另一文件，同时给路径 | 训练端定点只读（需训练端配合，非本卡可自取） |
| **A8** | "未参与拟合或选参"的开发根 | dev 面板 1000–1019 已用于选参；`1020–1039` 只在 `reserved_seeds` 里，未登记为 ancestors | 登记 1020–1039 为预注册开发根（或另立新种子段），重算 `splits.json` 的登记摘要，并在看结果前写死 | 研究端（split 变更 + 重新冻结 `configs/v430/splits.json`） |

## 6 证据索引

| 结论 | 源文件 | 行号 / 键 | 来源类型 |
| --- | --- | --- | --- |
| 阶段身份 `th08-stage1-normal-team0`，difficulty 由语料固定 | `configs/v430/splits.json` | `ancestors[0].task_family` / `.note`；`roots.engine_binary_relative` | 研究端离线复算 |
| `team` = `shotType`（机体），不能用来切子窗 | `Document/Update/Alpha/v8.30.0/TRAINING_HOST_REPLY_01.md` | 第 48、49 行（§1） | 历史记录 |
| 无"从第 N 帧开始"的入口 | 同上 | 第 50 行（§1）、第 210 行（§9） | 历史记录 |
| `stage_clear` 两套定义、从未并测 | 同上 | 第 37–39 行（§1）、第 215 行（§9 第 6 项） | 研究端离线复算（字段缺失）+ 历史记录 |
| `EVENTS` 0x0080 位与 `stageClearEdgeFrame` 的实现存在 | `tools/rl/v400_stage_clear_replay.py`；`tools/rl/v400_p2_reward_detector.py` | `:579-586,628`；`:2486` | 研究端离线复算 |
| 成功见证：13,195 帧 / 6,598 决策 / stage 0→1 / 2,631,430 分 / `mask_trace_sha256=66c81344…` | `.tmp_v420_evalpulse/positive_control.json`（sha256 `9cdd9033…`） | `frames` / `decisions` / `stage_start` / `stage_end` / `stage_clear` / `stop_reason` / `exit` / `score` / `shoot.*` / `tape.*` | 研究端离线复算 |
| 成功见证的登记副本与说明 | `Document/Update/Alpha/v4.2.0/evidence/T08/eval_shoot_pulse.json`；`…/T08/EVAL_SHOOT_PULSE.md` | `positive_control.*`；`:164-190` | 历史记录 |
| 失败/不同后果见证：15,564 帧 game_over、13,195 帧处仅 141,379 分 | `.tmp_v420_evalpulse/negative_control.json`（sha256 `ecb58cb4…`）；`T08/clear_reachability.json` | 同名字段；`q2_experiment.runs[eval]` | 研究端离线复算 + 历史记录 |
| 分数平台 8,187 帧（换面等待对时间不敏感） | `T08/clear_reachability.json` | `q2_experiment.runs[eval].longest_constant_score` | 历史记录 |
| 策略分母 0/160，最深 3,914 帧 | `T08/CLEAR_REMEASURE.md` | `:3-8`、`:51-64` | 历史记录 |
| 射击脉冲规则（period 48 / 最后一帧 / 同丢 BOMB） | `touhou_rl/v400/teacher/shoot.py`；`T08/EVAL_SHOOT_PULSE.md` | `:50-55,58-80`；`:60-81` | 研究端离线复算 + 历史记录 |
| 磁带 SHA、槽数、首/末决策帧 | `.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed12-shard/slots.jsonl`；`meta.json` | `Get-FileHash`；`counts.slots=6588` / `counts.first_decision_frame=16` / `counts.last_decision_frame=13190` | 研究端离线复算 |
| 磁带完整性锚（可当恢复锚） | 同上 `meta.json` | `hash.corpus_hash_post_equal=3298`、`hash.corpus_frame_hashes_observed=13192`、`hash.dataset_hash_chain_breaks=0`、`hash.label_mismatches=0` | 研究端离线复算 |
| 磁带首槽自带 8 行 one-hot(7) 历史（非零行 pad） | `slots.jsonl` | 第 1 行，`history` / `history_frames` / `obs_render_frames` | 研究端离线复算 |
| 一个 2 帧决策内 mask 必须相同 | `touhou_rl/v420/counterfactual.py` | `:3924-3940`、`:3988-3989` | 研究端离线复算 |
| 双 mask 首个冲突帧 92、目标段 8,664、四条解阻已排除三条 | `Document/Update/Alpha/v4.3.1/evidence/R74/pair_cand_scope.json` | `:30-40`、`:42-58`、`:63` | 历史记录 |
| obs-v3 常量（栈 4 / play 112×96 / HUD 56×160 / 历史 8 / 零行 pad） | `touhou_rl/v400/student/obs_spec_v3.py`（import 运行） | `FRAME_STACK` / `PLAY_SHAPE` / `HUD_SHAPE` / `HUD_FRAMES` / `ACTION_HISTORY_*` | 研究端离线复算 |
| 600 帧上限 | `Document/Update/Alpha/v8.30.0/PIXEL_SLICE_PLAN.md` | `:25`、`:88` | 历史记录（研究端口径） |
| 19,887 帧上限 | `T08/CLEAR_REMEASURE.md`；`positive_control.json` | `:23,211`；`max_frames` | 历史记录 + 研究端离线复算 |
| 训练/开发/确认/适应/隔离/禁训/封存 全部分类 | `configs/v430/splits.json` | `splits.{train,dev,confirm,adapt,quarantine}`、`reserved_training_exclusion`、`confirm_batches`、`sealed` | 研究端离线复算 |
| 4 个 registry 摘要 4/4 相符 | 同上 + `Document/Update/Alpha/v4.3.0/registries/*.json` | `registries[*].sha256`（LF 归一后重算） | 研究端离线复算 |
| 训练根 328 / dev 20 / confirm 0 / adapt 0 / 隔离 12 | 同上 | `splits.*.ancestor_ids` | 研究端离线复算 |
| 曝光分类（17 used_for_training 等）与 dev 0 泄漏 | `v4.3.0/registries/pretraining_sources.json`；v8.30 回执 §5 | 第 121 行 | 历史记录 |

## 7 本卡明确不主张什么

1. **不主张存在短阶段**：全卡没有任何"短子阶段"的事件定义；最早可辨识的完成点仍是整段 stage 1 的 `stage_clear`。
2. **不主张 `stageClearEdgeFrame` 已知**：没有任何登记记录给出该字段的数值；本卡只有它的**实现**存在性（`v400_stage_clear_replay.py`）与机理出处。
3. **不主张两套完成口径一致**：从未并测（A2）。
4. **不主张"教师录制到过关帧"**：完成点的最后 3–4 帧来自 `--tail 64` 的重复动作，不在教师录制帧内（§1.2）。
5. **不主张 SHOOT 脉冲属于任务定义**：v8.30 回执 §13.4（第 381–384 行）给了 (a) 声明为任务定义 / (b) 扩展动作空间两条路，**本条未裁定**；本卡只是引用 (a) 的已实现规则与冻结周期 48（= 48 决策 × 2 帧 = 96 帧）。
6. **不主张 R74 的具体掩码**：只有帧号，没有 `0x….`-`0x….`（A7）。
7. **不主张 600 帧窗口能观测完成**：算术上不可能（4.55%）。
8. **不主张任何像素能力或像素可学**：本卡只冻结任务与信号口径；`EVAL_SHOOT_PULSE.md:190,277-279` 的边界（教师重放 ≠ 策略）同样适用于本卡。
9. **不主张 splits 的一致性能外推**：registry 摘要相符只证明**工作副本未漂移**，不证明历史实跑用了这些文件（同 v8.30 回执 §7 对 SOURCE_AUDIT 的边界）。
10. **本卡不含 `训练端本次实测` / `训练端本次离线检查`**：这两类结论只能由 B 节回执产生；A 的 A1/A2/A6 三条缺项依赖 B 的记录，属"随 B 一起回报"。
11. 不主张 A 节已可实施：可冻结的只是**不可变项**（阶段身份、起点类型、完成口径、合法动作、两类见证、祖先分类），**可变的"短"仍是空的**。
