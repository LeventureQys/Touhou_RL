# V830-ROUTE-INFO-01 回执：路线判别所需的最少信息

**工单**：`Document/Update/Alpha/v8.30.0/TRAINING_HOST_REQUEST.md`（V830-ROUTE-INFO-01）
**回执端**：DeepSeek 训练端主代理 + 执行代理（DeepSeek v4.1 flash）
**性质**：只读。本单未启动引擎、未评测、未采集、未训练、未安装依赖、未修改代码/协议/资产/暂停快照、未上传。WP05 保持暂停。

> **执行说明（需研究端知情）**：本单先按项目约定派给 4 个 Provider=DeepSeek / DeepSeek v4.1 flash 子代理，四者在长时间静默后无任何回执，按 AGENTS.md §7.3 全部停止（未再唤醒，避免占用并发槽）。为不让路线停滞，第 3、4 项及引擎面核查由训练端主代理做最小必要只读处置，命令与退出码列于 §8。这是"训练端子代理池未可用"的处置，不是并行盘点。

---

## 0. 身份

| 项 | 值 | 类别 |
| --- | --- | --- |
| 实际跑工具的检出 | `/mnt/d/WorkShop/Touhou_RL`（= Windows `D:\WorkShop\Touhou_RL`） | 本次只读核实 |
| 该检出 HEAD | `c885e6477f498741d12269942d5e3decf8c8e2ed`（v8.30.0 文档提交） | 本次只读核实 |
| `git status --short -uno` 行数 | 3,516（工作区存在大量既有未提交改动/删除，非本单产生） | 本次只读核实 |
| Python | `C:\...\Python312\python.exe 3.12.10`；WSL 侧 `/root/v400-venv/bin/python` **3.10.12**（引擎与工具用它） | 本次只读核实 |
| 时间及时区 | 2026-11-25T00:19:50+08:00（Asia/Shanghai） | 本次只读核实 |
| 第二个检出 | WSL `/root/Touhou_RL`，HEAD `d55c26a6c465b26f11e26ba4fc1bdb888e8b130b`（v4.0.0 期），**但引擎产物在此树下** | 本次只读核实 |
| 引擎（登记的 rl_v3） | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，sha256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` | 本次只读核实；与 `ASSET_SYNC` 登记 `691bfd77…` **一致** |
| rl_laser | sha256 `1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f` | 同上，与登记 `1462829b…` **一致** |
| 游戏数据 | `/root/v400-data/game` 存在 | 本次只读核实 |
| 运行中引擎进程 | Windows 0 个、WSL 0 个（本单全程无引擎） | 本次只读核实 |

**⚠ 身份分裂（须研究端知悉，构建身份本回合已解开）**：代码检出在 `c885e64`，而 `rl_v3` 等引擎二进制位于第二个检出 `/root/Touhou_RL`（其 git HEAD 停在 `d55c26a`）。二进制的 sha256 与 `ASSET_SYNC` 一致。**构建身份现已可核**：`third_party/th08-rl` 自身是 git 仓库且**工作树干净**，HEAD = `1523df26ec643b7276d2cf487b299ad932a982b7`；`build/rl_v3/CMakeCache.txt` 记 `th08-modern_SOURCE_DIR=/root/Touhou_RL/third_party/th08-rl`、`TH08_RL_HARNESS=ON`、`CMAKE_BUILD_TYPE=Debug`；`rlharness.cpp.o` 的 mtime 与源码 mtime 同为 2026-11-10 10:17:29（差 0.6 s），二进制 10:43:28 链接完成。**残留限制**：没有构建日志，所以"编译的就是这份内容"依赖 mtime + 干净工作树，不是可回放的哈希链——这一条仍按待核验的强度陈述。

---

## 1. 一个真实攻击阶段

| 字段 | 结论 | 证据 | 类别 |
| --- | --- | --- | --- |
| 任务族 | `th08-stage1-normal-team0`（stage 0 / normal / **shotType 0**；见下 `team` 更正） | `configs/v430/splits.json` → `ancestors[0].task_family`；`splits.json:roots.engine_binary_relative` | 本次只读核实 |
| 恢复入口 | `RESET <stage> <difficulty> <team> <seed> [practice]`；`ArmStart()` 在标题画面 Ready 状态被自身消费，**不能从外部强改 curState** | `rlharness.cpp:981,989-991`；`rlharness.hpp:83-91` | 本次只读核实 |
| 起止事件 | 起点 = `RESET` 后 `ResetFrameCount()` 的 frame 0（= 录像 trace 的第 0 字段）；**完成事件 = `stage_clear`** | `rlharness.hpp:68-73`；见下"完成定义" | 本次只读核实 |
| 完成定义（可核） | `EVENTS` 回包字段 `stageClear` / `stageClearEdge` / **`stageClearEdgeFrame`** / `stageClearScreenState` / `stageClearSeq`；`frameEventFlags` 的 **0x0080** 是 harness 派生位，取自 `Gui.impl->stageClearScreenState`，在 pump 循环里**逐帧**采样（对多帧 STEP 也精确） | `rlharness.cpp:82-124,755-794`（Handler 全文已读） | 本次只读核实 |
| **⚠ 代码库里有两套 `stage_clear` 定义（R1 必须冻结其一）** | (a) **评测工具口径**：`stage_clear = stage_end > stage_start`，即 `STATE` 的 stage 计数前进了。`v420_eval_shoot.py:667`、`v410_student_eval.py:1065`、`v410_baselines.py:361`、`v400_teacher.py:633`、`touhou_rl/v420/eval.py` 全用这一套，**无一读取 `EVENTS` 位**。(b) **harness 事件口径**：`EVENTS` 的 0x0080 + `stageClearEdgeFrame`，只有 `v400_p2_reward_detector.py`（`STAGE_CLEAR_BIT=0x0080`）与 `v400_stage_clear_replay.py` 用 | 上述行号（本次逐文件核对） | 本次只读核实 |
| 两套口径的差别 | (a) 只给"阶段变了"，**不给精确帧**；(b) 给**逐帧电平 + 精确边沿帧**。两者"应当"一致但**本单未验证一致**。若 R1 主指标要帧级精度，应冻结 (b) 并把 (a) 作为交叉校验 | — | 缺口/待核验 |
| 该表面是否在**已构建**二进制里 | **是**。`rl_v3` 含字符串 `OK frames=%llu frameEventFlags=%04x stageClear=%u stageClearEdge=%u stageClearEdgeFrame=%llu stageClearScreenState=%d stageClearSeq=%u` | `strings rl_v3/th08-modern` | 本次只读核实 |
| EVENTS 在正常对局是否可用 | **可用**。`g_ReplayManager==NULL` 时 fail-closed 回 `ERR events-unavailable`；但清位回调在 **record 模式**也注册，正常开局的阶段会累积 | `rlharness.cpp:757-777` | 本次只读核实 |
| 动作库（**三套，别混用**） | ① 原始按钮：`MultiDiscrete(9,2,2,2)`=**72**（`encode_flat = move*8+shoot*4+focus*2+bomb`）；② **教师候选**：`cand = focus*9+move`=**18**，**射击恒开、bomb 恒关**；③ **学生动作**：`(cand, bomb)`=**36**。9 方向 = 0 idle/1 N/2 NE/3 E/4 SE/5 S/6 SW/7 W/8 NW | `touhou_rl/spec.py:343-345,384-407`；`touhou_rl/v400/gym/actions.py:5-7,19-22,102,105`；**缺口见 §13** | 本次只读核实 |
| 决策节拍 | 依工具而定：v4.2 T08 评测 `DECISION_FRAMES=2` / `COMMIT_FRAMES=2`；v8 系列用 H=120 微段。**本单不裁定，须在实施单冻结** | `R74/code_facts.constants_runtime` | 历史记录 |
| 射击约定（关键） | 过关**需要 SHOOT 上升沿**：每 48 决策，在该决策最后一个 STEP 帧 `mask & ~(BTN_SHOOT\|BTN_BOMB)`。旧仪器无条件 OR `BTN_SHOOT`、无释放路径，引擎永远看不到上升沿 | `v4.2.0/evidence/T08/CLEAR_REMEASURE.md:22-32` | 历史记录 ；**机理与后果见 §13** |
| 初始资源（**源码已核，替换先验**） | Normal / 非 practice / 非 replay 的普通开局：`livesRemaining = cfg->lifeCount = 2`（**2 条备用残机**）；死亡时先判 `GetLives() <= 0` → 是则 `showRetryMenu=1`（game over），否则 `AddLives(-1)` 并重生 ⇒ **共允许 3 次死亡**（= lifeCount+1 次机会）。`bombsRemaining = primaryShtFile->initialBombCount` —— **炸弹数来自机体射击文件，不是 cfg 的 bombCount=3**。`playerPower = 0` | `GameManagerSetup.cpp:105-130`；`GameManager.hpp:286-298`；`Player.cpp:1393-1400`；`Player.hpp:30,47` | 本次只读核实 |
| 资源语义的三个陷阱 | ① **practice 模式把 lifeCount 强制成 8**（`flags.isPracticeMode`），比自然开局多 6 条命，绝不能当作自然起点；② `difficulty >= EXTRA` 时 lifeCount 被强制为 2；③ 资源初始化整块在 `if (!flags.isReplay)` 里，**replay 路径不走这段**，所以录像回放不能用来推断自然开局资源 | `GameManagerSetup.cpp:110-130` | 本次只读核实 |
| 死亡的资源后果 | 死亡会 `SetPower(0)` 并掉出 5 个 `ITEM_POWER_FULL`，同时 `DecreaseSubrank(1600)` —— 即"首死"不只是少一条命，还改写 power 与 subrank | `Player.cpp:1355-1364` | 本次只读核实 |
| `team` 参数（**更正**） | `team` **不是**阶段内位置选择器，而是**机体/射击类型**：`ConsumePendingStart()` 里 `g_GameManager.shotType = g_startTeam;`。所以任务族名 `…-team0` = stage 0 / normal / **shotType 0**。`team>0` = 换机体，与 R1"冻结机体"冲突，**不能**用来切更短的子窗 | `rlharness.cpp:1142-1146,1163-1165` | 本次只读核实 |
| 机体（**本回合已定**） | `team` = `shotType`，零售录制元数据给出完整对照：**0 Border Team**、1 Magic Team、2 Scarlet Team、3 Ghost Team、4 Reimu、5 Yukari、6 Marisa、7 Alice、8 Sakuya、9 Remilia、10 Youmu、11 Yuyuko。所以任务族 `…-team0` = **Border Team**（灵梦+紫），全部既有运行都用它。R1 冻结机体 = 冻结 shotType，并因此**同时冻结了炸弹数**（见"初始资源"行） | `third_party/th08/config/replay-fixtures.json`（108 条 fixture 的 `shot`/`shotName` 对照）；`Player.cpp:1387` 用 `shotType` 选动画 | 本次只读核实 |
| 是否存在"从第 N 帧开始"的引擎杠杆 | **没有**。RESET 只有 `stage/difficulty/team(shotType)/seed/practice`，无起始帧参数；中局状态只能靠**前缀磁带重放 + `BRANCH`/`PARK`/`SPAWN`** 到达（v8.29 已用 `--reuse-park`） | `rlharness.cpp:981,1004,1142-1165`；`rlharness.hpp:43-51` | 本次只读核实 |

**结论**：一个真实、可恢复、完成事件可在帧级核对的阶段**成立**（`th08-stage1-normal-team0`，完成 = `stage_clear`）。但"短攻击阶段"这一具体切片**尚未定义**——现有全部证据都是整段 stage 1（需 13,195 帧），没有"短"子阶段的事件定义。

**而且引擎层没有"从第 N 帧开始"的入口**：`RESET` 只接受 `stage/difficulty/team(=shotType)/seed/practice`，`team` 换的是机体而不是段位。因此中局起点**只能**由「合法前缀磁带重放 → `PARK`/`BRANCH` 固化状态」得到，而这正是 §3 里被双 mask 磁带卡住的那条路。**结论：R1 若要用比整段更短的切片，前置条件是拿到一决策一 mask 的干净磁带（需授权重录）；否则任务就是整段 13,195 帧。**

---

## 2. 现存成功与失败见证

| 项 | 值 | 证据 | 类别 |
| --- | --- | --- | --- |
| **真实成功**（唯一一条，本次已从产物 JSON 逐字段核对） | 同一条**教师录像带**在修正仪器下：`stage_clear=true`、`frames=13195`、`score_final=2631430`、`deaths=0.0`、`stop_reason="stage_clear"`、`decisions=6598`、`shoot_frames=13058`。负对照（长 period）→ `stage_clear=false`、`frames=15564`、`score_final=163945`、`deaths=3.0`、`stop_reason="game_over"`。`shoot_pulse_period=48`（= 48 决策 × 2 帧 = 96 帧，与 `configs/v420/rl-engine-{audit,pilot}.json` 冻结值一致） | `T08/clear_reachability.json`、`T08/eval_shoot_pulse.json`（本次原样读取字段）；叙述见 `T08/CLEAR_REACHABILITY.md`、`T08/EVAL_SHOOT_PULSE.md` | 历史记录（产物本次已核） |
| **独立于教师磁带的可解性见证**（零售录像） | 零售录像库里 **Normal + Border Team + 非 practice + 非 extra** 的 fixture 有 **2 条**；其中 `normal-0-b`（`th8_ud2cd5.rpy`，sha256 `ba9ca721a8fe80d015c2b99461804f12891d3fbad842ed5b12829887177d49ba`）的 `stages[0]` 记 Stage 1 的 `inputRecords=9203`、`endScore=1802425`，且其后仍有 Stage 2/3… ⇒ **该局确实通过了 Stage 1**。这是本项目之外的人类录像，与我们的动作约束不同，**只能支持"任务可解"**，不能充当同约束成功见证 | `third_party/th08/config/replay-fixtures.json`（108 条，全部 `practice=false`、`extended=false`） | 本次只读核实 |
| 该成功的动作约束 | 动作库同 18/72 登记入口；**SHOOT 脉冲 period=48**；stage 0 / normal | `CLEAR_REMEASURE.md:22-23` | 历史记录 |
| 该成功的性质 | **教师录像带重放**，不是像素策略成绩，也不是新阶段成绩 | 同上 | 历史记录 |
| **真实失败** | 4 臂 × 2 口径 × 20 dev seed = **160 局，`stage_clear` = 0**，errors=0、invalid=0；最深 3,914 帧（D3/seed1010/s1n），距过关 13,195 帧差 **9,281 帧**；160 局无一达 12,000 帧 | `T08/CLEAR_REMEASURE.md:3-8,49-64` | 历史记录 |
| 失败的三类分离 | 阶段完成 0；首死 = `death_stop:20`（diag 口径，帧 1,597.9–1,757.5 均值）；技术故障 0 | 同上 §3 | 历史记录 |
| 同约束程度 | 成功与失败**同一引擎 sha `691bfd77…`、同一 `--max-frames 19887`、同一 SHOOT 脉冲仪器**；差别在"教师录像带 vs 策略 checkpoint" | `CLEAR_REMEASURE.md:16-25` | 历史记录 |
| 旧仪器结果不可用 | 旧 `v410_student_eval.py` 的 `cand_to_mask` 无条件 OR `BTN_SHOOT` → **此前所有 0/N 都不是"能否过关"的测量** | `CLEAR_REMEASURE.md:27-32` | 历史记录 |

**结论**：**有同约束真实成功见证，也有同约束真实失败见证**，且二者身份一致。可以支持 R1 的"任务可解性"判断。需要提醒的两点：(a) 成功见证是教师录像带重放，不是策略；(b) 成功需要的 13,195 帧是**整段 stage 1**，与"短攻击阶段"的设想还有距离。

---

## 3. 像素与执行历史

| 字段 | 结论 | 证据 | 类别 |
| --- | --- | --- | --- |
| 合法前缀 | **不成立（已定位到首个冲突）**。旧配对磁带 prefix **从 frame 92 起就有双 mask 决策**，而 `_pair_cand` 的**唯一调用点在 `replay()`**（校验录制磁带，不是实时策略），第 92 帧即抛错；目标段落在 **frame 8,664** | `R74/pair_cand_scope.json:19,30-40` | 历史记录 |
| 实时臂是否也有该冲突 | **没有**。实时臂每 commit 块只提交一个 mask（`COMMIT_FRAMES = 2×1 = 2`），历史**按构造**满足"一决策一 mask" | `R74/pair_cand_scope.json:34,38` | 历史记录 |
| 四条解阻选项（已排除三条） | 引擎存档：**无该动词**；重录干净磁带：需资产授权；改用 <92 帧前缀：早期稀疏局面，C4 已判 0/4 不合格；替换 prefix 强行重放：**可到达但无效**（起始状态不是录制状态，而"0 不符"是三臂可比前提） | `R74/pair_cand_scope.json:42-58` | 历史记录 |
| 连续像素 | 有：obs-v3 = playfield + HUD + 动作历史，T08 为纯像素评测（引擎 STATE 仅作裁判） | `CLEAR_REMEASURE.md:11-12,194` | 历史记录 |
| 观测维度（v8 放雷头，**本次已解出全部构成**） | `OBS_DIM=106` = `10 + 6×12 + 5×4 + 4`（`v431_arm_a.feature_dim()`）。逐项：**10 基础** px/384、py/448、(power−64)/64、lives/8、bombs/8、graze/2000、bullet_count/300、laser_count/10、enemy_count/20、decision_index/300；**12 颗最近致命弹 ×6** 相对 dx、dy、vx/10、vy/10、half_width/16、逼近速度；**4 个最近可碰撞敌 ×5** 相对 dx、dy、half_w/384、half_h/448、life/5000；**激光 ×4** 相对 dx、dy、width/64、damageable。再拼 3 个 explicit = 109 | `tools/rl/v431_arm_a.py:65-68,88-152`；`tools/rl/v8u0_features.py:166-169,586-611` | 本次只读核实 |
| 像素契约 | `PIXELS` 回 `glReadPixels(0,0,640,480,GL_RGBA,GL_UNSIGNED_BYTE)` → **640×480 RGBA8**，**原点在左下**；导出 PPM 时翻成自上而下并丢 alpha；`checksum` 是 RGBA 的 FNV‑1a‑64；同帧字节也发布到共享块（`SHM`） | `rlharness.cpp:1182-1215` | 本次只读核实 |
| `touhou_rl/obs.py::encode` 的性质（**关键**） | 它**不是像素观测，是状态/特权观测**：输入是 `GameState` + `BulletField`（引擎真值）。`self_feat` **24 维** = rel_x、rel_y、两个保留 0、focus、`lives/8`、`bombs/8`、`power/128`、graze、score、clock_time、time_orbs、`stage/6`、`stage_clear_flag`、弹数/700、近距数/120 + 8 条射线；另加最近 `64×6` 弹丸特征与 `16×8` 危险网格 | `touhou_rl/obs.py:33-39,91-186` | 本次只读核实 |
| 泄漏点定位（**修正后的准确说法**） | **两个向量必须分开看**：① v8 的 106 维（`v431_arm_a`）**全是引擎真值**（自机坐标、power、lives、bombs、graze、弹丸/敌/激光的精确位形），**但不含 `stage_clear_flag`** —— 它是特权观测，按 `PIXEL_SLICE_PLAN §3` **不能给像素 actor 用**，但**不泄漏 R1 主指标**；② 泄漏主指标的是**另一条管线** `touhou_rl/obs.py::encode`，其 `self_feat[13]` = `float(state.stage_clear_flag != 0)`，而该向量在 72,472 帧录像上恒为 1.0 | `tools/rl/v431_arm_a.py`；`touhou_rl/obs.py:151`；`touhou_rl/vision_metrics.py:15-17` | 本次只读核实 |
| 可见性已知/未知 | 已知：SHOOT 上升沿是过关必要条件（仪器级证据）。未知：过关瞬间的画面先兆、自机/威胁/资源/阶段先兆在框内的**可见性覆盖率**未测 | — | 缺口/待核验 |

**结论**：像素通道存在，但**当前没有任何满足"一决策一 mask"的合法前缀磁带**，这是输入 3 的硬缺口；且 **index 13 的 `stage_clear_flag` 与 R1 主指标（阶段完成）直接同源**，任何像素候选继承旧 obs 前必须删掉该通道，否则主指标本身泄漏进观测。本单未启动 render、未做可见性探针（像素侧的正式契约见 §12：obs-v3 已冻结且边界合规）。

---

## 4. 同约束控制器与像素候选

| 项 | 结论 | 证据 | 类别 |
| --- | --- | --- | --- |
| 可部署强参照（固定动作） | 18 个恒动作里在 4 个 train 窗口选出 **#12**，4 个 dev 窗口均值 **172.4**（per-window 258.0/180.5/221.0/30.0） | `R51/baseline_ladder.json:31-64` | 历史记录 |
| 神谕 vs 可部署 | 每窗最优神谕 258.8；池均值 111.7；神谕膨胀 +86.4（×1.5） | 同上 R51:19-28 | 历史记录 |
| 旧学习臂 A | 均值 **94.1**，**低于可部署恒动作 78.3，也低于池均值 17.6** | 同上 R51:82-86 | 历史记录 |
| 教师/状态控制器 | 可解性参照：教师录像带（真实过关，见 §2）；教师另可用真值。**教师 100/100 类的整段动作成绩不能充当缩减任务的正对照** | `PIXEL_SLICE_PLAN.md:62`（研究端口径）+ `CLEAR_REMEASURE.md` | 历史记录/研究端 |
| 特权混合臂标注 | 凡依赖特权距离/库存的放雷规则，只能标为**特权/混合参照**；G1/T2 移动读出为 softmax 抽样、v8.28 学习头为 argmax，成绩同时含移动读出差异 | `SOURCE_AUDIT.md:57`；研究端口径 | 历史记录 |
| 像素候选现状 | **无**可用于本阶段的像素学习权重（v8.29 演练用的 `bombdag.pt` 是放雷头，非本任务像素 actor）。可用**接线入口**：`PIXELS`/`SHM`/`READBACK` 命令 + obs-v3 管线 + `rlharness` 的 `SetReadback`/`ReadbackMicros` | `rlharness.hpp:96-105`；`rlharness.cpp` PIXELS 分支 | 本次只读核实 |
| 汇报口径风险 | 旧 `S2` 右半（臂 A ≤ C-const-best）对任何非神谕方法都会触发 → **不可证伪的停止条件**；真正卡住像素臂的是 `_pair_cand` 与 `PARK`，不是该触发 | `R51/baseline_ladder.json:75-79` | 历史记录 |

**结论**：**可解性参照齐备**（教师录像带 + 已构建引擎 + EVENTS 判决）；**可部署强参照齐备**（恒动作 #12 = 172.4，可直接用作 R2 的"固定/开环"对照底本）。**缺像素候选权重**，只能先做接线。另须提醒：该任务上旧学习臂 94.1 已被一个恒动作（172.4）明显解释，这正是 `ROUTE_DECISION.md` 第 4 条要防的情形，R2 必须把它作为强参照而非弱参照。

---

## 5. 少量开发祖先

| 项 | 值 | 证据 | 类别 |
| --- | --- | --- | --- |
| 祖先登记总数 | **360**（`splits.json.ancestors`） | `configs/v430/splits.json` | 本次只读核实 |
| 可直接分给训练/调试 | `train` = **328** 条（源为 `teacher_query|d1-teacher-fast`，D1 语料分片） | 同上 `splits.train.ancestor_ids` | 本次只读核实 |
| 未参与拟合与选参的开发面板 | `dev` = **20**（seeds **1000–1019**，`model_selection|v420-dev-panel`）；**1020–1039 保留**未登记为祖先 | 同上 `splits.dev` | 本次只读核实 |
| 正式确认 | `confirm` = **0**（v4.3.0 未冻结确认批） | 同上 `splits.confirm` | 本次只读核实 |
| 适应保留 | `adapt` = **0** | 同上 `splits.adapt` | 本次只读核实 |
| 隔离 | `quarantine` = **12**（S1b dev 1000–1011，种子来源被拒） | 同上 `splits.quarantine` | 本次只读核实 |
| 永久禁训种子 | 60000–60199 | 同上 `reserved_training_exclusion` | 本次只读核实 |
| 封存结果 | `sealed.present_on_host=false`、`read_sealed_results=false` | 同上 | 本次只读核实 |
| 曝光/同源（**本回合已解，原为缺口**） | 曝光不记在 `splits.json.ancestors` 上，而是记在冻结登记 `Document/Update/Alpha/v4.3.0/registries/pretraining_sources.json` 的 `status` 上：25 个来源 = **17 used_for_training / 2 used_for_selection / 3 evaluated_only / 1 rejected_not_used / 1 rule_based_no_training_data / 1 executed_instrument_only**；祖先枚举 22 个 `complete`、3 个 `family_level`。**关键核对：20 个 dev 面板种子（1000–1019）在任何 `used_for_training` 来源的祖先里出现 0 次**（`dev-seed-NNNN` 与数字 `seedNNNN` 两种写法都查过） | `v4.3.0/registries/pretraining_sources.json`（本次解析） | 本次只读核实 |
| 登记表完整性 | 4 个冻结登记（task_families / pretraining_sources / holdout_families / runs）按 `digest_scope: text/lf-normalized` 重算 sha256，**4/4 与 `splits.json` 记录的摘要相符**，无漂移 | `configs/v430/splits.json.registries` + 四个登记文件（本次重算） | 本次只读核实 |
| v8.29 新采根 | 8 个演练祖先（`reh-seed3201…`），`used_for_training:false`、`lineage_new:true`，但**前缀是被评测策略自己滚出来的**（起点不中立） | `v8.29.0/RESEARCH_REVIEW_PACKET.md:55-71,112-116` | 历史记录 |

**结论**：支持一个小任务族的**少量开发祖先是有的**（20 个 dev 面板 + 328 训练源），不必预设 8 根。两点限制：(a) 4 个旧开发窗 11/12/17/18 已用于 BC/RL 与开发，**不能充当新的独立留出集**；(b) 曝光已在冻结登记里完整登记、dev 面板 0 泄漏，但 v8.29 自采根的起点依赖被评测策略——**"起点中立"仍待研究端决定**（`RESEARCH_REVIEW_PACKET §4.3`）。

---

## 6. 一个切片的实际成本

| 环节 | 实测 | 证据 | 类别 |
| --- | --- | --- | --- |
| T08 单批 = 5 seed | **115–223 s**（批墙钟），即 ≈ **23–45 s/局**；`--max-frames 19887`；25 批退出码全 0，errors=0/invalid=0 | `CLEAR_REMEASURE.md:217-245` | 历史记录 |
| 单局上限 | `--max-frames 19887`（≈ 13,195 过关帧的 1.5 倍） | 同上 `:211` | 历史记录 |
| 前缀/恢复拆分 | **本任务没有拆分**；可得范围只有两种口径（diag=`--stop-on-death` / s1n=三命） | 同上 §9 | 缺口/待核验 |
| 渲染/读回单独成本 | harness 自带计量：`ReadbackFrames()` / `ReadbackMicros()`（"differencing two STEP runs" 可测读回成本）；**尚未被任何成本登记使用** | `rlharness.hpp:96-105` | 本次只读核实 |
| 采集 1 盘 4,800 帧 | ≈ **4.2 s**；5,600 帧 ≈ 9.6 s | `v8.29.0/RESEARCH_REVIEW_PACKET.md:121-126` | 历史记录 |
| 96 单臂（单档） | 176 s（30 s 档）/ 168 s（40 s 档）；两档 96 对/192 单臂 ≈ **344 s** | 同上 | 历史记录 |
| 切片规划上限 | 每条轨迹至多 **600 受控游戏帧**，前缀重放/记忆预热/恢复核验**另计** | `PIXEL_SLICE_PLAN.md:25,88` | 研究端口径 |
| 可用起止时段与分钟数 | **未知**：本单无调度记录、无可承诺窗口 | — | 缺口 |

**结论**：一个端到端切片的量级已知（单局 23–45 s、≤19,887 帧），但**没有前缀/恢复/渲染的拆分实测**，也**没有可用时段**。按工单要求不外推并发、不承诺完成日期。

---

## 7. 附带核实：SOURCE_AUDIT 在本机的身份

研究端 `SOURCE_AUDIT.md` 明确要求"训练端如文件身份不同，先报告差异"。本单核对结果：**完全一致，无差异**。

| 项 | 研究端声明 | 训练端本次实测 | 结论 |
| --- | --- | --- | --- |
| `v8m0_bombrl.py` | `fd0ac4ef…4cb41`；BC 960、977–979；指标 989–992 | sha256 **逐字符相同**；960 `BCEWithLogitsLoss()`、977 `# two logits == one logit…`、978 `[:, 1]`、989 `softmax` | 一致 |
| `v8p0_bombrl900.py` | `71bfbeb6…122a`；1012、1029–1031；1041–1044 | sha256 相同；命中 1012/1029/1030/1041 | 一致 |
| `v8t0_densereward.py` | `e7f004e5…21e1`；1050、1067–1069；1079–1082 | sha256 相同；命中 1050/1067/1068/1079 | 一致 |
| `v8u0_features.py` | `c4cde38f…0aac`；1190、1207–1209；1219–1222 | sha256 相同；命中 1190/1207/1208/1219 | 一致 |
| 网络形状 | `Linear(109,64)→Tanh→Linear(64,2)` | `FEATURE_DIM=109`(106+3)、`BOMB_HIDDEN=64`、`N_BOMB=2`，`make_bomb_head` 在 563–571 | 一致 |
| 源码常数 | 裁剪 1.0、200 steps、batch 256、lr 1e-2、正例比 0.5 | `GRAD_CLIP=1.0`(198)、`BC_STEPS=200`、`BC_BATCH=256`、`BC_LR=1e-2`、`BC_POS_FRACTION=0.5`(208) | 一致 |

**边界**：这只证明**当前工作区文件的身份**与审计基线相同，**不证明**四次历史实跑执行的就是这些文件（`SOURCE_AUDIT.md:56` 的"历史实现身份未核验"仍然成立）。

---

## 8. 实际命令与退出码

```text
# 1) 工单指定的身份命令（在 /mnt/d/WorkShop/Touhou_RL 执行）
git rev-parse --show-toplevel                        -> 0  D:/WorkShop/Touhou_RL
git rev-parse HEAD                                   -> 0  c885e6477f498741d12269942d5e3decf8c8e2ed
git status --short --untracked-files=no              -> 0  3516 行
python -B -c "…sys.version…datetime…"                -> 0  3.12.10 2026-11-25T00:19:50.737904+08:00

# 2) 第二检出的身份
wsl -e bash -lc 'cd /root/Touhou_RL && git rev-parse HEAD'
                                                     -> 0  d55c26a6c465b26f11e26ba4fc1bdb888e8b130b
wsl -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && git rev-parse HEAD'
                                                     -> 0  c885e6477f498741d12269942d5e3decf8c8e2ed
wsl -e bash -lc '/root/v400-venv/bin/python -c "import sys;print(sys.version.split()[0])"'
                                                     -> 0  3.10.12

# 3) SOURCE_AUDIT 四个 runner 的 sha256（Windows Get-FileHash）
                                                     -> 0  4/4 与研究端一致

# 4) 引擎面只读核查（自写脚本，仅 sha256sum/strings/stat）
bash .tmp_v830_engine_probe.sh                       -> 0  build 产物 + sha 前缀
bash .tmp_v830_provenance_probe.sh                   -> 0  mtime/字符串/Handler 位置
bash .tmp_v830_events_gate.sh                        -> 0  EVENTS 的 replay-manager 闸门
bash .tmp_v830_team_probe.sh                         -> 0  team=shotType 实证、PIXELS 契约
bash .tmp_v830_probe_harness.py (venv python)        -> 0  协议动词 25 个 + StateExport 字段

# 5) 结构摘要（不整读大文件）
python -B -c "json.load(configs/v430/splits.json)…"   -> 0  顶层 16 键、ancestors 360、splits 5 类

# 6) 进程核查（关声纪律的现状）
Windows Get-Process *th08*                           -> 0  0 个
wsl 'ps -eo comm | grep -c th08'                     -> 0  0 个
```

最短错误原文：无。本单没有失败命令。

---

## 9. 可独立推进项 / 仅需补充的最少输入

**可独立推进（不需要新授权）**
1. `stage_clear` 判决链路已可核，可作为 R1 主指标；建议下一步先做**只读**的判决冒烟设计（不启引擎）。
2. 删除 `touhou_rl/obs.py:151` 的 `stage_clear_flag` 通道属**代码变更**，须另单授权；本单只报告不修改。
3. 无前缀拆分的成本可用 harness 自带 `ReadbackMicros()` 计量，不必新造仪表。

**仅需研究端定的最少事项**
1. **任务切片（二选一，已无第三条路）**：(a) 接受整段 stage 1（13,195 帧，≈ 3.7 分钟/局）作为 R1 任务；或 (b) 授权**重录一决策一 mask 的干净磁带**，再用「前缀重放 → `PARK`」切后期短子窗。`team` 不能用于此（它是机体/射击类型）。按研究端规则，切片须先按来源与脚本边界定，**不得看新成绩后筛容易窗**。
2. **合法前缀**：即上面 (b) 的授权。不授权则输入 3 的缺口无法在现有磁带内解开（引擎无存档动词、<92 帧前缀已被 C4 判 0/4、替换 prefix 属"可到达但无效"）。
3. **起点中立**：接受策略自生起点，还是要求教师/固定脚本生成的起点（`RESEARCH_REVIEW_PACKET §4.3`）。
4. **强参照冻结**：R2 的固定/开环对照是否就用恒动作 **#12（172.4）**这一族；它在本任务上曾明显压过旧学习臂（94.1）。
5. **可用时段**：本机可用于实施单的实际起止时间与分钟数（本单无此记录）。
6. **主指标口径**：`stage_clear` 冻结为 (a) stage 计数前进，还是 (b) `EVENTS` 0x0080 + `stageClearEdgeFrame`。建议 (b) 为主、(a) 为交叉校验；一旦冻结就写进实施单，不在看结果后改。同时注意 (a) 的现有实现都在"每决策 2 帧"的旧节拍下，与新切片节拍的关系需重新确认。

**本回执不主张**：路线可行性、像素可学、任务已冻结、成本已按切片拆分。R0 设计复核与接线通过都不改变 G410 任何能力门状态。

---

*训练端本次实测 = §0/§1(引擎面)/§3(obs 泄漏)/§5(splits)/§6(ReadbackMicros 能力)/§7；
历史记录 = §2/§3(前缀)/§4/§6(T08 与 v8.29 数字)；
研究端离线判断 = ROUTE_DECISION / WORK_PLAN / PIXEL_SLICE_PLAN 的口径；
待核验 = 可用时段、框内可见性覆盖率。*

**本回执内的自我更正（留痕）**：初稿曾把 `team` 当作阶段内位置选择器并据此推断"可用 `team>0` 切短子窗"。经 `rlharness.cpp:1164` `g_GameManager.shotType = g_startTeam` 实证，`team` 是机体/射击类型，该推断**已撤回**；相应地把"短切片"的唯一可行路径改为"重录干净磁带 + PARK"。初稿另曾据 `strings | grep -cx STAGE_CLEAR` 得 0 而怀疑二进制不含该表面，随后的完整字符串核查显示它是以 `stageClear=…` 形式存在，该怀疑**已撤回**。第三，初稿曾把 `touhou_rl/obs.py` 的 `encode()` 与 v8 的 106 维 `OBS_DIM` 当成同一向量，实测不是（`encode()` 是 24 self + 64×6 弹丸 + 16×8 网格的特权状态向量）；已改为分列并标明"106 维构成待核验"。
第五，§12 初稿把 obs-v3 的 occupancy F1 写成“未通过的门槛”，不够准确——D-14 已正式把验收口径从 F1 改判为闭环 S1N 过关率，F1 仅作代理指标；现已按裁决原文改写。
第六，§13 初稿写成“旧 0/N 从此有了机理层面的解释”，这是**过度断言**。clear_reachability.json 明写旧 0/N “are still valid negative results”，且“no existing 0/N number is explained by it”（那批策略在 ≤4,964 帧就死，离 13,195 帧差 8,000+ 帧）；已改写为“测量能力缺口”并新增 §13.5 专记边界。

---

## 10. 留给实施单的运行纪律：静音（R2 一开引擎就必须先做）

本节不属于六项设计输入，但它是下一次**真的启动引擎**之前必须落地的前置条件，写进回执以免实施时被漏掉。

| 项 | 现状（本次实测） | 证据 | 类别 |
| --- | --- | --- | --- |
| 机内 `th08.cfg` 静音统计 | **找到 105 个、解码 105 个、错误 0；`muted` = 0，`audible` = 105**（一律 `musicMode=1, playSounds=1, music=39, sfx=36`）。分布：`.tmp_v420_data` 71、`.tmp_v410_wp05_round1` 32、`.tmp_v430_o2` 2 | 本次跑 `tools/rl/v431_audio_cfg_audit.py`（退出码 0） | 本次只读核实 |
| 该审计是否可信/稳定 | 重跑后产出的 `R14/audio_cfg_audit.json` 与 HEAD 里的 blob **完全相同**（`37b3e21d930adbdcd959e005fce7f602ae1cc8d2`），即结论逐字节可复现、且自 2026-11-15 未变化 | `git hash-object` vs `git rev-parse HEAD:…` | 本次只读核实 |
| 根因 | 引擎没有音频开关；`th08.cfg` 是 60 字节二进制结构体，写 `key=value` 会被静默丢弃。`/root/v400-data/game/th08.cfg` **本身**就是可出声的，A1/扩容启动器用 `copytree` 把它复制进各私有 data 目录，且未设 `SDL_AUDIODRIVER` | `AGENTS.md §9.1–9.2` | 研究端/历史口径 |
| 静音是否改变游戏逻辑 | 已实测哈希中性（单状态）：R18 在 `playSounds=0` + `SDL_AUDIODRIVER=dummy` 下重跑 A1 state 0 的 18 个首动作，36,180 次逐帧前缀哈希 **0 不匹配**，18/18 判定与每个 `first_death_frame` 逐位相同。**只覆盖 1 个状态，不能外推** | `AGENTS.md §9.2` | 历史记录 |

**实施单开引擎前必须做的三件事**（缺一不得开跑，也不得用“WSL 没有声卡”充当理由）：

1. 给该次运行的私有 data 目录放静音 cfg：`python tools\rl\v400_make_cfg.py --out "<data-dir>\th08.cfg"`，再用 `--inspect … --json` 确认 `musicMode/playSounds/musicVolume/sfxVolume` 全 0。
2. 启动环境显式设 `SDL_AUDIODRIVER=dummy`（`touhou_rl` 里已有多个 v4.0 脚本这么做，**不要**删掉或绕过）。
3. 启动后、跑实验前跑 `python tools\rl\v431_audio_cfg_audit.py`，目标目录必须落在 `muted`；落在 `audible` 就先关声再继续。

**给研究端的提示**：新开的运行**不得再增加 `audible` 计数**。若实施单要求复用一个既有私有 data 目录，那条目录的 cfg 必须先被替换成静音模板并留下替换前后的解码值——这会改动私有目录内容，属需要授权的资产操作，请在那张单里写明。

### 10.1 源 cfg 的逐字节现状（本次实测）

| 文件 | bytes | sha256 | musicMode | playSounds | musicVolume | sfxVolume | 判定 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `/root/v400-data/game/th08.cfg`（活的源） | 60 | `48bf7621da5879ef5ec1392fcac2cc0a85c313912a24fb33b4cddccf0d28f07a` | 1 | 1 | 39 | 36 | **audible** |
| `th08.cfg.orig`（留底） | 60 | `48bf7621…0d28f07a`（**与活的源逐字节相同**） | 1 | 1 | 39 | 36 | audible |
| `th08.cfg.dirty-20261109-001521`（留底） | 60 | `4664d8d44cb845a9faac07bd9a7973ce24fab7234bf082fa5dc3b743bc2d6574` | 1 | 1 | 100 | 80 | audible（更响） |

三点由此确定：

- **§9.2 的前置条件已满足**：`th08.cfg.orig` 与 `th08.cfg.dirty-*` 都在，且活的源与 `.orig` **逐字节相同**——即源文件自 2026-11-07 起没有漂移，回滚是零成本的。
- **根因确认到字节**：0x1F=1、0x20=1、0x27=39、0x28=36。104 个可出声的派生 cfg 都能回溯到这个文件。
- 三个 cfg 的 `lifeCount = 2`、`bombCount = 3`。**注意两者都不是"游戏内资源"的权威来源**：源码里 `livesRemaining = cfg->lifeCount`（所以 lifeCount=2 确实是 2 条备用残机，game over 发生在 `GetLives() <= 0`），而 `bombsRemaining = primaryShtFile->initialBombCount` —— **炸弹数由机体射击文件决定，`cfg.bombCount=3` 只是菜单默认值**（仅在 `Supervisor.cpp:1394-1395` 与 `TitleScreen.cpp:1171` 作为默认写回）。详见 §1 的"初始资源"行。

### 10.2 两种修法，以及为什么应当选最小改法

| 修法 | 做法 | 结果 sha256 | 副作用 | 适用 |
| --- | --- | --- | --- | --- |
| (a) 整模板替换 | 把 `v400_make_cfg.py` 生成的 60 字节模板覆盖源 | `cc062cfbf9fd2c14a1225376e182beffc52f964c342c187b409045f495d5f0fa` | **不止关声**：`controllerMapping` 被清成全 0（源里是键盘映射 `0000010002000400ffffffffffffffff03000000`），`opts` 由 `0x1` 变 `0x6001`（`preloadMusic=true`、`disableVsync=true`）。**会改变输入映射与呈现路径** | 适合给 harness 自建的**私有 data 目录**放模板（§9.3 第 1 步），**不适合**直接覆盖共享源 |
| (b) 最小四字节改法（**推荐用于源**） | 只把 0x1F、0x20、0x27、0x28 置 0，其余 56 字节不动 | `cfa292ae10ee1907df7c637d013386851486dadf7a97bb5adf12bcf21dd14638` | **diff = 4 字节**；`controllerMapping`、`opts=0x1`、`lifeCount=2`、`bombCount=3`、`defaultDifficulty`、`windowed` 全部保持；解码 `problems: []` | 用于替换 `/root/v400-data/game/th08.cfg`（§9.2 的"最省事正确做法"，但以最小改动实现） |

**已就绪的产物（本次只写进仓库内新路径，`/root/v400-data` 一字未动）**

- `.tmp_v830_mute/th08.cfg` — 模板，60 B，`cc062cfb…`，解码 `musicMode/playSounds/musicVolume/sfxVolume` 全 0、`problems: []`
- `.tmp_v830_mute/th08.cfg.mute-only` — 最小改法产物，60 B，`cfa292ae…`，与源仅差 4 字节

**仍需授权的步骤（不要由代理自行执行）**：把上述任一文件落到 `/root/v400-data/game/th08.cfg` 属于**资产操作**——§8.2 保护 `/root/v400-data`，而 §9.2 建议换这一个文件；`ASSET_SYNC` 目前登记的是 `48bf7621…`，一旦替换**必须**同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`。请把这一步写进实施单，并指定用 (a) 还是 (b)。

**第四处更正（同轮内，两次）**：§1 初稿写"自然开局：3 残机、bomb 按机体"，是**未核实的先验**。第一次改成 cfg 字节（`lifeCount=2`、`bombCount=3`）；随后读引擎源码发现**两处仍不准确**：`lifeCount=2` 确实是 2 条备用残机（game over 在 `GetLives() <= 0`，共 3 次死亡），但**炸弹数并不来自 `cfg.bombCount`**，而是 `primaryShtFile->initialBombCount`（机体射击文件决定），且 `playerPower` 初始为 0、practice 模式会把 lifeCount 强制成 8。现已按源码重写该行并单列三个陷阱，`cfg.bombCount=3` 只作菜单默认值记录。

---

## 11. R2 前置的离线可核部分（本回合已跑，全绿）

R2 的第一份交付是"真实像素 → actor → 动作 → 实际帧 → 独立终态"的接线，其中**帧序/帧证明/终态核对这一层不需要引擎**，可以先在离线验证。本回合跑了这部分，用作接线前的基线（它**不能**代替真实引擎轨迹）。

| 范围 | 命令 | 结果 |
| --- | --- | --- |
| 帧栅栏 + 帧证明 + 历史证明 | `python -m pytest -p no:cacheprovider --basetemp .tmp_v830_pytest_run -q touhou_rl/tests/test_framefence.py touhou_rl/tests/test_frame_proof.py touhou_rl/tests/test_phase2_history_proof.py` | **27 passed / 0.38 s / 退出码 0** |
| 渲染像素一致性 + 确定性 + 像素死亡/配对 | `python -m pytest -p no:cacheprovider --basetemp .tmp_v830_pytest_run2 -q tests/test_v400_render_pixel_consistency.py tests/test_v400_render_pixel_determinism.py tests/test_v400_pixel_death.py tests/test_v400_pixel_pair.py` | **142 passed / 6.49 s / 退出码 0** |

两点说明，避免误读：

1. 这两批**没有启动任何引擎**（跑前跑后 `th08-modern` 进程数在 Windows 与 WSL 都是 0）。其中两个渲染测试只**断言** `SDL_AUDIODRIVER == "dummy"` 这一契约，不 spawn 引擎。
2. 它们证明的是**审计/判分机器没有坏**，不证明引擎轨迹能跑通、也不证明像素可学。按 `PIXEL_SLICE_PLAN` 的口径，这属于"离线边界测试可验证计分实现，不能填补缺失的真实成功或真实死亡"。

---

## 12. R2 的像素契约：`obs-v3` 已冻结、基本合规，且带着一条已记录的负结果

R2 需要"框内像素 + 自己执行历史"的观测。本回合确认**不必新造管线**：仓库里已有冻结的 `obs-v3`，而且它的输入边界符合 v8.30 的像素 actor 要求。以下全部由模块本身读出（离线，无引擎）。

### 12.1 契约（`touhou_rl/v400/student/obs_spec_v3.py`）

| 项 | 值 | 含义 |
| --- | --- | --- |
| `OBS_SPEC_VERSION` | `obs-v3` | 版本标识 |
| `PLAYFIELD_ORIGIN` / `PLAYFIELD_SHAPE` | `(32,16)` / `(384,448)` | 从 640×480 读回里裁 `x=32..416, y=16..464`，即**框内画面**（与 `v420/rl.py:3494` 的 `pixels[16:464, 32:416]` 一致） |
| `PLAY_EXTENT_WH` / `PLAY_SHAPE` / `PLAY_BLOCK` | `(96,112)` / `(112,96)` / `(4,4)` | 384×448 按 4×4 块降采样到 96×112 |
| `FRAME_STACK` | `4` | 播放面是 4 帧栈，配 `DECISION_FRAMES=2` ⇒ 历史跨度 6 帧 |
| `HUD_X0` / `HUD_X1` / `HUD_BANDS` / `HUD_SHAPE` / `HUD_FRAMES` | `480` / `640` / `((48,72),(80,96),(96,112))` / `(56,160)` / `1` | HUD 只取右栏 x=480..640 的三条行带，单帧 |
| `GRAY_SHIFT` / `GRAY_WEIGHTS` | `8` / `(77,151,28)` | 灰度化方式 |
| `NORMALIZATION` | `float32 / 255.0`，逐平面，无 mean/std，范围 [0,1] | 归一化 |
| `ACTION_HISTORY_SHAPE` / `_ORDER` / `_PAD` | `(8,18)` / `oldest-first a[t-8]…a[t-1]` / `all-zero row` | **自己实际执行的动作历史**（18 路 one-hot） |
| `BYTES_PER_DECISION` | `52112` | = play 4×10752 + hud 8960 + act 144。**R2 估磁盘就该用这个数** |
| 冻结摘要 | `obs_spec_v3.obs_spec_v3_digest()` = `ef2fbd15385f26f425db2bfe855d105ee009023acb08c26916e811963553a105`，与 `tests/test_v420_contracts.py:25` 的字面量**逐字符相同** | 本次重算通过 |

### 12.2 输入边界判定：合规（两个"特权平面"开关都关）

- 三个平面分别是**框内像素**（play）、**HUD 像素**（hud）、**自己执行的动作**（act_hist）——正好是 `PIXEL_SLICE_PLAN §3` 允许像素 actor 读的三类。
- **`SELF_STATE_PRESENT = False`** 与 **`DANGER_SUMMARY_PRESENT = False`**（obs-v3 与 obs-v2 都是）：既没有把自机状态真值、也没有把弹幕/威胁真值额外塞进观测。
- `act_hist` 的复位语义写死在契约里：**在 shard 起点与每一次阶段切换时清零**（`obs_spec_v3.py:15-17`）。R2 的"接管时记忆如何重建"应当直接引用这一条，不要自创填充方式。
- **帧栈的 burn-in 目前没有定义（真实缺口，须实施单冻结）**：契约只钉死了栈形（4 帧 `t-3..t`，oldest-first）与 `act_hist` 的清零时机，**没有**规定接管时帧栈要预热多少帧、不足时如何填充。现存的 `warmup_frames` 是 RESET 之后像素回暖的 `advanced` 帧数（`touhou_rl/v420/rl.py:1237-1257`），**与帧栈 burn-in 不是同一件事**，不能混用。`PIXEL_SLICE_PLAN` 第 50 行要求实施单明确记录“前缀末尾可用帧、burn-in 长度、填充方式、记忆重置条件和实际执行的前缀动作”，这一项必须在派单时冻结。
- 因此 R2 **不需要新的像素管线**，可以直接用 `obs-v3`；本单没有启动 render，所以"真实轨迹上是否逐帧可重建"仍未验证。

### 12.3 两个必须写进实施单的保留

| 保留 | 内容 | 为什么要紧 |
| --- | --- | --- |
| 教师辅助标签 | `POLICY_AUX_LABEL` = `verified_surviving_action_set`，来源 `TeacherLabel.verified_mask[18]`（`11-P1 §3`），定义"教师已验证在该决策选 c 会存活"。模块明确说**它不是输入、不额外存平面**，且"是 mask 不是概率，若要当 BCE 目标必须单独登记损失" | 路线允许辅助标签作为可检验的辅助选择，但**必须单列声明**，不能算进"像素自我改进" |
| **已裁决的负结果（不是"悄悄没过"）** | `PROBE_OUTCOME`：EXP-P3-010 在 held-out seeds 15–16 上测自机 48 px 内的占用率 F1，**"no candidate reached F1 >= 0.90, so the T2 stop-loss condition fired"**。随后 **`D-14`（2026-11-10）裁决**：仍然采纳分辨率 B(96×112)，但**把验收标准从"occupancy F1 ≥ 0.90"改为闭环分数**（`G3 S1N` 过关率：stage 1 normal、确认种子），F1 只作为**记录在案的代理指标**。B 的 F1 = 0.8502（预注册口径）/ 0.8661（事后最优阈值） | 所以正确的说法是：**该像素信息量代理指标未达 0.90，且验收口径已被正式改判为闭环过关率**。这既不是"已通过"，也不是"未处理"——R2 必须交的是**闭环结果**，不能拿 F1 或"接线通过"充当；而 F1 值被写进 digest，正是为了让读者不会误以为它过了门 |

**一句话**：观测侧已经有一份冻结、可复算、边界合规（无自机/威胁真值平面）的像素契约可用；但它的像素信息量代理指标**未达 0.90**，且验收标准已被 `D-14` 正式改判为**闭环过关率**。R2 的"观测是否有控制价值"必须用闭环结果回答，不能用接线、loss 或 F1 代替。

---

## 13. 动作契约里的一个结构性缺口：标准动作空间**无法表达"不射击"**，而过关需要 SHOOT 上升沿

这条是本次最影响 R1 可行性的发现，单独列出。

### 13.1 仓库里其实有三套动作空间

| 空间 | 定义 | 出处 |
| --- | --- | --- |
| 原始按钮组合 | `MultiDiscrete(9,2,2,2)` = 9 移动 × 射击 × focus × bomb = **72** | `touhou_rl/spec.py:389-407` |
| **教师候选** | `cand = focus * 9 + move` → **18** 个，**射击恒开、bomb 恒关** | `touhou_rl/v400/gym/actions.py:5-6,22,102` |
| 学生动作 | `(cand, bomb)` → **36** 个 | 同上 `:7,105` |

`MOVES` 的 9 个方向是：0 idle、1 N、2 NE、3 E、4 SE、5 S、6 SW、7 W、8 NW（+y 向下）。

### 13.2 缺口：`cand_to_mask` 无条件 OR 上 SHOOT

```python
mask = _MOVE_BITS[move] | BTN_SHOOT      # 无条件
if focus: mask |= BTN_FOCUS
if bomb:  mask |= BTN_BOMB
```

反向函数更直接地 fail-closed：

```python
if not mask & BTN_SHOOT:
    raise ValueError("mask does not shoot; not representable as a candidate")
```

其 docstring 的理由是"'not shooting' is never useful here"。**于是标准契约里"不射击"根本不是合法动作。**

### 13.3 但过关恰恰需要一个 SHOOT 上升沿

`T08/CLEAR_REACHABILITY.md` / `EVAL_SHOOT_PULSE.md` 的实测（本回执 §2 已核到产物字段）：同一条教师磁带，

- `shoot_pulse_period = 48`（每 48 决策在该决策最后一个 STEP 帧放掉 `SHOOT|BOMB`）→ **`stage_clear=true`，13,195 帧，2,631,430 分**；
- 释放周期极大（等价于从不释放）→ **`game_over`，15,564 帧**。

也就是说：**没有 SHOOT 上升沿就过不了关**，而"放掉射击"在 18/36 动作空间里**无法表达**。

### 13.4 推论（R1 必须处理）

1. **只用标准动作契约 + 无释放路径，"阶段完成"在构造上不可达。** `clear_reachability.json` 的裁决原文：`stage_clear_reachable_through_evaluator_channel = false`；机制是"`cand_to_mask` 把 `BTN_SHOOT` OR 进每个 mask、评测器没有释放路径，于是每帧都按着射击；引擎需要一个 `SHOOT` 上升沿才能离开等待状态，而评测器与 harness 都不提供"。
2. 过关所依赖的 SHOOT 释放来自**外生的运行器约定**（T08 的 `--shoot-pulse-period 48`），**不是策略能决定的东西**。所以任何"像素 actor 通了第一面"的说法，严格写法应当是"**像素 actor + 固定外生脉冲**"。
3. R1 冻结动作契约时只有两条路，必须选一条并写进实施单：
   - **(a)** 把 SHOOT 脉冲**作为任务定义的一部分**声明（周期、释放帧位置都冻结），并明确"策略不可控"；或
   - **(b)** **扩展动作空间，给策略一个 shoot 位，然后重新登记任务**——这正是研究端 R1 原文的口径："若 18 动作/禁 bomb 不可解……**应恢复所需合法动作并重新登记任务**"。
4. 裁决还给了三条**实施纪律**（`q3_fix.must_also`），R1 必须照抄：**把 `shoot_pulse_period` 记进 episode 身份**；**拒绝合并不同 `shoot_pulse_period` 的摘要**；**修复前写下的行若要比较就必须重跑，不得混用**。周期本身也要注意：`48 决策 × 2 帧 = 96 帧`，与 `configs/v420/rl-engine-{audit,pilot}.json` 已冻结的 `rollout.shoot_pulse_period` 一致；教师默认的 24 是配 4 帧决策的。

### 13.5 本条**不能**被用来改写历史结论（重要，防止误用）

同一份产物明确划了界，R1 引用时必须连边界一起引：

| 产物原话 | 含义 |
| --- | --- |
| `impact_on_existing_zero_clear_counts: "none measured: 1285 episode records, 1 clears"` | 扫了 **87 个文件 / 1,285 局**，只有 1 个 clear（且那是 reward audit 自己的 clear 重放）。**旧 0/N 不受本缺口解释** |
| `one_line: "…but no existing 0/N number is explained by it, because every evaluated episode ended in a death at <= 4964 frames, 8,000+ frames short of the clear"` | 旧策略在 ≤4,964 帧就死了，离过关的 13,195 帧差 8,000+ 帧——**它们本来也过不了** |
| `cannot_claim`: "the 0/N counts are unchanged and are still valid negative results for the reason they were reported" | **旧 0/N 仍是有效的负结果**，只是原因与通道无关 |
| `cannot_claim`: "the recorded tape is a teacher recording, not a policy: replaying it through the evaluator channel proves the channel is broken, not that any student could clear" | 13,195 帧那条是**教师录音**，它只证明通道坏了，**不证明任何学生能过关** |
| `cannot_claim`: "the fix, once applied, would not make any policy clear: it only makes a clear *reportable*" | 修好通道只是让过关**可被报告**，不会让任何策略变强 |

**所以 §13 的正确读法是**：这是一个**测量能力**缺口（不能报告过关），不是"旧失败被推翻"，也不是"通道修好就能过"。R1 若把主指标定为阶段完成，**必须先解决可报告性**，同时**不得**回头宣称历史上的 0/N 无效。

### 13.6 若选 (a)：脉冲的精确规则已经实现好了，R1 直接复用即可

`touhou_rl/v400/teacher/shoot.py` 就是这份约定的唯一实现，**R1 不要重写**。

**引擎级根因（比评测器层面更靠前）**：`GameManager.cpp:381-389` 判定离开阶段结束状态的条件是 `(g_CurFrameInput & 0x1001)` **非零且与上一帧不同**——`0x1001` = SHOOT(0x1) | ENTER(0x1000)。所以「按住 SHOOT」**不算上升沿**；而 `rlharness.cpp` 没有 `0x1000` 通道，**客户端必须自己发上升沿**。

| 规则 | 定义 | 出处 |
| --- | --- | --- |
| 触发判据 | `release_due(i, period) = ((i + 1) % period == 0)` —— 每第 `period` 个决策（1 起算）触发一次 | `shoot.py:50-55` |
| 释放帧位置 | **只释放该决策的最后一帧**：`[(mask, decision_frames-1), (mask & ~SHOOT & ~BOMB, 1)]`；`decision_frames == 1` 时退化为 `[(held, 1)]` | `shoot.py:58-80` |
| 同时丢掉 BOMB | 释放帧必须也去掉 BOMB——"the release frame must not be the frame that spends a bomb" | `shoot.py:67-69` |
| bomb 归属 | bomb 只挂在**第一个**子决策上（"the next decision presses bomb"，不是整段按住） | `commit_block` `shoot.py:83-109` |
| 周期取值 | 教师默认 `SHOOT_PULSE_PERIOD = 24`，配 **4 帧决策** = 96 帧；**2 帧决策的等价取值是 48**（T08 用的就是 48）。原话："Any period works: the transition state waits indefinitely for the edge" | `shoot.py:43-47`；`T08/clear_reachability.json.q3_fix` |
| 代价 | 占射击时间 `1/(period × frames_per_decision)`，默认 96 帧里 1 帧 ≈ **1.04%** | `shoot.py:22-25` |
| 可核计数 | `mask_summary()` 给出 `shoot_frames` / `release_frames`，可与见证的 `shoot_frames=13058` 对照 | `shoot.py:118-129` |

**给实施单的一句话**：若 R1 选 (a)，直接 `from touhou_rl.v400.teacher.shoot import release_due, commit_frames`，并把 `period` 与 `decision_frames` 一起写进任务身份；`48 决策 × 2 帧 = 96 帧` 与已冻结的 `rollout.shoot_pulse_period` 对齐，不要另发明周期。

---

## 14. 回执自检（本回合新增，机器核对）

本文档经 9 次提交修改，为防止跨节自相矛盾，本回合加了一次**离线自检**（脚本 `.tmp_v830_audit.py`，只读本文件）。

| 检查维度 | 判据 | 结果 |
| --- | --- | --- |
| 章节编号 | `## N.` 必须从 0 连续 | **0–13 连续，无缺号无重号** |
| 交叉引用 | 文中 `§N` 必须指向存在的节 | **全部可解析**（原先把 R51 的行号区间误写成章节引用形式，现已区分为 `R51:19-28` / `R51:82-86`） |
| 12 个关键事实 | 必须在文中出现 | **12/12 命中**：13,195 / 2,631,430 / 15,564 / 19887 / 52112 / `ef2fbd15…` / `691bfd77…` / `48bf7621…` / `cfa292ae…` / `37b3e21d…` / `1523df26…` / `381-389` |
| 被推翻的旧说法 | 只允许出现在"自我更正"记录里 | **仅第 279 行**，且该行是更正记录本身（引用旧文） |
| 表格完整性 | 不得有缺列的表格行 | **0 处畸形** |
| 控制字符 | 不得有 `\x00-\x1f`（换行/制表除外） | **0 个** |
| 证据分类词 | 只用约定词汇 | 本次只读核实 42 处、历史记录 26 处、"缺口" 13 处、研究端 16 处 |

自检结论：**上述六个维度无矛盾**。这只说明文档内部一致，**不**说明其结论已被新实验验证。

---

## 15. 实施前的命名占用核对（在写第一行代码之前必须先解决）

`PIXEL_SLICE_PLAN §7` 给了三个"未来待实现"的路径。本回合核对了它们是否可用，并发现一处**会实际咬人的命名冲突**。

### 15.1 三个规划路径当前都是空的

| 规划路径 | 现状 |
| --- | --- |
| `touhou_rl/v830/pixels.py` | **不存在**（`touhou_rl/` 下没有任何 `v8*` 包） |
| `tests/test_v830_pixels.py` | **不存在** |
| `tools/rl/v8_30.py` | **不存在** |

也就是说没有半成品需要接管，也不会有并行写同一入口的情况。

### 15.2 但 `v830_` 这个名字**已经被 v8.3.0 占了**

`tools/rl/v830_bombpol.py` 的首行是 `"""v8.3.0 BOMBPOL-001 …"""` —— **本仓库里 `v830_` 指的是 v8.3.0，不是 v8.30.0。**

两套既有约定都能印证这个歧义：

| 约定 | 实例 | 含义 |
| --- | --- | --- |
| 包目录 `vXYZ` = X.Y.Z | `touhou_rl/v430/` | **v4.3.0**（不是 v4.30.0） |
| v8.x 工具用 `v8<字母>0` | `v8a0`=8.10.0 … `v8u0`=8.28.0（21 个工具，非单调：12/17/27 各有两字母共用，8.13.0 无工具） | v8.x.0 线 |
| 最近的版本换了写法 | `v829_pair.py` 等 5 个 | **v8.29.0** |

按 `v829_` 的最新写法类推，v8.30.0 会写成 **`v830_`** —— 正好撞上 v8.3.0 的 `v830_bombpol.py`。所以：

- 规划里的 `tools/rl/v8_30.py`（下划线）**是好的**，它天然避开了冲突，建议保留；
- 但 **`touhou_rl/v830/` 会被读成 v8.3.0**，建议在实施单里换成一个无歧义的包名（例如 `touhou_rl/v8300/` 或 `touhou_rl/v8_30/`），并把选择写进单子；
- 若真出现 `touhou_rl/v830/`，将来 `import touhou_rl.v830` 与 v8.3.0 的放雷线（`v830_bombpol.py`）在读者和工具链里都容易混。

### 15.3 给实施单的最小要求

1. **显式登记版本→路径映射**（例如"v8.30.0 ↔ `tools/rl/v8_30.py` + `touhou_rl/<选定的新包名>/`"），不要靠推断。
2. **不要**新建 `touhou_rl/v830/`。
3. 新建的包/工具在首次提交时就写清版本号，避免与 v8.3.0 的 `v830_` 线混淆。

这条只是**命名与路径占用**核对，不涉及任何行为或能力判断；三项均为本单只读核实。

---

## 16. 需求覆盖核对：工单的六个「最少回答」是否逐条答到

§14 查的是**内部一致**，本节查的是**需求覆盖**——把工单六个输入格的"最少回答"拆成可核对的子问题，逐条确认回执里有没有对应答案。

| 输入 | 子问题数 | 覆盖 |
| --- | ---: | ---: |
| 1 一个真实攻击阶段 | 8 | 8 |
| 2 现存成功与失败见证 | 6 | 6 |
| 3 像素与执行历史 | 6 | 6 |
| 4 同约束控制器与像素候选 | 3 | 3 |
| 5 少量开发祖先 | 3 | 3 |
| 6 一个切片的实际成本 | 4 | 4 |
| **合计** | **30** | **30** |

**核对中发现的一个真实缺口（已补）**：第 3 项要求回答"接管前帧栈/记忆如何重建"。`act_hist` 的清零时机与栈形（4 帧 `t-3..t`）在契约里有定义，但**帧栈的 burn-in 长度与填充方式没有任何现有定义**；现存的 `warmup_frames` 是 RESET 后像素回暖的帧数，与 burn-in 不是同一件事。已在 §12.2 补成一条**实施单必填项**。

**核对中的一次误报（留痕）**：该项最初被判缺失，是因为核对脚本用英文 `stage change` 作探针，而回执写的是"阶段切换"。修正探针后通过——**误报来自探针，不是回执缺内容**。

两点边界：

1. 本核对只证明"每个子问题在文中**有对应答案**"，**不证明答案正确**；正确性由各节标注的证据路径与 §14 的一致性检查支撑。
2. 与 §14 一样，这**不是**实验验证。工单要求的"训练端本次只读核实"到此闭合；R1 冻结任务与 R2 接线的实质结论，仍然要等实施单。

> **配套文件**：可勾选的实施提案见 [TRAINING_HOST_PROPOSAL_R1.md](TRAINING_HOST_PROPOSAL_R1.md) 与 [TRAINING_HOST_PROPOSAL_R2_P1.md](TRAINING_HOST_PROPOSAL_R2_P1.md)（训练端草稿，待研究端批准；不含任何开工）。
