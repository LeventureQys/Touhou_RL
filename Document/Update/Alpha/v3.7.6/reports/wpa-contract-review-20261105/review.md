# WP-A 合同与动作对齐审查（只读，含一处必修 P0）

- 适用版本：v3.7.6（真实 F2 回放示范行为克隆分支）
- 审查对象：[f2_labels.py](<../../../../../../touhou_rl/f2_labels.py>) · [f2_dataset.py](<../../../../../../touhou_rl/f2_dataset.py>) · [v380/replay.py](<../../../../../../touhou_rl/v380/replay.py>) · [tactical_sequence_train.py](<../../../../../../touhou_rl/tactical_sequence_train.py>) · [tactical_feature_cache.py](<../../../../../../touhou_rl/tactical_feature_cache.py>) · [frame_proof.py](<../../../../../../touhou_rl/frame_proof.py>) · [framefence.py](<../../../../../../touhou_rl/framefence.py>) · 三份真实 `exact` 采集（`third_party/th08/build/rl/exact/v376-f2-*-20261104-r1`）
- 证据：[对齐探针 JSON](<alignment-probe.json>) · 探针源码 [v376_wpa_alignment_probe.py](<../../../../../../tools/rl/v376_wpa_alignment_probe.py>)
- 本文**不**改变任何门状态。真实训练仍为 `blocked / pending_visual_review`；本审查不构成 G3/G4 通过。

---

## 0. 结论

**必须先修（已修）→ 修复后可继续，但仍受画面复核阻断。**

发现 1 项 P0：**回放记录索引被当成 `stageFrames`，而真正的回放游标是 `logicFrame`**。在出现过中场对白的回放里，两者永久错位，导致该段之后的每一行标签取到 149（train）/393（validation）个逻辑帧之前的输入。test 来源恰好没有对白，所以这条错误只在两个来源上发生、且**不会**被现有 build/audit 发现。

已在本次会话修复并加回归测试；修复使用游戏自己发布的已消费输入做逐行证明（9000/9000 全中），不是换一个假设。

| 结论项 | 结果 |
|---|---|
| 观测—按键—生效—位移时间线是否成立 | 成立，已用 9000 帧逐帧证明（见 §1） |
| 标签索引是否与观测因果对齐 | **修复前：否（P0-1）**；修复后：是，且被逐行证明 |
| 是否存在未来信息泄漏 | 否（经修复后；`observation` 只含 ≤ 当前帧的像素与已消费历史） |
| test 是否泄漏进缓存/训练/选模 | 否（`tactical_feature_cache.py:88`、`tactical_sequence_train.py:194`） |
| 是否有文档把 BC 写成原 G4 通过 | 否 |
| 放行判定 | `必须先修 → 已修 → 允许继续 WP-B/WP-C` |

---

## 1. 观测—按键—生效—位移时间线（已证实）

游戏侧每个逻辑帧的固定顺序（逆向源码 + 链优先级）：

| 步骤 | 位置 | 内容 |
|---|---|---|
| 1 | `Global.hpp:52-70` | 链优先级：GameManager=2 → 回放播放=6 → Player=9 → 跳过控制=18 |
| 2 | `GameManager.cpp:580` | `gameplayFrameCounter++`（即 proof 的 `logicFrame`） |
| 3 | `ReplayManager.cpp:329-330` | `g_GuiMessageInputCurrent = *(u16*)replayInputCursor; cursor += 2`（消费 **record[L-1]**） |
| 4 | `Player.cpp:618-639` | 用 `g_GuiMessageInputCurrent` 计算自机移动/射击 |
| 5 | `Player.cpp:1072-1075` | **仅当无对白**才 `stageActiveFrames++`（即 proof 的 `stageFrames`） |
| 6 | `main.cpp:348→370→381→385` | RunCalcChain → RunDrawChain → Present → `FrameFence::OnDrawnFrameEnd()` |
| 7 | `framefence.py:546-583` | 抓两次像素、读内存、`hold_stable`、`ack` |
| 8 | `exact_capture.py:1142` | `inputReadback = state.gui_input_current`（写进 proof 行） |

因此：**观测（logicFrame L 的像素）= 已应用 record[L-1] 之后的状态**；**BC 标签应为 record[L]（该观测之后才生效的下一步决策）**；**历史动作应为 record[L-1]**。

三份真实来源的逐帧证明（[探针](<alignment-probe.json>)）：

| 来源 | `gui_input_current == records[logicFrame-1]` | `gui_input_previous == records[logicFrame-2]` | 速度方向 == record[L-1] | 速度方向 == record[L] |
|---|---|---|---|---|
| train | **3000/3000** | **3000/3000** | 2013/2013 | 1853/2013 |
| validation | **3000/3000** | **3000/3000** | 1747/1747 | 1585/1747 |
| test | **3000/3000** | **3000/3000** | 1781/1781 | 1620/1781 |

即：回放游标 = 逻辑帧，且「本帧画面里的速度」与 record[L-1] 100% 一致、与 record[L] 只有 92%。这是源码语义与实测双向一致的证据，不依赖「现有测试通过」。

---

## 2. P0-1：标签按 `stageFrames` 索引 → 对白之后永久错位（已修）

### 机制

`stageActiveFrames` 在对白期间停走（`Player.cpp:1072`），而回放游标照旧每个逻辑帧消费一条记录（`ReplayManager.cpp:329`；对白快进时靠 `Global.cpp:198` 的 `RESTART_FROM_FIRST_JOB` 让同一个绘制帧内连续推进 3 个逻辑帧）。两者一旦分开，**永远不会再对齐**。

修复前 [f2_dataset.py](<../../../../../../touhou_rl/f2_dataset.py>) 第 310-312 行硬编码 `stage_frame_offset=0, aligned_stage_frame=frame`，`f2_labels.py:255` 用 `index = frame - 0` 取记录：调用方给的是 `stageFrames`，于是 train/validation 在中场对白之后全部取到过期输入。

### 影响面（真实数据实测）

| 来源 | 出现停走的位置 | 停走样本数 | 最大滞后（逻辑帧） | 标签与正确值不同的行 | 其中「本来有效却被写错」 | 其中「旧索引判 unknown、实际有效」 | 受影响的候选窗口 |
|---|---|---|---|---|---|---|---|
| train | `stageFrame 5451`（sample 2712） | 288 | **149**（≈2.5 s） | 195 | 120 | 75 | **3/43**（i2820 21 行、i2880 41 行、i2940 44 行） |
| validation | `stageFrame 5706`（sample 2850） | 150 | **393**（≈6.5 s） | 63 | 0 | 63 | 0/45（该段不在候选窗内） |
| test | 无停走 | 0 | 0 | 0 | 0 | 0 | 0/45 |

这不是「偏一帧」的语义争议，而是**取了 2.5～6.5 秒前的按键**：标签与画面里的动作因果无关。同时它掩盖了 75 行本可监督的 train 样本（被写成 unknown、零梯度）。

### 最小复现（修复前）

```powershell
python tools/rl/v376_wpa_alignment_probe.py --json <out.json>
# 修复前预测：train/validation 的 readback_matches_consumed_record 仍 3000/3000（证明游标=logicFrame），
# 而 stage_frame_equals_logic=false，stage_frame_lag_samples>0；test 全等。
```

### 修复

1. [f2_dataset.py](<../../../../../../touhou_rl/f2_dataset.py#L338-L350>)：`logic = current_proof['logicFrame']`，取 `records[logic-1]`/`records[logic-2]` 与同 hold 内存里的 `gui_input_current`/`gui_input_previous` 逐行比对，不一致即 `F2DatasetError`；标签索引、历史索引、`steps` 时间基准全部改为 logic 时钟，`steps['stage_frame']` 保留图像键，`alignment_diagnostics.stage_clock_lag` 把滞后显式记录进 manifest。
2. [f2_labels.py](<../../../../../../touhou_rl/f2_labels.py#L241-L275>)：删除 `stage_frame_offset` 语义，改为显式 `record_index` + `observation_frame` + `consumed_mask` + `previous_mask`，并在 `_replay_candidate` 内**重新解码并复验**这两条输入字；错位、越界、来源不符一律 unknown（零梯度），不再能被「一个 offset 数字」蒙过去。
3. 回归：`test_replay_clock_is_the_logic_clock_not_the_stage_clock`（构造 100 帧钟差的端到端夹具，按 stage 索引会得到不同标签）、`test_misbound_consumed_input_rejected`（消费字被挪一帧即拒绝构建）、`test_unaligned_actions_never_supervised[wrong_consumed]`。

### 不能靠改文档解决

`stageFrames` 仍然是**图像键**（PNG 窗口 f-4/f-2/f 与 hold 分组都用它），这次只把**回放记录索引**与**行时间基准**换成逻辑时钟；两者不可互相替代，已在模块 docstring 中写明。

---

## 3. P1：合同强度问题（不影响本轮构建，但影响「独立复核」的效力）

**P1-1 「独立像素标注」被特权坐标钉死。** [f2_dataset.py](<../../../../../../touhou_rl/f2_dataset.py#L317-L320>) 要求 `observation['player_px']` **精确等于**内存态 `player_x+32, player_y+16`；`_verify_teacher` 又要求 `aim_player_px` 与内存自机位置在 `position_px` 容差内、`pixel_tracks` 与 `slot:N` 坐标在容差内。于是「像素测得的自机位置」在合同上必须复刻内存值——它是**一致性检查**，不是独立测量。结合 `reviewer.id` 是自由字符串（无身份验证），文档中「独立像素标注」的强度应写成「在特权坐标容差内的一致性标注」。结论不变（仍需人看），但不要把该字段当独立证据。

**P1-2 弹体槽生命周期未受保护。** `_verify_teacher` 只校验「该帧该 slot 唯一存在且坐标一致」；同一 `slot:N` 在不同帧复用（不同弹体）时，像素轨迹只要逐点对上就能拼成一条「轨迹」，合同不检查轨迹的速度连续性/瞬移。真正会拦住它的是 `aim_cross_px_per_frame`（速度只由前两个可见样本推导，且只要求指向自机）。→ 复核时必须在 review 里显式写明生命周期划分，不能靠校验器兜底。

**P1-3 「两次瞄准发射」是弱判据。** `emissions` 只要求：(a) ≥2 个不同帧，(b) 同一个 `emitter_id` 字符串，(c) `origin` 与首个可见样本在容差内一致、`velocity` 与「样本 1→2 的位置差」在容差内一致，(d) `velocity` 与「origin→标注自机点」方向不被判为背离。它**没有**要求弹道在发射时刻真正穿过自机当时位置（只要求指向标注点，而标注点允许是 `aim_player_frame ≤ frame` 的任一帧）。因此「恰好朝自机早前位置飞去」的弹流也能通过。→ 结论仍是「这是可复核的结构合同，不是瞄准性的证明」。

---

## 4. P2：命名与语义不一致（记录在案，不阻塞）

| 编号 | 问题 | 位置 | 影响 |
|---|---|---|---|
| P2-1 | `hold_stable` 只是「ack 前仍在同一 hold」的活性检查，**不**证明「本帧输入 == 上帧输入」 | `framefence.py:491-492,582` | 名称容易被读成「同一按键保持」，不能当输入稳定的证据 |
| P2-2 | `input_mask` 帧证明比较的是**物理键盘**（`g_CurFrameInput`，回放播放时为 0），与标签无关 | `framefence.py:598`、`framefence.cpp:525` | 它证明不了任何回放对齐；真正的守卫是本次新加的消费字复验 |
| P2-3 | 采样间隔是 2 个逻辑帧（`hold_every=2`，实测钟差 2/4/6），但 `capture_interval` 写成 `[frame, frame]`、`frame_delta` 恒为 1 | `f2_dataset.py:385-393` | 读 `steps` 的下游若把相邻行当「相邻帧」会错；`frame_delta` 本身按逻辑帧算是对的 |
| P2-4 | `exact-*.npz` 里的 `action` 是 `gui_input_current`（**已消费/过去**动作），与 F2 BC 的「下一步动作」语义相反 | `exact_capture.py:1414` | 将来若有人直接拿 npz 的 action 当 BC 标签，会整体偏一帧；建议在 shard schema 注释里显式标注 past-action |
| P2-5 | `previous_actual_option=3`、`previous_target=[0,0]`、`target_valid=False` 是 F2 分支的常量占位 | `f2_dataset.py:379-392` | 与「不适用即 null/mask=false」一致，但训练侧要清楚 option/target 头永远不参与 |

---

## 5. 八个必答问题逐条回答

1. **观测帧对应前一帧还是后一帧？** 观测 = logicFrame L 的像素，此时已应用 record[L-1]；**标签必须是 record[L]**（下一步决策）。证据：§1 的源码顺序 + 9000/9000 消费字匹配 + 速度方向 100%/92% 对比。
2. **`stageFrames → record index → candidate` 偏移是否正确？** 不正确。正确关系是 `record_index = logicFrame`，`consumed = logicFrame-1`；`stageFrames` 只在「从未出现对白」时与 `logicFrame` 相等，一旦停走即永久滞后。已修。
3. **`previous_actual_candidate` / `effective_frame` / `hold` 语义是否一致？** 修复后一致：三者都由「本帧实际消费的 record[L-1]」推出，`hold` 是该 record 在回放里连续相同的长度，`effective_frame = L-1`。修复前存在题目所说的坏情况：对停走后的行，「已执行历史」指向与画面无关的 149 帧前的输入。
4. **是否给了部署时拿不到的信息？** 修复后没有：`observation` 只含 ≤ 当前帧图像（SHA 绑定）、`observed_through ≤ frame`、`bullet_tracks` 只允许 ≤ 当前帧；历史字段只含已消费动作。标签 record[L] 是「此刻要做的决策」，不是观测里的信息。
5. **像素标注是否被内存坐标反向污染？** 是（P1-1）：自机位置必须等于内存值，瞄准点被内存值容差约束。这是设计选择，但文档不能把它称为独立测量。
6. **槽位复用是否会拼错轨迹 / 两次瞄准判据是否成立？** 见 P1-2、P1-3：会，且判据较弱；靠复核者显式标注 + 容差，不靠校验器。
7. **test 是否只在审计阶段读取？** 是。`f2_dataset.audit_dataset` 有意重算全部 split；`tactical_feature_cache.build` 跳过 test（`tactical_feature_cache.py:88`），`tactical_sequence_train.prepare` 只加载非 test（`tactical_sequence_train.py:194`），cache/report 里 `test_tensor_loaded=False`、`test_cache_created=False`；`seq.train` 报告 `test_used=False`。
8. **是否有文档/报告把 BC 写成了原 G4 通过？** 没有。manifest/event/report 全部带 `objective='f2_behavior_cloning'`、`original_G4=False`；`check-gate` 打印明确限定语；`任务清单.md` 把 U05 系列保持 `not_run`。

---

## 6. 复核命令

```powershell
# 1. 对齐证据（只读，不改数据）
python tools/rl/v376_wpa_alignment_probe.py --json Document/Update/Alpha/v3.7.6/reports/wpa-contract-review-20261105/alignment-probe.json

# 2. 与本审查相关的合同测试
python -B -m pytest touhou_rl/tests/test_f2_labels.py touhou_rl/tests/test_f2_dataset.py `
  touhou_rl/tests/test_f2_end_to_end.py touhou_rl/tests/test_f2_training_gate.py `
  touhou_rl/tests/test_v376_review_pack.py -q -p no:cacheprovider --basetemp .tmp/wp-a-tests

# 3. 全量回归（见 README 登记的新日志）
python -B -m pytest touhou_rl/tests tests -q -p no:cacheprovider --basetemp .tmp/v376-full-20261105-wpa
```

> 环境注记：本机 pytest 无法写 `%TEMP%\pytest-of-*`（`WinError 5`），必须用 `--basetemp` 指向工作区；这与上一轮 `regression-20261104-r3` 的命令一致，不是本轮引入的问题。

---

## 7. 对后续工作包的影响

- **WP-B（视觉复核）不受影响**：复核用 `stageFrames` 做图像键是对的（PNG 与 f-4/f-2/f 窗口都按它），本次只改回放记录索引。复核模板无需改字段。
- **WP-C（冻结 plan）**：可以照原计划冻结；但必须在计划里写明「行时间基准 = logic 时钟，图像键 = stage 时钟」，并把 `alignment_diagnostics.stage_clock_lag` 作为已知数据事实写入。
- **WP-D（build/audit）**：新合同会在 build 时逐行复验消费字；任何对齐漂移直接失败，不会静默出数据。audit 会重算同一检查。
- **WP-E**：缓存绑定源码哈希，源码已变（`f2_labels.py`/`f2_dataset.py`），必须重建缓存，不能复用旧 cache。
- **仍然阻断**：本修复不产生任何 review、不改变 `pending_visual_review`，真实训练继续 blocked。
