# V831-SLICE-01 补遗：A7 关闭、A5 需修正、两处口径已钉死

**主机代号**：轩神开播了（WSL2 `ubuntu`，20 vCPU / 23 GiB，RTX 4070 SUPER）
**时间及时区**：2026-11-26 09:0x – 09:4x +08:00
**性质**：对 [`TRAINING_HOST_REPLY_01.md`](TRAINING_HOST_REPLY_01.md) 的**补遗**（addendum），不是新工单的回执。**未启动任何引擎进程**，全部为只读源码阅读与既有产物复算。
**检出与 HEAD**：`/mnt/d/WorkShop/Touhou_RL`，HEAD `11807614ca8453d62e48eff6ca4c4df804cd1bae`（与本单主回执同一提交，期间无新提交）。

---

## 0 三句话结论

1. **A7 关闭**：R74 的"双 mask"首冲突源文件是 **D1 canonical 语料** `.tmp_v400_d1/d1/shards/d1-s0d1-n3371-seed12-shard/labels.jsonl`（`457728a9…ccbd`），不是 T04 的 2 帧磁带；两个冲突掩码是 **0x0085(133) → 0x0084(132)**，位于 `decision_frame=92` 那一行的第 3、4 帧（`step_frames=[149,149,133,132]`）。**R74 记的"帧 92"是该行的 `decision_frame` 标号，`_pair_cand` 实际在第 96 帧抛错**，两者差 4 帧。
2. **A5 的结论要改**：8 个 seed-12 shard 里**每一处**双 mask 冲突都**只差 `BTN_SHOOT` 一位**，真正的候选级冲突 **0 处**；s0d1 的 137 处冲突恰好等于 `shoot_release=True` 的行数，且首 4 个释放行的间隔是 **96 帧 = 48 决策 × 2 帧**。⇒ "双 mask"不是磁带缺陷，而是 **SHOOT 脉冲帧混进了"一决策一 mask"的判定**。存在一条**不需要重录任何资产**的解阻路径（只需改动作历史重建，不改 STEP 掩码，故 HASH 链不变）。
3. **两处待核验已钉死**：`bombs_remaining` 来自 `primaryShtFile->initialBombCount`，**`cfg->bombCount` 从不参与资源初始化**；13,118 → 13,195 的 **77 帧** 是脚本化过关序列的长度，期间**恰好有一次输入上升沿（13,153）**，阶段前进在**该边沿之后 42 帧**——所以"过关帧 = 输入边沿帧"这个假设**被实测否定**。

---

## 1 A7：pair prefix 源文件与两个具体掩码

### 1.1 源文件身份

| 项 | 值 |
| --- | --- |
| 路径 | `.tmp_v400_d1/d1/shards/d1-s0d1-n3371-seed12-shard/labels.jsonl` |
| 字节 / sha256 | **5,822,072 B** / **`457728a988717ae2e5deae425196a43a6d0de4cbb21f18b4c41630f9d3aaccbd`** |
| 同目录 `shard.json` | 1,298,565 B / `d48bc569e1c4cb6e644770403f75144342f98f7995d16cd68b725a679ccafc0f` |
| 行数 / 帧数 | 3,298 行 × 4 帧 = **13,192 帧**（与 v8.31 教师磁带的 13,192 个观测帧哈希同数） |
| 每行字段（关键） | `frame` / `decision_frame` / `decision_frames=2` / **`frames_per_decision=4`** / `step_frames`(4) / `shoot_release` / `bomb` / `chosen_plan` / `scores`(18) / `verified_mask`(18) / `hash_pre` / `hash_post` |

### 1.2 冲突点（逐字节）

```
row[23]  decision_frame = 92   frame = 92
step_frames  = [149, 149, 133, 132]        # 覆盖 replay 帧 93..96
hash_pre     = 7e9b64232d230a26
hash_post    = b131114d28bd2a4c
shoot_release= true        bomb = 0
pair 判定（replay() 在 frame%2==0 时比较 masks[i-1] 与 masks[i]）
  frame 94 : 149 vs 149  -> 通过
  frame 96 : 133 vs 132  -> 冲突   0x0085 -> 0x0084
```

`_pair_cand`（`touhou_rl/v420/counterfactual.py:3924-3940`，owner class `_StudentSession`）抛出的**原文**：

```
frame 96: the recorded two-frame decision carries two different masks
(0x0085 then 0x0084); the action history cannot be rebuilt
```

`0x0084 = 0x0085 & ~BTN_SHOOT`，即**唯一的差别是 SHOOT 位**。该掩码也**不是候选可表示的**（`mask_to_cand` 抛 `mask does not shoot`）——`_pair_cand` 的两条 fail-closed 里会先命中"双 mask"这一条。

### 1.3 对 R74 措辞的更正（重要）

| R74 原文 | 实测 |
| --- | --- |
| "首个冲突**帧** = 92" | `_pair_cand` 在 **replay 帧 96** 抛错；**92 是该行的 `decision_frame` 标号**，也是 `frame` 字段值 |
| "每条磁带的 prefix 从 frame 92 起就有双 mask 决策" | 成立但需精确化：**从第 24 个 4 帧块的最后一对（帧 95/96）起** |
| "目标段 = 帧 8,664" | 本补遗**未复核**这一条（属 R74 原文，仍为历史记录） |

**8 个 seed-12 shard 全部一致**（首个冲突同在 `row[23]`、`decision_frame=92`、抛错帧 96）：

| shard | 行 | 帧 | 首个冲突 | 两掩码 | `labels.jsonl` sha256 |
| --- | ---: | ---: | ---: | --- | --- |
| `d1-s0d1-n3371` | 3,298 | 13,192 | 96 | 133→132 | `457728a9…ccbd` |
| `d1-s1d1-n4000` | 4,000 | 16,000 | 96 | 133→132 | `5e6cfd89…4091` |
| `d1-s2d1-n4000` | 4,000 | 16,000 | 96 | 133→132 | `5814b140…7eaf` |
| `d1-s3d1-n3833` | 2,019 | 8,076 | 96 | 133→132 | `50bb0f0d…0c67` |
| `d1-s4d1-n4000` | 4,000 | 16,000 | 96 | 133→132 | `0c803bfc…4b8b` |
| `d1-s5d1-n4000` | 4,000 | 16,000 | 96 | **1→0** | `018e7fb8…f366` |
| `d1-s6d1-n3836` | 3,218 | 12,872 | 96 | **1→0** | `f9e9aaa5…38fb` |
| `d1-s7d1-n3218` | 3,218 | 12,872 | 96 | **1→0** | `5a072211…cdeb` |

> stage 1–5 的两个值是 `0x0085/0x0084`（候选 12 松扳机），stage 6–8 是 `0x0001/0x0000`（候选 0 松扳机）。**同一位置、同一机理**，说明这是记录器的结构属性，不是某条磁带偶然。

**另一处必须点明的口径**：`t02c/rebuild` 与 `t02c/replay_data` 下**没有** `labels.jsonl`（后者是私有游戏数据目录），所以 R74 与 A 卡里"可能是 T04 2 帧磁带"的猜测**不成立**；`corpus_prefix()` 只吃带 `step_frames` 的 4 帧行。

---

## 2 A5 需要修正：这是 SHOOT 脉冲的编码产物，不是磁带缺陷

### 2.1 全语料复算（8 个 shard × 每一对 2 帧）

| shard | 原始双 mask 冲突 | 其中仅差 `BTN_SHOOT` | **真正的候选级冲突** |
| --- | ---: | ---: | ---: |
| `d1-s0d1-n3371` | **137** | **137** | **0** |
| `d1-s1d1-n4000` | 166 | 166 | **0** |
| `d1-s2d1-n4000` | 166 | 166 | **0** |
| `d1-s3d1-n3833` | 84 | 84 | **0** |
| `d1-s4d1-n4000` | 166 | 166 | **0** |
| `d1-s5d1-n4000` | 166 | 166 | **0** |
| `d1-s6d1-n3836` | 134 | 134 | **0** |
| `d1-s7d1-n3218` | 134 | 134 | **0** |

**判据**：把一对掩码双双清掉 `BTN_SHOOT` 后再比较；清后相等 ⇒ 该冲突只由扳机解释。**没有一个 shard 出现清后仍不等的对。**

### 2.2 三条互相独立的自洽证据

1. **数目对得上**：s0d1 的冲突 137 条 = `shoot_release=True` 的行数 **137**（3,298 行中）。
2. **相位对得上**：s0d1 首 4 个释放行的 `decision_frame` 是 **92, 188, 284, 380**，步长 **96 帧 = 48 决策 × 2 帧**——正是 `configs/v420/rl-engine-*.json` 与 T08 冻结的 `shoot_pulse_period = 48`。
3. **非释放行不冲突**：有 1,665 行 `step_frames` 有 ≥2 个不同掩码但 `shoot_release=False`，它们的变化都发生在**两对之间**（如 `[1,1,69,69]` = 候选 0,0 然后 16,16），**每一对内部相等**，因此不触发。也就是说"4 帧块 = 2 个 2 帧决策"这一读法是对的，错的只是**把松扳机帧当成换动作**。

### 2.3 为什么这条推翻了 A5 的"必须重录磁带"

A 卡把 A5 写成"中局合法前缀不存在（R74 四选项已排除三条）"，并把"重录一决策一 mask 的干净磁带"列为唯一解阻项（需资产授权）。实测表明这条路**可能不必要**：

- `replay()` 里 `_pair_cand` **只用于一件事**：把"这一帧的掩码"折成 `cand` 推进 `action_history`（`:3989`）。**STEP 用的掩码直接来自 `prefix.masks`（`:3962`），与 `_pair_cand` 无关。**
- 所以只需在**动作历史重建**这一步容忍"仅差 SHOOT 位"的成对掩码（把 `mask | BTN_SHOOT` 再送 `mask_to_cand`，或对清掉 SHOOT 后的两值判等），**STEP 掩码逐帧不变 ⇒ HASH 链逐帧不变 ⇒ R74 的"0 不符"前提不被破坏**。
- 这属于 `touhou_rl/v420/counterfactual.py` 的代码改动，**不在任何已授权白名单内**，本补遗**未实施**，只提出。是否改、以及改哪一侧（改 `_pair_cand`，还是在 `corpus_prefix()` 里额外带一份"纯净候选序列"），由研究端与新工单决定。

**本补遗不主张**这条路径一定能跑通：改完后 `replay()` 是否真的能到 8,664 帧、以及到达后的状态是否仍是"录像里的那个状态"，需要**一次真机前缀重放**才能回答（不是本补遗的结论）。

---

## 3 `bombs_remaining` 口径：已钉死

`src/GameManagerSetup.cpp:123-130`（Normal、非 replay、非 practice 的自然开局路径）：

```cpp
if (!g_GameManager.flags.isReplay)
{
    configMode = gameManager->cfg->lifeCount;
    g_GameManager.globals->livesRemaining = static_cast<f32>(configMode);
    g_GameManager.UpdateAntiTamper();
    g_GameManager.SetBombCount(static_cast<i32>(
        g_Player.primaryShtFile->initialBombCount));   // <- 炸弹来自机体射击文件
}
```

- **`livesRemaining` ← `cfg->lifeCount`**（Normal 为 2；`GameManagerSetup.cpp:111-113` 把 Extra 设为 2、practice 设为 8）。
- **`bombsRemaining` ← `primaryShtFile->initialBombCount`**，与 `cfg->bombCount` **无关**。
- `cfg->bombCount` 在整个源码里**只有三处**：`Supervisor.cpp:1395` 写默认 3、`TitleScreen.cpp:1171` 写 3、`Supervisor.cpp:1445` 读它做一次范围校验（`>= 4` 判脏）。**没有一处参与资源初始化。**
- 同一逻辑在 `Player.cpp:1401`（重生时重置炸弹）、`GameManager.cpp:783`、`AsciiManager.cpp:1475` 重复出现。

⇒ 主回执 §6.1 里标"待核验"的 `bombs_remaining=3` 现在确定：**它是 Border Team 射击文件的 `initialBombCount`，与 `cfg.bombCount=3` 的数值相同纯属巧合**。前一轮"无法区分"的原因是二者恰好都是 3；源码层面并不存在歧义。

---

## 4 13,118 → 13,195 的 77 帧：机理与一条被否定的假设

### 4.1 从既有产物重放出的逐帧时间线（无引擎）

用 `teacher/decisions.jsonl` 的 `segments`（每段真实 `mask`/`advanced`）重建逐帧序列：

| 量 | 重建值 | `result.json` | 一致 |
| --- | ---: | ---: | --- |
| 受控帧 | **13,195** | 13,195 | ✅ |
| SHOOT 上升沿（要求前一帧无 SHOOT） | 137 | 138 | 差 1，见 §4.4 |

关键帧：

| 事件 | 帧 |
| --- | ---: |
| `EVENTS.stageClear` 首次为 1（= `stageClearScreenState != 0`） | **13,118** |
| 该波内最后一次输入上升沿 | **13,153**（恰在 13,118 之后 **35** 帧） |
| `STATE.stage` 首次前进 0→1 | **13,195** |
| 过关屏 → 前进 | **77 帧** |
| 边沿 → 前进 | **42 帧** |

**`[13,118, 13,195]` 区间内只有一次上升沿（13,153）**。

### 4.2 源码路径（两条不同的路，不能混为一谈）

- 电平位来源：`rlharness.cpp:107` `StageClearLevel()` = `g_Gui.impl->stageClearScreenState != 0`；该字段由 **`Gui.cpp:687-688`** 在 `GUI_MSG_SHOW_STAGE_RESULTS` 里置 `stageClearScreenState = 1` 并 `stageClearSequenceActive = 1`。⇒ **13,118 是过关屏出现的帧。**
- 阶段前进来源：**`GameManager.cpp:357-359`**
  ```cpp
  g_GameManager.AdvanceToNextStage();                        // :358
  g_Supervisor.curState = SupervisorState_GameManagerReInit; // :359
  ```
  这条在"非 replay 且非 6A/6B/Extra"分支里**直接执行**，并在 :331-345 先查 `clockTime >= 12`（那是"全部通关"的另一条路，本局 `clockTime=0`）。
- 而 `shoot.py` 引用的那条 `(g_CurFrameInput & 0x1001)` 上升沿判据在 **`GameManager.cpp:388-396`**，它清的是 `STAGE_TRANSITION_STATE_MASK` 并跳 `nextSupervisorState`——**与 :358 的 `AdvanceToNextStage()` 是两条不同的路**。

### 4.3 被实测否定的假设

`touhou_rl/v400/teacher/shoot.py` 的注释断言"游戏只在 `SHOOT|ENTER` 上升沿离开过关态"。若该断言对**本局这条**路径成立，前进帧应当等于边沿帧 **13,153**；实测是 **13,195**，**晚 42 帧**。⇒ **"过关帧 = 输入边沿帧"不成立**（至少对本局、本路径不成立）。

**但脉冲本身是否仍必需，本补遗不否定**：T08 的负对照（`period=1,000,000,007`，永不释放）确实得不到 `stage_clear`，所以"某个环节需要上升沿"有经验支撑；源码里 :388 那条路仍需边沿。**两条路各管什么，本补遗没有逐步追踪。**

### 4.4 顺带钉死一个计数口径

`result.shoot.shoot_rising_edges = 138` 而逐帧重建是 137，差 1 的来源：仪器在**每段 STEP**上计 `if shoot and not previous_shoot`，且 `previous_shoot` 初值为 0，于是**第一段（帧 1，mask 0x0041 含 SHOOT）也算一次上升沿**；逐帧重建要求"前一帧无 SHOOT"，因此不计首帧。**两个数都对，只是定义不同**，后续比较时不可混用。

### 4.5 要彻底裁定，最便宜的一次实验

**只改脉冲相位、不换任何别的东西**：同一磁带跑 `shoot_pulse_period = 24`（上升沿落在不同帧），观察 13,195 是跟着边沿走（⇒ 前进 = 边沿 + 固定延迟）还是跟着过关屏走（⇒ 前进 = 过关屏 + 77 的固定序列）。一次教师轨迹即可裁定，**但需要研究端给预算**——本补遗不动用引擎。

---

## 5 两处工单缺陷的可直接套用补丁

### 5.1 pytest 目标行（`TRAINING_HOST_REQUEST_01.md:70-73`）

把**一行**拆成两行并按解释器分段（文件一个不丢）：

```bash
# WSL python 3.10：能收集的两个 + 新文件
"$V831_PY" -B -m pytest -p no:cacheprovider --basetemp "$V831_RUN/pytest" -q \
  tests/test_v8_31_slice.py tests/test_v400_obs_v3.py

# Windows python 3.12：三个只有 Windows 能收集的文件（在本机单独执行并回传退出码）
#   python -m pytest -p no:cacheprovider -q \
#     touhou_rl/tests/test_framefence.py touhou_rl/tests/test_frame_proof.py \
#     touhou_rl/tests/test_phase2_history_proof.py
```

理由：`touhou_rl/memory.py:23` 在 import 期执行 `ctypes.WinDLL("kernel32")`，WSL 下**连收集都失败**（实测 rc 2），会把 `set -e` 的脚本卡在引擎启动之前。

### 5.2 像素帧关系句（`TRAINING_HOST_REQUEST_01.md:40` 与 `PIXEL_SLICE_PLAN`）

原句隐含 `header.frame == EVENTS.frames − 1`。按 rl_v3 实测（`rlharness.cpp:1362` + `main.cpp:143-146` + `shm.py` 契约）应为：

> **STEP 返回时 `header.frame == EVENTS.frames`（该 STEP 自己的 `PublishState` 写的自增值）；共享块里的 RGBA 字节属于 `EVENTS.frames − 1` 那一帧。判定用字节帧，标签只作 `{F−1, F}` 的二值分类；任何第三值 fail-closed。**

（`v400_stage_clear_replay.py:547` 的 `− 1` 属于更早的 `rl_events` 构建，不可直接搬用。）

---

## 6 分类声明

- **训练端本次离线检查**（无引擎、无新资产）：§1 全部文件身份与两侧掩码、§2 八个 shard 的全量配对复算、§3 源码引用、§4.1 逐帧时间线重建（13,195/137 与产物自洽）、§4.2 源码行号。
- **历史记录**：R74 的"目标段 8,664"、T08 负对照、`shoot.py` 的上升沿判据、`configs/v420` 的 `shoot_pulse_period=48`。
- **待核验**：§2.3 那条改法能否真的让前缀跑到 8,664 且状态仍等于录像状态；§4.5 的相位实验；§4.2 两条路各自服务于哪一类过关。
- **不主张**：本补遗不主张 A5 已解阻、不主张脉冲可去掉、不主张 R74 的目标段帧号、不主张 `stage_clear` 可精确到单帧、不主张任何像素或策略能力。

---

## 7 建议的下一步（按代价从低到高）

| # | 动作 | 代价 | 谁做 |
| ---: | --- | --- | --- |
| 1 | 接受 §3、§4.1、§4.4 三个口径钉死，更新 A 卡的 A7 与"资源/终点"表述 | 零 | 研究端 |
| 2 | 就 §2.3 的 `_pair_cand` 改法**立项**（新工单 + 白名单），并用**一次**真机前缀重放验证到 8,664 | 1 次引擎 + 代码改动 | 研究端立项 → 训练端执行 |
| 3 | §4.5 的相位实验（`period=24` 一次教师轨迹）裁定"过关帧"的依赖 | 1 次引擎（≈8.5 分钟） | 研究端给预算 |
| 4 | 把 §5 的两处补丁合进下一张工单模板 | 零 | 研究端 |
| 5 | A3（截断见证）、A8（开发根）仍按主回执口径待决 | —— | 研究端 |
