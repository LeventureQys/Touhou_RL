# v3.5.0 S0：P 点契约受控实验（预注册）

- 日期：2026-10-24
- 依据：[状态与事件契约](<./状态与事件契约.md>) §4–§5、[执行计划](<../执行计划.md>) §3.2
- 性质：**预注册实验协议**。阈值、场景、覆盖规则、样本量与失败处理在此写定，跑完后不得修改。
- 目的：在真机上确认 `power`/自动收集线/Shift/收集动作的六个假设（H-P1…H-P6），为 S3 的 `CollectPower` 提供可执行契约，而不是照抄社区说法。

---

## 0. 预注册修订（2026-10-24，正式采集之前）

这些修订发生在**任何正式 S0/P 契约数据之前**，依据是 1,200 决策的工程 smoke（只用来暴露量具问题，不作为证据）。阈值、判据方向与“失败即停止”均未改变，改的是“用哪个量具测哪条假设”。

| # | 原写法 | 修订后 | 为什么 |
|---|---|---|---|
| R1 | 只跑 teacher + 上移覆盖 | 增加 **replay 臂**：播放一份人类 Lunatic 录像，记录同一 trace | teacher + 覆盖在 2,400 帧内死亡 2 次、power 最高 37，无法观测 H-P1 的 power=128；人类录像会自然收满 P，且不依赖我们的策略 |
| R2 | H-P3 用“首次 state 0→1” | H-P3 只用 **受控臂** 的首次 transition | 人类可能先不按 Shift 越过 128，之后再聚焦；此时首次 transition 的 y 必然远离线，不能用来检验“线在哪” |
| R3 | H-P5 对所有 crossing | H-P5 只用 **触发条件成立** 的 crossing（`focus_mode!=0` 或 `power>=128`） | `ItemManager.cpp:267-271` 的自动收集本身就有 focus/power 条件；不满足条件的 crossing 本来就不应触发 |
| R4 | H-P6 只允许“item 消失”解释 power 增 | 允许 `ReplayManager.frameEventFlags & 0x40`（ITEM_COLLECTED）解释 blind window 内的收集 | frameskip=2 时，P 可能在同一决策的两帧内生成并被收集，前一条快照里看不到它；单靠槽位消失会把合法收集误判为违约 |
| R5 | 单次读帧事件标志 | 在 `env._wait_frames` 的每次轮询中 OR 累积 `frameEventFlags`，并把 H-P6 的归属窗口放宽到 **前后各 1 个决策** | 帧标志按“模拟帧”清零，power 是等待结束后读的；一次收集可能落在决策边界另一侧。累积 + ±1 窗口是“同一事件不被采样边界切断”的最小修正，不改变 H-P6 的方向（power 增必须有收集证据） |
| R6 | H-P3 要求 `|y-line| ≤ 2 px` | 改为“线落在 `[cur_y, prev_y]` 区间内，且 `|y-line| ≤ 4.5 px`” | frameskip=2、聚焦速度 2 px/帧 → 每个决策走 4 px；首次采样到线下的 y 最多比线低 4 px，±2 是 1 帧分辨率下的阈值，不适用于 2 帧决策 |
| R7 | H-P5 对所有触发 crossing 计漏检 | 排除 crossing 时 `player_state ∉ {0,3}` 的行，并单独报 `excluded_player_dying` | 源码 `ItemManager.cpp:273` 明确在 DYING/SPAWNING 时不自动收集；在死亡瞬间越过线而漏收不是契约失败 |

R4 的 `frameEventFlags` 读取仍是只读（`g_ReplayManager` +0xDA 的 u16），不改变游戏行为；它只是把“看不见的收集”变成可复核的事件。R5 的累积只影响探针（`EnvConfig.expose_frame_events`），默认关闭，不改变既有训练/评估。R6/R7 只修正“采样分辨率/合法例外”，不改变 H-P3“线在 128”、H-P5“触发后全部 P 进入 auto-collect”的方向。

---

## 1. 为什么需要单独做这个实验

v3.5.0 研究方案把“P 点是条件性资源投资”作为 H3；计划 §3.2 要求先确认：

1. `power` 最大值；
2. P 槽未满判据；
3. 25% 收集线的真实 y；
4. Shift 是按住还是按下沿；
5. 25% 线是否收集屏幕内全部 P；
6. 收集前后 item 状态与 power 增量。

这些不是“读一个标量”就能回答的：`pointItemValueLine` 是 shot 文件里的字段，`power<128` 还受 `focusMode`/`shotType` 影响，`state` 从 0→1 是逐帧逻辑。必须用受控场景同时看 **自机 y、item state、power、focus_mode** 四路信号。

源码依据（只读，不改第三方）：

- `ItemManager.cpp:267-280`：`state==AUTOCOLLECT || (player.y < pointItemValueLine && (power>=128 || focusMode!=UNFOCUSED || shotType==1||6))` → 所有 item 进入 `AUTOCOLLECT`；
- `ItemManager.cpp:403-454`：小 P `+1`，`power>=0x80` 时 `SetPower(0x80)` 并转换剩余 power item；
- `Player.cpp:637-752`：`focus = g_GuiMessageInputCurrent & 4`，按住时每帧 `focusMode=FOCUSED`；
- `Player.hpp:22-56`：`pointItemValueLine` 在 `PlayerRawShtFile + 0x1C`。

---

## 2. 场景与工具

| 项 | 值 |
|---|---|
| 游戏 | TH08 1.00d 重建 bugfix 镜像（`third_party/th08/build/play`） |
| 难度 / 队伍 | Lunatic / Border Team（team 0，shotType 0） |
| 模式 | 练习 Stage 1（`practice_stage=0`），不是 replay |
| 决策 | `frameskip=2`，每决策读状态/敌人/道具/shot 参数 |
| 采集 | `v350-trace/1` 逐决策 JSONL + `bullets/*.npz` |
| 控制 | `AvoidPilot` + 受控上移覆盖（见 §3） |
| episode 上限 | 每局 `max_steps=3000`（6000 帧），或自然终止 |
| 重试 | 最多 3 局；仍未 crossing → H-P3 `not_tested`，S0 fail |

### 2.1 两个臂

| 臂 | 场景 | 用途 | 样本 |
|---|---|---|---|
| A 受控臂 | Lunatic / Border Team / 练习 Stage 1；`AvoidPilot` + 上移覆盖 | H-P3（线位置）、H-P4（Shift 按住）、H-P5（全部 P 进入 auto-collect） | 2 局 × 3000 决策 |
| B replay 臂 | 同一游戏，`th08_replay/Th8_ud2558.rpy`（Border Team Lunatic）播放 | H-P1（power 上限）、H-P2（满后不涨）、H-P5/P6（真实收集规模） | 1 局 × 4500 决策 |

replay 臂不发送任何输入，只读回放输入与状态；它的 `action` 由 `g_GuiMessageInputCurrent` 解码，仅用于记录，不参与控制。两个臂写入同一个 `probe/p-contract/` 目录，`episodes.jsonl` 用 `mode` 区分（`p-contract` / `replay`）。

### 2.2 控制覆盖（只在 A 臂启用）

每个决策：

```text
items = read_items(); power = state.power; y = state.player_y
p_exists = 任一 item.type in {0,2,4} 且 isInUse
if power < 128 and p_exists and y > 110:
    action = [move=1 (N), shoot=1, focus=1, bomb=0]   # 上移 + 聚焦
elif y <= 110:
    action = [0, 1, 1, 0]                              # 停在线上方继续聚焦
else:
    action = AvoidPilot.act(...)                       # 默认局部躲避
```

- 覆盖只改变移动/聚焦，不改变 shoot；不要求这个策略活到结束，只要求 crossing；
- 若检测到 `best_clearance < 2 px`，放弃本决策覆盖，改用教师动作（安全优先）；
- 覆盖事件写入 trace 的 `controller.override` 字段，保证可复核。

### 2.3 真机输入前置

- 游戏窗口必须能接收键盘（S0 发现：管理员权限置顶的任务管理器会让 `SetForegroundWindow`/`SendInput` 全部失败；这是环境问题，不是契约问题）；
- 探针启动时打印 `foreground == game_hwnd` 检查；若 false，直接失败并提示，不静默采空数据。

---

## 3. 观测与记录

每决策记录（除契约 §7.1 的公共字段外）：

```json
"p_contract": {
  "line": 128.0,
  "auto_speed": 10.0,
  "box": 24.0,
  "power": 12.0,
  "focus_mode": 1,
  "player_y": 120.0,
  "p_items": [{"slot": 0, "type": 0, "x": 100.0, "y": 300.0, "state": 0}],
  "state_transitions": [{"slot": 0, "from": 0, "to": 1}],
  "collected": [{"slot": 0, "type": 0, "power_delta": 1}]
}
```

消失判定：

- 上一决策 `isInUse`，本决策不 `isInUse` → 消失；
- 若消失时 `state==AUTOCOLLECT` 或 `player_y < line + 8` → 记为 `collected`，并记录同窗口 `power` 增量；
- 若消失时 `state==default` 且 `y > PLAYFIELD_Y+PLAYFIELD_H+16` → 记为 `offscreen`，不算收集。

---

## 4. 判定表（预注册）

| 编号 | 通过标准 | 数据来源 |
|---|---|---|
| H-P1 | 至少 1 次 `power == 128.0`；全部记录 `power <= 128.0`；满后不再出现正增长 | trace `resources.power`（两臂合并） |
| H-P2 | 满 power 后不再出现正 power 增量；`power<128` 时允许 | `power` 差分（两臂合并） |
| H-P3 | **A 臂**首次 `P item.state 0→1` 的决策：`line` 落在 `[cur_y, prev_y]` 内，且 `|cur_y-line| ≤ 4.5 px`（frameskip=2 的 4 px 步长 + 0.5 余量）；`line` 落在 120–136 | 受控臂 `state_transitions` |
| H-P4 | 按住 Shift 的连续 ≥10 决策 `focus_mode=1`；松开后 2 决策内 `focus_mode=0` | `focus_mode`（两臂合并） |
| H-P5 | **触发条件成立**（focus 或 power=128）且 `player_state∈{0,3}` 的 crossing 决策，所有 in-use P（type∈{0,2,4}）在 ≤2 决策内进入 `AUTOCOLLECT`；漏检 ≤1 | `state_transitions` + `items`（两臂合并） |
| H-P6 | 每个窗口 `power_delta` ≥0；正增量必须有消失的 P 或 `frameEventFlags & 0x40` 解释；不存在“power 增但两者都无” | `collected` + `power` + `frame_events`（两臂合并） |

### 4.1 关于“大 P 的增量”

`CollectPowerBig` 的具体增量在源码里不是 1；实验不预设，只检查“同一窗口 power 增量 / 消失 P 数”在两次 crossing 间一致且可解释。若大 P 的增量无法确定，H-P6 只对小 P 判定，并在报告里写明“大 P 未闭合”。

### 4.2 失败处理

| 失败 | 处理 |
|---|---|
| H-P1/P2 fail | P 资源契约无效，S3 `CollectPower` 不启动；先修 `power` 读/写口径 |
| H-P3 fail | 检查 `primaryShtFile` 偏移或 shotType 假设；不调动作权重 |
| H-P4 fail | 输入契约无效，先修 Shift 按住语义；禁止用“按下沿”糊过去 |
| H-P5 fail | 记录反例（哪个 item 没进入 auto-collect、当时 y/state），S3 收集条件暂缓 |
| H-P6 fail | 收集事件口径无效，S0 总判定 fail |

---

## 5. 样本与切分

- 正式记录：`probe/p-contract/episodes.jsonl` + `trace/*.jsonl`；
- A 臂 2 局 × 3000 决策；B 臂 1 局 × 4500 决策（6000 帧不足，smoke 实测 6000 帧只有 power 103）；**若 B 臂功率仍未到 128，允许把 B 臂延长到 6000 决策一次**，不改判据；
- 每次 crossing 的 episode 独立计；同一 episode 的多次 crossing 可以合并证据，但 H-P3 以 A 臂“首次 transition”计；
- 全部 episode 参与 S0-5 的整局切分检查（同一 episode 不跨 train/validation）。

---

## 6. 复现命令（实现后补路径，不改变判据）

```powershell
# A 臂：受控收集
python -m touhou_rl.probe35 p-contract `
  --out Document/Update/Alpha/v3.5.0/probe/p-contract `
  --run-dir .tmp/v350-p-contract-session `
  --episodes 2 --max-steps 3000 --frameskip 2

# B 臂：人类录像补充（追加到同一目录）
python -m touhou_rl.probe35 p-contract --append `
  --out Document/Update/Alpha/v3.5.0/probe/p-contract `
  --run-dir .tmp/v350-p-replay-session `
  --episodes 1 --max-steps 4500 --frameskip 2 `
  --replay th08_replay/Th8_ud2558.rpy
```

产物：

```text
probe/p-contract/
  episodes.jsonl
  trace/p-contract-ep01.jsonl / replay-ep01.jsonl
  bullets/p-contract-ep01.npz / replay-ep01.npz
  p-contract.json          ← H-P1…H-P6 判定
```

---

## 7. 不做

- 不为了“收集率好看”而修改覆盖规则；
- 不把 25% 线的社区值（112/128/其他）直接写进代码而不读 `pointItemValueLine`；
- 不在真机输入不可用时用合成数据冒充 crossing；
- 不在本实验里训练任何策略。
