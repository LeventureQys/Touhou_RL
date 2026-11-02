# v3.7.1 P0 预注册：实验有效性修复

- 日期：2026-10-26
- 状态：**运行前冻结**（冻结对象：`configs/p0-clock-v1.json` 与本文）
- 范围：只修实验本身（时钟、输入生效、等逻辑帧曝光、顺序、记录）。**不涉及**视界/beam 调整、网络扩大、学生重训。
- 对应门禁：C-01、C-02、C-03、C-04、C-05（冻结部分）、C-06（回归）

## 1. 为什么必须先做 P0

v3.7.0 确认集用 `frames // 2 = 900` 次决策终止，而不是 `gameplay_frame >= 1800`：

| 证据 | 值 |
|---|---|
| pair 1 subject frame span | `[2, 1845]` |
| pair 2 subject frame span | `[2, 1854]` |
| baseline frame span | **未记录** |

subject 每决策规划约 14 ms，规划期间游戏继续推进、旧动作保持，因此：

1. subject 实际经历的逻辑帧比名义窗口多 45–54 帧；
2. 每个动作的实际保持时长 = `frameskip + 规划耗时`，随策略不同而不同；
3. baseline 没有任何等价曝光记录，两臂的"同样 1800 帧"是假设而非测量。

在这些数字修好之前，任何 `+29%` 或 `-29%` 的点估计都不能归因于算法。

## 2. 修复定义（可在无游戏环境下验证）

### 2.1 时钟

- 唯一终止依据：`gameplay_frame >= start_frame + frames`（`frames = 1800`）。
- 决策数只是派生量（`planned_slots = frames // frameskip`），永不作终止条件。
- 帧回退（`frame < last`）分类为 `rollback`：开始新 epoch、该局判技术失败、曝光夹在 `[0, frames]`，**不允许负曝光**。
- 帧停滞超过 `stall_s = 3.0` 秒：`stall`，判技术失败，不伪装成"窗口跑完"。
- 读到 `supervisor_state != 2`：`left_game`，判技术失败。
- 等待超时：`timeout`，`reached=False`，**不得**当作已到达目标帧。
- 帧跳跃超过 `jump_threshold = 12`：记录 `jump` 及其 delta（允许发生，但必须可见）。

### 2.2 曝光相等

动作调度是固定栅格：

```text
决策 k：观测帧 O_k = start + k*frameskip
        写键帧 S_k = O_k + actuation_lag（默认 1）
        动作 k 曝光 [S_k, S_{k+1})，长度恒为 frameskip
```

因此**规划耗时不再改变单个动作的曝光长度**：

- 规划在 `actuation_lag` 帧内完成 → `hold_frames = frameskip`；
- 规划超时 → 记录 `send_late_frames > 0`、`hold_frames > frameskip`，是**被测量**的劣化，而不是静默的曝光不等；
- 窗口总长度始终是 1800 帧（`exposure_frames` 由首末帧计算，不由决策数计算）。

`actuation_lag = 0` 保留为诊断模式，用于复现 v3.7.0 的混杂（差异必须可见）。

### 2.3 输入生效

- `send_frame`：写键前后各读一次 `gameplay_frame`；两次不同则记录区间而不是点。
- `effective_frame`：逐帧读引擎 `g_CurFrameInput`（`GameState.input_mask`），要求
  1. 出现过"未匹配"帧在前，且
  2. 匹配连续 ≥ 2 帧（settling）。
- 无法解析时输出 `status = unknown_interval` 且 `effective_frame = null`；若首次观测即已匹配，输出 `already_active`。
- 只有在"动作确实改变了按键集合"的决策上统计生效解析率（`changed_resolution_rate`），避免重复动作把比率虚高。

### 2.4 顺序与 RNG

- 每对按 `balanced_random`（种子固定）决定 AB/BA，顺序在**运行前**写入记录。
- TH08 RNG 不由 reader 暴露 → 记录 `rng_control = "unavailable"`，并额外记录每个 reset 的
  `start_frame / stage / deaths / bombs / initial_pool`（弹池指纹）。
  **不得**把"同一 session"写成"同一初始弹幕"。

### 2.5 事件归属

- 死亡通过累计计数器 `deaths` 的增量判定，归属区间为 `(上一观测帧, 本帧]`。
- 区间完全落在窗口内 → `inside`，计入主指标。
- 区间跨越窗口边界 → `boundary_unknown`，**不计入**死亡数，但事件保留在 `death_events` 中并计入 `boundary_unknown` 计数。
- 计数器回退（新局/续关）→ `counter_reset` 事件，不得算成负死亡。

## 3. 通过条件

| 门禁 | 通过条件 |
|---|---|
| C-01 | 规划耗时 0/1/3/10 帧时 `exposure_frames` 完全相同且 `= frames`；回退/停滞/离开游戏均判技术失败且曝光 ≥ 0 |
| C-01c | 同一动作序列经同一管线两次回放，逐帧位置完全相同（假世界为硬条件；真机只报告散度） |
| C-02 | 两臂 `window`、`end_frame`、`exposure_frames`、`hold_frames` 分布完全相同 |
| C-02b | 窗口内死亡按帧归属；跨界标 `boundary_unknown` 且不计入 |
| C-03 | `send_frame != effective_frame` 可测；变更动作的解析率非空；不可解析时为区间 |
| C-04 | 注入 0/5/15/30/50 ms 延迟时曝光帧不变，只有迟到帧数与 hold 变化 |

任一不满足 → P0 未完成，**不得**进入 P1 的解释阶段。

## 4. 已知局限（预先声明）

1. 假时钟门禁只证明"时钟/终止/归属逻辑正确"，**不证明**真机输入延迟。真机输入延迟需 C-03-live 的独立运行。
2. 真机 RNG 不可固定，因此逐帧一致性不是真机硬条件；真机只能报告散度与重复运行。
3. `effective_frame` 是"引擎消费该输入的第一帧"，不是"按键物理到达"的时刻；两者之间的差值不可测，报告不得混用。
4. `frame_gaps` 记录未观测帧数；跳跃越大，事件归属区间越宽，`boundary_unknown` 越可能出现。这是诚实代价，不是缺陷。

## 5. 运行命令

```text
python -m touhou_rl.escape experiment gates \
  --out Document/Update/Alpha/v3.7.1/data/c01-c04-gates.json
python tools/rl/v371_recompute.py report \
  --data Document/Update/Alpha/v3.7.1/data \
  --out Document/Update/Alpha/v3.7.1/reports/P0-实验有效性审计.md
python tools/rl/v371_manifest.py \
  --out Document/Update/Alpha/v3.7.1/evidence/manifest.json
```
