# v3.7.1 P1 预注册：最小公平单因素对照

- 日期：2026-10-26
- 状态：**运行前冻结**（冻结对象：`configs/p1-fair-v1.json` 与本文）
- 前置条件：P0 门禁 C-01…C-04 全部 pass（见 `preregistration/p0-clock.md`）
- 本文件只定义 **P1**；P2 的逐项消融见 `p2-ablation-v1.json`，P3 的正式确认集必须在 P1 结果出来之后再预注册。

## 1. P1 要回答的问题

> 在**同一个观测/输入管线、同一逻辑帧窗口、同一 frameskip、同一视界、同一 bomb 协议**下，
> 只把 baseline 的局部控制换成"baseline + 一个宏观路线 veto"，死亡数会怎么变？

v3.7.0 的回答不可用，因为它同时改动了七项（规划器、视界、beam、排名、bomb、执行管线、记录）。P1 把它压到**一项**。

## 2. 两臂定义

| 项目 | baseline | hybrid_veto |
|---|---|---|
| 局部策略 | `AvoidPilot(PilotConfig())` | **同一份** `AvoidPilot(PilotConfig())` |
| 视界 | `horizon_steps=10` × frameskip 2 = **20 帧** | `horizon_decisions=10` × frameskip 2 = **20 帧** |
| beam / continuation | `quota_per_action=2` | `beam_width=2`，`continuation_decisions=0` |
| trajectory 排名 | pilot 自己的 `bottom_bias=0.3`、`switch_penalty=0.05` | **不变**：只有 veto 生效的那一步才由控制器给 move/focus |
| bomb 协议 | pilot 原生（`bomb_min_clearance=-1.5`，`bomb_cooldown=40`） | **完全不变**：veto 只覆盖 move 与 focus，`shoot`、`bomb` 两位沿用 baseline |
| frameskip | 2 | 2 |
| 观测/输入管线 | `escape/harness.py` 同一循环 | 同一循环 |
| 窗口 | `[start_frame, start_frame+1800)` | 同 |

**唯一差异**：当逃逸控制器给出 `intent ∈ {evacuate, emergency}` 且**找到路线**（`route_id != null`）时，
用控制器的 `(move, focus)` 替换 pilot 的，并保留 pilot 的 `shoot / bomb`。

- veto 触发率会被逐决策记录（`intent`、`route_id`、`reason`），报告必须给出触发次数与覆盖率。
- 若 veto 触发率为 0，本实验**不是**"路线能力无效"的证据，而是"在 practice stage 0 / Lunatic 的 1800 帧窗口里该 veto 从未达到高置信条件"，必须如实这样写。

## 3. 样本量与功效（预先算出，不事后补）

用 v3.7.0 确认集实测的配对差标准差作为规划假设：`sd = 1.442` 死/局，baseline 均值 `1.0333`。
命令与输出可复现：

```text
python tools/rl/v371_recompute.py power --sd 1.442 --baseline 1.0333
```

| 目标相对变化 | 绝对效应（死/局） | 所需对数（80% 功效） |
|---|---:|---:|
| 10% | 0.103 | 1529 |
| 15% | 0.155 | 680 |
| 20% | 0.207 | 383 |
| 30% | 0.310 | 170 |
| 40% | 0.413 | 96 |
| 50% | 0.517 | 62 |

结论（**必须在报告里重复**）：

- **n = 30 只能检出约 71% 的相对变化**（MDE = 0.738 死/局）。C-07 的 "≥20% 相对下降" 在 n=30 下**不可能成立**，除非真实效应远大于 20%。
- 因此 P1 的 30 对**不是**确认检验，而是**估计量**：它的用途是给出新协议下的 `sd`，用来给 P3 定样本量。
- 若新协议的 `sd` 明显下降（等曝光应当降低一部分方差），P3 的 n 由 `power` 命令用新 `sd` 重算后写进 P3 预注册。**不得**在看到 P1 的均值后倒推样本量。

## 4. 指标

主指标（唯一判定用）：

- `deaths_in_window`：`[start_frame, start_frame+1800)` 内的死亡数，按帧归属；跨界死亡标 `boundary_unknown` 且不计入。

次要指标（只报告，不判定）：

- 首次死亡帧、后续死亡帧；
- `bombs_in_window`；
- 实际曝光帧（`exposure_frames`）、`hold_frames` 分布、`send_late_frames`；
- `plan_ms` p50/p95、`deadline_miss`；
- 生效输入解析率（仅统计按键集合发生变化的决策）；
- veto 触发次数与触发后动作改变率。

L1（局部）指标**不得**替代 L2（整局）：v3.7.0 的 24 帧安全率 subject 92% / baseline 75% 而整局死亡更多，正是尺度错配的例子。

## 5. 排除与失败规则（运行前固定）

| 情形 | 处理 |
|---|---|
| 时钟回退 `rollback` | 该局技术失败；曝光夹在 `[0, 1800]`，不产生负值 |
| 帧停滞 `stall`、离开游戏 `left_game`、等待超时 `timeout` | 技术失败 |
| 快照连续不稳定（超过 `max(8, slots/4)` 次） | 技术失败 |
| 单局墙钟超过 120 s | 该局标 `wall_budget` |
| 死亡归属区间跨窗口边界 | 事件保留，计入 `boundary_unknown`，**不计入**死亡数 |
| 技术失败的对 | 从死亡均值中剔除，但**必须在报告里逐对列出**；若失败对数 > 3，该次运行作废 |
| 任何"看起来异常"的对 | 不得事后剔除；只能作为敏感性分析另列 |

## 6. 顺序与 RNG

- 每对的臂顺序由 `balanced_random`（`seed=0`）在**运行前**决定并写入数据文件顶部。
- TH08 RNG 不由 reader 暴露 → 记录 `rng_control = "unavailable"`。
- 每个 reset 记录 `start_frame / stage / deaths / bombs / initial_pool`（弹池指纹）。
- **禁止**把"同一 session"写成"同一初始弹幕"。若两臂的 `initial_pool` 指纹不同，报告必须说明配对是在"同一时间邻近的两次 attempt"意义上成立，而不是同一世界。

## 7. 判定

P1 不做出 "改善/恶化" 的确认判定，只输出：

1. 点估计与配对 95% 区间；
2. 新协议下的配对差 `sd`；
3. 两臂实际曝光是否相等（应完全相同）；
4. veto 触发率；
5. 明确的 `underpowered` / `not established` 声明（若区间跨 0）。

只有 P3 的确认集（样本量按第 3 节方法用 P1 的 `sd` 重新计算）才可以引用 C-07 的 20% 门槛。

## 8. 运行命令

```text
python -m touhou_rl.escape experiment pair \
  --pairs 30 --order balanced_random --seed 0 \
  --frames 1800 --frameskip 2 --actuation-lag 1 \
  --horizon 10 --continuation 0 --beam 2 \
  --difficulty 3 --stage 0 \
  --confirm-foreground \
  --out Document/Update/Alpha/v3.7.1/data/live-pairs.json
```

重算：

```text
python tools/rl/v371_recompute.py pairs \
  --data Document/Update/Alpha/v3.7.1/data/live-pairs.json
```

## 9. 如果真机不可用

若 TH08 无法启动或被会话禁止前台输入：

- P1 **不得**用合成场景替代（`escape/evaluate.py` 的 synthetic world 没有 bomb，碰撞即终止，见 README §5）；
- 交付物应停在 P0：`configs/`、`preregistration/`、`evidence/`、假时钟门禁数据和 P0 报告；
- P1 诚实状态是 `blocked_on_live_access`，并附上尝试记录（命令、退出码、错误）。
