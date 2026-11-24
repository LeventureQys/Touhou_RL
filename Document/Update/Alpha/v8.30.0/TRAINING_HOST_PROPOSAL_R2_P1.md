# R2 / P1 像素接线提案（训练端草稿，**待研究端批准**）

**性质**：提案，不是实施单，不启动引擎。
**目标**：把"一条真实像素 → actor → 动作 → 实际帧 → 独立终态"的完整轨迹跑通并留下可核证据（`PIXEL_SLICE_PLAN` 的 **P1**）。
**不包含**：学习、选模、泛化、任何能力声明。P1 通过只说明"接线可运行"，**不说明视觉策略有效**。
**依赖**：`TRAINING_HOST_PROPOSAL_R1.md` 的 D1（任务切片）、D2（脉冲）、D6（burn-in）、D7（静音）。若 D1 选 (B) 重录磁带，本提案随之推迟。

---

## 决策清单（请逐项勾选）

| 编号 | 决策 | 训练端建议 | 理由 |
| --- | --- | --- | --- |
| **E1** | 观测契约 | **`obs-v3`**：play `(4,112,96)` + hud `(1,56,160)` + act_hist `(8,18)`，uint8/255；**必须把 digest `ef2fbd15…` 写进产物** | 已冻结、边界合规（`SELF_STATE_PRESENT=False`、`DANGER_SUMMARY_PRESENT=False`），不必新造管线 |
| **E2** | 每条轨迹受控帧 | **600**（前缀重放、预热、恢复核验**另计**，不得藏进 600） | `PIXEL_SLICE_PLAN §2.3` 的上限；成本可测 |
| **E3** | 帧栈 burn-in | **重放 3 帧**填 `t-3..t`，`act_hist` 前置全零行；`burn_in_frames=3` 入身份 | 与 R1-D6 一致；**这是设计值不是既有契约** |
| **E4** | 冒烟用的"策略" | **固定策略**：事前冻结的合法动作序列（或恒动作 #12），**不使用任何学习权重、不读教师 mask** | P1 只证接线；固定策略能发键**不**证明视觉有效 |
| **E5** | 输入隔离检查 | **两层**：① 静态列出 actor 实际调用与每个张量来源；② 离线保持像素与实际动作不变、扰动裁判的资源/死亡/阶段字段，核对 **actor 输入摘要、记忆、输出三者不变** | `PIXEL_SLICE_PLAN §4 P0` 的原文要求；只验证隔离，不替代真实轨迹 |
| **E6** | 终态判决 | **`EVENTS` `stageClear` + `stageClearEdgeFrame` 为主，`STATE.stage` 计数交叉校验**（同 R1-D3） | 两口径同时记录才能独立重算 |

---

## E1 输入边界的落地（对着 `obs-v3` 逐字段说清）

| obs-v3 分量 | 来源 | 判定 |
| --- | --- | --- |
| play `(4,112,96)` | 640×480 读回裁 `x=32..416, y=16..464`（384×448）后 4×4 块降采样，取 `t-3..t` | **框内像素** —— 允许 |
| hud `(1,56,160)` | 读回右栏 `x=480..640` 的三条行带 `((48,72),(80,96),(96,112))` | **HUD 像素** —— 允许（路线允许 actor 读 HUD 可见资源像素） |
| act_hist `(8,18)` | **自己实际执行**的动作，oldest-first，全零填充 | **自身执行历史** —— 允许 |

**明确禁止 actor 读取**：引擎玩家坐标、弹幕/激光/敌机真值、`livesRemaining`/`bombsRemaining`/`playerPower` 数值、`stage`、命中/死亡事件、真实 `done`、**教师 `verified_mask`**、裁判结果。这些只能进隔离审计侧。

## E5 的两层实现（可直接照做）

**第 1 层（静态）**：产出 `actor_inputs.json`，逐项列出 `tensor → shape → dtype → 来源函数:行号`，并断言每个来源都属于 {读回像素, 自身动作历史}。

**第 2 层（离线扰动）**：固定同一组像素与动作历史，只改裁判侧的 `lives/bombs/power/stage/death` 字段，重算 actor 输入摘要与输出；要求**逐位不变**。任一项变化即判"隔离失败"，停止 P1。

## E2/E3 一局的帧与数据预算

| 项 | 值 |
| --- | --- |
| 受控帧 | ≤ 600（另计前缀重放与预热） |
| 决策数（2 帧/决策） | ≈ 300 |
| 数据量 | 300 × 52,112 B ≈ **15.6 MB**/轨迹（`BYTES_PER_DECISION`） |
| 目标墙钟 | 由首批实测填写，**本提案不外推** |

## E6 每条轨迹必须留下的字段

```text
run_id, ancestor_id, seed, stage, difficulty, shot_type,
shoot_pulse_period, decision_frames, burn_in_frames, takeover_frame,
controlled_frames, prefix_replay_frames, warmup_frames, total_frames,
per_decision: (logical_frame, pixel_frame, decision_index, pixel_digest,
               requested_mask, executed_mask, hold_frames),
stage_clear_events_bit, stage_clear_edge_frame, state_stage_start, state_stage_end,
terminal_kind, deaths, lives_remaining, bombs_remaining, power_final,
engine_sha256, obs_spec_digest, data_dir, exit_code
```

`pixel_digest` 用 harness 的 `PIXELS` 回包 `checksum`（RGBA 的 FNV-1a-64），这样"像素确实来自正确帧"可被外部复算。

## 验收判据（照 `PIXEL_SLICE_PLAN §4 P1` 原文落地）

| 判据 | 含义 |
| --- | --- |
| 真实像素进入 actor | `pixel_digest` 与同帧 `PIXELS` 回包一致 |
| 动作作用于预定后续帧 | 请求 mask 与执行 mask 的对应关系可核，`hold_frames` 符合 `decision_frames` |
| 日志实际动作与历史一致 | `act_hist` 只含**真实生效**的动作；未生效/失败的输入不得记为已执行 |
| 终态可独立重算 | 两条口径都记录，且能被第三方用日志重算 |
| **合法死亡也是有效轨迹** | 请求后合法死亡**保留**，不得为了拿满窗重试 |
| 技术失败单列 | 帧丢失、错位、恢复哈希不符、终态不明 → 记技术失败并报**首个差异位置** |

## 四种结果 → 四种下一步

| 结果 | 结论 | 下一步 |
| --- | --- | --- |
| 完整真实像素轨迹 + 四判据通过 + 隔离检查通过 | 接线通过 | 才可提名 P2 小学习实施单 |
| 首个恢复/像素时序/动作生效/历史冲突可复现 | 接线失败 | 在版本范围内定位修复，保留原记录 |
| 同约束教师拿不到真实成功 | 正对照失败 | 先查可解性/动作契约，**不**归咎像素学习 |
| 无合法前缀 / 身份或终态不明 | 待证 | 报具体文件/帧/错误，重采另立工单 |

## 本提案明确**不**主张

- 不主张像素可学、不主张能过关、不主张 G410-3/G410-4 成立。
- 不主张离线边界测试能替代真实轨迹（它只能验证判分实现）。
- 不主张 E3 的 burn-in 是既有契约。
- 不主张 P1 通过就等于 R2 的"学习带来改善"那条也成立——那是另一张单。

## 开工前置（缺一不得开跑）

1. R1 的 D1/D2/D6 已批准；
2. 静音三件套（D7：私有 data 目录静音 cfg → `SDL_AUDIODRIVER=dummy` → 启动后 `v431_audio_cfg_audit.py` 落在 `muted`）；
3. 产物写**新路径**，不改既有证据、不覆盖资产；新资产按 `ASSET_SYNC` 登记。

**在收到批准前，本提案不启动任何引擎。**
