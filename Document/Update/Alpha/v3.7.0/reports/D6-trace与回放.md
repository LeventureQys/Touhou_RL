# v3.7.0 D6 报告：trace sidecar、离线回放与归因

- 日期：2026-10-25
- 任务：开发计划 D6 / R2（trace schema、原因码/时延/观测年龄、分层回放、死亡前 N 帧归因）
- 结论类型：**记录与回放就绪**；A-01…A-08 仍 `not_started`（D7）。
- 产物：`touhou_rl/escape/trace.py`、`render.py`、CLI `record`/`render`、`tests/test_escape_trace.py`（7 项）、[data/replays/](<../data/replays/index.json>)。

---

## 1. 记录格式

每个决策一行 JSONL，四个已有契约的记录 + 计时 + 结果：

| 字段 | 内容 |
|---|---|
| `observation` | `EscapeObservation` 的标量字段（玩家、帧、完整性、坐标契约、池计数、观测年龄、动作延迟） |
| `hazards_ref` | 弹池存放在 NPZ sidecar，JSON 只记数量（否则 90 决策约 4 MB） |
| `forecast` | 摘要：节点数、丢弃实体、是否 calibrated、计划状态、计划耗时、打分模式 |
| `plan` | 完整 `RoutePlan`：动作序列、18 首动作状态与理由、余量、关闭时刻、耗时 |
| `decision_record` | 完整 `ControlDecision`：意图、接受动作、拒绝动作与证据、回退、bomb 理由 |
| `timing` / `outcome` | 控制器耗时；本决策是否命中、推进帧数 |

**为什么弹池不内联**：`EscapeObservation` 的 hazards 列表在密集场景有数百个盒子。内联后单决策约 4 KB→40 KB，90 决策的 trace 会到 4 MB。sidecar 与 v3.5.0 的布局一致（positions/velocities/halves/offsets），回放时按 `sidecar_index` 取回。这个有损选择写在每条记录的 `hazards_ref` 里，不是隐式省略。

## 2. 回放与归因

- `render_text`：决策表（意图、路线、首动作、计划状态、关闭时刻、最小余量、回退）＋ 归因摘要。
- `render_png`：单帧分层——红色危险盒、蓝色计划路线、绿色实际轨迹、黄色自机。**真值与预测分色分层**，避免“图好看”被当成预测正确的证据。
- `attribute`：死亡前 N 帧的意图/计划状态统计，分类为 `clean`、`no_route_at_decision`、`hit_while_route_feasible`。

## 3. 四个样例（已落盘）

| 样例 | 决策 | 结果 | 归因 | 用途 |
|---|---:|---|---|---|
| E-05 | 30 | 存活 60 帧 | clean；窗口内 evacuate 8 | **提前撤离**：压升地板逼近前已在走廊内 |
| SC-15 | 90 | 存活 180 帧 | clean；local 5 + evacuate 3 | **拒绝死胡同**：轨迹从 20 px 安全口穿过 |
| E-11 | 3 | 第 6 帧命中 | `no_route_at_decision`；unknown 3 | **无路**：区域覆盖，控制器不假装有解 |
| E-06 | 8 | 第 16 帧命中 | `no_route_at_decision`；evacuate 3 + unknown 5 | **失败回放**：0.65 px 缝确实不可过 |

文件：`data/replays/{E-05,SC-15-0,E-11,E-06}.{jsonl,npz,md,png}`、`index.json`。trace 体积 14–528 KB（SC-15 90 决策 528 KB）。

SC-15 的回放图可以直接看出：轨迹从墙的下方 20 px 缺口穿过，上方 40 px 的宽口通向 x=340 的封闭口袋——这正是 D4 里“宽口是死胡同”的场景。

## 4. 已知限制

| 项 | 说明 |
|---|---|
| 预测覆盖网格未持久化 | 只存摘要；需要时由 sidecar + 配置重算（同 D3 的 31 节点） |
| 观测里的 hazards 不内联 | 见 §1；契约校验需先按 sidecar 回填 |
| `deadline_miss` | 合成环境无墙钟截止，恒 False；真机由 D8 填 |
| PNG 不含时间轴动画 | 单帧分层；逐帧序列可以用每决策一张图，未做 GIF |
| trace 体积 | 90 决策 528 KB；真机长局需要分片或压缩（D8） |

## 5. 复现

```powershell
python -m pytest touhou_rl/tests -k escape --basetemp=.tmp/v370-d6 -p no:cacheprovider
python -m touhou_rl.escape record --edge E-01 --out tmp/traces --plan-horizon 4 --plan-continuation 1
python -m touhou_rl.escape render --trace tmp/traces/E-01.jsonl --out tmp/replays
python -m touhou_rl.escape render --trace tmp/traces/E-01.jsonl --out tmp/replays --text-only
```

## 6. 下一步（D7）

1. 把 `evaluate` 命令从桩变成真：按冻结的 dev/formal 种子跑 17 族 × 30，统计 A-01…A-08。
2. A-01：确定性穿越/窄缝/时间窗口/第二波/远处封路实例全部通过，漏检为 0。
3. A-02/A-03：有解家族成功率与按时撤离率；A-04：相对旧 pilot 的可避免封死率；A-07：入坑拒绝；A-08：零散弹回归。
4. 先跑 dev 集，冻结配置后再跑 formal；formal 结果只写一次。
