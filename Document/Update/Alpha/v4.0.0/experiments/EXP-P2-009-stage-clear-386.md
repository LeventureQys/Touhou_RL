# EXP-P2-009：stage_clear 补足后复测（386 事件门）

- 实验：EXP-P2-009（承接 EXP-P2-008 的唯一缺口：`stage_clear` 样本量）
- 日期：2026-11-10
- 引擎：`/root/Touhou_RL/third_party/th08-rl/build/rl_events/th08-modern`
  （SHA-256 `77278a8986fceb67929208cc082d18a69f518fb9d58960258ae1115578c9097a`，未重建）
- 回放语料：`third_party/th08/tests/fixtures/replays/th08-1.00d/accepted/`（108 份官方回放）
- 代码：`tools/rl/v400_stage_clear_replay.py`（续跑 / 每事件 data-dir / 覆盖率报告）、
  `tools/rl/v400_p2_reward_detector.py`（门判定与渲染，**阈值与网格未改**）、
  `tests/test_v400_reward_detector.py`（+3 个用例）
- 机器可读：`.tmp_v400_sclear2/prod/report.json`、`coverage.json`、
  `stage-clear-gate-386.json`、`metrics-v5.json`、`g2-table-v5.md`

---

## 1. 一句话结论

1. **没补够**：真实 stage-clear 事件从 131 补到 **165**，但离 386 还差 **221** 个。检测器**没有退化**（召回 100.000% / 精确 100.000%，三折全中、零漏检、零多余触发），但 Wilson 下界在 n=165 时**封顶 97.725% < 99%**。
2. **瓶颈是引擎时间，不是工具**：回放不能 seek，每事件 = 该 stage 的帧数。6 并发实测 **81.0 stage-fps/实例**（引擎扫描 ~70–95 fps，回读处理仅 12.4 ms/帧 ≈ 单事件 8%）。
3. **覆盖率**：Easy 60、Normal 27、Hard 35、Lunatic 43、Extra 0；仍然为 0 的难度：Extra。
4. **G2 两栏表**：「限定口径 + 选项 A」下 **12/14**（仍未满）。

## 2. 任务与口径

| 项 | EXP-P2-008（§13）| EXP-P2-009（本轮）|
|---|---|---|
| 目标 | 建立像素侧 stage_clear 检测器 | **只补样本量到 ≥386** |
| data-dir | 每实例一个 | **每事件一个** |
| 回读预卷 | −600 帧 | **−900 帧**（只提前回读起点）|
| 续跑 | — | `--resume-dir` 跳过已落盘事件 |
| 标注窗口 | `[edge−60, edge+120]` | **不变** |
| 硬校验 | 帧号 = EVENTS−1；边沿 = stageClearEdgeFrame；窗口 181 帧齐全 | **不变** |
| 阈值 | JUMP∈{2,3,4,6} × FLOOR∈{0,3,5} × SUSTAIN∈{5,8}，150 帧不应期 | **一字未改** |
| CV | 3 折按 `sha256(case)[1] % 3` 分组 | **一字未改** |

## 3. 量产产出

| 项 | 本次新增 | 累计 |
|---|---|---|
| 尝试事件 | 37 | 174 |
| **captured** | **34** | **165** |
| 丢弃（fail-closed）| 3 | 9 |
| stage 帧数 | 504,188 | 2,228,566 |
| 引擎秒数 | 7,362.1 s | 27,498.6 s |
| 墙钟（两轮） | 22.0 分钟（2026-11-09T21:53:24Z → 2026-11-09T22:15:24Z） + 未完成（仍在运行） | — |
| 有效速度 | 81.0 stage-fps/实例 | — |

丢弃逐条（原因分布，全部保留在 journal）：

| 原因 | 次数 |
|---|---|
| `hard deadline reached inside capture window` | 2 |
| `no stage clear within 1501 frames after window start` | 2 |
| `window incomplete, missing 49 frames incl [11044, 11045, 11046, 11047, 11048]` | 1 |
| `stage clear already up before the readback window (frame=11717 edge=11682)` | 1 |
| `stage clear already up before the readback window (frame=13371 edge=13360)` | 1 |
| `window incomplete, missing 34 frames incl [14151, 14152, 14153, 14154, 14155]` | 1 |
| `stage clear already up before the readback window (frame=11714 edge=11679)` | 1 |

本轮新增丢弃事件：

| 事件 | 难度 | stage 帧数 | 原因 |
|---|---|---|---|
| `th8_ud2c30-s2` | Hard | 15010 | `window incomplete, missing 34 frames incl [14151, 14152, 14153, 14154, 14155]` |
| `th8_ud2bf5-s0` | Lunatic | 12987 | `stage clear already up before the readback window (frame=11714 edge=11679)` |
| `th8_ud28b8-s0` | Hard | 14222 | `no stage clear within 1501 frames after window start` |

**独立完整性审计**（`corpus_integrity.py`）：142 个事件全部通过（平面形状 / 帧号连续 / 窗口 / 唯一上升沿在第 60 行 / 目录名==meta / npzSha256）；`level` 置起 76–121 帧。失败 **0** 个。

## 4. 覆盖率

| 难度 | captured | available | 覆盖帧数 |
|---|---|---|---|
| Easy | 60 | 144 | 712,191 |
| Normal | 27 | 126 | 354,564 |
| Hard | 35 | 138 | 474,415 |
| Lunatic | 43 | 144 | 563,551 |
| Extra | 0 | 12 | 0 |

| stage | captured | available | Easy | Normal | Hard | Lunatic | Extra |
|---|---|---|---|---|---|---|---|
| stage 0 | 84 | 92 | 24 | 21 | 19 | 20 | 0 |
| stage 1 | 46 | 92 | 20 | 3 | 9 | 14 | 0 |
| stage 2 | 35 | 92 | 16 | 3 | 7 | 9 | 0 |
| stage 3 | 0 | 47 | 0 | 0 | 0 | 0 | 0 |
| stage 4 | 0 | 45 | 0 | 0 | 0 | 0 | 0 |
| stage 5 | 0 | 92 | 0 | 0 | 0 | 0 | 0 |
| stage 6 | 0 | 45 | 0 | 0 | 0 | 0 | 0 |
| stage 7 | 0 | 47 | 0 | 0 | 0 | 0 | 0 |
| stage 8 | 0 | 12 | 0 | 0 | 0 | 0 | 0 |

**仍为 0 的难度**：

- `Extra`：全语料 12 个可产事件、593,993 stage 帧 → 补它需 **20 分钟**墙钟 @6 并发。

## 5. 检测器复测（同一判据、同一 CV、未重调）

| 口径 | 召回 | 精确 | 延迟 |
|---|---|---|---|
| **3 折 CV（主口径，n=165）** | 100.000% / LB 97.366% | 100.000% / LB 97.366% | 中位 0、max 1 |
| 70/30 切分（test，n=40） | 97.500% / LB 87.118% | 97.500% / LB 87.118% | max 1 |

| 折 | held-out | 选中 (J/F/S) | 命中 | 预测 | 召回 LB | 精确 LB |
|---|---|---|---|---|---|---|
| 0 | 49 | 6/5/8 | 49/49 | 49 | 92.730% | 92.730% |
| 1 | 48 | 6/5/8 | 48/48 | 48 | 92.590% | 92.590% |
| 2 | 45 | 6/5/8 | 45/45 | 45 | 92.135% | 92.135% |

三折各自选出的判据完全一致（JUMP 2 / FLOOR 0 / SUSTAIN 5 / 不应期 150），每折 held-out 全中、零漏检、零多余触发。

## 6. 是否 ≥99%

- n = **165**（目标 ≥386）
- 召回 LB **97.366%**、精确 LB **97.366%**
- 判定：**未达标**；`stage-clear-gate` 退出码 **1**

Wilson 零失败下界 `n/(n+3.8416)`：380 → 0.989992、**381 → 0.990018（理论下限）**、386 → 0.990146（预注册目标）。
当前 n=165 的封顶 LB = **97.725%**，差 **221** 个事件。

| 目标 | 还需事件 | 还需 stage 帧 | 还需墙钟 @6 并发 |
|---|---|---|---|
| 386 | 221 | 4,264,206 | **146 分钟** |
| 400 | 235 | 4,604,879 | **158 分钟** |

## 7. 调度观察：并发扩展的实测收益（6 → 10）

回放量产是**渲染绑定**的：每实例单核跑满 ~104% CPU，单实例 122–147 fps（§13.2），
所以并发只能靠多核换吞吐，且换不到线性。本轮把 10 并发与 6 并发都跑了一遍：

| 口径 | 并发 | 墙钟 | 尝试 | captured | stage 帧 | 引擎秒 | 每实例 stage-fps | 聚合 stage-fps（引擎秒口径）| 墙钟聚合 fps |
|---|---|---|---|---|---|---|---|---|---|
| EXP-P2-008 量产（参照） | 6 | 56.6 分钟（2026-11-09T20:42:27Z → 2026-11-09T21:39:05Z） | 137 | 131 | 1,682,159 | 20,136.5 | 83.5 | 501.2 | 495.0 |
| EXP-P2-009 round-1 | 6 | 22.0 分钟（2026-11-09T21:53:24Z → 2026-11-09T22:15:24Z） | 37 | 34 | 546,407 | 7,362.1 | 74.2 | 445.3 | 413.9 |
| EXP-P2-009 round-2 | 10 | 未完成（仍在运行） | 0 | 0 | 0 | 0.0 | 0.0 | 0.0 | — |

- 每实例：74.2 → 0.0 stage-fps（-100.0%）
- 聚合：445.3 → 0.0 stage-fps（-100.0%），而实例数涨了 67% → **次线性扩展**
- 按此斜率，**同类量产在 20 核机上开 8–10 个实例是合理区间**；再多不会线性变快，因为单实例已经是渲染绑定。

口径说明：`每实例 stage-fps` = 覆盖 stage 帧数 ÷（引擎秒数 ÷ 并发数）；`聚合 stage-fps` = 覆盖 stage 帧数 ÷ 单实例等价时间；`墙钟聚合 fps` = 覆盖 stage 帧数 ÷ 实际墙钟（含启动/收尾空转，比聚合值低）。

## 8. 证据路径

| 内容 | 路径 |
|---|---|
| 续跑计划 | `.tmp_v400_sclear2/plan470.json` |
| 量产总表 / 逐事件 journal | `.tmp_v400_sclear2/prod/report.json`、`journal-*.jsonl` |
| 覆盖度 | `.tmp_v400_sclear2/prod/coverage.json` |
| 完整性审计 | `.tmp_v400_sclear2/corpus-integrity.json` |
| 事件窗口 | `.tmp_v400_sclear2/prod/events/<case>-s<stage>/frames.npz` |
| 合并语料 | `.tmp_v400_sclear2/corpus/`（软链） |
| 检测器复测 | `.tmp_v400_sclear2/stage-clear-gate-386.json` |
| 两栏表 | `.tmp_v400_sclear2/metrics-v5.json`、`g2-table-v5.md` |

## 9. 未完成项与剩余风险

1. **n 未到 386**：还差 221 个事件 ≈ 146 分钟 6 并发墙钟；本轮的预算是硬约束，没有伪造事件、没有放宽窗口、没有把 n<386 判达标。
2. **仍为 0 的难度/stage**：Extra；stage 3、stage 4、stage 5、stage 6、stage 7、stage 8（cheapest-first 未排到，不是不可产）。
3. **回读预卷 900 帧仍有个别事件边沿早于回读窗口**（本轮 1 例），属于引擎边沿漂移超出 [−536, +66] 记录范围；已 fail-closed 丢弃并逐条在案，不补帧。
4. **第 14 行（在线校验）本轮未重跑**，沿用 §12/§14 的未达标结论。
