# EXP-P2-008：stage_clear 回放量产与像素检测器 + WP5 像素侧死亡检测

- 实验：EXP-P2-008（承接 EXP-P2-007 的两项缺口：stage_clear 门、WP5 在线校验）
- 日期：2026-11-10
- 引擎：`/root/Touhou_RL/third_party/th08-rl/build/rl_events/th08-modern`
  （SHA-256 `77278a8986fceb67929208cc082d18a69f518fb9d58960258ae1115578c9097a`，
  `TH08_RL_HARNESS=ON`，带 `EVENTS 0x0080 = STAGE_CLEAR` 与 `stageClearEdgeFrame`）
- 回放语料：`third_party/th08/tests/fixtures/replays/th08-1.00d/accepted/`（108 份官方回放，
  Easy/Normal/Hard/Lunatic 各 24 + Extra 12）
- 机器可读：
  - `.tmp_v400_sclear/prod/report.json`（量产总表）、`summary.json`
  - `.tmp_v400_sclear/stage-clear-gate.json`（像素检测器 held-out）
  - `.tmp_v400_sclear/pixel-death-ncc.json`（WP5 像素侧死亡，四版判据见 §5）
  - `.tmp_v400_sclear/stage_inventory.json`（每 stage 精确帧数）
- 代码：`tools/rl/v400_stage_clear_replay.py`（新增）、
  `tools/rl/v400_p2_reward_detector.py`（新增 `stage_clear_*` / `player_patch_ncc` /
  `pixel_death_*` / `stage-clear-gate` 子命令；两栏表 stage_clear 行改为「召回+精确都要 LB」）

---

## 1. 一句话结论

1. **回放确实能廉价量产过面事件**：`rl_events` 引擎能播放全部 108 份官方回放，
   `EVENTS 0x0080` 在每个 stage 结束时真实置起、`stageClearEdgeFrame` 精确。用 WP1 差分
   retrace 产物统计出全语料 **564 个可解析的真实过面事件**。
2. 但**「廉价」是相对的**：回放不能 seek，每个事件必须从该 stage 起点模拟到结尾。
   harness 实测 **122–147 fps/实例**（渲染绑定），6 并发下 386 个最便宜 stage 仍需
   **约 126 分钟**引擎时间 —— 150 分钟总预算内**拿不到 386 个**。本轮按预算量产
   **131 个真实事件**（全部通过 fail-closed 校验）。
3. **像素检测器做出来了且可分**：判据是 `play64` 上「上亮下暗」的**跳变**
   （`vgrad[t] - vgrad[t-1] >= JUMP` 且 `vgrad[t..t+4] >= FLOOR`，150 帧不应期；
   纯电平判据在 stage 2 上会失败，已改为跳变判据）。
   主指标用 **3 折按回放交叉验证**（每个事件都由没见过它的回放选出阈值判定）：
   **事件级召回 100.00%（LB 97.15%，n=131）
   / 精确 100.00%（LB 97.15%）/ 延迟 max 1 帧**。
4. 但 **n=131 < 386**，Wilson 下界封顶 `n/(n+3.84)` → **97.15% < 99%**，
   `stage_clear` 门按「样本量不足」判**未达标**。这是**样本量问题，不是检测器问题**：
   要把 LB 抬到 99% 需要 ≥386 个事件，即追加约 **109 分钟** 6 并发引擎时间。
5. **WP5 像素侧死亡检测未过门**：四版判据里最好的（机体模板 NCC 崩塌）held-out 召回
   **16.41%（LB 14.14%，n=908）/ 精确 17.43%（LB 15.03%，n=855）/ 延迟中位 0、max 10**。
   根因已量化：死亡的视觉信号真实存在（NCC 1.0→0.38，+0～4 帧），但非死亡帧也有 1% 低 NCC，
   简单阈值不可能同时拿到 99% 精确；需要学习型检测器（play96 小 CNN）或带遮挡处理的跟踪器，
   本轮时间盒内没做，**不伪造**。

---

## 2. 回放量产：能产出多少真实过面事件

### 2.1 事件总量（来自 WP1 retrace 产物，先算账后跑）

`stage_clear` 的最便宜来源是**每个 stage 的结尾**。`.tmp_v400_replay_retrace_108/` 里每份回放都有

- `on/attempt-1/complete.json` 的 `stageFrames`：**每个 stage 的精确帧数**（当次重放的引擎真值）；
- `replay.stages`：该回放实际录到的 stage 列表（含 Extra 的 stage 7）。

| 项 | 值 |
|---|---|
| 官方回放 | 108 份 |
| 可解析 stage 条目 | 588 |
| **可解析为真实事件**（回放文件存在 + stageFrames>0） | **564** |
| 事件总帧数 | 11,781,533 |
| 最便宜 386 个 stage 的帧数 | 6,368,927 |
| 386 × 6 并发 @140 fps | **约 126 分钟**（单实例 45,492 s ÷ 6） |
| 最便宜 100 / 200 / 300 个 | 1.16M / 2.62M / 4.39M 帧 |

**结论（先回报）**：总量够（564 ≥ 386），但 150 分钟预算内 6 并发只能买到
`6 × 60 × 140 × 150 ≈ 7.5M` 帧理论上限，实际被启动/回读/进程开销吃掉约 25%，
真实可交付约 **131 个**。要 386 个需追加约 **109 分钟**（见 §6）。

### 2.2 单事件成本实测

| 阶段 | 帧 | 实测 |
|---|---|---|
| harness 启动 + 回放挂载 | ~85 帧 | 2–3 s |
| stage 回放（`TH08_AUTOPLAY_STAGE=<stage>`） | stageFrames（8.2k–28k） | 122–147 fps |
| 回读窗口（edge 前 600 帧起逐帧 `READBACK 1` + `/dev/shm` 读像素） | ~600–900 帧 | ~110 fps |
| 合计（8,224 帧的 Easy stage 0） | — | **78.6 s / 事件**，470 KB/事件 |

**关键事实：回放不能 seek。** 引擎按 replay 输入流逐帧推进，`stageClearEdgeFrame` 只在
模拟到该帧时才出现；所以「每事件成本 = 该 stage 的帧数」，没有捷径。这也是 42 h 估算的来源。

### 2.3 量产口径（写进代码的 fail-closed 规则）

`tools/rl/v400_stage_clear_replay.py`（新增）：

1. **plan**：读 `complete.json` 的 `stageFrames`，按帧数从便宜到贵排序，取前 N 个；
2. **worker**：每事件独立 `--data-dir`（`th08.dat`/`thbgm.dat` 软链、`score.dat`/`th08.cfg` 每实例复制），
   `TH08_AUTOPLAY_REPLAY=<abs .rpy>`、`TH08_AUTOPLAY_STAGE=<stage>`；
3. 扫描段用 `STEP 0 120` + 每段 `EVENTS` 轮询，若在回读窗口前就见到 `stageClear=1`
   → 记 *pre-roll missed*（**不猜帧、不补帧**）；
4. 预测 `capture_start = startup + stageFrames - 600`，此后**逐帧** `STEP 0 1` + `EVENTS` + `HUD` +
   读共享内存像素；像素帧号必须等于 `EVENTS frames - 1`（`AfterDrawFrame` 在 `++g_frameCount`
   之前发布），否则重读一次、仍不符则整事件判失败；
5. 事件边沿必须同时满足：观测到的上升沿 == 引擎 `stageClearEdgeFrame`；窗口 `[edge-60, edge+120]`
   181 帧齐全；否则**整事件丢弃并写 journal**；
6. 事件写 `events/<case>-s<stage>/frames.npz`（`play` 64×64、`hud` 56×160、`thumb` 32×32、
   逐帧 16 个标量、`level`/`screenState`/`seq`/`flags`/`hudFields`）+ `meta.json`（含引擎/回放 SHA）。

### 2.4 实际产出

| 项 | 值 |
|---|---|
| 尝试事件 | 137 |
| **captured（计入 n）** | **131** |
| fail-closed 丢弃 | 6（原因：{"hard deadline reached inside capture window": 2, "window incomplete, missing 49 frames incl [11044, 11045, 11046, 11047, 11048]": 1, "stage clear already up before the readback window (frame=11717 edge=11682)": 1, "no stage clear within 1501 frames after window start": 1, "stage clear already up before the readback window (frame=13371 edge=13360)": 1}） |
| 难度分布 | Easy 53, Hard 22, Lunatic 32, Normal 24 |
| 引擎秒数合计 | 20136.5 s（6 并发 → 墙钟约 55.9 min） |
| 覆盖 stage 帧数 | 1,682,159 |
| 有效速度 | 83.5 fps/实例 |
| 每事件数据 | play64 64×64 + hud 56×160 + thumb 32×32 + 16 标量，181 帧（−60/+120） |
| `0x0080` 电平持续 | 76–121 帧（窗口内） |
| 边沿 vs retrace stage 帧数漂移 | 中位 -99 帧、范围 [-536, +66] |

---

## 3. stage_clear 像素检测器

### 3.1 可分离性（用 train 回放的 181 帧窗口）

| 特征 | pos 中位 | neg 中位 | AUC | 方向 |
|---|---|---|---|---|
| `brightFrac` | 0.00 | 0.00 | 0.004 | low |
| `vgrad` | 7.36 | -3.03 | 0.808 | high |
| `colour` | 24.41 | 33.37 | 0.245 | low |
| `bMean` | 38.47 | 46.40 | 0.287 | low |
| `playMotion` | 1.69 | 0.53 | 0.693 | high |
| `gradX` | 10.85 | 7.51 | 0.680 | high |
| `playStd` | 11.14 | 12.40 | 0.320 | low |
| `gradY` | 10.45 | 8.11 | 0.651 | high |

### 3.2 判据与阈值来源

- **判据**：`vgrad = mean(play64[0:16]) - mean(play64[48:64])`（play64 行=竖直轴）。
  游戏内画面下亮上暗（`vgrad` 明显为负）；过面屏换成「上亮下暗」的结算面板。
  事件帧 = `vgrad[t] - vgrad[t-1] >= JUMP` 且 `vgrad[t..t+SUSTAIN-1] >= FLOOR`；
  150 帧不应期（同一次过面屏的淡出与下一面开场不重复报）。
- **为什么不用纯电平**：`vgrad >= 6` 的电平判据在 **stage 2** 上失败——该面背景本身就上亮下暗，
  窗口起点电平已为真、找不到上升沿（3 个事件全丢）。改成「跳变 + 保持上亮」后，
  `JUMP ∈ [2,6]`、`FLOOR ∈ [0,5]` 都能 100% 命中。
- **阈值来源**：只在 **train 回放** 上网格搜索（`JUMP ∈ {2,3,4,6}`、`FLOOR ∈ {0,3,5}`、
  `SUSTAIN ∈ {5,8}`），按「事件级召回 Wilson LB，其次精确 Wilson LB」选；
  主指标用 **3 折按回放交叉验证**（每个事件都由没见过它的回放选出的阈值判定），
  held-out 回放不参与选择。
- 单帧误触发靠 `SUSTAIN` 抑制，不应期挡掉同屏二次上升沿。

### 3.3 held-out 结果

| 口径 | 召回 | 精确 | 延迟 |
|---|---|---|---|
| **3 折交叉验证（主口径，n=131）** | **100.00% / LB 97.15%** | **100.00% / LB 97.15%** | 中位 0、max 1 帧 |
| 单次 70/30 切分（test，n=37） | 100.00% / LB 90.59% | 100.00% / LB 90.59% | max 1 帧 |
| 帧级（状态判据 `vgrad>=FLOOR`，n=11329 正 / 12214 判正） | 72.35% / LB 71.52% | 67.11% / LB 66.27% | — |

| 折 | held-out 事件 | 选中 (JUMP/FLOOR/SUSTAIN) | 命中 | 预测 | 召回 LB | 精确 LB |
|---|---|---|---|---|---|---|
| 0 | 46 | 6/5/8 | 46/46 | 46 | 92.29% | 92.29% |
| 1 | 45 | 6/5/8 | 45/45 | 45 | 92.13% | 92.13% |
| 2 | 40 | 6/5/8 | 40/40 | 40 | 91.24% | 91.24% |

---

## 4. 与 G2 两栏表的关系

- `stage_clear` 行（第 13 行）从「无检测器」变成「有检测器、100%/100%、但 n 不足」；
  行判定规则改为**召回与精确的 Wilson LB 都要 ≥0.99**（此前只看召回）。
- **「限定口径 + 选项 A」下 14 行不全是达标（12/14）**：完整 14 行表见 G2 报告 §14.6。

---

## 5. WP5 像素侧死亡检测（摘要，全文见 G2 报告 §14）

| 版本 | 判据 | held-out 召回 / 精确（n） | 结论 |
|---|---|---|---|
| v1 | 全屏块差爆点 + 块消失 | 0% / 0%（1 次触发） | 淘汰 |
| v2 | 亮像素骤降 + 全屏均值下降 | 14.3% / 9.0%（n=855） | 淘汰（仅能分「bomb 空→屏幕转暗」一类） |
| v3 | 机体窗口运动崩塌 | （转置轴错误，已弃用） | 弃用 |
| v4 | 机体 5×5 patch 零均值 NCC 崩塌 | **16.41% / 17.43%（LB 14.14% / 15.03%）**，延迟中位 0 | 最好，但远未过门 |

**根因量化**：死亡窗口内 NCC 最小值 p50=0.377、p95=0.502（正常帧 p01=0.383）——
信号真实（像素在事件后 0–4 帧可见，§12 的 +29 帧确实是 HUD 刷新问题），
但简单阈值必然带到两位数假阳性。在线校验行**仍未达标**。

---

## 6. 复现命令与退出码

```bash
# 1) 每 stage 帧数清单（来自 WP1 retrace 产物；exit 0）
wsl -d Ubuntu bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_sclear/inventory.py"

# 2) 选 386 个最便宜 stage（写 plan.json；exit 0）
wsl -d Ubuntu bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_stage_clear_replay.py plan --target-events 386 --output .tmp_v400_sclear/plan.json"

# 3) 6 并发量产（每实例独立 data-dir；deadline 由脚本内 date 计算；exit 0）
wsl -d Ubuntu bash -lc "bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_sclear/run_production.sh"

# 4) 汇总（写 report.json / summary.json；exit 0）
wsl -d Ubuntu bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_stage_clear_replay.py report --out .tmp_v400_sclear/prod"

# 5) 像素检测器 held-out（写 stage-clear-gate.json；达标 exit 0，未达标 exit 1）
wsl -d Ubuntu bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_p2_reward_detector.py stage-clear-gate --events .tmp_v400_sclear/prod/events --out .tmp_v400_sclear/stage-clear-gate.json"

# 6) WP5 像素侧死亡四版判据（exit 0）
wsl -d Ubuntu bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_sclear/pixel_death_ncc.py"

# 7) 单测（Windows；61 passed）
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_sclear/pytest -q tests/test_v400_reward_detector.py
```

---

## 7. 局限与下一步

1. **n=131 < 386**：要过 `stage_clear` 门需要再跑约 **109 分钟**（6 并发）到 386 个事件；
   plan 已按最便宜优先排好，直接重跑 `worker` 即可续。
2. **回放无 seek**：这是成本的物理下限；除非引擎提供「从 stage 内任意帧恢复」的快照/回放 seek，
   否则每事件必付一个 stage 的帧数。
3. **mid-stage 起的轻微时序漂移**：`TH08_AUTOPLAY_STAGE=s` 从该 stage 开头起放，长 stage 的
   实测边沿与 retrace 帧数会漂几十到几百帧；本轮用「预测 -600 帧开始回读 + 420 帧环形缓冲」
   兜住，仍有 6 个事件因窗口不齐被 fail-closed 丢弃（已写 journal，未计入 n）。
4. **像素侧死亡检测**需要学习型检测器（play96 小 CNN，正样本 908 个已就位）或带遮挡处理的
   模板跟踪器；本轮只做到「可分离性 + 简单判据上界」。
5. **stage_clear 在线回路**未跑（时间盒）；判据是纯像素、无前视（`sustain-1` 帧延迟），
   接线成本低，列为下一步。
