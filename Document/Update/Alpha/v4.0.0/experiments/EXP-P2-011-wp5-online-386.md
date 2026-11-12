# EXP-P2-011：WP5 在线校验到 ≥386/类（G2 第 14 行）

- 实验：EXP-P2-011（承接 EXP-P2-007 §12、EXP-P2-008 §14 的在线缺口）
- 日期：2026-11-10
- 引擎：`/root/Touhou_RL/third_party/th08-rl/build/rl_events/th08-modern`
  （SHA-256 `77278a8986fceb67929208cc082d18a69f518fb9d58960258ae1115578c9097a`，**未重建**）
- 代码：`tools/rl/v400_stage_clear_replay.py`（新增 `--online-detector`：把门自己那条
  vgrad 判据做成**逐帧流式**形式，嵌在回放量产回路里）、
  `tools/rl/v400_pixel_death.py`（新增按 checkpoint 推断通道数的加载路径）、
  `.tmp_v400_g2online/wp5_online386.py`（实时回路驱动 + 四类评分）
- 机器可读：`.tmp_v400_g2online/episodes-*.jsonl`、`wp5-summary.json`、
  `.tmp_v400_sclear2/prod12/report.json`（其 `online` 块）、`.tmp_v400_wp5/bengeval*.json`

---

## 1. 预注册（**在运行之前**写死，运行后不得改动）

冻结时间：2026-11-10T02:20Z（早于任何一局在线采集；生产回路 02:13:48Z 起跑时判据常量已定）。

### 1.1 四类的真值与检测器（判据与阈值**一律不新调**）

| 类 | 真值（引擎侧，只用于评分，不进检测器） | 在线检测器 | 常量来源 |
|---|---|---|---|
| death | `EVENTS & 0x0004` 上升沿 | `play96` 7 帧因果窗 → 小 CNN（`v400_pixel_death.PixelDeathNet`，width 32，88,721 参数），阈值/不应期取自 checkpoint 的 `valThreshold`/`config`，告警规则 = 上升沿 +1 帧确认 + 60 帧不应期 | EXP-P2-008 §14 判据 v5（学习型）；本轮**不重训、不在线调阈值** |
| bomb | `EVENTS & 0x0001` 上升沿 | HUD `bombs` 严格递减（同一把尺 = EXP-P2-004） | 与离线完全相同 |
| game_over | HUD `gameOverFlag` 连续段（起点） | `play96` 平均 \|x[t]−x[t−60]\| < **1.736** 的上升沿 | EXP-P2-006 §4.3 预注册阈值，未改 |
| stage_clear | `EVENTS & 0x0080` 上升沿 | `vgrad = mean(play64[0:16]) − mean(play64[48:64])`；`Δvgrad ≥ JUMP` 且随后 `SUSTAIN` 帧 `vgrad ≥ FLOOR`，150 帧不应期 | 门自己 `thresholds` 块报的 **jump 2.0 / floor 0.0 / sustain 5 / refractory 150**（`.tmp_v400_sclear2/stage-clear-gate-386.json`），**一字未改** |

补充约定：
- death 另报 **bomb 用光子集**（事件帧 `bombs == 0`，即 §12.3 的 +29 帧消解路径）的召回；
- death 另报 **HUD 计数器参考行**（`lives` 严格递减），并**标注其循环性**（该行不是像素检测器，只作对照）；
- 在线延迟 = 告警帧 − 事件帧；stage_clear 的流式形式需 `sustain` 帧前瞻，因此**同时**记录「判定帧（eventFrame）」和「可确认帧（alarmFrame）」。

### 1.2 驱动策略与样本量（固定配比，不按结果调整）

| 家族 | 策略 | 占比 | 作用 |
|---|---|---|---|
| `still` | 只按住射击键（mask = `cand_to_mask(0,0)`），不移动 | **3/4** | 稳定产生 death 与 game_over（3 条命耗尽） |
| `random-b35` | 每 6 帧换一个随机方向（18 候选之一），bomb 以 0.35 概率按下 | **1/8** | 把 bomb 存量打光 → 触发 +29 帧消解路径 |
| `random-b08` | 同上，bomb 概率 0.08 | **1/8** | bomb 存量保留对照组 |

- 每局：Stage 3、**Lunatic（d3）**、team 0、每实例独立 `--data-dir`、**一进程一局**（T7 之前 `RESET` 只能一次）；
- 每局帧数上限 2400；gameOverFlag 连续 30 帧后提前收局；
- 种子：`21100 + episode`（不复用 T6 的确认种子 60000–60199）；
- **目标样本量：每类 ≥ 386 个真值事件**（Wilson 零失败下界 386/(386+3.8416) = 0.99015）；
  预注册局数 520 局（still 390 / random-b35 65 / random-b08 65），按 `index % instances` 分片并发。
- **教师策略未纳入**（已声明的偏离）：教师需要 v2h checkpoint 与 obs 管线进回路，本轮时间盒内不做；
  两档 random 已覆盖「bomb 打光 / bomb 保留」这一对照，而它正是 death 两条消解路径的分界。

### 1.3 切分与判定

- **整局切分**：一局 = 一个进程 = 一个独立样本；在线回路**不做任何训练**，因此不存在「训练集泄漏」，
  但仍按整局报告（每局的事件数、告警数逐局在 `episodes-*.jsonl`）；
- 判定：每类的召回与精确 **Wilson 95% 下界 ≥ 0.99**，且 n ≥ 386；任一不满足 → 该类**未达标**；
- 第 14 行只有当**四类全部达标**时才记达标；
- 停机：预注册局数跑完、或时间盒（35 分钟）用完、或该类样本已 ≥ 500 则停；不因为「差一点」加跑。

## 2. 做法（在线回路）

每帧：`STEP <mask> 1` → `EVENTS`（death / bomb / stage_clear 位，仅做真值）→ `HUD`
（`lives` / `bombs` / `gameOverFlag`）→ 共享内存整帧回读 → `play96` → 三个在线判据当场判。
无回放、无前视（game_over 只用 `t−60`，CNN 只用 `t−6..t`），检测器**看不到** `EVENTS`/`HUD` 真值。

计时按阶段切开（`step` / `events` / `hud` / `capture` / `infer`），用来回答
「推理是否拖慢引擎」：对照为同一回路关掉推理的 `control` 局。

stage_clear 的在线行由**回放量产回路**给出（`--online-detector`），而不是另跑一遍实时回路：
同一条判据在逐帧到达的像素流上边到边算，引擎真值（`stageClearEdgeFrame`）只用于评分。

## 3. 结果

### 3.1 stage_clear 在线（第 14 行的 stage_clear 部分）

在线判据 = 冻结判据（jump 2.0 / floor 0.0 / sustain 5 / refractory 150）的**逐帧流式形式**，
嵌入回放量产回路，像素只进不出，引擎 `stageClearEdgeFrame` 只用于评分。**判据与阈值未改。**

先做等价性验证（用已落盘的 195 个真实事件逐帧重放）：**告警集合与离线判据 195/195 逐一相同**，
判定延迟固定 **+4 帧**（`sustain` 前瞻），耗时 **0.0123 ms/帧**。

**真连续流的结果（55 个事件，回读起点 → edge+120，中位 783 帧）——这是本轮最重要的发现：**

| 匹配口径 | 命中 | 召回 | 首个告警相对 edge |
|---|---|---|---|
| 门的冻结窗 `[edge, edge+10]` | **15 / 55** | **27.27%（LB 17.28%）** | 中位 **−112 帧**，范围 [−172, +1] |
| 允许早检的窗 `[edge−150, edge+10]`（只报告） | **55 / 55** | **100.00%** | 同上 |

**解释**：像素侧的过面板**比引擎的 `stageClear` 电平位早约 112 帧**开始出现，冻结判据在真连续流上
**每次都在过渡开始时正确触发**；但门的匹配窗锚在引擎位上升沿，于是把 40/55 次**提前命中**记成漏检
（同时，150 帧不应期又把这些事件在 `edge` 处的二次触发压掉）。
**切片评估（181 帧，起点=edge−60）把这段提前量裁掉了**，所以切片口径（248/250 = 99.20%）与
连续流口径（15/55 = 27.27%）差 72 个百分点 —— 差别在**匹配窗锚点**，不在检测器能力。

⇒ 第 13 行**不能靠补样本过门**，也不能靠「连续流评估」过门（连续流在冻结窗下更差）。
要让这一行真正达标，只有两条路，**两条都需要负责人签字改判据**：
1. 把事件锚点改成像素过渡起点（或把匹配窗左拓到 `[edge−150, edge+10]`）；
2. 或者让引擎把 `stageClear` 电平位改成与像素面板同时置起（引擎侧改动，属 T7 范畴）。

本轮量产因主机时钟跳变（见 §3.4）只跑出 **250** 个事件；带真连续流的只有 **55** 个。
事件级门判定见 §3.2 与 G2 报告 §16.4。

### 3.2 第 13 行：两个口径并列（n=250）

| 口径 | 命中 | 召回 / Wilson LB | 精确 / Wilson LB | 漏检事件 |
|---|---|---|---|---|
| **冻结判据 2.0/0.0/5（EXP-P2-009 公布）** | 248 / 250 | **99.20% / LB 97.13%** | 99.20% / LB 97.13% | `th8_ud2cce-s1`、`th8_ud2ce5-s2` |
| 门的 3 折 CV（每折重选，fold 选中 3.0/0.0/5） | 248 / 250 | 99.20% / LB 97.13% | 100.00% / LB 98.47% | `th8_ud2945-s2`、`th8_ud2cb3-s2` |
| 网格点 3.0/0.0/5（仅报告） | 249 / 250 | 99.60% / LB 97.77% | — | `th8_ud2cce-s1` |
| 网格点 4.0/5.0/8（仅报告） | **250 / 250** | 100.00% / LB 98.49% | — | 无 |
| 网格点 6.0/5.0/8（fold 1 曾选中） | 248 / 250 | 99.20% / LB 97.13% | — | 面板渐入的 2 个 |

**两个口径都不达标，而且样本量本身也还不达标**：n=250 时即使 250/250，Wilson 下界也只有
250/(250+3.84) = **98.49% < 99%**（要到 386 才有 99.01%）。所以第 13 行本轮判**未达标**有两条独立理由：
**n 不足**（时钟跳变导致量产中断）与**漏检 2 个**（冻结判据 248/250）。

**两个漏检的机制（逐事件查清，与「面板渐入」不同）**：这两个事件在回读窗口**开头**（第 3 帧 / 第 8 帧）
各有一次梯度跳变（Δ 16.06 / 2.16），而判据的 **150 帧不应期**把第 60 帧的真边沿压掉了：

| 事件 | 窗口内预测 | `vgrad[edge]` | `Δvgrad@edge` | 面板首次点亮（相对 edge）|
|---|---|---|---|---|
| `th8_ud2cce-s1` | `[3]` | 8.42 | +17.70 | 0 |
| `th8_ud2ce5-s2` | `[8]` | 9.92 | −1.49 | 0 |

全语料 250 个事件里，**只有这 2 个**在 `[1, edge)` 内出现过候选（`diag_predict.py`），因此漏检率 ≈0.8%。
`th8_ud2ce5-s2` 的 `vgrad` 在窗口第 6–8 帧就已经 ≥8（面板早于引擎边沿约 50 帧出现），
**即使换成连续流评估也会在 [edge, edge+10] 之外触发并被不应期压掉**；
`th8_ud2cce-s1` 则是开-关-开的三段式（第 3 帧点亮、第 59 帧又落回 −9.3、第 60 帧再亮）。

**放宽匹配窗的诊断（只报告，不用于判定）**：把窗口从 `[edge, edge+10]` 放宽到 `[edge−8, edge+10]`，
冻结判据仍是 **248/250**、严格点仍是 **248/250** —— **放宽窗口对这两个漏检完全无效**，
因为它们缺的不是窗口而是「不应期之前的那次误触发」。

> **口径敏感性（必须与上面一起读）**：181 帧切片被当成独立流评估时，窗口起点的抖动会先占掉
> 150 帧不应期，从而吃掉真边沿。生产回路已经把**整段逐帧 `vgrad` 流**落盘（`events/*/stream.npz`，
> 从回读起点到 edge+120，约 1,100 帧 / 4 KB），供下一轮做**连续流评估**；本轮 250 个事件里
> 有 55 个带流（改动后捕获的），其余 195 个只有 181 帧切片。

### 3.3 death CNN：修好加载器后重训，仍远未达标

| 项 | 旧 checkpoint（2026-11-10 早） | 本轮重训（ckpt2） |
|---|---|---|
| 加载 | `load_checkpoint` 按 6 通道重建 → **size mismatch，根本加载不了**（上一轮「CNN 半途」的真实原因） | 改成从 `state_dict` 推断通道数后正常加载 |
| 特征腿 | 5 条（训练时代码） | 6 条（当前代码，含 appearance novelty） |
| 负样本构成 | 25,291 respawn + **57** 普通帧（采样器把 respawn 先取满预算） | 21,673 respawn + **32,651** 普通帧（`respawn_share=0.5`，两池都随机抽） |
| 训练 | 12 epoch / CPU | 12 epoch / **RTX 4070 SUPER，979 s**（≤25 min 预算内） |
| val（整回放切分）| 最好 25.2% 召回 / 19.0% 精确 | 最好 **40.8% 召回 / 33.0% 精确**（epoch 10） |
| test（164 shard / 394 事件 / 212,188 帧，整回放切分）| **8.38% 召回 / LB 6.03%**；**2.08% 精确 / LB 1.49%**；1,585 次告警 | 见 `offline-test-ckpt2.json`（同一把尺）|
| CPU 推理 | — | **678 fps（10 线程）/ 555 fps（1 线程）**，1.47 ms/帧，满足「CPU ≥200 fps」 |

**结论**：负样本构成的修正把 val 召回从 25% 抬到 41%，但离 99% 差两个数量级；
这个规模（89k 参数、212k 帧、单一 stage 3 视角）的 CNN 在本轮预算内**做不到**第 14 行要的
10 帧窗 99% 召回 + 99% 精确。

### 3.4 中断与偏离（如实记录）

1. **量产被主机时钟跳变打断**：2026-11-10 约 02:50Z 之后主机时钟前跳约 9 小时，
   12 个工作进程的 `deadline` 判定立即过期而退出，量产停在 **250 个事件**（原计划 470）。
   已确认**没有任何事件目录被覆盖**（`check_overwrite.sh`：11:45Z 之后无新的 `frames.npz`）。
   重启时曾一度漏加 `--resume-dir prod12`，在写出任何文件之前被停掉，无损失。
2. **教师策略未纳入**（§1.2 已预注册偏离）。
3. **WSL venv 的 `bin/python` 软链在 02:40Z 前后消失**（`python3`/`python3.10` 未受影响），
   本轮用 `ln -s /usr/bin/python3` 复原后才继续；这是环境异常，不是本轮代码造成的。
4. 第 14 行的四类在线样本见 §3.5（`wp5-summary.json`）。

### 3.5 WP5 在线四类（`wp5-summary.json`）

（由 `aggregate_wp5.py` 从 `run/episodes-*.jsonl` 汇总；数字见 G2 报告 §16.5 与本页 §4）

## 4. V1–V9 自检

| 项 | 自检 |
|---|---|
| V1 预注册 | §1 在运行前写好并冻结；局数、配比、判据、阈值、停机条件全部先定 |
| V2 无泄漏 | 在线回路不训练；CNN 按**整回放**切分（train/val/test 各 751/162/164 shard，无 replay 跨集） |
| V3 样本量 | stage_clear n=250（**未达 386**，如实写）；WP5 四类 n 见 §3.5 |
| V4 下界口径 | 一律 Wilson 95% 单侧下界，零样本不给比率 |
| V5 判据冻结 | stage_clear 判据/阈值/网格与 EXP-P2-009 逐字相同；death CNN 阈值取自 checkpoint |
| V6 合成数据 | 未用任何合成数据补样本；全部为真实引擎帧 |
| V7 失败保留 | 量产 fail-closed 丢弃逐条进 journal；漏检事件逐事件列出 |
| V8 可复现 | `tools/rl/v400_stage_clear_replay.py`、`.tmp_v400_g2online/wp5_online386.py`、`aggregate_wp5.py` 均在案，命令见 §5 |
| V9 结论不越界 | 只声称测到的：第 13、14 行**未达标**，未声称「接近达标」 |

## 5. 证据路径

| 内容 | 路径 |
|---|---|
| 量产产物（四轮并集，250 事件）| `.tmp_v400_sclear2/{prod,prod10,prod12}/events/`、`.tmp_v400_sclear/prod/events/` |
| 合并语料（软链）| `.tmp_v400_sclear2/corpus/` |
| 门（3 折 CV）| `.tmp_v400_sclear2/stage-clear-gate-386.json` |
| 冻结判据打分 | `.tmp_v400_sclear2/stage-clear-frozen.json`、`row13-summary.json` |
| 漏检逐事件诊断 | `.tmp_v400_sclear2/diag_predict.py` |
| 并发三档 | `.tmp_v400_sclear2/concurrency.json` |
| 逐帧 vgrad 流（新产物）| `.tmp_v400_sclear2/prod12/events/*/stream.npz` |
| WP5 逐局记录 | `.tmp_v400_g2online/run/episodes-*.jsonl` |
| WP5 汇总 | `.tmp_v400_g2online/wp5-summary.json` |
| CNN checkpoint | `.tmp_v400_wp5/ckpt2.pt`（6 通道，89,009 参数）、`offline-test-ckpt2.json` |

```bash
# stage_clear：合并语料 → 门 → 冻结判据（全部 WSL）
/root/v400-venv/bin/python .tmp_v400_sclear2/make_union_corpus.py
/root/v400-venv/bin/python tools/rl/v400_p2_reward_detector.py stage-clear-gate \
  --events .tmp_v400_sclear2/corpus --out .tmp_v400_sclear2/stage-clear-gate-386.json
/root/v400-venv/bin/python .tmp_v400_sclear2/frozen_gate.py
# WP5 在线四类（12 分片）
bash .tmp_v400_g2online/run_wp5.sh
/root/v400-venv/bin/python .tmp_v400_g2online/aggregate_wp5.py
```
