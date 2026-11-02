# P0 在帧栅栏变体上的重判（续轮26）

- 日期：2026-11-02。预注册：[p0-fence-round26.md](<../preregistration/p0-fence-round26.md>)（执行前写入；执行中只改过"被打断怎么处理"，未改任何判据）。
- 实际命令：

```powershell
python tools\rl\p0_fence_round26.py --out-dir Document/Update/Alpha/v3.7.3/evidence/p0-fence-round26
python tools\rl\p0_fence_round26_derive.py
python -m pytest touhou_rl/tests/test_p0_fence_round26.py touhou_rl/tests/test_framefence.py -q -p no:cacheprovider --basetemp .tmp_p0fence_pytest
```

- 本轮只做一件事：把 [P0 出口判定](<P0-exit-round15.md>) 冻结的三条欠账（像素同帧 unknown、动力学无 20 次重复、边界各一次）**在带呈现栅栏的变体镜像上重跑**。不训练、不产标签、不改 `capture_dataset.py`、不动 canonical 镜像。

## 结论

**10 条预注册判据 G1–G10 全部通过，P0 的"像素—状态精确同帧"从 `unknown` 变为"在指定变体镜像 + 指定环境指纹下有条件的 `exact`"。**

| 判据 | 结果 | 关键数字 |
|---|---|---|
| G1 ticket 逐帧 +1 | **通过** | demo 239/239、practice 7911/7911 次转移增量恒为 1 |
| G2 逻辑帧与 ticket 同速 | **通过** | 同上，两侧都恒为 1 |
| G3 呈现速率 | **通过** | 59.991 Hz（游戏侧 QPC 计算，240 帧） |
| G4 输入生效为整数帧 | **通过** | C1 20/20、C2 20/20，`lag_in == 1` |
| G5 聚焦/斜向同为整数帧 | **通过** | C3 20/20、C4 20/20，`lag_in == 1` |
| G6 位移不早于输入、不晚于 +1 帧 | **通过** | 80/80，`lag_mv == 2 == lag_in + 1` |
| G7 动力学数值 | **通过** | 4.0 / 4.0 / 2.0 / 2.82843017578125 px 每帧，各 20/20 |
| G8 四边界 | **通过** | 376 / 8 / 16 / 432，各 2 个连续静止帧才判饱和 |
| G9 反向旧方向 | **通过** | 20/20 次"恰好 1 帧、恰好 4 px"；切轴 10/10 旧轴 +4 px 后停 |
| G10 像素—状态同帧 | **通过** | 60/60 样张、13 项检查 × 60 = **780/780**；`pixels_match_backbuffer` 60/60 |
| G11 旧读法对照（测量，非门） | 已测量 | 60 次读取里 2 次跨 ticket（3.3%），摘要不一致 0 次——**本次场景太弱，不能当反证** |

一句话：**"发送 → 引擎掩码 → 位移"现在是 1 + 1 = 2 个整数帧，反向时旧方向恰好多走 1 帧 4 px，四边界与 4/2 px 速度在 20 次独立重复下逐位一致；像素与内存状态在 780 项检查下同帧。**

## 与 v3.7.4 的差别

v3.7.4 证明的是**机制**：attract demo、不发按键、不测边界/动力学、不比对 `read_state`。本轮在同一变体镜像上**重跑 P0 的门**，并且第一次在"自机受控 + 会死亡 + 多局拼接"的条件下测：

| 维度 | v3.7.4 | 本轮（round26） |
|---|---|---|
| 场景 | attract demo（自机无敌） | demo（时钟/像素/旧读法）+ practice（输入/动力学/边界） |
| 按键 | 无 | 80 次试验 + 4 次推边 + 30 次反向/切轴 |
| 状态比对 | 有（13 项） | 有，且 60/60 全过 |
| 边界/动力学 | 未测 | 四边界 + 四类各 20 次独立重复 |
| 跨局 | 单局 | 4 局（自机 18 次被打断后重开，见"失败与重跑"） |

## 实测

镜像身份：`third_party/th08/build_fence/th08.exe`，905,728 B，SHA-256 `BC1894FE…`，`.text` SHA-256 `34A8070B…`；地址表 `frame-fence-addresses.json` SHA-256 `79237A29…`（map `CE7B6672…`）。环境指纹：Windows 11 10.0.26300、i5-14600KF、RTX 4070 SUPER（驱动 617.42）、Python 3.12.10、窗口 640×480、D3D8 `COPY` 呈现、无 vsync。所有数据来自**变体镜像**，不是 canonical。

### M1 时钟（demo，240 帧，不发按键）

| 指标 | 数值 |
|---|---|
| 被 hold 的帧 / 不同 ticket | 240 / 240 |
| 呈现速率（游戏侧 QPC） | **59.991 Hz** |
| ticket 增量 / 逻辑帧增量 | 239/239 恒为 1；239/239 恒为 1 |
| hold 时长（仅 telemetry 读取） | 中位 **20 µs**、最大 34 µs |
| hold 预算耗尽 | M1 段 0 次 |
| 自机状态 / 活跃弹 | `player_state=0` 240/240；弹数 5→653 |

### M5 像素—状态（demo，60 张）

| 指标 | 数值 |
|---|---|
| 接受 / 丢弃 | **60 / 0** |
| 13 项检查 | **780/780 通过**（每张 13 项） |
| `pixels_match_backbuffer`（GDI 截图 == 游戏后台缓冲逐字节） | **60/60** |
| `pixels_frozen`（hold 内两次 BitBlt 相同） | 60/60 |
| 不同 ticket / 不同像素哈希 / 不同 PNG 哈希 | 60 / 60 / 60（无旧画面复用） |
| 读取管线耗时（中位） | 总 8.08 ms = BitBlt 4.62 + 状态 0.79 + 摘要 2.47 + 后台缓冲拷贝 0.17 |
| 场景弹数 | 0–5（本段几乎空场，摘要类检查仍然逐字节比对整池） |

### M2/M3 输入与动力学（practice，80 次独立试验）

| 类 | n | `lag_in` | `lag_mv` | 单帧位移 |
|---|---:|---|---|---|
| C1 横向普通 | 20 | 1（20/20） | 2（20/20） | **4.0**（min=max） |
| C2 纵向普通 | 20 | 1（20/20） | 2（20/20） | **4.0**（min=max） |
| C3 横向聚焦 | 20 | 1（20/20） | 2（20/20） | **2.0**（min=max） |
| C4 斜向普通 | 20 | 1（20/20） | 2（20/20） | **2.82843017578125**（min=max） |

- 斜向实测相对 `4/√2` 偏差 **1.08e-6**，且四类内 20 次逐位相同（浮点完全相等）。
- 位移与 telemetry 报告的该帧速度**逐次相等**（`delta_equals_velocity` 全部为真）。
- 松开按键：**80/80** 次试验在松开后第 2 帧停止位移——与按下侧对称。

**这说明什么**：`发送 → 掩码` 是 1 帧，`掩码 → 位移` 又是 1 帧，所以"发送 → 生效"是 **2 个整数帧**，不是一个区间。旧报告只能写"首次匹配在发送后 1 逻辑帧"，因为当时把"掩码变化"当成了"生效"；本轮把这两件事分开了：

```
帧 F   （hold 内打字）掩码=旧，位移=旧
帧 F+1            掩码=新（lag_in=1），位移=旧
帧 F+2            掩码=新，位移=新（lag_mv=2）
```

### M4 边界与方向保持（practice）

| 方向 | 饱和值 | 期望 | 推到底帧数 | 静止帧数 |
|---|---:|---:|---:|---:|
| x 右 | **376.0** | 376 | 57 | 2 |
| x 左 | **8.0** | 8 | 96 | 2 |
| y 上 | **16.0** | 16 | 99 | 2 |
| y 下 | **432.0** | 432 | 108 | 2 |

反向（4 个方向 × 5 次 = 20 次）：**旧方向恰好再多走 1 帧、恰好 4 px**，20/20 全中，符号随旧方向（向右/向下为 +4，向左/向上为 −4）。切轴（2 组 × 5 次 = 10 次）：旧轴 **+4 px 一帧后停**，新轴从下一帧开始，10/10。

这条旧规则（"切轴时旧方向仍走 1 帧 4 px"，见 [P0-boundaries-round8](<P0-boundaries-round8.md>)）第一次有了逐帧证据，并且解释了它为什么一定是 1 帧：掩码→位移本身就是 1 帧延迟。

### M6 旧读法对照（demo，关闭 hold）

60 次旧读法：跨 ticket 2 次（3.3%），弹池/敌人摘要、玩家位置、帧号不一致 **0 次**。对照 v3.7.4 在 43 发弹的 stage 5 demo 上测到的 20.83% 跨 ticket / 4.17% 弹池摘要不一致，**本轮这个场景（0–5 发弹）太弱，不能当反证，也不能当"旧读法可用"的证据**。摘要只能证伪，这条没有变。

## 三条欠账分别怎么闭合

| 台账行 | 之前 | 本轮证据 | 现在的状态 |
|---|---|---|---|
| 截图与状态精确同帧 | unknown（GDI 无 Present fence） | 60/60 样张 13 项全过，GDI 与后台缓冲逐字节相同 | **判据内 passed**（只在变体镜像 + 本环境） |
| 每类 20 次独立动力学重复 | failed（只有 9–36 个转移） | 四类 × 20 次，位移逐位相同 | **passed** |
| 四边界 | partial（每边界 1 次） | 四方向各 1 次饱和（静态 2 帧）+ 20 次反向/10 次切轴 | **passed**（边界值本身仍只有每次 1 次饱和，见限制） |

## 解冻规则 v2（可执行版）

`frame_proof=exact`（允许像素派生几何标签进入精确监督）**必须同时满足**：

1. 采集来自**帧栅栏变体**，且数据清单记录 `exeSha256`、`.text` SHA、地址表 SHA、map SHA；
2. 每张样张记录 `ticket` 与 `holdPhase==1` 的一对证据（不是"读了摘要就算"）；
3. 每张样张通过全部 13 项检查，其中 `pixels_match_backbuffer` 必须为真；
4. 记录环境指纹（OS 版本、GPU/驱动、窗口模式、客户区尺寸、输入后端）；
5. `frame_proof=exact` **不得**出现在 canonical 镜像或纯 GDI 采集的数据上——那些仍是 `alignment_unknown`。

任何一条不满足 → 退回 `alignment_unknown`，不得用"摘要相符""±1 帧可容忍"替代。

## 补充测量：采样器浪费与修复（同日，轮后）

round26 正式数据用的是默认采样器。上面限制 4 的诊断用一个 A/B 实测确认，并在同一台机器、同一变体镜像、同一 attract demo 上量化：

```powershell
python tools\rl\fence_sampler_throughput.py --samples 60
```

两种模式都**包含与真实采集相同的逐样张 PNG + JSON 落盘**（这是把下一张样张推到 hold 预算里的原因）：

| 指标 | A：等新 ticket（round26 默认） | B：`accept_current=True` |
|---|---:|---:|
| 样张 / 丢弃 | 60 / 0 | 60 / 0 |
| 13 项检查 | **780/780** | **780/780** |
| 60 张总墙钟 | 15.78 s | **1.33 s** |
| 吞吐 | 3.8 张/s | **45.2 张/s（11.9×）** |
| `wait` 中位 / 最大 | 232.9 ms / 242.9 ms | **3.1 µs / 4.9 µs** |
| hold 预算耗尽 | **59** | **0** |
| 不同 ticket / 不同像素哈希 | 60 / 60 | 60 / 60 |

A 模式复现了正式轮的 235 ms / 60 次预算耗尽（诊断成立）；B 模式把它们压到微秒级并且**一项检查都没少**。B 的上限不再是 hold，而是读取管线 8.16 ms + I/O 13.3 ms ≈ 21 ms/张，即"每帧都采样"的理论上限约 45–60 张/s。

注意：这是**工程项**，不是判据项——两种模式的样张都通过全部 13 项检查，`frame_proof=exact` 的成立条件一个字没变。默认行为（不传 `accept_current`）保持原样，round26 正式数据仍由默认模式产生。

## 失败与重跑（必须与结论同读）

本轮正式数据之前有三次未完成执行，全部保留在 `evidence/p0-fence-round26/attempt-{1,2,3}/`：

| 尝试 | 结束原因 | 处理 |
|---|---|---|
| attempt-1 | 旧规则"一次被打中就整场中止"：practice 第 8 次被打中后生命耗尽，只完成 C1 20/20、C2 16/16 | 预注册改为"打断只作废当次试验并重做"，判据不变 |
| attempt-2 | 重构后忘记在进入游戏后 arm holds，每帧等 hold 超时 3 s（19 分钟只写出空 trials） | 修 `_start_practice_game` 补 arm，并加"连续 3 次空试验即中止"与单局 240 s 预算 |
| attempt-3 | `gap_summaries` 未初始化，practice 起步即 `NameError` | 修后重跑 |
| **attempt-4（正式）** | 全部 114 项测量完成 | 本报告所有数字 |

正式轮 practice 共 4 局：18 次打断（第 1 局 8 次、第 2 局 7 次、第 3 局 3 次），每次打断作废当次试验并重做，逐条记在 `summary.json → verdict.interruptions`。**没有任何一次试验被打断后"补成通过"**：每类仍需 20 次完整试验才计入判据。

## 残留限制（必须一起读）

1. **不是 canonical 镜像**：exact-match / 版本一致性 / canonical SHA 声明都不适用于 `build_fence/th08.exe`；变体全局地址整体 `+0x3000`，读取必须用变体地址表。
2. **"窗口 DC == 后台缓冲"仍是环境结论**：本次为 D3D8 `COPY`、无 vsync、窗口 640×480、未最小化/未被 DWM cloak。换驱动、换呈现模式、缩放窗口都要重测。
3. **hold 成本分两种**：M1 只读 telemetry 时中位 **20 µs**；M5 完整取样（两次 BitBlt + 状态 + 摘要 + 后台缓冲）中位 **8.08 ms**。长期采集吞吐要按后者估。
4. **M5 采样器每次烧掉一个 hold 预算——已定位并修复，收益已实测**：round26 的 60 次 `holdBudgetSkips`、`wait` 中位 235 ms，原因是 `FenceFrameSampler` 在 ack 之后才进入等待，撞上游戏已经在等的下一帧 hold，只能等它 250 ms 预算到期。**不影响同帧性**（13 项检查全过）。修法：新增 `accept_current=True`（消费"当前正在 hold 的那一帧"而不是"等新的一帧"），默认关闭、行为不变，单测覆盖。实测见下节。
5. **M6 对照太弱**：场景 0–5 发弹，摘要几乎不变；"旧读法仍然 unknown"这条要靠 v3.7.4 的 43 发弹对照，本轮不提供新反证。
6. **边界值每次只有 1 次饱和**：四方向各 1 次（静止 2 帧判定），反向/切轴才有 20/10 次重复。把边界值当"标定模型"仍然不成立，只能当已观测样本。
7. **练习局靠重开续命**：stage 1 Lunatic 对一个近乎静止的自机约 10 秒就打死一次；114 项测量用了 4 局共 7,915 个被 hold 的帧（约 132 秒游戏时间）。换关卡/难度会改变打断分布，但不改变判据。
8. **未接入正式采集**：`capture_dataset.py` 未改，`frame_proof=exact` 字段尚未预注册进数据集清单。本轮只给出解冻规则，不宣布任何数据集合格。
9. **G11 不是门**：它是测量；本轮数字（3.3% 跨 ticket）不能用来论证旧读法好坏。

## 复现

```powershell
# 1) 变体构建 + 地址表（不触碰 canonical 产物）
cd third_party\th08
python scripts\build.py --build-type fence -j 1
python scripts\fence-addresses.py build_fence\th08.map build_fence\frame-fence-addresses.json
cd ..\..

# 2) 本轮（两次启动：attract demo + practice，自动重开至多 5 局）
python tools\rl\p0_fence_round26.py --out-dir Document/Update/Alpha/v3.7.3/evidence/p0-fence-round26
python tools\rl\p0_fence_round26_derive.py

# 3) 不启动游戏的单元测试
python -m pytest touhou_rl/tests/test_p0_fence_round26.py touhou_rl/tests/test_framefence.py -q -p no:cacheprovider
```

运行会夺取前台焦点（菜单只吃焦点窗口的按键），并在结束时关闭自己启动的游戏；不读取用户会话、不改游戏进程内存。

## 证据索引

| 主题 | 文件 |
|---|---|
| 全部判据与元数据 | [summary.json](<../evidence/p0-fence-round26/summary.json>) |
| 报告用的派生数字 | [derived.json](<../evidence/p0-fence-round26/derived.json>) |
| M1 逐帧 telemetry（240 帧） | [trace.jsonl](<../evidence/p0-fence-round26/trace.jsonl>) |
| M5 样张记录（60 条，含 13 项检查与 PNG SHA） | [samples.jsonl](<../evidence/p0-fence-round26/samples.jsonl>)、`samples/*.png` |
| M6 旧读法（60 条） | [legacy.jsonl](<../evidence/p0-fence-round26/legacy.jsonl>) |
| 80 次试验原始帧序列 | [trials.json](<../evidence/p0-fence-round26/trials.json>) |
| 四次推边原始帧序列 | [boundaries.json](<../evidence/p0-fence-round26/boundaries.json>) |
| 反向/切轴 30 次原始帧序列 | [carry.json](<../evidence/p0-fence-round26/carry.json>) |
| 采样器 A/B 吞吐（含逐样张 PNG） | [sampler-throughput/summary.json](<../evidence/p0-fence-round26/sampler-throughput/summary.json>) |
| 三次未完成执行 | `attempt-1/`、`attempt-2/`、`attempt-3/` |

## 本轮改动清单

- 新增：`tools/rl/p0_fence_round26.py`、`tools/rl/p0_fence_round26_derive.py`、`tools/rl/fence_sampler_throughput.py`、`touhou_rl/tests/test_p0_fence_round26.py`、`Document/Update/Alpha/v3.7.3/preregistration/p0-fence-round26.md`、本报告。
- 修改：`touhou_rl/framefence.py`（`FenceFrameSampler` 新增可选 `on_hold` 钩子与 `accept_current` 选项；默认行为不变，单测覆盖"钩子在 hold 内被调用且 ack 在其后""默认仍等新 ticket""accept_current 消费当前 hold 且不复用已消费 ticket"）。
- 未改动：`third_party/th08/build/th08.exe` 字节、`capture_dataset.py`、任何训练/标签产物、canonical 地址常量。
