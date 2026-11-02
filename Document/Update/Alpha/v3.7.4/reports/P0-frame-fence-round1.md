# P0 同帧证明：帧栅栏变体（round1）

## 结论

**"画面与内存状态是否同一帧"从"软件层证明不了"变成了"游戏内部显式契约 + 可复核"。** 采用 `DATA-ACQUISITION.md` 第 5 节的方案 B（研究变体构建），但不再依赖外部调试器或时序估计：游戏在每个**已呈现**帧的 `Present()` 之后发布 ticket + 状态摘要，并可**在栅栏内自旋等待读取方 ack**。hold 期间游戏不推进任何逻辑帧、不产生新 present，因此此窗口内的 `BitBlt` 与 `ReadProcessMemory` 属于同一帧；游戏同时发布它自己算的弹池/敌人摘要和后台缓冲副本，读取方用自己的读取复算校验，而不是相信时间假设。

实测（本机，stage 5 attract demo，240 帧）：**240/240 帧全部通过 13 项同帧检查（3120/3120）**；**240/240 张 GDI 截图与游戏自读回的后台缓冲逐字节相同**；240 帧互不相同（没有旧画面复用）；保证期间仍以 **59.87 Hz** 呈现（hold 中位 7.44 ms、最大 8.97 ms，0 次预算超时）。对照组（不 hold、沿用旧读法）120 次读取里 **20.83% 跨了 ticket**，其中**至少 4.17% 的弹池读取可以证明不是该帧的状态**。

## 机制：三层，缺一不可

| 层 | 做什么 | 为什么需要 |
|---|---|---|
| **ticket** | `GameWindow::Render` 在 `Present()` 返回后、且该次确实跑了绘制链时，递增 ticket 并把 `gameplayFrameCounter`、玩家位置/速度/状态、focus、输入、活跃弹数写进共享节 | 给"这一帧"一个单调编号；只有 present 之后才发布，所以客户区内容必然是该帧 |
| **hold** | 读取方请求后，游戏写完 telemetry、置 `holdPhase=1`、自旋到 `ackTicket==ticket` 或预算耗尽 | 消除 GDI "没有呈现栅栏" 与读取非原子：游戏停住后，截图与内存读取天然同一帧 |
| **复核** | 游戏按自己的内存算 FNV-1a 摘要（子弹池 `0x601` 槽的 state/x/y、敌人数组 481 槽的 flags1/life/x/y），并把后台缓冲按 RGB24 复制一份 | 读取方用**自己的** `ReadProcessMemory`/`BitBlt` 结果复算；摘要/像素不一致就说明不是同一帧，不必信任时序 |

共享节：`Local\th08.frame_fence.v1`（控制 0x000 / telemetry 0x020 / 像素 0x1000，共 925,696 B）。**游戏不写游戏状态；读取方也不写游戏进程**——ack 与控制字走共享节，`ReadProcessMemory` 依旧是只读。

同帧判定读到的是 `ticket == T && holdPhase == 1` 这一对：ticket 是最后一个被写的字段，且写完即进入 hold，所以一旦观察到这对值，T 的全部 telemetry 都已落盘、游戏已停住。读取方在最后一次读之后**再次**确认这对值仍然成立才 ack；若预算耗尽（游戏已继续）则整帧丢弃重来，绝不放宽。

## 与 canonical 镜像的隔离

- 新增 `BuildType.FENCE`，输出到 `build_fence/`（已加入 `.gitignore`），编译定义 `/DFIX_REALLY_BAD_BUGS /DTH08_FRAME_FENCE`。**不是重建产物，不能支撑 exact-match 声明。**
- `src/main.cpp` 只插入三处 `#ifdef TH08_FRAME_FENCE` 块；canonical 构建里它们被预处理掉。重建后 `.text` SHA 仍为 `7B5B3C8AB5023BFA7960DA309338A01982F651DC6B1E7734BEE92437202ED6DD`，与 `build/play/th08.exe`（`CDD23BCA…`）比对：**整文件只差 PE 链接时间戳的 3 字节（0x118–0x11A）**，其余字节完全相同；已把 `build/th08.exe` 恢复成原字节，文档中的 SHA 继续成立。
- 变体多一个 TU 使 `.text`/`.rdata` 增长，**所有全局地址整体 +0x3000**（结构偏移不变）。因此新增 `scripts/fence-addresses.py` 从链接 map 生成地址表，`touhou_rl/spec.py` 支持 `TH08_ADDRESS_FILE` 覆盖；用 canonical map 生成的表与 `spec.py` 的 13 个硬编码地址**逐项一致**（已写成单元测试）。
- 踩坑记录：变体最初用 `normal` 语义构建，启动即 AV。原因不是 fence 代码，而是**非 bugfix 的重建镜像本身不可运行**（`Supervisor::CheckVersion` 路径）。fence 必须叠在 bugfix 上；这条已写进 `docs/FRAME_FENCE.md`。

## 实测

镜像身份：`build_fence/th08.exe` 898,048 → 905,728 B，SHA-256 `BC1894FE…`，`.text` SHA-256 `34A8070B…`；地址表 `build_fence/frame-fence-addresses.json` SHA-256 `79237A29…`。会话客户端 640×480，输入后端 PostMessage（本轮**没有发送任何按键**，实验用的是标题画面 attract demo，stage 5、`isDemo=1`、`isReplay=1`，暖机后活跃弹 43）。

| 指标 | 主轮（240 帧） | 复现轮（120 帧） |
|---|---:|---:|
| 通过全部 13 项检查 | **240/240** | **120/120** |
| player 位置/速度/状态/focus | 100% | 100% |
| `gameplayFrame` == ticket 的帧号 | 100% | 100% |
| 弹池摘要（`0x601` 槽） | 100% | 100% |
| 敌人摘要（481 槽） | 100% | 100% |
| 输入/活跃弹数/stage 帧 | 100% | 100% |
| hold 期间两次截图相同（像素冻结） | 100% | 100% |
| **GDI 截图 == 后台缓冲（逐字节）** | **240/240** | **120/120** |
| 去重后的不同帧数 | **240** | **120** |
| hold 中位 / p95 / 最大 (µs) | 7442 / 8001 / 8969 | 7883 / 10578 / 90000（1 次预算耗尽） |
| 整条读取管线（中位 µs） | 7446 | 7879 |
| 其中 BitBlt 中位 / 状态读取 / 摘要 | 4052 / 747 / 2448 | 4195 / 763 / 2708 |
| ticket 增量恒为 1 | 是 | 否（漏了 1 个 hold，见下） |
| 实测呈现速率 | **59.87 Hz** | 56.88 Hz |

复现轮里出现 **1 次 hold 预算耗尽（90 ms）**：读取方（Python）被调度走，那一个 hold 没有等到 ack 就超时；于是 ticket 增量出现 2。这**不是**同帧性失败——被采样的帧仍然全部通过 13 项检查；它只说明"读取方没接住的帧不会被伪造"，ticket/`logicFrame` 增量与 `holdBudgetSkips` 把丢帧**显式暴露**出来，而不是悄悄混进数据。

> 表里的 hold 数值取自 telemetry 的 `holdLastUs`，即**读取时刻刚结束的那一次 hold**（游戏在 ack 之后才写这个字段）。逐帧连续 hold 时它与本次 hold 等价：读取方管线耗时中位数 7446 µs 与它 7442 µs 基本一致，可互相印证。

### 对照组：旧读法（不 hold、不钉帧）

同一进程、同一场景，关闭 hold，用 `legacy_sample` 复刻现有管线（先读 telemetry → 再 `read_state` + 6.3 MB 弹池 + 10 MB 敌人数组 → 复算摘要）：

| 指标 | 主轮（120 次） | 复现轮（60 次） |
|---|---:|---:|
| 读取期间 ticket 已变化 | 20.83% | 23.33% |
| 弹池摘要与起始 telemetry **不一致** | **4.17%** | **8.33%** |
| 敌人摘要不一致 | 0% | 18.33% |
| 玩家位置不一致 | 0.83% | 3.33% |
| `gameplayFrame` 不一致 | 0.83% | 6.67% |

**这些比例是下界，不是"错误帧比例"**：摘要只能**证伪**。旧读法即使在摘要上碰巧相符（场景里弹池一帧内没变、敌人几乎不动），也**无法证明**自己读到的是哪一帧——这正是 `DATA-ACQUISITION.md` 第 4 节说的"unknown"。敌人摘要在主轮 100% 相符，恰好说明**弱摘要不能当同帧证据**。

## 为什么这解决了问题

1. **像素归属**：ticket 只在 `Present()` 返回后发布；本次实测该窗口的客户区像素与游戏自读回的后台缓冲逐字节相同（240/240），因此"GDI 抓到的是不是刚呈现的那一帧"从猜测变成**可判定**（相同即同一帧；不同即丢弃重采）。
2. **状态原子性**：hold 期间无逻辑推进、无 present，`read_state`（含 6.3 MB 弹池与 10 MB 敌人数组）是**同一时刻**的完整快照；`holdPhase` 复核把"预算到期导致的半帧"排除掉。
3. **独立复核**：摘要由游戏算、读取方复算，两侧算法与字节序在代码与测试里固定；不依赖"发送→生效约 1 帧"之类的时序标定。
4. **可观测的失败**：ticket 增量、`logicFrame` 增量、`holdBudgetSkips`、`hold_stable`、`pixels_frozen` 都是显式字段，丢帧/过期一律可见，不会被当成有效样本。

## 残留限制（必须一起读）

- **变体镜像不是 canonical**：exact-match、版本一致性、canonical SHA 声明都不适用于 `build_fence/th08.exe`；换构建/改源码后地址表必须重新生成（工具已提供）。要用于正式数据，必须把"用的是哪个变体镜像 + 地址表哈希"写进数据清单。
- **hold 占用帧时间**：本次 hold 中位 7.44 ms，仍在 16.7 ms 预算内（59.87 Hz）。若读取方更慢，游戏会降速或跳帧；生产采集应用 `holdEvery` 降频，或把 10 MB 敌人摘要移出逐帧路径。**本轮只验证 hold 成本，没有做长期吞吐标定。**
- **D3D8 `Present()` 返回 ≠ 显示管线完成**：本次"窗口 DC 与后台缓冲逐字节相同"是**当前 GPU/驱动/窗口模式下的实测结论**（`COPY` swap effect、无 vsync）。换机器或改呈现模式需要重测，不能当成跨平台保证。
- **像素相等只覆盖客户区 640×480**：没有覆盖窗口缩放、遮挡、DPI 虚拟化等场景；采集时必须保持"窗口未最小化、未被 DWM cloak"，否则旧有的 52,888 张重复图事故仍会重现（本轮的 240 帧全不同，但那是本次条件）。
- **摘要只能证伪**：即使 hold 关闭，摘要相符也不代表同帧（敌人摘要已给出反例）。只有 `ticket + holdPhase` 组合构成证明。
- **未接入训练/标签管线**：本轮只交付机制、验证与工具，`capture_dataset.py` 等仍未改用 fence；把 `frame_proof=exact` 字段写进正式数据集是下一步的工作，且需要新的预注册。可直接使用的采集入口：[frame_fence_capture.py](<../../../../../tools/rl/frame_fence_capture.py>)（PNG + `capture.jsonl` + `manifest.json`，每帧带 ticket/逻辑帧/位置/摘要/校验结果；实测 60 帧、0 丢弃、780 项校验全过）。

## 复现

```powershell
# 1) 变体构建（不触碰 canonical 产物）
cd third_party\th08
python scripts\build.py --build-type fence -j 1
python scripts\fence-addresses.py build_fence\th08.map build_fence\frame-fence-addresses.json

# 2) 实验（自动启动游戏、等 attract demo、hold 采集 + 旧读法对照）
cd ..\..
python tools\rl\frame_fence_round1.py --frames 240 --legacy-frames 120 --warmup-seconds 12 `
  --out-dir Document/Update/Alpha/v3.7.4/evidence/frame-fence-round1

# 3) 单元测试（不启动游戏）
python -m pytest touhou_rl/tests/test_framefence.py -q -p no:cacheprovider

# 4) canonical 未被改动的证据
cd third_party\th08
python scripts\build.py --build-type bugfix -j 1   # .text = 7B5B3C8A…，整文件仅差 PE 时间戳
```

## 证据索引

| 主题 | 文件 |
|---|---|
| 主轮完整记录（逐帧检查、耗时、ticket、sha256） | [frames.json](<../evidence/frame-fence-round1/frames.json>) |
| 主轮汇总 | [summary.json](<../evidence/frame-fence-round1/summary.json>) |
| 复现轮汇总 | [replication/summary.json](<../evidence/frame-fence-round1/replication/summary.json>) |
| 同一帧的 GDI 截图 / 游戏后台缓冲（两张 PNG 的 SHA-256 相同） | [frame-1904-gdi.png](<../evidence/frame-fence-round1/frame-1904-gdi.png>)、[frame-1904-backbuffer.png](<../evidence/frame-fence-round1/frame-1904-backbuffer.png>) |
| 游戏侧实现与协议 | [framefence.cpp](<../../../../../third_party/th08/src/framefence.cpp>)、[framefence.hpp](<../../../../../third_party/th08/src/framefence.hpp>)、[FRAME_FENCE.md](<../../../../../third_party/th08/docs/FRAME_FENCE.md>) |
| 变体构建接入 | [configure.py](<../../../../../third_party/th08/scripts/configure.py>)、[build.py](<../../../../../third_party/th08/scripts/build.py>)、[fence-addresses.py](<../../../../../third_party/th08/scripts/fence-addresses.py>) |
| 读取端 | [framefence.py](<../../../../../touhou_rl/framefence.py>)、[spec.py](<../../../../../touhou_rl/spec.py>)（`TH08_ADDRESS_FILE`） |
| 实验驱动 | [frame_fence_round1.py](<../../../../../tools/rl/frame_fence_round1.py>) |
| 精确样本采集工具（PNG + JSONL + manifest） | [frame_fence_capture.py](<../../../../../tools/rl/frame_fence_capture.py>) |
| 单元测试（9 项） | [test_framefence.py](<../../../../../touhou_rl/tests/test_framefence.py>) |

## 本轮改动清单

- 新增：`third_party/th08/src/framefence.{hpp,cpp}`、`third_party/th08/scripts/fence-addresses.py`、`third_party/th08/docs/FRAME_FENCE.md`、`touhou_rl/framefence.py`、`touhou_rl/tests/test_framefence.py`、`tools/rl/frame_fence_round1.py`、`tools/rl/frame_fence_capture.py`。
- 修改：`third_party/th08/src/main.cpp`（3 处 `#ifdef`）、`third_party/th08/scripts/{configure.py,build.py}`（新增 `fence` 构建类型）、`third_party/th08/.gitignore`（忽略 `build_fence/`）、`third_party/th08/docs/RE_HANDOFF.md`（指针）、`touhou_rl/spec.py`（地址表覆盖，默认行为不变）。
- 未改动：`third_party/th08/build/th08.exe` 的字节（已恢复原 SHA）、任何训练/标签产物、`capture_dataset.py` 采集管线。
