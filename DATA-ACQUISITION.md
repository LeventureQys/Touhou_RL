# 真机数据获取能力说明（读本项目前先看这篇）

> 一句话：**画面能抓、内存能读、按键能发，这三件事都已经在本仓库实现并实测过；"抓到的画面与读到的状态是不是同一帧"原先在软件层面证明不了，现在已由帧栅栏（frame fence）研究变体解决——游戏在 Present 之后发布 ticket 并可停在栅栏内等待读取方确认，240/240 帧实测通过 13 项同帧检查、GDI 截图与游戏后台缓冲逐字节相同。** 该能力只存在于非 canonical 的变体镜像（见第 5 节），canonical 镜像字节未变。

本文只讲"能拿到什么数据、怎么拿、边界在哪"，不需要读其它文档。所有数字都有落盘证据，链接在文末。

---

## 1. 结论表

| 能力 | 能不能做 | 现状 | 证据 |
|---|---|---|---|
| 读游戏内存（只读，不注入、不改字节） | **能** | 已实现并长期使用；本会话还修好了输入全局变量的偏移错误 | [memory.py](<touhou_rl/memory.py>)、[state.py](<touhou_rl/state.py>)、[输入纠错报告](<Document/Update/Alpha/v3.7.3/reports/P0-input-readback.md>) |
| 抓游戏实时画面 | **能** | 已验证 640×480 客户区截图，290 张连续采样，单帧 2.46 ms（中位） | [winapi.py](<touhou_rl/winapi.py>)、[连续采样报告](<Document/Update/Alpha/v3.7.3/reports/P0-capture-round5.md>) |
| 发按键 / 读回游戏实际消费的输入 | **能** | SendInput 与 PostMessage 两种后端；实测"发送→生效"通常滞后 1 逻辑帧 | [session.py](<touhou_rl/session.py>)、[两帧动作与释放实测](<Document/Update/Alpha/v3.7.3/reports/P0-motion-round4.md>) |
| 起停游戏、进练习关、重置世界 | **能** | 已有完整会话管理（启动→标题→菜单→实战） | [session.py](<touhou_rl/session.py>)、[env.py](<touhou_rl/env.py>) |
| **证明画面与内存状态同帧** | **能**（仅变体镜像） | 帧栅栏：Present 后 ticket + hold + 游戏侧摘要/后台缓冲复读；240/240 帧全检查通过 | [帧栅栏报告](<Document/Update/Alpha/v3.7.4/reports/P0-frame-fence-round1.md>)、[framefence.cpp](<third_party/th08/src/framefence.cpp>)、[framefence.py](<touhou_rl/framefence.py>) |
| 用 canonical 镜像证明同帧 | **不能** | GDI 截图没有"呈现栅栏"；帧计数在自机更新之前 | [截图新鲜度审查](<Document/Update/Alpha/v3.7.3/reports/P0-capture-round4.md>)、[对齐分析](<Document/Update/Alpha/v3.7.3/reports/P0-alignment-round7.md>) |

---

## 2. 内存读取：怎么读、读到什么

**方法**：`OpenProcess` + `ReadProcessMemory`，模块基址由 Toolhelp32 快照解析，**只读**；不写内存、不改代码、不注入 DLL（[memory.py](<touhou_rl/memory.py>)）。帧栅栏的 ack/控制字写在游戏自己创建的共享节里，走的是 `MapViewOfFile` 而不是 `WriteProcessMemory`，读取方对游戏进程**仍然没有任何写权限**。

**能读到的字段**（[state.py](<touhou_rl/state.py>)）：

| 类别 | 内容 |
|---|---|
| 时钟 | 逻辑帧计数、关卡帧、当前 stage、supervisor 状态、是否 demo/replay |
| 自机 | 位置、速度、状态（alive/dying/spawn/invuln）、焦点模式、判定点半径 |
| 资源 | 残机、炸弹、power、得分、graze、符卡状态 |
| 弹幕 | 完整弹池（容量 `0x601` 槽），每槽位置/速度/尺寸/状态/禁用标志 |
| 敌人 | 敌人数组（类型、位置、HP、旗标）、Boss 指针、Boss HP |
| 掉落物 | item 数组 |
| 输入 | 引擎侧当前/上一帧输入、GUI 侧当前/上一帧输入（即 Player 真正消费的那一路） |
| 回放 | ReplayManager 输入游标（用于回放播放与真值对照） |

**本会话的实测纠正**：原先 `gameplay input` 与 `GUI previous` 两个字段按 +2 字节读取，实际全局变量之间是 +4 间距，导致读到的其实是 padding——3625 次轮询里输入位恒为 0。修复后实测：引擎侧输入与 GUI 侧 5161 次轮询里 5153 次一致，差异是帧内更新相位。**这类错误说明"读到 0"必须当成可疑，而不是当成事实。**

---

## 3. 画面抓取：怎么抓、有多快

**方法**：`GetDC(hwnd)` + `BitBlt` 到自建 DIB（32 位、top-down），返回 `uint8[H, W, 3]` RGB；ROI 按原生分辨率裁剪（[winapi.py](<touhou_rl/winapi.py>)、[vision.py](<touhou_rl/vision.py>)）。

**已实测**：

| 项目 | 数字 |
|---|---|
| 客户区 | 640×480（原生） |
| ROI | playfield (32,16,384,448) / hud_score (424,36,216,34) / hud_status (424,88,216,126) |
| 单帧抓取耗时 | 中位 2.46 ms，最大 6.56 ms（含 RGB 复制） |
| 连续采样 | 290 张 / 122 个逻辑帧（约 120 Hz 观察，游戏 60 Hz） |
| 全黑帧 | 0 |

**已知风险**（同一个文件里有历史事故记录）：窗口不被合成时（锁屏、显示器关闭、最小化），`GetDC` 会一直返回**最后一帧旧画面**，而 `BitBlt` 仍然报告成功。仓库曾经因此产出 52,888 张完全相同的图（占某数据集的 73%）。现在的代码会检查最小化/被遮挡，并把重复检测写进采集流程，但**这仍然不能证明"这一帧是新的"**。

---

## 4. 为什么"同帧"证明不了（canonical 镜像）

三条独立的事实：

1. **渲染顺序**：主循环先跑计算链（逻辑帧在此推进），再按 frameskip 决定是否绘制，最后才 `Present`（[main.cpp](<third_party/th08/src/main.cpp#L338-L368>)）。所以"逻辑帧号 = F"时，像素可能还是 F−1 甚至更早。
2. **读取非原子**：`read_state` 先读帧计数、再读自机位置（两次 `ReadProcessMemory`），中间游戏可能已经前进一帧。
3. **GDI 无栅栏**：`BitBlt` 只是从窗口 DC 复制，不触发也不等待 D3D 的 Present；`GdiFlush` 只能保证 GDI 批次完成，不是呈现完成。

实测后果：289 张截图里，只有 66 张能得到完整的"像素位置 ↔ 内存轨迹"候选集合，其余 223 张只能标 `unknown`；候选里还出现"同一位置相隔 8 帧"的多解。结论是：**canonical 素材足够做"这一刻大概在哪"，不够做"这一像素属于哪一逻辑帧"的精确监督。**

### 4.1 现在怎么解决：帧栅栏（研究变体，已实测）

同样的三条事实在**变体镜像**里被游戏自己消除（[帧栅栏报告](<Document/Update/Alpha/v3.7.4/reports/P0-frame-fence-round1.md>)、实现 [framefence.cpp](<third_party/th08/src/framefence.cpp>)、说明 [FRAME_FENCE.md](<third_party/th08/docs/FRAME_FENCE.md>)）：

- **ticket**：`GameWindow::Render` 在 `Present()` 返回后（且该次确实跑了绘制链）把 ticket、`gameplayFrameCounter`、自机位置/速度/状态、输入、活跃弹数写进命名共享节 `Local\th08.frame_fence.v1`。
- **hold**：读取方请求后游戏置 `holdPhase=1` 并在栅栏内自旋，直到读取方 ack 或预算耗尽。hold 期间**不推进逻辑帧、不产生新 present**，所以此窗口内的 `BitBlt` 与 `ReadProcessMemory` 必然同帧；读取方读完最后一次读后复核 `ticket+holdPhase` 仍成立才 ack。
- **复核**：游戏按自己的内存算 FNV-1a 摘要（子弹池 `0x601` 槽、敌人数组 481 槽）并把后台缓冲复制成 RGB24；读取方用自己的读取/截图复算，摘要或像素不一致就丢帧，不靠时序推断。

实测（stage 5 attract demo，240 帧）：**240/240 帧通过全部 13 项同帧检查（3120/3120）**；**240/240 张 GDI 截图与游戏后台缓冲逐字节相同**，240 帧互不相同；保证期间 **59.87 Hz**（hold 中位 7.44 ms、最大 8.97 ms，0 次预算超时）。对照组（不 hold、沿用旧读法）120 次读取中 20.83% 跨 ticket、**至少 4.17% 的弹池读取可证明不是该帧**。

**注意**：该能力来自非 canonical 变体镜像，不能支撑 exact-match/版本一致性声明；canonical 镜像字节未变（canonical `.text` SHA-256 `7B5B3C8A…`，整文件仅 PE 时间戳不同）。

---

## 5. 四条路的现状（B 已落地）

| 方案 | 做法 | 代价 | 现状 |
|---|---|---|---|
| A. 调试器断点 | 在 `Present` 之后设断点；命中时读状态 + 抓图 | 附加调试器改变帧时序，需单独标定 | 未采用：hold 已提供等价同步点，且没有调试器改时序的问题 |
| **B. 研究变体构建（已实现）** | 重建源码加 Present 后栅栏 + hold + 摘要/后台缓冲复读，另建非 canonical build | 变体不能用于 exact-match；全局地址整体平移，需要变体地址表 | **已落地**：[FRAME_FENCE.md](<third_party/th08/docs/FRAME_FENCE.md>)、[报告](<Document/Update/Alpha/v3.7.4/reports/P0-frame-fence-round1.md>) |
| C. 桌面合成抓取 | DWM/DXGI 抓合成后桌面 + 高精度时间戳 | 仍拿不到游戏内部帧号 | 不需要：本变体的像素与后台缓冲逐字节相同，已直接给出归属 |
| D. 接受 ±1 帧 | 标签里做误差膨胀 | 最弱 | 不需要（只在无法使用变体时作为退路） |

注意：**注入 DLL / hook `Present` / 改目标字节都不在允许范围内**（重建仓库自身规则也禁止改目标字节）。B 是另建变体并明确标注非 canonical；读取方也**不写游戏进程**（ack 走共享节，`ReadProcessMemory` 保持只读）。

---

## 6. 所以现在能产出什么、不能产出什么

**可以**（已有实现）：
- 逐帧的**游戏状态轨迹**（位置、弹池、敌人、资源、时钟）→ 用于教师、参考、物理校验。
- **动作与状态配对**（回放驱动或真机发送，含实际 hold 帧数、输入读回区间）。
- **截图与状态并存的样本**，其中同步性标 `unknown` → 用于视觉表征、目标检测、位置回归等**容错**任务。
- 边界/动力学标定：x[8,376]、y[16,432]、普通 4 px/帧、聚焦 2 px/帧、"发送→生效"约 1 帧。
- **变体镜像上的同帧样本**：像素 ↔ 状态 ↔ ticket ↔ 逻辑帧号，带游戏侧摘要可复核；读取端 [framefence.py](<touhou_rl/framefence.py>) 已实现 hold 协议与 13 项校验，采集入口 [frame_fence_capture.py](<tools/rl/frame_fence_capture.py>)（PNG + JSONL + manifest，实测 60 帧 0 丢弃）。**尚未接入 `capture_dataset.py`**，正式数据集需要新的预注册。
- **该能力已用于 P0 重判（2026-11-02）**：[round26 报告](<Document/Update/Alpha/v3.7.3/reports/P0-fence-round26.md>) 在同一变体镜像上重跑 P0，10 条预注册判据全过（60/60 样张 780/780 检查、ticket/logic 帧逐帧 +1、发送→掩码 1 帧、掩码→位移 1 帧、四类动力学各 20/20 逐位相同、四边界 376/8/16/432），并给出 `frame_proof=exact` 的解冻规则 v2。**采样器浪费也已修**：默认"等新 ticket"模式在逐样张落盘时只有 3.8 张/s（`wait` 中位 232.9 ms、60 张烧掉 59 个 hold 预算），`accept_current=True` 后 **45.2 张/s（11.9×）**、预算耗尽 0、13 项检查仍 780/780。

**不可以**：
- 用 canonical 镜像把截图当作"精确几何标签"（例如"这一帧里这颗子弹的准确坐标"）用于训练。
- 在变体镜像上做 exact-match / 版本一致性声明（变体多一个 TU，全局地址整体 +0x3000）。
- 用旧缓存冒充完整同步弹池（历史上有 64 弹截断的缓存，不能当危险真值）。
- 把"摘要相符"当作同帧证据：摘要只能**证伪**（实测对照组里敌人摘要在不 hold 时也会 100% 相符）。

---

## 7. 一句话复述

**读内存、抓画面、发按键都已经跑通；"两份数据是不是同一帧"也不再是空口：变体镜像里游戏自己在 Present 后发 ticket、可停下来等读取方确认，并交出它算的摘要与后台缓冲供复核，240/240 帧实测全中。** 剩下的工作是把这条能力接进正式采集与标签预注册，而不是继续讨论 ±1 帧。

---

## 8. 证据索引

| 主题 | 文件 |
|---|---|
| 内存读取实现 | [touhou_rl/memory.py](<touhou_rl/memory.py>)、[touhou_rl/state.py](<touhou_rl/state.py>) |
| 截图与输入后端 | [touhou_rl/winapi.py](<touhou_rl/winapi.py>)、[touhou_rl/session.py](<touhou_rl/session.py>) |
| 视觉 ROI 与缩放 | [touhou_rl/vision.py](<touhou_rl/vision.py>) |
| 输入读回纠错（含 3625 次恒 0 的证据） | [P0-input-readback.md](<Document/Update/Alpha/v3.7.3/reports/P0-input-readback.md>) |
| 发送/释放/边界实测 | [P0-motion-round4.md](<Document/Update/Alpha/v3.7.3/reports/P0-motion-round4.md>)、[P0-boundaries-round8.md](<Document/Update/Alpha/v3.7.3/reports/P0-boundaries-round8.md>) |
| 连续采样（290 张、2.46 ms） | [P0-capture-round5.md](<Document/Update/Alpha/v3.7.3/reports/P0-capture-round5.md>) |
| 截图新鲜度源码审查 | [P0-capture-round4.md](<Document/Update/Alpha/v3.7.3/reports/P0-capture-round4.md>) |
| 候选帧集合与 unknown 分析 | [P0-alignment-round7.md](<Document/Update/Alpha/v3.7.3/reports/P0-alignment-round7.md>) |
| P0 出口判定与冻结规则 | [P0-exit-round15.md](<Document/Update/Alpha/v3.7.3/reports/P0-exit-round15.md>) |
| **帧栅栏：机制、实测、复现** | [P0-frame-fence-round1.md](<Document/Update/Alpha/v3.7.4/reports/P0-frame-fence-round1.md>)、[frames.json](<Document/Update/Alpha/v3.7.4/evidence/frame-fence-round1/frames.json>)、[复现轮](<Document/Update/Alpha/v3.7.4/evidence/frame-fence-round1/replication/summary.json>) |
| 帧栅栏实现与构建变体 | [framefence.cpp](<third_party/th08/src/framefence.cpp>)、[framefence.py](<touhou_rl/framefence.py>)、[FRAME_FENCE.md](<third_party/th08/docs/FRAME_FENCE.md>) |
| 精确样本采集工具（PNG + JSONL + manifest） | [frame_fence_capture.py](<tools/rl/frame_fence_capture.py>) |
| **P0 重判（变体镜像，10 判据全过）** | [P0-fence-round26.md](<Document/Update/Alpha/v3.7.3/reports/P0-fence-round26.md>)、证据 `Document/Update/Alpha/v3.7.3/evidence/p0-fence-round26/`、工具 [p0_fence_round26.py](<tools/rl/p0_fence_round26.py>) |
| 变体构建与地址表工具 | [configure.py](<third_party/th08/scripts/configure.py>)、[build.py](<third_party/th08/scripts/build.py>)、[fence-addresses.py](<third_party/th08/scripts/fence-addresses.py>) |
| 帧栅栏单元测试（9 项） | [test_framefence.py](<touhou_rl/tests/test_framefence.py>) |

**当前运行的二进制身份**（实测）：`third_party/th08/build/play/th08.exe`，898,048 字节，SHA-256 `cdd23bca0c724941866efda5da280be67eb6cdd935f83548045c2a8e037f1ddf`（`.text` SHA-256 `7b5b3c8ab5023bfa7960da309338a01982f651dc6b1e7734bee92437202ed6dd`）。这是**重建版**，不是原始 `resources/th08.exe`（840,704 字节，SHA-256 `330fbdbf…`）；两者不可混用，exact-match 结论只对后者成立。

**帧栅栏变体镜像身份**（研究用，非 canonical）：`third_party/th08/build_fence/th08.exe`，905,728 字节，SHA-256 `bc1894fed48beac739f628836629b8b1b1c1b63dbe0097239c2762cb4c8d8565`，`.text` SHA-256 `34a8070b4854672c92a66c95472f9df7fb3b83c0afeb6fbd15bbaed45a8593c7`；地址表 `build_fence/frame-fence-addresses.json`（`TH08_ADDRESS_FILE` 指向它才能读该镜像）。变体全局地址相对 canonical 整体 `+0x3000`，结构偏移不变。
