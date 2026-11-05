# Touhou_RL

> 用像素学会玩东方 Project · Teaching an agent to play Touhou Project from pixels alone

**状态** Alpha / 研究阶段 · **最新记录版本** v3.7.4 · **当前清理计划** v3.7.5 · **测试** 1958 passed, 1 xfailed（2026-11-02 实测 127 s）· **环境** Windows 10/11 + Python 3.12 + PyTorch CUDA

**导航** [中文](#中文) · [English](#english) · [先读：真机数据获取能力说明](<DATA-ACQUISITION.md>) · [当前版本状态](<Document/Update/Alpha/v3.7.4/README.md>) · [v3.7.5 清理计划](<Document/Update/Alpha/v3.7.5/README.md>) · [过程日志](<Document/Dialogs/>)

> **一句话**：基础设施已经跑通并能被审计；**"会玩"还没有做到**。本文把已成立与未成立的分开写，每条都带证据路径。

---

## 中文

### 0. 三十秒版本

这是一个「**只通过画面学会玩东方永夜抄（TH08 1.00d）**」的研究项目。

- **部署形态**只允许输入框选出的游戏区域像素：不读内存、不注入、不依赖具体游戏版本。
- **训练期**可以读游戏内存，但它只是**特权教师**——产出标签、参照系与审计证据，其结果永远不许进入部署观测。
- **终极目标**是可迁移的基模：换一作东方，给定框选区域，它自己学规则、自己微调自己。
- **现在**没有任何"会玩"的可部署模型（默认模型从未被替换）。已经成型的是一条**可审计的真实游戏实验基础设施** + 一条**被反复证否后变得清晰的卡点清单**。

### 1. 目标与边界

| | 内容 |
| --- | --- |
| **终极目标** | 只从框选画面完成游戏，并能迁移到东方系列其它作品：换一作、给一个框，自己学规则、自己微调。 |
| **特权的定位** | 内存特权状态 →（教师/标注/参照系）→ 视觉学生 →（部署）→ 只看画面。读内存不是最终形态，是生产监督信号的手段。 |
| **明确不做** | 不注入 DLL、不 hook `Present`、不改目标字节、不写游戏进程内存；不靠改阈值/改视界/改主指标方向让某个门"通过"；不把未实测的门写成通过。 |
| **成功声明的边界** | 合成门通过只叫"受控场景中可行"；真机门通过才叫"改善指定协议下的效果"；视觉门通过才叫"该能力可视觉部署"。三者不可互相冒充。 |

### 2. 路线图

**已完成（按时间顺序，判定都来自预注册门禁）**

| 阶段 | 版本 | 交付 | 判定 |
| --- | --- | --- | --- |
| 真机访问基础设施 | v1.0.0–v1.0.2 | 只读内存、位图抓帧、按键发送、菜单/会话管理、回放采集 | ✅ 成立并长期使用 |
| 视觉表征与策略 | v1.0.2–v1.0.4 | 框选→裁剪→缩放的帧契约、共享 CNN、18 路价值头、两阶段蒸馏 | 🟡 表征成立（2.41 px / 相关 0.82）；闭环 fail（1627.7 帧 < 4000） |
| 数据流水线 | v1.0.3 | `collect → process → train`，`manifest.json` 交接与漂移审计 | ✅ 成立，至今仍在用 |
| PPO + 安全教师 | v2.0.0 | 18 动作 PPO、候选×持续时间安全教师、episode holdout | 🟡 r11 通过 19 项**有限离线**门；不是真机证明 |
| 特权阶梯 | v3.1–v3.3 | BC18 / 多轮 PPO / 加权诊断 | ❌ 未达 A-04 |
| 离线信息门 | v3.4–v3.5 | Stage 0 的 V/Q 门、S0 状态契约、区域探针 | ❌ S1 fail → S2 不启动 |
| 战术消融 | v3.6.0 / v3.6.1 | 七策略离线消融 + 两条策略的真机配对 | ❌ ENEMY +3.63%（门槛 15%）、POWER 死亡 1.95→4.60；无策略接入控制环 |
| 时空选路（撤离） | v3.7.0 | `escape` 包：逐帧 AABB、分时危险、路线搜索、意图状态机、trace | 🟡 合成 A 门 7 项通过（117/117、可避免封死 −92.3%）；真机 B-01/B-02/B-03 fail |
| 合成课程审计 | v3.7.1–v3.7.3 | 28 个有效世界、教师 V2 28/28、蒸馏负结果、像素信息探针、特权审计 | 🟡 卡点从"信息缺失"改写为"目标缺失" |
| 同帧证明 | v3.7.4 | 帧栅栏变体（ticket + hold + 摘要/后台缓冲复读）+ 采集工具 | ✅ 240/240 帧通过；canonical 镜像未改 |

**正在做（v3.7.5 清理版本）**

1. 按 v3.7.5 清理计划收口所有未完成目标；
2. 先完成 exact 采集接入、manifest、F2 标签预注册、契约和 CLI；
3. 再执行关系型目标、P4 训练和 P5 因果闭环；
4. 游戏缺失时，P3 真实采集、P6 真机配对和 P7 发布保持 blocked/not_run，不使用合成或工程冒烟替代。

**之后**

真实像素战术训练（等预注册数据集）→ 真机公平配对（严格等曝光 + AB/BA 顺序）→ 视觉部署门（只像素）→ 跨作品输入层（框选归一化，分辨率/画风无关）→ 可迁移基模。

### 3. 技术架构

```text
┌─ 数据获取（真机，只读） ─────────────────────────────────────────┐
│ 内存: OpenProcess + ReadProcessMemory（无注入/无写）              │
│ 画面: GetDC + BitBlt → RGB 640×480（中位 2.46 ms/帧）             │
│ 输入: SendInput / PostMessage（实测"发送→生效"约 1 逻辑帧）       │
│ 同帧: 帧栅栏变体 ticket + hold + 摘要/后台缓冲复读（仅研究镜像）  │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌─ 特权层（训练期专用） ───────────────────────────────────────────┐
│ state.py 状态/弹池(0x601 槽)/敌人(481 槽)/资源/输入/replay 游标    │
│ escape: 逐帧 AABB 碰撞、分时危险、参考求解器、路线搜索、意图状态机 │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌─ 数据流水线 ─────────────────────────────────────────────────────┐
│ collect（唯一占用真机的阶段）→ process（校验/打标/冻结划分）      │
│ → train（只认 manifest）→ inspect（manifest 与磁盘是否仍一致）    │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌─ 学习与控制 ─────────────────────────────────────────────────────┐
│ 表征: 共享 CNN 主干 → 状态头 + 18 路价值头（vmodels*.py）         │
│ 策略: BC / 蒸馏 / PPO18 / 宏观路线层（v3.7.2）                    │
│ 基线: pilot.py 的 20 帧首动作束搜索（冻结，作为对照）             │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌─ 验收与证据 ─────────────────────────────────────────────────────┐
│ 预注册 → 门禁判据 → 报告 → SHA-256 证据清单 → 双跑逐行可复现     │
└──────────────────────────────────────────────────────────────────┘
```

规模：`touhou_rl/` 127 文件 31,175 行、`touhou_rl/escape/` 35 文件 11,387 行、`touhou_rl/pipeline/` 7 文件 935 行、测试 141 文件 25,651 行、`tools/` 89 文件 32,562 行。

### 4. 技术经历

这条线不是一次做成的。下面每一项都是**先做、被数据推翻、再改**留下的结论，而不是设计时就知道的事。

**4.1 版本演进与被推翻的判断**

| 版本 | 做了什么 | 被数据推翻了什么 |
| --- | --- | --- |
| v1.0.2 | 第一批视觉落地：帧契约、视觉分片、CNN、两阶段训练、只像素部署 | "能看"成立但没用：真机 1,239 帧 **低于**随机策略的 2,418 帧 |
| v1.0.3 | 原生分辨率分区录制、多流视觉模型 | 定位到真实故障：数据被冻结 + 解码错配（[根因分析](<Document/Update/Alpha/v1.0.3/根因分析-数据冻结与解码错配.md>)） |
| v1.0.4 | 空间网络 v3 拿下 A-01/A-02 | 离线达标 ≠ 闭环可用：教师后悔值 1.27 px，但真机 1627.7 帧、**逐帧 argmax 翻转 50.7%** |
| v2.0.0 | PPO18 + 候选×持续时间安全教师 + episode holdout | 只通过"有限离线门"，转成真机能力没有成功（v2.0.0 记录在本机文档目录，含 18 GB 会话数据，未入库） |
| v3.1–v3.5 | 特权阶梯、离线信息门、区域探针 | 区域探针 S1 fail，按停止线不启动 S2——**不硬推** |
| v3.6.x | 七策略离线消融 + 两条策略真机配对 | 单指标陷阱：CollectPower 功率 +1069% 却让死亡 1.95→4.60；六处结论自我更正 |
| v3.7.0 | 时空选路（提前撤离而非近弹微操） | 合成 117/117 通过，真机反而更差（1.333 vs 1.033 死/局）、弹量 2.1 vs 0.1 |
| v3.7.1–v3.7.3 | 查真机实验是否公平、重造有效课程、测"信息在不在像素里" | 旧合成世界可被**单一恒定动作**通过 → R11/R13/R15/R16 聚合结论作废；蒸馏离线 0.92 而闭环 14–29% |
| v3.7.4 | 把"画面与状态是否同帧"从不可证明变成可证明 | 旧读法在 120 次读取里 20.83% 跨 ticket、至少 4.17% 可证明不是该帧 |

**4.2 六条被数据逼出来的硬结论**

1. **"能看"很容易，"用得对"很难。** v1.0.2：危险图相关 0.82、格子命中 0.90，自机位置误差却 48 px；教师 top1 只有 0.13。v1.0.4 把位置压到 2.41 px 后，闭环仍打不过随机——感知不是瓶颈。
2. **离线准确率不是能力。** v3.7.3：合成像素 masked 准确率 0.913–0.942，闭环留出只有 26.2% / 19.0%，且行为形态是"恒定 0/84、两段式 84/84"。高准确率可以和"根本不会用这个信息"共存。
3. **聚合指标会说谎。** 旧合成世界里有大量可被**单一恒定动作**通过的例子（F1 恒 right/left 各 8/8）；CollectPower 的功率提升是真的，死亡翻倍也是真的。所以现在每个结论都要求**六条非学习基线全灭 + 独立参考见证**才承认有效。
4. **采到截图 ≠ 采到这一帧。** 窗口不被合成时 `BitBlt` 会一直成功返回旧画面，仓库曾因此产出 52,888 张完全相同的图（占某数据集 73%）；读内存的"读到 0"也曾被当成事实（实际是按 +2 而非 +4 读到了 padding）。现在"读到 0 / 重复帧"一律先当可疑。
5. **几何与时序必须按游戏实际，不能按直觉。** 碰撞是**逐帧 AABB**，连续扫掠会多报约 34%；旧 pilot 的圆近似在 45° 盒接触时仍报 +0.67 px 余量；按键后第 1 帧仍用旧速度（输入沉降 1 帧，2,903 对快照中 96.62% 可精确重建）；旧 pilot 的 20 帧视界可退化为 8 帧，并整体丢弃 300 px 外的静止弹墙。
6. **实验设计比调参重要。** 曾存在两个**伪参数**（`--seed`、`--w-coord`，改与不改结果逐位相同），现用 AST 网锁死；真机配对曾按"决策次数"而非"游戏帧"终止，导致两臂曝光不等；一个 (0,0) 幽灵弹曾让"首步完整"检查假通过。修这些的收益大于调任何超参。

### 5. 已达成

**5.1 真机数据获取（可复现、有实测数字）**

| 能力 | 实测 |
| --- | --- |
| 只读内存 | 弹池 `0x601` 槽、敌人 481 槽、自机/资源/输入/replay 游标；只读，不注入、不写字节 |
| 抓帧 | 640×480 客户区，单帧中位 **2.46 ms**（最大 6.56 ms）；290 张连续采样跨 122 个逻辑帧 |
| 发键与读回 | SendInput / PostMessage；实测"发送→生效"约 **1 逻辑帧**；边界 x[8,376]、y[16,432]，普通 4 px/帧、聚焦 2 px/帧 |
| 会话管理 | 启动 → 标题 → 菜单 → 实战 → reset（4.1–4.4 s），60 fps |
| **同帧证明** | 帧栅栏变体：**240/240** 帧通过 13 项检查（3120/3120），GDI 截图与游戏后台缓冲**逐字节相同**，保证期间 59.87 Hz，hold 中位 7.44 ms |
| 对照 | 不 hold 的旧读法：120 次读取中 **20.83%** 跨 ticket、至少 **4.17%** 可证明不是该帧 |

细节与边界见 [DATA-ACQUISITION.md](<DATA-ACQUISITION.md>)、[帧栅栏报告](<Document/Update/Alpha/v3.7.4/reports/P0-frame-fence-round1.md>)、[FRAME_FENCE.md 快照](<Document/Update/Alpha/v3.7.4/FRAME_FENCE-镜像.md>)。

**5.2 数据资产与流水线**

- 回放库 **148 份**：本项目新入库 40 份（4,581,131 输入帧 ≈ **21.2 小时**，结构与真机回放 40/40 通过、二进制重复 0）+ 重建仓库自带 108 份。Lunatic 语料由 24 份扩到 **39 份（+77%）**。
- 三段式流水线 + `manifest.json` 契约：`train` **不认没有 manifest 的数据集**，并核对分片字节数、总帧数、以及"磁盘上多出来的分片"（等于数据漂移）；验证集索引用 `(seed, val_fraction)` 推导并**记录索引哈希**，改数据不重跑 `process` 会直接失败。见 [pipeline/README.md](<touhou_rl/pipeline/README.md>)。

**5.3 学习与策略（能做什么、到哪一步）**

| 能力 | 最好成绩 | 判定 |
| --- | --- | --- |
| 视觉表征（位置/危险） | 位置误差 **2.41 px**、危险图逐帧相关 **0.82** | ✅ 达成 |
| 视觉策略排序 | 0–15 px 危险桶后悔值 **1.27 px** | ✅ 离线达成；闭环 ❌ |
| 合成时空选路 | 有解专项 **117/117**、可避免封死 **−92.3%**、入坑 0 帧 | ✅ 合成达成 |
| 实时性 | 控制回路 P95 **17.4–18.9 ms**、超时率 **0.11–0.22%** | ✅ 达标 |
| 真机主效果 | 死亡 1.333 vs 基线 1.033 死/局 | ❌ 未达成 |
| 真机资源约束 | bomb 2.1 vs 基线 0.1 颗/局 | ❌ 未达成 |
| 视觉学生接管 | 移动准确率 **9.75%**（随机 11.11%） | ❌ 未达成 |

**5.4 工程与验收纪律（这才是这个仓库最硬的部分）**

- **预注册**：八份预注册写定后才跑，阈值不看结果改（[v3.6.0 总索引](<Document/Update/Alpha/v3.6.0/总索引.md>)）。
- **作废声明**：v1 formal 评估、R11–R16 的聚合结论都**公开作废并留档**，而不是悄悄覆盖。
- **可复现性实测**：同一条命令连跑两次逐行比对（排除计时字段），并因此抓出"提交的数据是用旧门禁代码生成的"这一真实漂移。
- **机器可读结算**：门禁表由代码从评估数据生成（`evaluate.settle()`），避免手抄数字漂移。
- **测试**：1958 passed / 1 xfailed / 4 subtests（127 s），全部**不需要启动游戏**。

### 6. 未达成与当前卡点

1. **没有任何可部署的"会玩"模型**：默认网络从未被替换；v3.6.x 的可复现产物是"配置 + 种子 + trace"（无 checkpoint）。
2. **真机主效果不成立**：v3.7.0 的 30 对确认集点估计更差，且 v3.7.1 明确指出该实验**未严格等曝光**（按决策次数而非游戏帧终止），所以"劣化 29%"这个数字本身也不能当因果结论。
3. **真实像素战术训练尚未开始**：同帧能力刚打通（v3.7.4），还没接进正式采集与标签预注册。
4. **合法标签源很窄**：特权审计显示 28 个有效世界里只有 **F2 类 4 例**同时满足"教师不读世界参数"与"对象在真机可观测"。
5. **卡点已定位但未解决**：像素里**有**方向信息（预告渲染上直读 12/12 + 480/480，遮挡后 0/492），小 CNN 两朝向训练也能到 0.896–0.917；失败原因是**目标没有要求模型做"缺口相对自机"的比较**，而不是信息缺失。
6. **跨作品输入层未做**：目前只有 `PlayfieldBox` 一个参数，没有自动框选/分辨率自适应。

### 7. 快速开始

不需要启动游戏即可验证绝大部分内容：

```powershell
# 1) 全量测试（1958 passed，约 2 分钟，不需要游戏进程）
python -m pytest touhou_rl/tests tests -q

# 2) 合成场景八门离线评估（A 门）
python -m touhou_rl.escape evaluate --split dev --workers 16 --dev-seeds 12 `
    --out Document/Update/Alpha/v3.7.0/data/d7-dev-eval-12x2-seeds12.json

# 3) 流水线自检：manifest 与磁盘是否还一致
python -m touhou_rl.pipeline inspect --manifest <dataset>/manifest.json

# 4) 只读探针：真机回放可用性（需要 TH08 与一份 .rpy）
python tools/probes/check_replays.py --dir th08_replay --seconds 4
```

需要真机时：准备一份 **TH08 1.00d（日版）** 于运行目录（`th08.exe` / `th08.dat` / `thbgm.dat`），前台窗口 + SendInput 是能走通菜单的后端（PostMessage 会在菜单导航失败）。同帧采集需要**非 canonical 的帧栅栏变体**（`build_fence/`），读取方走共享节 ack，对游戏进程仍然只读。

依赖：Windows 10/11、Python 3.12、PyTorch（CUDA，已在 RTX 4070 SUPER 12GB / i5-14600KF / 32GB 上实测）、NumPy、Pillow、pytest。

### 8. 仓库结构

| 路径 | 作用 |
| --- | --- |
| `touhou_rl/` | 核心：状态读取、抓帧、输入、环境、模型、训练、评估 |
| `touhou_rl/escape/` | v3.7.x 时空选路：逐帧 AABB、分时危险、路线搜索、意图状态机、trace |
| `touhou_rl/pipeline/` | `collect / process / train / inspect` 三段式流水线 + manifest 契约 |
| `touhou_rl/tests/`、`tests/` | 1959 项测试（141 文件 25,651 行） |
| `tools/probes/`、`tools/rl/` | 探针与实验驱动（89 文件 32,562 行） |
| `Document/Update/Alpha/<version>/` | 21 个版本的正式文档：需求、设计、预注册、报告、数据、证据清单 |
| `Document/Dialogs/` | 33 份增量过程日志（教学/追溯用途，超 100 KB 归档到 `Archived/`） |
| `th08_replay/` | 本项目新入库的 40 份录像 |
| `third_party/th08/` | TH08 逆向重建工程（git submodule），用于地址/语义研究与变体构建 |

### 9. 文档与证据地图

按"想回答什么问题"来找：

| 问题 | 入口 |
| --- | --- |
| 到底能拿到什么数据、边界在哪 | [DATA-ACQUISITION.md](<DATA-ACQUISITION.md>) |
| 现在整体到哪一步、下一步做什么 | [v3.7.4 状态](<Document/Update/Alpha/v3.7.4/README.md>)、[v3.7.3 状态与下一任务卡](<Document/Update/Alpha/v3.7.3/README.md>) |
| 七条战术为什么全被否决 | [v3.6.0/3.6.1 总索引](<Document/Update/Alpha/v3.6.0/总索引.md>) |
| 撤离能力的逐门判定（含 fail） | [v3.7.0 版本结算](<Document/Update/Alpha/v3.7.0/reports/版本结算.md>) |
| 同帧证明怎么做的、边界在哪 | [帧栅栏报告](<Document/Update/Alpha/v3.7.4/reports/P0-frame-fence-round1.md>) |
| 每条声称的状态与证据 | [能力台账](<Document/Update/Alpha/v3.7.3/reports/ledger-round17.md>) |
| 合成学习线为什么卡住 | [收敛结论](<Document/Update/Alpha/v3.7.3/reports/synthesis-round25.md>) |
| 项目解剖（数据流/控制流/陷阱） | [交接文档](<Document/Update/Alpha/v1.0.4/交接文档-项目解剖.md>) |
| 最初的"需求文档"原文 | [存档](<Document/Update/Alpha/v1.0.2/需求文档-原始README存档.md>) |

### 10. 工程纪律（读代码前先读这段）

1. **未实测的门一律写"未实测"**，不写成通过；分母一并写出。
2. **作废要留档**，不静默改数；改结论写在新版本目录，不回写旧结果。
3. **单指标不足以判定**：主指标仍是固定逻辑帧窗口内的死亡数/首死时间，局部安全率不能替代它。
4. **反伪通过**：一个世界/场景必须"六条非学习基线全灭 + 独立参考见证"才算有效；参考求解器要区分 `no_solution_certified` / `budget_exhausted` / `model_unknown` / `feasible`。
5. **测试数不是效果门**：不删失败测试求绿，也不拿测试数当能力证据。

### 11. 致谢与声明

- 东方 Project 及其角色、素材版权归 **ZUN / 上海アリス幻樂団** 所有。本仓库**不附带**任何游戏本体文件（`th08.exe` / `th08.dat` / `thbgm.dat`），使用前请自备合法副本。
- `third_party/th08` 是社区维护的 TH08 逆向重建工程（git submodule），本仓库用它做地址/语义研究与**非 canonical** 的研究变体构建；变体不能用于 exact-match 或版本一致性声明。
- 本仓库记录的所有"实测"数字都对应 `Document/Update/Alpha/<version>/evidence/` 下的落盘证据；没有证据的数字不应被当作结论引用。

---

## English

### 0. Thirty-second version

This is a research project on **learning to play Touhou 8 (Imperishable Night, TH08 1.00d) from pixels alone**.

- **Deployment** may only consume pixels of a user-selected playfield box: no memory reads, no injection, no dependency on a specific game build.
- **Training** may read game memory, but only as a **privileged teacher** — it produces labels, reference frames and audit evidence, and never feeds the deployed policy.
- **The end goal** is a transferable base model: swap in another Touhou title, give it a box, and it learns the rules and fine-tunes itself.
- **Today** there is no deployable "plays the game" model (the default model has never been replaced). What exists is an **auditable real-game experiment infrastructure** plus a **sharply defined list of blockers**, each one obtained by disproving an earlier hypothesis.

### 1. Goal and boundaries

| | Content |
| --- | --- |
| **End goal** | Play from the boxed screen region only, and transfer to other Touhou titles: new game, one box, self-learned rules, self fine-tuning. |
| **Role of privilege** | Memory state → (teacher / labels / reference) → vision student → (deploy) → pixels only. Reading memory is a means of producing supervision, not the product. |
| **Explicitly out of scope** | No DLL injection, no `Present` hooks, no patching target bytes, no writes into the game process; never move a threshold, horizon or metric direction to make a gate pass; never write an unmeasured gate as passing. |
| **Claim boundaries** | A synthetic gate passing means "feasible in a controlled scenario"; a live gate means "improved under a specified protocol"; a vision gate means "this ability is visually deployable". They may not stand in for each other. |

### 2. Roadmap

**Done (in order; every verdict comes from a pre-registered gate)**

| Stage | Version | Delivered | Verdict |
| --- | --- | --- | --- |
| Real-game access | v1.0.0–v1.0.2 | Read-only memory, bitmap capture, key sending, menu/session management, replay capture | ✅ Works, used ever since |
| Vision representation and policy | v1.0.2–v1.0.4 | Box→crop→resize frame contract, shared CNN, 18-way value head, two-stage distillation | 🟡 Representation works (2.41 px / corr 0.82); closed loop fails (1627.7 frames < 4000) |
| Data pipeline | v1.0.3 | `collect → process → train` with `manifest.json` hand-off and drift audit | ✅ Works, still in use |
| PPO + safety teacher | v2.0.0 | 18-action PPO, candidate × duration safety teacher, episode holdout | 🟡 r11 passed 19 **offline-only** gates; not a live proof |
| Privileged ladder | v3.1–v3.3 | BC18, repeated PPO, weighted diagnostics | ❌ A-04 not reached |
| Offline information gates | v3.4–v3.5 | Stage 0 V/Q gates, S0 state contract, region probe | ❌ S1 failed → S2 not started |
| Tactical ablations | v3.6.0 / v3.6.1 | Seven-strategy offline ablation + two live paired comparisons | ❌ ENEMY +3.63% (threshold 15%), POWER deaths 1.95→4.60; nothing wired into the control loop |
| Space-time routing (escape) | v3.7.0 | `escape` package: per-frame AABB, time-indexed danger, route search, intent state machine, traces | 🟡 7 synthetic A-gates pass (117/117, avoidable sealing −92.3%); live B-01/B-02/B-03 fail |
| Synthetic curriculum audit | v3.7.1–v3.7.3 | 28 valid worlds, teacher V2 28/28, distillation negative result, pixel-information probe, privilege audit | 🟡 Blocker redefined from "information missing" to "objective missing" |
| Same-frame proof | v3.7.4 | Frame-fence variant (ticket + hold + digest/back-buffer re-read) + capture tooling | ✅ 240/240 frames pass; canonical image untouched |

**In progress (v3.7.4 → next)**

1. Wire the fence into the official capture path: `capture_dataset.py` uses `FenceFrameSampler` and stamps `frame_proof=exact` plus variant/address-table hashes.
2. Pre-register the `frame_proof=exact` manifest fields — no P3 dataset is declared valid before that.
3. Change the **objective** per the pixel-information probe: regress `gap_center − player_x` directly (relational read-out), with both orientations and edge positions covered by the training distribution.
4. Narrow P3's legal label source to the F2 family (objects observable in the real game) — or narrow the version scope and say so explicitly.

**Later**

Real-pixel tactical training (waiting on a pre-registered dataset) → fair live paired comparison (strictly equal frame exposure, AB/BA order) → vision deployment gate (pixels only) → cross-title input layer (box normalisation, resolution/style independent) → transferable base model.

### 3. Architecture

```text
┌─ Data acquisition (real game, read-only) ────────────────────────┐
│ memory: OpenProcess + ReadProcessMemory (no injection, no write) │
│ frames: GetDC + BitBlt → RGB 640×480 (median 2.46 ms/frame)      │
│ input : SendInput / PostMessage (measured ~1 logic frame lag)    │
│ fence : variant ticket + hold + digest/back-buffer re-read       │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌─ Privileged layer (training only) ───────────────────────────────┐
│ state.py: player / bullet pool (0x601 slots) / enemies (481) /   │
│ resources / input / replay cursor                                │
│ escape : per-frame AABB, time-indexed danger, reference solver,  │
│ route search, intent state machine                               │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌─ Data pipeline ──────────────────────────────────────────────────┐
│ collect (only stage that occupies the game) → process (validate, │
│ label, freeze split) → train (manifest required) → inspect       │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌─ Learning and control ───────────────────────────────────────────┐
│ representation: shared CNN trunk → state head + 18-way value head│
│ policies: BC / distillation / PPO18 / macro route layer          │
│ baseline: pilot.py 20-frame first-action beam search (frozen)    │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
┌─ Acceptance and evidence ────────────────────────────────────────┐
│ pre-registration → gate criteria → reports → SHA-256 manifests → │
│ double-run line-by-line reproducibility                          │
└──────────────────────────────────────────────────────────────────┘
```

Scale: `touhou_rl/` 127 files / 31,175 lines, `touhou_rl/escape/` 35 files / 11,387 lines, `touhou_rl/pipeline/` 7 files / 935 lines, tests 141 files / 25,651 lines, `tools/` 89 files / 32,562 lines.

### 4. Technical journey

None of this was designed up front. Each item below is a conclusion that survived being disproved by data.

**4.1 Version history and what overturned earlier beliefs**

| Version | What was built | What the data overturned |
| --- | --- | --- |
| v1.0.2 | First vision batch: frame contract, vision shards, CNN, two-stage training, pixels-only deployment | "It can see" held, but was useless: 1,239 live frames — **worse** than a random policy's 2,418 |
| v1.0.3 | Native-resolution partitioned capture, multi-stream vision model | Found a real fault: frozen data plus decode mismatch ([root-cause analysis](<Document/Update/Alpha/v1.0.3/根因分析-数据冻结与解码错配.md>)) |
| v1.0.4 | Spatial network v3 met A-01/A-02 | Offline numbers are not closed-loop skill: teacher regret 1.27 px, but 1627.7 live frames and **50.7% per-frame argmax flips** |
| v2.0.0 | PPO18 + candidate × duration safety teacher + episode holdout | Passed only "limited offline gates"; never became live capability (v2.0.0 records stay in the local doc tree — 18 GB of session data — and are not published) |
| v3.1–v3.5 | Privileged ladder, offline information gates, region probe | Region probe failed at S1 and S2 was never started — a stop line, honoured |
| v3.6.x | Seven-strategy offline ablation + two live paired runs | The single-metric trap: CollectPower gained +1069% power while deaths went 1.95→4.60; six conclusions self-corrected |
| v3.7.0 | Space-time routing (leave early instead of micro-dodging late) | Synthetic 117/117, yet live got worse (1.333 vs 1.033 deaths/episode) and bombs 2.1 vs 0.1 |
| v3.7.1–v3.7.3 | Checked whether the live experiment was even fair; rebuilt valid curricula; measured whether direction lives in pixels | Old synthetic worlds were passable by a **single constant action** → R11/R13/R15/R16 aggregates voided; distillation 0.92 offline vs 14–29% closed loop |
| v3.7.4 | Turned "are pixels and state the same frame?" from unprovable into provable | The old read path crossed tickets in 20.83% of 120 reads, and ≥4.17% provably belonged to another frame |

**4.2 Six hard lessons**

1. **Seeing is easy; using it is hard.** v1.0.2 got danger-map correlation 0.82 and cell hit rate 0.90, yet player position error was 48 px and teacher top-1 only 0.13. v1.0.4 cut position error to 2.41 px and the closed loop *still* lost to random — perception was never the bottleneck.
2. **Offline accuracy is not capability.** v3.7.3: masked accuracy 0.913–0.942 offline, but held-out closed loop 26.2% / 19.0%, with behaviour shapes of "constant 0/84" and "two-phase 84/84". High accuracy coexists with "cannot use the information".
3. **Aggregate metrics lie.** Many old synthetic worlds were passable by a **single constant action** (F1: constant right/left 8/8). CollectPower's power gain was real, and so was the doubled death rate. That is why a world now only counts when **six non-learning baselines all fail and an independent reference finds a witness**.
4. **A captured screenshot is not a captured frame.** When the window is not composited, `BitBlt` keeps returning the last picture and still reports success — one dataset was 52,888 identical images (73% of it). Reading "0" from memory was once taken as fact (wrong stride read padding). Today, zeros and duplicate frames are treated as suspicious first.
5. **Geometry and timing follow the game, not intuition.** Collisions are **per-frame AABB**; continuous sweeps over-report by ~34%. The old pilot's circle approximation still reported +0.67 px margin on a 45° box contact. The first frame after a key press keeps the old velocity (1-frame input settling; 96.62% of 2,903 snapshots reconstructed exactly). The old 20-frame horizon could degenerate to 8 and drop a whole static 300 px wall.
6. **Experiment design beats hyper-parameters.** Two **fake flags** existed (`--seed`, `--w-coord`: results bit-identical either way) and are now locked down by an AST net. Live pairing once terminated on decision count instead of game frames, giving the two arms unequal exposure. A (0,0) ghost bullet made the "first snapshot complete" check pass falsely. Fixing these paid more than any tuning.

### 5. What has been achieved

**5.1 Real-game data access (measured, reproducible)**

| Capability | Measured |
| --- | --- |
| Read-only memory | Bullet pool `0x601` slots, 481 enemy slots, player/resources/input/replay cursor; read-only, no injection, no byte writes |
| Frame capture | 640×480 client area, median **2.46 ms** per frame (max 6.56 ms); 290 consecutive samples across 122 logic frames |
| Input and read-back | SendInput / PostMessage; measured ~**1 logic frame** from send to effect; bounds x[8,376], y[16,432], 4 px/frame normal and 2 px/frame focused |
| Session management | Boot → title → menu → gameplay → reset (4.1–4.4 s), 60 fps |
| **Same-frame proof** | Frame-fence variant: **240/240** frames pass 13 checks (3120/3120), GDI screenshots **byte-identical** to the game's back buffer, 59.87 Hz sustained, hold median 7.44 ms |
| Control group | Old read path without hold: **20.83%** of 120 reads crossed tickets, ≥**4.17%** provably not that frame |

Details and limits: [DATA-ACQUISITION.md](<DATA-ACQUISITION.md>), [frame-fence report](<Document/Update/Alpha/v3.7.4/reports/P0-frame-fence-round1.md>), [FRAME_FENCE.md snapshot](<Document/Update/Alpha/v3.7.4/FRAME_FENCE-镜像.md>).

**5.2 Data assets and pipeline**

- **148 replays**: 40 newly added (4,581,131 input frames ≈ **21.2 hours**; 40/40 pass both structural and live-replay checks; zero binary duplicates) plus 108 bundled fixtures. The Lunatic corpus grew from 24 to **39 replays (+77%)**.
- Three-stage pipeline with a `manifest.json` contract: `train` **refuses datasets without a manifest** and cross-checks shard byte sizes, total frames and shards present on disk but absent from the manifest (data drift). Validation indices are derived from `(seed, val_fraction)` and their **hash is recorded**, so changing data without re-running `process` fails loudly. See [pipeline/README.md](<touhou_rl/pipeline/README.md>).

**5.3 Learning results (what works, and how far)**

| Capability | Best result | Verdict |
| --- | --- | --- |
| Vision representation | Position error **2.41 px**, per-frame danger correlation **0.82** | ✅ Achieved |
| Vision policy ranking | Regret **1.27 px** in the 0–15 px danger bucket | ✅ Offline; ❌ closed loop |
| Synthetic space-time routing | Solvable cases **117/117**, avoidable sealing **−92.3%**, 0 frames inside pockets | ✅ Achieved in synthetic |
| Real-time budget | Control loop P95 **17.4–18.9 ms**, timeout rate **0.11–0.22%** | ✅ Achieved |
| Live main effect | 1.333 vs baseline 1.033 deaths/episode | ❌ Not achieved |
| Live resource constraint | 2.1 vs baseline 0.1 bombs/episode | ❌ Not achieved |
| Vision student takeover | Move accuracy **9.75%** (random 11.11%) | ❌ Not achieved |

**5.4 Engineering and acceptance discipline (the strongest part of this repo)**

- **Pre-registration**: eight plans frozen before running; thresholds never moved after seeing results ([v3.6.0 index](<Document/Update/Alpha/v3.6.0/总索引.md>)).
- **Void declarations**: the v1 formal evaluation and the R11–R16 aggregates were publicly **voided and archived**, not quietly overwritten.
- **Reproducibility tested**: the same command run twice and compared line by line (timing fields excluded) — which caught the real drift "committed data came from older gate code".
- **Machine-readable settlement**: gate tables are generated from evaluation data by `evaluate.settle()`, so numbers cannot drift through manual copying.
- **Tests**: 1958 passed / 1 xfailed / 4 subtests in 127 s, **none of which need the game running**.

### 6. What has not been achieved

1. **No deployable model that plays.** The default network has never been replaced; the reproducible artefact of v3.6.x is "config + seeds + traces" (no checkpoint).
2. **The live main effect does not hold.** The v3.7.0 30-pair confirmation set is a worse point estimate, and v3.7.1 showed the experiment was **not strictly equal-exposure** (it terminated on decision count, not game frames) — so even "29% worse" cannot be claimed as causal.
3. **Real-pixel tactical training has not started.** Same-frame capture only just became possible (v3.7.4) and is not yet wired into capture and label pre-registration.
4. **The legal label source is narrow.** The privilege audit shows only **4 F2-family worlds** out of 28 valid worlds satisfy both "teacher does not read world parameters" and "object exists in the real game".
5. **The blocker is located but unsolved.** Direction information **is** in the pixels (12/12 + 480/480 on telegraph renders, 0/492 when occluded; a small CNN reaches 0.896–0.917 when trained on both orientations). The failure is that **the objective never asks the model to compare the gap against the player**, not that information is missing.
6. **No cross-title input layer.** Only a single `PlayfieldBox` parameter exists; there is no auto-boxing or resolution adaptation.

### 7. Quick start

Most of the project is verifiable without launching the game:

```powershell
# 1) Full test suite (1958 passed, ~2 min, no game process needed)
python -m pytest touhou_rl/tests tests -q

# 2) Offline synthetic A-gates
python -m touhou_rl.escape evaluate --split dev --workers 16 --dev-seeds 12 `
    --out Document/Update/Alpha/v3.7.0/data/d7-dev-eval-12x2-seeds12.json

# 3) Pipeline self-check: does the manifest still match the disk?
python -m touhou_rl.pipeline inspect --manifest <dataset>/manifest.json

# 4) Read-only probe: are the replays usable in the real game?
python tools/probes/check_replays.py --dir th08_replay --seconds 4
```

For real-game work you need a **TH08 1.00d (Japanese)** copy in the run directory (`th08.exe`, `th08.dat`, `thbgm.dat`); a foreground window plus SendInput is the backend that navigates menus (PostMessage fails there). Exact-frame capture requires the **non-canonical frame-fence variant** (`build_fence/`); the reader acknowledges through a shared section and still has no write access to the game process.

Requirements: Windows 10/11, Python 3.12, PyTorch (CUDA; measured on RTX 4070 SUPER 12 GB / i5-14600KF / 32 GB), NumPy, Pillow, pytest.

### 8. Repository layout

| Path | Purpose |
| --- | --- |
| `touhou_rl/` | Core: state reading, capture, input, environment, models, training, evaluation |
| `touhou_rl/escape/` | v3.7.x space-time routing: per-frame AABB, time-indexed danger, route search, intent machine, traces |
| `touhou_rl/pipeline/` | Three-stage `collect / process / train / inspect` pipeline plus the manifest contract |
| `touhou_rl/tests/`, `tests/` | 1959 tests (141 files, 25,651 lines) |
| `tools/probes/`, `tools/rl/` | Probes and experiment drivers (89 files, 32,562 lines) |
| `Document/Update/Alpha/<version>/` | Official records for 21 versions: requirements, design, pre-registration, reports, data, evidence manifests |
| `Document/Dialogs/` | 33 incremental process logs (teaching/traceability; archived past 100 KB) |
| `th08_replay/` | The 40 replays newly added by this project |
| `third_party/th08/` | TH08 reverse-engineering project (git submodule) used for address/semantics research and variant builds |

### 9. Docs and evidence map

| Question | Entry point |
| --- | --- |
| What data can we actually get, and where are the limits? | [DATA-ACQUISITION.md](<DATA-ACQUISITION.md>) |
| Where are we, and what is next? | [v3.7.4 status](<Document/Update/Alpha/v3.7.4/README.md>), [v3.7.3 status and next-task card](<Document/Update/Alpha/v3.7.3/README.md>) |
| Why were all seven tactics rejected? | [v3.6.0/3.6.1 index](<Document/Update/Alpha/v3.6.0/总索引.md>) |
| Gate-by-gate verdicts for escape (including failures) | [v3.7.0 settlement](<Document/Update/Alpha/v3.7.0/reports/版本结算.md>) |
| How same-frame proof works and where it stops | [frame-fence report](<Document/Update/Alpha/v3.7.4/reports/P0-frame-fence-round1.md>) |
| Status of every claim, with evidence | [capability ledger](<Document/Update/Alpha/v3.7.3/reports/ledger-round17.md>) |
| Why the synthetic learning line is stuck | [convergence conclusion](<Document/Update/Alpha/v3.7.3/reports/synthesis-round25.md>) |
| Project anatomy (data flow / control flow / traps) | [hand-off document](<Document/Update/Alpha/v1.0.4/交接文档-项目解剖.md>) |
| The original requirement statement | [archive](<Document/Update/Alpha/v1.0.2/需求文档-原始README存档.md>) |

### 10. Engineering discipline (read before reading the code)

1. **Any unmeasured gate is written "not measured"**, never "passed"; denominators are always printed.
2. **Voids are archived**, never silent edits; corrections live in the new version directory and never rewrite old results.
3. **One metric is never enough**: the main metric stays deaths / time-to-first-death inside a fixed logic-frame window; local safety rates cannot replace it.
4. **Anti-false-pass**: a world or scenario only counts when six non-learning baselines all fail and an independent reference finds a witness; the reference solver must separate `no_solution_certified` / `budget_exhausted` / `model_unknown` / `feasible`.
5. **Test counts are not effect gates**: failing tests are never deleted to get green, and test counts are never used as capability evidence.

### 11. Credits and notices

- Touhou Project, its characters and assets are © **ZUN / Team Shanghai Alice**. This repository ships **no** game files (`th08.exe`, `th08.dat`, `thbgm.dat`); you must own a legitimate copy to run anything live.
- `third_party/th08` is a community reverse-engineering / reconstruction project (git submodule). It is used here for address and semantics research and to build a **non-canonical** research variant; that variant may not be used for exact-match or version-consistency claims.
- Every "measured" number in this repository has on-disk evidence under `Document/Update/Alpha/<version>/evidence/`. A number without evidence should not be quoted as a conclusion.
