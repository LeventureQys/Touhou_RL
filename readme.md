需求文档

我简单说一下这个项目的终极目标

这个项目的终极目标就是通过框选游戏区的方式，让强化学习学习如何游玩东方系列游戏，最好是可以训练一个基模，然后让其可以根据不同的游戏，让其自己游玩，自己学习规则，自己微调自己。

最好通过框选和自学，通过用户选定好特定的框做好特定的定义，然后就可以让他自己学习如何控制键盘自己游玩这个游戏。

整个项目偏教学，所以我希望你写好整个项目的过程日志，写在Document\Dialogs 目录下，这个日志最好是增量的。每当当前的日志超过100kb之后，将整个日志截断，旧的日志放到Document\Dialogs\Archived目录下

另外，我希望你每个过程都能够尽可能以教学的目的来循循渐进地推进工作，以及留好痕迹，让我可以追溯到你是如何思考整个框架和细节的构建的。

---

## 目标与定位（2026-10-18 方向澄清，优先于下面所有内容）

### 终极目标

**只通过视觉**（框选游戏区域）完成游戏，并具备拓展到东方系列其它作品的能力：
不读内存、不注入、不依赖具体游戏版本。理想形态是可以迁移的基模——换一个东方作品，
给定框选区域，它能自己学规则、自己微调自己。

### 读内存的角色：训练期的特权教师，不是部署形态

`touhou_rl/` 目前几乎全部能力（精确子弹池、运动学、教师、奖赏、评估）都建立在"读内存"上。
这套逆向成果的定位是**特权信息来源**：在训练阶段提供一个"上帝视角"的标注器与参照系，
它的价值在于生产高质量监督信号，而不是最终的运行方式。

正确的关系是：

```
内存特权状态 ──(教师 / 标注 / 参照系)──▶ 视觉学生 ──(部署)──▶ 只看框选画面
```

### 视觉改造进度（2026-10-18 第一批已落地）

改造前，全仓 `Conv2d` 数量为 0，`obs.py` 文档承诺的 `profile.state_source` 只在注释里存在，
抓帧只用于截图。现在视觉层是一等公民：

| 层 | 文件 | 状态 |
| --- | --- | --- |
| 帧契约（框选 → 裁剪 → 面积平均缩放 → 帧堆叠） | `touhou_rl/vision.py` | ✅ 已实现并有单测 |
| 视觉数据分片（帧 + 特权标签） | `touhou_rl/vision_data.py` | ✅ 已实现并有单测 |
| 采集（回放自走 + 抓帧） | `touhou_rl/vision_capture.py` | ✅ 实测 28 帧/s |
| 模型（共享 CNN 主干 + 状态头 + 18 路价值头） | `touhou_rl/vmodels.py` | ✅ 84 万参数 |
| 训练（表征 → 策略，两阶段） | `touhou_rl/vision_train.py` | ✅ 已跑通 |
| 部署（**只用像素**接真机） | `evaluate.py --vision-checkpoint` | ✅ 1,239 帧 / 28.4 决策每秒 |
| 契约文档 | `Document/Update/Alpha/v1.0.2/视觉框架-契约v2.md` | ✅ |

**首批实测数字（4,524 帧 Lunatic Stage 1，2.7 分钟采集）**：

- 表征：危险图相关 **0.82**、格子命中率 0.90 → **确实"能看见子弹"**；
- 自机位置误差 **48 px**（384×448 场地内）→ 还不够准；
- 策略：教师 top1 0.13、**后悔值 3.98 px**（特权学生是 2.34 px）→ 还不够用；
- 真机：**1,239 帧**（随机策略 2,418）→ 会动，不会躲。

一句话：**"能看"初步成立，"看得准"和"会玩"还没成立**，而且这只用了 2.7 分钟数据。

### 仍未做（下一批）

1. **跨作品输入层**：目前只有 `PlayfieldBox` 一个参数，还没有自动框选/分辨率自适应；
2. **视觉 DAgger**：现有视觉数据全来自人类回放轨迹，学生自己走到的状态没有标签；
3. **标签补全**：激光与敌人在 `state.py` 里能读，但没进观测 → 视觉学生自然也学不到；
4. **自机位置热图头**：48 px 的位置误差来自 `AdaptiveAvgPool` 丢空间精度，应改成热图回归。

### 纠偏后的优先级

| 优先级 | 任务 | 状态 | 为什么在这个位置 |
| --- | --- | --- | --- |
| 1 | **视觉表征学生**：画面 → 特权观测（子弹/自机/Boss/激光） | 🟡 已跑通，精度待提 | 可离线评估（回归误差），不需要游戏时间，是"能不能看"的第一道门 |
| 2 | **视觉策略学生**：以特权教师为标签蒸馏（含 DAgger） | 🟡 已跑通，会动不会躲 | 把"会看"变成"会玩"，教师已经证明该状态空间下可行 |
| 3 | **跨作品输入层**：框选区域归一化（分辨率/画风无关） | ⬜ 未做 | 基模的前提；与 1、2 解耦，可并行 |
| 4 | 特权侧继续补强（激光、敌人、Boss 进观测） | ⬜ 未做 | 教师越强，学生上限越高 |

> 下面的「录像库」「训练流水线」等内容仍然有效——它们是**教师侧**的数据与工具，
> 但需按本节重新理解其定位：录像与内存数据是教材，不是最终产品。

---

## 数据流水线（collect → process → train）

三段式，每段只做一件事，段与段之间靠 `manifest.json` 交接。详细文档在代码旁：

| 阶段 | 作用 | 成本 | 文档 |
| --- | --- | --- | --- |
| **collect** 采集 | 游戏 → 原始分片（特权状态或视觉帧 + 标签） | **唯一需要占用真实游戏的阶段** | [collect/README.md](<touhou_rl/pipeline/collect/README.md>) |
| **process** 处理 | 校验分片、统计、离线打教师标签、钉死训练/验证划分 | 离线，秒到分钟 | [process/README.md](<touhou_rl/pipeline/process/README.md>) |
| **train** 训练 | 读 manifest → 训练 → 写可审计的 `train-report.json` | 离线，GPU | [train/README.md](<touhou_rl/pipeline/train/README.md>) |

流水线总览见 [touhou_rl/pipeline/README.md](<touhou_rl/pipeline/README.md>)。

```powershell
# 1) 采集（占游戏，约 2.7 分钟得 4.5k 视觉帧）
python -m touhou_rl.pipeline collect --modality vision `
    --replay th08_replay/Th8_ud3936.rpy --out .../datasets/vision-lunatic --max-seconds 160

# 2) 处理（校验 + 打教师标签 + 写 manifest）
python -m touhou_rl.pipeline process --raw .../datasets/vision-lunatic

# 3) 训练（读 manifest，拒绝没有 manifest 的数据集）
python -m touhou_rl.pipeline train --model vision `
    --manifest .../datasets/vision-lunatic/manifest.json --out .../runs/vision-v2

# 4) 随时审计：manifest 与磁盘是否还一致
python -m touhou_rl.pipeline inspect --manifest .../datasets/vision-lunatic/manifest.json
```

**为什么值得这么分**：此前三个环节各自重新推导"数据里有什么"，出过一次真实故障——
一次中断的采集让某个分片只有半截数据，而训练报告仍声称用了某个划分。
现在 `train` 不认没有 manifest 的数据集，并且会核对分片字节数、总帧数、
**以及磁盘上是否存在 manifest 之外的分片**（训练器会扫描整个目录，多出来的分片等于数据漂移）。

划分用 `(seed, val_fraction)` 推导并**额外记录验证集索引的哈希**，所以改数据不重跑 process 会直接失败，
而不是悄悄换一套验证集。

---

## 录像库

这个项目的数据来源是**真机录像**。东方 8 的回放文件逐帧记录了「玩家按了什么」，把它放进游戏的
`replay/` 目录播放，就能一边播放一边只读采样，得到 (状态, 动作) 对——既不需要人上手打，
也不需要注入输入。目前仓库里有两处录像：

| 位置 | 份数 | 来源 | 说明 |
| --- | ---: | --- | --- |
| `th08_replay/` | **40** | 本项目新入库（2026-10-18） | 全部为完整流程录像；逐份清单见下 |
| `third_party/th08/tests/fixtures/replays/th08-1.00d/accepted/` | 108 | 重建仓库自带 | 含 Lunatic 24 份；元数据见 `third_party/th08/config/replay-fixtures.json` |

新入库的 40 份合计 **4,581,131 个输入帧 ≈ 21.2 小时**游戏时间，其中 Lunatic 15 份
（230 万帧），使 Lunatic 语料从 24 份（298 万帧）扩到 **39 份（528 万帧，+77%）**。

### 可用性检查（40 份逐份实测）

两层检查，缺一不可：只做结构会漏掉「能解析但放不出来」，只做回放会漏掉「能放但校验和不符」。

| 层 | 方法 | 结果 |
| --- | --- | --- |
| 结构 | `third_party/th08/scripts/replay_file.py` 的 `inspect()`：魔数/版本、文件长度、校验和、LZSS 解压尺寸、exe 身份、慢速模式、偏移、输入编码 | **40/40 通过** |
| 真机回放 | `tools/probes/check_replays.py`：装进临时运行目录 → 走菜单 → 实际播放 → 读 `is_replay` / `gameplay_frame` / `g_GuiMessageInputCurrent` | **40/40 通过**，均 4 秒推进 241 帧（= 60 fps） |

- 与既有 108 份字节重复：**0**；这 40 份之间互相重复：**0**；
- 专家输入通道覆盖率中位数 **99.6%**（`g_GuiMessageInputCurrent` 非零帧占比），
  最低 8.7% 是 Stage 1 开头角色本来就不操作的那几秒；
- 全部 40 份都是完整流程（普通回放 6 面 1→Final B；Extra 为单面）；
- `clearState` 全部为 2，但**不能**据此宣称这 40 份都是 1CC：`ReplayManager.cpp:453` 对
  `shotType > 3`（8 个单体机体）强制写 2，该字段只在 4 个组合机体上代表「Final B 无续关通关」。

结果文件：`third_party/th08/build/rl/probe-v103/replay-inventory.json`（结构 + 元数据）、
`third_party/th08/build/rl/probe-v103/replay-check-all.json`（真机回放）。

### 清单（40 份，按难度 → 机体排序）

| 文件 | 机体 | 难度 | 路线 | 输入帧数 | 时长 |
| --- | --- | --- | --- | ---: | ---: |
| `Th8_ud2558.rpy` | Border Team | Lunatic | B | 148,618 | 41.3 分钟 |
| `Th8_udg005.rpy` | Border Team | Lunatic | B | 144,555 | 40.2 分钟 |
| `Th8_ud0067.rpy` | Magic Team | Lunatic | B | 154,765 | 43.0 分钟 |
| `Th8_ud24e5.rpy` | Scarlet Team | Lunatic | B | 153,240 | 42.6 分钟 |
| `Th8_ud2529.rpy` | Scarlet Team | Lunatic | B | 191,832 | 53.3 分钟 |
| `Th8_ud238d.rpy` | Ghost Team | Lunatic | B | 141,715 | 39.4 分钟 |
| `Th8_ud5022.rpy` | Ghost Team | Lunatic | B | 151,572 | 42.1 分钟 |
| `Th8_udIIII.rpy` | Ghost Team | Lunatic | B | 145,343 | 40.4 分钟 |
| `Th8_ud0059.rpy` | Youmu | Lunatic | B | 151,110 | 42.0 分钟 |
| `Th8_ud0063.rpy` | Youmu | Lunatic | B | 141,699 | 39.4 分钟 |
| `Th8_ud0439.rpy` | Youmu | Lunatic | B | 157,610 | 43.8 分钟 |
| `Th8_ud2110.rpy` | Youmu | Lunatic | B | 181,825 | 50.5 分钟 |
| `Th8_ud243c.rpy` | Youmu | Lunatic | B | 137,089 | 38.1 分钟 |
| `Th8_ud3936.rpy` | Youmu | Lunatic | B | 154,023 | 42.8 分钟 |
| `Th8_udx315.rpy` | Youmu | Lunatic | B | 150,351 | 41.8 分钟 |
| `Th8_ud2be5.rpy` | Magic Team | Hard | B | 149,179 | 41.4 分钟 |
| `Th8_ud2563.rpy` | Reimu | Hard | B | 147,493 | 41.0 分钟 |
| `Th8_ud256a.rpy` | Alice | Hard | B | 145,120 | 40.3 分钟 |
| `Th8_ud2601.rpy` | Alice | Hard | B | 143,928 | 40.0 分钟 |
| `Th8_ud2a95.rpy` | Remilia | Hard | B | 148,130 | 41.1 分钟 |
| `Th8_ud2aa6.rpy` | Remilia | Hard | B | 144,464 | 40.1 分钟 |
| `Th8_ud2596.rpy` | Yuyuko | Hard | B | 138,323 | 38.4 分钟 |
| `Th8_ud2970.rpy` | Marisa | Normal | B | 140,719 | 39.1 分钟 |
| `Th8_ud18d9.rpy` | Border Team | Extra | Extra | 72,198 | 20.1 分钟 |
| `Th8_ud17f4.rpy` | Magic Team | Extra | Extra | 69,042 | 19.2 分钟 |
| `Th8_ud187a.rpy` | Magic Team | Extra | Extra | 68,248 | 19.0 分钟 |
| `Th8_ud18e5.rpy` | Scarlet Team | Extra | Extra | 72,240 | 20.1 分钟 |
| `Th8_ud17b0.rpy` | Ghost Team | Extra | Extra | 71,001 | 19.7 分钟 |
| `Th8_ud1897.rpy` | Ghost Team | Extra | Extra | 70,541 | 19.6 分钟 |
| `Th8_ud1978.rpy` | Reimu | Extra | Extra | 54,405 | 15.1 分钟 |
| `Th8_ud1709.rpy` | Yukari | Extra | Extra | 75,334 | 20.9 分钟 |
| `Th8_ud1ab1.rpy` | Yukari | Extra | Extra | 65,656 | 18.2 分钟 |
| `Th8_ud1858.rpy` | Marisa | Extra | Extra | 49,229 | 13.7 分钟 |
| `Th8_ud1ffb.rpy` | Marisa | Extra | Extra | 49,774 | 13.8 分钟 |
| `Th8_ud1936.rpy` | Alice | Extra | Extra | 64,848 | 18.0 分钟 |
| `Th8_ud175f.rpy` | Sakuya | Extra | Extra | 54,633 | 15.2 分钟 |
| `Th8_ud1935.rpy` | Remilia | Extra | Extra | 65,588 | 18.2 分钟 |
| `Th8_ud1fab.rpy` | Remilia | Extra | Extra | 77,986 | 21.7 分钟 |
| `Th8_ud18ee.rpy` | Youmu | Extra | Extra | 69,778 | 19.4 分钟 |
| `Th8_ud1a53.rpy` | Youmu | Extra | Extra | 67,927 | 18.9 分钟 |

| 难度 | 份数 | 总输入帧 | 合计时长 |
| --- | ---: | ---: | ---: |
| Lunatic | 15 | 2,305,347 | 640.4 分钟 |
| Hard | 7 | 1,016,637 | 282.4 分钟 |
| Normal | 1 | 140,719 | 39.1 分钟 |
| Extra | 17 | 1,118,428 | 310.7 分钟 |
| **合计** | **40** | **4,581,131** | **1272.5 分钟（21.2 小时）** |

### 怎么用

一份录像的采集耗时约等于它的回放时长（60 fps 实时播放，144 万帧 ≈ 21 小时是全部采集的总成本）：

```powershell
# 采一份录制成 (状态, 动作) 数据集；--stage -1 是默认值 = 自动取该回放的第一面
python -m touhou_rl.capture_dataset `
  --replay th08_replay/Th8_ud3936.rpy `
  --out third_party/th08/build/rl/datasets/lunatic-youmu --tag lunatic-youmu --stage -1
```

Extra 回放只能从第 8 面开始（它只有一个面），硬写 `--stage 0` 会让菜单宏一路走到超时——
所以默认值改成了自动识别。

新增录像后重新核对（三条命令互不依赖，可只跑其中一条）：

```powershell
# 1) 结构与元数据 + 重复检测
python tools/probes/inventory_replays.py --dir th08_replay `
    --out third_party/th08/build/rl/probe-v103/replay-inventory.json
# 2) 真机回放验证（每份约 9 秒）
python tools/probes/check_replays.py --dir th08_replay --seconds 4
# 3) 重新生成本节的表格
python tools/probes/render_replay_table.py `
    --inventory third_party/th08/build/rl/probe-v103/replay-inventory.json --out replay-table.md
```

> 注意：这份录像库只解决「有样本」的问题。当前模型的瓶颈不是样本数量，而是
> 「一次前向推理能否代替 20 帧规划」的精度问题，结论与证据见
> `Document/Update/Alpha/v1.0.2/推进报告-教师与蒸馏.md`。
