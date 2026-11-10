# EXP-P0-058：WP8 三份真实回放 × 每份 200 帧双端像素配对（G0.4 规模扩展）

- 日期：2026-11-09
- 目标：把 EXP-P0-056 的「1 回放 × 48 帧」扩到 G0.4 要求的「**3 份真实回放 × 每份 200 帧**」，
  给出每份与合计统计、最差帧清单、人工目视复核结论与 P3-WP4 部署增强建议。
- 前置：EXP-P0-056（首例，Stage 1 开场 48 帧）、EXP-P0-043（fail-closed）、EXP-P0-047/053
- 环境：
  - Windows `sys.platform=win32`（RTX 4070 SUPER / 驱动 617.42，windowed 640×480，
    D3D8 COPY，no vsync），Python 3.12.10
  - WSL Ubuntu，私有 Xvfb（`xvfb-run -a`，实际分配 `:107`）+ Mesa 软件 GL
    （`LIBGL_ALWAYS_SOFTWARE=1`、`LP_NUM_THREADS=1`）
- Windows 侧构建：`third_party/th08/build_fence/th08.exe`，905,728 B，
  SHA256 `bc1894fed48beac739f628836629b8b1b1c1b63dbe0097239c2762cb4c8d8565`，
  `.text` `34A8070B4854672C92A66C95472F9DF7FB3B83C0AFEB6FBD15BBAED45A8593C7`，
  地址表 `79237a295567edf0bad6c557bcc1aaf8a7d1db369d2c54429fbc1b779b85634a`，
  `th08.map` `ce7b6672f9488909a5f700e6a52900c65358a4201b0de5e62ee904f8a752bb54`
- Linux 侧构建：`/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern`，
  SHA256 `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb`，
  源码 HEAD `101ce5eb9f442e9fbfd5e7e10e99495b4873d1ff`

## 结论

**完成。** 3 份不同回放 × 每份 200 个连续 `logicFrame` 全部完成 Windows + Linux 采集、
无损 PNG 转换与逐像素配对，共 **600 对帧**，三份 `pixel-report.json` 与一份
`suite-summary.json` 落盘。

合计（600 对，RGB 通道，未掩蔽 fps）：

| 区域 | 平均绝对差 | 差异像素比例 | 最大通道差 |
| --- | --- | --- | --- |
| playfield `x=[32,416)` | **0.143384** | **0.277006** | **132** |
| HUD（其余列） | **0.590419** | **0.178127** | **251** |
| 整帧去掉 fps 读数（附加） | 0.194470 | 0.133632 | 132 |
| HUD 去掉 fps 读数（附加） | 0.271838 | 0.130661 | **15** |

三条关键实测结论：

1. **没有任何一帧存在内容缺失。** 600 帧里 playfield 的「最差 8×8 格子」只有 4/64 个
   像素超过 16/255；≥64 的像素合计 344 个（110,592,000 个 playfield 像素样本的 3.11e-6），
   ≥128 的像素**全程只有 2 个**（各占 1 帧 1 像素）。弹幕、敌机、Boss、背景、HUD 文字与
   数值两侧逐一对应（见 §五 目视复核）。
2. **HUD 的 250/251 全部来自 fps 读数这一个矩形**（`y=[463,480)`、`x=[512,581)`）：
   Windows 侧 200 帧出现 **20 / 16 / 22 种**不同读数（真实被 hold 后的帧率，如 `19.08fps`），
   Linux 侧该矩形 200 帧**逐字节相同**（`60.00fps`，三份回放同 1 种）。
   **HUD 去掉该矩形后，600 帧内没有任何像素差超过 15/255**；整帧去 fps 后的最大值 132
   来自 playfield 里 §五(4) 的那 2 个孤立像素。
3. **跨侧状态在两侧都发布的字段上逐帧位相等**：`player_x`/`player_y`（浮点位模式）、
   `score`、`lives`、`power`、`player_state`、`stage` 在 **600/600 帧**全部相等，
   `difficulty`、`shot` 与回放头一致。这**不是**状态等式证明（两侧摘要算法不同），
   但它把「像素差异不能归因」的范围缩小到了摘要算法差异，而不是观测到的状态分歧。

**不宣称 G0 通过**：报告 `status = measured-not-G0-pass`；
`--state-pairing same-logic-frame` 下 `state_consistency = not-asserted`；
strict 默认在同一批真实数据上 **exit 3**（三份均如此，且不写报告）。

## 一、回放与窗口选择

### 1.1 三份回放

| 用例 | 回放文件 | SHA256 前缀 | 难度/机体 | stage | 该 stage 输入帧数 |
| --- | --- | --- | --- | --- | --- |
| `r1-normal-stage4a` | `th08_replay/Th8_ud2970.rpy` | `e621086d…` | Normal / Marisa（solo, shot 6） | Stage 4A（idx 3） | 27,633 |
| `r2-lunatic-stage5` | `th08_replay/Th8_ud0059.rpy` | `62b02a17…` | Lunatic / Youmu（shot 10） | Stage 5（idx 5） | 30,854 |
| `r3-hard-stage4b` | `th08_replay/Th8_ud2563.rpy` | `4457c855…` | Hard / Reimu（solo, shot 4） | Stage 4B（idx 4） | 29,714 |

三份都是零售原件录制的完整通关回放（`executable` 元数据**未记录**：套件 JSON 未保存
`executable` 元数据；`extended=False`，非 slow-mode）；回放 SHA-256 在 Windows 与 Linux 两侧逐份一致
（见 `pixel-report.json` 的 `replay_sha256`）。

### 1.2 窗口规则（在与像素比较无关的前提下预先固定）

任务要求「不要都取 Stage 1 开场标题卡」，且至少 1 份含弹幕/Boss。Windows 侧回放必须从
stage 起点实时播放，而游戏本体受 60fps 限制，因此在 180 s 预算内可到达的深度有限
（约 4000–6000 logicFrame）。为保证窗口**确实有弹幕**，采用一条**在像素比较之前**就固定的
内容规则：

> 从 `startLogicFrame = 1800` 起，取**第一个** `activeBullets ≥ 40` 的帧作为窗口起点，
> 上限 `maxSeekFrame = 4000`；整段 seek 过程的弹幕剖面（每 60 帧一点）原样记录在
> Windows manifest 的 `capture.seek.profile` 里。

`activeBullets` 来自栅栏 telemetry，本身与像素无关；规则在采集前写死在
`tools/rl/v400_wp8_pair_suite.py` 的 `DEFAULT_SPECS` 中，且被单元测试固定
（`test_load_specs_accepts_the_builtin_three_replays` 要求 3 个 `(replay, stage)` 互不相同）。

实测触发情况（`suite-summary.json` → `windowRule`）：

| 用例 | stoppedBy | 触发帧 | 触发时弹幕数 | seek 期间最大弹幕 |
| --- | --- | --- | --- | --- |
| r1 | `bullet-threshold` | 2351 | 43 | 69 |
| r2 | `bullet-threshold` | 1800 | 49 | 608 |
| r3 | `bullet-threshold` | 2010 | 51 | 138 |

三份窗口的**每一帧都有弹幕**（`framesWithBullets = 200/200`），密度如下：

| 用例 | activeBullets min / mean / max | 分段均值（每 20 帧） |
| --- | --- | --- |
| r1 | 39 / 59.4 / 74 | 42.7 → 68.9（递增） |
| r2 | 44 / 139.4 / **206** | 46.3 → 204.3 → 184.7 |
| r3 | 37 / 47.0 / 74 | 50.3 → 40.1 |

即覆盖：**4A 中段弹幕**、**5 面 Lunatic 高密度弹幕（峰值 206 发）**、**4B Hard 弹幕**，
没有任何一份是 Stage 1 开场标题卡。

### 1.3 实际窗口

| 用例 | logicFrame 区间 | 帧数 | 连续 |
| --- | --- | --- | --- |
| r1 | **2352–2551** | 200 | 是（`logicFramesConsecutive=true`） |
| r2 | **1801–2000** | 200 | 是 |
| r3 | **2011–2210** | 200 | 是 |

三份的 stage 索引由 Windows 侧 `state.stage` 回读断言（`observedStage` = 3 / 5 / 4），
Linux 侧由 `--stage` 传同一整数并在采集内断言 `f1 == stage`。**两侧 stage 语义同一**：
Windows 回放菜单的 stage 光标值就是引擎 stage 索引（`TitleScreen.cpp`
`TITLE_REPLAY_MENU_SELECT_STAGE` 直接以 stage 索引作为光标），Linux 侧
`TH08_AUTOPLAY_STAGE` 亦为 stage 索引（`TitleScreen.cpp:211` 读入后
`while (stage < MAX_STAGES && TH08_REPLAY_STAGE_DATA(replay, stage) == NULL) ++stage;`）。

## 二、双端采集

### 2.1 Windows（研究栅栏构建）

- 每帧在 `GameWindow::Render` 末尾被栅栏 hold，读状态、GDI 抓屏、fence backbuffer 三者
  在同一 hold 内完成；13 项校验（`hold_stable`/`pixels_frozen`/`pixels_match_backbuffer`/
  `bullet_digest`/`enemy_digest`/`gameplay_frame`/`stage_frames`/…）**600/600 帧全通过**。
- 采集前均确认无残留 `th08.exe`（工具内 `residual_game_processes()` 主动拒绝启动）。
- 单次墙钟 **62.4 / 54.2 / 56.6 s**（工具自带 `--deadline-seconds 170` 硬闸刀，任务上限 180 s）。
- 采样器 200 次 `next()`、0 次重试、0 次 hold 预算跳过
  （`sampler.samples=200, retries=0, attempts=200`；`holdBudgetSkips=0`）。
- 栅栏 hold 数：r1 2308、r2 1759、r3 1969。
- PNG：RGBA 640×480，alpha 为常量 255（栅栏像素源是 RGB24），比较只用 RGB。

### 2.2 Linux（env 角色 + 私有 Xvfb + 私有 data-dir）

- `TH08_AUTOPLAY_REPLAY` + `TH08_AUTOPLAY_STAGE` 生效，`replay_start` 三份均为
  `isReplay=1`、`replay_frame_counter=0`、`gameplay_frame_counter=0`、
  `stage` 分别 3 / 5 / 4，`difficulty` 1 / 3 / 2，`shot` 6 / 10 / 4（与回放头一致）。
- 每帧取共享块 RGBA 原始字节 + 25 字段状态，`HASH` 回复与共享块 `stateHash`
  逐帧交叉校验、`header.frame` 与 `STATE frames` 逐帧校验。
- 私有 Xvfb：`--xvfb auto` 强制重新 exec 到 `xvfb-run -a`（不信任继承的 `DISPLAY=:0`）。
- **私有 `--data-dir`**：`/root/v400-data/runtime3/<case>/game`，其中 `score.dat`、`th08.cfg`
  从 `/root/v400-data/game/*.orig` 字节复制，`th08.dat`、`thbgm.dat` 只读符号链接
  （原由见 §七）。三份采集内 `replay_start`/`stage` 断言全通过。
- 墙钟：采集 **42.7 / 30.8 / 37.6 s**，引擎子进程 `exit_code = 0`。

### 2.3 转 RGBA PNG（Windows 侧）

- `convert` 校验 raw SHA256 未变 → 行翻转（OpenGL 自下而上）→ 无损 RGBA PNG →
  回读 `array_equal` → 模式/尺寸；三份均 `exit 0`，各 200 张。
- 原始 SHM RGBA 保留在 `<case>/linux/raw/*.rgba`（每份 200 × 1,228,800 B）。

## 三、配对比较

### 3.1 每份统计（200 对，RGB，fps 未掩蔽）

| 用例 | playfield MAD | playfield 差异比例 | playfield maxch | HUD MAD | HUD 差异比例 | HUD maxch | 整帧去 fps MAD | 整帧去 fps maxch | HUD 去 fps MAD | HUD 去 fps maxch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r1 Stage 4A | **0.138379** | 0.253777 | **132** | 0.584623 | 0.178727 | 251 | 0.191987 | 132 | 0.273173 | 15 |
| r2 Stage 5 | **0.153007** | 0.318349 | **93** | 0.595456 | 0.176763 | 250 | 0.198640 | 93 | 0.267750 | 15 |
| r3 Stage 4B | **0.138766** | 0.258893 | **131** | 0.591176 | 0.178890 | 251 | 0.192784 | 131 | 0.274592 | 15 |
| **合计（按帧数池化）** | **0.143384** | **0.277006** | **132** | **0.590419** | **0.178127** | **251** | **0.194470** | **132** | **0.271838** | **15** |

逐帧范围（min–max）：

| 用例 | playfield MAD | playfield 差异比例 | HUD MAD |
| --- | --- | --- | --- |
| r1 | 0.124982 – 0.162109 | 0.229291 – 0.305203 | 0.320603 – 0.712242 |
| r2 | 0.145132 – 0.166511 | 0.303434 – 0.337918 | 0.262465 – 0.693164 |
| r3 | 0.122528 – 0.181682 | 0.229725 – 0.340847 | 0.272024 – 0.700616 |

与 EXP-P0-056 的 48 帧（playfield 0.130595）相比，弹幕窗口的 playfield MAD 只上升
0.008–0.023（0.138–0.153），说明**内容密度本身几乎不改变渲染差量**；差异是逐像素的
光栅化/取整噪声，而不是随画面复杂度放大的系统性偏差。

### 3.2 最差 10 帧

「最差」按任务口径 = `max(playfield MAD, HUD MAD)`（fps 未掩蔽）。

| # | 用例 | logicFrame | playfield MAD | HUD MAD | 整帧去 fps MAD | maxch |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | r1 | 2474 | 0.12623 | **0.71224** | 0.18622 | 71 |
| 2 | r1 | 2469 | 0.14347 | 0.71128 | 0.19622 | 87 |
| 3 | r1 | 2468 | 0.14259 | 0.71065 | 0.19544 | 51 |
| 4 | r1 | 2470 | 0.14502 | 0.71063 | 0.19689 | 46 |
| 5 | r1 | 2475 | 0.12756 | 0.71039 | 0.18628 | 21 |
| 6 | r1 | 2472 | 0.14341 | 0.70983 | 0.19560 | 88 |
| 7 | r1 | 2476 | 0.12741 | 0.70958 | 0.18586 | 60 |
| 8 | r1 | 2471 | 0.14398 | 0.70931 | 0.19574 | 75 |
| 9 | r1 | 2473 | 0.13226 | 0.70898 | 0.18854 | 65 |
| 10 | r3 | 2177 | 0.13513 | 0.70062 | 0.19133 | 41 |

这 10 帧的共性：HUD MAD 高（0.70–0.71）而 playfield MAD 正常，整帧去 fps 后回落到
0.186–0.197 —— 即它们**只是 Windows 侧那一刻 fps 读数跳变**（hold 后帧率掉到 ~15 fps），
不是画面变差。这也是为什么下面的目视复核不用这个口径选帧。

按 **playfield MAD** 的最差 10 帧全部落在 r3 的 2060–2070（0.17208–0.18168），
按 **整帧去 fps** 的最差 10 帧同样集中在该段（0.21278–0.21900），第二高为 r2 的
0.20769。该段是 4B 面的一段高密度背景+弹幕叠加画面。

### 3.3 严格模式对照（同一批真实数据）

三份均：`python tools\rl\v400_pixel_pair.py --windows … --linux … --frames 200 --out …-strict.json`
→ **exit 3**，stderr `BLOCKED: frame <N>: state mismatch; cannot attribute pixel difference`，
且**未写出报告文件**（`reportWritten=false`）。默认 strict 未被放宽。

## 四、跨侧字段级状态核对（600/600 帧）

两侧摘要算法不同（Windows `bulletDigest`/`enemyDigest`，Linux 25 字段 FNV-1a），
`same-logic-frame` 只强制 replay SHA 与 `logicFrame` 集合。本卡**额外**做了一次字段级核对
（`tools/rl/v400_wp8_pair_suite.py::state_cross_check`），比较双方都原样发布、且可由
同一位模式判等的字段：

| 字段 | 判定 | r1 | r2 | r3 | 合计 |
| --- | --- | --- | --- | --- | --- |
| `player_x` | 浮点位模式相等 | 200/200 | 200/200 | 200/200 | **600/600** |
| `player_y` | 浮点位模式相等 | 200/200 | 200/200 | 200/200 | **600/600** |
| `score` | 整数相等（`f4`） | 200/200 | 200/200 | 200/200 | **600/600** |
| `lives` | 浮点位模式相等（`f7`） | 200/200 | 200/200 | 200/200 | **600/600** |
| `power` | 浮点位模式相等（`f9`） | 200/200 | 200/200 | 200/200 | **600/600** |
| `player_state` | 整数相等（`f13`） | 200/200 | 200/200 | 200/200 | **600/600** |
| `stage` | 整数相等（`f1`） | 200/200 | 200/200 | 200/200 | **600/600** |
| `logicFrame` | 相等（`f3`） | 200/200 | 200/200 | 200/200 | **600/600** |
| `difficulty`（回放头 vs `f24`） | 相等 | 1=1 | 3=3 | 2=2 | 一致 |
| `shot`（回放头 vs `f23`） | 相等 | 6 | 10 | 4 | 一致 |

`mismatches = 0`（三份）。**这不是状态等式证明**：`graze`、子弹/敌机身份、RNG 种子
等只有一侧发布或两侧口径不同，因此报告仍写 `state_consistency = not-asserted`，
本表只把结论限定为「双方都发布的这些字段在 600 帧上完全一致」。

## 五、人工目视复核

### 5.1 方法

1. **选帧规则**（在被比较的测量值上固定，不看图）：`select_review_frames()` 取
   ①每份回放「整帧去 fps MAD」最差的 1 帧，②再取全局 playfield MAD 最高、且未被①选中的
   3 帧。避免用「HUD MAD」选帧，否则只会选中 fps 跳变帧。
2. 每帧生成三联图：`Windows | Linux | 8×放大的绝对差`，底部再叠 4× 放大的「最差 40×40 块」。
3. 对差异最大的位置再做定点放大（6–8×，三联），并做连通域/阈值解剖。

产出的图（均已实际读取，路径见 §十一）：

| 图 | 内容 |
| --- | --- |
| `per-replay-r1-normal-stage4a-f02400.png` | r1 最差帧全幅三联 |
| `per-replay-r2-lunatic-stage5-f01987.png` | r2 最差帧全幅三联 |
| `per-replay-r3-hard-stage4b-f02067.png` | r3 最差帧全幅三联 |
| `global-03/04/05-r3-f02068/02070/02069.png` | 全局 playfield 最差三帧 |
| `r1-f2400-hud-text.png` | HUD 顶部文字区 4×（x 416–640, y 0–140） |
| `r1-f2400-fps.png` | fps 读数 8× |
| `r1-f2495-px-375-244.png` | r1 最大通道差像素 8×（playfield 内孤立点） |
| `r3-f2101-px-415-86.png` | r3 最大通道差像素 8×（playfield 右边界） |
| `r2-f1846-px-73-412.png` | r2 最大通道差像素 8× |
| `r2-f1987-playfield.png` | r2 最差帧弹幕区 3×（逐发比对） |
| `r1-f2400-playfield.png` | r1 最差帧弹幕区 3× |

### 5.2 目视结论（逐图）

**（1）内容一致，无缺失。** r1 f2400、r2 f1987、r3 f2067 三张全幅三联图里，两侧的
弹幕（蓝色圆弹、粉色/白色蝶弹、白粉爆散环）、Boss 立绘、背景（竹林/木板走廊）、
玩家自机、判定点、左上角 `HiScore`/`Score` 数字、`Player`/`Spell` 星数、`Power MAX`、
`Graze`、`Point`、`Time`、难度标签、标题 logo、左下角减速指示（`-50%-0%`、`X-97`）
**逐个对得上、位置一致、数值一致**。r2 f1987 的 3× 弹幕区放大图里，两发的蝶弹、
`+1`/`+3` 增益数字、蓝色大圆、`水` 字弹幕逐一对位，差分面板几乎全黑。

**（2）唯一成片的高差是 fps 读数，属于 HUD 数值差异，不是渲染差异。**
8× 放大图 `r1-f2400-fps.png` 直接读出：Windows `19.08fps`，Linux `60.00fps`，
差分面板只留下数字笔画。定量佐证：Linux 侧该矩形在 200 帧里**逐字节相同**
（三份回放同一个 SHA，即恒定 `60.00fps`），Windows 侧却有 20 / 16 / 22 种不同读数。
这是栅栏 hold 之后打印真实帧率的必然结果：`hold_every=1` 把每帧压到 ~15–19 fps。

**（3）HUD 文字/面板是「同一字形、不同取整」，不是字体差异。**
`r1-f2400-hud-text.png` 里两侧 `HiScore 0762271050` 的字形、字宽、字距、位置完全一致；
差分面板呈现沿字边的高频散点（放大 8 倍后才可见），而不是字形轮廓整体偏移或粗细变化。
定量：HUD 去掉 fps 矩形后，600 帧内**没有任何像素差超过 15/255**，逐帧平均差
0.268–0.275（约 13% 的 HUD 像素有 1–15/255 的差）。这是**同一字模在两条渲染栈上的
覆盖/混合取整差**，不是「系统缺字体」或「换字体」。

**（4）playfield 的最大通道差是孤立单像素，不是缺失对象。**
`r1-f2495-px-375-244.png`：全帧最大 132 的那一像素位于竹子背景纹理的高对比边缘，
差分面板只有一个孤立亮点，周围全黑；`r3-f2101-px-415-86.png`：131 的那一像素落在
playfield 右边界列 `x=415`，同样是单点。定量：600 帧里 ≥128 的像素**总共 2 个**
（各 1 帧 1 像素），≥64 的最多 4 个/帧，最差 8×8 格子只有 4/64 个像素 >16。
若是「漏画一发弹/一个敌机」，必然出现成块（数十像素）的高差区域——实测不存在。

**（5）r2 有固定位置的低幅差（≥16 但 <64）。** `r2-f1846-px-73-412.png` 及
`crop-playfield-0`（y 416–464, x 41–89）显示：两侧左下角减速指示 `-50.0%` / `X-97`
文字完全相同，差集中在几个固定坐标 `(412,73)`/`(419,71)`/`(440,65)`/`(391,79)`
（200/200 帧都出现）。因为该指示是**静止**的，同一处字形边缘在每个帧都产生同样的
取整差——这是位置固定性，不是内容变化。

**综上：差异性质 = fps 数值差 + 字形/面板混合取整差 + 背景纹理边缘的孤立单像素取整差。
没有一处是内容缺失，也没有缩放/位移证据（≥76% 像素逐位相同，99% 的像素差 ≤4/255，
99.9% ≤9/255，见 §5.3 包络）。**

### 5.3 差异解剖（600 帧定量）

以「整帧去掉 fps 读数」为口径（550,848,600 个像素样本）：

| 指标 | R | G | B |
| --- | --- | --- | --- |
| 平均绝对差 | 0.0778 | 0.0586 | 0.0580 |
| RMS | 0.5204 | 0.3145 | 0.3400 |

max-channel 分位数（去 fps 矩形）：p50 = 0，p90 = 1，p99 = 4，p99.9 = 9，p99.99 = 13，
最大 132。**逐位相同**的像素占 **76.4%**；playfield 内逐位相同占 **72.3%**，
≥16 的仅占 2.1e-5。

playfield 阈值统计（playfield 为 184,320 px/帧，每份 200 帧 × 184,320 px = 36,864,000 px；307,200 px 是整帧像素）：

| 用例 | ≥16 帧数 / 像素数 | ≥64 帧数 / 像素数 | ≥128 帧数 / 像素数 | 最差 8×8 格子 |
| --- | --- | --- | --- | --- |
| r1 | 199/200，1117 px | 49/200，55 px | **1/200，1 px** @ (244,375) | 4/64 |
| r2 | 200/200，1745 px | 134/200，232 px | **0/200，0 px** | 2/64 |
| r3 | 199/200，1029 px | 50/200，57 px | **1/200，1 px** @ (86,415) | 3/64 |

「最差 8×8」的定义：在 playfield（`x=[32,416)`）内**滑窗 8×8、步长 1 像素**，统计每个窗口内
max-channel 差 **>16/255** 的像素数，取全帧最大值（上限 64）。

HUD 去掉 fps 矩形后，600 帧里最差 8×8 格子 = **0/64** 个像素 >16。

## 六、P3-WP4 部署增强建议

**依据**：§5.3 的包络。部署侧（P3-WP4）要做的增强应当**只覆盖实测存在的差异**；
下面每个强度都给出「实测锚点」和「超出实测多少倍」的判断。

| 增强项 | 实测锚点 | 建议强度 | 理由 / 不该做什么 |
| --- | --- | --- | --- |
| **滤波（模糊/AA 抖动）** | maxch p99.9 = 9/255，p99.99 = 13/255；playfield 通道 RMS **0.408–0.476**（R 0.4547 / G 0.4758 / B 0.4080）；整帧去 fps 通道 RMS **0.314–0.520**（R 0.5204 / G 0.3145 / B 0.3400）；逐位相同 72–76% | 高斯 σ ≤ **0.3 px**，概率 ≤ **0.1**；或 3×3 均值核权重 ≤ **0.02** | 差异是**逐像素取整**、集中在高对比边缘，成片结构为零。σ=0.5 px 的模糊会把高对比边缘改 20–60/255，是实测包络的 2–6 倍；σ≥1 px 属于另一个分布 |
| **色彩（亮度/对比/gamma）** | 三通道平均差 0.078/0.059/0.058（无通道偏置）；HUD 去 fps 后 600 帧内 maxch ≤ **15** | 亮度/对比抖动 ≤ **±2/255**（≈±0.8%）；每通道增益 ≤ **±0.5%**；gamma ≤ **±0.02**；概率 ≤ **0.3** | 两侧没有系统性色偏（RGB 平均差同量级），所以「大色偏」不是真实差异。±5% 增益 = ±12.75/255，已是 HUD 上界的 6 倍、平均差的 25 倍。**不要**做 HSV 大范围色相旋转（实测无证据） |
| **缩放 / 几何** | 内容逐一对位；99% 像素差 ≤4/255；无重采样环、无块状位移；最差情况是边界列 x=415 的单像素 | 等比缩放 ∈ **[0.995, 1.005]** 且概率 ≤ **0.05**；或 1 px 平移概率 ≤ **0.1**；禁止非等比拉伸 | 任何 0.9–1.1 的 random-resized-crop 都会产生 20–100 倍于实测的几何差异，并破坏「同一 logicFrame 同一画面」的前提 |
| **JPEG / 压缩噪声** | 差异**无 8×8 块结构**（最差 8×8 格子 4/64）、无振铃；p99.9 = 9/255 | 质量 ∈ **[90, 100]**，概率 ≤ **0.2**；**不要**低于 85 | JPEG q=95 的平滑区误差 ~1–2/255（与实测同量级），q=75 的边缘误差 ~5–15/255 且引入 8×8 块结构——实测里根本不存在这种结构，属于分布外。若必须模拟低质量，应按「块内均值保持 + 边缘抖动」另建，而不是直接压 JPEG |
| **HUD fps 矩形**（工程建议，非增强） | Windows 20/16/22 种读数 vs Linux 恒定；该矩形是唯一 >128 差异来源 | **在预处理里掩蔽或把该矩形归一化**（`y=[463,480)`、`x=[512,581)`，或按 ms 定时重定位） | 这是**数值差**，训练/部署侧不该把它当视觉信号；掩蔽后整帧 maxch 从 251 降到 132，HUD MAD 从 0.590 降到 0.272 |
| **总体结论** | playfield MAD 0.143、76% 像素逐位相同 | 现有像素差**不构成部署阻塞**；增强强度的量级应是「±1–2/255 级噪声 + 极轻几何抖动」，而不是常规 ImageNet 式强增广 | 现在的训练增广若用 ±10% 亮度、随机缩放 0.8–1.2、JPEG q=50，其强度是两条真实渲染栈差异的 **10–100 倍**，会把「同一帧」教成「不同帧」 |

补充两条配套建议：

1. **不要用 HUD MAD 做回归门限**。它的方差几乎全部来自 fps 读数（本批最差 10 帧的
   HUD MAD 0.70–0.71 全部是 fps 跳变帧）。门槛应建在
   `whole_frame_excluding_fps_readout` 上（推荐 ≤ 0.25，实测 0.192–0.199）。
2. **把字段级核对纳入常规门禁**。`state_cross_check` 的 8 个字段在 600/600 帧相等，
   成本是读两份 manifest；一旦某份回放出现 `score`/`player_x` 不等，说明两侧**状态已经
   分歧**，此时像素报告不可用于归因，应直接 fail-closed。

> 说明：以上强度是**从实测包络反推的建议值**，尚未做训练侧 A/B 验证；
> 它们回答的是「多大强度还算在真实跨栈差异之内」，不是「多大强度能提升泛化」。

## 七、共享模板污染与整改（必须单列）

**事实经过**：

1. 本轮最初（第一次 Linux 采集）沿用了 EXP-P0-056 的调用方式，没有传 `--data-dir`，
   于是 `tools/rl/v400_linux_frame_capture.py` 用了默认
   `--data-dir /root/v400-data/game` —— 这是**共享、本应只读**的零售数据目录。
   TH08 在回放结束/退出时会写回 `score.dat`（HiScore）与 `th08.cfg`，因此共享模板被**实际改写**。
   现场证据（WSL）：`score.dat.dirty-20261109-001521`（16,049 B）、
   `score.dat.dirty-20261109-002226`（17,776 B，对应本轮第二次运行）、
   `th08.cfg.dirty-20261109-001521`（60 B）。共享模板随后被恢复为 `.orig` 值。
2. **受污染的数据一律作废并重采**。第一代 Linux 采集（`--data-dir` 指向共享模板，
   HiScore 像素可能被写脏的模板影响）整体重命名作废，不进入任何统计：
   - `.tmp_v400_wp8_pair3_suite/<case>/linux.dirty-shared-template-20261109/`
   - `.tmp_v400_wp8_pair3_suite/suite-steps-linux.dirty-shared-template-20261109.json`
3. 第二代整改（私有 data-dir）虽然数据正确，但**私有目录溯源步骤自身失败**
   （脚本里 `sha256sum … readlink -f …` 少了分隔符，`sha256sum: invalid option -- 'f'`），
   无法自证「私有目录里的 `score.dat` 是从 `.orig` 复制来的」。按同一原则作废重采：
   - `.tmp_v400_wp8_pair3_suite/<case>/linux.rerun1-unproven-runtime-20261109/`
   - `.tmp_v400_wp8_pair3_suite/suite-steps-linux.rerun1-unproven-runtime-20261109.json`
4. 第三代（本卡 §二/§三 的正式证据）完成整改并通过全部溯源断言。

**整改措施（已落地在 `tools/rl/v400_wp8_pair_suite.py`）**：

- `WSL_SHARED_DATA` 只作为**复制来源**；每个用例一份私有
  `/root/v400-data/runtime3/<case>/game`：
  - `score.dat`、`th08.cfg` ← `…/game/*.orig` 的**字节复制**；
  - `th08.dat`、`thbgm.dat` ← **只读符号链接**（约 500 MB，不复制、不改）；
  - 复制后立刻 `sha256sum` 并把摘要写进 `linux-runtime.json`（`pristineEqual` 必须为 true）。
- 采集命令必须带 `--data-dir '<私有目录>'`；单元测试
  `test_linux_capture_passes_the_private_data_dir` **断言**命令里是私有路径、
  且**不得**出现 `--data-dir /root/v400-data/game`。
- 每个用例采集前后各做一次共享模板 `sha256sum`（`shared-template-before/after-<case>`、
  末尾 `shared-template-after`），与 `.orig` 摘要比对；任一不符 → `cmd_linux` 返回非 0，
  **报告而不修复、不删除**任何文件。
- 每次采集后再对私有目录做 `sha256sum`（`runtime-after`）。实测三份都是
  `rewritten = ["score.dat"]` —— 这正是「游戏确实写盘了，但写在了私有副本里」的证据。

**整改后复核（本卡正式 run）**：

```text
shared-template-before  score.dat 60e49abc8f0a955aa8fa883769a30c66913417be453632c9a8a99b0b8a5c5191
                        th08.cfg  48bf7621da5879ef5ec1392fcac2cc0a85c313912a24fb33b4cddccf0d28f07a
shared-template-after-r1 / r2 / r3（每份采集后）both clean
shared-template-after   score.dat 60e49abc…（= .orig）
                        th08.cfg  48bf7621…（= .orig）
```

**结论：共享模板已恢复且在本轮正式采集中未被再次写入**（`before.clean = true`、
`after.clean = true`、三份逐用例 clean、`failures = []`）。
残留的 `*.dirty-*` 备份文件**未删除**，原样留在 `/root/v400-data/game/` 供审计。

## 八、命令与退出码

所有步骤由套件统一执行并把「完整命令行 / 退出码 / 墙钟」写入
`.tmp_v400_wp8_pair3_suite/suite-steps-<step>.json`。

### 8.1 套件入口

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python tools\rl\v400_wp8_pair_suite.py --out .tmp_v400_wp8_pair3_suite plan      # exit 0
python tools\rl\v400_wp8_pair_suite.py --out .tmp_v400_wp8_pair3_suite windows   # exit 0
python tools\rl\v400_wp8_pair_suite.py --out .tmp_v400_wp8_pair3_suite linux     # exit 0（正式 run；前两代分别 exit 1，已作废）
python tools\rl\v400_wp8_pair_suite.py --out .tmp_v400_wp8_pair3_suite compare   # exit 0
python tools\rl\v400_wp8_pair_suite.py --out .tmp_v400_wp8_pair3_suite summary   # exit 0
```

### 8.2 Windows 采集（三次，均 exit 0）

```powershell
python tools\rl\v400_win_fence_capture.py --out .tmp_v400_wp8_pair3_suite\r1-normal-stage4a\windows `
  --run-dir .tmp_v400_wp8_pair3_suite\r1-normal-stage4a\windows-run `
  --replay-file D:\WorkShop\Touhou_RL\th08_replay\Th8_ud2970.rpy `
  --replay-stage 3 --start-logic-frame 1800 --seek-bullet-threshold 40 --max-seek-frame 4000 `
  --frames 200 --label r1-normal-stage4a
```

| 用例 | 退出码 | 墙钟 | framesCaptured | logicFrame | 连续 | 13 项校验 |
| --- | --- | --- | --- | --- | --- | --- |
| r1 (`Th8_ud2970`, stage 3) | 0 | 62.376 s（工具内 59.067 s） | 200 | 2352–2551 | 是 | 200/200 |
| r2 (`Th8_ud0059`, stage 5) | 0 | 54.161 s（工具内 51.003 s） | 200 | 1801–2000 | 是 | 200/200 |
| r3 (`Th8_ud2563`, stage 4) | 0 | 56.566 s（工具内 53.394 s） | 200 | 2011–2210 | 是 | 200/200 |

单次均 ≤ 180 s（工具自带 `--deadline-seconds 170`；实测最大 62.4 s）。
运行前后 `Get-Process th08` 均无输出（无残留进程）。

### 8.3 Linux 采集 + 转换（三次，均 exit 0）

```powershell
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && python3 tools/rl/v400_linux_frame_capture.py capture \
  --out .tmp_v400_wp8_pair3_suite/r1-normal-stage4a/linux \
  --targets-file .tmp_v400_wp8_pair3_suite/r1-normal-stage4a/targets.json \
  --replay '/root/v400-data/pair3/Th8_ud2970.rpy' --stage 3 \
  --data-dir '/root/v400-data/runtime3/r1-normal-stage4a/game'"          # exit 0，42.703 s

python tools\rl\v400_linux_frame_capture.py convert \
  --capture .tmp_v400_wp8_pair3_suite\r1-normal-stage4a\linux\capture.json  # exit 0，8.951 s
```

| 用例 | capture exit / 墙钟 | convert exit / 墙钟 | framesCaptured | 引擎 exit |
| --- | --- | --- | --- | --- |
| r1 | 0 / 42.703 s | 0 / 8.951 s | 200 | 0 |
| r2 | 0 / 30.790 s | 0 / 9.882 s | 200 | 0 |
| r3 | 0 / 37.608 s | 0 / 8.715 s | 200 | 0 |

### 8.4 配对比较（三次，均 exit 0）

```powershell
python tools\rl\v400_pixel_pair.py `
  --windows .tmp_v400_wp8_pair3_suite\r1-normal-stage4a\windows\windows-manifest.json `
  --linux   .tmp_v400_wp8_pair3_suite\r1-normal-stage4a\linux\linux-manifest.json `
  --frames 200 --state-pairing same-logic-frame `
  --out .tmp_v400_wp8_pair3_suite\r1-normal-stage4a\pixel-report.json      # exit 0（8.385 s）
```

报告头：`schema=v400-wp8-paired-pixel-audit/2`、`status=measured-not-G0-pass`、
`state_pairing=same-logic-frame`、`state_consistency=not-asserted`、`pairs=200`。
`r2` / `r3` 同形（8.260 s / 8.243 s）。

### 8.5 严格模式对照（三次，均 exit 3）

```powershell
python tools\rl\v400_pixel_pair.py --windows …\windows-manifest.json --linux …\linux-manifest.json `
  --frames 200 --out .tmp_v400_wp8_pair3_suite\r1-normal-stage4a\pixel-report-strict.json
```

- 三份退出码均 **3**，stderr `BLOCKED: frame <N>: state mismatch; cannot attribute pixel difference`
- 三份均**未写出** `pixel-report-strict.json`（`reportWritten = false`，fail-closed）

### 8.6 单元测试

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_wp8_pair3_pytest -q tests/test_v400_wp8_pair_suite.py
```

- 退出码 **0**，**28 passed**（含：窗口连续性/重复/乱序拒绝、跨侧字段核对与一 ULP 邻居检出、
  池化统计、审查选帧去重、三联图布局、私有 `--data-dir` 回归、共享模板 SHA 校验、
  子进程退出码记录）

## 九、变更文件

| 文件 | 变更 |
| --- | --- |
| `tools/rl/v400_win_fence_capture.py` | **新增**。把上一轮临时脚本 `.tmp_v400_wp8_pair_win_capture.py` 正式化：精确 seek 到 `--start-logic-frame`（逐 hold ack，不读 bulk state）、预登记的弹幕密度窗口规则（`--seek-bullet-threshold` / `--max-seek-frame` / `--profile-every`）、stage 索引断言、残留 `th08.exe` 拒绝启动、`--deadline-seconds` 硬闸刀、`logicFrame` 连续性检查、13 项校验与完整 state evidence |
| `tools/rl/v400_wp8_pair_suite.py` | **新增**。`plan/windows/linux/compare/summary/all` 六步编排；私有 WSL `--data-dir` 构建与溯源；共享模板 before/after SHA 闸门；`windows_logic_frames()` 连续性校验（Linux 目标直接取自 Windows 实测帧号，两侧不可能漂移）；字段级跨侧核对；池化统计；最差帧三口径；审查选帧与三联图生成；每步命令行/退出码/墙钟落盘 |
| `tests/test_v400_wp8_pair_suite.py` | **新增**。28 项离线测试（合成 fixture 仅作测试输入，不进入证据目录） |
| `.tmp_v400_wp8_pair3_suite/`、`.tmp_v400_wp8_pair3_*` | 证据目录、审查图、解剖脚本与探针脚本（临时区，未提交） |
| `Document/Update/Alpha/v4.0.0/experiments/EXP-P0-058-WP8-three-replays-200f.md` | 本卡 |

**未改动**：`tools/rl/v400_pixel_pair.py`（严格默认 `strict` 与全部语义原样保留，
本卡三次 strict 对照即其未被放宽的证据）、`tools/rl/v400_linux_frame_capture.py`
（沿用其既有 `capture`/`convert`，只通过命令行给它私有 `--data-dir`）、
`third_party/th08` 源码与构建、零售原件、依赖与协议/ABI；未删除、未提交任何仓库文件
（作废的采集代次是**重命名**保留，不是删除）。

## 十、限制（必读）

1. **Windows 侧是研究构建，不是零售原件。** `build_fence/th08.exe`（905,728 B）自带帧栅栏
   插桩并重新编译；零售原件（840,704 B）本轮未参与。任何「Windows 行为 = 零售行为」的推断
   在本卡中都不成立。
2. **状态一致性只到字段级，不是证明。** Windows 发布 `bulletDigest`/`enemyDigest`
   与玩家位模式/标量，Linux 发布 25 字段 + FNV-1a；两者不是同一摘要函数。§四 的 8 字段
   600/600 相等是**强时间对齐 + 部分状态一致**证据，但 `graze`、子弹身份、RNG 种子
   未做跨侧比较，因此像素差异仍**不能完全归因于渲染**。报告一律写
   `state_consistency = not-asserted`。
3. **两侧渲染栈不同。** Windows 是 D3D8 + 真实 GPU（RTX 4070 SUPER / 617.42）；
   Linux 是 Xvfb + Mesa **软件** GL（`LIBGL_ALWAYS_SOFTWARE=1`、`LP_NUM_THREADS=1`）。
   因此 0.143 的 playfield MAD 至少混合了「实现差异 + 驱动/光栅化差异」，本卡不能拆分这两者。
4. **HUD fps 读数按构造不同。** Windows 打印被 hold 后的真实帧率（本批 200 帧内 16–22 种读数，
   ~15–19 fps），Linux 恒定 `60.00fps`。HUD 主指标**仍按任务要求包含它**，
   看 HUD 数值时必须同时看 `hud_excluding_fps_readout`。
5. **像素来源不同。** Windows PNG 是栅栏 hold 期间的窗口 GDI 抓屏（并校验与游戏自身
   backbuffer 副本逐字节相同），alpha 是合成的常量 255（栅栏像素源是 RGB24）；
   Linux PNG 是 GL FBO 的 `glReadPixels` RGBA（alpha 为真实 GL alpha，在绘制后、
   Present 前由 `AfterDrawFrame` 发布）。比较只用 RGB。
6. **窗口深度受 180 s 预算限制。** Windows 回放必须实时播放（60fps 上限），本轮最远只到
   logicFrame 2551（约 42 s 游戏内时间）。Boss 全流程、符卡切换瞬间、死亡/Bomb、
   Practice/Spell Practice 起点、Stage 6/Extra 均未覆盖；每份 200 帧 = 3.3 s，
   只覆盖了该 stage 的一个 3 秒片段（虽有弹幕）。
7. **fps 矩形是经验值**（`y=[463,480)`、`x=[512,581)`，沿用 EXP-P0-056 实测并
   在 600 帧上再次成立：该矩形是唯一 >128 的差异来源）。换分辨率/字体/HUD 布局后
   必须重新测量，不能当常量。
8. **目视复核是单人、单轮。** 结论基于 3 张全幅三联图 + 8 张定点放大图 + 600 帧定量解剖；
   没有第二个人独立复核，也没有做逐帧全量人眼确认。
9. **不是 G0 通过。** 报告 `status = measured-not-G0-pass`；G0.4 仍需零售原件路径与
   （若验收要求）跨实现统一状态摘要。

## 十一、证据路径

- 汇总：`.tmp_v400_wp8_pair3_suite/suite-summary.json`（600 帧池化统计、最差三口径、
  `windowRule`、`crossCheck`、`sharedTemplate`、审查图清单）
- 每份报告：`.tmp_v400_wp8_pair3_suite/{r1-normal-stage4a,r2-lunatic-stage5,r3-hard-stage4b}/pixel-report.json`
- Windows：`.tmp_v400_wp8_pair3_suite/<case>/windows/windows-manifest.json` + `frames/*.png`
- Linux：`.tmp_v400_wp8_pair3_suite/<case>/linux/capture.json`、`linux-manifest.json`、
  `raw/*.rgba`、`frames/*.png`、`targets.json`
- 运行溯源：`.tmp_v400_wp8_pair3_suite/linux-runtime.json`
  （私有 data-dir、复制摘要、每用例共享模板前后 SHA、`runtime-after` 改写证据）
- 步骤记录：`.tmp_v400_wp8_pair3_suite/suite-steps-{windows,linux,compare,summary}.json`
- 差异包络：`.tmp_v400_wp8_pair3_suite/difference-envelope.json`（600 帧全量）
- 审查图：`.tmp_v400_wp8_pair3_suite/review/`（6 张三联图）、
  `.tmp_v400_wp8_pair3_suite/review-crops/`（7 张定点放大图）、
  `.tmp_v400_wp8_pair3_suite/anatomy/<case>-<frame>/anatomy.json` + `diff-x8.png` + `crop-*.png`
- 作废代次（保留，不参与统计）：
  `.tmp_v400_wp8_pair3_suite/<case>/linux.dirty-shared-template-20261109/`、
  `.tmp_v400_wp8_pair3_suite/<case>/linux.rerun1-unproven-runtime-20261109/`、
  `.tmp_v400_wp8_pair3_suite/suite-steps-linux.dirty-shared-template-20261109.json`、
  `.tmp_v400_wp8_pair3_suite/suite-steps-linux.rerun1-unproven-runtime-20261109.json`
- 解剖/探针脚本：`.tmp_v400_wp8_pair3_anatomy.py`、`.tmp_v400_wp8_pair3_crop.py`、
  `.tmp_v400_wp8_pair3_block_check.py`、`.tmp_v400_wp8_pair3_playfield_anatomy.py`、
  `.tmp_v400_wp8_pair3_envelope.py`、`.tmp_v400_wp8_pair3_fps_variation.py`、
  `.tmp_v400_wp8_pair3_runtime_probe.sh`

## 十二、剩余阻塞

1. **零售原件与授权采集**：本轮只有研究构建；零售 Windows 采集路径仍未打通
   （EXP-P0-043 的 fixture/data 缺口不变）。
2. **跨实现状态摘要**：要么把两侧映射到同一组可比较的状态投影，要么定义逐字段容差并
   写明证明范围。本卡 §四 的 8 字段核对是这一步的起点，但还不等于摘要统一。
3. **样本覆盖**：3 份回放 × 1 个 3 秒窗口；仍需 Boss 全流程、符卡切换、死亡/Bomb、
   Practice/Spell Practice 起点与 Stage 6/Extra。
4. **fps 掩蔽政策**：需在 WP8 验收口径里明确「HUD 主指标是否掩蔽 fps 读数」，
   否则 HUD 的 251 会长期主导最大通道差。
5. **独立目视复核**：本卡的单人复核需要第二人独立确认（尤其是「无内容缺失」这一结论）。
