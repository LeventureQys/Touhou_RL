# EXP-P0-062（v4.0.0 WP8）：零售原件 ↔ Linux harness 同 logicFrame 200 帧配对（G0.4 真零售对照第一步）

- 日期：2026-11-09
- Provider / 模型：DeepSeek / DeepSeek v4.1 flash
- 环境：Windows（`sys.platform=win32`，Python 3.12，numpy / Pillow）+ WSL2 Linux
  （`xvfb-run -a` 私有 Xvfb `:99`，软件 OpenGL，`LIBGL_ALWAYS_SOFTWARE=1`）
- 零售侧：`third_party/th08/resources/th08.exe`（**零售原件**，非研究构建）
  - 840,704 B，SHA256 `330fbdbf58a710829d65277b4f312cfbb38d5448b3df523e79350b879213d924`
  - PE：`machine=0x014C`、`subsystem=2`、`imageBase=0x400000`、`DllCharacteristics=0x0000`（无 `DYNAMIC_BASE`）
- 回放：`th08_replay/Th8_ud2970.rpy`（stage 0 / Stage 1，Normal / Marisa）
  - SHA256 `e621086d814fba4838a2c84e4081162524c3c7e84d1bc4ca3f6621f9ca52d96f`（两侧一致）
- 改动工具：
  - `tools/rl/v400_retail_win_capture.py`（新增 `windows-retail` manifest 输出、RGBA 帧写出）
  - `tools/rl/v400_pixel_pair.py`（新增 `--state-pairing common-projection`，默认 `strict` 不变）
  - `tests/test_v400_retail_win_capture.py`、`tests/test_v400_pixel_pair.py`（新增用例）
- 产物（全部在 `.tmp_v400_wp8_retail_pair*`，未提交、未删除）：
  - 零售：`.tmp_v400_wp8_retail_pair_win/`（`report.json`、`windows-retail-manifest.json`、`frames/` 400 个 PNG）
  - Linux：`.tmp_v400_wp8_retail_pair_linux/`（`capture.json`、`linux-manifest.json`、`frames/` 200 PNG、`raw/` 200 RGBA）
  - 报告：`.tmp_v400_wp8_retail_pair_report/pixel-report.json`
  - 诊断（一次性脚本）：`.tmp_v400_wp8_retail_pair_lag.py`、`.tmp_v400_wp8_retail_pair_crop.py`、`.tmp_v400_wp8_retail_pair_crops/`

---

## 结论（先给答案）

| # | 问题 | 结论 | 关键证据 |
| --- | --- | --- | --- |
| 1 | 零售原件能否连续抓 200 帧并给出可配对 manifest | **成立** | `--phase capture --consecutive-frames 200 --no-score-dat`，退出码 **0**，`elapsed 16.974 s`；连续帧 **58…257**（200 帧，Δ 全为 +1）；200/200 帧由「BitBlt 前后两次 `gameplayFrameCounter` 相同」证明同帧；`closeExitCode=0`；`thirdPartyUnchanged=true`；manifest `source="windows-retail"`，SHA256 `a0a24c91…64f2b` |
| 2 | Linux harness 能否按**同一批** logicFrame 精确取 200 帧 | **成立** | `--targets-file` 传 58…257，退出码 **0**，`wall 9.44 s`，`framesCaptured=200`，引擎退出码 0；私有 `--data-dir`、私有 `xvfb`；`replay_start` 的 `gameplay_frame_counter=0`（回放帧号从 0 起）；manifest `source="linux-harness"`，SHA256 `fde5b9eb…5a6e0` |
| 3 | 公共状态投影是否逐帧完全相等 | **成立（200/200）** | `--state-pairing common-projection` 退出码 **0**；`state_consistency="asserted-on-projection"`；`score, lives, bombs, deaths, power, graze, stage` 每个字段 `equal=200 / total=200` |
| 4 | 像素统计是否可交付 | **成立** | playfield 平均绝对差均值 **0.4989**（中位 **0.1089**）、差异像素比例均值 **0.2638**、最大通道差 **255**；HUD 对应 0.2711 / 0.1340 / 250；去除 fps 矩形后 whole-frame 0.3884 / 0.1168 / 255；最差 10 帧与分布见第五节 |

**一句话**：这是第一次把 WP8 的"左右像素对"从**研究构建**换成**零售原件**，并在 200 个相同
`logicFrame` 上把双方状态投影压成 7 个字段做**逐帧精确相等**判定（200/200 通过）。
但它**不是 G0.4 通过**：只覆盖 1 个回放 / 1 个 stage，且这 200 帧里 7 个投影字段**全程为常量**
（见 4.3），所以它证明的是"同帧对齐 + 无分歧"，不是"有动态变化时仍逐帧一致"。

---

## 一、这一步补的是哪个缺口

G0.4 原先的证据链是「Windows 研究构建（带 frame fence）↔ Linux harness」：两侧跑的是同一份
`th08-rl` 代码，只是渲染后端/平台不同。它无法回答"重实现与**零售原件**在同一回放同一帧上是否
一致"。

本卡把左腿换成零售件后，必须同时解决三件事，缺一不可：

1. **帧号可对齐**：零售件的 `gameplayFrameCounter`（`g_GameManager+0x3DDC0`）与 Linux harness
   的 `f3` 是同一个全局；Linux 侧实测回放起点 `f3=0`，零售侧首个可抓帧为 58，且 58…257 上
   **同号帧的画面比对明显优于 ±1 帧**（同帧 playfield MAD 0.08–0.17，±1 帧 1.5–2.8，见 5.3），
   所以"同一 logicFrame"是同一逻辑帧，而不是靠墙钟猜的；
2. **两侧可比较**：零售侧没有 frame fence，同帧性只能靠 BitBlt 前后的计数器夹逼；两侧的状态
   摘要算法不同（零售是内存投影，Linux 是 25 字段 FNV-1a），所以不能沿用 `strict`；
3. **状态一致性可断言**：新增 `common-projection`，只对双方都真有的 7 个字段做**位级**相等判定，
   不相等即 fail-closed。

---

## 二、零售侧：200 帧 + `windows-retail` manifest

### 2.1 命令与退出码

```
python tools/rl/v400_retail_win_capture.py --phase capture --no-score-dat \
  --consecutive-frames 200 --capture-seconds 120 --budget-seconds 175 --settle-seconds 2.0 \
  --out .tmp_v400_wp8_retail_pair_win --run-dir .tmp_v400_wp8_retail_pair_win_run
# 退出码 0，elapsed 16.974 s（预算 175 s，单次运行 ≤180 s）
```

`--no-score-dat` 是 EXP-P0-060 得出的硬性前置：run-dir 里**绝不能**预置重建构建写出的
`score.dat`（会 `0xC0000005`）。本次 run-dir 是全新装配的，零售件启动后**自己**写出了
`score.dat`（16,034 B，SHA256 `403b8c5f18…70501`），与模板 `play_fence/score.dat`
（17,773 B / `60e49abc…`）不同——这正是"没有被预置"的物证。

装配前先确认无残留 `th08.exe`：`Get-Process th08` 为空；运行结束后再查仍为空。

正式 200 帧之前先用同一工具跑了一次 5 帧 smoke（`.tmp_v400_wp8_retail_pair_smoke`，
`--consecutive-frames 5`，退出码 0，8.568 s，帧 60…64），确认 manifest 写出与 RGBA 帧正确后
才跑 200 帧；smoke 与正式 run 使用**不同的** run-dir，两者都用 `--no-score-dat`。

### 2.2 帧清单与同帧证明

- 连续帧号 **58, 59, …, 257**（200 帧，`validation.frameDelta` 全为 `1`，`consecutive=true`）；
- `sameFrameProvenAll=true`：每帧的 `frameCounterBefore == frameCounterAfter == logicFrame`；
- 抓取循环：`attempts=3398`、`accepted=3210`、`duplicateFrames=3010`、`skippedAttempts=188`；
- 首帧 `t=6.6025 s`、末帧 `t=9.9053 s`（自 `Popen` 起）⇒ 200 帧 3.303 s ≈ **60.55 fps**；
- `distinctPngSha256=169/200`：stage 1 开场有静态/重复画面（EXP-P0-060 已见 60/61 同图），
  但**帧号仍是严格 +1**；
- `closeExitCode=0`（零售件干净退出）、`thirdPartyUnchanged=true`（`resources/th08.exe`、
  `th08.dat`、目标 `.rpy` 前后 SHA256 一致）。

### 2.3 manifest 结构（每帧含 `logicFrame` / `image` / `state_evidence`）

`.tmp_v400_wp8_retail_pair_win/windows-retail-manifest.json`
（274,422 B，SHA256 `a0a24c91bb1d30272fecae796eb16fe4d10fbe23f63ed234ac2433465a564f2b`）：

```
schema          v400-wp8-retail-frame-manifest/1
source          windows-retail
replay_sha256   e621086d814fba4838a2c84e4081162524c3c7e84d1bc4ca3f6621f9ca52d96f
replay_stage    0
retail_image    size=840704 sha256=330fbdbf…d924 verified=true
capture.sameFrameProof   BitBlt 前后 gameplayFrameCounter（g_GameManager+0x3DDC0）必须相同
capture.frameFence       absent（零售件没有 frame fence，这是唯一可用的同帧证明）
capture.pngMode          RGBA 640×480（GDI 无 alpha，补不透明 255，已在 manifest 里写明）
frames[]        logicFrame / image（相对 PNG）/ image_sha256 / state_sha256 /
                state_evidence{ fields{7 字段}、field_bits{32 位精确值}、
                                frameCounterBefore/After、sameFrameProven、
                                memory{gameplayFrame, stage, flags, replayMode,
                                       currentSpell, playerState, playerX/Y,
                                       stageFrames, runFrames} }
```

`state_sha256` 是零售侧自己的摘要（canonical JSON：`algorithm + logicFrame + 投影 bits`），
**不与 Linux 的 `state_sha256` 比较**；两侧 SHA 只在 manifest 头部做回放同一性校验。

---

## 三、Linux 侧：同一批 logicFrame 的 200 帧

### 3.1 私有 `--data-dir`（不得用共享 `/root/v400-data/game`）

```
/root/v400-data/runtime3/wp8-retail-pair0/game/
  score.dat   <- cp /root/v400-data/game/score.dat.orig   (60e49abc…，pristine)
  th08.cfg    <- cp /root/v400-data/game/th08.cfg.orig    (48bf7621…，pristine)
  th08.dat    -> symlink /root/v400-data/game/th08.dat
  thbgm.dat   -> symlink /root/v400-data/game/thbgm.dat
```

运行前后对共享目录做 SHA256 对照：`score.dat` `60e49abc…`、`th08.cfg` `48bf7621…`
**前后完全一致**（共享模板没被写）；而私有目录里的 `score.dat` 被引擎改写为
`adfa4022e2ce0f58f53f4dd967c27e87bc054817a587e155d67c3c4dc0884805`
——写入确实落在私有副本里。

### 3.2 命令、退出码、帧数与 manifest

```
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && python3 tools/rl/v400_linux_frame_capture.py capture \
  --out .tmp_v400_wp8_retail_pair_linux \
  --targets-file .tmp_v400_wp8_retail_pair_win/targets.json \
  --replay /root/v400-data/pair3/Th8_ud2970.rpy --stage 0 \
  --data-dir /root/v400-data/runtime3/wp8-retail-pair0/game"
# 退出码 0，墙钟 9.2 s；capture.json: status=captured, framesCaptured=200, wallSeconds=9.44

python tools/rl/v400_linux_frame_capture.py convert \
  --capture .tmp_v400_wp8_retail_pair_linux/capture.json
# 退出码 0，5.8 s；200 张 RGBA PNG + linux-manifest.json
```

- `targets.json` 的 200 个帧号来自**零售侧 manifest**（两遍法：先零售定帧号，再让 Linux 按
  `--targets` 精确 seek），不是人工挑的；
- 私有 `xvfb`：`display=:99`（`xvfb-run -a` 自建，未继承 WSLg 的 `:0`）；
- 共享内存块 `v400wp8pair8571`；`HASH` 与 shm header 的 frame/stateHash 逐帧校验通过（工具内建）；
- 引擎退出码 **0**，`stderr_tail` 为空；
- `replay_start`: `gameplay_frame_counter=0`、`replay_frame_counter=0`、`stage=0`、
  `harness_frame=92`（标题/载入泵了 92 帧后才进入回放，回放帧号从 0 起）；
- `linux-manifest.json`：200 帧（58…257），SHA256
  `fde5b9eb42a193f7d685a798decbb4d86ad44a2111f91b110e8761fede45a6e0`；
- 二进制 `/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern`，
  SHA256 `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb`，
  git HEAD `101ce5eb9f442e9fbfd5e7e10e99495b4873d1ff`。

---

## 四、`common-projection`：投影一致性

### 4.1 实现（`tools/rl/v400_pixel_pair.py`）

新增第三种模式 `--state-pairing common-projection`；**`strict`（默认）与 `same-logic-frame`
的源码路径、源校验和结论措辞都未改动**（`strict` 仍只接受 `windows-fence`，仍要求两侧
`state_sha256` 相等）。

- 输入源：`common-projection` 只接受 `source="windows-retail"` 的 Windows manifest +
  `source="linux-harness"` 的 Linux manifest；两侧回放 SHA256 必须相同，`logicFrame` 集合必须
  完全相同且数量等于 `--frames`；
- 共同投影：`score, lives, bombs, deaths, power, graze, stage`
  - 零售侧取 `state_evidence.field_bits`（7 个字段全部是精确 32 位模式）；
  - Linux 侧取 `f4 score / f5 graze / f6 deaths / f7 lives / f8 bombs / f9 power / f1 stage`
    （`f6..f9` 本来就是 float bit pattern）；
  - 比较用**整型位模式**，不设容差；`-0.0` 与 `0.0` 会被判为不等；
- 全部相等：`state_projection.frames_equal=200/200`、`state_consistency="asserted-on-projection"`，
  报告列出字段名与每帧两侧数值；
- 任一不等：**fail-closed**，不写报告，退出码 **3**，stderr 指出首个不等帧与字段
  （单测 `test_common_projection_fails_closed_on_the_first_unequal_field` 覆盖）。

### 4.2 结果：200/200 完全相等

```
python tools/rl/v400_pixel_pair.py \
  --windows .tmp_v400_wp8_retail_pair_win/windows-retail-manifest.json \
  --linux   .tmp_v400_wp8_retail_pair_linux/linux-manifest.json \
  --frames 200 --state-pairing common-projection \
  --out .tmp_v400_wp8_retail_pair_report/pixel-report.json
# 退出码 0

MEASURED 200 frames (common-projection); G0.4 still requires three real replays and visual review
PROJECTION: score, lives, bombs, deaths, power, graze, stage equal on 200/200 frames (asserted-on-projection)
```

`pixel-report.json`（883,560 B，SHA256 `cfdb77c093ba18f9cc1af8fefb2117b9882b7d347756d3c316ba6ab4e78cd017`）
的 `state_projection`：

```
fields           [score, lives, bombs, deaths, power, graze, stage]
float_fields     [bombs, deaths, lives, power]
frames_equal     200
frames_total     200
field_equality   score 200/200, lives 200/200, bombs 200/200, deaths 200/200,
                 power 200/200, graze 200/200, stage 200/200
note             仅 7 字段投影的精确相等；RNG/子弹/敌机等不在投影内
```

### 4.3 这个结论的强度（必须如实说明）

这 200 帧里 **7 个投影字段全部是常量**：`score=0`、`lives=2.0`、`bombs=3.0`、`deaths=0.0`、
`power=0.0`、`graze=0`、`stage=0`（两侧逐帧相同）。原因是 stage 1 的开场窗口（帧 58…257）里
还没有可得分/可擦弹的内容。

因此本次通过证明的是：

- **对齐成立**：两侧回放 SHA256 相同，且同号帧的画面比对显著优于 ±1 帧（5.3 的 ±1 搜索），
  说明两个 `logicFrame` 指的是同一条回放的同一帧；
- **无分歧**：在 200 个同帧上，凡是双方都真有的 7 个字段没有任何一个不等；
- **不证明**：得分/残机/火力等字段**发生变化时**两侧是否仍逐帧一致（本窗口无变化），更不证明
  RNG、子弹、敌机、道具等投影外状态一致。

要补上"动态区间"的证据，应把窗口放到有得分/擦弹的区间（例如同一回放的 stage 3，或
`EXP-P0-056` 用过的窗口）——这属于下一步，不在本卡结论内。

### 4.4 负向对照：改掉一帧的 `score`，必须 exit 3

为了证明"不等就 fail-closed"不只是单测里的路径，用**真实 manifest 的副本**做了负向对照
（`linux-manifest-negctl.json`，只在副本里把 `logicFrame=150` 的 `f4`（score）从 `0` 改成
`999`，并在该帧 `state_evidence.note` 里写明是负向对照；原始 manifest 未改动）：

```
python tools/rl/v400_pixel_pair.py \
  --windows .tmp_v400_wp8_retail_pair_win/windows-retail-manifest.json \
  --linux   .tmp_v400_wp8_retail_pair_linux/linux-manifest-negctl.json \
  --frames 200 --state-pairing common-projection \
  --out .tmp_v400_wp8_retail_pair_failclosed/pixel-report.json
# 退出码 3

BLOCKED: frame 150: common projection mismatch on ['score'] (1/200 frames differ); score: retail=0 linux=999
# 报告未写出（Test-Path = False）
```

负向对照**不是证据**，只证明该模式在投影不等时确实拒绝并指出首个不等帧与字段。

---

## 五、像素统计（同帧、RGB、640×480）

### 5.1 总表（`pixel-report.json.aggregate`）

区域定义沿用工具：playfield = `x∈[32,416)`；HUD = 其余列；FPS 不遮罩；RGB 三通道。
下表 `MAD` = 每帧平均绝对差，`ratio` = 差异像素比例。

| 区域 | MAD 均值 | MAD 最小 | MAD 最大 | ratio 均值 | ratio 最大 | 最大通道差 |
| --- | --- | --- | --- | --- | --- | --- |
| **playfield** | **0.4989** | 0.0754 | 9.8666 | **0.2638** | 0.99984 | **255** |
| **HUD** | **0.2711** | 0.0297 | 9.3599 | **0.1340** | 0.99714 | **250** |
| whole frame（去 fps 矩形） | 0.3884 | 0.0646 | 9.5967 | 0.1168 | 0.99860 | 255 |
| HUD（去 fps 矩形） | 0.2211 | 0.0297 | 9.1880 | 0.0872 | 0.99687 | 217 |

FPS 矩形（工具内已记录）：`y∈[463,480)`、`x∈[512,581)`；它只影响"去 fps"辅助统计，
上表 playfield/HUD 的数字**没有**遮罩 fps。

playfield MAD 分位：p50 **0.1090**、p75 **0.1635**、p90 1.7473、p95 1.8595、p99 6.9654；
分布是双峰的：**160 帧 ≤0.2**，0 帧落在 0.2–1.0，**37 帧落在 1–2**，**3 帧 >2**。

200 帧里**没有一对 PNG 逐字节相同**（`windows_png_sha256 == linux_png_sha256` 为 0 对）；
零售侧 169 个不同 PNG，Linux 侧 200 个不同 PNG。

### 5.2 最差 10 帧（按 playfield MAD 排序）

| logicFrame | playfield MAD | HUD MAD | playfield ratio | HUD ratio | 最大通道差 |
| --- | --- | --- | --- | --- | --- |
| 58 | 9.8666 | 9.3599 | 0.9998 | 0.9971 | 237 |
| 59 | 6.9654 | 6.6677 | 0.9998 | 0.9960 | 236 |
| 60 | 2.0755 | 0.4820 | 0.3204 | 0.1118 | 236 |
| 76 | 1.9770 | 0.3027 | 0.3505 | 0.1444 | 236 |
| 69 | 1.9220 | 0.3069 | 0.3300 | 0.1441 | 235 |
| 66 | 1.9199 | 0.4428 | 0.3226 | 0.1392 | 236 |
| 232 | 1.8836 | 0.3201 | 0.3923 | 0.1824 | 227 |
| 135 | 1.8719 | 0.0735 | 0.4471 | 0.1209 | 227 |
| 75 | 1.8630 | 0.3227 | 0.3464 | 0.1391 | 236 |
| 91 | 1.8595 | 0.3248 | 0.3828 | 0.1408 | 164 |

### 5.3 差异是什么（用诊断脚本量化，不是猜）

1. **帧 58/59 是"开场淡入"错位**：整帧亮度比（零售/Linux，playfield 均值）
   帧 58 = **11.33 / 20.92 = 0.5414**，帧 60 = **1.0086**，帧 200 = 1.0073，帧 257 = 0.9987。
   即零售件在 logicFrame 58 仍处于 stage 开场淡入（全屏压暗 54%），Linux 构建在 f3=58 时淡入
   已结束；到帧 60 两侧亮度已一致。这是**呈现相位差**（淡入时钟比逻辑帧晚约 2 帧），不是状态差。
2. **HUD fps 读数差异（已知且预期）**：零售 HUD 显示 `60.31fps`，Linux harness 固定 `60.00fps`；
   同一位置的两位数 fps 大字在帧 58 分别是 `60`（零售，已饱和）与 `58`（Linux 预热），
   这也是帧 58/59 HUD MAD 高达 9.36/6.67 的主要原因之一。
3. **右侧装饰层相位差（1–2 区间共 37 帧，多数 ~1.8–1.9 MAD；如 66/69/75/76/91/135/200/232）**：
   放大差异热图后，强差异像素集中在 playfield 右缘 `x≈400–440`（含 HUD 左缘）的窄竖条
   （周期性出现的装饰/卷轴层）与雾边缘的稀疏像素；对这些帧做 ±1 帧搜索，
   `MAD(retail N, linux N−1) ≈ 0.08–0.17`，而 `MAD(retail N, linux N) ≈ 1.6–1.9`，说明**该装饰层
   的动画相位差 1 帧**；当该层处于暗淡相位时（如帧 100/250/253–257），同帧 MAD 回落到
   0.08–0.12。这也是 160 帧 ≤0.2 与 37 帧 1–2 双峰分布的原因。
4. 帧 60 之后**没有系统性整体偏移**：160 帧的同帧 MAD ≤0.2（若整帧晚 1–2 帧，这个量级不可能
   出现）；对 100/250/255 等帧做 ±1…±3 搜索，最小值都落在**同一 logicFrame**。帧 200 等少数
   帧的 ±1 最小值落在 N−1，仍属于第 3 条的装饰层相位差，且其同帧 MAD（1.6–1.9）远低于整帧
   位移的量级（帧 58/59 的 6.97/9.87）。

### 5.4 单人目视（只做了有限目视，不是完整人工评审）

- 目视了帧 58 的左右整图（零售 vs Linux）与帧 200 的差异热图
  `.tmp_v400_wp8_retail_pair_crops/diff-200.png`；
- 目视了帧 58/60/200/257 的右下角 4× 放大裁剪中的 4 张（共生成 8 张，
  `.tmp_v400_wp8_retail_pair_crops/`）；
- 结论：同帧画面是同一场景（Score 等 HUD 文本一致、自机位置一致、背景构图一致），差异集中在
  上述装饰层与 fps 字型；**未**对 200 帧逐帧做人工评审。

---

## 六、限制（全部是真实约束，不是免责声明）

1. **零售件没有 frame fence**：同帧性只能由「BitBlt 前读一次、BitBlt 后再读一次
   `gameplayFrameCounter`，两者相同」证明；它保证"抓到的像素属于这个计数器值"，但**不能**保证
   呈现缓冲与逻辑帧严格同拍（GDI 双缓冲/呈现延迟不在控制内）。这是零售侧与 `windows-fence`
   构建最本质的差别。
2. **7 字段投影在本次窗口内是常量**（见 4.3）：200/200 相等是"无分歧"证据，不是"动态区间也
   一致"的证据。
3. **投影不是全状态**：RNG、子弹、敌机、道具、BGM/SE 时钟都不在投影内；`state_sha256` 两侧
   算法不同，**不能**互相比较，只作溯源保留。
4. **HUD fps 差异**：零售 `60.31fps` vs Linux `60.00fps`（HUD 值差异，非渲染差异），帧 58/59
   还叠加了两位数 fps 大字从 58→60 的预热差异。
5. **帧 58/59 的开场淡入相位差**（约 2 帧）与**右侧装饰层 1 帧相位差**是"呈现"差异；本卡只
   定位、量化，不解释其实现原因（不在本卡范围）。
6. **单人目视**：仅 4 帧整图 + 8 张裁剪 + 1 张热图，未做完整人工评审；G0.4 要求的人工视觉
   评审仍未完成。
7. **只覆盖 1 个回放 / 1 个 stage / 200 帧**：G0.4 要求 3 个真实回放 × 200 连续帧；本卡是
   第 1 条腿（且是 stage 0 开场，不是子弹密集区）。
8. **零售 run-dir 必须 `--no-score-dat`**：预置重建构建的 `score.dat` 会让零售件 `0xC0000005`
   崩溃（EXP-P0-060 已定位）；本次 run-dir 的 `score.dat` 是零售件自建（16,034 B / `403b8c5f…`）。
9. **Linux 侧必须私有 `--data-dir`**：共享 `/root/v400-data/game` 会被引擎写脏；本次用
   `runtime3/wp8-retail-pair0/game`，共享目录前后 SHA256 未变、私有 `score.dat` 被改写
   （`adfa4022…`），写入去向已闭环。
10. **单次运行预算**：Windows 单次 ≤180 s；本次 16.974 s（200 帧一次通过，未使用 60 帧降级）。

---

## 七、变更文件与验收命令

变更（均在允许范围内）：

| 文件 | 变更 |
| --- | --- |
| `tools/rl/v400_retail_win_capture.py` | 新增 `PROJECTION_FIELDS`/`retail_state_projection`/`retail_state_digest`/`to_rgba`/`build_retail_manifest`/`write_retail_manifest`；`capture` phase 写出 RGBA 帧与 `windows-retail-manifest.json`；短抓取也落盘（诚实降级） |
| `tests/test_v400_retail_win_capture.py` | 新增 10 个用例（投影精确性/缺字段拒绝、manifest 结构、非连续帧拒绝、未证明同帧拒绝、空集拒绝、RGBA 合成、digest 区分帧与 float 位型） |
| `tools/rl/v400_pixel_pair.py` | 新增 `common-projection` 模式、`project_retail`/`project_linux`/`projection_display`、按模式限定 Windows 源、报告 `state_projection`；`strict` 默认路径不变 |
| `tests/test_v400_pixel_pair.py` | 新增 9 个用例（投影相等/首个不等字段 fail-closed/float 位级比较/缺字段/replay 与帧集仍强制/strict 仍拒绝 retail/CLI 写报告与不写报告） |
| `Document/Update/Alpha/v4.0.0/experiments/EXP-P0-062-WP8-retail-vs-linux-paired.md` | 本卡 |

验收命令与结果：

| 命令 | 退出码 | 结果 |
| --- | --- | --- |
| `python -m compileall -q tools/rl/v400_pixel_pair.py tools/rl/v400_retail_win_capture.py` | 0 | 语法通过 |
| `python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests/test_v400_pixel_pair.py tests/test_v400_retail_win_capture.py` | 0 | **45 passed**（0.46 s） |
| 零售 capture（2.1 节命令） | 0 | 200 帧，16.974 s |
| Linux capture（3.2 节命令） | 0 | 200 帧，9.44 s |
| Linux convert | 0 | 200 PNG + manifest，5.8 s |
| `v400_pixel_pair.py --state-pairing common-projection`（4.2 节命令） | 0 | 200/200 投影相等 |

未提交、未删除任何文件；`third_party/th08` 与零售 exe 未被写（零售报告
`thirdPartyUnchanged=true`）；运行前后 `Get-Process th08` 均为空。

---

## 八、G0.4 状态

- 本轮把 WP8 的对照从「研究构建替代零售」升级为**真零售 vs Linux**，并首次给出
  `state_consistency="asserted-on-projection"`（7 字段 × 200 帧，200/200）。
- 距 G0.4 仍缺：**另外 2 个真实回放的 200 帧**、**动态区间**（得分/擦弹/残机发生变化）的投影
  一致性、**完整人工视觉评审**。
- `pixel-report.json` 的 `status` 仍是 `measured-not-G0-pass`，这是刻意的：本卡不把"一个
  stage 的 200 帧"当成 G0.4 通过。
