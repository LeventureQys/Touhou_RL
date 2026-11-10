# EXP-P0-063（v4.0.0 WP8）：零售原件 ↔ Linux harness 三个动态回放窗口的 200 帧配对（G0.4 可判定性）

- 日期：2026-11-09
- Provider / 模型：DeepSeek / DeepSeek v4.1 flash
- 环境：Windows（Python 3.12，numpy/Pillow）零售原件 + WSL2 Linux（`xvfb-run -a` 私有 Xvfb `:99`，软件 OpenGL）
- 零售侧：`third_party/th08/resources/th08.exe`（零售原件，840,704 B，SHA256 `330fbdbf…d924`，`verified=true`）
- Linux 侧：`/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern`，SHA256 `591c521d…d3eb`，git HEAD `101ce5eb…1ff`
- 产物：`.tmp_v400_wp8_retail_pair3/{r1,r2,r3}/`（零售 `win/`、Linux `linux/`、`targets.json`、`pixel-report.json`）、
  目视复核图 `.tmp_v400_wp8_retail_pair3_review/`

---

## 结论（先给答案）

| # | 问题 | 结论 | 关键证据 |
| --- | --- | --- | --- |
| 1 | 三份回放是否都在**动态窗口**（`activeBullets>0` 且投影字段变化）取到 200 连续帧 | **成立** | r1 `score` 198 个不同值/197 次变化、`graze` 4/3；r2 `score` 200/199、`graze` 3/2；r3 `score` 200/199、`graze` 9/8。窗口弹幕数见一.3（fence 遥测 37–206） |
| 2 | 零售侧能否从指定 logicFrame 起抓 | **成立** | 新增 `--start-logic-frame`；三次 seek 到达 2352/1801/2011（probes 15349/11445/13081），首帧即请求帧，`startRequestHonoured=true` |
| 3 | 投影 200/200 逐帧相等 | **成立 ×3** | `common-projection` 三次均 exit **0**，`score, lives, bombs, deaths, power, graze, stage` 每字段 200/200（见三） |
| 4 | 像素统计是否可交付 | **成立** | 逐份与合计见四；同帧 playfield MAD 均值 r1 4.78 / r2 2.02 / r3 5.03，合计 3.94 |
| 5 | 差异是什么性质 | **1 帧呈现相位差，不是内容缺失/混合错误** | 600 帧中 210 帧（r1 77 / r2 52 / r3 81）零售第 N 帧像素 ≈ Linux 第 **N−1** 帧（playfield MAD 0.137–0.154）；按「同帧或前 1 帧取优」对齐后 **600/600 帧 MAD ≤ 0.1821**（均值 **0.1434**），对齐差异图近乎全黑 |

**一句话**：三份回放（Normal / Lunatic / Hard，含 `Th8_ud0059`）各 200 个**同号且动态变化**的 logicFrame 上，
7 字段投影 **600/600 位级相等**；同帧像素差偏大（最高 17.28 MAD）**完全由零售 GDI 抓取路径的 1 帧呈现相位差解释**，
按相位对齐后两侧渲染几乎逐像素相同（MAD 0.14）。这补上了 EXP-P0-062「常量投影」和「单回放」两个缺口，
但**仍不等于 G0.4 通过**：同帧像素判据在当前无 fence 的零售抓取方式下不可稳定通过，且投影仍只有 7 个字段。

---

## 一、三份回放与窗口选择

### 1.1 回放

| 编号 | 回放 | SHA256 | 难度/机体 | stage | 备注 |
| --- | --- | --- | --- | --- | --- |
| r1 | `th08_replay/Th8_ud2970.rpy` | `e621086d…2d96f` | Normal / Marisa | 3（Stage 4A） | 与 EXP-062 同一回放、不同 stage |
| r2 | `th08_replay/Th8_ud0059.rpy` | `62b02a17…9a5e9` | Lunatic / Youmu | 5（Stage 5） | **与 EXP-058 相同回放** |
| r3 | `th08_replay/Th8_ud2563.rpy` | `4457c855…fad59` | Hard / Reimu | 4（Stage 4B） | 第三条腿，4B 路线 |

两侧 manifest 的 `replay_sha256` 逐份相同（见 5.3 的完整性检查）。

### 1.2 窗口如何选出（不是人工挑帧）

窗口规则沿用 `.tmp_v400_wp8_pair3_suite/suite-summary.json`（EXP-058，研究 fence 构建在同一 replay/stage 上
按「首个 `activeBullets ≥ 40` 的 logicFrame 起 200 帧」选出的窗口）：

| 编号 | 窗口 logicFrame | 弹幕（fence 遥测：min/mean/max） | 触发帧 |
| --- | --- | --- | --- |
| r1 | 2352…2551 | 39 / 59.4 / 74，200/200 帧有弹幕 | 2351 |
| r2 | 1801…2000 | 44 / 139.4 / **206**，200/200 帧有弹幕 | 1800 |
| r3 | 2011…2210 | 37 / 47.0 / 74，200/200 帧有弹幕 | 2010 |

**诚实说明**：零售件不发布 bullet 计数，Linux 本轮的 `STATE` 25 字段也不含 bullet 计数；`activeBullets`
来自研究 fence 构建（与 Linux 同一套逻辑）在**相同 replay/stage/logicFrame** 上的遥测，属「回放已知弹幕段」证据。
本轮两侧在 2352/1801/2011 的投影值与该套件的 Linux 侧逐帧值一致，说明窗口落点相同；弹幕在画面中也可直接看到（第五节点检）。

### 1.3 零售侧 seek（新增能力）

`tools/rl/v400_retail_win_capture.py` 新增 `--start-logic-frame N`：回放开始后只读 `gameplayFrameCounter` 等待到 N，
再开始连续抓取；报告新增 `seek{requestedStart,probes,firstProbeFrame,lastProbeFrame,reached,seconds}` 与
`validation{startLogicFrameRequested,firstCapturedFrame,startRequestHonoured}`。首帧严格等于请求帧（不是「>= 之后取到哪算哪」）：

| 编号 | 请求 | 首帧 | seek 探针 | seek 秒 | 抓取区间 | 连续 | sameFrameProvenAll |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r1 | 2352 | **2352** | 15349 | 38.18 | **2352…2551** | true（Δ≡1） | true（200/200） |
| r2 | 1801 | **1801** | 11445 | 29.03 | **1801…2000** | true | true |
| r3 | 2011 | **2011** | 13081 | 32.50 | **2011…2210** | true | true |

---

## 二、命令与退出码

零售侧（三份分别执行，单次 ≤180 s；运行前 `Get-Process th08` 为空，运行后仍为空）：

```
python tools/rl/v400_retail_win_capture.py --phase capture --no-score-dat \
  --consecutive-frames 200 --start-logic-frame <S> --capture-seconds 120 --budget-seconds 175 \
  --settle-seconds 2.0 --replay <rpy> --replay-stage <st> \
  --out .tmp_v400_wp8_retail_pair3/<r>/win --run-dir .tmp_v400_wp8_retail_pair3/<r>/win-run
```

| 运行 | 退出码 | 耗时 | closeExitCode | thirdPartyUnchanged | distinctPng/200 |
| --- | --- | --- | --- | --- | --- |
| r1 | **0** | 55.782 s | 0 | true | 151 |
| r2 | **0** | 50.841 s | 0 | true | 159 |
| r3 | **0** | 49.880 s | 0 | true | 150 |

Linux 侧（`targets.json` 由零售 manifest 的 `logicFrame` 列表生成；私有 `--data-dir` + 私有 Xvfb）：

```
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && python3 tools/rl/v400_linux_frame_capture.py capture \
  --out .tmp_v400_wp8_retail_pair3/<r>/linux --targets-file .tmp_v400_wp8_retail_pair3/<r>/targets.json \
  --replay /root/v400-data/pair3/<rpy> --stage <st> --data-dir /root/v400-data/runtime3/wp8-retail3-<r>/game"

python tools/rl/v400_linux_frame_capture.py convert --capture .tmp_v400_wp8_retail_pair3/<r>/linux/capture.json
```

| 运行 | capture 状态 | 引擎退出码 | 墙钟 | display / shm | convert |
| --- | --- | --- | --- | --- | --- |
| r1 | captured, 200 帧 | 0 | 28.951 s | `:99` / `v400wp8pair411` | exit 0 |
| r2 | captured, 200 帧 | 0 | 23.588 s | `:99` / `v400wp8pair766` | exit 0 |
| r3 | captured, 200 帧 | 0 | 25.843 s | `:99` / `v400wp8pair402` | exit 0 |

（r2 的 capture 一次是通过 PowerShell 管道截断读取的，shell 退出码被管道关闭掩盖；以 `capture.json`
`status=captured / framesCaptured=200 / process.exit_code=0` 为准，未重跑。）

比较（三次均 exit 0；`--frames 200 --state-pairing common-projection`）：

```
python tools/rl/v400_pixel_pair.py --windows .tmp_v400_wp8_retail_pair3/<r>/win/windows-retail-manifest.json \
  --linux .tmp_v400_wp8_retail_pair3/<r>/linux/linux-manifest.json --frames 200 \
  --state-pairing common-projection --out .tmp_v400_wp8_retail_pair3/<r>/pixel-report.json
```

测试：`python -m pytest -p no:cacheprovider --basetemp .tmp_v400_wp8_retail_pair3_pytest -q
tests/test_v400_retail_win_capture.py tests/test_v400_pixel_pair.py` → **45 passed**，exit 0
（既有用例；本轮新增的 `--start-logic-frame` 与 `field_variation` 由三次真实运行覆盖，**未新增单测**，见七）。

---

## 三、投影一致性（`common-projection`，200/200 × 3）

| 编号 | frames_equal | state_consistency | 字段相等 |
| --- | --- | --- | --- |
| r1 | **200/200** | asserted-on-projection | 7 字段各 200/200 |
| r2 | **200/200** | asserted-on-projection | 7 字段各 200/200 |
| r3 | **200/200** | asserted-on-projection | 7 字段各 200/200 |

`pixel-report.json.state_projection.field_variation`（本轮新增的汇总字段）给出**每个字段的 distinct 值与变化次数**
（`distinct_patterns` = 32 位位型个数；`changes` = 相邻帧值改变次数；first/last 为窗口首末值）：

| 字段 | r1 distinct / changes | r2 distinct / changes | r3 distinct / changes |
| --- | --- | --- | --- |
| **score** | **198 / 197**（76,034,765→76,941,378） | **200 / 199**（255,147,077→255,413,295） | **200 / 199**（92,382,654→92,532,254） |
| lives | 1 / 0（6.0） | 1 / 0（7.0） | 1 / 0（6.0） |
| bombs | 1 / 0（5.0） | 1 / 0（7.0） | 1 / 0（3.0） |
| deaths | 1 / 0（0.0） | 1 / 0（0.0） | 1 / 0（0.0） |
| power | 1 / 0（128.0） | 1 / 0（128.0） | 1 / 0（128.0） |
| **graze** | **4 / 3**（9,404→9,413） | **3 / 2**（21,290→21,296） | **9 / 8**（12,794→12,836） |
| stage | 1 / 0（3） | 1 / 0（5） | 1 / 0（4） |

即每个窗口都至少有 `score` 和 `graze` 两项在变化（不是「常量即通过」）；`score` 在 r2/r3 每帧都变。
`lives/bombs/deaths/power` 在本窗口内确实不变，如实列出，不当作动态证据。

---

## 四、像素统计（同帧、RGB、640×480，`pixel-report.json.aggregate`）

区域定义沿用工具：playfield `x∈[32,416)`；HUD = 其余列；FPS 不遮罩（另给「去 fps 矩形」辅助统计）。
`MAD` = 每帧平均绝对差；`ratio` = 差异像素比例；`maxch` = 最大通道差。

| 编号 / 区域 | MAD 均值 | MAD 最小 | MAD 最大 | ratio 均值 | ratio 最大 | maxch |
| --- | --- | --- | --- | --- | --- | --- |
| r1 playfield | 4.781574 | 0.124570 | 17.057849 | 0.412176 | 0.709787 | 255 |
| r1 HUD | 0.495856 | 0.267543 | 1.454560 | 0.176325 | 0.183333 | 251 |
| r1 whole（去 fps） | 3.063475 | 0.184135 | 10.655802 | 0.234928 | 0.441497 | 255 |
| r1 HUD（去 fps） | 0.461489 | 0.269056 | 1.413942 | 0.132943 | 0.140340 | 251 |
| r2 playfield | 2.018878 | 0.145132 | 12.366206 | 0.381839 | 0.625640 | 255 |
| r2 HUD | 0.504447 | 0.262207 | 1.551655 | 0.174926 | 0.184351 | 251 |
| r2 whole（去 fps） | 1.390974 | 0.192679 | 7.922055 | 0.179563 | 0.342050 | 255 |
| r2 HUD（去 fps） | 0.440040 | 0.264405 | 1.511973 | 0.130646 | 0.140636 | 251 |
| r3 playfield | 5.032912 | 0.122815 | 17.280380 | 0.416922 | 0.762375 | 255 |
| r3 HUD | 0.557251 | 0.269859 | 1.336155 | 0.177247 | 0.185433 | 251 |
| r3 whole（去 fps） | 3.234807 | 0.183129 | 10.744398 | 0.240466 | 0.474726 | 255 |
| r3 HUD（去 fps） | 0.511656 | 0.271926 | 1.283944 | 0.133815 | 0.140647 | 251 |
| **合计 playfield** | **3.944455** | **0.122815** | **17.280380** | **0.403646** | **0.762375** | **255** |
| **合计 HUD** | **0.519185** | **0.262207** | **1.551655** | **0.176166** | **0.185433** | **251** |
| **合计 whole（去 fps）** | **2.563085** | **0.183129** | **10.744398** | **0.218319** | **0.474726** | **255** |
| **合计 HUD（去 fps）** | **0.471062** | **0.264405** | **1.511973** | **0.132468** | **0.140647** | **251** |

**注意**：这些「同帧」统计被第五节的 1 帧呈现相位差显著放大，不能单独当作渲染保真度结论。

---

## 五、最差 10 帧与差异性质（本轮最重要的发现）

### 5.1 最差 10 帧（按同帧 playfield MAD 排序）

| # | 回放 | logicFrame | 同帧 MAD | **N−1 对齐 MAD** | HUD MAD | playfield ratio | maxch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | r3 | 2060 | 17.2804 | **0.1435** | 0.7717 | 0.7280 | 255 |
| 2 | r3 | 2091 | 17.1028 | **0.1718** | 1.3362 | 0.7239 | 255 |
| 3 | r1 | 2391 | 17.0578 | **0.1435** | 1.0051 | 0.6923 | 255 |
| 4 | r3 | 2093 | 15.6738 | **0.1578** | 1.1507 | 0.7075 | 255 |
| 5 | r1 | 2543 | 15.5243 | **0.1488** | 0.8682 | 0.7092 | 255 |
| 6 | r1 | 2542 | 15.2827 | **0.1499** | 0.8709 | 0.7084 | 255 |
| 7 | r3 | 2096 | 14.4475 | **0.1586** | 1.1300 | 0.6919 | 255 |
| 8 | r3 | 2071 | 14.4409 | **0.1808** | 1.0282 | 0.7624 | 255 |
| 9 | r3 | 2076 | 14.3565 | **0.1675** | 1.0118 | 0.7545 | 255 |
| 10 | r1 | 2526 | 14.3526 | **0.1336** | 0.8954 | 0.6862 | 255 |

**10/10 帧的 N−1 对齐 MAD 都只有 0.13–0.18**，比同帧小两个数量级。

### 5.2 全量相位分析（同帧 vs 前 1 帧）

对每一帧计算 `MAD(retail N, linux N)` 与 `MAD(retail N, linux N−1)`：

| 编号 | 同帧更优帧数 | N−1 更优帧数 | 同帧优帧均值 | N−1 优帧均值 | **两相取优均值 / 最大** |
| --- | --- | --- | --- | --- | --- |
| r1 | 123 | **77** | 0.1386 | 0.1371 | **0.1380 / 0.1615** |
| r2 | 148 | **52** | 0.1529 | 0.1535 | **0.1530 / 0.1665** |
| r3 | 119 | **81** | 0.1378 | 0.1414 | **0.1393 / 0.1821** |
| 合计 | 390 | **210** | — | — | **0.1434 / 0.1821（600/600 ≤0.3）** |

同帧 MAD 分布是双峰的：r1 ≤1 有 123 帧、>5 有 77 帧、1–5 有 0 帧；r2 148/51/1；r3 119/81/0。
**「>5」的帧恰好就是「N−1 更优」的帧**——说明这些帧零售抓到的像素属于**上一帧的呈现**，不是渲染内容不同。

### 5.3 1 帧呈现相位差的直接证据

1. **r1 帧 2391**：同帧 playfield MAD 17.06；retail 2391 vs linux **2390** 只有 **0.1435**。
   同帧并排图上，零售的法阵背景是灰蓝色环、Linux 是品红色星阵（看起来像混合/颜色完全不同）；
   但把零售 2391 与 Linux **2390** 并排后两张图几乎逐像素相同（含法阵颜色、Boss、弹幕、HUD 数字）。
   这说明法阵背景颜色在相邻帧间快速变化，差异是**相位**而非混合模式错误。
2. **HUD 数字**：r1 帧 2391 的 HUD score 区域 `MAD(retail 2391, linux 2390)=0.773`，而同帧是 5.717；
   同帧图上可见 Score/Point/Time 读数字面不同（如 `609/800` vs `613/800`、`2511` vs `2513`），
   而投影字段（内存 score/graze）逐帧位级相等——HUD 显示滞后 1 帧，状态没有滞后。
3. **对齐后差异图**：`.tmp_v400_wp8_retail_pair3_review/r1-2391-phase-aligned-diff-x8.png` 的 playfield 近乎全黑，
   只剩边缘一像素噪声；同帧差异图 `.tmp_v400_wp8_retail_pair3_review/r1-2391-diff-x8.png` 整个 playfield 全亮。
4. 该相位不是整段固定的：同一窗口内约 60–74% 帧同帧最佳、26–40% 帧 N−1 最佳，逐帧翻转，
   与「GDI BitBlt 与呈现缓冲竞争」的抓取方式一致（零售件无 fence，EXP-062 已列为限制 1）。

**差异性质判定**：不是字体缺失（HUD 字形/布局两侧一致，只是数字滞后 1 帧）、不是混合模式错误或内容缺失
（对齐后逐像素相同，Boss/弹幕/背景/法阵都在），而是**抓取路径的 1 帧呈现相位差**。

### 5.4 目视复核（实际读了哪些图）

- `.tmp_v400_wp8_retail_pair3_review/r1-2391-side.png`：同帧左右整图（差异看似巨大）；
- `.tmp_v400_wp8_retail_pair3_review/r1-2391-diff-x8.png`：同帧差异×8（playfield 全亮）；
- `.tmp_v400_wp8_retail_pair3_review/r1-2391-phase-aligned-side.png`：retail 2391 ↔ linux 2390 并排（几乎相同）；
- `.tmp_v400_wp8_retail_pair3_review/r1-2391-phase-aligned-diff-x8.png`：相位对齐差异（playfield 近黑）；
- `.tmp_v400_wp8_retail_pair3_review/r2-1990-side.png`、`r3-2060-side.png`：另两份的最差帧同帧并排；
- `.tmp_v400_wp8_retail_pair3_review/r1-2352-side.png`：窗口首帧（同帧 MAD 0.134，HUD 数字完全一致）；
- `.tmp_v400_wp8_retail_pair3_review/r1-2391-hud3x.png`、`r1-2391-score8x.png`、`r1-2391-hudrows8x.png`、
  `r1-2352-pointtime6x.png`：HUD 数字放大核对。

**目视结论**：三份的最差帧在**相位对齐后**都是同一场景（Boss、弹幕位置、自机、背景、法阵、HUD 布局一致），
未发现内容缺失或图层丢失；同帧看到的「巨大差异」全部由 1 帧相位与被其带动的 HUD 数字造成。
**未做** 600 帧逐帧人工评审，只看了上述 8 张图 + 3 个最差帧的量化对齐。

---

## 六、完整性检查

- 两侧 `replay_sha256` 逐份相同（r1 `e621086d…`、r2 `62b02a17…`、r3 `4457c855…`），帧集完全相同（200/200）；
- `windows-retail-manifest.json` SHA256：r1 `d5af917e…9f77b`、r2 `0f3c0480…5abc0`、r3 `b11499c4…6c042`；
  `linux-manifest.json` SHA256：r1 `a6d9fdde…453f`、r2 `01d26e62…157a`、r3 `6384dd30…be08`；
- 零售三次 `sameFrameProvenAll=true`（BitBlt 前后 `gameplayFrameCounter` 相同）、`closeExitCode=0`、
  `thirdPartyUnchanged=true`（零售 exe / `th08.dat` / 回放前后 SHA256 未变）；
- 零售 run-dir 全部 `--no-score-dat` + 全新目录；运行前后 `Get-Process th08` 均为空；
- Linux 三个私有 data-dir（`runtime3/wp8-retail3-{r1,r2,r3}/game`）：共享模板 `score.dat` `60e49abc…`、
  `th08.cfg` `48bf7621…` **前后未变**；三个私有 `score.dat` 均被引擎改写
  （`a0db9b33…` / `9d5b265d…` / `b5098b0d…`），写入去向闭环。

---

## 七、限制（真实约束）

1. **同帧像素不可稳定取得**：零售件无 fence，GDI BitBlt 与呈现缓冲竞争；本轮 210/600 帧抓到 N−1 呈现。
   这既解释了同帧 MAD 偏大，也意味着**「同帧像素相等」不能作为零售件的判据**，除非改变抓取方式（双抓/呈现确认/fence 注入）。
2. **投影仍只有 7 个字段**：RNG、子弹、敌机、道具、HUD 的 Point/Time 都不在投影内；`state_sha256` 两侧算法不同、不可互比。
   相位对齐后逐像素相同**提示**子弹/敌机也一致，但这不是投影内的断言。
3. **弹幕计数来自 fence 构建遥测**，不是零售/Linux 本轮直接测量（见 1.2 的诚实说明）。
4. **1 帧相位可能是抓取竞态而非引擎固定行为**：同一窗口内逐帧翻转；本卡只量化，不解释实现原因。
5. **未新增单测**：`--start-logic-frame` 与 `field_variation` 由三次真实运行覆盖，既有 45 个用例全过；
   时间预算内未补 seek 的假 session 单测。
6. **目视有限**：8 张图 + 3 个最差帧的量化对齐，不是完整人工视觉评审。
7. r2 的一次 capture 退出码被 PowerShell 管道截断掩盖（见二），以 `capture.json` 为准。

---

## 八、G0.4 自评：这份证据够不够判？

**够用来判「状态一致性 + 三动态回放」这一半，不够用来判「同帧像素等价」这一半。**

- **成立的部分**：3 个真实回放 × 200 连续 logicFrame × 7 字段投影 **600/600 位级相等**，
  且每个窗口都动态（`score` 变化 197–199 次，`graze` 变化 2–8 次），不再是 EXP-062 的常量窗口；
  这足以把「重实现与零售原件在同回放同帧上状态一致」从 1 例扩到 3 例，并覆盖 Normal/Lunatic/Hard。
- **更有力的部分**：相位对齐后 600/600 帧 playfield MAD ≤0.1821（均值 0.1434），说明两侧渲染几乎逐像素相同；
  这比投影更强（它包含子弹/敌机/背景/法阵），只是以「N 或 N−1」的形式给出。
- **不能判的部分**：若 G0.4 的像素门要求**同号帧逐像素相等/低于阈值**，当前零售抓取路径（无 fence）**做不到**，
  本轮同帧 playfield MAD 最高 17.28 就是证据；必须先解决呈现相位（否则任何同帧阈值都会被 1 帧滞后击穿）。
- **结论**：本卡把 G0.4 从「只对齐、不动态」推进到「三动态回放状态一致 + 相位对齐渲染近似一致」，
  但 G0.4 仍**未通过**；剩余关键项是「同帧呈现相位的可判定化（fence 或呈现确认）」与「完整人工视觉评审」。

---

## 九、变更文件

| 文件 | 变更 |
| --- | --- |
| `tools/rl/v400_retail_win_capture.py` | 新增 `--start-logic-frame`、seek 块与 `seek` 报告、`validation.startLogicFrameRequested/firstCapturedFrame/startRequestHonoured`；`ok` 判定加入 start 请求满足 |
| `tools/rl/v400_pixel_pair.py` | `state_projection` 新增 `field_variation`（每字段 distinct_patterns / distinct_values / changes / first / last），仅汇总，不改判据 |
| `Document/Update/Alpha/v4.0.0/experiments/EXP-P0-063-WP8-retail-dynamic-3replays.md` | 本卡 |

未提交、未删除；`third_party/th08`、零售 exe 未被写；无共享 data-dir 使用。
