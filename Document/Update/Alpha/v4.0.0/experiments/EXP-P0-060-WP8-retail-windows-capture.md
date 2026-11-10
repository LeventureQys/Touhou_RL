# EXP-P0-060（v4.0.0 WP8）：零售原件在 Windows 上运行 / 读内存 / GDI 截图的三项可行性 + 20 帧最小闭环

- 日期：2026-11-09
- 环境：Windows（`sys.platform=win32`），Python 3.12.10，numpy 2.5.4，Pillow 13.0.0，
  NVIDIA RTX 4070 SUPER（零售件自报 `T&L HAL`），交互式桌面会话（未锁屏，前台为 DSH 窗口）
- 目标镜像：`third_party/th08/resources/th08.exe`（**零售原件**，非研究构建）
  - 大小 840,704 B，SHA256 `330fbdbf58a710829d65277b4f312cfbb38d5448b3df523e79350b879213d924`
  - PE 事实（本卡实测）：`machine=0x014C`（i386）、`subsystem=2`（GUI）、`imageBase=0x00400000`、
    `DllCharacteristics=0x0000`（**无** `DYNAMIC_BASE`）、5 个节，`.text` va 0x2000 / vsize 0xB1B78
- 数据资产：`third_party/th08/build/play_fence/`（`th08.dat` 46,838,025 B、
  `thbgm.dat` 449,961,024 B、`th08.cfg`、`score.dat`）——只读取、`shutil.copy2` 复制到私有目录，
  每个 phase 都对 `resources/th08.exe` / `th08.dat` / 目标 `.rpy` 做前后 SHA256 比对，
  报告里 `thirdPartyUnchanged: true`（8/8 次运行）
- 工具：`tools/rl/v400_retail_win_capture.py`（新增）、
  地址表 `.tmp_v400_retail_win_addresses.json`（新增，SHA256 `a9b9f316…8b5da8`）
- 回放：`th08_replay/Th8_ud2970.rpy`（41,398 B，stage 0，Normal）

---

## 结论（先给答案）

| # | 问题 | 结论 | 关键证据 |
| --- | --- | --- | --- |
| 1 | 能否用零售原件跑起来并进入回放 | **成立（附带一个必须满足的数据条件）** | 进程启动 → 窗口 `class=BASE`、标题含 `ver 1.00d`、客户区 640×480 → 0.5–2.4 s 进标题 → `menu.enter_replay` 闭环导航成功 → `supervisorState=2`、`flags=0x0C`（含 `FLAG_REPLAY=0x8`）、`gameplayFrame=119` 递增；全程 3.69 s；关闭退出码 **0** |
| 2 | 能否从零售进程内存读到状态 | **成立** | 195 次采样，`gameplayFrameCounter` 150→390（195 个不同值，跨度 240，3.9946 s ⇒ **60.08 fps**），Δ∈{1,2}，无倒退；`score/lives/bombs/deaths/power/graze/stage` 全部合理且与 HUD 数字一致；独立偏移扫描确认 `g_GameManager+0x3DDC0` |
| 3 | 能否用 GDI 截图 | **成立** | `winapi.GdiCapture`（`GetDC`+`BitBlt`）在零售进程上直接出图：标题画面 640×480 / 171,762 色 / 742,333 B；回放画面 640×480 / 22,999 色 / 181,745 B，HUD 文字（Score/HiScore/残机/残符/fps）清晰可读 |
| 4 | 最小闭环（20 个连续 logicFrame × 1 PNG + 内存值） | **成立** | 连续帧 **59…78**（全部 +1），20 张 PNG + 20 张 HUD 裁剪，每帧都由「BitBlt 前后两次 `gameplayFrameCounter` 相同」证明同帧；19/20 张 PNG 逐字节不同 |

**但必须记录的头号事实**：零售原件在本机的**首 3 次运行全部崩溃**（`0xC0000005`，模块 `th08.exe`，
偏移恒为 `RVA 0x000A5CFF`），根因是**用重建构建写出的 `score.dat` 喂给零售件**；换成让零售件自己生成
`score.dat` 后 4/4 次运行全部干净退出（退出码 0）。详见第一节。

---

## 一、关键发现：零售件不是"跑不起来"，而是被"外来 score.dat"打死

### 1.1 首次运行：进标题后 3 秒死

第一次 `--phase launch` 用「`play_fence` 里的 `score.dat`（17,773 B）」装配数据目录：

- 进程启动，0.409–2.356 s 出窗口，标题画面 PNG 正常（742 KB、171,762 色，见
  `.tmp_v400_retail_win_launch/01-title.png`）；
- 之后 `menu.ensure_title` 拿不到窗口（`hwnd=0`），所有内存读开始返回 `OSError`；
- `session.close()` 观测到 `closeExitCode = 3221225477` = `0xC0000005`（访问违规）；
- Windows 事件日志（`postMortem` 自动采集）：

```
出错应用程序名称： th08.exe，异常代码： 0xc0000005
错误偏移： 0x000a5cff
Faulting 应用程序路径： D:\WorkShop\Touhou_RL\.tmp_v400_retail_win_rundir\th08.exe
```

同一秒 `rundir/backup/` 被零售件写入了 `score_1.261109.bak`（17,773 B）——即零售件**读过**那个
外来 score.dat，并从中取出过版本串。

### 1.2 对照实验：完全不去观测也照样死（3/3）

为排除"是我们的读内存/截图把游戏弄死"，新增 `--phase idle` 三个手臂：

| 手臂 | 观测行为 | 结果 |
| --- | --- | --- |
| A（`.tmp_v400_retail_win_idle_none2`） | **只** `EnumWindows`/`GetExitCodeProcess` 轮询，不读内存、不 BitBlt | 第 **4.424 s** 进程消失，`exitCode=3221225477`，日志里同样的 `0xc0000005 / 0x000a5cff` |
| B/C | 读内存 / BitBlt | 同样在 ~4 s 死（同样的偏移） |

结论：崩溃与观测无关，是零售件自身的启动路径问题。

### 1.3 隔离变量：`score.dat`

唯一被改动的输入是 `score.dat`：**不复制**它（`--no-score-dat`，让零售件自己创建）后：

- `--phase idle --idle-read` 22.0 s 稳定存活，`supervisorState` 恒为 1，退出码 0；
- 后续 `launch/memory/capture` 三次运行全部退出码 0。

尺寸对比（三份互不相同）：

| 文件 | 大小 |
| --- | --- |
| 重建构建写出的 `play_fence/score.dat` | 17,773 B |
| 零售件崩溃瞬间自建的 `score.dat` | 275 B |
| 零售件正常运行后定型的 `score.dat` | 16,035 B |

因此：**`th08.dat`/`thbgm.dat` 与零售件是兼容的（它们被原样使用，游戏能正常读盘、正常渲染），
而重建构建写出的 `score.dat` 与零售 1.00d 不兼容，会让零售件在字符串比较里解引用坏指针。**

### 1.4 崩溃点定位（无 PDB 的代码同一性）

零售件没有符号。研究构建是逐函数匹配的，所以用"字节窗口 + 链接映射"反查：

```
python .tmp_v400_retail_win_crash_sym.py     # 退出码 0
```

- 零售 `RVA 0x000A5CFF` → 文件偏移 `0x000A5AFF`；
- 该处 32 字节窗口 `cc558bec5756538b4d10e3278bd98b7d088bf733c0f2aef7…`
  在研究构建 `build/play/th08.exe` 中唯一命中：`RVA 0xAB9FF` → `VA 0x4AB9FF`；
- 研究构建 `build/th08.map` 里该窗口落在 **`_strncmp` = 0x004ABA00**（下一个符号 `_strcpy` = 0x004ABA40），
  即崩溃点 = **`strncmp + 0x17`**；
- 崩溃处的原始字节是 `f3 a6` = `rep cmpsb`（字符串比较循环），`0xC0000005` 说明 ESI/EDI 里是
  空指针或悬垂指针。

这与 `third_party/th08/docs/RUNTIME_ISSUES.md` 的 **RT-004** 是同一个机理
（"unchecked `strchr(..., '\n') + 1` 得到 `0x1`，随后 `strncmp` 触发 0xC0000005"），
区别只是：RT-004 发生在**重建构建**上，本卡在**零售原件**上复现了同一类字符串解引用崩溃。
具体是 `strncmp` 的哪个调用者（版本串校验 / ScoreDat 版本比较）**未证实**，不在本卡结论内。

---

## 二、可行性 1：零售原件启动并进入回放 ✅

命令（`--no-score-dat` 为 1.3 节得出的必要条件）：

```
python tools/rl/v400_retail_win_capture.py --phase launch --no-score-dat \
  --run-dir .tmp_v400_retail_win_rundir_fresh2 --out .tmp_v400_retail_win_launch2
# 退出码 0，elapsed 8.187 s
```

时间线（`report.json` 的 `events`，单位 s，自 `Popen` 起）：

| 事件 | t |
| --- | --- |
| `process_started` (pid 10832) | 0.008 |
| 窗口稳定 + 读句柄/开句柄/挂 reader | 0.515 |
| `title_screen`（`g_Supervisor+0x15C == 1`） | 0.875 |
| `replay_keys_sent`（closed_loop 导航完成） | 3.693 |
| `replay_playback ok=true` | 3.693 |

- 窗口：`class=BASE`、`hwnd=0x1E06C0`、`visible=true`、客户区 **640×480**、
  标题为日文「東方永夜抄 ～ Imperishable Night. ver 1.00d」（本机 ANSI 代码页 CP936 下取回为乱码，
  与 `touhou_rl/winapi.py` 首部注释的原因一致；原始串已按 UTF-8 hex 存入 `report.json` 便于追溯）
- 导航走的是**内存闭环**（`navigation.path = "closed_loop"`，不是盲按键）：说明零售件的
  `TitleScreen` 结构偏移与研究构建一致——`title_cursor=0`、`title_screen_current=0`、
  `title_screen_state=1`、`title_timer2=1188`、`title_replay_count≥1` 全部读对，
  `menu.goto_start_menu_item`/`enter_replay` 才能在 3.2 s 内走完
  StartMenu→ReplayList→StageSelect→PlaybackMode→播放。
- 播放开始后的内存快照（`result.playbackProbe`）：

```
supervisorWanted=2  supervisorState=2  flags=0x0C(FLAG_REPLAY=0x8)  replayMode=0
gameplayFrame=119  stageFrames=119  runFrames=119  stage=0  inGameMenu=0
globalsPointer=62679008  score=0  graze=0  deaths=0.0  lives=2.0  bombs=3.0  power=0.0
playerState=3（入场无敌）  guiInputCurrent=1  guiInputPrevious=1
```

- 截图：`.tmp_v400_retail_win_launch2/01-title.png`（标题画面）、
  `03-playback.png` + `03-playback-hud.png`（回放中的游戏画面）
- 同一目录的 `log.txt` 是零售件自己写的 Shift-JIS 运行日志，直接证明引擎初始化正常：

```
東方動作記録 ---------------------------------------------
ウィンドウモードで起動します          ← 我们的 th08.cfg 被接受（windowed=1）
垂直同期を取りません                  ← disableVsync opt 被接受
T&L HAL で動作しま～す
…
DirectSound は正常に初期化されました
使えるパッドが存在しないようです、残念
DirectInput は正常に初期化されました
```

---

## 三、可行性 2：从零售进程读状态 ✅

```
python tools/rl/v400_retail_win_capture.py --phase memory --no-score-dat \
  --run-dir .tmp_v400_retail_win_rundir_fresh2 --out .tmp_v400_retail_win_memory
# 退出码 0，elapsed 12.437 s
```

### 3.1 用的地址（全部来自 `config/reccmp-globals.csv` / `config/globals.csv` 的**零售**地址）

| 符号 | 零售地址 | 与重建 `spec.py` 默认值 |
| --- | --- | --- |
| `g_Supervisor` | `0x017CE758` | 0x017DC380（不同） |
| `g_GameManager` | `0x0160F508` | 0x0161C7A0（不同） |
| `g_Player` | `0x017D5EF8` | 0x017E3120（不同） |
| `g_TitleScreen`（指针） | `0x018BDE08` | 0x018CB030（不同） |
| `g_Gui` | `0x0160F428` | 0x0161C680（不同） |
| `g_CurFrameInput` | `0x0164D528` | 0x0165A808（不同） |
| `g_LastFrameInput` | `0x0164D530` | 0x0165A804（不同） |
| `g_GuiMessageInputCurrent` | `0x0164D52C` | 0x0161C674（不同） |
| `g_GuiMessageInputPrevious` | `0x0164D534` | 0x0161C678（不同） |

结构体偏移沿用重建的 `C_ASSERT` 值，本卡**逐项实测验证**了下面这些：
`OFF_SUP_CUR_STATE=0x15C`、`OFF_GM_GAMEPLAY_FRAME=0x3DDC0`、`OFF_GM_CURRENT_STAGE=0x3DDC4`、
`OFF_GM_FLAGS=0x3DBAC`、`OFF_GM_GLOBALS=0x8`、`OFF_GLOBALS_SCORE=0x8`、`OFF_GLOBALS_LIVES=0x74`、
`OFF_GLOBALS_DEATHS=0x64`、`OFF_TITLE_CURRENT_SCREEN=0x14428`。

> 注意：零售件把 `g_GuiMessageInputCurrent/Previous` 放在相距 **8** 字节处，而重建的
> `GameStateReader` 用「Previous − Current = 4」的相对距离去解包，会直接
> `struct.error: requires a buffer of at least 10 bytes … (actual 8)`——所以
> `tools/rl/v400_retail_win_capture.py` 内置了 `RetailStateReader`，**每个全局都按自己的绝对地址读**，
> 只填 `GameState` 字段，全部只读。（未改 `touhou_rl/state.py`、`session.py`、`menu.py`。）

### 3.2 帧计数器：单调 + 60 fps + 与 HUD 一致

- 195 次采样（间隔 20 ms），`gameplayFrame` = 150 → 390，**195 个互不相同的值**，
  跨度 240 帧 / 3.9946 s ⇒ **60.08 fps**；`frameDeltaMin=1`、`frameDeltaMax=2`（无跳帧、无倒退）
- `monotonicProblems: []`（`gameplayFrame`/`score`/`deaths` 均无倒退）
- 资源字段：`lives=2.0`、`bombs=3.0`、`deaths=0`、`power=0.0`、`graze=0`、`stage=0` 恒定（stage 0 起始）
- **HUD 对照**：同一次运行的 `03-playback.png` 上，HUD 显示
  `HiScore 0000000000 / Score 0000000000 / Player ★★（2）/ Spell ★★★（3）/ Power 0 / Graze 0 /
  Point 0/100 / Time 0/2500 / Normal / 60.00 fps`，与内存 `score=0、lives=2.0、bombs=3.0、
  power=0.0、graze=0` **逐项一致**（`Player ★★`↔`lives=2.0`，`Spell ★★★`↔`bombs=3.0`）。
- **独立偏移扫描**（不信任预设偏移）：把 `g_GameManager` 起始 253,488 B 读两遍、间隔 0.5 s，
  找所有"增量 ≈ 30"的 u32，得到 7 个候选，`0x3DDC0` 在列且 `120 → 150`：

```
hits = 0x3db94, 0x3dba0, 0x3dba4, 0x3ddc0, 0x3de04, 0x3de10, 0x3de14
0x3ddc0: before=120 after=150 delta=30     ← OFF_GM_GAMEPLAY_FRAME 命中
0x3de10: before=120 after=150 delta=30     ← OFF_GM_RUN_ACTIVE_FRAMES
0x3de14: before=120 after=150 delta=30     ← OFF_GM_STAGE_ACTIVE_FRAMES
```

---

## 四、可行性 3：GDI 截图 ✅

`touhou_rl/winapi.py` **已有**客户区抓图能力（`GdiCapture`：`GetDC(hwnd)` + `BitBlt` 到 32 位
自顶向下 DIB，返回 `uint8[H,W,3]` RGB），本次**未修改 `winapi.py`**——只在其上补了 PNG 落盘与
HUD 裁剪（用 Pillow），放在工具里。

| 画面 | 路径 | 字节 | 尺寸 | 颜色数 |
| --- | --- | --- | --- | --- |
| 零售标题画面 | `.tmp_v400_retail_win_launch2/01-title.png` | 742,333 | 640×480 | 171,762 |
| 零售回放画面（含 HUD） | `.tmp_v400_retail_win_launch2/03-playback.png` | 181,745 | 640×480 | 22,999 |

- 标题画面是完整的**零售原版标题美术**（月光、剪影、"東方 Project / Project Shrine Maiden /
  少女祈祷中 Now Loading..." 与版权英文），不是占位图；
- 回放画面里 HUD、符卡星数、难度 `Normal`、右下 `60.00 fps` 全部清晰可读；
- 说明 `BitBlt` 在零售件（D3D8 窗口模式、非全屏、`COPY` 呈现）上拿到的就是真实呈现帧，
  与 `winapi.py` 注释里"1.98 ms/帧、被遮挡也返回真实帧"的结论一致。

---

## 五、最小闭环：20 个连续 logicFrame ×（1 张 PNG + 内存状态）✅

```
python tools/rl/v400_retail_win_capture.py --phase capture --no-score-dat \
  --run-dir .tmp_v400_retail_win_rundir_fresh2 --out .tmp_v400_retail_win_capture \
  --settle-seconds 2 --consecutive-frames 20
# 退出码 0，elapsed 7.783 s
```

- 同帧证明：每个样本在 `BitBlt` **前后各读一次** `g_GameManager+0x3DDC0`，只有两次相等才接受，
  因此该 PNG 归属的 logicFrame 是确定的（`sameFrameProvenAll: true`）；
- 采集统计：`attempts=234`、`accepted=216`、`duplicateFrames=196`（同帧重复采样被丢弃）、
  `skippedAttempts=18`（跨帧竞争）；
- **连续帧号：59, 60, 61, …, 78（20 帧，全部 +1）**；
- 20 张 PNG + 20 张 HUD 裁剪 = 40 个文件，位于 `.tmp_v400_retail_win_capture/frames/`；
- 19/20 张 PNG 逐字节不同 ⇒ 每个 logicFrame 至少拿到了 19 个不同的真实画面。

| # | logicFrame | 读前==读后 | score | lives | bombs | deaths | power | graze | stage | playerState | t(s) | PNG 字节 | PNG sha256[0:16] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 59 | 59==59 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.608 | 137,917 | 3dea5e2ed1d5febd |
| 1 | 60 | 60==60 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.617 | 137,698 | 7d9187e3ed13b052 |
| 2 | 61 | 61==61 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.633 | 137,698 | 7d9187e3ed13b052 |
| 3 | 62 | 62==62 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.650 | 140,890 | 58db24f3e8b8dc57 |
| 4 | 63 | 63==63 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.667 | 143,142 | bf9fb49d83960b81 |
| 5 | 64 | 64==64 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.683 | 145,178 | 583919fbc94a74f6 |
| 6 | 65 | 65==65 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.700 | 146,381 | 12a3cb0bac180f58 |
| 7 | 66 | 66==66 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.716 | 146,360 | 1b2885d751d4ba08 |
| 8 | 67 | 67==67 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.734 | 147,491 | 0f1ff3d314297407 |
| 9 | 68 | 68==68 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.750 | 149,362 | 7fbf0bc9e18d86e9 |
| 10 | 69 | 69==69 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.766 | 150,276 | 321eff16a3bcf7b3 |
| 11 | 70 | 70==70 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.784 | 151,958 | 69c23cb1461440c9 |
| 12 | 71 | 71==71 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.800 | 151,561 | 2f857a5188e3fa20 |
| 13 | 72 | 72==72 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.816 | 152,254 | 25e3547b2d7c14b5 |
| 14 | 73 | 73==73 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.833 | 154,087 | 7e91b8efabec5653 |
| 15 | 74 | 74==74 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.850 | 155,332 | 6ca9661f8858a91e |
| 16 | 75 | 75==75 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.867 | 156,354 | 5190881a528e9f85 |
| 17 | 76 | 76==76 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.884 | 156,122 | de8c94d4779e7d73 |
| 18 | 77 | 77==77 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.900 | 155,962 | 3bb39f9e5859525e |
| 19 | 78 | 78==78 | 0 | 2.0 | 3.0 | 0 | 0.0 | 0 | 0 | 3 | 5.916 | 157,408 | e7fa755fffe7358e |

完整记录（每帧 30+ 字段、PNG 绝对路径、HUD 路径、逐帧 SHA256）在
`.tmp_v400_retail_win_capture/report.json` 的 `frames[]`。

`frame-000060.png` 与 `frame-000061.png` 字节相同（都是 137,698 B / `7d9187e3…`）：
两帧都被计数器证明是**不同的 logicFrame**，但呈现像素一致——这段正好是 stage 入场卡
（`playerState=3` 无敌、画面基本静止），GDI 抓到的是同一张呈现帧。见第七节限制 ③。

---

## 六、全部命令与退出码

```
# 1) 首跑（用 play_fence 的 score.dat）：partial，崩溃
python tools/rl/v400_retail_win_capture.py --phase launch --out .tmp_v400_retail_win_launch \
  --run-dir .tmp_v400_retail_win_rundir --force-run-dir
→ exit 6；closeExitCode 3221225477；WER 0xc0000005 @ RVA 0x000A5CFF

# 2) 对照实验 A：完全不观测，仍崩
python tools/rl/v400_retail_win_capture.py --phase idle --idle-seconds 15 \
  --out .tmp_v400_retail_win_idle_none2
→ exit 4；4.424 s 崩溃，同一偏移

# 3) 换新 score.dat（--no-score-dat）：22 s 存活
python tools/rl/v400_retail_win_capture.py --phase idle --idle-seconds 20 --idle-read \
  --no-score-dat --run-dir .tmp_v400_retail_win_rundir_fresh2 --force-run-dir \
  --out .tmp_v400_retail_win_idle_freshscore2
→ exit 0

# 4) 启动 + 进回放
python tools/rl/v400_retail_win_capture.py --phase launch --no-score-dat \
  --run-dir .tmp_v400_retail_win_rundir_fresh2 --out .tmp_v400_retail_win_launch2
→ exit 0

# 5) 内存校验
python tools/rl/v400_retail_win_capture.py --phase memory --no-score-dat \
  --run-dir .tmp_v400_retail_win_rundir_fresh2 --out .tmp_v400_retail_win_memory
→ exit 0

# 6) 20 连续帧闭环
python tools/rl/v400_retail_win_capture.py --phase capture --no-score-dat \
  --run-dir .tmp_v400_retail_win_rundir_fresh2 --out .tmp_v400_retail_win_capture --settle-seconds 2
→ exit 0

# 7) 崩溃点符号定位
python .tmp_v400_retail_win_crash_sym.py
→ exit 0（strncmp + 0x17）

# 8) 单测
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q \
  tests/test_v400_retail_win_capture.py
→ exit 0，15 passed
```

每次运行的 `report.json` 都含：镜像身份、地址表 SHA256、`addressesUsed`、`structOffsetsUsed`、
`events` 时间线、`thirdPartyUnchanged`、`sessionLog`、`closeExitCode`、`elapsedSeconds`。

工具自身缺陷（已修，如实记录）：`.tmp_v400_retail_win_idle_none` 与
`.tmp_v400_retail_win_idle_freshscore` 两次 `exit 6` 是**工具 bug**
（读 `TitleScreen` 时缓冲区按 0x14 分配却读 0x6C 偏移，`struct.error`），与零售件无关；
修好后重跑得到 `idle_none2`（崩溃）与 `idle_freshscore2`（存活）。

---

## 七、限制与未验证项

1. **崩溃根因只到"层"**：已确定是零售件对**重建构建写出的 `score.dat`**（17,773 B ≠ 零售自建
   16,035 B）做的字符串比较里解引用坏指针（`strncmp + 0x17`，`rep cmpsb`），
   但**具体调用者**（`Supervisor::CheckVersion` 的版本串 vs `ScoreDat` 的版本字段）未证实——
   需要 CDB/x64dbg 挂零售件，本次任务未做。
2. **为什么崩溃发生在标题画面之后而不是启动瞬间**：首跑里标题画面 PNG 是正常的（零售原版美术），
   说明 `Supervisor` 初始化、`th08.dat` 读盘、D3D8 初始化都已成功；崩溃点在标题画面渲染后
   0.1–1.5 s。该时序差异未进一步解释。
3. **"同帧"不等于"像素一定变"**：零售构建没有 frame-fence 翻译单元，无法把逻辑帧摁住。
   本工具用"BitBlt 前后 `gameplayFrameCounter` 相同"证明像素归属哪一帧，但**不能**保证
   相邻 logicFrame 的呈现像素必然不同（frame 60/61 就相同）。若下游要"每帧必新"，
   需要另设像素差分门（本次没有实现）。
4. **HUD 对照是人工目视**：没有 OCR。HUD 数字与内存值的逐项对照由本卡人工完成
   （Score/Player/Spell/Power/Graze/Point/Time/fps），工具只负责把 HUD 区域裁成
   `frame-XXXXXX-hud.png` 存档。
5. **`score` 在采样窗口内恒为 0**：回放从 stage 0 起始卡开始，4 s 内还没有任何得分入账，
   所以"score 不倒退"这一条在本次证据里是**平凡成立**；有意义的得分单调性证据只有
   HUD 与内存同为 0 的一致性。
6. **`g_Player` 的位置偏移未单独验证**：读到 `playerX=376.0, playerY=384.0`（合理但未与画面
   逐像素对齐），本卡不使用该字段做结论。
7. **一次一个进程**：`Touhou 08 App` 是会话级互斥量，所有 phase 串行、每次运行前检查
   `list_process_ids("th08.exe")` 必须为空；8 次运行的 `otherInstances` 都是 `[]`。
8. **单次运行 ≤ 180 s**：`--budget-seconds` 默认 170 s，实测最长 53.8 s（那次是崩溃后的超时等待），
   成功的四次运行为 7.8–23.7 s。
9. **未使用管理员权限/未改注册表**：零售件不需要安装或注册，全程普通权限运行；
   没有触发 UAC。`__COMPAT_LAYER=HIGHDPIAWARE` 只对子进程生效（DPI 兼容层，不写系统设置）。

---

## 八、复现步骤（干净目录）

```powershell
Set-Location D:\WorkShop\Touhou_RL
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
Get-Process th08 -ErrorAction SilentlyContinue        # 必须为空
python tools/rl/v400_retail_win_capture.py --phase idle --idle-seconds 10 --idle-read `
  --no-score-dat --run-dir .tmp_v400_retail_win_rundir_fresh2 --force-run-dir `
  --out .tmp_v400_retail_win_repro1                   # 期望 exit 0
python tools/rl/v400_retail_win_capture.py --phase launch --no-score-dat `
  --run-dir .tmp_v400_retail_win_rundir_fresh2 --out .tmp_v400_retail_win_repro2
python tools/rl/v400_retail_win_capture.py --phase memory --no-score-dat `
  --run-dir .tmp_v400_retail_win_rundir_fresh2 --out .tmp_v400_retail_win_repro3
python tools/rl/v400_retail_win_capture.py --phase capture --no-score-dat `
  --run-dir .tmp_v400_retail_win_rundir_fresh2 --out .tmp_v400_retail_win_repro4
```

反向复现崩溃（**故意**用重建的 score.dat）：去掉 `--no-score-dat` 并把 `--run-dir` 指到一个
已经被 `prepare_run_dir` 复制过 `play_fence/score.dat` 的目录（例如本次的
`.tmp_v400_retail_win_rundir`），期望在 3–5 s 内 `0xC0000005 @ 0x000A5CFF`。

---

## 九、产物清单

| 路径 | 内容 |
| --- | --- |
| `tools/rl/v400_retail_win_capture.py` | 工具（`launch`/`memory`/`capture`/`idle` 四个 phase，含 `RetailStateReader`） |
| `tests/test_v400_retail_win_capture.py` | 15 个单测（不启动游戏） |
| `.tmp_v400_retail_win_addresses.json` | 零售 1.00d 地址表（`th08-map-addresses/1`） |
| `.tmp_v400_retail_win_crash_sym.py` | 崩溃点代码同一性定位 |
| `.tmp_v400_retail_win_launch2/` | 可行性 1 证据：`report.json`、`01-title.png`、`03-playback.png`、HUD 裁剪 |
| `.tmp_v400_retail_win_memory/` | 可行性 2 证据：`report.json`（195 采样 + 偏移扫描） |
| `.tmp_v400_retail_win_capture/` | 可行性 4 证据：`report.json` + `frames/`（20 PNG + 20 HUD） |
| `.tmp_v400_retail_win_idle_none2/` | 对照实验 A：不观测也崩（4.424 s） |
| `.tmp_v400_retail_win_idle_freshscore2/` | 换 score.dat 后 22 s 存活 |
| `.tmp_v400_retail_win_launch/`、`.tmp_v400_retail_win_idle_none/`、`.tmp_v400_retail_win_idle_freshscore/` | 失败/无效运行的原始报告（含 WER 事件、`startupDialog` 为空、工具 bug 的 traceback） |

**未修改**：`third_party/th08/` 任何内容、零售 `th08.exe`、`th08.dat`/`thbgm.dat`/`score.dat` 原件、
`touhou_rl/winapi.py`、`touhou_rl/state.py`、`touhou_rl/session.py`、`touhou_rl/menu.py`、依赖。
所有运行都在 `.tmp_v400_retail_win*` 私有目录内，数据资产是 `copy2` 复制（不是硬链接），
8/8 次运行的 `thirdPartyUnchanged` 均为 `true`。
