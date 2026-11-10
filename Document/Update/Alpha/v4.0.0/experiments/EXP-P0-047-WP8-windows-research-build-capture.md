# EXP-P0-047：WP8「Windows 侧真实渲染帧」来源可行性核查

日期：2026-11-08。Provider=DeepSeek，模型=DeepSeek v4.1 flash。

本实验只做**可行性核查**，不构成 WP8 完成，也不代表任何 exact-match 声明。零售原件
`third_party/th08/resources/th08.exe` 全程只读；`third_party/th08` 源码与构建未改动；
未写任何伪造 PNG。

## 结论

1. **前提需要修正**：`third_party/th08/build/play/th08.exe`（898,048 B，SHA256
   `cdd23bca…1ddf`）**没有编入帧栅栏**。它与 `build/th08.exe` 字节完全相同，是
   **bugfix 构建**（`build/play/runtime-manifest.json` 的 `exe.buildType` 就是
   `"bugfix"`），镜像里不存在任何栅栏字符串或栅栏调用点。
   **真正编入帧栅栏的是** `third_party/th08/build_fence/th08.exe`（905,728 B，SHA256
   `bc1894fe…8565`），它已被 `tools/rl/frame_fence_round1.py: prepare_fence_template()`
   装配进固定 data-dir `third_party/th08/build/play_fence/`（exe 同 SHA，并含
   `th08.dat` / `thbgm.dat` / `score.dat` / `th08.cfg`）。
2. **Q2 可行**：Windows 上确实有一条可复制命令，对**恰好 1 个** `logicFrame` 产出 PNG +
   状态哈希。本次实跑成功：退出码 **0**，`samples=1`、`discarded=0`、**13/13** 同帧校验
   全部为 true（含 `pixels_match_backbuffer`、`bullet_digest`、`enemy_digest`）。
3. **Q3 最小命令**：见下文「最小可执行命令」。注意 `--mode demo` 可用，
   `--mode practice` 在本机被输入注入卡住（退出码 1），卡点已定位到具体函数。

## 一、研究构建是否编入帧栅栏（Q1 证据）

### 镜像鉴别

| 镜像 | 字节 | SHA256 | `.text` SHA256 | 栅栏标记 |
|---|---|---|---|---|
| `build/play/th08.exe` | 898,048 | `CDD23BCA0C724941866EFDA5DA280BE67EB6CDD935F83548045C2A8E037F1DDF` | `7B5B3C8A…` | **无** |
| `build/th08.exe` | 898,048 | 同上（字节相同） | `7B5B3C8A…` | **无** |
| `build_fence/th08.exe` | 905,728 | `BC1894FED48BEAC739F628836629B8B1B1C1B63DBE0097239C2762CB4C8D8565` | `34A8070B…` | 有 |
| `build/play_fence/th08.exe` | 905,728 | 同 `BC1894FE…` | `34A8070B…` | 有 |
| `resources/th08.exe`（零售，只读） | 840,704 | `330FBDBF58A710829D65277B4F312CFBB38D5448B3DF523E79350B879213D924` | `1F204188…` | 无 |

栅栏标记（ASCII 实测）：`Local\th08.frame_fence.v1`、`framefence: ready name=`、
`framefence: EXCEPTION code=`、`framefence: CreateFileMapping failed`；以及仅栅栏镜像
才导入的 `AddVectoredExceptionHandler`。

### 调用点（不是"有符号"，而是"被调用"）

- 链接图 `build_fence/th08.map` 中有
  `?OnDrawnFrameEnd@FrameFence@@YIXXZ` = **`0x004764A0`**。
- 在 `build_fence/th08.exe` 的 `.text` 里扫描 `E8 rel32`，**恰好 1 处**调用该地址：
  `0x004338EA`，位于 `?Render@GameWindow@th08@@QAE?AW4RenderResult@2@XZ`（`main.obj`）。
- `build_fence/.text` 原始大小 772,608 B，`build/play` 为 765,440 B，差 **7,168 B**。
- 源码侧：`src/main.cpp:326-387`（`#ifdef TH08_FRAME_FENCE` 记 `drewThisFrame`，
  `Present()` 之后 `if (drewThisFrame) FrameFence::OnDrawnFrameEnd();`）。
- 构建侧：`scripts/configure.py` 的 `BuildType.FENCE = 6`（`builddir="build_fence"`、
  追加 `/DTH08_FRAME_FENCE`、把 `framefence` 加入 TU 列表）；当前 `build.ninja` 就是 fence
  图（`builddir = build_fence`、`cl_flags = $cl_flags /DTH08_FRAME_FENCE`、链接
  `$builddir/framefence.obj`）。`build/play_fence` 里没有任何栅栏条件，它是**运行目录**。

### 地址表（栅栏镜像移动了每个全局）

- `build_fence/frame-fence-addresses.json` 的 `mapSha256` =
  `CE7B6672F9488909A5F700E6A52900C65358A4201B0DE5E62EE904F8A752BB54`，与实测
  `build_fence/th08.map` 哈希**一致**（表是新鲜的）；表文件自身 SHA256 =
  `79237A295567EDF0BAD6C557BCC1AAF8A7D1DB369D2C54429FBC1B779B85634A`。
- 栅栏表 `g_Supervisor=0x017DE380` / `g_BulletManager=0x00F640B8`，规范常量是
  `0x017DC380` / `0x00F620B8`（整体 +0x2000）。
- `TH08_ADDRESS_FILE` 机制有效：设置后 `touhou_rl.spec` 导入即得
  `ADDR_SUPERVISOR=0x017DE380`，`ADDRESS_SOURCE.addressFile` 指向栅栏表。
- 副产物：`third_party/th08/build/frame-fence-addresses.json` 已 **stale**（记录
  `3564CC1B…`，实测 `build/th08.map` 为 `C10F3F3C…`）。`touhou_rl/exact_capture.py` 用的是
  `build_fence/` 那份，因此不影响 exact 路径，但不要把 build/ 那份当成栅栏镜像的表。

### 运行期活性（有界探测，未写 PNG）

`.tmp_v400_wp8_winprobe_fence_live.py`（约 12 s，启动后立即关闭）：

- fence 镜像启动到标题 **3.52 s**，客户区 **640x480**；
- 共享段 `Local\th08.frame_fence.v1` 发布：`magic=0x38454E46`（'FNE8'）、`version=1`、
  `blockSize=925,696`（= 0x1000 + 640·480·3）、`pid` 与本次 session 一致；
- 开启 `CTL_ENABLE` 后 `ticket` 由 0 → 1 → …（每绘制/呈现帧 +1）；
- 开启 `CTL_PIXELS` + hold 后：`holdPhase==1`、`pixelTicket==ticket`、
  `digestTicket==ticket`（`bulletDigest=0xd3982253`、`enemyDigest=0x592b3105`）、
  像素 `480x640x3` 可读、非黑像素比 **0.9916**；ack 后 ticket 继续递增。
- `pixelTicket==ticket` 说明 **D3D8 后备缓冲 `LockRect` 在本机可用**（这是最容易被
  驱动/WDDM 破坏的一环）。

## 二、现有 exact 工具在 Windows 上是否有一条可复制命令（Q2）

是，但需要按"谁有 CLI"区分：

| 入口 | 是否有 CLI | `--help` 退出码 | 能否产出 PNG + 状态哈希 |
|---|---|---|---|
| `tools/rl/frame_fence_capture.py` | 有 | **0** | **能**（PNG + `capture.jsonl` + `manifest.json`） |
| `tools/rl/v380_fence_probe.py` | 有 | **0** | 能（`ExactFrame` 记录 + PNG） |
| `touhou_rl/exact_capture.py` | **无**（库） | `python -m touhou_rl.exact_capture --help` → **0 且零输出**；`python touhou_rl\exact_capture.py --help` → **1**（`ImportError`） | 只能被 `capture_replay(frame_proof="exact")` 调用 |
| `touhou_rl/capture_dataset.py` | 有（需 `-m`） | `python -m touhou_rl.capture_dataset --help` → **0**；直接按路径跑 → **1**（相对导入） | 该 CLI **没有** `--frame-proof`，exact 路径不可由此命令行直达 |

关键限制（必须如实记录）：**这些工具都不能"指定 logicFrame=N 再抓"**。它们抓的是
"下一个被 hold 住的绘制帧"，并在记录里写出它的 `logicFrame`。因此"单帧"指
"一条命令只产出一帧"，不是"按帧号寻址"。

## 三、最小可执行命令（Q3，已实跑）

```powershell
cd D:\WorkShop\Touhou_RL
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python tools\rl\frame_fence_capture.py --frames 1 --mode demo --hold-every 1 `
  --out-dir .tmp_v400_wp8_winprobe_out_demo --run-dir .tmp_v400_wp8_winprobe_run_demo
```

退出码 **0**，`wallSeconds=37.168`，`samples=1`、`discarded=0`、`checksPassed=13`，
`fenceFinal.ticket=2141`、`holdMaxUs=11771`、`holdBudgetSkips=0`。

产物（真实采集，非合成）：

- `.tmp_v400_wp8_winprobe_out_demo/frames/00000-ticket-002140.png`
  406,824 B，PNG 640x480、bit depth 8、color type 2（RGB），文件 SHA256
  `C4B73559E1044A4CF28DDA076DFE9D56130C2A9827A1ECCFBE8C81237E896196`；
- `capture.jsonl` 一条记录：`ticket=2140`、`logicFrame=522`、`stageFrames=522`、
  `activeBullets=242`、`bulletDigest=804247521`、`enemyDigest=711840888`、
  玩家 `x=182.0 y=352.0 vy=-4.0`、`isDemo=true`、`isReplay=true`，
  `checks` 13 项全 true（`hold_stable`、`player_position`、`player_velocity`、
  `player_state`、`focus_mode`、`gameplay_frame`、`stage_frames`、`input_mask`、
  `active_bullets`、`bullet_digest`、`enemy_digest`、`pixels_frozen`、
  `pixels_match_backbuffer`）；
- `manifest.json`：`exeSha256=BC1894FE…`、`textSha256=34A8070B…`、
  `addressFileSha256=79237A29…`（研究构建 + 固定地址表的自证）。

`pixels_match_backbuffer` 为 true 意味着 GDI 抓到的客户区像素与游戏自己发布的后备缓冲
**逐字节相等**——这正是"Windows 侧真实渲染帧"要的证据类型。

## 四、卡点（practice / replay 路线）

同一工具换 `--mode practice` 会失败：

```powershell
python tools\rl\frame_fence_capture.py --frames 1 --mode practice --hold-every 1 `
  --out-dir .tmp_v400_wp8_winprobe_out --run-dir .tmp_v400_wp8_winprobe_run
# 退出码 1
# TimeoutError: could not move start menu cursor to item 3
#   at touhou_rl/menu.py:102 in goto_start_menu_item()
#   at touhou_rl/menu.py:201 in enter_game()
```

已定位到具体位置与缺失项：

- 文件/函数：`touhou_rl/menu.py:90-102`（`goto_start_menu_item`）→ `touhou_rl/menu.py:201`
  （`enter_game`）。5 s 内数十次方向键后 `title_cursor` 未变化，说明**键根本没进到游戏**。
- 输入后端：`th08.cfg` 被写成 `dontUseDirectInput`（`touhou_rl/config_file.py:167`
  设置 `OPT_DONT_USE_DIRECTINPUT`），于是 `GameSession` 选 `winapi.PostMessageKeyboard`
  （运行日志：`input backend: PostMessage (dontUseDirectInput)`）。
- 缺失项：`menu.tap()` **只在 `SendInputKeyboard` 分支**调用
  `winapi.force_foreground(...)`（`touhou_rl/menu.py:35-37`）；`PostMessageKeyboard` 分支
  从不激活窗口。当前会话里游戏窗口可创建、可渲染、可被 GDI 抓取，但没有被前台化，
  因此依赖线程键盘状态的 `dontUseDirectInput` 路径收不到按键。
- 注意：这不是"抓帧"能力的问题——`--mode demo` 不依赖任何输入，已证明抓帧链路完整。

## 五、可重复命令与退出码

| # | 命令 | 退出码 | 说明 |
|---|---|---|---|
| 1 | `python .tmp_v400_wp8_winprobe_preflight.py` | **0** | 镜像鉴别 + 地址表 + 依赖（numpy 2.5.4、PIL 13.0.0） |
| 2 | `python .tmp_v400_wp8_winprobe_callsite.py` | **0** | 栅栏调用点 1 处，`Render@GameWindow` |
| 3 | `python .tmp_v400_wp8_winprobe_fence_live.py` | **0** | 13 项活性探测全 PASS |
| 4 | `python tools\rl\frame_fence_capture.py --frames 1 --mode demo …` | **0** | 单帧实采（见 §三） |
| 5 | `python tools\rl\frame_fence_capture.py --frames 1 --mode practice …` | **1** | 卡点（见 §四） |
| 6 | `python tools\rl\frame_fence_capture.py --help` | **0** | |
| 7 | `python tools\rl\v380_fence_probe.py --help` | **0** | |
| 8 | `python -m touhou_rl.exact_capture --help` | **0** | 零输出：无 CLI |
| 9 | `python touhou_rl\exact_capture.py --help` | **1** | `ImportError`（相对导入） |
| 10 | `python -m touhou_rl.capture_dataset --help` | **0** | 无 `--frame-proof` |
| 11 | `python touhou_rl\capture_dataset.py --help` | **1** | `ImportError`（相对导入） |

运行环境：Windows，`sys.platform=win32`，Python 3.12.10，交互式桌面可用，无其它
`th08.exe` 进程。日志：`.tmp_v400_wp8_winprobe_preflight.log`、
`.tmp_v400_wp8_winprobe_callsite.log`、`.tmp_v400_wp8_winprobe_fence_live.log`、
`.tmp_v400_wp8_winprobe_capture.log`、`.tmp_v400_wp8_winprobe_capture_demo.log`。

## 六、状态与下一步最小动作

- WP8 的"Windows 侧真实渲染帧来源"：**可行，来源已确认存在**（研究构建
  `build_fence/th08.exe` + 固定 data-dir `build/play_fence` + fence 共享内存协议）。
- 但 **WP8 未完成**：目前只有 1 帧 demo 帧；demo 是回放驱动（`isReplay=true`），
  不是策略可控帧；practice 路线的输入注入仍卡住。
- 下一步最小动作（按顺序）：
  1. 用已验证的 demo 路线再抓 2 帧，确认 `logicFrame` 严格单调、PNG 两两不同
     （排除"同一帧写两次"），仍用 `--frames` 小值，不做整批采集；
  2. 修 practice 输入：在 `PostMessageKeyboard` 分支也调用 `winapi.force_foreground`
     （或改回 `SendInputKeyboard` + 前台化），使 `menu.goto_start_menu_item` 能在 5 s 内
     移动光标——这是 `tools/rl` 侧改动，**不触及** `third_party/th08`；
  3. 只有 practice/replay 路线打通并抓到可核验的多帧之后，才谈 WP8 的像素清单与
     exact 采集，本文不构成该结论。
