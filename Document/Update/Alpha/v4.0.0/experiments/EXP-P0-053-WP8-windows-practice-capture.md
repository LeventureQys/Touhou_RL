# EXP-P0-053：WP8 Windows `--mode practice` 采集打通（前台化 + 引擎真实输入路径）

- 日期：2026-11-08
- 环境：Windows，`sys.platform=win32`，Python 3.12.10，numpy 2.5.4，Pillow 13.0.0，
  NVIDIA RTX 4070 SUPER / 617.42，交互式会话 session 1
- 镜像：`third_party/th08/build_fence/th08.exe`，905,728 B，
  SHA256 `BC1894FED48BEAC739F628836629B8B1B1C1B63DBE0097239C2762CB4C8D8565`，
  `.text` SHA256 `34A8070B4854672C92A66C95472F9DF7FB3B83C0AFEB6FBD15BBAED45A8593C7`，
  地址表 `frame-fence-addresses.json` SHA256
  `79237A295567EDF0BAD6C557BCC1AAF8A7D1DB369D2C54429FBC1B779B85634A`
- 前置：EXP-P0-047（研究构建可采帧已确认，practice 卡在 `goto_start_menu_item`）

## 结论

**打通。** `--mode practice` 现在可退出码 0 产出真实 640×480 帧；`--frames 2`
两次采样 `logicFrame` 严格单调（571 → 573）、两张 PNG 逐字节不同。

但根因与 047 的推测**不同**：只给 `PostMessage` 分支补前台化**不够**——TH08 的 Windows
构建根本不看 `WM_KEYDOWN`，它读 `GetKeyboardState`，而投递消息不改线程键盘状态。
真正的修法是"前台化 + 把输入后端升级为 scan-code `SendInput`"。另外
`winapi.force_foreground` 的**一次性校验**本身会把已经成功的焦点切换判成失败。

## 一、失败根因（四层，逐条有证据）

### 1.1 PostMessage 在本引擎上证明无效（代码 + 实测）

- `touhou_rl/config_file.py:167` 写 `OPT_DONT_USE_DIRECTINPUT` →
  `Supervisor.cpp:898` 令 `g_Supervisor.keyboard = NULL` →
  `Global.cpp:729-731` 走 `GetKeyboardState(keyboardState)`，`KEYBOARD_KEY_PRESSED`
  判 `keyboardState[x] & 0x80`（`Global.cpp:434`）。
- `GetKeyboardState` 返回**调用线程输入队列**的键状态；投递的 `WM_KEYDOWN` 不写入该数组。
- 窗口过程也不会兜底：`main.cpp` 的消息循环只处理 `WM_ACTIVATEAPP`（`main.cpp:531`），
  整个 `third_party/th08/src` 里没有任何 `WM_KEYDOWN/WM_KEYUP` 分支。
- 实测（`.tmp_v400_wp8_practice_probe.py`，退出码 0）：3 次 `VK_DOWN` 投递，
  `title_cursor` 0→0→0→0，`moved=false`。

所以 047 §四"补 PostMessage 分支的前台化即可"的判断**不成立**：前台化后投递仍然无效。

### 1.2 `force_foreground` 的一次性校验把成功判成失败

旧实现调用 `SetForegroundWindow` 后**立刻**读 `GetForegroundWindow()`。实测
（`.tmp_v400_wp8_practice_fgtest.py`，退出码 0）：

```
"SetForegroundWindow": {"returned": true, "lastError": 0, "foregroundNow": 0}
```

调用返回非零，但同一毫秒 `GetForegroundWindow()` 还是 `0`（前台尚未落到目标窗口）。
于是 `force_foreground` 返回 `False` → `menu.tap` 记录
`warning: could not take foreground for SendInput`，而焦点其实已经切过去了。
`SPI_GETFOREGROUNDLOCKTIMEOUT` 在本机为 1,768,876 ms，靠"超时自然放开"不可行，
必须走 `AttachThreadInput` + 轮询校验。

### 1.3 环境因素：锁屏期间没有任何前台可言

调查早期本机处于锁屏态（`GetForegroundWindow()` 返回
`class='Windows.UI.Core.CoreWindow' title='Windows 默认锁屏界面'`，`LockApp` PID 13620
与 `LogonUI` PID 3984 在场）。该状态下：

- `force_foreground` 必然失败（锁屏窗口占着前台）；
- `SendInput` 返回 `PermissionError(13, '拒绝访问', None, 5)`；
- 游戏启动间歇性弹 `DirectInput / Direct3D` 初始化失败对话框
  （`rl-session.log`: `boot attempt 1 failed: th08 start-up dialog: … DirectInput … Direct3D …`）。

桌面恢复交互（`WTSINFOEX.SessionFlags=1`，前台回到 DSH 主窗口）后，同样的代码全部通过。
**这是环境前置条件，不是代码缺陷**：锁屏时任何人都无法给这个引擎送真实按键。

### 1.4 已排除的"等价机制"：`AttachThreadInput` + `SetKeyboardState`

`.tmp_v400_wp8_practice_queueprobe.py`（退出码 0）把本线程附到游戏窗口线程后写
`SetKeyboardState`，`GetKeyboardState` 读回 `128`（写读一致），但 4 次 `VK_DOWN`
后 `title_cursor` 仍为 0（`movedDown=false`）。结论：键状态挂在**线程**而非队列上，
`AttachThreadInput` 不共享它，此路不通。这一条负结果写在这里，避免以后重复尝试。

## 二、改动（仅 `touhou_rl/menu.py`、`touhou_rl/winapi.py`）

### `touhou_rl/winapi.py`

1. `force_foreground(hwnd, timeout_s=2.0, attempts=3)` 重写：
   - 先 `_nudge_input()`（零位移 `SendInput` 鼠标移动）让本进程成为"最后输入进程"；
   - `ShowWindow(SW_RESTORE/SW_SHOW)` + `BringWindowToTop`；
   - 同时把本线程 `AttachThreadInput` 到**当前前台线程**与**目标线程**，
     `SetForegroundWindow` + `SetFocus` + `SetActiveWindow` 后解附；
   - **轮询** `GetForegroundWindow()`（每 20 ms）判定成功，失败再用
     `SwitchToThisWindow` 兜底；最多 3 轮。
2. 新增 `foreground_summary()`：当前前台窗口的 hwnd/class/title，一行字符串。
3. 新增 `input_window_state(hwnd)`：目标窗口 `visible/iconic/cloaked/foreground`。
4. 新增常量 `SW_SHOW`、`MOUSEEVENTF_MOVE` 与
   `SetActiveWindow`/`SwitchToThisWindow` 原型。

### `touhou_rl/menu.py`

1. 新增 `_ensure_engine_input(session) -> bool`：
   - 前台化失败 → 记一条带 `input_window_state()` + `foreground_summary()` 的告警并返回
     `False`（不再无声失败）；
   - 前台化成功且后端是 `PostMessageKeyboard` → 先 `release_all()`，
     再把 `session.keyboard` 就地换成 `SendInputKeyboard`，日志写
     `input backend upgraded PostMessage -> SendInput: …`。
     升级是**自memo** 的（升级后类型即 `SendInputKeyboard`），无需额外状态。
2. `tap()` 对两种后端都调用 `_ensure_engine_input()`（原来是 `isinstance(...SendInputKeyboard)`
   才前台化，PostMessage 分支从不前台化 —— 047 指出的那处）。
3. `_ensure_foreground_if_needed()` 改为同一入口（兼容旧调用）。
4. `goto_start_menu_item()` 超时异常补上诊断：卡住的 cursor 值、后端类型、
   `input_window_state()`、`foreground_summary()`。
5. 新增模块常量 `SENDINPUT_UPGRADE_REASON`，把"为什么投递无效"的代码级理由
   （`Global.cpp:731` / `main.cpp:531`）写在被引用的地方。

未改：`session.py`、`config_file.py`、任何 `third_party/th08` 文件、依赖、零售原件。

## 三、验收命令与真实产物

### 3.1 第一步：`goto_start_menu_item` 5 s 内移动 title cursor

```powershell
cd D:\WorkShop\Touhou_RL
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python .tmp_v400_wp8_practice_menuprobe.py
```

退出码 **0**，结果（`.tmp_v400_wp8_practice_menuprobe.log`）：

| 字段 | 值 |
|---|---|
| `backendAtStart` | `PostMessageKeyboard` |
| `cursorStart` | 0 |
| `secondsToPractice` | **0.323 s**（目标 item 3，预算 5 s） |
| `cursorAfter` | 3 |
| `backendAfter` | `SendInputKeyboard` |
| `secondsBackToStart` | 0.323 s（回到 item 0） |
| `foregroundBefore/After` | `foreground=0xAD0966 class='BASE' title='… Imperishable Night. ver 1.00d'` |

运行日志（同文件 `log` 字段）含升级行：
`23:38:20 input backend upgraded PostMessage -> SendInput: TH08 reads GetKeyboardState (Global.cpp:731) and its message loop handles no WM_KEYDOWN (main.cpp:531), so posted keys cannot reach the engine`。

### 3.2 `--mode practice --frames 1`

```powershell
python tools\rl\frame_fence_capture.py --frames 1 --mode practice --hold-every 1 `
  --out-dir .tmp_v400_wp8_practice_f1_out --run-dir .tmp_v400_wp8_practice_f1_run
```

退出码 **0**（日志 `.tmp_v400_wp8_practice_f1.log`）。
`manifest.json`：`wallSeconds=14.364`、`samples=1`、`discarded=0`、`checksPassed=13`、
`fenceFinal={ticket:767, holdCount:1, holdMaxUs:10498, holdBudgetSkips:0}`。

产物（真实采集，非合成）：

- PNG `.tmp_v400_wp8_practice_f1_out/frames/00000-ticket-000766.png`
  **417,855 B**，640×480、mode `RGB`，
  文件 SHA256 `774D5186A2E88941569AC3144676763CA17376BAC66212819E12048CA8761D6A`
- `capture.jsonl`（1 条）：`ticket=766`、**`logicFrame=559`**、`stageFrames=559`、
  `runFrames=559`、`stage=0`、`isPractice=true`、`isDemo=false`、`isReplay=false`、
  `inputMask=0`、`activeBullets=84`、`bulletDigest=2893594990`、
  `enemyDigest=3443436408`、玩家 `x=192.0 y=384.0 vx=0.0 vy=0.0 state=0 focus=0`、
  `imageSha256`（像素）`05F941FC07D5AAD776223A401CCC5B75387AE6586CD88AE9D038333937E8FB6C`、
  `readerWorkUs=10486.1`；
  `checks` 13 项全 `true`：`hold_stable`、`player_position`、`player_velocity`、
  `player_state`、`focus_mode`、`gameplay_frame`、`stage_frames`、`input_mask`、
  `active_bullets`、`bullet_digest`、`enemy_digest`、`pixels_frozen`、
  `pixels_match_backbuffer`。

`pixels_match_backbuffer=true` = GDI 抓到的客户区像素与游戏自发布后备缓冲**逐字节相等**。

### 3.3 `--mode practice --frames 2`（单调性 + 不重复写）

```powershell
python tools\rl\frame_fence_capture.py --frames 2 --mode practice --hold-every 1 `
  --out-dir .tmp_v400_wp8_practice_f2_out --run-dir .tmp_v400_wp8_practice_f2_run
```

退出码 **0**（日志 `.tmp_v400_wp8_practice_f2.log`）。
`manifest.json`：`wallSeconds=14.363`、`samples=2`、`discarded=0`、`checksPassed=26`、
`fenceFinal={ticket:769, holdCount:3, holdMaxUs:90000, holdBudgetSkips:1}`。

| # | ticket | logicFrame | PNG | 字节 | 文件 SHA256 | 像素 SHA256 (`imageSha256`) | bullets / bulletDigest / enemyDigest |
|---|---|---|---|---|---|---|---|
| 0 | 766 | **571** | `00000-ticket-000766.png` | 420,964 | `9D20FC0143885E58A2E60FB2BC2C3FB9A73992EA96F44000E84863F1363FF696` | `4341F55B58A4A68E2B8B205A40F92874046CEA4C9CF0ACD8149294A92374598E` | 99 / 4184000854 / 3945872487 |
| 1 | 768 | **573** | `00001-ticket-000768.png` | 419,140 | `6F88166C86FB396C259AE0FDB706492C78FBF878097F506E9B1C5EFC6FCD8FCD` | `F36F86C4FD95B3B033F721A95D2BD66412AA77448F7DD3F9F31489ECB0473EF5` | 98 / 2593984971 / 87853393 |

- 两张 PNG 均 640×480 / mode `RGB`；
- `logicFrame` 571 < 573，**严格单调**；
- 像素哈希与文件哈希两两不同 ⇒ **不是同一帧重复写**；
- 两条记录 `checks` 各 13 项全 `true`（合计 26）。

注意：`logicFrame` 571→573 跨了 2（不是 1），因为 `holdBudgetSkips=1`——`--hold-max-us`
默认 90,000 µs，慢速 PNG 写入让栅栏放弃了中间那一帧。要拿到**相邻** logicFrame 需提高
`--hold-max-us`，本卡未做（见 §六）。

## 四、单元测试

新增 `tests/test_v400_menu_foreground.py`（8 个用例，不启动游戏，用 fake session 锁定决策逻辑）：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests/test_v400_menu_foreground.py
# 8 passed in 0.60s   退出码 0
```

用例覆盖：前台成功时 PostMessage→SendInput 升级且先 `release_all`、前台失败时保留
PostMessage 且告警、SendInput 会话不被降级、`goto_start_menu_item` 下行/回绕上行方向、
超时异常的诊断字段（`stuck at 0` / 后端名 / `hwnd`）、非 start menu 屏幕直接报错、
`force_foreground` 对已前台窗口幂等返回 `True`。

回归（同命令，退出码均 0）：`touhou_rl/tests/test_core.py` **30 passed**、
`tests/test_v400_env.py` **2 passed**、`python -m compileall -q touhou_rl` 退出码 0。

## 五、只读调查：Windows 侧有没有"按回放文件播放"的入口

**有，且本次修复后实测可用。** 逐项回答：

1. **`tools/rl/frame_fence_capture.py` 只有 demo/practice** —— 是。
   `parser.add_argument("--mode", choices=["demo", "practice"], default="demo")`
   （第 40 行），没有 replay。
2. **v3.x 是否已有 replay 驱动采集** —— 有，入口是
   `tools/rl/v380_fence_probe.py --mode replay`：第 46 行 `--mode` 含 `replay`，
   第 48-51 行 `--replay-file`（默认 `th08_replay/Th8_ud2970.rpy`），
   第 208-213 行执行 `menu.install_replay(run_dir, replay_path, index=1)` +
   `menu.enter_replay(session, replay_index=0, stage=args.replay_stage)`——
   即走**菜单导航**而不是任何环境变量。其它复用 `menu.enter_replay` 的还有
   `touhou_rl/capture_dataset.py`（`capture_replay`，`PROOF_EXACT`）、
   `touhou_rl/exact_capture.py`（`replay=` 配置 + `_wait_replay_running`）、
   `touhou_rl/vision_capture.py`、`touhou_rl/probe35.py`、`tools/probes/*`。
   `tools/rl/v380_replay_catalog.py` 只**离线解码** `.rpy`（无播放），
   `tools/rl/v390_probe.py` 里没有 replay 播放调用。
   **实测**（本次修复后）：

   ```powershell
   python tools\rl\v380_fence_probe.py --mode replay --frames 3 --hold-every 1 `
     --replay-stage 0 --save-png 2 `
     --out .tmp_v400_wp8_practice_replay_out --run-dir .tmp_v400_wp8_practice_replay_run
   ```

   退出码 **0**，`verdict.passed=true`、`accepted=3`、`rejected=0`、`cropFailures=0`、
   `aliveAtFrame=303`、`wallSeconds=11.906`、`skippedFrames=0`；
   `proven-frames.jsonl` 的 `logicFrame` 为 **305、306、307**（逐帧 +1）；
   `labels.jsonl` 三条 `mask=1`、`playerX=376.0`；
   PNG `frames/00000.png` 242,770 B / SHA256
   `110294770257807079A7E99DE527B67D0FC4222CEF931BF56708F4BDC25BB23A`，
   `frames/00001.png` 242,298 B / SHA256
   `A8F61521EBCC38E170DD97A445F02A422CDD9644B88D62EF9C8BF894DBD08CD9`
   （均 640×480 RGB）。
3. **Windows 构建是否支持 `TH08_AUTOPLAY_REPLAY`** —— **不支持**。
   `build_fence/th08.exe` 内 ASCII 与 UTF-16LE 搜索 `AUTOPLAY` / `REPLAY` / `TH08_` 前缀
   均为 **0 命中**；`third_party/th08` 源码树同样 0 命中。该变量只出现在
   `tools/rl/v400_human_reference_autoplay.py`（第 6、443、497 行），服务对象是
   **Linux/WSL harness**；而 `third_party/th08-rl/src/rlharness` 在本检出里是**空目录**
   （0 个文件），`.gitmodules` 只登记了 `third_party/th08`，所以那份 Linux autoplay
   补丁连源码都不在当前检出中。

**对"Windows 与 Linux 播同一份 .rpy 并对齐 logicFrame"的结论**：
Windows 侧只能走菜单驱动的 replay 路线（现已被本次修复打通），两边用同一 `.rpy`
（先比对文件 SHA256）即可；对齐键应当用 **`logicFrame` + replay 文件 SHA + stage 编号**。
不要用栅栏 `ticket`：`ticket` 是 `framefence.py` 描述的**进程内单调帧计数器**
（本次两次运行的首个记录都是 `ticket=766`，各自 `fenceFinal.ticket` 为 767/769），
跨进程不可比。本次未做 Linux 侧实跑（当前检出无可运行的 Linux harness），
故该对齐**尚未闭环**。

## 六、仍未解决项

1. **锁屏/无人交互时无法送输入（环境门，非代码可解）**。本次调查早段本机锁屏，
   表现为 `force_foreground=False`、`SendInput`=`ERROR_ACCESS_DENIED(5)`、
   游戏偶发 `DirectInput/Direct3D` 启动失败对话框。现在代码会 fail-closed 打出
   `input_window_state()` + `foreground_summary()`，但**采集前必须先确认桌面已解锁**。
2. **`--frames 2` 的 logicFrame 是 571→573，不是相邻帧**（`holdBudgetSkips=1`）。
   要相邻帧需把 `--hold-max-us` 提到大于单次读窗成本（本轮 `readerWorkUs≈10.5-12 ms`，
   但 PNG 落盘把整个 hold 拖到 90 ms 上限），未在本卡验证。
3. **Windows 与 Linux 的同 `.rpy` logicFrame 对齐未闭环**：Windows 侧 replay 路线已能跑
   （305/306/307），Linux harness 在本检出不可运行（`third_party/th08-rl` 为空子树）。
4. **`PostMessageKeyboard` 在 Windows 上已无实际作用**：`menu` 会在首次 `tap` 时把它升级成
   `SendInputKeyboard`，因此"后台并行实例无需焦点"这一性质在 Windows 上仍然不成立
   （它在 Windows 上本来也从未成立过）。若将来要为 Windows 找回真正的前台无关输入，
   只能改引擎侧（本任务禁止）。
5. **本次全部实跑均为单实例、串行**；每轮运行前都确认过无残留 `th08.exe`。三条验收运行
   分别为 14.364 s / 14.363 s / 11.906 s，远低于 120 s 上限。（唯一超 120 s 的是
   `.tmp_v400_wp8_practice_fgdiag.py`：它继承 `GameSession.start()` 的"3 次 × 45 s 引导
   重试"设计，在锁屏态下跑满约 140 s 后失败——是诊断脚本，不是采集运行。）
6. **`third_party/th08` 子模块的 `-dirty` 状态是本卡之前就存在的**（HEAD 仍是父仓记录的
   `08c7f00f…`，脏在 `src/framefence.cpp`、`scripts/fence-addresses.py` 等栅栏构建文件）。
   本卡没有改动 `third_party/th08` 的任何文件。

## 七、复现清单（命令 → 退出码 → 证据）

| # | 命令 | 退出码 | 证据 |
|---|---|---|---|
| 1 | `python .tmp_v400_wp8_practice_probe.py` | **0** | 投递无效 + `SendInput` 被拒的原始证据（`.tmp_v400_wp8_practice_probe.log`） |
| 2 | `python .tmp_v400_wp8_practice_queueprobe.py` | **0** | `AttachThreadInput`+`SetKeyboardState` 负结果（`…queueprobe.log`） |
| 3 | `python .tmp_v400_wp8_practice_fgtest.py` | **0** | `SetForegroundWindow` 返回 True 但 `foregroundNow=0` |
| 4 | `python .tmp_v400_wp8_practice_desktop.py` | **0** | 输入桌面 `Default`、`SendInput` 探测、WTS 会话状态 |
| 5 | `python .tmp_v400_wp8_practice_lockstate.py` | **0** | `WTSINFOEX.SessionFlags` |
| 6 | `python .tmp_v400_wp8_practice_menuprobe.py` | **0** | 0.323 s 内 cursor 0→3→0、后端升级（`.tmp_v400_wp8_practice_menuprobe.log`） |
| 7 | `python tools\rl\frame_fence_capture.py --frames 1 --mode practice …` | **0** | §3.2（`.tmp_v400_wp8_practice_f1.log`） |
| 8 | `python tools\rl\frame_fence_capture.py --frames 2 --mode practice …` | **0** | §3.3（`.tmp_v400_wp8_practice_f2.log`） |
| 9 | `python tools\rl\v380_fence_probe.py --mode replay --frames 3 …` | **0** | §五（`.tmp_v400_wp8_practice_replay.log`） |
| 10 | `python -m pytest … -q tests/test_v400_menu_foreground.py` | **0** | 8 passed |
| 11 | `python -m pytest … -q touhou_rl/tests/test_core.py` | **0** | 30 passed |
| 12 | `python -m pytest … -q tests/test_v400_env.py` | **0** | 2 passed |
| 13 | `python -m compileall -q touhou_rl` | **0** | 语法通过 |

> 表中 `.tmp_v400_wp8_practice_*` 脚本是本卡的**诊断证据**，不是运行期依赖；
> 清理它们不影响 `touhou_rl/` 的行为。
