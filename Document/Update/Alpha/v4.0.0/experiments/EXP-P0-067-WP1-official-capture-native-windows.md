# EXP-P0-067-WP1 — 官方零售采集在 Windows native 下的可行性判定

- 日期：2026-11-09（WSL `Linux 6.18.54.1-microsoft-standard-WSL2`）
- Provider / model：DeepSeek / DeepSeek v4.1 flash
- 范围：**只做可行性判断**。未跑 108 份，未安装任何软件，未改依赖，未改 `third_party/th08` 源码/构建，未改零售 EXE，未删除文件，未伪造任何 `rows.jsonl` / `complete.json`。
- 结论先行：**不可行（Windows 原生），且当前 WSL 官方环境下同样不可行。**

---

## 0. 结论摘要

| 问题 | 判定 | 依据 |
|---|---|---|
| 官方工具依赖什么环境 | **Linux 用户态 + Wine + GDB（或 Linux ptrace）+ X 显示 + xdotool** | 无任何 Windows 分支；见 §1 |
| 属于 (a) Windows+CDB/WinDbg 吗 | **否** | 代码里没有 `cdb`/`windbg`/`DebugActiveProcess`/`ntdll` 任何引用 |
| 属于 (b) ptrace / Wine 特有路径吗 | **是（Wine 特定路径）** | `--wine` 为必选语义、`WINEPREFIX/WINEARCH=win32`、`wineboot`、`wineserver`、`gdb --args wine th08.exe` |
| 属于 (c) 需要研究构建/插桩吗 | **否** | `--candidate` 省略即走 `product="retail"`，零售原件直跑 |
| Windows 原生「按官方用法」可行吗 | **不可行** | 第一个失败点在 Wine 之前：`capture-replay.py:198` symlink 特权 |
| 本机有 CDB/WinDbg 吗 | **没有** | 只有 `dbgcore/dbghelp/srcsrv/symsrv` 四个 DLL，无 `cdb.exe` |
| 官方完整环境在哪 | **WSL Ubuntu 22.04，依赖齐备** | gdb 12.1 / wine-6.0.3 / xdotool / Xvfb / gcc / i386 wine 全部存在 |
| 补齐 Python 缺口后单份实跑 | **0 行** | attestation 通过、4 个断点装载、200 s 内 0 命中 |
| RPM 自建零售 trace 现实吗 | **不现实（逐位一致目标）** | 57 h 实时下限 + 帧栅栏/读撕裂两条保真度硬伤 |

---

## 1. 官方工具的环境依赖（附 文件:行）

工具链由 `capture-replay.py` 一个启动器 + 两个 observer（GDB 脚本 / Linux ptrace tracer）组成。

### 1.1 决定性依赖：Wine + GDB

| 依赖 | 位置 | 说明 |
|---|---|---|
| Wine ELF loader（非 shell wrapper） | `capture-replay.py:110` | `--wine` 默认 `/usr/lib/wine/wine`，帮助文本明写 "Wine ELF loader (not the shell wrapper)" |
| Wine 版本必须可执行 | `capture-replay.py:212` | `subprocess.check_output([args.wine, "--version"])` 写进 metadata；Windows 上必抛 `FileNotFoundError` |
| 32 位 Wine 前缀 | `capture-replay.py:243` | `WINEARCH=win32`、`WINEPREFIX=<output>/wine` |
| Wine 音频后端屏蔽 | `capture-replay.py:246` | `WINEDLLOVERRIDES=winepulse.drv,...=d` |
| 前缀初始化 | `capture-replay.py:278` | `subprocess.run(["wineboot","-u"])` |
| Wine 注册表写入 | `capture-replay.py:281-283` | `wine reg add HKCU\Software\Wine\Drivers /v Audio` |
| **GDB 驱动 Wine 启动 th08.exe** | `capture-replay.py:299-301` | `Popen(["gdb","-q","--interpreter=mi2","--args", args.wine, "<runtime>/th08.exe"])` |
| GDB/MI 会话配置 | `capture-replay.py:328-339` | `mi-async on`、`disable-randomization off`、`debuginfod off`、`handle SIGUSR1/2 nostop noprint pass` |
| 启动 | `capture-replay.py:340` | `-exec-run` |
| **映射字节 attestation** | `capture-replay.py:343-357` | 轮询 `-data-read-memory-bytes <attestAddress> 13`，比对 `attestBytes`；注释明写 "Wine execs its loader during startup. Install the hardware breakpoint after that exec so the debug registers survive." |
| 装载 observer | `capture-replay.py:359` | `source <replay-external-observer.py>`（GDB Python 脚本） |
| 继续执行 | `capture-replay.py:362` | `-exec-continue` |
| 收尾 | `capture-replay.py:394` | `wineserver -k` / `wineserver -w` |

### 1.2 决定性依赖：X 显示 + xdotool（菜单输入）

| 依赖 | 位置 | 说明 |
|---|---|---|
| X 显示（私有 Xvfb） | `capture-replay.py:249-274` | 用 `socket.AF_UNIX` 探测 `\0/tmp/.X11-unix/X<n>`（抽象套接字，Windows 无此语义），再 `Popen(["Xvfb", ...])` |
| Xvfb 就绪判定 | `capture-replay.py:269-272` | 仅 `time.sleep(.5)` 后 `poll() is None` 即认作可用 |
| 菜单按键 | `replay-external-observer.py:110,141,172` | `subprocess.run(["xdotool","keydown"/"keyup", ...])` |
| 输入通道性质 | `replay-external-observer.py:3` | docstring：`Menu input uses XTest key events on the capture's isolated X display.` |
| `DISPLAY` 透传 | `capture-replay.py:251-252,271` | `--display` 或私有 Xvfb 编号 |

### 1.3 决定性依赖：GDB Python API 与硬件执行断点

| 依赖 | 位置 | 说明 |
|---|---|---|
| `import gdb` | `replay-external-observer.py:5` | 必须跑在 GDB 内嵌解释器里，普通 CPython 无法导入 |
| 内存读取 | `replay-external-observer.py:28` | `gdb.selected_inferior().read_memory(address, size)` |
| **硬件执行断点** | `replay-external-observer.py:54-56` | `gdb.Breakpoint(f"*0x{boundary:08X}", gdb.BP_HARDWARE_BREAKPOINT)` |
| 断点数量上限 | `capture-replay.py:170` | 注释明写 "x86 has four hardware breakpoint slots"；observer 用满 4 个（boundary/completion/stageStart/delete，`:54,:239,:252,:264`） |
| 寄存器级帧内读取 | `replay-external-observer.py:283-284,326,342-343` | `gdb.parse_and_eval("$esp"/"$ebp"/"$eip")` |
| ptrace 备选后端 | `capture-replay.py:131,284-294` | `--observer-backend ptrace` → `replay-ptrace.py`；其 docstring 明写 "Linux x86-64 tracer ... process_vm_readv" |

### 1.4 依赖：Python ≥ 3.11

| 依赖 | 位置 | 说明 |
|---|---|---|
| `hashlib.file_digest` | `capture-replay.py:90` | `hashlib.file_digest(stream, "sha256")`，该 API 自 **Python 3.11** 起才有 |
| 纯 stdlib 模块 | `pe_image.py` / `replay_file.py` | 只 import `hashlib`/`struct`/`pathlib`/`dataclasses`，无第三方包要求 |

### 1.5 判定

> 官方工具**没有任何 Windows 代码路径**。`platform` 只在 `capture-replay.py:211` 用于写 metadata 字符串，
> 不产生分支。`--observer-backend` 的合法取值只有 `("gdb","ptrace")`（`:131`）。
>
> 因此归类为 **(b) 需要 Wine 特有路径**：本质是一条 **Linux 用户态 + Wine32 + GDB 硬件断点 + Xvfb + XTest** 的管线。
> 它**不需要**研究构建或插桩（`--candidate` 省略即零售原件），也**不依赖** Windows 调试器。

补充一条容易误判的结构性事实：**菜单导航本身挂在断点上**。
`replay-external-observer.py:146-152` 中 `menu()` 只在 `stop(completion=True)` 且 `self.stage is None` 时被调用，
即依赖 `completionBoundary`（`0x00441F52`，位于 `GameWindow::Render` 内）先命中。
**断点不命中 ⇒ 不发键 ⇒ 游戏永远进不到 Replay 菜单 ⇒ 断点更不会命中**，这是一个自锁环。

---

## 2. 本机依赖盘点（只读，未安装）

| 依赖 | 状态 | 证据 |
|---|---|---|
| `cdb` / `windbg` 在 PATH | **缺失** | `where.exe cdb` / `where.exe windbg` 均 "Could not find files"，退出码 1 |
| Windows Kits 调试器 | **仅存 4 个 DLL** | `C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\` = `dbgcore.dll`、`dbghelp.dll`、`srcsrv.dll`、`symsrv.dll`（共 5 项含目录，**无 cdb.exe**） |
| `C:\Program Files*` 深度 4 内 `cdb/windbg/ntsd/kd.exe` | **0 命中** | 递归 `-Include` 搜索返回空 |
| WinDbg (Store/Appx) | **未安装** | `Get-AppxPackage -Name '*WinDbg*'` 空 |
| `gdb`（Windows 侧） | **缺失** | `where.exe gdb` 未找到 |
| `gdb`（WSL 侧） | **存在 12.1** | `GNU gdb (Ubuntu 12.1-0ubuntu1~22.04.2) 12.1` |
| `wine`（WSL 侧） | **存在 wine-6.0.3** | `/usr/bin/wine`、`/usr/lib/wine/wine` 均在；`/usr/lib/i386-linux-gnu/wine` 存在（32 位支持齐） |
| `xdotool` / `Xvfb` / `gcc`（WSL） | **存在** | `/usr/bin/xdotool`、`/usr/bin/Xvfb`、`/usr/bin/gcc` |
| Python（Windows） | **3.12.10** | 有 `hashlib.file_digest`；`pefile`/`PIL`/`numpy` 可导入 |
| Python（WSL） | **3.10.12，缺 `file_digest`** | Ubuntu 22.04.5 只有 `/usr/bin/python3.10`；`python3.11/3.12/3.13` 全无；无 venv、无 conda/pyenv |
| WSL 发行版 | **Ubuntu 22.04.5 LTS，Running** | `wsl -l -v` |

### 2.1 缺失项清单（仅记录，未安装）

1. **Windows 原生 x86 硬件断点调试器**：`cdb.exe` / `windbg.exe` 缺失。这是 Windows 原生路线唯一的正解工具，装它属于「安装软件」，本轮禁止。
2. **WSL 的 Python ≥ 3.11**：`capture-replay.py:90` 需要 `hashlib.file_digest`。Ubuntu 22.04 自带 3.10.12，安装 3.11 属于「安装软件」，本轮禁止。
   - 为在不安装的前提下取得「Wine 路径到底能不能跑」这一决定性信息，本轮用一个**纯 stdlib、仅经 `PYTHONPATH` 载入、明确披露**的兼容垫片补上了这一个 API：
     `.tmp_v400_official_native_shim/sitecustomize.py`。
     它不安装任何包、不改依赖版本、不触碰 `third_party/th08`，并且已用**零售原件的正确 SHA-256 自校验通过**（见 §3.2）。**该垫片是探测手段，不是官方环境的一部分。**

---

## 3. 最小尝试：命令、退出码、产物行数

三组实验，总计约 4.5 分钟机时（单份回放，未跑 108 份）。

选取的回放：`third_party/th08/tests/fixtures/replays/th08-1.00d/accepted/th8_ud2cd2.rpy`
（在 148 个本地 `.rpy` 中 `max(inputRecords)=22,292` 为最小，用于把单次运行压进预算；它是**探测样本**，不是 108 fixture 之一。）

公共参数：零售原件 `third_party/th08/resources/th08.exe`（840,704 B，SHA-256 `330fbdbf…d924`，脚本 `:176` 强校验通过）、
游戏数据 `third_party/th08/build/play_fence/th08.dat` / `thbgm.dat`。

### 3.1 Windows 原生（Python 3.12.10，原生 exe，无 Wine）

```powershell
python third_party\th08\scripts\capture-replay.py `
  --target third_party\th08\resources\th08.exe `
  --game-data third_party\th08\build\play_fence\th08.dat `
  --bgm-data  third_party\th08\build\play_fence\thbgm.dat `
  --output-dir .tmp_v400_official_native_win_a `
  --replay third_party\th08\tests\fixtures\replays\th08-1.00d\accepted\th8_ud2cd2.rpy
```

**退出码 1。首个错误：**

```
File "third_party\th08\scripts\capture-replay.py", line 198, in main
    (runtime / name).symlink_to(data)
OSError: [WinError 1314] 客户端没有所需的特权:
  '...\third_party\th08\build\play_fence\th08.dat' -> '...\.tmp_v400_official_native_win_a\runtime\th08.dat'
```

- 产物：2 个文件（`runtime\th08.exe` 840,704 B + `runtime\replay\th8_01.rpy` 27,310 B）。
- `rows.jsonl` **不存在**，`complete.json` **不存在**，连 `metadata.json` 都未生成（失败点在 `:178` 目录创建之后、`:229` 写 metadata 之前）。
- 说明：失败在 **Wine 之前**——即使补上 `SeCreateSymbolicLinkPrivilege`（开发者模式/管理员），下一步 `:212` 仍会因无 `wine` 抛 `FileNotFoundError`，再下一步 `:266` 无 `Xvfb`、`:299` 无 `gdb`。

### 3.2 WSL 官方环境，未加垫片（证明 Python 缺口真实存在）

```powershell
wsl -d Ubuntu -- bash -lc 'python3 /mnt/d/WorkShop/Touhou_RL/third_party/th08/scripts/capture-replay.py ...'
```

**退出码 1。首个错误：**

```
File ".../capture-replay.py", line 176, in main
    if args.target.stat().st_size != 840704 or sha(args.target) != TARGET_SHA:
File ".../capture-replay.py", line 90, in sha
    return hashlib.file_digest(stream, "sha256").hexdigest()
AttributeError: module 'hashlib' has no attribute 'file_digest'
```

- 产物：**0 个**（连输出目录都未创建）。

垫片自校验（证明垫片给出的是正确哈希，不是绕过校验）：

```
python: 3.10.12
file_digest present: True
sha: 330fbdbf58a710829d65277b4f312cfbb38d5448b3df523e79350b879213d924
matches TARGET_SHA: True
```

### 3.3 WSL 官方环境，gdb 后端，单份回放有界实跑（决定性实验）

```bash
# .tmp_v400_official_native_shim/probe2.sh
export PYTHONPATH=/mnt/d/WorkShop/Touhou_RL/.tmp_v400_official_native_shim
export PYTHONDONTWRITEBYTECODE=1                      # 防止写 third_party/th08/scripts/__pycache__
timeout 300 python3 third_party/th08/scripts/capture-replay.py \
  --target third_party/th08/resources/th08.exe \
  --game-data third_party/th08/build/play_fence/th08.dat \
  --bgm-data  third_party/th08/build/play_fence/thbgm.dat \
  --output-dir .tmp_v400_official_native_wsl_b \
  --replay third_party/th08/tests/fixtures/replays/th08-1.00d/accepted/th8_ud2cd2.rpy \
  --timeout 200 --keep-runtime
```

**结果：退出码 1，耗时 261 s。**

```
Captured 0 frames
File ".../capture-replay.py", line 380, in main
    raise RuntimeError("Incomplete capture; inspect gdb.log and rows.jsonl")
RuntimeError: Incomplete capture; inspect gdb.log and rows.jsonl
```

**产物行数：**

| 产物 | 大小 / 状态 |
|---|---|
| `rows.jsonl` | **0 字节（0 行）** |
| `menu.jsonl` | **0 字节（0 行）** |
| `rng-events.jsonl` / `score-events.jsonl` / `item-events.jsonl` / `effect-events.jsonl` | 均为 **0 字节** |
| `complete.json` | **缺失** |
| `game.log` / `runtime/log.txt` | **均未生成**（零售版连启动日志都没写出来） |
| `gdb.log` | 12,458 B |
| `metadata.json` | 3,581 B |
| `xvfb.log` | 264 B |

**环境与原件全部对得上（`metadata.json`）：**

```json
product      = "retail"
environment  = {"python":"3.10.12","host":"Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.35",
                "wine":"wine-6.0.3 (Ubuntu 6.0.3~repack-1)","observerBackend":"gdb"}
observer     = {"type":"GDB hardware breakpoints","frame":"0x00452490","completion":"0x00441F52"}
executableSha256 = 330fbdbf58a710829d65277b4f312cfbb38d5448b3df523e79350b879213d924
gameDataSha256   = 9d7edf43b8ddd347cbb641836f6b5050745dd936f688daebbf9382ca557043bb
bgmDataSha256    = 2c2ef05f9ff6f43f752dae5da519a2477372c3f8875b7b44996a8f69fdad4d89
```

**`gdb.log` 里的关键事实（说明「不是断点没装上」）：**

1. attestation **通过**：
   `10^done,memory=[{begin="0x00452490",...,contents="558bec51894dfca1b4d06401c1"}]`
   —— 与 `observer.json` 的 `attestBytes` 完全一致，说明零售 `.text` 已映射在 `0x00452490`。
2. **4 个硬件执行断点全部创建成功**：
   `=breakpoint-created,...addr="0x00452490"`（frame）、`"0x00441f52"`（completion）、`"0x00439bc7"`（stageStart）、`"0x00453080"`（delete），`times="0"`。
3. `-exec-continue` 之后 200 s 内**没有任何 `*stopped` 于这些地址**；
   唯一一次停点是收尾的 `-exec-interrupt --all`（`stop()` 里对 `0xf7f57569 in ?? ()` 的 SIGINT），随后 `[Inferior 1 (process 2710) killed]`。

**`xvfb.log`（可疑首因，未在本轮证实）：**

```
_XSERVTransmkdir: Mode of /tmp/.X11-unix should be set to 1777
_XSERVTransSocketCreateListener: failed to bind listener
_XSERVTransSocketUNIXCreateListener: ...SocketCreateListener() failed
_XSERVTransMakeAllCOTSServerListeners: failed to create listener for unix
```

私有 Xvfb 的 unix 监听器绑定失败；而启动器在 `:269-272` 只等 0.5 s 就认定显示可用，
因此游戏可能是在一个**已经死掉的 DISPLAY** 上启动的。

### 3.4 与历史三次的关系（重要）

EXP-P0-014 / 037 / 045 的 0 行结论来自 **ptrace 后端**；本次是**默认 gdb 后端**，得到**同样的 0 行**。
因此「后端选错」已被排除。共同的可归因事实是：

> 零售 TH08 在 Wine 6.0.3 下**从未跑进渲染/计算循环**（`GameWindow::Render` 区域的 completion 断点 0 命中、连 `log.txt` 都没写出），
> 所以 frame boundary 与 completion 都不可能命中。

这是 **Wine 运行时层面的阻塞**，不是调试器层面的阻塞。至于它究竟卡在 X 初始化、D3D、字体还是音频，本轮预算内**未定位**，不作结论。

---

## 4. Windows 原生是否可行

**结论：不可行。** 分三层，任何一层都足以否决：

1. **工具层（决定性）**：官方采集器是 Linux+Wine+GDB 管线，`:110/:212/:243/:266/:278/:299/:394` 全部写死 Wine/X11/GDB 语义；
   代码中不存在 Windows 分支，(a)(c) 均不成立。要在 Windows 原生复现，等价于**重写 observer**，不是「换个参数」。
2. **权限/环境层**：即便只跑启动器，`capture-replay.py:198` 已因 `WinError 1314` 失败（symlink 特权）。
3. **调试器层**：本机**没有** CDB/WinDbg（§2），而 Windows 原生路线的正解恰恰就是「x86 硬件执行断点调试器」。
   零售版需要 4 个槽位（`:170` 明确 x86 只有 4 个），这与 CDB 的 `ba e` 能力对得上——但工具不在。

### 4.1 解阻所需的具体前置（三选一）

| 路径 | 缺什么 | 是否本轮可做 |
|---|---|---|
| A. Windows 原生调试器 | 安装 CDB/WinDbg（Windows SDK Debugging Tools / WinDbg Store 版）；并**新写一个 CDB 版 observer**（GDB Python 脚本不能直接用） | **否**（禁止安装；且需新增大量代码） |
| B. 修好 Wine 运行时 | 让零售 TH08 在 Wine 下真正进入渲染循环：排查 Xvfb 就绪判定（`:269-272` 只等 0.5 s）、改用 `--display :0`（WSLg 已有 `X0`）、或升级 Wine 到能跑该游戏的版本 | 部分是（`--display :0` 是一次低成本单变量实验） |
| C. 上游参考 trace | 需要有可信来源、含完整 `rows.jsonl` + `complete.json` 且能校验 manifest `retailTrace.sha256` 的产物 | **否**（EXP-P0-045 已证本地无此物） |

> 注意 B 的一个反面证据：EXP-P0-037 那次用的就是 `--display :0`，同样是 0 行。
> 所以「Xvfb 绑定失败」是**可疑首因但不足以解释全部历史失败**，不能把它当作唯一根因。

---

## 5. 替代路评估：用 `ReadProcessMemory` 自己重建零售 trace

### 5.1 体积与时间

- 总量 = **12,311,460** frame（= 108 fixture 的 `retailTrace.calculationFrames` 之和，与任务给出的 12,311,460 **完全吻合**；另 `sum(inputRecords)=12,224,700`）。
- 零售逻辑帧率 60 fps，**实时下限** = 12,311,460 ÷ 60 = **205,191 s = 3,419.85 min = 56.998 h ≈ 57.0 小时 ≈ 2.37 天**连续不断。
- 再加 108 次菜单导航（每次约 30 s）≈ +0.9 h ⇒ **合计约 58 小时**。
- **无法用快进压缩**：`--playback-mode 2`（零售内建快进）会把帧率推离 60 fps，轮询式 RPM 观测无法锁帧，只能被迫实时。

### 5.2 吞吐不是瓶颈

官方 observer 每帧只做约 8 次小读取、共约 600 B（`replay-external-observer.py:157,197-208`：`g_GameManager+0x3dba8` 0x220 B、
`GM+8` 指向的 0xa0 B、`player+0x2b4` 12 B、`rng` 8 B、`input` 2 B、`GM+0x30` 4 B 等），
单帧 16.67 ms 预算内轻松完成（µs 量级）。所以**性能不是替代路的障碍**。

### 5.3 保真度是硬伤（三条，均不可工程绕开）

1. **没有帧栅栏**：官方在硬件执行断点**停住**的时刻取数（`replay-external-observer.py:28,201-208`），
   天然保证「每个 calculation frame 恰好一行、且帧内一致」。RPM 轮询没有这个栅栏，
   漏一帧或重复一帧都会改变 `rows` 计数，而该计数必须等于 fixture 的 `calculationFrames`（对照 `complete.json` 的 `rows`），
   于是 SHA **必错**。
2. **读撕裂**：25 个字段分布在 6 个内存区（`GameManager` 全局指针、`g_Player`、`g_Rng`、`input`、`g_ReplayManager`、`GM+0x3dba8`）。
   断点停住读是原子的；对运行中进程读会拿到「从未同时存在」的组合（例如 `score` 已更新而 `lives` 尚未），
   这类值官方产线**永远不会产生**，SHA **必错**。
3. **失败代价高**：单份 fixture 按 44k–150k 帧算，一份要 12–42 min；任何一份不一致就整份重跑，且 108 份串行。

### 5.4 判定

> **「用 `ReadProcessMemory` 自己重建零售 trace」在「与零售版逐位一致」这个目标上不可行。**
>
> 它只能支撑**近似/像素配对**用途——这正是 `tools/rl/v400_retail_win_capture.py` 已经在做的事
> （PNG 帧 + `gameplayFrameCounter` 同帧证明 + `windows-retail-manifest.json`，`source="windows-retail"`），
> 那条线可以服务像素级对照，但**不能**产出 `retailTrace`，不能替代 G0.1 的参考轨迹。
>
> 若只问「57 小时现实吗」：**不现实**——2.37 天不间断、且这 2.37 天换来的还是**不受信**的轨迹（§5.3 第 1、2 条）。
> 反过来说，如果把目标降级为「近似轨迹 / 帧级像素配对」，则不需要 57 h，也不必按 60 fps 全量跑，按需取样即可。

---

## 6. 未验证边界（明确不主张）

- 未定位零售 TH08 在 Wine 6.0.3 下究竟卡在哪一步（无 `log.txt`、无 Wine 崩溃日志、无 `*stopped` 于游戏代码）。
- 未验证 `--display :0`（WSLg 已有 `/tmp/.X11-unix/X0`）是否能解救本次 0 行；这是下一轮最便宜的单变量实验。
- 未验证其它 Wine 版本、未验证 32 位字体/D3D 依赖。
- 未运行任何 108 fixture 之一（探测样本 `th8_ud2cd2.rpy` 只是本地最短回放）。
- 未产生、也未声称产生任何 `rows.jsonl` / `complete.json` 有效内容；两份输出目录中这两个文件分别为「不存在」与「0 字节」。
- 兼容垫片是**探测手段**，官方环境本身仍缺 Python ≥ 3.11。

## 7. 合规声明

- 未安装任何软件，未改动任何依赖版本，未新增 Python 包。
- `third_party/th08` 工作区未被本轮修改：`scripts/capture-replay.py`、`scripts/replay-external-observer.py`、
  `scripts/replay_file.py`、`scripts/pe_image.py`、`resources/th08.exe` 均为未修改状态；
  已用 `PYTHONDONTWRITEBYTECODE=1` 阻止写入 `scripts/__pycache__`。
  （子模块内 `src/main.cpp`、`scripts/build.py` 等既有改动是**此前工作留下的**，不是本轮动作。）
- 未删除任何文件。本轮新增仅限 `.tmp_v400_official_native*` 与本文档。
- 未伪造 rows/complete：`rows.jsonl` 0 字节即为真实结果，原样保留。

## 8. 产物清单

| 路径 | 内容 |
|---|---|
| `.tmp_v400_official_native_win_a\` | Windows 原生尝试：`runtime\th08.exe`(840,704 B)、`runtime\replay\th8_01.rpy`(27,310 B)；**无** rows/complete/metadata |
| `.tmp_v400_official_native_win_a.log` | PowerShell 转写（含 `WinError 1314` 完整 traceback） |
| `.tmp_v400_official_native_wsl_b\` | WSL gdb 后端尝试：`gdb.log`(12,458 B)、`metadata.json`(3,581 B)、`observer.json`(2,594 B)、`xvfb.log`(264 B)、`rows.jsonl`(**0 B**)、`menu.jsonl`(**0 B**)、4 个事件流(0 B)、`runtime\`（`--keep-runtime`） |
| `.tmp_v400_official_native_wsl_b.log` | WSL 输出转写（含 `Captured 0 frames` 与 `Incomplete capture`） |
| `.tmp_v400_official_native_shim\sitecustomize.py` | 已披露的 `hashlib.file_digest` 纯 stdlib 垫片（仅 `PYTHONPATH` 载入） |
| `.tmp_v400_official_native_shim\probe1.sh` / `probe2.sh` | 两个探测脚本（可复现 §3.2 / §3.3） |

## 9. 下一轮建议（按性价比排序）

1. **单变量**：同一命令加 `--display :0`，仍 `--timeout 200`。若 rows 仍 0，则 Xvfb 假设被排除，锁定为「零售在 Wine 下进不了渲染循环」。
2. 若确认是 Wine 运行时问题：在**不改源码/不装软件**的前提下，检查 `wine` 的 stderr 与 `WINEDEBUG=+relay` 下的最后调用，定位阻塞点。
3. 若确定要走 Windows 原生：需人工授权安装 CDB/WinDbg，并**新写**一个 CDB 版 observer（4 个 `ba e` 硬件执行断点 + `dd` 读内存 + `SendInput` 发键）。这是唯一能产 `retailTrace` 的 Windows 原生路线。
4. 在此之前，G0.1 的「108 份逐位一致」保持 **BLOCKED**，不得用 `.rpy` 解码、近似轨迹或研究构建冒充。
