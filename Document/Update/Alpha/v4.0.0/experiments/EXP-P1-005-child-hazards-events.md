# EXP-P1-005 子进程只读快照与 `EVENTS` 事件位

- 日期 / 工作包：2026-11-09 / P1「fork 子进程只读命令」+「`frameEventFlags` 事件导出」两个硬缺口，以及 `touhou_rl.v400.teacher` 包导入回归
- 假设：
  1. `CHILD <handle> HAZARDS/PLAYER/BULLETS/LASERS/ENEMIES` 可以在**真实 fork** 上回答，语义与父进程一致，且不推进任何一方的帧；
  2. 新增只读 `EVENTS` 能从 `g_ReplayManager->frameEventFlags` 读出逐帧事件位，并在真实对局中观测到 `REPLAY_FRAME_EVENT_PLAYER_HIT`（0x2）；
  3. 以上扩展**不改变** 25 字段 `HASH`：与 `build/rl` 的前 100 帧逐帧相同；
  4. `touhou_rl.v400.teacher` 在干净解释器里可导入，`__all__` 的每个名字都可访问，速度表只有一个定义。
- 结论：**全部成立**。
  - `HASH` 前 100 帧：`build/rl` 与 `build/rl_hazard2` **100/100 相同**（三组序列 sha256 均为 `6407d535…`）；每帧额外插入 `HAZARDS 16`+`PLAYER`+`EVENTS` 也不改变序列。
  - `CHILD 1 HAZARDS`：真实 fork 返回 `OK`，与父进程**逐字节相同**（frame 120，213 B）；大回复 5,208 B（frame 1432）同样逐字节相同。
  - `EVENTS`：不动策略第 **678** 次观测抓到 `frameEventFlags=0602`（`PLAYER_HIT`=0x2 | `PLAYER_DYING_STARTED`=0x200 | `EFFECT_SPAWNED`=0x400）；复位前明确回 `ERR events-unavailable`，不返回 0。
  - `build/rl`（sha256 `591c521d…`）与 `build/rl_hazard`（sha256 `8053dfaa…`）**构建前后 SHA 均未变**；新构建只落在 `build/rl_hazard2`（sha256 `f172399f…`）。
  - teacher 导入回归：`tests/test_v400_teacher_import.py` 8 项全过；`tests/test_v400_teacher.py` 20 项全过。

---

## 1. 任务 A：`touhou_rl.v400.teacher` 包导入

### 1.1 根因与接手时的实际状态

任务书描述的断点是：`hazard_tape.py` 写了

```python
from touhou_rl.v400.teacher.geometry import SpeedTable, bullet_boxes, enemy_boxes
```

而 `geometry.py` 里没有 `SpeedTable`（它只在 `motion.py`），于是 `import touhou_rl.v400.teacher` 直接 `ImportError`。

**接手时工作树里该行人已被改成 `from touhou_rl.v400.teacher.motion import SpeedTable`，裸导入已能通过**（实测 `OK 17`）。也就是说：断点本身被另一路先行绕过，但“速度表到底定义在哪”仍与任务书要求（`geometry.py`）不一致，而且没有回归测试锁住这件事。本次按任务书把**唯一来源**落在 `geometry.py`，并补测试；若回到任务书描述的原始状态，新测试会立刻失败（`geometry` 没有 `SpeedTable`）。

### 1.2 修复

| 文件 | 改动 |
|---|---|
| `touhou_rl/v400/teacher/geometry.py` | 新增**唯一**的 `SpeedTable` 定义：四个字段 `normal_axial/normal_diagonal/focused_axial/focused_diagonal`；`from_player()` 从 `PLAYER` 快照读取四个速度（缺字段抛 `ValueError`，不猜 0）；`axial`/`diagonal` 属性作为未聚焦速度的短名；保留 `check()`/`as_tuple()`；`PLAYER_SPEED_FIELDS` 给出线上字段名映射 |
| `touhou_rl/v400/teacher/motion.py` | 删掉重复的 dataclass，改为 `from touhou_rl.v400.teacher.geometry import SpeedTable` 重导出；`step_delta`/`split_cand` 不动。旧导入路径 `motion.SpeedTable` 继续可用，且与 `geometry.SpeedTable` 是同一个对象 |
| `touhou_rl/v400/teacher/hazard_tape.py` | 恢复任务书要求的导入方向：`from ...geometry import SpeedTable, bullet_boxes, enemy_boxes` |
| `tests/test_v400_teacher_import.py` | 新增顶层导入测试（8 项） |

`tests/test_v400_teacher_import.py` 断言：

1. `import touhou_rl.v400.teacher` 成功，`__all__` 非空、无重复、每个名字 `hasattr` 且非 `None`；
2. `geometry.SpeedTable is motion.SpeedTable is hazard_tape.SpeedTable`（单一来源，防再次分叉）；
3. 可从实测值构造、可从 `PLAYER` 形状对象构造；
4. 缺一个速度时 `from_player` 抛错（不补 0）；
5. `check()` 拒绝非正速度；
6. 换一个工作目录启动全新解释器，仍能导入且 `__all__` 数量一致。

### 1.3 结果

```text
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q \
    tests/test_v400_teacher_import.py tests/test_v400_teacher.py
28 passed in 0.93s
python -m compileall -q touhou_rl/v400/teacher   # exit 0
```

其中 `test_v400_teacher_import.py` 8 项、原有 `test_v400_teacher.py` 20 项。

---

## 2. 引擎改动（只改 `rlharness`，不碰游戏逻辑/25 字段/依赖）

### 2.1 新只读命令 `EVENTS`

`src/rlharness/rlharness.cpp:626-652`（`HandleEvents`），派发在 `:1280`：

```cpp
void HandleEvents()
{
    if (g_ReplayManager == NULL)
    {
        Reply("ERR events-unavailable g_replaymanager=0");
        return;
    }
    Reply("OK frames=%llu frameEventFlags=%04x",
          static_cast<unsigned long long>(FrameCount()),
          static_cast<unsigned>(g_ReplayManager->frameEventFlags));
}
```

来源与生效条件（均在仓库源码里核对）：

- 字段：`ReplayManager::frameEventFlags`（`ReplayManager.hpp:213`），`u16`；`REPLAY_FRAME_EVENT_PLAYER_HIT = 0x0002`（`ReplayManager.hpp:161-172`）。
- 置位：`Player.cpp:329/356/466` 等直接 `g_ReplayManager->frameEventFlags |= REPLAY_FRAME_EVENT_PLAYER_HIT`。
- 清零：`ReplayManager::CaptureFrameSyncState` 每帧开头 `frameEventFlags = NONE`（`ReplayManager.cpp:297`）；该回调在 **record 模式也注册**（`ReplayManager.cpp:240-243`，优先级 `CHAIN_PRIO_CALC_REPLAYMANAGER_LOW_PRIO=7`，早于 Player=9 / BulletManager=14），因此正常对局可用，`STEP` 之后读到的就是**刚推进那一帧**的事件位。
- 取不到时：`g_ReplayManager == NULL`（复位前、replay 链被删除后）明确回 `ERR events-unavailable`，**不返回 0 冒充“没有事件”**。
- **不进入 25 字段**：`TrackedFields` / `HASH` / `STATE` 一行未改；`EVENTS` 只读、不推进、不抽 RNG。

实测（见 §3.3）：复位前 `ERR events-unavailable g_replaymanager=0`；复位后 `OK frames=0 frameEventFlags=1400`。

### 2.2 子进程只读快照

父进程侧 `HandleChild` 原先只放行 `HELLO/HASH/QUIT/DEBUG/FRAME/PING/STATE` 与 `STEP`，其余一律 `ERR child-command`，所以 `CHILD <h> HAZARDS` 被拒。改动全部在 `src/rlharness/fork.cpp`：

| 位置 | 改动 |
|---|---|
| `fork.cpp:163` | 新增 `ForwardLine(from,to)`：**流式**转发一整行（8 KiB 分块，遇 `\n` 停止），不再用 4 KiB 行缓冲。`HAZARDS` 单行容量 256 KiB，固定缓冲会把真实回复截断成 `ERR child-read` |
| `fork.cpp:241` | `TemplateLoop` 把子进程回复改为 `ForwardLine(child.output, control)` |
| `fork.cpp:352` | 新增 `ChildReadOnlyCommand`：放行 `HELLO/HASH/QUIT/DEBUG/FRAME/PING/STATE/PLAYER/EVENTS/HAZARDS` 以及带参数的 `BULLETS `/`LASERS `/`ENEMIES `/`HAZARDS `；参数由子进程自己校验，非法时回 `ERR <cmd>-usage`（显式拒绝，不静默） |
| `fork.cpp:368` | `HandleChild` 用上述白名单取代“只允许 STEP”；`RESET` 等仍回 `ERR child-command`；回复经 `ForwardLine` 流式写回调用方 |

子进程 `g_resetReady` 的可用性论证：`PARK` 从**已经跑起 stage 的父进程** fork 出 template（`Ready()` 要求 `GAMEPLAY_SETUP_COMPLETE` 且 `globals != NULL`），`SPAWN` 再从 template fork 出子进程，因此 template/子进程**继承父进程 `RESET` 后的 `g_resetReady=1` 与 `globals`**；`Run()` 在子进程里重新进入命令循环，不重新复位。实测（§3.2）`CHILD HAZARDS/PLAYER/HASH/STATE` 全部 `OK`，且与父进程逐字节相同。若未来出现“PARK 早于 RESET”的调用顺序，子进程会在 `HandleHazards` 现有分支明确回 `ERR hazards-not-ready reset-required`，不会静默失败。

### 2.3 新构建目录

- 目录：`/root/Touhou_RL/third_party/th08-rl/build/rl_hazard2`（**新建，从未写入 `build/rl` / `build/rl_hazard`**）
- 配置参数（与 `build/rl` 的 CMakeCache 关键项逐项相同）：

```text
CMAKE_BUILD_TYPE=Debug
TH08_RL_HARNESS=ON
TH08_ENABLE_REPLAY_TRACE=OFF
TH08_DX8_SDK_ROOT=/root/Touhou_RL/third_party/th08-rl/scripts/prefix/mssdk
生成器：Ninja，编译器 /usr/bin/c++（x86-64）
```

- 退出码：`CMAKE_EXIT=0`；首次构建日志 `[55/55] Linking CXX executable th08-modern`，复核 `NINJA_EXIT=0`（`ninja: no work to do.`）
- 产物：`build/rl_hazard2/th08-modern`，sha256 `f172399f61b75baa8e62be46041ef756a710759f439e7fb07111f4747c78264e`

> `build/rl`：`591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb`（未变）
> `build/rl_hazard`：`8053dfaa3d5cde0f720ba31d7f63aeba1db929a6721999c1b95e8ac00ba7b32a`（未变）
> 构建前记录的对照文件：`.tmp_v400_child_sha_before.txt`；构建/运行后复核一致。

---

## 3. 验证

所有真机验证均为：`--rl-role shadow`（无窗口/null-device、无 `DISPLAY`）、`SDL_AUDIODRIVER=dummy`、`LIBGL_ALWAYS_SOFTWARE=1`、`GALLIUM_DRIVER=llvmpipe`、`LP_NUM_THREADS=1`，数据目录 `/mnt/d/WorkShop/Touhou_RL/.tmp/calibration`；引擎逐个启动、关闭（WSL 内并发 ≤ 1）。探针脚本：`.tmp_v400_child_probe.py`、`.tmp_v400_child_probe_big.py`；原始证据：`.tmp_v400_child_evidence.json`、`.tmp_v400_child_big.json`。

### 3.1 `HASH` 前 100 帧与 `build/rl` 一致

命令形状：`RESET 0 1 0 1`，然后 100 次 `STEP 0 1` + `HASH`，取每帧 `hash=`。

| 序列 | 二进制 | 交互 | 帧数 | 序列 sha256 | 首帧 | 末帧 |
|---|---|---|---|---|---|---|
| A（基准） | `build/rl` | `STEP`+`HASH` | 100 | `6407d535c523a8378c2976684a8444d6e902afc01ddb1f3e6f118b19e99372d2` | `f1df042d9373e879` | `f356a844ad8ae613` |
| B（扩展后） | `build/rl_hazard2` | `STEP`+`HASH` | 100 | 同上 | 同上 | 同上 |
| C（扩展后+只读命令） | `build/rl_hazard2` | 每帧 `STEP`+`HAZARDS 16`+`PLAYER`+`EVENTS`+`HASH` | 100 | 同上 | 同上 | 同上 |

- A vs B：**100/100 相同**；A vs C：**100/100 相同**（同时证明 `HAZARDS`/`PLAYER`/`EVENTS` 是只读的，不会扰动确定性）。
- 两个二进制各自的 `RESET` 回复 hash 也相同：`hash=7f04c7361749ed9f`（`setup_frames` 373 vs 388 不同是加载耗时路径，帧计数在 `RESET` 末尾归零，不影响状态）。
- `build/rl` 是扩展前的旧二进制：它不认识 `EVENTS`/`HAZARDS`（`ERR unknown-command`），所以只用作 HASH 基准。

### 3.2 `CHILD <h>` 只读快照在真实 fork 上可用

`RESET 0 1 0 1` → `STEP 0 120` → `PARK` → `SPAWN 1`，然后逐条转发：

```text
PARK                      -> OK park=1 template=1 pid=47268 frame=120
SPAWN 1                   -> OK spawned=1 pid=47269 handle=1 template=1
CHILD 1 HAZARDS           -> OK frames=120 nb=1 nl=0 ne=1 ...（213 B）
CHILD 1 PLAYER            -> OK frames=120 ... speedNormalAxial=40800000 ...
CHILD 1 EVENTS            -> OK frames=120 frameEventFlags=0000
CHILD 1 BULLETS 0 4       -> OK frames=120 count=1 ... bullets=1536:6:0:...
CHILD 1 LASERS 0 4        -> OK frames=120 count=0 ... lasers=
CHILD 1 ENEMIES 0 4       -> OK frames=120 count=1 ... enemies=0:1000005d:...
CHILD 1 HASH              -> OK hash=a79faafc95583aa9 frames=120
CHILD 1 STATE             -> OK frames=120 f0=120 ... f24=1
CHILD 1 STEP 0 1          -> OK advanced=1 frame=121 hash=d298c6ea4f8bfeff
CHILD 1 HASH              -> OK hash=d298c6ea4f8bfeff frames=121
FRAME                     -> OK frame=120          # 父进程没有被推进
CHILD 1 RESET 0 1 0 1     -> ERR child-command     # 非只读命令仍被拒
CHILD 1 QUIT              -> OK bye
```

一致性判定：

| 检查 | 结果 |
|---|---|
| 父/子 `HAZARDS` 回复逐字节相同 | **是**（各 213 B，`frames/nb/nl/ne` 全等） |
| 父/子 `PLAYER` 回复逐字节相同 | **是** |
| 父/子 `HASH` 相同（fork 时刻状态相同） | **是**（`a79faafc95583aa9`） |
| 子进程 `STEP` 只推进自己 | **是**（子 121，父仍 120） |
| 非只读命令仍拒绝且显式 | **是**（`ERR child-command`） |
| 子进程复位状态可用 | **是**（`HAZARDS` 未回 `ERR hazards-not-ready`） |

**大回复（跨过旧 4 KiB 行缓冲）**：在 stage 0、seed 1 的 frame 1432 上父进程 `HAZARDS` 为 **5,208 B**（57 子弹 + 14 敌人）。同一帧 `CHILD 1 HAZARDS` 回复也是 **5,208 B 且逐字节相同**；这正是 `ForwardLine` 流式转发的目标路径（旧实现会在 4 KiB 处截断成 `ERR child-read`）。

> 备注：`CHILD 1 EVENTS` 在同一帧回 `OK frames=1432 frameEventFlags=0c00`（`EFFECT_SPAWNED|BULLET_PATTERN_SPAWNED`），说明子进程的 `EVENTS` 读取也走通；该帧没有命中位。

### 3.3 `EVENTS` 观测到 `PLAYER_HIT=0x2`

真实对局（stage 0 / Normal / team 0 / seed 1），策略为**不动**（`STEP 0 1`，mask=0），逐帧读 `EVENTS`：

| 观测点 | 回复 | 解释 |
|---|---|---|
| 启动后、`RESET` 前 | `ERR events-unavailable g_replaymanager=0` | 没有 `ReplayManager`，**明确不可用**，不是 0 |
| `RESET` 后 | `OK frames=0 frameEventFlags=1400` | `EFFECT_SPAWNED|ENEMY_SPAWNED`（复位时 setup 帧的残留事件） |
| 第 678 次观测 | `OK frames=678 frameEventFlags=0602` | `PLAYER_HIT(0x2)` + `PLAYER_DYING_STARTED(0x200)` + `EFFECT_SPAWNED(0x400)`，**抓到 0x2** |

- 尝试次数：678 帧（`STEP 0 1` + `EVENTS` 各 678 次），第 1 次命中即停止；未出现 game over（`exit=1` 未触发）。
- 事件位是引擎写的原值，未做任何合成/推断；测试与卡里都没有把“代理判据”写成 `PLAYER_HIT`。
- 交叉对照：`build/rl`（旧二进制）对 `EVENTS` 回 `ERR unknown-command EVENTS`，即新命令是**追加**，旧引擎/旧客户端不会误解析。

### 3.4 Python 客户端（`touhou_rl/v400/gym/protocol.py`，追加式、向后兼容）

- `EventFrame(frame, frame_event_flags)` + `player_hit` + `names`；`REPLAY_FRAME_EVENT_*` 常量与引擎枚举一一对应。
- `parse_events_reply()`：`OK` 但缺 `frameEventFlags` → `HarnessError`（fail-closed）；`ERR` 在 `request()` 层就抛出。
- `HarnessClient.events()`；`child(handle, command)` 返回原始行；`child_snapshot()` 按 verb 分派；`child_hazards()/child_player()/child_events()` 复用父进程同一套 decoder（`parse_hazards_reply` / `parse_player_reply` / `parse_events_reply`），所以子/父语义不会漂移；`parse_child_reply()` 对未知 verb 原样返回。
- 现有 API 一个未改，仅新增；`CHILD` 行格式固定为 `CHILD <handle> <command>`，拒绝换行与非法 handle。

### 3.5 测试统计

```text
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q \
    tests/test_v400_teacher_import.py tests/test_v400_teacher.py \
    tests/test_v400_child_hazards.py tests/test_v400_hazards.py tests/test_v400_protocol.py
84 passed in 2.51s
```

- 新增 `tests/test_v400_child_hazards.py`：**18 passed**，含 2 个真实引擎测试（未 skip，实际启动 `build/rl_hazard2`）：
  - `test_live_child_read_only_snapshots_match_the_parent`
  - `test_live_child_forwards_a_hazards_reply_larger_than_4kib`
- 新增 `tests/test_v400_teacher_import.py`：**8 passed**。
- `python -m compileall -q touhou_rl/v400/teacher touhou_rl/v400/gym/protocol.py`：exit 0。

---

## 4. 复现

```bash
# 构建（WSL，-j8；不写 build/rl 与 build/rl_hazard）
cd /root/Touhou_RL/third_party/th08-rl
cmake -S . -B build/rl_hazard2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DTH08_RL_HARNESS=ON \
  -DTH08_ENABLE_REPLAY_TRACE=OFF \
  -DTH08_DX8_SDK_ROOT=$PWD/scripts/prefix/mssdk
ninja -C build/rl_hazard2 -j8 th08-modern

# 真机探针（HASH 100 帧 / CHILD 快照 / EVENTS 命中）
python3 .tmp_v400_child_probe.py
python3 .tmp_v400_child_probe_big.py        # >4 KiB 子进程回复

# Python 测试（Windows 侧）
#   $env:PYTHONPATH='D:\WorkShop\Touhou_RL'
#   python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q <files>
```

---

## 5. 限制与遗留

1. `CHILD` 子进程的可回答性依赖“`PARK` 之前父进程已经 `RESET`”。本卡场景与仓库现有 `PARK/SPAWN` 用法都满足；若未来 `PARK` 早于 `RESET`，子进程会回 `ERR hazards-not-ready reset-required`（显式），本卡未构造该场景。
2. `EVENTS` 在 `g_ReplayManager == NULL` 时按设计不可用；replay 链被删除后的状态没有单独构造，只依据 `DeleteReplayManager` 置 NULL（`ReplayManager.cpp:743`）+ 实测复位前回复。
3. 本卡只验证 shadow role（无窗口），没有启动 Windows 零售实例；零售 trace 采集未受影响。
4. `touhou_rl/v400/teacher/engine.py` 与 `hazard_tape.py` 的模块注释仍写着“`CHILD <handle> HAZARDS` 被拒绝”（EXP-P1-003 的旧事实）。本任务白名单只允许改 teacher 的 SpeedTable 导入，故未动这些注释；`probe_capabilities()` 若用 `build/rl_hazard2` 重跑，`child_hazards` 会记录 `OK`，建议后续工作包一并更新。
5. [EXP-P1-004](<EXP-P1-004-geometry-engine-agreement.md>) §1 “`frameEventFlags` 完全不可观测”的结论基于**没有 EVENTS 命令**的旧协议；本卡之后该结论作废（原卡数字与代理判据本身仍然有效）。
6. `.tmp_v400_child_build.log` 末尾的 `EXIT=True` 是首次后台命令里 PowerShell 提前展开 `$?` 的噪声；权威退出码来自复核脚本：`CMAKE_EXIT=0`、`NINJA_EXIT=0`。

---

## 6. 证据文件

| 内容 | 路径 |
|---|---|
| HASH/CHILD/EVENTS 原始证据 JSON | `.tmp_v400_child_evidence.json`（工作区根目录与 WSL 仓库各一份） |
| 大回复证据 JSON | `.tmp_v400_child_big.json` |
| 真机探针脚本 | `.tmp_v400_child_probe.py`、`.tmp_v400_child_probe_big.py` |
| 构建/复核日志 | WSL：`.tmp_v400_child_cmake.log`、`.tmp_v400_child_build.log`、`.tmp_v400_child_cmake_recheck.log`、`.tmp_v400_child_build_recheck.log` |
| 构建前 SHA 对照 | WSL：`.tmp_v400_child_sha_before.txt` |
| 补丁脚本（可审计） | WSL：`.tmp_v400_child_patch.py` |
| 新测试 | `tests/test_v400_child_hazards.py`、`tests/test_v400_teacher_import.py` |
