# EXP-P0-075（v4.0.0 WP1）：批跑卡死止血、预算 watchdog 与导航挂死诊断

- 日期：2026-11-09 12:21（现场卡死）→ 13:01 止血 → 13:08 批跑重启（本卡写于 13:12）
- 触发：批跑 `tools/rl/v400_retail_trace_batch.py`（pid 21072）在 fixture `easy-2-b`（`th8_ud2b76.rpy`，`stages=[0,1,2,3,5,7]`，`calculationFrames=98098`）卡死 **39 分钟**
- 目标：先止血，再修「预算不生效」的根因（watchdog），诊断导航挂死，恢复批跑并给出 ETA
- 硬约束：不改 `third_party/th08` / 零售 exe / 依赖；不伪造 journal；不删除任何既有证据；同一时刻只有一个 `th08.exe`
- 本卡只动两个工具文件（`tools/rl/v400_retail_trace_debugapi.py`、`tools/rl/v400_retail_trace_batch.py`）、两个测试文件、本卡、以及 `.tmp_v400_watchdog*`

## 1. 结论（TL;DR）

| 问题 | 实测答案 |
| --- | --- |
| 卡死是什么 | 观测器**主线程卡在一个无上限的 Win32 调用里**（`winapi.force_foreground` 的 `AttachThreadInput`/`SetForegroundWindow`/`SwitchToThisWindow` 段），不是在循环里空转 |
| 为什么 `--budget-seconds 1800` 没生效 | 预算**只在主循环里被采样**（`while time.perf_counter() < deadline`）；主线程卡在该调用内部后，主循环一次都没再执行，所以预算永远不会被观察，也不会写 `report.json` |
| 止血做了什么 | 按 batch → observer → game 顺序强杀 3 个进程；journal **追加** 1 条 `status="failed"`；原 3 条终态行和证据目录零改动 |
| watchdog 是否生效 | **生效**。7 个新单测（含"永久阻塞的假导航"）+ 1 个真实进程 e2e：主线程阻塞、watchdog 到点触发、写完 report/rows、清 DR、detach、杀游戏、进程**退出码 7** |
| 导航挂死根因 | **不是** stage 集合 / `stage 4` / 列表滚动。`easy-1-b` 的 stage 集合与 `easy-2-b` 完全相同 `[0,1,2,3,5,7]` 且 126,803 行全绿；短预算复跑 `easy-2-b` 导航 **7.6 s 就进回放**。真正的窗口在 `phase="attached"` → `phase="navigating"` 之间，那里唯一无上限的调用就是前台转移 |
| `easy-2-b` 是否已能跑 | **能跑**（attempt-2 正在跑）：13:08:09 进回放，13:10:15 已 15,107 行，61.5 行/s，ETA ≈ 24 min |
| 批跑是否重启 | **已重启**：batch pid **22328**，当前 fixture `easy-2-b`，`th08.exe` 单实例 pid 23436 |
| failed 重试规则 | 已改为 **"failed 可重试一次"**（`--max-attempts 2` 默认；第 2 次仍 failed 则 exhausted 留给人工，不再每轮重试），规则写进 `report.json.retryPolicy` 与启动日志 |
| 测试统计 | `tests/test_v400_retail_trace_debugapi.py` **44 passed**（37→44，+7）；`tests/test_v400_retail_trace_batch.py` **31 passed**（27→31，+4）；合计 **75 passed**，`compileall` 退出码 0 |
| ETA（剩 104 份） | 剩余 `calculationFrames` 合计 **11,880,150**；按实测 61.5 行/s ≈ **53.7 h**，按批跑历史均值 79 行/s ≈ **41.8 h**，即 **2026-11-11 白天**跑完（不含重试与大 fixture 超预算） |

一句话：**卡死是"无上限 Win32 调用 + 预算只在主循环采样"的组合；watchdog 把这类卡死从"永久"压到"预算+5 s"，导航本身没有 stage 4 依赖。**

## 2. 卡死时间线（全部来自现场证据）

| 时间（+08:00） | 证据 | 内容 |
| --- | --- | --- |
| 12:21:10 | `batch2.stdout.log` | `--- fixture 2/106 easy-2-b (th8_ud2b76.rpy, 98098 frames)`，`attempt 1 -> .tmp_v400_trace_batch\easy-2-b`，命令行带 `--budget-seconds 1800` |
| 12:21:11 | `Get-Process`（止血时） | `th08.exe` pid **3276** 启动 |
| 12:21:13 | `easy-2-b/debugger.log` 末行 | `[2.569] armed 15 threads (14 live); dr7 readback=['0x415','0x415']` |
| 12:21:13 | `easy-2-b/status.json` | `phase="attached"`、`pid=22424`、`gamePid=3276`、`armSeconds=2.57`；**此后 39 分钟不再更新** |
| 12:21:13 | `easy-2-b/rows.jsonl` | **0 字节**（`TraceCollector` 已创建该文件，但一行都没写） |
| 12:51:10 | 计算 | `12:21:10 + 1800 s` → 预算到点，**无任何反应** |
| 13:00:29 | `heartbeat.json` | 仍是 `rowsSoFar=0`、`fractionOfFixture=0.0`、`batchElapsedSeconds=4023.6` |
| 13:01:51 | 止血 | 强杀三进程；`hungSecondsSinceLastLog=2437.7 s`、`budgetOverrunSeconds=641.0 s` |

关键结构性事实（决定了根因）：`debugger.log` 最后一条日志和 `status.json` 的 `phase="attached"` 是同一次 `status.update(phase="attached", ...)` 前后写出的，而**下一个可观测效果**是 `phase="navigating"` + `foregroundOk` + 导航日志。两者之间代码只做四件事：

1. `report["attach"] = {...}`（纯内存）
2. `TraceCollector(...)`（创建 `rows.jsonl` —— 现场该文件存在且为 0 字节，**证明这一步已执行**）
3. `status.update(phase="attached", ...)`（**证明已执行**）
4. `keyboard = winapi.SendInputKeyboard()`（构造函数只有一个 `set()`，不可能阻塞）
5. `nav_lock`（本轮没有 `--nav-lock`，为 `None`）
6. `foreground_ok = winapi.force_foreground(hwnd)` ← **唯一的无上限内核调用集合**

`force_foreground` 的实现里，`AttachThreadInput(current, target_thread, True)` / `SetForegroundWindow` / `SwitchToThisWindow` 都是"等到目标线程处理输入队列"的同步调用；而目标进程的线程正被我们自己的 DR0..DR3 断点冻结，**两者可以互相等待**。它外层的重试循环和 `time.sleep(0.02)` 都是有界的，所以"永久卡住"只能发生在这三个调用内部。

> 未能进一步区分是 `AttachThreadInput` 还是 `SetForegroundWindow`：这需要卡死瞬间的线程栈。watchdog 现在会在触发时把 `sys._current_frames()` 写进 `watchdog-stacks.json`（见 §3.4），**下一次发生就能直接指认具体 API**。

## 3. 止血动作（可复核）

### 3.1 杀进程顺序

```
Stop-Process -Id 21072   # batch：唯一的 journal 写入者，先杀它 → 不会有"自己写一半"的条目
Stop-Process -Id 22424   # observer（pid 见 easy-2-b/status.json 的 "pid"）
Stop-Process -Id 3276    # th08.exe
```

先杀 batch 的原因：batch 的 `subprocess.run(..., timeout=1800+900)` 在 **13:06:10** 会自己到期，届时它会写入一条 `wall-clock timeout after 2700 s` 并进入 120 s 的 `guard_after`，很可能再追加一条 "residual th08.exe" 的 aborted 记录 —— 那会把现场搅浑。止血后 `Get-Process th08,python` 为 **0 个**。

### 3.2 journal 追加（第 4 行，原 3 行不动）

- 追加前：3 行 / 8103 B / sha256 `880B22DB…5973`；追加后：**4 行 / 10418 B**
- 写入方式：用批跑自己的 `append_journal()`（`fsync`、单行 JSON），脚本 `.tmp_v400_watchdog/append_failed_entry.py`
- 条目要点（完整字段见 journal）：

| 字段 | 值 |
| --- | --- |
| `fixture_id` / `file` | `easy-2-b` / `th8_ud2b76.rpy` |
| `status` / `attempt` | `failed` / `1` |
| `rows` / `rowsBytes` | `0` / `0` |
| `error` | `navigation hang; observer budget not enforced` |
| `beganAt` / `endedAt` | `2026-11-09T12:21:10+08:00`（batch 日志行）/ `2026-11-09T13:01:51+08:00`（止血时刻） |
| `elapsed_s` / `budgetOverrunSeconds` | `2441.0` / `641.0` |
| `timed_out` / `killedByOperator` | `true` / `true` |
| `statusAtHang` | `{phase:"attached", observerPid:22424, gamePid:3276, budgetSeconds:1800.0, updatedAt:"2026-11-09T12:21:13+0800"}` |
| `lastObserverLogAt` / `hungSecondsSinceLastLog` | `12:21:13+08:00` / `2437.721` |
| `exit_code` | `null`（观测器从未退出，没有退出码可记 —— 不编造） |
| `followUp` | `EXP-P0-075-WP1-batch-watchdog-nav-hang` |

- `guard_before.ok=true`（12:21:10 起跑时无 th08）用 `note` 注明是事后补记；`guard_after` 记 `ok:null` + `note` 说明为何是人工终止而不是等 guard —— 两处都写了来源，不冒充"当场观测"。
- **未删除**：`.tmp_v400_trace_batch/easy-2-b/{debugger.log(721 B), rows.jsonl(0 B), status.json(428 B)}` 全部原样保留；重跑写入 `attempt-2`（`next_attempt()` 只增不删）。

## 4. watchdog：实现与证据

### 4.1 设计

新增（`tools/rl/v400_retail_trace_debugapi.py`）：

- `class Watchdog`：`threading.Timer` 语义 + `fired` 闩锁，daemon 线程，`cancel()`/`rearm()`/`armed`
- `emit_timeout_state(...)`：超时收尾的**唯一实现**，每步单独 try/except（一步失败不影响其余）：
  1. `stacks` → `watchdog-stacks.json`（全线程 Python 栈，主线程标 `main-<tid>`）
  2. `rows` → `collector.finish(reason=…)` + `flush()`（**落盘内存里缓冲的行**；`ROW_BATCH=1024`，所以正常运行时 `rows.jsonl` 可能长时间是 0 字节，超时必须补写）
  3. `debugger` → `NativeDebugger.hard_stop()`：清 DR0..DR3 + `DebugActiveProcessStop`（**不 pump**，避免与卡住的主线程抢共用的 `DEBUG_EVENT` 缓冲区；句柄故意不关，进程马上退出）
  4. `game` → `terminate()` → `wait(5 s)` → `kill()`
  5. `status` → `phase="timeout"`
  6. `report` / `report-final` → `status="timeout"`、`result.exitCode=EXIT_TIMEOUT(7)`
- 触发时机：`--budget-seconds + --watchdog-grace`（grace 默认 **5 s**）。grace 让健康运行仍走"主循环自己到点 → `budget exhausted` → 正常评分"的优雅路径；只有主线程真卡住时才由 watchdog 兜底。
- 出口：`os._exit(EXIT_TIMEOUT)`，**保证进程退出码 7**（即使主线程还卡在系统调用里）。
- 已结束的运行不会被误杀：主循环退出后置 `run_state["ending"]=True`，watchdog 若此时才触发会 `rearm(max(grace,15))` 再等一次；`finally` 里 `watchdog.cancel()` 彻底解除。
- 顺带修的**第二个无上限点**：`foreground_with_timeout()` —— `force_foreground` 改在一次性 daemon 线程里跑，`join(--foreground-timeout)`（默认 **8 s**）超时就用 `foregroundOk=false` 继续，并在 `report.foregroundTransfer.timedOut` 留痕。这才是 §2 那个窗口的直接修复。
- 顺带加的**快速失败**：`--nav-deadline`（默认 **90 s**）—— 回放迟迟不开始就判 `failed`（exit 6）而不是耗到预算；`_grade` 里 `rows==0` 的 reason 会带上 `navigation.error`。

### 4.2 单测（7 个新增，全绿）

| 测试 | 断言 |
| --- | --- |
| `test_watchdog_fires_while_a_navigation_is_blocked_forever` | **任务要求的核心用例**：假导航永久阻塞（`Event().wait()`）时，watchdog 到点仍完成 rows 落盘、`hard_stop`、杀游戏、`status=timeout`、`result.exitCode=7`；并断言 `watchdog-stacks.json` 里能看到被阻塞的导航线程 |
| `test_watchdog_cancel_before_the_deadline_prevents_a_stale_timeout` | `cancel()` 后到点不再触发 |
| `test_watchdog_rearm_gives_a_finishing_run_its_grace` | `rearm()` 后可以再次触发（对应"已在收尾"的分支） |
| `test_timeout_ending_survives_a_wedged_row_stream` | `collector.finish()` 抛异常时，detach / 杀游戏 / report 仍然完成，错误记进 `watchdog.errors` |
| `test_navigation_deadline_fails_fast_without_waiting_for_the_budget` | 90 s 判据的真值表，含 0=禁用、已开播不算 |
| `test_foreground_transfer_is_bounded_so_a_blocked_attach_cannot_hang_the_run` | `force_foreground` 永久阻塞时 `foreground_with_timeout` 在 0.25 s 返回 `(False, True)` |
| `test_navigator_always_returns_even_when_playback_never_starts` | 导航线程在"永远到不了回放"时也会在有限时间内结束 |

### 4.3 真实进程 e2e（不是单测）

`.tmp_v400_watchdog/watchdog_e2e.py`：起一个真实的 stand-in 游戏子进程（`python -c time.sleep(600)`），主线程阻塞在假导航里，`Watchdog(3.0)` 触发后走**真实** `emit_timeout_state` + **真实** `os._exit`。

```
[  0.003] stand-in game pid=20768
[  0.012] watchdog armed for 3.0 s; blocking the main thread in a fake navigation
[  0.013] fake navigation entered; it never returns
[  3.021] watchdog fired; emitting the timeout ending
[  3.047] watchdog: forced finish(status='timeout') in 0.025 s; steps=['debugger','game','report','rows','stacks','status']
[  3.047] game alive after cleanup: False
[  3.047] exiting with 7
PROCESS EXITCODE=7
```

产物复核：`rows.jsonl` **4096 行**（缓冲行已落盘）、`report.json` `status=timeout`/`exitCode=7`、`watchdog-stacks.json` 主线程栈 `main-…: watchdog_e2e.py line 117, time.sleep(120)`、stand-in 进程**已消失**、shell 收到退出码 **7**。

## 5. 导航挂死：根因判定

### 5.1 先排除"stage 集合"假设

- `easy-1-b`（`th8_ud2c0a.rpy`）的 stage 集合与 `easy-2-b` **逐元素相同** `[0,1,2,3,5,7]`，journal 里 126,803 行 **ok**（1664.9 s）。
- 导航代码根本不看 stage 集合：`menu_key(screen, substate, cursor, state, sub_timer, start_stage, playback_mode)` 只比较 `cursor == start_stage` / `cursor == playback_mode`，而批跑永远传 `--start-stage 0`；回放列表是按**索引**选，不是按 stage 序号。
- 因此"按 stage 4 → 失败 → 重试死循环"和"replay 列表索引/滚动死锁"这两个假设**不成立**：代码里不存在按 stage 连续性推进的逻辑。

### 5.2 短预算复跑（`--budget-seconds 120`）证明该 fixture 导航正常

```
[   7.434] trace: rng origin of stage 0 = 90
[   7.633] navigation: replay playback detected {"flags":8,"isReplay":true,"isDemo":false,"stage":0,...}
[   7.636] playback started at 2026-11-09T13:04:56+0800; navigation lock not used
[   8.768] trace: row read mode=merged / trace: stage 0 began at frame None
[ 120.006] budget exhausted
[ 120.048] detached; cleared debug registers on 19 threads
[ 128.157] exit 6 status=partial elapsed=128.125s
```

`navPath=closed_loop`、`keyTaps=8`、`foregroundOk=true`、14325 行写入 `rows.jsonl`（2,399,690 B）、退出码 6（partial，符合"预算到点但回放没放完"的语义）、`th08` 自行终止、无残留。**结论：导航路径对该 fixture 无确定性缺陷，卡死是偶发。**

### 5.3 所以挂死点在哪

`phase="attached"`（12:21:13 后无更新）→ `phase="navigating"` 之间，唯一无上限的调用是 `winapi.force_foreground()`（详见 §2 的六步排除）。挂死在"等什么条件"上是：**`AttachThreadInput`/`SetForegroundWindow` 等目标进程线程服务输入队列，而该线程正被我们自己的硬件断点冻结在 `WaitForDebugEvent`/`ContinueDebugEvent` 之间**；主线程在等它，它又在等主线程的 `ContinueDebugEvent` —— 典型自锁，所以永不满足。

### 5.4 修了什么

| 层 | 修复 | 效果 |
| --- | --- | --- |
| 直接原因 | `foreground_with_timeout()`：前台转移搬进一次性 daemon 线程，`join(8 s)` | 该窗口从"无上限"变成"最多 8 s"，卡死不再可能 |
| 兜底 | 独立 watchdog（§4） | 任何其它阻塞（RPM / `WaitForDebugEvent` / 未来的 Win32）都只能撑到 `预算+5 s` |
| 快速失败 | `--nav-deadline 90 s` | "活着但永远进不了回放"的导航 90 s 判 failed，不烧 1800 s |
| 现场取证 | watchdog 写 `watchdog-stacks.json` | 下次同类卡死直接给出阻塞栈，不用再靠推理 |

## 6. 批跑重启与 `easy-2-b` 现状

- 重启命令：`python -u tools\rl\v400_retail_trace_batch.py --root .tmp_v400_trace_batch --first easy-2-b`（13:08:02）
- `retry policy: skip ['ok','mismatch']; retry failed up to 2 attempts` —— 日志实拍
- batch pid **22328**、observer pid 24380、游戏 pid **23436**（`Get-Process th08` 计数 = **1**）
- `easy-2-b` **attempt 2** → `.tmp_v400_trace_batch/easy-2-b/attempt-2`（attempt-1 证据原样保留）

| 时刻 | rows | 说明 |
| --- | ---: | --- |
| 13:08:09 | 0 | `playbackStarted`（closed_loop） |
| 13:08:20 | 2,724 | `rows.jsonl` 仍 0 字节是 `ROW_BATCH=1024` 的批量写，不是故障 |
| 13:08:47 | 5,208 | |
| 13:09:47 | 8,905 | 实测 **61.5 行/s** |
| 13:10:15 | 15,107 | 15.4%，ETA ≈ 1450 s → 约 **13:33** 收尾 |

`heartbeat.json` 同步：`state=running`、`fixtureId=easy-2-b`、`attempt=2`、`rowsSoFar` 递增、`lastDebugEventAgeSeconds=0.0`（调试事件流正常）。

## 7. 剩余 104 份里的"特殊 stage 集合"清单（manifest 统计）

`third_party/th08/config/replay-fixtures.json` 共 108 份，只有 5 种 stage 集合：

| stage 集合 | 全量 | 未跑 | 特殊性 |
| --- | ---: | ---: | --- |
| `[0,1,2,3,5,6]` | 24 | 24 | **无 4**、无 7、非连续 |
| `[0,1,2,4,5,7]` | 24 | 23 | 无 3、无 6、非连续 |
| `[0,1,2,4,5,6]` | 24 | 23 | 无 3、无 7、非连续 |
| `[0,1,2,3,5,7]` | 24 | 22 | **无 4**、无 6、非连续（`easy-1-b`/`easy-2-b` 属此族） |
| `[8]` | 12 | 12 | **只有 EX stage 8**、不从 0 开始（`extra-*`） |

按问题维度点名（**未跑**的 104 份里）：

- **无 stage 4：58 份** = `[0,1,2,3,5,6]` 24 份 + `[0,1,2,3,5,7]` 22 份 + `[8]` 12 份
- **含/只有 stage 8：12 份**（全是 extra）：`extra-0-extra, extra-1-extra, extra-2-extra, extra-3-extra, extra-4-extra, extra-5-extra, extra-6-extra, extra-7-extra, extra-8-extra, extra-9-extra, extra-10-extra, extra-11-extra`
- **不从 stage 0 开始：12 份**（同上 extra）
- **无 stage 6：45 份**（`[0,1,2,4,5,7]` 23 + `[0,1,2,3,5,7]` 22）；**无 stage 7：47 份**（`[0,1,2,3,5,6]` 24 + `[0,1,2,4,5,6]` 23）
- **非连续集合：92 份**（除 12 份 extra 外全部）

**风险排序（下一批最可能出问题的）**：

1. **12 份 extra（`[8]`）**：3 份已成功的 fixture 全是 `[0,1,2,…]` 族，**这条导航分支一次都没被真实跑过**（`start_stage=0` 但 replay 只有 stage 8），是下一张卡的头号候选。
2. **12 份 >135k 帧**：`normal-6-b, normal-8-b, hard-1-b, hard-8-b, lunatic-3-b, lunatic-4-b, lunatic-8-b, lunatic-11-b, normal-8-a, hard-8-a, lunatic-8-a, lunatic-10-a`（最大 149,986 帧）。按实测 61.5 行/s，135k 帧需 ~2195 s、150k 帧需 ~2439 s，**都会超过 1800 s 预算**而被截断成 failed/partial —— 建议为这些单独抬预算或单开卡处理。
3. 无 stage 4 的 58 份：已由 `easy-1-b`/`easy-2-b` 同族证明导航不受影响（§5.1），仅作记录。

## 8. 回归

```
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q \
    tests/test_v400_retail_trace_debugapi.py tests/test_v400_retail_trace_batch.py
-> 75 passed in 2.71s        # 44 (debugapi, +7) + 31 (batch, +4)

python -m compileall -q tools/rl/v400_retail_trace_debugapi.py tools/rl/v400_retail_trace_batch.py
-> exit 0
```

批跑侧改动（`tools/rl/v400_retail_trace_batch.py`）：

- `exhausted_entries(entries, max_attempts)` + `retry_policy()` + `--max-attempts`（默认 `DEFAULT_MAX_ATTEMPTS=2`）
- 语义：**failed 不是终态**（`TERMINAL_STATUSES=(ok,mismatch)`），默认重试；累计 2 条 failed 后视为 exhausted，跳过并把该 fixture 计为 failed 留人处理；`--skip-failed` 仍然表示"从不重试"
- 规则写进 `report.json.retryPolicy`（含 `rule` 文本）和启动日志
- 新增 4 个测试：`test_failed_entry_is_retried_once_then_left_exhausted`、`test_retry_policy_is_recorded_for_audit`、`test_main_retries_a_failed_fixture_once_and_keeps_the_old_evidence`（断言 `attempt-2` 新建且 attempt-1 的 `debugger.log` 未被删）、`test_main_leaves_a_twice_failed_fixture_exhausted`

## 9. ETA

| 口径 | 数值 |
| --- | --- |
| 未跑 fixture | **104**（journal：3 ok + 1 failed(easy-2-b)，其余 pending） |
| 剩余 `calculationFrames` 合计 | **11,880,150** |
| 单 fixture 实测速率（easy-2-b attempt-2） | **61.5 行/s** |
| 批跑历史均值（`heartbeat.json`） | 50.18 行/s（含失败 attempt）/ 79 行/s（前 3 份成功件） |
| ETA（按 61.5 行/s） | ≈ 193,200 s ≈ **53.7 h** → 约 **2026-11-11 19:00** |
| ETA（按 79 行/s） | ≈ 150,400 s ≈ **41.8 h** → 约 **2026-11-11 07:00** |
| 批跑自报投影 | `projectedRemainingSecondsByFixture=172,637 s`（47.9 h）、`byRows=236,735 s`（65.8 h） |

**取 42–54 h，即 2026-11-11 早上到傍晚**；另有 12 份大 fixture 可能因 1800 s 预算被截断、以及 ≤1 次 failed 重试的开销，实际落点更可能偏后。

## 10. 剩余风险与后续卡

1. **watchdog 还没在真实卡死上"响过"**：本次修复后 `easy-2-b` 一次就跑通（导航 7.6 s），真实偶发卡死没再复现。当前保证是"单测 + 真实进程 e2e"，以及"下次真卡住时 `watchdog-stacks.json` 会自动留下阻塞栈"。**下一次卡死 = 免费拿到具体 API 名。**
2. **`force_foreground` 内部到底是哪个 API**：已把它整段限制到 8 s，但精确归因要等 §10.1 的栈。
3. **12 份大 fixture 会被 1800 s 预算截断**（§7 风险 2）—— 需要单独决定抬预算还是接受 partial。
4. **12 份 extra（`[8]`）导航分支未验证**（§7 风险 1）—— 建议下一张卡先单跑 1 份 extra 再用短预算验证。
5. 批跑单实例约束仍靠 `tasklist` 轮询 + 起跑前 guard；本卡期间未启动任何并行实例。

## 11. 证据索引

| 用途 | 路径 |
| --- | --- |
| 卡死现场（原样保留） | `.tmp_v400_trace_batch/easy-2-b/{debugger.log, status.json, rows.jsonl}` |
| 卡死前的批跑输出 | `.tmp_v400_trace_batch/batch2.stdout.log`（12:21:10 之后无新行） |
| journal（4 行） | `.tmp_v400_trace_batch/journal.jsonl` |
| journal 追加脚本 | `.tmp_v400_watchdog/append_failed_entry.py` |
| 短预算复跑（证明导航正常） | `.tmp_v400_watchdog_nav2b/{debugger.log, report.json, rows.jsonl}`、`.tmp_v400_watchdog/nav2b.out.log` |
| watchdog 真实进程 e2e | `.tmp_v400_watchdog/watchdog_e2e.py`、`.tmp_v400_watchdog/e2e/{watchdog.log, report.json, rows.jsonl, watchdog-stacks.json}` |
| manifest 统计 | `.tmp_v400_watchdog/manifest_stats.py` |
| 重启后的批跑 | `.tmp_v400_trace_batch/batch3.stdout.log`、`.tmp_v400_trace_batch/heartbeat.json`、`.tmp_v400_trace_batch/easy-2-b/attempt-2/` |
