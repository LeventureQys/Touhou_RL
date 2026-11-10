# EXP-P0-068-WP1 — 零售轨迹观测器的 Windows 原生 Debug API 实现（无安装）

- 日期：2026-11-09（Windows，Python 3.12.10 **64 位**，零售原件 `third_party/th08/resources/th08.exe` 840,704 B / SHA-256 `330fbdbf…d924`）
- Provider / model：DeepSeek / DeepSeek v4.1 flash
- 工具：`tools/rl/v400_retail_trace_debugapi.py`（新增），测试 `tests/test_v400_retail_trace_debugapi.py`（新增，26 passed）
- 范围：单条回放 `third_party/th08/tests/fixtures/replays/th08-1.00d/accepted/th8_ud2cd2.rpy`
  （manifest fixture `easy-0-a`，`sha256=6e80fc7b…6146`，`calculationFrames=99113`，
  `retailTrace.sha256=81f535e3e01743255fe1edbe71740f0b27050702ce4d28848ce934d38be1643c`）
- 合规：**未安装任何软件**，未改 `third_party/th08` 源码/脚本，未改零售 exe，未改依赖，未删除文件，
  未伪造 rows/complete/SHA。每轮均校验 `thirdPartyUnchanged=true`（零售 exe / th08.dat / 回放的 SHA-256 前后一致）。

---

## 0. 结论摘要

| 问题 | 结果 |
|---|---|
| POC：1 个硬件执行断点能否命中并读到内存 | **能**。`0x00452490`（`ReplayManager::ControlPlaybackFrameAdvance`）命中 5/5，退出码 0；同时读到 `stage/gameFrame/flags/replayFrame/rng/input`，帧号 1→5 连续 |
| 4 断点各自命中数 | `frame 0x452490` = **4144**，`completion 0x441F52` = **1646**，`stageStart 0x439BC7` = **4226**，`deleteReplay 0x453080` = **0** |
| 首命中顺序 | `completion` → `stageStart` → `frame`（相对 0.000 / +3.095 s / +4.445 s） |
| 采集行数 vs `calculationFrames` | **17,843 vs 99,113（不等，差 81,270）** |
| rows SHA-256 vs manifest | **不等**。实测 `49f383dc5cfa0d1ef1755f536488e8282321019da7f6afdabc332eb91995911d`，manifest `81f535e3…1643c` |
| 是否达到「真零售位一致」 | **未达到**（行数不足是唯一已证原因；见 §5 的定量外推） |
| 单阶段证据 | 阶段 0 采到 **9463 行**，其中末帧后 **76 行是对同一帧的重复命中**；`endScores["0"]=2447070` **与 manifest 的 stage 0 `endScore=2447070` 完全相等**；`stageFrames["0"]=9387 = inputRecords(9389) − 2`，正符合官方 `valid_input_tail(2)` 判据 |
| SeDebugPrivilege | **未持有**（`AdjustTokenPrivileges` 返回 err=1300），但 `DebugActiveProcess` 仍成功（目标是本进程的子进程） |

判定：**Windows 原生 Debug API 路线成立**——读点、断点语义、行格式、`complete.json` 字段全部复刻成功，
且阶段性状态（stage 0 endScore）与 manifest 逐位吻合；**唯一未达标项是墙钟预算**（完整 trace 需 ≈930 s 连续播放，
远超单次 180 s 上限），因此 `rows.jsonl` 是真实的 17,843 行前缀，SHA 必然不同，如实报告，不做任何格式调整。

---

## 1. 实现要点：官方读点如何被复刻

依据 `third_party/th08/scripts/replay-external-observer.py` + `capture-replay.py`（`candidate is None` 分支）：

| 官方机制 | 本实现 |
|---|---|
| `gdb.Breakpoint("*0x…", BP_HARDWARE_BREAKPOINT)` ×4 | `DebugActiveProcess` + `Wow64GetThreadContext/Wow64SetThreadContext` 写 `DR0..DR3`，`DR7=0x455`（4 槽 execute，`R/W=LEN=00`，bit10 保留位置 1） |
| `gdb.selected_inferior().read_memory` | 在 debug 事件内 `ReadProcessMemory`（进程同样处于冻结态，故「帧内原子读」性质相同） |
| 4 个 RVA `0x452490/0x441F52/0x439BC7/0x453080` | 完全相同 |
| `attestAddress/attestBytes` + `completion*` + `extraAttestations` | 先做 4 组 13 字节 attestation（`0x452490`→`558bec51894dfca1b4d06401c1`、`0x441F48`→`b948f56401e8feaaffff8945fc`、另两个按 PE 同 RVA 取字节），**4/4 通过**后才武装断点 |
| `xdotool` 菜单状态机（`g_TitleScreen+0x14428` 与 `title[0..20]` 的 screen/substate/cursor/state/timer） | 同一读点 + `SendInput` 扫描码（Down/Z），保持「hold 3 tick、间隔 12 tick」的节奏 |
| 25 字段行、`json.dumps(row)+"\n"` | 完全一致（字段顺序、`rngGeneration = rng[+4] − rngOrigin`、`gm[0]/gm[1]`、`GM+0x30`） |
| `complete.json`：`rows/ended/errors/stages/stageFrames/endScores/rngOrigins` | 完全一致（键序、LF 换行） |

关键运行事实（`report.json: attach`）：`armedThreads=15`（含后续 `CREATE_THREAD` 追加，最终 18），
`dr0..dr3` 回读 = `0x452490 / 0x441f52 / 0x439bc7 / 0x453080`，`dr7MatchesRequest=true`。

---

## 2. POC 阶段（1 个断点）

```powershell
python tools\rl\v400_retail_trace_debugapi.py --phase poc --out .tmp_v400_trace_dbg_poc  --budget-seconds 150 --max-hits 5
python tools\rl\v400_retail_trace_debugapi.py --phase poc --out .tmp_v400_trace_dbg_poc2 --budget-seconds 150 --max-hits 5
python tools\rl\v400_retail_trace_debugapi.py --phase poc --out .tmp_v400_trace_dbg_poc3 --budget-seconds 150 --max-hits 5
```

| 轮次 | 退出码 | 结果 |
|---|---|---|
| 1 `.tmp_v400_trace_dbg_poc` | **6** | 断点**确实触发**，但以 `0x4000001E`（`STATUS_WX86_SINGLE_STEP`）上报，被当成应用异常放行 → 游戏以 `0x4000001E`（1073741854）死亡；`hits.frame=0` |
| 2 `.tmp_v400_trace_dbg_poc2` | **0** | 5 次命中，但 5 次全在同一帧（`gameFrame=1/replayFrame=1`，18 ms 内）→ 指令被反复执行，`EFLAGS.RF` 未被置位 |
| 3 `.tmp_v400_trace_dbg_poc3` | **0** | 修好后 5 次命中，帧号 **1,2,3,4,5** 连续 |

第 3 轮原始日志（`Eip == 0x00452490` 即目标地址）：

```
[  10.765] frame hit #1 eip=0x00452490 stage=0 gameFrame=1 flags=0x0000000C replayFrame=1 rng=e57e00005c000000 input=0000
[  10.766] frame hit #2 eip=0x00452490 stage=0 gameFrame=2 flags=0x0000000C replayFrame=2 rng=e3a30000fc010000 input=0000
[  10.766] frame hit #3 eip=0x00452490 stage=0 gameFrame=3 flags=0x0000000C replayFrame=3 rng=e3a30000fc010000 input=0100
[  10.767] frame hit #4 eip=0x00452490 stage=0 gameFrame=4 flags=0x0000000C replayFrame=4 rng=e3a30000fc010000 input=0100
[  10.782] frame hit #5 eip=0x00452490 stage=0 gameFrame=5 flags=0x0000000C replayFrame=5 rng=609b00009c030000 input=0100
```

- `flags=0x0C` = `isReplay(bit3) | replayInputEnabled(bit2)`，正好通过官方过滤式 `flags & 8 and not flags & 2`
  （`GM+0x3DBAC`，`GameManagerFlags` 位域，见 `GameManager.hpp:415` 的 `C_ASSERT`）。
- `rng[+0]=0x7ee5/0xa3e3/0x9b60`、`rng[+4]=0x5c/0x1fc/0x39c` 逐步前进，`input` 由 `0x0000`→`0x0001`，
  证明读到的是真实推进的零售状态，不是常量或撕裂样本。
- `report.json: poc.firstHits[*].dr6 = 0xFFFF0FF1`：**B0=1**，确认触发源是 DR0（frame 槽）。

---

## 3. 四断点阶段

```powershell
python tools\rl\v400_retail_trace_debugapi.py --phase four --out .tmp_v400_trace_dbg_four --budget-seconds 150 --four-seconds 15
```

**退出码 0，status=partial，耗时 44.1 s。**

| 断点 | 地址 | 角色（官方语义） | 命中数 |
|---|---|---|---|
| frame | `0x00452490` | `ReplayManager::ControlPlaybackFrameAdvance`，游玩计算之后；**「计算完成」帧栅栏，写行的位置** | **4144** |
| completion | `0x00441F52` | `GameWindow::Render` 内 `call RunCalcChain` 之后 5 字节；**驱动菜单导航 / 收尾判定** | **1646** |
| stageStart | `0x00439BC7` | `GameManager::OnUpdate` 入口；stage 起始时记录 RNG generation 原点 | **4226** |
| deleteReplay | `0x00453080` | `ReplayManager::DeleteReplayManager`；回放结束 | **0** |

- 首命中顺序 `completion → stageStart → frame`（14 ms/3.095 s/4.445 s）。与语义自洽：
  `Render`（completion）在菜单/加载期就会跑到，`OnUpdate` 与 `ControlPlaybackFrameAdvance` 只在进入游玩后才出现，
  且 `frame` 在 `OnUpdate` 之后（同一帧内后置）。
- `frame : completion ≈ 4144 : 1646 ≈ 2.52`：说明 `playbackMode=2`（零售内建快进）下一个渲染帧推进约 2.5 个计算帧，
  与 `ReplayManager.cpp:299-303`（`replayMode==2 && !HasBoss && frameCounter%5 != 4` → 重启链）的语义一致。
- 总命中 10,016，`eventsHandled=10161`，`resumeFlagSet=10016 / failures=0`，`singleStepUnarmed=0`，
  即**每一次命中都被精确归因到某个断点，没有丢事件、没有落到未武装地址**。
- `deleteReplay=0`：15 s 窗口内回放未结束。为验证第 4 槽，本轮在窗口结束前主动用暂停菜单退出
  （`esc,down,down,down,z`，见 `--no-try-delete-breakpoint` 开关），**仍未触发**；说明该序列没有真正结束回放
  （或该版本走的是别的销毁路径）。该断点只证明了「已武装且地址回读正确」，**未证明命中保持诚实**。

---

## 4. 行采集阶段

```powershell
python tools\rl\v400_retail_trace_debugapi.py --phase rows --out .tmp_v400_trace_dbg_rows --budget-seconds 178
```

**退出码 6，status=partial，耗时 186.1 s（游戏侧预算 178 s）。**

```
[   9.123] stageStart: stage=0 rngOrigin=90
[  10.423] trace: stage 0 began at frame None
[ 107.982] (stage 1 起点)
[ 178.009] budget exhausted
[ 178.009] trace: finished rows=17843 errors=['phase budget exhausted before the replay ended',
            'Stage 1 ended with 4738 unconsumed input records',
            'Stage 1 score differs from the original recording']
```

| 指标 | 值 |
|---|---|
| `rows` | **17,843** |
| `calculationFrames`（manifest） | **99,113** |
| 差值 | 81,270 |
| `rows.jsonl` 字节 / 行数 | 3,054,147 B / 17,843 行，**纯 LF，无 CR** |
| `rows.jsonl` SHA-256 | **`49f383dc5cfa0d1ef1755f536488e8282321019da7f6afdabc332eb91995911d`** |
| manifest `retailTrace.sha256` | `81f535e3e01743255fe1edbe71740f0b27050702ce4d28848ce934d38be1643c` |
| 相等？ | **否** |
| 每阶段行数 | stage 0 = **9463**，stage 1 = **8380** |
| `stageFrames` | `{"0": 9387, "1": 8380}` |
| `endScores` | `{"0": 2447070, "1": 5452286}` |
| `rngOrigins` | `{"0": 92, "1": 297450}` |
| 重复帧（同 stage+frame） | **76**，全部集中在 stage 0→1 切换处（calc 9388…9463） |
| 吞吐 | ≈ **106 行/s**（17,843 / 167.6 s） |

`complete.json`（原样，官方字段顺序，LF 结尾）：

```json
{"rows": 17843, "ended": false, "errors": ["phase budget exhausted before the replay ended", "Stage 1 ended with 4738 unconsumed input records", "Stage 1 score differs from the original recording"], "stages": [0, 1], "stageFrames": {"0": 9387, "1": 8380}, "endScores": {"0": 2447070, "1": 5452286}, "rngOrigins": {"0": 92, "1": 297450}}
```

前两行与最后一行（原样）：

```
[1, 0, 1, 1, 0, 0, 0, 1073741824, 1077936128, 0, 1128267776, 1136656384, 1056629064, 3, 32485, 0, 0, 0, 0, 0, 0, 60000, 0, 0, 0]
[2, 0, 2, 2, 0, 0, 0, 1073741824, 1077936128, 0, 1128267776, 1136656384, 1056629064, 3, 41955, 416, 0, 0, 0, 0, 0, 0, 60000, 0, 0, 0]
[17843, 1, 8380, 8380, 5452286, 82, 1065353216, 1073741824, 1077936128, 1124073472, 1128906586, 1135241892, 1045220557, 0, 5016, 288428, 5, 195, 1991, 10000, 2, 82140, 0, 0, 0]
```

### 4.1 为什么行数不足（定量，不是猜测）

- 观测吞吐 ≈ 106 行/s（含 debug 事件往返 + 每帧 8 次 `ReadProcessMemory` + JSON 行写入）。
  同样的链在不写行时（§3 四断点轮）≈ 200 帧/s，即**观测自身开销让帧率降到约一半**。
- 完整 trace 需要 `99113 / 106 ≈ 935 s ≈ 15.6 min` 连续播放；单次运行上限 180 s ⇒ 单次最多约 1.9 万行。
- 无法分片拼接：回放只能从所选 stage 起播，跨 stage 重启会改变阶段初始状态与 `rngOrigins`，
  拼出来的文件不是「同一次回放」的轨迹；本实验拒绝用重排/重编号去凑 SHA。

### 4.2 一个有价值的副产物：`calculationFrames` 不是「唯一帧数」

- stage 0 的 9463 行只覆盖 frame 1…9387（9387 个不同帧），**多出的 76 行是对同一帧的重复命中**，
  且全部落在 stage 0 通关演出期间（`ControlPlaybackFrameAdvance` 在清关链里被反复进入）。
- 逐阶段「不同帧」之和 = `(9389−2)+(13118−2)+(13175−2)+(20123−2)+(20158−2)+(22292−2) = 98,243`，
  而 manifest 的 `calculationFrames = 99,113`，差 **870 行**——数量级与本轮观察到的「每次 stage 切换约 76~几百次重复命中」吻合。
- 结论：**要 SHA 相等，必须同时复现这些 stage-clear 重复行**；官方 observer 同样不做去重（`stop()` 里没有去重逻辑），
  所以「每帧一行」是错误直觉。这一条对 G0.1 的行数校验很关键。

### 4.3 与 manifest 已经对齐的部分

- `endScores["0"] = 2,447,070` **等于** manifest stage 0 的 `endScore = 2,447,070`（精确相等）。
- `stageFrames["0"] = 9,387 = 9,389 − 2`，而官方完成判据 `valid_input_tail(2, final=False) == True`
  （`replay_file.py:19-24`）——即**阶段 0 的采集在官方判据下是完整且自洽的**。
- `rngOrigins` 有值且被行内 `rngGeneration` 正确引用（第 1 行 `rngGeneration=0`、第 16 行起 `416`）。

---

## 5. 失败点与原因（按发生顺序）

| # | 失败点 | 现象 | 原因 | 处置 |
|---|---|---|---|---|
| 1 | 硬件断点命中被误判为应用异常 | 游戏以 `0x4000001E` 死亡，`hits.frame=0`（POC 轮 1） | WOW64 下 32 位进程的硬件执行断点以 **`STATUS_WX86_SINGLE_STEP (0x4000001E)`** 上报给 64 位调试器，不是 `0x80000004`；原实现按「其他异常」`DBG_EXCEPTION_NOT_HANDLED` 放行 | 同时接受 `0x80000004` 与 `0x4000001E`（断点同理接受 `0x4000001F`）；已加单测锁定 |
| 2 | 命中后指令被反复执行 | 5 次命中同帧、18 ms 内完成（POC 轮 2） | WOW64 路径**不置位 `EFLAGS.RF`**，恢复后立即重触发同一断点 | 每次命中后 `EFlags |= 0x10000` 再 `Wow64SetThreadContext`；`resumeFlagSet=10016/0 失败` |
| 3 | 行数不足 | 17,843 / 99,113 | 完整 trace 需 ≈935 s 连续播放，单次预算 180 s | **未解**。只能加长预算或整段放行；本实验如实交 partial |
| 4 | `deleteReplay (0x453080)` 0 命中 | 已武装、地址回读正确，但 15 s+8 s 窗口内未命中 | 回放未结束（该断点只在 `DeleteReplayManager` 时触发）；暂停菜单 `esc,down×3,z` 未真正结束回放 | 如实记为「未证明」，不再猜测 |
| 5 | 退出时无法用 `WM_CLOSE` 正常关闭 | `game did not close in 8 s; terminating` | 回放进行中窗口不响应 `WM_CLOSE`（引擎只在暂停菜单处理退出） | 先 `DebugActiveProcessStop` + 清 DR，再 `WM_CLOSE`，8 s 后 `terminate()`；每轮 `detach.detached=true`、`clearedThreads=17/18` |
| 6 | 观测开销 | 写行后 ~106 帧/s，不写行 ~200 帧/s | 每帧 8 次 RPM + 事件往返 | 记录并外推（§4.1）；未做优化以避免改变读点语义 |

**未发生**：`SeDebugPrivilege` 拒绝（未持有但不需要，`DebugActiveProcess` 成功）、attestation 失败、丢 debug 事件、
落到未武装地址的 single-step（`singleStepUnarmed=0`）、`third_party` 被写入（`thirdPartyUnchanged=true` 全轮）。

---

## 6. 合规与产物

- 未安装软件；`SeDebugPrivilege` 的 `AdjustTokenPrivileges` 返回 err=1300，仅记录，未提权、未改令牌。
- `third_party/th08`：`resources/th08.exe`、`build/play_fence/th08.dat`、回放文件在每轮前后 SHA-256 一致
  （`report.json: thirdPartyUnchanged=true`）。零售 exe 只被**复制**到 `.tmp_v400_trace_dbg*/rundir/`。
- 游戏只在私有目录写自己的 `score.dat`（279 B）+ `backup/score_1.261109.bak`——这正是 `--no-score-dat`
  （不预置模板 `score.dat`）的意义：不让模板存档进入回放菜单。
- 未删除任何文件；每轮 run dir 各约 497 MB（`th08.dat` 46.8 MB + `thbgm.dat` 450 MB 的副本）。

| 路径 | 内容 |
|---|---|
| `tools/rl/v400_retail_trace_debugapi.py` | 新工具（`--phase poc\|four\|rows`，`report.json` + `debugger.log`） |
| `tests/test_v400_retail_trace_debugapi.py` | 26 个单测（DR7 编码、WOW64 异常码、WOW64_CONTEXT 716 B、25 字段行、menu 状态机、PE RVA 映射、attestation 正/负例、collector 端到端、verdict） |
| `.tmp_v400_trace_dbg_poc/` | POC 轮 1（失败）：`report.json` 24 KB、`debugger.log` |
| `.tmp_v400_trace_dbg_poc2/` | POC 轮 2（命中但同帧重复） |
| `.tmp_v400_trace_dbg_poc3/` | POC 轮 3（**成功**）：5 次命中，帧 1–5 |
| `.tmp_v400_trace_dbg_four/` | 四断点：`report.json`（含 `four.hits`、`firstHitOrder`、`armedDetails`） |
| `.tmp_v400_trace_dbg_rows/` | `rows.jsonl`（17,843 行 / 3,054,147 B）、`complete.json`、`report.json`（含 `verdict`） |
| `Document/Update/Alpha/v4.0.0/experiments/EXP-P0-068-WP1-native-debugapi-trace.md` | 本文档 |

## 7. 明确不主张

- **不主张**已产出与零售位一致的 `retailTrace`：行数 17,843 ≠ 99,113，SHA 不等。
- **不主张** `0x453080` 的命中语义已被验证（只验证了武装与地址回读）。
- **不主张** 其余 107 份 fixture 可采（本次只跑 1 份，且未跑完）。
- 未验证 `playbackMode=0`（正常速度）下的行速率；未验证暂停菜单退出路径。
- 未定位 stage 0→1 处 76 行重复的精确机制（只观测到它落在清关链内）。
- 未对 `rows.jsonl` 做任何重排、补行、去重或格式调整以接近 manifest SHA。

## 8. 下一步（按性价比）

1. **整段放行**：把单轮预算提到 ≈1000 s（需用户授权突破 180 s 纪律），或让 `rows` 阶段在
   `deleteReplay` 命中后自然结束；按 §4.1 的 106 行/s 外推，一次 ≈16 min 的连续运行即可覆盖 99,113 行。
2. 若 16 min 仍不可接受：先只采 **stage 0**（≈98 s，已验证可完整结束于 `inputRecords−2`），
   用 `endScores`/`stageFrames` 判据逐阶段验收，再讨论拼接是否可信（当前判断：**不可信**，不改）。
3. 复现 stage-clear 的 76+ 重复行规律（每次切换的重复行数），这是 `calculationFrames` 校验的关键。
4. 用暂停菜单或 `esc` 结束回放以验证 `0x453080` 命中（本轮序列无效）。

---

## 9. 完整单回放运行结果（G0.1 决定性一跑，2026-11-09）

单次**连续**运行把 `th8_ud2cd2.rpy` 跑完（不分片、不拼接），并让回放**自然结束**：
`0x453080 deleteReplay` 命中 1 次，`complete.json` 的 `ended=true`、`naturalReplayExit=true`。

### 9.0 本轮对工具的改动（均在授权白名单内）

| 改动 | 原因 |
|---|---|
| `MAX_BUDGET_SECONDS` `180 → 1800` | G0.1 授权单次 16–25 min；99,113 行按 ≈106 行/s 需 ≈935 s |
| `TraceCollector.finish(..., natural_exit=False)`，`complete.json` 追加 `naturalReplayExit` 键 | 官方 `DeleteObserver` 在 `0x453080` 调 `finish()`（**无 error**，故 `ended=true`）；旧实现把「回放自然结束」记进 `errors`，导致 `ended=false`——语义错误 |
| 阶段完整性判据改为官方 `replay_file.valid_input_tail`：非末 stage `2/3`、末 stage `6/7`，且 stage 序列必须**严格相等** | 与 `replay-external-observer.py:88-98` 逐行一致；旧实现用 `(2,3,6,7)` 且跳过末阶段 |
| **SHA 对比对象修正**：`retailTrace.sha256` 不是 `rows.jsonl` 的文件哈希 | 见 §9.4；旧实现拿文件哈希去比 manifest 字段，**判据本身错了**，必然不等 |
| `TraceCollector.stage_rows` + `_grade` 输出逐 stage 表（行数/末帧/未消费输入/tail/endScore） | 逐 stage 验收 |

测试：`tests/test_v400_retail_trace_debugapi.py` 在我的改动落地时 **30 passed**（新增自然结束、官方 canonical 哈希、
行数相等但字节不等、1800 s 上限四类用例）；并行会话在 10:07:51 又追加了 3 个行格式/字节一致性用例，
同一路径当前为 **33 passed**。`python -m compileall -q tools/rl` 与 `python -m pytest ...` 均退出码 0。

### 9.1 命令与退出码

```powershell
# 运行前确认无残留
Get-Process -Name th08 -ErrorAction SilentlyContinue      # -> 空

$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python tools\rl\v400_retail_trace_debugapi.py --phase rows --out .tmp_v400_trace_full_1 --budget-seconds 1500
```

| 项 | 值 |
|---|---|
| 工具自身退出码（`report.json:result.exitCode`，也是进程返回值） | **6**（`status=mismatch`） |
| 后台 job 包装层退出码 | 1（包装层对非零返回值的映射，不是工具的判定；见 §9.4 该 6 的来源） |
| 墙钟耗时 | **1200.368 s（20 min 0.4 s）**，预算 1500 s，**未耗尽** |
| 结束方式 | `deleteReplay` 命中 → 自然结束；`DebugActiveProcessStop` + 清 DR（19 线程）→ `WM_CLOSE` 8 s 未响应 → `terminate` |
| 残留进程 | **无**（`Get-Process -Name th08` 为空） |
| `thirdPartyUnchanged` | **true**（零售 exe / `th08.dat` / 回放 SHA-256 前后一致） |

### 9.2 行数 vs `calculationFrames`

| 指标 | 值 |
|---|---|
| `rows` | **99,113** |
| `calculationFrames`（manifest `easy-0-a`） | **99,113** |
| 差值 | **0** |
| `rows.jsonl` 字节 | 18,043,248 B |
| 行格式 | 99,113 行全部 **25 字段**，纯 LF，`calc` 从 1 连续到 99,113 |

**行数逐 stage 对照**（`stageFrames` = 该 stage 最后一行的 `frame`；官方 `valid_input_tail` 只允许 `2/3`（非末）与 `6/7`（末））：

| stage | name | manifest `inputRecords` | 观测行数 | 不同帧 | 重复行 | `stageFrames` | `inputRecords−last` | 允许 tail | `endScore` 观测 | manifest `endScore` | 相等 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Stage 1 | 9,389 | 9,463 | 9,387 | 76 | 9,387 | 2 | 2/3 | 2,447,070 | 2,447,070 | ✅ |
| 1 | Stage 2 | 13,118 | 13,192 | 13,116 | 76 | 13,116 | 2 | 2/3 | 10,352,151 | 10,352,151 | ✅ |
| 2 | Stage 3 | 13,175 | 13,242 | 13,173 | 69 | 13,173 | 2 | 2/3 | 23,602,972 | 23,602,972 | ✅ |
| 4 | Stage 4B | 20,123 | 20,212 | 20,121 | 91 | 20,121 | 2 | 2/3 | 42,644,410 | 42,644,410 | ✅ |
| 5 | Stage 5 | 20,158 | 20,277 | 20,156 | 121 | 20,156 | 2 | 2/3 | 59,772,037 | 59,772,037 | ✅ |
| 6 | Final A | 22,292 | 22,727 | 22,286 | 441 | 22,286 | **6** | 6/7 | 90,941,860 | 90,941,860 | ✅ |
| 合计 | — | 98,255 | **99,113** | 98,239 | **874** | — | — | — | — | — | — |

- 每 stage 行数 = 不同帧 + 重复行；99,113 − 98,239 = **874 行**是清关链里对同一 `frame` 的重复命中
  （`ControlPlaybackFrameAdvance` 在 stage-clear 被反复进入），**不是缺行、不是多算 stage**。
  这与 §4.2 的预判（98,243 不同帧 + 870 行重复 ≈ 99,113）在量级与机制上都吻合；差异来自末 stage 的 tail 是 6 而不是 2。
- stage 序列 `[0, 1, 2, 4, 5, 6]` 与 manifest **完全一致**（Stage 4A/index 3 在该回放中不存在）。

### 9.3 `complete.json` 三张表 vs manifest

原始文件（一行，LF 结尾，键序为官方顺序 + 末尾追加本轮新增键）：

```json
{"rows": 99113, "ended": true, "errors": [], "stages": [0, 1, 2, 4, 5, 6], "stageFrames": {"0": 9387, "1": 13116, "2": 13173, "4": 20121, "5": 20156, "6": 22286}, "endScores": {"0": 2447070, "1": 10352151, "2": 23602972, "4": 42644410, "5": 59772037, "6": 90941860}, "rngOrigins": {"0": 92, "1": 297450, "2": 812328, "4": 1397518, "5": 2155904, "6": 2404318}, "naturalReplayExit": true}
```

| 表 | 判定 | 说明 |
|---|---|---|
| `stageFrames` | **逐项相等（＝manifest 可推导值）** | 每个 stage 的末帧 = `inputRecords − tail`，6/6 命中官方 tail 集合（`2,2,2,2,2,6`） |
| `endScores` | **逐项相等（精确整数相等）** | 6 个 stage 的 `endScore` 与 manifest `stages[].endScore` **完全相同** |
| `rngOrigins` | **manifest 无对应字段**，改验内部一致性 | manifest 的 `retailTrace` 只有 `schemaVersion/calculationFrames/sha256`，`stages[]` 只有 `index/name/inputRecords/endScore`；`rngOrigins` 不在 manifest 中。本轮的证据是：6 个 stage 全部有 origin、均在首行之前写入；且 `{"0":92,"1":297450}` 与 §4 的 180 s 部分运行**完全相同**（逐位可复现） |
| `ended` / `naturalReplayExit` / `errors` | **true / true / `[]`** | 自然结束，无任何 error |

### 9.4 rows SHA 与 manifest：判据修正后**精确相等**

**先给结论**：

| 对象 | 值 | 与 manifest `retailTrace.sha256` 比 |
|---|---|---|
| `rows.jsonl` **文件** SHA-256 | `ed07eeb737eec88d0ac3a3b8a70ae97b369c41d20958a5cf93db99157119b0c4` | **不等**（18,043,248 B 的字节流） |
| 官方**规范哈希**（`compare-replay-traces.trace_sha256`） | `81f535e3e01743255fe1edbe71740f0b27050702ce4d28848ce934d38be1643c` | **相等** ✅ |
| manifest `retailTrace.sha256` | `81f535e3e01743255fe1edbe71740f0b27050702ce4d28848ce934d38be1643c` | — |

**为什么文件哈希本来就不该相等**（本轮查明的判据错误）：

`third_party/th08/scripts/analysis/record-replay-expectations.py:54-55` 把
`retailTrace.sha256` 写成 `report["referenceTraceSha256"]`，而
`compare-replay-traces.py:12-19` 的定义是：

```python
def trace_sha256(metadata, rows):
    """Hash the schema and little-endian uint32 rows, independent of JSON spacing."""
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata["schema"], sort_keys=True, separators=(",", ":")).encode() + b"\n")
    pack = struct.Struct("<" + "I" * len(metadata["schema"]["fields"])).pack
    for row in rows:
        digest.update(pack(*row))
    return digest.hexdigest()
```

即：**紧凑排序的 schema JSON + 每行 25 个 `<I` 小端 uint32**，明确「independent of JSON spacing」。
它**不是** `rows.jsonl` 的文件哈希，所以 `sha256_file(rows.jsonl)` 与它比较必然不等——这正是旧实现
`_grade()` 的缺陷（§4 那次 17,843 行报「SHA 不等」也源于同一判据错误，不是数据问题）。
本轮不改任何 rows/complete 字节，只把**对比对象**换成官方函数本身（直接 `importlib` 载入
`compare-replay-traces.py`，不重写算法），并同时报告两个哈希。

复现命令（用官方函数，不改数据）：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -c "import importlib.util,json,sys; from pathlib import Path; \
sys.path.insert(0,str(Path('third_party/th08/scripts').resolve())); \
s=importlib.util.spec_from_file_location('c','third_party/th08/scripts/compare-replay-traces.py'); \
m=importlib.util.module_from_spec(s); s.loader.exec_module(m); \
sc=json.loads(Path('third_party/th08/scripts/replay-external-schema.json').read_text()); \
rows=[json.loads(l) for l in Path('.tmp_v400_trace_full_1/rows.jsonl').read_text().splitlines()]; \
print(m.trace_sha256({'schema':sc}, rows))"
# -> 81f535e3e01743255fe1edbe71740f0b27050702ce4d28848ce934d38be1643c
```

**反例（证明该哈希不是恒真）**：

| 输入 | 官方规范哈希 | 与 manifest 比 |
|---|---|---|
| 本次 99,113 行 | `81f535e3…1643c` | 相等 |
| 本次**前 99,112 行**（删掉最后一行） | `ae13b8e060f9bc58d4bf1a4e575dd20aa867918b02a3cc53ced9ed51c05a6908` | 不等 |
| §4 的 17,843 行部分运行 | `b98665124646b18418975d9c0d04525567aa5655a5d9f4d5d9dab3a059b77d67` | 不等 |

**关于 `.tmp_v400_trace_full_1/report.json`**：它是**修正前判据**的产物，`status=mismatch`、`exitCode=6`、
`verdict.shaMatch=false`，理由是「文件 SHA 不等」。该文件**保持原样、未改写**（原始证据不被回填）。
修正后的独立核对写在同一目录的 `verification.json`（由 `.tmp_v400_trace_full_tools/analyze_trace.py`
读 rows/complete 后计算），字段为：
`rowsMatch=true`、`canonicalTraceSha256=81f535e3…1643c`、`shaMatch=true`、
`fileEqualsRetailTraceField=false`、`completeStageFramesMatch=true`、`completeEndScoresMatch=true`、
`completeTailsOk=true`、`rngOriginsComplete=true`。第二份回放用的是修正后的工具，故其 `report.json`
直接给出 `status=ok/exitCode=0`。

### 9.5 调试事件与时间线

| 时刻（s） | 事件 |
|---|---|
| 0.359 | 启动 `th08.exe`（复制到 `rundir`，零售原件未被写） |
| 1.979 | 窗口 640×480；**attestation 4/4 通过** |
| 1.980 | `DebugActiveProcess(22376)`，`dr7=0x415`（3 槽） |
| 2.520 | 已武装 15 线程（14 live） |
| 7.214 | 导航进入回放（`closed_loop`，8 次按键，无盲打回退） |
| 8.333 | stage 0 首行 |
| 107.199 / 255.097 / 409.807 / 649.702 / 884.250 | stage 1 / 2 / 4 / 5 / 6 首行 |
| **1192.265** | **`0x453080 deleteReplay` 命中**（`DeleteReplayManager`，回放自然结束） |
| 1192.275 | `rows=99113 errors=[]`，`complete.json` 落盘 |
| 1192.290 | `DebugActiveProcessStop` + 清 DR（19 个活线程），`detached=true` |
| 1200.302 | `WM_CLOSE` 8 s 无响应 → `terminate` |
| 1200.369 | 退出码 6（修正前判据），`elapsed=1200.368 s` |

事件统计（`report.json:hits`）：`frame=99,113`、`stageStart=99,607`（`OnUpdate` 60 Hz）、
`deleteReplay=1`；总命中 198,721，`resumeFlagSet=198,721 / failures=0`、
`singleStepUnarmed=0`、`unexpectedExceptions=[]`；`eventsHandled=198,915`。
即**每一次单步都精确归因到已武装地址、RF 每次都置位成功、没有丢事件**。
吞吐：99,113 行 / (1192.265−8.333) s ≈ **83.8 行/s**（末段 stage 6 降到 ≈72 行/s，与实体数量增加、帧率下降一致）。

### 9.6 第二份回放（可复现性）：**尝试被外部中断，2/108 未成立**

命令（修正后的工具，`Extra` 回放的 `startStage` 由官方规则取 stage index 8）：

```powershell
python tools\rl\v400_retail_trace_debugapi.py --phase rows --out .tmp_v400_trace_full_2 `
  --replay third_party\th08\tests\fixtures\replays\th08-1.00d\accepted\th8_ud2c7a.rpy `
  --start-stage 8 --budget-seconds 900
```

| 项 | 值 |
|---|---|
| manifest fixture | `extra-10-extra`，`calculationFrames=44,771`，`retailTrace.sha256=ddca7cf69f8517f95734d4651fd0fcca5ec26d09eb9807b4843fd8323e254b2c` |
| 导航 | **成功**：7.299 s 检测到回放（`stage=8`，`frame=0`），8.435 s 首行（`startStage=8` 的 Extra 光标命中） |
| 实际写到 | **28,680 行**（5,076,647 B），`frames` 1…28,680、全部 25 字段、纯 LF、`calc` 连续、无重复帧 |
| 结束状态 | **被外部终止**：`complete.json`/`report.json` **均未生成**（`debugger.log` 停在 8.435 s 的首行之后，符合播放期不写日志的预期；`rows.jsonl` 最后写入 10:08:44） |
| 工具退出码 | **无**（进程被终止，未走到 `finally`；后台包装层报 exit 1） |
| 我的残留进程 | `th08.exe` 21508 随调试器一起消失；**无我方残留**（零售 exe SHA 仍为 `330fbdbf…d924`） |

**为什么没有重试**：同一工作区里有**另一个并行实验**在同一时刻运行零售游戏——
`python tools/rl/v400_retail_trace_debugapi.py --phase rows --out .tmp_v400_trace_fmt1/full_extra2
--replay …/th8_ud2c65.rpy --start-stage 8 --budget-seconds 900`（python PID 6208，其 `th08.exe` PID 18384
自 10:09:31 运行）。零售游戏是**单实例 + 单窗口 + `SendInput` 前台焦点**的采集：两个 `rows` 运行并行会
互相抢焦点、互相污染 trace，且本工具启动时 `_running_instances()` 会直接以 exit 5 拒绝。
因此按要求「最多重试 1 次」的前提下，**本轮不重试**，也不去杀掉不属于本实验的 PID 18384。

**结论**：第二份回放**没有完成**，`2/108` **未成立**。可复现性（同一工具对第二份回放给出 manifest SHA）
本轮**未取得证据**；§9.7 的判定只覆盖 1/108。第二份回放在被终止前产出的 28,680 行内部自洽
（单一 stage 8、帧号连续无重复、字段宽度全为 25），但它**不是**一次完整采集，不能用于任何 SHA 主张。

### 9.7 判定

- **1/108 与真实零售位一致**（`easy-0-a` / `th8_ud2cd2.rpy`）：行数 = `calculationFrames` = 99,113、
  官方 `trace_sha256` = manifest `retailTrace.sha256`（`81f535e3…1643c`）、`complete.json` 的
  `stageFrames`/`endScores` 与 manifest 逐项相等、`ended=true`、`naturalReplayExit=true`。
- **2/108 未成立**（§9.6：第二份回放被外部终止，只写到 28,680/44,771 行，无 `complete.json`）。
  可复现性留待独占工作区时补跑。
- 单次连续运行即可覆盖 99,113 行，**不需要也不允许**分片拼接。
- 旧 §4 的「SHA 不等」结论应按 §9.4 修正为**判据错误**；数据本身没有为了凑哈希做任何改动。
- 本轮同时证明了两件此前未证的事：`0x453080 deleteReplay` 在**自然结束**时命中（1 次），
  以及 `complete.json` 的官方 `ended` 语义（自然结束必须 `ended=true`，不能记成 error）。

### 9.8 并行会话干扰（必须记录的环境事实）

本轮的两次游戏运行期间，**同一工作区里还有另一个会话在做同一个 G0.1 任务**：它使用
`.tmp_v400_trace_fmt1/full_extra2`、回放 `th8_ud2c65.rpy`、`--start-stage 8 --budget-seconds 900`，
python PID 6208 / 其 `th08.exe` PID 18384 自 **10:09:31** 起运行；同一时刻 `tests/test_v400_retail_trace_debugapi.py`
在 **10:07:51** 被追加了 3 个用例（`test_row_line_is_byte_identical_to_the_official_write_expression` 等）。

- 我的主运行（§9.1–9.5）在该干扰出现之前**已经完整结束**（工具 1192.265 s 收尾、1200.369 s 退出），
  因此 §9.2–9.5 的结论不受影响；运行的 `rows.jsonl`/`complete.json`/`report.json` 与零售 exe 均未被他人改写
  （零售 exe 事后复核仍为 `330fbdbf…d924`，`thirdPartyUnchanged=true`）。
- 我的第二份回放（§9.6）在该干扰下于 10:08:44 被**外部终止**，未重试——零售游戏单实例、单窗口、`SendInput`
  抢焦点，两路并行必然互相污染，且工具有 `_running_instances()` 单实例门。
- 因此**所有数字只归属于我这两次运行**：`1/108` 成立；`2/108` 未成立；不要把我未参与的那次 `full_extra2`
  运行与本文的数字混在一起引用。
