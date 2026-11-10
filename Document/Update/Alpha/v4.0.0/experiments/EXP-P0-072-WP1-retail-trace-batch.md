# EXP-P0-072-WP1 — 零售轨迹 108 份串行批跑驱动（可续跑）

- 日期：2026-11-09（Windows，Python 3.12.10 **64 位**，零售原件 `third_party/th08/resources/th08.exe` 840,704 B / SHA-256 `330fbdbf…d924`）
- Provider / model：DeepSeek / DeepSeek v4.1 flash
- 工具：`tools/rl/v400_retail_trace_batch.py`（**新增**），测试 `tests/test_v400_retail_trace_batch.py`（**新增**，27 passed）
- 观测器：`tools/rl/v400_retail_trace_debugapi.py` **未改动**（沿用 EXP-P0-068 的逐位一致路径）
- 输入：`third_party/th08/config/replay-fixtures.json`（108 条，`retailTrace.calculationFrames` 合计 **12,311,460** 帧）
- 判据：只认官方 `compare-replay-traces.trace_sha256()`（canonical sorted-schema JSON + 每行 25 个 `<I` LE u32），
  **不是** `rows.jsonl` 的文件 SHA-256
- 合规：未改 `third_party/th08`／零售 exe／依赖，未删除任何文件，未伪造 journal／report／SHA；
  每份前后各做一次「无残留 `th08.exe`」守卫。

---

## 0. 结论摘要

| 问题 | 结果 |
|---|---|
| 驱动是否可用 | **可用**。`--dry-run` 正确排出 108 份；单测 **27 passed**（含 stub 观测器端到端、守卫、续跑、退出码） |
| 串行约束 | 单进程、单子进程、单 `th08.exe`；每份前后各一次 `th08.exe` 守卫（等待 ≤120 s） |
| 续跑语义 | 只有 `ok` / `mismatch` 才算终态并跳过；`failed` 默认重试；重试用 `attempt-N` 子目录，**不删旧证据** |
| 已完成份数 | **1 / 108**（`easy-0-a` = `th8_ud2cd2`），`identical = 1`，`mismatch = 0`，`failed = 0` |
| 第一份是否逐位一致 | **是**。rows = 99,113 = `calculationFrames`；canonical = `81f535e3…1643c` = manifest；`sha_equal=true` |
| 单份实测耗时 | **1200.7 s**（驱动墙钟 1200.718 s，观测器自报 1200.675 s） |
| 串行 108 份估算 | **≈34–41 h**（按实测 82.5 行/s 全流程 → ≈41.4 h；按播放稳态 102 行/s → ≈33.5 h） |
| 本实验最终状态 | **partial 且已移交**：10:48 起另一路**已获授权的 2 实例并行实验**接管主机，本批跑按要求停止，不再重启 |

> 移交说明：串行 36 h 对本次交付窗口不可接受，主代理已裁决改走「串行导航 + 并行播放」流水线并行路线
> （`.tmp_v400_trace_par2`）。本卡的驱动、续跑语义、journal/report 契约与已完成的 1 份证据保留，
> 供并行驱动复用；**本卡不是 108/108 的完成报告**。

---

## 1. 驱动设计

### 1.1 调用链

每份 fixture 一次独立子进程，参数固定为：

```powershell
python -u tools\rl\v400_retail_trace_debugapi.py `
  --phase rows `
  --out .tmp_v400_trace_batch\<fixture_id> `
  --replay third_party\tests\fixtures\replays\th08-1.00d\accepted\<file> `
  --budget-seconds 1800
```

- **不带** `--with-score-dat`：观测器默认就是「不播种 score.dat」（等价于任务要求的 `--no-score-dat`）；
  驱动从不下发该开关，`test_observer_command_contract` 显式断言 `--with-score-dat` 不在命令行里。
- 每份使用**全新私有 out 目录 + 私有 `rundir`**，因此每份的运行条件与 EXP-P0-068 那份逐位一致的
  `th8_ud2cd2` 单跑同构（没有共享 score.dat／共享 rundir 带来的隐性状态）。
- 子进程 `cwd=D:\WorkShop\Touhou_RL`，环境剔除 `PYTHONPATH`（观测器自己 `sys.path.insert(ROOT)`），
  `PYTHONIOENCODING=utf-8`。
- Python 侧硬超时 = `--budget-seconds + --wallclock-slack`（默认 1800+900 s），只作「观测器没退出」的兜底；
  正常路径由观测器自己的 `--budget-seconds` 收尾。

### 1.2 目录布局

```
.tmp_v400_trace_batch/
  batch.log                     # 追加式会话日志（每次启动追加，保留重启历史）
  journal.jsonl                 # 每份结束立即 append + fsync
  report.json                   # 每份结束后原子重写（<name>.tmp → os.replace）
  heartbeat.json                # 每 30 s 原子重写：在飞 fixture、已采行数、批次计数
  <fixture_id>/                 # 第一次尝试
    rows.jsonl complete.json debugger.log status.json report.json rundir/
  <fixture_id>/attempt-2/       # 重试（不删 attempt-1）
```

### 1.3 journal 字段

每份结束**立即**追加一行并 `os.fsync`（`append_journal`：`flush()` + `os.fsync(fileno)`；
`test_append_journal_fsyncs_each_entry` 用计数器断言每条一次 fsync）：

| 字段 | 含义 |
|---|---|
| `fixture_id` | manifest `id` |
| `file` | 回放文件名 |
| `manifest_sha256` | **manifest `retailTrace.sha256`**（canonical 期望值，比较对象） |
| `actual_sha256` | 实测 **canonical** 轨迹哈希（`compare-replay-traces.trace_sha256`） |
| `rows` / `calculationFrames` | 实测行数 / manifest 帧数 |
| `rows_equal` / `sha_equal` | 行数判据 / canonical 判据 |
| `elapsed_s` | 驱动侧墙钟（含子进程启动与收尾） |
| `exit_code` | 观测器退出码（0=ok，6=mismatch/不完整，5=有残留实例，7=超时…） |
| `error` | 失败／不符原因（取自观测器 `report.json:result.reason`） |

附加可审计字段（不参与判据）：`status`、`attempt`、`out`、`replay`、`replay_sha256`（**.rpy 文件** SHA-256）、
`manifest_retail_trace_sha256`、`rows_file_sha256`（**`rows.jsonl` 文件** SHA-256，明确不是 canonical）、
`stage_frames_equal`、`end_scores_equal`、`ended`、`natural_replay_exit`、`tool_elapsed_s`、`timed_out`、
`guard_before`、`guard_after`、`beganAt`、`endedAt`。

> 三个 SHA 严格区分：`replay_sha256` = 回放文件、`rows_file_sha256` = 行文件字节、
> `actual_sha256` = **官方 canonical 轨迹哈希**。只有第三个参与判据。
> （`easy-0-a` 实测：`replay_sha256=6e80fc7b…6146`、`rows_file_sha256=ed07eeb7…0c4`、
> `actual_sha256=81f535e3…1643c`——三者互不相同，正好证明没有拿文件 SHA 冒充 canonical。）

### 1.4 状态与退出码

| journal `status` | 判定 | 续跑 |
|---|---|---|
| `ok` | `exit_code=0` 且 `rows_equal` 且 `sha_equal` | **跳过** |
| `mismatch` | `rows_equal` 但 `sha_equal=false`（有完整轨迹、哈希不同） | **跳过** |
| `failed` | 行数不足、崩溃、超时、残留实例、缺回放… | 重试（`--skip-failed` 可关闭） |

`report.json` 的退出码：`0` = 108/108 全 identical；`3` = partial（还有 remaining）；
`1` = 出现 mismatch 或 failed（优先于 3）。

### 1.5 report.json / heartbeat.json

`report.json` 每份结束后原子重写，字段：`total / compared / identical / mismatch / failed / remaining`、
`totalRowsPerSide`（`captured`、`expected`、`allManifest`，并写明定义）、
`timing`（`totalElapsedSeconds`、`averageSecondsPerFixture`、`averageRowsPerSecond`、
`projectedRemainingSecondsByFixture`、`projectedRemainingSecondsByRows`）、
`mismatches`（两侧 SHA）、`failures`、`journalFieldNotes`、`fixtures`（每份明细）、`state`、`abortedReason`。

`heartbeat.json` 每 30 s 重写：`fixtureId`、`rowsSoFar`、`rowsBytes`、`fractionOfFixture`、`counts`、`timing`、
`batchElapsedSeconds`，并透传观测器 `status.json` 的 `phase / rows / gamePid / foregroundOk /
lastDebugEventAgeSeconds`（进度检查不必读 18 MB 的 rows 文件）。

---

## 2. 续跑语义

1. 启动时读 `journal.jsonl`，`terminal_entries()` 取每份**最后一条** `ok`/`mismatch` 作为终态集合。
2. 终态集合里的 fixture 直接跳过（日志打印 `resume: N terminal of 108`）。
3. `failed` 的 fixture 重新排队，`next_attempt()` 给出 `attempt-2` 新目录，旧目录原样保留。
4. 每份结束立刻 `fsync` journal，然后才重写 report ⇒ 进程被强杀时最多丢「正在跑的那一份」的记账，
   已完成份数不会回退。
5. 重启即续跑：同一个 `--root` 再启动一次即可。

**生产环境实测过的续跑**（本实验唯一一次重启，§5.2）：

```
[2026-11-09T10:46:48] resume: 1 terminal of 108; pending 107, this session 107
[2026-11-09T10:46:48] --- fixture 1/107 easy-0-b (th8_ud2cdc.rpy, 107296 frames)
[2026-11-09T10:46:48] [easy-0-b] attempt 2 -> .tmp_v400_trace_batch\easy-0-b\attempt-2
```

- `easy-0-a` 被正确跳过（没有重建目录、没有重跑）；
- 上一轮被打断的 `easy-0-b`（attempt-1 留下 `rows.jsonl` / `status.json`，`rows=17616`）**没有被删除**，
  续跑自动进入 `attempt-2`。

测试覆盖：`test_main_resumes_and_skips_ok_and_mismatch`、`test_terminal_entries_only_skip_ok_and_mismatch`、
`test_next_attempt_never_deletes`、`test_run_one_retry_uses_a_new_attempt_directory`。

---

## 3. 串行约束（硬）

上一轮两路并行导致其中一份被终止（EXP-P0-068 §12）：零售件是**单实例 + 单窗口 + `SendInput` 前台焦点**，
两路并行会互抢键盘焦点与窗口。因此本驱动：

- 单线程主循环，逐份 `subprocess.run(...)`，同一时刻最多一个 `th08.exe`；
- **每份开始前**：`wait_for_no_instances(120 s, poll 5 s)`，发现残留 `th08.exe` 就等；
  120 s 仍不消失 ⇒ 记 `failed`（`error` 写明 pids）并**中止该份**（不启动观测器、不建目录）；
- **每份结束后**：同样检查一次；若仍有残留，立即停止整批（`abortedReason`），不让下一份再空等 120 s；
- 连续 `--max-guard-failures`（默认 2）次前置守卫失败 ⇒ 停止整批，避免 108×120 s 空转；
- 驱动自身不碰窗口、不注入按键；进度检查只读 `journal.jsonl` / `report.json` / `heartbeat.json`。

测试覆盖：`test_run_one_guard_blocks_before_creating_a_directory`、
`test_main_stops_when_a_residual_instance_never_goes_away`、`test_wait_for_no_instances_*`。

---

## 4. 性能：为什么没有改观测器

- 单份速率由**游戏实时推进速度**决定（每帧一次硬件断点停–读–续），驱动不是瓶颈
  （`easy-0-a` 全程驱动进程 CPU 仅 **0.08 s**，`Get-Process` 累计）。
- 播放稳态实测 **≈102 行/s**（heartbeat 两点采样：19,860 → 25,980 行 / 60 s；
  另有 60 s 处 6,480 行、692 s 处 62,160 行的一致斜率）。
- 观测器每帧约 9 次 `ReadProcessMemory` + 1 次写行，Python 侧开销约 0.5 ms 量级，
  而实测每帧约 9.8 ms（含导航与收尾的整段均值为 12.1 ms/行）⇒ Python 开销占比 <10%。
  压到 0 也只能换来 <10%，却要冒「行内容／行序被改」的风险。
- 因此本实验**未修改** `v400_retail_trace_debugapi.py`，也没有做「优化后复验短份」的动作——没有优化需要复验。

> 结论：单进程串行模式的墙钟下界由「12,311,460 帧 ÷ 约 100 行/s」决定，约 **34 h**；
> 这正是本路线被并行模式接管的原因（§5.3）。

---

## 5. 运行记录（时间线）

### 5.1 第一份：`easy-0-a`（`th8_ud2cd2.rpy`）——**逐位一致**

| 项 | 值 |
|---|---|
| fixture / 文件 | `easy-0-a` / `th8_ud2cd2.rpy`（`replay_sha256=6e80fc7b…6146`） |
| 开始 / 结束 | 10:23:23 / **10:43:24** |
| 行数 | **99,113**（= manifest `calculationFrames`），`rows_equal=true` |
| canonical 轨迹哈希 | **`81f535e3e01743255fe1edbe71740f0b27050702ce4d28848ce934d38be1643c`** |
| manifest `retailTrace.sha256` | 同上，`sha_equal=true` |
| 退出码 | **0**（`status=ok`） |
| 耗时 | **1200.718 s**（观测器自报 1200.675 s） |
| 结束方式 | `ended=true`、`naturalReplayExit=true`、`errors=[]` |
| 守卫 | 前 `waitedSeconds=0.094`、后 `0.109`，前后均 `ok`（无残留实例） |
| 第三方未改动 | `report.json: thirdPartyUnchanged=true` |
| 文件 SHA（仅记录，非判据） | `rows.jsonl` = `ed07eeb7…0c4`（18,043,248 B） |

逐阶段（`report.json:verdict.stageTable`，全部 `tailOk=true` / `endScoreMatch=true`）：

| stage | rows | lastFrame | unconsumed | expectedTail | endScore |
|---|---|---|---|---|---|
| 0 | 9,463 | 9,387 | 2 | 2–3 | 2,447,070 ✅ |
| 1 | 13,192 | 13,116 | 2 | 2–3 | 10,352,151 ✅ |
| 2 | 13,242 | 13,173 | 2 | 2–3 | 23,602,972 ✅ |
| 4 | 20,212 | 20,121 | 2 | 2–3 | 42,644,410 ✅ |
| 5 | 20,277 | 20,156 | 2 | 2–3 | 59,772,037 ✅ |
| 6 | 22,727 | 22,286 | 6 | 6–7 | 90,941,860 ✅ |

> 与 EXP-P0-068 的独立核对结果（`.tmp_v400_trace_full_1/verification.json`）
> **在 rows、canonical SHA、stageFrames、endScores、tail、rngOrigins 上全部一致**：
> 同一份零售原件、同一条回放、同一套读点，批跑驱动没有引入任何差异。

### 5.2 辅助字段 `stage_frames_equal` 的定义错误与修正

- **错误版本**（只影响本实验的前 1 条 journal 记录）：把 `sum(complete.stageFrames) == complete.rows`
  当作「stageFrames 对齐」。实测 `sum(stageFrames)=98,239`、`rows=99,113`，差 **874 行**，
  于是这条记录写成 `stage_frames_equal=false`。
- **真实原因**：`stageFrames` 记录的是每个阶段**最后一个 calculation frame**，而 `rows` 计的是
  **命中次数**——同一帧可能被重复命中（`verification.json: duplicateRowsTotal=874`：
  76+76+69+91+121+441）。**重复命中是官方观测器本来就有的性质**，`sum(stageFrames) != rows` 是正常现象，
  不是差异。
- **修正**（10:46 起生效）：改用官方不变式——每个阶段 `inputRecords − stageFrames` 必须落在
  `valid_input_tail` 窗口（非末阶段 2/3、末阶段 6/7）；若 `verdict.stageTable` 在，就直接用观测器自己
  逐阶段的 `tailOk` / `endScoreMatch`（`_stage_tails_ok()` 作为无 verdict 时的回退）。
  单测：`test_stage_tails_follow_the_official_invariant`、`test_run_one_ok_records_both_hash_kinds`。
- **对判据无影响**：`status` 只由 `exit_code`、`rows_equal`、`sha_equal` 决定；`stage_frames_equal`
  与 `end_scores_equal` 只是附加审计字段，不参与 canonical SHA 判定。`easy-0-a` 的 `status=ok`、
  `sha_equal=true` 不受该字段影响；其权威证据是同目录 `report.json:verdict.stageTable`（上表，全 true）。

### 5.3 被并行实验接管（本实验的终止原因）

| 时间 | 事件 |
|---|---|
| 10:23:23 | 批跑启动（PID **14884**），第 1 份 `easy-0-a` |
| 10:43:24 | `easy-0-a` **ok**，1/108 identical；第 2 份 `easy-0-b` attempt-1 开始 |
| 10:46:17 | 为落地 §5.2 的字段修正，主动停止批跑树（batch 14884 / observer 22304 / 其 `th08.exe`）；确认无残留 `th08.exe`；`easy-0-b` attempt-1 的部分证据（`status.json: rows=17616`）**原样保留** |
| 10:46:48 | 重启（PID **22704**）：`resume: 1 terminal of 108`，`easy-0-b → attempt-2` |
| 10:48:47 | **另一路 agent** 启动 `.tmp_v400_trace_par2/harness.py --tag p2 --budget 150 --instances extra-1-extra,extra-2-extra`（pid 9184），同时起 **2 个 `th08.exe`**（18756 / 20792），用 `--allow-existing-instances --nav-lock` |
| 10:49 | 批跑 22704 及其观测器（15888）、游戏（22148）**被外部强杀**：`batch.log` 无 session-end 行、无 stderr、journal 无新记录 |
| 10:52:44 | 核实：22704 已死；`journal.jsonl` 仍 **1 行**（`easy-0-a ok`）——已完成份数**未回退** |
| 10:52:55 | 主代理裁决：`trace_par2` 并行实验是**授权的**，本批跑停止、不再重启，所有权移交并行驱动 |

并行路实测量（只读观察，10:48:47–10:52:55）：

| 实例 | phase | rows（4 分钟后） | 说明 |
|---|---|---|---|
| `extra-1-extra` | `waiting-for-nav-lock`（status 停在 10:48:50） | **0** | 已拿到 nav.lock，但 4 分钟内无新行 |
| `extra-2-extra` | `waiting-for-nav-lock`（status 更新到 10:52:50） | **0** | 等锁 |

> **诚实结论：这一轮 2 实例并行没有可用的吞吐数字**——4 分 8 秒内两侧 `rows.jsonl` 都是 0 B，
> 都停在 `waiting-for-nav-lock`（单窗口 + 前台焦点仍是瓶颈，与 EXP-P0-068 §12 的判断一致）。
> 我无法据此声称并行更快；并行可行性需由接手的那路 agent 用自己的证据证明。

---

## 6. 进度表（本实验的真实进度）

| 检查时间 | 已完成 | identical | mismatch | failed | 剩余 | 均耗时 | 备注 |
|---|---|---|---|---|---|---|---|
| 10:43:24 | 1 / 108 | 1 | 0 | 0 | 107 | 1200.7 s | `easy-0-a` ok；`rows=99,113`，canonical = `81f535e3…1643c` |
| 10:49:35 | 1 / 108 | 1 | 0 | 0 | 107 | 1200.7 s | 批跑 22704 被并行实验强杀；journal 仍 1 行 |
| 10:52:55 | 1 / 108 | 1 | 0 | 0 | 107 | 1200.7 s | 按要求停止，所有权移交并行驱动 |

### 最终统计（partial）

| 项 | 值 |
|---|---|
| total | 108 |
| **compared** | **1** |
| **identical** | **1**（`easy-0-a`） |
| mismatch | 0 |
| failed | 0 |
| **remaining** | **107** |
| totalRowsPerSide | captured **99,113** / expected **99,113**（同一份）；allManifest 12,311,460 |
| 累计耗时 | **1200.718 s**（1 份）；另有 `easy-0-b` attempt-1 的 172.8 s 被丢弃重跑 |
| 单份均耗时 | **1200.7 s**（n=1，实测） |
| 串行 108 份估算 | 12,311,460 帧 ÷ 82.5 行/s ≈ **149,000 s ≈ 41.4 h**（全流程均值）；÷102 行/s ≈ **33.5 h**（播放稳态） |
| journal 行数 | **1** |
| report.json | `state=running`（进程已死，未被改写为 finished），`identical=1`、`remaining=107` |

**已完成的 fixture 列表**：`easy-0-a`（`th8_ud2cd2.rpy`，99,113 行，canonical `81f535e3…1643c`，1200.7 s，ok）。

---

## 7. 变更文件清单

| 文件 | 类型 | 说明 |
|---|---|---|
| `tools/rl/v400_retail_trace_batch.py` | **新增**（约 900 行） | 串行批跑驱动：fixture 加载、`th08.exe` 守卫、journal(fsync)、report(原子)、heartbeat、续跑、退出码 |
| `tests/test_v400_retail_trace_batch.py` | **新增**（27 tests） | 纯单测 + stub 观测器端到端；**不启动任何零售实例**（`INSTANCE_PROBE` 注入、`--observer` 指向 stub） |
| `tools/rl/v400_retail_trace_debugapi.py` | **未改动** | 沿用 EXP-P0-068 实现 |
| `Document/Update/Alpha/v4.0.0/experiments/EXP-P0-072-WP1-retail-trace-batch.md` | **新增** | 本卡 |
| `.tmp_v400_trace_batch/**` | 运行产物 | `batch.log`、`journal.jsonl`(1 行)、`report.json`、`heartbeat.json`、`easy-0-a/`、`easy-0-b/`(attempt-1 部分证据) |

---

## 8. 验收命令

```powershell
# 驱动单测（27 passed，含 stub 观测器端到端、守卫、续跑、退出码；不启动零售实例）
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_batch_pytest -q tests/test_v400_retail_trace_batch.py

# 观测器单测未改动，回归 33 passed
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_batch_pytest -q tests/test_v400_retail_trace_debugapi.py

# 计划（不产生任何运行）
python tools\rl\v400_retail_trace_batch.py --dry-run --first easy-0-a --limit 4 --root .tmp_v400_trace_batch

# 进度（只读）
Get-Content .tmp_v400_trace_batch\journal.jsonl
Get-Content .tmp_v400_trace_batch\report.json -Raw
Get-Content .tmp_v400_trace_batch\heartbeat.json -Raw

# 复核第 1 份的 canonical SHA（与 journal 的 actual_sha256 对照）
python -c "import json,importlib.util,sys;sys.path.insert(0,r'third_party/th08/scripts');spec=importlib.util.spec_from_file_location('o','tools/rl/v400_retail_trace_debugapi.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);print(m.canonical_trace_sha256('.tmp_v400_trace_batch/easy-0-a/rows.jsonl'))"
```

---

## 9. 遗留与交接

1. **驱动可直接复用**：并行驱动（`.tmp_v400_trace_par2`）若要沿用本卡的 journal / report 契约，
   `append_journal()`、`terminal_entries()`、`summarize()`、`exit_code_for()`、`_stage_tails_ok()`
   都是纯函数，可直接 import，无需复制逻辑。
2. **并行前必须解决的实测阻塞**：本轮 2 实例在 4 分 8 秒内 **0 行**，都停在 `waiting-for-nav-lock`。
   单窗口前台焦点是硬约束，需要「导航窗口串行 + 播放窗口并行」或「每实例独立桌面/会话」之一，
   且必须有**每实例的 canonical SHA 证据**才能声称等价。
3. **预算边界**：最大 fixture `calculationFrames=149,986`，按 102 行/s 需 1470 s，加导航收尾约 1570 s；
   `--budget-seconds 1800` 足够但余量只有 ~15%。更大语料需重新评估。
4. **`easy-0-b` attempt-1** 是被我主动终止的部分运行（`rows=17616`，无 report），
   保留为证据，**不得**当作采集结果使用；并行接管后如需重跑，仍走 `attempt-2` 语义。
