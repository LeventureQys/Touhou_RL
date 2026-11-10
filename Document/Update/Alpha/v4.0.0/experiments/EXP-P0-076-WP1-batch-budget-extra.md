# EXP-P0-076（v4.0.0 WP1）：批跑预算自适应、Extra 导航前置与超时脏状态

- 日期：2026-11-09 13:10 → 13:25（本卡写于 13:25）
- 前置：EXP-P0-075 §7（风险 1/2）、§10（后续卡 3/4）；EXP-P0-074 §11.2；EXP-P0-068 §9.6
- 触发：G0.1 批跑（`tools/rl/v400_retail_trace_batch.py`，pid 22328，`--budget-seconds 1800` 固定）里有两类**已知会失败**的 fixture：
  1. **12 份 >135k 帧**按实测 61.5–100 行/s 需 2,195–2,439 s，固定 1800 s 预算**必然截断** → 行数不足 → `mismatch`；
  2. **12 份 Extra（stage 集合 `[8]`）**在批跑默认 `start_stage=0` 下走 40 s 闭环保守超时 + 35 s 盲序列回退（实测 **47.663 s**），`--start-stage 8` 只需 **7.3 s**。
- 目标：预算按 fixture 自适应；Extra 导航前置；超时不再被当成成功；全部先做**代码 + 单测 + 离线验证**，不触碰在跑的批跑。
- 硬约束：不改 `third_party/th08` / 零售 exe / 依赖；不伪造预算/ETA；不终止正在跑的批跑；**不启第二个零售实例**（本卡全部工作都在无零售进程下完成）。
- 本卡只动：`tools/rl/v400_retail_trace_batch.py`、`tools/rl/v400_retail_trace_debugapi.py`（**仅** `MAX_BUDGET_SECONDS` 常量 = 参数/预算传递）、`tests/test_v400_retail_trace_batch.py`、`tests/test_v400_retail_trace_debugapi.py`、本卡、`.tmp_v400_budget/*`。

## 1. 结论（TL;DR）

| 项 | 结果 |
| --- | --- |
| 预算公式 | **`budget = clamp(ceil(calculationFrames / 50) + 300, 1800, 5400)`**（`--budget-seconds` 降级为**下限**，`--budget-max-seconds` 默认 5400） |
| 受影响面 | **97/108** 份被抬预算（`frames > 75,000` 即抬），最大 **3300 s**，**无**一份触及 5400 s 上限 |
| 12 份大 fixture | 3015 / 3043 / 3095 / 3120 / 3174 / 3176 / 3203 / 3244 / 3270 / 3275 / 3280 / **3300** s（见 §3） |
| Extra 注入 | 每份 fixture 都显式传 `--start-stage <manifest 首 stage index>`：普通件 `0`（与原默认等价）、**12 份 Extra 为 `8`** |
| Extra 语义依据 | 官方 `capture-replay.py` 把 `startStage` 默认成 `replay["stages"][0]["index"]`；`replay-external-observer.menu()` 用 `cursor == startStage` 选回放菜单项 → `[8]` 的正确值就是 **8** |
| Extra 证据 | `--start-stage 8`：**7.264 / 7.289 s** 进回放（两份历史日志）；`--start-stage 0`：40 s 超时 + 盲序列，**47.663 s**；两者前 34,200 行**逐字节相同** |
| 超时语义 | `timed_out=true` → `status="timeout"`，**永不**判 ok（即使 rows/hash 恰好相等），`rowsAreEvidenceOnly=true`；timeout 与 failed 一样**可重试一次**（`--max-attempts 2`） |
| 单测 | `test_v400_retail_trace_batch.py` **44 passed**（31→44，+13）；`test_v400_retail_trace_debugapi.py` **45 passed**（44→45，+1）；合计 **89 passed**（75→89，+14），`compileall` 退出码 **0** |
| 在跑的批跑 | **未触碰**：pid 22328 仍在跑（13:24:30：`easy-2-b` attempt-2 83,968/98,098 = 85.6%），`th08.exe` 计数 1；journal 仍 **4 行** |
| 能否重启 | **可以**，但要在 `easy-2-b` attempt-2 收尾（约 13:28–13:29）之后、且用 **STOP 文件** 让旧批跑自己退出（见 §8），避免留下 observer/游戏孤儿 |
| ETA | 行数投影**不变**（42–54 h，2026-11-11 早上到傍晚）；但旧方案里 12 份大 fixture **永远跑不完**，新方案把它们的 43,200 s「白跑两次 1800 s」换成约 28,300 s 一次性出结果 → 关键路径 **≈ −4.1 h**，另加 Extra 导航 **−8 min** |

一句话：**预算从"固定 1800 s"改成"按帧数自适应且有 5400 s 硬上限"，Extra 从"靠盲序列回退"改成"按官方 startStage=8 走闭环"，超时从"可能被 rows 洗白"改成"fail-closed 的 timeout"。**

## 2. 预算公式与实现（`tools/rl/v400_retail_trace_batch.py`）

### 2.1 公式

```
budget_seconds(frames) = min( max(1800, ceil(frames / 50) + 300), 5400 )
```

| 参数 | 值 | 依据 |
| --- | --- | --- |
| 行/s 下限 `BUDGET_ROWS_PER_SECOND` | **50** | 实测 61.5 行/s（EXP-P0-075 §6，`easy-2-b` attempt-2）、79–100 行/s（EXP-P0-074 §9）、83.8 行/s（EXP-P0-068 §9.5）。取 50 是为了让"按实测速率跑"的件留 19–100 % 余量 |
| 固定余量 `BUDGET_MARGIN_SECONDS` | **300 s** | attach ≈ 3 s + 导航 7–48 s + detach ≈ 8 s + 抖动 |
| 下限 `BUDGET_MIN_SECONDS` | **1800 s** | G0.1 授权口径（EXP-P0-068：单次连续运行，最长 1800 s） |
| 上限 `BUDGET_MAX_SECONDS` | **5400 s** | 一份卡死件不能吃掉整晚；同时与 observer 的 `MAX_BUDGET_SECONDS` **相等**（单测断言两者一致，避免"批跑计划了一个 observer 会拒绝的参数"） |

边界性质（都有单测）：

- 抬升阈值：`ceil(frames/50) + 300 > 1800 ⟺ frames > 75,000`；`frames = 75,000` 正好落在 1800（不抬）。
- `frames` 为 `None` / 0 / 负数 → 1800（不抛异常）。
- 上限**永不低于下限**：`fixture_budget_seconds(400_000, minimum=2000, maximum=1500) == 2000`。
- 用实测 61.5 行/s 计算时：`149,986 → 2,739 s`（比 50 行/s 口径短 561 s），说明 50 是保守值。

### 2.2 落地位置

| 位置 | 变化 |
| --- | --- |
| CLI | `--budget-seconds`（默认 1800，语义改为**下限**）、新增 `--budget-max-seconds`（默认 5400，`1..5400`，且必须 ≥ 下限） |
| `fixture_budget_seconds()` | 纯函数，公式的唯一实现 |
| `budget_policy()` | 规则文本 + 参数，写进 `report.json.budgetPolicy`、`heartbeat.json.budgetPolicy` |
| `run_one()` | 按 fixture 计算 `budget`，传 `--budget-seconds <budget>`；子进程墙钟上限 = `budget + --wallclock-slack`；entry 记 `budgetSeconds` / `budgetBaseSeconds` / `budgetMaxSeconds` / `wallclockGuardSeconds` |
| journal | 每条 `budgetSeconds` |
| report | `fixtures[].budgetSeconds`、`budgetPlan{plannedSeconds, plannedSecondsForPending, fixturesAboveBase, uncappedFixedBudgetSeconds}`、`budgetPolicy` |
| heartbeat | 当前件的 `budgetSeconds` + `budgetPolicy` + `budgetPlan` |
| 启动日志 | `budget: floor=1800 s cap=5400 s formula=max(floor, ceil(frames/50)+300); raised for 97 of 108 fixtures, max=3300 s` |
| `v400_retail_trace_debugapi.py` | `MAX_BUDGET_SECONDS: 1800.0 → 5400.0`（**只改这个常量**；不传就更宽的上限，采集语义零改动） |

## 3. 12 份 >135k 帧：预算表与影响

`third_party/th08/config/replay-fixtures.json` 里正好 12 份 >135k（与 EXP-P0-075 §7 清单逐项一致），合计 **1,729,405 帧**：

| # | fixture | file | calculationFrames | 纯播放 @61.5 行/s | 旧预算 | **新预算** | 余量 (@61.5) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `lunatic-3-b` | `th8_ud2cd4.rpy` | 149,986 | 2,439 s | 1800 | **3,300 s** | 861 s |
| 2 | `lunatic-8-a` | `th8_ud2c8b.rpy` | 148,988 | 2,423 s | 1800 | **3,280 s** | 857 s |
| 3 | `normal-8-b` | `th8_ud2c2c.rpy` | 148,719 | 2,418 s | 1800 | **3,275 s** | 857 s |
| 4 | `normal-8-a` | `th8_ud2c2d.rpy` | 148,475 | 2,414 s | 1800 | **3,270 s** | 856 s |
| 5 | `lunatic-10-a` | `th8_ud2ca6.rpy` | 147,168 | 2,393 s | 1800 | **3,244 s** | 851 s |
| 6 | `lunatic-11-b` | `th8_ud2c7b.rpy` | 145,103 | 2,359 s | 1800 | **3,203 s** | 844 s |
| 7 | `hard-8-b` | `th8_ud24f9.rpy` | 143,785 | 2,338 s | 1800 | **3,176 s** | 838 s |
| 8 | `lunatic-8-b` | `th8_ud2cc9.rpy` | 143,679 | 2,336 s | 1800 | **3,174 s** | 838 s |
| 9 | `hard-8-a` | `th8_ud262e.rpy` | 140,962 | 2,292 s | 1800 | **3,120 s** | 828 s |
| 10 | `hard-1-b` | `th8_ud2933.rpy` | 139,702 | 2,272 s | 1800 | **3,095 s** | 823 s |
| 11 | `lunatic-4-b` | `th8_ud2bf5.rpy` | 137,112 | 2,229 s | 1800 | **3,043 s** | 814 s |
| 12 | `normal-6-b` | `th8_ud2922.rpy` | 135,726 | 2,207 s | 1800 | **3,015 s** | 808 s |

**为什么固定 1800 s 必错**：截断发生在预算到点，observer 优雅退出（`status=partial`、exit 6），行数 < `calculationFrames` → 批跑判 `failed`（"incomplete trace"）→ 旧规则重试一次 → 再截断一次 → exhausted。即**每份 2×1800 = 3,600 s 全部白跑**，12 份合计 **43,200 s ≈ 12 h**，且 12 份永远拿不到结果。

**不止这 12 份**（这是公式要抬 97 份的原因）：

| 口径 | 纯行时间 >1800 s 的份数 | 阈值 |
| --- | ---: | --- |
| @61.5 行/s（实测下限） | **75/108** | frames > 110,700 |
| @79 行/s（历史均值） | 8/108 | frames > 142,200 |
| @100 行/s（快件） | 0/108 | — |

也就是说"恰好 12 份"只是 **>135k 的必错子集**；在 61.5 行/s 这个实测速率下，75 份都会撞 1800 s。公式按 50 行/s + 300 s 余量把它们全部覆盖到，实测速率跑时**预算不会成为耗时**（上限不是时长）。

## 4. 12 份 Extra（`[8]`）：参数注入方式与实测对比

### 4.1 语义（来自官方实现，不是猜的）

| 证据 | 内容 |
| --- | --- |
| `third_party/th08/scripts/capture-replay.py:157` | `args.start_stage = args.start_stage if args.start_stage is not None else replay["stages"][0]["index"]`（且要求 `start_stage ∈ replay.stages`） |
| `third_party/th08/scripts/replay-external-observer.py:136` | 回放菜单（`screen=7, substate=2`）按键 = `"z" if cursor == settings["startStage"] else "Down"` |
| `tools/rl/v400_retail_trace_debugapi.py:680` | 本地 `menu_key()` 与官方逐字同构 |

结论：`--start-stage` 是**官方 `startStage`**，既是"从哪个 stage 开始期望行"，也是"回放菜单光标要停在哪个位置"。`stages=[8]` 的 Extra 件正确答案就是 **8**，不是 0。

### 4.2 注入实现

```python
start_stage = start_stage_for(fixture)   # manifest 的第一个 stage index；无 stages 时 0
command = [..., "--budget-seconds", f"{budget:g}", "--start-stage", str(start_stage)]
```

- **每份**都显式传（普通件 `--start-stage 0` 与旧默认完全等价，日志/journal 可审计）；
- `--phase rows` 的 `TraceCollector.start_stage` 与 `_grade` 的 `sequence` 用同一个值，`[8]` 件期望序列仍是 `[8]`、末 stage tail 仍是 `(6,7)`；
- 单测断言：`12 份 extra → 8`、其余 `→ 0`、且 `start_stage ∈ manifest stages`。

### 4.3 实测：7.3 s vs 47.7 s（都是真实零售运行的日志）

| 运行 | 参数 | 导航路径 | 进回放 | 结果 |
| --- | --- | --- | ---: | --- |
| `.tmp_v400_trace_fmt1/full_extra2` | **`--start-stage 8`**（`th8_ud2c65.rpy`，`stages=[8]`） | `closed_loop`（日志无"blind"行） | **7.289 s** | 被外部终止：34,200/44,554 行，无 report |
| `.tmp_v400_trace_fmt1/full_extra2b` | **`--start-stage 8`** | `closed_loop` | **7.264 s** | 被终止：3,769 行 |
| `.tmp_v400_obsopt/r3_full` | **`--start-stage 0`**（默认；同一 replay） | `closed_loop` 40 s 超时 → **blind** | **47.663 s**（42.683 s 回退 + 4.98 s） | **完整 44,554 行，`status=ok`，canonical sha = manifest** |

**两条关键复核（本卡离线新做，只读）**：

1. `r3_full` 的 canonical sha = `d48351bf…58765f` = manifest `retailTrace.sha256` → **盲序列回退路径本身是能出正确 trace 的**，所以问题 2 是**墙钟与"依赖回退"**问题，不是已证明的正确性缺陷。用户前提"从未真实跑过"应修正为：**批跑没跑过 `[8]` 件，`--start-stage 8` 的闭环路径也没有一次完整跑完过**；默认 `start_stage=0` 的回退路径在批跑外被 EXP-P0-074 完整跑通过一次。
2. `full_extra2`（`--start-stage 8`）的前 **34,200 行与 `r3_full`（`--start-stage 0`）逐字节相同** → **`--start-stage 8` 不改变 trace 内容**，只改变导航耗时与所用的分支。
3. 12 份 Extra 参数与预算（`stage` 全为 `[8]`，合计 599,995 帧）：

| # | fixture | file | frames | budget | 旧 `--start-stage` | **新 `--start-stage`** |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `extra-0-extra` | `th8_ud2cd8.rpy` | 49,029 | 1800 | 0 | **8** |
| 2 | `extra-1-extra` | `th8_ud2c67.rpy` | 44,106 | 1800 | 0 | **8** |
| 3 | `extra-2-extra` | `th8_ud2c65.rpy` | 44,554 | 1800 | 0 | **8** |
| 4 | `extra-3-extra` | `th8_ud2ce0.rpy` | 45,822 | 1800 | 0 | **8** |
| 5 | `extra-4-extra` | `th8_ud2ce4.rpy` | 77,082 | **1842** | 0 | **8** |
| 6 | `extra-5-extra` | `th8_ud2cca.rpy` | 48,567 | 1800 | 0 | **8** |
| 7 | `extra-6-extra` | `th8_ud2cbc.rpy` | 51,084 | 1800 | 0 | **8** |
| 8 | `extra-7-extra` | `th8_ud2c0b.rpy` | 48,268 | 1800 | 0 | **8** |
| 9 | `extra-8-extra` | `th8_ud2cdf.rpy` | 54,772 | 1800 | 0 | **8** |
| 10 | `extra-9-extra` | `th8_ud2c47.rpy` | 44,767 | 1800 | 0 | **8** |
| 11 | `extra-10-extra` | `th8_ud2c7a.rpy` | 44,771 | 1800 | 0 | **8** |
| 12 | `extra-11-extra` | `th8_ud2cc0.rpy` | 47,173 | 1800 | 0 | **8** |

**回退安全性**：`--start-stage 8` 若光标假设有误，闭环 40 s 超时后**仍然会走盲序列**（`Navigator.run()` 的既有逻辑未改），最坏情况与今天完全一致；因此该参数**不会让导航变差**，只会命中更早。

## 5. 超时不再留脏：`status="timeout"`

### 5.1 规则

| 触发 | `timeoutSource` | 判定 |
| --- | --- | --- |
| 批跑 `subprocess.run(timeout=budget + slack)` 到期 | `"batch-wallclock"` | `status="timeout"` |
| observer 自身 watchdog 写出 `report.status == "timeout"`（exit 7） | `"observer-watchdog"` | `status="timeout"` |
| 其它非零退出 / 无 report | — | 维持 `failed`（原逻辑） |

- `timed_out=true` 的条目 **一定** `status="timeout"`，并在 entry 里写 `rowsAreEvidenceOnly=true` 与说明文本 "…rows captured, kept as evidence only"；**绝不**因为 `rows_equal`/`sha_equal` 为真而判 ok（fail-closed）。
- `status="timeout"` **不是**终态（`TERMINAL_STATUSES=(ok, mismatch)` 未变），与 `failed` 一样进入可重试集合 `RETRYABLE_STATUSES=("failed","timeout")`：`--max-attempts 2` → **重试一次**，第二次仍 timeout 则 exhausted 留给人工。
- `report.json` 新增 `timeout` 计数；`failures[]` 里每条带 `status`（`failed`/`timeout`）；`exit_code_for()` 把 timeout 计入 **exit 1**。
- 旧 journal 行语义不变：EXP-P0-075 那行人工补写的 `status="failed"` + `timed_out=true` 原样保留（历史不改写），只是现在也会被当成可重试项。
- schema 升级：`JOURNAL_SCHEMA → v400-retail-trace-batch-journal/2`、`REPORT_SCHEMA → v400-retail-trace-batch/2`（新增 status 值与字段）。旧行仍可读（`read_journal` 只认 `fixture_id`）。

### 5.2 为什么这条重要

旧逻辑下，"预算到点 + observer 优雅退出（exit 6, status=partial）" 会走 `failure` 分支（`failed`），但**如果**某次运行的 rows 恰好等于 `calculationFrames`、hash 又恰好相等（例如 kill 发生在 detach 之前），旧代码会把它判成 `ok`。现在这类边界一律 `timeout`，批跑不会用"被切断的运行"去凑 G0.1 的 108/108。

## 6. 单测

```
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q \
    tests/test_v400_retail_trace_batch.py tests/test_v400_retail_trace_debugapi.py
-> 89 passed in 2.68s
   test_v400_retail_trace_batch.py     44 passed  (31 -> 44, +13)
   test_v400_retail_trace_debugapi.py  45 passed  (44 -> 45, +1)

python -m compileall -q tools/rl/v400_retail_trace_batch.py tools/rl/v400_retail_trace_debugapi.py
-> exit 0
```

`tests/test_v400_retail_trace_batch.py` 新增 13 个：

| 测试 | 断言要点 |
| --- | --- |
| `test_fixture_budget_formula_and_examples` | 149,986→3300；135,726→3015；135,000→3000；99,113→2283；10/0/None/−5→1800；75,001→1801（ceil 而非 round）；75,000→1800（阈值边界） |
| `test_fixture_budget_cap_and_overrides` | 400,000→5400（上限）；`maximum=2000`→2000；`maximum=600`→1800（上限不得低于下限）；`rows_per_second=61.5`→2739；`minimum=2000`→3300 |
| `test_the_twelve_large_fixtures_get_a_raised_budget` | 真 manifest：12 份 id 与 EXP-P0-075 §6 **逐项一致**、全部 >1800、`lunatic-3-b == 3300`、max=3300、无一触及 5400；抬升集合严格等于 `frames > 75,000`（97 份） |
| `test_start_stage_is_the_first_manifest_stage_index` | `[8]→8`、`[0,…]→0`、空 `stages→0`、首项 `None` 时取下一个非空 |
| `test_the_twelve_extra_fixtures_use_start_stage_eight` | 真 manifest：恰好 12 份 `[8]` 且 id 为 `extra-0..11-extra`；每份 `start_stage_for` = 8/0；且 `start_stage ∈ stages` |
| `test_run_one_passes_the_adaptive_budget_and_start_stage` | 149,986 帧 + `[8]` → argv `--budget-seconds 3300` + `--start-stage 8`；entry 记 `budgetSeconds/budgetBaseSeconds/budgetMaxSeconds/wallclockGuardSeconds`；日志含 `budget=3300 s, startStage=8` |
| `test_run_one_keeps_the_floor_for_a_normal_fixture` | 10 帧 + `[0,1]` → `--budget-seconds 1800` + `--start-stage 0` |
| `test_a_wallclock_timeout_is_journalled_as_timeout_and_never_as_success` | 伪造"rows/hash 全对"的 report + `TimeoutExpired` → `status="timeout"`、`rowsAreEvidenceOnly=True`、`timeoutSource="batch-wallclock"`、rows 仍留证、错误串含 `wall-clock timeout after 1860 s (budget 1800 s)` |
| `test_an_observer_watchdog_timeout_is_also_status_timeout` | `report.status="timeout"`（exit 7）→ `status="timeout"`、`timeoutSource="observer-watchdog"`、采用 observer 自报的 `budgetSeconds=3300` |
| `test_timeout_is_retryable_and_counted_as_a_failure` | `terminal_entries` 不含 timeout；`exhausted_entries` 1 条→可重试、2 条→exhausted；report `timeout=1/failed=0`、`failures[0].status="timeout"`、exit code 1 |
| `test_report_and_heartbeat_carry_the_budget` | report `budgetPolicy/budgetPlan/fixtures[].budgetSeconds/startStage`、`journalFieldNotes` 含新字段；heartbeat 带 `budgetSeconds/startStage/budgetPolicy/budgetPlan` |
| `test_main_retries_a_timed_out_fixture_once_and_keeps_its_evidence` | journal 已有 1 条 timeout → main 跑 attempt-2（旧证据不删）→ `["timeout","ok"]`、report `timeout=0/identical=1` |
| `test_main_dry_run_prints_the_budget_and_start_stage` | dry-run 输出含 `budget=1800s startStage=0` 与预算公式 |

`tests/test_v400_retail_trace_debugapi.py` 新增 1 个：

| 测试 | 断言要点 |
| --- | --- |
| `test_budget_cap_covers_the_adaptive_per_fixture_budget` | `MAX_BUDGET_SECONDS ≥ 3300`；**批跑 `BUDGET_MAX_SECONDS` == observer `MAX_BUDGET_SECONDS`**；真 manifest 最大件的计划预算 == 3300 且 ≤ 上限 |

原有 75 个测试**全部保持通过**（`test_observer_command_contract` 里 `frames=10` + 1800 → 仍传 `1800`；`test_run_one_without_report…` 的 exit 7 不被误判为 timeout，因为判定只认 `report.status`，不认退出码）。

## 7. 离线验证（全部无零售进程）

| 产物 | 命令 | 结果 |
| --- | --- | --- |
| 108 份计划表 + 12+12 清单 + 在跑 journal 投影 | `python .tmp_v400_budget\budget_table.py` → `.tmp_v400_budget\budget_table.txt` | 97/108 抬升、max 3300、计划预算合计 284,718 s（固定 1800 口径 194,400 s）、`pending planned = 274,890 s` |
| 真实 CLI dry-run（不写任何文件） | `python -u tools\rl\v400_retail_trace_batch.py --root .tmp_v400_budget\dryrun --dry-run` → `.tmp_v400_budget\dry_run.txt` | 退出码 3（partial，dry-run 既有语义）；启动日志出现 budget/start-stage 两行；`lunatic-3-b … budget=3300s startStage=0`、`extra-2-extra … budget=1800s startStage=8` |
| 历史 Extra trace 的 canonical sha | `python .tmp_v400_budget\verify_extra_sha.py` → `.tmp_v400_budget\verify_extra_sha.txt` | `r3_full`（start 0）44,554 行 sha=`d48351bf…58765f`=manifest；`full_extra2`（start 8）34,200 行未完成 |
| start 0 vs start 8 前缀 | `python .tmp_v400_budget\compare_extra_prefix.py` → `.tmp_v400_budget\compare_extra_prefix.txt` | 前 **34,200 行逐字节相同** |
| 截断风险统计 | `python .tmp_v400_budget\risk_counts.py` → `.tmp_v400_budget\risk_counts.txt` | @61.5 行/s：75 份纯行时间 >1800 s；@79：8 份；@100：0 份 |

## 8. 重启步骤（等 `easy-2-b` 收尾后，用 STOP 文件干净退出）

**不要**直接 `Stop-Process` 旧批跑：那会留下正在跑的 observer + `th08.exe` 孤儿（批跑的 `guard_after` 也就观察不到了）。用批跑自带的 STOP 文件机制——它在**每份 fixture 开始前**检查：

```powershell
# 0) 确认 easy-2-b attempt-2 已收尾（journal 出现 easy-2-b 的 ok/mismatch 行）
Get-Content .tmp_v400_trace_batch\batch3.stdout.log -Tail 5
(Get-Content .tmp_v400_trace_batch\journal.jsonl | Measure-Object -Line).Lines   # 期望 5

# 1) 放 STOP：旧批跑跑完当前件后自己退出（state=aborted, 不杀 observer/游戏）
New-Item -ItemType File .tmp_v400_trace_batch\STOP | Out-Null

# 2) 等它自然退出（python pid 22328 消失）并确认无残留零售实例
while (Get-Process -Id 22328 -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 10 }
(Get-Process th08 -ErrorAction SilentlyContinue | Measure-Object).Count    # 期望 0

# 3) 删 STOP，用新代码续跑（同 root → 从 journal 断点续跑）
Remove-Item .tmp_v400_trace_batch\STOP
python -u tools\rl\v400_retail_trace_batch.py --root .tmp_v400_trace_batch `
    *> .tmp_v400_trace_batch\batch4.stdout.log
```

- `--budget-seconds` / `--budget-max-seconds` **不用传**（默认 1800 / 5400 即自适应）。
- 同 root 续跑：4–5 条终态行会被跳过，其余 104 份继续；不新建第二个零售实例。
- 若要跑完再统一收口，可继续沿用 `--max-attempts 2`（默认）。

### 8.1 预期差异（重启后第一屏日志就能看到）

| 位置 | 旧批跑（pid 22328） | 新批跑 |
| --- | --- | --- |
| 启动日志 | `retry policy: … retry failed up to 2 attempts`、`budget=1800 s …` | `retry policy: skip ['ok','mismatch']; retry ['failed','timeout'] up to 2 attempts` + `budget: floor=1800 s cap=5400 s formula=…; raised for 97 of 108 fixtures, max=3300 s` + `start-stage: passed from each fixture's first manifest stage index (Extra fixtures use 8)` |
| 每份日志 | `--- fixture i/n <id> (file, N frames)` | 追加 `budget <X> s, startStage <0/8>`；observer argv 多一个 `--start-stage` |
| observer 调用 | `--budget-seconds 1800` | `--budget-seconds <3015…3300>` 的 12 份大件、97 份 >75k 全部抬升；Extra 12 份加 `--start-stage 8` |
| journal 每条 | 无预算字段 | `budgetSeconds / budgetBaseSeconds / budgetMaxSeconds / wallclockGuardSeconds / startStage / timeoutSource`，status 可能是 `timeout` |
| `report.json` | schema `…batch/1` | schema `…batch/2`；新增 `timeout`、`budgetPolicy`、`budgetPlan`、`fixtures[].budgetSeconds/.startStage/.timedOut`、`failures[].status` |
| `heartbeat.json` | 无预算 | 当前件 `budgetSeconds` + `startStage` + `budgetPolicy` + `budgetPlan` |

**用新预算的 fixture**：97 份（`calculationFrames > 75,000`）；其中 12 份 >135k 从 1800 → 3015–3300 s，其余 85 份按公式小幅抬升（例如 `easy-1-b` 126,803 → 2837 s，旧版正是它在 1664.9 s 跑完的，抬升不影响耗时）。
**用 `--start-stage 8` 的 fixture**：12 份 `extra-0-extra … extra-11-extra`；其余 96 份显式传 `0`（与旧默认等价）。

## 9. 预期 ETA 变化

| 口径 | 旧方案 | 新方案 |
| --- | --- | --- |
| 行数投影（剩余 11,880,150 帧，EXP-P0-075 §9） | 61.5 行/s → 193,200 s ≈ **53.7 h**；79 行/s → **41.8 h** | **不变**（同一速率、同一帧数）；band **42–54 h** → 约 **2026-11-11 07:00–19:00** |
| 12 份大 fixture 的墙钟 | 每份 2×1800 = 3,600 s，合计 **43,200 s ≈ 12 h**，**0 行被接受**、12 份永久 failed | 每份 1 次 3,015–3,300 s 预算；@61.5 行/s 实际占用 ≈ 28,120 s + 12×~15 s ≈ **28,300 s ≈ 7.9 h**，**12 份全部出结果** |
| 关键路径差 | — | **≈ −14,900 s ≈ −4.1 h**，并消除 12 个"必然失败" |
| 12 份 Extra 导航 | 12 × 47.7 s ≈ 572 s | 12 × 7.3 s ≈ 88 s → **−484…−492 s ≈ −8 min** |
| 总预算上限（不是 ETA） | 108 × 1800 = 194,400 s ≈ 54.0 h | Σbudget = **284,718 s ≈ 79.1 h**（上限，不是时长；无一份触及 5400 s 上限） |

**读数注意**：预算上限变大**不等于** ETA 变长——上限只在"这一份真的比 50 行/s 还慢"时才花时间；正常件按行数收工。真正的墙钟变化是：旧方案在 12 份大件上**注定**白烧 12 h 且永远拿不到 G0.1 的 108/108，新方案用 7.9 h 把它们一次性做成。

## 10. 风险与局限

1. **`--start-stage 8` 的闭环命中还没有一次"跑完整份"**：历史两次 `--start-stage 8` 都在 34,200 / 3,769 行时被外部终止，没有 report。**首次真实的批跑 Extra 件就是验收**（批跑会拿 manifest canonical sha 判，失败会被记录，不会被洗白）。最坏情况是 40 s 后回退盲序列 = 今天的行为，不会更差。
2. **50 行/s 是保守下限**：若某份稳定在 36 行/s 级别，公式仍不够（需要 >2,500 s 的件会被安排 3,000+ s，但更慢的件会被 5400 s 上限挡住）。当前 108 份里最慢的历史实测是 61.5 行/s，没有更低样本。
3. **`timeout` 依赖 observer 写 `report.status="timeout"`**：若 observer 被 `os._exit` 之外的路径杀掉（如外部强杀），批跑只能看到 `TimeoutExpired`。两种情况都被判 `timeout`，但"observer 无 report 且自愿退出"仍按 `failed` 处理。
4. **`--budget-seconds` 语义变更**（从"值"变成"下限"）会影响旧文档里的说法：任何 `--budget-seconds 900` 的调用现在仍会被公式抬到 ≥1800。若确实要更短（例如冒烟），需用 `--budget-seconds 900 --budget-max-seconds 900`——但 `--budget-max-seconds < --budget-seconds` 会被 parser 拒绝，因此**无法**把上限压到 1800 以下（这是有意的 fail-closed：G0.1 不接受被缩短的运行）。
5. **本卡未跑任何零售实例**：所有"预算/ETA"数字来自 manifest + 历史日志 + 单测，不是本轮实测；真正的新预算性能要等重启后的 batch4 日志。

## 11. 证据索引

| 用途 | 路径 |
| --- | --- |
| 预算公式 + 108 份计划 + 12+12 清单 | `.tmp_v400_budget/budget_table.py` → `budget_table.txt` |
| 真实 CLI dry-run（无写入） | `.tmp_v400_budget/dry_run.txt` |
| 历史 Extra trace canonical sha 复核 | `.tmp_v400_budget/verify_extra_sha.txt` |
| start 0 vs start 8 逐字节前缀 | `.tmp_v400_budget/compare_extra_prefix.txt` |
| 截断风险份数 | `.tmp_v400_budget/risk_counts.txt` |
| `--start-stage 8` 导航 7.289 s | `.tmp_v400_trace_fmt1/full_extra2/debugger.log`、`full_extra2_console.txt` |
| `--start-stage 8` 导航 7.264 s | `.tmp_v400_trace_fmt1/full_extra2b/debugger.log` |
| `--start-stage 0` 盲序列 47.663 s + 完整匹配 | `.tmp_v400_obsopt/r3_full/{debugger.log, report.json, rows.jsonl}` |
| 在跑批跑（未触碰） | `.tmp_v400_trace_batch/{journal.jsonl(4 行), heartbeat.json, batch3.stdout.log, easy-2-b/attempt-2/}` |
| 单测 | `tests/test_v400_retail_trace_batch.py`、`tests/test_v400_retail_trace_debugapi.py` |
