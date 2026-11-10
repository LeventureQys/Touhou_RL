# EXP-P0-077 — WP7：stage 索引语义（0-based）全仓审计与修正（2026-11-09）

**结论：`stage` 在本仓库里是引擎的 0-based `stage_index`（`Stage 1 = 0`），而 bench 套件一度
按 1-based 声明，导致 `S1N` 实际跑的是**真 Stage 2**、却套用 Stage 1 的帧上限。已修正套件定义、
cap 查表、行字段、scoreboard 与 `--summarize` 分组；52 条历史行全部保留原数字并以追加式更正
记录标注为 Stage 2。修正后真实引擎 1 局验证：`RESET 0 1 0 1000 0` → `stage_name=Stage 1`、
`stage_start_observed=0`、`stage_observed=0`、cap `19887`（= ceil(1.5×13258)）、退出码 **0**。

本卡不重算任何历史数字，也不声称"把 Stage 2 的局改成了 Stage 1 的测量"。它只做两件事：
把 **stage 标签**改对，并把"混着读"变成"分开读"。

---

## 1. 已确认证据（沿用 EXP-P3-002 §6，并在本仓复核）

| 证据 | 位置 | 内容 |
|---|---|---|
| 引擎 stage 枚举 | `third_party/th08/src/ScoreDat.hpp:73-87` | `enum Stage { STAGE1, STAGE2, … STAGE6B, EXTRASTAGE, … }` → `STAGE1 = 0`，`Stage 1..6B = 0..7`，`Extra = 8` |
| 训练/评测入口契约 | `touhou_rl/v400/gym/env.py:43-63` | `StageStart.stage` 合法域 `0..8`（0-based；1-based 不会允许 0）；`F_STAGE = 1` 即 `STATE.f1` |
| 人类参考编号 | `Document/Update/Alpha/v4.0.0/bench/human_reference.json` | `stage_index 0` = `Stage1`，normal 13258 帧 → cap `ceil(1.5×13258) = 19887` |
| D0 实测 | `tools/rl/v400_d0_dual.py`（默认 `--stage 0`） | `--stage 0` 每局约 13.2k 帧换面，与 Stage 1 人类 13258 吻合（99.5%） |
| **本仓决定性证据** | `bench/results.jsonl` 52 条历史行 | 每行都有引擎自报的 `stage_observed`（= `STATE.f1`）**= 1**，即 Stage 2；而 `human_reference_stage` 被写成 `Stage1` |

`stage_observed` 是运行器在每局结束时直接读引擎 `STATE.f1` 写下的（`tools/rl/v400_bench_run.py`
`stage_observed=int(end_state["f1"])`），不是推断值。52 行全为 `1`，因此"这 40 局到底是
Stage 1 还是 Stage 2"不需要再做实验就能判定：**Stage 2**。

---

## 2. 判定表（当前值 / 正确语义 / 是否错误 / 影响的历史结果）

| # | 位置 | 当前值（修正前） | 正确语义 | 错误？ | 影响的历史结果 |
|---|---|---|---|---|---|
| 1 | `touhou_rl/v400/bench/suites.py:18` `STAGES = (1,…,8)` | 1..8（1-based） | 0..7 + Extra=8（0-based index） | **是** | 所有按 `suite.stages` 发的 `RESET` 整体偏移一个 stage |
| 2 | `suites.py:20-21` `S1N/S1L = (1,)` | `stage 1` | `stage 0`（Stage 1） | **是** | 40 局 dev20 + 6 局 provisional 全部落在 Stage 2；S1L 同理（lunatic） |
| 3 | `suites.py:22-24` `PSN/PSL/RUNN = STAGES` | `(1..8)` | `(0..7)` | **是** | 多 stage 套件至今被 runner 拒绝执行（无 stage-clear 检测器），无既成结果污染；但声明值本身是错的 |
| 4 | `human_reference.py:50-53` `SUITE_STAGE_NAMES{1:"Stage1",…}` | 键 1..8 | 键 0..7（+8:Extra） | **是** | 查表把"stage 1"映射成 `Stage1` 的 cap，与 #2 的偏移**互相抵消成看似自洽**：跑 Stage 2 却拿 Stage 1 的 19887 |
| 5 | `human_reference.py:39` `STAGE_NAMES` | 0-based（Stage1 在 0） | 同 | 否 | — |
| 6 | `human_reference.py:480-491` `cap_for_suite` | 用 `suite.stages[0]` 查表 | 用 0-based index 查表 | 否（算法对），**但依赖 #3/#4 才正确** | 修正后 `S1N → Stage1 → 13258 → 19887` 不变，语义变正确 |
| 7 | `tools/rl/v400_bench_run.py` CLI | 无 stage 参数，stage 来自 `suite.stages[0]` | 同 | 否（结构性） | 见 #2 的传导 |
| 8 | `tools/rl/v400_bench_run.py:378` 行字段 `stages` | `[1]` | `[0]` | **是** | 52 行历史行写的是 `stages:[1]`（= Stage 2），行内无 `stage_index`/`stage_name` |
| 9 | `tools/rl/v400_bench_run.py` 行字段 | 无 `stage_index`/`stage_name` | 新行必须同时写索引与人类可见名 | **是（缺字段）** | 历史行只能靠 `stage_observed` 反推 stage |
| 10 | `tools/rl/v400_bench.py` `--summarize` | 按 (agent, suite, cap_source) 分组 | 必须按 stage 区分 | **是** | Stage 2 的 40 局与 Stage 1 的局会被平均进同一格 |
| 11 | `touhou_rl/v400/bench/results.py` `summarize` | 同上 | 同上 | **是** | 同上 |
| 12 | `bench/scoreboard.md` 全部 "S1N" 行 | 标签 S1N + cap 19887 | 实际 Stage 2；cap 来自 Stage 1 | **是（标签）** | 数字本身是真测量，**标签与 cap 归属错** |
| 13 | `tools/rl/v400_g1_eval.py:49,53-69` | `S1N_STAGE = 0`，`PSN_STAGES = range(0,8)`，`STAGE_NAMES` 0-based | 同 | **否（P3 已修）** | G1 的 S1N 20 局是 `--stage 0`，**结论无需重述** |
| 14 | `tools/rl/v400_g1_eval.py:154-155` `PAIR_PROXY`/`stage±1` | 0-based 邻近 stage 代理 | 同 | 否 | — |
| 15 | `tools/rl/v400_wsl_vec_stress.py:596` | `c.reset(stage=0,…)` | 同 | 否 | 压测用的是 Stage 1 |
| 16 | `tools/rl/v400_teacher.py:3,93` | docstring 示例 `--stage 1`；help "stage index, 0..8" | 需写明 0-based 且 `1` 就是真 Stage 2 | **标签含糊** | 不影响历史数字；但 `--stage 1` 的示例会把读者带到 Stage 2 |
| 17 | `tools/rl/v400_student_eval.py:57-61` | `DEFAULT_STAGE = 0` + 0-based 注释 | 同 | 否（P3 已修） | `bc_v0_stage1` 那条 `--stage 1` 记录已被该文件标为"不是 S1N 行" |
| 18 | `touhou_rl/boss_probe.py:869` | `practice-stage-{cfg.practice_stage + 1}`（显示用） | 越界项，不在本次白名单 | 未见实测污染 | 仅作记录，未改（见 §8） |

---

## 3. 修正内容（只改确有错误处）

### 3.1 `touhou_rl/v400/bench/suites.py`

- 字段改名 `stages` → **`stage_indexes`**，并在 dataclass 里 `__post_init__` 校验 `0..8`；
  `stages` 保留为**同名别名的 property**（值相同、同为 0-based），避免"两个名字两种进制"。
- `STAGES = (1..8)` → **`STAGE_INDEXES = (0..7)`**；`S1N/S1L = (0,)`，`PSN/PSL/RUNN = (0..7)`。
- 新增单一词汇表 `STAGE_NAMES`（0-based，`0="Stage1"`）与 `stage_name()` / `stage_ordinal()`
  （`0 → "Stage 1"`）。人类可见名与 `human_reference.json` 完全一致，避免"Stage 1"与"Stage1"
  两套写法各自漂移。
- 新增 `Suite.stage_index` / `Suite.stage_name`（单 stage 套件才有，多 stage 为 `None`），
  供行字段与 scoreboard 直接取用。

### 3.2 `touhou_rl/v400/bench/human_reference.py`

- `SUITE_STAGE_NAMES` 键改为 **0-based**（`0:"Stage1" … 7:"Stage6B"`，并补 `8:"Extra"`，
  使未测量 stage 报"覆盖缺失"而不是"命名缺失"）。
- `cap_for_suite` 改读 `suite.stage_indexes`（缺失时回退 `suite.stages`），docstring 写明
  "0 就是 Stage1"。

### 3.3 `tools/rl/v400_bench_run.py`

- 行里新增 `stage_index` / `stage_name` / `stage_index_semantics`（单 stage 套件才写）。
- 新增**启动实测**字段 `stage_start_observed`（`RESET` 后立刻读 `STATE.f1`）与
  `stage_consistent`（`stage_start_observed == stage_index`）。这让"套件说 stage 0、引擎从
  Stage 2 起来"这种错误在行里直接可见，不需要事后推断。
- scoreboard 增加 `## Stage index (0-based)` 段与按 stage 标注的总表；每局表新增
  `Stage idx` / `Stage` 两列；human cap 行追加"该 cap 下实际跑过哪些 stage"；新增
  `## Stage-index corrections (append-only audit records)` 段，渲染追加式更正记录。
- 新增 `--correct-stage --suite S1N --reason ...`：扫描该套件中 stage 与套件声明不符的
  已测行，追加一条 `schema=v400-wp7-bench-stage-correction/1, status="corrected"` 记录
  （**不**写 `invalidated_started_at`，因此不会把任何行判废；`measured=false` 使其不会被
  伪造成一局）。
- `legacy`（非 dev20）原先与 provisional 混在一张表：人类参考 cap 的验证局会被写成
  "cap 不是人类参考"。已拆成 provisional 与 "human-referenced outside the dev set" 两张表。

### 3.4 `touhou_rl/v400/bench/results.py`

- 新增 `row_stage(row)` → `(stage_index, stage_name, source)`，`source ∈ {declared,
  stage_observed, unrecorded}`；**从不猜**：无 `stage_index` 时用引擎自报的 `stage_observed`，
  两者都没有则 `unrecorded`。
- `validate()` 强制 `stage_index ∈ 0..7`（拒 bool、拒 9）且 `stage_name == STAGE_NAMES[idx]`，
  `stage_name` 不得脱离 `stage_index` 单独出现。
- `summarize()` 的分组键加入 `stage_index` / `stage_name` / `stage_source`，表格新增
  `Stage idx` / `Stage` / `Stage source` 三列。
- `VOID_STATUSES` 加入 `"corrected"`，使更正记录不会被当成一局去校验。

### 3.5 `tools/rl/v400_bench.py`

- `--summarize` 新增 `--stage-index N`（可重复）：只汇总该 0-based stage 的行。这是"按新字段
  过滤旧行"的可执行形式：`--stage-index 0` 得到 Stage 1 表，历史 Stage 2 行自动排除。
- 模块 docstring 写明 0-based 与回退规则。

### 3.6 `tools/rl/v400_teacher.py`（仅标签）

- docstring 与 `--stage` help 写明 0-based：`0 = Stage 1`、**`1` 就是真 Stage 2**（也是该
  选项默认值），`S1N` 必须传 `--stage 0`。行为未改。

---

## 4. `bench/scoreboard.md` 的更正方式（不改写历史数字）

1. **没有删除、没有改写任何一行**：`results.jsonl` 从 53 行 → 55 行（1 条更正记录 + 1 条
   验证局），前 53 行字节未变（`append_row` 每次都校验"旧字节仍是前缀"）。
2. 追加的更正记录（`started_at` 无关、`created_at = 2026-11-09T12:30:34.379609+00:00`）：
   `status="corrected"`、`measured=false`、`declared_stage_index=0`、
   `observed_stage_indexes=[1]`、`affected_rows=52`、`corrected_labels=["S1N"]`，
   逐行列出 `started_at`/`seed`/`agent`/`frames`/`score` 供交叉核对。
   52 = 46 条仍在表内的行 + 6 条早已被 EXP-P0-048 判废的行（它们同样跑在 Stage 2）。
3. scoreboard 由 `--summarize` 重新生成（不手写数字），新增三段可核对的产物：
   - `## Stage index (0-based)`：0-based 语义说明 + 按 `(suite, stage_index, stage_name,
     cap_source, stage_source)` 汇总的表 → `S1N | 0 | Stage 1 | 1 行 | declared` 与
     `S1N | 1 | Stage 2 | 40 行 + 6 行 | stage_observed` 明确分开；
   - 每局表新增 `Stage idx`/`Stage` 两列（40 行全部显示 `1 | Stage2`）；
   - `## Stage-index corrections`：把上面的更正记录渲染成表（"这些行实际跑的是
     `stage_index 1` (Stage2)，套件现在声明 `stage_index 0` (Stage1)"），并说明
     "改正的是标签，永远不是测量数字"。
4. `--summarize` 对旧行的区分机制：`row_stage()` 优先读新字段 `stage_index`，缺失时用引擎
   自报的 `stage_observed`，并把来源写进 `Stage source` 列；两者都没有才写 `unrecorded`。
   因此旧行不必改动就能被正确分到 Stage 2，且不会被伪装成"声明的" stage。

---

## 5. 最小验证（1 局真实引擎）

命令（修正后的套件定义；未加 `--dev20`，故为"dev 集之外的验证局"）：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python tools/rl/v400_bench_run.py --agent random --suite S1N --seed 1000 --execute
```

| 字段 | 值 | 说明 |
|---|---|---|
| 退出码 | **0** | `append_row` 成功、`status=ok` |
| `reset_stage_start` | `RESET 0 1 0 1000 0` | stage 参数确实为 **0** |
| `stage_index` / `stage_name` | `0` / `Stage1` | 新字段：索引 + 人类可见名 |
| `stage_start_observed` | **0** | `RESET` 后引擎 `STATE.f1` = 0 → 真的从 Stage 1 起来 |
| `stage_observed` | **0** | 局末 `STATE.f1` 仍为 0（未换面） |
| `stage_consistent` | `true` | 声明与实测一致 |
| `max_frames` / `cap_source` | `19887` / `human_reference` | 与 human cap 一致 |
| `human_reference_stage` / `human_frames` | `Stage1` / `13258` | `ceil(1.5×13258)=19887` |
| `frames` / `gameplay_frame` | `1862` / `1864` | 真实局长度（random 在 Stage 1 约 1.9k 帧就 game over，**远低于 cap**，符合本卡 §6 说明） |
| `end` / `deaths` / `score` | `game_over` / `3.0` / `5115` | 真实 referee 读数 |
| `reset_hash` | `a23f5edebd0e7454` | 该次启动的 `RESET` 状态哈希 |
| `started_at` | `2026-11-09T12:30:04.036950+00:00` | 追加在 `results.jsonl` 第 54 行 |

与历史行的对照：历史 52 行 `stage_observed = 1`（Stage 2），本行 `= 0`（Stage 1）。这是同一
字段、同一读取路径下的差异，正是"0-based 修正生效"的直接证据。

---

## 6. 影响面：需要重述的历史结论

| 结论 / 产物 | 原表述 | 应重述为 |
|---|---|---|
| **WP7 dev20 40 局**（`results.jsonl` 第 14–53 行） | "S1N（Stage 1）20 局 × 2 策略，cap 19887" | **Stage 2 的 40 局，套用了 Stage 1 的 cap 19887**。数字（frames/score/deaths）不变，但它们描述的是 Stage 2；不能作为 Stage 1 基线，也不能与任何 `--stage 0` 的 S1N 结果同列 |
| **6 条 provisional 行**（第 1–6、8–13 行中的 6 条仍有效者） | "S1N provisional cap 3000" | 同样是 **Stage 2**；"provisional" 说的是 cap 来源，不是 stage |
| **scoreboard 的 `Deaths [95% CI]`/`Score [95% CI]`** | "S1N 的 dev 20-seed 基线" | 数值有效，但标签须写成 `S1N / Stage 2 (idx 1)`；Stage 1 的 S1N 目前只有 n=1（本卡 §5） |
| **G1 的 S1N 20 局** | "S1N" | **无需重述**：`tools/rl/v400_g1_eval.py` 用 `S1N_STAGE = 0`，即 `--stage 0` = Stage 1，cap 19887 与 stage 一致（P3 已修） |
| **D0（`v400_d0_dual.py`）** | "`--stage 0` 每局 13.2k 帧换面" | 无需重述；它是本卡判定 0-based 的正向证据 |
| **`bc_v0_stage1` 检查点评估** | 曾以 `--stage 1` 记录 | 已在 `v400_student_eval.py` 内标为"不是 S1N 行（真 Stage 2）"；若引用该评估，须写明 Stage 2 |
| **PSN/PSL/RUNN 套件声明** | "stages 1..8" | 改为 0..7；这些套件**从未被真实执行**（runner 明确拒绝多 stage），因此没有受污染的结果，只有受污染的**声明** |
| **`v400_wsl_vec_stress.py` 压测** | `reset(stage=0)` | 无需重述（Stage 1） |

一句话：**只有"bench 的 S1N 行"需要重述；G1 / D0 / 压测的 stage 是对的。**

---

## 7. 测试

```
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_stage_audit/pt8 -q \
    tests/test_v400_bench.py tests/test_v400_bench_runner.py tests/test_v400_human_reference.py
→ 94 passed
```

新增/改写的用例（`tests/test_v400_bench.py`）：

- `test_bench_suites_declare_zero_based_stage_indexes`：`S1N.stage_indexes == (0,)`、
  `stage_index == 0`、`stage_name == "Stage1"`、`1 not in stage_indexes`（1-based 读法必须为假）、
  多 stage 套件 `stage_index is None`、越界 `ValueError`。
- `test_bench_suite_stage_names_match_the_measured_reference_names`：suite 与
  `human_reference.STAGE_NAMES` 必须是同一套词汇（防止两处各漂移）。
- `test_bench_summarize_keeps_stages_in_separate_cells`：同一 suite 的 idx 0 / idx 1
  不得进同一格；旧行按 `stage_observed` 分组并标注来源。
- `test_bench_results_validate_enforces_the_zero_based_stage_pairing`：`(0,"Stage2")` 必须
  报错（正是 EXP-P0-077 的缺陷形态）、`stage_name` 脱离索引报错、`idx=9`/`bool` 报错。
- `test_bench_row_stage_prefers_the_declared_index_and_never_guesses`。
- 旧断言按新列更新（`unrecorded` 两列而非静默归到某一 stage）。

新增用例（`tests/test_v400_bench_runner.py`）：

- `test_run_episode_records_the_zero_based_stage_and_the_observed_start_stage`：
  `RESET 0 1 0 1000 0`、`stages == [0]`、`stage_index/name`、`stage_start_observed == 0`、
  `stage_consistent is True`。
- `test_a_stage_the_engine_did_not_start_is_flagged_not_silently_relabelled`：协议替身
  永远报 `f1=1` → 行必须 `stage_consistent is False`（而不是被悄悄改成 Stage 2）。
- `test_multi_stage_suite_refuses_and_records_no_stage_index`。
- `test_scoreboard_annotates_every_table_by_zero_based_stage_index`。
- `test_correct_stage_lists_rows_that_ran_another_stage_and_edits_nothing`（含"旧字节仍是
  前缀"的 append-only 断言、`observed_stage_indexes == [1]`、无 `invalidated_started_at`）。
- `test_correct_stage_refuses_when_every_row_already_carries_the_right_stage` /
  `..._requires_a_reason_and_a_suite` / `test_a_correction_record_is_not_read_back_as_a_measured_episode`。
- 协议替身 `state_reply(stage=...)` 不再硬编码 `f1=1`：替身必须回报它被 `RESET` 到的 stage，
  否则替身自己就是 1-based 污染源。

回归（未改动的相邻文件）：`tests/test_v400_g1_eval.py`、`tests/test_v400_teacher_import.py`
→ 25 passed；`tools/rl/v400_bench.py --summarize` 对真实 `results.jsonl` 退出码 0，
`--stage-index 0` / `--stage-index 1` 分别得到 Stage 1 表与 Stage 2 表。

---

## 8. 边界与未做

- **未重跑 40 局 dev20**：本卡只修正语义与标签，不生成"Stage 1 的 dev20"数字。Stage 1 的
  S1N 目前只有本卡 §5 的 n=1 验证局，**不构成基线**。
- **未改引擎/协议/依赖/第三方**，未重建 build；`RESET`/`STATE` 线格式未动。
- **未删除任何历史行或文件**；`results.jsonl` 只追加（53 → 55 行）。
- `PSN/PSL/RUNN` 仍是声明而未执行：runner 没有 stage-clear 检测器，本次只把声明改成 0-based，
  没有假装它们跑过。
- `touhou_rl/boss_probe.py:869` 有 `practice_stage + 1` 的显示字符串，属本次白名单之外，未改；
  它是 probe 的模式标签，未发现其进入 bench/训练数据。若后续要统一词汇表，应单独开卡。
- 机器上同时有零售 trace 批跑（Windows）与 8 个 `rl_hazard2` 引擎进程；本卡只新增 **1** 次
  WSL 引擎启动（并发 1），未干扰在跑批处理。
