# WP02.5 — 用一次真实教师死亡 run 补 `verified_dead` 样本

- 工作包：v4.1.0 WP02.5（只写最小工具 / 测试 / 证据；不训练、不采数据集、不改 v400、不 commit/push）
- 产物：`tools/rl/v410_teacher_death_probe.py`、`tests/test_v410_teacher_death_probe.py`、本文件、`dead_sample_summary.json`、`dead_sample.jsonl`
- 状态：**真实死亡 run 完成**（pytest 36 passed / compileall / WSL 真实 run 退出码 0，三者均实际执行）。这是**一次**真实 run 的死亡样本，**不是** G410-1 通过，**不是**新数据集，**不是**存活率。

---

## 0 一句话结论

在真实引擎 `rl_laser/th08-modern`（sha256 `1462829b…`）上，`fast` 教师在 **stage 7（引擎 0-based；`third_party/th08/src/ScoreDat.hpp:73-87` 的 `Stage 1..6B = 0..7`，故 7 = `STAGE6B`，见 EXP-P0-077 §1）/ seed 1 / difficulty normal** 跑出 **stop_reason=death**：3,218 decisions / 12,872 frames / `deaths_observed=1`，退出码 **0**。这 3,218 个 verified 计划里有 **7 个 `died=true`**，经 WP02.4 的 `convert_teacher_row_calibrated` 全部读作 **`verified_dead` 7**；其余 **`verified_alive` 3,171 / `unknown` 40（全为 model_mismatch）**。`dead_sample.jsonl` 落盘 **7 条**真实 `verified_dead` 计划。WP02.4 之前只能由合成测试覆盖的死亡分支，现在有了真实引擎上的样本。

---

## 1 范围与不变量

- 只读输入：引擎二进制 `/root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern`（sha256 `1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f`）、教师 `tools/rl/v400_teacher.py`、数据目录 `/root/v400-data/g1/inst0`、WP02.4 模块 `touhou_rl/v410/teacher_calibration.py` 与 `teacher_labels.py`（只 import、不修改）、参考语料 `.tmp_v410_wp02_audit/teacher1000.jsonl`（仅被 `teacher_calibration` import 时读取，用于推导 8.0 px 容差）。
- 新增文件严格限于白名单 5 个；`/tmp/v410_death_probe/` 是运行目录（不入库）。
- 未修改：`touhou_rl/v400/`、`v380/`、`v390/`、`tools/rl/v400_*`、`tests/test_v400_*`、`third_party/`、`touhou_rl/v410/` 既有文件、`tools/rl/v410*.py` 其他工具、`state.json`/`journal.jsonl`/`budget.jsonl`/`assets.json`、WP02 其他证据（`audit.json`/`AUDIT.md`/`LABELS.md`/`labels_summary.json`/`labels_calibrated_*`/`FAST20.md`/`fast20*`/`CALIBRATION.md`/`STATUS.md`）。
- 未训练、未采集数据集、未启用 `--save-observations`（不写 shard）、未 commit/push、未删除/覆盖任何旧文件。
- `git status --porcelain` 中新出现的仅 `?? tools/rl/v410_teacher_death_probe.py`、`?? tests/test_v410_teacher_death_probe.py`；证据目录整体本来就是未跟踪目录（`?? Document/Update/Alpha/v4.1.0/evidence/`），本次只在其下新增 3 个文件；无 tracked 文件因本工作包被修改。

---

## 2 工具与 fail-closed 规则

`tools/rl/v410_teacher_death_probe.py`（WSL 可运行）只做三件事：驱动一次 `tools/rl/v400_teacher.py`（**固定 argv 列表，不拼 shell**）、解析它写的 JSONL、用 **WP02.4 自己的** `convert_teacher_row_calibrated` 给每条 verified 计划打标签。它不重写策略、不重算几何、不改容差。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--engine` / `--expected-sha` | — / `1462829b…` | `--engine` 指到二进制时，同时把同一路径传给教师 `--binary`（教师 `--engine` 期望目录） |
| `--data-dir` | — | 游戏数据目录（`inst<k>`） |
| `--stage` / `--seed` / `--difficulty` | `7` / `1` / `normal` | `--stage` 是引擎 0-based 索引 |
| `--max-frames` / `--decisions` / `--mode` | `54605` / `20000` / `fast` | 直接转给教师 |
| `--out-jsonl` | `/tmp/v410_death_probe/teacher_stage7_seed1.jsonl` | 教师 JSONL |
| `--summary` / `--sample` / `--sample-limit` | `/tmp/…/dead_sample_summary.json` / `/tmp/…/dead_sample.jsonl` / `100` | 证据输出与样本上限 |
| `--attempts-log` | `<out-jsonl 目录>/attempts.jsonl` | 每次调用追加一条真实 attempt 记录 |
| `--teacher` / `--python` / `--timeout-s` / `--tolerance` | `tools/rl/v400_teacher.py` / 当前解释器 / `7200` / `None` | `--tolerance` 为空时用 WP02.4 的数据推导常数 |

实际传给教师的 argv（与本工作包任务文本逐字对齐，仅解释器前缀不同）：

```
/root/v400-venv/bin/python tools/rl/v400_teacher.py --decisions 20000 --max-frames 54605 \
  --stop-on-death --stage 7 --difficulty normal --seed 1 --mode fast --quiet \
  --out /tmp/v410_death_probe/teacher_stage7_seed1.jsonl --data-dir /root/v400-data/g1/inst0 \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern \
  --binary /root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern \
  --expected-sha 1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f
```

### 2.1 fail-closed（任一命中就 exit 1，且**不写** summary / sample）

- 教师退出码非 0（含超时 → `-9`、无法启动 → `-1`）；
- JSONL 有**任何**空行、坏 JSON、非对象行、未知 `kind`，或缺少 `kind=report`、或 decision 行数为 0；
- `report.status != "ok"`；
- `report` 与请求不符：`stage` / `seed` / `mode` / `difficulty`（教师写 0..3 索引，工具把 `normal` 映射成 1 再比）任一不同；
- `report.engine.match` 不是 `true`；
- 引擎二进制 SHA 与 `--expected-sha` 不符（**启动前**本地 hash，教师自己的 SHA 门是第二道）；
- WP02.4 的 `MODEL_MATCH_TOLERANCE` 不可用（参考语料缺失/不满足规则）时**不**退化成手写数值；
- verified 计划数 ≠ `decisions × verify_k`（记录不完整，不能当样本）；
- 任一行被 WP02.4 的标注契约拒绝。

失败时只把这一次尝试如实写进 `--attempts-log`（含退出码、错误信息），成功时才写 summary/sample。

### 2.2 样本规则

`dead_sample.jsonl` 每行是一条 **`died=true` 的 verified 计划**（schema `v410-teacher-dead-sample/1`），按 `(decision, plan_index)` 排序取前 `--sample-limit` 条；`died=false` 计划**永不**替补进来。没有死亡时该文件写空、summary 标 `no_dead_sample=true`。

---

## 3 目标 run（实际执行）

在 WSL2 Ubuntu（`/root/v400-venv/bin/python`，Python 3.10.12）里，工作目录 `/mnt/d/WorkShop/Touhou_RL`：

```bash
/root/v400-venv/bin/python tools/rl/v410_teacher_death_probe.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern \
  --expected-sha 1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f \
  --data-dir /root/v400-data/g1/inst0 \
  --stage 7 --seed 1 --difficulty normal --mode fast --max-frames 54605 --decisions 20000 \
  --out-jsonl /tmp/v410_death_probe/teacher_stage7_seed1.jsonl \
  --summary Document/Update/Alpha/v4.1.0/evidence/WP02/dead_sample_summary.json \
  --sample  Document/Update/Alpha/v4.1.0/evidence/WP02/dead_sample.jsonl \
  --attempts-log /tmp/v410_death_probe/attempts.jsonl
```

```
v410_teacher_death_probe: stage=7 seed=1 difficulty=normal mode=fast exit=0 stop_reason=death
  frames=12872 decisions=3218 verified_plans=3218 died_true=7 verified_dead=7
  verified_alive=3171 unknown=40 model_mismatch=40 first_death_frame=791
  no_dead_sample=False sample_rows=7
REAL_EXIT=0
```

### 3.1 尝试清单（不伪造、不重试）

| # | stage | seed | difficulty | mode | 结果 | 教师退出码 | stop_reason | decisions | frames | 说明 |
|---|---|---|---|---|---|---|---|---|---|---|
| smoke | 7 | 1 | normal | fast | 通（不用于证据） | 0 | `max_frames` | 3 | 12 | 仅验证工具/教师/引擎链路（`--max-frames 12`），单独 attempts 日志 `/tmp/v410_death_probe/smoke_attempts.jsonl` |
| **1** | **7** | **1** | **normal** | **fast** | **死亡 → 采用** | **0** | **`death`** | **3,218** | **12,872** | 目标 run，一次就拿到死亡；未触发“换 stage/seed 再试” |

`dead_sample_summary.json.attempts = 1`，`attempt_records[0]` 即上表第 1 行（exit 0 / `death` / 12,872 frames / died_true 7 / verified_dead 7）。

### 3.2 报告（`report` 行）

| 字段 | 值 |
|---|---|
| `status` / `stop_reason` | `ok` / **`death`** |
| `stage` / `stage_at_stop` / `stage_clear` | 7 / 7 / `false` |
| `seed` / `difficulty` / `mode` / `team` | 1 / 1（normal） / `fast` / 0 |
| `decisions` / `requested_decisions` | 3,218 / 20,000 |
| `frames_committed` / `survival_frames` | **12,872** / 12,872（= 3,218 × 4 帧） |
| `deaths_observed` / `deaths_final` | **1** / 1.0 |
| `stage_frames_at_stop` | 12,874 |
| `config` | horizon 32 / beam 24 / verify_k 1 / durations [1, 2] |
| `consistency` | `plans_verified=3218`、`agree_alive=3211`、**`geo_alive_engine_dead=7`**、`agree_dead=0`、`trajectory_mismatch=1515` |
| `frame_event_flags_player_hit` | 7 |
| `bombs_initial` / `bombs_final` / `bomb_decisions` | 3.0 / 0.0 / 3 |
| `events` | `["death_committed_at_3217"]` |
| `engine` | sha256 `1462829b…`，`match=true` |
| `realtime_multiple` / `decisions_per_second` | 0.8338 / 12.507 |
| 教师 wall（工具实测） | 260.215 s |

教师 JSONL：`/tmp/v410_death_probe/teacher_stage7_seed1.jsonl`，3,219 行（1 report + 3,218 decision），13,251,047 字节，sha256 `64bbabc9f0fef38b6dcf03e1d8ae5ed3034446c3d7862a6af2b63c9b34a0f717`。

---

## 4 真实数字

### 4.1 计数（`dead_sample_summary.json` → `counts`）

| 指标 | 值 |
|---|---|
| `decisions` | 3,218 |
| `verified_plans` | 3,218（= decisions × verify_k = 3,218 × 1，`plans_verified_ok=true`） |
| `died_true` | **7** |
| `died_false` | 3,211 |
| **`verified_dead`**（校准后） | **7** |
| `verified_alive`（校准后） | 3,171 |
| `unknown`（校准后） | 40 |
| `model_mismatch` | 40 |
| `incomplete` | 0 |
| `died_true_verified_dead_agree` | `true`（7 = 7） |
| `first_death_frame` | **791** |
| 旧保守口径（WP02.3，同一批计划重算） | `verified_alive` 1,703 / `verified_dead` 7 / `unknown` 1,508 |
| `trajectory_matches` | true 1,703 / false 1,515 / missing 0 |
| `max_position_error`（px） | count 3,218 / min 0.0 / median 0.0 / **max 51.431439** |
| `frames_advanced` | min = median = max = 32 |
| `first_action_of_dead_plans` | [0, 2, 3, 7, 11, 12, 15] |

`first_death`：decision 189、该决策点 frame 760、计划内 `death_frame` 偏移 31 → 绝对帧 **791**（`source = decision.frame + plan.death_frame`），`first_action=12`。注意：这是**第一次**“某个 verified 计划被判死”的时刻；那一决策并没有被提交成死亡（教师躲开了），本次 run 真正被提交的死亡发生在 `decision 3217`（`events: death_committed_at_3217`）。

### 4.2 样本 7 条（`dead_sample.jsonl`，4,815 字节，sha256 `9fa67938…`）

| decision | frame | first_action | duration | death_frame | frames_advanced | max_position_error | model_match | survival_status |
|---|---|---|---|---|---|---|---|---|
| 189 | 760 | 12 | 1 | 31 | 32 | 26.142138 | false | `verified_dead` |
| 1,595 | 6,384 | 3 | 2 | 31 | 32 | 24.0 | false | `verified_dead` |
| 3,123 | 12,496 | 0 | 1 | 31 | 32 | 24.970584 | false | `verified_dead` |
| 3,213 | 12,856 | 7 | 1 | 15 | 32 | 5.17157 | **true** | `verified_dead` |
| 3,214 | 12,860 | 11 | 1 | 11 | 32 | 4.242645 | **true** | `verified_dead` |
| 3,216 | 12,868 | 2 | 1 | 5 | 32 | 5.656854 | **true** | `verified_dead` |
| 3,217 | 12,872 | 15 | 1 | 1 | 32 | 51.431439 | false | `verified_dead` |

7 条全部 `survival_reason=died_true`、`model_match_tolerance=8.0`、`horizon_source=frames_advanced`、`required_frames=32`。样本只有 7 行是因为**这次 run 只有 7 个 died=true 计划**（上限 100 未被触及），不是截断。

---

## 5 与 WP02.4 的关系

- WP02.4 在既有 seed 1000 / stage 0 / `fast` 语料上（3,251 条 verified 计划、`deaths_observed=0`）算出容差 **8.0 px**，重标为 `verified_alive` 3,226 / **`verified_dead` 0** / `unknown` 25，并在 `limits` 里明说“死亡分支只由合成测试覆盖”。本工作包正是补这个洞：**不重算容差**（仍是同一个由 `.tmp_v410_wp02_audit/teacher1000.jsonl` 推导出的 8.0，summary 的 `calibration.tolerance_source` 记录了来源），只把一次真实死亡 run 喂进同一个 `classify_plan`。
- 真实样本复现了 WP02.4 的两条判定语义，且是**在真实数据上**：
  1. **引擎判定优先**：7 条 died 计划里有 4 条 `max_position_error` 为 24.0–51.43 px（远超 8.0 px 容差，`model_match=false`），仍然判 `verified_dead`——模型漂移不能把死亡改判成存活；
  2. **死亡优先于存活条件**：另外 3 条 `max_position_error` ≤ 8.0（5.17/4.24/5.66 px，`model_match=true`、`frames_advanced=32`）如果只看“模型匹配 + 帧数”会被读成 alive，但引擎的 `died=true` 把它们钉在 `verified_dead`。
- 同一批计划在旧口径（WP02.3，`trajectory_matches` @ 1e-3 px）下是 alive 1,703 / dead 7 / unknown 1,508；校准口径把 1,468 条漂移但活着的计划从 `unknown` 提到 `verified_alive`，**死亡数一条没变**（7 → 7）。这与 WP02.4 声明的“校准只移动 unknown→alive，不移动死亡”一致。
- 与 D1 的交叉核对（只读对比，不改 D1）：`.tmp_v400_d1/d1/shards/d1-s7d1-n3218-seed1-shard/shard.json` 记录 `stage=7, difficulty=1, seed=1, mode=fast, stop_reason=death, deaths_observed=1, decisions=3218, frames_committed=12872, decision_frames=2, frames_per_decision=4`。本次 run 同样是 7 / 1 / 1 / fast / `death` / 1 / 3,218 / 12,872——**同一 episode 形状**（D1 那片用的是 `rl_v3` 引擎 sha `691bfd77…`，本次是 `rl_laser` sha `1462829b…`，因此只能说死亡点一致，不能说逐位重放一致）。

---

## 6 验收命令与退出码（全部实际执行）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp025_pytest -q tests/test_v410_teacher_death_probe.py
```

```
....................................                                     [100%]
36 passed in 0.27s
PYTEST_EXIT=0
```

```powershell
python -m compileall -q tools/rl/v410_teacher_death_probe.py
```

```
COMPILEALL_EXIT=0
```

WSL 真实 run（§3 的命令）：

```
v410_teacher_death_probe: … exit=0 stop_reason=death frames=12872 decisions=3218 died_true=7 verified_dead=7 verified_alive=3171 unknown=40 model_mismatch=40 first_death_frame=791 no_dead_sample=False sample_rows=7
REAL_EXIT=0
```

测试覆盖（36 例，全部 fake，不启动引擎）：教师 argv 逐字对齐（含 `--binary`）、`--difficulty` 名→索引映射与 report 一致性检查、JSONL 解析（空行/坏 JSON/未知 `kind`/缺 report/无 decision 全部 fail-closed）、`died=true→verified_dead`（含 139.58 px 漂移仍判死）、漂移存活→`unknown/model_mismatch`、计划形状非法拒绝、计数与信号（trajectory_matches / max_position_error / frames_advanced）、`first_death` 绝对帧推导、样本只含 died 且限量且可空、attempt 日志索引与坏行拒绝、summary 必备字段、`main()` 的 7 条 fail-closed 路径（非 0 退出、坏行、SHA 不符不启动、report 非本 run、report 非 ok、引擎缺失、缺计划）、失败后成功仍记录两次尝试。

---

## 7 限制（这份证据不主张什么）

1. **单 run / 单 stage / 单 seed / 单 difficulty**：stage 7 / seed 1 / normal / `fast` 的**一次**死亡。其他 stage、seed、difficulty、mode 是否死、死几次，本证据一个字都没说。
2. **不是 G410-1**：没有评估任何门、阈值或发布检查；本工作包只是给 WP02.4 的 `verified_dead` 分支补真实样本。
3. **不是数据集**：没有训练、没有采集 shard、没开 `--save-observations`；7 条样本是**证据**，不是训练数据。
4. **不是存活率**：`deaths_observed=1` 是“这一局死了”，不是“stage 7 的死亡率”。
5. **`verified_dead` 的真实含义**：它是引擎对该 verified 计划回放的 `died=true` 判定经 WP02.4 `classify_plan` 透传；它证明“死亡语义在真实引擎上会出现”，不证明“每条 died 计划都被正确标注”，也不证明 WP02.4 的 8.0 px 容差是生存真值（它只是模型一致性阈值）。
6. **`first_death_frame=791` 的口径**：等于 `decision.frame + plan.death_frame`（`verify.py` 把 `death_frame` 记成计划视界内的 `2*slot+1`），不是 run 的死亡帧；run 的死亡帧是 12,872（`death_committed_at_3217`）。
7. **样本量 7 < 100**：上限没被触及；`died=false` 的计划没有被塞进样本充数（这是 WP02.4 `limits` 里“unknown 不是负样本”的同一条纪律）。
8. **与 D1 的一致性只到 episode 形状**：决策数/帧数/stop_reason 一致，但引擎 build 不同（`rl_v3` vs `rl_laser`），未做逐位重放比较。
9. 运行目录 `/tmp/v410_death_probe/`（教师 JSONL、日志、attempts 日志）**不入库**；仓库里只有 3 个证据文件。

---

## 8 复现

1. 先跑测试与编译（§6 两条命令）。
2. 在 WSL 里按 §3 的命令重跑；`--out-jsonl`/`--attempts-log` 指向 `/tmp`，`--summary`/`--sample` 指向证据目录。教师 SHA 门 + 工具启动前 hash 都会拒绝非 `1462829b…` 的引擎。
3. 用 `touhou_rl.v410.teacher_calibration` 复核任意一行：
   `convert_teacher_row_calibrated(row)` 对 `died=true` 的计划返回 `survival_status="verified_dead"`、`survival_reason="died_true"`，与 `dead_sample.jsonl` 的 `survival_status` 列一致。
4. 若某次重跑**没有**死亡：summary 会写 `no_dead_sample=true`、`dead_sample.jsonl` 为空——这是测量结果，不是失败，也**不会**用 `died=false` 计划伪造 `verified_dead`。
