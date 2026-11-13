# WP02.4 — 用预注册的模型一致性阈值重标教师 survival

- 工作包：v4.1.0 WP02.4（只写代码 / 测试 / 离线重标；不启动游戏、不训练、不采集、不 commit/push）
- 产物：`touhou_rl/v410/teacher_calibration.py`、`tests/test_v410_teacher_calibration.py`、本文件、`labels_calibrated_summary.json`、`labels_calibrated_sample.jsonl`
- 状态：**校准规则 + 离线重标完成**（pytest 59 passed / compileall / 真实重标 全部 exit 0）。这是对既有教师 JSONL 的一次离线重标，**不是** G410-1 通过，**不是**新数据集；校准阈值是模型一致性阈值，**不是**生存真值。

---

## 0 一句话结论

在 seed 1000 的真实教师 JSONL（3,251 条决策 / 3,251 条 verified 计划）上，用预注册规则（覆盖 ≥99% 的最小 grid 阈值，上限 16 px）从数据算出 **`MODEL_MATCH_TOLERANCE = 8.0 px`**（覆盖率 3,226/3,251 = **99.231%**）。按它重标：**verified_alive 3,226 / verified_dead 0 / unknown 25**，其中 **moving_alive 1,704、idle_alive 1,522**；旧保守口径（WP02.3，`trajectory_matches` @ 1e-3 px）在同语料上是 **795 / 0 / 2,456**，moving_alive **0**。两种口径并排写入同一份汇总，旧证据文件未被修改（sha256 仍为 `709acbb5…`）。

---

## 1 范围与不变量

- 只读输入：`.tmp_v410_wp02_audit/teacher1000.jsonl`（11,931,643 字节，sha256 `a1d6923e…2f434b`，3,252 行 = 1 条 report + 3,251 条 decision）。未写入、未移动、未删除。
- 只读参考：`Document/Update/Alpha/v4.1.0/evidence/WP02/labels_summary.json`（WP02.3）只用于交叉复核；文件 sha256 仍是 `709acbb5d8221072dc0a93299a104515b799ae4a05274f5824ed7de876fbfebd`，与 `LABELS.md` 记录一致，**未被覆盖**。
- 未启动游戏、未训练、未采集、未 commit/push；未删除/覆盖旧文件。
- 新增文件仅白名单内 5 个（见 §11）。`touhou_rl/v400/`、`v380/`、`v390/`、`tools/rl/v400_*`、`tools/rl/v410*.py`、`tests/test_v400_*`、`touhou_rl/v410/` 既有文件（含 `teacher_labels.py`、`verify.py`）、`third_party/`、`state.json`/`journal.jsonl`/`budget.jsonl`/`assets.json`、WP02 的 `audit.json`/`AUDIT.md`/`LABELS.md`/`labels_summary.json` 全部未改。`git status --porcelain` 中与本工作包相关的新增项只有 `touhou_rl/v410/`、`tests/test_v410_teacher_calibration.py`、`Document/Update/Alpha/v4.1.0/evidence/WP02/` 三个未跟踪路径；工作树里既有的 tracked 修改（`readme.md` 等 18 项）是本次工作前就存在的，本工作包未触碰。

---

## 2 预注册规则与常量

| 常量 | 值 | 说明 |
|---|---|---|
| `ERROR_GRID` | `(0.001, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)` | 候选阈值，单位 px，升序 |
| `COVERAGE_TARGET` | `0.99` | 覆盖目标（对 `died=false` verified 计划） |
| `ERROR_GRID_CAP` | `16.0` | 选择上限；grid 上界即 cap |
| `MODEL_MATCH_RULE` | `"smallest grid threshold with coverage >= 0.99 of died=false verified plans, capped at 16.0 px"` | 规则原文，写进 summary |
| `MODEL_MATCH_TOLERANCE` | **`8.0`** | **由 `_bootstrap_model_match_tolerance()` 在 import 时从参考语料算出**，不是手写 |
| `MODEL_MATCH_TOLERANCE_STATUS` | `"computed"` | 数据满足规则时的状态 |
| `HORIZON_SOURCE` | `"frames_advanced"` | alive 所依赖的 horizon 来源 |
| `DEFAULT_HORIZON` / `DEFAULT_DECISION_FRAMES` | `32` / `2` | `report.config.horizon` / `committed_frames_per_decision ÷ decisions_per_decision_frame` |

常量来源（`labels_calibrated_summary.json` 内亦记录同源字符串）：

```
D:\WorkShop\Touhou_RL\.tmp_v410_wp02_audit\teacher1000.jsonl
(sha256=a1d6923efd8398619632d086bd854677e27cfd501158d2aeaafb37c9ca2f434b,
 died_false_plans=3251, covered=3226, coverage=0.992310, mismatch_count=25,
 rule='smallest grid threshold with coverage >= 0.99 of died=false verified plans, capped at 16.0 px')
```

### 2.1 fail-closed 行为（不是"放宽"，是"报错"）

- `calibrate()`：若 `ERROR_GRID` 中 ≤16 px 的**任何**阈值都达不到 99%，抛 `TeacherCalibrationError`，并在消息里列出最大覆盖率与规则原文；**不扩大 grid、不降低 target、不返回 16 px**。
- 参考语料缺失/不可读/不满足规则时，`MODEL_MATCH_TOLERANCE = None`，状态分别为 `reference_dataset_missing` / `reference_unreadable` / `reference_rule_not_satisfied`，`require_model_match_tolerance()` 直接抛错；`convert_teacher_row_calibrated(row)` 在 `tolerance=None` 时调用它，因此**不会**退化成手写默认值。
- 计划形状不合规（`died` 非 bool、`max_position_error` 负数/NaN/Inf/字符串、`frames_advanced` 负数/bool、`duration<=0`、`first_action` 越界）一律抛 `TeacherCalibrationError`，不猜。
- 流式转换里坏 JSON、空行、坏形状行计入 `skipped` + `errors`；CLI 只要 `skipped > 0` 就 exit 3，带洞语料不能伪装成转换成功。

---

## 3 判定规则（`classify_plan`，逐条 verified 计划）

| `died` | `max_position_error` | `frames_advanced` | 结果 | `survival_reason` | `survival_detail` |
|---|---|---|---|---|---|
| `True` | 任意 | 任意 | **`verified_dead`** | `died_true` | `died_true` |
| `False` | 缺失 | — | `unknown` | `model_mismatch` | `max_position_error_missing` |
| `False` | `> tolerance` | — | `unknown` | `model_mismatch` | `error_exceeds_tolerance` |
| `False` | `<= tolerance` | `duration` 缺失 | `unknown` | `incomplete_execution` | `duration_missing` |
| `False` | `<= tolerance` | `null` | `unknown` | `incomplete_execution` | `frames_advanced_missing` |
| `False` | `<= tolerance` | `< max(horizon, duration*decision_frames)` | `unknown` | `incomplete_execution` | `frames_advanced_below_required` |
| `False` | `<= tolerance` | `>= max(horizon, duration*decision_frames)` | **`verified_alive`** | `model_match_frames_advanced_ge_required` | `alive` |

- **引擎判定优先**：`died=True` 无论位置误差多大都是 `verified_dead`；校准不能把死亡改判成存活。
- **`unknown` 不是负样本**：`died=False` 但模型不匹配/执行不完整，一律 `unknown`，两个 reason 只有 `model_mismatch` / `incomplete_execution` 两类。
- 模型检查先于帧数检查（两者都不满足时 reason 记为 `model_mismatch`，`survival_detail` 明确到分支）。
- 边界包含：`error == tolerance` 算匹配，`frames_advanced == required` 算足够。
- 每条计划新增 `model_match`（`bool` 或 `null`）、`model_match_tolerance`、`horizon_source="frames_advanced"`；`max_position_error` / `trajectory_matches` / `died` / `frames_advanced` 原样保留。

---

## 4 命令与退出码（全部实际执行）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
```

### 4.1 pytest（绕过 `.pytest_cache`）

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp024_pytest -q tests/test_v410_teacher_calibration.py
```

```
...........................................................              [100%]
59 passed in 0.64s
PYTEST_EXIT=0
```

> `59 passed` 是稳定值；`in 0.64s` 是当次实测，重跑会浮动。用例覆盖：规则选择（最小达标 grid / 16 px cap / 不达标 fail-closed）、`died=True→dead`、`error=4+frames=32→alive`、`error=9→unknown/model_mismatch`、帧不足→`unknown/incomplete_execution`、未验证候选 `query unknown`、同 `first_action` 多计划保留、旧/新口径计数不同且都写出、空行/坏 JSON/坏行跳过计数、真实语料阈值表、CLI 退出码。

### 4.2 编译校验

```powershell
python -m compileall -q touhou_rl/v410
COMPILEALL_EXIT=0
```

> exit 0 只代表语法可通过，不等于测试通过；测试统计见 4.1。

### 4.3 真实重标（CLI）

```powershell
python -m touhou_rl.v410.teacher_calibration --input .tmp_v410_wp02_audit/teacher1000.jsonl --summary Document/Update/Alpha/v4.1.0/evidence/WP02/labels_calibrated_summary.json --sample Document/Update/Alpha/v4.1.0/evidence/WP02/labels_calibrated_sample.jsonl --sample-limit 100
```

```
teacher_calibration: decisions=3251 report_rows=1 skipped=0 tolerance=8.0 coverage=0.99231 alive=3226 dead=0 unknown=25 model_mismatch=25 incomplete=0 moving_alive=1704 idle_alive=1522
teacher_calibration: old conservative alive=795 dead=0 unknown=2456 (WP02.3, not rewritten)
teacher_calibration: summary -> Document\Update\Alpha\v4.1.0\evidence\WP02\labels_calibrated_summary.json
teacher_calibration: sample -> Document\Update\Alpha\v4.1.0\evidence\WP02\labels_calibrated_sample.jsonl (100 rows, sha256=2ce3fcca2d7f047b02ca9af6de9d30780892229ef478fedef007f5281ed454bf)
CLI_EXIT=0
```

### 4.4 证据可复现性校验（只读）

```powershell
python -c "import json;from touhou_rl.v410 import teacher_calibration as tc;s=json.load(open('Document/Update/Alpha/v4.1.0/evidence/WP02/labels_calibrated_summary.json',encoding='utf-8'));c=tc.stream_convert_calibrated('.tmp_v410_wp02_audit/teacher1000.jsonl',tolerance=s['calibration']['selected_tolerance']);assert c['alive']==s['calibrated']['alive'] and c['dead']==s['calibrated']['dead'] and c['unknown']==s['calibrated']['unknown'] and c['moving_alive']==s['calibrated']['moving_alive'];print('reproducible:',c['alive'],c['dead'],c['unknown'],c['moving_alive'],c['coverage'])"
```

```
reproducible: 3226 0 25 1704 0.99231
EXIT1=0
```

```powershell
python -c "import json;rows=[json.loads(l) for l in open('Document/Update/Alpha/v4.1.0/evidence/WP02/labels_calibrated_sample.jsonl',encoding='utf-8')];assert len(rows)==100;assert all(r['schema']=='v410-teacher-label-calibrated/1' and len(r['verified'])==1 for r in rows);p=rows[0]['verified'][0];assert {'died','frames_advanced','max_position_error','trajectory_matches','model_match','model_match_tolerance','horizon_source'}<=set(p);print('sample ok:',len(rows),'decisions',rows[0]['decision'],'..',rows[-1]['decision'])"
```

```
sample ok: 100 decisions 0 .. 3250
EXIT2=0
```

- `labels_calibrated_summary.json` 的校准/计数块与重新跑 `stream_convert_calibrated` 的结果**逐字段相等**（alive/dead/unknown/moving_alive/coverage）。
- `cross_check` 已把重算的旧口径与 `labels_summary.json` 对照：`reference={alive:795, dead:0, unknown:2456}`、`recomputed={alive:795, dead:0, unknown:2456}`、`matches=true`、`source_sha256_matches=true`。

---

## 5 阈值表与选择依据

来自 `calibration.threshold_table`（覆盖分母 = 3,251 条 `died=false` verified 计划；moving = `first_action != 0`）：

| threshold (px) | covered | coverage | mismatch | moving covered | moving coverage | 达到 99% | 选中 |
|---|---|---|---|---|---|---|---|
| 0.001 | 795 | 24.454% | 2,456 | 0 | 0.00% | 否 | |
| 0.5 | 795 | 24.454% | 2,456 | 0 | 0.00% | 否 | |
| 1.0 | 795 | 24.454% | 2,456 | 0 | 0.00% | 否 | |
| 2.0 | 1,217 | 37.435% | 2,034 | 44 | 2.560% | 否 | |
| 4.0 | 2,203 | 67.764% | 1,048 | 766 | 44.561% | 否 | |
| **8.0** | **3,226** | **99.231%** | **25** | **1,704** | **99.127%** | **是（首个）** | ✔ |
| 16.0 | 3,243 | 99.754% | 8 | 1,719 | 100.000% | 是 | |

选择依据（summary 原文）：`threshold 8.0 px is the smallest grid value whose coverage (3226/3251 = 0.992310) reaches the 0.99 target; the next smaller grid value 4.0 px only covers 2203/3251 = 0.677638`。

> 4.0 px 差 1,023 条（31.5 个百分点）才达标，8.0 px 一次跨过 → 这是 grid 分辨率上的自然台阶，不是"刚好挑一个好看的数"。

---

## 6 新旧口径对比（同语料、同计划、并排写入）

| 口径 | 规则 | alive | dead | unknown |
|---|---|---|---|---|
| 旧保守（WP02.3） | `died=False` + `trajectory_matches=True`（`verify.py` @1e-3 px）+ `frames_advanced >= duration*frames_per_decision` | **795**（24.454%） | 0 | **2,456**（75.546%） |
| 校准（WP02.4） | `died=False` + `max_position_error <= 8.0` + `frames_advanced >= max(32, duration*2)` | **3,226**（99.231%） | 0 | **25**（0.769%） |
| Δ | | **+2,431** | 0 | **−2,431** |

- `moving_alive`：旧 0（795 条 alive 全是 `first_action=0`）→ 新 **1,704**；`idle_alive` 1,522。
- `dead` 两种口径都是 0：本语料 `report.deaths_observed=0`，校准**没有**产生死亡，也**没有**把任何死亡改写。
- `dying` 之外唯一被移动的是 `unknown → alive`（2,431 条），来源就是 `died=False` 且位置误差落在 8 px 内的计划。

### 6.1 `per_first_action`（计划数 / 校准 alive / 旧口径 alive）

| action | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plans | 1532 | 69 | 290 | 326 | 71 | 34 | 212 | 208 | 42 | 0 | 8 | 179 | 198 | 16 | 3 | 35 | 16 | 12 |
| 校准 alive | 1522 | 69 | 287 | 325 | 71 | 34 | 206 | 207 | 40 | 0 | 8 | 179 | 198 | 15 | 3 | 34 | 16 | 12 |
| 旧 alive | 795 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

> 18 个动作里 17 个有存活正样本；action 9 本来就没有计划（0 条）。每行合计 3,251 / 3,226 / 795。

---

## 7 为什么 8 px 合理

1. **1e-3 px 回答的是另一个问题**。`verify.py::POSITION_TOLERANCE=1e-3` 是"引擎是否逐位复现搜索模型轨迹"的精确模型检查：`trajectory_matches = (max_position_error <= 1e-3)`。它检验的是**运动模型同步**，不是**生存**。把它当生存标签，结果就是 795 条 alive 全是 `first_action=0`、`duration=1`、`error=0.0` 的 idle 计划 —— 对机动动作零信息量。这是 WP02.4 要修的"单位错误"，不是阈值偏小 3 个数量级的小问题。
2. **8 px 来自预注册规则，不是手选**。规则在任何数字出现前就写死：覆盖 99% 的最小 grid 值、上限 16 px。实测 0.001/0.5/1.0 都只有 795；2.0 到 1,217；4.0 到 2,203（67.8%）；8.0 到 3,226（99.231%）**首次达标**；16.0 虽到 99.754%，但规则取最小达标值，因此不会贴 cap 选 16。代码里没有字面量 `8.0` 的容差常量：`MODEL_MATCH_TOLERANCE` 在 import 时由 `calibrate_teacher_file(DEFAULT_TEACHER_JSONL)` 算出并记录 sha256/覆盖/选中理由。
3. **8 px 与实测漂移分布同量级**。全语料 `max_position_error` 中位 3.414 px、最大 139.58 px；8 px ≈ 2.3× 中位，仍把最差尾部（25 条 >8 px，最大 139.58）留在 `unknown`，没有为了"凑满 100%"去选 16 或更大。
4. **校准只决定"能否当正样本"，不决定生死**。`died` 是唯一生存真值：`died=True` 无论误差多大都 `verified_dead`。因此 25 条 >8 px 的计划不是"被判死"，而是"模型漂移太大、不能确信是干净正样本" → `unknown`，WP03 不得把它当负样本。
5. **修复对机动动作真实有效**：moving coverage 从旧口径 0 → 8 px 下 1,704/1,719 = 99.13%，moving_alive 从 0 → 1,704。这是 WP02.4 的核心收益，而不是把 idle 计数换个名字。
6. **没有改 `verify.py`**（不在白名单）：`trajectory_matches` 字段仍保持 1e-3 语义，重标发生在标签层。下游若看到 `trajectory_matches=False` + `survival_status="verified_alive"`，那是**预期**的：前者是模型精确性，后者是（校准后的）计划可用性。

---

## 8 真实重标数字（seed 1000 / mode fast / stage 0 / difficulty 1）

源：3,252 行 = **1 report + 3,251 decision**；`report.decisions=3251`、`deaths_observed=0`、`config={horizon:32, beam:24, verify_k:1, durations:[1,2]}`、`decisions_per_decision_frames=2`、`committed_frames_per_decision=4`。

| 指标 | 值 |
|---|---|
| `decisions` / `report_rows` / `skipped` | 3,251 / 1 / 0 |
| `verified_plans_total` | 3,251 |
| `tolerance` | **8.0** |
| `coverage`（died=false 且 error≤8） | **0.992310**（3,226 / 3,251） |
| `calibrated.verified_alive` | **3,226**（99.231%） |
| `calibrated.verified_dead` | **0** |
| `calibrated.unknown` | **25**（`model_mismatch` 25，`incomplete_execution` 0） |
| `moving_alive` / `idle_alive` | **1,704** / **1,522** |
| `model_match` true / false / missing | 3,226 / 25 / 0 |
| `died_true_plans` / `died_false_plans` | 0 / 3,251 |
| `horizon.source` / `value` / `decision_frames` | `frames_advanced` / 32 / 2 |
| `horizon.verification_horizon_missing` | **true**（3,251/3,251 行无该字段） |
| `sample.rows` / 类别 | 100 / alive 75、dead 0、unknown 25 |

> 汇总顶层另有 `horizon_source="frames_advanced"` 与 `verification_horizon_missing=true`（与 `horizon` 块同值）；`counts.verified_plans` 与 `counts.verified_plans_total` 同一个数（3,251）——前者是 WP02.4 要求的键名，后者沿用 WP02.3 契约，避免两个消费方读到不同字段。

### 8.1 样本 `labels_calibrated_sample.jsonl`

100 行、370,519 字节、sha256 `2ce3fcca2d7f047b02ca9af6de9d30780892229ef478fedef007f5281ed454bf`；按 `decision` 升序覆盖 0..3250；确定性分层（按决策级校准类别等额配额，缺类余量按类别顺序补齐，再按 `decision` 排序），重跑字节一致。因为全语料只有 25 条 `unknown`，样本把 25 条全部收进来 + 75 条 alive —— 不是前 100 行前缀抽样。

### 8.2 与 WP02.3 证据的交叉复核

`labels_calibrated_summary.json` 的 `cross_check`：`available=true`、`sha256=709acbb5…`、`reference={verified_alive:795, verified_dead:0, unknown:2456}`、`recomputed={795,0,2456}`、`matches=true`、`source_sha256_matches=true`。**旧文件只读**，没有重写、没有改名、没有删除。

---

## 9 限制（这些数字能说什么、不能说什么）

1. **8 px 不是生存阈值**。它只回答"这条 died=false 计划的模型漂移是否小到可以当正样本"。不能说"8 px 内一定不死"，也不能用它反推死亡概率。
2. **25 条 >8 px 仍是 `unknown`**。它们是漂移尾部（最大 139.58 px），不是死亡，也不得在训练里当负样本。
3. **`verified_dead = 0` 仍未被真实数据覆盖**。源 report 写明 `deaths_observed=0`；death 分支只有合成测试覆盖（`test_died_plan_is_verified_dead_whatever_the_error_is` 等）。
4. **horizon 字段 100% 缺失**。`horizon_source="frames_advanced"`、`verification_horizon_missing=true`：alive 的依据是"计划实际推进了 `max(32, duration*2)` 帧"，本语料 3,251 条 `frames_advanced` 全为 32。若未来语料帧数更小，会落进 `incomplete_execution` 而不是被放行。
5. **单次运行范围**：seed 1000、mode fast、stage 0、difficulty 1。换 stage/mode 必须重新校准（规则可重算，8.0 不能当信念搬家）。
6. **常量依赖参考语料**：`MODEL_MATCH_TOLERANCE` 在 import 时从 `.tmp_v410_wp02_audit/teacher1000.jsonl` 计算；该文件缺失时常量是 `None` 且 `require_model_match_tolerance()` 抛错（干净 checkout / 临时目录被清理时会表现为"不可用"，而不是悄悄用 8.0）。
7. **没有改 `verify.py` 的 `POSITION_TOLERANCE`**（不在白名单）：`trajectory_matches` 字段仍是 1e-3 语义；本模块是标签层重标，不是引擎校验修改。
8. **样本里没有 dead 行**（语料 0 死亡），`dead` 分支的可视化证据仍缺。
9. **`coverage` 的分母是 `died=false` 计划**；语料无死亡所以等于全部 3,251 条。有死亡的新语料里，`coverage` 只统计 died=false，`dead` 不参与。

---

## 10 WP03 接入规则（硬约束）

1. **只把 `survival_status == "verified_alive"` 当正样本**。这是本次重标唯一授权的用法。
2. **`unknown` 既不是正样本，也不是负样本**。当成负样本会把"模型不同步"训练成"会死"；当成正样本会把漂移计划当干净成功。两者都禁止。
3. **负样本只能来自 `verified_dead`（真实 `died=True`）**。当前语料 0 条 → WP03 要负样本必须等真实死亡 run，不能用 `unknown` 凑。
4. **把 `model_match` / `model_match_tolerance` / `horizon_source` 一起写进训练样本元数据**，让"这条正样本来自 8 px 校准"可追溯；不要用 `trajectory_matches` 判断存活。
5. **`query_status` 仍只是"问过/没问过"**（`verified` / `unknown`），不是 survival，不参与正负样本划分。
6. **枚举校验**：训练侧用 `SURVIVAL_STATUSES` / `SURVIVAL_REASONS` 校验标签，别把 `"verified_alive"` 写死在多处。
7. **换语料先重校准**：`calibrate_teacher_file(新语料)`；若 fail-closed，先查模型漂移（WP02.2/WP04），**不要**手工挑一个能过的阈值。若新语料仍选 8.0，记录新语料的覆盖率与 mismatch 数，不要复用旧字符串。

---

## 11 产物清单

| 文件 | 字节 | sha256 |
|---|---|---|
| `touhou_rl/v410/teacher_calibration.py` | 78,168 | `c0917561f47e71c47fa71966c844ea32fe1eca1cd6a56352c1ea378067a8097c` |
| `tests/test_v410_teacher_calibration.py` | 37,384 | `51ad7b987639ecfcab468b212838f3915e69a819b83c23f151b664de5b03e5a2` |
| `Document/Update/Alpha/v4.1.0/evidence/WP02/labels_calibrated_summary.json` | 30,135 | `5ce5764336b636a4f6ccc6d0d54201326c3f2b2e9db42575610fa93df976dc54` |
| `Document/Update/Alpha/v4.1.0/evidence/WP02/labels_calibrated_sample.jsonl` | 370,519 | `2ce3fcca2d7f047b02ca9af6de9d30780892229ef478fedef007f5281ed454bf` |
| `Document/Update/Alpha/v4.1.0/evidence/WP02/CALIBRATION.md` | 本文件 | 自引用不写入；见交付回报 |

只读对照：`labels_summary.json`（14,207 字节，`709acbb5…`）与 `audit.json` 未改；`teacher1000.jsonl` 未改（11,931,643 字节，`a1d6923e…`）。`touhou_rl/v410/` 下 `teacher_labels.py` 等既有文件未改。

---

## 12 不宣称（`labels_calibrated_summary.json` 的 `not_claimed` 原文）

- 不是 G410-1 通过：这是 WP02.4 的模型一致性校准 + 一次既有教师 JSONL 的离线重标，没有评估或满足任何 gate/阈值/发布检查。
- 不是新数据集：输入是既有的 seed 1000 / mode fast / stage 0 教师 JSONL，没有启动、录制、训练或采集。
- 不是生存真值：校准阈值是在本语料上拟合的**模型一致性**阈值，不是生存判据；引擎的 `died` 才是唯一生存判定，`verified_dead` 依据的正是它。
