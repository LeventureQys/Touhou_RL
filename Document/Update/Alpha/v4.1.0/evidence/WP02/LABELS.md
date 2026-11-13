# WP02.3 — 新教师标签写出契约与离线转换

- 工作包：v4.1.0 WP02.3（只写代码 / 测试 / 离线转换；不启动游戏、不训练、不采集、不 commit/push）
- 产物：`touhou_rl/v410/teacher_labels.py`、`tests/test_v410_teacher_labels.py`、本文件、`labels_summary.json`、`labels_seed1000_sample.jsonl`
- 状态：**契约与离线转换完成**（pytest 60 passed / compileall / 真实转换 全部 exit 0）。这是标签契约 + 一次离线转换，**不是** G410-1 通过，**不是**新数据集，**不是**能力结论。

---

## 0 一句话结论

对 seed 1000 的真实教师 JSONL（3,251 条决策）做离线转换：1 条 `report` 行被跳过，3,251 条决策全部转成 `v410-teacher-label/1`，0 行被拒；每条 `verified` 明细（`died` / `death_frame` / `frames_advanced` / `trajectory_matches` / `max_position_error` / `events` / `end_xy`）逐条保留并各自给出 `survival_status`；18 个候选各自的 `query_status` 区分"验证问过 / 没问过"。严格判定结果为 **verified_alive 795 / verified_dead 0 / unknown 2,456**。D1 那种"`verified_mask=True` 就当活着"的读法不再成立：本样本 `verified_mask` 为 True 的行有 3,251 条，而能被判成 alive 的只有 795 条。

---

## 1 范围与不变量

- 只读输入：`.tmp_v410_wp02_audit/teacher1000.jsonl`（11,931,643 字节，sha256 `a1d6923efd8398619632d086bd854677e27cfd501158d2aeaafb37c9ca2f434b`，3,252 行 = 1 条 report + 3,251 条 decision）。未写入、未移动、未删除该文件。
- 只读参考：`Document/Update/Alpha/v4.1.0/evidence/WP02/audit.json`（WP02.1），本文件引用它的 D1 结论；未修改。
- 未启动游戏、未训练、未采集、未 commit/push。
- 未修改白名单外文件：`touhou_rl/v400/`、`v380/`、`v390/`、`tools/rl/v400_*`、`tools/rl/v410*.py`、`tests/test_v400_*`、`touhou_rl/v410/` 既有文件（`teacher_audit.py` 等）、`third_party/`、`state.json`/`journal.jsonl`/`budget.jsonl`/`assets.json`、WP02 的 `audit.json`/`AUDIT.md` 全部未动（`audit.json` 的 sha256 仍是 `faa697ca...`，与 `AUDIT.md` 记录一致）。
- 未删除/覆盖旧文件；新增文件仅白名单内的 5 个。

---

## 2 契约：`v410-teacher-label/1`

常量：`LABEL_SCHEMA="v410-teacher-label/1"`、`QUERY_STATUSES=("verified","unknown")`、`SURVIVAL_STATUSES=("verified_alive","verified_dead","unknown")`。

### 2.1 一条标签行

| 字段 | 含义 | 本样本 |
|---|---|---|
| `schema` / `kind` | `v410-teacher-label/1` / `label` | — |
| `decision` / `frame` | 源行决策序号与帧 | 0..3250 |
| `verified_mask` | 源行 18 元素 bool 数组，**原样保留** | 每行恰 1 个 True |
| `verified_mask_means` | 固定 `"verification_ran_not_survival"`：mask=True 只表示"验证返回过这个首动作"，不是存活 | 3,251 行 |
| `query_status` | 18 项：`verified` 当且仅当 `verified` 列表里存在该 `first_action`；否则 `unknown`。未验证候选**永不写 dead** | verified 3,251 / unknown 55,267 |
| `verified` | 源 `verified` 列表**逐条原样**（所有键保留），另加 `plan_index` / `survival_status` / `survival_reason` / `required_frames`；同一 `first_action` 的多条计划**不合并** | 每行 1 条 |
| `survival_status` | 与 `verified` 等长的状态列表（按 `plan_index` 顺序） | 见 §2.2 |
| `survival_counts` | 该行三种状态计数 | — |
| `raw_geo_score` | 源 `geo_score` 数组（18 个 float，逐字保留；契约名如此，不在本模块重算） | 18 个/行 |
| `post_verify_scores` | 只有行内自己给出 `unverified_discount` 时才推导 `{unverified_discount, source, verified_actions, unverified_actions, implied_raw_geo_score}`；否则 `null` | 全 `null` |
| `chosen_plan` / `margin` / `bomb` / `frame_event_flags` / `events` / `timing_ms` | 源行同名字段原样透传（缺失即 `null`） | 有 |
| `source_horizon` / `source_horizon_key` / `horizon_missing` | 行内 `verification_horizon`（或 `horizon`）；都没有则 `null` + `true` | `null` / `true` 全部 |
| `frames_per_decision` / `frames_per_decision_source` | 一个决策槽推进的帧数及来源（参数 > `row.frames_per_decision` > `row.decision_frames` > report 推导） | `2`，来源 report 商 |
| `plan_count` / `verified_mask_matches_verified_actions` | 已验证计划条数；mask 与实际 verified 首动作集合是否一致 | 1 / 全部一致 |

### 2.2 存活判定（对每条 verified 计划独立判定，fail-closed）

| `died` | `trajectory_matches` | `frames_advanced` | `duration * frames_per_decision` | 结果 |
|---|---|---|---|---|
| `True` | 任意 | 任意 | 任意 | `verified_dead`（`died_true`） |
| `False` | 缺失 | — | — | `unknown`（`trajectory_matches_missing`） |
| `False` | `False` | — | — | `unknown`（`trajectory_matches_false`） |
| `False` | `True` | 任意 | `duration`/帧预算缺失 | `unknown`（`duration_missing` / `frames_per_decision_missing`） |
| `False` | `True` | `null` | 已知 | `unknown`（`frames_advanced_missing`） |
| `False` | `True` | `< 需要帧数` | 已知 | `unknown`（`frames_advanced_below_required`） |
| `False` | `True` | `>= 需要帧数` | 已知 | **`verified_alive`** |

`died=False` 本身**不构成 alive**；`verified_mask=True` 不参与存活判定。

### 2.3 fail-closed 规则

- `geo_score`/`scores` 必须恰好 18 个数值（bool 不算数值）；`verified_mask` 恰好 18 个 bool；`first_action` 必须是 0..17 的 int（bool 拒绝）；`died` 必须 bool；`frames_advanced` 必须 int 或 `null`；`duration` 必须 >=1 int 或缺失；`kind` 缺失/非 `decision`/`report` 一律报错。
- 违反者抛 `TeacherLabelError`；`stream_convert` 里坏 JSON、空行、坏形状行计入 `skipped` 并写入 `errors`（最多记 20 条，总数在 `error_count`）；CLI 只要 `skipped > 0` 就 **exit 3**，带洞的语料不能伪装成转换成功。

---

## 3 命令与退出码（全部实际执行）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
```

### 3.1 pytest

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp023_pytest -q tests/test_v410_teacher_labels.py
```

```
............................................................             [100%]
60 passed in 0.11s
PYTEST_EXIT=0
```

> `60 passed` 是稳定值；`in 0.11s` 是当次实测，重跑会随机器浮动。

### 3.2 编译校验

```powershell
python -m compileall -q touhou_rl/v410
COMPILEALL_EXIT=0
```

> exit 0 只代表语法可通过，不等于测试通过；测试统计见 3.1。

### 3.3 真实转换（CLI）

```powershell
python -m touhou_rl.v410.teacher_labels --input .tmp_v410_wp02_audit/teacher1000.jsonl --summary Document/Update/Alpha/v4.1.0/evidence/WP02/labels_summary.json --sample Document/Update/Alpha/v4.1.0/evidence/WP02/labels_seed1000_sample.jsonl --sample-limit 100
```

```
teacher_labels: decisions=3251 report_rows=1 skipped=0 alive=795 dead=0 unknown=2456 horizon_missing=3251 verified_plans=3251 query_verified=3251 query_unknown=55267
teacher_labels: summary -> Document\Update\Alpha\v4.1.0\evidence\WP02\labels_summary.json
teacher_labels: sample -> Document\Update\Alpha\v4.1.0\evidence\WP02\labels_seed1000_sample.jsonl (100 rows, sha256=74792575588c9a6df47ec474ad0b6be2e4c4bbcd155f0910700833d45926612d)
CLI_EXIT=0
```

### 3.4 证据可复现性校验（只读，不写文件）

```powershell
python -c "import json;from touhou_rl.v410 import teacher_labels as tl;c=tl.stream_convert('.tmp_v410_wp02_audit/teacher1000.jsonl');d=json.load(open('Document/Update/Alpha/v4.1.0/evidence/WP02/labels_summary.json',encoding='utf-8'));assert tl.summarize(c)==d['labels'];print('summary reproducible:',c['decisions'],c['alive'],c['dead'],c['unknown'],c['horizon_missing'])"
```

```
summary reproducible: 3251 795 0 2456 3251
EXIT1=0
```

```powershell
python -c "import json;from touhou_rl.v410 import teacher_labels as tl;rows=[json.loads(l) for l in open('Document/Update/Alpha/v4.1.0/evidence/WP02/labels_seed1000_sample.jsonl',encoding='utf-8')];assert len(rows)==100;assert all(r['schema']==tl.LABEL_SCHEMA and len(r['query_status'])==18 and len(r['verified_mask'])==18 and all('died' in p for p in r['verified']) for r in rows);print('sample ok:',len(rows),'decisions',rows[0]['decision'],'..',rows[-1]['decision'])"
```

```
sample ok: 100 decisions 0 .. 335
EXIT2=0
```

- `labels_summary.json` 的 `labels` 块与重新跑 `stream_convert` 得到的计数**逐字段相等**（`tl.summarize(counts) == summary["labels"]`）。
- 样本 100 行逐行校验：`schema`/`kind` 正确、`query_status` 18 项且取值合法、`verified_mask` 18 个 bool、`raw_geo_score` 18 项、`verified[i]` 含 `died`/`death_frame`/`frames_advanced`/`trajectory_matches`/`max_position_error`，决策号 0..335 严格升序无重复。

---

## 4 真实转换数字（seed 1000 / mode fast / stage 0 / difficulty 1）

源文件：3,252 行 = **1 条 report + 3,251 条 decision**；`report.decisions=3251`，`report.deaths_observed=0`，`config={horizon:32, beam:24, verify_k:1, durations:[1,2]}`，`decisions_per_decision_frames=2`，`committed_frames_per_decision=4`。

| 指标 | 值 |
|---|---|
| `report_rows` | 1（不写标签） |
| `decisions` / `written` | 3,251 / 0（summary 走 `out_path=None`，只计数不写行；样本另走一次，写 100 行） |
| `skipped` / `error_count` | 0 / 0 |
| `verified_plans_total` | 3,251（每决策恰 1 条 verified 计划，`verify_k=1`） |
| `survival.verified_alive` | **795**（24.454%） |
| `survival.verified_dead` | **0** |
| `survival.unknown` | **2,456**（75.546%） |
| `horizon_missing.decisions` | 3,251（100%） |
| `query.entries` | 58,518（3,251 × 18） |
| `query.verified` / `query.unknown` | 3,251（5.5556%）/ 55,267 |
| `frames_per_decision.values` / `source` | `[2]` / `report.committed_frames_per_decision/report.decisions_per_decision_frames`（4/2） |
| `mask_mismatch_decisions` | 0 |
| `decisions_without_verified_plans` | 0 |
| `post_verify_scores_present` | 0 |
| `plan_signals` | `died_true=0`、`died_false=3,251`、`trajectory_matches_true=795`、`false=2,456`、`missing=0` |
| `consistency` | `decisions_match_report=true`、`deaths_match_report=true`（0 == 0） |
| `diagnostics` | `alive_first_actions=[0]`、`alive_only_for_action_0=true`、`alive_equals_trajectory_matches_true=true` |

### 4.1 `per_first_action`（verified 计划数 / alive / dead / unknown）

| action | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plans | 1532 | 69 | 290 | 326 | 71 | 34 | 212 | 208 | 42 | 0 | 8 | 179 | 198 | 16 | 3 | 35 | 16 | 12 |
| alive | 795 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dead | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| unknown | 737 | 69 | 290 | 326 | 71 | 34 | 212 | 208 | 42 | 0 | 8 | 179 | 198 | 16 | 3 | 35 | 16 | 12 |

### 4.2 样本 `labels_seed1000_sample.jsonl`

100 行、204,203 字节、sha256 `74792575588c9a6df47ec474ad0b6be2e4c4bbcd155f0910700833d45926612d`；按决策号排序，覆盖决策 0..335；类别 **alive 50 / dead 0 / unknown 50**。抽样规则是确定性的：按"决策级存活类别"等额配额（缺类不占配额、余量按类别顺序补齐）后按 `decision` 排序，重跑字节一致——不是前 100 行的前缀抽样（前缀里 100 条全是 unknown，什么也证明不了）。

### 4.3 支撑事实（只读统计，用于解释 §4 与 §6）

- `frames_advanced`：3,251 条全部 = 32（= `config.horizon`）。
- `duration`：1 共 2,685 条，2 共 566 条；795 条 alive 的 `duration` 全为 1、`first_action` 全为 0。
- `max_position_error`：795 条 alive 全为 **0.0**；2,456 条不匹配的最小 1.414212、中位 4.0、最大 139.583771（431 条恰为 4.0）。
- `chosen_plan.first_action` 与 verified 计划首动作一致：3,251/3,251；`bomb` 全为 0；`events` 全部含 `survive`。

---

## 5 与 D1 旧标签的差异

D1 结论引用 `audit.json`（WP02.1）的 `findings`：`verified_mask_true_is_not_survived`、`d1_rows_cannot_recheck_survival`、`bomb_without_player_hit_proves_engine_death_of_a_verified_plan`。

| 维度 | D1 行（`v400-teacher-obs-1`） | 本契约（`v410-teacher-label/1`） |
|---|---|---|
| `died` / `death_frame` | 不写（`verified` 列表整段被丢，`chosen_plan` 是 `search.Plan`，无该字段） | 逐条保留在 `verified[i]`，并驱动 `verified_dead` |
| `frames_advanced` / `trajectory_matches` / `max_position_error` | 被丢弃（audit 记录 34,013 行全部 `absent`） | 逐条保留；`trajectory_matches` 是 alive 的必要条件 |
| `verified_mask` | 每行 True，被当成"活着"误读 | 保留，但打 `verified_mask_means="verification_ran_not_survival"`；不参与存活判定 |
| 存活 | 无法复核 | `survival_status`（每条计划）+ `survival_reason` + `required_frames`，可逐条复算 |
| 查询覆盖 | 无此概念 | `query_status` 18 项：`verified` / `unknown`，未验证候选不写 dead |
| 分数语义 | `scores` 实为"验证后"分数（verified 替换 + 未验证 -5），却看不出 | `raw_geo_score` 命名点明来源，`post_verify_scores` 只在行内给出折扣时推导 |
| horizon | 不写（audit：`verification_horizon` 34,013 行全 absent） | `source_horizon` + `horizon_missing` 显式记录缺失 |
| 数字对照 | 34,013 决策、`verified_mask=True` 34,013、无法判定任何存活 | 3,251 决策、mask=True 3,251、**alive 795 / dead 0 / unknown 2,456** |

> 两侧样本不同（D1 是 10 个 shard 的抽样，本文件是单次 seed 1000 运行），所以只能比较**语义**，不能直接比较比例。

---

## 6 限制（这些数字能说什么、不能说什么）

1. **alive 只覆盖 action 0（idle）且 duration=1**。795 条 alive 的 `first_action` 全是 0、`duration` 全是 1、`max_position_error` 全是 0.0；移动动作的引擎轨迹与搜索模型偏差 1.41–139.58 px（中位 4.0），`trajectory_matches=False` → `unknown`。所以 795 不是"795 次机动存活"，而是"引擎未判死且模型位置被完全复现"的 idle 计划；生存标签目前对机动动作几乎无信息量。
2. **`unknown` 不等于"死"**。2,456 条 unknown 的引擎 `died` 也是 `False`（32 帧内没判死），但滚动轨迹与搜索模型不一致（`verify.py` 把它解释为"tape 过期/运动模型不符"，需要重录），因此契约拒绝升级为 alive，也不写成 dead。
3. **`verified_dead = 0` 未被真实数据覆盖**。源 report 写明 `deaths_observed=0`，death 分支只由合成测试覆盖（`test_verified_dead_keeps_death_frame_and_verdict` 等）。
4. **`horizon_missing` 为 100%**：所有决策行都没有 `verification_horizon`/`horizon`。契约不因它阻塞 alive（alive 依据 `duration*frames_per_decision` 帧数可自证），但"horizon 缺失"这一事实被记录，不当作通过。
5. **`frames_per_decision` 不是行内字段**，由 report 的 `committed_frames_per_decision / decisions_per_decision_frames = 4/2 = 2` 推导；若 report 缺失或不可整除，所有非死计划一律 `unknown`（`frames_per_decision_missing`）。
6. **`post_verify_scores` 全为 `null`**：行内没有 `unverified_discount`，契约不拿代码常量 5.0 冒充行内字段（该常量只在 `labels_summary.json` 的 `score_semantics` 里作为代码事实记录）。
7. **没有改 v4.0 writer**（禁止修改 `touhou_rl/v400/`），所以还没有产出新的 D1-like 语料；本文件只是"契约 + 离线转换 + 一个样本"。
8. **样本是确定性分层抽样**（alive 50 / unknown 50 / dead 0），不是随机抽样，也不是前 100 行；样本里能看到 alive 行，但看不到 dead 行。
9. **单次运行**：seed 1000、mode fast、stage 0 normal、difficulty 1；不代表其它 stage/mode/难度。

---

## 7 下一建议

1. **WP03 只能把 `survival_status="verified_alive"` 当正样本**；`unknown` 既不能当正样本也不能当负样本（否则会把"模型不同步"训练成"会死"）。负样本需要真实死亡 run。
2. **补行字段**（录制侧）：`verification_horizon`、`frames_per_decision`、`unverified_discount`（或直接写 post-verify 数组）。否则本契约里 horizon 永久 missing、`post_verify_scores` 永久 null。
3. **先修 `trajectory_matches`**：移动动作系统性偏差 1.4–139.6 px，说明搜索运动模型/tape 与引擎不同步；在修好之前，alive 只覆盖 idle，生存标签没有区分度。这属于 WP02.2 / WP04 的模型一致性工作，不是本契约能修的。
4. **用一次会产生死亡的真实 run 复核 `verified_dead`**（本样本 0 条），确认 death 分支在真实数据上的字段与数字。
5. **落盘新语料时**：新增 v4.1 writer 调用 `convert_teacher_row`，在 shard 元数据里记 `LABEL_SCHEMA`；不要改 v4.0 writer，也不要回写旧 D1 分片。
6. **给 WP03 传枚举**：训练侧用 `QUERY_STATUSES` / `SURVIVAL_STATUSES` 做校验，别把 `"verified_alive"` 之类的字符串写死在几处。

---

## 8 产物清单

| 文件 | 字节 | sha256 |
|---|---|---|
| `touhou_rl/v410/teacher_labels.py` | 55,894 | `e1e30e32397535dd2ea1437e873d463f267c734169b3ddc000d9c528963faa6f` |
| `tests/test_v410_teacher_labels.py` | 30,772 | `1c9e85e97df431fa29eb5b1682c280b47d2bda28ecd72088ef7d13f68be6109c` |
| `Document/Update/Alpha/v4.1.0/evidence/WP02/labels_summary.json` | 14,207 | `709acbb5d8221072dc0a93299a104515b799ae4a05274f5824ed7de876fbfebd` |
| `Document/Update/Alpha/v4.1.0/evidence/WP02/labels_seed1000_sample.jsonl` | 204,203 | `74792575588c9a6df47ec474ad0b6be2e4c4bbcd155f0910700833d45926612d` |
| `Document/Update/Alpha/v4.1.0/evidence/WP02/LABELS.md` | 本文件 | 自引用不写入；见交付回报 |

未修改任何既有文件（`audit.json` sha256 仍为 `faa697ca558ece5cd62916c9d6c6ce9b967c9e2ee33fab78fef12af327899241`，与 `AUDIT.md` 记录一致），未删除、未覆盖，未 commit/push。

---

## 9 不宣称（`labels_summary.json` 的 `not_claimed` 原文）

- 不是 G410-1 结果：这是 WP02.3 的标签契约 + 一次既有教师 JSONL 的离线转换，没有评估或满足任何 gate/阈值/发布检查。
- 不是新数据集：输入是既有的 seed 1000 / mode fast / stage 0 教师 JSONL，没有启动、录制、训练或采集。
- 不是能力结论：不从这些标签计数推断胜率、生存率、教师强度或策略质量。
- 不是 D1 重写：`.tmp_v400_d1` 分片、v4.0 writer、WP02.1 audit 均未改动；本文件只定义新标签行必须携带什么。
