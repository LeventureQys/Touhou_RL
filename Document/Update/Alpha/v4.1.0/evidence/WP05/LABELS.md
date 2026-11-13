# WP05.2 — DAgger raw teacher 标签 → calibrated `labels_v410`（seed2000 真实转换与审计）

状态：**WP05.2 已完成（仅 seed2000 shard；未训练、未采新数据、未改任何既有 v410 文件）**

权威记录是 `labels_audit.json`（本文件只是它的索引与说明）。转换产物在
`.tmp_v410_wp05_labels/labels_v410.jsonl`（3,500 行，**不入库**）。

---

## 1. 这一步做了什么 / 没做什么

做：

- 新增适配层 `touhou_rl/v410/dagger/labels.py`：把 WP05.0 DAgger shard 的 raw
  teacher 行（`query_status` / `verified_mask` / `verified` / `raw_geo_score` /
  `chosen_plan` / `bomb` / `frame_event_flags`）转成 calibrated `labels_v410` 行；
- 用 WP02.4 既有的规则（不是新规则）给每个 verified plan 一个 `survival_status`，
  并写出 alive/dead/unknown 明细、候选级 mask 与审计字段；
- 对真实 seed2000 shard 做了一次真实转换 + 审计（CLI exit 0）；
- 65 个单测覆盖 alive/dead/unknown、形状违例、injected 字段、unknown 不作负例、
  obs 行数不匹配 fail-closed、CLI 退出码。

不做（本文件不主张）：

- 不训练、不采新数据、不启动引擎；
- 不是 G410-2，不是学生强度 / 通关率 / 死亡率结论；
- 不是 D1 work-row（DAgger 行没有 D1 的 `model_match` 通道，见 §3.2）；
- 不重新拟合 tolerance（记录 8.0 及其 WP02.4 出处，不在此重新推导）；
- 只转换了 seed2000 一个 shard，其他 shard 未读、未转。

---

## 2. 只读输入（已核对 sha256）

| 文件 | 事实 |
|---|---|
| `.tmp_v410_wp05_round1/shards/dagger-s0d1seed2000/shard.json` | schema `v410-dagger-shard/1`；seed 2000；stage 0；difficulty normal (index 1)；decisions 3500；frames 7000；`complete=true`；alignment 7000 检查点 `checkpoints_mismatched=0` / `all_equal=true`；teacher horizon 32、decision_frames 2 |
| `.../labels.jsonl` | rows 3500；bytes 8,699,478；sha256 `129a1edcefc4789f0114c54ef6aa119c3df4b24d0fc384fb927a63d3b00be033`（与 shard.json 记录一致） |
| `.../obs/play96x112.npy` | (3500, 4, 112, 96) uint8；sha256 `45a7bb7f64e6467b4f59bc7bbb2c676d9c9ba7249decc44e4efd48c0d000e347`（一致） |
| `.../obs/hud.npy` | (3500, 1, 56, 160) uint8；sha256 `a30c600db70cfd8991e2b1c158e30d5eeb809bc29863470a97fbb17b8b5ce488`（一致） |
| `.../obs/act_hist.npy` | (3500, 8, 18) uint8，全 0/1；sha256 `863371ba65f8b04b6bc1cb70b6882eba9dc42e680adab6598cf796119f9a9e36`（一致） |

原始行的 `survival_status` 在 **3500/3500** 行是 `null`（deferred，WP05.0 的明文契约），
`query_status_source` 是「written directly from verified_mask」。所以本步的校准不是
「重写已有结论」，而是**第一次**把 survival 通道算出来；raw 行一个键都没丢。

checkpoint / engine 是 shard.json 自报、本步未重跑：checkpoint
`.tmp_v410_wp04_arms/b0_seed0_last.pt`（WP04.2 B0 probe，731,802 参数）sha256
`3bc0ec4461701bebb2ffc551d61dffff77f15cd486125baf9db6eaafc741b3a6`；engine
`th08-modern` sha256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`。

tolerance 的出处（`labels_audit.json:calibration.module_tolerance`）：
`teacher_calibration.MODEL_MATCH_TOLERANCE = 8.0`，status `computed`，来自
`.tmp_v410_wp02_audit/teacher1000.jsonl`（sha256 `a1d6923e…`，died=false plans 3251，
覆盖 3226，coverage 0.992310，mismatch 25），规则
「smallest grid threshold with coverage >= 0.99 of died=false verified plans, capped at 16.0 px」。

---

## 3. 适配层（`touhou_rl/v410/dagger/labels.py`）

### 3.1 规则（WP02.4，不新造）

`convert_dagger_row(row, *, tolerance=8.0, horizon=32)` 对 `verified[]` 的每个 plan
调用 `touhou_rl.v410.teacher_calibration.classify_plan`（WP02.4 的同一实现）：

| 条件 | 结果 |
|---|---|
| `died=true` | `verified_dead`（引擎判决优先，无论 `max_position_error` 多小） |
| `died=false` 且 `max_position_error <= tolerance` 且 `frames_advanced >= max(horizon, duration × 2)` | `verified_alive` |
| 其余（误差缺失/超容差、帧数不足、horizon 未知） | `unknown`（`model_mismatch` / `incomplete_execution` / `horizon_missing`） |

`unknown` **永不**进入 `negative_mask`：`negative_mask` 恒等于 `dead_mask`
（只有引擎已验证的死亡才是负例），并写出 `negative_mask_rule` 明文。

行内新增：`verified[].survival_status/survival_reason/survival_detail/model_match/
required_frames/decision_frames/horizon`、`survival_status`（按 plan 顺序）、
`survival_counts`、`survival_reasons`、`alive_mask/dead_mask/unknown_mask/
positive_mask/negative_mask`、`calibration`、`model_match_tolerance`、
`horizon`/`horizon_source`/`horizon_missing`、`label_rule`、`raw_row_sha256`、
`provenance`。

### 3.2 明确不是 D1 work-row

每行显式 `d1_work_row=false`、`work_row_schema=null`，`provenance.note` 写明
「DAgger 行没有 D1 的 `model_match` 通道」。raw 行的 46 个键**一个不丢**
（原始 deferred 文本转存到 `provenance.raw_survival_status[_source]`），转化后
74 个键（新增 28 个校准字段）。不把 unknown 伪装成 D1 的负例，也不伪造
D1 的 `model_match` 明细。

### 3.3 `convert_shard(shard_dir, out_dir)`

fail-closed 清单：

- `shard.json` schema = `v410-dagger-shard/1`、`complete=true`、alignment 无 mismatch；
- `teacher.decision_frames` / `frames_per_decision` 必须 = 2（`protocol.DECISION_FRAMES`）；
- 三个 obs 平面存在、`uint8`、逐行形状正确、行数 = `decisions`、`act_hist` 全 0/1；
- `labels.jsonl` 行数 = `decisions` = obs 行数 = `shard.labels.rows`；
- 每行 `decision == 行号`、`obs_row == 行号`、`frame == 行号 × 2`、seed/shard/stage 与
  shard.json 一致；
- `query_status` 必须是 `verified_mask` 的严格互逆；`verified[].first_action` 必须在
  mask 内；`verified_count` / `verified_mask_count` 与数组一致；`raw_geo_score` 是 18 个
  有限数；
- `injected` 行的 `executed_cand` 必须 = `teacher_cand`，非注入行必须 = `student_cand`；
- sha256 校验（默认开）：labels.jsonl + 三个 obs 平面都与 shard.json 一致；
- 写临时文件 → 全量重读复核计数与 sha → 才落盘 `labels_v410.jsonl`；失败不留下输出。

输出：`labels_v410.jsonl`（每决策一行）+ `labels_v410.summary.json`。

---

## 4. 真实转换（实际命令与退出码）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m compileall -q touhou_rl/v410/dagger
# exit 0

python -m touhou_rl.v410.dagger.labels `
  --shard .tmp_v410_wp05_round1/shards/dagger-s0d1seed2000 `
  --out .tmp_v410_wp05_labels `
  --tolerance 8.0 `
  --audit-json Document/Update/Alpha/v4.1.0/evidence/WP05/labels_audit.json
# exit 0
# dagger_labels: decisions=3500 plans=3500 alive=3492 dead=0 unknown=8
#   model_mismatch=8 incomplete=0 query_verified=3500/63000 query_coverage=0.055556
#   injected=3500 disagreement=1167 tolerance=8.0 horizon=32
# dagger_labels: labels -> .tmp_v410_wp05_labels\labels_v410.jsonl
#   (3500 rows, sha256=3494caa6…b4cf2)
```

默认 tolerance 路径（不带 `--tolerance`）也跑过一次，输出到
`.tmp_v410_wp05_labels/default_tolerance_check/`：得到**同一个 sha256**
（`3494caa65470c2c9abcad4baeea234446ca789627eda69fad2b0fdc4f21b4cf2`），
`tolerance_source = "WP02.4 MODEL_MATCH_TOLERANCE"`、`tolerance = 8.0`。即默认会解析到
同一个 8.0，转换是确定性的。

---

## 5. seed2000 转换数字

### 5.1 决策 / plan

| 指标 | 数值 |
|---|---|
| decisions | **3,500** |
| verified plans | 3,500（每决策 1 个） |
| `verified_alive` | **3,492** |
| `verified_dead` | **0** |
| `unknown` | **8** |
| └ `model_mismatch` | 8 |
| └ `incomplete_execution` | 0 |
| └ `horizon_missing` | 0 |
| decision 级 alive / dead / unknown | 3,492 / 0 / 8 |

8 个 unknown 全部是「误差刚刚超容差」：`max_position_error` 落在
8.000003–8.000031 px（tolerance = 8.0），decision 分别是
**357 / 1177 / 1523 / 1765 / 1770 / 1784 / 1897 / 2336**。其余 3,492 个 plan 误差都
≤ 8.0（其中 1,591 个恰为 0.0）。

输入侧确定性事实（解释为什么 dead=0）：3,500 个 plan 全部 `died=false`、
`frames_advanced=32`（= horizon，满视界）、`duration` ∈ {1: 2,780 个；2: 720 个}；
`labels.jsonl` 的 `resources` 在 3,500/3,500 行都是 `deaths=0, lives=2`。

### 5.2 query 覆盖

| 指标 | 数值 |
|---|---|
| 候选网格 | 3,500 × 18 = **63,000** |
| `query_status == verified` 的候选 | **3,500** |
| candidate coverage | **5.5556%**（3,500 / 63,000） |
| 至少有一个 verified plan 的行 | 3,500 / 3,500（100%） |
| `query_status` 与 `verified_mask` 严格互逆 | 3,500 / 3,500 |
| `query_status` 与 `verified[].first_action` 一致 | 3,500 / 3,500 |

### 5.3 injected / executed（与 shard.json 交叉核对）

| 指标 | 数值 | shard.json 自报 |
|---|---|---|
| injected 行 | **3,500 / 3,500** | `injection.injections = 3500`，`injection_rate = 1.0`（β = 1.0） |
| 执行教师候选 | 3,500 | `injection.teacher_available = 3500` |
| 执行学生候选 | 0 | — |
| 学生/教师一致（agreement） | 2,333 | `injection.agreement = 2333` |
| 不一致（disagreement） | 1,167 | `injection.disagreement = 1167` |
| 执行炸弹 | 0 | — |

### 5.4 候选级 mask（每决策 18 个候选）

| mask | 合计 |
|---|---|
| `alive_mask` | 3,492 |
| `dead_mask` | 0 |
| `negative_mask`（= `dead_mask`） | **0** |
| `unknown_mask` | 59,508 |

`unknown_mask` = 18 × 3,500 − alive − dead，包含「从未被 query 的 59,500 个候选」和
「被 query 但结论是 unknown 的 8 个候选」。unknown 不计入负例。

### 5.5 输出文件

| 文件 | 规模 | sha256 |
|---|---|---|
| `.tmp_v410_wp05_labels/labels_v410.jsonl` | 3,500 行 / 19,661,798 B | `3494caa65470c2c9abcad4baeea234446ca789627eda69fad2b0fdc4f21b4cf2` |
| `.tmp_v410_wp05_labels/labels_v410.summary.json` | 转换汇总 | — |
| `Document/Update/Alpha/v4.1.0/evidence/WP05/labels_audit.json` | 汇总 + audit schema + verification | — |

两次运行（显式 8.0 / 默认解析）产物 sha256 相同 → 转换确定性。

---

## 6. 审计（`labels_audit.json`）

**结构检查全部 3,500/3,500 通过**（`checks` 字段）：labels_rows=3500、
shard_decisions=3500、obs_rows=3500、labels_rows_declared=3500；
decision / obs_row / frame / seed / shard / stage 对齐各 3500；
`query_status_mask_inverse`=3500、`query_status_plan_consistent`=3500、
`verified_first_action_in_mask`=3500、`injection_executed_consistent`=3500、
`agreement_matches_candidates`=3500（mismatched=0）、
`chosen_plan_present` / `chosen_plan_matches_chosen`=3500（mismatched=0）、
teacher_available=3500、bombs_executed=0、raw_survival_present=0、
horizon_missing_rows=0；`labels_sha256_verified=true`、`obs_sha256_verified=true`。

**写盘后独立重读**（`verification`）：rows=3500、alive=3492、dead=0、unknown=8、
bytes=19,661,798、sha256 同上、`matches=true`。

另有两项**不依赖 `labels.py` 自报**的独立复核：

1. 逐行重算原始 `labels.jsonl` 行的 sha256，与行内 `raw_row_sha256` 比对：3,500/3,500
   相等；raw 行键 0 缺失（+28 个校准键）；`injected / student_cand / teacher_cand /
   executed_cand / executed_bomb / draw / bomb / frame_event_flags / raw_geo_score /
   chosen_plan / teacher_slots / resources / hash_*` 与原始行逐字段相等：0 违例；
   `d1_work_row` 全部 `false`。
2. 用**既有**消费方 `touhou_rl.v410.student.s1_data.parse_label_row` 读全部 3,500 行：
   3,500/3,500 被接受，重建的 alive/dead 候选 = 3,492/0，与 `labels.py` 自报一致。
   （没有改 `s1_data`；这只是「行形状确实是既有 v410 消费者期望的 calibrated 行」的旁证。）

---

## 7. 测试

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp052_pytest -q tests/test_v410_dagger_labels.py
# 65 passed in 0.52s   (exit 0)
```

覆盖：alive；dead（含 `died` 优先于完美 match）；unknown（超容差、误差缺失、帧数
不足、duration 扩展 required frames）；unknown 不作负例；tolerance/horizon 合法性；
query_status 长度与词表；mask 非 bool；mask 计数；`verified_count`；query ↔ mask
互逆；plan 必须在 mask 内；`raw_geo_score` 有限性；injected 执行一致性；raw survival
冲突；D1 标记；`s1_data` 消费兼容；shard 级 fail-closed（行数 / obs 行数 /
act_hist 形状 / sha / `complete` / schema / alignment / frame / decision / 外行 /
空行 / CLI exit 2、3 且不留输出文件）。

`python -m compileall -q touhou_rl/v410/dagger` → exit 0（只代表语法，不等于测试通过）。

---

## 8. 不主张（not claimed）

- G410-2 未测量、未尝试；
- 没有训练、没有更新任何 learner；这些行还没有进入训练管线；
- 没有采新数据、没有启动引擎；只读了 seed2000 一个 shard；
- 不是 D1 work-row；DAgger 行没有 D1 的 `model_match` 通道；
- tolerance 8.0 是 WP02.4 的结果，本步只记录出处、不重新拟合；
- alive/dead 是**教师 plan 的验证判决**，不是学生强弱、不是通关率或死亡率结论；
- 其他 seed / shard（含 `.tmp_v410_wp05_smoke` 的 seed2000 / seed2001）没有转换。

---

## 9. 变更文件

新增（白名单内）：

- `touhou_rl/v410/dagger/__init__.py`
- `touhou_rl/v410/dagger/labels.py`
- `tests/test_v410_dagger_labels.py`
- `Document/Update/Alpha/v4.1.0/evidence/WP05/LABELS.md`
- `Document/Update/Alpha/v4.1.0/evidence/WP05/labels_audit.json`
- `.tmp_v410_wp05_labels/`（`labels_v410.jsonl`、`labels_v410.summary.json`、
  `default_tolerance_check/`；**不入库**）

未改：v400/v380/v390、`tools/rl/v400_*`、`tests/test_v400_*`、`third_party/`、v410
既有文件、`tools/rl/v410*.py`、`tests/test_v410*.py`、state/journal/budget/assets、
WP03/WP04 证据。没有 commit / push。

---

## 10. 剩余缺口与下一步建议

剩余（都不是本 WP 的交付物）：

- 首轮 50k 决策的 DAgger 数据还没采（WP05.0 只有 600 决策的 smoke；本步只转了一个
  3,500 决策 shard）；
- 其他已完成 shard 未转换（round1 目录里若有更多 seed，需要逐个跑同一 CLI）；
- `survival_status` 还没进训练（`student/bc.py` 自报 `survival_status_role=diagnostic_only`，
  本步不改它）；
- G410-2 未评测。

下一步建议：

1. 把同一 CLI 接进 WP05 批次入口：对 round1 每个 `complete=true` 的 shard 跑一次，
   聚合 `counts` / `checks` / `verification`（本步的 audit payload 已可直接聚合）；
2. 在 WP06 接入前确认训练侧只消费已验证信号：`negative_mask` 只用 `verified_dead`，
   `unknown` 保持不参与（本步已按此编码并有单测）；
3. 本轮真实数据 0 死亡，`verified_dead` 分支只在单测里被证明；等后续 shard 出现
   `died=true` 时，重点复核这些 plan 是否被正确排除出 `alive_mask`。
