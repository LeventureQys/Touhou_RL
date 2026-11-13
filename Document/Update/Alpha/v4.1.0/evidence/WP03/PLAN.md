# WP03.0 — D410-S1 / D410-MULTI 数据计划与 fail-closed 覆盖审计（证据）

- 工作包：v4.1.0 WP03.0
- 目的：先定义 D410-S1 / D410-MULTI 的数据计划与 fail-closed 覆盖校验器，再用已有的 20 个
  fast 教师原始 run 做一次**离线标签覆盖审计**；不启动游戏、不训练、不采集新数据。
- 代码：`touhou_rl/v410/dataset_plan.py`
- 测试：`tests/test_v410_dataset_plan.py`（35 例）
- 机器可读证据：`Document/Update/Alpha/v4.1.0/evidence/WP03/plan.json`、
  `Document/Update/Alpha/v4.1.0/evidence/WP03/s1_label_audit.json`
- 只读输入（WSL）：`/tmp/v410_fast20/1000.jsonl` … `1019.jsonl`，共 20 个 run，
  244,235,556 bytes；每个文件首行 `kind="report"`，其余为 `kind="decision"`，含
  `verified` 明细。
- 只读 import：`touhou_rl/v410/teacher_labels.py`、`touhou_rl/v410/teacher_calibration.py`；
  未修改 WP02 证据、v4.0/v3.8/v3.9 代码与 `tools/rl/v410*.py`。
- 环境：Windows 验收 Python 3.12.10；WSL 审计解释器 `/root/v400-venv/bin/python`
  （Python 3.10.12，WP02 的既有 venv）。
- 活动声明：**未启动游戏、未训练、未采集新数据、未 commit/push、未删除或覆盖旧文件**。
  所有审计数字由代码读取真实 JSONL 生成，本文件只引用生成结果。

## 0 一句话结论

**S1 的标签侧达标：66,412 个决策点、20 个种子、0 个重复键、66,030 alive / 0 dead /
382 unknown；但 20 个 fast 教师 run 没有任何观测字节，`observations_present=false`，
因此 S1 整体状态是 `blocked_missing_observations`，`passed=false`。**
D410-MULTI 只有计划、没有数据，校验器对空 manifest 明确失败（`failed_total_frames`）。
WP03.1 必须先真实采集观测，然后才能重跑本审计并谈 S1 覆盖通过。

## 1 实际执行的验收命令与退出码

### 1.1 单元测试（Windows，绕过 `.pytest_cache`）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp03_pytest -q tests/test_v410_dataset_plan.py
# 输出：35 passed in 0.18s
# 退出码：0
```

### 1.2 语法编译

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m compileall -q touhou_rl/v410
# 输出：（无）
# 退出码：0
```

### 1.3 真实 20-run 离线审计（WSL）

```bash
cd /mnt/d/WorkShop/Touhou_RL
/root/v400-venv/bin/python -m touhou_rl.v410.dataset_plan \
  --audit-dir /tmp/v410_fast20 \
  --audit-out Document/Update/Alpha/v4.1.0/evidence/WP03/s1_label_audit.json \
  --plan-out  Document/Update/Alpha/v4.1.0/evidence/WP03/plan.json
# dataset_plan: runs=20 decisions=66412 unique_decisions=66412 seeds=20 \
#   alive=66030 dead=0 unknown=382 duplicates=0 observations_present=false \
#   s1=blocked_missing_observations multi=failed_total_frames
# dataset_plan: audit -> .../WP03/s1_label_audit.json \
#   (sha256=c5d2f92aa2f0eb290990883259b193b0183688867773323e4b8582b083e542e6, bytes=19961)
# dataset_plan: plan -> .../WP03/plan.json
# 退出码：0
```

### 1.4 fail-closed 门检查（预期非零）

```bash
cd /mnt/d/WorkShop/Touhou_RL
/root/v400-venv/bin/python -m touhou_rl.v410.dataset_plan \
  --audit-dir /tmp/v410_fast20 --require-pass --quiet
# stderr: dataset_plan: FAILED (fail-closed): --require-pass set and coverage did not pass \
#   (s1=blocked_missing_observations, multi=failed_total_frames)
# 退出码：4
```

说明：1.3 的退出码 0 表示**审计证据成功生成**，不是覆盖通过；覆盖结论在 JSON 的
`s1_coverage.status` / `multi_coverage.status`。1.4 证明 `--require-pass` 会把同一份
审计判为失败并返回非零。

## 2 pytest 统计

| 指标 | 值 |
|---|---|
| collected | 35 |
| passed | 35 |
| failed | 0 |
| skipped | 0 |
| error | 0 |
| wall time | 0.18 s |
| 退出码 | 0 |
| 绕过 `.pytest_cache` | `-p no:cacheprovider --basetemp .tmp_v410_wp03_pytest` |

覆盖点：

- S1 计划字段、种子数（20）、`no_synthetic_padding=true`、观测平面；
- S1 validator 的 50,000 边界（49,999 失败）、20 seeds 边界（19 失败）、每个 seed > 0；
- `observations_present` 缺失 / `false` / 字符串 `"true"` 都得到
  `blocked_missing_observations`，完美标签也不能通过；
- `synthetic_padding=true` 失败；重复 decision key 失败；
- MULTI 的 2,000,000 / 150,000 边界、缺 stage、重复轨迹、缺观测、缺标签；
- 合成 teacher JSONL：report+decision、坏 JSON、未知 kind、重复 key、
  calibrated alive/dead/unknown、跨文件重复 `(seed, decision)`、seed 与文件名不一致、
  缺 report 行。

## 3 真实审计关键数字

来源：`Document/Update/Alpha/v4.1.0/evidence/WP03/s1_label_audit.json`
（schema `v410-s1-label-audit/1`，sha256 `c5d2f92a…e542e6`，19,961 bytes）。

| 指标 | 值 |
|---|---|
| runs | 20 |
| seeds | 1000–1019（20 个，missing/unexpected 均为空） |
| total_decisions | 66,412 |
| unique_decisions | 66,412 |
| per-seed decisions | 最小 3,251（seed 1000），最大 3,419（seed 1014） |
| verified_plans | 66,412 |
| alive（calibrated `verified_alive`） | 66,030 |
| dead（`verified_dead`） | 0 |
| unknown | 382 |
| died_true / died_false plans | 0 / 66,412 |
| query entries | 1,195,416（66,412 × 18） |
| query verified / unknown | 66,412 / 1,129,004 |
| query coverage | 0.055556（1/18：每个 decision 只有 1 个 verified plan） |
| duplicate decision keys | 0（within_run 0 + cross_file 0） |
| observations_present | **false** |
| decisions_with_observations | 0 |
| observation_keys_seen | `[]` |
| calibrated tolerance | 8.0 px（WP02.4 数据推导值） |
| decision_frames | 2 |
| synthetic_padding | false |
| S1 标签侧 `labels_cover_s1` | **true** |
| S1 整体 `status` / `passed` | **blocked_missing_observations / false** |

### 3.1 每个种子的 decisions / alive / unknown

| seed | decisions | alive | unknown |
|---:|---:|---:|---:|
| 1000 | 3,251 | 3,226 | 25 |
| 1001 | 3,275 | 3,259 | 16 |
| 1002 | 3,323 | 3,299 | 24 |
| 1003 | 3,323 | 3,309 | 14 |
| 1004 | 3,323 | 3,300 | 23 |
| 1005 | 3,299 | 3,279 | 20 |
| 1006 | 3,323 | 3,304 | 19 |
| 1007 | 3,347 | 3,332 | 15 |
| 1008 | 3,323 | 3,300 | 23 |
| 1009 | 3,371 | 3,349 | 22 |
| 1010 | 3,347 | 3,320 | 27 |
| 1011 | 3,299 | 3,285 | 14 |
| 1012 | 3,275 | 3,257 | 18 |
| 1013 | 3,299 | 3,282 | 17 |
| 1014 | 3,419 | 3,395 | 24 |
| 1015 | 3,275 | 3,260 | 15 |
| 1016 | 3,275 | 3,260 | 15 |
| 1017 | 3,371 | 3,354 | 17 |
| 1018 | 3,371 | 3,357 | 14 |
| 1019 | 3,323 | 3,303 | 20 |
| **合计** | **66,412** | **66,030** | **382** |

### 3.2 first_action 分布（verified plans）

| action | verified_plans | alive | dead | unknown |
|---:|---:|---:|---:|---:|
| 0 | 29,225 | 29,144 | 0 | 81 |
| 1 | 1,444 | 1,432 | 0 | 12 |
| 2 | 5,962 | 5,907 | 0 | 55 |
| 3 | 6,764 | 6,741 | 0 | 23 |
| 4 | 1,383 | 1,374 | 0 | 9 |
| 5 | 759 | 758 | 0 | 1 |
| 6 | 4,573 | 4,505 | 0 | 68 |
| 7 | 4,566 | 4,564 | 0 | 2 |
| 8 | 787 | 762 | 0 | 25 |
| 9 | **0** | 0 | 0 | 0 |
| 10 | 186 | 186 | 0 | 0 |
| 11 | 3,988 | 3,960 | 0 | 28 |
| 12 | 4,754 | 4,714 | 0 | 40 |
| 13 | 407 | 401 | 0 | 6 |
| 14 | 95 | 93 | 0 | 2 |
| 15 | 747 | 729 | 0 | 18 |
| 16 | 490 | 487 | 0 | 3 |
| 17 | 282 | 273 | 0 | 9 |
| **合计** | **66,412** | **66,030** | **0** | **382** |

action 0（idle）占 44.0%，action 9 从未被验证；这是 20 个 fast run 的实际分布，不是
采集目标，也不构成策略质量结论。

## 4 计划与 validator 结果（plan.json）

`plan.json`（schema `v410-wp03-plan-evidence/1`）包含：

- `s1_plan`：
  - scope `D410-S1`，stage 0，difficulty normal，team 0，seeds 1000–1019；
  - `min_decisions=50000`、`min_seeds=20`、`decision_frames=2`、`obs_spec="v410"`；
  - `required_label_fields`（含 `source_run_id` / `trajectory_id` / `obs_sha` /
    `action_history` / `verified` / `query_status` / `survival_status` 等）；
  - `obs_planes`：play 112×96 uint8（4-frame stack）、HUD 56×160 uint8（4-frame
    stack）、action_history 8 int64、bomb uint8；
  - `observations_required=true`、`labels_required=true`、
    `no_synthetic_padding=true`；trajectory-grouped 70/15/15 split。
- `multi_plan`：
  - scope `D410-MULTI`，stages 0–7，`total_frames_target=2,000,000`、
    `min_frames_per_stage=150,000`、每 stage 计划目标 250,000；
  - `unique_trajectories_required=true`、`min_unique_trajectories=8`；
  - start mix 目标 50% midway / 30% natural_late / 20% student_failure，并明确
    student_failure 在无学生轨迹时只能声明缺失，不能静默计 0；
  - 同样要求 observations + labels + no synthetic padding。
- `s1_coverage`：对 20-run 审计运行 `validate_s1_coverage` 的结果。
  - `labels_cover_s1=true`；标签检查全部通过（66,412 ≥ 50,000、20 ≥ 20、
    每个 seed > 0、0 duplicate、synthetic_padding=false）。
  - 唯一失败项 `observations_present_is_true`；`status="blocked_missing_observations"`、
    `passed=false`、`blocked=true`。
- `multi_coverage`：对 `--audit-dir` 之外的显式空 MULTI manifest
  （`total_frames=0`、无 stage、无 trajectory、obs/labels=false）运行
  `validate_multi_coverage` 的结果：`status="failed_total_frames"`、`passed=false`，
  并同时列出 missing stages 0–7、0 个 unique trajectory、obs/labels 缺失。
  这证明 MULTI 校验器不会把"计划存在"当成"数据存在"。
- `s1_label_audit`：审计文件路径、sha256、bytes、runs、decisions、seeds、
  observations_present 与 status，便于从 plan.json 直接核对原始审计。

`plan.json` 中 S1 覆盖结论原文：

```text
labels_cover_s1 = true
status          = blocked_missing_observations
passed          = false
reasons         = ["observations_present_is_true: observations_present is missing or not
                   explicitly true (False); labels alone cannot satisfy D410-S1"]
```

## 5 限制（必须连同数字一起读）

1. **无观测，这是硬阻塞。** 20 个 teacher JSONL 只有 state 标量、候选分数、
   `verified` 计划与 timings；没有 play/HUD 像素、4-frame stack、8-action history。
   因此 `observations_present=false`、`decisions_with_observations=0`。标签覆盖
   完整也不等于可训练数据完整。
2. **单一 scope。** 20 个 run 全是 fast / stage 0 / normal，不能外推为多面、
   多难度或确认种子覆盖。
3. **原始 teacher 行不是最终 label contract。** 审计用 calibrated converter 派生出
   `query_status` / `survival_status` 等，但原始行没有 `source_run_id` /
   `trajectory_id` / `trajectory_parent` / `obs_sha` / `action_history` /
   `start_mode` / `burn_in_frames`；这些必须在 WP03.1 的真实采集里写入。
4. **alive / unknown 的口径。** `alive=66,030` 是 WP02.4 calibrated 规则
   （`died=false` + `max_position_error <= 8.0` + 足够 `frames_advanced`），不是存活
   保证；382 个 `unknown` 不是死亡，不能当负样本。
5. **dead=0 是语料属性。** 20 个 run 的 report 都是 `deaths_observed=0`，死亡分支由
   WP02 合成测试覆盖，不是本语料覆盖。
6. **query coverage 只有 1/18。** 每个 decision 只验证了一个候选计划，
   `verified=66,412`、`unknown=1,129,004`；不能把未验证候选当安全负例。
7. **first_action 分布不是策略质量。** action 0 占 44.0%，action 9 为 0；这只是
   20 个 fast run 的行为分布。
8. **MULTI 完全没有数据。** 2,000,000 帧 / 每 stage 150,000 / 8 stages 目前只是
   计划与 fail-closed 校验器；`multi_coverage` 对空 manifest 失败是预期结果。
9. **WSL 解释器是 3.10.12。** Windows 验收测试用 3.12.10；WSL 审计沿用 WP02 的
   `/root/v400-venv`（3.10.12）。两类解释器都实际跑过。
10. **审计只证明标签计数与字段可读。** 它不证明像素可学习性、策略质量、G410-1
    通过或数据集验收。

## 6 下一建议（WP03.1 / WP03.2）

1. **WP03.1（必须先做）：真实采集 S1 观测。** 为 seeds 1000–1019 写 play 112×96、
   HUD 56×160、4-frame stack、8-action history、bomb，并为每个 decision 落
   `obs_sha`；同时写 `source_run_id`、`trajectory_id`、`trajectory_parent`、
   `engine_sha`、`protocol_sha`、`teacher_config_sha`、`start_mode`、`burn_in_frames`。
2. **重跑同一审计。** 用本次命令把 `observations_present` 变成显式 `true`；只有
   标签 + 观测都通过，`validate_s1_coverage` 才会给出 `passed=true`。
3. **多计划覆盖。** 当前每个 decision 只有 1 个 verified plan（query coverage 1/18）。
   WP03.1 应至少让一部分 decision 带 `verify_k>1`，使 `unknown`/`dead` 分支在
   真实数据里也有覆盖，而不是只靠合成测试。
4. **WP03.2：启动 MULTI 采集。** 先按每 stage 250,000 帧推进，保证 8 个 stage 各
   ≥150,000、总帧 ≥2,000,000、trajectory id 唯一；`student_failure` 起点在无学生
   轨迹时必须显式声明为未满足，不能静默置 0。
5. **保持 fail-closed。** 任何缺 obs、缺 label、synthetic padding、重复决策键、
   缺 stage、短 stage 或重复轨迹都应继续让校验器失败；不得为了"先出数据"放宽阈值。

## 7 not claimed

- 不是 G410-1 通过：本工作包只写计划与标签审计，没有跑任何门。
- 不是新数据集：未启动游戏、未训练、未采集，20 个 teacher JSONL 是只读输入。
- 不是观测数据集：`observations_present=false`，S1 被明确 blocked。
- 不是 D410-MULTI 数据：MULTI 只有预注册计划，校验器对空 manifest 失败。
- 不是存活真值：`verified_alive` 是 calibrated 模型一致性判定；引擎 `died` 才是
  存活裁决，`unknown` 永远不是死亡。
