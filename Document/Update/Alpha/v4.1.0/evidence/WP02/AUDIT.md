# WP02.1 — D1 教师标签语义与可审计性审计（只读）

- 工作包：v4.1.0 WP02.1（只读 D1 标签；不启动游戏、不训练、不采集）
- 产物：`touhou_rl/v410/teacher_audit.py`、`tests/test_v410_teacher_audit.py`、本文件、`audit.json`
- 状态：**审计执行完成**（audit / pytest / compileall 全部 exit 0）。审计结论是“D1 标签不能证明存活”，不是“教师差”，也不是 G410-1 通过或失败。

---

## 0 一句话结论

34,013 条真实 D1 决策标签里，`verified_mask=True` 出现 34,013 次，但 D1 行里**没有** `died` / `frames_advanced` / `trajectory_matches` / `verification_horizon`（14 个探测字段全部 `absent`），因此**无法从现有标签复核任何一个计划的存活**；同时有 **20 条 `bomb=1` 且 `frame_event_flags` 不含 PLAYER_HIT 位** 的决策，按 `decide.py` 的唯一合法代码路径只能由“所有 verified 计划都被引擎判死”触发 —— 这 20 行的 `verified_mask` 仍是 True、`chosen_plan.verified` 仍是 True、`chosen_plan.death_frame` 仍是 null，与“存活”行在标签上**完全不可区分**。

---

## 1 范围与不变量

- 只读：`.tmp_v400_d1/d1/shards/<shard>/labels.jsonl` 与 `<shard>/shard.json`；未写入、未移动、未删除任何 D1 文件。
- 只读代码：`touhou_rl/v400/teacher/{decide,verify,search,obs}.py`、`touhou_rl/v400/teacher/config.py`、`touhou_rl/v400/labels/events.py`（静态文本读取，不含 import 执行）；未修改 `touhou_rl/v400`、`v380`、`v390`、`tools/rl/v400_*`、`tests/test_v400_*`、`third_party/`。
- 未启动游戏、未训练、未采集、未 commit/push。
- 抽样：10 个分片，每片 `--per-shard 5000`；10 个分片实际都**少于** 5000 行，所以**全部为全量读取**（见 §3）。

---

## 2 命令与退出码（全部实际执行）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
```

### 2.1 pytest（绕过 `.pytest_cache`：`-p no:cacheprovider` + 独立 basetemp）

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp02_pytest -q tests/test_v410_teacher_audit.py
```

```
...............................                                          [100%]
31 passed in 0.15s
PYTEST_EXIT=0
```

### 2.2 编译校验

```powershell
python -m compileall -q touhou_rl/v410
COMPILEALL_EXIT=0
```

> 退出码 0 只代表语法通过，不等于测试通过；测试统计见 2.1。

### 2.3 真实审计（10 个分片，显式目录，`--per-shard 5000`）

```powershell
python -m touhou_rl.v410.teacher_audit --d1-shards `
  .tmp_v400_d1/d1/shards/d1-s0d1-n3371-seed1-shard `
  .tmp_v400_d1/d1/shards/d1-s0d1-n3371-seed10-shard `
  .tmp_v400_d1/d1/shards/d1-s1d1-n4000-seed10-shard `
  .tmp_v400_d1/d1/shards/d1-s1d1-n4000-seed11-shard `
  .tmp_v400_d1/d1/shards/d1-s3d1-n3833-seed10-shard `
  .tmp_v400_d1/d1/shards/d1-s3d1-n3833-seed11-shard `
  .tmp_v400_d1/d1/shards/d1-s6d1-n3836-seed10-shard `
  .tmp_v400_d1/d1/shards/d1-s6d1-n3836-seed11-shard `
  .tmp_v400_d1/d1/shards/d1-s7d1-n3218-seed1-shard `
  .tmp_v400_d1/d1/shards/d1-s7d1-n3218-seed10-shard `
  --per-shard 5000 --out Document/Update/Alpha/v4.1.0/evidence/WP02/audit.json
```

stdout：

```
teacher_audit: rows=34013 shards=10 stages=[0, 1, 3, 6, 7] verified_mask_true_candidates=34013 verified_decisions=34013 chosen_plan_death_frame_nonnull=0 absent_fields=died,frames_advanced,trajectory_matches,verification_horizon,horizon,verify_k,verified_plans,verified,query_status,survival_status,events,death_frame,timing_ms,extra
teacher_audit: wrote Document\Update\Alpha\v4.1.0\evidence\WP02\audit.json
AUDIT_EXIT=0
```

### 2.4 fail-closed 演示（同样实际执行，均为 exit 3）

```powershell
# 目录里没有 labels.jsonl
python -m touhou_rl.v410.teacher_audit --d1-shards .tmp_v400_d1/d1/shards/d1-s0d1-n3371-seed1-shard/obs --per-shard 10 --min-shards 1
# -> teacher_audit: FAILED (fail-closed): ...: no shard directories under this root
# FC1_EXIT=3

# 分片数少于 --min-shards
python -m touhou_rl.v410.teacher_audit --d1-shards .tmp_v400_d1/d1/shards/d1-s0d1-n3371-seed1-shard --per-shard 10 --min-shards 10
# -> teacher_audit: FAILED (fail-closed): only 1 shard(s) selected, at least 10 are required for the WP02.1 sample
# FC2_EXIT=3
```

坏 JSON / 空行 / 截断末行 / 缺必需字段 / 类型错误 / `scores` 非 18 长度 / `chosen_plan` 缺 `first_action` 的 fail-closed 行为由 pytest 覆盖（`test_bad_json_line_fails_closed_with_line_number`、`test_truncated_last_line_fails_closed`、`test_blank_line_fails_closed`、`test_required_field_violations_fail_closed` 的 9 个参数化用例），异常信息带**文件与 1-based 行号**。

---

## 3 抽样与语料（10 分片，全部全量读取）

| 分片 | stage(shard.json) | rows | rows==declared | capped | stop_reason | deaths_observed | checkpoint equal/mismatch |
|---|---|---|---|---|---|---|---|
| d1-s0d1-n3371-seed1-shard | 0 | 3370 | True | False | stage_changed | 0 | 6740 / 0 |
| d1-s0d1-n3371-seed10-shard | 0 | 3322 | True | False | stage_changed | 0 | 6644 / 0 |
| d1-s1d1-n4000-seed10-shard | 1 | 4000 | True | False | max_frames | 0 | 8000 / 0 |
| d1-s1d1-n4000-seed11-shard | 1 | 4000 | True | False | max_frames | 0 | 8000 / 0 |
| d1-s3d1-n3833-seed10-shard | 3 | 3833 | True | False | max_frames | 0 | 7666 / 0 |
| d1-s3d1-n3833-seed11-shard | 3 | 3833 | True | False | max_frames | 0 | 7666 / 0 |
| d1-s6d1-n3836-seed10-shard | 6 | 3572 | True | False | death | 1 | 7144 / 0 |
| d1-s6d1-n3836-seed11-shard | 6 | 1647 | True | False | death | 1 | 3294 / 0 |
| d1-s7d1-n3218-seed1-shard | 7 | 3218 | True | False | death | 1 | 6436 / 0 |
| d1-s7d1-n3218-seed10-shard | 7 | 3218 | True | False | max_frames | 0 | 6436 / 0 |
| **合计** | 0/1/3/6/7 | **34,013** | 10/10 | 0 | — | 3 | 68,026 / 0 |

- 10 个分片的 `shard.json` 均 `complete=true`，`labels.rows` 与实际行数一致，`hash_alignment.checkpoints_mismatched=0`。
- 附注：`.tmp_v400_d1/d1/shards` 下 137 个非隐藏分片全部可解析（逐行 JSON 校验通过、行数等于 `shard.json` 声明）；本审计只取其中 10 个，**不代表 137 片全量**。
- 每片 `--per-shard 5000` 未截断任何一文件（最大 4000 行），所以 `sample.capped=false`、`has_more_rows=false`。

---

## 4 关键数字（`audit.json` 实测）

### 4.1 已验证 / 存活

| 指标 | 值 |
|---|---|
| rows | 34,013 |
| decisions（`kind=="decision"`） | 34,013 |
| `verified_mask_true_candidates` | **34,013**（每次决策恰好 1 个） |
| `verified_decisions` | 34,013（100%） |
| `verified_count` 分布 | `{"1": 34013}`，zero=0，min=max=1，与 mask 不一致 0 |
| `chosen_plan` | null=0；`verified=true` 34,013；`verified=false` 0 |
| `chosen_plan.death_frame` 非空 | **0**；null 34,013 |
| `chosen_plan.first_action` 与 `verified_mask` 不一致 | 0 |
| `bomb` | 0 → 33,993；1 → **20** |
| `bomb=1` 且无 PLAYER_HIT 位（`frame_event_flags & 0x2 == 0`） | **20**（有 PLAYER_HIT：0；flags 不可用：0） |
| `bomb=1` 行中带 verified mask 的 | 20 / 20 |
| `bomb=1` 行中 `chosen_plan.death_frame` 非空 | 0 / 20 |

### 4.2 缺失字段（**不做通过处理**）

`missing_fields.absent`（14 个，全行皆为 absent）：

```
died, frames_advanced, trajectory_matches, verification_horizon, horizon, verify_k,
verified_plans, verified, query_status, survival_status, events, death_frame,
timing_ms, extra
```

`missing_fields.chosen_plan_absent`（4 个）：`died, frames_advanced, trajectory_matches, verification_horizon`。

实测 D1 行的 26 个顶层字段：

```
bomb, bomb_death_available, channel, chosen_plan, decision, decision_frame, decision_frames,
difficulty, frame, frame_event_flags, frames_per_decision, hash_post, hash_pre, kind, margin,
obs, pixel_channel, schema, scores, shoot_release, stage, start_frame, start_mode, step_frames,
verified_count, verified_mask
```

`chosen_plan` 的 19 个字段即 `search.Plan.as_dict()`，没有 `died`。

### 4.3 margin

| 指标 | 值 |
|---|---|
| 饱和在 `MARGIN_CAP=16.0` 的行 | **29,946 / 34,013 = 88.04%** |
| `margin < 4` | 32 |
| `margin < 8` | 319 |
| min / max | **0.203247** / 16.0 |
| 不同取值数 | 2,354 |
| null / NaN | 0 / 0 |

### 4.4 scores（top1-top2 间隔、并列）

| 指标 | 值 |
|---|---|
| `unverified_discount`（代码常量） | 5.0 |
| top1-top2 gap：min / p10 / median / mean / p90 / max | **5.0 / 5.0 / 5.0 / 5.036784 / 5.058925 / 17.028595** |
| gap < 5.0 的行（与折扣矛盾） | **0** |
| 精确并列（gap == 0） | **0** |
| 近似并列（gap ≤ 1e-6） | 0 |
| top1 落在 verified 候选上 | **34,013 / 34,013 = 100%** |
| 非有限分数（NaN/±Inf） | 0 |

### 4.5 frames（`frames_per_decision` / `decision_frames` 一致性）

| 指标 | 值 |
|---|---|
| `frames_per_decision` 分布 | `{"4": 34013}` |
| `decision_frames` 分布 | `{"2": 34013}` |
| 观测对 | `{"4/2": 34013}` |
| 朴素相等判断 | equal=0，**unequal=34,013** |
| 推导出的 replan 因子 | `{"2": 34013}` |
| 关系 `frames_per_decision == 2 * decision_frames` | **holds 34,013 / violated 0** |
| `decision_frame` 步长 | 检查 34,003，全部 `+4`，mismatch 0，非单调 0 |
| `decision` id 步长 | 检查 34,003，mismatch 0 |
| `obs.label_frames` 长度 == `decision_frames` | 检查 34,013，mismatch 0 |
| `frame == decision_frame` | 34,013 / 34,013 |

### 4.6 per-stage（按 `shard.json` 的 stage 归组，**不是**按行内字段）

| stage | shards | rows | verified_decisions | bomb=1（无 PLAYER_HIT） | margin 饱和 | death_frame 非空 |
|---|---|---|---|---|---|---|
| 0 | 2 | 6,692 | 6,692 | 0 | 6,435 | 0 |
| 1 | 2 | 8,000 | 8,000 | 0 | 7,417 | 0 |
| 3 | 2 | 7,666 | 7,666 | 8 | 6,113 | 0 |
| 6 | 2 | 5,219 | 5,219 | 7 | 4,224 | 0 |
| 7 | 2 | 6,436 | 6,436 | 5 | 5,757 | 0 |

### 4.7 行完整性计数（`problems`）

```json
{"row_difficulty_disagrees_with_shard": 10, "row_stage_disagrees_with_shard": 8}
```

- `verified_count` 与 mask 不一致：0；`chosen_plan` 与 mask 不一致：0；非有限分数：0；frames 关系违反：0；label_frames 长度不一致：0。
- 但行内 `stage` 恒为 0、`difficulty` 恒为 0：stage 1/3/6/7 的 8 个分片全部与 `shard.json` 的 stage 不符（见 §6.9）。

---

## 5 代码语义引用（静态、可复现）

`code_semantics_report()` 以**整块语句匹配**（逐行去缩进后精确比较）定位，不 import 教师代码、不写任何文件；`audit.json.code_semantics` 记录文件 sha256（`decide.py 9d93bbf9…`、`verify.py a4583bcf…`、`search.py 49b4504c…`、`obs.py c229c5b2…`）和每块行号。

| id | 位置 | 内容 | 证明 |
|---|---|---|---|
| code.01 | `decide.py:295-299` | `verified_mask = np.zeros(N_CAND…)` / `for verified_plan in verified:` / `verified_mask[cand] = True` | 所有 verified 计划**无条件**置 True；无 `died` 分支（全文无 `verified_mask[...] = False`） |
| code.02 | `decide.py:211` | `survivors = [item for item in verified if not item.died]` | 存活判定只用 `VerifiedPlan.died`，而该字段不进行 |
| code.03 | `decide.py:221-222` | `fallback = max(verified…)` / `chosen = … _replace(fallback.plan, verified=True)` | 全部 verified 计划都死时，`chosen.verified` 仍为 True |
| code.04 | `decide.py:391` | `death_frame=plan.death_frame,` | `chosen_plan.death_frame` 是**几何模型**的预测，不是引擎帧 |
| code.05 | `decide.py:300-302` | `scores[cand] -= float(config.weights.unverified_discount)` | 行内 `scores` 是**折扣后**分数 |
| code.06 | `verify.py:60-63` | `"died"` / `"death_frame"` / `"margin"` / `"frames_advanced"` | 引擎判定只存在于 `VerifiedPlan.as_dict()` |
| code.07 | `verify.py:64-65` | `"max_position_error"` / `"trajectory_matches"` | 同上 |
| code.08 | `search.py:70-74` | `Plan.as_dict()` 的 `first_action/duration/slots/margin/death_frame` | 写进 `chosen_plan` 的对象没有 `died` |
| code.09 | `obs.py:1055-1056` | `"margin": … float(label.margin)` / `"chosen_plan": … chosen.as_dict()` | D1 行只写 `chosen_plan`，`label.verified` 被丢弃 |
| code.10 | `obs.py:1049-1050` | `"frames_per_decision": int(self.meta…)` / `"decision_frames": obs_spec.DECISION_FRAMES` | 两个节奏字段来源不同 |
| code.11 | `decide.py:234` | `all_verified_die = bool(verified) and not survivors` | bomb 规则 1 的前提是“每个 verified 计划都死” |
| code.12 | `decide.py:257-259` | `if config.bomb_active and all_verified_die and bombs_available > 0:` … `bomb = 1` | bomb=1 分支 A：规则 1（不看 PLAYER_HIT） |
| code.13 | `decide.py:282` | `elif player_hit:` | bomb=1 分支 B：death-bomb，必须 `player_hit` |
| code.14 | `decide.py:278` | `player_hit = bool(events_frame.get("player_hit"))` | `player_hit` 与行内 `frame_event_flags` 同源（见下） |

`player_hit` 的定义（`touhou_rl/v400/gym/protocol.py` `EventFrame.player_hit`）：

```python
return bool(self.frame_event_flags & REPLAY_FRAME_EVENT_PLAYER_HIT)
```

即 **`row["frame_event_flags"] & 0x2`**。该位值由 `labels/events.py:47` 的 `REPLAY_FRAME_EVENT_PLAYER_HIT = 0x0002` 固定，`code_semantics.constants` 会比对。

镜像常量校验（全部 `holds=true`）：`DECISION_FRAMES=2`（config.py:16）、`N_CAND=18`（config.py:19）、`Weights.unverified_discount=5.0`（config.py:89）、`replan_decisions=2`（config.py:199）、`MARGIN_CAP=16.0`（config.py:26）、`REPLAY_FRAME_EVENT_PLAYER_HIT=0x2`（events.py:47）。

`code_semantics.dropped_verified_fields`（`VerifiedPlan.as_dict()` 有、D1 行没有、`Plan.as_dict()` 也没有）：

```
died, events, frames_advanced, max_position_error, trajectory_matches
```

---

## 6 发现（11 条，`audit.json.findings`）

### 6.1 【blocking】`verified_mask=True` 不等于 survived，且有 20 例引擎死亡的正面证据

- `decide.py:296-299` 对所有 `verify_plans` 返回的计划无条件置 `verified_mask[cand]=True`，不看 `died`；存活只由 `decide.py:211` 的 `survivors = [i for i in verified if not i.died]` 决定；全部死亡时 `decide.py:222` 仍然 `_replace(..., verified=True)`。
- 行内没有 `died`（§4.2），所以“verified 且死亡”的行与“verified 且存活”的行在字段上**无法区分**。
- **正面证据**：20 条 `bomb=1` 行的 `frame_event_flags` 为 `{0:6, 1024:4, 1088:1, 3072:1, 9216:5, 11264:3}`，**无一含 0x2（PLAYER_HIT）**。`decide.py` 只有两个 `bomb = 1` 分支：规则 1（`decide.py:257`，前提 `all_verified_die`，`decide.py:234`）与 death-bomb（`decide.py:282`，前提 `player_hit`）。无 PLAYER_HIT 位 ⇒ 只可能是规则 1 ⇒ **这些决策的 verified 计划被引擎判死**（本语料里每次只有 1 个 verified 计划，故等价于“该计划死亡”）。
- 这 20 行的 `verified_mask` 仍为 True、`chosen_plan.verified` 仍为 True、`chosen_plan.death_frame` 仍为 **null**。
- 分布：stage 3 = 8，stage 6 = 7，stage 7 = 5，stage 0/1 = 0。

### 6.2 【blocking】D1 行不能复核存活

`ObsShardWriter._label_row`（`obs.py:1040-1060`）只写 `chosen_plan`（`search.Plan`），`label.verified`（`TeacherLabel.verified`）从未落盘。`died`、`frames_advanced`、`trajectory_matches`、`verification_horizon` 在 34,013 行中 **present_rows=0**。README §WP02 第 2 条要求的“对所有 verified 计划核对 first_action、duration、died、frames_advanced、验证视界、trajectory_matches 与掩码”在当前 D1 上**不可执行**；缺字段记为 absent，**不记为通过**。

### 6.3 【blocking】`chosen_plan.death_frame` 是几何模型预测，且本采样中非空为 0

- 该字段是 `Plan.death_frame`（`search.py:74`），由 `_replace` 原样复制（`decide.py:391`），**不是**引擎的 `VerifiedPlan.death_frame`（`verify.py:61`）。所以“非空”只意味着“几何模型预测会死”，不能当作引擎死亡证据；反之“null”也不代表引擎判定存活。
- 本采样 `verified=true` 且 `death_frame` 非空 = **0 / 34,013**；因此“已验证但死亡”的实例**不是**由该字段体现，而是由 §6.1 的 20 条 `bomb=1` 行体现（这些行 `death_frame` 全为 null，即几何模型判活、引擎判死，属模型/引擎不一致方向之一）。
- 若未来某批标签出现 `death_frame` 非空，那也只是“几何模型预测死亡”，仍需 `died` 才能定性。

### 6.4 【warning】top1-top2 间隔被 `unverified_discount` 主导

`scores` 是折扣后分数（`decide.py:300-302`，discount=5.0）：verified 候选保留原始分，其余 17 个各减 5.0。`verify_k=1` 时每次只有 1 个 verified，故 **argmax(scores) 恒为 verified 候选（34,013/34,013 = 100%）**，gap 的最小值、中位数都**精确等于 5.0**（最大 17.03，来自真正较大的几何差）。因此：

- `gap` 不能当“教师置信度”；`gap - 5.0` 才是几何差的上界近似。
- “top1 = chosen action”是同义反复，不能当“教师偏好”的证据（正对 EXP-P3-004 的“只是学到哪个动作被挑去验证”伪目标风险）。
- 审计把 `gap < 5.0` 作为可证伪断言：实测 0 例，与代码一致。

### 6.5 【warning】每个决策只有 1 个候选被验证

`verified_count` 恒为 1（`mode=fast`、`verify_k=1`）：18 个首动作里 **17 个永远没有引擎证据**。`verified_mask=False` 只能是“未验证”，不能当“不安全”负例（README §WP03 特别规则）。

### 6.6 【warning】`margin` 在 88% 的行上饱和

`MARGIN_CAP=16.0`（`config.py:26`，`margin_term = min(margin, margin_cap)`）：29,946 行（88.04%）`margin == 16.0`，只表示“净空 ≥16px”。可用的低余量样本：`<4` 仅 32 行、`<8` 仅 319 行。WP02 第 4 条要的“低余量诊断集”只能从这些行里取，且量很小。

### 6.7 【info】`frames_per_decision=4` 与 `decision_frames=2` 不是同一个量

二者是乘积关系：`frames_per_decision = replan_decisions(2) × decision_frames(2) = 4`（`obs.py:1049-1050`、`config.py:199-200`）。34,013 行里“朴素相等”全部为假（unequal=34,013），但乘积关系 **100% 成立**（violated=0），`decision_frame` 步长 100% 为 +4，`obs.label_frames` 长度 100% 为 2。**把 4≠2 当成数据缺陷是误报**；可判的一致性项是“乘积关系 / 步长 / label_frames 长度”，本审计三项全通过。

### 6.8 【warning】同 stage 不同 seed 的分片共享标签内容前缀

标签内容指纹（排除 `hash_pre/hash_post/obs.shard/start_frame` 等来源字段）逐行比较：

| stage | 对比行数 | 前缀完全相同行数 |
|---|---|---|
| 0 | 3,322 | **638** |
| 1 | 4,000 | **258** |
| 3 | 3,833 | **1,298** |
| 6 | 1,647 | **316** |
| 7 | 3,218 | **323** |

即：同 stage 的两个 seed 在前 258–1,298 个决策（约 1,000–5,200 帧）内**标签内容逐字节相同**，之后才分叉；整个采样窗口相同的配对 0 个，每 stage 的 distinct label stream 仍为 2。含义：`hash_pre` 不同（引擎状态不同）但标签不同步变化，**分片数 ≠ 独立标签轨迹数**。WP02 第 4 条“至少 10 条独立训练轨迹”、WP03 的“同一轨迹切片不算独立 episode”都必须按标签内容去重，而不是按分片目录计数。

### 6.9 【warning】行内 `stage` / `difficulty` 恒为 0

34,013 行全部 `stage=0`、`difficulty=0`，而 `shard.json` 的 stage 为 1/3/6/7、difficulty=1（10 个分片全部 `row_difficulty_disagrees_with_shard`，8 个 `row_stage_disagrees_with_shard`）。因此：

- 任何按 `labels.jsonl` 行字段做的分面/难度分层都会塌成单一桶；本审计改用 `shard.json` 的 stage 归组（`d1_shards.stage_coverage_source` 已写明）。
- G410-1 的“逐面”结算与 WP08 课程分面不能依赖行内 `stage`。

### 6.10 【info】行完整性计数

`problems` 只有 §4.7 的两项（行内 stage/difficulty 与 shard 不符）；`verified_count` 与 mask、`chosen_plan` 与 mask、frames 关系、`label_frames` 长度、非有限分数均无异常。

### 6.11 【info】`chosen_plan` 与 `scores` 的其余一致性

`chosen_plan.first_action` 落在 `verified_mask=True` 的索引上：34,013/34,013；`chosen_plan.verified=false` 的行：0；`chosen_plan=null`：0。即 D1 的 fast 教师每次都给出且只给出一个“已验证 chosen”。

---

## 7 限制（明确不声称）

1. **不是 G410-1 教师门结论**：本审计只读标签，没有跑教师、没有确认种子集、没有通关率/Wilson 区间、没有逐面 PSN。
2. **不是能力结论**：没有学生、没有策略评估、没有训练、没有一次 rollout。
3. **不是全量 D1 审计**：10/137 分片，且每片最多 5,000 行（实际全部 ≤4,000 行，等于全量读取该 10 片）。不代表其余 127 片。
4. **缺失字段不构成通过**：`died`/`frames_advanced`/`trajectory_matches`/`verification_horizon` 的 absent 是**阻断**，不是满足。
5. **§6.1 的“20 例引擎死亡”是代码路径推断**：依据是 `decide.py` 只有两个 `bomb=1` 分支 + 行内 `frame_event_flags` 无 PLAYER_HIT 位 + `bomb_active/bomb_death` 默认 True。它**不是**逐计划的引擎判定记录；逐计划判定只能在 `verify.py` 里产生，未落盘。
6. **`chosen_plan.death_frame` 不被解释为引擎死亡**（见 §6.3）。
7. **标签前缀相同不证明引擎状态相同**：`hash_pre` 每行都不同，指纹故意排除来源字段。
8. **未修改任何 D1 文件**，也未修改 `touhou_rl/v400|v380|v390`、`tools/rl/v400_*`、`tests/test_v400_*`、`third_party/`、`state.json`/`journal.jsonl`/`budget.jsonl`/`assets.json`；未 commit/push。

---

## 8 对 WP02.2 / G410-1 的直接影响

**WP02.2（教师质量与可学习性）必须先做标签再生成，不能先做统计：**

1. **必须把 `TeacherLabel.verified` 落盘**（`obs.py::_label_row` 增加 `"verified": [item.as_dict() …]` 或至少 `died/frames_advanced/trajectory_matches/max_position_error/verification_horizon`）。`verify.py` 已经算好这些字段，只在写出时被丢弃（`code_semantics.dropped_verified_fields`）；这是纯写出改动，不需要重跑引擎就能验证写出格式（但历史 D1 无法补，重发标签需要跑教师 → 不在 WP02.1 允许范围）。
2. **新增 `query_status` / `survival_status`**（README §WP02 第 3 条）：`survival_status ∈ {verified_alive, verified_dead, unknown}`，来源是 `VerifiedPlan.died`，**不能**用 `verified_mask` 或 `death_frame` 冒充。
3. **top1-top2 / 熵 / 并列统计必须用折扣前的几何分**（`Plan.score` / `geo_score`），否则 5.0 的折扣会污染分布（§6.4）。D1 现在只存折扣后 `scores`，`chosen_plan.geo_score` 仅对 1 个候选可用。
4. **低余量诊断集**从 `margin<8` 的 319 行（或 `<4` 的 32 行）里取，并注意 88% 的行已饱和（§6.6）。
5. **“10 条独立轨迹”必须按标签流去重**（§6.8）：同 stage 两个 seed 的前 258–1,298 个决策标签内容逐字节相同，所以“10 个分片”不能直接当成 10 条互不重叠的独立轨迹；WP02 第 4 条的 1,000 个分层状态必须按标签内容与轨迹祖先去重后再计数。
6. **分面统计改用 `shard.json`**（§6.9），并建议在标签里修正 `stage`/`difficulty` 来源，否则训练侧的 stage 条件头/课程会全落在一个桶。

**G410-1 教师腿：**

- 教师腿要求“无执行与验证不一致”。当前 D1 **无法提供**该证据（没有逐计划 `died`/`trajectory_matches`/视界），所以 G410-1 教师腿**不能**用 D1 标签结账；必须在带 `verified` 明细的新采集上重算。
- 已有的**可引用**证据只有：`shard.json.hash_alignment` 全部 0 mismatch（68,026 个 checkpoint 全等，10 片）——这是“环境复现影子轨迹”的对齐证据，**不是**教师通关率证据。
- §6.1 的 20 条 `bomb=1` 行说明当前 fast 教师**至少 20 次**在“唯一 verified 计划被引擎判死”的情况下仍把它标成 verified；这 20 次是否最终导致通关/失败，需要引擎侧结果（`deaths_observed`、`stop_reason`），本审计不做推断。

---

## 9 复现（完整）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'

# 1) 单测（31 项，含 fail-closed / 并列 / 缺失字段 / frames 关系 / bomb 推断 / 只读校验）
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp02_pytest -q tests/test_v410_teacher_audit.py

# 2) 语法
python -m compileall -q touhou_rl/v410

# 3) 真实审计（10 分片；命令见 §2.3）
python -m touhou_rl.v410.teacher_audit --d1-shards <10 个分片目录> --per-shard 5000 `
  --out Document/Update/Alpha/v4.1.0/evidence/WP02/audit.json

# 4) 根目录展开（确定性：每 stage 取字典序前 2 个）
python -m touhou_rl.v410.teacher_audit --d1-shards .tmp_v400_d1/d1/shards `
  --stages 0,1,3,6,7 --per-stage 2 --per-shard 5000 --min-shards 10 --out <path>
```

`audit.json` 里 `schema="v410-teacher-audit/1"`，含 `generated_at`、`tool.command`、`shards`、`rows`、`per_stage`、`verified_mask_true_candidates`、`verified_decisions`、`chosen_plan_death_frame_nonnull`、`missing_fields`、`frames`、`scores`、`margin`、`bomb`、`code_semantics`、`findings`、`not_claimed`、`label_stream_diversity`。

---

## 10 变更文件（严格白名单）

| 文件 | 说明 |
|---|---|
| `touhou_rl/v410/teacher_audit.py` | 新增：`audit_label_rows` / `audit_shard` / `audit_corpus` / `code_semantics_report` / `build_findings` / CLI |
| `tests/test_v410_teacher_audit.py` | 新增：31 个测试 |
| `Document/Update/Alpha/v4.1.0/evidence/WP02/AUDIT.md` | 本文件 |
| `Document/Update/Alpha/v4.1.0/evidence/WP02/audit.json` | 真实审计证据（124,837 字节，sha256 `faa697ca558ece5cd62916c9d6c6ce9b967c9e2ee33fab78fef12af327899241`） |

未修改任何既有文件；未删除、未覆盖旧文件；未 commit/push。
