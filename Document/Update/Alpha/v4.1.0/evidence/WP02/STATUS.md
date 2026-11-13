# WP02 状态（教师审计 / fast 20 seeds）

- 时间：2026-11-13T02:00:00+08:00
- 结论：**WP02.1/02.2/02.3/02.4/02.5 全部完成**；fast20 = 20/20、calibrated alive 3,226、真实 `verified_dead` 7 条已取得。**WP02 仍为 in_progress**，因为 G410-1 正式 100 种子 Stage-1 确认与 PSN 8×100 未做；G0.1 为 91/108，剩余 17 个批量重试中。

## 1. WP02.1 教师标签审计（完成）

证据：[`audit.json`](audit.json)、[`AUDIT.md`](AUDIT.md)；`pytest tests/test_v410_teacher_audit.py` → **31 passed**；compileall exit 0。

关键数字：

- 10 个 D1 分片、**34,013 行**；`verified_mask_true_candidates = 34,013`；`verified_decisions = 34,013 (100%)`；**verified_count 恒为 1**（fast/verify_k=1，17/18 首动作无引擎证据）。
- `chosen_plan.verified=true` 34,013；`chosen_plan.death_frame` 非空 **0**；但 `death_frame` 是几何预测，不是引擎帧。
- 34,013 行全部 **缺少** `died`、`frames_advanced`、`trajectory_matches`、`verification_horizon`（chosen_plan 内同样缺）；**D1 无法复核 survived**。
- 20 条 `bomb=1` 且无 `PLAYER_HIT` 位：按 `decide.py` 唯一路径只能由“所有 verified 计划被判死”触发，但这些行 `verified_mask`/`verified` 仍 True、`death_frame` null；与存活行不可区分（代码路径推断，已标 `not_claimed`）。
- margin 饱和 16.0 占 88.04%；top1-top2 gap 中位数 5.0（折扣主导）；行内 `stage/difficulty` 恒 0；同 stage 两 seed 标签前缀相同 258–1,298 行（独立轨迹去重风险）。
- 代码语义：`decide.py:295-299` 对所有 verified 计划无条件 `verified_mask=True`；存活只由 211 行 `survivors=[... not i.died]` 决定；222 行全死时仍 `_replace(..., verified=True)`。

结论：**历史 D1 标签不能作为“已验证存活”或 G410-1 教师腿证据**；WP02.3 必须改写出格式（`verified: [VerifiedPlan.as_dict()]` + `query_status`/`survival_status`），再在新切片重跑教师采集。

## 2.1 WP02.4 模型一致性校准（完成）

证据：[`CALIBRATION.md`](CALIBRATION.md)、[`labels_calibrated_summary.json`](labels_calibrated_summary.json)、[`labels_calibrated_sample.jsonl`](labels_calibrated_sample.jsonl)。`pytest tests/test_v410_teacher_calibration.py` → **59 passed**；compileall exit 0。

- 根因：`verify.py::POSITION_TOLERANCE=1e-3` 是精确模型检查；移动动作的模型-引擎位置误差中位 3.414 px，99.23% ≤8 px，只有 idle 误差为 0。
- 预注册规则：取最小 grid 阈值使 died=false 计划覆盖 ≥99%，上限 16 px；结果 **8.0 px**（covered 3,226/3,251 = 99.231%；mismatch 25）。
- 校准口径：`died=false` + `frames_advanced` 满视界 + `max_position_error ≤ 8` → `verified_alive`；误差 >8 或执行不完整 → `unknown`；`died=true` → `verified_dead`。
- 结果：**alive 3,226 / dead 0 / unknown 25（model_mismatch 25）/ moving_alive 1,704 / idle_alive 1,522**；旧 WP02.3 保守口径 alive 795 / unknown 2,456 仍保留、未改写。
- 限制：当前语料 0 条 `verified_dead`；8 px 是模型一致性阈值，不是生存真值；engine `died` 才是生存真值。WP03 只把 `verified_alive` 当正样本，`unknown` 不作负样本。

证据：[`LABELS.md`](LABELS.md)、[`labels_summary.json`](labels_summary.json)、[`labels_seed1000_sample.jsonl`](labels_seed1000_sample.jsonl)。`pytest tests/test_v410_teacher_labels.py` → **60 passed**；compileall exit 0。

## 2. WP02.3 新标签写出契约（完成）

- 输入：`teacher1000.jsonl`（seed 1000，fast，3,251 decisions）真实教师行。
- 输出保留 `verified` 完整明细（`died`、`frames_advanced`、`trajectory_matches`、`max_position_error`、`death_frame`）以及 `query_status`（verified/unknown）与 `survival_status`（verified_alive/verified_dead/unknown）。
- 结果：**alive 795、dead 0、unknown 2,456**；`horizon_missing=3,251`（教师行没有 `verification_horizon`，所以 795 个 `trajectory_matches=true` 才判 alive，其余保守 unknown）；query 覆盖 verified 3,251 / unknown 55,267（每决策 1 个 verified，17 个 unknown）。
- 这证明 D1 丢字段的问题可以在 v410 层修复，且不需要修改 v400 教师；WP03 的标签生产应使用这个契约。
- **关键限制（必须先修）**：795 条 `verified_alive` **全部是 `first_action=0`（idle）、duration=1**；移动动作的 `trajectory_matches=false`（模型-引擎位置偏差 1.414–139.584 px，中位 4.0），因此当前 survival 标签只覆盖 idle，**没有移动安全监督的区分度**。在 `trajectory_matches` 修好前，WP03 不能把 `verified_alive` 当作可用的移动动作正样本。

## 3. WP02.2 fast 20 seeds（完成，fast gate passed）

- **20/20 stage_clear、0 死、frames 13,004–13,676（mean 13,282.4、median 13,292）、wall mean 473.3 s、exit 0**。
- fast_gate：`clears=20 ≥18`、`completed=20`、`execution_ok=true`、`verification_ok=true`，`passed=true`，recommendation=`fast`。
- 与 G1 v2h 对照：v2h 20/20、frames mean 13,532、wall mean 808.8 s；fast 平均少 249.6 帧（-1.84%），wall 更快。
- 注意：fast 的 `verify_k=1` 只给每决策 1 个候选引擎验证；它的**通过只说明过面/吞吐**，不能修复 WP02.1 的标签稀疏与 `verified_mask≠survived` 问题。

## 3.1 WP02.5 真实 verified_dead 样本（完成）

证据：[`DEATH.md`](DEATH.md)、[`dead_sample_summary.json`](dead_sample_summary.json)、[`dead_sample.jsonl`](dead_sample.jsonl)。`pytest tests/test_v410_teacher_death_probe.py` → **36 passed**；compileall exit 0。

- Stage 6B（引擎 stage 7）/ normal / seed 1 / fast：**stop_reason=death，12,872 帧，3,218 decisions，deaths=1**；engine SHA `1462829b…`。
- 3,218 个 verified plans：`died_true=7`、`died_false=3,211`；校准后 **verified_dead=7**、verified_alive=3,171、unknown=40（model_mismatch 40）。
- 7 条 dead 计划 first_action = [0,2,3,7,11,12,15]；样本 7 行全部 `died=true`、`survival_status=verified_dead`。
- 这补上了 WP02.4 的 0 死亡缺口；`verified_dead` 语义在真实引擎上出现。限制：单 run/单 stage/seed，不是 G410-1，也不证明死亡分布。

## 4. G0.1 外部阻塞（已解除，批量重试中）

- `lunatic-9-a` attempt 2 在桌面解锁后成功：102,204 行、`foregroundOk=true`、耗时 1,262 s；G0.1 从 90/108 变为 **91/108 identical、0 mismatch、17 failed**。
- 剩余 17 个 fixture 已启动串行批量重试（后台 job `pwsh-1510`，`--max-attempts 5`）；预计数小时，取决于是否有 fixture 再次失败。
- 仍需保持桌面解锁；若锁屏，前台导航会再次 `WinError 5`，失败会保留证据。

## 5. 下一步

1. 收 G0.1 剩余 17 个 fixture 的批量重试结果；已从 90/108 → 91/108，`lunatic-9-a` 在解锁后成功。
2. **WP02.5**：补真实 `verified_dead` 样本（当前语料 0 条），否则 `verified_dead` 语义只有合成单测。
3. **WP03**：按 fast 教师 + calibrated `teacher_labels` 生产新切片；只把 `verified_alive` 当正样本，`unknown` 不作负样本。
4. G410-1 的正式 100 种子/PSN 100 面仍单独执行，不拿 20 局替代。
