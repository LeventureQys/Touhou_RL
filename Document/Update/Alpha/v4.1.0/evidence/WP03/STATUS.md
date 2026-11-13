# WP03 状态（数据计划与标签覆盖）

- 时间：2026-11-13T03:55:00+08:00
- 结论：**D410-S1 通过（20 个纯 train seeds、66,143 decisions、obs true、duplicates 0、seed policy 全 train）；WP03.1c 完成。WP03 仍为 in_progress，因为 D410-MULTI（stage 2–7、2M 帧）与 DIAG/RECOVERY 尚未完成；G0.1 为 95/108。**

## 1. WP03.0 完成

证据：[`plan.json`](plan.json)、[`s1_label_audit.json`](s1_label_audit.json)、[`PLAN.md`](PLAN.md)；`pytest tests/test_v410_dataset_plan.py` → **35 passed**；compileall exit 0；`s1_label_audit.json` SHA256 `c5d2f92a…e542e6`。

- D410-S1 计划：stage 0 / normal / team 0 / seeds 1000–1019 / ≥50,000 unique decisions / 20 seeds / 无 synthetic padding / 必需 obs 平面。
- D410-MULTI 计划：8 面 / 2,000,000 帧 / 每面 ≥150,000 / 唯一轨迹与起点混合。
- 离线标签审计（20 个 fast 教师原始 JSONL）：
  - **20 runs、66,412 decisions、20 unique seeds、duplicate keys 0**；
  - 校准 survival：**alive 66,030 / dead 0 / unknown 382**；query 覆盖随每决策 1 verified + 17 unknown；
  - 仅标签覆盖已过 50k/20 seed；
  - **observations_present=false** → `against_s1.status="blocked_missing_observations"`、`passed=false`。
- 这是正确的 fail-closed：标签数量足够不等于数据集成立；WP03.1 必须采集真实 obs 平面（play96x112/hud/act_hist）并重新验证。

## 1a. WP03.1a D1 真实重标切片（部分完成）

证据：[`relabel_manifest.json`](relabel_manifest.json)、[`s1_partial_audit.json`](s1_partial_audit.json)、[`RELABEL.md`](RELABEL.md)；`pytest tests/test_v410_relabel_d1.py` → **30 passed**；compileall exit 0。

- D1 work 原始 labels 只保留 **15 个 shard**；与现有 obs 对齐后 **54,625 decisions**，`trimmed_decisions=7`（stage transition）。
- 校准 survival：**alive 54,341 / dead 6 / unknown 278**；`moving_alive=33,159`、`idle_alive=21,182`；旧保守口径 alive 7,270。
- 观测：`observations_present=true`，15 个 shard 的 play/hud/act_hist 行数与 decisions 对齐；15 个 `labels_v410.jsonl` 已写出（obs 引用不复制）。
- S1 partial：stage0 **8 shards / 26,625 decisions / 8 seeds（实际 seeds 1–8，不是计划 1000–1019）** → `shortfall`，差 **23,375 decisions / 12 seeds**。
- MULTI partial：仅 stage 0/1、218,500 帧，远低于 2M/8×150k → `shortfall`。
- 6 个 `verified_dead` 全部来自 `d1-s1d1-n4000-seed2-shard`；死亡分支统计意义有限。
- 对齐证据：按 decision+hash_pre+decision_frame 逐条校验；旧 labels verified_mask 与 work 行 54,625/54,625 一致；45 个 obs 平面 SHA 45/45 匹配。
- 结论：**真实 obs+calibrated labels 对齐链路已验证，但这不是完整 D410-S1/MULTI。**

## 1b. WP03.1b 新采集（完成，但 S1 被拒绝）

- 12 个 seeds（1000–1011）真实采集完成：**12/12 shards ok、39,011 decisions、obs-v3、work verified labels 全部保留**；smoke seed 1000 为 3,251 decisions / 13,004 frames。
- 与已有 8 train seeds（1–8）合并：**65,636 decisions / 20 seeds / obs true / duplicates 0**；表面达到 50k/20。
- **主代理拒绝 S1 passed**：实际 seeds = **1–8（train）+ 1000–1011（dev）**，train/dev 混用；`plan_seed_set.matches_plan=false`。最终 `s1_full_audit.json` 已改为 `passed=false`、`s1_accepted=false`、`status=rejected_seed_provenance`。详见 [`S1_REVIEW.md`](S1_REVIEW.md)、[`S1B.md`](S1B.md)。
- 这仍证明采集/重标机制可用（12 个新 shard 真实、obs SHA 匹配、work labels 保留），但 **D410-S1 数据集不成立**。

## 1c. WP03.1c 干净 train-seed S1（完成，S1 通过）

证据：[`S1C.md`](S1C.md)、[`s1c_full_audit.json`](s1c_full_audit.json)、[`s1c_run.json`](s1c_run.json)；`pytest tests/test_v410_s1c_audit.py` → **20 passed**；compileall exit 0。

- 12 个新 train seeds（9–20）真实采集完成；与已有 train seeds 1–8 合并：
  - **20 seeds = 1–20，全部为 train；dev overlap 0、confirm overlap 0**；
  - **66,143 decisions / 66,143 unique / duplicate keys 0 / synthetic false**；
  - **observations_present=true**，obs-v3 冻结 digest 全匹配；obs/work labels/calibrated labels 完整性 checks 全 true；
  - calibrated survival：**alive 65,818 / dead 0 / unknown 325**；min per-seed decisions 3,274；
  - `against_s1.passed=true`，`seed_policy` 所有检查 true，`review_response` 三条拒绝项全部关闭。
- 结论：**D410-S1 第一轮要求通过**（≥50,000 唯一决策点、20 独立 train seeds、每 2 帧可部署决策有标签、obs 真实落盘）。
- 限制：S1 语料 0 deaths；unknown 325 不能作负样本；不是 G410-1 正式教师 100 种子，也不是 D410-MULTI。

## 1d. WP03.2 D410-MULTI（未完成）

- 目标 2,000,000 真实提交帧、8 面各 ≥150,000；当前只有 S1 的 20 个 stage0 shard（约 265k 帧）与旧的 15 个 mixed shard；**MULTI 未通过**。
- 下一步需要 stage 2–7 真实采集与每面 ≥150k 校验。

## 5. 下一步

1. **WP03.2**：采集 D410-MULTI stage 2–7，目标 2M 帧、每面 150k；同样保留 obs+verified 明细。
2. G0.1 批量重试继续；G410-0 仍 in_progress，不能因 95/108 写 passed。
3. G410-1 正式 100 种子/PSN 100 面仍单独执行。
4. D410-DIAG / D410-RECOVERY 按 WP03 后续要求推进。
