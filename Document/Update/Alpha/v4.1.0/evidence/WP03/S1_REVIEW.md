# D410-S1 覆盖审计复核（主代理）

- 时间：2026-11-13T03:10:00+08:00
- 复核对象：[`s1_full_audit.json`](s1_full_audit.json)（WP03.1b 产物）
- 结论：**不接受 `passed=true`。当前合并集不能作为 D410-S1 通过证据。**

## 1. 表面达标项

`s1_full_audit.json` 报告：

- total_decisions **65,636**、unique_decisions **65,636**、unique_seeds **20**、shard_count 20；
- observations_present **true**、observation_rows 65,636、duplicate decision keys 0、synthetic_padding false；
- calibrated alive 65,346 / dead 0 / unknown 290；
- `against_s1.passed=true`（阈值检查全部通过）。

## 2. 阻塞级问题：train/dev 种子混用

`plan_seed_set` 明确记录：

- planned seeds：1000–1019；
- actual seeds：**1–8 + 1000–1011**；
- `matches_plan=false`。

其中 **1–8 是 train seeds**，**1000–1011 是 dev seeds**。D410-S1 是“第一面 BC 与感知预训练”数据；README 要求 Stage-1 数据按原始轨迹组切 train/val/test，训练种子固定且 dev 保留用于开发评估。把 train 与 dev seeds 混成一个训练语料会：

1. 使 dev seeds 进入训练，污染 WP05 的开发评估；
2. 使 S1 的“20 个独立起始种子”不是同一协议集合，不能按计划复现；
3. 使 `against_s1` 的阈值检查通过但**协议身份不成立**——这是覆盖校验器漏掉的约束，不是可接受的偏差。

`s1_full_audit` 已经把偏差写进 `plan_seed_set.note`，但没有让 `passed=false`；主代理复核据此拒绝该通过。

## 3. 当前可接受的部分

- 12 个新采集 shard 的真实 obs + work verified labels 对齐链路成立：39,011 decisions / 12 seeds / alive 38,858 / dead 0 / unknown 153；obs 与 work labels 均真实。
- 已有 8 个 train seed shard（1–8）的 obs+labels 也真实。
- 因此 **WP03.1b 的采集/重标机制通过**，但 **S1 数据集仍未通过**。

## 4. 下一步（WP03.1c）

- 采集 **12 个 train seeds（9–20）**，与已有 8 个 train seeds（1–8）合并为 **20 个 train seeds**；
- 不使用 dev seeds 1000–1011 作为 S1 训练语料；它们保留为 dev/诊断用途；
- 复核器必须显式检查 seed policy：20 seeds 全部属于 train，且不与 dev/confirm 重叠；
- 只有 clean-train S1 在 50k/20 seeds/obs 全部通过后，才可写 S1 passed。
