# G410-1 教师正式确认 100 种子汇总（教师腿）

- 时间：2026-11-13T14:38:00+08:00
- 范围：确认种子 **61300–61399**（100 个唯一 seed，四批 25 个）
- 证据（四份 JSONL + summary）：
  - [chunk1](g4101_s1n_chunk1.jsonl) / [summary](g4101_s1n_chunk1_summary.json) / [md](G4101_CHUNK1.md)：61300–61324
  - [chunk2](g4101_s1n_chunk2.jsonl) / [summary](g4101_s1n_chunk2_summary.json) / [md](G4101_CHUNK2.md)：61325–61349
  - [chunk3](g4101_s1n_chunk3.jsonl) / [summary](g4101_s1n_chunk3_summary.json) / [md](G4101_CHUNK3.md)：61350–61374
  - [chunk4](g4101_s1n_chunk4.jsonl) / [summary](g4101_s1n_chunk4_summary.json) / [md](G4101_CHUNK4.md)：61375–61399

## 数字（四批直接合并）

| 指标 | 值 |
|---|---:|
| runs / unique seeds | **100 / 100** |
| clears | **100** |
| deaths | **0** |
| stop_reason | 100 × stage_changed |
| frames total | **1,336,576** |
| frames mean / median | **13,365.76 / 13,292** |
| engine SHA | 100/100 `1462829b…0d2f` |
| plans_verified | 334,144 = decisions（verify_k=1） |
| raw 1e-3 trajectory_mismatch | **270,501 / 334,144 = 80.95%** |
| geo_dead_engine_alive / geo_alive_engine_dead | **0 / 0** |

- 教师过面率 **100/100 = 100%**，高于 G410-1 教师腿的 90/100 阈值。
- missing=[]、failed=[]、重试 0；引擎 SHA 全部匹配。

## 仍不能判 G410-1 通过的原因

1. **8px 轨迹审计只在 dev seed 1000 做过**：raw 75.55% → 8px **0.769%**（25/3,251），其中 8 条是过面前真实漂移。100 个确认种子的 8px 残余尚未计算/重跑；`v410_teacher_fast20.py` 的 `verification_ok` 只检查 mode/SHA/plans_verified，**不检查 trajectory_mismatch**。因此“无执行与验证不一致”不能据此宣称满足。
2. **数据腿未完成**：Stage 1 数据 66,143（D410-S1）可用，但多面/PSN 与祖先切分、起点分布未结算；G410-1 要 Stage 1 与多面分表。
3. 本汇总只证明教师过面与执行完成，不证明像素可学性，也不等于 G410-2。

## 结论

- 教师腿的**过面计数**已满足 90/100；但 G410-1 整体仍 **in_progress**，因为 100 种子 8px 审计与数据腿未闭合。
- 不允许把四批的 chunk `fast_gate=PASS` 或本 100/100 单独写成“G410-1 通过”。
