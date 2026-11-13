# G410-1 教师 chunk1 轨迹一致性审计（raw strict 1e-3）

- 输入：`g4101_s1n_chunk1.jsonl`（25 seeds、61300–61324，全部 stage_clear、0 death）
- 引擎 SHA：`1462829b…0d2f`；plans_verified 合计 **83,673**。
- `consistency.trajectory_mismatch` 合计 **67,554 / 83,673 = 80.74%**（`_modelled_positions` 误差 > 工具默认 tolerance 1e-3）。
- 同时：`geo_dead_engine_alive=0`、`geo_alive_engine_dead=0`、`agree_alive=83,673`、`agree_dead=0`。

## 解释与限制

1. 这里的 80.74% 是 **严格 1e-3** 的几何轨迹不一致，不等于死亡判定错误：25/25 实际过面、0 death，且几何与引擎死亡语义无矛盾。
2. WP02.4 的校准结论是：**8.0 px** 容差对 54,341 个 alive 决策的覆盖率 **99.231%**；严格 1e-3 不是 v4.1 标签的校准规则。
3. 但 G410-1 要求“无执行与验证不一致”。当前 `v410_teacher_fast20.py` 的 `verification_ok` 只检查 mode/engine SHA/`plans_verified == decisions * verify_k`，**不检查** `trajectory_mismatch`；因此不能把 chunk1 的 25/25 当作 G410-1 教师腿已通过。
4. 最终 100 种子判定必须同时报告：
   - raw strict 1e-3 mismatch 率；
   - 8 px 校准后的 mismatch 率（需要保留/重算每个 verified plan 的 max_position_error，或按 WP02.4 方式重跑校准）；
   - `geo_dead_engine_alive` / `geo_alive_engine_dead`；
   - 执行与验证不一致的具体 seed/plan。
5. 如果 8 px 校准后仍有实质 mismatch（或任一 geo 矛盾），先修教师/验证器或改用 v2h，不得继续用未审计标签扩大 G410-1 结论。

## 8px 校准审计（dev seed 1000，2026-11-13T12:09）

证据：[`g4101_tolerance_audit.json`](g4101_tolerance_audit.json)、[`G4101_TOLERANCE_AUDIT.md`](G4101_TOLERANCE_AUDIT.md)。

- 同一 3,251 个 verified plans：
  - raw 1e-3：trajectory_mismatch **2,456 / 3,251 = 75.55%**；
  - 8.0 px：trajectory_mismatch **25 / 3,251 = 0.769%**；
  - `geo_dead_engine_alive=0`、`geo_alive_engine_dead=0`。
- seed1000 仍 **13,004 帧、0 deaths、stage_clear**；引擎 SHA 匹配。
- 25 个残余中，**8 个是过面前最后 8 个决策的真实模型漂移**（139.58 px），不是阈值噪声；其余为边界/离群。
- 结论：raw 1e-3 的 80% 主要是严格阈值伪差；但 **0.769% 残余不能忽略**，G410-1 的“无执行与验证不一致”仍需在 100 种子正式口径下逐条解释/修复，不能直接判 pass。

## 状态

- **chunk1 只是 25/100 数据**；G410-1 仍 `in_progress`。
- 未修改教师代码；本审计为只读复核。
