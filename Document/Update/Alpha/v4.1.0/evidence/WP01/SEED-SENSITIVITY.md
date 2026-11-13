# WP01 种子敏感性校准（seed lever）

- 时间：2026-11-13T00:42:00+08:00
- 状态：**verified_with_warning**；这不是能力门，而是评测分母规则。
- 引擎：`rl_v3` SHA256 `691bfd77…68ef6`；stage 0 / normal / team 0；固定动作 `cand=0,bomb=0`，2 帧决策。
- 方法：逐决策比较 25 个 `STATE` 字段，**只忽略 `rng_seed` 本身**（它被 HASH 跟踪，会让跨 seed 的 hash 从第 0 帧就不同，造成“seed 已生效”的假象）。

## 1. 结果

| 探测 | 资源 | 决策/帧 | 第一次非 rng 分歧 | 说明 |
|---|---|---:|---|---|
| 第一命 | standard | 350 / 700 | **无** | 第一命 700 帧内三个 seed 的 gameplay 字段完全相同；still 在 706 帧首死，死在 seed 生效之前 |
| 首死后继续 | standard | 3,000 / 6,000 | `1_vs_2` frame **906**（score）；`1_vs_3` frame **952**（score） | 首死/重生后才出现 seed 依赖；随后 graze/power/items 也分叉 |
| 首死后继续 | practice | 3,000 / 6,000 | `1_vs_2` frame **970**（score）；`1_vs_3` frame **1078**（stage_generation） | 与 standard 同源，frame 略有差异 |

原始结果：[`seed_probe_first_life.json`](seed_probe_first_life.json)、[`seed_probe_standard.json`](seed_probe_standard.json)、[`seed_probe_practice.json`](seed_probe_practice.json)。脚本：[`probe_seed_first_life.py`](probe_seed_first_life.py)、[`probe_seed_standard.py`](probe_seed_standard.py)、[`probe_seed_practice.py`](probe_seed_practice.py)。

## 2. 为什么重要

1. **早期死亡局不是独立 seed 样本**：如果一局在约 frame 1,000 前结束（例如 still 在 706 帧首死），seed 1/2/3 产生的是同一条轨迹。把它们算成 3 个独立 episode 会高估有效样本量。
2. **基线三 seed 的差异来自策略 RNG，不是环境 seed**：random 的 1,410/1,806/782 帧来自 `default_rng(seed*1000+17)` 的动作差异；still/fixed 三 seed 帧数完全相同，因为都死在 seed 生效之前。
3. **G1 teacher 的 seed 差异是合理的**：G1 的 S1N 20 个 seed 帧数 13,196–13,964，存活超过 13k，已越过首死后的 seed 分叉点；这与“seed 在首死后才生效”一致，不说明 seed 全局无效。
4. **正式门必须报有效独立样本**：S1N/PSN/RUNN/G410-2/4/5/6/7 在跑 100 seed 前，必须先为对应策略/分段测出 first divergence frame；在此之前结束的 episode 只能算同一条轨迹，并在分母里显式去重。

## 3. 规则（写入 WP01 分母）

- 名称：`seed_sensitivity_rule`。
- 前置：对目标策略/分段跑一次固定动作或当前策略的 seed 敏感性校准，记录 `first_non_rng_divergence_frame`。
- 分母：`effective_n = distinct trajectories after grouping episodes that never reach the first divergence frame`。
- 报告：原始 episode 数、有效独立 episode 数、重复组、first divergence frame 必须同时出现；不得只报 100 局。
- 不通过条件：若校准显示目标策略在整个 episode 内从未越过 seed 分叉点，则这批 seed 不能作为独立样本，相关门不能以“N=100”结算。

## 4. 限制

- 只测了 stage 0 / normal / team 0 / 固定动作；其他面/策略/起点的 first divergence frame 必须单独校准。
- 只比较 25 个导出字段；未导出的敌人/弹幕内存不在比较范围内。
- 这次没有跑 teacher/student；G1 的引用是上下文，不是本次重测。
- 本文件是分母警告，不是能力结论；它不把任何 G410 门标为通过。
