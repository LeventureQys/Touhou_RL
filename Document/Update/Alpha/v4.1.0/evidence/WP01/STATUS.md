# WP01.1 / WP01.2 状态（主代理复核）

- 时间：2026-11-13T09:30:00+08:00
- 结论：**WP01 = passed（带 warnings）**。`check --wp WP01` 7 个子门全部 passed、exit 0；warnings：30 分钟运行有 1 次 budget_cut（seed2 reset345，399/1000 帧，mismatch=0）、baselines 是 engine-state 只读而非严格像素、seed_sensitivity 是 warning。**G410-0 已 passed**：G0.1 108/108 已完成并报告（[`G0.1.md`](G0.1.md)、[`g0.1-108.json`](g0.1-108.json)，0 mismatch/0 failed/0 timeout，12,311,460 行）。

## 1. 已独立复核的证据

| 项 | 结果 | 证据 |
|---|---|---|
| CPU/GPU 与前向一致性 | 37 passed；离线/在线 CPU max_abs 0.0（B=1）/1.19e-07（B=4）；CPU vs CUDA 2.38e-07 / 1.19e-07；argmax 全等；CUDA 未 skip | [`parity.json`](parity.json)、[`PARITY.md`](PARITY.md) |
| 真实引擎主切片 | 3 seeds × 3 resets × 1,000 frames = 9,000 帧；同 seed 三次重开逐帧相同；mismatch/errors/stale = 0/0/0 | [`consistency.json`](consistency.json) |
| 主切片独立复跑 | 3 个 seed 的 `sequence_sha256` 与首次运行逐字节相同 | [`consistency-independent.json`](consistency-independent.json) |
| 像素切片（role=env） | 1 seed × 2 resets × 200 帧；16 个采样点 play/hud 字节 SHA 相同；`pixel_identical=true` | [`pixels.json`](pixels.json) |
| 2 分钟稳定性 | 69 resets / 68,608 帧 / errors=0 / stale=0 | [`stability-2min.json`](stability-2min.json) |
| **30 分钟稳定性** | **1,039 resets / 1,038,399 帧 / errors=0 / stale=0 / wall=1800.003 s / seeds_identical=true**；仅 1 次墙钟截断（seed2 末次，399/1000 帧，`budget_cut`），前缀仍逐帧一致 | [`stability-30min.json`](stability-30min.json) |
| 信息边界/终止语义 | 151 passed；伪造裁判状态不改变 policy/feedback；privileged key 全部拒绝；终止优先级 game_over>stage_clear>first_death>cap | [`boundary.json`](boundary.json)、[`BOUNDARY.md`](BOUNDARY.md) |
| 非学习基线（真实引擎） | 39 passed；12 局（4 policies × 3 seeds）全部 first_death；overall 6,352 frame/s、3,176 decision/s；无异常 | [`baselines.json`](baselines.json)、[`BASELINES.md`](BASELINES.md) |
| 种子敏感性校准 | 首命 700 帧 seed-invariant；首死后 standard 约 frame 906/952、practice 约 970/1078 才出现非 rng 分歧；早期死亡局不能算独立 seed 样本 | [`seed_sensitivity.json`](seed_sensitivity.json)、[`SEED-SENSITIVITY.md`](SEED-SENSITIVITY.md) |
| 范围 | 未修改 `touhou_rl/v400/` 或 `tests/test_v400_*`；未训练、未采数据集 | `git status --porcelain` 无 v400 输出 |

## 2. 尚未完成 / 已知缺口

1. **G0.1 108/108：已完成**（[`G0.1.md`](G0.1.md)、[`g0.1-108.json`](g0.1-108.json)；108 identical、0 mismatch、0 failed、0 timeout、12,311,460 行）。G410-0 因此 passed。
2. **game_over / stage_clear / cap 的真实样本**：基线 12 局全部 first_death；当前只有 `classify_termination` 的合成单测。WP01 已通过，但这些真实终止样本仍应在 WP02/WP06 前补，避免把合成语义当真实覆盖。
3. **seed sensitivity**：首命 700 帧 seed-invariant；早期死亡局不能算独立 seed 样本。后续每个门必须按 `seed_sensitivity_rule` 报有效独立 episode，不得只报 N=100。

## 3. 下一步

- 启动 WP02 教师质量与标签可审计性：fast vs v2h 同构建/同节拍、`verified`/`survived`/`died` 语义审计、1,000 状态诊断集、death-bomb 隐藏 PLAYER_HIT 对照。
- G0.1 已收尾，不再需要前台重试；正式 G410-1 的 100 新确认种子/PSN 100 面安排在教师审计与预算确认之后。
