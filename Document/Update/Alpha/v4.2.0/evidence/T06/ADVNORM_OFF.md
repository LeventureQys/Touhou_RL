# T06 / N-a②：`ppo.advantage_normalisation: true → false`（50k，单 seed）

> **这不是 R1。** 奖励与终止取自引擎真值（`engine_truth`），是**训练期特权诊断**。本文件没有任何能力结论，也不构成 G420-4 的证据（本档只有 40 局；G420-4 要 100 局、≥50 过关、下界 ≥0.35）。
>
> **单 seed。** 只有 `run_seed = 0`；不外推到 3 seed / 1M / 5M。
>
> **n = 20 无区间。** 主判据是点比较；配对区间宽 ±220 / ±400 帧；40 局的过关数只有二项上界。点估计差不是效应量。

| 项 | 值 |
|---|---|
| 规格 | `Document/Update/Alpha/v4.2.0/experiments/T06-advantage-norm-spec.md` |
| 配置 | `configs/v420/rl-engine-advnorm-off.json` sha256 `ca1dca15393d7983b493f1f1fa5254d2c686f7ce7a27474febd95c5742bb9fbe` |
| 基线（逐字节照抄） | `configs/v420/rl-engine-r3-pilot.json` sha256 `83f16c8a34090ae2994e51835624fb2e9252adc9e668a4b653ac0ba4d1f0673c` |
| 唯一语义改动 | `ppo.advantage_normalisation` `true` → `false` |
| 训练 payload | `.tmp_v420_data/T06/advoff/pilot-advoff-a.json` sha256 `d5c23faa6bd7604eb96182de9a22240e1b6a83b865916a220c4004560bcfadc5` |
| checkpoint | `.tmp_v420_data/T06/checkpoints/advoff_seed0_50000f.pt` sha256 `f84e8b2e95a518d37f38d1552eca8065149c0b9b63ec63404a9ad86cf3cd4e73` |
| 预注册判据 | 冻结在配置的 `advantage_norm_only.criteria`（`frozen_before_run: true`） |

## 0 一句话判定

**R4 成立。** 主判据 fail ∧ 能力 0/40。**白化这一行不是这条差距的机制**（r3 配方、50k、单 seed 下）。三个候选里唯一零代码的那个被排除；N-a①/N-c 按 §1.2 被预测为 null；N-b 无良定义 ⇒ PPO 侧在本预算下的可调清单已经穷尽。

- 主判据（两腿合取）：`diag_first_death` 1,481.9 帧（线 1,636.8）⇒ **fail**；`s1n` 2,660.7 帧（线 3,011.5）⇒ **fail**；合取 = **fail**。
- 能力判据：40 局 `stage_clear` = **0/40**（diag 0 / s1n 0）⇒ **fail**；严格读法（每口径各 ≥1）= **fail**。
- **下一步**：只剩四条，都不是 PPO 系数：(a) 射击脉冲错配的量化与修复、(b) 3 seed × 1M（需另编预算）、(c) KL 判据修订后的 50k 复现、(d) 换观测/表征（2 帧节拍已被 H-a 证伪）。

## 1 变更：一个键，其余逐字节不变

`configs/v420/rl-engine-advnorm-off.json` 由 `configs/v420/rl-engine-r3-pilot.json` 逐字节复制，只改一个语义键，另有输出命名、私有 `cp -a` data 目录和冻结判据块：

| 叶路径 | 基线 | 本臂 |
|---|---|---|
| `ppo.advantage_normalisation` | `true` | **`false`** |
| `pilot.checkpoint_tag` | `r3` | `advoff` |
| `dev_eval.tag` | `R3` | `ADVOFF` |
| `dev_eval.runs_dir` | `T06/dev-r3` | `T06/dev-advoff` |

`tests/test_v420_rl.py` 里三个测试把这件事变成可检查的：逐叶差异断言、**逐行重建**（把新增块删掉、把 10 个改动的行换回基线行，字节必须与基线相同）、以及 36 个变异全被抓住的「牙齿」检查（`.tmp_v420_advoff/check_teeth.py`，36/36 caught，退出码 0）。

## 2 50k 训练读数

| 量 | 值 |
|---|---:|
| 帧 / 决策 | 50000 / 25005 |
| episode 数 | **22** |
| 终局分布 | `{"game_over": 22}` |
| 终局奖励次数（stage_clear bonus） | **0** |
| 训练内部 `stage_clear` 计数（`reward_totals`） | 0.0 |
| 死亡总数（`reward_totals.death`） | -16.5 |
| 更新数 | **49** |
| 墙钟 / fps | 1,007.9 s / 49.61 fps |
| 每更新 `max\|ΔW\|` 均值 | **2.32378e-04** |
| └ / `8 × actor_lr` | 0.968240 |
| └ 逐更新 min / median / max | 2.19155e-04 / 2.32436e-04 / 2.39126e-04 |
| └ sd（逐更新） | 4.01270e-06 |
| per-episode return 均值 / sd | -0.710 / 0.007 |
| 训练期首死均值 | 1,252.5 帧（22 局，752–1,702） |
| `advantage_sd` | 1.73393e-04（见下注） |

> **`advantage_sd` 的口径**：payload 里没有 advantage 统计（`rl.py:4399-4415` 不记），所以这个数是**同配方的审计槽读数** `1.73393e-04`（`advantage_mean = 9.24231e-04`、`min = 4.98903e-05`、`max/min = 19.936842` ≈ GAE 视界 `19.936872`、`mean/sd = 5.3303`）。白化开关只改 `adv_t`（`rl.py:2171-2174`），**不改** GAE 的输出，因此这个数与 r3 的审计槽逐比特同值；它不是本臂训练 rollout 上量出来的，训练 rollout 的同一量本版仍未落盘（规格 §1.5 的缺口）。

### 2.1 训练四等分首死（并列诊断，**不参与判定**）

两个边界规则都列出，因为既有三档的发表值分别由不同规则复现（r3@50k 的 6/5/5/6 −7.86% 来自 index-round；r3@200k 的 22/23/22/23 −1.94% 来自 index-floor；两者互相不可复现，见 `extract.py --self-test` 的输出）：

| 臂 | 局数 | index-round 分格 / Q4−Q1 | index-floor 分格 / Q4−Q1 |
|---|---:|---|---|
| R2 无锚 @50k | 24 | 6/6/6/6 -29.29% | 6/6/6/6 -29.29% |
| R2+KL @50k | 21 | 5/5/6/5 +17.99% | 5/5/5/6 +21.48% |
| r3 @50k | 22 | 6/5/5/6 -7.86% | 5/6/5/6 -8.87% |
| r3 @200k | 90 | 22/23/23/22 -1.73% | 22/23/22/23 -1.94% |
| **N-a② advnorm off @50k** | 22 | 6/5/5/6 +4.79% | 5/6/5/6 +10.87% |

本臂在 index-round 下是 **+4.79%**（对照 r3@50k 的 -7.86%、r2 的 -29.29%），在 index-floor 下是 **+10.87%**（对照 r3@200k 的 -1.94%）。每格只有约 5.5 局，**不得**据此说「白化让首死变长/变短」——只能说「有没有观察到随训练变坏」。

### 2.2 曲线首末值（payload 已有）

| 量 | 首 | 末 | 均值 |
|---|---:|---:|---:|
| approx_kl（T=8） | 0.000358001 | 0.00106734 | 0.00118543 |
| entropy | 2.45152 | 2.47845 | 2.51921 |
| clip_fraction | 0 | 0.00195312 | 0.00460379 |
| value_loss | 0.0130033 | 0.00837229 | 0.00742565 |
| policy_loss | 0.00882886 | -0.00532048 | 0.00373841 |
| grad_norm（裁剪前） | 8.4064 | 9.49536 | 9.06164 |
| KL_T1(A0‖π)（rollout 最大） | 0.355657 | 0.504964 | 0.389825 |

## 3 §7 fail-closed 追溯检查（`max|ΔW|`）

带：`2.38e-4 ± 5% = [2.2610e-04, 2.4990e-04]`（由来 `8 × 3e-05 × [0.9916, 1.0013] = [2.37984e-04, 2.40312e-04]`，更新步数 = `epochs 2 × decisions_per_update 512 / minibatch 128 = 8`）。

- **落在带内。** 每更新 `actor_max_abs_parameter_change` 的均值是 **2.32378e-04**（除以 `8 × actor_lr` 得 0.968240），落在 `2.38e-4 ±5% = [2.2610e-04, 2.4990e-04]` 内。四臂实测带是 `[2.37984e-04, 2.40312e-04]`（即 `/8lr ∈ [0.9916, 1.0013]`），本臂落在它**之下**（相对其下沿 -2.36%），方向是**步幅略小**而不是略大。⇒ **P-N2 成立（按均值判），本实验不被追溯认定为学习率类修订**：白化没有在调步幅，§1.2「Adam 已把步幅与优势尺度解耦」的前提保住了；后续同类改动不因此受额外限制。
- 逐更新极值 `[2.19155e-04, 2.39126e-04]`；落在硬带外的更新数 = **2**（四臂的逐更新极值同样略高于带的上沿，说明带是对**均值**定义的，所以判定用均值、离散度并列报出）。

| 臂 | 每更新均值 | / `8 × actor_lr` |
|---|---:|---:|
| R2 无锚 | 2.40315e-04 | 1.001311 |
| R2+KL | 2.38087e-04 | 0.992030 |
| r3 @50k | 2.38292e-04 | 0.992883 |
| r3 @200k | 2.37981e-04 | 0.991587 |
| **N-a② advnorm off** | 2.32378e-04 | 0.968240 |

**结论：落带内 ⇒ NOT 学习率类修订（后续同类改动不受额外限制）。**

## 4 两口径 20 dev（seed 1000–1019，严格串行，私有 `cp -a`）

仪器自检：五个既有臂在这次运行里被重算，全部复现其发表均值（`{"R0_untrained_anchored": [1636.8, 3011.5], "R2_no_anchor": [889.35, 1906.4], "R2_plus_KL": [1512.1, 2626.0], "r3_50k": [1589.2, 2891.6], "r3_200k": [1575.2, 2808.6]}`，problems = `[]`）。

| 口径 | 均值(帧) | n | sd | min–max | 过关 |
|---|---:|---:|---:|---|---:|
| diag_first_death | **1,481.9** | 20 | 271.2 | 818–1,939 | 0/20 |
| s1n | **2,660.7** | 20 | 348.8 | 2,184–3,324 | 0/20 |

逐局过关数：diag **0/20**、s1n **0/20**，合计 **0/40**（过关 seed：diag `[]`、s1n `[]`）。

逐局记录在 `.tmp_v420_advoff/dev_compare.json#summary.N_a2_advnorm_off.*.per_seed`（每 seed 的 `frames` / `clear` / `stop_reason`）。停止原因分布：diag `{"death_stop": 20}`、s1n `{"game_over": 20}`。

### 4.1 逐 seed 配对差

#### vs r3 @50k

| 口径 | Δ均值(帧) | sd | se | t | 95% t-区间 | 更好/更差 seed |
|---|---:|---:|---:|---:|---|---:|
| diag_first_death | -107.3 | 363.6 | 81.3 | -1.320 | [-277.5, +62.9] | 8/12 |
| s1n | -230.9 | 667.6 | 149.3 | -1.547 | [-543.4, +81.5] | 8/12 |

- diag_first_death 逐 seed 配对差（advnorm_off − r3 @50k）：1000:-308, 1001:+23, 1002:-631, 1003:-14, 1004:-592, 1005:+263, 1006:-330, 1007:-134, 1008:-385, 1009:+211, 1010:+48, 1011:+173, 1012:+439, 1013:-146, 1014:-394, 1015:+480, 1016:-917, 1017:-4, 1018:-95, 1019:+167
- s1n 逐 seed 配对差（advnorm_off − r3 @50k）：1000:-535, 1001:+328, 1002:-1572, 1003:+126, 1004:+190, 1005:+490, 1006:-888, 1007:-613, 1008:-1246, 1009:+152, 1010:+835, 1011:-254, 1012:-239, 1013:-114, 1014:-318, 1015:-30, 1016:-1382, 1017:+326, 1018:-521, 1019:+646

#### vs R0（未训练但带锚的 A0）

| 口径 | Δ均值(帧) | sd | se | t | 95% t-区间 | 更好/更差 seed |
|---|---:|---:|---:|---:|---|---:|
| diag_first_death | -154.9 | 315.8 | 70.6 | -2.194 | [-302.7, -7.1] | 7/12 |
| s1n | -350.9 | 799.3 | 178.7 | -1.963 | [-724.9, +23.2] | 6/14 |

- diag_first_death 逐 seed 配对差（advnorm_off − R0（未训练但带锚的 A0））：1000:+71, 1001:-889, 1002:-201, 1003:-77, 1004:-592, 1005:+150, 1006:+218, 1007:+4, 1008:+0, 1009:+208, 1010:-177, 1011:+245, 1012:+42, 1013:-218, 1014:-561, 1015:-410, 1016:-92, 1017:-3, 1018:-636, 1019:-180
- s1n 逐 seed 配对差（advnorm_off − R0（未训练但带锚的 A0））：1000:+350, 1001:-1182, 1002:+944, 1003:-198, 1004:-56, 1005:+391, 1006:-42, 1007:-1004, 1008:-918, 1009:-1415, 1010:+487, 1011:+784, 1012:+72, 1013:-611, 1014:-126, 1015:-2290, 1016:-702, 1017:-888, 1018:-606, 1019:-7

#### vs R2 无锚

| 口径 | Δ均值(帧) | sd | se | t | 95% t-区间 | 更好/更差 seed |
|---|---:|---:|---:|---:|---|---:|
| diag_first_death | +592.5 | 290.7 | 65.0 | +9.115 | [+456.5, +728.6] | 19/1 |
| s1n | +754.2 | 340.9 | 76.2 | +9.895 | [+594.7, +913.8] | 20/0 |

- diag_first_death 逐 seed 配对差（advnorm_off − R2 无锚）：1000:+463, 1001:+76, 1002:+464, 1003:+747, 1004:-157, 1005:+806, 1006:+878, 1007:+661, 1008:+481, 1009:+967, 1010:+657, 1011:+824, 1012:+574, 1013:+475, 1014:+835, 1015:+946, 1016:+516, 1017:+395, 1018:+363, 1019:+880
- s1n 逐 seed 配对差（advnorm_off − R2 无锚）：1000:+652, 1001:+581, 1002:+1027, 1003:+654, 1004:+627, 1005:+1152, 1006:+662, 1007:+552, 1008:+475, 1009:+889, 1010:+992, 1011:+1497, 1012:+937, 1013:+241, 1014:+1139, 1015:+800, 1016:+386, 1017:+247, 1018:+392, 1019:+1183

## 5 六臂对照

| 臂 | 更新 | 局 | 每更新 mean\|ΔW\| | 首死 Q4−Q1 (round) | (floor) | diag 均值(n) | 过关 | s1n 均值(n) | 过关 |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| R0 (untrained anchored A0) | — | — | — | — | — | 1,636.8 (20) | 0 | 3,011.5 (20) | 0 |
| R2 无锚 | 49 | 24 | 2.40315e-04 | -29.29% | -29.29% | 889.4 (20) | 0 | 1,906.4 (20) | 0 |
| R2 + KL 锚 | 49 | 21 | 2.38087e-04 | +17.99% | +21.48% | 1,512.1 (20) | 0 | 2,626.0 (20) | 0 |
| r3 @50k | 49 | 22 | 2.38292e-04 | -7.86% | -8.87% | 1,589.2 (20) | 0 | 2,891.6 (20) | 0 |
| r3 @200k | 196 | 90 | 2.37981e-04 | -1.73% | -1.94% | 1,575.2 (20) | 0 | 2,808.6 (20) | 0 |
| **N-a② advnorm off @50k** | 49 | 22 | 2.32378e-04 | +4.79% | +10.87% | 1,481.9 (20) | 0 | 2,660.7 (20) | 0 |

读法（**不是效应量**，n=20 的配对区间已列在 §4.1）：

- 对 r3@50k：diag **-107.3 帧**（t = -1.320，95% 区间[-277.5, +62.9]），s1n **-230.9 帧**（t = -1.547，95% 区间[-543.4, +81.5]）——两条区间都含 0，所以「关掉白化比 r3@50k 更差」只能说成**点估计方向**，不能说成已确立的效应。

- 对 R0：diag **-154.9 帧**（t = -2.194，95% 区间[-302.7, -7.1]，**不含 0**），s1n **-350.9 帧**（t = -1.963，95% 区间[-724.9, +23.2]，含 0）。对照 r3@50k 相对 R0 是 −47.6 / −119.9 帧（t = −0.48 / −0.50）：**本臂不是「回到 R0 线附近」，而是离 R0 更远了**，diag 这一腿在 20 局配对下已经越过 0。这与 R4 一致：去掉白化并没有把差距补上，反而略微扩大。

五条既有参照与 R0 一并在 §4 的仪器自检里复现；本臂的 checkpoint 是 `.tmp_v420_data/T06/checkpoints/advoff_seed0_50000f.pt`（sha `f84e8b2e95a518d37f38d1552eca8065149c0b9b63ec63404a9ad86cf3cd4e73`），回读证明 `loaded=True`、`bitwise_identical=True`、`logit_max_abs_diff=0.0`、`rows=16`。

## 6 四条裁决规则

| # | 组合 | 结论 |
|---|---|---|
| R1 | 主判据 hold ∧ 能力 hold | criterion 1 hold AND criterion 2 hold: the whitening line is load-bearing on the r3 recipe and is the first PPO-side single factor to push both rulers back to the R0 line AND to produce a clear.  Still NOT R1 (engine_truth privileged diagnostic), still single seed, still 20 episodes with no interval, and it must not be read as 'the whitening is the main cause of the collapse'. |
| R2 | 主判据 hold ∧ 能力 0/40 | criterion 1 hold AND criterion 2 fail: the whitening is a sufficient single factor for pushing both dev point estimates back to the R0 line, but the R0 line is not a capability line -- another 'point estimate inside the band, 0 clears'. |
| R3 | 主判据 fail ∧ 能力 hold | criterion 1 fail AND criterion 2 hold: point estimate and capability have come apart -- the dev means are below the R0 line yet one episode really cleared, so the R0 line is not a necessary condition and the 40-episode mean is not a sufficient statistic. |
| R4 | 主判据 fail ∧ 能力 0/40 | criterion 1 fail AND criterion 2 fail: the whitening line is not the mechanism of this gap (r3 recipe, 50k, single seed); the only zero-code candidate of the three is excluded, N-a1 / N-c are predicted null by the arithmetic and N-b has no well-defined form, so the PPO-side list at this budget is exhausted. |

**本臂落在 R4。** 主判据 fail ∧ 能力 0/40。**白化这一行不是这条差距的机制**（r3 配方、50k、单 seed 下）。三个候选里唯一零代码的那个被排除；N-a①/N-c 按 §1.2 被预测为 null；N-b 无良定义 ⇒ PPO 侧在本预算下的可调清单已经穷尽。

## 7 这不是 R1

- 奖励与终止用引擎真值（`reward.source = engine_truth`），这是**训练期特权诊断**；部署侧是 T=1 argmax 的像素策略，两者不是同一件事。
- 本档的 dev 评测是 `tools/rl/v410_student_eval.py` 的冻结口径，但它读的 checkpoint 是**用引擎真值奖励训练出来的**，所以 40 局的结果只能是特权诊断读数。
- G420-4 需要 100 局 / ≥50 过关 / Wilson 下界 ≥0.35。本档 40 局（0 过关）两处都不够：0/40 的 Wilson 上界是 0.087624，1/40 的区间是 [0.00443, 0.12882]。
- 因此本文件里**不存在**任何「能力达成」的表述；`R1` 这个规则名只是预注册矩阵里的一个格名，不是「R1 级证据」。

## 8 单 seed / n=20 无区间

- **单训练 seed**：只有 `run_seed = 0`。r3@50k / r3@200k 也是单 seed，所以三/五/六臂之间**不是同轮**比较，归因带「轮次差异」。
- **n = 20 点比较**：主判据是 `≥` 的点比较，不是区间比较。分辨率披露：r3@50k 的 20 局 sd 是 322.2（diag）/ 675.2（s1n）⇒ 均值标准误 72.05 / 150.98 ⇒ R0 线距 r3@50k 只有 0.66 / 0.79 个标准误；配对区间宽 ±220 / ±400 帧。**<5% 的差距在这台仪器上与噪声不可分。**
- **40 局过关数只有二项上界**：0/40 时 Wilson 95% 上界 0.087624；1/40 时 [0.00443, 0.12882]。**不得**把点估计差当效应量，也不得用 1/40 宣布能力。

## 9 方向性预测（跑前写死，与判据分开报）

| 编号 | 预测 | 实测 | 判定 |
|---|---|---|---|
| P-N1 | 首死四等分比 r3@50k 的 −7.86% 更负、且 T=8 熵更低（均值 < 2.4945） | 首死 Q4−Q1 = +4.79%（round 规则）；熵均值 2.5192（首 2.4515 → 末 2.4785，r3@50k 参照 2.4945 / 2.4307 / 2.5758） | 首死腿 fail；熵腿 fail |
| P-N2 | 每更新 `max\|ΔW\|` 落在 `2.38e-4 ±5%` | 均值 2.32378e-04（/8lr = 0.968240，低于四臂带下沿 -2.36%），逐更新极值 [2.19155e-04, 2.39126e-04] ⇒ hold | hold |
| P-N3 | 40 局 `stage_clear` 仍为 0 | 0/40 | hold |
| P-N4 | 前向对比的 argmax 翻转率 / KL_T1 不得优于 R2+KL 的 0.011719 / 0.005817 | 本档未做前向对比 | 不适用 |

## 10 复现

```bash
# 训练（单实例，整段 50k，不重启：活 episode 由 collect_rollout(resume=carry) 携带）
bash .tmp_v420_advoff/run_pilot.sh a
# 两口径各 20 局，四批 × 5 seed，严格串行，私有 cp -a data 目录
bash .tmp_v420_advoff/run_dev_stream.sh diag_first_death diag
bash .tmp_v420_advoff/run_dev_stream.sh s1n s1n
# 读数与判定（会先复现四个既有臂的发表数字，再读取本臂）
/root/v400-venv/bin/python .tmp_v420_advoff/extract.py --self-test
/root/v400-venv/bin/python .tmp_v420_advoff/pair.py --criteria
/root/v400-venv/bin/python .tmp_v420_advoff/build_evidence.py
```

## 11 剩余阻塞（据实）

| 阻塞 | 影响 | 解除需要 |
|---|---|---|
| 训练 rollout 的 advantage 统计仍不落盘 | 白化分母只能引用审计槽；事后归因精度受限 | 改 `rl.py` 记录 `adv_t` 统计（本轮白名单不允许） |
| 一个布尔同时切换偏移与尺度 | 单次运行无法分离是哪一半承重 | 跑 N-a①（需授权改 `rl.py` + 测试） |
| 比较基是既有产物，不是同轮 | 归因带轮次差异 | 同轮「r3 但 ε=δ=0」消融 |
| 射击脉冲 train/deploy 错配从未量化 | 它是「永远不可能过关」的必要条件之一，本档测不到 | 一次专门的帧级测量 |
| dev n=20 分辨率 ±220/±400 帧 | 判定是规则判定，不是效应量确认 | 更大 dev 分配（需协议/预算决策），或区间判据（须先声明再跑） |

*本文件由 `.tmp_v420_advoff/build_evidence.py` 从实测 payload 生成；机器可读副本 `advnorm_off.json`；`extract.py --self-test` 在生成前复现了四个既有臂。*
