# T06 / WP420-05 第一档：小规模真跑（50k 帧）与成本外推

> **这一档已被 [`RL_PILOT2.md`](RL_PILOT2.md) 取代（不是删除）。** 它记录的是两个已知缺陷下的
> 50k 真跑：①`reward_discount_horizon` 失败（τ = 30 s，γ^T ≈ 6.5e-04，终局过关奖励不可见），
> ②`collect_rollout` 每个 rollout 都 reset，导致 `episode 数 = 0`、没有任何终局奖励进入 PPO。
> 两者都已修复：折扣见 [`DISCOUNT_REVISION.md`](DISCOUNT_REVISION.md)，接线见
> [`RL_PILOT2.md`](RL_PILOT2.md) §1.1。本文件的 checkpoint 已原样保留为
> `T06/checkpoints/r2_seed0_50000f.disc-round1.pt`（sha `ca1f8fff…`），本文件的数字**不可**
> 与第二档的数字混用：它们的折扣、接线、吞吐口径都不同。

对象：v4.2.0 `T06`（WP420-05）**第一档**的小规模真跑。
运行产物：[`rl_pilot.json`](rl_pilot.json)（本文件全部数字由它生成）。
原始运行：`.tmp_v420_data/T06/pilot.json`、`.tmp_v420_data/T06/dev/merged-R{0,2}.json`。

**本档只跑 1 个训练 seed × 50,000 游戏帧。** 它不是 1M 帧档，也不是 3 seed。

> R2 uses engine-truth reward and termination, so it is a training-time privileged diagnostic: it is NOT strict pixel self-improvement (R1) and no result here may be reported as R1.

## 1 运行设置

| 项 | 值 |
|---|---|
| arm | `R2`（引擎真值奖励） |
| run seed | 0 |
| 训练 seed（引擎 episode） | [3000, 3001, 3002, 3003, 3004] |
| 帧预算 | 50,000 游戏帧（实跑 50,000） |
| 决策数 | 25,000 |
| 更新数 | 49 |
| episode 数 | 0 |
| 每环境 rollout | 512 决策 |
| 每 episode 帧上限 | 19887 |
| shoot pulse 周期 | 48 决策（96 帧） |
| device | `cpu` |
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern` |
| 初始 checkpoint | `/mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T06/checkpoints/a0_seed0_step5000.pt`（sha `96cb219eac83391f42b18ea656b5a6c5ac81e97cd1fd8c1e2c0d404aee6b89ad`） |
| 训练后 checkpoint | `/mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T06/checkpoints/r2_seed0_50000f.pt`（sha `ca1f8fffb7b4c804cb46c6c93f167100c9557da0f0f72c5dca94e25883052e0e`） |
| critic sidecar | `/mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T06/checkpoints/r2_seed0_50000f.critic.pt`（sha `cb4c347d8dccfeef104fb6cbcc60668d3d183afbf3b49e278c7f456bb6c527bd`） |
| 冻结评估器回读 | loaded=true、arm `B0`、参数 731802、logits 最大差 0、逐比特一致 true |
| 熵系数（事前冻结） | 0.001 |
| 采样温度（审计规则选出后冻结） | 8.0 |

## 2 真实吞吐与更新统计

| 量 | 实测 |
|---|---|
| 墙钟 | 2206.6 s |
| 其中采集（引擎 + 前向） | 1958.1 s |
| 其中 PPO 更新 | 112.8 s |
| 游戏帧/秒（含 reset、含争用） | **22.66** |
| 决策/秒 | 11.33 |

> 采集与更新分开记：采集是引擎渲染 + 模型前向，更新是 `epochs × minibatch` 的反向。
> 二者之和小于墙钟，差值是 episode 之间的引擎重启（每局一次 RESET，5–7 s）。

### 2.1 这条真跑的一个**已知缺陷**（必须和上面的数字一起读）

`episode 数 = 0` 不是"策略从不死"，而是**训练循环的接线错误**：
`collect_rollout` 在每次 rollout 开始时都调用了一次 `env.reset()`，于是 512 决策
（1,024 帧）就被切成一段新 episode；而一局的帧上限是 19,887、A0 的首死约 1,370 帧，
所以**没有任何一局跑完**，也就**没有任何终局奖励或 game_over 进入过 PPO**。
后果：`reward_totals` 为空、advantage 只来自进度项、value loss 只有约 0.006。

- 修法已实现：`collect_rollout(resume=...)` 接收上一段 rollout 的现场（当前观测、episode 编号、
  本局累计回报），连续 rollout 因此接着跑同一局；`tests/test_v420_rl.py` 的两项测试覆盖它
  （`test_collect_rollout_continues_one_episode_across_rollouts`、
  `test_collect_rollout_reports_an_episode_end_and_resets_for_the_next_one`）。
- **修正后的 50k 真跑没有在本档重跑**（主机被另一个采集任务占满、引擎排队），
  所以 §2 的 KL / 熵 / clip / 梯度描述的是"1,024 帧片段"上的更新，
  §3 的 R2 也只是 A0 加上 49 次很小的更新。这是本档最大的缺口，已登记在 §6。
- 审计（[`RL_AUDIT.md`](RL_AUDIT.md)）**没有**这个缺陷：它的真实 episode 都跑到了
  `game_over`（2,186 帧）、帧上限截断（68 帧）与真实过关（13,195 帧），
  所以终止、截断 bootstrap 与 episode 边界本身是被真实验证过的。

| 统计 | 均值 (sd) | 首 → 末 |
|---|---|---|
| approx KL | 0.004 (0.002) | 0.007 → 0.003 |
| 熵 | 2.5343 (0.1087) | 2.4456 → 2.6557 |
| clip fraction | 0.0518 (0.0352) | 0.124 → 0.034 |
| policy loss | -0.0125 | -0.0111 → 2.4088e-04 |
| value loss | 0.0063 | 0.0091 → 0.0064 |
| 梯度范数（裁剪前） | 5.091 | 6.5078 → 5.073 |
| actor 参数最大变化 | 2.402e-04 | 2.399e-04 → 2.385e-04 |
| episode 帧数 | n/a | — |
| episode 回报 | n/a | — |
| episode 结束分布 | {} | — |

训练曲线（等距抽样）：

| update | 帧 | KL | 熵 | clip | policy loss | value loss | grad |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1024 | 0.007 | 2.4456 | 0.124 | -0.0111 | 0.0091 | 6.508 |
| 4 | 5120 | 0.004 | 2.4678 | 0.057 | -0.0158 | 0.0042 | 5.375 |
| 8 | 9216 | 0.009 | 2.4422 | 0.147 | -0.0187 | 0.0085 | 6.075 |
| 12 | 13312 | 0.007 | 2.2475 | 0.086 | -0.015 | 0.0087 | 7.061 |
| 16 | 17408 | 0.003 | 2.3854 | 0.026 | -0.0131 | 0.0046 | 4.887 |
| 20 | 21504 | 0.007 | 2.5699 | 0.098 | -0.0119 | 0.0038 | 4.498 |
| 24 | 25600 | 0.005 | 2.3983 | 0.058 | -0.0117 | 0.0016 | 4.216 |
| 28 | 29696 | 0.005 | 2.6025 | 0.056 | -0.0123 | 0.0068 | 4.571 |
| 32 | 33792 | 0.002 | 2.6684 | 0.006 | -0.0075 | 0.0052 | 3.996 |
| 36 | 37888 | 0.004 | 2.6275 | 0.034 | -0.0121 | 0.0028 | 3.224 |
| 40 | 41984 | 0.005 | 2.6161 | 0.069 | -0.0152 | 0.0072 | 4.716 |
| 44 | 46080 | 0.007 | 2.627 | 0.083 | -0.0144 | 0.0037 | 3.742 |
| 48 | 50000 | 0.003 | 2.6557 | 0.034 | 2.4088e-04 | 0.0064 | 5.073 |

- critic 预热：200 步，value loss 0.0013 → 0.0017，
  预热前后 actor logits 逐比特不变 **true**。
- 奖励分量合计（全部 episode）：{}。

## 3 R0 与 R2 的 dev 局对比（两种口径；R0 20 局，R2 20 局）

- **R0** = A0 原样（`96cb219e…`），冻结策略、只评测；
- **R2** = 上面这条 50k 帧真跑训出来的 checkpoint；
- 两者都用**同一个冻结 evaluator**（`tools/rl/v410_student_eval.py`，经 `touhou_rl.v420.eval.build_legacy_eval_argv` 组装 argv），
  20 个 dev seed（1000–1019），每 5 个 seed 一条命令（单条 ≤10 min）。

### 3.1 R0（A0 原样）

| 口径 | 过关 | 帧均值 | 帧范围 | Wilson 95% | invalid | 结束原因 |
|---|---|---|---|---|---|---|
| `diag_first_death` | 0/20 | 1636.8 | 1397–2185 | 0–0.1611 | 0 | {"death_stop": 20} |
| `s1n` | 0/20 | 3011.5 | 2137–4964 | 0–0.1611 | 0 | {"game_over": 20} |

### 3.2 R2（50k 帧训练后）

| 口径 | 过关 | 帧均值 | 帧范围 | Wilson 95% | invalid | 结束原因 |
|---|---|---|---|---|---|---|
| `diag_first_death` | 0/20 | 758.2 | 755–794 | 0–0.1611 | 0 | {"death_stop": 20} |
| `s1n` | 0/20 | 1857.1 | 1717–2126 | 0–0.1611 | 0 | {"game_over": 20} |

### 3.3 配对（同 seed）

DIAG-FIRST-DEATH（首死即停）：

| seed | R0 帧 | R2 帧 | Δ | R0 clear | R2 clear |
|---:|---:|---:|---:|---|---|
| 1000 | 1409 | 755 | -654 | false | false |
| 1001 | 1728 | 755 | -973 | false | false |
| 1002 | 1499 | 794 | -705 | false | false |
| 1003 | 1557 | 755 | -802 | false | false |
| 1004 | 1410 | 757 | -653 | false | false |
| 1005 | 1605 | 755 | -850 | false | false |
| 1006 | 1397 | 757 | -640 | false | false |
| 1007 | 1408 | 755 | -653 | false | false |
| 1008 | 1431 | 755 | -676 | false | false |
| 1009 | 1492 | 757 | -735 | false | false |
| 1010 | 1588 | 757 | -831 | false | false |
| 1011 | 1413 | 757 | -656 | false | false |
| 1012 | 1897 | 757 | -1140 | false | false |
| 1013 | 1640 | 762 | -878 | false | false |
| 1014 | 2138 | 757 | -1381 | false | false |
| 1015 | 2185 | 757 | -1428 | false | false |
| 1016 | 1567 | 755 | -812 | false | false |
| 1017 | 1494 | 757 | -737 | false | false |
| 1018 | 2085 | 755 | -1330 | false | false |
| 1019 | 1793 | 755 | -1038 | false | false |

标准 S1N（允许残机继续到 game_over）：

| seed | R0 帧 | R2 帧 | Δ | R0 clear | R2 clear |
|---:|---:|---:|---:|---|---|
| 1000 | 2137 | 1720 | -417 | false | false |
| 1001 | 3430 | 1723 | -1707 | false | false |
| 1002 | 2185 | 1815 | -370 | false | false |
| 1003 | 2731 | 1815 | -916 | false | false |
| 1004 | 2456 | 1880 | -576 | false | false |
| 1005 | 2579 | 1770 | -809 | false | false |
| 1006 | 2572 | 1770 | -802 | false | false |
| 1007 | 3280 | 1867 | -1413 | false | false |
| 1008 | 3181 | 1889 | -1292 | false | false |
| 1009 | 3940 | 1816 | -2124 | false | false |
| 1010 | 2528 | 1717 | -811 | false | false |
| 1011 | 2540 | 1948 | -592 | false | false |
| 1012 | 3080 | 2126 | -954 | false | false |
| 1013 | 3034 | 2020 | -1014 | false | false |
| 1014 | 3078 | 1888 | -1190 | false | false |
| 1015 | 4964 | 1891 | -3073 | false | false |
| 1016 | 2886 | 1814 | -1072 | false | false |
| 1017 | 3472 | 2036 | -1436 | false | false |
| 1018 | 3092 | 1825 | -1267 | false | false |
| 1019 | 3065 | 1812 | -1253 | false | false |

配对差汇总：DIAG 均值 -878.6 帧、
正差 0/20；
S1N 均值 -1154.4 帧、
正差 0/20。
两个口径都 **0 次过关**，所以这些数字只是存活长度差异，**不是**能力结论。
R2 在两个口径下都**明显更短**（DIAG −878.6 帧 ≈ −54%、S1N −1154.4 帧 ≈ −38%，20/20 配对为负）：
这与 §2.1 的接线缺陷一致 —— 49 次更新只看到进度项、基线与终局信号都不在回路里，
于是把冻结的 BC 策略带偏了。它**不能**读成"RL 让策略变差"，只能读成
"这次接线的训练没有可用的学习信号，重跑前先按 §6 修好折扣视界与 episode 连续性"。

## 4 3 seed × 1M 帧的成本外推（串行墙钟）

以本档实测吞吐与更新开销为基准（争用下界；另给干净主机口径）：

| 量 | 实测基准 |
|---|---|
| 每帧采集耗时 | 0.0327 s |
| 每次更新耗时 | 2.3021 s |
| 每 update 帧数 | 1024 |
| 每 episode 平均帧数 | 2186 |
| 每 episode 引擎重启 | 6.5 s（单独计，不藏进吞吐） |

| seed | 采集 | 更新 | 重启 | 单 seed 1M 墙钟 | 更新数 |
|---:|---:|---:|---:|---:|---:|
| 0 | 32661.5 | 2248.1 | 2973.5 | **37883 s** (10.523 h) | 977 |
| 1 | 32661.5 | 2248.1 | 2973.5 | **37883 s** (10.523 h) | 977 |
| 2 | 32661.5 | 2248.1 | 2973.5 | **37883 s** (10.523 h) | 977 |

**3 seed × 1M 串行墙钟 ≈ 113649 s = 31.57 小时**（争用下界，实测 load ≈26）。
干净主机口径（同一档审计里 13,195 帧真实过关回放实测 86.8 fps）：
**3 seed × 1M ≈ 50227 s = 13.95 小时**。
5M 后续档（未排期、未编预算，按干净口径）：单 seed ≈ 23.2 h、3 seed ≈ 69.8 h。

> 外推用**实测**速率，并把更新与重启单列：one engine start per episode; the same 5-7 s the frozen ruler pays, counted separately so the training throughput is not flattered。
> 本档真跑时主机 load average 约 26（另一个采集任务占 8 核、多个引擎同时在跑），
> 所以 pilot 的 fps 是下界，干净口径那一行才是排期应该用的数。

## 5 这一档能主张什么

- 2 帧连接层在真实引擎上闭环跑通：观测 → 策略 → 2 帧动作 → 引擎真值奖励 → GAE → PPO → 新 checkpoint，
  且新 checkpoint 能被**冻结 evaluator** 读回并逐比特复现（§1）。
- 统计健康：KL、熵、clip fraction、value loss、梯度范数都有实测值且无 nan/inf；
  更新次数与帧数账目自洽。
- **不能**主张能力：50k 帧、1 seed、0 次过关。R0 = 0/20，R2 = 0/20（两种口径相同）。
- **也不能**把这条真跑当作"R2 有效/无效"的判据：[`RL_AUDIT.md`](RL_AUDIT.md) 的
  `reward_discount_horizon` **failed**（γ^T ≈ 6.5e-04，终局过关奖金比一次死亡罚分小 400 倍），
  所以这次训练是在"终局奖励几乎不可见"的奖励下跑的。它证明的是**回路与统计**，
  不是 RL 能不能改善。这一点必须先修，再谈 1M。

## 6 G420-R 据实状态：**not_passed**

| 缺口 | 现状 |
|---|---|
| **折扣视界与任务长度冲突** | **审计 failed 项**：γ^T at clear = 6.5e-04，重启前必须改 τ（按 220 s 任务长度，τ ≈ 300–400 s 才能让终局奖金保留 ~0.45）或改奖励结构 |
| 3 个训练 seed × 1M 帧 | **未跑**（本档只跑 1 seed × 50k） |
| 5M/run 后续档 | **未排期、未编预算** |
| 至少 2/3 run 的配对游戏指标为正 | **未测**（只有 1 个 run） |
| 过面（能力）指标 | **两臂两口径全 0**（R0 0/20、R2 0/20） |
| **训练循环在 rollout 边界重启了 episode** | **已修未重跑**：50k 真跑的 episode 数 = 0（见 §2.1），修正后的 50k 需要重跑 |
| 特权 critic | **未测**（本档 critic 与 actor 共享 trunk，非特权网络） |
| 严格像素自我改进（R1） | **未做**：R2 用引擎真值奖励，明确不属于 R1 |
| 更大 rollout / 稳定性扫描 | **未做**：512 决策/更新是首档起点，未按显存与稳定性调整过 |

## 7 边界

- 50k 帧的结论**不能**写成 1M 档结论；1 个 seed **不能**当 3 个 seed。
- R2 的成功/失败都不是"RL 不可行/可行"的结论，也不是 R1 的证据。
- 训练前后都是 0/20 过关：这是**有效负结果**，不是测量失败。
