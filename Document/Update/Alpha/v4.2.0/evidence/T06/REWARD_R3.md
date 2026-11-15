# T06 / WP420-05 第三档：奖励版本 `v420-reward-r3/1`（让奖励能"看见"首死存活）

对象：v4.2.0 `T06`（WP420-05）第三档。**本档是一次自立实验（self-standing new reward
version），不是 P4 授权的实验。** P4 的前提是 `diag < 1473`，而终稿
[`r2_kl_p3.json`](r2_kl_p3.json) 的实测是 **1512.1 帧（n=20，落在预注册带 [1473,1800] 内）**，
所以 `p4.branch_taken = ` **`not_authorized_premise_false`**；P3 本身是
diag hold / s1n `not_run` ⇒ **incomplete**。本档因此**不引用 P4 的授权**，只引用
[`R2_COLLAPSE_DIAGNOSIS.md`](R2_COLLAPSE_DIAGNOSIS.md) 的 H3 量化（奖励看不见首死存活）与
WP420-05 §5 的奖励约束，自行新建一个奖励版本。

**这是新建一个奖励版本，不是对 r2 的微调。** r2 的六个系数里**五个逐字节保留**
（`stage_clear_bonus 1.0`、`death_penalty 0.25`、`bomb_penalty 0.1`、`progress_total 0.5`、
`progress_score_full 2631430.0`），第六个 `survive_per_decision` 正是本档要动的那一项
（`0.0` → `ε = 1.2e-05`，且被 §1.3 的断言钉住）；另外新增两个**有界**的首命项
（`first_life_per_decision = δ`、`first_life_cap = 0.25`）。

> R3 uses engine-truth reward and termination, so it is a training-time privileged
> diagnostic: it is **NOT** strict pixel self-improvement (R1) and no result here may
> be reported as R1.

产物：

| 文件 | 内容 |
|---|---|
| [`reward_r3.json`](reward_r3.json) | 本档全部数字（审计 18 项、50k 统计、两口径 20 dev、四臂对照、预测判定） |
| `configs/v420/rl-engine-r3-aloudit.json` | 审计配置（任务白名单授权的文件名） |
| `configs/v420/rl-engine-r3-pilot.json` | 50k 训练配置 |
| `.tmp_v420_data/T06/r3/audit-r3-c.json` | **18 项全 pass 的审计 payload**（`audit-r3-b.json` 保留为 17/18 的失败尝试） |
| `.tmp_v420_data/T06/r3/pilot-r3-a.json` | 50k 原始 payload |
| `.tmp_v420_data/T06/dev-r3/R3-*.json` | 两口径各 4 个 5-seed 批次 |
| `.tmp_v420_r3/*.sh`、`dev_batch.py` | 本次运行的脚本与逐条 argv/墙钟记录（`dev_runs.jsonl`） |

## 0 一句话结论

r2 的盲区是**结构性的**：24/24 局以 `game_over` 结束 ⇒ 死亡项每局恒为 −0.75、
`var(death)=0`、`corr(return, 首死帧)=0.3545`。r3 在同一批真实回放上的
`corr(return, 首死帧)` 是 **0.9485**（Pearson）/ 0.9339（Spearman）/ 0.8986（成对一致率），
而 r2 控制表在同一仪器上是 **0.3545 / 0.3600 / 0.6196** —— 新判据 `reward_first_death_visibility`
在 r3 上 pass、在 r2 上 fail，两者都在跑之前就算完了。

**运行结论（三次运行全部完成）**：审计 **18/18 pass**（含新增的第 15 项）；50k 真跑
**22 局 / 49 更新 / 22 次终局奖励进入 PPO**（0 过关）；两口径 20 dev 全部跑完、0 invalid：
**diag 1589.2 帧**、**s1n 2891.6 帧**。跑前写死的预测（两口径都 ≥ R0：1636.8 / 3011.5）
**fail**，两条腿分别差 **−47.6 帧（−2.91%）** 与 **−119.9 帧（−3.98%）**。
四臂对照：R0 1636.8 / r2 无锚 889.35 / r2+KL 锚 1512.1 / **r3 1589.2**（diag），
s1n 3011.5 / 1906.4 / 2626.0 / **2891.6**。也就是：r3 把 r2→R0 缺口闭合了
**93.6%（diag）/ 89.2%（s1n）**，但仍差 3–4% 没到线。

## 1 r3 奖励表

### 1.1 逐项：单位、事件、计次规则

| 项 | r3 值 | 单位 | 触发事件 | 计次规则 | 版本 |
|---|---:|---|---|---|---|
| `stage_clear_bonus` | 1.0 | reward / 首面完成 | 引擎 stage index 前进（`stage_clear`） | **每局一次**（主要正事件，未改） | r2=r3 |
| `death_penalty` | 0.25 | reward / 每次引擎死亡事件 | `globals.deaths` 增加 | 每次死亡一次；首死**不**终止 episode（未改） | r2=r3 |
| `bomb_penalty` | 0.1 | reward / 每颗 bomb | `globals.bombsRemaining` 减少 | 每颗一次（未改） | r2=r3 |
| `progress_total` | 0.5 | reward / 走完首面的分数进度 | 分数增加，按 `progress_score_full` 归一 | 累积、**硬上限 0.5**（未改） | r2=r3 |
| `progress_score_full` | 2631430.0 | 分数点 | 真实首面过关的终局分数 | 归一化分母（未改） | r2=r3 |
| `survive_per_decision` | **1.2e-05** | reward / 决策（2 帧节拍） | **每一个决策**（引擎活着） | 每决策一次，无上限但**被断言约束**：`ε·T_max ≤ 0.25·progress_total` | r2: 0.0 → r3: ε |
| `first_life_per_decision` | **3.7890269778720825e-05** | reward / 决策（**第一条命**） | `globals.deaths == 0` 时的每个决策 | 每决策一次，**每局硬上限 `first_life_cap`**；δ 由 `first_life_cap / 6598` 推导，不许手调 | r3 新增 |
| `first_life_cap` | **0.25** | reward（该集的封顶值） | 不直接发放，只封 `first_life` | 每局最多一次 | r3 新增 |

两条计次规则的细节（写进代码注释与 `REWARD_UNITS`）：

1. **首死那一决策仍然算第一条命**：`first_life_delta()` 读的是"本决策死亡事件发生**之前**的
   `deaths` 计数"。理由：动作是在活着的时候选的，奖励必须对应该决策，而不是对应该决策的后果。
   因此 `first_life` 的决策数 = 首次死亡所在决策的**下标 + 1**。
2. **`survive` 与死亡无关**：它按决策发放，包含死亡之后的所有决策，所以它是"这一局活了多久"的
   一阶信号；`first_life` 是"第一条命活了多久"的信号——后者才是 `diag_first_death` 尺子量的东西。

### 1.2 为什么这能"看见"首死存活（结构论证）

r2 里 `return = progress(score) − 0.25·deaths + 1.0·clear`。在 24/24 都 `game_over` 的那批真实
轨迹上 `deaths ≡ 3`，所以 `death` 项是常数、`clear` 是 0，**回报的排序完全由 progress 决定**；
而 progress 只看分数，不看时间。H3 的实测是这一点的直接后果：
`corr(return, 首死帧)=0.3545` 对 `corr(return, 总帧)=0.7049`。

r3 里同一个 episode 的回报多了两项：

```
return = progress(score) − 0.25·deaths + 1.0·clear
       + ε·decisions                              # 活了多久
       + δ·min(decisions_until_first_death, 6598)  # 第一条命活了多久
```

`sig(decisions_until_first_death)` 项是**首死帧的单调函数**，而 `decisions` 与首死帧的相关性
（真实 24 局：`corr(decisions, first_death_frame) = 0.7213`）远高于 1，所以回报对首死帧的排序
从 0.3545 抬到 0.9485。这不是"加大某个常数的量级"，而是**加了一个以前根本不存在的自变量**。

### 1.3 有界断言（逐条复算）

配置里冻结、`resolve_reward()` 每次载入时**重算**并 fail-closed 的三条：

```
T_max      = rollout.episode_max_frames / DECISION_FRAMES = 19887 / 2 = 9943.5 决策
ε·T_max    = 1.2e-05 × 9943.5                = 0.119322
0.25·progress_total                          = 0.125
断言 ①  ε·T_max ≤ 0.25·progress_total        : 0.119322 ≤ 0.125        PASS (margin 0.005678)
断言 ②  ε·T_max + first_life_cap ≤ progress_total
       : 0.119322 + 0.25 = 0.369322 ≤ 0.5                              PASS (73.9% of budget)
断言 ③  first_life_per_decision × CLEAR_DECISIONS_MEASURED = first_life_cap
       : 3.7890269778720825e-05 × 6598 = 0.25                          PASS (residual 0.0)
```

再加两条"不得压过任务目标"的：

```
max_non_clear_return = progress_total + ε/(1-γ) + first_life_cap
                     = 0.5 + 1.2e-05×6000.5 + 0.25 = 0.822006
clear 的一局上界     = stage_clear_bonus + progress_total = 1.5
                     ⇒ 0.822006 < 1.5（必然）
first_life_cap 0.25  < stage_clear_bonus 1.0（必然）
```

即：**所有"存活形状"的奖励加在一起（0.369322）还不到半个进度预算**，而且非过关轨迹的总上界
（0.822006）严格低于一次过关（1.5）。裸生存常数压不过任务目标这件事，在 r3 里是**算术**，
不是承诺。

### 1.4 只改了奖励这一个因素

`configs/v420/rl-engine-r3-pilot.json` 对 `configs/v420/rl-engine-pilot-kl.json` 的扁平化 diff
（脚本 `.tmp_v420_r3/diff_configs.py`，可复跑）：

- **协议值一个没动**：`ppo.clip 0.2`、`grad_clip 0.5`、`actor_lr 3e-05`、`critic_lr 3e-04`、
  `entropy_coef 0.001`、`value_coef 0.5`、`epochs 2`、`minibatch 128`、
  `advantage_normalisation true`、`critic_warmup_steps 200`、`temperature 8.0`、
  `rollout.decisions_per_update 512`、`episode_max_frames 19887`、`shoot_pulse_period 48`、
  `train_seeds [3000..3004]`、`pilot.frame_budget 50000`、`tau 200 / gamma 0.9998333… /
  lambda 0.95`、`kl_anchor.beta 1.0`、`kl_anchor.space T1`、`max_kl_nats 0.05`。
- **变的**：`reward.version`（r2→r3）、`reward.survive_per_decision`（0→1.2e-05）、新增
  `reward.first_life_per_decision/first_life_cap`、新增 `reward_criterion` 与
  `audit.first_death_reference` 两个**判据块**、`ppo.kl_anchor.gate_mode`（见 §4）。
- **纯记账**：`data_dir`（改成本档的私有 `cp -a` 副本）、`dev_eval.tag/runs_dir`、
  `pilot.checkpoint_tag`、以及若干说明文字。

## 2 新判据：`reward_first_death_visibility`（跑前冻结）

### 2.1 判据

在**冻结的真实轨迹集**上（r2 那次 50k 真跑的 24 局 `game_over` 记录，
`.tmp_v420_data/T06/pilot2.json`，sha256 `b217c4c0…ce5036e`；每局恰好 3 次死亡 ——
这正是 r2 失明的那个条件），把每局按候选奖励表重新打分，然后要求：

| 检查 | 门槛 | 含义 |
|---|---:|---|
| Pearson `corr(return, first_death_frame)` | ≥ **0.80** | 首死越晚，回报越高（线性） |
| Spearman `corr` | ≥ **0.80** | 同上（秩） |
| 成对一致率（later first death ⇒ higher return） | ≥ **0.75** | 不依赖线性假设 |
| 参照集局数 | ≥ 8 | 24 |
| 死亡项在本参照集上恒定 | true | 仪器有效性条件 |

评分函数是**精确的**，不是近似：`progress` / `bomb` / `stage_clear` 直接取记录值（r3 没有改这三项的
系数或归一化），`death = −0.25 × deaths`，两个 r3 新项按闭式加进去
（`ε·decisions` 与 `δ·min(decisions, ⌈first_death_frame/2⌉, cap/δ)`）。r2 控制表 = 同一个函数把
两个 r3 项关掉（`RewardSpec.as_r2_control()`）。

### 2.2 为什么 r2 会 fail（用它自己的实测数字）

| 表 | Pearson | Spearman | 成对一致率 | 判定 |
|---|---:|---:|---:|---|
| r2（控制） | **0.3545** | 0.3600 | 0.6196 | **fail** |
| r3 | **0.9485** | 0.9339 | 0.8986 | **pass** |

r2 的 0.3545 与 [`R2_COLLAPSE_DIAGNOSIS.md`](R2_COLLAPSE_DIAGNOSIS.md) §H3 的实测值**逐位一致**
（同一个参照集、同一个统计量），所以"r2 会 fail"不是本档的主张，而是既有证据的复算。
审计里这一项只有在**r3 pass 且 r2 控制 fail** 时才 pass，因此一个 r2 也能过的判据不会混进来。

### 2.3 判据评在"总和"而不是"折扣"上

参照集只有**每局聚合**（帧数、决策数、首死帧、当年的 reward_totals），没有逐决策奖励流，所以新判据
用 episode-sum 回报；折扣口径由四类真实轨迹那一项（`reward_ordering` / `reward_discount_horizon`）
负责，两者在审计里是**两个不同的 item**，不混。

## 3 四类真实轨迹（过关 / 早死 / 拖延 / 滥用 bomb）

四类**全部是真实引擎轨迹**，在 18 项审计里实测（`audit-r3-c.json#classes`）：
过关类是那条唯一的真实首面过关录制的回放，其余三类是本次新跑的真实 episode
（各自的私有 `cp -a` data 目录）。γ = 0.9998333472214507，G = Σγ^t r_t，**不做任何归一化**。

| 轨迹类 | 结束 | 帧数 | 决策 | **折扣回报 G** | 未折扣 Σr | stage_clear | progress | death | bomb | survive | first_life |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 过关 (clear) | `stage_clear` | 13,195 | 6,598 | **+0.723182** | 1.829176 | 1.0 | 0.5 | 0 | 0 | 0.079176 | **0.250000** |
| 早死 (early) | `first_death` | 1,370 | 685 | **−0.189780** | −0.214769 | 0 | 0.001056 | −0.25 | 0 | 0.008220 | 0.025955 |
| 拖延 (delay) | `game_over` | 1,503 | 752 | **−0.661556** | −0.726341 | 0 | 0.001184 | −0.75 | 0 | 0.009024 | 0.013451 |
| 滥用 bomb | `cap` | 2,000 | 1,000 | **−1.066487** | −1.162132 | 0 | 0.009082 | −0.5 | −0.7 | 0.012000 | 0.016785 |

排序检查（**未折扣**口径 —— 奖励表本身的性质，长度不会污染它）：

| 检查 | 结果 | 数值 |
|---|---|---|
| `clear_gt_delay_undiscounted` | pass | margin **2.555517** |
| `clear_gt_early_death_undiscounted` | pass | margin **2.043945** |
| `clear_gt_bomb_abuse_undiscounted` | pass | margin **2.991308** |
| `clear_actually_cleared` | pass | true |
| `non_clear_upper_bound_below_clear` | pass | 非过关上界 **0.822006** < 过关 1.829176 |
| `survive_constant_bounded` | pass | ε·T_max 0.119322 ≤ 0.125；合并 0.369322 ≤ 0.5；δ 推导 residual 0.0 |

折扣口径（`reward_discount_horizon`，6 条决定性检查）：

| 检查 | 结果 | 数值 |
|---|---|---|
| `clear_gt_delay_discounted` | pass | +0.723182 > −0.661556，margin **+1.384738** |
| `clear_gt_early_death_discounted` | pass | margin **+0.912962** |
| `clear_gt_bomb_abuse_discounted` | pass | margin **+1.789670** |
| `terminal_bonus_is_not_negligible` | pass | 1.0 × γ^T(6598)=**0.332982** > 0.25 |
| `clear_terminal_visibility_floor` | pass | 0.332982 ≥ **0.30** |
| `clear_vs_early_death_discounted_margin` | pass | \|clear\|/\|early\| = **3.8106** ≥ 1.5 |

> 与 r2 同口径对照（[`RL_AUDIT.md`](RL_AUDIT.md) §4.1）：r2 的三条折扣边际是
> +1.206600 / +0.745578 / +1.617223，\|clear\|/\|early\| = 2.357271。r3 的边际更大，原因是
> 过关轨迹额外拿了 `survive 0.079176 + first_life 0.25`，而早死类只拿了
> 0.008220 + 0.025955 —— **这正是"奖励能看见首死存活"在四类轨迹上的样子**：
> 早死与过关的距离从 r2 的 0.745578 拉开到 0.912962。

## 4 KL 锚的判据修订（**显式声明**）

保留不动：β = **1.0**（仍只能上调）、测量空间 **T=1**（未加温度）、`max_t KL_T1(A0‖π_t)` 每次更新
仍被测量、记录、写进 payload。

**改的只有一件事**：`max_t KL_T1 ≤ 0.05 nat` 从"通过条件"降为"**记录并只用于诊断**"。
这是**判据修订**，不是放宽容忍度，理由与原始数字一起冻结在配置的
`ppo.kl_anchor.criterion_revision` 里：

| 预测 | 内容 | 原始数字 | 判定 |
|---|---|---|---|
| **P1** | `max_t KL_T1(A0‖π_t) ≤ 0.05 nat`（50k 全程） | 观测 **0.8793413160101484** nat（阈值的 **17.59 倍**） | fail |
| **P2** | 同一批 2×256 槽：argmax 翻转率 ≤ 0.10；各动作平均 Δlogit 最大绝对值 ≤ 0.5 | 翻转率 **0.05092592592592592**（≤0.10）；末态 KL_T1 mean 0.016757 / median 0.002707 / max 0.319511 / sd 0.038054（432 行）；Δlogit 一腿**未记录**，不作主张 | flip 腿 hold |
| **P3** | 20 dev：diag 落在 1473–1800 且 s1n 落在 2710–3313 | **diag 1512.1 帧（n=20，1117–1751）⇒ hold**（+39.1 帧）；s1n **not_run**；整条判据 **incomplete** | incomplete |
| P4 | 裁决 | `r2_kl_p3.json#p4.branch_taken =` **`not_authorized_premise_false`**（P4 前提 `diag < 1473` 为假） | 未授权 |

> **更正一处早期读法**：本档开工时 `r2_kl_p3.json` 还是进行中的版本，里面 P3 记的是
> `1425.0 帧 / fail`，那是**只跑了 seed 1000 一局（n=1）** 的试验性聚合；终稿的 20 局均值是
> **1512.1**，P3 的 diag 腿是 **hold**。本档一律用终稿数字：**1512.1（n=20）**，不再引用 1425.0。

即：锚**确实把部署空间按住了**（P2 的翻转率 5.1%），却仍然在单点最坏值上超阈 17.6 倍。
0.05 nat 量的是"整个 run 里最坏的那**一个**状态"，不是策略；任何强到有用的锚都过不了它。
所以判据本身被认定为问题，改为记录；**β 与 T=1 一字未动**。
本档 50k 的 `max_t KL_T1` 仍会如实报出（§7），并且**不再决定 run 的退出码**。

> 这次降级是**本档自行声明的判据修订**，不是 P4 授权的一部分：P4 既然未授权
> （`not_authorized_premise_false`），本档就不拿 P4 当依据。

代码上：`config.ppo.kl_anchor.gate_mode ∈ {fail_closed, diagnostic_only}`，缺省
`fail_closed`（r2 的两个配置与既有测试的语义完全不变）；r3 的两个配置声明 `diagnostic_only`
并**必须**同时提供 `criterion_revision` 的数字，否则 `resolve_kl_anchor` 直接拒绝载入。
## 5 第一个可证伪预测（**跑前写死**）

在跑 50k 之前、在跑任何 dev 批次之前写死，写死的位置是 `reward_r3.json#prediction` 与本节：

> **R3-P1**：20 dev 局，`diag_first_death` 的 `frames_mean` **≥ 1636.8** 且 `s1n` 的
> `frames_mean` **≥ 3011.5**（即至少回到 R0 的两种口径水平）。
> 任一腿不成立即整条 fail；只跑完一条腿则为 `incomplete`，不外推。

锚点来源：R0 = T04 冻结记录（diag 1636.8 / s1n 3011.5，各 n=20，
`.tmp_v420_data/T06/dev/merged-R0.json`）。选"回到 R0"而不是"超过 R0"是刻意的：
r3 只改奖励，50k / 49 次更新不足以产生能力，所以能检验的只有"**奖励修好之后，
漂移与奖励盲区都不再把首死存活打下去**"。

同一次运行里还报三条读法（不是预测，是解释用的对照）：与 R0、与 r2 无锚（889.35 / 1906.4）、
与 r2+KL 锚（**1512.1**，n=20）。

## 6 审计逐项（18 项，全 pass）

命令：`xvfb-run -a -s "-screen 0 1024x768x24" python tools/rl/v420.py rl --config
configs/v420/rl-engine-r3-aloudit.json --out .tmp_v420_data/T06/r3/audit-r3-c.json --execute`
→ **`AUDIT_EXIT=0`**，`status = pass`，`failed = []`，墙钟 **555.7 s**（争用下界，当时的 load 在 45→4 之间）。

| # | 检查 | 状态 | 关键实测 |
|---:|---|---|---|
| 1 | `obs_contract` | pass | play [4,112,96] / hud [56,160] / hist [8,18]，历史只含**已执行**动作 |
| 2 | `initial_logits` | pass | float32 sha **`c8b6df33929365ae…60d85`** —— 与 r2 审计**逐位相同** |
| 3 | `argmax_behaviour` | pass | 9 个不同动作，最大占比 0.0483 ≤ 0.6 |
| 4 | `sample_behaviour` | pass | T=8.0，17/256 个动作，最大占比 0.1797 |
| 5 | `initial_policy_degeneracy` | pass | T=1 熵 0.413 nat（退化）→ T=8.0 后 2.3948 nat |
| 6 | `joint_log_prob` | pass | exp 行和 [0.9999998, 1.0000002]，记录值一致 |
| 7 | `ratio_zero_update` | pass | `max\|ratio−1\| = 9.536747712e-07` ≤ 1e-06（与 r2 审计的 9.536747712e-07 同值） |
| 8 | `gae` | pass | 256 transition / 1 segment，与独立实现最大差 3.4e-21 |
| 9 | `gamma_lambda` | pass | γ = 0.9998333472214507（复算 true），λ = 0.95，视界 6000.5 决策 |
| 10 | `truncation_bootstrap` | pass | 1 个 truncated transition，bootstrap 1 |
| 11 | `death_vs_game_over` | pass | 主 episode：game_over、2186 帧、首死帧 1370、之后又跑 408 决策 |
| 12 | `reward_units` | pass | 6 项齐备（含 `first_life`）；clear 分解 `{survive 0.079176, progress 0.5, stage_clear 1.0, first_life 0.250000}`；`survival_bound` 全 pass |
| 13 | `reward_ordering` | pass | clear > delay / early / bomb，未折扣边际 2.555517 / 2.043945 / 2.991308 |
| 14 | `reward_discount_horizon` | pass | γ^T = 0.332982；折扣边际 +1.384738 / +0.912962 / +1.789670；\|clear\|/\|early\| 3.8106 ≥ 1.5 |
| **15** | **`reward_first_death_visibility`** | **pass** | **r3: Pearson 0.948483 / Spearman 0.933913 / 一致率 0.898551（276 对中 248 对）；r2 控制: 0.354535 / 0.360000 / 0.619565 → fail**；参照集 24 局，sha 校验 true |
| 16 | `critic_warmup` | pass | 预热前后 actor logits 逐位不变 true |
| 17 | `bomb_channel_zero` | pass | bomb 常量 0，越界调用被拒 |
| 18 | `forward_equivalence` | pass | 32 行，最大差 0，逐位一致 true |

**pass 18 / 18，fail 0。** 第 15 项是本档新增的那一条，它的 pass 条件是
"**r3 达标且 r2 控制表不达标**"，两半都在这次 payload 里（`observed` 与 `r2_control`）。

两次**被保留的失败尝试**（不删除、不重写）：

| 尝试 | 线程 | 结果 | 原因 |
|---|---|---|---|
| `a` | 默认 8 | 中止（28 min 只跑到第 3 个 episode） | 争用下 8 线程自旋，前向 952 ms |
| `b` | 1（`OMP_NUM_THREADS=1`） | **17/18**，`failed = ["ratio_zero_update"]` | `max\|ratio−1\| = 1.192e-06` > 冻结容差 1e-06；单线程改变了 float32 归约顺序（`initial_logits` sha `ff11ac0a…` ≠ `c8b6df33…`），容差没动，改用默认线程数复算 |
| `c` | 默认 8 + `OMP_WAIT_POLICY=PASSIVE` | **18/18 pass** | 数值与 r2 审计逐位一致，只关掉自旋 |

## 7 50k 真跑（单实例，KL 锚保留）

命令：`xvfb-run -a -s "-screen 0 1024x768x24" python tools/rl/v420.py rl --config
configs/v420/rl-engine-r3-pilot.json --out .tmp_v420_data/T06/r3/pilot-r3-a.json --execute`
→ **`PILOT_EXIT=0`**。

| 项 | 值 |
|---|---|
| 帧预算 / 实跑 | 50,000 / **50,000** |
| 决策数 | 25,005 |
| **episode 数** | **22**（全部 `game_over`，0 次 `stage_clear`） |
| **终局奖励进入 PPO** | **22**（`terminated_transitions = 22`，`truncated_transitions = 0`） |
| 更新数 | **49** |
| 墙钟 / 采集 / 更新 | **1012.4 s** / 817.7 s / 68.7 s |
| 吞吐 | **49.39 fps**、24.70 决策/s（争用下界，运行期间 load 4.4–7.1） |
| 每局帧数 | 均值 **2189.55**（sd 255.9，1798–2919）；r2 是 2047.17（1415–2950） |
| 每局回报 | 均值 **−0.710394**，**sd 0.0050217**；r2 是 −0.747441，sd 0.0014040（**sd 提高 3.58 倍**） |
| 奖励分量合计 | `death −16.5`（66 次死亡）、`progress 0.070547`、`survive 0.289080`、**`first_life 0.511708`**、`stage_clear 0.0` |
| 训练后 checkpoint | `T06/checkpoints/r3_seed0_50000f.pt` sha **`83f1ac44…4b4685`**；critic sidecar `51d741e5…f0921d` |
| 冻结评估器回读 | loaded=true、arm `B0`、参数 731802、**logits 最大差 0**、逐位一致 true |

统计（49 次更新）：

| 量 | 均值 (sd) | 首 → 末 |
|---|---|---|
| approx KL（T=8） | 0.001298 (0.000197) | 0.001303 → 0.001003 |
| 熵（T=8） | 2.4945 (0.0516) | 2.4307 → 2.5758 |
| clip fraction | 0.00337 (0.00318) | 0.00195 → 0.0 |
| value loss | 0.008157 (0.002022) | 0.010413 → 0.010590 |
| 梯度范数（裁剪前） | 9.046 (1.537) | 8.628 → 9.284 |
| actor 参数每次更新最大变化 | 2.3829e-04 (1.64e-06) | 2.3830e-04 → 2.3532e-04 |

**critic 误差与回报尺度的比**：`sqrt(value_loss) = 0.09032`，是回报 sd（0.005022）的
**18.0 倍**（r2 是 0.08832 / 0.001404 = **62.9 倍**）。50k 的预算仍然不够把 critic 拟合好，
但**信噪比改善了 3.5 倍** —— 这是奖励方差变大（而尺度不变）的直接后果。

### 7.1 训练过程中的首死（本档自己的四等分）

| 区间 | 局数 | 帧数 | **首死帧** | 回报 | first_life | survive | progress |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 6 | 2178.5 | **1289.5** | −0.709250 | 0.024433 | 0.013074 | 0.003244 |
| Q2 | 5 | 2097.6 | 1197.6 | −0.712255 | 0.022696 | 0.012588 | 0.002460 |
| Q3 | 5 | 2279.0 | 1229.8 | −0.708638 | 0.023310 | 0.013678 | 0.004375 |
| Q4 | 6 | 2202.7 | **1188.17** | −0.711450 | 0.022513 | 0.013218 | 0.002818 |
| 变化 | — | +1.11% | **−7.86%** | −0.3102% | −7.86% | +1.10% | −13.1% |

对照 r2 的同一张表（[`R2_COLLAPSE_DIAGNOSIS.md`](R2_COLLAPSE_DIAGNOSIS.md) §H3）：
首死 **−29.29%**、回报 **+0.241%**，比值 `|Δreturn%| / |Δfirst_death%| = 0.0082`。
r3 的同一是 **0.0395**（4.8 倍），且首死只掉 7.86% 而不是 29.29%。
另外 r3 训练期的首死均值是 **1227.41**（22 局，806–1692），r2 那 24 局按四等分是 ~1002.3。

> 注意：这一节是**训练期特权诊断**（奖励用引擎真值），不是能力结论；单 seed、50k、0 过关。
## 8 两口径 20 dev 与四臂对照

### 8.1 运行方式（串行，私有 `cp -a` data 目录）

全部通过 `touhou_rl.v420.rl.dev_eval_invocation()` 组装 argv（同一个 fail-closed 构造器，
只用一份私有的 `cp -a` data 目录：`.tmp_v420_data/T06/engine-data-r3-dev/{diag,s1n}`，
每个口径一份，绝不与训练/审计共用），**一条命令 = 1 个 ruler × 5 个 dev seed**，
一条跑完再跑下一条（`.tmp_v420_r3/dev_batch.py` 的逐条 argv/墙钟记录在
`.tmp_v420_r3/dev_runs.jsonl`）。checkpoint = `r3_seed0_50000f.pt` sha `83f1ac44…4b4685`。

| 批次 | 口径 | 墙钟 | 退出码 | 帧数 |
|---|---|---:|---:|---|
| 1000–1004 | diag | 142.6 s | 0 | 1788, 816, 1929, 1494, 1410 |
| 1005–1009 | diag | 156.1 s | 0 | 1492, 1945, 1546, 1816, 1489 |
| 1010–1014 | diag | 136.0 s | 0 | 1363, 1485, 1500, 1568, 1971 |
| 1015–1019 | diag | 133.2 s | 0 | 1295, 2392, 1495, 1544, 1446 |
| 1000–1004 | s1n | 214.0 s | 0 | 3022, 1920, 4701, 2407, 2210 |
| 1005–1009 | s1n | 214.4 s | 0 | 2480, 3418, 2889, 3509, 2373 |
| 1010–1014 | s1n | 354.5 s | 0 | 2180, 3578, 3391, 2537, 3270 |
| 1015–1019 | s1n | 306.3 s | 1* | 2704, 3566, 2258, 3007, 2412 |

\* 最后一条的 **评估器** 写出的 summary 是完整的（文件内 `exit_code: 0`，5 局全
`game_over`、各 3 次死亡、0 invalid），非零来自 `xvfb-run` 包装层；provenance 行如实记
`rc=1`，帧数从 summary 文件本身读出。逐局核对：5 局全 `game_over`、各 3 次死亡、
`error: None`、`invalid` 0、文件内 `exit_code: 0`。**没有任何一局失败**。

单条最长 **354.5 s**，全部 ≤ 10 分钟。dev 批次运行时 load 已降到 **1.4–2.5**，
所以这一段的墙钟**不是**争用下界（审计与 50k 那两段才是，见 §9）。

### 8.2 聚合（每口径 n=20）

| 口径 | 过关 | `frames_mean` | median | min–max | sd | `stop_reason` | invalid | errors |
|---|---:|---:|---:|---|---:|---|---:|---:|
| `diag_first_death` | **0/20** | **1589.2** | 1500 | 816–2392 | 322.2 | `{"death_stop": 20}` | 0 | 0 |
| `s1n` | **0/20** | **2891.6** | 2889 | 1920–4701 | 675.2 | `{"game_over": 20}` | 0 | 0 |

Wilson 95%（0/20）：[0, 0.1611]（两口径相同）。

### 8.3 四臂对照

| 臂 | checkpoint sha | diag 均值 | s1n 均值 | 过关 | n | 来源 |
|---|---|---:|---:|---|---:|---|
| **R0**（A0，无训练） | `96cb219e…` | **1636.8**（1397–2185） | **3011.5**（2137–4964） | 0/20, 0/20 | 20 | T04 冻结记录 → `T06/dev/merged-R0.json` |
| **R2 无锚** | `aefcc083…` | **889.35**（733–1365） | **1906.4**（1636–2337） | 0/20, 0/20 | 20 | `T06/dev2/merged-R2.json` |
| **R2 + KL 锚** | `07fa0c69…` | **1512.1**（1117–1751） | **2626.0**（2070–3479） | 0/20, 0/20 | 20 | `T06/dev-kl/R2kl-*.json`（P3 任务的流，本档只读；每批次 sha 见 `kl_arm_reference.json`） |
| **R3（本档）** | `83f1ac44…` | **1589.2**（816–2392） | **2891.6**（1920–4701） | 0/20, 0/20 | 20 | `T06/dev-r3/R3-*.json` |

增量读法：

| 比较 | diag | s1n |
|---|---:|---:|
| R3 − R2 无锚 | **+699.85 帧**（+78.7%） | **+985.2 帧**（+51.7%） |
| R3 − R2+KL | **+77.1 帧**（+5.10%） | **+265.6 帧**（+10.11%） |
| R3 − R0 | −47.6 帧（−2.91%） | −119.9 帧（−3.98%） |
| 闭合 R2→R0 缺口的比例 | **93.6%**（KL 锚单独是 83.3%） | **89.2%**（KL 锚单独是 65.1%） |

配对（同 seed，n=20）：

| 配对 | 口径 | Δ 均值 | 正 / 负 / 零 | 范围 |
|---|---|---:|---|---|
| R3 − R0 | diag | **−47.6** | 8 / 11 / 1 | −912 … +825 |
| R3 − R0 | s1n | **−119.9** | 8 / 12 / 0 | −2260 … +2516 |
| R3 − R2 无锚 | diag | **+699.85** | **20 / 0 / 0** | +53 … +1433 |
| R3 − R2 无锚 | s1n | **+985.2** | 19 / 1 / 0 | −79 … +2599 |

逐 seed（diag，R0 → R3）：1000 1409→1788、1001 1728→816、1002 1499→1929、1003 1557→1494、
1004 1410→1410、1005 1605→1492、1006 1397→1945、1007 1408→1546、1008 1431→1816、
1009 1492→1489、1010 1588→1363、1011 1413→1485、1012 1897→1500、1013 1640→1568、
1014 2138→1971、1015 2185→1295、1016 1567→2392、1017 1494→1495、1018 2085→1544、1019 1793→1446。

### 8.4 预测判定：**fail**（两腿都没到 R0）

| 腿 | 门槛 | 实测 | 偏离 | 判定 |
|---|---:|---:|---:|---|
| `diag_first_death` ≥ 1636.8 | 1636.8 | **1589.2** | **−47.6 帧（−2.91%）** | **fail** |
| `s1n` ≥ 3011.5 | 3011.5 | **2891.6** | **−119.9 帧（−3.98%）** | **fail** |
| 整条 R3-P1（两腿都成立才 hold） | — | — | — | **fail** |

**据实结论**：r3 把两种口径都**大幅**从 r2 无锚拉回来（+78.7% / +51.7%），也**在 KL 锚之上
继续前进**（+5.10% / +10.11%），但**没有达到**跑前写死的"回到 R0"这条线，两腿各差
2.91% / 3.98%。这是一次**可证伪预测的失败**，不是"接近通过"。不过它同时给出了方向一致的证据：
在同一个 50k / 49 更新预算里，**奖励可见性**（r3）是继 KL 锚之后**第二个**能把首死存活往上推的
单因素，且它的增量（+77.1 / +265.6）比 KL 锚当年补回来的量（+622.75 / +719.6）小，但方向相同。
两臂都是 **0/20 过关**，所以这里没有任何能力结论。

## 9 并发与墙钟口径

这一节的数字都是**争用条件下观测值**，不是吞吐能力结论。

### 9.1 实测并发

本档在同一台 20 vCPU 主机上与另外三个任务并行（**不是本档起的**，本档没杀任何进程）：
T05 D3 采集（`rl v420.py collect`）、P3 任务的两条 20-dev 流、KL 任务的两条 20-dev 流。
本档自身**任何时刻最多 1 个重进程**（审计 → 50k → dev 批次严格串行）。

| 时间（UTC） | loadavg (1 min) | 同机在飞的引擎任务 | 本档进程 |
|---|---:|---:|---|
| 19:34（审计 a 启动） | **80.6** | 6 | 审计尝试 a（8 线程，已中止） |
| 20:05（审计 b 启动） | 66.8 | 5 | 审计尝试 b（1 线程，17/18 后失败） |
| 20:24（审计 c 启动） | 54.4 | 5 | 审计尝试 c（默认线程 + 被动等待） |
| 20:46（50k 启动） | 4.4 | 2 | 50k pilot |
| 21:05（diag 批启动） | 2.4 | 2 | dev 批次（串行） |
| 21:16（s1n 批运行中） | 1.4 | 2 | dev 批次（串行） |

同一台机器上另外三个任务的进程在 20:44 之后陆续结束，load 从 80 掉到 1–2；
本档的 **50k 与全部 dev 批次是在 load 1.4–7.1 下跑的**，只有**审计**跨过了高争用段
（45–80）。因此：审计的 555.7 s 是争用下界；50k 的 49.39 fps / 1012.4 s 与 dev 的
逐批墙钟**接近安静期**，但仍不是"独占主机的基准"。

### 9.2 torch 线程：为什么这是环境问题而不是协议问题

同一台机器上量同一个 `ProbeNet` 前向（`.tmp_v420_r3/_thread_probe.py`，b=1）：

| 线程数 | 每次前向 | 备注 |
|---:|---:|---|
| 1 | 6.180 ms | 争用最重时（load 80） |
| 2 | 15.758 ms | 同上 |
| 4 | 200.078 ms | 同上 |
| 8 | **951.594 ms** | 同上（torch 默认取 8） |
| 8 + `OMP_WAIT_POLICY=PASSIVE` | **2.934 ms** | load 45 时；同一线程数，关掉自旋等待 |

也就是说：**默认 8 线程在争用下把一次 3 ms 的前向变成 950 ms**（150 倍），
而 8 个这样的进程互相叠加就是 load 80 本身。关掉 OpenMP 自旋等待后，线程数不变、数值不变，
速度回到 3 ms。

**所以本档保留 torch 默认线程数（8）**，只设 `OMP_WAIT_POLICY=PASSIVE`、
`GOMP_SPINCOUNT=0`、`KMP_BLOCKTIME=0`。这一点是被审计自己**逼出来的**：

- 审计尝试 b 用 `OMP_NUM_THREADS=1`，17/18 项 pass，唯一失败的是 `ratio_zero_update`：
  `max|ratio−1| = 1.192e-06` 对冻结容差 **1e-06**（r2 审计当时实测 9.5367e-07，刚好在容差内）。
  单线程改变了 float32 卷积的归约顺序（`initial_logits` 的 float32 sha 从 r2 审计的
  `c8b6df33…` 变成 `ff11ac0a…`），于是这条**以 1e-6 为容差、恰好卡在浮点噪声上的**门就翻了。
- **本档没有改那个容差**（那是协议常量，不是本档可以动的东西），而是把线程数改回默认值、
  只关掉自旋等待，让数值环境与 r2 审计那次一致。
- 尝试 a/b/c 三次的日志全部保留在 `.tmp_v420_r3/logs/audit-{a,b,c}.log`，
  失败的 payload 也保留（`audit-r3-b.json`，`failed = ["ratio_zero_update"]`）。

### 9.3 哪些数字不能用

- 本档报的 **fps / 每帧墙钟 / 每批次墙钟** 按段区分：**审计**（555.7 s，跨越 load 45→4）是
  争用下界；**50k**（1012.4 s / 49.39 fps，load 4.4–7.1）与 **dev**（每批 133–355 s，
  load 1.4–2.5）接近安静期，但都不是"独占主机"的基准。安静期（load ≈ 26）r2 的 50k 实测是
  61.6 fps / 811.5 s，可作参照。
- 单条 dev 命令的 ≤10 分钟上限**这次全部满足**（最长 354.5 s = 5.9 min）；实际墙钟逐条记录在
  `.tmp_v420_r3/dev_runs.jsonl`（含 `loadavg_before`），报告里按批次如实列出，不用部分结果外推。
- 审计/训练/评测的**语义**不受线程设置影响：奖励、PPO、折扣、rollout 长度、温度一个都没动；
  线程数只影响浮点归约顺序与墙钟。
## 10 边界与据实的下一步

### 10.1 不能主张什么

1. **不能说 r3 达标**：预测 fail（diag −47.6、s1n −119.9）。也不能说"等于 R0"——
   预注册规则是 `≥`，不是 ±10% 带。
2. **不能说过关**：r3 与所有参考臂都是 0/20（两口径）。
3. **不能说 1M 档行为**：单 seed、50k、49 更新；不外推到 1M / 3 seed。
4. **不能把 r3 说成 R1**：奖励用引擎真值（死亡/分数/炸弹/stage），是训练期特权诊断。
5. **不能把 dev 的逐 seed 差异分解成"奖励造成的"**：本档只有一条 r3 臂与 4 条历史臂，
   没有同一 50k run 的奖励消融（例如"r3 但 ε=δ=0"同轮跑）。
6. **不能把墙钟/fps 当吞吐能力**：审计与 50k 是在 load 45–80（审计早段）与 4.4–7.1（50k）
   下测的；dev 批次是在 load 1.4–2.5 下测的。三段不同，不能互相折算。
7. **不能引用 P4 的授权**：P4 的 `branch_taken = not_authorized_premise_false`。

### 10.2 据实的下一步（按证据强度排序）

1. **先把预算这一维单独测一遍**：同一个 r3 奖励表、同一套 KL 锚，跑 **200k 帧**
   （[`RL_PILOT2.md`](RL_PILOT2.md) §6 已经建议过"1M 之前先放一个 200k 检查点"）。
   理由是可测的：r3 已经把 `corr(return, 首死帧)` 做到 0.9485、把训练四等分的首死塌陷从
   −29.29% 降到 −7.86%、把 critic RMSE/回报 sd 从 62.9× 降到 18.0×，
   却仍然离 R0 差 3–4%；现在唯一没被检验的变量是**这 49 次更新够不够把排序信号换成帧数**。
   若 200k 仍不到 R0，那么"奖励可见性"就不是这条差距的主因，需要换问题。
2. **不要用调 ε/δ 去追那 47.6 帧**。ε 的上界是断言 `ε·T_max ≤ 0.25·progress_total`，
   δ 由 `first_life_cap / 6598` 推导；在看到 dev 之后动系数，正是本档设计要避免的那种失败模式。
   若确实要第二个奖励版本，必须在跑之前冻结新的判据与新的预测（照本档 §2 与 §5 的形式）。
3. **可以补的那一条消融**（成本最低、信息量最大）：同一 50k、同一锚，把 r3 的 ε 与 δ 置 0
   （`RewardSpec.as_r2_control()` 已经能构造），得到"**只差奖励可见性**"的同轮对照。
   这样 §8.3 的四臂就变成同轮五臂，可以把 +77.1 / +265.6 归因到奖励而不是轮次差异。
4. **策略侧的下一个候选**（若 200k 仍不够）：`advantage_normalisation = true` 把优势幅度
   抹平，这是诊断 H3 点名的第二个机制；它是 PPO 改动，不是奖励改动，必须单独预注册。
## 11 复现与验收

所有命令在 WSL 下、`ROOT420=/mnt/d/WorkShop/Touhou_RL`、`PY420=/root/v400-venv/bin/python`
下执行；脚本在 `.tmp_v420_r3/`。**单实例串行**，任何时刻只有一条重进程。

```bash
# 1) 审计（18 项；全 pass 才允许训练）—— 555.7 s
bash .tmp_v420_r3/run_audit.sh c        # -> T06/r3/audit-r3-c.json, exit 0

# 2) 50k（KL 锚 beta=1.0，T=1；gate_mode=diagnostic_only）—— 1012.4 s
bash .tmp_v420_r3/run_pilot.sh a        # -> T06/r3/pilot-r3-a.json, exit 0

# 3) 两口径 20 dev：一条命令 = 1 ruler x 5 seed，串行（每批 133–355 s）
bash .tmp_v420_r3/run_dev_stream.sh diag_first_death diag
bash .tmp_v420_r3/run_dev_stream.sh s1n s1n

# 4) 汇总证据（只读输入，写 T06/r3/reward_r3.json，再拷进 evidence/T06/）
/root/v400-venv/bin/python .tmp_v420_r3/kl_arm_reference.py
/root/v400-venv/bin/python .tmp_v420_r3/build_reward_r3.py
cp .tmp_v420_data/T06/r3/reward_r3.json \
   Document/Update/Alpha/v4.2.0/evidence/T06/reward_r3.json
```

验收（本档实跑）：

| 命令 | 侧 | 结果 | 退出码 |
|---|---|---|---:|
| `pytest -p no:cacheprovider --basetemp $PWD/.tmp_v420_data/tests/R3 -q tests/test_v420_rl.py` | WSL | **50 passed**（基线 44 + 新增 6） | **0** |
| 同上（`--basetemp .tmp_v420_data/tests/win-R3`） | Windows | **50 passed** | **0** |
| `python -m compileall -q touhou_rl/v420` | WSL / Windows | — | **0 / 0** |
| `python tools/rl/v420.py --help`、`... rl --help` | WSL | 两个子命令都正常 | **0 / 0** |
| 全量 10 个 `tests/test_v420_*.py` | WSL | **381 passed**（其中本档的 `test_v420_rl.py` = 50；其余 9 个文件本档**没改**，它们的数量变化来自并行任务） | **0** |

新增的 6 个测试（**只追加**，不删不改既有 44 个）：

1. `test_the_r3_table_is_the_r2_table_plus_two_bounded_survival_terms` —— r2 六个系数逐字节不变
   （`survive_per_decision` 除外，它正是从 0 变成 ε 的那一项）、新增项正确、**版本号与 config 一致**
   （`config.reward.version == rl.REWARD_VERSION_R3 == "v420-reward-r3/1"`）、每项都有 unit/counted。
2. `test_the_r3_survival_constant_obeys_its_declared_bound` —— 复算
   `ε·T_max = 0.119322 ≤ 0.125`、合并 0.369322 ≤ 0.5、δ 推导 residual 0.0，
   并断言校验器**拒绝** 5 种破坏它的表。
3. `test_the_r3_first_life_term_is_bounded_and_tracks_the_first_death` —— 首死更晚 ⇒ 首命项更大；
   cap 生效；`survive` 与死亡无关；并让它走一遍真实 `GameEnv2F` 的记账路径。
4. `test_the_new_criterion_passes_on_r3_and_is_failed_by_the_r2_numbers` —— 真实 24 局上
   r3 pass / **r2 控制表 Pearson 0.3545 fail**（与诊断逐位一致）；另有一组纯算术用例；
   并断言下限**只能上调不能下调**。
5. `test_the_kl_bound_is_a_declared_criterion_revision_in_r3_not_a_relaxation` —— β/空间/阈值未动；
   `gate_mode=diagnostic_only` 必须带 `criterion_revision` 数字（P1 0.8793413160101484、
   P2 flip 0.05092592592592592、P3 diag **1512.1 / n=20 / hold** + s1n `not_run` + `incomplete`、
   P4 `not_authorized_premise_false`）；诊断模式不改退出码，默认模式仍然 fail-closed。
6. `test_the_new_audit_criterion_is_frozen_in_the_configs_and_in_the_module` —— 判据、参照集 sha、
   门槛与模块常量在**两个**配置里一致，且冻结标记为 true。

### 11.1 变更文件（白名单内）

| 文件 | 变更 |
|---|---|
| `touhou_rl/v420/rl.py` | 版本化奖励表（r2 语义保留、r3 新增）；`first_life` 记账与上限；`survival_bound()` / `clear_derived_first_life()`；`first_death_visibility()` + `FIRST_DEATH_VISIBILITY_FLOOR`；审计第 18 项；`kl_anchor.gate_mode` + `apply_kl_anchor_diagnostic`；payload 的 `reward_version` 改为读 config 的声明版本 |
| `configs/v420/rl-engine-r3-aloudit.json` | **新建**（任务白名单授权的文件名），审计配置 |
| `configs/v420/rl-engine-r3-pilot.json` | **新建**，50k 配置 |
| `tests/test_v420_rl.py` | **只追加** 6 个测试 + 该节的 helper |
| `Document/Update/Alpha/v4.2.0/evidence/T06/REWARD_R3.md` | **新建**，本文件 |
| `Document/Update/Alpha/v4.2.0/evidence/T06/reward_r3.json` | **新建**，本档全部数字 |
| `.tmp_v420_data/T06/r3/`、`.tmp_v420_data/T06/dev-r3/`、`.tmp_v420_data/T06/engine-data-r3*`、`.tmp_v420_r3/` | 运行产物与脚本（`.tmp_*`，不入库） |

既有文件一个都没改：`configs/v420/rl-engine-{audit,pilot,pilot-kl}.json`、
`evidence/T06/` 里的既有文件、其它 tests、其它 `touhou_rl/**`、`tools/**`、`ASSET_SYNC.*` 全部未动。

### 11.2 剩余阻塞

| 阻塞 | 影响 | 解除需要 |
|---|---|---|
| 单 seed / 50k | 不能说"r3 有效/无效"，只能说"在 49 次更新内它把两口径各推了 +77.1 / +265.6 帧、仍差 3–4%" | 200k 中间档（§10.2 第 1 条）或 3 seed × 1M |
| 没有同轮"r3 但 ε=δ=0"的消融 | +77.1 / +265.6 不能严格归因到奖励（只能归因到"奖励 + 轮次"） | 同一 50k 再跑一条 `as_r2_control()` 臂（约 17 min + 8 批 dev） |
| R2+KL 的 20 dev 是**另一个任务**的产物 | 四臂里只有这一臂不是本档亲跑（逐批 sha 已记录） | 若要完全自持，需要重跑那条臂（20+20 局） |
| dev 只有 20 局、无置信区间 | 3–4% 的差距在 n=20 上不能给区间比较 | 更多 dev seed（但 confirm 池 61000–61499 本档禁止读） |
