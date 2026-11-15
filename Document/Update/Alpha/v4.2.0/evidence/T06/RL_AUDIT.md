# T06 / WP420-05：零更新与数值审计（折扣修订后复跑）

对象：v4.2.0 `T06`（WP420-05 引擎回报 RL 诊断）**第一档**的零更新审计，**第二次运行**
（唯一一次折扣修订 τ 30 s → 200 s 之后）。
运行产物：[`rl_audit.json`](rl_audit.json)（本文件的全部数字都由它生成）。
原始运行：`.tmp_v420_data/T06/audit2.json`
（`tools/rl/v420.py rl --config configs/v420/rl-engine-audit.json --out ... --execute`，**退出码 0**，墙钟 465.2 s）。

**结论：PASS。17 项全部 pass，`failed = []`。**
上一轮唯一失败项 `reward_discount_horizon` 已按
[`DISCOUNT_REVISION.md`](DISCOUNT_REVISION.md) 的唯一一次修订修复：τ 30 s → 200 s、
γ → 0.9998333472214507；**奖励表系数逐字节未动**。

本档**没有**训练任何策略：唯一动过优化器的地方是 critic 预热（actor 及其共享特征冻结）
和零更新路径（前向 + 反向，**不** step）。

> R2 uses engine-truth reward and termination, so it is a training-time privileged
> diagnostic: it is NOT strict pixel self-improvement (R1) and no result here may be
> reported as R1.

## 1 协议与身份（跑之前冻结，跑之后不改）

| 项 | 值 |
|---|---|
| 契约 | `v420-rl/1` / `obs-v420-2f` / `action-v420-2f` |
| 奖励版本 | `v420-reward-r2/1`（来源 `engine_truth`；系数与上一轮完全相同） |
| 终止语义版本 | `v420-termination/1` |
| 折扣版本 | **`v420-discount/2`**（`/1` 是 τ=30 s 的初始配置，已由唯一一次修订取代） |
| 初始化 checkpoint | `/mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T06/checkpoints/a0_seed0_step5000.pt` |
| checkpoint sha256 | `96cb219eac83391f42b18ea656b5a6c5ac81e97cd1fd8c1e2c0d404aee6b89ad` |
| 与源文件一致 | true；声明 sha 校验 true（full sha256） |
| 参数量 | actor 731802、critic 129（Linear(128,1) on the actor trunk output） |
| obs-v3 / obs-v420-2f digest | `ef2fbd15385f26f425db2bfe855d105ee009023acb08c26916e811963553a105` |
| torch / numpy | 2.14.1+cu130 / 2.2.6 |
| device | `cpu` |
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern` |
| **引擎 data 目录** | `/mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T06/engine-data-gamesrc`（`cp -a` 私有副本，**不是** `/root/v400-data/game`；见 §5.2） |
| 观测 | play `[4, 112, 96]` uint8、hud `[56, 160]`、act_hist `[8, 18]`；历史只含**已执行**动作 |
| 动作 | 18 个 move/focus，**bomb 固定 0**（不训 bomb 通道） |
| 每决策帧数 | 2 逻辑帧 |

### 1.1 折扣（修订后复算一次）

| 量 | 值 |
|---|---|
| Δt | `0.03333333333333333` s = 2/60（契约常量，不是自由参数） |
| τ | **`200.0` s**（唯一一次修订，原 30.0 s） |
| γ = exp(-Δt/τ) | **0.9998333472214507**（模块复算一致：true；与旧发布值不同：因为 τ 变了） |
| GAE λ | `0.95` |
| 有效视界 | **6000.5 决策** ≈ 200.017 s |
| γ^T at T=6,598 | **0.33298205922092694** |
| 同一量在 τ=30 s（被替换） | 6.548455766e-04 |
| 可见性下限（config 冻结） | `γ^T_clear_min = 0.30`、`|clear|/|early_death| ≥ 1.5` |

候选 τ 表、事前规则与"为什么这不是放宽容忍度"见
[`DISCOUNT_REVISION.md`](DISCOUNT_REVISION.md)（数字见
[`discount_revision.json`](discount_revision.json)）。

### 1.2 奖励表（R2，引擎真值；单位与系数一起冻结在 `configs/v420/rl-engine-audit.json`）

| 项 | 值 | 单位 / 角色 |
|---|---|---|
| `stage_clear_bonus` | 1.0 | reward per first-face completion；primary positive event |
| `death_penalty` | 0.25 | reward per engine death event；auxiliary（首死**不**终止 episode） |
| `bomb_penalty` | 0.1 | reward per bomb spent；auxiliary |
| `progress_total` | 0.5 | reward for traversing the whole first face by score；**硬上限**（累计封顶） |
| `progress_score_full` | 2631430.0 | 归一化分母：真实过关轨迹的终局分数（本档实测） |
| `survive_per_decision` | 0.0 | **显式为零**：没有裸生存常数 |

`progress_bound`：累计上限 0.5，每决策速率 1.900108e-07。
因此**任何**不结束的拖延轨迹最多只能拿到 0.5 的进度奖励，永远低于一次过关（1.5）。
修订只动了 τ：上表六个系数与本轮审计之前**完全一致**（`scope.unchanged` 逐项记录在
[`discount_revision.json`](discount_revision.json)）。

### 1.3 PPO（冻结值）

clip `0.2`、梯度裁剪 `0.5`、actor lr `3e-05`、critic lr `0.0003`、
熵系数 `0.001`（事前从 0 / 0.001 中冻结这一个）、epochs `2`、minibatch `128`、
每环境 rollout `512` 决策、采样温度 `8.0`（审计 §3.10 的事前网格规则选出后冻结）。

## 2 零更新审计逐项结果（17 项）

| 检查 | 状态 | 关键实测 |
|---|---|---|
| `obs_contract` | **pass** | play [4,112,96] / hud [56,160] / hist [8,18]，历史只含已执行动作 true |
| `initial_logits` | **pass** | shape [256,18]，range 55.923，两次前向逐比特一致 true |
| `argmax_behaviour` | **pass** | 9 个不同动作，最大占比 0.0483 ≤ 0.6 |
| `sample_behaviour` | **pass** | T=8.0，17/256 个动作，最大占比 0.1797 |
| `initial_policy_degeneracy` | **pass** | T=1 熵 0.413 nat（退化）→ T=8.0 后 2.3948 nat |
| `joint_log_prob` | **pass** | exp 行和 [0.999999821, 1.000000238]，记录值一致 true |
| `ratio_zero_update` | **pass** | max\|ratio-1\| = 9.536747712e-07 ≤ 1e-06，参数逐比特未变 true |
| `gae` | **pass** | 256 transition / 1 segment，与独立实现最大差 **3.388e-21** |
| `gamma_lambda` | **pass** | γ = 0.9998333472214507（复算 true），λ = 0.95，视界 6000.5 决策 |
| `truncation_bootstrap` | **pass** | 1 个 truncated transition，bootstrap 1（critic bias 1.0） |
| `death_vs_game_over` | **pass** | 结束 game_over，首死帧 1370 之后继续 408 决策 |
| `reward_units` | **pass** | 5 项齐备，survive=0.0，progress 上限 0.5 未越界 |
| `reward_ordering` | **pass** | clear_gt_delay 2.2488 / clear_gt_early_death 1.7489 / clear_gt_bomb_abuse 2.6909（未折扣） |
| `reward_discount_horizon` | **pass** | γ^T at clear = **0.332982059**；discounted 边际 +1.2066 / +0.7456 / +1.6172；可见性下限 0.30 pass；\|clear\|/\|early\| = 2.357 ≥ 1.5 pass |
| `critic_warmup` | **pass** | 预热前后 actor logits 逐比特不变 **true**，最大差 0 |
| `bomb_channel_zero` | **pass** | bomb 常量 0，越界调用被拒 |
| `forward_equivalence` | **pass** | 32 行，最大差 0，逐比特一致 true |

**pass 17 / 17，fail 0。** `--execute` 退出码 **0**（上一轮因 1 项 fail 为 3）。

## 3 关键项细节

### 3.1 观测契约（`obs_contract`）

play `[4, 112, 96]`、hud `[56, 160]`、act_hist `[8, 18]`，
dtype `{'play': 'uint8', 'hud': 'uint8', 'act_hist': 'uint8'}`。历史窗口逐行独立重建并与模型实际输入**逐字节**比较
（256 行，`history_is_executed_only=true`）：每个决策的观测只含它**之前**已执行的动作。

### 3.2 初始 logits（`initial_logits`）

shape `[256, 18]`、finite `true`、两次前向逐比特一致 `true`、
float32 sha256 `c8b6df33929365ae98be5735b2ba8818d774bf641ae2ad693086fef4c0606d85`、范围 [-26.8796, 29.0432]、
range 55.9227、sd 8.0786。

### 3.3 argmax 与随机采样

argmax（部署规则）：256 个决策、9 个不同动作、最大占比 **0.0483**（阈值 0.6）。
采样：256 次抽取、17 个不同动作、最大占比 0.1797，温度 8.0；
温度 1.0 时同一批观测只抽到 9 个动作、最大占比 0.2578。

### 3.4 联合 log_prob（`joint_log_prob`）

256×18 矩阵：每行 exp 之和 ∈ [0.999999821, 1.000000238]；采集时记录的 `log_prob`
与矩阵逐元素一致 `true`；同一行 batch 前向与单行前向的差 9.536743164e-07
（容差 1e-4，float32 卷积重结合的正常量级）。

### 3.5 PPO ratio（零更新）

重算的 ratio：min 0.999999046 / max 1.000000954，`max|ratio-1|` = **9.536747712e-07**
（容差 1e-06）。真实 `ppo_update` 路径（前向 + 两个损失 + 反向，**不** step）：
`max|ratio-1|` = 9.536743164e-07；4.0 个 minibatch 之后 actor 与 critic 参数逐比特未变 `true`。

### 3.6 GAE（`gae`）

256 个 transition、1 个 segment（episode 边界分段）。与一份**独立实现**
（显式写出 `A_t = Σ_k (γλ)^k δ_{t+k}`，并从数组取 `V(s_{t+1})`）比较：
advantage 最大差 **3.388e-21**、return 最大差 3.388e-21；
advantage 均值 3.151889e-06、sd 6.506739e-06、范围 [0, 3.214101e-05]。
（γ 变大后同一条 rollout 的 advantage 量级基本不变，这是预期的：进度项每决策极小。）

### 3.7 γ / λ 复算（`gamma_lambda`）

模块复算 `exp(-Δt/τ)` = **0.9998333472214507**，与发布常量一致 true；
`1/(1-γ)` = **6000.5 决策**（200.017 s）；`γ^T`(T=6598) = **0.33298205922092694**。
同一条目同时报告被替换的 τ=30 s 的 γ（0.9988895059442793）与它的 γ^T（6.548456e-04），
所以"修订前后的对比"不需要回到旧证据里翻数字。

### 3.8 奖励单位（`reward_units`）

过关分解 `{'survive': 0.0, 'progress': 0.5, 'death': 0.0, 'bomb': 0.0, 'stage_clear': 1.0}`；
拖延 `{'survive': 0.0, 'progress': 0.0011837670012122693, 'death': -0.75, 'bomb': 0.0, 'stage_clear': 0.0}`；
早死 `{'survive': 0.0, 'progress': 0.001055699752605998, 'death': -0.25, 'bomb': 0.0, 'stage_clear': 0.0}`；
滥用 bomb `{'survive': 0.0, 'progress': 0.00908213404878713, 'death': -0.5, 'bomb': -0.7, 'stage_clear': 0.0}`。
`survive` 常数 = 0.0，进度累计上限 = 0.5，两类轨迹的 `progress` 均未越界。

### 3.9 truncation bootstrap 与 death / game_over

帧上限打断：68 帧、34 个决策里出现 **1 个 truncated**、0 个 terminated。
该 transition 的 bootstrap = **1**（检查时把 critic bias 设为 1.0，否则零初始化的 critic
会让"0 对 0"的比较毫无信息量），对应 advantage -0.000166653
= r + γ·V(s_T) − V(s_t)；其余 transition 的 bootstrap 全为 0。

真实一整局：结束原因 **game_over**、2186 帧、首死帧 1370、首死之后**又跑了 408 个决策**、
death 事件 3 次、终局 deaths 3.0，折扣回报 -0.646074。
即：`death` 是事件，`game_over` 才是终止；两者由冻结契约分类器
（`touhou_rl.v420.contracts.classify_termination`）判定，本模块没有第二套定义。

### 3.10 初始策略采样退化与温度修法（`initial_policy_degeneracy`）

- 温度 1.0（A0 原样）：平均行熵 **0.41264227** nat、有效动作数 1.5108、采样最大占比 0.2578。
  **这就是 WP420-05 §5 说的"初始策略采样已经退化"**：A0 的 softmax 实质上是确定性的。
- 事前冻结的修法：在网格 [1,2,4,8,16,32] 上取**最小**且同时满足两个事前阈值的温度
  （行熵 ≥ 1.0 nat 且有效动作 ≥ 6.0）→ 选中 **T = 8.0**。
- 修法只改**探索**：正温度不改变 argmax，而部署（冻结 evaluator 与 R0）是 argmax。
- 修后：行熵 2.3948 nat（0.8286 × ln18）、有效动作 10.9664、argmax 占比 0.0483。

| T | 行熵 (nat) | 有效动作 | 满足阈值 |
|---:|---:|---:|---|
| 1.0 | 0.412642 | 1.5108 | false |
| 2.0 | 0.849673 | 2.3389 | false |
| 4.0 | 1.61467 | 5.0262 | false |
| 8.0 | 2.394838 | 10.9664 | true |
| 16.0 | 2.760518 | 15.808 | true |
| 32.0 | 2.858443 | 17.4344 | true |

### 3.11 critic 预热（`critic_warmup`）

200 步、batch 256、lr 0.0003；冻结：actor 及其共享 trunk；训练：critic。
value loss 3.191677e-11 → 1.444518e-04，预热后 explained variance -43216.31
（γ 变大后回报尺度变大，这个数只用于证明 critic 真的在动）。
**actor logits 预热前后逐比特不变：true**（最大绝对差 0；前后 sha256
`9f7dfc7d1542b3737b4c70a12eadb6356236d36b13b7fd8bf19ee85cf74f14a6` 相同）。

### 3.12 前向等价（`forward_equivalence`）

critic 读的是 actor 的 trunk 输出，所以"显式特征路径"必须与冻结的 `model(...)` 完全一致：
32 行，最大绝对差 0，逐比特一致 **true**。

### 3.13 bomb 通道

`bomb_channel_zero`：bomb 常量 0，18 个动作只有 move/focus；训练时传入 bomb=1 会被拒绝
（只有审计的 bomb 滥用类走显式 override）。

## 4 四类真实轨迹与排序（奖励表本身的性质）

四类**全部是真实引擎轨迹**：

- **过关**：回放本项目唯一一条真实 Stage-1 过关录制（shard
  `.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed12-shard/slots.jsonl`、
  sha `f759de7dacc78343dec8825da5de14942e749b8a5a2c5319304e513d1618b7b9`、seed 12、
  6588 槽 + 64 个尾部决策）；
- **早死 / 拖延 / 滥用 bomb**：本次新跑的真实 episode（各自的私有 data 目录）。

| 轨迹类 | 结束原因 | 帧数 | 折扣总回报 G | 未折扣 Σr | clear | progress | death | bomb | survive |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 过关 (clear) | stage_clear | 13195 | **+0.523500** | 1.5 | 1 | 0.5 | 0 | 0 | 0 |
| 早死 (early death) | first_death | 1370 | **-0.222079** | -0.248944 | 0 | 0.001056 | -0.25 | 0 | 0 |
| 拖延 (stall) | game_over | 1503 | **-0.683100** | -0.748816 | 0 | 0.001184 | -0.75 | 0 | 0 |
| 滥用 bomb (bomb abuse) | cap | 2000 | **-1.093724** | -1.190918 | 0 | 0.009082 | -0.5 | -0.7 | 0 |

> G = Σ_t γ^t r_t，γ = 0.9998333472214507，**不做任何归一化**。

排序检查（**未折扣**口径 —— 奖励表本身的性质，长度不会污染它）：

| 检查 | 状态 | 数值 |
|---|---|---|
| `clear_gt_delay_undiscounted` | **pass** | margin = 2.248816 |
| `clear_gt_early_death_undiscounted` | **pass** | margin = 1.748944 |
| `clear_gt_bomb_abuse_undiscounted` | **pass** | margin = 2.690918 |
| `clear_actually_cleared` | **pass** | true |
| `non_clear_upper_bound_below_clear` | **pass** | non_clear_bound = 0.5 < 1.5 |
| `survive_constant_absent` | **pass** | survive_per_decision = 0 |

"不结束的拖延"上界 0.5 低于一次过关 1.5 —— 这正是**没有**裸生存常数的直接后果。
修订后这个性质在折扣口径下同样成立：过关的折扣回报 0.5235 也高于 0.5 的上界。

### 4.1 折扣视界（`reward_discount_horizon`）：修订后 pass

同一批轨迹，按 `configs/v420/rl-engine-audit.json` 冻结的 γ = 0.9998333472214507 折扣。
判定是**显式 6 条决定性检查**（`decisive_checks`）：

| 检查 | 状态 | 数值 |
|---|---|---|
| `clear_gt_delay_discounted` | **pass** | 0.523500 > -0.683100，margin +1.206600 |
| `clear_gt_early_death_discounted` | **pass** | 0.523500 > -0.222079，margin +0.745578 |
| `clear_gt_bomb_abuse_discounted` | **pass** | 0.523500 > -1.093724，margin +1.617223 |
| `terminal_bonus_is_not_negligible` | **pass** | 1.0 × 0.332982 > 0.25（一次死亡罚分） |
| `clear_terminal_visibility_floor` | **pass** | γ^T = 0.332982 ≥ **0.30**（显式下限） |
| `clear_vs_early_death_discounted_margin` | **pass** | \|clear\|/\|early\| = **2.357271** ≥ **1.5** |

过关轨迹共 6598 个决策（约 219.9 s），终局奖励的折扣权重 **γ^T = 0.3329820592**：
一次过关的 1.0 奖金在到达终点时值 **0.332982**，是**一次死亡罚分 0.25 的 1.33 倍**
（修订前是 6.548e-04，即 0.0026 倍）。

**报告但不作判据**的两项（`observed_only`）：在 TH08 第一面里，"拖得更久"必然多付死亡罚分
（过关之外没有别的结束方式，任何跑到 game_over 的 episode 都花光 3 条命），所以
"拖延 > 早死"检验的是死亡系数的量级，而不是"生存项是否压过任务目标"：

| 观察项 | 结果 | 数值 |
|---|---|---|
| `delay_gt_early_death_undiscounted` | does not hold | margin = -0.499872 |
| `delay_gt_bomb_abuse_undiscounted` | holds | margin = +0.442102 |
| `delay_gt_early_death_discounted` | does not hold | margin = -0.461021 |
| `delay_gt_bomb_abuse_discounted` | holds | margin = +0.410624 |

## 5 连接层的三个实测发现

### 5.1 引擎需要 SHOOT 的上升沿才能离开过关状态

同一段录制动作、不加脉冲时，真实引擎跑满 13,272 帧、分数停在 141,379、
`stage_clear` 永不触发；加"每 48 决策松一帧 SHOOT"之后，同一条轨迹在 13,195 帧过关、
分数 2,631,430。所以 `shoot_pulse_period` 属于**动作层**（不是奖励、不是策略输出），
2 帧节拍下取 48 决策（= 96 帧）。一个每帧都按住 SHOOT 的策略**永远不可能过关**。

### 5.2 引擎 data 目录改为 fail-closed（本轮修掉的偏差）

上一轮 dev 评测把 `/root/v400-data/game` 直接交给冻结 evaluator 的 `--data-dir`，
引擎因此**写掉了那里的 `score.dat`**。本轮把它改成硬失败：

- `configs/v420/rl-engine-{audit,pilot}.json` 的 `data_dir` 现在是
  `.tmp_v420_data/T06/engine-data-gamesrc`（`cp -a` 出来的私有副本，474 MB，3.1 s）；
- `touhou_rl.v420.rl.check_data_dir` 在三处调用：`load_rl_config`、
  `private_data_dir`（每个 episode 的私有目录准备）、`dev_eval_invocation`
  （冻结 evaluator 的 argv 组装）；
- 拒绝的标记：`/root/v400-data`、`.tmp_v400_`、`.tmp_v410_`、
  `Document/Update/Alpha/v4.0.0`、`v4.1.0`、以及 v4.2.0 的 `evidence/T02`–`T05`；
- 单测覆盖（`tests/test_v420_rl.py::test_the_engine_data_dir_is_fail_closed_on_legacy_trees`）：
  7 个禁止路径（POSIX 与 Windows 两种写法）、3 种空值、config 载入拒绝、dev-eval argv 拒绝；
- 本轮运行里 `data_dir_check.data_dir` = `.tmp_v420_data/T06/engine-data-gamesrc`，
  审计 payload 逐字记录了这条路径（`rl_audit.json` 顶层 `data_dir`）；
- 校验：`/root/v400-data/game/score.dat` 的 mtime 停在 `2026-11-14 00:36:50`
  （本轮第一次运行之前），本轮**没有**写过那个目录。

### 5.3 A0 的采样分布是退化的

见 §3.10：必须先用事前冻结的温度规则处理分布，而不是靠加长训练掩盖。

## 6 边界（不能主张什么）

- 本档**没有**任何策略改进证据：它只证明连接层、奖励、终止、折扣与统计在数值上成立。
- R2 用引擎真值奖励与终止 ⇒ **不是**严格像素自我改进（R1）。不得写成 R1 的结果。
- 一次审计、一个 checkpoint、一个 seed 的数值正确性，不等于 3 seed × 1M 的训练结论。
- 17/17 pass 只说明"门"是干净的；`G420-R` 仍是 **not_passed**（缺口见
  [`RL_PILOT2.md`](RL_PILOT2.md) §6）。
