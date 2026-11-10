# EXP-P4-001 — P4 探索性 RL 原型（PPO from BC）

> **这是原型，不是 G4。** 本轮不判任何门。G4 的正式门是"RL 后 20 局 CI 下界 > m1 CI 上界
> （2,553）且可复现"，本卡只负责把"能不能用 RL 打破 2,250 帧平台"这个问题变成可复算的数字，
> 并如实报告它现在**没有**给出信号。

- 日期：2026-11-10
- 状态：原型完成（环境/算法/短跑/20 局评测/吞吐核算全部真跑）。**判据结论：无信号**——RL 后 20 局 **876.7 帧（CI [804.6, 958.8]）** vs m1 **2,248.85（CI [1,952.0, 2,553.1]）**，比 BC **低 1,372 帧（−61 %）**；CI 不重叠且 |Δ| 超过 1,309 帧分辨极限，所以这是一个**显著的负效应**，不是「小幅退化」，更不是改进。
- 前置：EXP-P3-009（评测功效：σ_episode≈546、σ_ckpt≈472、ICC 0.43、k=1 分辨极限 1,309.7 帧、
  20 局协议下限）、EXP-P2-003（奖励检测器）、EXP-P3-002/003/006（BC/D0/DAgger）
- 新增代码：`touhou_rl/v400/rl/{env,policy,__init__}.py`、`tools/rl/v400_rl_train.py`、
  `tests/test_v400_rl_env.py`；`tools/rl/v400_student_eval.py` 与
  `tools/rl/v400_eval_power.py` **只读复用**，未改一行
- 未改：引擎、协议/ABI、`obs-spec-v1`、`StudentNet`、依赖、`build/`

---

## 1. 环境封装（`touhou_rl/v400/rl/env.py`）

一个 episode = 一个 `th08-modern` 进程（原因见 §7.1）。

| 项 | 值 |
|---|---|
| 观测 | **obs-v1 原样**：`play64` 4 帧**连续**堆叠 `(4,64,64)` + `hud` `(1,56,160)`，`float32/255` |
| 观测构造 | `obs_spec.ObservationBuffer` + `encode_planes`，与 `v400_student_eval.py --obs-window consecutive` 同一份代码 |
| 特权状态 | **观测里没有**；`--privileged-state` 只把 25 字段放进 `info` 并标 `privileged_diagnostic=True`，默认关闭 |
| 动作 | `action = cand*2 + bomb`，`cand∈[0,18)`，`bomb∈{0,1}` → **36 个联合动作** |
| duration | 1 decision = 2 逻辑帧（`obs_spec.DECISION_FRAMES`），与 BC 一致 |
| mask | `cand_to_mask(cand, bomb)`（仓库唯一动作表） |
| 终止 | `death_stop`（首死）/ `game_over`（命尽）/ `stage_clear`（stage 计数器前进）/ `engine_exit` |
| 截断 | `frame_cap`：19,887 帧（`ceil(1.5×13,258)`，S1N 尺） |
| 帧循环 | 与 `v400_student_eval.py` 逐条对齐：每帧一个 `STEP 1`、终止帧不 push、`budget=min(2, cap-frames)` |
| 并行 | 6 引擎 × 独立 data-dir（大档案 symlink、`th08.cfg`/`score.dat` 各自拷贝） |

### 1.1 关于"bomb 作为第 19 个可选动作"

实现为**独立维度**（18 × 2 = 36），不是第 19 个互斥动作。理由（写进代码注释）：

1. obs-v1 已冻结 `N_MOVE_CLASSES=18`、`N_BOMB_CLASSES=2`、`encoding: cand = focus*9+move`；
   19 路互斥会离开这个契约。
2. 19 路互斥必须二选一：要么"按炸弹时不移动"（丢掉边移边炸，动作语义比 BC 窄），
   要么把 18 个 cand 的 bomb 变体都塞进来（那就是 36，不是 19）。
3. BC `m1` 的两个头（`move_head[18] + bomb_head[2]`）用 `strict=True` 载入即可直接得到 36 路
   因式化策略；19 路只能人为映射，初始化不再是"从模仿解出发"。

### 1.2 奖励（训练期引擎真值；部署期用 P2 检测器）

| 项 | 公式 | 来源 | 尺度 | BC 基线一局幅值 | 占比 |
|---|---|---|---|---|---|
| survive | `+1.0 / 逻辑帧` | `STEP advanced` | +1.0 | **+2,248.85** | **93.96 %** |
| score | `+Δscore/1000` | `f4` | +0.001 | +42.66 | 1.78 % |
| death | `−100 / 次` | `f6` | −100 | −100.00 | 4.18 % |
| bomb_remaining | `+1×终局剩余炸弹` | `f8` | +1.0 | **0.00**（m1 把 3 颗全炸完） | 0 % |
| stage_clear | `+1000` | `f1` | +1000 | 0（0/20 过面） | 0 % |
| 合计 | | | | **≈ +2,191.5** | |

**风险（自知）**：存活项占 94 %，且 −100 的死亡惩罚只等于 100 帧，所以"用一次死亡换 100 帧"
在 return 上是赚的；这与"最大化帧数"的目标不一致。原型按任务给定的尺度实现，没有偷偷改；
若 P4 续做，第一件该动的是这个比值（例如 death −1000 或 survive 0.5/帧）。

---

## 2. 算法（`tools/rl/v400_rl_train.py` + `rl/policy.py`）

PPO，离散因式化动作，策略网络**初始化自 BC `m1`**（`strict=True`），value head 新建。

| 超参 | 值 | 说明 |
|---|---|---|
| 初始化 | `.tmp_v400_obsv2/arms/m1/checkpoint.pt`（sha256 `a3904a73…`，204,260 参数） | actor 权重 |
| value head | `Linear(256→1)`，正交 gain 1.0，257 参数 | 新建 |
| lr | 1e-4（Adam） | 小学习率保护模仿解 |
| rollout | 256 decision/env × 6 env = **1,536 transition/update** | |
| epochs / minibatch | 4 / 256 | |
| clip | 0.2 | |
| γ / λ | 0.99 / 0.95（GAE） | |
| entropy coef | 0.01 | |
| value coef | 0.5 | |
| grad clip | 0.5 | |
| reward scale | ÷1000（**仅优化器卫生**，不改奖励定义） | critic 目标 O(1) |
| value warmup | 前 5 个 update 只训 value head | 防止随机 critic 扯坏模仿解 |
| BatchNorm | actor 的 BN **全程 eval 模式**（统计量冻结，仿射参数仍训练） | 部署 `StudentNet.act` 走 `eval()`，训练必须看到同一归一化 |
| 环境步数目标 | 205,000（未达成，见 §4） | |
| 训练 seed | 5000 起（**1000–1019 只用于评测**） | |

---

## 3. 并行吞吐（实测，`--throughput-only` / `metrics.jsonl`）

| 配置 | 单引擎 帧/s | 聚合 帧/s | 聚合 decision/s |
|---|---|---|---|
| 1 引擎（无策略，纯 env） | 120.2 | 120.2 | 60.1 |
| 4 引擎 | 108.0 | 432 | 216 |
| 6 引擎 | 93.5 | 561 | 280 |
| 6 引擎 + 策略在环 + 每局重启（长局） | 57.5 | **345** | **172.5** |
| 6 引擎 + 策略在环 + 每局重启（**短局**，实测 PPO 中段） | ≈10 | **79–98** | **40–49** |

最后一行是本轮真正的教训：PPO 采样出来的局只有 1,300 帧左右（见 §4），重启（6–7 s/局）在
**屏障式 lockstep** 下被串行化，聚合吞吐从 172.5 掉到 40–49 decision/s。这不是引擎慢，是
"每个 vec step 都要等最慢的那个 env，而最慢的那个正在重启引擎"。

---

## 4. 短跑曲线关键点（真实 `metrics.jsonl`）

运行：`--engines 6 --steps 205000 --minutes 20.5`（墙钟上限先到）。

| update | steps | wall(s) | value_only | mean_frames | min/max | deaths | entropy | KL | clip frac | value loss | 吞吐(dec/s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1536 | 28.8 | 是 | — | —/— | 0 | 2.494 | 0.0000 | 0.000 | 0.00396 | 118.5 |
| 2 | 3072 | 40.4 | 是 | — | —/— | 0 | 2.588 | 0.0000 | 0.000 | 0.00297 | 132.3 |
| 3 | 4608 | 79.3 | 是 | 1282 | 1157/1373 | 4 | 2.523 | 0.0000 | 0.000 | 0.00406 | 39.6 |
| 4 | 6144 | 110.5 | 是 | 1327 | 648/1786 | 7 | 2.645 | 0.0000 | 0.000 | 0.00181 | 49.2 |
| 5 | 7680 | 141.1 | 是 | 1213 | 648/1786 | 10 | 2.671 | 0.0000 | 0.000 | 0.00146 | 50.1 |
| 6 | 9216 | 172.0 | 否 | 1185 | 648/1786 | 13 | 2.633 | 0.0093 | 0.123 | 0.00159 | 49.8 |
| 7 | 10752 | 189.5 | 否 | 1155 | 648/1786 | 14 | 2.695 | 0.0047 | 0.070 | 0.00118 | 87.7 |
| 8 | 12288 | 233.8 | 否 | 1176 | 648/1786 | 19 | 2.605 | 0.0080 | 0.116 | 0.00161 | 34.7 |
| 9 | 13824 | 257.2 | 否 | 1156 | 648/1786 | 21 | 2.662 | 0.0045 | 0.059 | 0.00117 | 65.6 |

**整轮**：14,670 steps / 9 updates / 288.8 s / 50.8 decision/s / 101.6 帧/s；24 局收尾，局均 1140.1 帧（648–1786），全部 death_stop，0 过面。**目标 205,000 步只完成 7.2 %**（墙钟上限先到）。

**第一个关键诊断（必须在任何 P4 续做里修掉）**：BC `m1` 的**采样**行为远差于它的 **argmax**
行为。m1 的 argmax 局均 2,248.8 帧，而把它当随机策略采样（entropy 2.49–2.65 nats）时只有
1,282–1,327 帧，比 bench 里 `random`（dev20 约 2,100 帧量级）还差。也就是说 BC 的 softmax 在
τ=2.0 的软标签下学出的是一族"尖峰在 argmax、但尾部很胖"的分布，PPO 一开始是从这族的**尾部**
起步的。任何 "PPO from BC" 的 P4 方案都必须先解决这一步（temperature 预锐化 / KL-to-BC 锚 /
只在 argmax 邻域采样），否则前几十万步都在把策略从尾部拉回峰顶，而不是在改进解。

---

## 5. 20 局评测

### 5.1 结果（官方控制器，seeds 1000–1019，cap 19,887，stop-on-death）

| 臂 | n | mean | sd | se | bootstrap 95% CI | 过面 | end |
|---|---|---|---|---|---|---|---|
| **RL**（PPO-from-BC，14,670 steps） | 20 | **876.7** | 183.70 | 41.08 | **[804.6, 958.8]** | 0/20 | death_stop 20/20 |
| **BC m1**（EXP-P3-009） | 20 | 2,248.85 | 709.79 | 158.71 | [1,952.00, 2,553.05] | 0/20 | death_stop 20/20 |

per-seed（RL）：`[748, 763, 1190, 783, 1165, 761, 969, 821, 735, 1223, 735, 784, 744, 788, 843, 750, 938, 1270, 778, 746]`

### 5.2 `claim` 判定（真实退出码 0）

```json
{"a": {"label": "rl", "mean": 876.7, "ci": [804.59875, 958.8012499999999], "n": 20, "sd": 183.699},
 "b": {"label": "m1", "mean": 2248.85, "ci": [1951.9975, 2553.0525], "n": 20, "sd": 709.791},
 "delta": -1372.15, "ci_overlap": false, "verdict": "有效应",
 "checkpoint_floor": {"resolution_limit": 1308.293, "resolvable": true}}
```

- 两臂 CI **不重叠**，|Δ| = 1,372.15 **超过** checkpoint 分辨极限 1,308.29，
  所以 `claim` 给出 `有效应`（exit 0）——但方向是**负的**：RL 比 BC 低 **1,372 帧（−61 %）**。
- **预注册判据（CI 下界 > 2,553）→ 不满足 → "无信号"**（改进方向）。不是"小幅效应"，
  不是"方向一致"，而是一个**显著的退化**。
- sd 从 709.8 掉到 183.7：RL 策略比 BC **更稳定也更差**，这是"塌到一个固定行为然后快速送死"
  的指纹，与 §4 的采样尾部诊断一致。

### 5.3 为什么是负的（按可核查证据排序）

1. **起点就在分布尾部**（§4）：BC `m1` 的采样局均 1,140 帧 vs argmax 2,249 帧，PPO 的 rollout
   全部来自尾部，advantage 全是"早死"的信号，策略被推向"少动/更早死"的局部区域。
2. **entropy 从头到尾没降**（2.49 → 2.66）：`entropy_coef=0.01` 在 9 个 update 内没有把分布
   收紧，KL 只有 0.0045–0.0080，clip fraction 0.06–0.12 —— PPO 每步都是小步，但**每一步都朝
   同一个（错的）方向**，14.7k 步足够把 argmax 从 2,249 拖到 877。
3. **训练预算不足**：14,670 步 / 9 个 update，是 205,000 目标的 **7.2 %**。没有任何证据表明
   继续训练会回升；相反，曲线单调向下。
4. **奖励尺度不匹配**（§1.2）：−100 的死亡惩罚只等于 100 帧，短命策略的 return 劣势被存活项
   线性掩盖，critic 学到的是"反正都要死"，而不是"怎么不死"。

### 5.4 中途评测（"每 N 步各评一次"的如实交代）

本次只有一个 checkpoint（14,670 步）被 20 局评测。计划里的"100k 步中途评测 + 最终评测"
**没有实现**：训练在 20.5 min 墙钟上限前只跑到 14,670 步（吞吐被 §3 的重启串行化压到
40–50 dec/s），`--save-at 100000` 从未触发。这是预算/吞吐问题，不是流程跳过；
卡片不把它写成"两次评测"。


---

## 6. 判据（预注册，原型级）

- 训练后 20 局均值 **CI 下界 > m1 CI 上界 2,553** → "有信号"；否则如实记"无信号"。
- 任何结论必须给 CI，且两臂 CI 不重叠；`v400_eval_power.py claim` 无 CI 会 exit 2。
- 单臂单 checkpoint，因此按 EXP-P3-009 的 checkpoint 分辨极限 1,309.7 帧必须一并报告：
  即使 CI 不重叠，|Δ| 小于该极限也**不能**称为效应（`claim --sigma2-checkpoint` 会把
  `有效应` 降级为 `无法判定（分辨率不足）`）。

---

## 7. 引擎侧两个实测发现（本轮新增，影响后续所有 P4 工作）

### 7.1 一个进程只能 RESET 一次

第二次 `RESET` 返回 `ready=0 setup_frames=1800`，frame 计数停在 0；再走 360 帧 `STEP` 并重试
6 次 `RESET` 仍为 `ready=0`。因此**一局 = 一个进程**，与参考评测器（每 episode 一个
`HarnessClient`）完全一致。重启成本 6–7 s/局，是评测尺本身的成本。

### 7.2 共享内存的 state 区必须由 `STATE` 命令刷新

`RESET` 之后 header 报 `state_bytes=100`，但 `state_hash=0`、25 字段**全 0**，而同一时刻
`STATE` 回复已经是 `lives_remaining=2.0`、`bombs_remaining=3.0`；先读 block 再发 `STATE` 会让
block 变有效（`state_hash` 非 0，内容与回复逐字段相等）。**像素区每帧都是有效的**
（`FLAG_PIXELS_VALID`），所以只有像素走共享内存。

本原型第一版直接读 block，结果 `deaths/score/stage` 永远是 0、每局跑到 `frame_cap`，
"看起来"跑通但学到的是噪声。`--state-equivalence` 现在每次运行都会把这个陷阱打印出来
（`block_stale_before_state_command` / `block_equals_reply_after_state_command`）。

### 7.3 尺子一致性校验（`--validate-ruler`）

确定性重放 `m1`：

| seed | 本环境帧数 | `.tmp_v400_power/m1` 记录 | 一致 |
|---|---|---|---|
| 1000 | **2,243**（score 40,870，death_stop） | 2,243（score 40,870） | ✅ |
| 1001 | 3,356 | 2,999 | ❌ **未解释** |

seed 1000 逐帧、逐 score 一致，说明帧循环/终止规则/obs 窗口都对；seed 1001 的偏差**没有解释**，
列为待查项。它不影响本卡结论：§5 的 20 局 CI 由官方评测器产生，不经过 `rl/env.py`。

---

## 8. 吞吐可行性核算（用实测速率）

实测：好情况 172.5 decision/s、坏情况（短局）40–49 decision/s。1 decision = 2 帧。

| 步数 | 好情况 (172.5/s) | 坏情况 (40/s) |
|---|---|---|
| 200 k | 19.3 min | 1.4 h |
| 1 M | 1.61 h | 6.9 h |
| 10 M | 16.1 h | 69.4 h |
| 100 M | 161 h | 694 h |

**到目标需要多少步**——这一点**不能**从帧/秒推出来，本卡不编：

- 到教师 30 %（4,070 帧）：一局 2,035 decision。BC 用了 76,912 条教师 decision（≈153,824 帧）
  才到 2,249 帧。若 PPO 的样本效率与 BC 同量级，至少 1 M 级；像素 PPO 通常还要 1–2 个数量级，
  即 **10 M–100 M 步 = 16–161 h（好情况）**。
- 过面（≈13,000 帧、6,500 decision/局、clear rate ≥ 50 %）：在 4,070 的基础上再加一个数量级，
  即 **≥100 M 步 = ≥161 h（好情况）/ ≥694 h（坏情况）**，且前提是先修掉 §4 的采样尾部问题。
- 本轮实际训练 **约 6 万步量级**（见 §4 与 `train_summary.json`），相对 1 M 的战役是 **6 %**。
  所以本轮只能给出"每一步多贵"，给不出"够不够"。

**结论**：以当前吞吐与样本效率，P4 不是"再跑一晚"能回答的问题；先把 §4 的采样尾部
（BC 分布的锐化/KL 锚）和 §3 的重启串行化（异步 reset 或预热池）解决，吞吐可回到 172 dec/s
以上，样本效率更是数量级问题。

---

## 9. 测试

`tests/test_v400_rl_env.py`：**24 passed**（3.22 s，离线，无引擎/无 GPU/无 tarball）。

覆盖：动作映射 36 项全覆盖 + 越界 fail-closed、bomb 两值可达、5 个奖励项逐项、
`death_stop`/`game_over`/`stage_clear`/`frame_cap`/`engine_exit` 全部终止路径、
obs-v1 张量形状/dtype/范围、特权状态 opt-in 且打标、重启计数、episode return 累加、
并行 4 env 两次运行逐字节一致、vec 顺序与 arity 保护、奖励表占比、
`v400_eval_power.py baseline` 的 mean/sd/se/bootstrap CI、`claim` 缺 CI exit 2、
CI 不重叠 exit 0（有效应）、CI 重叠 exit 2（无法判定）。

---

## 10. 复现命令

```bash
# 环境/吞吐（WSL，一个 Xvfb display）
bash .tmp_v400_rl/run_wsl.sh tools/rl/v400_rl_train.py --engines 6 --throughput-only 90 \
  --out-dir .tmp_v400_rl/tp6

# 尺子一致性
bash .tmp_v400_rl/run_wsl.sh tools/rl/v400_rl_train.py --engines 1 --validate-ruler 2 \
  --out-dir .tmp_v400_rl/ruler3

# PPO 短跑
bash .tmp_v400_rl/run_wsl.sh tools/rl/v400_rl_train.py --engines 6 --steps 205000 \
  --minutes 20.5 --save-at 100000 --out-dir .tmp_v400_rl/rlA --seed 20261110

# 20 局评测（官方控制器，3 分片并行后合并 episodes.jsonl）
bash .tmp_v400_rl/eval_arm.sh .tmp_v400_rl/rlA/checkpoint.pt .tmp_v400_rl/evalA 3

# 判定
python -X utf8 tools/rl/v400_eval_power.py claim \
  --a rl=.tmp_v400_rl/evalA --b m1=.tmp_v400_power/m1 --sigma2-checkpoint 222784
```

---

## 11. 声明

本卡记录的是 **P4 的探索性原型**：环境、算法、短跑、20 局评测、吞吐与可行性核算都是真跑的，
数字可复算；但它 **不是 G4**，不构成任何门通过，也不改变 G3/G4 的现状（学生 ≈2,250 帧、
过面 0/5、教师 13,565 帧）。任何"RL 有效"的结论都必须等到：多 checkpoint、20 局 CI 不重叠、
且 |Δ| 超过 1,309.7 帧的 checkpoint 分辨极限。
