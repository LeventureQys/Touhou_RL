# v8.10.0 HAZFIX-001 — RUN

预注册：`Document/Update/Alpha/v8.10.0/PREREGISTRATION_HAZFIX_001.md`（FROZEN 2026-11-22，
sha256 `fbc2f1937d44166d…`，本轮一字未改）。
执行器：`tools/rl/v8a0_hazfix.py`（新文件；重放循环 import `tools/rl/v860_bombdag.py`，
sha256 `3433ae743cc224ca…`）。
训练端本机实测，2026-11-22 12:00–12:03（Asia/Shanghai）。

---

## 0. 结论：**P0 不成立 → `S-data`，不训练、不评测**

| P0 条件 | 要求 | 实测 | 判定 |
| --- | --- | --- | --- |
| ① 重放 5,040 状态与 v8.6 逐状态一致 | `features_sha256_k12` 命中率 ≥ 99% | **5,040 / 5,040 = 100.0000%** | **PASS** |
| ② 规则三段各 ≥50 正例与 ≥50 负例 | 3 段都要 | 3→**24** 正例、2→**118**、1→**11** | **FAIL** |

按预注册 §2.1 `S-data`：只发布重放对账与规则统计，不训练。`evidence/p0.json` 的
`verdict` 为 `S-data`，`--mode p0` 返回 rc=5。

**本轮没有产生任何 checkpoint，也没有任何评测 episode。** 没有 `train_*` / `eval_*`
文件，因为预注册的门没有过。

---

## 1. 先说两件必须纠正的前提（预注册 §0 的两条“缺陷”都判错了）

### 1.1 v8.6 的 obs 没有“丢最近子弹”，它存的是「最近 12 颗 **lethal** 子弹」

`touhou_rl` 侧的实现是：

* `v450_cf_label.compact_observation`：`live = [b for b in hazards.bullets if b.lethal]`
  → 按到玩家的**距离平方**排序 → 取前 12 → 存 `[x, y, vx, vy, half_width]`；
* `v431_arm_a.build_features`：同样的 `if b.lethal` 筛选、同样的排序、同样取前 12。

两者是同一次选择，所以 **obs 对冻结的 106 维特征是无损的**。本轮实测（纯离线，
`.tmp_v8a0_hazfix/diagnose_offline.py`）：

| 检验 | 结果 |
| --- | --- |
| 用 `v450_cf_label.features_of(obs, decision)` 重算 106 维，与行内 `features_sha256_k12` 比 | **5,040 / 5,040 全等** |
| `min(存储 12 颗距离)` vs `behaviour.nearest_lethal_px` | **4,155 / 4,155 全等**（1e-3） |
| `min(存储 12 颗距离)` vs `behaviour.nearest_all_px` | 2,312 / 4,155 相等 |
| `nearest_all_px` ≠ `nearest_lethal_px` 的行 | **1,843 / 4,155 = 44.36%** |

⇒ v8.9.0 独立复算里那条「只有 2,312/4,155 (56%) 含真最近」**不是选择缺陷、不是数据
丢失**，而是 **lethal / all 的语义差**：44.36% 的状态里有一颗**非致命**子弹比任何致命
子弹都更近。`R5` 的 `rule_gate` 用的是 `nearest_all_px`（over all bullets），
所以两者本来就不该相等。

**推论（对设计有用）**：预注册的 **B 臂**——“用最近 12 颗重算 106 维特征”——在这条
轨迹上会与 A 臂的特征**逐位相同**（`features_sha256_k12` 已验证无损），因此 B 臂不是
一个可检验的因子。

### 1.2 规则目标并非“无法离线算”

v8.6 的每一行都带 `behaviour.nearest_all_px`，它就是
`v800_bomb.nearest_bullet_distances(state, hazards)["nearest_all_px"]` —— **R5 判定的那个量**。
所以 v8.9.0 的 0 / 10 / 24 正例是**用 lethal-only 的存储 12 颗复算**得到的（同法复算得
34 个正例合计），不是“数据缺失的假象”，而是**换了一个量**在算。

本轮按预注册用 `nearest_all_px` 重算，三段仍是 24 / 118 / 11 —— **P0② 依然不成立**；
换成 lethal-only 读数则更差（合计 34 正例）。**两种读数都过不了 P0②。**

---

## 2. 重放与对账（P0 ①）

复用 `v860_bombdag` 的整套重放原语：`load_behaviour_policy`（`round2.pt` + sha256）、
`generator_seed = seed*1000+7`、`behaviour_decision`（同一状态、同一 106 维、同一
stochastic 18 路候选、同一 R5 雷位）、`start_parent`（HELLO + 2,165 checkpoint 前缀 +
PARK + DEBUG）。cell 的停止规则、每 episode 的 warm RESET + 前缀重走、2 帧决策、死亡
规则 `counterfactual._terminal_of`，全部逐行照抄 `v860_bombdag --mode collect`。

**只改一件事**：每个状态多写一行 hazard 记录。**36 路标签一个都没重采**
（`label_state36` 从未调用，没有 spawn 任何 child），标签语义不可能移动。

12 个 (seed ∈ {0,1,2}) × (window ∈ {13,14,15,16}) cell，每 cell `--quota 420`
`--decisions 300`，全部 `status=complete`，每个 cell 420 状态，合计 **5,040**。
单 cell wall 8.4–8.9 s（12 路并行，含前缀重走），前缀 hash 检查全部 0 mismatch。

| 对账项 | 结果 |
| --- | --- |
| 匹配键 (seed, window, episode, decision) | 5,040 匹配 / 0 缺失 / 0 多余 |
| `features_sha256_k12` | **5,040 / 5,040 = 1.000000**（阈值 0.99） |
| `frame` | 5,040 / 5,040 |
| `bombs_remaining` | 5,040 / 5,040 |
| 重放实测 `nearest_all_px` vs 标签 `behaviour.nearest_all_px` | **5,040 / 5,040 相等**，\|Δ\| 的 mean/median/min/max **全部 0.0** |
| 分段命中（bombs 3 / 2 / 1） | 624/624、3,378/3,378、1,038/1,038 |

v8.6.0 标签文件 sha256 `e1f47805efdd37bb92d15c51898aacbef0ca13995a200d7d2e087cd919543c68`
（与 v8.9.0 独立复算的 pin 一致，未动）。

---

## 3. 规则统计（P0 ②）

R5（`v8.4.0 BOMBRULE2-001 §1`，经 `v800_bomb.rule_gate("staged")`）：阈值
`bombs==3 → 80 px`、`2 → 60 px`、`1 → 40 px`，触发条件 `nearest_all_px < threshold`
**且** `bombs_remaining > 0`，严格小于；无子弹的状态永不触发。

| bombs | n | 正例 | 负例 | 正例率 | ≤80 px | ≤60 px | ≤40 px | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 624 | **24** | 600 | 3.8462% | 24 | 18 | 18 | ❌ 正例 < 50 |
| 2 | 3,378 | **118** | 3,260 | 3.4932% | 259 | 118 | 95 | ✅ |
| 1 | 1,038 | **11** | 1,027 | 1.0597% | 82 | 72 | 11 | ❌ 正例 < 50 |

**交叉核对（很有力）**：重放时行为策略**实际**触发了 R5 153 次，按阈值分档是
`{80 px: 24, 60 px: 118, 40 px: 11}` —— 与上表离线用记录下来的 `nearest_all_px`
重算的正例数**逐个相同**。即本文件里的规则实现与 `v800_bomb.rule_gate` 的语义一致。

### 3.1 `nearest_all_px` 分位数（px，按 `bombs_remaining`）

| bombs | n | mean | median | q1 | q3 | min | max | stdev |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 624 | 147.9903 | 154.2923 | 132.0202 | 182.0697 | 22.4827 | 195.0316 | 37.3593 |
| 2 | 3,378 | 225.5968 | 245.7441 | 150.2069 | 291.4143 | 12.1391 | 564.2277 | 94.5182 |
| 1 | 1,038 | 283.7982 | 239.2149 | 146.0968 | 412.4425 | 4.6041 | 572.7128 | 171.8117 |

（`nearest_lethal_px` 的同一张表在 `evidence/rule_target_stats.json`。）

空子弹状态 **0**，激光状态 **0**（`laser_nearest_px` 因此全为 `null`）。`player_state`：
**ALIVE 1,624 / INVULNERABLE 3,416** —— 68% 的重放状态落在某颗雷的无敌窗里，这也是为什么
`nearest_all_px` 能小到 4.6 px 而不死。

### 3.2 为什么三段正例不够（这是分布事实，不是 bug）

* `bombs=3` 只有 624 个状态：行为策略几乎立刻放雷，3 雷状态很快消失；
* `bombs=1` 的阈值只有 40 px，1,038 个状态里仅 11 个跨过（≤40 px 的行数也是 11）；
* 只有 `bombs=2`（3,378 状态、60 px）凑到 118 个正例。

⇒ 在 **这条 5,040 状态的轨迹上**，任何「bombs 3 / 2 / 1 各 ≥50 正例」的规则蒸馏设计都
过不了 P0。要过，必须先改**取样设计**（按 bombs 分层、提高 quota、或改阈值），
而不是把同样的 5,040 状态再跑一遍。

---

## 4. 门

| 门 | 结果 |
| --- | --- |
| **D3 保真门（5 条）** | **5/5 PASS**：`1_hello_role_shadow`、`2_debug_threads_1`、`3_prefix_2165_checkpoints`（2,165/2,165，0 mismatch，final frame 8,663）、`4_r24_18_constants`、`5_park_frame_8663`。本轮新跑：`v432_fork_arm.py --mode fidelity --seeds 12 --learner frozen` |
| **ckpt 断言**（start 与 final 两相） | 全过：seed0 `4abee069…`、seed1 `a93746bb…`、seed2 `c69960fc…` 逐字节相同 |
| 引擎 | `691bfd77b7eca336…` match |
| 预注册 pin | 8 条（含 `v860_bombdag.py` `3433ae74…`）全 match |
| 音频（AGENTS.md §9.3） | **14 / 14 个 `th08.cfg` 全部 muted，audible 0，decode error 0**；三个 data root 分别审计（主运行 12、fidelity gate 1、smoke 1）；`SDL_AUDIODRIVER=dummy`，`DISPLAY`/`WAYLAND_DISPLAY` 已 unset |
| 本进程残留 | **0**（`--data-dir-filter .tmp_v8a0_hazfix`；宿主上另有 61 个属于别的轮次的 `th08-modern`，未触碰） |

引擎占用：开工时宿主上有 59→74 个 `th08-modern`（9 个 v8.6.0 `--mode eval` 父进程在跑）。
按协议**等满 10 分钟**（11:50:44 → 12:00:54），复查仍有 60 个（6 个外来 eval 父进程，
load 7.35/20）才启动重放。全程未用 `pkill -f`。

---

## 5. 没做什么

* **没有训练 A 臂、没有训练 B 臂、没有任何评测 episode。** P0② 不成立 ⇒ `S-data`。
* 没有重采任何 36 路标签（重放只记录 hazard 字段）。
* 没有统计推断：本文件只有计数、比率、分位数与分布。P1–P4 属于独立分析 pass（预注册 §2）。
* 没有改 `touhou_rl/**`、`third_party/**`、任何既有版本目录或预注册，
  没有改 `v860_bombdag.py` / `v850_distill.py` / `v830_bombpol.py`（只 import），
  没有 `git add/commit`，没有删除/移动/覆盖任何资产。

---

## 6. 产物与 sha256

`Document/Update/Alpha/v8.10.0/evidence/HAZFIX/`（`manifest.json` 里是全量清单）：

| 文件 | bytes | sha256 |
| --- | --- | --- |
| `hazard_fields.jsonl` | 15,842,958 | `b6e800b392b682589f4fdc9db58d53b64823c1081d053518f309b067b57a5a93` |
| `replay_reconciliation.json` | 3,230 | `68beae160544f8ad6bafdb830006831d25e71c40d75d32b98f177678fa939cbc` |
| `rule_target_stats.json` | 4,473 | `8c81c207340831ddf5132fdc2fba9769194d3b3421a09135dcbe0856654aaf04` |
| `p0.json` | 1,065 | `b3a4db54269ac8b3503ec073bb52af919424d0311e6e9469f3577bd8cbad3bac` |
| `fidelity_d3.json` | 26,638 | `d86809c594eb73e93ba90f255994ccf2cc186919bf45467feb3968fa8f60bebf` |
| `ckpt_assertions.json` | 3,384 | `3df851107024df2a83be92941971d6b62da47db11048122452eb34e877072938` |
| `ckpt_assertions_final.json` | 3,384 | `465695f6e3d392adba39ced44706d2fd47b206883a6cab44f7a5f7b24cf81df3` |
| `preflight.json` | 3,747 | `17d5a448fabbaf528440d23a1f025105c33f311da7bc4c2c96c8f04a560a83dd` |

`hazard_fields.jsonl` 一行一个状态（5,040 行），字段：
`nearest_all_px` / `nearest_index` / `bullet_count` / `nearest_lethal_px` /
`nearest_lethal_index` / `nearest12`（按距离升序，含 `x, y, vx, vy, half_width, slot,
state, lethal, dist_px`）/ `player_state` / `player_state_name` / `invulnerable` /
`laser_count` / `laser_in_use` / `laser_nearest_px` / `features_sha256_k12` /
`behaviour`（cand / bomb / action / threshold_px）/ `frame` / `bombs_remaining` /
`player_x` / `player_y` / `seed` / `window` / `episode` / `decision`。

12 个 cell 的 `replay/seed*_w*.json`（每 cell 420 状态、gate、cfg、wall_s）一并发布。

**工作目录 `.tmp_v8a0_hazfix/` 保留**：`replay/`（12 cell jsonl + summary）、
`evidence/`、`logs/`（12 个 cell 的 stdout + `run.log`）、`commands.tsv`（每条命令的
exit / wall / argv）、`exec.sh`、`smoke/`（6 状态冒烟，隔离工作目录）、
`diagnose_offline.py` / `diagnose_offline.json`（§1 的离线诊断）。**没有新 ckpt。**

---

## 7. 偏差清单

1. **未训练、未评测**：P0② 不成立，按预注册 §2.1 `S-data` 停止。不是执行失败，是门不成立。
2. **预注册 §0 的两条缺陷诊断与实际不符**（§1）：obs 选择无损（5,040/5,040 特征哈希一致），
   规则目标本来就可离线算（`behaviour.nearest_all_px` 在每一行里）。已给出可复算的替代
   解释；预注册未改一字。
3. **B 臂不是可检验因子**：既然 obs 无损，B 臂的 106 维特征会和 A 臂逐位相同。
4. 重放开跑时宿主仍有 60 个外来 `th08-modern`；已按协议等满 10 分钟，未 `pkill -f`，
   未碰别人的进程/资产。
5. `--mode rule-stats` 额外记录了 `nearest_lethal_px` 与 ≤80/≤60/≤40 的行数用于并排比对；
   R5 判定**只用** `nearest_all_px`。
6. 12 路并行（预注册 §3 的估算），单 cell wall 8.4–8.9 s，比估算的 5 min 快得多——
   因为重放不跑 36 路标签（不 spawn child）。
7. P0 判定用 `--mode p0` 返回 rc=5（非零）作为门失败的机器可读信号；`exec.sh` 因此
   在 `p0` 阶段以 rc=5 收尾，是**预期行为**，不是脚本错误。

---

## 8. 不能声称什么

* 不能声称「规则蒸馏可行」或「规则蒸馏不可行」——本轮**没有训练任何模型**，
  P0 在进入训练之前就不成立；§3.2 说的是**这条取样轨迹**上三段正例不够。
* 不能声称「v8.6 的 obs 有 bug」——反过来，本轮证明了它对 106 维特征无损。
* 不能声称「最近子弹已补回，所以观察退化被修复」——观察本来就没退化；本轮补的是
  **记录字段**（`nearest_all_px`、最近 12 颗的距离/速度、`player_state`）。
* 不能声称「通关」；不能把不同预算的均值直接比较；本轮没有 episode 级结果。
