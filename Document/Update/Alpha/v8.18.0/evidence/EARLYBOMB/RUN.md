# v8.18.0 EARLYBOMB-001 — RUN

**预注册**：`Document/Update/Alpha/v8.18.0/PREREGISTRATION_EARLYBOMB_001.md`（sha256 `d4bbb00946ce184914981d306ea540288c39cf06d3d6b77c2d7cb7737f00a57e`，4,301 B，FROZEN）
**单因子**：采样位置（相对死亡的提前量）；其余全部冻结（`S` + 不放雷、900 决策死亡即停、36 路 × H=120/240/600、A/B 两臂）。
**主机**：训练端（WSL2，`/root/v400-venv`，20 核 / 23 GB；RTX 4070 SUPER 未使用）
**生产者**：`tools/rl/v8j0_earlybomb.py`（唯一新增代码。`touhou_rl/**`、`third_party/**`、既有版本目录/预注册/证据、`v8i0_nobomb.py` 一律未改；无 `git add/commit`）
**判定**：`S-2bomb` →（按 `player_state == ALIVE` 过滤）→ **四门全过**，`train_horizon = H=600`，A/B 各 3 种子已训练并完成 18 组评测。

---

## 0. 本轮最重要的实测：死亡检测滞后 = 27 决策 = 54 帧

v8.17 的 600 个状态全是 `DYING`，但它没有测过「DYING 从哪里开始」。本轮把 `player_state` **逐决策**记进 pass 1 的轨迹，分界线是直接测出来的：

| 量 | 实测值 | 来源 |
| --- | --- | --- |
| 死亡被检测到的决策 | `death_decision = 125`（ep0）、`550`（ep1） | 冒烟 s0/w12 |
| 最后一个 `player_state == ALIVE` 的决策 | **98** | 同上 |
| **死亡检测滞后** | **27 决策 = 54 帧（≈1.8 s @30 fps）** | `death_decision − last_ALIVE` |
| `before_death` **27–40**（54–80 帧） | 全 **`ALIVE`**，`bomb=1` 分支 **`spent = 1`**（18/18 每状态） | 冒烟 21 状态 |
| `before_death` **20–26**（40–52 帧） | 全 **`DYING`**，`bomb=1` 分支 **`spent = 2`**（18/18 每状态） | 同上 |

⇒ **「普通放雷（扣 1 颗）」与「死亡炸弹（扣 2 颗）」的精确分界 ≈ 死前 54 帧（27 决策）。**

这条同时解释了 v8.17 的结局：它的采样窗是**最后 20 决策 = 40 帧**，整窗落在 54 帧滞后之内 ⇒ 100 % `DYING` ⇒ 每次按雷扣 **2** 颗。**不是标签错了，是采样窗落在了死亡动画内部。**

全量数据（447 状态）复核该分界是**结构性的**、不是单条 episode 的偶然：

| 池 | 状态数 | `ALIVE` | `spent == 1` 分支 | `spent == 2` 分支 | P0c |
| --- | ---: | ---: | ---: | ---: | ---: |
| 整窗（`before_death ∈ [20,40]`） | 447 | **314 (70.2 %)** | 5,652 (70.2 %) | **2,394 (29.8 %)** | 70.2 % |
| 仅 `ALIVE` | 314 | 314 (100 %) | **5,652** | **0** | **100 %** |
| 仅 `DYING` | 133 | 0 | 0 | 2,394 | **0 %** |

`spent == 1` 与 `ALIVE` **完全共变**（70.2461 % vs 70.2461 %，逐分支无例外）：引擎的扣雷数就是 `player_state` 的函数。

---

## 1. 设计（单因子）

| 项 | v8.17.0 | **v8.18.0** |
| --- | --- | --- |
| 行为策略 | `S` + 不放雷 | **不变**（`--bomb-rule none`，bomb 恒 0） |
| 轨迹预算 | 900 决策，死亡即停 | 不变（不死则丢弃；连续 10 条不满足则跳过该 cell） |
| 采样位置 | 最后 20 决策（40 帧） | **`decisions_before_death ∈ [20, 40]`（40–80 帧）** |
| 每状态标签 | 36 路 × H=120/240/600 | 不变（一次跑 H=600，`alive(H)=min(alive(600),H)`；冒烟直接验过 H=120/240：0 mismatch） |
| 目标状态数 | 600 | 600（实际标注 447 / `ALIVE` **314**，见 §6） |
| 训练 | A/B × 3 种子 × 200 epoch | 不变（`H=600`） |

**采样窗为什么是 `[20,40]` 而不是预注册的 `[30,60]`**：预注册 §1 写的是 `[30,60]`（60–120 帧）。本轮先用它跑冒烟，实测 `decisions_before_death ∈ [41,60]` 的状态在 **H=120 的 diff = 0/7**——不是数据坏了，而是**逻辑必然**：`bomb = 0` 从 82–120 帧以外看还能活过 120 帧，H=120 当然测不到雷的价值。按工单收窄到 `[20,40]`（40–80 帧），H=120 的 diff 立刻变成可测。**先测量、后收窄**；原始 `[30,60]` 的部分标注留档为 `smoke/labels_smoke_seed0_w12_EARLY30-60.partial`。

**`min_death_decision` 守卫（新增，工单要求）**：窗口是相对死亡的，「死得太早」会把窗口退化成 prefix 后最初的几十帧。冒烟实测到一次：`DIED at decision 45` ⇒ `[30,60]` 被裁成决策 `0..15`（prefix 后 0.5 s），那里 `bomb=0` 当然活过 H=120 ⇒ diff 是结构性 0。因此**死亡决策 < 300 的 episode 丢弃**（工单允许放宽到 120；实际用到 `{120, 300}` 两档，见 §6）。

---

## 2. P0a–P0d（判定按预注册 §2.1 的 `S-2bomb`）

`--mode stats`：447 状态 / 1,341 行，8 个 cell 有状态（12 个 cell 目录全部落盘）。

| 判据 | 整窗（447） | **`ALIVE` 池（314，判定用）** | 门 | 结果 |
| --- | --- | --- | --- | --- |
| **P0a** `player_state == ALIVE` | **314/447 = 70.2 %** | **314/314 = 100 %** | ≥ 90 % | 整窗 FAIL → 过滤后 **PASS** |
| **P0b** `bombs_remaining > 0` | 447/447 = 100 %（`{3: 447}`） | **314/314 = 100 %** | ≥ 50 % | **PASS** |
| **P0c** honoured 分支里 `spent == 1` | 5,652/8,046 = 70.2 % | **5,652/5,652 = 100 %** | ≥ 90 % | 整窗 FAIL → 过滤后 **PASS**（**与 v8.17 的判别式**） |
| **P0d** H=120 | 18/447 = 4.0 % | **18/314 = 5.73 %** | ≥ 20 % | fail |
| **P0d** H=240 | 50/447 = 11.2 % | **50/314 = 15.92 %** | ≥ 20 % | fail |
| **P0d** H=600 | 227/447 = 50.8 % | **227/314 = 72.29 %** | ≥ 20 % | **PASS** |

- **判定**：`S-2bomb` → 按 `player_state == ALIVE` 过滤（预注册 §2.1 明文处方），四门在 `ALIVE` 池上全过（P0d 由 **H=600** 达门）⇒ **`train`，`train_horizon = 600`**（最小达门视界）。整窗与 `ALIVE` 池两套 P0 都写进 `earlybomb_stats.json`（`p0_all` / `p0_alive`）。
- **P0d 的形状本身就是结论**：雷的价值只在长视界出现——H=120 **5.73 %**、H=240 **15.92 %**、H=600 **72.29 %**。在「距死 40–80 帧且仍 ALIVE」的位置，`bomb = 0` 的 18 个候选里**总有一个**能再活 2–4 秒（H=120/240 的 critical = **0/314**），但活不到 10 秒。**雷买到的是「再活 3–10 秒」，不是「多活一口气」。**
- `delta` 分位（ALIVE 池，H=600）：均值 333.8、中位 199、q1 187、q3 550、min 176、max 560（冒烟单 episode 口径）；全量 H=600 critical = 63/314 = 20.1 %。

---

## 3. 训练 A/B（`train_summary.json` 读数）

`Policy(106,64,36)`、Adam 1e-3、batch 256、**200 epoch**、无早停、dev 全程未读。**样本 = 314 个 `ALIVE` 状态**（`labels_earlybomb_alive.jsonl`，sha256 `305639df…`），**H = 600**（判定给出的最小达门视界）。A = `BCE(36) + 0.5·BCE_bomb(best-bomb 目标)`；B = 仅 `BCE(36)`。aux 正例 **262/314 = 83.4 %**。

| 臂 | seed | 首/末 epoch loss | train BCE | train acc | aux acc | ckpt sha256（前 16） |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| A | 0 | 1.0506 → 0.3291 | 0.2196 | 0.9061 | 0.8822 | `4d96a536448aa9e8` |
| A | 1 | 1.0385 → 0.3308 | 0.2208 | 0.9056 | 0.8917 | `b4675badee558b04` |
| A | 2 | 1.0671 → 0.3292 | 0.2244 | 0.9038 | 0.8853 | `df676ef4bdb86177` |
| B | 0 | 0.6980 → 0.2123 | 0.2122 | 0.9140 | 0.8344 | `d6b0681e4cd7214d` |
| B | 1 | 0.7037 → 0.2107 | 0.2105 | 0.9139 | 0.8344 | `44ea6ac1ba554b09` |
| B | 2 | 0.7125 → 0.2096 | 0.2094 | 0.9153 | 0.8344 | `8bc5b1a9bd4c2981` |

（B 臂的 aux acc 只是未训练的边际读数，不是监督目标。6 个 ckpt 各 39,281 B。完整 sha256 见 `train_summary.json`。）

---

## 4. 评测（18 组 = 2 臂 × 3 档 × 3 种子 × 4 窗 × 300 决策）

全部 **18/18 `status=complete`、`episodes_total=1200`**；18 组各自私有 data 目录 + 静音 `th08.cfg` + 唯一 tag（v8.17 的 10/18 失败模式是把 18 个进程塞进同一个 timestamp tag）。**总 21,600 个 episode。**

### 4.1 三档 × 池化（4 窗 × 3 种子 × 300 = 3,600 episode/臂/档）

| 臂 | 决策 | episodes | 活满 | 活满率 | Wilson 95 % CP | 用雷总数 | 平均用雷/ep |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| **A** | 300 | 3600 | 3592 | **99.78 %** | 99.56 % – 99.89 % | 10800 | 3.000 |
| **A** | 600 | 3600 | 1042 | **28.94 %** | 27.49 % – 30.45 % | 10800 | 3.000 |
| **A** | 900 | 3600 | 151 | **4.19 %** | 3.59 % – 4.90 % | 10800 | 3.000 |
| **B** | 300 | 3600 | 3596 | **99.89 %** | 99.71 % – 99.96 % | 10800 | 3.000 |
| **B** | 600 | 3600 | 1120 | **31.11 %** | 29.62 % – 32.64 % | 10800 | 3.000 |
| **B** | 900 | 3600 | 180 | **5.00 %** | 4.33 % – 5.76 % | 10800 | 3.000 |

### 4.2 逐窗口（池化三个种子，每格 900 episode）

| 臂 | 决策 | w11 | w12 | w17 | w18 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **A** | 300 | 99.89 % | 99.33 % | 100.00 % | 99.89 % |
| **A** | 600 | 30.44 % | 18.56 % | 35.89 % | 30.89 % |
| **A** | 900 | 3.00 % | 3.44 % | 5.44 % | 4.89 % |
| **B** | 300 | 99.89 % | 99.67 % | 100.00 % | 100.00 % |
| **B** | 600 | 30.44 % | 26.67 % | 34.78 % | 32.56 % |
| **B** | 900 | 3.22 % | 5.33 % | 7.00 % | 4.44 % |

### 4.3 逐种子（池化四个窗口，每格 1,200 episode）

| 臂 | 决策 | seed 0 | seed 1 | seed 2 |
| --- | ---: | ---: | ---: | ---: |
| **A** | 300 | 100.00 % | 99.42 % | 99.92 % |
| **A** | 600 | 27.58 % | 29.50 % | 29.75 % |
| **A** | 900 | 3.50 % | 4.25 % | 4.83 % |
| **B** | 300 | 99.92 % | 99.83 % | 99.92 % |
| **B** | 600 | 31.92 % | 31.00 % | 30.42 % |
| **B** | 900 | 4.83 % | 5.00 % | 5.17 % |

### 4.4 用雷分布（P3：出现「只用 1–2 颗」的 episode？均值 < 3.00？）

| 臂 | 决策 | episodes | 用 0 颗 | 用 1 颗 | 用 2 颗 | 用 3 颗 | 平均 | 中位 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **A** | 300 | 3600 | 0 | 0 | 0 | **3600** | **3.000** | 3 |
| **A** | 600 | 3600 | 0 | 0 | 0 | **3600** | **3.000** | 3 |
| **A** | 900 | 3600 | 0 | 0 | 0 | **3600** | **3.000** | 3 |
| **B** | 300 | 3600 | 0 | 0 | 0 | **3600** | **3.000** | 3 |
| **B** | 600 | 3600 | 0 | 0 | 0 | **3600** | **3.000** | 3 |
| **B** | 900 | 3600 | 0 | 0 | 0 | **3600** | **3.000** | 3 |

**P3 不成立**：21,600 个 episode 里**没有一个是「只用 1–2 颗」**，平均用雷 **3.000**（= 上限），中位 3。

### 4.5 并排（活满率；n 为 episode 数）

| 方案 | 300 决策 | 600 决策 | 900 决策 |
| --- | ---: | ---: | ---: |
| v8.3.0 BOMBPOL（S，不放雷） | 99.94 % | 19.44 % | 0.61 % |
| v8.6.0 BOMBDAG（S + R5 规则监督） | 99.94 % | **29.58 %** | 2.94 % |
| v8.17.0 CONSERVE（最后 20 决策采样） | — | **14.10 %** | — |
| v8.4.0 R5 脚本规则（非学习） | 100.00 % | 77.11 % | 17.61 % |
| v8.12.0 | 98.75 % | — | — |
| **v8.18.0 臂 A（H=600 训练，40–80 帧采样）** | **99.78 %** (n=3600) | **28.94 %** (n=3600) | **4.19 %** (n=3600) |
| **v8.18.0 臂 B（H=600 训练，40–80 帧采样）** | **99.89 %** (n=3600) | **31.11 %** (n=3600) | **5.00 %** (n=3600) |

### 4.6 判据（预注册 §2）

| # | 判据 | 实测 | 判定 |
| --- | --- | --- | --- |
| **P1** | 臂 B 的 600 决策活满率 **> 19.44 %**（v8.3）且 **> 14.1 %**（v8.17） | **31.11 %**（CP 29.62–32.64），n=3600 | **成立**（+11.67 pp / +17.01 pp） |
| **P2** | 臂 A 与 B 并排 | A 28.94 % / B 31.11 % | A 低于 B **2.17 pp**；两者都远超 v8.3 / v8.17 |
| **P3** | 出现「只用 1–2 颗」的 episode **且** 平均用雷 < 3.00 | 1–2 颗：**0/21,600**；平均 **3.000** | **不成立** |
| **P4** | 900 决策与 v8.3（0.61 %）、v8.6（2.94 %）、R5（17.61 %）并排 | A 4.19 % / B 5.00 % | 高于 v8.3 / v8.6，远低于 R5 脚本规则 |

**A/B 差**：B（无 aux）**优于** A（有 aux）2.17 pp（600 决策；A 27.49–30.45 % 与 B 29.62–32.64 % 的 Wilson 区间**重叠**，因此这只是描述性差异，不是显著性结论）。aux 目标在 H=600 上 83.4 % 为正，等价于「只要有机会就放雷」，它把策略往「更早、更频繁按雷」推——这与 P3（两臂都 3.000）一致，但没有带来存活收益。

**与工单预期的对照**：预期「训练在 H=600 ≡ 用光三颗雷 ≡ v8.6 附近（≈29.58 %），不会超过 R5 的 77.11 %」。实测 **A 28.94 % / B 31.11 %**，落在 v8.6 的 29.58 % 两侧；用雷 **3.000**（用光）；900 决策 4–5 %（远低于 R5 的 17.61 %）。**预期成立**：在「距死 40–80 帧」处，H=600 的标签几乎总是「`bomb=1` 更优」，于是学到的仍是「一有机会就放」，而不是「省着用」。

---

## 5. 保真门 / 静音 / 进程

- **D3 保真门 5/5 PASS**（`--mode fidelity` 把门委托给 `v432_fork_arm.py --mode fidelity`：`rc=0`，`assertions=5`，`all_passed=True`，`d3_all_pass=True`，17.7 s）。每窗口的 gate 另记在 `collect/seed*_w*.json` 的 `gate`（`hello_role_shadow` / `park_ok` / `park_frame=8663` / `threads_1` / prefix `2165/2165` checkpoints、0 mismatch）。
- **ckpt 断言**：`ckpt_assertions.json`（start）与 `ckpt_assertions_final.json`（final）**all_pass=True**；`round2.pt` = `4abee069…` / `a93746bb…` / `c69960fc…`；引擎 sha256 `691bfd77…`。
- **静音（AGENTS.md §9.3）**：每个私有 data 目录由 `prepare_data_dir` 现生成静音 `th08.cfg`，评测逐窗口断言 `cfg.muted_four_zero`（失败即 refuse to start）；`v400_make_cfg.py --inspect` 复核 `musicMode=0, playSounds=0, musicVolume=0, sfxVolume=0`（60 B，`problems: []`）。`--mode audio-audit`：**89 个 cfg、89 个 muted、0 audible**。`SDL_AUDIODRIVER=dummy`，`DISPLAY`/`WAYLAND_DISPLAY` 已 unset。
- **进程**：`--mode process-audit --data-dir-filter .tmp_v8j0_earlybomb` → **0** 个本轮引擎进程残留。

---

## 6. 偏差（如实记录）

1. **采样窗 `[20,40]` 而非预注册的 `[30,60]`**（工单指示，2026-11-22）：`[30,60]` 下 H=120 的 diff 结构性为 0（实测 7 状态：H=120 0/7、H=240 0/7、H=600 6/7）。**先测量后收窄**，原始标注留档 `smoke/labels_smoke_seed0_w12_EARLY30-60.partial`。
2. **`min_death_decision` 守卫**：两档实测（`{120, 300}`）。4 个 cell（`s0w18`、`s1w11`、`s1w17`、`s2w18`）在 300 与 120 两档下都是「连续 10 条早死」⇒ **0 状态**；`s0w11`、`s1w18`、`s2w12` 用 120 档各拿到 21/21/42 状态。全轮 `early_death_episodes`（被丢弃的早死 episode）= **131** 条。每 cell 的 `death_decision` 序列在 `collect/seed*_w*.json` 的 `episodes[].death_decision` 与 `death_decisions_rejected`。
3. **状态数 447（`ALIVE` 314），未达 600**：`death ≥ 300` 在 v8.17 同策略的 36 条死亡 episode 里只有 5/36 = 13.9 %；工单明确「短额就按实际 n 训练，不要补采到失真」，故按 **314** 训练（`train_summary.json` 的 `states`）。
4. **`S-2bomb` 过滤**：判定用 `player_state == ALIVE` 池（314）；整窗（447）的 P0 同时落盘（`p0_all` / `p0_alive` / `by_player_state`）。
5. **label 成本 7.3 s → 12–15.7 s/状态**：7.3 s 是**低竞争**观测（冒烟时 1 个 parent）；全量采集时本机同时有 v8.17 的 18 组评测在跑（峰值 **174** 个 `th08-modern` / 20 核），单状态标注升到 12.2–15.7 s。**并发导致，不是代码变慢**（36 个 child 仍串行，`max_concurrent_children = 1`）。
6. **一次评测组的瞬时失败并重试成功**：`arm B d600 seed 1` 首次启动时 `FidelityError: window 11: reset hash 'af6314501cf1a19e' != recorded '7c8705fc88f94c17' (§2.3 first checkpoint)`（当时 123 个引擎在跑）。该组已被丢弃并**重跑**（`--tag v8j0-eval-B-d600-s1-retry`），重跑完成为 `status=complete, episodes_total=1200, survived=372`，与前一次失败无数据混用。
7. 冒烟第一次（`window=auto`，死亡决策 45）被工单判为退化窗后**主动中止**，其 5 个状态的部分标注留档 `smoke/labels_smoke_seed0_w11.jsonl.partial`。
8. **证据体积 361 MB**：18 组评测的 `*.jsonl` 每行含 `actions_sampled`（300/600/900 个整数），是体积主因。若需精简可只保留 `*.summary.json`（各 6.3 KB）与 `diagnostics.json`。

---

## 7. 不能声称什么

- **不能声称「学会了资源分配」**：A 臂含规则先验（best-bomb aux），B 臂无；P3 明确失败（两臂都用光 3 颗、0 个 episode 只用 1–2 颗）。
- **不能把本轮与 v8.17 直接比较而不说明采样位置不同**：v8.17 采最后 40 帧（全 DYING、扣 2 颗），本轮采 40–80 帧（ALIVE、扣 1 颗）。这是**同一单因子的两个水平**，不是同一实验的两次重复。
- **不能声称跨窗口泛化**：训练/评调用同一批 dev 窗（11/12/17/18，窗之间大量重叠的固定 prefix），逐窗数字见 §4.2，w12 明显最差（A 18.56 %）。
- **不能声称「通关」**：30 s 档最好也只到 **31.11 %**，仍远低于 R5 脚本规则的 77.11 %。
- **不能把小样本 A/B 差当显著性**：A 与 B 的 600 决策 Wilson 区间重叠；A/B 差异只是描述性的。
- **不能把 900 决策的 5.00 % 与 R5 的 17.61 % 相比**：前者是纯学习策略，后者是脚本规则（含距离先验）。
- **不能把「H=600 训练」的结果外推为「提前放雷有用」**：H=120 的 diff 只有 5.73 %，本轮的存活收益来自「更早按雷 + 长视界」，不是「省雷」。

---

## 8. 产物与身份

`Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/`（77 个文件，manifest 见同目录 `manifest.json`）

| 文件 | 说明 |
| --- | --- |
| `summary.json` | 一页汇总（P0 / train / 18 组评测 / 产物 sha256） |
| `RUN.md` | 本文件 |
| `earlybomb_stats.json` | P0a–P0d，整窗 + `ALIVE` 池两套 |
| `labels_earlybomb.jsonl` / `labels_earlybomb_alive.jsonl` | 447 状态的 1,341 行 / 314 状态的训练池 |
| `smoke.json` | 冒烟：逐决策 `player_state` 表、滞后、P0、三视界 diff |
| `diagnostics.json` / `bomb_usage.json` | P1–P4 的描述性读数（**无检验、无区间**，检验属独立分析轮） |
| `train_A/B_earlybomb_seed*.json` / `ckpt_A/B_earlybomb_seed*.pt` / `train_summary.json` | 训练报告与权重 |
| `eval_A/B_earlybomb_d{300,600,900}_seed{0,1,2}.jsonl` + `.summary.json` | 18 组评测（21,600 episode） |
| `collect/seed*_w*.json`（+ `collect/firstpass_min300/`） | 12 个 cell 的采集摘要（含 `min_death_decision=300` 第一轮被跳过的证据） |
| `preflight.json` / `ckpt_assertions*.json` / `fidelity_d3*.json` / `audio_audit.json` / `process_residue.json` / `commands.tsv` | 门与审计 |

关键 sha256（完整清单见 `manifest.json`）：见 §9。

---

## 9. 复现命令（摘要）

```bash
# 1) 门
python tools/rl/v8j0_earlybomb.py --mode preflight
python tools/rl/v8j0_earlybomb.py --mode assert-ckpt --phase start
python tools/rl/v8j0_earlybomb.py --mode fidelity                 # D3 5/5
# 2) 冒烟（1 条死亡 episode，自动选窗 + H 嵌套验证）
python tools/rl/v8j0_earlybomb.py --mode smoke --seed 0 --windows 12 \
  --window tight --min-death-decision 300
# 3) 采集（12 cell；先 300 档，被守卫清空的 7 个 cell 再用 120 档）
bash .tmp_v8j0_earlybomb/run_collect.sh
bash .tmp_v8j0_earlybomb/run_collect_min120.sh
# 4) P0 + 训练 + 评测
python tools/rl/v8j0_earlybomb.py --mode stats
bash .tmp_v8j0_earlybomb/run_train.sh
bash .tmp_v8j0_earlybomb/run_eval.sh
# 5) 收尾
python tools/rl/v8j0_earlybomb.py --mode assert-ckpt --phase final
python tools/rl/v8j0_earlybomb.py --mode diagnostics
python tools/rl/v8j0_earlybomb.py --mode audio-audit
python tools/rl/v8j0_earlybomb.py --mode process-audit --data-dir-filter .tmp_v8j0_earlybomb
python tools/rl/v8j0_earlybomb.py --mode publish
```
