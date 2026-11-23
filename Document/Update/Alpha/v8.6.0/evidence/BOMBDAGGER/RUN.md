# v8.6.0 BOMBDAGGER-001 — 运行记录（RUN）

预注册：`Document/Update/Alpha/v8.6.0/PREREGISTRATION_BOMBDAGGER_001.md`（FROZEN，2026-11-22）
执行：训练端 DeepSeek 执行代理，2026-11-22 11:11–12:2x（Asia/Shanghai）
状态：**完成**。结论分支按预注册 §2.1 为 **S-partial**（P1 成立但 < 50 %），且 **P3 未出现**——
详见 §7/§10 的解释与对本轮机制假设的反证。

---

## 0. 一句话结论

行为策略换成「v4.5.0 `S` + R5 放雷规则」后，标签集第一次真正覆盖了 `bombs_remaining ∈ {3,2,1}`
（5,040 状态：3→624、2→3,378、1→1,038），**P0 以 16.9× / 5.2× 的余量通过**；同一套 36 路反事实标签、
同一训练配方（`Policy(106,64,36)`、BCE、Adam 1e-3、200 epoch、无辅助损失）下，600 决策（20 s）活满率从
v8.3.0 的 **19.44 % 升到 29.58 %**（+10.14 pp，单侧 Fisher p = 4.98e-07，CP95 下限 28.10 % > 19.44 %），
**P1 成立**；但 **P2（≥50 %）不成立**（29.58 %，距 R5 规则 77.05 % 还差 47.47 pp），
**P3 不成立**（10,800 个 episode 里 **0 个**只用 1–2 颗雷，全部用满 3 颗）。

⇒ 「换状态分布」**提高了存活**，但**没有教会省雷**。原因不是数据量，而是标签信号本身（§8）：
`bomb=1` 在 99.92 % 的状态下是 safe，且在无敌窗口内按雷与不按**完全等价**（引擎忽略、不扣雷），
因此「见雷就按」是 BCE 的最优解。

---

## 1. 上一轮为什么中止，这次改了什么

上一任代理的采集在 12 格里有 11 格于 **decision 1** 中止，报

```
BombDagError: w13 d1 a1: bomb=1 with 2 bombs left did not spend one
```

**引擎语义（本轮复核为定论，不再另作解释）**：**雷不可叠加**。上一颗雷的无敌窗口（`player_state = 3
PLAYER_STATE_INVULNERABLE`）内再按 `BTN_BOMB` 会被引擎**忽略**，**不扣雷**。
所以「`bomb=1` 必须扣一颗」这条不变量在无敌窗口内**不成立**；decision 0 放了雷，decision 1 的整格状态
全部落在无敌窗口里，18 条 `bomb=1` 分支全部是 no-op ⇒ 旧不变量必然在 decision 1 触发。

本轮对 `tools/rl/v860_bombdag.py` 的判定逻辑改成**两分支**（任一成立即通过）：

| 分支 | 条件 | 语义 |
| --- | --- | --- |
| (a)  honoured | `bombs_after == bombs_before - 1` | 真的扣了一颗 |
| (b)  no-op | `bombs_after == bombs_before` **且** 该分支 `alive_frames` 与同候选 `bomb=0` 分支**完全相同** | 无敌窗口内按与不按等价 |

两者都不成立 ⇒ **只记一条 `anomaly`，不中止整格**。`noop_due_to_invuln=True` 的行**全部保留**
（它们是合法的反事实对）。

每行新增字段（本轮实测全部非空，示例见 §2.3）：

| 字段 | 含义 |
| --- | --- |
| `bomb_spent` (bool) | 行为策略实际执行的那条分支是否真的扣雷（`bombs_after == bombs_before-1`） |
| `noop_due_to_invuln` (bool) | 该分支是否「按钮被忽略且与 bomb=0 兄弟同生同死」 |
| `bombs_before` / `bombs_after` (int) | 该分支被标签时的雷数 / 120 帧后的雷数 |
| `bomb1_noop_count` / `bomb1_anomaly_count` / `bomb1_anomaly_cands` / `bomb1_branch_kind` | 该状态 18 条 `bomb=1` 分支的逐条判类与聚合 |
| `bomb1.{bombs_before,bombs_after,alive_frames,alive_frames_bomb0,bomb_spent,alive_equal_to_bomb0,noop_due_to_invuln,branch_kind,anomaly_count,anomaly_cands}` | 18 条 bomb=1 分支的完整逐条记录 |

`labels_stats.json` 同时报 **两种 noop 口径**（见 §3）与 `anomaly_count`。

---

## 2. 采集

### 2.1 规模、命令、耗时

12 格 = seeds {0,1,2} × train windows {13,14,15,16}，**每格 `--quota 420`**，`--decisions 300`（冻结协议，
每格 ep0 走满 300 决策 = 300 状态，ep1 从新前缀再补 120 状态；`--max-episodes 10` 未用满）。

```
./exec.sh collect          # 11:16:14 -> 11:39:04，12 进程并行，全部 rc=0
./exec.sh stats            # 11:39:08，rc=0，P0=PASS
```

| 项 | 值 |
| --- | --- |
| 状态数 | **5,040**（12 格 × 420，全部 `status=complete`） |
| 标签条数 | **181,440**（5,040 × 36） |
| episode 数 | 24（每格 2 个） |
| 采集墙钟 | **22 min 50 s**（12 格并行；单格 1,239.8–1,292.0 s） |
| 每状态标签耗时 | 单格均值 2.93–3.05 s（`label_cost_s.mean`） |
| 子进程完整性 | `spawn_ok = quit_ok = baseline_frame_ok = baseline_state_equal_parent = 181,440/181,440` |
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`（sha256 见 `preflight.json`） |

采集期的 `label_checks`（12 格汇总，`collect/*.json`）：

```
labels_36                          5040        labels_missing 0
bomb1_press_honoured              42066
bomb1_press_ignored               48654   (= bomb1_noop_due_to_invuln)
bomb1_anomalies                       0        states_with_bomb1_anomaly 0
bomb1_unspent_alive_differs           0
bomb0_spent_a_bomb                    0
states_all_bomb1_noop              2703        states_some_bomb1_noop 0
states_no_bomb1_noop               2337
states_without_a_bomb_left            0
```

`states_some_bomb1_noop = 0` 而 `all_noop = 2,703`、`no_bomb1_noop = 2,337`：
**同一状态内 18 条 bomb=1 分支同生同死**，没有中间态（§3.3 的退化性）。

### 2.2 P0：`bombs_remaining` 分布 —— **PASS**

来自 `labels_stats.json`（`--mode stats`，rc=0）：

| `bombs_remaining` | 状态数 | 占比 | P0 门槛 |
| --- | --- | --- | --- |
| 3 | 624 | 12.381 % | — |
| **2** | **3,378** | 67.024 % | ≥ 200 ✓（16.9×） |
| **1** | **1,038** | 20.595 % | ≥ 200 ✓（5.2×） |
| 0 | 0 | — | — |

对比 v8.3.0：`{3.0: 5166}`（5,166/5,166 全是 3 颗）。**本轮的单一改动确实把 2 与 1 带进了标签集。**

每格分布（`per_cell`，`{3, 2, 1}`）：

```
s0_w13 {2,269,149}  s0_w14 {2,418,-}  s0_w15 {2,311,107}  s0_w16 {176,207,37}
s1_w13 {2,271,147}  s1_w14 {2,371,47}  s1_w15 {2,305,113}  s1_w16 {259,161,-}
s2_w13 {2,267,151}  s2_w14 {2,312,106} s2_w15 {2,283,135}  s2_w16 {171,203,46}
```

（w14 的第 3 颗雷在这 420 个状态里没有放掉，w16 的第 1 颗雷要等到 decision 84–141 才按，故其 3 颗状态占比高：
w16 三格合计 606 个 `bombs=3`。）

### 2.3 每行字段示例（`labels_bombdag.jsonl` 第 1–2 行）

```
{'bombs_before': 3, 'bombs_after': 2, 'bomb_spent': True,  'noop_due_to_invuln': False,
 'bomb1_noop_count': 0,  'bomb1_anomaly_count': 0, 'bombs_remaining': 3, 'decision': 0}
   behaviour: cand=16 bomb=1 threshold=80px   invuln: ALIVE
{'bombs_before': 2, 'bombs_after': 2, 'bomb_spent': False, 'noop_due_to_invuln': True,
 'bomb1_noop_count': 18, 'bomb1_anomaly_count': 0, 'bombs_remaining': 2, 'decision': 1}
   behaviour: cand=13 bomb=1 threshold=60px   invuln: INVULNERABLE
```

---

## 3. bomb=1 不变量：两种口径 + 引擎事实

`labels_stats.json` 顶层与 `bomb1_invariant` 内同时给出：

| 指标 | 值 | 定义 |
| --- | --- | --- |
| `noop_frac` | **0.536310** | noop 分支 / (18 × 状态) —— **bomb=1 分支口径**（向后兼容名） |
| `noop_branch_frac` | **0.268155** | noop 分支 / (36 × 状态) —— **36 路动作分支口径** |
| `noop_state_frac` | **0.536310** | 「至少一条 bomb=1 分支是 noop」的状态 / 状态 |
| `noop_state_frac_all_18` | **0.536310** | 「18 条 bomb=1 分支全 noop」的状态 / 状态 |
| `anomaly_count` | **0** | 两分支都不成立的分支数（`anomaly_kinds = {}`） |
| `bomb1_invariant.spent` | 42,066 | honoured 分支 |
| `bomb1_invariant.noop_due_to_invuln` | 48,654 | no-op 分支 |
| `branches` | 90,720 | 5,040 × 18 |

按 `bombs_remaining` 分层：

| 层级 | 状态 | 分支 | honoured | noop | `noop_frac`(18路) | `noop_state_frac` | anomaly |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 624 | 11,232 | 11,232 | 0 | 0.000000 | 0.000000 | 0 |
| 2 | 3,378 | 60,804 | 25,794 | 35,010 | 0.575784 | 0.575784 | 0 |
| 1 | 1,038 | 18,684 | 5,040 | 13,644 | 0.730250 | 0.730250 | 0 |

### 3.1 行为策略自身也在大量空按

`bomb1_invariant.rows`：行为策略自己按雷的 **153** 行里，**119 行是 no-op（77.78 %）**，只有 34 行真的扣了雷。
即连行为策略（R5 规则）都有 3/4 的按键被无敌窗口吞掉——这是 v8.3.0/v4.5.0 记录里「98–99 % 按键被忽略」的同一机制。

### 3.2 被标签状态的三分之二落在「无敌标志」上

采集期 `states_by_player_state = {ALIVE: 1,624, INVULNERABLE: 3,416}`：**67.8 % 的标签状态
（3,416 / 5,040）在 park 时刻 `player_state = 3`**。这正是旧不变量必然误伤的根源，也是本轮 noop 计数的主要来源。

但 `player_state = 3` **不是** no-op 的充分条件（引擎在窗口边界、以及别的无敌来源上仍会接受按键）：

| 交叉表（状态数） | 18 条 bomb=1 全 noop | 18 条全 honoured | 合计 |
| --- | --- | --- | --- |
| `player_state = 3` (INVULNERABLE) | **2,702** | 714 | 3,416 |
| `player_state = 0` (ALIVE) | 1 | 1,623 | 1,624 |
| 合计 | **2,703** | 2,337 | 5,040 |

（`noop_branches_by_player_state = {ALIVE: 18, INVULNERABLE: 48,636}`，48,636 = 2,702 × 18。）
即：**no-op ⇒ 几乎必然是 `player_state = 3`（2,703 / 2,704 个状态的分支全 noop）**；反过来
`player_state = 3` 里有 **79.1 %（2,702 / 3,416）** 的状态按键被忽略，其余 714 个状态在窗口边界上仍扣雷。

### 3.3 本轮确认的引擎事实（供 v8.7.0 可用性掩码设计）

> **在 no-op 状态上，36 动作空间沿 bomb 维是退化的。**
> 雷的**可用性**只由「此刻按键是否会被引擎接受」决定，**与候选方向无关**：
> - **2,703 / 5,040（53.6 %）个状态**：**18 条 `bomb=1` 分支全部**是 no-op，且
>   `alive_frames` 与各自 `bomb=0` 兄弟**逐位相同**（`bomb1_unspent_alive_differs = 0`）⇒
>   `bomb=0` 与 `bomb=1` 是**同一个动作**，bomb 维完全不可区分；
> - **2,337 / 5,040（46.4 %）个状态**：18 条 `bomb=1` 分支全部 honoured（扣 1 颗），bomb=0 全部不扣；
> - `states_some_bomb1_noop = 0`：**没有「部分分支 noop」的中间态** —— 同一状态下 18 条分支同生同死，
>   因为按键是否被接受取决于玩家自身状态，与候选方向无关。
>
> 因此下一轮的掩码应是**状态级**的：在 no-op 状态（等价地 `player_state = 3` 且窗口未结束）把 bomb=1 的
> 18 条动作从 loss / 采样 / 评价里**折叠**（或直接改标为 `bomb=0`），否则有效动作数是 18 而不是 36。

---

## 4. 标签对照（bomb=1 vs bomb=0，描述性）

| 子集 | 状态 | bomb=0 `safe_rate` | bomb=1 `safe_rate` | 差 |
| --- | --- | --- | --- | --- |
| 全部 | 5,040 | 0.951543 | 0.999184 | +0.047641 |
| bombs=3 | 624 | 0.867610 | 1.000000 | +0.132390 |
| bombs=2 | 3,378 | 0.953473 | 0.999227 | +0.045754 |
| bombs=1 | 1,038 | 0.995718 | 0.998555 | +0.002837 |

**和 v8.3.0 一样：`bomb=1` 从不更差，且在 bombs=1/2 的状态上仍然「按了更安全」。**
这就是 §8 里「P3 不可能靠状态分布学会」的直接原因。

---

## 5. 训练

`./exec.sh train`（11:39:16–11:39:29，rc=0）——配方与 v8.3.0 完全相同，**无任何辅助损失**：

| seed | 状态 | 首 epoch BCE | 末 epoch BCE | 最终 BCE | 训练 acc | mean p(bomb=0) | mean p(bomb=1) | ckpt sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5,040 | 0.595201 | 0.039285 | 0.039121 | 0.983851 | 0.952741 | 0.999111 | `9c6d5d4a…` |
| 1 | 5,040 | 0.574116 | 0.039133 | 0.039002 | 0.983736 | 0.953274 | 0.999100 | `36a51090…` |
| 2 | 5,040 | 0.597460 | 0.039159 | 0.038994 | 0.983884 | 0.951180 | 0.999066 | `885dde70…` |

模型 `Policy(106,64,36)`（`Linear(106,64) → Tanh → Linear(64,36)`），BCEWithLogits，Adam lr=1e-3，
batch 256，200 epoch（12,600 步 / 种子），wall 1.86–2.16 s / 种子。
`mean p(bomb=1) = 0.999` ⇒ 策略对「按雷安全」几乎全信，**这是 P3 不成立的直接体现**。

---

## 6. 评测（300 / 600 / 900 决策 × 3 种子 × 4 dev 窗口 × 300 episode）

`./exec.sh grid`（11:39:45–12:05:13，9 进程并行，全部 rc=0）：

| 预算 | 活满 | 活满率 | CP95 | v8.3.0 | Δ vs v8.3.0 | 单侧 Fisher p | R5 规则 | 距 R5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 300 | 3,598 / 3,600 | **99.9444 %** | [99.7995 %, 99.9933 %] | 99.9444 % | **0.00 pp** | 1.000 | 100 % | −0.06 pp |
| 600 | 1,065 / 3,600 | **29.5833 %** | [28.0956 %, 31.1043 %] | 19.4444 % | **+10.14 pp** | **4.98e-07** | 77.05 % | −47.47 pp |
| 900 | 106 / 3,600 | **2.9444 %** | [2.4168 %, 3.5502 %] | 0.6111 % | **+2.33 pp** | 8.73e-15 | 19.38 % | −16.44 pp |

每种子（`eval_d*_seed*.jsonl`，各 1,200 episode）：

| 预算 | seed 0 | seed 1 | seed 2 |
| --- | --- | --- | --- |
| 300 | 1,199 / 1,200 = 99.9167 % | 1,200 / 1,200 = 100 % | 1,199 / 1,200 = 99.9167 % |
| 600 | 343 / 1,200 = 28.5833 % | 386 / 1,200 = 32.1667 % | 336 / 1,200 = 28.0000 % |
| 900 | 49 / 1,200 = 4.0833 % | 45 / 1,200 = 3.7500 % | 12 / 1,200 = 1.0000 % |

`alive_frames`（帧，每决策 2 帧）：d600 均值 1,027.0（中位 1,065），d900 均值 1,096.7（中位 1,052.5）。

### 6.1 P3：用雷分布 —— **不成立**

| 预算 | 用 0 颗 | 用 1 颗 | 用 2 颗 | 用 3 颗 | 用 1–2 颗 | 耗尽率 |
| --- | --- | --- | --- | --- | --- | --- |
| 300 | 0 | 0 | 0 | 3,600 | **0** | 1.0 |
| 600 | 0 | 0 | 0 | 3,600 | **0** | 1.0 |
| 900 | 0 | 0 | 0 | 3,600 | **0** | 1.0 |

`first_bomb_decision` 中位数 = **0**（d300/d600/d900 都是第一决策就按），
`last_bomb_decision` 中位数 d600 = 532。
按键被忽略的比例：d600 共 **737,978** 次 bomb=1 采样，**727,178** 次被引擎忽略（**98.54 %**）——
策略实际上把「按雷」当成了持续动作。

⇒ **预注册 §2 P3「必须出现只用 1–2 颗雷的 episode」未达成**；v8.3.0 的「100 % 用 3 颗」在 v8.6.0 **没有改变**。

---

## 7. 判据裁决（按预注册 §2）

| # | 判据 | 结果 | 裁决 |
| --- | --- | --- | --- |
| **P0** | `bombs_remaining` 2 与 1 各 ≥ 200 状态 | 3,378 / 1,038 | **PASS** |
| **P1** | d600 活满率 > 19.44 % | 29.5833 %（CP95 下限 28.10 %） | **PASS** |
| **P2** | 与 R5（77.05 %）差距；≥ 50 % 才算「学会一部分省雷」 | 29.5833 %（差 47.47 pp） | **FAIL** |
| **P3** | 出现只用 1–2 颗雷的 episode | 0 / 10,800 | **FAIL** |

按 §2.1 的停止条件表：**S-partial**（P1 成立但 < 50 %，判「方向对、量不够」）。

**但必须同时记下与 S-partial 相反的一条读数**：S-partial 的处方是「下一轮扩状态数到 10,000」。
本轮数据表明这条路**不会**产生省雷行为——因为瓶颈不在状态数，而在标签语义（§8）：只要 `bomb=1` 在
99.92 % 的状态下 safe、且在无敌窗口内是一次**免费**的（被忽略的）按键，BCE 的最优解就是「一直按」。
扩到 10,000 个状态只会把同一个信号放大。若下一轮仍以 P3 为目标，**需要改标签/动作空间（可用性掩码、
或把 noop 分支从 loss 中折叠）**，而不是改规模。

---

## 8. 为什么 P3 一定失败（机制，不是数据量）

1. `bomb=1` 的 `safe_rate` 在所有 `bombs_remaining` 层级上都 ≥ `bomb=0`（§4），且差值为正；
2. 在 53.6 % 的状态里（2,703 / 5,040，全在无敌标志上），`bomb=1` 与 `bomb=0` **完全是同一个动作**（§3.3），
   标签给两者相同目标；在其余状态里 `bomb=1` 均 honoured 且 safe 率更高；
3. 于是 36 路输出里 bomb 维的目标几乎恒为 1，`mean p(bomb=1) = 0.999`（§5）；
4. 评测时策略每决策都在按雷，98.54 % 被无敌窗口吞掉，真正扣雷只发生在窗口结束后的那一帧
   ⇒ `bombs_used` 恒为 3、`first_bomb_decision` 恒为 0。

「省雷」需要的是**代价信号**（用掉一颗雷会损失未来的生存机会），而 36 路反事实标签在 H=120 帧内
看不到这个代价：120 帧里雷的收益是即时的（无敌），代价在 120 帧外。这是本轮最重要的负面结论。

---

## 9. 与 v8.3.0 / R5 并排

| 端点 | v8.3.0 BOMBPOL-001 | **v8.6.0 BOMBDAGGER-001** | R5 规则（v8.4.0） |
| --- | --- | --- | --- |
| 行为策略（状态来源） | v4.5.0 `S`，**永不放雷** | v4.5.0 `S` + **R5 放雷规则** | 规则自身 |
| 标签集 `bombs_remaining` | `{3: 5166}` | `{3: 624, 2: 3378, 1: 1038}` | — |
| d300 活满率 | 99.9444 % | **99.9444 %** | 100 % |
| d600 活满率 | 19.4444 % | **29.5833 %** | 77.05 % |
| d900 活满率 | 0.6111 % | **2.9444 %** | 19.38 % |
| 每 episode 用雷 | 3（3,600/3,600） | **3（3,600/3,600）** | — |
| 只用 1–2 颗雷的 episode | 0 | **0** | — |
| 训练配方 | `Policy(106,64,36)`、BCE、Adam 1e-3、200e、无辅助损失 | **同** | — |
| 结论 | 不会省雷 | **存活↑（P1），仍不会省雷（P3）** | 规则上界参考 |

---

## 10. 偏差清单（与预注册/工单的差异，全部如实记录）

1. **状态数 5,040（预注册「≈5,000」）**：每格 `--quota 420`，12 格 × 420 = 5,040。中途主代理要求降到
   3,600（12 × 300），但当时 12 个进程已各到 350 状态、且禁止 kill，进程内 quota 不可改；按
   「不再新开格子、让进程自然收尾」执行，最终 5,040（ep0 的 12 × 300 = 3,600 已完整包含在内）。
   主代理随后确认「用 5,040，不要 kill」。
2. **`--mode stats` 之后 runner 换了字节**：见 §11。采集用 `8fab2e63…`，stats/train/eval 用 `3433ae74…`；
   差异仅限 `--mode stats` 的 noop 聚合字段与 docstring，**未触碰**标签生成、行为策略、训练或评测代码。
3. **`bomb=0` 分支扣雷不再中止整格**：原实现会 `raise`，本轮改为计数（`bomb0_spent_a_bomb`，
   实测 **0** 次）并写日志。理由同 §1：计数型异常不应中止整格。标签完整性检查（36 条标签齐全、
   子进程 frame/state 一致、spawn/quit 全成功、PARK frame 一致）**仍然 fail-closed**。
4. **`--mode stats` 新增两种 noop 口径**（`noop_branch_frac` / `noop_state_frac` 及 `*_all_18`）：
   按主代理要求加入，属报告字段，不改变任何既有字段。
5. **旧的 12 个 `*.partial` 文件**（上一任代理的失败残留，9 个只有 1 个状态、3 个 w16 各 191 个状态）
   在重采时按脚本既有行为被同名 `.partial` 覆盖/删除；它们的 `pins.self.sha256` 与错误信息仍保留在
   `logs/collect-*.out`（旧日志被覆盖）之外的本记录 §1 引文里。**未删除任何受保护资产。**
6. **冒烟证据**：`smoke/` 下保留了 10 状态冒烟（seed0 window13，`--quota 10`）的
   `labels_bombdag.jsonl` 与 `labels_stats.json`，其 `noop_frac = 0.9`、`anomaly_count = 0`。
   该格随后被全量重采覆盖，冒烟数据**未混入** 5,040 的主标签集。
7. **未做**：不 `git add/commit`；未改 `touhou_rl/**`、`third_party/**`、既有版本目录/预注册、其它 runner。

---

## 11. 两个 runner 字节（采集 ≠ 训练，必须并列记录）

| 用途 | 文件 | sha256 |
| --- | --- | --- |
| **采集**（生成 5,040 状态 / 181,440 标签） | `.tmp_v860_bombdag/runner_collect_8fab2e63.py`（快照） | `8fab2e633cf2d4be2cffb8d23380bad296b4f170c8b265c3fe13570ac7083b07` |
| **stats / train / eval / publish** | `tools/rl/v860_bombdag.py` | `3433ae743cc224cae436b6e2192654e30eabe5f05f9053218969e52bdc47cf8a` |

`3433ae74` 相对 `8fab2e63` 的差异**只有两处**：
1. `--mode stats` 里新增 `noop_branch_frac` / `noop_state_frac` / `noop_state_frac_all_18` 及其分层计数；
2. 两段 docstring（`--mode collect` / `--mode stats` 的字段说明）。

**没有任何一行属于**：标签生成（`label_state36` 调用与不变量判定）、行为策略（`behaviour_decision`）、
训练（`mode_train`）、评测（`mode_eval`）。每个 `collect/*.json` 的 `pins.self.sha256` 均为
`8fab2e63…`，每个 `train/seed*.json` 的 `pins.self.sha256` 均为 `3433ae74…`，可交叉核对。

---

## 12. 复现（WSL，仓库根 `/mnt/d/WorkShop/Touhou_RL`）

```bash
export SDL_AUDIODRIVER=dummy; unset DISPLAY WAYLAND_DISPLAY
export PYTHONPATH=/mnt/d/WorkShop/Touhou_RL PYTHONUNBUFFERED=1
PY=/root/v400-venv/bin/python
V860=/mnt/d/WorkShop/Touhou_RL/tools/rl/v860_bombdag.py
WORK=/mnt/d/WorkShop/Touhou_RL/.tmp_v860_bombdag

$PY $V860 --mode preflight
$PY $V860 --mode assert-ckpt --phase start
$PY $V860 --mode collect --seed 0 --windows 13 --quota 10 --tag v860-smoke-s0-w13   # 冒烟
$PY $V860 --mode stats --labels-out $WORK/smoke/labels_bombdag.jsonl \
                        --out $WORK/smoke/labels_stats.json                     # 冒烟 stats
cd $WORK && ./exec.sh collect && ./exec.sh stats && ./exec.sh train \
        && ./exec.sh eval-smoke && ./exec.sh grid && ./exec.sh diagnostics \
        && ./exec.sh audit && ./exec.sh final && ./exec.sh publish
```

---

## 13. 产物

`Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/`（`manifest.json` 给出每个文件的 sha256/bytes）：

- `labels_bombdag.jsonl`（5,040 行）、`labels_stats.json`（P0 + 两种 noop 口径 + `anomaly_count`）
- `train_seed{0,1,2}.json`、`ckpt_seed{0,1,2}.pt`、`train_summary.json`
- `eval_d{300,600,900}_seed{0,1,2}.jsonl` + `.summary.json`（9 × 1,200 episode）
- `bomb_usage.json`、`diagnostics.json`（P3 的用雷分布）
- `preflight.json`、`ckpt_assertions.json`、`ckpt_assertions_final.json`、`pilot.json`、`fidelity_d3.json`
- `audio_audit.json`（35/35 muted）、`process_residue.json`（本轮引擎进程 0）
- `collect/seed{0,1,2}_w{13,14,15,16}.json`（12 格采集摘要，含 `label_checks`）
- `labels_stats_smoke.json` / `smoke/`（10 状态冒烟）
- `runner_collect_8fab2e63.py`（采集用的 runner 快照）
- `exec.sh`、`commands.tsv`（每次调用的 rc / 墙钟 / argv）、`RUN.md`、`manifest.json`
