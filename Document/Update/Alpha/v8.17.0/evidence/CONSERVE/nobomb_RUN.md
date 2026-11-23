# v8.17.0 CONSERVE-001 — **`none` 分支（NOBOMB）**：一次不放雷，让死亡邻域必然带 3 颗雷

运行时刻：2026-11-22（Asia/Shanghai）。训练端（WSL），引擎
`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`
（sha256 `691bfd77…`，与冻结值一致）。

脚本：`tools/rl/v8i0_nobomb.py`（**由 `tools/rl/v8h0_conserve.py` 复制生成**，
`v8h0_conserve.py` 本身一个字节未改）。生成器与逐条替换清单：
`.tmp_v8i0_nobomb/make_v8i0.py`（每条替换都断言「在源文件里恰好出现一次」）。
工作目录：`.tmp_v8i0_nobomb/`。证据目录：本目录（所有文件名带 `nobomb`）。

---

## 0. 为什么开这个分支（预注册 §2.1 的 `S-nobomb` 兜底）

| 轮次 | 行为策略的放雷规则 | 死亡邻域 `bombs_remaining` |
| --- | --- | --- |
| v8.16.0 DEATHADJ | R5 分阶段 80/60/40 px | **0/1,431 带雷**（全部 0）⇒ `bomb=1` 是 no-op，diff 恒 0 |
| v8.17.0 CONSERVE（25 px） | `nearest_all_px < 25` | **0/27 带雷**（冒烟 `bombs 3->0, emits=15`）⇒ 同样 no-op |
| **本轮 `none`** | **从不放雷** | **600/600 带 3 颗雷** |

⇒ 任何**按距离触发**的规则都会在死前把雷花光；只有「一次不放」能保证死亡时手里有雷。
这是预注册 §1 里写明的第三种、也是最纯的做法，**不改预注册**，只走它的兜底条款。

## 1. 冒烟（先跑 1 条死亡 episode，rc=0）

`--mode collect --bomb-rule none --seed 0 --windows 11 --quota 20 --max-episodes 1
--decisions 900 --verify-h 1`，工作目录 `.tmp_v8i0_nobomb/smoke_work/`。

| 项 | 实测 |
| --- | --- |
| 死亡决策 | **45**（46 decisions / 92 frames；`bombs 3->3`，`emits=0`） |
| 采样窗 | 26..45（最后 20 个决策） |
| `bombs_remaining` | **20/20 状态 = 3**；`bombs_at_death = {3: 1}` |
| H=120 | critical 0/20，diff **20/20** |
| H=240 | critical 0/20，diff **20/20** |
| H=600 | critical 0/20，diff **20/20** |
| H 嵌套校验 | 直接跑 H=120/240 与 H=600 截断比对：`{120: 0, 240: 0}` mismatches |

典型行（d=26）：`alive_frames` 的 bomb=0 半区全部 39 帧，bomb=1 半区全部 600 帧
⇒ `best_bomb0 = 39`、`best_bomb1 = 600`、`delta = 561`。

## 2. P0a / P0b（12 格全 complete，600 状态 / 1800 行）

采集：3 种子 × 4 窗（11/12/17/18）× 50 状态，12 个进程并行，wall 426 s；
36 条死亡 episode，0 条 no-death，无「连续 10 条不死」跳过。

| H | 状态 | **P0a** `bombs_remaining > 0` | 门 ≥50 % | **P0b** `best bomb1 > best bomb0`（在带雷状态中） | 门 ≥20 % |
| ---: | ---: | --- | --- | --- | --- |
| 120 | 600 | **600 (100.0 %)** | PASS | **588 (98.0 %)** | PASS |
| 240 | 600 | **600 (100.0 %)** | PASS | **588 (98.0 %)** | PASS |
| 600 | 600 | **600 (100.0 %)** | PASS | **588 (98.0 %)** | PASS |

按 `bombs_remaining` 分列（三个 H 相同）：

| bombs | 状态 | armed | diff | critical(context) |
| ---: | ---: | ---: | ---: | ---: |
| **3** | **600** | **600** | **588** | 12 / 42（按 H） |
| 2 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 | 0 |

按距死亡决策数分列（H=120）：

| 距死亡 | 状态 | diff |
| --- | ---: | ---: |
| 0（死亡决策本身） | 24 | 12 |
| 1–4 | 96 | 96 |
| 5–9 | 120 | 120 |
| 10–19 | 360 | 360 |

delta（`best_bomb1 - best_bomb0`）分布：H=120 mean 96.1 / median 96.5 / q1–q3 88–107；
H=240 mean 213.7；H=600 mean 565.3。`best_bomb0` mean 21.5（0–39），
`best_bomb1` mean 117.6 / 235.2 / 586.8。

**判据说明（重要，工作订单要求）**：`critical`（`max(alive) < H`）**不是**本轮的门，
而且在「雷能救命」的数据上它会**失效**：bomb=1 把濒死救到**正好 H** 时
`max(alive) == H` ⇒ 被判为「不 critical」，但那一格恰恰是雷有价值的证据。
本轮 588 个 diff 状态里绝大多数就是这样（H=120：12/600 critical，H=600：42/600）。
因此 P0 只以 **diff** 为准。

## 3. 机制：这 600 个状态全部是 `DYING`，雷是以「死亡炸弹」兑现的

- `states_by_player_state = {"DYING": 600}` —— **每个采样状态的 `player_state` 都是 2
  (DYING)**，采样窗整段落在死亡动画里。
- 原因：`--bomb-rule none` 下每一次死亡都是「被击中」⇒ 进入死亡动画；而 harness 的
  `deaths` 计数（`_terminal_of` 的判据）比命中滞后约 40 帧 = **正好 20 个决策**，
  所以「死亡前最后 20 个决策」必然整段是死亡动画。
- 引擎实测的按雷语义（本轮新测）：在 DYING 态按下雷 = **死亡炸弹**，**一次扣 2 颗**
  （`bombs_start 3 -> bombs_end 1`，`decrement_frame = 2`），随后活满 600/600。
  计数：`bomb1_press_honoured = 10,584`、`bomb1_noop_due_to_invuln = 216`
  （= 12 个死亡决策态 × 18，`player_state` 已不可救）、`bomb0_spent_a_bomb = 0`、
  `bomb1_anomalies = 0`。
- **账本改动**：v8h0 的 `spent = (after == before - 1)` 假设「按一次扣一颗」，
  所以第一版冒烟把 18/18 判成 anomaly 并触发 fail-closed（rc 4，该格 `*.partial` 保留）。
  本轮改为「扣 ≥1 颗即 honoured」，新增 branch kind `spent_deathbomb_2` 与
  计数 `bomb1_spent_two_or_more`；**fail-closed 门保留**（某格有雷却一次都没扣 ⇒ 中止）。
- ⇒ **本轮的标签表达的是「死那一刻手里那颗雷的价值」**（deathbomb），
  不是「活着时提前放雷的价值」；`none` 分支没有、也不可能有后者。

## 4. 训练（A/B，H=120，600 状态 × 36 路，Policy(106,64,36)，Adam 1e-3，batch 256，200 epoch，无早停）

| 臂 | 目标 | loss 首→末（seed0） | final BCE（3 种子） | train acc | aux acc |
| --- | --- | --- | --- | ---: | ---: |
| **A** | BCE(36) + 0.5 × BCE_bomb(**best-bomb**) | 1.0433 → 0.1282 | 0.0603 / 0.0591 / 0.0600 | 0.9900 | 0.9800 |
| **B** | 仅 BCE(36) | 0.6954 → 0.0490 | 0.0490 / 0.0489 / 0.0489 | 0.9900 | 0.9800 |

A 臂的 aux 目标（工作订单口径）：取 `argmax_c max(alive[2c], alive[2c+1])` 的候选，
若该候选 `alive[2c+1] > alive[2c]` 则 1，否则 0 ⇒ **588/600 为正**。
`rule_bomb` 在本分支**恒 0**（行为策略不放雷），所以规则位不再用于训练，
只作对照记录（25 px 规则位 = 600/600）。

## 5. 评测（A/B × 300/600/900 × 3 种子 × 4 窗 × 300 episode = 每格 1,200 / 每臂 3,600）

18 格全部 `status=complete`（rc=0），每格 1,200 episode，合计 **21,600 episode**。

| 预算 | 臂 | 活满 | 活满率 | 对照：v8.3.0 `S`（不放雷） | 对照：v8.6.0（R5 监督） | 对照：R5 脚本规则 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 300 | A | 3,549 / 3,600 | **98.583 %** | 99.944 % | 99.944 % | 100 % |
| 300 | B | 3,540 / 3,600 | **98.333 %** | | | |
| 600 | A | 616 / 3,600 | **17.111 %** | 19.444 % | 29.583 % | 77.111 % |
| 600 | B | 666 / 3,600 | **18.500 %** | | | |
| 900 | A | 71 / 3,600 | **1.972 %** | 0.611 % | 2.944 % | 17.611 % |
| 900 | B | 75 / 3,600 | **2.083 %** | | | |

按种子（活满率）：

| 格 | seed0 | seed1 | seed2 |
| --- | ---: | ---: | ---: |
| A d300 | 0.980833 | 0.988333 | 0.988333 |
| A d600 | 0.172500 | 0.164167 | 0.176667 |
| A d900 | 0.021667 | 0.015000 | 0.022500 |
| B d300 | 0.977500 | 0.987500 | 0.985000 |
| B d600 | 0.180833 | 0.192500 | 0.181667 |
| B d900 | 0.022500 | 0.021667 | 0.018333 |

**读法**：A/B 两臂在 300/600 上都比「完全不放雷的 v8.3.0 `S`」**差**
（−1.4 / −1.2 个百分点 at 300；−2.3 / −0.9 at 600），在 900 上**好 3 倍**
（0.611 % → 1.97 / 2.08 %），但仍低于 v8.6.0 的 2.944 %。A 与 B 之间，
B（纯 36 路 BCE）在 600/900 上**略好于** A（17.111 vs 18.500、1.972 vs 2.083）
⇒ 本分支里 best-bomb aux 项**没有带来收益**。

**与其它轮的并排（600 决策为主）**：

| 轮次 | 臂 | 600 活满率 | 来源 |
| --- | --- | ---: | --- |
| **v8.17.0 NOBOMB（本轮）** | A | **17.111 %** | 本文件 §5（3,600 episode） |
| **v8.17.0 NOBOMB（本轮）** | B | **18.500 %** | 本文件 §5（3,600 episode） |
| v8.3.0 BOMBPOL | `S`（从不放雷） | 19.444 % | `v8.3.0/evidence/BOMBPOL/diagnostics.json` |
| v8.6.0 BOMBDAG | `S` + R5 规则监督 | **29.583 %** | `v8.6.0`/BASELINE 表 |
| v8.12.0 HYBRID | 98.33 % ← **主代理转述，口径未复核** | （其 98.33 % 疑为 300 档，不与本行的 600 并排） | 主代理转述 |
| v8.19.0 RULESWEEP30 / v8.20.0 UNIFIED | **24.17 %**（主代理转述，口径未复核） | | 主代理转述 |

本轮 A/B（17.1 / 18.5 %）低于 R5 监督线（29.58 %）、低于 sweep/unified 的 24.17 %，
也低于不放雷的 `S`（19.44 %）——**唯一正向的一项是 900 档的 3 倍**。
（v8.12 / v8.19 / v8.20 的数字由主代理转述，本轮**未在本机复核**其口径与样本量，
只作方向参考，不作为对比结论。）

**首轮 18 个评测全部作废（必须记录的失败）**：eval 分支漏传 `--tag`，`main()`
按当前秒给 engine 模式生成默认 tag，18 个进程同一秒启动 ⇒ 共用同一个
`data/eval-<timestamp>/w*` 目录，互相 `rmtree`/`copytree`：

- 一部分进程 `FileNotFoundError`（目录正被别的进程重建）；
- 另一部分拿到别的进程引擎**退出时写回的可出声 cfg** ⇒ §9.3 静音门 fail-closed
  `ForkError: refusing to start the engine: the generated th08.cfg is not muted`。

18 个 summary 全是 `status=failed`、`episodes_total=0`、`wall 0.02–0.27 s`。
修复：每个 eval 显式 `--tag v8i0-nobomb-eval-{arm}-d{budget}-s{seed}`（18 个不同目录）。
重跑前先跑 `eval-smoke`（arm A / d300 / seed0 / 5 episode / w11，独立 tag + 独立
`--out`）：`status=complete survived=5/5 bombs_used=15`、`cfg muted=True`、
`prefix 2165/2165, 0 mismatches`。首轮 5 个「跑完」的 summary 同样作废重跑
（它们的数据目录被并发重写过，不能当证据）。

## 6. 用雷分布（P3）

| 臂 | 预算 | `bombs_used` 分布 | 只用 1–2 颗的 episode | 用光 3 颗 |
| --- | ---: | --- | ---: | ---: |
| A | 300 / 600 / 900 | `{3: 3600}` | **0** | 3,600 / 3,600 |
| B | 300 / 600 / 900 | `{3: 3600}` | **0** | 3,600 / 3,600 |

**P3 不成立**：没有任何一局只用 1–2 颗雷 —— 学到的策略在每一局都**把 3 颗雷全部花光**
（与训练集「DYING 态按雷」一致：deathbomb 一次扣 2 颗，两轮 press 就清空）。
它既不「省」，也没有换来 600 档的存活。

## 7. 落盘与身份

- `labels_nobomb.jsonl`（pooled，1800 行）、`nobomb_stats.json`、12 个
  `collect/nobomb_seed{S}_w{W}.json`；
- `nobomb_preflight.json`、`nobomb_ckpt_assertions.json` / `…_final.json`、
  `nobomb_fidelity_d3.json`（D3 五条全 PASS）、`nobomb_recall_v816.json`；
- `train_A_nobomb_seed{0,1,2}.json`、`train_B_nobomb_seed{0,1,2}.json`、
  `ckpt_A_nobomb_seed*.pt`、`ckpt_B_nobomb_seed*.pt`、`nobomb_train_summary.json`；
- `eval_{A,B}_nobomb_d{300,600,900}_seed{0,1,2}.jsonl` + `.summary.json`；
- `nobomb_diagnostics.json`、`nobomb_bomb_usage.json`、`nobomb_audio_audit.json`、
  `nobomb_process_residue.json`、`nobomb_commands.tsv`、`nobomb_work_manifest.json`、
  `manifest.json`。

## 8. 音频（AGENTS.md §9）

每个 data dir 由 `v450_cf_label.prepare_data_dir` 现场生成静音 `th08.cfg`
（`v400_make_cfg.py --out` + `--inspect`，四字段全 0），启动环境
`SDL_AUDIODRIVER=dummy`、`DISPLAY` / `WAYLAND_DISPLAY` 未设置。
`--mode audio-audit`：**13 个 cfg，13 decoded，13 muted，0 audible**。

## 9. 判定（按预注册 §2.1 的门逐条对）

| 门 | 判据 | 实测 | 结论 |
| --- | --- | --- | --- |
| **P0a** | 死亡邻域 `bombs_remaining > 0` ≥50 % | **100 %** | **PASS** |
| **P0b** | 其中 `best bomb1 > best bomb0` ≥20 % | **98.0 %** | **PASS**（三个 H 相同） |
| **P1** | 臂 B 的 600 决策活满率 > 19.44 %（v8.3.0） | **18.500 %** | **不成立** |
| **P3** | 出现「只用 1–2 颗」的 episode | **0 / 3,600**（全用光 3 颗） | **不成立** |
| P4 | 900 档与 v8.3/8.6/R5 并排 | 1.972 / 2.083 %（v8.3 0.611、v8.6 2.944、R5 17.611） | 记录 |

预注册的四种结局都不精确对应：`S-nobomb`（P0a 不成立）否、`S-nodiff`（P0b <20 %）否、
`S-ok`（P1 且 P3）否（P1 不成立）、`S-partial`（P1 成立但 P3 不成立）否（P1 也不成立）。
**本轮记为 `S-weak`：标签第一次能表达一颗雷的价值（P0b 98 %），但把它变成策略之后，
既没换来 600 档的存活，也没学会节省（每局仍花光 3 颗）。最硬的数字是 900 档：
不放雷的 `S` 0.611 % → 本轮 1.97–2.08 %（3 倍），仍低于 v8.6.0 的 2.944 %。**

## 10. 偏差清单（全部为记录在案的失败或口径变更）

1. **首轮 18 个评测全部 fail-closed 作废**：eval 分支漏传 `--tag` ⇒ 18 个进程同一秒
   共用同一个 data 目录 ⇒ `FileNotFoundError` / 「cfg not muted」。修复后重跑，
   18 格全部 `status=complete`（详见 §5 末尾）。
2. **首版冒烟采集 rc=4**（`bombs 3->3`、20 状态全 `DYING`）：v8h0 的
   `spent = after == before - 1` 假设「按一次扣一颗」，而死亡炸弹一次扣 **2** 颗，
   于是 18/18 被判 anomaly、fail-closed 中止该格（`*.partial` 保留）。
   本轮改为「扣 ≥1 颗即 honoured」，新增 kind `spent_deathbomb_2` 与
   `bomb1_spent_two_or_more`；**fail-closed 门保留**。
3. **A 臂 aux 目标改了口径**（工作订单要求）：不用 v8h0 的 25 px 规则位，改用
   `argmax_c max(alive[2c], alive[2c+1])` 的 best-bomb 位（588/600 为正）。
   `rule_bomb` 在本分支恒 0，规则位只作记录。
4. **`critical` 不作门**（工作订单要求）：它会把「救到正好 H」误判为「不需要」。
   本轮只在 §2 作 context 报告。
5. `nobomb_stats.json` 的**合并** `label_checks.bomb1_spent_two_or_more` 为 `null`
   （合并用的 key 列表早于本轮新增字段），但**每个 cell 的
   `collect_cells[*].label_checks.bomb1_spent_two_or_more` 是正确的**（合计 10,584）。
   这是记录字段的口径缺口，不影响任何统计量。
6. 未改 `touhou_rl/**`、`third_party/**`、`v8h0_conserve.py`、既有版本目录/证据；
   未 `git add/commit`；未删除/移动/覆盖资产。

## 11. 不能声称什么

- **不能声称「学会了资源分配」**：本分支的数据只覆盖死亡动画内的 deathbomb；
- **不能把 12 格当独立样本**，也不能把不同预算的均值直接比较；
- 不能声称「通关」，不能声称「保守规则/距离规则更好或更差」——本轮只有 `none` 一个行为策略；
- 不能把 `critical` 当作「雷没价值」的证据（见 §2 的判据说明）。
