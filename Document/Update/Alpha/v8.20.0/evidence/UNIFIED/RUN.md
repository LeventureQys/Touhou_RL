# v8.20.0 UNIFIED — 收紧阈值 60/40/20 在 600 / 900 / 1,200 决策三档统一重测

冻结预注册：[`PREREGISTRATION_UNIFIED_001.md`](../../../PREREGISTRATION_UNIFIED_001.md)
（sha256 `199a567ebe26198ef6892a2f2e37201143a119d86c4a66f958f7ca0f4ec44c42`）。
执行：训练端 WSL2 Ubuntu，引擎
`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`
（sha256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`）。
运行窗：2026-11-22 15:18–15:41（Asia/Shanghai）。**零训练、只评测。**

---

## 0. 一句话结论

**在 600 / 900 / 1,200 决策三档上、候选头与规则都不动的前提下，60/40/20 这条规则给出
98.7500 % / 24.1667 % / 2.0833 %；P1（≥95 %）成立、P2（对 v8.19 S1 的可重复性）成立 ⇒
判 `S-unified`：一套阈值（60/40/20）覆盖 20 s 与 30 s。** 40 s 档只有 2.0833 %，
与 R5 的 1.00 % 差 +1.0833 pp（Fisher p = 0.1097），**达不到预注册的 +3 pp「实质提升」线**。

| 判据 | 要求 | 观测 | 判定 |
| --- | --- | --- | --- |
| **P1（主）** | U-600 活满率 ≥ 95 % | **237/240 = 98.7500 %**，CP95 [96.3906, 99.7415] | **成立** |
| **P2** | U-900 与 v8.19 S1（58/240 = 24.1667 %）CP95 重叠 | **58/240 = 24.1667 %**，CP95 与 v8.19 **逐位相同** [18.8918, 30.0917]；**且 240/240 条 episode 逐条一致** | **成立（超出判据）** |
| **P3** | U-1200 与 R5 的 1.00 % 并排，≥ 3 % 即实质提升 | **5/240 = 2.0833 %**（+1.0833 pp，Fisher p = 0.1097） | **报；未达实质线** |
| **P4（描述）** | 用雷分布 | 见 §5 | 已报 |
| **停止条件** | — | **`S-unified`** | — |

---

## 1. 协议（本轮唯一动的东西是预算）

| 项 | 值 |
| --- | --- |
| 候选头 | `.tmp_v860_bombdag/ckpt/seed{0,1,2}/bombdag.pt` — `Policy(106,64,36)`，经 `v860_bombdag.load_policy36` |
| 采样 | `score[c] = max(logit[2c], logit[2c+1])`，`cand ~ softmax(score / 1.0)`，**每决策一次抽样** |
| 生成器 | 每进程一个 `torch.Generator().manual_seed(seed * 1000 + 7)`，在窗循环之前建，跨窗跨局连续消费 |
| 放雷规则 | `nearest_all_px < THRESHOLDS[bombs_remaining]` 且 `bombs_remaining > 0`；`THRESHOLDS = {3: 60, 2: 40, 1: 20}` px（v8.19 的 S1） |
| 距离度量 | `v800_bomb.nearest_bullet_distances`（v8.0.0 自己的代码，未重写） |
| 掩码 | 决策首帧 `cand_to_mask(cand, bomb)`，其后 `cand_to_mask(cand, 0)`；在 `v830_bombpol.run_episode36` 内 |
| 起点 | PARK 帧 **8663**；2 帧/决策；`budget` 决策 = `2 × budget` 帧 |
| 循环 | `seed → window → episode(20)`；3 种子 × 4 窗 × 20 = **每档 240 条** |
| 三档 | 600（1,200 帧）/ 900（1,800 帧）/ 1,200（2,400 帧） |

---

## 2. 保真门与身份断言

### 2.1 D3 保真门（5 条，`--mode d3`，exit 0）

| # | 断言 | 结果 |
| --- | --- | --- |
| 1 | `HELLO role=shadow` | **PASS** |
| 2 | `DEBUG threads=1` | **PASS** |
| 3 | 前缀 2,165 个 checkpoint 到帧 8663，0 不匹配 | **PASS** |
| 4 | R24 的 18 个常量 18/18 | **PASS** |
| 5 | PARK frame = 8663 | **PASS** |

`verdict = "PASS: §2.1-§2.5 hold on this binary and data dir"`，5/5，墙钟 15.445 s
（`d3_gate.json`，sha256 `48da324b…`）。**非 5/5 即整轮作废 —— 本轮不触发 `S-fork`。**

### 2.2 ckpt / 引擎 / 源文件断言（`--mode assert-ckpt start|final`，两次都 `all_pass=True`）

| 对象 | 期望 | 实测 | 匹配 |
| --- | --- | --- | --- |
| 引擎 `th08-modern` | `691bfd77…` | `691bfd77b7eca336…` | ✅ |
| `bombdag.pt` seed0（39,069 B） | `9c6d5d4a…` | `9c6d5d4ac7c9a612…` | ✅ |
| `bombdag.pt` seed1（39,069 B） | `36a51090…` | `36a51090d56149dc…` | ✅ |
| `bombdag.pt` seed2（39,069 B） | `885dde70…` | `885dde70c57f18bb…` | ✅ |
| `round2.pt` seed0/1/2（v4.5.0 `S`） | `4abee069… / a93746bb… / c69960fc…` | 全部匹配 | ✅ |
| 10 个受保护源文件 | 存在 | 10/10 存在 | ✅ |

### 2.3 冒烟（1 条：U-600 / seed0 / w11 / 600 决策，`--mode smoke`，exit 0）

```
survived=True  bombs_used=3  alive_frames=1200  frames=1200  decisions_executed=600
bomb_emits=9  bomb_decisions=[0,1,2,3,4,5,6,343,595]
nearest_at_emit=[24.9619, 23.8596, 19.659, 20.1392, 22.3554, 22.4882, 23.2529, 38.0295, 17.7136]
distinct_cands=18  墙钟 6.789 s
checks = {one_sample_per_decision: True, rule_bomb_matches_emits: True,
          counter_bookkeeping_closes: True, frames_consistent: True, threshold_respected: True}
gate = {hello_role_shadow: True, park_ok: True, park_frame: '8663', threads_1: True, all_pass: True}
cfg.muted_four_zero = True
```

判定：合理 —— 600 决策活满、3 颗雷全部用掉、候选 18/18 都被采到、5 项内部审计全过。
之后才启动 9 路并行。

---

## 3. 三档 × 逐窗口表（每格 3 种子 × 20 = 60 条）

### 3.1 活满率（CP95 为精确 Clopper–Pearson）

| 档 | w11 | w12 | w17 | w18 | **池化** | **CP95** |
| --- | --- | --- | --- | --- | --- | --- |
| **U-600**（20 s） | 60/60 = 100.0000 % <br> [94.04, 100] | 59/60 = 98.3333 % <br> [91.06, 99.96] | 58/60 = 96.6667 % <br> [88.47, 99.59] | 60/60 = 100.0000 % <br> [94.04, 100] | **237/240 = 98.7500 %** | **[96.3906, 99.7415]** |
| **U-900**（30 s） | 16/60 = 26.6667 % <br> [16.07, 39.66] | 11/60 = 18.3333 % <br> [9.52, 30.44] | 11/60 = 18.3333 % <br> [9.52, 30.44] | 20/60 = 33.3333 % <br> [21.69, 46.69] | **58/240 = 24.1667 %** | **[18.8918, 30.0917]** |
| **U-1200**（40 s） | 4/60 = 6.6667 % <br> [1.85, 16.20] | 0/60 = 0.0000 % <br> [0, 5.96] | 0/60 = 0.0000 % <br> [0, 5.96] | 1/60 = 1.6667 % <br> [0.04, 8.94] | **5/240 = 2.0833 %** | **[0.6798, 4.7946]** |

### 3.2 逐种子（每种子 4 窗 × 20 = 80 条）

| 档 | seed0 | seed1 | seed2 |
| --- | --- | --- | --- |
| U-600 | 80/80 = 100.0 % | 79/80 = 98.75 % | 78/80 = 97.5 % |
| U-900 | 20/80 = 25.0 % | 21/80 = 26.25 % | 17/80 = 21.25 % |
| U-1200 | 1/80 = 1.25 % | 3/80 = 3.75 % | 1/80 = 1.25 % |

12 个（种子, 窗）格的原始计数（**不是 12 个独立样本**，池化率才带 CP95）：

```
U-600 : w11 20/20 20/20 20/20 | w12 20/20 19/20 20/20 | w17 20/20 20/20 18/20 | w18 20/20 20/20 20/20
U-900 : w11  6/20  7/20  3/20 | w12  4/20  3/20  4/20 | w17  5/20  3/20  3/20 | w18  5/20  8/20  7/20
U-1200: w11  1/20  2/20  1/20 | w12  0/20  0/20  0/20 | w17  0/20  0/20  0/20 | w18  0/20  1/20  0/20
```

### 3.3 已发布对照并排

| 预算 | 本轮（60/40/20） | 对照 | 差 |
| --- | --- | --- | --- |
| 600 | **237/240 = 98.7500 %** | v8.12.0 H2s（80/60/40，同头同 12 格）**236/240 = 98.3333 %** | **+0.4167 pp**（Fisher「更差」p = 0.7752，「更好」p = 0.5000） |
| 900 | **58/240 = 24.1667 %** | v8.19.0 S1（同头同规则同种子）**58/240 = 24.1667 %** | **0.0000 pp**（逐条一致，见 §4） |
| 1,200 | **5/240 = 2.0833 %** | R5 / v8.1.0 A3（80 px，3,600 条）**36/3,600 = 1.0000 %** | **+1.0833 pp**（Fisher「更好」p = 0.1097） |

---

## 4. P1 / P2 / P3 与停止条件

### P1 — U-600 ≥ 95 %：**成立**
237/240 = **98.7500 %**，CP95 [96.3906, 99.7415]，下界 96.39 % > 95 %。
与 20 s 上原有的最优阈值 80/60/40（H2s 236/240 = 98.3333 %）相比 **+0.4167 pp**，
Fisher 单侧（U-600 更差）**p = 0.7752** —— **收紧到 60/40/20 在 20 s 上没有损失**。

### P2 — U-900 对 v8.19 S1 的可重复性：**成立，且强于判据**
- 池化：**58/240 = 24.1667 %**，CP95 **[18.8918, 30.0917]** —— 与 v8.19 S1 的 CP95 **逐位相同**（必然重叠）。
- 逐窗口：16 / 11 / 11 / 20，与 v8.19 S1 的 16 / 11 / 11 / 20 **完全相同**。
- **逐条**：240/240 条 episode 在 `survived`、`alive_frames`、`bombs_used`、`first_death_frame`、
  `bomb_emits` 五个字段上 **0 不匹配**；**240/240 条 episode 的 900 个 `actions_sampled`
  抽样序列完全相同**（`summary.json → verdict.P2.row_identity_vs_v819_S1 = identical: true`）。
- ⇒ 这套装置在 900 决策上是**确定性可复现**的。**注意这是同配置同种子的复现，不是独立重复**
  （见 §7 偏差 3）。

### P3 — U-1200 对 R5 的 1.00 %：**报，但未达「实质提升」线**
5/240 = **2.0833 %**（CP95 [0.6798, 4.7946]）对 R5 的 **1.0000 %**（36/3,600）：
**+1.0833 pp**，Fisher 单侧 p = 0.1097。预注册的门是 **≥ 3 %**，未达。
即：**收紧阈值到 40 s 只带来方向正确但幅度不足的变化；40 s 的瓶颈不是阈值远近。**

### 停止条件：**`S-unified`**
P1 成立（≥95 %）⇒ 判「**一套阈值（60/40/20）覆盖 20 s / 30 s**」，并把它定为推荐部署配置。
`S-switch`（P1 < 90 %）与 `S-mid`（90–95 %）**均未触发**；`S-fork`（保真门失败）未触发。

---

## 5. 用雷分布（P4，描述性）

| 档 | 用雷 0 / 1 / 2 / 3 | 均值 | 中位 | 标准差 | 雷耗 | 燃尽局数 | 规则触发次数 | 首放雷决策 | 末放雷决策 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U-600 | 0 / 0 / 22 / 218 | **2.9083** | 3 | 0.2892 | 698 | 218/240（90.83 %） | 2,365 | 全部 **0** | 中位 484（IQR 434.75–536.25；范围 239–597） |
| U-900 | 0 / 0 / 0 / 240 | **3.0000** | 3 | 0.0000 | 720 | 240/240（100 %） | 2,378 | 全部 **0** | 中位 515.5（466.75–563.25；315–714） |
| U-1200 | 0 / 0 / 0 / 240 | **3.0000** | 3 | 0.0000 | 720 | 240/240（100 %） | 2,396 | 全部 **0** | 中位 515.5（468.00–548.25；362–698） |

三条可直接读的事实：

1. **`first_bomb_decision` 在 720 条 episode 里全部是 0** —— PARK 停在帧 8663 时最近弹幕就已经
   在 60 px 以内，所以 60/40/20 的**第一条决策**就放雷。600 决策档里 22 条只用了 2 颗雷，是因为
   第三条决策时局已经结束，不是省雷。
2. **规则触发次数（2,365–2,396）远大于真正生效的雷数（698–720）**，约 **3.3–3.4 : 1**：
   一次放雷要过一帧才在 `bombs_remaining` 上体现，而规则在相邻决策上会重复触发
   （`bomb_ignored` 因而非零）。这与 v8.19 S1 的同一账目一致，不是本轮引入的行为。
3. **900 / 1,200 两档 240/240 局都把 3 颗雷用光**，且末放雷决策中位都在 515 附近 ——
   用雷节奏几乎不随预算变化，**预算加长只是让同一节奏多跑 300 / 600 帧**。

---

## 6. 命令、退出码与墙钟

运行器：`tools/rl/v8l0_unified.py`（收尾版 sha256 `ceae240c461f3a06c9c44a9612e42c07f4559ef132c3f2bdf6c48bd9600d1fe9`，
评测批次实际运行的版本为 `4f9ef984c7497d47…`，两者差异见 §7 偏差 6）。
Python：`/root/v400-venv/bin/python`（CPython 3.10.12，torch 2.14.1+cu130）。
所有命令都 `export SDL_AUDIODRIVER=dummy; unset DISPLAY WAYLAND_DISPLAY; export PYTHONPATH=/mnt/d/WorkShop/Touhou_RL`。

| # | 命令 | exit | 墙钟 |
| --- | --- | --- | --- |
| 1 | `v8l0_unified.py --mode preflight --phase start --waited-s 300 --out .tmp_v8l0_unified/preflight_start.json` | 0 | — |
| 2 | `v8l0_unified.py --mode assert-ckpt --phase start --engine <engine>` | 0 | — |
| 3 | `v8l0_unified.py --mode d3 --tag v8l0-fidelity --engine <engine>` | 0 | 15.445 s |
| 4 | `v8l0_unified.py --mode smoke --decisions 600 --policy-seed 0 --windows 11 --tag v8l0-smoke-U600-s0-w11 --engine <engine>` | 0 | 6.789 s |
| 5 | `--mode eval --decisions 600 --policy-seed 0 --tag v8l0-U600-d600-s0 …` | **0** | 203 s |
| 6 | 同上 `--decisions 600 --policy-seed 1 --tag v8l0-U600-d600-s1` | **0** | 198 s |
| 7 | 同上 `--decisions 600 --policy-seed 2 --tag v8l0-U600-d600-s2` | **0** | 208 s |
| 8 | `--decisions 900 --policy-seed 0 --tag v8l0-U900-d900-s0` | **0** | 269 s |
| 9 | 同上 `--decisions 900 --policy-seed 1 --tag v8l0-U900-d900-s1` | **0** | 271 s |
| 10 | 同上 `--decisions 900 --policy-seed 2 --tag v8l0-U900-d900-s2` | **0** | 280 s |
| 11 | `--decisions 1200 --policy-seed 0 --tag v8l0-U1200-d1200-s0` | **0** | 293 s |
| 12 | 同上 `--decisions 1200 --policy-seed 1 --tag v8l0-U1200-d1200-s1` | **0** | 281 s |
| 13 | 同上 `--decisions 1200 --policy-seed 2 --tag v8l0-U1200-d1200-s2` | **0** | 281 s |
| 14 | `--mode summary --budgets 600,900,1200` | 0 | — |
| 15 | `--mode audio-audit` | 0 | 37 cfg / 37 muted / **0 audible** |
| 16 | `--mode process-audit` | 0 | 本轮引擎进程 **0**（其他轮 110 台，未触碰） |
| 17 | `--mode assert-ckpt --phase final --engine <engine>` | 0 | `all_pass=True` |
| 18 | `--mode preflight --phase final --out .tmp_v8l0_unified/preflight_final.json` | 0 | — |
| 19 | `--mode publish` | 0 | — |

**评测批次 9/9 进程 exit code 全为 0**（原始表 `eval_exits.tsv`，逐条命令 `commands.tsv`）。
批次墙钟：**15:23:23 启动 → 15:28:16 全部结束 = 4 min 53 s**（最长单进程 293 s）；
720 条 episode 的 `wall_s` 合计 **1,773.8 s**（482.8 + 643.6 + 647.4）。

零 invalid：三档共 720 行，`invalid = 0`；每档 12/12 个格都是 20 行。
每条 episode 的 5 项内部审计（`one_sample_per_decision` / `rule_bomb_matches_emits` /
`counter_bookkeeping_closes` / `frames_consistent` / `threshold_respected`）720/720 全过。
每个窗口的 engine gate（`hello_role_shadow` / `park_ok` / `park_frame=8663` / `threads_1` /
`all_pass`）36/36 全过，前缀 2,165/2,165 个 checkpoint、0 不匹配。

### 音频（AGENTS.md §9.3）

- 每个私有 data 目录都由 `v450.prepare_data_dir` → `v432.build_private_data_dir` 用
  `tools/rl/v400_make_cfg.py` 生成 60 字节 `th08.cfg`；
- 每次启动前断言 `cfg.muted_four_zero == True`，否则 `UnifiedError` 直接失败（本轮未触发）；
- 进程内 `os.environ["SDL_AUDIODRIVER"] = "dummy"`，并 `pop("DISPLAY")` / `pop("WAYLAND_DISPLAY")`；
- 收尾 `--mode audio-audit`：本轮 data 树下 **37 个 `th08.cfg`，37 个 muted，0 个 audible**
  （36 个 (tag, window) 目录 + 1 个 fidelity 目录）。

---

## 7. 偏差与限制（必须一起读）

1. **开工前引擎数超限，按规则等了 5 分钟。** 引擎数在开工前一路是 **156 台**（15:17:58，
   `ps -eo comm` 统计）→ **153 台**（15:19:48，`preflight.json`）→ **99 台**（15:20:53，
   `preflight_start.json` 的 `history[0]`）→ **90 台**（15:23:20，`preflight_start.json` 当前值，
   `waited_s = 300`）。全部是**别轮**（v8i0）的进程，`own = 0`，远超「>6 先等 5 分钟」的门。
   等待期间只跑了 `--mode d3`（15:20:03，单进程、预注册要求的前置门）与 `--mode assert-ckpt`
   （纯哈希），**没有启动任何评测**；满 5 分钟、引擎数降到 90 之后才启动 9 路批次。
   全程**没有 `pkill -f`**，**没有触碰任何其他轮的进程**。结束时本轮引擎进程 **0**
   （`process_residue.json`：`matching = 0`，其他轮 110 台未被触碰）。
2. **三档同时启动（9 进程）而不是按档顺序。** 9 < v8.19 的 12 路，且批次实际只需 4 min 53 s；
   分档串行会多两次调度、多两次引擎数风险。每个进程都显式传了唯一 `--tag`
   （`v8l0-U{600,900,1200}-d{budget}-s{0,1,2}`），私有 data 目录 `<work>/data/<tag>/w<window>`，
   因此不存在 v8.17.0 那种同秒共用 data 目录的故障模式。
3. **P2 是「同配置同种子的确定性复现」，不是独立重复。** U-900 与 v8.19 S1 用了同一个候选头、
   同一条规则、同一个 `seed*1000+7`、同一个 12 格循环，所以「逐条一致」证明的是**装置可重复**，
   而不是「独立采样下率可重复」。**不能把它读成两次独立实验的一致性。**
4. **P3 的对照是跨臂的。** R5 的 1.00 %（36/3,600）是 **v8.1.0 A3**（v4.5.0 `S` 学生 + 固定
   80 px 规则，n = 3,600）的已发布格，**不是**本轮候选头 + R5 规则在 1,200 决策上的重跑。
   预注册把它固定为并排对象，所以本轮只做**描述性并排 + 参考性 Fisher**，不宣布优势。
   U-1200 的 n 是 240，R5 的 n 是 3,600，两者的 CP95 宽度差 4 倍。
5. **三个纯 stdlib 模式一度在系统 `python3` 下运行**（一条 shell 变量未展开的命令串），
   `audio_audit.json` / `process_residue.json` / `preflight_start.json` / `preflight_final.json`
   都已用 venv python 重跑，落盘的是 venv 版本；评测与 `assert-ckpt` / `summary` 一直用 venv python。
6. **`v8l0_unified.py` 在评测批次之后被改过两次**：只改 `--mode summary` 的报表代码
   （新增 P2 逐条一致性检查 `row_identity_vs_v819_S1`、preflight 的历史列表、publish 清单里
   加了 `preflight.json`）。评测路径（`mode_eval` / `run_one_episode` / `rule_decision` /
   `THRESHOLDS` / 生成器 / 循环）**未动**。三个版本号：
   `51bb5bdc8d34ee99…`（批次期间）→ `4f9ef984c7497d47…`（加 P2 逐条检查后）→
   `ceae240c461f3a06…`（收尾版，只多 publish 清单一行）。
   `ckpt_assertions_start.json` 记录的是批次期间那一版，`ckpt_assertions_final.json`
   记录的是收尾版。
7. **不能在 12 个（种子, 窗）格上做独立性推断**：同一进程里的 80 条 episode 共用一条生成器流，
   同窗的 60 条共用一个引擎父进程。池化率带精确 CP95，但**不构成 240 个 i.i.d. 样本**。
8. `bomb_emits` 与 `bombs_used` 不是同一个量：前者是规则触发次数，后者是 `bombs_remaining`
   真正减少的次数（§5 第 2 条）。引用「用了几颗雷」时只能引 `bombs_used`。

---

## 8. 不能声称什么（照预注册 §5）

- **不能声称「学会了资源分配」** —— 本轮只改手写阈值，雷是外部规则，不是策略头学出来的；
- **不能把 30 s 的 24.1667 % 与 20 s 的 98.75 % 直接比较** —— 不同预算；
- **不能声称「通关」**；
- **不能把 12 格当独立样本**；
- **不能声称 60/40/20 在 40 s 上更好** —— 只 +1.0833 pp，p = 0.1097，未过 3 pp 线。

---

## 9. 落盘清单

`Document/Update/Alpha/v8.20.0/evidence/UNIFIED/`

| 文件 | 内容 | sha256 |
| --- | --- | --- |
| `eval_U600_d600_w{11,12,17,18}_seed{0,1,2}.jsonl` | 12 × 20 行 | 见 `manifest.json` |
| `eval_U900_d900_w{11,12,17,18}_seed{0,1,2}.jsonl` | 12 × 20 行 | 见 `manifest.json` |
| `eval_U1200_d1200_w{11,12,17,18}_seed{0,1,2}.jsonl` | 12 × 20 行 | 见 `manifest.json` |
| `summary.json` | 逐档 × 逐窗口 + 池化 + CP95 + 用雷分布 + P1/P2/P3 + 停止条件 | `63400fc89139ad7a6051ae62cf0f672ac10926a139f0f3845a1e41baf7586ac0` |
| `RUN.md` | 本文件 | 见 `manifest.json` |
| `d3_gate.json` | D3 保真门 5/5 | `48da324bbbad6904031d4f10da02925fb35e5c5250ec89a373dbba67bb01e7db` |
| `ckpt_assertions_start.json` | 起：`all_pass=True` | `a1e5892fb7824322930506a9d08c5d7d7fa956e154d3f26177dfe000366c1f90` |
| `ckpt_assertions_final.json` | 终：`all_pass=True` | 见 `manifest.json` |
| `smoke.json` | 1 条冒烟 | `e8c80bb5c7c653d17b5a181ef7df01a0cb9a7cf6babeb2d2d406ef6861fc5e56` |
| `audio_audit.json` | 37 muted / 0 audible | `97c5a57d839ddb509489a957f0b80bcc89a9aaf0308ae6d7358d579caa32890b` |
| `process_residue.json` | 本轮引擎 0 | `88f4a762a02e29f97ae7b0e919d0f1e2eb53793584a3d7226f2487cf4743c323` |
| `preflight_start.json` | 引擎普查（含 5 分钟前后两次：99 → 90） | 见 `manifest.json` |
| `preflight.json` | 开工前第一次 `--mode preflight`（153 台） | 见 `manifest.json` |
| `preflight_final.json` | 结束普查（161 台，`own = 0`） | 见 `manifest.json` |
| `commands.tsv` | 9 条评测命令原文 | `13301e5e2ccb3cde28b49ebdc58270c1f9ee3199b4fabf18f7d924ff334b5c37` |
| `eval_exits.tsv` | 9 条评测的 exit code 与墙钟 | `0358eea5567cf1793ac8cc4fbf230fd90438c936806b3fcc3ab7d5fae5652945` |
| `manifest.json` | 全目录 sha256 清单 | — |
