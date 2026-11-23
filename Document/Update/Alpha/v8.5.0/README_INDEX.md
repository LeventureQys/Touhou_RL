# v8.5.0 索引 — DISTILL-001（把 R5 规则蒸馏进 36 动作策略）：**已发布 / 已独立重算**

**一句话**：单因子 = 在 bomb 头加 R5 规则的 BCE 辅助损失（λ = 0.5）。三档终值 **300 决策 99.8889 %（3,596/3,600）、600 决策 15.5000 %（558/3,600）、900 决策 2.8056 %（101/3,600）**。**P1（20 s > 19.4444 %）不成立且方向是变差**（单侧 greater p = 0.9999955、less p = 5.94356e−06、CP95 [14.3323 %, 16.7240 %] 整体在 19.4444 % 之下）；**P2（≥70 %）也不成立**（距 R5 的 77.1111 % 差 61.6111 pp）。但 **30 s 从 0.6111 %（22/3,600）升到 2.8056 %，4.59 倍（单侧 p = 8.83360e−14）**，10 s 基本不动（−0.0556 pp）。**原因是结构性的：5,166 个冻结训练态的 `bombs_remaining` 全为 3，R5 的 60 px / 40 px 两段零训练行，只有 80 px 段被实例化（正率 76.8873 %）——所以「分段规则」在原理上不可能在这批数据上被蒸馏。**

- 数据：9 格 × 1,200 条 = **10,800 条 episode**，`status: complete` ×9、0 空行、0 不可解析行、0 `invalid`
- 数据冻结：`diagnostics.json::created_at` = **2026-11-22T11:19:42+0800**；最后一个落盘格 = 11:17:44（`eval_d900_seed0.jsonl`）
- 引擎：`th08-modern` sha256 `691bfd77…8ef6`，start/final 两侧断言相符；**本文件的写作过程没有启动任何引擎**
- 统计：**做了**——P1 的单侧 Fisher + CP95 由本摘要（`SUMMARY_FOR_DECISION.md` §2）执行；p 值用**精确有理数**（`math.comb` + `Fraction`），CP95 用 Clopper–Pearson 反解。执行轮的 `diagnostics.json` 自称 `no_statistics`（它把统计留给分析轮）
- 证据：`manifest.json` **124/124** 逐文件 sha256 相符；9/9 summary 的 `out_sha256`/`out_bytes` 与副本相符；`verify.json all_pass: true`；D3 保真 **5/5**（2,165/2,165 前缀、0 mismatch）；音频 **38/38 muted、0 audible**；本轮引擎残留 **0**

---

## 1. 怎么读（三条路径）

| 你有多少时间 | 读什么 |
| --- | --- |
| **3 分钟** | 本文件 + [`SUMMARY_FOR_DECISION.md`](SUMMARY_FOR_DECISION.md) 的 **§0** 与 **§6.1（不能声称什么）** |
| **10 分钟** | [`SUMMARY_FOR_DECISION.md`](SUMMARY_FOR_DECISION.md) 全篇（§0 结论 → §2 判据逐条成立/不成立 → §3 机制 → §6 偏差清单） |
| **全过程** | [`evidence/DISTILL/RUN.md`](evidence/DISTILL/RUN.md)（执行轮记录：门与退出码、规则目标的结构性限制、训练、逐格逐窗口、有效放雷时点、偏差全表、机器负载） |

判据原文：[`PREREGISTRATION_DISTILL_001.md`](PREREGISTRATION_DISTILL_001.md)（FROZEN，83 行，4,073 字节，sha256 `e026bf612634de9acafffdccc03e9deb31636692e57a02cdd5e0ab567bbf342c`）。

---

## 2. 关键数字速查

### 2.1 三档终值（n = 3,600/档，9 格各 1,200 条）

| 档 | 活满 | 活满率 | 逐种子（各 1,200 条） | 逐窗口（各 900 条：w11/w12/w17/w18） | 用雷 |
| ---: | ---: | ---: | --- | --- | ---: |
| 300（10 s） | **3,596** | **99.8889 %** | 1,199 / 1,197 / 1,200 | 900 / 896 / 900 / 900 | 10,796 |
| 600（20 s） | **558** | **15.5000 %** | 233 / 143 / 182 | **78 / 19 / 236 / 225** | 10,800 |
| 900（30 s） | **101** | **2.8056 %** | 48 / 27 / 26 | **11 / 0 / 52 / 38** | 10,800 |

### 2.2 与两个参照并排

| 档 | **v8.5.0** | v8.3.0（纯反事实标签，我重算） | Δ vs v8.3.0 | R5 规则（v8.4.0 发布值） | Δ vs R5 |
| ---: | --- | --- | ---: | --- | ---: |
| 300 | **99.8889 %** | 99.9444 %（3,598） | −0.0556 pp | 100 % | −0.1111 pp |
| 600 | **15.5000 %** | 19.4444 %（700） | **−3.9444 pp** | **77.1111 %**（2,776） | **−61.6111 pp** |
| 900 | **2.8056 %** | 0.6111 %（22） | **+2.1944 pp（4.59×）** | **17.6111 %**（634） | −14.8056 pp |

### 2.3 判据逐条（预注册 §2 / §2.1）

| # | 冻结判据 | 结果 | 成立？ |
| --- | --- | --- | --- |
| **P1（主）** | 600 决策 > **19.4444 %**（单侧 Fisher + CP95） | **15.5000 %**；greater p = **0.9999955**、less p = **5.94356e−06**；CP95 **[14.3323 %, 16.7240 %]** | **不成立（显著更差）** |
| **P2** | 与 R5 差距；**≥ 70 %** 算成功 | 15.5000 % vs 77.1111 % → **−61.6111 pp**，达成率 20.10 % | **不成立** |
| **P3** | 300 / 900 并排（描述性） | 99.8889 % / 2.8056 %（vs 99.9444 % / 0.6111 %） | **描述性成立**（900 档 **4.5909 倍**，p = 8.83360e−14） |
| **P4** | 放雷行为（只报） | 首放决策 **0**；第 3 颗 honoured 中位 **206**；按键率 0.266 次/已执行决策（0.208 次/预算决策）；三颗花光 | **描述性成立** |
| `S-fork` | 保真门失败 → 作废 | D3 5/5、音频 38/38、残留 0、manifest 124/124、verify all_pass | **不触发** |
| `S-ok` / `S-partial` | 需 P1 成立 | P1 ✗ | **不触发** |
| **`S-fail`** | **P1 不成立** | 15.5000 % < 19.4444 % | **触发**（机制措辞「冲突」只对 20 s 成立，见摘要 §2.3） |

### 2.4 机制三行

| 层 | 实测 | 读法 |
| --- | --- | --- |
| **结构** | 5,166 态全部 `bombs_remaining = 3`；R5 的 60/40 px 段训练行 **0**；80 px 段正率 **3,972/5,166 = 76.8873 %**；`nearest_px` 中位 **36.3091** | 监督退化成「三颗雷时 80 px 内就放」，而它对 76.9 % 的态本来就成立 ⇒ **分段形状不可能被蒸馏** |
| **训练** | `BCE_bomb` 0.637 → **0.062**；bomb 决策 vs 规则 **98.01 / 98.24 / 98.01 %**；36 路准确率 89.71 / 89.70 / 89.66 % | **头学会了，学的是常数**；36 路头未被带偏 |
| **行为 / 结果** | 首雷仍在**决策 0**；第 3 颗 honoured 中位 **186 → 206**；按键率 **0.661 → 0.266**；三颗仍花光；20 s 损失全在 w11/w12（−91 / −129），30 s 增益全在 w17/w18（+45 / +33） | **不是「学会省雷」，是「换了一个分布」**；两档 alive 中位数都掉约 64 个决策 |

> **两条被证伪的说法**（不要引用）：①「辅助损失把放雷提前到决策 73–130」——盘上 `first_bomb_decision` 中位三档都是 **0**；②「20 s ≈ 8 %」——实测 **15.5000 %**（后者出自 v8.6.0 预注册，见摘要 §6.1 第 6 条 / §6.2 第 3 条）。

---

## 3. 目录清单

```
Document/Update/Alpha/v8.5.0/
├─ PREREGISTRATION_DISTILL_001.md        4,073   预注册（FROZEN，未改动）
├─ SUMMARY_FOR_DECISION.md                        决策摘要（§0–§6，P1/P2/P3 的检验在此）
├─ README_INDEX.md                                本文件
└─ evidence/DISTILL/                    125 个文件 / 75,863,189 字节（清单覆盖 124 个 / 75,845,862 字节，
    ├─ eval_d{300,600,900}_seed{0,1,2}.jsonl (9)  `manifest.json` 自身除外）
    │                                            原始数据（各 1,200 行、4 个 300 行的窗口块，w11→w12→w17→w18）
    ├─ eval_d{300,600,900}_seed{0,1,2}.summary.json (9)  每格的 status / out_sha256 / out_bytes / 计数
    ├─ rule_targets.jsonl               1,794,237  5,166 条 R5 规则目标（逐态 nearest_px / 阈值 / rule_bomb / reason）
    ├─ rule_target_stats.json               2,784  目标统计（**by_bombs_remaining 只有键 "3"**）+ 特征复现证明
    ├─ diagnostics.json                    40,410  三档逐种子/逐窗口、用雷、有效放雷时点、side_by_side（**无检验**）
    ├─ manifest.json                       17,327  124 个文件的 sha256 + 字节
    ├─ verify.json                        125,615  独立复算（另一套实现）：目标 / 每格 / 每文件，all_pass
    ├─ publish.json                        37,716  发布复核
    ├─ bomb_usage.json                     31,074  逐 episode 的 honoured 派生与分布
    ├─ v830_honoured.json                   5,648  v8.3.0 的有效放雷时点（对照组，只读来源）
    ├─ train_summary.json / train_seed{0,1,2}.json  训练报告（损失、准确率、ckpt sha256）
    ├─ ckpt_manifest.json / ckpt_assertions{,_final}.json  三个 ckpt 的 sha256 + 7 个被导入件的 pin
    ├─ preflight.json                       4,426  单因子声明、pin、冻结超参、launch_env（SDL_AUDIODRIVER=dummy）
    ├─ fidelity_d3.json                    26,626  D3 保真门（5/5 PASS、2,165/2,165）
    ├─ audio_audit.json                     4,669  静音审计（38/38 muted、0 audible）
    ├─ process_residue{,_final}.json          614  进程残留（matching 0 / 48 个别的轮的引擎未动）
    ├─ commands.tsv                        13,488  每次真实调用一行（exit / wall / argv；含 3 次重复的收尾步骤）
    ├─ logs/                             (25)      本轮全部 .out
    ├─ runs/                             (51)      逐格 run.log + 每窗口 w1X-stderr.log（**逐窗口推进的证据**）
    └─ scripts/                           (9)      exec/chain/chain2/grid/finish/status + tables.py / verify.py /
                                                   v830_honoured.py
```

逐文件 sha256 / 字节：`evidence/DISTILL/manifest.json` 的 `files`（**124 条**）。**`verify.json` 是唯一不被清单覆盖的文件**（它在清单封闭之后写，用来验证清单本身）。

---

## 4. 复现（离线，不启引擎）

**先读这一条**：`scripts/` 里三个脚本都是**工作目录 `.tmp_v850_distill/` 原件的副本**，路径由 `__file__` 推导，所以它们各按各的目录结构工作：

| 脚本 | 它找什么 | 会不会写文件 | 能否在 `evidence/DISTILL/` 下直接跑 |
| --- | --- | --- | --- |
| `scripts/tables.py` | 同目录的 `diagnostics.json` / `train_summary.json` / `rule_target_stats.json`（**三者都在本包里**） | **不写**，只打印到 stdout | **能，安全（只读）** |
| `scripts/verify.py` | 同目录的 `eval/`、`train/`、`rule_targets.jsonl`、`ckpt_manifest.json`，以及 `../.tmp_v830_bombpol/labels_bomb36.jsonl` | **会覆盖同目录的 `verify.json`** | **不能**：`eval/`、`train/` 与标签文件都不在发布树里，而且写入等于改动既有证据 |
| `scripts/v830_honoured.py` | 只读 `.tmp_v830_bombpol/eval/` | 默认写到 `evidence/DISTILL/v830_honoured.json`（**覆盖**） | **不建议** |

**安全的只读复现**（不写任何文件，实测可用）：

```powershell
$py = "python"
& $py Document\Update\Alpha\v8.5.0\evidence\DISTILL\scripts\tables.py
```

预期输出三档 `survived` = **3,596 / 558 / 101**（full-rate 99.89 % / 15.50 % / 2.81 %）、`bombs spent (honoured)` = 10,796 / 10,800 / 10,800，与 §2 一致。

要跑 `verify.py` 必须回到 `.tmp_v850_distill/`（那才是它被设计运行的位置，`eval/`、`train/`、`rule_targets.jsonl`、`ckpt_manifest.json`、`diagnostics.json` 都在那里），并预期它**重写** `verify.json`；发布副本的 sha256 见 §6。

---

## 5. 与姊妹轮 / 下游的关系

| 轮次 | 它给本轮什么 | 本轮给它什么 |
| --- | --- | --- |
| [v8.3.0](../v8.3.0/PREREGISTRATION_BOMBPOL_001.md) | 5,166 个状态 × 36 路反事实标签（原样复用，未重采）；99.94 / 19.44 / 0.61 % 的基线 | 20 s −3.94 pp、30 s +2.19 pp 的对照 |
| [v8.4.0](../v8.4.0/SUMMARY_FOR_DECISION.md) | **R5 规则**（`staged`：3→80 / 2→60 / 1→40 px）作为唯一监督源；R5 = 77.1111 % / 17.6111 % | R5 在本轮的对照值（差距 61.6111 / 14.8056 pp） |
| [v8.6.0](../v8.6.0/PREREGISTRATION_BOMBDAGGER_001.md)（下游，未跑） | — | **本轮的 P0 结论**：要让「省雷」可学，必须让训练态覆盖 `bombs_remaining ∈ {2,1}`（v8.6.0 的 P0 正是这条，要求各 ≥ 200 态）。**注意**：v8.6.0 预注册把本轮 20 s 写成「~8 %」，盘上是 15.50 %（摘要 §6.2 第 3 条） |

**下一步（按预注册的判定式）**：P1 不成立 ⇒ **`S-fail`**，冻结文本的动作是「下一轮改两阶段：先按规则训 bomb 头、冻结后再训候选头」。但该动作**目前没有对应的预注册**，而 v8.6.0 选的是另一条路（改状态分布、不再加辅助损失）。**这是需要决策的分叉，不是数据缺陷。**

---

## 6. 自哈希（实测）

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| `SUMMARY_FOR_DECISION.md` | 35,525 | `cbef998dba7011d0657b44382b6bad345a44b32cffd867c587c75cecceca3033` |
| `README_INDEX.md` | — | 本文件无法含自身哈希（见下） |
| `PREREGISTRATION_DISTILL_001.md` | 4,073 | `e026bf612634de9acafffdccc03e9deb31636692e57a02cdd5e0ab567bbf342c` |
| `evidence/DISTILL/RUN.md` | 32,935 | `713077c66479c1a5e60651c590697ed482063150e6be5056ba9998f1d2968e89` |
| `evidence/DISTILL/manifest.json` | 17,327 | `d8250577a5bc88177fd6aab3e13cb27ff31dc610aa661ffaf61c6704f728ea32` |
| `evidence/DISTILL/diagnostics.json` | 40,410 | `2d321a9a8d5cefc6f115fd535713efc9fc77217c80fc53b8f64379b62e4cf02d` |
| `evidence/DISTILL/verify.json` | 125,615 | `f078170f4e980001e999a4a1c0e0ecf472aff24000dac67ed92665c0c24e9add` |
| `evidence/DISTILL/publish.json` | 37,716 | `f354144a13e9946e477006601858fe09b58e501bed3ad8112681489017390066` |
| `evidence/DISTILL/rule_targets.jsonl` | 1,794,237 | `a7b788e18687ce7fdab1c99d2b645b44d2499dc35427597285982b825c9fdc83` |
| `evidence/DISTILL/commands.tsv` | 13,488 | `63321dcdb8854ee0057b308363faf8c0d81dc173fd1bc156178a6a47e5d23778` |
| `evidence/DISTILL/scripts/verify.py` | 10,848 | `da08cbb7904c1a004f275e6553e377cf064950c434cc25a843c2b2b734f18f15` |
| `evidence/DISTILL/scripts/tables.py` | 10,302 | `56842d2c4e538b0737a54168b3a0576562b30bfc908d14884a3cdbb1ca5a288a` |

> `README_INDEX.md` 的自哈希由**报告**给出（本文件不能包含自身哈希而不自指）。字节数与 sha256 都是 `Get-FileHash` / `hashlib` 在写入后实测。

---

## 7. 写入边界与合规

- **本索引与摘要的写作过程没有启动引擎、没有跑评测、没有重训**；全部数字来自 9 份已发布 JSONL 的离线逐行重算 + 已发布 JSON 的交叉核对。
- **没有改动任何既有证据**：`RUN.md` / `manifest.json` / `diagnostics.json` / `verify.json` / `publish.json` / `commands.tsv` / 9 份 eval JSONL 与 summary / `scripts/**` / `logs/**` / `runs/**` 全部保持发布时状态；**预注册未动**；没有 `git add` / `commit`；没有删除或移动任何资产；`.tmp_v850_distill/**` 与 `.tmp_v830_bombpol/**` 只读。
- **静音（AGENTS.md §9）**：执行轮 38/38 cfg muted；本轮写作零新增 cfg。
- **口径提醒**：三档是**三个不同终点**，不能按均值横向比较（`diagnostics.json → budget_comparison_warning`）；3,600 条不是独立样本（4 窗口 + 3 种子都是固定效应；w12 在 900 档三 seed 合计 **0/900**）。
