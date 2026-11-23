# 索引 — v8.1.0 放雷规则能撑多远：600 / 900 / 1,200 决策三档预算（BOMBFAR-001，接手从这里开始）

> v8.1.0 **只做一件事**：把 v8.0.0 那条已经证明有效的放雷规则（**A3，最近子弹 < 80 px 且还有雷 ⇒ 该决策首帧按雷**）从 **300 决策** 推到 **600 / 900 / 1,200 决策**（1,200 / 1,800 / 2,400 帧），看它还撑不撑得住。
> 单因子 = **预算**；策略固定为 v4.5.0 `S` 的三个**冻结** `round2.pt`；起点每次重新 `PARK` 到 frame **8,663**；4 dev 窗口 × 3 种子 × 每格 300 条 = **每档 3,600 episode**；**无雷基线 = A0 × 1,200 决策 × 种子 0 × 4 窗 × 300 = 1,200 条**；另有**预注册外**的 **A3 × 300 决策 × 3 种子 = 3,600 条控制臂**。
> **结论：600 决策 = 2,560/3,600 = 71.1111%（CP95 [69.5999%, 72.5879%]）、900 决策 = 480/3,600 = 13.3333%（[12.2391%, 14.4875%]）、1,200 决策 = 36/3,600 = 1.0000%（[0.7013%, 1.3818%]）；A0 × 1,200 = 0/1,200 = 0.0000%。三档没有一档是 100%。P1 成立（Fisher p = 3.865606e-575）。**
> **四条不可省的边界**：①**600 决策档没有复现 v8.0.0 的 100%**，原因是**预注册 §0 把 v8.0.0 的 300 决策 / 600 帧误写成「600 帧」**并当作 600 决策的复现检查点；控制臂 **A3 × 300 = 3,600/3,600 = 100%** 且与 v8.0.0 **逐 episode 15 字段一致** ⇒ **管线保真、跌幅来自段变长**。②**71.1111 % 与 13.4444 % 不可比**（1,200 帧 vs 600 帧；**同口径是 0.1667%**）。③**跨预算不是同一条轨迹**（600 档 1,040 条有效行里只有 6 条与 1,200 档同死亡帧）。④预注册的 P1 是**计数比较**，分析轮的 Fisher / CP95 / 同口径基线都是**补充证据**。

## 1. 你是谁，先读哪份

| 目的 | 读这份 | 规模 |
| --- | --- | --- |
| **要一个结论** | [`SUMMARY_FOR_DECISION.md`](SUMMARY_FOR_DECISION.md) | 六节：§0 一句话（四件事：600 档为何不是 100% / 三颗雷 15 秒烧光 / P1 成立 + 停止条件 / **同口径 A0 存活曲线**）→ §1 改了什么（含分析层与 runner 字节身份）→ §2 判据表（P1/P2/P3 + 停止条件）→ §3 有效性诊断（H1–H4 归因 + 分母差异 + A0 曲线）→ §4 并排 → §5 能/不能声称（11 条）→ §6 合规、七个开放项与**十一条文档间冲突** |
| **要核统计结论** | [`evidence/BOMBFAR/ANALYSIS.md`](evidence/BOMBFAR/ANALYSIS.md) + [`evidence/BOMBFAR/analysis.json`](evidence/BOMBFAR/analysis.json) | ANALYSIS.md **23,455 字节**（269 行 / 8 节：§0 结论 → §1 三档主表 + 逐种子 + 逐窗口 + 行级完整性 → §2 P1（**含同口径警告与 A0 存活曲线**）→ §3 复现检查（**双向核对**）→ §4 机制量化 → §5 停止条件 → §6 与 RUN.md 对数 → §7 不能声称 → §8 可复现性）；analysis.json **83,818 字节**（schema `v810-bombfar-analysis/1`、`created_utc = 2026-11-22T02:03:33Z`、`scope.engine_started = false`） |
| **要核某个数字** | [`evidence/BOMBFAR/diagnostics.json`](evidence/BOMBFAR/diagnostics.json) | 35,005 字节，sha256 `b401ef8b…`：`definitions` / `inputs`（逐文件 `rows`+`invalid_rows=0`+`sha256`）/ `budgets.{600,900,1200}`（各含 `per_seed`、`per_window`、`bomb_usage`、`exhaustion`、`last_bomb_decision`、`first_bomb_decision`、`remaining_decisions_after_last_bomb`）/ `control_d300` / `baseline.A0_d1200` / `cross_budget_rates` / `p1_criterion` / `reproduction_v800` |
| **要核「600 档为什么不复现 100 %」** | [`repro_v800.json`](evidence/BOMBFAR/repro_v800.json) + [`RUN.md`](evidence/BOMBFAR/RUN.md) §6 + `ANALYSIS.md` §3 | 391,100 字节：`fields_compared` **15 个**、3 种子各 `1,200 / 1,200`、`episodes_with_differences = 0`、`all_equal = true`；分析轮**双向核对**（声明 → 原始；原始 → 声明）+ 两侧 sha256 重算；RUN.md §6 是那条口径的完整说明（**未修改预注册**） |
| **要复跑实验** | [`RUN.md`](evidence/BOMBFAR/RUN.md) + [`commands.tsv`](evidence/BOMBFAR/commands.tsv) + [`scripts/exec.sh`](evidence/BOMBFAR/scripts/exec.sh) | RUN.md 19,610 字节（sha256 `e683b88a…`，250 行）：§1 设计与 runner 改动、§2 门与断言、§3 运行清单与墙钟、§4 三档结果、§5 逐条对判据、§6 复现检查、§7 **十条偏差**、§8 不能声称、§9 文件清单；`commands.tsv` 8,415 B（**22 条 argv，22/22 `exit = 0`**） |
| **要复跑统计** | `python .tmp_analysis_80/run_all.py` | 除 `created_utc` 外**逐字节可重生成**；`stats_core.self_test()` 全通过（scipy 1.18.1 对拍） |
| **要知道判据怎么定的** | [`PREREGISTRATION_BOMBFAR_001.md`](PREREGISTRATION_BOMBFAR_001.md) | 85 行，4,314 字节（FROZEN 2026-11-22，sha256 `b2609a35…`）：P1（各预算的活满率 + 与基线比较；**计数比较、无检验**）/ P2（雷耗尽时的剩余预算，**描述性**）/ P3（三档存活帧，**描述性、跨档不可比**）；`S-fork` / `S-100` / **`S-drop`** / **`S-floor`** |
| **要查机制（雷什么时候烧光）** | `analysis.json → tiers.*` / `diagnostics.json → budgets.*` | 三档第一颗雷**全部决策 0**；最后一颗**指令**中位 **433.0 / 435.5 / 435.0**（≈ **14.5 s**；**被接受**的在 600 档是 **432.5**）；耗尽 **98.3333% / 100% / 100%**；烧光后剩余中位 **166.5 / 463.5 / 764.0**（占预算 **27.75% / 51.50% / 63.67%**） |
| **要查同口径基线** | `analysis.json → a0_survival_curve` + `tiers_vs_horizon_matched_A0` | A0 × 1,200 决策这一条运行里「活到第 f 帧」：**600→235（19.5833%）/ 900→67（5.5833%）/ 1,200→2（0.1667%）/ 1,800→0 / 2,400→0**；同口径 Fisher **4.467089e-495 / 1.848037e-64 / 3.041551e-5** |
| **要查跨预算能不能逐条比** | `analysis.json → cross_budget_rng` + `ANALYSIS.md` §3.1 | **不能**：600 档 1,040 条有效行里只有 **6** 条与 1,200 档同死亡帧、`bomb_decisions` 前缀只有 **8/3,600** 相同；A0 上「截断后相同」只有 **125/1,200** |
| **要查逐种子撕裂** | `analysis.json → tiers.*.per_seed` | 600 档 **96.4167% / 93.6667% / 23.2500%**（极差 **73.17 pp**）；900 档 **14.6667% / 25.3333% / 0%**；1,200 档 **1.1667% / 1.8333% / 0%**（**seed 2 在 900 与 1,200 两档都是 0/1,200**） |
| **要查 runner 到底跑的是哪一份字节** | [`v800_bomb.used.py`](evidence/BOMBFAR/v800_bomb.used.py) + [`v800_bomb.pre_v810_patch.py`](evidence/BOMBFAR/v800_bomb.pre_v810_patch.py) + `ckpt_assertions*.json` | 本轮跑的是 **`29244683…`**（77,827 B，由「快照 + 4 处字面编辑」逐字节重建并验证）；快照 = **`7fcf5f26…`**（77,712 B，= **v8.0.0 的 runner 字节**）；`ckpt_assertions_final.json` 的 `f884a4be…` **不是**本轮运行的字节（并发 v8.2.0 回合 08:57:59 再改后的现状） |
| **要查默认路径有没有被改坏** | [`default_path_equivalence.json`](evidence/BOMBFAR/default_path_equivalence.json) | 911 字节：打补丁前后同默认参数各跑一次，**3/3** episode、每 episode 比较 **44** 个字段（排除 `handle` / `quit_reply` / `spawn_ms` / `wall_s`）、`all_equal = true` |
| **要查保真门（`S-fork`）** | [`fidelity_d3.json`](evidence/BOMBFAR/fidelity_d3.json) | 26,509 字节：**5/5 PASS**；每个 (cell, window) 另走一遍前缀 + PARK，`gate.all_pass = true` |
| **要查冻结对象** | [`ckpt_assertions.json`](evidence/BOMBFAR/ckpt_assertions.json)（start）+ [`ckpt_assertions_final.json`](evidence/BOMBFAR/ckpt_assertions_final.json)（final） | 各 2,675 字节；3 个 `round2.pt`（各 34,387 B）**两次都 3/3 match**；5 个受保护源逐条列出；引擎 `691bfd77…` = 冻结值 |
| **要查评测原始行** | `eval_A3_d{600,900,1200}_seed{0,1,2}.jsonl` + `eval_A0_d1200_seed0.jsonl` + `eval_A3_d300_seed{0,1,2}.jsonl`（**13 个 jsonl，各 1,200 行**）+ 13 份 `*.summary.json` | **15,600 条**：三档 10,800 + A0 1,200 + 控制臂 3,600；行级自检（`invalid = 0`、`error = 0`、`survived == alive_cap`、列表长度自洽、`used == honoured`）**全部通过** |
| **要查音频/进程合规** | [`audio_audit.json`](evidence/BOMBFAR/audio_audit.json) + [`process_residue.json`](evidence/BOMBFAR/process_residue.json) | 6,888 字节：**57 cfg / 57 decoded / 0 errors / 57 muted / 0 audible**、`all_muted = true`；487 字节：`matching = 0`、`all_pass = true`（`processes_total_by_engine_name = 73` 属并发回合，**不在过滤器内**） |
| **要查发布清单** | [`manifest.json`](evidence/BOMBFAR/manifest.json) | 9,643 字节：`file_count = 69`、`total_bytes = 33,701,296`、`excludes = "manifest.json"`。⚠️ **清单 09:36:38 发布，早于分析产物（10:03:33）** ⇒ 目录现在 **72 个文件 / 33,818,212 B**；分析轮自己也核对过 **39 个顶层文件**，`mismatch_count = 0`、`missing_count = 0` |
| **要知道这份索引之外还有什么** | [v8.0.0](../v8.0.0/SUMMARY_FOR_DECISION.md) · [v7.0.0](../v7.0.0/SUMMARY_FOR_DECISION.md) · [v7.1.0](../v7.1.0/SUMMARY_FOR_DECISION.md) · [v4.5.0](../v4.5.0/SUMMARY_FOR_DECISION.md) | **直接前驱**（放雷在 300 决策 = 100%、判 `S-ok`；本轮就是问「再往下推会怎样」）/ 控制频率 ⇒ 2 帧最优 / 在线自适应 `S-degrade` / 三个冻结 ckpt 与 `13.4444%` 的来源 |

> **分析产物是「后补」的，而且登记期间还在被重写**：`RUN.md` / `manifest.json` 的 mtime 是 **09:36:34 / 09:36:38**，`analysis.json` / `ANALYSIS.md` 的第一版是 **09:54:49**、本索引钉的最后一版是 **10:03:33**（登记期间实测被重写 ≥4 次：`analysis.json` 81,122 → 81,156 → **83,818** B；`ANALYSIS.md` **0（重写中）** → 21,752 → 21,767 → 23,290 → **23,455** B）。所以 ①`manifest.json` 的 69 条**不含**它们（分析轮在 `manifest_crosscheck.note` 里写明「deliberately not in the manifest」）；②本索引钉的是 **10:03:33 之后连续 ≥80 s 不变**的最后一版；③**0 字节中间态不可引用**；④**关键统计量在各版之间逐字相同**。
>
> **预注册不要求检验**：P1 只要求「给出各预算的活满率并与基线比较」（`p1_criterion.note`：`"a deterministic count/rate comparison, not a significance test"`），P2/P3 明写描述性。分析轮的 Fisher / CP95 / 同口径基线是**补充证据**，判定本身是**计数比较**。

## 2. 三十秒版本

- **只动了预算**：`--decisions` 从冻结的 300 直通到 600 / 900 / 1,200（runner 的 **4 处字面编辑**，默认仍 300）；放雷规则钉死 A3（80 px）；观测、模型、18 候选、死亡规则、起点 8,663、采样器与 RNG 全**一行未改**。
- **结果**：600 决策 **2,560/3,600 = 71.1111%**、900 决策 **480/3,600 = 13.3333%**、1,200 决策 **36/3,600 = 1.0000%**；无雷基线 A0 × 1,200 **0/1,200 = 0.0000%**；控制臂 A3 × 300 **3,600/3,600 = 100.0000%**。
- **P1 成立**：71.1111% > 13.4444%（预注册 §2 的字面判据）；附加单侧 Fisher **p = 3.865606e-575**（表 `[[2560,1040],[484,3116]]`）、CP95 **[69.5999%, 72.5879%]**；参照值 13.4444% 由分析轮从 v8.0.0 原始文件重算（484/3,600）。
- **600 档不是 100%，而且不是管线的问题**：预注册把 v8.0.0 的 **300 决策 / 600 帧**写成「600 帧」并拿去对 600 **决策**（1,200 帧）；**控制臂与 v8.0.0 在 15 个冻结字段上 3,600/3,600 逐 episode 一致**（双向核对 + 两侧 sha256）⇒ **管线保真、跌幅来自段变长**。
- **跨预算不是同一条轨迹**：600 档 **1,040 条**在自己上限内死亡，与 1,200 档同死亡帧的只有 **6/1,040**、`bomb_decisions` 前缀只有 **8/3,600**；A0 上「截断后相同」只有 **125/1,200** ⇒ **拿 600 档直接比 v8.0.0 在构造上就不成立**，这才是 300 控制臂存在的理由。
- **机制是「三颗雷开头烧光」**：第一颗雷在**三档与控制臂的全部 episode**上都是**决策 0**（起手 frame 8663 已满足 < 80 px）；最后一颗**指令**中位第 **433.0 / 435.5 / 435.0** 个决策 ⇒ **≈ 14.5 s**（60 fps、2 帧/决策）⇒ **三颗雷在开头 15 秒内烧光**。
- **雷耗尽 98.3333% / 100% / 100%**；烧光之后还剩中位 **166.5 / 463.5 / 764.0** 个决策**裸奔**（占预算 **27.75% / 51.50% / 63.67%**）—— 这就是 71.1111% 与 100% 的差距来源。
- **同口径基线（分析轮补充）**：A0 存活曲线 **19.5833% / 5.5833% / 0.1667% / 0% / 0%**（600 / 900 / 1,200 / 1,800 / 2,400 帧）⇒ 600 档的同口径基线是 **0.1667%**，不是 13.4444%；900 与 1,200 档的同口径基线都是 **0/1,200**。
- **逐种子在 600 档就已撕裂**：**96.4167% / 93.6667% / 23.2500%**（极差 **73.17 pp**）；900 / 1,200 两档 seed 2 都是 **0/1,200**。
- **停止条件**：**`S-drop` 在最低一档（600 决策 / 1,200 帧）即触发**（三档全跌破；该档用雷分布 **0 / 60 / 3,540**、耗尽 98.3333%）；**`S-floor` 触发**（A0 = 0/1,200）；**`S-100` 未触发**（1,200 决策 = 1.0000%）；**`S-fork` 未触发**（5/5、ckpt 两次 3/3、控制臂复现）。
- **默认路径没被改坏**：打补丁前后同默认参数 **3/3** episode、每 episode **44** 字段全同。
- **不能做**：不能把 71.1111% 与 13.4444% 当同口径（1,200 帧 vs 600 帧，同口径 0.1667%）；不能说「管线坏了 / v8.0.0 不可复现」；不能说「600 档复现了 100%」；不能说「A3 的雷白放了」（**没有**同段同种子的有雷/无雷配对臂）；不能做跨档均值比较（删失上限不同）；不能把 Fisher/CP95 说成「预注册要求的检验」；不能把 15,600 条当独立样本（**12 格**，逐种子极差 73.17 pp）；不能把 `ckpt_assertions_final.json` 的 runner sha 当成本轮运行的字节。

## 3. 证据一览（`evidence/BOMBFAR/`，**72 个文件** = 清单 69 ＋ `manifest.json` 自身 ＋ 2 个分析产物）

| 组 | 文件 | 这一组在做什么 |
| --- | --- | --- |
| 判据 | （上一级）[`PREREGISTRATION_BOMBFAR_001.md`](PREREGISTRATION_BOMBFAR_001.md) | P1（三档活满率 + 基线比较，**计数比较、无检验**）/ P2（耗尽时的剩余预算，描述）/ P3（三档存活帧，描述、**跨档不可比**）；`S-fork` / `S-100` / **`S-drop`** / **`S-floor`** |
| **分析** | [`analysis.json`](evidence/BOMBFAR/analysis.json)（81,156 B）+ [`ANALYSIS.md`](evidence/BOMBFAR/ANALYSIS.md)（21,767 B）+ `.tmp_analysis_80/{analyze_v810.py,run_all.py,stats_core.py}`（**未发布**） | P1 的 Fisher/CP95、**A0 存活曲线与同口径基线**、复现的双向核对、`cross_budget_rng`、停止条件、`manifest_crosscheck`、`run_md_crosscheck`、`self_test` |
| 三档网格 | `eval_A3_d{600,900,1200}_seed{0,1,2}.jsonl`（**9 个，各 1,200 行**）+ 9 份 `*.summary.json` | **10,800 条**的主力数据 |
| 无雷基线 | `eval_A0_d1200_seed0.jsonl`（1,200 行）+ summary | **0/1,200**；只有 **1 个种子**；**同一条运行还提供存活曲线** |
| 复现控制臂 | `eval_A3_d300_seed{0,1,2}.jsonl`（**3 个，各 1,200 行**）+ 3 份 summary | **预注册外**；与 v8.0.0 逐 episode **15 字段一致**，不参与三档并列表 |
| v8.0.0 复现 | [`repro_v800.json`](evidence/BOMBFAR/repro_v800.json) + `scripts/repro_v800.py` | 3 种子 × 1,200 逐 episode；**注意**：两份文件 sha256 **不同**（`6cf4db90…` vs `016e5a3f…`），一致的是**字段**不是字节 |
| 诊断 | [`diagnostics.json`](evidence/BOMBFAR/diagnostics.json) + `scripts/diagnose.py` | 三档 + 控制臂 + A0 的 n/率/存活帧/用雷分布/耗尽/首末雷位置/剩余预算；`cross_budget_rates` 只并排**同分母率** |
| 放雷审计 | [`bomb_audit.json`](evidence/BOMBFAR/bomb_audit.json) + `scripts/diagnose.py` | schema `v810-bombfar-bomb-audit/1`、`tiers` 五键、`all_pass = true`；**不是** v8.0.0 那个 `--mode audit` 的产出（glob 不匹配），同名不同物 |
| 门 1：保真 | [`fidelity_d3.json`](evidence/BOMBFAR/fidelity_d3.json)、[`default_path_equivalence.json`](evidence/BOMBFAR/default_path_equivalence.json) | 5/5 断言；默认路径 **3/3** episode × **44** 字段全同 |
| 门 2：冻结 | [`ckpt_assertions.json`](evidence/BOMBFAR/ckpt_assertions.json)、[`ckpt_assertions_final.json`](evidence/BOMBFAR/ckpt_assertions_final.json) | 3 个 ckpt **两次都 3/3**、5 个受保护源、引擎哈希；**runner sha 有两个值**（`29244683…` = 本轮跑的、`f884a4be…` = 并发回合改后的现状） |
| runner 字节 | [`v800_bomb.used.py`](evidence/BOMBFAR/v800_bomb.used.py)、[`v800_bomb.pre_v810_patch.py`](evidence/BOMBFAR/v800_bomb.pre_v810_patch.py)、`scripts/reconstruct.py` | `29244683…`（77,827 B，本轮实际运行）由「快照 + 4 处字面编辑」**逐字节重建并验证**；快照 `7fcf5f26…`（77,712 B）= v8.0.0 的 runner |
| 执行 | [`RUN.md`](evidence/BOMBFAR/RUN.md)、[`commands.tsv`](evidence/BOMBFAR/commands.tsv)、`logs/*.out`（**23 个**）、`scripts/`（**7 个**：`exec.sh` / `diagnose.py` / `repro_v800.py` / `reconstruct.py` / `regress_check.py` / `regress.sh` / `show.py`） | **22 条 argv / 22 条 `exit = 0`**；每个 cell 的完整 stdout；`RUN.md` §7 **十条偏差**全列 |
| 音频/进程合规 | [`audio_audit.json`](evidence/BOMBFAR/audio_audit.json)、[`process_residue.json`](evidence/BOMBFAR/process_residue.json) | **57 个** `th08.cfg` 全 muted、**audible 0**；本轮引擎进程 **0**（并发回合的 73 个在过滤器外） |
| 发布 | [`manifest.json`](evidence/BOMBFAR/manifest.json) | `file_count = 69` / `total_bytes = 33,701,296` / `excludes = "manifest.json"`；登记时对 **13 个关键文件**重算，**13/13 相符**；分析轮对 **39 个顶层文件**核对，`mismatch_count = 0` |

## 4. 判据与停止条件（预注册 §2 / §2.1）

| 条目 | 条件 | 本轮结果 |
| --- | --- | --- |
| **P1（主）** | 600 决策 **> 13.4444%**（作为复现检查）；900 / 1,200 各给率并与不放雷基线比较 | ✅ **成立**：600 = **71.1111%**（Fisher **3.865606e-575**、CP95 [69.5999%, 72.5879%]）；900 = **13.3333%**、1,200 = **1.0000%**；A0 × 1,200 = **0.0000%**（**分母 1,200 ≠ 3,600**）。⚠️ **600 档不是「复现 100%」**——它是 1,200 帧的率 |
| **P2（描述）** | 雷耗尽时的剩余预算比例：3/3 占多少、它们是否仍活满整段 | 3/3 占比 **98.3333% / 100% / 100%**；其中仍活满 **70.6215% / 13.3333% / 1.0000%**；最后一颗**指令**中位第 **433.0 / 435.5 / 435.0**，之后剩 **166.5 / 463.5 / 764.0**（**27.75% / 51.50% / 63.67%**） |
| **P3（描述）** | 三档存活帧均值/中位（**跨档不可比**） | 1155.3233 / 1400.1844 / 1413.8400；中位 1200 / 1396 / 1396.5 —— **删失上限 1,200 / 1,800 / 2,400 不同** |
| `S-fork` | 保真门失败 | **未触发**（5/5、ckpt 两次 3/3、每窗前缀 + PARK、控制臂复现） |
| **`S-100`** | 1,200 决策仍 100% | **未触发**（1.0000%，36/3,600） |
| **`S-drop`** | 某一档跌破 100% | ✅ **触发，且在最低一档**：三档全跌破（71.1111% / 13.3333% / 1.0000%），跌破预算 = **600 决策 / 1,200 帧**；该档用雷分布 **0 / 60 / 3,540**、3/3 耗尽 **98.3333%** |
| **`S-floor`** | 1,200 决策的不放雷基线也为 0 | ✅ **触发**（A0 = 0/1,200，与 v5.1.0 一致 ⇒「无需特殊处理」） |
| **实际结果** | **`S-drop` + `S-floor` 触发；`S-100` / `S-fork` 未触发** | 预注册对 `S-drop` 的处置：**下一轮训练 36 动作（含放雷）的策略，DAgger 覆盖该预算** |

> **必须一起读的一句**：`S-drop` 与 `S-floor` **同时**触发不是矛盾——`S-drop` 说的是「有雷也掉下 100%」，`S-floor` 说的是「没雷在 1,200 决策上也是 0」。两者合起来的读法是：**雷把率从 0% 抬到 1.0000%（1,200 决策）、13.3333%（900）、71.1111%（600），但没有一档保住 100%。**

## 5. 开放项

1. **`S-drop` 的处方还没做**：预注册写的是「下一轮训练 **36 动作（含放雷）** 的策略，**DAgger 覆盖该预算**」。本轮只**定位**了跌破档（600 决策 / 1,200 帧）与该档用雷分布（0 / 60 / 3,540），**一条处方都没跑**。
2. **规则本身该先改**：三颗雷在开头 14.5 s 烧光，因为规则**起手就触发**且**不看「已经有一颗在生效」**（76,198 次指令里 65,458 次被无敌帧忽略，忽略占比 85.91%）。「**lethal-only + 不重复放 + 留一颗给后半段**」是最省的下一个变体（v8.0.0 §3.4 已备好 lethal-only 口径的读数：4,755–8,017 条决策）。
3. **900 档的同口径基线是分析轮补的，不是预注册给的**：预注册 §1 引的 v5.1.0/v5.4.0「0/1,200」是 **600 帧口径**；分析轮用 A0 存活曲线补出了 1,800 帧 = 0/1,200。**若要正式当判据用，需要新预注册。**
4. **1,200 档的 A0 只有 1 个种子**（1,200 条 vs 3,600 条）⇒ 只能并排率、必须带 `n`；`RUN.md` §7 偏差 4 已披露。
5. **seed 2 是结构性例外**：600 档 23.2500%、900 与 1,200 档都是 **0/1,200**，且其 1,200 决策档 `max = 1,682 < 2,400`。**任何「平均而言」都要先说明这是 3 个种子的池化**（极差 73.17 pp）。
6. **没有「有雷/无雷」的配对臂**：本轮 A0 与 A3 的段长不同、generator 流因段长而不同（§3.1 H4 已量化）⇒ 「A3 的雷有没有白放」**本轮无法回答**（`RUN.md` §8 明列）。
7. **分析产物的字节在登记期间被反复重写**（`analysis.json` 81,122 → 81,156 → **83,818**；`ANALYSIS.md` **0** → 21,752 → 21,767 → 23,290 → **23,455**）⇒ 引用时**只能引 10:03:33 之后稳定的那一版**；`created_utc` 现为 **`02:03:33Z`**（早前版本里 `ANALYSIS.md` 写 `01:55:53Z`、`analysis.json` 写 `01:56:45Z`，相差 52 s，现版已一致）。

## 6. 与相邻版本的关系

| 版本 | 与 v8.1.0 的关系 |
| --- | --- |
| [v8.0.0](../v8.0.0/SUMMARY_FOR_DECISION.md) | **直接前驱**（也是本轮的「第 0 档」）：它在 **300 决策 / 600 帧** 上得到 A0 **484/3,600 = 13.4444%** → A1/A2/A3 **各 3,600/3,600 = 100%**，40/60/80 px **三档全部顶到天花板**，判 **`S-ok`**，并留下处方「下一轮要么拉长段、要么禁掉决策 0 的那颗雷、要么把每局雷预算压低」。**本轮走的是第一条（拉长段）。** ⚠️ **引用 v8.0.0 的 `n` 时注意**：本轮预注册 §0 记的 A1/A2/A3 `n` = 3,357 / 3,420 / 3,446 是**冻结时的不完整快照**，v8.0.0 实测**各 3,600**。 |
| [v7.0.0](../v7.0.0/SUMMARY_FOR_DECISION.md) | **为什么这一路线会走到动作集**：控制频率 1/2/4 帧 ⇒ 2 帧最优、维度关闭，处方是「转动作集」；动作集是冻结协议常量 ⇒ 需要用户授权改协议（2026-11-22「协议 v2」）⇒ v8.0.0 先测放雷。 |
| [v7.1.0](../v7.1.0/SUMMARY_FOR_DECISION.md) | **同为 v4.5.0 `S` 三个冻结 ckpt 上的「改一样东西」**：v7.1.0 改**权重**（零样本 11.08% → 自适应后 1.00%，`S-degrade`，**拒绝做检验**）；v8.0.0/v8.1.0 改**动作**（13.4444% → 100% → 71.1111% / 13.3333% / 1.0000%）。**可以并排读「改权重 vs 改动作」，但不能把两者的率放进同一张「学习进展」表。** |
| [v4.5.0](../v4.5.0/SUMMARY_FOR_DECISION.md) | **三个冻结 ckpt 与 `13.4444%` 的来源**。本轮的复现控制臂（A3 × 300 决策）与 v8.0.0 **逐 episode 15 字段一致**，而 v8.0.0 的 A0 又与 v4.5.0 的冻结评测 **3,600/3,600 × 17 字段一致** ⇒ **A3×300 与本轮所有数字都能一路回溯到 v4.5.0 的同一批字节。** |
| [v5.1.0](../v5.1.0/SUMMARY_FOR_DECISION.md) / [v5.4.0](../v5.4.0/SUMMARY_FOR_DECISION.md) | **`S-floor` 的参照**：预注册引它们的「不放雷满段 0/1,200」作基线。⚠️ 那两个数是 **600 帧（300 决策）口径**，**不是** 900 决策口径（本轮 `RUN.md` §5 已指出）⇒ **900 档的同口径基线只能由本轮 A0 的存活曲线补**（分析轮已补：1,800 帧 = 0/1,200）。 |

## 7. 路径与复现约定

| 项 | 值 |
| --- | --- |
| 执行环境 | WSL，仓库 `/mnt/d/WorkShop/Touhou_RL`（Windows 侧 `D:\WorkShop\Touhou_RL`）；运行窗口 **08:39:34 → 09:34:02（+08:00）= 54 min 28 s**；22 条 `wall_s` 合计 **23,215.893 s**（10 个 cell **并发** + 并发 v8.2.0 回合把 `load average` 推到 ≈ 48） |
| 分析环境 | **Windows 侧**独立跑：`Python 3.12.10` / `numpy 2.5.4` / `Windows-11-10.0.26300-SP0`；`scipy 1.18.1` 只用于 `self_test` 对拍；**0 次引擎** |
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，sha256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`（= 冻结值） |
| 初始策略 | `.tmp_v450_cf/ckpt/seed{0,1,2}/round2.pt`（各 **34,387 B**；`4abee069…` / `a93746bb…` / `c69960fc…`）—— **冻结**，start 与 final **各 3/3** |
| 本轮产物 | **0 个 checkpoint**（纯评测、无训练）。工作目录 `.tmp_v810_bomb/`（`arms/` 13 份 jsonl + summary、`work/data/` **57 个私有 `th08.cfg`**、`regress/`（`before.jsonl` / `after.jsonl`，**未随证据发布**）、`logs/`、`py/`）与分析目录 `.tmp_analysis_80/` 都按 AGENTS.md §8.1 留在本机、不入 Git |
| runner | `tools/rl/v800_bomb.py`（**只改 4 处字面编辑**，默认 `--decisions 300` 不变）。本轮运行的字节 = [`v800_bomb.used.py`](evidence/BOMBFAR/v800_bomb.used.py)（77,827 B，`29244683…`）；补丁前快照 = [`v800_bomb.pre_v810_patch.py`](evidence/BOMBFAR/v800_bomb.pre_v810_patch.py)（77,712 B，`7fcf5f26…` = v8.0.0 的 runner）。**现盘文件已被并发 v8.2.0 回合改到 `f884a4be…`**，所以**要复跑本轮必须用 `v800_bomb.used.py` 那份字节** |
| 复跑入口（评测） | `bash Document/Update/Alpha/v8.1.0/evidence/BOMBFAR/scripts/exec.sh`（顺序见 `RUN.md` §3：`assert-ckpt start` → `d3 fidelity` → 3 个新路径探针 → A0×1200 → 三档 × 3 种子 → 控制臂 → `diagnose.py` → 音频/进程审计 → `assert-ckpt final`） |
| 复跑入口（统计） | `python .tmp_analysis_80/run_all.py`（**除 `created_utc` 外逐字节相同**）；`repro_v800.py` 重算 §3 的逐 episode 比对；`reconstruct.py` 重建本轮 runner 字节；`regress.sh` + `regress_check.py` 重做默认路径等价 |
| 单点复跑（最小切片） | `python tools/rl/v800_bomb.py --mode eval --bomb-rule r80 --policy-seed 0 --windows 12 --episodes 300 --decisions 600 --out <dir>`（**必须用 `v800_bomb.used.py` 的字节**；`--decisions` 是本轮新加的直通参数） |
| 只读约束 | 本轮只改 `tools/rl/v800_bomb.py`（4 处）+ 只写 `Document/Update/Alpha/v8.1.0/**` 与 `.tmp_v810_bomb/**`；分析层只写 `analysis.json` / `ANALYSIS.md` 与 `.tmp_analysis_80/**`；未触 `touhou_rl/**`、`third_party/**`、既有版本目录、**任何预注册**；**无 `git add`/`commit`**、**无 `pkill -f`**、未删除/移动/覆盖任何资产 |
| 音频（AGENTS.md §9） | 三段都做：①`exec.sh` 里 `SDL_AUDIODRIVER=dummy` + `unset DISPLAY WAYLAND_DISPLAY`，且 `v450_cf_label.prepare_env()` 再设一次；②每个私有 data 目录用 `v400_make_cfg.py --out` 生成 `th08.cfg` + `--inspect` 四字段全 0，否则 `build_private_data_dir` **拒绝启动引擎**；③事后审计 **57/57 muted、0 audible**；本轮引擎进程 **0** |
| 不要做的事 | 不要写「600 档复现了 100%」（是 **71.1111%**）；不要说「管线坏了 / v8.0.0 不可复现」（控制臂 3,600/3,600、15 字段一致）；不要把 **71.1111%** 与 **13.4444%** 当同口径（**同口径是 0.1667%**）；不要说「A3 的雷白放了」（**没有**同段同种子的有雷/无雷配对臂）；不要做跨档均值比较（删失上限 1,200/1,800/2,400）；不要把 Fisher/CP95 说成「预注册要求的检验」（预注册只要计数比较）；不要把 15,600 条当独立样本（**12 格**，逐种子极差 **73.17 pp**）；不要把 `ckpt_assertions_final.json` 的 `f884a4be…` 当成本轮运行的 runner 字节（那是 `29244683…`）；不要把 A0 的 1,200 条与 A3 的 3,600 条摆成严格对照（分母不同、A0 只有 1 个种子）；不要引用 `ANALYSIS.md` 的 **0 字节**中间态 |
