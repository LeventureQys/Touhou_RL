# 索引 — v8.0.0 先用脚本规则测「放雷」值多少（BOMB-001，接手从这里开始）

> v8.0.0 **只做一件事**：把「放雷」从一个**被禁用的动作**（18 个候选 = 9 方向 × 2 focus，射击恒开、放雷恒关）变成一个**外部脚本规则**，然后用同一批冻结 checkpoint、同一预算测它值多少帧。
> 单因子 = **放雷规则**：`A0` 从不放雷 / `A1` 最近子弹 **< 40 px** / `A2` **< 60 px** / `A3` **< 80 px**（且 `bombs_remaining > 0` ⇒ 该决策**首帧**发 `BTN_BOMB`）。策略固定为 v4.5.0 `S` 的三个冻结 `round2.pt`（stochastic），预算固定 **300 决策 / 600 帧**，起点 PARK 到 frame **8,663**，4 dev 窗口 × 3 种子 × 每格 300 条 = **14,400 episode**。
> **结论：A0 484/3,600 = 13.4444%（CP95 [12.3462%, 14.6024%]，存活帧均值 216.9714）→ A1/A2/A3 各 3,600/3,600 = 100.0000%（CP95 [99.8976%, 100.0000%]，600.00）；三个阈值一档不差。**
> **判定（分析轮已跑完）：P1 成立（Fisher + Holm）、P2 成立（置换 20,000）⇒ `S-ok` 触发**；`S-fork` / `S-nohelp` / `S-harm` 未触发。
> **四条不可省的边界**：放雷是**脚本规则、不是策略**；三档都顶到天花板 ⇒ **阈值不可分辨（饱和），不是「阈值无效」**（臂间 Fisher p 正好 1.0 = **没有信息**）；**P2 的 p 停在下限 `1/20,001`**，只能当方向与量级、不能当精确显著性；**+383.03 帧是臂间差**，逐 episode 反事实无法判定。

## 1. 你是谁，先读哪份

| 目的 | 读这份 | 规模 |
| --- | --- | --- |
| **要一个结论** | [`SUMMARY_FOR_DECISION.md`](SUMMARY_FOR_DECISION.md) | 六节：§0 一句话（七件事：决策 0 / 3 颗雷 / 2 帧延迟 / 天花板 / **P1 成立** / **P2 成立** / **`S-ok`**）→ §1 改了什么（含分析层）→ §2 判据表（P1 Fisher+Holm / P2 置换 / P3 / 停止条件）→ §3 有效性诊断（三重保真 + 天花板 + all-bullets vs lethal-only）→ §4 并排 → §5 能/不能声称（11 条）→ §6 合规、七个开放项与**九条文档间冲突** |
| **要核统计结论** | [`evidence/BOMB/ANALYSIS.md`](evidence/BOMB/ANALYSIS.md) + [`evidence/BOMB/analysis.json`](evidence/BOMB/analysis.json) | ANALYSIS.md **19,383 字节**（sha256 `347f995e…`，238 行 / 11 节：§0 结论 → §1 主表 + **§1.0 行级自检 14 项 × 4 臂** + §1.1 P1 + §1.2 P2 → §2 判定 → §3 P3 → §4 管线保真 → §5 决策 0 → §6 天花板 → §7 停止条件 → §8 与 RUN.md 对数 → §9 不能声称 → §10 文件）；analysis.json **81,317 字节**（sha256 `f14ade73…`；schema `v800-bomb-analysis/1`、`created_utc = 2026-11-22T02:03:28Z`、`scope.engine_started = false`） |
| **要核某个数字** | [`evidence/BOMB/diagnostics.json`](evidence/BOMB/diagnostics.json) | 148,597 字节，sha256 `e6c40297…`；`definitions` / `arms`（A0–A3）/ `pooled` / `a0_vs_v450` / `no_statistics`；`status = complete`、`error = null` |
| **要核放雷记账**（本轮核心） | [`evidence/BOMB/bomb_audit.json`](evidence/BOMB/bomb_audit.json) | 378,209 字节，sha256 `24af8d45…`：`rule_table`、`distance_definition`、`smoke`（逐决策 trace：`emit_nearest_px` / `emit_nearest_lethal_px` / `emit_remaining_after_{first,last}_frame` / `emit_latency_frames`）、`arms.*.first_emit_decision_stats` / `latency_frames` / `invariants` |
| **要复跑实验** | [`evidence/BOMB/RUN.md`](evidence/BOMB/RUN.md) + [`commands.tsv`](evidence/BOMB/commands.tsv) + [`scripts/exec.sh`](evidence/BOMB/scripts/exec.sh) | RUN.md 21,059 字节（sha256 `01deca41…`，337 行）：§1 设计与等价性证明、§2 门表、§3 冒烟、§4 全网格、§5 审计、§6 **七条偏差**、§7 不能声称 + 开放项、§8 文件与环境；`commands.tsv` 11,372 B（**34 条 argv**：31 条 `exit = 0`，3 条失败后有重跑） |
| **要复跑统计** | `python .tmp_analysis_80/run_all.py` | 除 `created_utc` 外，`analysis.json` 与 `ANALYSIS.md` **逐字节可重生成**；`stats_core.self_test()` 与 **scipy 1.18.1** 对拍（Fisher 最大相对误差 **4.32e-16**、Clopper-Pearson 最大绝对误差 **2.11e-14**） |
| **要知道判据怎么定的** | [`PREREGISTRATION_BOMB_001.md`](PREREGISTRATION_BOMB_001.md) | 80 行，4,456 字节（FROZEN 2026-11-22，sha256 `8629d73c…`）：P1（三个阈值各对 A0 的单侧 Fisher + **Holm**）/ P2（**置换 20,000**）/ P3（描述）；`S-fork` / `S-nohelp` / **`S-ok`** / `S-harm` |
| **要查 A0 是不是「另一份实现」** | `ANALYSIS.md` §4 + [`equiv_check.json`](evidence/BOMB/equiv_check.json) | 分析轮**自己重算**（不引 `equiv_check` / `diagnostics.a0_vs_v450`）：3,600/3,600 行、**17 个共有字段**、`rows_with_differences = 0`，两边的 full-600 逐种子都是 **220 / 143 / 121**；`equiv_check` 另证「runner 复制对 `v432_fork_arm.run_episode` 等价」**4/4**（含动作序列 126/300/130/135） |
| **要查机制** | [`analysis.json`](evidence/BOMB/analysis.json) → `caliber_checks` + `p3` | `first_bomb_decision` = 0（min=中位=max，100%）；`emit_nearest_px_first` **均值 25.2279 / 中位 23.1229 / min 18.2876 / max 36.3780 px** ⇒ 决策 0 的按键**在三档阈值之内**；`last_bomb_decision` 中位 A1 **6.0** / A2 **133.0** / A3 **132.0**；最后一次**被接受的按雷**中位 A1 **0.0** / A2 **128.0** / A3 **126.0** |
| **要查保真门（`S-fork`）** | [`evidence/BOMB/fidelity_d3.json`](evidence/BOMB/fidelity_d3.json) | 26,500 字节：`all_pass = true`、`deviations = []`、5/5（`role=shadow` / `threads=1` / 前缀 **2,165** 0 mismatch、末帧 8,663 / 18 个常数 **18/18** / `PARK frame=8663`）。⚠️ 分析轮**离线不能重跑**这条门（`recorded_gate_not_reverifiable_offline = true`） |
| **要查冻结对象** | [`evidence/BOMB/ckpt_assertions.json`](evidence/BOMB/ckpt_assertions.json) | 2,675 字节、`phase = "final"`、`all_pass = true`：3 个 `round2.pt`（各 **34,387 B**）3/3、5 个受保护源、引擎 `691bfd77…`、**runner 自身** `7fcf5f26…` |
| **要查评测原始行** | `eval_A{0,1,2,3}_seed{0,1,2}.jsonl` + 12 份 `*.summary.json` | 12 个 jsonl（各 1,200 行 × 4 臂 = **14,400** 条），每条带 `alive_frames` / `first_death_offset` / `survived` / `return_decisions` / 该 episode 的放雷审计 / `ckpt` sha256 / generator seed |
| **要查音频/进程合规** | [`audio_audit.json`](evidence/BOMB/audio_audit.json) + [`process_residue.json`](evidence/BOMB/process_residue.json) | 5,986 字节：`scope = .tmp_v800_bomb/work/data`，**52 cfg / 52 decoded / 0 errors / 52 muted / 0 audible**、`all_muted = true`；486 字节：`matching = 0`、`pgrep_returncode = 1`、`all_pass = true` |
| **要查发布清单** | [`evidence/BOMB/manifest.json`](evidence/BOMB/manifest.json) | 8,452 字节：`file_count = 62`、`total_bytes = 24,380,174`、`excludes = "manifest.json"`。⚠️ **清单早于分析产物 83 分钟**（08:40:10 vs 10:03:28）⇒ 目录现在 **65 个文件 / 24,489,326 B**，多出来的 2 个就是分析产物 |
| **要知道这份索引之外还有什么** | [v8.1.0](../v8.1.0/SUMMARY_FOR_DECISION.md) · [v7.1.0](../v7.1.0/SUMMARY_FOR_DECISION.md) · [v7.0.0](../v7.0.0/SUMMARY_FOR_DECISION.md) · [v4.5.0](../v4.5.0/SUMMARY_FOR_DECISION.md) | **直接后继**（预算推到 600/900/1,200 决策 ⇒ **71.1111% / 13.3333% / 1.0000%**；A3×300 控制臂 3,600/3,600 逐 episode 一致）/ 在线自适应 `S-degrade` / 控制频率 ⇒ 2 帧最优、维度关闭 / 本轮三个冻结 ckpt 与 `13.4444%` 的来源 |

> **分析产物是「后补」的，而且登记期间还在被重写（必须知道）**：`RUN.md` 与 `manifest.json` 的 mtime 是 **08:40:07 / 08:40:10**，`analysis.json` / `ANALYSIS.md` 的第一版是 **09:49:35**、本索引钉的最后一版是 **10:03:28**（登记期间实测被重写 ≥6 次：`analysis.json` 76,294 → 78,196 → 80,736 → 80,731 → 81,197 → **81,317** B；`ANALYSIS.md` 17,932 → **0（重写中）** → 18,925 → 19,027 → 18,990 → 19,281 → **19,383** B）。因此 ①`manifest.json` 的 62 条**不含**它们；②本索引钉的是 **10:03:28 之后连续 ≥80 s 字节不变**的最后一版；③**0 字节的中间态不可引用**，早期版本的 sha256 也不保留。
>
> **本轮有统计推断，而且只在本索引的 §4 之下**：`RUN.md` 自己声明「不含统计推断」——它确实没有；`analysis.json` / `ANALYSIS.md` 补的就是那一步（`scope.engine_started = false`，**0 次引擎**）。

## 2. 三十秒版本

- **只打开了一个被禁用的动作**：观测（106 维，早已含 `bombs_remaining/8`）、模型、**18 候选**、死亡规则、起点 8,663、预算 **300 决策 / 600 帧**、RNG `seed*1000+7` **一行未动**；唯一变量是「该决策首帧要不要发 `BTN_BOMB`」。
- **规则三段**：最近任意 `HAZARDS` 子弹的欧氏距离 `< T`（40 / 60 / 80 px）**且** `bombs_remaining > 0` ⇒ 首帧 `cand_to_mask(cand, bomb=1)`，其余帧 `bomb=0`；A0 从不置位。
- **结果**：A0 **484/3,600 = 13.4444%**（存活帧均值 **216.9714** / 中位 **107.0**）→ A1/A2/A3 **各 3,600/3,600 = 100.0000%**（均值/中位/q1/q3/min/max **全为 600.0**）⇒ **+3,116 条 / +86.5556 pp / +383.03 帧**。
- **P1 成立**：三臂零失败，单侧 Fisher 的精确 `log10 p = −1521.5834` ⇒ `p_sci = 2.609474e-1522`（double 下溢，不是 0），**Holm 校正后 7.828422e-1522**，CP95 **区间不重叠**（A0 `[12.3462%, 14.6024%]` vs 三臂 `[99.8976%, 100.0000%]`）。
- **P2 成立**：观测差 **+383.0286 帧**、置换 **20,000 次**、`count_ge = 0` ⇒ **p = 4.999750e-05**，Holm 后 **1.499925e-4**。**这个 p 正好是 20,000 次置换的分辨率下限 `1/20,001`**（`resolution_note`：是「要求的检验的性质，不是测到的效应量」）。
- **判定 `S-ok` 触发**（P1 与 P2 同时成立）⇒ 预注册的处置是下一轮**为 36 动作（18 × bomb∈{0,1}）重采标签并训练**。`S-fork` / `S-nohelp` / `S-harm` 未触发（反向 Fisher 三条 `reversed_p = 1.0`）。
- **不是「抓时机」，是「起手就放」**：决策 0 的最近子弹距离在 3,600 条上 **max 只有 36.3780 px**（均值 25.2279 / 中位 23.1229 / min 18.2876）⇒ **本来就在 40/60/80 三档之内**；10,800 条放雷 episode 的第一颗雷**全部**在决策 0。
- **雷是有限资源**：每局 **3** 颗（14,400 行全为 3），`bombs_used` max = **3**；一颗雷就够撑满 **≥ 66%** 的 episode（A1 用 1 颗的有 2,384 条；A1 最后一次**被接受的按雷**中位在第 **0** 个决策）。
- **按键后第 2 帧才扣雷**：`latency_frames = {"2": 4880 / 5734 / 6288}` —— 单值 **2**，覆盖全部 **16,902** 次被接受的按键；约 **86–89%** 的按键被无敌帧忽略（34,891 → 4,880 等）。
- **阈值不可分辨 = 饱和**：三臂 `alive_frames` 分布**完全相同**，48 个单元 **48/48 都是 300/300**，**臂间 Fisher p = 1.000000e+0（表完全相同 ⇒ 没有信息，不是没有差异）**；端点不退化（A0 有非平凡失败率）、**资源也没用完**（每局只用 1.36 / 1.59 / 1.75 颗）⇒ **端点先饱和、资源还没耗尽**。
- **A0 三重保真**：①分析轮独立重算 3,600/3,600 行 × **17 个共有字段**、0 差异；②与冻结 `run_episode` 循环等价 **4/4**（含动作序列）；③两处字段差被点名（`alive_frames` 是新增、`baseline.bombs` 是新增）。
- **不能做**：不能写「阈值无关 / 阈值等效」；不能说「策略学会了放雷」；不能把放雷臂与历史 18 动作结果池化；不能说「一颗雷值 383 帧」（臂间差 ≠ 逐 episode 增量）；不能把 `p = 4.99975e-05` 当精确显著性/效应量；不能把 p 当「已按 12 格聚类」；不能外推 300 决策 / 600 帧；不能把 14,400 条当独立样本。

## 3. 证据一览（`evidence/BOMB/`，**65 个文件** = 清单 62 ＋ `manifest.json` 自身 ＋ 2 个分析产物）

| 组 | 文件 | 这一组在做什么 |
| --- | --- | --- |
| 判据 | （上一级）[`PREREGISTRATION_BOMB_001.md`](PREREGISTRATION_BOMB_001.md) | P1（三阈值单侧 Fisher + **Holm**）/ P2（**置换 20,000**）/ P3（描述）；`S-fork` / `S-nohelp` / **`S-ok`** / `S-harm` |
| **分析** | [`analysis.json`](evidence/BOMB/analysis.json)（80,731 B）+ [`ANALYSIS.md`](evidence/BOMB/ANALYSIS.md)（18,990 B）+ `.tmp_analysis_80/{analyze_v800.py,run_all.py,stats_core.py}`（**未发布**） | **P1/P2 的唯一判定来源**；`scope.engine_started = false`；`fidelity_a0_vs_v450` 自己重算；`caliber_checks`（决策 0 / 天花板）；`stop_conditions`；`self_test`（scipy 对拍）；`run_md_crosscheck`（9 条全一致） |
| 网格数据 | `eval_A{0,1,2,3}_seed{0,1,2}.jsonl`（**12 个，各 1,200 行**）+ 12 份 `*.summary.json` | **14,400 条** episode 的唯一数据源；每条含存活字段 + 该 episode 的完整放雷审计 + ckpt sha256 + generator seed |
| 放雷审计 | [`bomb_audit.json`](evidence/BOMB/bomb_audit.json) | `smoke`（逐决策 trace）+ `arms.A{0..3}`（指令/扣雷/忽略/分布/`first_emit_decision_stats`/`latency_frames`/6+3 条不变量） |
| 诊断 | [`diagnostics.json`](evidence/BOMB/diagnostics.json) | `definitions` / `arms` / `pooled` / `a0_vs_v450` / `no_statistics`；**无检验**（检验在分析层） |
| 对照 | [`equiv_check.json`](evidence/BOMB/equiv_check.json) + `ANALYSIS.md` §4 | ①循环等价 4/4（代码）；②A0 与 v4.5.0 冻结评测 3,600/3,600 × 17 字段（数值）。**两条是不同命题，不能互相顶替** |
| 门 1：保真 | [`fidelity_d3.json`](evidence/BOMB/fidelity_d3.json) | 5/5 断言、前缀 2,165、末帧 8,663、18 个常数候选、PARK、`deviations = []`；**离线不可重跑**（已如实记录） |
| 门 2：冻结 | [`ckpt_assertions.json`](evidence/BOMB/ckpt_assertions.json) | 3 个 ckpt 3/3、5 个受保护源、引擎哈希、**runner 自身** sha256（`phase = "final"`） |
| 执行 | [`RUN.md`](evidence/BOMB/RUN.md)、[`commands.tsv`](evidence/BOMB/commands.tsv)、`logs/*.out`（**22 个**）、`scripts/`（**7 个**：`exec.sh` / `status.sh` / `finalize.sh` / `tables.py` / `numbers.py` / `inspect_smoke.py` / `show.py`） | 34 条 argv（31 exit 0；`smoke-a3` 的 1→7→0 与 `equiv-a0` 的 8→0）；每个进程的完整 stdout；`RUN.md` §6 **七条偏差**全列 |
| 音频/进程合规 | [`audio_audit.json`](evidence/BOMB/audio_audit.json)、[`process_residue.json`](evidence/BOMB/process_residue.json) | **52 个** `th08.cfg` 全 muted、**audible 0**、`shared_audit_untouched`；本轮引擎进程 **0** |
| 发布 | [`manifest.json`](evidence/BOMB/manifest.json) | `file_count = 62` / `total_bytes = 24,380,174` / `excludes = "manifest.json"`；登记时对 **9 个关键文件**重算，**9/9 相符**。**不含分析产物**（清单发布在先） |

## 4. 判据与停止条件（预注册 §2 / §2.1，**已由分析轮跑完**）

| 条目 | 条件 | 本轮结果 |
| --- | --- | --- |
| **P1（主）** | 某个放雷臂的活满 600 帧条数 > A0；三个阈值各做**单侧 Fisher**，**Holm 校正** | ✅ **成立**：`log10 p = −1521.5834`、`p_sci = 2.609474e-1522`（double 下溢）、**Holm 后 7.828422e-1522**、三条 `holm_reject_0.05 = true`、CP95 **不重叠** |
| **P2** | 某个放雷臂的存活帧均值 > A0；**置换 20,000 次（种子 0）** | ✅ **成立**：差 **+383.0286 帧**、`count_ge = 0 / 20000`、**p = 4.999750e-05**（= 分辨率下限 `1/20,001`）、**Holm 后 1.499925e-4** |
| **P3（描述）** | 雷的消耗量、放雷后的存活增量、`bombs_remaining` 掉到 0 的比例 | **只报**：扣雷 4,880 / 5,734 / 6,288；耗尽 **1.7778% / 5.4444% / 9.3056%**；`bombs_used` 均值 1.3556 / 1.5928 / 1.7467；**增量是臂间差，不是逐 episode** |
| `S-fork` | 保真门失败 | **未触发**（D3 5/5 在案 + A0 逐条复现；引擎侧一半**离线不可复算**，已记录） |
| `S-nohelp` | P1 与 P2 都不成立 | **未触发**（两条都成立） |
| **`S-ok`** | P1 **或** P2 成立 | ✅ **触发** ⇒ 下一轮为 **36 动作（18 × bomb∈{0,1}）** 重采标签并训练，再与 18 动作策略同协议对比 |
| `S-harm` | 某个放雷臂显著**更差**（反向 Fisher p<0.05） | **未触发**（反向 p 三条都是 `1.000000e+0`、`significant_at_0.05 = false`） |
| **实际结果** | — | **`S-ok`**（`verdict.stop_condition = "S-ok"`、`triggered = ["S-ok"]`） |

> **必须一起读的一句**：`S-ok` 的处置是「**为 36 动作重采标签并训练**」——那是**一条新的、更贵的路线**，不是本轮的结论。本轮的结论只有两条：①**一个脚本规则就把 13.4444% 顶到 100%**（P1/P2 都成立）；②**这个设计在 600 帧上分不出 40/60/80 px**（饱和，不是无效）。

## 5. 开放项

1. **`S-ok` 的处方还没做**：预注册写的是「下一轮为 **36 动作（18 × bomb∈{0,1}）** 重采标签并训练，再与 18 动作策略同协议对比」。本轮**只测了脚本规则**，一条处方都没跑——这是下一轮的预注册内容。
2. **天花板必须先打破**（`RUN.md` §7.2 / `ANALYSIS.md` §6 的三条路，**本轮一条都没试**）：①把段拉长（→ **v8.1.0 已做** 600/900/1,200 决策）；②禁掉**决策 0** 的那颗雷；③把每局雷预算压到规则要用的量以下。**没有这三者中的任何一个，40/60/80 px 永远是 100% vs 100% vs 100%。**
3. **lethal-only + 不重复放**的规则没试：全子弹口径 35,515–58,968 条决策「低于阈值」，只看致命弹只剩 **4,755–8,017**（7.4–7.5×）。预注册 §2.1 的 `S-harm` 处方「只在无路可走时放」正是这个方向。
4. **逐 episode 的反事实无法判定**：+383.03 帧是**臂间差**；要拆「这一条因为放雷多活了多少帧」，必须新增「同一 episode 有雷/无雷各跑一遍」的配对臂。
5. **`smoke` 块是重跑不是预跑**：`bomb_audit.json::smoke` 来自 08:37:38 的 `smoke-a3-rerun`（`RUN.md` §6 偏差 2：合并 bug 丢过预跑块、已修）；数字与预跑一致（49 emits / 7 honoured / 7 bombs / first emit decision 0），但它不是那条预跑。
6. **P1/P2 的 p 没有按 12 格聚类**：分析按**行**算（每臂 3,600 行），可辩护单位是 **12 格**。若要聚类校正，需要新的分析轮声明并重算。
7. **分析产物的字节在登记期间被反复重写**（`analysis.json` 76,294 → 78,196 → 80,736 → 80,731 → 81,197 → **81,317**；`ANALYSIS.md` 17,932 → **0** → 18,925 → 19,027 → 18,990 → 19,281 → **19,383**）⇒ 引用时**只能引 10:03:28 之后稳定的那一版**，`created_utc` 是 **`02:03:28Z`**（不是 09:49 那版的 `01:49:35Z`，也不是 09:51 那版的 `01:51:57Z`）。**关键统计量在各版之间逐字相同**。

## 6. 与相邻版本的关系

| 版本 | 与 v8.0.0 的关系 |
| --- | --- |
| [v8.1.0](../v8.1.0/SUMMARY_FOR_DECISION.md) | **直接后继，也是本轮的「段长」续篇**：把预算从 **300 决策** 推到 **600 / 900 / 1,200 决策**（1,200 / 1,800 / 2,400 帧），固定 A3（80 px），并补 **A0 × 1,200 决策** 无雷基线。结果 **71.1111% → 13.3333% → 1.0000%**，**不是 100%**；且它加跑的 **A3 × 300 决策控制臂 = 3,600/3,600、与本轮逐 episode 15 字段全同** ⇒ 本轮的 100% **可复现**，跌幅来自段变长。⚠️ 引用本轮的 `n`：v8.1.0 预注册 §0 记的 A1/A2/A3 `n` = 3,357 / 3,420 / 3,446 是 **in-flight 快照**，**本轮实测各 3,600**（该轮 `RUN.md` §7 偏差 7 与 `ANALYSIS.md` §8 都已说明）。 |
| [v7.0.0](../v7.0.0/SUMMARY_FOR_DECISION.md) | **为什么换到动作集**：控制频率 1/2/4 帧在 600 帧预算下 2 帧最优 ⇒ 该维度**关闭**，处方是「转**动作集**」；而动作集 `N_MOVE_FOCUS = 18` 是冻结协议常量 ⇒ 改动作集等于改协议/ABI ⇒ 用户在 2026-11-22 明确授权「**协议 v2**」，本轮据此先测**放雷**（不加候选、只加外部规则）。 |
| [v7.1.0](../v7.1.0/SUMMARY_FOR_DECISION.md) | **同为「改一样东西」的另一型，但方向相反**：v7.1.0 是**在线改权重**（零样本 11.08% → 自适应后 1.00%，`S-degrade`，且**拒绝做检验**）；本轮是**不动权重、只加一个外部按键规则**（13.4444% → 100%，**做了 Fisher + Holm 与 20,000 次置换，判 `S-ok`**）。两轮都以 **v4.5.0 `S` 三个冻结 ckpt** 为起点 ⇒ **可以并排读「改权重 vs 改动作」**，但**不能**把两者的率放进同一张「学习进展」表；也不能把「本轮做了检验」当成 v7.1.0 的补。 |
| [v4.5.0](../v4.5.0/SUMMARY_FOR_DECISION.md) | **本轮三个冻结 ckpt 与 `13.4444%` 的来源**，而且本轮 A0 **逐 episode 复现**了它（3,600/3,600、**17 个字段**、0 差异，分析轮独立重算）——这是 v7.1.0 做不到的（那轮 100 条/格、流也不同）。**所以 A0 与 v4.5.0 是同一个数，不是两个接近的数。** |
| [v5.7.0](../v5.7.0/SUMMARY_FOR_DECISION.md) | **不要混谈**：它用 **1,200 帧预算**（= 600 决策）；本轮 **600 帧**（300 决策）。本轮**不能**外推到 600/900 决策（预注册 §5 明令）——那条外推是 v8.1.0 做的，而且**掉到了 71.1111%**。 |

## 7. 路径与复现约定

| 项 | 值 |
| --- | --- |
| 执行环境 | WSL，仓库 `/mnt/d/WorkShop/Touhou_RL`（Windows 侧 `D:\WorkShop\Touhou_RL`），`/root/v400-venv/bin/python`；`RUN.md` §2 记录**开工前工作树已经是脏的**（数百条早前回合的修改，均非本轮产生），本轮新增路径只有 `tools/rl/v800_bomb.py` 与 `Document/Update/Alpha/v8.0.0/**` |
| 分析环境 | **Windows 侧**独立跑：`Python 3.12.10` / `numpy 2.5.4` / `Windows-11-10.0.26300-SP0`（`analysis.json::environment`）；`scipy 1.18.1` 只用于 `self_test` 对拍；**0 次引擎**。**登记期间分析轮仍在重写**，本索引钉的是 10:03:28 版 |
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，sha256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` |
| 初始策略 | `.tmp_v450_cf/ckpt/seed{0,1,2}/round2.pt`（各 **34,387 B**；`4abee069…` / `a93746bb…` / `c69960fc…`）—— **冻结**，`assert-ckpt` 3/3 相符 |
| 本轮产物 | **0 个 checkpoint**（纯评测、无训练）。工作目录 `.tmp_v800_bomb/`（`arms/` 12 份 jsonl + summary、`work/data/` **52 个私有 `th08.cfg`**、`equiv_check.json`、`logs/`、`py/`）与分析目录 `.tmp_analysis_80/`（`analyze_v800.py` / `run_all.py` / `stats_core.py`）都按 AGENTS.md §8.1 留在本机、不入 Git |
| runner | `tools/rl/v800_bomb.py`（新增，sha256 `7fcf5f265cfd1460fcf009fca2ad596152687dafb82ab83913211e92a49434e0`，记在 `ckpt_assertions.json::v800_bomb_py`；v8.1.0 侧留有同字节快照 `v800_bomb.pre_v810_patch.py`，77,712 B） |
| 复跑入口（评测） | `bash Document/Update/Alpha/v8.0.0/evidence/BOMB/scripts/exec.sh`（`RUN.md` §2 的步骤序列：`assert-ckpt --phase start` → `d3 fidelity` → `smoke` → `equiv` → 12 个 `--mode eval` → `audit` / `audio-audit` / `process-audit` → `diagnostics` → `assert-ckpt --phase final` → `manifest`） |
| 复跑入口（统计） | `python .tmp_analysis_80/run_all.py`（**除 `created_utc` 外逐字节相同**）；单点自检 `stats_core.self_test()` |
| 单点复跑（最小切片） | `python tools/rl/v800_bomb.py --mode eval --bomb-rule r80 --policy-seed 0 --windows 12 --episodes 300 --out <dir>`（默认 `--decisions 300`；v8.1.0 之后该参数可改，但**改它就不再是本轮的条件**） |
| 只读约束 | 本轮只写 `tools/rl/v800_bomb.py`、`Document/Update/Alpha/v8.0.0/**`、`.tmp_v800_bomb/**`；分析层只写 `analysis.json` / `ANALYSIS.md` 与 `.tmp_analysis_80/**`；未触 `touhou_rl/**`、`third_party/**`、既有版本目录、预注册；**无 `git add`/`commit`**、**无 `pkill -f`**、未删除/移动/覆盖任何资产 |
| 音频（AGENTS.md §9） | 每次启动前由 `build_private_data_dir` 用 `v400_make_cfg.py --out` 重新生成静音 `th08.cfg` 并 `--inspect`，四字段不全 0 就**拒绝启动**（`cfg.muted_four_zero = true` 记在每个 summary 里）+ `SDL_AUDIODRIVER=dummy` + `DISPLAY`/`WAYLAND_DISPLAY` 未设置；审计 **52/52 muted、0 audible**；本轮引擎进程 **0** |
| 不要做的事 | 不要写「阈值无关 / 阈值等效 / 阈值无效」（是**饱和**；臂间 p = 1.0 是**没有信息**）；不要说「策略学会了放雷」；不要把放雷臂与历史 18 动作结果池化；不要说「一颗雷值 383 帧」（臂间差 ≠ 逐 episode 增量）；不要把 `p = 4.99975e-05` 当精确显著性/效应量（是分辨率下限）；不要把 p 说成「已按 12 格聚类」；不要外推 300 决策 / 600 帧；不要把 14,400 条当独立样本（**12 格**）；不要引用 `used_at_most_eight_per_episode` 的 8（真实上限 **3**）；不要把 `manifest.json` 的 62 条当成目录全部（**65**）；不要引用 `ANALYSIS.md` 那份 **0 字节**中间态或 `01:49:35Z` 的旧 `created_utc` |
