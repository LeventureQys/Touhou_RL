# v8.4.0 索引 — BOMBRULE2-001（放雷规则 R0–R5）：**已发布 / 已重算**

**一句话**：单因子 = 放雷规则；600 决策（20 s）下 **R5 77.11 %（2,776/3,600，p = 3.53e−09）与 R3 75.72 %（2,726/3,600，p = 5.32e−06）显著优于 R0 的 71.11 %**，**R1 67.44 % / R4 25.81 % / R2 17.22 % 显著更差**；**R5 冲上限（seed2 仅 34.0 %），R3 压方差（最差种子 68.4 %）**；900 决策档的 R5 **17.61 %**（634/3,600，逐种子 18.75 / 34.08 / 0.00 %）只作描述性。

- 数据冻结：**2026-11-22T11:15:25+08:00**（`diagnostics.json::frozen_at`）
- 引擎：**发布过程没有启动任何引擎**；数据来自被中止的执行代理已落盘的 21 份 JSONL（`engine_started = false`）
- 统计：**做了**——单侧 Fisher + Holm(5)，p 值用**精确有理数**计算，`scipy.stats.fisher_exact` 交叉校验（相对差 ≤ 1.7e−15；R2/R4 在双精度下溢，精确值 3.23e−490 / 1.41e−335）
- 证据：**21/21 文件完整**、**21/21 副本 sha256 == 执行器 `out_sha256`**、**334,517 次按雷 0 违例**、**93/93 cfg muted**

---

## 1. 怎么读（三条路径）

| 你有多少时间 | 读什么 |
| --- | --- |
| **3 分钟** | 本文件 + [`SUMMARY_FOR_DECISION.md`](SUMMARY_FOR_DECISION.md) 的 **§0** 与 **§5（能声称 / 不能声称）** |
| **10 分钟** | [`SUMMARY_FOR_DECISION.md`](SUMMARY_FOR_DECISION.md) 全篇（§0 结论 → §2 判据表 → §3 有效性诊断 → §6 合规与开放项） |
| **全过程** | [`evidence/BOMBRULE2/RUN.md`](evidence/BOMBRULE2/RUN.md)（中止代理留下的现场、d900 时间线、逐格完整性、Fisher+Holm 全表、规则不变量、偏差清单、发布清单） |

判据原文：[`PREREGISTRATION_BOMBRULE2_001.md`](PREREGISTRATION_BOMBRULE2_001.md)（FROZEN，79 行，3,703 字节，sha256 `0576ddc03440f8352958829380878051b57fb29eff2750c0ec1833a8a13bc30b`）。

---

## 2. 关键数字速查（600 决策，n = 3,600/臂）

| 臂 | 规则 | 活满 | 活满率 | Δ vs R0 | 单侧 p（>R0） | Holm(5) 单侧 | 每局用雷 | 末次放雷（中位） |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **R0** | `r80`（对照） | 2,560 | 71.1111 % | — | — | — | 2.9833 | 433 |
| R1 | `conserve` | 2,428 | 67.4444 % | −3.6667 pp | 0.999661 | 1（n.s.） | 2.9917 | 420.5 |
| R2 | `delay200` | 620 | 17.2222 % | −53.8889 pp | 1 | 1（n.s.） | 0.4853 | 441 |
| R3 | `double60` | 2,726 | **75.7222 %** | **+4.6111 pp** | **5.32236e−06** | **2.12894e−05** | 2.5911 | **458** |
| R4 | `wall` | 929 | 25.8056 % | −45.3055 pp | 1 | 1（n.s.） | 1.0067 | 418 |
| **R5** | `staged` | **2,776** | **77.1111 %** | **+6.0000 pp** | **3.53114e−09** | **1.76557e−08** | 2.9392 | **458** |

**逐种子（每格 1,200 条）**：

| 臂 | seed 0 | seed 1 | seed 2 |
| --- | ---: | ---: | ---: |
| R0 | 1,157（96.42 %） | 1,124（93.67 %） | 279（23.25 %） |
| R1 | 1,124（93.67 %） | 1,071（89.25 %） | 233（19.42 %） |
| R2 | 277（23.08 %） | 193（16.08 %） | 150（12.50 %） |
| R3 | 985（82.08 %） | 920（76.67 %） | **821（68.42 %）** |
| R4 | 516（43.00 %） | 250（20.83 %） | 163（13.58 %） |
| **R5** | **1,190（99.17 %）** | **1,178（98.17 %）** | 408（34.00 %） |

**决策侧的两个事实**：① 「显著更差」用的是**双侧族**（补充报告）——预注册的单侧（greater）族在构造上无法证明「更差」；② R0/R1/R5 都在**决策 0** 放第一颗雷（停车状态最近子弹 18.29–36.38 px），差别在**后两颗的时点**。

**900 决策（R5，完整格，描述性）**：**634/3,600 = 17.6111 %**（225 / 409 / 0），R0 记录 13.3333 %（480/3,600 = 176/304/0）。**seed2 = 0.00 %**。

---

## 3. 目录清单

```
Document/Update/Alpha/v8.4.0/
├─ PREREGISTRATION_BOMBRULE2_001.md        3,703   预注册（FROZEN，未改动）
├─ SUMMARY_FOR_DECISION.md                26,821   决策摘要（§0–§6）
├─ README_INDEX.md                         本文件
└─ evidence/BOMBRULE2/                 116 个文件 / 69,433,015 字节（清单覆盖其中 115 个，`manifest.json` 自身除外）
   ├─ eval_R{0..5}_d600_seed{0,1,2}.jsonl (18)   d600 原始数据（原名，byte-identical；各 1,200 行）
   ├─ eval_R5_d900_seed{0,1,2}.jsonl      (3)    d900 原始数据（原名；各 1,200 行，完整格）
   ├─ diagnostics.json                 175,407   本包重算的全部统计（含 Fisher+Holm、规则不变量、d900 快照史）
   ├─ manifest.json                     23,003   115 个文件的 sha256 / 字节 / 行数 + self_check + aggregate_sha256
   ├─ RUN.md                            32,989   执行记录（0–11 节）
   ├─ refs/                              (37)   ckpt 断言、D3 保真、A/B、rulecheck、补丁 diff、27 份 commands TSV、
   │                                            CROSSCHECK.json（21/21）、COPY_INDEX.json（81 份来源副本）、
   │                                            v800_bomb_used.py（执行器本体，sha256 5d85bcb0…d7bf）
   ├─ logs/                              (33)   本轮全部 .out（wave / 逐格 / smoke / A/B / D3）
   ├─ smoke/                              (5)   bomb_smoke_R{1..5}.json
   ├─ ab/                                 (8)   A/B 60 条 before/after/repro + 补丁前执行器
   └─ scripts/                            (9)   本包 4 个脚本 + 执行代理遗留 5 个（含一个陈旧 .pyc）
```

逐文件 sha256 / 字节 / 行数 / 源文件比对标志：`evidence/BOMBRULE2/manifest.json` 的 `files[]`（**115 条**，其中 21 条带 `copied_from` 与 `source_sha256_equals_copy_sha256`）。

---

## 4. 三份文档与清单的自哈希

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| [`SUMMARY_FOR_DECISION.md`](SUMMARY_FOR_DECISION.md) | 27,051 | `39a3318746b7cb66551bb61f060fcf42c8abad1f678532933fe9ba0b638e8433` |
| [`evidence/BOMBRULE2/RUN.md`](evidence/BOMBRULE2/RUN.md) | 32,988 | `e5019b4c4a012c43ee8b0bde0a40ec27c54e7b7a967b41753a433cb0803da662` |
| [`evidence/BOMBRULE2/diagnostics.json`](evidence/BOMBRULE2/diagnostics.json) | 175,407 | `9fef417dae058beae369209ab3cb0bef4e07333add9b0b36f322c88dcfdcd7f5` |
| [`evidence/BOMBRULE2/manifest.json`](evidence/BOMBRULE2/manifest.json) | 23,003 | `5bf84a0bd7482bf98fc2ab8dfaa4b4ec179dc899ec0d372c1f5c04139aced2b2` |
| [`evidence/BOMBRULE2/refs/CROSSCHECK.json`](evidence/BOMBRULE2/refs/CROSSCHECK.json) | 32,865 | `a27e04475bf538f78c778a9d1c6f36a02e169fe7d8a273d6aa7ec8cf3caa462f` |
| [`evidence/BOMBRULE2/refs/COPY_INDEX.json`](evidence/BOMBRULE2/refs/COPY_INDEX.json) | 21,733 | `088f7969f05909f4eb059e1d3e7ea977a7ac5cd2f078c0dfbf7206d4d7a82281` |

`manifest.json` 自带两级自校验：`aggregate_sha256`（对 `files[]` 的 `<sha256>  <relpath>\n` 串求哈希）与 `self_check.value`（对去掉 `self_check` 的整个 payload 求规范化 JSON 的哈希）。**两者都已独立复验通过**，且 115/115 个文件在盘上的 sha256 / 字节数与清单一致。

**`README_INDEX.md` 自身没有内嵌自哈希**（写入后再计算会使内容改变）。它的 sha256 应由统一的登记轮次计算并写入 `ASSET_SYNC.sha256`。

---

## 5. 复现（**不需要引擎**）

三个脚本都在 `evidence/BOMBRULE2/scripts/`，只读已发布副本（并比对 `.tmp_v840_bomb` 源文件 sha256），只写 `evidence/BOMBRULE2/`：

| 脚本 | 作用 |
| --- | --- |
| `publish_v840.py` | 复制 21 份 JSONL（源/副本 sha256 双算并断言相等）、重算全部统计、精确 Fisher+Holm、规则不变量、d900 快照史、R0 对 v8.1.0 复现 |
| `crosscheck_v840.py` | 21 格三方比对（副本 vs 执行器 `summary.json` vs 日志末行）→ `refs/CROSSCHECK.json` |
| `manifest_v840.py` | 逐文件 sha256 / 字节 / 行数 → `manifest.json`（含 `aggregate_sha256` 与 `self_check`） |
| `copy_refs_v840.py` | 81 份 provenance 副本（logs / smoke / ab / refs / commands / 执行器本体）→ `refs/COPY_INDEX.json` |

```powershell
$py = "python"
$ev = "Document\Update\Alpha\v8.4.0\evidence\BOMBRULE2\scripts"
& $py "$ev\publish_v840.py" --no-copy --frozen-at=2026-11-22T11:15:25+08:00   # 逐字节重现 diagnostics.json
& $py "$ev\crosscheck_v840.py"                                              # 期望 cells checked: 21, failing: 0
& $py "$ev\manifest_v840.py"                                                # 重写 manifest.json（自校验写入）
```

`--frozen-at` 钉住唯一会记「当前时间」的字段，因此上述命令**逐字节重现** `9fef417d…d7f5`（175,407 B）。

---

## 6. 这个包里**没有**什么（预注册期待但未产生）

| 缺件 | 原因 / 本包的处理 |
| --- | --- |
| `bomb_audit.json` | 执行器超时，`finalize.sh` 未执行 → 本包用 **334,517 次按雷的离线规则审计**替代（`diagnostics.json → arms_d600.*.rule_invariants`，0 违例） |
| `audio_audit.json` | 同上；静音证据改为日志里的 **93 条 `cfg muted=True` / 0 条 `False`** |
| `process_residue.json` | 同上（未执行；本包不伪造） |
| `ckpt_assertions_final.json` | `assert-ckpt --phase final` 未执行（**只有 start**，`all_pass: true`） |
| 执行器自己的 `diagnostics.json` / `manifest.json` | 未生成 → 本包的这两份是**发布方离线补做**的（schema 为 `v840-bombrule2-*`） |
| d900 的退出码（seed0/1） | 孤儿进程，wrapper 已死 → 完成性由 1,200 行 + `status=complete` + `summary.json` 三证 |
| 900 决策的 R0 / R3 格 | 预注册只要求「最佳臂」，本轮只有 R5 |
| R5 vs R3 的检验 | 预注册未登记该比较 → **不做**（SUMMARY §5 明确「不能声称 R5 > R3」） |

**与任务书的差异（如实列出）**：任务书把 d900 描述为「不完整、15.01 %、逐种子 17.4/33.2/0.0」。签发时确实是 1,001/971/1,200 行的中途快照；本轮发布时两个孤儿写手已跑完，**发布值是完整格的 17.6111 %（225/409/0）**；该快照的前缀复算是 **504/3,172 = 15.889 %**（逐种子 17.982 / 33.3677 / 0.00），而 **15.01 % 与 504/3,172 不自洽**（504/3,600 = 14.000 %）。三个时刻都写进 `RUN.md §2.2` 与 `SUMMARY §6.2`，发布值只用完整格。

---

## 7. 登记说明

**本轮不登记 `ASSET_SYNC.md` / `ASSET_SYNC.sha256`**（按要求留给统一的登记轮次）。应当登记的清单与 sha256 在 [`evidence/BOMBRULE2/RUN.md`](evidence/BOMBRULE2/RUN.md) §10 与 `manifest.json`（115 个文件）。

**约束遵守**：本轮只新增 `Document/Update/Alpha/v8.4.0/**` 下的文件（`evidence/BOMBRULE2/**` + `SUMMARY_FOR_DECISION.md` + 本文件）；`.tmp_v840_bomb/**` **只读**（无创建 / 修改 / 删除）；既有版本目录、预注册、其他证据**未改动**；**未启动引擎**；**未 `git add` / `commit`**；**未删除或移动任何资产**。
