# v8.2.0 索引 — BOMBLETH-001（lethal-only 放雷）：**部分完成**

**一句话**：预注册的 36 格只有 **14 格完整**；被收窄后的 **12 格设计 12/12 完整**；**核心对照 L2（≤5 px）vs 80 px 只在 600 决策档成立（20.75% vs 71.11%，−50.36 pp）**，1,200 决策档两者互撞地板（1.56% vs 1.00%），**900 / 1,500 两档不完整 ⇒ 本轮不宣布任何停止条件**。

- 数据冻结：**2026-11-22T09:42:39+08:00**
- 引擎：**本轮封包过程中没有启动任何引擎**（全部为盘上数据的复制 / 解码 / 计数）
- 统计：**没有做**（预注册 §2 的 P1/P2 未执行）

---

## 1. 怎么读（三条路径）

| 你有多少时间 | 读什么 |
| --- | --- |
| **3 分钟** | 本文件 + [`SUMMARY_FOR_DECISION.md`](SUMMARY_FOR_DECISION.md) 的 **§0** 与 **§5（能声称 / 不能声称）** |
| **10 分钟** | [`SUMMARY_FOR_DECISION.md`](SUMMARY_FOR_DECISION.md) 全篇（§0 结论 → §2 判据表 → §3 有效性诊断 → §6 冲突清单） |
| **全过程** | [`evidence/BOMBLETH/RUN.md`](evidence/BOMBLETH/RUN.md)（中止原因与范围、逐格完整性表、`bombs_used=0` 行数、窗口配对、机制证据、偏差清单、登记清单） |

判据原文：[`PREREGISTRATION_BOMBLETH_001.md`](PREREGISTRATION_BOMBLETH_001.md)（FROZEN，82 行，3,806 字节，sha256 `867d366e14b01bba401b02aa6781473d4ec671a635bd1fd0fab88be747bcef82`）。

---

## 2. 关键数字速查（全部为**完整格**，种子配对）

| 预算 | L1（≤3 px） | L2（≤5 px） | A3（80 px，引用 v8.1.0） |
| ---: | ---: | ---: | ---: |
| 300 决策 | — | — | **100.00%**（3,600/3,600） |
| **600 决策** | **0.36%**（13/3,600） | **20.75%**（747/3,600） | **71.11%**（2,560/3,600） |
| 900 决策 | 0.00%（0/2,400，仅 seeds 1,2） | **无完整格** | 12.67%（304/2,400，同 seeds） |
| **1,200 决策** | **0.00%**（0/3,600） | **1.56%**（56/3,600） | **1.00%**（36/3,600） |
| 1,500 决策 | 无 | 无 | **无格** |

- L1 **不是「从不触发」**：8 个完整格共触发 **80 次**（`bombs_used=1` 的 80 行；`bombs_used=0` 的 **9,520 / 9,600 行 = 99.17%**），触发距离 **2.833–2.996 px**，触发后只有 **13.75%** 活满。
- 机制：**2 帧决策粒度**（36 格逐行 `decision_frames: 2`）⇒ 死亡局里「最小观测 lethal 距离始终 > 阈值」的比例：**L1 98.75–99.83%**、**L2 59.7–77.4%**，最小距离中位数仅 **~5.3 px** ⇒ 3–5 px 的判据**几乎不可观测**，不只是「太晚」。
- 代价：L2 每局只用 **0.60** 颗雷（80 px 是 **2.965** 颗），**省 79% 的雷，没换来更远**。

---

## 3. 目录清单

```
Document/Update/Alpha/v8.2.0/
├─ PREREGISTRATION_BOMBLETH_001.md      3,806   预注册（FROZEN，未改动）
├─ SUMMARY_FOR_DECISION.md             22,466   决策摘要（§0–§6）
├─ README_INDEX.md                     本文件
└─ evidence/BOMBLETH/                218 个文件 / 35,187,843 字节
   ├─ RUN.md                           31,847   执行记录
   ├─ diagnostics.json                 68,674   离线诊断（本包生成，engine_started=false）
   ├─ audio_audit_recheck.json         36,031   静音全量复核（103/103 muted）
   ├─ SNAPSHOT_MANIFEST.json          111,952   210 个文件的逐文件 sha256 / 行数 / 完整性
   ├─ SNAPSHOT_MANIFEST.freeze1_093441.json  112,599   第一次冻结的留痕
   ├─ eval_L1_*.jsonl  (8)             ——      完整格（原名）
   ├─ eval_L2_*.jsonl  (6)             ——      完整格（原名）
   ├─ eval_L1/L2/L3_*.jsonl.partial (22) ——    不完整格（`.partial` 后缀，**不入表**）
   ├─ eval_*.summary.json / *.summary.json.partial (36)   执行器逐格摘要
   ├─ refs_v810/ (13)                  90,550   v8.1.0 A3（80 px）参考格，sha256 冻结
   ├─ refs/ (42)                       72,301   ckpt 断言、D3 保真、S-ab、静音审计、逐格 commands TSV
   ├─ logs/ (50)                       69,614   `.out` 日志（含 `wave.out` / `pool.out`）
   ├─ smoke/ (5)                    3,147,340   smoke 原始记录（L1 / L1_rerun / L2 / L3 / L1_60ep）
   ├─ ab/ (4)                         227,808   S-ab 改动前后各 60 条
   └─ scripts/ (27)                   169,729   本轮脚本 + 归档的改动前执行器
```

完整 210 个文件的逐文件 sha256、字节数、行数与完整性判定：`evidence/BOMBLETH/SNAPSHOT_MANIFEST.json`（`files[]` 的每一条都有 `relpath / pub_sha256 / pub_bytes / lines / complete / snapshot_at`）。

---

## 4. 三份文档与两份清单的自哈希

| 文件 | 字节 | sha256 |
| --- | ---: | --- |
| [`SUMMARY_FOR_DECISION.md`](SUMMARY_FOR_DECISION.md) | 22,466 | `0432aa19a6fbafc12b5bffab21171baa82a115861f30acd342b6eafad417ac57` |
| [`evidence/BOMBLETH/RUN.md`](evidence/BOMBLETH/RUN.md) | 31,847 | `a134be54f8d814be363d6ec42686adc9f2c32b484f97c15eb544e0c537dfc5d4` |
| [`evidence/BOMBLETH/diagnostics.json`](evidence/BOMBLETH/diagnostics.json) | 68,674 | `2ffa0b37f1080a701f8b20b5f10c2bece31185d012be55f018f0bd1b6db20b15` |
| [`evidence/BOMBLETH/SNAPSHOT_MANIFEST.json`](evidence/BOMBLETH/SNAPSHOT_MANIFEST.json) | 111,952 | `f13cd6801b4d0438f16104a5ccfdbf67a37c9b226dd096f1e8878997ca2f3938` |
| [`evidence/BOMBLETH/SNAPSHOT_MANIFEST.freeze1_093441.json`](evidence/BOMBLETH/SNAPSHOT_MANIFEST.freeze1_093441.json) | 112,599 | `e7e33734a3fd3fe933761a978d5d66953a8fcd16b8b24b13eb35a6f32e0abed3` |
| [`evidence/BOMBLETH/audio_audit_recheck.json`](evidence/BOMBLETH/audio_audit_recheck.json) | 36,031 | `56f1c39651859424b153c27570a088ecd80d735b590233a838ddc615a0e6ef24` |

**`README_INDEX.md` 自身没有内嵌自哈希**（写入后再计算会使内容改变）。它的 sha256 应由统一的登记轮次计算并写入 `ASSET_SYNC.sha256`。

---

## 5. 复现（**不需要引擎**）

三个脚本都在 `evidence/BOMBLETH/scripts/`，只读 `.tmp_v820_bomb` 与 `.tmp_v810_bomb`，只写 `evidence/BOMBLETH/`：

| 脚本 | 作用 | sha256 |
| --- | --- | --- |
| `v820_publish_snapshot.py` | 冻结快照：逐文件流式复制、丢掉撕裂的尾行、同时算源与副本的 sha256、按 1,200 行判定完整性并加 `.partial` | `5f1fab728d5e353164aa07ad57e1a9d28bfa81134c617db3b3ab1ff0c671b04f` |
| `v820_analyze_snapshot.py` | 只用已发布副本生成 `diagnostics.json`（逐格 / 逐窗口、完整格配对、触发可观测性） | `b86ce65f398fc6bafa3f8b1b9e1375aa5471d8bac744bb36608f3005b0b9e532` |
| `v820_audio_recheck.py` | 用 `tools/rl/v400_make_cfg.py --inspect` 解码全部 `th08.cfg`，补执行器那次审计之后的缺口 | `8d5487c6597d540da4a6a58da1ceb1da879c123a454415235cdb2da17ef0c13f` |

```powershell
$py = "<python>"
& $py Document\Update\Alpha\v8.2.0\evidence\BOMBLETH\scripts\v820_analyze_snapshot.py
& $py Document\Update\Alpha\v8.2.0\evidence\BOMBLETH\scripts\v820_audio_recheck.py   # 退出码 0 == 无 audible
```

---

## 6. 这个包里**没有**什么（预注册期待但未产生）

| 缺件 | 原因 |
| --- | --- |
| `bomb_audit.json` | 执行器没跑到 `audit` 步骤；`finalize.sh` 的 5 步全部未执行 |
| 执行器自己的 `diagnostics.json` | 同上；本包的 `diagnostics.json` 是**执行代理离线补做**的 |
| `manifest.json` | 同上；本包用 `SNAPSHOT_MANIFEST.json` 替代，**schema 不同**（`v820-bombleth-snapshot-manifest/1`） |
| `ckpt_assertions_final.json` | `assert-final` 未执行（只有 start） |
| `process_residue.json` | `process-audit` 未执行 |
| L3 的 36 格（本轮自跑） | 09:16 收窄后不再自跑，改为引用 v8.1.0 的 13 个 `summary.json` |
| 900 / 1,500 档的完整格 | 900 档 L2 三格 rc=4 断管；1,500 档 9 格 rc=143 被 SIGTERM |
| 任何统计结果 | P1（Fisher + Holm）/ P2 未执行 |

---

## 7. 登记说明

**本轮不登记 `ASSET_SYNC.md` / `ASSET_SYNC.sha256`**（按要求留给统一的登记轮次）。
**应当登记的清单与 sha256 在 [`evidence/BOMBLETH/RUN.md`](evidence/BOMBLETH/RUN.md) §8**（14 个完整格、22 个 `.partial` 格、13 个 v8.1.0 参考格、6 个运行级证据、归档的改动前执行器），完整 210 个文件的哈希在 `SNAPSHOT_MANIFEST.json`。

**约束遵守**：本轮只新增 `Document/Update/Alpha/v8.2.0/**` 下的文件；`.tmp_v820_bomb` 与 `.tmp_v810_bomb` **只读**（无创建 / 修改 / 删除）；既有版本目录与预注册**未改动**；**未启动引擎**；**未 `git add` / `commit`**。
