# W01c 前置事实：本机只有 **1 条** 记录在案的真实 Stage-1 通关

日期：2026-11-15，Asia/Shanghai。执行者：主代理（只读盘点，用于裁定 CONTRACT 的 3+3 能否凑齐）。
不启动引擎、不修改任何证据。真正的重放实测由 W01c 执行代理负责（见后续 `W01C_PREFLIGHT.md`）。

## 1 方法

对下列目录做**有界**扫描（只读 `*.json`，单文件上限 4 MiB，总上限 40,000 文件；不读像素、不读 `.pt`）：

```text
.tmp_v420_data  .tmp_v420_evalpulse  .tmp_v420_clearcheck  .tmp_v420_accept
.tmp_v420_remeasure  .tmp_v400_sclear  .tmp_v400_sclear2
Document/Update/Alpha/v4.2.0/evidence
```

判定口径：**顶层记录自身**声明 `"stage_clear": true`（不接受嵌套引用、不接受摘要里"曾有通关"的说法）。

## 2 结果

```text
scanned_json = 19,404     skipped_too_big = 1     top_level_clear_records = 1
distinct (frames, score, action_sequence_sha256) triples = 1
```

唯一一条：

| 字段 | 值 |
| --- | --- |
| 记录 | `.tmp_v420_evalpulse/positive_control.json` |
| stage / seed | 0 / 12 |
| frames | 13,195 |
| score | 2,631,430 |
| stop_reason | `stage_clear` |
| deaths | 0.0 |
| `action_sequence_sha256` | `5e9d66c4f5685da9f0ea56f3105656b7210721124f8b74a12813b49089995cd3` |

## 3 这条唯一通关已在本次会话被**逐位复现**

主代理独立重跑正对照（严格按 `.tmp_v420_evalpulse/20_control_pulse.sh` 的配方，私有 data dir）：

```text
exit=0  wall_s=125.9  harness_role=env  error=None
frames=13195  score=2631430  stage_clear=True  stop_reason=stage_clear  deaths=0.0
action_sequence_sha256 = 5e9d66c4f5685da9f0ea56f3105656b7210721124f8b74a12813b49089995cd3   ← 与记录逐位相同
observed_input_sha256  = 315a2ab6552edcba23a40f49069150e52eac67f034407e9480f48a66f47c6212   ← 与记录逐位相同
not_a_policy_result=True  not_a_gate=True
```

即：**环境可复现，通关磁带可复现，哈希逐位一致。** 这条可以作为 CONTRACT 的 1 个 success 来源。

## 4 对 CONTRACT 的直接影响（必须正面记录）

`PLAN_SPEC.json` `datasets.CONTRACT` 要求 `success_sources: 3, failure_sources: 3, repeats_per_source: 2`。

按 §2 的盘点：

| 类别 | 可用来源 | 缺口 |
| --- | ---: | --- |
| success | **1**（seed12 磁带 + pulse 48） | **缺 2** |
| failure | 待 W01c 实测（负对照 = 同一磁带关掉 pulse → `game_over`；seed10/11 是否通关未知） | 取决于实测 |

关键点：**把同一磁带重放多次不能算成多个来源**——`TRAINING_PLAN.md` §4 明确"相邻帧、裁剪、重标、重复重放、由同一起点生成的多个分支属于同组"。所以 success 侧**现在最多 1**。

补齐 3 个 success 只有三条路，且都不是 W01 能独自完成的：

1. **seed10 / seed11 实测能通关** → 但它们采集时的记录是 `stage_clear: False`（`slot_probe` 已复核 seed10），
   加上 tail 延伸 + pulse 后是否通关属未知，正是 W01c 步骤 3 要测的。
2. **新的真实通关** → 需要真正会玩的策略（即 W03/W04 的目标本身），W01 阶段不可能有。
3. **修订 CONTRACT 定义**并说明理由（例如：仪器校准只需 1 正 + 1 负配对，3+3 留给后续有更多通关时再满足）。

## 5 建议（供裁定与 W12 结算引用）

- **不要把 W01 判成"CONTRACT 3+3 已满足"。** 现状是 success 1/3。
- W01 的**仪器校准**意义（正负结果都能报、裁判与输入一致）已由 §3 的单对正负对照**真实达成**；
  这部分可以接受。
- `CONTRACT.success_sources = 3` 应作为**未闭合的缺口**登记，阻塞条件明确：**缺少第 2、3 条真实通关磁带**。
  这属于"数据不足"，不是"代码失败"，也不应通过重复重放同一磁带冒充。
- 若 seed10/11 实测能通关，缺口降为 1；若都能通关则缺口为 0——以 W01c 实测为准，本文不预判。

## 6 追加：seed10 实测**也能通关**（本条晚于 §2 的扫描时点）

§2 的扫描是 **2026-11-15 11:18** 的快照，当时只有 1 条 `stage_clear: true` 记录。
W01c 随后对 seed10 磁带做真实重放（pulse=48，绕过 shard sha pin，`shard_pin_bypassed=true`），**通关**：

| 字段 | seed12（原有记录） | seed10（本次新产出） |
| --- | --- | --- |
| `stage_clear` | True | **True** |
| `frames` | 13,195 | **13,291** |
| `score` | 2,631,430 | **2,868,224** |
| `deaths` | 0.0 | **0.0** |
| `stop_reason` | `stage_clear` | **`stage_clear`** |
| `action_sequence_sha256` | `5e9d66c4…` | **`3cb366a0…`** |
| 记录 | `.tmp_v420_evalpulse/positive_control.json` | `.tmp_v430_replay/runs/step3_seed10.json` |

**这是两个真正不同的成功来源**：帧数、分数、动作序列哈希三项都不同，不是同一条磁带的重放。

两条修正：

1. **"采集时的 `stage_clear` 记录"不能预测"重放能否通关"。** seed10 的 shard 在采集时记录的是
   `stage_clear: False`（`v420_slot_probe` 复核过），但加上 tail 延伸与 pulse 后重放却通关。
   所以 §4 的"必须实测"是对的，任何"采集纪录 = 重放结果"的推断都不成立。
2. §4 的 success 缺口更新为：**已知至少 2 个**（seed12、seed10），不是 1 个。
   seed11 正在实测；若也通关，success 侧**凑齐 3 个**。

**正式计数仍保留一条限制**：seed10 这次走的是"绕过 shard sha pin"的分类路径
（详见执行代理的 `W01C_PREFLIGHT.md`）。要升为**正式 CONTRACT 来源**，还需在同一 instrument 身份下、
以未绕过的路径再确认一次（或把它的 shard sha 正式登记进来源清单）。这一步尚未做。

## 7 追加：seed11 也通关 —— **三个磁带全部通关，success 3/3 达成**

| seed | frames | score | deaths | `action_sequence_sha256` | 结论 |
| --- | ---: | ---: | ---: | --- | --- |
| 12 | 13,195 | 2,631,430 | 0.0 | `5e9d66c4…` | stage_clear |
| 10 | 13,291 | 2,868,224 | 0.0 | `3cb366a0…` | stage_clear |
| 11 | 13,291 | 2,768,761 | 0.0 | `19be3bc6…` | stage_clear |

三条**动作序列哈希互不相同**，分数也不同（seed10 与 seed11 帧数同为 13,291 但分数与哈希不同），
所以是**三个独立的成功来源**，不是同一轨迹的重复。

**因此 `CONTRACT.success_sources = 3` 在实测层面已达成**（3 个真实通关磁带，各可重复重放）。
仍需保留的限制：seed10/11 走的是"绕过 shard sha pin"的分类路径，
升为**正式来源**需未绕过路径复核或正式登记 shard sha。

**failure 侧仍未达标**：目前只有 1 个（seed12 + pulse 关掉 → `game_over`）。
已要求执行代理补测 seed10/11 的 pulse-off 配对，以凑齐 failure 3/3。

