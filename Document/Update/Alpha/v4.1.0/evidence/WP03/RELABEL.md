# WP03.1a — D1 现存 obs + 原始 work `verified` 明细的 calibrated 重标（证据）

- 工作包：v4.1.0 WP03.1a（只做离线重标切片：**不启动游戏、不训练、不采集新数据、不 commit/push**）
- 代码：`tools/rl/v410_relabel_d1.py`
- 测试：`tests/test_v410_relabel_d1.py`（30 例）
- 机器可读证据：[`relabel_manifest.json`](relabel_manifest.json)、[`s1_partial_audit.json`](s1_partial_audit.json)
- 标签输出：`.tmp_v410_wp03_relabel/<shard>/labels_v410.jsonl`（15 个文件、54,625 行、202,401,419 bytes，**不入库**；只写标签，**不复制 obs**）
- 只读输入：`.tmp_v400_d1/d1/shards/<shard>/`（`obs/play96x112.npy`、`hud.npy`、`act_hist.npy`、旧 schema `labels.jsonl`、`shard.json`）、`.tmp_v400_d1/work/<shard>/labels.jsonl`（work 原始 `verified` 明细）、`touhou_rl/v410/teacher_labels.py`、`teacher_calibration.py`（只读 import，未修改）
- 状态：**真实 obs + calibrated labels 重标切片完成并自检通过；D410-S1 / D410-MULTI 仍是 `shortfall`，不得写成通过。**

## 0 一句话结论

D1 只有 **15 个 shard** 仍保留 work 原始 `labels.jsonl`（含逐 plan 的 `verified` 明细）。把这 15 个 shard 的 work decisions 按 `decision + hash_pre + decision_frame` 与现有 obs 逐条对齐：**54,632 条 work decisions 中 54,625 条有对应观测行**，7 条 stage transition 尾行按规则裁剪并逐条记录；calibrated 重标为 **alive 54,341 / dead 6 / unknown 278（278 全为 `model_mismatch`，0 `incomplete_execution`）**。stage 0 切片 **26,625 decisions / 8 seeds** 对 S1 的 50,000 / 20 是 `shortfall`（差 **23,375 decisions / 12 seeds**）；全部 15 shard 对 MULTI 的 2,000,000 帧 / 每面 150,000 / 8 面同样是 `shortfall`（**218,500 帧 / 2 面**）。**观测+标签对齐链路已真实跑通，但这不是完整 D410-S1 或 D410-MULTI。**

---

## 1 实际执行的验收命令与退出码

三条命令均已真实执行（Windows，`PYTHONPATH=D:\WorkShop\Touhou_RL`），最后一次运行的退出码：

| # | 命令 | 结果 | 退出码 |
|---|------|------|--------|
| 1 | `python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp031_pytest -q tests/test_v410_relabel_d1.py` | `30 passed in 0.50s` | **0** |
| 2 | `python -m compileall -q tools/rl/v410_relabel_d1.py` | 无输出 | **0** |
| 3 | `python tools/rl/v410_relabel_d1.py --d1-root .tmp_v400_d1/d1 --work-root .tmp_v400_d1/work --out .tmp_v410_wp03_relabel --manifest Document/Update/Alpha/v4.1.0/evidence/WP03/relabel_manifest.json --s1-audit Document/Update/Alpha/v4.1.0/evidence/WP03/s1_partial_audit.json` | 见下方 summary JSON；耗时约 10 s（含 2.85 GB obs 的 sha256 校验） | **0** |

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp031_pytest -q tests/test_v410_relabel_d1.py
python -m compileall -q tools/rl/v410_relabel_d1.py
python tools/rl/v410_relabel_d1.py --d1-root .tmp_v400_d1/d1 --work-root .tmp_v400_d1/work --out .tmp_v410_wp03_relabel --manifest Document/Update/Alpha/v4.1.0/evidence/WP03/relabel_manifest.json --s1-audit Document/Update/Alpha/v4.1.0/evidence/WP03/s1_partial_audit.json
```

命令 3 的 stdout（最后一次运行）：

```json
{"manifest": "Document/Update/Alpha/v4.1.0/evidence/WP03/relabel_manifest.json", "s1_audit": "Document/Update/Alpha/v4.1.0/evidence/WP03/s1_partial_audit.json", "shards_processed": 15, "decisions": 54625, "work_decisions": 54632, "trimmed_decisions": 7, "alive": 54341, "dead": 6, "unknown": 278, "model_mismatch": 278, "horizon_missing": 54625, "stage0_decisions": 26625, "stage0_seeds": 8, "s1_status": "shortfall", "s1_shortfall_decisions": 23375, "s1_shortfall_seeds": 12, "multi_status": "shortfall", "multi_frames": 218500}
```

生成的证据文件（本次新建，未覆盖任何旧文件）：

| 文件 | bytes | sha256 |
|------|-------|--------|
| `relabel_manifest.json` | 94,761 | `90beab1331046fd1eca742c12aa7a4ba8b2296975327a71d1b7cc0efff36ca12` |
| `s1_partial_audit.json` | 8,263 | `1eff4b36bd530a05fcf7b6801a1a365b2c6d3f0367e92b09d8246926b2028ed6` |
| `tools/rl/v410_relabel_d1.py` | — | `d669f7a76995f1816595fcd2a9294b9cfad094dc21801dd3291459a353f7b4a4` |
| `tests/test_v410_relabel_d1.py` | — | `55772eef67eacef9800cce6d66a54a1f56088f0636aa56b1f2be2b772562a9d1` |

---

## 2 输入扫描（脚本实际枚举，不假定数量）

`.tmp_v400_d1/d1/shards/` 下共 **142 个目录**：137 个具名 shard + 5 个 `.partial*` 工作目录（`.d1-s1d1-n4000-seed1-shard.partial` 及 4 个 `.partial.failed-*`）。

`.tmp_v400_d1/work/` 下 47 个目录，但与 shard 目录**同名**的只有 **15 个**，且这 15 个都真的有 `labels.jsonl` → 可重标切片 = **15 个 shard**（与任务预期一致，按实际枚举结果写入 manifest 的 `source`）：

- **8 个 stage 0 shard**（`d1-s0d1-n3371-*`，seeds 1–8，每个 3,274–3,371 decisions）；
- **7 个 stage 1 shard**（`d1-s1d1-n4000-*`，seeds 2–8，每个 4,000 decisions）；
- 其余 **127 个** shard 只有 obs、没有 work labels → 不在切片内（`shard_dirs_without_work_labels=127`），不做任何猜测式补全。

要点：**work JSONL 是这些观测上逐 plan `verified` 明细的唯一保留副本**。D1 shard 侧的旧 `labels.jsonl`（`v400-teacher-obs-1`）只有 `verified_mask`/`verified_count`，没有任何一个 plan 的 `died`/`max_position_error`/`frames_advanced`，无法单独做 calibrated 重标。

---

## 3 对齐规则与裁剪

### 3.1 对齐键（逐条验证，不是按位置硬截断）

对每个 decision `i`：

1. shard obs 行数 == shard `labels.jsonl` 行数 == `shard.json.decision_count`（三者必须一致）；
2. work decision 必须从 0 连续：中间缺行 / 重复行 → `RelabelAlignmentError`；
3. work `decision == i`、shard `decision == i`；
4. shard `hash_pre` == work `hash_pre["hash"]`（work 侧是 `{"hash": str, "frames": int}` 结构，shard 侧是裸字符串，脚本按 hash 归一化比较）；
5. shard `decision_frame` == work `hash_pre["frames"]`；
6. work 行数不足 shard obs 行数 → 直接失败（有观测行没有标签）。

`shard.json` 里记录的 `labels.sha256` / 每个 obs 平面的 `sha256` 与磁盘文件逐个复核；不一致即拒绝该 shard（不允许对"像素已变"的 obs 重标）。

### 3.2 frame 语义（看起来像错位、其实不是）

- work 行的 `frame` 是**该 decision 执行之后**的帧：`4*(i+1)`；
- shard 行的 `frame` / `decision_frame` 是**执行之前**的帧：`4*i`；
- `obs` 第 `i` 行是 decision `i` 的 pre-commit 观测（`decision_time_before_commit`）。

所以对齐键用 `hash_pre`（在 pre-commit 帧上记录）+ `decision_frame`，**不用 `frame`**；每行标签同时写出 `frame`（work 值）、`decision_frame`、`obs_row`，避免下游混淆。

### 3.3 真实裁剪明细（`trimmed_decisions=7`）

只有 stage 0 的 7 个 shard 多出 1 条尾行，均为 stage transition 决策（obs 在该决策之前结束）：

| shard | 裁剪 decision | work 行号 | work frame | decision_frame | 原因 |
|---|---|---|---|---|---|
| `d1-s0d1-n3371-seed1-shard` | 3370 | 3372 | 13484 | 13480 | `stage_transition_trim` |
| `d1-s0d1-n3371-seed2-shard` | 3322 | 3324 | 13292 | 13288 | `stage_transition_trim` |
| `d1-s0d1-n3371-seed3-shard` | 3346 | 3348 | 13388 | 13384 | `stage_transition_trim` |
| `d1-s0d1-n3371-seed4-shard` | 3274 | 3276 | 13100 | 13096 | `stage_transition_trim` |
| `d1-s0d1-n3371-seed6-shard` | 3274 | 3276 | 13100 | 13096 | `stage_transition_trim` |
| `d1-s0d1-n3371-seed7-shard` | 3346 | 3348 | 13388 | 13384 | `stage_transition_trim` |
| `d1-s0d1-n3371-seed8-shard` | 3322 | 3324 | 13292 | 13288 | `stage_transition_trim` |

- `d1-s0d1-n3371-seed5-shard`：work 3,371 == shard 3,371 → **裁剪 0**（该 shard 的 transition 决策有 obs）。
- 7 个 stage 1 shard：work 4,000 == shard 4,000 → **裁剪 0**。
- 每个 shard 都校验 `对齐 decisions + 裁剪 decisions == work report.decisions`（如 3370 + 1 == 3371），不吻合直接失败。

### 3.4 fail-closed 条件（测试逐条覆盖）

中间缺行 / 重复 decision / obs 行数与 shard decisions 不符 / work 行数不足 / `hash_pre` 不符 / `decision_frame` 不符 / 尾行没有越过最后一个观测帧 / 缺 report 行 / 坏 JSON / obs shape 或 sha256 与 `shard.json` 不符 / `shard.json.decision_count` 不符 → 全部 `RelabelAlignmentError` / `RelabelInputError`，不写 manifest、CLI 退出码 2 或 3。空行允许（计入 `blank_lines`），坏 JSON 不允许。

### 3.5 与旧 labels 的一致性核对

- 54,625 个对齐 decision 上，旧 shard `labels.jsonl` 的 `verified_mask` 与 work 行 `verified_mask`**逐个相同：54,625/54,625，差异 0**；`verified_count` 也 54,625/54,625 相同 → 旧标签与 work 标签确实是同一批决策。
- 新标签行内 `verified_mask_matches_verified_actions` 全为 true（0 个 false）→ 声明为 verified 的候选动作与 `verified` 明细一致。
- 45 个 obs 平面（15 shard × play/hud/act_hist）sha256 **45/45** 与 `shard.json` 记录一致。

---

## 4 真实重标数字

### 4.1 总计（`relabel_manifest.json.totals`）

| 指标 | 数值 |
|---|---|
| shards processed | **15** |
| decisions（对齐、有 obs） | **54,625** |
| work decisions（原始） | **54,632** |
| trimmed decisions | **7**（全部 `stage_transition_trim`） |
| verified plans | **54,625**（每决策恰好 1 个 verified plan） |
| alive / dead / unknown | **54,341 / 6 / 278** |
| unknown 细分 | `model_mismatch` **278**，`incomplete_execution` **0** |
| moving_alive / idle_alive | **33,159 / 21,182** |
| died_true / died_false plans | **6 / 54,619** |
| horizon_missing | **54,625**（work JSONL 不写 `verification_horizon`，与 WP02.4 的 3,251/3,251 同因） |
| 旧保守口径（WP02.3，`trajectory_matches` @1e-3） | alive **7,270** / dead **6** / unknown **47,349** |
| query_status | `verified` **54,625** / `unknown` **928,625**（每决策 1 verified + 17 unknown） |
| 帧数（obs 对齐） | **218,500** = 54,625 × 4 |
| 帧数（work report `survival_frames`） | **218,528**（含被裁剪 7 行的 28 帧） |
| 校准规则 | `MODEL_MATCH_TOLERANCE=8.0 px`、`decision_frames=4`（`report.committed_frames_per_decision`）、`horizon=32`（`report.config.horizon`） |

按 stage 拆分：

| stage | shards | seeds | decisions | frames（obs） | alive | dead | unknown |
|---|---|---|---|---|---|---|---|
| 0 | 8 | 1–8 | 26,625 | 106,500 | 26,488 | 0 | 137 |
| 1 | 7 | 2–8 | 28,000 | 112,000 | 27,853 | 6 | 141 |

### 4.2 每个 shard（`relabel_manifest.json.shards`）

| shard | stage | seed | decisions | trimmed | alive | dead | unknown | model_mismatch | 旧保守 alive |
|---|---|---|---|---|---|---|---|---|---|
| `d1-s0d1-n3371-seed1-shard` | 0 | 1 | 3,370 | 1 | 3,355 | 0 | 15 | 15 | 705 |
| `d1-s0d1-n3371-seed2-shard` | 0 | 2 | 3,322 | 1 | 3,303 | 0 | 19 | 19 | 645 |
| `d1-s0d1-n3371-seed3-shard` | 0 | 3 | 3,346 | 1 | 3,327 | 0 | 19 | 19 | 655 |
| `d1-s0d1-n3371-seed4-shard` | 0 | 4 | 3,274 | 1 | 3,255 | 0 | 19 | 19 | 641 |
| `d1-s0d1-n3371-seed5-shard` | 0 | 5 | 3,371 | 0 | 3,362 | 0 | 9 | 9 | 633 |
| `d1-s0d1-n3371-seed6-shard` | 0 | 6 | 3,274 | 1 | 3,259 | 0 | 15 | 15 | 875 |
| `d1-s0d1-n3371-seed7-shard` | 0 | 7 | 3,346 | 1 | 3,332 | 0 | 14 | 14 | 689 |
| `d1-s0d1-n3371-seed8-shard` | 0 | 8 | 3,322 | 1 | 3,295 | 0 | 27 | 27 | 620 |
| `d1-s1d1-n4000-seed2-shard` | 1 | 2 | 4,000 | 0 | 3,962 | **6** | 32 | 32 | 291 |
| `d1-s1d1-n4000-seed3-shard` | 1 | 3 | 4,000 | 0 | 3,984 | 0 | 16 | 16 | 209 |
| `d1-s1d1-n4000-seed4-shard` | 1 | 4 | 4,000 | 0 | 3,983 | 0 | 17 | 17 | 761 |
| `d1-s1d1-n4000-seed5-shard` | 1 | 5 | 4,000 | 0 | 3,984 | 0 | 16 | 16 | 112 |
| `d1-s1d1-n4000-seed6-shard` | 1 | 6 | 4,000 | 0 | 3,982 | 0 | 18 | 18 | 172 |
| `d1-s1d1-n4000-seed7-shard` | 1 | 7 | 4,000 | 0 | 3,979 | 0 | 21 | 21 | 152 |
| `d1-s1d1-n4000-seed8-shard` | 1 | 8 | 4,000 | 0 | 3,979 | 0 | 21 | 21 | 110 |

6 个 `verified_dead` 全部来自 `d1-s1d1-n4000-seed2-shard`；而 15 份 work report 的 `deaths_observed` 都是 **0**。两者不矛盾：`died` 是 shadow replay 对整个 plan 的判定，`deaths_observed` 是这局实际玩家死亡计数；重标保留 plan 级 `died=True`，不把它解释成 run 死亡。

### 4.3 观测平面与 SHA256

| 平面 | 行数（总） | bytes（总） | sha256 与 `shard.json` 一致 |
|---|---|---|---|
| `obs/play96x112.npy` | 54,625 | 2,349,319,830 | 15/15 |
| `obs/hud.npy` | 54,625 | 489,447,830 | 15/15 |
| `obs/act_hist.npy` | 54,625 | 7,873,830 | 15/15 |

obs 总 bytes **2,846,641,490**（约 2.85 GB），**未复制**任何字节；manifest 里每个 shard 记录三个平面的相对路径、shape、dtype、bytes、sha256。`obs.copied=false`。

### 4.4 标签输出

- 15 个 `labels_v410.jsonl`，共 **54,625 行 / 202,401,419 bytes**，路径 `.tmp_v410_wp03_relabel/<shard>/labels_v410.jsonl`（`.tmp_*` 不入库）；每个文件的行数、bytes、sha256 记录在 manifest 的 `labels_output`。
- 每行 schema `v410-teacher-label-calibrated/1`（WP02.4 行契约），含 `decision`、`frame`、`decision_frame`、`obs_row`、`obs`（shard/row/decision_frame/planes 相对路径）、`query_status`（18 项）、`survival_status`、`verified`（逐 plan `died`/`death_frame`/`frames_advanced`/`max_position_error`/`survival_reason`/`survival_detail`/`model_match`/`conservative_*`）、`calibration`、`work_line`。
- 仅写标签文件：输出目录下 `*.npy` 数量 = **0**（测试 `test_relabel_writes_labels_only_never_observations` 固定该约束）。

---

## 5 S1 partial 审计（**shortfall，不是 passed**）

`s1_partial_audit.json` 只统计 work report 记录 `stage == 0` 的 **8 个 shard**（D1 shard 侧 `labels.jsonl` 的 `stage` 字段对全部 15 个 shard 都写 0，包括 7 个 stage 1 run，因此**只作为记录、不用来选择**）：

| 指标 | 数值 |
|---|---|
| stage 0 shards | 8（seeds 1–8） |
| decisions（对齐、有 obs） | **26,625** |
| work decisions | 26,632 |
| trimmed | 7 |
| unique seeds | **8** |
| observations_present | **true**（26,625 行观测，逐 shard 行数与 decisions 相等） |
| alive / dead / unknown | **26,488 / 0 / 137** |
| 帧数 | 106,500（obs）/ 106,528（report） |
| `against_s1.status` | **`shortfall`**，`passed=false` |
| `shortfall_decisions` | **23,375**（= 50,000 − 26,625） |
| `shortfall_seeds` | **12**（= 20 − 8） |
| seeds 是否等于 S1 计划的 1000–1019 | **否**（实际 1–8，`seed_set_matches_s1_plan=false`） |

说明：任务预期写作 **26,632**，那是 **work decisions** 数；有 obs 可对齐的 **26,625** 才是本切片可用的 decisions（差 7 即 3.3 节的裁剪）。两个数字都写进审计（`decisions` / `work_decisions`），`shortfall` 结论不变。**不得把这个切片写成 S1 passed / S1 完成。**

---

## 6 MULTI partial（**shortfall**）

| 指标 | 实际 | 目标 | 差距 |
|---|---|---|---|
| stages | 2（0、1） | 8 | 缺 6 面 |
| frames（obs 对齐） | 218,500 | 2,000,000 | **1,781,500** |
| stage 0 帧 | 106,500 | ≥150,000/面 | 43,500 |
| stage 1 帧 | 112,000 | ≥150,000/面 | 38,000 |
| decisions | 54,625（work 54,632） | — | — |

`multi_partial.status="shortfall"`、`passed=false`，`checks` 逐项写明未达标原因。

---

## 7 与 D1 旧 labels / WP02 参考的差异

1. **旧 D1 `labels.jsonl` 没有可重标的明细**：schema `v400-teacher-obs-1` 只保留 `verified_mask` / `verified_count`，没有 `verified`（逐 plan 的 `died`/`max_position_error`/`frames_advanced`/`trajectory_matches`/`end_xy`），也没有 `geo_score`、`state`、`consistency`、`events` 等 work 顶层键；重标把这些明细从 work JSONL 恢复出来，并加上 `query_status`/`survival_status`。
2. **决策集合一致**：54,625 个对齐决策的 `verified_mask`/`verified_count` 与旧标签 100% 相同；差异只在于裁剪掉的 7 条 transition 决策（旧标签本来也没有它们的 obs）。
3. **旧标签的 `stage` 字段不可信**：15 个 shard 的旧 `labels.jsonl` 全部写 `stage=0`，包括 7 个实际 stage 1 的 run（`state.stage=1`）；manifest 用 work report 的 `stage`，并把旧字段记为 `shard_label_stage`。
4. **`frame` 语义不同**：旧 shard 行 `frame == decision_frame == 4*i`，work 行 `frame == 4*(i+1)`；新标签同时给出两者 + `obs_row`。
5. **与 WP02.4 参考语料同一规则、不同口径基数**：WP02.4 在 seed 1000 的 3,251 条 r 上得到 calibrated 3,226 alive / 0 dead / 25 unknown、旧保守 795/0/2,456；本切片按同一规则（tol 8.0 px）但有 `decision_frames=4`、`horizon=32`，得到 54,341 / 6 / 278 与旧保守 7,270 / 6 / 47,349。差异主要来自旧保守口径在 4 帧/决策下要求 `duration*4` 帧且必须 `trajectory_matches=True`（D1 的 `trajectory_matches` 几乎全为 false），这正是 WP02.4 校准要解决的问题。
6. **`horizon_missing` 全为 true**：work JSONL 不声明 `verification_horizon`；calibrated 规则用 `frames_advanced >= max(32, duration*4)` 作为 horizon（`horizon_source="frames_advanced"`），与 WP02.4 的处理一致。
7. **6 个 `died=True` 的 plan 与 report `deaths_observed=0` 并存**：前者是 plan 级 shadow replay 判定，后者是本局玩家死亡计数；重标两者都保留，不互相覆盖。

---

## 8 限制与"不得声称"

**限制**

1. 只有 **15/142** 个 shard 目录保留 work `labels.jsonl`；另外 127 个 shard 有 obs 但没有逐 plan `verified` 明细，无法离线重标（不猜、不补）。
2. **work JSONL 是这批观测上 `verified` 明细的唯一保留副本**；本工具只读它（记录 sha256），未改写、未复制。
3. 切片只覆盖 **1 个难度（difficulty=1）、2 个 stage（0/1）**；不是多难度/多面数据集。
4. stage 1 的 seeds 只有 2–8（缺 1），stage 0 是 1–8；两组 seeds 都不是 D410-S1 计划里的 1000–1019。
5. 观测是 pre-commit（`decision_time_before_commit`），work 行 `frame` 是 post-commit；对齐靠 `hash_pre` + `decision_frame`，标签行显式带 `obs_row`，下游不得用 `frame` 当 obs 行号。
6. 7 条 transition 尾行没有 obs，已裁剪并逐条记录；它们仍计入 work decisions（54,632），不计入对齐 decisions（54,625）。
7. `alive` 是 WP02.4 的 calibrated model-consistency 判定，不是生存保证；`unknown` 不是负样本。
8. 6 个 dead plan 只出现在一个 stage 1 shard；本切片对死亡分支的统计意义有限（与 WP02.4 的 0 dead 语料相比只多了 6 条）。

**不得声称**

- 不得写成 **D410-S1 通过**（stage 0 只有 26,625/50,000 decisions、8/20 seeds，`status=shortfall`）。
- 不得写成 **D410-MULTI 通过**（2/8 面、218,500/2,000,000 帧、每面均 <150,000）。
- 不得写成"新数据集/新采集"：没有启动游戏、没有训练、没有采集，标签来自既有 work JSONL 的离线重标。
- 不得把 `.tmp_v410_wp03_relabel/` 的标签当作已入库数据集：它没有复制 obs，且 `.tmp_*` 不入库。

---

## 9 复现与产物清单

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'

# 1. 单元测试（30 passed，exit 0）
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp031_pytest -q tests/test_v410_relabel_d1.py

# 2. 语法校验（exit 0）
python -m compileall -q tools/rl/v410_relabel_d1.py

# 3. 真实重标：15 个 shard（exit 0），写 manifest + S1 partial 审计
python tools/rl/v410_relabel_d1.py --d1-root .tmp_v400_d1/d1 --work-root .tmp_v400_d1/work --out .tmp_v410_wp03_relabel --manifest Document/Update/Alpha/v4.1.0/evidence/WP03/relabel_manifest.json --s1-audit Document/Update/Alpha/v4.1.0/evidence/WP03/s1_partial_audit.json
```

产物：

| 路径 | 说明 |
|---|---|
| `tools/rl/v410_relabel_d1.py` | `discover_shards` / `align_labels` / `relabel_shard` / `run_relabel` + CLI |
| `tests/test_v410_relabel_d1.py` | 30 例：裁剪、fail-closed、calibrated 映射、manifest/SHA、S1/MULTI shortfall、坏 JSON/空行、不复制 obs |
| `Document/Update/Alpha/v4.1.0/evidence/WP03/relabel_manifest.json` | schema `v410-relabel-manifest/1`，15 条 shard 记录 + totals + stages + S1/MULTI 摘要 |
| `Document/Update/Alpha/v4.1.0/evidence/WP03/s1_partial_audit.json` | schema `v410-s1-partial-audit/1`，stage 0 的 shortfall 审计 + `multi_partial` |
| `.tmp_v410_wp03_relabel/<shard>/labels_v410.jsonl` | 15 个标签文件（54,625 行 / 202,401,419 bytes），不入库、不含 obs |
