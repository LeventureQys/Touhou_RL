# V831-ROADMAP-01 阶段回执 12：WP05 标签链路可行性——需最小适配，成本仅 6.3 分钟

**工单**：[`LONG_TERM_DEVELOPMENT.md`](LONG_TERM_DEVELOPMENT.md)（V831-ROADMAP-01）· 承接 [`TRAINING_HOST_REPLY_11.md`](TRAINING_HOST_REPLY_11.md)
**主机代号**：轩神开播了（Windows 11 家庭版 / i5-14600KF；WSL2 `ubuntu`，20 vCPU，23 GiB，RTX 4070 SUPER）
**时间及时区**：2026-11-27 07:58 – 08:05 +08:00
**执行**：DSH 主代理派发 1 个执行代理做可行性判定（只读分析 + 实际调用），主代理独立复跑其验收（10 项）
**检出**：`D:\WorkShop\Touhou_RL`，HEAD `1233598796b1b6f1d734950a4113af917da7b844`（分支 `v400`）
**引擎启动**：**0 次**（本回执全部为只读分析与离线调用）

---

## 0 结论摘要（四条）

1. **DAgger 的标签链路可行，但需要最小适配：写一个小重标注工具（约 120 行 + 20 行前缀胶水），不要试图把学生行伪装成 legacy shard。**
   两条既有加载路径都 **fail-closed**（异常原文见 §3），缺的是 shard 级三件套（`shard.json`/`labels.jsonl`/`targets.jsonl`）与 15 个行级字段；
   而"伪装"方案要伪造 4 帧 `step_frames`、T04 `slot_id`、`executed.source="step_frames"` —— 那是对学生 rollout 的**不实陈述**，被否掉。
2. **状态可重建性已验到锚点级。** 学生的每个决策状态 = `RESET seed` + mask 前缀；
   磁带 `lead_in` 恰好 16 帧 + 磁带 slot mask 拼出 0..entry，且在入口帧与学生行 0 的 `hash_pre`/`hash_post` **逐位相同**。
   **entry 之后磁带 mask 与学生 mask 有 84/96 行不同**（65 vs 33）——所以后段必须用学生自己的 mask，两段在入口由 HASH 对齐。
3. **教师查询 API 已定位**：`touhou_rl/v400/teacher/decide.py:129 decide(engine, config, *, committed, extra_actions, frame, stage, difficulty, consistency, bombs, include_items, record_source, shoot_release)`，
   内部 18 路打分 `search.beam_search`、逐候选验证 `verify.verify_plans`；需要**活着的 shadow 进程**与 `PARK`/`SPAWN`/`CHILD` 三个命令，**没有 snapshot/restore ABI**。
   关键性质：`decide(record_source="child")` **不推进主进程**（`reslot.py:1482` 有断言）→ 一次 RESET + 一次前缀重放后可以"边重放边逐决策查询"。
4. **成本极低**：6 个 rollout × 约 900 决策 = **5,400 次查询**，只需 **6 次引擎启动 / 6 次 RESET**，
   墙钟 **≈377 s（6.3 min）**：教师 359.5 s + 前缀重放 1.9 s + 启动/RESET 12.5 s + 协议 2.8 s。
   对照"每状态 RESET + 重放"的写法是 **≈4,600 s（77 min）**——**约 13 倍贵**，所以要走长会话。

---

## 1 身份

| 项 | 值 |
| --- | --- |
| 分析产物 | `.tmp_v832_dagger01/feasibility.json`（17,976 B）/ `feasibility.md`（8,232 B，80 行）/ `probe_feasibility.py`（22,521 B）/ `verify_feasibility.py`（17,406 B） |
| 样例语料 | `.tmp_v832_takeover/scale/`（96 决策，`--save-obs` 的桩 client 产物） |
| 教师入口 | `touhou_rl/v400/teacher/decide.py:129`、`search.beam_search`、`verify.verify_plans(engine, tape, plans, *, config, tolerance=0.001, template=None, batch=True)` |
| 重放入口 | `touhou_rl/v420/counterfactual.py` 的 `Prefix`(:245) 与 `_Session`（reset:2968 / probe_once:2982 / replay:3001 / decide:3071） |
| 磁带前缀工具 | `v8_32_takeover.lead_in_masks`(:988) / `slot_masks`(:968) |

---

## 2 命令与退出码

| # | 命令 | rc | 结果 |
| ---: | --- | ---: | --- |
| 1 | `python -B .tmp_v832_dagger01\verify_feasibility.py`（主代理复跑） | **0** | **10/10 PASS** |

10 项检查：平面形状、8 条行级拒绝原文、`reslot.load_shard` 三级失败原文、`dataset_2f`/`counterfactual` 失败原文、15 个缺失键、96/96 掩码算术、前缀 64 帧 + entry HASH + 84/96 mask 差异、成本算术、AST 扫描无引擎构造、引擎模块不在 import 图。

---

## 3 兼容性实测（异常原文，全部 fail-closed）

| 路径 | 结果 |
| --- | --- |
| `dataset_2f.open_corpus_planes(scale/)` | **通过**：`play[96,4,112,96]` / `hud[96,56,160]` / `act_hist[96,8,18]` uint8 全对 |
| `dataset_2f.load_dataset_shard` | `Dataset2FError: …\corpus96\targets.jsonl does not exist` |
| `counterfactual.read_shard` | `CounterfactualError: shard.json does not exist: …` |
| `reslot.load_shard`（先补 `shard.json`、再改 `frames_per_decision`、再补 `labels.jsonl`） | 三级：`shard.json does not exist` → `frames_per_decision 2 is not 4` → `row 0: step_frames must be 4 masks, got None` |
| `reslot.check_label_record(行 0)` | 8 条拒绝：`slot_id None`、`emitted None`、`duration None`、`scores must be 18 numbers`（×2）、`verified_mask must be 18 flags`、`verified_count None != the mask's 0`、`executed.source 'policy_act' is not 'step_frames'`、`does not carry a two-frame teacher profile` |
| **已兼容（96/96）** | `cand_to_mask(cand,bomb)==mask`、`commit_frames==decision_span(d)`、`obs_render_frames==obs_window(d)`、`history_frames==history_window(d)`、`len(history)==8`、哈希链无断点 |

→ **坏的只是 shard 形状与标签字段，观测与帧尺是干净的。**

---

## 4 适配方案（推荐 b）

**方案 b（推荐）**：约 120 行的小工具 + 20 行前缀胶水。
`tape prefix → _Session.replay（逐槽 HASH 校验）→ 每决策 step_frame×2 + hash_post 断言 → decide(record_source="child") → reslot.decision_view(...) → 写 scores/verified_mask/verified_count/chosen/chosen_verified/margin/bomb/label_source="teacher_2f" → reslot.check_label_record 当门`。

**方案 a（否掉）**：把学生行补成 legacy shard 形状，需要 4 帧 `step_frames`、frame0 链、T04 `slot_id`、`executed.source="step_frames"`——2 帧语料没有 4 帧前驱，只能伪造。

---

## 5 成本

| 口径 | 数值 |
| --- | ---: |
| 查询次数 | **5,400**（6 rollout × 900） |
| 内部帧数 | 5,400 × 64 = 345,600 |
| 引擎启动 | **6**（每 rollout 一个长会话；不是每状态一个子进程） |
| 墙钟 | **≈377 s（6.3 min）** |
| 对照（每状态 RESET+重放） | ≈4,600 s（77 min），约 **13×** |

---

## 6 风险与待核验

1. **样例是桩 client 产出**（checkpoint 23 B / 1,234 参数、HASH 是决策计数器）：HASH 校验的"形状"已验证，**鉴别力未验**，真机需再确认一次。
2. **前缀重放必须用 `TeacherEngine(role="shadow")`**，不能复用 runner 的 env client（角色不同）。
3. **磁带是 entry 之前 mask 的唯一来源**：`--save-obs` 若脱离磁带单独归档，前缀就无法重建——采集时必须同时登记磁带身份。
4. 本轮**未做**卡 v2（`S1N-MB2` 的分级口径重冻），顺延到下一轮。

---

## 7 分类声明

- **训练端本次离线（含实际调用）**：§3 的全部异常原文、§4 的方案与代码行号、§5 的成本算术。
- **训练端本次实测**：无引擎启动；96 决策样例来自此前的桩 client 产物。
- **待核验**：真机 HASH 鉴别力、shadow 角色下的重放、教师查询的真实耗时（377 s 是按历史 66.58 ms/query 折算）。

---

## 8 本回执**不**主张

1. 不主张 DAgger 已经能跑：只证明**链路可行 + 适配方案 + 成本量级**，工具还没写。
2. 不主张 377 s 是真实墙钟：它是按历史单查询成本折算的估算。
3. 不主张 `--save-obs` 的语料可用于训练：它还没有教师标签，且样例是桩 client 的。
4. 不主张卡 v2 已完成：本轮没做。

---

## 9 下一项（R10）

| # | 任务 | 代价 |
| ---: | --- | --- |
| 1 | 写 `tools/rl/v8_32_relabel.py`（方案 b，约 140 行）+ 单测 | 0 次引擎 |
| 2 | 真机冒烟：单根 100 决策的"采 → 标"闭环，验证 shadow 重放与 HASH 鉴别力 | 2 次启动 |
| 3 | 卡 v2 冻结（Tier A 强 + 血量削减% 分级 + 确定性 + 2/6） | 0 次引擎 |
| 4 | 扩大 (root, entry) 到 9 组合，测刀锋宽度 | 3 次启动 |
| 5 | WP05 最小采集（失败配置约 193 MB）+ 重标注 + 重训 + 复测 | 约 12 次启动 |

---

## 10 证据索引

| 结论 | 路径 | 身份 |
| --- | --- | --- |
| 可行性判定 | `.tmp_v832_dagger01/feasibility.json` / `feasibility.md` / `verify_feasibility.py` / `probe_feasibility.py` | `f8164e18…` / `a14b40b9…` / `abba954f…` / `5a397fbe…`（10/10） |
| 样例语料 | `.tmp_v832_takeover/scale/` | 96 决策、53.5 KB/决策 |
| 教师 API | `touhou_rl/v400/teacher/decide.py:129` | `decide(...)` + `beam_search` + `verify_plans` |
| 重放/会话 | `touhou_rl/v420/counterfactual.py` | `Prefix`:245、`_Session` reset:2968 / replay:3001 / decide:3071 |
