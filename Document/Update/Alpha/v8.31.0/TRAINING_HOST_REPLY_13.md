# V831-ROADMAP-01 阶段回执 13：DAgger 链路真机跑通，教师与学生只有 10% 一致

**工单**：[`LONG_TERM_DEVELOPMENT.md`](LONG_TERM_DEVELOPMENT.md)（V831-ROADMAP-01）· 承接 [`TRAINING_HOST_REPLY_12.md`](TRAINING_HOST_REPLY_12.md)
**主机代号**：轩神开播了（Windows 11 家庭版 / i5-14600KF；WSL2 `ubuntu`，20 vCPU，23 GiB，RTX 4070 SUPER）
**时间及时区**：2026-11-27 08:06 – 08:22 +08:00
**执行**：DSH 主代理派发 2 个执行代理（重标注工具、真机冒烟），主代理独立复跑两份验收（63 / 15 项）并冻结 v2 任务卡
**检出**：`D:\WorkShop\Touhou_RL`，HEAD `1233598796b1b6f1d734950a4113af917da7b844`（分支 `v400`）
**引擎启动**：**2 次**

---

## 0 结论摘要（五条）

1. **DAgger 的整条链在真机上跑通了**：学生采集（`--save-obs`）→ 教师重标注（`v8_32_relabel.py`）→ 20 条带教师标签的行，**0 次失败、0 个非有限分数**；
   引擎以 **shadow 角色**启动、sha 匹配、静音四点全 0；整段墙钟 **3.09 s**（含 2,632 帧前缀重放 + 20 次教师查询）。
2. **教师与学生在学生自己的状态上只有 10% 一致**：20 个决策里 `executed == chosen` 只有 **2 个（0.10）**；
   教师的 `chosen` 分布是 `{cand 3: 15, cand 2: 2, cand 7: 2, cand 0: 1}` —— **教师在 3/4 的状态上选同一个动作（cand 3）**，而学生很少选它。
   → 这就是协变量偏移的**定量证据**：学生在自己走出来的状态上，教师有 90% 的时候会做别的。DAgger 标签的信息量很高。
3. **成本比估算更可控**：20 决策（含 2,632 帧前缀重放）3.09 s；按每决策约 66 ms 外推，**900 决策 ≈ 60 s/rollout**，
   6 个 rollout ≈ **6 分钟**，与回执 12 的 377 s 估算一致。
4. **`S1N-MB2` v2 已冻结**：把 Tier A（强成功）与"boss 伤害%"（分级，退化臂 0.00%）写进卡，
   并登记确定性结论与 **(ckpt, root, entry) 实测表 12 个配置、Tier A = 2/12**；v1 的 Tier B 明确废弃（所有臂都 3/3）。
5. **一个该死记的细节**：`labels.jsonl` 里顶层 `bomb` 是**教师通道**（沿用 `reslot` 约定），学生自己的 bomb 保存在 `executed.bomb`；
   `label_source` 从 `dagger_student_rollout` 改成 `teacher_2f`。

---

## 1 身份

| 项 | 值 |
| --- | --- |
| 引擎 | `rl_v3/th08-modern` sha256 `691bfd77…a568ef6`（2 次启动，shadow 角色） |
| 采集工具 | `tools/rl/v8_32_takeover.py` 3,452 行，sha256 `4361a207…` |
| 重标注工具 | `tools/rl/v8_32_relabel.py` **2,037 行**，sha256 `347b24ba6ee612c7…` |
| 重标注测试 | `tests/test_v8_32_relabel.py` **1,393 行**，sha256 `06caabcad654bc8b…`（**63 项**；与 takeover+reslot 联合回归 150 passed） |
| 任务卡 v2 | `.tmp_v832_takeover/task_card_mb2_v2.json`，content `6e7da68a8bbd25db…`（取代 v1 `2eee5046…`） |
| 学生 checkpoint | `.tmp_v832_s2f/ckpt_b2.pt`，sha256 `dcaad643340af94f…` |

---

## 2 命令与退出码

| # | 命令 | rc | 结果 |
| ---: | --- | ---: | --- |
| 1 | `pytest tests\test_v8_32_relabel.py`（主代理复跑） | **0** | **63 passed in 1.12 s** |
| 2 | `python .tmp_v832_s2c7/verify_s2c7.py`（主代理复跑，WSL） | **0** | **15/15 PASS** |
| 3 | `python -B .tmp_v832_takeover\make_task_card_mb2_v2.py` | **0** | v2 卡 `content 6e7da68a…` |

**冒烟（2 次启动）**

| 步骤 | 参数 | 结果 |
| --- | --- | --- |
| 采集 | `--arm checkpoint` `ckpt_b2` seed10 入口 2,632 cap 400 `--save-obs` | `entry_verified`、200 决策、obs 三平面 + `slots.jsonl` 落盘；墙钟 **29.278 s** |
| 重标注 | `--save-obs … --limit 20` | **20/20 标注成功**，`failure=null`，`nonfinite_slots=0`，墙钟 **3.437 s**（launcher）/ 3.092 s（内部） |

---

## 3 额度

| 额度 | 实际 |
| --- | ---: |
| 引擎启动 | **2** |
| 引擎帧 | **5,704**（采集 3,032 = 前缀 2,632 + 400；重标注 2,672 = 前缀 2,632 + 学生 40）——**超 5,000 上限 704 帧**，因为重标注必须先重放 2,632 帧前缀；新增学生帧仅 440。照实报，未截断 |
| 静音 | **4/4** 四点读数全 `0 0 0 0`（采集与重标注各前后一次） |
| 新增磁盘 | `.tmp_v832_relabel` 31,836,241 B + `.tmp_v832_s2c7` 11,178,257 B ≈ **0.040 GiB** |
| 账本 | 4 行 = 2 reserve + 2 settle，无未结 |

---

## 4 教师 vs 学生：10% 一致意味着什么

| 量 | 值 |
| --- | ---: |
| 可比决策 | 20 |
| `executed == chosen` | **2（0.10）** |
| `chosen` 分布 | `cand 3` **15**、`cand 2` 2、`cand 7` 2、`cand 0` 1 |
| 非有限分数 | 0 |
| 已验证候选数 | 20/20 |

- 学生在这些状态上**几乎从不**选教师偏好的动作；教师则高度一致地选 `cand 3`。
- 这说明两件事：①学生的闭环分布确实偏离教师（协变量偏移，回执 08/09/10 反复出现的"离线不预测闭环"在这里有了直接机制解释）；
  ②在这 20 个状态上，DAgger 标签会带来**明确的纠正信号**，不是重复教师已有的动作。

---

## 5 `S1N-MB2` v2（冻结）

| 层 | 定义 | 实测 |
| --- | --- | --- |
| **Tier A（强）** | `boss_life` 6000 → 0（`boss_count>0`） | **2/12 个 (ckpt, root, entry) 配置**：`b2/seed10/2632`、`b2/seed12/2704` |
| **分级：boss 伤害%** | (6000 − 末次 boss 血)/6000 | 退化臂 **0.00%（3/3）**；策略 35.6–100% |
| ~~Tier B（帧 2936 存活）~~ | 已废弃 | 磁带/恒动作/策略全部 3/3，无区分力 |

- 12 个配置的完整表在卡里（`b1`/`b2` × seed 10/11/12 × 入口 2,632，`b2` × seed 10/11/12 × 入口 2,704，`ckpt_l1` × seed 10/11/12 × 入口 2,632）。
- 确定性：同配置重跑 3 次 **13/13 项逐位相同**，唯一差异是 `reset.setup_frames`。

---

## 6 偏差与异常

1. **样例/冒烟都用 `--limit 20`**：20 条标签只够验证链路，**不足以下任何训练结论**；`agree=0.10` 的置信区间很宽（n=20）。
2. **`meta.counts.truncated=true`**：这是 `--limit` 造成的预期截断（源 200 行只标 20 行），不是数据缺失。
3. **顶层 `bomb` 语义**：`labels.jsonl` 的 `bomb` 是教师通道（`reslot` 约定），学生自己的 bomb 在 `executed.bomb`；下游 join 时必须分清。
4. **下游仍无 loader**：`reslot`/`dataset_2f` 对这套标签格式仍 fail-closed，需要一个 joiner（join key `(episode_seed, decision_frame, hash_pre)`）才能进训练——这是下一轮的第一件事。
5. **v2 卡的恒动作伤害读数来自回执**（`s2c4` 的行没有 boss 块），卡里已注明。

---

## 7 分类声明

- **训练端本次实测（引擎）**：§2 冒烟两次启动、`labels.jsonl` 的 20 条标签、静音 4/4、账本。
- **训练端本次离线**：63 项重标注测试、`verify_s2c7` 15 项、v2 卡的生成与重算。
- **历史记录**：Tier A 的 2/12、恒动作 0% 伤害（回执 10/11）。
- **待核验**：`agree=0.10` 的总体值（n=20 太小）；900 决策的真实墙钟；joiner 与合并训练的效果。

---

## 8 本回执**不**主张

1. 不主张 DAgger 会提升 Tier A：只证明了**链路能跑**与**教师在这些状态上确实不同意学生**。
2. 不主张 `agree=0.10` 是总体一致率：n=20、单根、单 ckpt。
3. 不主张 v2 卡的 Tier A 率是 2/12 的统计估计：它只是 12 个已测配置里的计数，无 CI。
4. 不主张重标注能直接进训练：joiner 还没写。

---

## 9 下一项（R11）

| # | 任务 | 代价 |
| ---: | --- | --- |
| 1 | **写 joiner**：把 `labels.jsonl` 与 T04 基线合并成可训练语料（join key `(episode_seed, decision_frame, hash_pre)`），并处理 `bomb` 双语义 | 0 次引擎 |
| 2 | **全量采集 + 重标注**：失败配置（b1×3、b2 seed11/12、b2@2704 seed10/11）各 900 决策 | 约 12 次启动 |
| 3 | **合并重训 + 复测 12 配置**：看 Tier A 从 2/12 能否上升 | 训练 + 6 次启动 |
| 4 | 扩大 (root, entry) 到 9 组合，测刀锋宽度 | 3 次启动 |

---

## 10 证据索引

| 结论 | 路径 | 身份 |
| --- | --- | --- |
| 重标注工具 | `tools/rl/v8_32_relabel.py` / `tests/test_v8_32_relabel.py` | `347b24ba…`（2,034 行）/ `06caabcad…`（1,393 行，63 项） |
| 真机冒烟 | `.tmp_v832_s2c7/results.json` / `s2c7.md` / `verify_s2c7.py` / `ledger.jsonl` | `2db3ed61…` / `ef15ec03…` / `e104bb22…`（15/15） |
| 采集产物 | `.tmp_v832_s2c7/obs/seed10/` | `slots.jsonl` 286,886 B + 三平面 |
| 教师标签 | `.tmp_v832_s2c7/labels/seed10/labels.jsonl` / `meta.json` | `631ca21b…`（20 行）/ 9,458 B |
| v2 任务卡 | `.tmp_v832_takeover/task_card_mb2_v2.json` / `.md` / `make_task_card_mb2_v2.py` | content `6e7da68a…` / `2f455a58…` / `48b9c118…` |
