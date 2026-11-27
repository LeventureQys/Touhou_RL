# V831-ROADMAP-01 阶段回执 14：WP05 数据集建成——3,116 条学生态教师标签，一致率 10.6%

**工单**：[`LONG_TERM_DEVELOPMENT.md`](LONG_TERM_DEVELOPMENT.md)（V831-ROADMAP-01）· 承接 [`TRAINING_HOST_REPLY_13.md`](TRAINING_HOST_REPLY_13.md)
**主机代号**：轩神开播了（Windows 11 家庭版 / i5-14600KF；WSL2 `ubuntu`，20 vCPU，23 GiB，RTX 4070 SUPER）
**时间及时区**：2026-11-27 08:22 – 08:42 +08:00
**执行**：DSH 主代理派发 2 个执行代理（joiner、WP05 采集+重标注），主代理独立复跑两份验收（48 / 25 项）
**检出**：`D:\WorkShop\Touhou_RL`，HEAD `1233598796b1b6f1d734950a4113af917da7b844`（分支 `v400`）
**引擎启动**：**12 次**（6 采集 + 6 重标注）

---

## 0 结论摘要（五条）

1. **WP05 的训练数据建成**：6 个**未达成 Tier A** 的配置上采了 6 套学生 rollout 并全部重标注，得到
   **3,116 条"学生状态 + 教师标签"的行**（`labels.jsonl`），0 次失败、0 个非有限分数、`chosen` 100% 落在 `verified_mask` 内。
2. **教师与学生一致率只有 10.62%**（331/3,116）；逐配置 0.070 / 0.078 / 0.091 / 0.099 / 0.134 / 0.179。
   教师 `chosen` 高度集中在 **`cand 3`（63.1%，1,966/3,116）**，用到 16/18 个候选。
   → 学生自己走出来的状态上，教师几乎总是想做别的事；这批标签的纠正信息量很高。
3. **入口状态与 checkpoint 无关**：`b1` 与 `b2` 在 (seed 11, 入口 2,632) 上得到同一个 `entry_hash` `9edbdd05…`；
   6 个配置的入口三重证明（collect `entry_verified` / relabel `meta.entry.verified` / `labels row0.hash_pre`）**6/6 成立**。
4. **joiner 落地**：`v8_32_join.py` 2,130 行 + 25 项测试，把 T04 基线（19,772 行）与 dagger 标签合并成
   **下游 `s2e_train.merge_shards` 能直接读**的语料（实测 19,772 行全断言通过）。
   一个已记录的限制：`readout01.gate_corpus` 是单 episode 门，**构造上**不可能接受多 episode 合并语料（诊断已写进 meta）。
5. **额度**：12 次启动 / 44,336 引擎帧 / 400.1 s 墙钟 / 179.9 MB（WP05 部分），静音 24/24 读数全 `0 0 0 0`，账本 24 行无未结。

---

## 1 身份

| 项 | 值 |
| --- | --- |
| 引擎 | `rl_v3/th08-modern` sha256 `691bfd77…a568ef6`（12 次启动全部一致） |
| joiner | `tools/rl/v8_32_join.py` **2,130 行**，sha256 `8b6ea1382dff0d05…` |
| joiner 测试 | `tests/test_v8_32_join.py` **887 行**，sha256 `6349151732bcecc8…`（**25 项**） |
| 重标注工具 | `tools/rl/v8_32_relabel.py` 2,037 行，sha256 `347b24ba…` |
| checkpoint | `ckpt_b1.pt` `9548e3a8…` / `ckpt_b2.pt` `dcaad643…` |
| 磁带 | `.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed{10,11,12}-shard/` |

---

## 2 命令与退出码

| # | 命令 | rc | 结果 |
| ---: | --- | ---: | --- |
| 1 | `python .tmp_v832_wp05/verify_wp05.py`（主代理复跑，WSL） | **0** | **48/48 PASS** |
| 2 | `pytest tests\test_v8_32_join.py`（主代理复跑） | **0** | **25 passed in 3.39 s** |

**6 个配置（全部未达成 Tier A）**

| # | ckpt | seed | 入口 | 采集帧（前缀+接管） | 决策/obs/标注行 | 一致率 | 采集 s | 标注 s |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | b2 | 11 | 2,632 | 3,452（2,632+820） | 410 | 0.134 | 34.3 | 21.7 |
| 2 | b2 | 12 | 2,632 | 3,568（2,632+936） | 468 | 0.179 | 36.4 | 23.6 |
| 3 | b2 | 10 | 2,704 | 4,062（2,704+1,358） | 679 | 0.099 | 42.9 | 40.3 |
| 4 | b2 | 11 | 2,704 | 4,006（2,704+1,302） | 651 | 0.078 | 41.5 | 40.0 |
| 5 | b1 | 10 | 2,632 | 3,616（2,632+984） | 492 | 0.091 | 37.5 | 25.2 |
| 6 | b1 | 11 | 2,632 | 3,464（2,632+832） | 416 | 0.070 | 34.5 | 22.3 |

合计标注 **3,116 行**，一致 **331 行（0.1062）**。

---

## 3 额度

| 额度 | 实际 |
| --- | ---: |
| 引擎启动 | **12**（6 采集 + 6 重标注，全部 rc=0，无重试） |
| 引擎帧 | **44,336**（采集 22,168 + 标注 22,168；学生驱动 12,464） |
| 墙钟 | **400.1 s**（采集 227.1 + 标注 173.0） |
| 静音 | **24/24** 四点读数全 `0 0 0 0`；源树三处指纹前后相同 |
| 新增磁盘 | WP05 **179.9 MB**（obs 166.9 + labels 11.0）；joiner 目录 4.29 GB（含 3 份同 sha 中间语料，见 §6-3） |
| 账本 | 24 行 = 12 reserve + 12 settle，无未结 |

---

## 4 教师标签的形态

| 量 | 值 |
| --- | ---: |
| 合并一致率 | **331/3,116 = 0.1062** |
| 逐配置一致率 | 0.070 / 0.078 / 0.091 / 0.099 / 0.134 / 0.179（中位 0.095） |
| `chosen` 分布（合并） | **`cand 3` 1,966（63.1%）**、2:360、7:295、0:293、4:52、6:49、1:32、11:25、12:22、其余 <10 |
| 非有限分数 | 0 行 / 0 槽 |
| `chosen` 落在 `verified_mask` 内 | 100% |

→ 教师在这批状态上高度偏向同一个动作（cand 3），而学生很少选它；这正是"学生分布偏移"的具体形状。
训练时要注意类不平衡（cand 3 占 63%）。

---

## 5 joiner 与下游

`v8_32_join.py build --baseline <T04 片>×3 --dagger <labels.jsonl>×6 --out <dir>` 输出：

| 项 | 说明 |
| --- | --- |
| 行字段 | `provenance`(t04\|dagger)、重编号的 `slot_id`/`shard_key`、`episode_seed`/`decision_frame`、`history`(8)、`executed_cand`/`executed_bomb`/**`mask`（执行 mask）**、**`bomb`（标签通道）+`bomb_source`**、`scores`(18)/`verified_mask`(18)/`verified_count`、`chosen`/`chosen_verified`/`chosen_score`/`margin`、`label_source`/`label_verified`、`hash_pre`/`hash_post`、`obs{play,hud,act_hist,row,source_row}`、`source{side,ordinal,...}`、`provenance_source`、`relabel` |
| 关键语义 | 顶层 `bomb` 是**标签通道**（基线用其 `bomb`、dagger 用 `teacher_bomb`）；学生自己的 bomb 在 `executed_bomb`；`mask` 只与 `(executed_cand, executed_bomb)` 往返 |
| 下游实测 | `readout01.Corpus → s2e_train.merge_shards → s2e_train.view_corpus → dataset_2f.check_targets_record`，19,772 行全过 |
| 构造性限制 | `readout01.gate_corpus` 是单 episode 门（`decision_frame_rule_breaks`、`hash_chain_breaks`），**多 episode 合并语料不可能通过**；已写入 `meta.downstream.readout01_gate_corpus` |

---

## 6 偏差与异常

1. **6 次采集全部 `legal_death`**：学生都在 cap 内先死，决策数 410–679（合计 3,116），低于按 cap 估算的约 5,500。
   这是学生真实行为，不是失败；标注行数因此等于各自 obs 行数。
2. **磁盘自检曾误报一次**：driver 把私有目录里指向游戏树的软链（th08.dat + thbgm.dat，共约 500 MB）算进"已写字节"，
   在第 2 次启动后误触停止；已改为跳过 symlink 并**按 reserve 续跑**，没有重复启动任何 launch（总数仍 12）。
3. **joiner 目录有 3 份同 sha 的中间语料**（`real3` / `real3b` / `corpus_t04_3shard`），使 `.tmp_v832_join` 达 4.29 GB。
   主代理已确认其中一份 `corpus/` 是权威产物；**清理另外三份属于删除操作，下一轮由主代理先解析绝对路径再执行并登记**，本轮不动。
4. **`agree` 的口径**：`executed_cand == chosen`（学生实际执行的动作是否等于教师首选）。它不等于"教师会对"，只是"教师想做别的"。

---

## 7 分类声明

- **训练端本次实测（引擎）**：§2 的 12 次启动、3,116 条标签、静音 24/24、账本。
- **训练端本次离线**：25 项 joiner 测试、48 项 WP05 验收、joiner 的真实数据构建与下游加载断言。
- **历史记录**：Tier A 2/12、恒动作 0% 伤害（回执 10/11）。
- **待核验**：这批标签训练后的 Tier A 变化（下一轮）；cand 3 类不平衡的影响。

---

## 8 本回执**不**主张

1. 不主张这批标签能提升 Tier A：数据建成了，但**还没训练、没评测**。
2. 不主张 10.62% 是总体一致率：它只覆盖 6 个配置的 3,116 个状态。
3. 不主张 joiner 的输出可直接用于生产训练：只验证了 `s2e_train` 路径能读，未做完整训练跑。
4. 不主张 `readout01.gate_corpus` 需要修改：它对单 episode 是对的，多 episode 需要另一条路径。

---

## 9 下一项（R12）

| # | 任务 | 代价 |
| ---: | --- | --- |
| 1 | **合并语料**：T04 三片（19,772）+ 6 套 dagger（3,116）= 22,888 行 | 0 次引擎 |
| 2 | **重训**（2,000 / 6,000 步）并与同预算的纯 T04 基线对照 | 约 2–3 min GPU |
| 3 | **复测 12 个 (ckpt, root, entry) 配置**，看 Tier A 能否从 2/12 上升 | 12 次启动 |
| 4 | 清理 joiner 的三份同 sha 中间语料（先解析路径再删） | 0 次引擎 |

---

## 10 证据索引

| 结论 | 路径 | 身份 |
| --- | --- | --- |
| WP05 汇总 | `.tmp_v832_wp05/results.json` / `wp05.md` / `verify_wp05.py` / `ledger.jsonl` | `857dd237…` / `0cb8596d…` / `2ea1ac40…`（48/48）/ `ae162ecb…` |
| 标签 6 套 | `.tmp_v832_wp05/labels/<name>/labels.jsonl` | 合计 3,116 行 |
| 观测 6 套 | `.tmp_v832_wp05/obs/<name>/{slots.jsonl,obs/*.npy}` | 53.5 KB/决策 |
| joiner | `tools/rl/v8_32_join.py` / `tests/test_v8_32_join.py` | `8b6ea138…`（2,130 行）/ `63491517…`（887 行，25 项） |
| joiner 真实构建 | `.tmp_v832_join/corpus/`（19,772 行） | `slots.jsonl` `8069c75d…`、`play` `6d2f954e…` |
