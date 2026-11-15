# T05 证据：WP420-04 第三轮（D3，β=0）DAgger 收尾轮

> 机器可读镜像：[`dagger_d3.json`](dagger_d3.json)。本文件由 `.tmp_v420_t05d3/write_report.py` 从该 JSON 渲染，JSON 由 `.tmp_v420_t05d3/summarise.py` 从**运行产物**生成（合并后的采集报告、逐命令 `D3_segments.jsonl`、独立复读 `verify_d3.json`、`analysis_d3.json`、`aggregate_d3.json`、`train_d3.json`、二十份 evaluator summary，以及只读的 D1/D2/T04 既有证据），不手写、不四舍五入后再抄。

> **本轮是三轮配方的最后一轮**：β=0 自主决策累计到 **51,671**，β=0 轮次 **2/2**。**G420-3 仍为 `not_passed`** —— 决策量两项到位，起点类别与整面过关两项没有到位，见 §8。两套 20 局仍然 **0/20 过关**。

---

## 0 结论摘要

| 问题 | 回答 | 依据 |
|---|---|---|
| 采集是否跑通（env + shadow 教师 + 双引擎 HASH 对齐） | **是**：27 个 episode、27,000 决策、54,432 帧、27,000 次教师查询，**108,432 个 HASH 校验 0 不一致** | §2 |
| β=0 下 `injection_rate` 是否真的为 0 | **是**：注入 0/27,000，`injection_rate = 0.0`，`label_only.passed = true` | §2 |
| 学生与教师是否真的分歧 | 是：agreement 7,807/27,000 = **28.91%**，即标签纠正率 **71.09%** | §2、§4 |
| 有效祖先数 | **27** 个 train seed （3042–3068），与 D1（3000–3004）、D2（3010–3029）**互不相交** | §4 |
| 聚合是否按 50/30/20 | **是**：27,000 / 16,200 / 10,800 = 0.500000 / 0.300000 / 0.200000；**三源首次真正独立**，bootstrap 未启用 | §5 |
| 新 checkpoint 能否被冻结 evaluator 加载 | **能**：`load_back.loaded = true`，731,802/731,802 参数，重载后 logits **max abs diff = 0.0** | §6 |
| 两种口径的 20 局 | 标准 S1N mean **3,025.20**、DIAG 首死 mean **1,716.45**，两套都 **0/20 与 0/20 过关** | §7 |
| 相对 A0 的配对差 | DIAG **+79.65 帧**（+4.87%）、S1N **+13.70 帧**（+0.45%） | §7 |
| 累计 β=0 自主决策 | **51,671** = D2 24,671 + D3 27,000，门槛 50,000：**达到** | §8 |
| G420-3 | **not_passed**：决策量两项到位，起点类别、整面过关仍未到位 | §8 |
| 停止条件 | **触发**：三轮已满、整面过关 0/40、最好的 DIAG 首死相对 A0 改善 **16.21%**（阈值 30%） | §9 |

一句话：**D3 把 β=0 自主决策推到 51,671、把两轮 β=0 补齐、把 50/30/20 的三源第一次做成真正独立的三源，训练链路与两口径评测再次完整走通；但能力侧没有任何过关证据 —— 两套仍 0/20，DIAG 首死相对 A0 只有 4.87%。三轮配方到此结束，停止条件给出明确裁决（§9）。**

---

## 1 运行命令与退出码

全部在资产主机 WSL 内、`cwd=/mnt/d/WorkShop/Touhou_RL`、`PYTHONPATH=` 仓库根。

### 1.1 采集：逐 seed 一条命令（多数远低于 10 分钟上限；受并发的少数见 §1.3）

主轮计划 27,000 决策 = 27 个 episode × 1,000 决策，**由 27 条独立 `collect` 命令完成**，每条的 argv / 开始结束时间 / 退出码 / 墙钟逐条记录在 [`D3_segments.jsonl`](../../../../../.tmp_v420_data/T05/D3_segments.jsonl)。

| 项 | 值 |
|---|---:|
| 命令条数（= episode 数） | **27**（期望 27） |
| 退出码集合 | **[0]**（全 0：true） |
| 单条墙钟总和 | **11,308 s** |
| 单条墙钟最长 | **1,584 s**（计划上限 600 s；超出的原因见 §1.3） |
| 单条墙钟最短 | **126 s** |
| 平均每决策 | **0.419 s**（主机有其他任务在跑） |
| 墙钟 > 600 s 的命令数 | **4 / 27**（并发争用，见 §1.3） |

每条命令的形状（同一条 argv，只换 seed 与输出路径）：

```bash
xvfb-run -a -s "-screen 0 1024x768x24" \
  /root/v400-venv/bin/python tools/rl/v420.py collect \
  --config .tmp_v420_t05d3/collect-seed<SEED>.json \
  --out .tmp_v420_data/T05/D3seg/seed<SEED>.json --execute
```

### 1.2 逐条命令记录（全部 exit 0）

| seed | 墙钟 (s) | 退出码 | 决策 | 帧 | 注入 | 停止原因 |
|---|---:|---:|---:|---:|---:|---|
| seed3042 | 377 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3043 | 459 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3044 | 323 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3045 | 427 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3046 | 174 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3047 | 258 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3048 | 250 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3049 | 163 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3050 | 127 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3051 | 234 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3052 | 290 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3053 | 321 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3054 | 307 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3055 | 307 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3056 | 304 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3057 | 311 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3058 | 439 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3059 | 510 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3060 | 717 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3061 | 1,584 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3062 | 1,547 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3063 | 997 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3064 | 343 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3065 | 156 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3066 | 126 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3067 | 131 | 0 | 1,000 | 2,016 | 0 | `decisions` |
| seed3068 | 126 | 0 | 1,000 | 2,016 | 0 | `decisions` |

### 1.3 并发事实（墙钟是争用条件下的观测值，不是吞吐承诺）

本轮与另一个任务（T06 的引擎回报 RL 诊断）共用同一台机器。采集期间 T06 曾同时起
**6 个 env 会话**（`.tmp_v420_data/T06/engine-data-p3/{diag,diag2,s1n,s1n2}` 与 `.tmp_v420_data/T06/data/kl-eval-{diag,s1n}`），主机 **load average 一度到 80.06**（20 核；观测于 2026-11-13T19:36Z，用 `.tmp_v420_t05d3/host_state.sh` 记录，逐样本的 load 与活跃引擎数另外写在 `.tmp_v420_data/T05/host_load_d3.log`）。

- 于是**同一形状的命令**墙钟从 127 s 涨到 1,000 s 以上：本次最慢的一条是 **1,584 s**，最快 126 s，共 4 条超过 600 s。
- 本轮的应对是**串行 + 逐条记录 + 失败重试，从不杀别人的进程**；这些墙钟是**争用条件下的观测值**，不能当作这台机器上 DAgger 采集的吞吐。
- 采集的失败模式也没有被掩盖：任一条命令非零退出、`injection_rate != 0`、`hash_mismatches != 0` 或 epilogue 停止原因非法，驱动都会立刻停在原地（`collect_all.sh` 的 fail-closed 分支）；本轮没有触发。

### 1.4 聚合 / 训练 / 评测

| 步骤 | 命令 | 退出码 |
|---|---|---:|
| 合并逐命令报告 | `python .tmp_v420_t05d3/merge_round.py main` | **0** |
| 独立复读 shard | `python .tmp_v420_t05d3/verify.py $T05/D3.json --out $T05/verify_d3.json` | **0**（`ok = true`） |
| shard 分布分析 | `python .tmp_v420_t05d3/analyse.py $T05/D3 $T05/analysis_d3.json D3` | **0** |
| 生成运行时配置 | `python .tmp_v420_t05d3/make_configs.py --with-aggregate` | **0** |
| 聚合 50/30/20 | `python .tmp_v420_t05d3/aggregate.py` | **0** |
| 独立复读聚合 | `python .tmp_v420_t05d3/validate_aggregate_d3.py` | **0** |
| 训练 | `bash .tmp_v420_t05d3/train.sh` | **0** |
| DIAG 20 局 | `bash .tmp_v420_t05d3/run_all_evals.sh D3 diag`（4 条命令、每条 5 局） | **0**（4/4 batch） |
| 标准 S1N 20 局 | `bash .tmp_v420_t05d3/run_all_evals.sh D3 s1n`（4 条命令、每条 5 局） | **0**（4/4 batch） |

- **单条命令 ≤10 分钟**：两套评测共 8 条命令，DIAG 的 4 条墙钟 [136.761, 138.024, 133.619, 133.683]（合计 542.09 s，最长 138.02 s）、S1N 的 4 条 [207.105, 226.186, 216.146, 215.327]（合计 864.76 s，最长 226.19 s）；全部 batch 退出码 [0, 0, 0, 0] / [0, 0, 0, 0]。
- 采集每条命令最长 1,584 s：主机上同时有别的任务在跑 env 实例，本轮的策略是**串行 + 记录 + 失败重试**，从不杀别人的进程。

### 1.5 测试

| 侧 | 范围 | 结果 | 退出码 |
|---|---|---|---:|
| WSL | `tests/test_v420_dagger.py`（验收命令） | **38 passed** | 0 |
| WSL | 10 个 v420 测试文件（全量） | **378 passed** | 0 |
| Windows | `tests/test_v420_dagger.py`（`--basetemp .tmp_v420_pytest_win13`） | **38 passed** | 0 |
| Windows | 10 个 v420 测试文件（全量） | **377 passed + 1 skipped** | 0 |
| Windows | `python -m compileall -q touhou_rl\v420` | 无输出 | 0 |
| WSL | `tools/rl/v420.py --help` / `collect --help` / `train --help` | 各 0 | 0 |

测量时刻 2026-11-13T21:16Z（WSL）/ 2026-11-14T05:16+08（Windows），原始输出在 `.tmp_v420_data/T05/acceptance_d3.txt`。**基线读数要小心**：任务单里的 360 是本轮之前的快照，而 `tests/test_v420_{contracts,counterfactual,rl,train,...}.py` 在本轮运行期间被另一个并行任务持续追加（本轮没有改它们），所以「10 个文件」的总数由当前树决定、不由本轮决定；本轮只对 `tests/test_v420_dagger.py` 负责 —— 它在两侧都是 **38 passed**，其中既有 33 个未改、本轮追加 5 个。

`tests/test_v420_dagger.py` 本轮**只追加**了 5 个测试（§8 的三条要求各覆盖一次，外加算术可失败性与聚合短缺口声明），既有 33 个测试一个未改。

---

## 2 采集契约与实测

### 2.1 固定身份

| 角色 | 进程 | 身份 |
|---|---|---|
| env（学生闭环 + 像素） | `th08-modern --rl-role env` | 引擎 `691bfd77…` |
| shadow（教师标注） | `th08-modern --rl-role shadow` | 同上 |
| 学生 | **D2 的 checkpoint**（D1 训练后 → D2 训练后） | `/mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T05/checkpoints_d2/d2_seed0_step5000.pt`，sha256 `9f284e48…`, 731802 参数，经冻结 `tools/rl/v410_student_eval.py` 的 loader 与 `ProbePolicy` 载入 |
| 教师 | `fast` @ `replan_decisions=1`、`shoot_pulse_period=48` | horizon 32 / beam 24 / verify_k 1 / 1 iteration，每决策必查（27,000 次） |

**学生身份是本轮的关键改动**：D1 的 rollout 学生是 T04 的 A0，D2 的 rollout 学生是 D1 的产物，D3 的 rollout 学生是 **D2 的产物**（sha256 `9f284e48c2304a519308b947f3d9e6705a49d9404dbc3ee5b596c2a4ea3751b6`）。冻结 loader 按配置里声明的 digest 校验，换文件会被拒绝。

### 2.2 两帧节拍与冷启动

- 决策周期 2 帧；观测 = 4 帧栈 + HUD + 8 条**已执行**的 2 帧历史，与 `obs-v420-2f` 契约逐字段对齐。
- 决策帧 0–14 是冷启动：学生自己执行、不记录、不补零。因此记录的决策从 `decision_frame=16` 开始。
- 实测 54,432 帧 / 27,000 决策 = **2.016 帧每决策**（正好 2，未出现折叠）。

### 2.3 注入与一致率

| 项 | 实测 | 判定 |
|---|---:|---|
| 决策 | **27,000** | — |
| 注入 | **0** | — |
| `injection_rate` | **0.0** | β=0 必须为 0：**通过** |
| label-only 审计 | `passed = true`、`label_only = true` | 通过 |
| 教师可用 | 27,000 / 27,000 | 每决策都有标签 |
| agreement | **7,807**（28.91%） | 标签纠正率 **71.09%** |

`agreement` 的定义固定为「教师 `chosen.slots[0]` == 学生在同一决策点的 argmax」；β=0 时学生动作就是执行动作，所以**这条数据流的 100% 是学生自主访问的状态**，教师只写标签、从不替学生执行。

### 2.4 HASH 校验（fail-closed）

| 项 | 实测 |
|---|---:|
| 对齐边界总数 | **108,432**（每 episode 4,016） |
| 不一致 | **0** |
| 决策间 HASH 链断点 | **0** |
| 逐帧 HASH 不符 | **0** |
| 独立复读 `verify_d3.json` | `ok = true`，注入行 0，无标签行 0 |

复读脚本 `verify.py` 是**另写**的只读检查器：它重新打开每个 `slots.jsonl` 与 `meta.json`，重算 label-only 规则、HASH 链、身份字段与行数，不复用采集器的写路径。

---

## 3 起点类别（计划三类 vs 本轮实测）

| 起点类别 | 计划目标 | 本轮实测 | 说明 |
|---|---:|---:|---|
| 自然开局 `natural` | 50% | **27,000 / 100%** | 每个 episode 从 `RESET(seed)` 起，学生自己控制冷启动 |
| 教师中后段 `curriculum_teacher_mid` | 30% | **0** | **未实现**：需要分支/重放的课程起点，采集器**拒绝**而不是模拟 |
| 上一轮失败前状态 `prev_round_failure` | 20% | **0** | 同上；D2 已存在，但把它做成起点是新写分支/重放代码，不是少量适配 |

`start_kind_gap` 在 round 报告里显式记录这两类为 0 与原因；配置一旦请求这两类，`build_round_config` 直接 `DaggerError`（测试 `test_curriculum_start_is_refused_not_simulated` 覆盖）。**这是缺口，不是「三类都做了」。**

---

## 4 轨迹、祖先与动作分布

| 项 | 实测 |
|---|---:|
| episode seed（有效祖先） | **27**：3042–3068 |
| 每 episode 决策 | **1,000** |
| 唯一 `hash_pre` 总数 | **27,000 / 27,000**（唯一率 1.0000） |
| 每个决策索引上的不同状态数 | 27–27（均值 27，1000 个索引上 27 个 seed 全不同） |
| 教师 18 路 scores 跨 seed 相同 | 4910 / 26000（0.188846） |
| 学生 idle（cand 0）占比 | **0.3152** |
| 学生用到多少个候选 | **13 / 18** |
| 学生动作熵 | **2.7641 bits** |
| 首死帧（每 episode） | 中位 **1,588**、范围 1,088–1,970 |
| 有死亡的 episode | **23 / 27** |
| D1 窗口（2,016 帧）内死亡 | **28** |

与 D2 的分布对照（同一条分析脚本、同一份契约）：

| 项 | D2 | D3 | 差 |
|---|---:|---:|---:|
| idle 占比 | 0.3289 | 0.3152 | -0.0137 |
| 不同候选数 | 16 | 13 | — |
| 动作熵 (bits) | 2.7925 | 2.7641 | — |
| agreement | 0.2846 | 0.2891 | — |
| 首死均值（帧） | 1,747.80 | 1,665.30 | — |
| 祖先集合 | 20 个 | 27 个 | 交集 0 个 |

读法：D3 的 27 条轨迹与 D1（5 个）、D2（20 个）的 seed 集合**完全不相交**，所以这批状态是新的访问，不是把旧轨迹改个名字。agreement 从 D2 的 0.2846 变到 0.2891，idle 占比 0.3289 → 0.3152：**分布变了，但没有任何「变得更安全」的证据** —— 首死帧在这两轮之间几乎没动，能力结论只能来自 §7 的评测。

---

## 5 聚合：50 / 30 / 20（三源首次真正独立）

| 来源 | 行数 | 实际占比 | 目标占比 | 来源说明 |
|---|---:|---:|---:|---|
| `current_round` | **27,000** | 0.500000 | 0.50 | D3 本轮 27 个 shard 的全部行，一行不删 |
| `history_round` | **16,200** | 0.300000 | 0.30 | **D2 本轮采集的 shard**（24,671 槽可用） |
| `fixed_bc` | **10,800** | 0.200000 | 0.20 | 被消费的 `v420-2f-dataset/1` 语料，与 history 切片**不相交** |
| **合计** | **54,000** | 1.000000 | 1.00 | 43,200 train / 10,800 holdout |

**bootstrap 声明**：`applied = false`、`history_round_target = 16,200`、`history_round_from_round = 16,200`、**`history_round_topup_from_legacy = 0`**、`shortfall = 0`、`legacy_rows_used = 10,800`（语料共 19,772 槽）。

这就是 D3 相对 D1/D2 的数据侧实质变化：

- D1：没有更早的轮，「最近历史轮」30% **全部**由语料 bootstrap；
- D2：D1 只提供 5,000 槽，30% 里 9,803 行仍需语料 top-up；
- **D3：D2 的 24,671 槽足以付清 30% 的 16,200 槽，top-up 为 0，bootstrap 未启用** —— 三份来源第一次真的是三份。

独立复读（`validate_aggregate_d3.py`，另写的只读检查）：`ok = true`、`failed = []`。

| 检查 | 结果 |
|---|---|
| `slots_rows` | 通过 |
| `rows_by_source` | 通过 |
| `shares_match_reported` | 通过 |
| `history_fixed_disjoint` | 通过 |
| `current_history_disjoint` | 通过 |
| `no_duplicate_origin_within_source` | 通过 |
| `labels_ok` | 通过 |
| `targets_separation` | 通过 |
| `slot_ids_ordered` | 通过 |
| `planes_rows` | 通过 |
| `planes_dtype` | 通过 |
| `beta_zero_rows_carry_no_injection` | 通过 |
| `collection_planes_match_source` | 通过 |
| `holdout_is_20pct` | 通过 |

数据集身份：`slots.jsonl` `faad8ef4…`、`targets.jsonl` `a23697a5…`、`meta.json` `0432433a…`；目录 `/mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T05/agg_d3/dataset/d1-s0d1-n3300-seed3000-shard`。

---

## 6 训练（一个新 checkpoint）

### 6.1 协议（事前冻结在 `configs/v420/dagger-D3.json#training`）

| 项 | 值 |
|---|---|
| 臂 / 目标列 | `A0` / `targets.jsonl:executed_cand` |
| 种子 / 步数 / batch | **0** / **5,000** / **128** |
| 优化器 / lr / wd / clip | adam / 0.0001 / 0.0 / 0.5 |
| 设备 | **cuda**（NVIDIA GeForce RTX 4070 SUPER，12281.5 MiB） |
| 确定性 | `torch.manual_seed=0`、`numpy=0`、`random=0`、`cudnn.deterministic=true`、`cudnn.benchmark=false`、`use_deterministic_algorithms=true`（strict，`warn_only=false`）、`CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| 模型 | 冻结 `ProbeNet`，**731,802** 参数 |
| 行数 / epochs | 43,200 train / 10,800 holdout；**15** epochs |

**一个必须声明的选择**：新模型**从随机初始化训练**，**不是**从 D2 warm-start。冻结训练器没有 warm-start 路径，加一条要改 `touhou_rl/v420/train.py`，不在本任务写权限内。所以「沿用 A0/D1/D2 超参」指协议相同，起点不同。

### 6.2 离线结果（都是语料内测量，不是能力）

| 行集 | CE | top-1 | macroR |
|---|---:|---:|---:|
| train | 0.329890 | 0.895069 | 0.498418 |
| holdout | 0.729505 | 0.772870 | 0.378803 |

参考标尺：均匀基线 `ln 18 = 2.8904`。holdout 是**这一份 50/30/20 混合语料**的语料内切分，构成变了，所以**不能**与 T04 的 A0 或 D2 的 holdout 直接比。

层 1（256 槽、300 步全批）：loss 2.895919 → **0.000036**、top1 0.015625 → 1.000000、proper safe-set 0.011719 → 0.378906。**这是记忆，不是泛化，也不是 G420-2。**

### 6.3 checkpoint 与被冻结 evaluator 读回的证明

| 项 | 值 |
|---|---|
| 文件名 | **`d3_seed0_step5000.pt`**（训练器先写 `a0_seed0_step5000.pt`，再由 `rename_checkpoint.py` 校验 sha256 后改名） |
| 路径 | `/mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T05/checkpoints_d3/d3_seed0_step5000.pt` |
| 大小 | 2,940,699 B |
| **sha256** | `91d11d22187bc25b15bc6eb36932aaf30f30c548ffa7ce076e7a57fbdd68a603` |
| schema | `v410-wp04-bc-arm-checkpoint/1` |
| arm / family / model_class / parameters | `B0` / `probe` / `touhou_rl.v410.student.diagnostics.ProbeNet` / 731,802 |
| `v420_arm` / `v420_target` | `A0` / `executed` |
| epochs | 15 |
| **读回** | `tools/rl/v410_student_eval.py` 成功：`loaded=true`、`parameters=731,802`、`parameters_pinned=731,802`、`parameters_match_pin=true`；8 行上训练模型与重载模型 logits **max abs diff = 0.0**（cpu (both sides)） |
| 改名校验 | `status = renamed`、`sha256_equal = true`、与既有 checkpoint 逐字节相同：false |

---

## 7 评测：两种口径各 20 个开发局

| 口径 | argv 差异 | 语义 |
|---|---|---|
| **标准 S1N** | **不带** `--stop-on-death` | 死亡可承受（3 条命），跑到 `game_over` / `stage_clear` / 19,887 帧 cap |
| **DIAG-FIRST-DEATH** | 带 `--stop-on-death` | 首次死亡即结束该局（历史 0/20 口径） |

两者同一个冻结 evaluator（`tools/rl/v410_student_eval.py`）、同一引擎、同一组 dev seed 1000–1019、同一 19,887 帧 cap；argv 由 `touhou_rl.v420.eval.build_legacy_eval_argv` 构造，不手拼 shell 字符串。

### 7.1 汇总

| 口径 | checkpoint | 局数 | 过关 | frames 均值 | median | min–max | deaths 均值 |
|---|---|---:|---:|---:|---:|---|---:|
| DIAG | **D3（本轮）** | 20 | **0** | **1,716.45** | 1,610.50 | 1,122–2,349 | 1.00 |
| DIAG | D2（参照） | 20 | — | 1,902.10 | — | — | — |
| DIAG | D1（参照） | 20 | — | 1,687.25 | — | — | — |
| DIAG | A0（参照） | 20 | — | 1,636.80 | — | — | — |
| S1N | **D3（本轮）** | 20 | **0** | **3,025.20** | 2,822.00 | 2,179–5,195 | 3.00 |
| S1N | D2（参照） | 20 | — | 3,185.70 | — | — | — |
| S1N | D1（参照） | 20 | — | 3,028.10 | — | — | — |
| S1N | A0（参照） | 20 | — | 3,011.50 | — | — | — |

- 两套 **errors = 0、deploy_input = 20/20 与 20/20**；`bombs_used` 均值 0.0。
- Wilson 95%（0/20）：DIAG [0.0000, 0.1611]、S1N [0.0000, 0.1611]；G420-4 的 three conditions **全部 not_met**。
- DIAG 主环境帧合计 34,329.0 帧（1,716.45 × 20）；S1N 合计 60,504.0 帧。

### 7.2 与 D2 / D1 / A0 / B0 的**逐 seed 配对差**（同一 seed 相减）

**DIAG**

| vs | 均值差 | 中位差 | 最小 / 最大 | 胜 / 负 / 平 | 参照均值 | 相对变化 |
|---|---:|---:|---|---:|---:|---:|
| **D2** | **-185.65** | -70.00 | -1488 / +712 | 7 / 13 / 0 | 1,902.10 | -9.76% |
| **D1** | **+29.20** | -28.50 | -950 / +866 | 10 / 10 / 0 | 1,687.25 | +1.73% |
| **A0** | **+79.65** | +36.50 | -837 / +855 | 13 / 7 / 0 | 1,636.80 | +4.87% |
| **B0** | **+263.20** | +394.50 | -1117 / +954 | 15 / 5 / 0 | 1,453.25 | +18.11% |

逐 seed 差值（D3 − 参照）：

| seed | 帧数 | vs D2 | vs D1 | vs A0 |
|---|---:|---:|---:|---:|
| 1000 | 1,454 | -491 | -195 | +45 |
| 1001 | 1,434 | -51 | -139 | -294 |
| 1002 | 2,168 | -266 | +866 | +669 |
| 1003 | 1,572 | +126 | -712 | +15 |
| 1004 | 1,922 | +452 | +350 | +512 |
| 1005 | 2,201 | +712 | -93 | +596 |
| 1006 | 1,432 | -1388 | +28 | +35 |
| 1007 | 1,412 | -39 | -950 | +4 |
| 1008 | 2,016 | +257 | +443 | +585 |
| 1009 | 1,828 | -343 | +356 | +336 |
| 1010 | 2,168 | +223 | +445 | +580 |
| 1011 | 1,940 | +374 | +396 | +527 |
| 1012 | 1,935 | -74 | +190 | +38 |
| 1013 | 1,122 | -1069 | -688 | -518 |
| 1014 | 1,301 | -133 | -262 | -837 |
| 1015 | 1,487 | -451 | -104 | -698 |
| 1016 | 1,367 | -578 | -85 | -200 |
| 1017 | 2,349 | +580 | +654 | +855 |
| 1018 | 1,649 | -66 | +241 | -436 |
| 1019 | 1,572 | -1488 | -157 | -221 |

**S1N**

| vs | 均值差 | 中位差 | 最小 / 最大 | 胜 / 负 / 平 | 参照均值 | 相对变化 |
|---|---:|---:|---|---:|---:|---:|
| **D2** | **-160.50** | -275.50 | -1850 / +2282 | 9 / 11 / 0 | 3,185.70 | -5.04% |
| **D1** | **-2.90** | -221.50 | -1730 / +2360 | 9 / 11 / 0 | 3,028.10 | -0.10% |
| **A0** | **+13.70** | +177.00 | -2785 / +1723 | 11 / 9 / 0 | 3,011.50 | +0.45% |
| **B0** | **+144.25** | -55.50 | -1180 / +2152 | 9 / 11 / 0 | 2,880.95 | +5.01% |

逐 seed 差值（D3 − 参照）：

| seed | 帧数 | vs D2 | vs D1 | vs A0 |
|---|---:|---:|---:|---:|
| 1000 | 2,520 | -845 | -516 | +383 |
| 1001 | 2,362 | -306 | -547 | -1068 |
| 1002 | 3,810 | +331 | -383 | +1625 |
| 1003 | 2,228 | -1100 | -999 | -503 |
| 1004 | 3,474 | +581 | +395 | +1018 |
| 1005 | 3,193 | +659 | -138 | +614 |
| 1006 | 3,032 | -1850 | +365 | +460 |
| 1007 | 2,383 | -953 | -1730 | -897 |
| 1008 | 3,618 | +302 | +587 | +437 |
| 1009 | 2,587 | -489 | +334 | -1353 |
| 1010 | 4,087 | +1053 | +932 | +1559 |
| 1011 | 2,732 | -245 | -305 | +192 |
| 1012 | 2,912 | +199 | +258 | -168 |
| 1013 | 2,492 | -1104 | -501 | -542 |
| 1014 | 3,240 | +1078 | +924 | +162 |
| 1015 | 2,179 | -1311 | -908 | -2785 |
| 1016 | 2,222 | -912 | -800 | -664 |
| 1017 | 5,195 | +2282 | +2360 | +1723 |
| 1018 | 3,857 | +922 | +1695 | +765 |
| 1019 | 2,381 | -1502 | -1081 | -684 |

**读法**：D3 相对 A0 的 DIAG 首死改善是 +4.87%，相对 D2 是 **-9.76%**（7 胜 13 负）。**相对 D2 为负**：多一轮聚合+训练并没有把学生推得更久，D3 的首死比 D2 更早，说明这条配方在这里已经不再单调改善。 逐 seed 差从 -837 到 +855 全幅摆动，这不是一个可区分的改善，更不是恢复能力；两套仍然 0/20。

---

## 8 G420-3：据实状态 `not_passed`

| 要求（README §5/§7） | 本轮实测 | 判定 |
|---|---|---|
| β=0 轮次数 ≥ 2 | **2/2**（D2, D3） | **满足** |
| β=0 自主决策 ≥ 50,000 | **51,671** = D2 24,671 + D3 27,000 | **满足** |
| 聚合三源与比例 | 0.500000 / 0.300000 / 0.200000，三源独立、无 top-up | **满足** |
| 起点分表（自然/课程） | 自然 27,000；教师中后段 0；上一轮失败 0 | **不满足** |
| 谱系（有效祖先） | **27** 个，与 D1/D2 不相交 | **满足** |
| 标签与每轮评测 | 18 路 scores/verified_mask/教师动作逐决策全记（27,000 次）；两种口径各 20 局 | **满足** |
| 整面过关 | 两套 0/20（三轮合计 0/120 episode） | **不满足** |

**本轮可以主张的**：

1. 一次真实运行产出了 **27,000 条 β=0 自主决策**（54,432 帧、27 个互不相同的祖先），`injection_rate = 0.0`、**108,432 个 HASH 校验 0 不一致**；
2. 这 27,000 条与 D2 的 24,671 条合计 **51,671**，把计划的两项决策量要求**都推到位**；
3. 聚合第一次做到三源独立（50/30/20 精确，bootstrap 未启用）；
4. 新 checkpoint 能被冻结 evaluator 原样加载（91d11d22187bc25b15bc6eb36932aaf30f30c548ffa7ce076e7a57fbdd68a603），两口径各 20 局完整测完。

**本轮不能主张的**：

1. **任何过关**：两套 0/20；
2. **起点三类**：只有自然类，教师中后段与上一轮失败前都是 0；
3. **「恢复有效」**：DIAG 首死相对 A0 +4.87%、S1N +0.45%，逐 seed 大幅摆动；
4. **区间结论**：只有 1 个训练 seed、20 个开发局；
5. **G420-3 通过**：决策量两项到位，但起点类别与整面过关两项没有到位。

---

## 9 停止条件的裁决（本轮最重要的输出）

计划规则：**三轮之后**若仍无整面过关、且**开发首死相对 A0 改善不足 30%**，结束当前 DAgger 配方，转已登记的标签/输入或 RL 分支。

| 项 | 值 |
|---|---|
| 已完成的轮次 | **3 / 3**（D1、D2、D3） |
| 参照 | A0 DIAG-FIRST-DEATH mean over dev seeds 1000-1019 = **1,636.80 帧** |
| 整面过关 | D3 **0 / 40**（两套 20 局）；三轮合计 0 / 120 |
| 阈值 | 改善 ≥ 30% |

| 轮次 | DIAG 首死均值 | 相对 A0 改善 |
|---|---:|---:|
| **D1** | 1,687.25 | **3.08%** |
| **D2** | 1,902.10 | **16.21%** |
| **D3** | 1,716.45 | **4.87%** |

**结论：停止条件 触发。**

- 「三轮之后」已满足：D1/D2/D3 三轮全部完成并各有自己的两口径 20 局。
- 「仍无整面过关」已满足：D3 两套 0/20，三轮合计 0/120 个开发局没有一次 `stage_clear`。
- 「改善不足 30%」已满足：三轮里最好的一轮（**D2**）的 DIAG 首死相对 A0 只有 **16.21%**，距 30% 门槛差 13.79 个百分点。

**触发后应转哪条分支**（计划已登记，不是新决定）：

1. **WP420-02 标签分支（首选）**：引擎验证的候选动作 + 续行验证，能产出真正有信息量的安全集合目标（A1），而不是继续模仿单一 chosen 动作。三轮最好的 DIAG 首死改善只有 16.21%，说明「照抄教师动作」这条监督信号没有把学生推到可过关的控制上；先修标签，再谈数据量。
2. **WP420-03 输入分支**：单因素输入变化（全局 192×224 / RGB / 局部高分辨率窗）或特权状态诊断臂 A2。D3 的 idle 占比 0.3152、只用到 13/18 个候选，是「像素+历史是否够用」这一问题的直接线索。
3. **WP420-05 引擎回报 RL 诊断**：已登记且不依赖本配方成功，可并行推进。

**这不是项目停止条件。** 停的是 DAgger 这一条配方链，不是目标；上述三条分支任意一条产出第一面候选，都继续走 WP420-06 的确认与 Windows 复验。

---

## 10 边界（必须与上面的数字一起读）

1. ONE beta = 0 round with one training seed and 20 development episodes per ruler: 27000 self-driven decisions, 27 ancestors.
2. the pilot is not counted toward G420-3 (the same rule D2 used), so the cumulative number stays comparable across rounds.
3. start kinds: natural only; teacher-mid and previous-round-failure are 0 and are never simulated.
4. the checkpoint is trained from a fresh init on the 50/30/20 aggregate, not warm-started from D2 (the frozen trainer has no warm-start path).
5. the holdout numbers are an in-corpus split of THIS aggregate; they are not comparable across rounds with different corpora.
6. the confirm batch 61000-61499 was never read; dev 1000-1019 counts as development exposure only.
7. no interval is claimed for the 20-episode means; a 0/20 pass rate has a Wilson 95% upper bound of about 0.161.

---

## 11 产物与复现

| 类别 | 路径 |
|---|---|
| 采集报告（合并） | `.tmp_v420_data/T05/D3.json`（+ `D3/<shard>/{slots.jsonl,obs/,meta.json}`，27 个 shard） |
| 逐命令记录 | `.tmp_v420_data/T05/D3_segments.jsonl`（argv / 起止 / 退出码 / 墙钟 / 报告摘要） |
| pilot | `.tmp_v420_data/T05/D3pilot.json`、`analysis_d3pilot.json`、`verify_d3pilot.json` |
| 复读 | `.tmp_v420_data/T05/verify_d3.json`、`.tmp_v420_data/T05/analysis_d3.json` |
| 聚合 | `.tmp_v420_data/T05/aggregate_d3.json`、`validate_aggregate_d3.json`、`agg_d3/dataset/…` |
| 训练 | `.tmp_v420_data/T05/train_d3.json`、`train_d3.log`、`train_d3.stdout`、`rename_d3.json` |
| checkpoint | `/mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T05/checkpoints_d3/d3_seed0_step5000.pt`（sha256 `None`） |
| 评测 | `.tmp_v420_data/T05/eval_d3/D3-{diag,s1n}-s*.json`（10 个 batch）+ `logs/`（逐 batch 墙钟） |
| 脚本 | `.tmp_v420_t05d3/*.sh`、`*.py` |
| 代码 | `touhou_rl/v420/dagger.py`, `tools/rl/v420.py`, `configs/v420/dagger-D3.json`, `tests/test_v420_dagger.py` |

复现顺序（每步都有非零退出即停）：

```bash
python .tmp_v420_t05d3/make_configs.py            # 逐 seed 的运行时配置
bash   .tmp_v420_t05d3/collect_all.sh pilot       # 2 个 seed 的分布检查（不计数）
bash   .tmp_v420_t05d3/collect_all.sh main        # 27 条 collect 命令，逐条记录
bash   .tmp_v420_t05d3/post_collect.sh            # 合并/复读/分析/聚合/校验/训练/改名
bash   .tmp_v420_t05d3/run_all_evals.sh D3 4      # diag + s1n，每条命令 ≤4 局
python .tmp_v420_t05d3/summarise.py               # 写 dagger_d3.json
python .tmp_v420_t05d3/write_report.py            # 由该 JSON 渲染本文件
```

（`collect` 对已存在的 shard **fail-closed**：同一 seed 的 shard 已存在时返回 `status="refused"`、退出码 2，不会追加。`--help` 退出码 0；dry-run 形式退出码 0。）

