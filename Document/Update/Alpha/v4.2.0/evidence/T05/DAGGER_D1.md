# T05 证据：WP420-04 第一轮（D1，β=0.25）有界 DAgger

> 机器可读镜像：[`dagger_d1.json`](dagger_d1.json)。本文件每个数字都在那里，且由
> `.tmp_v420_t05/summarise.py` 从**运行产物**生成（采集报告 `T05/D1.json`、五个 shard 的
> `slots.jsonl`、`T05/aggregate.json`、`T05/train.json`、十个 evaluator summary、以及只读的
> `evidence/T04/dev_eval.json`），不手写、不四舍五入后再抄。
>
> **本轮只做 D1。D2/D3（β=0）没有开始，也没有用任何方式替代。**
>
> **G420-3 = not_passed**，缺口见 §8。这不是能力声明：本轮的两套 20 局都是 **0/20 过关**。

---

## 0 结论摘要

| 问题 | 回答 | 依据 |
|---|---|---|
| 采集器是否真的跑通（环境 + shadow 教师 + 双引擎 HASH 对齐） | **是**：5 个 episode、5,000 决策、10,080 帧、5,000 次教师查询，**20,080 个 HASH 校验 0 不一致** | §2 |
| 注入比例是否等于 β | **24.50%**（1,225/5,000），目标 25%；5 个 shard 分别 22.3%–25.6% | §2.3 |
| 学生与教师是否真的分歧 | **是**：agreement **31.64%**（1,582/5,000），即 68.36% 的决策上教师的选择与学生 argmax 不同 | §2.3 |
| 有效祖先数 | **5**（seed 3000–3004，每个 1,000 决策）；5,000 个决策的 `hash_pre` **互不相同**（唯一率 1.0），1,000 个决策索引上**每个索引都恰有 5 个不同状态** | §4 |
| 聚合是否按 50/30/20 | **是**：5,000 / 3,000 / 2,000 = 0.5 / 0.3 / 0.2（独立复读 12 项检查全过） | §5 |
| 新 checkpoint 能否被冻结 evaluator 加载 | **能**：`load_v410_checkpoint` 读回成功，731,802/731,802，重载后 logits **max abs diff = 0.0** | §6.3 |
| 两种口径的 20 局 | 标准 S1N mean **3,028.10**、DIAG 首死 mean **1,687.25**，两套都 **0/20 过关** | §7 |
| 相对 A0 的配对差 | S1N **+16.60 帧（+0.55%）**、DIAG **+50.45 帧（+3.08%）**；11 胜 9 负 | §7.3 |
| G420-3 | **not_passed**：β=0 轮次 0/2、β=0 自主决策 0/50,000 | §8 |
| 停止条件 | **未触发**（规则按三轮判；本轮 D1 的 DIAG 首死相对 A0 只有 +3.08%，远低于 30%） | §9 |

一句话：**采集器、聚合、训练、两种口径评测这条链路第一次在 v4.2.0 的 D1 上完整走通，数据侧
（注入率、agreement、HASH、祖先）全部据实可核；能力侧没有任何改善证据 —— DIAG 首死 +3.08%、
标准 S1N +0.55%，两套 0/20，且只有一个训练 seed 与 20 个开发局。**

---

## 1 运行命令与退出码

全部在资产主机 WSL 内、`cwd=/mnt/d/WorkShop/Touhou_RL`、`PYTHONPATH=` 仓库根。

### 1.1 采集（验收命令）

```bash
xvfb-run -a -s "-screen 0 1024x768x24" \
  /root/v400-venv/bin/python tools/rl/v420.py collect \
  --config configs/v420/dagger-D1.json \
  --out $PWD/.tmp_v420_data/T05/D1.json --execute      # 退出码 0，报告 wall_s = 753.736 s
```

- UTC **13:33:15Z → 13:46:23Z**（含进程起停 788 s ≈ 13.1 min；报告内的 `wall_s=753.736` 只计
  采集本体）。
- 环境角色需要自己的 Xvfb（`xvfb-run -a`），shadow 教师由 `TeacherEngine` 自己剥掉 `DISPLAY`。
- **串行约束已遵守**：执行前有另一个任务（T03）的 env 引擎在跑，本轮用
  `.tmp_v420_t05/collect_guarded.sh` **等待 430 s** 直到主机上没有外来 `th08-modern`
  （日志逐行记录等待），**没有杀任何别人的进程**；第一次尝试即成功（attempt 1 exit 0）。
- 两个引擎各有**私有 data 副本**（`T05/work/envdata`、`T05/work/shadowdata`，从
  `T05/evaldata` 各 `cp -a` 一份），`/root/v400-data/**` 全程未写。

### 1.2 聚合、训练、评测

| 步骤 | 命令 | 退出码 | 墙钟 |
|---|---|---:|---:|
| 聚合数据集 | `python .tmp_v420_t05/aggregate.py` | 0 | 一次写入 514 MB |
| 训练配置生成 | `python .tmp_v420_t05/make_train_config.py` | 0 | — |
| 训练 | `python tools/rl/v420.py train --config .tmp_v420_t05/train-D1.json --out $V420_DATA/T05/train.json --execute` | **0** | 193.233 s（报告 `wall_s`；其中训练 118.766 s） |
| DIAG 首死 20 局 | `bash .tmp_v420_t05/run_eval.sh diag 4`（5 条命令、每条 4 局） | **0**（5/5 batch 全 0） | 499.827 s |
| 标准 S1N 20 局 | `bash .tmp_v420_t05/run_eval.sh s1n 4`（5 条命令、每条 4 局） | **0**（5/5 batch 全 0） | 788.797 s |

- **单条命令 ≤ 10 分钟**：10 条评测命令最长 171.722 s（2.9 min），最短 92.418 s。
- 两条独立复验（结果原样收在 `dagger_d1.json#verification`）：`verify_d1.py`（五个 shard 复读：
  行数/记录的 sha256/HASH 链/逐帧 hash/平面非空/18 宽标签/injection 与 agreement 重算 →
  `problems = []`）、`validate_aggregate.py`（聚合数据集复读，**12 项检查全过、failed = []**）。
  两者都是**另写**的只读脚本，不复用采集器的写路径。
- **一次必须记录的事故**：第一次调用评测包装脚本时，bash 的 `GROUPS` 是保留变量，分组数组
  被静默忽略，`--seeds` 收到 `["0"]`，`build_legacy_eval_argv` 按 seed 分区**直接拒绝**，
  **0 个 episode 被启动、0 个进入任何分母**（两份 log 里那一行 `seeds=0 exit=1` 就是它）。
  修好后重跑；DIAG 那次的包装脚本又因我在它运行期间去掉文件 BOM 而以退出码 2 结束（**5/5
  batch 仍全部 exit 0**，见 §7.5 的确定性复跑）。这两条都写在这里，不隐藏。

### 1.3 测试

| 侧 | 范围 | 结果 | 退出码 |
|---|---|---|---:|
| WSL | 8 个既有 v420 测试文件 | **290 passed** | 0 |
| WSL | 加上 `tests/test_v420_dagger.py`（9 个文件） | **315 passed** | 0 |
| Windows | 8 个既有文件 | **289 passed + 1 skipped** | 0 |
| Windows | 加上 `tests/test_v420_dagger.py` | **314 passed + 1 skipped** | 0 |
| Windows | `python -m compileall -q touhou_rl\v420` | 无输出 | 0 |

**基线口径已更新**：任务单里写的 267 是 `PROGRESS.md` §8 的旧快照；本轮**没有修改**那 8 个
文件（它们和整个 v4.2.0 代码一样是未跟踪新文件），现在同一组 8 个文件实测 WSL **290**、
Windows **289+1 skip**。新增 25 个测试即 `tests/test_v420_dagger.py`。
（另有一个并发任务在改 `touhou_rl/v420/counterfactual.py`；那 8 个文件的测试数由当前树决定，
不由本轮决定，本轮三次测量都是 290 / 289+1。）

---

## 2 采集契约与实测

### 2.1 两个引擎的固定身份

| 角色 | 进程 | 身份 |
|---|---|---|
| env（学生闭环 + 像素） | `th08-modern --rl-role env` | 引擎 `691bfd77…`（sha256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`） |
| shadow（教师标注 + 被注入动作的对照轨迹） | `th08-modern --rl-role shadow` | 同上 |
| 学生 | A0 checkpoint | `.tmp_v420_data/T04/checkpoints/a0_seed0_step5000.pt`，sha256 `96cb219e…6b89ad`，731,802 参数，经**冻结** `tools/rl/v410_student_eval.py` 的 loader 与 `ProbePolicy` 载入 |
| 教师 | `fast` @ `replan_decisions=1`、`shoot_pulse_period=48` | horizon 32 / beam 24 / verify_k 1 / 1 iteration；实测 **43.81–46.88 ms/query**（5 个 shard 分别 43.807 / 44.522 / 46.882 / 46.260 / 46.188 ms） |

两点必须与数字一起读：

1. `replan_decisions` 在 `touhou_rl/v400/teacher/decide.py` 里**没有被读取**（`grep` 无命中），
   它只是调用侧的约定；本采集器每个决策**只执行 `chosen.slots[0]` 一个 slot**（也就是两帧），
   所以"两个 2 帧节拍 = 一个教师 slot"这件事由采集器的执行规则保证，而不是由该参数保证。
2. `shoot_pulse_period=48`（单位是**决策**）保证每 48 个决策中有一帧松 SHOOT（=96 帧一次），
   该释放**在 env 与 shadow 上同时施加**并按帧记录（`masks[1] = mask & ~SHOOT & ~BOMB`，
   行内 `release_shoot=true`）。5 个 shard 各 20 次释放（1,000/48 = 20.8 → 20）。

### 2.2 两帧节拍与冷启动

- 决策周期 2 帧；观测 = 4 帧栈（`d-4..d-1`）+ 该帧 HUD（`d-1`）+ 8 条**已执行**的 2 帧历史
  （`d-16..d-2`），与 `obs-v420-2f` 契约逐字段对齐（`obs.play_frames` / `obs.hud_frame` /
  `obs.history_frames` 逐槽写出）。
- 决策帧 0–14 是语料声明的**冷启动**：由学生执行、**不记录、不补零成槽**（每 episode 8 个决策，
  5 个 episode 共 40 个）。因此记录的 1,000 个决策从 `decision_frame=16` 开始，历史恰有 8 条。
- 逐帧推进（每个决策两帧、每帧一个 `STEP … 1` + 一次渲染），一次决策**从不跨帧折叠**：
  行内 `masks` / `frame_hashes` 都是 2 项。

### 2.3 注入与一致率（实测 vs 目标）

| shard（seed） | 决策 | 注入 | 注入率 | agreement | 一致率 | bomb 决策 | SHOOT 释放 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3000 | 1,000 | 223 | 0.223 | 308 | 0.308 | 0 | 20 |
| 3001 | 1,000 | 252 | 0.252 | 377 | 0.377 | 0 | 20 |
| 3002 | 1,000 | 256 | 0.256 | 370 | 0.370 | 1 | 20 |
| 3003 | 1,000 | 255 | 0.255 | 233 | 0.233 | 1 | 20 |
| 3004 | 1,000 | 239 | 0.239 | 294 | 0.294 | 1 | 20 |
| **合计** | **5,000** | **1,225** | **0.2450** | **1,582** | **0.3164** | **3** | **100** |

- **目标 β = 0.25，实测 24.50%**（−0.5 pp；5 个独立 RNG 流各 1,000 次伯努利抽样，这在抽样噪声内）。
- `agreement` 的定义固定为"教师 `chosen.slots[0]` == 学生在同一决策点的 argmax"，**与是否注入
  无关**；`injected` 的定义固定为"该决策的 draw < β 且教师有计划"。两个定义都写进
  `meta.injection.definitions` 与每行的 `source` 字段，测试里逐行验证。
- **68.36% 的决策上两者不同**：这正是 DAgger 需要的数据，也是 A0 的 argmax 与教师选择的真实差距。
- 只有 3 个决策执行了 bomb（都是教师注入的死亡炸弹路径），三行都记录 `bomb=1` 与 `masks` 里的
  bomb 位；学生本身从不按 bomb（B0 头为 0）。

### 2.4 HASH 校验（fail-closed）

| 项 | 实测 |
|---|---:|
| 对齐边界总数 | **20,080**（每个 shard 4,016 = 16 冷启动帧 + 1,000 pre + 2,000 帧 + 1,000 post） |
| 不一致 | **0** |
| 决策间 HASH 链（`hash_post == 下一行 hash_pre`） | 4,995 对，**0 断点** |
| `hash_post == 最后一帧 hash` | 5,000/5,000 |

- 每个决策**前后**都取两边的 HASH；每帧执行后也逐帧比对（`frame_hashes`）。任何一次不一致都会
  立即停止该轮：已写的行**保留**在 shard 里（`meta.complete=false` + `alignment_failed=true` +
  `first_mismatch`），并返回退出码 3。本轮**没有触发**这条路径。
- 冷启动帧同样比对（16/16 每 shard），所以"记录段之外"也没有漂移。

### 2.5 一个已知的小缺口（不事后改证据）

`shard/meta.json` 里**没有**独立的 `contract` 与 `student` 块：五个 shard 是在
`dagger.py` 补上这两个块之前写下的。契约版本仍然写在**全部 5,000 行 slot**（每行
`contract`/`obs_contract`/`action_contract`）与 **round 报告** `D1.json#contract` 里，学生身份
写在 round 报告的 `student` 与每个 checkpoint 的 `v420_arm` 里；`meta.json` 里只有
`schema`/`shard_key`/`slots`/`obs`/`teacher`/`env`/`injection`/`alignment`。

**本轮没有事后改写那五个 `meta.json`**，因为 `D1.json#files.meta_sha256` 逐个钉住了它们的哈希，
改了就会让报告与磁盘不一致 —— 那比缺一个字段更糟。代码已经补上（`dagger.py` 现在写
`contract` 与 `student`，并有测试覆盖），下一次采集的 meta 会带这两个块。这条同时写在
`dagger_d1.json#boundaries`。

---

## 3 起点类别（计划的三类 vs 本轮实际）

| 起点类别 | 计划目标 | 本轮实测 | 说明 |
|---|---:|---:|---|
| 自然开局 `natural` | 50% | **5,000 决策 / 100%** | 每个 episode 从 `RESET(seed)` 起，学生自己控制冷启动 |
| 教师中后段 `curriculum_teacher_mid` | 30% | **0** | **未实现**：需要分支/重放的课程起点，本采集器**拒绝**而不是模拟 |
| 上一轮失败前状态 `prev_round_failure` | 20% | **0** | 同上（本轮之前也不存在"上一轮"） |

- 采集器对 `starts` / `start_kind` 配置**fail-closed**：请求 `curriculum_teacher_mid` 或
  `prev_round_failure` 会直接 `DaggerError`（测试 `test_curriculum_start_is_refused_not_simulated`
  覆盖），shard 的 `meta.json` 里 `unavailable_start_kinds` 显式列出这两类、`start_kind_counts`
  给出 0，round 报告里 `start_kind_gap` 给出原因。
- **因此本轮的起点分表只有一个非零格**：这是缺口，不是"三类都做了"。

---

## 4 轨迹、祖先与动作分布

| 项 | 实测 |
|---|---:|
| episode seed（有效祖先） | **5**：3000/3001/3002/3003/3004，各 1,000 决策 |
| 决策索引数 | 1,000（每 seed 都有 1,000） |
| 每个索引上的不同 `hash_pre` 数 | **恰为 5**（min=max=mean=5），1,000/1,000 个索引"五 seed 全不同" |
| 唯一 `hash_pre` 总数 | **5,000 / 5,000（1.0）** |
| 冷启动以外的重复轨迹 | **0** |
| 教师 18 路 scores 跨 seed 相同 | 240/4,000 对（6.0%） |

读法：**5 条轨迹是真不同的**（每个状态哈希都唯一），所以"祖先数 5"不是把同一条轨迹改个 seed
名。只有 6% 的跨 seed 同索引状态给出完全相同的教师评分（开局几何一致的那些点），其余都不同。

动作分布（决策数）：

| cand | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 11 | 12 | 14 | 15 | 16 | 17 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 学生 argmax | 1,597 | 3 | 334 | 652 | 0 | 0 | 634 | 794 | 0 | 0 | 200 | 278 | 0 | 251 | 257 | 0 |
| 实际执行 | 1,551 | 8 | 376 | 846 | 16 | 9 | 579 | 796 | 6 | 2 | 184 | 241 | 1 | 185 | 198 | 2 |

- 学生只用到 **10/18** 个候选，且 `cand 0` 占 31.9% —— 与 T04 的 idle 偏斜结论一致；教师注入把
  执行侧扩到 **16/18** 个候选。这是"当前候选的错误区域"的量化形状。

---

## 5 聚合：50 / 30 / 20（含 bootstrap 配方）

**固定比例**（计划）：当前轮 50% / 最近历史轮 30% / 固定 BC 20%。

| 来源 | 行数 | 实际占比 | 来源说明 |
|---|---:|---:|---|
| `current_round` | **5,000** | **0.500000** | 本轮 D1 的五个 shard，一行不删 |
| `history_round` | **3,000** | **0.300000** | 被消费的 `v420-2f-dataset/1`（19,772 槽）中随机（seed 42050）抽 3,000 行 |
| `fixed_bc` | **2,000** | **0.200000** | **同一语料**中与上一行集合**不相交**的 2,000 行 |
| **合计** | **10,000** | **1.000000** | 8,000 train / 2,000 holdout（`row_index % 5 == 4`，按来源 1,009 / 578 / 413） |

**bootstrap 配方（必须声明）**：v4.2.0 还没有更早的 DAgger 轮，所以"最近历史轮"和"固定 BC"
**都来自同一份被消费语料**，只是**互不相交**的两块行。**比例是计划的，来源还不是三项独立的。**
这一句同时写进 `meta.aggregate.bootstrap_recipe` 与 `dagger_d1.json#aggregate`。

独立复读（`validate_aggregate.py`，另写的只读检查，12/12 通过）：

- 行数 / 每来源行数 / 比例（误差 ≤1e-6）/ history 与 fixed 不相交 / 同来源无重复 origin /
  10,000 行标签字段全合法（18 宽 scores + 18 宽 mask + `verified_count` 自洽 +
  `chosen_verified` 与 mask 一致）/ targets 与 slots 两列不混 / `slot_id` 有序 /
  三个平面行数一致且 dtype=uint8 / **50 行 D1 平面与源 shard 逐字节相同** / holdout 恰 20%。

数据集身份：`meta.json` sha256 `8ff2db62…fd3d06`、`slots.jsonl` `df286f03…d1846c`、
`targets.jsonl` `6151560e…f73ddd`；目录
`.tmp_v420_data/T05/agg/dataset/d1-s0d1-n3300-seed3000-shard`（514 MB）。
目录名的 `d1-s0d1-n3300-seed{seed}-shard` 形态来自**冻结训练器自己的模板**，是名字不是断言；
`meta.counts.slots = 10000` 才是行数。

---

## 6 训练（一个新 checkpoint）

### 6.1 协议（事前冻结在 `configs/v420/dagger-D1.json#training`，训练配置由它生成）

| 项 | 值 |
|---|---|
| 臂 / 目标列 | `A0` / `targets.jsonl:executed_cand`（DAgger 轮里即"实际执行"的 β 混合动作） |
| 种子 / 步数 / batch | **0** / **5,000** / **128** |
| 优化器 / lr / wd / clip | Adam / 1e-4 / 0.0 / 0.5 |
| 设备 | **cuda**（RTX 4070 SUPER，12,281.5 MiB） |
| 确定性 | `torch.manual_seed(0)`、`numpy.seed(0)`、`random.seed(0)`、`cudnn.deterministic=true`、`cudnn.benchmark=false`、`use_deterministic_algorithms(true)`（**strict，非 warn_only**）、`CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| 模型 | 冻结的 `ProbeNet`，**731,802** 参数 |
| epochs / steps_per_epoch | 80 / 63 |

**一个必须声明的选择**：新模型**从随机初始化训练**，**不是**从 A0 warm-start。冻结训练器没有
warm-start 路径，加一条要改 `touhou_rl/v420/train.py`，那不在本任务的写权限内。所以本轮
"沿用 A0 超参"指协议相同，起点不同。

### 6.2 离线结果（**都是语料内测量，不是能力**）

| 行集 | CE | top-1 | macroR | proper safe-set |
|---|---:|---:|---:|---:|
| train（8,000） | **0.004175** | 0.999125 | 0.999380 | 0.5855 |
| holdout（2,000） | **5.179169** | 0.495000 | 0.258245 | 0.3875 |
| train（另一列 `chosen_verified`） | 6.477883 | 0.5855 | 0.512232 | 0.5855 |
| holdout（另一列） | 7.619916 | 0.3875 | 0.168769 | 0.3875 |

- 参考标尺：均匀基线 `ln 18 = 2.8904`。**holdout CE 5.179 高于均匀基线**，top-1 0.495 比随机高
  但远未可用；train CE 0.0042 是 80 个 epoch 的记忆。
- **不能与 T04 的 A0 数字直接比**：holdout 是**这一份 50/30/20 混合语料**的语料内切分，构成变了，
  所以 A0 的 2.802 与本轮的 5.179 不是同一个量。它只能说明"这份聚合语料被拟合到什么程度"。
- 层 1（256 槽、300 步全批）：loss 2.905933 → **4.4e-05**、top1 0.007812 → 1.0、safe-set 0.011719
  → 0.566406；留出 CE 12.221。**这是记忆，不是泛化，也不是 G420-2。**
- 吞吐 5,346.16 samples/s、42.10 steps/s、634,944 次样本呈现；曲线单调下降无发散
  （100 步 loss 2.234 → 5,000 步 0.004728）。

### 6.3 checkpoint 与被冻结 evaluator 读回的证明

| 项 | 值 |
|---|---|
| 路径 | `.tmp_v420_data/T05/checkpoints/a0_seed0_step5000.pt` |
| 大小 | 2,940,699 B |
| **sha256** | `645098519922ea69acde92b2a3c5422f564c2971fef1495a4c99b401f1f02b99` |
| schema | `v410-wp04-bc-arm-checkpoint/1` |
| arm / family / model_class / parameters | `B0` / `probe` / `touhou_rl.v410.student.diagnostics.ProbeNet` / 731,802 |
| seed / epochs / last_epoch | 0 / 80 / 80 |
| `v420_arm` / `v420_target` | `A0` / `executed` |
| **读回** | `load_v410_checkpoint` 成功：`loaded=true`、`parameters=731802`、`parameters_pinned=731802`、`epochs=80`；8 个留出行上训练模型与重载模型 logits **max abs diff = 0.0**（两侧都在 CPU） |

---

## 7 评测：两种口径各 20 个开发局

### 7.1 口径必须分开

| 口径 | argv 差异 | 语义 |
|---|---|---|
| **标准 S1N** | **不带** `--stop-on-death` | 死亡可承受（3 条命），跑到 `game_over` / `stage_clear` / 19,887 帧 cap |
| **DIAG-FIRST-DEATH** | 带 `--stop-on-death` | 首次死亡即结束该局（历史 0/20 口径） |

两者同一个冻结 evaluator（`tools/rl/v410_student_eval.py`）、同一引擎（`691bfd77…`）、
`--stage 0 --difficulty normal --max-frames 19887 --obs-window consecutive --device cpu`、dev
seed **1000–1019**；argv 全部由 `touhou_rl.v420.eval.build_legacy_eval_argv` 构造（不手拼
shell 字符串），逐条 argv 见 `dagger_d1.json#evaluation.<suite>` 与 `.tmp_v420_data/T05/eval/logs/`。
数据目录是**副本** `T05/evaldata`（`th08.dat`/`thbgm.dat` 指向只读原件的符号链接）。

### 7.2 汇总

| 口径 | checkpoint | 局数 | 过关 | frames 均值 | median | min–max | deaths 均值 | stop_reason |
|---|---|---:|---:|---:|---:|---|---:|---|
| DIAG | **D1（本轮）** | 20 | **0** | **1,687.25** | 1,582.0 | 1,302–2,362 | 1.00 | `death_stop=20` |
| DIAG | A0（参照） | 20 | 0 | 1,636.80 | 1,562.0 | 1,397–2,185 | 1.00 | `death_stop=20` |
| DIAG | B0（参照） | 20 | 0 | 1,453.25 | 1,374.0 | 815–2,386 | 1.00 | `death_stop=20` |
| S1N | **D1（本轮）** | 20 | **0** | **3,028.10** | 3,033.5 | 2,162–4,193 | 3.00 | `game_over=20` |
| S1N | A0（参照） | 20 | 0 | 3,011.50 | 3,049.5 | 2,137–4,964 | 3.00 | `game_over=20` |
| S1N | B0（参照） | 20 | 0 | 2,880.95 | 2,977.0 | 1,900–4,798 | 3.00 | `game_over=20` |

- 两套 **errors = 0、invalid = 0、deploy_input = 20/20**；`bombs_used` 均值 0.0；
  `action0_share` 均值 DIAG 0.2883、S1N 0.3228。
- Wilson 95%（0/20）：**[0.0000, 0.1611]**（两套相同）；G420-4 三个条件
  （`episodes` / `successes` / `wilson_lower`）**全部 not_met**，唯一满足的是"部署输入"——
  但这不改变任何判定。
- DIAG 主环境帧合计 **33,745**；S1N 合计 **60,562**；两套合计 **94,307 帧**。
- A0/B0 的逐局数字**不是重跑**，而是只读引用 `evidence/T04/dev_eval.json`（同一 evaluator、
  同一引擎、同一 20 个 seed、同一 cap），所以配对差成立；B0/20k 在 T04 已与历史基准逐 seed 精确
  一致，这条一致性也是本轮引用的前提。

### 7.3 与 A0 / B0 的配对差（同一 seed 相减）

| 口径 | vs | 均值差 | 中位差 | 胜/负 | 相对变化 |
|---|---|---:|---:|---|---:|
| DIAG | **A0** | **+50.45 帧** | +69.0 | **11 / 9** | **+3.08%** |
| DIAG | B0 | +234.00 帧 | +203.5 | 15 / 5 | +16.10% |
| S1N | **A0** | **+16.60 帧** | +115.5 | **11 / 9** | **+0.55%** |
| S1N | B0 | +147.15 帧 | +273.5 | 12 / 8 | +5.11% |

逐 seed 的 DIAG 差（D1 − A0）：1000 **+240**、1001 −155、1002 −197、1003 **+727**、1004 +162、
1005 **+689**、1006 +7、1007 **+954**、1008 +142、1009 −20、1010 +135、1011 +131、1012 −152、
1013 +170、1014 **−575**、1015 **−594**、1016 −115、1017 +201、1018 **−677**、1019 −64。

**读法**：11 胜 9 负、均值 +3.08%，但逐 seed 差从 **+954**（seed 1007）到 **−677**（seed 1018）
全幅摆动 —— 这**不是**一个可区分的改善，更不是"恢复能力"。它在 DIAG 口径上比 A0 略好、在标准
S1N 上几乎持平，两套都 0/20。

### 7.4 单条命令 ≤ 10 分钟（逐 batch）

| 口径 | 5 条命令的墙钟（s） | 最长 |
|---|---|---:|
| DIAG | 102.673 / 112.943 / 95.812 / 95.981 / 92.418 | **112.943（1.9 min）** |
| S1N | 171.722 / 169.249 / 148.546 / 144.000 / 155.280 | **171.722（2.9 min）** |

### 7.5 一次确定性复跑

DIAG 套件在包装脚本修好后**完整重跑了一次**（同样 5 条命令、全部 exit 0）：**20/20 个 episode
的帧数逐个完全相同**（差异 0 个，两者均值都是 1,687.25）。第一跑的原始 JSON 保留在
`.tmp_v420_data/T05/eval/run1/`。所以本节的逐 seed 数字是可复现的，不是主机时序噪声。

---

## 8 G420-3：据实状态 `not_passed`

**计划要求**（README §5 WP420-04 / §7 G420-3）：**两轮 β=0 累计至少 50,000 个自主决策**，加聚合
来源与自然/课程起点分表（以及谱系、标签、每轮评测）。

| 要求 | 本轮实测 | 判定 |
|---|---|---|
| β=0 轮次数 ≥ 2 | **0/2**（本轮是 β=0.25 的 D1；D2/D3 未开始） | 不满足 |
| β=0 自主决策 ≥ 50,000 | **0/50,000** | 不满足 |
| 聚合来源与比例 | 50/30/20 **达成**，但"最近历史轮"与"固定 BC"同源（bootstrap） | 部分 |
| 自然 / 课程起点分表 | 自然 **5,000**；教师中后段 **0**；上一轮失败 **0** | 不满足（缺 2 类） |
| 谱系（有效祖先） | **5** 个 seed，状态唯一率 1.0 | 达成 |
| 标签与每轮评测 | 18 路 scores/verified_mask/教师动作逐决策全记；两种口径 20 局 | 达成 |

**本轮可以主张的**：采集器在一次真实运行中产出了 5,000 条**带完整 18 路教师标签**的
β=0.25 决策，注入率 24.50%、agreement 31.64%、**20,080 个 HASH 校验 0 不一致**、5 个互不相同的
祖先；这些数据按 50/30/20 聚合成 10,000 槽并被训练真实消费，产出的 checkpoint 能被冻结
evaluator 原样加载。

**本轮不能主张的**：

1. **任何 β=0 自主覆盖**（0 个决策）；
2. 起点分表的三类（只有自然类）；
3. "恢复有效"——DIAG +3.08%、S1N +0.55%、两套 0/20；
4. 任何区间结论——只有 1 个训练 seed、20 个开发局，且没有配对区间；
5. 方法可推广——聚合的 history/fixed 同源，还不是三项独立来源。

---

## 9 停止条件的触发判断（不提前宣布结束）

计划的规则（README §5）：**三轮之后**若仍无整面过关、且**开发首死相对 A0 改善不足 30%**，
结束当前配方，转已登记的标签/输入或 RL 分支。

| 项 | 值 |
|---|---|
| 已完成轮次 | **1 / 3**（D1） |
| 是否触发 | **未触发** —— 规则按三轮判，一轮不是三轮 |
| 现有数字 | DIAG 首死 1,687.25 vs A0 1,636.80 = **+3.08%**（阈值 30%） |
| 整面过关 | 标准 S1N **0/20**、DIAG **0/20** |

把这组数字现在就写下来，是为了让第三轮的判定是**比较**而不是回忆。它**不**表示配方已经失败，
也**不**授权提前结束；同时它确实说明：如果 D2/D3 沿用同一配方，要跨过 30% 门槛需要**大幅**
改变开发首死，而不是再 +3%。

---

## 10 边界（必须与上面的数字一起读）

1. **只做 D1**：一个 β=0.25 轮、一个训练 seed（0）、每口径 20 个开发局。D2/D3 未开始。
2. **没有区间比较**：20 局的均值不带区间，也不能外推成过关率（0/20 的 Wilson 上界是 0.1611）。
3. **不是能力**：两种口径都 0/20 过关；DIAG 首死只作诊断。
4. **起点缺两类**（§3），聚合的 history/fixed 同源（§5）。
5. **训练起点不同**：新模型是随机初始化 + 5,000 步，不是从 A0 warm-start（§6.1）。
6. **holdout 不可跨语料比**：它是这份混合语料的语料内切分（§6.2）。
7. **确认批次未读**：61000–61499 全程未读；dev 1000–1019 早已暴露，只算开发曝光。
8. **没有跑的**：D2（β=0）、D3（β=0）、新 checkpoint 的 seeds 1/2、微段层 2、A1/A2 臂、
   确认批次。

---

## 11 产物与复现

| 类别 | 路径 |
|---|---|
| 采集报告 | `.tmp_v420_data/T05/D1.json`（+ `T05/D1/<shard>/{slots.jsonl,obs/,meta.json}`，5×52 MB） |
| 采集日志 / 主机等待 | `.tmp_v420_data/T05/collect.log`、`collect_guarded.sh` 的 job 输出 |
| 聚合 | `.tmp_v420_data/T05/aggregate.json`、`.tmp_v420_data/T05/agg/dataset/…`（514 MB） |
| 聚合复读 | `.tmp_v420_data/T05/verify_d1.json`（shard 复读）、`.tmp_v420_data/T05/validate_aggregate.json`（数据集复读） |
| 训练 | `.tmp_v420_data/T05/train.json`、`train.log`、`T05/checkpoints/a0_seed0_step5000.pt` |
| 评测 | `.tmp_v420_data/T05/eval/D1-{diag,s1n}-*.json`（10 个 batch summary + 10 个 driver）、`eval/run1/`（第一跑）、`eval/logs/` |
| 脚本 | `.tmp_v420_t05/*.sh`、`*.py`（`collect_guarded.sh`、`aggregate.py`、`make_train_config.py`、`train.sh`、`run_eval.sh`、`run_all_evals.sh`、`summarise.py`、`verify_d1.py`、`validate_aggregate.py`） |
| 代码 | `touhou_rl/v420/dagger.py`（新）、`tools/rl/v420.py`（只接 `collect`）、`configs/v420/dagger-D1.json`（新）、`tests/test_v420_dagger.py`（新） |

复现顺序（每步都有非零退出即停）：

```bash
bash .tmp_v420_t05/make_evaldata.sh
bash .tmp_v420_t05/collect_guarded.sh 2400 3          # 采集（会等主机空闲）
python .tmp_v420_t05/aggregate.py                     # 50/30/20
python .tmp_v420_t05/make_train_config.py
bash .tmp_v420_t05/train.sh
bash .tmp_v420_t05/run_all_evals.sh 4                 # diag + s1n，每条命令 ≤4 局
python .tmp_v420_t05/summarise.py                     # 写本文件的机器可读镜像
```

（`collect` 对已存在的 shard 目录 **fail-closed**：同一 `shard_root` 再跑一次会返回
`status="refused"`、退出码 **2**，并列出五个既有 shard，不会追加。复现到同一路径前必须换
`shard_root`。验收用的 dry-run 形式（不带 `--execute`）退出码 0；`rl` 子命令仍是
`not_implemented`、退出码 3；`--help` 退出码 0。）
