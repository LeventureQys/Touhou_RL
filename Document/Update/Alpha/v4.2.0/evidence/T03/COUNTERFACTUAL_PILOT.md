# T03 反事实首动作试点：20 状态 × 18 首动作的实引擎验证（WP420-02 pilot）

日期：2026-11-13，Asia/Shanghai。资产主机（本机 WSL2），`cwd=/mnt/d/WorkShop/Touhou_RL`，
Python `/root/v400-venv/bin/python`（3.10.12），引擎 `rl_v3`
`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`（sha `691bfd77…7a568ef6`），
分支角色 `shadow`（`env -u DISPLAY`，无 GL、无 xvfb）。

**范围声明（必读）**：本文件是一次 **20 状态成本与有效性试点**，用真实引擎跑完
20 状态 × 18 首动作 × 两档视界（H=32、H=120）。它**不是 G420-1**：门要求 200 状态全候选、
≥100 个有区分度的有效状态、≥10 条祖先与重复分支一致，本轮全部不满足（见 §8）。
本文件不写“G420-1 通过”，不把几何分数当引擎真值，不补造正例。

原始数据：`counterfactual_pilot.json`（本目录，1.56 MB，含两档逐分支裁决、安全集合、
逐状态 HASH 校验计数、前缀掩码 RLE、成本与控制实验）。
完整逐帧 trace 与逐分支明细在运行目录 `.tmp_v420_data/T03/`（`pilot_h32.json`、
`pilot_h120.json`、`raw/`），不在证据目录里重复 43,200 行 ×2 的帧记录。

---

## 0. 结论摘要

1. **两档都跑完，零 unknown、零 fail-closed 触发**：H=32 与 H=120 各有 **360 条分支**
   （20 状态 × 18 候选），五类结果都是 `engine_alive = 360`，
   `queried / executed / engine_death / unknown = 0`。
2. **本轮的核心实测结论是“无区分度”**：两档共 720 条分支里**没有一条死亡**，
   20 个状态在两档下的安全集合**全部等于 18**（`empty=0`、`all_safe=20`、`proper=0`）。
   也就是说：**在固定教师续行（`fast` @ `replan_decisions=1`，每 2 帧重规划）下，
   这些状态对 18 个首动作没有分辨力**——包括“旧模型首次死亡前 0.5 s”的状态。
   这是有效负结果，不是管线失败（§5、§7）。
3. **分支确实是 18 个不同动作**：每个状态 18 条分支的第一帧引擎 HASH **18/18 互不相同**，
   到 H 时的末帧 HASH 仍有 **8–18 个互不相同**（H=32 均值 15.4，H=120 均值 15.8）。
   所以“全存活”不是“18 条分支跑成了同一条”。
4. **前缀 HASH 逐帧校验通过**：H=32 档 **311,490** 次、H=120 档 **311,490** 次
   （两档同样的 20 个状态），**0 处不一致**。学生轨迹前缀（shadow 回放 env 录制的
   逐帧掩码）是逐比特可复现的。
5. **死手检测有正对照**：把学生的**自己**的后续掩码接在同一状态上，shadow 在
   **帧 1541 / 1928 / 1797** 检出死亡，与 env 角色记录的首次死亡帧**完全相同**，
   死亡 HASH `8b573c0b…` / `d84b6be2…` / `ed0a9bec…` 也与轨迹末帧一致（§9.1）。
   所以“本轮 0 死亡”不是检测器看不见死亡。
6. **重复分支一致（试点规模）**：3 对 (状态, 候选) 各跑两次，逐帧 trace 的 sha256
   **3/3 完全相同**，裁决相同（§9.2）。门要求的是 200 状态全量重复，不是这个 3 对。
7. **成本**：H=32 档 **599.3 s（9 min 59 s）**、H=120 档 **1,572.0 s（26 min 12 s）**；
   教师查询 5,400 / 21,240 次，教师内部帧 345,600 / 1,359,360（**均为查询数 × 64**，
   逐档精确相等），实测 61.6 / 60.0 ms/查询，与 `TEACHER_2F_COST.md` 的 66.58 ms
   同量级但更低（本轮主机空载，那份文件是争用条件下的上界）。
8. **距离 G420-1**：状态 20/200、proper 状态 **0/100**、祖先 3/10、重复一致 3 对（门要 200 状态）、
   学生续行对照 0/50。**最关键的缺口不是数量而是区分度**：按本轮继续扩到 200 个同类状态，
   proper 仍会是 0（§8）。

---

## 1. 本次实现

### 1.1 `touhou_rl/v420/counterfactual.py`（新增实现，替换原 not_implemented 占位）

| 部分 | 内容 |
| --- | --- |
| 纯函数层（stdlib + `contracts`，无 numpy/torch/引擎） | 语料/轨迹读取、前缀 RLE 与 HASH 链、分层抽样与去重、预注册校验、成本模型、`summarise`、`merge_reports`、`g420_1_verdict` |
| 引擎层（惰性 import，仅 `--execute` 路径） | `probe_student_traces`（`role="env"`，xvfb）、`_Session`（`role="shadow"`，每 seed 一个进程）、`_run_one_branch`、`run_pilot` |

三个必须实现的入口：

* `select_states(config) -> (list[StateRef], selection)`：分层抽样 + 去重 + 来源与限制记录；
* `run_branch` = `run_pilot(...)`/`_run_one_branch(...)`：RESET → 前缀重放（HASH 校验）→
  候选 2 帧 → 教师续行；
* `summarise(report) -> summary`：安全集合大小直方图、空/全安全/proper 计数、
  教师续行死亡数、unknown 数、每状态祖先与位置，并内嵌 `g420_1_verdict`。

结果分类**只**用 `contracts.classify_branch_outcome`，安全集合**只**用
`contracts.safe_action_set`；`unknown` 不会变成 `engine_death`，也不会进入安全集合
（由 contracts 的分支顺序保证，测试另钉了一遍）。

### 1.2 分支语义（每一条分支都这样跑）

```text
RESET(seed, stage=0, difficulty=1, team=0)
  -> 校验 RESET HASH == 前缀 frame 0 的 HASH（不等则重试一次干净 RESET）
  -> 按记录的逐帧掩码重放 1..decision_frame（逐帧比对记录的 HASH）
  -> 校验停放点 HASH == 该状态冻结的 expected_hash（不等即 fail-closed，不写结果）
  -> 施加候选首动作 恰好 2 帧（DECISION_FRAMES）
  -> 循环 { 教师 decide()@当前帧 -> commit_block(chosen.slots[:1]) -> 逐帧 STEP + STATE }
     直到 命中 H / 首次死亡 / 游戏结束 / 本地过面
```

* 候选 2 帧用 `cand_to_mask(cand, 0)`（与语料/学生同一动作表，SHOOT 常开、bomb 不用）。
* 教师续行固定为 `dataclasses.replace(get_config("fast"), replan_decisions=1,
  shoot_pulse_period=48)`：`horizon=32`、`decision_frames=2`、`beam=24`、`verify_k=1`，
  即 `TEACHER_2F_COST.md` 实测的 2 帧节拍；`shoot_release` 用**全局决策序号**
  （`frame // 2`）配 48 决策周期，保持 96 帧一次的 SHOOT 脉冲。
* 逐帧记录：`[frame, engine_hash, deaths, lives, bombs, score, stage, gameplay_frame]`，
  每分支另存 `trace_sha256` 与 `prefix_runs`（RLE）+ `state_hash`，**分支可重建**，
  没有新增任何快照序列化 ABI。
* 终止条件（事前定义，都是有效完成）：到达 H、首次死亡、游戏结束、本地过面。
  超时/缺帧/进程故障 = `unknown` 并写原因。

### 1.3 CLI 接入（只动 `counterfactual` 子命令）

```text
v420.py counterfactual --config ... [--out ...]                       # 默认 dry-run：冻结状态清单 + 成本估算
v420.py counterfactual --config ... --execute --probe-student         # env 角色录制学生首死（需 xvfb-run）
v420.py counterfactual --config ... --execute [--horizon 32|120] \
        [--state-range A:B] [--states-file frozen.json]               # 真实分支
v420.py counterfactual --config ... --merge 5 个 shard.json --out merged.json
```

`train` 的接线（另一个代理的改动）没有被覆盖；`collect` / `rl` 仍是 `not_implemented`（退出码 3）。

---

## 2. 状态清单（两档完全相同，先冻结后跑）

冻结文件：`.tmp_v420_data/T03/states_frozen.json`（schema `v420-counterfactual-plan/1`，
`frozen: true`，20 条，index 0..19），由**第一次 dry-run** 写出，之后两档都带
`--states-file` 读取它。

| # | seed | decision_frame | 层 | 选择理由 | 死亡前？ | 死亡帧 | 距死亡 | 并列宽度 | 祖先轨迹位置 |
| ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 0 | 10 | 60 | teacher_tie | 语料决策 15：18 个候选里 17 个分数完全相同（114.938981），选中的那个 119.938981（差 5.0） | 否 | – | – | 17 | 0.0045 |
| 1 | 10 | 480 | teacher_normal | 普通状态，固定网格槽 0（决策 120） | 否 | – | – | 0 | 0.0364 |
| 2 | 10 | 1600 | teacher_normal | 固定网格槽 1（决策 400） | 否 | – | – | 0 | 0.1212 |
| 3 | 10 | 2800 | teacher_normal | 固定网格槽 2（决策 700） | 否 | – | – | 0 | 0.2121 |
| 4 | 10 | 1240 | student_death_proximate | 旧模型首死（1541）前 300 帧 = 5.00 s | 是 | 1541 | 300 | 0 | 0.8047 |
| 5 | 10 | 1376 | student_death_proximate | 首死前 165 帧 = 2.75 s | 是 | 1541 | 165 | 0 | 0.8929 |
| 6 | 10 | 1508 | student_death_proximate | 首死前 30 帧 = 0.50 s（窗口最近端） | 是 | 1541 | 30 | 0 | 0.9786 |
| 7 | 11 | 60 | teacher_tie | 同上（seed 11 的并列行也是决策 15） | 否 | – | – | 17 | 0.0045 |
| 8 | 11 | 480 | teacher_normal | 固定网格槽 0 | 否 | – | – | 0 | 0.0364 |
| 9 | 11 | 1600 | teacher_normal | 固定网格槽 1 | 否 | – | – | 0 | 0.1212 |
| 10 | 11 | 2800 | teacher_normal | 固定网格槽 2 | 否 | – | – | 0 | 0.2121 |
| 11 | 11 | 1628 | student_death_proximate | 首死（1928）前 300 帧 | 是 | 1928 | 300 | 0 | 0.8444 |
| 12 | 11 | 1760 | student_death_proximate | 首死前 165 帧 | 是 | 1928 | 165 | 0 | 0.9129 |
| 13 | 11 | 1896 | student_death_proximate | 首死前 30 帧 | 是 | 1928 | 30 | 0 | 0.9834 |
| 14 | 12 | 60 | teacher_tie | 同上 | 否 | – | – | 17 | 0.0045 |
| 15 | 12 | 480 | teacher_normal | 固定网格槽 0 | 否 | – | – | 0 | 0.0364 |
| 16 | 12 | 1600 | teacher_normal | 固定网格槽 1 | 否 | – | – | 0 | 0.1213 |
| 17 | 12 | 1496 | student_death_proximate | 首死（1797）前 300 帧 | 是 | 1797 | 300 | 0 | 0.8325 |
| 18 | 12 | 1632 | student_death_proximate | 首死前 165 帧 | 是 | 1797 | 165 | 0 | 0.9082 |
| 19 | 12 | 1764 | student_death_proximate | 首死前 30 帧 | 是 | 1797 | 30 | 0 | 0.9816 |

配额（预注册，先写进配置，不是事后挑的）：`student_death_proximate = 9`（3 seed × 3 offset）、
`teacher_normal = 8`（3/3/2 轮转分配）、`teacher_tie = 3`（每 seed 1 个）。
每 seed 实际条数：10 → 7、11 → 7、12 → 6。

### 2.1 “旧模型首次死亡前 0.5–5 秒”是怎么得到的（不是猜的）

旧模型 = 冻结的 20k 学生 `.tmp_v410_wp05_train_beta05_20k/b0_s0_dagger_step20000.pt`
（sha `15ae1fd5…7215bd95`，731,802 参数，arm B0，obs-v3），经 v4.1.0 冻结 loader 加载。

**教师语料是另一条轨迹**（同一个 RESET 之后教师走的路），所以“学生首死前”的状态不可能从
教师 `labels.jsonl` 里读出来。本轮因此**先真跑一次学生自身轨迹**：

* `--probe-student`：`role="env"`（obs-v3 需要像素，故 xvfb），
  18 路 argmax、bomb 恒 0、逐帧 `STEP` 并记录该帧 `mask` + 引擎 `HASH` + `STATE`，
  首次死亡即停；
* 结果：seed 10 → **1541 帧**（23.9 s）、seed 11 → **1928 帧**（28.3 s）、
  seed 12 → **1797 帧**（27.7 s），`stop_reason=first_death`；
* RESET HASH 与语料 `hash_pre` **逐 seed 完全相同**
  （`c2b684a7a564c2d3` / `7c8705fc88f94c17` / `83d953750afaea5a`），
  说明学生轨迹与语料从**同一个** RESET 出发；`select_states` 把这一点做成硬检查
  （不一致就拒绝抽样）；
* 这一层是**诊断定位，不是评测**：3 个 seed 都是 D410-S1 语料 seed，而该 checkpoint
  就在这批语料上训练，所以这 3 个数字只能说明“它在这些种子上从哪里开始死”，
  不能当泛化成绩（trace 里 `not_an_evaluation: true`）。

3 个首死帧（1541 / 1928 / 1797）与 v4.1.0 记录的 20k 首死均值约 1,689 帧同量级，
是自洽的旁证，但本文件不用它作门。

### 2.2 去重与样本量限制

* 去重规则：**同一祖先轨迹内**两个状态 `decision_frame` 相距 < **20 帧**只保留优先级高的一个
  （优先级：死亡层最近端优先、普通层网格槽序、并列层最靠近顶部的 plateau）。
  不同祖先（同 seed 的教师轨迹 vs 学生轨迹）互不去重——它们是不同的状态。
* 实际结果：候选 20 个，`dropped=0`、`dropped_in_selection=0`（3 个死亡偏移 30/165/300 帧
  本身就相隔 ≥135 帧；网格槽相隔 ≥400 帧）。去重规则在这个样本上没有触发，
  但它是**规则**不是巧合：把 `dedup_min_gap_frames` 调到 400 时，抽样会拒绝
  同 seed 相邻偏移（单测钉住了这一点）。
* **祖先数量：3 条 seed 祖先 + 2 种轨迹类型 = 6 条不同记录轨迹**
  （`corpus:…seed10/11/12` 与 `student:seed10/11/12`）。
  **WP420-02 要求 ≥10 条互不相同的训练祖先，本轮不具备，也不假装具备。**
* 前缀深度上限 `max_prefix_frames = 4000`：普通层只用决策 120/400/700（帧 480/1600/2800），
  因为**每条分支都要自己 RESET + 重放前缀**（2 帧教师续行不能 fork，见 §3.1），
  前缀越深，360 条分支的固定成本越高。

---

## 3. 为什么这样做（结构性约束）

1. **`PARK` 只在 shadow 主进程合法，`CHILD` 只转发只读命令 + `STEP`**
   （`fork.cpp::ChildReadOnlyCommand` 没有 `PARK`/`SPAWN`）。2 帧教师续行必须在主进程里跑，
   所以每条分支都要付一次 `RESET` + 一次前缀重放；本轮实测 `RESET` 0.536 s、
   前缀重放 8,760–8,792 fps。`PILOT_COST.md` 的 `BRANCH`/`SPAWN` 省不下这笔钱。
2. **`record_child_tape()` 自己会 `probe_child_hazards()`，而该探针会 `PARK`**，
   顶掉 `decide` 刚取的模板 → `ERR spawn-template expected=2`。
   本模块在每个引擎进程 RESET 之后、第一次 `decide()` 之前跑一次
   `probe_capabilities()` + `probe_child_hazards()`（`TEACHER_2F_COST.md` §3.2 的修法），
   本轮 720 条分支没有出现该错误。
3. **数据目录一律用副本**：每个 seed 组把 `/root/v400-data/s1c/<shard>/env`
   复制到 `.tmp_v420_data/T03/run/<tag>/data-s<seed>/`，引擎只写副本；
   `/root/v400-data/**`、`.tmp_v410_*` 全程只读（`counterfactual` 只读 `labels.jsonl`、
   `shard.json` 与学生 trace）。跑完 `rmtree` 副本。
4. **教师配置只能库层自组装**：`tools/rl/v400_teacher.py` 没有任何 CLI 开关能改
   `replan_decisions`，所以用 `dataclasses.replace(get_config("fast"), replan_decisions=1)`。
   本轮没有走采集器（`v400_dagger_collect.py` 的 4 帧守卫会拒绝 2 帧节拍，且不在白名单内）。

---

## 4. 两档结果

状态集合**两档完全相同**（同一 `states_frozen.json`）。命令与环境见 §11。

### 4.1 H=32 档（既有短视界参照）

| 项 | 值 |
| --- | ---: |
| 状态 / 分支 | 20 / **360** |
| `queried` / `executed` / `engine_death` / `engine_alive` / `unknown` | **0 / 0 / 0 / 360 / 0** |
| 安全集合大小直方图 | `{18: 20}`（其余 0..17 全为 0） |
| 空安全集合数 | **0** |
| 全 18 安全数 | **20** |
| proper 集合数（0 < |S| < 18） | **0** |
| 教师续行下死亡数 | **0** |
| `unknown` 数 | **0** |
| 逐状态 HASH 校验（前缀） | **311,490 次 / 0 处不一致** |
| 末帧 HASH 互不相同数（每状态，18 条分支） | min 8 / max 18 / 均值 15.4 |
| 墙钟 | **599.3 s（9 min 59 s）**＝5 个 shard 的进程内合计（111.0–135.0 s/片）+ 合并 1.9 s；5 条命令各自 116.6–141.6 s（含解释器/进程启动） |

逐状态：20 个状态每个都是 `safe=18`、`engine_death=0`、`engine_alive=18`、`unknown=0`、
`queries=270`（18 × 15）、状态墙钟 **17.3–35.3 s**。

### 4.2 H=120 档

| 项 | 值 |
| --- | ---: |
| 状态 / 分支 | 20 / **360** |
| `queried` / `executed` / `engine_death` / `engine_alive` / `unknown` | **0 / 0 / 0 / 360 / 0** |
| 安全集合大小直方图 | `{18: 20}` |
| 空安全集合数 | **0** |
| 全 18 安全数 | **20** |
| proper 集合数 | **0** |
| 教师续行下死亡数 | **0** |
| `unknown` 数 | **0** |
| 逐状态 HASH 校验（前缀） | **311,490 次 / 0 处不一致** |
| 末帧 HASH 互不相同数 | min 8 / max 18 / 均值 15.8 |
| 墙钟 | **1,572.0 s（26 min 12 s）**＝5 个 shard 的进程内合计（276.4–371.9 s/片）+ 合并 0.8 s；5 条命令各自 291.1–387.0 s |

逐状态：每状态 `safe=18`、`engine_alive=18`、`queries=1062`（18 × 59）、
状态墙钟 **41.7–97.2 s**（浅的并列状态 42 s，深的学生死亡状态 92–97 s）。

### 4.3 两档对比（同一状态集合）

| 量 | H=32 | H=120 | 比值 |
| --- | ---: | ---: | ---: |
| 分支 | 360 | 360 | 1 |
| 教师查询 | 5,400 | 21,240 | 3.93× |
| 教师内部帧 | 345,600 | 1,359,360 | 3.93× |
| 外层分支帧 | 11,520 | 43,200 | 3.75× |
| 墙钟 | 599.3 s | 1,572.0 s | 2.62× |
| 每分支墙钟 | 1,665 ms | 4,367 ms | 2.62× |
| 死亡 | 0 | 0 | — |
| proper 集合 | 0 | 0 | — |

视界 ×3.75 只换来墙钟 ×2.62，因为 `RESET` + 前缀（360 × 0.536 s + 360 × 前缀）
在两档里是同一笔固定成本：H=32 时它占 246.8/599.3 = **41%**，H=120 时 247.9/1,572.0 = **16%**。
所以“短视界参照”在**每分支成本**上并不便宜 3.75 倍，只便宜约 2.6 倍。

---

## 5. 这个结果说明什么（区分度为零，是有效负结果）

1. **固定教师续行把首动作的影响抹平了**：候选只跑 2 帧（33 ms），之后教师每 2 帧重新规划一次，
   于是任何首动作造成的偏差在 2 帧内就被教师纠正。在 H=32/H=120 上，
   连“旧模型首死前 0.5 s”的状态也能被教师从**任意**首动作救回来。
2. 这与 WP420-02 §4 的描述一致，也正是它要求另外做**学生续行对照**的原因：
   “若教师续行安全、学生续行失败，优先查后续控制和状态覆盖”。
   本轮没有做学生续行（那是 50 状态主 pass 的事），所以本轮的 18 条分支测的是
   **教师可救性**，不是“这个首动作危险”。
3. **因此“安全集合 = 18”不是“18 个动作都同样好”**：它们只说明“在这 120 帧里都没死”。
   按计划的并列规则，这类状态可用于普通 BC，但**不给“区分危险”的信息量**，
   也不计入 proper。
4. 想让这类反事实产生分辨力，下一轮只有三条路（都已在计划里，不是新发明）：
   （a）更长的 H（教师仍会救，但代价线性上升）；（b）真正危险的状态（不是“学生死前 0.5 s”，
   而是学生已经无法恢复的状态）；（c）**学生续行**，让后续控制反映学生自己的错误。
   §8 把它写成 G420-1 的具体缺口。

---

## 6. 成本实测

### 6.1 教师与引擎

| 量 | H=32 | H=120 | 口径 |
| --- | ---: | ---: | --- |
| 教师查询 | **5,400** | **21,240** | 每分支 (H−2)/2 次（15 / 59），18 分支 × 20 状态 |
| 教师内部帧 | **345,600** | **1,359,360** | 实测 = 查询 × **64**（`horizon 32 × (1 + verify_k 1)`），逐档精确相等 |
| 引擎帧（前缀 + 分支） | **485,280** | **516,960** | 前缀 473,760 + 分支 11,520 / 43,200 |
| `RESET` 次数 / 合计 | 360 / 192.87 s | 360 / 193.81 s | **0.536 s/次** |
| 前缀重放 | 53.88 s（**8,792 fps**） | 54.08 s（**8,760 fps**） | 每条分支重放自己的前缀 |
| 教师墙钟 | 332.88 s（**61.64 ms/查询**） | 1,275.22 s（**60.04 ms/查询**） | 含 record+search+verify |
| 候选 2 帧 | 0.178 s | 0.178 s | 可忽略 |
| 总墙钟 | **599.26 s** | **1,571.95 s** | 三项相加 = 579.6 / 1,523.1 s，占 **96.7% / 96.9%** |

### 6.2 与 `TEACHER_2F_COST.md` 外推的对比

| 量 | 该文件外推/实测 | 本轮实测 | 读法 |
| --- | ---: | ---: | --- |
| 单次教师查询（`fast`，2 帧节拍） | 66.58 ms（depth 100）、86.18 ms（depth 1650） | **61.64 ms（H=32）/ 60.04 ms（H=120）** | 本轮主机空载；那份文件是争用（loadavg 12–19）条件下的上界 |
| 内部帧/查询 | 64（精确公式） | **64（精确）** | 一致，预算口径可用公式 |
| `RESET` | 0.704 s（中位） | **0.536 s** | 空载更快 |
| 前缀重放 | 5,861 fps（中位） | **8,760–8,792 fps** | 空载更快，与上一轮 §5 的 6,000–6,700 fps 同量级偏高 |
| 每分支（2 帧教师续行） | 43.7–61.4 ms 只是 fork 侧，不含每分支开局 | **1,665 ms（H=32）/ 4,367 ms（H=120）** | 结构不同：本模块每分支必须自己 RESET + 重放 |
| 20 状态 pilot 预测 | 约 21–35 s（fork 策略） | 不适用 | fork 策略不能做教师续行 |

**预注册成本模型 vs 实测**（模型见 `counterfactual.py::COST_MODEL`，来自上述两份成本文件）：

| 档 | 预测墙钟 | 实测墙钟 | 比值 | 预测查询 / 实测 | 预测内部帧 / 实测 |
| --- | ---: | ---: | ---: | --- | --- |
| H=32 | 696.8 s | 599.3 s | **0.86** | 5,400 / 5,400 | 345,600 / 345,600 |
| H=120 | 1,759.7 s | 1,572.0 s | **0.893** | 21,240 / 21,240 | 1,359,360 / 1,359,360 |

结论：**成本模型偏保守约 11–14%**（主要来自 `RESET` 0.704→0.536 s 与
`ms/query` 66.58→60–62 ms），符合“空载比争用快”的预期。按 200 状态 × 18 × H=120 外推：
**查询 212,400、内部帧 13.59 M、墙钟约 15.7 min × 10 = 2.6 h**（10 份 20 状态的线性外推，
不含每命令固定开销）。

### 6.3 预算占比（v4.2.0 README §8）

| 额度 | 上限 | 本轮用掉（两档合计） | 占比 |
| --- | ---: | ---: | ---: |
| 教师查询 | 650,000 | 26,640 | **4.1%** |
| 教师内部帧 | 64,000,000 | 1,704,960 | **2.7%** |
| 外层反事实续行帧 | 1,500,000 | 54,720 | **3.6%** |
| 引擎帧（含前缀/分支） | — | 1,002,240 | 记录备查 |

未触发 80% 停线。

---

## 7. 并列 / 几何 / 安全集合的处理声明

1. **几何打平不组成安全集合**：安全集合只装 `engine_alive` 与 `executed`
   （`contracts.safe_action_set`），`unknown` / `queried` / `engine_death` 一律不装。
   本模块另有一条测试钉住“未知永不变成死亡、也永不变成安全”。
2. **本语料在 argmax 上根本没有并列**：seed 10/11/12 的 3,300 / 3,300 / 3,298 行里，
   **0 行**的最高分被两个候选共享；最小 top-2 差恰为 **5.0**（决策 10–15）。
   所以“并列”只能按**分数平台**（plateau）取：本轮的并列层选的是
   **决策 15**——18 个候选里 **17 个**分数完全相同（114.938981），选中的那个 119.938981。
   这就是计划里说的“几何打平”的真实形态，`reason` 字段逐条记录。
3. **全存活状态的声明**：20 个状态在两档下都是 18/18 存活，
   `all_candidates_safe=20`、`proper=0`。按计划：**这些状态可作普通 BC 数据，
   但不提供“区分危险”的信息量**；它们不进 proper，也不改变分母让门通过。
4. **空 / 不完整 / 续行失败的状态不补造正例**：
   * 本轮 0 个空集合、0 个不完整状态（重跑后）；H=120 首次运行有 1 个状态（#11）
     因 1 条 unknown 分支而 `complete=false`、`safe=17`——那 17 不是“18 减 1 补出来的”，
     而是 17 条已验证存活 + 1 条**明确 unknown**；该状态**没有**被当成 proper 正例，
     也没有补任何标签。
   * `unknown` 的原因逐条写进 `unknown_reasons` 与分支的 `error`。
5. **`executed` 为什么是 0**：每帧都读 `STATE`，所以“是否存活”总是已知，
   分类落在 `engine_alive`（而不是“提交满 H 但没有存活读取”的 `executed`）。
   这是分类器的正常分支，不是漏统计。

---

## 8. G420-1 的当前状态：**不满足**，距离如下

`summary.g420_1` 由代码从数据算出（`counterfactual.g420_1_verdict`），不是手写结论：

| 门的要求 | 现状 | 还差 |
| --- | ---: | ---: |
| 200 个状态，全部 18 候选验证 | **20** | +180 个状态（即 +3,240 条分支） |
| ≥100 个 proper 有效状态（0 < |S| < 18） | **0** | **+100**，且**不是靠加状态能修**（见下） |
| 每个状态 18/18 分支完整 | 20/20（720 条分支全有裁决） | 已满足（试点规模） |
| 重复分支一致 | 3 对 3/3 逐比特相同（§9.2） | 门要的是 200 状态全量重复（0/20 计入） |
| ≥10 条互不相同的训练祖先 | **3** 条 seed 祖先（6 条轨迹） | +7 条祖先（当前语料只有 3 个 train seed shard） |
| 50 个失败相关状态的学生续行对照 | **0** | +50 个状态 × 至少 3 个首动作 |

**最关键的缺口是第 2 行，不是数量**：本轮 720 条分支零死亡、20 个状态全是 `|S| = 18`。
如果按同样的“固定教师续行 + H≤120”扩到 200 个同类状态，**proper 仍会是 0**。
要让 G420-1 可达，必须先做一件性质上不同的事，再扩量：

* **学生续行**（计划 §5 的 50 状态对照）：把“后续控制”从教师换成冻结学生，
  才可能看到“教师救得回来、学生救不回来”的区分度；
* 或**真正危险的状态**：不是“学生首死前 0.5 s”，而是学生已经进入不可恢复区域的状态
  （例如首死前 0–10 帧，或用分支结果筛出的高危状态）；
* 或**更长视界**：H=120 在教师手里太短；按本轮 4.37 s/分支线性估算，
  H=600 约 22 s/分支，200×18 就是 22 h，需要重排预算。

---

## 9. 两个控制实验（证明上面的 0 不是假的 0）

### 9.1 死亡检测正对照：用学生自己的后续动作

把“学生首死前 30 帧”的状态重放到 shadow，然后**不再问教师**，而是直接施加学生
**自己记录的**后续掩码，看 shadow 是否在记录的位置检出死亡：

| seed | 状态帧 | 记录首死帧 | shadow 检出死亡帧 | 死亡 HASH | 前缀逐帧 HASH 校验 | 结果 |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 10 | 1508 | 1541 | **1541**（+33 帧） | `8b573c0bf12ef1cc` | 1,508 / 0 不一致 | 一致 |
| 11 | 1896 | 1928 | **1928**（+32 帧） | `d84b6be24b1c099a` | 1,896 / 0 不一致 | 一致 |
| 12 | 1764 | 1797 | **1797**（+33 帧） | `ed0a9bec83a7d2f4` | 1,764 / 0 不一致 | 一致 |

死亡时 `deaths 0→1`、`lives=2.0`（正常残机死亡，不是 game over），
`_terminal_of` 判为 `first_death`/`engine_dead=True`。
**所以本轮 720 条分支的 0 死亡是“教师真的没死”，不是“检测器坏了”。**
（复算：`.tmp_v420_t03/death_control.py` → `.tmp_v420_data/T03/death_control.json`）

### 9.2 重复分支一致（试点规模）

同一 (状态, 候选) 连跑两次，比较**逐帧 HASH trace 的 sha256** 与裁决：

| 状态 | seed | 帧 | 层 | 候选 | 两次 trace sha256 | 裁决 | 一致 |
| ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| 0 | 10 | 60 | teacher_tie | 0 | `1c5479c4b8b38686` / 同 | engine_alive / 同 | 是 |
| 2 | 10 | 1600 | teacher_normal | 5 | `ee056ba9b783441d` / 同 | engine_alive / 同 | 是 |
| 6 | 10 | 1508 | student_death_proximate | 17 | `b19ac52fa29488e2` / 同 | engine_alive / 同 | 是 |

3/3 逐比特一致（同状态同候选，两次独立 RESET + 重放）。
（复算：`.tmp_v420_t03/repeat_check.py` → `.tmp_v420_data/T03/repeat_agreement.json`）

---

## 10. 一次真实故障与 fail-closed 证据（没有掩盖）

H=120 档第一次跑 `state-range 8:12` 时，状态 #11（seed 11、帧 1628）的 **cand 12**
前缀重放在**第 1 帧**就与记录不一致（1,628 处全不一致），其余 17 条分支同一状态下全部通过。

* 该分支**被判为 `unknown`**（`error` 原文记录在报告里），**没有**被算成死亡、
  **没有**被算成安全，状态标记 `complete=false`、`safe=17`；
* 该 shard `status=failed`、**退出码 5**；那份报告原样保留为
  `raw/h120_s8-12_FAILED.json`（也在证据 JSON 的 `incidents` 里）；
* 复现尝试：单独重跑同状态 cand 11/12/13 → **全部复现成功**
  （RESET HASH 与记录一致、1,628/1,628 帧 HASH 一致、停放点 HASH 一致），
  所以这是一次**不可复现的单次瞬态**，不是可重现的分支错误；
* 处置：给分支加了**显式 RESET-HASH 校验 + 一次干净重试**（此时尚未提交任何动作，
  重试不改变实验语义；第二次仍不一致就 fail-closed），然后重跑该 shard：
  **341.4 s、退出码 0、0 unknown、重试计数 0**（瞬态没有再次出现）。
* 该 shard 的报告里 `provenance.module_sha256 = cda7449e…`，其余四个 shard 是
  `ccbd91d7…`（改前）。差别只是“多了一次 RESET-HASH 校验与重试路径”，
  在重试计数为 0 时对分支结果无影响；合并报告 `shards[]` 逐 shard 记录了这一点，
  没有用一个 sha 代表两份代码。

---

## 11. 复算命令与退出码

全部用 `wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v420_t03/<x>.sh` 调用，
`ROOT420=/mnt/d/WorkShop/Touhou_RL`、`PY420=/root/v400-venv/bin/python`、
`V420_DATA=$ROOT420/.tmp_v420_data`。

### 11.1 学生首死探测（env 角色，需 xvfb）

| # | 命令 | 退出码 | 覆盖 |
| ---: | --- | ---: | --- |
| 1 | `PYTHONPATH=$ROOT420 xvfb-run -a -s "-screen 0 1024x768x24" $PY420 tools/rl/v420.py counterfactual --config configs/v420/counterfactual-pilot.json --out $V420_DATA/T03/student_probe.json --execute --probe-student` | **0** | 3 seed 学生首死 1541 / 1928 / 1797，23.9 / 28.3 / 27.7 s |

### 11.2 状态冻结（dry-run，不起引擎）

| # | 命令 | 退出码 |
| ---: | --- | ---: |
| 2 | `env -u DISPLAY $PY420 tools/rl/v420.py counterfactual --config configs/v420/counterfactual-pilot.json --out $V420_DATA/T03/states_frozen.json` | **0**（20 状态，可用，成本估算 5 shard × 5.85 min @H=120） |

### 11.3 H=32 档（每条命令都远低于 10 分钟上限）

| # | 命令（省略公共前缀 `$PY420 tools/rl/v420.py counterfactual --config configs/v420/counterfactual-pilot.json`） | 退出码 | 墙钟 |
| ---: | --- | ---: | ---: |
| 3 | `--out raw/h32_s0-4.json --execute --horizon 32 --state-range 0:4 --states-file states_frozen.json` | 0 | 116.6 s |
| 4 | 同上 `--state-range 4:8` | 0 | 120.6 s |
| 5 | 同上 `--state-range 8:12` | 0 | 135.9 s |
| 6 | 同上 `--state-range 12:16` | 0 | 116.7 s |
| 7 | 同上 `--state-range 16:20` | 0 | 141.6 s |
| 8 | `--out pilot_h32.json --merge raw/h32_s{0-4,4-8,8-12,12-16,16-20}.json` | **0** | 1.9 s / 0.5 s（两次合并，第二次为最终） |

### 11.4 H=120 档

| # | 命令（同上前缀） | 退出码 | 墙钟 |
| ---: | --- | ---: | ---: |
| 9 | `--out raw/h120_s0-4.json --execute --horizon 120 --state-range 0:4 --states-file states_frozen.json` | 0 | 291.1 s |
| 10 | 同上 `--state-range 4:8` | 0 | 327.0 s |
| 11 | 同上 `--state-range 8:12` | **5**（1 条 unknown，见 §10） | 339.1 s |
| 12 | 同上 `--state-range 12:16` | 0 | 300.6 s |
| 13 | 同上 `--state-range 16:20` | 0 | 387.0 s |
| 14 | `--out pilot_h120.json --merge ...` | **5**（继承 #11 的失败） | 0.8 s |
| 15 | **重跑** `--state-range 8:12`（加 RESET-HASH 校验后） | **0** | 341.4 s |
| 16 | `--out pilot_h120.json --merge ...` | **0** | 0.9 s / 0.8 s（最终） |

### 11.5 控制与复核

| # | 命令 | 退出码 | 覆盖 |
| ---: | --- | ---: | --- |
| 17 | `env -u DISPLAY $PY420 .tmp_v420_t03/death_control.py` | 0 | §9.1 死亡检测正对照 |
| 18 | `env -u DISPLAY $PY420 .tmp_v420_t03/repeat_check.py` | 0 | §9.2 重复一致 3/3 |
| 19 | `env -u DISPLAY $PY420 .tmp_v420_t03/diag_reset.py` | 0 | §10 瞬态不可复现 |
| 20 | `$PY420 .tmp_v420_t03/check_reports.py` | 0 | 18/18 首帧 HASH 互异、安全集合复算一致、帧号自洽、引擎帧 = 前缀+分支 |
| 21 | `$PY420 .tmp_v420_t03/make_evidence.py` | 0 | 写出 `counterfactual_pilot.json` |

### 11.6 派发里的验收命令：跑了哪几条、为什么分片

派发给出的三条验收命令与本轮实跑的关系：

| 派发命令 | 本轮 | 说明 |
| --- | --- | --- |
| `pytest ... tests/test_v420_counterfactual.py` | **原样跑，退出码 0，46 passed** | 见 §12.1 |
| `tools/rl/v420.py counterfactual --config configs/v420/counterfactual-pilot.json --out $PWD/.tmp_v420_data/T03/pilot.json --execute` | **dry-run 原样跑，退出码 0**（写出冻结清单 `pilot.json`）；`--execute` 形式按 5 个 `--state-range` 分片 + `--merge` 跑 | 这条命令在 H=120 下要跑全部 20 状态 = **约 26 min**，超过派发自己规定的“单条命令不超过 10 分钟”；分片的 argv/退出码/墙钟见 §11.3–11.4，实测每片 4.6–6.5 min。分片只切状态，不切实验：同一 config、同一 `--states-file`、同一 H=120 |
| `tools/rl/v420.py --help` | **原样跑，退出码 0** | 8 个子命令齐全 |

（`--execute` 的合并结果就是 `pilot_h120.json`；`pilot.json` 是 dry-run 写出的冻结清单，
两者内容不同但都在 `.tmp_v420_data/T03/` 下，文件名不冲突。）

---

## 12. 测试与边界

### 12.1 两侧测试统计（本轮实跑）

| 侧 | 命令 | 结果 | 退出码 |
| --- | --- | --- | ---: |
| WSL | `PYTHONPATH=$PWD /root/v400-venv/bin/python -m pytest -p no:cacheprovider --basetemp $PWD/.tmp_v420_data/tests/T03 -q tests/test_v420_counterfactual.py` | **46 passed** in 12.14 s | **0** |
| WSL | 基线六文件 `test_v420_{contracts,replay_audit,eval,rebuild,reslot,dataset_2f}.py` | **197 passed** in 14.81 s | **0** |
| WSL | 全部 v420 测试文件（含本文件与 T04 的 `train`） | **267 passed** in 30.59 s | **0** |
| Windows | `python -m compileall -q touhou_rl\v420` | 无输出 | **0** |
| Windows | `python -m pytest --basetemp .tmp_v420_pytest_win7 -q tests/test_v420_counterfactual.py` | **46 passed** in 8.99 s | **0** |
| Windows | 基线六文件 | **196 passed, 1 skipped** in 2.42 s | **0** |
| 两侧 | `tools/rl/v420.py --help` | 8 个子命令都在 | **0** |

（Windows 那 1 个 skip 是基线文件里既有的平台 skip，与本轮改动无关；
改动前后 `counterfactual` 的差别是：改前 `--help` 里没有本子命令、`counterfactual`
任何调用都返回 `not_implemented` / 退出码 3，改后 dry-run 退出码 0。）

### 12.2 测试覆盖（`tests/test_v420_counterfactual.py`，46 个，无引擎/无游戏/无 numpy）

合成语料与合成学生 trace 写在 `tmp_path` 里。覆盖：RLE 往返与畸形拒绝、
语料 hash 链断裂拒绝、行掩码数量错误拒绝、学生 trace 缺口与错名拒绝、
分层覆盖（普通/死亡前/并列三类都在、三个 seed 都在、偏移落在 0.5–5 s 内）、
`decision_frame` 必为块边界且 HASH 有记录、20 帧去重规则（含把 gap 调到 400
验证规则本身）、预注册校验（视界、候选子集、配额求和、dev seed 拒绝）、
成本模型（查询数 = (H−2)/2 × 分支、内部帧 = 查询 × 64）、
`unknown` 永不变成死亡也永不进安全集合、空/全安全/proper 三分、unknown 原因归并、
`g420_1_verdict` 不把试点写成门、合并拒绝重叠/混视界/缺口/不同冻结清单、
冻结清单往返与缺口拒绝、`trim_report` 只去 trace 保留 digest、
以及“模块顶层不 import numpy/torch/引擎”。

### 12.3 未做（明确记录，不当作已做）

* 50 状态“冻结学生续行”对照（WP420-02 §5）——这是让本轮结果产生区分度最直接的下一步；
* 200 状态全量、逐状态 18 候选之外的任何扩展；
* 像素学生的分支续行（`PILOT_COST.md` §9.3 已判定只能在 env 角色重放，
  50 状态 × 3 臂 ≈ 35 min–3.8 h，不在 20 状态试点预算内）；
* 跨面/低余量/贴边/资源操作/面过渡等其余分层（本轮只有普通/死亡前/并列三类）；
* 全量重复一致（只做了 3 对 2 次，见 §9.2）。

### 12.4 不改动声明

旧 ABI、`touhou_rl/v420/{contracts,replay_audit,eval,rebuild,reslot,dataset_2f,batch_2f,train,diagnostics,__init__}.py`、
`configs/v420/{audit-time,legacy-reproduce,learn-controlled,local,protocol}.json`、
`tools/rl/v410_student_eval.py`、`touhou_rl/v400|v410/**`、旧证据、`ASSET_SYNC.*` 全程未改；
`.tmp_v410_*` 与 `/root/v400-data/**` 只读（引擎只写自己的副本）；
confirm 种子 61000–61499 与保留段 60000–60199 全程未读；状态只来自 seed 10/11/12（train split）。
本轮新增/改动的文件只有：`touhou_rl/v420/counterfactual.py`、`tools/rl/v420.py`
（只接 `counterfactual`）、`configs/v420/counterfactual-pilot.json`、
`tests/test_v420_counterfactual.py`、本目录两个新证据文件、
`.tmp_v420_t03/`（脚本）与 `.tmp_v420_data/T03/`（运行数据）。
