# T08 / CLEAR_REACHABILITY：评测器的动作通道能否触发 `stage_clear`？

对象：v4.2.0 `T08`。问题来自 [`T06/RL_AUDIT.md`](../T06/RL_AUDIT.md) §5.1 的实测
（同一段录制动作：不加脉冲 13,272 帧、分数停在 141,379、`stage_clear` 永不触发；
加"每 48 决策松一帧 SHOOT"后 13,195 帧过关、分数 2,631,430），与主代理对
`tools/rl/v410_student_eval.py` 的观察（全文件 `shoot`/`pulse` 零命中）之间的矛盾。

**本档只做诊断，不改任何实现**：`touhou_rl/**`、`tools/**`、`configs/**`、`tests/**`
逐字节未动（`git status` 里本档只新增 T08 两个文件与 `.tmp_v420_clearcheck/`）。
全部数字由 [`clear_reachability.json`](clear_reachability.json) 承载，该 JSON 由脚本从
两次真实引擎回放的产物生成，不含手抄数字。

> **一句话判定：`stage_clear` 在评测器的动作通道下不可达（机制上不可能被报告）；但既有
> 0/N 过关数字一条都不受它影响——因为所有被评测的 episode 都在 ≤ 4,964 帧内以死亡结束，
> 离过关所需的 13,195 帧还差 8,231 帧以上。**

---

## 1 问题一：评测器实际发出的 SHOOT 状态是什么

**答案：每一帧都按住 SHOOT（`0x0001`），由 `cand_to_mask` 无条件 OR 进去，默认值就是
"开"，评测器里没有任何释放路径。**

| 环节 | 位置 | 事实 |
|---|---|---|
| 决策 → 掩码 | `tools/rl/v410_student_eval.py:967` | `mask = gym_actions.cand_to_mask(cand, 0)` —— 全文件唯一构造掩码的地方 |
| 像素预热 | `tools/rl/v410_student_eval.py:941` | `client.step(gym_actions.cand_to_mask(0, 0), 1)`，同一个构造器 |
| 唯一的 STEP | `tools/rl/v410_student_eval.py:975` | `step_reply = client.step(mask, want)`，`want=1`、每决策 2 帧（`decision_steps("consecutive",2) -> (2,1)`，见 `tools/rl/v400_student_eval.py:191-213`） |
| 掩码构造器 | `touhou_rl/v400/gym/actions.py:58,70` | `def cand_to_mask(cand, bomb=0)`；`mask = _MOVE_BITS[move] \| BTN_SHOOT`，**SHOOT 无条件置位**，唯一可调位是 `bomb` |
| 解码器反向约束 | `touhou_rl/v400/gym/actions.py:86` | `mask_to_cand` 对不含 SHOOT 的掩码直接拒绝："not representable as a candidate" |
| 上线路 | `touhou_rl/v400/gym/protocol.py:694` | `STEP {mask} {frames}`，掩码原样写给引擎 |
| 引擎角色 | `tools/rl/v400_student_eval.py:168-169` | `--data-dir … --rl-control 0,1 --rl-role env`（v410 第 222 行直接继承这个 client） |
| 协议语义 | `Document/Update/Alpha/v4.0.0/10-P0-训练场.md:187,190,235,237` | `--rl-control <fd_in>,<fd_out>` 进 `RlLoop`（读命令→执行→回话）；智能体掩码白名单 `0x00F7`，**ENTER(0x1000) 被清掉**；换面需要 `SHOOT\|ENTER` 的**上升沿**，按住不算 |

**释放路径核查（真实命令与退出码，见 `.tmp_v420_clearcheck/evidence_greps.log`）**

```text
grep -nE "shoot|SHOOT|pulse|BTN_" tools/rl/v410_student_eval.py   -> exit 1（0 命中）
grep -nE "shoot|SHOOT|pulse|BTN_" tools/rl/v400_student_eval.py   -> exit 1（0 命中）
grep -rn release_due --include=*.py touhou_rl tools               -> exit 0，10 处命中，
     全在 teacher/collector/训练侧（teacher/shoot.py、v420/{rl,counterfactual,dagger}.py、
     tools/rl/v400_teacher.py、tools/rl/v4{00,10}_dagger_collect.py），**两评测器各 0 处**
```

掩码枚举（`.tmp_v420_clearcheck/masks.log`，`32_masks.py` 退出码 0）：`cand=0..17` 的
18 个掩码**全部**带 `0x0001`，`candidates whose mask clears SHOOT: [] (count 0)`。
所以评测器能发出的动作集合在结构上不含"松 SHOOT"这一项。

**顺带证伪了文档里的 ENTER 注入**：`10-P0 §4.2` 第 237 行说"处于等待换面的状态时 harness
每隔一帧注入一次 ENTER 脉冲"。本轮 15,564 帧里没有任何一帧出现非智能体位；
`touhou_rl/v400/teacher/shoot.py` 的 docstring 早已指出 `rlharness.cpp` 没有 `0x1000` 路径。
**结论：上升沿只可能来自客户端主动松 SHOOT，而评测器没有这条路径。**

---

## 2 问题二：决定性实验（同一条录制轨迹，两种通道）

### 2.1 输入

RL_AUDIT §4 用的那条**唯一真实过关录制**（先按证据定位，再校验 sha）：

```text
.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed12-shard/slots.jsonl
sha256 f759de7dacc78343dec8825da5de14942e749b8a5a2c5319304e513d1618b7b9  (与 configs/v420/rl-engine-audit.json audit.clear_shard_sha256 一致)
```

动作表由**训练侧自己的加载器** `touhou_rl.v420.rl.load_clear_schedule` 重建（slot0 的 8 个
前导决策 + 6,588 个 `executed_cand` + 64 个尾部重复 = **6,660 个决策**，seed 12、stage 0、
difficulty normal）。两条通道**动作表完全相同、帧预算相同（19,887，即评测器自己的上限）、
STATE 逐帧读取**，唯一差别是每 48 个决策的最后一帧是否丢掉 `SHOOT|BOMB`。

### 2.2 命令与退出码

| 步骤 | 命令 | 退出码 |
|---|---|---|
| 私有 data 目录 | `bash .tmp_v420_clearcheck/00_setup_data.sh` | 0（`setup-exit=0`，两份 474 MB `cp -a` 到 `.tmp_v420_data/T08/clearcheck-{eval,pulse}`） |
| A 评测器通道 | `bash .tmp_v420_clearcheck/10_replay.sh eval 19887 clear_eval` | 0（`replay_exit=0`） |
| B 加脉冲 | `bash .tmp_v420_clearcheck/10_replay.sh pulse 19887 clear_pulse` | 0（`replay_exit=0`） |
| A+B 连跑 | `bash .tmp_v420_clearcheck/20_full.sh` | 0（`=== done eval_rc=0 pulse_rc=0 ===`） |
| 重复跑（确定性） | `bash .tmp_v420_clearcheck/21_repeat.sh` | 0（`done eval_rc=0 pulse_rc=0`） |
| 数字提取 | `python .tmp_v420_clearcheck/40_analyze.py` | 0 |
| 既有产物扫描 | `python .tmp_v420_clearcheck/scan_existing_evals.py --out …` | 0 |

回放用的是**评测器自己的 client**（`tools/rl/v400_student_eval.py::EnvHarnessClient`，
v410 第 222 行 import 的同一个类）、评测器自己的 `decision_steps`、评测器自己的
`cand_to_mask`；脚本 `.tmp_v420_clearcheck/replay_clear.py` 只读这些模块，不改它们。

### 2.3 三个数字

| 量 | **A 评测器通道（不加脉冲）** | **B 同一轨迹 + 每 48 决策松 1 帧** |
|---|---|---|
| 是否触发 `stage_clear` | **否**（`stage_end` 仍是 0） | **是**（`stage_start 0 -> stage_end 1`，`stop_reason=stage_clear`） |
| 帧数 | **15,564**（越过 13,195 之后又跑了 2,369 帧，仍不过关） | **13,195** |
| 分数 | 停住时 **163,945**；**第 13,195 帧时正是 141,379**；第 13,272 帧 141,659 | **2,631,430** |
| 结束原因 | `game_over`（deaths 3，lives 0） | `stage_clear`（deaths 0） |
| SHOOT 统计 | 15,564/15,564 帧全部按住；**上升沿 1 次**（只有开局那一次） | 13,058 帧按住、137 帧松开；**上升沿 138 次** |
| 决策数 | 7,782（含 1,122 个尾部重复） | **6,598** |
| 跑分冻结 | 分数 141,379 从第 **5,058** 帧冻结到第 **13,245** 帧（8,187 帧无任何得分） | 最长冻结只有 1,599 帧（5,058–6,657），随后继续得分并过关 |

**三条独立交叉验证**（说明这确实复现了既有那条过关轨道，不是新造的轨迹）：

1. B 的 13,195 帧 / 2,631,430 分 / 6,598 决策，与 [`RL_AUDIT.md`](../T06/RL_AUDIT.md) §4
   表格逐字一致，也与 [`reward_r3.json`](../T06/reward_r3.json) 的 `classes.clear`
   （`frames 13195`、`score 2631430.0`、`decisions 6598`）逐字一致；
2. A 的分数停住值 **141,379** 与文档记录的无脉冲值完全相同；
3. 两次独立运行（run1/run2）的 `frames`、`score`、`stop_reason`、掩码轨迹 sha256
   **逐项相同**（`determinism.identical = true`）。

**关于文档里的 13,272**：本轮 A 跑到 15,564 帧才 `game_over`，第 13,272 帧时分数已经
从 141,379 微升到 141,659。所以 13,272 是当年那次测量自己的"停在这里"的帧号（停止口径），
不是机制差异；**分数停在 141,379 这个事实两边完全一致**，而过关与否的差别是类别差别，
不是帧号差几十帧。

### 2.4 机制观察（本实验测到但未分离的部分）

两条通道在第 **2,044** 帧首次出现分数差异（B 的第一个松 SHOOT 帧在第 96 帧，此后
子弹/击杀时序混沌分叉）。A 通道在第 5,058 帧起**整整 8,187 帧没有任何得分**，`stage`
仍为 0；B 通道同一帧位置也进入一次静默，但只用 1,599 帧就在第 6,657 帧恢复并一路打到
第 13,195 帧过关。也就是说，**按住 SHOOT 卡住的不只是最后那次换面，还有中途的一次等待
状态**（候选解释：对话等待，见 `10-P0 §4.2` 第 238 行"对话用 SKIP、不生效就改用 SHOOT
脉冲"；`teacher/shoot.py` 的 docstring 则描述为"run survives, the playfield stays empty"）。
本实验**没有**把"对话等待"与"换面等待"分离开，只测到"只有加脉冲的通道能离开并过关"。

---

## 3 问题三：修复方案与最小验证（只写方案，不改实现）

### 3.1 方案

改动点只有两处 STEP 循环（评测器是唯一出口，两代评测器同一个 bug）：

* `tools/rl/v410_student_eval.py`，`run_episode` 的 STEP 循环（第 975 行）；
* `tools/rl/v400_student_eval.py`，同一循环（第 511 行）。

规则与训练侧完全一致（复用冻结实现，不新写一套）：

```python
from touhou_rl.spec import BTN_BOMB, BTN_SHOOT
from touhou_rl.v400.teacher.shoot import release_due

release = release_due(steps, SHOOT_PULSE_PERIOD)     # (i + 1) % period == 0
...
for _unit in range(units):
    step_mask = mask
    if release and _unit == units - 1 and _unit > 0:  # 每决策最后一帧
        step_mask = mask & ~BTN_SHOOT & ~BTN_BOMB
    step_reply = client.step(step_mask, want)
```

* `SHOOT_PULSE_PERIOD = 48`：与 `configs/v420/rl-engine-{audit,pilot}.json` 的
  `rollout.shoot_pulse_period = 48` 一致（48 决策 × 2 帧 = 96 帧，与
  `touhou_rl/v420/rl.py:1026,1040` 的训练路径同一个数；`teacher/shoot.py` 的默认
  24 是给 4 帧决策用的，不能直接搬到 2 帧节拍）；
* 只在 `units > 1` 时松（`consecutive` 窗口下每决策 2 个 STEP，与训练一致）；
* **必须同时**把 `shoot_pulse_period` 写进 episode identity，并让
  `--merge-summaries` 拒绝混合不同脉冲的记录——否则修好前后的行不可比；
* 修复前后的记录**只能重跑，不能混表**。

### 3.2 最小验证（已执行）

`replay_clear.py --mode pulse` **就是**这个补丁在 `tools/` 之外的等价实现：同一个 client、
同一个 `decision_steps`、同一个掩码构造，只有"每 48 决策的最后一帧丢掉 `SHOOT|BOMB`"这一
条差别。结果：

| 期望 | 实测 | 判定 |
|---|---|---|
| `stage_clear = true` | `true` | pass |
| `frames = 13,195` | `13,195` | pass |
| `score = 2,631,430` | `2,631,430` | pass |

（`99_verify.sh` 的 invariants 段 `ALL_PASS`，退出码 0。）

**所以"加了脉冲就能过关"在评测器同一控制协议上得到验证；"不加脉冲永远不过关"也在同一
条轨迹上得到验证（15,564 帧、3 条命死光、`stage` 始终 0）。**

---

## 4 受影响范围（不重测，只给清单）

方法：把**冻结评测器写过的每一个 episode 记录**扫一遍（87 个文件、含 per-shard 与 merged
的重复计数共 1,285 条记录；confirm 种子 60000–60199 的记录一律跳过），看有没有哪条
episode 真的跑到过"等待换面"的位置（≈13,195 帧）。详见
[`.tmp_v420_clearcheck/existing_evals_scan.json`](../../../../../../.tmp_v420_clearcheck/existing_evals_scan.json)。

**扫描边界（刻意排除，且不引用其中任何数字）**：

* **教师 S1N 汇总（G410-1，`tools/rl/v400_teacher.py`）**——那是**另一条通道**，而且
  `tools/rl/v400_teacher.py:442` 本来就用 `release_due` 松 SHOOT；这些文件里带
  **confirm 种子 61000–61499**，本任务不得读取，因此显式排除。它作为**正向对照**只用
  已公开的结论（`AGENTS.md` §8.1：G410-1 教师 100/100 S1N 过关），不重读任何一行记录。
* 训练侧 Dagger/RL rollout（`GameEnv2F`，`shoot_pulse_period=48`）：本来就有脉冲，
  不属于"评测器通道"。

| 文件 | 数字 | 实测帧数 | 需要重测？ |
|---|---|---|---|
| `v4.1.0/evidence/WP04/EVALARMS.md` | B0/B1/B2/B3 各 **0/20**（共 80 局） | 743–3,021，全部 `death_stop` | **否**（离 13,195 差 ≥10,000 帧） |
| `v4.1.0/evidence/WP05/EVAL_DAGGER.md` | DAgger **0/20** | 743–4,048，全部 `death_stop` | **否** |
| `v4.2.0/evidence/T06/RL_PILOT2.md` | R0 **0/20**、R2 **0/20**（s1n 与 diag_first_death 两个口径） | 733–4,964，`game_over`/`death_stop` | **否** |
| `v4.2.0/evidence/T06/{r2_collapse,r2_kl_p3,rl_kl_anchor,advnorm_off}.json` 与 `dev-kl/dev-r3/dev-r3b200` | 各轮 **0 过关** | 816–4,701 | **否** |
| `v4.2.0/evidence/T06/RL_AUDIT.md` §4 | clear 类 **13,195 帧 / 2,631,430** | 13,195 | **否**（走的是训练环境 `GameEnv2F`，`shoot_pulse_period=48` 本来就在） |

**汇总：1,285 条记录里 `stage_clear` 只有 1 条，就是 RL_AUDIT 的 clear 类（训练通道回放）；
除它之外最深的评测 episode 是 4,964 帧，离过关长度差 8,231 帧。**
因此**受影响的既有数字＝空集**：这条机制 bug 至今没有把任何一次"本可过关"误判成"不过关"。

**它会在什么时候开始咬人**：任何一次评测里如果有策略活到等待状态（这条轨道上是
5,000+ 帧，过关是 13,195 帧），评测器都会把它记成 `frame_cap`/`game_over` 且
`stage_clear=false`——也就是说**修好之前，评测器结构上无法报告一次过关**，0/N 里的
"N" 目前是"0 次过关，且不可能报告 1 次过关"。

---

## 5 不能主张什么

* **本档没有重测任何能力数字**：没有训练、没有 checkpoint 评测、没有 dev seed 运行；
  0/N 一个都没变，也一个都不需要因为本 bug 而变。
* **录制轨迹不是策略**：用它回放只能证明"评测器通道坏了"，不能证明任何学生能过关；
  也不构成"修好后学生就能过关"的证据——修复只是让"过关"这件事**可被报告**。
* **不构成对 0/N 结论的推翻**：本档的判定是"机制不可达、但既有结论不受影响"，
  不是"既有 0/N 是测量伪影"。
* **没有分离"对话等待"与"换面等待"**：只测到按住 SHOOT 时打分区冻结 8,187 帧。
* **没有碰** confirm 种子 60000–60199、D410-S1 corpus 种子 1–20、dev 种子（仅只读既有
  记录）、没有 gate/Wilson/开发成绩主张。
* 没有修改任何 `tools/**`、`touhou_rl/**`、`configs/**`、`tests/**`；修复方案是**提议**，
  代码一行未改。

---

## 6 复现

```bash
ROOT420=/mnt/d/WorkShop/Touhou_RL; PY420=/root/v400-venv/bin/python
cd $ROOT420
bash .tmp_v420_clearcheck/00_setup_data.sh                 # 私有 data 目录（cp -a）
bash .tmp_v420_clearcheck/30_evidence_greps.sh             # Q1 行号级证据（grep 退出码在日志里）
$PY420 .tmp_v420_clearcheck/32_masks.py                    # 18 个候选掩码全带 SHOOT
bash .tmp_v420_clearcheck/20_full.sh                       # A/B 两次真实回放（约 4 分钟）
bash .tmp_v420_clearcheck/21_repeat.sh                     # 重复跑，验证确定性
$PY420 .tmp_v420_clearcheck/40_analyze.py                  # 三个数字
$PY420 .tmp_v420_clearcheck/scan_existing_evals.py \
    --out .tmp_v420_clearcheck/existing_evals_scan.json    # 受影响范围
$PY420 .tmp_v420_clearcheck/50_emit_evidence.py \
    --out Document/Update/Alpha/v4.2.0/evidence/T08/clear_reachability.json
bash .tmp_v420_clearcheck/99_verify.sh                     # 断言 + 交付物 sha256
```

引擎：`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`
（sha256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`）；
env 角色 `xvfb-run -a -s "-screen 0 1024x768x24"`；data 目录是
`.tmp_v420_data/T08/clearcheck-{eval,pulse}`（`cp -a` 的私有副本，**未写**
`/root/v400-data`、`.tmp_v410_*`、`.tmp_v420_data/{T02c,T02d,T04,T05,T06}/**`）。
