# P2 有效世界上的教师能力审计与训练/评估分组冻结（round19）

## 一句话

在 R18 冻结的 **28 个"非常量可解"的 development 世界**上，把两个教师原样放进去跑：**旧 `TacticalTeacher` 只过 14/28（F1 0/12、F2 4/4、F3 10/12），`TacticalTeacherV2` 过 28/28**；两边合计 **42 次成功里没有任何一次是"动作恒定"（退化 0 次）**，旧教师在 F1 的 4 个恒定 episode 全部失败并被单独标注为"恒定但未成功"；**没有任何一族出现"两个教师都 0 可行"**（旧教师在 F1 全灭，但 V2 在该族 12/12），单教师的失败按原样保留、未放宽判据、未换世界、未调教师；每族都有 R18 参考见证，**28/28 见证用独立 `verify()` 重放通过**；六条冻结基线 **168/168 行完整复现** manifest；14 组按族内索引交替冻结为 **train 7 组 / validation 7 组（事件 14/14）**，同组同 split、镜像与本体同组、零泄漏。

**这是合成 development 世界、未训练策略（训练步数 0、无 optimizer、不导入 `TacticalPolicy`）、无 checkpoint（`checkpoint = null`）、未开正式门、不运行游戏、不产生教师标签。** 教师是**先验规则/前瞻规划器**，不是学习到的策略。

## 隔离与边界

- 只新增 4 个路径：[本轮工具](<../../../../../tools/rl/v373_valid_teacher_eval.py>)、[本轮测试](<../../../../../tests/test_v373_valid_teacher_eval.py>)、本报告、[evidence 目录](<../evidence/validteacher-round19/preregistration.json>)。
- **不改旧模块**：`tactical_teacher.py`、`tactical_teacher_v2.py`、`tactical_teacher_validation.py`、`tactical_scenarios.py`、`tactical_synthetic_closed_loop.py`、`tactical_data.py`、`tools/rl/v373_world_filter.py` 逐个在预注册里冻结 SHA，报告时全部未变（`identities_unchanged = true`）。
- **不改 R18 证据**：`worldfilter-round18/run/manifest.json`、`run/sha256.json`、`preregistration.json` 与 `reports/P1-worldfilter-round18.md` 四个文件 SHA 一并冻结；28 个世界**只读加载**，不重新生成、不改参数、不换世界、不加预算。
- **没有 test split**（`test_split = false` 有断言，加载阶段就拒绝任何非 `development` 的 scenario）；**不评估任何 checkpoint 或策略**（`policy_evaluations = 0`）；**不训练**（`training_steps = 0`）；**不运行游戏**（`game = false`）。
- 唯一的教师接触面：对每个世界、每个教师调用 `decide()`，**只读**；不写任何标签文件。

## 预注册判据（先冻结、再跑，写进 `preregistration.json`）

执行顺序是 **测试 → 预注册 → 测试 → 唯一一次 run → 测试**。判据写死在 [preregistration.json](<../evidence/validteacher-round19/preregistration.json>) 的 `criterion`（SHA `48a8ccb24846c9194e29c81ae10cb77070aa2eac02ebf0e6a203c9629d8aa605`），冻结 **13 个身份 SHA**（本轮工具/测试、旧世界过滤器、六个旧模块、R18 manifest/sha256/prereg、R18 报告）。`verify_prereg()` 在 run 之前重新推导 `te.criterion()` 并逐字段比对，**判据漂移就直接拒绝执行**；manifest 的 SHA 也在加载时与 R18 的 `run/sha256.json` 对账。

| 项 | 内容（预注册原文归档在 criterion 中） |
|---|---|
| 世界 | 只读加载 R18 `run/manifest.json` 的 28 个世界 = 14 组 × 镜像 2；F1 12 / F2 4 / F3 12；组 6 / 2 / 6；**不重生成、不改参数、不换世界** |
| 闭环 | 与 R18 完全一致：`tactical_teacher_validation.advance`，每决策 2 逻辑帧，horizon 160 = 80 决策，sticky 碰撞，**无像素、无渲染、无游戏** |
| 教师 | `TacticalTeacher`、`TacticalTeacherV2` **原样**，只调用 `decide()`；**驱动器拥有物理**（与独立 `verify` 共用同一个 `advance`），不用两个教师自带的 `rollout()`（原因见"做法 2"） |
| survived | 跑满 160 帧且无碰撞 |
| goal_satisfied | F3：某个采样状态 `canonical_x <= goal`；F1/F2：按 R18 `outcome()` 口径恒为真（它们没有越线目标） |
| feasible | `survived and goal_satisfied`，与 R18 `outcome()` 逐字段同义 |
| first_hit | 首个 sticky 碰撞的逻辑帧，从未碰撞为 null |
| 独立 verify | `tactical_teacher_validation.verify(world, actions)` 对完整 80 动作**非提前终止**重放：feasible / rejected_witness / unknown；与记录值不一致就抛错 |
| 恒定/退化 | 80 个决策里**只有一个不同动作**即"恒定"；若同时 feasible 则标 `degenerate`，**单独报告、绝不计入教师能力**；`capability_success = feasible and not 单一动作` |
| 恒定交叉核对 | 每个恒定 episode 必须与 R18 `extra_constants` 探针一致（同一动作的常量基线），不一致直接 abort |
| 动作改变 | `switches` = 80 个决策里相邻动作不同的次数；按教师、按族报 total/mean/median/max/有改变的事件数 |
| 族判定 | 有 ≥1 次 `capability_success` 为 `capable`，否则 `not_capable`；两个教师都 `not_capable` 记 `both_failed` **并保留为失败**，旁边给出该族参考见证数 |
| 基线对照 | 六条 R18 基线用 **R18 自己的** `policies()/causal_loop()/outcome()` 重跑，必须逐字段复现 manifest（含完整动作表），否则 abort；这是"世界与物理没变"的硬门 |
| 参考 | 只读重放 R18 的 28 个见证，仅作"该族至少可解"的对照，**永远不是教师成功的判据** |
| 分组 | 一组 = 本体 + 镜像（原子）；族内按组名（candidate 索引）排序后**交替** train/validation，索引 0 起为 train；F2 两组 → 1 train + 1 validation；每族都有两个 split；同组不跨 split |
| 放松条款 | 不放宽阈值、不换世界、不调教师、不扩动作集、不追加尝试、不抽新种子；只跑一次，无 rerun、无 repair |

## 做法

1. **只读加载**：`load_worlds()` 校验 schema、`development_only`、`test_split=false`、28/14/12-4-12 计数、逐世界 `verdict=valid`、六基线全失败、`any_constant_feasible` 为空、参考见证 80 动作且 `witness_replay=true`，并把 manifest SHA 与 R18 `sha256.json` 对照。
2. **驱动器拥有物理**：每个决策用 `tactical_teacher_validation.advance`（R18 与独立验证器用的同一 step 函数）执行，再喂给教师下一帧。**不使用教师自带的 `rollout()`**：`tactical_teacher.rollout` 走的是 `tactical_scenarios.advance`，其 coverage/gap/window/threshold 是**硬编码默认值**（`(100,284)`、`(180,204)`、`34..104`、`180`），根本不是这 28 个世界，用它产生的 episode 与验证器语义不同。这不是改教师——只调用 `decide()`，教师代码一行未动。
3. **逐决策落盘**：`frame / action / option / trigger_reason / confidence / candidate / x_before / x_after / hit / actual_hold` 共 80 条，加完整 **160 逻辑帧动作**（每决策 2 帧）、`actions_sha256`、option/trigger 直方图、`switches`、`distinct_actions`。
4. **独立 verify**：`metrics()` 从 `initial_state` 重新重放一遍，给出 `verify` 状态、`first_hit`、`survived`、`reached_goal`、final x；与驱动器记录值交叉断言。
5. **六基线复算**：直接 import R18 的 `policies/ causal_loop/ outcome`，把 `feasible/survived/goal_satisfied/first_hit/decisions` 与**完整动作表**逐字段比 manifest。
6. **恒定判定 + 交叉核对**：见判据表。
7. **分组**：`split_plan()` 确定性生成，`assert_split()` 硬断言，`run/split.json` 落盘（含每组 13 字段参数签名与两个 event）。
8. **落盘约束**：所有 JSON 用 `x` 模式打开，所有目录 `exist_ok=False`；重复运行直接失败。`run/sha256.json` 覆盖 30 个 run 产物，evidence 根目录另有 `sha256.json` 覆盖全部 37 个文件。

## 结果

### 1) 分族 feasible / survived（判据口径，退化单列）

| 族 | 教师 | 例数 | feasible | survived | goal_satisfied | first_hit 非空 | 单一动作 episode | **恒定成功（退化）** | **能力成功** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 | 旧 TacticalTeacher | 12 | **0** | 0 | 12* | 12 | 4 | **0** | **0** |
| F1 | TacticalTeacherV2 | 12 | **12** | 12 | 12* | 0 | 0 | **0** | **12** |
| F2 | 旧 TacticalTeacher | 4 | **4** | 4 | 4* | 0 | 0 | **0** | **4** |
| F2 | TacticalTeacherV2 | 4 | **4** | 4 | 4* | 0 | 0 | **0** | **4** |
| F3 | 旧 TacticalTeacher | 12 | **10** | 10 | 12 | 2 | 0 | **0** | **10** |
| F3 | TacticalTeacherV2 | 12 | **12** | 12 | 12 | 0 | 0 | **0** | **12** |
| 合计 | 旧 TacticalTeacher | 28 | **14** | 14 | 28 | 14 | 4 | **0** | **14** |
| 合计 | TacticalTeacherV2 | 28 | **28** | 28 | 28 | 0 | 0 | **0** | **28** |

\* F1/F2 没有越线目标，`goal_satisfied` 按 R18 口径恒真；**只有 F3 的 goal 列有信息**。F3 的 12/12 goal 里含旧教师那 2 例失败：它第 70 帧就判死，但动作仍被继续执行到 160 帧，于是第 76 帧越过了 goal 线，`goal_satisfied` 因此仍为真——**这正是 R17/R18"goal 不能挽救失败、feasible 必须同时 survived"的实际例子**（判据用 `feasible`，不用 goal）。

### 2) 恒定成功：0 例（单独标注，不计入能力）

- **42 次成功（14 + 28）里没有一次是"整个 episode 只用一个动作"**，所以本轮没有任何成功需要被降级，`degenerate_success_total = 0`，`constant_successes = {teacher: [], teacher_v2: []}`。
- 但**确实有 4 个恒定 episode**：旧教师在 `wf18-F1-002-m0/m1`、`wf18-F1-006-m0/m1` 上从头到尾 `0×80`（恒 hold），全部**失败**（首次碰撞帧 76 / 40，正好是 wall_frame）。它们被记为 `single_action = true`、`degenerate = false`（因为没成功），并且**逐个与 R18 的常量探针交叉核对**：动作 0 对应 R18 `extra_constants["0"]/["9"]`，两处 `feasible = false`，与教师 episode 的失败完全一致（`consistent = true`）。**这正是"动作恒定不可能成功"的逐世界证据**——R18 已证 6 个常量全失败，本轮 4 个恒定 episode 全部复现该结论。
- 判据对"恒定但失败"和"恒定且成功"分别处理：前者只标 `single_action`，后者才标 `degenerate` 并从能力里剔除。测试用构造数据覆盖了"恒定且成功 → 退化、能力为 0"这一支（真实数据里它没有出现，而代码路径必须正确）。

### 3) 每个教师每族"改变了多少次动作"

| 族 | 教师 | switches 合计 | 每例均值 | 中位数 | 最大 | 有改变的事件数 / 例数 |
|---|---|---:|---:|---:|---:|---:|
| F1 | 旧 | **8** | 0.667 | 1 | 1 | 8 / 12 |
| F1 | V2 | **24** | 2.000 | 2 | 2 | 12 / 12 |
| F2 | 旧 | **88** | 22.000 | 22 | 22 | 4 / 4 |
| F2 | V2 | **88** | 22.000 | 22 | 22 | 4 / 4 |
| F3 | 旧 | **142** | 11.833 | 11.5 | 17 | 12 / 12 |
| F3 | V2 | **76** | 6.333 | 7 | 8 | 12 / 12 |
| 合计 | 旧 | **238** | 8.500 | 9 | 22 | 24 / 28 |
| 合计 | V2 | **188** | 6.714 | 4.5 | 22 | 28 / 28 |

读法：旧教师在 F1 上最多只改 1 次动作（4 例 0 次、8 例 1 次），是"**动一小段就永久 hold**"的两段式；V2 在 F1 固定改 2 次（±2 粗调 → ±1 细调 → hold）。F2 上两个教师的**动作流逐位相同**（4/4），所以 88 是同一套发射同步微移。F3 上 V2 用 3–8 次改变完成"等窗口—越线—（有时）回稳"，旧教师用 10–17 次但时机取自硬编码窗口。

### 4) 逐世界明细（feasible / survived / first_hit / switches / 不同动作数）

`first_hit` 为逻辑帧，`—` 表示整段无碰撞。完整 80 决策、160 帧动作、option/trigger 直方图都在 `run/worlds/<family>/<event>.json`。

| 事件 | 旧 feasible | 旧 first_hit | 旧 switches | 旧动作 | V2 feasible | V2 first_hit | V2 switches | V2 动作 |
|---|---|---:|---:|---|---|---:|---:|---|
| wf18-F1-002-m0 | 0 | 76 | 0 | [0] | 1 | — | 2 | [0,1,2] |
| wf18-F1-002-m1 | 0 | 76 | 0 | [0] | 1 | — | 2 | [-2,-1,0] |
| wf18-F1-005-m0 | 0 | 48 | 1 | [-2,0] | 1 | — | 2 | [0,1,2] |
| wf18-F1-005-m1 | 0 | 48 | 1 | [0,2] | 1 | — | 2 | [-2,-1,0] |
| wf18-F1-006-m0 | 0 | 40 | 0 | [0] | 1 | — | 2 | [-2,-1,0] |
| wf18-F1-006-m1 | 0 | 40 | 0 | [0] | 1 | — | 2 | [0,1,2] |
| wf18-F1-007-m0 | 0 | 60 | 1 | [-2,0] | 1 | — | 2 | [0,1,2] |
| wf18-F1-007-m1 | 0 | 60 | 1 | [0,2] | 1 | — | 2 | [-2,-1,0] |
| wf18-F1-009-m0 | 0 | 52 | 1 | [-2,0] | 1 | — | 2 | [0,1,2] |
| wf18-F1-009-m1 | 0 | 52 | 1 | [0,2] | 1 | — | 2 | [-2,-1,0] |
| wf18-F1-010-m0 | 0 | 56 | 1 | [-2,0] | 1 | — | 2 | [0,1,2] |
| wf18-F1-010-m1 | 0 | 56 | 1 | [0,2] | 1 | — | 2 | [-2,-1,0] |
| wf18-F2-119-m0 | 1 | — | 22 | [0,1] | 1 | — | 22 | [0,1] |
| wf18-F2-119-m1 | 1 | — | 22 | [-1,0] | 1 | — | 22 | [-1,0] |
| wf18-F2-399-m0 | 1 | — | 22 | [0,1] | 1 | — | 22 | [0,1] |
| wf18-F2-399-m1 | 1 | — | 22 | [-1,0] | 1 | — | 22 | [-1,0] |
| wf18-F3-000-m0 | 1 | — | 10 | [-2,0,1] | 1 | — | 7 | [-2,-1,0] |
| wf18-F3-000-m1 | 1 | — | 10 | [-1,0,2] | 1 | — | 7 | [0,1,2] |
| wf18-F3-001-m0 | 1 | — | 17 | [-2,0,1] | 1 | — | 6 | [-2,-1,0] |
| wf18-F3-001-m1 | 1 | — | 17 | [-1,0,2] | 1 | — | 6 | [0,1,2] |
| wf18-F3-002-m0 | 1 | — | 13 | [-2,0,1] | 1 | — | 8 | [-2,-1,0,1] |
| wf18-F3-002-m1 | 1 | — | 13 | [-1,0,2] | 1 | — | 8 | [-1,0,1,2] |
| **wf18-F3-003-m0** | **0** | **70** | 8 | [-2,0,1] | 1 | — | 7 | [-2,-1,0] |
| **wf18-F3-003-m1** | **0** | **70** | 8 | [-1,0,2] | 1 | — | 7 | [0,1,2] |
| wf18-F3-004-m0 | 1 | — | 13 | [-2,0,1] | 1 | — | 3 | [-2,0] |
| wf18-F3-004-m1 | 1 | — | 13 | [-1,0,2] | 1 | — | 3 | [0,2] |
| wf18-F3-005-m0 | 1 | — | 10 | [-2,0,1] | 1 | — | 7 | [-2,-1,0,1] |
| wf18-F3-005-m1 | 1 | — | 10 | [-1,0,2] | 1 | — | 7 | [-1,0,1,2] |

代表动作流（游程，决策数；1 决策 = 2 逻辑帧）：

| 事件 | R18 参考见证 | 旧教师 | V2 |
|---|---|---|---|
| wf18-F1-002-m0（gap 88–104，wall 76） | `[+1×12, 0×68]` | `[0×80]` | `[+2×4, +1×1, 0×75]` |
| wf18-F1-006-m0（gap 144–164，wall 40） | `[-1×12, 0×68]` | `[0×80]` | `[-2×5, -1×1, 0×74]` |
| wf18-F1-007-m0（gap 152–172，wall 60） | `[+2×4, 0×76]` | `[-2×5, 0×75]` | `[+2×3, +1×1, 0×76]` |
| wf18-F2-119-m0 | `[-2×4, 0×76]` | `[0×1, (+1×1, 0×5)×10, +1×1, 0×18]` | 同左（**逐位相同**） |
| wf18-F3-000-m0（window 56–76，thr 198，goal 174） | `[0×8, -1×40, 0×32]` | `[0×1, +1×1, 0×11, +1×1, 0×3, -2×21, 0×11, +1×1, 0×11, +1×1, 0×18]` | `[-2×8, -1×1, 0×18, -2×5, -1×1, 0×33, -2×1, 0×13]` |
| **wf18-F3-003-m0（window 80–108，thr 186）** | `[0×8, -1×72]` | `[0×1, +1×1, 0×10, +1×1, 0×4, -2×25, 0×14, +1×1, 0×23]` → **`-2` 起于帧 34、死在帧 70** | `[-2×14, -1×1, 0×24, -2×5, -1×1, 0×28, -1×1, 0×6]` |

### 5) 失败保留与参考见证对照（判据不放宽）

- 旧教师的失败**逐条保留**在 `run/summary.json` 的 `failure_preserved.teacher` 与 14 个失败世界各自的 JSON 里：**F1 全 12 例**（首次碰撞帧 76/76/48/48/40/40/60/60/52/52/56/56，**与各组 `wall_frame` 逐一相等，12/12**）+ **`wf18-F3-003-m0/m1`**（首次碰撞帧 70/70）。`verify` 状态全部是 `rejected_witness`，没有任何一次被改写成"未知"或"可解"。
- V2 **0 失败**。
- **"两个教师在某族都 0 可行"的情况没有出现**：`both_failed_families = []`。旧教师在 **F1 单方面 0/12**（一族全灭，只是因为 V2 在该族 12/12 才没有构成"双失败族"）；这一点单独报告，不因另一教师的成功而淡化。
- 该族的参考见证对照：**F1 的 12 个世界 R18 都有完整见证**（manifest `reference.status = feasible`，`checked` 8–17 个程序），本轮又用 `verify()` 把 12/12 重放为 feasible。**也就是说 F1 不是"无解"，是旧教师解不出**。同理 F3-003 的两个世界也有完整见证（`checked = 51`，`[0×8, -1×72]`）且重放通过，V2 也过了——失败只属于旧教师。
- 各族参考见证数（= R18 该族世界数）：F1 12/12、F2 4/4、F3 12/12，**28/28 重放为 feasible**（`reference_replay.status = {feasible: 28}`）。`unknown = 0`，本轮没有出现"参考找不到见证"的世界（R18 已把这类世界记 `unknown` 并丢弃，没有进入本集）。

### 6) 六条冻结基线复现（计数口径复用，168/168）

用 R18 自己的 `policies()/causal_loop()/outcome()` 重跑 hold / right / left / alternate(每 16 决策) / random(37318) / weak(教师首帧动作)：

| 对照项 | 结果 |
|---|---|
| 基线行数 | **168 = 28 世界 × 6 条** |
| 逐字段一致（feasible / survived / goal_satisfied / first_hit / decisions） | **168/168** |
| 完整动作表一致（80 动作逐位） | **168/168** |
| weak 的 `teacher_first` 与 `equivalent_baseline` | **28/28 一致** |
| `baseline_reproduction.matches` | **true**，mismatches = `[]` |

因此"世界没被换、物理没被改"是可验证的，不是声明。R18 的计数口径（`feasible = survived and goal`、`first_hit = sticky 首帧`、弱基线投影到支持候选）被原样复用，没有另立一套。

### 7) 冻结的训练/评估分组（`run/split.json`）

规则：一组 = 本体 + 镜像（原子）；族内按组名（candidate 索引）排序，**索引 0 起交替** train / validation；F2 只有 2 组 → 索引 0 train、索引 1 validation。

| 族 | 组 | 族内索引 | split | 事件（本体 / 镜像） | 本体 start_x |
|---|---|---:|---|---|---:|
| F1 | wf18-F1-002 | 0 | **train** | m0 / m1 | 56.0 |
| F1 | wf18-F1-005 | 1 | validation | m0 / m1 | 216.0 |
| F1 | wf18-F1-006 | 2 | **train** | m0 / m1 | 204.0 |
| F1 | wf18-F1-007 | 3 | validation | m0 / m1 | 128.0 |
| F1 | wf18-F1-009 | 4 | **train** | m0 / m1 | 104.0 |
| F1 | wf18-F1-010 | 5 | validation | m0 / m1 | 216.0 |
| F2 | wf18-F2-119 | 0 | **train** | m0 / m1 | 204.0 |
| F2 | wf18-F2-399 | 1 | validation | m0 / m1 | 208.0 |
| F3 | wf18-F3-000 | 0 | **train** | m0 / m1 | 282.00 |
| F3 | wf18-F3-001 | 1 | validation | m0 / m1 | 220.80 |
| F3 | wf18-F3-002 | 2 | **train** | m0 / m1 | 235.60 |
| F3 | wf18-F3-003 | 3 | validation | m0 / m1 | 317.20 |
| F3 | wf18-F3-004 | 4 | **train** | m0 / m1 | 276.72 |
| F3 | wf18-F3-005 | 5 | validation | m0 / m1 | 274.16 |

计数：**组 train 7 / validation 7；事件 train 14 / validation 14**；每族都有两个 split（F1 3/3、F2 1/1、F3 3/3）。`split.json` 里每组还带 **13 字段参数签名**（`signature` 与 `params` 两处，字段名在 `signature_fields`）和两个 event 的 `event_id / mirror / start_x`。

断言（写入 `split.json.assertions`，`assert_split()` 硬校验）：

| 断言 | 状态 |
|---|---|
| 一组恰好 2 个 event，镜像标志互补 | ✅ 14/14 |
| 同组同 split（**同组不跨 split**） | ✅ 14/14 |
| 镜像与本体同组、同 split、同签名、start_x 互补（和 = 384） | ✅ 14/14 |
| 28 个 event 各属且仅属一个 split | ✅ |
| 每族都有 train 与 validation | ✅ F1/F2/F3 |
| F2：1 train + 1 validation | ✅ |
| 无 `test` split、无 `holdout` 等其它标签 | ✅（未知标签会被拒绝） |

### 8) 分组内诊断（**非判据**，供后续学习参考）

| 族 | split | 旧 feasible / 例数 | V2 feasible / 例数 |
|---|---|---:|---:|
| F1 | train | 0/6 | 6/6 |
| F1 | validation | 0/6 | 6/6 |
| F2 | train | 2/2 | 2/2 |
| F2 | validation | 2/2 | 2/2 |
| F3 | train | 6/6 | 6/6 |
| F3 | validation | 4/6 | 6/6 |

旧教师的 2 个 F3 失败都落在 validation（`wf18-F3-003`）——这是**先冻结分组、后看结果**的自然分配，没有为了"好看"调整任何一组。数值上 train/validation 的难度不对称是事实，报告出来而不是藏起来。

## 机制：为什么是这个结果

**旧 `TacticalTeacher` 在 F1 全灭的原因是硬编码几何，不是世界太难。** 它的 F1 规则用的是 R8 时代写死的坐标：`180 <= cx <= 204` 视为"可靠内部缺口"，transit 目标 `(60, 88)`，`cx > 88` 才向左。28 个 R18 世界的缺口位置是另一套（88–104、144–164、152–172、248–264、256–268），于是：

| F1 组 | 世界 gap | wall | 旧教师最终停位（canonical） | 落在缺口内？ | first_hit |
|---|---|---|---:|---|---:|
| 002 | 88–104 | 76 | 56（`cx > 88` 为假 → 直接 hold） | ❌ | 76 |
| 005 | 248–264 | 48 | 200（落在它自认的 180–204 缺口里） | ❌ | 48 |
| 006 | 144–164 | 40 | 204（同上，起手就在 180–204 区间） | ❌ | 40 |
| 007 | 152–172 | 60 | 88（到 88 就停） | ❌ | 60 |
| 009 | 152–172 | 52 | 88 | ❌ | 52 |
| 010 | 256–268 | 56 | 200 | ❌ | 56 |

**12/12 首次碰撞帧恰好等于该组 `wall_frame`**：它只是站在一个"自认为安全"的坐标上等墙压过来。这与 R17 观察到的"训练网络在 F1 全部死在 wall_frame"是同一个现象，只不过这里是规则教师。

**V2 在 F1 12/12 的原因是它被直接告知缺口。** `target()` 用 `s.gap` 构造 `(gap[0]+2, gap[1]-2)` 区域，`preferred()` 用 ±2 粗调、±1 细调，最后**12/12 都停在 `gap[0]+4`**（92/252/160/156/260，canonical），全部落在缺口内。所以 F1 对 V2 不难，但它证明的是"读到 gap 参数 + 12 帧前瞻就能精确停车"，**不是战术理解**。

**F3 的对比最干净。** 旧教师的 F3 规则用写死的 `34 <= frame < 104` 当窗口，`-2` 一律从帧 34 起手（不看世界自己的 `window`）。它能否活下来，完全取决于它固定的越线时刻是否恰好落在该世界的窗口里：6 个组的越线帧是 58 / 52 / 50 / **70** / 56 / 56，而窗口起点是 56 / 48 / 40 / **80** / 48 / 56，前 5 组恰好落在窗口内（所以 10/12 过），`wf18-F3-003` 的窗口起点 80 更晚，于是帧 70 时 `cx = 181.2 ≤ 186` 而 `frame+1 = 71 < 80`，按 F3 的 depletion 规则**当场判死**（first_hit 70，两个镜像完全一致）。V2 用真实 `s.window`：先停在阈值上方（如 F3-003 停在 201.2），等帧 78 才开始越线，**在窗口边界上完成穿越**，所以 12/12 过。这条差异是"硬编码窗口 + 固定起手 vs 读世界参数 + 前瞻"的直接证据。

**F2 的 4/4 不是 V2 的功劳。** V2 的 `target()` 对 F2 返回 None，于是 `preferred()` 退回**与旧教师字面相同的发射同步微移**（`0 <= age < 4 and x < 368` 时给 `sign`）；12 帧 beam 没有覆盖它。结果**两个教师的 80 动作逐位相同（4/4）**。加上 R18 已指出 F2 的 4 例其实是 2 组几乎同一世界（start 204 vs 208），F2 的 4/4 只能当"这条发射同步规则在这个参数点上能活满 160 帧"，不能当 V2 搜索能力的证据。

## 判定（有界，不做显著性声明）

1. **教师能玩这批世界**：V2 28/28；旧规则教师 14/28，且**按族分裂**——F2 4/4、F3 10/12、F1 0/12。
2. **没有任何一次成功是退化的**：42 次成功全部是非恒定 episode（`degenerate_success_total = 0`）；4 个恒定 episode 全部失败且被单独标注、交叉核对一致。因此"42 次成功"里没有一条需要被打折。
3. **没有"双失败族"**：`both_failed_families = []`。旧教师在 F1 的 0/12 作为**单教师失败**原样保留；判据未放宽、世界未更换、教师未调参、未追加尝试。
4. **失败旁边有见证**：所有失败世界在 R18 manifest 里都有完整参考见证，本轮 28/28 重放通过——失败是"该教师解不出"，不是"世界不可解"。
5. **分组已冻结**：14 组 → 7 train + 7 validation（事件 14/14），同组不跨 split、镜像与本体同组、每族都有两个 split、F2 1/1；分组只看族内索引，不看任何本轮结果。
6. **不做显著性声明**：14 组、28 例、镜像不是独立样本，3 个族的样本量 12/4/12；本报告只报观测与机制，不宣称泛化、不宣称战术理解、不宣称可部署。

## 执行与退出码

```powershell
# 1) 判据单测（预注册之前）
python -m pytest tests/test_v373_valid_teacher_eval.py -q -p no:cacheprovider --basetemp .tmp/v373-validteacher-round19-tests-01
#    15 passed, 1 skipped，exit 0
# 2) 冻结预注册（判据 + 分组规则 + 13 个身份 SHA）
python -m tools.rl.v373_valid_teacher_eval preregister
#    exit 0；28 事件 / 14 组 / manifest SHA 9bd3a1b8…
# 3) 预注册后复跑同一测试（冻结的测试文件仍通过）
python -m pytest tests/test_v373_valid_teacher_eval.py -q -p no:cacheprovider --basetemp .tmp/v373-validteacher-round19-tests-02
#    15 passed, 1 skipped，exit 0
# 4) 唯一一次教师评估（后台 job，自行收取）
python -m tools.rl.v373_valid_teacher_eval run
#    exit 0；28 世界 + split + summary + sha256 共 30 个产物；168/168 基线复现；见证 28/28
# 5) run 之后第三次测试（此时 run 相关断言真正执行）
python -m pytest tests/test_v373_valid_teacher_eval.py -q -p no:cacheprovider --basetemp .tmp/v373-validteacher-round19-tests-03
#    16 passed，exit 0
# 6) 只读复核
python -m tools.rl.v373_valid_teacher_eval verify
#    exit 0；artifacts 30 / baselines 168 / reference_replay 28 / groups 14 / degenerate 0
```

退出码依次 **0 / 0 / 0 / 0 / 0 / 0**。日志（PowerShell 5.1 `Out-File -Encoding utf8`）：[tests-first.log](<../evidence/validteacher-round19/tests-first.log>)、[tests-after-preregister.log](<../evidence/validteacher-round19/tests-after-preregister.log>)、[tests-completion.log](<../evidence/validteacher-round19/tests-completion.log>)、[preregister.log](<../evidence/validteacher-round19/preregister.log>)、[execution.log](<../evidence/validteacher-round19/execution.log>)、[command-results.md](<../evidence/validteacher-round19/command-results.md>)。

- **run 只跑一次，没有 first-failure、没有 rerun、没有 repair pass**：30 个产物一次写全，`run/` 之后未被改写（`verify` 复核 30/30 SHA 一致）。教师单决策成本约 4 ms（V2 的 12 帧 beam，CPU），`run` 墙钟约 8.5 s（`split.json` 20:52:09.226 → `sha256.json` 20:52:17.747，不含解释器启动与 import）。
- **披露**：预注册之前的测试阶段会**端到端跑 1 个世界**（`wf18-F1-002-m0`）作为契约测试。它不参与判据选择、不参与分组、也没有被用来改教师或世界；判据是模块常量，在 run 之前被冻结并逐字段复核，而 28 个世界的完整结果只存在于 `run/`。

## 限制（必须与结论一起读）

1. **V2 是一个被直接告知世界几何的有界前瞻规划器**（`target()` 读 `s.gap` / `s.window` / `s.threshold` / `s.goal`，12 帧 beam、真实碰撞模型）。它 28/28 说明"这些世界对一个知道参数、能前瞻 6 步的规划器是可玩的"，**不说明战术理解，也不是学习到的策略**。
2. **旧教师在 F1 全灭主要是硬编码坐标的错**（`180..204`、`60..88`、`88`、`34..104`），不是"世界太难"或"F1 不可解"——12/12 都有见证，V2 12/12 过。因此"旧教师 0/12"不能反过来当作世界有效性的强度证据。
3. **F2 的 4/4 信息量很低**：4 例其实是 2 组几乎同一世界（仅 `start_x` 204 vs 208），且两个教师的动作流**逐位相同**——F2 上是同一条发射同步微移，V2 的搜索没有贡献。F2 的 validation 只有 1 组（2 例）。
4. **F1 的成功依赖精确停车**（R18 限制 5 依旧成立）：V2 12/12 都停在 `gap[0]+4`，缺口宽只有 12–20 px。这是几何构造，不是"会玩"。
5. **恒定判定只覆盖"整个 episode 单一动作"**。"两段式"（改 1 次后永久 hold）**不算恒定**，但旧教师的 F1 恰好是这种形态（8 例 1 次切换、4 例 0 次）且全灭；V2 的 F1 是 3 段。也就是说：**本轮没有出现"退化成功"，但不等于所有成功都复杂**。
6. **分组是确定性交替，不是分层抽样**：不看结果，因此 train/validation 难度不对称（旧教师在 F3 的 2 个失败全在 validation）。F2 只有 2 组，validation 只有 1 组，用它算 validation 成功率会极不稳定。
7. **`goal_satisfied` 对 F1/F2 恒真**（R18 口径），只有 F3 的 goal 有信息；F3 那 2 例是"**死后才越线**"（第 70 帧判死、第 76 帧过 goal），所以 goal 绝不能替代 survived（与 R17 一致）。
8. **镜像不是独立样本**：14 组、28 例，有效独立几何只有 14 个（F2 实际只有 2 个且几乎相同）。不做显著性、不做 seed 级重复（教师是确定性的，没有 seed 自由度）。
9. **未训练、无 checkpoint、未开正式门、不运行游戏**：本轮不产生任何可部署产物，也没有验证"学习能否学会"这些问题——`split.json` 只是把后续学习要用的划分冻结下来。
10. **一次运行**：没有第二套 seed/参数/顺序来检验稳定性；教师确定性，两次跑必然逐位相同，但报告只依赖唯一一次落盘。

## 对既有结论的处置

| 既有结论 | 处置 |
|---|---|
| R18："28 个世界非常量可解，六基线全 0，28/28 有见证" | **确认并补上"能玩"的证据**：V2 28/28、旧教师 14/28 可行；168/168 基线复现；28/28 见证重放通过 |
| R18："教师首帧动作重复"的弱基线全 0 | **不等于教师不行**：旧教师的**完整规则**在 14/28 上能过（F2 4/4、F3 10/12）。弱基线只用了 1 个决策，低估了教师 |
| R17："F1（避难所）几乎全灭，网络都死在 wall_frame" | **对旧规则教师同样成立**（0/12、12/12 死于 wall_frame、两段式），**但对 V2 不成立**（12/12 停在 `gap[0]+4`）。F1 的难点是"精确停车 + 读到 gap"，不是不可解 |
| 任何在 R11/R16 旧 32 世上算出的成功率 | 继续**作废**（R18 已判；本轮不碰旧集） |
| 后续学习该用哪套世界与怎么分 | 用 `run/manifest.json` 的 28 个世界 + 本轮 [split.json](<../evidence/validteacher-round19/run/split.json>) 的 7/7 组划分；validation 仅 14 例（F2 仅 1 组）需在口径中注明；旧集只留诊断用途 |

本轮不修改 `ledger-round17.md`（属上一轮台账）；建议父级在 round19 台账登记：教师能力首次按族量化（旧 14/28、V2 28/28）、退化成功 0、无双重失败族、F1 的旧教师失败归因于硬编码几何、六基线 168/168 复现、训练/评估分组 7/7 冻结。

## 证据文件

- [preregistration.json](<../evidence/validteacher-round19/preregistration.json>)：判据（`criterion`）、预期计数、分组规则与预期、13 个身份 SHA；SHA `48a8ccb2…`。
- [run/split.json](<../evidence/validteacher-round19/run/split.json>)：14 组的 split、13 字段参数签名、两个 event、全部断言。
- [run/summary.json](<../evidence/validteacher-round19/run/summary.json>)：分族/分 split 汇总、族判定、失败清单、退化清单、基线复现、见证重放、教师模块 SHA、`identities_unchanged`。
- `run/worlds/F1|F2|F3/<event_id>.json`：28 个逐世界证据（80 决策 × 2 教师、160 帧动作、option/trigger 直方图、独立 verify、first_hit、goal、survived、恒定核对、六基线、见证重放）。
- [run/sha256.json](<../evidence/validteacher-round19/run/sha256.json>)：30 个 run 产物的 SHA（已复核 0 处不符；`run/sha256.json` 自身不在表内）。
- [sha256.json](<../evidence/validteacher-round19/sha256.json>)：evidence 根目录 37 个文件的 SHA（复核 0 处不符；两个 `sha256.json` 自身不在表内，目录内共 39 个文件）。
- 命令与退出码：[command-results.md](<../evidence/validteacher-round19/command-results.md>)；日志：`tests-first.log`、`tests-after-preregister.log`、`tests-completion.log`、`preregister.log`、`execution.log`。
