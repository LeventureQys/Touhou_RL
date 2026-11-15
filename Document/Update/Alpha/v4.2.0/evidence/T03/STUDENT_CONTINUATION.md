# T03s：把后续控制从固定教师换成冻结学生（WP420-02 §5 第二条腿）

日期：2026-11-13（Asia/Shanghai）。主机 WSL2，`cwd=/mnt/d/WorkShop/Touhou_RL`，
Python `/root/v400-venv/bin/python`（3.10.12），引擎 `rl_v3`
`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`（sha `691bfd77…7a568ef6`，
`check_engine_binary` 校验通过）。分支角色 **`env`**（`xvfb-run -s "-screen 0 1024x768x24"`）。

**范围声明（必读）**：本文件是 T03 试点之后计划要求的那一步**换性质实验**——
把分支的「后续控制」从固定教师换成**冻结学生**，看安全集合是否出现区分度。
它做的是 WP420-02 §5 的**第二条腿**（50 个失败相关状态的学生续行对照）在本轮的
**24 个状态 × 3 条臂**切片，**不是** G420-1：门要求 200 状态全 18 候选、≥100 个有区分度
有效状态、≥10 条祖先，本轮全部不满足（见 §6）。本文件**不写** “G420-1 通过”，
不把几何分数当引擎真值，不给空/不完整集合补正例。

原始数据：`student_continuation.json`（本目录，496 KB，含 24 状态 × 3 臂的逐臂裁决、
死亡帧、`lives`、安全集合、逐状态前缀 RLE 与 HASH、`verified_mask` 普查、重复一致、
成本与控制实验）。完整逐帧 trace 在运行目录 `.tmp_v420_data/T03/contrast_student.json`
（同一次运行，未裁剪）与 `.tmp_v420_data/T03/raw_student/*.json`（4 个分片）里，
证据目录只保留 `trace_sha256` 与帧数，不重复 ~8.7k 行帧记录。

---

## 0 结论摘要

1. **换后续控制之后，区分度出现了**：24 个失败相关状态 × 3 条臂（H=120）共 **72 条臂**，
   裁决 **`engine_death = 29` / `engine_alive = 43` / `unknown = 0`**，退出码 0、零 fail-closed。
   **「测过的臂里既有死亡又有存活」的状态数 = 6**——这正是教师续行轮里为 **0** 的那个量。
2. **同一状态、同一 H，教师续行全存活、学生续行死亡**：9 个状态同时有 T03 教师续行参照
   （18/18 存活），其中 **4 个**在本轮学生续行下出现死亡（§5.4 的 2×2：
   教师全存活 × 学生死亡 = 4，教师全存活 × 学生全存活 = 5，教师有死亡 = 0）。
3. **最强的形态**：状态 11（seed 11、帧 1760、首死前 165 帧）里，**T03 用教师续行验证过
   存活的候选 1 与 7**，在冻结学生续行下分别在帧 1837/1805 死亡，而学生自己的动作 0 存活；
   状态 15（首死前 30 帧）相反——学生自己的动作 0 在帧 1928 死亡，而**教师续行验证过的候选 1
   存活过 H**。所以区分度不是“学生总是更差”，而是“首动作 + 后续控制的组合确实改变结果”。
4. **空集合 6 个**（三臂全死，首死前 30–90 帧，学生已无法恢复）、**全存活 12 个**
   （首死前 ≥120–300 帧，死亡落在 H=120 之外）、**全死 6 个、proper（本轮口径）6 个**，
   安全集合大小直方图 `{0:6, 1:5, 2:1, 3:12}`。
5. **学生臂是可信的，不是“每步都乱走”**：24/24 个状态上，学生在重建后的 obs-v3 栈与动作历史上
   的前向 argmax **逐状态复现了录制轨迹在该决策上的动作**；10/10 次学生臂死亡**落在录制首死帧上**
   （1541 / 1928 / 1797），与 T03 §9.1 的死亡检测正对照吻合。
6. **教师查询 0 次**。教师动作是**读语料标签**（`labels.jsonl` 的 `chosen_plan.first_action`），
   不是查询；§5.4 的教师侧参照也是读 T03 已有报告。
7. **计划指定的第三条臂（`verified_mask` 的已验证替代）在本轮是 `n/a`**，且有数据证据：
   语料 **9,898/9,898** 行的 `verified_count` 恒为 1 且那一个 verified 候选**就是**教师选中的动作；
   T02d/T04 的 2 帧数据集 **19,772/19,772** 个 slot 同样只有 1 个 verified 且等于该 slot 的 `chosen`
   （§4.3）。所以没有“不是教师/学生动作的已验证候选”。替代地，9 个有 T03 参照的状态
   用 **T03 引擎验证存活**的候选补上第三臂（11 条臂）；其余状态用**明确标注未验证**的
   语料 rank-2/rank-3 候选做第 4 条臂（23 条臂），见 §4.4/§4.5。
8. **重复一致 3/3**：同一 (状态, 臂) 在同一进程里跑两次，逐帧 HASH trace 的 sha256 与裁决
   **逐比特相同**（含一条死亡臂、一条存活臂、一条来自 T03 参照的臂）。
9. **成本**：4 个分片、72 条臂 + 3 条重复臂，**996.1 s（16 min 36 s）**、127,025 引擎帧、
   3,643 次学生前向、75 次 RESET，实测 **127.5 fps / 7.84 ms/帧 / 1.68 ms/前向 / 13.83 s/臂**；
   另有一次**争用条件下**的 3 状态成本探针（9 臂，173.7 s，19.3 s/臂，§7）。
10. **G420-1 仍不通过**：24/50 失败相关状态、6 个有区分度（门要 100）、3 条祖先（门要 10）、
   3 对重复（门要 200 状态全量）；本轮甚至没有做 18 候选验证（每条臂只测 1 个首动作）。

---

## 1 本轮换了什么性质（相对 T03 试点）

T03 试点用**固定教师续行**（`fast` @ `replan_decisions=1`，每 2 帧重规划）跑完
20 状态 × 18 候选 × H=32/120，共 720 条分支**零死亡**、`proper = 0`。它的结论是：
候选只跑 2 帧（33 ms），之后教师每 2 帧纠正一次，于是任何首动作的偏差都在 2 帧内被抹平。

本轮把**后续控制**换掉，只换这一个性质：

| | T03 试点 | 本轮 T03s |
| --- | --- | --- |
| 首动作 | 18 个候选全测 | **3 条臂**（教师 / 学生 / 已验证替代），H=120 |
| 后续控制 | 固定教师（每 2 帧重规划） | **冻结 20k 学生**（每 2 帧前向 argmax，bomb 恒 0） |
| 角色 | shadow（无像素，无 xvfb） | **env**（obs-v3 需要像素；fork 子进程出不了像素） |
| 每臂成本 | 一次 RESET + 前缀 + 教师查询（60 ms × 59） | 一次 RESET + 前缀 + 60 次学生前向（≈1.7 ms） |
| 状态 | 20（含 9 个失败相关） | **24 个失败相关**（首死前 30–300 帧的 8 偏置网格 × 3 seed） |
| 结果 | `engine_alive = 720`，`proper = 0` | `engine_death = 29`，**有区分度 6 个** |

这不是“同一实验加长”：教师续行下这些状态对 18 个首动作没有分辨力，学生续行下它们有了。

---

## 2 本次实现（只增量，未改坏教师路径）

| 部分 | 内容 |
| --- | --- |
| `touhou_rl/v420/counterfactual.py` | 新增 §4b 纯逻辑层（`contrast_states` / `corpus_teacher_action` / `verified_alternatives` / `plan_arms` / `contrast_arm_outcomes` / `validate_contrast_plan` / `estimate_contrast_cost` / `load_teacher_reference` / `teacher_reference_for` / `engine_verified_candidates` / `verified_mask_census` / `summarise_contrast` / `g420_1_contrast_verdict` / `merge_contrast_reports` / `frozen_contrast_states_payload`）+ §10 引擎层（`load_student_policy` / `recorded_decision_cand` / `_StudentSession` / `_run_student_arm` / `_run_contrast_state`）+ §11 入口（`dry_run_contrast_payload` / `execute_contrast` / `merge_contrast_files`） |
| 教师路径 | `summarise` / `merge_reports` / `g420_1_verdict` / `_run_one_branch` / `execute` / `_Session` **一行未改**；`trim_report` 只多处理 `arms` 的 trace（追加，不改 `branches` 行为）；`load_states_file` 只多接受一个新的 schema |
| `tools/rl/v420.py` | 只给 `counterfactual` 加 `--rollout-policy {teacher,student}`（默认 `teacher`），并让它路由到学生模式的三条路径（dry-run / merge / execute）；其它子命令未动 |
| `configs/v420/counterfactual-student.json` | 新建：H=120、8 偏置网格、3 seed、checkpoint sha、T03 参照路径、重复臂、预算与成本模型 |
| `tests/test_v420_counterfactual.py` | **只追加** 23 个纯逻辑测试（假 env / 假学生，不启动引擎）：两条 rollout 的路由、`unknown` 不判死、死亡帧记录、臂集合去重与替代动作选择、census、参照读取、摘要与合并（见 §11） |

一条臂的确切语义（每条臂都这样跑）：

```text
RESET(seed, stage=0, difficulty=1, team=0)          # env 角色，xvfb
  -> 校验 RESET HASH == 前缀 frame 0 的 HASH（不等则重启进程重试一次）
  -> 按录制的逐帧掩码重放 1..decision_frame，逐帧比对 HASH
     （只对最后 4 帧做 encode_planes_v3，栈内容与录制时一致；每 2 帧重建一次动作历史）
  -> 校验停放点 HASH == 该状态冻结的 expected_hash（不等即 fail-closed）
  -> 在停放点做一次学生前向（=「学生在该状态的 argmax」，也是本臂的候选若 arm=student）
  -> 施加候选首动作 恰好 2 帧
  -> 循环 { 学生前向 -> 提交 2 帧 } × ((H-2)/2)
  -> 到达 H / 首次死亡 / 游戏结束 / 本地过面 = 有效完成；超时/缺帧/进程故障 = unknown
```

---

## 3 状态清单（24 个失败相关状态，**24/50**）

规则与 T03 试点相同（学生首死前 0.5–5 s，向下取整到录制的 4 帧块，同一祖先内 <20 帧去重），
但把 T03 的 3 个偏置（30/165/300）扩展成**固定 8 偏置网格** `30, 60, 90, 120, 165, 210, 255, 300`，
每 seed 8 个状态。**T03 冻结的 9 个失败相关状态被这个网格逐帧原样复现**
（1508/1376/1240、1896/1760/1628、1764/1632/1496），所以本轮的 24 个状态是试点那 9 个的**超集**。

冻结清单：`.tmp_v420_data/T03/contrast_states_frozen.json`（schema
`v420-counterfactual-student-plan/1`，`frozen: true`，24 条，index 0..23），
由 dry-run 在**任何引擎启动之前**写出，之后 4 个分片都带 `--states-file` 读它。

| # | seed | decision_frame | 首死帧 | 距首死 | 层 | 是否试点冻结 |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 10 | 1240 | 1541 | 300 | student_death_proximate | 是 |
| 1 | 10 | 1284 | 1541 | 255 | student_death_proximate | 否 |
| 2 | 10 | 1328 | 1541 | 210 | student_death_proximate | 否 |
| 3 | 10 | 1376 | 1541 | 165 | student_death_proximate | 是 |
| 4 | 10 | 1420 | 1541 | 120 | student_death_proximate | 否 |
| 5 | 10 | 1448 | 1541 | 90 | student_death_proximate | 否 |
| 6 | 10 | 1480 | 1541 | 60 | student_death_proximate | 否 |
| 7 | 10 | 1508 | 1541 | 30 | student_death_proximate | 是 |
| 8 | 11 | 1628 | 1928 | 300 | student_death_proximate | 是 |
| 9 | 11 | 1672 | 1928 | 255 | student_death_proximate | 否 |
| 10 | 11 | 1716 | 1928 | 210 | student_death_proximate | 否 |
| 11 | 11 | 1760 | 1928 | 165 | student_death_proximate | 是 |
| 12 | 11 | 1808 | 1928 | 120 | student_death_proximate | 否 |
| 13 | 11 | 1836 | 1928 | 90 | student_death_proximate | 否 |
| 14 | 11 | 1868 | 1928 | 60 | student_death_proximate | 否 |
| 15 | 11 | 1896 | 1928 | 30 | student_death_proximate | 是 |
| 16 | 12 | 1496 | 1797 | 300 | student_death_proximate | 是 |
| 17 | 12 | 1540 | 1797 | 255 | student_death_proximate | 否 |
| 18 | 12 | 1584 | 1797 | 210 | student_death_proximate | 否 |
| 19 | 12 | 1632 | 1797 | 165 | student_death_proximate | 是 |
| 20 | 12 | 1676 | 1797 | 120 | student_death_proximate | 否 |
| 21 | 12 | 1704 | 1797 | 90 | student_death_proximate | 否 |
| 22 | 12 | 1736 | 1797 | 60 | student_death_proximate | 否 |
| 23 | 12 | 1764 | 1797 | 30 | student_death_proximate | 是 |

* 24 个状态全部落在**冻结学生自己**的录制轨迹上（`kind = student_trace`），因为学生续行的
  动作历史必须能从**逐帧（2 帧决策）**的掩码重建；教师语料是 4 帧块，其中 137/6,600 个块的两个
  子帧掩码不同，无法无歧义地重建 2 帧动作历史（这些块会被 `_pair_cand` fail-closed 拒绝）。
* 3 条 seed 祖先（10/11/12，train split）、3 条轨迹记录（`student:seed10/11/12`）。
  **门要 ≥10 条互不相同的训练祖先，本轮 3 条，不假装具备。**
* 状态数 **24/50**（计划 §5 要 50 个失败相关状态）：24 ≥ 本轮下限 20，距 50 还差 26 个。

---

## 4 每状态 3 条臂：规则与来源（每条 H=120）

### 4.1 教师动作（slot 1）

`labels.jsonl` 第 `frame // 4` 行（同一决策块的 `chosen_plan.first_action`），
即**同一 RESET、同一决策时刻上教师录制的选择**。逐状态记 `teacher_cand` 与
`teacher_cand_source`。**这是读标签，不是教师查询**：本轮 `teacher_queries = 0`。

边界（必须与数字一起引用）：学生轨迹状态上，“教师在这个状态的动作”并不存在于录制里
（教师语料是同一次 RESET 的**另一条**轨迹），所以这条臂的准确定义是
“教师在同一决策槽上的录制动作”，不是“教师在当前状态的决策”。

### 4.2 学生动作（slot 2）

在重建后的状态下做**一次**学生前向（4 帧像素栈 + 8 条动作历史 + HUD，`ProbePolicy`）
取 18 路 argmax，bomb 恒 0（与冻结 v4.1.0 evaluator 同一 actuation 规则）。这条臂
**先跑**，因为它的第一次前向同时定义了学生自己的首动作，并给出一个端到端一致性检查：
**24/24 个状态上该 argmax 等于录制轨迹在该决策上的动作**（§5.5）。

### 4.3 已验证替代（slot 3）：本轮 **`n/a`**，有普查证据

计划指定的来源是 T02d/T04 数据集的 `verified_mask`。普查结果（`verified_mask_census`）：

| 来源 | 行数 | `verified_count` 直方图 | 单一 verified == 教师选中 | 有“别的已验证候选” |
| --- | ---: | --- | ---: | ---: |
| 语料 `labels.jsonl`（seed 10/11/12 = 3,300 + 3,300 + 3,298） | 9,898 | `{1: 9898}` | **9,898** | **0** |
| T02d/T04 2 帧 `slots.jsonl`（3 个 seed shard） | 19,772 | `{1: 19772}` | **19,772**（对 `chosen`） | **0** |

语料是 `verify_k=1` 录制的：每个 4 帧块只把**教师选中的那一个**候选送进引擎验证，
所以 `verified_mask` 永远只有 1 个 true，而它就是教师动作。**因此“不是教师/学生动作的
已验证候选”不存在**，这条臂在 24 个状态上全部记 `n/a`（24 条 n/a 行，原因逐条写在
`arm_plan.arms[].reason`）。**没有用未验证动作冒充它。**

### 4.4 T03 引擎已验证替代（第三臂的替代来源，11 条臂）

9 个有 T03 参照的状态（试点冻结的那 9 个）从 `counterfactual_pilot.json` 的 H=120 报告里取
**引擎在该状态上报告过 alive** 的候选（`t03_engine_alive` 池），取最小 index 且不等于教师/学生动作。
这是**引擎真值**（该分支真的跑过并存活），不是几何分数；来源与 sha 记在
`teacher_reference`。它在 3 条臂里有 8 条存活、3 条死亡（§5.3）——**正是这条臂给出了
“教师续行验证过存活、学生续行却死亡”的直接对照**。

### 4.5 未验证的 ranked 填充臂（第 4 条臂，23 条臂）

WP420-02 §5 还要求“三者有重复就补第 4 条臂”。本轮 24 个状态里 **10 个**的教师动作与
学生动作相同（`cand 0` 占语料 argmax 的 44%），加上 §4.3 的 `n/a`，实际可跑的不同首动作
往往只有 2 个。为了让每个状态仍有 **3 个不同首动作**，缺少的槽位用**语料 rank-2 / rank-3 候选**
填充，臂名 `teacher_ranked`，`source = corpus_ranked`，`source_verified = false`，
行内 `reason` 明确写 “NOT a verified action”。它的**候选选择**来自教师目标函数的排序
（几何），但它的**裁决**仍是引擎真值，且摘要里另外给了
`proper_without_unverified_filler`（把这个填充臂全部剔除后的计数，§5.3），
读者可以自己判断区分度是否依赖它。

### 4.6 去重、重复与臂数

* **重复的首动作只跑一次**：教师动作 == 学生动作时，slot 2 记 `status = duplicate`
  （不是 `run`），该次 rollout 同时作为两条臂的证据（`armed_as = ["student","teacher"]`）。
  在同一个后续控制下，两个相同的首动作就是同一条轨迹，跑第二次只会是**假的重复实验**。
* 每条臂恰好 **3 个不同首动作**（24/24 个状态）；槽位最多 4。
* **重复一致**：3 个 (状态, 臂) 在同一进程里跑两遍（§5.6）。

---

## 5 结果（24 状态 × 3 臂 = 72 条臂，H=120）

`status = pass`、`exit_code = 0`；**`unknown = 0`**、fail-closed 触发 0 次；
前缀逐帧 HASH 校验 **115,200 次 / 0 处不一致**。

### 5.1 每状态 × 每臂

（`lives` 一律为 2.0：首次死亡后剩 2 条残机，属正常残机死亡，不是 game over；
`death frame` 是**绝对引擎帧**，与状态帧同坐标系。）

| # | seed | 状态帧 | 距首死 | 臂 | cand | 裁决 | 死亡帧 | lives | 学生决策 | 墙钟 s |
| ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 10 | 1240 | 300 | student+teacher | 0 | engine_alive | – | 2.0 | 59 | 10.50 |
| 0 | 10 | 1240 | 300 | t03_engine_verified | 1 | engine_alive | – | 2.0 | 59 | 10.31 |
| 0 | 10 | 1240 | 300 | t03_engine_verified | 2 | engine_alive | – | 2.0 | 59 | 10.26 |
| 1 | 10 | 1284 | 255 | student+teacher | 0 | engine_alive | – | 2.0 | 59 | 10.58 |
| 1 | 10 | 1284 | 255 | teacher_ranked | 1 | engine_alive | – | 2.0 | 59 | 10.73 |
| 1 | 10 | 1284 | 255 | teacher_ranked | 2 | engine_alive | – | 2.0 | 59 | 10.65 |
| 2 | 10 | 1328 | 210 | student+teacher | 0 | engine_alive | – | 2.0 | 59 | 11.04 |
| 2 | 10 | 1328 | 210 | teacher_ranked | 1 | engine_alive | – | 2.0 | 59 | 11.04 |
| 2 | 10 | 1328 | 210 | teacher_ranked | 2 | engine_alive | – | 2.0 | 59 | 10.92 |
| 3 | 10 | 1376 | 165 | t03_engine_verified | 0 | engine_alive | – | 2.0 | 59 | 11.54 |
| 3 | 10 | 1376 | 165 | t03_engine_verified | 1 | engine_alive | – | 2.0 | 59 | 11.14 |
| 3 | 10 | 1376 | 165 | student+teacher | 3 | engine_alive | – | 2.0 | 59 | 11.32 |
| 4 | 10 | 1420 | 120 | student+teacher | 7 | engine_alive | – | 2.0 | 59 | 11.30 |
| 4 | 10 | 1420 | 120 | teacher_ranked | 11 | engine_alive | – | 2.0 | 59 | 12.02 |
| 4 | 10 | 1420 | 120 | teacher_ranked | 13 | engine_alive | – | 2.0 | 59 | 11.39 |
| 5 | 10 | 1448 | 90 | teacher_ranked | 6 | engine_alive | – | 2.0 | 59 | 12.15 |
| 5 | 10 | 1448 | 90 | student+teacher | 7 | **engine_death** | 1541 | 2.0 | 46 | 11.20 |
| 5 | 10 | 1448 | 90 | teacher_ranked | 8 | **engine_death** | 1526 | 2.0 | 38 | 10.98 |
| 6 | 10 | 1480 | 60 | teacher_ranked | 2 | **engine_death** | 1543 | 2.0 | 31 | 11.26 |
| 6 | 10 | 1480 | 60 | teacher | 3 | **engine_death** | 1541 | 2.0 | 30 | 11.44 |
| 6 | 10 | 1480 | 60 | student | 7 | **engine_death** | 1541 | 2.0 | 30 | 12.38 |
| 7 | 10 | 1508 | 30 | t03_engine_verified | 0 | **engine_death** | 1544 | 2.0 | 17 | 11.75 |
| 7 | 10 | 1508 | 30 | student | 3 | **engine_death** | 1541 | 2.0 | 16 | 11.11 |
| 7 | 10 | 1508 | 30 | teacher | 6 | **engine_death** | 1551 | 2.0 | 21 | 11.89 |
| 8 | 11 | 1628 | 300 | teacher | 0 | engine_alive | – | 2.0 | 59 | 13.48 |
| 8 | 11 | 1628 | 300 | t03_engine_verified | 1 | engine_alive | – | 2.0 | 59 | 13.12 |
| 8 | 11 | 1628 | 300 | student | 3 | engine_alive | – | 2.0 | 59 | 13.20 |
| 9 | 11 | 1672 | 255 | student+teacher | 0 | engine_alive | – | 2.0 | 59 | 13.26 |
| 9 | 11 | 1672 | 255 | teacher_ranked | 1 | engine_alive | – | 2.0 | 59 | 13.61 |
| 9 | 11 | 1672 | 255 | teacher_ranked | 2 | engine_alive | – | 2.0 | 59 | 13.65 |
| 10 | 11 | 1716 | 210 | student+teacher | 0 | engine_alive | – | 2.0 | 59 | 14.09 |
| 10 | 11 | 1716 | 210 | teacher_ranked | 1 | **engine_death** | 1834 | 2.0 | 58 | 14.49 |
| 10 | 11 | 1716 | 210 | teacher_ranked | 2 | **engine_death** | 1835 | 2.0 | 59 | 14.03 |
| 11 | 11 | 1760 | 165 | student | 0 | engine_alive | – | 2.0 | 59 | 14.58 |
| 11 | 11 | 1760 | 165 | t03_engine_verified | 1 | **engine_death** | 1837 | 2.0 | 38 | 14.27 |
| 11 | 11 | 1760 | 165 | teacher | 7 | **engine_death** | 1805 | 2.0 | 22 | 13.71 |
| 12 | 11 | 1808 | 120 | teacher | 0 | **engine_death** | 1910 | 2.0 | 50 | 14.46 |
| 12 | 11 | 1808 | 120 | teacher_ranked | 1 | **engine_death** | 1841 | 2.0 | 16 | 13.67 |
| 12 | 11 | 1808 | 120 | student | 4 | **engine_death** | 1928 | 2.0 | 59 | 14.86 |
| 13 | 11 | 1836 | 90 | student | 5 | **engine_death** | 1928 | 2.0 | 45 | 14.62 |
| 13 | 11 | 1836 | 90 | teacher_ranked | 6 | **engine_death** | 1871 | 2.0 | 17 | 13.97 |
| 13 | 11 | 1836 | 90 | teacher | 7 | **engine_death** | 1886 | 2.0 | 24 | 14.36 |
| 14 | 11 | 1868 | 60 | student | 0 | **engine_death** | 1928 | 2.0 | 29 | 14.44 |
| 14 | 11 | 1868 | 60 | teacher | 2 | **engine_death** | 1898 | 2.0 | 14 | 14.10 |
| 14 | 11 | 1868 | 60 | teacher_ranked | 4 | **engine_death** | 1925 | 2.0 | 28 | 14.42 |
| 15 | 11 | 1896 | 30 | student | 0 | **engine_death** | 1928 | 2.0 | 15 | 14.30 |
| 15 | 11 | 1896 | 30 | t03_engine_verified | 1 | **engine_alive** | – | 2.0 | 59 | 15.31 |
| 15 | 11 | 1896 | 30 | teacher | 12 | **engine_death** | 1927 | 2.0 | 15 | 14.39 |
| 16 | 12 | 1496 | 300 | student | 0 | engine_alive | – | 2.0 | 59 | 12.46 |
| 16 | 12 | 1496 | 300 | t03_engine_verified | 1 | engine_alive | – | 2.0 | 59 | 12.30 |
| 16 | 12 | 1496 | 300 | teacher | 3 | engine_alive | – | 2.0 | 59 | 12.20 |
| 17 | 12 | 1540 | 255 | teacher | 2 | engine_alive | – | 2.0 | 59 | 12.55 |
| 17 | 12 | 1540 | 255 | student | 3 | engine_alive | – | 2.0 | 59 | 12.64 |
| 17 | 12 | 1540 | 255 | teacher_ranked | 4 | engine_alive | – | 2.0 | 59 | 12.66 |
| 18 | 12 | 1584 | 210 | teacher | 2 | engine_alive | – | 2.0 | 59 | 12.85 |
| 18 | 12 | 1584 | 210 | teacher_ranked | 4 | engine_alive | – | 2.0 | 59 | 13.10 |
| 18 | 12 | 1584 | 210 | student | 7 | engine_alive | – | 2.0 | 59 | 13.24 |
| 19 | 12 | 1632 | 165 | teacher | 0 | engine_alive | – | 2.0 | 59 | 14.02 |
| 19 | 12 | 1632 | 165 | t03_engine_verified | 1 | engine_alive | – | 2.0 | 59 | 13.24 |
| 19 | 12 | 1632 | 165 | student | 11 | engine_alive | – | 2.0 | 59 | 13.73 |
| 20 | 12 | 1676 | 120 | student+teacher | 0 | engine_alive | – | 2.0 | 59 | 13.72 |
| 20 | 12 | 1676 | 120 | teacher_ranked | 1 | engine_alive | – | 2.0 | 59 | 13.59 |
| 20 | 12 | 1676 | 120 | teacher_ranked | 2 | engine_alive | – | 2.0 | 59 | 13.48 |
| 21 | 12 | 1704 | 90 | student+teacher | 0 | **engine_death** | 1797 | 2.0 | 46 | 13.67 |
| 21 | 12 | 1704 | 90 | teacher_ranked | 1 | **engine_alive** | – | 2.0 | 59 | 13.77 |
| 21 | 12 | 1704 | 90 | teacher_ranked | 2 | **engine_alive** | – | 2.0 | 59 | 13.77 |
| 22 | 12 | 1736 | 60 | student | 0 | **engine_death** | 1797 | 2.0 | 30 | 13.30 |
| 22 | 12 | 1736 | 60 | teacher_ranked | 1 | **engine_alive** | – | 2.0 | 59 | 14.10 |
| 22 | 12 | 1736 | 60 | teacher | 7 | **engine_death** | 1856 | 2.0 | 59 | 14.18 |
| 23 | 12 | 1764 | 30 | student | 0 | **engine_death** | 1797 | 2.0 | 16 | 13.33 |
| 23 | 12 | 1764 | 30 | t03_engine_verified | 1 | **engine_death** | 1794 | 2.0 | 14 | 13.30 |
| 23 | 12 | 1764 | 30 | teacher | 7 | **engine_death** | 1808 | 2.0 | 21 | 13.51 |

臂类型合计（同一 rollout 可属两条臂名，`armed_as`）：教师 24（13 存活 / 11 死亡）、
学生 14（14 / 10）、T03 引擎已验证 11（8 / 3）、未验证 ranked 填充 23（16 / 7）。

### 5.2 每状态汇总（安全集合 = 本轮测过的存活臂集合）

| # | seed | 帧 | 距首死 | 臂数 | 裁决 | 安全集合 | 既有死亡又有存活 | 剔除填充臂后 | 教师轮 \|S\|（T03） |
| ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 0 | 10 | 1240 | 300 | 3 | 0/1/2 alive | [0,1,2] | 否 | 否 | 18 |
| 1 | 10 | 1284 | 255 | 3 | 0/1/2 alive | [0,1,2] | 否 | 否 | – |
| 2 | 10 | 1328 | 210 | 3 | 0/1/2 alive | [0,1,2] | 否 | 否 | – |
| 3 | 10 | 1376 | 165 | 3 | 0/1/3 alive | [0,1,3] | 否 | 否 | 18 |
| 4 | 10 | 1420 | 120 | 3 | 7/11/13 alive | [7,11,13] | 否 | 否 | – |
| **5** | 10 | 1448 | 90 | 3 | 6 alive, 7 dead, 8 dead | **[6]** | **是** | 否 | – |
| 6 | 10 | 1480 | 60 | 3 | 全死 | **[]** | 否 | 否 | – |
| 7 | 10 | 1508 | 30 | 3 | 全死 | **[]** | 否 | 否 | 18 |
| 8 | 11 | 1628 | 300 | 3 | 0/1/3 alive | [0,1,3] | 否 | 否 | 18 |
| 9 | 11 | 1672 | 255 | 3 | 0/1/2 alive | [0,1,2] | 否 | 否 | – |
| **10** | 11 | 1716 | 210 | 3 | 0 alive, 1/2 dead | **[0]** | **是** | 否 | – |
| **11** | 11 | 1760 | 165 | 3 | 0 alive, 1/7 dead | **[0]** | **是** | **是** | 18 |
| 12 | 11 | 1808 | 120 | 3 | 全死 | **[]** | 否 | 否 | – |
| 13 | 11 | 1836 | 90 | 3 | 全死 | **[]** | 否 | 否 | – |
| 14 | 11 | 1868 | 60 | 3 | 全死 | **[]** | 否 | 否 | – |
| **15** | 11 | 1896 | 30 | 3 | 0/12 dead, 1 alive | **[1]** | **是** | **是** | 18 |
| 16 | 12 | 1496 | 300 | 3 | 0/1/3 alive | [0,1,3] | 否 | 否 | 18 |
| 17 | 12 | 1540 | 255 | 3 | 2/3/4 alive | [2,3,4] | 否 | 否 | – |
| 18 | 12 | 1584 | 210 | 3 | 2/4/7 alive | [2,4,7] | 否 | 否 | – |
| 19 | 12 | 1632 | 165 | 3 | 0/1/11 alive | [0,1,11] | 否 | 否 | 18 |
| 20 | 12 | 1676 | 120 | 3 | 0/1/2 alive | [0,1,2] | 否 | 否 | – |
| **21** | 12 | 1704 | 90 | 3 | 0 dead, 1/2 alive | **[1,2]** | **是** | 否 | – |
| **22** | 12 | 1736 | 60 | 3 | 0/7 dead, 1 alive | **[1]** | **是** | 否 | – |
| 23 | 12 | 1764 | 30 | 3 | 全死 | **[]** | 否 | 否 | 18 |

### 5.3 安全集合统计（本轮口径）

| 量 | 值 |
| --- | ---: |
| 状态 / 臂 | 24 / **72**（每条臂 3 个不同首动作） |
| `queried` / `executed` / `engine_death` / `engine_alive` / `unknown` | 0 / 0 / **29** / **43** / **0** |
| 安全集合大小直方图（0..4） | `{0: 6, 1: 5, 2: 1, 3: 12, 4: 0}` |
| 空安全集合数（测过的臂全死） | **6** |
| 全存活状态数 | **12** |
| 全死状态数 | **6** |
| **既又有死亡又有存活（本轮 proper 定义）** | **6** |
| 其中剔除未验证填充臂后仍成立 | **2**（状态 11、15） |
| 18 候选口径的 proper（只在“18 个候选全测过”的状态上才计入，本轮 0 个） | **0** |
| 每状态至少 3 条臂 | **24/24** |
| 来自已验证来源的臂（`verified_mask` / T03） | **11**（全部来自 T03；`verified_mask` 为 0） |
| 前缀逐帧 HASH 校验 | **115,200 次 / 0 处不一致** |
| `unknown` | **0**（无原因可列） |

**本轮的 proper 定义写清**：`0 < |S| < 测过的臂数`，即“这一状态测过的 3 条臂里既有死亡又有存活”。
这是教师续行轮里为 0 的那个量；它**不是**门的 18 候选定义（`0 < |S| < 18`），
两者不混用、不相加。

### 5.4 与教师续行轮的 2×2 对照

同一状态、同一 H=120，行 = T03 教师续行（18 候选）的裁决，列 = 本轮学生续行是否出现死亡。
只有 9 个状态同时有两边数据（本轮另外 15 个偏置不在 T03 试点里）：

| | 学生续行有死亡 | 学生续行全存活 |
| --- | ---: | ---: |
| **教师续行 18/18 全存活** | **4** | **5** |
| 教师续行有死亡 | 0 | 0 |

* 4 个“教师续行全存活、学生续行死亡”的状态：**7（seed 10/帧 1508/首死前 30）、
  11（11/1760/165）、15（11/1896/30）、23（12/1764/30）**。这是 WP420-02 §5
  预设的形态：“若教师续行安全、学生续行失败，优先查后续控制和状态覆盖”。
* 5 个两边都存活：0、3、8、16、19（首死前 165–300 帧，死亡落在 H 之外）。
* 教师续行有死亡的行是 0——与前一轮 720 条分支零死亡一致。
* **注意方向**：状态 11 里死的是**教师动作**（cand 7）和 T03 验证过的候选 1，活的是**学生自己**的动作 0；
  状态 15 里死的是学生自己的动作 0，活的是 T03 验证过的候选 1。所以这不是“学生一定更差”，
  而是“首动作 + 后续控制确实改变结果”——正是要区分的那种状态。

### 5.5 学生臂复现录制轨迹（本轮最强的自检）

| 检查 | 结果 |
| --- | --- |
| 学生在状态上的前向 argmax == 录制轨迹在该决策的动作 | **24 / 24** |
| 学生臂死亡时死亡帧 == 录制首死帧 | **10 / 10**（1541 / 1928 / 1797，逐 seed 精确相同） |
| 学生臂存活且录制死亡在 H 之外 | 14 / 14（首死帧距状态 ≥ 121 帧） |
| 状态前缀 HASH 逐帧校验 | 115,200 / 0 不一致 |

这说明 env 角色里的 obs-v3 栈（最后 4 帧 render）、动作历史（最近 8 个 2 帧决策）和
冻结学生的前向都重建正确；也说明 §5.1 的死亡不是“学生乱走”造成的噪声，而是**录制轨迹本来就在那里死**。

死亡与视界的边界很清晰：首死前 **120 帧整**（状态 12：1808+120 = 1928）算死亡；
**121 帧**（状态 4：1420+121 = 1541 > H）算存活。H=120 的截断效应必须与这些数字一起读。

### 5.6 重复一致（试点规模）

| 状态 | seed | 帧 | 臂 | cand | 两次 trace sha256 | 裁决 / 死亡帧 | 一致 |
| ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| 5 | 10 | 1448 | teacher_ranked（rank 3） | 8 | `73f376f6…3fc7493` / 同 | engine_death / 1526 同 | 是 |
| 6 | 10 | 1480 | teacher | 3 | `fad7607d…1f13f50f` / 同 | engine_death / 1541 同 | 是 |
| 8 | 11 | 1628 | t03_engine_verified | 1 | `936c32cc…d59995` / 同 | engine_alive / – 同 | 是 |

3/3 逐比特一致（同状态、同臂、同进程、两次独立 RESET + 重放）。

---

## 6 G420-1 的当前状态：**不通过**（据实），距离如下

`summary.g420_1` 由代码从数据算出（`counterfactual.g420_1_contrast_verdict`），不是手写结论。

| 门的要求 | 现状 | 还差 |
| --- | ---: | ---: |
| 200 个状态，全部 18 候选验证 | **24**（且本轮每状态只测 3 个首动作，**没有**做 18 候选） | 176 个状态 × 18 候选 |
| ≥100 个有区分度的有效状态 | 本轮口径 **6**（剔除填充臂后 2） | 94（按本轮 6/24 = 25% 的比率，200 状态只外推到约 50，仍不到 100） |
| 50 个失败相关状态的学生续行对照 | **24** | **26** |
| 每个状态 18/18 分支完整 | 本轮不适用（3 条臂，`complete = true`） | 需要 18 候选的完整裁决 |
| 重复分支一致 | **3 对 3/3** 逐比特一致 | 门要的是 200 状态全量重复 |
| ≥10 条互不相同的训练祖先 | **3** 条 seed 祖先（3 条轨迹记录） | 7 条 |

**距“200 状态全候选”还差什么（具体到可执行）**：

1. **176 个状态 × 18 候选**的教师续行验证。按 T03 实测 4.37 s/分支（H=120，shadow，
   含每分支自己的 RESET + 前缀）≈ **3,168 条分支 ≈ 3.8 h**（不含 20 状态试点已跑的部分）。
2. **26 个失败相关状态的学生续行对照**。本轮实测 13.83 s/臂、3 臂/状态 →
   26 状态 × 3 臂 ≈ **18 min**（空载主机）。
3. **若把这 200 个状态也放到学生续行下跑 18 候选**：3,600 条臂 × ~13.8 s ≈ **13.8 h**，
   这是“用学生续行当验证协议”的真实价格（也是本轮 6/24 这个比率能不能撑到 100 的关键——
   3 臂只采样了 18 个首动作里的 3 个，18 候选下的混合集合概率只会更高，但没有实测）。
4. **祖先数量**：当前语料只有 3 个 train seed shard（10/11/12），补到 10 条需要新的 train seed
   或新的轨迹类型，这一项本轮**无法**通过加状态解决。
5. **门本身的读法**：G420-1 的 100 个“有效状态”定义在 **18 候选**验证上，而 T03 已经证明
   **教师续行下 18 候选零死亡**（proper = 0）。本轮证明了**学生续行下区分度存在**，
   但**没有**把“18 候选 + 学生续行”跑出来。所以本轮**不改变**门的通过状态；
   要让门可达，必须（a）把 200 状态的主体验证也放到学生续行（或其它能产生死亡的续行）下，
   或者（b）先按计划改门/改协议——这两件事都不是本轮能自行宣布的。

---

## 7 成本

### 7.1 3 状态成本探针（派发要求：先测 3 个状态再决定 20 还是 10）

`.tmp_v420_data/T03/raw_student/cost3.json`，`--state-range 0:3`，3 状态 × 3 臂 = 9 臂：

| 量 | 值 |
| --- | ---: |
| 状态 / 臂 | 3 / 9 |
| 引擎帧 / 学生前向 | 12,636 / 540 |
| 墙钟 | **173.7 s**（2 min 54 s） |
| 每臂 | **19.30 s** |
| 帧率 / 每帧 | **72.76 fps / 13.744 ms** |
| 每前向 | **36.2 ms** |
| 裁决 | `engine_alive = 9`，`unknown = 0` |

**这批数字是争用条件下的**：跑这条命令时另一个代理正在跑 T05 实验
（两个 `th08-modern` 实例，`loadavg 13.85`），所以帧率只有 72.8 fps、学生前向被
CPU 饥饿拉到 36 ms（正常 1.6–1.7 ms，§7.2）。它仍然是**有效上界**，也是把
“20 状态 × 3 臂”排成 4 条短命令而不是 1 条长命令的依据：
按 19.3 s/臂估算，20 状态 × 3 臂 ≈ 19 min，必须分片。

### 7.2 正式跑的实测（4 个分片，空载主机 `loadavg ~0.1`）

| 分片 | 状态 | 臂 | 引擎帧 | 学生前向 | 墙钟 s | fps | 每臂 s | 每前向 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `--state-range 0:6` | 6 | 18 | 27,905 | 1,085 | 216.0 | 129.2 | 12.00 | 1.623 |
| `--state-range 6:12` | 6 | 18 | 34,197 | 903 | 271.8 | 125.8 | 15.10 | 1.724 |
| `--state-range 12:18` | 6 | 18 | 32,814 | 743 | 258.9 | 126.8 | 14.38 | 1.627 |
| `--state-range 18:24` | 6 | 18 | 32,109 | 912 | 249.4 | 128.7 | 13.86 | 1.751 |
| **合计（进程内）** | **24** | **72 + 3 重复** | **127,025** | **3,643** | **996.1** | **127.5** | **13.83** | **1.681** |

分解（合计）：`RESET` 75 次 / **54.3 s**（0.72 s/次，无一次被拒）、前缀重放
**119,756 帧 / 833.1 s（143.7 fps）**、分支帧 **7,269 帧**、学生前向 **6.12 s**、
`encode_planes_v3` **7.23 s**、整段墙钟 **996.1 s**。每帧 **7.84 ms**。
**教师查询 0、教师内部帧 0。**

读法：

* **成本主项是前缀重放**（833 s / 996 s = **84%**），因为 env 角色不能 fork、每条臂都要
  自己 RESET + 重放到停放点；学生前向只占 0.6%，`RESET` 占 5.5%。
* 因此“减少臂数”省不下大头：24 状态 × 3 臂 → 72 臂，若压到 2 臂/状态是 48 臂，
  但前缀要付 48 次而不是 72 次，约省 1/3（~11 min 而不是 16.6 min）。
* 预注册成本模型（`CONTRAST_COST_MODEL`，用 `PILOT_COST.md` 的争用单价 13.28 ms/帧、
  每臂一次引擎启动 5.89 s）对 24 状态 × 3 臂预测 **36.54 min**，实测 **16.6 min**，
  比值 **0.45**：空载主机每帧 7.84 ms（而非 13.28），且一个进程按 seed 复用、
  只在 RESET 上付费，而不是每臂重启一次引擎。**模型偏保守，方向与 T03 §6.2 一致。**
* 预算占比（v4.2.0 README §8）：外层反事实续行帧 7,269 / 1,500,000 = **0.5%**；
  教师查询 0 / 650,000；教师内部帧 0 / 64,000,000。未触发任何 80% 停线。
  （前缀重放的 119,756 帧是“重放已录制轨迹”，不是新增反事实续行帧，单列备查。）
* 产物里的 `cost.branch_s` 本次为 0.0（该字段在本次运行里声明了但没有累加，属报表疏漏）；
  逐臂墙钟在 `states[].arms[].wall_s`，其合计就是本节的整体墙钟，未受影响。

---

## 8 边界与限制（必读，全部据实）

1. **只测 3 个首动作，不是 18 个**。本轮的安全集合是“这 3 条臂里谁存活”，
   `proper = 6` 是本轮口径，**不能**当成 18 候选的 proper，也**不进** G420-1 的分母/分子。
2. **第三条臂（`verified_mask` 的已验证替代）是 `n/a`**，理由与证据见 §4.3（9,898/9,898 与
   19,772/19,772 行都只验证教师选中的那一个动作）。**没有**用未验证动作冒充它。
   替代来源只有两种，都逐条标注：(a) T03 引擎验证存活（11 条臂，`source_verified = true`）；
   (b) 未验证的语料 rank-2/rank-3 填充（23 条臂，`source_verified = false`）。
   摘要另给 `proper_without_unverified_filler = 2`，让读者自己看区分度是否依赖 (b)。
3. **教师动作是标签，不是查询**：`teacher_queries = 0`。学生轨迹状态上教师没有在该状态决策过，
   所以这条臂是“教师在同一决策槽上的录制动作”（教师语料是同一次 RESET 的另一条轨迹）。
   真正的“教师在该状态的动作”需要教师查询，本轮预算不允许（也不该：本轮是学生续行实验）。
4. **教师续行参照只有 9 个状态**（T03 试点冻结的那 9 个），另外 15 个偏置没有 18 候选数据，
   所以 2×2 只覆盖 9 个；`states_without_reference = 15` 与原因逐条记在
   `teacher_round_missing_reason`。
5. **H=120 固定**，学生续行只跑到 120 帧；首死前 ≥121 帧的状态在 H 内不可能死，
   “全存活 12 个”里有 6 个是**截断效应**而不是“这些状态安全”。状态 12（正好 120 帧）与
   状态 4（121 帧）就在边界两侧，必须一起读。
6. **`unknown = 0`**：本轮没有超时、缺帧或进程故障，所以 `unknown_reasons` 为空。
   一次都没有触发 fail-closed（RESET HASH 重试 0 次、前缀重试 0 次）。
   `unknown` 在任何情况下都不进安全集合、不算死亡（有测试钉住）。
7. **状态全部来自冻结学生自己的轨迹**（seed 10/11/12，train split）；这些 seed 是 D410-S1
   语料 seed，checkpoint 在这批语料上训练过，所以“它在这里首死”是**诊断定位，不是评测**。
   本轮没有读 dev/confirm 种子（60000–60199、61000–61499）。
8. **只测了 3 条祖先轨迹**，也没有做跨面/低余量/贴边等其它分层——本轮的层只有
   “学生首死前 0.5–5 s”一类。
9. **`lives` 全是 2.0**：首死扣 1 条、剩 2 条，没有 game over；分支在首次死亡处停止。

---

## 9 开发期发现并修掉的问题（已修，最终数据不受影响）

1. **`reset_hash` 被当成标量用**：`_StudentSession.reset_with_retry` 返回 `(frame, hash)`，
   第一版 `_run_student_arm` 直接把元组当 hash 比较，导致每次 RESET 都“不匹配”→ 全部
   `unknown`（首次冒烟的 3 条臂全 unknown，退出码 5）。修好之后同一条命令 1 状态 3 臂
   40.4 s、退出码 0。
2. **`recorded_decision_cand` 差一个决策**：第一版取的是“到达该帧的动作”（`masks[frame-1]`），
   而决策 `frame` 的动作是 `masks[frame], masks[frame+1]`。于是“学生 argmax 是否复现录制动作”
   的检查在 offset 30 的状态上假报 False。改正后 24/24 全部复现。
3. **`verified_mask_census` 对 T04 2 帧数据集比错了对象**：第一版把 verified 候选与
   `executed_cand` 比，而 4 帧块只验证教师选中的候选、块的第 2 个 2 帧槽执行的是块的
   第 2 个子动作，于是把 2,117/6,592 个 slot（当时只扫到 seed 10 那个分片）误算成
   “有别的已验证候选”。改成与 `chosen` 比、并扫全 3 个 seed 之后为 **0/19,772**，
   与语料普查（0/9,898）一致（同时记录 6,288 个 slot 的 `executed_cand != chosen`——
   这是 4 帧块第 2 槽的正常形态，不是“已验证替代”）。证据 JSON 里同时保留改前计数与改后计数
   （`census_recomputed`），臂的裁决、安全集合与成本**未**被重算。

---

## 10 命令与退出码

全部用 `wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v420_t03s/<x>.sh` 调用；
`ROOT420=/mnt/d/WorkShop/Touhou_RL`、`PY420=/root/v400-venv/bin/python`、
`V420_DATA=$ROOT420/.tmp_v420_data`。env 角色一律套
`xvfb-run -a -s "-screen 0 1024x768x24"`，引擎 data 目录每次 `cp -a` 到运行目录的副本。

| # | 命令 | 退出码 | 墙钟 | 覆盖 |
| ---: | --- | ---: | ---: | --- |
| 1 | `PYTHONPATH=$ROOT420 $PY420 tools/rl/v420.py counterfactual --config configs/v420/counterfactual-student.json --rollout-policy student --out $V420_DATA/T03/contrast_states_frozen.json`（dry-run，`env -u DISPLAY`） | **0** | 1 s | 24 状态冻结清单 + 成本估算（36.54 min @3 臂） |
| 2 | `run_contrast.sh 0:1 smoke02`（开发冒烟） | **0** | 41 s | 1 状态 3 臂，全部 alive |
| 3 | `run_contrast.sh 7:8 smoke09`（开发冒烟，正对照） | **0** | 46 s | offset 30 状态 3 臂全死，学生臂死亡帧 1541 == 录制 |
| 4 | `run_contrast.sh 0:3 cost3`（**3 状态成本探针**） | **0** | 183 s | 9 臂，§7.1（争用） |
| 5 | `run_contrast.sh 0:6 s0-6` | **0** | 221 s | 6 状态 18 臂 |
| 6 | `run_contrast.sh 6:12 s6-12` | **0** | 290 s | 6 状态 18 臂 |
| 7 | `run_contrast.sh 12:18 s12-18` | **0** | 273 s | 6 状态 18 臂 |
| 8 | `run_contrast.sh 18:24 s18-24` | **0** | 263 s | 6 状态 18 臂 |
| 9 | `merge_contrast.sh`（`--merge` 4 个分片） | **0** | 1 s | `contrast_student.json` |
| 10 | `summarize_contrast.sh` → `summary_contrast.txt` | **0** | 1 s | 本文件 §5 的表 |
| 11 | `make_evidence.sh` → 本目录 `student_continuation.json` | **0** | 1 s | 证据 JSON（裁 trace + census 重算） |
| 12 | `crosstab.py` | **0** | 1 s | §5.5 / 每偏置 / 每 seed 交叉表 |
| 13 | `accept_wsl.sh both` | **0** | 68 s | §11 |
| 14 | Windows：`python -m compileall -q touhou_rl\v420` + `pytest ... tests/test_v420_counterfactual.py` | **0 / 0** | 12 s | §11 |

**没有任何一条命令超过 10 分钟**：最长的是分片 #5–#8（221–290 s）与成本探针（183 s）。

---

## 11 测试与两侧统计（本轮实跑）

| 侧 | 命令 | 结果 | 退出码 |
| --- | --- | --- | ---: |
| WSL | `PYTHONPATH=$PWD /root/v400-venv/bin/python -m pytest -p no:cacheprovider --basetemp $PWD/.tmp_v420_data/tests/T03s -q tests/test_v420_counterfactual.py` | **69 passed** | **0** |
| WSL | 全部 v420 测试（8 个文件） | **290 passed** | **0** |
| WSL | `tools/rl/v420.py --help` | 8 个子命令 + `--rollout-policy` | **0** |
| Windows | `python -m compileall -q touhou_rl\v420` | 无输出 | **0** |
| Windows | `python -m pytest -p no:cacheprovider --basetemp .tmp_v420_pytest_win8 -q tests/test_v420_counterfactual.py` | **69 passed** in 11.5 s | **0** |

基线：改前 `tests/test_v420_counterfactual.py` **46 passed**，全部 v420 **267 passed**；
改后 **69 / 290**，其它文件仍全绿（差值是本轮追加的 23 个测试）。

新增测试覆盖（全部纯逻辑，假 env / 假学生，不启动引擎）：

* **两条 rollout 的路由**：`ROLLOUT_POLICIES == ("teacher","student")`；教师 pass 接受 H=32、
  学生 pass 只接受 H=120（32/600 都被拒）；`validate_contrast_plan` 的偏置窗口校验。
* **学生模式与教师模式的事实分离**：学生摘要用 `proper_this_round`，
  `unknown` 不进任何真值表，`teacher_queries = 0`。
* **`unknown` 不判死**：注入 `TimeoutError` 后 `outcome = unknown`、`engine_dead = None`、
  不进安全集合；全 unknown 状态既不算 empty 也不算 all-dead。
* **死亡帧记录**：假引擎在帧 66 置 deaths=1，臂的 `first_death_frame = 66`、
  trace 末行帧号 66、`deaths_end = 1.0`、`lives_end = 2.0`、每帧都有一行。
* **臂集合去重与替代动作选择**：教师==学生 → `duplicate` 且只跑一次；缺 verified 时
  slot 3 记 `n/a`（带原因，不用未验证动作冒充）；填充臂按 rank 2/3 补到 3 个不同首动作；
  有 T03 池时第三臂来自引擎验证（`source_verified = true`）。
* 另加：`verified_mask` 普查、T03 参照读取与 2×2、摘要的 proper/empty/all-alive/all-dead、
  合并拒绝重叠与缺口、`trim_report` 也裁 `arms` 的 trace。

---

## 12 复算与产物路径

| 路径 | 内容 |
| --- | --- |
| `.tmp_v420_data/T03/contrast_states_frozen.json` | 24 状态冻结清单（引擎启动前写出） |
| `.tmp_v420_data/T03/raw_student/{s0-6,s6-12,s12-18,s18-24,cost3,smoke*}.json` | 分片报告（含逐帧 trace） |
| `.tmp_v420_data/T03/contrast_student.json` | 合并后的完整报告（含逐帧 trace，本次运行未裁剪） |
| `.tmp_v420_data/T03/contrast_student_evidence.json` | 与证据目录同一份裁剪版 |
| `Document/Update/Alpha/v4.2.0/evidence/T03/student_continuation.json` | **本目录证据 JSON**（496 KB） |
| `.tmp_v420_t03s/*.sh`、`*.py` | 本次全部脚本：`run_contrast.sh`、`merge_contrast.sh`、`summarize_contrast.py/.sh`、`crosstab.py`、`make_evidence.py/.sh`、`accept_wsl.sh` |
| `.tmp_v420_t03s/summary_contrast.txt`、`crosstab` 输出 | 本文件 §5 的原始抽取 |

复算（全部幂等，除 `run_contrast.sh` 会重跑引擎）：

```bash
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v420_t03s/merge_contrast.sh      # 重算合并报告
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v420_t03s/summarize_contrast.sh  # 重算 §5 的表
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v420_t03s/make_evidence.sh       # 重算证据 JSON
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v420_t03s/accept_wsl.sh both     # 两侧测试 + --help
```

---

## 13 不改动声明

`touhou_rl/v420/contracts.py`、`replay_audit.py`、`eval.py`、`rebuild.py`、`reslot.py`、
`dataset_2f.py`、`batch_2f.py`、`train.py`、`diagnostics.py`、`__init__.py`、
`configs/v420/{audit-time,legacy-reproduce,learn-controlled,local,protocol,counterfactual-pilot}.json`、
`Document/Update/Alpha/v4.2.0/evidence/T02/**`、`evidence/T04/**`、其它 tests、
`tools/rl/v410_student_eval.py`、`touhou_rl/v400|v410/**`、旧证据、`ASSET_SYNC.*` 全程未改；
`.tmp_v410_*` 与 `/root/v400-data/**` 只读（引擎只写自己的副本，跑完删除）；
confirm 种子 61000–61499 与保留段 60000–60199 全程未读；状态只来自 seed 10/11/12（train split）；
没有并发布两个引擎实例（本轮 4 个分片串行，另一个 T05 实验是**别的代理**的进程，
只在 §7.1 的成本探针期间与之重叠，已标注）；没有 git add/commit/push。

本轮新增/改动的文件只有：`touhou_rl/v420/counterfactual.py`、`tools/rl/v420.py`
（只接 `counterfactual` 的 `--rollout-policy`）、`configs/v420/counterfactual-student.json`（新建）、
`tests/test_v420_counterfactual.py`（只追加）、本目录两个新证据文件、
`.tmp_v420_t03s/`（脚本）与 `.tmp_v420_data/T03/`（运行数据）。
