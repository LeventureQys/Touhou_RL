# 逐 2 帧槽的教师验证标签（T02d / reslot）

日期：2026-11-13，Asia/Shanghai。资产主机（本机 WSL2），`cwd=/mnt/d/WorkShop/Touhou_RL`，
Python `/root/v400-venv/bin/python`（3.10.12），引擎 `rl_v3`
`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`
（sha256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`，
与 `shard.json.engine_shadow.sha256` **逐字符相同**，运行前校验），
教师 **shadow** 角色（`env -u DISPLAY`，不渲染像素）。git HEAD
`c3e116fd45d708dc8cbe73d4bb49cd960f835405`。

**范围声明**：本文件产出**逐 2 帧槽的教师验证标签**，补上 G420-0 的 L1 腿最后一块缺口，
并给出它在**与 T02c 重建语料 join 之后**是否成立。它**不**做语义判定、**不**产出
G420-1 结论、**不**改写 T02c 的任何产物：T02c 的 `rebuild/**` 全程只读（本文件 §9
列出双向 sha256）。教师只被"重新查询"，没有任何一次教师动作被提交给引擎。

---

## 0 结论摘要

1. **3 个 shard 全部成功**（seed 10/11/12，退出码全 0）：查询
   **19,796** 个槽边界（= 2 × 9,898 决策，**正好每 2 帧一次**），
   写出 **19,796** 条标签记录，其中 **19,772** 条是语料槽、24 条是每 shard 8 个冷启动
   边界（查询了、但不是样本，显式标记 `emitted=false`）。
2. **每个槽都有自己的 18 路 verified 标签**：19,772/19,772 = **100%**。T02c 之后语料的
   verified 覆盖是 9,886/19,796 = 49.94%，本任务把**另外 9,886 个槽**（每块的第 2 槽，
   T02c 只能记成 `executed_slot1`）补上了真正的教师评分。
3. **HASH 零差异**：本任务 19,796 次 `hash_pre` + 19,796 次 `hash_post` +
   39,592 个逐帧哈希，**0 不一致**；`RESET` 哈希与第一行 `hash_pre` 逐 shard 相等。
   join 层面共 **29,694 个边界校验 + 79,184 个帧哈希，0 差异**。
4. **G420-0 的 L1 腿成立（passed）**，6 条判据逐条通过（§4）：每 2 帧一槽、每槽有
   自己的 obs+action+label、history 8×2 帧、无未来、HASH 0 差异、逐槽断言 0 违反。
5. **这是 2 帧节拍下的新查询，不是旧标签的平移或插值**。实测：第 1 槽
   （决策帧 `4i+2`）的 18 路排序与同块第 0 槽**不同**的有 **7,718/9,886 = 78.07%**，
   verified 候选不同 **5,406/9,886 = 54.68%** —— 任何"移位/插值"都不可能产生这些值。
   附带的最强校验是：第 0 槽（决策帧 `4i`，与旧教师查询同一个状态）的 18 路向量
   **9,750/9,886 = 98.62%** 与旧标签逐值（6 位小数）相同，剩余 136 行差异有明确
   机理（§6.1）。
6. **成本**：19,796 次查询 / **1,266,944** 教师内部帧（= 19,796 × 64，与 T03 的
   `horizon × (1 + verify_k)` 公式**逐个相符**），墙钟 **1,415.8 s = 23.6 min**
   （进程墙钟 1,492 s = 24.9 min，3 条命令分别 561 / 499 / 432 s）。
   占 650k 查询上限的 **3.08%**、64M 内部帧上限的 **2.00%**；与 T03 累计后
   **35.76% / 23.24%**，剩 **417,562 次查询 / 49,123,968 内部帧**。
7. **本任务无剩余阻塞**。T04 训练所需的"每槽 18 路 verified 标签"已在 3 个 shard
   （train split，seed 10/11/12）上全部落地；剩余工作量是**规模**（seed 13–20）而不是
   可行性，另需 T02c 侧对 seed 13–20 的重建（非本任务范围）。

---

## 1 方法与口径

### 1.1 一个新模块，一条新路径

新建 `touhou_rl/v420/reslot.py`（sha256 `f4af917d…3ed83`，3 次运行内不变），
入口 `python -m touhou_rl.v420.reslot`：

```bash
PYTHONPATH=$ROOT420 $PY420 -m touhou_rl.v420.reslot \
  --repo-root $ROOT420 --v420-data $V420_DATA --seeds <N> \
  --run-id T02d --output-kind reslot \
  --out-json $V420_DATA/T02d/logs/reslot_seed<N>.json
```

输出（白名单内，`$V420_DATA/.tmp_v420_data/T02d/`）：

| 路径 | 内容 |
| --- | --- |
| `T02d/reslot/<shard-key>/teacher_2f.jsonl` | 每行一个槽边界（19,796 行/shard，10.2 MB） |
| `T02d/reslot/<shard-key>/meta.json` | 契约版本、槽计数、HASH 校验、profile 快照、engine/data_dir/源码身份、教师内部帧数（15 KB） |
| `T02d/data/<shard-key>/` | 引擎数据**私有副本**（64 KB：`score.dat`/`th08.cfg` 复制，两个大 `.dat` 仍是只读符号链接） |
| `T02d/logs/` | 逐 shard 报告 JSON、墙钟、join 审计 |

### 1.2 一个槽的时序（顺序不可换）

```
RESET 0 1 0 <seed> 0            # stage/difficulty/team/seed 同 shard.json
  → HASH == rows[0].hash_pre    # 不等则整 shard 放弃，一条都不写
  → probe_capabilities()
  → probe_child_hazards()       # 必须在第一次 decide 之前，见下
  → 对每个槽边界 d = 0,2,4,...：
        decide(engine, cfg, committed=<上一槽的已执行候选>,
               frame=d, record_source="child", shoot_release=False)
        → 断言 engine.frame() 仍是 d（查询不得移动权威进程）
        → STEP step_frames[d]   1 帧
        → STEP step_frames[d+1] 1 帧
        → 记录该槽条；若 d 是块末（d = 4i+2）则校验 HASH == rows[i].hash_post
```

**探针顺序是承重的**：`record_child_tape()` 内部自己会调 `probe_child_hazards()`，而该
探针**自己 `PARK`**，会把 `decide()` 刚取的模板顶掉，录制子进程的 `SPAWN <旧 id>` 被拒：
`EngineError: 'SPAWN 1' -> ERR spawn-template expected=2`（T03 §3.2 已复现定位）。
模块在 `RESET` 之后、第一次 `decide()` 之前无条件跑这两个探针（判决按进程缓存），
并在 `child_hazards.available` 为假时 **fail-closed**。本次 3 个 shard 的探针判决都是
`CHILD <h> HAZARDS accepted`，全程未出现该错误。

### 1.3 提交出去的掩码来自 `step_frames`，不来自教师

**本次是「重放 + 重查询」，不是「重新驱动」**：每一帧掩码都是语料 `step_frames` 解码
出来的（`C.mask_to_cand` → `C.cand_to_mask` 往返相等才接受），教师的 `chosen` 只被
**记录**、从不提交。这就是 `hash_post` 相等有意义的原因：影子逐字节走了录制轨迹，
所以 T02c 重建出的观测与本文件写出的标签描述**同一批帧**。每条记录都带
`executed.source = "step_frames"`，且 `check_label_record` 会把其它取值判为违约。

### 1.4 教师录像走的那 2 帧：`committed` = 上一槽的已执行候选

`decide()` 的 `committed` 是"上一轮计划的剩余部分"（11-P1 §2 step 1）。2 帧节拍下上一轮
计划只有一个槽（2 帧），所以每次查询传 **1 个**已执行候选 —— 它是**过去**的动作
（决策帧 `d-2`），语料的**未来**掩码从不进入教师。记录里以
`committed_tape` / `committed_frames` 留痕（第 0 槽为 `[]`）。
这与 T03 的 2 帧协议一致（`commits = actions + commits[len(actions):]` 在 `R=1` 时退化为
"上一次提交的那一个动作"）。

### 1.5 `shoot_pulse_period`：本次**缩放为 48**

它的单位是**决策**。出厂值 24 在 4 帧节拍下 = 96 帧一次"松 SHOOT"；在 2 帧节拍下
同样的 24 会变成 48 帧一次。本次用 `dataclasses.replace(..., shoot_pulse_period=48)`
把它**缩放**成仍等于 **96 帧**（记录里 `teacher.shoot_pulse_frames = 96`）。

**为什么不影响本次用途**：`decide()` 只在 `label.extra["shoot_pulse_period"]` 里回显这个
值，**不读取它做任何分支**；产生"松 SHOOT 那一帧"的是
`touhou_rl.v400.teacher.shoot.release_due()`，它只在**采集器的提交路径**里被调用
（`tools/rl/v400_teacher.py:442`），而本次模块既不调用它、也不提交教师动作。
本次取的三样东西 —— `scores[18]`、`verified_mask`、`chosen` —— 与它无关；
提交给引擎的掩码一律来自 `step_frames`。所以缩放**不改一个字**的标签，
只让记录里的数值在 2 帧节拍下仍然读作"96 帧"。若不缩放（保持 24）结果完全相同；
`--shoot-pulse-period` 可改，`meta.json` 会记录实际值。

### 1.6 profile：与旧语料同一个 `fast`，但节拍不同

```
replace(get_config("fast"), replan_decisions=1, shoot_pulse_period=48)
horizon=32  decision_frames=2  beam=24  verify_k=1  durations=(1,2)  replan_decisions=1
内部帧/查询 = horizon × (1 + verify_k) = 32 × 2 = 64
```

profile 全字段快照 + sha256 `5545c1993ddcbf5e6f0fa401bd28806e183cfd0be66f348c217dc86838bb3db2`
写进每 shard 的 `meta.json`，也写进**每条**记录（`teacher.profile_sha256`）。
`fast` 与旧语料 `shard.json.mode="fast"` 一致：`horizon/beam/verify_k/durations`
与 4 帧采集时**完全相同**，只有节拍（每 2 帧决策一次）和提交槽数（1 而不是 2）不同。

### 1.7 记录字段

每条 `teacher_2f.jsonl` 记录（契约字段在前，溯源在后）：

| 组 | 字段 |
| --- | --- |
| 身份 | `schema=v420-teacher-2f/1`、`contract/obs_contract/action_contract`、`slot_id`、`emitted`、`cold_start`、`episode_seed`、`decision_frame`、`commit_frames`、`block_index`、`slot_in_block`、`duration`/`duration_frames`（=2） |
| 已执行（来自语料） | `executed.cand/bomb/mask/frames`（2 个逐帧掩码）、`executed.source="step_frames"` |
| 教师（本次查询） | `scores`（18）、`verified_mask`（18）、`verified_count`、`chosen`（= `first_action`）、`chosen_verified`、`chosen_score`、`teacher_duration`、`bomb`、`margin`、`teacher_death_frame` |
| 重放溯源 | `hash_pre`、`hash_post`、`events`、`frame_event_flags`、`bombs_left`、`timing_ms`（record/search/verify/events）、`committed_tape`、`committed_frames` |
| profile | `teacher.{mode,horizon,decision_frames,beam,verify_k,replan_decisions,shoot_pulse_period,internal_frames,profile_sha256}` |
| 语料 | `corpus.{version,key,source_row,source_decision_frame,slot_in_block_frames,legacy_label}` |

`slot_id` 与 T02c `slots.jsonl` **同一命名**：`slot_id = (decision_frame - 16) // 2`。
也就是说槽 0 在决策帧 16、槽 1 在决策帧 18；决策帧 `4i` 与 `4i+2` 各成一槽。
决策帧 `0..14` 的 8 个边界 `slot_id` 为 `-8..-1`、`emitted=false`：
它们**被查询**（所以文件行数正好 `2 × 决策数`），但观测/历史窗口会落到第 0 帧之前，
不是样本。`tests/test_v420_reslot.py` 直接读 T02c 的 `slots.jsonl` 前 8 行与末 2 行
校核这一命名（只读）。

### 1.8 逐槽自检 + fail-closed

写盘前每个槽过 `check_label_record()`（写与审计用**同一个**实现）：

`decision_frame`、`commit_frames`、`duration`、`slot_id/emitted/cold_start`、
`scores_length`、`scores_finite`、`verified_mask`、`verified_count`（= 掩码求和）、
`verified_argmax`（该槽有 verified 时 `chosen` 必须是被 verified 的候选且
`scores[chosen]` 是 verified 里的最大值）、`chosen_range`、`bomb`、`margin`、
`executed_mask`（`cand_to_mask(cand,bomb) == mask` 且 `source=="step_frames"`）、
`executed_matches_corpus`、`teacher_profile`。

任一条不满足 → 该 shard 立刻失败；`RESET`/`hash_pre`/`hash_post` 不一致同理。
记录先写进 `teacher_2f.jsonl.tmp-<pid>`，**只有**自检通过后才 `os.replace` 到最终路径，
失败时临时文件被删除 → 失败不会留下半个语料（测试 `test_hash_mismatch_*` 与
`test_malformed_label_*` 断言了这一点）。

---

## 2 逐 shard 结果

| seed | 决策 | 帧 | 槽（=2×决策） | 语料槽（emitted） | 冷启动 | 有 18 路 scores | 有 verified 候选 | verified flags | 教师查询 | 内部帧 | `hash_pre`/`hash_post`/逐帧 | 墙钟（进程 / in-process） | ms/query | exit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 10 | 3,300 | 13,200 | 6,600 | 6,592 | 8 | 6,600 | 6,600 | 6,600 | 6,600 | 422,400 | 3,300 / 3,300 / 13,200 | 561 s / 533.4 s | 80.81 | 0 |
| 11 | 3,300 | 13,200 | 6,600 | 6,592 | 8 | 6,600 | 6,600 | 6,600 | 6,600 | 422,400 | 3,300 / 3,300 / 13,200 | 499 s / 472.8 s | 71.64 | 0 |
| 12 | **3,298** | 13,192 | 6,596 | 6,588 | 8 | 6,596 | 6,596 | 6,596 | 6,596 | 422,144 | 3,298 / 3,298 / 13,192 | 432 s / 409.6 s | 62.11 | 0 |
| **合计** | **9,898** | **39,592** | **19,796** | **19,772** | **24** | **19,796** | **19,796** | **19,796** | **19,796** | **1,266,944** | **9,898 / 9,898 / 39,592** | **1,492 s / 1,415.8 s** | **71.52** | **0** |

- **verified 覆盖率**（两种分母，必须分开读）：
  **有 verified 候选的槽 = 19,796/19,796 = 100%**（每 shard 100%）；
  **verified flag 占 18 路向量的比例 = 19,796 / (19,796×18) = 5.56%**（每槽恰好 1 个
  verified 候选，因为 `verify_k=1`，见 §6.2）。
- **HASH 校验数**：每个 shard `hash_pre` = 决策数、`hash_post` = 决策数、逐帧 = 帧数，
  不一致 **0**、`first_mismatch=null`、`reset_hash_matches_first_pre=true`。
- **教师内部帧** = 19,796 × 64 = 1,266,944，逐 shard 为 6,600×64 / 6,600×64 / 6,596×64，
  与 `horizon × (1 + verify_k)` 公式**逐槽相符**（T03 已在 773 条查询行上验证该公式）。
- **引擎侧真正推进的帧** = 39,592（每帧一个 `STEP mask 1`，与语料 13,200+13,200+13,192
  逐帧一致），不含教师 fork 里的帧。
- seed 12 只有 3,298 决策 / 13,192 帧：该 shard `stop_reason="stage_changed"`、
  `stage_clear=true`、`stopped_early=true`（T02c §4.5 已记录）。**做 T04 的覆盖率分母
  必须用实际行数，不能用 3,300。**

---

## 3 与 T02c 的 rebuild 产物 join 之后（只读）

join 键 = `slot_id`（等价于 `decision_frame`），T02c 19,772 行 ↔ 本任务 19,772 条
`emitted` 记录，**一一对应无缺漏**（`set(rebuild.slot_id) == set(reslot.slot_id)`，
每 shard 6,592 / 6,592 / 6,588）。join 只读 T02c 的 `2f/slots.jsonl`、`meta.json`、
`obs/*.npy`，未改一个字节（§9 给出双向 sha256）。

| 完整性项 | 计数 | 说明 |
| --- | ---: | --- |
| `slots_total` | **19,772** | 3 个 shard 合并后的槽数（6,592 + 6,592 + 6,588） |
| 有自己 play 4 帧栈 | **19,772** | `play96x112.npy[slot_id]` 非全零；4 帧窗口 = `d-4..d-1` |
| 有自己 HUD | **19,772** | `hud.npy[slot_id]` 非全零；HUD 帧 = `d-1` |
| 有自己已执行动作 | **19,772** | `executed.cand/mask` 与 T02c 的 `cand/mask/bomb` **逐槽相同**（两条独立解码路径：T02c 从 `step_frames` 直接解，本任务从同一个 `step_frames` 走 `mask_to_cand` 往返） |
| 有自己 verified 教师标签 | **19,772** | 每槽 18 个 `scores` + 18 位 `verified_mask` + `verified_count=1`；`scores` 长度非 18 的 **0** 条、`verified_count` 与掩码不一致的 **0** 条、`scores[chosen]` 不是 verified 最大值的 **0** 条 |
| 8 条真实历史 | **19,772** | T02c 的 `history` 与**本任务独立解码**的已执行动作在 `d-16,d-14,…,d-2` 上**逐条相等**（19,772/19,772）；跨度 16 帧，每槽 8 条 |
| 自检违反数 | **0** | T02c 3 个 shard `self_check.passed=true` 且 `violations_total=0`；本任务 3 个 shard 同样 |
| HASH 不一致数 | **0** | join 层共 **29,694** 个边界校验（T02c 9,898 个 `hash_post` + 本任务 9,898 `hash_pre` + 9,898 `hash_post`）与 **79,184** 个帧哈希（两侧各 39,592），差异 0 |

补充（"两条腿描述同一批帧"的直接证据）：

- `commit_frames`、`block_index`、`slot_in_block` 在 19,772 个槽上两侧**逐字段相同**；
- `act_hist[slot_id]` 与 `onehot(history)` 逐槽相等（19,772/19,772）；
- 观测窗口恰为 `d-4..d-1`（19,772/19,772），`commit_frames == [d, d+2)`（19,772/19,772）。

---

## 4 G420-0 的 L1 腿是否成立：**成立（passed）**

判据逐条（数据来自 `reslot_2f.json` 的 `l1_leg`，全部 `passed=true`）：

| # | 判据 | 实测 | 判定 |
| ---: | --- | --- | --- |
| 1 | **每 2 帧一槽** | 2 × 9,898 = **19,796** 次查询，一个决策边界不漏；合并语料 19,772 = 19,796 − 24 冷启动；每槽 `decision_frame = 16 + 2·slot_id` 且 `commit_frames=[d,d+2)`（19,772/19,772） | **PASS** |
| 2 | **每槽有自己的 obs + action + label** | play 19,772、HUD 19,772、已执行动作 19,772、verified 教师标签 19,772；`scores` 形状违约 0、`verified_count` 违约 0、argmax 违约 0 | **PASS** |
| 3 | **history 8×2 帧** | 19,772/19,772 个槽的 8 条历史都是**真实执行过**的 2 帧动作（`d-16..d-2`，跨度 16 帧），并与 T02c 的记录逐条相等 | **PASS** |
| 4 | **无未来** | 历史最后帧 `d-2`、观测最后帧 `d-1`，均 `> d` 的帧数 = **0**；观测窗口非过去的 0 条；标签在第 d 帧的查询之后才提交第 d/d+1 帧的掩码 | **PASS** |
| 5 | **HASH 0 差异** | 29,694 边界校验 + 79,184 帧哈希，mismatch **0**；`RESET` 哈希逐 shard 等于第一行 `hash_pre` | **PASS** |
| 6 | **逐槽断言 0 违反** | 两侧自检 `violations_total` 合计 **0**（本任务 3 shard × 0，T02c 3 shard × 0） | **PASS** |

也就是说：**在 seed 10/11/12 的合并语料上，"每 2 帧一个槽、每槽有自己的
obs+action+verified 标签、history 8×2 帧、无未来、HASH 0 差异、逐槽断言 0 违反"
全部成立**。这**不是** G420-0 整门通过（G420-0 还含部署输入边界测试与冻结模型
在线/离线前向一致性等其它腿），也**不是** G420-1：本文件不产生任何安全集合、
不做语义判定，标签是否"用得上"由 T04 判定。

---

## 5 三方对照

| 项 | 旧语料（v4.1.0 4 帧 shard） | T02c：仅 rebuild | **合并 + T02d reslot** |
| --- | --- | --- | --- |
| 决策节拍 | 每行提交 4 帧（2 个槽压在一行） | 每 2 帧一槽 | 每 2 帧一槽 |
| 槽数登记 | 19,796 个逻辑槽，只登记 9,898 行 | 期望 19,796 / 写出 19,772（+24 冷启动排除） | **19,772，每槽一行** |
| 观测覆盖 | 9,898/19,796 = **50%** | 19,772/19,796 = 99.88% | **19,772/19,772 = 100%** |
| HUD 覆盖 | 每块 1 行（帧 `4i-1`）= 50% | 19,772/19,796 = 99.88%（槽 1 用帧 `4i+1`） | **100%** |
| 已执行动作 | 19,796（`step_frames[0]`/`[2]` 可解码，但槽 1 无独立行） | 19,772 行 | **19,772 行，`source="step_frames"`** |
| 教师 18 路标签 | 9,898（只覆盖槽 0）= 50% | **9,886/19,796 = 49.94%（未改变）** | **19,772/19,772 = 100%**（每槽 1 个 verified flag = 5.56% 的向量位） |
| 槽 1 的标签 | 不存在 | `executed_slot1`、`label_verified=false`、**无 scores** | **`scores`+`verified_mask`+`chosen`（2 帧节拍的新查询）** |
| `history` 跨度 | 8 × 4 = 32 帧 | 8 × 2 = 16 帧 | 8 × 2 = 16 帧 |
| HASH 粒度 | 每决策 1 个边界（6,600 对记录值） | 9,898 个 `hash_post` + 39,592 帧哈希 | **+9,898 `hash_pre` +39,592 帧哈希 → 29,694 边界 / 79,184 帧** |
| 墙钟 | 采集时约 1 天（含环境进程） | 688.3 s（3 shard，`role=env` 渲染像素） | **+1,415.8 s（3 shard，`role=shadow` 无像素）** |
| 可训练样本 | 9,898 verified + 9,898 BC 目标 | 9,886 verified + 9,886 BC 目标 | **19,772 条"obs+action+verified 标签"齐全的槽** |

---

## 6 边界（必须与上面的"成立"一起引用）

### 6.1 这是 2 帧节拍下的新查询，不是旧标签的平移或插值

- **第 1 槽（决策帧 `4i+2`）从来没有被旧教师查询过**。它是本次的**新查询**：
  它的 18 路排序与同块第 0 槽不同的占 **78.07%**，verified 候选不同的占 **54.68%**。
  "把槽 0 的分数搬过去"或用插值补一个向量，都会和这 7,718/5,406 条实测矛盾。
- **第 0 槽**（决策帧 `4i`）与旧教师查询**同一个状态、同一个 profile**，所以本次
  重新查询的结果 **9,750/9,886 = 98.62%** 与旧标签逐值（6 位小数）相同 ——
  这是"同规则在同状态上重算一次"的重现，不是复制（记录里没有任何字段被写死成旧值；
  `timing_ms.record/search/verify` 每次都是真实测量）。
- 剩下 **136 行**（1.38%）有差异，机理明确：教师录像走用的是
  `committed`（上一轮计划的剩余部分），2 帧节拍下它是**上一槽 1 个动作**
  （§1.4），而旧采集器在 4 帧节拍下传的是**上一块 2 个动作**；于是录像头 2 帧
  （旧的是头 4 帧）走的不是同一动作，瞄准玩家的敌弹方向随之不同，
  18 条候选的终端几何分因此最多变动 **10.81**（136 行里多数只动 1–18 个分量）。
  这 136 行**恰好证明标签是算出来的**：如果是复制或平移，就不可能既有 98.62% 的
  逐 bit 相符、又在有分歧的地方给出不同的 verified 候选（`verified_mask` 与
  `chosen` 一致率 9,825/9,886 = 99.38%）。
- **因此不能把两边的 `scores[18]` 当"逐值可比"来用**：第 1 槽的向量在旧语料里
  **不存在**（没有可比对象），第 0 槽的重现只是校验而不是"同一份数据"。
  任何"新旧 scores 做配对统计"的分析都必须先声明它比的是哪一批槽，
  并且不能用旧标签去代表 2 帧节拍的标签。

### 6.2 `verify_k=1` 的含义：每槽只有 1 个 verified flag

`fast` 的 `verify_k=1` 意味着每次查询只把 1 个候选送进引擎 fork 校验，另外 17 个
候选的分数带 `-5.0` 的 `unverified_discount`。所以：
**verified 槽覆盖率 100%，但 verified flag 只占向量的 5.56%（1/18）**。
这与旧语料完全一致（旧语料 9,898 行全部 `verified_count=1`），不是本任务引入的
退化。若要更多 verified 候选，需要改 `verify_k`（成本与内部帧按 §7 线性放大），
这是 T04 之前的产品决策，本文件不擅自改。

### 6.3 seed 12 只有 3,298 决策

`stop_reason="stage_changed"`、`stage_clear=true`。所以 3 个 shard 的分母是
6,600 + 6,600 + 6,596 = 19,796（**不是** 19,800），合并语料 19,772
（**不是** 19,776）。T04 做覆盖率时必须用**实际行数**（T02c §4.5 同结论）。

### 6.4 冷启动 8 槽被查询但不入语料

决策帧 `0,2,…,14` 的 8 个边界每 shard 都被查询并写进 `teacher_2f.jsonl`
（`slot_id=-8..-1`、`emitted=false`、`cold_start=true`），因为它们的 4 帧观测栈或
8 条历史会落到第 0 帧之前。**不做 pad、不复制第一帧、不留白名单例外**；
24 条这样的记录在 3 个 shard 里各 8 条，join 时按 `slot_id >= 0` 排除。

### 6.5 没有真实时间戳

与旧语料、T02c 一致：只有引擎逻辑帧号（`FRAME`/`STEP` 的 `frame` 字段）。
本文件任何"时序一致"都指**帧号对齐**，不声称时间戳。

### 6.6 role 与像素

本任务教师用 **shadow**（`role="shadow"`，无 GL、不渲染）。shadow **不能**发布像素，
所以观测腿**必须**来自 T02c 的 `role="env"` 重建（688.3 s / 3 shard）；
本任务只补标签腿（1,415.8 s / 3 shard）。两条腿的引擎 sha 相同
（`691bfd77…`），角色不同是有意的。

### 6.7 数据目录用副本，源资产只读

每个 shard 先把 `/root/v400-data/s1c/<key>/env` 复制到
`$V420_DATA/T02d/data/<key>/`（64 KB：`score.dat`/`th08.cfg` 复制，两个大 `.dat`
仍是只读符号链接），再让引擎在该副本上跑。`source_score_dat_unchanged=true`
（3 个 shard 的 `score.dat` 运行前后 sha256 相同），`/root/v400-data/**` 与
`.tmp_v410_*` 全程只读。

---

## 7 资源账（650k 查询 / 64M 教师内部帧）

上限口径沿用 T03：**650,000 次教师查询**、**64,000,000 教师内部帧**。

| 项 | 查询数 | 内部帧 |
| --- | ---: | ---: |
| **T02d 正式重标（3 shard）** | **19,796** | **1,266,944** |
| T02d 冒烟（240 槽 + 2 协议查询） | 242 | 15,488 |
| **T02d 合计** | **20,038** | **1,282,432** |
| 占上限（T02d 单独） | **3.083%** | **2.004%** |
| T03 已用（`evidence/T03` §7.3） | 212,400 | 13,593,600 |
| **累计（T03 + T02d）** | **232,438** | **14,876,032** |
| 占上限（累计） | **35.760%** | **23.244%** |
| **剩余** | **417,562** | **49,123,968** |

- 每次查询 **64** 内部帧（`32 × (1 + 1)`），19,796 × 64 = 1,266,944 —— 与实测
  逐 shard 相等（6,600/6,600/6,596 次查询各 × 64）；T03 的公式在 773 条查询行上
  0 例外，本次在 19,796 条上再次成立。
- 单价：**71.52 ms/query**（13.98 queries/s），逐 shard 80.81 / 71.64 / 62.11 ms
  —— 与 T03 的 66.58 ms（depth 100）/ 86.18 ms（depth 1,650）区间吻合，
  且本次是**一条连续重放**（0 次额外 `RESET`、0 次前缀重放）。
- 若再做 seed 13–20（8 个 shard，约 8 × 6,600 = 52,800 次查询 / 3,379,200 内部帧）：
  查询 **+8.12 个百分点**（累计 **43.88%**）、内部帧 **+5.28 个百分点**（累计 **28.52%**），
  仍在两道上限之内。
- 磁盘：发布物 30.5 MB（3 × 10.2 MB 标签 + 3 × 15 KB meta），引擎数据副本 256 KB，
  日志 < 100 KB。**没有**新增 GB 级资产。

---

## 8 命令与退出码

全部用 `wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v420_t02d/<x>.sh` 调用。
3 个 shard **拆成 3 条独立命令**（每条一个 seed），单条最长 561 s < 10 min。

| # | 命令 | 退出码 | 墙钟 | 覆盖 |
| ---: | --- | ---: | ---: | --- |
| 1 | `smoke_engine.sh` | 0 | ~5 s | 协议冒烟：`RESET` 哈希、`STEP` 回复字段、2 次 decide |
| 2 | `run_smoke.sh` | 0 | 17.5 s | 240 槽 / 240 查询（`--decisions 120`），`ms/query=72.76` |
| 3 | `run_full.sh 10 …` | **0** | **561 s** | seed 10 全量（6,600 槽） |
| 4 | `run_full.sh 11 …` | **0** | **499 s** | seed 11 全量（6,600 槽） |
| 5 | `run_full.sh 12 …` | **0** | **432 s** | seed 12 全量（6,596 槽） |
| 6 | `run_join.sh` | **0** | 5.0 s | 与 T02c join + 完整性审计（只读） |
| 7 | `pytest.sh`（WSL 验收） | **0** | 1.5 s | `16 passed` |
| 8 | `python -m compileall -q touhou_rl\v420`（Windows） | **0** | — | 语法 |
| 9 | Windows `pytest tests/test_v420_reslot.py` | **0** | 0.3 s | `16 passed` |

验收命令（逐字）：

```bash
# WSL
PYTHONPATH=$ROOT420 $PY420 -m pytest -p no:cacheprovider \
  --basetemp $V420_DATA/tests/T02d -q tests/test_v420_reslot.py     # 16 passed, exit 0
```

```powershell
# Windows
python -m compileall -q touhou_rl\v420                                  # exit 0
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v420_pytest_win3 -q tests/test_v420_reslot.py
                                                                        # 16 passed, exit 0
```

测试覆盖（`tests/test_v420_reslot.py`，纯逻辑，注入假引擎/假教师，不连引擎）：
2 帧槽边界与 `slot_id` 命名（含**只读**校核 T02c `slots.jsonl` 前 8 行与末 2 行）、
scores 长度 18、`verified_count` 与掩码一致、`scores[chosen]` 必须是 verified 最大值、
`chosen` 不在 verified 里必须被拒、提交掩码来自 `step_frames` 而不是 `chosen`、
HASH 不一致必须抛错且**不落盘**、`RESET` 哈希不一致必须在第一次查询前抛错、
畸形标签（17 个分数 / chosen 越界）必须抛错且不落盘、探针顺序
（`reset → probe_capabilities → probe_child_hazards → 第一次 decide`）、
配置守卫（`role=env`、`replan_decisions=2`、输出越界、`.tmp_v410_` 路径、
dev/confirm/legacy_confirm 种子）。

---

## 9 原始证据（本目录）

| 文件 | 内容 |
| --- | --- |
| `RESLOT_2F.md` | 本文件 |
| `reslot_2f.json` | join + 完整性审计的机读结果：逐 shard 计数、`cross_check`、`obs`、`history`、`labels`（含 136 行差异的直方图）、`hash`、`self_check`、`identity`、`label_reproduction`、`budget`、`wall_clock`、`l1_leg`（6 条判据逐条 `passed`） |
| `reslot_seed10.json` / `reslot_seed11.json` / `reslot_seed12.json` | 逐 shard 运行报告（`touhou_rl.v420.reslot` 的 `--out-json`）：`counts`、`hash`、`teacher`（含 profile 全字段与 `capabilities`/`child_hazards` 探针原文）、`engine`、`data_dir`、`source_identity`（模块 sha256）、`self_check`（逐检查项） |

产物（`.tmp_v420_data`，不进 Git）：

| 路径 | sha256 | 字节 |
| --- | --- | ---: |
| `T02d/reslot/d1-s0d1-n3300-seed10-shard/teacher_2f.jsonl` | `0d81bc4d…97d1a6` | 10,167,501 |
| `T02d/reslot/d1-s0d1-n3300-seed11-shard/teacher_2f.jsonl` | `f23145c5…204a21` | 10,166,434 |
| `T02d/reslot/d1-s0d1-n3300-seed12-shard/teacher_2f.jsonl` | `e4e27845…2931e0fc` | 10,155,775 |
| `T02c/rebuild/…seed10-shard/2f/slots.jsonl`（**只读**） | `b215c00a…492548` | 5,984,901 |
| `T02c/rebuild/…seed11-shard/2f/slots.jsonl`（**只读**） | `ba163416…a640bba` | — |
| `T02c/rebuild/…seed12-shard/2f/slots.jsonl`（**只读**） | `ef5fb00f…c8bd7a69` | — |

**只读足迹**（运行后独立复核，`.tmp_v420_t02d/footprint.py`）：

- 旧语料 `labels.jsonl`：seed 10/11/12 = `011f6c1c…` / `bc663759…` / `f79f8ab6…`，
  mtime 全为 `2026-11-13 03:33`（与 `shard.json.labels.sha256` 一致，未被触碰）；
- 源数据 `/root/v400-data/s1c/*/env/score.dat`：`d7089834…` / `832f1813…` / `165fbd25…`，
  mtime 全为 `2026-11-13 03:31`；
- T02c 的 `2f/slots.jsonl` 在本任务运行后重新哈希，与 join 时记录的值**仍然相同**
  （`b215c00a…` / `ba163416…` / `ef5fb00f…`，mtime 17:50 / 17:54 / 17:58），
  证明审计读到的是 T02c 的稳定产物，本任务既未改也未在它写入过程中读半成品。

**复算**：

```bash
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v420_t02d/run_join.sh   # 重算 reslot_2f.json
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v420_t02d/pytest.sh     # 16 passed
```

## 10 变更文件（白名单内）

| 文件 | 状态 |
| --- | --- |
| `touhou_rl/v420/reslot.py` | 新建（1,880 行），唯一实现：`reslot_legacy_shard(config)` + `main()` |
| `tests/test_v420_reslot.py` | 新建（585 行，16 个测试） |
| `Document/Update/Alpha/v4.2.0/evidence/T02d/**` | 新建：本文件 + 4 个 JSON |
| `.tmp_v420_data/T02d/**` | 新建：标签、meta、data 副本、logs、pytest basetemp |
| `.tmp_v420_t02d/**` | 新建：运行脚本与复算脚本 |

未修改：`touhou_rl/v420/{contracts,replay_audit,eval,counterfactual,rebuild,__init__}.py`、
`tools/rl/v420.py`、`configs/v420/**`、任何既有测试、`touhou_rl/v400/**`、
`touhou_rl/v410/**`、`tools/rl/**`、旧证据、`ASSET_SYNC.*`、T02c 的任何产物。
