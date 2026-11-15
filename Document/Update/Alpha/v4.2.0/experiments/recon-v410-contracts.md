# v4.1.0 契约侦察备忘录（只读）

- 状态：**只读侦察完成**；本文件不含任何实现、未运行引擎、未启动训练/采集、未执行 git 写操作。
- 适用对象：v4.2.0 T01（契约/入口）与 T02（历史复现 + 两帧契约核对）。
- 侦察样本（唯一数据集实例）：`.tmp_v410_s1c/d1/shards/d1-s0d1-n3300-seed10-shard/`（stage 0, seed 10, mode `fast`, 3,300 decisions, 13,200 frames, `stop_reason=max_frames`, `stage_clear=false`, `deaths_observed=0`）。
- 配套校准标签（只读）：`.tmp_v410_s1c/relabel/d1-s0d1-n3300-seed10-shard/labels_v410.jsonl`；WP03 全量审计：`Document/Update/Alpha/v4.1.0/evidence/WP03/s1c_full_audit.json`（`passed=true`, 20 shards, 66,143 decisions）。
- 本次读过的代码（精确路径，未对仓库根做递归扫描）：
  - `tools/rl/v400_d1_collect.py`（1,857 行）
  - `tools/rl/v400_teacher.py`、`tools/rl/v400_d0_dual.py`、`tools/rl/v400_student_eval.py`
  - `touhou_rl/v400/student/obs_spec.py`、`obs_spec_v2.py`、`obs_spec_v3.py`、`obs_v3_dataset.py`
  - `touhou_rl/v400/teacher/config.py`、`search.py`、`decide.py`、`obs.py`
  - `touhou_rl/v400/gym/{actions,shm}.py`
  - `touhou_rl/v410/student/s1_data.py`、`touhou_rl/v410/student/bc.py`、`touhou_rl/v410/dagger/train.py`
  - `tools/rl/v410_student_eval.py`
- 只读复现脚本（本次新建，位于允许目录 `.tmp_v420_recon/`）：
  - `inspect_shard.py`：`shard.json` / `labels.jsonl` 前 3 行与前 8 个 decision、三个 `npy` 的形状与栈重叠检验；
  - `decode_slots.py`：逐行解码 `step_frames`→`mask_to_cand`、`act_hist` 与 `first_action` 对账、窗口算术与 `duration/slots` 分布。

> **路径更正（重要，影响 T01 的 import 清单）**：任务书给出的 `touhou_rl/v410/s1_data.py`、`touhou_rl/v410/bc.py`、`touhou_rl/v410/train.py` **不存在**。真实位置是
> `touhou_rl/v410/student/s1_data.py`、`touhou_rl/v410/student/bc.py`、`touhou_rl/v410/dagger/train.py`（另有 `touhou_rl/v400/student/probe/train.py`）。下文一律使用真实路径。

---

## Q1 一个 4 帧采集块如何映射到 2 个 2 帧决策槽？

**结论一句话：** 采集器每个 label 行推进 **4** 个逻辑帧（= 2 个 2 帧槽），`decision_frames=2` 描述的是**该行标签动作的作用长度（第一个槽）**，不是行间步长；因此 `decision_count=3300 × frames_per_decision=4 = frames_committed=13200`，`last_decision_frame=13196=(3300−1)×4` 完全自洽。

### 代码定位

`tools/rl/v400_d1_collect.py:118-119`
```python
#: 4 logic frames per teacher decision (``replan_decisions 2 x decision_frames 2``).
FRAMES_PER_DECISION = 4
```

`tools/rl/v400_d1_collect.py:962-1003`（每个 label 行一个 4 帧块；决策帧步长恒为 4）
```python
last = len(labels) - 1
pending: dict[str, Any] | None = None
for index, row in enumerate(labels):
    absolute = start_frame + index * FRAMES_PER_DECISION
    relative = index * FRAMES_PER_DECISION
    ...
    sample = buffer.stack_for(absolute, episode_start=(start_mode == "shard" and index == 0))
    ...
    for offset in range(FRAMES_PER_DECISION):
        frame = absolute + offset
        render_frame, current_hash = _step_exact(
            env, int(row["step_frames"][offset]), frame,
            f"{key}: decision {index} frame {offset}")
```

`tools/rl/v400_d1_collect.py:1062-1063`（落盘口径：`frames = decisions × 4`）
```python
writer.meta["decisions"] = int(written)
writer.meta["frames"] = int(written * FRAMES_PER_DECISION)
```

4 帧块本身由 teacher CLI 的“2 个槽 × 2 帧”拼出：`tools/rl/v400_teacher.py:473-484`
```python
actions = list(chosen.slots[: config.replan_decisions])
commits = actions + commits[len(actions):]
block = commit_block(
    actions, bomb=bomb, decision_frames=config.decision_frames,
    release_shoot=release)
```
`tools/rl/v400_teacher.py:539-541`（`step_frames` = 逐帧展开的 4 个 mask）
```python
payload["step_frames"] = [
    int(mask) for mask, count in block for _ in range(int(count))
]
```
`touhou_rl/v400/teacher/config.py:199-200`：`replan_decisions: int = 2`、`decision_frames: int = DECISION_FRAMES`；`touhou_rl/v400/student/obs_spec.py:37`：`DECISION_FRAMES: Final = 2`。
`touhou_rl/v400/student/obs_spec_v3.py:66-68`（同一句话的规范表述）：
```python
#: A student decision is 2 logic frames (``obs_spec.DECISION_FRAMES``); a D0 CLI
#: decision commits ``2 x 2 = 4`` logic frames and labels only its first slot.
DECISION_FRAMES: Final = obs_spec.DECISION_FRAMES
```
`touhou_rl/v400/teacher/search.py:59-66`（一个 `slots[i]` 展开为 2 个相同 mask，故 4 帧 = 2 槽）：
```python
@property
def masks(self) -> tuple[int, ...]:
    from touhou_rl.v400.gym.actions import cand_to_mask
    out: list[int] = []
    for cand in self.slots:
        out.extend([cand_to_mask(cand, 0)] * 2)
    return tuple(out)
```

### 数据实例（shard.json）
| 字段 | 实测值 | 说明 |
|---|---|---|
| `frames_per_decision` | `4` | 行间帧步长 |
| `decision_frames` | `2` | 单槽=标签动作长度 |
| `decision_count` / `rows` / `decisions` | `3300` | 行数 |
| `frames` / `frames_committed` / `stage_frames[1]` | `13200` | `3300×4` |
| `first_decision_frame` | `0` | |
| `last_decision_frame` | `13196` | `(3300−1)×4` |
| `hash_checkpoint_pairs` | 6600 条、`checkpoints_equal=6600` | 每行 2 个检查点；**唯一帧号 3301 个、最大帧 13200** |
| 最后一行的 `hash_post` 帧 | `13200` | 与 `frames_committed` 一致 |

`shard.json.hash_checkpoint_pairs` 前 3 条与末条实测：
```json
{"decision":0,"boundary":"pre","frame":0,...}
{"decision":0,"boundary":"post","frame":4,...}
{"decision":1,"boundary":"pre","frame":4,...}
{"decision":3299,"boundary":"post","frame":13200,"shadow":"8d10c356f304ad3c","env":"8d10c356f304ad3c","equal":true}
```

### 对 v4.2.0 的直接影响
- `frames_per_decision=4` 与 `decision_frames=2` **不是矛盾**，是“行步长 vs 标签动作长度”两个量；T01 的 `SlotRecord.decision_frame` 必须用 `index*4` 生成，`commit_frames` 用 `(d, d+1)`（槽0）/`(d+2, d+3)`（槽1）。
- 槽总数 = `decision_count*2 = 6600`，而 `frames_committed = 13200`；T02 报告“槽位覆盖率”时，分母用 6600、分子用 3300（见 Q3）。
- `cap` 判定（T02 的 `frame_cap`）在旧数据里等价于 `frames_committed=13200` 且 `stop_reason="max_frames"`；本样本无死亡、无换面，**不能**用来验证 `first_death`/`game_over`/`stage_clear` 分支。

---

## Q2 `chosen_plan` 的 `first_action` / `duration` / `slots`(16) 各是什么语义？

**结论一句话：** `first_action`=槽 0 的候选动作（唯一的学习目标，`scores[18]`/`verified_mask` 只对它）；`duration`=**规划根节点**把首动作保持多少个 2 帧槽（搜索超参，取值 1..4，本 shard 只出现 1/2），与实际执行几个槽无关；`slots` 是**未来** 16 个 2 帧槽（32 帧搜索视野），槽 0 = `first_action`，槽 1 起是规划，**已执行但无独立标签**。

### 代码定位

`touhou_rl/v400/teacher/search.py:27-33`（三个字段的定义）
```python
@dataclass(frozen=True)
class Plan:
    """One planned continuation: the first action, how long it is held, the rest."""
    first_action: int
    duration: int
    slots: tuple[int, ...]
```
`touhou_rl/v400/teacher/config.py:595-613`（`duration` 是“根节点保持长度”，提交长度恒为 2 槽）
```python
#: ``durations = (1, 2)``  ->  ``(1, 2, 3, 4)``
...
#: Nothing about the *executed* plan changes shape: ``decide``/the CLI commit
#: ``replan_decisions = 2`` slots either way (11-P1 §2 replans every R slots), so
#: durations 2/3/4 commit the same two masks and differ only in how far the plan
#: assumed the hold would carry -- which is the information the score needs.
_V2J_DURATIONS = (1, 2, 3, 4)
```
`touhou_rl/v400/teacher/config.py:199-203`：`replan_decisions=2`、`durations=(1,2)`（本 shard `mode="fast"`，`touhou_rl/v400/teacher/config.py:277` `FAST = TeacherConfig(mode="fast", horizon=32, ...)`）。
`slots` 长度 = `TeacherConfig.slots`（`touhou_rl/v400/teacher/config.py:251-252`：`return self.horizon // self.decision_frames`）= `32 // 2 = 16`，并受 `limit = min(slots, len(tape)//2)`（`touhou_rl/v400/teacher/search.py:1093-1094`）截断——本 shard 3,300 行实测**全部为 16**。
`touhou_rl/v400/teacher/search.py:1049-1057`（`slots` 来自束搜索的整条未来动作序列；槽 0 被写成 `first`）
```python
plan = Plan(
    first_action=first,
    duration=duration,
    slots=tuple(int(x) for x in slots[chosen]),
```
`tools/rl/v400_teacher.py:473`：提交时只取前 `replan_decisions` 个槽
```python
actions = list(chosen.slots[: config.replan_decisions])
```

### 数据实例（3,300 行实测）
| 量 | 实测 |
|---|---|
| `chosen_plan.slots` 长度 | 全部 `16` |
| `duration` 直方图 | `{1: 2712, 2: 588}`（无 3/4） |
| `first_action` 与 `step_frames[0]` | `step_frames[0] == cand_to_mask(first_action, 0)` **3300/3300** |
| `slots[1]` 与 `step_frames[2]` | `step_frames[2] == cand_to_mask(slots[1], 0)` **3300/3300** |
| `slots[1] != first_action` | **1740/3300 = 52.7%** |
| `step_frames[0] == step_frames[2]` | 1560/3300 |
| `bomb` 非零 | 0 行；`verified_count` 全为 1 |

第 0 行（`inspect_shard.py` 输出）：
```text
d=0 step_frames=[65,65,65,65]  first_action=7  duration=2
    slots=[7,7,7,0,0,0,0,0,0,0,0,0,0,0,0,0]
```
末行（`decode_slots.py` 解码）：
```text
decision=3299 decision_frame=13196 duration=1 first_action=2 slots[:4]=[2,16,0,0]
step_frames=[145,145,69,69]
mask_to_cand(145) -> cand 2, bomb 0
mask_to_cand(69)  -> cand 16, bomb 0
```
> **“`duration=2` 时实际执行了几个 2 帧动作？”→ 恒为 2 个槽（4 帧），与 `duration` 无关。**
> 反证实例：末行 `duration=1`，却仍然执行了槽 0（cand 2）与槽 1（cand 16）两个 2 帧动作。

### 对 v4.2.0 的直接影响
- `SlotRecord.cand` 的槽 0 可直接用 `first_action`（3300/3300 与实际 mask 吻合）；槽 1 **不能**从 `duration` 推，只能从 `slots[1]` 或 `step_frames[2]` 解。
- 任何“`duration` = 决策覆盖帧数”的理解都会把 4 帧块误算成 1/2 个槽；T01 单测应显式钉住 `duration ∈ {1,2}` 时槽数仍为 2。
- `slots[2:]`（本样本多为 0）是**未执行的规划**，不得进 act_hist、不得当标签、不得当“已执行动作”。

---

## Q3 原始标签是否只覆盖第一个槽？第二槽是否有独立标签？

**结论一句话：** 是，只有槽 0 有标签；4 帧块里的第二个 2 帧决策（帧 `[d+2, d+4)`）**没有任何独立标签**——`shard.json` 自己声明了 `label_is_first_slot_not_continuation=true`，行 schema 里也不存在第二槽的 `scores`/`verified_mask`/`margin`。

### 代码定位

`shard.json → alignment`（来源是**冻结模块常量** `touhou_rl/v400/teacher/obs.py:89-113` 的 `ALIGNMENT` 字典，由 `TeacherObsShardWriter` 原样写入 `shard.json`，见同文件 `1109`/`1183`；采集器只负责把它落盘）：
```json
"observation_time": "decision_time_before_commit",
"label_time": "scores[18] ranks the plan's first student decision (slot 0),
               applied to frames [decision_frame, decision_frame + decision_frames)",
"label_is_first_slot_not_continuation": true,
"decision_frames": 2,
"action_history_excludes_current_action": true
```
`touhou_rl/v400/teacher/obs.py:351-359`（标签作用帧只有 `decision_frames=2` 个）
```python
return {
    "decision_frame": frame,
    "render_frames": frames,
    "newest_render_frame": newest,
    "p3_window_last_row": p3_i,
    "p3_label_frame": p3_label,
    "label_frame": frame,
    "label_frames": [frame + offset for offset in range(obs_spec.DECISION_FRAMES)],
}
```
检查点却是 4 帧后才对齐（`tools/rl/v400_d1_collect.py:1002-1003`），这正是“4 帧块只有 2 帧带标签”的实现形态：
```python
pairs.append(_pair(index, "post", absolute + FRAMES_PER_DECISION,
                   str(row["hash_post"]), current_hash))
```

### 数据实例
1. 行 schema 唯一（3,300 行只有 1 种 key 集合，26 个 key），**没有** `slots`/`second_action`/`scores_slot1` 之类字段：
```text
['bomb','bomb_death_available','channel','chosen_plan','decision','decision_frame',
 'decision_frames','difficulty','frame','frame_event_flags','frames_per_decision',
 'hash_post','hash_pre','kind','margin','obs','pixel_channel','schema','scores',
 'shoot_release','stage','start_frame','start_mode','step_frames','verified_count',
 'verified_mask']
```
2. 末行 `step_frames=[145,145,69,69]`：槽 0 = cand 2（= `first_action`，有 `scores`/`verified_mask`），槽 1 = cand 16（**只存在于 `step_frames[2]`/`slots[1]`，无任何标签字段**）。
3. 52.7% 的行（1740/3300）两个槽动作不同，说明“第二槽 = 第一槽”不能当近似。
4. 校准标签 `labels_v410.jsonl` 同样只保留一份 `chosen_plan` + `verified[]`（`plan_count=1`），`verified[0].decision_frames=4`、`horizon=32`，即“验证计划以 4 帧为提交块”，不提供第二槽的独立评分。

### 对 v4.2.0 的直接影响
- 旧数据对 2 帧契约的**标签覆盖率 = 3300/6600 = 50%**；T02 必须把这个数字写成分母/分子，不能用“3300 行 ≈ 3300 个 2 帧决策”掩盖。
- `SlotRecord.label_source` 只有两种取值可用：槽 0 = `teacher_slot0`；槽 1 = **`none`**（`label_verified=false`）。任何把槽 0 标签复制给槽 1 的做法都属于“凭空补标签”，禁止。
- 槽 1 的**已执行动作**是可恢复的（`step_frames[2]`→`mask_to_cand`，3300/3300 与 `slots[1]` 一致），但那是“行为数据”不是“教师标签”，必须分开登记（见 §9 补救路径）。

---

## Q4 `act_hist` 的真实来源：教师建议动作还是实际执行动作？

**结论一句话：** 数据集 `act_hist` 由 `chosen_plan.first_action` 重建（教师提交动作），在该 teacher-driven shard 上它**确实等于实际执行**的槽 0 mask（3300/3300），但它是**每 4 帧才取一个槽**的 8 个历史（跨度 32 帧），因此 v4.2.0 要求的“过去 8 个已实际执行的 2 帧动作”（跨度 16 帧、包含槽 1）**未被满足**；evaluator 侧则完全独立，用策略自己的 `ActionHistoryBuffer`。

### 代码定位（数据集侧）

`tools/rl/v400_d0_dual.py:903-944`
```python
def v3_action_history(labels: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    """Rebuild the obs-v3 ``act_hist`` (N,8,18) from the aligned label rows.
    The reconstruction is obs-v2's frozen rule: a D0/teacher-driven row executes
    ``chosen_plan.first_action`` at its decision, so ``build_action_history`` can
    rebuild ``a[t-8]..a[t-1]``.
    """
    cand, stage = obs_spec_v2.reconstruct_from_label_rows(labels)
    history = obs_spec_v2.build_action_history(cand, stage)
    ...
    info = {... "source": ("chosen_plan.first_action (teacher-driven dual shard; the action "
                           "the teacher committed at that decision)"), ...}
```
`touhou_rl/v400/student/obs_spec_v2.py:282-290`（决定“执行动作”的唯一分支）
```python
else:
    plan = row.get("chosen_plan") or {}
    if "first_action" not in plan:
        raise ObsSpecV2Error(...)
    cand[index] = int(plan["first_action"])
```
`touhou_rl/v400/student/obs_spec_v2.py:247-254`（行 t = 决策 t−8..t−1；一“决策”=一次 4 帧提交）
```python
for t in range(count):
    if boundaries[t]:
        segment_start = t
    first = max(segment_start, t - ACTION_HISTORY_DECISIONS)
    for offset, source in enumerate(range(t - 1, first - 1, -1)):
        history[t, ACTION_HISTORY_DECISIONS - 1 - offset] = action_one_hot(int(cand[source]))
```
采集器落盘：`tools/rl/v400_d1_collect.py:880-882`（强制行数=决策数）与 `1124-1147`（把 `act_hist` 写进 sample）。

### 代码定位（evaluator 侧，与数据集无关）

`tools/rl/v410_student_eval.py:947-948`
```python
buffer = obs_spec_v3.ObservationBufferV3()
history = ActionHistoryBuffer()
```
`tools/rl/v410_student_eval.py:953-963`（喂给策略的是**自己的**历史）
```python
while frames < max_frames:
    play_u8 = buffer.stack()
    hud_u8 = buffer.hud()
    hist_u8 = history.rows()
    ...
    cand, bomb_logit = policy.act_with_bomb(
        obs_spec_v3.normalize_play(play_u8),
        obs_spec_v3.normalize_hud(hud_u8),
        obs_spec_v3.normalize_action_history(hist_u8).reshape(
            obs_spec_v3.ACTION_HISTORY_SHAPE))
```
`tools/rl/v410_student_eval.py:1008-1010`（**先执行后入历史**，每决策 1 条 = 1 个 2 帧动作）
```python
# The history at decision t must hold a[0..t-1]; push only after the
# action has been applied, so the current decision can never leak in.
history.push(cand)
```
身份声明：`tools/rl/v410_student_eval.py:889-891` `"act_hist 8x18 one-hot of the policy's own last 8 decisions"`。

### 数据实例
| 检查 | 结果 |
|---|---|
| `act_hist[t][-1] == labels[t-1].first_action`（t=1..3299） | **3299/3299**，0 失配 |
| `act_hist[t][-2] == labels[t-2].first_action`（t=2..3299） | 3298/3298 |
| `act_hist[0]` | 8 行全零（`zero_rows_at_start=8`，`first_unpadded_row=8`） |
| `step_frames[0] == cand_to_mask(first_action,0)` | **3300/3300** ⇒ 教师建议动作 == 实际执行的槽 0 |
| `act_hist` 是否含槽 1 动作 | **否**（槽 1 的 cand 16 之类从不入历史） |
| 时间跨度 | 数据集 8 决策 × 4 帧 = **32 帧**；部署 8 决策 × 2 帧 = **16 帧** |
| 执行顺序（t=8 起） | `[7,7,7,7,7,7,7,7]`（首 8 决策的 `first_action` 全为 7） |

### 对 v4.2.0 的直接影响
- 语义判定：**“教师建议”与“实际执行”在该 shard 上重合（槽 0），因此 `act_hist` 不是伪造的**；但它是“每 4 帧抽 1 个槽”的历史，**不满足** “过去 8 个已实际执行的 2 帧动作”。
- T01 的 `check_history_excludes_current` 之外，还必须有一条 `history_stride_frames == DECISION_FRAMES(2)` 的检查：旧数据该值为 4，必须判 fail（可解释 fail，不是未知差异）。
- evaluator 侧 act_hist 是“自己的历史”，语义正确；**T02 的“模型输入与前向比较”若直接对比数据集 act_hist 与在线 act_hist，差异是预期的、可解释的**（32 帧 vs 16 帧 + 缺槽 1），不能当模型 bug。

---

## Q5 obs 的 4 帧堆叠：连续逻辑帧还是隔帧？相邻 decision 的窗口是否重叠？

**结论一句话：** 4 帧是**严格连续的 4 个逻辑帧** `[d−4, d−1]`（oldest-first），在旧数据里相邻 decision 的窗口**完全不重叠**（步长 4）；但在 2 帧决策节拍下窗口**必然重叠 2 帧**——这正是旧栈采样与部署栈采样的结构性差异。

### 代码定位

`touhou_rl/v400/teacher/obs.py:328-359`（窗口定义，并自证 `frames[i-3..i] -> mask[i+1]`）
```python
def decision_window(decision_frame: int, *, stack: int = obs_spec.FRAME_STACK) -> dict[str, Any]:
    ...
    newest = frame - 1
    frames = [newest - stack + 1 + offset for offset in range(stack)]
    ...
    return {"decision_frame": frame, "render_frames": frames, "newest_render_frame": newest, ...}
```
`touhou_rl/v400/teacher/obs.py:570-583`（拒绝任何非 `d-4..d-1` 的缓冲）
```python
if len(self.frames) != self.stack:
    raise ObservationError(...)
actual = [int(row.frame) for row in self.frames]
if actual != expected:
    raise ObservationError(
        f"decision frame {decision_frame}: buffered renders {actual} != "
        f"required window {expected}")
```
`tools/rl/v400_d1_collect.py:987-1001`（**逐帧** STEP + 逐帧读像素，所以 4 个 render 是连续帧）
```python
for offset in range(FRAMES_PER_DECISION):
    frame = absolute + offset
    render_frame, current_hash = _step_exact(
        env, int(row["step_frames"][offset]), frame, ...)
    ...
    play, hud, _header, stats = _read_pixels(env, f"render {render_frame}")
    ...
    buffer.push(RenderFrame(frame=render_frame, play=play, hud=hud))
```
栈顺序冻结：`touhou_rl/v400/student/obs_spec.py:40-43` `FRAME_STACK=4`、`STACK_ORDER="oldest-first t-3,t-2,t-1,t"`；`touhou_rl/v400/student/obs_spec_v3.py:69-71` 继承之。

### 数据实例（逐行窗口 + 像素比对）
```text
d=1: render_frames=[0,1,2,3]      pad=0
d=2: render_frames=[4,5,6,7]      pad=0
d=3: render_frames=[8,9,10,11]    pad=0
d=4: render_frames=[12,13,14,15]  pad=0
d=0: render_frames=[0,0,0,0]      pad=3   ← 只有 decision 0 有 startup padding
```
- `obs.render_frames == [d-4, d-3, d-2, d-1]`：**3299/3299**（所有非 padding 行，0 例外）。
- 相邻 decision 栈逐行比对（`play96x112.npy`，`np.array_equal`）：decision 0↔1、1↔2、2↔3、3↔4 **没有任何一对相同行**，即**窗口不重叠**（步长 4）。
- 同一栈内 4 行两两不同（decision 1：对角为 True、其余 False），排除“4 帧重复填充”。
- `hud[0] != hud[1]`：HUD 取窗口最新帧（`t = d−1`），逐 decision 变化。

### 对 v4.2.0 的直接影响
- 旧数据：窗口步长 = 4 帧、无重叠；2 帧部署：窗口步长 = 2 帧、**相邻窗口共享 2 帧**。`SlotRecord.obs_render_frames` 必须是 `d-4..d-1`，在 2 帧节拍下 `d` 与 `d+2` 的窗口有 2 帧交集——T01 的 `check_contiguous_render_frames` 要允许交集（`d-2..d+1`），不能写成“必须不重叠”。
- evaluator 的 `obs_window="consecutive"` 只保证“一帧一 render”，**不保证窗口不重叠**；`tools/rl/v410_student_eval.py:772-784` 的 `matches_training_obs=True` 仅就“连续性”成立，T02 引用这句时必须补一句步长差异，否则会误称“与训练观测一致”。

---

## Q6 evaluator 的决策/终止契约

**结论一句话：** 一决策 = 2 帧（`--max-frames 19887` 是帧上限、奇数时最后一决策只剩 1 帧）；终止由“引擎 `exit`+`game_over`”判 `game_over`、由 `STATE.stage` 增大判 `stage_clear`、由首死 + `--stop-on-death` 判 `death_stop`，三者互斥；每局记录里 `stop_reason`/`exit`/`deaths`/`lives_end`/`stage_clear` 足以分离首死与 `game_over`，但**没有 `first_death_frame` 字段**。

### 代码定位

`tools/rl/v410_student_eval.py:953-1016`（主循环与三种终止）
```python
while frames < max_frames:
    ...
    budget = min(obs_spec_v3.DECISION_FRAMES, max_frames - frames)
    units, want = decision_steps(obs_window, budget)
    steps += 1
    ended: tuple[str, str] | None = None
    for _unit in range(units):
        step_reply = client.step(mask, want)
        advanced = int(step_reply.get("advanced", 0))
        frames += advanced
        ...
        if step_reply.get("exit") == "1":
            if step_reply.get("game_over") == "1":
                ended = ("game_over", "game_over")
            else:
                ended = ("engine_exit", "engine_exit")
            break
        state = state_from_reply(client.request("STATE"))
        current = _readings(state)
        if int(state["stage"]) > stage_start:
            ended = ("stage_clear", "controller")
            break
        if current["deaths"] > previous["deaths"]:
            if current["lives"] <= 0:
                ended = ("game_over", "game_over")
                break
            if stop_on_death:
                ended = ("death_stop", "controller")
                break
        previous = current
        play, hud = obs_spec_v3.encode_planes_v3(shm.pixels())
        buffer.push(play, hud)
        renders += 1
    history.push(cand)          # 1008-1010: 先执行后入历史
    if ended is not None:
        stop_reason, exit_reason = ended
        break
else:
    stop_reason = "frame_cap"
    exit_reason = "controller"
```
`tools/rl/v400_student_eval.py:191-213`（`consecutive` 模式：2 次单帧 STEP、2 个 render）
```python
if obs_window == "consecutive":
    return int(budget), 1
return 1, int(budget)
```
`tools/rl/v410_student_eval.py:1894-1897`（`--max-frames` / `--stop-on-death` 的来源）
```python
parser.add_argument("--max-frames", type=int, default=19887,
                    help="19887 = human_reference cap for S1N")
parser.add_argument("--stop-on-death", action="store_true",
                    help="end the episode on the first death (G1's rule)")
```
`tools/rl/v400_student_eval.py:13-18`：`19887 = ceil(1.5 × 13258 measured human Stage1 frames)`，0-based stage 语义。

终止词表与记录字段：`tools/rl/v410_student_eval.py:184-189`
```python
VALID_STOP_REASONS = frozenset({
    "game_over", "death_stop", "stage_clear", "frame_cap", "engine_exit",
    "reset_failed", "error",
})
VALID_EXITS = frozenset({"game_over", "controller", "engine_exit", "none"})
```
`tools/rl/v410_student_eval.py:1049-1088`（落到每局记录的字段）
```python
record.update(
    frames=int(frames), steps=int(steps), renders=int(renders),
    pixels_warmup_frames=int(pixels_warmup_frames),
    deaths=float(finish["deaths"]), lives_end=float(finish["lives"]),
    bombs_start=..., bombs_end=..., bombs_used=...,
    score=int(finish["score"]), score_start=...,
    state_frame=int(end_state["frame"]), gameplay_frame=int(end_state["gameplay_frame"]),
    stage_start=int(stage_start), stage_end=int(stage_end),
    stage_clear=bool(stage_end > stage_start),
    stop_reason=stop_reason, exit=exit_reason,
    actions={...}, action_sequence_sha256=..., policy_input_sha256=...,
)
```
必填字段与交叉校验：`tools/rl/v410_student_eval.py:1096-1142`（`REQUIRED_EPISODE_FIELDS`，含 `stop_on_death`/`decision_frames`/`stage_clear`/`stop_reason`/`exit`）与 `1230-1260`
```python
if int(record["decision_frames"]) != int(obs_spec_v3.DECISION_FRAMES):
    raise ...
if record["frames"] > record["max_frames"]:
    raise ...
if record["stage_clear"] != (record["stage_end"] > record["stage_start"]):
    raise ...
```

### 分离“首死 vs game_over”的可用字段
| 字段 | 用途 | 限制 |
|---|---|---|
| `stop_reason` | `death_stop`=首死即停；`game_over`=死亡或引擎退出；`stage_clear`=换面；`frame_cap`=打满 | 需要 `--stop-on-death` 才会出现 `death_stop` |
| `exit` | `controller`（我方停）/`game_over`（引擎判）/`engine_exit`/`none` | 与 `stop_reason` 联读 |
| `deaths` | 累计死亡数（`STATE.deaths`） | **无逐次死亡帧号** |
| `lives_end` | 残机数，`<=0` 时判 `game_over` | |
| `stage_clear` / `stage_start` / `stage_end` | 换面与 `game_over` 天然互斥 | `frame_cap` 不算通关 |
| `frames` / `steps` / `renders` / `pixels_warmup_frames` | 帧分母与首死帧（仅 `death_stop` 时 `frames` 即首死帧） | `frames` 含 warmup |
| `error` / `stop_reason∈{reset_failed,error}` | 技术无效局，必须留在分母 | `FAILED_STOP_REASONS`（`tools/rl/v410_student_eval.py:188`） |

### 数据实例与缺口
- 本 shard `shard.json`：`stop_reason="max_frames"`、`stage_clear=false`、`deaths_observed=0`、`report.channel_a={"stop_reason":"max_frames","stage_clear":false,"deaths_observed":0}` ⇒ 该样本**不含**任何死亡/换面实例，Q6 的结论来自代码而非该 shard。
- `--max-frames 19887` 为奇数：`budget = min(2, 19887 - frames)` 使最后一决策只推进 1 帧（`decision_steps("consecutive", 1) → (1,1)`），因此“19887 帧”存在**半个决策**；T02 的 2,000 帧（偶数）不受影响，但对比旧 20k 报告时必须说明这一点。

### 对 v4.2.0 的直接影响
- `classify_termination` 可直接映射：`death_stop→first_death`、`game_over→game_over`、`stage_clear→stage_clear`、`frame_cap→cap`、`error/reset_failed→error`。
- **必须新记录** `first_death_frame`：旧 evaluator 不开 `--stop-on-death` 时无法回填该帧号（只能给 `deaths` 计数）。T01 的 `SlotRecord`/episode 记录需新增字段，T02 的在线审计必须在死亡当帧落盘。
- `stage_clear` 与 `game_over` 在同一次 `for _unit` 循环内按顺序判定并 `break`，二者不会同时成立；T01 的单测可直接构造 `STATE.stage+1` 与 `exit=1,game_over=1` 两个互斥用例。

---

## Q7 像素时间戳：现有 shard 是否记录每帧真实时间戳/渲染序号？

**结论一句话：** 没有。现有 shard 只记录**引擎逻辑帧号**（`FrameCount`，RESET 归零），既无墙钟时间戳、也无渲染序号/发布序号；逐帧只有帧号，且每帧 HASH 也未存（每 decision 只存 2 个边界 HASH）。

### 代码定位

`touhou_rl/v400/gym/shm.py:3-19`（共享内存头部字段全集 —— 就是全部可得信息）
```text
[0, 64)     header: magic u32, version u32, frame u64, state hash u64,
                    pixel width u32, pixel height u32, state bytes u32, flags u32
...
The engine is the only writer.  A reader takes the header's frame counter to tell
whether it is looking at a frame it has already consumed.
```
`touhou_rl/v400/gym/shm.py:80-94`（header 解包，只有 `frame` 这一个时间量）
```python
def header(self) -> dict[str, int]:
    magic, version, frame, state_hash, width, height, state_bytes, flags = (...)
    return {..., "frame": frame, ..., "flags": flags}
```
采集器只把帧号写进两层元数据：`tools/rl/v400_d1_collect.py:993-1001`（`_read_pixels` 返回 `stats`，不含时间）与 `_pair(...)`（`tools/rl/v400_d1_collect.py:752-755`，只有 `decision/boundary/frame/shadow/env`）。

### 数据实例（`shard.json` / `labels.jsonl` 实有字段）
| 类似“时间”的字段 | 实际内容 | 是否逐帧 |
|---|---|---|
| `labels.jsonl → obs.render_frames` | 4 个**逻辑帧号**（如 `[0,1,2,3]`） | 是（每决策 4 个帧号，非时间） |
| `labels.jsonl → obs.newest_render_frame` / `label_frames` | 逻辑帧号 | 是（推导量） |
| `shard.json → hash_checkpoint_pairs[].frame` | 每决策 2 个边界帧号（6600 条） | 否（每 4 帧 2 个） |
| `shard.json → created_at` | shard 级墙钟（`2026-11-13T03:31:22`） | 否（单值） |
| `shard.json → report.channel_b.pixel_ms_per_frame` | 聚合耗时（1.072 ms） | 否（均值） |
| `labels.jsonl → timing_ms` | **不存在**（原 26-key schema 无此字段） | — |
| 逐帧 HASH / 逐帧时间戳 / 渲染序号 | **缺失** | — |

`shard.json` 内不存在 `timestamp` / `rendered_at` / `seq` / `wall` 等键（实测键集合见 Q1 表格与 `inspect_shard.py` 输出）。

### 对 v4.2.0 的直接影响
- T02 的“像素时间”**不能从旧 shard 回填**，只能在新的在线审计里落地（`audit_live_episode` 必须自己记 `captured_at`/单调时钟 + 帧号 + 渲染序号）。
- 旧数据能给的最强对齐证据是“逻辑帧号 + 每决策 2 个 HASH（6600/6600 全等）”；T02 若声称“像素时间已核对”，只能指“帧号对齐”，不得暗示有时间戳。
- 逐帧 HASH 缺失 ⇒ 分支/重放审计若要逐帧验证，需要新记录（T02/T03 的 `SlotRecord` 必须带逐帧 HASH 或逐帧 mask+HASH 摘要）。

---

## Q8 现有 load/parse 入口：T01 复用 reader 与 evaluator 应调用哪些现成函数？

**结论一句话：** 读**原始旧 shard**用 `touhou_rl.v400.student.obs_v3_dataset.load_teacher_shard_v3`（自带 schema/digest/窗口/act_hist 校验，不需要 WP03 审计）；读**校准标签**用 `touhou_rl.v410.student.s1_data.{load_shards, read_label_rows}`（需要 `s1c_full_audit.json` + `labels_v410.jsonl`，**不能**解析原始 `labels.jsonl`）；评测直接以 importlib 方式调 `tools/rl/v410_student_eval.py` 的 `run_episode`。

### 可 import 的完整符号名（实测存在、签名可查）

**A. 原始旧 shard reader（推荐 T01 主路径）** — `touhou_rl/v400/student/obs_v3_dataset.py`
| 符号 | 位置 | 说明 |
|---|---|---|
| `obs_v3_dataset.load_teacher_shard_v3(shard_dir, *, verify_sha=True, verify_history=True)` | 91-207 | 返回 `TeacherShardV3`；拒绝 schema/digest/形状/sha/alignment/窗口算术/history 不符 |
| `obs_v3_dataset.TeacherShardV3` | 208-301 | `.play` `.hud` `.act_hist` `.records` `.meta` `.frames_per_decision`；`shard[i]` 给出 dict；`normalized(i)` 给出模型输入 |
| `obs_v3_dataset.load_teacher_samples_v3(paths, *, min_samples=..., obs_root=..., require_observations=...)` | 304+ | 批量 samples + `obs_unresolved` |
| `obs_v3_dataset.expected_action_history(records)` | 78-88 | 仅按 `first_action` 重建（= 旧口径，T01 不能拿它当 2 帧口径） |

**B. 校准标签 reader（需要审计 + relabel）** — `touhou_rl/v410/student/s1_data.py`
| 符号 | 位置 |
|---|---|
| `s1_data.load_shards(audit_path, *, repo_root, seeds=TRAIN_SEEDS, require_files=True, require_all=True)` | 128-217 |
| `s1_data.read_label_rows(ref, *, limit=None)` | 418-443 |
| `s1_data.parse_label_row(payload, *, shard, seed, row)` | 319-415 |
| `s1_data.ShardRef` / `s1_data.LabelSample` | 96-126 / 222-307 |
| `s1_data.ShardPlanes` / `s1_data.S1Dataset` | 449-487 / 488+ |
| `s1_data.pack_batch(play, hud, act_hist)` | 866-884 |
| `s1_data.PLAY_SHAPE/ACT_HIST_SHAPE/HUD_SHAPE/PLAY_FILE/HUD_FILE/ACT_HIST_FILE/TRAIN_SEEDS` | 51-70 |

**C. BC 数据装配（T01 若要复用 loader）** — `touhou_rl/v410/student/bc.py`
`s1_data` 之上的 `bc.make_loaders(audit_path, split, ...)`（866-899）、`bc.load_split_plan(...)`（578-589）、`bc.S1BCLoader`（650-775，`iter_batches` / `_gather` 走 `pack_batch`）、`bc.BCBatch`（623-647）、`bc.plan_split`（506-549）、`bc.DEFAULT_AUDIT`（107-108 = `Document/Update/Alpha/v4.1.0/evidence/WP03/s1c_full_audit.json`）。
`touhou_rl/v410/dagger/train.py:534,577,675-680,1253-1259` 复用了同一套 `s1_data.ACT_HIST_FILE` + `pack_batch`，可作“组装方式”参考，**不要**改它。

**D. obs 编码/缓冲** — `touhou_rl/v400/student/obs_spec_v3.py` 与 `obs_spec.py`、`obs_spec_v2.py`
`obs_spec_v3.{encode_planes_v3, ObservationBufferV3, normalize_play, normalize_hud, normalize_action_history, check_action_history? →见v2, check_obs_spec_v3, obs_spec_v3_digest, PLAY_SHAPE, HUD_SHAPE, ACTION_HISTORY_SHAPE, FRAME_STACK, DECISION_FRAMES, PLAY_BLOCK, PLAY_EXTENT_WH}`；
`obs_spec_v2.{build_action_history, reconstruct_from_label_rows, ActionHistoryBuffer, action_one_hot, check_action_history, history_digest, ACTION_HISTORY_DECISIONS}`；
`obs_spec.{DECISION_FRAMES, FRAME_STACK, STACK_ORDER, HUD_FRAMES, N_MOVE_CLASSES, ACTION_DURATION_FRAMES, TEACHER_DURATION_UNITS}`。

**E. 窗口算术（唯一实现）** — `touhou_rl/v400/teacher/obs.py`
`obs.decision_window(decision_frame, *, stack=4)`、`obs.DecisionObservationBuffer`（`reset/push/stack_for`）、`obs.TeacherObsShardWriter`、`obs.check_obs_version`、`obs.obs_version_spec`、`obs.RenderFrame`、`obs.require_rendered`、`obs.FROZEN_OBS_SPEC_V3_DIGEST`。

**F. 动作编码** — `touhou_rl/v400/gym/actions.py`
`actions.cand_to_mask(cand, bomb=0)`（58-75）、`actions.mask_to_cand(mask) -> (cand, bomb)`（78-98）、`actions.CAND_MASKS`、`actions.N_CAND`；`touhou_rl.spec.BTN_SHOOT/BTN_FOCUS/BTN_BOMB`。

**G. 评测器（复用，不重写）** — `tools/rl/v410_student_eval.py`（无包路径，需 `importlib.util.spec_from_file_location`；该文件自身 import torch）
`run_episode(*, engine, data_dir, checkpoint, policy, seed, stage, difficulty, max_frames, shm_name, stop_on_death=False, timeout=240.0, obs_window="consecutive")`（818-1090）、`ProbePolicy`（677-760）、`load_v410_checkpoint(path, *, device="cpu")`（485-653）、`validate_episode_record(...)`（1155+）、`summarise(...)`（1549-1668）、`parse_seeds`（257+）、`check_seed_policy`（231-254）、`OBS_SHAPES`（179-183）、`VALID_STOP_REASONS`（184-187）、`EnvHarnessClient`/`decision_steps`（由 `_V400` 转发，199-225）；
`tools/rl/v400_student_eval.py` 的 `decision_steps(obs_window, budget)`（191-213）与 `OBS_WINDOWS`。

**H. 状态解析 / shm** — `touhou_rl/v400/labels/state.state_from_reply`、`touhou_rl/v400/gym/shm.{SharedBlock, FLAG_PIXELS_VALID}`。

### 必须新写的适配点（旧代码没有）
1. **原始行 schema 适配器**：`parse_label_row` 要求 `raw_geo_score` + `query_status` + `verified`，且 `kind ∈ {None,"label"}`；原始 `labels.jsonl` 是 `kind="decision"`、只有 `scores/verified_mask/chosen_plan/step_frames`。**直接喂 `read_label_rows` 会 raise**。要么用 `load_teacher_shard_v3(...).records`（保留原始 dict），要么新写 `parse_legacy_decision_row`。
2. **1 行 → 2 个 `SlotRecord`** 的展开器（Q1/Q3）：槽 0 = `first_action` + 有标签；槽 1 = `slots[1]`/`step_frames[2]` 解码 + `label_source="none"`。
3. **2 帧口径的 act_hist 重建器**（Q4）：按 `step_frames` 每 2 帧解一次 `mask_to_cand`，`shoot_release` 帧要先把 SHOOT 位补回（见 §9）。
4. **终止分类器**（Q6）：需要新增 `first_death_frame`；旧记录没有该字段，只能 `stop_reason=="death_stop"` 时取 `frames`。
5. **像素时间戳**（Q7）：T02 在线审计新建字段；旧 shard 无源。
6. **legacy evaluator argv 构造器**（T01 `eval.py`）：不得 shell 拼接，必须含 `--max-frames 19887 --stop-on-death --stage 0 --difficulty normal --obs-window consecutive`。
7. **旧口径与 2 帧口径的显式分栏**：任何训练/评测输出必须标 `history_contract ∈ {legacy_4f, v420_2f}`，禁止混用（T01-contract-spec §1.1 已要求）。

### 阻塞与风险（实测发现）
| # | 阻塞/风险 | 证据 | 影响 |
|---|---|---|---|
| B1 | 任务书给的模块路径不存在 | `Test-Path touhou_rl/v410/{s1_data,bc,train}.py` 全部 False；真实路径在 `student/`、`dagger/` | T01 import 清单必须用真实路径，否则首行即 ImportError |
| B2 | `s1_data.read_label_rows` **不能**读原始 `labels.jsonl` | `parse_label_row` 319-415 需要 `raw_geo_score`；`read_label_rows` 433-436 拒绝 `kind="decision"` | T01 若计划“复用 reader 读旧 shard”，必须走 A 路径 |
| B3 | 旧 shard 无逐帧时间戳/渲染序号 | `shm.py:3-19` 头部仅 8 字段；`labels.jsonl` 无 `timing_ms` | T02 的“像素时间”腿只能新采集，不能回填 |
| B4 | 旧数据死亡/换面样本缺失（本 shard） | `stop_reason=max_frames`、`deaths_observed=0`、`stage_clear=false` | `first_death`/`game_over`/`stage_clear` 分类只有代码级证据，需 T02 在线局补真实实例 |
| B5 | `s1c_full_audit.json` 位于 **v4.1.0 证据目录**（只读） | `Document/Update/Alpha/v4.1.0/evidence/WP03/s1c_full_audit.json`，`passed=true` | T01 只读引用即可；不得写回、不得改 sha |
| B6 | 校准标签行 `frame` != `decision_frame` | `labels_v410.jsonl` 行 0：`frame=4`、`decision_frame=0`、`decision_frames=4` | `parse_label_row` 用 `payload["frame"]` 当 `LabelSample.frame`，会得到“提交后帧”；T01 若按帧号对齐须用 `decision_frame` |

---

## §9 T01/T02 建议接口

### 9.1 可复用符号（按调用顺序）

```text
# 入口/配置
touhou_rl.v420.contracts.*                                   # T01 新写（规格已存在）

# 读旧 shard（原始，含 step_frames）
touhou_rl.v400.student.obs_v3_dataset.load_teacher_shard_v3
touhou_rl.v400.student.obs_v3_dataset.TeacherShardV3
touhou_rl.v400.student.obs_v3_dataset.expected_action_history     # 只作对照，不作 2 帧口径

# 读校准标签（需要 WP03 审计；T01 audit-assets 用）
touhou_rl.v410.student.s1_data.load_shards
touhou_rl.v410.student.s1_data.read_label_rows
touhou_rl.v410.student.s1_data.parse_label_row
touhou_rl.v410.student.s1_data.ShardRef / LabelSample / ShardPlanes / S1Dataset
touhou_rl.v410.student.s1_data.pack_batch
touhou_rl.v410.student.bc.make_loaders / load_split_plan / S1BCLoader / BCBatch / DEFAULT_AUDIT

# obs 编码与窗口
touhou_rl.v400.student.obs_spec_v3.encode_planes_v3
touhou_rl.v400.student.obs_spec_v3.ObservationBufferV3
touhou_rl.v400.student.obs_spec_v3.normalize_play / normalize_hud / normalize_action_history
touhou_rl.v400.student.obs_spec_v3.check_obs_spec_v3 / obs_spec_v3_digest
touhou_rl.v400.student.obs_spec_v3.{PLAY_SHAPE,HUD_SHAPE,ACTION_HISTORY_SHAPE,FRAME_STACK,DECISION_FRAMES}
touhou_rl.v400.teacher.obs.decision_window / DecisionObservationBuffer / RenderFrame
touhou_rl.v400.teacher.obs.check_obs_version / obs_version_spec / FROZEN_OBS_SPEC_V3_DIGEST

# 历史重建（旧口径 = 只读对照；2 帧口径需新写，见 9.2-③）
touhou_rl.v400.student.obs_spec_v2.build_action_history
touhou_rl.v400.student.obs_spec_v2.reconstruct_from_label_rows
touhou_rl.v400.student.obs_spec_v2.ActionHistoryBuffer
touhou_rl.v400.student.obs_spec_v2.check_action_history / action_one_hot / history_digest

# 动作编码（槽 1 恢复的唯一合法路径）
touhou_rl.v400.gym.actions.cand_to_mask / mask_to_cand / CAND_MASKS
touhou_rl.spec.BTN_SHOOT / BTN_FOCUS / BTN_BOMB

# 评测（T02 真实引擎，importlib 载入，不改原文件）
tools.rl.v410_student_eval.run_episode
tools.rl.v410_student_eval.ProbePolicy / load_v410_checkpoint
tools.rl.v410_student_eval.validate_episode_record / summarise
tools.rl.v410_student_eval.parse_seeds / check_seed_policy / OBS_SHAPES / VALID_STOP_REASONS
tools.rl.v400_student_eval.decision_steps / OBS_WINDOWS
touhou_rl.v400.labels.state.state_from_reply
touhou_rl.v400.gym.shm.SharedBlock / FLAG_PIXELS_VALID
```

### 9.2 必须新写的适配点（T01 白名单内）

1. `parse_legacy_decision_row(row) -> LegacyRow`：接受原始 26-key 行（`kind="decision"`），抽出 `decision/decision_frame/frames_per_decision/decision_frames/step_frames/hash_pre/hash_post/chosen_plan{first_action,duration,slots}/scores/verified_mask/bomb/shoot_release/obs.render_frames`。
2. `expand_block_to_slots(row) -> tuple[SlotRecord, SlotRecord]`：槽 0（`label_source="teacher_slot0"`，`label_verified=bool(verified_mask[first_action])`）、槽 1（`label_source="none"`，`cand` 由 `mask_to_cand(step_frames[2])` 得，`mask` 原样保留）。
3. `build_history_2f(rows) -> np.ndarray[(N,8,18)]`：按**每 2 帧一个槽**重建 8 个“已实际执行”动作：
   - 逐行取 `a0 = mask_to_cand(step_frames[0])`、`a1 = mask_to_cand(step_frames[2] | BTN_SHOOT)`；
   - 交错序列 `[a0(0), a1(0), a0(1), a1(1), …]`，`history[t] = 前 8 条`，跨 shard/换面清零（沿用 `obs_spec_v2.build_action_history` 的清零语义，但输入换成 2 帧序列）；
   - `| BTN_SHOOT` 是幂等兜底：137 个 `shoot_release` 行**只**清 `step_frames[3]` 的 SHOOT 位（`step_frames[2]` 永远带 SHOOT），所以取 `step_frames[2]` 本身不会触发 `mask_to_cand` 的 “mask does not shoot” 异常；若将来改用槽 1 的尾帧 `step_frames[3]`，则必须先补 SHOOT 位。
4. `check_history_stride(history_frames) -> None`：断言相邻历史条目帧差 == 2；旧 `first_action` 口径为 4，必须判 fail 并计入 `unexplained_diffs=0` 的“可解释 fail”。
5. `classify_termination(stop_reason, exit_reason, deaths, lives_end, stage_clear, frames, stop_on_death)`：`death_stop→first_death`、`game_over→game_over`、`stage_clear→stage_clear`、`frame_cap→cap`、`error/reset_failed→error`；`first_death_frame = frames` 仅当 `stop_reason=="death_stop"`，否则 `None`（**明确 unknown，不猜**）。
6. `pixel_timestamp_record`（T02 在线）：`{frame, monotonic_ns, wall_utc, render_seq, hash}`，逐帧落盘；旧数据该字段整体为 `missing`。
7. `build_legacy_eval_argv(config)`：输出 `["python","tools/rl/v410_student_eval.py","--engine",...,"--max-frames","19887","--stop-on-death","--stage","0","--difficulty","normal","--obs-window","consecutive","--seeds","1000-1019"]`（不用 shell 拼接）。

### 9.3 若原 4 帧标签不覆盖第二槽：最小可行补救路径（**不得凭空补标签**）

**前提事实**：槽 1 有**已执行动作**（`step_frames[2]`，3300/3300 与 `slots[1]` 一致），但**没有** `scores[18]`、`verified_mask`、`margin`、`verified[]` 条目。

1. **不复制、不推导标签**：槽 1 的 `label_source` 固定 `"none"`、`label_verified=false`；`scores`/`verified_mask` 字段保持 `None`。禁止把槽 0 的 `scores` 平移、禁止用 `slots[1]` 反推“教师认可”。
2. **只把槽 1 当“行为数据”用**：
   - 进 `act_hist`（输入侧）——这是唯一不涉及标签语义的用法；
   - 作为可选的 executed-action 辅助目标，必须单独登记（`target_kind="executed_slot1"`）并且**不参与** `verified`/安全集合/教师一致性统计。
3. **槽 0 保持唯一监督目标**：`target = first_action`（3300 行），训练时按“2 帧节拍、50% 标签覆盖率”标注，而不是伪装成 6600 个有标签决策。
4. **显式缺口登记（按槽逐项，这是最容易写错的一栏）**：

   | 槽位 | 4 帧窗口来源 | play 像素 | HUD | 教师标签 | 已执行动作 |
   |---|---|---|---|---|---|
   | 槽 0（决策帧 `4i`） | `play[i]`（帧 `4i-4..4i-1`） | 有 | 有（帧 `4i-1`，`hud[i]`） | 有 | 有（`step_frames[0]`） |
   | 槽 1（决策帧 `4i+2`） | **无现成行**，但**可拼**：`concat(play[i][2:4], play[i+1][0:2])` = 帧 `4i-2..4i+1` | 可重建（不新增渲染） | **缺失**（`hud.npy` 每 4 帧只存 1 条 = 帧 `4i-1`，槽 1 需要帧 `4i+1`） | **无** | 有（`step_frames[2]`） |

   依据：`touhou_rl/v400/teacher/obs.py:580-583` 的 `stack_for` 返回 `"hud": self.frames[-1].hud.copy()`（HUD 只取窗口最新帧），而 `tools/rl/v400_d1_collect.py` 每个决策只 `writer.add` 一条 `hud`；`play` 的行 = 该决策的 4 个 `render_frames`（数据实测 `d=1 → [0,1,2,3]`、`d=2 → [4,5,6,7]`）。
   实测验证（`.tmp_v420_recon/check_slot1_window.py`）：`i=1` 时槽 1 决策帧 `4·1+2=6`，拼接窗口帧号 `[2,3,4,5]`，4 行两两不同（`any-duplicate=False`），两半分别取自 `play[1]` 与 `play[2]` 且互不相同；`i=2`（帧 `[6,7,8,9]`）、`i=5`（帧 `[18,19,20,21]`）同样成立 ⇒ **play 侧可无损拼出槽 1 的 4 帧窗口**。

   因此登记口径应为：`slots_total=6600`、`slots_with_own_label=3300`、`slots_action_only=3300`、`slots_play_reconstructible=3300`（槽 1）、`slots_hud_missing=3300`（槽 1）。
   **槽 1 = 有动作、无标签、play 可拼、HUD 缺**——所以槽 1 不得作为训练样本直接入模；若要 2 帧完整观测必须重采（T05），T01/T02 只登记 gap。
5. **要真正补第二槽标签**：只能重采（教师按 2 帧节拍查询），属 T05/T02 之后的新采集任务；T01/T02 阶段只登记为 gap，不得用近似替代。

---

## 附：本次核对的数字（可复现）

| 项 | 值 |
|---|---|
| `labels.jsonl` 行数 | 3300（文件 5,741,707 B；`shard.json.labels.sha256=011f6c…`） |
| `obs/play96x112.npy` | `(3300,4,112,96)` uint8，141,926,922 B |
| `obs/hud.npy` | `(3300,56,160)` uint8，29,568,522 B |
| `obs/act_hist.npy` | `(3300,8,18)` uint8，475,722 B |
| `bytes_per_decision` | 52,112 |
| HASH 检查点 | 6600/6600 equal，`first_mismatch=null` |
| 窗口 == `[d-4..d-1]` | 3299/3299（d=0 为 pad=3） |
| `step_frames[0]==cand_to_mask(first_action,0)` | 3300/3300 |
| `step_frames[2]==cand_to_mask(slots[1],0)` | 3300/3300 |
| `step_frames[3] != step_frames[2]` | 137（全部 `shoot_release=True`、全部恰为清 SHOOT 位） |
| `slots[1] != first_action` | 1740/3300（52.7%） |
| `act_hist[t][-1] == first_action[t-1]` | 3299/3299 |
| `duration` 直方图 | `{1: 2712, 2: 588}` |
| `bomb` 非零 | 0 |
