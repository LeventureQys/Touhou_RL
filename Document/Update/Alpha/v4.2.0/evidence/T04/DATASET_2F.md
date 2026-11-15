# T04 证据：2 帧语料 join（T02c × T02d）与可训练性验证

> 机器可读镜像：[`dataset_2f.json`](dataset_2f.json)（本文件每个数字都在那里；它由
> `.tmp_v420_t04a/write_evidence_json.py` 从实测产物生成，不手写）。
>
> 本文件只回答一件事：**是否存在一份「每 2 帧一槽、每槽有自己的 obs+HUD+已执行动作+
> 8 条真实历史+逐槽 verified 教师标签、HASH 0 差异」的语料，并且它能被真正训练。**
> 它**不**报任何学习结果：没有泛化、没有能力、没有门结论。

---

## 0 结论摘要

| 问题 | 回答 | 依据 |
|---|---|---|
| T02c 的 obs/action 与 T02d 的逐槽标签能否 join | **能**，按 `(shard_key, slot_id)` 且校验 `decision_frame`；19,772/19,772 全对上，孤儿 0、缺失 0、重复 0 | §2/§3 |
| join 后总槽数 | **19,772**（seed 10/11/12 = 6,592 / 6,592 / 6,588） | §3 表 1 |
| 有 18 路 scores 的槽 | **19,772 / 19,772** | §3 表 1 |
| 有 ≥1 个 verified 动作的槽 | **19,772 / 19,772**（`verified_slot_coverage = 1.0`） | §3 表 1 |
| `chosen_verified` 为真的槽 | **19,772 / 19,772**（无 verified 动作的槽 **0**，单列计数） | §3 表 1 |
| `executed_cand ≠ chosen_verified_cand` 的槽 | **6,288**（= 2,117 / 2,066 / 2,105） | §3 表 1 |
| 自检违反数 | **0**（0 unexplained），逐 shard 全 0 | §4 |
| HASH | 逐槽 `hash_post == 下一个槽 hash_pre` **19,772/19,772**；链头 3/3 与冷启动末条 `hash_post` 相同；与 T02C/T02D meta 的 HASH 计数**全对账通过** | §4.3 |
| 一次前向 + 一次显式 CE（256 槽，覆盖 3 seed） | executed：**CE 4.090642 / top1 0.65625**；chosen_verified：**CE 3.060187 / top1 0.683594**（ln18 = 2.890372） | §5.2 |
| 256 槽 × 300 步过拟合 | executed：loss **4.090642 → 1e-06**，top1 **0.65625 → 1.0**；chosen_verified：loss **3.060187 → 0.0**，top1 **0.683594 → 1.0** | §5.3 |
| **G420-0 L1 的数据前提是否成立** | **成立**：一份 19,772 槽、逐槽自有 obs/HUD/已执行动作/8 条真实历史/逐槽 verified 教师标签、HASH 0 差异的语料**已经存在且可被优化** | §6 |
| L1 是否因此通过 | **否**。L1 要求这份语料**被消费**（进入训练），属 T04 正题；本任务只交付数据前提 | §6 |

一句话：**T02c 的观测半 + T02d 的标签半，现在是一份 19,772 槽、自检 0 违反、
逐槽 verified、HASH 全对账的 2 帧语料；它已经通过「一次前向 + 显式 CE + 256 槽过拟合」
的可训练性验证。这仍然是 L1 的数据前提，不是 L1 本身。**

---

## 1 输入（只读）与产物

| 角色 | 路径 | 规模 | 只读使用 |
|---|---|---|---|
| 观测/动作半（T02c） | `.tmp_v420_data/T02c/rebuild/<shard-key>/2f/{slots.jsonl, obs/*.npy, meta.json}` | 19,772 槽；3 个 shard | ✅ 不写 |
| 标签半（T02d） | `.tmp_v420_data/T02d/reslot/<shard-key>/teacher_2f.jsonl` + `meta.json` | 每 shard 6,600/6,600/6,596 行（含 8 个冷启动） | ✅ 不写 |
| **join 产物** | `.tmp_v420_data/T04/dataset/<shard-key>/` | 19,772 槽；≈36 MB（slots+targets+meta；obs 是 symlink） | 新建 |

每 shard 产物：

```
dataset/<shard-key>/
  slots.jsonl      6,592 / 6,592 / 6,588 行：契约字段 + 教师标签 + chosen + 已执行动作 + 两半 provenance
  targets.jsonl    同数行：{slot_id, executed_cand, chosen_verified_cand, chosen_verified, bomb}
  obs/             play96x112.npy / hud.npy / act_hist.npy  → **相对 symlink** 到 T02c 原始 .npy
                   + SOURCE.json（mode/来源/每个文件的 sha256）
  meta.json        counts、逐文件 hash、两侧源 meta 的 HASH 数字、自检结果、limits
dataset/index.json 3 个 shard 的清单与 totals
```

`obs` 不复制、不重编码：T02c 的 `.npy` 就是训练输入，`meta.obs.files[*].sha256` 记录其字节
身份（`play96x112.npy` 每 shard 283,508,864 B / `hud.npy` 59,064,448 B / `act_hist.npy` 949,376 B）。
若目标文件系统拒绝 symlink（Windows 无开发者模式），实现会退回 `shutil.copy2` 并把
`kind` 记成 `copy, fallback_from=link` —— 这条回退只在两侧单测的临时目录里触发过，真实产物
是 symlink。

---

## 2 join 规则：`(shard_key, slot_id)`，且逐字段校验

两份产物由**不同进程**写出，冷启动记账也不同（T02d 把 8 个冷启动边界写成
`slot_id = -8 … -1`、`emitted=false`，因此它的标签文件比 T02c 的 slots 文件**早 8 行**）。
所以「第 8 行对第 8 行」不是对齐，只是巧合。本实现：

1. 读取 T02c `slots.jsonl` 时**按 slot_id 流式校验**：`slot_id == 行号` 且
   `decision_frame == 16 + 2·slot_id`（槽 0 = 帧 16，与 `rebuild`/`reslot` 的编号
   `(d-16)//2` 一致）；
2. T02d `teacher_2f.jsonl` 只收 `emitted=true` 的行，按 `slot_id` 建索引；`slot_id<0` 的
   `emitted=true`、重复 `slot_id` 都直接报错；
3. 逐槽按 `slot_id` 取标签，并校验 **`decision_frame` 相等**、`episode_seed` 相等、
   标签自身 `corpus.key == shard_key`、`emitted is True`；
4. 校验**已执行动作三方一致**：T02c 的 `(cand, bomb, mask)` == T02d 的
   `executed.{cand,bomb,mask}`，且 `mask_to_cand(mask) == (cand, bomb)` 与
   `cand_to_mask(cand, bomb) == mask` 双向可逆；
5. 校验观测/历史窗口**由帧号推出而非采信**：`obs_render_frames == [d-4..d-1]`、
   `history_frames == [d-16, d-14, …, d-2]`、`history` 长度 8、`check_no_future`、
   `check_history_excludes_current`（带帧号证明，重复动作合法）；
6. 校验标签形状：`scores[18]` 全有限、`verified_mask[18]` 0/1、
   `verified_count == sum(mask)`；
7. **`chosen` 由本模块重算**：只在 `verified_mask==1` 的候选里取 argmax（并列取最小索引），
   并与 T02d 记录的 `chosen`/`chosen_verified` **比对**；不一致就中止该 shard；
8. 收尾校验：`join` 后 `written == corpus_rows == len(labels)`；存在没有任何语料槽的标签
   （孤儿）、没有标签的语料槽（缺失）、重复标签，全部 **fail-closed 中止**，且输出目录
   只在自检通过后才从 `<shard>.tmp-<pid>` 改名为正式目录（旧目录改名成 `.stale-<stamp>`，
   不删除）。

**没有按行号拼接的路径**。单测 `test_join_by_key_not_by_row_order` 直接把「语料第 0 行 ×
标签第 0 行（= 冷启动 `slot_id=-8`）」喂给 join，断言它报 `join key mismatch`；
`test_join_fails_closed_on_a_shifted_slot_id` 把某个标签的 `decision_frame` 换成下一个槽的帧，
断言报「不描述同一时刻」。

### 2.1 契约缺口（登记，不改）

`touhou_rl/v420/contracts.py` 的冻结枚举 `LABEL_SOURCES = (teacher_slot0, teacher_slot1, none)`
里没有「逐槽 2 帧重查」这个名字。本模块**不修改** contracts（另一 owner 的文件，白名单外），
而是自带 `LABEL_SOURCES_DATASET = ("teacher_2f", "none")` 并用
`contract_label_sources()` 在每份 `meta.json` 里登记
`unknown_to_contract=["teacher_2f"]`。这与 T02c 对 `executed_slot1` 的处理方式一致，
缺口仍记为 [PROGRESS §6.2](../../PROGRESS.md)。

---

## 3 join 后计数（实测）

### 表 1 每 shard 与总计

| seed | shard | T02c 槽 | T02d emitted | 冷启动标签 | 18 路 scores | ≥1 verified | `chosen_verified` | 无 verified（单列） | `executed≠chosen` | `executed==chosen` | targets 行 | 墙钟 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | `d1-s0d1-n3300-seed10-shard` | 6,592 | 6,592 | 8 | 6,592 | 6,592 | 6,592 | **0** | 2,117 | 4,475 | 6,592 | 3.97 s |
| 11 | `d1-s0d1-n3300-seed11-shard` | 6,592 | 6,592 | 8 | 6,592 | 6,592 | 6,592 | **0** | 2,066 | 4,526 | 6,592 | 4.11 s |
| 12 | `d1-s0d1-n3300-seed12-shard` | 6,588 | 6,588 | 8 | 6,588 | 6,588 | 6,588 | **0** | 2,105 | 4,483 | 6,588 | 4.17 s |
| **合计** | 3 个 | **19,772** | **19,772** | **24** | **19,772** | **19,772** | **19,772** | **0** | **6,288** | **13,484** | **19,772** | **12.25 s** |

- `verified_slot_coverage = 1.0`、`verified_flags_total = 19,772`、`verified_flags_mean = 1.0`：
  T02d 的 `verify_k=1` 使**每槽恰好 1 个**候选带引擎判定（`verified_count` 直方图只有
  `{"1": 19772}`）。所以 `chosen` 是 **1-of-18 的 verified 目标**，不是 18 路排名，
  另外 17 个 score 带 `-5.0` 的未验证折扣。
- **标签覆盖从「半份」变成「整份」**：T02c 里 `label_source=teacher_slot0`（verified）
  9,886 槽、`executed_slot1`（未验证、无 scores）9,886 槽；join 后 **19,772 槽全部**
  带 T02d 的逐槽 verified 标签，原来的 9,886 个「只有行为目标」的槽不再缺 verified 标签。
  这一点用 `corpus_label_source_counts = {teacher_slot0: 9,886, executed_slot1: 9,886}`
  与 `label_source_counts = {teacher_2f: 19,772}` 两栏并排记录，**没有任何标签被复制或平移**。
- 两份目标**确实不同**：6,288 槽（31.8%）的 `executed_cand ≠ chosen_verified_cand`。
  这就是为什么 `targets.jsonl` 必须把两者分开写（见 §4.2）。

### 3.1 一条真实的 joined 记录（seed 10, slot 0）

```json
{"slot_id": 0, "decision_frame": 16,
 "obs_render_frames": [12,13,14,15], "history_frames": [0,2,4,6,8,10,12,14],
 "history": [7,7,7,7,7,7,7,7],
 "executed_cand": 7, "bomb": 0, "mask": 65,
 "scores": [113.041667,113.041667,112.923816,112.875,112.923816,113.041667,
            113.277369,118.375,113.277369,113.041667,113.041667,112.982741,
            112.958333,112.982741,113.041667,113.159518,113.208333,113.159518],
 "verified_mask": [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0], "verified_count": 1,
 "chosen": 7, "chosen_verified": true, "chosen_score": 118.375,
 "label_source": "teacher_2f", "label_verified": true,
 "hash_pre": "dcb130a7b8a7851f", "hash_post": "f61e822580d29c6b",
 "provenance": {"join_key": {"shard_key": "d1-s0d1-n3300-seed10-shard", "slot_id": 0},
                "corpus": {"label_source": "teacher_slot0", "label_verified": true, "obs_row": 0},
                "label": {"teacher_chosen": 7, "margin": 16.0,
                          "teacher_profile_sha256": "5545c199…"}}}
```

同一槽的 `targets.jsonl` 行：

```json
{"slot_id": 0, "decision_frame": 16, "executed_cand": 7,
 "chosen_verified_cand": 7, "chosen_verified": true, "bomb": 0}
```

第一个 `executed ≠ chosen` 的槽是 seed 10 slot 13（帧 42）：执行 `cand=16`，
教师 verified argmax 是 `cand=0`（`scores[0]=119.958333`，其余 17 个 114.958333；
`verified_mask` 只在 0 处为 1）。

---

## 4 自检

`self_check_dataset()` 重新读回已发布目录，用**与写入时同一个** `check_dataset_record()`，
再叠加文件级检查。违反数必须全 0 才能声称「2 帧语料可用」。

### 4.1 逐 shard 结果

| shard | 行 / targets | violations_total | unexplained_diffs | passed |
|---|---|---:|---:|---|
| seed 10 | 6,592 / 6,592 | **0** | **0** | ✅ |
| seed 11 | 6,592 / 6,592 | **0** | **0** | ✅ |
| seed 12 | 6,588 / 6,588 | **0** | **0** | ✅ |

### 4.2 检查项与执行次数（seed 10 为例，其余 shard 同构）

| 检查 | 每次执行的次数 | 违反 |
|---|---:|---:|
| `contract_tags` / `decision_span` / `commit_frames` / `obs_render_frames` | 6,592 | 0 |
| `history_length` / `history_frames` / `history_no_future` / `history_excludes_current` | 6,592 | 0 |
| `history_plane`（存盘 `act_hist` 行 == 记录 history 的 one-hot） | 6,592 | 0 |
| `executed_mask_roundtrip`（`mask ↔ (cand,bomb)` 双向） | 6,592 | 0 |
| `scores_length` / `scores_finite` / `verified_mask_length` / `verified_count` | 6,592 | 0 |
| `chosen_verified_argmax` / `chosen_matches_teacher` / `chosen_verified_flag` | 6,592 | 0 |
| `label_source` / `label_verified` / `no_fabricated_verified` | 6,592 | 0 |
| `provenance`（记录里写的是哪一对 (shard_key, slot_id)） | 6,592 | 0 |
| `slot_id_order` / `decision_frame_rule` / `episode_seed` | 6,592 | 0 |
| `join_missing_label` / `join_orphan_label` / `join_duplicate_label` | 1 | 0 |
| `targets_rows` / `targets_key` / `targets_separation` / `targets_domain` / `targets_consistency` | 1 / 6,592 | 0 |
| `obs_rows` / `obs_shape_dtype` / `obs_blank_play` / `obs_blank_hud` | 1 / 3 / 1 / 1 | 0 |
| `hash_chain`（逐槽 `hash_pre == 上一槽 hash_post`） | 6,592 | 0 |
| `hash_reconcile`（与 T02C/T02D 两侧 meta 对账，见 §4.3） | 1 | 0 |
| `source_unchanged`（两个输入文件的 sha256 仍是构建时记录的值） | 4 | 0 |

`targets_separation` 的含义：`targets.jsonl` 的 `executed_cand` 必须等于记录的
`executed_cand`、`chosen_verified_cand` 必须等于记录的 `chosen`、`chosen_verified` 必须一致、
`chosen_verified_cand is None ⇔ chosen_verified is False`。**两种目标不能互换、不能互相填充**；
单测 `test_targets_rows_cannot_be_swapped` 把两栏对调/置空，断言检查器报错。

### 4.3 HASH 对账（与两半的 meta 逐项比对）

| 项 | T02C meta | T02D meta | 本语料 | 一致 |
|---|---:|---:|---:|---|
| seed 10 `hash_post` 检查 / 全等 | 3,300 / 3,300 | 3,300（mismatches 0） | 6,592 逐槽 hash 对 | ✅ |
| seed 11 同上 | 3,300 / 3,300 | 3,300（mismatches 0） | 6,592 | ✅ |
| seed 12 同上 | 3,298 / 3,298 | 3,298（mismatches 0） | 6,588 | ✅ |
| 逐帧 HASH（两侧观测） | 13,200 / 13,200 / 13,192 | 13,200 / 13,200 / 13,192 | — | ✅ |
| `reset_hash_matches_first_pre` | true | true | — | ✅ |
| 逐槽链断裂 | — | — | **0** | ✅ |
| 链头（首槽 `hash_pre` == 冷启动末条 `hash_post`） | — | — | **3/3 shard** | ✅ |
| 源文件 sha256 仍为构建时的值 | — | — | 4/4 文件 × 3 shard | ✅ |

`hash_post_checks` = 3,300 是**四帧块**数，逐槽对是 6,592 条 —— 两个口径都记在
`meta.json#hash` 里，不混用。

---

## 5 可训练性验证（只证明可训练）

脚本：`.tmp_v420_t04a/trainability.py`（结果 [`dataset_2f.json`](dataset_2f.json) →
`trainability`）。checkpoint 用 `tools/rl/v410_student_eval.py::load_v410_checkpoint`
（importlib 只读加载）：`B0` / `ProbeNet` / **731,802** 参数 /
sha256 `3bc0ec4461701bebb2ffc551d61dffff77f15cd486125baf9db6eaafc741b3a6`。

### 5.1 取 256 槽

`select_spread` 按 shard 长度比例分配、shard 内等距取行（不是头部切片）：
**86 / 85 / 85**（seed 10/11/12），合计 **256**；`verified coverage = 1.0`；
其中 **73 槽**的两个目标不同。

### 5.2 一次前向 + 一次显式 CE（两个目标各一次，同一次前向）

打包走 `touhou_rl.v410.student.s1_data.pack_batch`（`pack_source = s1_data.pack_batch`，
并与 `obs_spec_v3` 的冻结归一化逐行比对通过）：`play (256,4,112,96)` / `hud (256,1,56,160)` /
`act_hist (256,144)` / `target (256,) int64`；`check_batch` 与 `check_batch_bindings` 均 0 违反。

| 目标 | 交叉熵 | top-1 | 备注 |
|---|---:|---:|---|
| `executed_cand` | **4.090642** | **0.65625** (168/256) | 行为克隆目标 |
| `chosen_verified_cand` | **3.060187** | **0.683594** (175/256) | 逐槽 verified 教师目标 |
| 参考：均匀分布 `ln 18` | 2.890372 | 1/18 | 只作标尺 |

单次前向 256 槽耗时 **0.092 s**（CPU），logits 形状 `(256,18)`、全有限。

> **这两个数字不是评测。** B0 是在 D410-S1（train seeds 1–20）上训练出来的，而 10/11/12
> 属于那批 seed，所以连「初始 top-1」都不能当作对该 checkpoint 的评价。它只说明：
> **这份语料的张量形状、归一化与标签索引，冻结模型能吃下去，且 CE 是有定义、能算的。**

### 5.3 256 槽 × 300 步过拟合（T04 学习检查层 1 的最小版）

协议在脚本里**先固定**（Adam，lr 3e-3，weight decay 0，全批 256，300 步，seed 0，CPU），
没有根据结果调参、没有扫 lr、没有换步数。

| 目标 | loss 起点 → 终点 | top-1 起点 → 终点 | 用时 |
|---|---|---:|---:|
| `executed` | **4.090642 → 1e-06** | **0.65625 → 1.0** | 65.9 s（0.217 s/步） |
| `chosen_verified` | **3.060187 → 0.0** | **0.683594 → 1.0** | 64.8 s（0.216 s/步） |

loss 曲线端点（每 25 步）：

| step | 0 | 25 | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| executed loss | 4.090642 | 0.000809 | 1.1e-05 | 2e-06 | 1e-06 | 1e-06 | 1e-06 | 1e-06 |
| executed top1 | 0.65625 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| chosen_verified loss | 3.060187 | 6.4e-05 | 6e-06 | 2e-06 | 1e-06 | 1e-06 | 0.0 | 0.0 |
| chosen_verified top1 | 0.683594 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

> **明确声明（必须与上面的数字一起读）：**
> 这只证明「**这批数据能被优化**」—— 张量与标签之间没有矛盾、梯度能流、目标可达。
> 256 槽对 731,802 参数是**几乎必然的记忆**，25 步内就到 top-1 = 1.0 恰恰说明它衡量的是
> 记忆而不是学习。它**不是**泛化、**不是**能力、**不是** G420-2，也**不构成**任何门结论。

---

## 6 对 G420-0 L1 的输入

**问题**：现在是否存在一份「每 2 帧一槽、每槽有自己的 obs + HUD + 已执行动作 + 8 条真实历史 +
逐槽 verified 教师标签、HASH 0 差异」的语料？

**回答：存在。** `.tmp_v420_data/T04/dataset/`，版本 `v420-2f-dataset/1`：

| 判据 | 实测 |
|---|---|
| 规模 | **19,772 槽**（train seeds 10/11/12 = 6,592 / 6,592 / 6,588） |
| 每槽自有 4 帧连续 `play`（`d-4..d-1`） | 19,772 / 19,772（`obs_render_frames` 逐槽校验） |
| 每槽自有 HUD（帧 `d-1`） | 19,772 / 19,772（`hud.npy` 行 = 槽行，非空） |
| 每槽自有已执行动作 | 19,772 / 19,772（`executed_cand`，mask 双向可逆） |
| 8 条**真实**历史（`d-16..d-2`，无当前/未来） | 19,772 / 19,772（`history_frames` + `check_no_future` + `check_history_excludes_current`） |
| 逐槽 18 路教师 `scores` + `verified_mask` | 19,772 / 19,772 |
| 逐槽 verified 目标（`chosen_verified`） | 19,772 / 19,772；无 verified 动作的槽 **0** 个（单列，不冒充） |
| HASH 0 差异 | 逐槽链 19,772/19,772、链断裂 0、与 T02C/T02D 两侧 meta 全对账 |
| 自检 | 违反 **0**、unexplained **0**、三 shard `passed=true` |
| 可训练 | 一次前向 + 两次显式 CE 有定义；256 槽 300 步 loss→~0 |

**但这仍然只是 L1 的「数据前提」。** L1 的判据是「所有**已消费**样本每 2 帧一槽、每槽有自己
观测与标签」。这份语料目前**还没有被任何训练运行消费**：
本任务交付的是「接成一份可训练语料」，训练本身的接入（数据加载器 → 训练循环 → 学习检查
层 1/2、G420-2 的判定）属于 **T04 正题**。因此：

- **L1 数据前提：成立**（本文所有数字）；
- **L1 判定：仍 fail**，理由从「标签只有半份」变成「这份语料未进入训练」；
- 本文件**不**改动 [G420_0.md](../T02/G420_0.md) §6 的判定，也不把它算作门通过。

---

## 7 变更文件（严格白名单）

| 文件 | 变更 |
|---|---|
| `touhou_rl/v420/dataset_2f.py` | **新建**：`Dataset2FConfig` / `build_2f_dataset` / `build_dataset_record`（join 与逐字段校验）/ `choose_verified` / `check_dataset_record` / `build_targets_record` / `check_targets_record` / `target_vector` / `history_onehot` / `self_check_dataset` / `load_dataset(_shard)` / `dataset_summary` / `contract_label_sources` |
| `touhou_rl/v420/batch_2f.py` | **新建**：`pack_planes`（复用 `s1_data.pack_batch`，并与 `obs_spec_v3` 逐行交叉验证）/ `Batch2F`（`BCBatch` 形状 + `to_bc_batch()`）/ `build_batch` / `select_spread` / `select_slice` / `verified_indices` / `check_batch` / `check_batch_bindings` / `one_hot_targets` |
| `tests/test_v420_dataset_2f.py` | **新建**：29 条纯逻辑测试（小 fixture，无引擎、无真实数据） |
| `Document/Update/Alpha/v4.2.0/evidence/T04/DATASET_2F.md` | **新建**（本文件） |
| `Document/Update/Alpha/v4.2.0/evidence/T04/dataset_2f.json` | **新建**（机器可读镜像） |
| `.tmp_v420_data/T04/dataset/**` | **新建**：join 后的 3 个 shard + `index.json` |
| `.tmp_v420_data/T04/{dataset_build.json,trainability.json}` | 新建：构建报告与可训练性报告 |
| `.tmp_v420_t04a/*` | 运行脚本（`probe.sh`、`pytest_wsl.sh`、`build_dataset.sh`、`trainability.sh`、`report_numbers.py`、`write_evidence_json.py` 等） |

**未改动**（白名单外，全部只读）：`touhou_rl/v420/{contracts,replay_audit,eval,counterfactual,rebuild,reslot,__init__}.py`、
`tools/rl/v420.py`、`tools/rl/v410_student_eval.py`、`configs/v420/**`、任何既有 tests、
`touhou_rl/v400/**`、`touhou_rl/v410/**`、`.tmp_v420_data/T02c/**`、`.tmp_v420_data/T02d/**`、
`.tmp_v410_*`、旧证据、`ASSET_SYNC.*`。没有 `git add/commit/push`；没有读 confirm
seed 61000–61499，也没有碰 60000–60199。

> 说明：代码定稿前 T04 目录被重建过几次，每次都会把旧目录改名成
> `<shard>.stale-<stamp>`（不删除）。定稿后用同一协议重建一次、自检通过（0 违反），
> 之后删除了这些**内容相同的被替代副本**，只留最终 3 个 shard。它们都是本项目自己的
> 19,772 槽语料，不含任何其他来源数据。

---

## 8 验收命令与统计

```bash
# WSL（资产侧）
ROOT420=/mnt/d/WorkShop/Touhou_RL; PY420=/root/v400-venv/bin/python; V420_DATA=$ROOT420/.tmp_v420_data
PYTHONPATH=$ROOT420 $PY420 -m compileall -q touhou_rl/v420                       # exit 0
PYTHONPATH=$ROOT420 $PY420 -m pytest -p no:cacheprovider \
  --basetemp $V420_DATA/tests/T04 -q tests/test_v420_dataset_2f.py                # 29 passed, exit 0
```

```powershell
# Windows
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m compileall -q touhou_rl\v420                                            # exit 0
python -m pytest -p no:cacheprovider --basetemp .tmp_v420_pytest_win4 -q tests/test_v420_dataset_2f.py
                                                                                 # 29 passed, exit 0
```

| 侧 | compileall | pytest | 退出码 |
|---|---|---|---|
| WSL（`/root/v400-venv`） | 0 | **29 passed** | 0 / 0 |
| Windows（`python 3.12.10`） | 0 | **29 passed** | 0 / 0 |

构建与可训练性脚本（同一 WSL 解释器，`PYTHONPATH=$ROOT420`）：

| 脚本 | 结果 | 退出码 |
|---|---|---|
| `.tmp_v420_t04a/build_dataset.sh` | 3 shard、19,772 槽、自检 0 违反、12.25 s | 0 |
| `.tmp_v420_t04a/trainability.sh` | 一次前向 + 2×CE + 2×300 步过拟合、130 s | 0 |

测试覆盖（29 条）：

| 组 | 条数 | 断言的是什么 |
|---|---:|---|
| join 对齐 / 对齐键 | 5 | 正常 join 逐槽 `(shard_key, slot_id)`；缺标签、孤儿标签、按行号拼、标签 `corpus.key` 错 → 全部 fail-closed |
| `decision_frame` | 2 | 标签帧错位、整块移位 → 报错（不是"取最近的行"） |
| 执行动作 / mask / history | 3 | `executed` 三方不一致、`mask` 不能解码回 `(cand,bomb)`、history ≠ 8 或含未来 → 报错 |
| 无 verified 动作的槽 | 2 | 单列计数、`chosen=None`/`chosen_verified=false`，可做 executed 目标但**不能**做 verified 目标 |
| `chosen` | 3 | 必须是 verified 集合的最大值索引（并列取最小）、越界/伪造被检查器抓到 |
| 两种目标 | 2 | `targets.jsonl` 分栏写入；对调/置空被 `targets_separation`/`targets_consistency` 抓到 |
| batch 形状 / dtype / 归一化 / 复用 | 3 | `(B,4,112,96)`/`(B,1,56,160)`/`(B,144)`/`(B,) int64`、`uint8/255`、`pack_batch` 与 `obs_spec_v3` 逐元素相等 |
| `act_hist` 扁平化顺序 | 2 | 第 k 块的热点必须是 `k*18+cand`；列优先（转置）布局被拒绝 |
| 自检 | 3 | 正常语料 0 违反；被破坏的 HASH 链与 history 平面各被单独抓到 |
| 选择 / 路径 / seed | 4 | 等距选择确定性且不重复、越界切片被拒、输出必须在 `v420_data` 下且不得落在旧资产、confirm seed 被拒 |

---

## 9 边界（必须与上面的「通过」一起读）

1. **不是学习结果**：一次前向与 256 槽过拟合只证明「数据能被优化」。没有 dev/confirm 评测，
   没有泛化测量，没有能力主张，G420-2 未开始。
2. **B0 的初始 top-1 不是评测**：B0 在 D410-S1（train seeds 1–20）上训练，10/11/12 在其中，
   训练集与这 256 槽同源。
3. **`chosen` 是 1-of-18**：`verify_k=1` 使每槽恰有 1 个候选带引擎判定；另外 17 个 score 带
   `-5.0` 未验证折扣。所以它不是 18 路排名，也不能当排名用。
4. **两种目标不可混用**：6,288 槽（31.8%）的 `executed ≠ chosen`。`targets.jsonl` 分栏就是
   为了让下游**显式**选一个；把它俩相加/平均/互相填充，正是 G420-1 要抓的缺陷。
5. **冷启动仍然排除**：24 / 19,796 个边界（帧 0–14）没有 4 帧观测或 8 决策历史，**没有填充**。
6. **obs 是 symlink**：本语料不重编码像素，`play` 字节仍是 T02c 的；`meta.obs.files[*].sha256`
   记录身份。若把语料移到别处，需要跟着搬 T02c 的 `.npy`（或用 `obs_mode=copy` 重建）。
7. **HASH 是「同一条重放轨迹」的证明**：逐槽链只说明标签行对应的重放与语料是同一轨迹，
   不重新测引擎。
8. **契约缺口未补**：`teacher_2f` 仍不在 `contracts.LABEL_SOURCES` 里（白名单外），
   每份 meta 都登记了 `unknown_to_contract`。
9. **只有 3 个 train seed**：语料不代表全量训练集，也不代表 dev/confirm。

---

## 10 产物清单

| 路径 | 大小 | 内容 |
|---|---|---|
| `.tmp_v420_data/T04/dataset/<shard-key>/slots.jsonl` | 10.1 MB × 3 | 19,772 条 joined 记录 |
| `.tmp_v420_data/T04/dataset/<shard-key>/targets.jsonl` | 1.6 MB × 3 | 19,772 条双目标 |
| `.tmp_v420_data/T04/dataset/<shard-key>/meta.json` | 6 KB × 3 | counts / hash / obs / source / 自检 |
| `.tmp_v420_data/T04/dataset/<shard-key>/obs/` | symlink | play/hud/act_hist + `SOURCE.json` |
| `.tmp_v420_data/T04/dataset/index.json` | 3.9 KB | 3 shard 清单 + totals |
| `.tmp_v420_data/T04/dataset_build.json` | 36.8 KB | 构建报告（含逐 shard 自检明细） |
| `.tmp_v420_data/T04/trainability.json` | 9.2 KB | 一次前向 + 2×CE + 2×300 步曲线 |
| `Document/Update/Alpha/v4.2.0/evidence/T04/dataset_2f.json` | 56.7 KB | 本文件的机器可读镜像 |
