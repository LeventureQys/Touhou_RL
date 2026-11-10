# EXP-P1-020 — teacher v2f：与底部软罚同构的连续顶部/贴角罚分（PSN 0.625 → ≤0.5）

- **日期**：2026-11-09
- **状态**：**完成（v2f 8 面完整轮，`partial=false`）→ PSN 未达标**：8 面死亡 **5 / 8 = 0.625**（目标 ≤4 死 / ≤0.5）。顶部连续罚分**按设计生效**（v2a 的 2/2 顶角死亡被搬走、Stage4B +38.6 %、Stage6A +41.7 %），但**死亡数不变**（5/5 死于 `bomb_exhausted` + `wall_dense`）。按协议门 **S1N 20 局未执行**；G1 **仍不判过**。**v2g（p_top=3.0）第二臂也跑完 8 面：同样 5 死 / 0.625，且死亡更靠底（4/5 在 py 411–424）**（§8.1）。
- **上游**：[EXP-P1-017](EXP-P1-017-teacher-v2e.md)（v2e：Boss 本体代价打偏，PSN 仍 0.625；**`top_pin` 已是主导死因 3/5**）、[EXP-P1-012](EXP-P1-012-g1-evaluation.md)（G1 门与帧上限表）、[EXP-P1-015](EXP-P1-015-teacher-v2-bottom-band.md)（v1 → v2a：底部软罚 `1.5/2.0` 把贴底率 36–67 % 压到 11–26 %，PSN 5/8 = 0.625；v2c 硬剪/v2d 二值顶罚 8.0 均未达标）
- **产物**：`touhou_rl/v400/teacher/config.py`（`Weights.top_*`/`corner_*` 与 v2f/v2g 两档）、`search.py`（`Plan.top`/`corner`、`score_plan`、`_finish_roots`）、`label.py`（反序列化）、`decide.py`（`_replace` 带上新项）、`tools/rl/v400_teacher.py`（观测 `chosen_top`/`chosen_corner`）、`tests/test_v400_teacher.py`（+8 项）、`.tmp_v400_v2f/`
- **一句话**：以 **v2a 逐项不变**为基底，把"顶部"从 `high_penalty=3.0` 的**二值**开关改成与底部**同构的连续软罚**（`top_penalty·(200−y)/180`，1.5 一档）并加一个**贴角**平罚（1.0），二值 `high_penalty` **保留**，然后完整重测 8 面。

---

## 1. 目标与口径

| 项 | 口径 |
|---|---|
| 目标 | **PSN 每面死亡 ≤ 0.5**（8 面口径），即 **8 面死亡合计 ≤ 4** |
| 判死依据 | **引擎 verdict 唯一**：`STATE.f6`（deaths）增量；到帧上限**不写成"过面"** |
| 过面 | `STATE.f1` 离开起跑面（`stage_clear=true`），否则记 `max_frames` 或 `death` |
| 帧上限 | EXP-P1-012 §2 同一张表：Stage1 19,887 / Stage2 27,788 / Stage3 30,014 / Stage4A 41,583 / **Stage4B 45,000（fallback）** / Stage5 38,513 / **Stage6A 45,000（fallback）** / Stage6B 54,605 |
| 其它 | normal、seed 1、每面各自 `RESET`、无 `--follow-stage`、**6 并发**、每 worker 独立 data-dir |
| 完整轮 | 8 面各 1 局，**没有 partial**：8/8 面都要有真实 JSONL |
| S1N 门 | 只有 PSN 达标才跑；seeds 1000–1019、stage 0、cap 19,887、≤6 并发、独立 data-dir、**≥18/20 且每局 0 死** |

## 2. 配置与理由

### 2.1 逐项配置（v2a 是唯一基线）

| 参数 | v1 (`full`) | v2a（基线） | **v2f** | v2g（备档，未跑） |
|---|---|---|---|---|
| horizon / beam / verify_k / max_iterations | 48 / 64 / 3 / 3 | 48 / 64 / 3 / 3 | **48 / 64 / 3 / 3** | 同 v2f |
| `bottom_penalty` / `bottom_y` / `bottom_span` | 0 / 400 / 32 | 1.5 / 400 / 32 | **1.5 / 400 / 32（保留）** | 同 v2f |
| `bottom_extra_penalty` / `bottom_extra_y` / `bottom_extra_span` | 0 / 420 / 12 | 2.0 / 420 / 12 | **2.0 / 420 / 12（保留）** | 同 v2f |
| **`top_penalty` / `top_y` / `top_span`** | 0 / 200 / 180 | 0 / 200 / 180 | **1.5 / 200 / 180（新增）** | **3.0** / 200 / 180 |
| **`corner_penalty` / `corner_x`** | 0 / (64, 312) | 0 / (64, 312) | **1.0 / (64, 312)（新增）** | 1.0 / (64, 312) |
| `high_penalty` / `high_y` | 3.0 / 200 | 3.0 / 200 | **3.0 / 200（保留）** | 同 v2f |
| `edge_penalty` / `edge_band` | 1.0 / 24 | 同 | **同 v2a** | 同 v2f |
| `escape_weight` | 0 | 0 | **0**（v2b 的逃脱项不采用） | 0 |
| `boss_penalty` / `boss_radius` | 0 / 48 | 0 / 48 | **0**（EXP-P1-017 已证伪：5/5 死亡点 65–343 px，48 px 一次都没触发） | 0 |
| `bomb_rule1_max_death_frame` | `None` | 12 | **12（保留）** | 12 |
| `bomb_rule1_min_bombs_left` | 0 | 0 | **0（保持关闭）** | 0 |
| `bomb_rule1_max_margin` | `None` | `None` | **`None`（保持关闭）** | `None` |
| `clip_y`（硬剪） | `None` | `None` | **`None`（保持关闭）** | `None` |

`v2f`/`v2g` 注册在 `config.CONFIGS`，`PROFILES = (V2A, V2B, V2C, V2D, V2E, V2F, V2G, V3A, V3B)`；CLI 的 `--mode` 由 `choices=tuple(sorted(CONFIGS))` 自动暴露，默认仍是 `full`。

### 2.2 三个决定与理由

**(a) 保留二进制 `high_penalty=3.0`，不与连续项合并。**
`high` 是"进入 y<200 这条带"的**门票**，连续项是"带内深度"的**斜率**。两者分工：

- 若删掉 `high`、只留 `1.5·(200−y)/180`，则在 `y=190` 只罚 0.083 分 —— 等于**放开** y∈(150,200) 这一段，而这正是 v2a 六个"逐帧相同"的面赖以稳定的中场上沿排序；v2a 在这一段是靠 3.0 的平罚把它们挡在下面的。
- 若保留 `high`，`y=190` 仍收 3.0（**与 v2a 逐位一致**），而 `y=20` 收 3.0+1.5=4.5，于是"带内"第一次有了**梯度**：从 190 往 20 走，每 12 px 多付 0.1 分，到顶角共多付 1.5+1.0=2.5 分。
- 因此 v2f 对**带外**行为是 v2a 的严格复制（这可由 `p_top=0` 时逐位相等的单测钉住），只改**带内如何排**。这也是"与底部同构"的准确含义：底部是 `bottom 1.5 + extra 2.0` 的纯连续项（底部本来就没有二值项），顶部是"二值门票 + 连续深度"。

**(b) 为什么首档取 `p_top=1.5`（而不是像 v2d 那样 8.0）。**
底部软罚的实测有效剂量是 `1.5`（贴底率 36–67 % → 11–26 %），v2f 取同一剂量；而 v2d 的 8.0 是加在**二值**项上的，它让 `y=199` 与 `y=20` 同罚 8.0（带内仍无梯度），且与"留 1 颗弹/margin 门槛"这两个已被证伪的守卫混在一起，效果不可归因。1.5 的增量在顶角合计 `1.5 + 1.0 = 2.5` 分，是 margin 上限 16 分的 **15.6 %**，与底部 3.5 分（21.9 %）同量级；若首轮不够，`v2g` 的 3.0 是同构的下一档（顶角 4.0 分 = 25 %）。

**(c) 为什么还要贴角项。** 3/5 的死亡点 `px` 距左右边 < 56 px（`py` 21.7 / 51.8 / 27.3），只罚"顶"会把"正上方中央"和"顶角"混为一谈；贴角平罚 1.0 让**同时在顶带内且在左右 64/312 之外**的末端多付一分。它只在 `y < high_y` 时生效，因此和 `high`/`top` 是同一坐标系里的三个可观测项，JSONL 里分别记 `high`/`top`/`corner`。

### 2.3 评分式（v2f，living plan）

```
score = 100 + min(margin,16) + 4·align + 2·min(items,1) − 1·edge − 3·high
        − [1.5·max(y−400,0)/32 + 2.0·max(y−420,0)/12]        # v2a 底部软罚（保留）
        − [1.5·max(200−y,0)/180]                              # top（新增，连续）
        − [1.0 · 1{x<64 ∨ x>312} · 1{y<200}]                  # corner（新增，平罚）
```

死亡分支不变：`score(dead) = death_frame / horizon · 10`（不含任何几何项）。
`top`/`corner` 只作用于 living plan；`p_top=0 ∧ corner=0` 时两项恒为 0，**v2a 逐位不变**（单测 `test_v2f_at_zero_top_weights_is_bit_identical_to_v2a`）。

## 3. 实现（改了哪些行）

| 文件 | 改动 | 说明 |
|---|---|---|
| `touhou_rl/v400/teacher/config.py` | `Weights.top_penalty/top_y/top_span/corner_penalty/corner_x`（默认全 0/200/180/(64,312)）+ `top_penalty_at()` / `corner_penalty_at()`；`_V2F_WEIGHTS`+`V2F`、`_V2G_WEIGHTS`+`V2G`、`CONFIGS`、`PROFILES` | 标量口径唯一来源；`0.0` 时 v1/v2a/v2e/v3 逐位不变 |
| `touhou_rl/v400/teacher/search.py` | `Plan.top`/`Plan.corner`（`as_dict`）；`score_plan` 的新项与 `parts`；`_finish_roots` 的向量式（与标量**逐元素同一表达式**）；`beam_search_reference` 的 `Plan(...)` | 两条搜索路径由单测钉住相等 |
| `touhou_rl/v400/teacher/label.py` | `Plan.from_dict` 读 `top`/`corner`（缺省 0.0） | 旧 JSONL 仍可反序列化 |
| `touhou_rl/v400/teacher/decide.py` | `_replace()` 一并复制 `boss`/`top`/`corner`（`score` 本来就逐位复制） | 只补诊断字段，不改任何决策 |
| `tools/rl/v400_teacher.py` | `bomb_rule1` 观测块加 `chosen_top`/`chosen_corner` | 仅观测：可从 JSONL 审计"顶部罚分是否真的计入选中计划" |
| `tests/test_v400_teacher.py` | +8 项（公式/贴角/档注册/与 `high` 的叠加/翻转机制/向量=标量/`p_top=0` 逐位不变/label 往返） | 既有 97 项全绿 |

## 4. 评测协议（实际执行）

- 驱动：`.tmp_v400_v2f/round.sh` → 复用 `.tmp_v400_v2/psn_v2.py`（只加 `--mode`，**未改** `tools/rl/v400_g1_eval.py`）：`--mode v2f --tag round1 --jobs 6 --timeout 9000 --data-dir-root /root/v400-data/g1`。
  - 为什么是 9000 而不是 EXP-P1-017 的 4200：本轮启动时机器上已有 v3 扫描（6 教师）与 D0 1 路，`th08-modern` 17 个、load 35，而"活下来的面"比"早死的面"更慢（要走到 cap）。第一次以 4200 启动后 **0 个面完成**即主动中止（见 §10 #3），正式轮改用 9000，**8 面全部自然结束，无一面触到超时**（最大 3,231 s）。
- cap 由 `v400_g1_eval.psn_caps(load_human_reference())` 现算（= EXP-P1-012 表），每面 1 局、seed 1、normal、`--stop-on-death`、无 `--follow-stage`。
- 6 并发、每 worker 独立 data-dir（`/root/v400-data/g1/inst0..5`）。
- 每面一个 JSONL + 日志，完成即追加 `runs.jsonl`；表由 `.tmp_v400_v2f/table.py` 生成。
- 判达标 = **8 面死亡合计 ≤ 4**（PSN = 合计/8 ≤ 0.5）。

## 5. 结果：v2f 8 面完整轮（`partial=false`）

驱动 `.tmp_v400_v2f/round.sh`：`--mode v2f --tag round1 --jobs 6`，cap 见表，seed 1、normal、`--stop-on-death`。**8/8 面各 1 局、各自真实 JSONL、8/8 `exit=0`、8/8 引擎 SHA 一致（`f172399f…264e`）。**

| stage | 面 | cap | 上限来源 | 过面 | frames | 死亡 | stop | exit | 已用弹 | 墙钟 s | **v2a frames** | v2a 死亡 | 与 v2a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | Stage3 | 30,014 | `ceil(1.5×20,009)` | false | 23,172 | **1** | death | 0 | 3.0 | 2,035 | 25,052 | 1 | **变差 −1,880** |
| 3 | Stage4A | 41,583 | `ceil(1.5×27,722)` | **true** | 29,564 | **0** | stage_changed | 0 | 1.0 | 3,231 | 30,952 | 0 | 仍过面（更早过面 −1,388） |
| 4 | Stage4B | 45,000 | fallback | false | **24,068** | **1** | death | 0 | 3.0 | 2,266 | 17,364 | 1 | **变好 +6,704（+38.6 %）** |
| 5 | Stage5 | 38,513 | `ceil(1.5×25,675)` | false | 21,660 | **1** | death | 0 | 3.0 | 2,443 | 21,504 | 1 | 基本不变 +156 |
| 6 | Stage6A | 45,000 | fallback | false | **19,584** | **1** | death | 0 | 3.0 | 2,383 | 13,820 | 1 | **变好 +5,764（+41.7 %）** |
| 7 | Stage6B | 54,605 | `ceil(1.5×36,403)` | false | 17,564 | **1** | death | 0 | 3.0 | 1,837 | 17,564 | 1 | **逐帧相同** |
| 0 | Stage1 | 19,887 | `ceil(1.5×13,258)` | **true** | 13,484 | **0** | stage_changed | 0 | 0.0 | 757 | 13,484 | 0 | **逐帧相同** |
| 1 | Stage2 | 27,788 | `ceil(1.5×18,525)` | **true** | 21,740 | **0** | stage_changed | 0 | 0.0 | 1,275 | 21,740 | 0 | **逐帧相同** |

- **8 面死亡合计 = 5，均值 = 0.625 → PSN = 0.625（目标 ≤ 0.5）→ 未达标**；过面 **3/8**（与 v2a 相同的三面：Stage1/2/4A）；测满 8/8 面。
- 帧数合计 **170,836**（v2a 同轮 161,480，**+9,356 = +5.8 %**）；单面最大 3,231 s，整轮墙钟 **3,456 s**（6 并发，18:47:22 → 19:44:58）。
- 并发环境：本轮启动时 **`pgrep -xc th08-modern` = 17、load average 35.3**（v3 扫描 6 个教师 + D0 1 路），运行中 load 冲到 **38–42**；帧数与 v2e 卡 §10 的结论一致（load 只改墙钟，不改帧数）——Stage6B/Stage1/Stage2 的 frames 与 v2a 逐个相同即为证。
- 第一轮曾在 18:44:56 以 `--timeout 4200` 启动，因上述并发导致"长命面 >70 min"的截断风险，**在没有任何面完成时**主动中止（`ROUND_EXIT=143`，`runs.jsonl` 为空），18:47:22 以 `--timeout 9000` 重跑。**本卡没有任何数据来自那次中止的运行。**

## 6. 顶部罚分做了什么：v2a 的 2/2 顶角死亡被搬走，但死亡数不变

先校正前提：**以 v2a（本轮基线）计，顶角死亡是 2/5**（Stage4B `py=51.8`、Stage6A `py=38.6`）；EXP-P1-017 的 "3/5" 是 **v2e** 的（多出的 Stage3 `py=21.7` 是 v2e 的 Boss 项造成的一次分叉，v2a 的 Stage3 本来就死在底部 `py=412.4`）。v2f 把 v2a 的 **2 个顶角死亡全部搬走**：

| 面 | v2a 死亡点 (px, py) | **v2f 死亡点 (px, py)** | v2a frames | v2f frames | 变化 | 选中计划顶部罚分触发率 |
|---|---|---|---|---|---|---|
| Stage4B | (16.5, **51.8**) 顶角 | (**330.9, 424.0**) 底右角 | 17,364 | **24,068** | **+6,704** | 3 % |
| Stage6A | (89.0, **38.6**) 顶部 | (**243.4, 408.0**) 底部 | 13,820 | **19,584** | **+5,764** | 1 % |
| Stage3 | (186.3, 412.4) 底部 | (184.0, 407.2) **底部** | 25,052 | 23,172 | −1,880 | 0 % |
| Stage5 | (226.3, 392.2) 中场 | (109.8, 386.7) **中场** | 21,504 | 21,660 | +156 | 2 % |
| Stage6B | (192.0, 345.6) 中场 | (192.0, 345.6) **中场（逐帧相同）** | 17,564 | 17,564 | 0 | 0 % |
| Stage4A | 过面 | 过面（y=384） | 30,952 | 29,564 | −1,388 | 1 % |

**`top_pin` = 0/5（v2a 是 2/5），`bottom_pin` 从 1/5 升到 4/5。** 也就是说：**罚分治好了"贴顶"，但死亡只是换了边**。

### 6.1 机制：它不是在"稳定避开顶角"，而是在**带边缘打破平局**，随后确定性分叉

`.tmp_v400_v2f/divergence.py` 逐决策对比 v2f 与 v2a 的 `chosen_plan`（`first_action`/`duration`/`end_xy`）：

| 面 | 首次分叉 | v2f 选择 | v2a 选择 | 该决策后顶部罚分计入的决策数 | 均值（非零时） | 最大 |
|---|---|---|---|---|---|---|
| Stage3 | 决策 609（frame 2,440） | act 7, end_y **188.5** | act 8, end_y 183.5 | 5 | 0.208 | 0.292 |
| Stage4B | 决策 581（frame 2,328） | act 0, end_y **191.9** | act 0, end_y 186.3 | 190 | 0.597 | 1.035 |
| Stage5 | 决策 314（frame 1,260） | act 0, end_y **171.0** | act 0, end_y 166.3 | 88 | 0.768 | 1.533 |
| Stage6A | 决策 1,949（frame 7,800） | act 6, end_y **198.3** | act 0, end_y 191.5 | 70 | 0.600 | 1.406 |
| Stage6B / Stage1 / Stage2 | **无分叉**（逐帧相同） | — | — | 0 | 0 | 0 |

- 顶部罚分在 **0–3 %** 的决策上进入选中计划，非零时均值只有 **0.21–0.77 分**（最大 1.53 分，即 `y≈20` 的满额 + 贴角）。
- 首次分叉全部落在 **y ≈ 171–198**：那是 `top_y=200` 的**带入口**，罚分只有 0.01–0.24 分；也就是说，起作用的是"在近似平局处加了 0.01 分"，而不是"在顶角加了 2.5 分把计划推开"。引擎对相同输入是确定的，于是 0.01 分 → 另一个 action → 整条轨迹分叉 → 24,068 / 19,584 帧。
- **诚实口径**：因此 `+38.6 %` / `+41.7 %` **不是剂量-反应曲线**，而是各一条确定性轨迹；Stage3 的 −1,880 同理。`top` 项本身只在 4B 的 190 / 6A 的 70 个决策上真正计费，其余面为 0。

**逐决策证据（`.tmp_v400_v2f/one_decision.py`，相同 frame、相同状态下的两个候选末端）**：

| 面/决策 | 轮次 | 选中 | end_xy | margin | high | top | corner | **score** |
|---|---|---|---|---|---|---|---|---|
| Stage4B 决策 581（frame 2,328） | v2a | act 0 | (372.686, 186.260) | 16.0（饱和） | 1.0 | 0 | 0 | **112.000** |
| 同上 | v2f | act 0 | (375.029, 191.917) | 16.0（饱和） | 1.0 | **0.067** | **1.0** | **110.933** |
| Stage6A 决策 1,949（frame 7,800） | v2a | act 0 | (201.221, 191.515) | 12.75 | 1.0 | 0 | 0 | **113.726** |
| 同上 | v2f | act 6 | (198.392, 198.343) | 12.75 | 1.0 | **0.0138** | 0 | **113.701** |

把 v2a 的那个末端**放到 v2f 的评分下**重算：Stage4B 得 `112.000 − 1.5·13.74/180 − 1.0 = 110.885` → 比 v2f 选中的 110.933 **低 0.047 分**；Stage6A 得 `113.726 − 1.5·8.485/180 = 113.655` → 比 v2f 的 113.701 **低 0.046 分**。**两次"改变 2 万帧走向"的分叉，都只值 0.05 分。**

- 机制上的原因：`beam_search` 的剪枝只看 `(died_now, −margin, action)`，**权重只进入 `_finish_roots` 的末端打分**（`search.py` 中权重唯一的入口），所以新项**不能塑造 beam**，只能在"beam 已经留下的末端"里挑一个。Stage4B 那次两个候选末端**都在右上角**（x≈373–375 > 312），`corner` 对两者都收 1.0，真正打破平局的是 `top` 的深度差（0.1145 vs 0.067）。
- 这解释了"为什么同 1.5 的权重在 3 个面上逐帧不变、在 2 个面上 +40 %"：不是权重强弱，而是**那个面上是否存在 margin 饱和的近似平局**。也因此**不能**用加大权重来外推收益。


## 7. 剩余死因：5/5 是"弹幕墙 + 弹用光"，不是边缘

`.tmp_v400_v2f/causes.py`（JSONL 代理标签，口径同 EXP-P1-017 §8，非逐帧像素分类器）：

| 面 | frames | 死亡 | 末帧 py | 末帧 px | 末 30 决策均值 py | 余弹 | 末决策 48 帧盒均值 | 代理标签 |
|---|---|---|---|---|---|---|---|---|
| Stage3 | 23,172 | 1 | 407.2 | 184.0 | 374.3 | 0 | 137 | `bottom_pin` + `bomb_exhausted` + `wall_dense` |
| Stage4B | 24,068 | 1 | 424.0 | 330.9 | 412.7 | 0 | 96 | `bottom_pin` + `corner` + `bomb_exhausted` + `wall_dense` |
| Stage5 | 21,660 | 1 | 386.7 | 109.8 | 380.8 | 0 | **706** | `bomb_exhausted` + `wall_dense` |
| Stage6A | 19,584 | 1 | 408.0 | 243.4 | 421.1 | 0 | 309 | `bottom_pin` + `bomb_exhausted` + `wall_dense` |
| Stage6B | 17,564 | 1 | 345.6 | 192.0 | 348.0 | 0 | 48 | `bomb_exhausted` + `wall_dense` |

1. **5/5 死时 0 弹**（`bombs_spent` 全为 3.0）：与 EXP-P1-016/017 完全一致，"用光后死"没有被任何边缘项改变。
2. **4/5 死在场地底部**（py 386.7–424.0，其中 3 个 >400 即在底部带内），而底部带本来就有 `1.5 + 2.0 = 3.5` 的连续罚分——只比顶部满额（`3.0 + 1.5 + 1.0 = 5.5`，或贴角 `edge` 再 +1.0）低 2.0 分。**底部罚分并不比新的顶部罚分弱多少，但死亡还是发生在底部**，说明"边缘罚分的强弱"不是约束。
3. 真正的约束是 **`wall_dense`（末决策 48 帧致死盒均值 48–706）+ 0 弹**：48 帧的视野里已经没有任何"活着且不被墙压住"的末端可选。Stage5 的 706 盒/帧 与 EXP-P1-016 的"bullet_wall 6/6"是同一件事。

## 8. 是否 ≤ 0.5、S1N 与 v2g

- **PSN = 5/8 = 0.625 > 0.5 → 未达标。** 过面 3/8。
- **S1N 20 局：未执行（协议门未过）。** 与 EXP-P1-017 同一口径：只有 PSN ≤ 0.5 才跑 S1N；不把"没跑"写成"通过"，也不拿 v2f 的 Stage1 单面（13,484 帧 0 死过面，与 v2a **逐帧相同**、顶部罚分触发率 0 %）冒充 20 局结果。**G1 仍不判过。**
- **v2g（p_top=3.0）：在 v2f 轮结束后（机器转为空闲：load 2.0）按任务书条件启动，6 并发、同 cap、同 seed。** 先验判断写在 §2.2（b）：v2f 已经把顶部死亡清零、死亡迁移到底部，**加大顶部权重在机制上无法处理底部/中场死亡**；跑 v2g 是为了把"更强一档"这个预登记分支也测掉，而不是因为它有希望过门。结果见 §8.1。

### 8.1 v2g（`p_top=3.0`）第二臂：8 面完整跑完，**仍是 5 死 / 0.625**

驱动 `.tmp_v400_v2f/round_v2g.sh`（19:51:02 启动，机器空闲：load 2.0、`th08-modern` 6 个），同 cap、同 seed、6 并发、独立 data-dir。**8/8 面 `exit=0`、8/8 SHA 一致、无 face 触到 7,200 s 超时**；整轮 2,848 s，单面最大 2,658 s。

| stage | 面 | cap | 过面 | **v2g frames** | 死亡 | stop | exit | 墙钟 s | v2f frames | v2a frames | 与 v2f |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | Stage3 | 30,014 | false | 22,912 | **1** | death | 0 | 1,686 | 23,172 | 25,052 | −260 |
| 3 | Stage4A | 41,583 | **true** | 29,036 | **0** | stage_changed | 0 | 2,600 | 29,564 | 30,952 | −528（仍过面） |
| 4 | Stage4B | 45,000 | false | 24,136 | **1** | death | 0 | 1,882 | 24,068 | 17,364 | +68 |
| 5 | Stage5 | 38,513 | false | **29,004** | **1** | death | 0 | 2,658 | 21,660 | 21,504 | **+7,344（cap 的 75.3 %）** |
| 6 | Stage6A | 45,000 | false | 15,132 | **1** | death | 0 | 1,380 | 19,584 | 13,820 | **−4,452** |
| 7 | Stage6B | 54,605 | false | 17,564 | **1** | death | 0 | 1,523 | 17,564 | 17,564 | **逐帧相同** |
| 0 | Stage1 | 19,887 | **true** | 13,484 | **0** | stage_changed | 0 | 643 | 13,484 | 13,484 | **逐帧相同** |
| 1 | Stage2 | 27,788 | **true** | 21,740 | **0** | stage_changed | 0 | 1,088 | 21,740 | 21,740 | **逐帧相同** |

- **8 面死亡合计 = 5，PSN = 0.625 → 仍未达标**；过面 3/8；帧数合计 **173,008**（v2f 170,836、v2a 161,480）。
- **加倍的顶部权重没有减少死亡，反而把死亡压得更靠底**：4/5 死在 `py` **411.3–424.0**（v2f 是 386.7–424.0），`top_pin` 0/5，`bomb_exhausted` + `wall_dense` 5/5；顶部罚分计入选中计划的比例 **0–2 %**（v2f 0–3 %）。
- **面级上是两次互相抵消的分叉**：Stage5 **29,004**（+7,344，全卡最大单面收益，走到 cap 的 75.3 %）vs Stage6A **15,132**（−4,452）；Stage3/4A 略差于 v2f、Stage4B 基本不变、Stage6B/Stage1/Stage2 逐帧相同。
- **结论**：3.0 不是 1.5 的剂量-反应升级，而是**另一条确定性轨迹**；两档都停在 5 死 → **"顶部/贴角罚分"这条轴上没有 ≤0.5 的解**。v2g 同样**不作为 G1 候选档**，也不触发 S1N。


## 9. 测试（`tests/test_v400_teacher.py`）

新增 **8 项**（既有 97 项保持全绿）：

| 测试 | 钉住什么 |
|---|---|
| `test_top_band_penalty_matches_the_published_formula` | 顶部连续项公式：`y=210/200` 为 0、`y=110` 为 0.75、`y=20` 为 1.5、`y=16` 为 `1.5·184/180`（与底部一样**不截断**）；单调；v2g=3.0；`full/v2a/v2b/v2c/v2d/v2e/v3a/v3b` 全部为 0 |
| `test_top_corner_surcharge_is_flat_and_only_inside_the_top_band` | 贴角项：`x<64 ∨ x>312` 且 `y<200` 时平罚 1.0；`x==64`/`x==312` 边界**严格不在角内**；`y≥200` 不生效；中场不生效；旧档为 0 |
| `test_v2f_profile_is_v2a_plus_the_top_band_and_keeps_high_penalty` | 档注册：H/beam/k/iterations 与 v2a 相同；底部 1.5/2.0 与全部共享项逐项相等；`high_penalty` 仍 3.0、`top=1.5`、`corner=1.0`；v2g `top=3.0`；无留弹/无 margin 门槛/无硬剪/12 帧窗口；`CONFIGS`/`PROFILES`/CLI/`full` 默认 |
| `test_top_band_stacks_with_high_penalty_on_the_living_score_only` | 与 `high` 的交互：顶角处 `high=1` 与 `top=1.5`、`corner=1.0` **同时**计费，总分比 v2a 恰低 2.5；`y=205` 处与 v2a **逐位相等**；死亡分支三项均为 0 |
| `test_top_band_flips_the_choice_from_the_top_corner_to_the_midfield` | 机制：3.5 px 的 margin 优势在 v2a 选顶角、在 v2f 被 1.5+1.0 的顶角代价翻转；存下的分数是真实分数；被拒计划分数更低（不是平局） |
| `test_v2f_search_matches_the_reference_on_a_top_heavy_tape` | 向量式 ≡ 标量式：自机起点放在顶部时 `beam_search` 与 `beam_search_reference` 的 `scores` 逐位相等，`score/top/corner/margin/end_xy/death_frame/bottom/high` 逐项相等，且**确实有计划被扣分**（`top>0` 与 `corner>0` 都出现） |
| `test_v2f_at_zero_top_weights_is_bit_identical_to_v2a` | `p_top=corner=0` 时 `replace(v2f.weights, …) == v2a.weights`（字段级相等），且同一 tape 上 `np.array_equal` 分数逐位相同 → **v2a 逐位不变** |
| `test_top_terms_round_trip_through_the_plan_and_the_label` | `Plan.as_dict` 带 `top`/`corner`，`TeacherLabel` 往返保留，**旧 JSONL（缺这两个键）仍能反序列化**为 0.0 |

| 时点 | 命令 | 结果 |
|---|---|---|
| v2f 8 面并发运行期间 | `python -m pytest … tests/test_v400_teacher.py --deselect …::test_raster_search_is_faster_than_the_exact_search_on_a_dense_tape` | **102 passed, 2 skipped, 1 deselected, exit=0**（105 项 = 97 既有 + 8 新增） |
| 8 面跑完、机器空闲（load 2.0） | 全量同命令（含 raster 性能项与 v3 性能项） | **103 passed, 2 skipped, 0 failed, exit=0**（105 项，含 raster 与 v3 两个性能项，4.47 s） |

## 10. 命令与退出码

| # | 命令 | 退出码 / 结果 |
|---|---|---|
| 1 | `python -m compileall -q touhou_rl tools/rl`（Windows） | **0** |
| 2 | `wsl bash .tmp_v400_v2f/smoke.sh`（400 帧、15 决策、stage 0、`--mode v2f`） | **SMOKE_V2F_EXIT=0**；JSONL 里 `mode=v2f`，`chosen_plan` 含 `top`/`corner`，`bomb_rule1` 含 `chosen_top`/`chosen_corner` |
| 3 | `wsl bash .tmp_v400_v2f/round.sh`（**第一次**，18:44:56，`--timeout 4200`） | **ROUND_EXIT=143**（主动中止）：并发 load 35–42 下长命面有 >70 min 截断风险；**0 个面完成、`runs.jsonl` 为空**，数据未使用 |
| 4 | `wsl bash .tmp_v400_v2f/round.sh`（**正式**，18:47:22，`--timeout 9000`） | **ROUND_EXIT=0**，`PSN_V2_DONE profile=v2f tag=round1 cases=8`，8/8 面 `exit=0`；整轮 3,456 s |
| 5 | `wsl /root/v400-venv/bin/python .tmp_v400_v2f/table.py round1-v2f round1-v2a` | **0**，写出 `.tmp_v400_v2f/table-round1-v2f.md`（§5 表） |
| 6 | `wsl /root/v400-venv/bin/python .tmp_v400_v2f/causes.py` | **0**，写出 §6/§7 的死亡点与代理死因表 |
| 7 | `wsl /root/v400-venv/bin/python .tmp_v400_v2f/divergence.py` | **0**，写出 §6.1 的首次分叉与顶部罚分触发率 |
| 8 | `wsl /root/v400-venv/bin/python .tmp_v400_v2f/records.py round1-v2f` | **0**，8 条 `runs.jsonl` 的全部字段（SHA、cap、frames、deaths） |
| 9 | `python -m pytest …`（并发期，排除 raster 性能项） | **0**，102 passed / 2 skipped / 1 deselected |
| 10 | `wsl bash .tmp_v400_v2f/round_v2g.sh`（第二臂，19:51:02 起） | **ROUND_EXIT=0**，`PSN_V2_DONE profile=v2g tag=round1 cases=8`，8/8 `exit=0`，整轮 2,848 s |
| 11 | `python -m pytest …`（空闲全量） | **0**，103 passed / 2 skipped / 0 failed（105 项，含两个性能项） |

**并发环境如实记录**：v2f 轮启动时 `pgrep -xc th08-modern`=**17**、`load average`=**35.3**（v3 扫描 6 教师 + D0 1 路 + 我的 6 教师），运行中峰值 **38–42**；v2g 轮在 **load 2.0** 的空闲机器上启动。两轮的 cap/seed/协议完全相同，因此帧数可比（v2e 卡 §10 已证 load 只影响墙钟）。

## 11. 结论与下一步

**结论：v2f 是一个"机制达成、门未过"的改动；v2g（3.0）复现了同一结论。**

1. **它做到了要求的事**：把"顶部"从二值 3.0 变成连续 `1.5·(200−y)/180` + 贴角 1.0（保留二值 `high`），并且 **v2a 的 2/2 顶角死亡全部消失**（Stage4B `py` 51.8→424.0、Stage6A 38.6→408.0），两个面分别 **+38.6 % / +41.7 %** 帧数；三个过面的行为不变（Stage1/Stage2 逐帧相同，Stage4A 仍过面且早 1,388 帧过）。`top_pin` 从 2/5 降到 **0/5**。
2. **但 PSN 仍是 0.625（5/8）**：死亡**换了边而不是消失** —— `bottom_pin` 从 1/5 升到 4/5，`bomb_exhausted`/`wall_dense` 仍是 5/5。底部带本来就有 3.5 分软罚（顶部满额 5.5），死亡仍发生在底部，说明**"边缘罚分强弱"不是当前约束**。
3. **v2g（p_top=3.0）完整 8 面：仍 5 死 / 0.625**，且死亡更靠底（4/5 在 `py` 411.3–424.0）；面级上 Stage5 **29,004**（+7,344）与 Stage6A **15,132**（−4,452）互相抵消。**两档之间不是剂量-反应**——这直接反驳"把顶部权重再加一档就会更好"。
4. **诚实口径**：+38.6 %/+41.7 % 是两次**确定性分叉**的结果（首次分叉都在 `y≈171–198` 的带边缘、罚分只有 0.01–0.24 分），不是剂量-反应；权重只在 `_finish_roots` 的末端打分里生效，**不能塑造 beam**，所以同一权重在 3 个面上逐帧不变、在 2 个面上 +40 %。
5. **S1N 未跑、G1 仍不判过**（协议门：PSN > 0.5）。

**下一步（按证据强度，不含本轮未验证的猜测）**：

1. **停止再加边缘项**：顶部已到 4.5–5.5 分（v2g 满额 7.5）、底部 3.5 分，而 4/5 死亡仍发生在场地底部（3–4 个在底部带内） —— 再加"底/顶"权重只会把死亡推到场地上部/中部，不会减少死亡数（本卡两档就是这条规律的直接实验）。
2. **真正的约束是"墙 + 没弹"**：5/5 死在 0 弹、末决策 48 帧致死盒均值 48–706。可测的下一步是 (a) 搜索预算（v3a/v3b 已在另一条工作流跑，本卡不重复）；(b) 与底部带配合的**逃逸空间**项（v2b 的 `escape_weight` 从未在"0 弹 + 底带"的死亡点上被单独检验）；(c) bomb 策略（12 帧窗口之外的放弹时机）。
3. **本卡的 `top`/`corner` 项可保留**（默认 0、纯新增、逐位可关），但它不是把 PSN 从 0.625 压到 ≤0.5 的杠杆；**v2f/v2g 都不应作为 G1 的候选档**。
