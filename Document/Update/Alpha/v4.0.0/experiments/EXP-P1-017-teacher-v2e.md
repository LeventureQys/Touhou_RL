# EXP-P1-017 — teacher v2e：Boss 本体代价 + 撤留弹（PSN 0.625 → ≤0.5）

- **日期**：2026-11-09
- **状态**：**完成（8 面完整轮，`partial=false`）→ PSN 未达标**：8 面死亡 **5 / 8 = 0.625**（目标 ≤4 死 / ≤0.5）。按协议门 **S1N 20 局未执行**（见 §7）。G1 **仍不判过**
- **上游**：[EXP-P1-012](EXP-P1-012-g1-evaluation.md)（G1 门：S1N 20/20、PSN 6/8 = 0.750 不达标）、[EXP-P1-015](EXP-P1-015-teacher-v2-bottom-band.md)（v1 → v2a：PSN 5/8 = 0.625、Stage4A 首次过面；v2c 硬剪被否、v2d Mixed）、[EXP-P1-016](EXP-P1-016-death-analysis.md)（v1 死亡分析：`bullet_wall` 6/6、`bomb_exhausted` 6/6、`bottom_pin` 5/6、`top_pin` 2/5、**`boss body` 5/6**、`laser` 1/6）
- **产物**：`touhou_rl/v400/teacher/config.py`（v2e 档）、`search.py`（Boss 本体代价项）、`label.py`（`Plan.boss` 反序列化）、`tools/rl/v400_teacher.py`（仅 `--mode` 帮助文本）、`tests/test_v400_teacher.py`（新增 10 项）、`.tmp_v400_v2e/`（驱动、8 面 JSONL、日志、`runs.jsonl`、表、复现实验）
- **一句话**：以 v2a 为基底（底部软罚 1.5/2.0 + 规则 1 的 12 帧窗口），撤掉"留 1 颗弹"与 margin 门槛，并给候选末端加 Boss 本体代价 `−3.0·clip(48−d,0,48)/48`，然后完整重测 8 面。**结论：该代价在 8 面上的 5 个死亡点全部不触发（死亡时到 Boss 本体 65–343 px，全部 > 48 px），6 面与 v2a 逐帧相同，PSN 仍是 0.625。**

---

## 1. 目标与口径

| 项 | 口径 |
|---|---|
| 目标 | **PSN 每面死亡 ≤ 0.5**（8 面口径），即 **8 面死亡合计 ≤ 4** |
| 判死依据 | **引擎 verdict 唯一**：`STATE.f6`（deaths）增量；到帧上限**不写成"过面"** |
| 过面 | `STATE.f1` 离开起跑面（`stage_clear=true`），否则记 `max_frames` 或 `death` |
| 帧上限 | EXP-P1-012 §2 的同一张表：Stage1 19,887 / Stage2 27,788 / Stage3 30,014 / Stage4A 41,583 / **Stage4B 45,000（fallback）** / Stage5 38,513 / **Stage6A 45,000（fallback）** / Stage6B 54,605 |
| 其它 | normal、seed 1、每面各自 `RESET`、无 `--follow-stage`、6 并发、每 worker 独立 data-dir |
| 完整轮 | 8 面各 1 局，**没有 partial**：8/8 面都有各自的真实 JSONL（§6） |

## 2. v2e 配置与理由

### 2.1 配置（逐项）

| 参数 | v1 (`full`) | v2a | **v2e** |
|---|---|---|---|
| horizon / beam / verify_k / max_iterations | 48 / 64 / 3 / 3 | 48 / 64 / 3 / 3 | **48 / 64 / 3 / 3** |
| `bottom_penalty` / `bottom_y` / `bottom_span` | 0 / 400 / 32 | 1.5 / 400 / 32 | **1.5 / 400 / 32（保留）** |
| `bottom_extra_penalty` / `bottom_extra_y` / `bottom_extra_span` | 0 / 420 / 12 | 2.0 / 420 / 12 | **2.0 / 420 / 12（保留）** |
| `escape_weight` | 0 | 0 | **0**（v2b 的逃脱加分未采用） |
| `high_penalty` | 3.0 | 3.0 | **3.0**（v2d 的 8.0 未采用） |
| `boss_penalty` / `boss_radius` | 0 / 48 | 0 / 48 | **3.0 / 48（新增）** |
| `bomb_rule1_max_death_frame` | `None` | 12 | **12（保留）** |
| `bomb_rule1_min_bombs_left` | 0 | 0 | **0（撤掉 v2c/v2d 的留弹）** |
| `bomb_rule1_max_margin` | `None` | `None` | **`None`（撤掉 v2c/v2d 的 margin 门槛）** |
| `clip_y` | `None` | `None` | **`None`（撤掉 v2c 的硬剪）** |
| `align_weight` / `edge_penalty` / `death_scale` / `margin_cap` | 4.0 / 1.0 / 10.0 / 16.0 | 同 | **同 v2a（逐项相等）** |

`v2e` 注册在 `config.CONFIGS`，`PROFILES = (V2A, V2B, V2C, V2D, V2E)`（本卡落地时的工作树快照；之后另一条并行工作流追加了 v3a/v3b，见 §10）；CLI `--mode v2e` 由 `choices=tuple(sorted(CONFIGS))` 自动暴露，默认仍是 `full`。

### 2.2 Boss 本体代价：为什么选 (b) 而不是 (a)

两个可选实现（任务书 §3）：**(a)** `align` 的目标点排除 Boss 本体；**(b)** 候选末端落在任一 Boss 本体 AABB 的 48 px 内 → 扣 `−3.0·clip(48−d,0,48)/48`。

**选 (b)**，理由三条：

1. **只惩罚"贴脸"，不改道中行为。** `align` 的目标是 `target_x(tape)`（最近的一个"引擎自己判定为致死"的敌人世界 x，无敌人时退化为自机 x）。方案 (a) 会把目标改成"某只小怪的 x"或"自机 x"，等于**同时改掉每一面（含已 0 死过面的 Stage1/2/4A）的排序基准**；而 (b) 是**纯新增项**，`boss_penalty=0` 时逐位等于原式，可独立开关。
2. **它测的正是被量到的量。** EXP-P1-016 的证据是"Stage6B 死在距 Boss 本体 102 px 处"，且 Boss 本体本身就是致死 AABB（死亡帧 458 个致死盒里含 body）；而 `margin` 项在 16 px 饱和，所以"离 100 px 的物体 40 px"与"离子弹 16 px"在旧评分里同分。
3. **a 与 b 不是同一件事。** `align` 是瞄准奖励（朝 Boss 打输出），(a) 会让 plan 在 Boss 战里失去对齐信号；死亡并不来自"没瞄准"，而来自"站得太近"。

实现上 `boss_body_boxes()` **复用** `geometry.enemy_boxes()` 的既有表达式（`worldPosition ± hitboxDimensions/3`），只把行过滤成 `flags1 & ENEMY_FLAG_BOSS`（`engine.py:46`，bit 1），因此"Boss 本体 AABB"和引擎判定致死的那个 AABB 是**同一个函数**算出来的。

### 2.3 撤留弹 / 撤 margin 门槛的证据

| 证据 | 数值 | 结论 |
|---|---|---|
| 同一面（Stage5）可放 bomb 3 颗时存活 | **21,504 帧** | 留弹的收益假设不成立 |
| 同一面 0 颗可放时存活 | **2,428 帧**（8.9× 差） | bomb 是存活资源，不是该囤的储备 |
| v2a 死亡点 | `bomb_exhausted` 5/5（"用光后死"） | 留 1 颗的规则只作用于**用光前**，救不了末段 |
| 规则 1 在 v1 的触发点 | 最短预测死亡 27–47 帧 | 12 帧窗口已足以抑制"为 48 帧预报浪费 bomb" |

故 v2e **不带** `bomb_rule1_min_bombs_left`（=0）与 `bomb_rule1_max_margin`（=`None`），只保留 12 帧窗口。

## 3. 实现（改了哪些行）

| 文件 | 改动 | 说明 |
|---|---|---|
| `touhou_rl/v400/teacher/config.py` | `Weights.boss_penalty`（默认 0.0）/`boss_radius`（48.0）+ `boss_penalty_at(distance)`；`_V2E_WEIGHTS`、`V2E`、`CONFIGS`、`PROFILES` | 标量口径唯一来源；`0.0` 时 v1/v2a 逐位不变 |
| `touhou_rl/v400/teacher/search.py` | `Plan.boss` 字段（`as_dict`）；`boss_body_boxes()` / `boss_clearance()` / `_boss_rows()`；`score_plan(..., boss_dist=)`；`_finish_roots(..., boss_boxes=)`；`beam_search` 与 `beam_search_reference` 各取"末端帧"的 Boss 本体并扣分 | 向量式与标量式**逐元素同一表达式**，两条搜索路径由测试钉住相等 |
| `touhou_rl/v400/teacher/label.py` | `Plan.from_dict` 读 `boss`（缺省 0.0） | 旧 JSONL 仍可反序列化 |
| `tools/rl/v400_teacher.py` | 仅 `--mode` 帮助文本 | 无参数/观测语义变化 |

评分式（v2e，living plan）：

```
score = 100 + min(margin,16) + 4·align + 2·min(items,1) − 1·edge − 3·high
        − [1.5·max(y−400,0)/32 + 2.0·max(y−420,0)/12]      # v2a 底部软罚（保留）
        − [3.0·clip(48 − d_boss, 0, 48)/48]                  # v2e Boss 本体代价（新增）
```

`d_boss` = 末端自机中心到"末端帧"最近 Boss 本体 AABB 的欧氏间隙（AABB 内为 0）；末端帧无 Boss（或 Boss 无几何导出）时该项恰为 0。代价只作用于 living plan（死亡分支仍是 `death_frame/H·10`）。

## 4. 评测协议（实际执行）

- 驱动：`.tmp_v400_v2e/round.sh` → 复用 `.tmp_v400_v2/psn_v2.py`（只加 `--mode`，**未改** `tools/rl/v400_g1_eval.py`）：`--mode v2e --tag round1 --jobs 6 --timeout 4200 --data-dir-root /root/v400-data/g1`。
- cap 由 `v400_g1_eval.psn_caps(load_human_reference())` 现算（= EXP-P1-012 表），每面 1 局、seed 1、normal、`--stop-on-death`、无 `--follow-stage`。
- 6 并发、每 worker 独立 data-dir（`/root/v400-data/g1/inst0..5`，`th08.cfg` 六份 md5 相同，`score.dat` 各自独立）。
- 每面一个 JSONL + 日志，完成即追加 `runs.jsonl`；表由 `.tmp_v400_v2e/table.py` 从 `runs.jsonl` 生成（`.tmp_v400_v2e/table-round1-v2e.md`）。
- 引擎 SHA 门通过（8/8 面同 `f172399f…264e`），8/8 面 `exit=0`。
- 判达标 = **8 面死亡合计 ≤ 4**（PSN = 合计/8 ≤ 0.5）。

## 5. 结果：v2e 8 面完整轮（含与 v2a 的逐面对比）

| stage | 面 | cap | 过面 | frames | 死亡 | stop | exit | bomb 已用 | Boss 证据 | 墙钟 s | **v2a frames** | **v2a 死亡** | 与 v2a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | Stage3 | 30,014 | false | **5,040** | **1** | death | 0 | 3.0 | present, life 11,557（max 16,000） | 354 | 25,052 | 1 | **变差 −20,012** |
| 3 | Stage4A | 41,583 | **true** | **30,952** | **0** | stage_changed | 0 | 3.0 | present, life 1（max 60,000） | 2,862 | 30,952 | 0 | **逐帧相同** |
| 4 | Stage4B | 45,000 | false | **17,364** | **1** | death | 0 | 3.0 | present, life 479（max 60,000） | 1,208 | 17,364 | 1 | **逐帧相同** |
| 5 | Stage5 | 38,513 | false | **21,504** | **1** | death | 0 | 3.0 | present, life 75（max 60,000） | 1,667 | 21,504 | 1 | **逐帧相同** |
| 6 | Stage6A | 45,000 | false | **14,112** | **1** | death | 0 | 3.0 | life→0（max 60,000） | 1,088 | 13,820 | 1 | **变好 +292** |
| 7 | Stage6B | 54,605 | false | **17,564** | **1** | death | 0 | 3.0 | life→0（max 60,000） | 1,314 | 17,564 | 1 | **逐帧相同** |
| 0 | Stage1 | 19,887 | **true** | **13,484** | **0** | stage_changed | 0 | 0.0 | life→0（max 60,000） | 578 | 13,484 | 0 | **逐帧相同** |
| 1 | Stage2 | 27,788 | **true** | **21,740** | **0** | stage_changed | 0 | 0.0 | life→0（max 19,000） | 1,163 | 21,740 | 0 | **逐帧相同** |

- **8 面死亡合计 = 5，均值 = 0.625 → PSN = 0.625（目标 ≤0.5）→ 未达标**；过面 3/8；测满 8/8 面。
- 帧数合计 **141,760**（v2a 同轮 161,480，**−19,720**）。
- **与 v2a 的差异只有两处**：Stage3 从 25,052 → 5,040（更差），Stage6A 从 13,820 → 14,112（+292，仍死）。其余 **6 面与 v2a 完全逐帧相同**（`frames` 连个位数都一样：30,952 / 17,364 / 21,504 / 17,564 / 13,484 / 21,740）。
- **bomb 统计**：6 面 `bombs_spent = 3.0`（3 颗全用光，其中 Stage4A 也用了 3 颗但**过面**）；Stage1/Stage2 为 0.0。`bomb_decisions`（决策行里 `bomb=1` 的计数，含同一次放弹的连续决策）为 Stage3 2 / Stage4A 2 / Stage4B 2 / Stage5 3 / Stage6A 3 / Stage6B 20 / Stage1 0 / Stage2 0，**实际消耗以 `bombs_spent` 为准**；Stage6B 的 20 说明"放弹标签连续保持了多次决策"，不是 20 次放弹。
- Stage4A 过面时**最后观测到的** Boss life 只有 **1**（`boss_analysis.boss_life_last`）；这条只是观测，不声称因果。

## 6. 为什么没达标：Boss 本体代价在 5 个死亡点上**全部不触发**

这是本卡最重要的结论，证据链三层：

1. **触发条件**：只有末端距 Boss 本体 AABB **< 48 px** 才扣分（`clip(48−d,0,48)/48`）。实测 5 个死亡点到 Boss 本体最近距离是 **65 / 124 / 218 / 246 / 343 px**（§8 的表），**全部 > 48 px**；选中的计划里 `boss` 项在 1,260 个 Stage3 决策上 **0/1,260 非零**。
2. **因此 6 面逐帧相同**：Boss 项没进过任何一个被选中的计划，搜索排序与 v2a 完全一致，`frames` 逐个相同（stage 0/1/3/4/5/7）。**唯一被改动的是 Stage3**：该面在决策 758 处 `geo_score` 出现 **−0.06 … −0.59** 的差异（另有 `±5.0` 的"未验证折扣"翻转），并把该决策的"已验证集合/选中计划"换掉（首次分歧点：v2a 选 1 号、v2e 选 12 号），随后轨迹分叉，最终 5,040 帧死于左上角（`py=21.7`），而不是 v2a 的 25,052 帧死于底部。
3. **该分歧可复现，不是噪声**：把 Stage3 用 v2e 重跑一次（`.tmp_v400_v2e/repro.sh`），结果**逐位复现**：5,040 帧、`end=(29.657, 21.657)`、bombs 0/3 —— 同一个死亡点、同一个位置。→ 引擎对相同输入是确定的，Stage3 的 25,052 → 5,040 **是配置效应，不是运行抖动**（对照：v2a 的复现跑因 `SPAWN … Broken pipe` 引擎崩溃 `exit=5`，见 §10）。

**根因**：任务书给出的是 EXP-P1-016 在 **v1** 上量到的死亡（Stage6B 距 Boss 本体 102 px、`py=84.5`）。但 v2a 已经把死亡点从 Boss 身边挪开了 —— v2a 的 Stage6B 死亡点是 `py≈346`、距 Boss **218 px** 的弹幕墙，不是 102 px 的贴脸；48 px 的半径对 v2e 自己的 5 个死亡点一个都够不到。同时 v2e 对 Stage3 的那次分叉把死亡从"底部"换成了"左上角"（`top_pin`），这正是 EXP-P1-016 已经点出的第二个死法，v2e 没有任何一项在治它。

## 7. S1N 20 局重判：**未执行（协议门未过）**

协议规定"若 PSN 达标 → 立刻用同一档重跑 S1N 20 局"。本轮 **PSN = 0.625 > 0.5（5 > 4 死）**，门未过，因此：

- S1N **没跑**，没有 20 局数据可报；**G1 仍不判过**（G1 需要 S1N ≥18/20 且 PSN ≤0.5 同时成立）。
- 不把"没跑"写成"通过"，也不把 v2a 的历史 S1N 结果（EXP-P1-012 的 20/20）当成本档的结果。
- 若强行跑 S1N，本档在 Stage1（8 面里的 stage0）13,484 帧 0 死过面，与 v2a 逐帧相同，但那**不能**替代 PSN 门。

## 8. 剩余死因（JSONL 代理判读，非 EXP-P1-016 的逐帧分类器）

EXP-P1-016 的分类器要重放 180 帧像素（引擎 + xvfb），本轮没跑；下表只用 `v400_teacher.py --out` 的 JSONL 做**代理标签**，口径写在 `.tmp_v400_v2e/causes.py` 的 docstring 里（`bottom_pin`=`py>400`、`top_pin`=`py<100`、`corner`=贴 y 边且 `px` 距左右边界 <48、`bomb_exhausted`=`bombs==0`、`boss_body`=距 Boss <48 px、`boss_pattern`=距 Boss <448 px、`wall_dense`=该决策 48 帧 tape 的致死盒均值 ≥40）。

| 面 | frames | 死亡 | 末帧 py | 末 30 决策均值 py | 余弹 | 离 Boss | tape 盒均值 | 代理标签 |
|---|---|---|---|---|---|---|---|---|
| Stage3 | 5,040 | 1 | **21.7** | 40.2 | 0 | 65 | 234 | `top_pin` + `corner` + `bomb_exhausted` + `boss_pattern` + `wall_dense` |
| Stage4B | 17,364 | 1 | **51.8** | 28.4 | 0 | 246 | 329 | `top_pin` + `corner` + `bomb_exhausted` + `boss_pattern` + `wall_dense` |
| Stage5 | 21,504 | 1 | 392.2 | 397.8 | 0 | 343 | **720** | `bomb_exhausted` + `boss_pattern` + `wall_dense` |
| Stage6A | 14,112 | 1 | **27.3** | 26.3 | 0 | 124 | 619 | `top_pin` + `corner` + `bomb_exhausted` + `boss_pattern` + `wall_dense` |
| Stage6B | 17,564 | 1 | 345.6 | 348.0 | 0 | 218 | 48 | `bomb_exhausted` + `boss_pattern` + `wall_dense` |

**读法**：

1. **5/5 死时 0 弹**（`bomb_exhausted`）：与 EXP-P1-016 一致，"用光后死"没有因为撤掉留弹而改变 —— 因为 v2a 本来就没留弹，撤掉的是 v2c/v2d 的规则，对 v2e 无影响。
2. **3/5 死在顶角**（Stage3 `py=21.7`、Stage4B `py=51.8`、Stage6A `py=27.3`，且 `px` 距边 <56）：`top_pin` 已是**主导死法**，而 v2e 里没有任何一项在治它（`high_penalty` 只是"在 y<200 时扣 3.0"的常数，对"顶角"和"场地中部上方"一视同仁；`edge_penalty` 只 1.0）。
3. **2/5 死在场地中部**（Stage5 `py≈392` + 盒均值 720、Stage6B `py≈346`）：密集弹幕墙，属搜索/规避能力问题（Stage5 只走到 cap 38,513 的 56%）。
4. **0/5 是 `boss_body`**：死亡点到 Boss 本体 65–343 px，48 px 半径够不到 —— 与 §6 的结论互相印证。

## 9. 测试

```
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_v2e_pytest_idle2 -q tests/test_v400_teacher.py
```

| 时点 | 结果 |
|---|---|
| 实验期间（并发跑 8 面，排除 `test_raster_search_is_faster_than_the_exact_search_on_a_dense_tape`） | **91 passed, 2 skipped, 1 deselected, exit=0**（当时 94 项 = 既有 84 + 本卡新增 10） |
| 空闲时（全量，含 raster 性能项） | **95 passed, 2 skipped, 0 failed, exit=0（97 项）** |

- 97 = 84（既有）+ 10（本卡）+ 3（同一工作树里**并行**进行的 EXP-P1-019 `v3a/v3b` 工作流新增；与本卡无关，但也在绿）。
- 本卡新增 10 项：Boss 代价的饱和/边界（48 px 处 0、24 px 处 1.5、体内 3.0）、AABB 间隙（标量 vs 向量逐元素相等）、只取 Boss 行（小怪/Boss 无几何都不算）、只作用于 living plan、v2e 档注册（撤留弹 + 撤 margin + 保留 12 帧窗口 + 保留底部软罚 + 无硬剪 + CLI 暴露）、末端选择被 Boss 代价翻转（v2a 选"贴脸但 margin 高 5"的端点、v2e 选"远 80 px 但 margin 低 1 + 底部罚"的端点）、所有末端都贴脸时仍给 living plan、`beam_search` 与 `beam_search_reference` 在带 Boss 的 tape 上逐位相等、v2e 下 `bombs=1` 时规则 1 重新会放弹（v2c 不会）、12 帧窗口行为不变。
- `test_raster_search_is_faster_than_the_exact_search_on_a_dense_tape` 在并发时会偶发，因此**实验期间排除、空闲时单独验证通过**（含在上面的 95 passed 里）。

## 10. 命令与退出码

| # | 命令 | 退出码 / 结果 |
|---|---|---|
| 1 | `python -m compileall -q touhou_rl tools/rl`（Windows） | **0** |
| 2 | `wsl bash .tmp_v400_v2e/smoke.sh`（v2e 400 帧冒烟：15 决策、stage 0、`--mode v2e`） | **SMOKE_V2E_EXIT=0**，JSONL 里 `mode=v2e`、`chosen_plan.boss` 字段存在 |
| 3 | `wsl bash .tmp_v400_v2e/round.sh`（8 面完整轮，6 并发） | **ROUND_EXIT=0**，`PSN_V2_DONE profile=v2e tag=round1 cases=8`，8/8 面 `exit=0` |
| 4 | `wsl /root/v400-venv/bin/python .tmp_v400_v2e/table.py round1-v2e round1-v2a` | **0**，写出 `.tmp_v400_v2e/table-round1-v2e.md`（§5 的表） |
| 5 | `wsl /root/v400-venv/bin/python .tmp_v400_v2e/causes.py` | **0**，写出 §8 的代理死因表 |
| 6 | `wsl bash .tmp_v400_v2e/repro.sh`（Stage3 × {v2a, v2e} 复现） | v2e **REPRO_v2e_EXIT=0**（逐位复现 5,040 帧）；v2a **REPRO_v2a_EXIT=5**（`write failed on 'SPAWN 3736': [Errno 32] Broken pipe`，引擎崩溃，未取得基线复现） |
| 7 | `python -m pytest … tests/test_v400_teacher.py`（实验期间，排除 raster 性能项） | **0**，91 passed / 2 skipped / 1 deselected |
| 8 | `python -m pytest … tests/test_v400_teacher.py`（空闲全量） | **0**，95 passed / 2 skipped |

**并发环境说明（如实记录）**：

- 本轮 6 并发教师跑在 `--jobs 6`（Engine 并发 = 6 teacher × 1 主引擎 + 1 个 <1 s 的 SPAWN 录制子进程），data-dir 为 `/root/v400-data/g1/inst0..5`。
- 同一工作树里另有一条**并行工作流**（EXP-P1-019 的 `v3a`/`v3b`）在 18:04 之后向 `config.py` 追加了两档、18:11 向 `tests/test_v400_teacher.py` 追加了 3 项测试，并在 18:26 之后自己启动了 v3a/v3b 的 8 面轮（`/root/v400-data/v3*`）。本轮的 8 个教师进程**全部在 18:04 之前启动**（最后一个 stage0 于 17:55 启动），因此 8 面结果用的都是本卡描述的 v2e 代码；`search.py`/`label.py` 自 17:34 起未被改动。
- 该并行负载把 Stage4A 的墙钟从 v2a 的 2,538 s 抬到 2,862 s（+13%），但**帧数逐个相同**，说明负载只影响时间不影响结果；v2a 的复现跑崩溃（#6）也是在 13 个 th08-modern 同时运行的窗口里发生的。

## 11. 结论与下一步（按证据强度排序）

**结论**：v2e 是一个**结构上正确但打偏了**的改动 ——

1. 它把"Boss 本体代价"做成可独立开关的纯新增项（`boss_penalty=0` 时逐位等于 v1/v2a），实现与口径都对，测试也钉住了；但 **48 px 半径覆盖不到 v2a 实际死亡点到 Boss 的距离（65–343 px）**，所以在 6 面上逐帧等于 v2a，只在 Stage3 造成一次分叉（且变差）。
2. **PSN 仍是 0.625（5/8），没有从 0.625 压到 ≤0.5**；S1N 未跑，G1 不判过。
3. 撤掉留弹/margin 门槛**在本轮不可观测**：v2a 本来就没有这两条规则，所以"撤掉"不产生行为差异（v2e ≡ v2a 的 6 面即为证据）。要检验"留弹有害"，只能用 v2c/v2d 与 v2a/v2e 对比，而那两组已有数据（v2c 明显更差、v2d Mixed）已经给出方向。

**下一步（按证据强度）**：

1. **顶角项（最强证据）**：3/5 死亡在顶角（`py` 21.7 / 51.8 / 27.3）。现有 `high_penalty` 是 y<200 的二值常数 3.0，且 `edge_penalty` 只 1.0，两者叠加（4.0）抵不过一次 margin 差的 16 分。应把"贴顶/贴角"做成**连续**的、与底部软罚同构的项（例如 `top_penalty·max(top_y−y,0)/span` + `corner` 项），而不是继续加 Boss 半径。
2. **Boss 半径要按实测重定**：若仍要保留 Boss 项，半径必须 ≥ 实测死亡距离的量级（65–343 px 的中位约 218 px），48 px 只覆盖"已经在体内"的情形，等价于没有。
3. **Stage5（盒均值 720、只走到 cap 56%）是搜索能力问题**，与任务书 §4 的"`horizon 48→64` / `beam 64→96`"假设一致；但按本轮实测的墙钟（Stage5 单面 1,667 s），horizon 64 的完整 8 面轮需要 ≈1.4× 时间（约 4,000 s），单轮预算内只能跑一次，**可作 v2f 的唯一一次尝试**，且应当先只在 Stage5 + Stage6B 上做时间预算估算。
4. **`laser` 仍是规划盲区**（`laser_hits` 缺 `startTime`/`duration`，且不许改引擎）：Stage6A 的死亡（`py=27.3`、盒均值 619）不能排除激光，只能等引擎侧导出。
