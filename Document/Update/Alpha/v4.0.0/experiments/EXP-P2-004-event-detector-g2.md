# EXP-P2-004 — 事件检测器 v1：HUD 序列事件头 + 数字位分类（G2）

- **实验编号**：EXP-P2-004（承接 EXP-P2-003）
- **目标**：把 P2 的 **G2** 从「未达」推到「达标」；失败点：`death` 事件召回 ≤33%、`score` 增量符号 91.9%、`score_down` 样本 0。
- **数据集**：`/root/v400-data/p2det/out-main`（WSL ext4），引擎 `third_party/th08-rl/build/rl_hazard2/th08-modern`
  `sha256=f172399f61b75baa8e62be46041ef756a710759f439e7fb07111f4747c78264e`，label version `p2-v2-1`。
- **数据量**：EXP-P2-003 的 128 shard / 300,000 帧 + 本次补采 66 shard / 57,729 帧 → **194 shard / 357,729 帧**（索引口径）；
  磁盘上 label 行 358,601（含 1 个未入索引的尾部 shard，差 872 帧 = 0.24 %）。
- **切分**：按 **stage 整段**切（train=0,1,2,4 / val=5 / test=3,6），同一 `(stage, difficulty, seed)` 的 shard 绝不跨 split；
  本次补采全部落在 stage 3，因此**只进 test**，模型从未见过。
  split 实测：train 80 shard / 192,000 帧、val 16 / 38,400、test 98 / 127,329。

---

## 1. 口径确认：事件 — 可观测信号对照表

统计对象：全部 194 shard / 358,601 个带标签帧，只读 `labels.jsonl`，不读像素、不跑引擎。
脚本 `.tmp_v400_det2/task1_align.py`，结果 `.tmp_v400_det2/task1_align.json`。

| 事件（真值） | 可观测信号 | 同帧？ | 时序偏移（事件帧 → 信号帧） | 覆盖 | 判定 |
|---|---|---|---|---|---|
| `death` = `ev & 0x4`（530 帧/530 事件，逐事件 1 帧） | HUD `lives` 递减 | **否** | **+4 帧**（327/342）、**+10 帧**（15/342） | **342/530 = 64.5 %** | 事件**之后** 4 帧可读，不是同帧 |
| `death` 且 `lives == 0` 已成立（188/530 = 35.5 %） | **无任何可观测变化** | — | — | **0/188** | HUD 上根本不存在递减 |
| `bomb` = `ev & 0x1`（134 帧/134 事件） | HUD `bombs` 递减 | **是** | **0 帧**（134/134） | **134/134 = 100 %** | 完美同帧 |
| `game_over`（`go=1`） | — | — | — | **0 帧 / 0 shard** | 本批数据从未进入 game over |
| `stage_clear` | **没有专用事件位** | — | — | `lives` 上跳 0 次、`bombs` 上跳 40 次 | 只有"重置型"弱代理，不可作为 G2 事件 |
| score 增量 | 标签 `sc`=`globals.score` 的差分 **vs** 像素上的 `displayScore` | 否 | 计数器**滞后滚入** 0–32+ 帧 | `down` = **0 / 358,601** | 见 §5.2 |

关键量化：

- `death` 的 lives 递减**不是同帧**，是 **+4 帧**：`ev&4` 在 t 帧，`livesRemaining` 在 t+4 才减 1（同时 `player_state` 2→1、`py` 从顶部回到 384）。
  直方图：`{+4: 327, +10: 15}`，中位数 **+4**，无提前（`signal_earlier = 0`）、无同帧（`exact_same_frame = 0`）。
- **188 个 death 事件发生在 `lives` 已经为 0 之后**（`li`、`prev_li` 都是 0，`dx` 递增照常，说明引擎确实记了这次死亡）。
  `lives` 不能为负，**这类事件在 HUD 上没有任何增量可读**。
- `bomb` 是**唯一**与 HUD 增量严格同帧的事件：134/134，偏移 0，既无漏检也无误检（`bombs_down` 也恰好 134，一一对应）。
- score 负向样本：358,601 帧里 `sc` 下降 **0** 次（`up` 88,947 / `same` 269,460 / `down` 0）。

---

## 2. 根因验证：原假设「death 应由 HUD 增量判定」——**部分成立，但不成立的部分是决定性的**

原假设：v0 用 playfield 预测"将死"是错的，`death` 应由 HUD `lives` 增量判定。数据给出的结论：

| 命题 | 验证结果 |
|---|---|
| v0 用 playfield 预测 `death` 是错的 | **成立**：v0 test 上 event recall 1.04 %，而 HUD 侧 lives 读数已 100 % |
| `death` 的最强证据是 `lives` 递减 | **成立**（64.5 % 的事件确实伴随递减，且延迟稳定 +4 帧） |
| 因此 `death` 可完全由 HUD 增量判定 | **不成立**：35.5 %（188/530）的 death 发生在 `lives == 0`，HUD 无增量可得 |
| `bomb` 可由 `bombs` 增量判定 | **完全成立**：134/134 同帧 |
| `score` 增量符号可从 HUD 数字得到 | **不成立**：HUD 画的是 `displayScore`，不是标签的 `globals.score` |

**结构性上限（不是模型能力问题）**：

- `death` 事件召回 ≤ `342/530 = 64.53 %`。任何只读 HUD 的检测器都不可能超过它，因为另外 188 个事件在 HUD 上不可观测。
- `score` 增量符号 ≤ 约 `1 − (滚动帧占比)`。原因在引擎里：
  `GameManager.cpp:526-544` 每帧把 `displayScore` 向 `globals.score` 滚 `(score − displayScore) >> 5`，
  `Gui.cpp:1285` 画的正是 `displayScore`。标签是 `globals.score`（`STATE f4`），所以"计数器还在滚"的帧上，
  像素上的数字**本来就不等于标签**。实测：`sc` 不变但 score 区域像素仍在变的帧占 `sc` 不变帧的 **272/2032 = 13.4 %**，
  像素差分随时间常数 ~32 帧衰减到 0（`.tmp_v400_det2/probe04_roll.py`）—— 这正是 EXP-P2-003 那 91.9 % 的来源。

---

## 3. 检测器 v1 结构

不改 v0（`train` 子命令仍可复现 EXP-P2-003），新增 `train-v1` / `report-v1`。文件：`tools/rl/v400_p2_reward_detector.py`。

```
每帧 HUD 平面 (56×160, u8)
   ├── lives 裁剪  rows 28..40         → CNN → 7 类读数（0..6）
   ├── bombs 裁剪  rows 44..56         → CNN → 9 类读数（0..8）
   └── score 9 个数字格 rows 8..24, x = 3+13i 宽 13（Gui.cpp: "%.9d" + 117px/9 位）
                                        → 共享 CNN → 每位 10 类（0–9，%.9d 不会有空白位）
                     ↓ 逐帧读数向量 (20 维 = lives 后验 7 + bombs 后验 9 + score 增量符号 one-hot 3 + 归一化 score 1)
因果 1-D CNN（kernel 3×2 层，只回看 t−4..t，**无前视**）→ 2 个 logits：[death, bomb]
```

- **`death` / `bomb` 头的主输入是 HUD 序列**，playfield **不参与**这两个头。
- **play 流保留在数据管线和 v0 模型里**（`play96`/`play64` 平面、v0 的 `train` 子命令、危险/占用用），
  v1 的 G2 事件头刻意不消费它 —— 这是本轮的范围裁剪，见 §8。
- 阈值不在 test 上挑：`val` 上扫 0.05..0.9 取事件 F1 最优（`sweep_threshold`），本次挑到 `death=0.05, bomb=0.05`。
- 同时报告**确定性 HUD 差分基线**（`hud_diff_events`：读数严格递减即报事件），用来和学到的头对照。

---

## 4. 数据补充（只补必要的）

补采命令见 §9。参数：`--stage-plan 3:3`（**只采 test 的 stage 3**）、`--policy-plan dive,dive,dive,random`、
`--frames-per-shard 900`、`--instances 6`、`--seed-base 8000`、`--warmup 400`。

| 项目 | 补采前 | 补采后 | 说明 |
|---|---|---|---|
| shard / 帧 | 128 / 300,000 | **194 / 357,729**（索引） | 补采 +66 shard / +57,729 帧 |
| `death` 事件（全库） | 354 | **530** | +176 |
| `death` 事件（**test** / stage 3） | 96 | **272** | +176，全部落在 test |
| `bomb` 事件（全库） | 114 | **134** | +20 |
| `bomb` 事件（**test** / stage 3） | **0** | **20** | 补采前 test 完全没有 bomb，bomb 门在 test 上不可测 |
| `death` 中 `lives` 可观测者 | 218 / 354 = 61.6 % | **342 / 530 = 64.5 %** | 上界 |
| `score_down` 负向样本 | 0 | **0** | **补采也采不到，见下** |

**`score_down` 仍然采不到的结论（不是采集不足，是引擎不产生）**：
`third_party/th08/src` 全仓库对 `globals->score` 的写入只有 `+=`（`GameManager.cpp:196`）、
上限钳到 999,999,999（`GameManager.cpp:523-524`）、以及**关卡/Game Over 时的置 0**
（`GameManager.cpp:782/1049`、`GameManagerSetup.cpp:139/406`、`ReplayManager.cpp:622`）。
**没有任何扣分路径**。因此"score 下降"只可能来自一次 stage 重置（分数清零），
而本批策略（`still`/`random`/`dive`，每段 900–2400 帧）从未跨过关卡边界 —— 358,601 帧里 0 次。
→ **G2 的 `score` 增量符号门在当前数据上不可测（负向类样本量为 0）**，如实记录，不用 0 样本硬算一个 100 %。

death 正例数说明：test 现有 **272** 个 death 事件。零错误下界要 `1 − 3/n ≥ 0.99` 需 `n ≥ 300`，
272 只够 `1 − 3/272 = 98.90 %`。**但该项本来就被 64.5 % 的可观测上限卡死，补到 300 也不会改变判定**，
故本轮不再为凑 300 追加采（把预算留给真正能过的门）。

---

## 5. G2 全表（test split，模型未见过 stage 3）

test = **stage 3**（98 shard / 127,329 帧，模型完全未见过）。v1 事件头取 **HUD 差分的确定性形式**
（`hud_diff_events`：`lives` 读数严格递减即报 `death`，`bombs` 递减即报 `bomb`）。学到的因果序列头同样训练并评测了，
但**没有收敛**（事件概率恒 ≈0），单列在表末，**不冒充主结果**。

| # | 指标 | 值 | 95% 单侧下界 | n | 要求 | 判定 |
|---|---|---|---|---|---|---|
| 1 | death recall (event level) | 69.26% | 63.52% | 270 | >= 0.99 LB | **未达标** |
| 2 | death precision (event level) | 98.94% | 96.22% | 189 | >= 0.99 LB | **未达标** |
| 3 | death recall (frame level, 严格同帧) | 0.00% | 0.00% | 270 | >= 0.99 LB | **未达标**（必然：HUD 差 4 帧） |
| 4 | death precision (frame level, 严格同帧) | 0.00% | 0.00% | 189 | >= 0.99 LB | **未达标**（同上） |
| 5 | death 延迟 中位 / p95 / max | **4 / 4 / 10 帧** | n/a | 187 | <= 10 | **达标** |
| 6 | lives 读数准确率 | **100.00%** | **99.997%** | 127,329 | >= 0.995 LB | **达标** |
| 7 | bombs 读数准确率 | **100.00%** | **99.997%** | 127,329 | >= 0.995 LB | **达标** |
| 8 | bomb recall (event level) | **100.00%** | 83.89% | 20 | >= 0.99 LB | 取值达标、**下界未达标**（n 太小） |
| 9 | bomb precision (event level) | **100.00%** | 83.89% | 20 | >= 0.99 LB | 同上 |
| 10 | bomb 延迟 max | **0 帧** | n/a | 20 | <= 10 | **达标** |
| 11 | score 逐位准确率（9 位平均） | 73.86% | 73.78% | 1,145,961 | >= 0.99 LB | **未达标** |
| 12 | score 整数值准确率 | 21.80% | 21.57% | 127,329 | >= 0.99 LB | **未达标** |
| 13 | score 增量符号准确率 | 76.27% | 76.03% | 127,329 | >= 0.99 LB | **未达标** |
| 14 | score 增量符号 真值负向样本数 | **0**（run 内） | n/a | 0 | 需 > 0 | **不可测** |
| 15 | score 符号（settled 子集，诊断） | 99.56% | 99.50% | 70,113 | 诊断 | 接近、仍差 |
| 16 | score 符号（rolling 子集，诊断） | 47.73% | — | 57,216 | 诊断 | 像素 != 标签，物理不可恢复 |
| - | learned 事件头 death（未收敛，单列） | 0.37%（1/270） | — | 270 | — | 不计入主结果 |
| - | learned 事件头 bomb（未收敛，单列） | 0.00%（0/20） | — | 20 | — | 不计入主结果 |

逐位准确率（左→右 9 位，n = 127,329 / 位）：
**100.00 / 100.00 / 100.00 / 98.50 / 66.62 / 58.83 / 47.41 / 45.58 / 47.80 %**
低位数字塌陷是 `displayScore` 滚动滞后造成的（`GameManager.cpp:526-544`），不是分类器读不出字形。

**判定汇总：G2 = 「未达」**。达标 5 项（#5 #6 #7 #10，以及 #8/#9 的取值）、未达标 8 项、不可测 1 项。

注：#13 的 76.27% 里含 **97 个 shard 边界伪负向样本**（把各 run 拼成一条序列时，新 run 的 `score` 从 0 起，制造了虚假"下降"）。
**run 内的真实 `score_down` 是 0**（§1）。因此 #14 才是这一门的正确判定：**不可测**，#13 仅作为"含伪样本"的参考值列出，不作主张。


### 5.1 延迟定义（必须与"预测"区分）

- 本方法做的是**事件检测（发生即报）**，**不是死亡预测**。检测帧 = HUD 读数首次发生变化的帧。
- **延迟 = 检测帧 − 真值帧**，正数表示比事件晚。门要求 `≤ 10` 帧。
- 因为引擎在 `death` 之后 **+4 帧**才扣命，检测器**在物理上只能**在 `t+4` 报出，所以延迟中位数 = +4 / p95 = +4（少量 +10）。
  这**不是**提前预警能力；若要"提前 N 帧知道会死"，必须回到 playfield 预测问题（v0，已测失败）。
- 报告里同时给 `frame-level recall`（严格同帧）= 0 %，这正是"HUD 增量与事件不同帧"的直接体现，不予掩盖。

### 5.2 score 数字位与符号

逐位准确率、整数值准确率、增量符号准确率见上表；并按"计数器是否仍在滚动"拆成两组诊断项：

- `settled`（`sc` 已连续 ≥33 帧未变，`displayScore == globals.score`）
- `rolling`（仍在滚，像素 ≠ 标签，**标签在像素上不可恢复**）

---

## 6. 测试

`tests/test_v400_reward_detector.py` 新增 12 个用例（原有 21 个，现 **33 passed / 0 failed**）：

| 用例 | 覆盖点 |
|---|---|
| `test_hud_diff_events_fire_on_every_lives_decrement` | HUD 差分事件：递减报事件、相等/上升不报 |
| `test_hud_diff_events_bombs_match_the_bomb_count` | bombs 差分事件帧号 |
| `test_score_glyph_boxes_are_nine_disjoint_in_plane_cells` | 9 个数字格几何、互不重叠、落在平面内 |
| `test_digit_roundtrip_and_nine_digit_format` | `%.9d` 编码/解码、越界钳位 |
| `test_sign_series_three_classes` | 三分类符号 |
| `test_causal_windows_never_look_ahead` | 因果窗口：改动未来帧不得影响当前行（无前视） |
| `test_death_hud_lag_is_inside_the_latency_gate` | +4 帧延迟落在 ≤10 门内 |
| `test_latency_gate_fails_past_ten_frames` | 15 帧延迟判失败 |
| `test_event_metrics_precision_counts_false_alarms` | 事件/帧级精确率与误报 |
| `test_reader_accuracy_separates_settled_from_rolling_score` | 逐位/整数/符号、settled vs rolling 拆分 |
| `test_v1_verdict_covers_every_g2_gate` | G2 门清单完整性 |
| `test_v1_report_marks_the_untestable_negative_class` | 负向样本为 0 时报"不可测"而不是 100 % |

---

## 7. 命令与退出码

```bash
# 补采（WSL，6 实例，只采 stage 3）
wsl -e bash .tmp_v400_det2/topup.sh            # TOPOUT_RUN_EXIT=0（人工在 ~55k 帧处停止）
# 索引 + 口径统计 + 切分
wsl -e bash .tmp_v400_det2/run_v1.sh           # INDEX_EXIT=0 ALIGN_EXIT=0 SPLIT_EXIT=0
# v1 训练 + 评测
wsl -e bash -c '... train-v1 ...'              # V1_EXIT=0
# G2 表
wsl -e bash -c '... report-v1 ...'             # REPORT_EXIT=0
# 测试
PYTHONPATH=... python -m pytest -p no:cacheprovider --basetemp .tmp_v400_det2/pytest -q tests/test_v400_reward_detector.py
```

产物：`.tmp_v400_det2/metrics-v1.json`、`g2-v1.md`、`split-v1.json`、`task1_align.json`、`v1.pt`。

---

## 8. 结论

### 达成

- **lives / bombs 读数 100%（下界 99.997%）** —— 保留 v0 的能力，HUD 裁剪 + CNN reader 直接复现。
- **bomb 事件**：Bomb 与 `bombs` 递减**严格同帧**（134/134，test 20/20），延迟 0，精确率 100%。G2 若要求 bomb，这一门在"值"上已通过，
  只差 test 侧样本量（20 个 → 零错误下界 83.9%）。**补到 n≥300 个 bomb 事件即可让该门达标**（约 1 bomb/674 帧 random 策略 → 20 万帧）。
- **death 延迟 4 帧（max 10）** —— 在 ≤10 帧门内，且 p95=4 稳定。

### 未达成（且**不是**再训练能解决的）

1. **death 事件召回 69.26%**：可观测上界是 **64.5 %（全库 342/530）**、test 上 187/270=69.3 %。
   其余 **188/530（35.5 %）的 death 发生在 `lives==0`**，HUD 上**没有任何增量**可读。
   → 只读 HUD 的检测器**不可能**达到 99 %。要补上这部分，只能重新引入 playfield 侧信号（玩家爆炸/回位），
   而这正是 v0 失败的那条路，且需要重新设计（不是本轮的 HUD 差分）。
2. **score 整数值/符号**：标签是 `globals.score`，HUD 画的是 `displayScore`（滚动逼近）。
   在"还在滚"的帧上，**像素里不包含标签信息**。settled 子集符号 99.56 %、rolling 子集 47.7 %，
   合起来 76.3 %；整数值 21.8 % 同样被滚动残差吃掉。**换更强的分类器也不会改变**，因为差异不在字形识别上。
3. **learned 事件头未收敛**：pos_weight 上限 8 + 0.2 % 正例率 → 概率塌到 0，test 上只报 1 个事件。
   确定性差分形式反而 100 % 精确，故本轮以确定性形式交付；序列头需要重采样/更长的训练预算。

### 对 P2 的结论

- 「`death` 由 HUD 增量判定」这条路**在 64.5 % 处封顶**，G2 的 death 召回门与 score 符号门在当前**标签定义与数据**下不可达。
  要继续推 G2，必须先决定二选一：
  (a) 把 `death` 的定义改为「`lives` 递减事件」并声明 `lives==0` 的重复死亡不在范围内（需要改标签口径，不是改模型）；
  (b) 承认 score 门要用 `displayScore` 作标签（需要引擎多导出一个字段，属于协议变更，本轮禁止）。
- 在现有口径下，最诚实的状态是：**G2 未达**，且已把"为何未达、差多少、门在哪"全部量化。


## 9. 范围与限制（如实）

1. **play 流未接入 v1 事件头**：数据说明事件标签与 HUD 增量共变、与 playfield 像素无关，故本轮把 play 只保留在数据管线与 v0 模型里，
   没有实现 v1 的 play 辅助头（危险/占用）。这是**部分交付**，不是"已保留"。
2. `stage_clear` 没有事件位，本卡只给出弱代理（`lives`/`bombs` 上跳），不主张任何 stage_clear 召回。
3. `game_over` 在 358,601 帧中一次都没出现，`go` 门不可测。
4. 补采在 ~55k/90k 帧处因预算被人工停止；`stats` 子命令在该次运行中被 `pkill` 连带终止（`STATS_EXIT=143`），
   `index` 已正常重建，统计改由 `task1_align.py` 直接读 label 行得到。
