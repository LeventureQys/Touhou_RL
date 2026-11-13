# WP05.0 — v4.1.0 DAgger 采集器适配与 β-mixing 管线 smoke

状态：**WP05.0（管线验证）**。本文件只规定 WP05.0 的实现契约，并把
`Document/Update/Alpha/v4.1.0/README.md` §WP05 已预注册的课程、预算与止损抄录
为可执行参数。**本文件与本次 smoke 都不构成 G410-2，也不构成任何训练结果。**

- 上游预注册：`Document/Update/Alpha/v4.1.0/README.md:348-366`（WP05 目标、每轮顺序、
  G410-2 定义、止损）。
- 输入学生：`.tmp_v410_wp04_arms/b0_seed0_last.pt`
  （`v410-wp04-bc-arm-checkpoint/1`，arm `B0`，`ProbeNet`，731,802 参数，obs-v3）。
- 引擎：`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`
  SHA256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`。
- 数据模板：`/root/v400-data/game`（只读，每个 episode 派生私有可写副本）。
- 采集器：`tools/rl/v410_dagger_collect.py`（本工作包新增；v4.0 的
  `tools/rl/v400_dagger_collect.py` 只读参考，未修改、未导入）。
- 测试：`tests/test_v410_dagger_collect.py`（不启动引擎）。
- 证据：`Document/Update/Alpha/v4.1.0/evidence/WP05/dagger_smoke.json`。

---

## 1. WP05.0 的边界

**做：** 把 v4.0 的 DAgger 采集器适配到 v410 B0 + obs-v3，跑 2 个 episode 的
β-mixing 管线 smoke，证明以下链路真实可用：

1. v410 B0 checkpoint 加载（`ProbeNet`，731,802 参数，obs-v3 形状）；
2. obs-v3 像素 readback（`play 4×112×96`、`hud 1×56×160`、`act_hist 8×18`）；
3. 每 2 帧一次 argmax 决策（`obs_spec_v3.DECISION_FRAMES = 2`）；
4. 按 seed 抽 `draw`、以 β 概率执行教师动作、否则执行学生动作；
5. shadow 教师在**学生实际访问的状态**上打标签（`scores[18]` / `chosen` /
   `verified` / `verified_mask`）；
6. env×shadow 双引擎 HASH checkpoint 对齐（每个决策前后各一次）；
7. `shard.json` + `labels.jsonl` + `obs/*.npy` 落盘，且保留 raw teacher 明细；
8. fail-closed：checkpoint / 教师 / 动作 / β / HASH 任一不成立就不写 shard。

**不做（明确不声称）：**

- 不训练（本采集器不更新任何参数，也不写 checkpoint）；
- 不做 50k 首轮、不做 250k 后续轮；
- 不做 G410-2（不跑 100 局确认集，不报过关率）；
- 不做 arm 比较、不做 dev 20 局评测、不做 UCB/选模；
- 不读 confirm split `60000-60199`，不读 dev pool `1000-1099`，不重采 D410-S1 语料
  `1-20`；
- 不修改 `touhou_rl/v400|v380|v390`、`tools/rl/v400_*`、`tests/test_v400_*`、
  `third_party/`、v410 既有文件、state/journal/budget/assets、WP03/WP04 证据。

2 个 episode 是**管线验证**，不是能力测量：它只能说"这条链路跑通了、产物可审计"，
不能说"学生变强了"或"配方有效"。

---

## 2. β 课程（预注册，README §WP05.2）

| 轮次 | β（教师执行概率） | 学生侧含义 | 备注 |
|---|---|---|---|
| R1 | **1.0** | 100% 执行教师动作 | 教师访问分布，**不能**称作学生自主闭环 |
| R2 | **0.5** | 一半教师、一半学生 | 本 WP05.0 smoke 使用的 β |
| R3 | **0.25** | 教师只做少量纠正 | 第 3 轮做 3 个训练种子确认性比较 |
| R4 | **0.0** | 纯学生执行、教师只打标签 | 真正的学生分布 |
| R5 | **0.0** | 同上，最终轮 | 最终至少两轮 β=0，且 β=0 学生访问决策点 ≥ 50,000 |

规则（均由采集器强制）：

- **每个状态都用实际执行历史构造学生输入**：`act_hist` 只 push **实际执行**的动作
  （`executed_cand`，可能是教师动作），在动作生效后 push，当前决策不会泄漏进历史。
- 教师没有 `chosen_plan` 的决策**永不注入**（`inject_decision(.., None) == False`），
  否则上报的 β 与数据不符。
- β 非法（非有限、不在 `[0,1]`）→ 拒绝启动（exit 2），不写任何 shard。
- `draw` 由 `numpy.random.default_rng(rng_seed + seed)` 产生，逐 episode 独立；
  `draw_sequence_sha256` 写入 `shard.json`，可复核。

---

## 3. 起点采样 50 / 30 / 20（预注册，README §WP05.3）

**"起点"指 episode 的起始状态从哪来**（采集器只负责 rollout 与打标签；起点由采集
配置决定，属于 WP05.x）：

| 份额 | 起点类型 | 定义 |
|---|---|---|
| **50 %** | 关首 | 阶段开局的自然起点（严格评测恒用自然开局；起点采样只用于训练数据） |
| **30 %** | 教师中后段 | 教师轨迹中段/后段的状态（教师能安全续行的困难段） |
| **20 %** | 上一轮学生失败前 5 秒 | 上一轮学生在死亡前约 5 秒内的状态 |

- 起点必须**重建状态与像素历史**并**核验 HASH**（采集器的 `align_pair` /
  `require_aligned` 就是这条规则的最小实现）。
- 严格评测（G410-2、dev 20 局）**始终自然开局**，不使用本节采样。
- WP05.0 只跑自然开局（`--stage 0 --difficulty normal`，`RESET` 后直接开始），
  因此本 smoke 不验证起点采样的 50/30/20，只验证"起点=自然开局"这条路径。

---

## 4. 聚合采样 50 / 30 / 20（预注册，README §WP05.4）

**"聚合"指训练集的采样权重**（DAgger 的 A；由 WP05.x 训练侧实现）：

| 份额 | 来源 | 定义 |
|---|---|---|
| **50 %** | 当前轮 | 本轮新采集的 β-mixed 决策 |
| **30 %** | 最近历史轮 | 上一轮（必要时再上一轮）的 shard |
| **20 %** | 固定 BC 数据 | D410-S1 固定切片，保证不遗忘、锚住离线分布 |

- 必须**记录真实采样权重**（每轮一条聚合记录，含 shard 列表、权重、行数）。
- 危险状态最多上调到普通状态的 **5 倍**，且必须报告覆盖变化；不得把所有非 idle
  状态一律当危险。
- 历史轮次保留策略：β 越高的轮次权重越低（R1 teacher 访问分布不得主导训练集）。

WP05.0 **不实现**聚合采样器，只保证每轮产物（`shard.json` 的 seed / β /
injection / alignment / obs sha256 / labels sha256）足以让聚合侧无歧义地加权。

---

## 5. 预算：首轮 50k，后续 250k（预注册，README §WP05.1）

| 轮次 | 决策点预算 | 说明 |
|---|---|---|
| R1 | **50,000** | 首轮用于证明流水线（本 WP05.0 smoke 只是它的 1/83 缩小版：2×≤300） |
| R2–R5 | **250,000** / 轮 | 默认每轮预算 |
| 加速条件 | **500,000** / 轮 | 仅当 8 小时内可完成**且**有收益（离线 val top-1 与 dev 截断存活同向改善） |

- 预算按**决策点**计，不按帧计；1 决策 = 2 帧（obs-v3 决策周期）。
- 每轮必须记录：真实标注决策/秒（分环境、教师推演、像素、加载、重开成本）、
  注入率、教师可用率、HASH 对齐率、学生/教师分歧率。
- 本 WP05.0 smoke 的对应参数：`--decisions 300 --max-frames 1200 --beta 0.5`，
  2 个 episode，seeds `2000`、`2001`（train 范围，允许；confirm/dev/corpus 全部拒绝）。

---

## 6. 停止条件

### 6.1 单 episode（采集器强制，全部 fail-closed）

| 条件 | 结果 |
|---|---|
| 决策数达到 `--decisions` | `stop_reason=decisions`，写 shard |
| 提交帧达到 `--max-frames` | `stop_reason=frame_cap`，写 shard |
| 首次死亡且 `--stop-on-death` | `stop_reason=death_stop`，写 shard |
| 阶段推进（`stage_clear`） | 丢弃跨越 transition 的那 1 个决策（v4.0 语义），写 shard |
| game over / engine exit | `stop_reason=game_over|engine_exit`，写 shard |
| **教师无法标注该状态** | 默认**不写 shard**（exit 4）；`--keep-labelled-segment` 才保留已标注段 |
| **HASH 不一致** | `AlignError` → **不写 shard**，staging 目录删除 |
| **动作越界 / β 非法 / checkpoint 不符 / obs 契约漂移 / 教师缺失** | 拒绝或失败，**不写 shard** |
| 0 决策 | 失败，不写 shard |

### 6.2 单轮（WP05.x 实现）

- 达到本轮决策预算 → 正常结束；
- HASH 对齐率 < **1.0** → 该轮作废，复核采集侧（不允许"轻微不一致"）；
- 实现注入率偏离 β 超过 **±0.05**（二项 3σ 之外）→ 该轮作废，复核 RNG / seed；
- 教师可用率 < **99.5 %**（`teacher_unavailable` 频繁）→ 按 README 止损条款改恢复课程
  或教师，不得重复蒸馏无解状态；
- 学生/教师分歧率在 β=0 轮 < 1 %（学生完全复制教师）或 > 95 %（学生无学习信号）→
  该轮标注为诊断异常，不进入训练集比较；
- 未过 G410-2 前**不消耗正式确认集**（confirm seeds 由评测入口独占）。

### 6.3 整个配方（预注册止损，README §WP05 止损）

- 三轮后仍零过关，且截断存活相对受控 BC 的改善 **< 30 %** → 结束当前配方，
  复核失败状态；该条件**只**决定停止这个配方，不代表结束整个项目；
- 优先执行 WP04 的表示/标签替代分支，并继续 WP06；
- 若教师在学生恢复状态也无安全续行 → 改恢复课程或教师，而非重复蒸馏无解状态；
- 选择最终候选后冻结所有身份再评一次；WP07 的 Windows 腿补齐 G410-2。

---

## 7. 产物 schema（WP05.0 实际写出）

```
<shard-root>/shards/<key>/shard.json      # 决策/帧/seed/β/injection/alignment/obs/labels/checkpoint/engine
<shard-root>/shards/<key>/labels.jsonl    # 每决策一行：student/teacher/executed + raw teacher + resources/events
<shard-root>/shards/<key>/obs/play96x112.npy   # (N,4,112,96) uint8
<shard-root>/shards/<key>/obs/hud.npy          # (N,1,56,160) uint8
<shard-root>/shards/<key>/obs/act_hist.npy     # (N,8,18)    uint8
```

`labels.jsonl` 每行关键字段：

- mixing：`beta`、`draw`、`injected`、`student_cand`、`teacher_cand`、
  `teacher_slots`、`executed_cand`、`executed_bomb`、`agreement`、`mixing_source`；
- raw teacher（**不丢**）：`raw_geo_score[18]`（教师存的折扣后分数）、
  `verified_mask[18]`、`verified[]`（`died` / `death_frame` / `margin` /
  `frames_advanced` / `max_position_error` / `trajectory_matches` / `end_xy`）、
  `chosen_plan`、`chosen`、`chosen_verified`、`margin`、`bomb`、
  `bomb_death_available`、`frame_event_flags`、`teacher_events`；
- `query_status[18]`：**直接写出**，取值为 `verified` / `unknown`，与 `verified_mask`
  严格互逆（`touhou_rl.v410.student.s1_data.parse_label_row` 反推同一映射）；
- `survival_status`：**留空并标注 deferred**（DAgger shard 没有 D1 的 `model_match`
  通道，校准后的 survival 状态由后续 v410 relabel 生成，**不伪造**）；
- `resources`：决策点的 `deaths` / `lives` / `bombs` / `score`（README §5 的资源与事件）；
- alignment：`hash_pre` / `hash_post` / `env_hash_pre` / `env_hash_post` /
  `hashes_equal_*` / `frames_advanced`。

`shard.json` 关键字段：`complete`（只有全链路通过才为 `true`）、`decisions`、
`frames`、`beta`、`injection`（含 `injection_rate` / `disagreement_rate` /
`mean_draw`）、`alignment`（`method=hash-checkpoint`、`boundaries`、
`checkpoints_equal`、`checkpoints_mismatched=0`、`all_equal`）、`checkpoint`
（path / sha256 / arm / model_config / parameters）、`engine`（binary / sha256）、
`obs`（shape / dtype / 每平面 sha256）、`labels`（rows / sha256 / raw_teacher_rows_kept）、
`teacher`（mode / replan_decisions / decision_frames / unverified_discount）、
`not_claimed`。

决策周期说明：`obs_spec_v3.DECISION_FRAMES = 2`，而 v4.0 fast 教师的一个计划块是
`replan_decisions × decision_frames = 2 × 2 = 4` 帧（= 两个 v410 决策宽）。因此
采集器**每决策提交 1 个动作 / 2 帧**，只执行教师计划的第一个 slot，其余 slot 只用于
打标签；`teacher.decision_frames` 必须等于 2，否则拒绝运行。这样 obs-v3 的
`frames[t-3..t]` 与训练数据同网格，而不是被拉成 4 帧决策。

---

## 8. 本次 smoke 的定位（重要）

| 声称 | 是否成立 |
|---|---|
| 采集器能在真实引擎上跑完 2 个 episode 并写出可审计 shard | 由 `dagger_smoke.json` 决定 |
| env×shadow HASH 在每个决策点一致 | 由 `alignment` 决定（必须 `checkpoints_mismatched=0`） |
| β-mixing 的 draw / 注入 / 执行记录自洽 | 由 `injection` 与 `labels.jsonl` 决定 |
| 学生变强 / 配方有效 | **不成立**（未训练、未评测） |
| G410-2 通过 | **不成立**（未跑 100 局确认集） |
| 首轮 50k 决策 | **不成立**（本次最多 600） |
| 起点/聚合采样 50/30/20 已验证 | **不成立**（WP05.x 才实现） |

WP05.0 的验收就是：**管线跑通 + 产物自洽 + fail-closed 在真实引擎上被证明过一次**
（本仓库已在调试中真实观察到该路径：act_hist 形状违例 → 无 shard、exit 4、
report `status=failed`）。

---

## 9. WP05.0 实测（2 episodes，2026-11-12T21:22:49Z，exit 0）

权威记录是 `dagger_smoke.json`；本节只是索引。复现命令见该文件的 `config.argv`。

| 指标 | seed 2000 | seed 2001 | 合计 |
|---|---|---|---|
| frames | 600 | 600 | 1,200 |
| decisions | 300 | 300 | 600 |
| stop_reason | `decisions` | `decisions` | — |
| injections（`draw < β`） | 148 | 129 | 277 |
| injection rate（β=0.5） | 0.493333 | 0.43 | 0.461667 |
| mean draw | 0.494012 | 0.535830 | — |
| teacher_available | 300/300 | 300/300 | 600/600 |
| agreement | 172 | 120 | 292 |
| disagreement | 128 | 180 | 308（0.513333） |
| HASH checkpoints | 600/600 equal | 600/600 equal | 1,200/1,200（rate 1.0） |
| act_hist = 执行历史 | 0 违例 | 0 违例 | 0 |
| wall | 35.16 s | 51.57 s | 86.77 s |

独立审计（`labels.jsonl` 逐行重算，不依赖采集器自报）：

- `injected == (draw < β)`，且 `executed == teacher if injected else (student, bomb=0)`：600/600；
- `query_status` 与 `verified_mask` 严格互逆、`verified[].first_action` 均在 mask 内：600/600；
- `hash_pre == env_hash_pre`、`hash_post == env_hash_post`：600/600；
- 三个 obs 平面 sha256 与 `shard.json` 记录一致，形状/`dtype` 与 obs-v3 一致；
- `act_hist[t]` 恰好是 `executed[0..t-1]` 的 one-hot 窗口（"学生输入由实际执行历史构造"）：0 违例；
- `labels.jsonl` 行数 = `decisions` = obs 行数；`shard.json.complete == true`。

观测事实（不是能力结论）：B0 在 300 个决策里只用了 **7** 个不同动作，教师用了 **8**
个；两局都没有死亡（`deaths=0`，`lives=2` 全程），分数 0 → 649 / 763。

**仍然不是 G410-2。**

