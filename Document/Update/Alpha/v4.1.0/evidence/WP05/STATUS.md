# WP05 状态（DAgger）

- 时间：2026-11-13T05:30:00+08:00
- 结论：**WP05.0 DAgger 管线 smoke 通过（2 episodes/600 decisions，HASH 1200/1200 对齐）**；正式 50k 第一轮尚未开始。WP05 仍为 in_progress。

## 1. WP05.0 管线 smoke（完成）

证据：[`dagger_smoke.json`](dagger_smoke.json)、[`PLAN.md`](PLAN.md)；`pytest tests/test_v410_dagger_collect.py` → **75 passed**；compileall exit 0。

- 学生：v410 B0_s0（731,802 参数，obs-v3 像素），β=0.5；seeds 2000/2001；stage0 normal。
- 2 episodes：**600 decisions / 1,200 frames / 600 renders**；stop_reason=decisions；无 error。
- 注入：**277/600 = 46.17%**；teacher_available=100%；student-teacher agreement **292/600 = 48.67%**，disagreement **308/600 = 51.33%**。
- HASH 对齐：**1,200/1,200 checkpoints equal，0 mismatch**；obs 三平面 SHA 已记录；checkpoint SHA 3bc0ec44…、engine SHA 691bfd77…。
- 输出：2 个 shard（`labels.jsonl` + obs + shard.json），运行目录 `.tmp_v410_wp05_smoke/`。
- 这是管线验证，不是 G410-2，也不是 50k 第一轮；n=2 不构成开发成绩。

## 1.1 WP05.2 DAgger 标签转换（完成）

证据：[`LABELS.md`](LABELS.md)、[`labels_audit.json`](labels_audit.json)；`pytest tests/test_v410_dagger_labels.py` → **65 passed**；compileall exit 0。

- seed2000 shard 真实转换：**3,500 decisions**；alive **3,492**、dead **0**、unknown **8**（全部 model_mismatch，误差 8.000003–8.000031）。
- query coverage 3,500/63,000 = 5.5556%；query_status↔verified_mask 互逆 3,500/3,500；injected 3,500/3,500（β=1.0），agreement 2,333 / disagreement 1,167。
- 输出 `labels_v410.jsonl` 3,500 行 / 19,661,798 B / SHA `3494caa6…f21b4cf2`；obs/labels SHA 与 shard.json 一致；`s1_data.parse_label_row` 100% 接受。
- 规则复用 WP02.4 `classify_plan`，tolerance 8.0；不新增/不重拟合；unknown 不作负例。
- 限制：本轮 0 死亡，verified_dead 仅单测覆盖；仅转换 seed2000，50k 首轮未完。

## 1.2 WP05.3 DAgger 训练 smoke（完成，正向描述性）

证据：[`TRAIN.md`](TRAIN.md)、[`train_smoke.json`](train_smoke.json)；`pytest tests/test_v410_dagger_train.py` → **34 passed**；compileall exit 0；回归 labels 65 passed。

- 6 个 complete shards：**21,000 decisions / 42,000 frames**；alive 20,963 / dead 0 / unknown 37。
- 混合：S1 25,600（20.000%）+ DAgger 102,400（80.000%）=128,000 samples；unknown 不作负例。
- B0 731,802，chosen one-hot CE，1,000 steps×128：loss 2.904→0.546、top1 **0.7549**、macroR 0.2052；DAgger top1 0.8200、S1 0.4947；2,079.9 samples/s。
- 2 个 dev episodes（n=2，非成绩）：
  - seed1000：**1,609 帧** vs B0 baseline 1,398（Δ+211）；
  - seed1001：**1,200 帧** vs 818（Δ+382）；
  - 均 death_stop、0 过关；`n=2 / is_a_gate=false / not_a_development_score=true`。
- 限制：仅 1,000 steps、6 shards/21k、0 verified_dead；不能写 G410-2 或 WP05 通过；需补齐 50k、β=0.5 轮与 ≥20 dev episodes。

## 1.3 WP05.4 DAgger checkpoint 20 局开发评测（完成，描述性）

证据：[`EVAL_DAGGER.md`](EVAL_DAGGER.md)、[`eval_dagger_20.json`](eval_dagger_20.json)。

- checkpoint：WP05.3 的 B0+S1 20%/DAgger 80%、1,000 steps（SHA e771314d…）。
- 结果：**0/20 过关、20/20 death_stop**；frames mean **1,574.9**、median **1,375**、min 743、max 4,048；idle **70.47%**。
- 对比 B0 baseline（0/20、mean 1,453.25、idle 45.81%）：mean +8.4%，但 idle 明显升高，仍 0 过关。
- n=20、无 CI；工具 verdict `is_a_gate=false`、`not_a_development_score=true`；**不是 G410-2**。
- 结论：DAgger 21k/1,000-step 只有微弱存活改善，远不足以过门；需要 50k 全量、β=0.5/多轮以及动作多样性改进。

## 1.4 WP05.5 50k DAgger checkpoint 20 局（完成，描述性）

证据：[`EVAL_50K.md`](EVAL_50K.md)、[`train_50k.json`](train_50k.json)。

- 15 个 complete shards：**52,500 decisions / 105,000 frames**（seed2024 在固定列表后才 complete，未并入；round1 总计 56,000）。
- 训练预算与 WP05.3 完全相同：1,000×128、seed0、S1 20%/DAgger 80%；loss 0.8336、top1 **0.7579**、macroR 0.2081。
- 20 dev：**0/20 过关、20/20 death_stop**；mean **1,643.25**、median **1,580.5**、min 1,261、max 2,371、idle **72.06%**。
- 对比 21k（0/20、mean 1,574.9、median 1,375、idle 70.47%）：mean **+4.34%**、median +205.5、idle +1.59pp；配对 12 长/8 短。
- 仍 0 过关、n=20/单 seed；不是 G410-2。数据量增加只有小幅存活改善，下一步必须做 β=0.5/多轮与动作多样性。

## 1.5 WP05.6 β=0.5 第二轮 DAgger（进行中）

- seeds 2025–2039、β=0.5，batchA 重启后（seed2027+）当前 **3 个 shards**：
  - seed2025：injection **0.4943**、agreement 80.11%、HASH 7,000/7,000；
  - seed2026：injection **0.5026**、agreement 79.31%、HASH 7,000/7,000；
  - seed2027：injection **0.5011**、agreement 77.23%、HASH 7,000/7,000；
  - seed2028：injection **0.5106**、agreement 77.23%、HASH 7,000/7,000；
  - seed2029：injection **0.5023**、agreement 80.14%、HASH 7,000/7,000；
  - seed2030：injection **0.4969**、agreement 78.94%、HASH 7,000/7,000；
  - seed2031：injection **0.5003**、agreement 82.06%、HASH 7,000/7,000；
  - seed2032：injection **0.5040**、agreement 77.26%、HASH 7,000/7,000；
  - seed2033：injection **0.4917**、agreement 82.57%、HASH 7,000/7,000；
  - seed2034：injection **0.4943**、agreement 80.26%、HASH 7,000/7,000；
  - seed2035：injection **0.5217**、agreement 83.00%、HASH 7,000/7,000；
  - seed2036：injection **0.5043**、agreement 78.11%、HASH 7,000/7,000；
  - seed2037：injection **0.4989**、agreement 82.97%、HASH 7,000/7,000；
  - seed2038：injection **0.5014**、agreement 77.00%、HASH 7,000/7,000；
  - seed2039：injection **0.4886**、agreement 78.63%、HASH 7,000/7,000。
- 最终 15 个 shards：**52,500 decisions / 105,000 frames**；injection **26,295/52,500 = 0.500857**；agreement **41,819 = 79.66%**；disagreement 20.34%；HASH **105,000/105,000 equal**。
- **50k 目标已达标**；WP05.7 已用完全相同 1,000-step/S1 20% 预算训练 β=0.5 学生并跑 20 dev，结果待出；不是 G410-2。
- WP05.7 训练离线（描述性）：loss **0.9564**、top1 **0.7164**、macroR **0.1406**（β1 50k 为 0.7579/0.2081；β=0.5 训练更难，尚未收敛）；DAgger 流 top1 0.7724。
- 20 dev 评测进行中；闭环结果才是判据。

## 1.6 WP05.7 β=0.5 20 局（完成，负结果）

证据：[`EVAL_BETA05.md`](EVAL_BETA05.md)、[`train_beta05.json`](train_beta05.json)。

- 15 shards/52,500 decisions；1,000×128、seed0、S1 20%/DAgger 80%。
- 训练离线：loss **0.9564**、top1 **0.7164**、macroR **0.1406**（β1-50k 0.7579/0.2081）。
- 20 dev：**0/20 过关、20/20 death_stop**；mean **1,339.95**、median **1,321**、min 1,101、max 1,747；idle **71.07%**。
- 对比 β1-21k（1,574.9）、β1-50k（1,643.25）：**β=0.5 更差**；至少在这个 1,000-step 预算下没有收益。
- 不能把 β=0.5 当解法；下一步先测 5,000 steps（欠训练）再谈分布。

## 1.7 WP05.8 β=0.5 5,000 步训练（训练完成，20 dev 待出）

- 同一 15 shards/52,500、同 seed/mix；只把 steps 从 1,000 改为 5,000。
- 训练：loss **0.6586**（1k 0.9564）、top1 **0.7993**（0.7164）、macroR **0.2733**（0.1406）；1,512.8 samples/s；checkpoint `692fdc61…`。
- 控制：前 1,000 步 loss 与 1k 逐点一致，确认是续训；唯一差异是后 4,000 步。
- 20 dev 等 G410-1 chunk4 结束后跑；尚无闭环结果，不能下结论。

## 1.8 WP05.8 β=0.5 5k 步 20 局（完成，显著改善但仍 0/20）

证据：[`train_beta05_5k.json`](train_beta05_5k.json)、[`EVAL_BETA05_5K.md`](EVAL_BETA05_5K.md)。

- 训练：5,000 steps；loss **0.6586**、top1 **0.7993**、macroR **0.2733**；checkpoint `692fdc61…`。
- 20 dev：**0/20**、mean **1,689.1**、median **1,605**、min 1,530、max 2,213、idle **62.84%**、unique actions **12.5**。
- 对比 1k（1,339.95/71.07%/6.7）与 β1-50k（1,643.25/72.06%）：mean 和 idle 均改善，**仍 0 过关**。
- 结论：欠训练是部分原因，但不足以解释 0 过关；下一步做 20k 步缩放与动作平衡诊断。

## 1.9 WP05.9 β=0.5 20k 步 20 局（完成，饱和但仍 0/20）

证据：[`train_beta05_20k.json`](train_beta05_20k.json)、[`EVAL_BETA05_20K.md`](EVAL_BETA05_20K.md)。

- 训练：20,000 steps；loss **0.3245**、top1 **0.9005**、macroR **0.5904**；checkpoint `b0_s0_dagger_step20000.pt`。
- 20 dev：**0/20**、mean **1,714.65**、median **1,619.5**、min 1,523、max 2,332、idle **60.87%**、unique 12.45。
- 1k→5k→20k：mean 1,339.95→1,689.1→1,714.65；idle 71.07→62.84→60.87；**三档全部 0/20**。
- 结论：续训到 20k 只带来小幅存活改善，已接近饱和；瓶颈不在步数/离线拟合，必须换数据分布（用 20k 学生做第二轮 DAgger）或做动作平衡/闭环诊断。

## 1.10 idle/动作分布诊断（完成，根因线索）

证据：[`IDLE_DIAG.md`](IDLE_DIAG.md)、[`idle_diag.json`](idle_diag.json)；15 shards/52,500 decisions；labels SHA 15/15 未改。

- 教师 idle **66.16%**；采集学生 idle 73.69%；executed 69.89%；agreement 79.66%，其中 idle–idle 占 agreement 的 80.54%。
- **44.58% 决策的 18 路分数完全打平**，78.50% 与次优打平；idle 标签 **67.38% 来自打平行**；idle 在 52,500 决策里**从未是唯一最优**。
- 同分数随机 tie-break 的离线反事实 idle 仅 4.67%；“取第一个最大值”把 idle 推到 66.16%。
- 训练标签先验 idle 61.72%，20k dev idle 60.87%（差 −0.84pp）；策略在拟合 tie-break 先验，不是单纯视觉失败。
- 学生自转移 90.43%，最长单一动作 1,334；31.89% 决策在 ≥512 run 内（97.09% idle）——“idle 冻结”。
- 建议：idle 降权/重采样（≤43.96%）、tie-break/等价集合、第二轮采集反卡死；不应继续加步数。

## 1.11 WP05.12 idle 降权训练（进行中）

- 同一 52.5k/5,000 步预算，只改 loss 的 idle 权重（目标 idle share 0.44），验证先验假设；20 dev 待出。

## 1.12 WP05.12 idle 降权 20 局（完成，轻微改善但仍 0/20）

- 同一 52.5k/5,000 步，只把 idle 权重降到目标占比 0.44（idle multiplier 0.4874，其余 1.4628）。
- 训练：loss **0.6568**、top1 **0.7973**、macroR **0.3142**（5k none 为 0.6586/0.7993/0.2733；macro 提升）。
- 20 dev：**0/20**、mean **1,693.3**、median **1,610.5**、min 1,520、max 2,011、idle **59.77%**、unique 10.2。
- 对比 5k none（1,689.1/62.84%）：mean +4.2、idle −3.07pp；仍 0 过关。
- 结论：idle 先验是因素之一，但单纯 loss 降权不足以改变闭环；需 tie-break/等价集合或新数据分布。

## 2. 下一阶段（WP05.1）

- 首轮 50k 正式完成：**16 episodes / 56,000 decisions / 112,000 frames**；HASH **112,000/112,000 equal**、注入 1.0、agreement 68.83%、disagreement 31.17%。
- batch A 11/20（exit 4 仅表示 8700s 预算截断），batch B 5/5 补齐；合并后 `round1.json` status ok，独立审计 0 违例。
- **50k 数据量已达标**；WP05.5 用 50k 的 1,000-step 训练已完成，20-dev 评测进行中；仍不是 G410-2。
- 调度决定：保留 WP04.5 四臂评测；主跑到 `--max-seconds 8700` 后用 seeds 2020+ 追加补充批次，不降 50k target，并合并进 round1.json/ROUND1.md。
- 下一步：继续首轮剩余 shards；完成后写 round1.json/ROUND1.md，再按 β=0.5 进入第二轮。
