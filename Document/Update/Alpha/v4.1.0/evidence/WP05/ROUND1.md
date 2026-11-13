# WP05.1 — 第 1 轮 DAgger 采集（v410 B0 / obs-v3，β=1.0）

状态：**已完成采集（target 50,000 decisions 已达标：56,000）**。本文件与
`round1.json` 都**不训练、不评测、不成门**：没有更新任何参数，没有跑 G410-2，
也没有跑 50k β=0 轮。第 1 轮是 **β=1.0（100 % 执行教师动作）**，访问的是
**教师分布**，不能当作学生自主闭环。

- 上游：`Document/Update/Alpha/v4.1.0/evidence/WP05/PLAN.md`（WP05.0 契约）、
  `Document/Update/Alpha/v4.1.0/README.md` §WP05（预注册课程/预算/止损）。
- 采集器：`tools/rl/v410_dagger_collect.py`（WP05.0 产物，本轮**未修改**）。
- 学生：`.tmp_v410_wp04_arms/b0_seed0_last.pt`
  SHA256 `3bc0ec4461701bebb2ffc551d61dffff77f15cd486125baf9db6eaafc741b3a6`
  （arm `B0`，`ProbeNet`，731,802 参数，obs-v3）。
- 引擎：`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`
  SHA256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`
  （采集器 `--expected-engine-sha` 强制校验通过）。
- obs 契约：`obs-v3`，digest `ef2fbd15385f26f425db2bfe855d105ee009023acb08c26916e811963553a105`；
  `play96x112 (4,112,96)` / `hud (1,56,160)` / `act_hist (8,18)`，`decision_frames=2`，`frame_stack=4`。
- 数据模板：`/root/v400-data/game`（只读；每个 episode 派生私有 `data/env-<seed>`、`data/shadow-<seed>`）。
- 权威证据：`Document/Update/Alpha/v4.1.0/evidence/WP05/round1.json`（合并报告）、
  运行目录 `.tmp_v410_wp05_round1/`（shards / logs / 原始批次报告 / 审计，**不入库**，
  被 `.gitignore` 的 `.tmp_*/` 覆盖）。

---

## 1. 参数映射（与工单字面参数的差异，必须先读）

工单写的 `--target-decisions` / `--episodes` / `--seed-start` **在采集器里不存在**
（`--help` 已确认）。实际可用的等价开关是 `--seeds`（支持区间）、`--decisions`
（每 episode 上限）、`--max-seconds`（整轮 wall 预算）。因此：

| 工单要求 | 实际参数 |
|---|---|
| seed-start=2000、episodes=20 | `--seeds 2000-2019`（20 个 train seed） |
| target-decisions=50000 | **没有该开关**：由“episodes × decisions”与 `--max-seconds` 隐含；本轮以总量 ≥50,000 为验收，实际 56,000 |
| decisions=3500（每集上限） | `--decisions 3500` |
| max-frames=20000 | `--max-frames 20000` |
| beta=1.0、stage0 normal、stop-on-death | `--beta 1.0 --stage 0 --difficulty normal --stop-on-death`（默认） |
| out-dir `.tmp_v410_wp05_round1`、summary `.../WP05/round1.json` | `--shard-root .tmp_v410_wp05_round1 --work-dir .tmp_v410_wp05_round1/logs --out .../WP05/round1.json` |

`--mode fast`（v4.0 fast 教师）、`--device cpu`、`--rng-seed 0`（每 episode
`rng_seed = 0 + seed`，即 2000…2024）、`--trim-transition`（默认开）、
`--keep-labelled-segment` 关（教师无法标注即整集不写 shard）、`--continue-on-error` 开。

## 2. 实际命令与退出码

两次串行调用，**参数除 seed 区间/报告路径外完全相同**（`config.argv` 原文见 `round1.json`）。

批次 A（主跑，`--max-seconds 8700`）：

```bash
cd /mnt/d/WorkShop/Touhou_RL
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_dagger_collect.py \
  --checkpoint .tmp_v410_wp04_arms/b0_seed0_last.pt \
  --seeds 2000-2019 --beta 1.0 --decisions 3500 --max-frames 20000 \
  --stage 0 --difficulty normal --mode fast \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --data-root .tmp_v410_wp05_round1/data \
  --shard-root .tmp_v410_wp05_round1 \
  --work-dir .tmp_v410_wp05_round1/logs \
  --continue-on-error --max-seconds 8700 \
  --out Document/Update/Alpha/v4.1.0/evidence/WP05/round1.json
# EXIT_CODE=4  (预算到点停止：计划 20 集，实跑 11 集 → 未跑满即退出码 4，数据本身 status=ok)
```

批次 B（补充，见 §5；同一契约，无参数降低）：

```bash
  ... --seeds 2020-2024 --max-seconds 9000 \
  --out .tmp_v410_wp05_round1/round1_batchB.json
# EXIT_CODE=0  (5/5 集全部产出 shard)
```

- 批次 A：`2026-11-12T21:29:37Z` 起，`wall_s=9169.939`，11 集，38,500 decisions，
  `exit_code=4`。
- 批次 B：`2026-11-13T00:12:04Z` 起，`wall_s=2509.532`，5 集，17,500 decisions，
  `exit_code=0`。
- 合并：`round1.json` 由 `.tmp_v410_wp05_round1/derive_round1.py` 从两份**原始采集器报告**
  （`round1_batchA.json` = 批次 A 原文、`round1_batchB.json`）逐字段合并：`totals` 由
  16 条 episode 记录求和，其余字段原样保留，`round1.source_reports` 记录来源。合并脚本
  只做求和与求比率，不引入任何新测量值。
- `exit_code=4` 的语义是采集器“没有跑满计划集数”（fail-closed），不是数据损坏；
  16/16 集 `status=ok`、`problems=[]`、`checkpoints_mismatched=0`。

## 3. 真实结果

| 指标 | 值 |
|---|---|
| episodes（有 shard） | **16**（计划 20；批次 A 11 + 批次 B 5） |
| episodes 失败 | 0 |
| decisions（标注决策点） | **56,000**（目标 50,000，`target_met=true`） |
| frames（提交帧） | 112,000（每决策 2 帧） |
| 每集 decisions / frames | 3500 / 7000（16 集全部 `stop_reason=decisions`，即达决策上限） |
| injection_rate（实测） | **1.0**（56,000/56,000，与 β=1.0 一致，偏差 0.0） |
| teacher_available | **56,000/56,000 = 1.0** |
| agreement（师生首动作一致） | **38,545**（68.83 %） |
| disagreement | **17,455**（**31.17 %**，`disagreement_rate=0.311696`） |
| HASH 对齐（env×shadow，每决策前后各一次） | **112,000/112,000 equal，0 mismatched，rate=1.0** |
| 总 wall（两次调用相加） | **11,679.471 s**（3 h 14 min 39 s） |
| 实测吞吐（计入争用） | **4.79 decisions/s**（`samples/s` 同义，定义见 `round1.json.round1.samples_per_s_definition`） |
| 无争用吞吐（批次 B 实测） | **6.97 decisions/s**（17,500 decisions / 2,509.5 s） |
| 单集 wall | min 488.2 s / mean 729.9 s / max 1193.8 s |
| 数据体量 | shards 2.9 GB（16 × 183 MB）+ data 1.4 MB |

### 3.1 每 episode

| seed | decisions | frames | injections | agreement | disagreement | HASH equal/total | stop | wall_s |
|---|---|---|---|---|---|---|---|---|
| 2000 | 3500 | 7000 | 3500 | 2333 | 1167 | 7000/7000 | decisions | 561.309 |
| 2001 | 3500 | 7000 | 3500 | 2387 | 1113 | 7000/7000 | decisions | 586.168 |
| 2002 | 3500 | 7000 | 3500 | 2378 | 1122 | 7000/7000 | decisions | 1025.471 |
| 2003 | 3500 | 7000 | 3500 | 2409 | 1091 | 7000/7000 | decisions | 1151.286 |
| 2004 | 3500 | 7000 | 3500 | 2468 | 1032 | 7000/7000 | decisions | 1167.223 |
| 2005 | 3500 | 7000 | 3500 | 2469 | 1031 | 7000/7000 | decisions | 1193.782 |
| 2006 | 3500 | 7000 | 3500 | 2462 | 1038 | 7000/7000 | decisions | 654.861 |
| 2007 | 3500 | 7000 | 3500 | 2454 | 1046 | 7000/7000 | decisions | 634.339 |
| 2008 | 3500 | 7000 | 3500 | 2480 | 1020 | 7000/7000 | decisions | 1121.688 |
| 2009 | 3500 | 7000 | 3500 | 2375 | 1125 | 7000/7000 | decisions | 559.655 |
| 2010 | 3500 | 7000 | 3500 | 2344 | 1156 | 7000/7000 | decisions | 513.663 |
| 2020 | 3500 | 7000 | 3500 | 2385 | 1115 | 7000/7000 | decisions | 488.183 |
| 2021 | 3500 | 7000 | 3500 | 2434 | 1066 | 7000/7000 | decisions | 506.739 |
| 2022 | 3500 | 7000 | 3500 | 2446 | 1054 | 7000/7000 | decisions | 493.297 |
| 2023 | 3500 | 7000 | 3500 | 2275 | 1225 | 7000/7000 | decisions | 496.405 |
| 2024 | 3500 | 7000 | 3500 | 2446 | 1054 | 7000/7000 | decisions | 524.686 |

观测事实（**不是能力结论**）：16 集都是 `stop_reason=decisions` —— 教师执行的
7000 帧里既没有触发 `death_stop`，也没有 `stage_clear`；即 3500 决策（7000 帧 ≈ 117 s
游戏时间）内没有死、也没有过关。学生的 disagreement 在 29.1 %–35.0 % 区间
（每集 1020–1225 次），说明 B0 与教师在这些状态上确实不同，DAgger 有信号可学。

## 4. 独立审计（不依赖采集器自报）

`.tmp_v410_wp05_round1/audit_round1.py` 只读 `shard.json` + `labels.jsonl` + `obs/*.npy`
重算，56,000 行全过，`verdict=ok`、`problems=[]`
（结果：`.tmp_v410_wp05_round1/audit_round1.json`）：

| 检查 | 结果 |
|---|---|
| `labels.jsonl` 行数 = `decisions` = obs 行数（每 shard） | 16/16 通过 |
| `labels.jsonl` sha256 = manifest、三平面 `.npy` sha256 = manifest | 16/16 通过 |
| obs 形状/dtype：`(N,4,112,96)`、`(N,1,56,160)`、`(N,8,18)` uint8 | 16/16 通过 |
| `injected == (draw < beta)` | 56,000/56,000（0 违例） |
| `executed == teacher`（注入时）/ `executed == student, bomb=0`（未注入时），`mixing_source` 相符 | 56,000/56,000（0 违例） |
| `agreement == (student_cand == teacher_cand)` 且 `teacher_available == (chosen_plan != None)` | 56,000/56,000（0 违例） |
| `hash_pre == env_hash_pre`、`hash_post == env_hash_post` | 56,000/56,000（0 违例） |
| `act_hist[t]` = 实际执行动作 `a[t-8..t-1]` 的 one-hot（bomb 不入历史，越界补全零行） | 0 违例 |
| `query_status` 与 `verified_mask` 严格互逆 | 56,000/56,000（0 违例） |

审计与采集器报告的数字独立吻合：56,000 decisions / 56,000 injections /
38,545 agreement / 17,455 disagreement。

## 5. 为什么是 16 集而不是 20（真实成本与争用）

- 单集成本由决策上限决定：3500 decisions = 7000 frames ≈ 117 s 游戏时间，
  无争用时 **488–587 s wall**（≈6.0–7.2 decisions/s），加载/重开/双引擎启动的
  固定开销约 10–20 s/集。
- 批次 A 的 `--max-seconds 8700` 是**整轮**预算，采集器在**每集开始前**检查，
  所以最后一集（2010）约在 8,650 s 启动、约 9,170 s 结束后才停：批次 A 实际 wall 9,170 s，
  11 集 = 38,500 decisions。
- 期间检测到另一个并行任务（`v410_student_eval.py` 的 WP04.5 四臂评测链，
  `.tmp_v410_wp04_eval_arms/run_arms.sh full B0 B1 B2 B3`）在与本采集争用 20 核机型，
  单集 wall 被拉高到 1,025–1,194 s（≈1.8–2.1×）。**这只影响 wall/吞吐口径，不影响
  数据正确性**：每决策的 env×shadow HASH 对齐率仍是 1.0，标签仍来自真实教师。
- 因此按既定口径（不降 target、不杀别人的进程）在批次 A 之后追加了批次 B
  （seeds 2020-2024，争用已消失，单集 488–525 s），把总量补到 56,000。
- 结论：**50k 目标本身约需 2.0 h 的无争用机器时间**（50,000 / 6.97 ≈ 7,170 s；该吞吐按
  批次 B 的端到端 wall 计，已含集间启动/加载开销）；若与 dev-20 评测并发，按 1.8–2×
  预算，即 50k ≈ 3.5–4 h。

## 6. 数据布局（运行目录，不入库）

```
.tmp_v410_wp05_round1/
  shards/dagger-s0d1seed<seed>/          # 16 个，每个 183 MB
    shard.json                           # decisions/frames/β/injection/alignment/checkpoint/engine/obs/labels
    labels.jsonl                         # 每决策一行：draw/injected/student_cand/teacher_cand/executed_cand
                                         #   + 原始教师 raw_geo_score/verified/verified_mask/chosen_plan
                                         #   + resources/events/query_status/survival_status(deferred)
                                         #   + hash_pre/hash_post/env_hash_pre/env_hash_post
    obs/play96x112.npy  (N,4,112,96) uint8
    obs/hud.npy         (N,1,56,160) uint8
    obs/act_hist.npy    (N,8,18)     uint8
  data/env-<seed>/ data/shadow-<seed>/   # 每集私有引擎数据目录（score.dat 不复用）
  logs/run_round1.log logs/run_supplement.log   # 采集器 stdout 全文 + EXIT_CODE
  logs/*.exit                            # 退出码
  round1_batchA.json round1_batchB.json   # 采集器原始报告（合并前）
  audit_round1.py audit_round1.json       # 独立审计脚本与结果
  derive_round1.py                        # 合并脚本（只做求和/比率）
```

shard key 由采集器生成：`dagger-s0d1seed<seed>`（stage 0 / difficulty 1）。

## 7. 限制（必须与结论一起读）

1. **β=1.0 是教师分布**：每 3500 决策中 3500 次执行教师动作，学生只“提议”并被记录；
   这批数据**不是**学生自主闭环的分布，不能用于宣称学生能力。
2. **不是 50k β=0**：β=0 轮（纯学生执行）尚未开始，本轮不能替代它。
3. **不是 G410-2**：没有跑 100 局确认集，没有过关率，没有任何门的判定。
4. **未训练**：采集器不更新参数、不写 checkpoint；本文件不包含任何模型结论。
5. **只有自然开局**：WP05.3 的 50/30/20 起点采样与 WP05.4 的聚合采样都没有实现，
   所以这批数据只有“关首自然开局”这一类起点。
6. **单阶段**：只跑 `stage 0 / normal / team 0`，没有其它阶段/难度。
7. **wall 口径被争用污染**：批次 A 的 `wall_s` 含 1.8–2.1× 争用放大，真实无争用成本见
   §5；`round1.json` 的 `wall_s=11,679.471` 是两次调用 wall 之和，`samples/s=4.79`
   也是这个含争用口径。无争用参考值是 6.97 decisions/s。
8. **每 shard 的 `not_claimed` 是 WP05.0 的 smoke 样板文字**（例如“this smoke collects
   <=600”）：采集器（只读、本轮不得修改）把它硬编码进 `shard.json`。本轮权威免责声明是
   `round1.json` 的 `not_claimed_round1`；`round1.json.not_claimed` 仍是采集器原文，
   故意不改写，以便读者看出差异来源。
9. **`teacher.decision_frames=2`**：v410 每决策执行教师 2-slot 计划的第 1 个 slot，
   第 2 个 slot 只用于打标签；这是 obs-v3 的决策周期约束，不是“教师计划被截断”。

## 8. not_claimed

- 不是 G410-2（v4.1.0 验收门），未测量、未尝试；
- 不是 50k β=0 轮（本轮 β=1.0，教师分布）；
- 不是训练、不是评测、不是 arm 比较、不是选模；
- 不声称学生变强、不声称配方有效、不给任何模型结论；
- 不声称起点/聚合采样的 50/30/20 已验证；
- 不声称 `survival_status` 通道（留给 v410 relabel 阶段）；
- 不读 confirm split `60000-60199`，不读 dev pool，不重采 D410-S1 语料。

## 9. 下一建议

1. **R2：β=0.5**（预注册第二轮），起点同一批 train seed 之后（建议 `2025-2039`），
   `--decisions 3500 --max-frames 20000 --max-seconds 8700`，一次调用即可跑到 ~52k；
   目标仍按 ≥50k 决策点计，并复样本轮口径（注入率应在 0.5±0.05，教师可用率 ≥99.5 %）。
2. **R3/R4 之前先做一次 20 分钟的单集计时**（`--seeds 2030 --decisions 3500`）确认
   机器空闲，避免再次出现 1.8–2× 争用放大；**评测与采集串行**，不要并发。
3. **WP05.2 训练**：先用本轮 56k（β=1.0）与 D410-S1 固定切片做聚合采样的最后一次
   离线确认（50/30/20 权重、危险状态 ≤5×），再开训练；训练不属于本工作包。
4. **dev-20 评测**（WP04.3 口径）只作为训练后的开发标尺，不作为门；G410-2 留给 WP07
   的确认集。
5. 若要把吞吐做成可比的正式数字，建议单独跑一次“纯采集、机器空闲”的 30 分钟基准，
   把 6.97 decisions/s 固化成 round-2 的预算常数。
