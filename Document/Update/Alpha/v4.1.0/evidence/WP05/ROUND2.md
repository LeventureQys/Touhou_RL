# WP05.6 — 第 2 轮 DAgger 采集（v410 B0-50k / obs-v3，β=0.5）

状态：**已完成采集（目标 ≥50,000 decisions 已达标：52,500）**。本文件与
`round2.json` 都**不训练、不评测、不做 G410-2**：没有更新任何参数，没有跑
G410-2 确认集，没有跑 dev-20 评测，也没有跑 β=0 轮。第 2 轮是 **β=0.5**
（每决策 50 % 概率执行教师动作、50 % 执行学生 argmax），访问的是
**师生混合分布**：既不是第 1 轮的教师分布，也**不是**学生自主闭环。

- 上游：`Document/Update/Alpha/v4.1.0/evidence/WP05/ROUND1.md`（第 1 轮，β=1.0）、
  `PLAN.md`（WP05.0 契约）、`TRAIN.md` / `train_50k.json`（WP05.5 产出的 50k checkpoint）。
- 采集器：`tools/rl/v410_dagger_collect.py`，SHA-256
  `3b55f2996bd31e02618a84a6d37f4a6466f946c522f1e530a1c63b1ba20f608f`（**未修改**）。
- 学生（输入）：`.tmp_v410_wp05_train50k/b0_s0_dagger_step1000.pt`，2,941,187 B，
  SHA-256 `6ebe4ea91d81d9333bfa1ccc065fc3515bb25b1d3b90a2684cdaa8a0a69b8086`，
  schema `v410-wp04-bc-arm-checkpoint/1`，arm `B0`，`ProbeNet` 731,802 参数（obs-v3）；
  WP05.5 训练口径：S1 20 % / DAgger 80 %、1,000 steps、batch 128，loss 0.8336、top1 0.7579。
- 引擎：`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，SHA-256
  `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`
  （采集器 `--expected-engine-sha` 强制校验通过）。
- obs 契约：`obs-v3`，digest `ef2fbd15385f26f425db2bfe855d105ee009023acb08c26916e811963553a105`；
  `play96x112 (4,112,96)` / `hud (1,56,160)` / `act_hist (8,18)`，`decision_frames=2`，`frame_stack=4`。
- 数据模板：`/root/v400-data/game`（只读；每 episode 派生私有 `data/env-<seed>`、`data/shadow-<seed>`）。
- 权威证据：`Document/Update/Alpha/v4.1.0/evidence/WP05/round2.json`
  （SHA-256 `81bf1a5a34befa5a6d74ca022e1b69af26c444b9d6b9b03e074d8f091d42fb6f`，72,152 B）；
  运行目录 `.tmp_v410_wp05_round2/`（shards / logs / 两份原始批次报告 / 审计，**不入库**，
  被 `.gitignore` 的 `.tmp_*/` 覆盖）。

**一句话**：15 集 × 3,500 = **52,500 decisions / 105,000 frames**，注入 **26,295**
（**0.500857**，β=0.5），教师可用 **52,500/52,500 = 1.0**，师生一致 41,819
（79.66 %）/ 不一致 **10,681（0.203448）**，每决策 HASH 对齐
**105,000/105,000 = 1.0**（0 mismatch），15/15 集 `stop_reason=decisions`。

---

## 1. 参数映射（与工单字面参数的差异，必须先读）

采集器**没有** `--target-decisions`（`--help` 已确认，本轮未改采集器）。可用的等价开关
是 `--seeds`（支持区间）、`--decisions`（每集决策上限）、`--max-seconds`（整轮 wall 预算）：

| 工单要求 | 实际参数 |
|---|---|
| target-decisions ≥ 50,000 | **无该开关**：由 episodes × decisions 与 `--max-seconds` 隐含；本轮以总量 ≥50,000 验收，实际 52,500 |
| decisions=3,500（每集上限） | `--decisions 3500` |
| max-frames=20,000 | `--max-frames 20000` |
| beta=0.5 | `--beta 0.5` |
| seed-start 2025、15 集 | `--seeds 2025-2039`（重启批次为 `2027-2039`，见 §3） |
| stage 0 / normal / fast / stop-on-death | `--stage 0 --difficulty normal --mode fast`（`--stop-on-death` 默认开） |
| max-seconds=8,700 | 首次调用 `8700`；中止后重启按父代理指示 `9000` |
| 报告路径 | `--out .tmp_v410_wp05_round2/round2_batchA.json`（A1）/ `round2_batchA2.json`（A2） |

其余固定项：`--device cpu`、`--rng-seed 0`（每集 `rng_seed = 0 + seed`）、`--trim-transition`
（默认开）、`--keep-labelled-segment` 关（教师无法标注即整集 fail-closed，不写 shard）、
`--continue-on-error` 开。

## 2. 机器状态、争用时间线、中止与重启（本轮最重要的过程事实）

开工前按要求做了空闲确认；但两波并发教师任务先后进入同一台 20 核机器，直接改变了 wall 口径，
并导致一次**主动中止 + 重启**。时间线（UTC）：

| 时间 | 事件 | load(1m) |
|---|---|---|
| 01:29:05 | **空闲确认**：`ps` 无 `v410_student_eval` / `v410_dagger_collect`；load 0.05 | 0.05 |
| 01:29:50 | 单集计时探针启动（scratch `seed 3000`，不进数据） | 0.09 |
| 01:30:35 | **WP02 G410-1 chunk1**（`v410_teacher_fast20.py`，seeds 61300-61324，rl_laser 引擎，6 个 `v400_teacher.py` worker）启动 | → 63.9 |
| 02:01:00 | 探针**中止**：30.7 min 未跑完一集（CPU 2:56:36 ≈ 5.75 核），已证实不是卡死而是 CPU 饥饿；无 shard | 60.6 |
| 02:14:41 | chunk1 25/25 结束，teacher 进程归零 | 2.66 |
| 02:20:45 | **batchA 启动**（再次确认 load 0.09、无 v410 任务） | 0.09 |
| 02:20:45–02:35:30 | **干净口径**：seed 2025 = 436.329 s、seed 2026 = 437.326 s（3,500 decisions 各） | ~8 |
| 02:38 | **WP02 G410-1 chunk2**（seeds 61325-61349，同样 6 worker）启动 | → 63–67 |
| 03:06:35 | **batchA 被父代理 SIGTERM**（exit 143）：seed 2027 已空转 ~31 min（干净参考 437 s）；2025/2026 两个 shard 保留 | 61 |
| 03:21:31 | chunk2 结束（25/25） | 4.55 |
| 03:22:32 | **batchA2 重启**（`--seeds 2027-2039`，`--max-seconds 9000`），启动时 load 1.74 | 1.74 |
| 05:00:54 | **batchA2 结束，exit 0**（13/13 集全部产出 shard） | 4.07 |

两轮争用的机制（有实测依据，不是猜测）：采集器进程仍能拿到 832–927 % 的 CPU，但 `rl_v3`
引擎的 CPU 份额被压到 ~5 %（干净时 18.5 %），每个 `HASH`/`STEP`/`STATE` 往返的**调度延迟**
被拉高；torch 线程在等引擎回包时空转，烧 CPU 但不推进度。**它只污染 wall/吞吐，不影响
数据正确性**：每个决策边界仍然双端 `HASH` 校验，105,000/105,000 全部相等。

被中止的 batchA（A1）**没有写出原始报告**：采集器只在整轮结束时写 `--out`，SIGTERM 发生在
seed 2027 执行中，因此 `round2_batchA.json` 不存在。幸存的两个 shard（2025/2026）由
`.tmp_v410_wp05_round2/recover_batchA1.py` 从**采集器自己写的 `shard.json`** + 采集器 stdout 的
`wall=` 行重建成 collector 报告形状（`record_source=reconstructed_from_shard_json`，
批次级 `recovered: true`、`aborted: {…}`）。**没有估算任何数值**，且这两个 shard 同样通过了
§6 的独立审计。`round2.json` 的 `round2.episode_record_sources` 明确标注了这两条的来源。

## 3. 实际命令与退出码

三次调用，除 seed 区间 / 预算 / 报告路径外参数完全相同。均在 `cd /mnt/d/WorkShop/Touhou_RL`
下执行，包在 `xvfb-run -a -s "-screen 0 1024x768x24"` 里。

探针（只用于计时，**不产出数据**；中止，无退出码，`timing_probe_aborted.json`）：

```bash
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_dagger_collect.py \
  --checkpoint .tmp_v410_wp05_train50k/b0_s0_dagger_step1000.pt \
  --seeds 3000 --beta 0.5 --decisions 3500 --max-frames 20000 \
  --stage 0 --difficulty normal --mode fast \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --data-root .tmp_v410_wp05_round2/timing/data \
  --shard-root .tmp_v410_wp05_round2/timing \
  --work-dir .tmp_v410_wp05_round2/timing/logs \
  --continue-on-error --max-seconds 1800 \
  --out .tmp_v410_wp05_round2/timing/timing_probe.json
# 无 shard、无 report；2026-11-13T02:01:00Z 因争用被中止（wall≈1,870 s）
```

批次 A1（主跑；**被 SIGTERM**，`EXIT_CODE=143`）：

```bash
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_dagger_collect.py \
  --checkpoint .tmp_v410_wp05_train50k/b0_s0_dagger_step1000.pt \
  --seeds 2025-2039 --beta 0.5 --decisions 3500 --max-frames 20000 \
  --stage 0 --difficulty normal --mode fast \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --data-root .tmp_v410_wp05_round2/data \
  --shard-root .tmp_v410_wp05_round2 \
  --work-dir .tmp_v410_wp05_round2/logs \
  --continue-on-error --max-seconds 8700 \
  --out .tmp_v410_wp05_round2/round2_batchA.json
# EXIT_CODE=143（03:06:35Z 被 SIGTERM；2 集完成 = 7,000 decisions；无原始报告）
```

批次 A2（中止后**从 seed 2027 重启**；`EXIT_CODE=0`）：

```bash
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_dagger_collect.py \
  --checkpoint .tmp_v410_wp05_train50k/b0_s0_dagger_step1000.pt \
  --seeds 2027-2039 --beta 0.5 --decisions 3500 --max-frames 20000 \
  --stage 0 --difficulty normal --mode fast \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --data-root .tmp_v410_wp05_round2/data \
  --shard-root .tmp_v410_wp05_round2 \
  --work-dir .tmp_v410_wp05_round2/logs \
  --continue-on-error --max-seconds 9000 \
  --out .tmp_v410_wp05_round2/round2_batchA2.json
# EXIT_CODE=0（13/13 集全部产出 shard，未触发 wall 预算截断）
```

**没有 batchB**：A2 已让总量达到 52,500 ≥ 50,000，不需要再追加 seeds 2040+。
`round2.json` 的 `round2.stopped_on_wall_budget=false`。

批次级来源（`round2.json.round2.batches`）：

| 报告 | recovered | aborted | exit | episodes | decisions | wall_s | wall_s_productive |
|---|---|---|---|---|---|---|---|
| `.tmp_v410_wp05_round2/round2_batchA2.json`（SHA `15dc3248…ae20aa`） | false | false | **0** | 13 | 45,500 | 5,772.964 | — |
| `.tmp_v410_wp05_round2/round2_batchA1_recovered.json`（SHA `161219d7…47274c`） | **true** | **true** | **143** | 2 | 7,000 | 2,750.0 | 873.655 |

## 4. 真实结果

| 指标 | 值 |
|---|---|
| episodes（有 shard） | **15**（计划 15：seeds 2025-2039） |
| episodes 失败 | 0 |
| decisions（标注决策点） | **52,500**（目标 50,000，`target_met=true`） |
| frames（提交帧） | 105,000（每决策 2 帧） |
| 每集 decisions / frames | 3,500 / 7,000（15/15 集 `stop_reason=decisions`，即达决策上限） |
| injection_rate（实测） | **0.500857**（26,295/52,500，β=0.5，偏差 **+0.000857**；每集 0.4886–0.5217） |
| teacher_available | **52,500/52,500 = 1.0**（无 unlabelable 决策，无 `teacher_unavailable` 集） |
| agreement（师生首动作一致） | **41,819**（79.655 %） |
| disagreement | **10,681**（**20.345 %**，`disagreement_rate=0.203448`） |
| HASH 对齐（每决策前后各一次） | **105,000/105,000 equal，0 mismatched，rate=1.0** |
| 采集器 wall 合计（两次调用） | **8,522.964 s**（A1 2,750.0 + A2 5,772.964） |
| 其中落在 episode 内的 wall | **6,646.067 s**（15 集相加）；其余 1,876.897 s 是启动/重开与**被中止的空转** |
| shell 起止时钟合计 | 8,652 s（A1 02:20:45→03:06:35 = 2,750 s；A2 03:22:32→05:00:54 = 5,902 s） |
| 端到端（含被中止的探针与等待） | 01:29:50→05:00:54 = **12,664 s（3 h 31 min 04 s）** |
| 吞吐（含中止开销） | **6.16 decisions/s**（52,500 / 8,522.964） |
| 吞吐（只计 episode 内 wall） | **7.90 decisions/s**（52,500 / 6,646.067） |
| 单集 wall | min **418.167 s** / median 432.663 s / mean 443.071 s / max **611.471 s** |
| 干净口径单集 wall | 418–447 s（除 2032 的 611.471 s 异常，见 §7.6） |
| 数据体量 | shards **2.7 GB**（15 × ~183 MB）+ data 1.4 MB |

### 4.1 每 episode

| seed | decisions | frames | injections | rate | agreement | disagreement | HASH equal/total | stop | wall_s | record |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025 | 3500 | 7000 | 1730 | 0.494286 | 2804 | 696 | 7000/7000 | decisions | 436.329 | reconstructed |
| 2026 | 3500 | 7000 | 1759 | 0.502571 | 2776 | 724 | 7000/7000 | decisions | 437.326 | reconstructed |
| 2027 | 3500 | 7000 | 1754 | 0.501143 | 2703 | 797 | 7000/7000 | decisions | 432.663 | collector |
| 2028 | 3500 | 7000 | 1787 | 0.510571 | 2703 | 797 | 7000/7000 | decisions | 444.082 | collector |
| 2029 | 3500 | 7000 | 1758 | 0.502286 | 2805 | 695 | 7000/7000 | decisions | 446.799 | collector |
| 2030 | 3500 | 7000 | 1739 | 0.496857 | 2763 | 737 | 7000/7000 | decisions | 442.646 | collector |
| 2031 | 3500 | 7000 | 1751 | 0.500286 | 2872 | 628 | 7000/7000 | decisions | 439.934 | collector |
| 2032 | 3500 | 7000 | 1764 | 0.504 | 2704 | 796 | 7000/7000 | decisions | 611.471 | collector |
| 2033 | 3500 | 7000 | 1721 | 0.491714 | 2890 | 610 | 7000/7000 | decisions | 423.893 | collector |
| 2034 | 3500 | 7000 | 1730 | 0.494286 | 2809 | 691 | 7000/7000 | decisions | 424.625 | collector |
| 2035 | 3500 | 7000 | 1826 | 0.521714 | 2905 | 595 | 7000/7000 | decisions | 418.167 | collector |
| 2036 | 3500 | 7000 | 1765 | 0.504286 | 2734 | 766 | 7000/7000 | decisions | 420.01 | collector |
| 2037 | 3500 | 7000 | 1746 | 0.498857 | 2904 | 596 | 7000/7000 | decisions | 419.017 | collector |
| 2038 | 3500 | 7000 | 1755 | 0.501429 | 2695 | 805 | 7000/7000 | decisions | 425.362 | collector |
| 2039 | 3500 | 7000 | 1710 | 0.488571 | 2752 | 748 | 7000/7000 | decisions | 423.743 | collector |

观测事实（**不是能力结论**）：15 集全部 `stop_reason=decisions` —— 在 3,500 决策
（7,000 帧 ≈ 117 s 游戏时间）内既没有触发 `death_stop`，也没有 `stage_clear` / `frame_cap`。
`teacher_available` 在本轮是 1.0（教师对 B0-50k 访问到的每个状态都给出了计划），
这**不代表**教师对所有状态都可用，只代表这 52,500 个状态都可标注。

## 5. 独立审计（不依赖采集器自报）

`.tmp_v410_wp05_round2/audit_round2.py` 只读 `shard.json` + `labels.jsonl` + `obs/*.npy`
重算（结果 `.tmp_v410_wp05_round2/audit_round2.json`，SHA `3d50a6bf…89eea5`）：
**15 个 shard、52,500 行全过，`verdict=ok`、`problems=[]`**。

| 检查 | 结果 |
|---|---|
| `labels.jsonl` 行数 = `decisions` = obs 行数（每 shard） | 15/15 通过 |
| `labels.jsonl` / `shard.json` / 三平面 `.npy` sha256 = manifest | 15/15 通过 |
| obs 形状/dtype：`(N,4,112,96)`、`(N,1,56,160)`、`(N,8,18)` uint8 | 15/15 通过 |
| manifest `frames == 2 × decisions`、`frames_per_decision == 2` | 15/15 通过 |
| manifest alignment：`boundaries == 2 × decisions`、equal = boundaries、mismatched = 0 | 15/15 通过 |
| `injected == (draw < beta=0.5)` | **52,500/52,500**（0 违例） |
| `executed == teacher`（注入时）/ `executed == student, bomb=0`（未注入时），`mixing_source` 相符 | **52,500/52,500**（0 违例） |
| `agreement == (student_cand == teacher_cand)` 且 `teacher_available == (chosen_plan != None)` | **52,500/52,500**（0 违例） |
| 行内 `beta` 均为 0.5、`hashes_equal_*` 标志与原始比较一致 | 0 违例 |
| `hash_pre == env_hash_pre` 且 `hash_post == env_hash_post` | **105,000/105,000**（0 违例，rate **1.0**） |
| `act_hist[t]` = 实际执行动作 `a[t-8..t-1]` 的 one-hot（bomb 不入历史，越界补零行） | 0 违例 |
| `query_status` 与 `verified_mask` 严格互逆 | 52,500/52,500（0 违例） |
| 记录的 draw 统计（min/mean/max）由行内重算一致 | 15/15 通过 |

审计与采集器报告**独立吻合**：52,500 decisions / 26,295 injections / 41,819 agreement /
10,681 disagreement / 105,000 HASH boundaries（0 mismatched）。审计的
`injection_rate=0.500857`、`disagreement_rate=0.203448` 与 `round2.json` 完全一致；
这也从数据侧证明 β=0.5 是**实测**的（不是把注入率写成 1.0 的伪造）。

审计脚本自测：用 round1 的 `dagger-s0d1seed2000` shard 以 `--expect-beta 1.0` 运行，
得到 3,500/2,333/1,167、HASH 7,000/7,000、0 违例，与 `ROUND1.md` 公布值逐项一致。

## 6. 数据布局（运行目录，不入库）

```
.tmp_v410_wp05_round2/
  shards/dagger-s0d1seed2025 … dagger-s0d1seed2039/   # 15 个，每个 ~183 MB
    shard.json      # decisions/frames/β/injection/alignment/checkpoint/engine/obs/labels
    labels.jsonl    # 每决策一行：draw/injected/student_cand/teacher_cand/executed_cand
                    #   + 原始教师 raw_geo_score/verified/verified_mask/chosen_plan
                    #   + resources/events/query_status/survival_status(deferred)
                    #   + hash_pre/hash_post/env_hash_pre/env_hash_post/hashes_equal_*
    obs/play96x112.npy (N,4,112,96)  obs/hud.npy (N,1,56,160)  obs/act_hist.npy (N,8,18)
  data/env-<seed>/ data/shadow-<seed>/   # 每集私有引擎数据目录（score.dat 不复用）
  logs/batchA.log logs/batchA2.log logs/timing.log logs/wait_then_A2.log
  logs/batchA.exit (143) logs/batchA2.exit (0)
  round2_batchA2.json                    # 采集器原始报告（13 集，exit 0）
  round2_batchA1_recovered.json          # 2025/2026 的重建记录（exit 143，显式标注 recovered）
  timing_probe_aborted.json              # 中止的计时探针记录（无 shard）
  audit_round2.py audit_round2.json      # 独立审计脚本与结果
  recover_batchA1.py derive_round2.py render_tables.py status.sh
  timing/                                # 探针的私有 data 与空 staging（无 shard）
  derive_dryrun.json derive_dryrun2.json audit_selftest_round1.json  # 合并/审计自测
```

shard key 由采集器生成：`dagger-s0d1seed<seed>`（stage 0 / difficulty 1）。

## 7. 限制（必须与结论一起读）

1. **β=0.5 是混合分布**：每 3,500 决策中约 1,750 次执行教师动作、约 1,750 次执行学生动作，
   数据里两者都有。它比第 1 轮（β=1.0，纯教师）更接近学生分布，但**不是**学生闭环。
2. **不是 50k β=0**：β=0 轮（纯学生执行）尚未开始，本轮不能替代它；β=0 轮还要处理
   死亡/过关造成的 episode 长度不均。
3. **不是 G410-2**：没有跑确认集 60000-60199，没有过关率、没有死亡率的门判定。
4. **未训练**：采集器不更新参数、不写 checkpoint；本文件不含任何模型结论。
   52,500 条数据是**下一轮训练的输入**，本轮没有用它训练。
5. **2025/2026 两条记录是重建的**：原始 `round2_batchA.json` 因 SIGTERM 从未写出。
   重建只读采集器写的 `shard.json` 与它自己 stdout 的 `wall=` 行；
   数据本身（labels/obs/HASH）由 §5 的独立审计验证，未做任何估算或补写。
6. **wall 口径被两波争用污染**：A1 的 2,750 s 里有 ~1,876 s 是 seed 2027 的空转；
   干净口径见 2025/2026（436.329 / 437.326 s）。A2 的 seed 2032 是 611.471 s 的异常值
   （该时段 load 一度到 ~20，但按 CPU 采样的前 4 名只有本采集器与两个 `rl_v3` 引擎，
   未见 teacher/评测进程；**未捕获确切原因**，候选是该集 183 MB shard 落盘到 `/mnt/d`
   DrvFs 的 IO 压力或外部任务的短时 IO，未证实），其余 12 集为 418–447 s。
   所有吞吐数字都**同时**给出含争用与只计 episode 两种口径。
7. **只有自然开局**：WP05.3 的 50/30/20 起点采样与 WP05.4 的聚合采样仍未实现，
   所以这批数据只有“关首自然开局”这一类起点。
8. **单阶段**：只跑 `stage 0 / normal / team 0`，没有其它阶段/难度。
9. **`teacher_available=1.0` 的取值范围**：只覆盖 B0-50k 在 β=0.5 下访问到的 52,500 个状态；
   不能外推成“教师对所有状态可用”。
10. **每 shard 的 `not_claimed` 是 WP05.0 的 smoke 样板文字**（如“this smoke collects <=600”）：
    采集器（只读、本轮不得修改）把它硬编码进 `shard.json`。本轮权威免责声明是
    `round2.json` 的 `not_claimed_round2`；`round2.json.not_claimed` 仍是采集器原文，
    故意不改写，以便读者看出差异来源。
11. **`teacher.decision_frames=2`**：v410 每决策执行教师 2-slot 计划的第 1 个 slot，
    第 2 个 slot 只用于打标签；这是 obs-v3 的决策周期约束。

## 8. not_claimed

- 不是 G410-2（v4.1.0 验收门），未测量、未尝试；
- 不是 50k β=0 轮（本轮 β=0.5，仍有半数教师动作）；
- 不是训练、不是评测、不是 arm 比较、不是选模；
- 不声称学生变强、不声称配方有效、不给任何模型结论；
- 不声称 round 1（56k, β=1.0）与 round 2（52.5k, β=0.5）哪个更好——本轮没有训练也没有评测；
- 不声称起点/聚合采样的 50/30/20 已验证；
- 不声称 `survival_status` 通道（留给 v410 relabel 阶段）；
- 不读 confirm split `60000-60199`，不读 dev pool，不重采 D410-S1 语料。

## 9. 下一建议

1. **WP05.7 训练**（不属于本工作包）：用本轮 52,500（β=0.5）与 round 1 的 56,000（β=1.0）、
   S1 的 46,397 行做混合；混合比例与轮次权重必须**先预注册**，不要事后挑。
2. **β=0 轮要单独预注册**：纯学生闭环会出现死亡/过关导致的 episode 长度不均，
   需要先定好“每集上限 / 总决策数 / 失败集是否计入”的口径。
3. **采集与其它重任务串行**：本轮两波 WP02 教师任务把 ~2.0 h 的干净工作量拖成
   3 h 31 min 端到端（含中止与重启）。建议在启动 50k 级采集前，把 teacher 数据生产排到别的时段，
   并在 batchA 开始后用 `load`/`ps` 做前后各一次记录。
4. **不要重跑 seeds 2025-2039**：shard 已存在，采集器会 fail-closed 拒绝覆盖；
   需要补充数据时用新 seed（2040+）或新 shard-root，且**不得覆盖**已有 shard。
5. **relabel 阶段**补 `survival_status`；本轮 15 个 shard 的行里该字段仍是 `deferred`。
6. 若要把吞吐固化成预算常数，用本轮干净口径 **418–447 s/集（≈7.9 decisions/s）**，
   不要用含争用的 6.16 decisions/s。
