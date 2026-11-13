# WP05.9 — β=0.5 15 shards（52,500 decisions）20,000 步受控缩放训练 + 20 dev 局（与 1k / 5k / β1-50k 并排）

> **一句话**：把与 WP05.8 完全相同的 15 个 complete β=0.5 shard（52,500 decisions，标签与 5k 版**逐字节相同**）、相同的 mix/优化器/seed/模型换到 **20,000 步**（2,560,000 samples，5k 的 4 倍）后，dev 20 局仍然是 **0/20 过关**，mean **1714.65** 帧（1k 1339.95 → 5k 1689.10 → 20k 1714.65），idle **60.87 %**。离线三项指标（loss 0.9564 → 0.6586 → 0.3245；top1 0.7164 → 0.7993 → 0.9005；macroR 0.1406 → 0.2733 → 0.5904）随预算单调改善，因此 **0/20 不再能由「训练预算不足」解释**，剩下的解释是策略质量 / 状态覆盖
>
> **边界**：这是 WP05.9 的描述性对照，**不是 G410-2**，不设门、不是 pass rate、不是 development score；没有采数据、没有新 DAgger 轮、没有动 confirm 种子。20 局只跑 dev seeds 1000-1019。

生成时间（UTC）：`2026-11-13T07:05:02Z`　·　证据：[`train_beta05_20k.json`](train_beta05_20k.json)

## 0. 关键数字一览

| 项 | 值 |
|---|---|
| 训练步数 | **20,000**（batch 128，AdamW，lr 3e-4 / wd 1e-4 / clip 5，seed 0） |
| 训练 samples | **2,560,000**（S1 512,000 = 20 %，DAgger 2,048,000 = 80 %） |
| 数据 | 15 个 complete shard，**52,500 decisions / 105,000 frames**，β=0.5，stage 0 / normal |
| 标签 | WP05.2 CLI ×15，tolerance 8.0，**与 5k 转换逐字节相同** |
| 模型 | B0 ProbeNet，**731,802 参数**（pinned 731,802） |
| 训练指标 | loss **0.324472**，top1 **0.900506**，macroR **0.590401**（17 类，in-sample 训练批） |
| 吞吐 | 4176.4 samples/s，32.63 steps/s，训练 613.0 s，总 elapsed 625.8 s |
| checkpoint | `b0_s0_dagger_step20000.pt`，2,941,293 B，SHA-256 `15ae1fd5b9a6e0cbbddfd668bb342444bdb44db123b5f7e56abf3c5e7215bd95` |
| dev 20 局 | **0/20 过关**，mean **1714.65**，median **1619.50**，min 1523，max 2332 |
| 动作 | idle **60.87 %**（逐局均值 61.14 %），unique **12.45**，最高频动作 `idle` 占 **60.87 %**，熵 1.8660 bit（满 4.1699） |
| 停止原因 | 全部 `death_stop`（20/20），death 均值 1.00 |
| 引擎 | rl_v3 `691bfd77…` |

## 1. 交付物

| 路径 | 说明 |
|---|---|
| `Document/Update/Alpha/v4.1.0/evidence/WP05/train_beta05_20k.json` | 完整证据（15 shard、训练、20 局、对比、40 项 fail-closed 检查） |
| `Document/Update/Alpha/v4.1.0/evidence/WP05/EVAL_BETA05_20K.md` | 本文件 |
| `.tmp_v410_wp05_train_beta05_20k/` | 运行目录（日志、labels、train_run.json、eval_dev20.json、checkpoint；不提交） |

运行目录里的脚本（均不入库）：`enumerate_shards.py`、`convert_labels.sh`、`verify_labels.py`、`run_train.sh`、`run_eval.sh`、`build_evidence_20k.py`、`render_md.py`、`check_evidence.py`。

## 2. 命令与退出码（逐步实测）

### 2.1 enumerate complete round-2 shards

```bash
# /root/v400-venv/bin/python .tmp_v410_wp05_train_beta05_20k/logs/enumerate_shards.py
# EXIT=0
```

### 2.2 convert 15 shards with the unmodified WP05.2 CLI

```bash
# bash .tmp_v410_wp05_train_beta05_20k/logs/convert_labels.sh -> /root/v400-venv/bin/python -m touhou_rl.v410.dagger.labels --shard .tmp_v410_wp05_round2/shards/dagger-s0d1seed<seed> --out .tmp_v410_wp05_train_beta05_20k/labels/dagger-s0d1seed<seed> --tolerance 8.0
# EXIT=0
# per-shard seconds=[4, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2]
# exit_code_source: derived from the 15 recorded per-shard exit codes (convert_labels.sh exits with their OR)
```

### 2.3 independently verify converted labels (rows + SHA-256 + 5k parity)

```bash
# /root/v400-venv/bin/python .tmp_v410_wp05_train_beta05_20k/logs/verify_labels.py
# EXIT=0
```

### 2.4 train B0, 20,000 steps (everything else identical to WP05.8)

```bash
# bash .tmp_v410_wp05_train_beta05_20k/logs/run_train.sh -> /root/v400-venv/bin/python -m touhou_rl.v410.dagger.train train --steps 20000 --batch-size 128 --seed 0 --s1-share 0.2 --device auto --shard-seeds 2025-2039 --shards-root .tmp_v410_wp05_round2/shards --labels-dir .tmp_v410_wp05_train_beta05_20k/labels --workdir .tmp_v410_wp05_train_beta05_20k --out .tmp_v410_wp05_train_beta05_20k/train_run.json
# EXIT=0
# payload_elapsed_seconds=625.8473
# NOTE: the nvidia-smi header/footer lines in run_train.sh AND run_eval.sh lost their trailing character when the driver scripts were written, so both logged 'Format modifier is not recognized.'; both drivers still exited 0 and the authoritative GPU numbers are the train_run.json gpu block recorded below
```

### 2.5 20 dev episodes (20,000-step checkpoint)

```bash
# bash .tmp_v410_wp05_train_beta05_20k/logs/run_eval.sh -> xvfb-run -a -s '-screen 0 1024x768x24' /root/v400-venv/bin/python tools/rl/v410_student_eval.py --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern --data-dir /root/v400-data/game --checkpoint .tmp_v410_wp05_train_beta05_20k/b0_s0_dagger_step20000.pt --seeds 1000-1019 --stage 0 --difficulty normal --max-frames 19887 --stop-on-death --out .tmp_v410_wp05_train_beta05_20k/eval_dev20.json
# EXIT=0
```

### 2.6 build this evidence file (fail closed)

```bash
# /root/v400-venv/bin/python .tmp_v410_wp05_train_beta05_20k/logs/build_evidence_20k.py
# EXIT=0
```

## 3. shards 与标签（15 个，WP05.2 规则）

| seed | complete | decisions | frames | β | stop_reason | injected | agreement | disagreement | alive / dead / unknown | rows | SHA 校验 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025 | True | 3,500 | 7,000 | 0.5 | decisions | 1,730 | 2,804 | 696 | 3494 / 0 / 6 | 3,500 | labels+obs OK |
| 2026 | True | 3,500 | 7,000 | 0.5 | decisions | 1,759 | 2,776 | 724 | 3451 / 2 / 47 | 3,500 | labels+obs OK |
| 2027 | True | 3,500 | 7,000 | 0.5 | decisions | 1,754 | 2,703 | 797 | 3482 / 0 / 18 | 3,500 | labels+obs OK |
| 2028 | True | 3,500 | 7,000 | 0.5 | decisions | 1,787 | 2,703 | 797 | 3490 / 0 / 10 | 3,500 | labels+obs OK |
| 2029 | True | 3,500 | 7,000 | 0.5 | decisions | 1,758 | 2,805 | 695 | 3490 / 0 / 10 | 3,500 | labels+obs OK |
| 2030 | True | 3,500 | 7,000 | 0.5 | decisions | 1,739 | 2,763 | 737 | 3452 / 5 / 43 | 3,500 | labels+obs OK |
| 2031 | True | 3,500 | 7,000 | 0.5 | decisions | 1,751 | 2,872 | 628 | 3493 / 0 / 7 | 3,500 | labels+obs OK |
| 2032 | True | 3,500 | 7,000 | 0.5 | decisions | 1,764 | 2,704 | 796 | 3489 / 0 / 11 | 3,500 | labels+obs OK |
| 2033 | True | 3,500 | 7,000 | 0.5 | decisions | 1,721 | 2,890 | 610 | 3466 / 3 / 31 | 3,500 | labels+obs OK |
| 2034 | True | 3,500 | 7,000 | 0.5 | decisions | 1,730 | 2,809 | 691 | 3491 / 0 / 9 | 3,500 | labels+obs OK |
| 2035 | True | 3,500 | 7,000 | 0.5 | decisions | 1,826 | 2,905 | 595 | 3426 / 3 / 71 | 3,500 | labels+obs OK |
| 2036 | True | 3,500 | 7,000 | 0.5 | decisions | 1,765 | 2,734 | 766 | 3491 / 2 / 7 | 3,500 | labels+obs OK |
| 2037 | True | 3,500 | 7,000 | 0.5 | decisions | 1,746 | 2,904 | 596 | 3458 / 3 / 39 | 3,500 | labels+obs OK |
| 2038 | True | 3,500 | 7,000 | 0.5 | decisions | 1,755 | 2,695 | 805 | 3401 / 3 / 96 | 3,500 | labels+obs OK |
| 2039 | True | 3,500 | 7,000 | 0.5 | decisions | 1,710 | 2,752 | 748 | 3494 / 0 / 6 | 3,500 | labels+obs OK |
| **合计** | **15/15** | **52,500** | **105,000** | 0.5 | decisions | **26,295** | **41,819** | **10,681** | 52068 / 21 / 411 | **52,500** | 15/15 |

- query coverage `0.055556`，injection rate `0.500857`，agreement rate `0.796552`。
- 15 个 shard 的 rollout checkpoint 是同一个：`6ebe4ea91d81d9333bfa1ccc065fc3515bb25b1d3b90a2684cdaa8a0a69b8086`（β=0.5 的 D1 学生），因此「采数据的策略」和「被训练的策略」不会混淆。
- **与 WP05.8 的 5k 标签逐字节相同**：15/15 个 shard 的 `labels_v410.jsonl` SHA-256 完全一致（缺参考 0）。这是「20k 相对 5k 只改了 `--steps`」这一说法能被验证的原因。

## 4. 训练（20,000 步）

### 4.1 预算与混合

| 项 | 值 |
|---|---|
| steps × batch | 20,000 × 128 = **2,560,000 samples** |
| 优化器 | AdamW，lr 0.0003，wd 0.0001，grad clip 5 |
| 训练 seed | 0 |
| 混合 | S1 20.00 % / DAgger 80.00 %（nominal 0.2 / 0.8） |
| S1 训练集 | 46,397 行，seeds 1-14 |
| DAgger 集 | 52,500 行，15 shard |
| **流重复次数** | S1 **×12**，DAgger **×40** |
| split | 只读 train（val/test 未读） |

> 关键含义：20,000 步把同一批 46,397 / 52,500 行分别重放约 12 / 40 遍。**加预算买到的是重复，不是新状态。**

### 4.2 指标（in-sample 训练批，不是留出精度）

| 指标 | 20k | 5k | 1k |
|---|---|---|---|
| loss | **0.324472** | 0.658562 | 0.956430 |
| top1 | **0.900506** | 0.799258 | 0.716406 |
| macro recall（17 类） | **0.590401** | 0.273256 | 0.140587 |
| majority baseline top1 | 0.617158 | — | — |
| top1 − majority | +0.283348 | — | — |

- 首 100 步 loss `1.4864` → 末 100 步 `0.1103`。
- 标签规则：one_hot(chosen_plan.first_action) cross entropy (B0)；`unknown_used_as_negative` = `False`（unknown 标签不当负样本）。
- per-class recall（0-17）：0=0.980、1=0.528、2=0.705、3=0.895、4=0.500、5=0.635、6=0.707、7=0.909、8=0.600、10=0.299、11=0.639、12=0.645、13=0.336、14=0.391、15=0.517、16=0.370、17=0.380

### 4.3 训练曲线（每 100 step 一个 block，按 1k/5k/10k/15k/20k 取窗口）

| 到第 N 步 | 末 500 步 loss | 累计 loss | 末 500 步 DAgger top1 | 末 500 步 S1 top1 | 末 500 步 DAgger macroR |
|---|---|---|---|---|---|
| 1,000 | 0.8409 | 0.9564 | 0.8004 | 0.5113 | 0.1620 |
| 5,000 | 0.4529 | 0.6586 | 0.9019 | 0.6821 | 0.4164 |
| 10,000 | 0.2352 | 0.4871 | 0.9759 | 0.7410 | 0.8179 |
| 15,000 | 0.1380 | 0.3879 | 0.9914 | 0.8222 | 0.9551 |
| 20,000 | 0.1202 | 0.3245 | 0.9926 | 0.8403 | 0.9703 |

- 20,000 步的累计 loss（按 block 的 count 加权）为 `0.324472`，与 `metrics.loss` = `0.324472` 一致（1e-6 内，构建器检查过）。
- loss 在 5k 之后**继续下降，没有饱和**：1k→5k 降 0.2979，5k→20k 降 0.3341（4 倍步数）。**离线指标没有收益递减，dev 却已经平了**——这正是第 6.4 节要展开的对比。
- 确定性：本 run 的**前 5,000 步与 WP05.8 的 5k run 逐 block 完全一致**（比较 50 个 block，最大绝对差 0.0，exact=True），说明标签、seed、mix、device 全都对齐，唯一变化确实是 `--steps`。

### 4.4 吞吐与 GPU

| 项 | 值 |
|---|---|
| samples/s | 4176.39 |
| steps/s | 32.63 |
| 训练秒 / 总 elapsed | 613.0 s / 625.8 s |
| 数据准备秒 | 237.4 s（占 38.73 %） |
| step 中位/最大 | 0.0295 s / 0.6603 s |
| GPU | NVIDIA GeForce RTX 4070 SUPER，peak alloc 252.11 MiB / reserved 434.0 MiB / total 12281.5 MiB |
| device | `cuda`（requested `auto`，cuda available True，cudnn deterministic True，torch 2.14.1+cu130，python 3.10.12） |

> **墙钟不可与 1k/5k 受控比较**：1k/5k 两臂训练时有 G410-1 teacher batch 共用同一台 20 核主机（samples/s 分别 776.3 / 1512.8），本次 20k 独占主机 （4176.4），所以吞吐差异是环境差异，不是算法差异。

## 5. 20 dev 局（seeds 1000-1019）

### 5.1 汇总

| 项 | 值 |
|---|---|
| 设置 | stage 0 / normal / cap 19,887 帧 / stop-on-death |
| 局数 | 20（n_measured 20） |
| 过关 | **0/20**（clear_rate 0.0000） |
| Wilson 95 % | [0.0000, 0.1611]（描述性，非门） |
| frames mean / median | **1714.65** / **1619.50** |
| frames min / max | 1523 / 2332 |
| frames sd（样本） | 207.6693 |
| death 均值 | 1.00 |
| idle share | **60.87 %**（逐局均值 61.14 %） |
| unique actions 均值 | **12.45** |
| 最高频动作 | `idle`（index 0）占 **60.87 %** |
| top-3 动作占比 | 89.52 % |
| 动作熵 | 1.8660 bit / 满 4.1699 bit |
| 停止原因 | 全部 `death_stop` |
| 单局墙钟合计 | 452.9 s（20.7–28.6 s/局） |

- 原始值：`idle_share` = `0.6087`（pooled），`idle_share_by_episode_mean` = `0.6114`，`unique_actions_mean` = `12.45`，`dominant_action_share` = `0.6087`，`entropy_bits` = `1.8660`。

> 20 局里最好的一局只有 2,332 帧，是 cap 19,887 的 11.73 %。**没有一局接近过关**。

### 5.2 逐局

| seed | frames | deaths | stop_reason | 动作数 | idle 占比 | score |
|---|---|---|---|---|---|---|
| 1000 | 1,612 | 1.0 | death_stop | 15 | 60.79 % | 12,471 |
| 1001 | 1,970 | 1.0 | death_stop | 14 | 64.06 % | 18,920 |
| 1002 | 1,532 | 1.0 | death_stop | 11 | 65.01 % | 12,124 |
| 1003 | 1,762 | 1.0 | death_stop | 12 | 63.00 % | 15,428 |
| 1004 | 1,627 | 1.0 | death_stop | 11 | 60.32 % | 13,677 |
| 1005 | 1,597 | 1.0 | death_stop | 12 | 62.95 % | 13,187 |
| 1006 | 1,683 | 1.0 | death_stop | 12 | 61.52 % | 14,616 |
| 1007 | 1,764 | 1.0 | death_stop | 13 | 64.29 % | 15,495 |
| 1008 | 1,687 | 1.0 | death_stop | 12 | 55.45 % | 16,245 |
| 1009 | 1,962 | 1.0 | death_stop | 15 | 57.08 % | 33,718 |
| 1010 | 1,589 | 1.0 | death_stop | 13 | 62.89 % | 9,669 |
| 1011 | 1,540 | 1.0 | death_stop | 12 | 60.00 % | 11,983 |
| 1012 | 1,571 | 1.0 | death_stop | 12 | 62.98 % | 11,735 |
| 1013 | 1,588 | 1.0 | death_stop | 13 | 62.97 % | 13,228 |
| 1014 | 1,849 | 1.0 | death_stop | 12 | 58.38 % | 17,149 |
| 1015 | 1,523 | 1.0 | death_stop | 10 | 64.70 % | 10,969 |
| 1016 | 1,959 | 1.0 | death_stop | 14 | 56.84 % | 18,866 |
| 1017 | 1,538 | 1.0 | death_stop | 9 | 64.89 % | 11,960 |
| 1018 | 1,608 | 1.0 | death_stop | 11 | 62.19 % | 12,578 |
| 1019 | 2,332 | 1.0 | death_stop | 16 | 52.57 % | 32,014 |

### 5.3 动作分布（20 局合计 17,151 decisions）

| # | 动作 | 次数 | 占比 |
|---|---|---|---|
| 0 | `idle` | 10,440 | 60.87 % |
| 1 | `N` | 67 | 0.39 % |
| 2 | `NE` | 546 | 3.18 % |
| 3 | `E` | 2,371 | 13.82 % |
| 4 | `SE` | 71 | 0.41 % |
| 5 | `S` | 53 | 0.31 % |
| 6 | `SW` | 555 | 3.24 % |
| 7 | `W` | 2,543 | 14.83 % |
| 8 | `NW` | 23 | 0.13 % |
| 9 | `focus-idle` | 0 | 0.00 % |
| 10 | `focus-N` | 6 | 0.03 % |
| 11 | `focus-NE` | 162 | 0.94 % |
| 12 | `focus-E` | 148 | 0.86 % |
| 13 | `focus-SE` | 40 | 0.23 % |
| 14 | `focus-S` | 4 | 0.02 % |
| 15 | `focus-SW` | 103 | 0.60 % |
| 16 | `focus-W` | 14 | 0.08 % |
| 17 | `focus-NW` | 5 | 0.03 % |

- idle 定义：candidate 0 = focus 0 + move 0 (no direction held)。
- 分布**没有塌成单一动作**（12.45 个不同动作/局，熵 1.8660 bit），但明显偏向 `idle`。

## 6. 与 1k / 5k / β1-50k 并排（描述性）

> descriptive side-by-side only. The three beta=0.5 arms share the same 15 shards, the same 52,500 converted label rows (verified byte-identical), the same mix, optimiser, training seed 0, model and 20 dev seeds, so the only controlled change is the step budget 1,000 -> 5,000 -> 20,000. The beta=1.0 arm differs in the injection coefficient and is not a controlled budget comparison. No interval, no significance test, no gate, no decision rule

### 6.1 训练指标

| 臂 | steps | samples | loss | top1 | macroR | samples/s | elapsed s |
|---|---|---|---|---|---|---|---|
| β0.5 1k | 1,000 | 128,000 | 0.956430 | 0.716406 | 0.140587 | 776.3 | 193.8 |
| β0.5 5k | 5,000 | 640,000 | 0.658562 | 0.799258 | 0.273256 | 1512.8 | 461.4 |
| **β0.5 20k** | 20,000 | 2,560,000 | 0.324472 | 0.900506 | 0.590401 | 4176.4 | 625.8 |
| β1.0 50k | 1,000 | 128,000 | 0.833632 | 0.757922 | 0.208093 | 2112.1 | 69.0 |

### 6.2 dev 20 局

| 臂 | 过关 | mean | median | min | max | idle | idle（逐局均值） | unique | 最高频占比 | 熵 bit |
|---|---|---|---|---|---|---|---|---|---|---|
| β0.5 1k | 0/20 | 1339.95 | 1321.00 | 1,101 | 1,747 | 71.07 % | 71.29 % | 6.7 | 71.07 % | 1.2689 |
| β0.5 5k | 0/20 | 1689.10 | 1605.00 | 1,530 | 2,213 | 62.84 % | 63.17 % | 12.5 | 62.84 % | 1.8410 |
| **β0.5 20k** | 0/20 | 1714.65 | 1619.50 | 1,523 | 2,332 | 60.87 % | 61.14 % | 12.45 | 60.87 % | 1.8660 |
| β1.0 50k | 0/20 | 1643.25 | 1580.50 | 1,261 | 2,371 | 72.06 % | 72.55 % | 8.8 | 72.06 % | 1.4289 |

### 6.3 差值（20k − 参照）

| 项 | 20k − 1k | 20k − 5k | 20k − β1-50k |
|---|---|---|---|
| clear rate | 0.0000 − 0.0000 | 0.0000 − 0.0000 | 0.0000 − 0.0000 |
| mean frames | +374.70（+27.96 %） | +25.55（+1.51 %） | +71.40（+4.35 %） |
| median frames | +298.50 | +14.50 | +39.00 |
| idle share | -10.20 pp | -1.97 pp | -11.19 pp |
| unique actions | 12.45 − 6.7 | 12.45 − 12.5 | 12.45 − 8.8 |
| train loss | 0.3245 − 0.9564 | 0.3245 − 0.6586 | 0.3245 − 0.8336 |
| train top1 | 0.9005 − 0.7164 | 0.9005 − 0.7993 | 0.9005 − 0.7579 |
| train macroR | 0.5904 − 0.1406 | 0.5904 − 0.2733 | 0.5904 − 0.2081 |
| samples 比 | ×20.00 | ×4.00 | ×20.00 |

**唯一受控变量**：β0.5 三臂共用同一批 shard、同一批 52,500 行标签（已逐字节验证）、同一 mix / 优化器 / 训练 seed 0 / 模型 / 20 个 dev seed，所以受控变化只有 steps 1,000 → 5,000 → 20,000。β1.0 50k 臂的注入系数不同，只作参照，**不是受控预算比较**。

### 6.4 WP05.9 的核心问题：0/20 还能不能由预算 / 动作崩溃解释

**问题**：20,000 步之后仍然是 0/20，这还能用「训练预算不足」解释吗，还是说它是一个「再加预算也修不好」的动作崩溃 / 策略质量问题？

**短答**：**训练预算已经不能解释 0/20。**从 5,000 步到 20,000 步（4 倍 samples）把训练 loss 从 0.6586 降到 0.3245、macroR 从 0.2733 提到 0.5904（×2.16），末 500 步在 DAgger 行上 top1 0.9926 / macroR 0.9703；但 dev 过关仍是 0/20，mean 只动了 +25.55 帧（+1.51 %）。策略也没有塌成单动作（18 维直方图，id 0 `idle` 占 0.6087，熵 1.8660 / 满 4.1699 bit，每局 12.45 个不同动作）；剩下的限制是单轮 DAgger 的状态覆盖，加上持续偏被动/idle 的动作选择。

#### (a) 离线预算轴：没有饱和，还在快速拟合

| 预算 | loss | top1 | macroR |
|---|---|---|---|
| 1,000 | 0.956430 | 0.716406 | 0.140587 |
| 5,000 | 0.658562 | 0.799258 | 0.273256 |
| 20,000 | 0.324472 | 0.900506 | 0.590401 |

- 三项离线指标随步数**单调且大幅改善**，20k 是三个 β0.5 臂里最好的：相对 5k，loss -0.3341、top1 +0.1012、macroR +0.3171。
- top1 0.9005 比 majority baseline 0.6172 高 +0.2833；baseline 本身就有 0.6172，说明动作先验本来就很重。
- 预算买到的是重复：S1 流被重放 **12 遍**、DAgger 流被重放 **40 遍**，没有新增状态。

#### (a2) 分岔：离线指标继续变好，dev 已经平了

| 从 5k 到 20k（4 倍步数） | 变化 |
|---|---|
| train loss | 0.6586 → 0.3245（÷2.03） |
| train top1 | 0.7993 → 0.9005（+0.1012） |
| train macroR | 0.2733 → 0.5904（×2.16） |
| 末 500 步 DAgger top1 | **0.9926** |
| 末 500 步 DAgger macroR | **0.9703** |
| 末 500 步 S1 top1 | 0.8403 |
| **dev mean frames** | 1689.10 → 1714.65（+25.55 帧，+1.51 %） |
| **dev 过关** | 0/20 → **0/20** |
| dev idle | 62.84 % → 60.87 %（-1.97 pp） |
| dev unique | 12.5 → 12.45 |

**这是 WP05.9 最重要的一个观察**：同 4 倍预算把离线 loss 砍半、macroR 翻倍（末 500 步在采集到的 DAgger 行上 top1 0.9926、macroR 0.9703，已经接近记住标签），dev 存活却只动 +25.55 帧（+1.51 %）。**拟合能力不是瓶颈，dev 上的瓶颈在别处。**

#### (b) dev 轴：预算换来更长存活，但换不到过关

| 预算 | 过关 | mean | median | idle | unique |
|---|---|---|---|---|---|
| 1k | 0/20 | 1339.95 | 1321.00 | 71.07 % | 6.7 |
| 5k | 0/20 | 1689.10 | 1605.00 | 62.84 % | 12.5 |
| 20k | 0/20 | 1714.65 | 1619.50 | 60.87 % | 12.45 |

- mean 从 1339.95 → 1689.10 → 1714.65，20k 相对 1k +374.70 帧、相对 5k +25.55 帧——**增量在缩小**，而 cap 是 19,887 帧。
- 过关数始终是 0/20。**如果 0/20 是欠训练，4 倍预算应该能动它；它没动。**

#### (c) 动作崩溃轴：没有崩成单动作，但持续偏被动

| 预算 | 最高频动作占比 | top-3 占比 | 熵 bit | idle | unique |
|---|---|---|---|---|---|
| 1k | 71.07 % | 98.13 % | 1.2689 | 71.07 % | 6.7 |
| 5k | 62.84 % | 89.04 % | 1.8410 | 62.84 % | 12.5 |
| 20k | 60.87 % | 89.52 % | 1.8660 | 60.87 % | 12.45 |

- 20k 的合并动作分布熵 1.8660 bit（满 4.1699，即 44.75 %），最高频动作 `idle` 占 60.87 %，top-3（idle + E + W）占 89.52 %：**不是退化成单动作，但明显偏被动**。
- 但 idle 一直很高（1k 71.07 % → 5k 62.84 % → 20k 60.87 %），策略在大部分 decision 上「不按方向」，靠被动漂移活到 1715 帧左右就死。
- 动作丰富度在 1k→5k 明显上升（unique 6.7 → 12.5），5k→20k 基本持平、甚至略降（12.5 → 12.45）：**再加预算没有继续拓宽动作**，也仍然产生不了能过 stage 0 的策略。

#### (d) 结论

**0/20 在 20,000 步上被复现了，所以它不是 1k / 5k 预算不够造成的欠训练。**

1. **训练预算不再能解释 0/20。**4 倍步数把离线 loss 从 0.6586 砍到 0.3245、macroR 从 0.2733 提到 0.5904，末 500 步在采集到的 DAgger 行上 top1 已到 0.9926；但 dev 只多活 +25.55 帧（+1.51 %），过关仍是 0/20。**优化器/拟合能力不是瓶颈。**
2. **动作没有塌成单动作，但一直偏被动。**18 维直方图熵 1.8660 bit（满 4.1699），每局 12.45 个不同动作，`idle` 占 60.87 %，E/W 再占 13.82 % / 14.83 %。idle 从 1k 的 71.07 % 到 5k 的 62.84 % 到 20k 的 60.87 %——在降，但一直过半。
3. **剩下的解释是状态覆盖 + 分布偏移。**单轮 DAgger 的 52,500 条决策被重放 40 遍，模型能复现采集状态上的教师动作，却在**自己走出来的状态**上仍以「不按方向」为主，20 局全部 `death_stop`，最好一局 2,332 帧（cap 的 11.73 %）。

用英文写进证据文件的同一结论：

> 0/20 is replicated at 20,000 steps, so it is not an undertraining artefact of the 1,000- or 5,000-step budgets. The budget axis is exhausted for dev: 4x steps bought a 1.5 % survival gain and no clears, while the same 4x cut the offline loss in half and more than doubled macro-recall. The binding constraints left are (a) state coverage -- one DAgger round of 52,500 decisions replayed ~40x cannot represent the states the policy itself visits, and (b) a persistent idle/passive action bias at decision time. More steps over the same rows are not the lever; a new collection round or a different objective (e.g. penalising idle, or on-policy correction) would be, and both are out of scope for WP05.9

**下一步若还要验证**：

- a second DAgger collection round (new on-policy states) is out of scope for WP05.9: no data was collected here and this run does not authorise it
- an offline held-out check on different seeds would separate 'memorised the train batches' from 'generalises'; the DAgger-window top1 of 0.9926 strongly suggests memorisation, which in-sample train metrics cannot by themselves distinguish
- a training-seed sweep (seeds 1-2) would give a variance estimate; one seed cannot separate a poor seed from a poor recipe
- an explicit action-distribution diagnostic (idle penalty / entropy bonus) would test whether the passive bias, rather than the state coverage, is the first thing to break

## 7. 限制

1. one arm (B0), one training seed (0), one budget triple (1k/5k/20k): no variance estimate, so budget cannot be fully separated from seed noise
2. 52,500 round-2 decisions from a single DAgger round; at 20,000 steps the DAgger stream is replayed ~40x and the S1 stream 12x, so the extra budget is repetition rather than new state coverage
3. in-sample train metrics; there is no held-out offline evaluation
4. dev episodes are the 20 dev seeds only (1000-1019); G410-2 is not measured
5. wall time and samples/s are not comparable with the 1k/5k arms, which ran while a G410-1 teacher batch shared the 20-core host; the 20k run had the host to itself
6. the two nvidia-smi lines in run_train.sh and run_eval.sh logged a format error (a dropped trailing character in the driver scripts); both drivers exited 0 and GPU evidence therefore comes from the train_run.json gpu block, not from the log header

补充（对比层的已知混淆）：

- one training seed (0) and one checkpoint per arm: no variance estimate
- the 20k run reuses the 52,500 rows ~40 times (DAgger stream epochs 40) and 46,397 S1 rows 12 times, so the extra budget is repetition of the same states, not new coverage
- the beta=1.0 50k arm used injection_rate 1.0 while the beta=0.5 shards used injection_rate 0.500857, so that contrast is descriptive only
- offline metrics are in-sample train-batch metrics, not held-out accuracy
- 20k trained on an otherwise idle host; the 1k and 5k arms shared the host with a G410-1 teacher batch, so throughput/wall-clock is not comparable even though the samples/s differ by ~1.6x
- the dev episodes are the 20 dev seeds only; G410-2 is not measured and no pass/fail threshold is defined or implied

## 8. not_claimed / 边界

**明确不主张**：

- this is NOT G410-2: no confirm seeds were run and no gate is evaluated
- this is NOT a pass rate, a development score, or a four-arm comparison verdict
- the 20 dev episodes are descriptive only; no interval or significance test is attached
- no data was collected, no new DAgger round was run, and no shard was regenerated
- the offline metrics are in-sample train-batch metrics, not held-out accuracy
- a single 0/20 at 20,000 steps does not prove the approach is dead, only that more steps over the same 52,500 rows do not clear stage 0
- the beta=1.0 contrast in this file is descriptive; that arm is not a controlled budget comparison

**范围声明**：

- 只读输入：`.tmp_v410_wp05_round2/shards/`（15 个 complete，未改动）、`touhou_rl/v410/dagger/train.py`、`labels.py`、`tools/rl/v410_student_eval.py`。
- 本次**没有**采数据、没有新 DAgger 轮、没有确认新种子、没有跑 G410-2、没有动 v400/v380/v390、tools/rl/v400_*、tests、third_party/、state/journal/budget/assets。
- 写入白名单只有三项：`train_beta05_20k.json`、`EVAL_BETA05_20K.md`、`.tmp_v410_wp05_train_beta05_20k/`。
- 没有 commit / push。

**fail-closed 自检**：构建器 `build_evidence_20k.py` 跑完 **40/40** 项检查（任何一项不过就 exit 3、不写证据文件）。其中独立重算的项：checkpoint SHA-256 与字节数、20 局 frames 的 mean/median/min/max、idle 占比、过关数、loss 曲线累计值、15 个 `labels_v410.jsonl` 的 SHA-256。

