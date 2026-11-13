# WP04.4 — B0_s0 在真实引擎上的 20 局 dev 开发局（闭环落差量化）

- 时间：2026-11-13T05:12:08+08:00（WSL 侧 2026-11-12T21:12:08Z）
- 结论：**B0_s0（BC，20 epoch）在 dev seeds 1000–1019 的 20 局真实 `rl_v3` 开发局中 0/20 过关、20/20 首死停止；首死帧 mean 1,453.25 / median 1,374 / min 815 / max 2,386。动作 45.81% 为 idle，向下（S/SE）合计仅 0.25%，focus 仅 6.54%。** 这 20 局是 dev 开发局量化，**不是 G410-2、不是正式确认集、不是门、不是四臂比较**。
- 范围：只评测；**未训练、未采数据、未跑 confirm split（60000–60199 被入口硬拒绝）、未改评测工具、未改既有证据**。

## 0. 关键数字一览

| 项 | 值 |
|---|---|
| checkpoint | `.tmp_v410_wp04_arms/b0_seed0_last.pt`（B0，train seed 0，20 epoch，731,802 参数） |
| checkpoint SHA-256 | `3bc0ec4461701bebb2ffc551d61dffff77f15cd486125baf9db6eaafc741b3a6` |
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern` |
| 引擎 SHA-256 | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` |
| 评测入口 | `tools/rl/v410_student_eval.py`（只读使用；SHA-256 `c63100eaf668eaedd6422fdf662e38e320f620ca18041107ab501b4375c20818`） |
| 口径 | S1N：`--stage 0` + `normal` + `--max-frames 19887` + `--stop-on-death`，2 帧/决策，obs-v3 真实像素 |
| dev seeds | 1000–1019（20 个，全部 dev；无 confirm、无语料 seed） |
| episodes / 退出码 | **20 / exit 0**（无 `error`、无 `reset_failed`） |
| stage_clear | **0 / 20 = 0.0%**；Wilson 95% **[0.0000, 0.1611]**（仅描述，**不是 G410-2**） |
| 首死帧 | n=20，**mean 1,453.25**，median 1,374.0，min 815，max 2,386，sd 403.22，p25 1,225.75，p75 1,619.5 |
| 死亡数 | total 20，mean 1.0（`--stop-on-death` 的构造值，不是整局死亡分布） |
| stop_reason | `death_stop` × 20（无 `stage_clear`、无 `frame_cap`、无 `engine_exit`） |
| 决策数 | 14,537 |
| 动作 | idle **45.81%**（6,660/14,537），E 14.69%，W 13.74%，SW 13.43%，NE 5.12%，focus 合计 6.54%，**S 0.12% / SE 0.14%** |
| 帧利用 | mean 为 cap 的 7.31%，max 为 cap 的 12.00%，**0 局触到 19887 帧上限** |
| 离线指标（同 checkpoint） | train top1 0.9769 / val top1 **0.6280** / val macro recall **0.3080**（val 9,874 样本） |
| 实际墙钟 | 逐局 `wall_s` 合计 417.226 s；批跑 21:04:26Z → 21:12:08Z（约 7 min 42 s） |

## 1. 交付物与范围

| 文件 | 作用 |
|---|---|
| [`tools/rl/v410_student_eval.py`](../../../../../../tools/rl/v410_student_eval.py) | 只读使用的评测入口（未修改，mtime 04:56 早于本次运行） |
| [`eval_b0_20.json`](eval_b0_20.json) | 本次 20 局的机器可读证据（77,815 bytes；20 条 episode + aggregate + verdict/limits） |
| [`EVAL20.md`](EVAL20.md) | 本文件：命令、退出码、20 局表、汇总、限制、下一建议 |
| `.tmp_v410_wp04_eval20/` | 运行目录（`run_eval20.sh`、`meta.txt`、`run.log`、`stats.json`、`md_fragments.txt`、`summarize_eval20.py`、`validate_eval20.py`），**不入库** |

本次只新增 `eval_b0_20.json` 与 `EVAL20.md`，运行产物只写 `.tmp_v410_wp04_eval20/`（该目录被 `.gitignore` 的 `.tmp_*/` 忽略，不入库）。**未修改** `tools/rl/v410_student_eval.py`、`tests/test_v410_*`、`touhou_rl/v410/`、`third_party/`、`.tmp_v410_wp04_arms/`、`state/journal/budget/assets`、WP03/WP04 既有证据文件（`BC.md` / `DIAG.md` / `EVAL.md` / `bc_smoke.json` / `diag.json` / `eval_smoke.json` 的 mtime 均早于本次运行）。未 commit / push。工作区在本次任务前已有大规模未提交状态（含 `Document/` 清理删除记录），与本任务无关，本任务未触碰。

> 并发写入说明：`v4.1.0/STATUS.md`（05:08:50）、`v4.1.0/journal.jsonl`（05:14:20）、`v4.1.0/state.json`（05:14:25）在本次评测运行期间被**本任务之外**的并行操作更新；本报告不引用其内容，也不对其负责。

## 2. 实际命令与退出码

在 WSL2 中从 Windows 侧工作区运行（引擎与数据在 `/root`，仓库在 `/mnt/d/WorkShop/Touhou_RL`）：

```bash
cd /mnt/d/WorkShop/Touhou_RL
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_student_eval.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp04_arms/b0_seed0_last.pt \
  --seeds 1000-1019 --stage 0 --difficulty normal --max-frames 19887 --stop-on-death \
  --out Document/Update/Alpha/v4.1.0/evidence/WP04/eval_b0_20.json
# wrote Document/Update/Alpha/v4.1.0/evidence/WP04/eval_b0_20.json (exit 0)
# EXIT_CODE=0
```

- 运行包装脚本：[`.tmp_v410_wp04_eval20/run_eval20.sh`](../../../../../../.tmp_v410_wp04_eval20/run_eval20.sh)；日志 `run.log`，元数据 `meta.txt`（`START_UTC=2026-11-12T21:04:26Z`、`END_UTC=2026-11-12T21:12:08Z`、`EXIT_CODE=0`）。
- 运行前复核两个 SHA（与任务给定一致）：

```bash
sha256sum .tmp_v410_wp04_arms/b0_seed0_last.pt \
  /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern
# 3bc0ec4461701bebb2ffc551d61dffff77f15cd486125baf9db6eaafc741b3a6  b0_seed0_last.pt
# 691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6  th08-modern
```

- 独立复核（只读，不启动引擎）：用入口自身的 `validate_episode_record()` 重验 20 条记录，并检查 seed 序列、`error=null`、`frames ≤ 19887`、`exit_code=0`：

```bash
/root/v400-venv/bin/python .tmp_v410_wp04_eval20/validate_eval20.py
# VALIDATED: 20/20 episode records pass ... ; no error/reset_failed; frames <= 19887; exit_code 0
# EXIT_CODE=0
```

所有统计数字由脚本从 JSON 计算（`.tmp_v410_wp04_eval20/summarize_eval20.py` → `stats.json`），本文件只转抄，不手写统计量：

```bash
/root/v400-venv/bin/python .tmp_v410_wp04_eval20/summarize_eval20.py > .tmp_v410_wp04_eval20/stats_stdout.txt
# EXIT_CODE=0
```

## 3. 20 局逐局表

`frames` 是 `--stop-on-death` 口径下的**首死帧数**；`deaths` 恒为 1.0（构造值，不列）；`lives_end=2` 表示 3 残机开局后的第 1 次死亡；全部 `stage_start=0 → stage_end=0`，即无一局离开 Stage 1。

| seed | frames | steps | stop_reason | clear | score | lives_end | bombs | idle% | unique | wall_s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 1398 | 699 | `death_stop` | false | 13102 | 2 | 0 | 42.2% | 12 | 20.34 |
| 1001 | 818 | 409 | `death_stop` | false | 1512 | 2 | 0 | 50.9% | 8 | 14.40 |
| 1002 | 1251 | 626 | `death_stop` | false | 7438 | 2 | 0 | 49.7% | 11 | 18.84 |
| 1003 | 815 | 408 | `death_stop` | false | 1507 | 2 | 0 | 61.5% | 7 | 14.34 |
| 1004 | 2386 | 1193 | `death_stop` | false | 30207 | 2 | 0 | 44.1% | 14 | 30.56 |
| 1005 | 1379 | 690 | `death_stop` | false | 5165 | 2 | 0 | 38.6% | 13 | 20.27 |
| 1006 | 1369 | 685 | `death_stop` | false | 5030 | 2 | 0 | 47.4% | 9 | 20.02 |
| 1007 | 1664 | 832 | `death_stop` | false | 8774 | 2 | 0 | 50.0% | 14 | 23.08 |
| 1008 | 1476 | 738 | `death_stop` | false | 8520 | 2 | 0 | 39.7% | 11 | 21.16 |
| 1009 | 1457 | 729 | `death_stop` | false | 5629 | 2 | 0 | 38.4% | 10 | 20.73 |
| 1010 | 1214 | 607 | `death_stop` | false | 5414 | 2 | 0 | 45.5% | 12 | 18.37 |
| 1011 | 1224 | 612 | `death_stop` | false | 7589 | 2 | 0 | 34.8% | 9 | 18.74 |
| 1012 | 1214 | 607 | `death_stop` | false | 7456 | 2 | 0 | 49.8% | 10 | 18.42 |
| 1013 | 2239 | 1120 | `death_stop` | false | 33295 | 2 | 0 | 49.1% | 14 | 28.78 |
| 1014 | 1256 | 628 | `death_stop` | false | 9512 | 2 | 0 | 45.4% | 11 | 18.77 |
| 1015 | 1361 | 681 | `death_stop` | false | 8167 | 2 | 0 | 48.3% | 11 | 19.39 |
| 1016 | 1486 | 743 | `death_stop` | false | 8924 | 2 | 0 | 50.5% | 8 | 20.86 |
| 1017 | 1894 | 947 | `death_stop` | false | 24666 | 2 | 0 | 49.2% | 13 | 25.16 |
| 1018 | 1231 | 616 | `death_stop` | false | 7475 | 2 | 0 | 43.7% | 9 | 18.73 |
| 1019 | 1933 | 967 | `death_stop` | false | 19457 | 2 | 0 | 43.8% | 12 | 26.26 |

## 4. 汇总（全部由 `eval_b0_20.json` 计算）

### 4.1 过关与首死

| 指标 | 值 |
|---|---|
| episodes | 20（seeds 1000–1019） |
| stage_clear | **0 / 20 = 0.0%** |
| Wilson 95% 过关率区间 | **[0.0000, 0.1611]**（0/20；仅描述性，**不是 G410-2**，也不是正式确认集） |
| 首死帧 mean / median | **1,453.25 / 1,374.0** |
| 首死帧 min / max | **815 / 2,386** |
| 首死帧 sd / p25 / p75 | 403.22 / 1,225.75 / 1,619.5 |
| 死亡数 total / mean | 20 / 1.0（`stop-on-death` 构造） |
| stop_reason | `death_stop: 20` |
| 触到 19887 帧上限的局 | 0 |
| mean / max 帧占 cap | 7.31% / 12.00% |
| 决策总数 | 14,537 |

### 4.2 动作分布（20 局合计 14,537 次决策，argmax，无采样）

| 动作 id | 名称 | 次数 | 占比 |
|---|---|---|---|
| 0 | idle | 6,660 | 45.81% |
| 1 | N | 40 | 0.28% |
| 2 | NE | 745 | 5.12% |
| 3 | E | 2,135 | 14.69% |
| 4 | SE | 20 | 0.14% |
| 5 | S | 17 | 0.12% |
| 6 | SW | 1,953 | 13.43% |
| 7 | W | 1,998 | 13.74% |
| 8 | NW | 18 | 0.12% |
| 9 | focus-idle | 0 | 0.00% |
| 10 | focus-N | 1 | 0.01% |
| 11 | focus-NE | 234 | 1.61% |
| 12 | focus-E | 623 | 4.29% |
| 13 | focus-SE | 5 | 0.03% |
| 14 | focus-S | 0 | 0.00% |
| 15 | focus-SW | 73 | 0.50% |
| 16 | focus-W | 11 | 0.08% |
| 17 | focus-NW | 4 | 0.03% |

- idle 45.81%；水平横移（E+W）28.43%；向下（SE+S+SW）13.69% 但几乎全是 **SW**，纯 S+SE 仅 **0.25%**；focus 动作合计 6.54%（focus-idle 与 focus-S 为 0）。
- 逐局 idle 比例 34.8%（1011）–61.5%（1003）；逐局 unique action 7–14 个（并集 16/18，`focus-idle` 与 `focus-S` 从未被选中）。
- 只记录分布，**不据此推断死因**；方向性解释需要另立的诊断实验。

### 4.3 与 WP04.3 smoke（同 checkpoint）的一致性

WP04.3 的 2 局 smoke（seeds 1000/1001，同 checkpoint、同引擎、同 S1N 口径）与本次 20 局中重叠的两个 seed **逐项完全一致**——不只是帧数，连 policy 输入哈希与动作序列哈希都相同：

| seed | frames smoke / 本次 | steps | score | stop_reason | 直方图 | `policy_input_sha256` | `action_sequence_sha256` |
|---|---|---|---|---|---|---|---|
| 1000 | 1,398 / **1,398** | 699 / 699 | 13,102 / 13,102 | `death_stop` / `death_stop` | 相同 | **相同**（`41d171cc…`） | **相同** |
| 1001 | 818 / **818** | 409 / 409 | 1,512 / 1,512 | `death_stop` / `death_stop` | 相同 | **相同**（`cad86d0e…`） | **相同** |

**结论：与 WP04.3 完全一致，且是比特级可复现**（这两个 seed 上策略输入序列与动作序列的 SHA-256 都逐字节相同）。这同时说明本次 runner 与 WP04.3 的接线没有漂移；但它只覆盖 2 个 seed，其余 18 个 seed 未重复跑，因此**不能**外推为"20 个 seed 全部可复现"。

### 4.4 闭环落差（仅上下文，不作比较结论）

同一 S1N 口径下，v4.0.0 的 teacher 参考行（`Document/Update/Alpha/v4.0.0/experiments/EXP-P1-012-g1-evaluation.md` §5.3）为 **20/20 过关、0 死亡、mean 13,565.2 帧**（min 12,812 / max 15,204）。B0_s0 的 mean 首死帧 1,453.25 ≈ 该参考的 **10.71%**，过关 **0/20 vs 20/20**。

这**不是**头对头臂比较：政策不同（v400 teacher vs v4.1.0 BC 学生）、工具不同（`v400_student_eval.py` vs `v410_student_eval.py`）、没有共同随机化、n=20 也没有做区间比较；只用于说明"离线 BC 基线在闭环里远未接近参考水平"的量级，**不构成效应量或 G410-2 结论**。

## 5. 限制（明确不声称）

1. **B0 基线，不是 RL 策略**：B0 是 D410-S1 上的 chosen-one-hot 行为克隆（20 epoch），离线 train top1 0.9769 / val top1 0.6280 / val macro recall 0.3080，已明显过拟合；本报告量化的是**这个 BC checkpoint 的闭环表现**，不是"v4.1.0 学生"的最终能力。
2. **只有 B0**：B1（probe + bomb head，731,931 参数）与 B2/B3（spatial 家族）在当前只读入口下会被 fail-closed 拒绝；**没有 B1/B2/B3 的 20 局数据，因此没有任何四臂比较**。
3. **不是门、不是确认集**：seed 只用 dev 1000–1019；confirm 60000–60199 与 D410-S1 语料 seed 1–20 被入口硬拒绝，本次未触达。Wilson 区间 [0.0000, 0.1611] 只是 0/20 的描述性区间，**不是 G410-2 的过关率区间**，也不能写成"过关率上限 16.1%"这类门结论。
4. **`--stop-on-death` 口径**：每局在首死处停止，`deaths` 恒为 1.0、`lives_end` 恒为 2.0，因此**没有整局死亡分布、没有 game-over 口径、没有 score/PSN/人类对齐指标**；有信息量的量是**首死帧数**。
5. **JSON 自带标签是工具写死的**：`eval_b0_20.json` 里 `kind="dev-smoke"`、`wp="WP04.3"`、`schema="v410-student-eval-smoke/1"`、`limits` 仍写 "This is a SMOKE"，因为入口是**只读使用、未修改**。本文件是 WP04.4 的 20 局 dev 运行，标签未改；`verdict` 五项（`is_a_gate` / `is_a_development_score` / `is_a_pass_rate` / `is_a_four_arm_comparison` / `conclusion_allowed`）全部为 `false`，这是刻意的。`ci_reportable=true` 只是工具在 n≥20 时的标记，工具本身仍不计算区间；本报告的区间由外部脚本计算。
6. **只测了一个 BC 训练种子**：B0 还有 seed1/seed2 checkpoint（离线 val top1 0.6210 / 0.6271），本次未评测，因此没有"BC 训练种子方差"。
7. **无因果归因**：动作分布（idle 45.81%、S/SE 0.25%）只是记录，不能据此断言死因是"不向下躲"或"act_hist 冻结"；需要预注册的消融/诊断实验。
8. **可复现性覆盖有限**：只有 seeds 1000/1001 有 WP04.3 的重复运行可对比；其余 18 个 seed 只跑了一次。
9. **未训练、未采数据、未跑测试套件**：本次没有重跑 `tests/test_v410_student_eval.py`（入口未修改），也没有改动任何 v400/v380/v390 代码。

## 6. 下一建议（按优先级）

1. **B1/B2/B3 loader（正式四臂的前置）**：在**新**入口或经评审的扩展中实现——B1 = probe 家族 + bomb head（模型 731,931 参数，动作 `(cand, bomb)`，需记录 bomb 决策分布）；B2/B3 = spatial 家族（B2 2,600,951 参数 / B3 2,601,464 参数），必须复用各自 `build_model` 的构造参数并 `strict=True` 加载。loader 必须继续 fail-closed：arm 认领、参数量三方一致、`model_class` 匹配、拒绝未知 schema；不得偷偷复用当前 B0 路径。
2. **正式四臂 20 局**：WP04.2 的 B2/B3 补齐后，四臂各跑同一组 dev seeds 1000–1019（同 stage/cap/stop-on-death/obs-v3/2 帧决策），每臂报告 0/20 式过关计数、首死帧 mean/median/区间、动作分布；**先冻结口径与预注册**，再跑，不得事后挑 seed 或 epoch。
3. **过拟合处置先于扩评**：预注册写的是 last epoch（val 0.6280），而 val top1 在 epoch 4–5 附近最高（约 0.68）。若要评"最佳 val epoch"，必须另立预注册，不能在本结果上事后选择。
4. **可复现性补强**：对全部 20 个 seed 重跑一次（约 7 分钟），比对 `policy_input_sha256` / `action_sequence_sha256`；当前只有 2 个 seed 有重复证据。
5. **诊断而非调参**：在不动正式评测口径的前提下，另立预注册做 `act_hist` zero/shuffle 消融与首死 forensics（20 局回放），验证 obs-v3 第三平面是否被使用、首死是否集中在固定阶段；结论只用于下一轮设计。
6. **B0 训练种子方差**：用 B0 seed1/seed2 各跑 20 局（每臂约 7 分钟），判断 0/20 是否是单 checkpoint 的偶然；仍属 dev，不触 confirm。
7. **confirm split 继续封存**：直到四臂 loader、20 局协议与判定口径全部冻结；入口的硬拒绝必须保留。

## 7. 复现清单

```bash
# WSL2 Ubuntu；仓库在 /mnt/d/WorkShop/Touhou_RL
cd /mnt/d/WorkShop/Touhou_RL
sha256sum .tmp_v410_wp04_arms/b0_seed0_last.pt \
          /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_student_eval.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp04_arms/b0_seed0_last.pt \
  --seeds 1000-1019 --stage 0 --difficulty normal --max-frames 19887 --stop-on-death \
  --out Document/Update/Alpha/v4.1.0/evidence/WP04/eval_b0_20.json
# EXIT_CODE=0

/root/v400-venv/bin/python .tmp_v410_wp04_eval20/summarize_eval20.py > .tmp_v410_wp04_eval20/stats_stdout.txt
/root/v400-venv/bin/python .tmp_v410_wp04_eval20/validate_eval20.py
```
