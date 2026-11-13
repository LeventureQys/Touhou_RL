# WP05.4 — WP05.3 DAgger checkpoint 在真实引擎上的 20 局 dev 开发局（与 B0 baseline 并排描述）

- 时间：2026-11-13T07:53:38+08:00（WSL 侧 2026-11-12T23:53:38Z）
- 结论：**WP05.3 的 DAgger checkpoint（B0 + S1 20 % / DAgger 80 %，1,000 steps × batch 128）在 dev seeds 1000–1019 的 20 局真实 `rl_v3` 开发局中 0/20 过关、20/20 首死停止；首死帧 mean 1,574.9 / median 1,375 / min 743 / max 4,048；idle 70.47 %。** 与 B0 baseline（0/20、mean 1,453.25、median 1,374、idle 45.81 %）并排：过关计数相同（0/20），首死帧均值 **+121.65 帧（+8.37 %）**，但逐 seed 中位数差 **−8.5 帧**（9 个 seed 变长 / 11 个变短），idle **+24.65 pp**。**这是描述性 dev 测量：不是门、不是开发成绩、不是 G410-2、不是确认集**；n=20 的 Wilson 过关区间只作描述，不作显著性/效应量/判定。
- 范围：只评测；**未训练、未采数据、未做 β=0.5 轮、未做 50k 全量、未跑 confirm split（60000–60199 被入口硬拒绝）、未改评测工具、未改既有证据**。

## 0. 关键数字一览

| 项 | 值 |
|---|---|
| checkpoint | `.tmp_v410_wp05_train/b0_s0_dagger_step1000.pt`（schema `v410-wp04-bc-arm-checkpoint/1`，arm **B0**，`ProbeNet`，**731,802** 参数） |
| checkpoint 训练配方 | B0 + **S1 20 % / DAgger 80 %**（25,600 / 102,400 samples），固定 1,000 steps × batch 128，seed 0，AdamW 3e-4；`epochs=1` 是 WP04 schema 的占位（不是 1 个 epoch） |
| checkpoint SHA-256 | `e771314d736bf5bd4df0651cab9d9a9780aa067da3a7fbc892cd6ad5cec66824`（与任务给定一致，运行前复核） |
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern` |
| 引擎 SHA-256 | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`（与任务给定一致） |
| 评测入口 | `tools/rl/v410_student_eval.py`（**只读使用，未修改**；SHA-256 `2db2caaf9089c1698cef9b6a682991a7594ebac47ae93244c3d9932591eddddc`，mtime 06:00:33 早于本次运行） |
| 口径 | S1N：`--stage 0` + `normal` + `--max-frames 19887` + `--stop-on-death`，2 帧/决策，obs-v3 真实像素（`consecutive`） |
| dev seeds | 1000–1019（20 个，全部 dev；无 confirm、无语料 seed） |
| episodes / 退出码 | **20 / exit 0**（无 `error`、无 `reset_failed`） |
| stage_clear | **0 / 20 = 0.0 %**；Wilson 95 % **[0.0000, 0.1611]**（仅描述，**不是 G410-2、不是过关率门**） |
| 首死帧 | n=20，**mean 1,574.9**，median 1,375.0，min **743**，max **4,048**，sd 669.54，p25 1,235.75，p75 1,602.0，sum 31,498 |
| 死亡数 | 每局 1.0（`--stop-on-death` 的构造值，不是整局死亡分布）；`lives_end` 全部 = 2.0 |
| stop_reason | `death_stop` × 20（无 `stage_clear`、无 `frame_cap`、无 `engine_exit`） |
| 决策数 | **15,756** |
| 动作 | idle **70.47 %**（11,103/15,756）；E 10.62 %；W 13.67 %；SW 1.99 %；NE 1.69 %；focus 合计 **1.40 %**；**S 0 次**、S+SE 仅 0.05 % |
| 逐局 idle | 63.44 %（seed1004）– 88.44 %（seed1011）；逐局 unique action 4–13 个（并集 13/18） |
| 帧利用 | mean 为 cap 的 **7.92 %**，max 为 cap 的 **20.36 %**，**0 局触到 19,887 帧上限** |
| 离线指标（同 checkpoint，in-sample） | train loss 0.8324 / top1 **0.7549** / macro recall 0.2052（128,000 samples；WP05.3 未读 val/test） |
| 实际墙钟 | 逐局 `wall_s` 合计 1,123.357 s；批跑 23:34:52Z → 23:53:38Z（约 **18 min 46 s**；见 §5 关于并发采集对墙钟的影响） |
| B0 baseline（同口径，WP04.4） | 0/20、mean **1,453.25**、median **1,374.0**、min 815、max 2,386、idle **45.81 %**、决策 14,537 |
| 与 B0 差异 | 过关 0 vs 0；首死帧 mean **+121.65**、median **+1.0**、min −72、max +1,662；idle **+24.65 pp**；决策 +1,219 |

## 1. 交付物与范围

| 文件 | 作用 |
|---|---|
| [`tools/rl/v410_student_eval.py`](../../../../../../tools/rl/v410_student_eval.py) | 只读使用的评测入口（未修改；SHA `2db2caaf…`，mtime 06:00:33 早于本次 07:34:52 启动） |
| [`eval_dagger_20.json`](eval_dagger_20.json) | 本次 20 局的机器可读证据（102,971 B；SHA-256 `5194a913ebd898a984c5402617fbaef1c3cfe150469e9be2d07ec7c41cf9cd7b`；20 条 episode + aggregate + arm_summary + verdict/limits） |
| [`EVAL_DAGGER.md`](EVAL_DAGGER.md) | 本文件：命令、退出码、20 局表、汇总、与 baseline 对比、限制、下一建议 |
| `.tmp_v410_wp05_eval20/` | 运行目录（`run_eval20.sh`、`meta.txt`、`run.log`、`exit_code.txt`、`stats.json`、`stats_stdout.txt`、`derived.json`、`validate_stdout.txt`、`summarize_eval20.py`、`validate_eval20.py`、`derived.py`、`check_md.py`、`audit_whitelist.sh`），**不入库** |

本次只新增 `eval_dagger_20.json` 与 `EVAL_DAGGER.md`，运行产物只写 `.tmp_v410_wp05_eval20/`（被 `.gitignore` 的 `.tmp_*/` 忽略）。**未修改** `tools/rl/v410_student_eval.py`、`touhou_rl/v400|v380|v390`、`tools/rl/v400_*`、`tests/`、`third_party/`、`.tmp_v410_wp05_train/`、`state/journal/budget/assets`、WP03/WP04/WP05 既有证据（`eval_b0_20.json` SHA `2cb3e0f7…` 与 mtime 05:12:06 未变）。未 commit / push，未删除/覆盖任何旧文件。

> 并发写入说明：本次运行期间 `WP05` 的 50k 首轮采集（`tools/rl/v410_dagger_collect.py`，seeds 2000+，PID 2459296/2459318）仍在运行。它只影响本机的墙钟时间（逐局 28–139 s，对比 B0 的 14–31 s），**不影响帧数**：本报告的两个重叠 seed 与 WP05.3 的 2 局 smoke 逐字节一致（§4.5）。

## 2. 实际命令与退出码

在 WSL2 中从 Windows 侧工作区运行（引擎与数据在 `/root`，仓库在 `/mnt/d/WorkShop/Touhou_RL`）：

```bash
cd /mnt/d/WorkShop/Touhou_RL
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_student_eval.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp05_train/b0_s0_dagger_step1000.pt \
  --seeds 1000-1019 --stage 0 --difficulty normal --max-frames 19887 --stop-on-death \
  --out Document/Update/Alpha/v4.1.0/evidence/WP05/eval_dagger_20.json
# wrote Document/Update/Alpha/v4.1.0/evidence/WP05/eval_dagger_20.json (exit 0)
# EXIT_CODE=0
```

- 运行包装脚本：[`.tmp_v410_wp05_eval20/run_eval20.sh`](../../../../../../.tmp_v410_wp05_eval20/run_eval20.sh)；日志 `run.log`，元数据 `meta.txt`（`START_UTC=2026-11-12T23:34:52Z`、`END_UTC=2026-11-12T23:53:38Z`、`EXIT_CODE=0`），`exit_code.txt` = `0`。
- 运行前复核两个 SHA（与任务给定一致）：

```bash
sha256sum .tmp_v410_wp05_train/b0_s0_dagger_step1000.pt \
  /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern
# e771314d736bf5bd4df0651cab9d9a9780aa067da3a7fbc892cd6ad5cec66824  b0_s0_dagger_step1000.pt
# 691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6  th08-modern
```

- 独立复核（只读，不启动引擎）：用入口自身的 `validate_episode_record()` 重验 20 条记录，并检查 seed 序列、`error=null`、`frames ≤ 19887`、`exit_code=0`、checkpoint/engine SHA 与任务给定一致：

```bash
/root/v400-venv/bin/python .tmp_v410_wp05_eval20/validate_eval20.py
# VALIDATED: 20/20 episode records pass ...; seeds 1000-1019; no error/reset_failed;
#   frames <= 19887; exit_code 0; checkpoint SHA e771314d...; engine SHA 691bfd77...;
#   not_a_gate/not_a_development_score/not_g410_2 all true
# EXIT_CODE=0
```

- 所有统计数字由脚本从 JSON 计算（`.tmp_v410_wp05_eval20/summarize_eval20.py` → `stats.json`；`derived.py` → `derived.json`），本文件只转抄，不手写统计量：

```bash
/root/v400-venv/bin/python .tmp_v410_wp05_eval20/summarize_eval20.py > .tmp_v410_wp05_eval20/stats_stdout.txt   # EXIT_CODE=0
/root/v400-venv/bin/python .tmp_v410_wp05_eval20/derived.py > .tmp_v410_wp05_eval20/derived.json              # EXIT_CODE=0
```

## 3. 20 局逐局表

`frames` 是 `--stop-on-death` 口径下的**首死帧数**；`deaths` 恒为 1.0（构造值，不列）；`lives_end=2` 表示 3 残机开局后的第 1 次死亡；全部 `stage_start=0 → stage_end=0`，即无一局离开 Stage 1；`Δ` = 本 checkpoint 帧数 − B0 baseline（WP04.4 同 seed）。

| seed | frames | steps | stop_reason | clear | score | lives_end | bombs | idle% | unique | Δ vs B0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 1609 | 805 | `death_stop` | false | 13483 | 2 | 0 | 65.22% | 11 | **+211** |
| 1001 | 1200 | 600 | `death_stop` | false | 9689 | 2 | 0 | 74.67% | 7 | **+382** |
| 1002 | 1183 | 592 | `death_stop` | false | 11991 | 2 | 0 | 78.21% | 7 | −68 |
| 1003 | 1343 | 672 | `death_stop` | false | 6938 | 2 | 0 | 77.98% | 7 | **+528** |
| 1004 | 2292 | 1146 | `death_stop` | false | 49645 | 2 | 0 | 63.44% | 10 | −94 |
| 1005 | 1375 | 688 | `death_stop` | false | 14401 | 2 | 0 | 64.68% | 7 | −4 |
| 1006 | 1375 | 688 | `death_stop` | false | 15457 | 2 | 0 | 67.15% | 7 | +6 |
| 1007 | 1347 | 674 | `death_stop` | false | 17609 | 2 | 0 | 73.00% | 8 | −317 |
| 1008 | 1364 | 682 | `death_stop` | false | 17817 | 2 | 0 | 65.98% | 7 | −112 |
| 1009 | 1523 | 762 | `death_stop` | false | 12608 | 2 | 0 | 65.62% | 8 | +66 |
| 1010 | 2050 | 1025 | `death_stop` | false | 37249 | 2 | 0 | 68.20% | 13 | **+836** |
| 1011 | 743 | 372 | `death_stop` | false | 2874 | 2 | 0 | 88.44% | 4 | **−481** |
| 1012 | 1375 | 688 | `death_stop` | false | 15602 | 2 | 0 | 69.77% | 7 | +161 |
| 1013 | 1189 | 595 | `death_stop` | false | 5568 | 2 | 0 | 74.29% | 9 | **−1050** |
| 1014 | **4048** | 2024 | `death_stop` | false | 68202 | 2 | 0 | 74.85% | 8 | **+2792** |
| 1015 | 1348 | 674 | `death_stop` | false | 7148 | 2 | 0 | 78.49% | 7 | −13 |
| 1016 | 1805 | 903 | `death_stop` | false | 31109 | 2 | 0 | 65.34% | 7 | +319 |
| 1017 | 1557 | 779 | `death_stop` | false | 12279 | 2 | 0 | 69.06% | 9 | −337 |
| 1018 | 1191 | 596 | `death_stop` | false | 8998 | 2 | 0 | 72.65% | 7 | −40 |
| 1019 | 1581 | 791 | `death_stop` | false | 25630 | 2 | 0 | 64.73% | 7 | −352 |

- 最大局 seed1014 **4,048 帧**（cap 的 20.36 %）；最小局 seed1011 **743 帧**（idle 88.44 %）。
- 全部 `bombs_used=0`：B0 没有 bomb 头，部署动作空间只有 `(cand, bomb=0)`（入口口径固定）。

## 4. 汇总（全部由 `eval_dagger_20.json` 计算）

### 4.1 过关与首死

| 指标 | 值 |
|---|---|
| episodes | 20（seeds 1000–1019） |
| stage_clear | **0 / 20 = 0.0 %** |
| Wilson 95 % 过关率区间 | **[0.0000, 0.1611]**（0/20；仅描述性，**不是 G410-2**，也不是正式确认集） |
| 首死帧 mean / median | **1,574.9 / 1,375.0** |
| 首死帧 min / max | **743 / 4,048** |
| 首死帧 sd / p25 / p75 | 669.54 / 1,235.75 / 1,602.0 |
| 首死帧 sum | 31,498 |
| 死亡数 total / mean | 20 / 1.0（`stop-on-death` 构造） |
| stop_reason | `death_stop: 20` |
| 触到 19,887 帧上限的局 | 0 |
| mean / max 帧占 cap | 7.92 % / 20.36 % |
| 决策总数 | 15,756 |

### 4.2 动作分布（20 局合计 15,756 次决策，argmax，无采样）

| 动作 id | 名称 | 次数 | 占比 | B0 占比 | Δ |
|---|---|---|---|---|---|
| 0 | idle | **11,103** | **70.47%** | 45.81% | **+24.65 pp** |
| 1 | N | 9 | 0.06% | 0.28% | −0.22 pp |
| 2 | NE | 267 | 1.69% | 5.12% | −3.43 pp |
| 3 | E | 1,674 | 10.62% | 14.69% | −4.06 pp |
| 4 | SE | 8 | 0.05% | 0.14% | −0.09 pp |
| 5 | S | **0** | **0.00%** | 0.12% | −0.12 pp |
| 6 | SW | 314 | 1.99% | 13.43% | **−11.44 pp** |
| 7 | W | 2,154 | 13.67% | 13.74% | −0.07 pp |
| 8 | NW | 6 | 0.04% | 0.12% | −0.09 pp |
| 9 | focus-idle | 0 | 0.00% | 0.00% | 0.00 pp |
| 10 | focus-N | 0 | 0.00% | 0.01% | −0.01 pp |
| 11 | focus-NE | 86 | 0.55% | 1.61% | −1.06 pp |
| 12 | focus-E | 122 | 0.77% | 4.29% | −3.51 pp |
| 13 | focus-SE | 3 | 0.02% | 0.03% | −0.02 pp |
| 14 | focus-S | 0 | 0.00% | 0.00% | 0.00 pp |
| 15 | focus-SW | 9 | 0.06% | 0.50% | −0.45 pp |
| 16 | focus-W | 1 | 0.01% | 0.08% | −0.07 pp |
| 17 | focus-NW | 0 | 0.00% | 0.03% | −0.03 pp |

- idle **70.47 %**（逐局均值 71.09 %）；水平横移（E+W）24.30 %；focus 动作合计 **1.40 %**（B0 6.54 %）；向下（SE+S+SW）2.04 %，其中纯 S **0 次**、S+SE 仅 0.05 %。
- 逐局 idle 63.44 %（seed1004）–88.44 %（seed1011）；逐局 unique action 4–13 个（并集 13/18；`S`、`focus-idle`、`focus-N`、`focus-S`、`focus-NW` 从未被选中）。
- 只记录分布，**不据此推断死因**。一个纯描述性背景：WP05.3 的 DAgger 标签（teacher chosen plan）多数类基线 top1 是 **0.6860**（[`TRAIN.md`](TRAIN.md) §5），即教师标签本身高度偏向单一动作；学生把这个边际复制成 70.47 % idle 与"teacher 数据占比 80 %"在方向上是相容的。**这只是描述，不是因果归因，也不是"策略退化"的证明**。

### 4.3 与 B0 baseline 并排（同引擎、同 S1N 口径、同 dev seeds）

| 指标 | WP05.3 DAgger checkpoint | B0 baseline（WP04.4） | 差异 |
|---|---|---|---|
| checkpoint | `b0_s0_dagger_step1000.pt`（SHA `e771314d…`） | `b0_seed0_last.pt`（SHA `3bc0ec44…`） | 不同 checkpoint |
| 训练数据 / 预算 | S1 20 % + DAgger 80 %，1,000 steps × 128 | 仅 D410-S1，20 epochs × 363 steps | 不同配方 |
| episodes | 20 | 20 | 相同 |
| stage_clear | **0 / 20 = 0.0 %** | **0 / 20 = 0.0 %** | 相同（0 vs 0） |
| 首死帧 mean | **1,574.9** | **1,453.25** | **+121.65（+8.37 %）** |
| 首死帧 median | **1,375.0** | **1,374.0** | **+1.0** |
| 首死帧 min / max | 743 / 4,048 | 815 / 2,386 | −72 / **+1,662** |
| 首死帧 sd / p25 / p75 | 669.54 / 1,235.75 / 1,602.0 | 403.22 / 1,225.75 / 1,619.5 | 分布更宽 |
| idle 比例 | **70.47 %**（逐局均值 71.09 %） | **45.81 %**（逐局均值 46.12 %） | **+24.65 pp** |
| 决策总数 | 15,756 | 14,537 | +1,219 |
| stop_reason | `death_stop` × 20 | `death_stop` × 20 | 相同 |
| 平均 score（描述性） | 19,214.85 | 10,941.95 | +8,272.90 |
| 墙钟合计 | 1,123.357 s | 417.226 s | 不可比（见 §5） |

### 4.4 逐 seed 差异分布（描述性，不构成效应估计）

- 9 个 seed 首死帧变长、11 个变短、0 个持平；逐 seed Δ 的 **median = −8.5 帧**（典型 seed 略差），**mean = +121.65**（被少数长局拉高）。
- 拉升主要来自 seed1014 **+2,792**、seed1010 +836、seed1003 +528、seed1001 +382、seed1016 +319；拖低主要来自 seed1013 **−1,050**、seed1011 −481、seed1019 −352、seed1017 −337、seed1007 −317。
- 逐 seed Δ 的 sd = 743.36，范围 −1,050 … +2,792。**均值为正、中位数为负**这一并存本身就是描述：两个 checkpoint 的差距在 20 个 seed 上方向不一致，**没有做显著性检验、没有预注册判定规则、没有效应量**。

### 4.5 与 WP05.3 的 2 局 smoke 逐字节一致（可复现性）

WP05.3 用同一 checkpoint、同引擎、同 S1N 口径跑过 seeds 1000/1001（[`.tmp_v410_wp05_train/eval_dev.json`](../../../../../../.tmp_v410_wp05_train/eval_dev.json)）。本次 20 局中重叠的两个 seed **全部相同**——帧数、steps、score、动作直方图、`action_sequence_sha256`、`policy_input_sha256` 都一致：

| seed | frames smoke / 本次 | steps | score | stop_reason | 直方图 | `policy_input_sha256` | `action_sequence_sha256` |
|---|---|---|---|---|---|---|---|
| 1000 | 1,609 / **1,609** | 805 / 805 | 13,483 / 13,483 | `death_stop` / `death_stop` | 相同 | **相同** | **相同** |
| 1001 | 1,200 / **1,200** | 600 / 600 | 9,689 / 9,689 | `death_stop` / `death_stop` | 相同 | **相同** | **相同** |

这说明本次 runner 与 WP05.3 的接线没有漂移，且这两个 seed 上**策略输入与动作序列是比特级可复现**的；其余 18 个 seed 只跑了一次，**不能**外推为"20 个 seed 全部可复现"。

## 5. 限制（明确不声称）

1. **不是门、不是开发成绩、不是 G410-2、不是确认集**：只用 dev seeds 1000–1019；confirm 60000–60199 与 D410-S1 语料 seed 1–20 被入口硬拒绝，本次未触达。Wilson 区间 [0.0000, 0.1611] 只是 0/20 的描述性区间，**不是过关率上限**，不能写成任何门结论。JSON 的 `verdict` 五项（`is_a_gate` / `is_a_development_score` / `is_a_pass_rate` / `is_a_four_arm_comparison` / `conclusion_allowed`）全部为 `false`，`not_a_gate` / `not_a_development_score` / `not_g410_2` 全部为 `true`。
2. **两个 checkpoint 的配方不同，差异不能归因于 DAgger**：B0 是纯 D410-S1 BC（20 epochs × 363 steps = 7,260 steps），本次是 S1 20 % + DAgger 80 % 的 1,000 steps × 128。数据、步数、优化轨迹都不同，**+121.65 帧不是"DAgger 带来的增益"的估计**；要比较配方需冻结同一预算并预注册。
3. **`--stop-on-death` 口径**：每局在首死处停止，`deaths` 恒为 1.0、`lives_end` 恒为 2.0，因此**没有整局死亡分布、没有 game-over 口径、没有 score/PSN/人类对齐指标**；有信息量的量是**首死帧数**。
4. **JSON 自带标签是工具写死的**：`eval_dagger_20.json` 里 `kind="dev-smoke"`、`wp="WP04.3"`、`schema="v410-student-eval-smoke/1"`、`limits` 仍写 "This is a SMOKE"，因为入口是**只读使用、未修改**。本文件是 WP05.4 的 20 局 dev 运行，标签未改。`ci_reportable=true` 只是工具在 n≥20 时的标记，工具本身仍不计算区间；本报告的区间由外部脚本计算。
5. **入口版本与 baseline 的入口版本不同**：当前 `tools/rl/v410_student_eval.py` SHA `2db2caaf…`，WP04.4 记录的是 `c63100ea…`（之后为 WP04.5 四臂扩展改过，不是本任务改的）。baseline 由旧版入口产出；不过重叠 seeds 1000/1001 的帧数与 WP05.3 smoke 完全一致，说明这次版本差**没有**改变这两个 seed 的闭环结果，但**没有**对其余 18 个 seed 做版本交叉验证。
6. **只有一个 checkpoint、一个训练 seed、1,000 步**：没有 seed 方差、没有收敛声明；`epochs=1` 只是 WP04 schema 占位（实际是 1,000 steps 的 last-step 权重）。
7. **idle 70.47 % 只记录不解释**：它可能与 DAgger 教师标签的多数类先验（in-sample top1 0.6860）一致，但本次没有做消融/诊断，**不能**据此断言"策略只会发呆"或"死因是 idle"；也不能据此排序两个 checkpoint 的"好坏"。
8. **死亡通道未被 DAgger 数据覆盖**：WP05.3 的 21,000 条 DAgger 决策里 `verified_dead=0`（37 条 `unknown` 全是 model-mismatch），本次也没有死亡标签可用；闭环保真但**死亡相关学习信号仍缺**。
9. **墙钟不可比**：本次运行与 50k 采集并发（逐局 28–139 s），B0 baseline 是空闲时跑的（14–31 s）；帧数不受影响（§4.5），但耗时数字不能用于吞吐比较。
10. **未重跑测试套件**：入口未修改，本次没有重跑 `tests/test_v410_student_eval.py`；也没有改动任何 v400/v380/v390 代码、没有训练、没有采数据。

## 6. 下一建议（按优先级）

1. **50k 全量首轮 → 同一固定预算复跑 20 局**：用 WP05.1 的 50k DAgger 数据（当前 7 个 shard 已完成，仍在补 seeds 2020+），保持**完全相同的固定预算**（1,000 steps × 128、seed 0、S1 20 %），再跑同一组 dev seeds 1000–1019；本报告是这条"数据量–效果"曲线上的 **21k / 1,000-step 参考点**。只有预算与口径冻结后才允许并排。
2. **β = 0.5 第二轮**：同一批 shard 采集时把教师注入率降到 0.5，训练同一预算，比较 student–teacher disagreement、离线 top1/macroR 与闭环 idle/首死帧；先离线，再决定是否值得跑 dev 20 局。
3. **多 seed / 多轮**：β=1.0 → β=0.5 循环 2–3 轮，每轮 1,000 steps，训练 seed 0/1/2，记录 seed 方差；只有 ≥20 dev episodes 且**预注册判据**时才允许写"效果"。
4. **拉长预算**：1,000 steps 的 in-sample top1 0.7549（val 未读）；在预注册后把预算提到 5k–10k，看 idle 是否下降、首死帧是否同向变化。不得在本结果上事后挑 epoch/step。
5. **死亡通道覆盖**：下一轮 DAgger 采集需要覆盖 `verified_dead`（当前为 0），否则"死亡前状态"的学习信号只能靠单测模拟。
6. **confirm split 继续封存**：直到四臂 loader、20 局协议与判定口径全部冻结；入口硬拒绝必须保留。**本报告不构成 G410-2 的任何输入。**

## 7. 复现清单

```bash
# WSL2 Ubuntu；仓库在 /mnt/d/WorkShop/Touhou_RL
cd /mnt/d/WorkShop/Touhou_RL
sha256sum .tmp_v410_wp05_train/b0_s0_dagger_step1000.pt \
          /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_student_eval.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp05_train/b0_s0_dagger_step1000.pt \
  --seeds 1000-1019 --stage 0 --difficulty normal --max-frames 19887 --stop-on-death \
  --out Document/Update/Alpha/v4.1.0/evidence/WP05/eval_dagger_20.json
# EXIT_CODE=0

/root/v400-venv/bin/python .tmp_v410_wp05_eval20/summarize_eval20.py > .tmp_v410_wp05_eval20/stats_stdout.txt  # EXIT_CODE=0
/root/v400-venv/bin/python .tmp_v410_wp05_eval20/validate_eval20.py                                     # EXIT_CODE=0
```
