# WP05.7 — β=0.5 第 2 轮 52,500 decisions 同预算重训 B0 + 20 dev episodes（与 β=1.0 21k/50k 并排）

- 时间：2026-11-13T13:48:11+08:00（训练 13:06:32–13:09:56，dev 评测 13:41:26–13:48:11）
- 结论：**已完成（描述性，不是门）**。按实际枚举，`.tmp_v410_wp05_round2/shards/` 下有
  **15 个 `complete=true`** 的 β=0.5 shard（每个 3,500 decisions / 7,000 frames），合计
  **52,500 decisions / 105,000 frames**；15/15 用 WP05.2 CLI 转成 `labels_v410`（全部 exit 0，
  labels/obs SHA-256 校验通过，行数 52,500）。B0（731,802 参数）在**与 WP05.3 完全相同的预算**下
  重训：`steps=1000 × batch=128`、seed 0、S1 20 % / DAgger 80 %（实际 25,600 / 102,400 样本）、
  AdamW lr 3e-4 / wd 1e-4 / clip 5.0、只读 D410-S1 train seeds 1-14；训练 loss **0.9564** /
  top1 **0.7164** / macroR **0.1406**，**776.3 samples/s**，checkpoint SHA
  `a11ba0b7…ede25d5b`。20 个 dev episodes（seeds 1000-1019，S1N stage0/normal/cap19887/
  stop-on-death）在真实 `rl_v3` 引擎跑满：**0/20 通关、20/20 `death_stop`；frames mean 1,339.95、
  median 1,321.0、idle 71.07 %**。与 β=1.0 并排：21k（mean 1,574.9 / median 1,375 / idle 70.47 %）
  与 50k（mean 1,643.25 / median 1,580.5 / idle 72.06 %）都是 **0/20**，本轮 mean **低 234.95 帧
  （−14.92 %）** / **低 303.30 帧（−18.46 %）**，median 分别 **−54.0** / **−259.5**，idle 分别
  **+0.60 pp** / **−1.00 pp**。**本对比是描述性的：无区间、无显著性检验、不写门、不是 G410-2；
  且两轮的注入 β 与 rollout 学生同时变了（见 §7.3），不能读成 β 的因果效应。**
- 证据：[`train_beta05.json`](train_beta05.json)（84,690 B）；运行目录
  `.tmp_v410_wp05_train_beta05/`（labels 15 个 / checkpoint / `train_run.json` /
  `eval_dev20.json` / `dryrun_eval_attempt2.json` / `logs/`，被 `.gitignore` 的 `.tmp_*/` 覆盖，不入库）。

## 0. 关键数字一览

| 项 | 值 |
|---|---|
| complete shards | **15**（seed2025-2039），每个 3,500 decisions / 7,000 frames |
| 聚合 decisions / frames | **52,500 / 105,000**（= 50k 目标的 105 %） |
| 采集注入率（shard.json） | injected **26,295 / 52,500 = 0.500857**（β=0.5；教师可用 52,500/52,500 = 1.0） |
| 师生一致 / 不一致 | **41,819（79.66 %） / 10,681（20.34 %）** |
| rollout 学生 | `6ebe4ea9…69b8086`（= WP05.5 50k B0 step1000，非首轮 baseline） |
| 生存标签（teacher-plan） | alive **52,068**、dead **21**、unknown **411**（全部 model_mismatch） |
| query 覆盖 | 52,500 / 945,000 = **5.5556 %**（每 shard 3,500/3,500） |
| 标签转换 | WP05.2 CLI ×15，**15/15 exit 0**；单 shard 6–9 s |
| S1 训练集 | 46,397 rows（seeds 1-14，train split only；val/test 从未打开） |
| 混合 | 名义 **S1 20 % / DAgger 80 %**；实际 **25,600 / 102,400 = 20.000 % / 80.000 %**，每样本权重 1.0 |
| 模型 | B0 `ProbeNet`，**731,802** 参数，chosen one-hot CE |
| 训练预算 | **1,000 steps × 128 = 128,000 samples**（与 WP05.3/WP05.5 逐项相同）；seed 0 |
| checkpoint | `.tmp_v410_wp05_train_beta05/b0_s0_dagger_step1000.pt`，2,941,187 B，SHA-256 `a11ba0b74547515023d98145c915e0afb7d5eaafc3bea8d9d11b2429ede25d5b` |
| 训练指标（in-sample） | loss **0.9564**、top1 **0.7164**、macroR **0.1406**；首 step 2.9003 → 末 step 0.6633 |
| 吞吐 / GPU | **776.3 samples/s / 6.06 steps/s**；RTX 4070 SUPER（峰值 252.11 MiB allocated） |
| 训练 wall | 进程 204 s（payload `elapsed_seconds` 193.79 s）；**同机 G410-1 chunk3 6 并发教师**，load 45–59 |
| dev 口径 | S1N：`--stage 0` + `normal` + `--max-frames 19887` + `--stop-on-death` + `--obs-window consecutive` |
| dev 结果（n=20） | **0/20 通关**；frames mean **1,339.95**、median **1,321.0**、min 1,101、max 1,747、sd 122.88；idle **71.07 %**；unique actions 6.7 |
| β=1.0 21k 参照 | 0/20；mean 1,574.9、median 1,375.0、min 743、max 4,048、sd 669.54；idle 70.47 %；unique 7.85 |
| β=1.0 50k 参照 | 0/20；mean 1,643.25、median 1,580.5、min 1,261、max 2,371、sd 315.64；idle 72.06 %；unique 8.8 |
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，SHA-256 `691bfd77…a568ef6` |

## 1. 交付物

| 文件 | 作用 |
|---|---|
| [`train_beta05.json`](train_beta05.json) | 本轮机器可读证据：输入 shards/转换逐 shard 退出码、标签聚合、混合、预算、训练指标与吞吐、checkpoint、commands+exit_code、20 局汇总与逐局记录、与 β1 21k/50k 的并排与差值、两次评测尝试、scope/not_claimed/limits |
| `EVAL_BETA05.md` | 本文件：命令与退出码、结果表、β1 对比、争用口径、限制 |
| `.tmp_v410_wp05_train_beta05/` | 运行目录（不入库）：`labels/dagger-s0d1seed*/{labels_v410.jsonl,labels_v410.summary.json}`、`b0_s0_dagger_step1000.pt`、`train_run.json`、`dryrun_eval.json`、`dryrun_eval_attempt2.json`、`eval_dev20.json`、`logs/{convert_labels.sh,train.log,run_eval.sh,run_eval_attempt2.sh,eval_dev20_attempt1_aborted.log,eval_dev20_attempt2.log,build_evidence.py}` |

只读使用、**未改动**：`touhou_rl/v410/dagger/train.py`（WP05.3 CLI）、`touhou_rl/v410/dagger/labels.py`（WP05.2 CLI）、
`tools/rl/v410_student_eval.py`、WP05.0/05.6 shards、WP03 S1 audit
[`s1c_full_audit.json`](../WP03/s1c_full_audit.json)、WP05.5 [`train_50k.json`](train_50k.json) /
[`EVAL_50K.md`](EVAL_50K.md)、WP05.3 [`train_smoke.json`](train_smoke.json) /
[`eval_dagger_20.json`](eval_dagger_20.json)。**未采集任何数据、未跑确认集、未做 G410-2。**

## 2. 命令与退出码（逐步实测）

### 2.1 枚举 complete shards

```powershell
foreach ($i in 2025..2039) {
  $j = Get-Content ".tmp_v410_wp05_round2\shards\dagger-s0d1seed$i\shard.json" -Raw | ConvertFrom-Json
  "$i complete=$($j.complete) decisions=$($j.decisions) frames=$($j.frames) beta=$($j.beta)"
}
# 15/15 complete=True decisions=3500 frames=7000 beta=0.5
# EXIT_CODE=0；合计 decisions=52500 frames=105000
```

判据与 WP05.5 相同：只取 `complete=true` 的 shard，本轮 15 个全部达标（52,500 ≥ 50,000），
**没有伪造、没有补数**；15 个 shard 的 `beta` 全为 **0.5**、`stage=0`、`difficulty=normal`、
`stop_reason=decisions`、`checkpoint.sha256` 全为 `6ebe4ea9…`（同一个 rollout 学生）。

### 2.2 标签转换（WP05.2 CLI，15 次）

```bash
bash .tmp_v410_wp05_train_beta05/logs/convert_labels.sh
#   /root/v400-venv/bin/python -m touhou_rl.v410.dagger.labels \
#     --shard .tmp_v410_wp05_round2/shards/dagger-s0d1seed<seed> \
#     --out   .tmp_v410_wp05_train_beta05/labels/dagger-s0d1seed<seed> \
#     --tolerance 8.0
# 2025..2039 全部 exit=0（6–9 s/shard；日志 logs/convert_exit_codes.txt）
# CONVERT_ALL_DONE fail=0
```

每次输出形如 `decisions=3500 plans=3500 alive=3494 dead=0 unknown=6 model_mismatch=6
incomplete=0 query_verified=3500/63000 query_coverage=0.055556 injected=1730 disagreement=696
tolerance=8.0 horizon=32`，并写出 `labels_v410.jsonl` + `labels_v410.summary.json`。
转换后逐 shard 复核：`labels` 行数 = `shard.json.decisions`、`checks.labels_sha256_verified` /
`checks.obs_sha256_verified` 均 true；训练入口 `train.py` 读入时再校验 SHA-256 并逐行解析。

### 2.3 训练（WSL2，CUDA，与 WP05.3/WP05.5 同预算）

```bash
bash .tmp_v410_wp05_train_beta05/logs/run_train.sh
#   /root/v400-venv/bin/python -m touhou_rl.v410.dagger.train train \
#     --steps 1000 --batch-size 128 --seed 0 --s1-share 0.2 --device auto \
#     --shard-seeds 2025-2039 \
#     --shards-root .tmp_v410_wp05_round2/shards \
#     --labels-dir .tmp_v410_wp05_train_beta05/labels \
#     --workdir .tmp_v410_wp05_train_beta05 \
#     --out .tmp_v410_wp05_train_beta05/train_run.json
# [WP05.3] DAgger: 52500 rows with a chosen target from 15 shards
# [WP05.3] D410-S1 train: 46397 rows from seeds 1-14
# dagger_train: steps=1000 samples=128000 s1=25600 dagger=102400
#   loss=0.9564 top1=0.7164 macroR=0.1406 776.3 samples/s
#   checkpoint=.tmp_v410_wp05_train_beta05/b0_s0_dagger_step1000.pt
# TRAIN_EXIT=0 wall_seconds=204
```

### 2.4 dev 接线 dry run（不是证据）

```bash
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_student_eval.py --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp05_train_beta05/b0_s0_dagger_step1000.pt \
  --seeds 1000 --stage 0 --difficulty normal --max-frames 60 --stop-on-death \
  --out .tmp_v410_wp05_train_beta05/dryrun_eval_attempt2.json
# DRYRUN_EXIT=0（两次评测前各跑一次，均 exit 0）
```

### 2.5 真实 20 episodes —— 两次尝试，只用第 2 次

命令固定为脚本文件（避免 PowerShell→WSL 引号转义）：

```bash
# 第 1 次：bash .tmp_v410_wp05_train_beta05/logs/run_eval.sh
# 第 2 次：bash .tmp_v410_wp05_train_beta05/logs/run_eval_attempt2.sh
timeout-friendly 形式：
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_student_eval.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp05_train_beta05/b0_s0_dagger_step1000.pt \
  --seeds 1000-1019 --stage 0 --difficulty normal --max-frames 19887 --stop-on-death \
  --out .tmp_v410_wp05_train_beta05/eval_dev20.json
```

| 尝试 | 时间 | 结果 | 退出码 | 是否进汇总 |
|---|---|---|---|---|
| attempt 1 | 13:10:15–13:24:47 | **4/20 被 SIGTERM 中止**（wall 855 s） | **143**（128+15） | **否** |
| attempt 2 | 13:41:26–13:48:11 | **20/20 完整**，`eval_dev20.json` exit_code=0 | **0** | **是** |

- **attempt 1 中止原因（如实记录）**：G410-1 teacher chunk3 批次（`v410_teacher_fast20.py`，
  6 个并发教师实例 × 25 seeds，13:03:13 启动）占满同一台 20 核机器（load 45–74），dev 评测从
  seed1004 起被**饿死**：单局 wall 从正常的 ~19 s 涨到 84 / 228 / 178 / 134 s，
  由上级代理在 4/20 处主动终止，避免无界挂起。**中止不影响训练证据，checkpoint 与训练指标未改。**
  attempt1 日志与退出码保留在 `logs/eval_dev20_attempt1_aborted.log` 与
  `logs/eval_exit_code_attempt1_aborted.txt`，其 4 局（1404 / 1453 / 1325 / 1294 帧）**不进任何统计**。
- **attempt 2 干净重跑**：脚本先 `kill -0 367018` 轮询等 chunk3 进程退出（**等待 945 s**，
  13:25:13 → 13:41:26，chunk3 期间 25/25 seed 落盘），随后空载开跑：等待结束时 load 5.86、
  结束时 load 2.50，20 局逐局 wall 16.4–23.5 s（合计 382.40 s，进程 wall 396 s），
  与 β1-50k 那轮的 467.16 s 同量级。
- **可复现性交叉核对**：attempt1 与 attempt2 的**同名 seed 逐局完全一致**——
  seed1000 frames 1404/steps 702/score 10877、1001 1453/727/18698、1002 1325/663/9447、
  1003 1294/647/25404，四局全等。说明争用只改变 wall，**不改变策略行为**（同 seed 确定性回放）。
- 20 局全部 `harness_role="env"`、`error=null`、`stage_start=0 → stage_end=0`、`lives_end=2.0`；
  `deaths=1.0` 是 `--stop-on-death` 的构造值。

### 2.6 证据构建与自检

```bash
/root/v400-venv/bin/python .tmp_v410_wp05_train_beta05/logs/build_evidence.py
# wrote .../evidence/WP05/train_beta05.json (84690 B)
# EXIT=0
```

`build_evidence.py` 只做聚合与交叉核对，**fail-closed（不一致即 exit 3）**：eval `exit_code=0`、
episode 数 = 20、seeds 严格等于 1000-1019、逐局 `error=null`、逐局 `checkpoint_sha256` 与训练
checkpoint 一致、逐局 `engine_sha256` 与头部一致、mean/median/idle 独立重算后与 evaluator 的
`arm_summary`/`aggregate` 对照、每 shard 的 `labels_sha256_verified`/`obs_sha256_verified` 为 true
且 labels 行数 = shard decisions、聚合 decisions = 52,500。任一不满足即拒绝写出。

## 3. shards 与标签转换（15 个，WP05.2 规则）

每个 shard 的 `labels.jsonl` 行数、`shard.json.decisions`、obs 三平面行数与 SHA-256 全部一致
（summary 中 `checks.labels_sha256_verified` / `checks.obs_sha256_verified` 均 true）。

| shard | decisions | frames | alive | dead | unknown | injected | agreement | disagreement | labels SHA-256（前 8） |
|---|---|---|---|---|---|---|---|---|---|
| dagger-s0d1seed2025 | 3,500 | 7,000 | 3,494 | 0 | 6 | 1,730 | 2,804 | 696 | `170775f2` |
| dagger-s0d1seed2026 | 3,500 | 7,000 | 3,451 | 2 | 47 | 1,759 | 2,776 | 724 | `d4bdc39f` |
| dagger-s0d1seed2027 | 3,500 | 7,000 | 3,482 | 0 | 18 | 1,754 | 2,703 | 797 | `8800fcd0` |
| dagger-s0d1seed2028 | 3,500 | 7,000 | 3,490 | 0 | 10 | 1,787 | 2,703 | 797 | `f9bf46fd` |
| dagger-s0d1seed2029 | 3,500 | 7,000 | 3,490 | 0 | 10 | 1,758 | 2,805 | 695 | `b9365376` |
| dagger-s0d1seed2030 | 3,500 | 7,000 | 3,452 | 5 | 43 | 1,739 | 2,763 | 737 | `d5221680` |
| dagger-s0d1seed2031 | 3,500 | 7,000 | 3,493 | 0 | 7 | 1,751 | 2,872 | 628 | `a44029cf` |
| dagger-s0d1seed2032 | 3,500 | 7,000 | 3,489 | 0 | 11 | 1,764 | 2,704 | 796 | `c8c853e4` |
| dagger-s0d1seed2033 | 3,500 | 7,000 | 3,466 | 3 | 31 | 1,721 | 2,890 | 610 | `868339ba` |
| dagger-s0d1seed2034 | 3,500 | 7,000 | 3,491 | 0 | 9 | 1,730 | 2,809 | 691 | `e00db162` |
| dagger-s0d1seed2035 | 3,500 | 7,000 | 3,426 | 3 | 71 | 1,826 | 2,905 | 595 | `609f84bf` |
| dagger-s0d1seed2036 | 3,500 | 7,000 | 3,491 | 2 | 7 | 1,765 | 2,734 | 766 | `2b06fc3b` |
| dagger-s0d1seed2037 | 3,500 | 7,000 | 3,458 | 3 | 39 | 1,746 | 2,904 | 596 | `f45bee9a` |
| dagger-s0d1seed2038 | 3,500 | 7,000 | 3,401 | 3 | 96 | 1,755 | 2,695 | 805 | `414a20cd` |
| dagger-s0d1seed2039 | 3,500 | 7,000 | 3,494 | 0 | 6 | 1,710 | 2,752 | 748 | `e50fb7aa` |
| **合计** | **52,500** | **105,000** | **52,068** | **21** | **411** | **26,295** | **41,819** | **10,681** | — |

- 逐 shard labels SHA-256 全文见 `train_beta05.json:shards[].labels.sha256`。
- 411 个 `unknown` 全部是 model_mismatch（误差 > tolerance 8.0，与 WP05.2 同口径）。
- 每行恰 1 个候选被验证：`query_verified_candidates` 52,500 / `candidate_grid` 945,000 = 5.5556 %。
- **β=0.5 与 β=1.0 数据的两点实质差异**：(a) 注入率 0.500857 vs 1.0；(b) 师生一致率 **79.66 %** vs
  **68.76 %**（β=1.0 15 个 shard 聚合 36,099/52,500，见 `EVAL_50K.md` §3）——混合分布下教师更容易
  与学生一致，符合 β<1 的定义（50 % 的决策执行学生动作，命中率自然更高）。
- **WP05 两轮标签审计里第一次出现 verified_dead**：本轮 dead = **21**（β=1.0 的 21k 与 50k
  两次转换均为 **0**），分布在 7 个 shard（2026: 2、2030: 5、2033: 3、2035: 3、2036: 2、
  2037: 3、2038: 3）。比例仅 0.04 %，样本太小，不构成死亡通道覆盖，也不能用来解释任何 dev
  差异；只记录事实。

## 4. 数据混合（与 WP05.3/WP05.5 同一规则）

- **名义**：S1 20 % / DAgger 80 %，chosen one-hot CE；无 soft target、无 class weight、无 candidate mask。
- **规则**：每 step S1 行数 = `floor((step+1)·batch·0.2) − floor(step·batch·0.2)`；batch 内顺序每 step 随机打乱；
  每样本 loss 权重恒 1.0；`unknown` 只出现在 softmax 分母，从不作为负例或 target。

| 项 | 值 |
|---|---|
| S1 samples | **25,600（20.000 %）** |
| DAgger samples | **102,400（80.000 %）** |
| 总 samples | 128,000（= 1,000 × 128） |
| 每 step S1 行数 | 25 行 × 400 steps + 26 行 × 600 steps |
| 采样流消耗 | S1 1 个 epoch（46,397 rows 中抽 25,600）；DAgger **2 个 epoch**（52,500 rows 中抽 102,400） |
| rows_without_target | S1 0 / DAgger 0 |
| S1 majority baseline top1 | 0.4382（与 β1 两轮完全相同，同一 S1 训练集） |
| DAgger majority baseline top1 | **0.6619**（β1-50k 为 0.6891；本轮目标分布更分散） |

## 5. 训练指标（in-sample，训练批次上实测）

| 指标 | 本轮 β=0.5（52.5k） | β=1.0 21k | β=1.0 50k（52.5k） |
|---|---|---|---|
| loss | **0.9564** | 0.8324 | 0.8336 |
| top1 | **0.7164** | 0.7549 | 0.7579 |
| macroR（17 类） | **0.1406** | 0.2052 | 0.2081 |
| majority baseline top1 | 0.6172 | — | 0.6389 |
| loss 首 step → 末 step | 2.9003 → 0.6633 | 2.9041 → 0.5242 | 2.9041 → 0.5242 |
| loss 前 100 / 后 100 | 1.4864 / 0.8295 | 1.4551 / 0.6307 | 1.4551 / 0.6307 |
| 吞吐（samples/s） | **776.3**（争用下） | 2,079.9 | 2,112.1 |
| 训练 wall | 204 s（payload 193.79 s） | 68.05 s | 71.9 s（payload 69.02 s） |

分源：

| 源 | samples | loss | top1 | macroR | majority baseline |
|---|---|---|---|---|---|
| S1 | 25,600 | 1.6626 | 0.4923 | 0.1209 | 0.4382 |
| DAgger | 102,400 | 0.7799 | 0.7724 | 0.1405 | 0.6619 |

- **梯度裁剪**：`clip=5.0`；preclip mean 1.6776 / max 5.0604，postclip mean 1.6776 / max 4.99999906
  （β1-50k 为 preclip mean 1.8168 / max 4.5120，postclip 与 preclip 相同即未触发裁剪）。
- **GPU**：`cuda`，RTX 4070 SUPER，`max_memory_allocated` 252.11 MiB / reserved 434.0 MiB（与 β1 两轮逐字节相同）。
- 读法（**重要，避免过度解读**）：本轮 offline 拟合**比 β1 两轮都差**（loss 更高、top1/macroR 更低），
  但目标分布本身也更分散（DAgger majority baseline 0.6619 vs 0.6891），所以 top1 的差距里有一部分
  是分布差异、不是单纯的拟合变差；macroR 与前 17 类逐类 recall（类 3：0.6238 vs 0.7094；类 7：0.6279 vs
  0.7366）确实更低。离线指标都是**训练批次上的 in-sample 指标**，不是留出精度。

## 6. dev 20 局（S1N，真实引擎）

口径：`--stage 0`（S1N）+ `normal` + `--max-frames 19887` + `--stop-on-death` +
`--obs-window consecutive`；seeds **1000-1019**（dev 段 1000-1099；确认集 60000-60199 从未读）；
引擎 SHA `691bfd77…a568ef6`；checkpoint `a11ba0b7…ede25d5b`。

| seed | frames | steps(=决策) | score | idle 占比 | unique actions | stop_reason |
|---|---|---|---|---|---|---|
| 1000 | 1,404 | 702 | 10,877 | 0.7108 | 7 | death_stop |
| 1001 | 1,453 | 727 | 18,698 | 0.5887 | 7 | death_stop |
| 1002 | 1,325 | 663 | 9,447 | 0.6923 | 7 | death_stop |
| 1003 | 1,294 | 647 | 25,404 | 0.7836 | 7 | death_stop |
| 1004 | 1,747 | 874 | 27,910 | 0.6659 | 7 | death_stop |
| 1005 | 1,308 | 654 | 9,487 | 0.6911 | 6 | death_stop |
| 1006 | 1,324 | 662 | 9,876 | 0.6903 | 7 | death_stop |
| 1007 | 1,322 | 661 | 9,617 | 0.6899 | 7 | death_stop |
| 1008 | 1,101 | 551 | 6,357 | 0.7604 | 7 | death_stop |
| 1009 | 1,399 | 700 | 10,277 | 0.6857 | 6 | death_stop |
| 1010 | 1,298 | 649 | 26,119 | 0.7720 | 7 | death_stop |
| 1011 | 1,190 | 595 | 8,082 | 0.7697 | 7 | death_stop |
| 1012 | 1,320 | 660 | 9,607 | 0.7379 | 7 | death_stop |
| 1013 | 1,308 | 654 | 9,487 | 0.7110 | 6 | death_stop |
| 1014 | 1,308 | 654 | 9,487 | 0.6911 | 6 | death_stop |
| 1015 | 1,308 | 654 | 9,487 | 0.6911 | 6 | death_stop |
| 1016 | 1,335 | 668 | 8,191 | 0.7141 | 7 | death_stop |
| 1017 | 1,290 | 645 | 27,457 | 0.7860 | 7 | death_stop |
| 1018 | 1,328 | 664 | 9,756 | 0.6928 | 7 | death_stop |
| 1019 | 1,437 | 719 | 11,445 | 0.7330 | 6 | death_stop |

汇总（n=20，全部 `error=null`）：**通关 0/20（0.0 %）**；frames **mean 1,339.95** / **median 1,321.0** /
min 1,101 / max 1,747 / **sd 122.88**；deaths mean 1.0（`--stop-on-death` 构造）；stop_reason 20/20
`death_stop`；决策总数 13,403；**idle 71.07 %**（逐局均值 71.29 %，idle 定义 = candidate 0 = focus 0 + move 0）；
unique actions 均值 **6.7**；逐局 wall 合计 382.40 s。

动作分布（18 候选，占比）：idle **0.7107**、W 0.1412、E 0.1295、NE 0.0104、SW 0.0021、
focus-NE 0.0022、focus-E 0.0040，其余（含全部 focus-* 除 NE/E 外）**0**。

## 7. 与 β=1.0 并排对比（描述性，不写显著性、不写门）

### 7.1 dev 20 局同 seeds 并排

| 项 | **β=0.5 52.5k（本轮）** | β=1.0 21k | β=1.0 50k（52.5k） |
|---|---|---|---|
| 通关 | **0/20** | 0/20 | 0/20 |
| clear_rate | **0.0** | 0.0 | 0.0 |
| mean frames | **1,339.95** | 1,574.9 | 1,643.25 |
| median frames | **1,321.0** | 1,375.0 | 1,580.5 |
| min / max | **1,101 / 1,747** | 743 / 4,048 | 1,261 / 2,371 |
| sd | **122.88** | 669.54 | 315.64 |
| idle share | **71.07 %** | 70.47 % | 72.06 % |
| unique actions（均值） | **6.7** | 7.85 | 8.8 |
| stop_reason | 20/20 `death_stop` | 20/20 `death_stop` | 20/20 `death_stop` |
| checkpoint SHA | `a11ba0b7…` | `e771314d…` | `6ebe4ea9…` |
| DAgger 语料 | 52,500（β=0.5 第 2 轮） | 21,000（β=1.0 第 1 轮） | 52,500（β=1.0 第 1 轮） |
| 逐局 wall 合计 | 382.40 s（空载） | — | 467.16 s |

### 7.2 差值（本轮 − 参照）

| 对比 | mean 差 | mean 差 % | median 差 | idle 差 | clear 差 | loss 差 | top1 差 | macroR 差 |
|---|---|---|---|---|---|---|---|---|
| vs β=1.0 **21k** | **−234.95** | **−14.92 %** | **−54.0** | **+0.60 pp** | 0 | +0.1240 | −0.0385 | −0.0646 |
| vs β=1.0 **50k** | **−303.30** | **−18.46 %** | **−259.5** | **−1.00 pp** | 0 | +0.1228 | −0.0415 | −0.0675 |

可以说的（描述性）：本轮 **0/20 通关、idle 仍在 ~71 %**，与两个 β=1.0 checkpoint 完全同一档；
mean/median 都更低，且**离散度小得多**（sd 122.9，两个参照分别 669.5 / 315.6）——本轮的局全部
落在 1,101–1,747 帧之间，**没有出现 β1-21k 的 4,048 帧、β1-50k 的 2,371 帧那种长尾**，
即"更稳定地死得更早"，而不是"偶尔逃得更远"。
**不能说的**：不能说 β=0.5 让策略变差（或变好）。三个理由：(a) 每条臂只有 1 个训练 seed，
无方差估计；(b) 0/20 对 0/20，通关通道在本口径下没有分辨力；(c) 见 §7.3 的混淆。

### 7.3 混淆项（必须与数字一起读）

1. **rollout 学生不同**：β=1.0 的 15 个 shard 由 `3bc0ec44…`（首轮 baseline）以
   `injection_rate=1.0` 采集；β=0.5 的 15 个 shard 由 `6ebe4ea9…`（**WP05.5 的 50k B0**，
   `injection_rate=0.500857`）采集。也就是说本轮比较的是"**β=0.5 的第二轮数据**"对
   "**β=1.0 的第一轮数据**"——注入系数与 rollout 学生**同时**变了，不是纯 β 消融。
2. **样本量/轮次不同**：21k 是 6 个 shard、50k 是 15 个 β=1.0 首轮 shard、本轮是 15 个 β=0.5
   二轮 shard；训练预算与 20 个 dev seeds 相同，但语料来源批次不同。
3. **wall 不可比**：β1 两轮在空载机器上测量；本轮训练与实验期间同机有 G410-1 chunk3
   6 并发教师（见 §8），吞吐被压到 776 samples/s。
4. 单 seed、单 checkpoint、20 局：**无区间、无显著性检验、无预注册决策规则**。

## 8. 机器争用记录（wall 口径为什么会变）

| 阶段 | 时间（+08:00） | load(1m) | 现象 |
|---|---|---|---|
| 训练开始 | 13:06:32 | 59.08 | chunk3（6 教师）在跑；吞吐 776.3 sps（空载参照 2,112） |
| 训练结束 | 13:09:56 | 47.81 | 训练 wall 204 s（参照 71.9 s，≈2.8×） |
| dev attempt 1 | 13:10:15–13:24:47 | 45.07 → 73.93 | 逐局 wall 84–228 s，4/20 后 SIGTERM |
| dev attempt 2 等待 | 13:25:13–13:41:26 | 63.01 → 5.86 | 等 chunk3 退出（25/25 seed 落盘） |
| dev attempt 2 实跑 | 13:41:35–13:48:11 | 5.86 → 2.50 | 逐局 wall 16.4–23.5 s，20/20 完整 |

- 争用只影响 **wall**，不影响**策略行为**：attempt1 与 attempt2 的 4 个同名 seed 逐局
  frames/steps/score 完全相同（§2.5）。
- 训练指标不受影响：`--device auto` 走 CUDA，GPU 侧峰值 252 MiB，争用体现在 dataloader
  （`data_fraction` 0.4562）与 step 调度（median step 0.1381 s vs 空载 ~0.06 s）。
- 因此**不要**把本轮的 776.3 samples/s / 204 s 与 β1 的 2,112 / 71.9 s 当作模型或实现差异。

## 9. 限制与 not_claimed

**限制**
- 单臂（B0）、单训练 seed（0）、1,000 steps：无方差估计，不能比较训练方差。
- 一轮 DAgger（β=0.5 第 2 轮），52,500 decisions ≈ 目标的 105 %，但没有第 3 轮、没有 β=0 闭环。
- 离线指标是**训练批次上的 in-sample** 指标，不是留出精度；`val` 段为空。
- dev 评测只跑 20 个 dev seed（1000-1019），S1N 单关；不构成开发分数、不是通关率、不是四臂对比。
- wall 与吞吐与 β1 两轮**不可比**（同机 chunk3 并发，见 §8）。
- 20 局里 `deaths=1.0` 是 `--stop-on-death` 的构造值；`lives_end=2.0` 说明是掉残机而非耗尽全部残机。
- 只有一个动作通道被观测到发生变化（idle + E/W），focus 分支几乎全为 0，**动作多样性本身也很低**。

**not_claimed（与 `train_beta05.json` 一致）**
- **不是 G410-2**：v4.1.0 的验收门既未测量也未尝试；确认集 seeds 60000-60199 从未读取。
- 不对 20 局结果做显著性检验、不给置信区间、不设通过/失败门。
- **不声称 β=0.5 优于或劣于 β=1.0**：单臂单 seed、rollout 学生不同、0/20 对 0/20 无分辨力。
- 不声称任何 RL/PPO 训练：B0 是 chosen one-hot CE 的行为克隆基线。
- 不使用 attempt1 的 4 局 partial 结果（仅记录，已排除在全部统计之外）。
- 不声称死亡通道被覆盖（本轮 verified_dead 仅 21 条，占 0.04 %）。
