# WP05.5 — 15 个完整 DAgger shards（52,500 decisions）同预算重训 B0 + 20 dev episodes

- 时间：2026-11-13T09:10:00+08:00
- 结论：**已完成（描述性，不是门）**。按实际枚举，`.tmp_v410_wp05_round1/shards/` 下有 **15 个 `complete=true`** 的 WP05.0 DAgger shard（每个 3,500 decisions / 7,000 frames），合计 **52,500 decisions / 105,000 frames（≥ 50,000）**；全部用 WP05.2 CLI 转成 `labels_v410`（15/15 exit 0，labels/obs SHA 校验通过）。B0（731,802 参数）在**与 WP05.3 完全相同**的预算下重训：`steps=1000 × batch=128`、seed 0、S1 20 % / DAgger 80 %（实际 25,600 / 102,400 样本）、AdamW lr 3e-4 / wd 1e-4 / clip 5.0、只读 D410-S1 train seeds 1-14；训练 loss **0.8336** / top1 **0.7579** / macroR **0.2081**，2,112.1 samples/s，checkpoint SHA `6ebe4ea9…a69b8086`。20 个 dev episodes（seeds 1000-1019，S1N：stage0/normal/cap19887/stop-on-death）在真实 `rl_v3` 引擎上跑满：**0/20 过关、20/20 `death_stop`；frames mean 1,643.25、median 1,580.5、idle 72.06 %**。与 WP05.3 的 21k checkpoint 的 20 局（0/20、mean 1,574.9、median 1,375、idle 70.47 %）相比：mean **+68.35 帧（+4.34 %）**、median **+205.5**、idle **+1.59 pp**，仍是 0 过关。**本对比是描述性的：同预算、同口径、同 20 个 dev seeds，但 DAgger 语料同时变了数量和采集批次；无区间、无显著性检验、不写门、不是 G410-2。**
- 证据：[`train_50k.json`](train_50k.json)（55,310 B）；运行目录 `.tmp_v410_wp05_train50k/`（labels 15 个 / checkpoint / `train_run.json` / `eval_dev20.json` / logs，`.gitignore` 的 `.tmp_*/` 已忽略，不入库）。

## 0. 关键数字一览

| 项 | 值 |
|---|---|
| complete shards | **15**（seed2000-2010、2020-2023），每个 3,500 decisions / 7,000 frames |
| 聚合 decisions / frames | **52,500 / 105,000**（= 50k 目标的 105 %） |
| 生存标签（teacher-plan） | alive **52,416**、dead **0**、unknown **84**（全部 model_mismatch） |
| query 覆盖 | 52,500 / 945,000 = **5.5556 %**（每 shard 3,500/3,500） |
| 注入 / 一致性 | injected **52,500/52,500**（β=1.0）；agreement **36,099（68.76 %）** / disagreement **16,401（31.24 %）** |
| 标签转换 | WP05.2 CLI ×15，**15/15 exit 0**；单 shard 1.14–1.29 s；seed2000-2005 的 labels SHA 与 WP05.3 **逐字节相同** |
| S1 训练集 | 46,397 rows（seeds 1-14，train split only；val/test 从未打开） |
| 混合 | 名义 **S1 20 % / DAgger 80 %**；实际 **25,600 / 102,400 = 20.000 % / 80.000 %**，每样本权重 1.0 |
| 模型 | B0 `ProbeNet`，**731,802** 参数，chosen one-hot CE |
| 训练预算 | **1,000 steps × 128 = 128,000 samples**（与 WP05.3 逐项相同）；seed 0 |
| checkpoint | `.tmp_v410_wp05_train50k/b0_s0_dagger_step1000.pt`，2,941,187 B，SHA-256 `6ebe4ea91d81d9333bfa1ccc065fc3515bb25b1d3b90a2684cdaa8a0a69b8086` |
| 训练指标（in-sample） | loss **0.8336**、top1 **0.7579**、macroR **0.2081**；首 step 2.9041 → 末 step 0.5242 |
| 吞吐 / GPU | **2,112.1 samples/s / 16.5 steps/s**；RTX 4070 SUPER（峰值 252.11 MiB allocated） |
| dev 口径 | S1N：`--stage 0` + `normal` + `--max-frames 19887` + `--stop-on-death` + `--obs-window consecutive` |
| dev 结果（n=20） | **0/20 过关**；frames mean **1,643.25**、median **1,580.5**、min 1,261、max 2,371、sd 315.64；idle **72.06 %**；unique actions 8.8 |
| 21k 参照（WP05.3 checkpoint，同 20 seeds） | 0/20；mean **1,574.9**、median **1,375**、idle **70.47 %**；min 743、max 4,048 |
| 对比（描述性） | mean **+68.35（+4.34 %）**、median **+205.5**、idle **+1.59 pp**；配对 12 局更长 / 8 局更短，单局 −2,474…+1,213 |
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，SHA-256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` |

## 1. 交付物

| 文件 | 作用 |
|---|---|
| [`train_50k.json`](train_50k.json) | 本次 50k 重训的机器可读证据：输入 shards / 转换逐 shard 退出码、标签聚合、混合、预算、训练指标与吞吐、checkpoint、20 局 dev 汇总与逐局记录、与 21k 的并排对比、commands+exit_code、scope/not_claimed/limits |
| `EVAL_50K.md` | 本文件：命令与退出码、结果表、21k 对比、限制 |
| `.tmp_v410_wp05_train50k/` | 运行目录（不入库）：`labels/dagger-s0d1seed*/labels_v410.jsonl` + summary、`b0_s0_dagger_step1000.pt`、`train_run.json`、`dryrun_eval.json`、`eval_dev20.json`、`logs/{convert,train,eval_dev20}.log`、`logs/run_eval20.sh`、`logs/build_evidence.py` |

只读使用：`touhou_rl/v410/dagger/train.py`（WP05.3 CLI，未改动）、`touhou_rl/v410/dagger/labels.py`、`tools/rl/v410_student_eval.py`、WP05.1 shards、WP03 S1 audit [`s1c_full_audit.json`](../WP03/s1c_full_audit.json)、WP05 既有证据（`train_smoke.json`、`eval_dagger_20.json`）。

未修改 `v400/v380/v390`、`tools/rl/v400_*`、`tests/test_v400_*`、`third_party/`、v410 既有文件、`tools/rl/v410*.py`、tests、`state/journal/budget/assets`、WP03/WP04/WP05 既有证据；未训练超过 1,000 steps；未采新数据；未 commit/push；未删除/覆盖旧文件。

## 2. 命令与退出码（实际执行）

所有 Windows 命令先设 `$env:PYTHONPATH='D:\WorkShop\Touhou_RL'`。

### 2.1 枚举 complete shards

```powershell
$dirs = Get-ChildItem '.tmp_v410_wp05_round1/shards' -Directory | Sort-Object Name
$rows = foreach ($d in $dirs) {
  $j = Get-Content (Join-Path $d.FullName 'shard.json') -Raw | ConvertFrom-Json
  [pscustomobject]@{ Name=$d.Name; Seed=$j.seed; Complete=$j.complete;
                     Decisions=$j.decisions; Frames=$j.frames }
}
$rows | Format-Table -AutoSize
# 15 个 complete=True，decisions=3500，frames=7000（seed2000-2010、2020-2023）
# EXIT_CODE=0
```

### 2.2 标签转换（WP05.2 CLI，15 次）

```powershell
$seeds = @(2000..2010) + @(2020..2023)
foreach ($seed in $seeds) {
  python -m touhou_rl.v410.dagger.labels `
    --shard .tmp_v410_wp05_round1/shards/dagger-s0d1seed$seed `
    --out   .tmp_v410_wp05_train50k/labels/dagger-s0d1seed$seed `
    --tolerance 8.0
}
# 2000..2010,2020..2023 全部 exit 0（1.14–1.29 s/shard；日志 .tmp_v410_wp05_train50k/logs/convert.log）
# EXIT_CODE=0（循环整体）
```

每次输出形如 `decisions=3500 plans=3500 alive=3492 dead=0 unknown=8 model_mismatch=8 incomplete=0 query_verified=3500/63000 query_coverage=0.055556 injected=3500 disagreement=1167 tolerance=8.0 horizon=32`，并写出 `labels_v410.jsonl` + `labels_v410.summary.json`。

### 2.3 训练（Windows，CUDA，与 WP05.3 同预算）

```powershell
python -m touhou_rl.v410.dagger.train train `
  --steps 1000 --batch-size 128 --seed 0 --s1-share 0.2 --device auto `
  --shard-seeds 2000-2010,2020-2023 `
  --shards-root .tmp_v410_wp05_round1/shards `
  --labels-dir .tmp_v410_wp05_train50k/labels `
  --workdir .tmp_v410_wp05_train50k `
  --out .tmp_v410_wp05_train50k/train_run.json
# dagger_train: steps=1000 samples=128000 s1=25600 dagger=102400
#   loss=0.8336 top1=0.7579 macroR=0.2081 2112.1 samples/s
#   checkpoint=.tmp_v410_wp05_train50k\b0_s0_dagger_step1000.pt
# EXIT_CODE=0（进程 elapsed 71.9 s；payload elapsed 69.02 s）
```

### 2.4 dev 接线 dry run（不是证据）

```bash
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_student_eval.py --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp05_train50k/b0_s0_dagger_step1000.pt \
  --seeds 1000 --stage 0 --difficulty normal --max-frames 60 --stop-on-death \
  --out .tmp_v410_wp05_train50k/dryrun_eval.json
# wrote ... (exit 0)  EXIT_CODE=0
```

### 2.5 真实 20 episodes（WSL，`xvfb-run`）

命令固定为 `.tmp_v410_wp05_train50k/logs/run_eval20.sh`（脚本化以避免 PowerShell→WSL 引号转义）：

```bash
cd /mnt/d/WorkShop/Touhou_RL
timeout 5400 xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_student_eval.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp05_train50k/b0_s0_dagger_step1000.pt \
  --seeds 1000-1019 --stage 0 --difficulty normal --max-frames 19887 --stop-on-death \
  --out .tmp_v410_wp05_train50k/eval_dev20.json
# WSL_EXIT=0；eval_dev20.json: exit_code=0；20/20 episodes；逐局 wall 合计 467.16 s
```

- 运行备注：评测一共启动过三次。第 1 次在同一命令下跑完 seed1000 后被人工中止（未写输出文件）；第 2 次因 PowerShell 把单引号串传给 WSL 时丢失内层引号，`xvfb-run` 立即报 `usage error: need a command to run`（未产生输出）；第 3 次改为执行脚本文件后完整跑满 20 局，exit 0。本文件只使用第 3 次的 `eval_dev20.json`。
- 20 局全部 `harness_role="env"`、`error=null`、`stage_start=0 → stage_end=0`、`lives_end=2.0`；`deaths=1.0` 是 `--stop-on-death` 的构造值。

### 2.6 证据构建与自检

```powershell
python .tmp_v410_wp05_train50k/logs/build_evidence.py
# wrote Document/Update/Alpha/v4.1.0/evidence/WP05/train_50k.json (55310 B)
# EXIT_CODE=0

python -c "import json; d=json.load(open('Document/Update/Alpha/v4.1.0/evidence/WP05/train_50k.json', encoding='utf-8')); print(d['schema'], len(d['shards']), d['decisions'], d['dev_eval']['n'])"
# v410-wp05-dagger-train-50k/1 15 52500 20
# EXIT_CODE=0
```

`build_evidence.py` 只做聚合与交叉核对：复核 evaluated checkpoint SHA 与训练 checkpoint 一致（不一致即退出非 0）、episode 数 = 20、seeds = 1000-1019、eval `exit_code=0`；mean/median/idle 由原始逐局记录独立重算，并与 evaluator 的 `arm_summary` 对照。

## 3. shards 与标签转换（15 个，WP05.2 规则）

每个 shard 的 `labels.jsonl` 行数、`shard.json.decisions`、obs 三平面行数与 SHA-256 全部一致（summary 中 `checks.labels_sha256_verified` / `checks.obs_sha256_verified` 均 true）；`train.py` 读入时再校验 `labels_v410.jsonl` 的 SHA-256 并按 `s1_data.parse_label_row` 逐行解析。

| shard | decisions | frames | alive | dead | unknown | injected | agreement | disagreement | labels SHA-256（前 8） |
|---|---|---|---|---|---|---|---|---|---|
| dagger-s0d1seed2000 | 3,500 | 7,000 | 3,492 | 0 | 8 | 3,500 | 2,333 | 1,167 | `3494caa6` |
| dagger-s0d1seed2001 | 3,500 | 7,000 | 3,494 | 0 | 6 | 3,500 | 2,387 | 1,113 | `196c7a97` |
| dagger-s0d1seed2002 | 3,500 | 7,000 | 3,497 | 0 | 3 | 3,500 | 2,378 | 1,122 | `f82d3954` |
| dagger-s0d1seed2003 | 3,500 | 7,000 | 3,492 | 0 | 8 | 3,500 | 2,409 | 1,091 | `286b76dd` |
| dagger-s0d1seed2004 | 3,500 | 7,000 | 3,492 | 0 | 8 | 3,500 | 2,468 | 1,032 | `1a9b09b9` |
| dagger-s0d1seed2005 | 3,500 | 7,000 | 3,496 | 0 | 4 | 3,500 | 2,469 | 1,031 | `fcd9c162` |
| dagger-s0d1seed2006 | 3,500 | 7,000 | 3,491 | 0 | 9 | 3,500 | 2,462 | 1,038 | `8cbd3cf4` |
| dagger-s0d1seed2007 | 3,500 | 7,000 | 3,497 | 0 | 3 | 3,500 | 2,454 | 1,046 | `aac97e5d` |
| dagger-s0d1seed2008 | 3,500 | 7,000 | 3,496 | 0 | 4 | 3,500 | 2,480 | 1,020 | `e1be87c7` |
| dagger-s0d1seed2009 | 3,500 | 7,000 | 3,492 | 0 | 8 | 3,500 | 2,375 | 1,125 | `212613a2` |
| dagger-s0d1seed2010 | 3,500 | 7,000 | 3,497 | 0 | 3 | 3,500 | 2,344 | 1,156 | `41158a4a` |
| dagger-s0d1seed2020 | 3,500 | 7,000 | 3,495 | 0 | 5 | 3,500 | 2,385 | 1,115 | `3fcc824c` |
| dagger-s0d1seed2021 | 3,500 | 7,000 | 3,494 | 0 | 6 | 3,500 | 2,434 | 1,066 | `2637e913` |
| dagger-s0d1seed2022 | 3,500 | 7,000 | 3,496 | 0 | 4 | 3,500 | 2,446 | 1,054 | `c938ec24` |
| dagger-s0d1seed2023 | 3,500 | 7,000 | 3,495 | 0 | 5 | 3,500 | 2,275 | 1,225 | `55e95147` |
| **合计** | **52,500** | **105,000** | **52,416** | **0** | **84** | **52,500** | **36,099** | **16,401** | — |

- 84 个 `unknown` 全部是 model_mismatch（误差 > tolerance 8.0，与 WP05.2 同口径）；本轮 **0 个 verified_dead**，DAgger 侧死亡通道仍未由真实数据覆盖。
- 每行恰 1 个候选被验证：`query_verified_candidates` 52,500 / `candidate_grid` 945,000 = 5.5556 %。
- **可复现性交叉核对**：seed2000-2005 六个 shard 的 `labels_v410.jsonl` SHA-256 与 WP05.3 记录的完全相同（`3494caa6 / 196c7a97 / f82d3954 / 286b76dd / 1a9b09b9 / fcd9c162`），说明转换是确定性的、与 21k 那次用的是同一规则。
- 本轮实际参与训练的是 15 个 shard；batch B 的 seed2024 在我固定 seed 列表后才收尾（`dagger-s0d1seed2024` 于 2026-11-13T08:53:54+08:00 变为 `complete=true`，3,500 decisions）。按 52,500 ≥ 50,000 的判据本轮不需要它，故**未使用**，只在 `train_50k.json:inputs.shard_selection` 记录（如并入则为 56,000 decisions）。

## 4. 数据混合（与 WP05.3 同一规则）

- **名义**：S1 20 % / DAgger 80 %，chosen one-hot CE；无 soft target、无 class weight、无 candidate mask。
- **规则**：每 step S1 行数 = `floor((step+1)·batch·0.2) − floor(step·batch·0.2)`；batch 内顺序每 step 随机打乱；每样本 loss 权重恒 1.0；`unknown` 只出现在 softmax 分母，从不作为负例或 target。

| 项 | 值 |
|---|---|
| S1 samples | **25,600（20.000 %）** |
| DAgger samples | **102,400（80.000 %）** |
| 总 samples | 128,000（= 1,000 × 128） |
| 每 step S1 行数 | 25 行 × 400 steps + 26 行 × 600 steps |
| 采样流消耗 | S1 1 个 epoch（46,397 rows 中抽 25,600）；DAgger **2 个 epoch**（52,500 rows 中抽 102,400；WP05.3 为 5 个 epoch） |
| rows_without_target | S1 0 / DAgger 0 |

## 5. 训练指标（in-sample，训练批次上实测）

B0：731,802 参数（`ProbeNet`），AdamW lr 3e-4 / wd 1e-4 / clip 5.0，seed 0，1,000 steps × batch 128，CUDA（torch 2.14.1+cu132 / Python 3.12.10 / Windows 11）。

| 指标 | 50k 全窗口 | 21k 全窗口（WP05.3） | 50k S1 | 21k S1 | 50k DAgger | 21k DAgger |
|---|---|---|---|---|---|---|
| loss | **0.8336** | 0.8324 | 1.6632 | 1.6730 | **0.6262** | 0.6223 |
| top1 | **0.7579** | 0.7549 | 0.4969 | 0.4947 | **0.8232** | 0.8200 |
| macro recall | **0.2081** | 0.2052 | 0.1268 | 0.1265 | **0.2384** | 0.2333 |
| majority baseline top1 | 0.6389 | — | 0.4382 | 0.4382 | 0.6886 | 0.6860 |

- 学习进程（50k）：首 step loss **2.9041** → 末 step **0.5242**；前 100 步均值 1.4551 → 后 100 步均值 0.6307。
- 最后 100 步（step 901–1000，50k）：总 loss 0.6307；S1 1.5044 / top1 0.5328 / macroR 0.1748；DAgger 0.4123 / top1 0.8867 / macroR 0.4313。
- 梯度（50k）：clip 前均值 1.8168（最大 4.5120），**全程未触发 clip**（21k 的 preclip 最大 6.2939、触发过 5.0）。
- 吞吐（50k）：**60.60 s** 训练环 / 128,000 samples = **2,112.1 samples/s、16.5 steps/s**；数据装载 33.75 s（**55.69 %**），中位 step 59.8 ms，首 step 715.7 ms（CUDA 预热）。
- GPU：RTX 4070 SUPER，峰值 allocated **252.11 MiB** / reserved **434.0 MiB**（总 12,281.5 MiB）。
- 口径：这 128,000 个样本就是本次训练消费的批次，**是 in-sample 优化诊断，不是 held-out 准确率**；按预注册只读 train split（seeds 1-14），val/test 从未打开。
- 注意：50k 与 21k 的 DAgger 采样 epoch 数不同（2 vs 5），所以 per-source top1/loss 的并排只是**同口径复算**，不是"数据更多所以更准"的证据。

## 6. 20 个 dev episodes（真实引擎）

- checkpoint：`b0_s0_dagger_step1000.pt`（SHA `6ebe4ea9…a69b8086`，schema `v410-wp04-bc-arm-checkpoint/1`，arm B0，731,802 参数）；evaluator 复核 SHA 一致。
- 引擎：`rl_v3` `th08-modern` SHA `691bfd77…7a568ef6`；`harness_role="env"`、`error=null`、`exit_code=0`。

| seed | frames | steps(=决策) | score | wall_s | stop_reason | stage_clear |
|---|---|---|---|---|---|---|
| 1000 | 1,323 | 662 | 6,858 | 19.82 | `death_stop` | false |
| 1001 | 1,261 | 631 | 6,217 | 19.19 | `death_stop` | false |
| 1002 | 1,852 | 926 | 12,531 | 26.07 | `death_stop` | false |
| 1003 | 1,333 | 667 | 7,087 | 20.31 | `death_stop` | false |
| 1004 | 1,326 | 663 | 6,888 | 20.21 | `death_stop` | false |
| 1005 | 1,440 | 720 | 10,428 | 21.22 | `death_stop` | false |
| 1006 | 1,314 | 657 | 6,817 | 19.86 | `death_stop` | false |
| 1007 | 1,963 | 982 | 15,499 | 26.50 | `death_stop` | false |
| 1008 | 1,886 | 943 | 15,606 | 25.43 | `death_stop` | false |
| 1009 | 1,739 | 870 | 15,186 | 24.24 | `death_stop` | false |
| 1010 | 1,558 | 779 | 13,328 | 22.41 | `death_stop` | false |
| 1011 | 1,956 | 978 | 16,374 | 26.29 | `death_stop` | false |
| 1012 | 1,984 | 992 | 16,995 | 27.01 | `death_stop` | false |
| 1013 | 1,587 | 794 | 11,679 | 22.93 | `death_stop` | false |
| 1014 | 1,574 | 787 | 11,513 | 22.39 | `death_stop` | false |
| 1015 | 1,968 | 984 | 23,984 | 27.18 | `death_stop` | false |
| 1016 | 1,326 | 663 | 6,888 | 20.22 | `death_stop` | false |
| 1017 | 1,778 | 889 | 14,646 | 24.76 | `death_stop` | false |
| 1018 | 2,371 | 1,186 | 24,335 | 31.18 | `death_stop` | false |
| 1019 | 1,326 | 663 | 6,888 | 19.95 | `death_stop` | false |

- 汇总：**0/20 过关、20/20 `death_stop`**；frames mean **1,643.25**、median **1,580.5**、min 1,261、max 2,371、sd 315.64；score mean 12,487.35；16,436 次决策池化。
- 动作分布（16,436 次决策）：idle **11,844（72.06 %）**、W 1,922（11.69 %）、E 1,609（9.79 %）、NE 404（2.46 %）、SW 343（2.09 %）、focus-NE 113（0.69 %）、focus-E 104（0.63 %）、focus-SW 55（0.33 %），其余 ≤19；每局平均 unique actions **8.8**（21k：7.85）。
- 动作序列/输入 SHA 与逐局原始记录在 `.tmp_v410_wp05_train50k/eval_dev20.json`（103,006 B）。

## 7. 与 WP05.3 21k checkpoint 的受控对比（描述性）

**控制关系**：命令相同、预算逐项相同（1,000 steps × 128、seed 0、S1 20 %/DAgger 80 %、AdamW 3e-4/wd 1e-4/clip 5.0、128,000 samples、25,600/102,400）、S1 训练集相同（46,397 rows，seeds 1-14）、dev 口径与 seeds 相同（1000-1019、S1N、cap 19887、stop-on-death）、引擎相同。**唯一变化是 DAgger 语料：6 shards/21,000 → 15 shards/52,500 decisions**（同时多出 batch B 的采集批次，故不是"纯数据量"对照）。

| 项 | 50k（本次） | 21k（WP05.3） | 差 |
|---|---|---|---|
| 过关 | **0/20** | 0/20 | 0 |
| frames mean | **1,643.25** | 1,574.9 | **+68.35（+4.34 %）** |
| frames median | **1,580.5** | 1,375 | **+205.5** |
| frames min / max | 1,261 / 2,371 | 743 / 4,048 | — |
| frames sd | 315.64 | 669.54 | — |
| idle（池化） | **72.06 %** | 70.47 % | **+1.59 pp** |
| unique actions / 局 | 8.8 | 7.85 | +0.95 |
| 训练 loss / top1 / macroR | 0.8336 / 0.7579 / 0.2081 | 0.8324 / 0.7549 / 0.2052 | +0.0012 / +0.0030 / +0.0029 |

逐 seed 配对（同一 seed、同一引擎与口径；只描述帧数差）：

| seed | 50k frames | 21k frames | Δ |
|---|---|---|---|
| 1000 | 1,323 | 1,609 | −286 |
| 1001 | 1,261 | 1,200 | +61 |
| 1002 | 1,852 | 1,183 | +669 |
| 1003 | 1,333 | 1,343 | −10 |
| 1004 | 1,326 | 2,292 | −966 |
| 1005 | 1,440 | 1,375 | +65 |
| 1006 | 1,314 | 1,375 | −61 |
| 1007 | 1,963 | 1,347 | +616 |
| 1008 | 1,886 | 1,364 | +522 |
| 1009 | 1,739 | 1,523 | +216 |
| 1010 | 1,558 | 2,050 | −492 |
| 1011 | 1,956 | 743 | +1,213 |
| 1012 | 1,984 | 1,375 | +609 |
| 1013 | 1,587 | 1,189 | +398 |
| 1014 | 1,574 | 4,048 | −2,474 |
| 1015 | 1,968 | 1,348 | +620 |
| 1016 | 1,326 | 1,805 | −479 |
| 1017 | 1,778 | 1,557 | +221 |
| 1018 | 2,371 | 1,191 | +1,180 |
| 1019 | 1,326 | 1,581 | −255 |

- 配对汇总：**12 局更长、8 局更短**，配对差中位数 **+140.5**，范围 **−2,474（seed1014）… +1,213（seed1011）**，均值 +68.35。
- 读法：均值/中位数都略高于 21k，但单局差异的符号并不一致、量级跨度很大（21k 有一局 4,048 帧、50k 最长 2,371），且两个 checkpoint 都是 0 过关。**因此只能说"这 20 个 dev seeds 上首死帧数的均值略高"，不能据此写"数据量带来提升"、更不能写效应估计或显著性**：n=20、单 seed、语料同时换了数量与批次、离线指标是 in-sample。
- 与 B0 baseline（WP04.3，同口径，0/20、mean 1,453.25）相比：本次 mean +190.0 帧；同样只是描述性（数据混合、训练预算都不同）。

## 8. 限制（不声称）

1. **不是 G410-2**：无 confirm split（60000–60199 从未读取）、无四臂矩阵、无预注册判据、无通过率门。
2. **不是开发成绩/不是门**：20 局是开发测量，`clear_rate = 0` 不是 pass rate，工具 verdict 仍是 `is_a_gate=false`、`is_a_development_score=false`、`conclusion_allowed=false`。
3. **不是"50k 数据带来提升"的因果证据**：语料从 21k→52,500（并加入 batch B 的 seeds 2020-2023）与采集批次同时变化；n=20、单训练 seed、单预算，无区间、无检验。
4. **不是 β=0.5 轮**：15 个 shard 全部 β=1.0（教师全注入）；β=0.5 第二轮未跑。
5. **离线指标是 in-sample**：128,000 个训练批次上的 loss/top1/macroR 是优化诊断，不是 held-out 泛化；val/test 从未打开。
6. **死亡通道未覆盖**：52,500 行 0 个 `verified_dead`，`negative_mask` 恒空；unknown 仅 84 个（全部 model_mismatch）。
7. **只有 B0 / 1 个 seed / 1,000 步**：无 seed 方差、无收敛声明；checkpoint 是第 1,000 步的 last-step 权重。
8. **数据管线仍是瓶颈**：数据装载占训练环 55.69 %，GPU 峰值仅 252 MiB；当前吞吐不代表模型上限。
9. **未并入 seed2024**：该 shard 在 seed 列表固定后才 complete（若并入为 56,000 decisions），本轮按 52,500 ≥ 50,000 的口径交付。
10. **未做权重消融**：S1 20 % 是照搬 WP05.3 的预注册值；S1 上 top1 0.4969 / macroR 0.1268 仍明显低于 DAgger 上 0.8232 / 0.2384，只记录不解释。

## 9. 下一建议

1. **β=0.5 第二轮**：用同一批 shard 采集时把注入率降到 0.5，仍用同一 `train.py`、同一预算训练，先看离线 top1/macroR 与 student–teacher disagreement 分布，再决定是否值得开 dev 评测。
2. **多 seed / 多轮**：每轮固定 1,000 步、训练 seed 0/1/2，记录 seed 方差；只有预注册判据 + ≥20 episodes 时才允许比较"效果"。
3. **把 16 个 shard（56,000）作为下一个点**：seed2024 已 complete，可在同一预算下再补一个数据量点，保持"只改语料"的对照设计。
4. **动作多样性 / idle 偏高**：本次 idle 72.06 %、12/20 局帧数变长但 0 过关；值得在离线侧检查 chosen one-hot CE 下的类别不平衡与 idle 先验（不改协议的前提下做诊断）。
5. **数据管线**：预取/合并 uint8 批次到 pinned memory 或按 shard 预组 batch，把 55.69 % 的数据时间压下去，再评估是否提升步数预算。
6. **死亡通道**：需要能产生 `verified_dead` 的采集/标注切片（当前 0 条），否则 DAgger 侧始终学不到"会死的候选"。
