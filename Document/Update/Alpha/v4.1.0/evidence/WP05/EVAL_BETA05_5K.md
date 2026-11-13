# WP05.8 — β=0.5 15 shards（52,500 decisions）5,000 步受控训练 + 20 dev 局（与 1k / β1-50k 并排）

- 时间：2026-11-13T13:53:58–14:35:43+08:00（训练 13:53:58–14:01:56；随后等 G410-1 chunk4 退出等了
  1,500 s（14:02:08–14:27:08）；dev 20 局 14:27:58–14:35:43）
- 结论（**描述性，不是门**）：把 WP05.7 的 1,000 步换成 **5,000 步**（其余预算/混合/种子/数据逐项不变）后，
  **离线指标大幅改善**（loss 0.9564 → **0.6586**，top1 0.7164 → **0.7993**，macroR 0.1406 → **0.2733**，
  约 1.94 倍），**20 dev 存活帧 mean 1,339.95 → 1,689.10（+349.15 帧，+26.06 %）**、median 1,321.0 →
  **1,605.0（+284.0）**、idle 71.07 % → **62.84 %（−8.22 pp）**、unique actions 6.7 → **12.5**。
  但 **clear 仍是 0/20**，20/20 `death_stop`。因此：**1,000 步确实是欠训练**（同样的数据、同样的种子，
  多 4,000 步明显变好），但 5 倍预算**不足以**换来一次通关——预算不是唯一瓶颈。
  与 β1-50k（mean 1,643.25 / median 1,580.5 / idle 72.06 %）相比，本 arm 的 mean 高 **+45.85 帧（+2.79 %）**、
  median 高 **+24.5**、idle 低 **9.22 pp**，但两者的注入系数与 rollout 学生不同，**不是受控预算对比**。
- 证据：[`train_beta05_5k.json`](train_beta05_5k.json)（127,483 B）；运行目录
  `.tmp_v410_wp05_train_beta05_5k/`（labels 15 个 / checkpoint / `train_run.json` / `eval_dev20.json` /
  `verify_labels.json` / `logs/`，被 `.gitignore` 的 `.tmp_*/` 覆盖，不入库）。

## 0. 关键数字一览

| 项 | 值 |
|---|---|
| complete shards | **15**（seed2025-2039），每个 3,500 decisions / 7,000 frames，`beta=0.5`、`stage=0`、`normal`、`stop_reason=decisions` |
| 聚合 decisions / frames | **52,500 / 105,000**（= 50k 目标的 105 %） |
| 采集注入率（shard.json） | injected **26,295 / 52,500 = 0.500857**（β≈0.5） |
| 师生一致 / 不一致 | **41,819（79.66 %）/ 10,681（20.34 %）** |
| rollout 学生 | `6ebe4ea9…a69b8086`（15 个 shard 同一 checkpoint，= WP05.5 50k B0 step1000） |
| 生存标签（teacher-plan） | alive **52,068**、dead **21**、unknown **411**（全部 model_mismatch） |
| query 覆盖 | 52,500 / 945,000 = **5.5556 %**（每 shard 3,500/3,500） |
| 标签转换 | WP05.2 CLI ×15，**15/15 exit 0**，6–8 s/shard；独立复核 52,500 行 + SHA-256 全对 |
| S1 训练集 | 46,397 rows（seeds 1-14，train split only；val/test 从未打开） |
| 混合（实测） | 名义 **S1 20 % / DAgger 80 %**；实际 **128,000 / 512,000 = 20.000 % / 80.000 %**，样本权重 1.0 |
| 训练预算 | **5,000 steps × 128 = 640,000 samples**（与 WP05.7 的 1,000×128 相比只改步数）；seed 0 |
| 数据重复 | S1 流 **3 epochs**、DAgger 流 **10 epochs**（预算加在**同一批行**上，不是新数据） |
| 模型 | B0 `ProbeNet`，**731,802** 参数，chosen one-hot CE |
| checkpoint | `.tmp_v410_wp05_train_beta05_5k/b0_s0_dagger_step5000.pt`，2,941,187 B，SHA-256 `692fdc61dc473ff10657d2a0b1a37434d3b24561fedb9798a41a38086bdbd120` |
| 训练指标（in-sample） | loss **0.6586**、top1 **0.7993**、macroR **0.2733**；首 step 2.9003 → 末 step 0.4223 |
| 吞吐 / GPU | **1512.8 samples/s / 11.82 steps/s**；RTX 4070 SUPER（峰值 252.11 MiB allocated / 434.0 MiB reserved） |
| 训练 wall | 进程 **478 s**（payload `elapsed_seconds` 461.37 s，其中 `train_seconds` 423.05 s、`data_seconds` 174.71 s） |
| 确定性续训 | 前 1,000 步的 11 个记录点与 WP05.7 **逐点完全相同**（max \|Δloss\| = 0.0），即 5k = 1k 的续训 |
| dev 口径 | S1N：`--stage 0` + `normal` + `--max-frames 19887` + `--stop-on-death` + `--obs-window consecutive` |
| dev 结果（n=20） | **0/20 通关**；frames mean **1,689.10**、median **1,605.0**、min 1,530、max 2,213、sd 204.38；idle **62.84 %**；unique actions **12.5** |
| β=0.5 1k 参照（同数据） | 0/20；mean 1,339.95、median 1,321.0、idle 71.07 %、unique 6.7 |
| β=1.0 50k 参照 | 0/20；mean 1,643.25、median 1,580.5、idle 72.06 %、unique 8.8 |
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，SHA-256 `691bfd77…7a568ef6` |
| G410-1 chunk4 | 13:51:16 启动 → **14:26:59 结束，exit 0，25/25 seed 落盘**；评测**等它退出后**才开始（等了 1,500 s，未向教师发任何信号） |

## 1. 交付物

| 文件 | 作用 |
|---|---|
| [`train_beta05_5k.json`](train_beta05_5k.json) | 本轮机器可读证据：shards/转换退出码、标签聚合、混合、预算、训练指标与曲线、吞吐、checkpoint、commands+exit_code、20 局汇总与逐局、与 1k/β1-50k 的并排与差值、fail-closed 校验清单、scope/not_claimed/limits |
| `EVAL_BETA05_5K.md` | 本文件 |
| `.tmp_v410_wp05_train_beta05_5k/` | 运行目录（不入库）：`labels/dagger-s0d1seed*/{labels_v410.jsonl,labels_v410.summary.json}`、`b0_s0_dagger_step5000.pt`、`train_run.json`、`verify_labels.json`、`eval_dev20.json`、`dryrun_eval.json`、`logs/{convert_labels.sh,run_train.sh,run_eval.sh,build_evidence_5k.py,verify_labels.py,enumerate_shards.py,…}` |

只读使用、**未改动**：`touhou_rl/v410/dagger/train.py`、`labels.py`、`tools/rl/v410_student_eval.py`、
WP05.0/05.6 shards、WP03 S1 audit、WP05.7 [`train_beta05.json`](train_beta05.json) / [`EVAL_BETA05.md`](EVAL_BETA05.md)、
WP05.5 [`train_50k.json`](train_50k.json) / [`EVAL_50K.md`](EVAL_50K.md)。
**未采集任何数据、未跑确认集、未做 G410-2、未 commit。**

## 2. 命令与退出码（逐步实测）

### 2.1 枚举 complete shards

```powershell
wsl -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python \
  .tmp_v410_wp05_train_beta05_5k/logs/enumerate_shards.py'
# 15/15 complete=True decisions=3500 frames=7000 beta=0.5 stage=0 normal stop=decisions
# 合计 N=15 DECISIONS=52500 FRAMES=105000  rollout_checkpoints={6ebe4ea9…}
# ENUMERATE_OK，EXIT=0
```

判据与 WP05.7 相同：只取 `complete=true`，15 个全达标（52,500 ≥ 50,000），**没有伪造、没有补数**。

### 2.2 标签转换（WP05.2 CLI ×15）

```bash
bash .tmp_v410_wp05_train_beta05_5k/logs/convert_labels.sh
#   /root/v400-venv/bin/python -m touhou_rl.v410.dagger.labels \
#     --shard .tmp_v410_wp05_round2/shards/dagger-s0d1seed<seed> \
#     --out   .tmp_v410_wp05_train_beta05_5k/labels/dagger-s0d1seed<seed> \
#     --tolerance 8.0
# 2025..2039 全部 exit=0，6–8 s/shard；日志 logs/convert_exit_codes.txt
# CONVERT_ALL_DONE fail=0   EXIT=0
```

### 2.3 标签独立复核（重新读文件 + 重新算 SHA-256）

```bash
/root/v400-venv/bin/python .tmp_v410_wp05_train_beta05_5k/logs/verify_labels.py
# 15 个 shard：rows=3500 == decisions、labels SHA-256 == summary 记录、obs SHA-256 verified=True
# 聚合 decisions=52500 frames=105000 rows=52500
# query_coverage=0.05555555555555555 injection_rate=0.5008571428571429 agreement_rate=0.7965523809523809
# VERIFY_LABELS_OK   EXIT=0
```

### 2.4 训练（WSL2，CUDA，与 WP05.7 同预算只改步数）

```bash
bash .tmp_v410_wp05_train_beta05_5k/logs/run_train.sh
#   /root/v400-venv/bin/python -m touhou_rl.v410.dagger.train train \
#     --steps 5000 --batch-size 128 --seed 0 --s1-share 0.2 --device auto \
#     --shard-seeds 2025-2039 \
#     --shards-root .tmp_v410_wp05_round2/shards \
#     --labels-dir .tmp_v410_wp05_train_beta05_5k/labels \
#     --workdir .tmp_v410_wp05_train_beta05_5k \
#     --out .tmp_v410_wp05_train_beta05_5k/train_run.json
# [WP05.3] DAgger: 52500 rows with a chosen target from 15 shards
# [WP05.3] D410-S1 train: 46397 rows from seeds 1-14
# dagger_train: steps=5000 samples=640000 s1=128000 dagger=512000
#   loss=0.6586 top1=0.7993 macroR=0.2733 1512.8 samples/s
#   checkpoint=.tmp_v410_wp05_train_beta05_5k/b0_s0_dagger_step5000.pt
# TRAIN_EXIT=0 wall_seconds=478
```

### 2.5 dev 接线 dry run（60 帧，不算证据）

```bash
# 由 run_eval.sh 在等完 chunk4 后自动执行：
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_student_eval.py --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp05_train_beta05_5k/b0_s0_dagger_step5000.pt \
  --seeds 1000 --stage 0 --difficulty normal --max-frames 60 --stop-on-death \
  --out .tmp_v410_wp05_train_beta05_5k/dryrun_eval.json
# DRYRUN_EXIT=0
```

### 2.6 等 G410-1 chunk4 → 20 局真实评测

```bash
bash .tmp_v410_wp05_train_beta05_5k/logs/run_eval.sh
# 1) 轮询 `pgrep -f v410_teacher_fast20`，每 60 s 一次，上限 2400 s；命中则继续等
#    → 14:02:08 开始等，14:27:08 结束，waited=1500 s（chunk4 于 14:26:59 exit 0）
# 2) dry run（见 2.5），DRYRUN_EXIT=0
# 3) 20 局：
#    xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
#      tools/rl/v410_student_eval.py --engine …/rl_v3/th08-modern --data-dir /root/v400-data/game \
#      --checkpoint …/b0_s0_dagger_step5000.pt \
#      --seeds 1000-1019 --stage 0 --difficulty normal --max-frames 19887 --stop-on-death \
#      --out .tmp_v410_wp05_train_beta05_5k/eval_dev20.json
# EVAL_EXIT=0 wall_seconds=465 waited=1500s
```

- **没有为了评测杀教师进程**：等 chunk4 自己退出（`/tmp/v410_g4101_chunk4/exit_code.txt = 0`，
  `finished_at=2026-11-13T14:26:59+08:00`，25 个 per-seed jsonl 齐全）。
- 等完后 load 从 ~45 掉到 4.89，逐局 wall 20.88–27.92 s，与 WP05.7 attempt-2 的 16.4–23.5 s 同量级（略高，
  因为 5k 学生每局平均多活约 350 帧）。

### 2.7 证据构建（fail-closed）

```bash
/root/v400-venv/bin/python .tmp_v410_wp05_train_beta05_5k/logs/build_evidence_5k.py
# wrote …/evidence/WP05/train_beta05_5k.json (127483 B)
# BUILD_EXIT=0
```

任一交叉校验不一致即 **exit 3** 拒绝写出：eval `exit_code=0`、episode 数 = 20、seed 严格等于 1000-1019、
逐局 `error=null`、逐局 `checkpoint_sha256` 与训练 checkpoint 一致、逐局 `engine_sha256` 与头部一致、
独立重算的 mean/median/idle 与 evaluator 的 `arm_summary`/`aggregate` 一致（1e-6）、train steps=5000 且
samples=640000、15 shard 52,500 decisions/52,500 rows/105,000 frames、labels+obs SHA 全验证、
前 1,000 步 loss 与 WP05.7 逐点相同。

## 3. shards 与标签（15 个，WP05.2 规则）

| shard | decisions | frames | alive | dead | unknown | mm | injected | agree | disagree | labels SHA-256 |
|---|---|---|---|---|---|---|---|---|---|---|
| dagger-s0d1seed2025 | 3,500 | 7,000 | 3,494 | 0 | 6 | 6 | 1,730 | 2,804 | 696 | `170775f2` |
| dagger-s0d1seed2026 | 3,500 | 7,000 | 3,451 | 2 | 47 | 47 | 1,759 | 2,776 | 724 | `d4bdc39f` |
| dagger-s0d1seed2027 | 3,500 | 7,000 | 3,482 | 0 | 18 | 18 | 1,754 | 2,703 | 797 | `8800fcd0` |
| dagger-s0d1seed2028 | 3,500 | 7,000 | 3,490 | 0 | 10 | 10 | 1,787 | 2,703 | 797 | `f9bf46fd` |
| dagger-s0d1seed2029 | 3,500 | 7,000 | 3,490 | 0 | 10 | 10 | 1,758 | 2,805 | 695 | `b9365376` |
| dagger-s0d1seed2030 | 3,500 | 7,000 | 3,452 | 5 | 43 | 43 | 1,739 | 2,763 | 737 | `d5221680` |
| dagger-s0d1seed2031 | 3,500 | 7,000 | 3,493 | 0 | 7 | 7 | 1,751 | 2,872 | 628 | `a44029cf` |
| dagger-s0d1seed2032 | 3,500 | 7,000 | 3,489 | 0 | 11 | 11 | 1,764 | 2,704 | 796 | `c8c853e4` |
| dagger-s0d1seed2033 | 3,500 | 7,000 | 3,466 | 3 | 31 | 31 | 1,721 | 2,890 | 610 | `868339ba` |
| dagger-s0d1seed2034 | 3,500 | 7,000 | 3,491 | 0 | 9 | 9 | 1,730 | 2,809 | 691 | `e00db162` |
| dagger-s0d1seed2035 | 3,500 | 7,000 | 3,426 | 3 | 71 | 71 | 1,826 | 2,905 | 595 | `609f84bf` |
| dagger-s0d1seed2036 | 3,500 | 7,000 | 3,491 | 2 | 7 | 7 | 1,765 | 2,734 | 766 | `2b06fc3b` |
| dagger-s0d1seed2037 | 3,500 | 7,000 | 3,458 | 3 | 39 | 39 | 1,746 | 2,904 | 596 | `f45bee9a` |
| dagger-s0d1seed2038 | 3,500 | 7,000 | 3,401 | 3 | 96 | 96 | 1,755 | 2,695 | 805 | `414a20cd` |
| dagger-s0d1seed2039 | 3,500 | 7,000 | 3,494 | 0 | 6 | 6 | 1,710 | 2,752 | 748 | `e50fb7aa` |
| **合计 15** | **52,500** | **105,000** | **52,068** | **21** | **411** | **411** | **26,295** | **41,819** | **10,681** | — |

`mm` = `model_mismatch`；`unknown` 全部等于 `model_mismatch`（教师模型版本对不上的候选，不当负样本）。
15 个 shard 的 `beta` 全为 0.5、`stage=0`、`difficulty=normal`、`stop_reason=decisions`、
`checkpoint.sha256` 全为 `6ebe4ea9…`。

## 4. 训练（5,000 步）

### 4.1 预算与混合

| 项 | 值 |
|---|---|
| steps / batch | 5,000 × 128 = **640,000 samples** |
| 优化器 | AdamW，lr 3e-4，wd 1e-4，grad clip 5.0 |
| 混合（名义） | S1 20 % / DAgger 80 % |
| 混合（实测） | S1 **128,000** / DAgger **512,000** = 20.000 % / 80.000 % |
| 数据重复 | S1 46,397 行 × **3 epochs**；DAgger 52,500 行 × **10 epochs** |
| 梯度范数（preclip） | mean 1.9440，max 5.0604 |
| 设备 | CUDA，RTX 4070 SUPER；峰值 252.11 MiB allocated / 434.0 MiB reserved |

### 4.2 指标（in-sample 训练批，不是留出精度）

| 指标 | WP05.7 1k | **WP05.8 5k** | 变化 |
|---|---|---|---|
| loss | 0.9564 | **0.6586** | −0.2978 |
| top1 | 0.71641 | **0.79926** | +0.0828 |
| macroR（17 类） | 0.14059 | **0.27326** | **×1.94** |
| 多数类基线 top1 | 0.61718 | 0.61715 | — |
| S1 源 top1 / macroR | 0.4923 / 0.1209 | **0.6151 / 0.2350** | +0.123 / +0.114 |
| DAgger 源 top1 / macroR | 0.7724 / 0.1405 | **0.8453 / 0.2722** | +0.073 / +0.132 |
| 首 step → 末 step loss | 2.9003 → 0.6633 | 2.9003 → **0.4223** | — |

### 4.3 训练曲线（每 100 step 的 loss，按 500 step 抽样）

| steps | loss | steps | loss |
|---|---|---|---|
| 1–100 | 1.4864 | 2501–2600 | 0.6093 |
| 501–600 | 0.8660 | 3001–3100 | 0.5854 |
| 1001–1100 | 0.7870 | 3501–3600 | 0.5335 |
| 1501–1600 | 0.7132 | 4001–4100 | 0.5108 |
| 2001–2100 | 0.6557 | 4501–4600 | 0.4967 |
|  |  | 4901–5000 | 0.4468 |

前 1,000 步的 loss 轨迹与 WP05.7 **逐点相同**（1.4864 / 1.0612 / 0.9548 / … / 0.8295），这是同 seed、
同数据、同混合顺序的确定性重放；所以两个 checkpoint 的**唯一差别**就是多出的 4,000 步。

### 4.4 吞吐与墙钟（**不与 β1 受控比较**）

| 项 | 值 |
|---|---|
| 吞吐 | **1512.8 samples/s / 11.82 steps/s**（WP05.7 1k：776.3 samples/s / 6.06 steps/s） |
| `train_seconds` / `data_seconds` | 423.05 s / 174.71 s（data_fraction 0.413） |
| 单 step | median 0.0725 s，max 1.4085 s（= 首 step） |
| 进程 wall | **478 s**（payload `elapsed_seconds` 461.37 s） |
| 同机负载 | 训练期间 G410-1 chunk4 的 6 个并发教师实例在跑（训练窗口起点 load 52.52；1k 那轮起点 59.08） |

两轮吞吐不可当受控对比：1k 那轮首 step 与 data 时间的占比更高，teacher 批次负载也不同；
本文件只把吞吐当作本轮观测值记录。

## 5. 20 dev 局（seeds 1000-1019）

### 5.1 汇总

| 项 | 值 |
|---|---|
| 通关 | **0 / 20**（clear_rate 0.0） |
| 终止原因 | `death_stop` **20/20**（`--stop-on-death` 构造值） |
| frames | mean **1,689.10**、median **1,605.0**、min 1,530、max 2,213、sd 204.38 |
| idle share | **62.84 %**（按局均值 63.17 %） |
| unique actions（按局均值） | **12.5** |
| decisions | 16,896（20 局合计） |
| deaths | 每局 1.0 |
| 逐局 wall | 20.88–27.92 s，合计 451.43 s；驱动器 wall 465 s |
| 引擎 / checkpoint | `691bfd77…` / `692fdc61…`（20 局逐局核对一致，`error=null`） |
| seed 政策 | 全部落在 dev 区间 1000-1099；确认集 60000-60199 与 D410-S1 语料 seeds **未被请求** |

### 5.2 逐局

| seed | frames | steps | score | stop | wall s |
|---|---|---|---|---|---|
| 1000 | 1,615 | 808 | 13,543 | death_stop | 21.49 |
| 1001 | 1,532 | 766 | 10,829 | death_stop | 21.54 |
| 1002 | 1,530 | 765 | 10,794 | death_stop | 21.63 |
| 1003 | 1,548 | 774 | 11,444 | death_stop | 21.51 |
| 1004 | 1,548 | 774 | 11,434 | death_stop | 22.03 |
| 1005 | 1,610 | 805 | 12,882 | death_stop | 22.20 |
| 1006 | 1,553 | 777 | 11,814 | death_stop | 21.30 |
| 1007 | 1,593 | 797 | 12,607 | death_stop | 22.20 |
| 1008 | 1,574 | 787 | 12,926 | death_stop | 21.35 |
| 1009 | 1,867 | 934 | 17,006 | death_stop | 24.10 |
| 1010 | 1,770 | 885 | 14,542 | death_stop | 23.15 |
| 1011 | 1,531 | 766 | 11,138 | death_stop | 20.88 |
| 1012 | 2,151 | 1,076 | 18,583 | death_stop | 27.10 |
| 1013 | 2,213 | 1,107 | 26,418 | death_stop | 27.92 |
| 1014 | 1,676 | 838 | 13,976 | death_stop | 21.96 |
| 1015 | 1,928 | 964 | 17,512 | death_stop | 24.90 |
| 1016 | 1,784 | 892 | 22,164 | death_stop | 22.48 |
| 1017 | 1,601 | 801 | 12,514 | death_stop | 21.35 |
| 1018 | 1,549 | 775 | 11,514 | death_stop | 21.07 |
| 1019 | 1,609 | 805 | 12,872 | death_stop | 21.27 |

### 5.3 动作分布（20 局合计 16,896 decisions）

| 动作 | idle | N | NE | E | SE | S | SW | W | NW | focus-NE | focus-E | focus-SE | focus-SW | focus-W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 次数 | 10,618 | 125 | 502 | 2,048 | 124 | 114 | 557 | 2,379 | 65 | 157 | 137 | 2 | 53 | 15 |
| 占比 | 62.84 % | 0.74 % | 2.97 % | 12.12 % | 0.73 % | 0.67 % | 3.30 % | 14.08 % | 0.38 % | 0.93 % | 0.81 % | 0.01 % | 0.31 % | 0.09 % |

（其余 4 个 focus 方向为 0。1k 参照：idle 71.07 %、只用到 6 个方向类，无 N/SE/S/NW。）

## 6. 与 1k / β1-50k 并排（描述性）

### 6.1 训练指标

| arm | steps × batch | samples | loss | top1 | macroR | 吞吐 |
|---|---|---|---|---|---|---|
| β=0.5 1k（WP05.7，同数据） | 1,000×128 | 128,000 | 0.9564 | 0.7164 | 0.1406 | 776.3 samples/s |
| **β=0.5 5k（本轮）** | **5,000×128** | **640,000** | **0.6586** | **0.7993** | **0.2733** | 1512.8 samples/s |
| β=1.0 50k（WP05.5） | 1,000×128 | 128,000 | 0.8336 | 0.7579 | 0.2081 | 2112.1 samples/s |

### 6.2 dev 20 局

| arm | clear | mean | median | min / max | sd | idle | unique actions |
|---|---|---|---|---|---|---|---|
| β=0.5 1k（同数据，1/5 预算） | **0/20** | 1,339.95 | 1,321.0 | 1,101 / 1,747 | 122.88 | 71.07 % | 6.7 |
| **β=0.5 5k（本轮）** | **0/20** | **1,689.10** | **1,605.0** | 1,530 / 2,213 | 204.38 | **62.84 %** | **12.5** |
| β=1.0 50k（WP05.5） | **0/20** | 1,643.25 | 1,580.5 | 1,261 / 2,371 | 315.64 | 72.06 % | 8.8 |
| β=1.0 21k（WP05.3） | 0/20 | 1,574.9 | 1,375.0 | 743 / 4,048 | 669.54 | 70.47 % | 7.85 |

### 6.3 差值

| 对比 | mean | median | idle | unique | clear |
|---|---|---|---|---|---|
| 5k **vs** β=0.5 1k（受控：只改步数） | **+349.15 帧（+26.06 %）** | **+284.0** | **−8.22 pp** | +5.8 | 0/20 → 0/20 |
| 5k **vs** β=1.0 50k（非受控） | +45.85 帧（+2.79 %） | +24.5 | −9.22 pp | +3.7 | 0/20 → 0/20 |

**怎么读**：

- 与 1k 的对比是**受控**的：同 15 个 shard、同 labels、同 seed 0、同 S1/DAgger 混合、同优化器、同 dev 口径，
  且前 1,000 步逐点重放。唯一的自变量是**多 4,000 步**。在这个意义上，1,000 步的 0/20 与 mean 1,339.95
  **确实有欠训练成分**：多 4,000 步让离线 macroR 翻倍、dev 存活 +26 %、idle −8 pp、动作多样性近乎翻倍。
- 但 **5 倍预算仍然 0/20**，20 局全部 `death_stop`，帧数分布整体上移（min 1,530 vs 1k 的 1,101，mean 与
  median 同步抬高），说明它学到的是"更久地闪避"，而不是"过面"。**预算不是唯一瓶颈**。
- 与 β=1.0 50k 的对比**不是**受控预算对比：注入系数（1.0 vs 0.500857）与 rollout 学生（3bc0ec44… vs
  6ebe4ea9…）都不同。本 arm 略高 2.79 %、idle 低 9.22 pp，只能当作描述性旁证。

## 7. 限制

1. **一个 arm、一个训练种子（0）、一组预算对比（1k vs 5k）**：没有任何方差估计；20 局的对比不能把
   "预算"与"噪声"分开，也没有区间/显著性检验（本文件不写门）。
2. 离线指标是**训练批上的 in-sample** 指标（同一批 128 样本），不是留出精度。
3. 5,000 步用的是**同一批** 46,397 S1 + 52,500 DAgger 行：预算换来的是**重复**（3 / 10 epochs），
   不是新数据；仍然只有**一轮** DAgger。
4. **只测了 1,000 与 5,000 两个点**，不能推断"更多步会继续变好"或"5,000 是最优"。
5. 训练期间同机有 G410-1 chunk4 的 6 个并发教师实例（训练窗口起点 load 52.52），吞吐/墙钟不能与其他轮次
   受控对比；dev 是在 chunk4 **退出后**才跑的（等了 1,500 s），这一点与 WP05.7 attempt-2 一致、与 attempt-1 不同。
6. dev 只有 20 个 dev seed（1000-1019）；**G410-2 未测**。
7. `touhou_rl/v410/dagger/train.py` 写进 `train_run.json` 的 `scope`/`not_claimed` 是 WP05.3 的通用模板
   （仍写着 "1,000 steps"、"not a beta=0.5 round"），对本次运行**已过时**；本文件把它原样保留在
   `raw_tool_payload_scope` 并显式声明不据此作任何主张。

## 8. not_claimed / 边界

- **不是 G410-2**：没有 confirm 集、没有四臂矩阵、没有预注册决策规则。
- **不碰确认集 60000-60199**：评测器的 seed 政策与逐局记录都显示只请求了 dev 1000-1019。
- **没有采集新数据**、没有跑 collection rollout、没有重跑教师、没有确认种子；引擎只被评测器启动。
- **不声称 β=0.5 比 β=1.0 好/差**：两者注入系数与 rollout 学生不同。
- **不声称 5,000 步"修好了" S1N**：clear 仍是 0/20。
- **不声称任何带区间的结论**：20 局对比是描述性的，没有显著性检验、没有门。
- 未 commit / 未 push；未修改白名单以外的任何文件（含 v400/v380/v390、tools/rl/v400_*、
  tests、third_party、既有 v410 文件、state/journal/budget/assets、其它 WP 证据）。
