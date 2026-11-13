# WP05.12 — 同一 52.5k β=0.5 数据、同一 5,000 步预算下的 idle 类降权受控训练 + 20 dev 局

> **一句话**：与 WP05.8 的 5k arm **同数据、同标签、同混合、同预算、同 seed、同优化器**，只把 loss 里
> 的 idle 类权重降到 **0.7129**（其余 17 类 1.4628），让**有效 idle 份额 61.72 % → 44.00 %**（对齐
> D410-S1 的 43.96 %）。离线目标确实按设计动了（macroR 0.2733 → **0.3142**，非 idle 16 类 recall 全升、
> idle −3.45 pp），但闭环几乎没动：**0/20 通关不变**，frames mean 1,689.10 → **1,693.30（+4.20 帧，
> +0.25 %）**、median 1,605.0 → **1,610.5（+5.5）**、idle 62.84 % → **59.77 %（−3.08 pp）**、unique
> actions 12.5 → **10.2（−2.3）**。**结论（描述性，不是门）：在这个预算下，loss 侧的 idle 先验不是闭环
> 0/20 的主因**；把先验从 61.72 % 压到 44 %（−17.72 pp）只换来 3.08 pp 的 dev idle 下移和 4 帧均值变化
> （n=20、sd≈173，完全在噪声内）。IDLE_DIAG 的 H1 falsifier 条件（新先验 44 % 而 dev idle 仍 >58 %）
> **被触发**：dev idle 59.77 %。但注意这只推翻「**loss 先验**决定 dev idle」这一条，**标签侧**先验（R1 去平局
> 重标注）尚未测。

- 时间：2026-11-13T15:25:52–15:50:19+08:00（训练 15:25:52–15:30:34，**282 s**；dev 20 局
  15:30:44–15:50:19，**1,164 s**）
- 证据：[`train_classweight.json`](train_classweight.json)（73,974 B，机器可读）；运行目录
  `.tmp_v410_wp05_classweight/`（`b0_s0_dagger_step5000.pt` / `train_run.json` / `eval_dev20.json` /
  `dryrun_eval.json` / `logs/`，被 `.gitignore` 的 `.tmp_*/` 覆盖，不入库）
- 引擎：`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，SHA-256 `691bfd77…7a568ef6`
  （与 5k / 20k 同一二进制）

## 0. 关键数字一览

| 项 | 值 |
|---|---|
| 唯一改动 | `--class-weight-mode idle-cap --idle-target-share 0.44`（其余参数与 WP05.8 逐项相同） |
| loss 有效 idle 份额 | **44.00 %**（label prior 61.716 % → implied share 0.44；对齐 D410-S1 43.9554 %） |
| 权重表 | idle **0.712943**、其余 17 类 **1.462754**；样本加权平均权重 **1.000000**（loss 尺度不变） |
| 数据 / 预算 | 15 个 β=0.5 shard（52,500 decisions）+ D410-S1 train 46,397 行；**5,000 × 128 = 640,000** 样本；seed 0 |
| 混合（实测） | S1 128,000 / DAgger 512,000 = **20.000 % / 80.000 %**；S1 流 3 epochs、DAgger 流 10 epochs |
| 训练指标（in-sample） | loss **0.6568**、top1 **0.7973**、macroR **0.3142**（5k none：0.6586 / 0.7993 / 0.2733） |
| 逐类 recall | idle 0.9636 → **0.9292（−3.45 pp）**；非 idle 16 类（有标签的 16 类）**全部上升**（+0.95 ~ +9.69 pp） |
| 加权目标 | 加权 CE mean **0.8389**（末 100 步 0.5564）；同时记录未加权 CE 以保持与 none 可比 |
| checkpoint | `.tmp_v410_wp05_classweight/b0_s0_dagger_step5000.pt`，2,941,699 B，SHA-256 `f447664c…53a4919` |
| 吞吐 / GPU | **2,556.29 samples/s / 19.97 steps/s**；RTX 4070 SUPER 峰值 **252.11 MiB** allocated / 434.0 MiB reserved |
| 训练 wall | 驱动 **282 s**（payload `elapsed_seconds` 272.24 s，`train_seconds` 250.36 s、`data_seconds` 110.97 s） |
| dev 口径 | S1N：`--stage 0` + `normal` + `--max-frames 19887` + `--stop-on-death` + `--obs-window consecutive` |
| **dev 20 局** | **0/20 通关**（`death_stop` 20/20）；frames mean **1,693.30**、median **1,610.50**、min 1,520、max 2,011、sd 172.94；idle **59.77 %**（逐局均值 60.15 %）；unique actions **10.2** |
| 与 5k none 差 | mean **+4.20（+0.25 %）**、median +5.5、idle **−3.08 pp**、unique **−2.3**、clear 0/20 → 0/20 |
| 与 20k none 差 | mean **−21.35（−1.25 %）**、median −9.0、idle **−1.11 pp**、unique −2.25、clear 0/20 → 0/20 |
| 并发争用 | round3 DAgger batchA（`v410_dagger_collect`，seeds 2040-2054，β=0.5）**全程在跑**；只观测、**未发任何信号、未 kill**；load 15.07 → 22.79 |

## 1. 交付物

| 文件 | 作用 |
|---|---|
| [`train_classweight.json`](train_classweight.json) | 机器可读证据：设计、命令+退出码、18 行权重表、训练指标/曲线/吞吐/GPU、checkpoint SHA、20 局汇总与逐局、动作直方图、与 5k/20k 并排与差值、争用记录、fail-closed 校验、limits/not_claimed |
| `EVAL_CLASSWEIGHT.md` | 本文件 |
| `.tmp_v410_wp05_classweight/logs/` | 运行日志与驱动：`run_train.sh` / `run_eval.sh` / `build_evidence.py` / `train.log` / `eval_dev20.log` / `train_exit_code.txt` / `eval_exit_code.txt` |

只读使用、**未改动**：`.tmp_v410_wp05_round2/shards/*`、`.tmp_v410_wp05_train_beta05_5k/labels/*`
（WP05.8 的转换输出，SHA 在训练时重新核对通过）、`touhou_rl/v410/student/{bc,s1_data}.py`、
`tools/rl/v410_student_eval.py`、WP05 既有证据（`idle_diag.json` / `train_beta05_5k.json` /
`train_beta05_20k.json`）。**未采数据、未跑教师、未确认种子、未做 G410-2、未 commit。**

## 2. 变更（白名单内）

| 文件 | 变更 |
|---|---|
| `touhou_rl/v410/dagger/train.py` | 新增可选类权重：`--class-weight-mode {none,invfreq,idle-cap}`（默认 `none`）、`--idle-target-share`（默认 0.44）；`class_weight_table()` / `expected_mix_counts()` / `class_weighted_ce()`；run JSON 增 `class_weight` 块与 `sampled_label_counts`；`none` 时 checkpoint 的 `config` **不写** class_weight（默认产物逐字节不变） |
| `tests/test_v410_dagger_train.py` | 追加 13 个用例（共 17 个 test，含 2 处 parametrize）：三种模式、目标份额自洽、fail-closed、`none` 与改造前指标逐值一致、加权只改 loss 不改批次、CLI 非法参数 |

## 3. 类权重设计（只改 loss 权重，不重采样、不改标签）

对每个训练行 `i`，loss 为

```
CE_weighted = Σ_i w[y_i] · CE_i / Σ_i w[y_i]          （weighted mean）
w_c          = m_c / Σ_j (p_j · m_j)                  （使 Σ_c p_c · w_c = 1）
```

其中 `p_c` 是固定步数流上的**期望标签先验**（`s1_quota × S1 频率 + dagger_quota × DAgger 频率`，
`s1_quota = floor(5000×128×0.2) = 128,000`、`dagger_quota = 512,000`）。`idle-cap` 取

```
m_idle = λ = t·(1 − p_idle) / (p_idle·(1 − t))        t = 0.44, p_idle = 0.61716038
m_other = 1
```

解得 `λ = 0.487398`、归一化因子 `0.683642` → **w_idle = 0.712943**、**w_other = 1.462754**，
隐含 idle 份额 **0.440000**（自检 `|q_idle − t| < 1e-9` 不过即 fail-closed）。

### 3.1 权重表（18 类，`weights[target]` 查表）

| idx | 类 | 640k 流内计数 | 先验 p_c | 权重 w_c | 隐含份额 q_c |
|---|---|---|---|---|---|
| 0 | **idle** | 394,983 | 0.617160 | **0.712943** | **0.440000** |
| 1 | N | 6,981 | 0.010907 | 1.462754 | 0.015955 |
| 2 | NE | 27,710 | 0.043298 | 1.462754 | 0.063334 |
| 3 | E | 65,977 | 0.103089 | 1.462754 | 0.150793 |
| 4 | SE | 6,832 | 0.010675 | 1.462754 | 0.015615 |
| 5 | S | 6,177 | 0.009652 | 1.462754 | 0.014118 |
| 6 | SW | 25,447 | 0.039761 | 1.462754 | 0.058161 |
| 7 | W | 61,952 | 0.096800 | 1.462754 | 0.141595 |
| 8 | NW | 4,994 | 0.007803 | 1.462754 | 0.011414 |
| 9 | focus-idle | **0** | 0.000000 | 1.462754 | 0.000000 |
| 10 | focus-N | 738 | 0.001153 | 1.462754 | 0.001687 |
| 11 | focus-NE | 13,407 | 0.020948 | 1.462754 | 0.030642 |
| 12 | focus-E | 13,165 | 0.020571 | 1.462754 | 0.030090 |
| 13 | focus-SE | 2,143 | 0.003349 | 1.462754 | 0.004898 |
| 14 | focus-S | 761 | 0.001190 | 1.462754 | 0.001740 |
| 15 | focus-SW | 5,016 | 0.007838 | 1.462754 | 0.011466 |
| 16 | focus-W | 2,340 | 0.003656 | 1.462754 | 0.005348 |
| 17 | focus-NW | 1,376 | 0.002150 | 1.462754 | 0.003145 |

- 交叉校验：算出的 `p_idle = 0.61716038` 与 `idle_diag.json` 的 `label_prior_idle_mix = 0.61716`
  **一致**（同一 52,500 行 + 同一 46,397 S1 行、同一 20/80 配额）。
- 采样实测：`sampled_label_counts[0] = 394,983 / 640,000 = 0.617145`，与期望先验差 7.5e-6 →
  采样流与设计一致。
- `focus-idle`（idx 9）在语料里 **0 行**，权重不可能生效；这与 IDLE_DIAG 的
  「class 9 是构造性缺失」一致，不是被学掉的。
- 权重**只**进 loss：`mix.realized` 的 `sample_weight` 标记为 `per-class`，`s1/dagger` 行数、
  顺序、shuffle、标签、`rows_without_target` 全部不变（`label_resampled=false`、
  `labels_changed=false`）。

## 4. 命令与退出码（逐步实测）

### 4.1 pytest（Windows，验收命令）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp0512_pytest -q tests/test_v410_dagger_train.py
# 51 passed in 5.50s        EXIT=0   （34 个改造前用例 + 17 个新用例；收尾复核重跑 51 passed in 6.04s）
```

WSL venv 复跑：`/root/v400-venv/bin/python -m pytest … tests/test_v410_dagger_train.py` →
**51 passed in 24.69s，EXIT=0**。

### 4.2 compileall

```powershell
python -m compileall -q touhou_rl/v410/dagger     # EXIT=0（仅语法，不等于测试通过）
```

### 4.3 训练（WSL2 / CUDA，5,000 步）

```bash
/root/v400-venv/bin/python -m touhou_rl.v410.dagger.train train \
  --steps 5000 --batch-size 128 --seed 0 --s1-share 0.2 --device auto \
  --shard-seeds 2025-2039 \
  --shards-root .tmp_v410_wp05_round2/shards \
  --labels-dir .tmp_v410_wp05_train_beta05_5k/labels \
  --workdir .tmp_v410_wp05_classweight \
  --out .tmp_v410_wp05_classweight/train_run.json \
  --class-weight-mode idle-cap --idle-target-share 0.44
# [WP05.3] class weights: mode=idle-cap idle_prior=0.617160 idle_weight=0.712943
#          idle_implied_share=0.440000 mean_weight=1.000000
# [WP05.3] DAgger: 52500 rows with a chosen target from 15 shards
# [WP05.3] D410-S1 train: 46397 rows from seeds 1-14
# dagger_train: steps=5000 samples=640000 s1=128000 dagger=512000 loss=0.6568
#   top1=0.7973 macroR=0.3142 2556.3 samples/s checkpoint=…/b0_s0_dagger_step5000.pt
# dagger_train: class_weight mode=idle-cap idle_weight=0.712943
#   idle_implied_share=0.440000 weighted_loss=0.8389
# TRAIN_EXIT=0 wall_seconds=282
```

### 4.4 20 dev 局（WSL2 / xvfb / S1N）

```bash
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python tools/rl/v410_student_eval.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp05_classweight/b0_s0_dagger_step5000.pt \
  --seeds 1000-1019 --stage 0 --difficulty normal --max-frames 19887 --stop-on-death \
  --out .tmp_v410_wp05_classweight/eval_dev20.json
# DRYRUN_EXIT=0（60 帧接线检查，不算证据）
# EVAL_EXIT=0 wall_seconds=1164
```

### 4.5 证据构建（fail-closed）

```bash
/root/v400-venv/bin/python .tmp_v410_wp05_classweight/logs/build_evidence.py
# wrote …/evidence/WP05/train_classweight.json (73974 B)     BUILD_EXIT=0
```

构建器在以下任一条件下**拒绝写文件**：run schema 不符、`class_weight.applied/mode` 不是 idle-cap、
隐含份额 ≠ 0.44（1e-6）、权重表不是 18 维、平均权重 ≠ 1、预算不是 640,000/5,000、
评测 checkpoint SHA ≠ 训练 SHA、磁盘 checkpoint SHA ≠ 记录的 SHA、dev seed 不是 1000-1019、
直方图和不等于决策数、`arm_summary.clears` 与逐局 `stage_clear` 不一致。

## 5. 训练（5,000 步）

### 5.1 预算与混合

| 项 | 值 |
|---|---|
| steps × batch | 5,000 × 128 = **640,000** 样本（与 5k none 同一预算） |
| 混合（实测） | S1 **128,000（20.000 %）** / DAgger **512,000（80.000 %）**；每步 25 或 26 行 S1（2,000/3,000 步） |
| 数据重复 | S1 流 **3 epochs**、DAgger 流 **10 epochs**（预算加在同一批行上，不是新数据） |
| 优化器 | AdamW lr 3e-4 / wd 1e-4 / clip 5.0；grad pre-clip mean 2.5111、max 4.9883（**未触发** clip 上限 5.0） |
| 模型 | B0 `ProbeNet`，**731,802** 参数，chosen one-hot CE（无人为 bomb 头） |

### 5.2 指标（in-sample 训练批，不是留出精度）

| 指标 | 5k none（WP05.8） | **5k idle-cap（本轮）** | 差 |
|---|---|---|---|
| loss（未加权 CE） | 0.6586 | **0.6568** | −0.0018 |
| top1 | 0.7993 | **0.7973** | **−0.20 pp** |
| macroR | 0.2733 | **0.3142** | **+4.09 pp（+15.0 %）** |
| 加权 CE（本轮优化目标） | —（= loss） | **0.8389** | — |
| loss 首 step / 末 step | 2.9003 / 0.4223 | 2.9003 / 0.4312 | 首步一致 |
| loss 首 100 / 末 100 步 | 1.4864 / 0.4468 | 1.4639 / 0.4361 | — |

**批次未变的证据**：第 1 步的未加权 CE 为 `2.900337696`（本轮）vs `2.900337458`（5k none），
差 **2.4e-7**（仅 `F.cross_entropy` 默认均值与 `per-row → mean` 的归约顺序差）→ 第一步吃到的是
**同一批 128 个样本**。

### 5.3 逐类 recall（同一训练批，in-sample）

| 类 | none 5k | idle-cap 5k | Δ |
|---|---|---|---|
| idle | 0.9636 | 0.9292 | **−3.45 pp** |
| N | 0.1274 | 0.1997 | +7.23 pp |
| NE | 0.3475 | 0.4171 | +6.96 pp |
| E | 0.7669 | 0.8101 | +4.33 pp |
| SE | 0.0999 | 0.1584 | +5.85 pp |
| S | 0.2538 | 0.3507 | **+9.69 pp** |
| SW | 0.3434 | 0.4232 | +7.98 pp |
| W | 0.7865 | 0.8196 | +3.31 pp |
| NW | 0.1972 | 0.2447 | +4.75 pp |
| focus-idle | —（0 行） | —（0 行） | — |
| focus-N | 0.0054 | 0.0149 | +0.95 pp |
| focus-NE | 0.2964 | 0.3644 | +6.80 pp |
| focus-E | 0.3560 | 0.3966 | +4.06 pp |
| focus-SE | 0.0051 | 0.0341 | +2.90 pp |
| focus-S | 0.0013 | 0.0183 | +1.70 pp |
| focus-SW | 0.0723 | 0.1061 | +3.38 pp |
| focus-W | 0.0204 | 0.0302 | +0.98 pp |
| focus-NW | 0.0022 | 0.0234 | +2.12 pp |

**读法**：降权在**离线目标上完全按设计生效**——idle 让出 3.45 pp recall，16 个有标签的非 idle 类无一例外
上升，macroR 涨 4.09 pp。这排除了「权重没生效」这一解释；下面 dev 的平坦结果不是实现 bug。

### 5.4 训练曲线（每 100 步；`loss` 未加权、`weighted` 为优化目标）

| 步 | loss | weighted | S1 loss | DAgger loss |
|---|---|---|---|---|
| 1–100 | 1.4639 | 1.7712 | 2.0701 | 1.3123 |
| 601–700 | 0.8494 | 1.0905 | 1.5723 | 0.6687 |
| 1,201–1,300 | 0.7483 | 0.9599 | 1.4239 | 0.5795 |
| 1,801–1,900 | 0.6857 | 0.8742 | 1.3010 | 0.5319 |
| 2,401–2,500 | 0.6126 | 0.7833 | 1.1935 | 0.4674 |
| 3,001–3,100 | 0.5833 | 0.7495 | 1.2034 | 0.4283 |
| 3,601–3,700 | 0.4995 | 0.6378 | 1.1114 | 0.3465 |
| 4,201–4,300 | 0.4530 | 0.5764 | 1.0670 | 0.2995 |
| 4,801–4,900 | 0.4282 | 0.5399 | 1.0423 | 0.2747 |
| 4,901–5,000 | 0.4361 | 0.5564 | — | — |

### 5.5 吞吐、GPU、墙钟

| 项 | 值 |
|---|---|
| samples/s / steps/s | **2,556.29 / 19.97**（5k none 为 1,512.8 / 11.82，**不可比**：并发负载不同） |
| 中位 / 最大步时 | 0.0403 s / 1.1007 s（首步含 CUDA 预热） |
| data 占比 | 110.97 s / 250.36 s = **44.33 %**（numpy 侧 gather 是瓶颈之一） |
| GPU 峰值 | **252.11 MiB** allocated、**434.0 MiB** reserved（12,281.5 MiB 总） |
| wall | 驱动 282 s；payload `elapsed_seconds` 272.24 s |
| checkpoint | 2,941,699 B、`f447664c660933af9c2840dd2c8e43051e9757c2f520fa9706178c92b53a4919`（**磁盘重算一致**） |

## 6. 20 dev 局（seeds 1000-1019）

### 6.1 汇总

| 项 | 值 |
|---|---|
| 通关 | **0 / 20**（clear_rate 0.0；Wilson 95 % 上界 0.1611，描述性） |
| 终止原因 | `death_stop` **20/20**（`--stop-on-death` 构造值） |
| frames | mean **1,693.30**、median **1,610.50**、min 1,520、max 2,011、sd **172.94** |
| idle share | **59.77 %**（逐局均值 60.15 %） |
| unique actions（按局均值） | **10.2**（逐局 7–15） |
| decisions | 16,938（20 局合计） |
| deaths | 每局 1.0 |
| 逐局 wall | 49.69–67.58 s，合计 1,127.52 s；驱动器 1,164 s（含 dry run） |
| 引擎 / checkpoint | `691bfd77…` / `f447664c…`（20 局 `error=null`） |
| seed 政策 | 只请求 dev 1000-1019；确认集 60000-60199 与 D410-S1 语料 seeds **未被请求** |

### 6.2 逐局

| seed | frames | steps | score | idle % | unique | wall s |
|---|---|---|---|---|---|---|
| 1000 | 1,538 | 769 | 11,789 | 64.11 | 8 | 54.83 |
| 1001 | 1,945 | 973 | 18,284 | 51.70 | 12 | 65.21 |
| 1002 | 2,011 | 1,006 | 19,959 | 55.17 | 15 | 64.91 |
| 1003 | 1,812 | 906 | 19,481 | 55.85 | 9 | 55.83 |
| 1004 | 1,631 | 816 | 13,749 | 60.91 | 9 | 50.51 |
| 1005 | 1,776 | 888 | 22,108 | 60.02 | 8 | 54.23 |
| 1006 | 1,862 | 931 | 16,681 | 58.32 | 10 | 58.40 |
| 1007 | 1,607 | 804 | 13,372 | 61.19 | 13 | 54.58 |
| 1008 | 1,945 | 973 | 28,681 | 52.42 | 11 | 56.86 |
| 1009 | 1,549 | 775 | 11,587 | 63.61 | 9 | 50.17 |
| 1010 | 1,520 | 760 | 10,860 | 64.87 | 10 | 49.69 |
| 1011 | 1,613 | 807 | 12,853 | 62.95 | 13 | 51.38 |
| 1012 | 1,608 | 804 | 13,153 | 60.82 | 9 | 58.99 |
| 1013 | 1,546 | 773 | 11,899 | 63.13 | 8 | 56.64 |
| 1014 | 1,538 | 769 | 11,829 | 63.46 | 7 | 55.57 |
| 1015 | 1,535 | 768 | 11,749 | 64.06 | 8 | 55.30 |
| 1016 | 1,535 | 768 | 11,749 | 64.06 | 8 | 56.17 |
| 1017 | 1,945 | 973 | 17,338 | 54.78 | 12 | 67.58 |
| 1018 | 1,550 | 775 | 11,083 | 63.74 | 13 | 54.09 |
| 1019 | 1,800 | 900 | 16,061 | 57.89 | 12 | 56.58 |

（wall 是**争用下**的值，见 §8；帧数由引擎按帧预算推进，不是实时墙钟的函数。）

### 6.3 动作分布（20 局合计 16,938 decisions）

| idx | 类 | 次数 | 占比 |
|---|---|---|---|
| 0 | idle | 10,123 | **59.77 %** |
| 1 | N | 22 | 0.13 % |
| 2 | NE | 512 | 3.02 % |
| 3 | E | 2,509 | **14.81 %** |
| 4 | SE | 39 | 0.23 % |
| 5 | S | 60 | 0.35 % |
| 6 | SW | 521 | 3.08 % |
| 7 | W | 2,796 | **16.51 %** |
| 8 | NW | 53 | 0.31 % |
| 9 | focus-idle | 0 | 0.00 % |
| 10 | focus-N | 0 | 0.00 % |
| 11 | focus-NE | 130 | 0.77 % |
| 12 | focus-E | 141 | 0.83 % |
| 13 | focus-SE | 2 | 0.01 % |
| 14 | focus-S | 0 | 0.00 % |
| 15 | focus-SW | 26 | 0.15 % |
| 16 | focus-W | 3 | 0.02 % |
| 17 | focus-NW | 1 | 0.01 % |

- idle + E + W = **91.09 %**（5k none：89.04 %）→ 闭环行为**更集中**，不是更分散。
- 竖直/斜向 6 类（N/NE/SE/S/SW/NW）合计 **7.13 %**（5k none：8.80 %）→ 离线 recall 涨了，
  闭环里反而**少用**了。

## 7. 与 5k none / 20k none 并排（描述性）

### 7.1 训练

| arm | steps × batch | samples | loss | top1 | macroR | 吞吐 |
|---|---|---|---|---|---|---|
| β0.5 5k none（WP05.8） | 5,000×128 | 640,000 | 0.6586 | 0.7993 | 0.2733 | 1,512.8 samples/s |
| **β0.5 5k idle-cap 0.44（本轮）** | **5,000×128** | **640,000** | **0.6568** | 0.7973 | **0.3142** | 2,556.3 samples/s |
| β0.5 20k none（WP05.11） | 20,000×128 | 2,560,000 | 0.3245 | 0.9005 | 0.5904 | — |

### 7.2 dev 20 局

| arm | clear | mean | median | min / max | sd | idle | unique | idle+E+W |
|---|---|---|---|---|---|---|---|---|
| β0.5 5k none | **0/20** | 1,689.10 | 1,605.0 | 1,530 / 2,213 | 204.38 | 62.84 % | 12.5 | 89.04 % |
| **β0.5 5k idle-cap 0.44** | **0/20** | **1,693.30** | **1,610.5** | 1,520 / 2,011 | **172.94** | **59.77 %** | **10.2** | **91.09 %** |
| β0.5 20k none | **0/20** | 1,714.65 | 1,619.5 | 1,523 / 2,332 | 207.67 | 60.87 % | 12.45 | — |

### 7.3 差值

| 对比 | mean | median | idle | unique | clear |
|---|---|---|---|---|---|
| idle-cap **vs** 5k none（受控：只改 idle 权重） | **+4.20 帧（+0.25 %）** | +5.5 | **−3.08 pp** | **−2.3** | 0/20 → 0/20 |
| idle-cap **vs** 20k none（同时差预算，非受控） | −21.35 帧（−1.25 %） | −9.0 | −1.11 pp | −2.25 | 0/20 → 0/20 |

**怎么读（这是本轮的核心判断）**：

1. **权重生效，闭环不动。** 离线：macroR +4.09 pp、16/16 有标签的非 idle 类 recall 上升、idle recall −3.45 pp；
   闭环：mean +0.25 %、median +5.5 帧、idle −3.08 pp、clear 仍 0/20。把「先验」在 loss 里压 17.72 pp，
   只换来 dev idle 3.08 pp 的下移和 4 帧均值变化（n=20、sd≈173、SE≈39 帧 → 完全在噪声内）。
   **因此在这个预算下，"idle 先验"不是闭环 0/20 的主因**；继续在先验上做文章不会把 0 变 1。
2. **H1 的 falsifier 被触发（但边界要说清）。** IDLE_DIAG H1 的推翻条件是「新先验 ~45 % 而 dev idle
   仍 >58 %」：本轮 dev idle **59.77 %**，满足。但要精确：本轮改的是 **loss 侧**先验（样本加权），
   **标签分布仍是 61.72 %**；IDLE_DIAG 想到的「先验」还有**标签侧**那一条（R1：改 tie-break 重新标注，
   H2）。所以结论写成：**"loss 侧 idle 先验决定 dev idle" 不成立**；标签侧先验是否也会被闭环忽略，
   本轮没有测。
3. **闭环在"更集中"而不是"更分散"。** 唯一略降的 idle 被 E/W 吃掉一部分（E 12.12→14.81 %、
   W 14.08→16.51 %），竖直/斜向 6 类从 8.80 % 降到 7.13 %，unique 12.5→10.2，sd 204→173（上尾
   从 2,213 收到 2,011）。这更像「同一个卡死/贴边循环换了个更窄的动作集」，而不是「学会了新动作」。
   与 IDLE_DIAG H3（idle 标签由采集学生的长 run 供应）和 H4（竖直方向被饿死）方向一致。
4. **和 20k 的关系。** 20k（4 倍预算、无权重）mean 1,714.65、idle 60.87 %；本 arm（1/4 预算 + 权重）
   mean 1,693.30、idle 59.77 %。两者差 21 帧（−1.25 %）、idle 差 1.11 pp —— **预算加 4 倍、或把 idle
   先验压到 44 %，闭环都停在 ~1,700 帧 / ~60 % idle / 0 通关**。这与 IDLE_DIAG R4（"不要在现有语料上
   加步数"）一致，并补上一条：**在现有语料上改 loss 权重的边际收益也很小**。

## 8. 并发争用记录（只观测，不干预）

| 项 | 训练窗口 | dev 窗口 |
|---|---|---|
| 并发 round3 DAgger | `v410_dagger_collect.py --seeds 2040-2054 --beta 0.5`（PID 1049814/1049835）**在跑** | **同一进程仍在跑**（15:50 结束时仍在） |
| load average | 15.61 起、17.62 止 | 15.07 起、**22.79** 止 |
| GPU | 1,584 MiB / 0 %（首尾） | 1,770 MiB / 8 %（结尾） |
| 操作 | 仅 `pgrep`/`nvidia-smi` 记录 | 同左；**未 kill、未发信号、未改它的输出** |
| 影响 | 吞吐 2,556 samples/s 与 5k 的 1,512 不可比 | 逐局 wall 49.7–67.6 s（5k 为 20.9–27.9 s，约 **2.5×**） |

引擎按**帧预算**推进（`--max-frames 19887` + `--stop-on-death`），不是实时推进，所以争用主要落在
墙钟/吞吐上；但这一点无法在本轮内独立验证，已写进 §10 限制 7。

## 9. 测试（51 passed，旧 34 不破坏）

新增用例（`tests/test_v410_dagger_train.py`，17 个 test / 13 个函数）：

- 三种模式：`none` 全 1 且 `applied=false`；`invfreq` 每个出现类隐含份额相等（1/17）、缺失类权重 0；
  `idle-cap` 隐含 idle 份额 = 0.44 ± 1e-9、idle 权重严格小于其他类、`λ` 与闭式解一致、平均权重 = 1。
- 目标份额不可达即拒绝：`t ≥ p_idle`（0.617、0.7、0.9 三档）→ fail-closed。
- 表本身的 fail-closed：非法 mode、长度不是 18、负计数、全零、`t ∉ (0,1)`、idle 计数为 0、非法
  `idle_action`。
- `expected_mix_counts` 的配额缩放与非法输入。
- `class_weighted_ce` 与手算 weighted mean 一致（含 `unweighted_ce`）。
- **默认行为不变**：`none` 的 loss/top1/macroR/分源 loss 与改造前记录的
  `2.85571825504303 / 0.125 / 0.07142857142857142 / 2.8797269662221274 / 2.850177764892578`
  逐个一致（per-source 容差 1e-5，跨平台归约顺序；其余 1e-9），且 `mix.rule` 字符串逐字相同、
  `sample_weight == 1.0`、checkpoint 的 `config` **没有** class_weight 键。
- `none` 显式传入 == 默认（metrics / mix / class_weight 全等）。
- `idle-cap` 的微型训练：与 `none` 的**批次/标签计数/数据集完全相同**（只 loss 权重变），
  `metrics.weighted_loss`、`blocks[].weighted_loss` 出现，checkpoint `config.class_weight` 记录权重表。
- CLI：非法 `--class-weight-mode`、非法 `--idle-target-share`（0 / 1.5 / −0.1）→ `EXIT_TRAIN_FAILED(3)`
  且不写 out 文件；parser 默认值 pinned（`none` / 0.44，步数仍 1,000、share 仍 0.2）。

## 10. 限制

1. **一个 arm、一个训练种子（0）、一个权重点（t=0.44）**：没有任何方差估计；4.20 帧的均值变化在
   sd≈173、n=20 的噪声内，不能当作效应量。也不是权重扫描。
2. 离线指标是**训练批上的 in-sample** 指标（同一批 128 行），不是留出精度。
3. 5,000 步仍是**同一批** 46,397 S1 + 52,500 DAgger 行的重复（3 / 10 epochs），还是一轮 DAgger。
4. 本轮改的是 **loss 侧**先验；**标签侧**先验（tie-break 重标注，IDLE_DIAG R1/H2）**没有测**。
5. 权重只在 chosen one-hot CE 上生效，不改状态分布：如果 idle 标签来自采集学生在长 run 里卡住的状态
   （H3），loss 权重不可能改变这些状态的到达频率——这正是本轮 3 pp 量级的解释候选。
6. 与 5k/20k 的对比都只有**一个训练 seed**，20 局的差值不能与 seed 噪声分开，也没有区间/显著性检验
   （本文件不写门）。
7. 训练与 dev 全程有 round3 DAgger 采集并发（load 15→23）：墙钟与吞吐**不可**与其它轮次受控比较；
   帧数按引擎帧预算推进，理论上与争用无关，但本轮没有隔离复跑可以证明。
8. dev 只有 20 个 dev seed（1000-1019）；**G410-2 未测**，确认集 60000-60199 未被请求。
9. 引擎 `train.py` 的 `scope.purpose` 仍是 WP05.3 模板（写着 "1,000 steps"、"not a beta=0.5 round"），
   本轮把它原样记录在 `train.scope_as_written_by_the_tool`，**不据此作任何主张**。

## 11. not_claimed / 边界

- **不是 G410-2**：无 confirm 集、无四臂矩阵、无预注册决策规则、无门。
- **不是开发得分**：0/20 是描述性计数；Wilson 上界 0.1611 只是二项描述，不是通过率。
- **不是新数据**：未采集、未重跑教师、未确认种子、未改引擎；游戏只被评测器启动 20 次。
- **不声称 idle 先验"无关"**：本轮只否证了「loss 侧先验决定 dev idle / 决定闭环结果」这一条弱命题；
  标签侧先验与状态分布仍未分离。
- **不声称权重"没用"**：它在离线目标上明确生效（macroR +4.09 pp、16/16 有标签非 idle 类 recall 上升），只是没有
  转成闭环收益。
- **不声称 5,000 步够 / 20k 是最优**：两个点都停在 0/20。
- **不声称任何带区间的结论**。
- 未 commit / 未 push；未修改白名单以外任何文件（含 v400/v380/v390、`tools/rl/v400_*`、
  `tools/rl/v410*.py`、`tests/test_v410*` 其它文件、`third_party/`、v410 其它既有文件、
  state/journal/budget/assets、其它 WP 证据）。
