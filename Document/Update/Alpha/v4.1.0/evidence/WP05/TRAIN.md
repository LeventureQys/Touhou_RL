# WP05.3 — DAgger shards 标签转换 + S1+DAgger 混合 B0 训练 smoke + 2 dev episodes

- 时间：2026-11-13T08:10:00+08:00
- 结论：**已完成**。6 个 `complete=true` 的 WP05.0 DAgger shard 全部用 WP05.2 CLI 转成 `labels_v410`（21,000 decisions / 42,000 frames，labels+obs SHA 全部校验通过、聚合 counts 一致）；B0（731,802 参数）在 **S1 20 % / DAgger 80 %（按样本数，实际 25,600 / 102,400）** 混合上跑满 **固定 1,000 steps × batch 128**（loss 0.8324 / top1 0.7549 / macroR 0.2052，2,079.9 samples/s，RTX 4070 SUPER）；训练后 checkpoint 在真实 `rl_v3` 引擎上跑 dev seeds 1000/1001 两局（1,609 / 1,200 帧，均首死停止、0 次过关）。**n=2 不是开发成绩、不是通过率、不构成 G410-2 结论**；本次不采数据、不做 50k 正式轮、不做 β=0.5 轮、不改 G410-2。
- 证据：[`train_smoke.json`](train_smoke.json)（49,530 B）；运行目录 `.tmp_v410_wp05_train/`（labels/checkpoint/logs/eval，`.gitignore` 的 `.tmp_*/` 已忽略，不入库）。

## 0. 关键数字一览

| 项 | 值 |
|---|---|
| DAgger shards | 6 个（seed2000–2005），每个 3,500 decisions / 7,000 frames，`complete=true` |
| 标签转换 | WP05.2 CLI ×6，全部 exit 0；tolerance 8.0、horizon 32 |
| 聚合 decisions / frames | **21,000 / 42,000**（50k 的 42 %） |
| 生存标签（teacher-plan） | alive **20,963**、dead **0**、unknown **37**（全部 model_mismatch） |
| query 覆盖 | 21,000 / 378,000 = **5.5556 %**；query_status↔verified_mask 每 shard 3,500/3,500 |
| 注入 | injected **21,000/21,000**（β=1.0）；student–teacher agreement **14,444** / disagreement **6,556**（68.78 % / 31.22 %） |
| S1 训练集 | 46,397 rows（seeds 1-14，train split only；val/test 从未打开） |
| 混合 | 名义 **S1 20 % / DAgger 80 %**；实际 **25,600 / 102,400 = 20.000 % / 80.000 %**，每样本权重 1.0 |
| 模型 | B0 `ProbeNet`，**731,802** 参数，chosen one-hot CE |
| 训练预算 | **1,000 steps × batch 128 = 128,000 samples**；AdamW 3e-4 / wd 1e-4 / clip 5.0；seed 0 |
| checkpoint | `.tmp_v410_wp05_train/b0_s0_dagger_step1000.pt`，2,941,187 B，SHA-256 `e771314d736bf5bd4df0651cab9d9a9780aa067da3a7fbc892cd6ad5cec66824` |
| 训练指标（全窗口） | loss **0.8324**、top1 **0.7549**、macroR **0.2052**；首 step 2.9043 → 末 step 0.5459 |
| 吞吐 | **2,079.9 samples/s / 16.25 steps/s**；训练环 61.54 s（数据 32.91 s = 53.5 %），中位 step 60.6 ms |
| GPU | NVIDIA GeForce RTX 4070 SUPER（12,281.5 MiB）；峰值 allocated 252.11 MiB / reserved 434.0 MiB |
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，SHA-256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` |
| dev 口径 | S1N：`--stage 0` + `normal` + `--max-frames 19887` + `--stop-on-death` + `--obs-window consecutive` |
| dev 结果 | seed1000 **1,609 帧**（805 steps，score 13,483）；seed1001 **1,200 帧**（600 steps，score 9,689）；均 `death_stop`、0 过关 |
| B0 baseline（WP04.3，同口径） | seed1000 1,398 帧 / seed1001 818 帧；本次 **+211 / +382** |
| pytest | `tests/test_v410_dagger_train.py` → **34 passed**，exit 0 |
| compileall | `python -m compileall -q touhou_rl/v410/dagger` → exit 0 |

## 1. 交付物

| 文件 | 作用 |
|---|---|
| [`touhou_rl/v410/dagger/train.py`](../../../../../../touhou_rl/v410/dagger/train.py) | WP05.3 训练入口：shard/conversion 交叉校验 + 聚合 counts、S1+DAgger 混合 loader、固定 1,000 步 B0 smoke、checkpoint 写出、`report` 证据合并；**不启动游戏、不采数据、不训练其他 arm** |
| [`tests/test_v410_dagger_train.py`](../../../../../../tests/test_v410_dagger_train.py) | 34 项 CPU 合成单测：shard/conversion/HASH 校验、loader、target 规则、unknown 不作负例、混合权重、CHECKPOINT 可被 WP04.3 evaluator 加载、report、CLI fail-closed |
| [`train_smoke.json`](train_smoke.json) | 本次 smoke 的机器可读证据（labels/shards/decisions、混合权重、训练指标、2 dev 结果、not_claimed/limits） |
| `.tmp_v410_wp05_train/` | 运行目录：`labels/dagger-s0d1seed2000..2005/`（labels_v410.jsonl + summary）、`b0_s0_dagger_step1000.pt`、`train_run.json`、`eval_dev.json`、`dryrun_eval.json`、`logs/train.log`；**不入库** |

未修改 `v400/v380/v390`、`tools/rl/v400_*`、`tests/test_v400_*`、`third_party/`、`tools/rl/v410*.py`、`tests/test_v410_*` 既有文件、`state/journal/budget/assets`、WP03/WP04/WP05 既有证据；未 commit/push；未删除/覆盖任何旧文件。

## 2. 命令与退出码（实际运行）

所有 PowerShell 命令先设 `$env:PYTHONPATH='D:\WorkShop\Touhou_RL'`。

### 2.1 标签转换（WP05.2 CLI，6 次）

```powershell
foreach ($seed in 2000..2005) {
  python -m touhou_rl.v410.dagger.labels `
    --shard .tmp_v410_wp05_round1/shards/dagger-s0d1seed$seed `
    --out   .tmp_v410_wp05_train/labels/dagger-s0d1seed$seed `
    --tolerance 8.0
}
# 2000=0 2001=0 2002=0 2003=0 2004=0 2005=0
```

每次 CLI 输出形如 `decisions=3500 plans=3500 alive=3492 dead=0 unknown=8 model_mismatch=8 incomplete=0 query_verified=3500/63000 query_coverage=0.055556 injected=3500 disagreement=1167 tolerance=8.0 horizon=32`，并写出 `labels_v410.jsonl` + `labels_v410.summary.json`。

### 2.2 训练 smoke（Windows，CUDA）

```powershell
python -m touhou_rl.v410.dagger.train train `
  --steps 1000 --batch-size 128 --seed 0 --s1-share 0.2 --device auto `
  --workdir .tmp_v410_wp05_train --out .tmp_v410_wp05_train/train_run.json
# dagger_train: steps=1000 samples=128000 s1=25600 dagger=102400
#   loss=0.8324 top1=0.7549 macroR=0.2052 2079.9 samples/s
#   checkpoint=.tmp_v410_wp05_train\b0_s0_dagger_step1000.pt
# EXIT_CODE=0（进程 elapsed 70.7 s；payload elapsed 68.05 s）
```

日志：`.tmp_v410_wp05_train/logs/train.log`。

### 2.3 dev 评测（WSL，`xvfb-run`）

先做一次 60 帧接线 dry run（不是证据，只验证新 checkpoint 能被 evaluator 加载）：

```bash
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_student_eval.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp05_train/b0_s0_dagger_step1000.pt \
  --seeds 1000 --stage 0 --difficulty normal --max-frames 60 --stop-on-death \
  --out .tmp_v410_wp05_train/dryrun_eval.json
# wrote .tmp_v410_wp05_train/dryrun_eval.json (exit 0)
# EXIT_CODE=0
```

真实 2 episodes（命令在 `/mnt/d/WorkShop/Touhou_RL` 下执行）：

```bash
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_student_eval.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp05_train/b0_s0_dagger_step1000.pt \
  --seeds 1000,1001 --stage 0 --difficulty normal --max-frames 19887 --stop-on-death \
  --out .tmp_v410_wp05_train/eval_dev.json
# wrote .tmp_v410_wp05_train/eval_dev.json (exit 0)
# EXIT_CODE=0（外层 elapsed 98.3 s；episode wall 53.65 s + 42.07 s）
```

### 2.4 证据合并

```powershell
python -m touhou_rl.v410.dagger.train report `
  --run .tmp_v410_wp05_train/train_run.json `
  --eval .tmp_v410_wp05_train/eval_dev.json `
  --baseline Document/Update/Alpha/v4.1.0/evidence/WP04/eval_smoke.json `
  --out Document/Update/Alpha/v4.1.0/evidence/WP05/train_smoke.json
# dagger_report: decisions=21000 frames=42000 dev={1000: 1609, 1001: 1200}
#   baseline={'1000': 1398, '1001': 818}
# EXIT_CODE=0
```

`report` 会复核 evaluated checkpoint 的 SHA-256 与训练 checkpoint 一致（不一致即拒绝），并要求恰好 2 个 dev episodes。

### 2.5 验收命令

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp053_pytest -q tests/test_v410_dagger_train.py
# 34 passed in 4.75s
# EXIT_CODE=0

python -m compileall -q touhou_rl/v410/dagger
# EXIT_CODE=0
```

## 3. 标签转换结果（6 shards，WP05.2 规则）

每个 shard 的 `labels.jsonl` 行数、`shard.json.decisions`、obs 三平面行数与 SHA-256 全部一致（CLI 内 `verify_sha` 默认开启，summary 中 `labels_sha256_verified` / `obs_sha256_verified` 均为 true）；`train.py` 读入时再对 `labels_v410.jsonl` 重新计算 SHA-256 与 summary 比对，并按 `s1_data.parse_label_row` 逐行校验。

| shard | decisions | frames | alive | dead | unknown | injected | agreement | disagreement | labels SHA-256（前 8） |
|---|---|---|---|---|---|---|---|---|---|
| dagger-s0d1seed2000 | 3,500 | 7,000 | 3,492 | 0 | 8 | 3,500 | 2,333 | 1,167 | `3494caa6` |
| dagger-s0d1seed2001 | 3,500 | 7,000 | 3,494 | 0 | 6 | 3,500 | 2,387 | 1,113 | `196c7a97` |
| dagger-s0d1seed2002 | 3,500 | 7,000 | 3,497 | 0 | 3 | 3,500 | 2,378 | 1,122 | `f82d3954` |
| dagger-s0d1seed2003 | 3,500 | 7,000 | 3,492 | 0 | 8 | 3,500 | 2,409 | 1,091 | `286b76dd` |
| dagger-s0d1seed2004 | 3,500 | 7,000 | 3,492 | 0 | 8 | 3,500 | 2,468 | 1,032 | `1a9b09b9` |
| dagger-s0d1seed2005 | 3,500 | 7,000 | 3,496 | 0 | 4 | 3,500 | 2,469 | 1,031 | `fcd9c162` |
| **合计** | **21,000** | **42,000** | **20,963** | **0** | **37** | **21,000** | **14,444** | **6,556** | — |

- 37 个 `unknown` 全部来自 model-mismatch（同 WP05.2 seed2000 的口径：误差 8.000003–8.000031 > tolerance 8.0），本轮 **0 个 verified_dead**，因此 DAgger 侧死亡通道未被真实数据覆盖（仅单测覆盖）。
- 每个 shard 的 `query_verified_candidates` 都是 3,500（= decisions，每行恰 1 个候选被验证），`candidate_grid` 63,000，coverage 3,500/63,000 = 5.5556 %。
- `labels_v410.jsonl` 路径：`.tmp_v410_wp05_train/labels/dagger-s0d1seed2000..2005/`（每目录另含 `labels_v410.summary.json`）。

## 4. 数据混合（预注册口径 + 实际测量）

- **名义**：S1 20 % / DAgger 80 %，chosen one-hot CE，无 soft target、无 class weight、无 candidate mask。
- **规则**：每 step 的 S1 行数 = `floor((step+1)·batch·0.2) − floor(step·batch·0.2)`；batch 内顺序每 step 随机打乱（顺序不携带来源信息）。每样本 loss 权重恒为 1.0；`unknown` 候选只出现在 CE 的 softmax 分母，**从不作为负例、也不进 target**。
- **实际**（整个 1,000 步累积）：

| 项 | 值 |
|---|---|
| S1 samples | **25,600（20.000 %）** |
| DAgger samples | **102,400（80.000 %）** |
| 总 samples | 128,000（= 1,000 × 128） |
| 每 step S1 行数 | 25 行 × 400 steps + 26 行 × 600 steps |
| 采样流消耗 | S1 1 个 epoch（46,397 rows 中抽 25,600）；DAgger 5 个 epoch（21,000 rows 中抽 102,400） |
| rows_without_target | S1 0 / DAgger 0（全部行都有 chosen_plan） |
| unknown candidates（诊断） | S1 788,749 / DAgger 357,000（= 行数 ×17，全部只作诊断） |

## 5. 训练指标（in-sample，训练批次上的实测）

B0：731,802 参数（`ProbeNet`），AdamW lr 3e-4 / wd 1e-4 / clip 5.0，seed 0，固定 1,000 steps、batch 128、CUDA（torch 2.14.1+cu132 / Python 3.12.10 / Windows 11）。

| 指标 | 全窗口（128,000） | S1（25,600） | DAgger（102,400） |
|---|---|---|---|
| loss（chosen one-hot CE） | **0.8324** | 1.6730 | 0.6223 |
| top1 | **0.7549** | 0.4947 | 0.8200 |
| macro recall | **0.2052** | 0.1265 | 0.2333 |
| majority baseline top1 | — | 0.4382 | 0.6860 |

- 学习进程：首 step loss **2.9043** → 末 step **0.5459**；前 100 步均值 1.4340 → 后 100 步均值 0.6143。
- 最后 100 步（step 901–1000）：S1 loss 1.5370 / top1 0.5184 / macroR 0.1699；DAgger loss 0.3836 / top1 0.8898 / macroR 0.4157。
- 梯度：clip 前均值 1.8516（最大 6.2939），clip 后均值 1.8489（最大 5.0）。
- 吞吐：61.54 s 训练环 / 128,000 samples = **2,079.9 samples/s、16.25 steps/s**；数据装载 32.91 s（53.5 %），中位 step 60.6 ms，首 step 602.4 ms（CUDA 预热）。
- GPU：RTX 4070 SUPER，峰值 allocated **252.11 MiB** / reserved **434.0 MiB**（总 12,281.5 MiB）。
- 指标口径说明：这 128,000 个样本就是本次训练消费的批次，**是 in-sample 优化诊断，不是 held-out 准确率**；本次按预注册只读 train split（seeds 1-14），val/test 从未打开，因此不存在 val/test 指标。

## 6. 2 个 dev episodes（真实引擎，n=2）

- checkpoint：`b0_s0_dagger_step1000.pt`（SHA `e771314d…c66824`，schema `v410-wp04-bc-arm-checkpoint/1`，arm B0，731,802 参数）。
- 引擎：`rl_v3` `th08-modern`，SHA `691bfd77…7a568ef6`；`harness_role="env"`、`pixels_warmup_frames=0`、`error=null`、`exit_code=0`。

| seed | frames | steps(=决策) | deaths | lives_end | score | stop_reason | stage_clear | stage_start→end | wall_s | baseline frames | Δ frames |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | **1,609** | 805 | 1.0 | 2.0 | 13,483 | `death_stop` | false | 0→0 | 53.65 | 1,398 | **+211** |
| 1001 | **1,200** | 600 | 1.0 | 2.0 | 9,689 | `death_stop` | false | 0→0 | 42.07 | 818 | **+382** |

- `deaths=1.0` 是 `--stop-on-death` 的构造值（首次死亡即停），不是"整局只死一次"；有信息量的是首死帧数。两局 `lives_end=2.0`、`stage_end=0`、**0 次过关**。
- 动作分布（1,405 次决策池化）：idle 973（**69.25 %**）、E 183（13.0 %）、W 170（12.1 %）、SW 36（2.6 %）、NE 26（1.9 %），其余 ≤6；每局平均 unique actions 9.0。相比 WP04.3 基线（idle 42 %/51 %），本次 idle 明显更高——**只记录，不据此推断原因**。
- 动作序列 SHA：seed1000 `4d8732fd…406988d`；seed1001 `138e67a8…6b8b1a94`；policy input SHA 两局分别 `e15026de…be93e15d` / `26d42a0b…5a71be`。
- **n=2 不是开发成绩**：无置信区间、不能排名、不能称"过门"；`train_smoke.json:dev_eval` 显式带 `n=2 / is_a_gate=false / not_a_development_score=true / ci_reportable=false`。Δ 帧数只描述"这两个种子上的首死帧数差"，**不是效应估计**：两个 checkpoint 的数据混合、训练预算、训练目标都不同。

## 7. 测试覆盖（34 项，`tests/test_v410_dagger_train.py`，实际收集数）

全部 CPU、合成数据、无引擎、不读真实 182 MB shard、不写 `tmp_path` 之外。

- **权重/混合（4 项）**：1,000×128 精确 25,600 S1 / 102,400 DAgger、每步 25/26、share∈(0,1) 拒绝 0/1/负/>1、step 越界拒绝。
- **shard/转换交叉校验（5 项）**：完整 shard 接受；`complete=false` 拒绝；frames 不一致拒绝；conversion decisions 不一致拒绝；labels 文件被追加后 SHA 不一致拒绝；聚合 6 shard counts 正确。
- **loader/target/unknown（7 项）**：target 逐行等于 chosen；无 `chosen_plan` 行被排除并计数（8 行中排除 2 行）；unknown 候选即使分数更高也绝不当 target（chosen 100 vs unknown 300 仍取 5）；dead 候选行仍按 chosen 训练；非 `dagger-label` kind 拒绝；行数与 decisions 不一致拒绝；行级 out-of-range 拒绝。
- **planes/流（4 项）**：DAgger hud 的 `(N,1,56,160)` 与 S1 的 `(N,56,160)` 两种布局都接受、坏布局拒绝；gather 返回归一化 float32 平面与 target；shuffled stream 确定性 + 跨 epoch 续取；S1 只读 train seeds 1-14（val/test 文件被删仍成功 → 证明未打开）。
- **训练 smoke（2 项）**：3 步 CPU 跑通并记录 loss/top1/macroR、per-source counts 与混合一致、`test_split_read=False`；checkpoint 可被 WP04.3 evaluator 的 `load_v410_checkpoint` 以 B0/731,802 参数严格加载（`strict=True`、eval、冻结梯度）；同 seed 两次运行 loss/top1/mix 完全相同（确定性）。
- **report（3 项）**：run + 2 episodes 合并出 schema/labels/decisions/mix/train/dev_eval/not_claimed/limits；checkpoint SHA 不一致拒绝；非 2 episodes 拒绝。
- **CLI（5 项）**：缺 labels root → exit 2 且不写 out；`--steps 0` → exit 3 且不写 out；`--s1-share 1.0` → exit 3；`report` 缺 run → exit 2 且不写 out；parser 默认值 = 预注册预算（1,000/128/seed0/0.2，lr 3e-4、wd 1e-4、clip 5.0）。

## 8. 限制（不声称）

1. **不是 50k 正式轮**：只用了已完成的 6 个 shard（21,000 decisions，约 50k 的 42 %），seed 2020+ 的补充批次未跑。
2. **不是 β=0.5 轮**：这 6 个 shard 全部 β=1.0（教师全注入）。
3. **不是 G410-2**：无 confirm split（60000–60199 未读）、无 20-episode 协议、无四臂矩阵、无通过率/性能门。
4. **不是开发成绩**：2 个 dev episode 没有区间、不能排名；与 WP04.3 B0_s0 的 +211/+382 帧只是描述性差异（数据混合与训练预算不同，不是效应估计）。
5. **离线指标是 in-sample**：128,000 个训练批次上的 loss/top1/macroR 是优化诊断，不是 held-out 泛化；本次按预注册不读 val/test。
6. **只有 B0 / 1 个 seed / 1,000 步**：无 seed 方差、无收敛声明（checkpoint 是第 1,000 步的 last-step 权重，`epochs=1` 只是 WP04 schema 的占位）。
7. **死亡通道未覆盖**：本轮 DAgger 21,000 行 0 个 `verified_dead`，`negative_mask` 恒空；unknown/dead 只由单测覆盖。
8. **数据管线是瓶颈**：数据装载占训练环 53.5 %（32.91 / 61.54 s），GPU 峰值仅 252 MiB；当前吞吐不代表模型上限。
9. **只读保守混合**：S1 20 % 是预注册值，本轮没有做权重消融；S1 上 top1 0.4947 / macroR 0.1265 明显低于 DAgger 上 0.8200 / 0.2333，只记录不解释。

## 9. 下一建议

1. **补齐 50k 第一轮**：跑 seed2020+ 约 29,000 decisions，合并 `round1.json`/`ROUND1.md` 后，用**完全相同的固定预算**（1,000 steps × 128、seed 0、S1 20 %）复跑本 smoke，形成可比的"数据量-效果"点。
2. **β=0.5 第二轮**：同一批 shard 采集时把注入率降到 0.5，用同一 `train.py` 训练同一预算，比较 student–teacher disagreement 分布与离线 top1/macroR；先离线，再决定是否值得开 dev 评测。
3. **多轮 / 多 seed**：β=1.0 → β=0.5 循环 2–3 轮，每轮固定 1,000 步、seed 0/1/2，记录 seed 方差；只有 ≥20 个 dev episodes 且预注册判据时才允许比较"效果"。
4. **把数据管线移出瓶颈**：预取/合并 uint8 批次到 pinned memory、或按 shard 预组 batch，把 53.5 % 的数据时间压下去，再评估是否值得把预算从 1,000 步提到 5k–10k。
5. **S1 权重消融（预注册后再做）**：20 % vs 50 % 两档、同一预算与 seed，检查 S1 top1 与闭环首死帧数是否一起改善；本轮 20 % 是照搬 WP05 计划，不是最优值。
