# WP04.1 — 固定标签规则与步数预算：D410-S1 最小 BC 训练器 + B0/B1 单种子离线 smoke

- 时间：B0 run `2026-11-12T20:24:32Z`、B1 run `2026-11-12T20:24:52Z`、`bc_smoke.json` `2026-11-12T20:24:58Z`
  （即 2026-11-13 +08:00；时间戳取自各 JSON 的 `generated_at`）
- 输入：**D410-S1**（`WP03/s1c_full_audit.json`，`passed=true`）—— 20 个 train shards / 20 seeds（1–20）/ **66,143 decisions**；
  三平面 `(N,4,112,96)` / `(N,56,160)` / `(N,8,18)` uint8 + calibrated `labels_v410.jsonl`
- 设备：**CUDA**，NVIDIA GeForce RTX 4070 SUPER（capability 8.9，12,281.5 MiB），CUDA 13.2，torch **2.14.1+cu132**，Python **3.12.10**
- 范围：**只做最小 BC 训练器与离线 smoke**。未接引擎、未启动游戏、未采数据、**未训练正式四臂**（B2/B3 未跑）、
  **未读 test split**、未做任何 test 调参、未 commit/push、未改 v400/v380/v390/third_party/v410 既有文件
- 结论：训练链路**真实跑通**（exit 0）。B0/B1 各 1 epoch × 363 step + 46,397 样本，train/val 指标、梯度、吞吐、
  GPU 峰值、last-epoch checkpoint 全部落盘；脚本级吞吐 **≈3.9–4.0k samples/s（30.7–31.4 step/s）**，GPU 峰值 **252 MiB**。
  **这不构成 WP04 通过，也不是 G410-2**。

## 0. 关键数字一览

| 项 | B0（chosen one-hot CE） | B1（B0 + bomb BCE aux 0.5） |
|---|---|---|
| 模型 / 参数量 | diagnostics `ProbeNet` / **731,802** | `BombAuxNet` / **731,931**（+129 = bomb head） |
| 训练样本 / 验证样本 | 46,397 / 9,874（test 9,872 未读） | 同 |
| steps（batch 128） | **363 / epoch**，跑满 | **363 / epoch**，跑满 |
| samples_seen | 46,397 | 46,397 |
| **train** loss / top1 / macro recall | **1.6898 / 0.4895 / 0.1066** | **1.7093 / 0.4849 / 0.1026** |
| **val** loss / top1 / macro recall | **1.4220 / 0.5613 / 0.1635** | **1.4491 / 0.5482 / 0.1508** |
| 多数类基线（train / val top1） | 0.4396 / 0.4412 | 同 |
| 吞吐（train loop） | **4,012.7 samples/s**，31.39 step/s，median step 16.1 ms | **3,927.2 samples/s**，30.73 step/s，median 16.8 ms |
| 数据占比（memmap gather） | 43.4%（5.01 s / 11.56 s） | 42.4%（5.00 s / 11.81 s） |
| eval 吞吐 | 6,614.6 samples/s（1.49 s） | 6,397.0 samples/s（1.54 s） |
| GPU 峰值（allocated / reserved） | **252.11 / 436.0 MiB** | **252.11 / 436.0 MiB** |
| 梯度范数 mean/min/max（363 步） | 2.0939 / 0.8908 / 5.2658；**363/363 非零**，0 NaN | 2.4478 / 0.9948 / 5.7717；363/363 非零，0 NaN |
| 触发 grad clip(5.0) 的步数 | 1（post-clip max 5.0） | 1（post-clip max 5.0） |
| 整进程耗时 | 17.97 s | 18.28 s |
| checkpoint | `.tmp_v410_wp04_bc/b0_seed0_last.pt`（2,944,307 B） | `.tmp_v410_wp04_bc/b1_seed0_last.pt`（2,945,401 B） |

- val loss **低于** train loss 不是错误：train loss 是**整个 epoch 的均值**（含前几十步高损失），val 是 **epoch 结束后**统一评估；
  epochs=1 时二者不是同一时刻的量，`train.history[0].loss == loss_first_epoch == loss_last_epoch` 只是 1 epoch 的必然结果
- val top1（0.5613）高于多数类基线（0.4412）→ 1 epoch 内确实学到了**超出先验**的动作区分；
  macro recall 只有 **0.1635**，`predicted_distribution` 集中在动作 0（72.2%，目标占比 44.1%）→ 长尾动作基本没学会
- B1 三项 val 指标都比 B0 差（loss +0.0271，top1 −0.0131，macroR −0.0127）。**单种子 / 1 epoch / 辅助标签恒为 0**，
  三者都不足以支持任何“aux 有效/无效”的结论，只记录数值

## 1. 预注册（训练前固定，写入 `configs/v410/bc_s1_arm.json`）

`bc.py` 内 `PREREGISTRATION` 是 fail-closed 参照，`load_arm_config()` 逐字段比对配置；改动任一被校验字段都会报错而不是静默换实验。
机器可读副本在 `bc_smoke.json → preregistration`。

### 1.1 标签规则

| 字段 | 值 | 含义 |
|---|---|---|
| `primary_target` | `chosen_onehot_ce` | 主目标 = `chosen_plan.first_action` 的 one-hot 交叉熵 |
| `soft_target` | `false` | 不用 KL / verified-support 软目标 / τ；WP04.0 已证明 S1 上软目标退化为 one-hot CE（98.27% KL 来自 −5.0 discount） |
| `bomb_aux` | `binary_cross_entropy_with_logits` | 辅助项 = 该行 `bomb` 标签的 BCE |
| `bomb_aux_weight` | `B0=0.0` / `B1=0.5` | 只有 B1 有 bomb head 且参与损失；权重与 head 必须同时存在，否则报错 |
| `unknown_role` | `diagnostic_only` | `query_status`/`survival_status` **不进损失**：无候选 mask、无 class weight、无 margin/ranking 项 |
| `unknown_used_as_negative` | `false` | unknown/dead 候选**从不作负例**；非 chosen 候选唯一出现的位置是 CE 的 softmax 分母 |
| `rows_without_chosen` | `excluded_and_counted` | 无 chosen_plan 的行从 CE 中剔除并计数，**不映射到任何类别**（本次真实数据 0 行） |

- 明确写出与 WP04.0 报告措辞的差异：`DIAG.md` 说“其余 17 个 logit 不进损失”，严格说 CE 的 softmax 分母仍包含全部 18 个 logit；
  本任务“unknown 不作负例”的准确含义是**没有候选级负标签/权重/排序项**，这一点在 `label_rule` 与测试里都固定了

### 1.2 切分（按 seed 整轨迹 70/15/15）

| split | seeds | 说明 |
|---|---|---|
| train | **1–14** | 46,397 行 |
| val | **15–17** | 9,874 行 |
| test | **18–20** | 9,872 行，**只登记不读取**（`split.test_split_read=false`、`checks.test_split_read=false`） |

- 同一 shard 的行**永不跨 split**：切分按 seed，`assert_shard_atomic()` 记录 `cross_split_shards=0`、`shards_per_split={train:14,val:3,test:3}`
- 测试 `test_train_and_val_loaders_never_open_test_shards` 用 monkeypatch 记录真正被打开的 shard：只有 seed 1–17 被打开
- 20 seed 的切分是**字面冻结**的：`plan_split()` 对 1–20 的结果若与 1–14/15–17/18–20 不符，直接 raise

### 1.3 步数预算与优化器

| 字段 | 值 |
|---|---|
| epochs | **20**（正式预算；本次 smoke 用 `--epochs 1`，见 §9 限制） |
| batch size | **128**（显存不足才允许 64，且必须记录；本次**未触发**，`batch_size_fallback.used=false`） |
| steps_per_epoch | `ceil(train_samples / batch_size)` = `ceil(46397/128)` = **363**，**固定跑满**（`_fit` 校验 `steps == 363×epochs`，不符即 raise） |
| optimizer | AdamW，`lr=3e-4`，`weight_decay=1e-4`，`grad clip=5.0` |
| 增强 | 关闭 |
| eval batch | 128（WP04.0 实测 batch 1 vs 8 在 CUDA 上差 ~4e-4，故固定单一批大小） |
| 记录字段 | `samples_seen` / `steps` / `samples_per_second` / `gpu.max_memory_{allocated,reserved}_mib` |

### 1.4 模型

| arm | 类 | 参数量 | 说明 |
|---|---|---|---|
| **B0** | `touhou_rl.v410.student.diagnostics.ProbeNet` | **731,802** | 与 WP04.0 完全同一参数空间（测试断言与 `diag.build_model` 一致） |
| **B1** | `touhou_rl.v410.student.bc.BombAuxNet` | **731,931** | B0 + `Linear(trunk=128, 1)` bomb head（129 参数）+ aux weight 0.5；move logits 与 B0 **逐位相同**（测试 `torch.equal`） |
| B2 / B3 | —— | 2.6 M | **不在本 smoke 范围**，`arm_spec`/CLI 均拒绝 |

### 1.5 checkpoint

- 位置：`.tmp_v410_wp04_bc/<arm>_seed<seed>_last.pt`（`.tmp_*/` 已被 `.gitignore` 忽略，**不入库**）
- 内容：**最后 epoch 的 `state_dict`** + `model_config` + arm spec + 参数数 + **切分**（`split.as_json()`）+ budget（steps/steps_per_epoch/samples_seen）+ **train/val 指标** + 完整预注册 config + 优化器设置

## 2. 命令与退出码（实际运行，PowerShell，均先设 `$env:PYTHONPATH='D:\WorkShop\Touhou_RL'`）

| # | 命令 | 退出码 |
|---|---|---|
| 1 | `python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp041_pytest -q tests/test_v410_student_bc.py` | **0**（`53 passed in 5.54s`） |
| 2 | `python -m compileall -q touhou_rl/v410/student` | **0** |
| 3 | `python -m touhou_rl.v410.student.bc --arm B0 --seed 0 --epochs 1 --out .tmp_v410_wp04_bc/b0_s0.json` | **0** |
| 4 | `python -m touhou_rl.v410.student.bc --arm B1 --seed 0 --epochs 1 --out .tmp_v410_wp04_bc/b1_s0.json` | **0** |
| 5 | `python -m touhou_rl.v410.student.bc --assemble .tmp_v410_wp04_bc/b0_s0.json .tmp_v410_wp04_bc/b1_s0.json --out Document/Update/Alpha/v4.1.0/evidence/WP04/bc_smoke.json --summary` | **0** |
| 6 | （回归，非验收项）`python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp041_pytest -q tests/test_v410_student_diagnostics.py` | **0**（`51 passed`，WP04.0 未被破坏） |

- pytest 统计：**53 passed in 5.54s**，全部合成 fixture、**CPU、不读写真实 shard**；`--basetemp .tmp_v410_wp041_pytest` 绕过 `.pytest_cache` 的 ACL 问题
- 日志：`.tmp_v410_wp04_bc/{b0_s0,b1_s0,assemble,acceptance_pytest}.log`；run JSON：`.tmp_v410_wp04_bc/{b0_s0,b1_s0}.json`

### 测试覆盖（53 项，对应本任务验收要求）

| 覆盖点 | 代表用例 |
|---|---|
| 按 seed 切分 / 整 shard 不跨 split | `test_plan_split_is_the_frozen_20_seed_cut`、`test_make_loaders_groups_whole_seeds`、`test_shard_atomicity_is_checked`（14/3/3） |
| test split 从不打开 | `test_train_and_val_loaders_never_open_test_shards`（monkeypatch 记录被打开的 shard） |
| target 对齐（行→平面→one-hot） | `test_targets_follow_the_chosen_plan_and_the_pixel_rows`（逐行比对 memmap 原始行与 `pack_batch`） |
| B0/B1 forward + 参数空间 | `test_b0_is_the_diagnostics_parameter_space`、`test_b1_is_b0_plus_the_bomb_head`、`test_b1_move_head_is_bitwise_the_b0_network`、`test_forward_rejects_wrong_plane_shapes` |
| loss / grad | `test_b0_loss_is_the_one_hot_cross_entropy`、`test_b1_loss_adds_half_the_bomb_bce`、`test_gradients_are_non_zero_finite_and_clipped`、`test_gradient_clipping_binds_when_the_norm_is_large`、`test_a_few_steps_reduce_the_loss_on_one_batch` |
| checkpoint 字段 | `test_train_arm_writes_a_last_epoch_checkpoint`（schema/arm/seed/params/state_dict/split/metrics/config） |
| unknown 不参与 CE | `test_rows_without_a_chosen_plan_are_excluded_not_mapped`、`test_unknown_candidates_never_become_negatives`（三种标签状态 → targets/onehot/bomb 完全相同）、`test_unverified_chosen_plan_is_still_the_target` |
| CLI 参数 | `test_cli_parser_defaults_and_choices`、`test_cli_runs_and_writes_the_run_json`、`test_cli_returns_2_on_a_bad_audit`、`test_cli_assemble_path` |
| 预注册不可漂移 | `test_pre_registration_file_still_matches_the_frozen_rules`、`test_canonical_config_matches_the_file`、`test_config_tampering_is_refused`（6 种篡改） |
| 指标 | `test_metrics_macro_recall_uses_present_classes_only`、`test_confusion_matrix_orientation`、`test_evaluate_reports_the_split_and_never_a_negative_example`、`test_evaluate_b1_reports_the_bomb_head` |

## 3. 数据与切分（实际读到的量）

| 项 | 值 |
|---|---|
| shard / seed 数 | 20 / 20（`seed_class=train`，dev 1000–1099、confirm 60000–60199 由 `s1_data` fail-closed 拒绝） |
| 总 decisions | 66,143 |
| train / val / test 行 | **46,397 / 9,874 / 9,872** |
| train seed 覆盖 | 14 个 seed 全在（`summary.shards=14`） |
| 每行 chosen | 46,397 / 46,397（**0 行缺 chosen**） |
| chosen ∈ verified | **100%**（46,397 / 46,397） |
| verified 候选 / alive / dead | 46,397 / **46,174** / **0** |
| `bomb` 正例（train） | **0 / 46,397**（20 个 labels 文件全量扫描也是 0 / 66,143） |
| robust 生存状态 unknown | 223 行（46,397 − 46,174，保守规则，非 dead） |
| 加载/对齐 | `s1_data.read_label_rows` 全量校验（verified 数组 vs `query_status`、`survival_counts` 一致性） |

- **`bomb` 标签在 D410-S1 上恒为 0**（本次 smoke 读到 56,271 行正例 0；全量 66,143 行扫描同为 0）。
  因此 B1 的辅助 BCE 退化为“恒预测 0 的先验”：val `bomb_bce` 从 ln2≈0.693 降到 **0.000895**，`bomb_accuracy=1.0`。
  B1 与 B0 的差异**不能**作为 aux 标签收益的证据

## 4. B0 / B1 指标（真实 D410-S1，CUDA）

### 4.1 train / val 汇总

| 指标 | B0 train | B0 val | B1 train | B1 val |
|---|---|---|---|---|
| loss（chosen one-hot CE） | 1.6898 | **1.4220** | 1.7093 | **1.4491** |
| top1 | 0.4895 | **0.5613** | 0.4849 | **0.5482** |
| macro recall | 0.1066 | **0.1635** | 0.1026 | **0.1508** |
| 样本数 | 46,397 | 9,874 | 46,397 | 9,874 |
| 多数类基线 top1 | 0.4396 | 0.4412 | 0.4396 | 0.4412 |
| 出现类别数 | 17 / 18（动作 9 缺失） | 17 / 18 | 17 / 18 | 17 / 18 |
| 非零 recall 的类别数 | —— | 7 / 17 | —— | 8 / 17 |
| bomb BCE / accuracy | —— | ——（B0 无 aux） | 0.000895（val） | **0.000895 / 1.0** |

### 4.2 val 动作分布（B0，目标 vs 预测）

| 动作 | 0 | 2 | 3 | 7 | 11 | 12 | 其它 11 类 |
|---|---|---|---|---|---|---|---|
| 目标占比 | 0.4412 | 0.0877 | 0.1014 | 0.0701 | 0.0612 | 0.0702 | ≤0.0244 各 |
| B0 预测占比 | **0.7221** | 0.1191 | 0.0560 | 0.0617 | 0.0090 | 0.0215 | 0（collapse） |
| B0 recall | 0.941 | 0.447 | 0.364 | 0.561 | 0.129 | 0.237 | 动作 6 = 0.101，其余 10 类 = 0 |

- 读法：1 epoch 后模型已学到“动作 0 最常被选 + 动作 2/3/7 有结构”，但预测分布**向多数类坍塌**（72.2% vs 44.1%），
  长尾类 recall 为 0 → 这是 **363 step 预算下的欠拟合**，与 WP04.0“256 样本 40 step 不够”的结论方向一致
- val 混淆矩阵 18×18 在 `bc_smoke.json → arms.<arm>.val.confusion_matrix`（行=目标，列=预测）；
  `per_class_recall` / `per_class_support` / `target_distribution` / `predicted_distribution` 同处

## 5. 吞吐与 GPU

| 项 | B0 | B1 |
|---|---|---|
| train loop 时长 | 11.5625 s | 11.8142 s |
| wall（含 loop 内 val） | 13.0558 s | 13.3583 s |
| eval 时长（1 次 val） | 1.4928 s | 1.5435 s |
| samples/s（train loop） | **4,012.72** | **3,927.24** |
| steps/s | 31.39 | 30.73 |
| median step | 16.1 ms | 16.8 ms |
| 第一步（含 CUDA 预热） | 391.6 ms | 431.0 ms |
| 数据 gather 占比 | 43.4%（5.0143 s） | 42.4%（5.0038 s） |
| eval samples/s | 6,614.56 | 6,397.00 |
| GPU 峰值 allocated | **252.11 MiB** | **252.11 MiB** |
| GPU 峰值 reserved | **436.0 MiB** | **436.0 MiB** |
| 显存总量 | 12,281.5 MiB | 同 |
| batch 回退 | 未触发（batch 128） | 未触发 |

- 数据面（memmap gather + `pack_batch` 归一化）占总时长 ~43%，是当前**唯一显著非计算开销**；
  正式 20 epoch × 4 臂 × 3 seed（≈12 次 run × 20 × 11.6 s ≈ 46 分钟纯训练）在此之前可先决定是否重叠 IO
- 显存占用 <3%，batch 128 远未到上限；**正式预算无需回退到 64**

## 6. 梯度与优化器

| 项 | B0 | B1 |
|---|---|---|
| 步数 / 非零 / NaN | 363 / **363** / 0 | 363 / **363** / 0 |
| mean / min / max（clip 前） | 2.0939 / 0.8908 / 5.2658 | 2.4478 / 0.9948 / 5.7717 |
| first / last（clip 前） | 1.8148 / 3.5005 | 1.6908 / 3.3448 |
| 触发 clip 的步数 | 1 | 1 |
| post-clip 最大值 | 5.0 | 5.0 |

- 梯度**全程非零、无 NaN**，优化器链路正常；只有 1 步超过 clip=5.0，说明该 lr 下裁剪基本不参与，但保留它以固定预算
- 训练末段梯度**上升**（B0 3.50，B1 3.34）：模型仍在从“多数类先验”向逐类区分移动，不是发散——epoch 均值 CE 已从
  随机初始化的 ln18≈2.890 降到 1.6898（B0）；本次只有 1 个 epoch，**不构成“loss 曲线单调”的证据**
  （`loss_first_epoch == loss_last_epoch` 是 1 epoch 的必然结果）

## 7. checkpoint

| arm | 路径 | 大小 | sha256（前 16） |
|---|---|---|---|
| B0 | `.tmp_v410_wp04_bc/b0_seed0_last.pt` | 2,944,307 B | `05ee3ab7c0069440…` |
| B1 | `.tmp_v410_wp04_bc/b1_seed0_last.pt` | 2,945,401 B | `f69ccb737960ef23…` |

- 完整 sha256 在 `bc_smoke.json → arms.<arm>.checkpoint.sha256`；`in_repo=false`、`gitignored=true`
- 校验：`torch.load` 后 `schema=v410-wp04-bc-checkpoint/1`、`state_dict` 键集合与 `build_model(arm)` 完全一致、
  `split.seeds` = 1–14 / 15–17 / 18–20、`budget.steps=363`、`metrics.val.loss` 与该 run 的 val loss 一致（测试已固定）
- B1 的 checkpoint 多 129 参数（bomb head），文件大 1,094 B

## 8. 标签规则落实（可复核）

1. **主目标**：`bc_loss()` 只做 `F.cross_entropy(logits, one_hot(chosen))`；`label_rule.query_status_used=false`、
   `survival_status_used=false`、`unknown_used_as_negative=false` 写在每个 run JSON 里
2. **unknown 不进损失**：`test_unknown_candidates_never_become_negatives` 用三份只改候选状态的标签文件（全 unknown /
   额外 verified_dead / 额外 verified）验证 targets、one-hot、bomb **完全相同**；`dead_candidates` 只作为诊断计数
3. **无 chosen 的行剔除而不是映射**：`rows_without_target` 计数（本次 0）+ `one_hot_targets()` 拒绝 −1（fail-closed）
4. **不做候选级负例**：没有 mask、class weight、margin、ranking 项；`known`/`unknown` 之间不加任何对比损失
5. **不做软目标**：`soft_target=false` 在 config 与代码双重固定；`target_distribution`（τ 网格）等 WP04.0 诊断**未接入**训练

## 9. 限制（不声称）

1. **不是 WP04 通过**：每个 arm 只有 **1 个训练种子 × 1 epoch（363/7,260 步，5% 预算）**，不是预注册的 20 epoch × 3 seeds × 四臂
2. **不是 G410-2**：没有任何闭环、存活率、控制能力或引擎侧结果；游戏未启动
3. **test split 未读**：seeds 18–20 只在计划里登记，**没有 test 指标**，也没有 test 调参
4. **B0 vs B1 不可下结论**：种子数 1、epoch 1、且 `bomb` 标签恒为 0（辅助项退化为负先验）；B1 三项 val 指标略差只能记录
5. **macro recall 低（0.10–0.16）不代表模型上限**：363 步内预测分布向动作 0 坍塌，属预算不足
6. **val loss < train loss** 是“epoch 均值 vs epoch 末评估”的时刻差，不是数据泄漏（val seeds 15–17 从未进梯度）
7. **吞吐是单机单卡实测**：数据 gather 占 43%，换存储/进程模型会变；`cudnn.deterministic=True` 已开
8. **checkpoint 只是 last-epoch 权重**，不是收敛模型，也不是可部署产物

## 10. 下一建议（按优先级）

1. **正式训练前把预算改成“步数”而不是“epoch”**：本次 363 step/epoch 下 macro recall 仍 ~0.16，
   建议正式四臂用 **≥7,260 step（20 epoch）** 并**同时报 macro recall**；只看 top1 会被多数类掩盖
2. **正式四臂（B0–B3，20 epoch × 3 seeds）**：同一数据、同一切分（1–14/15–17/18–20）、同一曝光顺序、
   同 batch/lr/clip；每臂 3 个训练种子给均值±极差；B2/B3（2.6 M）在此之前先确认显存与吞吐（当前 <3% 显存，风险低）
3. **先修 `bomb` 标签**：S1 全 0，B1 的辅助项现在只学“负先验”；要么在数据侧产出具信息量的 bomb 标签，
   要么在四臂设计里把 B1 改成一个有信息量的辅助任务（例如 verified/alive 辅助头，但需另立预注册）
4. **处理“多数类坍塌”**：正式训练建议增加 per-class recall / balanced accuracy 对照，或按类采样；
   否则 20 epoch 后 top1 可能主要由动作 0 支撑
5. **act_hist 消融**（承接 WP04.0 建议）：zero/shuffle 两个对照，作为 obs-v3 第三平面被真实使用的验收项
6. **吞吐**：需要时把 memmap 预取放到 DataLoader worker 或提前转 float32 缓存（当前 43% 时间在 gather），
   本次未做，因为单次 run 只有 ~18 s
7. **IO/验收口径**：跨机复现需固定 `cudnn.deterministic=True`；不同 eval batch 之间按 1e-3 容差比较

## 11. 复现清单

```
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp041_pytest -q tests/test_v410_student_bc.py   # exit 0, 53 passed
python -m compileall -q touhou_rl/v410/student                                                          # exit 0
python -m touhou_rl.v410.student.bc --arm B0 --seed 0 --epochs 1 --out .tmp_v410_wp04_bc/b0_s0.json     # exit 0
python -m touhou_rl.v410.student.bc --arm B1 --seed 0 --epochs 1 --out .tmp_v410_wp04_bc/b1_s0.json     # exit 0
python -m touhou_rl.v410.student.bc --assemble .tmp_v410_wp04_bc/b0_s0.json .tmp_v410_wp04_bc/b1_s0.json --out Document/Update/Alpha/v4.1.0/evidence/WP04/bc_smoke.json --summary   # exit 0
```

- 代码：`touhou_rl/v410/student/bc.py`（新增，只读 import `diagnostics.py` / `s1_data.py`）、配置 `configs/v410/bc_s1_arm.json`、
  测试 `tests/test_v410_student_bc.py`；证据 `Document/Update/Alpha/v4.1.0/evidence/WP04/bc_smoke.json`
- 配置在 `bc_smoke.json` 里出现两次：`config` 与 `preregistration`（同一份 `configs/v410/bc_s1_arm.json` 的有效内容）
- 运行产物（checkpoint / run JSON / 日志）在 `.tmp_v410_wp04_bc/`，**不入库**；`git status` 确认本次**没有修改任何既有跟踪文件**
- 本文件所有数字均来自上述命令的实际输出、`bc_smoke.json` 以及 `.tmp_v410_wp04_bc/{b0_s0,b1_s0}.json`（CLI 自己的 run JSON），未手写
