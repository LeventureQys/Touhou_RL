# D410-S1 学生侧四项短诊断（WP04.0）

- 时间：2026-11-12T20:07Z（`diag.json → generated_at`）
- 输入：已通过的 **D410-S1**（`WP03/s1c_full_audit.json`，`passed=true`）——20 个 train shards、
  **66,143 decisions**、真实 obs-v3 三平面、calibrated labels；audit sha256 `a8fdadda…`（完整值在 `diag.json → inputs.audit_sha256`）
- 设备：**CUDA**，NVIDIA GeForce RTX 4070 SUPER（capability 8.9，cuda 13.2），torch **2.14.1+cu132**，Python 3.12.10
- 范围：**只做诊断**。未训练正式四臂模型、未采数据、未启动游戏、未读 dev 1000–1099 / confirm 60000–60199
- 结论：**四项诊断全部实际跑通**（exit 0）。小集合拟合在**规定的 20 epoch（40 step）内只到 80.47%**，用**完全相同超参**再加 18 步（≈9 epoch）即达 **96.88%** → 缺口是**优化预算**，不是代码/标签；输入敏感性显示探针**对空间位移极敏感、对 act_hist 近乎不敏感**；离线确定性**重复前向 bitwise 一致**，批大小差为 CUDA kernel 级 **3.99e-4**；标签 KL 的 **98.27%** 由 −5.0 unverified discount 造出（τ=1）。

## 0. 关键数字一览

| 项 | 结果 |
|---|---|
| 小集合拟合（20 epoch / 40 step，AdamW 3e-4，batch 128，无增强） | loss 2.8856 → 1.5273；top1 **0.8047**（门 0.95 **未过**） |
| 同上 + 18 步（同超参） | top1 **0.96875**，loss 0.4993（门过） |
| 梯度范数（58 步，无一步为 0） | min 0.4984 / mean 1.4874 / max 2.6485 / last 2.0693 |
| 探针参数量 | **731,802**（预算 0.4–0.8 M 内） |
| 输入敏感性（12 对 × 5 扰动 = 60 次） | top1 改变 **19/60**；logits L2 mean 2.003，max-abs-Δ 2.935 |
| 离线确定性 | repeat **0.0（bitwise）**；reorder **0.0**；batch(1 vs 8) **3.986e-4**（CPU 交叉 1.43e-6） |
| 标签覆盖 | 每行恰好 **1** 个 verified（候选占比 **5.56% = 1/18**），unknown **94.44%**；verified_dead **0** |
| 标签 KL（τ=1） | KL(raw‖uniform) **2.3169**，其中 discount 造出 **2.2732 = 98.27%**；undiscounted 仅 **0.0437** |
| 几何偏好 | **82.10%** 的行 undiscounted top-gap **= 0**（几何分数并列，偏好由 discount 决出） |

## 1. 命令与退出码（实际运行，PowerShell）

前置（每次都设）：`$env:PYTHONPATH='D:\WorkShop\Touhou_RL'`

| # | 命令 | 退出码 |
|---|---|---|
| 1 | `python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp04_pytest -q tests/test_v410_student_diagnostics.py` | **0** |
| 2 | `python -m compileall -q touhou_rl/v410/student` | **0** |
| 3 | `python -m touhou_rl.v410.student.diagnostics --out Document/Update/Alpha/v4.1.0/evidence/WP04/diag.json --workdir .tmp_v410_wp04 --summary` | **0** |

- pytest：**51 passed in 5.03s**（51 个用例，全部合成数据，CPU，不读真实 obs 全量；`--basetemp .tmp_v410_wp04_pytest` 绕开 `.pytest_cache` 的 ACL 问题）
- 诊断 3 的整跑耗时 **7.77 s**（`diag.json → elapsed_seconds`）：label 解析 2.92 s、选样 0.68 s、
  小集合拟合 1.01 s（+扩展 0.20 s）、敏感性 0.14 s、确定性 0.05 s、标签信号 2.37 s
- 复跑一致性：`cudnn.deterministic=True`、`cudnn.benchmark=False`，连续两次整跑的
  loss / top1 / grad / 敏感性 / 确定性数字**逐位相同**（对照日志在 `.tmp_v410_wp04/run_a.json`、`run_b.json`）
- 探针 checkpoint：`.tmp_v410_wp04/small_fit_model.pt`（2.9 MB，含 model_config/参数数/state_dict/train）
- 确定性第一处不一致：`.tmp_v410_wp04/determinism_first_mismatch.pt`

## 2. 输入与选样（可复核）

- 20 shards / 20 train seeds（1–20），66,143 行；obs 三平面形状 `(N,4,112,96)`/`(N,56,160)`/`(N,8,18)` uint8，
  路径由 WP03 audit 的 `obs.play96x112.file` 与 `calibrated_labels.path` 直接给出（seeds 1–8 在 `.tmp_v400_d1`，9–20 在 `.tmp_v410_s1c`）
- 标签完整性：0 行缺 chosen、0 行缺 verified、0 个 verified_dead、`query_status ∈ {unknown, verified}`、
  `survival_status ∈ {unknown, verified_alive}`；**325** 行（0.49%）的 verified 计划 calibrated 生存状态为 `unknown`（保守规则，非 dead）
- 选样规则（`diag.json → sample_selection.label_rule`）：**tier A**＝该行恰好 1 个 verified 候选、
  该候选 alive、chosen 等于折后 top1、折后 margin ≥ 1.0 且**折前（undiscounted）margin ≥ 0.1**；
  **tier B**（折前 margin = 0 的"纯 discount"行）只用于补足。按目标动作 round-robin，单动作上限 32
- 结果：**256 样本，tier A = 256，tier B = 0**（可选池 tier A = 780 行，tier B = 65,363 行）；
  覆盖 **16/18** 个动作（0、9 号动作在 S1 中没有"清楚偏好"行），单动作最多 17 条，
  **多数类基线 6.64%**；20 shard / 20 seed 全部覆盖
- 折后 margin：5.10 / 5.17 / 9.32（min/median/max）；折前 margin：0.10 / 0.17 / 4.32
- 输入重复冲突：`distinct_inputs = 256`，**0** 个"同输入不同标签"组 → 95% 门不可能被矛盾监督卡住
- unknown **从不作为负例**：训练目标是唯一 verified 动作 one-hot，其余 17 个 logit 不进损失

## 3. 诊断 1：小集合拟合

协议（`configs/v410/bc_s1_smoke.json`）：AdamW `lr=3e-4`、`weight_decay=1e-4`、`batch=128`、**20 epoch**、
**增强关闭**、交叉熵（等价于 one-hot 目标的 KL）、256 样本 → **每 epoch 2 步，共 40 步**；模型 731,802 参数
（play 4×112×96 → 3 层 conv（24/48/64）→ 1×1(64→16) → flatten 2688 → 128；hud → 64；act_hist → 64；trunk 128 → 18）。

| epoch | 1 | 5 | 10 | 15 | 19 | 20 |
|---|---|---|---|---|---|---|
| train loss | 2.8856 | 2.7265 | 2.4460 | 1.9416 | 1.6367 | **1.5273** |
| train top1 | 0.0664 | 0.2031 | 0.5430 | 0.7109 | 0.8008 | **0.8047** |

- 40 步时 eval loss **1.4373**、top1 **0.8047** → **20 epoch 内未达 95%**
- 扩展（**超参与数据完全不变**，只是继续跑）：**+18 步**（"epoch 等量" 9 epoch）→ top1 **0.96875**、loss **0.4993**，
  `steps_to_target=18`；总计 58 步
- 梯度：58/58 步非零，min 0.4984（第 1 步之后没有衰减到 0 的迹象），末步 2.0693
- 判定：**不是代码/标签/优化器问题**。(a) loss 20 个 epoch 单调下降、末段仍陡；(b) 每步梯度非零、无 NaN；
  (c) 0 个同输入冲突标签，说明目标自洽；(d) 同超参再 18 步即过门；(e) 两次整跑逐位一致（可复现）。
  真实原因是**步数预算**：256 样本 × batch 128 = **40 step**，对 73 万参数、16 类、256 个近重复状态而言太小
- 该 95% 只说明**记忆 256 个固定样本**的能力，**不是泛化**（12 个探针状态上探针 top1 只有 3 个等于教师动作，见 §4）

## 4. 诊断 2：输入敏感性（12 对，同一权重）

同一 decision 只改输入，不换权重、不换标签；下表是 12 对的统计（每对 5 种扰动）。

| 扰动 | logits L2 mean / max | max-abs-Δ mean / max | KL(clean‖pert) mean | top1 改变 | 改动像素比例 |
|---|---|---|---|---|---|
| roll 整体环形平移 (dy=3, dx=8) | 4.482 / 6.676 | 2.311 / 2.923 | 0.5869 | **9/12** | 89.9% |
| 遮挡自机 16×12 补丁 | 1.054 / 2.283 | 0.572 / 1.442 | 0.0532 | 1/12 | 1.68% |
| 仅平移亮层（弹幕）+4/+6 | 4.319 / 6.046 | 2.141 / 2.935 | 0.5729 | **9/12** | 74.5% |
| 打乱 act_hist 8 行顺序 | 0.086 / 0.128 | 0.044 / 0.077 | 0.00020 | **0/12** | 8.1% 行 |
| act_hist 全零 | 0.074 / 0.090 | 0.039 / 0.059 | 0.00012 | **0/12** | 5.6% 行 |
| 合计（60 次测量） | 2.003 | 2.935 | 0.2427 | **19/60** | — |

- 逐对数字在 `diag.json → sensitivity.pairs_detail`（12 条，含 clean_top1、teacher_action、每种扰动的 L2/max-abs-Δ/top1 是否翻转）
- 读法：探针的 logits 由**亮层的空间位置**主导（roll 与"只移弹幕"都造成 ≈4.3–4.5 L2、9/12 翻转）；
  自机补丁只改 1.7% 像素却造成 1.05 L2（自机标记是最亮物体，符合预期）；
  **act_hist 分支几乎冻结**（≤0.13 L2、0/12 翻转）——8×18 的历史在这一小模型 + 58 步里没有获得功能
- 12 对中探针 top1 仅 3 对等于教师动作（seed5/8/9），其余为记忆失败/未泛化 → 与 §3 的"记忆而非泛化"一致
- **不声称控制能力**：这里没有问题域闭环、没有存活率、没有策略优劣

## 5. 诊断 3：离线确定性

同一帧 + 同一历史，`model.eval()` + `no_grad`，12 行探针集：

| 比较 | GPU 最大绝对差 | 位级 | CPU 交叉检查（同权重） |
|---|---|---|---|
| 同一 batch 连续 forward 两次 | **0.0** | ✅ bitwise | 0.0 ✅ |
| 同一行 batch=1 vs batch=8 | **3.986e-4** | ❌ | **1.43e-6**（≤1e-5 ✅） |
| 同一 batch 反序 | **0.0** | ✅ bitwise | 1.43e-6 |

- 容差表（`determinism.tolerance_table`）：1e-6 ✗、1e-5 ✗、1e-4 ✗、**1e-3 ✅**、1e-2 ✅
- 结论：**重复前向 bitwise 一致**；**批大小不变性在 CUDA 上只到 ~1e-3**（float32 卷积按 batch 选不同 kernel/归约顺序），
  同一权重在 CPU 上 1.43e-6 → 差异来自**设备 kernel**，不是模型/代码逻辑
- 第一处不一致已保存：`determinism_first_mismatch.pt`（探针集第 0 行＝seed1 row 2866，动作 10，Δ=3.99e-4）
- 记录的环境：`cudnn.enabled=True`、`cudnn.deterministic=True`（本轮显式固定，保证整跑可复现）、
  `cudnn.benchmark=False`、`torch.are_deterministic_algorithms_enabled()=False`

## 6. 诊断 4：标签信号审计（66,143 行）

`raw_geo_score` 是教师 `geo_score` 的**原样**数组，其中 **unverified 候选已被减去 5.0**；把 5.0 加回
`unknown` 候选即得"折前"分数（`undiscounted`）。四种读法在 τ 网格上比较：

| τ=1（主） | top1=chosen | 熵 (nats) | KL→uniform | unverified 质量占比 | 梯度 L2（均匀学生处） |
|---|---|---|---|---|---|
| raw（折后，实际训练用） | 1.0000 | 0.5735 | **2.3169** | 9.39% | 0.8752 |
| undiscounted（折前） | 0.9996 | 2.8467 | 0.0437 | 93.60% | 0.0298 |
| chosen（one-hot） | 1.0000 | 0.0 | 2.8904 | 0% | 0.9718 |
| verified（verified 支撑上重归一） | 1.0000 | 0.0 | 2.8904 | 0% | 0.9718 |
| verified_uniform | 1.0000 | 0.0 | 2.8904 | 0% | 0.9718 |

- **KL 归属**：`KL(raw‖uniform) = 2.3169`，`KL(undiscounted‖uniform) = 0.0437`
  → discount 造出 **2.2732 nats = 98.27%**；`KL(raw‖one-hot(verified)) = 0.0988`；
  按候选组精确拆分：verified 部分 **+2.5297**、unverified 部分 **−0.2128**（unverified 组低于 1/18 的均匀份额，因此是**负贡献**）
- **几何偏好缺口**：**82.10%** 的行折前 top-gap **恰好为 0**（教师几何分并列），
  把 discount 加回后 top1 改变的比例只有 **0.0438%（29 行）**、`undiscounted top1 = verified` 的比例 99.96%
  → discount 的作用是**把并列变成严格偏好**，很少直接改 argmax
- **τ 敏感性**（同一批行）：unverified 质量占比 0.07%（τ=0.5）→ 9.39%（τ=1）→ 56.39%（τ=2）→ 85.72%（τ=5）；
  discount 的 KL 占比始终 **94.7%–98.3%**
- **覆盖**：verified 候选占比 **1/18 = 5.56%**，unknown **94.44%**；**每行恰好 1 个 verified（100% 行）**；
  `chosen ∈ verified` **100%**；verified_dead **0**；325 个 verified 计划的生存状态为 `unknown`
- **对 WP04 的直接含义**：S1 的"verified 集合"**恒为单元素**，所以
  "集合损失（在 verified 支撑上分配质量）"在 S1 上**退化**为对单动作的 one-hot 交叉熵；
  软目标的信息量几乎全来自 discount，教师几何分只贡献 0.0437 nats/行（≈1.9%）。
  若四臂里的 soft/set 臂需要非退化的教师分布，**必须先改标签规则（或给出按 τ 的显式软目标）**，而不是换损失函数

## 7. 限制（不声称）

`diag.json → not_claimed` 的 5 条原文（下列 1–5），加上本次运行补充的 2 条操作性说明（6–7）：

1. 无控制能力结论：不跑游戏、不测存活、不评估闭环表现
2. 不构成正式四臂 20-epoch 训练：探针是 0.4–0.8 M 参数的诊断网络，**不是** P3 目标架构（`model_v3` 2.5 M）
3. 未读 dev/confirm：只用 train seeds 1–20（audit 与代码双重 fail-closed）
4. 95% 门衡量的是**256 个固定样本的记忆**，不是泛化（探针在 12 个未训练状态上只有 3/12 与教师一致）
5. 敏感性数字只描述**一个已拟合探针在 12 对状态上**的行为，不是目标架构的性质
6. 小集合拟合的"18 步扩展"是诊断（区分预算与缺陷），**不是**正式训练预算结论
7. 确定性是 float32 + CUDA 的实测值；批大小差异 3.99e-4 属设备 kernel 级，未做 float64 参考实现

## 8. 下一建议（按优先级）

1. **先修标签规则再谈四臂**：S1 的 verified 支撑恒为 1，soft/set 臂在定义上退化。建议在 WP04 正式训练前
   明确 τ（当前隐含 τ≈1：discount 5.0 对应 5 nats）并决定是否用 `undiscounted` 分数生成本征上更平的软目标；
   否则四臂中的 KL/集合臂与 one-hot CE 臂不可区分
2. **小集合拟合只作为"可学习性"门槛**，正式训练前把 20 epoch 的**步数预算**写死（例如 batch 64 或 ≥160 step），
   或在配置里显式给出 `steps` 而非 `epochs`——本次 40 步在 16 类、256 样本上确定不足
3. **act_hist 需要单独验证**：本探针对其几乎不敏感（0/12 翻转、≤0.13 L2）。正式训练应加一条"history 消融"
   对照（zero/shuffle）来证明该输入确实被使用，否则 obs-v3 的第三平面可能是死输入
4. **确定性写进验收**：GPU 上批大小差异 ~4e-4，评估若按不同 batch 跑必须在 1e-3 容差内比较，
   或统一 eval batch=1；跨机器复现需固定 `cudnn.deterministic=True`
5. 正式训练若要在 S1 上跑满 66,143 决策，建议先用本次的 780 行 tier-A 子集做一次学习率/步数扫描，
   再进四臂，以免把预算问题误判成架构问题

## 9. 复现清单

```
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp04_pytest -q tests/test_v410_student_diagnostics.py   # exit 0, 51 passed
python -m compileall -q touhou_rl/v410/student                                                                  # exit 0
python -m touhou_rl.v410.student.diagnostics --out Document/Update/Alpha/v4.1.0/evidence/WP04/diag.json --workdir .tmp_v410_wp04 --summary   # exit 0
```

代码：`touhou_rl/v410/student/{__init__,s1_data,diagnostics}.py`、配置 `configs/v410/bc_s1_smoke.json`、
测试 `tests/test_v410_student_diagnostics.py`。运行产物（checkpoint / 两处不一致 / 复跑对照）在 `.tmp_v410_wp04/`，不入库。
本文件全部数字均由上述命令生成；`diag.json` 是对应机器可读证据。
