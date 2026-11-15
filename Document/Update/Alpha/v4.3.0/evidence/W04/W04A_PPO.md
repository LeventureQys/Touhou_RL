# W04A — PPO 纯函数损失与诊断（合成切片）

**范围声明**：本文件只记录 `touhou_rl/v430/ppo.py` 的纯函数实现与合成单测。
不涉及 rollout、引擎、真实数据、训练或能力评测。**合成测试不是游戏/能力证据**
（见 `tests/test_v430_ppo.py` 文件头）。

- 工作包：W04A（W04 的 PPO 纯函数切片）
- 规格来源：`DEVELOPMENT_GUIDE.md` §7.2 / §7.4、`PLAN_SPEC.json` → `ppo`
  （`clip_ratio=0.2`、`value_coefficient=0.5`、`entropy_coefficient=0.001`、
  `max_grad_norm=0.5`、`mean_kl_epoch_stop=0.02`；系数**不硬编码**，全部由调用方传入）
- 依赖：`torch` + stdlib（`math` / `dataclasses` / `typing`）。**不 import**
  引擎、`touhou_rl.v400`、`touhou_rl.v420`、`touhou_rl.v430.learn`、
  `touhou_rl.v430.models`、`sequence`、`contracts`。无 rollout、无文件 I/O、
  无全局状态、无 seed。

## 1. 变更文件（严格白名单内）

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `touhou_rl/v430/ppo.py` | 新建 | 8 个纯函数 + `PpoError` + 2 个结果 dataclass |
| `tests/test_v430_ppo.py` | 新建 | 38 项合成测试（含参数化） |
| `Document/Update/Alpha/v4.3.0/evidence/W04/W04A_PPO.md` | 新建 | 本文件 |
| `Document/Update/Alpha/v4.3.0/evidence/W04/EXIT_CODES_W04A.txt` | 新建 | 验收命令退出码 |

未修改任何既有文件（`contracts/actions/session/evaluate/observation/sequence/learn.py`、
`models/**`、`data/**`、`v400|v420/**`、`tools/rl/**`、`configs/**`、其它
`tests/test_v430_*.py`、`evidence/{W00,W01,W02,W03,W09}/**`、`registries/**`、
`PLAN_SPEC.json`、`OBS_CONTRACT_GAP.md`、`GATE_TABLE.md` 全部未触碰）。

## 2. 验收命令与退出码（真实执行）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m compileall -q touhou_rl/v430/ppo.py
# exit 0

python -m pytest -p no:cacheprovider --basetemp .tmp_v430_pytest_w04a -q tests/test_v430_ppo.py
# 38 passed in 1.57s / exit 0

python -m pytest -p no:cacheprovider --basetemp .tmp_v430_pytest_w04a_reg -q tests/test_v430_learn.py tests/test_v430_sequence.py
# 88 passed in 8.83s / exit 0
```

（与 `EXIT_CODES_W04A.txt` 同一轮捕获；数值改动后已重跑并重新捕获。）

`compileall` 退出码 0 只代表语法通过；真实结论来自两条 pytest。

## 3. API 与语义

| 函数 | 返回 | 关键拒绝条件（均 `PpoError`） |
| --- | --- | --- |
| `ratio(logp_new, logp_old)` | `exp(new-old)` 张量 | 非张量/非浮点、形状不一致、空批次、输入 NaN/Inf、比值溢出到 Inf |
| `clipped_surrogate(adv, new, old, *, clip_ratio)` | `ClippedSurrogate(loss, clip_fraction, ratio_mean)` | `clip_ratio<=0`/非有限/非实数/`bool`、形状或长度不一致、空批次、NaN/Inf |
| `value_loss(values, returns, *, clip=None, old_values=None)` | 标量损失 | `clip` 无 `old_values`（禁止静默降级 MSE）、`old_values` 无 `clip`（拒绝忽略引用）、`clip<=0`、形状不一致、空批次、NaN/Inf |
| `entropy_bonus(entropies, *, coef)` | `-coef*mean(H)` | `coef<0`、`coef` 非有限、空批次、NaN/Inf |
| `kl_estimators(new, old)` | **两个** float：`(k1, k3)` | 形状不一致、空批次、NaN/Inf；结果非有限 |
| `total_loss(p, v, e, *, value_coef, entropy_coef)` | 加权和标量 | 系数非有限、`value_coef<0`、`entropy_coef<0`、非标量/非有限损失项 |
| `gradient_diagnostics(params, *, reference_grads=None)` | `GradientDiagnostics` | 参数为裸张量、非浮点、引用长度未按位置对齐、引用形状不一致、无可用对照、零范数导致余弦无定义 |

设计要点：

1. **fail-closed 是对称的**：值损失不仅拒绝「有 `clip` 无 `old_values`」，也拒绝
   「有 `old_values` 无 `clip`」，因为后者意味着引用值会被静默忽略——与降级
   MSE 同类的隐藏行为。
2. **符号约定集中一处**：`entropy_bonus` 自己加负号，`total_loss` 只做加权，因此
   不存在「传原始熵进来忘记取负」之外的隐式翻转点；负系数一律报错而非自动取正。
3. **KL 只做诊断**：`kl_estimators` 在 float64 上 `detach()` 后计算，丢弃 `logp_new`
   的自图，避免「量 KL 顺手加了一条梯度路径」；docstring 明确 k1 可负、k3 恒非负、
   **不要混用**（阈值必须绑定到某一个估计量）。
4. **`clip_fraction` 的定义**：被裁样本 = 截断分支严格小于未截断分支
   （`clip(r)·A < r·A`），即 `min` 真正选中了截断项；用 float64 求均值，故 2/4 精确为
   `0.5`，3/4 精确为 `0.75`。
5. **诊断不污染梯度**：`gradient_diagnostics` 不修改 `.grad`、不 `zero_grad`；
   `None` 参数与 `grad is None` 都整项跳过（缺失 ≠ 零梯度，不得计入 `n_zero`）
   ；含 NaN/Inf 的梯度整项计入 `n_nonfinite` 并从 L2 中剔除；`n_finite` 为实际
   进入范数的张量数。
6. **无隐式同步**：仅在返回诊断时对已 detach 的 CPU 标量做 `float(...)`
   （等价 `.item()`），损失路径内不读取标量值。

## 4. 14 条要求的测试对号

| # | 要求 | 测试 |
| --- | --- | --- |
| 1 | `ratio` 身份为 1 | `test_ratio_identity_is_one`、`test_ratio_equals_exp_of_difference` |
| 2 | 裁剪边界 + `clip_fraction` 精确比例 | `test_clip_boundary_and_clip_fraction_exact`、`test_clip_fraction_zero_when_all_inside_band` |
| 3 | 手算对照 `-mean(min(rA, clip(r)A))` | `test_clipped_surrogate_hand_computed` |
| 4 | `clip_ratio<=0` / `inf` 抛错 | `test_clip_ratio_invalid_rejected`（参数化 5 例）、`test_clip_ratio_bool_and_text_rejected` |
| 5 | 无 `clip` 时等于 MSE | `test_value_loss_without_clip_is_mse`；另 `test_value_loss_clipped_matches_hand_computation` |
| 6 | `clip` 无 `old_values` 抛错（不降级） | `test_value_loss_clip_without_old_values_raises`、`test_value_loss_clip_invalid_rejected` |
| 7 | 熵项符号为负且等于 `-coef*mean(H)` | `test_entropy_bonus_is_negative_weighted_mean`、`test_entropy_bonus_zero_coefficient_is_zero` |
| 8 | `coef<0` 抛错 | `test_entropy_coefficient_negative_rejected` |
| 9 | KL 返回两个，恒等时均为 0，`k3>=0` | `test_kl_estimators_returns_two_and_zero_at_identity` |
| 10 | `k1` 可负、`k3` 不可负 | `test_kl_k1_can_be_negative_k3_cannot` |
| 11 | `total_loss` 加权手算 | `test_total_loss_weighted_sum_hand_computed`、`test_total_loss_keeps_gradient_path`、`test_total_loss_invalid_coefficients_rejected` |
| 12 | 梯度诊断（范数/跳过/计数/余弦冲突） | `test_gradient_diagnostics_norm_skip_and_cosine_conflict`、`test_gradient_diagnostics_accepts_generator_parameters`、`test_gradient_diagnostics_rejects_bad_reference_alignment` |
| 13 | 长度/形状/空/NaN 抛 `PpoError` | `test_shape_mismatch_empty_and_nan_rejected`、`test_ratio_overflow_is_rejected_not_clamped` |
| 14 | 文件头注明合成测试 | `test_file_header_declares_synthetic_scope`、`test_ppo_module_imports_no_engine_or_learn` |

额外：`test_functions_are_reproducible_and_do_not_move_the_input`（同输入同输出、
不改写入参，对应「可复现、无随机、无全局状态」）。

## 5. 实测数值（并列预期）

### 5.1 裁剪边界与 `clip_fraction`（`clip_ratio=0.2`，区间 `[0.8, 1.2]`）

构造：`A=[+1, -1, +1, -1]`，`r=[1.50, 0.50, 1.10, 0.95]`。

| 项 | 实测 | 预期 |
| --- | --- | --- |
| `clip_fraction` | `0.5` | `0.5`（4 个里 2 个被裁：`r=1.5>1.2` 且 `A>0`；`r=0.5<0.8` 且 `A<0`） |
| `ratio_mean` | `1.0125` | `1.0125` = `(1.5+0.5+1.1+0.95)/4` |
| `loss` | `-0.1375` | `-((1.2)+(-0.8)+(1.1)+(-0.95))/4` = `-0.55/4` = `-0.1375` |

三样本手算对照（`A=[1,-1,2]`，`r=[1.5,0.5,1.1]`，`clip=0.2`）：
`min` 项 = `1.2, -0.8, 2.2`，均值 `0.8666…`。

| 项 | 实测 | 预期 |
| --- | --- | --- |
| `loss` | `-0.8666666666666667` | `-2.6/3` = `-0.8666666666666667` |
| `clip_fraction` | `0.6666666666666666` | `2/3` |

### 5.2 `kl_estimators` 两个估计量

| 案例 | `k1` 实测 | `k3` 实测 | 预期 |
| --- | --- | --- | --- |
| `logp_new==logp_old=[-0.5,-1.0,0.3]` | `0.0` | `0.0` | 两者恰为 0 |
| 故意偏差 `new=[-1.0,-2.0]` vs `old=[-1.1,-2.2]`（`d=[0.1,0.2]`） | `-0.15000000000000013` | `0.012500000000000022` | `k1=-0.15`（可为负）、`k3=mean(d²)/2=0.0125`（恒非负） |

`k1<0` 与「同一对张量交换参数顺序后 `k1=+0.15`、`k3` 不变」均已断言，说明两者
不可混用。

### 5.3 `gradient_diagnostics`

场景：`w.grad=[3,4]`、`None` 项、`grad is None` 项、零梯度项、`inf` 梯度项；
引用 `reference=[-3,-4]`（零梯度项配零引用，`inf` 项配 `[1.0]`）。

| 项 | 实测 | 预期 |
| --- | --- | --- |
| `grad_norm` | `5.0` | `sqrt(3²+4²)=5.0`（`inf` 与 `None` 均被跳过） |
| `n_nonfinite` / `n_zero` / `n_finite` | `1` / `1` / `2` | `inf` 计 1；零梯度计 1；`grad is None` 不计入任何一类 |
| `cosine`（反向引用） | `-1.0` | `-1.0` |
| `conflict` | `True` | `cos<0` |
| `n_reference_pairs` | `2` | 仅 `[3,4]` 与零梯度项有可用对照 |
| `cosine`（同向引用） | `1.0`，`conflict=False` | `+1.0` |

### 5.4 其余手算对照

| 检查 | 实测 | 预期 |
| --- | --- | --- |
| `value_loss` MSE（`v=[1,2,3]`，`R=[1.5,1.5,4]`） | `0.5` | `(0.25+0.25+1)/3` = `0.5` |
| 截断值损失（`v=[2,1.4]`，`R=[1,1]`，`v_old=[1,1]`，`c=0.5`） | `0.58` | `(1.0+0.16)/2` = `0.58` |
| `entropy_bonus(H=[0.5,1.5,2.5], coef=0.001)` | `-0.0015` | `-0.001*1.5` |
| `total_loss(1,2,-0.5; 0.5,0.001)` | `1.9995` | `1+1-0.0005` |

## 6. 未做（本切片明确排除）

- 优势标准化（在有效 actor 样本上、截断前统计留在 W04B 的 batch 组装侧）；
- GAE/return/bootstrap（只读复用 `sequence.gae`，本文件不 import）；
- `max_grad_norm` 梯度裁剪、优化器步进、KL 早停、minibatch 迭代；
- burn-in / hidden 重算 / 序列切段（§7.1，属 W04B）；
- 任何真实 rollout、引擎、数据或能力评测。

## 7. TODO 清单

| TODO | 内容 | 归属 |
| --- | --- | --- |
| `TODO(W04A): 1` | `max_grad_norm=0.5` 的全局梯度裁剪函数（本切片只要诊断，不裁剪） | W04B |
| `TODO(W04A): 2` | 优势标准化 + 有效 actor 样本 mask + 标准化前统计落 result JSON | W04B |
| `TODO(W04A): 3` | KL 早停（`mean_kl_epoch_stop=0.02`）与「用 k1 还是 k3」的预注册绑定 | W04B |
| `TODO(W04A): 4` | `gradient_diagnostics` 目前按位置对齐 `reference_grads`；若 W04B 的 BC 梯度只覆盖部分参数，需增加按参数名匹配的变体 | W04B |
| `TODO(W04A): 5` | GPU 上 `float(...)` 读取会产生同步；本切片 CPU-only，正式训练若上 GPU 应改为批量聚合后单次读取 | W04C |

剩余阻塞：无。本切片自包含，不依赖引擎/数据/checkpoint。
