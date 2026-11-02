# P4 冻结骨干空间特征的条件化融合 → 按组留出闭环（round22）

## 一句话

R21 把 F1 的覆盖带与缺口画进输入之后仍然失败，并且把失败定位得很干净：**方向**——base 组 18 格中 15 格走对、镜像组 18 格中 15 格走反，所有死亡仍在 `wall_frame`，同一 head/世界/起点只因"画面上有没有预告"就从"停在缺口 248"变成"hold 在 216"。本轮只改一个**结构**：把冻结骨干的 F8（28×24）汇成 **94 维**空间向量（7×6 平均池化 42 + 列 profile 24 + 行 profile 28），经一层**权重与偏置严格为 0** 的 `Linear(94→256)` 加到原有 256 维 features 上，再送进**未改动的 N1 头**；A 臂走同一条代码路径但融合层旁路并冻结在 0（等价 R21 A）。3 seeds × 200 updates、R21 的 telegraph 像素、R19 冻结 split、R20/R21 超参一律不变，在 validation 14 事件上闭环。**结果：F1 留出完全没动——A 2/2/2、S 2/2/2（`unchanged`，逐 seed Δ=0），而且 14 个 validation 事件的 first hit 在三个 seed 上全部逐事件相同（14/14 ×3）。** 机制读数才是本轮的新增量：**方向翻转率一格没变**（base 6/9 走对、镜像 3/9 走对，两臂完全相同；A 的 3 格镜像走对全部来自 `007-m1`）；**融合层学到的东西 94.4% 是一个常数再偏置**（delta 范数 4.40 vs 冻结 features 2.99，其中随画面变化的部分只有 0.246，占 5.6%）；把 delta 关掉（不改任何参数）时 S 从 3/14 掉到 0/14，说明 delta 有力、但只作用于**停车点**、从不作用于**镜像方向**。而"空间信息不可用"这个读法被本轮自己的探针**否定**：同一个冻结骨干的 **256 维 trunk features 在留出几何上线性可解码缺口所在侧 98.5%**（λ 扫描 95–98%），94 维向量只有 67%（且逐折不稳 100/50/52%，λ 扫描 52–68%，中心误差 55 px ≈ 直接用均值 55 px）；94 维列 profile 与其镜像逆序的相关只有 0.39–0.44。所以 R21 的负结果既不是"看不见"、也不是"输入里没有空间信息"，而是**输出形态（预告窗口里 95–97% 的决策是 hold，动的时候走底座先验）+ 这个 94 维池化向量太粗（一列 = 16 native px，12–20 px 的缺口约一列宽）**。本轮不宣称战术理解，不产生可部署产物。

## 隔离与边界

- 只新增 4 个路径：[本轮工具](<../../../../../tools/rl/v373_spatial_fuse.py>)、[本轮测试](<../../../../../tests/test_v373_spatial_fuse.py>)、本报告、[evidence 目录](<../evidence/spatialfuse-round22/preregistration.json>)。
- **不改旧模块**：`tactical_policy.py`（父类 `forward_features` 一字未改，变体是**子类**）、`vmodels3.py`、`vmodels.py`、`vision.py`、`tactical_synthetic_pixels.py`、`tactical_teacher_validation.py`、`tactical_scenarios.py`、`tactical_synthetic_closed_loop.py`、`tactical_teacher*.py`、`macro_policy.py`、`tools/rl/v373_valid_curriculum.py`（`encode` 的同序特征/基线、`arm_loss`、`six_baselines`、`behaviour`、`compact`、`write_*` 全部原样复用）、`v373_telegraph.py`（渲染器、`run_event_rendered`、`closed_loop`、`gap_trace`、`gap_diagnostics` 原样复用）、`v373_seed_confirm.py`、`v373_world_filter.py`，共 **33 个身份 SHA** 在预注册里冻结、`verify` 逐个复核一致。
- **不改旧证据**：R18 `run/manifest.json`、R19 `run/split.json`、R21 的 `preregistration.json` / `run/records.json` / `run/summary.json` / `run/checkpoints.json` / `run/f1-gap.json` / `run/baselines.json`、两份旧报告一并冻结；**训练像素一次都没有重渲染**——14 个 train 事件的 `pixels.npz`/`history.pt`/`labels.pt`/`rollout.json` 全部按 **R21 冻结 SHA** 逐字节绑定后才允许使用（`records.bindings`，14 × 4 = 56 个 SHA）。
- **没有重新生成世界、没有调用教师、没有新标签**：`teacher_calls = 0`、`new_worlds = false`、`new_pixels = false`；train 闭环用的就是那 14 条 R21 已冻结的教师路线（本轮只**读**它们，不重规划）。
- **没有 test split**、**未开启正式门**、**不产生可部署 checkpoint**（`deployable = false`）、**不运行游戏**、**不调参**（`tuning = false`）。
- **控制臂不是"重建的 R21 A"，而是逐位相同的 R21 A**：三个 seed 的 checkpoint SHA 与 R21 的 `d52551bb…` / `fb9233f1…` / `49ff13d9…` 逐一相同，并且 6 个 (arm, split) 组合的**每一步**（`candidate`/`actual_action`/`x_after`/`hit`）都与 R21 存盘的闭环完全相同（`equivalence.all_equal = true`）。这是本轮所有 S-vs-A 比较的前提，也是"两臂除融合外完全一致"的唯一可检验形式。

## 干预：融合层是什么、不是什么

### 1) 唯一改动

| 元素 | 值 |
|---|---|
| 空间源 | `F8 [B,96,28,24]`（冻结骨干 `PlayfieldTrunk` 的第二个输出）；`F4 [B,64,56,48]` 从同一次调用里取出并**锁定形状**，但按预注册**不使用** |
| 空间向量（94 维，无参数） | `concat( adaptive_avg_pool2d(F8,(7,6)).mean(channel).flatten → 42, F8.mean(channel,row) → 24, F8.mean(channel,column) → 28 )` |
| 融合参数 | 一层 `Linear(94,256,bias=True)`，**权重与偏置严格为 0**（`nn.init.zeros_`）= **24 320** 个参数 |
| 融合 | `fused = features + projection(spatial)`；**A 臂直接 `return features`（旁路，不做任何算术）并把该层冻结在 0（`requires_grad=False`，活跃参数增量 = 0）** |
| 头 | `TacticalPolicy.forward_features`，只改一行（见下） |
| 不变量 | 世界、世界参数、R19 split、R21 渲染器与像素、教师动作、标签、打包、编码器、loss 权重、优化器、lr、clip、序列长、batch、update 数、seed 集合、闭环 runner、条件、掩码 |

**base_logits 仍来自原始 features 的 `backbone.policy_head`**，不是从融合特征重算：`SpatialHead` 只在 `inp`（GRU 输入）与 `residual` 的条件向量里看到融合后的 features，`base_logits` 与 R20/R21 完全同源，残差头的初值语义不变。

### 2) 两处被迫的结构修订（都写进了预注册，也都有单测）

**(a) 父类 `forward_features` 会把输入 detach。** 父类第一行是
`features, base_logits = features.detach(), base_logits.detach()`。这对"头建在冻结特征上"是对的，但对**上游可训练层**是致命的：`detach()` 切断了 loss 到融合参数的唯一路径，于是**融合层 200 步全程拿到 0 梯度，S 臂会与 A 臂逐位相同，本轮变成空转**。这不是推测，单测 `test_the_parent_detach_would_freeze_the_fusion_forever` 直接测了：给两个头塞入同样的非零 residual（否则零初始化 residual 本身也会挡住所有 features 侧梯度），过父类时 `d(loss)/d(fusion weight)` 是 **None**，过变体时是 **> 0**。

因此本轮的头是**子类** `SpatialHead(TacticalPolicy)`，其 `forward_features` 是父类函数体**逐行相同、只少一句 `features.detach()`**：
- `test_variant_body_is_the_parent_body_with_one_line_changed`：用 `inspect.getsource` 取出两个函数体、去 docstring/空行/注释后**逐行比对**，64 行 vs 64 行、**只有 1 行不同**，且那一行正是 `features, base_logits = features.detach(), …` → `base_logits = base_logits.detach()`；
- `test_variant_forward_is_the_parent_forward_bitwise`：同一份权重、同一批真实数据行 + 一批随机张量，`policy_logits`/`residual_logits`/`option_logits`/`target`/`slack`/`urgency`/`termination_logits`/`hidden` **逐位相等**。

也就是说：**数值上前向完全等价，唯一差别是 autograd 图上多了一条边**。父模块文件未被修改。

**(b) `forward_core` 不返回 F4/F8。** 所以按 `vmodels3` 里的同一调用顺序自行取：
`checked = backbone._checked_regions(regions)` → `backbone.playfield(backbone._playfield_input(checked['playfield']))`。
这不是"看起来一样"，而是**被测出来一样**：`test_the_spatial_maps_are_the_tensors_forward_core_used` 用取到的 F4/F8 把 `forward_core` 的**后半段**（`locate` → `decode_location` → `location_mlp` → `local_features` / `global_features` / 两个 HUD 分支 → `trunk` → `policy_head`）重算一遍，与 `forward_core` 的 `heatmap_logits`/`position`/`features`/`policy_logits` **逐位相等**。所以"这两张图就是冻结骨干当时用的那两张"是被测出来的事实。

### 3) 分辨率：这个向量装得下 12–20 px 的缺口吗

模型看到的 playfield 是 **192 列**（native 384 px 的一一对应 2:1），F8 的列方向是 **24** 列：**一列 = 8 model px = 16 native px**；7×6 池化后一格 = **64 native px**。而 F1 的缺口只有 **12–20 px**，也就是**约一列 F8 宽**。这一点在预注册时就算好并冻结（`fusion.resolution`）：**这个向量在构造上就是粗的**，它当然可以编码"左边有槽/右边有槽"，但很难编码 12–20 px 的停车点。本轮不做任何"换更细的图"的补救——那是第二个因子。

### 4) 初始等价（本轮的单测硬约束）

- `test_fusion_layer_starts_at_exactly_zero`：两种臂的融合层权重与偏置都严格为 0；两臂的 `fusion.state_dict()` SHA 相同（`64a8389b…`）。
- `test_step0_of_S_is_bitwise_arm_A`：真实数据行上 A（旁路）与 S（零初始化但启用）的 `policy_logits` **逐位相等**、`hidden` 逐位相等、masked 动作相同、delta 的 `abs().max() == 0`。
- `test_initial_head_sha_is_the_frozen_table`：每个 seed 的初始 head SHA 等于 R21 预注册里冻结的 `66e5f85f…` / `f7d03e20…` / `276573d1…`（即 R16/R20/R21 的那张表）。
- `test_fusion_gradient_is_zero_at_step0_and_live_from_step1`：**step 0 的融合梯度恰好是 0**（原因与融合无关：residual 头本身也是零初始化，step 0 没有任何 features→logits 的路径），从 step 1 起为正；训练日志里 `nonzero_steps = 199/200`。

## 预注册判据（先冻结、再跑）

判据写死在 [preregistration.json](<../evidence/spatialfuse-round22/preregistration.json>) 的 `criterion`（SHA `8ff1f14b6c071cbe63b2eae9d153596ab1ba667d6779bea5dc72f3d487681297`），冻结 **33 个身份 SHA**，其中：

| 类别 | 冻结内容 |
|---|---|
| 源码 | 15 个：本轮工具 + 本轮测试、R21 工具 + R21 测试、R20 工具 + R20 测试、`tactical_policy.py`/`macro_policy.py`/`vmodels3.py`/`vmodels.py`/`vision.py`、像素与教师与场景与闭环 5 个、`v373_seed_confirm.py`/`v373_world_filter.py`、`tactical_sequence_train.py`/`tactical_feature_cache.py`/`tactical_data.py` |
| 数据 | R18 `manifest.json` SHA `9bd3a1b8…`；R19 `split.json` SHA `fa35b381…`；**R21 的 14 个 train 资产逐个 SHA**（56 个绑定） |
| worlds digest | train `abb70913aabc9610…`（14 事件）/ validation `5c7d0b110b11369f…`（14 事件）/ 合计 `b47dd5f4b7001e48…`——与 R20/R21 完全一致 |
| 底座 | `third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt`（SHA `74abde8a…`） |
| R21 参照 | `preregistration.json`、`run/records.json`、`run/summary.json`、`run/checkpoints.json`、`run/f1-gap.json`、`run/baselines.json` 逐一 SHA |
| 融合形状/数值 | F4 `[56,48]`、F8 `[28,24]`、spatial `[80,94]`、投影 `94→256` = 24 320 参数；**94 维向量 SHA `83a75f73…`**（14 事件 ×80 行，逐个事件的 min/max/mean/std 也冻结）；14 个事件的 feature 范数 |
| 初值 | 每 seed 的 head SHA + 两臂的融合层 SHA（必须相同） |
| 几何 | 28 个事件的 start/gap/gap_screen/wall_frame/preview_window/decisions_in_preview |
| 判据要点 | 干预只有一个（融合层）；A 必须逐位复现 R21 A，否则报 `control_not_reproduced` 并**撤回** S-vs-A 结论；改进判定 = F1 留出逐 seed 的 S−A（任一更大=improved、全等=unchanged、任一更小=worse）；机制指标只报计数与 px，不做显著性声明；若镜像方向没有改善，**必须**在"底座先验"与"向量太粗"之间给出证据支持的判断，不能留空 |

`verify_prereg()` 在 `run` 之前与之后各重算一次：33 个身份 SHA、criterion SHA、worlds digest、**94 维空间向量（重跑一次完整的 14 事件编码）**、空间统计、28 条几何、R21 参照记录。任何漂移都直接拒绝执行。

## 数据：R21 的 14 个 train 事件 + 冻结的空间向量

用 R21 的 14 个 train 事件（F1×6 / F2×2 / F3×6，教师 14/14 `verify()` 可行，`failures = []`），资产总 **2 714 834 B（2.71 MB）**，与 R21 逐字节相同；本轮**额外**产出 14 个 `spatial/<event>.npz`（94 维 float32，共 **149 991 B**）。空间向量的统计（预注册时冻结、run 时复算一致）：

| 事件族 | 94 维 min | max | mean | std |
|---|---:|---:|---:|---:|
| F1 ×6 | 0.141 421 – 0.150 075 | 0.281 514 – 0.294 803 | 0.179 287 – 0.179 535 | 0.015 932 – 0.016 436 |
| F2 ×2 | 0.145 641 – 0.147 288 | 0.266 498 | 0.174 931 – 0.174 938 | 0.013 241 – 0.013 255 |
| F3 ×6 | 0.144 748 – 0.152 252 | 0.266 498 – 0.313 815 | 0.192 275 – 0.194 742 | 0.019 912 – 0.020 380 |

**注意这个统计本身就说明了问题的一半**：94 维向量的 **mean/std ≈ 11**，也就是说它的"常数部分"比"随画面变化的部分"大一个量级；而冻结 features 的每行 L2 范数是 **2.84–3.16**。下面的"融合层学到什么"会看到这两件事的直接后果。

## 训练：2 臂 × 3 seeds，一次跑完

| 检查点 | 初始 head（前 16） | 末 head（前 16） | 活跃参数 | loss@0 | loss@200 | 离线 masked（已见） | F1 离线 | 训练秒 | 同 seed 与 R21 A 逐位相同 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| A-s7 | `66e5f85fe6b8e186` | `d52551bb43e4d49d` | 429 200 | 2.9321 | 0.1666 | 0.9143 | 0.9625 | 11.22 | **是** |
| A-s17 | `f7d03e205f378179` | `fb9233f11bb2c556` | 429 200 | 2.9321 | 0.1672 | 0.9161 | 0.9667 | 10.28 | **是** |
| A-s27 | `276573d10853f3e7` | `49ff13d9384fb2d6` | 429 200 | 2.9321 | 0.1643 | 0.9170 | 0.9688 | 10.64 | **是** |
| S-s7 | `66e5f85fe6b8e186` | `0ef292e5164062cb` | **453 520** | 2.9321 | 0.1665 | 0.9161 | 0.9667 | 11.59 | —（融合臂） |
| S-s17 | `f7d03e205f378179` | `77dda383da6094bb` | **453 520** | 2.9321 | 0.1666 | 0.9170 | 0.9688 | 10.80 | — |
| S-s27 | `276573d10853f3e7` | `49510b080d01afb7` | **453 520** | 2.9321 | 0.1666 | 0.9170 | 0.9688 | 11.17 | — |

完整 64 位 SHA 见 [run/checkpoints.json](<../evidence/spatialfuse-round22/run/checkpoints.json>)。

- **同 seed 两臂初始 head SHA 逐位相同**（7/17/27 全部），且等于 R16/R20/R21 冻结值；`loss@0` 六个都是 **2.9321**（同一批行、同一初值）。
- **融合层活跃参数增量 = 24 320**（94×256+256），A 臂 = **0**；两臂的 `total` 都是 **2 026 474**（冻结骨干 1 507 162 + head 494 992 + 融合 24 320），差别只是 A 的融合 24 320 个参数不可训练（`heads_total = 494 992`、`fusion_total = 24 320`）。
- **融合梯度**：step 0 恰好 0.0（结构原因：residual 头也是零初始化，见上），此后 199/200 步非零，末值 0.032–0.037、最大 0.112–0.114。
- **delta 大小**（训练日志的 `delta_norm` 是**该事件 80 行 delta 的 Frobenius 范数**，不是单行范数：÷√80 ≈ 8.94 才是每行范数）：末值 38.56–39.17 → **每行约 4.31–4.38**；训练结束后在全部 1120 行上重新测的每行均值是 **4.38–4.45**（见下），两者一致，且都是冻结 features 范数（2.99）的 **1.46–1.49 倍**。
- 离线 masked 准确率（**已见** 14 事件）六个都从 0.0491 升到 0.914–0.917，F1 分族 0.9625–0.9688；**两臂的离线拟合几乎一模一样**——融合没有改变"拟合标签"这件事。

## 评估

### 1) 留出侧校验先行

R18 自己的 `policies()/causal_loop()/outcome()` 在 validation 14 + train 14 世界上重跑六条冻结基线，**168 行与 R21 的 `run/baselines.json` 逐行相同**（工具内硬断言，含 `event/policy/feasible/survived/first_hit`）：

| 族 | hold | right | left | alternate | random | teacher_first_action |
|---|---|---|---|---|---|---|
| F1（6） | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| F2（2） | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| F3（6） | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| 合计（14） | **0/14** | **0/14** | **0/14** | **0/14** | **0/14** | **0/14** |

**`holdout_filter_valid = true`，`passing_events = []`。** 同一批 telegraph 渲染上，未训练 **B0 在 validation 0/14、train 0/14**；B0 的 F1 死亡帧 48/48/60/60/56/56，与 R21 完全相同。

### 2) 主指标 A：F1 留出的方向翻转率

分类口径（预注册写死）：用 `advance` 语义把已执行动作重建为**逐逻辑帧**的 x（含 2 帧决策网格从不采样的奇数帧），取 `start → wall_frame-1` 的 **canonical** 位移方向；`toward` = 该方向指向缺口的 canonical 中心，`hold` = 零净位移，`away` = 相反；`screen_direction` 是同一段位移在**原始 x** 上的方向（底座先验作用的那一维）。三个 F1 留出组的起点都在缺口 canonical 左侧，所以 "canonical 右" 在 base 世界等于"屏幕右"、在镜像世界等于"屏幕左"。

| 臂 | 组 | 格数 | 走对（toward） | 原地 hold | 走反（away） | 停进缺口 | 停车误差 \|px\| 均值 / 最大 | 全程没进过缺口 | 屏幕向右 / 向左 / hold |
|---|---|---:|---:|---:|---:|---:|---|---:|---|
| **A（无融合）** | base | 9 | **6** | 3 | 0 | 3 | 13.33 / 32 | 6 | 6 / 0 / 3 |
| | 镜像 | 9 | **3** | 0 | 6 | 3 | 45.33 / 72 | 6 | 6 / 3 / 0 |
| **S（空间融合）** | base | 9 | **6** | 0 | 3 | 3 | 16.00 / 40 | 6 | 6 / 3 / 0 |
| | 镜像 | 9 | **3** | 1 | 5 | 3 | 36.44 / 72 | 6 | 5 / 3 / 1 |
| **B0（冻结底座）** | base | 3 | 3 | 0 | 0 | 0 | 96.00 / 144 | 0 | 3 / 0 / 0 |
| | 镜像 | 3 | **0** | 0 | 3 | 0 | 166.67 / 192 | 3 | 3 / 0 / 0 |

**镜像组走对率：A = 3/9 = 33.3%，S = 3/9 = 33.3%——一格没变。** 三格都是 `007-m1`（S 的 x=160/160/168，仍停在 `[152,172]` 里）。base 组走对率也一样是 6/9。逐 seed 完全一致：

| seed | A base（toward/hold/away） | A 镜像 | S base | S 镜像 |
|---|---|---|---|---|
| 7 | 2 / 1 / 0 | 1 / 0 / 2 | 2 / 0 / 1 | 1 / 1 / 1 |
| 17 | 2 / 1 / 0 | 1 / 0 / 2 | 2 / 0 / 1 | 1 / 0 / 2 |
| 27 | 2 / 1 / 0 | 1 / 0 / 2 | 2 / 0 / 1 | 1 / 0 / 2 |

**S 唯一改掉的运动是 005 组**：base 的 `005-m0` 从 A 的"原地 hold 在 216（差 32 px）"变成"**屏幕向左**走到 208（差 40 px）"——也就是在 base 组里**多走了 3 格错误方向**（A 的 3 格 hold 变成 S 的 3 格 away）；镜像的 `005-m1` 从"屏幕向右到 184"变成 s7 的"hold 在 216"、s17/s27 的"屏幕向右到 208"。方向结构因此是：**A 的 base 9 格里 6 走对 + 3 hold + 0 走反；S 的 base 9 格里 6 走对 + 0 hold + 3 走反；两臂的镜像组都只有 3 格走对（A 6 格走反，S 5 格走反 + 1 格 hold）。**

### 3) 主指标 B：停车位置与误差 px

`wall_frame-1`（最后一个安全帧，三个世界都是奇数帧）的 canonical x 与带符号误差（`0` = 在缺口内；负 = 差多少没到左边界；正 = 越过右边界多少 px），括号内是距缺口 0 的像素差。三 seed 相同处标 ×3：

| 事件 | 起点 | 缺口 | wall | A（×3 seeds） | S（×3 seeds） |
|---|---:|---|---:|---|---|
| `005-m0` | 216 | [248,264] | 48 | **hold 216（−32）** | **屏幕向左 208（−40）** |
| `005-m1` | 216 | [248,264] | 48 | 屏幕向右 184（−64） | s7 hold 216（−32）；s17/s27 屏幕向右 208（−40） |
| `007-m0` | 128 | [152,172] | 60 | **停进 160（0）** | **停进 160（0）** |
| `007-m1` | 128 | [152,172] | 60 | **停进 160（0）** | **停进 160 / 160 / 168（0）** |
| `010-m0` | 216 | [256,268] | 56 | 屏幕向右 248（−8） | **完全相同 248（−8）** |
| `010-m1` | 216 | [256,268] | 56 | 屏幕向右 184（−72） | **完全相同 184（−72）** |

也就是说：**6 个 F1 留出事件里 3 个的停车位置与带符号误差逐 px 相同**（`007-m0`、`010-m0`、`010-m1`），`007-m1` 三个 seed 都停进缺口但 s17/s27 停在 168 而不是 160（误差都是 0），**只有 `005` 两组的运动方向变了，而且 base 组那次是往错误方向动的**。B0 对照：base 走到 300/316/376（越过 +36/+144/+108），镜像走到 84/8/64（差 −164/−144/−192）——典型的"恒屏幕向右"。

### 4) 逐族 feasible / survived 与逐事件 first hit

`feasible` 与 `survived` 在每个格子里都相等，下表只列 `feasible`：

| 族 | n | B0 | A-s7 | A-s17 | A-s27 | S-s7 | S-s17 | S-s27 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 | 6 | 0 | **2** | **2** | **2** | **2** | **2** | **2** |
| F2 | 2 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| F3 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **合计** | 14 | **0** | **3** | **3** | **3** | **3** | **3** | **3** |

**逐事件 first hit（逻辑帧，`none` = 整段无碰撞）**：两臂**逐格完全相同**。

| 事件 | B0 | A-s7 | A-s17 | A-s27 | S-s7 | S-s17 | S-s27 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wf18-F1-005-m0` | 48 | 48 | 48 | 48 | 48 | 48 | 48 |
| `wf18-F1-005-m1` | 48 | 48 | 48 | 48 | 48 | 48 | 48 |
| `wf18-F1-007-m0` | 60 | **none** | **none** | **none** | **none** | **none** | **none** |
| `wf18-F1-007-m1` | 60 | **none** | **none** | **none** | **none** | **none** | **none** |
| `wf18-F1-010-m0` | 56 | 56 | 56 | 56 | 56 | 56 | 56 |
| `wf18-F1-010-m1` | 56 | 56 | 56 | 56 | 56 | 56 | 56 |
| `wf18-F2-399-m0` | 92 | 52 | 52 | 52 | 52 | 52 | 52 |
| `wf18-F2-399-m1` | 108 | **none** | **none** | **none** | **none** | **none** | **none** |
| `wf18-F3-001-m0/m1` | 80 / 14 | 52 / 52 | 52 / 52 | 52 / 52 | 52 / 52 | 52 / 52 | 52 / 52 |
| `wf18-F3-003-m0/m1` | 66 / 34 | 62 / 62 | 62 / 62 | 62 / 62 | 62 / 62 | 62 / 62 | 62 / 62 |
| `wf18-F3-005-m0/m1` | 84 / 20 | 58 / 58 | 58 / 58 | 58 / 58 | 58 / 58 | 58 / 58 | 58 / 58 |

**14/14 格 first hit 在三个 seed 上全部两臂相同**（`comparison.identical_first_hit_cells = [14, 14, 14]`，`moved_first_hit = {}`）。这是比"F1 计数相同"更硬的"没变"：**没有一个事件在任何一个 seed 上换了碰撞帧**。

**所有 F1 死亡仍在 `wall_frame`**：两臂 6 个 head 的 validation **24** 次 F1 死亡 + train **26** 次 + B0 的 **12** 次 = **62 次**，全部等于各自世界的 `wall_frame`（48 / 60 / 56 / 76 / 40 / 52），与 R21 的 51+12 次同一种死法（`f1-gap.json` 里逐行可查：没有任何一行的 `hit_frame` 不等于 `wall_frame`）。

**动作分布与形态**（validation 14×80 = 1120 个已执行动作）：

| 网络 | hold | right | left | 切换中位数 | 最大切换 | `切换≤1` 事件 | 恒定事件 | `切换≤1` 且可行 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A-s7 | 0.9429 | 0.0375 | 0.0196 | 1.0 | 2 | 13/14 | 0 | 3 |
| A-s17 | 0.9429 | 0.0375 | 0.0196 | 1.0 | 2 | 13/14 | 0 | 3 |
| A-s27 | 0.9411 | 0.0384 | 0.0205 | 1.0 | 2 | 13/14 | 0 | 3 |
| S-s7 | 0.9402 | 0.0366 | 0.0232 | 1.0 | 2 | 12/14 | 0 | 3 |
| S-s17 | 0.9366 | 0.0384 | 0.0250 | 1.0 | 3 | 12/14 | 0 | 3 |
| S-s27 | 0.9366 | 0.0384 | 0.0250 | 1.0 | 3 | 12/14 | 0 | 3 |

**形态没变**：恒定 0/84，两段式 **75/84**（A 13/14 ×3 = 39/42，S 12/14 ×3 = 36/42），中位切换 1，S 的最大切换从 2 抬到 3（`005` 那两组多走了一次）。融合没有把策略从"动一下就 hold"里推出来。

### 5) 判别证据：底座偏置 vs 空间信息不可用

这是本轮真正要回答的问题。四组独立读数，全部是只读、可复算的：

#### (a) 底座先验：同一批像素、不同 mirror，预测分布几乎一致

冻结 B0（就是残差偏移所依附的那个 `policy_head`）在**同一批 telegraph 渲染**上的 masked argmax，按事件分（22 = 预告窗口内的决策数）：

| 事件 | mirror | 预告窗口内 屏幕右 / 左 / hold | 全 80 决策 屏幕右 / 左 / hold |
|---|---|---:|---:|
| `F1-005-m0` | 否 | **15 / 6 / 1** | 73 / 6 / 1 |
| `F1-005-m1` | 是（缺口在屏幕左） | **17 / 3 / 2** | 74 / 4 / 2 |
| `F1-007-m0` | 否 | **18 / 2 / 2** | 76 / 2 / 2 |
| `F1-007-m1` | 是（缺口在屏幕左） | **22 / 0 / 0** | 80 / 0 / 0 |
| `F1-010-m0` | 否 | **19 / 2 / 1** | 77 / 2 / 1 |
| `F1-010-m1` | 是（缺口在屏幕左） | **16 / 3 / 3** | 74 / 3 / 3 |

**镜像与否，B0 都 68–100% 投"屏幕向右"**；镜像世界里缺口在屏幕左，它没有任何一格把这当作"该往左"。这就是"底座偏置"的**预测分布级**证据（不是从行为反推的）。

同一张表上的蒸馏头更极端：A 在预告窗口内 **20–22/22 都是 hold**（只有 `005-m0` 两格向左、`005-m1` 两格向右），S 是 **19–22/22 hold**（`005-m0` 三格向左、`005-m1` 两格向左）。也就是说，**预告窗口里"什么都不做"本身就是学到的默认**，方向问题只发生在极少数真的动了的那几步。

#### (b) 线性可解码性：信息早就在输入里，但这个向量没有

用**只用 train 像素**的探针（F1 六个 train 事件、预告窗口内的决策行；ridge，标准化按训练折拟合并留一几何组）问："缺口在屏幕哪一侧 / 缺口中心在屏幕哪个 x，能不能从表示里线性读出来？"

| 表示 | 侧别准确率（留出几何） | 中心 x 误差 px（均值基线 54.67） | λ 扫描（8 个量级）侧别 | λ 扫描中心 px |
|---|---:|---:|---|---|
| **冻结 256 维 trunk features**（R20/R21 头的输入） | **0.985**（1.00 / 0.90 / 0.955） | **32.69** | **0.953 – 0.985** | 32.6 – 46.3 |
| **本轮 94 维池化向量**（融合层的输入） | **0.674**（1.00 / 0.50 / 0.523） | **55.20**（≈ 均值基线） | **0.523 – 0.680** | 49.6 – 63.8 |

**这是本轮最硬的一条**：R20/R21 的头**本来就看到**一个在留出几何上 **98.5% 线性可解码缺口所在侧**的 256 维向量（8 个 λ 量级里最低也有 95.3%），而本轮加进去的 94 维池化向量只有 **67%**，且**逐折极不稳**（100% / 50% / 52%），中心误差 55 px 与"直接猜训练均值"（54.7 px）没有区别。**"空间信息不可用"这个解释因此被否定**；反过来，"这个 94 维向量不足以承载 12–20 px 的精确定位"被证实。

（诚实标注两点：① 逐行相关的 `offset` 目标两臂都学不动——39.1 px 与 35.2 px，而"猜均值"只要 8.3 px，因为该目标自身方差很小、且模式就是"停在 0"；这个目标不作为结论依据。② 探针 p≈n，λ=1e-4 时几乎是插值器，所以结论只以**整条 λ 扫描**的形式给出，不以单个 λ 为准。）

#### (c) 融合层实际学到了什么：94.4% 是常数，5.6% 才随画面变化

把训练后的融合层在**全部 1120 行**上拆成"常数" `projection(mean spatial vector)` 与"条件化" `delta − 常数`：

| 检查点 | Δ 每行范数 | 常数部分 | 条件化部分 | **条件化占比** | 冻结 features 范数 | Δ / features |
|---|---:|---:|---:|---:|---:|---:|
| S-s7 | 4.398 | 4.398 | 0.246 | **5.59%** | 2.991 | 1.47 |
| S-s17 | 4.380 | 4.380 | 0.245 | **5.59%** | 2.991 | 1.46 |
| S-s27 | 4.448 | 4.448 | 0.249 | **5.59%** | 2.991 | 1.49 |

**融合层学到的几乎是一个常数偏移**（而且比它自己加进去的 features 还大 1.5 倍）：它把头的**工作点**挪了，而不是把"缺口在哪"接进去。这与 (b) 里"这个向量只能到 67%"是一致的：给一个几乎不含可线性解码空间信息的向量，零初始化的线性层当然只能学到它的均值。

#### (d) 同一份权重的 delta on/off 消融：delta 有用，但只作用于停车点

把**已训练**的 S 检查点在评估时 `fusion.enabled = False`（旁路 delta，**不改任何参数、不重训**），在 6 个 F1 留出事件上重跑闭环：

| 检查点 | 位改变的决策 / 480 | 可行（14 事件，含 delta） | 可行（14 事件，无 delta） |
|---|---:|---:|---:|
| S-s7 | 20（4.17%） | 3 | **0** |
| S-s17 | 22（4.58%） | 3 | **0** |
| S-s27 | 22（4.58%） | 3 | **0** |

逐格看（×3 seeds 一致）：有 delta 时 `007-m0/m1` **停进缺口（0 px）、`007-m1` 甚至不碰撞**；无 delta 时 `007-m0` 差 16 px、`007-m1` 差 24 px 或直接 hold，全部死在 `wall_frame=60`。`005-m0` 从屏幕向左（−40）变回屏幕向右（−24），`010-m0` 从 −8 变 −32，`010-m1` 从 −72 变 −48。

**结论**：delta 不是装饰品——去掉它 S 直接崩到 0/14；但它改变的是"走多远/停在哪"，**从不改变镜像组的屏幕方向**（`005-m1`、`010-m1` 在 on/off 两种情况下都是屏幕向右）。

#### (e) 输入镜像探针：功率极小，但给出一个事实

把每个 head 的**已存闭环轨迹**按它自己的 x 轨迹重新渲染并逐决策重放（单测锁死：重放必须逐决策复现存盘的 masked argmax，**7 个网络全部 80/80 = 1.0**），在预告窗口的每个决策上让**同一个网络状态**再看一次横向镜像的 playfield（HUD 不动）：

| 网络 | 窗口内行数 | 翻转数 | 无条件翻转率 | 其中"真动"的行 | 动时翻转 | hold 行 | 重放保真 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 132 | 17 | 12.9% | 123 | 13.8% | 9 | 1.0 |
| A-s7/s17/s27 | 132 | 4 | 3.0% | **4** | **4/4** | 128 | 1.0 |
| S-s7 | 132 | 2 | 1.5% | 5 | 2/5 | 127 | 1.0 |
| S-s17/s27 | 132 | 3 | 2.3% | 6 | 3/6 | 126 | 1.0 |

**这个探针的功率很小，必须这样读**：蒸馏头在预告窗口里 **95–97% 的决策是 hold**（A 128/132、S 126–127/132，而 B0 只有 9/132），所以"翻转率"的分子分母只有 4–6 步；A 的 4 步全翻、S 的 5–6 步翻 2–3 步，方向上是"动的时候确实受画面影响"，但 **n≈5 不能当速率**。另外它把 player 一起镜像了，所以翻转不能单独归因给缺口。它给出的可靠事实只有一个，而且很重要：**预告窗口里的默认动作是 hold，方向偏好只作用在极少几步上**。

#### (f) 列 profile 不是镜像对称的

同一组 m0/m1 的 94 维列 profile 与其镜像逆序比较：相关 **0.387 / 0.437 / 0.391**，平均绝对差 0.00996/0.01047/0.01102，相对差 **0.458 / 0.483 / 0.507**（相对"合并标准差"）。F8 的列轴就是屏幕 x 轴，一个"读出布局"的骨干在这里本应接近镜像对称；实测只有 0.39–0.44，说明**池化后的列 profile 并不携带干净的屏幕横向布局**。

#### 判别小结

| 读法 | 支持证据 | 反对证据 | 本轮裁定 |
|---|---|---|---|
| **底座偏置主导** | B0 在镜像世界仍 68–100% 投屏幕右 (a)；两臂镜像走对率都是 3/9 (2)；预告窗口 95–97% hold (e)；A 只有 3 格靠 `007-m1` 蒙对 (2) | 256 维 features 已可解码缺口侧 98.5% (b)，说明"信息够了"——这与"偏置主导"不矛盾，恰恰是它的前提 | **成立** |
| **空间信息不可用** | 94 维向量只得 67% 且逐折不稳 (b)；列 profile 非镜像对称 (f)；融合层 94.4% 学成常数 (c) | 同一骨干的 256 维 features 已可解码 98.5% (b)，而那就是 R21 头的输入 | **对"输入"不成立；对"这个 94 维向量"成立** |

所以 R21 的负结果不能归因于"信息缺失"（本轮实测否证），也不能归因于"补上了就会被用"（本轮实测未发生）；**约束在输出形态与这个向量的分辨率上**。

### 6) train 闭环（**已见**：就是用来生成标签的那 14 个事件）

| 族 | n | B0 | A-s7 | A-s17 | A-s27 | S-s7 | S-s17 | S-s27 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 | 6 | 0 | 1 | 1 | 2 | **2** | **2** | **2** |
| F2 | 2 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| F3 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **合计** | 14 | **0** | 2 | 2 | 3 | **3** | **3** | **3** |

**"已见"上 S 确实多赢了一格**：`006-m0` 三个 seed 全部从"差 16 px / 死"变成"停进缺口"（A-s27 本来也进），F1 已见 1/1/2 → 2/2/2。但**镜像组在已见里也照样 3/9 走对、6/9 走反、0 格停进缺口**（A 与 S 完全相同）。也就是说：融合让"已经见过的 base 世界"多停进一个，但**没有改变镜像世界的方向结构**，而留出侧连那一格都没拿到。

## 与 R21 的对比（判据的结论）

| 臂 | seed | R21 F1 留出 | R22 F1 留出 | Δ | R21 合计 | R22 合计 | Δ | 14 事件 first hit 相同数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A（=R21 A，逐位复现） | 7 / 17 / 27 | 2 / 2 / 2 | 2 / 2 / 2 | 0 / 0 / 0 | 3 / 3 / 3 | 3 / 3 / 3 | 0 / 0 / 0 | **14 / 14 / 14** |
| S（空间融合） | 7 / 17 / 27 | — | 2 / 2 / 2 | — | — | 3 / 3 / 3 | — | **14 / 14 / 14**（对 A） |

按预注册判定：**S 对 A = `unchanged`（三个 seed 的 F1 差全为 0）**；而 A 对 R21 = **逐位相同**，所以"本轮 A 就是 R21 A"不是假设而是被检验过的事实。

- **总数没动**，但**运动确实变了**：`005-m0` base 组 3 格 hold → 屏幕向左，`005-m1` 3 格屏幕向右 184 → hold/向右 208；其余 3 个 F1 单元格的停车位置与误差逐 px 相同（`007-m1` 三个 seed 都停进缺口，只是 s17/s27 停在 168 而不是 160）。
- **机制指标一格没动**：镜像走对率 3/9 → 3/9；`in_gap` 3 → 3（base 与 mirror 都是）。所以本轮不能声称"方向被修好了"。
- **两臂的 first-hit 逐事件完全相同**（14/14 ×3）：这意味着**没有一个事件的碰撞时刻被融合改变**，只有碰撞之前的路径变了。

## 判定（有界，不做显著性声明）

1. **控制成立**：A 的三个 checkpoint SHA 与 R21 完全相同，6 个 (arm, split) 的闭环**每一步**与 R21 相同（`equivalence.all_equal = true`）；初值 SHA 与 R16/R20/R21 表一致。本轮的两臂比较因此是干净的单因子比较。
2. **唯一干预的结论：F1 留出没有变好也没有变坏——`unchanged`**（2/2/2 vs 2/2/2，逐 seed Δ=0，14/14 first hit 相同）。按预注册规则，这就是"没变"，不重跑、不调参。
3. **"空间条件化"这个结构修订失败，但失败得很具体**：
   - **不是没梯度**：融合层 199/200 步有非零梯度，delta 从 0 长到每行 ~4.4（比 features 本身大 1.5 倍）。
   - **不是没影响**：关掉 delta 后 S 从 3/14 掉到 0/14，4–5% 的决策改变。
   - **是"学成了再偏置"**：delta 的 **94.4% 是常数**、只有 **5.6% 随画面变化**；改变的是停车点，不是镜像方向。
   - **是"向量太粗"**：94 维向量对缺口侧别只有 67%（逐折 100/50/52%），对中心 x 的误差等同瞎猜；而它来自一个**一列 = 16 native px** 的 28×24 图，F1 缺口只有 12–20 px。
4. **"空间信息不可用"被本轮否证**：同一个冻结骨干的 **256 维 trunk features 在留出几何上线性可解码缺口侧 98.5%（λ 扫描 ≥95.3%）**、中心 x 误差 32.7 px（基线 54.7）。所以 R21 的失败**不能**再归因于"输入里没有空间信息"。
5. **方向偏置仍是主因，而且现在有预测级别的证据**：冻结 B0 在镜像世界（缺口在屏幕左）仍然 16–22/22 投"屏幕向右"；蒸馏头在预告窗口内 95–97% 的决策是 hold，动的那几步方向由底座先验决定。两臂镜像走对率都是 3/9，其中 3 格全部来自同一个事件 `007-m1`。
6. **本轮不宣称任何战术理解**，也不声称"空间融合无用"：只说"**这一个 94 维池化向量 + 零初始化线性 + 200 updates + 单路线 one-hot** 的组合，没有改变镜像方向"。

## 为什么仍失败（机制，逐条）

- **信息够，形态不够。** R20/R21 的头输入里已经有 98.5% 线性可解码的缺口侧别信息，但它的输出形态是"一次两段式"：在 1120 个已执行动作里 **94% 是 hold**，中位切换 1 次。预告窗口内更极端：**95–97% 的决策是 hold**。要在 45 帧里"早动、动对方向、到点停"，需要三段以上，而这套监督（单路线 one-hot、200 updates、batch 1）把模型稳定地推到了"动一下然后 hold"。
- **融合层在数值上只能学成常数。** 94 维向量的 mean/std ≈ 11，即"同一个几乎不变的向量"；零初始化的线性层第一步之后得到的梯度方向就主要是"均值方向"，200 步 Adam 后 **94.4% 的 delta 范数是常数**。它把 residual 头的工作点整体挪了 1.5 倍 features 的尺度，于是改变了停车点（±8→±40 px），却没有改变"屏幕向右"这个方向前提。
- **这个向量的分辨率与任务不匹配。** 一列 F8 = 16 native px，F1 缺口 12–20 px；7×6 池化后一格 = 64 px。侧别（左/右半场）勉强可读（67%，且逐折不稳），**停车点（±10 px）在这个向量里基本不存在**。因此"给它一个池化的全局空间向量"不可能解决"停进 24 px 宽的空档"。
- **镜像轴本身在池化后不干净。** 列 profile 与其镜像逆序相关只有 0.39–0.44、相对差 0.46–0.51；一个理想的"布局读取器"在这里应接近 1。这解释了为什么 94 维向量对侧别的可解码性远低于 256 维 features（后者还含 `global_features` 的坐标通道与 `location_stats`）。
- **底座先验没有被任何东西挑战。** 两臂的 `base_logits` 完全同源（融合只进 GRU/residual 的条件向量），KL 锚本轮未开（两臂都是 R20/R21 的 control loss）。融合层能改的只有 residual 的输入，而 residual 的 18 维输出里"屏幕向右"那一维本来就被底座 logits 抬着。

## 限制（必须与结论一起读）

1. **n = 3 个 seed、7 个留出组**，F1 留出只有 3 组 / 6 个事件；镜像不是独立样本。`2/6`、`3/9` 这样的数字是**计数**，不是速率。
2. **本轮只改了一个结构**（融合层 + 头部一行 detach），因此"F1 没学会"不能外推成"空间条件化无用"。**没有做的消融**：更细的空间源（F4 的 48 列 = 8 native px/列）、不做池化直接取列向量、非零初始化、给融合层加 LayerNorm、把融合接到 `base_logits` 而不是 features、以及 KL 锚下的同一融合（本轮两臂都是 control loss）。这些都是**下一轮的单因子**，本轮一条都没测。
3. **输入镜像探针功率极小**（分子分母 4–6 步），且把 player 一起镜像；它的结论只用于"窗口内默认是 hold"这一条。
4. **线性可解码性探针 p≈n**（94/256 维 vs 128 行），只以 λ 扫描给结论；`offset` 目标因方差太小不可用，已如实标注、未用于结论。
5. **delta on/off 消融是反事实**：S 的头是在**有 delta** 的输入上训练的，关掉 delta 是分布外输入，所以"无 delta 0/14"应读作"delta 承担了头的输入基准"，不是"delta 提供信息"。
6. **教师不是学习到的策略**：`TacticalTeacherV2` 被直接告知世界几何，是 12 帧 beam 规划器；14/14 只说明"这些世界对知道参数、能前瞻 6 步的规划器可玩"。
7. **F1 的胜利条件是 12–20 px 的精确停车**，是几何构造，不是"会玩"；F3 是卡时机的越线。
8. **数据是合成固定 y 世界**：两帧动作、无射击、无炸弹、无真实 TH08 像素与判定；horizon 固定 160。
9. **只有一种超参配置**（200 updates / lr 1e-3 / clip 1 / batch 1 / 单路线 one-hot）；没有扫"更多 update 是否就学会"。
10. **有一次预先披露的冒烟运行**：预注册之前在 `.tmp/spatialfuse-smoke/` 用同一份冻结代码跑过一次完整流水线（产物写 scratch），其数值未用于任何结论；它与正式 run 的 **6 个 checkpoint 文件 SHA、14 个闭环文件、`f1-gap.json`、`probes.json`、`records.json`、`baselines.json` 全部逐字节相同**（只有含墙钟时间的 `timings.json` 与 `checkpoints.json` 里的秒数不同），因此本轮可以声明"两次独立进程逐位相同"。
11. **不产生任何可部署产物**：六个 `research-heads.pt` 是研究用 head，`deployable = false`、`formal_gate = false`、`game = false`。

## 对既有结论的处置

| 既有结论 | 处置 |
|---|---|
| R21："F1 留出 A 3/3/3→2/2/2、B 2/3/3→2/2/2；方向 base 15/18 走对、镜像 15/18 走反；死亡 100% 在 `wall_frame`" | 本轮把 A 臂**逐位复现**（checkpoint SHA 与 6×14×80 步全部相同），因此 R21 的 A 数字在本轮是可对照的基线，而不是引用；R21 的方向/死亡结构也在 S 臂上原样重现 |
| R21："把缺口补进画面仍不足以学会" | **再次确认，并且更硬**：不只 F1 计数相同，**14/14 事件的 first hit 两臂完全相同** |
| R21 的隐含推论"策略拿不到/用不上缺口的精确横向位置" | **一分为二**：'拿不到'被本轮否证——同一骨干的 256 维 features 已能 98.5% 线性解码缺口侧别；'用不上'被本轮证实——预告窗口内 95–97% 的决策是 hold，方向由底座先验决定 |
| "给一个空间向量就能把方向补上" | **必须撤回**：94 维池化向量把侧别可解码性从 98.5% 降到 67%，融合层 94.4% 学成常数，镜像走对率 3/9 → 3/9 |
| R20/R17："KL 锚改变分布但不改形态" | 本轮两臂都**没有**锚，形态仍是两段式（75/84）、hold 0.937–0.943、中位切换 1——说明"两段式"不是锚造成的 |
| R20："两段式 84/84、恒定 0/84" | **保持**（75/84 两段式、0/84 恒定） |
| R22 自己的架构观察 | **新增一条可复用的负结果**：`TacticalPolicy.forward_features` 会 detach 输入，任何建在它上游的可训练模块都拿不到梯度；零初始化的 `residual` 还使 **step 0 的 features 侧梯度恒为 0**。以后要接上游模块，只能用变体（本轮做法）或改父类 |
| 后续该怎么做 | 不要在"再给一个池化向量"这条路上继续。下一步应把**空间源的分辨率**（F4 列向量 / 不池化 / 列 softmax 定位）与**输出形态**（多段式、显式停止头、逐帧 stop 信号）分别作为**预注册的单因子**去测；同时保留本轮的四组判别探针作为验收口径，因为它们能区分"信息不在"与"信息在但没用" |

本轮不修改 `ledger-round17.md`（属上一轮台账）；建议父级在 round22 台账登记：**空间条件化融合 = F1 留出 `unchanged`（2/2/2 对 2/2/2，14/14 first hit 相同）**、**镜像走对率 3/9 → 3/9（一格没变）**、**融合层 94.4% 学成常数再偏置（条件化只占 5.6%）**、**delta on/off：3/14 → 0/14（4.2–4.6% 决策改变，方向不变）**、**256 维 features 已可解码缺口侧 98.5% 而 94 维向量只有 67% → "信息不可用"被否证**、**预告窗口内 95–97% 的决策是 hold**、**F1 死亡仍 100% 在 `wall_frame`（62 次）**、**R21 A 被逐位复现**。

## 执行与退出码

```powershell
# 1) 单测（写工具期间共 15 轮，逐条归因见 command-results.md）
python -m pytest tests/test_v373_spatial_fuse.py -q -p no:cacheprovider --basetemp .tmp/v373-spatialfuse-tests-15
#    22 passed in 6.09s，exit 0
# 2) 旧轮回归（旧模块未被改动）
python -m pytest tests/test_v373_telegraph.py -q -p no:cacheprovider --basetemp .tmp/v373-r21-regression-r22
#    27 passed in 15.30s，exit 0
python -m pytest tests/test_v373_valid_curriculum.py -q -p no:cacheprovider --basetemp .tmp/v373-r20-regression-r22
#    21 passed in 18.87s，exit 0
# 3) 冒烟（scratch，预先披露）：预注册 → run → verify，共 5 轮
python -m tools.rl.v373_spatial_fuse preregister --out .tmp/spatialfuse-smoke   # exit 0
python -m tools.rl.v373_spatial_fuse run --out .tmp/spatialfuse-smoke           # exit 0，约 296 s
python -m tools.rl.v373_spatial_fuse verify --out .tmp/spatialfuse-smoke        # exit 0
# 4) 正式冻结预注册（判据 + 33 个身份 SHA + worlds digest + 94 维向量 SHA + 空间统计 + 28 条几何 + R21 参照）
python -m tools.rl.v373_spatial_fuse preregister
#    exit 0；无 stdout；criterion SHA 8ff1f14b…；preregistration.json SHA 9eed34e1…
# 5) 唯一一次 run（后台 job，自行收取）：
python -m tools.rl.v373_spatial_fuse run
#    exit 0；墙钟 ≈ 298 s（data 1.355 s / 训练 69.016 s（其中纯优化 65.697 s）/ 闭环 168.614 s / 探针 58.634 s）
# 6) 只读复核
python -m tools.rl.v373_spatial_fuse verify
#    exit 0；81 run 产物 / 83 根产物 / 70 资产（含 14 个空间向量）/ 6 checkpoint /
#    14 闭环文件 / 28 事件文件 / 168 基线行 / 84 条 F1 机制行 / 7 个翻转探针网络
# 7) run 之后复跑测试（冻结的测试文件仍通过）
python -m pytest tests/test_v373_spatial_fuse.py -q -p no:cacheprovider --basetemp .tmp/v373-spatialfuse-tests-final
#    22 passed in 6.11s，exit 0
```

退出码依次 **0 / 0 / 0 / 0 / 0 / 0 / 0 / 0**（1 次正式 preregister、1 次正式 run、3 次正式 verify：run 之后、写入日志之后、写入 command-results 之后）。命令、日志与测试迭代（共 15 轮，其中 **4 个真实缺陷**在工具/测试里：①`r20.all_names()` 返回的是 R20 的 A/B 两臂，导致闭环在 `B-s7` 上 KeyError；②`inference_mode` 产出的 tensor 不能当 autograd 输入，融合层拿不到梯度；③`replay_images` 多放了一张初始帧，使输入镜像探针的栈整体错位一格；④探针脚本文本比对里 `capture[2]['playfield'][0]` 索引错维度）都在 [command-results.md](<../evidence/spatialfuse-round22/command-results.md>)。

## 证据文件

- [preregistration.json](<../evidence/spatialfuse-round22/preregistration.json>)：判据（`criterion`，SHA `8ff1f14b…`）、**33 个身份 SHA**、sources（R18 manifest / R19 split / 底座 / R21 六件产物）、worlds digest、逐组 split、**融合形状与 94 维定义**、**空间向量 SHA `83a75f73…` 与逐事件统计**、**14 个事件的 feature 范数**、**28 条几何**、**14 个事件的资产绑定**、R21 参照记录（含逐 seed F1 留出与六头方向计数）、每 seed 初始 head SHA 与两臂融合初值 SHA。文件 SHA `9eed34e19aeaa510af65c2b0d703960d7cdbf567b6e5f31bca0aa0ac419185d7`。
- [run/records.json](<../evidence/spatialfuse-round22/run/records.json>)：14 个事件的**R21 资产绑定**（pixels/history/labels/rollout 的 path + SHA + 字节）、`spatial` 14 条引用（path/SHA/字节/形状）、`totals`（2 714 834 B + 149 991 B）、`failures = []`、空间向量定义。
- `run/spatial/<event_id>.npz`：14 × `[80,94]` float32 空间向量（deflate），SHA 与预注册冻结的 `83a75f73…` 一致。
- [run/checkpoints.json](<../evidence/spatialfuse-round22/run/checkpoints.json>) 与 `run/train/<name>/{initialization.json,training.json,research-heads.pt}`：6 个 checkpoint 的初始/末 head 与融合 SHA、活跃参数（429 200 / 453 520）、逐 step loss、**融合梯度范数与 delta Frobenius 范数序列**、离线指标、`matches_round21_control`。
- [run/baselines.json](<../evidence/spatialfuse-round22/run/baselines.json>)：六条 R18 基线的 **168 行**（validation 84 + train 84），与 R21 逐行相同。
- [run/f1-gap.json](<../evidence/spatialfuse-round22/run/f1-gap.json>)：**84 行 F1 机制诊断**（7 网络 × 6 事件 × 2 split），每行含起点、缺口（canonical + 屏幕）、窗口、窗口内决策数、canonical/屏幕方向、首次移动帧、首次进入缺口帧与决策序号、`wall_frame-1` 的 x、是否在缺口内、**带符号误差 px**、机制分类、`toward/hold/away` 判定。
- `run/closed-loop/<B0|A-s*|S-s*>-<validation|train>.json`：**14 个文件**，每条 14 事件 × 80 决策（`raw_unmasked_candidate`/`candidate`/`actual_action`/`x_before`/`x_after`/`hit`/`first_collision_frame`）+ 分族聚合 + `behaviour`。
- `run/events/<event_id>.json`：**28 个逐事件 JSON**（14 train 含教师路线与资产绑定、28 个都含几何、B0/六头闭环、六条基线、F1 机制行、动作分布）。
- [run/probes.json](<../evidence/spatialfuse-round22/run/probes.json>)：本轮的判别证据全集 —— `flip`（7 网络 × 132 行 + 逐事件 + 重放保真）、`prediction`（B0 与六头的逐事件 masked argmax 分布，全部 80 决策与窗口内 22 决策）、`base_bias`、`spatial`（94 维与 256 维的留出几何可解码性 + **8 个 λ 的扫描** + 逐折明细）、`symmetry`（镜像对列 profile）、`fusion`（常数/条件化分解、权重与偏置范数）、`ablation`（**同一份权重的 delta on/off 逐格对照**）。
- [run/summary.json](<../evidence/spatialfuse-round22/run/summary.json>)：本轮汇总（分族/分臂/分 seed、逐事件 first hit、`comparison`（S−A 与**逐 seed 相同 first hit 格数**）、`equivalence`（控制臂逐位复现）、`validation.direction`（方向与停车）、`probes`、`round21_reference`、`environment`、全部 `flags`、`limitations`）。
- [run/sha256.json](<../evidence/spatialfuse-round22/run/sha256.json>)（81 个 run 产物）与 [sha256.json](<../evidence/spatialfuse-round22/sha256.json>)（根目录 83 个文件）；已用 `verify` 逐个复核，0 处不符。
- 命令与日志：[command-results.md](<../evidence/spatialfuse-round22/command-results.md>)、`logs/{preregister,execution,verify,tests-after-run,regression-round20,regression-round21}.log`（在 run 之后写入，因此不在根 `sha256.json` 表内；写入后重跑 `verify` 仍 exit 0）。
