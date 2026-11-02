# P4 F1 数据规模单因子：6 组 → 12 组训练集（round25）

## 一句话

R18/R19 的 F1 只有 6 组（3 训练 + 3 留出，训练侧 6 个事件）；R21–R24 在留出侧反复看到同一个形状：base 组朝缺口，镜像组**屏幕向右**（R24 的 A 臂 base 6/9、镜像 3/9，屏幕上 12 右/3 左/3 hold）。本轮只检验**数据规模**这一个假设，只做 F1、只用 A 臂（无辅助头）：用 R18 的同一生成器与同一判据，在预注册的更大预算（seed 25001 / 400 个候选）下搜到 **16 个新组（32 例，含镜像）**，与 R8/R9/R10/R16/R18/R19 的 59 个冻结签名零重复；再按组重新划分为 **train 12 组 / held-out 10 组**（实测 22 组，交替 + 一次已记录的翻转），其中 **R19 原冻结的 3 组 F1 留出保持留出**，作为跨集对照。训练两个只有训练集不同的 A 臂：**S = 旧 6 组（12 事件）**、**L = 新 train 12 组（24 事件）**，各 3 seeds、200 updates、R20/R21 其余超参与 R21 telegraph 渲染全部不动；同 seed 两臂初始 head SHA 逐位相同。

**判定：`scale_not_the_cause`（规模不是主因）。** 在**新 held-out** 上，L 臂在"缺口在屏幕左侧、屏幕向右先验是错的"7 个格里，**flip = 0/0/0**（三个 seed），与 S 臂、与冻结基线 B0 完全相同；镜像子集（2 格）也是 **0/6**。P2 成立：L 在"缺口在屏幕右侧"的 7 格里 keep **7/7/7**（21/21），即 base 的行为被完整保留。放大到 12 组/24 事件没有让模型走过**一次**屏幕向左。

**但这个 null 必须与三件事一起读**：

- **原始"镜像走对率"在新 held-out 上是几何假象**：新留出组混合了缺口在屏幕左右两侧，冻结 B0（恒屏幕向右）自己就在镜像组拿 **5/7**、base 组 **2/7**；S 与 L 的原始计数（base 6/21、镜像 15/21）**恰好等于 B0 的比率**。所以本轮的主指标必须在预注册里就改成"按缺口所需的屏幕方向条件化"（need-left 7 格 / need-right 7 格）。这个混杂是 scratch 冒烟跑暴露的，指标在**正式预注册之前**改定（criterion SHA `d2ae92b6…` → `5ad3ea35…`），冒烟的任何数值都没有进入本轮证据。
- **S 与 L 不是嵌套的**：S 是旧 6 组（含 R19 冻结的 3 个留出组），L 是 3 个旧 train 组 + 9 个新组。所以两臂同时变了"规模"和"组成"；不过两臂的 flip 都是 0，这个 null 对组成差异是稳健的。S 在 R19 冻结留出上的读数是**已见**读数（它训练过那 3 组），从不当留出结论使用。
- **200 updates 固定**：S 看到 16.7 个 epoch，L 只有 8.3 个。L 的 flip=0 不能排除"24 个事件在 200 步里训不动"；但 S 的 flip 也是 0，而 L 的离线 masked 准确率只比 S 低约 0.4 个百分点（0.968 vs 0.972），所以"规模不够"与"预算不够"在本轮**无法完全分离**，只能确定：**把世界数翻倍本身没有产生任何一次向左**。

本轮不宣称战术理解，不产生可部署产物，不开启正式门，不运行游戏，不调参。

## 隔离与边界

- 只新增 4 个路径：[本轮工具](<../../../../../tools/rl/v373_f1_scale.py>)、[本轮测试](<../../../../../tests/test_v373_f1_scale.py>)、本报告、[evidence 目录](<../evidence/f1scale-round25/preregistration.json>)（含 `run/`、`logs/`、`command-results.md`）。
- **不改旧模块**：工具 `import` 了 `v373_world_filter`（R18 的生成器/六基线判据/签名）、`v373_telegraph`（R21 的 telegraph 渲染器、`gap_diagnostics`、`run_event_rendered`、`closed_loop`）、`v373_valid_curriculum`（R20 的 `encode/infer/arm_loss/offline/Network/compact/behaviour/six_baselines`）、`v373_spatial_fuse`（R22 的 `cell_diagnostic/group_summary/mechanism_block`）、`v373_seed_confirm`（R16 的 `fresh_policy/state_sha/initial_head_shas`）与两个教师/像素模块，**一个字符都没有改**。预注册冻结 **36 个身份 SHA**（含本轮工具与测试、5 个旧工具、12 个 `touhou_rl` 模块、基线 `policy.pt`、R18 manifest、R19 split、R21 records、R24 summary、R19 的 12 个逐世界教师记录），`verify` 逐个复核一致。
- **新旧证据的边界**：只读 R18 manifest / R19 split / R19 逐世界教师记录 / R21 records / R24 summary；不删除、不改写、不覆盖。
- **教师调用**：只为 30 个被渲染的事件各跑一次 `TacticalTeacherV2.rollout`（R21 的 `render_event` 内部），30/30 `verify = feasible`；**没有新教师定义**，teacher 模块 SHA 在 36 个身份里。6 个旧 train 事件（002/006/009）的渲染结果与 R21 存盘**逐字节相同**（pixels/history/labels 三个 SHA 全等）；12 个旧组事件的 80 动作与 R19 的 `teachers.teacher_v2.actions` **12/12 相同**。
- **没有 test split**、**未开启正式门**、**不产生可部署 checkpoint**（`deployable = false`）、**不运行游戏**、**不调参**（`tuning = false`）。
- **所有 JSON 以 `x` 打开、所有权重以 `xb` 打开、所有目录 `exist_ok=False`**（工具直接调用 R20 的 `write_json/write_torch/new_dir`，测试逐个覆盖）。
- **真实 CUDA、FP32、两个 TF32 关闭**：`environment` 记录 `NVIDIA GeForce RTX 4070 SUPER`、`torch 2.14.1+cu132`、`tf32_matmul = false`、`tf32_cudnn = false`、`matmul_precision = highest`。

## 设计与唯一因子

| 臂 | 训练集 | 组 / 事件 | epoch（200 updates） | 损失 |
|---|---|---|---|---|
| **S** | R18/R19 的**全部 6 个旧 F1 组** | 6 / 12 | 16.67 | R20 的 control loss（无 KL、无辅助头） |
| **L** | 新划分的 **train 12 组** | 12 / 24 | 8.33 | 同上（逐行相同） |

- 两个训练集只差"喂进去的世界"；网络、损失、优化器（Adam 1e-3）、clip 1.0、batch 1、seeds 7/17/27、200 updates、初始化规则全部是 R20/R21 的代码对象（`import` 而非复制）。
- **同 seed 两臂初始 head SHA 逐位相同**（`66e5f85fe6b8e186…` / `f7d03e205f378179…` / `276573d10853f3e7…`，也就是 R16/R20/R21/R24 记录的同一个初值），所以差异只能来自训练集。
- 训练只做 F1；没有辅助头、没有 R23/R24 的标签。

## 搜索：R18 同一生成器/判据，更大预算

预注册的搜索规格（`preregistration.json` → `search` / `criterion.search`）：

| 项 | 值 |
|---|---|
| 族 | **F1 only** |
| 生成器 | `tools.rl.v373_world_filter.draw_geometry`（未改，`DOMAINS['F1']` 原样冻结） |
| 判据 | `tools.rl.v373_world_filter.evaluate_world`（未改）：恒 hold/right/left、每 16 次决策的左右交替、种子随机、教师首动作弱基线**六条全部失败**，且独立 `solve()` 给出完整 80 动作见证并 `verify()` 非提前终止重放通过 |
| seed / 预算 / 目标 | **25001 / 400 个候选 / 16 组（32 例）** |
| 放松条款 | 预算内凑不满就如实报数量与逐候选原因；**不放宽任何阈值、不改任何已有世界、运行后不加预算** |

实际（唯一一次正式 run，[run/search.json](<../evidence/f1scale-round25/run/search.json>)）：

| 量 | 值 |
|---|---:|
| 实抽候选 | **50**（命中目标即停，预算 400 未用尽） |
| 收入 | **16 组 / 32 例** |
| 丢弃 | 34（invalid **27**、unknown **7**、签名重复 **0**、抽样不可用 **0**） |
| `target_met` / `budget_exhausted` | true / false |
| 单动作常量可通过（补充诊断，非判据） | **0 / 32** |
| 与冻结签名重复 | **0** |

冻结签名普查（[run/search.json](<../evidence/f1scale-round25/run/search.json>)）：R8 dev/val **21**、R9 **12**、R10 **8**（仍是 R8 子集）、R16 **12**、R18 **14**、R19（同一批 14 组的 F1 6 组）；并集 **59**。16 个新组签名与并集交集为空、组内互不相同。

（预注册同时冻结了"新增 < 8 组 → train 用所有可得组、并声明规模差异不足以支撑强结论"的短差分支，测试覆盖该分支；本轮 50 个候选就命中了 16 组目标，**未触发**。）

**16 个新组（本体；镜像为 `384 - start_x`，同组同 split）**：

| 组 | coverage | gap | wall | start | 见证 checked | 相对恒 hold 改变步数 |
|---|---|---|---:|---:|---:|---:|
| `wf25-F1-001` | (8,376) | (200,220) | 44 | 168 | 13 | 12 |
| `wf25-F1-003` | (8,372) | (128,142) | 44 | 182 | 8 | 12 |
| `wf25-F1-007` | (12,376) | (256,268) | 56 | 308 | 8 | 12 |
| `wf25-F1-008` | (8,376) | (120,140) | 60 | 24 | 14 | 24 |
| `wf25-F1-012` | (8,372) | (120,136) | 72 | 80 | 13 | 12 |
| `wf25-F1-015` | (8,372) | (176,196) | 52 | 152 | 17 | 4 |
| `wf25-F1-019` | (12,376) | (304,318) | 76 | 366 | 8 | 12 |
| `wf25-F1-020` | (8,376) | (272,288) | 64 | 336 | 8 | 12 |
| `wf25-F1-023` | (8,372) | (144,164) | 44 | 112 | 13 | 12 |
| `wf25-F1-028` | (8,376) | (168,184) | 52 | 224 | 8 | 12 |
| `wf25-F1-033` | (12,376) | (272,292) | 56 | 372 | 3 | 12 |
| `wf25-F1-040` | (12,376) | (248,260) | 60 | 356 | 3 | 12 |
| `wf25-F1-042` | (8,376) | (296,310) | 44 | 248 | 13 | 12 |
| `wf25-F1-046` | (12,376) | (152,166) | 48 | 206 | 8 | 12 |
| `wf25-F1-048` | (8,376) | (144,164) | 44 | 212 | 8 | 12 |
| `wf25-F1-049` | (8,376) | (272,286) | 56 | 326 | 8 | 12 |

15/16 组的见证是"短促平移后停在缺口里"（12 步），`008` 是 24 步，`015` 是 4 步；32 例全部 `reference.witness_replay = true`。

## 按组重新划分（不含泄漏）

规则（冻结在 `criterion.split`）：组顺序 = R18/R19 的 6 个 F1 组（R19 顺序）+ 新组按接受顺序；**偶数下标 → train，奇数 → held-out**；3 个冻结的 R19 验证组（`005/007/010`）**强制 held-out**；若 train < 12 且 held-out > 5，从队尾把 held-out 翻到 train（每次翻转都记录）；镜像原子，组不跨 split。

| 组来源 | 组数 | 事件数 |
|---|---:|---:|
| train | **12**（3 个旧 train + 9 个新） | 24 |
| held-out | **10**（3 个冻结 R19 验证组 + 7 个新） | 20 |
| 其中 R19 原冻结留出（跨集对照） | 3 | **6** |
| 其中新 held-out | **7** | **14** |

- 实际翻转 **1 次**：`wf25-F1-049` 从 held-out 翻到 train（"train floor 12"）。翻转原因与组名逐条在 [run/split.json](<../evidence/f1scale-round25/run/split.json>)。
- **断言（`split.assertions.ok = true`，逐条在 summary）**：train/held-out 组不相交且覆盖全部 22 组；R19 冻结验证组全在 held-out；每组的 base/mirror 同 split；train 与两个评估集（`r19_frozen` 6 例 + `new_heldout` 14 例）**事件不相交**；**L 臂没有训练任何评估事件**；S 臂恰好是那 6 个旧组。
- **S 臂的泄漏是设计的一部分，必须披露**：S = 旧 6 组，包含 R19 冻结的 `005/007/010`，所以 S 在 `r19_frozen` 上的读数是**已见**读数，**从不当留出结论**。L 臂在这 3 组上完全干净（`large_arm_trains_on_r19_frozen = false`）。

## 主指标：为什么必须按"缺口在屏幕哪一侧"条件化

这是本轮最容易被读错的地方，写在预注册的 `criterion.judgement.geometry` 里。

- R22 的 `toward` 是**canonical** 方向；canonical 在 base 世界等于屏幕方向，在镜像世界翻转。冻结基线 B0 恒"屏幕向右"，所以它在"缺口在屏幕右侧"的格上必然 `toward`，在"缺口在屏幕左侧"的格上必然 `away`。
- **R19 冻结留出恰好是纯的**：3 个 base 事件的缺口全在屏幕右侧、3 个镜像全在屏幕左侧，所以 R24 的"镜像 toward"就是"先验错的那一半"。
- **本轮新 held-out 不是纯的**：7 个新留出组里，5 个 base 的缺口在玩家左侧（先验错），2 个 base 在右侧；镜像随之变成 2 个"先验错"、5 个"先验对"。于是 **B0 自己在镜像组就能拿 5/7**，原始"镜像 toward"完全被几何主导。

因此预注册的主指标是**方向条件化**的：

- `need_left`：缺口的**屏幕**中心在玩家起点**左侧**（冻结的屏幕向右先验是错的）；本轮新 held-out 每个网络 **7 格 = 5 base + 2 镜像**。
- `flip`：在 need_left 格上实际**屏幕向左**。
- `need_right`：相反（先验对）；本轮 **7 格 = 2 base + 5 镜像**。
- `keep`：在 need_right 格上实际**屏幕向右**。
- **P1** = L 在新 held-out 的 need_left 格上的 flip 数/seed（7 格）；≥2 个 seed > 0 为 `scale_helps`，三个 seed 全 0 为 `scale_not_the_cause`。
- **P1-mirror** = 其中镜像子集（2 格），单独报；这是 R19/R24 镜像组的直接类比。
- **P2** = L 在 need_right 格上的 keep 率**每个 seed 都 ≥ 0.5** 才叫 base 保持。
- **P3** = R19 冻结留出（其 need_left 3 格**恰好就是 3 个镜像事件**）对 S/L 与 R24 A 的对照；S 是已见读数，不承担主结论。

**披露**：这个混杂是 scratch 冒烟跑（`.tmp/f1scale-smoke`，第 6 次尝试）发现的——当时原始"镜像 toward"给出 5/7，而 B0 也是 5/7。指标在**正式预注册之前**改成上述条件化版本（criterion SHA `d2ae92b6…` → `5ad3ea35…`），测试同步更新，冒烟的数值一处也没有进入正式证据；正式 run 的 6 个 checkpoint SHA 与第 7 次冒烟**逐位相同**（见 §执行与退出码）。

## 训练：2 臂 × 3 seeds，一次跑完

| checkpoint | 集 | 初始 head（前 16） | 末 head（前 16） | ckpt（前 16） | loss@0 | loss@200 | epoch | 事件 | 秒 | 离线 masked（已见） |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| S-s7 | S | `66e5f85fe6b8e186` | `570ea6bd91f14ef9` | `4c4dd3c76af01656` | 2.9321 | 0.0742 | 16.67 | 12 | 10.80 | 0.9719 |
| L-s7 | L | `66e5f85fe6b8e186` | `258449969d262282` | `8273e25762a45447` | 2.9321 | 0.1046 | 8.33 | 24 | 10.41 | 0.9677 |
| S-s17 | S | `f7d03e205f378179` | `8ae9a830cc8fff4c` | `399684c9d4f07e7b` | 2.9321 | 0.0731 | 16.67 | 12 | 10.17 | 0.9708 |
| L-s17 | L | `f7d03e205f378179` | `476f8bac96d37f42` | `7f6a4080af458b84` | 2.9321 | 0.1030 | 8.33 | 24 | 9.96 | 0.9688 |
| S-s27 | S | `276573d10853f3e7` | `81250392c9eba45a` | `4153be1471efc32c` | 2.9321 | 0.0724 | 16.67 | 12 | 10.51 | 0.9729 |
| L-s27 | L | `276573d10853f3e7` | `69b454bfd5dc44f0` | `58b76828384cb96c` | 2.9321 | 0.1051 | 8.33 | 24 | 9.96 | 0.9677 |

- 同 seed 两臂初始 head SHA **逐位相同**；A 臂三个 `loss@0` 都是 **2.9321**（= R20/R21/R22/R23/R24 的 control 初值）。
- 末 loss 都在 0.07–0.11，离线 masked 准确率 0.968–0.973：两臂都在**已见**事件上学会了教师的首动作，L 只是略低（epoch 少一半）。
- 冻结骨干训练后逐位不变（工具内 `torch.equal` 硬断言）。

## 评估

### 1) 留出侧校验先行

R18 的六条基线在两个评估集上重跑：`r19_frozen` 6 例、`new_heldout` 14 例，**全部 0 条 feasible**（`filter_valid = true`、`passing_events = []`）。冻结 B0 在两个集上 feasible 都是 **0**。

### 2) 可行性 / first hit（两个 held-out 集）

| 集 | 事件 | B0 | S-s7 | S-s17 | S-s27 | L-s7 | L-s17 | L-s27 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `r19_frozen`（跨集对照） | 6 | 0 | 1 | 1 | 1 | **0** | **0** | **0** |
| `new_heldout`（新） | 14 | 0 | 4 | 3 | 3 | 3 | 3 | 2 |

- 新 held-out 上 L 合计 8、S 合计 10：**规模翻倍没有提高可行性**（反而少 2 格，n 很小，不作显著性声明）。
- R19 冻结留出上 L **0/6**（从未训练这 3 组），S **1/6**（训练过，`010-m0` 停进缺口）；R24 A 在同一 6 例上是 **2/6**。

### 3) P1（主判定）：need_left 格上的 flip

| 网络 | seed 7 | seed 17 | seed 27 | 合计 / 21 | 镜像子集（2 格/seed） |
|---|---:|---:|---:|---:|---:|
| **L（大训练集）** | **0** | **0** | **0** | **0** | **0 / 6** |
| S（旧 6 组） | 0 | 0 | 0 | 0 | 0 / 6 |
| B0（冻结基线，只评估一次） | 0 | 0 | 0 | 0 / 7 | 0 / 2 |

**7 个 need_left 格（5 base + 2 镜像）在三个 seed 上、两臂、B0 全部走屏幕向右**。逐事件：`003-m0`、`008-m1`、`015-m1`、`020-m0`、`028-m0`、`040-m0`、`046-m0`——两臂都 `away`/`hold`，没有一次向左。**P1 = `scale_not_the_cause`。**

### 4) P2：need_right 格上的 keep

| 网络 | seed 7 | seed 17 | seed 27 | keep 率 |
|---|---:|---:|---:|---:|
| **L** | **7/7** | **7/7** | **7/7** | **1.000** |
| S | 7/7 | 7/7 | 7/7 | 1.000 |

**P2 = `base_kept`**：只要缺口在屏幕右侧，两臂都继续向右；L 的 8 个 `in_gap_at_wall` 格（3 base + 5 镜像）与 S 的 10 个（3 base + 7 镜像）**全部落在 need_right 一侧**，没有一格靠"向左"停进去的（见下表）。

### 5) 原始 base/镜像计数（几何混杂，只作对照）

`new_heldout`（每格 = 网络 × 14 事件；S/L 是 3 seed 合计 42 格）：

| 臂 | 组 | 格数 | toward / hold / away | toward 率 | 停进缺口 | 平均 \|误差\| px | 最大 \|误差\| | 屏幕 右/左/hold |
|---|---|---:|---|---:|---:|---:|---:|---|
| B0 | base | 7 | 2 / 0 / 5 | 0.286 | 0 | 142.29 | 192 | 7 / 0 / 0 |
| B0 | 镜像 | 7 | **5** / 0 / 2 | 0.714 | 0 | 122.86 | 168 | 7 / 0 / 0 |
| S | base | 21 | 6 / 1 / 14 | 0.286 | 3 | 63.81 | 116 | 20 / 0 / 1 |
| S | 镜像 | 21 | **15** / 0 / 6 | 0.714 | 7 | 32.38 | 112 | 21 / 0 / 0 |
| L | base | 21 | 6 / 2 / 13 | 0.286 | 3 | 58.48 | 116 | 19 / 0 / 2 |
| L | 镜像 | 21 | **15** / 0 / 6 | 0.714 | 5 | 30.86 | 112 | 21 / 0 / 0 |

**S、L 的 toward 计数与 B0 完全相同**（base 2/7、镜像 5/7 的比率），这就是"原始镜像率不能当 flip 率"的直接证据。屏幕向右占比：B0 **100%**、S **97.6%**（41/42）、L **95.2%**（40/42）；两个臂在 `new_heldout` 上一次屏幕向左都没有（S 是 41 右 + 1 hold，L 是 40 右 + 2 hold）。

`r19_frozen`（跨集对照，每臂 9 格）：

| 臂 | 组 | toward / hold / away | 停进缺口 | 平均 \|误差\| px | 屏幕 右/左/hold |
|---|---|---|---:|---:|---|
| B0 | base | 3 / 0 / 0 | 0 | 96.00 | 3 / 0 / 0 |
| B0 | 镜像 | 0 / 0 / 3 | 0 | 166.67 | 3 / 0 / 0 |
| S | base | 6 / 0 / 3 | 3 | 14.67 | 6 / 3 / 0 |
| S | 镜像 | 1 / 2 / 6 | 0 | 52.44 | 6 / 1 / 2 |
| L | base | 6 / 3 / 0 | 0 | 18.67 | 6 / 0 / 3 |
| L | 镜像 | **0 / 0 / 9** | 0 | 74.67 | 9 / 0 / 0 |

- **L 在冻结留出上的方向条件化读数**：need_left 3 格（= 3 个镜像事件）flip **0**；need_right 3 格 keep **2/3**；所以 L 不是"读对了没用"，而是**根本没向左**。
- 对照 R24 A：镜像 toward **3/9**（同一量 = flip 3/9）、base toward 6/9、feasible 2/6。L 的镜像 flip 0/9、feasible 0/6：**在这条旧标尺上，扩大数据没有带来任何改善**（注意 L 没训练过这 3 组，而 R24 A 训练过另外 3 个 F1 组 + F2/F3，两组数据与 epoch 都不同，只能作描述性对照）。

### 6) 逐事件：新 held-out

`verdict/signed_error_px`（signed error：0 = 停进缺口，负 = 短于左边缘，正 = 越过右边缘）。`need` 是缺口所需的**屏幕**方向：

| 事件 | 镜像 | need | 起点 | 缺口 | B0 | S-s7 | S-s17 | S-s27 | L-s7 | L-s17 | L-s27 | firsthit S / L |
|---|---|---|---|---:|---|---|---|---|---|---|---|---|---|
| `wf25-F1-003-m0` | base | **left** | 182 | (128,142) | away/172 | hold/40 | away/56 | away/56 | hold/40 | hold/40 | away/56 | 44 / 44 |
| `wf25-F1-003-m1` | 镜像 | right | 182 | (128,142) | toward/−118 | toward/0 | toward/8 | toward/8 | toward/0 | toward/0 | toward/8 | None / None |
| `wf25-F1-008-m0` | base | right | 24 | (120,140) | toward/120 | toward/−48 | toward/−48 | toward/−48 | toward/−32 | toward/−32 | toward/−32 | 60 / 60 |
| `wf25-F1-008-m1` | 镜像 | **left** | 24 | (120,140) | away/−112 | away/−112 | away/−112 | away/−112 | away/−112 | away/−112 | away/−112 | 60 / 60 |
| `wf25-F1-015-m0` | base | right | 152 | (176,196) | toward/120 | toward/0 | toward/0 | toward/0 | toward/0 | toward/0 | toward/0 | None / None |
| `wf25-F1-015-m1` | 镜像 | **left** | 152 | (176,196) | away/−168 | away/−40 | away/−48 | away/−40 | away/−32 | away/−32 | away/−32 | 52 / 52 |
| `wf25-F1-020-m0` | base | **left** | 336 | (272,288) | away/88 | away/88 | away/88 | away/88 | away/88 | away/88 | away/88 | 64 / 64 |
| `wf25-F1-020-m1` | 镜像 | right | 336 | (272,288) | toward/−164 | toward/0 | toward/0 | toward/0 | toward/0 | toward/0 | toward/0 | None / None |
| `wf25-F1-028-m0` | base | **left** | 224 | (168,184) | away/192 | away/56 | away/56 | away/56 | away/56 | away/56 | away/56 | 52 / 52 |
| `wf25-F1-028-m1` | 镜像 | right | 224 | (168,184) | toward/−36 | toward/24 | toward/16 | toward/16 | toward/32 | toward/24 | toward/32 | 52 / 52 |
| `wf25-F1-040-m0` | base | **left** | 356 | (248,260) | away/116 | away/116 | away/116 | away/116 | away/116 | away/116 | away/116 | 60 / 60 |
| `wf25-F1-040-m1` | 镜像 | right | 356 | (248,260) | toward/−128 | toward/48 | toward/48 | toward/48 | toward/32 | toward/32 | toward/32 | 60 / 60 |
| `wf25-F1-046-m0` | base | **left** | 206 | (152,166) | away/188 | away/88 | away/88 | away/88 | away/72 | away/72 | away/72 | 48 / 48 |
| `wf25-F1-046-m1` | 镜像 | right | 206 | (152,166) | toward/−134 | toward/0 | toward/0 | toward/0 | toward/8 | toward/8 | toward/8 | None / 48 |

读法：

1. **7 个 need_left 事件全部向右**（`003-m0`、`008-m1`、`015-m1`、`020-m0`、`028-m0`、`040-m0`、`046-m0`）；两臂与 B0 同形，两臂的 |误差| 是 32–116 px，B0 是 88–192 px。
2. **need_right 的 7 个事件两臂都向右**，其中 `015-m0`（两臂 0 px）、`020-m1`（两臂 0 px）、`003-m1`（S 0/8/8、L 0/0/8）停在缺口内或只差 8 px，`046-m1`（S 0、L 8）也在缺口内或只差 8 px；`008-m0` 向右但幅度不足（−32~−48 px）；`028-m1` 越过 16–32 px；`040-m1` 越过 32–48 px。
3. **L 与 S 的区别只在幅度与时机，不在方向**：同事件同 seed 的 `firsthit` 大多相同（003-m0 都是 44、008 都是 60、020 都是 64、028 都是 52、040 都是 60、046-m0 都是 48）；`046-m1` 上 S 满 horizon 存活而 L 在 48 帧撞死，`003-m1` 上 L 三个 seed 只死一个、S 死两个。

### 7) 逐事件 first hit（新 held-out，`None` = 满 horizon 未撞）

| 事件 | B0 | S-s7 | S-s17 | S-s27 | L-s7 | L-s17 | L-s27 |
|---|---|---|---|---|---|---|---|
| `wf25-F1-003-m0` | 44 | 44 | 44 | 44 | 44 | 44 | 44 |
| `wf25-F1-003-m1` | 44 | None | 44 | 44 | None | None | 44 |
| `wf25-F1-008-m0` | 60 | 60 | 60 | 60 | 60 | 60 | 60 |
| `wf25-F1-008-m1` | 60 | 60 | 60 | 60 | 60 | 60 | 60 |
| `wf25-F1-015-m0` | 52 | None | None | None | None | None | None |
| `wf25-F1-015-m1` | 52 | 52 | 52 | 52 | 52 | 52 | 52 |
| `wf25-F1-020-m0` | 64 | 64 | 64 | 64 | 64 | 64 | 64 |
| `wf25-F1-020-m1` | 64 | None | None | None | None | None | None |
| `wf25-F1-028-m0` | 52 | 52 | 52 | 52 | 52 | 52 | 52 |
| `wf25-F1-028-m1` | 52 | 52 | 52 | 52 | 52 | 52 | 52 |
| `wf25-F1-040-m0` | 60 | 60 | 60 | 60 | 60 | 60 | 60 |
| `wf25-F1-040-m1` | 60 | 60 | 60 | 60 | 60 | 60 | 60 |
| `wf25-F1-046-m0` | 48 | 48 | 48 | 48 | 48 | 48 | 48 |
| `wf25-F1-046-m1` | 48 | None | None | None | 48 | 48 | 48 |

所有死亡仍然发生在 `wall_frame` 那一帧（F1 的致命窗口起点），与 R21–R24 一致；`015-m0` 与 `020-m1` 上两臂三 seed 全部满 horizon 存活，`046-m1` 上 S 三个 seed 满 horizon、L 三个 seed 都在 48 帧撞死。

## 判定（有界，不做显著性声明）

1. **P1（主判定）= `scale_not_the_cause`**：新 held-out 的 7 个 need_left 格，L 三个 seed **flip 0/0/0**（0/21），S 与 B0 也是 0；镜像子集 0/6。**把 F1 从 6 组扩到 12 组、把训练事件从 12 扩到 24，没有产生一次"缺口在左就走左"。**
2. **P2 = `base_kept`**：need_right 的 7 格 keep **7/7/7**（21/21），且 L 与 S 在这一读数上完全相同。base 的"向右先验"没有被扩大数据破坏——它也没有被改善。
3. **原始"镜像走对率"在新集上是几何**：B0 自己 5/7，S 与 L 的原始计数与 B0 逐格同率（base 6/21、镜像 15/21）。**任何"镜像从 0/9 升到 15/21"的读法都是错的**；条件化后的镜像 flip 是 0/6。
4. **R19 冻结留出（跨集对照）**：L **0/6 feasible、镜像 flip 0/9**、keep 2/3；S 1/6（已见读数）；R24 A 2/6、镜像 toward 3/9。**在这条旧标尺上，扩大数据没有带来改善**；因为 L 未训练这 3 组而 R24 A 训练过另外的 F1 组，这个对比只能描述、不能归因。
5. **规模不是主因，至少在这个尺度与预算上**：两臂、三个 seed、21 个 need_left 格、0 次向左；L 的可行性与方向计数都没有超过 S。反过来，"200 updates 对 24 个事件只有 8.3 个 epoch"这个预算混淆无法排除，所以本轮能说的是**"把世界数翻倍不是解"**，不能说的是"任何规模都不会是解"。
6. **不宣称战术理解**，也不声称"数据无用"：本轮只测了「F1 only + control loss + S 6 组 vs L 12 组 + 200 updates」这一个组合。

## 为什么（机制，逐条）

1. **方向先验是冻结骨干的，不是训练集的**：两臂的初始 head 相同，末 head 不同（6 个 SHA 互异），离线准确率都到 0.97，但在需要向左的格上两臂都输出屏幕向右。数据规模改变的是"在哪些世界上拟合教师的首动作"，没有改变"首动作在屏幕坐标里的方向分布"——训练集里绝大多数教师首动作仍是向右/停。
2. **新 held-out 的 need_left 格里 5 个是 base**：这与 R19 冻结留出（need_left 全是镜像）不同。一个只看"镜像/非镜像"的模型在这里会得到相反的归纳偏置；两臂都没有利用这个结构。
3. **L 的 12 个训练组里有 6 个 base-need-left 组**（旧 `006`，新 `007/019/033/048/049`），也就是说"缺口在起点左侧的 base 世界"在 L 的训练集里占一半，但 L 在留出的 base-need-left 上仍然全部向右：**覆盖面增加并不等于学会了条件方向**。
4. **可行性没有提高的原因**：F1 的可行要求"在 wall 前停进 12–20 px 的槽"。L 的 8 个 `in_gap_at_wall` 格（base 3 + 镜像 5，21 格网络合计）**全部落在 need_right 格上**；need_left 格上方向就错，幅度再准也没用；S 与 L 的死亡帧都与 B0 同帧，说明失败仍是**方向**而不是时机。
5. **原始计数与 B0 同率**说明：在这批世界上，两个学到的策略在"方向"这一层与冻结基线不可区分；它们与 B0 的差别只在幅度（平均 |误差| 从 122–142 px 降到 31–64 px）与少数停进缺口。

## 限制（必须与结论一起读）

1. **S 臂训练过 R19 冻结留出**（`005/007/010`），它在 `r19_frozen` 上的读数是已见读数；本轮所有主结论只使用 `new_heldout`，L 臂在两个评估集上都干净。
2. **两臂不嵌套**：S = 旧 6 组，L = 3 个旧 train 组 + 9 个新组，规模与组成同时变化。两臂 flip 都是 0，null 对组成差异稳健；但"纯规模"的因果不能从这一轮单独建立。
3. **n = 3 seeds、7 个新 held-out 组**；need_left 只有 7 格（5 base + 2 镜像），镜像子集只有 2 格；镜像不是独立样本，seed 也不是独立世界。
4. **200 updates 固定**：S 16.7 epoch、L 8.3 epoch。L 的 flip=0 不能区分"规模不够"与"预算不够"；S 同样 0，而且 S 的训练集里本来就有 5 个镜像 need-left 事件（`002/005/007/009/010` 的 m1）和 1 个 base need-left（`006-m0`），它仍然 0——所以两臂的 0 都不能用"没见过向左"解释。
5. **指标是条件化后的**：原始"镜像 toward"在新集上被几何主导（B0 5/7），所以本轮的主判定用 need_left/need_right；这个修正在正式预注册之前完成，但它是被 scratch 冒烟跑暴露的，必须与"冒烟数值未用于证据"一起读。
6. **F1 的胜利条件是 12–20 px 的精确停车**，是几何构造，不是"会玩"；`010-m0` 上 L 只差 4 px 就死，说明幅度余量极小。
7. **教师不是学出来的策略**：`TacticalTeacherV2` 被告知世界几何；标签是它的首动作。
8. **合成固定 y 世界**：两帧动作、无射击/炸弹、无真实 TH08 像素与判定、horizon 固定 160。
9. **只有一种超参配置**（200 updates / lr 1e-3 / clip 1 / batch 1）、一个损失（control，无 KL、无辅助头）、一次运行；没有调参、没有正式门、没有可部署产物。
10. **scratch 冒烟跑**：正式预注册前在 `.tmp/f1scale-smoke` 跑过 7 次完整流程（前 5 次各暴露一个本轮工具/测试的真实缺陷，第 6 次跑通但暴露了原始"镜像 toward"的几何混杂，第 7 次用改定后的指标成功），其任何数值都没有进入证据；正式 run 与第 7 次冒烟的 6 个 checkpoint SHA **逐位相同**，因此两者是同一个计算。逐条见 [command-results.md](<../evidence/f1scale-round25/command-results.md>)。

## 对既有结论的处置

| 既有结论 | 处置 |
|---|---|
| R24："辅助任务把策略推成屏幕向右（18 右/0 左/0 hold）" | **确认并扩展到无辅助头的 A 臂**：新 held-out 上 L 40/42 右、0 左（95.2%），S 41/42 右；需要向左的 21 个格上 0 次向左 |
| R24："F1 留出 base 走对 9/9" | **必须按方向条件化读**：R19 冻结留出的 3 个 base 缺口全在右侧，所以 9/9 是 keep；在新的混合 held-out 上，base 的原始 toward 只剩 6/21，而 keep 仍是 21/21。原始 base 数字是数据集性质，不是策略能力 |
| R24："镜像走对 3/9 → 0/9" | **在条件化读数上确认**：新 held-out 的镜像 need_left 子集 flip **0/6**；原始镜像 toward 15/21 与 B0 的 15/21 相同，是几何 |
| R24："辅助头在留出上退化 / 8/9 读对却没用" | 本轮不训练头，不涉及；但"514 参数的头退化"与本轮的"策略方向不随数据规模变"是两条独立证据，都指向**不是数据量** |
| R24 后续建议 "(c) 类别与事件解耦（跨事件平衡的 batch / 按组留一）" | **本轮先排除了"只加世界"这条路**：12 组/24 事件、同一 200 步预算下没有出现任何条件向左。下一步应把**结构与耦合**作为单因子，而不是继续加世界 |
| "数据规模是镜像失败的主因"（本轮假设） | **不支持**：P1 全 0、P2 保持、可行性未提高；只能确认"翻倍世界数不是解" |
| R18 的 6 组 F1 世界 | **不改**：16 个新组与 59 个冻结签名零重复，旧世界的参数、判定、动作一个都没有改 |

本轮不修改任何旧台账；建议父级在 round25 台账登记：**F1 数据规模 6 组 → 12 组（12 → 24 事件）在 200 updates 下没有产生任何一次条件向左（need_left flip 0/21，镜像 0/6），判定 `scale_not_the_cause`；base 的屏幕向右先验保持（keep 21/21）；新 held-out 的原始镜像 toward 15/21 与冻结基线 B0 同率，是几何混杂；R19 冻结留出上 L 0/6 feasible、flip 0/9，未优于 R24 A 的 2/6、3/9。**

## 执行与退出码

```powershell
# 1) 单测（写工具期间共 5 轮，逐条归因见 command-results.md）
python -m pytest tests/test_v373_f1_scale.py -q -p no:cacheprovider --basetemp .tmp/v373-f1scale-tests-final-pre
#    31 passed in 3.51s，exit 0
# 2) 正式冻结预注册（判据 P1/P1-mirror/P2/P3 + 36 个身份 SHA + 搜索规格 + 划分规则 + 期望计数）
python -m tools.rl.v373_f1_scale preregister
#    exit 0；criterion SHA 5ad3ea35a02ab4aa86d673bb908fc9f0826b4a8719dc8ab49e5479cfa5a065d8
#    preregistration.json SHA d450dd4d6694f6077a4f699ef73b2a53b1169ec599e72345ad71c8315fd25b89
# 3) 唯一一次正式 run（后台 job，自行收取）：
python -m tools.rl.v373_f1_scale run
#    exit 0；墙钟 ≈ 235 s（search 2.403 / data 36.563 / encode 2.558 / training 65.799 / closed loop 122.986）
# 4) 只读复核
python -m tools.rl.v373_f1_scale verify
#    exit 0；180 条 run SHA（run/ 共 181 个文件）/ 182 条根 SHA / 120 资产绑定 / 6 checkpoint /
#    14 闭环文件 / 140 个 F1 格
# 5) run 之后复跑测试
python -m pytest tests/test_v373_f1_scale.py -q -p no:cacheprovider --basetemp .tmp/v373-f1scale-tests-after-run
#    31 passed，exit 0
# 6) 相邻套件回归（R18 生成器/判据）
python -m pytest tests/test_v373_world_filter.py -q -p no:cacheprovider --basetemp .tmp/v373-f1scale-regression-r18
#    15 passed，exit 0
# 7) 写入日志后再复核（日志在 run 之后写入，故不在根 sha256 表内）
python -m tools.rl.v373_f1_scale verify
#    exit 0
```

退出码依次 **0 / 0 / 0 / 0 / 0 / 0 / 0**（正式 preregister + 正式 run + 两次正式 verify + run 前后各一次测试 + R18 回归）。另有 **7 次 scratch 冒烟**（`.tmp/f1scale-smoke`，前 5 次各暴露一个本轮缺陷，第 6 次跑通但暴露了原始指标的几何混杂，第 7 次用改定后的指标成功），不计为完成、不产生证据；其 6 个 checkpoint SHA 与正式 run **逐位相同**。命令、日志与测试迭代都在 [command-results.md](<../evidence/f1scale-round25/command-results.md>)。

## 证据文件

- [preregistration.json](<../evidence/f1scale-round25/preregistration.json>)：判据（**P1/P1-mirror/P2/P3 + 几何混杂说明**，SHA `5ad3ea35…`）、**36 个身份 SHA**、搜索规格（seed 25001 / 预算 400 / 目标 16 / R18 F1 域原样）、划分规则与下限、两臂定义、训练超参、评估协议、初始化 per-seed SHA、冻结签名普查 59、R24 参照记录、`expected` 计数。文件 SHA `d450dd4d…`。
- [run/search.json](<../evidence/f1scale-round25/run/search.json>)：**50 个候选**的逐条结果（27 invalid / 7 unknown / 0 重复 / 0 不可用）、16 个接受组的完整 R18 判据证据（六基线 feasible/survived/first_hit + 完整 80 动作基线表 + 独立 `solve` 见证动作 + 六常量诊断）、冻结普查、世界 id 与 scenario。
- [run/split.json](<../evidence/f1scale-round25/run/split.json>)：22 组的 split、组内两事件的 scenario/`scenario_sha256`/start、`counts`/`floors`/`flips`/`shortfall`、`train_events`/`heldout_events`/`new_heldout_events`/`r19_frozen_events`。
- [run/records.json](<../evidence/f1scale-round25/run/records.json>)：30 个渲染事件的 120 个资产绑定（path + SHA + 字节 + shapes）、`teacher_verified = true` 30/30、每事件的 `render_freeze` 与 `containment`（6 个旧 train 事件与 R21 逐字节相同、12 个旧组事件与 R19 教师动作相同）、`totals`（3 511 046 B、103 134 690 changed pixels）。
- `run/assets/<event_id>/{pixels.npz,history.pt,labels.pt,rollout.json}`：30 个事件的 R21 telegraph 渲染资产（F1 训练事件 `pixels.npz` 约 95 KB，全部 `xb` 写入，SHA 在 records 里逐个绑定）。
- `run/train/<S|L>/<S|L>-s<seed>/{initialization.json,training.json,research-heads.pt}`：**6 个 checkpoint** 的初始/末 head SHA、逐 step loss、epoch、活跃参数、离线指标。
- [run/checkpoints.json](<../evidence/f1scale-round25/run/checkpoints.json>)：6 个 checkpoint 的 SHA 与数据集描述。
- [run/baselines.json](<../evidence/f1scale-round25/run/baselines.json>)：两个评估集的 R18 六基线（6 × 20 行），`filter_valid = true`、`passing_events = []`。
- `run/closed-loop/<B0|S-s*|L-s*>-<r19_frozen|new_heldout>.json`：**14 个文件**，每文件 6 或 14 个事件 × 80 决策（`actual_action`/`x_before`/`x_after`/`hit`/`first_collision_frame`）+ 分族聚合 + `behaviour` + `heads_sha256`。
- [run/f1-gap.json](<../evidence/f1scale-round25/run/f1-gap.json>)：**140 行 F1 机制诊断**（7 网络 × 20 个 held-out 事件）：起点、缺口、窗口、方向、首次移动帧、`in_gap_at_wall`、`signed_error_px`、机制分类、`toward/hold/away`、`screen_direction`。
- `run/events/<event_id>.json`：**20 个逐事件 JSON**（两个评估集），含六基线、7 个网络的闭环读数（feasible/survived/firsthit/decisions/switches）、每格 F1 诊断、该事件被哪些臂训练过。
- [run/summary.json](<../evidence/f1scale-round25/run/summary.json>)：本轮汇总（`search`/`split`/`data`/`initialization`/`checkpoints`/两个 `sets`（含 `conditioned` 与原始 `direction`）/`judgement`（P1/P1-mirror/P2/P3 + pooled + raw_mirror_toward）/`round24_reference`/`timings`/`environment`/`flags`/`limitations`）。
- [run/sha256.json](<../evidence/f1scale-round25/run/sha256.json>)（180 条，不含自身；run/ 共 181 个文件）与 [sha256.json](<../evidence/f1scale-round25/sha256.json>)（根目录 182 个文件）；`verify` 逐个复核，0 处不符。
- 命令与日志：[command-results.md](<../evidence/f1scale-round25/command-results.md>)、`logs/{preregister,execution,verify,tests-before-prereg,tests-after-run,regression-round18,verify-after-logs}.log`（在 run 之后写入，不在根 `sha256.json` 表内；写入后重跑 `verify` 仍 exit 0）。
