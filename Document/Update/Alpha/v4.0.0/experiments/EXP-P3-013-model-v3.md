# EXP-P3-013：学生 v3（IMPALA-ResNet + 保留空间）架构、训练器与单测

- 日期 / 工作包：2026-11-10 / T4（[P3 重启任务书 §4 T4](../tasks/2026-11-10-P3重启任务书.md)）
- 对应规格：[13-P3 §2.3–2.4](../13-P3-像素学生.md)、[obs-spec-v3](../obs-spec-v3.md)（D-14 冻结）
- 状态：**架构 + 训练器 + 单测 + 前向性能已完成；训练段留空**（D1 仍在采集，本轮按任务书"本轮不训练"执行）
- 代码：`touhou_rl/v400/student/model_v3.py`、`touhou_rl/v400/student/train_v3.py`、
  `tools/rl/v400_student_train_teacher.py`（新增 `--obs v3` / `--arch v3`）、
  `tools/rl/v400_student_eval.py`（新增 `--obs v3` / `--arch v3`）、
  `tests/test_v400_model_v3.py`；父提交 `77025ae`（本卡与代码同一个 T4 提交，
  `git log -1 --stat` 即该提交）

## 预注册（训练运行之前写；本卡的训练段在运行后才填）

- 假设：把场地流的**全局平均池化**换成**保留空间**的读出头，同时把输入换成 obs-v3
  （96×112 + 8×18 act_hist）、把容量从 20 万参数提到 2–6 M，学生的 S1N 开发集成绩会比
  `m1`（EXP-P3-009：2,248.85 帧）高。
- 主指标：`S1N`（`--stage 0 --difficulty normal`，cap 19887，`--obs-window consecutive`）
  20 个开发种子 1000–1019 的每局存活帧数，取均值与 95% bootstrap 区间
  （`tools/rl/v400_student_eval.py` 的 `frames_stats`）。
- 判据：按 EXP-P3-009 的比较规则，**必须**用 `tools/rl/v400_eval_power.py claim` 判定；
  每臂 ≥3 个独立训练 checkpoint（训练种子 `--seed 0/1/2`），每个 checkpoint 跑满
  20 个开发种子；两臂区间不重叠且差值超过 checkpoint 分辨极限才算"成立"。没有 CI 或
  区间重叠 → 一律写"无法判定"。
- 样本量：3 个 checkpoint × 20 局 = 60 局/臂；对照臂 = EXP-P3-009 的 `m1` 记录。
- 停止条件：3 个 checkpoint 训完后按上述规则判定一次；不做中途改判据。
- 不在本轮执行：上面这段**全部**属于训练段（D1 就绪后由主代理启动）。本卡本轮只填
  架构、参数、前向性能、单测、checkpoint 兼容与已知缺失；训练段的表格留空。
- 前向性能（本卡唯一含数字的部分）**不是**假设检验：它是固定协议的测量（batch=32、
  5 次预热、CPU 10 次 / GPU 30 次 / 训练步 20 次，取中位数），协议写死在
  `.tmp_v400_p3v3/bench_v3.py`，可原样重跑。

## 1. 架构

```text
play (B,4,112,96)                                            # obs-v3 场地平面
  └─ IMPALA-ResNet encoder
       stage1: conv3x3(4→32)  + BN + ReLU → maxpool2 → ResBlock(32) ×2   → (B,32,56,48)
       stage2: conv3x3(32→64) + BN + ReLU → maxpool2 → ResBlock(64) ×2   → (B,64,28,24)
       stage3: conv3x3(64→64) + BN + ReLU → maxpool2 → ResBlock(64) ×2   → (B,64,14,12)
       —— 到此为止没有全局池化，输出仍是 14×12 的空间图 ——
  └─ spatial readout: conv1x1(64→16) + BN + ReLU → flatten(16·14·12 = 2688) → Linear 512 + ReLU

hud (B,1,56,160)                                             # obs-v1 同一块 HUD 绿通道
  └─ conv5x5(1→16) s2 → conv3x3(16→32) s2 → conv3x3(32→32) s2 → (B,32,7,20)
     → flatten(4480) → Linear 96 + ReLU                       # 独立小卷积，同样不池化

act_hist (B,8,18) → flatten(144) → Linear 64 + ReLU           # obs-v2 的 8 个历史决策

trunk: concat(512 + 96 + 64 = 672) → Linear 512 + ReLU
heads: move_logits[18] · bomb_logits[2] · value[1]
perception（只在训练时算，部署 `aux=False` 不付这份算力）
  (B,64,14,12) → nearest×2 + conv3x3(64→32)+BN+ReLU → (B,32,28,24)
               → nearest×2 + conv3x3(32→16)+BN+ReLU → (B,16,56,48)
               → nearest×2 + conv3x3(16→16)+BN+ReLU → (B,16,112,96)
               → 1×1 头：occupancy(1) · self_heatmap(1) · danger(3, horizons 4/8/16)
```

- 辅助标签网格 `(112,96)` **恰好等于** obs-v3 的场地平面，也等于
  `touhou_rl/v400/labels/spec.py` 的 `GRID_HEIGHT=112 / GRID_WIDTH=96`
  （448×384 的 1/4），三者是同一个 4×4 整数块均值口径——辅助头输出直接和标签对齐，
  不需要重采样。
- 没有 GRU：13-P3 §2.3 把它列为可选项（WP6 消融），本轮不引入第二个自由变量；
  `act_hist` 已经承担 8 个历史决策的记忆。

### 1.1 "保留空间"选了哪一种，为什么

二选一里**选了"展平空间特征图"**（1×1 通道压缩 + flatten + Linear），没有选
"空间 softmax 关键点 + 坐标通道"。理由：

1. **稠密辅助头本来就需要这张图。** 占用图 / 自机热图 / 危险图是 112×96 的逐格标签，
   必须从空间特征图上读。展平让它成为策略和感知头**共用的同一张图**；关键点做法只能
   留下每个通道的一个期望坐标，感知头得再挂一套读出，等于同一份信息两套表示。
2. **多目标状态。** 空间 softmax 把每个通道压成一个期望点：左右各一颗子弹和两颗都
   靠左可能给出相近的坐标。躲弹要同时看多个威胁，展平读出的每个权重绑定一个
   `(行, 列, 通道)` 单元，可以独立表达。
3. **它是对假设更强的检验。** 重启的假设是"位置信息丢了"（任务书 §0.2 原因 2）。
   展平读出的每个权重只对应一个格子，学不到位置就是**可证明地**学不到；无监督关键点
   坐标是一个平移不变的摘要，会把效应糊掉（EXP-P3-011 要量化的正是这一项）。
4. **参数预算。** `2688 → 512` 是 1.38 M 参数，加上其余部分正好落进预注册的
   2–6 M 带（实测 2,602,490），关键点做法会让模型停在约 0.4 M，要靠无关的地方补容量
   才够得着带内下限。

单测把这一条做成了可执行断言：把同一个亮块从场地左侧平移到右侧，
**全局平均池化的摘要余弦相似度 0.9993**（几乎相同），而展平读出头的输出相对差
**27.6 %**（`test_spatial_readout_separates_what_average_pooling_cannot`）。

### 1.2 参数表（实测，`torch.manual_seed(0)` 构建）

| 模块 | 参数 | 占比 |
|---|---:|---:|
| 场地流 IMPALA-ResNet（3 stage，32/64/64，每 stage 2 个残差块） | 389,824 | 15.0 % |
| 空间读出（1×1 64→16 + flatten 2688→512） | 1,377,840 | 52.9 % |
| HUD 小卷积（1→16→32→32，flatten 4480→96） | 444,640 | 17.1 % |
| 动作历史（144→64） | 9,280 | 0.4 % |
| trunk（672→512） | 344,576 | 13.2 % |
| 移动头 18 类 / bomb 头 2 类 / 价值头 | 9,234 / 1,026 / 513 | 0.4 % |
| 感知辅助头（3 次上采样 + 3 个 1×1 头） | 25,557 | 1.0 % |
| **合计** | **2,602,490** | 100 % |

- 预注册区间 2–6 M：`model_v3.PARAMETER_BUDGET = (2_000_000, 6_000_000)`，构造函数对
  **规范架构**直接断言（`check_parameter_budget`），越界抛
  `ParameterBudgetError`；单测同时断言 2,602,490 这个具体值，任何结构改动都会先打断测试。
- 对照：`m1` 是 204,260 参数（`model.py` V0 冒烟），本架构是它的 **12.7 倍**。

## 2. 前向性能（实测，batch = 32）

机器：NVIDIA GeForce RTX 4070 SUPER，torch 2.14.1+cu132，CPU 线程 14，Windows 侧；
同机 WSL 另有 4–8 个 D1 probe 引擎在跑，CPU 数字受此影响（GPU 侧不受）。
命令（可原样重跑）：

```powershell
$env:PYTHONPATH=(Get-Location).Path
python .tmp_v400_p3v3/bench_v3.py        # 写 .tmp_v400_p3v3/bench.json
```

| 项 | 第 1 次 | 第 2 次 | 样本/秒（中位） |
|---|---:|---:|---:|
| CPU 前向（部署路径，`aux=False`）中位 | 54.959 ms | 56.499 ms | ≈ 566–582 |
| GPU 前向（部署路径）中位 | 2.258 ms | 2.241 ms | ≈ 14,172–14,279 |
| GPU 训练步（`aux=True` 前向 + 反向 + AdamW）中位 | 14.776 ms | 14.245 ms | **2,166–2,246** |

- 波动范围：CPU 53.5–63.1 ms，GPU 前向 2.19–3.01 ms，GPU 训练步 12.96–17.76 ms。
- GPU 训练峰值显存：**565 MB**（batch 32，含感知头）。
- 按 2,166–2,246 样本/秒投影：50 万决策一个 epoch ≈ **223–231 秒**（任务书 §4 T5 的
  每轮采集规模）。这是**投影**，不是真训练吞吐；真实吞吐还要加上 memmap 取批与标签
  解析，训练段的卡片会用 `samples_per_second` 字段记录实测值。
- 部署侧含义：单决策前向 2.26 ms（GPU），远低于 2 帧决策周期（≈33 ms @60fps），
  真机评测（T6）不会成为瓶颈。

## 3. 单测

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p3v3/pytest -q tests/test_v400_model_v3.py
# 22 passed
```

`tests/test_v400_model_v3.py`：**22 项，全部通过（22 passed，2.85 s）**，覆盖任务书
要求的每一项：

| 要求 | 用例 |
|---|---|
| 输入输出形状 play `(B,4,112,96)` / hud `(B,56,160)` / act_hist `(B,8,18)` | `test_input_output_shapes`（含 hud `(B,1,56,160)` 与 act_hist `(B,144)` 两种写法）、`test_wrong_shapes_fail_closed` |
| 参数量 2M–6M | `test_parameter_count_is_inside_the_preregistered_band`（含 2,602,490 的钉值断言、越界抛错） |
| 场地编码器无 `AdaptiveAvgPool2d(1)` | `test_no_adaptive_pool_in_the_model`（**反射遍历** `model.modules()`，并断言输出仍是 `(2,64,14,12)`） |
| 保留空间真的起作用 | `test_spatial_readout_separates_what_average_pooling_cannot` |
| checkpoint 能存能读 + digest | `test_checkpoint_round_trip_and_digest`、`test_checkpoint_records_arch_and_obs_spec` |
| obs-v3 digest 不匹配 fail-closed | `test_checkpoint_fails_closed_on_another_obs_spec`（digest / version / schema / arch / parameters 五种篡改） |
| 辅助头无标签时跳过、权重归零 | `test_aux_loss_is_skipped_with_zero_weight_without_labels`、`test_student_v3_loss_effective_weights_match_the_spec` |
| 旧 checkpoint 不回归 | `test_v1_and_v2_checkpoints_still_load`、`test_the_three_checkpoint_loaders_do_not_cross` |
| 训练器端到端 | `test_train_v3_runs_end_to_end_without_aux_labels`、`test_train_v3_uses_aux_labels_when_a_shard_carries_them`、`test_train_v3_refuses_a_mis_shaped_aux_label`、`test_train_v3_refuses_a_corpus_below_the_contract` |
| 工具路由 | `test_train_tool_routes_obs_v3_without_changing_the_old_routes`、`test_eval_tool_routes_obs_v3_and_stamps_the_record` |

端到端用例用 4 个**契约合法**的合成 obs-v3 分片（经 `tools/rl/v400_d0_dual.py` 的
`assemble` 写出，因此带着真 digest、真 sha256、真 act_hist 重建），跑通
`train_v3.main`：分片 2/1/1 切分、`gate_waived=true`、`loss_weights_effective`
在无辅助标签时为 `{kl:1.0, bomb:1.0, aux:0.0, value:0.25}`、有辅助标签时为
`aux:0.5` 且 breakdown 列出三个活动头。

全量回归（任务书 §2 第 9 条的命令，`-k v400`）：

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p3v3/pytest_full -q tests touhou_rl/tests -k v400
# 1564 passed, 6 skipped, 2937 deselected in 35.69s   (exit 0)
```

**0 failed**。相邻套件（`tests/test_v400_student.py`、`test_v400_obs_v3.py`、
`test_v400_obs_v2.py`）单独跑 **78 passed**，确认两处工具改动没有动到 obs-v1/v2 的
既有行为。

## 4. checkpoint 兼容

| 证据 | 值 |
|---|---|
| 新 schema | `v400-student-checkpoint-3` |
| 记录的 `arch` | `v3` |
| 记录的 `obs_spec_version` / `obs_spec_digest` | `obs-v3` / `ef2fbd15385f26f425db2bfe855d105ee009023acb08c26916e811963553a105`（= 冻结值） |
| 记录的参数量 / 预算 / 规范架构标记 | `2,602,490` / `[2000000, 6000000]` / `true` |
| 冒烟 checkpoint（**随机初始化，不是训练结果**） | `.tmp_v400_p3v3/checkpoint-v3-smoke.pt`，10,468,383 B，SHA-256 `bc5cbaff68793d79201534a0ce77cca5fa77fe1736d7f11c6c3f2deb911ea25c` |
| 重新读取后 logits 一致 | move / bomb 逐元素一致（`allclose`），加载后为 `eval` 模式 |
| 篡改 fail-closed | digest → `ObsSpecV3Mismatch`；version → `ObsSpecV3Mismatch`；schema / `arch` / `parameters` → `ModelV3Error` |

旧格式没有回归：`model.py`（`v400-student-checkpoint-1`）与 `model_v2.py`
（`v400-student-checkpoint-2`，`arch=mlp|gru`）的 checkpoint 仍然只由各自的 loader 读写，
三个 loader 互不串读（`test_the_three_checkpoint_loaders_do_not_cross`）；
`model.py` / `model_v2.py` 两个冻结文件**没有改动**（本提交不含这两个文件）。
eval 工具对 v3 checkpoint 额外要求记录 `obs_spec_version/digest`，所以 v3 的 episode
不会和 v1/v2 的 row 混在同一份 summary 里（`validate_episode_record` 的
`expected_spec`）。

## 5. 已知缺失（写清，不当作通过）

1. **感知辅助标签还没接入。** D1 目前不产出占用图 / 自机热图 / 危险图；训练器按
   `aux/occupancy.npy`、`aux/self_heatmap.npy`、`aux/danger.npy`（每分片、第一维 = 决策数、
   网格 `(112,96)`、危险 3 通道按 4/8/16 排序）寻找，找不到就**跳过并把有效权重记为 0**。
   因此本轮训练段的 `loss_weights_effective.aux` 会是 `0.0`，感知头建好但未训练。
2. **价值头目标是代理量，不是回报。** D1 每一行只有 `verified_count` 与 `margin`，
   没有 return-to-go。默认 `--value-target verified-share`（教师验证为存活的 18 个候选
   占比），可选 `margin`（饱和到 16 px）。两者都写进 metrics 的 `value_target.known_gap`。
   规格 13-P3 §2.3 的"教师回合内的剩余存活/回报"要等 T3 落轨迹级字段。
3. **未加 GRU。** 13-P3 §2.3 的 GRU(256) 是可选项（WP6 消融），本轮为了不引入第二个
   自由变量没有做；`act_hist` 是唯一的时序输入。
4. **obs-v3 的已知限制被继承，不是本卡的缺陷。** 输入 B 的 48 px 占用 F1 = 0.8502
   （预注册口径）/ 0.8661（事后最优阈值），低于 0.90，按 D-14 以闭环为验收；这一点写在
   obs-v3 digest 里。
5. **训练段空白。** 没有训练、没有 S1N 评测、没有 `claim`，因此本卡**不构成**
   "v3 比 m1 好/差"的任何结论，也不是 G3。
6. **tau 固定 2.0。** 沿用 EXP-P3-008 的预注册值，使得 T4 与 obs-v2 两臂可比；没有重跑
   tau 扫描。

## 6. 训练结果（留空，D1 就绪后填）

| 项 | 值 |
|---|---|
| D1 规模（帧 / 决策 / 分片） | 待填 |
| 训练命令 | 待填 |
| 墙钟 / GPU 时 / 显存峰值 | 待填（架构侧实测：565 MB @ batch 32） |
| `samples_per_second`（实测） | 待填（合成 batch 投影 2,166–2,246） |
| 3 个 checkpoint 的 SHA-256 | 待填 |
| 每 checkpoint 20 开发种子的 `frames_stats` | 待填 |
| 与 `m1` 的 `v400_eval_power.py claim` 输出 | 待填 |
| `loss_weights_effective`（是否真的用上 aux） | 待填 |

## 有效性自检（V1–V9）

本卡只对**"架构 / 训练器 / 单测 / 前向性能满足 T4 的结构性验收"**作结论，
不对学生强弱作结论（训练段空白）。逐条：

| # | 检查项 | 回答与证据 |
|---|---|---|
| V1 | 输入里是否真的包含目标信息 | **是（限本卡结论范围）**。obs-v3 是 D-14 冻结的输入；位置可表征性用平移实验证（GAP 余弦 0.9993 vs 读出相对差 27.6 %）。输入是否**足够**躲避属于 T2a/F1 = 0.8502 的已知限制，已写进 §5。 |
| V2 | 标签方差 / 类平衡 / 正例数 | **未执行**——本轮不加载 D1、不训练。该检查属于训练段，卡片 §6 留空。 |
| V3 | 数值是否在物理合理范围 | **是（就本卡数字）**：参数量 2,602,490 ∈ [2 M, 6 M]；前向毫秒为正、重复 10/30 次一致；训练步 12.96–17.76 ms。没有精度类指标。 |
| V4 | 统计单位是否为独立 episode | **不适用**：本轮无 episode 级指标。前向性能的统计单位是"一次前向"，样本量（10/30/20 次）写在协议里。 |
| V5 | 平凡基线 | **已用**：结构基线就是 `m1`（204,260 参数的 V0 冒烟）与"全局平均池化"读出头，两者都在卡里给出了对照数字（12.7 倍参数；平移实验）。 |
| V6 | 训练场景与评估场景分开 | **不适用**（无训练、无评估）。训练器实现上仍强制整分片 70/15/15 切分并拒绝同 seed 跨切分。 |
| V7 | 换 seed 结论是否成立 | **部分**：性能数字两次独立运行（两套不同 seed 的合成 batch），中位数差 <3 %；单测用固定 seed，可复现。学生强弱无结论。 |
| V8 | 是否混入特权信息 | **是——输入侧为零特权**：训练器只喂 obs-v3 平面 + act_hist，`verified_mask` 只进诊断、不进 loss（`VERIFIED_MASK_USED_IN_LOSS = False`）；本卡未使用引擎。 |
| V9 | 是否用了预注册指标 | **是**：训练判据（3 ckpt × 20 种子的 `claim`）在预注册段写死，未改；性能数字是固定协议的测量，不是假设检验，协议写死在 `bench_v3.py` 里。 |

## 结论（本轮范围）

**成立**——限定为本轮范围：`model_v3.py` 实现了 13-P3 §2.3 的目标架构（IMPALA-ResNet
3 stage 32/64/64、每 stage 2 残差块、trunk 512、18+2 头、价值头、感知辅助头），
**场地编码器不含任何 `AdaptiveAvgPool2d` / `AdaptiveMaxPool2d`**（反射断言），
用展平空间特征图保留位置；参数量 **2,602,490**（2–6 M 内，构造时断言）；
`train_v3.py` 端到端可跑（合成 D1 冒烟通过，辅助标签缺失时优雅跳过且权重归零）；
`tests/test_v400_model_v3.py` **22 passed / 0 failed**；前向 GPU 2.24–2.26 ms、
CPU 55–56 ms（batch 32），GPU 训练吞吐 2,166–2,246 样本/秒、显存 565 MB。
**不含**任何"v3 学生更强"的结论——那需要 §6 的训练与 `claim`。

## 成本

- 墙钟：本轮约 0.5 小时（含两次性能测量与测试）。
- GPU 时：< 1 分钟（合成 batch，未训练）。
- 磁盘新增：约 10 MB（一个随机初始化的冒烟 checkpoint，在 `.tmp_v400_p3v3/`，不入库）。

## 证据

- `.tmp_v400_p3v3/bench.json`、`.tmp_v400_p3v3/bench_v3.py`（性能原始数据与协议）
- `.tmp_v400_p3v3/checkpoint-v3-smoke.pt`（随机初始化的架构冒烟 checkpoint）
- `.tmp_v400_p3v3/regress_full.txt`（全量 `-k v400` 回归输出）
- `.tmp_v400_p3v3/pytest/`（单测 basetemp）
- 入库的代码即证据：`touhou_rl/v400/student/model_v3.py`、`train_v3.py`、
  `tests/test_v400_model_v3.py`

## 下一步

1. D1 就绪后按 §6 跑 3 个种子（`--seed 0/1/2`），每个 checkpoint 20 开发种子，
   用 `v400_eval_power.py claim` 与 `m1` 比较；把上表填满。
2. T3 产出辅助标签后，把 `aux/*.npy` 按 §5 的约定放进分片，重跑一次训练，
   记录 `loss_weights_effective.aux` 是否变成 0.5。
3. T5 的预算用本卡的实测吞吐（2,166–2,246 样本/秒）而不是合成投影。
4. 若要 GRU 或 tau 的消融，按 WP6 单独开卡片，不改本卡的判据。
