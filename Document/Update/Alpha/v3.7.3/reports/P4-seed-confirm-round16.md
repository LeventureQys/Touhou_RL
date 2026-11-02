# P4 多 seed 复核与全新确认世界（round16）

## 结论摘要

**R15 单 seed 的「unweighted 1/8、weighted 3/8」不能直接当作稳定效应量：unweighted 同一配置换 seed 就在 0–3/8 之间波动（1/3/0），R15 的 1 恰好落在自身 seed 波动内；但 weighted 在旧 8 例上三个 seed 都是 3/8（极差 0），且在本轮首次使用、从未参与像素训练或评估的 24 个确认世界上，weighted 三个 seed 的 feasible 为 9/7/6，unweighted 为 3/6/3。方向性（weighted ≥ unweighted）在 6 组 seed 配对中 5 组更高、1 组持平、0 组更低，因此不能把 R15 的差异整体归为纯噪声；但也不能据此宣称加权带来可靠收益。**

关键反证与限制同时成立，必须一起读：

- 同一确认渲染上，**未训练基线 B0 得到 9/24**（F1 8/8、F2 0/8、F3 1/8），与 weighted 最好的 seed 持平、高于 unweighted 均值。也就是说，加权相对未训练基线在**总体**上并没有胜出。
- 两个训练臂都**丢失了 B0 的 F1 行为**（B0 8/8，unweighted 1–5/8，weighted 4–5/8）；weighted 相对 unweighted 的确认世界优势主要来自 F1 与 F2，F3 两边都接近全灭。
- 离线指标方向与闭环相反：weighted 的 train/val accuracy 在 3 个 seed 上都低于 unweighted，val moving macro 在两个 seed 上更低。加权不是通过提高离线模仿率起作用的。
- 确认世界是 12 组几何 × 2 镜像 = 24 例，镜像对参数签名相同、**不是独立样本**；seed 只有 3 个。本轮不做显著性检验、不宣称统计结论。
- 本轮是合成研究，**不开启正式 P3/P5/T21 门，不产生可部署 checkpoint，不运行游戏**。旧 8 validation 已反复暴露，只是连续性对照。

## 隔离、预登记与公式

只新增[本轮 runner](<../../../../../tools/rl/v373_seed_confirm.py>)、[对应测试](<../../../../../tests/test_v373_seed_confirm.py>)、本报告和[evidence 目录](<../evidence/seed-confirm-round16/preregistration.json>)；不修改旧模块、不改 R10/R13/R15 证据、不覆盖旧 run、不读取被封存的 R8 test 定义（`definitions()` 只过滤 `development`/`validation` 并断言无 `test`）。

先跑 CPU 契约，再冻结[preregistration](<../evidence/seed-confirm-round16/preregistration.json>)（SHA `3e5e9f0d3e09ddfae3647239b7c1946817334cb77327af8f40882821b9b42310`），再执行唯一一次训练+评估，最后汇总。prereg 冻结 68 个身份：R10 `records.json` 及其 48 个 pixel/history/label 资产、baseline、R11 闭环 prereg、R15 prereg 与结果、R9 confirmation manifest、相关模块与本轮源码/测试，全部按 SHA 核验。

- 动作权重**直接复用 R15 模块的同一函数**（`tools/rl/v373_weighted_diagnostic.py`，源码 SHA `a2c0d9710e021f9767cd4b139d8e27ce33418859f0485ccdfadfe037a06d5e2f`），没有另写一份：`u_c=min(5,sqrt(N/(K*n_c)))`，`w_c=u_c/Z`，只在 train 标签上统计。本轮从数据重算的结果与 R15 冻结的 `train_weights` **逐字段完全相等**：`N=640`、`K_present=5`、counts `{hold0:530, right3:44, left7:44, rightfocus12:11, leftfocus16:11}`、weights `{0:0.64769, 3/7:2.24791, 12/16:4.49582}`。
- 公式 spec 摘要 SHA `51711002bb92e0051f061401ca1513206bf61d744a9961896fdfe306257c5253`（文本+计数+权重+归一化），公式文本 SHA `dd57860916701edbaa41c09a49aa4f4d7fd29f1f1a911dd3c7bd996159e325a4`。
- 训练数据只取原 R10 **train 8 events**（640 有效行）；validation 8 events 不参与训练、不用于权重/epoch/选模。固定 200 updates，无早停、无按结果调参。

## 训练：2 臂 × 3 seed，fresh 初始化

每臂每个 seed 在构造前重新 `torch.manual_seed(seed)`；两臂结构完全相同（都是 N1，只有 action loss 处理不同），因此同 seed 两臂的初始 head state 必须逐字节相同——prereg 冻结了 per-seed 初始 SHA，运行时再次断言。**seed 7 的初始 SHA 与 R15 完全一致（`66e5f85f…`）**，seed 7 两臂的 final head SHA 及旧 8 例逐事件 firsthit 也与 R15 逐位相同，说明本轮训练/闭环可复现 R15 的单 seed 结果。

| arm-seed | init head SHA | final head SHA | train acc / moving | val acc / moving | 优化秒 |
|---|---|---:|---:|---:|---:|
| unweighted-s7 | 66e5f85fe6b8 | 8a06c56e86d8 | 94.53 / 46.02 | 92.66 / 47.22 | 10.39 |
| weighted-s7 | 66e5f85fe6b8 | ba346b8e9841 | 90.00 / 41.48 | 88.28 / 44.44 | 10.27 |
| unweighted-s17 | f7d03e205f37 | 2f2d91c7f257 | 93.59 / 46.02 | 91.72 / 47.22 | 10.27 |
| weighted-s17 | f7d03e205f37 | 58c974ca4321 | 92.66 / 42.61 | 90.94 / 45.14 | 10.62 |
| unweighted-s27 | 276573d10853 | a03685b92812 | 92.97 / 38.64 | 91.09 / 39.58 | 10.48 |
| weighted-s27 | 276573d10853 | d0b9f9a7409c | 92.66 / 42.61 | 90.31 / 44.44 | 10.72 |

初始（未训练）6 个 head 的 train/val accuracy 与 moving macro 均为 0。表中为 unmasked18 的单路线一致率 / moving macro recall（%）；horizontal_masked、每族、每类 confusion 与 support 在 6 份 `run/train/*/training.json` 与 [checkpoints.json](<../evidence/seed-confirm-round16/run/checkpoints.json>) 中，未只保留好结果。active 参数 429,200，heads 494,992，含冻结 backbone 总 2,002,154；三个 seed 相同。backbone 逐 state tensor 严格不变。**weighted 的离线 accuracy 三个 seed 都低于 unweighted**，val moving macro 两低一高（s27 反向），闭环差异不能解释成“更好地模仿标签”。

## 全新确认世界（prereg 内冻结，只用于闭环确认）

12 组参数（F1/F2/F3 各 4 组）× 2 镜像 = 24 例，`split='confirmation'`，全部用 `tactical_teacher_validation.ValidationWorld` 构造、`make_world` 生成，渲染/碰撞/`advance` 与 R11 完全同一路径。参数在 prereg 中冻结，之后未改动（worlds digest `6e170fae589293a50248913de09609255754cb2774daee5ac9a2ef4d39a01720`）。

- 与 R8 development/validation（21 个几何签名）、R9 confirmation（12）、R10 records（8，且已验证是 R8 development/validation 子集）按 R9 测试同款签名逐字段比较，**0 重复**；每对镜像签名相同。
- 只做了**算术可达性**检查（F1 在 wall 前能否到达安全带、F3 窗口结束时能否低于阈值且 horizon 内可达 goal、F2 发射间隔与横向空间），24 例全部通过；**没有**对确认世界调用教师策略、`solve`/`programs` 或生成任何标签，也没有用参考解筛除“不可行”世界。因此“这些世界一定可被某条路线通关”不是本轮结论；B0 与 6 个 checkpoint 的实测是唯一参照。
- 结构上确认世界的 F1 对基线较容易（B0 8/8），F3 仍然很难（B0 1/8）。这会影响解释：F1 反映训练是否保留基线行为，而非加权收益。

## 确认世界闭环（24 例；feasible / survived）

横向 mask `[0,3,7,9,12,16]`、深度 3 因果渲染、实际执行历史、sticky 碰撞并跑满 160 逻辑帧、每事件 hidden reset；逐事件 row（含 80 个决策与实际 trace）保存在 [各臂闭环 JSON](<../evidence/seed-confirm-round16/run/closed-loop/weighted-s7-confirmation.json>)（此处链接 weighted-s7 确认集，其余同名文件同目录）。B0 用同一 runner/renderer、只取冻结 baseline 的 policy logits，**只是这些渲染上的参照，不是真实 B0 能力**。

| arm-seed | F1（8） | F2（8） | F3（8） | 总（24） |
|---|---|---|---|---|
| B0 | 8 / 8 | 0 / 0 | 1 / 1 | **9 / 9** |
| unweighted-s7 | 1 / 1 | 2 / 2 | 0 / 2 | 3 / 5 |
| unweighted-s17 | 5 / 5 | 1 / 1 | 0 / 0 | 6 / 6 |
| unweighted-s27 | 1 / 1 | 2 / 2 | 0 / 0 | 3 / 3 |
| weighted-s7 | 5 / 5 | 3 / 3 | 1 / 1 | 9 / 9 |
| weighted-s17 | 4 / 4 | 3 / 3 | 0 / 1 | 7 / 8 |
| weighted-s27 | 4 / 4 | 2 / 2 | 0 / 2 | 6 / 8 |

F3 的 goal 与 survived 需要分开看（到达 goal 后仍可能碰撞，goal 不能挽救失败）：

| arm-seed | F3 goal / survived / feasible（各 8） |
|---|---|
| B0 | 4 / 1 / 1 |
| unweighted-s7 | 2 / 2 / 0 |
| unweighted-s17 | 8 / 0 / 0 |
| unweighted-s27 | 0 / 0 / 0 |
| weighted-s7 | 4 / 1 / 1 |
| weighted-s17 | 5 / 1 / 0 |
| weighted-s27 | 2 / 2 / 0 |

unweighted-s17 的 F3 goal 8/8，但其中 **6/8 在碰撞前就已到达 goal**（goal 帧 42–50，碰撞 140–158），另 2 例在 46 帧即碰撞、goal 是碰撞后继续左移才达到。也就是说它早期确实执行了穿过窗口的左移，失败更可能在后续避弹/时序，而不是“完全不会往左走”；但 F3 的 goal 列本身会被“碰撞后继续跑满 horizon”抬高，只有在 survived 前提下的 feasible 才有意义。F1/F2 的 goal 恒真，feasible 与 survived 相同。

### 3 seed 均值 / 极差（不做事后挑选）

| 指标 | unweighted 均值 / 极差 | weighted 均值 / 极差 |
|---|---|---|
| 确认 all feasible（/24） | 4.00 / 3 | 7.33 / 3 |
| 确认 all survived（/24） | 4.67 / 3 | 8.33 / 1 |
| 确认 F1 feasible（/8） | 2.33 / 4 | 4.33 / 1 |
| 确认 F2 feasible（/8） | 1.67 / 1 | 2.67 / 1 |
| 确认 F3 feasible（/8） | 0.00 / 0 | 0.33 / 1 |
| 旧 val feasible（/8） | 1.33 / 3 | 3.00 / 0 |
| 旧 val survived（/8） | 1.33 / 3 | 3.00 / 0 |

### 确认世界逐事件 firsthit（逻辑帧；— 为完整无碰撞）

| 确认事件 | B0 | u-s7 | u-s17 | u-s27 | w-s7 | w-s17 | w-s27 |
|---|---:|---:|---:|---:|---:|---:|---:|
| confirm-F1-00-m0 | — | 58 | 58 | 58 | 58 | 58 | 58 |
| confirm-F1-00-m1 | — | 58 | — | — | 58 | 58 | 58 |
| confirm-F1-01-m0 | — | 46 | — | — | — | 46 | — |
| confirm-F1-01-m1 | — | 46 | 46 | 46 | 46 | 46 | 46 |
| confirm-F1-02-m0 | — | — | — | — | — | 54 | — |
| confirm-F1-02-m1 | — | 54 | 54 | 54 | 54 | 54 | 54 |
| confirm-F1-03-m0 | — | 62 | — | — | — | — | — |
| confirm-F1-03-m1 | — | 62 | — | — | — | 62 | — |
| confirm-F2-00-m0 | 118 | — | — | 60 | 76 | — | 76 |
| confirm-F2-00-m1 | 96 | 76 | 96 | 96 | 78 | 76 | 78 |
| confirm-F2-01-m0 | 90 | — | — | — | — | — | — |
| confirm-F2-01-m1 | 94 | 70 | 94 | 94 | 72 | 70 | 72 |
| confirm-F2-02-m0 | 94 | 62 | 94 | 94 | — | 62 | — |
| confirm-F2-02-m1 | 94 | 62 | — | 88 | — | 62 | 68 |
| confirm-F2-03-m0 | 110 | 76 | 110 | 110 | 108 | 74 | 106 |
| confirm-F2-03-m1 | 106 | 72 | 106 | 106 | 102 | 72 | 78 |
| confirm-F3-00-m0 | 62 | 140 | 98 | 140 | 140 | 98 | 140 |
| confirm-F3-00-m1 | 146 | — | 142 | 144 | 140 | 98 | — |
| confirm-F3-01-m0 | 64 | 96 | 96 | 152 | 96 | 96 | 96 |
| confirm-F3-01-m1 | 158 | — | 152 | 152 | — | 96 | — |
| confirm-F3-02-m0 | 72 | 46 | 46 | 46 | 46 | 106 | 46 |
| confirm-F3-02-m1 | 46 | 46 | 46 | 46 | 46 | 46 | 46 |
| confirm-F3-03-m0 | 102 | 154 | 102 | 154 | 154 | 102 | 102 |
| confirm-F3-03-m1 | — | 102 | — | 158 | 154 | 102 | 154 |

观察：weighted 的优势不是“全面更晚碰撞”。F1 的 46/58/62 早碰撞在多数臂上相同；weighted 主要在若干 F1/F2 事件上完整存活（—），unweighted 在这类事件上更常早撞。F3 两臂都普遍在 46–158 间碰撞，没有稳定差异。

## 旧 8 validation 连续性对照（已反复暴露，不是独立 test）

同一 runner、同一 8 例，仅作连续性对照；R11 的 B0 2/8 是外部参照（R11 闭环 cuDNN TF32=true，实现路径也不同；B0 本身仍是同一 baseline 的 logits），本轮不重复 B0。

| arm-seed | F1（2） | F2（4） | F3（2） | 总（8） |
|---|---|---|---|---|
| unweighted-s7 | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| unweighted-s17 | 2 / 2 | 0 / 0 | 1 / 1 | 3 / 3 |
| unweighted-s27 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| weighted-s7 | 2 / 2 | 0 / 0 | 1 / 1 | 3 / 3 |
| weighted-s17 | 1 / 1 | 2 / 2 | 0 / 0 | 3 / 3 |
| weighted-s27 | 1 / 1 | 2 / 2 | 0 / 0 | 3 / 3 |

逐事件 firsthit：

| validation 事件 | u-s7 | u-s17 | u-s27 | w-s7 | w-s17 | w-s27 |
|---|---:|---:|---:|---:|---:|---:|
| validation-F1-structure-00-m0 | — | — | — | 40 | — | — |
| validation-F1-structure-00-m1 | — | — | 40 | 40 | 40 | 40 |
| validation-F2-structure-00-m0 | 104 | 112 | 92 | 76 | 92 | 76 |
| validation-F2-structure-00-m1 | 102 | 136 | 88 | 74 | 88 | 74 |
| validation-F2-structure-01-m0 | 76 | 76 | — | 72 | — | 76 |
| validation-F2-structure-01-m1 | 102 | 102 | — | 52 | — | 52 |
| validation-F3-structure-00-m0 | 94 | — | 136 | 94 | 136 | 94 |
| validation-F3-structure-00-m1 | — | 138 | 94 | 94 | 94 | 94 |

weighted 三个 seed 的 3/8 由不同事件组成（s7 靠 F1 两例+F3，s17/s27 靠 F1 一例+F2 两例），不是同一 3 例的重复。

## 判定：噪声与否

1. **数值层面的 R15 差异不稳定。** unweighted 同配置换 seed 的旧 val feasible 是 1/3/0；R15 的 1/8 只是其中一个抽样。用 3/8 去和 1/8 比较“提高 2 例”没有稳定基准。
2. **方向层面不能归为纯噪声。** weighted 在旧 val 三 seed 恒为 3/8，在全新 24 例上三 seed 均高于同 seed 的 unweighted（9>3、7>6、6>3），6 组配对 5 高 1 平 0 低。就这些渲染而言，“加权至少不差于未加权”是一致的。
3. **但收益不是通用的。** 确认世界总体 B0 9/24 ≥ weighted 均值 7.33、> unweighted 均值 4.00；两个训练臂都损失了 B0 的 F1 能力；F3 两边都几乎全灭；weighted 离线 accuracy 三个 seed 都更差。更准确的描述是：**加权在旧数据上稳定地少损失一点闭环成功、并在新世界上比 unweighted 更少伤害基线式 F1 行为，同时略微提高 F2 存活；没有证据表明它提升了总体能力或 F3。**
4. 因此本轮**不支持**“R15 的 weighted 优势是可靠收益”，也**不支持**“R15 差异纯属噪声”。按预注册口径，这只是 3 seed、12 组镜像几何的合成确认，不做显著性声明，不换 baseline、不改权重、不追加 epoch。

## 执行、第一失败与恢复

```powershell
python -m pytest tests/test_v373_seed_confirm.py -q -p no:cacheprovider --basetemp .tmp-r16-tests-03
python -m tools.rl.v373_seed_confirm preregister
python -m tools.rl.v373_seed_confirm run
python -m pytest tests/test_v373_seed_confirm.py -q -p no:cacheprovider --basetemp .tmp-r16-tests-04
python -m tools.rl.v373_seed_confirm complete
```

退出码依次为 **0 / 0 / 1 / 0 / 0**。测试在 prereg 前与修复后均为 **9 passed**（[tests-first.log](<../evidence/seed-confirm-round16/tests-first.log>)、[tests-completion.log](<../evidence/seed-confirm-round16/tests-completion.log>)）；完整训练/闭环日志见 [execution-first.log](<../evidence/seed-confirm-round16/execution-first.log>)，命令、退出码与环境的机器可读记录见 [execution-notes.json](<../evidence/seed-confirm-round16/execution-notes.json>)。首次测试调用曾有 1 个测试自身的 AST 断言错误（把本轮自定义的 `verify` 定义误判为教师 solver 调用），修正测试后通过；当时尚未 prereg、未训练。训练与评估均在该 prereg 后一次执行。

`run` 的**第一失败已保留**（[first-failure.json](<../evidence/seed-confirm-round16/run/first-failure.json>)，SHA `9aafdd45…`）：6 个 checkpoint 与全部 **216 条闭环事件**（B0 24 + 6×24 确认 + 6×8 旧 val）都已落盘完成后，**在最后的汇总聚合处**崩溃（把已聚合的 B0 families 又传给 `families()`，`TypeError: string indices must be integers`）。训练、编码、闭环本身没有失败。

恢复方式：修正该一行聚合错误，并在同一新模块里加入只读 `complete` 命令，从 `run/` 已冻结产物重建汇总；**没有重训、没有重新编码、没有重跑任何闭环、`run/` 目录未被改写**。`complete` 会逐项核验：6 个 checkpoint 文件 SHA、checkpoint 记录与 training 记录一致、confirmation worlds digest、以及每条存储事件**重算出的 family 聚合必须等于存储值**。结果在 [run-completion/summary.json](<../evidence/seed-confirm-round16/run-completion/summary.json>)（SHA `3f024c57…`）与 [notes.json](<../evidence/seed-confirm-round16/run-completion/notes.json>)（SHA `9533b3d4…`）。

**必须披露的身份变化**：prereg 冻结的模块 SHA 是 `e4b86995…`，修复+新增只读恢复命令后为 `ba1f1370…`；[notes](<../evidence/seed-confirm-round16/run-completion/notes.json>) 记录了两个 SHA 与变更原因。其余 **67/68 个身份**（R10 全部资产、R11/R15/R9 冻结文件、旧模块、本轮测试文件等）仍完全匹配；改动的两处都不在训练/评估路径上，且所有闭环产物都产生于改动之前。若要求“摘要也必须由 prereg 冻结源码写出”，本轮只能标记为**部分满足**，不以重跑掩盖。

## 更正（round17 追加，不改上文）

R17 在复现本轮的 unweighted 臂时发现：**本文"逐事件 firsthit"表（旧 val 那 6 列）与 `run/closed-loop/*-validation.json` 存储数据不一致**，列值被错位填充。主线已用脚本从存储文件重算核对：

- 旧 val 的**汇总表**（F1/F2/F3/总，第 123–130 行）与存储数据**完全一致**：u 1/3/0、w 3/3/3，分族也一致。
- 但**逐事件 firsthit 表**（第 134–143 行）对 u-s7 与 w-s7 两列明显错位（例如存储里 u-s7 的 F1 两例是 40/40，表中写成 "—"；w-s7 的 F3-m1 是 "—"，表中写成 94）。
- 以存储 JSON 为准的正确值：

| validation 事件 | u-s7 | u-s17 | u-s27 | w-s7 | w-s17 | w-s27 |
|---|---:|---:|---:|---:|---:|---:|
| F1-00-m0 | 40 | — | 40 | — | — | — |
| F1-00-m1 | 40 | — | 40 | — | 40 | 40 |
| F2-00-m0 | 76 | 112 | 76 | 104 | 92 | 92 |
| F2-00-m1 | 74 | 136 | 74 | 102 | 88 | 88 |
| F2-01-m0 | — | 76 | 72 | 76 | — | — |
| F2-01-m1 | 52 | 102 | 52 | 102 | — | — |
| F3-00-m0 | 136 | — | 94 | 94 | 136 | 136 |
| F3-00-m1 | 94 | 138 | 94 | — | 94 | 94 |

结论层面不受影响（汇总与分族计数正确、A 臂与 R16 存储逐位相同已由 R17 复核），但**引用逐事件细节时必须用存储 JSON**，不要引用上表旧版本。另注：w-s17 与 w-s27 的逐事件结果完全相同，而两者 final head SHA 不同；未进一步解释，仅记录。

## 保留边界

- 所有 JSON 用 `x` 模式、权重用 `xb`、输出目录 `exist_ok=False`；旧 R10/R13/R15 证据未改。
- 确认世界只用于闭环确认：未训练、未参与选权重/epoch/选模、未产生教师标签；评估路径不调用 `solve`/教师策略（`run_event` 仅对**实际执行**的动作做 replay 以生成 environment_metrics 的 verdict，不影响动作选择）。
- 不把 24 例当独立大样本（12 组镜像、3 seed）；不宣称统计显著、不宣称真实泛化、不支持部署或战术理解；不开启 P3/P5/T21，不产可部署 checkpoint，不运行游戏。
