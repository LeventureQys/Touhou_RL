# P5 合成横向 toy 闭环开发研究（round11，非正式 P5/T21）

## 结论先行

**固定研究 N1 在这 8 个已暴露 validation 事件上，横向 mask 闭环全程成功 2/8；同 mask 的 B0 也是 2/8，没有观察到总体生存/全程目标成功率改善。** 固定 hold 与训练集频率 mode 均为 0/8。F2、F3 中 N1 全部碰撞，不能把 round10 的 91.41% 单路线标签一致率解释为闭环可靠性。

这是**仅横向、合成像素、开发集闭环研究**。原 8 validation 已用于 round10 优化指标评估，不是独立 test；镜像也不是独立模板。未运行游戏，未新增训练、调参、换 checkpoint、失败补标或 teacher 重标；未修改正式 loader、renderer、heads 或旧 run；正式 P5/T21、真实数据门、部署资格均未通过。

## 先冻结，再执行

[预注册 JSON](<../evidence/synthetic-closed-loop-round11/preregistration.json>) 在模型闭环前落盘，固定 8 个 event/spec、四臂、两种动作条件、动作映射、指标、取图顺序和资产 SHA。固定 prereg SHA 为 `4ae47790c2c94a014d04fd608f1141e20a30269d4c64970de06b04b093352406`。

| 冻结对象 | SHA256 |
|---|---|
| round10 固定末步研究 heads | `85cc2f4ed93c6e513d76ffde559ca61e2c3061afb9522694d9831ee2838d81d7` |
| 原 B0 baseline | `74abde8a996c13fae832913e4e5533bc9490688bc4c0aa73560e695f065d14be` |
| 原 records | `cb229c99ddb95e17ec201807dc3a336de0a9edf558329529f56ed1328fa2a30e` |

研究 heads 来自 [固定研究权重](<../evidence/synthetic-learning-round10/run-s7-n1-200/research-heads.pt>)，基座来自 [B0 checkpoint](<../../../../../third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt>)。独立 [闭环实现](<../../../../../touhou_rl/tactical_synthetic_closed_loop.py>) 手动检查研究 schema `v373-synthetic-render-only-heads/1`、原 baseline SHA、strict heads state_dict；**没有修改正式 checkpoint loader**。执行后全部冻结身份一致，详见 [summary](<../evidence/synthetic-closed-loop-round11/run/summary.json>) 与 [逐资产 SHA](<../evidence/synthetic-closed-loop-round11/run/sha256.json>)。

训练集 mode 仅统计原 train8 的 640 个有效单路线标签：candidate0 为 530/640，mode=0。因此 mode 和固定 hold 是两个预登记控制臂，但本次实际策略相同；绝不利用 validation 频率确定 mode。

## 动作与闭环合同

主比较为 `horizontal_masked`：**所有臂在 argmax 前使用同一个固定动作域 mask** `[0,3,7,9,12,16]`，分别对应 toy 两帧动作 `[0,+2,-2,0,+1,-1]`。候选遵循 `focus*9+move`；0/9 分别为两种 focus 的静止，3/7 为不 focus 左右，12/16 为 focus 左右。9 虽与 0 的位移相同，history 保留实际 candidate9，不重编码为0。

该 mask 是预注册的**横向任务定义**，不是依危险状态触发的 shield，不是完整 18 动作环境。上下/斜向候选不能投影到水平后冒称完整动作。补充 `unmasked` 条件中，网络选到不支持候选即标记 `invalid_action`、无实际动作、立即技术失败终止；无 fallback、无 shield、无 teacher 救援。预注册列出了 candidate0..17 的完整 focus/move/vector/支持性表。

环境每次只以实际选中 candidate 映射的动作调用原 `advance`，推进两逻辑帧；实际当前位置又决定后续瞄准弹出生方向。下一输入重新渲染实际 state，不重放教师轨迹。原 renderer 只显示当前可见几何/已出生弹/实际玩家，HUD 固定，不添加 F1 提前墙、F3 未来窗口或目标提示。

输入深度3，与 dataset 一样按时间旧到新排列 `[max(0,f-4),max(0,f-2),f]`，即当前与实际过去两张决策帧，初始重复填充。每事件 GRU `hidden=None/reset=true`；第一步 BOS。仅执行成功后提交实际 candidate、hold2、delta2；previous_option 恒 BOS3，target 无效/零。预测 aux option/target 只参与 N1 内部未监督条件，不宣称已被采用为战术。

每 step 的 [实例 trace](<../evidence/synthetic-closed-loop-round11/run/horizontal_masked/validation-F3-structure-00-m0/N1/trace.json>) 记录像素 NPZ 引用与实际帧索引、原始/最终候选、实际动作/hold、前后位置、hidden reset、执行历史、碰撞、invalid/fallback。对应 [实际像素](<../evidence/synthetic-closed-loop-round11/run/horizontal_masked/validation-F3-structure-00-m0/N1/pixels.npz>) 可重建每个输入 stack；每个条件/事件/臂均有独立 trace 与像素，不改输入、不补标签。

## 主结果：同 mask 横向任务

全程 feasible = 无技术失败、推进完整160帧、无碰撞，且 F3 曾在实际采样状态达到 canonical x≤goal。F1/F2 无额外目标条件。结果 denominator 是事件，不是640帧独立样本。

| 族 | 事件数 | B0 全程成功 | 固定 hold | train mode | N1 全程成功 |
|---|---:|---:|---:|---:|---:|
| F1 | 2 | 2/2 | 0/2 | 0/2 | 2/2 |
| F2 | 4 | 0/4 | 0/4 | 0/4 | 0/4 |
| F3 | 2 | 0/2 | 0/2 | 0/2 | 0/2 |
| 总计 | 8 | **2/8** | **0/8** | **0/8** | **2/8** |

所有横向 masked 臂都完成8×80决策，invalid=0，fallback=0。碰撞后继续推进 toy 到 horizon 以记录完整轨迹；**hit 为 sticky，后续达到目标不能挽回失败**。末端剩余空间/弹幕 occupancy 等环境指标包含碰撞后的状态，不赋予安全价值，不按复活/真实游戏生存解释。

首碰撞按原 advance 输出、2帧粒度记录，非精确逐逻辑帧死亡时间：

| validation 事件（顺序与预注册一致） | B0 | hold/mode | N1 |
|---|---:|---:|---:|
| F1-00-m0 | 无 | 40 | 无 |
| F1-00-m1 | 无 | 40 | 无 |
| F2-00-m0 | 112 | 46 | 112 |
| F2-00-m1 | 92 | 46 | 136 |
| F2-01-m0 | 76 | 28 | 76 |
| F2-01-m1 | 102 | 28 | 102 |
| F3-00-m0 | 58 | 94 | 138 |
| F3-00-m1 | 142 | 94 | 138 |

F2 在4例中仅1例 N1 比同 mask B0 晚碰撞，其余相同；所有例仍失败，不能宣传总体收益。F3 N1 两例都曾达到目标，但138帧碰撞；B0 m1碰撞还略晚，为142。F1 两例成功也不足以说明学到未来墙结构：相同 B0 已成功，且未来墙在出现前不可见。

## 补充：unmasked 支持性审计

| 臂 | unsupported/实际观察决策 | invalid 事件 | 完整 horizon | 全程成功 |
|---|---:|---:|---:|---:|
| B0 | 8/8（100%） | 8/8 | 0/8 | 0/8，技术失败而非可比生存分数 |
| hold | 0/640 | 0/8 | 8/8 | 0/8 |
| train mode | 0/640 | 0/8 | 8/8 | 0/8 |
| N1 | 0/640 | 0/8 | 8/8 | 2/8 |

B0 每事件第一决策即选到不支持的竖向/斜向候选，所以未执行；**不能外推为它后续640步的动作率，也不能拿这张表声称 N1 打败了完整 B0**。另在 horizontal_masked 的实际轨迹上，B0 原始未mask argmax有625/640（97.65625%）不受支持，N1为0/640；这是另一轨迹条件的诊断，不是 unmasked 继续执行的结果。

## 实际图数与证据

按预注册每族取前最多3个**不同真实 validation 事件**，不复制/造例：F1只有2例，F3只有2例，均不足3；F2有4例，固定取前3。总7张，每张叠加四臂 masked 实际位置；不是三张图假装三个独立事件。

- F1（实际2）：[m0](<../evidence/synthetic-closed-loop-round11/run/figures/validation-F1-structure-00-m0.png>)、[m1](<../evidence/synthetic-closed-loop-round11/run/figures/validation-F1-structure-00-m1.png>)。
- F2（实际选3/4）：[00-m0](<../evidence/synthetic-closed-loop-round11/run/figures/validation-F2-structure-00-m0.png>)、[00-m1](<../evidence/synthetic-closed-loop-round11/run/figures/validation-F2-structure-00-m1.png>)、[01-m0](<../evidence/synthetic-closed-loop-round11/run/figures/validation-F2-structure-01-m0.png>)；第四例仍有完整 JSON/像素。
- F3（实际2）：[m0](<../evidence/synthetic-closed-loop-round11/run/figures/validation-F3-structure-00-m0.png>)、[m1](<../evidence/synthetic-closed-loop-round11/run/figures/validation-F3-structure-00-m1.png>)。

图通过 PIL 原生输出；本会话图像读取工具因模型不支持图像输入拒绝预览，未宣称人工视觉验收。数值结论依据 JSON trace，而不是图形外观。

## 执行与验证

```powershell
python -m touhou_rl.tactical_synthetic_closed_loop preregister
python -m pytest touhou_rl/tests/test_tactical_synthetic_closed_loop.py -q -p no:cacheprovider --basetemp .tmp/v373-round11-closed-loop-tests-02
python -m touhou_rl.tactical_synthetic_closed_loop run --device cuda
```

GPU run退出0；真实 RTX4070 SUPER，batch1，冻结 eval/inference，无 AMP/optimizer。记录 [执行说明](<../evidence/synthetic-closed-loop-round11/execution-notes.json>)：PyTorch2.14.1+cu132、CUDA13.2，独立同环境进程观察默认 matmul TF32=false、cuDNN TF32=true、float32 precision=highest。本研究实现未更改默认精度，也未使用 round10 encoder 的 `_fp32_mode` 显式关闭 cuDNN TF32，所以不声称与原离线 batch16 编码数值逐位一致；未因结果修改已冻结源码重跑。

独立 [stub 单测](<../../../../../touhou_rl/tests/test_tactical_synthetic_closed_loop.py>) **6 passed in1.66s**，覆盖：实际左右候选改变当前位置和后续像素/弹出生因果；unmasked unsupported 无执行终止；固定相同动作域mask；candidate9不丢失/不提交aux；事件hidden reset；深度3 padding与HUD恒定；碰撞后完整 horizon 仍不算成功。首次测试1failed/5passed仅因fixture horizon6使原 firing条件 `frame<horizon-30` 永不发弹，修正fixture到40后通过，未修改实现/prereg/checkpoint。

## 研究边界

当前证据支持“原研究 N1 可以在因果实际状态输入下执行横向闭环，但闭环成功仅2/8，与同mask B0持平，F2/F3失效仍明显”，**不支持正式 P5/T21、不支持18动作完整环境、不支持真实游戏安全或泛化提升**。8例仅4个验证模板组（F1/F3各1、F2为2）与镜像；不做小样本显著性宣称。下一步若继续必须另外预注册独立数据与评价，不可回填本轮失败为新输入后仍称冻结测试。
