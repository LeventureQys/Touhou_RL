# 单因素加权动作监督：开发对照（续轮15）

## 身份与隔离

只改一个因素：N1 动作损失的类别权重。数据仍为原 R10 的 train 8 事件；validation 8 事件不参与训练、不参与权重计算、不用于选步数/选权重。渲染器、教师、世界动力学、网络结构、容量、初始化、优化器和预算全部不变。这是合成研究，不开启正式 P3/P5/T21 门，不产生可部署 checkpoint，不运行游戏，不替换默认模型。

[预登记](<../evidence/weighted-round15/preregistration.json>)在训练前冻结，含源码、测试、原 records、48 个像素/历史/标签资产及基座 SHA（SHA `c94b41a033417a1b505d8b9a47326ee2c5e4f5e18a2a06a71026a45e99d8b361`）。

**权重公式（只由 train 640 条有效标签计算）**：`u_c = min(5, sqrt(N/(K_present·n_c)))`，再按 train 平均归一 `w_c = u_c / (Σ n_c u_c / N)`。缺失类别没有目标，不设权重；`L_action = Σ_valid w_y·(-log p[y]) / Σ_valid w_y`，其余损失项与系数不变。

| 候选 | train 计数 | 归一权重 |
|---|---:|---:|
| 0 hold |530|0.647690|
| 3/7 左右 |44/44|2.247910|
| 12/16 聚焦左右 |11/11|4.495821|

两臂**初始 head 状态 SHA 完全相等**（`66e5f85f…a8f7799`），fresh 初始化、同 seed7、round-robin、full80、batch1、Adam1e-3、clip1、200 updates、FP32 且 matmul/cuDNN TF32 均关闭。真实 RTX 4070 SUPER 一次 batch16 编码共享内存特征 2.73s，未落盘生产 cache。

## 实际命令

```powershell
python -m pytest touhou_rl/tests/test_v373_weighted_diagnostic.py touhou_rl/tests/test_v373_arm_diagnostic.py -q -p no:cacheprovider --basetemp .tmp/v373-weighted-round15-tests
python -m tools.rl.v373_weighted_diagnostic preregister --out Document/Update/Alpha/v3.7.3/evidence/weighted-round15
python -m tools.rl.v373_weighted_diagnostic run --out Document/Update/Alpha/v3.7.3/evidence/weighted-round15 --device cuda
```

测试 14 passed（1.23s）。run 退出 0，两臂各 200 步，无增 epoch、无按 val 选模。

**过程失败披露**：子任务报告其分析步骤首次因缺少 `PYTHONPATH` 退出 1；保存的[首次执行日志](<../evidence/weighted-round15/execution-first.log>)中训练与闭环逐事件行完整、无异常堆栈，随后仅重跑分析得到 summary，训练证据未被覆盖。原始日志按 UTF-16 保存，读取需 `-Encoding Unicode`。

## 结果

[完整汇总](<../evidence/weighted-round15/run/summary.json>)、逐资产 SHA 见[sha256](<../evidence/weighted-round15/run/sha256.json>)、两臂训练与初始化见各自 `training.json`/`initialization.json`。

| 指标（validation 离线） | unweighted | weighted |
|---|---:|---:|
| accuracy |92.656%|88.281%|
| 5 类 macro balanced recall |57.51%|54.39%|
| moving macro recall |47.22%|44.44%|
| F1 recall [hold,left,right] |[.987,.944,.944]|[.942,.806,.972]|
| F2 聚焦左右 recall |[0,0]|[0,0]|
| F3 聚焦左右 recall |[0,0]|[0,0]|

| 闭环（horizontal_masked，8 已暴露 validation） | unweighted | weighted |
|---|---:|---:|
| 全事件 feasible |**1/8**|**3/8**|
| F1 |0/2|2/2|
| F2 |1/4|**0/4**|
| F3 |0/2|1/2|

逐事件 firsthit：unweighted `F1 40/40、F2 76/74/—/52、F3 136/94`；weighted `F1 —/—、F2 104/102/76/102、F3 94/—`。

## 解读（不作能力结论）

1. 加权把闭环从 1/8 提到 3/8，但**不是全面改善**：F1 +2、F3 +1，而 F2 由 1/4 变 0/4，即换来了原本成功的那个事件失败。总体净 +2 只基于 8 个事件、单一 seed，不能排除噪声。
2. 离线 accuracy 和两类 moving 指标都**下降**，说明"整体一致率"与闭环不是同一件事：加权牺牲了占多数的 hold/普通移动，换取少数类的梯度份额，但没有换取到真正需要的聚焦微移。
3. **目标是修复聚焦移动，但两种聚焦 recall 在两臂仍全为 0**。权重加大到 4.5 仍为 0，说明问题不只是类别不平衡：F2/F3 的聚焦左右在渲染中与普通移动同方向、仅速度不同，像素层面的可区分性与 credit assignment 可能是主因。这是下一轮应单因素检验的假设，不是已证结论。
4. F1 加权后 left recall 从 .818 降到 .455，却闭环 2/2，再次说明离线单路线一致率不是闭环预测器。
5. R15 的 unweighted 臂（1/8）与 R14 N1（1/8）独立复现，事件级结果一致，说明该对照臂可复现。

## 限制

- 单 seed、8 事件、开发 validation 已反复暴露，不是独立 test，也不构成统计显著性。
- 只监督单条教师路线的首动作；其他候选未被证伪。
- 闭环沿用 R11 的 horizontal_masked 语义，与真实游戏无关。
- 由于加权改变了梯度尺度，两臂末步 loss 不可直接比较；未做学习率匹配消融。

下一步已冻结并执行多种子（7/17/27）× 两臂，以及在**从未用于像素训练或评估的新世界**上的一次确认；结论以那份证据为准，本轮不据单 seed 差异调权重或加 epoch。
