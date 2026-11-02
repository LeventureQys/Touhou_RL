# P0 候选帧集合与几何误差区间（round7）

## 结论及证据身份

**本轮不再尝试用软件 counter 证明 Present fence。只诊断每张高置信 player 像素定位 q，在真实、离散 memory 几何轨迹上的兼容帧集合。phase 仍 unknown，`phase_verified=False`、`real_pixel_gate_passed=False`、`bulletpool_same_frame_proven=False`。player 对应不证明 bulletpool 同帧。**

已读取 [round6 bank 报告](<P0-pixel-bank-round6.md>)、[确认采集报告](<P0-confirmation-round6.md>)、[bank 模块](<../../../../../touhou_rl/tactical_pixel_bank.py>)、[probe 模块](<../../../../../touhou_rl/tactical_pixel_probe.py>)及既有开发/确认输出。**round6 确认 bank 汇总在 round7 设计前已经暴露；本轮不是新的独立 blind 确认**。这里的 prereg 只保证本次逐 q 确认 analysis 前冻结代码和开发规则，不倒推宣称从未见过确认数据。未来规则验证必须另采新片段，本轮没有游戏、输入或训练。

新增 [分析模块](<../../../../../touhou_rl/tactical_alignment_intervals.py>)、[合成测试](<../../../../../touhou_rl/tests/test_tactical_alignment_intervals.py>)、[冻结 prereg](<../evidence/alignment-round7/prereg.json>)和[首次确认分析完整输出](<../evidence/alignment-round7/confirmation-analysis-x1.json>)；旧模块、旧数据及旧结果不修改。直接消费既有 bank 的逐图定位，不重新 NCC、不从确认提模板。

## 运行前冻结：不能确认再拟合

顺序：实现与合成测试通过（12 passed）→ 仅读取[开发输出](<../evidence/pixel-bank-round6/dev-round5-x1.json>)执行 `freeze` → 检查 [prereg](<../evidence/alignment-round7/prereg.json>)已落盘 → 才执行确认 `analyze`。两命令退出0，输出均 `x` 模式拒绝覆写。

- 冻结 offset 为开发 `counter_lag_diagnostic.scores` 中 **lag=0** 的 `anchor_offset=(32,16)`。不挑确认最优 lag，不采用确认报告中拟合出的约 `(32.05884,15.94116)`。
- 欧氏几何残差阈值 **2px（含边界）**，沿用旧 probe 的误差预算；不根据确认误差缩放。固定候选窗口 `reference_counter + [-8,+8]`。
- 每个稳定 q 的参考 f 为相等的 pre/post gameplay counter。对全部17个候选 g，计算 `e_g=q-(32,16)-p_g`。候选集合 `C(q)={g: ||e_g||₂≤2px}`，不选最近赢家、最佳 lag 或中位 frame。
- memory 只从真实 pre/post exact counter 建表；同 counter 的不同位置、任何非有限值均冲突。全窗口任一缺帧/冲突即整张 q **unknown**，不插值、不按首末择优。部分可观测候选仍输出，但明确非完整集合，不进入通过统计。
- pixel duplicate 使用稳定、高置信同 counter 定位的 **全 pairwise 直径≤1px**；超过1px全部 unknown（比旧 probe 相对第一点更保守）。不稳定 capture bracket 亦 unknown。低置信图不强造 q。
- 每张 q 单独保留；同 counter 一致的重复 capture 不均值归并成新 q。统计是 capture 计数，**不是独立样本数或置信区间**。81 q 只有35个不同参考counter；66完整 q 为33个不同参考counter。
- 输出每个真实候选 frame、相对 frame、完整17帧逐点残差、候选残差 min/max、`p_g-p_f` 的 xy 与径向区间；所有区间是离散集合 hull，**不是插值、概率区间、真实相位上下界**。
- “全部±1”仅在完整非空集合上有布尔值；unknown/空集为 null，不用空集的 vacuous truth。负相对frame表示更早 memory（与旧 probe 的正lag含义相反）。
- 动态：`||p_f-p_(f-1)||>0.25px`。转向来自 trace 非零方向改变，停顿不重置前方向；动态且距转向≤3 counter 单列。以上规则均写入 prereg。

运行验证源码、probe、开发输入 SHA 与 prereg 一致，并验证确认 trace SHA 存在于 bank 的输入哈希中。首次分析输入 SHA 完整保存于[结果](<../evidence/alignment-round7/confirmation-analysis-x1.json>)。

## 确认结果：时间多解不等于空间超预算

[既有确认 bank](<../evidence/pixel-bank-round6/confirmation-round6-x1.json>)共289张：81高置信 q，208低置信 unknown。

| 分层（分母是高置信 capture） | n | 完整非空 | 全部候选在±1 | 含±1外候选 | unknown | 完整空集/几何超预算 |
|---|---:|---:|---:|---:|---:|---:|
| 全部高置信 |81|66 (81.48%)|59 (72.84%)|7 (8.64%)|15 (18.52%)|0 (0%)|
| 动态 |78|63 (80.77%)|59 (75.64%)|4 (5.13%)|15 (19.23%)|0|
| 静止 |3|3 (100%)|0|3 (100%)|0|0|
| 动态且转向±3 |32|28 (87.50%)|28 (87.50%)|0|4 (12.50%)|0|
| 全部转向±3（含静止） |33|29 (87.88%)|28 (84.85%)|1 (3.03%)|4 (12.12%)|0|

完整非空内部，59/66=89.39% 全部±1，7/66=10.61%含±1外候选。**这只是窗口内几何兼容集合，不是“真实显示延迟在±1”的证明**；无法由现有数据排除窗口以外的等位位置/相位。

- unknown 15张：不稳定 bracket 9张，pixel duplicate冲突6张，后者counter为303、336。本片段无 memory counter冲突，完整窗口缺帧导致的 unknown 为0；缺帧/冲突分支有合成测试。
- 全289图的 unknown 为 **208+15=223/289=77.16%**；完整有候选为66/289=22.84%。低conf不能从高conf表里消失。全图全部候选±1为59/289=20.42%；含±1外为7/289=2.42%。
- “超预算”须区分：**时间预算±1之外**为7/81；**2px几何预算内无候选的完整空集**为0/81。unknown 不计成几何超预算或通过。
- trace 转向counter为262、274、282、298、309、320、331、349。动态/转向分层沿固定规则，不从成功帧重新挑窗口。

### 多frame的真实例子（不硬选）

| capture index | 参考 f | 候选 memory frames | 相对frame集合 | 相对memory误差xy区间 |
|---|---:|---|---|---|
|244、245|343|343,351|{0,8}|x=[0,0], y=[0,0]|
|247、248|344|344,345,346,347,348,349,350|{0,1,2,3,4,5,6}|x=[0,0], y=[0,0]|
|249、250|345|344,345,346,347,348,349,350|{-1,0,1,2,3,4,5}|x=[0,0], y=[0,0]|
|251|346|344,345,346,347,348,349,350|{-2,-1,0,1,2,3,4}|x=[0,0], y=[0,0]|

以上7张候选残差均约0.242636px。343与351是**离散重复位置**，相对frame hull `[0,8]`绝不表示1..7也兼容。其余是静止重复位置。动态分类只看参考前一步，所以343/344仍动态；不能因标签“dynamic”便强迫唯一 frame。

## 未来误差膨胀标签：仅诊断比例

固定考察2/4/8px三种膨胀预算，规则是完整非空且 `max_(g∈C)||p_g-p_f||≤预算`，不是确认后挑更宽阈值。

本片段66个完整 q 的候选 memory **恰好全部与参考位置相同**，最大相对位移0px；故三个膨胀预算结果相同：66/81=81.48%高置信、66/289=22.84%全截图可进入未来 player-only 标签候选诊断；动态63/78=80.77%，动态转向28/32=87.50%。完整非空内部是66/66=100%，**不是全数据100%可用，也不意味着2px已校准为真实误差保证**。

这些比例只说明冻结诊断的空间歧义不大于给定预算。**2px残差阈值是候选筛选定义，不是已被独立验证的真实误差界；不能从筛后的66张兼容位置/位移再自证误差已界定。真实frame可能因定位误差、模板误识别或窗口截断而根本不在候选集合中；因此不能据此自动进行label inflate。** 像素残差、定位偏置/身份不确定性、memory读数延迟、窗口外相位、bulletpool时相均未被证明；不能直接生产安全训练标签/生产cache，也不能令phase或正式pixel gate true。未来需要新片段冻结验证，并独立处理bulletpool对应，再讨论标签的额外误差膨胀。只有player空间多解重合，无法据此把bulletpool对齐到该frame。

## 测试与复现

新测试覆盖精确2px闭边界、17帧逐点比较、静止17解、非相邻重复位置保留离散集合、缺帧/微小memory冲突/非有限值unknown、不稳定bracket、窗口边缘、全pairwise pixel duplicate冲突、一致duplicate逐capture保留、空集非vacuous ±1、开发冻结不准确认拟合、配置篡改拒绝、索引对应及输出拒绝覆写。

```powershell
python -m pytest touhou_rl/tests/test_tactical_alignment_intervals.py touhou_rl/tests/test_tactical_pixel_bank.py touhou_rl/tests/test_tactical_pixel_probe.py -q -p no:cacheprovider --basetemp .tmp/alignment-round7-regression-x1
```

结果：**35 passed**，使用全新 basetemp，无旧pytest缓存依赖。

首次运行顺序：

```powershell
python -m touhou_rl.tactical_alignment_intervals freeze --development Document/Update/Alpha/v3.7.3/evidence/pixel-bank-round6/dev-round5-x1.json --output Document/Update/Alpha/v3.7.3/evidence/alignment-round7/prereg.json
python -m touhou_rl.tactical_alignment_intervals analyze --prereg Document/Update/Alpha/v3.7.3/evidence/alignment-round7/prereg.json --bank Document/Update/Alpha/v3.7.3/evidence/pixel-bank-round6/confirmation-round6-x1.json --trace Document/Update/Alpha/v3.7.3/evidence/capture-round6/trace.json --output Document/Update/Alpha/v3.7.3/evidence/alignment-round7/confirmation-analysis-x1.json
```

上述产物已存在，命令将拒绝覆写；复现必须用新输出路径，不删除旧证据。未启动游戏、未训练、未改旧模块。
