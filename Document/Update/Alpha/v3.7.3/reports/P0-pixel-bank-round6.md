# P0 privileged pixel bank（round6）

## 定性结论与边界

**仅离线特权模板诊断，不是正式 pixel sync，不是部署视觉，不是 sprite 身份证明。`real_pixel_gate_passed=False`，display phase 与最终同步始终 unknown。** 本子任务没有发送游戏输入；只读取旧开发归档，以及主线冻结后另外采集的独立确认归档。

新增[独立 bank 模块](<../../../../../touhou_rl/tactical_pixel_bank.py>)与[测试](<../../../../../touhou_rl/tests/test_tactical_pixel_bank.py>)；旧[单模板模块](<../../../../../touhou_rl/tactical_pixel_probe.py>)不修改。复用其 full-playfield NCC 和 `counter_lag_diagnostic`，不调用旧 CLI，不覆写 round5 的任何输出。

## 先冻结，再开发运行

顺序：完成实现与合成测试 → `freeze` 将[prereg](<../evidence/pixel-bank-round6/prereg.json>)、[config](<../evidence/pixel-bank-round6/bank-config.json>)、[source SHA](<../evidence/pixel-bank-round6/source-sha.json>)写入新目录 → 从旧开发前60逻辑帧按固定 slots 提取 bank → [freeze manifest](<../evidence/pixel-bank-round6/freeze.json>)封存 → 通知主线可以独立采确认 → 运行旧 dev。**未根据 dev NCC 成功率改配置或换模板**。确认仅加载既有 bank，绝不从确认图像提取 anchor。

所有 JSON 用文件模式 `x`；bank archive 用 `xb`。freeze 要求目录不存在，run 在计算前拒绝既有输出。每次 run 验证 bank/config/prereg/anchor/source 的 SHA256；禁止静默覆写与确认期调参。

## 固定 anchor 与置信规则

旧 capture-round5 第一 counter=240，预定相对 offsets=`[0,10,28,39,48,57]`，均严格位于 `[0,60)`；与旧输入计划对应 stationary/right/left+up/down/right/stationary。每个 offset 只取 exact counter 的最低 capture index；即使该 capture 不稳定、越界或常量也不以另一个 capture 替补。选点不计算 NCC。模板最多6个，24×24，memory+origin=(32,16) 仅用于这6次 privileged crop。

| offset | counter | capture index | input |
|---:|---:|---:|---|
|0|240|0|stationary+z|
|10|250|24|right+z|
|28|268|66|left+up+z|
|39|279|93|down+z|
|48|288|115|right+z|
|57|297|136|stationary+z|

6 slots 全部稳定且非常量，详见[anchors 与每槽状态](<../evidence/pixel-bank-round6/anchors.json>)。模板与身份未人工确认，不把 privileged crop 视作 sprite 语义标注。

- 每张图、每模板 raw+gradient 均搜索固定全 playfield `(32,16,384,448)`；定位 API 不接收当前 memory/state。
- 保持旧阈值 NCC≥0.65、远处峰差≥0.06、raw/gradient距离≤3px，没有下调。
- 至少2个双模式合格模板；**所有**合格模板位置 pairwise距离≤3px，才输出 median candidate；任何合格模板冲突都 unknown，不挑最高分赢家。低conf模板保留逐模式证据，不算 support；不足2 support unknown。
- 与旧单模板相比，bank 本身不使用每帧 RAM state/focus来挑模板；不宣称已经覆盖 focus、死亡或透明闪烁。输入 capture 必须由主线维持 alive/focus边界，未来状态变化仍不可推广。
- dev 排除6个 anchor capture index。注意相同 counter 的重复 screenshot 仍可能存在，因此 dev 只是开发结果，不宣称独立图像证据；counter lag 内部会归并 duplicate counter。独立确认不共享这6张 anchor。

## Lag 与运动覆盖（固定）

调用原 `counter_lag_diagnostic(max_lag=8,min_samples=12)`。保持 stable bracket、冲突 memory counter 剔除、冲突 pixel duplicate counter 剔除、所有 lag 完全相同公共 counter、固定 anchor offset拟合、±0.35px候选与2px error规则。样本不足只输出探索性 scores，永不 gate true。

额外单独量化公共 counter 内 dynamic 样本（相邻唯一 memory position 位移>0.25px），至少12个 dynamic samples；由 trace 的非零方向变化定义 turn，停顿不重置前一非零方向；至少3个 turn 在±3 counter内有 dynamic common sample。不靠静止重复截图凑样本。即使满足，也仍是 counter candidate、phase unknown，不是正式同步。

## 开发与确认结果

[首次 dev 原始输出](<../evidence/pixel-bank-round6/dev-round5-x1.json>)：290 captures，排除6 anchor后91/284高置信，193 unknown；14个高置信 capture 因不稳定 bracket 被 lag 排除。36个全lag公共counter中34 dynamic、2 static；8个非零方向转向中4个在±3 counter内覆盖（285/312/321/347），达到固定运动覆盖条件。memory/pixel冲突counter均0。候选lag=[0]，RMSE=0.679381px，offset=(32,16)；+1 RMSE=2.154902，-1 RMSE=2.631831。**这是 counter candidate，不是同步=0，不是正式门。** 相同 counter 仍可能重复 anchor图像；dev不作独立确认。

[独立确认首次原始输出](<../evidence/pixel-bank-round6/confirmation-round6-x1.json>)：主线在 freeze 信号后按新非周期 schedule 另采 capture-round6，本子任务只离线消费。289 captures，81 confident、208 unknown；9个高置信capture因不稳定 bracket排除。公共counter=33，其中31 dynamic、2 static；8个非零方向转向中5个在±3 counter内覆盖（262/274/298/309/331）。memory冲突counter=0，pixel重复counter冲突=[303,336]，均保守剔除，未挑第一/最后。候选lag=[0]，RMSE=0.274160px，offset≈(32.05884,15.94116)；+1 RMSE=2.183553，-1 RMSE=2.817945。

| 固定指标 | old dev | independent confirmation |
|---|---:|---:|
| captures / excluded anchors |290 / 6|289 / 0|
| confident / unknown |91 / 193|81 / 208|
| 所有lag公共counter |36|33|
| dynamic / static common |34 / 2|31 / 2|
| covered turns / observed turns |4 / 8|5 / 8|
| counter candidate |[0]|[0]|
| candidate RMSE px |0.679381|0.274160|
| phase/status |unknown|unknown|
| formal real pixel gate |False|False|

**确认在固定配置下有足够动态/转向覆盖，并复现counter-based [0]候选；但正式同步结论仍unknown。** 208/289图像仍unknown，NCC没有sprite身份语义证明、跨模板与raw/gradient不独立、capture稳定RAM bracket没有Present fence或像素frame编号。因此不称正式pixel sync，更不能把counter候选包装成phase_verified或正式gate true。没有依确认结果改模板、阈值、源码或覆盖标准；首次结果即保留。

确认命令（相同冻结bank，输出x不覆盖）：

```powershell
python -m touhou_rl.tactical_pixel_bank run --frames Document/Update/Alpha/v3.7.3/evidence/capture-round6/frames.npz --trace Document/Update/Alpha/v3.7.3/evidence/capture-round6/trace.json --bank-dir Document/Update/Alpha/v3.7.3/evidence/pixel-bank-round6 --role confirmation --output Document/Update/Alpha/v3.7.3/evidence/pixel-bank-round6/confirmation-round6-x1.json
```

所有输入SHA与freeze SHA均在两份原始结果内。freeze、dev、confirmation三命令均退出0；旧单模板源码 `git diff --stat`为空。

## 测试

```powershell
python -m pytest touhou_rl/tests/test_tactical_pixel_bank.py touhou_rl/tests/test_tactical_pixel_probe.py -q -p no:cacheprovider --basetemp .tmp/pixel-bank-round6-tests-x1
```

**23 passed in 2.88s，退出0。** 新bank 9例 + 旧probe 14例；覆盖 exact slot无替补、duplicate counter不能择优、全场远距离定位、至少双模板支持、跨模板矛盾unknown、远处重复峰unknown、动态/至少3转向与静止堆样本拒绝、anchor排除/phase unknown、阈值冻结与原尺寸、输出拒绝覆写。

## 复现命令

```powershell
python -m touhou_rl.tactical_pixel_bank freeze --frames Document/Update/Alpha/v3.7.3/evidence/capture-round5/frames.npz --trace Document/Update/Alpha/v3.7.3/evidence/capture-round5/trace.json --out Document/Update/Alpha/v3.7.3/evidence/pixel-bank-round6
python -m touhou_rl.tactical_pixel_bank run --frames Document/Update/Alpha/v3.7.3/evidence/capture-round5/frames.npz --trace Document/Update/Alpha/v3.7.3/evidence/capture-round5/trace.json --bank-dir Document/Update/Alpha/v3.7.3/evidence/pixel-bank-round6 --role development --output Document/Update/Alpha/v3.7.3/evidence/pixel-bank-round6/dev-round5-x1.json
```

上述产物已存在时命令会拒绝覆写；不是通过清空旧目录重跑的指引。
