# P0输入读回与动力学纠错（续轮1）

## 发现

初次校准保存的input_mask在所有3625个轮询中恒为0，不能证明输入生效。静态检查发现[读取实现](<../../../../../touhou_rl/state.py#L409-L412>)从LAST_FRAME_INPUT读取4字节，再以两个连续u16解析；而[地址定义](<../../../../../touhou_rl/spec.py#L28-L34>)的CUR_FRAME_INPUT与LAST相隔4字节。因此第二个u16并不是声明的当前输入地址。这是已有读取链路的明确风险，本轮未覆盖旧工作树文件。

此外，游戏[Player消费通道](<../../../../../third_party/th08/src/Player.cpp#L618-L639>)是GUI输入；新增校准中显式保存gui_input_current/previous。旧harness假环境测试通过不能证明它读取的真实通道正确。正式P0冻结前必须由唯一负责人修复并回归该已有路径，不能默默绕过后声称旧共同管线已过门。

## 实际复测

用户先前明确允许本轮前台短试，复测同样10组/60次，不扩大为长时间性能试验。

```powershell
python -m pytest touhou_rl/tests/test_tactical_calibration.py -q -p no:cacheprovider --basetemp .tmp/v373-calibration-analysis-02
python -m tools.rl.v373_calibrate --confirm-foreground --out Document/Update/Alpha/v3.7.3/evidence/live-calibration-02
python -m touhou_rl.tactical_calibration --source Document/Update/Alpha/v3.7.3/evidence/live-calibration-02/report.json --out Document/Update/Alpha/v3.7.3/evidence/live-calibration-02/input-analysis.json
```

全部退出码0；测试3通过。真机10.77秒，零死亡，已释放按键并关闭游戏。GUI通道60/60观察到输入从不匹配到匹配的区间，而非用首次匹配伪造单点延迟。

## 为什么第一轮速度不能冻结

首轮普通/聚焦3.0/1.5是四帧发送窗口的位移平均，混合了输入尚未生效的帧。复测中对实际GUI按键匹配、连续单位帧且player_state=0的子集分析，普通/聚焦/斜向中位为4.0/2.0/约4.0px每帧。普通仍有0位移异常；此子集仅9/12/12个单位帧转移，不是每类别20次独立稳定校准，不能据此冻结动力学。

两个结果并不矛盾：发送窗口平均包含延迟，稳定生效子集更接近实际速度。不能把旧动作延迟简单吸收到降低运动速度里，那会使最迟撤离时间标签系统性错误。

## 已实施最小修复

主agent作为本轮唯一读取路径负责人，读过现有文件后仅定点修改[state.py](<../../../../../touhou_rl/state.py>)：engine当前输入按+4读取，GUI previous按+4读取并把GUI缓冲扩到8字节；没有改执行循环、没有覆盖已有其它修改。新增非零padding反例单测，避免全零假内存掩盖错误。

```powershell
python -m pytest touhou_rl/tests/test_tactical_input_offsets.py touhou_rl/tests/test_state_entities.py touhou_rl/tests/test_experiment_harness.py touhou_rl/tests/test_tactical_calibration.py -q -p no:cacheprovider --basetemp .tmp/v373-offset-fix-01
```

退出码0：38 passed in1.07s。上述复测02发生在修复前，GUI current当时本已正确，因此其GUI分析有效；engine current与GUI previous需要修复后的下一次真机核验，不追溯篡改旧报告。全量回归另记录。

全量回归也已执行：

```powershell
python -m pytest touhou_rl/tests tests/test_tactical_tasks.py -q -p no:cacheprovider --basetemp .tmp/v373-round1-full-01
```

退出码0：1365 passed，1 xfailed，4 subtests passed，55.47s。

[复测原始报告](<../evidence/live-calibration-02/report.json>)、[输入分析](<../evidence/live-calibration-02/input-analysis.json>)。尚待修复后真机读取复核、边界/聚焦斜向、沉降/释放间隔、渲染新鲜度验证。P0仍部分完成，P3精确监督禁止开启。
