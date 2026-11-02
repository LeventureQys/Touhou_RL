# P0修复后真机核验与因果数据审计（续轮2）

## 真机核验

在已有短校准授权内运行：

```powershell
python -m tools.rl.v373_calibrate --confirm-foreground --out Document/Update/Alpha/v3.7.3/evidence/live-calibration-03
```

退出码0，21.87秒；出生后等待alive，动作前中立输入4帧，10组×8动作，普通/聚焦/斜向/聚焦斜向各20次。已释放按键、关闭本次游戏。[原始报告](<../evidence/live-calibration-03/report.json>)不可覆盖。

修复后engine current不再恒0：5161个动作轮询全部非零，5153/5161与GUI current一致，其余8次是帧内更新相位差，不能强行视作不同输入失败。GUI previous也能观察到真实上帧按键，不再读padding。

稳定按键匹配、player_state=0、相邻单位帧的速度分别4.0、2.0、约4.0、约2.0px/帧（36/36/33/33个转移），未观测到这些子集的速度分散。

**重要失败边界：本轮窗口末累计1死，34个trial端点包含非alive状态。**原脚本仅用死亡计数是否在短trial内变化作为排除，因死亡计数延迟更新而漏掉这些情况。不能把原报告中的no_death_trials=20理解为20次合格动力学重复。已修复后续采集脚本，追加invalid_player_state排除；旧报告仍原样保留。未重新开长采集掩盖污染。

```powershell
python -m touhou_rl.tactical_calibration --source Document/Update/Alpha/v3.7.3/evidence/live-calibration-03/report.json --out Document/Update/Alpha/v3.7.3/evidence/live-calibration-03/input-analysis-v2.json
```

退出码0，分析使用alive+正确实际消费按键的逐帧子集，80次均得到生效区间。详见[分析v2](<../evidence/live-calibration-03/input-analysis-v2.json>)。P0仍缺边界、释放/沉降完整区间及渲染新鲜度，不能宣称精确数据门通过。

## 因果数据审计接口

新增[数据审计](<../../../../../touhou_rl/tactical_data.py>)及[测试](<../../../../../touhou_rl/tests/test_tactical_data.py>)，要求事件inputs与labels分离；拒绝未来/特权部署输入、未验证alignment、不完整弹池、尚未生效动作、缺失实际hold、事件内时钟回退、重复event、SHA漂移、跨split同模板。

目前仅结构检查，不能用一个verified字符串独立认证像素或教师；完整数据验收仍需来源证明。没有把校准截图打包冒充P3事件，没有生成正式训练标签。

```powershell
python -m pytest touhou_rl/tests/test_tactical_data.py touhou_rl/tests/test_tactical_calibration.py touhou_rl/tests/test_tactical_input_offsets.py -q -p no:cacheprovider --basetemp .tmp/v373-round2-gates-02
python -m tools.rl.tactical data-audit --help
```

退出码0，9测试通过；data-audit具备实际help，可审计指定manifest并在失败时返回2和落盘错误。尚无正式manifest可通过。
