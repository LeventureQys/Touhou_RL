# P0两帧动作、释放与右边界短试（续轮4）

## 实际运行

```powershell
python -m tools.rl.v373_motion_probe --help
python -m tools.rl.v373_motion_probe --confirm-foreground --out Document/Update/Alpha/v3.7.3/evidence/motion-round4
python -m pytest touhou_rl/tests/test_tactical_motion_probe.py -q -p no:cacheprovider --basetemp .tmp/v373-motion-round4-01
```

全部退出码0，单测2通过。本轮新probe在每次read检查前台、alive、死亡计数、playing与clock；任何异常立即中止并释放按键，避免上轮死亡污染继续采集。Shoot开启、bomb关闭，不减速游戏，不运行模型。

## 观测

- 20次左右交替2帧命令，动作前中立3帧、释放后再观察3帧。
- 20/20观察到实际GUI输入生效，首次匹配均在发送后1逻辑帧；记录前一不匹配帧与首次匹配帧区间，不声称精确墙钟延迟点。
- 20/20观察到释放至仅shoot输入的变化。
- 右移55帧，到x376饱和，未越界；随即左移8帧到x348，符合一帧输入延迟后7帧×4px移动的现象。逐读回原始记录保留，不以端点推断独立精确更新相位。
- 全程alive、零死亡、时钟无回退；已释放所有按键并关闭本次游戏。

[原始report](<../evidence/motion-round4/report.json>)、[summary](<../evidence/motion-round4/summary.json>)。

## 回归

```powershell
python -m pytest touhou_rl/tests/test_tactical_motion_probe.py touhou_rl/tests/test_tactical_calibration.py touhou_rl/tests/test_tactical_input_offsets.py touhou_rl/tests/test_experiment_clock.py touhou_rl/tests/test_experiment_harness.py -q -p no:cacheprovider --basetemp .tmp/v373-motion-round4-regression-01
```

退出码0，45 passed in1.07s。

## 尚未证明

右边界仅一次推挤与反转，不是各类别20次边界重复；未测左/上/下边界，未把普通命令的延迟推广至全部聚焦/斜向模式。两帧命令存在生效滞后，不能直接把send frame视为action effective frame。P0完整动力学门仍未过。

截图同frame括号仍不是Present fence。现有GDI实现有历史冻结DC风险；源码时序显示逻辑counter更新早于条件Draw/Present，需要下一轮通过自机像素位移时间序列与状态轨迹配对标定。当前所有校准图仍不作正式精确几何监督。
