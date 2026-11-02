# P0左/上/下边界及反转（续轮8）

实际命令：

```powershell
python -m tools.rl.v373_motion_probe --confirm-foreground --protocol boundaries --out Document/Update/Alpha/v3.7.3/evidence/boundaries-round8
python -m pytest touhou_rl/tests/test_tactical_motion_probe.py -q -p no:cacheprovider --basetemp .tmp/v373-boundaries-round8-tests-01
python -m tools.rl.v373_motion_probe --help
```

均退出0，测试2passed。真实游戏frame240→536，零死亡、全程alive，失去前台/死亡/回退会立即中止；已释放按键并关闭本次游戏。Shoot开、bomb关，无模型部署，不减速游戏。

| 段 | 最终坐标 | 观测 |
|---|---|---|
| 左边界55帧 | (8,384) | x下界8饱和 |
| 向右反转8帧 | (36,384) | 恢复28px，含1帧生效滞后现象 |
| 上边界105帧 | (40,16) | y下界16，切方向时旧右动作多移动4px |
| 向下反转8帧 | (40,44) | 恢复28px |
| 下边界112帧 | (40,432) | y上界432饱和 |
| 向上反转8帧 | (40,404) | 恢复28px |

[原始trace](<../evidence/boundaries-round8/report.json>)、[逐段summary](<../evidence/boundaries-round8/summary.json>)。结合round4右边界376，本机体原生运动边界观测为x[8,376]、y[16,432]。每个边界仅一次饱和与反转，不冒充每类别20次误差分布；当前所有边界短试均非战术性能测试。

## 诊断动力学小契约

新增[运动包络](<../../../../../touhou_rl/tactical_motion_envelope.py>)：理想已生效动作按普通4/聚焦2px每帧、斜向归一化和已观测四边界裁剪；`old_action_carryover`对离散lag区间逐候选先执行旧方向、再执行新方向，输出端点集合，不把滞后平均成慢速新动作。它不是碰撞安全可达集，不修改现有teacher/执行循环。

```powershell
python -m pytest touhou_rl/tests/test_tactical_motion_envelope.py touhou_rl/tests/test_tactical_motion_probe.py -q -p no:cacheprovider --basetemp .tmp/v373-motion-envelope-round8-01
```

退出0，5passed in0.14s。轴切换(36,384)旧右1帧再上104帧→(40,16)，右边界旧右1帧再左7帧→(348,384)，与观测端点一致；这只是已观测样本拟合，不独立证明所有输入延迟恒1帧。

切轴后x多4px是实际旧动作保持证据，不能事后把上移起点改成40抹去输入滞后。正式教师的可达时间必须计入发送/生效区间与已执行历史。P0仍缺完整重复标定与像素—弹池对齐认证，不能因四边界已观测便宣布门通过。
