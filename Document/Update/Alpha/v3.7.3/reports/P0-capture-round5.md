# P0连续像素采样（续轮5）

## 实际运行

```powershell
python -m tools.rl.v373_capture_probe --help
python -m tools.rl.v373_capture_probe --confirm-foreground --out Document/Update/Alpha/v3.7.3/evidence/capture-round5
```

退出码0。用户授权范围内执行非周期转向/停顿的小范围普通动作，RAM缓冲RGB与状态，结束关闭游戏/释放按键后才压缩NPZ落盘。未用PNG编码阻塞实时采样；未改默认GDI/执行循环、未改游戏速度、未跑策略模型。采样前等待alive，任何death/前台丢失/回退即中止，本轮完整结束。

## 结果

[原始报告](<../evidence/capture-round5/report.json>)、[trace](<../evidence/capture-round5/trace.json>)、[图像归档](<../evidence/capture-round5/frames.npz>)。

- 290张640×480 RGB截图，覆盖122个不同游戏计数。
- 244次捕获前后frame相同，其余跨frame，全部保留区间。
- 全图122种hash，168次相邻完全重复；这是高于游戏帧率观察产生的现象，**不能凭此直接证明新鲜或冻结**。
- 本次状态trace内没有同gameplay_frame出现不同位置；只限此trace，不抵消源码发现的潜在帧内相位风险。
- 捕获含RGB连续复制中位2.46ms、最大6.56ms；未包括完整模型和输入发送。
- near-black检测0张；这也不证明可见对象同步。

[字节检查](<../evidence/capture-round5/byte-audit.json>)由离线读取NPZ与trace实算。所有phase_verified仍False、alignment仍unknown，未将本轮归档包装为训练事件。

## 验证

```powershell
python -m pytest touhou_rl/tests/test_tactical_capture_audit.py touhou_rl/tests/test_tactical_motion_probe.py -q -p no:cacheprovider --basetemp .tmp/v373-round5-capture-unit-01
python -m compileall -q tools/rl/v373_capture_probe.py
```

退出码0，9测试通过。下一步一次特权模板提取后全playfield独立像素定位，报告confidence/unknown与counter-based候选lag，不用当前内存附近小窗把结果强制拉向lag0。定位器本身不等于部署视觉模型。
