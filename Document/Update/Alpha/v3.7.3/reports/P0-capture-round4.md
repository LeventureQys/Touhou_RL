# P0截图新鲜度只读审查与离线估计器（续轮4）

## 已证实的工程边界

- [GDI实现](<../../../../../touhou_rl/winapi.py#L349-L353>)记录过窗口DC冻结造成52,888重复样本的历史；BitBlt成功不保证新图像。[当前grab](<../../../../../touhou_rl/winapi.py#L444-L457>)没有Present fence或图像帧戳。
- 游戏[main循环](<../../../../../third_party/th08/src/main.cpp#L338-L368>)先calc，再条件draw，再Present；逻辑frame在calc内推进，且自机更新可能晚于计数器。截图前后frame相同既不是状态原子性证明，也不是已显示该帧的证明。
- [自机绘制](<../../../../../third_party/th08/src/Player.cpp#L1514-L1519>)用field origin加player position；原生640×480客户区的playfield ROI为(32,16,384,448)。不同尺寸/DPI不静默缩放认证。
- GdiFlush只能保证当前线程GDI batch与CPU读DIB的顺序，不能变成D3D/DWM同步保证。grab后再flush已经复制出的image无效；未来若用独立审计wrapper，应BitBlt→flush→复制DIB，或丢弃首次grab副本后flush再复制。默认capture本轮未改。

以上源代码审查不是运行二进制exact-match或图像实测通过。当前模型read_image不支持图像，未完成肉眼复核。

## 最小后续实测方案

独立采样器以RAM缓冲采样，不同步写PNG；保存截图时钟区间、frame/player/frame连续读、窗口尺寸/可见性及RGB。原生RGB数组需copy成contiguous，避免BGRA倒通道负stride视图失效。

离线独立多模板定位自机像素，搜索整个playfield或轨迹包络，不能只搜当前内存附近强行得到lag0。先在独立开发段固定sprite anchor偏差；低置信/多峰/闪烁/死亡/帧内位置变化均unknown。选择含非周期转向/启停/双轴运动的段，扫描lag[-8,8]，缺帧不插值；候选含>1帧不得通过±1门。全屏hash变化仅证明变化，不证明自机新鲜。

## 本轮实际代码与验证

新增[离线lag估计器](<../../../../../touhou_rl/tactical_capture_audit.py>)：消费已独立定位的像素位置与已确认phase的memory轨迹；同时拟合固定sprite anchor以揭示恒速/静止不可辨识，输出candidate lag集合而非强行argmin。**没有实现真实像素检测器/采样器，没有真实数据同步结论。**即便工程输入识别lag在±1，real_pixel_gate_passed恒为False。

```powershell
python -m pytest touhou_rl/tests/test_tactical_capture_audit.py touhou_rl/tests/test_tactical_motion_probe.py -q -p no:cacheprovider --basetemp .tmp/v373-capture-round4-01
```

退出码0，9passed in0.15s。覆盖非周期轨迹注入lag -1/0/1/3、正负符号、+3不得过一帧预算、恒速/静止不可辨识、phase未知不虚构插值。更多真实定位/周期模糊/冻结DC测试仍待后续实测采样器实现。
