# 连续像素独立确认动作协议（续轮6）

在新capture-round6采集与查看图像前登记。旧capture-round5已用于开发，不把其43可信定位当留出结果。

- 独立确认使用同一团队/难度/关卡/截图后端/无bomb；不同非周期动作序列。
- 数据目录固定为`Document/Update/Alpha/v3.7.3/evidence/capture-round6`，保存290左右高频观察但不预设必须数量；窗口固定122逻辑帧，死亡/前台失去立即中止。
- 新动作sequence由v373_capture_probe的confirmation协议定义并随report保存；不依据定位输出挑片段。
- 多模板bank方案、阈值、代码SHA先冻结，再运行独立确认；不根据确认结果调参。若失败，只交付unknown/失败证据并另开后续开发修订，不重写结果。
- 像素位置来源仍含初始特权模板，不作为纯部署视觉。counter候选lag与真实Present相位认证分开，real_pixel_gate不能由NCC分数直接开启。
- 模板开发采集与确认session不称同世界；只是截图时序工程检查，不是策略生存配对。
