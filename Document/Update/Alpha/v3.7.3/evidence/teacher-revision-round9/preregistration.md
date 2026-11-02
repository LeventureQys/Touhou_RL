# R9 preregistration — 2026-11-01

先登记后实现/rollout。只新增tactical_teacher_v2.py、新tests及本R9证据/报告，不改旧teacher/world/reference。R8全部18 development+24已暴露validation作为开发回归，不称独立泛化；原R8 test24条不得构造/执行/修改。

v2协议：F1读取coverage/gap/internal_gap/wall_frame，可靠gap和左右外出口按几何距离/到达时间选目标，禁止旧常量识别gap。每次先验证完整2逻辑帧。F3依据实际window/threshold/goal规划必要左移，不为避碰永远放弃目标；短期碰撞优先，以实际自机位置的发射因果模拟有限候选（每一首动作后最多12逻辑帧、每步五动作的受限beam=20，目标规则尾随）；F2可复用发射后微移规则，碰撞优先。不得调用reference solve/verify挑动作，有限搜索失败仅unknown不证明无解。短期安全与进度冲突优先安全并保留失败。未知模型维持unknown。

开发阶段允许根据已暴露R8调试，必须留存每次开发运行失败。开发结束后冻结v2/runner源码SHA及新确认manifest，之后不得改v2。新确认为每族4组镜像共24例：F1新coverage/gap/wall/start；F2新cadence/start/origin/burst；F3新cadence/window/threshold/goal/start组合。参数完整签名不得与R8任何split重复，新确认未在冻结前rollout。manifest拒绝覆盖。

评价：R8 solve有限223程序独立搜索，不接收teacher决策；每个返回正例verify完整160帧重放，旧teacher与v2亦独立完整重放。reference未找到即unknown；不把teacher失败当无解。逐事件保存完整动作、决策、轨迹、因果发射及碰撞采样帧。每族>=3 SVG对比v2/旧teacher/reference。开发与新确认分别汇总；所有失败保存，不根据新确认调参。source SHA before/after及原R8manifest SHA记录。测试用全新basetemp且禁pytest cache。

不产生正式标签、不启动游戏/像素/训练、不打开正式gate；没有B0、真实几何校准、精确latest界或统计泛化宣称。新确认是小型人为结构组合留出、镜像非独立样本。
