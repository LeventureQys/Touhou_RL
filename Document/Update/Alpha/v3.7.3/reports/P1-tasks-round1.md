# P1 tasks round1：受限 F4/F5 开发任务

2026-11-01。本轮只新增独立任务和测试，不修改现有 scenarios、teacher、reference、执行或训练模块。先阅读升级计划、版本 README、仓库 README、现有三个模块与场景/教师测试，再保存[预注册](<../evidence/tasks-round1/preregistration.md>)，之后实现和评估。未运行游戏、未生成正式训练标签、未打开验证/测试模板。

## 小契约及实现

- 新增[独立任务模块](<../../../../../touhou_rl/tactical_tasks.py>)、[测试](<../../../../../tests/test_tactical_tasks.py>)。
- x=0..40、y固定400，2逻辑帧/动作，40帧完整事件，动作每帧速度{-2,-1,0,1,2}；每个逻辑帧检测危险、碰撞累积不可消除。单位为合成坐标，不冒充已标定TH08像素。
- F4：从x10出发，经[18,22]通道，必须无碰撞到完整时域且最终进入[30,34]。等待任务通道1..16帧关闭；立即撤离反例通道1..10帧开放，随后关闭。停在起点虽可生存，永不记目标成功。
- F5窄缝：12..20帧仅[18,22]安全，最终目标[20,22]。零散危险：6..10帧[15,17]危险，14..18帧[19,21]危险，最终目标[24,26]。两者都要求全程最大位移≤10，不能以生存或大幅迁移冒充局部恢复。
- 每类3个坐标偏移×左右镜像，共24个development实例；镜像共享template_group。所有构造只接受development，不提供validation/test生成入口。
- 独立有限BFS以时间、位置和最大位移展开，预算100000，不读取现有教师/参考或诊断动作。搜索正例从初始世界完整重放；模型未知、预算不足、时域不完整均为unknown。碰撞或目标失败只拒绝本条见证，不推导无解。环境转移与重放共享小契约，故这不是对物理实现的独立形式验证。

## 预注册结果

以下均为目标成功数/6，不是分类准确率，也不是学得策略收益：

| 任务 | 独立搜索正例 | 静止 | 立即迁移 | 等待后迁移 | 局部恢复 |
|---|---:|---:|---:|---:|---:|
| F4 等待后通过 | 6 | 0 | 0 | 6 | 不适用 |
| F4 立即撤离反例 | 6 | 0 | 6 | 0 | 不适用 |
| F5 窄缝 | 6 | 0 | 0 | 不适用 | 6 |
| F5 零散危险 | 6 | 0 | 0 | 不适用 | 6 |

六项预注册任务构造检查全部true，有限搜索24/24获得正例。F4静止生存12/12、目标成功0/12；F5零散危险的立即大迁移生存6/6、目标成功0/6，证明这里明确区分生存与任务目标。每实例保存所有诊断臂、参考路线、每2帧状态、实际动作/hold、首次目标到达帧、最大位移及任务SHA，见[汇总](<../evidence/tasks-round1/summary.json>)及同目录逐实例JSON。

未换基线：冻结的hold、immediate_transit、wait_then_transit、local_recover保持预注册定义，首轮运行未作结果驱动修订。它们是新任务构造用诊断程序，不是原视觉B0；不能将参考搜索优于这些程序解读为教师或网络收益。

## 命令、退出码与测试

工作目录为仓库根；新basetemp均首次创建，禁用cacheprovider避开既有缓存权限问题，未删除或修复任何原缓存。

```powershell
python -m pytest tests/test_tactical_tasks.py -q -p no:cacheprovider --basetemp .tmp/tasks-round1-20261101-a
# exit 0; 30 passed in 0.29s

python -m touhou_rl.tactical_tasks --output Document/Update/Alpha/v3.7.3/evidence/tasks-round1
# exit 0; 24 instances, 24 positive witnesses, 6 construction checks true

python -m pytest tests/test_tactical_tasks.py touhou_rl/tests/test_tactical_scenarios.py touhou_rl/tests/test_tactical_teacher.py -q -p no:cacheprovider --basetemp .tmp/tasks-round1-20261101-b
# exit 0; 41 passed in 0.37s
```

机器可读记录见[execution](<../evidence/tasks-round1/execution.json>)。测试覆盖：镜像、确定性、逐逻辑帧碰撞/粘性、等待/立即撤离双向反例、F5局部恢复与位移约束、搜索正例完整重放、未知模型、零/小预算、未完成时域、非法动作/几何、development-only。

## 边界与后续

本轮补充的是有明确目标的受限玩具任务，不接入原教师、模型或正式监督。零散危险以时间-位置区间表示，不模拟完整子弹、激光、动态发射器或二维精控；横向固定y世界不能证明真实窄缝控制非劣。独立搜索仍使用同一环境转移，不作无解证明或可靠最迟启动上界。

尚未封存独立验证/测试模板、进行双人标签复核、原视觉策略公平比较、每族三段可视回放或真实像素同步确认。**完整P1/P2均未过门**；本轮只验证F4/F5任务构造与正例重放，不应导出正式成功标签。下一步应在这些冻结目标上接入受测教师并记录失败，再由主负责人安排独立模板封存及真实对应场景，不将这里的24/24搜索见证当教师24/24。
