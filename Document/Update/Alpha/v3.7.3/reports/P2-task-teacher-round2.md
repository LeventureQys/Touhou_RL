# P2 task-teacher round2：冻结F4/F5开发任务规则教师

2026-11-01。只新增[受测模块](<../../../../../touhou_rl/tactical_task_teacher.py>)和[根测试](<../../../../../tests/test_tactical_task_teacher.py>)及本轮证据/报告；不改旧task契约或其他旧模块，不游戏输入、不训练、不生成正式监督。

## 结果

| 冻结任务 | 教师目标成功 | 生存 | 独立solve可行正例 | 失败 |
|---|---:|---:|---:|---:|
| F4 wait_pass | 6/6 | 6/6 | 6/6 | 0 |
| F4 immediate_exit | 6/6 | 6/6 | 6/6 | 0 |
| F5 narrow_gap | 6/6 | 6/6 | 6/6 | 0 |
| F5 scattered | 6/6 | 6/6 | 6/6 | 0 |

[汇总](<../evidence/task-teacher-round2/summary.json>)记录源码SHA、预注册SHA与每族计数；每实例JSON保存decision、物理目标区域、触发/终止/失败、完整2帧执行轨迹和actual_hold、位移及首次到达、独立参考。所有24例均保留，formal_training_labels始终false。本轮无实际教师失败；测试故意制造的失败验证保存机制，不把它计入开发结果或制造失败率。

**预注册时序披露：** [预注册](<../evidence/task-teacher-round2/preregistration.md>)及[holdout结构设计](<../evidence/task-teacher-round2/holdout-design.json>)在24例评估CLI前保存。但此前36项单测已参数化跑过全部24例闭环，已知其开发成功，并运行过单一实例的故意失败保存测试；原预注册对先验暴露的表述不够充分，现保留原文及SHA，在[execution补记](<../evidence/task-teacher-round2/execution.json>)明确披露。因此本轮是开发规则冻结后的复验，**不是结果盲预注册、独立确认或泛化成绩**，不能用“预注册”措辞掩盖开发暴露。

## 规则与隔离

- F4不看subtype来选择等待或立即撤离：从当前状态检查最大速度直线走廊的未来时间占用；若当前可通立即走，否则检查安全静止等待后的首个可通决策时刻。每2帧重算，不缓存硬编码序列。wait_pass会在frame14启动，因为到达危险带时已过frame16；并非按诊断臂固定等到16。已经越过关闭通道的状态也不会因subtype而重启等待。
- 走廊检查只模拟单一不分支直线规则，非solve/诊断动作或穷举搜索；无见证不推导无解，不声称可靠最迟启动上界。未来危险是当前特权task给出的已知模型，不是像素可见未来。
- F5先逐逻辑帧检查每个候选的2帧碰撞及局部位移界，安全候选中再按朝目标速度1的局部偏好选择；危险优先，可临时远离目标。无安全候选记中断/unknown并保留失败，不当不可行证据。
- 受测decide/rollout完全不调用solve、verify_route、diagnostic_actions；仅独立评估在rollout完成后运行verify_route及solve。单测把三者monkeypatch成抛错，24例闭环仍通过；另修改dangerous但不改subtype，验证等待/立即选择随危险时间改变。
- 模型未知、已碰撞、时域结束但目标未到均显式非成功；F5位移超界也不能仅以生存结算。评估拒绝覆盖已有结果，防止后续修订抹除失败。

## SVG与逐决策证据

横轴synthetic x、纵轴逻辑帧向下：红色危险、绿色目标、蓝色教师、灰色独立BFS参考，右侧每2帧标选项及动作。图不是TH08像素或实机回放。代表实例：

- [F4等待JSON](<../evidence/task-teacher-round2/dev-task-wait_pass-00.json>) / [SVG](<../evidence/task-teacher-round2/dev-task-wait_pass-00.svg>)。
- [F4立即撤离JSON](<../evidence/task-teacher-round2/dev-task-immediate_exit-00.json>) / [SVG](<../evidence/task-teacher-round2/dev-task-immediate_exit-00.svg>)。
- [F5窄缝JSON](<../evidence/task-teacher-round2/dev-task-narrow_gap-00.json>) / [SVG](<../evidence/task-teacher-round2/dev-task-narrow_gap-00.svg>)。
- [F5零散危险JSON](<../evidence/task-teacher-round2/dev-task-scattered-00.json>) / [SVG](<../evidence/task-teacher-round2/dev-task-scattered-00.svg>)。

其余20例同目录均有完整JSON及SVG，不只选成功片段。没有原视觉B0比较，灰色参考不是B0。

## 命令与退出码

仓库根运行，新basetemp首次使用，禁用cacheprovider；未删除或修复旧缓存。

```powershell
python -m pytest tests/test_tactical_task_teacher.py -q -p no:cacheprovider --basetemp .tmp/task-teacher-round2-20261101-a
# exit 0; 36 passed in 0.14s（预注册之前，开发先验暴露）

python -m touhou_rl.tactical_task_teacher --output Document/Update/Alpha/v3.7.3/evidence/task-teacher-round2
# exit 0; 教师24/24目标成功；solve 24/24；预注册之后

python -m pytest tests/test_tactical_tasks.py tests/test_tactical_task_teacher.py touhou_rl/tests/test_tactical_scenarios.py touhou_rl/tests/test_tactical_teacher.py -q -p no:cacheprovider --basetemp .tmp/task-teacher-round2-20261101-b
# exit 0; 77 passed in 0.47s
```

测试覆盖闭环全40帧目标与独立重放、mirror/offset等变、当前状态重评估、危险时间而非subtype硬编码、F5第二逻辑帧碰撞否决及背离目标避险、未知模型/已碰撞/未达目标、无局部见证非无解、失败产物保留/SVG和覆盖拒绝、预注册缺失拒绝。全部命令详见[execution](<../evidence/task-teacher-round2/execution.json>)。

## holdout封存状态与结论边界

[设计文件](<../evidence/task-teacher-round2/holdout-design.json>)SHA256：`881c4a1cb80bb558a1f5dc2becac2ba064b6be2b114cf6e4beb1970e3a1a3d66`。仅封存未来结构意向：validation双交替门/移动安全口袋；test重开多门与中间避难点/交错局部障碍岛。不是镜像或偏移；所有现开发模板及派生变体仍归development。未来必须由独立保管者在另行授权的契约下落实精确参数/seed/hash并审查结构独立性，才形成真正可执行封存集。

**pending_task_contract_development_only**：当前Task仅development、固定4种几何，seed和template_group不改变危险拓扑，不能创建独立holdout。没有实例化、solve、verify_route或教师评估这些未来模板；本轮不修改task契约，也不假称已封存可执行独立测试实例。

24例实际只有4个固定危险几何的镜像/偏移开发派生。结果支持受测规则教师在这些冻结玩具任务上闭环控制，不支持真实窄缝控制非劣、F1/F2/F3/F6、像素泛化、完整P1/P2或正式训练监督。环境转移与参考重放共享契约，不是独立物理形式证明。
