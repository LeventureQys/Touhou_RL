# P1 template round3：独立结构模板与冻结评估

日期：2026-11-01。**仅合成，不游戏、不训练；未通过或宣称通过完整P1/P2出口。**

## 1. 预注册与改动范围

已先读取[既有世界](<../../../../../touhou_rl/tactical_tasks.py>)、[既有教师](<../../../../../touhou_rl/tactical_task_teacher.py>)、[round2 holdout设计](<../evidence/task-teacher-round2/holdout-design.json>)与[升级计划](<../升级执行计划.md>)，先写[本轮预注册](<../evidence/template-round3/preregistration.md>)后实现和评估。

新增[独立模板模块](<../../../../../touhou_rl/tactical_task_templates.py>)及[模板单测](<../../../../../tests/test_tactical_task_templates.py>)。**范围例外说明**：原要求“勿改现有模块”，主代理随后明确授权在[世界dangerous接口](<../../../../../touhou_rl/tactical_tasks.py#L88-L91>)增加唯一两行duck-type hook，任务提供dangerous方法则转发；其余世界和教师不改。该例外已在评估前更新预注册。没有动态函数克隆、全局monkeypatch或teacher调参。

TemplateTask兼容现有start_x、canonical_x、physical_x、target、family、model_known、horizon；DangerRegion参数化闭区间时间和空间，outside支持局部safe gate。直接复用原world.advance、goal_reached、verify_route、solve和原teacher.rollout；保持动作-2..2、2帧、边界0..40、F5累计最大位移≤10。未知模型与有限搜索无见证仍报unknown，不判不可行。

本轮结构设计是新授权合同，不将round2未实例化的moving/reversing pocket、reopening-with-retreat建议说成已完成。

## 2. 结构拆分与封存

| split | 内容 | 实例 | 结构group |
|---|---|---:|---:|
| development | 精确复现single wait/exit gate、single narrow safe gate、scattered pair | 24 | 4 |
| validation | F4 double blocked gates（两个时间/空间间隔及目标区）；F5 double sequential safe gates（两个间隔/目标区） | 24 | 2 |
| sealed test | F4 staggered narrow gates三段；F5多段局部区域四段 | 12 | 2 |

参数变体不冒充独立拓扑；同一结构的时间/目标参数、所有镜像与offset(-1,0,1)留在同group和同split。结构独立性基于单段→双段→三/四段，不基于改名或seed。**仅8个结构group而非60个独立模板；样本相关，不能当60个独立泛化证据。**

[完整manifest](<../evidence/template-round3/manifest.json>)包含精确seed、目标、危险区域、group、变换和各项hash；[hash清单](<../evidence/template-round3/hashes.json>)含source/预注册/manifest。test始终仅序列化元数据；没有instantiate、危险查询、advance、goal、reference或teacher调用；接口不提供test执行开关，instantiate拒绝test。

manifest SHA256：`9050ba1699dbb3cc0f3bbb23f6c01ec4f8e7bdb4ea04c35a02bdd73865835b6a`。

## 3. 冻结评估结果

先teacher rollout与路线重放，再独立BFS参考（100000扩展预算）；只跑开发和验证，未依验证修teacher/模板或阈值。每条事件保留动作、候选、决策、终止、失败、逐步轨迹、参考见证与hash；结果拒绝覆盖。

| split / family | goal成功 | 完整生存 | 参考可行见证 | 失败 |
|---|---:|---:|---:|---:|
| development F4 | 12/12 | 12/12 | 12/12 | 0 |
| development F5 | 12/12 | 12/12 | 12/12 | 0 |
| validation F4 | 12/12 | 12/12 | 12/12 | 0 |
| validation F5 | 12/12 | 12/12 | 12/12 | 0 |

[评估汇总](<../evidence/template-round3/summary.json>)记录两split的严格诊断条件均满足，但这**不是P1/P2全门**。本次没有验证失败，未杜撰失败；程序对未来失败保留原始结果，不覆写或降门。

评估前后source hash一致；teacher source SHA256为`689471ab6e1848c2e5a2c9b4cb413b0adb3bfa74d72e2a2970f16a6128541cdd`。world source hash对应授权hook之后冻结的源文件，不表示它相对round2毫无改动。

## 4. 接口与命令结果

```powershell
python -m touhou_rl.tactical_task_templates --output Document/Update/Alpha/v3.7.3/evidence/template-round3
# exit 0；development 24/24，validation 24/24；test_executed=false
# 相同输出目录再次调用将拒绝覆盖。新实验目录须先有preregistration.md。

python -m pytest tests/test_tactical_tasks.py tests/test_tactical_task_teacher.py tests/test_tactical_task_templates.py -q -p no:cacheprovider --basetemp .tmp_template_round3_regression
# 72 passed in 0.47s；exit 0
```

新单测独立运行6 passed in 0.11s；测试封存部分只做manifest拆分/hash完整性，不执行封存世界。首次未指定basetemp时5 passed/1 fixture error：系统Temp pytest目录Windows权限拒绝；完整记录保留于[命令结果](<../evidence/template-round3/command-results.md>)。未改权限；随后使用新工作区临时目录。所有命令前台完成，无后台任务。

## 5. 边界与下一步

本轮仅固定y=400的一维合成单位、已知几何/时间特权信息，无像素、真实弹池或游戏动力学证据。双门成功是这个固定合同的结构外推诊断，不是完整真实游戏泛化，也不是统计独立确认。F4双blocked gate可等待后直接通过，不宣称强制经过中间safe pocket或实现round2交替门原提案。

未覆盖F1/F2/F3/F6、状态依赖发射、B0、真实回放、统计非劣、P0真机校准、正式监督与完整P1/P2出口。封存test仍未评估，不得引用为成功证据。下一步如需更复杂反向/中间等待要求，另行预注册新结构合同，不能回调本轮验证或解封测试调teacher。
