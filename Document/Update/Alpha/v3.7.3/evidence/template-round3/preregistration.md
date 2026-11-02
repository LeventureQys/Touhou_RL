# Template round3 预注册（评估前冻结）

日期：2026-11-01；范围：仅合成、无游戏IO、无训练、无正式监督标签。

## 合同与独立性

本轮新增独立模块；主代理在预注册后、评估前明确授权唯一例外：tactical_tasks.dangerous 开头添加 hasattr(task, 'dangerous') duck-type hook，转发 task.dangerous(frame,x)，其余世界代码与 tactical_task_teacher 不改。不动态克隆或 monkeypatch。Task duck typing 保留 canonical_x / physical_x / start_x / target / family / horizon / model_known 接口；动作、边界、2帧步进、目标和F5位移≤10不变。

round2 holdout-design 是未实例化的提案，本轮新合同替代其未来模板建议，不声称执行其原 moving/reversing pocket 或 reopening-with-retreat 设计。

## 冻结结构与精确参数

- development：既有 F4 single gate wait_pass/immediate_exit、F5 single narrow gate/scattered pair 的精确危险；各6个镜像/offset变体，共24。
- validation：F4 double gates 两组（空间段[18,22]/[26,28]，时间[1,16]/[23,40]；及[16,18]/[24,26]，时间[1,12]/[21,40]），目标分别[30,34]/[28,30]；F5 double safe gates 两组（时间[10,14]/[20,26]，safe[18,20]/[22,24]，目标[24,26]；以及[8,12]/[18,24]，safe[14,16]/[18,20]，目标[20,22]）。每组镜像×offset(-1,0,1)，共24。不同时间/目标参数不冒充独立拓扑：F4整个double-gate为同一结构group，F5同理。
- sealed test：F4 三段 staggered narrow gate，F5 四段局部危险区域；仅声明序列化参数、manifest和SHA256，禁止实例化、danger/advance/goal/solve/verify/teacher/rollout调用。每结构group六个镜像/offset变体，共12。测试只能检查封存manifest拆分和hash完整，不评估这些世界。
- 同一结构group所有镜像/偏移/时间参数变体不跨split。完整manifest固定seed、精确危险矩形、目标、结构group；hash采用sorted-key紧凑UTF8 JSON SHA256。

## 冻结评估

先生成manifest/hash，再评估开发+验证各一次；拒绝覆盖结果。teacher完全复用现有规则；开发和验证都仅评估，不调参、不按结果修teacher。每事件先teacher rollout及正例重放，再独立BFS，max_expansions=100000；有限搜索无见证只报unknown，不判无解。目标成功不能替换为仅生存。保留每次失败、完整动作/决策/轨迹、独立参考及hash。F4/F5各自报告成功数、生存数、参考可行数及失败，不合并声称通关；不根据参考行动给teacher供策。

本轮诊断门：各split每族goal_success全部成功，且参考可行全部见证；失败原样报告，不改门槛。此严格诊断不等于计划P1/P2正式出口；六族、真实游戏、B0比较、统计非劣和完整P1/P2均未覆盖。

## 接口与证据

`python -m touhou_rl.tactical_task_templates --output Document/Update/Alpha/v3.7.3/evidence/template-round3`：要求本预注册已存在；生成manifest+hash，开发和验证评估；没有test执行选项。

`python -m pytest tests/test_tactical_task_templates.py -q -p no:cacheprovider`：测试开发/验证兼容及封存manifest/hash；禁止执行test世界。

记录原world/teacher源文件SHA256、模板源文件、预注册和manifest SHA256；评估前后原文件hash相等。源文件/manifest/预注册hash不一致时拒绝评估。测试不会解封测试世界。证据在本目录，结论在 reports/P1-template-round3.md。
