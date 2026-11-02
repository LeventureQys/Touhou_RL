# task-teacher-round2 预注册（24开发实例首次运行前）

日期：2026-11-01。范围：冻结 tactical_tasks 的24个 development 实例；只新增受测规则教师模块及根tests测试，不修改旧task/teacher/reference/执行/训练模块，不启动游戏，不生成正式监督。已阅读P1-tasks-round1和升级执行计划。本文在执行24实例开发评估CLI前落盘；36项单测（含单一开发实例的故意失败保存测试）已通过，不是预注册的24实例评估。

## 冻结规则

- 每2帧从当前特权task/state重新决策，无离线动作表，不调用solve/verify_route/diagnostic_actions选择动作；独立评估阶段在教师rollout完成后才运行verify_route及solve。
- F4：以目标区域为端点，按最高2单位/帧的直线走廊检测完整剩余时域危险。当前走廊可通则立即迁移；否则仅在原地等待逐帧安全、且某个未来2帧决策时刻走廊可通时等待。此为单一规则走廊的充分检查，不是分支搜索，不声称最迟启动或不可行。等待/立即由危险时间及走廊决定，不以subtype选择序列。
- F5：先对全部{-2,-1,0,1,2}候选执行逐帧的2帧碰撞检查及最大位移≤10筛选，再按朝目标速度1的局部偏好及最小动作选取。局部碰撞否决优先于目标，不作宏观大迁移；无即时安全候选记录unknown/interrupted，不把有限规则失败当无解。
- 危险由冻结task环境定义；可读当前task及其已知危险时间模型，不能把该特权教师外推为像素部署。未知模型不声称安全。
- 成功仅按原完整40帧、无碰撞、最终进入目标，以及F5位移界定义。到达目标前或仅生存都不成功；到达目标后仍需完整时域安全保持。

## 运行、产物和失败规则

运行 `python -m touhou_rl.tactical_task_teacher --output Document/Update/Alpha/v3.7.3/evidence/task-teacher-round2`。

每例保存task/hash、完整decision（action、option、trigger、物理目标、每候选2帧碰撞/位移界、走廊departure）、actual_hold=2的执行状态、目标到达/位移、终止/失败、独立verify_route、solve见证及指标、SVG时空图。24例全部保留；若规则失败不删除、不筛除、不生成正式成功监督、不用参考路线回填教师动作。CLI拒绝覆盖已有结果。

分别报告4个subtype各6例的目标成功、生存、独立参考可行与失败列表；不混成教师全P2通过。预期检查：F4等待与立即撤离反例、F5局部界和2帧优先。只报告实测，不因结果修改契约/阈值。任何后续规则修订须新预注册目录，保留本轮。单测用新basetemp禁用cacheprovider，另跑旧task及原scenario/teacher回归。

## 独立holdout：只设计封存，不执行

见holdout-design.json：未来validation/test必须分别使用真正独立的时空结构模板组，不把本24例镜像/offset跨split，开发模板全部归development。当前Task仅允许development、固定4个subtype和危险几何，无法表示这些独立模板；状态pending_task_contract_development_only。本轮不修改task契约、不实例化/求解/评价holdout，不给出holdout性能。

## 结论边界

24例是4种固定合成危险几何的镜像/偏移开发复验，不是24个独立模板，不能估泛化。没有B0公平比较、独立留出、真实像素/动力学校准、F1/F2/F3/F6或正式P1/P2门通过声明。正式训练标签始终false。
