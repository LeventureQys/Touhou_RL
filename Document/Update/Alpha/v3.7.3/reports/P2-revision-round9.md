# P2 — controlled geometry teacher v2, revision round9

2026-11-01。**完成受控合成教师修订与新组合确认；P2正式gate未开启/未通过。** 不产生正式标签，不游戏、不读像素、不训练。不改旧teacher、world、R8 reference和R8 test。

## 1. 顺序与隔离

先写[preregistration](<../evidence/teacher-revision-round9/preregistration.md>)，随后新增[v2策略与CLI](<../../../../../touhou_rl/tactical_teacher_v2.py>)及[合同tests](<../../../../../touhou_rl/tests/test_tactical_teacher_v2.py>)。先用R8全部18 development+24已暴露validation作开发回归，再冻结[新确认manifest](<../evidence/teacher-revision-round9/confirmation-manifest.json>)，最后一次执行24新确认实例。**旧validation重跑全部属于开发，不能称独立泛化。** 新确认4组/族、每组两镜像；镜像同group，不当独立样本。参数签名合同对照R8全部定义（只读取test参数、不构造或执行test世界），完整签名不重复。

R8原24 test既未执行也未修改。本轮新确认均是首次运行的新coverage/gap/wall/start或cadence/window/threshold/goal/start组合；机制仍相同且由人选邻近组合，不是随机统计样本或未见机制泛化。确认后未调整v2。runner拒绝覆盖已有phase结果和manifest。

初次合同6通过2失败已保留于[开发日志](<../evidence/teacher-revision-round9/development-log.md>)：一处坏fixture缺aim_x、一处全屏coverage无出口导致empty min。均在首次开发批量rollout/冻结前修正。首次开发rollout后没有调策略，开发碰撞失败为空；未删旧teacher失败。独立完整失败清单见[all-arm-failures](<../evidence/teacher-revision-round9/all-arm-failures.json>)。

## 2. v2约束

- F1按实际coverage边界找左右外出口，若internal_gap且实际gap足够宽，亦纳入带2px内缩的可靠gap；选最近区域，不再认180..204或88常量。实际wall_frame在每个候选的世界推进中逐逻辑帧参与碰撞，而不是另造墙时钟。
- F3保留必须到goal的进度。实际window前只接近threshold安全右侧，不越早左危险区；开窗后目标在goal和threshold左侧留裕量。读取实际window/threshold/goal，非固定34/104。安全可能临时覆盖目标动作，不通过hold冒充完成；最终verify同时检查到goal。
- F2保留实际发射后微移。全部族先完整模拟候选的**两个逻辑帧**，再至多6步/12帧、beam20的有限因果候选。每次发射根据候选分支实际自机位置重新定向，非复用固定未来弹路。排序先立即不碰撞、再短期安全深度、再区域进度/规则一致性。全部首动作碰撞时记录bounded unknown，不宣称无解。
- 策略类不调用solve/verify/programs；AST合同验证。R8 solve只存在于独立audit入口，不能用来选teacher动作。privileged合成几何前瞻是受控teacher能力，**不是网络可见未来或真实策略能力**。

## 3. 开发回归（R8暴露数据，不是独立验证）

来源：[分split详细summary](<../evidence/teacher-revision-round9/detailed-summary.json>)、[开发执行审计](<../evidence/teacher-revision-round9/development/execution-audit.json>)。

| 已暴露split | family | 实例/reference正例 | v2成功 | 旧teacher成功 |
|---|---|---:|---:|---:|
| R8 development | F1 | 6/6 | 6/6 | 6/6 |
| R8 development | F2 | 6/6 | 6/6 | 6/6 |
| R8 development | F3 | 6/6 | 6/6 | 4/6 |
| R8 validation（开发用途） | F1 | 8/8 | 8/8 | 2/8 |
| R8 validation（开发用途） | F2 | 8/8 | 8/8 | 8/8 |
| R8 validation（开发用途） | F3 | 8/8 | 8/8 | 8/8 |

修复了R8记录的F1几何硬编码失败和development F3两例碰撞。不是对独立val再次取得泛化的证据。开发v2失败清单[空列表](<../evidence/teacher-revision-round9/development/failures.json>)保留。

## 4. 冻结后新确认

来源：[确认summary](<../evidence/teacher-revision-round9/confirmation/summary.json>)、[确认执行审计](<../evidence/teacher-revision-round9/confirmation/execution-audit.json>)。

| family | group/实例 | reference正例 | v2成功 | 旧teacher成功 |
|---|---:|---:|---:|---:|
| F1 | 4/8 | 8/8 | 8/8 | 2/8 |
| F2 | 4/8 | 8/8 | 8/8 | 8/8 |
| F3 | 4/8 | 8/8 | 8/8 | 8/8 |

v2新确认24/24成功；[失败列表](<../evidence/teacher-revision-round9/confirmation/failures.json>)为空，并非没有边界的证明。旧teacher在F1 group00/02/03各两镜像失败，均保留完整证据，reference均见证可行。F3这里没有相对旧teacher成功率增益；F2也没有。新确认F2预留矩形实际bullet-frame occupancy：v2=0、旧=0、reference=134，**相对旧teacher没有污染改善**，reference目标仅寻可行不优化污染。F3 occupancy三者均0；final room总和v2=1136px、旧=1024px、reference=396px，仅是toy末态诊断，不能推真实恢复空间优势。F1旧teacher含死亡轨迹，其final room不可当安全备用空间。reference不是待击败的性能baseline。

## 5. 独立reference、完整重放与SVG

沿用R8 bounded solve最多223个open-loop程序及verify，不读teacher决策。每个正例完整160逻辑帧重放（80个2帧动作），世界内部每逻辑帧实际发射/移动/碰撞。有限搜索无见证仍unknown；本轮66实例均有正例，不说明搜索完备。

每事件JSON记录三臂完整动作/轨迹/发射/碰撞采样帧、v2与旧决策。首碰撞记录仍2帧采样粒度，不能冒充精确帧；死亡后toy轨迹继续到horizon。额外[完整性脚本](<../evidence/teacher-revision-round9/integrity_check.py>)独立重新verify三臂，66个reference正例全重放，66 SVG XML解析通过，见[integrity audit](<../evidence/teacher-revision-round9/integrity-audit.json>)。SVG是时间—canonical x图，非游戏回放，未栅格视觉验收。

| family | 三个新确认对比SVG（另有相邻同名JSON） |
|---|---|
| F1 | [新外出口](<../evidence/teacher-revision-round9/confirmation/confirmation-F1-00-m0.svg>) · [新内部gap](<../evidence/teacher-revision-round9/confirmation/confirmation-F1-02-m0.svg>) · [旧假gap](<../evidence/teacher-revision-round9/confirmation/confirmation-F1-03-m0.svg>) |
| F2 | [新cadence15](<../evidence/teacher-revision-round9/confirmation/confirmation-F2-00-m0.svg>) · [新origin](<../evidence/teacher-revision-round9/confirmation/confirmation-F2-01-m0.svg>) · [新burst](<../evidence/teacher-revision-round9/confirmation/confirmation-F2-02-m0.svg>) |
| F3 | [窗口40..96](<../evidence/teacher-revision-round9/confirmation/confirmation-F3-00-m0.svg>) · [窗口28..92](<../evidence/teacher-revision-round9/confirmation/confirmation-F3-01-m0.svg>) · [窗口44..110](<../evidence/teacher-revision-round9/confirmation/confirmation-F3-02-m0.svg>) |

## 6. Source与命令核验

[manifest](<../evidence/teacher-revision-round9/confirmation-manifest.json>)及两phase execution audit保存源码before/after一致：

| source | SHA256 |
|---|---|
| v2与runner | `47d39e1c02c32865ed5db46befdc1ab22d7178ded86eb1333409930ffd648e95` |
| 旧teacher | `c23c9b91ba8c692265f4171dd430d08c508d5884f26370581943727680d0ad73` |
| R8 world/reference | `86608f2774553caa473e6d5d0e35c44fe42e12064e5ec6ebe1d3e4913bf3d92b` |
| scenarios | `471f73debcbb58ee3f184948b2368c6e453a59898c7286e68b29b9a863d84906` |

```powershell
python -m touhou_rl.tactical_teacher_v2 development --output Document/Update/Alpha/v3.7.3/evidence/teacher-revision-round9
python -m touhou_rl.tactical_teacher_v2 freeze --output Document/Update/Alpha/v3.7.3/evidence/teacher-revision-round9
python -m touhou_rl.tactical_teacher_v2 confirmation --output Document/Update/Alpha/v3.7.3/evidence/teacher-revision-round9
python -m pytest touhou_rl/tests/test_tactical_teacher_v2.py touhou_rl/tests/test_tactical_teacher_validation.py touhou_rl/tests/test_tactical_scenarios.py touhou_rl/tests/test_tactical_teacher.py -q -p no:cacheprovider --basetemp .tmp_teacher_revision_round9_final_a
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python Document/Update/Alpha/v3.7.3/evidence/teacher-revision-round9/integrity_check.py
```

最终**28 passed in0.21s**，全新basetemp，禁cache。三次策略CLI均exit0。完整性脚本首次误以带点版本路径为Python模块调用，报ModuleNotFoundError，无世界执行；改为独立脚本+显式PYTHONPATH后exit0，不涉及策略源码改动。

## 结算与边界

仅支持这个明确几何、固定y、已知因果发射模型中受控v2修订。没有精确latest上界、真实B0对照、游戏校准、F4/F5/F6或人工独立审查。beam有限可能漏可行策略，目标距离启发式也可能走入长时死局；未见失败不等于可靠全域。确认不扩展成正式成功标签，**不授权训练，不开启正式gate**。
