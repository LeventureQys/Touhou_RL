# P2 — independent synthetic teacher validation, round8

2026-11-01。**本轮完成独立合成留出审计，不是P2完整通过，不产生正式标签。**
未启动游戏、未读像素/B0 checkpoint、未训练、未改旧teacher/scenario/reference。

## 1. 先登记，再执行

已读[升级计划](<../升级执行计划.md>)、[R2登记](<../evidence/tactical-r2-prereg.md>)、[R2结果](<P1-P2-tactical-development.md#L136-L154>)及现有teacher/reference/scenarios。
新增独立[validation模块](<../../../../../touhou_rl/tactical_teacher_validation.py>)和[测试](<../../../../../touhou_rl/tests/test_tactical_teacher_validation.py>)。
先落盘[preregistration](<../evidence/teacher-validation-round8/preregistration.md>)，实现中跑9项合同单测（含预声明validation几何负例；非批量留出评估），再以单独preregister命令写入[manifest](<../evidence/teacher-validation-round8/manifest.json>)，之后才批量evaluate。严格说source哈希manifest在合同单测之后封存，而非所有单测rollout之前；不能声称机器可核验manifest先于一切rollout。协议文本和全部参数定义在批量validation之前登记，未按单测/validation结果调teacher或baseline。
冻结teacher SHA256 `c23c9b91ba8c692265f4171dd430d08c508d5884f26370581943727680d0ad73`；冻结评估源码SHA256 `86608f2774553caa473e6d5d0e35c44fe42e12064e5ec6ebe1d3e4913bf3d92b`；before/after相同。

| split | 每族group | 每族实例 | 总实例 | 执行 |
|---|---:|---:|---:|---|
| development | 3 | 6 | 18 | 是 |
| validation | 4 | 8 | 24 | 是 |
| test | 4 | 8 | 24 | **否，仅manifest封存** |

镜像同group同split；完整group参数签名跨split不重复。这里是人为预声明的小型结构/参数组留出，不是随机抽样、独立真实关卡或未见机制泛化；镜像对不是独立样本。seed仅标识，不宣称随机功效。

- F1：开发保留旧墙/内部缝隙形式，validation变覆盖区、关闭时间、内部缝位置/宽度。
- F2：保留**每次实际发射时对实际自机x定向**；validation变起点、cadence、speed、origin，并含两个burst组。镜像连同发射源一起镜像。测试实证动作改变后继发射方向。
- F3：保留R2 `depletion_pressure=True`、早左危险/晚右耗尽和必须到达左目标。development是R2原34/104/180机制，另取起点/cadence组合；validation由新模块显式参数化窗口、阈值、目标及一组origin。**这是新合成世界的压力参数/几何留出，不是旧Scenario本身具备新结构或游戏机制泛化。**

不依据validation调整teacher、参数、baseline或reference预算。

## 2. 冻结评价与有限参考

旧TacticalTeacher直接调用，按新世界因果推进；没有修改其88、180..204、34..104、128等硬编码。
固定R1 broad comparator：canonical动作-2/+2每16次决策（32逻辑帧）交替；另报hold、premature-cross、frame112 late-cross。**不是B0，也不与R0替换baseline的结果拼作收益。**

独立reference不读取teacher动作/选项：最多223个预登记open-loop programs（hold；11启动时刻×4方向×5长度；2个cadence微移）。对原始世界逐动作、每动作完整2帧重放，内部每逻辑帧做出生、移动和碰撞检查。正例只认证160帧有限horizon；F3同时要求实际达到目标。未命中只返回unknown，不构造不可行、精确截止或安全标签。

所有事件保存完整动作/实际hold=2、决策/目标/trigger、轨迹、发射、首碰撞采样帧、baseline和reference；首碰撞输出粒度2帧（不冒充精确碰撞时刻）。命中后仍继续跑至horizon，故末端空间/发射污染也包括死亡后toy轨迹，**不能把死亡臂的末端room当可用安全空间**。

## 3. 实测结果

来源：[summary](<../evidence/teacher-validation-round8/summary.json>)、[execution audit](<../evidence/teacher-validation-round8/execution-audit.json>)。
每项reference均有有限horizon正例；teacher成功分母因此等于实例总数。

| split/family | reference正例 | teacher验证成功 | broad成功 | hold成功 | premature / late成功 |
|---|---:|---:|---:|---:|---:|
| development F1 | 6/6 | 6/6 | 0/6 | 2/6 | 6/6 / 2/6 |
| development F2 | 6/6 | 6/6 | 6/6 | 4/6 | 0/6 / 4/6 |
| development F3 | 6/6 | **4/6** | 0/6 | 0/6 | 0/6 / 0/6 |
| validation F1 | 8/8 | **2/8** | 0/8 | 2/8 | 8/8 / 2/8 |
| validation F2 | 8/8 | 8/8 | 8/8 | 0/8 | 0/8 / 0/8 |
| validation F3 | 8/8 | 8/8 | 0/8 | 0/8 | 0/8 / 0/8 |

- F1 validation低于候选90%门；teacher几何硬编码失败必须保留，不能纳入成功监督。
- F2 validation预留矩形实际bullet-frame occupancy：teacher **0**、broad **412**；生存均8/8。birth-ray计数反而76 vs66，说明出生射线计数不等于占用。development occupancy0 vs214，生存均6/6。仅为这个toy emitter和冻结broad的正诊断，不是全局污染/真实B0优势。
- F3 validation均生存并到达目标，但teacher final room总和992px vs broad1344px，不能声称横向空间数值收益；broad均死亡，room更不能当安全备用空间。两者预留矩形occupancy均0。development F3 teacher两例弹撞，保留4/6负结果，不因validation较好删除。

### 失败清单

[全部8个失败](<../evidence/teacher-validation-round8/failures.json>)，均为已重放碰撞而不是无解证明；每个都有独立reference正例：

- development F3 group02两镜像：首碰撞记录38帧。speed9/cadence20/start344组合下既有规则在迁移初期撞旧瞄准弹。
- validation F1 group00两镜像：40帧碰撞。teacher在88停下，而新覆盖左沿72，硬编码出口并不在安全侧。
- validation F1 group02两镜像：48帧碰撞。实际可靠缝208..224，起点216；旧teacher不承认此缝并强行左移撞墙（hold可生存）。
- validation F1 group03两镜像：44帧碰撞。实际缝220..236，起点192；旧teacher错误认定180..204内部可靠而保持不动。

## 4. delay扫描：仅已见证窗口，不是假精确门通过

F1/F3共28事件，每事件两个prefix（hold=0；continue-current=初始teacher动作），逐2帧扫描0..160，**56×81=4536**条delay记录，无二分/单调性推断。2104条正例各存完整80个两帧动作；2432条未见证为unknown，保存在[unknown集合](<../evidence/teacher-validation-round8/unknown.json>)，包括prefix已碰撞和有限搜索无正例。所有2104正例另行从零复验prefix及全horizon；见[witness integrity audit](<../evidence/teacher-validation-round8/witness-integrity-audit.json>)。

| validation group（镜像相同） | hold见证窗口 | continue-current见证窗口 | teacher持续进展启动 |
|---|---|---|---|
| F1 00 | [0,18] | [0,160] | 0 |
| F1 01 | [0,34] | [0,160] | 0 |
| F1 02 | [0,160] | [0,160] | 不适用：起点在真实可靠缝 |
| F1 03 | [0,36] | [0,36] | unknown：teacher不动 |
| F3 00 | [0,48] | [0,48] | 34 |
| F3 01 | [0,40] | [0,40] | 34 |
| F3 02 | [0,40] | [0,40] | 34 |
| F3 03 | [0,36] | [0,36] | 34 |

窗口只表示预算内连续**已见证**偶数delay。continue-current在F1 00/01取-2，prefix已撤出，延迟160仅表示整段初始动作持续可以生存，**不意味着可以静等160帧才撤离**；F3初始动作0，两个prefix恰好重合，依然各保存。

启动定义是目标方向连续6逻辑帧进展且总位移>=8px；相对已见证最大delay，有启动者未观察到late，F1两组无可评价启动。该诊断绝不覆盖碰撞失败：F1 00启动0仍撞墙。**所有事件exact_latest_upper_bound=null；“晚于精确最迟上界<=10%”门没有建立、没有通过。** 未见证点不能外推不可行，160只为扫描horizon，非真实终局上界。

## 5. SVG与证据完整性

42事件各有JSON+SVG（F1/F2/F3各14，其中validation各8，超过每族3段）。SVG显示teacher、冻结broad、reference、战术/目标、横向轨迹及已见证窗口；只是toy时间-x图，不伪称原策略像素回放。XML解析42/42通过；本轮未做栅格化外观验收。

| family | 三个validation示例（均有相邻同名JSON） |
|---|---|
| F1 | [出口硬编码失败](<../evidence/teacher-validation-round8/validation-F1-structure-00-m0.svg>) · [实际缝被错过](<../evidence/teacher-validation-round8/validation-F1-structure-02-m0.svg>) · [假内部缝失败](<../evidence/teacher-validation-round8/validation-F1-structure-03-m0.svg>) |
| F2 | [origin左偏](<../evidence/teacher-validation-round8/validation-F2-structure-00-m0.svg>) · [origin右偏](<../evidence/teacher-validation-round8/validation-F2-structure-01-m0.svg>) · [burst](<../evidence/teacher-validation-round8/validation-F2-structure-02-m0.svg>) |
| F3 | [窗口42..94](<../evidence/teacher-validation-round8/validation-F3-structure-00-m0.svg>) · [窗口26..90](<../evidence/teacher-validation-round8/validation-F3-structure-01-m0.svg>) · [窗口46..112](<../evidence/teacher-validation-round8/validation-F3-structure-02-m0.svg>) |

## 6. 命令与核查

```powershell
python -m touhou_rl.tactical_teacher_validation preregister --output Document/Update/Alpha/v3.7.3/evidence/teacher-validation-round8
python -m touhou_rl.tactical_teacher_validation evaluate --output Document/Update/Alpha/v3.7.3/evidence/teacher-validation-round8
python -m pytest touhou_rl/tests/test_tactical_teacher_validation.py touhou_rl/tests/test_tactical_scenarios.py touhou_rl/tests/test_tactical_teacher.py --basetemp .tmp_teacher_validation_round8_20261101_final -p no:cacheprovider -q
```

登记66条，评估exit0；最后回归 **20 passed in0.16s**。新测试9项覆盖split/镜像参数签名隔离、test拒绝执行、F2实际瞄准/镜像、完整两帧路线、F3压力/目标、delay-prefix、非单调窗口保留unknown、旧teacher硬编码负例、manifest拒绝覆盖。初次collection发现新模块声明列表括号语法错误，**在manifest冻结/首次evaluate之前**修复；随后9测试及最终20测试通过。未变更冻结后的评估源码。

## 结算

完成本轮独立合成validation/有限horizon见证/双prefix全偶数扫描及负证据。**P2完整NOT PASSED**：F1泛化失败、development F3失败、没有精确late上界、没有B0真实像素对照/校准、F4/F5/F6/独立人工复核仍缺。teacher和baseline保持冻结；test24条未执行；不创建正式成功标签、不训练。本轮结果不能授权后续训练或真机性能结算。
