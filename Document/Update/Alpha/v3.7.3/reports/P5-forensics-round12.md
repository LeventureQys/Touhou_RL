# P5 失败取证（round12；仅 forensic，非正式 P5/T21）

## 结论

原 round11 **horizontal_masked N1 的 8 条 validation 实际 trace，全部从 initial 按 actual_action 完整重放通过**：640 次决策的前后 x、sticky hit、firstcollision 与末端状态一致；另一个独立逐逻辑帧实现也与原 world 一致。原网络成绩仍是 **2/8 成功、6/8 碰撞**，不因任何反事实研究改变。

在预登记的 **6 个失败 × 每例 2 个代表状态**，独立参考 schedule 找到 **44/60 个候选的完整有限后缀 witness**，其余16个为 unknown，均有首两逻辑帧碰撞。12个代表状态都至少存在一个独立验证的安全完整替代后缀。因此，这些**特定实际前缀下、toy 有限 horizon 内**的死亡存在可避免 witness；这不是仅依据 teacher disagreement 作出的归因，更不是 N1 已能执行该路线的证据。

单独的 privileged teacherV2 rescue 从这12状态续跑到160帧，**12/12 完整 prefix+suffix 经独立物理重放可行**。它使用可信几何模型，不是网络闭环分数、不是部署shield，也没有修改原 trace。

## 输入、预算与先冻结后运行

读取了 [round11 报告](<P5-synthetic-round11.md>)、[闭环实现](<../../../../../touhou_rl/tactical_synthetic_closed_loop.py>)、[teacherV2](<../../../../../touhou_rl/tactical_teacher_v2.py>) 和 [ValidationWorld](<../../../../../touhou_rl/tactical_teacher_validation.py>)。新实现为 [forensics module](<../../../../../touhou_rl/tactical_failure_forensics.py>)，单测为 [forensics tests](<../../../../../touhou_rl/tests/test_tactical_failure_forensics.py>)。

[预注册](<../evidence/failure-forensics-round12/preregistration.json>) 在研究 run 之前落盘，SHA256：

`d180cc51ce77b214f60f22f2bd1b95ef09e0772f971b4e208dd9098baf98c2a2`

固定：8个原validation事件，只用 horizontal_masked/N1；最多6失败；每例仅取 `[原报告firstcollision−16, firstcollision)` 中第一个与最后一个可信、尚未hit的决策状态；teacher比较也仅在这最多16逻辑帧窗口内。候选域为 `[-2,-1,0,1,2]`，每候选最多223个原world有限参考program，**不做巨大状态搜索**。两个focus的stay对应同toy action0；比较是实际位移动作，不把 candidate0/9差别称为动作分歧。

冻结151个文件SHA，包含旧round11全部run产物（各臂trace/像素/控制结果/图）、旧预注册和summary、旧输入权重/records、相关源码及新源码/测试。[运行后SHA](<../evidence/failure-forensics-round12/old-input-sha256-after.json>) 与冻结值全部一致。未覆盖旧run；本轮未训练、未产正式labels、未把val混入训练、未运行游戏、未开启正式P5/T21。

## 独立验证方法

1. **原实际trace完整重放**：从initial重放全部80个actual_action（包括碰撞后suffix），检查candidate映射、hold2、执行前后x、sticky hit、每行firstcollision、全部81个采样位置和horizon。
2. **独立逐逻辑帧物理实现**：不调用原advance或teacher作为裁判，分别实现每帧发射、实际当前位置aim、移动/clamp、弹位置及碰撞、F1墙、F3窗口压力。保留原horizon，不能为取单帧把horizon截短而改变发射截止。其每两帧端点与原world完整重放相等。
3. **候选首2帧**：每个代表状态的每个候选保留两条逻辑帧记录（x/hit/action）。即刻hit仍只记 `unknown`（候选已碰撞不等于全状态不可行）。
4. **参考后缀**：固定实际完整prefix + 一个forced候选 + 原world `programs` 的固定有限schedule suffix，从initial以独立物理重算所有后续出生/瞄准，不套用旧trace未来bulletlist。搜索和验证不调用teacher；每个feasible保留完整actions、prefix、candidate、suffix及独立全程逐帧验证trace。要求完整160帧、无hit，F3还要求原偶数决策采样达到goal。
5. **privileged rescue**：另跑teacherV2，从同一实际状态到horizon生成suffix；再由独立实现从initial重放完整prefix+suffix。保留全部rescue动作、决策和独立验证trace，不计入原网络成绩。

这是有限正向witness，不是viability kernel或精确最迟安全期限。未见证一律unknown，不能以teacher短视搜索或参考program穷尽这个小类来宣称一般不可行。

## first divergence、late reaction、stays、switch

首碰撞“原/逻辑”分别为原2帧分辨率与本轮逐帧解析。first divergence只指**该16帧窗口内**的首次实际动作与teacher建议不同，不是事件全程首次分歧。stay为action0的决策数；switch为相邻决策动作值改变数，另保留正负直接反转数。碰撞前统计含触发首碰撞的最后一次动作，不含碰撞后动作。

| 失败事件 | 首碰撞 原/逻辑 | 窗口 | 首分歧 | 分歧/8 | 窗口stay / switch | 碰撞前stay / 决策数 | 碰撞前switch / 正负反转 |
|---|---:|---|---:|---:|---:|---:|---:|
| F2-00-m0 |112/112|96–110|110|1|8 / 0|18 / 56|1 / 0|
| F2-00-m1 |136/136|120–134|128|2|8 / 0|30 / 68|2 / 1|
| F2-01-m0 |76/75|60–74|60|8|0 / 0|0 / 38|0 / 0|
| F2-01-m1 |102/102|86–100|90|2|8 / 0|10 / 51|1 / 0|
| F3-00-m0 |138/138|122–136|122|8|8 / 0|31 / 69|2 / 1|
| F3-00-m1 |138/138|122–136|122|8|8 / 0|32 / 69|3 / 2|

48个可信未hit决策比较中29个disagreement。**全部6例的窗口内实际action不切换**：五例一直stay，F2-01-m0一直action+2。预登记late reaction定义为首次窗口disagreement之后、首碰撞前的第一次实际action change及其lag；这里全部不存在，故时间/lag为null，不能虚构“晚了几帧”的准确反应时。F2-00-m0甚至直到110才与teacher不同，而96处actual0也有安全后缀witness；说明早期动作本身并不充分决定死亡，后续持续动作很重要。

F1两例完整重放无碰撞，未强行生成失败窗口或代表状态；它们不是0次teacher分歧的全程比较结果，而是未做teacher失败窗口比较。

## 独立参考候选与privileged rescue

下面“feasible首动作”均有完整实际prefix + 该候选 + 固定finite suffix独立witness；省略者为unknown，**不叫infeasible**。实际first动作在每例早代表状态均仍有witness，但在最后代表状态均已首2帧hit。

| 事件 | 早状态：actual / teacher → feasible首动作 | 晚状态：actual / teacher → feasible首动作 | rescue早/晚 |
|---|---|---|---|
| F2-00-m0 |96：0 / 0 → −2,−1,0,+1,+2|110：0 / −2 → −2|feasible / feasible|
| F2-00-m1 |120：0 / 0 → −2,−1,0,+1,+2|134：0 / +1 → −2,+1,+2|feasible / feasible|
| F2-01-m0 |60：+2 / 0 → −2,−1,0,+1,+2|74：+2 / −2 → −2|feasible / feasible|
| F2-01-m1 |86：0 / 0 → −2,−1,0,+1,+2|100：0 / −2 → −2|feasible / feasible|
| F3-00-m0 |122：0 / +2 → −2,−1,0,+1,+2|136：0 / +2 → −2,−1,+1,+2|feasible / feasible|
| F3-00-m1 |122：0 / −2 → −2,−1,0,+1,+2|136：0 / −2 → −2,−1,+1,+2|feasible / feasible|

实际只需140个schedule尝试便取得44个正witness；另16候选首两帧已经hit，不搜索后缀。预算上限223×60未被扩张。每个失败在最后未hit决策处至少有一个其他首动作的独立有限安全witness，支持“从这个toy实际前缀存在避免本次死亡的完整路线”。但只有两个时刻，不扫描完整期限，不能称它们为唯一、最早或最迟必须反应点；也不把12个状态或60候选伪装成独立validation事件。

完整证据：[summary](<../evidence/failure-forensics-round12/run/summary.json>)、[证据SHA](<../evidence/failure-forensics-round12/run/sha256.json>)。

- F2-00：[m0](<../evidence/failure-forensics-round12/run/validation-F2-structure-00-m0.json>)、[m1](<../evidence/failure-forensics-round12/run/validation-F2-structure-00-m1.json>)。
- F2-01：[m0](<../evidence/failure-forensics-round12/run/validation-F2-structure-01-m0.json>)、[m1](<../evidence/failure-forensics-round12/run/validation-F2-structure-01-m1.json>)。
- F3-00：[m0](<../evidence/failure-forensics-round12/run/validation-F3-structure-00-m0.json>)、[m1](<../evidence/failure-forensics-round12/run/validation-F3-structure-00-m1.json>)。
- F1重放：[m0](<../evidence/failure-forensics-round12/run/validation-F1-structure-00-m0.json>)、[m1](<../evidence/failure-forensics-round12/run/validation-F1-structure-00-m1.json>)。

## 执行与测试

```powershell
python -m pytest touhou_rl/tests/test_tactical_failure_forensics.py -q -p no:cacheprovider --basetemp .tmp/v373-round12-forensics-tests-02
python -m touhou_rl.tactical_failure_forensics preregister
python -m touhou_rl.tactical_failure_forensics run
```

**11 passed in 1.44s**，新basetemp；preregister/run均exit0。首次测试11failed仅因新fixture遗漏Scenario必填template_group/seed，修正fixture后、预注册前重跑通过；无研究结果驱动的源码修改。覆盖六种family×mirror物理一致、原trace篡改拒绝、teacher被禁止调用时参考仍可运行、完整前缀/首2帧碰撞、预算0为unknown、unknown model/hit guard，以及固定program数量与stay/switch统计。

## 归因边界

观察到的末段固定动作与碰撞、teacher建议分歧、独立存在安全后缀，是三类不同证据。这里**不是“teacher不同→N1死因就是不听teacher”**；teacher有特权几何和有限lookahead，reference只证实一些存在性路线。toy model是否覆盖真实游戏、神经输入是否包含足够可判别信息、如何学到及时动作都未由本轮解决。保持原8例已暴露开发validation地位，不补标、不训练、不换checkpoint、不把rescue折算网络成绩；正式P5/T21及真实游戏资格仍未通过。
