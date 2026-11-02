# P4 三臂等预算 fresh 初始化诊断（round14）

## 结论与隔离

**B1 / N0 / N1 的 8 个已暴露 validation 横向闭环成功分别为 0/8、1/8、1/8；没有 N1 相对 N0 的总体成功收益。** N1 离线标签一致率92.66%，但五类 macro balanced recall57.51%，两种 focus 移动 recall 都为0；高 accuracy 不等于学会少数纠正动作，更不等于生存。

本轮仅新增[独立 runner](<../../../../../tools/rl/v373_arm_diagnostic.py>)及[对应测试](<../../../../../touhou_rl/tests/test_v373_arm_diagnostic.py>)，不修改旧训练、formal loader、renderer 或任何旧证据。只原 R10 的16完整event（train8/validation8）；不读取 dagger 数据、不触及 test、不采样失败补标、不调用 teacher/shader、不运行游戏，不是正式训练/P5/T21，不产可部署候选。8val已经反复暴露，闭环是**开发闭环，不是独立验证**。所有失败都保留，无增加epoch或结果驱动修改。

## 预登记、训练与精度

先冻结[preregistration](<../evidence/arms-round14/preregistration.json>)（源码、测试、原records、48个pixel/history/label资产及baseline SHA），然后执行测试，最后一次运行三臂；SHA `1e43e06382ad3bfbd6e0cce7ca837bb0ee5bb4b4029157d455b0848826eb9955`。执行前后核验全部冻结身份不变。[执行记录](<../evidence/arms-round14/execution-notes.json>)、[结果汇总](<../evidence/arms-round14/run/summary.json>)和[结果SHA](<../evidence/arms-round14/run/sha256.json>)均独立落盘，JSON以`x`、权重以`xb`、输出目录以`exist_ok=False`拒绝覆盖。

- 同一个原 frozen v3-st backbone，各臂构造前**重新设seed7**，fresh heads；无checkpoint/warmstart，N1也不继承R10。遵循现有 TacticalPolicy 初始化：其他heads为fresh random，residual weight/bias标准zero-init，不擅改旧模块。这里重置点在backbone加载后，与R10在加载前设seed的位置不同，不宣称复现R10具体权重。
- 各200次Adam更新，lr1e-3、clip1、full80、batch1，原train顺序round-robin，每事件25次；无按val选模型/步数/超参。仅action单路线正例有监督，aux option/target/termination等未监督，不称学会战术分层。
- 原16×80帧只做**一次真实GPU batch16** frozen编码，共享内存特征，未生产磁盘cache。RTX4070 SUPER、PyTorch2.14.1+cu132，编码2.757s。
- 编码、初末评估、训练及闭环均FP32/noAMP，matmul TF32=false、cuDNN TF32=false、float32 precision=highest。B0只引用R11已有2/8不重复。**R11闭环cuDNN TF32=true而本轮false，且fresh初始化/RNG重置点不同，因此本轮与R11差异不是纯结构因果比较。** 本轮三臂精度协议一致。

### 参数预算

active为`requires_grad=True`且纳入Adam的参数；不是“有非零梯度”或“已监督”的参数。heads_total包含未激活诊断分支，total另外包含同一1,507,162参数冻结backbone。

|arm|active / optimizer|heads_total|total 含backbone|优化秒|首/末loss|
|---|---:|---:|---:|---:|---|
|B1|77,968|494,992|2,002,154|5.454|3.015779 / 0.878964|
|N0|428,828|494,668|2,001,830|10.303|3.015779 / 0.317603|
|N1|429,200|494,992|2,002,154|9.984|3.015779 / 0.286058|

三臂backbone逐state tensor严格不变。B1关闭GRU训练，N0/N1关闭memoryless训练，N0另关闭option_embedding。预算非等参数；等的是数据、更新、seed、精度及优化协议，不能将B1差异只归因于记忆。

## 初始与末步离线指标

以下为unmasked18的严格candidate单路线一致率 / macro balanced recall（%），不是安全集合recall；train/val各640有效行。每个arm初始与最终、每split每family、unmasked及同R11 horizontal_masked两条件的完整5×6 confusion、support和per-class recall都保存在各自 [B1 training](<../evidence/arms-round14/run/B1/training.json>)、[N0 training](<../evidence/arms-round14/run/N0/training.json>)、[N1 training](<../evidence/arms-round14/run/N1/training.json>)，另含逐事件truth/prediction与200步loss，未只保留好结果。

|arm|初始train acc/macro|初始val acc/macro|末步train acc/macro|末步val acc/macro|
|---|---|---|---|---|
|B1|0 / 0|0 / 0|82.81 / 20.00|83.13 / 20.00|
|N0|0 / 0|0 / 0|91.88 / 54.28|90.16 / 55.36|
|N1|0 / 0|0 / 0|94.53 / 56.59|92.66 / 57.51|

### 每族末步validation acc / macro（%）

|族|B1|N0|N1|
|---|---|---|---|
|F1（2event）|86.25 / 33.33|93.13 / 80.61|97.50 / 90.67|
|F2（4event）|90.00 / 33.33|86.25 / 31.94|88.13 / 32.64|
|F3（2event）|66.25 / 20.00|95.00 / 58.63|96.88 / 59.20|

F1/F2分别只出现3类，所以macro只平均该族有support的类；缺类recall为null，不能与五类macro直接作难度排序。F2三臂两种移动focus recall均0，moving macro为0；F3四种moving macro为B1=0、N0=49%、N1=49%。B1完全塌缩hold，val532/640本就是hold，83.13%只是类别不均衡。

### validation末步完整action confusion

行是真实`[hold,left,right,leftfocus,rightfocus]`，列为同序再加`other`；对应candidate `[0,7,3,16,12]`。**严格candidate统计**，candidate9即使位移同0仍计other，不混同movement semantic。所有arm相同support `[532,36,36,18,18]`。macro为有support五类recall无权平均，而非总体accuracy。

```text
B1                   N0                    N1
532 0 0 0 0 0        512 8 12 0 0 0         525 1 6 0 0 0
 36 0 0 0 0 0          0 34 2 0 0 0           0 34 2 0 0 0
 36 0 0 0 0 0          0 5 31 0 0 0           1 1 34 0 0 0
 18 0 0 0 0 0         15 1 2 0 0 0           16 0 2 0 0 0
 18 0 0 0 0 0         15 0 3 0 0 0           16 0 2 0 0 0
```

|arm|hold recall|left|right|leftfocus|rightfocus|macro|
|---|---:|---:|---:|---:|---:|---:|
|B1|100%|0%|0%|0%|0%|20.00%|
|N0|96.24%|94.44%|86.11%|0%|0%|55.36%|
|N1|98.68%|94.44%|94.44%|0%|0%|57.51%|

## 同R11横向mask actual-history闭环

复用原`run_event`、`pixel_stack`、renderer、实际`advance`：mask `[0,3,7,9,12,16]`，深度3过去到现在，事件hidden reset/BOS，执行后提交真实candidate/hold2/delta2；不提交预测aux目标。每臂使用**arm-aware自定义network callback**，手工检查新研究schema `v373-arm-diagnostic-heads/1`、baseline SHA、arm config并strict load该臂heads；B1/N0不塞进N1的FrozenNetworks。正式loader不变。

每臂8×80真实决策、完整160帧，无invalid/fallback。碰撞sticky且继续完整horizon，firsthit为两逻辑帧分辨率，goal即使碰撞后达到也不能挽救失败。F1/F2 goal列仅为无需附加目标的恒真，不是额外任务成功。完整[ B1闭环](<../evidence/arms-round14/run/B1/closed-loop.json>)、[N0闭环](<../evidence/arms-round14/run/N0/closed-loop.json>)、[N1闭环](<../evidence/arms-round14/run/N1/closed-loop.json>)及各event trace/pixels均保留。

|族|事件数|B1 survival / goal / feasible|N0 survival / goal / feasible|N1 survival / goal / feasible|
|---|---:|---|---|---|
|F1|2|0 / 2 / 0|0 / 2 / 0|0 / 2 / 0|
|F2|4|0 / 4 / 0|1 / 4 / 1|1 / 4 / 1|
|F3|2|0 / 0 / 0|0 / 1 / 0|0 / 1 / 0|
|总计|8|0 / 6 / **0**|1 / 7 / **1**|1 / 7 / **1**|

### 每族每事件 firsthit（逻辑帧；—为完整无碰撞）

|validation事件|B1|N0|N1|
|---|---:|---:|---:|
|F1-00-m0|40|40|40|
|F1-00-m1|40|40|40|
|F2-00-m0|46|102|76|
|F2-00-m1|46|104|74|
|F2-01-m0|28|—|—|
|F2-01-m1|28|94|52|
|F3-00-m0|94|138|136|
|F3-00-m1|94|94|94|

N0与N1成功的是同一个F2事件；三条其余F2 N1碰撞均早于N0，但小样本不能推出稳定优劣。F3一例达到goal仍碰撞，另一例未达到goal且碰撞。N1较高离线accuracy没有转成更高闭环成功；此轮不足以支持条件分层结构优势。

## 执行、测试和保留边界

```powershell
python -m tools.rl.v373_arm_diagnostic preregister
python -m pytest touhou_rl/tests/test_v373_arm_diagnostic.py -q -p no:cacheprovider --basetemp .tmp/v373-arms-round14-tests-01
python -m tools.rl.v373_arm_diagnostic run
```

依次exit0；**6 passed in1.14s**，覆盖confusion/other/缺类、exclusive输出、原16事件与无test/组隔离、各arm trainable branch与骨干冻结、研究schema拒绝。GPU job自行收取exit0，无失败训练或失败测试尝试，未重跑任何arm。每arm完整初末指标与loss、手动研究heads、8条实际轨迹及像素均在[evidence目录](<../evidence/arms-round14/preregistration.json>)所界定的新run保存；权重不是formal tactical schema。

不把8事件、4组验证模板与镜像当成独立大样本，不宣称统计显著/真实泛化。结果支持“B1偏频率塌缩、N0/N1能学习普通方向但focus少数类失败且闭环弱”；**不支持继续堆epoch、不支持正式部署或战术理解，也不改写R11/R13原有失败。**
