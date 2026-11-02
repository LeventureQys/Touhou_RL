# v3.7.3 首轮网络工程与优化诊断

日期：2026-11-01。**状态：工程链路通过；正式T21/P4未通过，不可接管，不是玩法改善checkpoint。**

## 新接口

- [策略](<../../../../../touhou_rl/tactical_policy.py>)：N1预测战术/目标条件化；N0不使用这些动作条件；B1无记忆诊断；固定18动作、残差零初始化，骨干冻结。
- [事务runner](<../../../../../touhou_rl/tactical_runner.py>)：propose不提交；commit只确认实际动作/实际hold；cancel、死亡及异常reset；固定duration=1。未接入真机循环。
- [训练工程入口](<../../../../../touhou_rl/tactical_train.py>)：仅工程smoke与前置数据门检查。**完整真实事件训练器仍待P3数据契约后实施，不将当前CLI当完整训练交付。**

## 实际执行

工作目录：`D:\WorkShop\Touhou_RL`。

```powershell
python -m tools.rl.tactical --help
python -m touhou_rl.tactical_train --help
python -m pytest touhou_rl/tests/test_tactical_train.py -q -p no:cacheprovider --basetemp .tmp/v373-train-gate-20261101
```

退出码0，门禁测试3通过。CLI只暴露实现项，没有伪造计划中的collect/train/live-pair等命令。

```powershell
python -m pytest touhou_rl/tests/test_tactical_policy.py touhou_rl/tests/test_tactical_runner.py touhou_rl/tests/test_tactical_train.py -q -p no:cacheprovider --basetemp .tmp/v373-tactical-integrated-20261101
```

退出码0，25 passed in 1.54s。包括CPU/CUDA sidecar加载、SHA绑定、零初始化、预测条件反传、因果性、padding/reset、未知标签mask和事务历史。

```powershell
python -m tools.rl.tactical engineering-smoke --baseline third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt --out Document/Update/Alpha/v3.7.3/evidence/engineering-optimization-s7-final --steps 100 --device cuda --seed 7
```

退出码0。N1、seed7、FP32、batch4×32步，100次真实Adam更新约2.58s（38.81步/s），loss3.47056→0.00676，人造标签拟合100%，骨干逐参数精确不变。有效可训练参数429,200，峰值allocated约49.5MiB。不使用GPU利用率作为训练有效性的证据。

完整[实测报告](<../evidence/engineering-optimization-s7-final/report.json>)。输出权重使用独立engineering-smoke-only schema，**不能由正式tactical checkpoint加载器当候选策略部署**。早先未冻结inactive分支的诊断也保留在独立目录，不覆盖。

## 不能从这里推出什么

- 人造特征不是真实像素事件，标签不是可信玩法教师；这不是T21的8事件过拟合门。
- 100%仅为工程标签，不能等同离线玩法准确率、更不能等同闭环生存。
- 未跑正式四臂三seed、留出集、P5闭环或P6/P7。
- B1目前为本轮无记忆战术诊断接口；正式四臂比较前须冻结其与旧宏观头的准确关系及参数预算。

## 全量回归

```powershell
python -m pytest touhou_rl/tests -q -p no:cacheprovider --basetemp .tmp/v373-regression-20261101
```

退出码0：1330 passed，1 xfailed，4 subtests passed，92.14s。保留既有xfail，未删除失败测试求绿；此轮后的场景单因素修订另跑针对测试。

下一步以P1/P2失败修订和P0同步标定优先；数据门未满足前不通过增加epoch制造训练进展。
