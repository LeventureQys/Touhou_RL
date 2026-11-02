# 首轮失败状态补采与等预算训练负结果（续轮13）

## 预登记与隔离

[实验预登记](<../evidence/dagger-training-round13-prereg.md>)先于两臂优化。仅synthetic renderer研究，不开启真实数据门、不算正式P5/T21，不替换默认模型。原8validation与其R12forensic恢复点不进入训练。

数据子任务仅原8train模板跑冻结R10 N1：网络4/8成功，4失败在碰撞前16帧切teacher，4/4完整prefix+suffix独立重放通过。仅suffix108行有效监督，网络prefix全invalid；无碰撞F1四事件不自监督，完整保留invalid。详见[采集报告](<P5-dagger-data-round13.md>)。

合并时原16event资产与新增8event全部复制到新根，保留原SHA/group，4invalid补采跳过，实际训练12event、748有效行。原验证8事件复制字节不变。新增merge测试拒绝把validation group当augmentation train。

## 实际命令

工作目录仓库根。

```powershell
python -m tools.rl.v373_synthetic_learning --records Document/Update/Alpha/v3.7.3/evidence/synthetic-learning-round10/data/records.json --baseline third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt --out Document/Update/Alpha/v3.7.3/evidence/dagger-training-round13/control-original --steps 200 --device cuda --init-checkpoint Document/Update/Alpha/v3.7.3/evidence/synthetic-learning-round10/run-s7-n1-200/research-heads.pt
python -m tools.rl.v373_merge_synthetic --original Document/Update/Alpha/v3.7.3/evidence/synthetic-learning-round10/data/records.json --augmentation Document/Update/Alpha/v3.7.3/evidence/synthetic-dagger-round13/records/records.json --out Document/Update/Alpha/v3.7.3/evidence/dagger-training-round13/mixed-data
python -m tools.rl.v373_synthetic_learning --records Document/Update/Alpha/v3.7.3/evidence/dagger-training-round13/mixed-data/records.json --baseline third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt --out Document/Update/Alpha/v3.7.3/evidence/dagger-training-round13/augmented --steps 200 --device cuda --init-checkpoint Document/Update/Alpha/v3.7.3/evidence/synthetic-learning-round10/run-s7-n1-200/research-heads.pt
python -m tools.rl.v373_compare_research --checkpoint Document/Update/Alpha/v3.7.3/evidence/dagger-training-round13/control-original/research-heads.pt --out Document/Update/Alpha/v3.7.3/evidence/dagger-training-round13/control-closed-loop
python -m tools.rl.v373_compare_research --checkpoint Document/Update/Alpha/v3.7.3/evidence/dagger-training-round13/augmented/research-heads.pt --out Document/Update/Alpha/v3.7.3/evidence/dagger-training-round13/augmented-closed-loop
```

全部退出0。相同warm start SHA85cc2f4e…d81d7、fresh Adam、seed7、200更新、FP32、clip1、batch1 full event，round-robin。不同数据导致每个原event曝光次数不同，这是补采干预的一部分，不声称逐原事件曝光匹配。没有验证选步数/选权重。

## 结果

| 指标 | R10初始研究权重 | 原数据继续200步 | 加补采200步 |
|---|---:|---:|---:|
| 离线val单路线一致率 |91.41%|93.44%|93.13%|
| 闭环F1 |2/2|1/2|0/2|
| 闭环F2 |0/4|1/4|1/4|
| 闭环F3 |0/2|0/2|0/2|
| 闭环全成功 |2/8|2/8|**1/8**|
| 优化壁钟 |10.61s|11.19s|10.46s|
| 骨干严格不变 |是|是|是|

训练报告：[control](<../evidence/dagger-training-round13/control-original/report.json>)、[augmented](<../evidence/dagger-training-round13/augmented/report.json>)；闭环原始：[control](<../evidence/dagger-training-round13/control-closed-loop/summary.json>)、[augmented](<../evidence/dagger-training-round13/augmented-closed-loop/summary.json>)。

**无补采收益，反而F1退化。** F3两臂firsthit均136；补采没有修复R11已有F3失败。F2改善各1但事件不是同一个，不能只挑成功例宣传稳定改善。样本极小且开发验证反复暴露，不作显著性或确认结论。

## 边界与下一步

1. 不再为这组负结果加epoch/无限补采。先检查108稀有纠正标签在事件平均loss中的权重、原局部保持约束缺失、完整80步回放与部署history差异；一次只改一因素。
2. 当下只action训练，未监督战术/目标，不能把N1名义结构当已学战术分层；可能应先同预算N0/B1对照。
3. 原val不混train，原heads/run不覆盖。两新heads保留研究schema不可部署；真实门仍pending。
4. closed-loop沿R11默认精度，与FP32抽特征协议的微差仍需单独消融；本轮两臂使用相同闭环实现，不据此抹去负结果。

```powershell
python -m pytest touhou_rl/tests/test_tactical_synthetic_dagger.py touhou_rl/tests/test_tactical_synthetic_merge.py touhou_rl/tests/test_tactical_synthetic_learning.py -q -p no:cacheprovider --basetemp .tmp/v373-round13-dagger-integration-01
```

退出0，9passed in2.30s。
