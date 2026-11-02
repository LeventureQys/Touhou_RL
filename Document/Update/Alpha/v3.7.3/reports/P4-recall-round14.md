# 关键移动标签召回失败（续轮14）

冻结R10和R13补采研究heads，对原8validation离线轨迹逐类混淆诊断；不训练、不选权重、不用test。

```powershell
python -m tools.rl.v373_action_recall --records Document/Update/Alpha/v3.7.3/evidence/synthetic-learning-round10/data/records.json --baseline third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt --checkpoint Document/Update/Alpha/v3.7.3/evidence/synthetic-learning-round10/run-s7-n1-200/research-heads.pt --out Document/Update/Alpha/v3.7.3/evidence/action-recall-round14-original.json
python -m tools.rl.v373_action_recall --records Document/Update/Alpha/v3.7.3/evidence/synthetic-learning-round10/data/records.json --baseline third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt --checkpoint Document/Update/Alpha/v3.7.3/evidence/dagger-training-round13/augmented/research-heads.pt --out Document/Update/Alpha/v3.7.3/evidence/action-recall-round14-augmented.json
```

均exit0。moving macro recall对本族有真实标签支持的非0/9候选等权平均；只监督一条teacher路线，不代表所有必要安全动作召回。

| validation | R10 accuracy | R10 moving recall | R13aug accuracy | R13aug moving recall |
|---|---:|---:|---:|---:|
| F1 |96.88%|86.36%|97.50%|86.36%|
| F2 |86.25%|**0%**|88.75%|**0%**|
| F3 |96.25%|49.00%|97.50%|50.00%|
| 全部 |91.41%|47.22%|93.13%|47.92%|

F2的标签移动完全未命中，离线整体大部分靠hold。与R12持续不反应吻合，但这仍不是唯一死亡因果证明；不同于teacher动作也可能可行，须独立路线见证。F1离线变好却闭环退化说明分布偏移仍严重。不得仅凭准确率或多训练200步结算学习成功。

结果完整矩阵：[R10](<../evidence/action-recall-round14-original.json>)、[R13aug](<../evidence/action-recall-round14-augmented.json>)。均绑定checkpoint/records SHA。

```powershell
python -m pytest touhou_rl/tests/test_tactical_action_recall.py touhou_rl/tests/test_tactical_metrics.py -q -p no:cacheprovider --basetemp .tmp/v373-action-recall-round14-01
```

exit0，6passed in1.09s。下一步先固定数据/预算三臂对照，不在本结果上立即调类权重；之后若改权重须单因素新预登记与闭环验收。
