# 合成像素观测别名与单路线标签冲突（续轮12）

固定round10 records及资产，不训练、不更改labels或renderer。逐行哈希原始预处理像素堆叠、实际上一动作历史，并用递推hash表达整段因果观测历史；未来行不进入当前hash。

```powershell
python -m tools.rl.v373_alias_audit --records Document/Update/Alpha/v3.7.3/evidence/synthetic-learning-round10/data/records.json --out Document/Update/Alpha/v3.7.3/evidence/synthetic-alias-round12.json
python -m pytest touhou_rl/tests/test_tactical_alias_audit.py touhou_rl/tests/test_tactical_metrics.py -q -p no:cacheprovider --basetemp .tmp/v373-alias-round12-01
```

退出0，6tests通过。[完整alias输出](<../evidence/synthetic-alias-round12.json>)。

| 输入等价关系 | rows | unique inputs | 冲突组/涉及行 | 确定性单路线一致率上限 |
|---|---:|---:|---:|---:|
| 当前3帧stack |1280|802|6 /18|99.53125%|
| stack+实际上一步历史 |1280|802|6 /18|99.53125%|
| 整段因果输入历史 |1280|1210|4 /12|99.6875%|

train/validation各自整历史上限均99.6875%。完全相同历史的4冲突组都在frame0：相同位置初始空场，F1教师立即向出口移动，F2教师保持。单帧不可区分是真实数据约束，没有添加teacher hint/家族标识掩盖。

**不能从这些冲突推导不可避免死亡**：不同teacher动作可能都存在安全后继；它们是单路线示范偏好冲突，不是互斥安全真值。上限只适用于该冻结输入/标签的精确一致率，不是最优玩法上限。

反过来，99.69%意味着精确别名数量很小，不能把93.75%训练拟合或F2/F3闭环全败都归因于隐藏信息。近似视觉特征碰撞、动态感知不足、稀有移动标签、误差积累与分布偏移仍需逐失败证据。当前审计不衡量近似相似性，不越权声称唯一原因。
