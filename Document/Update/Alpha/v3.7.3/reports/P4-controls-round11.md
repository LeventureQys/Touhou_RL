# 合成像素离线频率/历史/视觉控制（续轮11）

冻结round10末步研究checkpoint，不重训、不选epoch、不访问封存test。本轮只是开发验证轨迹上的离线控制；闭环结果单独报告。

```powershell
python -m tools.rl.v373_synthetic_controls --records Document/Update/Alpha/v3.7.3/evidence/synthetic-learning-round10/data/records.json --baseline third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt --checkpoint Document/Update/Alpha/v3.7.3/evidence/synthetic-learning-round10/run-s7-n1-200/research-heads.pt --out Document/Update/Alpha/v3.7.3/evidence/synthetic-controls-round11.json
```

退出0，真实CUDA前向，无优化步。输出[原始controls](<../evidence/synthetic-controls-round11.json>)绑定研究checkpoint/records SHA。

训练标签频率：hold candidate0=530/640（82.81%），左右普通各44、左右聚焦各11。只用训练集确定频率mode，不读取validation频率选基线。mode恰好hold。

| validation | N1正常历史 | 去实际动作历史 | 首帧视觉固定且去历史 | hold/mode |
|---|---:|---:|---:|---:|
| F1（2事件） | 96.88% | 87.50% | 86.25% | 86.25% |
| F2（4事件） | 86.25% | 73.75% | 72.50% | **90.00%** |
| F3（2事件） | 96.25% | 96.25% | 31.25% | 66.25% |
| 全部8事件 | 91.41% | 82.81% | 65.63% | 83.13% |

F2离线一致率低于hold，说明整体91.41%不能结算诱导成功。F3对初始固定视觉敏感，但history去除几乎无差，不足以证明记忆必要性。去历史只是推理期输入消融，不是同预算训练N0/B1；固定首帧视觉不是有效物理轨迹，不作生存反事实。

B0在这些渲染教师轨迹单路线标签上一致率0，可能常选toy不支持的上下/斜向候选；不能当真实B0能力退化。后续水平mask闭环必须两臂一致并单列原18候选unsupported/invalid率，不偷偷映射方向。

本轮不更改权重/renderer/阈值。下一步闭环按实际候选推进世界、因果渲染和commit历史，才可说明单事件存活/目标是否有效；仍不代表正式P5或真机改善。
