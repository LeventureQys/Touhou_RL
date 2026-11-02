# P4按事件离线诊断（续轮9）

新增[tactical_metrics](<../../../../../touhou_rl/tactical_metrics.py>)与[测试](<../../../../../touhou_rl/tests/test_tactical_metrics.py>)，为后续训练报告准备事件级诊断，不跑正式数据、不改现有cache /2计算协议。

- action：可靠正例集合一致率，未知行单独计数，不当错或安全成功。
- option：独立mask的可行集合一致率。
- target：独立mask MSE；仅预测/标签方向均非零时报告角度误差，零向量不伪造0°。
- termination：可靠标签的3×3混淆矩阵；padding无贡献。
- 同时给等事件平均和按有效行加权平均，避免长事件淹没小事件；各指标有各自有效分母。

没有从离线分类推导撤离提前量、返身、诱导污染或生存；这些必须由闭环事件轨迹评价。当前MVP训练器仍按validation loss选研究权重，本模块尚未集成或替代正式行为选模，不宣称P4门完成。

```powershell
python -m pytest touhou_rl/tests/test_tactical_metrics.py touhou_rl/tests/test_tactical_policy.py -q -p no:cacheprovider --basetemp .tmp/v373-metrics-round9-01
```

退出0，22passed in1.61s。测试覆盖unknown与独立mask、零向量角度undefined、termination padding、等事件均值与行加权差异。
