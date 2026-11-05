# 真机公平配对预注册 v1

- 状态：blocked: game_required
- 适用版本：v3.7.5
- 目的：在共同管线和等逻辑帧曝光下比较 baseline 与 subject
- 当前限制：本机没有游戏，未执行

## 协议

- 先 shadow，再进行探索性 AB/BA；
- 两臂使用同一截图、输入后端、frameskip、资源和 reset 协议；
- 以 gameplay_frame 固定窗口结束，不以决策次数结束；
- 记录 start_frame、end_frame、send_frame、effective_frame、hold 和分项延迟；
- 死亡前保留至少 120 帧；
- 区分观测、预测、路线、执行、输入延迟、资源和 unknown 归因。

## 探索规模

首轮建议 10 对 AB/BA。该规模只能验证管线和估计方差，不能声明改善。

## 进入独立确认

必须先通过 P5 闭环门和 P6 时序门，再冻结独立测试集、样本量、阈值和排除规则。

## 主门

固定逻辑帧窗口死亡数下降目标至少 20%，配对差 95% 区间上界小于 0；bomb 和局部生存不得明显退化。

没有游戏时保持 blocked，不使用合成结果代替真机结果。
