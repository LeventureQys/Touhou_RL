# P4合成渲染像素首次学习实测（续轮10）

## 身份与门禁

**这是合成渲染像素的冻结特征可学习性研究，不是P3真实像素、不算正式T21，不是可部署候选。** 真实alignment/teacher/data门均未改成passed，默认模型未替换。独立训练入口拒绝真实manifest和real_pixel_training标志，没有把生产gate加bypass。

[预登记](<../evidence/synthetic-learning-round10-prereg.md>)先于优化。数据16完整事件，train8/validation8，按8模板group分离，镜像同split；来源为已暴露R8开发/验证组，不触及封存test/R9确认。每事件80步，当前像素渲染不加入未来teacher目标/option/clock提示；隐墙出现前不可见、F3未来时窗未通知，所以任务并不保证完全可辨识。

## 实际运行

```powershell
python -m tools.rl.v373_synthetic_learning --records Document/Update/Alpha/v3.7.3/evidence/synthetic-learning-round10/data/records.json --baseline third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt --out Document/Update/Alpha/v3.7.3/evidence/synthetic-learning-round10/run-s7-n1-200 --steps 200 --device cuda
```

首次退出1：vision loader的map_location不自动把module参数移到CUDA，输入GPU而权重CPU。优化尚未开始、输出目录未创建。修复为显式`.to(device)`后同命令退出0，不覆盖已有训练run。

真实RTX4070 SUPER CUDA执行：

| 项目 | 实测 |
|---|---:|
| 冻结真实v3-st编码16×80步 | 2.83s |
| N1优化200步 | 10.61s |
| 优化吞吐 | 18.86step/s |
| action单路线一致率 train | 0 → 93.75% |
| action单路线一致率 validation | 0 → 91.41% |
| 第一步/末步loss | 3.01578 / 0.30614 |
| 基座逐参数不变 | 是 |

完整[run报告](<../evidence/synthetic-learning-round10/run-s7-n1-200/report.json>)、[loss](<../evidence/synthetic-learning-round10/run-s7-n1-200/losses.json>)、[artifact SHA](<../evidence/synthetic-learning-round10/run-s7-n1-200/sha256.json>)。研究权重使用`v373-synthetic-render-only-heads/1`，正式部署loader不能当作tactical checkpoint加载。

## 与正式计划的差异

- 这是80步整事件、batch1、单seed7固定200步研究；不是正式burnin16+seq32/batch4/三seed四臂比较。
- 只action正例标签，option/target/termination无可靠监督所以不报准确率。不将未训练aux头视作战术理解。
- 仅教师轨迹单一路线正例，其他动作并非都已证伪；标签一致率不是安全/最优动作集合覆盖。
- 未按验证选步数或checkpoint，报告固定末步；验证仍属开发诊断，不是封存test。
- 未做网络闭环世界重置、网络自身失败状态补采；不能从91.41%推出生存收益。
- train未达到计划候选95%小集一致率，且任务信息不足/风格差异，不据此扩大模型或盲加epoch。

## 工程验证

```powershell
python -m pytest touhou_rl/tests/test_tactical_synthetic_pixels.py touhou_rl/tests/test_tactical_synthetic_learning.py touhou_rl/tests/test_tactical_metrics.py -q -p no:cacheprovider --basetemp .tmp/v373-round10-render-metrics-final-01
```

退出0，12passed in1.21s。下一步固定当前run作B0/B1/N0诊断与逐族/闭环失效检查，先确认预测是否只学动作频率/时间节奏；正式真实训练仍需真实数据门。
