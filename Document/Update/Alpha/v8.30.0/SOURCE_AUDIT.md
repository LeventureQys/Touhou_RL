# v8.30｜放雷头 BC 目标与读出的源码审计

**第二稿定位：** 本文的源码发现保留为局部实现依据。是否开展 BC 初始化研究由 [路线决策](ROUTE_DECISION.md) 决定；修复此问题不是选择主路线的充分条件，也不自动启用克隆或 RL 预算。

2026-11-24，GPT 研究端。本次仅审阅研究端检出及做标准库数学复算；未运行训练端引擎、未加载权重、未恢复 WP05。基点为 `539a4e4f9261fff46c21a66746ad18d6bc7b72a1`。这里的 SHA 是研究端 Windows 工作副本身份，不能替代训练端历史实跑身份。

## 1. 已确认的实现问题

四个 runner 的 BC 热启动以双输出网络的第 1 个 logit `z1` 单独计算 `BCEWithLogitsLoss(z1, y)`；实际推理、训练后克隆指标和后续 REINFORCE 使用的是两个 logit 的 softmax。模型第 0 个输出 `z0` 未固定为零，因此这两种概率不同：

```text
BC 优化的概率：       sigmoid(z1)
部署所用的放雷概率： softmax([z0, z1])[1] = sigmoid(z1 - z0)
一致的目标：         CE([z0, z1], y) = BCEWithLogitsLoss(z1 - z0, y)
```

| 研究端源码 | BC 定义及使用位置 | softmax/argmax 克隆指标位置 | SHA256 |
| --- | --- | --- | --- |
| `tools/rl/v8m0_bombrl.py` | 960、977–979 行 | 989–992 行 | `fd0ac4ef82772ec8b6ac9820268d5d15a4d3e21191b1dba1c58ba63475e4cb41` |
| `tools/rl/v8p0_bombrl900.py` | 1012、1029–1031 行 | 1041–1044 行 | `71bfbeb6011179d0373896c1efdd973a760eb3f1d7adad55478cc2a6511d1222` |
| `tools/rl/v8t0_densereward.py` | 1050、1067–1069 行 | 1079–1082 行 | `e7f004e5dc71c6e77cf83373c8daa95993f1bdb5a8c52910504b6408239221e1` |
| `tools/rl/v8u0_features.py` | 1190、1207–1209 行 | 1219–1222 行 | `c4cde38f954f20433932c3dade77afb5882c0a42bb0c94070c8bee380a848aac` |

`v8u0_features.py:563–571` 创建 `Linear(109,64) → Tanh → Linear(64,2)`，没有固定 `z0` 的约束。BC 注释“two logits == one logit”漏掉了 logit 差值。其在线动作在 718–726 行使用双输出 softmax/argmax；REINFORCE 更新在 1293–1298 行使用双输出 `log_softmax`，本次没有发现该处同样使用单 logit。

影响范围限定为：**热启动优化目标与部署概率不一致**。这可能影响初始化校准、动作分布及后续优化起点；尚不能判定影响大小，也不能把全部历史失败归因于它。

## 2. 可复算的数学证据

对正标签 `y=1`，取 logits `[4,3]`，旧 BC loss 为 `0.04858735`，部署放雷概率却为 `0.26894142`，对应 CE 为 `1.31326169`。将两个 logit 同减 3 得到 `[1,0]`，部署概率及 CE 不变，旧 BC loss 却变成 `0.69314718`。

旧目标对两个输出 logit 的梯度为 `[0, sigmoid(z1)-y]`；一致目标为 `[y-p, p-y]`，其中 `p=sigmoid(z1-z0)`。旧目标对第 0 输出没有直接梯度；共享隐层仍会影响两个输出，不能把第 0 logit 称为整个训练中完全不变。

研究端已执行以下标准库命令，退出码 **0**，三个检查全部成立。此处是数学复算，不是 PyTorch 实现测试或真实克隆验收。

```powershell
@'
import math
sp=lambda x: max(x,0.0)+math.log1p(math.exp(-abs(x)))
for z0,z1 in ((4.,3.),(1.,0.)):
    p=1/(1+math.exp(z0-z1))
    print({'logits':[z0,z1], 'p_bomb_softmax':p,
           'old_BCE_y1':sp(-z1), 'CE_y1':sp(z0-z1),
           'BCE_margin_y1':sp(-(z1-z0)),
           'grad_old':[0.,1/(1+math.exp(-z1))-1],
           'grad_CE':[1-p,p-1]})
    assert abs(sp(z0-z1)-sp(-(z1-z0))) < 1e-12
assert abs(sp(4.-3.)-sp(1.-0.)) < 1e-12
assert abs(sp(-3.)-sp(0.)) > 0.6
print('math_checks=3/3; source_execution=none; engine_execution=none')
'@ | python -B -
```

## 3. 历史结论应保留的边界

- v8.28 发布记录 `evidence/FEATURES/RUN.md:233–235` 声称从前版复制且 BC 数学未动，与本次发现的复制链相容；这仍不能证明四次历史实跑都执行了当前文件。须回传每次实跑 runner、关键依赖、BC 产物和数据的 SHA，以及原命令、日志和退出码。拿不到时标为“历史实现身份未核验”。
- G1/T2 的移动是 softmax 抽样，而 v8.28 学习头的冻结移动采用 argmax。两者成绩同时含移动读出差异。v8.29 `README.md:17` 已明确承认；本次不能把历史成绩差全部解释为放雷能力差。
- G1 的阈值直接使用本档剩余预算（`v8r0_urgency.py:584–592,704–706`），20/30/40 秒不是同一固定策略的嵌套生存曲线。整面部署必须另定义可用预算输入。
- G1 历史 30、40 秒首雷请求也在 decision 0（v8.26 `extra_numbers.md:79–80`）。首步请求放雷不能单独作为克隆失败或策略失败门；须核对实际扣库存和其后的状态。
- 四个旧开发窗 11/12/17/18 已用于 BC/RL 及开发，不能充当新的独立留出集。修改损失不能消除这种曝光。

## 4. 下一步只回答一个局部问题

按 [V830-BC-01](PREREGISTRATION_BC_001.md) 比较旧目标 L0 与一致目标 L1：相同网络、初值、数据、逐步 batch 索引、优化器和预算，先核验数学及离线克隆，再做一个 seed 的真实克隆冒烟。规则教师冻结为 **60/40/20 px**，不同时换成 T2 或 G1。v8.29 的新祖先仍用于其自己的配对比较，不能进入本轮拟合或校准。

源码常数已核对：`v8u0_features.py:198–208` 为梯度裁剪 1.0、200 steps、**batch 256（128 正例 + 128 负例）**、Adam 学习率 `1e-2`、正例采样比例 0.5。保持这些常数才能把本轮变化限定到损失定义。类别重采样的校准问题另行记录，本轮不同时修正。

本版仅产生源码审计与实验设计；未修复业务代码、未完成 PyTorch 梯度测试、未完成离线训练或真实引擎验证，也不支持关于“学习上限”的结论。

## 5. 身份复核命令

以下命令已在研究端执行，退出码均为 **0**：

```powershell
git rev-parse HEAD
Get-FileHash -Algorithm SHA256 -LiteralPath tools\rl\v8m0_bombrl.py,tools\rl\v8p0_bombrl900.py,tools\rl\v8t0_densereward.py,tools\rl\v8u0_features.py | Select-Object Path,Hash
rg -n -C 4 "BCEWithLogitsLoss|two logits|full_logits" tools/rl/v8m0_bombrl.py tools/rl/v8p0_bombrl900.py tools/rl/v8t0_densereward.py tools/rl/v8u0_features.py
```

本文行号与 SHA 均绑定该研究端检出。训练端如文件身份不同，先报告差异，不得以同名路径代替同一版本证明。
