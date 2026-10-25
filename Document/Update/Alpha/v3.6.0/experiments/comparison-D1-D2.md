# v3.6.0 T1 消融：D1（局部 pilot）vs D2（人类回放）

- D1：`Document\Update\Alpha\v3.6.0\experiments\T1-ablation\ablation.json` — 9 局 / 21156 决策；速度口径 `config`；LOEO 9 折；生存标签 death_60 (auc)
- D2：`Document\Update\Alpha\v3.6.0\experiments\T1-ablation-D2\ablation.json` — 6 局 / 30000 决策；速度口径 `measure`；LOEO 6 折；生存标签 min_clearance_60 (spearman)

> 两个运行的模型、标签定义、阈值与置换次数完全相同；差别只来自**数据源**、**折数**、**速度口径**，
> 以及 D2 因 `death_60` 阳性为 0 而触发的**预注册生存标签回退**。
> 因此**绝对值不可直接比较**，只比较增量方向与是否过门。

## 1 逐策略对照

| 策略 | D1 判定 | D1 增量 | D1 p | D2 判定 | D2 增量 | D2 p | 方向一致 | 变化 |
|---|---|---:|---:|---|---:|---:|---|---|
| LocalDodge | ✅ pass | n/a | n/a | ✅ pass | n/a | n/a | — |  |
| REGION | ❌ fail | -0.0148 | 1 | ❌ fail | -0.0018 | 1 | 是 |  |
| LURE | ❌ fail | +0.0203 | 0.005 | ❌ fail | +0.0014 | 0.005 | 是 |  |
| ENEMY | ❌ fail | +0.0699 | 0.005 | ❌ fail | +0.1100 | 0.005 | 是 |  |
| POWER | ✅ pass | +0.0405 | 0.005 | ✅ pass | +0.0053 | 0.005 | 是 |  |
| BOSS | ⚠ unstable | -0.0107 | 0.990 | ❌ fail | -0.0668 | 1 | 是 | unstable → **fail** |
| EMERGENCY | ❌ fail | +0.0022 | 0.005 | ⚠ underpowered | n/a | n/a | — | fail → **underpowered** |
| HISTORY | ❌ fail | -0.0004 | 0.990 | ❌ fail | +0.0057 | 0.005 | **否** |  |

## 2 全局门

| 门 | D1 | D2 |
|---|---|---|
| G-1 pass 策略数 >= 3 | 2 → 未过 | 2 → 未过 |
| G-2 通过块的增量点估计为正且 p<0.05 | 0 → 通过 | 0 → 通过 |
| G-3 ALL vs BASE death_60 (auc) vs min_clearance_60 (spearman) ≥ +0.05 | -0.0210 → 未过 | +0.0084 → 未过 |
| G-4 无块的置换 p 与主结果符号相反 | 0 → 通过 | 0 → 通过 |

- D1 `enters_T2` = **False**；D2 `enters_T2` = **False**
- 增量方向翻转的策略：['HISTORY']

## 3 预注册预测核对（预测写在 D2 采集之前）

| 编号 | 预测 | 结果 | 证据 |
|---|---|---|---|
| P1 | POWER 在 D2 仍然 pass | 命中 | D2 verdict=pass |
| P2 | EMERGENCY 在 D2 的增量显著高于 D1 | 不可判（标签功效不足） | D1 Δ=+0.0022；D2 判定=underpowered |
| P3 | REGION 在 D2 的生存增量仍 ≤ 0 | 命中 | D2 Δ=-0.0018 |
| P4 | BOSS 在 D2 从 unstable 变为可判（stable） | 命中 | D2 verdict=fail stability=stable |
| P5 | HISTORY 仍然 fail | 命中 | D2 verdict=fail |
| P6 | death_60 落入 underpowered 分支（阳性 <100） | 命中 | death_60 阳性 = 0 |

## 4 口径与诊断差异

| 项 | D1 | D2 |
|---|---|---|
| 生存标签 | death_60 (auc) | min_clearance_60 (spearman) |
| 空特征 pooled 指标 | 0.3170 | -0.0792 |
| 空特征局均指标 | 0.5000 | n/a |
| 近常量二值标签 | [] | ['p_collected_60', 'bomb_used_60'] |
| 零散布置换单元格 | [] | [] |
| unstable 单元格 | ['BOSS'] | [] |
| pooled/局均符号不一致 | ['BOSS', 'EMERGENCY::bomb_used_60'] | [] |

## 5 各局实测移动速度（D2，`--speed-mode measure`）

| episode | speed | focus_speed | 轴样本 | 回退 |
|---|---:|---:|---:|---|
| replay-ep01 | 5 | 1.90 | 1279 | — |
| replay-ep02 | 5 | 1.90 | 1396 | — |
| replay-ep03 | 4 | 2 | 1318 | — |
| replay-ep04 | 5 | 2.20 | 1276 | — |
| replay-ep05 | 4 | 2.30 | 1451 | — |
| replay-ep06 | 5 | 1.90 | 1078 | — |

