# v3.6.1 执行计划·POWER 补遗：T2 安全门的修正 + CollectPower 臂

- 日期：2026-10-25
- 依据：[v3.6.1 执行计划-T2](<./执行计划-T2.md>) §4.4、[T2-ENEMY 报告](<./reports/T2-ENEMY报告.md>) §4
- 性质：**预注册补遗**。写定后才开始 POWER 采集。
- 只改"安全门怎么评"，不改主指标、不改任何阈值方向、不改 ENEMY 的任何结论。

---

## 1 为什么要改安全门

ENEMY 的 20 对实测暴露了一个缺陷：**T2-E2（生存帧数比 ≥90%）退化**。

| 事实 | 数值 |
|---|---|
| practice stage 1 全长 | 约 16,000 帧 |
| `max_steps=3000` 覆盖 | 6,000 帧（37%） |
| 40 局中截断于 `max_steps` 的 | **40 / 40** |
| 两臂平均生存帧 | baseline 6002.7 / tactic 6002.7（**恒等**） |

也就是说 E2 在两臂上永远相等，判它"通过"只是因为它不可能失败。**一个不能失败的门不是门。**

两个可选修法：把 `max_steps` 提到 9000 让整面跑完（每局 5 分钟，40 局 ≈ 3.3 小时），或者换一个**在截断下仍有区分力**的安全量。本轮选后者，理由是：死亡与 bomb 使用在截断窗口内本来就是可观测的，而"活得更久"在固定时长对比里本来就没有意义。

---

## 2 修正后的安全门（POWER 与今后所有截断协议共用）

| 编号 | 判据 | 阈值 | 与旧版关系 |
|---|---|---:|---|
| T2-E3 | 每对**死亡次数**增量（tactic − baseline）的均值 | ≤ **+0.10** | 不变（ENEMY 轮已用） |
| T2-E4 | 每对**bomb 使用次数**增量的均值 | ≤ **+0.50** | **新增** |
| T2-E5 | 平均 clearance 的相对变化 | ≥ **−20%** | **新增** |
| ~~T2-E2~~ | ~~生存帧数比~~ | — | **停用**：截断下恒等，无区分力 |

安全仍是硬约束：三条里任一不过 → `fail_safety`，无论主指标多好。

E4 的理由：CollectPower 的收益可以用"多扔 bomb 换生存"来伪造，必须堵住。
E5 的理由：往线上跑会系统性降低 clearance，若降幅超过 20%，那就是"用安全换 power"，不能算战术收益。

---

## 3 POWER 臂的定义（沿用 v3.6.1 §4.4）

| 项 | 值 |
|---|---|
| primary | `safe_power_gain_per_frame` = Σ max(0, Δpower) over 帧，仅当 clearance ≥ **6 px**；除以这些帧数 |
| 通过阈值 | 相对提升 **≥15%** 且配对 bootstrap p<0.05 |
| 次指标 | P 收集数、Boss 出现时 power、峰值 power、`up_collect`/`hold_collect`/`unsafe` 次数、生存帧、死亡、bomb、贴墙率、动作熵 |

臂定义（`tactics.collect_power_override`，常数已固定）：

```text
当且仅当：power < 128
      且 存在屏内 power 类道具
      且 pilot 所选动作的 clearance ≥ 6 px
  则：player_y > 110 → 强制 [上移, 射击, 聚焦]
      player_y ≤ 110 → 强制 [停留, 射击, 聚焦]
否则：完全不干预（pilot 原动作）
```

中止原因逐决策记录（`power_max` / `no_p_item` / `unsafe` / `up_collect` / `hold_collect`）。

---

## 4 采集参数（与 ENEMY 轮一致，只换臂）

| 项 | 值 |
|---|---|
| 臂排布 | `--tactic none --tactic collect_power`，40 局交错 |
| `max_steps` | 3000（与 ENEMY 轮相同，便于横比） |
| `arm_seed` | **20261026**（不同种子，避免与 ENEMY 轮共用热身序列） |
| 难度 / 机体 / 起点 / frameskip | Lunatic / team 0 / practice stage 1 / 2 |

### 4.1 采前烟测（已做，写下作为预测依据）

2 局 × 1500 决策，同一热身：

| 臂 | power 结束值 | 覆盖动作 |
|---|---:|---|
| baseline | **8** | 无 |
| collect_power | **67** | up_collect 509、hold_collect 64、unsafe 35、no_p_item 841 |

**8 → 67**，约 8 倍。这说明臂本身有效；本轮要回答的是"在完整的 20 对上，安全代价是多少"。

---

## 5 预注册预测

| 编号 | 预测 | 若失败说明什么 |
|---|---|---|
| P-P1 | 主指标 ≥ +15%（预期远高于此） | 若不到 15%，说明安全窗口（clearance ≥6 px）恰好是 P 不可达的窗口，收集收益与安全严格互斥 |
| P-P2 | 死亡增量 ≤ +0.10 | 收集路线确实危险，战术必须降级为"极安全窗口才收" |
| P-P3 | clearance 相对降幅 ≤ 20% | 同上 |
| P-P4 | bomb 增量 ≤ +0.50 | 收集在用 bomb 兜底 |

P-P2/P-P3/P-P4 任一失败 → `fail_safety`，即使 P-P1 极好。

---

## 6 停止线

1. 可用对 < 16/20 → `underpowered`。
2. 任一臂 complete 率 < 95% → 先查记录器。
3. baseline 臂 power 终值与 ENEMY 轮的 baseline（均值 5.4、峰值 14.4）相差 >3 倍 → 环境变了，先对比设置。
4. 看到结果后不得改 6 px 安全门、110 px 收集带、32 px 归属半径或热身参数。
5. 不得用 ENEMY 轮的结论解释 POWER 的结果。

---

## 7 交付物

```text
Document/Update/Alpha/v3.6.1/
  执行计划-POWER补遗.md        # 本文件
  data/probe/t2-power/         # 40 局
  experiments/t2-power/paired.{json,md}
  reports/T2-POWER报告.md
```
