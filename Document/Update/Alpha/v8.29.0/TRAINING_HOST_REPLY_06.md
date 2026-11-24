# V829-PAIR-01 §3 实施回执（五）｜两档试点形状打通 · REHEARSAL（非结论）

> ⚠️ **本文数字全部是管线演练值，不是关于 G1 与 T2 的证据。**
> 8 个「祖先」由被评测策略自己滚出的前缀生成，只有 8 个 seed。判定 `UNDETERMINED` 只说明**报表拒绝乱下结论**，
> 不说明任何策略优劣。任何 Δ 都不得被引用。

| 项 | 值 |
| --- | --- |
| 日期/时区 | 2026-11-24，`+08:00`（Asia/Shanghai） |
| 实测窗口 | 12:30:09 → 12:37:20（+0800） |
| 主机代号 | `DSH-WIN-01` / `DSH-WSL-01` |
| 依据 | `PREREGISTRATION_PAIR_001.md` §2、§4 |
| 性质 | 演练；不训练、不恢复 WP05 |

---

## 1. 本轮修掉的两个真问题

预注册 §2 的形状是「每祖先 × 3 ckpt × 2 配对 RNG × **2 个预算档**」，且**两档必须从同一接管帧开跑**。
这两个约束各自撞上一个设计错误：

| # | 症状 | 根因 | 修法 |
| ---: | --- | --- | --- |
| 1 | 同一祖先的两档 root 被守卫判 `ANCESTOR_INCONSISTENT` + `DUPLICATE_ANCESTOR` | `budget_frames` 被当成「祖先自洽字段」与「祖先身份键」的一部分 | 自洽字段去掉预算；新增 `ANCESTOR_BUDGET_SET`（档位须是 {1800,2400} 子集）；`DUPLICATE_ANCESTOR` 键加预算；builder 加顶层 `budgets:[1800,2400]` 轴 |
| 2 | 混合两档的清单被 `BUDGET_ROOT_MISMATCH` **整份拒绝** | `--budget-frames` 被实现成「所有 root 必须同档」的断言 | 改成**选择器**（与 `--window` 同类）：先 window 后 budget，剩者为本次 root 集；空集 → `BUDGET_TIER_ABSENT`；单档清单输出**逐字节不变** |

修完后：守卫 `selftest 57/57`；`--budget-frames 1800` 在 96-root 混合清单上 `selected 48 root(s), skipped 48 (other tiers)`，
`2400` 对称；两档 `--mode check` 都 `would_start_engine:false`。

## 2. 96-root 清单（预注册的 cell 数）

`8 祖先 × 3 ckpt × 2 RNG × 2 档 = 96`，守卫**两档各 584 项检查全过**、不加 `--allow-exposed`：

```text
ancestors 8 · per-ancestor 12 · ckpt 3 · pair_index {0,1} · budgets {1800,2400}
pair_rng 96 个全不重复 · takeover_frame 全为 2399 · total_frames 全为 4800 · headroom {1, 601}
```

## 3. 两档真机（各自的 48 cell / 96 单臂）

| 档位 | 墙钟 | 行数 | 满段 | PARK | 子进程 | invalid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1800 帧（30 s） | **176 s** | 96 | 4 | 8（省 40） | 96 | 0 |
| 2400 帧（40 s） | **168 s** | 96 | 0 | 8（省 40） | 96 | 0 |
| 合计 | **344 s** | 192 | 4 | 16 | 192 | 0 |

全部 `th08-modern` 前后 0/0、私有 cfg 四字段全 0、`audible=0`。

## 4. 报表层首次「全门通过」

```text
GATES ok=True
  DATA_INTEGRITY        ok   no hard row-level refusal
  ANCESTOR_COUNT        ok   8 ancestor(s) in the rows
  MANIFEST_SINGLE_TABLE ok   one frozen manifest digests every row
  SCORING_EVIDENCE      ok   no pair whose score disagrees with its raw fields
  CHECKPOINT_AXIS       ok   checkpoint groups per budget {1800: 3, 2400: 3} vs expected 3
Δ30 = 0.000000 (+0.00 pp)  pos/tie/neg = 0/8/0
Δ40 = 0.000000 (+0.00 pp)  pos/tie/neg = 0/8/0
full-rate G1 2/48 = 4.17% · T2 2/48 = 4.17%（30 s）；30 s 档；40 s 档两者皆 0/48
inventory: 192 rows, 72 deduction(s), 48 exhausted, 0 death-with-bomb, 144 insufficient
verdict primary_30s: UNDETERMINED
```

**这是整条链第一次跑完整主批**：从 96 root 清单 → 两档真机 → 5 个门全过 → 判定路径真的产出结论。
逐祖先 2×2 全部是 `neither`（两臂都死）或并列，没有任何一格出现臂间差异。

## 5. 成本修正（我上一轮的外推偏乐观）

| 量 | 上一轮估计 | 本轮实测 |
| --- | ---: | ---: |
| 单档 48 cell / 96 臂 | 71 s | **176 s（30 s 档）/ 168 s（40 s 档）** |
| 完整两档试点 | ≈ 2.5 min | **≈ 6 min（344 s）** |

原因：上一轮取 `takeover=2999`，祖先在那个起点上死得早（很多格 100–400 帧就结束）；
本轮取预注册要求的 `takeover=2399`，回合更长（1800/2400 帧跑满的格变多），单臂时间随之上升。
⇒ **排期按 6 分钟/完整两档试点**，另加 8 盘采集的 ~34 s。

## 6. 一个必须转达的设计约束：接管帧 vs 预算档

本轮实测把这条硬冲突钉死了：

| tape 长度 | 接管帧 N | 1800 档 | 2400 档 | 演练结果 |
| ---: | ---: | --- | --- | --- |
| 4,800 帧 | 2,999 | ✓ | **✗**（2999+2400=5399 > 4800） | 只能跑单档 |
| 4,800 帧 | 2,399 | ✓ | ✓ | 两档都能跑，**但 192 格无任何臂间差异**（Δ=0） |
| **5,600 帧** | **3,199** | ✓（余 601） | ✓（余 1） | 已实测 build + guard 双档通过 |

⇒ **要「更晚的接管帧」+「两档都跑」，就必须采更长的录像**。
实测成本：1,400 块（5,600 帧）**9.6 秒**采完、0 死亡、哈希链 1,401 检查 0 失配。
对照：同一盘 5,600 帧录像用 `N=3399`（3399+2400=5799 > 5600）会被 builder 拒 `BUDGET_EXCEEDS_TOTAL`。

## 7. 合规

本轮 **17 次模板 spawn**：两档各 8 次 PARK + 1 次长录像采集；另有 192 个 fork 子进程。
会话累计 **69 次模板 spawn**。未训练、未恢复 WP05、未改既有资产；新增产物仅在
`.tmp_v829_collect/seed3301-long/`、`.tmp_v829_info/s20|s21/`、`.tmp_v829_pair/eval/`。

## 8. 剩余

1. **2400 帧档在 `N=2399` 下全部早死**（0 满段）⇒ 若研究端希望 40 s 档也有信息量，接管帧不宜太早。
2. 祖先筛选准则（研究端定）：8 个新采祖先在 `N=2399` 下依然 0 臂间差异，说明**该起点没有区分力**。
3. 自然起点策略（研究端定）。
