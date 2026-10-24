# v3.5.0 S0：P 点契约实验报告

- 日期：2026-10-24
- 依据：[P点契约实验（预注册）](<./contract/P点契约实验.md>)、[状态与事件契约](<./contract/状态与事件契约.md>)
- 性质：S0 的 P 点行为契约验证；不训练策略，不改动作空间
- 结论：**H-P1…H-P6 全部通过**（受控臂 + replay 臂），S3 的 `CollectPower` 具备可执行契约
- 证据：`probe/p-contract/`（`episodes.jsonl`、`trace/*.jsonl`、`bullets/*.npz`、`manifest.json`、`p-contract.json`）

---

## 1 一页数字

| 量 | 实测 | 来源 |
|---|---:|---|
| `pointItemValueLine`（自动收集线） | **128.0** | `PlayerRawShtFile + 0x1C`，6 局全部一致 |
| 首次 auto-collect 时自机 y | **124.627**（前一条 128.627） | 受控臂 ep01，decision 468 |
| 线与采样区间 | 128.0 ∈ [124.627, 128.627]，偏差 3.373 px（2 帧步长 4 px 内） | 同上 |
| `itemCollectionBoxSize` | 24.0 | shot 参数 |
| `itemAutoCollectSpeed` | 10.0 | shot 参数 |
| power 上限 | **128.0**，无 >128 记录 | replay 臂 ep04 起手即 128.0；全部 18,000+ 决策 `max_power=128` |
| 满 power 后正增长 | **0 / 16 窗口** | H-P2 |
| Shift 最长连续聚焦 | **428 决策**；松开后 2 决策内 `focus_mode=0` | H-P4 |
| 触发 crossing 的 P 漏检 | **0 / 51 crossings**（1 个 crossing 因 `player_state=2` 排除） | H-P5 |
| power 增量的未解释窗口 | **0**（P 消失或 `frameEventFlags&0x40` 解释） | H-P6 |

## 2 假设逐条

| 编号 | 判据（预注册） | 实测 | 判定 |
|---|---|---|---|
| H-P1 | 至少一次 power=128；无 >128 | max 128.0，over_max 0 | **通过** |
| H-P2 | 满后无正增长 | 16 个“上一决策已满”的窗口，0 次正增长 | **通过** |
| H-P3 | A 臂首次 transition：线在 `[cur_y, prev_y]`，偏差 ≤4.5 px，线 120–136 | 128.0 ∈ [124.627, 128.627]，diff 3.373，line 128.0 | **通过** |
| H-P4 | 聚焦连续 ≥10 决策且松开后 2 决策内归零 | 最长 428，release_ok=true | **通过** |
| H-P5 | 触发且存活 crossing 的全部 in-use P 在 ≤2 决策进入 AUTOCOLLECT，漏检 ≤1 | 51 crossings，missed_total 0，排除死亡 crossing 1 | **通过** |
| H-P6 | power 正增长必须由消失 P 或 ITEM_COLLECTED 解释 | 86 个消失窗口，unexplained 0 | **通过** |

## 3 样本与臂

| 臂 | episode | 决策 | 场景 | 用途 |
|---|---|---:|---|---|
| A 受控 | `p-contract-ep01` | 3,000 | Lunatic / Border Team / 练习 Stage 1；`AvoidPilot` + 上移聚焦覆盖 | H-P3/H-P4/H-P5 |
| A 受控 | `p-contract-ep02` | 3,000 | 同上，独立一局 | 重复性 |
| B replay | `replay-ep01` | 6,000 | `Th8_ud2558.rpy` 第 1 面，人类输入 | H-P1/P2/P5/P6 规模 |
| B replay | `replay-ep02` | 6,000 | 同上 | 重复 |
| B replay | `replay-ep03` | 8,000 | 同上，延长到 16k 帧 | power 升到 116，未满 |
| B replay | `replay-ep04` | 400 | 同一 replay 从第 2 面起播 | power 起手 128，验证上限 |

总决策 **26,400**（A 臂 6,000；B 臂 20,400）。所有 episode 的 `complete_rate` ≥ 0.995（A 两局 1.0 / 0.9997；B 局 0.996–1.0）。

## 4 关键读数解释

### 4.1 线在哪里

- `pointItemValueLine` 不是社区口口相传的“屏幕 25%”，而是 shot 文件里的 **场地 y=128.0**；
- 场地范围 y=16..464，128 正好是 1/4 高度，所以“25% 线”的说法在数值上成立，但必须以字段为准；
- 受控臂在 128.627 → 124.627 的决策里触发首个 auto-collect；线落在区间内，偏差 3.373 px 完全由 frameskip=2 的 4 px 步长解释。

### 4.2 为什么 replay 臂的“首次 transition”不在线上

replay 臂的人类第一次越过 128 时没有按 Shift（power<128），因此不触发 auto-collect；他后来在 y≈86 聚焦时才触发。这正是契约 R2 把 H-P3 限定在受控臂的原因：**线是触发条件的阈值，不是“任何越线都会收集”**。

### 4.3 收集动作的时间关系

- `frameEventFlags & 0x40` 与 power 增量在同一决策或相邻决策内出现；R5 的 ±1 窗口把所有 86 个消失窗口解释完；
- 未出现“power 增但无 P 消失且无 ITEM_COLLECTED”的窗口；
- 未出现“P 消失但 power 下降”的窗口。

## 5 预注册修订（R1–R7）

见 [P点契约实验](<./contract/P点契约实验.md>) §0。全部修订发生在正式 P 契约数据之前；阈值方向未改变：

- R1 replay 臂：teacher+覆盖 2,400 帧内最高 power 37，无法观测 H-P1；
- R2 H-P3 只判受控臂；
- R3 H-P5 只判“触发条件成立”的 crossing；
- R4 允许 `frameEventFlags` 解释盲窗收集；
- R5 帧事件按轮的每次都 OR + ±1 决策归属窗口；
- R6 H-P3 分辨率改为 4 px 步长 + 区间；
- R7 H-P5 排除 DYING/SPAWNING 的 crossing（源码明确不自动收集）。

## 6 对 S3 的直接约束

```text
CollectPower 只在以下全成立时启动：
  power < 128.0
  屏幕内存在 type ∈ {0,2,4} 且 isInUse 的 P
  player_state ∈ {0,3}                # DYING/SPAWNING 不收集
  自机可安全到达 y < 128.0 的路线     # 路线风险由 S2/S3 另判
启动后：
  hold Shift（focus_mode=1）
  移动到 y < 128.0
  等待全部 in-use P 进入 state=1
  收集完成后离开该区域
中止：
  区域风险超过阈值
  预计路线穿越危险区
  power 已达 128（不再为 P 冒险）
```

**不包含**：收集路线本身的安全性、收集期间的大范围弹幕规避、收集对 Boss 输出的机会成本。这些仍是 S2/S3 的工作。

## 7 本实验没有证明的

- 不证明“为了 P 值得放弃多少 clearance”；
- 不证明收集路线能在 S2 的大范围危险区里安全；
- 不证明 power 满后仍应收集（本实验只证明“满后不再涨”，策略上应停止）；
- 不证明 Boss 战中的收集时机。

## 8 复现

```powershell
# A 臂
python -m touhou_rl.probe35 p-contract `
  --out Document/Update/Alpha/v3.5.0/probe/p-contract `
  --run-dir .tmp/v350-p-contract-session --episodes 2 --max-steps 3000

# B 臂（追加）
python -m touhou_rl.probe35 p-contract --append --episode-start 3 `
  --out Document/Update/Alpha/v3.5.0/probe/p-contract `
  --run-dir .tmp/v350-p-replay-session --episodes 1 --max-steps 8000 `
  --replay th08_replay/Th8_ud2558.rpy

# C 上限观测（第 2 面起播，power 起手 128）
python -m touhou_rl.probe35 p-contract --append --episode-start 4 `
  --out Document/Update/Alpha/v3.5.0/probe/p-contract `
  --run-dir .tmp/v350-p-replay-stage2 --episodes 1 --max-steps 400 `
  --replay th08_replay/Th8_ud2558.rpy --replay-stage 1
```

> 注：B 臂 6,000 决策 ×2 与 8,000 决策 ×1 是同一 replay 的多次延长；`episodes.jsonl` 已按 episode 去重，保留最后一次运行的行。
