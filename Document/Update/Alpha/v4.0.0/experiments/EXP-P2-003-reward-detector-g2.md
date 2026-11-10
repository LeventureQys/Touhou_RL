# EXP-P2-003 像素回报检测器 v0 + G2 指标

- 日期 / 工作区：2026-11-09 / P2-WP3（回报检测器 v0）、P2-WP4 轻量版（96×112 vs 64×64）。
- 依 12-P2 §3.3（1/4 网格）、§4（G2 门）；前置 [EXP-P2-002](<EXP-P2-002-labels-v2-events.md>)（标签 v2、EVENTS 真实位、HUD 延迟 0–1 帧）、[EXP-P0-030](<EXP-P0-030-WP4-WSL-16-instance-stress.md>)（16 实例）。
- 结论：**数据与管线打通；G2 只过 lives/bombs 两条，death 三条（召回/精确/延迟）与 score 符号一条未过**。全部数字都是**检测器成绩**，不是 RL 成绩。检测器 v0 的失败原因是可定位的（见 §7），不是数据造假或口径模糊。

## 1. 预注册（跑之前固定）

1. 标签只来自 `EVENTS`（`0x1 bomb` / `0x2 hit` / `0x4 death` / `0x200 dying`）与 `STATE`（lives/bombs/score/…）；**像素只作输入，从不反推标签**。
2. death 头预测"未来帧的 death 位"：`window=1` 指下一帧，`window=4` 指 `t+1..t+4` 内出现 death 位。两种都在 §6 如实给出。
3. 切分按**关卡整段**：train 关卡 {0,1,2,4} / val {5} / test {3,6}。同一 shard（同一局、同一 seed）只属于一个 split；`build_split()` 会因"关卡的 shard 跨 split""引擎 SHA 混用""标签版本混用""split 空"直接 fail-closed。
4. 阈值只在 val 上选（帧级 F1 最大），test 只跑一次；不允许用 test 调阈值。
5. 置信下界：比例用 Wilson 单侧 95%，零错误用 rule-of-three（`1-3/n`）。G2 的 99% 门需要 ≥299 个零错误正例，99.5% 门需要 ≥599。
6. 不重建 build、不改引擎/协议/依赖、不删文件。

## 2. 数据规模（真实）

| 项 | 值 |
|---|---|
| 帧数 | **300,000**（16 实例 × 128 shard，全部 `complete=true`，16/16 worker 退出码 0） |
| 墙钟 | 483.8 s，聚合 **~620 fps**（16 实例、`LP_NUM_THREADS=4`、单 Xvfb） |
| 像素体积 | **7,142,400,000 B = 6.65 GiB**（≤10 GB 门）：`play96` 3.0 GiB + `play64` 1.14 GiB + `hud` 2.5 GiB；每帧 23,808 B |
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl_hazard2/th08-modern`，SHA `f172399f61b75baa8e62be46041ef756a710759f439e7fb07111f4747c78264e`，128/128 shard 一致 |
| 关卡 | 0:38,400 / 1:38,400 / 2:76,800 / 3:69,600 / 4:38,400 / 5:38,400（**关卡 6 未采到**：分片计划没轮到 (6,2) 就用完了预算，如实记录） |
| 难度 | 1:108,000 / 2:76,800 / 3:115,200 |
| 策略 | `dive` 153,600 / `random` 76,800 / `still` 69,600 帧 |

### 2.1 正例数（真实 `EVENTS` 位）

| 事件 | 帧级正例 | 事件簇 | 门（≥299） |
|---|---|---|---|
| `death` (0x4) | **354** | 354 | **达标**（零错误 95% 下界 = 1−3/354 = **99.15%**） |
| `bomb` (0x1) | 114 | 114 | 未达标 |
| `hit` (0x2) | 19,056 | 4,588 | 达标 |
| `dying_started` (0x200) | 357 | 357 | — |
| `effect_spawned` (0x400) | 216,247 | — | — |

`score` 增量符号：上升 69,749 帧 / 不变 230,123 帧 / **下降 0 帧**。送死策略**没有**产生扣分样本，与 P2-WP2 的 0 一致；score 符号的负向类在本数据集里仍无样本。

### 2.2 为什么必须加 `dive` 策略（关键实测）

任务原本假定 `random`/`still` 送死率高。**实测不成立**：

| 探针 | 帧数 | death | 每 1000 帧 |
|---|---|---|---|
| 16 实例 30 s，`random`+`still` 计划 | 16,161 | 1 | **0.06** |
| 16 实例 30 s，`dive` | 17,478 | 46 | **2.63** |
| 主跑（2/3 dive） | 300,000 | 354 | 1.18 |

按 0.06/1000 帧，300k 帧只能得到 ~18 个 death 正例，G2 的 299 门**不可能**达到。因此在采集器里加了显式的采集策略 `dive`（一直按 `UP|SHOOT` 停在场地顶边，那里是敌机与弹幕的生成处），它是**采集参数**（和 `bomb_prob` 同级），逐 shard 记录在 `spec.source` 里。主跑策略环 = `random,still,dive,dive,dive,dive`。**354 个 death 里绝大多数来自 dive**，G2 表给按策略分解（§6.3），不把 dive 的正例说成 random 的成绩。

## 3. 观测规格（存储的是什么）

| 平面 | 形状 | 内容 | 每帧字节 |
|---|---|---|---|
| `play96` | 96×112 uint8 | 场地 384×448（屏幕 (32,16) 起）4×4 块均值，BT.601 整数亮度 `(77R+151G+28B)>>8` | 10,752 |
| `play64` | 64×64 uint8 | 同一场地 6×7 块均值（原生 64，不是从 96 重采样） | 4,096 |
| `hud` | 56×160 uint8 | 三条数字带**原生分辨率**绿色通道：score 行 48–72、lives 80–96、bombs 96–112，列 480–640 | 8,960 |

**HUD 为什么不下采样**：把同样的带子降 2× 后（80×64），`000000000` 的 4×4 数字糊成 2×2 团块、lives 的星形图标不可数——≥99.5% 的读数门在下采样 HUD 上不可能达成。因此 `hud` 平面按原生分辨率存，`play` 平面承担 96/64 的分辨率对照。

## 4. 检测器 v0

| 项 | 值 |
|---|---|
| 结构 | 双流小 CNN，**1,752,967 参数**：play 流 3×Conv(16/32/64, s2)+BN+ReLU+GAP；hud 流 2×Conv(16/32, s2)+BN+ReLU+Flatten→Linear(17920→96)；拼接后 5 个头 |
| 头 | `death`(2) / `bomb`(2) / `lives`(7 类 0–6) / `bombs`(9 类 0–8) / `score_delta`(3 类 −/0/+) |
| 输入 | play = 4 帧堆叠；hud = 当前帧单张 |
| 训练样本 | 36,000 采样帧（train split 内全部 213 个 death 正例 + 近死亡难负例 40% + 均匀帧），batch 256，AdamW lr 2e-3 |
| 训练成本 | RTX 4070 SUPER（WSL CUDA）：`play96` 3 epoch **7.04 s**、4 epoch **9.28 s**（2.32 s/epoch，15.5k 样本/s，峰值显存 387 MB）；`play64` 3 epoch **4.51 s**、4 epoch **5.94 s**（1.49 s/epoch，24.2k 样本/s，峰值 270 MB） |
| 推理成本 | test 69,600 帧：`play96` 2.29 s、`play64` 1.30 s（含 eval 全部 5 个头） |

数据加载（36k 样本 × 4 帧堆叠）与评测（"整段 shard 顺序推理"）是主要墙钟，**单次训练 5–10 s**，所以成本对比几乎全在数据带宽上：`play64` 的输入字节是 `play96` 的 38%。

## 5. G2 指标表（test split = 关卡 3，69,600 帧 / 32 shard / 96 个 death 事件）

变体 A：`window=1`、3 epoch、death 正类权重 60。变体 B：`window=4`、4 epoch、正类权重上限 8（校准）。两版都在同一份数据、同一 split 上，阈值均来自 val。

| 指标 | A play96 | A play64 | B play96 | B play64 | 要求 | 判定 |
|---|---|---|---|---|---|---|
| death 帧级召回 | 0.00% (0/96) | 1.04% (1/96) | 0.52% (2/384) | **14.06%** (54/384) | ≥99% 下界 | **未达标** |
| death 帧级精确率 | 0.00% (0/1) | 33.33% (1/3) | 25.00% (2/8) | **75.00%** (54/72) | ≥99% 下界 | **未达标** |
| death 事件级召回 | 1.04% (1/96) | 2.08% (2/96) | 3.13% (3/96) | **33.33%** (32/96) | ≥99% 下界 | **未达标** |
| death 事件级精确率 | 100% (1/1) | 100% (2/2) | 100% (3/3) | **96.97%** (32/33) | ≥99% 下界 | **未达标**（n 太小，下界 84.7%） |
| death 延迟（最大帧） | 3 (n=1) | 4 (n=2) | 5 (n=3) | **5** (n=32；中位 2、p95 4) | ≤10 | **达标** |
| lives 读数准确率 | 100% (69,600) | 100% (69,600) | 100% (69,600) | **100%** (69,600) | ≥99.5% 下界 | **达标**（Wilson 下界 99.9944%） |
| bombs 读数准确率 | 100% (69,600) | 100% (69,600) | 100% (69,600) | **100%** (69,600) | ≥99.5% 下界 | **达标**（Wilson 下界 99.9944%） |
| score 增量符号准确率 | 90.23% (69,568) | 90.56% (69,568) | 88.42% (69,568) | **91.90%** (69,568) | ≥99% 下界 | **未达标**（下界 91.70%） |

- 95% 单侧下界：lives/bombs 用 Wilson（69,600 全对 → 99.9944%）；death/score 的实测下界一律远低于 99%，见 `metrics-*.json` 的 `lower_bound_95`。
- **score 符号的负向类样本数 = 0**（`score_down` 整份数据 0 帧），因此 91.9% 这个数只覆盖"上升/不变"两类，不能声称能识别扣分。
- 口径说明：帧级与事件级**都是检测器成绩**。事件级用完整 shard 序列（不采样），匹配窗口 ±10 帧、一对一贪心匹配，未匹配的预测事件计为假正例。

### 5.1 按策略分解（test split，事件级）

| 变体 / 分辨率 | 策略 | 帧数 | 真事件 | 命中 | 事件召回 | 预测事件 | 命中 | 事件精确率 |
|---|---|---|---|---|---|---|---|---|
| B play96 | `dive` | 38,400 | 48 | 3 | 6.25% | 3 | 3 | 100% |
| B play96 | `still` | 31,200 | 48 | **0** | **0.00%** | 0 | 0 | — |
| B play64 | `dive` | 38,400 | 48 | 32 | **66.67%** | 33 | 32 | **96.97%** |
| B play64 | `still` | 31,200 | 48 | **0** | **0.00%** | 0 | 0 | — |
| A play96 | `dive` | 38,400 | 48 | 1 | 2.08% | 1 | 1 | 100% |
| A play96 | `still` | 31,200 | 48 | 0 | 0.00% | 0 | 0 | — |
| A play64 | `dive` | 38,400 | 48 | 2 | 4.17% | 2 | 2 | 100% |
| A play64 | `still` | 31,200 | 48 | 0 | 0.00% | 0 | 0 | — |

val（关卡 5，全部 `random`）：A play96 事件召回 17.8%（8/45）、A play64 22.2%（10/45）、B play96 15.6%（7/45）、B play64 37.8%（17/45）。

**这是本卡最重要的负面结论**：检测器在 `dive` 分布上学到了东西（B play64 事件召回 66.7%、精确率 97%），但对 `still` 的 48 个死亡事件**一个都没报**（四个变体全是 0/48）。也就是说 v0 学到的是"顶部弹幕密集"这个 dive 特有的线索，而不是跨策略通用的"死亡前兆"。把 dive 造出来的正例当成通用能力会直接高估检测器——这正是 §7 要写明的风险。

## 6. P2-WP4：96×112 vs 64×64

| 维度 | play96 | play64 | 结论 |
|---|---|---|---|
| 参数 | 1,752,967 | 1,752,967 | 完全相同（只换输入分辨率） |
| 输入字节/样本 | 43,008 | 16,384 | 64×64 只需 38% |
| 训练（4 epoch） | 9.28 s | 5.94 s | 快 1.56× |
| 采样吞吐 | 15.5k/s | 24.2k/s | 快 1.56× |
| 峰值显存 | 387 MB | 270 MB | 少 30% |
| death 事件级召回 | 3.13% | **33.33%** | 64×64 明显更好 |
| death 事件级精确率 | 100%（n=3） | 96.97%（n=33） | 64×64 的 n 大 11 倍才可比 |
| 延迟（最大） | 5 | 5 | 都 ≤10 |

**结论（供 WP5 冻结观测规格）**：在 v0 这个尺度上，**64×64 不仅更便宜，而且更强**——事件级召回从 3.1% 升到 33.3%（同一 split、同一配方、同一阈值来源），且这 33.3% 里 dive 子集是 66.7%。合理的解释是 6×7 块均值把"子弹接近玩家"的对比度聚了起来，而 4×4 块在灰度高分辨率下反而更受背景纹理干扰（本实验只有单 seed/单配方，**不足以证明 96×112 永远更差**，只说明"没有证据要求 96×112，而 64×64 已足够且更省"）。WP5 若要冻结，建议先按 64×64 冻结，把 96×112 作为可选项而不是默认。注意：两者的 `still` 召回都是 0/48，所以"64×64 更好"只在 dive 域内成立，不构成通用优势。

## 7. 限制与失败归因（不掩饰）

1. **death 检测器基本不可用**：事件级召回 3%–33%，帧级召回 ≤14%。原因可定位，按证据排序：
   - **策略域偏移（最严重）**：test 里 `still` 的 48 个死亡事件在四个变体下**全部漏检（0/48）**，而 `dive` 的 48 个事件在 B play64 下命中 32 个。模型学的是 dive 特有的"顶部弹幕密集"外观，不是跨策略的死亡前兆。用 dive 造正例解决了"有没有正例"，但**没有解决"正例是否代表通用死亡"**。
   - 阈值跨关卡不迁移。val（关卡 5，`random`）上变体 B `play96` 的事件召回 37.8%、`play64` 20.6%，同一阈值到 test（关卡 3）只剩 3.1% / 33.3%（其中 dive 66.7%、still 0%）。v0 的概率标定不可信，**阈值迁移本身是一个待解问题**。
   - 正类权重把概率推饱和：变体 A 用 60× 权重，val 上 38,400 帧里报出 2,401 次"death"，精确率 0.87%——典型过报。
   - 1/4 灰度网格对"死亡前一刻"的信息量本就有限：玩家自机在网格里约 2×2 像素，子弹约 1 像素。
2. **`window=4` 是预报窗口，不是延迟作弊**：延迟仍按"检测帧 − 真值帧"算（负值=提前），变体 B `play64` 的 32 次命中里 1 次是提前检测，其余中位 2 帧、最大 5 帧，都 ≤10。
3. **数据分布不均衡**：test 只有关卡 3（关卡 6 未采到），且由 `dive`/`still` 组成；`random` 的 death 正例全在 train/val。G2 因此是"跨关卡 + 跨策略"的测量，不是同分布内测。
4. **score 符号**：负向类 0 样本，无解；除非采集能触发扣分的行为（当前 `random`/`still`/`dive` 都不扣分）。
5. **单 seed、单配方**：两版变体各一次训练，没有方差估计，不能当作稳定结论；分辨率差异（33% vs 3%）幅度很大，值得复跑确认。
6. **G2 的 death 门在当前配方下不可能达标**（需要 ≥99% 下界），本卡按要求只报"未达标"和差距，不调参刷分。

## 8. 复现命令

```bash
# WSL：主跑（16 实例，300k 帧，约 8 分钟）
cd /mnt/d/WorkShop/Touhou_RL
export PYTHONPATH=/mnt/d/WorkShop/Touhou_RL
ENGINE=/root/Touhou_RL/third_party/th08-rl/build/rl_hazard2/th08-modern
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v400_p2_bulk_collect.py run --engine "$ENGINE" \
  --data-root /root/v400-data/p2det --out-dir /root/v400-data/p2det/out-main \
  --instances 16 --frames-total 300000 --frames-per-shard 2400 --lp-threads 4 \
  --warmup 400 --seed-base 7000 --max-seconds 1500 \
  --policy-plan random,still,dive,dive,dive,dive
/root/v400-venv/bin/python tools/rl/v400_p2_bulk_collect.py stats --out-dir /root/v400-data/p2det/out-main
/root/v400-venv/bin/python tools/rl/v400_p2_bulk_collect.py index --out-dir /root/v400-data/p2det/out-main

# WSL：切分 + 训练 + G2 表
/root/v400-venv/bin/python tools/rl/v400_p2_reward_detector.py split \
  --index /root/v400-data/p2det/out-main/index.json --out /root/v400-data/p2det/split.json
/root/v400-venv/bin/python tools/rl/v400_p2_reward_detector.py train \
  --index /root/v400-data/p2det/out-main/index.json --split /root/v400-data/p2det/split.json \
  --resolution play64 --epochs 4 --death-window 4 --pos-weight-cap 8 \
  --out /root/v400-data/p2det/metrics-play64-w4.json --checkpoint /root/v400-data/p2det/play64-w4.pt
/root/v400-venv/bin/python tools/rl/v400_p2_reward_detector.py report \
  --metrics /root/v400-data/p2det/metrics-play64-w4.json --out /root/v400-data/p2det/g2.md

# Windows：测试（21 个用例）
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_det/pytest -q tests/test_v400_reward_detector.py
```

产物：`/root/v400-data/p2det/out-main/{plan.json,index.json,stats.json,shards/*}`、`split.json`、`metrics-play96.json`、`metrics-play64.json`、`metrics-play96-w4.json`、`metrics-play64-w4.json`、`g2.md`、`g2-w4.md`、`play96.pt`/`play64.pt`/`*-w4.pt`。

## 9. 产物与改动

- 新增 `tools/rl/v400_p2_bulk_collect.py`（并行采集器：`prepare/worker/run/stats/index`，紧凑平面 + 断点续跑 `attempt-NN` 永不覆盖）。
- 新增 `tools/rl/v400_p2_reward_detector.py`（切分 + 训练 + 评测 + G2 表）。
- 新增 `tests/test_v400_reward_detector.py`（21 用例）。
- 未改引擎、未改协议、未改依赖、未删除任何文件。
