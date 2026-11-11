# 观测规格 obs-v3（T2 收尾冻结 / T3 D1 的输入契约）

- 日期：2026-11-10
- 状态：**已冻结**（`OBS_SPEC_VERSION = "obs-v3"`，规范摘要 `obs_spec_v3_digest()` =
  `ef2fbd15385f26f425db2bfe855d105ee009023acb08c26916e811963553a105`）
- 代码：`touhou_rl/v400/student/obs_spec_v3.py`（唯一实现；本文档是它的契约说明）
- 采集器：`tools/rl/v400_d0_dual.py --obs v3`（默认仍是 `--obs v1`，逐字节不变）
- 读取器：`touhou_rl/v400/student/obs_v3_dataset.py`
- 上游：T2a 感知探针 [EXP-P3-010](<experiments/EXP-P3-010-perception-probe-resolution.md>)、
  负责人裁定 [D-14](<31-决策记录.md>)、[obs-spec-v1](<obs-spec-v1.md>) 的 play/HUD 约定、
  [obs-spec-v2](<obs-spec-v2.md>) 的 `act_hist` 契约
- 下游：T3（D1 数据集）、T4（model-v3 + BC）、T5（DAgger）、T6（真机部署）都必须按本规格构造输入

---

## 1. 冻结（先读这一节）

1. obs-v3 = **obs-v1 的 play/HUD 约定**（同一读回、同一亮度、同一 HUD 条带、同一 4 帧堆叠、
   同一归一化、同一动作表）+ **场地平面改为 96×112** + **obs-v2 的 `act_hist`**（8×18）。
2. obs-v1 与 obs-v2 的文件**一个字节都没动**：`obs_spec.py` 的 sha256 =
   `0d47daf1bddc2b6fb8a33eb54a558f2d40f4a8c3b3d878155035d914d455578b`，
   `obs_spec_v2.py` 的 sha256 = `388b9bd86625200cda8355e4f2c3ddff8deaa993f5fab3a71946dd762a7b6370`，
   两个规范摘要仍是 `24f0ddac…364a381` / `95110743…4dba3`。四个字面量同时写在
   `obs_spec_v3.py`、`touhou_rl/v400/teacher/obs.py` 和 `tests/test_v400_obs_v3.py` 里，
   任何一处漂移都会让 import 或测试失败，而不是让一次训练悄悄跑在别的规格上。
3. 改变观测含义（分辨率、通道、帧堆叠、决策周期、归一化、HUD 是否包含、动作空间语义、
   动作历史窗口）都必须新开版本号、新文档，并让旧 checkpoint/数据集显式不兼容。
4. 摘要：`python -c "from touhou_rl.v400.student.obs_spec_v3 import obs_spec_v3_digest; print(obs_spec_v3_digest())"`
5. `shard.json` 记录 `obs_spec_version="obs-v3"`、`obs_spec_digest`、`obs_version="v3"` 与每个平面的
   shape/size/sha256；`dataset.load_teacher_shard`（v1 读取器）仍然**拒绝** v3 分片，
   `obs_v3_dataset.load_teacher_shard_v3` 拒绝 v1/v2 摘要。

### 1.1 张量布局（与任务书的写法对照）

任务书写 "play (4,96,112)"，按惯例指的是「4 帧 × 96×112 的场地」；**数组布局是
`(4, 112, 96)`**，即 `(stack, rows=height, cols=width)`，与 EXP-P3-010 的输入 B 张量
`(1, 112, 96)` 逐元素相同。原因是：384/96 = 4 列、448/112 = 4 行，只有 **4×4 整数块**
才能逐字节复现 B；`(96, 112)` 的数组布局要 384/112 = 3.43 列，不存在整数块。
`PLAY_SHAPE = (112, 96)`、`PLAY_EXTENT_WH = (96, 112)`，两个常量都在模块里，避免歧义。

---

## 2. 观测张量

| 项 | 冻结值 | 依据 |
|---|---|---|
| 场地分辨率 | **96×112**（`play96x112`，灰度，uint8），数组形状 `(112, 96)` | T2a（EXP-P3-010）：四臂全部 < 0.90，止损触发 → D-14 采用 B |
| 通道 | 1（灰度）。亮度 = `(77R + 151G + 28B) >> 8`（整数 BT.601，与 obs-v1 同一表达式） | `obs_spec.GRAY_WEIGHTS/GRAY_SHIFT` |
| 下采样 | 场地 384×448 从屏幕 `(32,16)` 起，按 **4 行 × 4 列**块做整数均值 `sum // 16`，原生 96×112；**无插值** | `PLAY_BLOCK=(4,4)`；`encode_planes_v3` 与 `probe.native.block_mean(native, (112,96))` 逐字节相同 |
| 帧堆叠 | **4** 帧，最旧在前（`t-3..t`）；episode 开头重复最早一帧补齐 | 与 obs-v1 相同 |
| HUD 条带 | **包含**，`hud` 56×160 uint8 绿色通道：score 行 48–72、lives 80–96、bombs 96–112，列 480–640，只取当前 1 帧 | 与 obs-v1 **逐字节相同**（同一编码器路径，测试锁定） |
| 动作历史 | `act_hist` **8×18** uint8，行 0 = `a[t-8]`，行 7 = `a[t-1]`；one-hot（`cand=focus*9+move`）；分片起点与**换面**处清零；**不含当前动作**、不含 bomb 位 | 完全沿用 obs-v2（`ACTION_HISTORY_*` 常量直接引用 `obs_spec_v2`） |
| 归一化 | `float32(uint8) / 255.0`，每平面独立，范围 [0,1]，不减均值不除标准差 | 与 obs-v1/v2 相同（`/255.0`，**不做** `*2-1`） |
| 输入张量 | `play (4,112,96) float32`、`hud (1,56,160) float32`、`act_hist (144,) float32` | `normalize_play` / `normalize_hud` / `normalize_action_history` |
| 排除输入 | 特权状态（`self_state`）、lives/bombs 差分（`danger_summary`） | 与 obs-v2 相同，`present=false` 写进摘要 |

**帧率/决策周期**：不变——一个学生决策 = **2 个逻辑帧**；一个 D0 CLI 决策提交
`replan_decisions × DECISION_FRAMES = 4` 帧，标签只标它的**第一个槽**。

**动作空间**：不变——`cand = focus*9 + move`（18 类）× bomb（2 类），shoot 恒开。

**每决策字节**（平面负载，不含 `labels.jsonl`）：

```text
play      4 × 112 × 96  = 43,008 B
hud       56 × 160      =  8,960 B
act_hist  8 × 18        =    144 B
合计     BYTES_PER_DECISION = 52,112 B
```

磁盘上每个平面还有固定 522 B 的 `.npy` 头（3 个平面 = 1,566 B/分片），
实测 200 决策分片 = 10,423,966 B → **52,119 B/决策**（含头）；`labels.jsonl` 另计
（实测 1,683 B/决策）。v1 的同口径平面负载是 25,344 B/决策，v3 是它的 **2.056×**。

---

## 3. 与教师的接口（学生输出 ↔ 教师 scores），以及**面向策略的辅助标签**

与 obs-v1 §3 完全一致（`scores[18]`/`verified_mask[18]`/`bomb`/`chosen_plan`/`margin`，
`π_teacher = softmax(scores/τ)`，KL 方向 `KL(π_teacher‖π_student)`，bomb 用 BCE），
另外**新增一条 D-14 要求的辅助标签**：

| 项 | 契约 |
|---|---|
| 名称 | `verified_surviving_action_set`（教师「验证存活」动作集合） |
| 来源 | 每个决策的 `TeacherLabel.verified_mask[18]`（11-P1 §3），`labels.jsonl` 已经带着它，**不需要新平面** |
| 定义 | 候选 `c` 属于该集合 ⟺ `verified_mask[c] == True`，即教师在其搜索horizon内验证过「选 `c` 能活」 |
| 消费者 | T4/EXP-P3-011 的**危险状态一致率**：在教师余量 < 8 px 的状态上，学生 argmax 落在该集合里的比例 |
| 作用 | 支撑 D-14 的**闭环判据**（占用 F1 只作代理指标记录） |
| 不是稠密标签 | 它是掩码不是概率；若要当 BCE 目标训练，必须在 T4 单独登记该损失 |

读取器把它暴露为 `sample["verified_actions"]`（升序的 cand 列表），并写进
`load_teacher_samples_v3(...)["policy_aux_label"]`。

---

## 4. 为什么是 96×112：2a 探针结果与 D-14 裁定

### 4.1 EXP-P3-010 主表（留出 seeds 15–16，thr 0.5，3 个训练种子）

| 输入 | 输入像素 | 预注册 micro-F1（3 seed 均值） | 事后最优阈值 F1 | 定位器中位误差 | 结论 |
|---|---|---|---|---|---|
| A 64×64 | 4,096 | 0.5633 | 0.6290 | — | 不达标 |
| **B 96×112** | 10,752 | **0.8502** | **0.8661** | — | **采用（D-14）** |
| C 192×224 | 43,008 | 0.8392 | 0.8540 | — | 不达标 |
| D 64×64 + 128×128 局部窗 | 16,384 | 0.7397 | 0.7458 | **11.59 px > 4 px 门限** | **淘汰** |

- 「预注册 micro-F1」= §0.1 口径（阈值 0.5、seeds 15–16、每臂 3 个训练种子的均值），
  是选择规则用的指标；「事后最优阈值 F1」= `.tmp_v400_p3diag/aggregate.json` 的
  max 聚合 × 阈值扫描（负责人引用的 0.629 / 0.866 / 0.854 就是这一列）。
  **两种读法下四臂都 < 0.90**，因此预注册的止损条件（任务书 §4 T2）触发。
- D 被淘汰的直接原因是**部署合法性**：局部窗中心只能来自像素定位器，而定位器在留出集上
  中位误差 11.59 px，远超 4 px 门限；用真实中心的 0.7397 只是上界，不能当作可部署成绩。

### 4.2 负责人裁定（D-14，2026-11-10）

- **采用 B（96×112）继续**；
- **判据从「占用 F1 ≥ 0.90」改为「闭环成绩」**：占用 F1 只作代理指标记录，
  验收看 G3 的 `S1N` 过关率（stage 1 normal、确认种子）；
- 在 T3/T4 增加**面向策略的辅助标签**（教师「验证存活」动作集合，见 §3）来支撑闭环判据。

### 4.3 已知局限（必须写进门报告，不能当成绩）

> **B 的占用 F1 是 0.8502（预注册口径）/ 0.8661（事后最优阈值），< 0.90。
> 这是 obs-v3 的已知局限，不是「已通过」的门。**

这条句子同时冻结在规范摘要里（`resolution.ruling.known_limitation`），
所以任何声称「obs-v3 达到 0.90」的说法都与摘要字节矛盾。

---

## 5. 采集器：`--obs v3`

```powershell
# Windows 侧只看参数（不跑引擎）
$env:PYTHONPATH=(Get-Location).Path
python tools/rl/v400_d0_dual.py --help
```

```bash
# WSL 侧真实采集（单实例；这里用独立 data-root，避免抢别人正在用的引擎）
env -u DISPLAY xvfb-run -a /root/v400-venv/bin/python tools/rl/v400_d0_dual.py \
  --phase all --decisions 200 --stage 0 --difficulty 1 --seed 1 --mode fast --obs v3 \
  --work-dir .tmp_v400_obsv3/work --out-dir .tmp_v400_obsv3/d0 \
  --data-root /root/v400-data/obsv3 --data-template /root/v400-data/game \
  --engine-limit 4 --engine-pattern 'th08-modern --data-dir /root/v400-data/obsv3' \
  --shard-key dual-s0d1seed1-v3
```

- `--obs` 取值 `v1`（默认）/`v3`；**默认行为不变**：v1 仍然只写 `obs/play64.npy` + `obs/hud.npy`。
- `--obs v3` 时：env 通道用 `obs_spec_v3.encode_planes_v3` 写 96×112 play 平面，
  assemble 阶段从**同一批标签行**重建 `act_hist`（`chosen_plan.first_action`，分片起点清零）
  并写 `obs/play96x112.npy` + `obs/hud.npy` + `obs/act_hist.npy`；`shard.json` 记录
  `obs_spec_version="obs-v3"` 与 digest。
- `--obs` 与标签/哈希无关：HASH 对齐闸门一个字都没改（`--trim-transition`、
  `--allow-early-stop` 的语义也不变），所以 v1/v3 的轨迹证据可以互相对照。

### 5.1 200 决策冒烟（2026-11-10 真跑，stage 0 / normal / seed 1 / 单实例）

```text
channel A (shadow labels) : 200/200 decisions, wall 17.96 s
channel B (env pixels)    : 800 frames, 400/400 HASH checkpoints equal, wall 18.97 s
assemble                  : decisions 200, obs_bytes_per_decision 52,119
load gate                 : samples 200/200, play (4,112,96), hud (56,160),
                            obs_spec_digest ef2fbd15…553a105, alignment
                            decision_time_before_commit
total wall                : 40 s (1 引擎实例；生产批跑同时占 12 个，未受影响)
```

**HASH 对齐 100%**（400/400 检查点相同，`trimmed_transition=null`）。
读取器回读：`play (200,4,112,96) uint8`、`act_hist (200,8,18) uint8`、`hud (200,56,160)`；
`act_hist` 起点 8 行为全零 padding，第 8 行首个满窗；`require_observations=True` 闸门
200/200 通过、0 未解析。

### 5.2 200 万帧的磁盘估算（D-05 上限 300 GB）

| 项 | 每帧/每决策 | 200 万帧（= 50 万决策） |
|---|---|---|
| 平面负载（play+hud+act_hist） | 52,112 B/决策 | **26.06 GB** |
| 平面实测（含 3 个 `.npy` 头） | 52,119 B/决策 | **26.06 GB** |
| `labels.jsonl`（scores/mask/plan/HASH…） | 1,683 B/决策 | **0.84 GB** |
| 合计 | 53,802 B/决策 | **26.90 GB** |

占 D-05（磁盘新增 ≤ 300 GB）的 **9.0%**；是 v1 同规模平面（12.67 GB）的 2.06×。

---

## 6. Fail-closed 规则（已实现并有测试）

| 情况 | 行为 |
|---|---|
| shard.json 的 `obs_spec_version`/`obs_spec_digest` 不是冻结 obs-v3 | `ObsSpecV3Mismatch` → `DatasetError`（v1 读取器同样拒绝 v3 分片） |
| 平面文件名/shape/dtype/size/sha256 与 `shard.json` 不符 | `DatasetError`，不加载 |
| `act_hist` 与按标签行重建的结果不逐字节相同 | `DatasetError`（「存的历史不是这条轨迹的历史」） |
| 标签行 `decision`/`frame`/`render_frames` 算术不符、padding 出现在非 0 决策 | `DatasetError` |
| 标签行既无 `dagger.executed_cand` 也无 `chosen_plan.first_action` | `DualError`/`ObsSpecV2Error`（不猜） |
| `require_observations=True` 且某个标签行解析不到可读分片 | `DatasetInsufficient`（消息里带真实计数） |
| 采集器 `--obs` 取值非法 | argparse `SystemExit`（choices 只有 v1/v3） |

---

## 7. 复现/校验命令

```powershell
# 契约测试（28 用例：v1/v2 逐字节、v3 编码逐字节等于探针 B、act_hist 因果与清零、
# 采集器默认 v1、--obs v3 写平面、v3 读取器往返、digest 冻结、磁盘估算）
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_obsv3/pytest -q tests/test_v400_obs_v3.py
```

```powershell
# 回读真实冒烟分片并打印样本数/形状/digest/字节估算
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python .tmp_v400_obsv3/readback_v3.py
```

---

## 8. 修订记录

| 日期 | 版本 | 修改 | 原因 |
|---|---|---|---|
| 2026-11-10 | obs-v3 | 初版冻结：obs-v1 的 play/HUD 约定 + 场地 96×112（4×4 整数块，= EXP-P3-010 输入 B 逐字节）+ obs-v2 的 `act_hist`（8×18） | T2a 四臂全部 < 0.90 触发止损，负责人 D-14 裁定采用 B、判据改闭环，并新增「验证存活动作集合」辅助标签 |
