# EXP-P3-007 教师 v2j：动作持续时间集 `{1,2}` → `{1,2,3,4}`

- 日期 / 切片：2026-11-10 / **P3-WP5-b**（EXP-P3-005 §8 的 P0：检验"18 个候选几乎是同一个动作"）
- 依赖：[EXP-P3-005](<EXP-P3-005-margin-resolution.md>) §7–§8（P0 = 动作持续时间集；
  "18 个候选的位移只差 2–8 px、margin 在 16 px 饱和、`align` 仅 0.021 点/px"）、
  [EXP-P1-021](<EXP-P1-021-laser-damageable.md>)（`v2h` = 本档的父档）、
  [EXP-P3-004](<EXP-P3-004-label-signal-2x2.md>) §6（窗口尺 `consecutive`）

## 0. 结论先说

1. **`v2j` 按任务卡注册完成**：与 `v2h` 逐字段相同（`weights` 是**同一个实例**），只把 `durations`
   从 `(1, 2)` 改成 `(1, 2, 3, 4)`；`FAST`/`FAST3`/`V2A`/`V2H`/`V2I`/`V3A`/`V3B` 全部未动（单测逐档钉住）。
   `durations` 本来就是 `TeacherConfig` 的字段，`search.py` **一行未改**，根网格 36 → 72。
2. **采集 8 片 / 26,346 决策 / 54.5 min / HASH 52,692-52,692 = 100 %**（2 worker，引擎 `rl_laser`
   `1462829b…`），8/8 片加载门全绿。单决策 **186–236 ms**（低负载 186–193 ms，`loadavg` 10–16 时
   235–236 ms；全批平均 **199.4 ms**）；对照 `v2i` 的 114–116 ms 是**低负载 ×1.63、全批 ×1.73**。
   其中搜索相 **21.0 → 39.5 ms（×1.88）**；单 worker 5.19–5.37 决策/s，2 worker 合计 8.06 决策/s。
3. **标签分辨率出现本轮唯一的质变**：复原后 `top1−top2 > 0.1` 的决策占比
   **2.2159 % → 9.5612 %（×4.31）**，p90 0.0795 → 0.0875、p99 0.1667 → 0.1953；argmax 最大类
   0.4222 → 0.4023、归一化熵 0.7244 → 0.7449；**23.7 % 的决策选了 3/4 槽**（`v2h` 结构上不可能）。
   但 `chosen_plan.margin` 仍 97.31 % 顶在 16 px，240 条真实 tape 上"`margin < 16` 的计划个数"
   **逐位相同（0.0875 → 0.0875）** ⇒ EXP-P3-005 §8 的两个机制只兑现了一个（端点散布 +19.2 %、
   原始 `top1−top2` ×1.99），**"把计划压到 16 px 以下"没有发生**。
4. **离线**：E（8 片）val top-1 **0.4527**、macroR **0.0999**、KL 0.5389（`v2i` E：0.3288 / 0.0879 /
   0.8019）；F（27 片）0.5584 / **0.1695** / 0.4181（`v2i` F：0.5493 / 0.1471 / 0.4129）。
   `geo_restore=on` 把 top-1 打到 0.2141 / 0.3853（目标变平），KL 反而降到 0.026 / 0.024。
5. **闭环（新尺 `consecutive`）**：E **1,044.6**、F **1,462.2**、Eg 734.0、Fg 714.0；`v2i` 同口径是
   1,209.6 / 1,816.0，v2 基线 1,718.8，教师 30 % 门 4,070。**配对提升是负的**（E −165.0，t=−1.03；
   F −353.8，t=−1.07；Eg −475.6，Fg −1,102.0），两条预注册腿（≥ +300 帧、macroR ≥ 0.20）**都不过**；
   只有 F 的帧数 ≥ 教师 30 %（35.9 %）。**判据 FAIL。**
6. **回答"9.56 % 的 `>0.1` 有没有转化成帧数"：没有，而且方向是负的。** 分辨率 ×4.31、离线 top-1
   +0.124（E）都没能变成闭环帧数；`v2j` 是**第二个**被证伪的"标签分辨率 → 闭环"通道（第一个是
   EXP-P3-005 的 `margin_cap`/连续项）。**`durations` 是一个真实的、可用的标签侧旋钮（与 `margin_cap`
   不同：它确实改写了 73 % 的分数向量和 66.7 % 的承诺时长），但它不是闭环瓶颈。**

## 1. `v2j` 档：改了什么、为什么只改这一处

`touhou_rl/v400/teacher/config.py` 新增 `V2J`（注册进 `CONFIGS` 与 `PROFILES`）：

```python
_V2J_DURATIONS = (1, 2, 3, 4)
V2J = TeacherConfig(mode="v2j", horizon=48, beam=64, verify_k=3, max_iterations=3,
                    weights=_V2H_WEIGHTS,          # 同一个 Weights 实例，不是复制
                    bomb_rule1_max_death_frame=12, durations=_V2J_DURATIONS)
```

| 字段 | `v2h` | `v2j` |
|---|---|---|
| `durations` | (1, 2) | **(1, 2, 3, 4)** |
| `weights` | `_V2H_WEIGHTS` | **同一个实例**（`V2J.weights is V2H.weights`） |
| `horizon` / `beam` / `verify_k` / `max_iterations` | 48 / 64 / 3 / 3 | 同 |
| `replan_decisions` / `decision_frames` | 2 / 2 | 同 |
| `bomb_rule1_max_death_frame` | 12 | 同 |
| `bomb_rule1_min_bombs_left` / `max_margin` / `clip_y` | 0 / None / None | 同 |
| `shoot_pulse_period` / `record_items` | 60 / False | 同 |

`dataclasses.replace(V2J, mode="v2h", durations=(1, 2)) == V2H` 逐字段成立
（`tests/test_v400_teacher.py::test_v2j_is_v2h_with_the_duration_set_widened_and_nothing_else`）。
`FAST`/`FAST3`/`V2A`/`V2H`/`V2I`/`V3A`/`V3B` 的 `durations` 全部仍是 `(1, 2)`
（`test_v2h_v2i_and_the_earlier_profiles_keep_their_duration_set` 逐档钉住；`V2H` 另有一条
**逐字段重新声明**的相等断言）。

### 1.1 为什么"持续时间"是唯一能分开 18 个候选的旋钮

EXP-P3-005 §7.1/§7.2 已经把 `margin_cap` 判死（钳位在 `search._frame_margin` 与 raster 的
`uint16` 低字节，`Weights.margin_cap` 只是最后一道 `min`），并测出"教师侧唯一可动的连续项"
把真实偏好率从 1.20 % 拉到 2.22 % 就到底了。剩下没动过的物理量是**首动作的承诺时长**：

* `decision_frames = 2`、`durations = (1, 2)` ⇒ 首动作只被承诺 2 或 4 个逻辑帧；
* 队伍 0 的实测速度（`EXP-P1-001 §3`）是 4.0 / 2.83 / 2.0 / 1.41 px/帧（轴/斜 × 普通/聚焦）
  ⇒ 首动作的位移只有 **2.8–16 px**，而 18 个候选共享这 18 个方向 × 2 种速度；
* 于是 18 个端点挤在一个 20 px 的圆里，`align`（4.0 点 / 192 px = 0.021 点/px）与已在 16 px
  饱和的 margin 都产生不了梯度。

把承诺时长扩到 4 槽（8 帧，最多 **32 px**，典型 16 px）**只改根网格**：
`18 × len(durations)` = 36 → 72 个根，`slot < duration` 的根跳过前 `duration` 轮扩展，
其余（束宽、权重、打分函数、验证、提交节奏 `replan_decisions`）一律不动。

**实现事实（本轮实测）**：`durations` 是纯配置，`search.py` 一行未改 —— `beam_search` 的
`limit = min(config.slots, len(tape)//2)`、`by_duration` 的根分配、`start_slot` 的跳过逻辑
本来就按 `config.durations` 工作（`v1` 起就是可配置的），所以"新增一个档"就足以打开 4 槽。

## 2. 采集（stage 0 / normal / seeds 1–8，引擎 `build/rl_laser`）

- 采集器：`tools/rl/v400_d0_dual.py --phase all`（channel A shadow 出标签、channel B env 出像素，
  HASH 逐检查点对齐），`--mode v2j`；`--mode` 的 choices 早就是教师注册表
  `tuple(sorted(CONFIGS))`，零改动（`tools/rl/v400_teacher.py` 只更新了 `--mode` 的帮助文本）。
- 引擎：`/root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern`，
  sha256 `1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f`（`--expected-sha` 同值，
  与 EXP-P3-005 同一把二进制；`v2h` 的激光判定窗需要它）。
- 并发：**2 个 worker**（2 teacher × shadow/child），`--engine-limit 6`、`--threads 1`；
  实测 `pgrep -c th08-modern` 5–7，`loadavg` 3–8（机器上同时有零售 trace 批跑、DAgger R2 与
  P2 上量三个外部任务，采集期 `loadavg` 4–12）。
- 参数：`--decisions 3300 --max-frames 20000 --stop-on-death --allow-early-stop --trim-transition`，
  `--target 26400`。

| shard | seed | decisions | frames | stop_reason | HASH 检查点 | 对齐 | channel A wall | channel B wall | wall | obs bytes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| dual-s0d1seed1 | 1 | 3,300 | 13,200 | decisions | 6,600 | 6,600/6,600 | 636.1 s | 125.7 s | 770.8 s | 83,636,244 |
| dual-s0d1seed2 | 2 | 3,274 | 13,100 | **stage_changed** | 6,548 | 6,548/6,548 | 610.2 s | 126.6 s | 746.4 s | 82,977,300 |
| dual-s0d1seed3 | 3 | 3,300 | 13,200 | decisions | 6,600 | 6,600/6,600 | 616.6 s | 163.4 s | 799.1 s | 83,636,244 |
| dual-s0d1seed4 | 4 | 3,300 | 13,200 | decisions | 6,600 | 6,600/6,600 | 610.4 s | 165.0 s | 790.3 s | 83,636,244 |
| dual-s0d1seed5 | 5 | 3,300 | 13,200 | decisions | 6,600 | 6,600/6,600 | 779.6 s | 153.5 s | 950.3 s | 83,636,244 |
| dual-s0d1seed6 | 6 | 3,300 | 13,200 | decisions | 6,600 | 6,600/6,600 | 777.1 s | 150.8 s | 941.7 s | 83,636,244 |
| dual-s0d1seed7 | 7 | 3,274 | 13,100 | **stage_changed** | 6,548 | 6,548/6,548 | 609.5 s | 120.9 s | 740.3 s | 82,977,300 |
| dual-s0d1seed8 | 8 | 3,298 | 13,196 | **stage_changed** | 6,596 | 6,596/6,596 | 615.6 s | 122.1 s | 746.9 s | 83,585,556 |
| **合计** | 1–8 | **26,346** | **105,396** | 5× decisions + 3× stage_changed | **52,692** | **52,692/52,692 = 100 %** | 5,255.1 s | 1,128.0 s | **3,267.3 s（54.5 min）** | **667,721,376** |

`progress.json`：`decisions=26346 / shards_done=8 / shards_failed=0 / hash_alignment_rate=1.0 /
hash_checkpoints=52692 / hash_checkpoints_equal=52692 / load_verified_shards=8 / elapsed_s=3267.3 /
decisions_per_s=8.0635 / complete=false / partial=true` —— **`partial` 只是"26,346 < target 26,400"的算术标记**
（seeds 2/7/8 在 3,274/3,274/3,298 决策处**自然换面**，每片都比 3,300 少 1–26 条），不是失败：
8/8 片 `complete=true`、`shards_failed=0`。

`load_gate.json`（逐片 `dataset.load_teacher_samples(require_observations=True)`）：8/8 片
`matches=true`、`all_digest_ok=true`（`obs_spec_digest 24f0ddac…`）、`unresolved_observations=0`、
**`max_chosen_margin_px = 16.0`**、`margin_clamped_at_16 = true`。

**每决策成本**：channel A（teacher）= 5,255.1 s / 26,349 ≈ **199.4 ms/决策**（逐片 186–236 ms）；
channel B（env 回放 + 像素）≈ **42.8 ms/决策**（32–50 ms，比 v2i 的 34 ms 高，因为叠加了外部任务的
`loadavg` 4–12）。见 §7。

## 3. 分辨率与散布：`v2h`（= `v2i`）/ `v2j` 三档对比

### 3.1 标签级（`analyze_labels.py`，口径与 EXP-P3-005 §3 完全一致）

`restored = scores + 5·(1−verified_mask)` 复原后取 `top1 − top2`；`chosen_plan` 的
`duration`/`margin`/`end_xy` 直接读。`v2h` 与 `v2i` **是同一列**：EXP-P3-005 §7.1 已实测
`v2h`/`v2i` 的 18 路分数逐位相同（`margin_cap` 是惰性的），本轮同一份脚本在 `v2i` 根上复算得
`>0.1 = 2.2159 %`、`p90 = 0.0795`，与 EXP-P3-005 §3 逐值相同。

| 指标 | 旧 `k=1`（D0 26,648） | `v2h` = `v2i`（26,400） | **`v2j`（26,346）** | v2j vs v2h |
|---|---:|---:|---:|---:|
| `verified_count` mean | 1.00 | 3.00 | 3.00 | — |
| 存盘 `top1−top2` 恰等于 5.0 | 82.16 % | 0.00 % | 0.00 % | — |
| 复原后 `top1−top2` 均值 | 0.014488 | 0.016801 | **0.023118** | **+37.6 %** |
| **p50** | 0.0 | 0.0 | 0.0 | 不变 |
| **p90** | 0.058925 | 0.079545 | **0.087521** | **+10.0 %** |
| p99 | 0.117851 | 0.166667 | **0.195263** | +17.2 % |
| max | 5.573148 | 4.827411 | 2.959560 | −38.7 % |
| **`>0.1` 占比** | 1.2008 % | 2.2159 % | **9.5612 %** | **+331.6 % 相对 / +7.35 pp（×4.31）** |
| `>1.0` 占比 | 0.1013 % | 0.0795 % | 0.0607 % | −23.6 % |
| argmax 类数（共 18） | 17 | 17 | **17** | 不变（动作 9 仍从未被选中） |
| argmax 最大类占比 | 0.4375 | 0.4222 | **0.4023** | −0.020（更分散） |
| argmax 归一化熵 | 0.6871 | 0.7244 | **0.7449** | +2.8 % |
| 教师目标熵（τ=2） | 2.2687 | 2.1200 | 2.1194 | ≈ 0 |
| **`chosen_plan.margin` 恰 16 px** | 96.06 % | 96.41 % | **97.31 %** | **+0.90 pp（钳位依旧）** |
| `chosen_plan.margin` < 16 px | 3.94 % | 3.59 % | 2.69 % | −0.90 pp |
| **`chosen_plan.duration` 直方图** | `[_, 21865, 4783, 0, 0]` | `[_, 20547, 5853, 0, 0]` | **`[_, 17908, 2188, 3058, 3192]`** | **23.7 % 选 3/4 槽** |
| `duration` 均值 | 1.179 | 1.222 | **1.679** | +37.4 % |
| chosen 端点 `end_x` 标准差（跨决策） | 102.03 | 99.62 | 100.24 | +0.6 % |
| chosen 端点 `end_y` 标准差（跨决策） | 29.48 | 34.53 | 26.10 | −24.4 % |
| bomb 正例 | 0 | 0 | 0 | — |

argmax 直方图（18 类，顺序 = 动作 0..17）：

| root | 直方图 |
|---|---|
| 旧 `k=1` | `[11659, 616, 2279, 2646, 548, 335, 1799, 1877, 285, 0, 73, 1707, 1934, 194, 53, 309, 224, 110]` |
| `v2h` = `v2i` | `[11146, 786, 1824, 2238, 652, 435, 1423, 2422, 556, 0, 151, 1846, 1470, 233, 141, 669, 226, 182]` |
| **`v2j`** | `[10600, 714, 1851, 2428, 600, 359, 1641, 2359, 520, 0, 160, 1528, 1575, 348, 142, 805, 484, 232]` |

**读法**：

1. **唯一的质变是 `>0.1` 占比：2.2159 % → 9.5612 %（×4.31）**。这是"教师真正能排序 18 个动作"
   的决策比例，EXP-P3-005 只把它从 1.20 % 抬到 2.22 %，本轮一次抬到 9.56 %。p90 只涨 10 %
   （0.0795 → 0.0875）说明增量全在**尾部**：不是"整体更平的地方变陡"，而是**多出一批真正分开的决策**。
2. **首动作分布更分散，但只分散一点**：最大类占比 0.4222 → 0.4023、熵 0.7244 → 0.7449，
   动作 0 从 43.75 % 降到 40.24 %、动作 17 从 0.69 % 升到 0.88 %；17 类、动作 9 依旧为 0。
   和 `k=1`（0.4375 / 0.6871）比，这是三步里第二步，仍不是"18 类均匀"。
3. **margin 依旧被钳位**：`chosen_plan.margin` 恰 16 px 的比例反而升到 97.31 %（v2i 96.41 %）——
   `v2j` 没有、也不可能靠 durations 绕过 `search._frame_margin` 的 16 px 钳位。
4. **行为真的变了**：23.7 % 的决策选择 3/4 槽（v2h 结构上不可能，上限 2 槽），`duration` 均值
   1.222 → 1.679。这是标签层面的"行为分布"变化，不是分数噪声。

### 3.1b 同状态配对（`hash_pre` 相同的行）

`v2i` 与 `v2j` 从同一个 `RESET` 出发，逐决策比较 `hash_pre`（决策点状态哈希）。**8 个 seed 全部在
决策 15 首次分叉**（前 15 个决策的提交动作相同，之后不同），所以可配对的同状态行是
**120 行**（15 × 8）：

| 量（120 行同状态） | 值 |
|---|---:|
| 18 路复原分数**逐位相同** | **26.7 %**（即 73.3 % 的分数向量被 durations 改写） |
| argmax 相同 | 93.3 %（6.7 % 的决策换了首动作） |
| chosen 首动作相同 | 93.3 % |
| **chosen `duration` 相同** | **33.3 %**（66.7 % 的决策换了承诺时长） |
| 验证集合相同 | 86.7 % |
| chosen 端点相同 ⟺ 位移 > 0 | **66.7 % 移动** |
| chosen 端点位移均值 / 中位 / p90 | **10.65 / 16.00 / 16.00 px** |
| `top1−top2` 差（v2j − v2i）均值 / 中位 / p90 | −0.0523 / −0.0976 / 0.0 |

**这张表要说的是**：在同一批状态下，`durations` 把 73 % 的分数向量改写、把 2/3 的决策的承诺
时长改写、把 2/3 的端点推移**恰好 16 px**（4 槽位移），但只有 6.7 % 的决策换了首动作、
`top1−top2` 的差值是**负的**（−0.052）。最后一条很重要：在**开局那 15 个决策**里，`v2h` 的分差
比 `v2j` 更大 —— §3.1 的全量 `>0.1` ×4.31 不是靠开局这一段拉起来的，而是靠中后段。样本只有
120 行，只能作为"机制确实生效"的旁证，不能当作效应量。

### 3.2 真实 tape 上的 **18 动作位移散布**（本轮新增测量）

标签文件只存"被选中的那一个计划"，所以 §3.1 的 `end_xy` 只能回答"被选中的端点云有多宽"。
"18 个候选彼此离多近"必须回到真实 tape 上量：`tape_dispersion.py` 走**出厂 child 录制路径**
（与采集器同一条路），对每一条真实 tape 在**同一条 tape** 上跑 `v2h` 与 `v2j` 两次搜索。

采样：seed 1 的 stage 0 轨迹上取 4 个 60 决策窗口（决策 0 / 600 / 1,200 / 2,000，即帧 0 / 2,400 /
4,800 / 8,000），共 **240 条真实 tape**。窗口靠"回放采集器自己提交的 mask"到达，并用采集标签里的
`hash_post` **逐窗口校验哈希一致**（`replay_hash.match=true`，回放与采集状态逐位相同，
所以这 240 条 tape 就是教师真实面对的状态分布，不是另造的）。

| 量（240 条 tape 均值） | `v2h`（durations 1,2） | `v2j`（1,2,3,4） | 变化 |
|---|---:|---:|---|
| 18 端点云 RMS 半径（px） | **6.300** | **7.510** | **+19.2 %** |
| `end_x` 标准差（px） | 2.542 | **3.453** | +35.9 % |
| `end_y` 标准差（px） | 5.279 | 5.990 | +13.5 % |
| 18 端点最大两两距离（px） | 20.946 | 24.657 | +17.7 % |
| `\|end − start\|` 均值（px） | 8.148 | 8.941 | +9.7 % |
| `\|end − start\|` 标准差（px） | 3.662 | 4.752 | +29.8 % |
| `\|end − start\|` 最大（px） | 15.713 | 18.839 | +19.9 % |
| 原始 `top1 − top2`（无 −5 折扣） | **0.01687** | **0.03357** | **+99.0 %（恰好 ×1.99）** |
| 18 个计划里 `margin < 16` 的个数 | **0.0875** | **0.0875** | **0（完全没变）** |
| 18 个计划的最小 margin（px） | 15.877 | 15.877 | 0 |
| 端点云变宽的窗口数 | — | **82 / 240（34 %）** | — |
| argmax 变化的决策数 | — | **14 / 240（5.8 %）** | — |
| 搜索耗时（早期 / 中期窗口） | 21.0 / 51.6 ms | 39.5 / 79.0 ms | **×1.88 / ×1.53** |

**逐窗口（RMS 半径 / top1−top2）**：

| 窗口（回放决策数） | `v2h` RMS | `v2j` RMS | `v2h` gap | `v2j` gap | 云变宽 |
|---|---:|---:|---:|---:|---:|
| 0（开局，帧 0） | 6.11 | **7.14** | 0.0155 | **0.0282** | 26/60 |
| 600（帧 2,400） | 7.32 | **10.82** | 0.0517 | **0.1058** | **42/60** |
| 1,200（帧 4,800） | 6.07 | 6.07 | 0.0000 | 0.0000 | 0/60 |
| 2,000（帧 8,000） | 5.70 | 6.02 | 0.0002 | 0.0003 | 14/60 |

**读法（这一节是本轮最重要的机制证据）**：

1. **"端点散布"这一半成立，但量级是 +19 %，不是卡片预期的 ×2–4**：18 端点云的 RMS 半径
   6.30 → 7.51 px，`end_x` 标准差 +36 %，最大两两距离 +18 %。原因在逐候选的 argmax：4 槽根
   只在**少数状态**上赢（240 个决策里 82 个云变宽、14 个 argmax 改变），多数决策仍然选 1–2 槽。
   位移之所以只 +10 %，是因为教师**没有**把首动作总是推到 4 槽 —— 它只在 align 真的更好时才推。
2. **"把计划压到 16 px 以下"这一半不成立**：240 条 tape、每条 18 个计划（共 4,320 个计划）里，
   `margin < 16` 的计划个数在 `v2h` 与 `v2j` 下**完全相同**（均值 0.0875、最小 margin 15.877）。
   机制原因是可解释的：分数里 margin 与 align 的边际价值是 1 点/px 对 0.021 点/px，所以一个
   净空更小、align 更好的长持计划几乎永远不会赢过短持计划。**EXP-P3-005 §8 的两个机制里，
   只有"align/端点散布"这一条被观测到**。
3. **分差翻倍但绝对水平仍极小**：`top1 − top2` 从 0.0169 到 0.0336（×1.99），这是"18 个候选
   被分开了"的**相对**证据；但它仍然只有 0.03 px 量级 —— 一个 margin 点的 3 %。
   在这个量级上，四舍五入到 6 位小数之后仍然"平"（§3.1 的 `>0.1` 占比才是它变成标签信号的证据）。
4. 窗口 1,200（帧 4,800）是**完全惰性**的一段：18 个候选分数逐位相同（gap 恰 0.0），`v2j` 与
   `v2h` 的端点云逐位相同。这说明这个旋钮的作用**依赖状态**，不是全局提升。

### 3.3 首动作分布：更分散，但不是"从塌缩到均匀"

"首动作"= 标签里 `chosen_plan.first_action`（教师真正提交的那个动作），与 §3.1 的 argmax 直方图
（对 `scores` 取 argmax，含验证选优）**略有不同**，因为验证会在同分候选间换一个：

| 量 | `v2h` = `v2i` | **`v2j`** | 变化 |
|---|---:|---:|---|
| 直方图（动作 0..17） | `[11144, 786, 1824, 2238, 651, 435, 1422, 2423, 555, 0, 151, 1846, 1471, 233, 141, 670, 227, 183]` | `[10594, 714, 1851, 2428, 600, 360, 1640, 2359, 520, 0, 160, 1527, 1576, 349, 141, 810, 485, 232]` | — |
| 最大类占比 | 0.4221（动作 0） | **0.4021** | −0.020 |
| 非零类数 | 17 | 17 | 动作 9（聚焦不动）**仍为 0** |
| 归一化熵 | 0.7245 | **0.7451** | +2.8 % |
| 动作 0 占比 | 42.21 % | 40.21 % | −2.0 pp |
| 动作 10 / 15 / 16 / 17 占比 | 0.57 % / 2.54 % / 0.86 % / 0.69 % | **0.61 % / 3.07 % / 1.84 % / 0.88 %** | 小动作类别普遍上升（+0.04…+0.98 pp） |

**读法**：首动作分布**确实更分散**（最大类 −2 pp、熵 +2.8 %），而且增加量集中在**低频动作**
（15/16/17 号：东南/南/西南方向的非聚焦），这与"更长的承诺时长让斜向/后退动作有机会赢"一致。
但它离"均匀"还很远（40 % 的决策仍是动作 0），**动作 9 依旧一次都没被选中** ——
和 EXP-P3-004/005 的结构性塌缩是同一件事，只是松了一格。

## 4. 训练（两种口径 × `geo_restore` off/on，同一配方）

- 配方与 armA/armB/E/F 完全相同：width 48 / hidden 256（**204,260 参数**）、τ=2.0、
  `KL(π_t‖π_s)+BCE`、AdamW lr 2e-3、batch 256、seed 0、`--train-seconds 420`（≤10 min）。
- **口径 E / Eg**：`--d0-root .tmp_v400_v2j/v2j`（新 8 片），`--accept-partial-d0`（26.4k < 50k 契约，
  与 EXP-P3-005 的 E 同设计）。
- **口径 F / Fg**：`--d0-root .tmp_v400_v2j/v2j --extra-root .tmp_v400_margin/d0_916
  --extra-root .tmp_v400_dagger/d0_extra` = **27 片**（新 8 + D0 seeds 9–16 8 片 + DAgger R1 11 片，
  26,864 + 23,400 决策），与 EXP-P3-005 的 F 逐分片相同。
- `geo_restore`：E/F 用**关闭**（与 EXP-P3-005 §4 的 E/F 配对）；Eg/Fg 用 `--geo-restore`
  （EXP-P3-004 里唯一 5/5 同向、t=24.9 的因子）。

| 臂 | root（分片） | 决策 | split（train/val/test） | τ | 参数 | step / 样本 | wall | 跑满预算 |
|---|---|---|---:|---|---:|---:|---:|---|
| **E** | v2j 8 片 | 26,346 | 19,774（seeds 1–6）/ 3,298（seed 8）/ 3,274（seed 7） | 2.0 | 204,260 | 15,037 / 3,849,472 | **461.7 s** | 是（`aborted_at_deadline=true`） |
| **F** | v2j 8 + D0 9–16 + DAgger 11 = **27 片** | 76,610 | 52,052（19 片）/ 12,294（4 片）/ 11,982（4 片） | 2.0 | 204,260 | 14,753 / 3,776,768 | **537.7 s** | 是 |
| **Eg** | E + `--geo-restore` | 26,346 | 同 E | 2.0 | 204,260 | 15,871 / 4,062,976 | 475.5 s | 是 |
| **Fg** | F + `--geo-restore` | 76,610 | 同 F | 2.0 | 204,260 | 15,657 / 4,008,192 | 599.5 s | 是 |

E/Eg 带 `--accept-partial-d0`（26,346 < 50,000 契约，`gate_waived=true`，与 EXP-P3-005 的 E 同设计）；
F/Fg 不豁免。E 与 F 的 `split_signature` 与 EXP-P3-005 的 E/F 同结构（val=seed 8、test=seed 7，
DAgger 与 D0-9..16 的分片也按同一规则落位），所以 §4.1 与 §5.2 的对比是**配对的**。

### 4.1 离线指标（整 split 全量）

| 臂 | val top-1 | val top-3 | **val macroR** | val KL | val 教师熵 | test top-1 | **test macroR** | test KL | val 预测类数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **E**（8 片） | **0.4527** | 0.5940 | 0.0999 | 0.5389 | 2.1178 | 0.3919 | 0.0969 | 0.6247 | 13 |
| **F**（27 片） | **0.5584** | 0.6999 | **0.1695** | 0.4181 | 2.2642 | 0.5031 | **0.1543** | 0.4699 | 15 |
| **Eg**（8 片 + geo） | 0.2141 | 0.4036 | 0.0829 | **0.0256** | 2.8682 | 0.1732 | 0.0891 | 0.0240 | 17 |
| **Fg**（27 片 + geo） | 0.3853 | 0.4630 | 0.0749 | **0.0238** | 2.8666 | 0.3454 | 0.0688 | 0.0227 | 10 |
| 参照 `v2i` E（EXP-P3-005） | 0.3288 | 0.4721 | 0.0879 | 0.8019 | 2.1098 | 0.4724 | 0.0971 | 0.5910 | 12 |
| 参照 `v2i` F（EXP-P3-005） | 0.5493 | 0.7103 | 0.1471 | 0.4129 | 2.2642 | 0.5370 | 0.1426 | 0.4274 | 12 |
| 参照 `checkpoint-v2`（27 片，k1/fast） | 0.5912 | — | 0.1912 | 0.3473 | — | 0.5766 | 0.1887 | 0.3542 | — |
| 参照 armA（8 片，k1/fast） | 0.5656 | — | 0.2514 | 0.4176 | 2.2608 | 0.5696 | 0.2487 | 0.3957 | 16 |

**读法**：

1. **8 片口径是明确变好**：E 的 val top-1 0.3288 → **0.4527**（+0.124，+37.7 %）、KL 0.8019 → 0.5389
   ——"可学的排序"确实变多了，和 §3.1 的 `>0.1` ×4.31 方向一致。这是本轮最干净的一条"标签侧证据"。
2. **27 片口径几乎不动**：F 的 val top-1 0.5493 → 0.5584（+0.009）、macroR 0.1471 → 0.1695、
   KL 0.4129 → 0.4181（略差）。原因结构性：27 片里只有 8 片是 v2j 标签（30 %），另外 19 片是旧的
   `fast` 标签，把新分辨率稀释掉了。**这也说明 E→F 的 +0.106 主要是数据量，不是分辨率。**
3. **`geo_restore` 在这批标签上不再是"更准"，而是"更平"**：target 熵 2.12 → 2.87、KL 0.026（几乎
   完美拟合一个极平的目标）、top-1 却掉到 0.21/0.39。EXP-P3-004 里它是唯一 5/5 同向的闭环因子
   （+463 帧），本轮它在 v2j 标签上**把策略压成常数动作**（§5.1 的 sd=0），所以这条因子与 v2j
   的组合是**负交互**，不能照搬。

## 5. S1N 5 局（新尺 `consecutive`）

- 尺：`tools/rl/v400_student_eval.py --seeds 1000,1001,1002,1003,1004 --max-frames 19887
  --stage 0 --difficulty normal --stop-on-death --obs-window consecutive`，引擎
  `build/rl/th08-modern`（sha256 `591c521d…`，与 EXP-P3-002/003/004/005 同一把尺、同一引擎），
  **1 引擎/进程、2 进程并发**（2 引擎）。

### 5.1 逐局

| seed | **E**（8 片） | **F**（27 片） | **Eg**（8 片+geo） | **Fg**（27 片+geo） | `v2i` E | `v2i` F | v2 基线 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 1,263 | 1,171 | 734 | 714 | 943 | 1,763 | 1,644 |
| 1001 | 965 | 1,543 | 734 | 714 | 1,233 | 2,907 | 1,852 |
| 1002 | 1,025 | 814 | 734 | 714 | 945 | 1,293 | 1,409 |
| 1003 | 992 | 1,397 | 734 | 714 | 1,399 | 1,324 | 1,885 |
| 1004 | 978 | 2,386 | 734 | 714 | 1,528 | 1,793 | 1,804 |
| **均值** | **1,044.6** | **1,462.2** | **734.0** | **714.0** | 1,209.6 | 1,816.0 | 1,718.8 |
| 中位数 | 992 | 1,397 | 734 | 714 | 1,233 | 1,763 | 1,804 |
| sd / se | 124.1 / 55.5 | 585.1 / 261.6 | **0.0 / 0.0** | **0.0 / 0.0** | 264.0 / 118.1 | 653.7 / 292.3 | — |
| 均值 score | 25,933.0 | 21,382.0 | 903.4 | 2,367.4 | 17,767.6 | 29,373.8 | 25,114.2 |
| 过关 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| 首死停 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| `renders` | = frames | = frames | = frames | = frames | = frames | = frames | = frames |

`obs_window=consecutive`、`renders == frames`（每决策 2 次渲染）、引擎 sha `591c521d…`、`bombs_used=0`
—— 全部行都通过 `window_alignment` 校验，**没有混用旧 strided 尺**。

**Eg/Fg 的 sd=0 是"退化"，不是"稳定"**：5 个种子给出**逐帧相同**的 734 / 714 帧、score 836–1,148 /
2,002–2,611，说明策略对 RNG 种子完全不敏感（同一个（近似）常数动作序列，在第一个弹幕波就撞死）。
这正是 `geo_restore` 把目标压平到熵 2.87、KL 0.024 之后学生学到"均值动作"的后果，**不能**当作
"方差小、结果可靠"来读。

### 5.2 配对（同一组 5 个评测种子）

| 对比 | 均值差（帧） | sd | 配对 t | 同向种子 | ≥ +300 帧 |
|---|---:|---:|---:|---:|---|
| **E − v2i/E**（同 8 片，只换标签来源） | **−165.0** | 358.0 | **−1.03** | 2/5 | ✗ |
| **F − v2i/F**（同 27 片，只换标签来源） | **−353.8** | 736.8 | **−1.07** | 2/5 | ✗ |
| Eg − v2i/E（8 片 + geo） | −475.6 | 264.0 | −4.03 | 0/5 | ✗ |
| Fg − v2i/F（27 片 + geo） | −1,102.0 | 653.7 | −3.77 | 0/5 | ✗ |
| E − v2（基线，混杂数据量） | −674.2 | 267.6 | −5.63 | 0/5 | ✗ |
| F − v2（基线） | −256.6 | 479.8 | −1.20 | 1/5 | ✗ |
| F − E（数据量，27 vs 8） | **+417.6** | 644.5 | 1.45 | 3/5 | ✓（但非本卡判据） |
| Fg − Eg（数据量，geo on） | −20.0 | 0.0 | 0.00 | 0/5 | ✗ |

**读法**：预注册的对照是"同数据量、只换标签来源"，两组都是**负值**（E −165、F −354）。
两者的 |t| 都 ≈1（n=5、se 55–262），所以严格的统计说法是"**测不出提升，点估计为负**"；
但判据是"≥ +300 帧"，而这里连方向都不对，**判据是不通过，不是"未达显著"**。
`v2j − v2` 也都是负的（E −674、F −257），说明"教师换档"这件事在闭环上没有一次是正收益。
唯一仍然稳定的效应是**数据量**（F − E = +417.6，3/5，t=1.45），与 EXP-P3-004/005 的方向一致。

### 5.3 为什么可能是负的：一个结构性的、可检验的解释

`v2j` 的 3/4 槽计划（占 23.7 %）是"把首动作按住 8 帧"的计划，但 `decide`/CLI 的提交节奏是
`replan_decisions = 2` 槽（4 帧）——**被打了分的那个计划从来没有被执行完**。对 2 槽计划这个
不一致不存在（`slots[:2]` 就是全部），所以受影响的正好是"新增的那部分标签"：

* 教师为了给 4 槽计划打分，把**未来 8 帧**的净空/激光/边界都算进了分数，于是它的 argmax 会倾向
  "按住一个方向更久"——但学生执行时 4 帧后就要重新决策，而学生的策略是在**标签分布**上训练的，
  不是在这个"半截执行"上训练的；
* 结果：标签分辨率升高（学生更容易拟合 argmax），但**拟合的是执行不到的意图**；
  离线 top-1 上升（0.33 → 0.45）、闭环下降（−165 帧）这个组合与它一致。

这个解释**没有被隔离验证**（要验证就得把 `replan_decisions` 也改成 4 再重采一轮，见 §8 P0），
所以它只是"与三条独立观测一致"的假说，不是结论。写在这里是为了让下一个人知道从哪里切开。

## 6. 判据（预注册）

| 判据 | 门槛 | E | F | Eg | Fg |
|---|---|---|---|---|---|
| 相对 v2h/v2i 同量口径的配对提升 | **≥ +300 帧** | **−165.0** ✗ | **−353.8** ✗ | −475.6 ✗ | −1,102.0 ✗ |
| macro recall（val） | **≥ 0.20** | **0.0999** ✗ | **0.1695** ✗ | 0.0829 ✗ | 0.0749 ✗ |
| S1N 均值 ≥ v2 基线（1,718.8） | 1,718.8 | 1,044.6（60.8 %）✗ | 1,462.2（85.1 %）✗ | 734.0 ✗ | 714.0 ✗ |
| ≥ 教师 30 %（4,070 帧） | 4,070 | 1,044.6（25.7 %）✗ | **1,462.2（35.9 %）** ✓ | 18.0 % ✗ | 17.5 % ✗ |

**判定：按预注册规则（两腿都要过）→ FAIL。** 三条如实标注：

1. **帧数腿不是"没测出"，是"方向反了"**：E/F 的点估计分别 −165 / −354 帧，`≥ +300` 的门槛连边都
   没碰到；配对 t 分别是 −1.03 / −1.07（n=5、se 55.5 / 261.6），所以**既不能声称显著变差，也不能
   声称"没有差别"**——能声称的是"**没有看到 +300 帧，且最好的点估计也低于 v2i**"。
2. **`macroR ≥ 0.20` 这条腿在两轮里从未被"重教师"通过**（EXP-P3-004 只有 armA 的 val 0.2514 过过，
   那是 `fast`/k1 标签）。本轮最高的 F 也只有 0.1695：**分辨率越高、macroR 越低**这个反向关系在
   §4.1 里是可见的（E 0.0999 < armA 0.2514，但 top-1 0.4527 < armA 0.5656 也低）。
3. **唯一过门的是 F 的"≥ 教师 30 %"（35.9 %）**，而它的对照 `v2i` F 是 44.6 % —— 也就是说这一条
   "过"是继承了数据量，不是 v2j 的功劳。

## 7. 成本核算：v2j 作为长期教师能不能接受

### 7.1 单决策耗时（对照 `v2h` ~93 ms/决策、`v2i` ~114–116 ms/决策）

| 口径 | 值 | 来源 |
|---|---:|---|
| `v2j` channel A（教师整条决策） | **186–193 ms/决策**（seeds 1–4，`loadavg` 3–5） | `channel_a.json` |
| `v2j` channel A（`loadavg` 10–16 时） | **235–236 ms/决策**（seeds 5/6） | 同上 |
| 全批平均 | **199.4 ms/决策**（5,255.1 s / 26,349） | §2 |
| **搜索相 `v2h` → `v2j`（开局窗口）** | **21.0 → 39.5 ms（×1.88）** | `tape_dispersion.json` |
| 搜索相（中期窗口，帧 4,800） | 51.6 → 79.0 ms（×1.53） | `tape_dispersion_mid.json` |
| 搜索相（满束闭式 `expansions` 比） | **30/17 = 1.76**（`limit=10` 的测试 tape） | 单测 |
| 每秒决策数（单 worker） | **5.19–5.37 决策/s**（636 s/3,300；610 s/3,275） | `channel_a.json` |
| 每秒决策数（2 worker 合计） | **8.06 决策/s**（26,346 / 3,267 s） | `progress.json` |
| 8 片采集墙钟（2 worker） | **54.5 min**（v2i 同口径 33.5 min，×1.63） | §2 |

### 7.2 判定：作为长期教师**可接受，但每次 DAgger 轮次要多付 ~60 % 的标签墙钟**

- **搜索只贵 ×1.88**（不是 ×2 以上）：根网格 36 → 72 是唯一成本，`record`/`verify`/`commit`
  三个相位的耗时不变，所以整条决策只从 114–116 ms 涨到 186–193 ms（**×1.63**）。
- **绝对水平仍然很低**：0.19 s/决策 × 26.4k 决策 = 84 min 串行 / 54.5 min（2 worker）。
  与 EXP-P3-005 的 33.5 min 相比多 21 min —— 对一个"每轮 DAgger 产 2.6 万条标签"的流水线，
  这个代价是**可接受**的（它比一次 S1N 评测波次长不了多少，也不是实时约束）。
- **没有实时性要求**：教师是离线标签器（`realtime_multiple` 0.02–0.03），慢 63 % 不改变任何
  契约；唯一需要关注的是"8 片采集是否仍能在一次会话内完成"——本轮实测 54.5 min，**是**。
- **不建议再用更大 durations**：`{1,2,3,4}` 已经吃掉 ×1.63 的预算，而 §3.2 显示 240 条 tape 里
  只有 34 % 的决策真正用上 3/4 槽；再加 `{5,6}` 会把根网格推到 108 而边际收益按同样的比例衰减
  （每多一档只是多 18 个根，且越长的 hold 越难赢过 align 的 0.021 点/px）。

## 8. 下一条最小改动

先写清本轮**已经排除**的东西，再给下一步：

| 已被证伪/排除 | 证据 |
|---|---|
| `Weights.margin_cap` 16→64（`v2i`） | EXP-P3-005 §7.1：钳位在 `search._frame_margin`，改权重不产出一条新标签 |
| 连续项 + 更深搜索（`v2h`/`v2i` 预算） | EXP-P3-005 §7.3：`>0.1` ×1.85，闭环 +97 / +43 帧（t≈0.3–0.4） |
| **动作持续时间集 1,2 → 1,2,3,4（本卡）** | `>0.1` **×4.31**、离线 top-1 +0.124，闭环 **−165 / −354 帧**（判据 FAIL） |
| 在 v2j 标签上再叠 `geo_restore` | Eg/Fg sd=0，闭环 734/714 帧（退化策略） |
| `verify_k`（EXP-P3-004） | 对标签分辨率零影响 |

| 优先级 | 改动 | 为什么是这一条 | 代价 |
|---|---|---|---|
| **P0** | **把 `durations` 与 `replan_decisions` 对齐**（`replan_decisions` 2 → 4，或 durations 只留 `{2,4}` 的倍数） | 本轮暴露的**结构性缺陷**：`v2j` 有 23.7 % 的决策选 3/4 槽，但 `decide`/CLI 只提交 `replan_decisions=2` 槽 —— 标签说的"我要按住这个方向 8 帧"，执行只走了 4 帧。这是"标签语义与执行语义不一致"，不是分辨率问题；它同时解释了为什么"更长的承诺"在闭环上没有收益：**学生从没真正执行过被标签打分的那个计划**。 | 纯配置（`replan_decisions` 是 `TeacherConfig` 字段），但会改变采集的提交节奏，必须重采一轮 8 片 |
| P1 | **数据量/DAgger 而不是再换教师** | 本卡唯一稳定的闭环效应仍是数据量（F − E = +417.6，3/5），且 EXP-P3-004/005 两次都测到 550–606 帧量级；DAgger R2 已经在跑，方向对 | 已有预算 |
| P2 | 真·提高 margin 分辨率：`config.MARGIN_CAP` 16 → 60 并贯通 `raster.MARGIN_Q_CAP` / `search` 束键 | 这是**唯一还没测过**的标签侧通道（EXP-P3-005 §8 P3），上限 63.75 px（1/4 px × 1 字节） | 改 `search.py`/`raster.py`（超出本任务允许范围），所有既有档的束排序键一起变，需逐档回归 |
| 不推荐 | 继续加 durations（`{1..6}`）、继续抬 `horizon`、继续调 `margin_cap` | 本轮已证 durations 是"真旋钮但非闭环瓶颈"；`horizon` 只加长视野不改变首动作位移（EXP-P3-005 §8 P1 原话）；`margin_cap` 惰性 | — |

## 9. 测试

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_v2j\pytest_final -q `
  tests\test_v400_teacher.py tests\test_v400_margin_resolution.py tests\test_v400_labels_signal.py `
  tests\test_v400_d0_dual.py tests\test_v400_student.py
# → 1 failed, 193 passed, 2 skipped，exit 1
#   唯一红 = test_child_recording_is_one_batch_and_matches_the_main_process_tape
#   （既有 v2h 激光字段 `lasers_damageable` 的行形状差异，EXP-P3-005 §9 已作为同一处已知红记录，
#    与 v2j 无关：注册 v2j 前后都红）
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_v2j\pytest1 -q tests\test_v400_teacher.py `
  -k "v2j or duration_set or v2f_profile_is_v2a_plus"
# → 6 passed, 110 deselected，exit 0
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_v2j\pytest_margin -q tests\test_v400_margin_resolution.py
# → 8 passed，exit 0（`v2j` 注册后，该文件的"未注册名"守卫已按授权改为 `v2k`）
```

`tests/test_v400_teacher.py` 新增 5 个用例 + 1 处**语义保持**更新：

1. `test_v2j_is_v2h_with_the_duration_set_widened_and_nothing_else`：`v2j.durations == (1,2,3,4)`、
   `v2j.weights is v2h.weights`（同一实例）、`replace(v2j, mode="v2h", durations=(1,2)) == v2h`
   逐字段相等；`horizon/beam/verify_k/max_iterations/replan_decisions/decision_frames/border/
   record_items/shoot_*/bomb_rule1_*/clip_y` 逐项与 `v2h` 相等；`CONFIGS`/`PROFILES` 注册、
   CLI `--mode v2j` 可选、"未注册名"守卫移到 `v2k`。
2. `test_v2h_v2i_and_the_earlier_profiles_keep_their_duration_set`：`V2H` **逐字段重新声明**后相等
   （任何一处被改动都会红）；`full/fast/fast3/v2a..v2i/v3a/v3b` 共 11 档的 `durations` 全部是 `(1,2)`。
3. `test_v2j_durations_reach_the_search_roots_and_the_executed_plan`：`roots` 36 → **72**、plan 的
   `duration ∈ {1,2,3,4}` 且确实有候选选到 >2；`simulate_plan` 的位移随 hold 线性
   （cand 1「北」：4.0 px/帧 × 2 帧 × duration，逐档 `pytest.approx`）；cand 9（不动）位移为 0。
4. `test_v2j_price_is_bounded_by_the_root_grid`：`roots` 比恰为 2.0，`expansions` 比
   **30/17 = 1.76**（满束闭式：`18·Σ(limit−d)`，`limit=10`），界 `1.6–2.0`；相对墙钟 < 4× `v2h`。
5. `test_v2j_folded_back_to_two_durations_is_bit_identical_to_v2h`：折回 `(1,2)` 后
   `beam_search` 的 18 路分数与 `v2h` **逐位相同**、且与 `beam_search_reference` 相同（`use_raster=False`）；
   另用一个 align 目标在 +88 px 的 tape 证明**宽档真的能改标签**（cand 3「东」的 duration 与 score 都变大）。
6. 既有 `test_v2f_profile_is_v2a_plus_the_top_band_and_keeps_high_penalty` 里"未注册名报错"的守卫由
   `v2j` 移到 **`v2k`**（`v2j` 本轮注册了），并加断言 `get_config("v2j").mode == "v2j"`；
   **没有**放松任何既有断言，`v2h`/`v2i` 的逐字段断言全部保留。

**另有一处跨文件、语义保持的更新（已获任务授权）**：
`tests/test_v400_margin_resolution.py::test_fast_v2a_and_v2h_keep_their_published_numbers` 里
EXP-P3-005 写的"未注册名"守卫 `get_config("v2j")` 必然失效（本轮注册了 `v2j`），已改为
`get_config("v2k")`；该文件 **8 passed**。这不是放松断言，而是把"最新档之后的名字"这个语义往后挪一位。

## 10. 命令与退出码（可复现）

```bash
# WSL：v2j 8 片采集（2 worker，2 teacher，引擎 rl_laser，sha 1462829b…）
wsl -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_v2j/launch_batch.sh
# → exit 0（后台批处理；日志 .tmp_v400_v2j/logs/batch_v2j.log；进度/对齐逐片见 progress.json）
wsl -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_v2j/status.sh
```

```powershell
# Windows：标签分辨率 + 端点散布（v2j vs v2i/v2h 的既有统计）
python .tmp_v400_v2j\analyze_labels.py .tmp_v400_v2j/v2j .tmp_v400_margin/v2i .tmp_v400_p3sig/k1
# → exit 0（写 label_resolution.json：roots[4] + pairing）
# 逐片加载门
python .tmp_v400_v2j\verify_load.py .tmp_v400_v2j/v2j          # → exit 0（写 load_gate.json）
```

```bash
# WSL：18 动作位移散布（真实 tape，4 个窗口 × 60 决策，1 引擎）
wsl -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_v2j/run_dispersion.sh 60 1 \
  .tmp_v400_v2j/tape_dispersion.json
wsl -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_v2j/run_dispersion.sh 60 1 \
  .tmp_v400_v2j/tape_dispersion_w600.json .tmp_v400_v2j/v2j/shards/dual-s0d1seed1/labels.jsonl 600
# → exit 0（写 tape_dispersion*.json；回放窗口带 hash_post 校验）
```

```bash
# WSL：四臂训练（同配方；E/F = geo_restore off，Eg/Fg = on）
wsl -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_v2j/run_train.sh
# → 见 §4（TRAIN_DONE；每臂 rc 记录在 logs/train_exits.log）
# WSL：S1N 5 局 ×4 臂（consecutive 新尺，2 进程 = 2 引擎）
wsl -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_v2j/run_evals.sh consecutive
# → 见 §5（EVALS DONE window=consecutive；每臂 rc 记录在 logs/eval_exits.log）
```

```powershell
python .tmp_v400_v2j\summarize_arms.py     # → exit 0（写 arm_table.json）
python .tmp_v400_v2j\summarize_eval.py     # → exit 0（写 eval_table.json）
python .tmp_v400_v2j\s1n_effects.py        # → exit 0（写 s1n_effects.json）
```

## 11. 变更与产物

| 文件 | 变更 |
|---|---|
| `touhou_rl/v400/teacher/config.py` | **新增** `V2J`（`v2h` 的同一个 `_V2H_WEIGHTS` 实例 + `durations=(1,2,3,4)`），注册进 `CONFIGS` 与 `PROFILES`；`FAST`/`FAST3`/`V2A`/`V2H`/`V2I`/`V3A`/`V3B` **未改**（测试逐档钉住） |
| `tools/rl/v400_teacher.py` | 只改 `--mode` 的**帮助文本**（`choices` 早就是 `tuple(sorted(CONFIGS))`，`v2j` 零改动即可选） |
| `tests/test_v400_teacher.py` | 新增 5 个 v2j 用例 + 1 处语义保持更新（"未注册名"守卫 `v2j` → `v2k`） |
| `tests/test_v400_margin_resolution.py` | 1 处语义保持更新（EXP-P3-005 的同一守卫 `v2j` → `v2k`；已获任务授权） |
| `Document/Update/Alpha/v4.0.0/experiments/EXP-P3-007-teacher-v2j-durations.md` | 本卡 |
| `.tmp_v400_v2j/*` | `v2j/`（8 片）、`logs/`、`batch.py`、`launch_batch.sh`、`status.sh`、`run_dispersion.sh`、`tape_dispersion.py`、`run_train.sh`、`run_eval.sh`、`run_evals.sh`、`launch_train.sh`、`launch_evals.sh`、`analyze_labels.py`、`verify_load.py`、`summarize_arms.py`、`summarize_eval.py`、`s1n_effects.py`、`tape_dispersion*.json`、`label_resolution.json`、`load_gate.json`、`arm_table.json`、`eval_table.json`、`s1n_effects.json` |

**未改**：引擎、协议（`gym/protocol.py`）、依赖、第三方、`build/`（`rl`/`rl_laser` 的 sha256 只读校验：
`591c521d…` / `1462829b…`）、`touhou_rl/v400/teacher/{search,raster,raster_numba,decide,verify}.py`
（**`durations` 本来就是配置项**，`search.py` 一行未改）、`touhou_rl/v400/student/*`、
`tools/rl/v400_d0_dual.py`、训练器。未删除任何文件，未提交。

## 12. 限制（不要过度解读）

1. **不是 G3、不是 PSN、不是 confirm split**；1 个 stage（stage 0 = 真 Stage 1）、1 个难度
   （normal）、n = 5 局。
2. `v2j` 只动 `durations`：`weights`/`horizon`/`beam`/`verify_k`/`max_iterations`/rule-1 窗口
   逐字段仍是 `v2h` 的，所以任何差异**只能**归因给根网格（18×2 → 18×4）。
3. 采集按"决策数"截断（`stop_reason=decisions`），另有 3 片自然 `stage_changed`（seeds 2/7/8）。
4. 窗口尺只有一把：本卡所有 S1N 数字都是 `--obs-window consecutive`；历史 `strided` 数字不比较。
5. 训练与评测都只有 1 个训练种子（seed 0）与 5 个评测种子，配对 t 的自由度是 4。
6. 机器上同时有零售 trace 批跑、DAgger R2、P2 上量三个外部任务，采集 wall 受负载影响
   （`loadavg` 3–16，seeds 5/6 的 channel A 因此慢了 26 %）；单决策耗时是在这个负载下测的，
   不代表空载吞吐上限。
7. **"负收益"不能读成"durations 有害"**：E/F 的配对差 −165 / −354 帧，|t| ≈ 1（n=5、se 55–262），
   严格说只支持"没有 +300 帧的提升，点估计低于 v2i"。要把它升级成"显著变差"需要 5→15 局以上
   或者更多训练种子。
8. **`durations` 与 `replan_decisions` 的语义不一致是本卡新发现的结构性问题，尚未修复**：
   `v2j` 有 23.7 % 的决策把首动作承诺到 3/4 槽（搜到 8 帧之后），但 `decide`/CLI 只提交
   `replan_decisions = 2` 槽（4 帧）就重新规划。也就是说**标签打分的那个计划从未被真正执行完**。
   这既可能是"分辨率涨了、闭环没涨"的原因，也可能是"闭环节点估计为负"的原因；它**没有**在本轮
   被隔离成一个实验（§8 P0 就是它）。
9. **18 动作散布只在 stage 0 的 4 个窗口（240 条 tape）上测过**：窗口 1,200 处完全惰性
   （分数逐位相同）、窗口 600 处 +48 %，说明效应强状态依赖；不能外推到其它 stage / 难度。
10. `margin < 16` 的候选数在两个档下**逐位相同**这条结论，是在**stage 0、普通难度、引擎
    `rl_laser`** 上测的；它支持"durations 不会把计划压进 margin 活着的区间"，但若换成弹幕更密的
    stage（4/5/6）需要重测。

> **更正注记（2026-11-10，EXP-P3-009）**：v2j 动作集的 `−165` / `−354` 两对臂是单 checkpoint × 5 局、每对只差一次训练；按新判据 = 无法判定（CI 重叠且低于分辨极限 1,309.7 帧）。原始数字不改。详见 [EXP-P3-009](<EXP-P3-009-eval-power.md>)。
