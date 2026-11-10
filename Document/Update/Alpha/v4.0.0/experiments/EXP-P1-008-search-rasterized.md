# EXP-P1-008 致死掩码栅格化：有弹幕时 search 430 → 75 ms/决策（5.7×）、判死逐点一致、整决策 0.72× 实时

- 日期 / 工作包：2026-11-09 / P1 教师（[11-P1](<../../11-P1-搜索教师.md>) §2.1、§3；承接 EXP-P1-007 §8 的「搜索是唯一大头」）
- 目标：
  1. 把每帧致死 AABB 栅格化到 playfield 网格，束搜索的碰撞/余量判定改成**查表**；
  2. **保留精确路径**（`beam_search_reference` + `beam_search(use_raster=False)`），逐项对比 `geo_score[18]` 与 `chosen_plan`；
  3. 判死结论必须一致；若不一致，以引擎 verify 为准统计分歧率并说明是否接受；
  4. 真实引擎（`build/rl_hazard2`）实测 search 分位与单决策总耗时、实时倍率；
  5. stage 1 normal seed 1 跑 `--max-frames 16000 --stop-on-death`，报过面结果；
  6. `tests/test_v400_teacher.py` 增补（栅格正确性、掩码与精确路径一致、等价性、性能回归）。
- 预注册（运行前固定，可证伪）：
  1. **判死逐点一致**：`LethalTable.query` 的 dead 与 `_frame_margin<=0` 在**每一个查询点**上必须完全相同（不是统计上接近）；
  2. **束搜索判死一致**：同一 tape 上新旧 search 的每个 `Plan.death_frame` 的 `None/非None` 必须相同；
  3. **性能**：有弹幕时 search ≤ 60 ms/决策（关卡级、真实引擎）；
  4. **不伪造**：margin 量化差异、面级（plan）分歧、未达标的性能门都照实列。
- 环境：`/root/Touhou_RL/third_party/th08-rl/build/rl_hazard2/th08-modern`（`sha256=f172399f…`，**未重建**），数据 `/root/v400-data/p1s1/game`，WSL 内一次一个引擎；Python 3.10 / numpy 2.2.6（WSL）与 Python 3.12 / numpy 2.5.4（Windows 单测）。**不引入新依赖**（numba 虽在 WSL venv 里，本卡未使用，见 §8）。
- **负载说明**：Windows 侧零售 trace 批跑同时在跑（按要求不干扰），所以 WSL 的绝对耗时偏高；离线 A/B（Windows）与真机 A/B（WSL）分开列，不混用。

---

## 1. 实现

代码落点：

| 文件 | 改动 |
|---|---|
| `touhou_rl/v400/teacher/raster.py` | **新增**：`lattice()` / `flat_index()` / `_cell_ranges()` / `_paint()` / `distance_transform()` / `LethalTable` / `build_table()` |
| `touhou_rl/v400/teacher/search.py` | `beam_search(..., use_raster=True)`：每帧查表；`_root_nodes` 接受 `table`；`detail` 增 `raster_cells` / `raster_cell_px` |
| `tests/test_v400_teacher.py` | 新增 6 项栅格/等价/性能测试；3 项既有「vectorised == reference」测试显式走精确路径（见 §7） |
| `tools/rl/v400_teacher.py` | **未改**（CLI 的 `--engine/--max-frames/--stop-on-death` 已够用） |

### 1.1 网格与两张掩码

- cell = **2 px**（`CELL`），`x∈[8,376]`、`y∈[16,432]` → **184 × 208 = 38,272 cell/帧**，48 帧 = 1.84 M cell；
- 每帧把 `HazardTape.box_array(frame)`（子弹半宽 `collisionSize/2`、敌人 `hitbox/3`，即 EXP-P1-001/004 的模型，**未改**）按 hurtbox 半宽扩张成 `[min-half, max+half]`——这个矩形**恰好**是「引擎会判为命中」的 hurtbox 中心集合，因为 AABB 判定是 `gap<=0`（含等号，接触即死）；
- 三档覆盖谓词（`_cell_ranges`）：`hit` = cell 方块与矩形**相交**；`core` = cell 方块**完全落在**某个矩形内；`centre` = cell 中心落在矩形内（本版未用）；
- 栅格化用**二维差分 + 两次 cumsum**：散布 O(矩形数) 个角点，`cumsum` O(cell)，与「矩形数 × cell 数」无关。两档谓词合并在一次 `np.unique`+散点 + 一次二维 cumsum 里。

### 1.2 判死为什么**精确**

- `not hit` → 该 cell 内**任何**点都不在致死区域内 → 活着（证明）；
- `core` → 该 cell 内**任何**点都在致死区域内 → 必死（证明）；
- `hit and not core` → 唯一不确定的一圈（约 1 cell 宽的边界带）：用**原封不动的 AABB 表达式**（`LethalTable.exact`，与 `_frame_margin` 同一次减法、同一个 `sqrt`、同一个 `MARGIN_CAP`）逐点重算。

所以查表只可能「少算」，不可能「改判」。§3.1 的 238 万点逐点比对 0 分歧是这条论证的实测。

边界带有多大（40 条真机 tape，`detail.raster_band_exact / raster_band_queries`）：**查询点的 9.98%（最大 26.2%）** 落在 `hit & ~core`，即 90% 的几何问题由一次 gather 回答，其余 10% 用原表达式精确重算——这 10% 也是 §8 里 `exact` 段耗时的来源。

### 1.3 margin 字段（量化，允许的差异源）

- 不是逐 cell 求 `min over boxes`（那还是 `cells × boxes`，等于没提速），而是对 `hit` 掩码做**带截断的精确欧氏距离变换**：
  1. 逐帧、逐列的「到最近 hit 的一维距离」（索引差分，`np.maximum.accumulate`/`minimum.accumulate`，分段按帧）；
  2. 沿列方向做 `min_k (k² + g(x+k))`，`|k| ≤ cap_cells`——截断到 `MARGIN_CAP=16 px` 后它是**精确**的（更大的偏移必然超过 cap）；
- 距离场跑在 **4 px 格**（`MARGIN_CELL`）上再最近邻铺回 2 px 表：48 帧 × 4,992 × 92 = 459 k cell，成本是 2 px 场的 1/4；
- 存成 **1/4 px 定点 uint8**（0…64）；`core` cell 的 margin 由超集掩码的距离变换自然为 0，不再二次打补丁。

### 1.4 查表

一张 `uint16` 表/帧：低字节 = margin 定点，高字节 = flags（bit0 `hit`、bit1 `core`）。一次 gather 同时回答「多远」和「死没死」：

```python
codes  = table[frame].ravel()[flat_index(centres)]
margins = (codes & 0xFF) * 0.25
flags  = codes >> 8
dead   = flags & 1                     # 超集
band   = dead & ((flags & 2) == 0)     # 唯一不确定带 -> exact()
```

### 1.5 已知边界（照实写）

- 距离场只覆盖 playfield：**刚好在边界外**的致死物（例如 y<16 的子弹）对死亡判定无影响（中心被 clamp 在界内），但会让贴边 cell 的 margin 被高报。实测 `|Δmargin|>4px` 的 cell 从 8,586（修 bug 前）降到 **21/38,272（0.05%）**，全部贴着上边界；
- margin 是 4 px 格 + 1/4 px 定点的量化值，不是精确值：见 §3.2 的实测分布。

---

## 2. 等价性口径

三条一起看，缺一不可：

1. **点级判死**（`LethalTable.query` vs `LethalTable.exact`）：逐点 `np.array_equal`；
2. **计划级判死**（新旧 `beam_search` 的 `Plan.death_frame is None`）：逐 root 比；
3. **`chosen_plan` / `geo_score[18]`**：新旧逐项比，margin 允许 ≤1 cell 的量化差异。

第 1 条是「查表没改判」，第 2、3 条是「量化 margin 改了束的排序」——两者必须分开报，否则会把「排序漂移」误说成「判死错」。

---

## 3. 离线 A/B（85 条真机 tape，Windows）

样本：WSL 引擎 stage 1 normal seed 1 逐决策录制 **1,400 条** 48 帧 tape（`source="child"`），取**最密的 85 条**（单帧致死 AABB 峰值 42，均值 **30.1 个/帧**）。同一 tape 上跑 `beam_search(use_raster=False)`（改动前的实现）与 `beam_search(use_raster=True)`。

### 3.1 判死一致率

| 口径 | 结果 |
|---|---|
| 点级（随机点 + 全 cell 中心，跨帧） | **2,380,000 点 / 0 分歧 / 100.000000%**（误判死 0、误判活 0） |
| plan 集（18 个首动作是否都有计划） | **85/85 相同** |
| 计划级 `death_frame` 的 `None/非None` | 1,530 个计划中 **39 个不同（2.55%）** |
| `Plan.slots` | 1,530 个中 **402 个不同（26.3%）** |
| 决策级 best action | 85 个决策中 **18 个不同（21.2%）** |
| `geo_score[18]` 最大差 | **105.46** |

**如何读这张表**：点级判死是**证明 + 实测**都精确的；计划级那 39 个分歧不是判死算错，而是**量化 margin 改写了束内排序**——新旧搜索在某个 root 上留下了不同的节点，其中一条路径本来会死。`geo_score` 的 105 差正是这种「活/死计划互换」造成的（活计划 ~116 分，死计划 `death_frame/H*10` 只有几十分），不是 margin 差 105 px。

### 3.2 margin 量化差异（活着且在两个搜索里都活的点）

| 指标 | 值 |
|---|---|
| 平均 \|Δmargin\| | **3.55 px** |
| 最大 \|Δmargin\| | 16.00 px（仅出现在「最近致死物在 playfield 外」的贴边 cell，见 §1.5） |
| 单测里（合成 tape，致死物全在界内） | 最大 \|Δ\| ≤ **2 × CELL = 4 px** |

### 3.3 search 耗时

| 实现 | p50 | p90 | max |
|---|---|---|---|
| 精确（改动前） | **259.72 ms** | 273.31 | 304.84 |
| 栅格（改动后，含建表） | **78.50 ms** | 137.27 | 200.22 |
| 建表（`build_table` 单项） | 33.61 ms | 34.47 | 35.13 |

p50 加速 **3.31×**；p90 加速 1.99×。建表 p50 的分解：`paint` 13.50 + `distance` 11.87 + `pack` 8.15 ms。
未达标：**78.5 ms > 60 ms**（差 31%），见 §8。

---

## 4. 真机 A/B（WSL，同一 tape 上新旧各跑一次）

命令与产物见 §9。跑到 frame 800（stage 1 的弹幕从约 frame 768 起）后，对同一决策点录一条 tape，**同一条 tape** 上跑新旧两版搜索，两个 winner 都送引擎 verify，然后提交栅格版 winner 的前 2 个子决策（4 帧）。

| 项 | 值 |
|---|---|
| 决策数 | 30 |
| 单帧致死 AABB 均值 | 7.61 |
| 精确 search | 均值 **114.68 ms**，p90 160.97 |
| 栅格 search | 均值 **67.04 ms**，p50 66.04，p90 67.96，max 86.06 |
| 加速 | **1.71×** |
| 计划级判死分歧 | **0 / 540 计划** |
| `Plan.slots` 分歧 | **0 / 540** |
| margin 差 | 0.00 px（该密度下两侧 margin 都是 `MARGIN_CAP`） |
| 引擎 verify 判死一致 | **30/30** |

真机 A/B 的密度（7.6 个/帧）低于离线那 85 条（30.1 个/帧），所以 1.71× < 3.31×：栅格版的固定成本（建表 ~33 ms + 束机器 ~18 ms）在低密度 tape 上占比更大。该 A/B 跑在 `pack`/`distance` 微优化**之前**（当时建表 36.6 ms，现 33.6 ms），所以 67.04 ms 里约有 3 ms 是旧建表的成本。

---

## 5. stage 1 关卡级（`--max-frames 16000 --stop-on-death`）

命令（WSL，一次一个引擎）：

```powershell
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_teacher.py \
  --decisions 4200 --max-frames 16000 --stop-on-death --stage 1 --difficulty normal --seed 1 \
  --mode full --quiet --out .tmp_v400_raster_stage1.jsonl > .tmp_v400_raster_stage1.log 2>&1"
```

结果：**max_frames 先到（16,000 帧 = 266.7 s 游戏内），0 死亡，未过面**；单决策 92.3 ms 均值 → **0.722× 实时**（EXP-P1-007 是 0.127×）。

### 5.1 结果（4000 决策 / 16,000 帧）

| 指标 | 实测 | 对照 EXP-P1-007 |
|---|---|---|
| 停止原因 | **`max_frames`** | 同（`max_frames`） |
| 决策数 | **4,000** | 1,500 |
| 存活帧数 | **16,000 帧**（266.7 s 游戏内） | 6,000 帧 |
| **是否过面** | **否**（`stage_clear=false`，停止时 `stage=1`） | 否 |
| 死亡次数 | **0**（`deaths=0.0` 全程） | 0 |
| bomb 使用 | **3 次**（`bombs_spent=3.0`，`bomb_decisions=3`） | 0 |
| search（有弹幕 1,683 决策） | **p50 74.49 / p90 83.17 / max 154.41 ms**，均值 **75.03** | **430.17 / 810.65 / 2400.19** |
| search（空 tape 2,317 决策） | p50 **17.69**（均值 17.74） | 38.9（空 tape 均值） |
| 单决策总耗时（有弹幕） | 均值 **139.32**，p50 135.06，p90 170.78 | 591.3 |
| 单决策总耗时（全部） | 均值 **92.32**，p50 58.37，p90 154.08 | 526.07 / 496.25 / 937.77 |
| 四段均值（全部） | record **27.05** / search **41.84** / verify **21.80** / commit **1.15** / events 0.05 | 45.62 / 430.17 / 46.25 / 3.12 / 0.20 |
| 墙钟 | **369.27 s**（6 min 09 s，另加启动 1.18 s + RESET 0.41 s） | 789.1 s |
| **实时倍率** | **0.7221×**（16,000 帧 / 369.27 s / 60） | 0.127× |
| 决策/秒 | 10.83 | 1.90 |
| 1× 实时门槛 | 66.67 ms/决策（4 帧/决策）；实测 92.32 → **差 1.38×** | 差 7.9× |
| 协议调用 | 727,106 次（1 个进程） | 275,296 次 |
| 样本形态 | 单帧致死 AABB **均值 33.04、峰值 281** | 均值 46.1、峰值 191 |
| 一致性（12,000 个已验证计划） | `agree_alive 11,974`、`agree_dead 0`、`geo_dead_engine_alive 0`、**`geo_alive_engine_dead 26`**、`trajectory_mismatch 11,964` | `agree_alive 4,496`、`geo_alive_engine_dead 4` |

要点：

1. **search 从 430.17 → 75.03 ms（有弹幕，均值）**，相对下降 **5.73×**；空 tape 17.7 ms（几乎全是束搜索机器的固定成本，与 EXP-P1-007 的 38.9 ms 同量级但机器负载更轻）。
2. 整决策 **526.07 → 92.32 ms**，实时倍率 **0.127× → 0.722×**（5.7×）。离 1× 还差 1.38×，缺口现在**不在 search**（41.8 ms 均值）而在 record 27.0 + verify 21.8 + search 41.8 = 90.6 ms 的三段平摊——搜索再快一倍也只能到 0.98×，要过 1× 必须同时压 record/verify（两者本卡未动）。
3. **`--max-frames 16000` 是先到的上限，不是过面**：stage 1 normal 在 266.7 s 游戏内没有结束，`stage_at_stop=1`。要拿过面结论得把帧上限提到 stage 1 真实长度或干脆不设上限。
4. 3 次 bomb 是 11-P1 §4 规则 1（所有已验证计划都会死 → 主动 bomb）在真实弹幕下第一次触发（EXP-P1-007 里 `bomb_decisions=0`）；`EVENTS` 全程可用但 `PLAYER_HIT` 仍为 0，规则 2 没触发。
5. `geo_alive_engine_dead 26`（几何判活、引擎判死）是已知的 fork 晚 1 帧偏差（EXP-P1-006/007 的 `trajectory_mismatch` 同源），不是本卡引入的；它由 label 如实记录，未做任何掩盖。

### 5.2 与 EXP-P1-007 的可比性（必须说清）

EXP-P1-007 的 1,500 决策是在「Windows 零售 trace 批跑同时运行」时测的，本卡的 4,000 决策在**同一台机器、同一份引擎、同一个 stage/seed**但负载更轻的时段跑。所以：

- **search 430 → 75 ms 里含有负载差异**，不能全记在栅格化的账上；
- **干净的对照只有 §3 的同 tape A/B**（同一时刻、同一 tape、两个实现各跑一遍）：30.1 个致死物/帧时 **259.72 → 78.50 ms，3.31×**；§4 的真机同 tape A/B（7.6 个/帧）**114.68 → 67.04 ms，1.71×**；
- non-search 三段也一起变快了（record 45.62 → 27.05、verify 46.25 → 21.80），这两段本卡**一行没改**，正是负载差异的证据；
- 结论口径：栅格化的净收益是 **1.7×–3.3×**（随弹幕密度），关卡级 0.127× → 0.722× 的实时倍率提升里还有一部分来自机器变闲。

---

## 6. 复现命令

```powershell
# 单测（Windows）
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_raster_pytest -q `
    tests/test_v400_teacher.py tests/test_v400_teacher_import.py
python -m compileall -q touhou_rl/v400/teacher tools/rl/v400_teacher.py

# 真机 tape 采集（WSL 引擎，一次一个）
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_raster_capture.py 1400"

# 离线 A/B（Windows，85 条真机 tape，新旧各跑一遍）
python .tmp_v400_raster_bench.py            # -> .tmp_v400_raster_bench.log

# 真机 A/B（WSL，同 tape 新旧对比 + 引擎 verify）
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_raster_ab.py 30"

# 关卡级（§5）
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_teacher.py \
  --decisions 4200 --max-frames 16000 --stop-on-death --stage 1 --difficulty normal --seed 1 \
  --mode full --quiet --out .tmp_v400_raster_stage1.jsonl > .tmp_v400_raster_stage1.log 2>&1"
```

产物（仓库根，`.tmp_v400_raster*` 前缀）：

| 文件 | 内容 |
|---|---|
| `.tmp_v400_raster_tapes.pkl` | 85 条真机 tape（最密的 1,400 条里的 top-80 + 5 条空 tape） |
| `.tmp_v400_raster_capture.py` | tape 采集脚本（child 录制，不移动权威影子） |
| `.tmp_v400_raster_bench.py` / `.log` | 离线 A/B：点级判死、plan 级判死、score/margin 差异、耗时分位 |
| `.tmp_v400_raster_ab.py` / `.tmp_v400_raster_ab.log` / `.tmp_v400_raster_ab.jsonl` | 真机 A/B 每决策明细 |
| `.tmp_v400_raster_stage1.jsonl` / `.log` | 关卡级每决策 JSONL |
| `.tmp_v400_raster_stage1_summary.json` | 关卡级分层统计（有弹幕/空 tape 的 search 分位、总耗时、实时倍率） |
| `.tmp_v400_raster_debug.py` / `_probe.py` / `_micro.py` | 定位 margin 离群 cell、查表耗时分解、栅格化微基准 |

退出码：单测 `0`（58 passed）、`compileall` `0`、离线 A/B `0`、真机 A/B `0`、关卡级 `0`（若停在帧上限/死亡则是「按预期停止」）。

---

## 7. 测试

```text
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_raster_pytest -q `
    tests/test_v400_teacher.py tests/test_v400_teacher_import.py
58 passed          # 50 (teacher) + 8 (import)
python -m compileall -q touhou_rl/v400/teacher tools/rl/v400_teacher.py   # exit 0
```

`tests/test_v400_teacher.py` 从 **44 增到 50**（+6），新增：

1. `test_raster_cell_index_covers_the_playfield` — 184×208、原点 (8,16)、四角 clamp 不回绕、cell 中心映射；
2. `test_raster_hit_is_a_superset_and_core_a_subset_of_the_lethal_region` — **掩码与精确路径的一致性**：`core` 的每个 cell 中心必死、`~hit` 的每个 cell 中心必活（这条一破，查表就可能凭空造死或漏死）；
3. `test_raster_death_verdict_is_exact_on_random_points` — 3,000 点 × 4 帧逐点 `array_equal(query.dead, exact.dead)`；
4. `test_raster_margin_stays_within_a_cell_of_the_exact_gap` — 合成 tape（致死物全在界内）上 `max|Δmargin| ≤ 2×CELL`；
5. `test_raster_search_keeps_every_death_verdict_of_the_exact_search` — 同一 tape 新旧 search 的 18 个 plan 的 `death_frame is None` 全一致，`|Δgeo_score| ≤ 2×CELL`；
6. `test_raster_search_is_faster_than_the_exact_search_on_a_dense_tape` — 性能回归：栅格 < 0.8 × 精确（实测约 0.4×）。

**既有测试的改动（必须说明）**：`_assert_same_search`、`test_beam_search_matches_brute_force_on_a_small_tape` 改为显式 `use_raster=False`。这三项钉的是「向量化实现 == pre-vectorisation reference 逐位相同」，改动后**断言一条没放松**（仍是逐字段 `==`），只是把被比对的路径写明白；栅格路径的等价性由上面第 3–5 项覆盖。**没有为了变绿而放宽任何既有断言**。

---

## 8. 剩余瓶颈与未达标说明

**性能门未过**：有弹幕时 search **75.03 ms 均值 / 74.49 ms p50**（关卡级 4,000 决策），离 60 ms 差 24%；离线同 tape A/B 是 78.5 ms p50（30.1 个/帧）。分解与下一步：

1. **建表 33.6 ms** 是最大单项（`paint` 13.5 + `distance` 11.9 + `pack` 8.2）。三条可行路线：
   - 用 WSL venv 里**已存在的 numba**（0.68.1，不算新依赖）把 `paint` 与距离变换写成 jit kernel：能量级降到 3–5 ms。这是最直接的一条，本卡未做（时间预算用尽，且需要 numpy/numba 双路径的等价测试）；
   - 把 `paint` 的二维 cumsum 换成「逐行一维差分 + 行覆盖位」（省一次 cumsum）；
   - `distance` 的 `np.where` 两条 int32 临时数组换成 bool 索引 (row, col) 累加。
2. **查表 44.9 ms**（78.5 − 33.6）：主要是每个 slot 对 41 k 个展开点做 `flat_index`（约 10 个 numpy 算子）+ 1 次 uint16 gather，再加上 **9.98% 的点要进 `exact()` 精确回退**（≈10 万点/决策 × 30 个盒 ≈ 3 M 点-盒）。可用 int16 定点把整条 margin 链路（`minimum`、排序 key）留在整数域省掉 uint16→float64；把边界带收窄（例如再加一档 1 px 的 `core`）可把回退比例压到 ~2%；更彻底是把整个束搜索搬进 numba。
3. **margin 精度**：4 px 格导致 §3.1 的 2.55% 计划级判死漂移与 21% best-action 漂移。把 margin 格改回 2 px 会让这两项大幅下降（代价：`distance` 段 ×4 ≈ +25 ms，必须先把 §1/§2 的建表降下来）。关卡级运行不受影响的部分见下条。
4. **点级判死已经是精确的**，所以「为提速改判死依据」这条红线没有触碰：死/活仍然是 AABB 表达式（几何）与引擎 verify（唯一判决），本卡只是把几何答案换成查表 + 边界带回退。关卡级的判死结果是 `agree_alive 11,974 / geo_alive_engine_dead 26 / geo_dead_engine_alive 0`，与 EXP-P1-007 同构（那 26 个是 fork 晚 1 帧的已知偏差）。
5. 若要「确保 plan 级也逐位一致」，可行方案是「栅格预筛 + 决赛圈精确重算」：束只用于剪枝，对每 root 保留的 top-k（k=64）用 `exact()` 重新打分再选 best——成本约 36×64 点 × 30–280 boxes ≈ 3–30 M 点-盒（10–40 ms），能把本卡的排序漂移压到 0，但会吃掉大部分收益。**本卡未实现**，列为下一步。
6. **record 27.0 / verify 21.8 ms** 现在和 search 同量级（关卡级三段各占 29% / 24% / 45%）。搜索达标后，1× 实时的门槛转到这两段：EXP-P1-007 §8 已给出「逐帧 `PLAYER` → 只读第 0 行」（144→96 条命令）与协议级多帧批量两条路线。

---

## 9. 修订记录

| 日期 | 修改 | 原因 |
|---|---|---|
| 2026-11-09 | 初版 | 致死掩码栅格化 + 查表束搜索；点级判死精确、p50 3.24×，未达 60 ms 门 |
