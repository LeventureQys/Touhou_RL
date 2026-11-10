# EXP-P1-010 numba 建表/查表加速与「栅格预筛 + 决赛圈」漂移实验（未达 40 ms、漂移未消除）

- 日期 / 工作包：2026-11-09 / P1 教师（[11-P1](<../../11-P1-搜索教师.md>) §2.1、§3；承接 EXP-P1-008 §8 的三条下一步）
- 目标：
  1. 把 `paint` / `distance` / `pack` 写成 numba `njit(cache=True)` kernel，建表 ≤ 10 ms；
  2. 查表阶段压到 ≤ 25 ms（减少 gather 次数、量化链路留在整数域）；
  3. **漂移消除**：实现「栅格预筛 + 决赛圈精确重算」，把 EXP-P1-008 的 39/1,530 `death_frame` 漂移与 18/85 best action 漂移压到 0（或量化并定级）；
  4. 性能验证：同 tape A/B（离线 85 条）、关卡级；
  5. 回归：`tests/test_v400_teacher.py`（既有 50 项 + numba/决赛圈/性能新增）；
  6. 不伪造：任何未达标项照实写，包括**本卡未完成的门**。
- 预注册（运行前固定，可证伪）：
  1. 点级判死仍必须 **2,380,000 点 0 分歧**；
  2. 建表 ≤ 10 ms（p50，JIT 预热后稳态）、查表 ≤ 25 ms（p50）、search ≤ 40 ms（p50）、单决策 ≤ 66.7 ms（1× 实时）；
  3. 决赛圈后 best action 一致率 ≥ 95%（最好 100%）、`death_frame` 一致率同样处理；
  4. 未达标项、剩余漂移来源与等级必须写明。
- 环境：WSL `/root/v400-venv`（Python 3.10 / numpy 2.2.6 / **numba 0.68.1，实验前已在环境中，不算新依赖**）；Windows 侧 Python 3.12 / numpy 2.5.4（**无 numba**，只跑 numpy 回退路径）。引擎与协议**未改**，未重建 build。
- **负载说明**：Windows 零售 trace 批跑同时在跑，WSL 绝对耗时偏高且抖动大（同一份代码两次测量可差 20–40%）。跨实现比较一律用**同一进程内交替测量**；本卡所有数字都注明来源运行。

---

## 1. 结论先行

| 门 | 目标 | 实测 | 结论 |
|---|---|---|---|
| 建表 | ≤ 10 ms | **2.04 ms p50**（85 条 tape；numpy 30.77） | **达标**（15.1×） |
| 查表（41,472 点单次 gather） | ≤ 25 ms | **0.156 ms/次**（numpy 0.501，3.2×） | **达标** |
| search（有弹幕，p50） | ≤ 40 ms | **50.91 ms**（EXP-P1-008 同机 78.50；精确 244.51） | **未达标**（差 27%） |
| 点级判死 | 2,380,000 点 0 分歧 | **2,380,000 / 0 分歧 / 100.000000%** | **达标** |
| best action 一致率 | ≥ 95% | **87.06%**（决赛圈后 88.24%） | **未达标** |
| `death_frame` 一致率 | 100% | **97.451%**（39/1,530，决赛圈后仍 39） | **未达标、未消除** |
| 单决策 ≤ 66.7 ms / 1× 实时 | 达标 | **未测**（预算耗尽，见 §5.1） | **未验证** |

**一句话**：numba 把建表 30.77 → **2.04 ms**、查表 0.501 → **0.156 ms/次**，两项超额达标；search p50 从 78.5 只压到 **50.91 ms**，**没有到 40 ms**；「栅格预筛 + 决赛圈精确重算」已实现、已测、已进回归，但**在 85 条真机 tape 上没有消除漂移**（39/1,530 与 EXP-P1-008 完全相同），因此**默认关闭**，定级**未解决**。

---

## 2. 实现

| 文件 | 改动 |
|---|---|
| `touhou_rl/v400/teacher/raster_numba.py` | **新增**：`build_table_kernel`（`prange` over frames）、`query_kernel`、`exact_kernel`、`probe()`/`warmup()`/`available()`/`reason()` |
| `touhou_rl/v400/teacher/raster.py` | `build_table(..., backend="auto"/"numba"/"numpy")`；`_build_numpy` 保留为参考实现；`_pack_boxes`（padded `(frames, max_boxes, 4)` + counts）；`LethalTable.backend` / `flat_array()` / `packed()` / `exact_points()`；`query()` 双后端 |
| `touhou_rl/v400/teacher/search.py` | `beam_search(..., finalist_extra=None)`；`_root_nodes` 根节点改为**精确** margin；`_refine_finalists()`（决赛圈）；`_tie_orders()`；`detail` 增 `raster_backend` / `finalist_extra` / `finalist_points` / `finalist_parents` |
| `tests/test_v400_teacher.py` | +6 项（numba 建表一致、numba 查表一致、决赛圈等价、`finalist_extra=0` 保持旧口径、精确路径仍可用），性能回归明确钉在**排序路径**（`finalist_extra=0`） |

### 2.1 建表 kernel（paint + distance + pack 三合一）

`build_table_kernel` 一帧一个 `prange` 任务，**逐操作复刻** numpy 实现：

1. paint：`_cell_ranges` 的 `hit`（相交）与 `core`（内含）两档谓词，用同一 `np.ceil/np.floor` 表达式逐矩形填充（numpy 侧是二维差分 + 两次 `cumsum`，布尔结果等价）；
2. distance：`_downsample_hit` 的 any-of-4 下采样 → 逐列「到最近 true 的一维距离」双向扫描 → 沿列 `min_k (k² + vertical)`，`k <= cap_cells` 截断；整数域，结果与 numpy 逐位相同；
3. pack：`out = lookup[sq] | ((hit | core<<1) << 8)`，低字节 margin 定点，高字节 flags。

**未改判死依据**：仍是同一个 AABB 表达式、同一张表、同一个 `MARGIN_CAP`；kernel 只是把同一算式搬进编译代码。

### 2.2 查表 kernel

`query_kernel` 把「flat index → 1 次 uint16 gather → 解 flags → 边界带精确回退」走完：

```text
code = table[frame*nx*ny + iy*nx + ix]      # 一次 gather，索引算术在整数域
hit = (code>>8)&1 ; core = (code>>8)&2 ; margin = (code&0xFF)*0.25
hit==0 -> 活 ; core!=0 -> 死（margin=0）; 其余 -> AABB 精确重算（band 计数）
```

与 numpy 版本的差别只是「不再为每个算子分配 41k 的临时数组」。**bit 级一致**：85/85 条 tape 整表相同；8 万随机点 `dead` 完全一致、`margin` 最大差 **0.000000 px**。

### 2.3 「栅格预筛 + 决赛圈精确重算」

`_refine_finalists`（每个 slot 一次）：

1. 用栅格表的 margin 得到本 slot 的近似 key（死节点 `1000+16-margin`，活节点 `-margin`）；
2. `argpartition` 取每 root **`keep_width + finalist_extra`** 个最好的节点作为决赛圈（只有这些还能进束）；
3. 决赛圈节点的 gap 用**精确 AABB** 重算：`min(exact(parent@2s), exact(child@2s+1))`，`died_now` 同样重算（parent/child 去重后各一次 `exact_kernel` 调用）；
4. 后续 `_exact_selection` 与**累积 margin** 都用重算后的值。

`finalist_extra=0` 回到 EXP-P1-008 口径。**默认值：0**（原因见 §3.3）。

---

## 3. 漂移：实现了、测了、没消除

### 3.1 判死一致率（保住的红线）

| 口径 | 结果 |
|---|---|
| 点级（85 条 tape × 7 帧 × 4,000 点 = **2,380,000 点**） | **0 分歧**（误判死 0、误判活 0）→ **100.000000%** |
| numba 表 == numpy 表 | **85/85 条 tape 整表逐位相同** |
| numba `query` == numpy `query` | 8 万点 `dead` 完全一致，`margin` 最大差 **0.000000 px** |
| numba `exact_points` == numpy `exact` | 48 帧 × 2,000 点逐位相同（`dead` 0 分歧、`margin` 最大差 **0.0**） |

### 3.2 计划级 / 决策级一致率（85 条真机 tape，同进程交替测量，1,530 个计划）

| 变体 | `death_frame` None/非None 分歧 | 一致率 | best action 一致 | 一致率 | `Plan.slots` 分歧 | max\|Δscore\| | 存活点 max\|Δmargin\| |
|---|---|---|---|---|---|---|---|
| 栅格排序（EXP-P1-008 口径） | **39 / 1,530** | 97.451% | **74 / 85** | **87.06%** | 404 | 105.46 | 15.79 px |
| 栅格 + 决赛圈（extra=8） | **39 / 1,530** | 97.451% | 75 / 85 | 88.24% | 485 | 108.57 | 14.26 px |
| 栅格 + 决赛圈（extra=16） | **39 / 1,530** | 97.451% | 76 / 85 | 89.41% | 493 | 108.57 | 14.16 px |

对照 EXP-P1-008 的 18/85 best action（67/85 一致），本卡的 74/85 **改善来自把根节点 margin 从查表值改成精确值**（`_root_nodes`），不是决赛圈。

**决赛圈没有起作用**：`death_frame` 分歧条数与 EXP-P1-008 **完全相同（39）**，best action 只 74 → 76，`max|Δmargin|` 仍有 14 px（若存活节点的 margin 真被精确重算，这个数应落到 float 误差量级）。

### 3.3 定级与剩余来源

- **定级：未解决（unacceptable）**。best action 87.06%、`death_frame` 97.451%，都低于预注册门（≥95% / 100%）。
- 已排除：表本身（整表逐位相同）、点级判死（2.38 M 点 0 分歧）、numba kernel 数值（逐帧与 numpy 逐位相同）、根节点 margin（已改精确，只影响 best action 不影响 `death_frame`）。
- 决赛圈**确实改变了路径**（同 tape 上 best plan 的 `slots` 从 `[5,5,5,0,…]` 变成 `[5,5,5,0,…,5,4,…]`，`detail.finalist_points=56,376`），但计划级分歧条数不变——**这是 432 次逐 slot 选择（18 root × 24 slot）的复利效应**，任何一处不精确都会在下游放大，不是某一个 slot 算错。
- 剩余可疑来源（**未在本卡证实**，按证据强弱）：
  1. **决赛圈边界并列**：栅格 key 有大量整数并列（margin 恰 0 或 16），参考实现用 `(action, row)` tie code 决定谁进束，而 `argpartition` 在并列时取舍任意；第一版用 tie-order 视图，实测同样没改善，说明并列不是唯一原因；
  2. **`extra` 的 slack 覆盖不了 24 个 slot 的复利**：即便单 slot 的选择与参考一致率很高，24 slot 连乘后仍会走岔；
  3. 参考实现 `(died_now, -margin, action, row)` 全序里还有与 margin 无关的分支（死/活互换），需逐 root 打印才能确认。
- 工程处置：**默认关闭**（`finalist_extra=None` → 0），代码与测试保留。

---

## 4. 耗时（85 条真机 tape，WSL，交替测量）

### 4.1 建表 / 查表

| 项 | p50 | p90 | max | mean |
|---|---|---|---|---|
| 建表 numpy（EXP-P1-008） | 30.77 | 31.72 | 33.66 | 28.87 ms |
| 建表 **numba** | **2.04** | 3.27 | 9.90 | 2.41 ms |
| 加速 | **15.1×** | | | |

单次 gather（41,472 点，同一张表）：numpy **0.501 ms** → numba **0.156 ms**（3.2×）。`exact_points`（2,880 点）：numpy 0.786 → numba **0.082 ms**（9.6×）。JIT 预热（`probe()`/`warmup()`）不计入；冷启动一次约 2–6 s，`cache=True` 后同机后续进程读缓存不再编译。

### 4.2 search 分位（有弹幕）

| 实现 | p50 | p90 | max | mean |
|---|---|---|---|---|
| 精确（`use_raster=False`） | 244.51 | 268.87 | 286.82 | 220.87 |
| 栅格（numpy 建表/查表，EXP-P1-008 原值） | 78.50 | 137.27 | 200.22 | — |
| **栅格 + numba（本卡，`finalist_extra=0`）** | **50.91** | 66.04 | 75.66 | 51.00 |
| 栅格 + numba + 决赛圈(8) | 60.55 | 75.39 | 93.32 | 59.79 |
| 栅格 + numba + 决赛圈(16) | 61.41 | 74.85 | 80.58 | 59.84 |

**74.5 → 50.91 ms（−32%）**；对 EXP-P1-008 的 78.50 是 1.54×，对精确路径 4.8×。**没有达到 40 ms**。

### 4.3 剩余耗时分解（同 tape，包装器测量，含负载）

- `finalist_extra=0`：总计 60.97 ms = `_exact_selection` **17.50** + 建表 **14.40** + query **6.66** + `_root_nodes` 1.88 + finish 0.65 + 其余（`moved`/`clip`/`minimum`/`keys`/gather/写回）**18.15**。
- `finalist_extra=8`：总计 75.15 ms = refine **20.62** + 建表 15.19 + select 11.56 + query 5.55 + exact_pts 4.62 + 其余 16.69。

**建表与查表都已不是瓶颈**（2 ms + 6.7 ms）。40 ms 的门现在卡在 **numpy 束搜索机器本身**：`_exact_selection` 的 partition/lexsort/布尔累积（≈17 ms）加上 `moved`/`clip`/`minimum`/gather 的 41k 元素算子链（≈18 ms），每 slot 各 ~0.7 ms × 24 slot × 2 段 ≈ 40 ms，与实测吻合。要过 40 ms 必须把整个 slot 机器搬进 numba（下一张卡）。

---

## 5. 未完成项与阻塞（照实写）

1. **单决策耗时 / 实时倍率 / 关卡级 1,000 决策：未测**。70 分钟预算在 numba 调试、85 条 A/B 与漂移诊断上用尽；真机 A/B（30 决策）与关卡级运行需独占 WSL 引擎（一次一个）且各自 3–20 分钟，本卡**没有跑**。因此「单决策 ≤ 66.7 ms / 1× 实时」**未验证**，不能声称达标。
2. **40 ms 性能门未过**：50.91 ms p50（差 27%），缺口在 numpy 束搜索机器（§4.3），不在本卡改动的两段。
3. **漂移未消除**：见 §3.3，定级 unacceptable，默认关闭决赛圈。
4. **`prange` 在查表 kernel 上比串行慢 17 倍**（41k 点 8.0 ms vs 0.156 ms），所以 `query_kernel`/`exact_kernel` 用串行 `njit`，只有建表 kernel 用 `prange`。这是实测选择，不是笔误。
5. **建表耗时随机器负载抖动**：85 条 A/B 的报数口径是 **2.04 ms p50**（安静时段）；同一份代码在负载高的时段用包装器测得 14–16 ms（§4.3 即该时段）。两个数都写出来，不取最好看的一个。
6. 一次操作事故并已修复：用 PowerShell `Set-Content` 回写 `tests/test_v400_teacher.py` 时把 UTF-8 中文注释二次编码成乱码（Python 仍能解析、测试仍绿）；已按「读 UTF-8 → 以 GBK 反向还原」修复，并用 `pytest` + `compileall` 复核（58 passed / 2 skipped，exit 0）。源码语义与断言未变。

---

## 6. 复现命令与产物

```powershell
# 单测（Windows，numpy 回退路径；2 项 numba 测试 skip）
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_numba_pytest_win -q tests/test_v400_teacher.py
python -m compileall -q touhou_rl/v400/teacher tests/test_v400_teacher.py
# 58 passed, 2 skipped / compileall exit 0

# 单测（WSL，numba 路径；numba 一致性测试真正执行）
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && export PYTHONPATH=/mnt/d/WorkShop/Touhou_RL && \
  /root/v400-venv/bin/python -m pytest -p no:cacheprovider --basetemp .tmp_v400_numba_pytest_wsl -q \
  tests/test_v400_teacher.py tests/test_v400_teacher_import.py"

# 85 条真机 tape A/B（点级判死 + 建表/搜索耗时 + 一致率）
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_numba_ab.py > .tmp_v400_numba_ab.log 2>&1"

# 微基准 / 阶段分解 / 建表线程稳定性 / 逐帧 exact 正确性
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_numba_micro.py"
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_numba_phase.py"
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_numba_build_t.py"
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_numba_frame_check.py"
```

| 文件 | 内容 |
|---|---|
| `.tmp_v400_numba_ab.py` / `.tmp_v400_numba_ab.log` | 85 条 tape：2,380,000 点点级判死、整表一致性、建表/搜索分位、计划级与决策级一致率 |
| `.tmp_v400_numba_micro.py` | 单次 gather / `exact_points` / 建表 / 三变体 search 交替计时 |
| `.tmp_v400_numba_phase.py` | monkey-patch 包装器做 search 阶段分解 |
| `.tmp_v400_numba_build_t.py` | 建表 p50 与预热/线程稳定性 |
| `.tmp_v400_numba_frame_check.py` | `exact_kernel` 逐帧正确性 + 决赛圈是否改变路径 |
| `.tmp_v400_numba_check.py` | 首轮冒烟（表一致、点级一致、初版耗时） |

退出码：Windows 单测 `0`（58 passed / 2 skipped）、`compileall` `0`、85 条 A/B `0`、微基准/阶段/建表/逐帧脚本 `0`、WSL 单测 `0`（68 passed）。

---

## 7. 测试

```text
Windows（numpy 回退路径）   58 passed, 2 skipped   exit 0
WSL（numba 路径）           68 passed, 0 skipped    exit 0   (60 teacher + 8 import)
```

`tests/test_v400_teacher.py` 从 **50 增到 58**（+6 新测试；2 项 numba 测试在无 numba 环境 skip），新增：

1. `test_raster_numba_build_is_identical_to_the_numpy_build` — 两后端整表 `np.array_equal`；
2. `test_raster_numba_lookup_matches_the_numpy_lookup` — `query` / `exact_points` 与 numpy 表达式逐点相同（numba 不可用则 skip）；
3. `test_finalist_stage_matches_the_exact_search` — 决赛圈版本的 `death_frame`、best action 必须与精确搜索相同，`|Δscore| <= 2×CELL`，且 `detail.finalist_points > 0`（证明阶段真的跑了）；
4. `test_finalist_stage_is_off_at_zero_and_keeps_the_table_verdict` — `finalist_extra=0` 不产生决赛圈字段，判死仍与精确路径一致；
5. `test_beam_search_without_raster_still_runs` — `use_raster=False` 精确路径仍与 `beam_search_reference` 逐位相同；
6. 性能回归测试改为钉**排序路径**（`finalist_extra=0`），保留 `raster < 0.8 × exact` 断言。

**既有断言一条没有放宽**：只把性能回归测试的比对目标写成 `finalist_extra=0`（决赛圈开销单独测，见 §4.2/§4.3）。

---

## 8. 修订记录

| 日期 | 修改 | 原因 |
|---|---|---|
| 2026-11-09 | 初版 | numba 建表 30.77→2.04 ms、查表 0.501→0.156 ms/次；search 78.50→50.91 ms（未达 40）；决赛圈实现但**未消除漂移**（39/1,530、11/85），默认关闭；真机 A/B 与关卡级未测 |
