# EXP-P1-006 教师单决策成本：搜索向量化 + 验证批量化 + 录制接口 + 还原模板池实测

- 日期 / 工作包：2026-11-09 / P1 教师性能（[11-P1](<!-- -->) §2、§5；承接 EXP-P1-003 的三段计时）
- 目标：把单决策的**算法三段**（录制 + 搜索 + 验证）从 EXP-P1-003 实测的 114.6 ms 压到 ≤66.7 ms（60 fps、1 决策 = 4 帧的 1× 实时基准），并为「在 fork 子进程里录制」预留接口；同时实测「新进程 + RESET + 重放」能否用 PARK/SPAWN 模板池降到 ≤200 ms。
- 预注册（运行前固定，全部可证伪）：
  1. 向量化束搜索必须与改造前实现**逐位一致**：同一 `HazardTape` 上 `geo_score[18]`、每个首动作的 `chosen_plan`（slots/margin/death_frame/end_xy/score/align/edge/high）、`detail`（roots/expansions）全部相等；对照 oracle 保留为 `beam_search_reference`；
  2. 验证批量化不得改变判死依据：死亡仍由引擎自己的 `deaths` 计数（`STATE f6`）决定，批量路径与逐条路径在同一模板上的 `died` / `death_frame` / `frames_advanced` / `max_position_error` 必须相同；
  3. `record(source="child")` 在当前引擎上必须 **fail-closed**（抛 `ChildHazardsUnavailable`），不得静默退回主进程录制；
  4. 模板池若不可用，必须如实记录 PARK/SPAWN 的真实语义并给出最小引擎改动；
  5. 50 决策真实运行的一致性计数（`agree_alive` / `trajectory_mismatch` / 双向不一致）不得比改造前差。
- 样本量：stage 1 normal seed 1，**50 决策**（每决策 1 次重规划，执行 2 个子决策 = 4 帧，共 200 帧），改造前后同一命令、同一二进制、同一数据目录；单测 31 项。
- 环境：`build/rl_hazard/th08-modern`（与 EXP-P1-003 同一构建，未重建），`/root/v400-data/p1s1/game`，WSL 内一次只跑一个引擎进程。

---

## 1. 优化项与收益

| # | 文件 | 改动 | 收益（单决策） |
|---|---|---|---|
| 1 | `teacher/search.py` | 束搜索向量化：36 个 `(首动作, 持续时长)` root 叠成一个数组，**每槽一次** numpy 调用（扩张 / AABB 间距 / 边界裁剪一次性做完）；每 root 的 top-`beam` 选择用「数值 key + 分区 + 精确 tie 码」代替 36 次 `lexsort`；无致死 AABB 的 tape 走等价快速路径 | search **55.93 → 18.17 ms**（隔离测量 16.9 ms） |
| 2 | `teacher/verify.py` + `teacher/engine.py` | 验证批量化：新增 `TeacherEngine.ask_many()` 把整段命令**一次写入**再按序读回；计划里连续的相同掩码合并成一条 `STEP <mask> <frames>`（采样计划多为 2–4 段），每个 run 边界读一次 `STATE`；一旦引擎计数上升，再在**同一模板的新 fork** 上逐槽复跑，精确定位死亡帧并保留逐槽位置 | verify **42.94 → 20.21 ms**（隔离测量 16.0 ms） |
| 3 | `teacher/hazard_tape.py` | `record(..., source="main"|"child", child_callable=...)`；`source="child"` 时经可注入的 `child_callable(frame)->hazard_dict` 取数，**主进程一步不动**；能力缺失立刻抛 `ChildHazardsUnavailable`（不看 callable、不退回主进程） | 接口就绪，当前引擎 fail-closed（见 §3） |
| 4 | `teacher/engine.py` | `probe_child_hazards()`：真 fork 一次、真发 `CHILD <h> HAZARDS 16`，把引擎的拒绝原文作为 `reason` 返回并缓存；`probe_capabilities()` 同步记录 | 上层可据此自动切换 |
| 5 | `teacher/pool.py` | `RecorderPool` / `RestoreBudget`：把还原拆成「进程启动 + RESET（固定）」与「重放（可池化）」，`acquire()` 在引擎不支持时抛 `ChildHazardsUnavailable`，`status()` 给出实测预算 | 见 §3：当前引擎下池化收益为 0 |
| 6 | `tests/test_v400_teacher.py` | 新增 11 项（搜索等价 3、验证等价 3、child 录制 4、模板池 1，含把旧实现当 oracle 的对照） | 31 passed |

### 1.1 搜索向量化为什么能保持逐位一致

旧实现对每个 root 逐槽做：`moved = pos + delta`、`gap0/gap1`、`gaps = min(gap0, gap1)`、`margins = min(prev, gaps)`、`died_now = gaps <= 0`、`successor = moved + delta`、`order = lexsort((action, -margins, died_now))[:beam]`。
新实现保留**每一个逐元素运算的先后顺序**（`center + 2.0*step` 仍只加一次、帧位置仍在 `+step` 后裁剪），只把「对 36 个 root 分别做」改成「对叠起来的数组做一次」；选择把字典序

```text
(died_now, -margin, action, row)
```

编码成数值 key：存活节点 `key = -margin`，本轮死亡节点 `key = 1016 - margin`（`margin` 被 `_frame_margin` 截到 `MARGIN_CAP=16`，所以任何死亡节点的 key 都大于任何存活节点），同 key 的先后用 `tie = action * n + row` 决定。每 block 用 `np.partition` 找第 `beam` 个 key，边界上的同 key 行按 `tie` 取前若干行，再对选中的 2304 行做一次小 `lexsort` —— 与旧实现的稳定 `lexsort` 结果完全相同。`died_now` 用的是**本轮** `gaps <= 0`，不是累计死亡帧（旧实现如此，这处怪癖被原样保留）。

`_frame_margin` 仍在原地保留给 `simulate_plan`（根节点与测试用），新批量路径用 `_margin_rows`：同一个表达式、同样的减法顺序，只按 1024 行分块以限制临时数组大小（分块不改变任何元素的值）。

### 1.2 验证批量化为什么仍是引擎判死

- 死亡判定原封不动：`STATE f6`（`deaths`）相对**rollout 开始前**的读数上升即为死亡；raise 的那一帧仍按 `2*slot+1` 归属；
- 批量路径为了少往返，只在**掩码变化处**（run 边界）读 `STATE`。一旦发现计数上升，就在同一 `PARK` 模板的**新 fork** 上按「每槽 STEP 2 + STATE」复跑一遍，死亡帧与逐槽位置由这次复跑给出——仍是引擎自己的计数，不是几何推断；
- 顺手修掉一个真实缺陷：旧实现的基线是「第一步之后」的第一次读数，于是**第 0 槽就死亡且之后不再增长**的计划会被判活。新实现把基线改成 fork 之后的第一次读数（`CHILD h STATE`），并有单测 `test_batched_verification_detects_a_slot_zero_death` 覆盖。本样本 150/150 个计划无一死亡，所以一致性计数不变。

---

## 2. 等价性证据

### 2.1 搜索：与改造前实现逐位一致

oracle = `beam_search_reference`（改造前的实现，原样保留，不参与决策）。对比字段：`scores`（`np.array_equal`）、每个首动作的 `plans[cand]` 全字段、`detail`。

| 样本 | 内容 | 结果 |
|---|---|---|
| 真实 tape | stage 1 seed 1 第 0 决策，48 帧，**48/48 帧 0 个致死 AABB** | 逐位一致 |
| 稠密 tape ×6 | 64 具 61 个致死 AABB 的合成 tape（子弹 2.6–8.0 宽 + 敌人 body） | 6/6 逐位一致 |
| 混合 tape | 前半空、后半稠密 | 逐位一致 |
| 整条车道墙 tape | 200 px 宽子弹覆盖全道，全部计划判死 | 逐位一致 |
| `beam=400`（不可剪枝） | 全展开对全展开 | 逐位一致 |
| FAST 配置（H=32/beam=24）、H=8/beam=4 | 参数路径差异 | 各自一致 |
| 8 条随机 tape | 随机 horizon {4,8,48}、起始点、0/3/20/80 个子弹、散布 5–90 px | 8/8 逐位一致 |

`detail` 的 `roots=36`、`expansions` 也逐样本相等（例如真实 tape 1,724,976 vs 1,724,976；稠密 tape 862,488 vs 862,488）。

### 2.2 验证：批量路径与逐条路径同一模板对比

真实引擎（`build/rl_hazard`）同一 `PARK` 模板上先跑 `batch=False`（逐条 `CHILD STEP`/`CHILD STATE`）、再跑 `batch=True`，3 个计划：

| 字段 | 结果 |
|---|---|
| `died` | 3/3 相同（全部 `false`） |
| `death_frame` | 3/3 相同（全部 `null`） |
| `frames_advanced` | 3/3 相同（48/48/48） |
| `max_position_error` | 3/3 相同（4.0 / 4.0 / 4.0 px，仍是 fork 晚 1 帧的已知偏差） |
| `positions` / `deaths` / `hashes` | **采样密度不同**：批量路径按 run 边界采样（每计划 2–4 个点），逐条路径每槽 1 点（24 个）；末尾采样点相同；死亡时批量路径会复跑成逐槽采样，密度也相同 |
| 耗时 | 逐条 **39.2 ms** → 批量 **16.0 ms**（同一进程、同一模板、交替测量） |

单测 `tests/test_v400_teacher.py` 里另用确定性 fake 引擎对比 `batch=True/False`，并在 `dies_at ∈ {None, 1, 5}` 三种死亡场景下断言 `died`/`death_frame`/`frames_advanced`/末点位置全部相同（无引擎依赖，可长期回归）。

### 2.3 端到端：改造前后 50 个决策的标签逐项相同

改造前（旧搜索 + 逐条验证）与改造后（向量化搜索 + 批量验证）两次 50 决策运行，逐决策比较 JSONL：

| 字段 | 结果 |
|---|---|
| `geo_score[18]` | **50/50 决策完全相同**（逐元素，含 4 位小数序列化后的值） |
| `chosen_plan`（first_action/duration/slots/margin/death_frame/end_xy/score/align/edge/high） | **50/50 相同** |
| `verified_mask` | 50/50 相同 |
| `verified`（died/death_frame/frames_advanced/max_position_error/trajectory_matches/events/end_xy） | **50/50 相同**（`max_position_error` 全部 4.0 px） |
| 每决策选中的首动作集合 | 完全相同 |

这条比单测更强：它同时覆盖「搜索逐位一致」与「批量验证判决一致」，且是在真实引擎上、真实 50 个决策点上的端到端比较。

---

## 3. 还原路径：模板池实测（结论：当前引擎下不可用，收益 0）

### 3.1 PARK/SPAWN 的真实语义（读源码 + 真机）

- `PARK`：父进程 `fork()` 出一个模板进程，模板进程只跑 `TemplateLoop`（`fork.cpp:264`）；父进程继续前进；再次 `PARK` 会杀掉上一个模板（`ReleaseTemplate`）。
- `SPAWN <t>`：模板再 `fork()` 出一个孙进程，孙进程重进命令循环、走私有管道（`fork.cpp:188`）。
- **模板是单向快照**：没有任何协议命令能把模板状态「装回」父进程（没有 `RESTORE`/`SWAP`）。所以把决策点 PARK 下来并不能让下一次决策的主进程回到决策点。
- **孙进程是 child**：在 `build/rl_hazard` 上它只答 `HELLO/HASH/QUIT/DEBUG/FRAME/PING/STATE/STEP`，`HAZARDS` 被 `ERR child-command` 拒绝（真机探针原文见 §3.2）。孙进程能**走到**下一个决策点（实测：从模板 SPAWN 后 `STEP` 2×2 帧，`STATE` 报 `px=8.0 py=384.0 frame=204`），但**不能录制**。

### 3.2 还原成本拆分（真实进程，50 决策同一二进制）

| 分段 | 实测 | 模板池能否省掉 |
|---|---|---|
| 进程启动 + `HELLO` | **1180.8 ms** | 否（新录制进程仍需 RESET 一次，见下） |
| `RESET`（每进程一次） | **415.3 ms** | 否 |
| 重放历史（200 帧 = 100 决策槽） | **19.9 ms** | **是**（池化后可在 warm 进程上重放） |
| `PARK` + 每计划 `SPAWN` | 0.6 ms/次 | 已经用于验证 |
| 合计 | **1616.0 ms** | 可池化部分仅 19.9 ms（1.2%） |

结论：**「新进程 + RESET + 重放」的 1.6 s 地板是进程启动 + RESET**，不是重放；而当前引擎里唯一能录制的进程必须是「自己 RESET 过的主进程」。因此在本机上按「一次只跑一个引擎」的约束，模板池对单决策还原的净收益为 **0**（重放的 3.97 ms/2 决策可以拿掉，但必须顺序地再做一次 1.6 s 的启动+RESET，总时长不变）。`RecorderPool.status()` 真机返回 `available=false`（`reason` 为 `'CHILD 1 HAZARDS 16' -> ERR child-command`），`acquire()` 直接抛 `ChildHazardsUnavailable`，不会退化成「静默再跑一遍主进程」。

### 3.3 最小引擎改动建议（按收益排序）

1. **`CHILD <h> HAZARDS` / `CHILD <h> PLAYER` 可读**（另一路已在做的就是这条）：孙进程能录制后，录制改为 `PARK`(决策点) + 每帧 `SPAWN/STEP/HAZARDS`（或一条子进程内的多帧录制命令），主进程完全不动 → **还原归零、录制也不用 48 次主进程往返**；本卡的 `record(source="child")` + `probe_child_hazards()` 就是这个接口。
2. **让 fork 出的子进程能 `PARK`/`SPAWN`**（或一条 `CHILD <h> PARK` + 子进程内的 `SPAWN`）：这样一次 RESET 之后整局都在子进程链里推进，`RESET` 每进程一次的限制彻底消失。
3. 若短期不做 1/2：把控制通道做成「一次写多条命令、按序读回」的**协议级**批量（本卡在 Python 侧用 `ask_many` 已经这么用，但每条命令仍要过一次父进程的 poll/forward，144 条命令仍有 ~16 ms）。

---

## 4. 真实 50 决策：改造前后对照

命令（两侧完全相同，仅 `--out` 不同）：

```powershell
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_teacher.py \
  --decisions 50 --stage 1 --difficulty 1 --seed 1 --mode full --quiet --out <out>.jsonl"
```

| 指标（每决策均值） | 改造前 | 改造后 | 变化 |
|---|---|---|---|
| 录制 | 13.21 ms | **13.15 ms** | 持平（本卡未改录制路径） |
| 搜索 | 55.93 ms | **18.17 ms** | **3.08× 更快** |
| 验证 | 42.94 ms | **20.21 ms** | **2.12× 更快** |
| **算法三段（录制+搜索+验证）** | **112.1 ms** | **51.53 ms** | **2.18× 更快** |
| 还原（harness 成本） | 1615.44 ms | 1615.79 ms | 不变（模板池不可用，见 §3） |
| 合计 | 1727.5 ms | 1667.3 ms | 1.04× |
| 决策/秒 | 0.5789 | 0.5998 | +3.6% |
| 实时倍率（含还原） | 0.0193× | 0.0200× | +3.6% |
| 实时倍率（仅算法三段） | 0.595× | **1.294×** | **≥1× 达标** |
| 决策数 / 状态 | 50/50 `ok` | 50/50 `ok` | 一致 |
| 死亡 / 存活帧 | 0 / 200 | 0 / 200 | 一致 |
| `agree_alive` | 150/150 | **150/150** | 不劣化 |
| `agree_dead` | 0 | 0 | 一致 |
| `geo_dead_engine_alive` / `geo_alive_engine_dead` | 0 / 0 | **0 / 0** | 不劣化 |
| `trajectory_mismatch` | 150/150 | **150/150** | 不劣化（4.0 px，仍是 fork 晚 1 帧的已知偏差） |
| 每决策选中的首动作集合 | {0,2,3,6,7,11,12,15} | {0,2,3,6,7,11,12,15} | 完全相同 |
| tape 致死 AABB/帧（min/max/mean） | 未记录 | **0 / 0 / 0**（48/48 帧全空，50 个决策全部如此） | 见 §6 样本说明 |
| 引擎进程数 / 协议调用次数 | 50 / 17409 | 50 / 11331 | 单决策命令数少 34.9% |

退出码：改造前 `0`、改造后 `0`。两侧 `probe` 一致：`CHILD 1 HAZARDS` → `ERR child-command`（改造后的报告里同时带结构化探针 `child_hazards: {available: false, reason: ...}`）。

### 4.1 单段实测（同一进程内交替测量，用于拆解收益）

| 对比 | 旧 | 新 |
|---|---|---|
| 搜索（真实决策点 tape） | 54.5–56.0 ms | **16.9–18.2 ms** |
| 验证（同一 `PARK` 模板，3 计划，逐条 vs 批量） | 39.2 ms | **16.0 ms** |
| 搜索（稠密合成 tape，61 个致死 AABB/帧 × 48 帧） | 450.0 ms | **421.3 ms**（1.07×） |
| 搜索（空 tape，`MARGIN_CAP` 快速路径） | 54.5 ms | **16.9 ms**（3.24×） |

> verify 在 50 决策的真实运行里是 20.21 ms（比单次对比的 16.0 ms 高），因为 `decide()` 的验证段还包含 `PARK`、3 次 `SPAWN`、逐计划的 `simulate_plan` + `_modelled_positions` 与运行间抖动；单次对比只测 `verify_plans`。

---

## 5. 是否达到 1× 实时

- **算法三段：达到。** 51.53 ms/决策 < **66.7 ms**（60 fps、1 决策 = 4 帧）→ **1.294× 实时**，决策率 19.4 决策/秒。目标是 ≤66.7 ms，实测余量 15.2 ms。
- **含还原：未达到。** 1667.3 ms/决策 → **0.0200× 实时**。差距全部来自 §3 的 1.6 s「新进程 + RESET」，与算法三段无关（还原占单决策的 96.9%）。
- 口径说明：`realtime_multiple` 是 CLI 的 `decisions_per_second * decision_frames / 60`；仅算法三段的口径用 `(record+search+verify)` 作分母，与 EXP-P1-003 §4.2 的「算法三段 0.58×」直接可比。

## 6. 剩余瓶颈与引擎侧建议

1. **还原 1615.8 ms（占 96.9%）** —— 进程启动 1180.8 ms + `RESET` 415.3 ms。这是 harness 成本，不是算法成本；在不改引擎的前提下无法用 PARK/SPAWN 去掉（§3.1：模板是单向快照，孙进程不能 `HAZARDS`）。**最小引擎改动：`CHILD <h> HAZARDS`/`PLAYER` 可读**（本卡已把 Python 侧接口与 fail-closed 探针准备好），届时录制改为「子进程推进 + 子进程录制」，主进程不动 → 还原归零。
2. **录制 13.15 ms** —— 48 帧 × (`HAZARDS` + `PLAYER` + `STEP`) + 初始 `HASH`，共 145 条一命令一往返。Python 侧还能再压：`ask_many()` 已经能把 48 帧的命令一次写入（同样的命令序列、同样的顺序），本卡没有把它接到 `record()` 上，以免在测量窗口里同时改两条路径；这是下一步最便宜的 ~8–10 ms。
3. **验证 16–20 ms** —— 144 条子命令里大多数时间仍是父进程的 poll/forward（每条 ~0.11 ms）。Python 侧已经把往返拼成一次写；再往下需要协议级批量（一条命令带多帧/多槽）或让子进程自己跑完整段掩码序列并回传死亡帧与位置。
4. **样本警告**：本卡 50 个决策的 tape **全部 48/48 帧 0 个致死 AABB**（stage 1 normal 的前 200 帧没有可致死弹幕），所以搜索的 `MARGIN_CAP` 快速路径是**实测工作负载**，不是人为构造；稠密合成 tape（61 个致死 AABB/帧）的对照说明一般情形只是 1.07× 而不是 3.24×，`_margin_rows` 的分块（1024 行）就是为此调的。下一张卡应在 stage 4–7 采样，那里才有真实弹幕与死亡。
5. `trajectory_mismatch` 仍是 150/150：fork 晚 1 帧应用掩码的已知偏差没有修（本卡不改判死依据，也不改验证的语义）；要修需要在验证前补 1 帧原语，或从 d−1 帧 `PARK`。
6. 决死 bomb 仍不可用（`STATE` 无 `frameEventFlags`），未在本次改动范围内。

---

## 7. 测试

```text
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_tperf_pytest -q \
    tests/test_v400_teacher.py tests/test_v400_teacher_import.py
39 passed in 1.18s           # 31 (teacher) + 8 (import, EXP-P1-005)
python -m compileall -q touhou_rl/v400/teacher   # exit 0
```

其中原有 20 项**全部保持通过**（无改动、无放宽），`tests/test_v400_teacher.py` 从 20 增到 31；另有 EXP-P1-005 的 8 项导入回归同样通过。覆盖：原有 20 项（半宽 / 含等号 / 敌人 body / 边界 / 1 帧延迟 / §3 逐项 / 束搜索对暴力 / label 往返 / 未验证 −5 / `HAZARDS` fail-closed / `Consistency` 双向 / §5 参数）；新增 11 项：

1. `test_vectorised_search_matches_the_reference_on_dense_and_empty_tapes`（稠密/全空/混合/`beam=400` 四种 tape，逐位一致）；
2. `test_vectorised_search_finds_unknown_closed_form_on_a_wall`（全道墙，两侧都判死且一致）；
3. `test_batched_verification_matches_the_per_command_judgement`（fake 引擎，`dies_at ∈ {None,1,5}`）；
4. `test_batched_verification_detects_a_slot_zero_death`（基线修复）；
5. `test_record_child_source_fails_closed_without_the_capability`（探针拒绝时 callable 不被调用、主进程 0 步）；
6. `test_record_child_source_requires_a_callable_and_is_a_bad_source_value`；
7. `test_record_child_source_walks_the_callable_and_leaves_the_main_process_alone`（happy path，主进程 0 步）；
8. `test_record_child_source_wraps_a_failing_callable`；
9. `test_reference_selection_is_used_when_aliasing_cannot_be_avoided`（NaN 兜底与 `lexsort` 一致）；
10. `test_recorder_pool_reports_the_engine_capability_and_fails_closed`；
11. `test_restore_budget_separates_fixed_from_poolable_cost`。

---

## 8. 复现命令与产物

```powershell
# 单测（Windows 侧）
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_tperf_pytest -q tests/test_v400_teacher.py
python -m compileall -q touhou_rl/v400/teacher

# 真实 50 决策（WSL 侧，一次只起一个引擎）
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_teacher.py \
  --decisions 50 --stage 1 --difficulty 1 --seed 1 --mode full --quiet --out .tmp_v400_tperf_after.jsonl"

# 离线等价性与计时（真实 tape + 合成 tape，不占引擎）
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_tperf_bench.py .tmp_v400_tperf_tape.pkl"

# 真机：验证批量等价、child 探针、还原分段
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_tperf_verify.py"
```

产物：`.tmp_v400_tperf_before.jsonl/.log`（改造前 50 决策）、`.tmp_v400_tperf_after.jsonl/.log`（改造后）、`.tmp_v400_tperf_tape.pkl`（真实 tape 样本）、`.tmp_v400_tperf_verify.json`（探针/分段/批量等价原始输出）。

## 9. 修订记录

| 日期 | 修改 | 原因 |
|---|---|---|
| 2026-11-09 | 初版 | 搜索向量化 + 验证批量化 + child 录制接口 + 模板池实测 |
