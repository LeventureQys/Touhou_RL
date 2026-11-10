# EXP-P1-003 教师核心：单决策闭环 + 三段计时（`touhou_rl/v400/teacher/`）

- 日期 / 工作包：2026-11-09 / P1-WP1 + P1-WP2 的首个切片（[11-P1](../11-P1-搜索教师.md) §2、§3、§4、§5）
- 假设：把「录制 → 几何束搜索 → 引擎 fork 验证」做成**一个决策的闭环**，每一项耗时单独计量，几何分数与引擎判决分开记录；`BRANCH`/`PARK`/`SPAWN` 的**真实能力**先探测再使用，缺什么就如实报告，不猜。
- 预注册（运行前固定）：
  1. 单决策能跑通并把录制 / 搜索 / 验证三段耗时分开报出；
  2. `geo_score[18]` 与每段最优续接计划可复现（同 seed 同 HASH）；
  3. 几何判死与引擎判决的**一致性计数**双向都报（`geo_dead_engine_alive` / `geo_alive_engine_dead`）；
  4. `frameEventFlags` 若未导出，`bomb_death_available=false` 且不伪造决死 bomb；
  5. 单测覆盖半宽 / 含等号 / 边界 / 1 帧延迟 / §3 逐项 / 束搜索对暴力 / label 序列化 / fail-closed。
- 样本量：stage 1 normal seed 1，**100 个决策**（每决策 = 1 次重规划，执行 R=2 个子决策 = 4 帧，共 400 帧）；单测 20 项。

---

## 1. 实现要点（模块清单）

| 文件 | 内容 |
|---|---|
| `touhou_rl/v400/teacher/config.py` | §3 权重（base 100、margin 上限 16、align×4、items×2 上限 1、edge −1、high −3、死亡 `death_frame/H×10`、未验证 −5）；§5 双模式（full H=48/R=2/beam=64/k=3/迭代≤3；fast H=32/beam=24/k=1/迭代 1） |
| `motion.py` | 精确运动模型：四个速度**从 `PLAYER` 读**（team 0 = 4.0/2.8284/2.0/1.4142），斜向两轴都用斜向速度，边界 x∈[8,376] y∈[16,432]，1 帧输入延迟（第 t 帧掩码 → 影响第 t+1 帧位置） |
| `hazard_tape.py` | `record()` + `HazardTape`：逐帧 `HAZARDS`+`PLAYER`（第 0 行在第一次 `STEP` **之前**采）；子弹 `collisionSize` 保持全宽，敌人 `worldPosition`+原始 `hitboxDimensions`+`flags1`，激光 state/timer/startOffset/end/width/angle/pos；`HAZARDS` 被拒时 `HazardExportUnavailable`（fail-closed） |
| `search.py` | 束搜索：18 首动作 × 持续时间 {1,2} 槽位为根，`beam` 宽，逐槽 18 扩张；§3 评分逐项；`geo_score[18]` + 每个首动作最优续接 |
| `verify.py` | 一次 `PARK` + 每计划一次 `SPAWN`，逐槽 `CHILD <h> STEP <mask> 2` + `CHILD <h> STATE`；死亡由引擎 `deaths` 计数决定；位置与模型逐帧比对（容差 1e-3 px）；`Consistency` 双向计数 |
| `label.py` | `TeacherLabel(scores[18], bomb, chosen_plan, margin, verified_mask)` + JSON 序列化 + 已验证/未验证分数改写（未验证 −5） |
| `decide.py` | 单决策编排（录制 → 搜索 → 验证 → 选择 → bomb 规则 → label），三段计时 |
| `engine.py` | shadow 进程封装：`RESET/STEP/HASH/HAZARDS/PLAYER/STATE/PARK/SPAWN/CHILD` + **能力探针** |
| `tools/rl/v400_teacher.py` | CLI：`--decisions N --stage S --difficulty D --seed X --mode full\|fast`，每步打印 `geo_score`/验证/`chosen_plan`/`margin`/分段耗时，末尾出总报告 |

### 1.1 公式落地（§3 逐项，`search.score_plan`）

```text
存活：100 + min(margin,16) + 4·align + 2·min(items,1) − 1·edge − 3·high
死亡：death_frame / H × 10
未验证的首动作：几何分 − 5（verified_mask 标明哪些是引擎验证过的）
```

- `margin` 逐帧取「自机 hurtbox AABB 到最近致死物 AABB 的欧氏间距」，对地平线取最小；按 §3 只在 16 处饱和，所以计算时每帧也按 16 截断（不改变评分，也不改变碰撞判定）。
- `align = 1 − |末端 x − 目标 x| / 192`，截到 [0,1]；目标是第 0 帧最近的**带 collision 位**敌人 `worldPosition.x`（`boss_slot` 未导出），没有敌人时取自机 x（该项中性）。
- `edge`：末端距 x 边界 < 24 px；`high`：末端 y < 200。
- `items`：本切片**恒为 0**（`ITEMS` 未录入 tape，`config.record_items=False`）。因为对所有计划都是常数，不影响排序；已在 label 的 `extra.items_recorded` 标注，不假装有值。

### 1.2 bomb 规则（§4）

```text
规则 1（主动 bomb）：所有验证计划都死 且 bombs > 0 → bomb=1        ← 已实现
规则 2（决死 bomb）：frameEventFlags & 0x2 (PLAYER_HIT) → bomb=1    ← 不可用
```

`STATE` 只导出 `f0..f24`，**没有 `frameEventFlags`**（本次探针逐字记录：`state_fields` 里不存在该字段）。所以 `bomb_death_available=false` 写进每个 label，并在事件里记 `death_bomb_unavailable_no_frame_event_flags`。**没有用 deaths 上升沿之类的相关量去替代它**——那会是伪造。

---

## 2. 引擎能力探针（真机，运行前实测，决定了整个架构）

`python tools/rl/v400_teacher.py --probe-only`（真实 shadow 进程，`build/rl_hazard` SHA `8053dfaa…`）：

| 探针 | 结果 | 含义 |
|---|---|---|
| `HELLO` | `role=shadow proto=1` | 影子进程可跑 |
| `HAZARDS 1024` | OK，`nb/nl/ne` + 三族行 | 批量致死物导出可用（limit 1024 被接受） |
| `BRANCH 0 1` | `branch=1 pid=… parent_frame=… advanced=1 fork_us=… child_us=… hash=… trace=…` | **只有逐帧 25 字段 HASH，没有任何致死物几何** |
| `CHILD <h> HAZARDS` | `ERR child-command` | fork 出来的分支**答不了** `HAZARDS`（父侧白名单只放 HELLO/HASH/QUIT/DEBUG/FRAME/PING/STATE + STEP） |
| `CHILD <h> STATE` / `STEP` | OK | 分支能推进、能读 25 字段 |
| `STATE` 字段 | `f0..f24` + `frames` | **无 frameEventFlags** |
| `RESET`（同进程第二次） | 之后 `HAZARDS` → `ERR hazards-not-ready reset-required` | `g_resetReady = ready && globals`，而 `ready` 需要标题画面消费 `ArmStart`；**RESET 每进程只能成功一次** |

**结论**：[11-P1 §2](../11-P1-搜索教师.md) 第 1 步「从 S 分支一条推演并逐帧记录致死物」在**当前 harness 上无法实现**——不是写法问题，是 `BRANCH` 的返回载荷里就没有几何，而能 fork 的进程又答不了 `HAZARDS`。

---

## 3. 因此采用的闭环（以及它的代价）

单决策：

```text
1) 录制：让权威 shadow 进程按「上一轮计划剩余 + 不动(聚焦)」前进 H 帧，逐帧取 HAZARDS+PLAYER
2) 搜索：在 tape 上做束搜索 → geo_score[18]
3) 验证：PARK 一次 + 每计划 SPAWN 一次 + 逐槽 CHILD STEP/STATE
4) 还原：RESET（每进程只有一次，见 §2）+ 重放已提交掩码 → 回到决策点
```

因为 (1) 必须移动权威进程、(4) 的 `RESET` 每进程只能用一次，本切片用**每决策一个新进程 + 确定性重放**来实现还原（等价于"检查点即重放"）。这一段是 harness 成本，不是算法成本，所以单独计量、单独报告。

---

## 4. 真实 100 决策（stage 1 normal seed 1，full 模式）

```text
python tools/rl/v400_teacher.py --decisions 100 --stage 1 --difficulty 1 --seed 1 --mode full
```

| 指标 | 数值 |
|---|---|
| 决策数 / 请求 | 100 / 100（`status=ok`） |
| 存活帧数 | **400 帧**（6.667 s 游戏时间） |
| 死亡数 | 0 |
| 引擎进程数 | 100（每决策一个，见 §3） |
| 引擎协议调用总数 | 39,809 次 |
| 引擎侧总耗时（协议往返） | 167.6 s |

### 4.1 三段耗时（每决策平均）

| 段 | 均值 | 说明 |
|---|---|---|
| 录制 | **13.61 ms** | 48 帧 × (STEP+HAZARDS+PLAYER) |
| 搜索 | **57.39 ms** | 18 首动作 × 2 持续时间 × beam 64，24 槽 |
| 验证 | **43.55 ms** | 3 个计划 × 48 帧（PARK 0.8 ms + SPAWN 0.6 ms/个，PARK 计入本段） |
| 还原（harness 成本） | **1629.61 ms** | 新进程启动 + RESET（约 0.4 s）+ 重放历史 |
| 合计 | **1744.15 ms** | |

### 4.2 速度

| 口径 | 数值 |
|---|---|
| 决策/秒 | 0.5733 |
| 游戏秒/墙钟秒（含还原） | **0.0191× 实时** |
| 算法三段（录制+搜索+验证） | 114.6 ms / 决策 → **0.58× 实时** |
| 实时基准 | 60 fps，1 决策 = 4 帧 = 66.7 ms |

### 4.3 几何 vs 引擎一致性

| 计数 | 数值 |
|---|---|
| 验证计划总数 | 300（100 决策 × k=3） |
| `agree_alive`（几何活 ∧ 引擎活） | **300** |
| `agree_dead` | 0 |
| `geo_dead_engine_alive`（几何判死但引擎存活） | **0** |
| `geo_alive_engine_dead`（几何判活但引擎判死） | **0** |
| 轨迹不一致（位置误差 > 1e-3 px） | 300 / 300（最大 6.83 px，原因见 §5.3） |

`geo_score` 区间 111.6–120.0；100 个决策里选中的首动作只有 `{0, 7}` 两个（stage 1 开头 400 帧内弹幕稀疏，所有计划 `margin` 都顶到 16 的上限，差异只剩 align/edge/high）。

---

## 5. 搜索验证发现的两个真实缺陷（第一个已修，第二个已定位）

### 5.1 已修：tape 第 0 行错标为第 d+1 帧

第一版 `record()` 是「先 `STEP` 再取 `HAZARDS`/`PLAYER`」，于是 tape 的第 0 行其实是**第 d+1 帧**，整个计划相对决策点偏移 1 帧。`verify.py` 的位置检查把它抓了出来（300 个计划里 30 个位置误差 > 1e-3 px，最大 16.0 px）。
修法：第 0 行在第一次 `STEP` **之前**采样（`HASH` 取决策点哈希），其余行各自帧采样后再推进。

### 5.2 已修（更严重）：`PARK` 在录制之后

第一版 `decide()` 的顺序是「录制 → 搜索 → `PARK` → `SPAWN` 验证」。可是录制已经把权威进程推到第 d+H 帧，于是 `PARK` 冻结的是**第 d+48 帧**的状态，验证根本没在决策点上做。位置检查同样抓到了（修前最大误差 20.0 px）。
修法：`decide()` 先 `PARK`（第 0 相位，计入 verify 段耗时），再录制，再把模板句柄传给 `verify_plans(..., template=...)`。

### 5.3 已定位、未修：fork 的验证天生晚 1 帧应用掩码

修完 5.2 后，位置误差从 20.0 px 降到 **4.000012 px**（6 决策复测，18/18 个计划同值；100 决策里最大 6.83 px）—— 正好是**一到两帧位移**的量级（4.0 px/帧）。

原因在输入延迟与 fork 的交互：引擎在第 t 帧锁存掩码、第 t+1 帧才用它移动（§2.2 的 1 帧延迟）。`CHILD <h> STEP <mask> 2` 是从第 d 帧开始推进两帧，但第 d 帧**已经过去**，子进程继承了父进程最后一次锁存的掩码，于是第 d+1 帧用的是旧掩码，新掩码只作用到第 d+2 帧 —— 每个 `STEP 2` 只产生 **1 次**新掩码位移。

影响：验证的**判决**（活/死、死亡帧）仍然由引擎自己的计数器给出，因此有效；但它验证的是「掩码晚 1 帧」的那条轨迹。要完全对齐，验证必须在应用计划前先补 1 帧原语（或从 d-1 帧 `PARK`）。本切片如实记录这个已知偏差（	rajectory_mismatch=300/300），不把它算成"一致"。

这条同时说明 11-P1 §2 第 4 步「验证结果与录制不一致 → 重录」为何是必需的：**引擎验证不是装饰，它会抓几何/时序侧的错**，本切片就被它抓到两次。

---

## 6. 测试

```text
python -m pytest -p no:cacheprovider -q tests/test_v400_teacher.py
20 passed in 0.82s
```

覆盖：子弹半宽 = `collisionSize/2`（4.0 宽的子弹在 1.175 px 处不判死、旧的全宽读法会判死）、含等号的接触判定、敌人 `worldPosition` + `hitbox/3`、非 FIRED/disabled 不致死、速度来自速度表（不是常量）、斜向合成等于轴向速度、边界钳制、1 帧延迟（第 0 帧原地还判死）、§3 每一项（margin 上限 / align / edge / high / 死亡分支必须低于任何存活计划）、束搜索对暴力枚举（beam ≥ 18² 时不可剪枝，逐首动作全等）、`TeacherLabel` 往返序列化、未验证 −5 折扣、`HAZARDS` 缺失时 fail-closed、`Consistency` 双向计数、§5 参数。

---

## 7. 结论

- **单决策闭环成立**：录制 → 搜索 → 引擎 fork 验证 → label，真实引擎上跑通；几何与引擎在 300 个验证计划上**完全一致**（0 个双向不一致）。
- **速度不达标**：§5 的「单影子进程 ≥1× 实时」**未达到**。含还原 0.019×；即使只算算法三段也只有 **0.59×**。原因是硬的：每决策要跑 48 帧录制 + 3×48 帧验证 = 192 个引擎帧，而行协议是「一命令一往返」（实测 shadow 帧 41 µs，但每次往返 0.1–0.3 ms），加上本切片被迫的每决策新进程 + RESET。
- 要达到 ≥1×，需要**引擎/协议侧**的改动（按优先级）：
  1. 让 fork 出来的分支能导出致死物几何（`CHILD <h> HAZARDS`，或在 BRANCH 结果缓冲里带逐帧几何）——这一条同时去掉"录制必须移动权威进程"和"还原"两项成本，是唯一的结构性解法；
  2. `RESET` 支持从停车模板 `SPAWN` 重进（§4.3 设计里本来就有，实现里没有），否则还原只能开新进程；
  3. 控制通道批量化（一次写多条命令再按序读回），把 192 次往返摊平。
- **§4 决死 bomb 不可用**（`STATE` 无 `frameEventFlags`），已在每个 label 标 `bomb_death_available=false`，未用相关量替代。

## 8. 剩余缺口

1. **决死 bomb**：需要 harness 导出 `frameEventFlags`（`ReplayManager.frameEventFlags`，10-P0 §5 已列，但 `STATE` 没实现）。
2. **激光几何无真实样本**：100 决策里 `nl` 恒为 0（与 EXP-P1-001 的 31,200 帧采样一致）；几何路径有单测但**没有真机证据**，`laser` 的 state/timer 判定窗口也因缺 `startTime/duration/hitboxStartTime` 无法复现。
3. **items 项恒 0**：`ITEMS` 未进 tape（成本考虑），对排序无影响但对绝对分有 −2 的上限偏差。
4. **重录迭代未接**：`config.max_iterations` 已参数化，`verify.reconcile()` 是显式 `NotImplementedError`，CLI 里没有跑 3 次重录循环——本切片先交付"单决策闭环 + 计时"。
5. **ECL 旋转光束**未建模（§2.1 表最后一行说只靠引擎验证兜底）。
6. 100 决策只到 400 帧、0 死亡，样本里**没有一次真实死亡**，所以 `geo_alive_engine_dead` 的计数是 0 但**未被真正考验**；需要更长/更难的局（stage 4–7 Lunatic）才能压这个指标。

## 9. 复现命令

```powershell
# 单测（Windows 侧）
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider -q tests/test_v400_teacher.py

# 能力探针 + 真实 100 决策（WSL 侧，一次只起一个引擎）
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_teacher.py --probe-only"
wsl -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v400_teacher.py \
  --decisions 100 --stage 1 --difficulty 1 --seed 1 --mode full --out .tmp_v400_teacher_full100.jsonl"
```

产物：`.tmp_v400_teacher_full100.jsonl`（每决策一条：`geo_score`、`verified`、`chosen_plan`、`margin`、`timing_ms`、`consistency`）+ 同名 `.log`（stdout 摘要与总报告）。

## 10. 修订记录

| 日期 | 修改 | 原因 |
|---|---|---|
| 2026-11-09 | 初版（单决策闭环 + 100 决策计时） | P1 首个切片；BRANCH 无几何、RESET 单次、无 frameEventFlags 三项引擎事实先探再写 |
