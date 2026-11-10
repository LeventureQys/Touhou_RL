# EXP-P1-004 几何判定与引擎判定的一致率（切片 S3）

- 日期 / 工作包：2026-11-09 / P1-WP1「几何正确性测试」（[11-P1 §6](<../11-P1-搜索教师.md>)），P1 关键正确性验证
- 假设：按 [11-P1 §2.1](<../11-P1-搜索教师.md>) 建模的自机 hurtbox AABB 与子弹/激光/敌人本体 AABB 的相交判定，在真实帧上与引擎自己的致死判定一致率 ≥ 99.9%（不含 ECL 旋转光束）。
- 预注册（运行前固定，未事后追加）：
  1. 主指标 = 逐帧一致率 `(TP+TN)/N ≥ 0.999`，N ≥ 1,000 个真实帧；
  2. 混淆矩阵以「几何撞 × 引擎击中」四格给出，且**所有**不一致帧逐条列出（帧号、几何判定、引擎信号、最近致死物距离与类型）；
  3. 引擎侧信号若不可直接观测，必须使用代理判据并**显式标注为代理**，不得写成 `PLAYER_HIT` 事件；
  4. 若发现系统性偏差（半宽取错、边界含等号处理错等），修正判定器后用**同一份已冻结数据**重跑，修正前后数字并列；
  5. 含 ECL 旋转光束的帧不计入（规格 §2.1 明确不建模）；
  6. 只有主指标成立才可写「几何模型可用于束搜索」。
- 结论：**成立**，但成立的是**修正后**的模型（`engine-phase`），不是 §2.1 字面直译的模型（`spec`）。字面直译的模型一致率 99.747%，差 7 次真实死亡，**全部漏判**，不满足 99.9%。修正点只有一个：子弹要在测试前按 `velocity` 推进一帧，并使用引擎测试时刻的 hurtbox。详见 §4。

---

## 1. 引擎事件信号的可用性（任务 1）

**结论：`frameEventFlags` 与 `PLAYER_HIT` 位在本构建里完全不可观测；只能用「死亡提交」做代理，且代理只覆盖「允许致死的那一次命中」。**

逐项核对（源码在 WSL 的 `third_party/th08-rl/src`）：

| 候选信号 | 可用 | 依据 |
|---|---|---|
| `STATE` 暴露 `frameEventFlags`（bit 0x2 = `PLAYER_HIT`） | **否** | `STATE` 只导出 25 个 replay-trace 字段（`rlharness.cpp:86-127` `TrackedFields`），`f0..f24` 里没有事件位。`frameEventFlags` 在 `ReplayManager.hpp:213`（偏移 0xda），其 `REPLAY_FRAME_EVENT_PLAYER_HIT` 由 `Player.cpp:329/356/466` 置位，但 harness 从不读取 `ReplayManager` |
| `HASH` 间接判定 | **否** | `HASH` 是上述同一 25 字段的 FNV-1a（`rlharness.cpp:170-179`），不含事件位，也不可逆 |
| `DEBUG` 间接判定 | **否** | `DEBUG` 只有 `curState/setupState/subthread/globals/stage/difficulty/shot/isReplay/isDemo/threads/frames`（`rlharness.cpp:665-682`） |
| `PLAYER` 含受伤/无敌状态 | **部分** | `playerState`（0=ALIVE,1=SPAWNING,2=DYING,3=INVULNERABLE）与 `invulnState`（= `playerState==INVULNERABLE`）。**没有**独立的「本帧被弹」布尔 |
| `STATE` 的 `deaths` / `livesRemaining` | **是（滞后）** | `f6 = FloatBits(globals.deaths)`、`f7 = FloatBits(globals.livesRemaining)`。它们在死亡**提交**时才变（`AddToDeaths` 在死亡炸弹窗口结束后，`Player.cpp:1326`），滞后 2–15 帧，只能做旁证 |
| `STATE f13` = `playerState` | **是** | `PLAYER.playerState` 的别名，本次用来交叉校验两次读取一致 |

**采用的代理判据（明确是代理）**：

```text
engine_hit(k) := playerState(k) == ALIVE(0)  AND  playerState(k+1) == DYING(2)
```

- 它是 `frameEventFlags & 0x2` 的**严格子集**，不是同一个量。`CheckLethalCollision`/`CheckBulletCollision` 在重叠后会**无条件**置 `PLAYER_HIT`，然后才判 `if (playerState != ALIVE) return;`（`Player.cpp:329-337`、`356-364`、`466-470`）——也就是说在 DYING / SPAWNING / INVULNERABLE 期间，`PLAYER_HIT` 会被置位但**没有任何可观测后果**，代理看不见这些帧。
- 反过来，`Player::Die()` 在置 `PLAYER_HIT` 的同一次调用里把 `playerState` 设为 DYING（`Player.cpp:535`），而 `Die()` 只在 `playerState == ALIVE` 时可达，所以 **ALIVE→DYING 的边沿与一次「允许致死的命中」是同一帧、一一对应**。
- 因此评估集**必须**限定在「自机 ALIVE」的帧上：那是代理与 `PLAYER_HIT` 唯一可比的区间。该限制在报告里以 `frames_alive_usable` 显式给出，不隐藏。
- 本数据集上代理与滞后计数器**完全一致**：7 次 ALIVE→DYING，7 次 `deaths` 上升，7 次 `livesRemaining` 下降（`dataset.engine_hits_on_evaluated=7`、`death_commit_events=7`、`lives_drops=7`）。这是代理正确性的独立旁证，但**它仍然只是代理**，不是事件位本身。

## 2. 样本量（任务 2）

- 引擎：`build/rl_hazard/th08-modern`，sha256 `8053dfaa3d5cde0f720ba31d7f63aeba1db929a6721999c1b95e8ac00ba7b32a`（与 [EXP-P1-001](<EXP-P1-001-harness-hazard-export.md>) §0 的构建同一份）
- 角色 `--rl-role shadow`（无窗口/null-device，无 DISPLAY），`SDL_AUDIODRIVER=dummy`，`LIBGL_ALWAYS_SOFTWARE=1`，`LP_NUM_THREADS=1`
- 数据目录：`/root/v400-data/geom/game`（`score.dat`/`th08.cfg` 从 `.orig` 复制，`th08.dat`/`thbgm.dat` 软链）——独立于并行任务的目录，不共享 score 写回
- 推进策略：8 段真实对局，每段一个**独立引擎进程**（同进程第二次 `RESET` 只会返回 `ready=0`，见 §7），`RESET <stage> <difficulty> 0 <seed> 0`，每段跳过 150 帧开场无敌期后再记录
- 逐帧记录：`HAZARDS`（子弹全字段 + 速度 + activeTimer、激光、敌人 `worldPosition`+`hitboxDimensions`）+ `PLAYER`（含 `hurtboxBoundsMin/Max`、`playerState`、`lives`、`.sht` 速度表）+ `STATE`（`f6`/`f7`/`f13`）
- 数据集：`.tmp_v400_geom/agreement.jsonl`，sha256 `d21b699d99a7632e…`，**5,000 帧**，`gaps_not_one=0`（逐帧连续，无跳帧）

| run | stage | 难度 | 策略 | 记录帧 | 自机状态分布（3 无敌 / 0 存活 / 2 死亡 / 1 复活） | 最多子弹 | 最多激光 | 最多敌人 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0（stage 1） | Normal | 不动 | 800 | 299 / 440 / 31 / 30 | 8 | 0 | 18 |
| 1 | 0 | Normal | 随机 18 动作 | 800 | 87 / 713 / 0 / 0 | 14 | 0 | 14 |
| 2 | 3（stage 4） | Lunatic | 不动 | 500 | 183 / 256 / 31 / 30 | 207 | 0 | 7 |
| 3 | 6（stage 7） | Lunatic | 不动 | 400 | 118 / 221 / 31 / 30 | 1,520 | 0 | 16 |
| 4 | 3 | Lunatic | 随机 | 800 | 315 / 424 / 31 / 30 | 359 | 0 | 17 |
| 5 | 5（stage 6） | Lunatic | 不动 | 500 | 265 / 174 / 31 / 30 | 687 | 0 | 14 |
| 6 | 5 | Lunatic | 随机 | 700 | 446 / 132 / 62 / 60 | 491 | 0 | 14 |
| 7 | 2（stage 3） | Lunatic | 不动 | 500 | 87 / 413 / 0 / 0 | 188 | 0 | 11 |

- **评估集（自机 ALIVE 且有下一帧）= 2,771 帧**，> 预注册的 1,000 帧下限
- **引擎击中（代理）= 7 次**，分布在 run 0/2/3/4/5/6
- 激光：**5,000 帧里 `nl` 全为 0**，与 [EXP-P1-001 §3](<EXP-P1-001-harness-hazard-export.md>) 的已知缺口一致（该卡在 31,200 帧里也没抓到活激光）。所以激光几何本次**零样本**，不参与一致率，也不假装参与
- ECL 旋转光束：本数据集未出现（无该类死亡），因此按预注册不计入

## 3. 几何判定器（任务 3）

实现：`touhou_rl/v400/teacher/geometry.py`（numpy，纯函数，不碰引擎），由 `tools/rl/v400_geometry_agreement.py analyse` 驱动。

规则与源码出处：

| 致死物 | 何时致死 | 判定几何 | 源码 |
|---|---|---|---|
| 子弹 | `state == 1`（FIRED）且 `collisionDisabled == 0` | 以 `position` 为中心、`collisionSize/2` 为半宽的 AABB（`collisionSize` 是**全宽**） | `Player.cpp:339-348` |
| 敌人本体 | `flags1` bit2，且 bit4/bit5/bit11 均为 0，且 `!bit27` | 以 `worldPosition` 为中心、`hitboxDimensions/3` 为半宽（`EnemyManager` 先 `/1.5`，`CheckLethalCollision` 再 `/2`） | `EnemyManager.cpp:851-853`、`EnemyManagerUpdate.cpp:287-291` |
| 激光 | `inUse` 且 `state == ACTIVE` | 把自机点旋转进激光坐标系后，与半长 `(end-start)/2`（`start>0` 时 ×0.7）、半厚 `width/4` 的矩形比较 | `Player.cpp:416-440`、`BulletManager.cpp:1069-1098` |
| ECL 旋转光束 | 脚本触发 | 不建模，靠引擎兜底 | 规格 §2.1 |

三个实现要点：

1. **边界含等号。** 引擎的拒绝式是 `hurtboxMin.x > boxMax.x` 之类的**严格**比较，所以「贴住」算撞。本实现照抄这个形状（`>=`/`>` 的取舍按引擎），不做 epsilon。
2. **float32 而非 float64。** 引擎是 x86 `float` 运算：`boundsMin.x = position->x - size->x / 2.0f`。线上数值是 float32 位模式，本实现用 `numpy.float32` 复现同样的加/减顺序。实测在 5,000 帧上 float64 与 float32 **0 处不同**（`precision_control.frames_where_float64_disagrees=0`），所以这一点在本数据上没有翻转任何判定，但保持 float32 是唯一自洽的选择。
3. **代理判据不被写成事件。** 报告里所有引擎侧字段名都是 `engine_hit`（代理）并带 `note`，没有一处写成 `PLAYER_HIT`。

## 4. 对齐、统计与不一致明细（任务 4）

### 4.1 对齐规则

引擎在**一帧之内**先做 `bullet->position += bullet->velocity`（`BulletManager.cpp:862-863`），**之后**才进入碰撞块（`BulletManager.cpp:899-950`）。所以第 k+1 帧的碰撞测试用的是 k 时刻位置**推进一帧后**的坐标，以及第 k+1 帧的 hurtbox。而命中后同一块里立刻 `bullet->state = BULLET_STATE_DESPAWNING`，因此**杀死你的那颗子弹在 k 和 k+1 两个快照里都看不见**：

- 在 k：它还没进 hurtbox（差 0.18–3.30 px，见 4.3）；
- 在 k+1：它已经变成 DESPAWNING，被 `state == FIRED` 过滤掉。

这不是实现疏漏，是逐帧点采样的结构性盲区。`engine-phase` 模型用行里已有的 `velXy` 把它放回去：`测试点 = position + velocity`，hurtbox 用 k+1 帧的。

`position_{k+1} = position_k + velocity_k` 这条恒等式在数据上直接验证过（`dataset.bullet_motion_probe`）：

```text
匹配的子弹对             : 446,666
|Δposition - velocity| 中位数 : 5.7e-06 px
p99                      : 0.0266 px
最大                     : 3.00 px（ECL 加速/变速的子弹，约占 11.8%）
```

即绝大多数子弹的运动就是一条直线段 `p_k + v`，与引擎的浮点加法同阶；剩余 11.8% 是带 transform 的变速弹，本模型对它们仍用一阶近似。

### 4.2 混淆矩阵与一致率（评估集 2,771 帧，7 次引擎击中）

**主结果（`engine-phase`：子弹推进一帧 + 引擎时刻 hurtbox + 正确半宽 + 含等号 + 敌人 flag 门控）**

| | 引擎击中=1 | 引擎击中=0 | 合计 |
|---|---|---|---|
| **几何撞=1** | **TP = 7** | **FP = 0** | 7 |
| **几何撞=0** | **FN = 0** | **TN = 2,764** | 2,764 |
| 合计 | 7 | 2,764 | 2,771 |

**一致率 = 2,771 / 2,771 = 100.000%**（不一致 0 帧）。预注册阈值 99.9% → **达标**。

### 4.3 修正前后并列（任务 6）

同一份冻结数据、同一个评估集，只换模型：

| 模型 | 说明 | TP | FP | FN | TN | 一致率 | 不一致 |
|---|---|---|---|---|---|---|---|
| `spec` | §2.1 字面直译：测试刚读到的快照（子弹不推进、hurtbox 用 k 帧） | 0 | 0 | **7** | 2,764 | **99.7474%** | 7 |
| `engine-phase` | `spec` + 子弹按 `velocity` 推进一帧 + 用 k+1 帧 hurtbox | **7** | 0 | **0** | 2,764 | **100.000%** | 0 |
| `collision-bit-only` | `engine-phase` 的敌人门控退化为只看 `flags1` bit2 | 0 | 0 | 7 | 2,764 | 99.7474% | 7 |
| `legacy-half-width` | 旧教师缺陷：`collisionSize` 当半宽、`hitboxDimensions/1.5` 当半宽、边界用严格不等 | 6 | **18** | 1 | 2,746 | **99.3143%** | 19 |

读法：

1. **系统性偏差确实存在，且就在 §2.1 的字面语义里。** `spec` 的 7 次不一致**全部**是漏判（FN），没有一次误判。原因统一：杀死自机的那颗子弹在采样相位上永远看不见。这是 100% 可复现的系统性偏差，不是噪声。
2. **旧教师的半宽错误被量出来了。** `legacy-half-width` 产生 18 次**假撞**（FP），一致率降到 99.31%。典型形态：run 6 第 507–517 帧连续 11 帧报「会撞」却没有死亡——子弹实际边缘离 hurtbox 还有 0.18–3 px，被当成 2 倍宽之后假撞。在束搜索里这类 FP 会直接把预算烧在"看似危险其实安全"的路径上。
3. **敌人 flag 门控在本数据上无差异。** `collision-bit-only` 与 `spec` 数字完全相同：本次 7 次死亡全部由子弹造成，而那 7 帧里 `damageable_enemies` 只有 run 0 一帧为 10（其余为 0），其余致死帧上没有"有 collision 位但被 flag 抑制"的敌人。所以这一项修正**本次未被数据检验**，只是按源码补上的（`EnemyManagerUpdate.cpp:287-291`）。这一点必须如实标注：它不是"验证通过的修正"，而是"未被本数据证伪的修正"。

### 4.4 全部不一致帧明细（`spec` 模型，7 帧）

`gap` = 几何模型测出的、hurtbox AABB 与"最近致死物"AABB 之间的欧氏空隙（0 表示相交）。全部 7 帧的 `gap` 都为正，即几何模型认为"差一点"。

| # | run | 帧内序号 | 引擎帧号 | 几何 | 引擎(代理) | 最近致死物 | gap (px) | 子弹数 | 敌人/其中可致死 | 下一帧几何 | 下一帧被消耗的子弹 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 526 | 677 | 0 | 1 | bullet slot 0 | 0.3744 | 7 | 11 / 10 | 0 | 2 |
| 2 | 2 | 342 | 493 | 0 | 1 | bullet slot 4 | 0.6999 | 113 | 1 / 0 | 0 | 12 |
| 3 | 3 | 307 | 458 | 0 | 1 | bullet slot 158 | 1.1590 | 1,520 | 16 / 0 | 0 | 2 |
| 4 | 4 | 510 | 661 | 0 | 1 | bullet slot 48 | 3.3017 | 328 | 9 / 0 | 0 | 6 |
| 5 | 5 | 260 | 411 | 0 | 1 | bullet slot 122 | 1.8213 | 569 | 14 / 0 | 0 | 35 |
| 6 | 6 | 189 | 340 | 0 | 1 | bullet slot 232 | 0.1812 | 299 | 14 / 0 | 0 | 44 |
| 7 | 6 | 518 | 669 | 0 | 1 | bullet slot 1309 | 0.9134 | 241 | 0 / 0 | 0 | 2 |

**归因（逐个核对后的分类）**：

| 归因 | 帧数 | 说明 |
|---|---|---|
| **读点时序 / 逐帧点采样**（子弹在两次快照之间穿过 hurtbox，且命中后被引擎在同一帧内 despawn） | **7 / 7** | 每一帧都满足三个条件：(a) `gap > 0` 但很小（0.18–3.30 px，与 3–6 px/帧的子弹步长同量级）；(b) 几何在**下一帧**依然为 0（`geom_at_next=0`），说明不是"再等一帧就对上"的简单错拍；(c) 下一帧确实有子弹消失（`despawned_bullets_next` = 2…44），与"命中即 despawn"一致 |
| ECL 旋转光束 | 0 | 数据里没有此类死亡；按预注册本就不计入 |
| 拖尾采样点（敌人 `trailSamples`） | 0 | 7 帧里 6 帧 `damageable_enemies=0`，run 0 那帧有 10 个可致死敌人但最近致死物是 bullet、gap 0.37 px。拖尾点是**已识别的未建模家族**（`EnemyManagerUpdate.cpp:295-311`，`trailFlags` 与 `trailSamples` 都未导出），本次样本未触发 |
| 判定边界（含等号处理） | 0 | 见 §5 的边界测试；且 7 帧的 gap 都远大于 1 ulp，不可能是等号取舍问题 |
| 半宽用错 | 0 | `spec`/`engine-phase` 的 7 次漏判与 `legacy` 的 18 次假撞方向相反、互不重叠；修正半宽不能消除漏判 |
| 加速器差异（float32/float64） | 0 | 5,000 帧上两种精度 0 处不同判定 |
| 敌人本体判定的调用条件（`shotType` / `HasAttachedEnemy` / `bombState` 门控） | 0 | 本次 7 次死亡全部由子弹触发 |

## 5. 测试

```text
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_geom_pytest -q tests/test_v400_geometry_agreement.py
24 passed in 2.01s

# 相邻层没被破坏（新增 geometry.py 没有改动协议/解析）
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_geom_pytest2 -q tests/test_v400_hazards.py tests/test_v400_protocol.py
38 passed in 0.12s
```

`tests/test_v400_geometry_agreement.py`（24 项）覆盖：子弹半宽 = `collisionSize/2`、**贴住算撞**与「差一个 ulp 不算撞」两侧、`state`/`collisionDisabled` 过滤、缺 `collisionSize` 时丢弃而不是当 0 宽、`engine-phase` 恰好推进一个 `velocity` 单位、**杀死自机的子弹在两个快照里都不可见而 `engine-phase` 能找回**、子弹远离时不会凭空造出命中、敌人半宽 = 原始 hitbox 的 1/3、`worldPosition` 缺失即丢弃、三个抑制 flag 各自的参数化用例、激光半厚 `width/4` / `start>0` 的 0.7 因子 / 旋转进激光坐标系 / 只认 ACTIVE、`MODELS` 与 `HURTBOX_PHASE` 一致性、最近致死物距离，以及在冻结数据集存在时的端到端回归（断言朴素模型漏掉全部 7 次死亡、`engine-phase` 一次不漏）。

## 6. 判定：几何模型是否可用于束搜索

**可用 —— 但只能用修正后的 `MODEL_ENGINE_PHASE`，不能用 §2.1 的字面直译。**

- 逐帧一致率 100.000%（2,771 帧 / 7 次真实死亡 / 0 不一致），超过预注册的 99.9%。
- 前提（必须写进 teacher 的实现，否则一致率立刻掉到 99.75%）：
  1. 子弹在测试前按 `velocity` 推进**恰好一帧**；
  2. hurtbox 用**引擎测试时刻**的那一帧（即计划的下一帧位置），不是当前帧；
  3. 半宽 = `collisionSize/2`；敌人半宽 = `hitboxDimensions/3`；敌人门控看 bit2 + bit4/bit5/bit11；
  4. 边界含等号；运算保持 float32。
- 对束搜索的含义：`spec` 的 7 次漏判是**漏掉真实死亡**，会让搜索把必死计划判成"存活"，这正是 G1 判据里"死因①（执行与验证不一致）"的来源；必须先修再用。修正后残余风险只剩三类**已知未建模**家族：ECL 旋转光束（规格已豁免）、敌人拖尾采样点、变速/加速弹的一阶近似。
- **诚实边界（必须一起读）**：正例只有 **7** 个。按 rule of three，0 次漏判在 95% 置信下的单事件漏判率上界是 `3/7 ≈ 43%`；按帧算则是 `3/2,771 ≈ 0.108%`，即逐帧不一致率 95% 上界 0.108%，**严格说 99.9% 的下界（99.892%）略低于阈值**。点估计 100% 达标，但"7 次正例"不足以把 99.9% 压成统计上确定的结论。要把置信区间收到 99.9% 以内，需要 ≥ 3,000 次真实击中（`3/N ≤ 0.001`），这在当前 3 条命/局的游戏规则下需要约 1,000 段对局。这是本卡最大的一条限制，后续 WP 应继续累加正例。

## 7. 复现命令与退出码

```powershell
# 采集（WSL 内，标准库即可；每段一个独立引擎进程）
wsl -d Ubuntu -- bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && PYTHONPATH=/mnt/d/WorkShop/Touhou_RL python3 tools/rl/v400_geometry_agreement.py collect --engine /root/Touhou_RL/third_party/th08-rl/build/rl_hazard/th08-modern --data-dir /root/v400-data/geom/game --out /mnt/d/WorkShop/Touhou_RL/.tmp_v400_geom/agreement.jsonl --runs 0:1:still:800:1000:1 0:1:random:800:1001:2 3:3:still:500:1002:1 6:3:still:400:1003:2 3:3:random:800:1004:3 5:3:still:500:1005:1 5:3:random:700:1006:4 2:3:still:500:1007:1 --seed 1 --warmup 150'

# 分析（Windows，numpy）
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python tools/rl/v400_geometry_agreement.py analyse --frames .tmp_v400_geom/agreement.jsonl --out .tmp_v400_geom/agreement.report.json

# 测试
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_geom_pytest -q tests/test_v400_geometry_agreement.py
```

| 命令 | 退出码 | 关键输出 |
|---|---|---|
| `collect` | 0 | `frames_written=5000`、`runs=8`、`wall_seconds=17.398`、`engine_sha256=8053dfaa…` |
| `analyse` | 0 | `frames=5000`、`frames_alive_usable=2771`、`engine_hits=7`、`engine-phase agreement=1.0` |
| `pytest` | 0 | `24 passed in 2.01s`；相邻层 `38 passed in 0.12s` |

产物：`.tmp_v400_geom/agreement.jsonl`（5,000 帧原始数据）、`.tmp_v400_geom/agreement.meta.json`（引擎 SHA、每段 RESET 回复、逐段错误）、`.tmp_v400_geom/agreement.report.json`（完整混淆矩阵、逐帧不一致明细、运动学探针）、`.tmp_v400_geom/agreement.engine.stderr.run*.log`。

## 8. 有效性自检

- **V1 真实帧**：5,000 帧全部来自 `rl_hazard` 二进制对真实游戏数据的推进，`gaps_not_one=0` 证明没有跳帧拼接；每段 RESET 回复（含 `setup_frames`）逐段保存在 meta 里。
- **V2 真实事件**：7 次 ALIVE→DYING 与 7 次 `deaths` 上升、7 次 `livesRemaining` 下降三者独立计数一致，死亡不是解析产物。
- **V3 朴素基线**：`spec` 是 §2.1 的字面直译，当作"什么都不修"的对照；它的 7 次错误先被观察、再被解释，不是事后编造归因。
- **V4 修正可判伪**：`engine-phase` 只在两个可指认的地方不同（推进一帧、用下一帧 hurtbox），并且 `bullet_motion_probe` 用 446,666 对实测数据证明 `p_{k+1}=p_k+v` 成立（中位误差 5.7e-6 px）。
- **V5 反向对照**：`legacy-half-width` 与 `collision-bit-only` 是同一数据上的负对照，说明指标确实会随模型变化，而不是恒等于 1。
- **V6 精度对照**：float32/float64 在 5,000 帧上 0 处不同，排除"精度取舍造成一致率"的可能。
- **V7 失败不隐藏**：激光零样本、敌人 flag 门控未被检验、变速弹一阶近似、正例仅 7 个，全部在 §2/§4.3/§6 显式写出。
- **未覆盖**：带活激光的真实帧（`nl` 全 0）；敌人拖尾采样点致死；ECL 旋转光束致死；`ERR hazards-too-large`；跨进程重放同一段的确定性（本次每段只跑一次，确定性由 EXP-P0-049 负责）。

## 9. 下一步

1. **把 `MODEL_ENGINE_PHASE` 定为 teacher 唯一可用的几何模型**，并在 `teacher/search.py` 里禁止 `bullet_advance=0` 的写法（留一个显式命名 `MODEL_SPEC` 作为文档化的错误对照）。
2. 累加正例到 ≥ 3,000 次真实击中，把 99.9% 从"点估计达标"变成"置信区间达标"。
3. 抓带活激光的帧（仍欠，见 EXP-P1-001 §3），补激光几何的真实样本。
4. 导出敌人 `trailFlags`/`trailSamples`（需要改 harness，当前构建没有），补上拖尾采样点这一未建模家族。
5. 观察到的副作用（本卡未展开，供后续核对）：`RESET` 的 `seed` 参数不改变轨迹——seed=1 与 seed=2 的同一 stage 同策略对局逐帧字节相同（本次 run 2/run 3 撞车后改用不同 stage 才得到真正不同的样本）。若后续需要"换种子确认"，得先查清 seed 在 harness 里被写到哪里。

## 10. 成本

GPU 0 h；墙钟约 50 min（其中真实引擎采样 17.4 s + 4 次试跑调整约 25 s，其余是分析与文档）；磁盘约 0.06 GB（`.tmp_v400_geom` 数据 56.8 MB + 报告 40 KB + 8 份引擎 stderr 各 255 B）。
