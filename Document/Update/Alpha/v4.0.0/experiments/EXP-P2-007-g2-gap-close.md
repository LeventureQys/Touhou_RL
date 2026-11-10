# EXP-P2-007 — G2 四个缺口的补齐：stage_clear 事件位、bomb/game_over 补采、WP5 在线校验

- **日期**：2026-11-10
- **上游**：[EXP-P2-004](EXP-P2-004-event-detector-g2.md)（death 天花板）、[EXP-P2-005](EXP-P2-005-hud-display-score.md)（HUD/displayScore）、
  [EXP-P2-006](EXP-P2-006-g2-gap-close.md)（两栏判定与 G2 报告落盘）、[EXP-P1-009](EXP-P1-009-stage-clear-diagnosis.md)/[EXP-P1-011](EXP-P1-011-multistage-progress.md)（过面诊断与真实过面帧号）
- **一句话**：把 G2 剩下的四个缺口（bomb 样本、game_over 观察窗、stage_clear 事件位、WP5 在线）逐个**用真实引擎**补上，
  不改判定、不改游戏逻辑，最后重出两栏表并明确写出「限定口径 + 选项 A」下是否全部达标。
- **禁止事项遵守**：没有重建/覆盖任何既有 build（五个 SHA 前后逐一比对，见 §1.2），没有删除文件，
  没有把不可测项算成达标，没有把 0 样本算成 100%。

---

## 0. 结论（先给）

| 缺口 | 本轮结果 | 是否关掉 |
|---|---|---|
| 1. bomb 样本 n=20 | **436 个事件**（held-out stage 3），召回/精确 100%/100%，**Wilson LB 99.1266%**，延迟 436/436 +0 帧 | **是（达标）** |
| 2. game_over 区间 26 | **408 个区间**，按 ≥1500 帧规则取 **406 个有效**，区间召回/精确 100%/100%，**Wilson LB 99.0627%**，延迟全部 89 帧 | **是（达标）** |
| 3. stage_clear 无事件位 | **事件位可用**：`EVENTS` 追加 `0x0080`，真实过面帧 13,598 置起，HASH 前 100 帧与 `rl_score` 完全一致 | **是（可测）**，但无像素侧检测器且 n=1 → 记「样本不足」 |
| 4. WP5 在线校验 | 3 局 / 7,200 帧已跑；在线 death 召回 **28.57%**（10 帧窗）/ 85.71%（30 帧窗），bomb 18/18、game_over 1/1 | **已执行，门未达标**（见 §5.3） |

---

## 1. `stage_clear` 事件位（缺口 3）

### 1.1 源码里的真实触发点

`EVENTS` 原来只导出 `ReplayManager::frameEventFlags`（`ReplayManager.hpp:213` 的 u16）。
过面屏在游戏源码里只有一个入口：

| 位置 | 行为 |
|---|---|
| `Gui.cpp:672` `case GUI_MSG_SHOW_STAGE_RESULTS:` | 填 bonus 计数（power/pointItems/graze/timeOrbs/stageBonus） |
| `Gui.cpp:686` | `this->stageClearScreenState = 1;` ← **过面屏出现的唯一一帧** |
| `Gui.cpp:687` | `g_GameManager.flags.stageClearSequenceActive = 1;` |
| `Gui.cpp:1117` | 结算动画推进 `stageClearScreenState++`（1→2） |
| `Gui.cpp:1119-1127` | `clockDisplayCurrent >= clockDisplayTarget` 且 `stageTransitionState==0` 时置 `stageTransitionState=2` |
| `Gui.cpp:2177` | 下一面 `Gui` 初始化把 `stageClearScreenState = 0` |

两处被读的字段（`GuiImpl::stageClearScreenState`、`GameManagerFlags::stageClearSequenceActive`）
**都不在 25 个 tracked 字段里**（`TrackedFields`，`rlharness.cpp:115-146`），harness 只读不写，
所以 `HASH`/`STATE` 不可能移动（§1.3 逐帧实测）。

### 1.2 实现（追加式，`rlharness.cpp`）

- 位选 **`0x0080`**：`ReplayManager.hpp:161-172` 的事件枚举已占用
  `0x0001/0x0002/0x0004/0x0020/0x0040/0x0100/0x0200/0x0400/0x0800/0x1000/0x2000`，
  **`0x0100` 是 `REPLAY_FRAME_EVENT_PAUSE_RECORDED`**，所以不能用一个"看起来空"的高位。
- 语义：**保持型电平**（过面屏期间一直置起，客户端每 N 帧轮询也看得到）；
  **精确起始帧**由新字段 `stageClearEdgeFrame` 给出，记账点放在 `PumpFrame()`（`PumpOneFrame()` 的包装）里，
  即**每推进一帧就记一次**，所以 `STEP mask 4` 这种多帧命令也不会把上升沿记错帧。
- 回包（前缀不变，只在尾部追加）：
  `OK frames=.. frameEventFlags=%04x stageClear=0/1 stageClearEdge=0/1 stageClearEdgeFrame=N stageClearScreenState=-1/0/1/2 stageClearSeq=0/1`
- `RESET` 一并清零，保证按局隔离。

构建（独立目录，既有 build 只读）：

```bash
# .tmp_v400_g2c/build_events.sh
cmake -S $SRC -B $SRC/build/rl_events -G Ninja -DCMAKE_BUILD_TYPE=Debug \
      -DTH08_RL_HARNESS=ON -DTH08_ENABLE_REPLAY_TRACE=OFF -DTH08_DX8_SDK_ROOT=$SDK
cmake --build $SRC/build/rl_events -j 16
# cmake_exit=0 build_exit=0
# 新二进制 77278a8986fceb67929208cc082d18a69f518fb9d58960258ae1115578c9097a
```

既有 build 前后 SHA（**全部未变**）：

| build | SHA-256（前=后） |
|---|---|
| `rl_score` | `d3386c9536888a1721ebcc93d16839c40a66eeee2f25aead52a107e1d9370070` |
| `rl` | `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb` |
| `rl_hazard` | `8053dfaa3d5cde0f720ba31d7f63aeba1db929a6721999c1b95e8ace00ba7b32a` |
| `rl_hazard2` | `f172399f61b75baa8e62be46041ef756a710759f439e7fb07111f4747c78264e` |
| `rl_laser` | `1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f` |

### 1.3 HASH 一致性：前 100 帧逐帧相同

`rl_score` 与 `rl_events` 各自 `RESET 0 1 0 4242 0`，同一逐帧输入序列（`i//7 % 2` 交替 `0x0001/0x0000`），
每帧记录 `STEP` 回包的 `hash` + `STATE` + `HUD` + `EVENTS`：

| 比较项 | 结果 |
|---|---|
| 帧数 | 100 / 100 |
| `hash` 不一致帧 | **0** |
| `frame` 不一致帧 | 0 |
| `STATE` 不一致帧 | 0 |
| `HUD` 不一致帧 | 0 |

证据：`.tmp_v400_g2c/hash_parity.json`（含两边的 `RESET` 回包与每帧 hash）。

### 1.4 真实过面对局：位在过面帧置起

用 EXP-P1-011 的真实过面输入流（`.tmp_v400_clear_stage0.jsonl`，3,419 条 `step_masks`，13,676 帧，
教师 `--stage 0 --difficulty normal --seed 1 --mode full`，源引擎 `rl_hazard2` `f172399f`）**原样重放到 `rl_events`**。

轨迹一致性（每 40 个决策核对 `frame/stage/px/py/lives/deaths`，共 N 次）：**divergence = 0**
→ 同一输入在 `rl_events` 上复现了同一条轨迹，墙钟 146.8 s，93 fps 左右。

过面窗口（`frameEventFlags` 是十六进制，`0480 = 0x0400(EFFECT_SPAWNED) | 0x0080(STAGE_CLEAR)`）：

| frame | rel | frameEventFlags | stageClear | stageClearScreenState | stageClearSeq | STATE.f1 (`currentStage`) |
|---|---|---|---|---|---|---|
| 13592 | -6 | `0400` | 0 | 0 | 0 | 0 |
| 13593 | -5 | `0000` | 0 | 0 | 0 | 0 |
| 13594 | -4 | `0000` | 0 | 0 | 0 | 0 |
| 13595 | -3 | `0400` | 0 | 0 | 0 | 0 |
| 13596 | -2 | `0000` | 0 | 0 | 0 | 0 |
| 13597 | -1 | `0000` | 0 | 0 | 0 | 0 |
| **13598** | **+0** | **`0480`** | **1** | **1** | **1** | **0** |
| 13599 | +1 | `0480` | 1 | 2 | 1 | 0 |
| 13600…13618 | +2…+20 | `0480`/`0080` 交替 | 1 | 2 | 1 | 0 |
| **13675** | +77 | `0480` | 1 | 2 | 1 | **1**（换面） |
| 13676 | +78 | `0480` | 1 | 2 | 1 | 1 |

- **位在真实过面帧 13,598 置起**，`stageClearScreenState` 同帧从 0 变 1，`stageClearSeq` 同帧置 1。
- 位一直保持到 13,676（79 帧），下一面 `Gui` 初始化时归零。
- **`STATE.f1` 比事件位晚 77 帧**才从 0 变 1 —— 这正说明「stage 变了」不能当事件标签，
  也解释了 EXP-P1-011 §3 里那条代理为什么只能做进度指示。
- 窗口内 ±20 帧（13,578–13,618）逐帧 `EVENTS`/`STATE` 全量在 `.tmp_v400_g2c/stage_clear_evidence.json`。

### 1.5 stage_clear 门（缺口 3 的门级结果）

见 §4.5。**如实结论**：位可用了，但**没有像素侧 stage_clear 检测器**，本轮只有 **1 个**真实过面事件，
所以这一门记「标签已可用 / 检测器缺 / 样本不足」，**不记达标**。

### 1.6 想再拿一个过面事件的尝试（失败，但得到一次独立复现）

`.tmp_v400_clear_multi.jsonl` 是 EXP-P1-011 的 36,000 帧 **两面**连过跑（`stage_transitions`：
13,676 帧 0→1、33,512 帧 1→2），也逐决策记了 `step_masks`。把它重放到 `rl_events`：

| 项 | 结果 |
|---|---|
| 第一次过面 | **13,598**，`frameEventFlags=0480`、`stageClearScreenState=1` —— **与第一次重放逐帧相同（独立复现）** |
| 第二次过面（33,512） | **没到**：重放在第 5,702 个决策 / 第 22,872 帧 `step_not_advanced` 结束 |
| 轨迹一致性 | `divergence_count=91`，从决策 3,434 起 `frame` 差恒 −60（换面后 `AdvanceToNextStage` 重置了帧计数器，教师的 `frame` 是它自己的累计计数），属于**计数器口径差**而非控制差 |

结论：**过面位在第二次独立重放里同样在 13,598 置起**（可复现性得到确认），
但**没能拿到第二个过面事件**，所以 stage_clear 门的样本仍然是 **n=1**。
根因是 harness 重放没有教师 `--follow-stage` 那条 supervisor 重建路径
（EXP-P1-011 §4.3 记录过：换面后 `GameplaySetupThread` 卡在 `while (stageTransitionState != 0) Sleep(17)`，
只有 `0x1001` 上升沿能放行），重放到第二面中途链退出。
**这条如实记为遗留，不用它来抬样本量。**

---

## 2. 补采量总表（全部真实退出码）

| 批次 | 根目录 | 策略 / 参数 | shard | 帧 | bomb | 退出码 | 墙钟 |
|---|---|---|---|---|---|---|---|
| bomb-1 | `/root/v400-data/p2det/g2c-bomb` | `random` bp=1.0 s3d3 seed9700+（含 2 个被 SIGTERM 截断的 partial，**不计入**） | 16 | 14,400 | 65 | 0（另有 2 worker 被我在调度时误杀，记为 -15） | 165 s |
| bomb-2 | `.../g2c-bomb2` | `random` bp=1.0 s3d3 seed9800+ | 94 | 84,000 | **371** | 0 / 0 | 836 s |
| bomb-cover | `.../g2c-bomb-cover` | `random` bp=1.0 s2d2 seed11000+ 与 `still` bp=1.0 同 stage | 27 | 24,000 | 73 | 0 | 475 s |
| game_over-1 | `.../g2c-go` | `dive` s3d3 `--hud` 2400 帧/局 seed20000 段首批（被我切配置时停掉，8 个完整 shard 全用） | 8 | 19,200 | — | 0 / 0（2 worker 收到 -15） | 165 s |
| game_over-2 | `.../g2c-go2` | `dive` s3d3 `--hud` **1700 帧/局** seed20000+ | 见 §4 | 见 §4 | — | 见 §4 | 见 §4 |
| game_over-3/4 | `.../g2c-go3`、`g2c-go4` | 同上，seed30000+ / 40000+ | 见 §4 | 见 §4 | — | 见 §4 | 见 §4 |

所有 shard 的 `meta.json` 里 `engine_sha256` 都是 `d3386c9536888a17…`（`rl_score`，EXP-P2-005 冻结的那个引擎）。

---

## 3. bomb 门（缺口 1）

### 3.1 为什么上一轮 bomb 是 0（根因，不是采样不足）

| 事实 | 证据 |
|---|---|
| `still` **结构上永远不放弹** | `touhou_rl/v400/labels/runner.py:265-275`：`mask()` 对 `still` 直接 `return BUTTON_SHOOT`，bomb 分支在它之后。实测：`still`+`bp=1.0` 900 帧 **0 个 bomb**，9 个 shard 共 8,100 帧仍 **0** |
| `dive`（`TopDivePolicy`）也不放弹 | `v400_p2_bulk_collect.py:665-686` 只置 `SHOOT|UP`（+漂移），从不置 `BUTTON_BOMB` |
| `random` 是唯一会放弹的策略 | `Policy.mask()` 的 `bomb_prob` 分支只在 `name == "random"` 时可达 |
| 采集器**已有** `--bomb-prob` | `v400_p2_bulk_collect.py:1229`（默认 0.02），无需新增参数 |

所以缺口 1 不是"样本不够"，是**策略选错**：EXP-P2-006 的 three 批补采全是 `dive`/`still`。
本轮改用 `random` + `bomb_prob=1.0`：每 6 帧尝试一次放弹，命中率受"每条命 3 个 bomb、死亡才补"限制，
实测稳态 **1 个 bomb 事件 / 215 帧**。

### 3.2 held-out 复测（stage 3 = test 切片，seed 9700+/9800+ 与既有 7000–8065、9100–9500 不相交）

检测器与 death 门**同一把尺**：`bombs` 计数器严格递减即报 bomb 事件；一对一匹配，预测落在 `[event, event+10]`。

| 口径 | 事件（分母） | 可观测 | 预测 | 召回 | Wilson LB | 精确 | Wilson LB |
|---|---|---|---|---|---|---|---|
| 按原文口径（全部 bomb 事件） | **436** | 436 | 436 | **100.00%** | **99.1266%** | **100.00%** | **99.1266%** |
| 限定口径（可观测事件） | 436 | — | 436 | **100.00%** | **99.1266%** | **100.00%** | **99.1266%** |

- `n=436 ≥ 386` → 零失败时 Wilson 95% **单侧**下界 99.1266% **≥ 99%** ⇒ **bomb 门达标**（两栏都达标）。
- 延迟：436/436 全部 **+0 帧**（bomb 计数器与 `BOMB_STARTED` 位**同一帧**变化）。
  这与 death 的 +4 帧不同，所以可观测窗口从事件帧本身算起（`max(1, event)`），否则会把 436 个事件全判成"不可观测"（第一版就是这么错的，见 §7.2）。
- 分根：`g2c-bomb` 65/65（LB 94.42%）、`g2c-bomb2` 371/371（LB 98.975%）；合并 436/436。

### 3.3 覆盖性（不同 stage / 不同策略）

| 批次 | stage | 策略 | shard | 帧 | bomb 事件 | 召回 | 精确 |
|---|---|---|---|---|---|---|---|
| held-out | 3（test） | `random` bp=1.0 | 110 | 98,400 | 436 | 100% | 100% |
| coverage | 2（train 切片） | `random` bp=1.0 | 18 | 16,200 | **73** | 100% | 100% |
| coverage（对照） | 2 | `still` bp=1.0 | 9 | 7,800 | **0** | **n=0（不是 0%）** | n=0 |

**读法**：同一确定性规则在 stage 2 也是 100%/100%（n=73，LB 95.00%），说明它不是 stage 3 的巧合；
`still` 的 0 事件是 §3.1 那条结构事实的实测复现，**不当成 0% 召回**（n=0 一律不可测）。
难度覆盖：held-out 全在 `difficulty=3`，这是本轮已知的**覆盖缺口**（未做 d1/d4 的 bomb 补采）。

---

## 4. game_over 门（缺口 2）

### 4.1 为什么上一轮区间召回只有 23.08%

EXP-P2-006 的 A 批（20 局 × 900 帧）在 `gameOverFlag` 上升沿之后只剩 **55 帧**，
而像素面板要 **89 帧**才静止（实测延迟中位数 89），所以检测器"来不及"；
同一份报告里 6 个 2400 帧的长局是 **6/6 全检出**。结论是**观察窗不足**，不是检测器失效。

### 4.2 本轮采样：1700 帧/局，为什么不是 2400

选项 A 第 2 条有两条硬要求：**每局观察窗 ≥ 1500 帧**、**跑到 game over 后再录 ≥600 帧**。
本批 `dive` s3d3 的 onset 实测在 **1043 帧**（`1700 - 657`），于是：

| 要求 | 1700 帧/局 | 是否满足 |
|---|---|---|
| 每局观察窗 ≥ 1500 帧 | 1700 | ✅ |
| onset 之后再录 ≥ 600 帧 | 1700 − 1043 = **657**（实测 min 651） | ✅ |

用 2400 帧/局在同等的引擎时间下只能做到 ~350 个区间，而零失败时 **Wilson 99% 下界需要 386 个**，
两者不可兼得；1700 帧同时满足两条采样规则，因此选它。**这是本轮唯一偏离任务书建议参数的地方**，
理由与算术都在上表里。

### 4.3 区间级复测（阈值未在评测集上调参）

阈值沿用 EXP-P2-006 §4.3 的**预注册规则**：`static_low = max(1.0, min(调参语料 gameplay play96 差)/2)`，
在 `g2test2`（288 shard / 161,000 gameplay 帧，min 3.4722）上算得 **1.7360956**、`change_high = 47.652`
—— 与上一轮**同一个数**，评测集（本轮新采的 4 个 root）完全没有参与阈值选择。

| 口径 | 区间 | 检出段 | 命中 | 区间召回 | Wilson LB | 区间精确 | Wilson LB |
|---|---|---|---|---|---|---|---|
| 全部区间（含 2 个截断局） | 408 | 406 | 406 | 99.51% | 98.23% | 100.00% | 99.063% |
| **有效区间（每局 ≥1500 帧，符合选项 A）** | **406** | 406 | 406 | **100.00%** | **99.0627%** | **100.00%** | **99.0627%** |

| 帧级 | tp | fp | fn | 精确 | Wilson LB | 召回 | Wilson LB |
|---|---|---|---|---|---|---|---|
| 有效 406 局 | 236,928 | **0** | 36,134 | **100.00%** | 99.9984% | 86.77% | 86.6395% |

- **延迟（区间起点 → 首个检出）**：中位 **89**、max **89**，406/406 全在 89 帧（直方图 `{"89": 406}`）。
- **观察窗实测**：每局 1700（中位）/ 1700（min）/ 2400（max）；onset 之后 657（中位）/ **651（min）** 帧；
  `shards_meeting_1500 = 406`、`shards_meeting_600_after_onset = 406`。
- **帧级 fn=36,134 全部来自面板未静止的那 89 帧**（89 × 406 ≈ 36,134，逐项对得上），
  不是检测器漏报：区间级已经 406/406。
- **那 2 个被排除的局**：worker 预算把最后一个 shard 截断成 **1,100 帧**（onset 后只剩 **57 帧**），
  面板来不及静止 → 依规则不是有效样本。**它们仍然如实列出**（口径一），没有藏起来。

### 4.4 game_over 门结论

按选项 A 的采样规则，**game_over 门达标**：区间召回/精确 100%/100%，Wilson LB **99.0627% ≥ 99%**，
样本 **406 ≥ 386**，延迟 89 帧（面板静止延迟，不是检测器延迟）。

---

## 5. WP5 在线校验（缺口 4）

### 5.1 做法

`tools/rl` 的离线门吃的是 shard 文件；在线校验吃的是**活引擎的当帧流**：
每帧 `STEP` 之后**同帧**读 `HUD`（计数器 + `gameOverFlag`）、`EVENTS`（引擎事件位，含新 `STAGE_CLEAR`）、
以及共享内存的像素回读，然后**用同一套规则**当场判事件，再和引擎真值逐帧比。
没有回放、没有重新训练、没有前视（`play96` 静止判据用 `t` 与 `t-60`，只看过去）。

| 项 | 值 |
|---|---|
| 局数 | **3**（`dive` s3d3 / `random` bp0.5 s3d3 / `random` bp0.8 s3d1） |
| 帧数 | **7,200**（每局 2,400） |
| 引擎 | `build/rl_events`（含 `STAGE_CLEAR` 位） |
| 规则 | death = `HUD lives` 严格递减；bomb = `HUD bombs` 严格递减；game_over = `play96` 差 < 1.736 的上升沿（EXP-P2-006 预注册阈值） |
| 真值 | death/bomb = `EVENTS 0x0004/0x0001` 上升沿；game_over = `HUD gameOverFlag` 连续段 |
| 证据 | `.tmp_v400_g2c/wp5-online.json`（+ 每帧 dump `wp5-online.<ep>.rows.json`） |

### 5.2 在线表

| 门 | 真值事件 | 预测 | 命中 | 召回 | Wilson LB | 精确 | Wilson LB | 延迟 |
|---|---|---|---|---|---|---|---|---|
| death（离线口径 10 帧窗） | 7 | 6 | **2** | **28.57%** | 8.22% | 33.33% | 9.68% | +4 / +29 两簇 |
| death（30 帧窗，**新增诊断行，不替代上一行**） | 7 | 6 | **6** | **85.71%** | 48.69% | 100.00% | 60.97% | 中位 29，max 29 |
| bomb | 18 | 18 | **18** | **100.00%** | 82.41% | 100.00% | 82.41% | **全部 +0** |
| game_over（区间级） | 1 | 1 | 1 | 100% | 20.65% | 100% | 20.65% | **89 帧** |
| game_over（帧级 tp/fp/fn） | — | — | 1228 / 0 / 89 | 93.24%（LB 91.76%） | — | 100.00%（LB 99.69%） | — | — |

**在线死亡门的结论是「不达标」，而且不是样本问题，是延迟问题**：7 个事件里 6 个有 HUD 递减，
但其中 4 个的递减发生在 **+29 帧**，超出原文「延迟中位数 ≤ 10 帧」的要求（在线中位数 29 > 10）。
逐局原始间隔（`death_raw_gaps`）：`[4, 4]`（dive 局）、`[29, 29]`（random 局）、`[29, 29]`（random 局）。

### 5.3 与离线表的差异（以及差异的根因）

| 差异 | 离线（EXP-P2-006） | 在线（本轮） | 根因 |
|---|---|---|---|
| death 延迟 | +4 帧 799 例、+10 帧 5 例，**max 10** | **+4（2 例）/ +29（4 例）** | 两条死亡消解路径。`GameManager.hpp:56` 有 `playerDeathDissolveMode : 2`；本轮实测：**还有 bomb 存量时 +4，bomb 用光时 +29** |
| death 召回（10 帧窗） | 100%（限定口径，n=804） | **28.57%**（n=7） | 同上：离线 test 切片里 +29 路径几乎不存在（§5.4 审计：908 事件里只有 2 例 +21、102 例 40 帧内无变化），在线语料（`bomb_prob` 高、bomb 常为 0）把它暴露出来了 |
| bomb 延迟 | — | **+0**（18/18） | bomb 计数器与 `BOMB_STARTED` 同帧变化 |
| game_over 延迟 | 89 帧（区间起点→首个检出） | **89 帧** | 一致：面板静止延迟与语料无关 |
| game_over 帧级 | tp 8,799 / fp 0 / fn 1,475 | tp 1,228 / fp 0 / fn 89 | 同一现象（fn = 面板未静止的 89 帧），量级随局数缩放 |

### 5.4 这是「线上发现」而不是「离线算错」：908 个事件的延迟重审计

`.tmp_v400_g2c/death_latency_audit.py`（只读 labels，不用像素）在离线 test 切片（406 shard / 908 事件）上重算
「death 事件 → 之后第一次 `lives` 递减」的间隔分布：

| 间隔（帧） | 事件数 |
|---|---|
| +4 | 799 |
| +10 | 5 |
| +21 | **2** |
| 40 帧内无变化 | **102** |

- 把窗口从 10 放宽到 30，只多回收 **2** 个事件（804 → 806，可观测率 88.55% → 88.77%），
  **离线结论不因此改变**：那 102 个是真的不可观测（`lives` 已经是 0）。
- 但这也说明 **离线 test 切片里几乎没有 +29 路径**，所以离线的「max 10 帧」是**语料特性**，
  不是普适保证。在线校验把它抓出来了 —— 这正是 WP5 存在的理由。
- **如实标注**：+21 的 2 例在 EXP-P2-006 里被归入「不可观测」桶（因为 10 帧窗内没有变化），
  本轮审计把它们标出来；这不改变 804/806 的计数，只补齐了一句注释。

### 5.5 在线像素侧的**未**建立的环节（不许含糊）

| 环节 | 状态 |
|---|---|
| `HUD` 计数器 ← 像素 | 离线用训练好的 reader 验证过（lives/bombs 逐帧 100%，LB 99.997%） |
| 在线「HUD 带是否变了」→ 计数器变 | **不成立**：3 局里 `lives` 只有 2 帧真的变，但 lives 像素带有 **59 帧**在变（bomb 带 59/67 帧），一致率仅 **97.62%** |

所以在线 death/bomb 行的输入是 `HUD` 的**读数值**（原文允许「逐帧 HUD/像素」，且该读数由离线 reader 以 100% 还原），
**不是**新做的像素侧事件检测器。想让在线门完全走像素，还需要一个 counter reader 在线推理（本轮没做，如实记为遗留项）。



---

## 6. 重出的 G2 两栏表与「限定口径 + 选项 A」判定

### 6.1 两栏表（机器可读 `.tmp_v400_g2c/metrics-v3.json`）

| # | 指标 | 要求 | 按原文口径 | 限定口径（death 可观测 + 独立 game_over） |
|---|---|---|---|---|
| 1 | death recall | LB >= 0.99 | 88.55% / LB 86.310% / n=908 → **未达标** | 100.00% / LB 99.524% / n=804 → **达标** |
| 2 | death precision | LB >= 0.99 | 99.75% / LB 99.100% / n=806 → **达标** | 99.75% / LB 99.100% / n=806 → **达标** |
| 3 | death latency (max) | <= 10 | 10 / n=804 → **达标** | 10 / n=804 → **达标** |
| 4 | death 可观测样本量 | >= 350 | 不适用 | 804 → **达标** |
| 5 | bomb recall | LB >= 0.99 | 100.00% / LB **99.127%** / n=**436** → **达标** | 100.00% / LB 99.127% / n=436 → **达标** |
| 6 | bomb precision | LB >= 0.99 | 100.00% / LB 99.127% / n=436 → **达标** | 100.00% / LB 99.127% / n=436 → **达标** |
| 7 | bomb 可观测样本量 | >= 350 | 不适用 | 436 → **达标** |
| 8 | lives 读数 | LB >= 0.995 | 100.00% / LB 99.997% / n=127,329 → **达标** | 同左 |
| 9 | bombs 读数 | LB >= 0.995 | 100.00% / LB 99.997% / n=127,329 → **达标** | 同左 |
| 10 | score 增量符号 | LB >= 0.99 | 100.00% / LB 99.997% / n=127,329 → **达标** | 同左 |
| 11 | game_over 区间召回 | LB >= 0.99 | 不适用（原文无此门） | 100.00% / LB **99.063%** / n=**406** → **达标** |
| 12 | game_over 区间精确 | LB >= 0.99 | 不适用（原文无此门） | 100.00% / LB 99.063% / n=406 → **达标** |
| 13 | stage_clear 召回 | LB >= 0.99 | n/a（检测器缺，n=0）→ **未达标** | n/a（检测器缺，n=0）→ **未达标** |
| 14 | 在线校验（WP5 全门） | 每个在线 LB >= 0.99 | n/a / 7,200 帧 → **未达标** | n/a / 7,200 帧 → **未达标** |

### 6.2 「限定口径 + 选项 A」是否全部达标？

**否 —— 14 行里 12 行达标，2 行不达标：**

| 不达标项 | 差什么 | 补齐需要 |
|---|---|---|
| stage_clear（第 13 行） | ①没有像素侧 stage_clear 检测器；②真实过面事件只有 1 个 | 检测器 + ≥386 次真实过面（按 13,676 帧/次、399 s/次，约 42 h 引擎时间） |
| 在线校验（第 14 行） | 在线 death 召回 28.57%（10 帧窗），30 帧窗 85.71%；根因是第二条死亡消解路径 +29 帧 | 把匹配窗口与「消解路径」解耦（例如按 `playerDeathDissolveMode` 分两档），或把在线语料的 bomb 存量做均衡 |

### 6.3 判定表述（不得含糊）

```
G2 = 未达（按原文口径）          ← 与原报告 §0 一致，未被本卡改变
G2（限定口径 + 选项 A）= 未达     ← 12/14 达标，stage_clear 与 WP5 在线两项未达标
签署状态 = 未签署（§7.2 签署栏保持空白）
```

三件事**不能**写：
1. 不能写「G2 达标 / PASS」——签署栏未签，且限定口径仍有 2 行未达标；
2. 不能把 stage_clear 的「事件位可用」写成「门达标」——没有检测器、n=1；
3. 不能把在线 death 的 30 帧窗 85.71% 当成 10 帧窗的结果——两行必须并排出现。

---

## 7. 过程中被推翻的三个判断与两次事故（如实记录）

### 7.1 bomb 的「不可观测窗口」第一版写错了

第一版把"可观测"定义成 `event+1 .. event+10` 内 `bombs` 递减（照抄 death 的写法）。
实测 436 个 bomb 事件**全部被判成不可观测**，因为 **bomb 计数器与事件位在同一帧变化**（延迟 0）。
改为从事件帧本身起算（`range(max(1, event), …)`）后 436/436 可观测。
这条差异本身是有信息的：death 的 HUD 滞后 4 帧，bomb 不滞后。

### 7.2 第一次调度时误杀了自己的两个 batch

用 `pkill -f 'tools/rl/v400_p2_bulk_collect.py worker'` 想停第一批 game_over，结果把同时跑着的
bomb 批 worker 一起 SIGTERM 了（两个批的 worker 命令行一模一样）。
已采到的 71 个 bomb、16 个 shard 全部保留（`exit_codes {"0": -15, "1": -15}`），
未完成 shard 按完整性过滤，不计入任何分母。之后的停止都改成按 data-root 精确匹配。

### 7.3 报告追加时的 heredoc 事故

第一次用 `bash -lc "…<<'MDEOF'…"` 追加 G2 报告时，PowerShell 把反引号当命令替换执行，
追加了 80 行被"吃掉反引号"的文本。已回滚到原文 392 行后重新用 `cat` 追加（无反引号解释），
并复核 `build/rl_events` 的 SHA 未变（`77278a89…`）—— 那次事故没有执行任何构建脚本。

---

## 8. 命令与退出码

```bash
# 1) 事件位构建（独立目录；五个既有 build 前后 SHA 未变）           exit 0
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_g2c/build_events.sh

# 2) HASH 前 100 帧一致性（rl_score vs rl_events，PARITY=PASS）        exit 0
wsl -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_g2c/drive_parity.py'

# 3) 真实过面重放（edge=13598，divergence=0，146.8 s）                  exit 0
wsl -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_g2c/replay_stage_clear.py'
wsl -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_g2c/stage_clear_evidence.py'

# 4) stage_clear 门（1 个事件，代理延迟 77 帧）                        exit 0
wsl -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_g2c/stage_clear_gate.py'

# 5) bomb 补采：raw random+bp1.0 s3d3（371 事件）                       exit 0 / 0
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_g2c/collect_bomb2.sh /root/v400-data/p2det/g2c-bomb2 /root/v400-data/p2det/work-g2c-bomb2 9800 84000
# 6) bomb 覆盖：s2d2 random(73) + still(0)                             exit 0
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_g2c/collect_bomb_cover.sh

# 7) bomb 门（held-out stage 3：436/436）                               exit 0
wsl -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_g2c/bomb_gate.py --root held-out=/root/v400-data/p2det/g2c-bomb --root held-out2=/root/v400-data/p2det/g2c-bomb2 --stage 3 --out .tmp_v400_g2c/bomb-gate.json'

# 8) game_over 补采（1700 帧/局；go2 跑满预算 exit 0，go3/go4 由我 SIGTERM 停止）
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_g2c/collect_gameover2.sh /root/v400-data/p2det/g2c-go2 /root/v400-data/p2det/work-g2c-go2 20000 2 400000
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_g2c/collect_gameover2.sh /root/v400-data/p2det/g2c-go3 /root/v400-data/p2det/work-g2c-go3 30000 1 200000
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_g2c/collect_gameover2.sh /root/v400-data/p2det/g2c-go4 /root/v400-data/p2det/work-g2c-go4 40000 1 200000

# 9) game_over 门（408 全部 / 406 有效窗）                              exit 0
wsl -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_g2c/gameover_gate2.py --tune plain=/root/v400-data/p2det/g2test2 --eval go=/root/v400-data/p2det/g2c-go --eval go2=/root/v400-data/p2det/g2c-go2 --eval go3=/root/v400-data/p2det/g2c-go3 --eval go4=/root/v400-data/p2det/g2c-go4 --min-episode-frames 1500 --out .tmp_v400_g2c/gameover-gate-valid.json'

# 10) WP5 在线（3 局 / 7,200 帧）                                       exit 0
wsl -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_g2c/wp5_online.py --engine /root/Touhou_RL/third_party/th08-rl/build/rl_events/th08-modern --data-dir /root/v400-data/game --out .tmp_v400_g2c/wp5-online.json --dump-rows'

# 11) 死亡延迟重审计（908 事件）                                        exit 0
wsl -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_g2c/death_latency_audit.py --root main=/root/v400-data/p2det/out-main:3 --root g2test=/root/v400-data/p2det/g2test:3 --root g2test2=/root/v400-data/p2det/g2test2:3 --out .tmp_v400_g2c/death-latency-audit.json'

# 12) 两栏表                                                            exit 0
wsl -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python .tmp_v400_g2c/assemble_g2c.py --bomb .tmp_v400_g2c/bomb-gate.json --gameover .tmp_v400_g2c/gameover-gate-valid.json --stageclear .tmp_v400_g2c/stage_clear_gate.json --online .tmp_v400_g2c/wp5-online.json --out .tmp_v400_g2c/metrics-v3.json'

# 13) 单测（Windows）                                                   exit 0，54 passed
#     $env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_g2c/pytest_run -q tests/test_v400_reward_detector.py
```

---

## 9. 单测

`tests/test_v400_reward_detector.py`：**48 → 54 passed**（+6，全部是新增门的测试）

| 新增测试 | 钉住什么 |
|---|---|
| `test_observable_bomb_scopes_separate_the_physical_ceiling` | bomb 两口径：可观测 20/30、召回 100%、精确 100%、延迟 +4 |
| `test_bomb_gate_needs_386_observable_events_for_a_99_percent_lower_bound` | n=200 时 LB < 0.99 且 verdict 判未达；n=400 时才达标 |
| `test_stage_clear_metrics_read_the_events_bit_and_measure_the_proxy_gap` | 位 = 0x0080 且不与引擎枚举冲突；无检测器时 **n=0（不是 0%）**；代理延迟 40 帧 |
| `test_v2_verdict_adds_the_stage_clear_row_only_when_the_bit_exists` | 旧 metrics 文件渲染不变；有位时新增行且 n=1 不可达标 |
| `test_online_validation_schema_needs_three_episodes_and_five_thousand_frames` | 在线 schema：<3 局或 <5000 帧一律 fail-closed |
| `test_online_validation_wilson_gate_is_not_a_point_estimate` | 6/6 = 100% 但 Wilson 不过 → 门不过 |

---

## 10. 遗留项（下一轮要接的）

| 遗留 | 现状 | 下一步 |
|---|---|---|
| stage_clear 检测器 | 只有标签位，无检测器 | 用 HUD `stageClearScreenState` 衍生特征或像素差分训一个；样本需 ≥386 次真实过面 |
| 在线 death 的第二条消解路径 | +29 帧（bomb 用光）超 10 帧窗 | 按 `playerDeathDissolveMode` 分两档统计，或先在离线语料里把 +29 路径采够再定窗 |
| 在线像素 → 计数器 | 在线只用了 `HUD` 读数值；像素带变化与计数器变化一致率仅 97.62% | 把离线 counter reader 接进在线回路逐帧推理 |
| bomb 难度覆盖 | 只有 difficulty=3（held-out）+ difficulty=2（覆盖） | 补 d1/d4 |
| `go3`/`go4` 的截断 shard | 2 个局 <1500 帧 | 已有完整性过滤，不影响判定；下轮把 `--frames-total` 设成 1700 的整数倍即可避免 |
