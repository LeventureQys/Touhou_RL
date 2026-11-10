# EXP-P2-002 标签 v2：真实 `EVENTS` 事件、数据集生成器、3,989 帧真实采集与 HUD 延迟

- 日期 / 工作包：2026-11-09 / P2-WP1（标签 v2 + HUD 滞后）、P2-WP2 最小可用版（数据生成器 + 有效性审计表）
- 依 12-P2 §3.1/§3.2/§4、[EXP-P2-001](<EXP-P2-001-labels-overlay.md>)（v1 代理标签与 1,600 帧基线）、[EXP-P1-005](<EXP-P1-005-child-hazards-events.md>)（`EVENTS` 导出落地，引擎 `f172399f…`）
- 结论：**切片成立**。
  1. 6 类事件里 **5 类现在是引擎真实事件位**（`hit`/`player_dying_started`/`effect_spawned`/`death`/`bomb`），另 2 类是显式 `proxy`（`game_over`/`extend`），`stage_clear` 仍 `unavailable`；每条都带 `event_source`。
  2. 真实采集 **7 个分片 / 3,989 帧**（3 个面、含 Lunatic 2,066 帧、2 种来源），`EVENTS` 通道 **3989/3989 帧可用**，抓到真实 `hit` 6 次、`death` 5 次、`bomb` 5 次。
  3. EVENTS↔代理一致性给出**两套键位**：未对齐（`hit(k)` vs `edge(k)`）edge 覆盖 0%；**对齐后（`hit(k)` vs `edge(k-1)`）5/5 = 100%**，唯一多出的 1 次 `hit`（死亡帧）无 DYING 边。
  4. HUD 延迟 7 行全部测到：lives/bombs/graze/point/time/power **中位数 0 帧**，score **中位数 1 帧**（滚动显示，仅 12/2586 次变化可归因）；全部 ≤ G2 的 10 帧前置。
  5. G2 的**数据量缺口**（不是检测器成绩）已按零错误二项下界给出：death/bomb 各差 **294 个正例 ≈ 234,554 帧**，`game_over`/`extend`/`stage_clear` 从未出现（不可由同分布补采解决）。
- 不包含：`reward-v1` 训练与检测器指标（WP3/WP5）、感知分辨率研究（WP4）。**本卡没有任何召回率/精确率/准确率数字**。

---

## 1. 预注册（运行前固定）

1. 标签**只能**来自 `EVENTS` + `STATE` + `PLAYER` + `HAZARDS`；像素只用于存储与「变化检测」，**不得反推标签**。
2. 事件来源三分：`"EVENTS"`（引擎 `frameEventFlags` 位）、`"proxy"`（状态导出判据）、`"unavailable"`（通道不存在，值 `None`）。`None` 表示「没观测到」，不等于「没发生」。
3. 有真实位的事件**以位为准**，同时保留状态代理在 `proxy` 子字段并给 `agreement`，便于同帧双通道对照；`bomb` 的代理在「同帧有死亡」时**拒判**（`ambiguous_with_death`），因为 `Player::UpdateDeathAndRespawn` 也会移动 bomb 存量。
4. HUD 延迟：像素只做**行带摘要是否变化**的检测；状态变化帧 `k` 与首个行带变化帧 `j∈[k, k+10]` 配对，**延迟 = j − k**（`j==k` 记 0）。行带在变化前 3 帧内动过的样本**丢弃**（滚动中的 score 不可归因）。
5. 采集并发：WSL 引擎并发 ≤ 1（脚本内 `pgrep -x th08-modern` 硬门，有别的引擎就 exit 9）；不重建任何 build，不改引擎/协议/依赖。
6. 分片续跑：`shards/<key>/attempt-NN/`，**从不覆盖、从不删除**；最新 attempt 的 `shard.json.complete=true` 才跳过。

---

## 2. 环境与真实数据来源

| 项 | 值 |
|---|---|
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl_hazard2/th08-modern`（WSL Ubuntu） |
| SHA-256 | `f172399f61b75baa8e62be46041ef756a710759f439e7fb07111f4747c78264e`，4,827,408 B |
| 角色 / 渲染 | `--rl-role env`，`xvfb-run -a -s "-screen 0 1024x768x24"`，`READBACK 1` + `SHM`（640×480 RGBA，1,228,992 B） |
| 游戏数据 | `/root/v400-data/p2/game`（独立 `score.dat`/`th08.cfg`） |
| 策略 | `random` = 18 动作（方向/聚焦每 6 帧重采样，射击常开）+ **每次决策 2% 概率追加 BOMB**；`still` = 只按 SHOOT |
| 确定性 | 每分片 `RESET <stage> <diff> 0 <seed> 0`；`seed` 作为分片身份，不声称是独立随机源（EXP-P1-004 §9） |
| Python | WSL `/root/v400-venv/bin/python`（numpy，无 Pillow；PNG 由 `labels/png.py` 用 zlib 写） |

---

## 3. 标签 v2

实现：`touhou_rl/v400/labels/events.py`（`event_labels_v2` / `EVENT_BITS` / `decode_event_flags` / `PROXY_CRITERIA`）、`spec.py`（`LABEL_VERSION="p2-v2-1"`，保留 `LABEL_VERSION_V1="p2-s4-1"`）。v1 的 `event_labels()` 原样保留（EXP-P2-001 数据仍可解析）。

### 3.1 事件定义

| 事件 | 真实位 / 判据 | `event_source` | 说明 |
|---|---|---|---|
| `hit` | `0x0002` `PLAYER_HIT` | `EVENTS` | 子弹/激光判定的命中位 |
| `player_dying_started` | `0x0200` `PLAYER_DYING_STARTED` | `EVENTS` | 死亡动画开始 |
| `effect_spawned` | `0x0400` `EFFECT_SPAWNED` | `EVENTS` | 特效生成（背景偏噪，只作辅助类） |
| `death` | `0x0004` `PLAYER_DEATH_COMMITTED` | `EVENTS` | 决死窗口结束后提交；代理 = `deaths`(f6) 增加 **或** `ALIVE→DYING` 边 |
| `bomb` | `0x0001` `BOMB_STARTED` | `EVENTS` | 代理 = `bombsRemaining` 下降且同帧无死亡（否则 `ambiguous_with_death`，值 `None`） |
| `game_over` | STEP 回复 `game_over=`（= `g_GameManager.showRetryMenu`，`rlharness.cpp HandleStep`） | `proxy` | 只有链要求退出的那一帧为真；附 `lives_zero` 交叉检查 |
| `extend` | `livesRemaining` 上升 | `proxy` | 本数据里续关不可达，故上升即 extend |
| `stage_clear` | — | `unavailable` | `GameManager.flags` bit9 未导出、无 replay-event 位；值恒 `None`（不是 `False`） |

`EVENTS` 通道不可用时（`g_ReplayManager==NULL`）：`hit`/`player_dying_started`/`effect_spawned` 变 `None` + `event_source="unavailable"` + 原因串；`death`/`bomb` 退回代理并标 `proxy`；**绝不写 0**。

### 3.2 每帧结构

`events = {hit, player_dying_started, effect_spawned, death, bomb, game_over, extend, stage_clear, channel, note}`，其中 `channel = {name:"EVENTS", available, frame_event_flags, bits, error}`；`death`/`bomb`/`extend` 带 `proxy{value, criterion, reason, components}` 与 `agreement`。

---

## 4. 真实采集（7 分片 / 3,989 帧）

命令：`.tmp_v400_p2v2/collect_suite.sh`（内部 `tools/rl/v400_p2_dataset.py suite …`）。

| 分片（`shards/…`） | 面 | 难度 | 来源 | 帧数 | 状态 | `hit` | `dying_started` | `death` | `bomb` | `danger` 真 | 占用非空 | 墙钟 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `random-s0d1-seed11-n600-b1-bp0.02` | 0 | Normal | random | **1,023**/600 | complete | 1 | 1 | 1 | 2 | 0 | 474 | 41.2 s |
| `random-s0d3-seed17-n600-b1-bp0.02` | 0 | Lunatic | random | 600 | complete | 0 | 0 | 0 | 2 | 0 | 457 | 28.3 s |
| `random-s1d2-seed14-n600-b1-bp0.02` | 1 | Hard | random | 600 | complete | 0 | 0 | 0 | 0 | 3 | 600 | 31.4 s |
| `random-s3d3-seed12-n600-b1-bp0.02` | 3 | Lunatic | random | 591 | complete | 2 | 1 | 1 | 1 | 16 | 422 | 34.3 s |
| `random-s3d3-seed16-n600-b1-bp0.02` | 3 | Lunatic | random | 575 | complete | 2 | 2 | 2 | 0 | 11 | 267 | 32.7 s |
| `still-s0d1-seed15-n300-b1` | 0 | Normal | still | 300 | complete | 0 | 0 | 0 | 0 | 0 | 140 | 16.4 s |
| `still-s3d3-seed13-n300-b1` | 3 | Lunatic | still | 300 | complete | 1 | 1 | 1 | 0 | 17 | 108 | 16.5 s |
| **合计** | 3 面 | {1,2,3} | random 3,389 / still 600 | **3,989** | 7/7 complete | **6** | **5** | **5** | **5** | **47** | 2,468 | — |

- **帧数说明**：第一个分片在第 1 次运行里因「记录循环没有在收满 `frames` 后 break」多采到 1,023 帧（真实数据，保留；`shard.json` 同时记录 `frames_requested=600 / frames_written=1023`）。修好后其余分片按请求停止；`random-s3d3-seed12`、`-seed16` 各差 9/25 帧，原因是 Lunatic 第 4 面开场无敌超过 `--warmup 700`，记录起点被推后（`warmup_steps` 见分片 meta），不是丢帧。
- **帧连续性**：同一分片内 `frame` 严格 +1（测试逐帧断言）；`frame − shm_frame == 1` 恒成立（`AfterDrawFrame` 在 `++g_frameCount` 之前发布像素，与 EXP-P2-001 §4.1 一致）。
- **`EVENTS` 可用性**：`channel_availability.EVENTS = {ok: 3989, err: 0}`，全部 7 个分片 `channel_errors.EVENTS = null`。
- 像素：3,989 张无损 PNG（640×480 RGBA），**1.98 GB**；标签 JSONL **94.0 MB**。总墙钟约 200 s（7 分片之和）。

### 4.1 事件计数（真实）

| 事件 | 正例 | 负例 | 缺失 | 来源 |
|---|---|---|---|---|
| `hit` | **6** | 3,983 | 0 | `{EVENTS: 3989}` |
| `player_dying_started` | **5** | 3,984 | 0 | `{EVENTS: 3989}` |
| `effect_spawned` | 2,261 | 1,728 | 0 | `{EVENTS: 3989}` |
| `death`（0x4） | **5** | 3,984 | 0 | `{EVENTS: 3989}` |
| `bomb`（0x1） | **5** | 3,984 | 0 | `{EVENTS: 3989}` |
| `game_over` | 0 | 3,989 | 0 | `{proxy: 3989}`（结构性：触发帧不记录，见 §8.4） |
| `extend` | 0 | 3,982 | 7 | `{proxy: 3989}`（7 个 run 首帧无前帧） |
| `stage_clear` | — | — | 3,989 | `{unavailable: 3989}` |

---

## 5. EVENTS ↔ 代理一致性（同一批帧）

原始：`audit-v2.json → event_consistency`。

| 比较 | 数字 |
|---|---|
| `EVENTS` 可用帧 | 3,989 / 3,989 |
| `EVENTS hit`(0x2) 次数 | **6** |
| 代理 `ALIVE→DYING` 次数 | **5** |
| 同帧（`hit(k)` vs `edge(k)`）两者皆有 | **0** |
| 同帧 仅 hit / 仅 edge | 6 / 5 |
| `edge_recall_by_hit`（未对齐键位） | 0.00% |
| **对齐后** `hit(k)` vs `edge(k-1)`：同时 / 仅 hit / 仅 edge / 皆无 | **5 / 1 / 0 / 3,976**（配对 3,982） |
| **对齐后 `edge_recall_by_hit`** | **100.00%** |
| `death`(0x4) vs `player_dying_started`(0x200) | 各 5；同帧 **0**；`dying_started_only` 5、`death_only` 5 |
| `death` 位 vs 状态代理 | 同时 5；仅位 **0**；仅代理 5 |
| `bomb`(0x1) vs 代理 | 位 5 / 代理 5 / 同时 5 / 仅位 0 / 仅代理 0 / 代理拒判 0 |

**键位结论（这是本次最重要的发现）**：`ALIVE→DYING` 边定义在帧对 `(k, k+1)` 上，含义是「转变发生在**第 k+1 帧内**」，而第 k+1 帧的 `EVENTS` 字才是该帧的事件。因此正确对齐是 `hit(k) vs edge(k-1)`，未对齐比较必然 0 命中。真实序列（`random-s0d1-seed11` 帧 1211–1214）：

| 帧 | `hit` | `player_dying_started` | `death`(0x4) | `deaths` 上升 | `ALIVE→DYING` 边 |
|---|---|---|---|---|---|
| 1211 | – | – | – | – | **True** |
| 1212 | **True** | **True** | – | – | – |
| 1214 | – | – | **True** | **True** | – |

即：DYING 转变 → 下一帧拿到 `0x2|0x200` → 再 2 帧拿到 `0x4` 提交。`death` 代理在「边帧」和「`deaths` 上升帧」各触发一次，故 5 次死亡产生 10 个代理正例（5 与位重合 + 5 为边帧），表中「仅代理 5」由此而来，**不是**通道不一致。

---

## 6. HUD 延迟（G2「延迟 ≤10 帧」前置）

原始：`hud-lag-v2.json`（`tools/rl/v400_p2_dataset.py hud-lag`）。像素只做行带摘要变化检测。

| 行 | 状态变化 | 可归因 | 匹配 | 中位 | p95 | 最大 | 直方图 | G2 前置（≤10） | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| lives | 5 | 5 | 5 | **0** | 0 | 0 | `{0:5}` | 满足 | tested |
| bombs | 8 | 8 | 8 | **0** | 0 | 0 | `{0:8}` | 满足 | tested |
| score | 2,586 | 12 | 12 | **1** | 1 | 1 | `{0:5, 1:7}` | 满足 | tested（见下） |
| graze | 46 | 26 | 26 | **0** | 0 | 0 | `{0:26}` | 满足 | tested |
| point | 34 | 25 | 25 | **0** | 0 | 0 | `{0:25}` | 满足 | tested |
| time | 271 | 105 | 105 | **0** | 0 | 0 | `{0:105}` | 满足 | tested |
| power | 54 | 45 | 45 | **0** | 0 | 0 | `{0:45}` | 满足 | tested |

- **没有 `untested` 行**：7 行窗口内都有状态变化，且都有可归因样本。
- `score` 特殊：行带动画率 **82.4%**（滚动 `displayScore` 逐帧变化），2,586 次状态变化里只有 **12** 次满足「变化前 3 帧行带稳定」，其余 2,574 次**丢弃**而不是猜。这 12 次给出中位 **1 帧**、最大 **1 帧**，与「先写 `globals.score`、`displayScore` 逐帧滚动追上去」的实现一致；同时说明 **score 行不能用逐帧状态变化测**，只能挑干净跳变（采集条件见 §8.5）。
- 与 EXP-P2-001 §4.3 的差异：旧实现从 `k-1` 起搜（可能把「早于状态变化」的行带变化算成延迟），本卡改为从 `k` 起、`延迟=j−k`，并在 `hud_lag_v2` 的 `alignment` 字段里写明。旧卡的 lives/graze 0 帧结论在新键位下仍然成立。
- `power`/`point`/`time` 本次真正测到（EXP-P2-001 时这三行是 `untested`）：power 23 个不同取值、point 0–22、time 0–193。

---

## 7. 有效性审计与 G2 **数据量**缺口

### 7.1 标签覆盖（真实）

| 标签 | 统计 |
|---|---|
| `player` | 全部帧 `complete`（PLAYER 通道 3989/3989 ok） |
| `hud` | `complete` 3,989/3,989；score 0–282,865；graze 0–77 |
| `occupancy` | 非空 **2,468** / 空 **1,521** / 缺失 0；`HAZARDS` 全部可用 |
| `future_danger`（H=16） | 真 **47** / 假 1,864 / `None` **2,078**（缺失率 **52.09%**）；`min_gap` p50 **86.00 px** |
| `future_danger` 缺失原因 | `player_state_INVULNERABLE` 占多数（死亡后长无敌）、`DYING`、`SPAWNING`、run 尾 16 帧 |
| HUD `lives` 分布 | `{0:189, 1:837, 2:2,963}` |
| HUD `bombs` 分布 | `{0:1,865, 1:450, 2:265, 3:1,409}` |
| HUD `power` | 0–22，23 个不同取值 |

`future_danger` 缺失 52% 是**真实状态约束**（非 ALIVE 时标签必须 `None`），不是解析失败；它同时说明「随机策略 + 3 条命」的数据里有一半时间处于无敌/重生，正例密度很低。

### 7.2 G2 数据量缺口（**不是检测器成绩**）

统计口径：零错误时单侧 95% 下界达到 99% 需 **299** 个正例（`ln0.05/ln0.99`），达到 99.5% 需 **598** 帧（`ln0.05/ln0.995`）；这是「数据够不够做判断」的下界，与模型好坏无关。

| 门 | 已有 | 需要 | 缺口 | 按当前出现率还需帧 |
|---|---|---|---|---|
| `death`（0x4） | 5 | 299 | **294** | **≈234,554** |
| `bomb`（0x1） | 5 | 299 | **294** | **≈234,554** |
| `player_dying_started`（0x200） | 5 | 299 | 294 | ≈234,554 |
| `hit`（0x2） | 6 | 299 | 293 | ≈194,797 |
| `effect_spawned` | 2,261 | 299 | 0 | 达标 |
| `game_over` | **0** | 299 | 299 | **不可由同分布补采**（见 §8.4） |
| `extend` | **0** | 299 | 299 | **不可由同分布补采**（未过分数阈值，需长局） |
| `stage_clear` | **0**（通道缺失） | 299 | 299 | **不可补采**（无导出、无事件位） |
| 残机/bomb 逐帧读数（99.5%） | 3,989 帧 | 598 | 0 | 达标（**只代表数据量够**） |
| 分数增量符号（99%） | 2,586 帧有非零增量（涨 2,586 / **跌 0**） | 299 | 0（正向）/ **299（负向）** | 正向达标；**负向 0 样本**（需要死亡扣分/续关） |
| 死亡延迟（事件数） | 5 | 30（提议下限） | **25** | 与 death 同一缺口 |

**结论（缺口 → 采集计划）**

1. **按当前分布外推需要 ~23 万帧才能把 death/bomb 正例堆到 299**：速率是 5/3,989 ≈ 0.125%。这不是「再多跑几倍同样的随机策略」能高效解决的，应**改分布**而不是改数量。
2. **改分布的三条路**（按性价比排序）：(a) 不死策略 + 高弹幕密度面（stage 4/6 Lunatic 后段、Boss 符卡）把 `future_danger` 正例和 `hit` 提上来；(b) 专门的「送死」策略（朝弹幕移动）把 death/bomb 提到每秒 1 个量级；(c) 回放/教师覆盖后段与过面（`stage_clear`/`extend`/`game_over` 只能靠这些）。
3. **不能靠补采解决**：`stage_clear`（无导出）、`game_over` 的逐帧标签（见 §8.4）。
4. **已经够的**：HUD 逐帧准确率所需帧数、occupancy 正负样本、`effect_spawned`、像素/标签对齐与完整率。

---

## 8. 数据集生成器（`tools/rl/v400_p2_dataset.py`，P2-WP2 最小可用版）

```
collect   一个分片（可续跑）      suite     一组分片（同续跑规则）
manifest  重写 manifest.json      stats     审计表 + G2 数据量缺口
hud-lag   逐行 HUD 延迟
```

- **`--source`**
  - `random`：18 动作均匀随机 + 射击常开 + `--bomb-prob`（默认 0）概率 BOMB；
  - `still`：只按 SHOOT；
  - `replay`：`TH08_AUTOPLAY_REPLAY` + `TH08_AUTOPLAY_STAGE` 环境变量，`STEP 0` 泵帧（回放自己写输入）；
  - `teacher`：**fail-closed**，`exit 3` + 明确理由（见 §8.3）。
- **`--stage/--difficulty/--seed/--frames`**（另 `--hold/--warmup/--bomb-prob/--role/--replay-path/--pixel-format {png,raw}`）。
- **产物**：`shards/<key>/attempt-NN/{labels.jsonl, pixels/frame-NNNNN.png, shard.json}` + 根 `manifest.json`。标签 `schema=v400-p2-dataset-label-2`（`pixel_format=raw` 时写 `.rgba` 原始字节 + 摘要）。
- **分片续跑**：`shards/<key>` 的最新 attempt 若 `complete=true` 则打印 `skipped: true` 跳过（本次真实发生：第一个分片被跳过、其余继续）；**从不覆盖、从不删除**，失败分片下次进 `attempt-NN+1`（本次 `random-s3d3-seed12` 就是被中断的 `attempt-01` + 完成的 `attempt-02`）。
- **每分片 meta**：引擎 SHA、`RESET`/`SHM`/`READBACK`/`DEBUG` 原始回复、逐通道 `channel_availability`/`channel_errors`、事件计数、`warmup_steps`、`shm_frame_mismatches`、`labels_sha256`、墙钟。

### 8.1 `replay` 源实测（未产出可用帧，如实记录）

`.tmp_v400_p2v2/replay_probe.sh`（`Th8_ud2970.rpy`，stage 0，120 帧请求）：

```text
{"shard": "replay-s0d1-seed0-n120-rp1", "complete": true, "frames_written": 120,
 "channel_errors": {"EVENTS": "ERR events-unavailable g_replaymanager=0",
                    "HAZARDS": "ERR hazards-not-ready reset-required",
                    "PLAYER": "ERR player-not-ready reset-required",
                    "STATE": "parse: STATE reply missing f0 (frame): 'OK state=undefined frames=1'"}}
REPLAY_EXIT=0     # 进程正常，但没有一帧进游戏
```

**结论**：`--rl-role env` + `SHM/READBACK` 下，`TH08_AUTOPLAY_REPLAY` 的入口在标题画面分支（`src/TitleScreen.cpp:450 → StartDeveloperReplayIfRequested()`，读同两个环境变量、装载 `.rpy`、直接切 `SupervisorState_GameManager`），但 720 泵帧内**没有进入该分支**：`g_ReplayManager=NULL`、`globals=NULL`、`HAZARDS/PLAYER` 全部 `not-ready`。EXP-P0-046 是在 `--rl-role shadow` 下跑通同一入口的，所以**回放源不是「不支持」，而是「env 角色下入口没被触发」**：需要先让标题画面自己走到该分支（例如在 `SHM/READBACK` 之前先泵若干帧，或新增一条 harness 命令直接触发），而这属于引擎/协议改动，本任务明确禁止。**回放帧数 = 0**，不进入数据集。

### 8.2 为什么 `teacher` 是 fail-closed

`teacher-v1`（`touhou_rl/v400/teacher/decide.py`）是**规划器**不是控制器：一个决策 = 前向录制 48 帧 → 18 首动作束搜索 → fork 引擎验证计划 → **新进程 + RESET + 重放已提交掩码**还原。EXP-P1-003 §4.1 实测 **1,744.15 ms/决策（0.5733 决策/秒）**，每决策执行 4 帧 ⇒ **≈436 ms/帧**；且还原路径每个决策换一个引擎进程，SHM 像素缓冲与 `frameEventFlags` 字都跨不过还原。按此速率 3,000 帧需 ≈22 分钟**纯教师开销**且不是连续轨迹。故 `collect --source teacher` 直接 `exit 3`，不生成任何「教师帧」。解阻条件：P1 提供「不动权威进程」的状态→动作策略。

### 8.3 失败关闭的真实例子（不是模拟）

- `EVENTS` 在无 `ReplayManager` 时：分片记录 `channel_errors.EVENTS`，该分片所有帧 `hit=None` + `unavailable`（replay 探针即此情形）。
- `HAZARDS` 不可用（replay 模式）：`occupancy=None`、`future_danger.reason="hazards_channel_unavailable: …"`，帧仍写入但明确标缺失。
- `STATE` 回 `OK state=undefined`：按协议错误处理（`state_from_reply` 抛错 → 该帧 `state=None`、`hud` 全 `None`），不当 0。

---

## 9. 测试

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p2v2_pytest -q tests/test_v400_labels.py
# 38 passed in 1.68s   (exit 0)

# 相邻层未被破坏（本卡未改 protocol/engine）
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p2v2_pytest_neighbors -q `
    tests/test_v400_protocol.py tests/test_v400_hazards.py tests/test_v400_geometry_agreement.py
# 62 passed in 2.12s   (exit 0，与 EXP-P2-001 §7 的 62 项一致)
```

`EVENTS` **协议层**解析（`parse_events_reply`：缺 `frameEventFlags` 抛 `HarnessError`、`ERR` 在 `request()` 层抛出）已由 `tests/test_v400_child_hazards.py`（EXP-P1-005）覆盖；本卡新增的是**标签层**的位解码与来源标注。

`tests/test_v400_labels.py` 从 24 项增到 **38 项**（新增 14 项，原有 24 项保持通过；3 个真实 v2 数据集测试在 manifest 未就绪时 skip，就绪后运行通过）：

| 新增测试 | 钉住什么 |
|---|---|
| `test_events_v2_bits_map_to_the_documented_engine_flags` | `0x0602 → hit/dying_started/effect_spawned=True`，其余位 `False`（不是 `None`）；每类带 `frame_event_flags` |
| `test_events_v2_every_entry_declares_a_legal_source` | 8 类全部在列、`event_source ∈ {EVENTS,proxy,unavailable}`；`stage_clear` 恒 `None` |
| `test_events_v2_missing_channel_is_null_not_false` | 通道缺失 → `hit=None`+`unavailable`+原因串；`death`/`bomb` 退代理并标 `proxy`；`game_over=None` |
| `test_events_v2_state_proxies_and_ambiguous_bomb` | `deaths` 上升 / bomb 下降 / 死亡帧 bomb 拒判 `ambiguous_with_death` / lives 上升 = extend |
| `test_events_v2_first_frame_has_no_previous_state` | run 首帧所有变化型代理为 `None` + `no_previous_frame`（不跨 run 比较） |
| `test_event_flags_decoder_is_bitwise_and_bounded` | 解码按位、`>u16` 不串位、`None → None` |
| `test_event_consistency_is_a_same_frame_2x2_table` | 同一批帧的 2×2 表与 `edge_recall_by_hit`；`dying_started_only/death_only`；`bomb` 拒判计数 |
| `test_g2_gap_is_a_data_volume_floor_not_a_detector_score` | 299/598 二项下界；报告任何层级都**没有** recall/precision/f1/accuracy 键 |
| `test_hud_lag_alignment_counts_the_same_frame_as_zero` | 合成「状态变 k、行带变 k+1」→ 中位 1；钉住 `j−k` 键位 |
| `test_hud_lag_reports_untested_without_a_state_change` | 无状态变化 → `tested=False`、`median=None`、给采集条件（不猜 0） |
| `test_real_v2_dataset_events_and_frames_are_consistent` | 真实 3,989 帧：`label_version=p2-v2-1`、≥2 面含 Lunatic、`EVENTS` 全可用、来源标注、`frame−shm_frame==1`、帧号连续、`hit>0` |
| `test_real_v2_dataset_pixels_exist_and_are_full_frames` | PNG 存在、640×480、`hud.score == state.score` |
| `test_real_v2_dataset_event_consistency_is_reported_on_the_same_frames` | 一致性表的 hit 计数与事件统计**同一批帧**一致 |
| `test_dataset_sources_fail_closed_before_touching_an_engine` | 未知源/replay 缺路径在构造期抛错；teacher 理由含 1744.15 ms/决策、0.5733 决策/秒；分片 key 是续跑身份 |

---

## 10. 限制与未覆盖（必须先读）

1. **样本量仍然很小**：真实 `hit` 6、`death` 5、`bomb` 5。本卡**没有**任何检测器指标；这 5 个正例对「召回率」没有统计意义（§7.2 已量化还差 294 个）。
2. **`game_over` 逐帧标签结构性为 0**：`STEP` 的 `game_over=1` 只出现在**链要求退出的那一帧**，而采集器在该帧结束分片、不写标签。分片 meta 有 `game_over_frame`（本次全为 `null`，即没有一局打到 game over）。要得到逐帧 `game_over` 标签必须改采集器：在退出帧再读一次 `STATE`/像素并作为终止帧记录（属于后续工作包）。
3. **`extend` 0 样本**：分数没到加命阈值（本数据最高 282,865）。需要长局或回放/教师。
4. **`stage_clear` 永不可观测**：无导出、无事件位；`None`，不是 `False`。
5. **`score` 行延迟只能挑干净跳变**：82.4% 的行带动画率意味着逐帧配对不可归因；本卡 12 个可归因样本给中位 1 帧，**不足以**替代 G2 的 10 帧门，需要「分数静默 ≥3 帧后跳变」的专门片段（例如 Boss 击破）。
6. **`future_danger` 半数为 `None`**：非 ALIVE 状态的 fail-closed 结果，正例仅 47（1.18%）；这是「随机策略大量时间在无敌/重生」的直接后果。
7. **`random` 的 BOMB 概率 2% 是采集参数**，不是引擎行为；本次 5 次 `bomb` 位与 5 次代理完全一致，但 `ambiguous_with_death` 分支**没有真实样本**（0 次），单测覆盖的是构造用例。
8. **第一个分片多采**（1,023 帧 vs 请求 600）：真实数据，保留；`shard.json` 同时记录 requested/written。它是本卡「记录循环收满即停」修复前后的证据，不是重试产物。
9. **`replay` 源 0 帧**（§8.1）：代码路径已实现并真实运行过（`exit 0`、120 帧、全通道 fail-closed），但 env 角色下没进回放分支。
10. **并发**：所有分片串行，脚本有 `pgrep -x th08-modern` 硬门；本次无其他引擎同时在跑，但墙钟只在「当时负载」下成立。
11. **像素 1.98 GB**：无损 PNG，每帧记 `rgba_sha256`/`png_sha256` 可复核；未做原始 RGBA + 外部压缩的存储优化。
12. **`tools/rl/v400_p2_labels.py` 未改**：v1 采集/叠加/审计保持 EXP-P2-001 可复现；v2 全部走 `tools/rl/v400_p2_dataset.py` + `labels/runner.py`，`labels/audit.py` 的 v2 函数兼容旧 JSONL。

---

## 11. 复现命令与退出码

```powershell
# 1) 真实采集（7 分片，3,989 帧；串行 + 并发硬门）
wsl -d Ubuntu -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p2v2/collect_suite.sh     # exit 0
#    末尾：{"shards_requested": 7, "shards_attempted": 7,
#           "frames_this_invocation": 3989, "complete_shards": 7,
#           "frames_written_complete_shards": 3989, "failures": 0}

# 2) 续跑（同样的命令再跑一次：已完成分片全部 skipped）
wsl -d Ubuntu -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p2v2/collect_suite.sh     # exit 0，全 skipped

# 3) teacher fail-closed
python tools/rl/v400_p2_dataset.py collect --engine X --data-dir Y --out-dir .tmp_v400_p2v2/teacher-probe `
  --source teacher --stage 0 --difficulty 1 --seed 0 --frames 10                   # exit 3（理由见 §8.2）

# 4) replay 源探针
wsl -d Ubuntu -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p2v2/replay_probe.sh       # exit 0（0 可用帧，§8.1）

# 5) 审计表 + G2 数据量缺口
python tools/rl/v400_p2_dataset.py stats --out-dir .tmp_v400_p2v2/dataset `
  --out .tmp_v400_p2v2/audit-v2.json --markdown .tmp_v400_p2v2/audit-v2.md           # exit 0

# 6) HUD 延迟
python tools/rl/v400_p2_dataset.py hud-lag --out-dir .tmp_v400_p2v2/dataset `
  --out .tmp_v400_p2v2/hud-lag-v2.json --max-lag 10                                  # exit 0

# 7) 单测
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p2v2_pytest -q tests/test_v400_labels.py   # 38 passed，exit 0
```

| 命令 | 退出码 | 关键输出 |
|---|---|---|
| `collect_suite.sh`（首跑） | 0 | 7/7 complete，3,989 帧，failures 0 |
| `collect_suite.sh`（再跑，实测） | 0 | 7/7 `skipped: true`（`reason: complete attempt exists`），`failures: 0` |
| `collect --source teacher` | **3** | `ERR teacher-unavailable: …1744.15 ms per decision…` |
| `replay_probe.sh` | 0 | 120 帧，四通道全 fail-closed，0 可用标签 |
| `stats` | 0 | `audit-v2.json` / `audit-v2.md`（18,584 B / 5,095 B） |
| `hud-lag` | 0 | `hud-lag-v2.json`（8,368 B），7 行全部 tested |
| `pytest tests/test_v400_labels.py` | 0 | `38 passed in 1.68s` |
| `pytest`（相邻三层） | 0 | `62 passed in 2.12s` |
| `python -m compileall` | 0 | labels 包 + 新工具语法通过 |

产物清单（均在仓库 `.tmp_v400_p2v2/`，gitignore 覆盖）：

| 路径 | 内容 |
|---|---|
| `dataset/shards/<key>/attempt-NN/labels.jsonl` | 3,989 帧标签 v2，合计 94.0 MB |
| `dataset/shards/<key>/attempt-NN/pixels/` | 3,989 张无损 640×480 PNG，1.98 GB |
| `dataset/shards/<key>/attempt-NN/shard.json` | 每分片证据（引擎 SHA、通道可用性、事件计数、墙钟、`labels_sha256`） |
| `dataset/manifest.json` | 7 分片 / 9,456 B，`frames_written_complete_shards=3989` |
| `audit-v2.json` / `audit-v2.md` | 事件计数、一致性、HUD 分布、G2 数据量缺口 |
| `hud-lag-v2.json` | 7 行延迟、直方图、丢弃计数、采集条件 |
| `collect_suite.sh` / `collect_suite.log` / `replay_probe.sh` / `replay_probe.log` | 采集与探针脚本及日志 |
| `.tmp_v400_p2v2_pytest/` | pytest basetemp（避开 `.pytest_cache` ACL） |

---

## 12. 下一步

1. **按 §7.2 改分布补采**：送死策略 + stage 4/6 Lunatic 后段/Boss，把 `death`/`bomb` 从 5 提到 ≥299，把 `hit` 提到 ≥299；`future_danger` 正例顺带上一档。
2. **游戏结束帧记账**：在 `STEP exit=1` 那一帧再读一次 `STATE`/像素并作为终止帧记录，`game_over` 才有逐帧标签（当前 0 样本是采集器的结构性缺口，不是引擎缺口）。
3. **回放入口**（§8.1）：先泵帧让标题画面走到 `StartDeveloperReplayIfRequested()`，或由负责人决定是否允许加一条 harness 命令；之后 `--source replay` 才能产出后段/过面/`stage_clear`。
4. 用本数据集冻结 `obs_spec` 的初步分辨率判断（WP4 输入），但正式决策仍按 12-P2 §4.2 预注册规则。
