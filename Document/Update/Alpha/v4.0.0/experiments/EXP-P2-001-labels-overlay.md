# EXP-P2-001 标签生成器与叠加目检（P2-WP1 最小可验收切片 S4）

- 日期 / 工作包：2026-11-09 / P2-WP1「标签生成器 `perceive/labels.py`：事件、HUD 标量、感知图；HUD 滞后测量」（[12-P2 §5](<../12-P2-像素感知与回报.md>)）
- 本切片范围（S4）：`touhou_rl/v400/labels/` 标签生成器 + 真实 1,600 帧采集 + 50 帧叠加目检图 + 有效性审计表骨架 + 单测。**不包含** `reward-v1` 训练（WP3）、感知模型/分辨率研究（WP4）、在线校验（WP5）。
- 结论：**切片成立**。标签全部由状态导出且逐帧落到真实帧上；1,600 帧像素与标签对齐（SHM 时间戳恒差 1，见 §4.1）；50 张叠加图与坐标换算实测值见 §5；审计表如实失败（§6），失败项与「补多少帧能解决 / 不能解决」已分开写清。

---

## 1. 预注册（运行前固定）

1. 标签只能来自状态导出（`HAZARDS` + `PLAYER` + `STATE`），**任何标签不得由像素反推**；像素只用于存储与目检。
2. 网格按 [12-P2 §3.3](<../12-P2-像素感知与回报.md>)：原分辨率光栅化后下采样到 1/4 ⇒ 384/4 × 448/4 = **96 × 112**，cell = 4 px。
3. `future_danger` 必须用 EXP-P1-004 验证过的 `MODEL_ENGINE_PHASE`（子弹按 velocity 推进 1 tick、用下一帧 hurtbox、`collisionSize/2`、`hitboxDimensions/3`、flag 门控、含等号、float32），在 H = 4/8/16 帧上判断「若不动是否死」，输出布尔 + 最小余量；非 ALIVE 或未来帧不足时输出 `None` + 原因（fail-closed）。
4. 坐标换算断言：屏幕坐标 = playfield 坐标 + (32,16)。若实测不是，用实测值并记录。
5. 缺失字段一律 `None`，**不得补零**；代理事件（`EVENTS` 不可观测）必须显式标注 `event_source="proxy"`。
6. 采集 ≥ 1,000 帧真实对局：stage 1 Normal + stage 4 Lunatic，固定 seed，`still` 与 `random` 各一段。

---

## 2. 环境与真实数据来源

| 项 | 值 |
|---|---|
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl_hazard/th08-modern`（WSL Ubuntu） |
| SHA-256 | `8053dfaa3d5cde0f720ba31d7f63aeba1db929a6721999c1b95e8ac00ba7b32a`，4,826,176 B |
| 角色 / 渲染 | `--rl-role env`，`xvfb-run -a -s "-screen 0 1024x768x24"`，`READBACK 1` + `SHM`（640×480 RGBA，1,228,992 B 段） |
| 游戏数据 | `/root/v400-data/p2/game`（`score.dat`/`th08.cfg` 为独立副本，`th08.dat`/`thbgm.dat` 软链；不使用其他任务的目录） |
| 采集策略 | `still` = 只按 SHOOT；`random` = 18 动作均匀随机（方向每 6 帧重采样，35% 概率 FOCUS），均不按 BOMB/MENU |
| 确定性 | 每段 `RESET <stage> <diff> 0 <seed> 0`；`RESET` 的 seed 不改变轨迹这一点沿用 EXP-P1-004 §9 的观察，本轮没有把 seed 当独立性来源 |

并发纪律：采集脚本启动前检查 `pgrep -x th08-modern`，有**任何**别的引擎在跑就拒绝启动（退出码 9）。本轮实际启动时无其他引擎；整个采集只跑 1 个引擎进程。

---

## 3. 标签定义表

实现：`touhou_rl/v400/labels/`（`spec.py`、`state.py`、`player.py`、`hud.py`、`events.py`、`occupancy.py`、`danger.py`、`png.py`；`overlay.py`/`audit.py` 只在分析侧用 Pillow/numpy）。驱动：`tools/rl/v400_p2_labels.py`。

### 3.1 `player`（来源 `PLAYER`）

| 字段 | 定义 | 缺失行为 |
|---|---|---|
| `x`,`y` | 自机 playfield 坐标（引擎 `g_Player.position.x/y`，float32） | 全 `None`，`missing=["position"]` |
| `screen` | `(x+32, y+16)`，与 §5 的换算同一函数 | 位置缺失则 `None` |
| `hurtbox` | 引擎导出的 `hurtboxMin/Max`（**不是** position ± size，避免 EXP-P1-004 §1 的半宽错误） | `None` + `missing` 列出缺的字段 |
| `hurtbox_size` | 上述 bounds 的宽高（实测 1.65 × 1.65 px） | 同上 |
| `focus`,`is_youkai` | `PLAYER focus/isYoukai` | `None` |
| `power`,`lives`,`bombs` | `PLAYER` 原始 float（**不取整**，HUD 的取整在 `hud` 里） | `None` |
| `player_state`,`player_state_name` | 0=ALIVE/1=SPAWNING/2=DYING/3=INVULNERABLE | `None` |
| `speeds` | `.sht` 四个速度（per-shot 数据，不是常数） | `None` |
| `complete` | 全部必需字段存在 | `False` |

### 3.2 `bullet_occupancy`（来源 `HAZARDS`）

| 字段 | 定义 |
|---|---|
| `width`,`height`,`cell` | 96 × 112，cell=4 playfield px |
| `occupied` | **任意**致死家族（子弹 AABB、敌人本体 AABB、激光旋转矩形）覆盖的格（P2 §3.3）；base64 打包位 |
| `bullet_occupied` | 仅子弹覆盖的格（任务字面要求）；base64 打包位 |
| `distance_uint8_b64` | 每格中心到最近**致死子弹 AABB** 的欧氏距离（0=在子弹内），uint8、0–255 px（>255 饱和）；按 32 个 box 分块精确求 `min hypot(dx,dy)`，不是按轴近似 |
| `occupied_cells`,`bullet_cells`,`laser_cells` | 计数（供审计直接读，不必解包） |
| `bullet_boxes`,`enemy_boxes`,`lasers_active` | 实际参与光栅化的对象数（geometry 过滤后） |
| `nearest_distance` | 本帧距离场最小值；无子弹时 `None`（不是 0） |
| `model` | `spec`（当前帧快照位置；`engine-phase` 的 1 tick 推进只用于 future_danger） |
| `missing` | 例：有子弹行但全被 geometry 丢弃（缺 `collisionSize`）→ `bullet_boxes_empty_despite_rows`，不静默出空网格 |

光栅化语义：闭区间重叠（含等号）——`ceil(min/cell)-1 .. floor(max/cell)`，即贴住格边界时两侧格都标记。这是故意与引擎的含等号致死判定一致，代价是边界格略偏保守（多标 1 格），在 §8 记为限制。

### 3.3 `future_danger`（来源 `HAZARDS` 的未来帧 + `PLAYER`）

| 字段 | 定义 |
|---|---|
| `would_die` | 未来 H=16 帧内，冻结自机（hurtbox 取当前帧 bounds）是否被 `MODEL_ENGINE_PHASE` 判为致死；非 ALIVE → `None` |
| `first_hit_offset` | 首次判死的未来帧偏移（1..H），未死为 `None` |
| `min_gap` | 未来各帧「冻结 hurtbox 与最近致死物」的最小空隙（0=相交）；无任何 hazard 的帧不计 |
| `per_horizon` | `{"4":…,"8":…,"16":…}`，覆盖不足的 horizon 为 `None`（run 尾部不猜 `False`） |
| `frames_considered`,`frames_with_hazard` | 实际参与的未来帧数 / 其中有 hazard 的帧数 |
| `applicable`,`reason` | `False` + `player_state_DYING` / `player_state_SPAWNING` / `player_state_INVULNERABLE` / `hurtbox_missing` / `insufficient_future_frames` |
| `model`,`still` | `"engine-phase"`、`true` |

语义边界：这是**反事实**标签——hazard 序列来自真实录制的未来帧，自机被冻结；会瞄准自机的 ECL 敌人不建模（§8）。

### 3.4 `hud`（来源 `STATE`，HUD 数值的真值）

| 字段 | STATE 字段 | 说明 |
|---|---|---|
| `score` | `f4 globals.score` | **`displayScore` 未导出**（HUD 实际画的是它），故 `display_score=None` 并在 `unavailable` 列出；本标签是 `globals.score` |
| `graze` | `f5 globals.graze` | |
| `point` | `f17 globals.pointItemsCollected` | |
| `time` | `f18 globals.currentTimeOrbs`（=`GetTimeOrbs()`） | |
| `lives` | `f7 FloatBits(livesRemaining)` | 取 `int()` 截断（与引擎 `i32` 转换一致） |
| `bombs` | `f8 FloatBits(bombsRemaining)` | 同上 |
| `power` | `f9 FloatBits(playerPower)` | 128 → 文本 `MAX`（`Gui.cpp:1381-1390`） |
| `point_total`,`time_total` | — | `nextPointItemExtendThreshold` / `lastSpellTimeOrbThreshold` 未导出 ⇒ `None`（不伪造） |
| `text` | — | HUD 会画的字符串：score `%.9d`、power `MAX`/数字、point/time `n/?` |
| `missing`,`complete` | — | 缺字段→`None` 并列出；不补零 |

HUD 坐标（`Gui.cpp:1166-1203, 1297-1390`）：score (488,56)、lives 图标 (488+16i,88)、bombs (488+16i,104)、power (488,136)、graze (488,152)、point (488,168)、time (488,184)。

### 3.5 `events`（`EVENTS`/`frameEventFlags` 不可观测）

| 事件 | 本切片值 | `event_source` | 定义 / 原因 |
|---|---|---|---|
| `death` | 布尔或 `None` | **`proxy`** | `playerState(k)==ALIVE(0) 且 playerState(k+1)==DYING(2)`；是 `frameEventFlags & 0x4` 的严格子集（EXP-P1-004 §1），不是事件位本身 |
| `hit` | `None` | `unavailable` | `REPLAY_FRAME_EVENT_PLAYER_HIT (0x2)` 在 `ReplayManager::frameEventFlags`，本构建不导出 |
| `bomb` | `None` | `unavailable` | `bombsUsed` 不导出；`bombsRemaining` 下降也被死亡处理影响，故不设代理 |
| `stage_clear` | `None` | `unavailable` | `GameManager.flags` bit9 不导出 |
| `game_over` | `None` | `unavailable` | `showRetryMenu` 不导出（`STEP` 回复里有 `game_over=`，但不是状态导出的一部分，且本数据 1,600 帧未触发） |
| `extend` | `None` | `unavailable` | `livesRemaining` 上升也发生在其他路径，不设代理 |

run 尾帧没有下一帧 → `death.value=None`，`reason="no_next_frame"`（不写成 `False`）。

---

## 4. 采集结果（真实 1,600 帧）

### 4.1 采集命令与每段统计

```powershell
wsl -d Ubuntu -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p2/collect.sh   # exit 0
# 内部：
# PYTHONPATH=/mnt/d/WorkShop/Touhou_RL xvfb-run -a -s "-screen 0 1024x768x24" \
#   /root/v400-venv/bin/python tools/rl/v400_p2_labels.py collect \
#   --engine /root/Touhou_RL/third_party/th08-rl/build/rl_hazard/th08-modern \
#   --data-dir /root/v400-data/p2/game \
#   --out .../.tmp_v400_p2/labels-s4.jsonl --frames-dir .../.tmp_v400_p2/frames \
#   --warmup 700 --runs 0:1:still:400:1000:1 0:1:random:400:1001:2 3:3:still:400:1002:1 3:3:random:400:1003:2
```

| run | stage | 难度 | 策略 | 帧数 | 帧号范围 | 状态分布（ALIVE/SPAWN/DYING/INVULN） | 最多子弹 | `death` 正例 | 段墙钟 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | Normal | still | 400 | 239–638 | 400 / 0 / 0 / 0 | 7 | 0 | 18.4 s |
| 1 | 1 | Normal | random | 400 | 239–638 | 400 / 0 / 0 / 0 | 6 | 0 | 18.7 s |
| 2 | 4 | Lunatic | still | 400 | 239–638 | 255 / 30 / 31 / 84 | 207 | **1** | 18.5 s |
| 3 | 4 | Lunatic | random | 400 | 239–638 | 400 / 0 / 0 / 0 | 338 | 0 | 19.6 s |

- 每段先「步进到首个 `playerState==ALIVE`」再开始记录：4 段均用 **238** 步走出开场无敌（上限 700），因此记录起点是帧 239、范围 239–638。
- **帧连续性**：每段 `gaps_not_one=0`，帧号严格 +1，没有跳帧拼接。
- **像素/状态对齐**：1,600/1,600 帧 `frame - shm_frame == 1`。原因是 `AfterDrawFrame()`（`main.cpp:134-139`）在 `++g_frameCount`（`main.cpp:141`）**之前**发布像素，所以 SHM 头里的帧号比 STEP 回复的帧号小 1，而像素内容就是该帧的渲染。数据集同时存 `frame` 与 `shm_frame`，把这条不变量留成可复核的证据。
- `lives`（`PLAYER`）与 `STATE f7` 逐帧一致：mismatch = 0。
- 总墙钟 75.1 s；JSONL 36,649,902 B（sha256 `760d54ab05fb194e8e17ee1aa2ce56713f88625e911cf71542abd40af4af6c10`）；PNG 1,600 个共 705.6 MB（无损，逐帧 640×480 RGBA）。

### 4.2 标签覆盖（真实计数）

| 标签 | 统计 |
|---|---|
| `player` | `complete` 1,600 / 1,600；hurtbox 缺失 0；状态 ALIVE 1,455 / INVULNERABLE 84 / DYING 31 / SPAWNING 30 |
| `hud` | `complete` 1,600 / 1,600；score 有值 1,600；lives {2: 1,486, 1: 114}；bombs {3: 1,600}；power {0: 1,600}；graze 0–30；score 0–6,520 |
| `bullet_occupancy` | 非空 **834**，空 **766**；仅子弹非空 **638**；有敌人本体 480 帧；激光格 0 帧；占用格均值 110.95、最大 1,053 |
| `future_danger` | applicable **1,455**；H=4 真 7/假 1,436/None 157；H=8 真 20/假 1,411/None 169；H=16 真 **43**/假 1,364/None 193；`min_gap`（n=795）p50 130.02 / p95 348.32 / max 433.56 px；`first_hit_offset` 覆盖 1–16 帧 |
| `events.death` | 正例 **1**（run 2，帧 493），负例 1,595，`None` 4（每段尾帧）；其余 5 类全部 `None`（通道缺失） |

`future_danger` 的 193 个 `None` 分解：`player_state_INVULNERABLE` 84、`insufficient_future_frames` 48、`player_state_DYING` 31、`player_state_SPAWNING` 30。缺失率 12.06% 是真实状态约束，不是解析失败。

### 4.3 HUD 滞后（P2 §3.2 的测量项）

用「行带像素摘要是否变化」作为**变化检测器**（不读像素值），把 `STATE` 变化帧与对应 HUD 行带的变化帧配对（两个字段同时变化的帧跳过）：

| HUD 行 | 窗口内状态变化 | 匹配到行带变化 | 滞后直方图 | 行带动画率 |
|---|---|---|---|---|
| lives | 1 | 1 | {0: 1} | 0.13% |
| graze | 26 | 26 | {0: 26} | 1.69% |
| bombs | 0 | — | — | 0.13% |
| power | 0 | — | — | 0% |
| point | 0 | — | — | 1.69% |
| time | 0 | — | — | 0.06% |

结论：**可测的两行（lives、graze）滞后为 0 帧**，即本数据里 HUD 与同帧状态一致，没有观察到 12-P2 §3.2 担心的 1 帧滞后。限制：bombs/power/point/time 在本窗口内没有状态变化（没放 bomb、power 恒 0），它们的滞后**未测**；score 行无法用此法测（`displayScore` 是滚动值且逐帧变化，不可归因），见 §8。

---

## 5. 叠加目检（50 帧）与坐标换算实测

产物：`.tmp_v400_p2/overlays-s4/`（**50 张 overlay PNG + `montage.png` = 51 个 PNG**，22.2 MB）。

- `overlay-00..49-<run>-f<frame>.png`：640×480，画布就是该帧自己的像素；叠加内容包括
  - playfield 白框（32,16,384×448）；
  - 自机点（红十字 + 中心点）与 hurtbox 白框；
  - 子弹占用网格（绿=子弹格，橙=敌人本体格，品红描边=激光格；每 8 格一条淡网格线）；
  - 右下「HUD label = state truth」面板：score/lives/bombs/power/graze/point/time 的标签值、`danger16`、`death_proxy`；
  - 底部帧身份（run/frame/stage/difficulty/policy/mask）与顶部换算说明。
- `montage.png`：50 张缩略图拼成 5 列 × 10 行（1600×2400）。
- 50 帧构成：run 0/1/2/3 = 14/9/18/9；still 32 / random 18；`death` 正例 1；`future_danger` 真 9 / 假 34 / None 7；空网格 22 帧、有子弹 26 帧。逐帧清单在 `overlays-s4/index.json`（含标签摘要）。
- 目检判据（人工）：自机红十字/白框是否落在画面上自机处；绿色格是否压在子弹上；右侧面板数值是否与 HUD 行一致。**已抽样目检** `overlay-00-run2-f493.png`（死亡帧）等：自机十字与 hurtbox 在自机上，绿色格与蓝/红子弹一一对应，面板 `score 000002600 / lives 2 / bombs 3` 与 HUD 行 `Score 000002600 / Player ★★ / Spell ★★★` 一致。

### 5.1 坐标换算断言（实测，不是照抄规格）

| 检查 | 方法 | 实测值 | 结论 |
|---|---|---|---|
| 规格来源 | `third_party/th08-rl/src/Background.cpp:740-743`：`viewport.X=32, Y=16, Width=384, Height=448` | — | 源码依据 |
| playfield 原点 | 50 帧平均的列/行边缘强度（`mean abs gray(x)-gray(x+1)`，边界索引 i ⇒ 原点 i+1） | 列边界索引 31 → **x=32**（强度 12.04，邻近 1.2–2.6）；行边界索引 15 → **y=16**（强度 16.40，邻近 0.8–2.3） | **实测 (32,16)** |
| 状态→屏幕平移（子弹） | 4,984 个「在子弹 AABB 内」的网格中心，在 ±8 px 内搜索使帧亮度最大的偏移，12 帧平均 | 峰值 **(0, −1) px**；峰值亮度 227.9，边缘基线 138.9 | 平移 = **(32,16)**，残余 ≤1 px 来自 4 px 网格量化 |
| 占用↔像素一致性 | 子弹格内 vs 格外的「白色弹核」像素密度差（26 帧） | 平均 **+0.342**（min 0.230 / max 0.638） | 状态导出的占用格确实压在渲染弹核上 |
| 焦点红点质心 | 焦点帧在 player+(32,16) 的 ±6 px 内红色像素质心（14 帧全部检出） | (−0.04, **+4.93**) px | 红点画在精灵空间、比状态点低约 5 px；这是**精灵内偏移**，不改变 (32,16) 的换算（子弹对齐峰值已独立确认） |

**断言结论：屏幕坐标 = playfield 坐标 + (32,16)，实测成立。** 代码中的唯一换算入口是 `touhou_rl/v400/labels/spec.py:playfield_to_screen`，单测固定 (32,16) 与逆变换。

---

## 6. 有效性审计表骨架（P2-WP2 前置）

原始产物：`.tmp_v400_p2/audit-s4.json` / `.tmp_v400_p2/audit-s4.md`（由 `tools/rl/v400_p2_labels.py audit --lag` 生成，exit 0；`passes=false` 是**预期**的 fail-closed 结果，不是脚本失败）。

| 标签类 | 正例 | 负例 | 缺失率 | 阈值 | 达标 | 缺口/需补采 |
|---|---|---|---|---|---|---|
| `event:death`（proxy） | 1 | 1,595 | 0.003 | 500 | 否 | 按当前出现率 **+798,400 帧** |
| `event:hit`（unavailable） | 0 | 0 | 1.000 | 500 | 否 | **不可补采**（EVENTS 通道缺失） |
| `event:bomb`（unavailable） | 0 | 0 | 1.000 | 500 | 否 | **不可补采**（通道缺失） |
| `event:stage_clear`（unavailable） | 0 | 0 | 1.000 | 500 | 否 | **不可补采**（通道缺失） |
| `event:game_over`（unavailable） | 0 | 0 | 1.000 | 500 | 否 | **不可补采**（通道缺失） |
| `event:extend`（unavailable） | 0 | 0 | 1.000 | 500 | 否 | **不可补采**（通道缺失） |
| `future_danger:positive` | 43 | 1,364 | 0.121 | 500（提议） | 否 | **+17,005 帧** |
| `future_danger:negative` | 1,364 | 43 | 0.121 | 500（提议） | 是 | 达标 |
| `occupancy:nonempty` | 834 | 766 | 0.000 | 500（提议） | 是 | 达标 |
| `occupancy:empty` | 766 | 834 | 0.000 | 500（提议） | 是 | 达标 |
| `frames_total` | 1,600 | — | — | 1,000 | 是 | 达标 |

HUD 取值分布（P2 §2.2 要求 0–8 每个取值 ≥1,000 帧）：

| HUD 标量 | 实测分布 | 结论 |
|---|---|---|
| lives | {2: 1,486, 1: 114} | 缺 0,1,3,4,5,6,7,8（1 只有 114 帧） |
| bombs | {3: 1,600} | 缺 0,1,2,4,5,6,7,8（从未放 bomb） |
| power | {0: 1,600} | 只有 1 个取值 |

**fail-closed 结论（缺什么 → 要补多少帧）**

1. **通道缺失，补采无法解决**：`hit`/`bomb`/`stage_clear`/`game_over`/`extend` 五类事件需要**新增 `EVENTS` 导出**（另一路代理正在做），不是多采帧能补的。
2. **补采可解决**：`event:death`（+798,400 帧 @ 1/1,600 的死亡率）、`future_danger:positive`（+17,005 帧 @ 43/1,600）。这两个数字都是按**本数据分布**外推；要真正降低缺口，应该改分布（更后段、更高密度弹幕、专门的 bomb/死亡策略），而不是无限重复同两段。
3. **分布缺失，补采同分布不可解决**：`lives` 的 0/3–8 与 `bombs` 的 0–2/4–8 从未出现——需要**不同关卡/策略组合**（放 bomb、死亡更多、续关、后段），不是重复 stage 1/4 前 400 帧。
4. **已达标**：总帧数、occupancy 正/负、future_danger 负例、player/HUD 完整率（100%）。

---

## 7. 测试

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p2_pytest -q tests/test_v400_labels.py
# 24 passed in 0.48s   (exit 0)

# 相邻层未被破坏（未修改这些模块）
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p2_pytest_neighbors -q \
    tests/test_v400_geometry_agreement.py tests/test_v400_hazards.py tests/test_v400_protocol.py
# 62 passed in 2.10s   (exit 0)
```

`tests/test_v400_labels.py`（24 项）覆盖：

- **坐标换算**：`(32,16)`、矩形、逆变换、与 `Background.cpp` 常量一致、网格 = 1/4 下采样；
- **网格占用**：与独立写的「闭区间重叠暴力枚举」逐格相等；贴边界时两侧格都标；空格帧距离饱和 255 且 `nearest=None`；距离是**精确** `min hypot`（构造两枚正交子弹，按轴近似会算错）；敌人 flag 门控（`NO_SPRITE` 抑制）；激光格与 `geometry.laser_hits` 同规则；编解码往返；缺 `collisionSize` 的子弹被丢弃并记入 `missing`；
- **future_danger 与几何模型一致**：用 `geometry.frame_verdict(MODEL_ENGINE_PHASE)` 独立重算 `would_die`/`first_hit_offset`/`min_gap`/`per_horizon`；构造「一帧后推进才命中」的子弹，断言 `MODEL_SPEC` 不命中而标签命中（证明真的用了 engine-phase）；非 ALIVE / 未来帧不足 → `None` + 原因；
- **HUD 字段映射**：f4/f5/f17/f18/f7/f8/f9 → score/graze/point/time/lives/bombs/power；`display_score`/`point_total`/`time_total` 为 `None`；`i32` 截断；缺字段 → `None` + `missing`（**不补零**），真 0 仍是 0；
- **真实数据**：1,600 帧逐帧断言 `label_version`、`frame-shm_frame==1`、占用解码计数自洽、`screen=x+32,y+16`、`hud.score==state.score`、帧号连续、PNG 存在且 640×480；并有 future_danger 与模型字段一致性抽查。

---

## 8. 限制与未覆盖（必须先读）

1. **事件通道缺失**：只有 `death` 一个代理正例（1 个）。这不是"检测器不行"，而是本构建根本观测不到 `frameEventFlags`；`hit`/`bomb`/`stage_clear`/`game_over`/`extend` 全部为 `None`，审计表按通道缺失处理。
2. **score 不是 HUD 真值**：HUD 画 `displayScore`，本构建不导出；标签只能是 `globals.score`。§4.3 的 score 行滞后因此**无法测量**（滚动值逐帧变化不可归因）。
3. **HUD 滞后只测到 2 行**：lives 1 次、graze 26 次变化，均为 0 帧滞后；bombs/power/point/time 在本窗口没有状态变化，滞后**未测**，不能把「0 帧」推广到所有行。
4. **激光零样本**：1,600 帧 `nl` 全为 0（与 EXP-P1-001 §3 / EXP-P1-004 §2 的已知缺口一致）。激光的占用格/距离按源码实现并与 `laser_hits` 单测对齐，但**没有真实帧验证**。
5. **future_danger 是反事实近似**：hazard 未来帧来自真实轨迹，自机被冻结；会瞄准自机的 ECL 敌人、以及「不动导致的不同未来」不建模。`min_gap` 的 p50 是 130 px，说明大量帧的"最近危险"离自机很远，正例只有 43 个。
6. **网格量化**：cell=4 px + 含等号光栅化 ⇒ 边界格偏保守（多标一格）；`distance` 以格中心计算，误差 ≤ 2√2 px。
7. **样本分布窄**：只采 stage 1 Normal 与 stage 4 Lunatic 的帧 239–638（开场后 400 帧），没有 Boss/符卡/过面；lives 只有 {2,1}、bombs 恒 3、power 恒 0；`death` 正例 1 个。按 rule of three，1 个正例对任何"召回率"都没有统计意义。
8. **像素存储 705.6 MB**：无损 PNG，不是原始 RGBA；每帧记录 `rgba_sha256` 与 `png_sha256`，可复核 PNG 解码后与捕获缓冲一致。若要节省空间应改原始 RGBA + 外部压缩，但那会把像素缓存与标签解耦，留给 P2-WP2 决定。
9. **`RESET` seed 不改变轨迹**：同一 (stage,difficulty,policy,seed) 会重放同样帧（EXP-P1-004 §9）；本卡的 4 段差异来自 stage/策略/seed 组合，但 seed 本身不是独立随机源。
10. **并发**：采集脚本有硬门（有任何 `th08-modern` 在跑就 exit 9）；本轮采集时无其他引擎，但整机仍有其他代理的短时 shadow 引擎，墙钟数字只在"当时负载"下成立。

---

## 9. 复现命令与退出码

```powershell
# 0) 环境（WSL）：数据目录、依赖
wsl -d Ubuntu -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p2/envcheck.sh     # exit 0

# 1) 真实采集（4 段，1,600 帧，像素 + 标签）
wsl -d Ubuntu -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_p2/collect.sh      # exit 0
#    输出：frames_written=1600, wall_seconds=75.124, engine_sha256=8053dfaa…

# 2) 50 帧叠加 + 拼接 + 坐标实测
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python tools/rl/v400_p2_labels.py overlay `
  --labels .tmp_v400_p2/labels-s4.jsonl --frames-dir .tmp_v400_p2/frames `
  --out-dir .tmp_v400_p2/overlays-s4 --count 50                             # exit 0

# 3) 审计表骨架（含 HUD 滞后）
python tools/rl/v400_p2_labels.py audit `
  --labels .tmp_v400_p2/labels-s4.jsonl --frames-dir .tmp_v400_p2/frames `
  --out .tmp_v400_p2/audit-s4.json --lag                                    # exit 0（passes=false 为预期 fail-closed）

# 4) 单测
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p2_pytest -q tests/test_v400_labels.py   # 24 passed，exit 0
```

| 命令 | 退出码 | 关键输出 |
|---|---|---|
| `envcheck.sh` | 0 | 引擎/数据/依赖/并发检查 |
| `collect.sh` | 0 | `frames_written=1600`、`wall_seconds=75.124`、`engine_sha256=8053dfaa…` |
| `overlay` | 0 | 50 张 overlay + `montage.png`（共 51 PNG）、`coords.json`、`index.json` |
| `audit --lag` | 0 | `audit-s4.json` / `audit-s4.md`，`passes=false`（预期） |
| `pytest tests/test_v400_labels.py` | 0 | `24 passed in 0.48s` |
| `pytest`（相邻三层） | 0 | `62 passed in 2.10s` |

产物清单（均在本仓库 `.tmp_v400_p2/`，gitignore 覆盖）：

| 路径 | 内容 |
|---|---|
| `labels-s4.jsonl` | 1,600 帧标签，36,649,902 B，sha256 `760d54ab…` |
| `labels-s4.meta.json` | 引擎 SHA、每段 RESET 回复、状态分布、SHM/READBACK 回复、耗时 |
| `frames/run{0..3}_*.png` | 1,600 张无损 640×480 像素，705.6 MB |
| `overlays-s4/` | 50 张叠加 PNG + `montage.png` + `coords.json` + `index.json`（22.2 MB） |
| `audit-s4.json` / `audit-s4.md` | 审计表骨架（真实计数 + fail-closed 结论） |
| `.tmp_v400_p2_pytest/` | pytest basetemp（避开 `.pytest_cache` ACL 问题） |

---

## 10. 下一步

1. **等 `EVENTS` 导出落地**后，把 `events` 的 5 个 `None` 换成真值（`frameEventFlags` 位 + `bombsUsed`/`GameManager.flags`），再重跑审计；死亡代理与真实 `PLAYER_DEATH_COMMITTED` 做一致性对照（EXP-P1-004 已给代理的独立旁证）。
2. **扩采分布**而不是重复同样两段：更后段/Boss、放 bomb 策略、死亡密集段（把 `death` 从 1 提到 ≥500，把 lives/bombs 每个取值 ≥1,000 帧）。
3. 把 `future_danger` 从「冻结自机」升级为「按候选动作推演」需要引擎分支（P1 的 `BRANCH`/`PARK`）；本切片刻意不做，避免把反事实标签冒充真实未来。
4. 用本数据集先冻结 `obs_spec` 的初步分辨率判断（P2-WP4 的输入），但正式决策仍按 §4.2 预注册规则。
