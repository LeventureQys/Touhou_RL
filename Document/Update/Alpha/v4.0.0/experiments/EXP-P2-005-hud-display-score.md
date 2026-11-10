# EXP-P2-005 — 导出 `displayScore`：score 门可修项的修复与 death 门口径做实

- **实验编号**：EXP-P2-005（承接 EXP-P2-004）
- **目标**：修掉 G2 两个「物理不可达」里的**可修那一个**（`score`），并把另一个（`death`）的口径问题用数据做实。
- **引擎**：新建 `third_party/th08-rl/build/rl_score`（`sha256=d3386c95…`），源码 = EXP-P1-021 的 `rl_laser` 源码 + 只读 `HUD` 命令；
  既有 `build/rl`、`build/rl_hazard`、`build/rl_hazard2`、`build/rl_laser` **未重建、未覆盖**（构建前后 SHA 相同，见 §1.3）。
- **数据**：`/root/v400-data/p2det/out-main`（194 shard；index 口径 357,729 帧 / 磁盘 358,601 行）。test = stage 3，
  **98 shard / 127,329 帧**，与 EXP-P2-004 完全相同。
- **一句话结论**：
  1. 新 `HUD` 命令导出 `displayScore`，HASH 前 100 帧与 `rl_laser` **逐帧相同**（0 mismatch）。
  2. 用同一 reader 复核：**像素里画的就是 `displayScore`**。修掉 9 格裁剪错位后，`displayScore↔像素` 整数一致率
     **100.00%**（127,329/127,329，Wilson LB 99.997%），符号准确率 **100.00%**（LB 99.997%）→ **≥99.5% / ≥99% 均达标**。
  3. 旧口径（`globals.score` 标签）在同一批帧上仍是 `73.86% / 21.80% / 76.27%`（逐位/整数/符号），**完全复现** EXP-P2-004；
     用同一个「满分 reader」去读 `globals.score` 标签仍只有 `79.60% / 47.93% / 79.81%` —— 差异只在标签，不在模型。
  4. `death`：全库 530 事件中 **342（64.5%）** 有 `lives` 递减、**188（35.5%）** 没有任何 HUD 变化。
     把事件范围限定为「有 HUD 可观测变化者」后，现有 HUD 差分检测器 test 召回 **100.00%** / 精确 **98.94%（index 口径）/98.95%（全标签行）**；
     不限定则 69.26% / 98.94%。188 个不可观测事件在 HUD 上**确实不可观测**（`lives`/`bombs` band 像素差全 0）。
  5. 附带发现（score 门的第二个独立根因）：EXP-P2-004 的 9 格 score 裁剪 `HUD_SCORE_X0=3` 与 `Gui.cpp:1299` 的
     实际绘制位置差 **5 px**（真值 `X0=8`，字形 ink 10–21/13 px pitch），旧 reader 每格只看得到字形左半 —— 这是低位数字
     塌陷（47.8%）的直接原因，与标签错配无关。

---

## 1. 引擎：只读命令 `HUD`

### 1.1 为什么必须多导出一个字段

`GameManager.cpp:535-557` 每帧把 `globals->displayScore` 向 `globals->score` 滚 `(gap)>>5`（至少 1），
`Gui.cpp:1299` 画的是 **`displayScore`**；而 25 字段 `STATE f4`/`HASH` 用的是 **`globals->score`**。
两者不同步的帧上，像素里根本没有标签信息 —— EXP-P2-004 已量化（13.4% 的 sc 不变帧像素仍在动）。
本命令把两个值同时导出，让标签消费者能选像素真正包含的那一个。

### 1.2 字段与语义

```
HUD
OK frames=<n> displayScore=<u32> globalsScore=<u32> lives=<int> bombs=<int> power=<u32>
   graze=<int> point=<int> time=<int> gameOverFlag=<u> displayAge=<u32> displayStale=<u32>
   scoreDisplayStep=<u32>
```

| 字段 | 引擎来源 | 语义 / 像素侧锚点 |
|---|---|---|
| `displayScore` | `ZunGlobals::displayScore`（Global.hpp:0x0） | **Gui.cpp:1299 实际绘制的分数** |
| `globalsScore` | `ZunGlobals::score`（0x8） | 权威总分，= `STATE f4`（旧标签） |
| `lives` | `(int)livesRemaining`（f32,0x74） | HUD 残机读数（Gui.cpp:1264-1273 图标数） |
| `bombs` | `(int)bombsRemaining`（f32,0x80） | HUD 炸弹读数（Gui.cpp:1275-1284） |
| `power` | `(int)playerPower`（f32,0x98） | Gui.cpp:1383 `"%d"` / `MAX` |
| `graze` | `ZunGlobals::graze`（i32,0xC） | Gui.cpp:1317 |
| `point` | `ZunGlobals::pointItemsCollected`（i32,0x30） | Gui.cpp:1322（HUD 的 `point` 行） |
| `time` | `ZunGlobals::currentTimeOrbs`（i32,0x3C） | Gui.cpp:1334（`GetTimeOrbs()` 直接返回它） |
| `gameOverFlag` | `g_GameManager.showRetryMenu ? 1 : 0` | 与 `STEP` 的 `game_over=` 同一个位；玩家最后一命死亡后置 1（Player.cpp:1395） |
| `displayAge` | `max(0, globalsScore - displayScore)` | 画出来的计数器还差多少分没追上 |
| `displayStale` | harness 计数（`STEP` 推进后比较） | 连续多少帧 `displayScore` 没动；0 = 本帧刚动过 |
| `scoreDisplayStep` | `ZunGlobals::scoreDisplayStep`（0x10） | 当前每帧滚动步长；归零 = 已追上 |

**只读性保证**：handler 只读 `g_GameManager.globals` 与 `showRetryMenu`；不推进时间、不绘制、不取 RNG、不写任何 25 字段；
`displayStale` 的维护放在 `HandleStep()` 的 `PumpOneFrame()` 之后（harness 局部变量），引擎状态零写入。
`HUD` **不在** `TrackedFields()`/`PublishState()`/`PublishPixels()` 里，也不在 SHM 布局里。

### 1.3 构建（独立目录，参数照抄 `build/rl`）

`cmake -S ... -B build/rl_score -G Ninja -DCMAKE_BUILD_TYPE=Debug -DTH08_RL_HARNESS=ON -DTH08_ENABLE_REPLAY_TRACE=OFF -DTH08_DX8_SDK_ROOT=scripts/prefix/mssdk`
（与 `build/rl/CMakeCache.txt` 的关键项逐条一致；脚本 `.tmp_v400_hud/build_score.sh`）。

| 项目 | 值 |
|---|---|
| `cmake_exit` / `BUILD_EXIT` | `0` / `0` |
| `build/rl_score/th08-modern` | 4,829,776 B，`sha256=d3386c9536888a1721ebcc93d16839c40a66eeee2f25aead52a107e1d9370070` |
| 既有 `build/rl` | `591c521d…` **未变**（构建前=构建后） |
| 既有 `build/rl_hazard` | `8053dfaa…` **未变** |
| 既有 `build/rl_hazard2` | `f172399f…` **未变** |
| 既有 `build/rl_laser` | `1462829b…` **未变** |

### 1.4 HASH 一致性：前 100 帧逐帧相同

`.tmp_v400_hud/parity/parity.json`：两个引擎各自 `RESET 3 1 0 4242 0` → warm-up 238 帧 → 同一组 100 个 `(mask, STEP 1)`，
每帧 `HASH` 对比。

```json
{"compared_frames": 100, "hash_mismatches": [], "hash_mismatch_count": 0, "identical": true}
```

即：新增 `HUD` 没有移动 25 字段哈希，G0.2 的位级口径不受影响。

---

## 2. `displayScore` ↔ HUD 像素一致性

### 2.1 实况对局 2,000 帧（`rl_score`，stage 3 / difficulty 1 / seed 4242）

逐帧 `STEP` → `HUD` → 从 SHM 取 HUD band（480-640 列、48-112 行，绿通道 56×160），用 9 格裁剪 reader 读数。

滚入序列（`displayAge` 首次 >0 的窗口，`.tmp_v400_hud/capture.log`）：

```
i=1345 f=1584 ds=0 gs=0  age=0  step=0 reader=0
i=1346 f=1585 ds=0 gs=0  age=0  step=0 reader=0
i=1347 f=1586 ds=0 gs=10 age=10 step=0 reader=0
i=1348 f=1587 ds=1 gs=20 age=19 step=1 reader=1
...（ds 每帧 +1，gs 停在 20）
i=1395 f=1634 ds=2 gs=30 age=28 step=1 reader=2
i=1396 f=1635 ds=3 gs=40 age=37 step=1 reader=3
```

- 同帧 `reader == displayScore`，而 `reader != globalsScore`：**像素侧看到的是 `displayScore`**。
- 2000 帧统计（**旧 reader + 旧裁剪 x0=3**，`.tmp_v400_hud/capture/capture-summary.json`）：
  `displayScore` 逐位 97.72% / 整数 81.85% / 符号 99.65%；同一批像素换 `globalsScore` 标签：90.75% / 67.35% / 99.60%。
  旧 reader 的整数误差来自 §2.3 的裁剪错位，不是标签。
- 修好裁剪 + 用 `displayScore` 标签重训后（`.tmp_v400_hud/capture-fixed/…`）：整数一致率 `100.00%（2,000/2,000）`，符号 `100.00%`（n=2,000）。

### 2.2 test 集 127,329 帧（同一批像素、同一 reader、两种标签）

`score-gate` 子命令（`tools/rl/v400_p2_reward_detector.py`）在 test 的 98 shard 上跑：

| 标签 | reader / 裁剪 | 逐位 | 整数 | 符号 | 符号 LB(95%) |
|---|---|---|---|---|---|
| `globalsScore` | EXP-P2-004 `v1.pt` / x0=3 | 73.86% | 21.80% | 76.27% | 76.03% |
| `displayScore` | EXP-P2-004 `v1.pt` / x0=3 | 88.86% | 33.85% | 91.97% | 91.82% |
| `displayScore` | `v1.pt` / x0=8（错位纠正，未重训） | 58.64% | 15.85% | 58.98% | 58.71% |
| **`displayScore`** | **重训 reader / x0=8** | **100.00%** | **100.00%** | **100.00%** | **99.997%** |
| `globalsScore` | 重训 reader / x0=8（同一次评测） | 79.60% | 47.93% | 79.81% | 79.59% |

第一行**逐位复现** EXP-P2-004 的 73.86/21.80/76.27 —— 说明评估管线没变，变化只来自标签与裁剪。
最后两行是同一份权重、同一批像素：换成 `globalsScore` 标签后，**满分 reader 也只能到 47.93% 整数** —— 差异在标签，不在模型。

### 2.3 第二个根因：9 格裁剪错位 5 px

在已存平面上量字形（`.tmp_v400_hud/measure_p2det.py`）：score 行 9 个字形 ink 落在 band 列
`10–21, 23–34, …, 114–125`（13 px 一组），绘制锚点 `Gui.cpp:1299` 的 x=488 − plane 起点 480 = **列 8**。
而 EXP-P2-004 的 `HUD_SCORE_X0 = 3`，每格 `[3+13i,16+13i)` 只覆盖字形左半 + 空白 → 低位数字必然塌陷
（EXP-P2-004 逐位 100/100/100/98.5/66.6/58.8/47.4/45.6/47.8 的形状就是这么来的）。
本轮用锚点 `X0=8` 重训后 127,329 帧**全对**，说明右侧 1 px 的余量（字形 ink 到 125、格子到 125）不影响识别。

- 证据 A：同一批像素换 `displayScore` 标签，旧裁剪逐位从 73.86% 升到 88.86%，但整数仍只有 33.85% —— 标签对了，字形仍读不全。
- 证据 B：旧 reader 换到 x0=8（它没见过的正确输入）反而掉到 58.64% —— 说明模型锁死在错误裁剪上，**必须重训**。
- 修复：`HUD_SCORE_X0_MEASURED = 8`（新增常量），`score_glyph_boxes(x0)/hud_crops(score_x0=…)/train_reader(score_x0=…)`
  参数化；旧常量 `3` 保留用于复现 EXP-P2-004。重训 reader 只用 **train split** 的 `displayScore` 标签（80 shard / 192,000 帧，
  由 `HUD` 命令重采，`.tmp_v400_hud/relabel-train`），test 完全未见过。

### 2.4 一致性判定

`displayScore ↔ 像素` 整数一致率 **`100.00%`**（n=127,329，Wilson 95% LB `99.997%`）→ **≥99.5% 达标**。

拆分（同一 reader，`settled 61,031 / active 51,834 / frozen 14,464 / go=1 43,075，全部 100.00%`）：settled / active roll / frozen / retry-menu 各自的整数与符号准确率见
`.tmp_v400_hud/score-splits.json`；`displayScore` 在 run 内**从不下降**（`down=0`），所以符号门的负类仍为 0，
只能对 `same/up` 两类宣称（见 §3.3）。

---

## 3. score 门重评

### 3.1 标签重采（不是推导）

用 `HUD` 命令对 test 98 shard **逐帧重采** `displayScore`（`.tmp_v400_hud/relabel/`，`shards/<key>.jsonl` 每帧一行）。
对齐校验：重放同一 `(stage,difficulty,seed)` + 同一冻结策略 mask 序列 + 同一 warm-up，
逐帧比对 `globalsScore/lives/bombs/power/graze/point/time` 与磁盘上原标签。

| 项目 | 值 |
|---|---|
| test shard / 帧 | 98 / **127,329** |
| 对齐（逐帧全等）shard / 帧 | **98 / 127,329** |
| 首个 mismatch | **0 个 shard 有 mismatch**（`relabel.json: shards_with_mismatch=[]`） |
| `displayScore` 覆盖率 | 100.0% |
| 滚动帧（`displayScore != globalsScore`） | 66,298 / 127,329 = **52.07%** |
| 其中「本帧刚动过」（`displayStale==0`） | `51,834（40.7%）` |
| 其中「冻结未动」（`displayStale>0`） | `14,464（11.4%）` |
| `gameOverFlag==1` 帧 | 43,075（33.8%，最后一命死亡后 retry 菜单期，HUD 不再滚） |
| `displayScore` 下降帧（run 内） | **0** |

### 3.2 新旧对比（同一批像素、同一 reader）

见 §2.2 表。要点：

- **旧（`globals.score` 标签）**：逐位 73.86%、整数 21.80%、符号 76.27%，与 EXP-P2-004 完全一致。
  其中 settled 子集（61,031 帧）整数 45.47%/符号 99.89% —— 连 settled 都读不出整数，正是裁剪错位。
- **新（`displayScore` 标签 + 修裁剪 + 重训 reader）**：逐位 `100.00%`、整数 `100.00%`、符号 `100.00%`。
- 逐位（左→右，重训 reader）：`100.00 / 100.00 / 100.00 / 100.00 / 100.00 / 100.00 / 100.00 / 100.00 / 100.00 %`。

### 3.3 判定

- **≥99% 符号准确率：`达标`**（`100.00%`，LB `99.997%`）。
- **但负类样本仍为 0**：`displayScore` 与 `globals.score` 一样在 run 内从不下降（全仓只有 `+=`、钳位、关卡/GameOver 置 0），
  所以「score_down 正例」这一门**在当前数据上依然不可测**。本轮能说的是：在 `same/up` 两类上，符号准确率
  `100.00%`（down=0/same=75,103/up=52,226）；不能声称"负类也达标"。
- 与 EXP-P2-004 不同，这次 score 门的**可测量部分已经达标**；剩余的唯一缺口是"数据里没有负类"，属采集/口径问题，
  不是模型能力问题（要采负类，必须让 run 跨过关卡重置或 Game Over 置 0 之后的帧）。

---

## 4. death 门口径：数据化建议（不改标签口径）

统计脚本 `.tmp_v400_hud/death_gate.py`，产物 `.tmp_v400_hud/death-gate.json`；全部只读已存 shard 的 labels + HUD 平面。

### 4.1 可观测性定义

一个 death 事件"在 HUD 上可观测" ⇔ 事件帧之后 10 帧内 `lives` 读数发生递减（这是 HUD 侧唯一的 death 信号，且实测延迟 +4/+10）。

### 4.2 计数（与 EXP-P2-004 对齐）

| 范围 | death 事件 | 可观测 | 不可观测 | 可观测占比 |
|---|---|---|---|---|
| 全库（194 shard / 358,601 标签行） | **530** | **342** | **188** | **64.53%** |
| test（98 shard，**index 口径 127,329 帧**） | 270 | 187 | 83 | 69.26% |
| test（同一批 shard 的全部标签行，多 2 帧尾） | 272 | 188 | 84 | 69.12% |
| train+val | 213 | 133 | 80 | 62.44% |

与 EXP-P2-004 的 342/530 = 64.5%、188 个不可观测**完全一致**；test 的 270/69.26%/98.94% 也与 EXP-P2-004 的 G2 表逐位一致
（index 口径）；多出的 2 个事件来自未入索引的尾部 shard，单独列出不混用。

### 4.3 限定口径 vs 不限定口径（现有 HUD 差分检测器，test）

| 口径 | 召回 | 精确 | 说明 |
|---|---|---|---|
| 不排除（index 口径 270 事件） | **69.26%**（187/270） | **98.94%**（187/189） | 与 EXP-P2-004 完全一致 |
| **限定为有 HUD 变化者（index 口径 187 事件）** | **100.00%**（187/187） | **98.94%**（187/189） | 漏检 = 0 |
| 不排除（全部标签行 272 事件） | 69.12%（188/272） | 98.95%（188/190） | 含 2 个尾部帧事件 |
| 限定为有 HUD 变化者（188 事件） | **100.00%**（188/188） | **98.95%**（188/190） | 漏检 = 0；2 个预测未在 +10 窗口内匹配到事件 |
| 全库同口径 | 100.00%（342/342） | 93.19%（342/367） | train+val 的精确率低一些（91.10%） |

**结论**：在"排除 `lives==0` 的 35.5%"这个限定口径下，现有检测器**召回 100.00%、精确 98.94%（index 口径）/98.95%（全标签行）**；
召回不再是问题，精确率的 2 个未匹配预测需要人工核对（多为事件位与扣命间隔 >10 帧的边界情况）。

### 4.4 那 188 个事件能否用其它可观测信号检出？——**不能（HUD 侧）**

对 188 个不可观测事件逐事件检查其后 10 帧（187 个可查，1 个落在未入索引的尾 shard）：

| 候选信号 | 命中 |
|---|---|
| `gameOverFlag` 变 1 | **0 / 187** |
| score 重置（>0 → 0） | **0 / 187** |
| `bombs` 变化 | **0 / 187** |
| `lives` 变化 | **0 / 187** |
| HUD `lives` band 像素差（相对事件帧） | **全 0**：mean/median/p95/max 全 0.0，nonzero_fraction=0.0 |
| HUD `bombs` band 像素差 | **全 0**：同上 |
| HUD `score` band 像素差 | mean 10.58 / nonzero 78.6% —— 但对照帧同样在动（mean 5.34 / nonzero 45.9%），是滚动计数器的一般运动，**不是 death 特异信号** |

即：这 188 个事件发生时，HUD 的 `lives`/`bombs` 两个数字区在 +10 帧窗口内**一个像素都没变**，
game over 画面、score 重置、炸弹变化也都没有出现。**HUD 侧不可观测**；唯一可能的信号只剩 playfield
（玩家爆炸/回位），那是 EXP-P2-003 v0 走过的路（test 事件召回 1.04%），不属于本口径，也不在 G2 的 HUD 事件头范围内。

### 4.5 建议（不改 death 标签口径）

1. **如果 G2 的 death 门保持不变**：结构上限 64.5%（全库）/69.1%（test），任何只读 HUD 的检测器都不可能过 99%，
   **不应**通过把口径改窄来"通过"G2 —— 本轮只把它数据化，不当作 G2 通过。
2. **如果允许改口径**：把 death 事件范围限定为「有 HUD 可观测变化者」，并同时声明 `lives==0` 的重复死亡不在范围内。
   在该口径下现有检测器 **召回 100.00%、精确 98.94%（index 口径 187 事件）/98.95%（全标签行 188 事件）**；
   但 n=187/188 的零错误下界只有 98.4%（`1-3/188`），仍不足以过 99% LB —— 需 **≥300 个可观测事件**（`1-3/300=99.0%`），
   本轮 test 只有 187/188 个，所以即使改口径，样本量也不够。
3. **如果既不排除又要 99%**：只能引入 playfield/音频等 HUD 之外的信号，并重新做 v0 那条路（已证明 1.04% 召回，不具备可行性论证）。

---

## 5. 测试

| 文件 | 结果 | 新增覆盖 |
|---|---|---|
| `tests/test_v400_protocol.py` | **14 passed** | `HUD` 解析 7 例：全字段、settled、可选附加字段、缺字段 fail-closed、ERR fail-closed、CHILD 转发解码、`hud()` 只发 `HUD` |
| `tests/test_v400_reward_detector.py` | **39 passed** | 6 例：measured 裁剪几何（x0=8、pitch 13、5px 差）、`hud_crops(score_x0)` 生效、逐位/整数/符号计数、逐 shard 符号（不跨 shard 造假负类）、`load_display_score` 对齐与缺行 fail-closed、99.5% Wilson 下界门 |

命令（WSL，绕过 `.pytest_cache`）：

```bash
PYTHONPATH=/mnt/d/WorkShop/Touhou_RL /root/v400-venv/bin/python -m pytest \
  -p no:cacheprovider --basetemp .tmp_v400_hud/pytest -q \
  tests/test_v400_protocol.py tests/test_v400_reward_detector.py
```

## 6. 命令与退出码

```bash
# 1) 补丁 + 构建（build/rl_score）
python3 /mnt/d/WorkShop/Touhou_RL/.tmp_v400_hud/patch_hud.py          # PATCH_EXIT=0
sh /mnt/d/WorkShop/Touhou_RL/.tmp_v400_hud/build_score.sh              # cmake_exit=0 BUILD_EXIT=0
# 2) HASH 前 100 帧一致性
xvfb-run -a /root/v400-venv/bin/python .tmp_v400_hud/verify_engine.py parity --frames 100   # 0
# 3) 实况 2000 帧 displayScore↔像素（旧 reader 与重训 reader 各一次）
xvfb-run -a /root/v400-venv/bin/python .tmp_v400_hud/verify_engine.py capture --frames 2000  # 0
xvfb-run -a /root/v400-venv/bin/python .tmp_v400_hud/verify_engine.py capture --frames 2000 \
  --checkpoint /root/Touhou_RL/.tmp_v400_hud/reader-display-x8.pt --score-x0 8 \
  --out /root/Touhou_RL/.tmp_v400_hud/capture-fixed                                  # 0
# 4) test 集重采 displayScore（98 shard / 127,329 帧，3 引擎）
sh /mnt/d/WorkShop/Touhou_RL/.tmp_v400_hud/run_relabel.sh              # RELABEL_EXIT=0
# 5) train 集重采（重训 reader 用）
sh /mnt/d/WorkShop/Touhou_RL/.tmp_v400_hud/run_relabel_train.sh        # RELABEL_TRAIN_EXIT=0
# 6) 重训 reader（x0=8 + displayScore 标签）
/root/v400-venv/bin/python .tmp_v400_hud/train_display_reader.py       # 0
# 7) score gate 矩阵
sh /mnt/d/WorkShop/Touhou_RL/.tmp_v400_hud/run_score_gate.sh           # 0（A/B/C 三次 score-gate 均 0）
# 7b) 分场景拆分（settled / active / frozen / retry-menu）
/root/v400-venv/bin/python .tmp_v400_hud/score_splits.py               # 0
# 8) death 门口径
/root/v400-venv/bin/python .tmp_v400_hud/death_gate.py                 # 0
# 9) 测试
#    §5 的两条 pytest 命令                                              # 0 / 0
```

## 7. 范围与限制（如实）

1. **score_down 负类仍为 0**：`displayScore` 与 `globals.score` 一样在 run 内不下降；本轮只能对 `same/up` 宣称 ≥99%。
   要补负类必须采到跨关卡/Game Over 重置的帧（本轮 127,329 帧里 `displayScore` 下降 0 次）。
2. **test 集 33.8% 的帧是 retry 菜单期**（`gameOverFlag=1`，HUD 冻结）：这些帧对读数很容易，已在 §2.4/§3.1 单列，
   不把它们混进"活跃滚动"的结论里。
3. **death 口径未改**：§4 只给数据化建议；G2 的 death 召回门在现行口径下仍不达标（test 69.26% / 结构上限 64.5%）；
   限定为「有 HUD 可观测变化者」后召回到 100%，但那是改口径，**不当作 G2 通过**。
4. `HUD` 是追加式命令：不进 25 字段、不进 SHM、不改协议版本；旧客户端解析前缀不受影响。
5. 重训 reader（`reader-display-x8.pt`）是完整 reader：lives/bombs/score 三个头都在 train split 的 `displayScore` 标签上重训过
   （同一份训练脚本），但本轮 G2 结论只用它的 **score 头**；lives/bombs 的 100% 仍以 EXP-P2-004 的 `v1.pt` 为准（未受影响）。
   重训产物是实验产物，**未替换** `.tmp_v400_det2/v1.pt`。
