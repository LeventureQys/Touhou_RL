# EXP-P1-016 — 死亡分析工具：6 个死亡点的 3 秒回放、GIF 与死因分类

- **日期**：2026-11-09
- **状态**：**完成**（工具落地、6/6 死亡点真实跑通、63 项单测通过）
- **上游**：[EXP-P1-012](EXP-P1-012-g1-evaluation.md)（G1：PSN 8 面中 6 面各死 1 次，死亡帧 21,308 / 28,616 / 12,908 / 9,172 / 13,484 / 7,652）、[EXP-P1-013](EXP-P1-013-teacher-obs-dataset.md)（教师逐决策 JSONL）、[EXP-P1-014](EXP-P1-014-D0-dual-channel.md)（shadow/env 双通道同轨迹前提）
- **产物**：`tools/rl/v400_death_analysis.py`（新增）、`tests/test_v400_death_analysis.py`（新增，63 项）、`.tmp_v400_death/`（6 × `death.json` + 6 × `replay.gif` + 逐帧 PNG + `summary.md/json/csv`）
- **一句话**：把「教师 v1 在第 3 面之后每面死 1 次」从一句判断变成可复核的证据——每次死亡都有 180 帧真实像素回放、一段 GIF、和一份按固定规则算出来的多标签死因，并据此给出教师 v2 的具体修改建议。

---

## 1. 目标与判定口径

WP4 要求三件事，缺一不算完成：

| # | 要求 | 本卡怎么算「做到」 |
|---|---|---|
| 1 | 死亡前 3 秒（180 帧 @60fps）逐帧回放 | 真实重放该局，`death_frame-180 … death_frame` 每帧 `STEP 1` 后读 `SHM` 640×480 RGBA → PNG；同时逐帧采 `PLAYER`/`HAZARDS`/`EVENTS`/`STATE` |
| 2 | 渲染出的视频 | 把同一批索引帧写成 GIF89a（`replay.gif`），用 Pillow 能解回来 |
| 3 | 死因分类 | 6 条**固定规则** + **多标签** + **优先级主因**，规则与阈值原样写进 `death.json`，可复算 |

**严格口径（写死在这里，防止后面滑）**：

1. **死亡帧只认引擎计数**：`STATE.deaths`（`f6`，float bits）上升，或 `EVENTS` 的 `0x4 PLAYER_DEATH_COMMITTED`。`PLAYER_HIT(0x2)`、`playerState=2` 是**前置信号**，可以早 2 帧报到，但**不许当死亡帧**（本卡 6 局里 Stage5 实际就是这样：hit@9168，commit@9170）。
2. **死亡帧的精度是「决策」不是「帧」**：教师只看一个决策（4 帧）提交完之后的死亡计数，所以 JSONL 只能把死亡钉到某个 4 帧决策内。工具因此要求重放出的 commit 帧落在 `[death_frame-4, death_frame]`，再**原样记录**偏移量（见 §5 的 `offset` 列），不四舍五入、不假装精确到帧。
3. **贴底带 = 死亡前 60 帧平均 `py > 400`**（`py` 取自逐帧 `PLAYER`，不是轨迹推算）。
4. **不可观测就写不可观测**：`HAZARDS` 报 0 条时不许编「最近致死物」；激光未激活时记 `gap=null`；Boss 不在场时 `boss_pattern` 不成立。
5. **帧数与符号都来自报告**：`--death-frame` 必须等于 JSONL 报告里的 `survival_frames`，不等就 fail-closed，不猜。

---

## 2. 复现方法：为什么不重跑教师

`PARK` 只在 `shadow` 角色可用，而像素只在 `env` 角色可读（EXP-P1-014 §1 已把这条**角色分裂**写成 P3 硬阻塞），所以「一边决策一边截像素」在单进程里做不到。本工具的做法是把**输入**而不是**决策**搬过来：

```
教师 JSONL（已有，.tmp_v400_g1/psn/psn_stage*.jsonl）
  └─ 每个决策的 step_masks = ["0x0041x2", "0x0041x2"]   ← 该决策提交的 (掩码, 帧数)
       └─ 逐决策拼接 = 整局逐帧输入带（tape），长度 = survival_frames
            └─ env 角色引擎：SHM + READBACK 1 + RESET <stage> 1 0 1 0
                 └─ 180 帧窗口内每帧 STEP 1 → 读 SHM 像素 → PNG
```

三点理由，都写进了工具 docstring：

1. **可复现性有据**：`RESET` + 逐帧掩码相同 ⇒ 轨迹相同（EXP-P0-019 的 167 个检查点、EXP-P0-049 的 5,000 帧零差异）。本卡不满足于引用，**把重放做了门**：重放必须在记录的位置真的死一次，否则 exit 3 且不写 `death.json`。
2. **教师不必重跑**：JSONL 里 `step_masks` 已经是权威进程实际吃到的掩码序列（`v400_teacher.py` 的 child 路径 `commit_block` 产物），重跑教师只会用 8,887 s 机时换一份相同的决策。
3. **tape 自身带校验**：每个决策的 `frame` 字段必须等于累计帧数；总帧数必须等于 `survival_frames`，也必须等于 `decisions × committed_frames_per_decision`。任一条不成立 → fail-closed。

引擎门：二进制 SHA-256 必须是 `f172399f61b75baa8e62be46041ef756a710759f439e7fb07111f4747c78264e`（与 EXP-P1-011/012/013 同一个 `rl_hazard2` build），不匹配 → exit 4，不产出任何分析。数据目录用 `--data-template /root/v400-data/g1/inst0` 复制小文件、符号链接 `th08.dat`/`thbgm.dat`（**绝不**改动原目录，也不碰引擎/协议/第三方）。

---

## 3. 工具与产物

| 文件 | 作用 |
|---|---|
| `tools/rl/v400_death_analysis.py`（新增，1,638 行） | `run`：replay+捕获+分类；`summary`：汇总表。纯逻辑（tape 解析、分类、GIF、汇总）与引擎逻辑（`capture`）在同一文件里但**分层隔离**：引擎相关 import 全部在 `capture()` 内，所以 Windows 主机能直接跑单测 |
| `tests/test_v400_death_analysis.py`（新增，571 行） | **63 项**：tape 重建 9 项、分类规则 20 项、EVENTS 位 3 项、GIF 9 项、PNG 2 项、汇总 6 项、fail-closed 11 项、语料一致性 1 项 |
| `.tmp_v400_death/psn_stage<N>/frames/frame_XXXXXX.png` | 逐帧 640×480 RGBA PNG（每帧带 `png_sha256` + `rgba_sha256`） |
| `.tmp_v400_death/psn_stage<N>/replay.gif` | 同一批帧的动画 GIF（默认 5 cs/帧 = 20 fps，`--gif-delay-cs` 可调） |
| `.tmp_v400_death/psn_stage<N>/death.json` | 死亡帧、末帧自机位置、最近致死物类型/距离、贴底判定、bomb、`frameEventFlags` 位分解、多标签 + 主因 + 规则原文 + 每个标签的证据数字 |
| `.tmp_v400_death/psn_stage<N>/frames.jsonl` | 逐帧记录：`mask`、`shm_frame`、`shm_state_hash`、`player`、`hazards`（含 `verdict` 最近致死物与 `laser` 几何）、`boss`、`events`、`state`、`teacher`（该帧所属决策的 `chosen_plan/margin/score`） |
| `.tmp_v400_death/psn_stage<N>/decisions.jsonl` | 窗口内每个决策的 `first_action/duration/margin/score/geo_score[18]` |
| `.tmp_v400_death/summary.md` / `summary.json` / `summary.csv` | 6 局汇总表 + 标签计数 + 假设检验 |
| `.tmp_v400_death/run_all.sh` / `run_one.sh` / `batch.log` | 逐局命令行、退出码、起止时间（2 并发，见 §10） |

命令行（与任务规格一致）：

```bash
/root/v400-venv/bin/python tools/rl/v400_death_analysis.py \
  --replay .tmp_v400_g1/psn/psn_stage2.jsonl --death-frame 21308 \
  --stage 2 --seed 1 --out .tmp_v400_death/psn_stage2 \
  --data-template /root/v400-data/g1/inst0
# 汇总
/root/v400-venv/bin/python tools/rl/v400_death_analysis.py \
  --mode summary --summary-root .tmp_v400_death
# 工具自带的 6 个死亡点清单（防止手抄错帧）
/root/v400-venv/bin/python tools/rl/v400_death_analysis.py --list-deaths
```

`--stage` / `--seed` 只用于**校验**：与 JSONL 报告不一致就 exit 3；工具从不假设重放用哪一面。

---

## 4. 死因分类规则（固定、写在代码里、随结果复制进 JSON）

判决函数 `classify_death()` 是纯函数（无引擎、无 IO），输入是窗口内的三条序列：`py_series`（逐帧自机 y）、`boxes_series`（逐帧致死盒数）、`laser_gaps`（逐帧到最近**激活**激光的距离），加死亡帧的 `bombs` 与 Boss 几何。默认阈值在 `Rules` 里，**在分析这 6 个死亡点之前就固定**。

| 标签 | 规则（`bottom` 等阈值的默认值） | 为什么这样定 |
|---|---|---|
| `bottom_pin` | 死亡前 **60** 帧的 `mean(py) > 400`（`py` 严格大于阈值） | 任务口径；60 帧 = 1 s，足够跨过一次走位而不被单帧抖动骗到 |
| `bomb_exhausted` | 死亡帧 `bombs <= 0` | 「有 bomb 没用」和「bomb 用完」是两件事，只有 `=0` 才算后者 |
| `bullet_wall` | 死亡前 **10** 帧中至少 **6** 帧的致死盒数 `>= 40`（致死盒 = `geometry.hazard_box_count`：`state==FIRED` 且未被 ECL 关碰撞的子弹 + `flags1` 带碰撞位的敌机本体） | 「弹幕墙」要的是**持续**密度而不是某一帧尖峰：取多数帧（6/10）而不是任意一帧，避免单帧爆发误判 |
| `laser` | 死亡前 **10** 帧内任一帧存在**激活**激光（`state==1` 且 `in_use!=0`），且其到自机 hurtbox 中心距离 `<= 32 px` | 距离用 `geometry.laser_hits`（`Player::CalcLaserHitbox` 的逐字实现），32 px ≈ 8 个自机半身位，够抓住"擦着激光死" |
| `boss_pattern` | 死亡帧有 Boss 行（`flags1 & ENEMY_FLAG_BOSS`）且到 hurtbox 中心距离 `< 448 px` | 448 = 整个 playfield 高度。这条是**场景**标签（"死在 Boss 战里"），它不声称"Boss 的某张符卡杀了你"——那需要符卡 ID，引擎没有导出 |
| `unknown` | 上面 5 条全不成立 | 宁可写 `unknown`，也不硬凑一个类别 |

**多标签**：一条死亡可以同时命中多条（实际 6 局全部命中 3 条）。`labels` 按优先级排序输出。

**主因（优先级）**：`laser > bullet_wall > boss_pattern > bottom_pin > bomb_exhausted`。

排序原则写清楚：**即时致死机制 > 场景 > 促成条件**。激光和弹幕墙是"当场把你打死的东西"；Boss 在场说明你在哪场战斗里；贴底和 bomb 耗尽是**状态**——它们不会自己杀人，所以永远不能当主因。这条顺序保证「6 局都贴底、都没 bomb」不会自动被写成「死因：贴底」。

`death.json` 同时写入 `rules`（阈值）、`rules_text`（人话规则）和 `classification.evidence`（每个标签的实测数字，例如 `bullet_wall.frames_at_or_above=10`、`boxes_series=[…]`、`laser.nearest_gap_px`、`bottom_pin.mean_py`），任何人可以拿这 6 个数字复算出同样的标签。

---

## 5. 6 个死亡点的结果

（结果表见 §5.2；每局的 `death.json` 都在 `.tmp_v400_death/psn_stage<N>/`。）

### 5.1 共同事实（全部来自引擎）

- 6 局全部 `RESET <stage> 1 0 1 0`（normal、team 0、seed 1、practice 0），与 EXP-P1-012 §6.1 命令行一致。
- 6 局重放**全部复现了记录中的死亡**：commit 帧与记录帧的偏移是 −3 / +0 / −2 / −2 / −2 / −3（全部落在记录决策的 4 帧窗口内）。没有一局需要"调一下才对"，也没有一局的偏移超出容差。
- 每局都产出 178–181 张 640×480 PNG + 1 个 GIF（12.9–19.4 MB）+ 1 份 `death.json` + `frames.jsonl`（逐帧 `PLAYER`/`HAZARDS`/`EVENTS`/`STATE`/教师计划）+ `decisions.jsonl`（含 `geo_score[18]`）。
- 6 局的 `PLAYER_HIT` 都早于 commit 1–4 帧（Stage3 21303→21305、Stage5 9168→9170、Stage6A 13480→13482、Stage6B 7647→7649），这正是 §1.2 那条口径要处理的现实情况。

### 5.2 逐局结果

「死亡帧」是重放实测的 **commit 帧**（`STATE.deaths` 上升那一帧），`offset` 是它相对记录帧 `--death-frame` 的偏移；`帧数` 是 PNG 序列长度（回看 180 帧，裁掉 commit 之后的帧）。

| 面 | 死亡帧 | offset | 帧数 | 主因 | 全部标签 | py | mean_py(60) | bombs | 致死盒 median/max | 激光 gap | Boss 距离 | EVENTS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Stage3 | 21305 | -3 | 178 | `bullet_wall` | `bullet_wall` + `boss_pattern` + `bottom_pin` + `bomb_exhausted` | 429.2 | 428.6 | 0 | 99 / 105 | — | 311 | `0x0404` |
| Stage4A | 28616 | +0 | 181 | `bullet_wall` | `bullet_wall` + `boss_pattern` + `bottom_pin` + `bomb_exhausted` | 432.0 | 410.0 | 0 | 682 / 697 | — | 373 | `0x0404` |
| Stage4B | 12906 | -2 | 179 | `bullet_wall` | `bullet_wall` + `boss_pattern` + `bottom_pin` + `bomb_exhausted` | 432.0 | 428.4 | 0 | 401 / 408 | — | 413 | `0x0404` |
| Stage5 | 9170 | -2 | 179 | `bullet_wall` | `bullet_wall` + `bottom_pin` + `bomb_exhausted` | 428.0 | 428.9 | 0 | 137 / 138 | — | — | `0x1404` |
| **Stage6A** | 13482 | -2 | 179 | **`laser`** | `laser` + `bullet_wall` + `boss_pattern` + `bottom_pin` + `bomb_exhausted` | 416.4 | 410.7 | 0 | 99 / 99 | **0.0** | 288 | `0x0406` |
| **Stage6B** | 7649 | -3 | 178 | `bullet_wall` | `bullet_wall` + `boss_pattern` + `bomb_exhausted` | **84.5** | **64.0** | 0 | 408 / 458 | — | **102** | `0x0404` |

补充证据（都在各局的 `death.json` / `frames.jsonl` 里）：

| 面 | 末帧最近致死物 | 窗口内最近 gap | 教师死前 margin | 教师 first_action/duration | 致死盒序列（最后 10 帧） | PNG / GIF |
|---|---|---|---|---|---|---|
| Stage3 | bullet #1373 gap 7.12 px | 0.74 px @21302 | **5.93 px** | 3 / 2 | 105,105,101,100,99,98,97,95,94,94 | 178 / 15.4 MiB |
| Stage4A | bullet #566 gap 0.53 px | 0.28 px @28613 | **0.90 px** | 7 / 1 | 697,692,688,688,682,682,679,675,671,671 | 181 / 14.1 MiB |
| Stage4B | bullet #83 gap 11.49 px | 1.06 px @12897 | 11.23 px | 10 / 2 | 405,403,401,397,395,393,408,402,401,400 | 179 / 18.5 MiB |
| Stage5 | bullet #1099 gap 4.69 px | 0.22 px @9167 | **4.00 px** | 1 / 1 | 136,136,136,137,137,137,138,138,137,137 | 179 / 12.6 MiB |
| Stage6A | **laser #15 gap 0.00 px** | 0.00 px @13480 | 16.00 px | 11 / 1 | 86,86,99,99,99,99,99,99,99,99 | 179 / 14.3 MiB |
| Stage6B | bullet #416 gap 5.75 px | 1.06 px @7624 | **4.00 px** | 14 / 1 | 411,408,404,401,398,394,392,458,455,452 | 178 / 12.3 MiB |

`EVENTS` 位分解（每局都写进 `death.json` 的 16 位表）：5 局是 `0x0404 = PLAYER_DEATH_COMMITTED | EFFECT_SPAWNED`；Stage5 多一个 `ENEMY_SPAWNED`（`0x1404`）；Stage6A 是 `0x0406`，**独有** `0x2 PLAYER_HIT` 且死亡前 3 帧都有 hit（`frames_with_hit=3`）。Stage6B 在 `0x0404` 之前的两帧出现过 `0x2602 = PLAYER_HIT | PLAYER_DYING_STARTED | EFFECT_SPAWNED | PLAYER_GRAZED`，然后是一帧干净的 `0x0000` —— 也就是说 hit 与 commit 之间隔了 2 帧，这正是 §1.2 那条口径存在的理由。

### 5.3 重放可信度：与原始 JSONL 逐点比对

本卡不接受"看起来对"。对每一局，把重放捕获到的每一帧与 JSONL 里**同一决策边界**（决策的 `frame` == 捕获帧）的 `state.px/py` 比对：

```
.tmp_v400_death/verify_replay.py
TOTAL boundary_checks=271 mismatches=0
stage2 (Stage3)  45 checks, 0 mismatch   jsonl_final=(100.520, 429.172)  replay_final=(100.520, 429.172)
stage3 (Stage4A) 46 checks, 0 mismatch   jsonl_final=(16.485, 432.000)  replay_final=(16.485, 432.000)
stage4 (Stage4B) 45 checks, 0 mismatch   jsonl_final=(47.314, 432.000)  replay_final=(47.314, 432.000)
stage5 (Stage5)  45 checks, 0 mismatch   jsonl_final=(49.456, 428.000)  replay_final=(49.456, 428.000)
stage6 (Stage6A) 45 checks, 0 mismatch   jsonl_final=(187.723, 416.444) replay_final=(187.723, 416.444)
stage7 (Stage6B) 45 checks, 0 mismatch   jsonl_final=(353.255, 84.485)  replay_final=(353.255, 84.485)
```

**271 个决策边界、271 个逐位相等的 `px/py`**。因此「重放 = 原局」不是假设，而是这 6 局各自被验证过的事实；`env` 角色与 `shadow` 角色在同 RESET + 同掩码下同轨迹（EXP-P0-019/049）也在这批数据上复现了。

---

## 6. 死因分布与结论

### 6.1 多标签分布（6 局）

| 标签 | 命中局数 | 作为主因 | 命中哪几局 |
|---|---|---|---|
| `bullet_wall` | **6 / 6** | **5** | 全部 6 局（10/10 帧 ≥ 40 致死盒，无一例外） |
| `bomb_exhausted` | **6 / 6** | 0 | 全部 6 局（死亡帧 `bombs ≡ 0`） |
| `bottom_pin` | **5 / 6** | 0 | Stage3 / 4A / 4B / 5 / 6A |
| `boss_pattern` | **5 / 6** | 0 | Stage3 / 4A / 4B / 6A / 6B（Stage5 是 Boss 离场后的道中） |
| `laser` | **1 / 6** | **1** | 只有 Stage6A（gap = 0.0 px，51 条激光） |
| `unknown` | 0 / 6 | 0 | — |

**主因排序**：`bullet_wall` 5 ＞ `laser` 1 ＞ 其余 0。也就是说：**6 次死亡里 5 次是"被弹幕墙压死"，1 次是"被激光贴身"**，两次都不是"贴底"或"没 bomb"本身。

### 6.2 与 EXP-P1-012「贴底 + bomb 用完」假设的一致性

**结论：一半成立，一半被本卡修正。**

| 假设的一半 | EXP-P1-012 的说法 | 本卡实测 | 判定 |
|---|---|---|---|
| **bomb 用完** | 6 次死亡全部在 3 颗 bomb 用尽之后（`bombs_spent=3.0`） | 6/6 死亡帧 `bombs=0`、`STATE.bombs_remaining=0` | ✅ **完全一致**（但注意：这是"预算耗尽"，不是"无用 bomb"——见 §7.3） |
| **贴底** | §6.4 写着"死亡位置 `py≈416–432`（游戏区底边）" | 5/6 贴底（`py` 416.4–432.0，`mean_py` 410.0–428.9）；**Stage6B 死在 `py=84.5`、窗口均值 64.0，是屏幕顶部** | ⚠️ **部分不成立** |

Stage6B 的反例是硬证据，不是解释问题：

- 该局 JSONL 自己的最后一帧 `state.px/py = (353.255, 84.485)`，与本卡重放逐位相同（§5.3），所以**原局本来就死在顶部**，EXP-P1-012 §6.4 的那句概括对这一局不成立。
- 死亡时 Boss（slot 0，life 926）在 `(251.7, 89.6)`，距自机 **101.7 px** —— 自机是**主动压到 Boss 本体旁边**被打死的，不是"贴底保守"。
- 因此教师 v1 的死法有**两种**：贴底摆烂（5 局）＋ 贴近 Boss 本体（1 局）。任何只针对"贴底"的修改都覆盖不到第二种。

另有一处需要修正的旧前提：EXP-P1-012 引用"P1 语料中 0 条激光"，而 **Stage6A 死亡帧有 51 条激光**（`lasers=51`，5/10 帧 gap ≤ 32 px，最近 gap = 0.0 px）。激光不是"这批数据里不存在"的危险，它已经杀过教师一次。

### 6.3 一句话结论

> 教师 v1 的死亡不是"运气不好"，而是**几何余量归零**：死亡前 10 帧每一帧都有 40 个以上的致死盒（Stage4A 高达 697 个），最近致死物在死亡窗口内一度贴到 **0.22 px**；教师搜索出的"最优"动作，其 `margin` 在 5/6 局只有 **0.9–11.2 px**。bomb 在死亡前就已经花光（6/6），贴底是其中 5 局的**促成条件**而非主因（Stage6B 甚至死在顶部）。

---

## 7. 给教师 v2 的建议

每条都指到 §5/§6 的具体数字，不是泛泛而谈。

### 7.1 贴底带惩罚（针对 5/6 局）

- **现状**：`Weights.edge_penalty=1.0`、`high_penalty=3.0`，但 `bottom_pin` 仍占 5/6，且 Stage4A/4B 的 `py` **恰好是 432.000**（playfield 下边界 `border_y=(16,432)`）——自机被自己的边界夹在墙上。
- **建议**：加一条**硬约束**而不是软罚分：候选动作的终点 `py > 400` 时**直接剪掉**（与 `bottom_band_y=400` 同一口径），只有 `py ≤ 400` 的解才进 beam。软罚分已被证明压不住（`edge_band=24` 只覆盖 408–432，而死亡点正在 428–432）。若担心道中必须贴底才能吃道具，可把它做成"只在 `record_items=False` 模式下硬剪"（当前配置正是 `record_items=False`）。
- **同时**：Stage3/4A/4B 的 `mean_py` 已经 410–429，说明不是"某一帧被挤下去"，而是**整段都在带内**。所以惩罚要作用于 plan 的**整条轨迹**（`horizon=48` 帧内任一点越界即剪），而不是只罚终点。

### 7.2 bomb 保留策略（针对 6/6 局）

- **现状**：6/6 死亡帧 `bombs=0`；教师的 `bomb_active` 规则允许在"bomb 可用且能改善评分"时用掉，结果 3 颗在死亡前全部花光，死亡瞬间没有任何兜底。
- **建议**：把 bomb 从"清屏资源"改成"保命资源"：
  1. **保留 1 颗**：`bombs_remaining == 1` 时禁止主动 bomb（只在 `margin < 4 px` 即"这一帧必死"时允许）。
  2. **触发条件改成 margin 驱动**：现在 bomb 决策看的是"用 bomb 后评分变不变好"，v2 应该看"不用 bomb 的下一个决策 `margin` 是否 < 阈值"——§5.2 里 4/6 局的死前 `margin` 是 0.9–5.9 px，这正是该按 bomb 的时刻。
  3. **记账**：把 `bombs_spent` 与"按 bomb 时的 margin"一起记进 JSONL，才能事后判断策略好坏（本卡发现现有 JSONL 里 `bomb=0/1` 有记录，但没有"当时 margin 是多少"的联动）。

### 7.3 Boss 末段保守度（针对 5/6 局 + Stage6B 反例）

- **现状**：`boss_pattern` 5/6 命中；Boss 距离 102–413 px。Stage4A 死时 Boss 只剩 **17 HP**（EXP-P1-012），Stage6A 死时 Boss **2192 HP**、Stage6B **926 HP**。
- **建议**：
  1. **Boss 存活时提高 `high_penalty` / 加"靠近 Boss 本体"惩罚**：Stage6B 的自机在 `py=84.5`、距 Boss 本体 102 px，而 Boss body 也是致死盒（`enemy_boxes` 已把它算进 458 个致死盒里），教师的评分显然没把这个 body 的代价算够。
  2. **按 Boss 血量分段调保守度**：Stage4A 已经打到 17 HP，此时"再苟 30 秒"的收益远大于"抢输出"；建议 v2 在 `boss_life < 5%` 时把 horizon 拉长、margin 权重加倍，避免"差 17 HP 却死了"。
  3. **末段（`boss_life` 低或阶段切换时）禁止贴边/贴顶**：Stage6B 的 `py=84.5` 与 Stage6A 的 `py=416.4` 是两个极端，共同点是**都贴着一条边走**。

### 7.4 激光必须进 hazard tape（针对 Stage6A）

- **现状**：`geometry.laser_hits` 已经在几何里，但 `laser_hits` 的 docstring 明确写着 `startTime`/`duration`/`hitboxStartTime`/`hitboxEndDelay` **未导出**，所以只能把 `state==ACTIVE` 当致死；Stage6A 死亡帧 **51 条激光、gap 0.0**，说明这个近似在真实 Boss 战里不够。
- **建议**：v2 的 `CHILD HAZARDS` 增加激光的 timer 字段（引擎侧是只读导出，不改协议语义的话可以先加一个 `LASERS <n>` 的扩展页），否则"激光贴脸"这类死亡在规划期就是盲区。

### 7.5 优先级建议（若 v2 只能改一处）

按本卡的证据强度排序：**① 贴底硬剪（5/6 局直接相关，改动最小、可立即验证）→ ② bomb 保 1 颗（6/6 局相关，但只在 4/6 局真正能救命）→ ③ 激光 timer 导出（1/6 局，但属于"永远看不见"的盲区）**。Boss 保守度排第三，因为它需要先有"Boss 本体代价"的量化，而当前评分里这个代价只有 `enemy_boxes` 一条路径。


---

## 8. 测试

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests/test_v400_death_analysis.py
# 63 passed
```

覆盖内容（按任务点对号）：

| 任务要求 | 对应测试 |
|---|---|
| 分类规则 | `test_all_rules_fire_on_a_textbook_bottom_death`、`test_unknown_when_nothing_fires`、`test_labels_are_multi_label_and_ordered_by_priority`、`test_primary_never_becomes_a_contributing_condition`、`test_classifier_records_the_rule_text_and_thresholds` |
| 贴底判定 | `test_bottom_pin_boundary_is_strictly_greater_than_threshold`（400.0 不算、400.01 算）、`test_bottom_pin_uses_the_last_60_frames_only`、`test_bottom_pin_not_fooled_by_a_short_average_over_the_whole_window` |
| 边界 | `test_bullet_wall_count_threshold_is_inclusive`（40 算、39 不算）、`test_bullet_wall_needs_six_of_ten_frames`、`test_laser_needs_an_active_laser_inside_the_radius`（31.9 / 32.1）、`test_laser_lookback_is_ten_frames`、`test_boss_pattern_requires_a_live_boss_inside_the_radius`（447 / 449）、`test_decision_for_frame_boundaries`、`test_classifier_handles_empty_series_without_crashing` |
| GIF 生成 | `test_stdlib_gif_is_a_real_gif89a`、`test_stdlib_gif_round_trips_through_pillow`（Pillow 解码回 RGB，逐像素相等）、`test_stdlib_gif_round_trips_a_large_noisy_frame`（跨多个 LZW 码宽台阶）、`test_pillow_writer_matches_the_stdlib_palette`、`test_write_gif_prefers_pillow_and_falls_back`、`test_stdlib_gif_rejects_an_empty_animation`、`test_stdlib_gif_rejects_mismatched_frame_sizes` |
| fail-closed | `test_load_replay_fails_closed_*`（6 项：缺文件/缺 report/无死亡/survival 不符/决策帧数漂移/JSON 损坏）、`test_run_one_refuses_*`（stage 不符、seed 不符、death-frame 不符）、`test_capture_refuses_*`（帧超出 tape、回看越界、引擎缺失）、`test_prepare_data_dir_refuses_an_incomplete_template`、`test_main_returns_usage_without_required_arguments`、`test_main_summary_mode_*` |

**没有测到的部分（写明）**：`capture()` 的真实引擎分支（`SHM`/`READBACK`/`xvfb` 路径）不在单测里，它需要 WSL 引擎和 `/dev/shm`；这部分由 §5 的 6 次真实运行与 §9 的退出码覆盖。单测里对它的覆盖只有「拒绝坏输入、不启动进程」这一半。

GIF 的实现选择也写在测试里：Windows 主机有 Pillow ⇒ 走 `write_gif_pillow`；WSL 的 venv **没有 Pillow 且不许装**，因此工具自带一个只用标准库的 GIF89a 编码器（`_lzw_encode` + `write_gif_stdlib`），并用 Pillow 反向解码来证明它写的是真 GIF（`test_stdlib_gif_round_trips_through_pillow` 逐像素比对）。**没有安装任何新依赖。**

---

## 9. 命令与退出码（全部真实）

| # | 命令 | 退出码 | 说明 |
|---|---|---|---|
| 1 | `python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests/test_v400_death_analysis.py` | **0** | **63 passed** |
| 2 | `python -m compileall -q tools/rl/v400_death_analysis.py` | **0** | 语法 |
| 3 | `wsl -e bash .tmp_v400_death/run_one.sh 5` | **0** | Stage5 单跑（首次尝试 exit 3，见下表） |
| 4 | `wsl -e bash .tmp_v400_death/run_all.sh 2 3 4 5 6 7` | **0** | **6/6 全部 exit 0**（16:05:46 → 16:20:53，14.1 min，2 并发） |
| 5 | `python tools/rl/v400_death_analysis.py --mode summary --summary-root .tmp_v400_death` | **0** | 写 `summary.{md,json,csv}`（6 行） |
| 6 | `python .tmp_v400_death/verify_replay.py` | **0** | 271 个决策边界、0 不匹配（§5.3） |
| 7 | `python tools/rl/v400_death_analysis.py --list-deaths` | **0** | 6 行死亡点表 |

逐局真实耗时（`run.log` / `batch.log`，墙钟）：

| 面 | 启动 | 结束 | 墙钟 | tape 帧数 | 其中快进 | 其中捕获 178–181 帧 |
|---|---|---|---|---|---|---|
| Stage3 | 16:05:46 | 16:11:52 | 366 s | 21,308 | 315.4 s | 9.4 s |
| Stage4A | 16:05:46 | 16:15:48 | 602 s | 28,616 | 537.0 s | 9.9 s |
| Stage4B | 16:11:54 | 16:16:09 | 255 s | 12,908 | 210.9 s | 11.1 s |
| Stage5 | 16:15:48 | 16:18:17 | 149 s | 9,172 | 114.8 s | 7.9 s |
| Stage6A | 16:16:10 | 16:20:47 | 277 s | 13,484 | 235.3 s | 8.0 s |
| Stage6B | 16:18:17 | 16:20:53 | 156 s | 7,652 | 123.7 s | 8.4 s |

（墙钟 − 快进 − 捕获 ≈ GIF 编码 + 分析写盘。**6 局每局只启动 1 个引擎进程**，`run_all.sh` 限 2 并发；Stage5 在批跑里重跑了一次是为了让 6 份产物来自同一条命令路径，见 §9 的失败清单。）

失败但**保留证据**的尝试（不隐藏）：

| 尝试 | 退出码 | 原因 | 处置 |
|---|---|---|---|
| 首跑 Stage5 | **3** | `the replay died at frame 9168 (event_hit), not at 9172` | 工具把 `PLAYER_HIT` 当死亡帧。**这是工具的口径错误，不是引擎问题** → 改成只认 `STATE.deaths`/`0x4`，hit/dying 降级为前置信号并把「commit 帧必须落在 4 帧决策内」写成显式门（§1.2） |

---

## 10. 局限与声明

1. **死亡帧精度 = 1 个决策（4 帧）**，不是 1 帧。JSONL 里教师只在提交完整个决策后比较死亡计数，所以记录的 `survival_frames` 只能把死亡钉到 4 帧窗口内；工具记录实测 commit 帧与偏移量（§5.2 的 `offset` 列，实测 −3…+0），不假装帧级精确。
2. **回看窗口以记录帧为锚**：窗口起点按 `--death-frame - 180` 取；实测 commit 帧比记录帧早 1–3 帧时，末尾帧被裁到 commit 帧，所以帧数是 178–181 而不是恒定 181（`window.frames` 与 `window.anchored_last_frame` 都写进 JSON）。这让 `mean(py)` 的尾部对齐真实死亡帧，代价是窗口长度少 1–3 帧。
3. **`boss_pattern` 是场景标签，不是符卡归因**：引擎不导出符卡 ID/弹幕来源，所以这条只能说"死在 Boss 战里且 Boss 在 448 px 内"，不能说"死在 X 符卡第 N 波"。
4. **本卡修正了上游卡的两处说法**，两处都有逐位证据：
   - EXP-P1-012 §6.4「死亡位置 `py≈416–432`」对 **Stage6B 不成立**：原局 JSONL 末帧 `state.py=84.485`，重放同为 84.485（§5.3），Boss 距自机 102 px。正确的说法是「6 局里 5 局贴底，1 局压到 Boss 本体」。
   - EXP-P1-012/013 里「P1 语料中 0 条激光」的前提在 **Stage6A 不成立**：死亡帧 `lasers=51`、gap 0.0 px，这是本批唯一的激光主因死亡。
5. **本卡的 2 并发是自限的**：`run_all.sh` 显式限制 2 个 worker（`MAXJOBS=2`），`batch.log` 里每次 launch 都带 PID 与时间戳。跑批期间 `pgrep -xc th08-modern` 读到 8–17，是因为**另一个教师 v2 调参任务**同时开着 6 个 worker 以及它们的瞬时子进程；本工具的引擎进程（命令行 `v400_death_analysis.py … --stage N`）始终只有 2 个，已用 `pgrep -af` 逐条核对。
6. **GIF 不是 mp4**：任务允许 "mp4/gif"，本卡选 GIF，因为 Pillow 已装在主机、WSL 侧用自带编码器，**零新依赖**。代价是文件大（每局 12.3–18.5 MiB）且只有 216+40 色调色板；GIF 的像素是**调色板索引**，不是原始 RGBA——原始 RGBA 以逐帧 PNG（无损）为准，`frames.jsonl` 里每帧都有 `rgba_sha256`。GIF 的保真度也验过：`psn_stage6/replay.gif` 的 179 帧里，首帧与末帧经调色板还原后与对应 PNG **逐像素相同**（`max_diff=0`）。
7. **没有测到的部分**:`capture()` 的真实引擎分支不在单测里（需要 WSL 引擎 + `/dev/shm` + Xvfb），由 §5 的 6 次真实运行、271 点比对（§5.3）与 §9 的退出码覆盖。
8. **没有改引擎/协议/依赖/第三方**，没有重建 build，没有删除任何文件；改动仅限任务许可的四处（工具、测试、本卡、`.tmp_v400_death/`）。引擎 SHA 仍是 `f172399f61b75baa8e62be46041ef756a710759f439e7fb07111f4747c78264e`（工具每次运行都校验，不匹配 exit 4）。
9. **总耗时超出 90 分钟预算**：真实墙钟 16:00→16:21（约 21 分钟纯跑批）+ 开发与测试。超时的原因是 §9 里那次 exit 3 的口径修正（必须在产出任何 `death.json` 之前修好），以及 Stage4A 单局 602 s 的快进。这一点如实记录，不含糊。
