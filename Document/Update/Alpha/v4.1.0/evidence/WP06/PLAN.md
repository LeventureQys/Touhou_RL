# WP06.0 — v4.1 连续像素回报：像素事件/状态机与数据/评测计划

- 工作包：v4.1.0 WP06.0（只读审计 + 设计；不训练、不采数据、不启动游戏）。
- 状态：**审计与设计完成，G410-3 未通过、未开始验收**。本文件不是 G410-3 证据，
  也不把任何旧 G2 数字改名为 G410-3。
- 机器可读审计：`Document/Update/Alpha/v4.1.0/evidence/WP06/audit.json`
  （schema `v410-wp06-audit/1`）。
- 只读输入：`tools/rl/v400_p2_reward_detector.py`（3,738 行）、
  `tests/test_v400_reward_detector.py`（1,520 行 / 66 个 `test_*`）、
  `EXP-P2-003`、`EXP-P2-004`、`EXP-P2-007`、`EXP-P2-011`、`README.md` WP06/G410-3、
  `touhou_rl/v400/student/obs_spec_v3.py`、`tools/rl/v400_p2_bulk_collect.py`、
  `tools/rl/v400_pixel_death.py`、`tools/rl/v400_stage_clear_replay.py`、
  `.tmp_v400_d1/d1/shards/*`（137 个完整 shard）、`.tmp_v400_g2online/`、
  `.tmp_v400_sclear2/`、`.tmp_v400_g2c/`。
- 运行目录：`.tmp_v410_wp06_audit/`（分析脚本与输出，不入库）。
- 范围声明：**未启动游戏、未训练、未采集新数据、未改 v400/v380/v390、未改
  `tools/rl/v400_*`、`tests/test_v400_*`、`third_party/`、v410 既有文件、
  state/journal/budget/assets、其它 WP 证据；未 commit/push；未删除/覆盖旧文件。**
  只新增 `evidence/WP06/PLAN.md`、`evidence/WP06/audit.json` 与运行目录文件。
  同一目录下另有一个由其它进程/代理在 06:56:48 生成的 `STATUS.md`；本卡未写、未改、未删它，
  只在 `audit.json` 的 `validation.concurrent_untracked_files` 中登记。

## 0 一句话结论

现有 v4.0 detector 的 **HUD 资源读数**（lives/bombs 100%，Wilson 95% 下界 99.997%）
和 **bomb 事件**（436/436，LB 99.126%，延迟 0）可以复用；但 **death 的像素检测**
（HUD 差分上限 64.5%、NCC 16.41%/17.43%、CNN 8.38%/2.08%）、
**stage_clear 的锚点语义**（像素过渡起点比引擎位早约 112 帧，旧门把 40/55 次提前命中记成漏检）、
**game_over 的首可见检测**（旧静态规则延迟 89 帧）都**不能直接用于 v4.1 的像素-only 输入**。
更关键的是，`.tmp_v400_d1` 的 `frame_event_flags` 是**每 4 帧一次的单帧快照**：
469,216 行里 death 位 **0**、bomb 位 **0**，而 45 个 shard 实际 `stop_reason=death`；
它**不能当事件真值**。v4.1 必须先冻结“首次可见证据”锚点，做全分辨率 HUD +
像素定位/局部窗 + 显式状态机，再谈 99% 事件门。

## 1 审计结论

### 1.1 可以复用

| 组件 | 证据/结果 | 复用方式 | 引用 |
|---|---|---|---|
| HUD 几何与裁剪 | lives/bombs 读数 100%，LB 99.997%；bomb 436/436 同帧 | 复用 9 位 score、lives/bombs 带的行列几何；改成**全分辨率彩色 HUD 区域**后重训/模板化 | `tools/rl/v400_p2_reward_detector.py:1389-1443`、`:1620-1675` |
| HUD 读数 CNN | v1 reader：lives 7 类、bombs 9 类、score 9×10 类 | 作为**学习型 fallback**，不是主判据；主判据先用图标/数字模板 | `:1639-1678`、`:1718-1774` |
| HUD 差分事件 | lives 递减即 death：69.26%/98.94%，延迟 4/4/10；bombs 递减即 bomb：100%，延迟 0 | 保留为**确定性确认层**；不能单独做 death（lives==0 的 188/530 无递减） | `:1473-1484`、`:2493-2558`、`:2684-2750` |
| 统计/匹配/失败关闭 | Wilson、rule-of-three、一对一 greedy、零样本不冒充 100% | 复用并**参数化置信水平**；v4.1 需加聚类 bootstrap | `:106-236`、`:179-217`、`:220-236`、`:2818-3001` |
| stage 过渡的流式形态 | `eventFrame`/`alarmFrame` 双时间、jump+sustain+refractory | 复用**流式状态机形态**与 vgrad 特征；锚点改成“像素可见起点” | `tools/rl/v400_stage_clear_replay.py:82-150`、`v400_p2_reward_detector.py:3088-3158` |
| 像素时序特征 | 7 帧因果窗；current/diff1/diff3/diff6/collapse/novelty 6 通道 | 复用到**玩家局部窗**；这是现成的最强 death 视觉特征 | `tools/rl/v400_pixel_death.py:241-325` |
| 事件规则 | 上升沿 + 确认延迟（延迟只加不减）+ refractory + prominence | 复用规则骨架；v4.1 的可见锚点窗 `before=0` | `v400_pixel_death.py:764-857` |
| 玩家定位基线 | 64×64 heatmap locator；T2a median 11.59 px（4 px 门未过） | 作为基线；v4.1 需全分辨率局部窗/跟踪，目标 median ≤2 px、P95 ≤4 px | `touhou_rl/v400/student/probe/net.py:157-223`、`obs_spec_v3.py:35-40`、`:133-141` |
| 数据/切分失败关闭 | index schema、SHA/标签版本混用、空 split、短平面全部非零退出 | 复用失败关闭思想；切分单位改为**原轨迹/祖先**，不是 stage | `v400_p2_reward_detector.py:267-397` |

### 1.2 不能直接用于 v4.1 像素-only

| 组件/结论 | 为什么不能直接用 | 数字/引用 |
|---|---|---|
| v0 playfield death 头 | 学到 dive 域“顶部弹幕密集”，`still` 48 个死亡 0 命中；跨关卡阈值不迁移 | EXP-P2-003 §5.1/§7；`v400_p2_reward_detector.py:637-689` |
| v1 学习型因果事件头 | 未收敛（test 只报 1/270）；pos_weight 上限 8 + 0.2% 正例 | EXP-P2-004 §3/§8；`:1681-1702`、`:1828-1871` |
| death 只靠 HUD 递减 | 全库 342/530 可观测（64.5%）；188/530 在 lives==0 之后，HUD 无递减 | EXP-P2-004 §1；`:2684-2750` |
| death 的 `+10` 帧/`+4` 帧旧窗 | 旧 G2 以引擎边沿后 10 帧计分；v4.1 以首次可见证据计分；`+29` 帧 HUD 刷新路径会超窗 | EXP-P2-007 §5.4；G410-3 明说与旧 G2 不同 |
| NCC 玩家模板 | 依赖引擎 `PLAYER` 的 px/py；v4.1 运行时禁止真值；且阈值不可分 | `:3402-3518`；EXP-P2-008 §14 |
| 全局 play96 death CNN | test 8.38%/2.08%；online 19.5%/4.6% @10；89k 参数 | EXP-P2-011 §3.3；`v400_pixel_death.py:1-42` |
| stage_clear 旧门 | 像素过渡起点中位 **−112 帧**（范围 −172..+1），旧窗 `[edge, edge+10]` 只 15/55；切片口径 248/250 是裁掉提前量后的假象 | EXP-P2-011 §3.1；`:3088-3158` |
| game_over 静态规则 | 需要 60 帧 lag，延迟 89 帧；WP5 round1 因 episode 在 game over 后 30 帧关闭而 0/415，round2 253/260 在 +88/89 | `.tmp_v400_g2online/wp5-summary.json`、`run2/`；`wp5_online386.py:72-73`、`:215-225` |
| 56×160 绿色 HUD | 丢颜色/图标/暂停菜单/完整客户区；v4.1 要求保留原始 HUD 区域 | `v400_p2_bulk_collect.py:85-101`、`:141-180`；`obs_spec_v3.py:84-91` |
| p2-v2-1 index/label 契约 | D1 是 `v400-teacher-obs-1`，没有 li/bo/sc 列，无 index；不能喂给 `train-v1` | `:79-82`、`:267-280`；D1 labels 首行 schema |
| D1 `frame_event_flags` | 每 4 帧一次快照：469,216 行 death=0、bomb=0、hit=95、dying=24、stage_clear 0x80=270；45 个 death shard 不含 death 位 | `.tmp_v410_wp06_audit/audit_verify.out.json`；`touhou_rl/v400/teacher/decide.py:164,277,304-311` |
| `stage_clear` 事件位 | 只有 `rl_events` 构建导出；是电平位（本例连续 79 帧），不是边沿；stage index 又晚 77 帧 | `.tmp_v400_g2c/stage_clear_gate.json`；`:2482-2486` |
| score 用 globals.score 作标签 | HUD 画的是 displayScore，滚动 0–32+ 帧；settled 符号 99.56%，rolling 47.73%；负向样本除 stage 重置外为 0 | EXP-P2-004 §5.2；`v400_p2_reward_detector.py:2201-2310` |

### 1.3 关键数字（审计实测）

| 项 | 值 | 来源 |
|---|---|---|
| v4.0 detector 行数/SHA | 3,738 行 / `88cbf7da…d75787242f0` | audit_verify.py |
| v1 HUD lives/bombs | 100% / LB 99.997% | EXP-P2-004 §5 |
| v1 death（HUD 差分） | 69.26% / 98.94%，延迟中位 4、p95 4、max 10，n=270/189 | EXP-P2-004 §5 |
| bomb（HUD 差分，G2C） | 436/436，LB 99.126%，延迟 0 | `.tmp_v400_g2c/bomb-gate.json` |
| game_over（静态规则，G2C） | 406/406，LB 99.063%，延迟 89 | `.tmp_v400_g2c/gameover-gate-valid.json` |
| stage_clear（1 个真实事件） | bit 0x80 连续 79 帧；stage flip +77；无像素检测器 | `.tmp_v400_g2c/stage_clear_gate.json` |
| stage_clear（像素，250 事件） | 切片 99.2%；连续流 15/55（27.27%），像素起点中位 −112 帧 | EXP-P2-011 §3.1 |
| NCC death | 16.41% / 17.43%，LB 14.14/15.03，n=908/855，延迟中位 0、max 10 | EXP-P2-008 §14 |
| CNN death | test 8.38% / 2.08%；online 19.5% / 4.6% @10；36.3% / 8.5% @30 | EXP-P2-011 §3.3；wp5-summary |
| D1 语料 | 137 shard / 469,216 decision / 1,876,864 frame / 8 个 Normal 面 / 25 seeds | audit_verify.py |
| D1 事件位 | death 0、bomb 0、hit 95、dying 24、stage_clear 0x80 270（全 469,216 行） | audit_verify.py |
| D1 结局 | death 停止 45、stage_changed 15（全是 stage 0）、max_frames 77 | audit_verify.py |
| WP5 round1 | 453 局 / 551,250 帧 / 1,325 death / 530 bomb / 415 game_over | wp5-summary |
| WP5 round2 | 285 局 / 372,891 帧 / 832 death / 260 game_over；253 个静态段在 +88/89，7 漏 | run2 重算 |
| Stage-clear 语料 | 250 个 case；Normal 仅 27 个（s0:21/s1:3/s2:3）；Normal s3–s7 **0** | combined/report.json |

### 1.4 数据缺口（与 WP06 要求逐条对照）

| WP06 要求 | 现有 | 缺口 |
|---|---|---|
| Normal 各面 | D1 覆盖 0–7（Normal）；stage-clear 语料只 0–2 | death/game_over/连续过面在 Normal s3–s7 几乎为空 |
| 资源增减 | D1 无 li/bo/sc 逐帧标签；p2v2 只有 7 shard/3,989 帧 | 需逐帧 HUD 真值（displayScore/lives/bombs）+ 原因标注 |
| death / death-bomb / 无 bomb 死亡 | WP5 有 1,325 death、139 bomb==0；D1 45 个 death 停止但无 death 位 | 需可见锚点 + 逐帧事件真值；death-bomb 需专门策略 |
| 遮挡 / 慢速 / Boss 转场 / 暂停 / 菜单 | 无系统数据；WP5 不含暂停/菜单 | 需专门采集与 unknown 标注 |
| 长无事件段 | 无 10h 等效连续流 | 需按面分层采集 |
| 按原轨迹/祖先隔离 | v0 按 stage；stage-clear 按 replay；D1 有 seed 但无祖先图 | 需轨迹/祖先 ID 与确认集封存 |
| 事件 ≥400 / Wilson 下界 | 旧门最大 n 436（bomb）/406（game_over）；death 全库 530 但可观测仅 342 | death 可观测 342 就不够；需 ≥400/类（建议 ≥550） |
| 像素终局/重开 | 无 | 需 100 次闭环，≥99 成功 |
| 像素-only 运行时 | 现有 detector 允许 HUD 像素，但 v0 输入 56×160；无 purity 审计 | 需禁止 EVENTS/STATE/HUD 真值读取并测试 |

## 2 v4.1 事件与状态机

### 2.1 状态（public）

| 状态 | 可见语义 | 进入条件 | 退出条件 |
|---|---|---|---|
| `ALIVE` | 游戏进行中；机体可见且稳定；HUD 可读；无菜单/过场 | 初始；respawn 完成；过面完成；菜单关闭 | hit/death/过面/菜单/unknown 证据 |
| `HIT_PENDING` | 命中后的短暂无敌/闪烁窗口；可能 death-bomb | 机体闪烁/白闪/局部 novelty 尖峰，或 hit 证据 | death 证据 → `RESPAWN`/`GAME_OVER`；bomb 消耗且存活 → `ALIVE`；窗口 `W_hit` 到期无死亡 → `ALIVE` |
| `RESPAWN` | 机体消失/消解，等待回位或终局 | death 证据（`A_death`）且不是终局 | 机体在底部重新出现并稳定 `K` 帧 → `ALIVE`；lives==0 且终局面板 → `GAME_OVER` |
| `STAGE_TRANSITION` | 过面过渡：Boss 死亡/白闪/面板/横幅 | 像素可见过渡起点 `A_stage_visible` | 新面 `ALIVE` 稳定；或菜单/unknown |
| `GAME_OVER` | 终局面板/Continue/Retry；lives==0 | 末命 death + 终局面板首可见 | 重开 → `ALIVE`；回标题 → `MENU` |
| `MENU` | 暂停/标题/设置等非终局菜单 | 菜单首可见（暂停、标题） | 关闭 → 回到上一状态；新游戏 → `ALIVE` |
| `UNKNOWN` | 状态不可判定（黑屏/加载/过场/定位与 HUD 同时失效） | 观测无效且无法维持上一状态 | 观测恢复；不自动补零，不补事件 |

说明：
- `RESPAWN` 内部有一个 `DYING` 子相（机体消失/消解、等待回位），它是 private 子相，
  不作为 public 状态，也不单独设门；public 状态仍是上面 7 个。
- `HIT_PENDING` 是**状态**不是 death；`death` 是事件，只在 `A_death` 发一次。
- `MENU` 与 `GAME_OVER` 分开：**终局 Retry/Continue 面板算 `GAME_OVER`**，不再重复计
  `menu` 事件；`menu` 事件只计暂停/标题/设置等非终局菜单。重开闭环记录 `menu_kind=retry`
  作为 `GAME_OVER` 的子事件，不计入 `menu` 分母，避免双重计数。
- 局部 HUD 不可读但机体可见时，**不是** `UNKNOWN`：维持上一状态，字段 `hud_unknown=true`；
  只有状态本身不可判定才进 `UNKNOWN`。
- `UNKNOWN` 期间发生的事件仍进**总分母**（`unobservable` 子集），不能删除；不能匹配就计漏检。

### 2.2 事件、可见锚点与计分

| 事件 | 可见锚点 `A`（首次可见证据，独立标注） | 运行时检测 | 引擎真值（只进离线标签） | 主延迟 | 次延迟 |
|---|---|---|---|---|---|
| `death` | 机体开始消失/消解，或机体位置出现死亡爆炸的第一帧；`A=min(可见类别)`，由标注员盲标 | 局部 patch 的 sprite-present/NCC/novelty 崩塌 + 状态机确认 | `EVENTS & 0x4` 上升沿 | `alarm − A` P95 ≤10，`before=0` | `alarm − engine` P95 ≤30，max ≤60 |
| `bomb` | bombs 图标计数下降的第一帧（同帧） | HUD 图标计数；HUD 不可读时 bomb 激活特效 | `EVENTS & 0x1` 上升沿 | `alarm − A` ≤10（目标 0） | `alarm − engine`（应 0） |
| `death_bomb` | 命中后 bomb 激活/消耗、且 lives 未下降 | `HIT_PENDING` + bombs 递减 + `W_confirm` 内无 lives 递减 | 无专用引擎位；用 `0x2` + bomb 位/输入日志 | 作为 `bomb` 子类报告 | 不计 death |
| `respawn` | 机体在底部重新出现的第一帧 | 局部 patch 重新出现 + 稳定 K 帧 | 无引擎位；可见标注 | 状态事件，诊断 | — |
| `stage_transition` | 过面序列首帧：Boss 死亡/白闪/面板/横幅，由标注员盲标；**不是** stage index 翻转 | vgrad jump + sustain；面板/横幅模板 | `EVENTS & 0x80` 上升沿（电平位）；`STATE.stage` 变化 | `alarm − A` P95 ≤10 | 报告 `alarm − engine_bit` 与 `alarm − stage_index` 的带符号提前量 |
| `game_over` | 终局面板/letterbox 首帧；**不是**面板静止 60 帧后 | 终局面板模板/边沿 + lives==0 历史 | `HUD gameOverFlag` 上升沿（标签） | `alarm − A` P95 ≤10 | 报告 `alarm − engine_flag` |
| `menu` | 暂停/标题菜单首帧 | 菜单模板/边沿 | 输入日志 + 可见标注（无引擎位） | `alarm − A` P95 ≤10 | — |
| `hit_pending` | 机体闪烁/白闪首帧 | 局部时序特征 | `EVENTS & 0x2`（标签） | 诊断，不作硬门 | — |

`death` 的可见证据类别（标注时必填一项，用于分层）：
1. `sprite_vanish`：机体 sprite 消失/消解（最强，末命也可见；+0..+4 帧）；
2. `death_effect`：机体位置出现死亡爆炸/特效，但 sprite 消失被遮挡；
3. `hud_lives_decrement`：仅当 1/2 都不可见时使用（+4 帧或 +29 帧，取决于 bomb 存量）；
4. `respawn_only`：只能从回位推断（最弱，单独报告）。

`W_hit`（hit→bomb/death 的 pending 窗口）与 `W_confirm`（bomb 后等 lives 的确认窗口）
必须在 WP06.1 用 dev 数据实测并冻结；当前 `W_confirm=35` 帧只是占位，不是门常数。

**锚点冻结规则：** 两名标注员独立盲标（不看 detector 输出、不看引擎帧），
不一致处第三人裁决；冻结 `anchor_version`、标注文件 sha256、裁决记录。
detector 上线后**不得**用其输出去改锚点；改了锚点就是新版本 + 新确认集。

### 2.3 death 延迟锚点与计分（G410-3 的关键改动）

- 旧 G2：`latency = detection − engine death bit`，窗 `[engine, engine+10]`。
- **G410-3 主判据：`visible_latency = alarm − A_death`，匹配窗 `[A_death, A_death+10]`；
  `alarm < A_death` 不算命中，计假正例。** 这禁止“提前猜中”被算成检测。
- 次判据：`engine_latency = alarm − engine_death_bit`，P95 ≤30、max ≤60；
  允许负值（视觉早于引擎位 1–2 帧），报告带符号分布。
- **全部真实死亡进总分母**，包括末命死亡、death-bomb 失败、过面/菜单期间的死亡；
  `unobservable` 样本单列（遮挡/黑屏/unknown），**不从分母删除**；其漏检如实计入。
- 延迟用 nearest-rank P95（`ceil(0.95*n)-1`），同时给 median/max/n；
  n=0 时不给通过，也不给 0% 或 100%。
- 旧 `+4/+29` HUD 刷新路径不再定义主延迟；`hud_lives_decrement` 只是可见证据类别 3。

### 2.4 过面双时间（WP06 要求 5）

| 时间 | 定义 | 用途 |
|---|---|---|
| `A_stage_visible` | 人类首次能看出“这一面要结束”的帧（Boss 死亡/白闪/面板/横幅首帧），独立标注 | **主判据锚点**；`alarm − A_stage_visible` P95 ≤10 |
| `E_stage_bit` | `EVENTS & 0x80` 上升沿（rl_events 构建）；电平位，连续约 79 帧 | 引擎正式位；只进离线标签 |
| `E_stage_index` | `STATE.stage` 翻转帧 | 引擎正式过面；只进离线标签 |
| `lead_bit = alarm − E_stage_bit` | 带符号；现有 vgrad 起点中位约 −112 帧 | **报告**，不用于主判据，不通过改引擎语义来“修正” |
| `lead_index = alarm − E_stage_index` | 带符号；现有 stage index 又晚 77 帧 | 报告 |

旧 G2 的 `[edge, edge+10]` 窗**不用于 G410-3**；旧 G2 仍按原协议单独结算。
不得把 `A_stage_visible` 事后定义为 detector 首次触发的帧；标注必须先于 detector 冻结。

### 2.5 资源/奖励语义

- HUD 读数目标：`lives ∈ 0..6`、`bombs ∈ 0..8`、score 9 位数字；全部**像素读数**，
  运行时禁止 `STATE/HUD/EVENTS` 真值。
- score 用 **displayScore**（像素上真实可见的值）作为奖励资源，不用 `globals.score`；
  同时离线记录 `globalsScore` 以量化滚动滞后。滚动未 settled 的帧标 `score_rolling=true`；
  奖励使用 settled 后的值或差分，不把滚动当真实变化。
- 资源变化必须**归因**，不能只看符号：
  - lives 下降 = death（必须伴随 death 事件/状态机确认）；
  - lives 上升 = extend（1UP）或 stage reset/continue；stage reset 单独标 `resource_reset`；
  - bombs 下降 = bomb 使用，或死亡处理移动 bomb 存量（P2-004 已指出死亡帧 bomb 下降歧义）；
    用 bomb 激活视觉 + death 状态消歧；
  - bombs 上升 = bomb pickup 或 stage reset；
  - score 下降 = stage reset/score 清零（引擎无扣分路径）；**不得**当受伤/惩罚；
  - score 上升 = 正常得分；stage reset 后的上升要扣掉 reset 基线。
- stage reset 窗口（过面/重开）内的资源跳变一律标 `cause=reset`，不产生 bomb/death 事件。
- `unknown` 读数不补零、不补上一值；按字段报告 `unknown_ratio` 与覆盖。

## 3 数据计划

### 3.1 Scope 与流

- 主 scope：**Normal（difficulty 1）全部 8 个 stage 索引 0–7**（Stage 1..6B；
  `ScoreDat.hpp:73-87`，见 WP02 DEATH.md §0）。team 固定一个（team 0），
  另一 team 只作外部 OOD，不进 G410-3 分母。
- 连续流：一局从 RESET 到 game over/重开，不中途 RESET；60 Hz 逐帧记录。
  失败/超时/异常按 fail-closed 记入清单，不按结果重跑。
- 引擎 SHA 冻结；一次采集批次一个 SHA；混 SHA 直接失败。
- 运行时 detector 只读像素；引擎 `EVENTS/STATE/HUD` 真值写**独立标签文件**，
  只用于离线评分，不能进 detector 进程。需要有 purity 审计（见 §4.5）。

### 3.2 覆盖矩阵（最低目标）

| 维度 | 分层 | 最低目标 |
|---|---|---|
| 面 | Normal s0–s7 | 每面 ≥30 个 stage_transition；每面 ≥1h 背景/长流 |
| death | 普通（bomb>0）、无 bomb（+29）、末命、death-bomb、过面/菜单期间死亡、遮挡死亡 | death 去重总分母 ≥400；每个分层 ≥100（分层可重叠；为覆盖 6 层按 ≥600 个 death 规划；末命/无 bomb 必采） |
| bomb | 普通 bomb、death-bomb、bomb pickup、死亡帧 bomb 移动 | bomb 事件 ≥400；death-bomb ≥100 |
| 过面 | 道中/Boss 后、白闪/面板/横幅、慢速渐变 | stage_transition ≥400；每面 ≥30 |
| game_over | 3 命耗尽、Continue/Retry、回标题 | ≥400 |
| menu | 暂停/恢复、标题/设置 | ≥400 |
| respawn | 有残机死亡后的回位、无敌闪烁 | ≥300（death 子集） |
| 资源 | lives/bombs/score 增减、extend、stage reset、bomb pickup | 每类 ≥100 个变化事件；逐帧标签 |
| 长无事件 | 8 面各 ≥1h；总计 ≥10h 等效 | 0 事件段，FP ≤1/游戏小时 |
| 遮挡/慢速 | Boss 特效、对话/黑屏、慢速聚焦 | 各 ≥100 帧段/≥30 事件 |

数量说明：`stage_transition` 400 次按每次 2–4 分钟约 13–27h 引擎时间；16 实例并行约 1–2h 墙钟。
`death`/`bomb` 用 still/dive 约 500/h / 200/h；`game_over` 约 160/h（WP5 round1：
1,325 death / 530 bomb / 415 game_over in 2.55h 等效），400 个约需 2–4h；多样性靠策略/面分层。
所有数字是**最低**目标，不是“重复同一事件 10 次凑 400”；重复回放只算复现检查。

### 3.3 采集分层与存储

| 层 | 内容 | 用途 |
|---|---|---|
| T0 每帧 | play96/play64、**全分辨率彩色 HUD 区域 160×64×3**、玩家局部 96×96×3（定位器中心）、输入 mask、逐帧标签 | 训练/评测主数据；raw 约 73 KB/帧，压缩后按实测 |
| T1 事件窗 | 每个事件 ±120 帧的**完整 640×480 RGBA** 无损 PNG | 标注可见锚点、复核、失败录屏 |
| T2 长流 | T0 的 compact 版；每 600 帧一张完整帧 | 10h 无事件/误报 |

- 预算上限 150 GB。WP06.1 先做**1 小时 pilot 的字节实测**（不训练、不采正式集），
  用实测压缩率算全量；超预算就缩小 T1 窗口/采样，不缩 T0 事件帧。
- 所有采集产物写 manifest（SHA、shard、seed、ancestor、stop_reason、帧数、事件计数），
  不覆盖旧 attempt；断点续跑用新 attempt 目录。
- 标签必须**逐帧**：`frame_event_flags`（每帧 EVENTS 字）、`STATE`（lives/bombs/score/stage）、
  `HUD`（displayScore/lives/bombs/gameOverFlag）、`PLAYER`（仅离线标签，用于定位误差评估）、
  输入日志、RESET/重开记录。**D1 的每 decision 快照不能用于事件真值。**
- 可见锚点标注：T1 全帧 + 独立标注工具；death/stage/game_over/menu 各两名标注员；
  冻结 anchor_version + sha；标注不一致率报告。

### 3.4 切分/祖先隔离

- 切分单位：**原轨迹及其祖先**（同一 seed、同一 RESET 链、同一 restart 祖先、
  同一 replay/重放派生）。同一祖先的所有派生流只能进一个 split；不允许“同一局换 10 个窗口”跨集。
- dev/confirmation 用**预注册 seed 段**；confirmation seed 段在冻结前不采、不看。
- 目标：train ≥60%、dev ≥20%、confirmation ≥20% 的**独立轨迹**；每类事件在 confirmation
  ≥400（建议 ≥550，见 §4.1）。
- 面/策略/难度分层在三个 split 内尽量平衡；任何 split 缺层直接失败关闭，不用重采样补。
- 祖先图与 split 清单写入 manifest；`build_split` 类检查：同一 ancestor 跨 split → 非零退出。

### 3.5 确认集封存

1. 冻结：detector/state machine/threshold/anchor 定义/评分脚本版本 + sha256；
   写 `confirmation-manifest.json`（seed 段、轨迹数、每类事件数、引擎 SHA）。
2. 封存：confirmation 流在冻结后采集或从已采但**未看**的预注册 seed 段打开；
   打开前不跑 detector、不调阈值、不看事件数。
3. 一次性：只开一次；失败就记失败，改任何组件都要新版本 + 新确认集，不能复用同一集。
4. 证据：per-stream/per-event JSONL、manifest、命令、退出码、失败录屏/超时；
   不是一张汇总表。
5. 旧 G2 数据/阈值/窗口不能改名成 G410-3；G410-3 的确认集与旧 G2 确认集分开登记。

## 4 评测计划

### 4.1 事件级 P/R 与下界

- 事件级一对一 greedy 匹配；预测 `p` 命中真值 `d` 当 `A_d ≤ p ≤ A_d+10`；
  每个预测最多命中一个事件；未命中预测计 FP。
- **正事件与预测事件各 ≥400**（README 最低）；建议 **≥550/类**：
  - README 口径：Wilson **95% 单侧**下界 ≥99%（z=1.96）；n=400 零失败时 LB=99.049%，刚好过；
  - 若“Wilson 99% 下界”被理解为 **99% 置信度**（z=2.326），n=400 零失败只有 **98.665%**，
    要 **n≥536** 才 ≥99%；因此建议 ≥550，报告同时给 95% 与 99% 置信下界。
  - 两种口径必须**事前选定并写进 manifest**；不能看结果再选。
- 聚类：cluster = 原轨迹祖先；先算 per-cluster recall/precision，再做 10,000 次 cluster
  bootstrap，给 99% 区间；报告 design effect。若 design effect >1.5，**cluster 区间为主判据**，
  Wilson 只作参考；否则 Wilson 为主、cluster 并列。
- 每类事件分别报：death、bomb、stage_transition、game_over、menu；
  `death_bomb`、`respawn`、`hit_pending` 作为子类/诊断。
- 零样本 = 不可测，不给 0%/100%，不给 pass；预测数 <400 也 fail-closed。

### 4.2 延迟门

| 事件 | 主延迟（可见锚点） | 次延迟（引擎） |
|---|---|---|
| death | P95 ≤10 帧，`before=0`；median/max/n 并列 | P95 ≤30，max ≤60；带符号 |
| bomb | P95 ≤10（目标 0） | 应 0 |
| stage_transition | P95 ≤10；`lead_bit`/`lead_index` 只报告 | 不设“必须晚于引擎” |
| game_over | P95 ≤10（首可见，不是静止后） | 报告 `alarm − gameOverFlag` |
| menu | P95 ≤10 | 输入日志 |
| respawn | 诊断 | — |

- P95 用 nearest-rank；n<20 时同时给逐事件延迟表，不能只给一个 P95。
- 若 detector 需要 `sustain` 确认，确认帧数**计入**延迟；不得用“离线可回看”当在线延迟。

### 4.3 资源读数与奖励

- lives/bombs：分段（600 帧窗口）算准确率；Wilson 95% LB ≥99.5%；同时报最差轨迹/窗口；
  unknown 比例 ≤1%/字段；低于门直接失败，不用长静止帧抬高置信度。
- score：逐位准确率、整数值、settled 符号；settled 符号 LB ≥99%；rolling 单列诊断；
  负向类（stage reset）单独报告；没有负向样本就写不可测。
- 奖励：用可见 displayScore + 资源/事件奖励；报每局累计奖励误差（与可见标签）、
  资源不一致次数、unknown 帧占比；不把 reset 当惩罚/奖励。

### 4.4 长流与重开

- 长无事件：10h 等效 Normal 连续流，8 面各 ≥1h；事件误报 ≤1 次/游戏小时；
  报背景段 FP、unknown 比例、资源不一致、累计奖励误差；0 FP 时给 Poisson/rule-of-three 上界。
- 重开闭环：≥100 次由**像素终局**（game_over/menu）驱动的菜单/重开；≥99 次自动成功；
  失败录屏 + 超时 + 失败原因；纯像素训练只能用此路径或事前固定截断。
  99/100 是点估计，同时报 Wilson 区间；不声称 99% 下界。

### 4.5 运行时纯度与失败关闭

- 运行时禁止读取 `EVENTS`/`STATE`/`HUD` 真值；detector 进程只收像素帧和自己的状态。
- 纯度测试：把引擎真值文件打乱/删除，detector 输出必须逐字节不变；导入审计禁止
  `touhou_rl.v400.gym.protocol` 等真值模块进入 runtime 路径。
- 失败关闭：缺 HUD/缺帧/帧序不连续/时间戳回退/混引擎 SHA/混 anchor_version/
  确认集被打开过/样本不足 → 非零退出，不产出一个“通过”。
- 任何 `unknown` 不能由引擎纠正；只能在离线分析里解释，不能回填到 runtime 输出。

## 5 分阶段实施（草案）

> 本节是**下一任务的设计**，本卡未执行；命令是草案，不代表已运行。

### WP06.1 离线标签/事件定义/状态机（无游戏）

- 新增（建议路径，下一任务白名单另行确认）：
  `touhou_rl/v410/reward/events.py`（事件定义、锚点、匹配、Wilson 95/99、cluster bootstrap）、
  `state_machine.py`（7 状态 + 内部 DYING 子相）、`labels.py`（逐帧标签契约/转换）、
  `audit.py`（对 D1/WP5/G2C/stage-clear 语料做只读覆盖审计）。
- 测试：`tests/test_v410_reward_events.py`、`tests/test_v410_reward_state_machine.py`。
- 验收草案：
  ```powershell
  $env:PYTHONPATH='D:\WorkShop\Touhou_RL'
  python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp06_pytest -q `
    tests/test_v410_reward_events.py tests/test_v410_reward_state_machine.py
  python -m compileall -q touhou_rl/v410/reward
  python -m touhou_rl.v410.reward.audit --corpus .tmp_v400_d1/d1 --out `
    .tmp_v410_wp06_audit/wp061_coverage.json
  ```
- 退出码：pytest 0；compileall 0；audit 0（只表示审计生成成功，不表示覆盖通过）。
- 必须产出的审计：D1 事件位 0/death、stage_clear 270、WP5 数字、stage-clear Normal 缺口；
  与 `audit.json` 的 `data_inventory` 对齐。

### WP06.2 全分辨率 HUD 读数（模板优先，学习 fallback）

- 新增 `touhou_rl/v410/reward/hud.py`：彩色 HUD 区域、9 位数字格、lives/bombs 图标模板、
  displayScore 读取、settled/rolling、unknown。
- 复用 v4.0 reader 架构/几何做 fallback；主判据必须可解释（模板/图标计数）。
- 验收草案（dev 集，非确认集）：
  ```powershell
  python -m touhou_rl.v410.reward.hud --corpus <dev-hud> --out <dev-hud-report.json>
  ```
  lives/bombs 分段 LB ≥99.5%；score settled 符号 LB ≥99%；unknown ≤1%/字段；
  失败样本逐帧列，不挑帧。

### WP06.3 像素事件 detector（定位/局部窗/状态机）

- 新增 `touhou_rl/v410/reward/player.py`：全分辨率玩家定位（模板 + 跟踪 + 置信度），
  目标 median ≤2 px、P95 ≤4 px（对比 T2a 11.59 px）。
- 新增 `detector.py`：局部 sprite-present/NCC/novelty、death/hit/respawn；
  vgrad+面板/横幅模板做 stage_transition；终局面板模板做 game_over；暂停/标题模板做 menu；
  融合 HUD 确认层 + 状态机；所有阈值在 dev 冻结。
- 验收草案：dev 上每类 ≥400（建议 550）事件、P/R 95% LB ≥99%、death 可见延迟 P95 ≤10、
  engine 延迟 P95 ≤30/max ≤60、FP ≤1/h；局部定位误差；unknown；纯度测试。

### WP06.4 开发集采集与冻结

- 按 §3 采集 Normal s0–s7 连续流、death/death-bomb/无 bomb/末命/遮挡/慢速/Boss/暂停/菜单/
  长无事件；按祖先隔离；dev 达标后冻结 detector/state machine/threshold/anchor。
- 验收草案：覆盖矩阵每格 ≥ 最低目标；manifest/SHA；字节预算；dev 门全过；
  冻结 `detector_version` 与 `anchor_version`；未打开 confirmation。

### WP06.5 确认集（一次性）

- 打开预注册 confirmation seed 段；只跑一次；不调阈值/不改锚点/不补样本。
- 验收草案：
  ```powershell
  python -m touhou_rl.v410.reward.eval --split <confirmation-split.json> `
    --detector <frozen> --out Document/Update/Alpha/v4.1.0/evidence/WP06/confirmation.json
  python tools/rl/v410.py check --wp WP06 --evidence Document/Update/Alpha/v4.1.0/evidence/WP06
  ```
  注意：`tools/rl/v410.py check --wp WP06` **当前未实现**（只实现 WP00/WP01），
  会按设计返回 exit 3；WP06.5 之前必须先实现 WP06 check（下一任务，不在本卡白名单）。
- 通过才写 G410-3；失败写有效负结果 + `not_claimed`，不改旧 G2。

## 6 G410-3 验收草案（全部满足才算通过）

| # | 门 | 通过条件 | 失败关闭 |
|---|---|---|---|
| 1 | 事件/锚点冻结 | 7 状态机 + 事件定义 + 可见锚点 + 评分脚本版本化；双标注/裁决记录 | 缺 anchor_version → 失败 |
| 2 | death | ≥400（建议 550）；P/R 95% LB ≥99%（99% 置信度口径需 n≥536）；可见延迟 P95 ≤10；engine P95 ≤30/max ≤60 | 末命/不可观测被删 → 失败 |
| 3 | bomb/death-bomb | ≥400；P/R LB ≥99%；延迟 ≤10；death-bomb 不冒充实 death | 死亡帧 bomb 下降误计 → 失败 |
| 4 | stage_transition | ≥400；P/R LB ≥99%；可见延迟 P95 ≤10；双时间（可见/引擎）都报告 | 用旧 `[edge,edge+10]` 冒充 → 失败 |
| 5 | game_over/menu | 各 ≥400；P/R LB ≥99%；首可见延迟 P95 ≤10 | 用“静止 60 帧”当首可见 → 失败 |
| 6 | 资源 | lives/bombs 分段 LB ≥99.5%；score settled 符号 LB ≥99%；unknown ≤1%；负向不可测如实写 | 长静止帧抬置信度 → 失败 |
| 7 | 长流 | 10h 等效；FP ≤1/游戏小时；unknown/资源不一致/累计奖励误差报告 | 无 0 事件段 → 失败 |
| 8 | 重开 | 100 次像素终局驱动；≥99 自动成功；失败录屏/超时齐全 | 用引擎终局 → 失败 |
| 9 | 确认集 | 封存、一次性、独立祖先/seed；冻结后不调阈值 | 打开过/复用 → 失败 |
| 10 | 纯度 | runtime 不读 EVENTS/STATE/HUD 真值；打乱真值输出不变 | 任一真值 import → 失败 |
| 11 | 证据 | per-stream/per-event JSONL + manifest + 命令 + 退出码 | 只有汇总表 → 失败 |

## 7 已执行（本卡）与未执行

已执行（只读，退出码 0）：
```powershell
python .tmp_v410_wp06_audit/audit_verify.py
# → .tmp_v410_wp06_audit/audit_verify.out.json；EXIT=0
# 断言：detector 3,738 行/SHA；D1 137/469,216/1,876,864；death 位 0；
# WP5 round1 1,325/530/415；round2 260/253；G2C bomb 436、game_over 406；
# stage_clear bit 128、79 帧、flip +77；Normal stage-clear 27。
```
辅助探针（同一运行目录，均无引擎/无训练）：
`probe02_events.py`、`probe03_inventory.py`、`probe04_flag_census.py`、
`probe05_hashes.py`、`probe06_stage_outcomes.py`、`probe08_g2c2.py`、
`probe12_wp5_round2.py`、`probe13_wp5_go_lag.py`、`probe15_wilson.py`。

未执行：pytest（避免写 `__pycache__`/缓存）、compileall、任何引擎/采集/训练/commit/push。
因此本卡**没有**“测试通过”或“G410-3 通过”的声明；`v400_p2_reward_detector.py` 的
66 个测试只做了静态清点，未运行。

## 8 不宣称（not claimed）

- 不是 G410-3 通过；没有新的 detector、没有确认集、没有运行时证据。
- 没有采数据、没有训练、没有启动游戏、没有改 v400/v410 代码或 state/journal/budget/assets。
- 没有证明像素-only death 能到 99%；现有证据反而显示 NCC/CNN/HUD 差分都不到。
- 没有证明 400 个事件在 99% 置信度下够；按 99% 置信度需 ≥536，建议 ≥550。
- 没有证明 stage_clear 的 `A_stage_visible` 就是旧 vgrad 起点；必须先独立标注确认。
- 没有证明 game_over/menu 的 10 帧首可见可达；旧规则是 89 帧。
- 没有证明 player locator 能达到 ≤2 px；T2a 基线是 11.59 px。
- 没有证明 D1 `frame_event_flags` 可作事件真值；实测 death/bomb 位为 0。
- 没有把旧 G2 的 95% 下界/旧窗/旧阈值改名为 G410-3。

## 9 下一任务（建议）

**WP06.1：离线事件/状态机与标签契约（无游戏）**——先实现 `events.py`/`state_machine.py`/
`labels.py`/`audit.py` 与测试，冻结“首次可见证据”定义，重跑本卡 `data_inventory`，
产出一份“现有语料能支持到什么、缺什么”的机器可读覆盖报告；然后才进入 WP06.2 HUD 与
WP06.3 像素 detector。不得先训练/先采数据/先调阈值。
