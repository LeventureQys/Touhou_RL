# WP06.1 — v4.1 连续像素回报：离线事件/状态机/标签契约

- 工作包：v4.1.0 WP06.1（按 WP06.0 审计实现离线契约 + 合成测试）。
- 状态：**契约冻结、合成测试通过；G410-3 未开始验收、detector 未实现。**
- 机器可读契约：[`contract.json`](contract.json)（schema `v410-wp06-contract/1`，
  contract_version `wp06.1-contract/1`，25 条转移 / 17 个事件定义 / 6 个代码文件 sha256）。
  该 JSON 由 `touhou_rl/v410/reward/__init__.py::build_contract()` 从代码中的冻结表生成
  （运行目录 `.tmp_v410_wp06_contract/build_contract.py`），不是手写副本。
- 只读输入：WP06.0 [`PLAN.md`](PLAN.md)、[`audit.json`](audit.json)、[`STATUS.md`](STATUS.md)，
  `README.md` WP06/G410-3 相关小节，v400 reward detector/HUD 代码（只读参考）。
- 代码：`touhou_rl/v410/reward/__init__.py`、`events.py`、`state_machine.py`、`labels.py`。
- 测试：`tests/test_v410_reward_events.py`（43）、`tests/test_v410_reward_state_machine.py`（28）。
- 范围声明：**未启动游戏/引擎、未采数据、未训练、未做 G410-3、未改白名单外文件、
  未 commit/push、未删除/覆盖旧文件。** 运行目录 `.tmp_v410_wp06_contract/` 与
  pytest basetemp `.tmp_v410_wp061_pytest/` 不入库。

## 0 一句话结论

WP06.1 只做**契约与合成验证**：7 个公开状态 + 25 条显式转移、可见/引擎两套锚点、
`[A_death, A_death+10]`（before=0）的可见延迟门、引擎延迟 P95≤30/max≤60、
资源归因（reset/extend/bomb/death-frame stock move）、逐帧标签 schema、
按原轨迹/祖先隔离的切分校验、一次性确认集封存、以及 400/536/550 两种置信口径的字段化。
71 个合成测试通过（exit 0），compileall 通过（exit 0）。
**这些数字只证明契约自洽，不证明任何像素 detector 能达到 99%。**

## 1 状态机（7 状态，流式逐帧）

`state_machine.RewardStateMachine` 只接收 `Observation`（frame + 像素可见标志 +
HUD 读数 + 可选离线 `EngineObservation`），逐帧返回 `StepResult`（状态、事件、
资源归因、`hud_unknown`、`unknown_reason`、私有子相）。**没有批处理回看，
不使用引擎真值做判定。**

| 状态 | 可见语义 | 进入条件 | 退出条件 |
|---|---|---|---|
| `ALIVE` | 正常游戏；机体可见、HUD 可读、无菜单/过场 | 初始；respawn 完成；过面完成；菜单关闭 | hit → `HIT_PENDING`；death → `RESPAWN`；终局面板 → `GAME_OVER`；过面 → `STAGE_TRANSITION`；菜单 → `MENU` |
| `HIT_PENDING` | 命中后的无敌/闪烁窗口；**不是 death** | `hit_visible` 上升 | `W_hit` 到期/DEATH_BOMB 确认 → `ALIVE`；death → `RESPAWN`；终局面板 → `GAME_OVER` |
| `RESPAWN` | 机体消失/消解，等待回位或终局；内部私有子相 `dying` | death 证据（可见或 HUD lives 下降） | 机体在底部稳定 K 帧 → `ALIVE`(+respawn)；终局面板或末命 `T_terminal` 超时 → `GAME_OVER`；菜单 → `MENU` |
| `STAGE_TRANSITION` | 过面序列：Boss 死亡/白闪/面板/横幅 | `A_stage_visible`（像素可见起点） | 新面稳定 → `ALIVE`；菜单 → `MENU`；death → `RESPAWN`；终局面板 → `GAME_OVER` |
| `GAME_OVER` | 终局面板/Continue/Retry（含 `menu_kind=retry` 子类型） | 终局面板首可见，或末命超时 | 重开 → `ALIVE`(+restart)；回标题菜单 → `MENU` |
| `MENU` | 非终局菜单：暂停/标题/设置 | 菜单首可见 | 关闭 → 回到进入前状态（`ALIVE`/`STAGE_TRANSITION`/`GAME_OVER`/`RESPAWN`）；新局 → `ALIVE` |
| `UNKNOWN` | 状态本身不可判定（黑屏/加载/过场，HUD 与机体线索同时失效） | 观测无效或无任何信号 | 观测恢复 → 恢复**上一状态**；不补零、不由引擎纠正 |

优先级（`STATE_PRIORITY`）：`UNKNOWN > GAME_OVER > MENU > STAGE_TRANSITION > RESPAWN > HIT_PENDING > ALIVE`。

关键语义：

- `RESPAWN` 的 `dying` 是私有子相，不是第 8 个公开状态。
- 局部 HUD 不可读但机体线索有效时**不进** `UNKNOWN`：保持状态并置 `hud_unknown=true`。
- `UNKNOWN` 期间不产生事件、不更新资源基线、不补零；恢复时重新取基线（不把跨 unknown
  的读数差当成 death/extend）。
- `UNKNOWN` 期间的真实事件仍留在分母里（离线 `unobservable` 子集），不能删除。

### 1.1 显式转移表（25 条，代码内 `TRANSITIONS`，非注释）

| # | from → to | trigger | emits |
|---|---|---|---|
| 1 | ALIVE → HIT_PENDING | `hit_visible` | hit_pending |
| 2 | HIT_PENDING → ALIVE | `w_hit_expired` | — |
| 3 | HIT_PENDING → ALIVE | `bomb_confirmed_no_life_loss` | bomb, death_bomb |
| 4 | HIT_PENDING → RESPAWN | `death_visible` | death |
| 5 | HIT_PENDING → GAME_OVER | `terminal_panel_visible` | death, game_over |
| 6 | RESPAWN → ALIVE | `respawn_stable` | respawn |
| 7 | RESPAWN → GAME_OVER | `terminal_panel_or_last_life_timeout` | game_over |
| 8 | RESPAWN → MENU | `menu_visible` | menu_enter |
| 9 | ALIVE → RESPAWN | `death_visible_or_life_loss` | death, life_loss |
| 10 | ALIVE → GAME_OVER | `terminal_panel_visible` | death, game_over |
| 11 | ALIVE → STAGE_TRANSITION | `stage_visible` | stage_clear_visible |
| 12 | STAGE_TRANSITION → ALIVE | `new_stage_stable` | — |
| 13 | STAGE_TRANSITION → MENU | `menu_visible` | menu_enter |
| 14 | STAGE_TRANSITION → RESPAWN | `death_visible` | death |
| 15 | STAGE_TRANSITION → GAME_OVER | `terminal_panel_visible` | death, game_over |
| 16 | ALIVE → MENU | `menu_visible` | menu_enter |
| 17 | MENU → ALIVE | `menu_closed` | menu_exit |
| 18 | MENU → STAGE_TRANSITION | `menu_closed_into_stage_transition` | menu_exit |
| 19 | MENU → GAME_OVER | `menu_closed_into_terminal_panel` | menu_exit |
| 20 | MENU → RESPAWN | `death_visible` | death |
| 21 | MENU → GAME_OVER | `terminal_panel_visible` | death, game_over |
| 22 | GAME_OVER → ALIVE | `restart_detected` | restart |
| 23 | GAME_OVER → MENU | `title_menu_visible` | menu_enter |
| 24 | ANY → UNKNOWN | `state_undecidable` | unknown_start |
| 25 | UNKNOWN → held state | `observations_recovered` | unknown_end |

表外的转移抛 `IllegalTransition`（不 drift）；`emits` 是该迁移的典型事件，具体事件
仍以实际证据为准（例如 ALIVE→RESPAWN 只有可见证据时才发 death，纯 HUD 下降时发
death+life_loss）。

### 1.2 窗口常数（`StateMachineConfig`）

`w_hit=12`、`w_confirm=35`、`respawn_stable_frames=6`、`respawn_timeout_frames=240`、
`stage_end_confirm_frames=5`、`menu_clear_frames=2`、`alive_stable_frames=3`、
`reset_window_frames=30`、`death_confirm_frames=0`、`strict_frame_continuity=true`。

**`w_hit`/`w_confirm` 被显式标注为 placeholder**（`placeholder_fields`、`frozen=false`），
必须用 dev 数据实测后才能冻结；本卡不声称它们是门常数。

### 1.3 引擎 oracle 的边界

- 默认 `allow_engine_oracle=False`：任何携带 `EngineObservation` 的观测直接抛
  `EngineOracleForbidden`（fail-closed）。
- 仅离线审计时允许开启；即使开启，引擎数据**只**做两件事：
  1. 给可见事件附 `offline_engine_frame`（算 `engine_latency`）；
  2. 在 `0x0080` 上升沿发 label-only 的 `stage_clear_engine` 事件。
- 测试 `test_engine_oracle_does_not_change_a_single_prediction` 证明：同一流在有/无
  oracle 下状态路径、事件类型、predicted/evidence 帧、转移日志完全一致，只有
  `offline_engine_frame` 不同。

## 2 事件、锚点与延迟

`EVENT_SPECS` 共 17 个事件类型；计分口径（G410-3 分母）只有 5 类：
`death`、`bomb`、`stage_transition`、`game_over`、`menu`。

| 事件 | role | 计入 | 锚点 | 引擎位（仅标签） | 主/次延迟 |
|---|---|---|---|---|---|
| `death` | scored | death | 可见：机体消失/消解或死亡特效首帧 | `0x0004` | 可见 P95≤10；引擎 P95≤30/max≤60 |
| `bomb` | scored | bomb | 可见：bombs 图标数下降首帧（应与引擎同帧） | `0x0001` | 可见 ≤10（目标 0）；引擎应 0 |
| `death_bomb` | diagnostic | bomb（子类） | 可见：`HIT_PENDING` 内 bomb 消耗且 `W_confirm` 内无 lives 下降 | `0x0002`+`0x0001` | 作为 bomb 子类报告 |
| `stage_clear_visible` | scored | stage_transition | 可见：过面序列首帧（Boss 死亡/白闪/面板/横幅） | `0x0080`（电平位） | 可见 P95≤10 |
| `stage_clear_engine` | label_only | — | 引擎：`0x0080` 上升沿 | `0x0080` | 只报告 `lead_bit` |
| `game_over` | scored | game_over | 可见：终局面板/letterbox 首帧（**不是**静止 60 帧） | 无（HUD gameOverFlag 仅标签） | 可见 P95≤10 |
| `menu_enter` | scored | menu | 可见：非终局菜单首帧 | 无 | 可见 P95≤10 |
| `life_loss` | diagnostic | — | 可见：HUD lives 下降（death 的确认，不重复计 death） | 无 | — |
| `menu_exit` | diagnostic | — | 可见：菜单关闭/新局 | 无 | — |
| `respawn` | diagnostic | — | 可见：机体在底部重新出现首帧 | 无 | 诊断 |
| `hit_pending` | diagnostic | — | 可见：闪烁/白闪首帧 | `0x0002` | 诊断 |
| `resource_reset` | diagnostic | — | 可见：reset 窗口内的资源跳变 | 无 | — |
| `resource_extend` | diagnostic | — | 可见：非 reset 窗口 lives 上升 | 无 | — |
| `bomb_stock_move` | diagnostic | — | 可见：死亡帧/消解期 bomb 存量移动 | 无 | — |
| `restart` | diagnostic | — | 可见：像素终局驱动的重开完成 | 无 | — |
| `unknown_start`/`unknown_end` | diagnostic | — | 观测有效性 | 无 | — |

### 2.1 两套锚点（绝不合并）

- `AnchorKind.VISIBLE`：`A` = 人类首次能看出该事件的那一帧，盲标冻结；
  `visible_anchor()` 只接受可见证据类（`sprite_vanish`/`death_effect`/
  `hud_lives_decrement`/`respawn_only`/`stage_sequence`/`terminal_panel`/...）。
- `AnchorKind.ENGINE`：`E` = 引擎位/索引帧（`engine_death_bit`、`engine_stage_clear_bit`、
  `engine_stage_index`...），**只进离线标签**，运行时禁止读取。
- `EventRecord` 同时给出 `predicted_frame`（alarm/确认帧）、`evidence_frame`（锚点）、
  `confidence`、`anchor_kind`、`evidence`、`subtype`、`offline_engine_frame`；
  `visible_latency = predicted - evidence`，可见锚点事件禁止 `predicted < evidence`（构造即报错）。

### 2.2 death 首次可见证据

`resolve_death_anchor()`：按 `sprite_vanish > death_effect > hud_lives_decrement >
respawn_only` 的优先级取**最早**可用帧；全不可见时**抛错**而不是返回 0，
必须用 `unobservable_truth(event, reason)`（`frame=None`，留在分母）。

延迟（`events.latency_gate` / `latency_summary`）：

- 主判据 `visible_latency`：匹配窗 `[A, A+10]`，`before=0`；**`alarm < A` 记假正例**，
  不允许"提前猜中"算命中。
- 次判据 `engine_latency`：P95≤30、max≤60，允许负值（视觉早于引擎位），报带符号分布。
- P95 = nearest-rank `ceil(0.95n)-1`，同时报 median/max/n；`n=0` 时 `status="zero_samples"`
  且 `pass=false`，不给 0% 或 100%。
- detector 的 sustain/确认帧**计入**延迟（`test_typical_alive_hit_respawn_alive_cycle`
  中 respawn 的确认 5 帧被计入 `visible_latency`）。

### 2.3 一对一匹配

`match_events(truth, predicted, window=(0,10))`：贪心一对一；未命中预测为假正例；
每个预测最多命中一个真值；空真值/空预测由 `proportion()` 显式返回 `zero_samples`。

## 3 资源变化归因（不允许把 reset 当 death）

`ResourceChange` 的 `cause` 是封闭词表：
`death / extend / bomb / bomb_pickup / death_frame_stock_move / reset / score`。

| 变化 | 条件 | cause | 事件 | 禁止 |
|---|---|---|---|---|
| lives ↓ | reset 窗口（`STAGE_TRANSITION`/`GAME_OVER`/`stage+restart` 后 30 帧） | `reset` | resource_reset | 计 death |
| lives ↓ | 非 reset 窗口 | `death` | death（若无可见证据则 evidence=`hud_lives_decrement`）+ life_loss | 计两次 death |
| lives ↑ | reset 窗口 | `reset` | resource_reset | 计 extend |
| lives ↑ | 非 reset 窗口 | `extend` | resource_extend | 计 death/reset |
| bombs ↓ | reset 窗口 | `reset` | resource_reset | 计 bomb |
| bombs ↓ | 死亡帧 / `RESPAWN` / `GAME_OVER` | `death_frame_stock_move` | bomb_stock_move | 计 bomb |
| bombs ↓ | 其他（含 `HIT_PENDING`） | `bomb` | bomb（+ 延迟确认 death_bomb） | 计 death |
| bombs ↑ | reset / 非 reset | `reset` / `bomb_pickup` | 对应 | 计 bomb |
| score ↓ | 任意 | `reset`（引擎无扣分路径） | resource_reset | 当受伤/惩罚 |
| score ↑ | 任意 | `score` | 无（奖励用 settled displayScore） | 当事件 |
| 读数 `None` | 任意 | 不产生变化 | 无 | 补 0/补上一值 |

跨 unknown 的读数差**不**回填（`test_an_unknown_reading_is_not_backfilled_into_a_death`）。

## 4 逐帧标签 schema（`labels.py`）

必填字段（`REQUIRED_LABEL_FIELDS`）：

`frame, state, event, visible_anchor, engine_anchor, resource_li, resource_bo,
resource_sc, provenance, trajectory_id, parent_trajectory_id, ancestor_id, split_group`

可选字段：`anchor_version, evidence, confidence, hud_unknown, unobservable,
unobservable_reason, score_settled, subphase`。序列化时带 `schema =
v410-wp06-frame-label/1`。

规则：

- `None` = unknown，**不是 0/False**；任一 `resource_li/bo/sc` 为 `None` 必须
  `hud_unknown=true`，否则校验失败。
- `visible_anchor` 必须 ≤ `frame`；`engine_anchor` 可早可晚（stage_clear 引擎位比像素
  起点晚约 112 帧是合法的）。
- 有 `event` 必须有锚点，或 `unobservable=true` + 非空 reason；两者不可同时成立。
- `provenance ∈ {annotation, pixel_detector, engine_offline, synthetic, mixed}`；
  `split_group ∈ {train, dev, confirmation}`。
- `FrameLabel.as_dict()` 先重新校验，坏行**不可能**被写进证据包。
- `labels_from_step(step, obs, ...)`：每个发射事件一行（同一帧可有 death + life_loss），
  无事件帧产出一行 `event=None` 的**负样本**。

## 5 轨迹/祖先隔离与确认集封存

`split_isolation_report()` / `require_split_isolation()`：

- 切分单位是**原轨迹及其祖先**（seed、RESET 链、restart 谱系、replay 派生）。
- 失败关闭：同一 `ancestor_id` 出现在两个 split；同一 `trajectory_id` 有两个祖先或两个
  split；空 split；split 的轨迹占比低于 `train 0.60 / dev 0.20 / confirmation 0.20`。
- 报告给出 `trajectory_counts`、`trajectory_fractions`、`ancestor_counts`、
  `ancestor_conflicts`、`trajectory_conflicts`、`empty_splits`、`below_minimum`。
- 报"同一局换 10 个窗口"不是独立样本（README WP06 第 3 条）。

`ConfirmationSeal`（`v410-wp06-confirmation-seal/1`）：

- 必填：`seal_id/created_at/anchor_version/detector_version/state_machine_version/
  engine_sha256/component_sha256/seed_ranges/trajectory_ids/event_counts`；
  `anchor_version="unfrozen"` **不能**封印，`component_sha256` 必须含 events/
  state_machine/labels 三个文件的 sha256。
- 一次性：`open_confirmation()` 只成功一次，第二次抛 `ConfirmationSealError`；
  `opened_at/opened_by/one_shot` 作为封存标志写入 manifest。
- `confirmation_readiness()` 检查：未开启、轨迹非空、无零样本类、达到 README 95% 计数口径。
- `validate_confirmation_labels()`：必须全部 `split=confirmation`、轨迹在 seal 列表内、
  `anchor_version` 匹配、`provenance` **不得**为 `synthetic`/`mixed`。

## 6 事件计数与 400/550（字段化，不在这里实现 detector）

`event_count_requirements()` / `event_count_report()`：

| 口径 | 置信度 | z | 零失败所需 n | 400 零失败下界 |
|---|---|---|---|---|
| README 95% 单侧 | 0.95 | 1.959964 | **381** | 0.990488（过） |
| 99% 置信度 | 0.99 | 2.326348 | **536** | 0.986651（不过） |

- `minimum_events_per_class=400`（README），`recommended_events_per_class=550`
  （两种口径都过且留余量）。
- `event_count_report()` 每类给 `meets_minimum / meets_readme_95 / meets_strict_99 /
  meets_recommendation` 与 `status`（`zero_samples`/`below_minimum`/
  `meets_95_reading_only`/`meets_both_readings`/`meets_recommendation`）；
  `n=0` 永远不是 pass，也不报 0%。
- `event_sample_report()` 同时给正样本（逐事件）、负样本（无事件帧）、
  `unobservable_labels`、`hud_unknown_labels`、`unknown_state_labels` 与计数就绪度。
- 这些是**契约校验**，不是 detector 结果；不实现 detector、不选阈值、不调参。

## 7 测试、命令与退出码（实际执行）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp061_pytest -q `
  tests/test_v410_reward_events.py tests/test_v410_reward_state_machine.py
# → 71 passed in 0.12s；EXIT=0（-p no:cacheprovider + 独立 basetemp，绕开 .pytest_cache）
python -m compileall -q touhou_rl/v410/reward
# → EXIT=0（仅语法检查，不等于测试通过）
```

覆盖（合成，无引擎/无真实帧/无训练）：

| 类别 | 测试 |
|---|---|
| 典型 alive→hit_pending→respawn→alive | `test_typical_alive_hit_respawn_alive_cycle` |
| death 首可见锚点 vs 引擎锚点 | `test_visible_death_anchor_differs_from_the_engine_anchor`；`test_engine_oracle_does_not_change_a_single_prediction` |
| stage_clear visible/engine 两事件 | `test_stage_clear_has_two_clocks_and_two_events` |
| game_over/menu 分离、retry 子类型 | `test_retry_panel_is_game_over_and_not_a_menu`；`test_pause_menu_enters_and_exits_without_events_of_death` |
| 资源 reset/extend/bomb/stock move | `test_stage_reset_is_attributed_to_reset_not_death`；`test_lives_extend_is_not_a_death_nor_a_reset`；`test_bomb_stock_move_on_the_death_frame_is_not_a_bomb`；`test_bomb_during_hit_pending_confirms_a_death_bomb` |
| unknown 不补零/不回填 | `test_unknown_state_never_zero_fills...`；`test_an_unknown_reading_is_not_backfilled_into_a_death` |
| 无事件长段（600 帧） | `test_long_alive_segment_has_no_events` |
| 轨迹隔离/祖先冲突/空 split/占比 | 4 个 `test_split_isolation_*` |
| 确认集一次性/封存/合成禁用 | 5 个 `test_confirmation_*` / `test_seal_*` |
| 坏输入 fail-closed | `test_malformed_observations_fail_closed`；`test_frame_sequence_regressions_and_gaps_fail_closed`；`test_unpinned_engine_data_and_bad_configs_are_rejected` |
| 匹配/延迟/计数口径 | `test_match_window_*`、`test_wilson_conventions_*`、`test_event_count_*` |

`contract.json.validation` 记录了命令、退出码、71 passed、43+28 分文件计数、
`engine_started=false`、`data_collected=false`、`trained=false`。

## 8 限制与不宣称

- **不是 G410-3 通过**：没有 detector、没有真实连续流、没有确认集运行。
- 没有采数据/训练/启动游戏；没有改 v400/v380/v390、`tools/rl/v400_*`、
  `tests/test_v400_*`、`third_party/`、既有 v410 文件、`tools/rl/v410*.py`、
  既有 `tests/test_v410_*`、state/journal/budget/assets、其它 WP 证据；未 commit/push。
- 合成测试只证明契约自洽，不证明像素事件可识别；WP06.0 的 NCC 16.41%/CNN 8.38%/
  HUD 差分 64.5% 仍是最新反证。
- `w_hit`/`w_confirm` 是 placeholder，不是门常数。
- `A_stage_visible` 尚未独立标注冻结；不能把旧 vgrad 起点当成可见锚点。
- 400 事件只过 README 95% 口径；99% 置信度需 ≥536，建议 550。
- 旧 G2 的 `[edge, edge+10]` 窗、95% 下界、阈值都没有被改名成 G410-3。

## 9 下一步

1. **WP06.2**：全分辨率彩色 HUD 读数（图标/数字模板优先，学习 fallback）+ 玩家定位
   （目标 median ≤2 px、P95 ≤4 px），把 `labels.py` 的 `resource_*` 真值接上逐帧 HUD。
2. **WP06.3**：像素事件 detector（局部窗 + 状态机 + HUD 确认层），在 dev 上按本契约
   的 5 类事件、延迟门与计数口径评测；仍不得打开确认集。
3. **WP06.4**：Normal s0–s7 连续流采集与冻结（含 men/death-bomb/无 bomb 死亡/遮挡/
   慢速/暂停/长无事件），用 `split_isolation_report` + `seal_confirmation` 固化。
4. **WP06.5**：一次性确认集；`tools/rl/v410.py check --wp WP06` 目前仍未实现，需先补。
