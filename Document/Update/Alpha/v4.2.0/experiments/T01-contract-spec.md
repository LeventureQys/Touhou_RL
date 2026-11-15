# T01 契约规格：obs-v420-2f / action-v420-2f

依据：[v4.2.0 版本计划](../README.md) §5（WP420-01/02/03）、[首轮执行任务单](../首轮执行任务单.md) §3（T01）。
本文件只定义**接口与判定规则**，不代表已实现、更不代表已通过。

- 状态：待实现（T01）
- 负责代理：Provider=DeepSeek / DeepSeek Flash
- 允许修改白名单：`touhou_rl/v420/{__init__,contracts,replay_audit,counterfactual,eval}.py`、`tools/rl/v420.py`、`configs/v420/`、`tests/test_v420_{contracts,replay_audit,eval}.py`、`Document/Update/Alpha/v4.2.0/evidence/T01/`

---

## 0 已由主代理实测确认的 v4.1.0 事实（用作实现与验收基准）

在 `.tmp_v410_s1c/d1/shards/d1-s0d1-n3300-seed10-shard`（3,300 行）上实测：

| 事实 | 实测值 | 证据 |
|---|---|---|
| 每行推进的逻辑帧 | **4**（`decision_frame` 步长全部为 4，直方图 `{4: 3299}`） | `.tmp_v420_slot_probe.py` |
| `frames_per_decision` / `decision_frames` | 4 / 2（同一行内自相矛盾） | `labels.jsonl`、`shard.json` |
| 槽位总数 vs 有自己观测的槽位 | 6,600 vs 3,300（50%） | `decision_count=3300`, `frames_committed=13200` |
| 标签归属 | `alignment.label_is_first_slot_not_continuation=true`，`scores[18]` 只评 slot 0，作用于帧 `[d, d+2)` | `shard.json.alignment` |
| 槽 1 的动作 | 每行可恢复（`chosen_plan.slots[1]`），但**无观测、无 scores、无 verified** | `labels.jsonl` |
| 逐帧实际 mask | `step_frames[0:2] == cand_to_mask(slots[0], bomb)` **3300/3300**；`step_frames[2:4] == cand_to_mask(slots[1], 0)` **3163/3300**，差 137 | `.tmp_v420_mask_probe.py` |
| 137 处差异的性质 | 全部 `shoot_release=True`，全部只发生在**块内第 4 帧**（`step_frames[3]` 清掉开火位，如 0x85→0x84） | `.tmp_v420_release_probe.py` |
| `act_hist` 组成 | `act_hist[r]` = `first_action[r-8..r-1]` 的 one-hot，前补零，**0 失配** | `.tmp_v420_hist_probe.py` |
| `act_hist` 时间跨度 | 8 个 4 帧决策 = **32 逻辑帧**；部署侧 8 个 2 帧决策 = **16 逻辑帧** | 上两行合并 |
| `act_hist` 是否含槽 1 的动作 | **否**（槽 1 已执行的动作从不进入历史） | 同上 |
| HASH 链 | `hash_post[i] == hash_pre[i+1]`，**0 断点** | `.tmp_v420_slot_probe.py` |
| `bomb` | 全 0（`bomb nonzero: 0`） | `.tmp_v420_hist_probe.py` |

结论（**G420-0 的直接输入**）：旧 4 帧数据**不能**直接当 2 帧数据使用。三项结构性不一致：
1. 槽位覆盖 50%（槽 1 有动作无观测无标签）；
2. 历史语义 32 帧 vs 部署 16 帧，且漏掉槽 1 实际执行动作；
3. 开火释放规则是**块周期**行为（每 4 帧末一次），在 2 帧节拍下不存在对应。

---

## 1 `touhou_rl/v420/contracts.py`

### 1.1 常量

```python
CONTRACT_VERSION = "v420-contract/1"
OBS_CONTRACT = "obs-v420-2f"
ACTION_CONTRACT = "action-v420-2f"
DECISION_FRAMES = 2          # 一次决策覆盖 [d, d+2)
FRAME_STACK = 4              # 最近 4 个已渲染帧，oldest-first
N_MOVE_FOCUS = 18            # focus*9 + move
BOMB_BINARY = 2
ACTION_HISTORY = 8           # 过去 8 个**已实际执行**的 2 帧动作
PLAY_SHAPE = (112, 96)
HUD_SHAPE = (56, 160)
LEGACY_BLOCK_FRAMES = 4      # v4.1.0 采集块，只用于识别旧数据
TERMINATION_KINDS = ("first_death", "game_over", "stage_clear", "cap", "error")
BRANCH_OUTCOMES = ("queried", "executed", "engine_death", "engine_alive", "unknown")
```

张量形状**不变**（obs-v3 相同）——变化的只有时间/动作语义。这样旧 checkpoint 仍能按原 loader 加载，只是必须显式标为「旧契约样本」，不能混入 2 帧训练。

### 1.2 必须实现的数据结构

`SlotRecord`（`dataclasses.dataclass`，或同等不可变结构）至少含：

| 字段 | 语义 |
|---|---|
| `slot_id` | episode 内单调递增槽号 |
| `episode_seed` | 训练/开发 seed |
| `block_index` / `slot_in_block` | 若来源是旧 4 帧块则记录，2 帧新数据 `slot_in_block` 无意义 |
| `decision_frame` | 绝对逻辑帧 d，决策点 |
| `commit_frames` | `(d, d+1)`，恰好 `DECISION_FRAMES` |
| `last_visible_render_frame` | `d - 1`（AfterDrawFrame 先发布再 `++g_frameCount`） |
| `key_send_frame` / `key_effective_frame` | 发送帧与真正生效帧；两者必须可区分且可解释 |
| `duration_frames` | 必须等于 `DECISION_FRAMES` |
| `cand` / `bomb` / `mask` | `cand_to_mask(cand, bomb)` 必须复算出 `mask` |
| `obs_render_frames` | 4 个**连续**已渲染帧 `d-4..d-1` |
| `history` | 8 个已执行 cand，oldest-first，**不含当前、不含未来** |
| `label_source` | `teacher_slot0` / `teacher_slot1` / `none` |
| `label_verified` | 该槽是否有自己的验证结果 |

### 1.3 fail-closed 校验函数（每条都要有对应测试）

1. `check_decision_span(start, duration)`：`duration == DECISION_FRAMES`；`start % DECISION_FRAMES == 0`。
2. `check_history_excludes_current(history, current)`：最后一个元素不得等于当前动作；长度必须为 8。
3. `check_no_future(history_frames, decision_frame)`：历史帧号必须 `< decision_frame`，否则报错。
4. `check_contiguous_render_frames(frames, decision_frame)`：必须是 `decision_frame-4 .. decision_frame-1` 且严格连续。
5. `classify_branch_outcome(...)`：返回 `BRANCH_OUTCOMES` 之一。**`unknown` 永远不得被映射成 `engine_death`**；未验证动作不得进安全集合。
6. `classify_termination(...)`：返回 `TERMINATION_KINDS` 之一，`first_death` 与 `game_over` **互斥且不得合并**；`stage_clear` 优先于 `cap`；`cap` 不等于通关。
7. `assert_seed_allowed(seed, purpose)`：沿用 v4.1.0 分区（train 1..49999 排除 1000..1099 与 60000..60199；dev 1000..1099；confirm 61000..61499）。
8. `require_confirm_enabled(config, candidate_freeze, exposure_registry)`：确认模式默认拒绝，必须同时满足「候选已冻结 + 曝光登记里该批次无冲突」。
9. `check_config_inputs(config)`：拒绝
   - 任何空的资产路径（`root420`/`py420`/`engine420`/`data_dir`/`v420_data` 为空或纯空白）；
   - 输出路径落在旧资产上：`.tmp_v400_`、`.tmp_v410_`、`/root/v400-data/d1`、`Document/Update/Alpha/v4.1.0/`、`Document/Update/Alpha/v4.0.0/` 前缀；
   - 已存在的 run 目录但未声明 `resume: true` 或 `resume_mode`。

---

## 2 `touhou_rl/v420/replay_audit.py`（`audit-time`）

### 2.1 纯函数层（可离线测试）

`audit_legacy_shard(shard_dir) -> LegacyShardFindings`

必须逐项输出并计数（每项都要有 `count` 与最多 5 条 `samples`）：

| 检查 | 期望的旧数据结果 | 判定 |
|---|---|---|
| `rows` / `slots_expected` / `slots_with_obs` / `slots_with_own_label` | 3300 / 6600 / 3300 / 3300 | 覆盖率 50% |
| `decision_frame_stride` 直方图 | `{4: n-1}` | ≠2 即 fail |
| `label_scope` | 只覆盖 slot 0 | fail（对 2 帧契约） |
| `slot1_action_recoverable` | 3300 | 仅作可恢复性记录 |
| `history_stride_frames` | 4（应为 2） | fail |
| `history_contains_slot1_action` | 0 | fail |
| `hash_chain_breaks` | 0 | 这一项 pass |
| `mask_decode_mismatches`（slot 0） | 0 | pass |
| `mask_decode_mismatches`（slot 1） | 137，且全部 `shoot_release` 且全部落在块内第 4 帧 | 记录为 explained |
| `bomb_nonzero` | 0 | 记录 |

`unexplained_diffs` = 所有「fail 且无 explanation 字段」的项。报告 `status` 仅当 `unexplained_diffs == 0` 时为 `pass`。旧数据预期 `status = fail`，但 `unexplained_diffs = 0`（每条失败都有实测解释）——**这正是 G420-0 需要的形态：失败原因全部可解释，不是未知差异**。

注意：`unexplained_diffs == 0` **不**等于 G420-0 pass。G420-0 还要求「新 2 帧样本」这一腿，见 §2.2。

### 2.2 在线层（`--execute`，T02 用）

`audit_live_episode(engine, data_dir, seed, frames, ...) -> LiveAudit`：

- 用固定策略（默认 `idle` 以外的确定性 `cycle`，或无 checkpoint 时 `cycle`）跑 `frames ≥ 2000` 帧；
- 每 2 帧记录一个 `SlotRecord`，逐帧记录：绝对逻辑帧、渲染帧号、决策点、按键发送帧、按键生效帧、实际持续长度；
- 断言与统计：
  - 每个决策的 4 个渲染帧严格连续、最新渲染帧 = `decision_frame - 1`；
  - `history` 长度 8、无未来、无当前；
  - 引擎 HASH 逐帧记录，重放同一段两次的 HASH 序列必须一致；
  - 相同设备上同一冻结模型对相同输入的前向必须可复现（CPU/GPU 差异单列）；
- 输出 `unexplained_diffs`，只有 0 才允许 T02 报 G420-0 的在线腿通过。

---

## 3 `touhou_rl/v420/eval.py`（`evaluate` / `summarize`）

**复用而非重写**：真实引擎评测一律通过调用既有
`tools/rl/v410_student_eval.py`（WSL + `xvfb-run` + `/root/v400-venv/bin/python`）完成。
`eval.py` 只负责：构造 argv、dry-run 打印、执行并捕获退出码、把 v410 评测 JSON 归一为 v420 摘要、算 Wilson 区间、做门判定。

- `build_legacy_eval_argv(config)` → `argv` 列表（不得用 shell 字符串拼接）。
- 默认 dry-run：输出 `{schema, mode: "dry-run", argv, would_write}`，退出码 0，**不启动引擎**。
- `--execute` 必须显式给出。
- `summarize(inputs, out)`：合并多个摘要，输出逐局记录 + 分母/分子 + Wilson 95% 区间；**技术无效的局（error/reset_failed）保留在分母里**，另计 `invalid`。
- 门判定：G420-4 = 100 局、success ≥ 50、Wilson 下界 ≥ 0.35、部署输入只有像素与自身历史。少于 100 局或含未解释 invalid 时如实标 `not_met`，不得外推。

### 3.1 `configs/v420/` 需交付的配置

| 文件 | 用途 | 状态 |
|---|---|---|
| `local.json` | 资产主机路径（T00 产出） | 已由 T00 负责 |
| `audit-time.json` | T02 的 audit-time 输入：3 个训练 seed × 2 次 × ≥2000 帧、`frames_per_decision: 2` | T01 新建 |
| `legacy-reproduce.json` | T02 的 B0/20k 各 3 dev seed × 2 次复现；显式列出 checkpoint 路径与 SHA | T01 新建 |
| `protocol.json` | 本契约常量镜像（与 `contracts.py` 的单测校验一致） | T01 新建 |

---

## 4 `tools/rl/v420.py`

只保留 8 个子命令：

```text
v420.py audit-assets --config ... --out ...
v420.py audit-time   --config ... --out ...
v420.py counterfactual --config ... --out ...   # T01: not_implemented
v420.py train        --config ... --out ...     # T01: not_implemented
v420.py collect      --config ... --out ...     # T01: not_implemented
v420.py rl           --config ... --out ...     # T01: not_implemented
v420.py evaluate     --config ... --out ...
v420.py summarize    --inputs ... --out ...
```

- 默认 dry-run；真实 run 必须 `--execute`。
- 未实现子命令：打印 `{"status": "not_implemented", "implemented_by": "T0x", ...}` 且**非零退出**（用 `EXIT_EVIDENCE = 3`）。
- 退出码沿用 v4.1.0：0 ok / 2 参数或协议错 / 3 证据或依赖不满足 / 4 环境或外部资源缺失 / 5 运行失败或超时 / 6 预算不足。
- `audit-assets` 复用 v4.1.0 的资产核验工具，不重写 SHA 逻辑，但**不得**修改 `ASSET_SYNC.sha256`。

---

## 5 测试要求（`tests/test_v420_{contracts,replay_audit,eval}.py`）

只测真实不变量，数量不预写，**禁止**用「合成通过」当能力证据：

`test_v420_contracts.py`
- 4 帧块 → 2 个槽，每槽 2 帧，边界不重叠；
- `history` 不含当前动作、不含未来动作（构造越界用例必须报错）；
- `unknown` 不映射为 `engine_death`；
- `first_death` 与 `game_over` 分离，`cap` 不判为 `stage_clear`；
- 确认 seed 默认被拒；`require_confirm_enabled` 三条件缺一即拒；
- `bomb` 缺省固定 0（旧 B0 无可用 bomb 策略），不得由事件推断；
- `check_config_inputs` 拒绝空路径、旧资产输出路径、未声明 resume 的既有目录。

`test_v420_replay_audit.py`
- 用 `--basetemp` 下的小 fixture（几行 labels + 小 npy）构造 4 类人为缺陷，断言被检出：
  (a) 槽 1 复用槽 0 标签；(b) `history` 含未来/当前动作；(c) HASH 链断点；(d) 槽 1 无观测；
- 增补：`shoot_release` 块周期差异必须被标为 **explained**，不得计入 `unexplained_diffs`；
- 真实 shard 的集成测试：`pytest.mark.skipif(not shard.exists())`，断言上面 §0 的实测计数（3300/6600、137、0 断点）。

`test_v420_eval.py`
- `build_legacy_eval_argv` 产出的 argv 逐项正确（引擎绝对路径、`--max-frames 19887`、`--stop-on-death`、seed 区间）；
- dry-run 不启动任何子进程（monkeypatch `subprocess.run` 断言未被调用）；
- 缺对象（checkpoint/engine/data）→ 退出码 3 或 4，且报告指明缺哪个；
- 确认模式默认拒绝；给了冻结点但曝光冲突仍拒绝；
- Wilson 区间：全部 100 局全过时下界 > 0.35 且 < 1.0；0/100 时上界 < 0.05；
- 技术无效局保留在分母：含 3 个 error 的 100 局，`denominator == 100`。

验收命令（在资产主机 WSL 内，`ROOT420=/mnt/d/WorkShop/Touhou_RL`）：

```bash
cd "$ROOT420"
PYTHONPATH="$ROOT420" "$PY420" -m pytest -p no:cacheprovider \
  --basetemp "$V420_DATA/tests/T01" -q \
  tests/test_v420_contracts.py tests/test_v420_replay_audit.py tests/test_v420_eval.py
PYTHONPATH="$ROOT420" "$PY420" tools/rl/v420.py --help
```

---

## 6 不得越界

- 不改 `touhou_rl/v400/**`、`touhou_rl/v410/**`、旧 tests、旧 configs、旧证据；
- 不改冻结 ABI / obs-v3 语义 / `obs_spec_v3_digest`；
- 不加第三方依赖；
- 不启动长时间引擎任务（T01 只跑单测与 `--help`；`audit-time --execute` 属于 T02）；
- 不删除、不移动、不覆盖任何 `.tmp_*` 资产；不 git add/commit/push。
