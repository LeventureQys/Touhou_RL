# WP01.2a — 信息边界、严格反馈与终止语义（证据）

- 工作包：v4.1.0 WP01.2a
- 模块：`touhou_rl/v410/boundary.py`
- 测试：`tests/test_v410_boundary.py`
- 机器可读证据：`Document/Update/Alpha/v4.1.0/evidence/WP01/boundary.json`
- 环境：Windows / Python 3.12.10 / numpy 2.5.4 / torch 2.14.1+cu132
- 活动声明：**未启动游戏、未训练、未采集数据、未读取封存资产、未 commit/push**
- 测试数据：全部为合成像素（`numpy.random.default_rng(20241113)`），不读任何 D1 分片或 checkpoint

## 1. 实际执行的验收命令与退出码

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp012_pytest -q tests/test_v410_boundary.py
# 输出：151 passed in 0.24s
# 退出码：0

python -m compileall -q touhou_rl/v410
# 输出：（无）
# 退出码：0
```

pytest 统计：**collected 151 / passed 151 / failed 0 / skipped 0 / error 0**，最终一次用时 0.24s
（多次重跑 0.24–0.30s）。
这些 151 是**单元测试用例数**，不是游戏运行局数，也不构成能力证据。

过程说明（不隐瞒）：首次运行收集 151 例，报 150 passed / 1 failed，原因是测试自身
`play_norm.reshape(4, 112, 95)` 对 43008 个元素做了非法 reshape；改成 `[:, :, :95]`
后重跑得到 151 passed、退出码 0。失败发生在测试文件，不在 `boundary.py`。

## 2. 信息边界

`PRIVILEGED_KEYS`（17 项，集合为 `frozenset`）必须始终留在像素边界之外：

```
deaths lives bombs stage game_over teacher_margin future_danger
verified_mask snapshot_start engine_reward engine_done referee_state
engine_score referee_input teacher_logits rng_state seed_state
```

- `strict_observation(play_stack, hud, act_hist)`：只返回 `play / play_norm / hud / act_hist`
  四键；dtype/形状/非数组错误经 `encode_observation` 抛 `ProtocolError`；返回数组不与入参别名。
- `assert_no_privilege(mapping)`：递归拒绝任意层级的特权 key，大小写/空白不敏感
  （`"  DEATHS "` 同样被拒）；用于严格训练/评测入口。
- `StrictPixelPolicy`：只接受数组（`logits`）或恰好四键的观测（`logits_from_observation`），
  因此不存在把裁判状态送进策略的通道；实例无状态（`vars(policy) == {}`）。
- `strict_feedback` / `classify_termination`：同样先过 `assert_no_privilege`，特权字段一律
  `ProtocolError`，绝不读取。

## 3. 严格反馈语义

允许字段（仅像素可观测）：`death_visible`、`stage_clear_visible`、`game_over_visible`、
`menu_visible`、`paused_visible`、`resource_delta`、`frame`。

- 返回恰好 `{reward, done, reset, sample_selected}`。
- 特权 key：`ProtocolError`（拒绝）；普通未知 key：忽略并写入可选 `record` 列表（可审计）。
- `done == reset == (death_visible or stage_clear_visible or game_over_visible)`。
- `reward = clip(resource_delta, [-1,1]) − 1·death + 1·stage_clear − 1·game_over`；
  menu/pause 帧 reward 固定 `0.0`（非游戏帧不给学习信号）。
- `sample_selected` 仅在“非 menu/pause 且 `frame % 2 == 0`（决策槽边界）”为真。

## 4. 终止语义

类型：`first_death | stage_clear | game_over | cap`（与 `protocol.TERMINATION_TYPES` 一致）。

优先级（高 → 低）：`game_over > stage_clear > first_death > cap`。

- `game_over_visible` 最高：失败局不得被残留的过面横幅掩盖。
- `stage_clear_visible` 高于死亡动画：把已过面记成死亡等于伪造失败。
- 终止事件永远压过 `cap`；`cap` 只在“无任何终止事件且 `frame >= cap`”时返回。
- 无事件且 `frame < cap` → 返回 `None`（仍未终止）。
- **menu/pause 不计作存活**：无终止证据时抛 `ProtocolError`，既不返回 `None`（存活），
  也不返回 `cap`；若同帧有明确终止像素证据，则终止事件优先。
- 非法输入（负 frame、bool、非正 cap、特权字段、非映射）一律 `ProtocolError`。

## 5. 资源套件

| suite | 允许资源 | 用途 |
| --- | --- | --- |
| `standard` | `S1N`、`RUNN` | 自然开局、标准残机/bomb |
| `practice` | `PSN` | 仅练习起点，不与标准局结果池化 |

交叉使用（`standard`+`PSN`、`practice`+`S1N`/`RUNN`）、未知名称、空选择、重复项、
非字符串均抛 `ProtocolError`；两个 suite 的结果永不合并。

## 6. 覆盖矩阵（检查项 → 用例数）

| id | 检查项 | 用例 |
| --- | --- | --- |
| B01 | 观测键恰好四键 | 3 |
| B02 | 观测输入契约（非数组/形状/dtype） | 6 |
| B03 | 特权字段全面拒绝（含嵌套、大小写、反馈与终止入口） | 57 |
| B04 | 策略仅数组/四键入口 | 3 |
| B05 | 策略确定性、无状态、对像素敏感 | 22 |
| B06 | 反馈只读像素字段 | 4 |
| B07 | 未知字段：忽略并记录；非法类型拒绝 | 14 |
| B08 | 终止优先级、事件压过 cap、非法参数拒绝 | 16 |
| B09 | menu/pause 不计存活 | 2 |
| B10 | cap 规则 | 1 |
| B11 | standard / practice 组合校验 | 21 |
| B12 | 入参不被改写、返回副本 | 1 |
| B13 | 无引擎活动（设计约束，0 用例） | 0 |
| B14 | `boundary_report()` 可 JSON 化且不伪造结果 | 1 |
| **合计** | | **151** |

## 7. policy stub 说明（必读）

`StrictPixelPolicy` 是 **interface stub, not a capability claim**：

- 固定线性投影，权重由 splitmix64 整数混合从固定 salt 生成，无 RNG、无全局状态、无学习；
- 输入仅 `play_norm`（float32/float64，`(4,112,96)`）、`hud`（uint8，`(56,160)`）、
  `act_hist`（uint8，`(8,18)`），输出 18 维 `move_logits`（float64）+ 1 个 `bomb_logit`；
- 相同像素/历史逐位相同（跨实例、跨调用序列）；任一像素变化都会改变输出（敏感性）；
- **它不是真实策略，没有训练、没有评测，不能据此声称任何能力、胜率或样本效率。**

## 8. 限制

1. `StrictPixelPolicy` 只是接口桩，固定权重，无训练；logits 的“好坏”未被评估，也不构成绩效证据。
2. 没有引擎反馈：本 WP 未接触引擎/零售件，`*_visible`、`resource_delta` 由本 WP 之外
   的像素采集端提供，`boundary.py` 只定义契约与校验。
3. `menu/pause` 采用 fail-closed（抛错）而非自动恢复策略，调用方须显式处理冻结帧。
4. `validate_resources` 只校验 suite/资源名称，无法验证一局真实资源。
5. 151 个用例是合成单元测试，不构成任何真实游戏结果；本 WP 未启动游戏、未训练、未采集。
6. CPU/GPU 一致性不在本 WP 范围（见 WP01.1），本文件不作相关声明。

## 9. 变更文件

| 文件 | 类型 |
| --- | --- |
| `touhou_rl/v410/boundary.py` | 新增 |
| `tests/test_v410_boundary.py` | 新增 |
| `Document/Update/Alpha/v4.1.0/evidence/WP01/BOUNDARY.md` | 新增 |
| `Document/Update/Alpha/v4.1.0/evidence/WP01/boundary.json` | 新增 |

未修改白名单外任何文件；未改 `protocol.py` / `adapters.py` / `evidence.py` / `parity.py` /
`runtime.py` / `tools/rl/*`，未改 v400/v380/v390，未改 `state.json` / `journal.jsonl` /
`budget.jsonl` / `assets.json`。
