# WP01.0 — v4.1.0 最小协议与观测/动作适配器

- 状态：**完成**（只写代码与测试；未启动游戏、未训练、未采集、未读大文件）
- 日期：2026-11-13
- 范围：`touhou_rl/v410/protocol.py`、`touhou_rl/v410/adapters.py`、`configs/v410/protocol.json`、`tests/test_v410_protocol.py`
- 证据：本文件 + [`commands.jsonl`](commands.jsonl)（每行一条命令记录）
- 协议摘要 `protocol_digest()`：`a58bb8b8075454c610f87305995fb584c7522290d3312eeb74c75daa2b501991`
- 最终验收：`57 passed`（exit 0）、`compileall` exit 0、config 解析 exit 0

## 1. 完成 / 未完成

已完成（本切片验收契约全部落地）：

| 契约 | 位置 | 证据（测试） |
| --- | --- | --- |
| 常量 `PROTOCOL_VERSION`…`TERMINATION_TYPES` 全部取指定值 | `protocol.py` | `test_protocol_constants_are_exact` |
| `ProtocolError(RuntimeError)` | `protocol.py` | `test_protocol_constants_are_exact` |
| `assert_seed_allowed`：train / dev / confirm / diagnostic 四类 purpose、排除区、错误信息含 seed+purpose+范围 | `protocol.py` | `test_seed_*`（7 个函数 / 29 个 item，含 cross-purpose） |
| `validate_action_history`：取最后 8 个、0..17、oldest-first、返回规范化 tuple、长度/范围拒绝 | `protocol.py` | `test_validate_action_history_*`（2 个） |
| `decision_alignment`：4 帧 block = 2 个 2 帧 slot、标签只属当前 slot | `protocol.py` | `test_decision_alignment_*`（3 个） |
| `protocol_digest`：canonical JSON 的 sha256，64 位小写 hex、稳定 | `protocol.py` | `test_protocol_digest_*`（2 个） |
| `normalize_plane`：随机 uint8 与 `astype(float32)/255.0` 逐元素且逐字节相等 | `adapters.py` | `test_normalize_plane_matches_float32_reference` |
| `stack_frames`：4 帧 oldest-first、不重排、长度/形状拒绝 | `adapters.py` | `test_stack_frames_*`（3 个） |
| `build_action_history`：8×18 one-hot、排除 current、前置 pad、reset 清空、越界拒绝 | `adapters.py` | `test_build_action_history_*`（7 个） |
| `encode_observation`：4 键 dict、不改写输入、shape/dtype 拒绝 | `adapters.py` | `test_encode_observation_*`（2 个） |
| `configs/v410/protocol.json`：可解析、常量与 Python 一致、与 `config_payload()` 完全相等 | `configs/v410/protocol.json` | `test_config_*`（5 个） |
| `assert_cpu_gpu_parity` 留待 WP01.1 | `adapters.py` | `test_assert_cpu_gpu_parity_is_deferred_to_wp011`（断言 `NotImplementedError`） |

未完成 / 明确不在本切片：

- `assert_cpu_gpu_parity`：仅占位并抛 `NotImplementedError`，**不构成能力声明**，CPU/GPU 一致性属 WP01.1。
- 未接入真实模拟器帧：全部测试使用合成 uint8 数组（`np.random.default_rng(20260401)`，并强制含 0/255 边界）。真实帧来源、HUD 裁剪、4 帧对齐需 WP01.1/WP02 用真实采集验证。
- 未修改 `tools/rl/v410.py`（白名单外），因此 `check --wp WP01` 仍返回 exit 3（未实现）。
- 未把 seed 策略接入 v3/v4.0 采样器或 CLI；本切片只交付协议与适配器。

## 2. 变更文件（严格白名单）

| 文件 | 说明 | 行数 |
| --- | --- | ---: |
| `touhou_rl/v410/protocol.py` | 新增：常量、`ProtocolError`、seed 分区与 `assert_seed_allowed`、`validate_action_history`、`decision_alignment`、`canonical_json`/`protocol_constants`/`protocol_digest`、`config_payload` | 383 |
| `touhou_rl/v410/adapters.py` | 新增：`normalize_plane`、`stack_frames`、`build_action_history`、`encode_observation`；`assert_cpu_gpu_parity` 占位 | 245 |
| `configs/v410/protocol.json` | 新增：`schema="v410-protocol-config/1"`、常量、`seed_ranges`、`termination_types`、`decision_alignment` | 70 |
| `tests/test_v410_protocol.py` | 新增：57 个测试 | 606 |
| `Document/Update/Alpha/v4.1.0/evidence/WP01/REPORT.md` | 新增：本报告 | — |
| `Document/Update/Alpha/v4.1.0/evidence/WP01/commands.jsonl` | 新增：9 条命令记录 | — |

未修改：`touhou_rl/v400/`、`v380/`、`v390/`、`tools/rl/v400_*`、`tests/test_v400_*`、`third_party/`、`Document/Update/Alpha/v4.0.0/`、`state.json`、`journal.jsonl`、`budget.jsonl`、`assets.json`，以及 `touhou_rl/v410/__init__.py`、`touhou_rl/v410/evidence.py`、`tools/rl/v410.py`（三者 mtime 仍为 WP00 写入时间：`__init__.py` 23:59:01、`evidence.py` 00:01:33、`tools/rl/v410.py` 未被触碰）。
未 commit / 未 push；未删除或覆盖旧文件。

副产物说明：

- `python -m compileall` 在 `touhou_rl/v410/__pycache__` 刷新字节码（编译校验必需，不涉及源码）。
- 未生成 `.tmp_v410_wp01_pytest`：本测试不使用 `tmp_path`，pytest 未物化 basetemp；`.pytest_cache` 的 mtime 仍为 2026-10-18，证明 `-p no:cacheprovider` 生效、未写缓存。

## 3. 实际命令与退出码

环境：Windows PowerShell，Python 3.12.10，numpy 2.5.4，pytest 9.1.1，cwd = `D:\WorkShop\Touhou_RL`，每次运行前 `$env:PYTHONPATH='D:\WorkShop\Touhou_RL'`。逐条 argv 见 `commands.jsonl`。

| # | 命令（简写） | 退出码 | 实际结果 |
| --- | --- | ---: | --- |
| 1 | `python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp01_pytest -q tests/test_v410_protocol.py`（首跑）| **1** | `1 failed, 54 passed in 0.32s`；失败原因是测试自身断言过严（要求 `diagnostic.rationale` 含子串 `not`，文案实为 `non-confirmation`）|
| 2 | 同上（修正该测试断言后重跑）| **0** | `55 passed in 0.19s` |
| 3 | `python -m compileall -q touhou_rl/v410 tools/rl/v410.py` | **0** | 无输出（仅语法通过，不等于测试通过）|
| 4 | `python -c "…json.loads(Path('configs/v410/protocol.json')…)…"` | **0** | `v410-protocol-config/1 2 4` |
| 5 | `python -c "…protocol_digest(); assert_seed_allowed(1,'train'), (1010,'dev'), (61000,'confirm')…"` | **0** | `digest a58bb8b8…01991` / `env-check 1 1010 61000` |
| 6 | `git status --porcelain -- touhou_rl/v410 configs/v410 tests Document/Update/Alpha/v4.1.0` | **0** | 仅 WP00 基线与本切片文件为未跟踪；无白名单外改动 |
| 7 | `python -m pytest … --collect-only -q tests/test_v410_protocol.py` | **0** | `55 tests collected in 0.12s`（当时状态，用于核对 item 分类）|
| 8 | `python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp01_pytest -q tests/test_v410_protocol.py`（补 `validate_action_history` 直测后，最终态）| **0** | `57 passed in 0.21s` |
| 9 | `python -c "…校验 commands.jsonl 每行字段…"` | **0** | `records 8 ok [1, 0, 0, 0, 0, 0, 0, 0]`（校验当时已有的前 8 条；本次校验自身为第 9 条）|

第 1 次失败的退出码 1 如实保留，未删除证据。

## 4. pytest 统计

```
$ $env:PYTHONPATH='D:\WorkShop\Touhou_RL'
$ python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp01_pytest -q tests/test_v410_protocol.py
.........................................................                  [100%]
57 passed in 0.21s
```

- 通过 57 / 失败 0 / 跳过 0 / 错误 0（单文件 `tests/test_v410_protocol.py`）
- item 分布：常量 1、`normalize_plane` 2、`stack_frames` 3、`build_action_history` 7、`validate_action_history` 2、`encode_observation` 2、CPU/GPU 占位 1、seed 策略 29（4+10+1+1+1+6+6，含 parametrize）、`decision_alignment` 3、digest 2、config 5 → 合计 57
- 三项硬性断言均被实际执行并通过：
  1. 随机 uint8（含 0 与 255）归一化结果与 `astype(np.float32)/255.0` 逐元素 `array_equal` **且** `tobytes()` 逐字节相等；
  2. `stack_frames` 输出顺序为 oldest-first（`out[0]` 为 `t-3`），且不做任何重排；
  3. `build_action_history` 排除 current、前置补零、8×18 one-hot、越界拒绝、`reset=True`/空列表清空。

## 5. 关键契约细节

- seed 分区（含边界，均闭区间）：
  - `train` 1..49999，排除 1000..1099 与 60000..60199；
  - `dev` 1000..1099（仅开发诊断 / 候选选择 / 预留开发检查）；
  - `confirm` 61000..61499（新正式确认批次，禁止用于 train）；
  - `diagnostic` 1..49999，排除 60000..60199，配置与报告均标注 **非确认**。
  - 错误示例：`assert_seed_allowed: seed 1019 is reserved and forbidden for purpose 'train': allowed 1..49999 excluding 1000..1099`；越界示例：`… seed 61000 is outside the allowed range for purpose 'train': allowed 1..49999`。
  - 非 `int`（含 `bool`、`float`、`str`、`None`、list）一律 `ProtocolError`。
- decision 对齐：4 帧 block = 2 个 2 帧 slot（`slot_frames=[[0,1],[2,3]]`）；`label_policy` 明确 “one label per decision slot … the second slot's label must never be used as the first slot's label”；`label_bleed_allowed=false`，并列出 3 条禁止的标签复用方式。非 2 的倍数的 block（5/3）、0、负数、字符串均拒绝。
- 动作历史语义：`actions` 为已执行动作（oldest-first）；若末尾等于 `current` 则只剔除该末尾项一次（更早的重复动作属合法历史，见 `test_build_action_history_current_dropped_only_once`）；不足 8 个时在**前面**补零；`reset=True`（或空列表 / `None`）返回全零 —— 对应 stage change。
- `encode_observation` 返回且仅返回 `play` / `play_norm` / `hud` / `act_hist` 四键；输入不被改写，输出与输入不共享内存（`np.shares_memory` 断言）。
- `configs/v410/protocol.json` 与 `protocol.config_payload()` **完全相等**（双向无漂移）；其去掉 `schema`/`notes` 后的 canonical JSON 的 sha256 等于 `protocol_digest()`（`test_config_digest_matches_python_digest`）。

## 6. 剩余阻塞

- 本切片无阻塞项：三条验收命令均已实际执行，最终退出码分别为 0 / 0 / 0（第 1 条首跑 exit 1 的原因与修复见第 3 节）。
- 需移交后续工作包的依赖（不属本切片阻塞）：
  1. CPU/GPU 一致性（WP01.1）缺 GPU 侧实现与固定 fixture，`assert_cpu_gpu_parity` 只能抛 `NotImplementedError`；
  2. 真实帧 / 真实 HUD 尚未接入，合成数组**不能**作为“真实采集已对齐”的证据；
  3. `tools/rl/v410.py` 未获授权修改，`check --wp WP01` 仍为 exit 3，WP01 证据门需在允许修改 CLI 的工作包中接入。

## 7. 下一建议

1. WP01.1：CPU(numpy)/GPU(torch) 归一化与堆叠的逐元素 + 逐字节一致性检查；固定 fixture（0/255 边界 + 随机 uint8），并用真实采集帧各跑一次。
2. 在 env/observation builder 中接入 `stack_frames` + `encode_observation`，补端到端 4 帧 rollover 测试（含首帧 warm-up 的填充策略决策）。
3. 在 seed 分配处调用 `assert_seed_allowed`，让 1000..1099 与 61000..61499 的误用直接 fail-closed。
4. 在允许修改 `tools/rl/v410.py` 的工作包中实现 `check --wp WP01`：校验本报告 + `commands.jsonl` + digest 一致性，作为 WP01 证据门。


## 8. 主代理独立复核（2026-11-13）

- 独立重跑 `pytest -q tests/test_v410_protocol.py`：**57 passed，exit 0**；`compileall -q touhou_rl/v410 tools/rl/v410.py`：**exit 0**；`configs/v410/protocol.json` 解析：**exit 0**（`v410-protocol-config/1 2 4`）。
- 独立抽查 seed 策略：train 接受 1/999/1100，拒绝 1000/1019/60000/60199/61000；dev 接受 1010；confirm 接受 61000；`protocol_digest()` 稳定为 `a58bb8b8075454c610f87305995fb584c7522290d3312eeb74c75daa2b501991`。
- `git status --porcelain -- touhou_rl/v400 tests/test_v400_protocol.py` 无输出：本切片未改 v400 代码/测试。
- 主代理复核后把 `adapters.build_action_history` 的 docstring 写清：`current` 提供且等于末项时只剔除一次；`current=None` 时 `actions` 视为仅过去动作。行为未变，测试仍 57 passed。
- `check --wp WP01` 仍为 exit 3（工具尚未实现 WP01 检查器）；**WP01 仍未 passed**，因为 WP01.1 的真实引擎一致性切片与 30 分钟运行尚未完成。
- 下一动作：WP01.1（真实引擎 3 训练种子 × 3 重开 × 1,000 帧一致性、CPU/GPU parity、至少 30 分钟小规模运行、信息边界测试）。
