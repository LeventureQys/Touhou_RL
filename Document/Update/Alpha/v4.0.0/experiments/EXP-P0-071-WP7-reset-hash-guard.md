# EXP-P0-071 — WP7：训练/评测启动的 `RESET` 状态哈希守卫（fail-closed，2026-11-09）

**结论：守卫已落地，只做"检测并拒绝"，不声称修根因。** 训练/评测启动现在会读取引擎
`RESET` 回复里的 `hash`，与期望值（`--expect-reset-hash` > `V400_EXPECT_RESET_HASH` >
同一次运行首个成功 reset）比较；不一致时 **不写 bench 行、退出码 4、错误消息含两侧 hash
与完整 `stage_start` 参数**。真实引擎 3 局（`random`/`S1N`/seed 1000/`--max-frames 3000`）
全部退出码 **0**，reset hash 全部为 `3eafc035dc4fe6a9`；负对照（故意给错期望）退出码 **4**
且未产生任何结果文件。本卡**不**解释 5.6% 分歧的成因，也**不**降低其发生概率——它把
"静默污染训练数据"变成"启动失败并显式报错"。

本卡闭合 [EXP-P0-070](EXP-P0-070-WP2-multithread-pixel-determinism.md) §5/§6 第 3 条留下的缺口：
"多进程启动时比对 `RESET` 回复 hash，不一致即 fail-closed"。

---

## 1. 缺口来源（EXP-P0-070 run c，实测）

- 4 次运行共 **18 次引擎启动**，其中 **1 次（≈5.6%）** 在 `RESET 0 1 0 1 0` 后落入不同
  StageStart 状态：该进程回复 `hash=795b9441ed0c8056`，其余 5 个进程回复 `7f04c7361749ed9f`。
- 分歧**早于任何 `STEP`**：第 1 帧 state hash 就已经不同（`6e05414e5ea646a2` vs
  `f1df042d9373e879`），且 `RESET` 回复仍报 `ready=1`。
- 旧启动路径（本卡修改前）只校验 `ready == 1`，因此**无法发现**该分歧：它会照常跑完一局，
  把"另一个初始状态下的轨迹"当成同一配置的测量值写进 bench/训练数据。EXP-P0-070 已证明
  这种差异会被误读成"渲染不确定"（或反过来把非确定数据当成确定）。
- 本卡只处理"启动处检测并拒绝"。**没有**证据表明分歧来自 LP 线程数、RNG、初始化顺序或并发
  启动压力；EXP-P0-070 §5 已明确不声称 LP=8 更易触发。本卡不做因果推断。

## 2. 实现

### 2.1 新模块 `touhou_rl/v400/bench/reset_guard.py`

- `normalize_reset_hash(value)`：只接受十六进制串（大小写归一为小写），空值/非 hex 一律
  `ValueError`。目的：**不允许**把 `"unknown"`、`"ready"` 这类占位符拿来互相比较后"相等"。
- `resolve_expected_reset_hash(explicit, environ)` → `(hash, source)`，优先级严格为：
  1. 显式参数 `--expect-reset-hash <hex>`（`source="cli"`）；
  2. 环境变量 `V400_EXPECT_RESET_HASH`（`source="env"`）；
  3. 都没有 → `(None, None)`，由守卫在第一个成功 reset 时 pin（`source="first-observed"`）。
  环境变量存在但不是 hex → 直接 `ValueError`（启动失败），不静默忽略。
- `ResetHashGuard`：
  - `check(reply, stage_start=...)`：读 `reply["hash"]`；期望为 `None` 时 pin 本次观测，
    之后所有 reset 都与它比较；
  - 回复里没有 `hash`/不可用 → `ResetHashUnavailable`（同样 fail-closed）；
  - 不一致 → `ResetHashMismatch`，携带 `expected`、`observed`、`stage_start`、`source`、
    `attempts`、`observed_hashes`，消息文本包含两侧 hash 与 stage_start；
  - `retries` 上限 `MAX_RESET_RETRIES = 5`，默认 `DEFAULT_RESET_RETRIES = 1`。
- `stage_start_command(...)`：按 `HarnessClient.reset` 的线格式生成
  `RESET <stage> <difficulty> <team> <seed> <practice>`，保证错误消息里写的是**实际发出的**
  参数，而不是事后重建的参数。

### 2.2 `tools/rl/v400_bench_run.py`

- 新增参数：
  - `--expect-reset-hash HEX`（argparse 类型即 `normalize_reset_hash`，非法值退 2）；
  - `--reset-retries N`（默认 1，范围 `[0, 5]`，越界退 2）。
- `run_episode(...)` 新增 `reset_guard=` / `expect_reset_hash=` / `reset_retries=` /
  `environ=`；CLI 构造一个 guard 并传入，便于同一进程内多 episode 共享（来源③）。
- 启动顺序（不变的部分保留）：`start` → `HELLO` → `RESET` → **`ready` 校验** → **guard 校验** →
  `STATE` → `STEP`。`ready` 校验原样保留，守卫只在其后增加一道 hash 校验。
- 不一致时：关闭当前引擎进程 → 用 `--reset-retries` 允许的次数**新起一个引擎进程**重试 →
  用尽仍不一致则抛 `ResetHashMismatch`（见 §2.4 为什么必须换进程）。
- 记录字段：
  - 成功行：`reset_verified=true`、`reset_hash`、`reset_hash_expected`、`reset_hash_source`、
    `reset_attempts`、`reset_observed_hashes`、`reset_stage_start`；
  - 拒绝记录：`status="error"`、`measured=false`、`reset_guard_failed=true`、两侧 hash、
    `reset_stage_start`、`reset_attempts`、`reset_observed_hashes`，**不含任何结果数字**。
- CLI 的 fail-closed 分支：若返回记录里 `reset_guard_failed` 为真，则**不调用 `append_row`**
  （`results.jsonl` 一行都不写，连 error 行也没有），把该 JSON 打到 stdout、人类可读消息打到
  stderr，返回 `EXIT_RESET_HASH_MISMATCH = 4`。
  - 与既有退出码区分：`1` = 失败但已记录（`status="error"` 行）、`2` = 参数错误、
    `3` = 未请求执行/无参考而拒绝、`4` = 起点未验证且**未写任何行**。

### 2.3 `tools/rl/v400_wsl_env.py`（reset 封装）

- `WslHarnessClient(..., expect_reset_hash=...)` 为**可选**：
  - 每次 `reset()` 都把原始 `hash` 追加到 `self.reset_hashes`（诊断用）；
  - 配置了 `expect_reset_hash` 时，在 `reset()` 内直接比对，不一致抛 `ResetHashMismatch`；
  - 默认 `None`，探测/其它调用方行为不变。
- probe CLI 增加 `--expect-reset-hash`，mismatch 退出码 4。
- **不在 client 上做重试**：bench runner 自己持有 guard 与重试循环，若把期望同时塞给 client，
  client 会在第一次不一致时立刻抛错，runner 的重试就不会发生（且会绕过"重试"语义）。

### 2.4 实测发现：同一进程重复 `RESET` 不可用

负对照第一次实现是"在同一进程里重发 `RESET`"，真实引擎的回答是：

```
OK reset=1 ready=0 setup_frames=1800 frames=0 stage=1 difficulty=1 team=0 seed=1000
   setup_state=0 hash=8643c48839284226
```

即同一进程第二次 `RESET` **不进入可玩状态**（`ready=0`）。因此该实现被改为"重试 = 新起一个
引擎进程"，并在单测里固定了"每个进程只收到一次 `RESET`、分歧进程不 `STEP`"这一形态。
旧实现的那次真实记录留在 `.tmp_v400_resetguard_negative_retry.jsonl`（当时还没改成 fail-closed，
所以它以普通 `status="error"` 行记录）；改完后该路径不再出现。

## 3. 单测

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider `
  --basetemp .tmp_v400_resetguard_t -q tests/test_v400_bench_runner.py tests/test_v400_env.py
# → 49 passed, exit 0
```

新增 15 个用例（`tests/test_v400_bench_runner.py` 12 个 + `tests/test_v400_env.py` 3 个）：

| 用例 | 断言要点 |
|---|---|
| `test_expectation_precedence_is_cli_then_environment_then_first_observed` | cli > env > first-observed；env 大小写/空白归一；非法 env 抛 `ValueError` |
| `test_guard_rejects_a_different_hash_with_both_values_and_stage_start` | 消息含两侧 hash 与 `RESET 1 1 0 1000 0`；`as_record()` 字段正确 |
| `test_divergent_reset_hash_is_fail_closed_and_writes_no_row` | 退出码 4、`results.jsonl` 不存在、无任何结果数字、stdout JSON 与 stderr 均含两侧 hash |
| `test_matching_reset_hash_runs_and_records_the_guard_provenance` | 一致时正常出 `status=ok` 行，`reset_hash_source="cli"`、`reset_attempts=1`，`validate()` 通过 |
| `test_a_mismatch_retried_once_succeeds_and_records_both_hashes` | 重试成功：`attempts=2`、观测序列为 `[坏, 好]`、**两个进程各只 1 次 RESET**、分歧进程未 STEP 且已关闭 |
| `test_retries_exhausted_still_fails_closed_without_a_row` | `--reset-retries 2` 全不一致 → 退出码 4、`attempts=3`、仍不写行 |
| `test_reply_without_a_hash_is_fail_closed` | 无 `hash` → 退出码 4、不写行、`reset_hash_mismatch=false` 但 `reset_guard_failed=true` |
| `test_environment_hash_expectation_is_used_when_no_cli_flag` | 无 CLI 参数时 `source="env"` 且一致即成功 |
| `test_explicit_hash_outranks_the_environment` | env 设为错误值时，显式参数仍然胜出并成功（`source="cli"`） |
| `test_env_hash_mismatch_is_fail_closed_without_a_row` | env 期望与引擎不符 → 退出码 4、不写行、消息含 `source=env` |
| `test_invalid_expected_hash_is_rejected_before_any_launch` | `--expect-reset-hash not-a-hash` → 退 2、不产生结果文件 |
| `test_a_shared_guard_makes_two_episodes_of_one_run_agree` | 共享 guard：首局 pin，第二局换引擎状态 → 第二局 fail-closed |
| `test_wsl_reset_records_the_engine_hash_without_an_expectation` | 封装默认记录 `reset_hashes`，不额外校验 |
| `test_wsl_reset_accepts_the_configured_hash` | 显式期望（大小写不同）通过，`guard.expected` 已归一 |
| `test_wsl_reset_fails_closed_on_a_configured_hash_mismatch` | 封装层抛错并带两侧 hash 与 stage_start，真实回复仍留档 |

> 说明：`touhou_rl/v400/gym/env.py`（`Th08Env.reset`）不在本任务允许修改范围内，本次未接守卫；
> 它属于 §6 的已知限制。

## 4. 真实引擎验证（3 局 + 负对照）

数据目录为**新建**私有目录 `/root/v400-resetguard/game`（`th08.cfg`、`score.dat` 及其
`.orig` 副本从 `/root/v400-data/game` 复制，`th08.dat`/`thbgm.dat` 为只读符号链接），
不使用共享模板。
引擎：`th08-modern` sha256 `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb`，
commit `f563df58c44037b44208c393a86540b434b0c83b`，WSL 内核 `6.18.54.1-microsoft-standard-WSL2`。

```powershell
# 第 1 局：不 pin 期望（source=first-observed）
python tools/rl/v400_bench_run.py --agent random --suite S1N --seed 1000 --execute `
  --max-frames 3000 --data-dir /root/v400-resetguard/game `
  --results .tmp_v400_resetguard_results_final.jsonl --scoreboard .tmp_v400_resetguard_scoreboard_final.md
# 第 2 局：--expect-reset-hash <第 1 局实测值>
# 第 3 局：$env:V400_EXPECT_RESET_HASH=<第 1 局实测值>
```

| 局 | 期望来源 | 退出码 | `reset_hash` | 期望 hash | attempts | frames | end | 结果文件 |
|---|---|---|---|---|---|---|---|---|
| 1 | first-observed | **0** | `3eafc035dc4fe6a9` | `3eafc035dc4fe6a9` | 1 | 2586 | game_over | 追加 1 行 |
| 2 | cli | **0** | `3eafc035dc4fe6a9` | `3eafc035dc4fe6a9` | 1 | 2586 | game_over | 追加 1 行 |
| 3 | env | **0** | `3eafc035dc4fe6a9` | `3eafc035dc4fe6a9` | 1 | 2586 | game_over | 追加 1 行 |

- 三局 `reset_hash` 一致（`{3eafc035dc4fe6a9}`）、三行 `status=ok`、`score` 均为 41193、
  `gameplay_frame` 均为 2588 → 同参数下轨迹一致，守卫**没有误报**。
- `reset_setup_frames` 分别为 330/331/336（引擎内部预热帧数有 ±6 帧抖动）而 state hash 相同：
  守卫比对的是状态哈希，不是预热帧数，因此不受该抖动影响。
- 三个来源各被真实路径覆盖一次（`first-observed` / `cli` / `env`）。

负对照（同参数，故意把期望值写成 `deadbeefdeadbeef`，仅作哨兵，非测量值）：

| 场景 | 命令差异 | 退出码 | `reset_attempts` | 观测序列 | 结果文件 |
|---|---|---|---|---|---|
| 不重试 | `--reset-retries 0` | **4** | 1 | `[3eafc035dc4fe6a9]` | **不存在**（一行未写） |
| 重试 1 次（新进程） | `--reset-retries 1` | **4** | 2 | `[3eafc035dc4fe6a9, 3eafc035dc4fe6a9]` | **不存在** |

- 不重试负对照的 stderr 实测文本（含两侧 hash 与 stage_start）：
  `RESET state-hash guard refused RESET 1 1 0 1000 0: ResetHashMismatch: RESET state-hash guard
  diverged after 1 attempt(s): expected hash=deadbeefdeadbeef (source=cli),
  observed hash=3eafc035dc4fe6a9, stage_start=RESET 1 1 0 1000 0; ...`
- 重试负对照证明"重试=两个新进程、每个各一次 `RESET`"在真实引擎上可行，且最终仍 fail-closed。

## 5. 该守卫能否解释并防止 5.6% 的不确定性？

**（a）能检测并拒绝（本卡已证明的部分）。**
只要期望 hash 来自外部（`--expect-reset-hash` 或 `V400_EXPECT_RESET_HASH`），任何落入不同
StageStart 的进程都会在**第一个 `STEP` 之前**被拒；没有 STEP，就没有可写入的测量值。同批
多实例场景下（共享 guard，或先以第 1 个实例 pin 再把该值作为 cli/env 期望分发）也能互相发现
分歧。三种来源在真实路径上各验证一次，负对照证明拒绝分支确实生效且不写行。

**（b）不能解释（明确不声称）。**
本卡没有测量 5.6% 的成因。EXP-P0-070 run c 的样本只有 18 次启动、1 次分歧，本卡未扩大样本，
也没有做受控复现（无法按需触发）。因此不声称与 LP 线程数、RNG、初始化顺序、并发启动压力或
引擎版本有关，更不声称"已修复根因"。

**（c）不能完全防止（条件限制，必须写清楚）。**
- 若只用来源③（first-observed）：守卫只保证"同一次运行内各实例一致"。如果**第一个**实例就
  漂移，该漂移会被当作期望值 pin 下来，后续实例反而与"错误的标准"对齐。要防这一情形，必须
  在启动前 pin 一个外部已知值（cli/env），或先做两次独立启动交叉验证再 pin。
- 重试不改变根因概率：默认 1 次重试只是把"启动即失败"的代价降低；若分歧是系统性的（例如
  数据/版本变了），期望值本身就会持续不匹配并持续 fail-closed，这需要人工更新期望值。
- 期望哈希与参数绑定：`3eafc035dc4fe6a9` 对应 `RESET 1 1 0 1000 0`（stage 1 / normal /
  seed 1000）。换 seed、stage、难度或 practice 位都必须重新登记期望值；本卡不跨参数复用。

**结论：**5.6% 的"静默污染"被堵住（从"可能进入训练数据"变成"启动失败 + 显式报错 + 零行写入"）；
5.6% 的"发生概率"未被本卡改变，根因仍未知。这是"检测并拒绝"，不是"修复"。

## 6. 剩余限制

1. `touhou_rl/v400/gym/env.py` 的 `Th08Env.reset` **未接守卫**（不在本任务允许修改范围）。任何
   绕过 `tools/rl/v400_bench_run.py` 直接构造 env 的训练路径仍然没有起点校验——这是下一步工作。
2. 未做 18 进程并发场景的按需复现（无法可靠触发 5.6%）；拒绝逻辑只由负对照证明，真实分歧场景
   本身在本卡中**未被重新观测到**（本轮 3 局 + 4 次负对照启动均未出现分歧）。
3. 期望值仓库化尚未做：目前靠命令行/环境变量传入；没有把"每个 (stage, difficulty, seed) 的
   期望 hash"登记成受版本控制的文件（这会引入新的"期望值是否可信"问题，需单独设计）。
4. 3 局真实运行只覆盖 stage 1 / normal / seed 1000 / 3000 帧上限；其它 stage、lunatic、practice
   路径未跑。
5. 守卫不校验 `STATE`/`STEP` 期间的 hash 漂移（EXP-P0-070 的第 1 帧分歧可由 `RESET` hash 覆盖；
   但若某进程在 reset 之后、step 之前内部漂移，本卡不检测）。
6. 未改引擎 C++/协议/依赖/第三方/零售件；未删除任何文件；未伪造 hash（负对照的
   `deadbeefdeadbeef` 在卡中明确标注为哨兵值，观测 hash 全部来自引擎真实回复）。

## 7. 证据文件

- 实现：`touhou_rl/v400/bench/reset_guard.py`（新增）、`tools/rl/v400_bench_run.py`（修改）、
  `tools/rl/v400_wsl_env.py`（修改）
- 单测：`tests/test_v400_bench_runner.py`、`tests/test_v400_env.py`
- 真实 3 局：`.tmp_v400_resetguard_results_final.jsonl`（3 行，全 `status=ok`）、
  `.tmp_v400_resetguard_scoreboard_final.md`
- 负对照（不重试）：`.tmp_v400_resetguard_negative_final.jsonl`（**不存在**，退出码 4）
- 负对照（重试 1 次）：`.tmp_v400_resetguard_negative_retry3.jsonl`（**不存在**，退出码 4，
  `reset_attempts=2`）
- 旧"同进程重发 RESET"实测：`.tmp_v400_resetguard_negative_retry.jsonl`（`ready=0` 记录，
  见 §2.4）
- 更早一轮 3 局（改重试实现前的等价路径，结论相同，保留作复现）：
  `.tmp_v400_resetguard_results.jsonl`
