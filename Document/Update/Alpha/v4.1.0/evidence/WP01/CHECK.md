# WP01 `check --wp WP01` fail-closed 证据检查器

- 时间：2026-11-13
- 范围：只写代码与测试，不启动游戏、不训练、不采集；`evidence/WP01/*.json` 只读，未改动任何证据文件。
- 结论：`python tools/rl/v410.py check --wp WP01 --evidence Document/Update/Alpha/v4.1.0/evidence/WP01` **exit 0**（真实证据 7 个子门全通过）；`check --wp WP00` 行为与输出字段保持不变，仍 exit 0。

## 1. 变更文件（严格白名单内，共 4 个）

| 文件 | 状态 | 作用 |
|---|---|---|
| [`touhou_rl/v410/wp01.py`](../../../../../touhou_rl/v410/wp01.py) | 新建 | `check_wp01(evidence_dir, *, strict=True)`：7 个子门 + 阈值常量 + 失败聚合 |
| [`tools/rl/v410.py`](../../../../../tools/rl/v410.py) | 改 `check` 分支 | `--wp WP01` 调用 `check_wp01`；WP00 分支与其余 WP 的「未实现/3」原样保留 |
| [`tests/test_v410_wp01_check.py`](../../../../../tests/test_v410_wp01_check.py) | 新建 | 41 个用例：合成证据的全通过/缺失/不满足/坏 JSON/WP00 回归/CLI 分派 |
| `CHECK.md` | 新建 | 本文件 |

未触碰：`touhou_rl/v400/`、`v380/`、`v390/`、`tools/rl/v400_*`、`tests/test_v400_*`、`third_party/`、`touhou_rl/v410/{protocol,adapters,boundary,evidence,parity,runtime}.py`、任何状态/证据 JSON。

## 2. 子门与阈值（全部写成模块常量，来源见常量注释）

| 子门 | 文件 | 阈值常量 | 判定要点 |
|---|---|---|---|
| `parity` | `parity.json` | `OFFLINE_VS_ONLINE_MAX_ABS=1e-5`（源自 `parity.json tolerances.cpu_offline_vs_online`）、`CPU_VS_CUDA_MAX_ABS=1e-4`（源自 `tolerances.cpu_vs_cuda`） | schema、`passed=true`、两个 `batch1.max_abs` 未超限、`argmax_all_equal=true`、`cuda_skipped` 不得为 true（本机有 CUDA，skip 不算通过）、子项 `skipped` 不得为 true |
| `consistency` | `consistency.json` | `MIN_CONSISTENCY_SEEDS=3`、`MIN_CONSISTENCY_RESETS=3`、`MIN_CONSISTENCY_FRAMES=1000` | `status=pass`、`config.seeds>=3`、`resets>=3`、`frames>=1000`、每个 seed `identical=true`/`mismatch_count=0`/`errors=stale=0`、`totals.errors=stale=0`、`failures` 为空 |
| `pixels` | `pixels.json` | — | `status=pass`、`pixel_check.enabled=true`、`identical=true`、`samples>0`；`config.role!=env` 记 warning |
| `stability_30min` | `stability-30min.json` | `MIN_STABILITY_WALL_SECONDS=1800.0` | `status=pass`、`mode=stability`、`stability.wall_seconds>=1800`、`errors=stale=0`、`seeds_identical=true`；`budget_cut`（含 `seeds[].resets[].budget_cut`）允许但必须进 `warnings` |
| `boundary` | `boundary.json` | `MIN_BOUNDARY_PASSED=150` | `pytest.passed>=150`、`failed=skipped=0`、`not_claimed` 非空且含 stub/not-capability 语义（`NOT_CLAIMED_MARKERS` × `NOT_CLAIMED_SUBJECTS`）、`policy_stub.capability_claim` 不得为 true |
| `baselines` | `baselines.json` | `MIN_BASELINE_RUNS=12`、`REQUIRED_BASELINE_POLICIES=("random","still","fixed_east","fixed_west")` | `status=pass`、`results>=12`、4 个 policy 全覆盖、每局有非空 `end` 与正 `frames`；`privilege` 必须显式声明且**不是严格像素**→ 只记 warning，不失败 |
| `seed_sensitivity` | `seed_sensitivity.json` | `MAX_SEED_DIVERGENCE_FRAME=1200`、`SEED_SENSITIVITY_STATUS="verified_with_warning"`、`SEED_POLICY_RULE_NAME="seed_sensitivity_rule"` | `status` 恰为 `verified_with_warning`、`probes.first_life.first_non_seed_divergence is null`、`standard_after_first_death` 的分歧 frame 全部 `<=1200`、`policy_rule.name` 匹配 |

任一子门缺失（记入 `missing`）或不满足（记入 `invalid`）→ `status="failed"`、`exit_code=3`；`skip` 一律不算通过。

### `strict` 语义

- `strict=True`（默认）：所有硬子门必检，并额外对辅助证据 `consistency-independent.json`、`stability-2min.json` 产出提示性 warnings（缺失/非 JSON/status 非 pass）。
- `strict=False`：**只**关闭上述辅助提示；硬子门判定完全不变，缺失或不满足仍 exit 3（由 `test_strict_false_does_not_relax_hard_gates` 证明）。

## 3. 验收命令与实际退出码

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp01check_pytest -q tests/test_v410_wp01_check.py tests/test_v410_evidence.py
python -m compileall -q touhou_rl/v410 tools/rl/v410.py
python tools/rl/v410.py check --wp WP01 --evidence Document/Update/Alpha/v4.1.0/evidence/WP01
python tools/rl/v410.py check --wp WP00 --evidence Document/Update/Alpha/v4.1.0/evidence/WP00
```

| # | 命令 | 实际输出 | 退出码 |
|---|---|---|---|
| 1 | pytest（WP01 新测试 + WP00 既有测试） | `52 passed in 0.30s` | **0** |
| 2 | `compileall -q touhou_rl/v410 tools/rl/v410.py` | 无输出 | **0** |
| 3 | `check --wp WP01`（真实证据） | `status=passed`，`missing=[]`，`invalid=[]`，warnings=3 | **0** |
| 4 | `check --wp WP00`（真实证据） | `status=passed`，`missing=[]`，`invalid=[]`，`resummed=verified`，warnings=6（与改动前一致） | **0** |

交叉验证：`check --wp WP02` 仍为 `{"schema":"v410-check/1","status":"failed","exit_code":3,"message":"... not implemented ..."}`，退出码 **3**（未实现分支未改动）。

pytest 统计：`tests/test_v410_wp01_check.py` 41 个用例 + `tests/test_v410_evidence.py` 11 个用例 = **52 passed / 0 failed / 0 skipped**，退出码 0。

额外回归（非验收要求，用于确认未破坏既有 v4.1.0 测试）：`tests/test_v410_{wp01_check,evidence,protocol,boundary,baselines,consistency}.py` 共 **319 passed**（exit 0），`tests/test_v410_parity.py` 另 **37 passed**（exit 0）。

## 4. 每个子门的实际值（真实证据，2026-11-13 运行时）

| 子门 | 状态 | 实测值 |
|---|---|---|
| `parity` | passed | `passed=true`；`offline_vs_online_batch1.max_abs=0.0`（<=1e-5），`argmax_all_equal=true`；`cpu_vs_cuda_batch1.max_abs=2.384e-07`（<=1e-4），`argmax_all_equal=true`；`cuda_skipped=false` |
| `consistency` | passed | `status=pass`；`config.seeds=[1,2,3]`（3 个），`resets=3`，`frames=1000`；`seeds.count=3` 且全部 `identical=true`/`mismatch_count=0`；`totals.errors=0`，`totals.stale=0`；`config.role=shadow` |
| `pixels` | passed | `status=pass`；`config.role=env`；`pixel_check.enabled=true`，`identical=true`，`samples=16`（>0） |
| `stability_30min` | passed | `status=pass`，`mode=stability`；`stability.wall_seconds=1800.003`（>=1800）；`errors=0`，`stale=0`，`seeds_identical=true`；`totals.resets=1039`，`totals.frames=1038399`；`budget_cut` 命中 1 个 reset |
| `boundary` | passed | `pytest.passed=151`（>=150），`failed=0`，`skipped=0`，`error=0`；`not_claimed.count=5`（含 "does not claim policy capability ..." 与 "synthetic unit tests"）；`policy_stub.capability_claim=false`；`activity.game_launched=false` |
| `baselines` | passed | `status=pass`；`results.count=12`；policies=`["fixed_east","fixed_west","random","still"]` 覆盖 4 个；`summary.overall.runs=12`；每局 `end=first_death` 且 `frames>0`；`privilege=engine_state_readonly` |
| `seed_sensitivity` | passed | `status=verified_with_warning`；`probes.first_life.first_non_seed_divergence=null`；`standard_after_first_death` 分歧 frame `1_vs_2=906`、`1_vs_3=952`（均 <=1200）；`policy_rule.name=seed_sensitivity_rule` |

## 5. Warnings（真实证据，共 3 条；均为软发现，不影响 exit 0）

1. `stability-30min.json: budget_cut recorded on 1 reset(s) ([{'seed': 2, 'reset': 345, 'iteration': 345, 'requested_frames': 1000, 'frames': 399, 'mismatch': 0}]); the truncated prefixes stay frame-identical and the stability.wall_seconds gate is still met`
   —— 与 [`STATUS.md`](STATUS.md) 记录的「仅 1 次墙钟截断（seed2 末次，399/1000 帧）」一致；截断被登记而非被忽略，且 `stability.wall_seconds=1800.003` 仍达标。
2. `baselines.json: privilege='engine_state_readonly' is engine-state read-only, not a strict-pixel measurement; these baseline numbers are not a pixel-only capability claim`
   —— 基线是引擎态基线，不是严格像素能力；按任务要求记 warning，不失败。
3. `seed_sensitivity.json: status is 'verified_with_warning'; the seed sensitivity warning stays attached to the WP01 evidence`
   —— 种子敏感性结论本身带警告（首死后才出现非 rng 分歧），警告随证据保留。

无 `missing`、无 `invalid`。

## 6. fail-closed 证明（41 个用例覆盖矩阵）

| 场景 | 用例 | 期望/实际 |
|---|---|---|
| 7 子门齐全且满足 | `test_all_gates_pass_exit_zero` | passed / exit 0 |
| 真实证据目录 | `test_real_evidence_passes` | passed / exit 0 |
| 缺 `parity.json` | `test_missing_parity_exits_three` | failed / exit 3，`missing` 含 `parity.json` |
| 证据目录整体不存在 | `test_empty_evidence_dir_reports_every_missing_file` | exit 3，7 个文件全部进 `missing` |
| 坏 JSON（截断） | `test_bad_json_exits_three` | exit 3，`invalid` 含 "not parseable JSON" |
| schema 不符 | `test_wrong_schema_exits_three` | exit 3 |
| `cuda_skipped=true` | `test_cuda_skipped_is_invalid` | exit 3（本机有 CUDA，skip 非通过） |
| 容差超限 / argmax 不等 | `test_parity_tolerance_exceeded`、`test_parity_argmax_mismatch` | exit 3 |
| consistency mismatch | `test_consistency_mismatch_exits_three` | exit 3 |
| seeds 少于 3 / stale 非 0 | `test_consistency_too_few_seeds`、`test_consistency_errors_or_stale` | exit 3 |
| pixels `identical=false` / 未启用 | `test_pixels_not_identical_exits_three`、`test_pixels_disabled_exits_three` | exit 3 |
| stability `wall<1800` / 非 stability / 非 seed 一致 | 3 个用例 | exit 3 |
| `budget_cut`（报告级与 reset 级） | `test_stability_budget_cut_is_a_warning`、`test_stability_per_reset_budget_cut_is_a_warning` | passed + warning |
| boundary `passed=149` / `skipped=2` / `not_claimed` 为空或不含语义 | 4 个用例 | exit 3 |
| baselines 少 policy / 少于 12 局 / 缺 `end` 或 `frames` | 3 个用例 | exit 3 |
| baselines `privilege` 非严格像素 | `test_baselines_privilege_warns_but_passes` | passed + warning |
| seed_sensitivity 缺失 / 首命分歧非空 / 分歧 frame 过晚 / 无分歧记录 / status 非 warning / rule 名不符 | 6 个用例 | exit 3 |
| `strict=False` 不放宽硬门 | `test_strict_false_does_not_relax_hard_gates` | exit 3 |
| WP00 回归（函数级） | `test_wp00_check_still_passes`、`test_wp00_check_still_fails_closed` | exit 0 / exit 3 |
| WP00 回归（CLI 字段） | `test_cli_wp00_output_fields_unchanged` | 字段集合仍为 `{schema, wp, status, exit_code, missing, invalid, warnings, resummed, message}`，exit 0 |
| WP01 CLI 分派 | `test_cli_wp01_dispatches_to_checker`、`test_cli_wp01_failure_exits_three` | exit 0 / exit 3 |
| 其他 WP 仍「未实现/3」 | `test_cli_other_wp_still_not_implemented` | exit 3 |

## 7. 限制（不构成能力声明）

1. 检查器**只做证据一致性校验**，不重跑 parity/consistency/基线，也不校验证据内记录的 sha256、checkpoint 或引擎二进制；证据本身的真实性由各 `*.md` 与原始产物负责。
2. `baselines.json` 是 `privilege=engine_state_readonly` 的引擎态基线：12 局全部 `first_death`，**没有**真实 `game_over` / `stage_clear` / `cap` 样本；终止类型仍只有合成单测覆盖（该缺口在 [`STATUS.md`](STATUS.md) 中已登记，本检查器不放宽也不掩盖）。
3. `seed_sensitivity.json` 的 `verified_with_warning` 意味着：首命 700 帧 seed-invariant，但首死后约 frame 906/952 起出现非 rng 分歧，早期死亡局不能当独立 seed 样本。检查器只校验该警告存在，不评估有效样本量。
4. 辅助证据 `consistency-independent.json`、`stability-2min.json` 只作提示，不作硬门（任务允许）。
5. 检查器不启动游戏、不训练、不采集；本次执行期间同样未启动游戏。
6. 工作树在本任务之前即不干净（WP00 `assets.json` 已登记 tracked modified + untracked），本任务只写入上表 4 个白名单路径；pytest 使用 `--basetemp .tmp_v410_wp01check_pytest`，该目录为测试临时产物。
