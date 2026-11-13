# WP02.2 — fast 教师在 20 个 Stage-1 开发种子上的真实评测（证据）

- 工作包：v4.1.0 WP02.2
- 目的：判断 `fast` 教师能否在 Stage-1 开发种子（1000–1019）上达到 **≥ 18/20 过面**，以决定后续教师采集用 `fast` 还是完整/`v2h`
- 工具：`tools/rl/v410_teacher_fast20.py`（只驱动 `tools/rl/v400_teacher.py`，不改教师策略）
- 测试：`tests/test_v410_teacher_fast20.py`（35 例，fake 运行器，不启动引擎）
- 机器可读证据：`Document/Update/Alpha/v4.1.0/evidence/WP02/fast20.jsonl`（20 行）、`fast20_summary.json`（schema `v410-fast20/1`）
- 运行目录（不入库）：`/tmp/v410_fast20/`（每个 seed 的教师原始 JSONL 与日志）
- 环境：WSL2 Ubuntu / Python 3.10.12 / numpy 2.2.6 / 20 vCPU；教师引擎 `/root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern`，SHA256 `1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f`（与 G1 v2h 同一构建）
- 活动声明：**未训练、未采数据集、未修改引擎/教师实现、未 commit/push**；只读引用 `.tmp_v400_laser/s1n/round1-v2h/runs.jsonl`

## 0 结论（一句话）

**fast 在 20/20 个 Stage-1 开发种子上过面（0 死亡、全部 `stage_changed`），`fast_gate` 通过（阈值 18/20），执行与验证一致——后续教师采集可用 `fast`。**
该结论只覆盖 Stage 1 / normal / 开发种子 1000–1019 / fast profile；**不是 G410-1 的正式 100 种子确认**（见 §7）。

## 1 实际执行的验收命令与退出码

```powershell
# 1) 单元测试（Windows，无引擎、无 WSL）
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_fast20_pytest -q tests/test_v410_teacher_fast20.py
# 输出：35 passed in 0.19s
# 退出码：0

# 2) 语法编译
python -m compileall -q tools/rl/v410_teacher_fast20.py
# 输出：（无）
# 退出码：0
```

```bash
# 3) WSL smoke：单 seed（1000），先证工具链
cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v410_teacher_fast20.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern \
  --expected-sha 1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f \
  --seeds 1000 \
  --out-jsonl /tmp/v410_fast20/smoke.jsonl \
  --out-summary /tmp/v410_fast20/smoke_summary.json --quiet --force
# [fast20] SMOKE: clears=1/1 frames_mean=13004.0 frames_median=13004.0 missing=[] failed=[] fast_gate=N/A (slice < threshold)
# [fast20] smoke 切片（1 个种子 < 阈值 18），fast_gate 不构成 WP02.2 的 18/20 判定
# SMOKE_EXIT=0

# 4) WSL 主跑：20 seeds
cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v410_teacher_fast20.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern \
  --expected-sha 1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f \
  --seeds 1000-1019 \
  --out-jsonl Document/Update/Alpha/v4.1.0/evidence/WP02/fast20.jsonl \
  --out-summary Document/Update/Alpha/v4.1.0/evidence/WP02/fast20_summary.json --quiet
# [fast20] PASSED: clears=20/20 frames_mean=13282.4 frames_median=13292.0 missing=[] failed=[] fast_gate=PASS
# [fast20] fast 达到 18/20 过面门（20/20），执行与验证一致：后续教师采集可用 fast
# MAIN_EXIT=0
```

补充说明（不隐瞒）：

- smoke 一共完整跑了三次，`frames` 都是 **13004**（同 seed 的确定性旁证），但退出码只在后两次被显式回显：
  1. 首轮（`--quiet`，无 `--force`，01:04:50→01:07:38，wall 155.9 s）：输出同样的 1/1 过面与 13004 帧；当时回显用的 `$?` 被 PowerShell 提前展开成 `True`，没有取到可观察的退出码（是 shell 转义写法错误，不是工具失败）。
  2. 第二次（`--force`）：捕获 `SMOKE_EXIT=0`；此时进度行仍显示旧的 `fast_gate=FAIL` 标签（smoke 切片不适用阈值，随后已改成 `N/A`）。
  3. 第三次（改标签后 `--force`）：wall 173.5 s，输出 `fast_gate=N/A (slice < threshold)`，`SMOKE_EXIT=0`。
- 主跑结束后，以**同一 argv（不加 `--force`）**再调用一次：20 个 seed 全部按"同 mode + 同引擎 SHA + exit 0"跳过，只重算 summary 与 JSONL，`MAIN_RESUME_EXIT=0`。这次幂等重算也验证了断点续跑逻辑。
- pytest 统计：**collected 35 / passed 35 / failed 0 / skipped 0 / error 0**，退出码 0。这 35 是单元测试用例数，**不是游戏运行局数**。
  开发过程中先有 34 passed；为固定"smoke 切片 < 阈值时不构成 gate、且不能被当成 pass"的语义，补了 1 例 `test_gate_on_a_one_seed_slice_is_not_applicable_and_never_a_pass`，最终 35 passed。

## 2 smoke：seed 1000（先证退出码 0 与 JSONL 可解析）

| 字段 | 值 |
|---|---|
| seed | 1000 |
| stage / difficulty | 0 / normal |
| stage_clear | **true** |
| frames | 13004 |
| deaths | 0 |
| stop_reason | `stage_changed` |
| exit_code | 0 |
| wall_s | 173.461（末轮；首轮 155.869） |
| report_mode | fast |
| engine sha256 / match | `1462…0d2f` / true |
| plans_verified | 3251（= decisions 3251 × verify_k 1） |
| execution_ok / verification_ok | true / true |
| smoke summary 的 status / pass / gate | `smoke` / `false` / `applicable=false`（1 < 18，**明确不是 18/20 门**） |

教师原始 JSONL：`/tmp/v410_fast20/1000.jsonl`（3252 行 = 1 行 `kind=report` + 3251 行 decision，可解析）。

## 3 主跑：20 局逐局结果

命令见 §1 第 4 条；实际时间 2026-11-13T01:07:47+0800 → 01:39:01+0800（墙钟 31 分 14 秒，6 路并发）。每局 `--decisions 10000 --max-frames 19887 --stop-on-death --stage 0 --difficulty normal --mode fast`，6 个 worker 各固定一个 `inst<k>` 数据目录。

| # | seed | worker | data_dir | stage_clear | frames | deaths | stop_reason | wall_s | exit |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1000 | 0 | inst0 | **true** | 13004 | 0 | `stage_changed` | 452.6 | 0 |
| 2 | 1001 | 1 | inst1 | **true** | 13100 | 0 | `stage_changed` | 457.4 | 0 |
| 3 | 1002 | 2 | inst2 | **true** | 13292 | 0 | `stage_changed` | 459.0 | 0 |
| 4 | 1003 | 3 | inst3 | **true** | 13292 | 0 | `stage_changed` | 458.1 | 0 |
| 5 | 1004 | 4 | inst4 | **true** | 13292 | 0 | `stage_changed` | 463.0 | 0 |
| 6 | 1005 | 5 | inst5 | **true** | 13196 | 0 | `stage_changed` | 457.9 | 0 |
| 7 | 1006 | 0 | inst0 | **true** | 13292 | 0 | `stage_changed` | 489.2 | 0 |
| 8 | 1007 | 1 | inst1 | **true** | 13388 | 0 | `stage_changed` | 494.9 | 0 |
| 9 | 1008 | 5 | inst5 | **true** | 13292 | 0 | `stage_changed` | 491.0 | 0 |
| 10 | 1009 | 3 | inst3 | **true** | 13484 | 0 | `stage_changed` | 497.0 | 0 |
| 11 | 1010 | 2 | inst2 | **true** | 13388 | 0 | `stage_changed` | 493.8 | 0 |
| 12 | 1011 | 4 | inst4 | **true** | 13196 | 0 | `stage_changed` | 488.6 | 0 |
| 13 | 1012 | 0 | inst0 | **true** | 13100 | 0 | `stage_changed` | 519.9 | 0 |
| 14 | 1013 | 5 | inst5 | **true** | 13196 | 0 | `stage_changed` | 536.0 | 0 |
| 15 | 1014 | 4 | inst4 | **true** | 13676 | 0 | `stage_changed` | 551.7 | 0 |
| 16 | 1015 | 1 | inst1 | **true** | 13100 | 0 | `stage_changed` | 529.5 | 0 |
| 17 | 1016 | 2 | inst2 | **true** | 13100 | 0 | `stage_changed` | 533.4 | 0 |
| 18 | 1017 | 3 | inst3 | **true** | 13484 | 0 | `stage_changed` | 547.9 | 0 |
| 19 | 1018 | 0 | inst0 | **true** | 13484 | 0 | `stage_changed` | 281.8 | 0 |
| 20 | 1019 | 1 | inst1 | **true** | 13292 | 0 | `stage_changed` | 264.3 | 0 |

口径说明：

- `stage_clear` 取教师报告自己的 `stage_clear`（引擎 stage index 前进），不是"活着到帧上限"。20 局的 `stop_reason` 全部是 `stage_changed`，没有一局是 `max_frames` 或 `death`。
- `frames` 是报告里的 `frames_committed`（权威播放长度），与 `survival_frames` 一致。
- `exit` 是 `tools/rl/v400_teacher.py` 的退出码，20/20 为 0。
- 最后两局（1018、1019）wall 明显更短（264–282 s）是因为它们收尾时只剩 2 个 worker 在跑，机器负载低；wall 的差异不是策略差异。

## 4 统计与 fast_gate

| 指标 | 值 |
|---|---|
| clear count | **20 / 20** |
| clear rate（声明分母） | **100.0%** |
| frames 均值 | **13282.4** |
| frames 中位数 | **13292.0** |
| frames 最小 / 最大 | 13004 / 13676 |
| deaths 合计 | **0**（20 局全部 0 死亡） |
| bombs_spent | 0.0（20 局） |
| stop_reason 分布 | `stage_changed` × 20 |
| exit_code 分布 | `0` × 20 |
| decisions 合计 / plans_verified 合计 | 66412 / 66412（verify_k=1，逐局等式成立） |
| 几何 vs 引擎一致性（合计） | `geo_dead_engine_alive` 0、`geo_alive_engine_dead` 0、`agree_alive` 66412、`agree_dead` 0、`plans_verified` 66412 |
| 单局 wall | 264.3 – 551.7 s（6 路并发）；合计 9466.8 s，均值 473.3 s |
| 并发实时倍率 | 0.41 – 0.85×（6 路并发下的吞吐，不代表单跑速度；单跑 smoke 为 13004 / 173.5 / 60 ≈ 1.25×） |

`fast_gate` 的定义与结果（阈值未改：仍是 18/20）：

```
rule            : clear_count >= 18 of 20 AND no execution/verification inconsistency
min_clears      : 18
required_clears : 18
applicable      : true
clears          : 20 / 20
execution_ok    : true   （exit 0 × 20、status=ok × 20）
verification_ok : true   （report mode=fast × 20、引擎 SHA match × 20、
                           plans_verified = decisions × verify_k = 66412 = 66412）
passed          : true
recommendation  : fast
note            : fast 达到 18/20 过面门（20/20），执行与验证一致：后续教师采集可用 fast
```

`trajectory_mismatch` 合计 53216 / 66412 个已验证计划（80.1%）。这不是死亡判定分歧：它是"几何预测轨迹与引擎回放轨迹在 32 帧视野内不完全逐帧一致"的既有量（`verify_k` 正是为这种分歧而做引擎验证；G1 v2h 20 局合计 193545/202980 ≈ 95.4%）。本批次真正与**死亡**相关的两个计数器 `geo_dead_engine_alive` / `geo_alive_engine_dead` 均为 **0**（v2h 20 局合计为 0 / 42，属同一类边界计数），且 0 死亡也意味着没有"双方都判死"的正样本可比。这里如实记录，不把 `trajectory_mismatch` 当作 fast 的通过依据，也不把它藏起来。

## 5 与 G1 v2h 原始 20 局对照

对照来源（**只读**，未修改）：`.tmp_v400_laser/s1n/round1-v2h/runs.jsonl`，SHA256 `abea27812dd9fd57e8e35263a16863119ccb21bc1f8b64df0c7cc965b11993fd`，profile `v2h`，20 行，20/20 过面，`stop_reason` 全为 `stage_changed`。

| 指标 | fast（本次） | v2h（G1 round 1） | 差 |
|---|---|---|---|
| clears | 20 / 20 | 20 / 20 | 0 |
| frames 均值 | 13282.4 | 13532.0 | **−249.6（−1.84%）** |
| frames 中位数 | 13292.0 | 13484.0 | −192 |
| frames 最小 / 最大 | 13004 / 13676 | 13196 / 13964 | — |
| deaths | 0 | 0 | 0 |

逐 seed 对照（Δ = fast − v2h；负值表示 fast 该局更早触发 stage clear）：

| seed | fast frames | v2h frames | Δ(fast−v2h) | fast clear | v2h clear |
|---|---|---|---|---|---|
| 1000 | 13004 | 13484 | −480 | true | true |
| 1001 | 13100 | 13676 | −576 | true | true |
| 1002 | 13292 | 13292 | 0 | true | true |
| 1003 | 13292 | 13292 | 0 | true | true |
| 1004 | 13292 | 13292 | 0 | true | true |
| 1005 | 13196 | 13388 | −192 | true | true |
| 1006 | 13292 | 13484 | −192 | true | true |
| 1007 | 13388 | 13196 | +192 | true | true |
| 1008 | 13292 | 13484 | −192 | true | true |
| 1009 | 13484 | 13868 | −384 | true | true |
| 1010 | 13388 | 13676 | −288 | true | true |
| 1011 | 13196 | 13580 | −384 | true | true |
| 1012 | 13100 | 13580 | −480 | true | true |
| 1013 | 13196 | 13964 | −768 | true | true |
| 1014 | 13676 | 13676 | 0 | true | true |
| 1015 | 13100 | 13388 | −288 | true | true |
| 1016 | 13100 | 13580 | −480 | true | true |
| 1017 | 13484 | 13484 | 0 | true | true |
| 1018 | 13484 | 13868 | −384 | true | true |
| 1019 | 13292 | 13388 | −96 | true | true |

汇总：14 局 fast 更早、1 局（1007）更晚、5 局相同；合计 −4992 帧。

解读与边界：

- 所有差值都是 **96 帧的整数倍**（96/192/288/384/480/576/768），说明差异落在"stage clear 的脚本边界位置"上，而不是"谁把 Boss 打得更快"。这个对照是**描述性**的：它说明 `fast` 在这 20 局上没有比 `v2h` 更差的过面表现，**不构成"fast 比 v2h 更强"的因果结论**。
- 两次运行使用同一引擎 SHA、同一 stage/difficulty、同一 19887 帧上限、同一批 seed。v2h 记录里的 `--decisions` 是 4979（= 19887/4 + 8，即覆盖帧上限所需），本次按 WP02.2 指定为 10000；两者都大于"帧上限 / 4 帧每决策"，有效决策上限相同。
- v2h 的 wall 均值约 808.8 s/局（496.2–890.1 s，当时同样是 6 路并发），仅作参考；本轮 fast 的 wall 受当时机器负载影响（见 §4），不能直接当作速度比。

## 6 `fast20_summary.json` 的机器可读结构

顶层键与含义（schema `v410-fast20/1`）：

| 键 | 内容 |
|---|---|
| `schema` | 固定 `v410-fast20/1` |
| `generated_at` | 生成时间（末次幂等重算 2026-11-13T01:46:03+0800；批次数据时间 01:07:47–01:39:01） |
| `status` / `pass` | `passed` / `true`（不完整批次为 `partial`，slice 为 `smoke`，未达门为 `failed`） |
| `mode` | `fast` |
| `engine` | 传入的 `--engine`（原样保留）、解析出的 `dir`/`binary`、`expected_sha256`、20 局实际报告到的 `sha256_seen`、`sha_match_all`、`preflight`（本地哈希结果） |
| `params` | stage 0、normal、max_frames 19887、decisions 10000、teacher/python/tmp_dir/timeout |
| `seeds` | `[1000 … 1019]` |
| `runs` | 20 条逐局摘要：seed、frames、deaths、stage_clear、stop_reason、exit_code、wall_s、worker、data_dir、engine_sha256/match、mode、decisions、verify_k、plans_verified、consistency、execution_ok、verification_ok |
| `summary` | 覆盖/clears/clear rate、frames 均值/中位数/min/max、deaths、wall、stop_reason 与 exit_code 直方图、`per_seed` 表 |
| `comparison_v2h` | v2h 来源路径与 SHA256、20/20、均值/中位数、逐 seed `paired` 与 `deltas`（14 快 / 1 慢 / 5 平） |
| `fast_gate` | 规则、阈值、是否适用、clears、执行/验证一致性清单、`passed`、`note`、`recommendation` |

`fast20.jsonl`：20 行，每行一局的原始摘要（与上表 `runs` 同构，含实际下发的教师 `command`），按 seed 升序，一局一行。

## 7 限制（本证据**不能**支撑的结论）

1. **单 stage**：只测 Stage 1（引擎 stage index 0）/ normal。其它 stage、其它难度、完整一周目都未测。
2. **开发种子**：1000–1019 是 Stage-1 开发种子，不是留出集；本轮也没有训练，因此不存在"对训练集过拟合"的问题，但同样不能当成未见种子的泛化证据。
3. **fast profile**：本轮只跑 `fast`（horizon 32 / beam 24 / verify_k 1 / max_iterations 1）。`v2h` 的配置是 48 / 64 / 3 / 3，两者不是同一搜索预算；因此"fast 也 20/20"只说明**这套 Stage-1 开发种子对 fast 足够**，不能推导"fast 在所有场景等价或优于 v2h"。
4. **不是 G410-1 的正式确认**：G410-1 要求的是正式 100 种子口径的确认；本轮 20 局只是 WP02.2 的教师选型门，**不能替代 G410-1，也不能替代 G1 的正式评测**。
5. **0 死亡**：20 局没有任何死亡样本，本证据无法比较 fast 与 v2h 的死亡分布/死亡原因。
6. **wall/吞吐不是单跑速度**：主跑是 6 路并发（`realtime_multiple` 0.41–0.85），单跑 smoke 为 ≈1.25×；不同负载下的 wall 不可直接横比。
7. **引擎单点**：结论绑定在 SHA256 `1462…0d2f` 的 `rl_laser/th08-modern` 构建上；换构建必须重跑。
8. **一致性口径**：`fast_gate` 的"验证一致"= 报告 `mode` 与请求一致 + 引擎 SHA match + `plans_verified == decisions × verify_k`；几何/引擎死亡分歧计数器为 0。`trajectory_mismatch`（80.1%）是既有度量，未用作通过条件，也不应被解读为"验证失败"。

## 8 复现与产物清单

- 工具：`tools/rl/v410_teacher_fast20.py`
  - 默认参数即 WP02.2 口径：`--seeds 1000-1019 --mode fast --stage 0 --difficulty normal --max-frames 19887 --decisions 10000 --workers 6`
  - `--engine` 既接受构建目录，也接受 `.../rl_laser/th08-modern` 这个二进制路径（此时同时下发 `--binary <同一路径>`，让教师哈希的正是任务点名的文件；`--engine` 参数原样保留在记录的命令里）
  - 6 个 worker 与 `inst0..inst5` 一一绑定（同一目录不会被两个进程并发使用）；数据根下实际存在 8 个 `inst`，本轮只用 0–5
  - 断点续跑：`--out-jsonl` 中"同 mode + 同引擎 SHA + exit 0"的 seed 会跳过；`--force` 强制重跑
  - 退出码：`0` 批次完整完成（门通过或 slice 小于阈值）；`1` 20 局跑完但未达 18/20 门；`2` 有 missing/failed 或参数/引擎预检被拒
- 测试：`tests/test_v410_teacher_fast20.py`（seed 解析、worker/inst 分配、报告解析、clear 统计、18/17 阈值判定、gate 语义、JSON 结构、fake 端到端）
- 运行目录：`/tmp/v410_fast20/<seed>.jsonl`（教师原始 JSONL）与 `<seed>.log`；`smoke*.json[l]` 为 smoke 产物；`main.log` 为主跑 stdout
- 未改动：`touhou_rl/v400/`、`v380/`、`v390/`、`tools/rl/v400_*`、`tests/test_v400_*`、`third_party/`、`touhou_rl/v410/`、`tools/rl/v410*.py`（本文件除外）、`state.json` / `journal.jsonl` / `budget.jsonl` / `assets.json`、以及 `.tmp_v400_laser/s1n/round1-v2h/runs.jsonl`（只读取并记录 SHA256）
