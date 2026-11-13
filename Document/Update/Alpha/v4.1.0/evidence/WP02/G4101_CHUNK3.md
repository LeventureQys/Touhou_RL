# G410-1 第三批确认种子（61350–61374）真实评测 — chunk 3（证据）

- 工作包：v4.1.0 **G410-1**（教师正式确认种子 61300–61399）的**第三批 25 个种子**
- 本文件范围：**只出数据**——不改教师、不训练、不做 G410-1 总结论
- 工具：`tools/rl/v410_teacher_fast20.py`（只读；只驱动 `tools/rl/v400_teacher.py`，不改教师策略）
- 教师 profile / 引擎：`fast` 教师 + `rl_laser` 引擎
- 机器可读证据：`Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk3.jsonl`（25 行）、`g4101_s1n_chunk3_summary.json`（schema `v410-fast20/1` + 附加注释块 `g4101_chunk3`）
- 运行目录（不入库）：`/tmp/v410_g4101_chunk3/`（25 份教师原始 JSONL + 25 份教师日志 + `main.log` + `started_at.txt` + `finished_at.txt` + `exit_code.txt` + 注释脚本）
- 环境：WSL2 Ubuntu / Python 3.10.12 / 20 vCPU；教师引擎 `/root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern`，SHA256 `1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f`
- 活动声明：**未训练、未采数据集、未改引擎/教师实现、未写 state/journal/budget/assets、未 commit/push**；**未读取 confirm 种子 61000–61099**

## 0 一句话

**25/25 局过面（`stage_clear=true`）、0 死亡、25/25 `exit 0`、引擎 SHA 全部匹配、`missing=[]`、`failed=[]`、0 次重试**——这是 G410-1 确认种子第三批（61350–61374）的**真实评测数据**。
**它不是 G410-1 的通过结论**：本批只有 25 个种子（< 100），G410-1 的口径是 61300–61399 共 100 个确认种子，本文件只覆盖其中第 3/4 段；即使把 chunk1、chunk2 与本批拼起来也只有 **75/100**，还差 61375–61399。

## 1 实际执行的命令与退出码

```bash
cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v410_teacher_fast20.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern \
  --expected-sha 1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f \
  --seeds 61350-61374 \
  --out-jsonl Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk3.jsonl \
  --out-summary Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk3_summary.json \
  --tmp-dir /tmp/v410_g4101_chunk3 --quiet --force
# [fast20] PASSED: clears=25/25 frames_mean=13361.1 frames_median=13292.0 missing=[] failed=[] fast_gate=PASS
# [fast20] fast 达到 18/20 过面门（25/25），执行与验证一致：后续教师采集可用 fast
# 批次退出码：EXIT_CODE=0        （/tmp/v410_g4101_chunk3/exit_code.txt）
```

- **退出码 0**（`wsl.exe` 透传的工具进程退出码，由 `bash -lc` 末尾 `echo $ec > exit_code.txt` 落盘到 `/tmp/v410_g4101_chunk3/exit_code.txt`）。工具自己在判定后 `return EXIT_PASS`，含义是"批次完整跑完、无 missing/failed"，**不是** G410-1 判定。
- 与任务给定命令**逐字一致**：`--engine` / `--expected-sha` / `--seeds 61350-61374` / 两个 `--out-*` / `--tmp-dir /tmp/v410_g4101_chunk3` / `--quiet` / `--force`，未改任何阈值或参数（`--min-clears` 未传，取工具默认 18；`--max-frames` 未传，取默认 19887；`--workers` 未传，取默认 6）。
- 启动 `2026-11-13T13:03:13+08:00`（`started_at.txt`），末局结束 `2026-11-13T13:41:14+08:00`（`finished_at.txt`）：**批次墙钟 2,281 s ≈ 38 min 1 s**，6 路并发。
- 教师单局参数（`runs[].command` 原样记在 JSONL 每行里）：`--decisions 10000 --max-frames 19887 --stop-on-death --stage 0 --difficulty normal --seed <seed> --mode fast --quiet --data-dir /root/v400-data/g1/inst<k> --engine <同上> --binary <同上> --expected-sha 1462…0d2f`。
- stdout/verdict 落在 `/tmp/v410_g4101_chunk3/main.log`；25 份教师报告 `61350.jsonl … 61374.jsonl`（行数 = 1 + `plans_verified`，逐局核对成立）、25 份教师日志 `61350.log … 61374.log` 完整保留（运行目录共 294 MB）。
- **重试：0 次**。本批无失败、无超时、无 missing；`runs[]` 中每个 seed 恰好一行，无重复 seed，未触发"同一 seed 最多 2 次重试"的路径。
- 数据目录用工具默认的 `/root/v400-data/g1/inst0..5`（与 chunk2 相同，按 worker 一一绑定，同一目录不会被两个进程并发使用）；**本批 25 局的 `runs[].command` 中没有任何一局把 `/root/v400-data/game` 当作 `--data-dir`**，该目录不在本批写入范围。说明：本批开始前该目录 `score.dat` / `th08.cfg` 的 mtime 是 `Nov 13 09:02`；本批进程已于 `13:41:14` 全部退出，随后 `13:43` 该目录被**另一个并发任务**改写（`tools/rl/v410_student_eval.py`，`rl_v3` 引擎，`--data-dir /root/v400-data/game`，约 13:42 启动），与本批无关。

## 2 逐局结果（25 行）

| # | seed | worker | data_dir | stage_clear | frames | deaths | stop_reason | wall_s | exit |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 61350 | 0 | inst0 | **true** | 13484 | 0 | `stage_changed` | 523.440 | 0 |
| 2 | 61351 | 1 | inst1 | **true** | 13196 | 0 | `stage_changed` | 510.466 | 0 |
| 3 | 61352 | 2 | inst2 | **true** | 13196 | 0 | `stage_changed` | 513.608 | 0 |
| 4 | 61353 | 3 | inst3 | **true** | 13388 | 0 | `stage_changed` | 521.963 | 0 |
| 5 | 61354 | 4 | inst4 | **true** | 13676 | 0 | `stage_changed` | 528.840 | 0 |
| 6 | 61355 | 5 | inst5 | **true** | 13388 | 0 | `stage_changed` | 516.755 | 0 |
| 7 | 61356 | 1 | inst1 | **true** | 13196 | 0 | `stage_changed` | 552.205 | 0 |
| 8 | 61357 | 2 | inst2 | **true** | 13292 | 0 | `stage_changed` | 555.823 | 0 |
| 9 | 61358 | 5 | inst5 | **true** | 13100 | 0 | `stage_changed` | 552.793 | 0 |
| 10 | 61359 | 3 | inst3 | **true** | 13868 | 0 | `stage_changed` | 589.442 | 0 |
| 11 | 61360 | 0 | inst0 | **true** | 13100 | 0 | `stage_changed` | 551.212 | 0 |
| 12 | 61361 | 4 | inst4 | **true** | 13100 | 0 | `stage_changed` | 549.264 | 0 |
| 13 | 61362 | 1 | inst1 | **true** | 13292 | 0 | `stage_changed` | 493.837 | 0 |
| 14 | 61363 | 2 | inst2 | **true** | 13196 | 0 | `stage_changed` | 498.503 | 0 |
| 15 | 61364 | 5 | inst5 | **true** | 13292 | 0 | `stage_changed` | 498.636 | 0 |
| 16 | 61365 | 0 | inst0 | **true** | 13292 | 0 | `stage_changed` | 501.533 | 0 |
| 17 | 61366 | 4 | inst4 | **true** | 13196 | 0 | `stage_changed` | 492.330 | 0 |
| 18 | 61367 | 3 | inst3 | **true** | 13484 | 0 | `stage_changed` | 499.917 | 0 |
| 19 | 61368 | 1 | inst1 | **true** | 13580 | 0 | `stage_changed` | 479.838 | 0 |
| 20 | 61369 | 2 | inst2 | **true** | 13484 | 0 | `stage_changed` | 478.405 | 0 |
| 21 | 61370 | 5 | inst5 | **true** | 13484 | 0 | `stage_changed` | 475.281 | 0 |
| 22 | 61371 | 4 | inst4 | **true** | 13484 | 0 | `stage_changed` | 479.548 | 0 |
| 23 | 61372 | 0 | inst0 | **true** | 13484 | 0 | `stage_changed` | 475.502 | 0 |
| 24 | 61373 | 3 | inst3 | **true** | 13484 | 0 | `stage_changed` | 454.617 | 0 |
| 25 | 61374 | 1 | inst1 | **true** | 13292 | 0 | `stage_changed` | 182.476 | 0 |

口径说明：

- `stage_clear` 用教师报告自己的 `stage_clear`（引擎 stage index 前进到 1），不是"活着到帧上限"；25 局 `stop_reason` 全为 `stage_changed`，**没有一局**是 `max_frames` 或 `death`。`stage_at_stop` 全为 1。
- `frames` = 报告的 `frames_committed`（权威播放长度）；25 局 `survival_frames == frames` 逐局成立。
- `exit` = `tools/rl/v400_teacher.py` 的单局退出码，25/25 为 0；报告 `status` 25/25 为 `ok`、`completed` 25/25 为 `true`、`report_mode` 25/25 为 `fast`、`report_seed == seed` 25/25 成立、`error` 25/25 为 `null`。
- `wall_s` 是单局墙钟（6 路并发期间 454.6–589.4 s；末尾 61374 独占机器，182.5 s）。
- 数据目录按 worker 固定：`inst0..inst5` 一一绑定；`/root/v400-data/g1` 下实际有 8 个 `inst`，本批只用 0–5（与 chunk2 一致）。

## 3 统计

| 指标 | 值 |
|---|---|
| 声明 / 记录 / 完成 | 25 / 25 / **25** |
| **clears** | **25 / 25**（100.0%，`clear_seeds` = 61350…61374，`not_cleared_seeds=[]`） |
| **missing** | **[]**（无缺失） |
| **failed** | **[]**（无失败） |
| frames 均值 | **13361.12**（工具日志 `13361.1`） |
| frames 中位数 | **13292.0** |
| frames 最小 / 最大 | 13100（61358、61360、61361）/ 13868（61359） |
| frames 合计 | 334,028 |
| deaths 合计 / 有死亡局数 | **0** / **0** |
| bombs_spent | 0.0（25 局） |
| stop_reason 分布 | `stage_changed` × 25 |
| exit_code 分布 | `0` × 25 |
| report mode 分布 | `fast` × 25 |
| plans_verified 合计 | 83,507（= decisions 83,507，`verify_k=1`，逐局等式成立） |
| 单局 wall | 182.476 – 589.442 s；合计 12,476.234 s，均值 499.049 s |
| realtime_multiple | 0.395 – 0.4988（6 路并发段）；61374 独占段 1.2358 |
| **批次墙钟** | **2,281 s（38 min 1 s）**，6 路并发 |
| 重试 | **0** |

frames 分布（96 帧为单位的门控边界值）：

```
13100 ×3, 13196 ×5, 13292 ×5, 13388 ×2, 13484 ×7, 13580 ×1, 13676 ×1, 13868 ×1
```

## 4 引擎与一致性

| 项 | 值 |
|---|---|
| `--engine` / `--binary` | `/root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern` |
| **预期 SHA256** | `1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f` |
| **25 局报告到的 SHA256 集合** | `["1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f"]`（只有一个值） |
| `engine.match` | 25/25 `true`；`engine.sha_match_all = true` |
| 预检（本地哈希） | `preflight.checked=true, match=true`（工具在任何教师进程启动前先拒绝错构建） |
| `execution_ok` | true（`exit 0` × 25、`status=ok` × 25） |
| `verification_ok` | true（mode=fast × 25、SHA match × 25、`plans_verified == decisions × verify_k` = 83,507） |
| 几何/引擎死亡分歧 | `geo_dead_engine_alive` 0、`geo_alive_engine_dead` 0、`agree_alive` 83,507、`agree_dead` 0 |
| `trajectory_mismatch` | 68,129 / 83,507（81.6%，既有度量，**不作为通过条件**，见 §6） |

`fast_gate`（工具自带的 WP02.2 门，**不是** G410-1 门）：

```
rule            : clear_count >= 18 of 25 AND no execution/verification inconsistency
min_clears      : 18
required_clears : 18
applicable      : true
declared        : 25
completed       : 25
clears          : 25 / 25
execution_ok    : true
verification_ok : true
passed          : true
recommendation  : fast
note            : fast 达到 18/20 过面门（25/25），执行与验证一致：后续教师采集可用 fast   ← 工具固定文案，阈值按 25 个声明种子计
```

**这条 gate 的含义仅限于"本 chunk 的 25 局跑完且一致"**：分母是本次声明的 25 个种子（`required_clears = 18`），不是 G410-1 的 100 种子口径。`summary.status=passed` / `pass=true` 同理。

工具 summary 里的 `comparison_v2h` 显示 `found=0`、`missing_in_v2h` 为全部 25 个 seed：那是**参照文件覆盖范围**的问题（`.tmp_v400_laser/s1n/round1-v2h/runs.jsonl` 只含开发种子 1000–1019），**不是**本批失败，也不影响任何门。

## 5 与 chunk1（61300–61324）、chunk2（61325–61349）及 WP02.2 `fast20`（开发种子 1000–1019）的描述性对照

同一引擎 SHA、同一 stage/difficulty、同一 19887 帧上限、同一 6 路并发与数据目录，只是种子不同：

| 指标 | 本批（61350–61374） | chunk2（61325–61349） | chunk1（61300–61324） | fast20（1000–1019） |
|---|---|---|---|---|
| clears | 25 / 25 | 25 / 25 | 25 / 25 | 20 / 20 |
| frames 均值 | **13361.12** | 13422.24 | 13387.68 | 13282.4 |
| frames 中位数 | **13292.0** | 13292.0 | 13292.0 | 13292.0 |
| frames 最小 / 最大 | 13100 / 13868 | 13100 / 15300 | 13004 / 15204 | 13004 / 13676 |
| deaths | 0 | 0 | 0 | 0 |
| 批次墙钟 | 2,281 s | 2,567 s | 2,592 s | — |
| 单局 wall 均值 | 499.0 s | 575.0 s | 586.3 s | 473.3 s |

三批合并的**描述性**合计（75 个确认种子，仍不是 G410-1 结论）：clears **75/75**、frames 合计 **1,004,276**、均值 **13,390.35**、deaths 0（中位数需 chunk1 的逐局帧数才能合并，本文件不臆造）。

这只是**描述性**对照：种子集不同，本批不是 chunk1/chunk2/fast20 的重复实验，也不构成"哪一段确认种子更容易/更难"的因果结论。两批 `realtime_multiple`（本批 0.395–0.4988 vs chunk2 0.294–0.575）与单局 wall 的差异**没有做归因实验**，只能说明墙钟/吞吐不可跨批比较：运行时段的负载构成不同（见 §6.5）。

## 6 限制（本证据**不能**支撑的结论）

1. **chunk only，不是 G410-1 通过**：G410-1 的确认种子范围是 61300–61399（100 个），本批只跑了 **61350–61374 共 25 个（第 3/4 段）**。25 < 100，本文件**既不是 G410-1 的中间结论，也不是任何形式的通过判定**；加上 chunk1/chunk2 也只有 75/100，仍需 61375–61399 一批，并按 G410-1 自己的口径单独出总结论。
2. **单 stage / 单难度**：只测 Stage 1（引擎 stage index 0）/ normal。其它 stage、其它难度、完整一周目未测。
3. **只跑 `fast` profile**：horizon 32 / beam 24 / `verify_k` 1 / max_iterations 1。本批不回答 `fast` 与完整/`v2h` 教师（48 / 64 / 3 / 3）的强弱关系。
4. **0 死亡**：25 局没有任何死亡样本，本批**不能**用于死亡分布、死亡原因或"fast 的死亡处理"结论。
5. **wall / 吞吐不可比**：整个运行窗口里本机还有另一个 WP05 `dagger` 采集任务（`python -m touhou_rl.v410.dagger.labels --shard .tmp_v410_wp05_round2/...`，不同数据目录）在跑。运行期间采样的 1 分钟 load average 为 48.9 / 57.2 / 66.3（20 vCPU），其中本批 6 个 `v400_teacher.py` 各占 ~280% CPU 是主要来源，`dagger` 进程约占 31%。因此 `realtime_multiple` 只有 0.395–0.4988（6 路并发段），61374 独占机器时才回到 1.2358。单局 wall 反映的是当时负载，不是策略性质。另：本批结束（13:41:14）后约 13:42 又有 `tools/rl/v410_student_eval.py`（`rl_v3` 引擎、`--data-dir /root/v400-data/game`）启动，它在本批窗口**之外**，未影响本批任何一局。
6. **`trajectory_mismatch` 不是失败**：68,129/83,507（81.6%）是"几何预测轨迹与引擎回放轨迹在视野内不完全逐帧一致"的既有量（chunk2 为 80.4%、chunk1 为 80.7%、fast20 为 80.1%）；`verify_k` 正是为此做引擎验证。与**死亡**相关的两个计数器均为 0。
7. **引擎单点**：全部结论绑定在 SHA256 `1462…0d2f` 的 `rl_laser/th08-modern` 构建上；换构建必须重跑。
8. **未触碰**：confirm 种子 61000–61099 未读取；未训练；未写 `state.json` / `journal.jsonl` / `budget.jsonl` / `assets.json`；未 commit/push；未改 `touhou_rl/v400`、`v380`、`v390`、`v410`、`tools/rl/v400_*`、`tools/rl/v410*.py`、`tests/`、`third_party/`、v4.1.0 既有文件（含 `G4101_CHUNK1.md`、`G4101_CHUNK2.md`、`g4101_s1n_chunk1*`、`g4101_s1n_chunk2*`、`fast20*`、`AUDIT.md`、`CALIBRATION.md`、`DEATH.md`、`LABELS.md`、`STATUS.md`、`audit.json`、`labels_*`、`dead_sample*`）。
9. **单批次自洽 ≠ 可外推**：25/25 只是本段种子的结果；它不能预测 61375–61399 也 25/25，也不能替代 G410-1 的 100 种子汇总。

## 7 下一批建议（61375–61399）

- 命令形状与本批完全一致，只改三个路径（种子、证据文件、运行目录），阈值/参数一律不动：

```bash
cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v410_teacher_fast20.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern \
  --expected-sha 1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f \
  --seeds 61375-61399 \
  --out-jsonl Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk4.jsonl \
  --out-summary Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk4_summary.json \
  --tmp-dir /tmp/v410_g4101_chunk4 --quiet --force
```

- 预算：本批 25 局 / 6 路并发实测 **38 min 1 s**（含并行负载）；若届时机器空闲、单局回到 ~450 s，估计 ~32 min。单局硬超时仍是 3600 s，无需调整。
- 工程建议：
  1. 若要与本批/ chunk2 的 wall 可比，尽量**避开**其它引擎密集任务（如 WP05 `dagger`），否则 wall 与 `realtime_multiple` 只反映负载；本批已如实记录该干扰。
  2. 保持 `--force`（chunk 文件是新建的）；若中途被杀，**去掉 `--force`** 即可断点续跑已完成的种子。
  3. 每批结束后按本文件的口径核三件事：`missing`/`failed` 为空、`engine.sha_match_all=true`、`plans_verified == decisions`（本批 25/25 成立）。
  4. 四批齐了（100/100）再谈 G410-1：**单批或拼接后的 chunk 数据都不是 G410-1 通过**，判定要按 G410-1 的 100 种子口径单独出。

## 8 产物清单

产物（均在白名单内）：

- `Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk3.jsonl`：25 行（44,719 B，SHA256 `2394dfc46926441768162e3eb89921124ddda6eea36d73e1101b1a9492afbe19`），按 seed 升序，每行一局（含实际下发的教师 `command`、frames/deaths/stage_clear/stop_reason/exit_code/wall_s/worker/data_dir/engine SHA/mode/decisions/verify_k/plans_verified/consistency/execution_ok/verification_ok）。
- `Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk3_summary.json`：schema `v410-fast20/1`，工具原样输出 + **另加一个 `g4101_chunk3` 注释块**（本批命令、批次墙钟、engine SHA、chunk-only 声明、retries=0；工具生成的 12 个顶层字段一个未改，写前用 JSON 往返逐字节核验、写后再从磁盘复核）。
- `Document/Update/Alpha/v4.1.0/evidence/WP02/G4101_CHUNK3.md`：本文件。
- `/tmp/v410_g4101_chunk3/`（不入库）：`61350–61374.jsonl`（教师原始报告+决策行，行数 = 1 + plans_verified，本批 3,276–3,468 行）、`61350–61374.log`（教师 stdout 报告）、`main.log`、`started_at.txt`、`finished_at.txt`、`exit_code.txt`，以及本批注释/核对用脚本 `annotate.py`（写入 summary 注释块）、`report.py`（派生统计与逐局核对）、`facts_chunk3.json`。

未改动：`touhou_rl/v400/`、`v380/`、`v390/`、`v410/`、`tools/rl/v400_*`、`tools/rl/v410*.py`、`tests/`、`third_party/`、v4.1.0 既有文件、其它 WP 证据、`state.json` / `journal.jsonl` / `budget.jsonl` / `assets.json`。
