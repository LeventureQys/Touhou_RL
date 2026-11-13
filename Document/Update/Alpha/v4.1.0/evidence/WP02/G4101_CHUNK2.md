# G410-1 第二批确认种子（61325–61349）真实评测 — chunk 2（证据）

- 工作包：v4.1.0 **G410-1**（教师正式确认种子 61300–61399）的**第二批 25 个种子**
- 本文件范围：**只出数据**——不改教师、不训练、不做 G410-1 总结论
- 工具：`tools/rl/v410_teacher_fast20.py`（只读；只驱动 `tools/rl/v400_teacher.py`，不改教师策略）
- 教师 profile / 引擎：`fast` 教师 + `rl_laser` 引擎
- 机器可读证据：`Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk2.jsonl`（25 行）、`g4101_s1n_chunk2_summary.json`（schema `v410-fast20/1` + 附加注释块 `g4101_chunk2`）
- 运行目录（不入库）：`/tmp/v410_g4101_chunk2/`（25 份教师原始 JSONL + 25 份教师日志 + `main.log` + `started_at.txt` + `finished_at.txt` + `exit_code.txt`）
- 环境：WSL2 Ubuntu / Python 3.10.12 / 20 vCPU；教师引擎 `/root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern`，SHA256 `1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f`
- 活动声明：**未训练、未采数据集、未改引擎/教师实现、未写 state/journal/budget/assets、未 commit/push**；**未读取 confirm 种子 61000–61099**

## 0 一句话

**25/25 局过面（`stage_clear=true`）、0 死亡、25/25 `exit 0`、引擎 SHA 全部匹配、`missing=[]`、`failed=[]`**——这是 G410-1 确认种子第二批（61325–61349）的**真实评测数据**。
**它不是 G410-1 的通过结论**：本批只有 25 个种子（< 100），G410-1 的口径是 61300–61399 共 100 个确认种子，本文件只覆盖其中第 2/4 段；即使把 chunk1（61300–61324）与本批拼起来也只有 **50/100**。

## 1 实际执行的命令与退出码

```bash
cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v410_teacher_fast20.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern \
  --expected-sha 1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f \
  --seeds 61325-61349 \
  --out-jsonl Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk2.jsonl \
  --out-summary Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk2_summary.json \
  --tmp-dir /tmp/v410_g4101_chunk2 --quiet --force
# [fast20] PASSED: clears=25/25 frames_mean=13422.2 frames_median=13292.0 missing=[] failed=[] fast_gate=PASS
# [fast20] fast 达到 18/20 过面门（25/25），执行与验证一致：后续教师采集可用 fast
# 批次退出码：EXIT_CODE=0        （/tmp/v410_g4101_chunk2/exit_code.txt）
```

- **退出码 0**（`wsl.exe` 透传的工具进程退出码，由 `bash -lc` 末尾 `echo EXIT_CODE=$?` 落盘到 `/tmp/v410_g4101_chunk2/exit_code.txt`）。工具自己在判定后 `return EXIT_PASS`，含义是"批次完整跑完、无 missing/failed"。
- 与任务给定命令的差异只有一处：追加 `--tmp-dir /tmp/v410_g4101_chunk2`（任务指定的运行目录，工具默认值 `/tmp/v410_fast20` 属于 WP02.2）。**其余参数逐字一致**，未改任何阈值/参数。
- 启动 `2026-11-13T10:38:28+08:00`（`started_at.txt`），首局开始 `10:38:28`，末局结束 `11:21:15`：**批次墙钟 2,567 s ≈ 42 min 47 s**，6 路并发。
- 教师单局参数（`runs[].command` 原样记录在 JSONL 每行里）：`--decisions 10000 --max-frames 19887 --stop-on-death --stage 0 --difficulty normal --seed <seed> --mode fast --quiet --data-dir /root/v400-data/g1/inst<k> --engine <同上> --binary <同上> --expected-sha 1462…0d2f`。
- stdout/verdict 落在 `/tmp/v410_g4101_chunk2/main.log`；25 份教师报告 `61325.jsonl … 61349.jsonl`（行数 = 1 + `plans_verified`，逐局核对成立）、25 份教师日志 `61325.log … 61349.log` 完整保留。
- **重试：0 次**。本批无失败、无超时、无 missing，未触发"同一 seed 最多 2 次重试"的路径；`runs[]` 中每个 seed 恰好一行，无重复 seed。
- 数据目录用工具默认的 `/root/v400-data/g1/inst0..5`（与 chunk1 相同，按 worker 一一绑定，同一目录不会被两个进程并发使用）；任务提到的 `/root/v400-data/game`（游戏本体数据）**未被本批写入**。

## 2 逐局结果（25 行）

| # | seed | worker | data_dir | stage_clear | frames | deaths | stop_reason | wall_s | exit |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 61325 | 0 | inst0 | **true** | 13196 | 0 | `stage_changed` | 640.394 | 0 |
| 2 | 61326 | 1 | inst1 | **true** | 13292 | 0 | `stage_changed` | 640.280 | 0 |
| 3 | 61327 | 2 | inst2 | **true** | 15300 | 0 | `stage_changed` | 872.842 | 0 |
| 4 | 61328 | 3 | inst3 | **true** | 13292 | 0 | `stage_changed` | 637.008 | 0 |
| 5 | 61329 | 4 | inst4 | **true** | 13676 | 0 | `stage_changed` | 656.584 | 0 |
| 6 | 61330 | 5 | inst5 | **true** | 13292 | 0 | `stage_changed` | 639.089 | 0 |
| 7 | 61331 | 3 | inst3 | **true** | 13196 | 0 | `stage_changed` | 635.363 | 0 |
| 8 | 61332 | 5 | inst5 | **true** | 13580 | 0 | `stage_changed` | 654.690 | 0 |
| 9 | 61333 | 1 | inst1 | **true** | 13772 | 0 | `stage_changed` | 663.115 | 0 |
| 10 | 61334 | 0 | inst0 | **true** | 13772 | 0 | `stage_changed` | 661.592 | 0 |
| 11 | 61335 | 4 | inst4 | **true** | 13196 | 0 | `stage_changed` | 636.724 | 0 |
| 12 | 61336 | 2 | inst2 | **true** | 13484 | 0 | `stage_changed` | 656.784 | 0 |
| 13 | 61337 | 3 | inst3 | **true** | 13196 | 0 | `stage_changed` | 578.472 | 0 |
| 14 | 61338 | 4 | inst4 | **true** | 13484 | 0 | `stage_changed` | 585.252 | 0 |
| 15 | 61339 | 5 | inst5 | **true** | 13292 | 0 | `stage_changed` | 572.798 | 0 |
| 16 | 61340 | 0 | inst0 | **true** | 13100 | 0 | `stage_changed` | 567.556 | 0 |
| 17 | 61341 | 1 | inst1 | **true** | 13196 | 0 | `stage_changed` | 567.964 | 0 |
| 18 | 61342 | 2 | inst2 | **true** | 13292 | 0 | `stage_changed` | 514.326 | 0 |
| 19 | 61343 | 3 | inst3 | **true** | 13292 | 0 | `stage_changed` | 473.623 | 0 |
| 20 | 61344 | 5 | inst5 | **true** | 13100 | 0 | `stage_changed` | 473.652 | 0 |
| 21 | 61345 | 0 | inst0 | **true** | 13388 | 0 | `stage_changed` | 486.594 | 0 |
| 22 | 61346 | 1 | inst1 | **true** | 13196 | 0 | `stage_changed` | 476.387 | 0 |
| 23 | 61347 | 4 | inst4 | **true** | 13388 | 0 | `stage_changed` | 479.670 | 0 |
| 24 | 61348 | 2 | inst2 | **true** | 13196 | 0 | `stage_changed` | 386.067 | 0 |
| 25 | 61349 | 3 | inst3 | **true** | 13388 | 0 | `stage_changed` | 218.115 | 0 |

口径说明：

- `stage_clear` 用教师报告自己的 `stage_clear`（引擎 stage index 前进到 1），不是"活着到帧上限"；25 局 `stop_reason` 全为 `stage_changed`，**没有一局**是 `max_frames` 或 `death`。`stage_at_stop` 全为 1。
- `frames` = 报告的 `frames_committed`（权威播放长度）；25 局 `survival_frames == frames` 逐局成立。
- `exit` = `tools/rl/v400_teacher.py` 的单局退出码，25/25 为 0；报告 `status` 25/25 为 `ok`、`completed` 25/25 为 `true`、`report_mode` 25/25 为 `fast`、`report_seed == seed` 25/25 成立。
- `wall_s` 是单局墙钟（6 路并发期间 386–873 s；末尾 61349 独占机器，218 s）。
- 数据目录按 worker 固定：`inst0..inst5` 一一绑定；`/root/v400-data/g1` 下实际有 8 个 `inst`，本批只用 0–5。

## 3 统计

| 指标 | 值 |
|---|---|
| 声明 / 记录 / 完成 | 25 / 25 / **25** |
| **clears** | **25 / 25**（100.0%，`clear_seeds` = 61325…61349，`not_cleared_seeds=[]`） |
| **missing** | **[]**（无缺失） |
| **failed** | **[]**（无失败） |
| frames 均值 | **13422.24**（工具日志 `13422.2`） |
| frames 中位数 | **13292.0** |
| frames 最小 / 最大 | 13100（61340、61344）/ 15300（61327） |
| frames 合计 | 335,556 |
| deaths 合计 / 有死亡局数 | **0** / **0** |
| bombs_spent | 0.0（25 局） |
| stop_reason 分布 | `stage_changed` × 25 |
| exit_code 分布 | `0` × 25 |
| report mode 分布 | `fast` × 25 |
| plans_verified 合计 | 83,889（= decisions 83,889，`verify_k=1`，逐局等式成立） |
| 单局 wall | 218.115 – 872.842 s；合计 14,374.941 s，均值 574.998 s |
| realtime_multiple | 0.294 – 1.040（6 路并发段 0.294–0.575；61349 独占段 1.040） |
| **批次墙钟** | **2,567 s（42 min 47 s）**，6 路并发 |
| 重试 | **0** |

frames 分布（96 帧为单位的门控边界值，另有 61327 的 15300）：

```
13100 ×2, 13196 ×7, 13292 ×6, 13388 ×3, 13484 ×2, 13580 ×1, 13676 ×1, 13772 ×2, 15300 ×1
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
| `verification_ok` | true（mode=fast × 25、SHA match × 25、`plans_verified == decisions × verify_k` = 83,889） |
| 几何/引擎死亡分歧 | `geo_dead_engine_alive` 0、`geo_alive_engine_dead` 0、`agree_alive` 83,889、`agree_dead` 0 |
| `trajectory_mismatch` | 67,452 / 83,889（80.4%，既有度量，**不作为通过条件**，见 §6） |

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

## 5 与 chunk1（61300–61324）及 WP02.2 `fast20`（开发种子 1000–1019）的描述性对照

同一引擎 SHA、同一 stage/difficulty、同一 19887 帧上限、同一 6 路并发与数据目录，只是种子不同：

| 指标 | 本批（61325–61349） | chunk1（61300–61324） | fast20（1000–1019） |
|---|---|---|---|
| clears | 25 / 25 | 25 / 25 | 20 / 20 |
| frames 均值 | **13422.24** | 13387.68 | 13282.4 |
| frames 中位数 | **13292.0** | 13292.0 | 13292.0 |
| frames 最小 / 最大 | 13100 / 15300 | 13004 / 15204 | 13004 / 13676 |
| deaths | 0 | 0 | 0 |
| 批次墙钟 | 2,567 s | 2,592 s | — |
| 单局 wall 均值 | 575.0 s | 586.3 s | 473.3 s |

两批合并的**描述性**合计（50 个确认种子，仍不是 G410-1 结论）：clears 50/50、frames 均值 **13404.96**、中位数 **13292.0**、合计 670,248、deaths 0。

这只是**描述性**对照：种子集不同，本批不是 chunk1/fast20 的重复实验，也不构成"确认种子比开发种子更容易/更难"的因果结论。

## 6 限制（本证据**不能**支撑的结论）

1. **chunk only，不是 G410-1 通过**：G410-1 的确认种子范围是 61300–61399（100 个），本批只跑了 **61325–61349 共 25 个（第 2/4 段）**。25 < 100，本文件**既不是 G410-1 的中间结论，也不是任何形式的通过判定**；加上 chunk1 也只有 50/100，仍需 61350–61374、61375–61399 两批，并按 G410-1 自己的口径汇总。
2. **单 stage / 单难度**：只测 Stage 1（引擎 stage index 0）/ normal。其它 stage、其它难度、完整一周目未测。
3. **只跑 `fast` profile**：horizon 32 / beam 24 / `verify_k` 1 / max_iterations 1。本批不回答 `fast` 与完整/`v2h` 教师（48 / 64 / 3 / 3）的强弱关系。
4. **0 死亡**：25 局没有任何死亡样本，本批**不能**用于死亡分布、死亡原因或"fast 的死亡处理"结论。
5. **wall / 吞吐不可比**：整个运行窗口里本机还有另一个 WP05 `dagger` 采集任务（`rl_v3` 引擎、不同数据目录）在跑，运行前后 load average ≈ 16.9（20 vCPU）。因此 `realtime_multiple` 只有 0.294–0.575（6 路并发段），61349 独占机器时才回到 1.040。单局 wall 反映的是当时负载，不是策略性质。
6. **`trajectory_mismatch` 不是失败**：67,452/83,889（80.4%）是"几何预测轨迹与引擎回放轨迹在视野内不完全逐帧一致"的既有量（chunk1 为 80.7%、fast20 为 80.1%）；`verify_k` 正是为此做引擎验证。与**死亡**相关的两个计数器均为 0。
7. **引擎单点**：全部结论绑定在 SHA256 `1462…0d2f` 的 `rl_laser/th08-modern` 构建上；换构建必须重跑。
8. **未触碰**：confirm 种子 61000–61099 未读取；未训练；未写 `state.json` / `journal.jsonl` / `budget.jsonl` / `assets.json`；未 commit/push；未改 `tools/rl/v410*.py`、`tools/rl/v400_*`、`third_party/`、`tests/`、v4.1.0 既有文件。

## 7 下一批建议（61350–61374）

- 命令形状与本批完全一致，只改三个路径（种子、证据文件、运行目录），阈值/参数一律不动：

```bash
cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v410_teacher_fast20.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern \
  --expected-sha 1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f \
  --seeds 61350-61374 \
  --out-jsonl Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk3.jsonl \
  --out-summary Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk3_summary.json \
  --tmp-dir /tmp/v410_g4101_chunk3 --quiet --force
```

- 预算：本批 25 局 / 6 路并发实测 **42 min 47 s**（含并行负载）；若届时机器空闲，按单局 473 s 估计约 35 min。单局硬超时仍是 3600 s，无需调整。
- 工程建议：
  1. 尽量**避开**与其它引擎密集任务（如 WP05 `dagger`）并行，否则 wall 与 `realtime_multiple` 只反映负载；本批已如实记录该干扰。
  2. 保持 `--force`（chunk 文件是新建的）；若中途被杀，**去掉 `--force`** 即可断点续跑已完成的种子。
  3. 每批结束后按本文件的口径核三件事：`missing`/`failed` 为空、`engine.sha_match_all=true`、`plans_verified == decisions`。
  4. 四批齐了再谈 G410-1：**单批/拼接后的 chunk 数据都不是 G410-1 通过**，判定要按 G410-1 的 100 种子口径单独出。

## 8 产物清单与未改动项

产物（均在白名单内）：

- `Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk2.jsonl`：25 行，按 seed 升序，每行一局（含实际下发的教师 `command`、frames/deaths/stage_clear/stop_reason/exit_code/wall_s/worker/data_dir/engine SHA/mode/decisions/verify_k/plans_verified/consistency/execution_ok/verification_ok）。
- `Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk2_summary.json`：schema `v410-fast20/1`，工具原样输出；**另加一个 `g4101_chunk2` 注释块**（本批命令、批次墙钟、engine SHA、chunk-only 声明、retries=0；工具生成的字段一个未改，已用 JSON 往返逐字节核验）。
- `Document/Update/Alpha/v4.1.0/evidence/WP02/G4101_CHUNK2.md`：本文件。
- `/tmp/v410_g4101_chunk2/`（不入库）：`61325–61349.jsonl`（教师原始报告+决策行，行数 = 1 + plans_verified）、`61325–61349.log`（教师 stdout 报告）、`main.log`、`started_at.txt`、`finished_at.txt`、`exit_code.txt`。

未改动：`touhou_rl/v400/`、`v380/`、`v390/`、`v410/`、`tools/rl/v400_*`、`tools/rl/v410*.py`、`tests/`、`third_party/`、v4.1.0 既有文件（含 chunk1 的 `G4101_CHUNK1.md`、`g4101_s1n_chunk1*`、`fast20*`、`AUDIT.md`、`CALIBRATION.md`、`DEATH.md`、`LABELS.md`、`STATUS.md`、`audit.json`、`labels_*`、`dead_sample*`）、其它 WP 证据、`state.json` / `journal.jsonl` / `budget.jsonl` / `assets.json`。
