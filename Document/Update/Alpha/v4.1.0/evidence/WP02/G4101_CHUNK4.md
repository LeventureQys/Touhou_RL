# G410-1 第四批确认种子（61375–61399）真实评测 — chunk 4（证据）

- 工作包：v4.1.0 **G410-1**（教师正式确认种子 61300–61399）的**第四批、也是最后一批 25 个种子**
- 本文件范围：**只出数据**——不改教师、不训练、不做 G410-1 总结论
- 工具：`tools/rl/v410_teacher_fast20.py`（只读；只驱动 `tools/rl/v400_teacher.py`，不改教师策略）
- 教师 profile / 引擎：`fast` 教师 + `rl_laser` 引擎
- 机器可读证据：`Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk4.jsonl`（25 行）、`g4101_s1n_chunk4_summary.json`（schema `v410-fast20/1` + 附加注释块 `g4101_chunk4`）
- 运行目录（不入库）：`/tmp/v410_g4101_chunk4/`（25 份教师原始 JSONL + 25 份教师日志 + `main.log` + `started_at.txt` + `finished_at.txt` + `exit_code.txt` + 核对脚本）
- 环境：WSL2 / Python 3.10.12（`/root/v400-venv`）/ 20 vCPU；教师引擎 `/root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern`，SHA256 `1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f`
- 活动声明：**未训练、未采数据集、未改引擎/教师实现、未 commit/push**；**未读取 confirm 种子 61000–61099**；`state.json` / `journal.jsonl` / `budget.jsonl` **不是本批写入**（见 §1 末条）

## 0 一句话

**25/25 局过面（`stage_clear=true`）、0 死亡、25/25 `exit 0`、引擎 SHA 全部匹配、`missing=[]`、`failed=[]`、0 次重试**——这是 G410-1 确认种子第四批（61375–61399）的**真实评测数据**。
**它不是 G410-1 的通过结论**：本批只是 25 个种子；G410-1 的口径是 61300–61399 共 100 个确认种子。四批拼起来现在是 **100/100 局数据齐、100/100 过面、0 死亡**（描述性合计，§5），但 G410-1 的判定必须按它自己的 100 种子口径**单独出**，本文件（以及任何单批/拼接 chunk 文件）都不构成该判定。

## 1 实际执行的命令与退出码

```bash
cd /mnt/d/WorkShop/Touhou_RL && /root/v400-venv/bin/python tools/rl/v410_teacher_fast20.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern \
  --expected-sha 1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f \
  --seeds 61375-61399 \
  --out-jsonl Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk4.jsonl \
  --out-summary Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk4_summary.json \
  --tmp-dir /tmp/v410_g4101_chunk4 --quiet --force
# [fast20] PASSED: clears=25/25 frames_mean=13292.0 frames_median=13292.0 missing=[] failed=[] fast_gate=PASS
# [fast20] fast 达到 18/20 过面门（25/25），执行与验证一致：后续教师采集可用 fast
# 批次退出码：EXIT_CODE=0        （/tmp/v410_g4101_chunk4/exit_code.txt，内容 "0"）
```

- **退出码 0**：`bash -lc` 里取 `${PIPESTATUS[0]}`（即 `tools/rl/v410_teacher_fast20.py` 进程的退出码）落盘到 `/tmp/v410_g4101_chunk4/exit_code.txt`，并经 `wsl.exe` 透传为宿主侧退出码 0。工具在判定后 `return EXIT_PASS`，含义是"批次完整跑完（无 missing/failed）且本 chunk 的 fast_gate 通过"，**不是** G410-1 判定。
- 与任务给定命令**逐字一致**：`--engine` / `--expected-sha` / `--seeds 61375-61399` / 两个 `--out-*` / `--tmp-dir /tmp/v410_g4101_chunk4` / `--quiet` / `--force`，未改任何阈值或参数（`--min-clears` 未传 → 默认 18；`--max-frames` 未传 → 默认 19887；`--workers` 未传 → 默认 6；`--decisions` 未传 → 默认 10000）。
- 启动 `2026-11-13T13:51:16+08:00`（`started_at.txt`），末局结束 `2026-11-13T14:26:59+08:00`（`finished_at.txt`）：**批次墙钟 2,143 s = 35 min 43 s**，6 路并发。`runs[]` 中首局 `started_at` = 13:51:16，末局 `finished_at` = 14:26:59，与墙钟自洽。
- 教师单局参数（`runs[].command` 原样记在 JSONL 每行里）：`--decisions 10000 --max-frames 19887 --stop-on-death --stage 0 --difficulty normal --seed <seed> --mode fast --quiet --data-dir /root/v400-data/g1/inst<k> --engine <同上> --binary <同上> --expected-sha 1462…0d2f`。
- stdout/verdict 落在 `/tmp/v410_g4101_chunk4/main.log`（两行，见上）；25 份教师报告 `61375.jsonl … 61399.jsonl`（每份行数 = 1 + `plans_verified`，本批 3,252–3,444 行，逐局核对成立）、25 份教师日志 `61375.log … 61399.log` 完整保留（运行目录共 292 MB）。
- **重试：0 次**。本批无失败、无超时、无 missing；工具本身不做自动重试，任务允许的"同一 seed 最多 2 次重试"**未被使用**，`runs[]` 中每个 seed 恰好一行。
- 数据目录：给定命令未带 `--data-root`，工具按默认用 `/root/v400-data/g1/inst0..5`（与 chunk1/chunk2/chunk3 相同，按 worker 一一绑定，同一目录不会被两个进程并发使用）。任务背景里写的 `/root/v400-data/game` **不能**作为 `--data-root`（该目录下没有 `inst<k>` 子目录，工具会直接拒绝），本批也**没有写入**它：`/root/v400-data/game/score.dat` mtime `13:47:56`、`th08.cfg` mtime `13:47:50`，均早于本批启动的 `13:51:16`。
- **`state.json` / `journal.jsonl` / `budget.jsonl` 不是本批写的**：`v400_teacher.py` 只写 `--out` 与 `--data-dir`，本批 25 个进程没有任何一条命令带状态路径。审计若看到它们变动，来源是编排层的记账（`journal.jsonl` / `budget.jsonl` mtime `14:27:44`，`state.json` mtime `13:51:04`，均在本批窗口边界之外/由外部写入）。

## 2 逐局结果（25 行）

| # | seed | worker | data_dir | stage_clear | frames | deaths | stop_reason | wall_s | exit |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 61375 | 0 | inst0 | **true** | 13388 | 0 | `stage_changed` | 509.055 | 0 |
| 2 | 61376 | 1 | inst1 | **true** | 13388 | 0 | `stage_changed` | 504.485 | 0 |
| 3 | 61377 | 2 | inst2 | **true** | 13388 | 0 | `stage_changed` | 505.942 | 0 |
| 4 | 61378 | 3 | inst3 | **true** | 13580 | 0 | `stage_changed` | 510.421 | 0 |
| 5 | 61379 | 4 | inst4 | **true** | 13292 | 0 | `stage_changed` | 502.502 | 0 |
| 6 | 61380 | 5 | inst5 | **true** | 13292 | 0 | `stage_changed` | 507.825 | 0 |
| 7 | 61381 | 4 | inst4 | **true** | 13100 | 0 | `stage_changed` | 477.867 | 0 |
| 8 | 61382 | 1 | inst1 | **true** | 13292 | 0 | `stage_changed` | 482.572 | 0 |
| 9 | 61383 | 2 | inst2 | **true** | 13100 | 0 | `stage_changed` | 477.130 | 0 |
| 10 | 61384 | 5 | inst5 | **true** | 13100 | 0 | `stage_changed` | 477.929 | 0 |
| 11 | 61385 | 0 | inst0 | **true** | 13484 | 0 | `stage_changed` | 492.457 | 0 |
| 12 | 61386 | 3 | inst3 | **true** | 13196 | 0 | `stage_changed` | 480.855 | 0 |
| 13 | 61387 | 4 | inst4 | **true** | 13196 | 0 | `stage_changed` | 464.973 | 0 |
| 14 | 61388 | 2 | inst2 | **true** | 13196 | 0 | `stage_changed` | 470.325 | 0 |
| 15 | 61389 | 5 | inst5 | **true** | 13292 | 0 | `stage_changed` | 475.604 | 0 |
| 16 | 61390 | 1 | inst1 | **true** | 13772 | 0 | `stage_changed` | 490.199 | 0 |
| 17 | 61391 | 3 | inst3 | **true** | 13676 | 0 | `stage_changed` | 490.605 | 0 |
| 18 | 61392 | 0 | inst0 | **true** | 13196 | 0 | `stage_changed` | 467.946 | 0 |
| 19 | 61393 | 4 | inst4 | **true** | 13004 | 0 | `stage_changed` | 454.183 | 0 |
| 20 | 61394 | 2 | inst2 | **true** | 13292 | 0 | `stage_changed` | 466.250 | 0 |
| 21 | 61395 | 5 | inst5 | **true** | 13100 | 0 | `stage_changed` | 461.092 | 0 |
| 22 | 61396 | 0 | inst0 | **true** | 13292 | 0 | `stage_changed` | 464.509 | 0 |
| 23 | 61397 | 1 | inst1 | **true** | 13388 | 0 | `stage_changed` | 465.149 | 0 |
| 24 | 61398 | 3 | inst3 | **true** | 13196 | 0 | `stage_changed` | 457.761 | 0 |
| 25 | 61399 | 4 | inst4 | **true** | 13100 | 0 | `stage_changed` | 186.941 | 0 |

口径说明：

- `stage_clear` 用教师报告自己的 `stage_clear`（引擎 stage index 前进到 1），不是"活着到帧上限"；25 局 `stop_reason` 全为 `stage_changed`，**没有一局**是 `max_frames` 或 `death`；`stage_at_stop` 25/25 为 1。
- `frames` = 报告的 `frames_committed`（权威播放长度）；25 局 `survival_frames == frames` 逐局成立。
- `exit` = `tools/rl/v400_teacher.py` 的单局退出码，25/25 为 0；报告 `status` 25/25 为 `ok`、`completed` 25/25 为 `true`、`report_mode` 25/25 为 `fast`、`report_seed == seed` 25/25 成立、`error` 25/25 为 `null`、`engine.match` 25/25 为 `true`、`jsonl_lines == 1 + plans_verified` 25/25 成立。
- `wall_s` 是单局墙钟（6 路并发段 454.2–510.4 s；末局 61399 独占机器时 186.9 s）。
- 数据目录按 worker 固定：`inst0..inst5` 一一绑定；`/root/v400-data/g1` 下实际有 8 个 `inst`，本批只用 0–5（与前三批一致）。
- 上表不是手抄：`/tmp/v410_g4101_chunk4/check_md.py` 把本文件 25 行表与 `g4101_s1n_chunk4.jsonl` 的 seed/worker/data_dir/stage_clear/frames/deaths/stop_reason/wall_s/exit_code 逐字段比对，**0 处差异**；`g4101_s1n_chunk4_summary.json` 的 `runs[]` 就是这 25 条记录（键值逐项相同，脚本按 JSON 规范化后比对相等）。

## 3 统计

| 指标 | 值 |
|---|---|
| 声明 / 记录 / 完成 | 25 / 25 / **25** |
| **clears** | **25 / 25**（100.0%，`clear_seeds` = 61375…61399，`not_cleared_seeds=[]`） |
| **missing** | **[]**（无缺失） |
| **failed** | **[]**（无失败） |
| frames 均值 | **13292.00**（工具日志 `13292.0`；与中位数恰好相等） |
| frames 中位数 | **13292.0** |
| frames 最小 / 最大 | 13004（61393）/ 13772（61390） |
| frames 合计 | 332,300 |
| deaths 合计 / 有死亡局数 | **0** / **0** |
| bombs_spent | 0.0（25 局） |
| stop_reason 分布 | `stage_changed` × 25 |
| exit_code 分布 | `0` × 25 |
| report mode 分布 | `fast` × 25 |
| plans_verified 合计 | 83,075（= decisions 83,075，`verify_k=1`，逐局等式成立） |
| 单局 wall | 186.941 – 510.421 s；合计 11,744.577 s，均值 469.783 s |
| realtime_multiple | 0.4392 – 0.4846（6 路并发段）；61399 独占段 1.1890 |
| **批次墙钟** | **2,143 s（35 min 43 s）**，6 路并发 |
| 重试 | **0** |

frames 分布（96 帧为单位的门控边界值）：

```
13004 ×1, 13100 ×5, 13196 ×5, 13292 ×6, 13388 ×4, 13484 ×1, 13580 ×1, 13676 ×1, 13772 ×1
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
| `verification_ok` | true（mode=fast × 25、SHA match × 25、`plans_verified == decisions × verify_k` = 83,075） |
| 几何/引擎死亡分歧 | `geo_dead_engine_alive` 0、`geo_alive_engine_dead` 0、`agree_alive` 83,075、`agree_dead` 0 |
| `trajectory_mismatch` | 67,366 / 83,075（81.1%，既有度量，**不作为通过条件**，见 §6） |

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

## 5 与前三批、以及 WP02.2 `fast20`（开发种子 1000–1019）的描述性对照

同一引擎 SHA、同一 stage/difficulty、同一 19887 帧上限、同一 6 路并发与数据目录，只是种子不同：

| 指标 | 本批（61375–61399） | chunk3（61350–61374） | chunk2（61325–61349） | chunk1（61300–61324） | fast20（1000–1019） |
|---|---|---|---|---|---|
| clears | 25 / 25 | 25 / 25 | 25 / 25 | 25 / 25 | 20 / 20 |
| frames 均值 | **13292.00** | 13361.12 | 13422.24 | 13387.68 | 13282.4 |
| frames 中位数 | **13292.0** | 13292.0 | 13292.0 | 13292.0 | 13292.0 |
| frames 最小 / 最大 | 13004 / 13772 | 13100 / 13868 | 13100 / 15300 | 13004 / 15204 | 13004 / 13676 |
| deaths | 0 | 0 | 0 | 0 | 0 |
| 批次墙钟 | **2,143 s** | 2,281 s | 2,567 s | 2,592 s | — |
| 单局 wall 均值 | 469.8 s | 499.0 s | 575.0 s | 586.3 s | 473.3 s |

**四批拼合的描述性合计（100 个确认种子，仍不是 G410-1 结论）**——由四份 chunk JSONL 逐行读取后合并计算（只读）：

| 指标 | 四批合计 |
|---|---|
| 局数 / 唯一 seed | **100 / 100**，种子区间 61300–61399 连续无缺 |
| clears | **100 / 100**（100.0%） |
| deaths | **0** |
| frames 合计 | **1,336,576** |
| frames 均值 / 中位数 | **13,365.76** / **13,292.0** |
| frames 最小 / 最大 | 13,004（61393）/ 15,300（chunk2） |

这只是**描述性**对照：种子集不同，本批不是前三批/fast20 的重复实验，也不构成"哪一段确认种子更容易/更难"的因果结论。各批 `realtime_multiple` 与单局 wall 的差异**没有做归因实验**，只能说明墙钟/吞吐不可跨批比较：运行时段的宿主负载构成不同（见 §6.5）。

## 6 限制（本证据**不能**支撑的结论）

1. **chunk only，不是 G410-1 通过**：G410-1 的确认种子范围是 61300–61399（100 个），本批只跑了 **61375–61399 共 25 个（第 4/4 段）**。25 < 100，本文件**既不是 G410-1 的中间结论，也不是任何形式的通过判定**。四批拼起来数据已齐（100/100 局、100/100 过面、0 死亡），但"数据齐"≠"G410-1 判定"：G410-1 的判定要按它自己的 100 种子口径、连同其独立审计项（如 8px 残差审计、数据腿）单独出，本文件不替代。
2. **单 stage / 单难度**：只测 Stage 1（引擎 stage index 0）/ normal。其它 stage、其它难度、完整一周目未测。
3. **只跑 `fast` profile**：本批教师报告自己的 `config` 是 `{"beam": 24, "horizon": 32, "verify_k": 1, "max_iterations": 1, "shoot_always": true, "shoot_pulse_period": 24, "durations": [1, 2]}`（25/25 一致）。本批不回答 `fast` 与完整/`v2h` 教师（48 / 64 / 3 / 3）的强弱关系。
4. **0 死亡**：25 局没有任何死亡样本（四批合计 100 局也是 0 死亡），本批**不能**用于死亡分布、死亡原因或"fast 的死亡处理"结论。
5. **wall / 吞吐不可比**：整个窗口里本机还有别的 v4.1.0 任务在跑（`/tmp` 下同时存在 `v410_cons_*`、`v410_seed_*` 等探测产物，仓库内 `.tmp_v410_wp05_train_beta05_5k/` 在本批窗口内持续写入 labels/ckpt）。运行期间采样的 1 分钟 load average 为 18.8（13:51）→ 43.2（14:22）→ 8.2（14:27），20 vCPU；本批 6 个 `v400_teacher.py` 各占 ~240% CPU 是主要构成之一，但 14:22 的峰值高于本批自身容量，说明还有其它并发负载（未在当时抓进程表，故不做归因）。因此 `realtime_multiple` 只有 0.4392–0.4846（6 路并发段），61399 独占机器时才回到 1.1890。单局 wall 反映的是当时负载，不是策略性质。
6. **`trajectory_mismatch` 不是失败**：67,366/83,075（81.1%）是"几何预测轨迹与引擎回放轨迹在视野内不完全逐帧一致"的既有量（chunk3 为 81.6%、chunk2 为 80.4%、chunk1 为 80.7%、fast20 为 80.1%）；`verify_k` 正是为此做引擎验证。与**死亡**相关的两个计数器均为 0。
7. **引擎单点**：全部结论绑定在 SHA256 `1462…0d2f` 的 `rl_laser/th08-modern` 构建上；换构建必须重跑。
8. **未触碰**：confirm 种子 61000–61099 未读取；未训练；**本批未写** `state.json` / `journal.jsonl` / `budget.jsonl`（这些文件在本批窗口外的 mtime 变动来自编排层记账，见 §1 末条）；未 commit/push；未改 `touhou_rl/v400`、`v380`、`v390`、`v410`、`tools/rl/v400_*`、`tools/rl/v410*.py`、`tests/`、`third_party/`、v4.1.0 既有文件（含 `G4101_CHUNK1/2/3.md`、`g4101_s1n_chunk1/2/3*`、`fast20*`、`AUDIT.md`、`CALIBRATION.md`、`DEATH.md`、`LABELS.md`、`STATUS.md`、`audit.json`、`labels_*`、`dead_sample*`）。
9. **单批次自洽 ≠ 可外推**：25/25 只是本段种子的结果；四批合计 100/100 也只是"这 100 个确认种子在 fast + 该引擎构建上的描述性结果"，不能外推到其它 stage/难度/教师/构建，也不替代 G410-1 的 100 种子判定。
10. **墙钟记账小偏差（留给汇总）**：编排层在 `budget.jsonl`/`journal.jsonl`（14:27:44）记的 chunk4 约为 ~2,280 s / 0.65 h，**实测批次墙钟是 2,143 s（0.595 h）**。差异不影响任何门；四批汇总时以 2,143 s 为准。

## 7 四批汇总建议（下一阶段）

1. **数据已齐**：61300–61399 共 100 个确认种子、100 局、100 过面、0 死亡、0 missing、0 failed、0 重试，分别落在四份 chunk 证据（`g4101_s1n_chunk1..4.jsonl` + 各自 summary）里，逐局 `command` 与报告字段都在。
2. **G410-1 判定要单独出**：不要用任何单份 chunk 的 `status=passed` / `pass=true` / `fast_gate=PASS` 当 G410-1 结论；判定文档应显式说明 100 种子口径、引擎 SHA、stage/difficulty、fast profile，并逐项引用四份 chunk 的 `missing`/`failed`/`engine.sha_match_all`/`plans_verified == decisions` 核对结果（四批均成立）。
3. **审计建议**：合并 100 局时可复用本文件的核对逻辑（`/tmp/v410_g4101_chunk4/report.py` 一类脚本）：每局 `report_seed == seed`、`jsonl_lines == 1 + plans_verified`、`survival_frames == frames`、`engine_sha256` 唯一；再核对四份 JSONL 的种子区间是否刚好拼成 61300–61399（本文件 §5 已做一次，结果连续无缺）。
4. **不要再跑第五批**：61300–61399 已全覆盖；若后续需要更多确认种子，应另开区间并重新登记，而不是重复本区间（重复运行会得到不同 seed 的同一性很强但不完全相同的结果，且会掩盖"是否真的 100 个"的边界）。
5. **口径卫生**：若之后要把 chunk4 与其它 WP 的墙钟一起汇总，用 2,143 s（实测）而不是 `budget.jsonl` 里的 ~2,280 s；跨批 wall 不可比（§6.5），只能作为工程量级参考。
6. **若要重跑**：命令形状与 §1 完全一致，去掉 `--force` 即可断点续跑已完成种子；单局硬超时默认 3600 s 无需调整。

## 8 产物清单

产物（均在白名单内）：

- `Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk4.jsonl`：25 行（44,719 B，SHA256 `5dbd64776c8c2595d629106a083ecb76be695451bd784f1da7269ac590bcd5b8`），按 seed 升序（61375→61399），每行一局（含实际下发的教师 `command`、frames/deaths/stage_clear/stop_reason/exit_code/wall_s/worker/data_dir/engine SHA/mode/decisions/verify_k/plans_verified/consistency/execution_ok/verification_ok）。
- `Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_s1n_chunk4_summary.json`：schema `v410-fast20/1`，工具原样输出的 12 个顶层字段 + **另加一个 `g4101_chunk4` 注释块**（本批命令、批次墙钟、engine SHA、chunk-only 声明、四批合计、retries=0）。写前把 12 个工具字段逐个用 JSON 往返比对确认未改，写后再从磁盘复核（工具原样输出 69,894 B / SHA256 `2eb8c6f9…1ce4d8`；加注释块后 74,269 B / SHA256 `0a558c72…569a4bd`，`status/pass/clears/wall_total_s` 等工具字段不变）。
- `Document/Update/Alpha/v4.1.0/evidence/WP02/G4101_CHUNK4.md`：本文件。
- `/tmp/v410_g4101_chunk4/`（不入库，67 个文件 / 291.4 MiB）：`61375–61399.jsonl`（教师原始报告 + 决策行，行数 = 1 + plans_verified，本批 3,252–3,444 行）、`61375–61399.log`（教师 stdout 报告，另有工具自己的 `main.log`）、`started_at.txt`、`finished_at.txt`、`exit_code.txt`，以及核对/注释用脚本 `report.py`（逐局核对与派生统计）、`check_md.py` / `check2.py`（本文件与 JSONL/summary 交叉核对）、`annotate.py`（写入 summary 注释块）、`fourbatches.py`（四批合计）、`peek*.py` / `peek_state.sh` / `recent*.sh`（只读核对）、`facts.txt`、`ps.txt` / `ps_now.txt`。

未改动：`touhou_rl/v400/`、`v380/`、`v390/`、`v410/`、`tools/rl/v400_*`、`tools/rl/v410*.py`、`tests/`、`third_party/`、v4.1.0 既有文件、其它 WP 证据、`state.json` / `journal.jsonl` / `budget.jsonl`。
