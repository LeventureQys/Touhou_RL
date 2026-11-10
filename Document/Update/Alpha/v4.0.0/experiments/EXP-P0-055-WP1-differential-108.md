# EXP-P0-055：WP1 退路 C — 108 份官方回放全量差分重放（harness ON vs 原样 OFF）

日期：2026-11-08 / 2026-11-09（UTC 15:51–17:35，本地跨零点）。Provider=DeepSeek，模型=DeepSeek v4.1 flash。

## 结论

**官方 accepted corpus 的 108/108 份真实回放全部通过「差分验证」：同一 replay 在 harness ON 构建（`build/rl_retrace`）与原样 portable OFF 构建（`build/portable-retrace`）下，以完全相同参数、同一纯记录模式播放到自然结束，25 字段追踪行文件逐字节一致，`complete.json` 也逐字节一致。`partial=false`，无超时、无失败、无重试。**

| 指标 | 值 |
|---|---|
| `total` | **108** |
| `recorded` | 108 |
| `compared` | 108 |
| `identical` | **108** |
| `different` | **0** |
| `timeout` | **0** |
| `failed` | **0** |
| `partial` | **false** |
| `remaining` | 0 |
| `allIdentical` | true |
| `totalRowsPerSide` | **12,311,460** |

> **`proofStrength` = `differential (harness ON vs portable OFF); not a bit-exact comparison against the retail executable`。**
> **这是差分验证，不是与零售版逐位一致。** 108/108 只证明「harness 构建没有改变原样构建的回放行为」，**不等同于** WP1/G0.1 要求的「108 份回放与零售轨迹逐位一致」。零售参考轨迹仍为 **0/108**，G0.1 状态**不因本卡翻转**；门报告必须写明证明力下降。

## 每分片完成数

| 分片 | journal 文件 | 记录数 | identical | different | timeout | failed | corruptLines | 行数合计（=计算帧） | 该片用例墙钟累计 |
|---|---|---|---|---|---|---|---|---|---|
| 1/4 | `shard-1.jsonl` | **27** | 27 | 0 | 0 | 0 | 0 | 3,119,757 | 4,087.9 s |
| 2/4 | `shard-2.jsonl` | **27** | 27 | 0 | 0 | 0 | 0 | 3,084,117 | 4,018.5 s |
| 3/4 | `shard-3.jsonl` | **27** | 27 | 0 | 0 | 0 | 0 | 3,061,181 | 4,009.6 s |
| 4/4 | `shard-4.jsonl` | **27** | 27 | 0 | 0 | 0 | 0 | 3,046,405 | 4,028.3 s |
| 合计 | 4 个 journal | **108** | **108** | 0 | 0 | 0 | 0 | **12,311,460** | 16,144.3 s |

- 每份 journal 均 27 行、零不可解析行（`shard-1..4.jsonl` 分别为 55,960 / 55,961 / 55,947 / 55,963 字节）。
- 分片按 `balanced_order`（预测负载降序）轮转分配，4 片各 27 份，无重复、无遗漏（`uniqueCases=108`、`duplicateCases=[]`）。

## 逐份一致性（108/108 聚合证明）

- **行数**：108/108 满足 `on.rows == off.rows == manifest.calculationFrames`；行数区间 **44,106 – 149,986**，均值 **113,995**，合计 **12,311,460**，与 `config/replay-fixtures.json` 的 `calculationFrames` 合计逐帧相同。
- **两侧 rows SHA-256**：108/108 两侧 SHA-256 **完全相同**（除逐字节相同外无其他可能；唯一不同的 SHA 数量为 0）。
- **两侧 `complete.json`**：108/108 两侧 SHA-256 **完全相同**（108 份各不相同，符合「每份 replay 各有一套完成元数据」）。
- **逐字节比较结果**：`comparison.rows.identical=108`、`comparison.completeJson.identical=108`，首个差异列表 `differences=[]` 为空，`notIdentical=[]` 为空。
- **每侧进程状态**：216 个 `side.json`（108 份 × ON/OFF）全部 `status=clean`、`completeEnded=true`、`completeErrors=[]`、`attempts=1`（**0 次重试**）；216 个退出码全部为 **0**，`timedOut` 全为 false。
- **行文件尺寸自洽**：12,311,460 行 × 100 B（25×u32）= **1,231,146,000 B/侧**（≈1.147 GiB），两侧合计 ≈2.29 GiB；无一份行文件尺寸不是 100 的整数倍。
- **难度覆盖**：Easy 24、Normal 24、Hard 24、Lunatic 24、Extra 12，合计 108。
- 逐份明细（行数 + 两侧 rows SHA-256 前 16 位 + 单侧耗时）见文末**附录**；完整 64 位 SHA 见 `report.json` 的 `cases[].on/off.rowsSha256` 与 `cases_108.tsv`。

## 耗时统计

| 统计量 | ON 侧 | OFF 侧 | 逐用例（两侧并行墙钟） |
|---|---|---|---|
| 合计 | 16,081.63 s（4h28m02s） | 16,061.92 s（4h27m42s） | **16,144.31 s（4h29m04s）** |
| 最小 | 40.638 s | 40.476 s | **40.743 s**（`th8_ud2c7a`，44,771 行） |
| 最大 | 270.787 s | 269.550 s | **271.114 s**（`th8_ud2cd4`，149,986 行） |
| 均值 | 148.904 s | 148.721 s | **149.484 s** |

- 实际墙钟（4 分片并行、片内两侧并行，共 8 个进程）：第一段（72 份）`15:51:23Z → 16:45:36Z` 计 **54m13s**（由先前进程完成）；本卡负责的 `--resume` 续跑段（36 份）`17:19:50Z → 17:35:03Z` 计 **15m13s**，在 60 分钟预算内；两段合计 **69m26s**。
- 吞吐：12,311,460 帧 / 16,144.31 s ≈ **762.6 帧/秒**（按用例墙钟、两侧并行计）。
- 全部用例均在 `--timeout 300` 上限内完成，未触发任何 124/137 超时码，重试次数 0。

## 构建与源树身份（只用两个独立目录）

| 变体 | 可执行文件 | SHA-256 | 字节 | `ReplayTraceRecord` 符号 |
|---|---|---|---|---|
| **ON**（harness） | `third_party/th08-rl/build/rl_retrace/th08-modern` | **`c2ce960dff8cf6d5eee4062c7c88cbad3c00c42dc4ed9c32d3af240819c6f79d`** | 4,807,664 | 存在 |
| **OFF**（原样 portable） | `third_party/th08-rl/build/portable-retrace/th08-modern` | **`f920e14004fb065955d9baae39cddfa575c78bcf90482f862b700dcf153bd2f8`** | 4,629,432 | 存在 |

- 两个 SHA-256 与 `EXP-P0-051` 记录的二进制**逐字节相同**，说明本轮**没有重建任何构建**，证据可追溯到同一对二进制。
- 源树：`third_party/th08-rl` HEAD `f563df58c44037b44208c393a86540b434b0c83b`，tracked 工作树 dirty=true；记录器源码 `src/modern/linux/replay_trace.cpp` SHA-256 `78cd4fe3834027cdd61014cff92f2ff998c33c490345cc6f93b4c27d92989398`。
- corpus manifest `third_party/th08/config/replay-fixtures.json` SHA-256 `0aaef96d389bf85ad1c77734ddcddbf1a2d80d7119b86f9f1a892ebfec959eaf`；accepted 目录 108 份 `.rpy`，全部与 manifest 的 `sha256` 逐一核对通过（0 缺失、0 多余、0 不符，工具 fail-closed）。
- 只读游戏数据 `/root/v400-data/game`：`th08.dat` `9d7edf43b8ddd347cbb641836f6b5050745dd936f688daebbf9382ca557043bb`（46,838,025 B）、`th08.cfg` `48bf7621da5879ef5ec1392fcac2cc0a85c313912a24fb33b4cddccf0d28f07a`（60 B）、`score.dat` `60e49abc8f0a955aa8fa883769a30c66913417be453632c9a8a99b0b8a5c5191`（17,773 B）；每份每侧均新建 runtime 目录，以符号链接只读挂入。
- **未使用 `build/rl` 与 `build/portable-replay`，未重建 `build/rl`**；muted 配置 SHA-256 `6401f8aa0a8ad7c0a2b4fc08174d7b6b9c2d8f58c110ee8329b397fdb7b18a52`（60 B，与上游 `muted_config()` 逐字节相同）。

## 运行参数

```text
xvfb-run -a -s "-screen 0 1024x768x24" timeout --signal=TERM --kill-after=5s 300s <exe> --data-dir <runtime>
```

```text
SDL_AUDIODRIVER=dummy LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=0 LP_NO_RAST=true
TH08_AUTOPLAY_REPLAY=./replay/audit.rpy  TH08_AUTOPLAY_STAGE=<该 replay 首个录制面>
TH08_RENDER_AUDIT_FAST=1  TH08_RENDER_AUDIT=0
TH08_REPLAY_TRACE_RECORD_ONLY=1
TH08_REPLAY_TRACE_ROWS=<runtime>/rows.bin
TH08_REPLAY_TRACE_COMPLETION=<case>/<side>/complete.json
（不设置 TH08_REPLAY_TRACE_REFERENCE / TH08_REPLAY_TRACE_MAXIMUM）
```

单份上限 `--timeout 300`、`--retries 1`、`--jobs 1`、`--parallel-sides`（同份 ON/OFF 并行），两侧参数完全一致。

## 执行记录（命令与退出码）

1. **残留进程清理**：`wsl -d Ubuntu -e bash -lc 'ps -eo pid,args | grep th08-modern'` → **0 命中**（唯一含 `th08`/`replay` 的行是 grep 自身命令行），故**未结束任何进程**，PID 列表为空。二次确认：`ps -eo pid,args` 全表 27 行，无 `th08-modern`、无 `--data-dir` 运行进程，驱动进程确已退出。
2. **续跑**（4 条命令并行，`N`=1..4）：

```bash
wsl -d Ubuntu -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && python3 /mnt/d/WorkShop/Touhou_RL/tools/rl/v400_replay_retrace.py \
  --on-build /root/Touhou_RL/third_party/th08-rl/build/rl_retrace/th08-modern \
  --off-build /root/Touhou_RL/third_party/th08-rl/build/portable-retrace/th08-modern \
  --all --shard N/4 --resume --timeout 300 --retries 1 --jobs 1 --parallel-sides \
  --output-dir /mnt/d/WorkShop/Touhou_RL/.tmp_v400_replay_retrace_108 --expected-total 108'
```

| 分片 | 续跑前 journal | `pending` | 续跑后 journal | 进程退出码 |
|---|---|---|---|---|
| 1/4 | 18/27 | 9/27 | **27/27** | **0**（该片最后完成，落盘报告 `allIdentical=true`） |
| 2/4 | 18/27 | 9/27 | **27/27** | **3**（先完成时其余片未跑完，脚本按设计返回 3=partial） |
| 3/4 | 18/27 | 9/27 | **27/27** | **3**（同上，先于 shard 4 完成） |
| 4/4 | 18/27 | 9/27 | **27/27** | **3**（同上） |

> 退出码 3 **不是失败**：工具在每个分片结束时都会把当时可见的全部 journal 合并写 `report.json`，先完成的分片看到的是未齐的 108，故按 `partial` 返回 3；四片自身的 27 份全部 `identical`、`attempts=1`、exit 0。最终一次的完整报告由第 3 步的 `--merge` 生成。

3. **合并**：

```bash
wsl -d Ubuntu -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && python3 tools/rl/v400_replay_retrace.py --merge \
  --on-build  /root/Touhou_RL/third_party/th08-rl/build/rl_retrace/th08-modern \
  --off-build /root/Touhou_RL/third_party/th08-rl/build/portable-retrace/th08-modern \
  --output-dir /mnt/d/WorkShop/Touhou_RL/.tmp_v400_replay_retrace_108 --expected-total 108'
# 输出 total=108 recorded=108 compared=108 identical=108 different=0 timeout=0 failed=0 partial=false remaining=0
# MERGE_EXIT=0
```

4. **只读复核脚本**（均放在证据目录内，不改动工具）：`summarize_108.py` → `SUMMARIZE_EXIT=0`；`appendix_108.py` → `APPENDIX_EXIT=0`；`crosscheck_108.py` → `CROSSCHECK_EXIT=0`；`verify_sides_108.py` → `SIDES_EXIT=0`；`timing_108.py` → `TIMING_EXIT=0`。

## 证明力与边界（必须读）

1. **差分，不是零售位一致**：`proofStrength=differential (not retail bit-exact)`。两侧共享同一份记录器源码与同一读点/字段定义；差分**只能**发现「构建差异导致的回放行为差异」，**不能**发现记录器自身对状态读取的偏差（读点、字段语义、边界）。
2. **不能替代 G0.1**：WP1/G0.1 要求「108 份回放与零售轨迹逐位一致」，零售参考轨迹仍为 **0/108**；本卡是退路 C 的全量证据，门报告须标注证明力下降，不得据此翻转 G0.1。
3. **行数对齐 ≠ 轨迹比对**：行数与 manifest `calculationFrames` 一致只说明计算帧计数一致；retail trace 是外部观察器的 22 字段 JSONL，与本记录器 25 字段定长行格式不同，SHA 不可互比。
4. **未验证 paced（非快进）模式**：全部使用 fast clock（`TH08_RENDER_AUDIT_FAST=1` + `LP_NO_RAST=true`）；本模式会给 `g_GameManager.replayMode` 置 2（上游原有行为，两侧相同）。
5. **观测面有限**：25 字段只覆盖分数、资源、位置位模式、RNG、输入、计时器等内部状态，不含渲染、音频、内存布局或其他时序观测。
6. **并发与耗时相关**：4 分片 × 2 侧同时运行（8 进程）下的单份耗时会高于单进程，本节耗时数据只对应本并发配置。
7. **记录瑕疵（不影响结论）**：`--merge` 写出的 `report.json.runConfig` 记录的是 merge 调用参数（`parallelSides=false`、`shard=null`），非分片运行参数；真实运行参数为 `--resume --timeout 300 --retries 1 --jobs 1 --parallel-sides --shard N/4`，逐份证据见 `shard-N.resume.log`、每份 `on|off/side.json`、以及 `report.json` 中每份 case 的 `shard` 字段。另：`--resume` 的提示行 `resume: 0 of 18 journaled cases skipped` 计数口径有瑕疵（实际跳过 18 份，紧随其后的 `pending 9 of 27` 正确）；为避免在运行中变更工具，本轮未改该文案，续跑语义本身正确（18 份未重跑，耗时与 journal 增长均为 9/片）。

## 证据与复现

- 合并报告：[`report.json`](../../../../../.tmp_v400_replay_retrace_108/report.json)（304,795 B，`version=2`，含 108 份 case 明细、4 分片汇总、构建身份、源树身份、游戏数据 SHA、`proofStrength`）。
- 汇总与复核产物（同目录）：`summary_108.json`、`cases_108.tsv`（108 行逐份明细）、`appendix_table.md`。
- 分片 journal：`.tmp_v400_replay_retrace_108/shard-{1,2,3,4}.jsonl`（各 27 行）；续跑日志：`shard-{1,2,3,4}.resume.log`；首段日志（18 份阶段）：`shard-{1,2,3,4}.log`（保留未改）。
- 逐份原始证据：`.tmp_v400_replay_retrace_108/<case>/{on,off}/{run.log,complete.json,result.json,command.json,side.json,runtime/rows.bin}`。
- 工具：[`v400_replay_retrace.py`](../../../../../tools/rl/v400_replay_retrace.py)（**本轮未修改**；既有单测 [`test_v400_replay_retrace.py`](../../../../../tests/test_v400_replay_retrace.py)）。

```powershell
wsl -d Ubuntu -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && python3 tools/rl/v400_replay_retrace.py --progress \
  --on-build /root/Touhou_RL/third_party/th08-rl/build/rl_retrace/th08-modern \
  --off-build /root/Touhou_RL/third_party/th08-rl/build/portable-retrace/th08-modern \
  --output-dir /mnt/d/WorkShop/Touhou_RL/.tmp_v400_replay_retrace_108 --expected-total 108'
# {"total":108,"recorded":108,"compared":108,"identical":108,"different":0,"timeout":0,"failed":0,"partial":false}
```

## 剩余阻塞

1. **零售位一致仍未达成（0/108）**：retail capture 的 0 帧失败（`EXP-P0-014/037/045`）未解除；本卡只提供退路 C 的全量差分证据。
2. **超时上限余量**：最大 fixture（149,986 行）单侧耗时 269.6–270.8 s，贴近 300 s 上限；若日后并发更高或机器更慢，需提高上限或单独重跑该份（本轮全部一次通过，未触发）。
3. **记录器语义未独立验证**：只有取得 retail `rows.jsonl`+`complete.json` 后，才能用同一记录器的 `TH08_REPLAY_TRACE_REFERENCE` 模式做真正的零售位比对。

## 附录：逐份明细（108/108，两侧逐字节一致）

行数/侧即两侧共同行数（`on.rows == off.rows`）；两侧 rows SHA-256 前 16 位相同、全部 64 位完全相同；「两侧相同」同时覆盖 rows 文件与 `complete.json`（108/108 全 ✅）。秒为单侧进程实际耗时，两侧并行执行。完整 64 位 SHA-256 见 `report.json` / `cases_108.tsv`。

| 分片 | 回放 | 难度 | 行数/侧 | ON rows SHA-256（前16位） | OFF rows SHA-256（前16位） | 两侧相同 | ON 秒 | OFF 秒 |
|---|---|---|---|---|---|---|---|---|
| 1 | `th8_ud2c0a.rpy` | Easy | 126,803 | `5d706de2d9d28480…` | `5d706de2d9d28480…` | ✅ | 156.4 | 154.3 |
| 1 | `th8_ud2b76.rpy` | Easy | 98,098 | `07cc4a75434a3bd6…` | `07cc4a75434a3bd6…` | ✅ | 104.3 | 104.6 |
| 1 | `th8_ud2bb6.rpy` | Easy | 122,814 | `78160da511805a14…` | `78160da511805a14…` | ✅ | 146.1 | 146.3 |
| 1 | `th8_ud2b7b.rpy` | Easy | 109,495 | `28aae91b54307783…` | `28aae91b54307783…` | ✅ | 123.0 | 119.4 |
| 1 | `th8_ud2603.rpy` | Normal | 129,300 | `835124791cbf393c…` | `835124791cbf393c…` | ✅ | 160.1 | 161.1 |
| 1 | `th8_ud2919.rpy` | Hard | 131,706 | `949a827c7ef3ab39…` | `949a827c7ef3ab39…` | ✅ | 161.1 | 160.7 |
| 1 | `th8_ud267e.rpy` | Hard | 125,026 | `aaaf4b141f8afed2…` | `aaaf4b141f8afed2…` | ✅ | 169.9 | 171.2 |
| 1 | `th8_ud2ac5.rpy` | Hard | 134,771 | `4dd86dca1ad6bc14…` | `4dd86dca1ad6bc14…` | ✅ | 187.4 | 188.4 |
| 1 | `th8_ud2cd4.rpy` | Lunatic | 149,986 | `223e77dbe8e0ccb3…` | `223e77dbe8e0ccb3…` | ✅ | 270.8 | 269.6 |
| 1 | `th8_ud2cac.rpy` | Lunatic | 113,660 | `daafe6db21fd7ca3…` | `daafe6db21fd7ca3…` | ✅ | 149.8 | 148.1 |
| 1 | `th8_ud2ce4.rpy` | Extra | 77,082 | `9c084edd94164b6d…` | `9c084edd94164b6d…` | ✅ | 120.9 | 121.2 |
| 1 | `th8_ud2cca.rpy` | Extra | 48,567 | `8e49884afdb36863…` | `8e49884afdb36863…` | ✅ | 67.0 | 67.1 |
| 1 | `th8_ud2c7a.rpy` | Extra | 44,771 | `1314414ad47471f6…` | `1314414ad47471f6…` | ✅ | 40.6 | 40.5 |
| 1 | `th8_ud2b75.rpy` | Easy | 106,420 | `361a07fff1f33d12…` | `361a07fff1f33d12…` | ✅ | 121.4 | 120.9 |
| 1 | `th8_ud2b92.rpy` | Normal | 128,279 | `70b94b1b68abcc4d…` | `70b94b1b68abcc4d…` | ✅ | 167.8 | 166.6 |
| 1 | `th8_ud2bd4.rpy` | Normal | 119,049 | `206abd7d896705b3…` | `206abd7d896705b3…` | ✅ | 154.4 | 153.2 |
| 1 | `th8_ud21f9.rpy` | Normal | 112,101 | `e3620fe24fa02be3…` | `e3620fe24fa02be3…` | ✅ | 144.9 | 147.1 |
| 1 | `th8_ud28a0.rpy` | Hard | 101,936 | `187c36762badfb14…` | `187c36762badfb14…` | ✅ | 115.9 | 115.6 |
| 1 | `th8_ud27df.rpy` | Hard | 117,783 | `0081a9724f156ec5…` | `0081a9724f156ec5…` | ✅ | 141.7 | 141.6 |
| 1 | `th8_ud2848.rpy` | Hard | 132,772 | `834cd5c3cee25bef…` | `834cd5c3cee25bef…` | ✅ | 169.7 | 168.6 |
| 1 | `th8_ud262e.rpy` | Hard | 140,962 | `50990192c5be4bf5…` | `50990192c5be4bf5…` | ✅ | 199.4 | 196.6 |
| 1 | `th8_ud286e.rpy` | Hard | 108,075 | `0c3ecd6a600e444d…` | `0c3ecd6a600e444d…` | ✅ | 133.2 | 134.0 |
| 1 | `th8_ud248d.rpy` | Hard | 127,290 | `4972fd03c5c3b3f0…` | `4972fd03c5c3b3f0…` | ✅ | 166.1 | 166.6 |
| 1 | `th8_ud2bca.rpy` | Lunatic | 114,724 | `3b42c232aaa68cd7…` | `3b42c232aaa68cd7…` | ✅ | 147.4 | 147.4 |
| 1 | `th8_ud2cd6.rpy` | Lunatic | 130,348 | `fa8260b511eab89d…` | `fa8260b511eab89d…` | ✅ | 169.4 | 167.8 |
| 1 | `th8_ud2c66.rpy` | Lunatic | 120,771 | `869302f5e86bf74d…` | `869302f5e86bf74d…` | ✅ | 170.0 | 171.1 |
| 1 | `th8_ud2ca6.rpy` | Lunatic | 147,168 | `321dff0fb3327c92…` | `321dff0fb3327c92…` | ✅ | 213.2 | 213.6 |
| 2 | `th8_ud2b78.rpy` | Easy | 118,416 | `5b18666cb87c5f0b…` | `5b18666cb87c5f0b…` | ✅ | 144.8 | 145.4 |
| 2 | `th8_ud2c0e.rpy` | Easy | 132,629 | `3b6fade83bde0024…` | `3b6fade83bde0024…` | ✅ | 147.2 | 146.6 |
| 2 | `th8_ud2cb3.rpy` | Easy | 129,211 | `ddd2e8eafadd1761…` | `ddd2e8eafadd1761…` | ✅ | 155.0 | 154.1 |
| 2 | `th8_ud2cc3.rpy` | Easy | 117,698 | `e3ae8342cbfb292b…` | `e3ae8342cbfb292b…` | ✅ | 137.1 | 136.8 |
| 2 | `th8_ud2cd5.rpy` | Normal | 100,979 | `a56c59af475b2f5f…` | `a56c59af475b2f5f…` | ✅ | 113.2 | 113.7 |
| 2 | `th8_ud2c88.rpy` | Normal | 123,844 | `8147be7a2dbc2413…` | `8147be7a2dbc2413…` | ✅ | 164.9 | 164.3 |
| 2 | `th8_ud279f.rpy` | Normal | 126,201 | `be0f348d19196ebd…` | `be0f348d19196ebd…` | ✅ | 158.1 | 156.4 |
| 2 | `th8_ud2b9c.rpy` | Normal | 122,716 | `d59081c356b929ca…` | `d59081c356b929ca…` | ✅ | 159.9 | 159.1 |
| 2 | `th8_ud2933.rpy` | Hard | 139,702 | `2c2433e5365940a5…` | `2c2433e5365940a5…` | ✅ | 194.0 | 191.5 |
| 2 | `th8_ud2ce5.rpy` | Lunatic | 113,221 | `26b59c8a96b3aecb…` | `26b59c8a96b3aecb…` | ✅ | 148.8 | 146.3 |
| 2 | `th8_ud2c70.rpy` | Lunatic | 106,063 | `e81b586fbc34a3fa…` | `e81b586fbc34a3fa…` | ✅ | 138.3 | 136.8 |
| 2 | `th8_ud2b8e.rpy` | Lunatic | 120,438 | `8d680fd05d35eed8…` | `8d680fd05d35eed8…` | ✅ | 165.9 | 166.5 |
| 2 | `th8_ud2c7b.rpy` | Lunatic | 145,103 | `4a7f1c3744939f38…` | `4a7f1c3744939f38…` | ✅ | 228.6 | 229.1 |
| 2 | `th8_ud2c0b.rpy` | Extra | 48,268 | `72af3f09d2b6443e…` | `72af3f09d2b6443e…` | ✅ | 66.5 | 66.3 |
| 2 | `th8_ud2cdf.rpy` | Extra | 54,772 | `3fe0d95e3b509702…` | `3fe0d95e3b509702…` | ✅ | 81.5 | 80.7 |
| 2 | `th8_ud2c47.rpy` | Extra | 44,767 | `65f4f31b3103275a…` | `65f4f31b3103275a…` | ✅ | 55.9 | 55.4 |
| 2 | `th8_ud2b6c.rpy` | Easy | 96,825 | `d241bb442bb1c7f7…` | `d241bb442bb1c7f7…` | ✅ | 104.9 | 105.0 |
| 2 | `th8_ud2cc4.rpy` | Easy | 131,458 | `7543db6da3151eba…` | `7543db6da3151eba…` | ✅ | 150.8 | 150.9 |
| 2 | `th8_ud2b70.rpy` | Easy | 107,722 | `71c4b34e55780c65…` | `71c4b34e55780c65…` | ✅ | 121.5 | 121.8 |
| 2 | `th8_ud2b3c.rpy` | Normal | 127,972 | `83ae497695d4fd63…` | `83ae497695d4fd63…` | ✅ | 162.0 | 161.1 |
| 2 | `th8_ud2ab0.rpy` | Normal | 134,376 | `51bf27167eb458de…` | `51bf27167eb458de…` | ✅ | 178.8 | 178.6 |
| 2 | `th8_ud2879.rpy` | Normal | 130,148 | `253e2a15b2d3d83e…` | `253e2a15b2d3d83e…` | ✅ | 155.0 | 156.0 |
| 2 | `th8_ud2c30.rpy` | Hard | 109,488 | `4bddf91b602ce155…` | `4bddf91b602ce155…` | ✅ | 133.1 | 131.7 |
| 2 | `th8_ud238c.rpy` | Hard | 111,944 | `488ceafb49427e95…` | `488ceafb49427e95…` | ✅ | 146.3 | 147.8 |
| 2 | `th8_ud2ccf.rpy` | Lunatic | 113,882 | `e90500c7054def98…` | `e90500c7054def98…` | ✅ | 145.0 | 146.2 |
| 2 | `th8_ud2cc8.rpy` | Lunatic | 127,286 | `971d6b0852984a6f…` | `971d6b0852984a6f…` | ✅ | 181.3 | 181.0 |
| 2 | `th8_ud2c8b.rpy` | Lunatic | 148,988 | `1d17bb4af3c0920f…` | `1d17bb4af3c0920f…` | ✅ | 266.4 | 267.7 |
| 3 | `th8_ud2cdc.rpy` | Easy | 107,296 | `bbd82ee6fbacea87…` | `bbd82ee6fbacea87…` | ✅ | 118.1 | 117.7 |
| 3 | `th8_ud2cde.rpy` | Easy | 103,875 | `f104b5ec6b7fa3c3…` | `f104b5ec6b7fa3c3…` | ✅ | 115.0 | 114.9 |
| 3 | `th8_ud1051.rpy` | Easy | 134,146 | `2e2a902bef23a1ab…` | `2e2a902bef23a1ab…` | ✅ | 171.1 | 170.2 |
| 3 | `th8_ud2bff.rpy` | Easy | 127,605 | `2335a4de9f86c9a7…` | `2335a4de9f86c9a7…` | ✅ | 153.9 | 153.2 |
| 3 | `th8_ud2ccc.rpy` | Normal | 113,791 | `618e0c650d4f4553…` | `618e0c650d4f4553…` | ✅ | 134.6 | 134.3 |
| 3 | `th8_ud2c2c.rpy` | Normal | 148,719 | `6439770c61f16551…` | `6439770c61f16551…` | ✅ | 251.0 | 250.4 |
| 3 | `th8_ud22fe.rpy` | Normal | 110,715 | `11e52a900cca38a7…` | `11e52a900cca38a7…` | ✅ | 145.6 | 145.1 |
| 3 | `th8_ud23d6.rpy` | Normal | 129,556 | `462df8efebd5da71…` | `462df8efebd5da71…` | ✅ | 162.5 | 162.0 |
| 3 | `th8_ud2a62.rpy` | Hard | 118,338 | `6d0049693536f1d5…` | `6d0049693536f1d5…` | ✅ | 158.1 | 157.3 |
| 3 | `th8_ud28b8.rpy` | Hard | 126,126 | `de5d46675c2dbbad…` | `de5d46675c2dbbad…` | ✅ | 166.4 | 166.5 |
| 3 | `th8_ud2227.rpy` | Hard | 120,173 | `45473f14bb9cb9bd…` | `45473f14bb9cb9bd…` | ✅ | 146.6 | 147.2 |
| 3 | `th8_ud24f9.rpy` | Hard | 143,785 | `0e4d409eae4e0641…` | `0e4d409eae4e0641…` | ✅ | 214.3 | 216.8 |
| 3 | `th8_ud2854.rpy` | Hard | 121,434 | `64c049f70ef32818…` | `64c049f70ef32818…` | ✅ | 158.5 | 159.2 |
| 3 | `th8_ud2bf5.rpy` | Lunatic | 137,112 | `44d27d809ada7ab3…` | `44d27d809ada7ab3…` | ✅ | 194.6 | 195.9 |
| 3 | `th8_ud2c82.rpy` | Lunatic | 127,016 | `5f447316dce1a724…` | `5f447316dce1a724…` | ✅ | 172.8 | 171.0 |
| 3 | `th8_ud2945.rpy` | Lunatic | 132,477 | `21e7b6b95f27cab0…` | `21e7b6b95f27cab0…` | ✅ | 181.1 | 181.4 |
| 3 | `th8_ud2c65.rpy` | Extra | 44,554 | `e322778b6e093b9f…` | `e322778b6e093b9f…` | ✅ | 62.5 | 62.2 |
| 3 | `th8_ud2cbc.rpy` | Extra | 51,084 | `37dbe8691b4f5b71…` | `37dbe8691b4f5b71…` | ✅ | 80.4 | 78.3 |
| 3 | `th8_ud2cc0.rpy` | Extra | 47,173 | `c6f8b0450701c353…` | `c6f8b0450701c353…` | ✅ | 69.2 | 69.4 |
| 3 | `th8_ud2cd2.rpy` | Easy | 99,113 | `e64ec6beec235ebc…` | `e64ec6beec235ebc…` | ✅ | 103.2 | 102.7 |
| 3 | `th8_ud2b72.rpy` | Easy | 123,487 | `57c1cedd012eaff8…` | `57c1cedd012eaff8…` | ✅ | 153.2 | 153.3 |
| 3 | `th8_ud2b74.rpy` | Easy | 95,280 | `dd1955bef658161b…` | `dd1955bef658161b…` | ✅ | 104.7 | 106.5 |
| 3 | `th8_ud2b86.rpy` | Normal | 128,399 | `3bc60ffdfab0e2e8…` | `3bc60ffdfab0e2e8…` | ✅ | 162.7 | 162.0 |
| 3 | `th8_ud2b1d.rpy` | Hard | 117,598 | `d079afc3fff3cc53…` | `d079afc3fff3cc53…` | ✅ | 144.7 | 144.4 |
| 3 | `th8_ud239d.rpy` | Hard | 108,947 | `4fa30408f9159529…` | `4fa30408f9159529…` | ✅ | 138.3 | 139.5 |
| 3 | `th8_ud2cce.rpy` | Lunatic | 112,876 | `7fad16a500a33ef9…` | `7fad16a500a33ef9…` | ✅ | 154.9 | 155.1 |
| 3 | `th8_ud2c9d.rpy` | Lunatic | 130,506 | `24bb470e01afc438…` | `24bb470e01afc438…` | ✅ | 174.3 | 176.2 |
| 4 | `th8_ud2b32.rpy` | Normal | 132,235 | `c9a36daf3ef9faf7…` | `c9a36daf3ef9faf7…` | ✅ | 162.9 | 163.4 |
| 4 | `th8_ud2b49.rpy` | Normal | 134,128 | `49a7d21960951ab1…` | `49a7d21960951ab1…` | ✅ | 182.3 | 182.1 |
| 4 | `th8_ud2922.rpy` | Normal | 135,726 | `67b5e418f8e44d93…` | `67b5e418f8e44d93…` | ✅ | 187.4 | 186.2 |
| 4 | `th8_ud2c44.rpy` | Hard | 122,914 | `50b72462b9b88e77…` | `50b72462b9b88e77…` | ✅ | 163.8 | 162.7 |
| 4 | `th8_ud2abc.rpy` | Hard | 130,503 | `e6cc3a3eb190ce8f…` | `e6cc3a3eb190ce8f…` | ✅ | 168.8 | 168.7 |
| 4 | `th8_ud2469.rpy` | Hard | 129,463 | `46fa8eda8671e28a…` | `46fa8eda8671e28a…` | ✅ | 170.7 | 171.0 |
| 4 | `th8_ud2c5a.rpy` | Lunatic | 112,701 | `aef195b5ce7a2497…` | `aef195b5ce7a2497…` | ✅ | 148.5 | 148.2 |
| 4 | `th8_ud2cc2.rpy` | Lunatic | 113,666 | `701b346ddb488cb8…` | `701b346ddb488cb8…` | ✅ | 147.5 | 147.2 |
| 4 | `th8_ud2cc9.rpy` | Lunatic | 143,679 | `368fa8194f68c433…` | `368fa8194f68c433…` | ✅ | 226.8 | 227.0 |
| 4 | `th8_ud2cd8.rpy` | Extra | 49,029 | `8e812c6d2c87afd6…` | `8e812c6d2c87afd6…` | ✅ | 68.1 | 67.5 |
| 4 | `th8_ud2c67.rpy` | Extra | 44,106 | `8dace2addc5a578f…` | `8dace2addc5a578f…` | ✅ | 62.2 | 64.5 |
| 4 | `th8_ud2ce0.rpy` | Extra | 45,822 | `a85639a09860d292…` | `a85639a09860d292…` | ✅ | 68.4 | 69.0 |
| 4 | `th8_ud2c68.rpy` | Easy | 126,114 | `ebe6c29892e94729…` | `ebe6c29892e94729…` | ✅ | 153.5 | 152.5 |
| 4 | `th8_ud2c54.rpy` | Easy | 121,373 | `08f408a2c8bd15c3…` | `08f408a2c8bd15c3…` | ✅ | 149.9 | 149.5 |
| 4 | `th8_ud2b6f.rpy` | Easy | 98,897 | `6cd67fa47f30da7b…` | `6cd67fa47f30da7b…` | ✅ | 105.4 | 105.6 |
| 4 | `th8_ud2b71.rpy` | Easy | 106,778 | `c41e4834e8cd7fff…` | `c41e4834e8cd7fff…` | ✅ | 120.5 | 117.4 |
| 4 | `th8_ud2b73.rpy` | Easy | 93,040 | `13670b9ee60ae8e5…` | `13670b9ee60ae8e5…` | ✅ | 102.4 | 103.4 |
| 4 | `th8_ud2bd8.rpy` | Normal | 127,542 | `0a221ff10fc9c10e…` | `0a221ff10fc9c10e…` | ✅ | 169.3 | 171.0 |
| 4 | `th8_ud2cc5.rpy` | Normal | 109,987 | `98d07fa5748342e6…` | `98d07fa5748342e6…` | ✅ | 144.0 | 142.8 |
| 4 | `th8_ud2991.rpy` | Normal | 127,006 | `7746cb141d3705d0…` | `7746cb141d3705d0…` | ✅ | 152.4 | 152.6 |
| 4 | `th8_ud2c2d.rpy` | Normal | 148,475 | `c6489406481fd6fe…` | `c6489406481fd6fe…` | ✅ | 249.3 | 247.5 |
| 4 | `th8_ud22fb.rpy` | Normal | 108,555 | `28702aa0ca00c619…` | `28702aa0ca00c619…` | ✅ | 132.2 | 132.8 |
| 4 | `th8_ud239c.rpy` | Hard | 128,362 | `2020c4d6ac9bdcb3…` | `2020c4d6ac9bdcb3…` | ✅ | 167.7 | 167.9 |
| 4 | `th8_ud2212.rpy` | Hard | 118,134 | `55dcc6ccd4342b30…` | `55dcc6ccd4342b30…` | ✅ | 155.2 | 155.1 |
| 4 | `th8_ud2cd7.rpy` | Lunatic | 119,685 | `3591f931f7a5a006…` | `3591f931f7a5a006…` | ✅ | 153.3 | 154.8 |
| 4 | `th8_ud2c74.rpy` | Lunatic | 102,204 | `1ab05343eaf72723…` | `1ab05343eaf72723…` | ✅ | 137.0 | 135.8 |
| 4 | `th8_ud2c77.rpy` | Lunatic | 116,281 | `6539aac694efbd50…` | `6539aac694efbd50…` | ✅ | 162.8 | 162.9 |
