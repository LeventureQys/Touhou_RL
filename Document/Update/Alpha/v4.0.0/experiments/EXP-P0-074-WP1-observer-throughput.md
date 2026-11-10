# EXP-P0-074（v4.0.0 WP1）：零售 trace 观测器每行开销 profiling 与吞吐实验

- 日期：2026-11-09 11:20–12:00（OBSOPT 独占窗口，批跑已按 `STOP` 停在 fixture 边界）
- 目标：把 `tools/rl/v400_retail_trace_debugapi.py` 的**每行**开销压下来，缩短 108 份批跑墙钟
- 硬约束：行内容/字段/顺序、断点语义、读点一律不变；判据只认官方 `trace_sha256()`
- 权威基线：`.tmp_v400_trace_batch/easy-0-a/rows.jsonl`（99,113 行，canonical `81f535e3…1643c`）
- 被否决路线（未重试）：多实例并行（EXP-P0-073，加速比 1.00×）

## 1. 结论（TL;DR）

| 问题 | 实测答案 |
| --- | --- |
| 每行 ≈12 ms 花在哪 | **th08.exe 自己算这一帧就占 94.6–95.1%**；观测器 Python 侧只有 **234 µs（2.3%）**，其中 97.2% 的墙钟是 `WaitForDebugEvent` 等游戏算完 |
| 低风险优化做了哪些 | 读窗口合并（12→8 次 RPM/行）、ctypes 缓冲复用、`DEBUG_EVENT`/`WOW64_CONTEXT` 复用、1024 行批量写、rng-origin 日志去重 |
| 优化收益 | 观测器每行 **234.48 → 217.94 µs（−16.5 µs，−7.05%）**；RPM 调用 **12.02 → 8.02 次/行（−33%）** |
| 行/s 前后 | 同一 replay 同一前 20,000 行：**98.239 → 98.172 行/s（−0.07%）**，即**墙钟无变化**（噪声内） |
| 前缀 30,000 行是否逐字节一致 | **是**。5,287,192 B，sha256 `64a083f0…808cd1`，30/30 个 1,000 行分块全同，0 处差异 |
| ≥1.5× 是否达成 | **否，且物理上不可达**。游戏自身 CPU 已占 94.6–95.1%，即使把 Python 侧**归零**，上限也只有 **1.022×**（ETA 40.67 h → 39.80 h） |
| 是否需暂停批跑才能切换 | **不需要**。批跑每个 fixture 都新起一个 `python -u <observer>` 子进程，换文件在下一个 fixture 生效，journal 不受影响 |

一句话：**这 12 ms 不是观测器的开销，是游戏自己的帧成本**；观测器微优化实测有效（−7%）但被游戏 9.6 ms/帧完全淹没。要继续压缩墙钟，杠杆在“游戏渲染/算帧速率”，不在观测器。

## 2. 方法与证据位置

| 用途 | 文件 |
| --- | --- |
| 基线观测器（逐字节冻结的副本，`ROOT` 深度与原文件一致） | `.tmp_v400_obsopt/prof/observer_baseline.py`（sha256 `39BBAC6C…C58D996A`，与原文件同哈希） |
| 分段计时 harness（不改进程内代码，monkeypatch 热路径） | `.tmp_v400_obsopt/prof_wrapper.py` |
| 基线 profile（20,000 行，优化前） | `.tmp_v400_obsopt/profile_before.json` |
| 优化后 profile（30,000 行） | `.tmp_v400_obsopt/profile_after.json` |
| 前缀逐字节比对器 | `.tmp_v400_obsopt/compare_prefix.py` |
| 单位成本微基准（无需游戏） | `.tmp_v400_obsopt/microbench.json` |
| 运行脚本 / 输出 | `.tmp_v400_obsopt/run_window.ps1`、`r1_before/`、`r2_after/`、`r3_full/` |

计时口径：`time.perf_counter_ns()` 分段累加；**窗口从第 1 行开始**（`on_row` 首次触发时清空累加器），attach/导航/收尾不计入；`th08.exe` 自身 CPU 用 `GetProcessTimes` 在每行采样。每个分段都同时记录调用次数与累计 ns，因此“每行调用数”是实测而非估算。

profile 自身开销：每行多 2 次 `GetProcessTimes` + 约 20 次 `perf_counter_ns` ≈ 2–4 µs/行（<0.04%），且**前后两次都带同样开销**，所以前后对比是同类比较。

## 3. Profile：每行 10.18 ms 到底花在哪（优化前，19,999 行）

窗口：`th8_ud2cd2.rpy`（easy-0-a 同款 replay）第 1–20,000 行，机器独占，行窗口 203.7 s。

| 分段 | 次/行 | µs/行 | 占行墙钟 |
| --- | ---: | ---: | ---: |
| **`WaitForDebugEvent`（游戏在跑）** | 2.005 | **9,896.77** | **97.23%** |
| `handle_event`（调试事件处理 = 观测器总工作） | 2.005 | 226.66 | 2.23% |
| ├ `Wow64GetThreadContext` | 2.005 | 40.66 | 0.40% |
| ├ `Wow64SetThreadContext`（写 `EFLAGS.RF`） | 2.004 | 44.77 | 0.44% |
| ├ 事件分发/簿记残余 | — | 60.11 | 0.59% |
| ├ `TraceCollector._on_frame`（整条取行路径） | 1.000 | 81.14 | 0.80% |
| │ ├ `ReadProcessMemory`（12 次） | 12.016 | 54.00 | 0.53% |
| │ ├ `compose_row`（struct.unpack + 建 list） | 1.000 | 6.84 | 0.07% |
| │ ├ `json.dumps(row)+"\n"` | 1.000 | 9.65 | 0.09% |
| │ ├ `write(rows.jsonl)` / `flush()` | 1.000 / 0.017 | 2.28 / 0.89 | 0.03% |
| │ └ `TraceCollector.log` | 0.004 | 0.43 | 0.00% |
| `ContinueDebugEvent` | 2.005 | 7.82 | 0.08% |
| 驱动循环残余（status/时间戳/条件判断） | — | 35.7 | 0.35% |
| **行墙钟合计** | | **10,179.2** | 100% |
| *参照：th08.exe 自身 CPU（kernel+user）* | | *9,630* | *94.61%* |

逐读点（每行 12 次 RPM）：

| 读点 | 次/行 | µs/行 |
| --- | ---: | ---: |
| `replay_manager_ptr`（4 B） | 2.004 | 17.94 |
| `frame_u32_4B`（manager→frame） | 2.012 | 7.35 |
| `gm_block_0x220`（544 B） | 1.000 | 7.63 |
| `globals_block_0xA0`（160 B） | 1.000 | 3.61 |
| `player_tail_12B` | 1.000 | 3.12 |
| `rng_8B` / `input_2B` / `player_first_1B` / `gm_plus_30` / `globals_ptr` | 各 1 | 2.72–2.96 |

**关键结构事实**：一行需要 **2 个硬件断点命中**（`frame` 0x452490 + `stageStart` 0x439BC7），即 `2.005` 个调试事件/行、2 次 Get/SetThreadContext、2 次 `ReadProcessMemory` 只为判 stageStart（`replay_manager` + `manager->frame`）。baseline journal 亦印证：`frame 99113 + stageStart 99607`、`eventsHandled 198916`、`resumeFlagSet 198721`。

把它换算到任务里的 **12.1 ms/行**（easy-0-a 全 fixture 1200.7 s / 99,113 行，含 6 面后期更重的关卡）：观测器侧仍是 ~0.22 ms，占 **1.8%**；**≈11.9 ms 是游戏自己**。

## 4. 每项优化与实测收益

全部改动只碰“怎么读/怎么写/怎么复用对象”，不碰行格式、断点、读点。

| # | 改动 | 实施 | 实测收益 |
| --- | --- | --- | --- |
| 1 | **读窗口合并**：`GM+8`+`GM+0x30` → 一个 44 B 窗口；`GM+0x3DBA8`(0x220)+`RNG`+`INPUT` → 一个 1,150 B 窗口；`PLAYER[0]`+`PLAYER+0x2B4` → 一个 704 B 窗口 | `WINDOW_OFFSET/WINDOW_BYTES/RNG_IN_WINDOW/INPUT_IN_WINDOW/GM_HEAD_BYTES/PLAYER_BYTES` 全部**从读地址推导**（无字面量），并保留 2 次依赖读（`REPLAY_MANAGER`→`manager`） | RPM **12.02 → 8.02 次/行（−4.0）**；RPM 时间 **54.00 → 43.50 µs/行（−10.5 µs）** |
| 2 | **ctypes 缓冲复用**：`WindowReader` 按 size 复用 `create_string_buffer`，复用同一个 `c_size_t`，直接调 `ReadProcessMemory`（原型钉在本模块自己的 `WinDLL` 上） | `.tmp_v400_obsopt/microbench.json`：单次 RPM ≈1.7–1.8 µs，`create_string_buffer` 另计 ≈0.1 µs | 与 #1 合计反映在 RPM µs 上；单独贡献 <2 µs/行 |
| 3 | **行批量写**：内存里 `"\n".join` 1,024 行再写一次 | `ROW_BATCH=1024`、`write_pending()`；`flush()` 语义保留（含 pending 落盘） | `write` **2.28 → 0.09 µs/行**，`flush` **0.89 → 0**（去掉每 60 行一次 syscall） |
| 4 | **日志里程碑化**：`set_rng_origin` 只在取值变化时打印 | 原来 stage 切换/终局时每帧一条 `rng origin …`（日志里看到 30+ 条/0.5 s，每条 `print`+`write`+`flush`） | `collector_log` **0.43 → 0.01 µs/行**（只在过渡窗口有明显效果） |
| 5 | **`DEBUG_EVENT` / `WOW64_CONTEXT` 复用** | 每行 ~4 次结构体分配 → 0 | 单测通过，实测落在噪声内（get+set 85.4 → 86.3 µs） |
| 6 | `SetThreadContext`（手置 RF）能否“一次覆盖多次命中” | **不能**：`EFLAGS.RF` 只抑制**一次**调试异常，每次命中都要重设（baseline `resumeFlagSet=198721 ≈ 2×rows`、失败 0），因此逐次 Set 是语义必需 | 保留原样 |
| 合计 | | | 观测器 **234.48 → 217.94 µs/行（−16.5 µs，−7.05%）**；行墙钟 **10,179.2 → 10,161.4 µs（−0.17%，噪声内）** |

**安全性设计（不是“信任”而是“自检”）**：合并窗口在**第一帧断点内**（进程已冻结）做一次等价性自检——把 3 个合并窗口的每个切片与官方逐地址读法逐字节比较，7 项全等才启用合并路径，否则永久回退到逐地址读（`merged_error` 记录原因）。真实运行日志：`[46.187] trace: row read mode=merged`。

## 5. 优化前后 profile 对比（同 replay、同前 20k 内容）

| 指标 | 优化前（R1） | 优化后（R2） | Δ |
| --- | ---: | ---: | ---: |
| 行墙钟 | 10,179.2 µs | 10,161.4 µs | −17.8 µs（−0.17%） |
| **行/s（前 20,000 行）** | **98.239** | **98.172** | **−0.07%** |
| 观测器工作（handle+continue） | 234.48 µs | 217.94 µs | **−16.54 µs（−7.05%）** |
| ├ Get/SetThreadContext | 85.43 µs | 86.28 µs | +0.85（噪声） |
| ├ `ReadProcessMemory` | 54.00 µs / 12.02 次 | 43.50 µs / 8.02 次 | −10.5 µs / −4.0 次 |
| ├ `_on_frame` 整条取行路径 | 81.14 µs | 64.19 µs | −16.95 µs |
| ├ `stream_write` + `flush` | 3.17 µs | 0.09 µs | −3.08 µs |
| └ `collector_log` | 0.43 µs | 0.01 µs | −0.42 µs |
| th08.exe 自身 CPU | 9,630 µs | 9,667 µs | +0.4%（同内容，噪声） |

结论：**优化真实生效（观测器侧 −7%，RPM 次数 −33%），但被游戏自身 9.6 ms/帧淹没，端到端墙钟不变**（−0.07% 在噪声内）。

## 6. 正确性证据

### 6.1 前缀 30,000 行逐字节一致（硬证据）

```
python .tmp_v400_obsopt/compare_prefix.py \
    .tmp_v400_trace_batch/easy-0-a/rows.jsonl \
    .tmp_v400_obsopt/r2_after/rows.jsonl 30000
```

| 项 | 值 |
| --- | --- |
| 结论 | `identical: true` |
| 行数 | 30,000 / 30,000（两侧行数一致） |
| 字节数 | 5,287,192 B（两侧完全相等） |
| 前缀 sha256（基线） | `64a083f06327d20d9bdb07bb5821683715b9d0c2102c50185e3a56ee48808cd1` |
| 前缀 sha256（优化后） | `64a083f06327d20d9bdb07bb5821683715b9d0c2102c50185e3a56ee48808cd1` |
| 1,000 行分块 | **30/30 全同**，`mismatch: {}` |

补充：**基线观测器（R1）前 20,000 行 vs 优化后（R2）前 20,000 行**也逐字节相同（sha `36e75213a82a9d1aaa1ed5dff669a59d`，20/20 分块），证明分段计时 harness 本身也不改行。

### 6.2 单元测试

`tests/test_v400_retail_trace_debugapi.py`：**37 passed**（原 33 + 新增 4）。

新增：
- `test_merged_row_windows_cover_exactly_the_official_read_points`：窗口常量从地址推导、覆盖 `GM+8..GM+0x34`、`RNG`/`INPUT`、`PLAYER..+0x2C0`
- `test_merged_row_path_is_byte_identical_to_per_address_reads`：同一宽内存镜像跑 200 行，合并路径 vs 逐地址路径 **rows.jsonl 逐字节相同**，且读次数实测 **10 次/行 → 6 次/行**
- `test_auto_row_read_mode_falls_back_without_a_process_handle`：无 handle 时自动回退、不重复探测
- `test_rows_are_joined_in_memory_and_written_in_batches`：缓冲期间文件 0 字节，`flush()` 后 5 行

### 6.3 完整短 fixture（硬判据 `trace_sha256()`）

`th8_ud2c65.rpy`（fixture `extra-2-extra`，44,554 计算帧，单面 stage 8），**优化后代码、无插桩、跑到自然结束**：

| 项 | 值 |
| --- | --- |
| `status` / `exitCode` | `ok` / **0** |
| 行数 vs manifest `calculationFrames` | **44,554 / 44,554**（`rowsMatch: true`） |
| **`canonicalTraceSha256`** | **`d48351bfe2d5e873a12ce956d03f6d35b420cab65ca134a687ba7b4c3c58765f`** |
| manifest 期望 | `d48351bf…58765f` → **`shaMatch: true`** |
| `ended` / `naturalReplayExit` | `true` / `true`（`deleteReplay` 断点命中 1 次，`errors: []`） |
| 阶段表 | `stages=[8]`、`stageFrames={'8': 44054}`、`rngOrigins={'8': 92}`、`endScores={'8': 188810015}` |
| `thirdPartyUnchanged` | `true`（`third_party/th08` 三个被监视文件前后哈希一致） |
| 断点命中 | `frame 44554 + stageStart 44636 + deleteReplay 1`（仍为 2 次/行，语义未变） |
| 墙钟 | 789.25 s（其中导航 47.7 s，行窗口 741.6 s → **60.1 行/s**，stage 8 比 easy 面重） |

即：**优化后的观测器在同一份 replay 上跑完全程，官方 `trace_sha256()` 与 manifest 完全相等**，并且 `rows.jsonl` 文件 sha 为 `3b30d73f…39d0c2`（可作为该 fixture 的完整基线）。

### 6.4 合并窗口在真实进程里被真正启用

三次真实运行的 `debugger.log` 都有 `trace: row read mode=merged`（自检通过），没有一次落到回退分支；`extra-2-extra` 的 44,554 行也全部由合并窗口产出，且 canonical sha 仍等于 manifest。



## 7. 行/s 与 ETA

口径：批跑 journal 的 `elapsed_s`（含启动/导航/收尾）与 profile 的行窗口分开列。

| 指标 | 优化前 | 优化后 |
| --- | ---: | ---: |
| 行/s（easy-0-a 全 fixture，journal 1200.718 s / 99,113 行，含启动） | 82.54 | 82.54（未变） |
| 行/s（easy-0-a 行窗口，日志 7.237→1192.318 s） | 83.63 | 83.57（×0.99932） |
| 行/s（本实验 20k 窗口，机器独占） | 98.239 | 98.172 |
| 每 fixture 固定开销（启动+导航+收尾） | 15.64 s | 15.64 s |
| **108 份剩余 ETA（12,105,051 帧 + 106×15.64 s）** | **146,396 s = 40.67 h** | **146,495 s = 40.69 h（+99 s）** |
| 理论上限（把观测器 Python 侧全部归零） | — | 143,292 s = **39.80 h（1.022×，省 0.87 h）** |

对照：批跑自身 heartbeat 的投影 `projectedRemainingSecondsByRows = 147,948 s`（用 82.54 行/s），与本表 40.67 h 同量级。

**≥1.5× 不成立，且不是实现问题**：要让行时间从 10.18 ms 降到 6.79 ms，需要游戏每帧从 9.63 ms 降到 6.2 ms；观测器侧就算全部消失也只有 1.022×。

## 8. 如何安全切换（是否需要暂停批跑）

**不需要暂停批跑，也不需要重启批跑进程。**

- 批跑 `run_one()` 每个 fixture 都 `subprocess.run([python, "-u", observer, ...])` 新起子进程；已运行的子进程持有的是**已加载的模块**，改磁盘文件不影响它。
- 因此在**任意 fixture 边界**（`rows.jsonl` 写完、journal 落盘之后、下一个 subprocess 启动之前）替换 `tools/rl/v400_retail_trace_debugapi.py` 即可，下一个 fixture 自动用新代码。
- journal 语义不变：`journal.jsonl` 每行仍是 `ok`/`mismatch` 终态，`latest_entries`/`terminal_entries` 逻辑没动；新代码写出的行与旧代码逐字节相同（§6.1），所以**同一批里混用两种代码也不会产生可比性缺口**。
- 原子替换建议：`Move-Item -Force` 覆盖（同目录 rename），避免半截文件被下一个 subprocess 读到。

本次窗口的操作顺序（已执行，可复用）：写 `.tmp_v400_trace_batch/STOP` → 等当前 fixture 正常结束（`batch.log`: `stop file present` → `session end ... exit=3`）→ 跑实验 → 删 STOP → 原命令重启：

```
python tools\rl\v400_retail_trace_batch.py --root .tmp_v400_trace_batch --first easy-0-a
```

（`easy-0-a`/`easy-0-b` 已是终态，重启会自动跳过；`--first` 只影响顺序，不影响 resume。）

**建议**：既然端到端收益是 0.0%，**不值得为它专门停批跑**；如果批跑已经因别的原因停着，换上也无害（少 4 次 RPM/行、少 1024 次/1000 行的 syscall，抖动更小）。

## 9. `th08.cfg` 渲染配置变量实验

动机：§3 已证明 94.6% 的行时间在 `th08.exe` 自己身上，唯一可能有量级的杠杆是“让游戏每帧算得更便宜”。TH08 的 `th08.cfg` 是 60 字节 `GameConfiguration`（`Supervisor.hpp:62-89`，见 `touhou_rl/config_file.py` 与 `tools/rl/v400_make_cfg.py`），**确实有渲染档位**：
`frameskipConfig`(0x23，0=full speed，界<3)、`effectQuality`(0x24，2=MAXIMUM，界<3)、以及 `GameConfigOpts`(0x38) 位：`DISPLAY_MINIMUM_GRAPHICS`(4)、`SUPPRESS_GOURAUD`(5)、`DISABLE_DEPTH_TEST`(6)、`DISABLE_COLOR_COMPOSITING`(8)、`DISABLE_FOG`(10)、`DISABLE_VSYNC`(14)。

方法：**只改 run-dir 里的副本**。`third_party/th08` 一个字节都没动；变体文件放在 `.tmp_v400_obsopt/cfg_minrender*/th08.cfg`，由 harness 在 `assemble_run_dir()` **之后**覆盖，并用 sha256 验证确实生效：

| 变体 | 文件 sha256 | frameskip / effectQuality / opts |
| --- | --- | --- |
| 模板（在用） `third_party/th08/build/play_fence/th08.cfg` | `48bf7621…0d28f07a` | 0 / 2 / `0x1` |
| min-render | `1cd32d23…f219837` | **2 / 0 / `0x571`**（bit0+4+5+6+8+10） |
| min-render + 静音 | `2d34daa7…17a58430` | 2 / 0 / `0x571` + `playSounds=0, musicMode=0(MIDI), vols=0` |

实测（同一 replay、easy-0-a 前 10,000 行窗口、同一台机器、独占）：

| 运行 | cfg | 行/s | th08 自身 CPU | 行是否与基线逐字节相同 |
| --- | --- | ---: | ---: | --- |
| R1（基线观测器） | 模板 | 98.24（前 20k 窗口） | 9.630 ms | — |
| R2（优化观测器） | 模板 | 98.17（前 20k 窗口） | 9.667 ms | **是**（30,000 行 vs 权威基线） |
| **R4（min-render）** | `0x571`/fs2/eff0 | **99.25** | 9.637 ms | **是**（前 9,999 行，sha `8c805542…`，10/10 分块） |
| **R5（min-render+静音）** | 同上+静音 | **98.95** | 9.649 ms | **是**（vs R4 前 9,999 行，10/10 分块） |

结论（负结果，但很有价值）：

1. **渲染档位不是瓶颈**：把画面质量压到最低 + 静音，游戏自身每帧 CPU 只从 9.667 → 9.637/9.649 ms（**−0.3%，噪声内**），行/s 的变化（+1.1%/−0.5%）落在同内容窗口的自然波动里。也就是说这 9.6 ms 不是纹理混合/雾/深度/混音花的，而是**核心算帧 + 呈现路径**花的。
2. **这些档位不改变 trace**：两个变体的行与基线/彼此**逐字节相同**（各 9,999 行、10 个分块全同）。即“改画面不改逻辑”这一假设成立——它们**安全但无用**。
3. **对像素的影响**：`SUPPRESS_GOURAUD`/`DISABLE_DEPTH_TEST`/`DISABLE_COLOR_COMPOSITING`/`DISABLE_FOG`/`effectQuality=0` 按定义就是会改变画面的；WP8 的像素/渲染确定性工作**不得**使用这些档位。G0.1（只看行）不受影响。
4. **canonical `trace_sha256()` 对 cfg 变体未跑完整 fixture**：本窗口预算不允许（extra-2-extra 一趟 ~13 min）。但“同一 replay 前 9,999 行逐字节相同”是比 sha 相等更强的逻辑不变性证据。若要补官方判据，命令是：
   ```
   $env:PROF_CFG='D:\WorkShop\Touhou_RL\.tmp_v400_obsopt\cfg_minrender\th08.cfg'
   $env:PROF_OBSERVER='D:\WorkShop\Touhou_RL\tools\rl\v400_retail_trace_debugapi.py'
   python -u .tmp_v400_obsopt\prof_wrapper.py .tmp_v400_obsopt\profile_cfg_full.json `
       --phase rows --out .tmp_v400_obsopt\r6_cfg_full `
       --replay third_party\th08\tests\fixtures\replays\th08-1.00d\accepted\th8_ud2c65.rpy `
       --budget-seconds 1700
   ```
   然后看 `r6_cfg_full/report.json` 的 `verdict.canonicalTraceSha256` 是否等于 `d48351bf…58765f`。



## 10. 风险与局限

1. **窗口代表性**：profile 用的是 easy-0-a 的**前 20,000 行**（第 0 面 + 第 1 面前段，10.18 ms/行）。全 fixture 平均是 12.1 ms/行（后期面更重），但结构比例不变（游戏 ≥94%）；行/s 的绝对值随内容波动很大（本实验观察到同一台机器上 36.8–140 行/s 的瞬时波动），**任何“行/s 提升”结论必须同内容窗口对比**——本文所有前后对比都是同一 replay 的前 20,000 行。
2. **profile 开销**：2–4 µs/行（<0.04%），前后同构。
3. **批量写的耐久性**：最多 1,024 行（~160 KiB）留在内存；进程被强杀会丢这 ≤1,024 行（`finish()`/超时路径会 `write_pending()` 落盘）。批跑侧 `count_rows()` 读书文件的进度会滞后 ≤1,024 行（~10 s），仅影响 heartbeat 显示。
4. **合并窗口依赖进程内存布局**：`GM+0x3DBA8..INPUT+2` 必须连续可读；不可读/不等价时自检会永久回退到逐地址读（日志 `row read mode=per-address (...)`），不会产出错行。
5. **未测**：像素/渲染确定性（本实验只看行）；`--allow-existing-instances` 并行路径（已否决）。

## 11. 后续（真正能压缩墙钟的方向，均需另行授权）

1. **渲染成本**（唯一量级足够的杠杆）：`th08.cfg` 的 `frameskipConfig`/`effectQuality`/`GameConfigOpts` 渲染位；见 §9 结论。
2. **fixture 级调度**：`extra-*` 单面 replay 默认参数下导航要 ~48 s（闭环保守超时→盲序列回退），12 份合计 ~8 min 一次性开销；按 fixture 传 `--start-stage 8` 可降到 ~7 s（历史 `.tmp_v400_trace_fmt1/full_extra2` 证据）。
3. **批跑投影口径**：heartbeat 用 82.54 行/s 单一均值外推 40 h，实际内容相关的 ±20% 波动远大于任何观测器优化，ETA 应给区间。
