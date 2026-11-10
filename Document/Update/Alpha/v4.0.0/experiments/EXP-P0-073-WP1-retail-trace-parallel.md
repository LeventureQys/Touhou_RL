# EXP-P0-073（v4.0.0 WP1）零售 trace 批跑并行化实验卡

- 日期：2026-11-09
- 负责人授权范围：`tools/rl/v400_retail_trace_batch.py`、`tools/rl/v400_retail_trace_debugapi.py`（仅多实例协同/日志）、`tests/test_v400_retail_trace_batch.py`、本卡、`.tmp_v400_trace_par*`
- 状态：**并行不成立（负结果）**；批跑已回退串行并重启

---

## 0. 结论（先看这里）

| 问题 | 结论 |
| --- | --- |
| 2 实例是否安全 | **否**。第二个零售实例无法完成初始化 |
| 4 实例是否安全 | **未做**（2 实例门已失败，禁止在有证据前扩大到 4） |
| 总吞吐 | 串行基线 82.54 行/s；双实例重叠窗口 **从未出现**，聚合吞吐 = 0 |
| 加速比 | **1.00x**（无收益） |
| 并行上限建议 | **1**（串行）；启用 >1 前必须换到能并发初始化的采集配置 |
| 批跑状态 | 已按**串行**重启，pid 19916，`easy-0-a` 正确跳过 |
| 108 份新预计末时间 | 2026-11-10 22:39（按 fixture 均值 35.7 h）～2026-11-11 04:04（按行数 41.1 h） |

**一句话**：``SendInput`` 焦点只用于菜单导航这个洞察本身成立（导航锁工作正常），但
**零售 TH08 是两个进程不能共存的独占显示目标**——第二个实例在初始化阶段就退出，
因此"串行导航 + 并行播放"这条路线在本机当前采集配置下不成立。

---

## 1. 串行批跑接管记录（如实记录）

按授权接管零售采集所有权，**终止了串行批跑**。所有 journal/证据均保留，未删除任何文件。

| 项 | 值 |
| --- | --- |
| 接管时驱动 pid | **22704**（其父 22040 为已退出的启动壳） |
| 该会话启动时间 | 2026-11-09T10:46:48（前一会话 pid 14884 于 10:46:17 中断后自动重启） |
| 其观察器子进程 | pid 15888 |
| 其游戏进程 | pid 22148（终止时已自行退出） |
| journal 行数 | **1**（终态：`easy-0-a` ok / `sha_equal=true` / rows=99113 / 1200.7 s） |
| 接管时 fixture | `easy-0-b` attempt-2，rows=9660（此前 attempt-1 中断于 17700 行） |
| 终止方式 | `Stop-Process` 22704、15888；确认 `th08.exe` 零残留 |
| STOP 文件 | 写出 `.tmp_v400_trace_batch/STOP` 防自动重启，重启前已删除 |

**已保留、未覆盖的事实**（沿用上一代理已验证结论，判据未改动）：

- `easy-0-a` = `th8_ud2cd2.rpy`，**99,113 行**，canonical `81f535e3…1643c` = manifest，1200.718 s；
- `stage_frames_equal` 已改用官方 tail 不变式（`verdict.stageTable[].tailOk` / `_stage_tails_ok`），
  **不是** 旧的 `sum(stageFrames)==rows`（清关重复命中使 rows 比 stageFrames 多 874）。本次改造
  完整保留该判据，`canonical SHA` 判定路径未触碰。

---

## 2. 方法

两份短 fixture（manifest 中 40k–50k 帧）：

| fixture | replay | 帧数 | manifest canonical sha256 |
| --- | --- | --- | --- |
| `extra-1-extra` | th8_ud2c67.rpy | 44,106 | `3c69ae01…30072` |
| `extra-2-extra` | th8_ud2c65.rpy | 44,554 | `d48351bf…58765f` |

观察器**已具备**多实例协同能力（本卡实验直接复用，未改断点语义/行格式）：

- `--allow-existing-instances`：解除"单实例"拒绝（默认仍拒绝，串行语义不变）；
- `--nav-lock PATH` + `--nav-lock-timeout`：`msvcrt.locking` 字节锁，只覆盖
  `force_foreground` → 菜单导航 → 回放开始的窗口，回放与断点采集不持锁；
- `<out>/status.json`：`phase / playbackStarted / playbackStartedAt / rows / foregroundOk /
  navLockWaitSeconds / navLockHeldSeconds / lastDebugEventAgeSeconds`。

驱动侧新增（向后兼容，`--parallel` 未启用，见 §6）：`wait_for_instance_limit(limit, …)`
（`limit=0` 即原 `wait_for_no_instances`）、`run_one(allow_existing=…, nav_lock=…, slot=…)`、
journal 新增 `playbackStartedAt / navPath / navLockWaitSeconds / navLockHeldSeconds /
parallelSlot / parallelAllowExisting` 审计字段。

实验台架：`.tmp_v400_trace_par2/harness.py`（一次性，位于授权临时目录），逐 5 s 采样
row 行数、`status.json`、以及观察器+游戏的 CPU 时间。

---

## 3. 实验一：同时启动（失败，证据 1）

`--tag p2`，budget 150 s，两实例**同时**启动，共享 nav lock。

| 实例 | 启动 | 窗口 | 线程 | score.dat | 结果 |
| --- | --- | --- | --- | --- | --- |
| `extra-1-extra` | 2.05 s 拿到窗口 | **640×480** | 15（14 live） | 有（280 B） | 2.60 s 拿到 nav lock → **卡死** |
| `extra-2-extra` | 2.25 s 拿到窗口 | **482×195** | 7（6 live） | **无** | nav lock 等 240 s **超时** |

现象：

1. `extra-1-extra` 状态永久停在 `phase=waiting-for-nav-lock`，`navLock.ok=true`，
   日志最后一行是 `nav-lock: acquired … after 0.00 s`，**之后 330 s 无任何日志**——
   卡在 `winapi.force_foreground()`（该函数为把窗口提到前台会
   `AttachThreadInput(当前线程, 当前前台窗口线程)`，而当时的前台窗口属于另一个被调试的
   零售进程），且**持锁不放**。
2. `extra-2-extra` 的游戏进程根本没有初始化成功：窗口 482×195（不是 640×480 游戏窗口）、
   仅 6 个存活线程（正常 14）、run dir 里没有 `score.dat`（正常实例会写 280 B）。
3. 两者 rows 均为 **0**，无一行 trace 产出；实验台架 330 s 后终止。
4. 结束后 `th08.exe` 零残留。

**结论 1**：同时启动时有两类冲突——(a) 前台窗口线程跨进程 `AttachThreadInput` 互锁
（症状：拿到 nav lock 后永久僵死）；(b) 第二个零售进程初始化失败（症状：482×195 对话框窗口）。

---

## 4. 实验二：串行导航 + 并行播放（失败，证据 2）

`--tag p2s`，budget 150 s，**stagger**：A 启动 → 等到 A `playbackStarted=true` → 才启动 B。
这正是"先串行导航 A → 串行导航 B → 两者同时播放"的目标形态，也规避了实验一的两类冲突。

| 实例 | 时间线 | 结果 |
| --- | --- | --- |
| `extra-1-extra` | 2.6 s 拿锁 → 47.7 s 回放开始（**nav lock 持锁 45.1 s**）→ 150 s budget 到 | **6212 行**（14.1%），exit 6（budget 用尽，非失败语义） |
| `extra-2-extra` | A 回放开始后才启动；2.17 s 拿到 482×195 窗口、6 live 线程 | 游戏 **4.7 s 自行退出（exit 0）**，**0 行**，exit 6 |

关键读数：

- `extra-1-extra` 播放窗口 (150.0 − 47.7) = **102.3 s 内 6212 行 = 60.7 行/s**
  （对比串行基线 82.54 行/s；本 fixture 为 stage 8 单独回放，行速率与 6 面 replay 不同量级，
  不宜直接当"劣化"证据，但**没有任何提速**）；
- `extra-2-extra` 的游戏进程在 4.7 s 内正常退出，`status.json` 无 `playbackStarted`；
- **两个实例从未同时处于 `playing`**（harness `firstSampleAllPlayingSeconds = null`），
  即聚合吞吐无法测量——不是"测得低"，而是**并行窗口根本不存在**；
- 结束后 `th08.exe` 零残留。

**结论 2**：即便把导航完全串行化、第二个实例只在第一个回放开始后才启动，
**第二个零售进程仍然无法初始化**（482×195 对话框窗口 → 6 线程 → 退出 0）。
最可能的机制是独占显示模式（DirectDraw 全屏）争用：第一个实例已持有显示目标，
第二个实例拿不到，弹框后退出。

补充观察：`extra-*` fixture 的导航耗时 **45.1 s**（非原先假设的 7–10 s），
因为要导航进 Extra Stage；即使并行成立，导航串行段也会是明显瓶颈。

---

## 5. 冲突现象汇总

| 现象 | 触发条件 | 证据 |
| --- | --- | --- |
| `force_foreground` 永久阻塞 | 另一实例窗口是前台 | p2：`extra-1-extra` 330 s 无日志、持 nav lock 不放 |
| 第二实例初始化失败 | 两零售进程共存 | p2/p2s：482×195 窗口、6 live 线程、无 `score.dat`，p2s 中 4.7 s 退出 0 |
| 窗口失焦 | 无（未观测到回放中失焦导致的停顿） | 两实例 `foregroundOk=true`；因并行窗口不存在，**未验证** |
| 调试权限冲突 | 未出现（`SeDebugPrivilege ok=False` 与串行时一致，非新增） | 两轮日志相同 |
| `ha` 类异常 | 未出现 | 两轮日志无 `ha`/断言类报错 |

---

## 6. 为什么没有继续做 4 实例 / `--parallel`

用户给定的步骤是**门控**的：`若 2 实例安全 → 扩到 4 实例`。2 实例门已有硬证据失败，
因此：

- **未做 4 实例实验**（做了只会重复同一失败，且违反"无证据就声称并行安全"红线）；
- **驱动未启用 `--parallel` 并行采集**：流水线调度器一旦打开，必然持续产出"第二个实例
  起不来"的失败条目，属于用代码掩盖负结果。已保留其**全部前置件**并在 `run_one` /
  `wait_for_instance_limit` 上做好了向后兼容并联调通过（27 项单测全绿），
  待采集配置能支持并发初始化后，只需在 main 循环加调度即可，不必重做底层。

已落地的可复用资产：观察器 `--nav-lock` / `--allow-existing-instances` / `status.json` 里程碑、
驱动的实例上限守卫与导航审计字段。这些在**单实例**下完全惰性，不改变现有串行行为。

---

## 7. 加速比与并行上限建议

- 串行基线：**82.54 行/s**（`easy-0-a`：99,113 行 / 1200.718 s）；
- 2 实例：并行重叠窗口 **不存在** ⇒ 聚合吞吐 0 行/s（重叠期），加速比 **1.00x**；
- 4 实例：未测；
- **并行上限建议：1**。在有新证据（例如容器/虚拟显示或窗口化采集配置被官方判据验证）
  之前，不要把批跑切到 N>1。

若要真正解锁并行，需要先解决"第二进程初始化失败"，候选方向（均未验证，且都会改变采集
配置、必须重新用 canonical `trace_sha256()` 验证）：

1. 窗口化/非独占显示配置（当前 `th08.cfg` 模板来自 `third_party/th08` 资产，**修改被禁止**；
   若要试，只能在实例私有 run dir 内改写并重新验证 canonical SHA）；
2. 第二台机器/第二个桌面会话（不同 `SessionId`），绕开独占显示与前台窗口竞争；
3. 放弃同机并行，改为**单实例流水线以外的加速**（例如缩短导航、复用 run dir）。

---

## 8. 回退方式

批跑已回退并重启为串行：

```powershell
# 1) 确认无残留
tasklist /FI "IMAGENAME eq th08.exe" /FO CSV /NH
# 2) 删除 STOP（若有）
Remove-Item .tmp_v400_trace_batch/STOP -ErrorAction SilentlyContinue
# 3) 串行重启（续跑语义不变：journal 终态自动跳过）
python -u tools/rl/v400_retail_trace_batch.py --root .tmp_v400_trace_batch --first easy-0-a
```

- 驱动 `--parallel` **未引入**，故无需回退命令行；`run_one` 新增参数默认 `allow_existing=0`
  = 原严格单实例语义；
- 观察器单实例行为不变（不加 `--nav-lock` / `--allow-existing-instances` 时与改造前等价）；
- 证据目录 `.tmp_v400_trace_par2/p2`、`p2s` 保留，供复核。

---

## 9. 重启后的批跑状态

| 项 | 值 |
| --- | --- |
| 驱动 pid | **19916** |
| 启动时间 | 2026-11-09T10:58:43+08:00 |
| 续跑 | `resume: 1 terminal of 108; pending 107, this session 107` ⇒ **`easy-0-a` 被跳过** |
| 当前 fixture | `easy-0-b`（th8_ud2cdc，107,296 帧）→ `attempt-3`（attempt-1/2 保留，未删除） |
| 模式 | 串行（单实例） |
| heartbeat | 单实例语义（`fixtureId` / `rowsSoFar` / `fractionOfFixture`），未出现多实例行 |

`easy-0-b` 已因本实验中断两次（attempt-1 17,700 行、attempt-2 9,660 行），
这些都**不是**失败条目，不会进 journal，续跑时以新 attempt 目录重跑。

---

## 10. 108 份新预计末时间

剩余 = 108 全部 12,311,460 帧 − 已完成 99,113 帧 = **12,212,347 帧**，107 个 fixture。

| 口径 | 用时 | 预计末时间（自 10:58:43 起） |
| --- | --- | --- |
| 按行速率 82.54 行/s（保守） | 41.1 h | **2026-11-11 04:04** |
| 按 fixture 均值 1200.7 s（与驱动 `projectedRemainingSecondsByFixture` 一致） | 35.7 h | **2026-11-10 22:39** |

两者差异来自 fixture 帧数分布（`extra-*` 仅 44k–55k 帧，短于 `easy-0-a` 的 99k）。
**真实末时间应落在 2026-11-10 22:39 ～ 2026-11-11 04:04 之间**，随已完成 fixture 的实际
行速率收敛。并行化未能缩短此区间。

---

## 11. 未完成 / 下一步

1. 并行化未落地（受阻于第二实例初始化失败），需要先解决显示独占；
2. `extra-1-extra` 导航耗时 45.1 s 一事值得单列——若将来启用并行，导航串行段是瓶颈；
3. 若要继续这条路线，建议下一步做一次**最小验证**：只保留一个实例，验证在实例私有 run dir
   内改写显示模式后 canonical `trace_sha256()` 是否仍等于 manifest（不改 `third_party`）。

---

## 12. 证据路径

| 内容 | 路径 |
| --- | --- |
| 串行批跑日志（含接管与重启） | `.tmp_v400_trace_batch/batch.log` |
| 唯一终态 journal | `.tmp_v400_trace_batch/journal.jsonl`（1 行） |
| 实验一（同时启动） | `.tmp_v400_trace_par2/p2/summary.json`、`p2/*/debugger.log`、`p2/*/status.json` |
| 实验二（串行导航+并行播放） | `.tmp_v400_trace_par2/p2s/summary.json`、`p2s/*/debugger.log`、`p2s/*/status.json` |
| 实验台架 | `.tmp_v400_trace_par2/harness.py` |
| 单测 | `tests/test_v400_retail_trace_batch.py`（27 passed） |
