# EXP-P0-064 — WP4 16 实例真实 Linux 引擎 PID / RSS / 线程 / 僵尸审计

- 日期：2026-11-09 02:14（Asia/Shanghai；约 2026-11-08 10:14 America/Los_Angeles）
- Provider / model：DeepSeek / DeepSeek v4.1 flash
- 范围：只补齐 WP4 规格（[10-P0-训练场.md §WP4 验收第 3 条](../10-P0-训练场.md#L137)）中**唯一未证项**——“16 环境并行”期间的**真实 Linux 引擎 PID、RSS、线程数、僵尸状态**，以及结束后的无僵尸/无残留。
- 本次未改引擎、协议、依赖、第三方；只改 `tools/rl/v400_wsl_vec_stress.py`、`tests/test_v400_wsl_vec_stress.py`，并新增本卡与 `.tmp_v400_wp4_pid_audit/*` 证据。未删除任何文件，未覆盖 EXP032/039/040/041 原记录。

## 1. 结论（先给判定）

| 项目 | 本次实测 | 判定 |
|---|---|---|
| 16/16 实例唯一映射到真实 Linux 引擎 PID | **16/16**（4 个时点中 3 个为 16/16，t=0 为 7/16，见 §5） | **PASS** |
| 同一实例匹配到多个 PID（ambiguous） | 0 个实例 | **PASS** |
| 采样不到（missing/未解析）实例 | 0 个实例（t=0 的 9 个 missing 全部在 t=15 唯一补齐，且全程无歧义） | **PASS** |
| 运行期间 `Z` 状态进程数 | 4 个时点均为 **0** | **PASS** |
| 结束后残留引擎进程（同一 run root） | **0** | **PASS** |
| 结束后 `Z` 状态进程数 | **0** | **PASS** |
| 是否可把 WP4 的 `Linux PID/zombie unknown` 改为 `PASS` | 可以，**但仅限“16×60 秒、shadow/null-device”这一窗口** | **PASS（限定）** |

脚本自动判定字段：`linux_pid_audit.verdict = "PASS"`，`reasons = []`，`unique_mapping_count = 16`，`ambiguous_instances = []`，`never_mapped_instances = []`，`pid_changes = []`。

**不因本卡而宣称完整 WP4 / G0 PASS。** 600 秒窗口的内存曲线、渲染模式（`VecTh08Env` 像素）与 1000-step 集成项不在本次证据范围内，见 §9。

## 2. 上次失败原因与本次采样器设计

| 上次失败原因 | 本次做法 |
|---|---|
| `ps` + `awk` 按 `--data-dir <root>/i-N` 匹配，**把采样器自己的 ps/awk 命令行也匹配进来**（自匹配） | **完全不用 ps/pgrep/grep/awk 管道**。只从 `/proc/*/cmdline`（NUL 分隔）枚举，要求 `argv[0]` **恰好等于**引擎绝对路径，且解析出的 `--data-dir` 值**恰好等于** `<run_root>/i-N`；采样器自身 pid 按 pid 精确排除。探针程序经 **stdin** 送入 `python3 -`，实测它在 `/proc` 里的自述是 `argv0="python3"`、`cmdline="python3 - /root/…/th08-modern"`、`data_dir=null`——即使命令行里含引擎路径也不会被匹配，这是自匹配的结构性排除，而不是靠排除规则兜底。 |
| 16 路样本写进同一个 dict 互相覆盖 | 每个时点、每个实例都是**独立记录**，按 `str(instance)` 键存进各自的 sample dict；sample 追加到列表（追加是原子的），不再有任何共享的 per-instance 字典。 |
| 之前把 Windows `wsl.exe` PID 当成引擎 PID（EXP-039/040） | 本次同时保留并区分 `wsl.exe` PID（`results[].pid`）与 **Linux 引擎 PID**（`linux_samples[].instances[i].pid`），两者逐实例并列于 §4。 |

匹配失败时的行为：某实例 0 个匹配 → `status="missing"`；同一实例 >1 个匹配 → `status="ambiguous"`（`pid=null`，`pids=[...]`，`rows` 保留全部候选）；`/proc` 采样整体失败 → 全部 `missing` 且 `reason="probe_unavailable"`。**任何情况下都不填猜测值。**

采样内容：`pid`、`ppid`、`/proc/<pid>/status` 的 `State`/`VmRSS`/`VmHWM`/`Threads`、`/proc/<pid>/stat` 的 `starttime_ticks`（用于 PID 复用甄别）、`/proc/<pid>/cmdline` 片段（映射证据）。时点固定 t≈0/15/30/59 秒（由 `_sample_offsets()` 按运行时长裁剪），结束后再采两次：`post_run`（16 个 worker 全部关闭后、目录清理前）与 `post_cleanup`（清理后）。

## 3. 验收命令与退出码

在 `D:\WorkShop\Touhou_RL` 执行：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python tools/rl/v400_wsl_vec_stress.py --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-dir /root/v400-data/game --instances 16 --seconds 60 --out .tmp_v400_wp4_pid_audit/16x60_linux_pid_audit.json > .tmp_v400_wp4_pid_audit/run_stdout.log 2> .tmp_v400_wp4_pid_audit/run_stderr.log
```

- 脚本退出码：**0**
- 墙钟时间：**60.522 秒**（请求 60 秒，上限 600 秒；`memory_audit_scope="bounded_request_not_600s_proof"`）
- `run_stderr.log` 为空（0 字节）

单测（本次实跑，含既有 4 个 16×600 证据用例）：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests/test_v400_wsl_vec_stress.py
```

- 退出码：**0**；`22 passed in 0.10s`（4 个既有用例 + 18 个新增用例；既有 `.tmp_v400_wp4_long.json`、`.tmp_v400_wp4_diag_039_600.json` 证据断言未被破坏）

## 4. 16 实例 Linux PID 映射证据（t=59 秒时点）

引擎绝对路径（`argv[0]` 精确匹配对象）：`/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern`
运行实例根：`/tmp/v400_wp4_diag_7820_396b3f1d085b404badda1e45ab589212`

| 实例 | Windows `wsl.exe` PID | **Linux 引擎 PID** | PPID | starttime_ticks | Threads | RSS@59s (kB) | RSS@15s (kB) | VmHWM (kB) | t=0 采样 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 13096 | **598** | 597 | 19057 | 1 | 72160 | 72160 | 90888 | ok |
| 1 | 2548 | **604** | 603 | 19059 | 1 | 71984 | 71984 | 90716 | ok |
| 2 | 5320 | **600** | 599 | 19058 | 1 | 72168 | 72168 | 90576 | ok |
| 3 | 5096 | **610** | 608 | 19060 | 1 | 72012 | 72012 | 90672 | ok |
| 4 | 2212 | **609** | 607 | 19060 | 1 | 71996 | 71996 | 90684 | ok |
| 5 | 18824 | **614** | 613 | 19061 | 1 | 72032 | 72032 | 90892 | missing |
| 6 | 10380 | **606** | 605 | 19059 | 1 | 72052 | 72052 | 90832 | ok |
| 7 | 9164 | **616** | 615 | 19062 | 1 | 72048 | 72048 | 90792 | missing |
| 8 | 1532 | **612** | 611 | 19061 | 1 | 72060 | 72060 | 90708 | ok |
| 9 | 12552 | **618** | 617 | 19063 | 1 | 72052 | 72052 | 90764 | missing |
| 10 | 4412 | **622** | 620 | 19064 | 1 | 71944 | 71944 | 90720 | missing |
| 11 | 19844 | **624** | 623 | 19066 | 1 | 71968 | 71968 | 90524 | missing |
| 12 | 18760 | **626** | 625 | 19068 | 1 | 72116 | 72116 | 90960 | missing |
| 13 | 5824 | **621** | 619 | 19064 | 1 | 72048 | 72048 | 90836 | missing |
| 14 | 8296 | **630** | 629 | 19069 | 1 | 71980 | 71980 | 90948 | missing |
| 15 | 12756 | **628** | 627 | 19068 | 1 | 72136 | 72136 | 90792 | missing |

- **16 个互不相同的 Linux PID**：598, 600, 604, 606, 609, 610, 612, 614, 616, 618, 621, 622, 624, 626, 628, 630。
- 每个引擎的 PPID 都是它自己的 `/init`（16 个互不相同的 interop 父进程 597…629，`argv0="/init"`），即 **16 路各自一条独立的 `wsl.exe` → WSL interop 会话**，不是共享一个 supervisor。
- 16 个 Linux PID 与 16 个 `wsl.exe` PID **一一对应且互不相等**（例如实例 0：`wsl.exe` 13096 ↔ Linux 598），证实 EXP-039/040 记录的 `pid` 确实是 Windows 侧 PID、不能冒充 Linux PID。
- t=59 时点实例 0 的原始 cmdline 证据：

  ```text
      598      597 S (sleeping) 1 72160 19057
      /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-dir /tmp/v400_wp4_diag_7820_396b3f1d085b404badda1e45ab589212/i-0 --rl-control 0,1 --rl-role shadow
  ```

- 4 个时点的 `unattributed_engine_processes` 全为 `[]`：期间没有“是引擎但 `--data-dir` 不属于本次 run root”的进程混入。

## 5. 各时点采样结果（含 t=0 的启动延迟说明）

| 时点 | 实际漂移 | `/proc` 扫描 | 引擎进程数 | ok | ambiguous | missing | `Z` 数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| t=0s | +0.219s | 43 | 7 | 7 | 0 | 9 | 0 |
| t=15s | +0.094s | 56 | 16 | **16** | 0 | 0 | 0 |
| t=30s | +0.079s | 56 | 16 | **16** | 0 | 0 | 0 |
| t=59s | +0.079s | 57 | 16 | **16** | 0 | 0 | 0 |

**t=0 只有 7/16 是真实的启动竞争，不是采样缺陷**：t=0 采样与 16 个 `wsl.exe` 同时启动，采样到达 `/proc` 时实例 5/7/9/10/11/12/13/14/15 的 ELF 尚未 `exec` 完成；这 9 个实例在 t=15 全部唯一映射，且全程从未 ambiguous、PID 从未变化（`pid_changes = []`）。因此“唯一映射 16/16”按实例判定成立；`strict_all_timepoints_instances` 为 7（即在 4 个时点都 ok 的，恰好是 t=0 已经在场的那 7 个）。表中不隐藏这一点。

t=0 那 7 个引擎的 RSS 处于启动中（3,316 / 11,548 / 11,936 / 17,316 / 23,736 / 37,720 / 37,772 kB），与 t=15 之后的稳定值（约 72 MB）不同，也说明这不是同一次快照被复用。

## 6. 各时点 RSS / Threads 摘要

| 时点 | 覆盖实例 | RSS min / 中位 / max (kB) | VmHWM max (kB) | Threads min / max | State 取值 |
|---:|---:|---|---:|---|---|
| t=0s | 7 | 3,316 / 17,316 / 37,772 | 37,772 | 1 / 1 | `R (running)` |
| t=15s | 16 | 71,944 / 72,048 / 72,168 | 90,960 | 1 / 1 | `R (running)`, `S (sleeping)` |
| t=30s | 16 | 71,944 / 72,048 / 72,168 | 90,960 | 1 / 1 | `R (running)`, `S (sleeping)` |
| t=59s | 16 | 71,944 / 72,048 / 72,168 | 90,960 | 1 / 1 | `R (running)`, `S (sleeping)` |

- **Threads：每个引擎在 4 个时点恒为 1**，与 stderr 的 `subthread=0` 一致（shadow/null-device 模式不建渲染子线程）。
- **RSS 在 t=15→t=59 的 44 秒内逐实例变化为 0 kB（+0.00%）**：16/16 全部 `delta = 0`（如实例 0 `72160 → 72160`，实例 3 `72012 → 72012`）。三个时点是三次独立采样（`/proc` 扫描数 56/56/57、漂移 0.094/0.079/0.079 各不相同），不是同一快照。
- `VmHWM`（启动期峰值）90,524–90,960 kB 高于当前 RSS，说明启动后内存已回落并保持平稳，未见增长。
- 泄漏口径仅限 **60 秒窗口、4 个采样点**；不构成 600 秒或长时间曲线的无泄漏证明。

## 7. 僵尸数与结束后残留检查

| 检查点 | `/proc` 扫描 | 引擎进程数（`argv0` 精确等于引擎路径） | 本 run root 残留 | `Z` 总数 | 引擎 `Z` |
|---|---:|---:|---:|---:|---:|
| t=0s | 43 | 7 | 0 | **0** | 0 |
| t=15s | 56 | 16 | 0 | **0** | 0 |
| t=30s | 56 | 16 | 0 | **0** | 0 |
| t=59s | 57 | 16 | 0 | **0** | 0 |
| 结束后 / 清理前（`post_run`） | 30 | **0** | **0** | **0** | 0 |
| 清理后（`post_cleanup`） | 30 | **0** | **0** | **0** | 0 |
| 独立事后复查（`posthoc_residual_probe.json`） | 25 | **0** | **0** | **0** | 0 |

- 目录清理：`CLEANUP_RESOLVED=/tmp/v400_wp4_diag_7820_396b3f1d085b404badda1e45ab589212`，`cleanup_exit_code=0`；沿既有白名单（原路径与 `realpath -e` 都必须匹配 `/tmp/v400_wp4_diag_?*`）只删本次诊断根。
- 清理后 WSL 内 `/tmp/v400_wp4_diag_*` 残留数：**0**。
- `remaining_wsl_process_lines=[]`、`no_wsl_processes=true`（Windows 侧 `tasklist /FI "IMAGENAME eq wsl.exe"`）。
- **归因声明**：本次运行结束后另有一次短暂出现的 `th08-modern --data-dir /root/v400-data/runtime3/wp8-retail3-r2/game --rl-role env` 进程（pid 768）。它**不属于本次 16 实例**：数据目录不在本次 run root 下，角色是 `env` 而非 `shadow`；`post_cleanup` 的 `/proc` 全量快照（30 个进程）里 `engine_process_count=0`，说明它当时尚未出现；复查时该 pid 已不存在。此处如实记录，避免被误读为 WP4 残留。

## 8. 本次运行的其他门（与既有流程一致）

- `all_instances_reached_deadline=true`、`all_exit_zero=true`、`all_error_free=true`、`cleanup_exit_code=0`。
- 总帧数 **1,265,773**，16/16 的 `steps == frames`，最短 worker `run_seconds=59.993`。
- stderr 尾部全部为正常信息（3 种：`setup_frames=8/9/10`），形如 `th08-modern: shadow null-device preflight=ok (… no SDL/GL/window/audio)` + `shadow startup setup_frames=… setup_state=0 subthread=0`，无错误。

## 9. 判定与证据边界

**判定：可以把 WP4 的 `Linux PID/zombie unknown` 改为 `PASS`（限定：16 实例 × 60 秒、shadow/null-device）。** 依据是本次真实运行中：16/16 实例唯一映射到 16 个互不相同的 `/proc` Linux PID（含 PPID、starttime、cmdline 证据），全程 0 个 ambiguous、0 个未解析，4 个时点及结束后两次复查的 `Z` 计数均为 0，结束后本 run root 残留为 0，`/tmp` 清理干净。

以下**不**由本卡证明，仍需保留“未证”：

1. **600 秒时间尺度**：本次是 60 秒窗口。EXP-039/040 的 `PASS-with-limit` 中“长时内存趋势未证”一项不因本卡改变；600 秒仍需同法采样（同一脚本把 `--seconds` 提到 600 即可覆盖 t=0/15/30/59，更长曲线需扩展时点）。
2. **渲染模式**：本次仍是 `--rl-role shadow`、null-device（无 SDL/GL/window/audio），不覆盖像素渲染的 `Th08Env`/`VecTh08Env`。
3. **fd 数与 PSS**：本次采集 State/VmRSS/VmHWM/Threads/PPID/starttime，未采集 fd 计数与 PSS；EXP-040 提出的 fd 项仍未证。
4. 因此 **完整 WP4 与 G0 仍不因本卡 PASS**；本卡只把“Linux PID / 僵尸”这一项从 `unknown` 升级为 `PASS（60 秒、shadow 窗口）`。

## 10. 复现与证据清单

```powershell
# D:\WorkShop\Touhou_RL 下
Get-FileHash -Algorithm SHA256 .\.tmp_v400_wp4_pid_audit\16x60_linux_pid_audit.json
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python .tmp_v400_wp4_pid_audit\summarize.py
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python .tmp_v400_wp4_pid_audit\crosscheck.py
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests/test_v400_wsl_vec_stress.py
```

| 证据 | 路径 | 说明 |
|---|---|---|
| 主证据 | [`.tmp_v400_wp4_pid_audit/16x60_linux_pid_audit.json`](../../../../../.tmp_v400_wp4_pid_audit/16x60_linux_pid_audit.json) | 166,205 字节；SHA256 `F181CC4F40518582AF4AEE3FCBC4F385D64A185E5D9EE19A68F5F9478713AA4F` |
| 事后独立复查 | [`.tmp_v400_wp4_pid_audit/posthoc_residual_probe.json`](../../../../../.tmp_v400_wp4_pid_audit/posthoc_residual_probe.json) | 25 进程、0 引擎、0 僵尸 |
| 运行 stdout | [`.tmp_v400_wp4_pid_audit/run_stdout.log`](../../../../../.tmp_v400_wp4_pid_audit/run_stdout.log) | 含 `CLEANUP_RESOLVED=…` |
| 运行 stderr | [`.tmp_v400_wp4_pid_audit/run_stderr.log`](../../../../../.tmp_v400_wp4_pid_audit/run_stderr.log) | 0 字节 |
| 汇总/交叉校验脚本 | [`.tmp_v400_wp4_pid_audit/summarize.py`](../../../../../.tmp_v400_wp4_pid_audit/summarize.py)、[`crosscheck.py`](../../../../../.tmp_v400_wp4_pid_audit/crosscheck.py) | 生成本卡表格 |
| 采样器 | [`tools/rl/v400_wsl_vec_stress.py`](../../../../../tools/rl/v400_wsl_vec_stress.py) | `_linux_proc_dump`/`_map_engines`/`_instance_records`/`_collect_linux_samples`/`_residual_report`/`_audit_verdict` |
| 单测 | [`tests/test_v400_wsl_vec_stress.py`](../../../../../tests/test_v400_wsl_vec_stress.py) | 22 passed（含既有 4 个 16×600 证据断言） |

关联记录：[EXP-P0-039 16×600](EXP-P0-039-WP4-16x600.md)（PASS-with-limit 不变）、[EXP-P0-040 独立审计](EXP-P0-040-WP4-independent-audit.md)（本卡回应其“无条件验收须补证”第 1 项）、[EXP-P0-041 采样器回退](EXP-P0-041-WP4-sampler-rollback.md)（本次重新实现并验证采样器）。
