# V829-PAIR-01 §3 实施回执（二）｜阻塞解除与验收层补齐

**DeepSeek 训练端 → 用户转交 → GPT 远端研究端**

| 项 | 值 |
| --- | --- |
| 日期/时区 | 2026-11-23，`+08:00`（Asia/Shanghai） |
| 实测窗口 | 2026-11-23 16:40 → 17:35（+0800） |
| 主机代号 | `DSH-WIN-01`（Windows）/ `DSH-WSL-01`（WSL2 Ubuntu，同一台机） |
| 授权 | 用户在上一轮之后明确要求「持续工作」，并已在此前授予全部权限；本轮据此做了**引擎实测**（见 §6 总账） |
| 性质 | 训练端自发实施，**不是**研究端评测工单的回执；不产生任何策略结论 |
| 当前状态 | 预注册 §3.1–§3.3 全部通过；**另外把「新起点」的三处技术阻塞真正解除了**（非 8663 接管帧、非 S1c 录像源、新祖先的谱系核对） |

> **一句话**：现在只要研究端给出「用哪盘录像、在哪个 4k−1 帧接管、预算多少、哪份 checkpoint」，训练端**不需要改任何被 pin 的既有文件**就能开跑 —— 四种起点组合已在真实引擎上跑通（§2）。

---

## 1. 交付物与身份（全部未提交、未登记）

| 文件 | 行数 | 字节 | SHA-256 |
| --- | ---: | ---: | --- |
| `tools/rl/v829_guard.py` | 1,147 | 54,154 | `8ec8517e73e1f0de5c2ca077d86a4dab93eb443afd710a703e8a8de47bf0fce5` |
| `tools/rl/v829_pair.py` | 6,070 | 316,555 | `45a5306ff426f21fdb515485d501a567a4a3a52c140c3324ee5ab9a2f893fea6` |
| `tools/rl/v829_report.py` | 2,691 | 131,470 | `c420a389a603fc7739c48195df94ccb22636f820dca0f7db9fdea042d4cc9732` |

清单（新路径，均可被 `v829_guard.py` 校验）：

| 清单 | roots | 字节 | SHA-256 前 16 |
| --- | ---: | ---: | --- |
| `.tmp_v829_pair/manifest/smoke-old-window.json` | 1（w11, 2400） | 4,103 | `a84faca53333c0cb` |
| `.tmp_v829_pair/manifest/smoke-old-window-1800.json` | 1（w11, 1800） | 4,311 | `6a1f9fd326d64171` |
| `.tmp_v829_pair/manifest/smoke-old-window-t3999.json` | 1（w11, N=3999） | 3,408 | `f81ed2a6e9b5af66` |
| `.tmp_v829_pair/manifest/smoke-two-windows.json` | **2**（w11+w12） | 5,283 | `f4e74554c29c5c0b` |
| `.tmp_v829_info/s10/manifest-dagger-3006.json` | 1（**dagger 闲置录像**） | 4,603 | `9a8561b7993d3afc` |

**被 pin 的文件一个字符未动**（运行前后 SHA-256 相同）：
`v432_fork_arm.py` `3323550607730854…` · `v450_cf_label.py` `b7ab0ebeee38df4f…` ·
`v830_bombpol.py` `8e40ded82ee0f545…` · `v431_dense_driver.py` `1079d93a46ed77c7…` ·
`v8r0_urgency.py` `98b28a8dd7c1f7c5…` · `v8o0_ruletight.py` `2351be25efd22dba…` · `v829_guard.py` 见上表。

---

## 2. 三处技术阻塞已解除（都有真机证据）

### 2.1 非 8663 的接管帧（原来三处硬编码只读断言挡死）

**关键约束（本轮实测确认，写进工具了）**：录像锚点每 4 帧一个（`v431_dense_driver.py:81-82`），
`build_masks` 要求 `target = N+1` 必须是锚点（`:88-89`）⇒ **合法接管帧必须是 `4k − 1`**。
8663 = 4×2166 − 1，所以「任意帧」的真实条件是 `N % 4 == 3`；`N % 4 == 0`（如 4000）**不合法**。

做法：`v829_pair.py` 在**自己进程内**改 importlib 模块对象的属性（必须覆盖 **7 个**对象，
因为真正跑 `walk_prefix` 的是 `v450._M432`），运行前后对三份文件做 SHA-256 比对，不一致即 fail-closed。

真机证据（window 11，N=3999）：

```text
[v829] prefix constants: takeover=3999 expected_checkpoints=999 target_frame=4000
       override_applied=True modules=7 files_on_disk_unmodified=True
[v450] window 11: parent up pid=1348159 prefix 999/999 checkpoints, 0 mismatches,
       PARK frame=3999, threads=1
```

21 项断言全部通过，其中两条最有说服力：
- `prefix.hash_checks == 999`（= 3999 // 4）、`hash_mismatches == 0`、`final_frame == 3999`、`park_frame == "3999"`；
- **`reset_hash = 7c8705fc88f94c17` 与 N=8663 的历次运行完全相同**（它是 frame 0 的重置哈希，与 N 无关），
  而 `final_step_hash` 从 `0299ade917af1401` 变成 `6568a22d1a42812c`（N 处状态，本就该变）。
  ⇒ 「同起点」判据与接管帧解耦，这是正确性的直接证据。
- 附带结果：这个起点上**两臂都跑满 2400 帧**（`full_segment=True`、`terminal=None`），
  说明「满段」这条计分路径也在真机上证过了（此前只有死亡分支被走过）。

### 2.2 非 S1c 的录像源（原来 `build_masks` 把路径写死）

做法：清单可声明 `prefix_tape` + `prefix_tape_sha256`（+ 可选 `rows`/`frames`），
工具用 `read_tape()` 读任意 4 帧锚点录像并**注入** `v450.masks_for`；未声明时行为与之前逐字节相同。

真机证据（S9 审计发现的**未被任何登记来源使用**的录像 `dagger-s0d1seed3006`，N=2999）：

```text
[v829] w3006: prefix_tape .../dagger-s0d1seed3006/labels.jsonl sha256 55fd2a62…4393fc
       rows=2088 frames_built=8352 target=3000 injected_into=2 object(s) files_on_disk_unmodified=True
[v450] window 3006: parent up pid=1348620 prefix 749/749 checkpoints, 0 mismatches,
       PARK frame=2999, threads=1
```

`read_tape` 与 `build_masks` 在三盘真实 S1c 录像上产出的 `masks`/`hashes`/`frames_built` **逐位相同**（离线自证）。
⇒ **录像源与接管帧已完全参数化**，产物里记 `source_kind="prefix_tape"` 与录像 SHA，不会与旧路径混淆。

### 2.3 新祖先的谱系核对（原来试点根本用不了 `--lineage-config`）

现状：`--lineage-config` 要求每根声明非空 `lineage_ancestor_ids`，否则 `LINEAGE_MISSING`；
但**新祖先本来就不在 `configs/v430/splits.json` 里**，于是试点上这个开关永远打不开。
已加 `lineage_new: true`：允许空 id，记 `LINEAGE_NEW`（只记录、exit 0）并写残差
`"new ancestor: exposure is declared by the run table, not provable against the frozen lineage"`；
`lineage_new` 与「命中 train/dev」同时出现仍判 `EXPOSURE_CONTRADICTS_LINEAGE`（自相矛盾）。

---

## 3. 计分与报表层（预注册 §4）

`tools/rl/v829_report.py`：逐祖先 2×2（both/only_G1/only_T2/neither）、
`Δ = 8 祖先差值的等权平均`、绝对满段率、正/平/负祖先数、**真去一法**（min/max 并报并列集合）、
三份 checkpoint 分组、Δ40、未决对的上下界、技术无效与待判单列、库存事件（扣减次数/耗尽后存活/带雷死亡）、
判定表（NOMINATE_G1 / KEEP_T2 / UNDETERMINED / REFUSED）。三级 fail-closed：Δ → bounds → verdict。

**合成验收 170/170 断言**（8 祖先 × 3 ckpt × 2 档 × 2 臂的手算对照，含 4:4 打平、缺档、技术无效、重复行、跨清单池化等反例）。
**真实冒烟数据上如实拒绝出 Δ**：4 文件 8 行 → `ANCESTOR_COUNT` / 单清单 / ckpt 轴三道门全 FAIL，给分母不给效应。

### 3.1 一个值得研究端知道的复现性发现

`--pairs` 的去一法曾在 Python 3.12 与 3.10 下选出**不同的祖先**（`anc4` vs `anc7`）。
根因不是哈希随机，而是 **CPython 3.12 把内置 `sum()` 的浮点累加改成了 Neumaier 补偿（gh-100425）**：
场景里「去掉 anc4」与「去掉 anc7」的均值数学上都等于 `1/14`，3.12 把它们压成同一个 double，
3.10 朴素累加得 `0.5000000000000001` vs `0.5`。
修法：所有发布的均值改用 `fractions.Fraction` 精确算、末尾只 `float()` 一次；极值按精确相等判并列。
⇒ 同一输入在 3.10.12 与 3.12.10 上 `--json` **逐字节相同**（`A4301DD6…`），
连跑 3 次 `--json` 也逐字节相同（可用 `SOURCE_DATE_EPOCH` 钉住时间戳）。
**这条对研究端的意义**：任何跨机器/跨版本的数字比对，都不应默认「浮点求和是稳的」。

---

## 4. 运行时能力

| 能力 | 状态 | 证据 |
| --- | --- | --- |
| 技术故障「重跑双方一次」 | 已实现、**仅离线验证** | 3 个单测（首次 invalid 后重跑、合法死亡不重跑、两次都 invalid 停在 attempt=2）；`attempt=2` 的两次 PARK 与 `.a2.trace.json` 只在单测里验过 |
| `--workers N`（1..4） | 已实现、**真机 2 worker 通过** | 见下 |
| `lineage_new` | 已实现、离线验证 | `LINEAGE_NEW` exit 0；矛盾声明 exit 1 |

**真机并发**（两窗两 cell，`--workers 2`）：

```text
[v829] mode=pair --workers 2: 2 cell(s) -> 1 batch(es); tags [...-w1, ...-w2]
[v829] worker 1/2 done rc=0   worker 2/2 done rc=0
[v829] merged 4 row(s) from 2 worker jsonl into ...-r7.jsonl sha256=f3aeda451cfe81de...
[v829] done: workers=2 batches=1 completed=2 failed=False rows=4
=== done rc=0 wall_s=8 ===  engines_after=0
```

逐项核对：两 worker 各自 `runs/<tag>-w<i>/` 与 `data/<tag>-w<i>/`，tag/数据目录**无重叠**；
两 worker 的 `audio_audit.json` 都是 `cfg_files=1, muted=1, audible=0, four_zero=1`；
合并后 4 行、单清单摘要、每 cell 的 `pair_rng`/`generator_seed` 与清单一致。

---

## 5. 本轮抓到并修掉的四个缺陷（都记来源）

| # | 缺陷 | 怎么发现的 | 修法 |
| ---: | --- | --- | --- |
| 1 | `v829_pair.py` 有一个 **PEP 701 多行 f-string**，在 Python 3.10 下是**语法错误** | 真机第一次跑就 `SyntaxError`；子代理当时用 **Windows 3.12** 的 `py_compile` 验收，所以没抓到 | 改成先算变量再拼串。**并把「验收必须用 `/root/v400-venv/bin/python`」写进所有后续子单** |
| 2 | 报表层去一法的并列极值写成单名断言 | 3.10 下 selftest 157/158 | 精确 Fraction + 并列集合（见 §3.1） |
| 3 | `--out` 的父目录不自建，PARK 成功后才崩 | 真机 r1 | 在 spawn 前 `mkdir(parents=True)`；事先删空目录后复跑证明可自建 |
| 4 | worker 的 stdout 日志目录导致子进程把自己判成 `TAG_NOT_UNIQUE` | 子代理跑 `--workers 1` 真机时暴露 | 日志改到 `runs/<tag>/worker-w<i>.log`，3 条单测锁定 |

**系统性教训（已采纳为规则）**：本机真实运行解释器是 WSL 的 **Python 3.10.12**，
而 Windows 侧是 3.12.10。**用错解释器做编译验收等于没验收** —— 这类缺陷不会在 3.12 上显形。

---

## 6. 引擎与合规总账（本轮）

| 用途 | spawn | 结果 |
| --- | ---: | --- |
| `w11-t3999-r5`（非 8663 首测） | 1 | exit 0，两臂满段 |
| `dagger3006-t2999-r6`（闲置录像首测） | 1 | exit 0（前三次尝试在 import/window/lineage 处被拒，**未 spawn**） |
| `conc-two-windows-r7`（2 worker） | 2 | exit 0，合并 4 行 |
| **执行代理未计划的一次**（`work-gate/runs/gate-pair-w1`） | 1 | 已被其自查披露：w11 PARK 8663、双臂 first_death、`invalid=0`、wall 6.997 s；音频 `muted=1/audible=0`、cfg 四字段全 0、`/root/v400-data` 未改写 |
| **合计** | **5** | 每次前后 `th08-modern` 计数 0/0；每次私有 cfg 四字段全 0；未 `pkill`、未碰他轮进程 |

**一处需要交代的误写（已逐字节还原）**：执行代理误跑了 `v431_audio_cfg_audit.py`（该工具无 `--root`、会写回固定路径），
改到了 `Document/Update/Alpha/v4.3.1/evidence/R14/audio_cfg_audit.json`。它随即用 `git show HEAD` 重定向还原。
**主代理独立复核**：与 HEAD blob **逐字节相同**（4,921 B、`sha256 ce8b36fc6fc8c4ec…`、0 个 CRLF、`git diff HEAD` 为空）；
`git status` 仍显示 `M` 只是 DrvFs 的 stat 缓存假象（Git 自己也警告 LF→CRLF 的将来转换）。
⇒ **该既有证据文件的内容未被改变**，但这次误写本身应记在案。

其余合规：未采集、未训练、未恢复 WP05、未 `git add/commit`、未上传、未删/移/覆盖任何资产；
`Document/` 只新增了本文件与上一份回执，未改既有内容。

---

## 7. 仍然缺什么（按谁能解决分类）

### 7.1 只有研究端能定

1. **干净祖先**：登记谱系里仍未曝光 = **0**；本轮 S9 磁盘审计找到 **4 盘未被任何登记来源使用**的录像
   （`dagger-s0d1seed{3006,3007,3008,3009}`，frames_built 8,352 / 8,736 / 5,568 / 9,024，4 帧锚点齐全，
   `≥2400 帧尾巴` 的最大接管帧分别是 5,951 / 6,335 / 3,167 / 6,623）。**它们只是 seed 未登记，不等于行为干净**
   （DAgger 行为来自在 seeds 1–25 上训过的 v4.0 学生）—— 算不算干净祖先，得研究端定性。
   8 个目标 → 认这 4 个还差 4 个；不认就差 8 个，须新采集。
2. **要不要把 `v829_guard.py` 的 schema 当冻结契约**（现在只是训练端提案，未登记未提交）。
3. **道中/Boss 二分判据**：清单无此字段，预注册要 4+4。
4. **接管帧定义**：A1 旧规则（首死前 30–120 帧）与 2,400 帧差 20–80 倍。

### 7.2 训练端下一轮可自行补齐

1. **重跑一次的真机证据**：现在只有单测。需要有控制地制造一次技术故障（例如中途 kill 引擎）验证两次 PARK、
   `gate-w<w>-a2.json`、`.a2.trace.json` 与 `superseded` 语义。
2. **4 worker 的资源竞争**：目前只验了 2 worker；4 路并发下的引擎吞吐与 `load` 未测。
3. **`--mode check` 与 `--workers` 不共存**（exit 2）：离线预演并发计划只能靠单测。
4. **`v829_report.py` 的路径分隔符**：跨 Windows/WSL 比对时 `inputs.files` 会不同（不影响数值）。

---

## 8. 命令与退出码（本轮可复算部分）

```text
# 全部在 WSL /mnt/d/WorkShop/Touhou_RL，/root/v400-venv/bin/python = Python 3.10.12
py_compile v829_guard.py / v829_pair.py / v829_report.py                 -> 0 / 0 / 0
v829_guard.py selftest                                                   -> 0   (34/34)
selftest_v829_report.py                                                  -> 0   (170/170)
v829_pair.py --help                                                      -> 0
v829_pair.py --mode check  （三个旧清单 + dagger 清单）                    -> 0 / 0 / 0 / 0
v829_pair.py --mode pair  w11 N=3999 2400                                -> 0   (2 行, 双臂满段)
v829_pair.py --mode pair  w3006 N=2999 2400 (dagger tape)                -> 0   (2 行)
v829_pair.py --mode pair  two-window manifest --workers 2                -> 0   (4 行合并)
v829_report.py --pairs '.tmp_v829_pair/eval/*.jsonl'                     -> 1   (如实拒绝出 Δ，给分母)
```

**引擎启动计数 5（含 1 次执行代理未计划的）、残留 0、音频违反 0。**

---

## 9. 补记（2026-11-23 17:50 +0800）：故障注入实验 —— 重跑规则的两条分支都在真机验过

预注册 §4 的「一次技术故障只按原 pair、起点、随机流和配置重跑双方一次」本轮**不再只有单测**。

### 9.1 行级故障（对局内杀引擎）→ 整对重跑一次

做法：起一对，等引擎进对局后 `kill -9` 只属于本次 `tag` 的 `th08-modern` 进程。

```text
[v829] w11: parent pid=1354892 PARK frame=8663 prefix=2165/2165 attempt=1
[v829] … G1 attempt=1 frames=0/2400 decisions=494 terminal=None invalid=True
        error=HarnessError: harness closed the control pipe
[v829] … T2 attempt=1 frames=0/2400 decisions=0   invalid=True
        error=BrokenPipeError: [Errno 32] Broken pipe
[v829] w11: attempt 1 finished: rows=2 invalid_rows=2 wall_s=3.27
        -> re-running the whole pair once (pre-registration sec. 4)
[v450] window 11: parent up pid=1354924 prefix 2165/2165 checkpoints, 0 mismatches, PARK frame=8663
[v829] … G1 attempt=2 frames=1966 decisions=982 invalid=False
[v829] … T2 attempt=2 frames=1966 decisions=982 invalid=False
[v829] done: 4 row(s) (2 in the final attempt), invalid=0, attempts=2, same_initial_rng=True
```

**产物**（`runs/v829-retry-fault-r10/`）：`gate-w11.json` + `gate-w11-a2.json`（**两次 PARK**）、
jsonl 里 **4 行**（attempt 1 的 2 行**保留**、attempt 2 的 2 行带 `supersedes` 行指纹）、
`run.json.attempts` 两条记录（含 `invalid_rows`、`wall_s`、`attempt_fingerprint`）。
**attempt 2 的数字与 r2/r4 完全一致**（1966 帧 / 982 决策 / emits 8 与 3）⇒ 杀掉重启后同随机流仍复现同一轨迹。

### 9.2 注入暴露的真缺口：**恢复失败当时不重跑** —— 已修并复验

第一版注入打在**前缀重放阶段**，得到：

```text
REFUSED [RECOVERY_FAILED] w11: v450.start_parent failed (attempt 1; HELLO -> 8663-frame prefix
        -> 2165 hash checks -> PARK -> DEBUG threads=1): HarnessError: harness closed the control pipe
```

exit 1、**没有重跑**。但 §4 把「无法恢复」明确列为技术无效 ⇒ 这是实现缺口，不是正确行为。
已修（`v829_pair.py`）：`start_parent` 在 **HELLO / 前缀 / 哈希 / PARK / DEBUG** 任一阶段抛异常都视同技术无效，
整对重跑一次，`attempts` 记 `phase="recovery"`、`kind ∈ {engine_crash, prefix_hash_mismatch, recovery_error}`。

**真机复验（在 PARK 边界 SIGKILL）**：

```text
[verify] attempt 1: real prefix walked (2165 checks, 0 mismatches, final_frame=8663); killing engine … (SIGKILL)
[verify] attempt 1: park raised BrokenPipeError
[v829] w11: attempt 1 finished: rows=0 invalid_rows=0 phase=recovery
        error=attempt 1 engine_crash: BrokenPipeError  -> re-running the whole pair once
[v450] window 11: parent up pid=1356486 prefix 2165/2165 checkpoints, 0 mismatches, PARK frame=8663
[v829] … G1 attempt=2 frames=1966 decisions=982 invalid=False
[v829] w11: attempt 1 failed in recovery -> the same PARK / pair / window / takeover / tape / ckpt / pair_rng
        was re-run once; the new PARK is gate-w11-a2.json
[v829] done: 2 row(s) … attempts=2 …           rc=0
```

**两次都失败**的分支也已真机验（持续 kill 直到两次尝试都死）：

```text
[v829] w11: attempt 1 finished: phase=recovery error=attempt 1 engine_crash: HarnessError: harness closed the control pipe
[v829] w11: attempt 2 finished: phase=recovery error=attempt 2 engine_crash: BrokenPipeError: [Errno 32] Broken pipe
[v829] w11: recovery failed on all 2 attempt(s) -> REFUSED [RECOVERY_FAILED]; run.json keeps all 2 record(s)
REFUSED [RECOVERY_FAILED] … no third attempt (pre-registration sec. 4 allows exactly one re-run)
```

⇒ **硬上限 2 次尝试被真机证死**，且 `run.json` 保留两条恢复失败记录。

### 9.3 修完之后「技术无效」的完整口径（写进工具 docstring 与 `--help`）

| 会重跑（一次） | 绝不重跑 |
| --- | --- |
| `start_parent` 在 HELLO / 前缀 / 哈希 / PARK / DEBUG 任一阶段抛异常 | 合法死亡 `first_death` |
| 任一臂行 `invalid=true` 或 `error` 非空 | 满段 `survived=true` |
| | `stage_clear` |
| | guard / preflight 拒绝（输入问题，不消耗第二次尝试） |

### 9.4 本轮引擎总账（修正 §6 的 5 次为 **12 次**）

| 用途 | spawn |
| --- | ---: |
| `w11-t3999-r5`（非 8663 首测） | 1 |
| `dagger3006-t2999-r6`（闲置录像首测） | 1 |
| `conc-two-windows-r7`（2 worker） | 2 |
| `retry-fault-r9`（注入打在**前缀阶段**，暴露 §9.2 缺口） | 1 |
| `retry-fault-r10`（注入打在**对局内**：attempt1 + 重跑的 attempt2） | 2 |
| `retry-parkkill-r11`（PARK 边界 SIGKILL：attempt1 + attempt2） | 2 |
| `retry-alwaysfail-r12`（两次都 kill） | 2 |
| 执行代理未计划的那 1 次（`work-gate/gate-pair-w1`，已披露） | 1 |
| **合计** | **12** |

全部满足：每次前后 `th08-modern` 计数 0/0（`engines_total_after=0`）、每次私有 `th08.cfg` 四字段全 0、
每次跑后 `audio_audit.json` 的 `audible=0`、未 `pkill` 他轮进程、`/root/v400-data` 未被改写。

### 9.5 工具身份变更

`tools/rl/v829_pair.py`：`297,545 B / 5,718 行 / 6cf4ea41…` → **`316,555 B / 6,070 行 / 45a5306f…`**
（§1 的表已同步）。其余两件工具身份未变。

### 9.6 并发调度实测：2 worker 最好，4 worker 反而更慢

四个旧窗作为四个 cell，1800 帧档，同 checkpoint：

| 配置 | cell 数 | 墙钟 | 每对有效成本 | 备注 |
| --- | ---: | ---: | ---: | --- |
| `--workers 1`（单对基线，r3） | 1 | **5 s** | 5.0 s | PARK 1.8 s + 两臂各 ~0.9 s |
| `--workers 2`（r7，2400 档） | 2 | **8 s** | 4.0 s | 略优于串行 |
| `--workers 4`（r13，1800 档） | 4 | **27 s** | 6.8 s | **比串行（≈20 s）还慢 35 %** |

原因：每个 worker 都要自己跑一次 8,663 帧前缀重放 + PARK（~1.8 s），4 路同时起引擎时 CPU 抢占把这段放大；
而单对的两臂本身只有 ~0.9 s，所以**启动开销主导，并发没有可摊薄的分母**。
⇒ **试点建议用 `--workers 2`（或 3），不要用 4。** 若把每 worker 的 cell 数提高（例如一个 worker 连跑 4 个 cell，
PARK 次数不变但引擎启动摊薄），才可能真正受益 —— 当前实现是「一个 worker 一个 cell」，所以没有这个收益。
（4 个 worker 的 tag / data dir / audio audit 全部独立且 `audible=0`，功能没问题，只是不划算。）

---

## 10. 补记（2026-11-23 18:05 +0800）：接管帧边界、清单生成器、会话总账

### 10.1 接管帧的可行范围（边界实测，window 11 / 13,200 帧录像 / 1800 帧预算）

| N | target | 检查点 | 失配 | PARK | `reset_hash` | `final_step_hash` | 结果 |
| ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| **7**（最小合法） | 8 | **1** | 0 | 7 | `7c8705fc88f94c17` | `daf305e86cbc515f` | G1 满段 / T2 死 1548 |
| 3,999 | 4,000 | 999 | 0 | 3,999 | `7c8705fc88f94c17` | `6568a22d1a42812c` | 双臂满段 |
| 8,663（历史） | 8,664 | 2,165 | 0 | 8,663 | `7c8705fc88f94c17` | `0299ade917af1401` | 见 r2/r3/r4 |
| **11,399**（最大可行） | 11,400 | **2,849** | 0 | 11,399 | `7c8705fc88f94c17` | `80cc0ac19d964271` | 双臂满段 |

**四个不同接管帧的 `reset_hash` 完全相同**（frame 0 的重置哈希），而 `final_step_hash` 各自不同
⇒ 「起点同一性」与接管帧**解耦**，这是参数化正确性的直接证据。
上界 = `total_frames − budget`，且必须 `≡ 3 (mod 4)`（13,200 − 1,800 = 11,400 ⇒ N=11,399）。

### 10.2 清单生成器：`v829_manifest_build.py`（4 个工具里的最后一块）

研究端只需给 `v829-sources/1` 小 JSON，训练端生成**通过守卫**的 `v829-pair-manifest/1`：
`sha256(tape)`、`rows × 4`、`hash_checkpoints = N // 4`、`window` 派生、`ckpt/engine` 实测 SHA 全自动，
16 个错误码（命中即不写清单），收尾**强制**调 `v829_guard.py check` 并落 `<out>.guard.json` sidecar。

**端到端实测**（`.tmp_v829_info/milestone_e2e.sh`，0 次引擎）：

```text
root milestone-old-w11     window 11   tape bc663759… rows 3300 frames 13200 takeover 8663 cp 2165
root milestone-dagger-3006 window 3006 tape 55fd2a62… rows 2088 frames  8352 takeover 2999 cp  749
wrote manifest 5290 B sha256 ee65f7f0aaab6116…      build_rc=0
guard (生成器内调) exit 0 PASS 18 checks            guard_rc=0（主代理独立重跑同 0）
IDEMPOTENT: yes（连跑两次 sha256 相同，不写时间戳）
pair --mode check → rc=0, would_start_engine: false, engines_after=0
```

⇒ **「研究端给来源 → 清单 → 预演」这条链已闭合**，中间不需要人工填任何派生字段。

### 10.3 会话引擎总账（22 次，全部音频合规）

| 段 | tag / 用途 | spawn |
| --- | --- | ---: |
| 第一段（回执 02） | r1 缺陷 / r2 2400 / r3 1800 / r4 确定性 | 4 |
| 第二段（回执 03 §6/§9.4） | r5 t3999 / r6 dagger / r7 worker×2 / 执行代理未计划 1 次 | 5 |
| 第三段（故障注入） | r9 前缀阶段 / r10 行级×2 / r11 恢复级×2 / r12 两次都失败×2 | 7 |
| 第四段（本轮） | r13 四 worker / boundary min / boundary max | 6 |
| **合计** | | **22** |

每次：前后 `th08-modern` 计数 0/0、私有 `th08.cfg` 四字段全 0、跑后 `audible=0`、
未 `pkill` 他轮进程、`/root/v400-data` 未改写。**没有采集任何新数据、没有训练、WP05 仍暂停。**

### 10.4 四个工具（最终身份）

| 工具 | 行数 | 字节 | SHA-256 前 16 | 离线自测 |
| --- | ---: | ---: | --- | --- |
| `v829_manifest_build.py` | 1,511 | 70,630 | `0191485b48f30468` | 20/20 负例 + 等价性 + 幂等 |
| `v829_guard.py` | 1,147 | 54,154 | `8ec8517e73e1f0de` | `selftest 34/34` |
| `v829_pair.py` | 6,070 | 316,555 | `45a5306ff426f21f` | 4 单元 + 2 故障注入单测 |
| `v829_report.py` | 2,691 | 131,470 | `c420a389a603fc77` | `selftest 170/170`（3.10 与 3.12 逐字节同） |
