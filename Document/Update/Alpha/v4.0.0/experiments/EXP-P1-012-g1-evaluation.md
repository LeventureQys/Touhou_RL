# EXP-P1-012 — G1 门测量：S1N 20 局 / PSN 8 面

- **日期**：2026-11-09
- **状态**：**完成**（S1N 20/20、PSN 8/8，`partial=false`；G1 判定见 §7）
- **上游**：[EXP-P1-011](EXP-P1-011-multistage-progress.md)（教师 v1 真过面 + 多面可观测）、[EXP-P1-009](EXP-P1-009-stage-clear-diagnosis.md)（过面诊断）、[EXP-P1-006](EXP-P1-006-teacher-perf.md)（教师吞吐）
- **产物**：`.tmp_v400_g1/`（逐局 JSONL + 汇总 JSON + 日志）、`tools/rl/v400_g1_eval.py`（新增）、`tests/test_v400_g1_eval.py`（新增）
- **一句话**：把 P1 的验收数字——**S1N 过关率（≥90%）**与 **PSN 每面平均死亡（≤0.5）**——用真实退出码测出来，并给出 G1 判定与失败清单；跑不完的部分显式标 `partial=true`，不猜数。

---

## 1. 门限与判定口径

| 指标 | 定义（本卡实际执行的） | 阈值 |
|---|---|---|
| **S1N clear rate** | `--stage 0 --difficulty normal`，seeds **1000–1019** 共 20 局，每局 `--max-frames ceil(1.5×13,258)=19,887`、`--stop-on-death`、`--quiet`；`stage_clear` 由引擎自己的 `STATE.f1` 是否离开 0 判定 | **≥ 90%**（同时给 95% Wilson 区间） |
| **PSN deaths/stage** | stage index **0..7** 各跑一局（normal、seed 1、`RESET <s> 1 0 1 0`），跑到「过面 / 死亡 / 帧上限」；每面死亡数取 `STATE.deaths` 增量 | **每面平均 ≤ 0.5** |

**严格口径（写死在这里，防止后面滑）**：

1. **`stage_clear` 只认引擎 verdict**：`STATE.f1`（`GameManager.currentStage`）离开起跑面才算过面。跑到帧上限仍活着记 `stop_reason=max_frames` + `stage_clear=false`，**不写成"过面"**，也不写成"失败"——它是一个独立的第三种结局。
2. **死亡**：`deaths_final - deaths_baseline`，来自 `STATE` 的死亡计数，不用"血量/残机"推断。
3. **两个分母**：`申报 N 局`（含未跑/失败的局）与`实际跑完的局`都要给，且过关率两个口径都算。失败局（引擎错误、超时、无 JSONL）保留日志、计入申报分母、从"跑完"分母剔除。
4. **Boss「被打死」**：只写可观测证据——`ENEMIES` 里 boss 行的 `life` 是否归零、Boss 消失后场上是否连续 ≥3 决策无敌人生存。观测不到就写 `not_observed`，不推断。
5. **不改口径凑阈值**：帧上限只由 `ceil(1.5×人类该面帧数)` 决定；缺人类数据的面用任务裁定的固定上限并标 `cap_source=fallback`，两种来源在表里逐面可见。

---

## 2. 人类参考与帧上限

参考文件：`Document/Update/Alpha/v4.0.0/bench/human_reference.json`（**注意**：任务里写的 `bench/human_reference.json` 在仓库根不存在，参考数据实际在 v4.0.0 包下的 `bench/`；本卡与工具都用这个真实路径）。取 `difficulty=normal` 的 `Th8_ud2970.rpy` 集，帧数是 `gameplay_frames`。

| stage index | 面 | 人类帧数 | 上限（规则） | cap 来源 |
|---|---|---|---|---|
| 0 | Stage1 | 13,258 | 19,887 | `ceil(1.5×13,258)` |
| 1 | Stage2 | 18,525 | 27,788 | `ceil(1.5×18,525)` |
| 2 | Stage3 | 20,009 | 30,014 | `ceil(1.5×20,009)` |
| 3 | Stage4A | 27,722 | 41,583 | `ceil(1.5×27,722)` |
| 4 | Stage4B | **无 normal 人类数据** | 45,000 | **fallback**（任务裁定固定值；同类面 Stage4A 的 1.5× 会是 41,583，仅作参照，不用） |
| 5 | Stage5 | 25,675 | 38,513 | `ceil(1.5×25,675)` |
| 6 | Stage6A | **无 normal 人类数据** | 45,000 | **fallback**（同上；同类面 Stage6B 的 1.5× 会是 54,605，仅作参照） |
| 7 | Stage6B | 36,403 | 54,605 | `ceil(1.5×36,403)` |

`human_reference.json` 的 `coverage.missing.normal` 自己就写着缺 `Stage4B / Stage6A / Extra`，与上表一致。**S1N 的上限 19,887 与 stage 0 相同**。

---

## 3. 工具与产物

| 文件 | 作用 |
|---|---|
| `tools/rl/v400_g1_eval.py`（新增） | 只做**编排**：按上面的 CLI 逐局拉起 `tools/rl/v400_teacher.py`、记墙钟与退出码、解析 `kind=report` 行、算 Wilson 区间、写汇总。**不碰策略、不重算标签、不覆盖帧上限** |
| `tests/test_v400_g1_eval.py`（新增） | 15 项单测：帧上限算术、Wilson 闭式、`stage_clear` vs `max_frames` 不许混、Boss 证据分级、两个分母、断点续跑 |
| `.tmp_v400_g1/s1n/<case>.jsonl` | 每局教师原始 JSONL（逐决策） |
| `.tmp_v400_g1/s1n_runs.jsonl` | 每局一行记录（命令、退出码、墙钟、`stage_clear`、帧数、死亡、bomb、Boss 证据） |
| `.tmp_v400_g1/psn_runs.jsonl` | 同上，PSN 8 面 |
| `.tmp_v400_g1/s1n_summary.json` / `psn_summary.json` / `g1_summary.json` | 汇总 |
| `.tmp_v400_g1/logs/<case>.log` | 每局子进程 stdout/stderr（`--quiet` 下即最终报告 + 任何引擎错误） |
| `.tmp_v400_g1/s1n_batch.log` / `psn_batch.log` | 批跑进度流 |

断点续跑：每条记录**跑完立即追加**；重跑同一命令会跳过已有 `completed=true` 的 case（`--force` 可强制重跑），因此超时/中断不会丢已得的局。

验收命令：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests/test_v400_g1_eval.py
# 15 passed
```

---

## 4. 已知结构性限制：换面后 `PARK` 不可用（不影响逐面 PSN）

在本轮 S1N 启动前，机器上另有一条 `--follow-stage --max-frames 44000` 的连续多面推进跑（`.tmp_v400_clear_multi.log`），它在 **2026-11-09 13:38** 以退出码非 0 结束，唯一输出是：

```json
{"status": "engine_error", "reason": "'PARK' -> ERR park-unsafe role=shadow-required threads=2 setup=1 null_device=1"}
```

源码判据（`third_party/th08-rl/src/rlharness/fork.cpp:259-268`，`HandlePark()`）：

```cpp
if (!Ready() || (display && *display) ||
    g_Supervisor.startupThreadState != SupervisorStartupThreadState_Idle ||
    g_Supervisor.subthreadActive || !g_Supervisor.d3dDevice) {
    printf("ERR park-unsafe role=shadow-required threads=%d setup=%u null_device=%d\n", ...);
```

`PARK`（child 录制路径的每一步前置）要求 **`startupThreadState == Idle` 且 `subthreadActive == 0`**。一次换面（`AdvanceToNextStage()` → `SupervisorState_GameManagerReInit`）会让 supervisor 进入重初始化/子线程活跃状态，于是**换面后的第一个 `PARK` 就被拒绝**，CLI 按 fail-closed 退出。

**对证据的影响（必须这样写）**：

- ❌ **「连续多面推进」（一次进程内 stage 0→1→2…）当前不可测**：不是教师打不过，而是录制路径在换面后拿不到 hazard 快照；任何"跨面平均死亡"都不能用 `--follow-stage` 得到。
- ✅ **「逐面（PSN）测量」不受影响**：每面各自 `RESET` 起跑，整局都在同一面内，不会跨过换面点。本卡的 PSN 就是这样跑的。
- ✅ **S1N 不受影响**：`--stop-on-death` 且**不带** `--follow-stage`，检测到 `STATE.f1` 变化即停，换面后不再 `PARK`。

---

## 5. S1N 结果（20 局，stage 0，normal，seeds 1000–1019）

### 5.1 命令

```bash
# 串行段（seed 1000–1003，4 局）——每局的等价 CLI：
/root/v400-venv/bin/python tools/rl/v400_teacher.py --decisions 4979 --max-frames 19887 \
  --stop-on-death --stage 0 --difficulty normal --seed 1000 --mode full --quiet \
  --out .tmp_v400_g1/s1n/s1n_seed1000.jsonl
# 并行段（seed 1004–1019，6 并行，每 worker 独立 data-dir）：
bash .tmp_v400_g1/run_s1n_parallel.sh   # 见 §9
# 两段都由 tools/rl/v400_g1_eval.py 驱动；退出码逐局记录在 s1n_runs.jsonl
```

`--decisions` 统一取 `cap//4+8 = 4,979`（每决策 4 帧），保证**绑定停止条件永远是 `--max-frames`/死亡/换面**，而不是决策数用尽。

### 5.2 逐局结果（20/20，全部真实退出码）

| # | seed | stage_clear | frames | deaths | bombs | wall_s | exit | stop_reason |
|---|---|---|---|---|---|---|---|---|
| 1 | 1000 | **true** | 13580 | 0 | 0.0 | 442.865 | 0 | stage_changed |
| 2 | 1001 | **true** | 13676 | 0 | 1.0 | 467.279 | 0 | stage_changed |
| 3 | 1002 | **true** | 13676 | 0 | 0.0 | 463.997 | 0 | stage_changed |
| 4 | 1003 | **true** | 13388 | 0 | 0.0 | 389.827 | 0 | stage_changed |
| 5 | 1004 | **true** | 13484 | 0 | 0.0 | 732.890 | 0 | stage_changed |
| 6 | 1005 | **true** | 13676 | 0 | 0.0 | 749.975 | 0 | stage_changed |
| 7 | 1006 | **true** | 13388 | 0 | 1.0 | 732.027 | 0 | stage_changed |
| 8 | 1007 | **true** | 13388 | 0 | 1.0 | 735.613 | 0 | stage_changed |
| 9 | 1008 | **true** | 13292 | 0 | 0.0 | 726.713 | 0 | stage_changed |
| 10 | 1009 | **true** | 15204 | 0 | 0.0 | 943.731 | 0 | stage_changed |
| 11 | 1010 | **true** | 13676 | 0 | 0.0 | 746.687 | 0 | stage_changed |
| 12 | 1011 | **true** | 13292 | 0 | 1.0 | 714.339 | 0 | stage_changed |
| 13 | 1012 | **true** | 13292 | 0 | 1.0 | 721.286 | 0 | stage_changed |
| 14 | 1013 | **true** | 13676 | 0 | 0.0 | 745.021 | 0 | stage_changed |
| 15 | 1014 | **true** | 13580 | 0 | 0.0 | 743.090 | 0 | stage_changed |
| 16 | 1015 | **true** | 13676 | 0 | 0.0 | 727.935 | 0 | stage_changed |
| 17 | 1016 | **true** | 12812 | 0 | 1.0 | 573.243 | 0 | stage_changed |
| 18 | 1017 | **true** | 13484 | 0 | 1.0 | 601.712 | 0 | stage_changed |
| 19 | 1018 | **true** | 13676 | 0 | 1.0 | 603.699 | 0 | stage_changed |
| 20 | 1019 | **true** | 13388 | 0 | 1.0 | 594.225 | 0 | stage_changed |

### 5.3 汇总（两个分母）

| 指标 | 值 |
|---|---|
| 申报局数 / 跑完局数 / 失败局数 | **20 / 20 / 0** |
| **S1N 过关率（跑完口径）** | **20/20 = 100.0%**，95% Wilson **[0.8389, 1.0000]** |
| **S1N 过关率（申报口径）** | **20/20 = 100.0%**，95% Wilson **[0.8389, 1.0000]** |
| 每局死亡（= 每面平均死亡，单面套件） | **0.0**（total 0，max 0，含死亡的局 0） |
| 平均帧数 | **13,565.2**（min 12,812 / max 15,204） |
| 帧上限利用 | 最大 15,204 = 上限 19,887 的 **76.4%**，无任何一局触到 `max_frames` |
| `exit_code` | **20 × 0** |
| `stop_reason` | **20 × `stage_changed`**（0 × `max_frames`、0 × `death`、0 × `engine_error`） |
| bomb | 9 颗（9 个 bomb 决策：1001/1006/1007/1011/1012/1016/1017/1018/1019 各 1；`bombs_initial=3` 每局） |
| 逐局墙钟合计 | **13,156.2 s**（3.65 h） |
| **批跑实际耗时** | **5,052.7 s = 84.2 min**（13:39:59 → 15:04:11，含串行段与并行切换） |
| 吞吐 | 串行段 4 局 3,630 s（≈908 s/局含切换）；并行段 16 局 ≈2,400 s（**≈150 s/局**，3.05× 于串行 460 s/局） |
| 实时倍率 | 串行段 0.515–0.577×；6 并行段 0.27–0.38×（单局变慢，总吞吐上升） |

**过关率门槛判定**：100% ≥ 90% **达标**；区间下界 83.9% < 90%，所以严格表述是「点估计 100%、95% 区间下界 83.9%；以 20 局样本无法在 95% 置信度下断言真实过关率 ≥90%，但观测到的 20/20 已满足门限的点估计判据」。这一点如实写在这里，避免把区间下界说成达标。

### 5.4 每局的 Boss 与死亡观测量

20 局全部一致：`boss_decisions` 1,893–2,478（= 过面决策数的 56–73%），`boss_first` 在决策 733 前后（帧 ≈2,936）以 **life 6,000** 出现，`boss_life_max`（报告级）= **60,000**，随后 boss 行 `life` 归零（每局 `boss_zero_life_rows ≥ 1`），再进入本面结束序列并在帧 12,812–15,204 之间把 `STATE.f1` 推到 1。`deaths_observed=0` 与 `frame_event_flags_player_hit=0` 一致 —— **Boss 是真被打死，不是超时/存活**。

### 5.5 并行不改变结果（同 seed 复现）

| seed | 串行 | 2 并行 | 一致？ |
|---|---|---|---|
| 1000 | 13,580 帧 / 0 死 / clear=true / 442.9 s | **13,580 / 0 / true / 460.8 s** | ✅ 逐位一致 |
| 1001 | 13,676 帧 / 0 死 / clear=true / 467.3 s | **13,676 / 0 / true / 459.1 s** | ✅ 逐位一致 |

证据：`.tmp_v400_g1/val/s1n_runs.jsonl`。⇒ 并行只改墙钟，不改轨迹。

### 5.6 关于中断的 seed 1004

串行段跑到第 5 局（seed 1004）时，为了切并行手动终止了该局，**该次运行没有产生结果记录**（教师只在结束时写 JSONL），随后在并行段以同一 CLI 重跑并计入：上表第 5 行的 13,484 帧 / 0 死 / clear=true 就是重跑结果。中断本身不构成数据点，也不进入任何分母。串行段因此只贡献 4 局（1000–1003）。

---

## 6. PSN 结果（8 面，逐面各自 RESET，normal，seed 1）

### 6.1 命令

每面一局、**各自 RESET 起跑**（`RESET <stage> 1 0 1 0`：normal、team 0、seed 1、practice=0），**不带** `--follow-stage`，所以换面即停；每 worker 一条独立 data-dir。以 stage 4 为例：

```bash
/root/v400-venv/bin/python tools/rl/v400_teacher.py --decisions 11258 --max-frames 45000 \
  --stop-on-death --stage 4 --difficulty normal --seed 1 --mode full --quiet \
  --out .tmp_v400_g1/psn/psn_stage4.jsonl --data-dir /root/v400-data/g1/inst4
```

`--decisions = cap//4+8`（每面不同），保证停止条件只可能是「过面 / 死亡 / 帧上限」。8 面同时跑（8 并行），驱动命令见 §9。

### 6.2 逐面结果（8/8，全部真实退出码）

| stage | 面 | cap | cap 来源 | clear | frames | deaths | bombs | Boss 证据 | exit | wall_s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Stage1 | 19,887 | `ceil(1.5×13,258)` | **true** | 13,676 | **0** | 0 | `life_reached_zero` 6,000→0 | 0 | 825.0 |
| 1 | Stage2 | 27,788 | `ceil(1.5×18,525)` | **true** | 20,684 | **0** | 2 | `life_reached_zero` 12,000→0 | 0 | 1,173.9 |
| 2 | Stage3 | 30,014 | `ceil(1.5×20,009)` | false | 21,308 | **1** | 3 | `life_reached_zero` 16,000→0（末段再起，停止时 3,725） | 0 | 1,298.7 |
| 3 | Stage4A | 41,583 | `ceil(1.5×27,722)` | false | 28,616 | **1** | 3 | `life_reached_zero` 60,000→0（停止时 **17**） | 0 | 2,031.2 |
| 4 | Stage4B | 45,000 | **fallback** | false | 12,908 | **1** | 3 | `present_at_stop` 60,000→15,897（min 367） | 0 | 904.4 |
| 5 | Stage5 | 38,513 | `ceil(1.5×25,675)` | false | 9,172 | **1** | 3 | `arena_empty_after_boss` 14,000→1,633 | 0 | 829.7 |
| 6 | Stage6A | 45,000 | **fallback** | false | 13,484 | **1** | 3 | `present_at_stop` 13,000→2,192（min **4**） | 0 | 1,097.6 |
| 7 | Stage6B | 54,605 | `ceil(1.5×36,403)` | false | 7,652 | **1** | 3 | `present_at_stop` 16,000→926 | 0 | 727.4 |

### 6.3 汇总

| 指标 | 值 |
|---|---|
| 申报 / 跑完 / 失败 | **8 / 8 / 0**（`partial=false`） |
| 过面数 | **2/8**（Stage1、Stage2） |
| **每面平均死亡** | **6/8 = 0.750**（阈值 ≤0.5 → **不达标**） |
| 死亡合计 / 有死亡的面 | 6 / 6（stage 2,3,4,5,6,7） |
| `exit_code` | **8 × 0** |
| `stop_reason` | 2 × `stage_changed`，**6 × `death`**，0 × `max_frames`，0 × `engine_error` |
| 逐面墙钟合计 | 8,887.9 s（148.1 min 机时） |
| **批跑实际耗时** | **2,031.2 s = 33.9 min**（15:09:15 → 15:43:06；8 并行 4.37× 加速） |
| 上限利用 | 只有 stage 3 跑到上限的 68.8%（28,616/41,583）；没有一面撞 `max_frames` |

### 6.4 死亡点解剖（`stage_frame` / boss / 弹幕密度都是引擎导出的，不是推断）

| stage | 面 | 死亡帧 | 死亡决策 | 面内帧 | 停止时 Boss life | 场上敌人 | tape 盒均值（min–max） | 剩余 bomb | 停止帧 `EVENTS` |
|---|---|---|---|---|---|---|---|---|---|
| 2 | Stage3 | 21,308 | 5,326 | 21,310 | **3,725（Boss 活着）** | 10 | 55.7（1–94） | **0** | `0x400` |
| 3 | Stage4A | 28,616 | 7,153 | 28,618 | **17（Boss 只剩 17 HP）** | 10 | 460.1（20–682） | **0** | `0x2400` |
| 4 | Stage4B | 12,908 | 3,226 | 12,910 | 15,897（60,000 的第 1 阶段） | 12 | 258.9（11–404） | **0** | `0xE02`（含 `0x2 PLAYER_HIT`） |
| 5 | Stage5 | 9,172 | 2,292 | 9,174 | 已离场（14,000→1,633 后 arena 空 30 决策） | 18 | 87.2（0–138，17 帧无盒） | **0** | `0x2E02`（含 `0x2`） |
| 6 | Stage6A | 13,484 | 3,370 | 13,486 | **2,192（min 4）** | 1 | 73.2（1–122） | **0** | `0x602`（含 `0x2`） |
| 7 | Stage6B | 7,652 | 1,912 | 7,654 | 926（16,000 的第 1 阶段） | 17 | 300.8（1–531） | **0** | `0x0` |

口径说明：`EVENTS` 的 `frame_event_flags` 里只有 `0x2 = PLAYER_HIT` 有客户端定义（`label.py:40`），其余位**不解释**，只报原始值。每局死亡时 `player_state(f13)=2`、`bombs=0`、活着过面的两面 `f13=1`。

**共同点（可观测、可复现）**：6 次死亡**全部发生在 3 颗 bomb 已用完之后**（`bombs_spent=3.0`，`bombs_left=0`）。也就是说这些死亡不是"有 bomb 没用"，而是**炸弹预算先耗尽、之后被弹幕压死**；死亡位置 `py≈416–432`（游戏区底边），与"贴底保守走位"一致。

---

## 7. G1 判定

| 门 | 指标 | 阈值 | 实测 | 判定 |
|---|---|---|---|---|
| **S1N** | stage 0 过关率（20 局） | ≥ 90% | **20/20 = 100.0%**，95% Wilson **[83.89%, 100%]** | ✅ **达标**（点估计；区间下界 83.9% 见 §5.3 的严格表述） |
| **PSN** | 每面平均死亡（8 面） | ≤ 0.5 | **6/8 = 0.750** | ❌ **不达标** |
| **G1** | 两项都要过 | 全部达标 | S1N ✅ / PSN ❌ | ❌ **G1 未达标** |

两个分母口径（S1N）：

| 口径 | 分母 | 分子 | 过关率 |
|---|---|---|---|
| 实际跑完的局 | 20 | 20 | **100.0%** |
| 全部申报局 | 20 | 20 | **100.0%** |

PSN 的分母没有歧义：申报 8 面，跑完 8 面，失败 0 面。

**为什么 PSN 不达标（一句话）**：教师 v1 的**过面能力到 Stage2 为止**——Stage1/Stage2 都能 0 死亡过面，但从 **Stage3 开始每一面都死 1 次**，且 6 次死亡时 3 颗 bomb 全部用尽。它不是"不会过面"，而是**在第 3 面之后的中后段没有存活余量**。

---

## 8. 失败清单（首个失败点）

按"第一个没过的门"排序。G1 的首个失败点是 **PSN 的 stage 2（Stage3）**：S1N 全过之后，这是第一个 `deaths > 0` 的面。

| # | 面 | 首个失败点 | 位置（帧 / 决策） | 死因可观测信息 | 离过面还差 |
|---|---|---|---|---|---|
| 1 | **Stage3**（stage 2） | Boss 战第 2 阶段末段 | 帧 **21,308** / 决策 5,326（面内 21,310） | Boss 活着 life 3,725；场上 10 敌；tape 盒 1–94/帧；bomb 0；`f13=2` | 死在第 1 个上限（30,014）的 71.0%；比人类 20,009 帧多活 1,299 帧后死 |
| 2 | **Stage4A**（stage 3） | **Boss 只剩 17 HP** | 帧 **28,616** / 决策 7,153 | Boss life 17（60,000→0→…→17）；10 敌；tape 盒 20–682/帧（本批最密）；bomb 0；`0xE02` 含 `PLAYER_HIT` | 差 17 HP 就过面；人类 27,722 帧，教师多活 894 帧 |
| 3 | **Stage6A**（stage 6） | Boss 末段 | 帧 **13,484** / 决策 3,370 | Boss life 2,192（本局曾到 **min 4**）；1 敌；bomb 0；`0x602` 含 `PLAYER_HIT` | 无人类参考（fallback 上限 45,000），是上限的 30.0% |
| 4 | **Stage6B**（stage 7） | Boss 第 1 阶段早段 | 帧 **7,652** / 决策 1,912 | Boss life 926（16,000）；17 敌；tape 盒 1–531/帧；bomb 0 | 人类 36,403 帧，教师只活到 21.0% |
| 5 | **Stage5**（stage 5） | **Boss 离场后的道中** | 帧 **9,172** / 决策 2,292 | Boss 已离场（14,000→1,633 后 arena 空 30 决策）；18 敌；17 帧无致死盒；bomb 0；`0x2E02` 含 `PLAYER_HIT` | 人类 25,675 帧，教师只活到 35.7% |
| 6 | **Stage4B**（stage 4） | Boss 第 1 阶段 | 帧 **12,908** / 决策 3,226 | Boss life 15,897/60,000；12 敌；bomb 0；`0xE02` 含 `PLAYER_HIT` | 无人类参考（同类面 4A 的 1.5× = 41,583，只跑到 31.0%） |

**非失败的结构性缺口**（不是"没过"，是"测不了"）：连续多面推进（一次进程内 0→1→2）受 §4 的 `PARK` 限制无法测量；本轮用逐面 RESET 绕开，所以 PSN 的数字是**逐面口径**，不能当成"一命通关到第 8 面"的成绩。

---

## 9. 命令与退出码（全部真实）

| # | 命令 | 退出码 | 说明 |
|---|---|---|---|
| 1 | `python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests/test_v400_g1_eval.py` | **0** | **17 passed** |
| 2 | `/root/v400-venv/bin/python tools/rl/v400_teacher.py --probe-only --stage 0 --difficulty normal --seed 1 --mode full` | **0** | SHA `f172399f…` match，`child_hazards.available=true`（`.tmp_v400_g1/probe.sh`） |
| 3 | `bash .tmp_v400_g1/run_s1n.sh`（串行段 seed 1000–1003） | 被我中断（无退出码） | 见 §5.6；已产出 4 条记录 |
| 4 | `bash .tmp_v400_g1/stop_serial.sh` | **0** | 停止串行批；残留进程 `none`；记录保留 4 条 |
| 5 | `bash .tmp_v400_g1/setup_data_dirs.sh` | **0** | `DATA_DIRS_OK`，8 个 data-dir |
| 6 | `bash .tmp_v400_g1/val_parallel.sh`（2 并行验证） | **0** | `VAL_EXIT=0`，与串行逐位一致 |
| 7 | `bash .tmp_v400_g1/run_s1n_parallel.sh`（6 并行，seed 1004–1019） | **0** | `S1N_PAR_EXIT=0`，20/20 记录齐全 |
| 8 | `bash .tmp_v400_g1/run_psn_parallel.sh`（8 并行） | **0** | `PSN_EXIT=0`，8/8 记录齐全 |
| 9 | `/root/v400-venv/bin/python tools/rl/v400_g1_eval.py summary` | **0** | 写 `g1_summary.json`（`.tmp_v400_g1/g1_summary.log`） |
| 10 | `/root/v400-venv/bin/python .tmp_v400_g1/psn_death_diag.py` | **0** | §6.4/§8 的死亡解剖数据来源 |

逐局退出码：S1N **20 × 0**；PSN **8 × 0**；合计 **28 次真实运行、28 次 exit 0**，没有一次 engine_error / fail_closed / 超时。

产物清单：

| 路径 | 内容 |
|---|---|
| `.tmp_v400_g1/s1n_runs.jsonl` | S1N 20 条记录（命令、seed、data-dir、并行度、墙钟、exit、帧、死亡、bomb、Boss 证据） |
| `.tmp_v400_g1/psn_runs.jsonl` | PSN 8 条记录（同上 + cap 来源） |
| `.tmp_v400_g1/s1n/*.jsonl`（20 个，224 MB） | 教师逐决策原始日志 |
| `.tmp_v400_g1/psn/*.jsonl`（8 个） | 同上 |
| `.tmp_v400_g1/logs/*.log` | 每局子进程 stdout/stderr |
| `.tmp_v400_g1/s1n_summary.json` / `psn_summary.json` / `g1_summary.json` | 汇总（含 Wilson 区间、两个分母） |
| `.tmp_v400_g1/val/s1n_runs.jsonl` | 2 并行验证记录 |
| `.tmp_v400_g1/s1n_batch.log` / `s1n_parallel.log` / `psn_parallel.log` / `val_batch.log` | 批跑进度流 |

---

## 10. 局限与声明

1. **`partial=false`**：两套测量都按申报跑满（S1N 20/20，PSN 8/8），没有未跑面。唯一的中断是 §5.6 的 seed 1004 串行尝试，已重跑且不计入任何分母。
2. **S1N 的区间**：20/20 的 95% Wilson 下界是 83.9%，低于 90% 门限。所以严格说法是「观测点估计 100% ≥ 90%，但 20 局样本不足以在 95% 置信度下断言真实过关率 ≥90%」。本卡不把区间下界写成达标。
3. **PSN 的 cap 口径**：stage 4 / stage 6 没有 normal 人类参考，用任务裁定的 **45,000 fallback**（同类面 proxy 会给出 41,583 / 54,605，已在 §2 与 `psn_runs.jsonl` 的 `proxy_note` 里列出）。其余 6 面全部是 `ceil(1.5×人类帧数)`，没有一面按结果调参。
4. **PSN 是逐面口径**：每面各自 RESET，seed 固定 1。它测的是"该面能不能过"，不测"一命从第 1 面打到第 8 面"；后者受 §4 的 `PARK` 限制当前不可测。
5. **并行只改墙钟**：6/8 并行下单局墙钟升到 727–2,031 s，但同 seed 复现（2 并行 vs 串行）逐位一致；所有 run 都记录了自己的 `parallel_jobs`、`worker`、`data_dir`。
6. **`deaths` 口径**：`STATE` 死亡计数增量，不用残机推算。S1N 20 局 `deaths=0` 与 `frame_event_flags_player_hit=0` 一致。
7. **Boss「被打死」的判据**：`ENEMIES` 里 boss 行 `life` 归零（`boss_zero_life_rows ≥ 1`）或 Boss 消失后 arena 连续 ≥3 决策清空；两者都没有时写 `present_at_stop`，不写"打死"。
8. **记录字段的先后差异**：串行段 4 条记录（seed 1000–1003）写于并行改造之前，因此没有 `parallel_jobs` / `worker` / `data_dir` / `display` 四个字段；它们的 data-dir 是教师默认 `/root/v400-data/p1s1/game`（命令行里没有 `--data-dir`，这一点在记录的 `command` 字段里可见）。并行段 16 条记录全部带这四个字段。本轮**没有**回填旧记录，避免事后改写原始产物。
9. **引擎与协议未改**：本轮只新增 `tools/rl/v400_g1_eval.py` 与 `tests/test_v400_g1_eval.py`，`tools/rl/v400_teacher.py` 的 diff 为空；引擎 SHA-256 仍是 `f172399f61b75baa…`（与 EXP-P1-011 相同），未重建、未改协议、未动第三方。


