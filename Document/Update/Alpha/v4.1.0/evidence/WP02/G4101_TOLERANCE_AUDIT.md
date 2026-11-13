# G410-1 轨迹一致性 8 px 校准审计（seed 1000 / fast / stage 0 / normal）

- 工作包：v4.1.0 **G410-1** 的"教师/验证器轨迹一致性"审计——**只审计**，不训练、不采确认种子、不改 v400 源码
- 只读输入：`tools/rl/v400_teacher.py`、`touhou_rl/v400/teacher/decide.py`、`touhou_rl/v400/teacher/verify.py`、`Document/Update/Alpha/v4.1.0/evidence/WP02/fast20.jsonl`（seed 1000 raw）、WP02.4 [`CALIBRATION.md`](CALIBRATION.md) / `labels_calibrated_summary.json`
- 运行目录（**不入库**）：`.tmp_v410_g4101_tol8/`（wrapper、teacher JSONL、日志、分析 JSON）
- 机器可读证据：[`g4101_tolerance_audit.json`](g4101_tolerance_audit.json)（schema `v410-g4101-tolerance-audit/1`）
- 状态：**审计完成**。这是对"raw 1e-3 mismatch 是阈值噪声还是教师/验证器真实问题"的一次实测判定，**不是 G410-1 通过**，也**不是**新标签/新数据集。

---

## 0 一句话结论

在 dev seed 1000（fast / stage 0 / normal / `rl_laser` 1462829b…）上，把 `verify_plans` 的 `tolerance` 临时设为 **8.0 px** 实测重跑：`trajectory_mismatch` 从 raw 1e-3 的 **2,456/3,251（75.546%）** 降到 **25/3,251（0.769%）**，`geo_dead_engine_alive=0`、`geo_alive_engine_dead=0` 不变，局仍然 **stage_clear=true / 13,004 帧 / 0 死亡**。**raw 1e-3 的 75.5% 绝大部分是"模型-引擎位置数值漂移"（中位 3.414 px）而不是"执行与验证的死亡判定矛盾"**；残余 25 条中 17 条是 8 px 边界的浮点尾巴、8 条是 stage clear 前最后 8 个决策的真实模型漂移（139.58 px）。这个结论**不能**替代 G410-1 的 100 确认种子正式口径。

---

## 1 范围与不变量

- **未改任何源码**：8 px 是运行时内存 patch，仓库里 `verify.py::POSITION_TOLERANCE` 仍是 `1e-3`，`decide.py` 一字未动。
- **未训练、未采数据集、未使用/未运行任何确认种子**（61300–61399 一个都没跑）；只阅读了既有 `G4101_CHUNK1/2` 文档用于对齐口径（文中出现种子号，但本审计没有运行它们、也没有引用其结论）。
- **未写** `state.json` / `journal.jsonl` / `budget.jsonl` / `assets.json`；**未 commit/push**；未触碰其它 WP 证据与 v410 既有文件。
- 新增/修改仅白名单内：本文件、`g4101_tolerance_audit.json`、运行目录 `.tmp_v410_g4101_tol8/`（gitignored）。
- 引擎：`/root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern`，SHA256 `1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f`（报告 `engine.match=true`，启动前 SHA 门通过）。
- 数据目录：`/root/v400-data/g1/inst0`（与 WP02 fast20 的 worker 0 相同）；`/root/v400-data/game` 未被本审计写入。

---

## 2 实际命令与退出码（真实执行）

### 2.1 8 px 是怎么打进去的（不改源码）

`.tmp_v410_g4101_tol8/run_tol8.py`（运行目录内）在 `runpy.run_path("tools/rl/v400_teacher.py", run_name="__main__")` **之前**执行：

```python
decide_module.verify_plans = functools.partial(verify_module.verify_plans, tolerance=8.0)
```

`decide.py` 在调用点按模块全局名解析 `verify_plans`，且该模块已进 `sys.modules`，所以 CLI 后续的 `from ...decide import decide` 复用同一份被 patch 的模块。wrapper 还做了两件事，避免"以为 patch 了其实没有"：

1. 包一层计数器，记录**真正到达** `verify.verify_plans` 的调用数、`tolerance` kwarg 与计划数；未达标则 exit 90/91。
2. 用 `importlib.import_module` 取子模块（`touhou_rl.v400.teacher.__init__` 把函数 `decide` 重导出，遮蔽了同名子模块属性，`import ... as m` 会绑到函数上——已绕开）。

### 2.2 命令

```bash
cd /mnt/d/WorkShop/Touhou_RL
bash .tmp_v410_g4101_tol8/run_tol8.sh
# 内部：PYTHONPATH=/mnt/d/WorkShop/Touhou_RL \
#       /root/v400-venv/bin/python -u .tmp_v410_g4101_tol8/run_tol8.py > .tmp_v410_g4101_tol8/run_tol8.log 2>&1
```

wrapper 展开后下发给 CLI 的 argv（`tol8_patch_meta.json` 原文记录）：

```text
--decisions 10000 --max-frames 19887 --stop-on-death --stage 0 --difficulty normal \
--seed 1000 --mode fast --quiet \
--out /mnt/d/WorkShop/Touhou_RL/.tmp_v410_g4101_tol8/seed1000_tol8.jsonl \
--data-dir /root/v400-data/g1/inst0 \
--engine /root/Touhou_RL/third_party/th08-rl/build/rl_laser \
--binary /root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern \
--expected-sha 1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f
```

与任务给定命令的**唯一差异**：`--out` 用了绝对路径（cwd 固定为 repo 根，与相对路径等价），其余参数逐字一致；没有改任何阈值/参数（`tolerance` 只在内存 patch 里，argv 里不出现）。

### 2.3 退出码与耗时

| 项 | 值 | 来源 |
|---|---|---|
| 教师进程退出码 | **0** | `tol8_patch_meta.json: exit_code` |
| 驱动脚本 `run_tol8.sh` 退出码 | **0** | `run_meta.json: exit_code` / `wsl.exe` 透传 `WSL_EXIT=0` |
| 教师自身 wall | **298.891 s** | `tol8_patch_meta.json: wall_s` |
| 脚本墙钟 | **299 s**（12:02:40 → 12:07:39 +0800） | `run_meta.json` |
| 预检 smoke（2 decisions / 8 帧，另跑一次） | 退出 0；当时 wrapper 输出 `verify_plans_calls=2`、`tolerances=[8.0]` | `smoke.jsonl`（3 行：1 report + 2 decision） |

> 299 s 比 WP02 fast20 的 452.6 s 短，只说明当时机器负载不同（另有 WP05 `dagger` 采集在跑），**不是**策略或引擎的变化。

### 2.4 patch 确实生效的运行时证据

| 证据 | 值 |
|---|---|
| `verify_plans_calls` | **3,251**（= 决策数） |
| `verify_plans_tolerances` | **[8.0]**（只有一个值） |
| `verify_plans_plans_seen` | **3,251** |
| JSONL 里 `trajectory_matches` 与 `(max_position_error <= 8.0)` 不一致的条目 | **0 / 3,251** |
| `patch_still_installed_after_cli` | true |

---

## 3 seed 1000 本局结果（8 px）

| 指标 | 值 |
|---|---|
| `status` / `stop_reason` | `ok` / `stage_changed` |
| `stage_clear` | **true**（`stage_at_stop=1`，`stage_transitions` 1 条：decision 3250、frame 13004、stage 0→1、stage 内 0 死亡） |
| **frames**（`frames_committed`/`survival_frames`） | **13,004**（两者相等；216.733 游戏秒） |
| decisions | **3,251**（`--decisions 10000` 未用尽，被过面终止） |
| deaths | **0**；`bombs_spent=0.0`、`bomb_decisions=0` |
| plans_verified | **3,251**（`verify_k=1`，每决策 1 条） |
| `died=true` 计划 | **0** |
| `decisions_per_second` / `realtime_multiple` | 11.2313 / 0.7488 |
| config | `horizon=32, beam=24, verify_k=1, max_iterations=1, durations=[1,2]` |

与 WP02 fast20 的 seed 1000 行（`decisions=3251`、`frames=13004`、`deaths=0`、`stage_clear=true`、`stop_reason=stage_changed`）**逐项相同**。

---

## 4 raw 1e-3 vs 8 px（同一局面、同一 3,251 条计划）

raw 一列 = WP02 fast20 seed 1000 的 `consistency`，并用同一份语料 `.tmp_v410_wp02_audit/teacher1000.jsonl`（sha256 `a1d6923e…2f434b`，即 WP02.4 校准用的那份，11,931,643 B）按计划级 `max_position_error` 复算；8 px 一列 = 本次重跑的 JSONL。

| 度量 | raw 1e-3 | **8.0 px（本次实跑）** | Δ |
|---|---|---|---|
| plans_verified | 3,251 | **3,251** | 0 |
| `trajectory_mismatch` | **2,456** | **25** | **−2,431** |
| mismatch 率 | **75.546%** | **0.769%** | **−74.777 pp** |
| `geo_dead_engine_alive` | 0 | 0 | 0 |
| `geo_alive_engine_dead` | 0 | 0 | 0 |
| `agree_alive` / `agree_dead` | 3,251 / 0 | 3,251 / 0 | 0 |
| `died=true` 计划 | 0 | 0 | 0 |
| WP02.4 校准口径 alive（`died=false` 且 `error ≤ 8`，两列都按计划级误差在各自语料上复算） | 3,226 | **3,226** | 0 |
| moving_alive / idle_alive | 1,704 / 1,522 | **1,704 / 1,522** | 0 |

> raw 一列的 3,226 **不是** WP02.3 的 795：795 就是 1e-3 那一行（`trajectory_matches=true` 且 `error ≤ 1e-3`），这里把同一份 raw 计划按 8 px 重新分类，所以 raw 与 8 px 两列在 8 px 口径下必然都是 3,226——它证明的是"两列是同一批计划、同一分类器"，不是两次不同的采样。

> 8 px 的 3,226/3,251 = **99.231%**，与 WP02.4 的覆盖率逐位一致；**25 条 mismatch 就是 WP02.4 的 25 条 `unknown`**，不是另一批计划。

### 4.1 阈值表（从本局计划级 `max_position_error` 复算，与 WP02.4 §5 对照）

| threshold (px) | covered | coverage | mismatch | moving covered | moving coverage | WP02.4 记录 | 一致 |
|---|---|---|---|---|---|---|---|
| 0.001 | 795 | 24.454% | 2,456 | 0 | 0.00% | 795 / 24.454% | ✔ |
| 0.5 | 795 | 24.454% | 2,456 | 0 | 0.00% | 795 | ✔ |
| 1.0 | 795 | 24.454% | 2,456 | 0 | 0.00% | 795 | ✔ |
| 2.0 | 1,217 | 37.435% | 2,034 | 44 | 2.560% | 1,217 / 37.435% | ✔ |
| 4.0 | 2,203 | 67.764% | 1,048 | 766 | 44.561% | 2,203 / 67.764% | ✔ |
| **8.0** | **3,226** | **99.231%** | **25** | **1,704** | **99.127%** | 3,226 / 99.231% | ✔ |
| 16.0 | 3,243 | 99.754% | 8 | 1,719 | 100.000% | 3,243 / 99.754% | ✔ |

误差分布（3,251 条）：min 0、中位 **3.414212**、均值 3.304276、p90 6.0、p95 6.828424、max **139.583771**；`error == 0` 恰好 795 条（全部 idle）。

### 4.2 残余 25 条不是一类东西

| 类别 | 条数 | 特征 |
|---|---|---|
| 8 px 浮点边界 | **17** | `8.0 < error < 8.001`（8.000003–8.000015），`died=false`、`frames_advanced=32` |
| stage clear 窗口的真实漂移 | **8** | `error = 139.583771`，decision **3243–3250**（frame 12,976–13,004），全部 `first_action=0`、`duration=1`，就在 stage 0→1 前最后 8 个决策 |

- 刚好落在 **8.0** 上的还有 **31** 条（按"边界包含"算 covered），另有 **6** 条 `7.999994`、**17** 条 `8.000003–8.000015`：**54 条挤在 8 px 线 ±1e-5 px 内**——8.0 是"刀刃"位置，不是空档。
- 17 条边界条目只超阈值 3e-6–1.5e-5 px，属数值尾巴；**8 条 139.58 px 是真实模型漂移**：stage 即将切换时几何模型不再复现引擎位置（模型不建模过场/换 stage 的那几帧）。这 8 条在本局**没有**造成死亡误判（`died=false`、局也过面），但它是"模型在 stage 边界失效"的可复现证据，**不能**归因于阈值噪声。

---

## 5 确定性：8 px 重跑就是一局干净的独立重放

`tolerance` 只进入 `VerifiedPlan.trajectory_matches`（以及 `Consistency` 计数），不参与计划选择、不参与提交。逐计划比对 `.tmp_v410_wp02_audit/teacher1000.jsonl` 与本次 `seed1000_tol8.jsonl`：

| 项 | 结果 |
|---|---|
| decision 行数 / 计划数 | 3,251 / 3,251 对 3,251 / 3,251（相等） |
| `first_action`、`duration`、`died`、`frames_advanced`、`max_position_error`、`end_xy` 差异 | **0**（3,251 条全部相等） |
| `trajectory_matches` 翻转 | **2,431**（原 false → 8 px 下 true；25 条 >8 仍 false） |
| 报告字段（status/stop_reason/stage_clear/stage_at_stop/decisions/frames/survival_frames/deaths/bombs/seed/stage/difficulty/mode） | **无差异** |

也就是说：本次实跑与 raw run **是同一局的两个判定**，差异只在容忍阈值，符合"tolerance 仅影响一致性度量"的代码语义（`verify.py:198`、`decide.py:204-207`）。

---

## 6 结论（谨慎版）

1. **raw 1e-3 的 75.5% 主要是严格阈值噪声，不是死亡判定矛盾**：同一 3,251 条计划在 8 px 下 mismatch 只剩 25 条（0.769%），位置误差中位 3.414 px——1e-3 量级根本不是这批模型-引擎同步的实际精度。`geo_dead_engine_alive` / `geo_alive_engine_dead` 两种口径都恒为 0，`agree_alive=3,251`，**没有任何"几何说活/引擎说死"或反向的矛盾**。
2. **但它不是"纯噪声到 0"**：残余 25 条里 8 条是 stage clear 前最后 8 个决策的 **139.58 px** 真实漂移（模型不重建换 stage 的那几帧），17 条是 8 px 的浮点边界。因此更准确的说法是"1e-3 的 75.5% 里，绝大多数是数值漂移；另有 8/3,251 = 0.246% 是模型在 stage 边界上的真实失效"。
3. **不构成 G410-1 通过**：只有 dev seed 1000 一局；G410-1 的教师腿口径是 61300–61399 共 100 个确认种子，且 `g4101_s1n_chunk1/2` 已经显示 raw mismatch 在 80% 量级、`v410_teacher_fast20.py` 的 `verification_ok` **不检查** `trajectory_mismatch`。100 种子正式口径必须同时报 raw 1e-3、8 px、geo 矛盾与具体 seed/plan，不能拿本文件替代。
4. **死亡语义这次没被真正检验**：本局 `died=true` 计划 0 条、局内死亡 0，所以"无矛盾"只覆盖了"两边都说活"这一格；`agree_dead` 与 `geo_alive_engine_dead` 的正例来自 WP02.5 的 7 条 `verified_dead` 与后续有死亡的 run，不是这次。
5. **`v2h` 不能当这个度量的解**：`POSITION_TOLERANCE` 在 `verify.py` 里被 full/fast/v2a…v2h 所有 profile 共用，换 v2h（`verify_k=3`）不会改变 1e-3 的语义，只会让每决策多验 2 条计划、mismatch **条数**按比例变大；**率**是计划构成的属性、可能略有不同，但不会因为换 profile 就消失。要动 1e-3 语义只能改 `verify.py`（v400 变更，需单独立项）或在标签层用 WP02.4 的 8 px（已存在）。

---

## 7 限制

1. **单种子/单 stage/单难度/单 profile**：seed 1000、stage 0（Stage 1）、normal、`fast`（`horizon=32, beam=24, verify_k=1`）。本审计不回答其它 seed/stage/难度、也不回答 `full`/`v2h` 的漂移分布。
2. **8 px 不是生存阈值**：它是 WP02.4 预注册的**模型一致性**阈值；`died` 才是生存真值。8 px 内不等于"一定不死"，>8 px 也不等于"会死"（本次 25 条全 `died=false`）。
3. **0 死亡使"无矛盾"变窄**：见 §6.4。
4. **边界敏感性**：31 条恰好 8.0、17 条只超 1e-5 px；若有人把阈值改成 8.001，mismatch 会变成 8 条（只剩 stage 边界那 8 条）。这本文件不给"应该改成多少"的建议，只记录 8.0 是预注册值、且是刀刃位置。
5. **wall 不可比**：运行窗口内另有 WP05 `dagger` 采集（`rl_v3` 引擎、不同数据目录、约 10 核）在跑；299 s 反映负载，不是策略性质。
6. **patch 只在内存里**：仓库的 `POSITION_TOLERANCE` 仍是 1e-3；任何不经过本 wrapper 的新 run 仍会写出 raw 口径的 `trajectory_matches`。
7. **一次性重放**：本局与 raw 的"逐计划相等"是同一引擎 SHA + 同 seed + 同数据目录下的确定性结果；换数据目录内容或引擎构建必须重测。

---

## 8 下一建议（不改变本审计结论）

1. **G410-1 教师腿的正式口径**：在 100 确认种子批次里，除现有 `consistency` 外，**同时输出**每条计划的 `max_position_error` 分布（或直接跑一个 8 px 版本），并**并列报告** raw 1e-3 / 8 px / geo 矛盾 / 具体 seed+plan；不要只报 `verify_k` 计数。
2. **把"执行与验证一致"定义在正确的地方**：门应落在 `geo_dead_engine_alive == 0` 且 `geo_alive_engine_dead == 0`（死亡语义），加上 WP02.4 的"`error ≤ 8` 才算干净正样本"（标签语义）；把 1e-3 当门会让 75% 的页面永远 fail。
3. **chunk3/4 前不需要修 tolerance**：0.769% < 5% 已说明继续用 fast 采集是可接受的；若要花工程预算，优先看 §4.2 的 8 条 stage 边界漂移（模型不重建换 stage 帧），而不是把 8.0 往上调。
4. **可选的加固**：在另一个 dev seed（1000–1019 内，不碰确认种子）复跑本 wrapper，确认 0.769% 与 25 条残差的构成是 seed 级稳定现象；本审计只做了 seed 1000。
5. **若未来要改 `verify.py::POSITION_TOLERANCE`**：那是 v400 源码变更，需独立工作包 + 回归（`tests/` 里对 `trajectory_matches` 语义的断言会一起变），本审计**没有**改它。

---

## 9 产物清单与未改动项

产物（白名单内）：

| 文件 | 字节 | sha256 |
|---|---|---|
| `Document/Update/Alpha/v4.1.0/evidence/WP02/G4101_TOLERANCE_AUDIT.md` | 本文件 | 自引用不写入 |
| `Document/Update/Alpha/v4.1.0/evidence/WP02/g4101_tolerance_audit.json` | — | 自引用：本文件内含 md 的 sha256；json 自身 sha256/字节数不写入自身 |
| `.tmp_v410_g4101_tol8/seed1000_tol8.jsonl`（不入库） | 11,918,830 | `bd2a30a88e0057da0bbb4855a8b4c37578d144016cd703608fb79b7f4a01c341` |
| `.tmp_v410_g4101_tol8/run_tol8.py` / `run_tol8.sh` | 6,654 / 954 | wrapper（内存 patch，不改源码） |
| `.tmp_v410_g4101_tol8/run_tol8.log` | 6,059 | `9d92706cc1810e43b9c5cbcce9698cbc5c9adc1ea307f3f1473054c942e18aee` |
| `.tmp_v410_g4101_tol8/tol8_patch_meta.json` / `run_meta.json` | 1,542 / 350 | patch 证明 / 退出码与墙钟 |
| `.tmp_v410_g4101_tol8/tol8_analysis.json` / `tol8_vs_raw.json` / `raw_analysis.json` / `residuals_gt8px.json` / `boundary_stats.py` / `analyze_tol8.py` / `compare_tol8_raw.py` / `dump_residuals.py` / `validate_evidence.py` / `smoke.jsonl` / `git_status_before.txt` / `git_status_after.txt` | — | 分析中间件与自查（不入库；`validate_evidence.py` 67 项全 PASS） |

未改动：`touhou_rl/v400/`（含 `teacher/verify.py` 的 `POSITION_TOLERANCE=1e-3`）、`touhou_rl/v380/`、`v390/`、`v410/` 既有文件、`tools/rl/v400_*`、`tools/rl/v410*.py`、`tests/`、`third_party/`、v4.1.0 既有证据（`fast20*`、`CALIBRATION.md`、`G4101_CHUNK1.md`、`G4101_CHUNK2.md`、`AUDIT.md`、`LABELS.md`、`STATUS.md`、`audit.json`、`labels_*`、`dead_sample*`）与其它 WP 证据；`state.json` / `journal.jsonl` / `budget.jsonl` / `assets.json`；未 commit/push。

---

## 10 不宣称

- **不是 G410-1 通过**：没有评估任何 gate/阈值/发布检查；100 个确认种子未运行、未采集、未使用（只在既有 chunk 文档里读到过口径与种子号）。
- **不是新数据集/新标签**：只是同一 dev seed 1000 在 8 px 下的重放，没有训练、没有采集、没有产生训练样本。
- **不是生存真值**：8 px 是模型一致性阈值；`died` 才是生存判定，本局 0 死亡。
- **不是"教师没问题"**：8 条 stage 边界 139.58 px 的模型漂移是可复现的教师/模型侧缺陷（虽未造成误判）。
