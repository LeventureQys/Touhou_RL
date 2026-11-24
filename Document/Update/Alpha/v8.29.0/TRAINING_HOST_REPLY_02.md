# V829-PAIR-01 §3 实施回执｜v8.29 配对入口 + 旧窗冒烟

**DeepSeek 训练端 → 用户转交 → GPT 远端研究端**

| 项 | 值 |
| --- | --- |
| 检查日期/时区 | 2026-11-23，`+08:00`（Asia/Shanghai，UTC+8） |
| 实测窗口 | 2026-11-23 11:30:45 → 11:34:16（+0800） |
| 主机代号 | `DSH-WIN-01`（Windows 侧）/ `DSH-WSL-01`（WSL2 Ubuntu，同一台机器） |
| 依据 | [`PREREGISTRATION_PAIR_001.md`](PREREGISTRATION_PAIR_001.md) §1、§3.1–§3.3、§4 |
| 性质 | **训练端在用户授权下的自发实施与冒烟**，不是研究端评测工单的回执。冒烟**不产生**任何新祖先的策略结论，只验收「适配器实现是否正确」 |
| 当前状态 | §3.1 保真门 ✅、§3.2 入口与清单读取器 ✅、§3.3 冒烟 ✅（两档各 1 对）；**新起点仍为 0**（见上一份回执 §3） |
| 只读/安全合规 | 引擎本次**是授权启动的**（冒烟必需）；启动前 `pgrep` = 0、结束后 = 0；**音频三件事全做**（见 §1.4）；未采集、未训练、未改删既有资产、未 `git add/commit`、未上传 |

> **一句话**：v8.29 的配对入口已经在旧窗上真跑通 —— 两次独立恢复逐位一致、两臂 RNG 同起点、每决策采样一次、帧级扣雷事件终于落盘；同时暴露出一个**有科学含义**的现象：在 40 秒档、本窗口、本 checkpoint 上，G1 比 T2 多发的 5 次请求**全部落在无敌帧、全部不兑现**，因此两臂状态逐决策相同、死在同一帧。

---

## 1. 交付物与验收（逐条对预注册 §3.1–§3.3）

### 1.1 新增文件（两个，均未提交、未登记）

| 文件 | 行数 | 字节 | SHA-256 |
| --- | ---: | ---: | --- |
| `tools/rl/v829_guard.py` | 1,147 | 54,154 | `8ec8517e73e1f0de5c2ca077d86a4dab93eb443afd710a703e8a8de47bf0fce5` |
| `tools/rl/v829_pair.py` | 3,229 | 162,962 | `d33cc38326e6068fb9746a8874850bfb8a5d8da6ae6d5b61928b370c1b3f0dbb` |

**`v829_guard.py`**：配对清单 `schema = v829-pair-manifest/1` 的 fail-closed 守卫，11→**13** 个错误码，`selftest` **34/34**。
**`v829_pair.py`**：四模式 `check / trace / pair / compare`；只复用既有已审计代码，**不复制决策循环**：
`v450_cf_label.start_parent`（start+HELLO+前缀重放+PARK）、`v450.prepare_data_dir`（私有 data dir + 静音 cfg）、
`v830_bombpol.run_episode36`（唯一决策循环）、`v8r0_urgency.thresholds_for`（G1 阈值）、`v8o0_ruletight.thresholds("T2")`（T2 阈值）。

清单（新路径 `.tmp_v829_pair/manifest/`）：

| 清单 | 字节 | SHA-256 | 档位 |
| --- | ---: | --- | ---: |
| `smoke-old-window.json` | 4,103 | `a84faca53333c0cb6bf66c5b914dfa1d0defbb01d5037e6429ad9dbf780b9434` | 2400 帧 / 1200 决策 |
| `smoke-old-window-1800.json` | 4,311 | `6a1f9fd326d64171b89866c3550f87134177d16514f0a5a144e72a3e266ebf90` | 1800 帧 / 900 决策 |

### 1.2 §3.1 保真门：**在旧窗 11 上真跑通过**

`gate-w11.json`（两次独立恢复，逐字段相同）：

| 断言 | 实测 | 判定 |
| --- | --- | --- |
| §2.1 HELLO `role=shadow` | `OK hello proto=1 harness=1 role=shadow frames=0` | ✅ |
| §2.3 前缀 2,165 检查点 0 不符 | `hash_checks=2165, hash_mismatches=0, first_mismatch=null, final_frame=8663` | ✅ |
| §2.5 PARK 在 8663 | `OK park=1 template=1 frame=8663`，`park_frame="8663"` | ✅ |
| §2.2 DEBUG `threads=1` | `OK curState=2 … threads=1 frames=8663` | ✅ |
| 录像来源 | `.tmp_v410_s1c/d1/shards/d1-s0d1-n3300-seed11-shard/labels.jsonl`（3,300 行 → 13,200 帧） | — |
| 前缀重放耗时 | `wall_s = 1.807` | — |

**两次独立恢复逐位一致**（`r1` 与 `r2` 是两次独立 spawn）：

```text
park_state_tuple r1: ('7c8705fc88f94c17', '0299ade917af1401', 2165, 0, 8663, '8663')
park_state_tuple r2: ('7c8705fc88f94c17', '0299ade917af1401', 2165, 0, 8663, '8663')
TWO RESTORES IDENTICAL: True
```

（即 `reset_hash`、`final_step_hash`、检查点数、失配数、末帧、PARK 帧六项全同。）

**一对之内两臂的起点更强于「两次恢复」**：一对只起**一个模板进程**（`parent pid=1343250`），
PARK 由它 fork 出的子进程执行（`park_reply: OK park=1 template=1 pid=1343258 frame=8663`），
两臂各自从同一个模板状态 fork 出来跑。⇒ 两臂起点**由同一个 PARK 状态保证**，不靠两次独立重放对齐。
`parent_gate` 在两条行里都记 `park_frame=8663`、`hello_role_shadow=true`。

### 1.3 §3.2 入口与清单读取器：fail-closed 真的会拒绝

| 场景 | 结果 | 退出码 |
| --- | --- | ---: |
| 清单声明曝光 true/true、不加 `--allow-exposed` | guard `EXPOSURE_CONFLICT`；本工具 `MANIFEST_REFUSED` | **1** |
| 清单声明 2400 档、却请求 `--budget-frames 1800` | `BUDGET_ROOT_MISMATCH`（清单就是运行表） | **1** |
| 清单在 guard 校验后、runner 读取前被改一个字节 | `MANIFEST_CHANGED`（对照旧逻辑会静默用被改的值 → 0） | **1** |
| 声明 `used_for_training=false` 但谱系把它放在 `train` | `EXPOSURE_CONTRADICTS_LINEAGE` | **1** |
| `parent_prefix_digest` 与本窗口录像 SHA 不符 | `PREFIX_RECORDING_MISMATCH` | **1** |
| 库入口 `evaluate(budget=2400.0/0/-1/None/"2400"/True/9999)` | `BUDGET_INVALID`（修前是静默通过） | **2** |
| 同一 JSON 对象内重复键（`takeover_frame` 先 999999 后 8663） | `MANIFEST_UNREADABLE`（修前 last-wins 通过） | **2** |
| 两根本质同一窗口、只改 `ancestor_id` 标签 | `DUPLICATE_WINDOW`（修前通过） | **1** |
| 本窗口真实谱系 | `lineage_split="train"`、`lineage_status="used_for_training"`、`lineage_unmatched=[]` | 0 |

独立复核（对抗式，65 次 CLI + 7 个突变体）抓出的 3 个可绕过缺陷 **M1/M2/M3 已修**：`selftest` 从 13 项升到 **34** 项，
4 个「改了也不报警」的突变体现在**全部被抓**（`DEFAULT_BUDGET`→1800/9999/0、`is_int` 收 bool、`>` 改 `>=`、digest 收大写）。
M4（曝光自证 + TOCTOU）在**接线侧**修：字节级一次读盘闭合 + `--lineage-config` 交叉核对。

### 1.4 §3.3 冒烟：8 项验收逐项

| 验收项 | 实测证据 | 判定 |
| --- | --- | --- |
| 两次恢复哈希一致 | 见 §1.2，六项全同 | ✅ |
| 每决策抽样次数 = 1 | `rng_draws = 983 = trace 长度`，`decisions_executed = 982`（差的 1 次是致死决策：被 trace 记录、不计 executed） | ✅ |
| 两帧 mask（雷位只在决策首帧） | 逐决策 trace 有 `masks` 字段；`emit_decrement_at = 2`（详见 §3） | ✅ |
| 计时边界 | 每行 `wall_s` + `spawn_ms` + 前缀 `wall_s=1.807`；见 §4 成本 | ✅ |
| 静音 | `audio_audit.json`：`cfg_files=1, decoded=1, errors=0, muted=1, audible=0, four_zero=1, ok=true`；四字段 `musicMode/playSounds/musicVolume/sfxVolume` **全 0**；`SDL_AUDIODRIVER=dummy`；`DISPLAY`/`WAYLAND_DISPLAY` 已 pop；四层链路（env / HarnessClient / cfg 重生成 / 跑后审计）逐层记录 | ✅ |
| 库存变化 | `bombs_start=3`（**回答了预注册的开放问题**）、`bombs_end`、`bombs_used`、`bombs_remaining_at_emit`、`bomb_available_at_emit`、`emit_decrement_at_list` 全部落盘 | ✅ |
| 故障退出处理 | 见 §5：一次真实 `FileNotFoundError` 被 fail-closed 成 `UNEXPECTED` exit 2，**引擎已退出、无残留**（`engines_after=0`） | ✅ |
| 同起点采样一致 | 两臂 `generator_initial_digest` 相同（2400 档 `23667ba0aad2c5ff`、1800 档 `be9f389e75e817c8`），`rng_same_initial_state=true` | ✅ |

**帧级扣雷事件**（v8.26 缺的那一层）现在这样落盘：

```text
G1 b2400: bomb_decisions  [0, 1, 2, 3, 4, 6, 352, 591]
          bomb_frames     [0, 2, 4, 6, 8, 12, 704, 1182]      ← 绝对帧偏移（2 帧/决策）
          bombs_remaining_at_emit [3, 2, 2, 2, 2, 2, 2, 1]
          bomb_available_at_emit  [True, False, False, False, False, False, True, True]
          emit_decrement_at_list  [2, None, None, None, None, None, 2, 2]
```

---

## 2. 冒烟结果（两档各 1 对，window 11，checkpoint `bombdag.pt` seed0）

| 档位 | 臂 | 规则 | frames | decisions | emits | used | honoured | ignored | wall_s | terminal |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2400 帧（40 s） | G1 | `URG-a0.50` | 1,966 | 982 | 8 | 3 | 3 | 5 | 1.3903 | first_death @ 10629 |
| 2400 帧（40 s） | T2 | `40/20/5` | 1,966 | 982 | 3 | 3 | 3 | 0 | 1.3417 | first_death @ 10629 |
| 1800 帧（30 s） | G1 | `URG-a0.50` | 1,538 | 768 | 9 | 3 | 3 | 6 | 0.9567 | first_death @ 10201 |
| 1800 帧（30 s） | T2 | `40/20/5` | 1,447 | 723 | 7 | 3 | 3 | 4 | 0.8351 | first_death @ 10110 |

两档 `invalid=0`、`full_segments=0`、`trace_alignment_ok=true`。
实测阈值表（逐决策落盘）：G1 `{B=3: 49.4975, B=2: 21.2132, B=1: 5.0}`（= `35·√(1200/(3·200))`，与预注册 §1 公式一致）；
T2 `{3: 40.0, 2: 20.0, 1: 5.0}`。

> **注意这不是策略结论**：只有 1 个祖先、1 个 checkpoint、1 个 RNG，且该祖先**已曝光**（`--allow-exposed` 才跑得起）。它只证明适配器把两臂的差异按协议落盘了。

---

## 3. 首个分叉：一个可直接交给研究端的观察

用新入口的 `--mode compare`（七字段严格比对 + 「首个行为差异」次判据）逐决策对齐两臂 trace：

| 档位 | 首个字段差异 | 首个**行为**差异 | `mask`/`bomb_request` 不同的决策 | `cand` 不同 | `inventory_after` 不同 |
| --- | --- | --- | ---: | ---: | ---: |
| 2400 帧 | 决策 0（threshold 49.4975 vs 40.0，构造使然） | 决策 1（绝对帧 8665） | **5** → `[1,2,3,4,6]` | **0** | **0** |
| 1800 帧 | 决策 0 | 决策 4（绝对帧 8671） | 108 | 99 | 58 |

- **40 秒档**：G1 多发的 5 次请求**全部在无敌帧、全部不兑现**（`bomb_available_at_emit=False`），
  因此 **`cand` 与 `inventory_after` 逐决策完全相同，两臂死在同一帧 10629**。
  ⇒ 在本窗口/本 checkpoint 上，40 秒档 G1 与 T2 的**状态轨迹等价**，差别只在「发了多少个无效请求」。
  这与历史 40 秒档差 4 局（34/240 vs 30/240）量级一致，也解释了为什么长档差异小。
- **30 秒档**（另一个 pair RNG）：先出现库存差异（决策 349），再出现动作差异（决策 356），
  此后两臂真正分叉（G1 活 1,538 帧、T2 活 1,447 帧）。
  决策 356 处 G1 看到 `nearest=10.23 px`、T2 看到 `nearest=100.43 px` —— 状态已经分叉，符合预注册 §1「同流容许状态分叉后选出不同动作」。

**口径修正（重要，供研究端写 §4 计分时用）**：`emit_decrement_at = 2`，
即**库存扣减在决策的第 2 帧才被观测到**（决策 = 2 帧）。因此「请求是否兑现」必须以**决策末帧**的读数判定，
不能用请求帧判定，也不能用「最后一次请求」直接当「最后一次扣雷」。

---

## 4. 成本实测（用于研究端排期）

| 环节 | 实测 |
| --- | ---: |
| 前缀重放 + PARK（每次 spawn） | **1.807 s** |
| 单臂单局（2400 帧） | 1.34–1.39 s |
| 单臂单局（1800 帧） | 0.84–0.96 s |
| 一「对」（PARK + G1 + T2，2400 帧） | **≈ 6 s 墙钟**（含启动与跑后审计） |
| 一「对」（1800 帧） | **≈ 5 s 墙钟** |
| 外推：一档 48 对 | ≈ **4.8 min**（串行，2400 帧档）；6 路并发 ≈ 1 min |
| 外推：两档 96 对 | ≈ **9 min**（串行） |

⇒ 与上一份回执的估计一致：**算时不是瓶颈**，瓶颈仍是「干净祖先 = 0」。

---

## 5. 冒烟中真实发生的一次失败（如实报告）

第一次运行（`tag=...-r1`，2400 档）在 **PARK 成功之后**因输出目录不存在而中止：

```text
[v450] window 11: parent up pid=1343199 prefix 2165/2165 checkpoints, 0 mismatches, PARK frame=8663, threads=1
Traceback (most recent call last):
  File "/mnt/d/WorkShop/Touhou_RL/tools/rl/v829_pair.py", line 2734, in mode_run
    with out.open("a", encoding="utf-8") as handle:
FileNotFoundError: [Errno 2] No such file or directory: '.tmp_v829_pair/eval/...jsonl'
REFUSED [UNEXPECTED] ... (treated as a technical failure)
```

- 行为是**正确的 fail-closed**：退出码 2、`engines_after=0`（引擎已退出、无残留进程）。
- 缺陷是**不友好**：`--out` 的父目录不会自建。当时的绕过办法是外部 `mkdir -p .tmp_v829_pair/eval .tmp_v829_pair/runs`
  （**未改代码**）。
- 副作用：`pid=1343199` 这次 spawn 留下的 `gate-w11.json`，正好被用作 §1.2 的「第二次独立恢复」对照。

**该缺陷已在本轮修掉并复跑验证**（见 §9）。

**其他已知未实现**（明确列出，不当作已完成）：预注册 §4 的「技术故障按原 pair/随机流重跑双方一次」没有自动化
（`invalid` → exit 1，人工用同清单 + 新 tag 重跑）；`--workers` 并发父进程；`--mode report` / Δ30 / 四格表统计。

---

## 6. 残余风险（交给研究端判断）

1. **曝光仍是半自证**：守卫没有 `splits.json` 输入；接线侧的 `--lineage-config` 交叉核对默认关闭。
   对**已曝光旧窗**我们必须显式 `--allow-exposed`，这一步是人为的。
2. **谱系 id 与录像文件之间是间接绑定**：清单声明祖先 id、前缀门钉录像 SHA，两者不是同一张证明；
   研究端若要强绑定，需要在清单里补「祖先 id ↔ 录像 SHA」的登记行。
3. **`total_frames` 是经验下界（11063 = 8663 + 2400），不是阶段长度**：只由「G1 d1200 34/240 局活满 2400 决策且 0 例 stage_clear」支撑。
   本机**没有任何索引字段**提供每祖先帧跨度（见上一份回执 §9 勘误第 4 条）。
4. **窗口身份键含 `budget_frames`**：同一物理窗口换标签 + 换档位仍可让它在 Δ30 里被算两次（工单冻结五元组的直接代价）。
5. **新起点（≠8663）仍跑不起来**：`walk_prefix`、`masks_for`、`run_episode36` 三处 `PREFIX_FRAMES` 硬编码只读断言，
   要动必须单独授权（本轮**没有**改这三个被 pin 的文件）。
6. 守卫 `selftest` 只钉码集与 34 条语义用例，不等于「清单里的数是真值」；证据门仍需人工复核。

---

## 7. 命令与退出码

```text
# 全部在 WSL /mnt/d/WorkShop/Touhou_RL，SDL_AUDIODRIVER=dummy，/root/v400-venv/bin/python
python -B -m py_compile tools/rl/v829_guard.py                     -> 0
python -B tools/rl/v829_guard.py selftest                           -> 0    (SELFTEST: ok 34/34)
python -B -m py_compile tools/rl/v829_pair.py                       -> 0
python -B tools/rl/v829_pair.py --help                              -> 0
python -B tools/rl/v829_pair.py --mode check --manifest ... --budget-frames 2400 \
        --allow-exposed --allow-legacy-window --lineage-config configs/v430/splits.json
                                                                    -> 0    would_start_engine=false
python -B tools/rl/v829_pair.py --mode pair --manifest smoke-old-window.json \
        --window 11 --budget-frames 2400 --episodes 1 --arm-order G1,T2 ... --keep-trace
                                                                    -> 0    rows=2 invalid=0
python -B tools/rl/v829_pair.py --mode pair --manifest smoke-old-window-1800.json \
        --window 11 --budget-frames 1800 ...                        -> 0    rows=2 invalid=0
python -B tools/rl/v829_pair.py --mode compare --trace-g1 ... --trace-t2 ...
                                                                    -> 1    VERDICT: DIFFER（差异 = 预期）
pgrep/ps 统计 th08-modern（每次运行前/后）                          -> 0 / 0
```

引擎启动次数：**3 次成功 spawn**（r1 冒烟中止、r2 2400 档、r3 1800 档）+ **0 次残留**。
音频：**每次 spawn 的私有 data dir 都复核四字段全 0**，跑后再审计，`audible=0`。

---

## 8. 下一步（仍需研究端/用户决策，不是训练端能自行推进的）

1. **新起点来源**：8 个（目标）新开发祖先、每根 1 窗、每根 ≥2,400 帧尾巴 —— 现在仍有 **0** 个可用干净祖先。
   两条路（新 seed 采集 / 动用未花费 confirm 段 61000·61100·61200·61400 各 100 seed）都需要一张来源工单。
2. **清单补 4 类字段**：`takeover_frame`、`total_frames`（含来源）、`parent_prefix_digest`、`verify_anchor`。
3. **接管帧定义**：沿用 A1 的「首死前 30–120 帧」只留 30–120 帧，与 2,400 差 20–80 倍。
4. **要不要把 `v829_guard.py` 当作冻结契约**：它现在是训练端提案（未登记、未提交）。研究端若接受，
   建议把 schema 与错误码表写进预注册，再登记 SHA。
5. **要不要授权改 `PREFIX_FRAMES`**：非 8663 的接管帧必须动到三处只读断言，这是新起点的前置条件。

---

## 9. 补记（2026-11-23 11:41 +0800）：缺陷修复 + 跨进程确定性

### 9.1 修了什么

`tools/rl/v829_pair.py` 新增 `ensure_parent_dir()`（`:574-588`，`path.parent.mkdir(parents=True, exist_ok=True)`，
**不 try/except、不吞 OSError**），在 `mode_run()` 的**引擎 spawn 之前**（`:2585` / `:2586`）为 `--out` 与 `--log` 建父目录；
`runs/<tag>` 仍按需创建；原 `out.open("a")` 未改。
⇒ `--out` 不可写仍是 **fail-closed**（`[UNEXPECTED]` exit 2），只是把失败提前到 spawn 之前，不再浪费一次 PARK。

### 9.2 修复后复跑（`tag=...-b2400-r4`，**事先删掉 `eval-r4/` 与 `runs/<tag>/`**）

时间线：`11:38:38` 三路径 `ABSENT` → 建空目录再 `rmdir` 确认不存在 → 直接跑 → **exit 0**，
目录由修复代码自己创建。两臂结果与 §2 的 r2 **完全一致**：

| 臂 | frames | decisions | emits | used | ignored | wall_s | terminal | `generator_initial_digest` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| G1 | 1,966 | 982 | 8 | 3 | 5 | 2.3414 | first_death | `23667ba0aad2c5ff` |
| T2 | 1,966 | 982 | 3 | 3 | 0 | 2.2721 | first_death | `23667ba0aad2c5ff` |

`rng_draws = 983/983`、`trace_alignment_ok = true`、`engines_after = 0`、
`audio_audit.json`：`cfg_files=1, errors=0, muted=1, audible=0, four_zero=1, ok=true, outcome=muted`。

### 9.3 顺带得到一个更强的结论：**跨进程确定性**

`r2`（11:31，`tag=...-r2`）与 `r4`（11:38，`tag=...-r4`）是**两次独立进程、两个独立私有 data 目录、两次独立 PARK**，
相同的清单 / checkpoint / `pair_rng`。逐决策比对：

```text
G1: 983 entries,  decision entries differing: 0
T2: 983 entries,  decision entries differing: 0
CROSS-PROCESS DETERMINISM (decision arrays identical): True
```

两份 trace 的字节数也相同（G1 560,496 B / T2 506,862 B）；差异只出现在 7 个易变键
（`tag`、`data_dir`、`parent pid`、`prefix.wall_s`、`spawn_ms`、`wall_s`），jsonl 23,881 → 23,882 B 也只是 `wall_s` 位数。
⇒ **同输入在同引擎上可逐决策复现**，这是预注册 §4「技术故障按原 pair/随机流重跑」的前提条件（重跑自动化仍未实现，见 §5）。

### 9.4 一个必须记住的接口行为

`--pair-rng` **不是**可以随手覆盖的参数：清单 `roots[i].pair_rng` 是登记值，
传不同值会被 `PAIR_RNG_CONFLICT` **拒绝**（exit 1，不启引擎）。
本轮复跑因此仍用清单声明的 `20261124` —— 要换种子必须**新开一个清单 root**，
这也正是 §6 残余风险第 4 条（窗口身份键含档位）的同一个设计取向：**清单就是运行表**。

### 9.5 本轮引擎使用总账

| 用途 | 次数 | 结果 |
| --- | ---: | --- |
| `tag=...-r1`（2400 档，输出目录缺陷） | 1 spawn | PARK 成功，随后 `UNEXPECTED` exit 2，**0 残留** |
| `tag=...-r2`（2400 档） | 1 spawn | exit 0，2 行 |
| `tag=...-b1800-r3`（1800 档） | 1 spawn | exit 0，2 行 |
| `tag=...-b2400-r4`（修复复跑） | 1 spawn | exit 0，2 行 |
| **合计** | **4 spawn** | **每次前后 `th08-modern` 计数 0/0；每次私有 cfg 四字段全 0；未 `pkill`、未碰他轮进程** |
