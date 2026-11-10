# EXP-P1-019 — teacher v3：更大搜索预算（v3a / v3b），PSN 0.625 → ≤ 0.5

- **日期**：2026-11-09
- **状态**：**完成后回填**（详见 §6/§8/§10）
- **上游**：[EXP-P1-015](EXP-P1-015-teacher-v2-bottom-band.md)（v1 → v2a：PSN 5/8 = 0.625、Stage4A 首次过面、**§10.4「先加大搜索预算再动权重」**）、[EXP-P1-016](EXP-P1-016-death-analysis.md)（6 局死亡分析）、[EXP-P1-017](EXP-P1-017-teacher-v2e.md)（v2e 被否：Stage3 5,040 帧即死，v2a 是 25,052）、[EXP-P1-012](EXP-P1-012-g1-evaluation.md)（G1 门与 cap 表）
- **产物**：`touhou_rl/v400/teacher/config.py`（`V3A`/`V3B` 两档）、`tools/rl/v400_teacher.py`（仅 `--mode` 帮助文本）、`tests/test_v400_teacher.py`（新增 3 项）、`.tmp_v400_v3/`（探针、驱动、每面 JSONL、`runs.jsonl`）
- **一句话**：**不动任何一个权重**——`v3a`/`v3b` 与 v2a **共用同一个 `Weights` 实例**——只把搜索预算从 `H48/beam64/k3` 提到 `H64/beam96/k3`（v3a）与 `H64/beam128/k4`（v3b），先用 30 决策探针确认单决策代价 ≤ 200 ms，再完整重测 8 面并（若达标）重判 S1N 20 局。

---

## 1. 目标与口径

| 项 | 口径 |
|---|---|
| 目标 | **PSN 每面死亡 ≤ 0.5**（8 面口径：Stage1/Stage2 回归 + Stage3/4A/4B/5/6A/6B 靶点），即 **8 面死亡合计 ≤ 4** |
| 判死依据 | **引擎 verdict 唯一**：`STATE.f6`（deaths）增量；到帧上限**不写成"过面"** |
| 过面 | `STATE.f1` 离开起跑面（`stage_clear=true`），否则记 `max_frames` 或 `death` |
| 帧上限 | EXP-P1-012 §2 同一张表：Stage1 19,887 / Stage2 27,788 / Stage3 30,014 / Stage4A 41,583 / **Stage4B 45,000（fallback）** / Stage5 38,513 / **Stage6A 45,000（fallback）** / Stage6B 54,605 |
| 其它 | normal、seed 1、每面各自 `RESET`、无 `--follow-stage`、每 worker 独立 data-dir |
| 完整轮 | 8 面各 1 局，**不许 partial**：缺一面即标 `partial=true`，**不得**与历史轮次拼接成"8 面均值" |
| 权重 | **零改动**。`v3a.weights is v2a.weights` 在测试里是断言，不是注释 |

## 2. 为什么是"加预算"（EXP-P1-015 §10.4）

§10.4 列的四个候选里，前三项（Boss body 代价、留弹、硬剪）都已在 EXP-P1-015/017 被实测否掉：硬剪 v2c 3/3 更差、留弹 v2d 5/5 仍死、Boss-body 代价 v2e 早期更差（Stage3 5,040 帧 vs v2a 25,052）。剩下的第 4 条是**搜索能力**。

支持"瓶颈在搜索而非权重"的直接证据：

| 观测 | 数值 | 含义 |
|---|---|---|
| `full` 档 Stage5 存活 | 21,504 / cap 38,513 = **56 %** | 不是开局即死，是**中后段**没算出来 |
| `full` 档 Stage6B 存活 | 17,564 / cap 54,605 = **32 %** | 同上 |
| v2a 的 5 个死亡面在 cap 上的占比 | 31 %–83 % | 每面都是"活了一大段才死" |
| v2a 死因（EXP-P1-016） | `bullet_wall` 6/6、`bomb_exhausted` 6/6、`bottom_pin` 5/6 | 是**局部弹幕墙**没被提前看见，而不是评分方向错 |

`H=48` 只覆盖 24 个决策 = 48 帧 ≈ 0.8 s 游戏时间；`beam=64` 在 18 分支 × 64 的宽度下每层只留 64 个节点。把两者一起加，是"看得更远 + 留得更多"的最小改动，而且**不需要新证据支撑新权重**——它不改变任何一项的符号。

### 2.1 两档配置

| 参数 | v1 (`full`) | v2a | **v3a** | **v3b** |
|---|---|---|---|---|
| `horizon` | 48 | 48 | **64** | **64** |
| `beam` | 64 | 64 | **96** | **128** |
| `verify_k` | 3 | 3 | **3** | **4** |
| `max_iterations` | 3 | 3 | 3 | 3 |
| `slots` | 24 | 24 | **32** | **32** |
| `durations` | (1,2) | (1,2) | (1,2) | (1,2) |
| `bottom_penalty` / `bottom_extra_penalty` | 0 / 0 | 1.5 / 2.0 | **同 v2a（共用对象）** | **同 v2a（共用对象）** |
| `boss_penalty` / `escape_weight` | 0 / 0 | 0 / 0 | **0 / 0** | **0 / 0** |
| `bomb_rule1_max_death_frame` | `None` | 12 | **12** | **12** |
| `bomb_rule1_min_bombs_left` / `bomb_rule1_max_margin` | 0 / `None` | 0 / `None` | **0 / `None`** | **0 / `None`** |
| `clip_y` | `None` | `None` | **`None`** | **`None`** |

实现上 `_V2A_WEIGHTS` 被 `V3A`/`V3B` **直接引用**（不是 `dataclasses.replace`，不是重新构造）：

```python
V3A = TeacherConfig(mode="v3a", horizon=64, beam=96,  verify_k=3, max_iterations=3,
                    weights=_V2A_WEIGHTS, bomb_rule1_max_death_frame=12)
V3B = TeacherConfig(mode="v3b", horizon=64, beam=128, verify_k=4, max_iterations=3,
                    weights=_V2A_WEIGHTS, bomb_rule1_max_death_frame=12)
```

因此"在 v2a 权重基础上"是**代码级恒等式**：任何后续编辑想把 v3a 的分数改掉，都必须先破坏 `v3a.weights is v2a.weights`，而那会在 `tests/test_v400_teacher.py::test_v3_profiles_register_a_larger_budget_and_keep_the_v2a_weights` 里立刻红。

两档都注册进 `config.CONFIGS`，`PROFILES = (v2a, v2b, v2c, v2d, v2e, v3a, v3b)`；CLI `--mode` 由 `choices=tuple(sorted(CONFIGS))` 自动暴露，默认仍是 `full`（v1 永不被重定义）。

### 2.2 为什么 `v3b` 连 `verify_k` 一起加

11-P1 §4 规则 1 的触发条件是"**所有被验证的**计划都会死"。`verify_k=3` 时第 4 个还活着的计划对规则 1 是**不可见**的，所以只加 `beam` 不加 `k`，等于在更宽的搜索上继续按旧的分辨率判"必死"——那会把加预算换来的更优解**直接浪费成一次多余的 bomb**。v3b 因此是 `beam 96→128` **且** `verify_k 3→4`。

## 3. 单决策耗时探针（§3 门槛：≤ 200 ms）

**协议**：正式 CLI (`tools/rl/v400_teacher.py`)，Stage 5，`--difficulty normal --seed 1`，`--decisions 30 --max-frames 400 --stop-on-death --follow-stage`，每档 30 个决策 = 120 帧；每档独立 data-dir（`/root/v400-data/v3/inst{0,1,2}`）。数字取 JSONL 汇总记录的 `timing_ms`（CLI 自己的**逐决策均值**，`tools/rl/v400_teacher.py:1045,1163`），包含 record + search + verify + observe + events + commit，即 8 面套件真正要付的代价。

| 档 | record | search | verify | commit | observe+events | **均值 ms/决策** | 中位 | 最大 | 相对 v2a | ≤200 ms | exit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v2a（基线，非本轮改动） | 19.47 | 19.03 | 13.62 | 0.54 | 0.20 | **52.86** | 51.65 | 62.11 | 1.00× | ✅ | 0 |
| **v3a** H64/b96/k3 | 30.48 | 43.51 | 18.34 | 0.67 | 0.22 | **93.22** | 91.43 | 111.34 | **1.76×** | ✅ | 0 |
| **v3b** H64/b128/k4 | 29.94 | 56.83 | 24.27 | 0.68 | 0.22 | **111.95** | 109.36 | 155.80 | **2.12×** | ✅ | 0 |

**判定：两档都 < 200 ms → 不触发"回退到 beam=96"，`v3b` 保留。**

**必须写明的限定**（否则这张表会被误读）：

1. 探针的 120 帧只覆盖 Stage 5 的**开头**（弹幕最稀疏的一段）。同面真实全程的平均代价显著更高：v2a round1 的 Stage5 实测 **21,504 帧 / 1,819 s ≈ 169 ms/决策**，是探针 52.86 ms 的 **3.2×**，原因是 tape 随关卡变密、raster 表变大、末端 `align` 目标移动。因此本表的用途是**档间相对代价**（1.76× / 2.12×）与"远低于门槛"，**不是**全程均值。
2. 探针与另一任务的 4 个引擎同时运行（`pgrep -xc th08-modern` 在探针期为 6–8 进程），所以数字里已经含真实争用。
3. 结论只对 Stage5 成立：Stage4A 是全场最重的面（v2a 2538 s / 30,952 帧 ≈ 164 ms/决策），按 1.76×/2.12× 外推，v3a ≈ 290 ms、v3b ≈ 350 ms **per decision**——这**超过** 200 ms。门槛按任务书的定义（Stage5、30 决策）执行，但这条外推必须留在卡里：**8 面的墙钟代价不是按 93/112 ms 算出来的，而是按 Stage4A 的 290/350 ms 算的**（见 §6.4 的墙钟表）。

## 4. 实现（改了哪些行，为什么）

| 文件 | 改动 | 说明 |
|---|---|---|
| `touhou_rl/v400/teacher/config.py` | 新增 `V3A` / `V3B` 两个 `TeacherConfig`；`CONFIGS` 加两项；`PROFILES` 加两项；引用既有 `_V2A_WEIGHTS` | **`Weights` 一个字段都没动**，`search.py` / `decide.py` **零改动**（预算只走 `horizon`/`beam`/`verify_k` 三个既有字段） |
| `tools/rl/v400_teacher.py` | **仅** `--mode` 的 help 文本（"…or v3a/v3b (EXP-P1-019: v2a weights with a larger search budget)"） | `choices` 由 `CONFIGS` 自动带出，无参数语义变化 |
| `tests/test_v400_teacher.py` | 新增 3 项（见 §8） | 新档注册 / H·beam·k 生效 / 性能上限 |
| `touhou_rl/v400/teacher/search.py` | **无改动** | 已能处理任意 `horizon`/`beam`；`slots = horizon/2` 与 `limit = min(slots, len(tape)//2)` 都是既有逻辑 |
| `touhou_rl/v400/teacher/decide.py` | **无改动** | 规则 1 的 12 帧窗口是 `bomb_rule1_max_death_frame`，两档都继承 v2a 的 12 |

即：**本轮的全部变量只有三个整数**。

## 5. 与 v2a 的对比（对照基线）

v2a（EXP-P1-015 round1，8 面完整轮，同 cap、同 seed、同 `--stop-on-death`）：

| stage | 面 | cap | clear | frames | deaths | stop |
|---|---|---|---|---|---|---|
| 0 | Stage1 | 19,887 | **true** | 13,484 | 0 | stage_changed |
| 1 | Stage2 | 27,788 | **true** | 21,740 | 0 | stage_changed |
| 2 | Stage3 | 30,014 | false | 25,052 | 1 | death |
| 3 | Stage4A | 41,583 | **true** | 30,952 | 0 | stage_changed |
| 4 | Stage4B | 45,000 | false | 17,364 | 1 | death |
| 5 | Stage5 | 38,513 | false | 21,504 | 1 | death |
| 6 | Stage6A | 45,000 | false | 13,820 | 1 | death |
| 7 | Stage6B | 54,605 | false | 17,564 | 1 | death |

**死亡合计 5 / 8 = 0.625，过面 3/8。** 本轮要打掉其中至少 1 个。

## 6. 8 面完整结果

（运行时回填）

## 7. 小结与下一步

（运行时回填）

## 8. 测试

```
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_v3\pytest_a -q tests\test_v400_teacher.py
```

新增 3 项：

1. `test_v3_profiles_register_a_larger_budget_and_keep_the_v2a_weights` —— 新档注册（`CONFIGS`/`PROFILES`/CLI `--mode`），且 `v3a.weights is v2a.weights`、`v3b.weights is v2a.weights` 为**同一对象**；`(H,beam,k,iter) = (64,96,3,3)/(64,128,4,3)`；`slots 24→32`；v2a 的 12 帧窗口、无留弹、无 margin 门槛、无硬剪**逐项断言**；`full`/`fast` 仍是原对象、默认仍是 `full`。
2. `test_v3_budget_reaches_the_horizon_the_beam_and_the_verification` —— 三个预算**真的被消费**：`SearchResult.slots` = min(H/2, frames/2)（96 帧 tape 上 v2a 停 24、v3a/v3b 走 32）；展开数只随 `beam` 动（同一 20 帧密集 tape，limit=10 槽：v2a **281,880** → v3a **416,664** → v3b **551,448**），且把 `weights` 换成 v2e 的权重后展开数**逐位相同**（证明多出来的算力是预算不是评分）；`select_plans` 分别返回 3/3/4 个待验证计划。
3. `test_v3_search_stays_within_the_measured_performance_ceiling` —— §3 门槛的单元侧锚点：展开数相对 v2a 的倍率被夹在 [1.4,1.6] / [1.8,2.1]（**比值**断言，换机器不变），单次搜索另有 1,000/1,200 ms 的宽松上界与 `< 3× v2a` 的相对上界（**不用绝对毫秒做硬门**，否则并发跑实验时会假红）。

`test_raster_search_is_faster` 在并发跑实验时会偶发，因此实验期间单独 `--deselect`，空闲后单独验证（见 §10）。

## 9. 复现命令与退出码

（运行时回填）
