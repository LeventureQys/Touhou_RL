# T08 / EVAL_SHOOT_PULSE：带 SHOOT 脉冲的 v4.2.0 评测器

对象：v4.2.0 `T08`。前序诊断
[`CLEAR_REACHABILITY.md`](CLEAR_REACHABILITY.md) 证明：`tools/rl/v410_student_eval.py`
的动作通道把 `BTN_SHOOT` 无条件带在每一帧上，结构上不可能产生引擎换面所需的
`SHOOT` 上升沿，因此**该仪器无法报告 `stage_clear`**；同一段录制轨迹加"每 48 决策
松一帧 SHOOT"后即可过关。本档交付**新仪器**实现该脉冲，并用三条真实实验证明它
（a）能跑通、（b）与旧仪器字段同构、（c）**确实能报告过关**。

> **一句话判定：新仪器 `tools/rl/v420_eval_shoot.py` 在录制轨迹上复现
> `stage_clear=true / 13,195 帧 / 2,631,430`（且掩码轨迹 sha256 与已证明的
> `replay_clear.py --mode pulse` 逐位相同）；同一轨迹把 period 改成永不释放的
> 1,000,000,007 则复现 `stage_clear=false / 141,379@13,195 / game_over`
> （掩码轨迹与 T08 的 eval 通道逐位相同）。旧仪器 `tools/rl/v410_student_eval.py`
> 逐字节未动（sha256 `2db2caaf…`），旧数字继续绑定旧仪器。**

全部数字由 [`eval_shoot_pulse.json`](eval_shoot_pulse.json) 承载；该 JSON 由脚本
[`40_verify.py`](../../../../../../.tmp_v420_evalpulse/40_verify.py) 与
[`30_emit_evidence.py`](../../../../../../.tmp_v420_evalpulse/30_emit_evidence.py)
从真实运行产物生成与复核，不含手抄数字。

---

## 1 交付物与只读边界

### 1.1 新增（白名单内）

| 文件 | 作用 |
|---|---|
| [`tools/rl/v420_eval_shoot.py`](../../../../../../tools/rl/v420_eval_shoot.py) | 新评测器：v4.1.0 全部语义 + SHOOT 脉冲 + 混表拒绝 + 录制轨迹对照模式 |
| [`tests/test_v420_eval_shoot.py`](../../../../../../tests/test_v420_eval_shoot.py) | 33 个 engine-free 测试（脉冲算术、fail-closed、身份、混表、旧仪器未改） |
| `Document/Update/Alpha/v4.2.0/evidence/T08/EVAL_SHOOT_PULSE.md` | 本档 |
| `Document/Update/Alpha/v4.2.0/evidence/T08/eval_shoot_pulse.json` | 机器生成的证据负载 |

临时目录（全部新增子目录）：`.tmp_v420_evalpulse/`（脚本、日志、两次对照的运行 JSON）、
`.tmp_v420_data/T08/evalpulse-dev-data`（5+5 dev 的私有 data）、
`.tmp_v420_data/T08/evalpulse-dev`（5+5 的两个汇总 JSON）、
`.tmp_v420_data/T08/evalpulse-ctl-pulse`、`.tmp_v420_data/T08/evalpulse-ctl-nopulse`。
三个 data 目录都是 `.tmp_v420_data/T06/engine-data-gamesrc` 的 `cp -a` 私有副本
（474 MB 各一份），**未写** `/root/v400-data`、`.tmp_v410_*`、
`.tmp_v420_data/{T02c,T02d,T04,T05,T06}/**`。

### 1.2 未改（本任务只读）

| 文件 | sha256 | git status |
|---|---|---|
| `tools/rl/v410_student_eval.py` | `2db2caaf9089c1698cef9b6a682991a7594ebac47ae93244c3d9932591eddddc` | `??`（本来未跟踪，仍未跟踪） |
| `tools/rl/v400_student_eval.py` | `7bccf2409cd31065a4e0e3148f03672fc764b1e562986e47404ed989e843c1a2` | 空（已跟踪，干净） |

两份文件在 `shoot|pulse|BTN_|release_due`（大小写不敏感）上各 **0 命中**，即旧仪器
至今没有任何释放路径——这与 [`clear_reachability.json`](clear_reachability.json)
`q1_channel.release_paths_searched` 一致，也是"旧数字绑定旧仪器"的机制前提。
`touhou_rl/**`、`configs/**`、其它 tests、T08 既有四份文件（`CLEAR_REACHABILITY.md`、
`clear_reachability.json`、`A1_SAFE_SET_PILOT.md`、`a1_safe_set_pilot.json`）以及
`v4.2.0` 下的 PROGRESS/state/journal/budget 均未写入（mtime 早于本档开始时间；
`git status` 里本档只新增上述两个代码/测试文件与 T08 的两个新文件）。

---

## 2 唯一行为差异：每 48 决策松一帧

### 2.1 规则

复用冻结的训练规则 `touhou_rl.v400.teacher.shoot.release_due(i, period)`：
决策 `i`（0 基）是释放决策当且仅当 `(i + 1) % period == 0`，即 `i = 47, 95, 143, …`。
该决策的**最后一个 STEP 帧**发 `mask & ~(BTN_SHOOT | BTN_BOMB)`，其余帧发
`gym_actions.cand_to_mask(cand, 0)` 原掩码：

```text
decision i=47, unit 0 of 2:  mask                  (SHOOT held)
decision i=47, unit 1 of 2:  mask & ~0x0003        (release: one frame)
decision i=48 … i=94      :  mask for both units
decision i=95, unit 1 of 2:  mask & ~0x0003
```

方向位与 FOCUS 位原样保留（是"松扳机"一帧，不是改移动）；`cand=0` 的释放帧因此是
`0x0000`，`cand=17` 的释放帧是 `0x0054`（`0x55 = FOCUS|NW|SHOOT`）。48 决策 × 2 帧 =
96 帧周期，与 `configs/v420/rl-engine-{audit,pilot}.json` 的
`rollout.shoot_pulse_period: 48` 和 `touhou_rl/v420/rl.py::SHOOT_PULSE_PERIOD` 一致；
`teacher/shoot.py` 的默认 24 是给 4 帧决策的，未采用。

### 2.2 实现复用（不是重写一遍评测器）

新文件通过 `importlib` 只读加载 v4.1.0 评测器并**调用**它：`EnvHarnessClient`
（v410 又只读复用 v400）、`decision_steps`、`load_v410_checkpoint`（arm 预注册、
`strict=True` 状态字典、参数钉）、`ProbePolicy`（argmax、bomb 头只读不执行）、
`check_seed_policy`/`parse_seeds`（dev 1000–1099，confirm 60000–60199 与 corpus 1–20 拒绝）、
`validate_episode_record`、`summarise`/`summarise_arm`。新文件自己只新增两件东西：
`pulse_step_mask`（上表那一行）与 `_run_loop`（v410 `run_episode` 的帧循环，逐行同构，
唯一差别是每帧掩码过 `pulse_step_mask`）。测试
`test_the_loop_sends_every_decision_frame_through_the_pulse` 直接断言
`client.step(` 在整个循环里只有两处：一处是像素预热 `cand_to_mask(0, 0)`，一处是
`client.step(step_mask, want)`——不存在绕过脉冲的第二个 STEP 出口。

### 2.3 fail-closed：没有"关掉脉冲"的隐藏开关

| 输入 | 行为 | 位置 |
|---|---|---|
| `--shoot-pulse-period 0 / 负数 / None / 非 int` | 拒绝（`EvalError`，CLI 退出码 2，**任何引擎进程之前**） | `check_shoot_pulse_period`，`main` 的第一段校验 |
| `--obs-window strided` | 拒绝：该窗口把一次决策压成**一个多帧 STEP**，掩码覆盖决策内每一帧，释放帧无法表达；拒绝而不是静默不脉冲 | `check_obs_window_pulse` |
| 候选集合 / 观测窗口 / 观测形状 | 无任何新开关；`--obs-window` 仍是 `OBS_WINDOWS` 两项，形状仍走 `obs_spec_v3` | `build_parser`，`policy_identity` |
| 测试 | `test_cli_has_no_pulse_off_switch`（选项名里没有 no-pulse/disable/…）、`test_cli_refuses_period_zero_before_any_engine`、`test_strided_window_is_refused` | `tests/test_v420_eval_shoot.py` |

---

## 3 字段同构证明（实测，非声明）

用**同一命令**（评估开关逐项相同：`--engine`、`--data-dir`、`--checkpoint`、
`--seeds 1000-1004`、`--stage 0`、`--difficulty normal`、`--max-frames 19887`、
`--obs-window consecutive`、`--device cpu`、`--timeout 240`）跑 R0 两口径，
产物分别是 `R0-old-1000-1004.json` 与 `R0-new-1000-1004.json`。字段差：

| 对象 | 旧仪器字段数 | 新仪器字段数 | 新增字段 | 缺失字段 |
|---|---|---|---|---|
| 逐局记录（episode） | 61 | 62 | `shoot_pulse_period` | 无 |
| 汇总（summary） | 28 | 29 | `shoot_pulse_period` | 无 |

* **字段集合**：`new_only_fields = ["shoot_pulse_period"]`、`missing_fields = []`（两侧
  5 局逐一相同）。逐局记录的类型表 `REQUIRED_EPISODE_FIELDS` 与 v4.1.0 **逐项相等**
  （测试 `test_episode_record_field_set_is_v410_plus_one` 直接对比两份表）。
* **身份值**：5 个种子逐局比较"非测量字段"，差异**只有**
  `schema`、`controller`、`shoot_pulse_period` 三个（`identity_value_differences_by_seed`
  五个种子完全一致）。也就是说候选规则、观测契约、checkpoint 身份、seed 策略、
  `max_frames`、`stop_on_death`、`decision_frames`、`window_alignment`、engine sha256
  等身份值逐字节相同。
* **校验器**：`validate_episode_record` 直接调用冻结的
  `v410.validate_episode_record`，只把副本里的两个**身份值**（`schema`/`controller`）
  换成冻结值，其余字节原样送检；新仪器自己的身份（新 schema、新 controller、
  `shoot_pulse_period ≥ 1`、字段集合）在送检前先行检查。因此 v4.1.0 的每一条
  语义断言（histogram 求和、stage/clear 一致性、arm 参数钉、bomb 块、失败局必须带
  error、dev seed 检查……）都仍然生效。
* **测量值**：种子 1000/1002/1004 两口径只有 `wall_s` 不同（动作序列与结果逐位一致），
  1001/1003 的 `frames/steps/score/action_sequence_sha256/...` 不同——这正是脉冲
  在策略通道里产生的真实分叉，而不是字段差异。

---

## 4 5+5 局 dev 结果（R0）

R0 = T04 A0 checkpoint（B0，731,802 参数，sha256 `96cb219e…`），dev seeds 1000–1004，
S1N 口径（stage 0 / normal / cap 19,887 / 不 stop-on-death）。

| seed | 旧仪器帧数 | 旧仪器分数 | 新仪器帧数 | 新仪器分数 | 新仪器释放决策数（`steps // 48`） | 两口径是否逐位一致 |
|---|---|---|---|---|---|---|
| 1000 | 2137 | 18,951 | 2137 | 18,951 | 22 | 是 |
| 1001 | 3430 | 26,258 | **2654** | **22,263** | 27 | 否 |
| 1002 | 2185 | 7,600 | 2185 | 7,600 | 22 | 是 |
| 1003 | 2731 | 12,827 | **2741** | **12,465** | 28 | 否 |
| 1004 | 2456 | 10,878 | 2456 | 10,878 | 25 | 是 |

* 两口径都是 **0/5 过关**（`stage_clear=false`，全部 `game_over`，deaths 3.0）：
  R0 在 2,137–3,430 帧内死亡，离 13,195 帧的过关长度差 ~10,000 帧；**脉冲不会让早死
  的策略过关**，它只让"过关"这件事可被报告。
* 旧仪器本次重跑**逐局重现**了 `.tmp_v420_data/T06/dev/merged-R0.json` 里冻结的
  R0-s1n 帧数（2137/3430/2185/2731/2456），证明旧仪器与 T04 批次仍是同一把尺
  （证据 `old_instrument.rerun_reproduces_the_frozen_t06_rows`，5/5 `match=true`）。
* 新仪器帧数均值 2434.6（旧 2587.8），中位数同为 2456——描述性数字，
  不是 gate、不是 development score、不是 clear rate。

---

## 5 正向对照：新仪器的通道能报告过关

命令（`.tmp_v420_evalpulse/20_control_pulse.sh`，退出码 0，wall 121.6 s）：

```text
xvfb-run -a … tools/rl/v420_eval_shoot.py --engine …/rl_v3/th08-modern \
  --data-dir .tmp_v420_data/T08/evalpulse-ctl-pulse \
  --replay-tape --shard .tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed12-shard/slots.jsonl \
  --tail 64 --stage 0 --difficulty normal --max-frames 19887 \
  --obs-window consecutive --score-trace-frames 13195 \
  --shoot-pulse-period 48 --out .tmp_v420_evalpulse/positive_control.json
```

| 量 | 期望（T08 已证） | 新仪器实测 |
|---|---|---|
| `stage_clear` | `true` | **`true`**（`stage_start 0 -> stage_end 1`，`stop_reason=stage_clear`） |
| 帧数 | 13,195 | **13,195** |
| 分数 | 2,631,430 | **2,631,430** |

附加的机制数字（同一次运行）：6,598 决策、`releases_planned = releases_applied = 137`、
SHOOT 帧 13,058 / 松键帧 137、上升沿 138、`mask_trace_sha256 =
66c81344b00f8766f7d9e860d1874b1141c699122f491ed1a8ed0aea9f179924`——
**与 [`clear_reachability.json`](clear_reachability.json)
`q2_experiment.runs[pulse].mask_trace_sha256` 逐位相同**，也与
`replay_clear.py --mode pulse` 的 13,195 / 2,631,430 / 6,598 决策逐项一致。
换言之：新评测器的"脉冲通道"和早已证明能过关的那条等价实现是**同一条通道**
（同一 client、同一 `decision_steps`、同一掩码构造、同一释放规则）。

这是唯一能证明"新仪器能报告过关"的证据；录制轨迹是教师录制，**不是策略**。

---

## 6 负向对照：差异确实来自脉冲

同一条轨迹、同一个通道、同一帧预算，只把 period 改成 1,000,000,007（大于任何可能
决策数 ⇒ `(i+1) % period` 永不成立 ⇒ 永不释放；`releases_planned=0`）。
命令 `.tmp_v420_evalpulse/21_control_nopulse.sh`，退出码 0，wall 138.9 s。

| 量 | 期望（T08 eval 通道） | 新仪器实测 |
|---|---|---|
| `stage_clear` | `false` | **`false`** |
| 第 13,195 帧的分数 | 141,379 | **141,379**（`score_at_frames["13195"]`） |
| 结束原因 | `game_over` | **`game_over`**（15,564 帧、终分 163,945、deaths 3） |

附加：`releases_applied = 0`、SHOOT 帧 15,564 / 松键帧 0、上升沿 1、
`mask_trace_sha256 = 92cea74d6f2b51663ef692cc0f360aec06a64923044120fc8b5d390cd1feba10`
——**与 T08 eval 通道逐位相同**。两条通道的差异只有 `shoot_pulse_period`
（48 vs 1,000,000,007），动作表、观测窗口、帧预算、种子完全相同；因此
`stage_clear` 的有无、13,195 vs 15,564、2,631,430 vs 141,379 **只能**来自脉冲。

---

## 7 旧数字仍然绑定旧仪器

1. **字节**：`tools/rl/v410_student_eval.py` sha256 `2db2caaf…`（测试常量固定 +
   证据 JSON 记录 + 每轮 `40_verify.py` 复核）；`tools/rl/v400_student_eval.py`
   sha256 `7bccf240…`。git 状态：v410 仍是未跟踪 `??`（本任务不 add），v400 干净。
   v410 自带测试 **152 passed**（Windows 与 WSL 两侧，未改动任何测试）。
2. **机制**：旧仪器 `shoot|pulse|BTN_|release_due` 0 命中，没有释放路径——它对
   `stage_clear` 的结构性不可报告性没有被新文件"顺手修掉"。
3. **身份**：新仪器写的每一局都带 `controller="tools/rl/v420_eval_shoot.py"`、
   `schema="v420-eval-shoot-episode/1"`、`shoot_pulse_period=48`；旧仪器写的每一局
   **三个字段一个都没有**（5 局实测 `shoot_pulse_period` 不存在，schema/controller
   仍是旧值）。读一行记录就能判定它出自哪台仪器。
4. **合并工具**：既有合并器 `touhou_rl/v420/eval.py::load_summary` 只接受
   `("v410-student-eval-smoke/1", "v410-student-eval-arms/1")`；新汇总的 schema 是
   `v420-eval-shoot-summary/1`，被它**当作 foreign schema 明确拒绝**（实测：
   `touhou_rl.v420.eval.load_summary(new)` 抛 `EvalError`，而同一函数读旧汇总正常）。
   因为 `touhou_rl/**` 不可改，这条拒绝是"按构造"实现的：换 schema 而不是改合并器。
5. **旧数字的性质**：R0 旧行 0/5（以及文档里 0/20、0/80 等）是**旧仪器的测量**；
   本档既没有重测它们的能力主张，也没有把新仪器的行混进去。任何需要和新仪器比较的
   行都必须重跑（见 §8）。

---

## 8 混表拒绝（episode identity）

* 逐局记录写 `shoot_pulse_period`；summary 顶层同样写一份；逐局与 summary 的 period
  必须相等，否则 `load_arm_summary` 拒绝（"a file may not mix periods"）。
* `--merge-summaries` 只接受本仪器的 summary；`check_same_pulse_period` 对
  `[48, 24]` 直接拒绝（"refusing to merge summaries with different
  shoot_pulse_period values … re-run, never mix"）；缺 `shoot_pulse_period` 的输入
  （即 v4.1.0 汇总）被拒绝；不同 schema 被拒绝。
* 既有合并器按 §7.4 拒绝新 schema，所以"把新旧两批一起合并"这件事在两条路径上
  都不可达。
* 拒绝不是纸面声明：`40_verify.py` 的 5 个拒绝探针（不同 period、v4.1.0 汇总、
  缺 period、`period=0`、`strided` 窗口）**实际执行**并全部成立
  （`mixed_table_refusal.*_refused = true`）。

---

## 9 测试与验收命令

| 侧 | 命令 | 结果 | 退出码 |
|---|---|---|---|
| Windows | `python -m pytest -p no:cacheprovider --basetemp .tmp_v420_evalpulse/pytest-win-final3 -q tests/test_v420_eval_shoot.py` | **33 passed** | 0 |
| WSL | `/root/v400-venv/bin/python -m pytest -p no:cacheprovider --basetemp .tmp_v420_evalpulse/pytest-wsl-final3 -q tests/test_v420_eval_shoot.py` | **33 passed** | 0 |
| Windows | 同上 `tests/test_v410_student_eval.py`（旧仪器自带测试，未改动） | **152 passed** | 0 |
| WSL | 同左 | **152 passed** | 0 |
| WSL | `/root/v400-venv/bin/python .tmp_v420_evalpulse/40_verify.py` | **55 passed, 0 failed → ALL_PASS** | 0 |
| WSL | `/root/v400-venv/bin/python .tmp_v420_evalpulse/30_emit_evidence.py --check` | 证据与现场产物一致（无漂移） | 0 |

测试覆盖（engine-free，不启动游戏）：脉冲算术（第 48/96/… 决策的最后一帧松键、
其余帧不松、单 STEP 决策永不释放、cand 17 的 `0x55 -> 0x54`）、`period=0/负数/None`
被算术/校验器/CLI 三处拒绝、`strided` 被拒、身份含 `shoot_pulse_period`、
记录字段集合 = v4.1.0 字段集合 + 1、冻结校验器仍生效（直方图/clear/controller 三例）、
summary = v4.1.0 summary + 1、混表拒绝（[48,24]、v4.1.0 汇总、缺 period、
episode/summary period 不一致）、同 period 合并成功、tape 记录必须逐帧记账
（shoot+no_shoot+warmup == frames）且释放数不得超过上限、`--replay-tape` 拒绝
`--stop-on-death`、以及 §7 的 sha256/git/源码 0 命中。

---

## 10 不能主张什么

* **不是能力结果**：正向对照回放的是教师录制轨迹，只证明"新仪器能报告过关"；
  它不证明任何学生能过关，也不构成"修好后学生就能过关"。
* **5+5 dev 行不是 gate**：n=5，无预注册判定规则、无 Wilson 门槛、无 confirm 分割；
  §4 的帧数/分数是描述性测量。
* **0/5 没有被推翻**：R0 依旧 0/5，且死亡帧数（2,137–3,430）距过关（13,195）差
  ~10,000 帧——脉冲不是这些 0 的原因，也不会把它们变成 1。
* **没有分离脉冲的两类作用**：本档没有区分"对话等待"与"换面等待"（前序诊断同样
  没有分开），只测到"只有带脉冲的通道能离开等待并过关"。
* **未碰 confirm/corpus**：没有读 confirm 种子 60000–60199、种子 61000–61499 的任何
  记录，没有 corpus 种子 1–20，没有打开 `v4.1.0/evidence/WP02/`；没有训练、没有 1M。
* 没有修改 `touhou_rl/**`、`configs/**`、其它 tests、T08 既有四份文件、PROGRESS/
  state/journal/budget；没有 git add/commit/push；没有杀任何进程。

---

## 11 复现

```bash
ROOT420=/mnt/d/WorkShop/Touhou_RL; PY420=/root/v400-venv/bin/python
cd $ROOT420
bash .tmp_v420_evalpulse/00_setup_data.sh          # 三个私有 data 目录（cp -a，474 MB 各一份）
bash .tmp_v420_evalpulse/10_dev_old.sh             # 旧仪器 5 dev seeds（约 3.3 分钟）
bash .tmp_v420_evalpulse/11_dev_new.sh             # 新仪器 5 dev seeds（约 3.0 分钟）
bash .tmp_v420_evalpulse/20_control_pulse.sh       # 正向对照（约 2.3 分钟）
bash .tmp_v420_evalpulse/21_control_nopulse.sh     # 负向对照（约 2.6 分钟）
$PY420 .tmp_v420_evalpulse/30_emit_evidence.py     # 证据 JSON
$PY420 .tmp_v420_evalpulse/40_verify.py            # 55 项断言 -> ALL_PASS
```

引擎：`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`
（sha256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`）；
env 角色 `xvfb-run -a -s "-screen 0 1024x768x24"`；每次运行 ≤ 10 分钟。
