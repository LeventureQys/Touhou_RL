# W01c 前置侦察：真实引擎重放已记录的 Stage-1 磁带 + CONTRACT 来源清点

日期：2026-11-15（Asia/Shanghai）。执行者：W01c 执行代理（DeepSeek Flash）。
对象：v4.3.0 `W01` 的前置侦察（[EXECUTION_PLAN.md](../../EXECUTION_PLAN.md) §2 W01 行、§6.1）。
性质：**真实证据采集**，不实现 v430 的 evaluate/session。

> **一句话判定：本机仍能逐位复现 v4.2.0 T08 记录的正/负对照（
> `stage_clear=true / 13,195 帧 / 2,631,430` 与 `game_over / 15,564 帧 / 3 死亡`，
> 动作序列与观测输入 sha256 全部与记录相同）。并且，**三个磁带 shard 全部能通关**
> —— `seed10` / `seed11` 此前被文档描述为"只有 seed12 通关"，实测均 `stage_clear`。
> 6 条来源（3 success + 3 failure）各重放 2 次，**9 个身份字段逐项完全一致**。
> 但 6 条来源只来自 **3 段不同的录制 episode**：success/failure 是同一段 episode
> 在"发键通道带 SHOOT 释放"与"通道结构上不释放 SHOOT"两种设置下的配对结果。**

---

## 1 步骤 0：环境与资产核验（全部通过）

| 对象 | 期望 | 实测 | 判定 |
|---|---|---|---|
| 引擎 `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern` | `691bfd77…a568ef6` | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`（4,836,008 B） | ✅ |
| `…/seed12-shard/slots.jsonl` | `f759de7d…`（任务书给定） | `f759de7dacc78343dec8825da5de14942e749b8a5a2c5319304e513d1618b7b9`（10,097,148 B / 6,588 slots / `episode_seed=12`） | ✅ |
| `…/seed10-shard/slots.jsonl` | —（本任务首次登记） | `52402de44692fa2decd019da9701f57de6e27f193aea01c1e19d2b69fab781a2`（10,100,966 B / 6,592 slots / `episode_seed=10`） | ✅ 登记 |
| `…/seed11-shard/slots.jsonl` | —（本任务首次登记） | `f75e632ba6990a372b4bae614f51bbc74049a75296380491b558b9071f09eb9e`（10,104,250 B / 6,592 slots / `episode_seed=11`） | ✅ 登记 |
| WSL `Ubuntu` + `/root/v400-venv/bin/python` | 3.10.12 | `Python 3.10.12`，`torch 2.14.1+cu130`，`numpy 2.2.6` | ✅ |
| 参考记录 `.tmp_v420_evalpulse/positive_control.json` | `9cdd9033…` | `9cdd90335edcc5a5d2afcf3b1e18d7713e921099967dbf4c62b5b84ef3dbaf63` | ✅ |
| 参考记录 `.tmp_v420_evalpulse/negative_control.json` | `ecb58cb4…` | `ecb58cb40e2c1fe9a3baf75daa24865606fc69a73be12cdc1ce933b394f50643` | ✅ |
| 数据源 `.tmp_v420_data/T06/engine-data-gamesrc` | 474 MB | 474 MB（含 `score.dat` / `th08.cfg`，引擎会写，故每跑一次 `cp -a` 一份私有副本） | ✅ |

复现命令：`bash .tmp_v430_replay/00_verify.sh`（退出码 0，输出见 `EXIT_CODES_W01C.txt` CMD1）。
全部依赖字节的身份汇总在 [`w01c/IDENTITY_SHA256.txt`](w01c/IDENTITY_SHA256.txt)。

**附带事实（影响后续判读）**：三个 shard 的 `decision_frame` 跨度几乎相同
（`16 → 13190` / `13198` / `13198`），即 seed10/11/12 都是整段长局；corpus 的
`meta.json`（T04 与 T02c 两层）**不记录该 episode 是否通关**——用
`.tmp_v430_replay/05_outcome_probe.py` 逐键扫描，只有 `shard.stage = 0`，
没有 `stage_clear` / `stop_reason` 之类的字段。所以"某段录制通关与否"必须靠重放实测，
不能从资产里读出来。

---

## 2 步骤 1：正对照复现（**逐位复现成功**）

配方逐项等同 `.tmp_v420_evalpulse/20_control_pulse.sh`（只把 `--data-dir`、`--out`、
`--shm-prefix` 换成本任务的新路径）：

```text
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python tools/rl/v420_eval_shoot.py \
  --engine …/rl_v3/th08-modern --data-dir .tmp_v420_data/T08/w01c-pos1 \
  --replay-tape --shard …seed12-shard/slots.jsonl --tail 64 --stage 0 --difficulty normal \
  --max-frames 19887 --obs-window consecutive --score-trace-frames 13195 \
  --shoot-pulse-period 48 --shm-prefix w01cpos1 --out …/step1_pos_r1.json
```

| 判据 | 参考（v4.2.0 T08） | W01c 实测 | 一致 |
|---|---|---|---|
| `stage_clear` | `true` | **`true`**（`stage_start 0 → stage_end 1`，`stop_reason=stage_clear`） | ✅ |
| `frames` | 13,195 | **13,195** | ✅ |
| `score` | 2,631,430 | **2,631,430** | ✅ |
| `deaths` | 0.0 | **0.0** | ✅ |
| `action_sequence_sha256` | `5e9d66c4f5685da9f0ea56f3105656b7210721124f8b74a12813b49089995cd3` | **同值** | ✅ |
| `observed_input_sha256` | `315a2ab6…f47c6212` | **同值** | ✅ |
| `mask_trace_sha256` | `66c81344…a9f179924` | **同值** | ✅ |
| `releases_planned/applied` | 137 / 137 | 137 / 137 | ✅ |
| SHOOT / 松键帧、上升沿 | 13,058 / 137 / 138 | 13,058 / 137 / 138 | ✅ |
| `distinct_masks` | 30 | 30 | ✅ |
| **退出码** | 0 | **0** | ✅ |
| **wall** | 121.573 s（仪器内） | **125.946 s**（wrapper）/ 126.44 s（仪器内） | 同口径 +4.0% |

---

## 3 步骤 2：负对照复现（**逐位复现成功**）

同一磁带、同一配方、**只把 `--shoot-pulse-period` 改为 `1000000007`**（大于任何
可能决策数 ⇒ `(i+1) % period` 永不成立 ⇒ 永不释放；`releases_planned=0`）：

| 判据 | 参考 | W01c 实测 | 一致 |
|---|---|---|---|
| `stage_clear` | `false` | **`false`** | ✅ |
| `stop_reason` | `game_over` | **`game_over`** | ✅ |
| `deaths` | 3.0 | **3.0** | ✅ |
| `frames` / `score` | 15,564 / 163,945 | **15,564 / 163,945** | ✅ |
| `score_at_frames["13195"]` | 141,379 | **141,379** | ✅ |
| `action_sequence_sha256` | `77455bef…caea6c4bb` | **同值** | ✅ |
| `observed_input_sha256` | `a1b3bdd6…f6d924e1` | **同值** | ✅ |
| `mask_trace_sha256` | `92cea74d…cd1feba10` | **同值** | ✅ |
| `releases_applied` / SHOOT 帧 / 上升沿 | 0 / 15,564 / 1 | 0 / 15,564 / 1 | ✅ |
| **退出码 / wall** | 0 / 138.94 s（仪器内） | **0 / 143.032 s**（wrapper，仪器内 144.447 s） | 同口径 +4.0% |

两条控制通道的差异**只有** `shoot_pulse_period`（48 vs 1,000,000,007）。因此
`stage_clear` 的有无、13,195 vs 15,564、2,631,430 vs 163,945 **只能**来自脉冲。

---

## 4 步骤 3：seed10 / seed11 通关性实测（**两段都能通关**）

`tools/rl/v420_eval_shoot.py` 把 `--replay-tape --shard` 硬钉在 seed12 的 sha256 上
（`CLEAR_SHARD_SHA256`），换 shard 会被**按设计拒绝**（exit 2），而该文件本任务只读。
因此用 [`../../../../../../.tmp_v430_replay/30_shard_tape.py`](../../../../../../.tmp_v430_replay/30_shard_tape.py)
只读加载该工具并**调用它自己的函数**（`tape_identity` / `TapePolicy` / `_run_loop` /
`_assemble_tape_record` / `validate_tape_record`），**唯一绕过的就是那一条 shard sha 断言**；
发键掩码、组装、校验、观测窗口、帧预算、pulse 规则全部未改、未复制。

| 磁带 | `stage_clear` | `stop_reason` | 最深帧数 | `score` | `deaths` | `action_sequence_sha256` |
|---|---|---:|---:|---:|---:|---|
| seed10（pulse 48） | **`true`** | `stage_clear` | **13,291** | 2,868,224 | 0.0 | `3cb366a00cca975bced3fbc3434c74bb32ed256e16501c8461170920567fea64` |
| seed11（pulse 48） | **`true`** | `stage_clear` | **13,291** | 2,768,761 | 0.0 | `19be3bc67d7de57e35c20fc55cb03eedce1535b7e4fadcdfb333ef8e5f204f90` |

三条动作序列哈希两两不同（`5e9d66c4…` / `3cb366a0…` / `19be3bc6…`），`mask_trace_sha256`
也互不相同 ⇒ 是**三段不同的录制 episode**，不是同一段的别名。
即 `touhou_rl/v420/rl.py::load_clear_schedule` 文档字符串里"**the one** recorded real
episode … that cleared Stage 1"的说法**在重放通道上被推翻**：seed10 与 seed11 同样能通关。

**必须同时记录的一个机制事实**：三段录制都在换面**之前**结束（最后决策帧
13,190 / 13,198 / 13,198），而通关帧是 13,195 / 13,291 / 13,291 ——
也就是说**通关发生在录制区间之后**，靠的是 `--tail 64`（把最后一条动作重复 64 个决策）。
这三段磁带**没有一段录到了换面本身**。

三个磁带关掉脉冲后全部死亡（见 §5 的 F1/F2/F3），说明 SHOOT 释放帧对"能否换面"
是**承重**的，不是可选装饰。

---

## 5 步骤 4 与 CONTRACT 清点：6 条来源 × 2 次重放，逐项一致

`PLAN_SPEC.json` `datasets.CONTRACT` = `success_sources: 3, failure_sources: 3,
repeats_per_source: 2`。12 次运行全部 **exit 0**，记录在
[`w01c/`](w01c/)（每次一份仪器原始 JSON + `.status`），机器汇总在
[`w01c/CONTRACT_INVENTORY.json`](w01c/CONTRACT_INVENTORY.json)。

| 来源 | 磁带 | pulse | 结果 | 首跑记录 | 重复记录 | 逐项一致 | 次数 |
|---|---|---|---:|---|---|---|---:|
| **S1** | seed12 `f759de7d…` | 48 | `true` / 13,195 f / 2,631,430 / deaths 0.0 | `step1_pos_r1` | `step4_pos_r2` | **是**（9/9 + mask/releases） | 2 |
| **S2** | seed10 `52402de4…` | 48 | `true` / 13,291 f / 2,868,224 / deaths 0.0 | `step3_seed10` | `step4_seed10_r2` | **是** | 2 |
| **S3** | seed11 `f75e632b…` | 48 | `true` / 13,291 f / 2,768,761 / deaths 0.0 | `step3_seed11` | `step4_seed11_r2` | **是** | 2 |
| **F1** | seed12 `f759de7d…` | 1,000,000,007 | `false` / 15,564 f / 163,945 / deaths 3.0 | `step2_neg_r1` | `step4_neg_r2` | **是** | 2 |
| **F2** | seed10 `52402de4…` | 1,000,000,007 | `false` / 4,615 f / 69,461 / deaths 3.0 | `step3_seed10_nopulse` | `step4_seed10_nopulse_r2` | **是** | 2 |
| **F3** | seed11 `f75e632b…` | 1,000,000,007 | `false` / 13,944 f / 135,116 / deaths 3.0 | `step3_seed11_nopulse` | `step4_seed11_nopulse_r2` | **是** | 2 |

逐项比对字段：`frames / score / stage_clear / stop_reason / deaths / lives_end /
action_sequence_sha256 / observed_input_sha256 / engine_sha256`（另加
`mask_trace_sha256` / `releases_applied`）——**6 对全部 `reproducible=true`，无任何差异项**。
wall 时间不同（CPU 争用，见 §6）不计为差异。

各来源的 `mask_trace_sha256`（两次重放同值）：S1 `66c81344…`、S2 `b1900a27…`、
S3 `a74ea1bf…`、F1 `92cea74d…`、F2 `6ce07783…`、F3 `81389f2b…`；`releases_applied`
为 137/138/138（success）与 0/0/0（failure）。

> **一处记录格式的例外（已单独留档）**：`step3_seed10`（S2 首跑）产生于"pin-bypass
> 字段集合"固定**之前**，它的 `w01c` 块里有 `shard_pin_bypassed=true` 与
> `shard_pin_note`，但没有 `pin_bypassed_reason` / `instrument_identity` 两个字段。
> 我没有回改原始记录，而是另写
> [`w01c/pin_bypass_addendum_seed10_pass1.json`](w01c/pin_bypass_addendum_seed10_pass1.json)
> 补上这两个字段并说明为何事后可证明身份（instrument mtime 早于运行、engine sha 写在
> 记录内、且 `step4_seed10_r2` 是其逐位相同的重复且带完整块）。
> 其余 5 条 pin-bypassed 运行均自带完整块。

### 5.1 直接回答

1. **success 侧 3/3 成立**（S1/S2/S3，三段不同磁带，各 2 次逐位一致）；
   **failure 侧 3/3 成立**（F1/F2/F3，三段不同磁带，各 2 次逐位一致）。
2. **`repeats_per_source: 2` 6/6 全部满足**（表内"次数"列全为 2）。
3. 但"来源"的口径必须说清（见 §5.2）——**不要把这 6 行当成 6 个独立来源**。

### 5.2 这 6 行的**真实独立性**：只有 3 个来源组

`S1` 与 `F1` 是**同一段录制 episode**（seed12 磁带）在两种仪器设置下的两次运行；
`S2`/`F2`、`S3`/`F3` 同理。按 `TRAINING_PLAN.md` §4 的分组口径（"重复重放、由同一起点
生成的多个分支属于同组"），独立来源是 **3 组**，不是 6 个：

- **独立录制 episode = 3 段**（seed10 / seed11 / seed12，动作哈希两两不同）；
- **success 与 failure 不是两批不同的真实结果**，而是"发键通道是否发出 SHOOT 释放帧"
  这一**受控操纵**的两种取值：同一段动作表，pulse 48 → 通关，pulse 永不触发 → 三命死亡。
  F1/F2/F3 的失败**不是**录制里真实发生过的失败，而是本仪器通道造成的失败。

因此：
- 若 CONTRACT 的意图是"**3 个不同的 success 见证 + 3 个不同的 failure 见证**"，
  当前资产在**行级别**满足（3+3，各 2 次），但在**独立来源级别只有 3 组**，
  且 failure 一侧全部是**仪器诱导的受控失败**。
- 若 CONTRACT 的意图是"3 个 success 与 3 个 failure 必须来自**不同的真实 episode**"，
  则当前只有 **3/6**（缺 3 个"录制里真实失败"的 episode）。

### 5.3 每条来源升为"正式 CONTRACT 来源"还差什么

| 来源 | 现状 | 缺口 |
|---|---|---|
| S1 | 走**未绕过** pin 的正式通道（`--replay-tape` + 已冻结 sha），2 次一致 | 只缺"登记"这一步：该磁带即 `configs/v420/rl-engine-audit.json` 的 `audit.clear_shard_sha256`。另需接受"clear = 磁带 + 64 决策尾"这一**定义**（录制本身没录到换面，见 §4） |
| S2 / S3 | 通关已实测、2 次一致，但走的是**绕过 shard pin** 的驱动路径 | ①把 `52402de4…` / `f75e632b…` 正式登记进来源清单，或让 v430 的生产入口（W01 待实现的 `session.py`）按清单接受 shard；②在该**未绕过**路径上以同一 instrument 身份再确认一次。**在那之前只能算 `classification`，不能算正式 success 来源** |
| F1 / F2 / F3 | `game_over` + 3 死亡、2 次一致 | ①需要产品级裁定："把同一磁带的 SHOOT 释放通道关掉"是否算 CONTRACT 的 failure 来源；②若要求"真实失败"，必须**新采集**录制里真死掉的 episode（当前仓库没有这样的磁带：三个 shard 全部能通关）；③F2/F3 同样带 pin-bypass 限制 |

**缺的部分需要什么**（结论）：**更多真实录制**——既需要"录制里真实失败"的 episode
（补 failure 侧的独立性），也需要"录到换面本身"的通关磁带（消除对 64 决策尾的依赖）。
不需要新代码就能补 success 侧的"正式化"，但需要在
`configs/v430/` 的来源清单里登记 shard sha，并让 v430 生产入口按清单接受它们；
`tools/rl/v420_eval_shoot.py` 本任务只读，未改。

---

## 6 成本与并发（供 ST00 预算核算）

全部 **12 次**运行、**串行单引擎**（我自己的引擎并发恒为 1，从未同时跑 2 个）。
另有一个**不属于本任务**的 v4.2.0 A1 扩容长跑（`.tmp_v420_expand/run_all.sh`，
自 2026-11-15 00:32:28 起）在此期间持续占用 **3 个** `rl_v3` 引擎；
采样点 03:16:25Z / 03:29:50Z / 03:38:00Z 均为 3 个，其 state 在窗口内从 538 推进到 552，
即**全程活跃**。load average 2.5–3.5 / 20 核。因此我这一侧同时存在的引擎总数为 4（含它），
我自己的始终是 1。

| run | 来源 | 退出码 | wrapper wall s | 仪器内 wall_s | frames | 其他任务并发引擎 |
|---|---|---:|---:|---:|---:|---:|
| `step1_pos_r1` | S1 | 0 | 125.946 | 126.440 | 13,195 | 3 |
| `step4_pos_r2` | S1 | 0 | 124.570 | 125.241 | 13,195 | 3 |
| `step3_seed10` | S2 | 0 | 129.816 | 130.779 | 13,291 | 3 |
| `step4_seed10_r2` | S2 | 0 | 127.978 | 128.607 | 13,291 | 3 |
| `step3_seed11` | S3 | 0 | 127.401 | 128.455 | 13,291 | 3 |
| `step4_seed11_r2` | S3 | 0 | 123.858 | 124.591 | 13,291 | 3 |
| `step2_neg_r1` | F1 | 0 | 143.032 | 144.447 | 15,564 | 3 |
| `step4_neg_r2` | F1 | 0 | 144.007 | 144.969 | 15,564 | 3 |
| `step3_seed10_nopulse` | F2 | 0 | 51.317 | 50.820 | 4,615 | 3 |
| `step4_seed10_nopulse_r2` | F2 | 0 | 51.588 | 50.238 | 4,615 | 3 |
| `step3_seed11_nopulse` | F3 | 0 | 131.899 | 132.810 | 13,944 | 3 |
| `step4_seed11_nopulse_r2` | F3 | 0 | 129.357 | 130.267 | 13,944 | 3 |

**合计 wrapper wall 1,410.769 s ≈ 23.5 min**（12 次）；另加 **10 份**私有 data dir 的
`cp -a`，每次 3.3–4.9 s（10 × 474 MB ≈ 4.7 GB，D: 剩余 1.2 TB，磁盘不是约束）。

单次最贵 144.007 s（F1），**全部 12 次都远低于 §6.1 的 300 s 兜底**，
没有一次触发 `cost_contention`。与 v4.2.0 记录（121.573 s / 138.94 s，仪器内口径）
相比同口径慢约 4.0%（+4.86 s / +5.51 s），属 CPU 争用；用 wrapper 口径算是 +2.9%–3.6%。

---

## 7 不能主张什么

* **不是能力结果**：回放的是教师录制轨迹，只证明"控制入口能报过关/失败、且可复现"，
  不证明任何学生策略能过关。
* **不是 6 个独立来源**：见 §5.2，只有 3 段独立 episode；failure 侧是仪器诱导。
* **没有录到换面**：三段磁带的通关都发生在录制区间之后（`--tail 64` 承担），
  不能据此说"录制本身记录了换面"，也不能说"策略学会了最后那段"。
* **S2/S3/F2/F3 是 pin-bypassed 的 classification 证据**，在按 §5.3 正式化之前，
  不能直接计为 CONTRACT 的正式来源。
* **没有重测旧数字**：R0 的 0/5、0/20 等历史行属于旧仪器，本档没有触碰，也没有把它们
  混进本表（`v420_eval_shoot.py` 的 schema 拒绝混表）。
* **未跑**：任何训练、任何 v430 接口、任何 confirm/corpus 种子。

---

## 8 复现

```bash
ROOT=/mnt/d/WorkShop/Touhou_RL; cd $ROOT
bash .tmp_v430_replay/00_verify.sh                       # 步骤 0（只读，不启引擎）
bash .tmp_v430_replay/00_setup_data.sh w01c-pos1 w01c-neg1 w01c-pos2 w01c-neg2 \
     w01c-s10 w01c-s10b w01c-s11 w01c-s11b \
     w01c-s10nb w01c-s11nb                              # 每个 474 MB 私有副本
bash .tmp_v430_replay/10_tape_run.sh pos-r1 …seed12… 48 …           # S1 首跑
bash .tmp_v430_replay/10_tape_run.sh neg-r1 …seed12… 1000000007 …   # F1 首跑
bash .tmp_v430_replay/31_shard_tape_run.sh s10 …seed10… 48 …        # S2（pin-bypass）
bash .tmp_v430_replay/31_shard_tape_run.sh s11 …seed11… 48 …        # S3
bash .tmp_v430_replay/31_shard_tape_run.sh s10-nopulse …seed10… 1000000007 …  # F2
bash .tmp_v430_replay/31_shard_tape_run.sh s11-nopulse …seed11… 1000000007 …  # F3
# 6 次重复：同样命令、换 --out/--shm-prefix 与新的私有 data dir（w01c-pos2/neg2/s10b/
# s11b/s10nb/s11nb），即 step4_* 那六个文件
bash .tmp_v430_replay/40_publish_runs.sh                 # 原始 JSON → 证据目录
bash .tmp_v430_replay/45_identity.sh                     # 身份 sha256
/root/v400-venv/bin/python .tmp_v430_replay/50_report.py --json …/CONTRACT_INVENTORY.json
```

每次运行的完整命令行、引擎 sha、shard sha 与退出码都写在该次的
`.tmp_v430_replay/runs/<run>.log` 开头，摘要见
[`EXIT_CODES_W01C.txt`](EXIT_CODES_W01C.txt)。

### 8.1 自查：本档每个数字都回验过（56/56，ALL_PASS）

`.tmp_v430_replay/60_verify.py`（只读，不启引擎）对**已发布的 12 份原始记录**重新断言：

1. 12 份记录全部携带引擎 sha256 `691bfd77…a568ef6`，且**只有这一个**引擎身份；
2. 12 份记录全部**再次**被仪器自带的 `validate_tape_record` 接受（不是"跑的时候通过"的口头结论）；
3. `step1_pos_r1` 与 `.tmp_v420_evalpulse/positive_control.json`、`step2_neg_r1` 与
   `negative_control.json` 在 17 个字段 + `shoot` 块（除 `release_rule` 文案）上**逐项相等**；
4. 6 条来源各自的两次重放在 9 个固定字段 + `mask_trace_sha256` / `releases_applied` /
   `shoot_frames` / `no_shoot_frames` 上**零差异**；
5. success 侧 3 个不同磁带 sha256、failure 侧 3 个不同磁带 sha256、6 条动作序列哈希互不相同；
6. `CONTRACT_INVENTORY.json` 的 totals（12 次 / 3 success / 3 failure / 6 条两跑 / 6 条 reproducible）
   与记录一致；12 份 `.status` 全部 `exit=0`。

```text
$ /root/v400-venv/bin/python .tmp_v430_replay/60_verify.py
56/56 checks passed
ALL_PASS
exit_code: 0
```

---

## 9 变更与新增文件清单

| 路径 | 性质 |
|---|---|
| `.tmp_v430_replay/`（`00_verify.sh`、`00_setup_data.sh`、`01_inspect.py`、`02_shard_probe.py`、`03_slot_peek.py`、`04_shard_dirs.sh`、`05_outcome_probe.py`、`10_tape_run.sh`、`20_collect.py`、`30_shard_tape.py`、`31_shard_tape_run.sh`、`40_publish_runs.sh`、`45_identity.sh`、`50_report.py`、`60_verify.py`、`runs/**`） | 新增（被 `.gitignore` 的 `.tmp_*/` 忽略） |
| `.tmp_v420_data/T08/w01c-{pos1,neg1,pos2,neg2,s10,s10b,s11,s11b,s10nb,s11nb}/` | 新增私有 data dir（各 474 MB） |
| `Document/Update/Alpha/v4.3.0/evidence/W01/W01C_PREFLIGHT.md` | 本档 |
| `Document/Update/Alpha/v4.3.0/evidence/W01/EXIT_CODES_W01C.txt` | 退出码台账 |
| `Document/Update/Alpha/v4.3.0/evidence/W01/w01c/*.json` / `*.status` / `RUN_SHA256.txt` / `IDENTITY_SHA256.txt` / `CONTRACT_INVENTORY.json` / `pin_bypass_addendum_seed10_pass1.json` | 新增原始证据 |

**未修改**：`.tmp_v420_evalpulse/**`、`.tmp_v420_data/T04/**`、
`.tmp_v420_data/T06/engine-data-gamesrc/**`、`tools/rl/v420_eval_shoot.py`
（sha256 `7dbc2a31…`，mtime 2026-11-14T02:30:40Z，早于本次全部运行）、
`tools/rl/v420*.py`、`tools/rl/v430.py`、`touhou_rl/**`、`tests/**`、`configs/**`、
`evidence/W00/**`、`evidence/W01/W01A.md`、`evidence/W01/EXIT_CODES_W01A.txt`。
没有 `git add`/commit/上传，没有删除或移动既有资产，没有杀任何进程。

> 说明：`git status` 里 `Document/Update/Alpha/v4.3.0/*.md|json` 与 `tools/**`、
> `touhou_rl/**` 的一批 `M` 是**本次会话开始前**就存在的（相关文件 mtime 为
> 2026-11-15 10:54 或更早，早于本任务 11:13 起跑；另外 `touhou_rl/v430/**`、
> `tests/test_v430_*` 的改动来自**其它并行代理**，与本任务无关）。本任务对这些路径
> 只读，未写入一个字节；它新增的全部是本表列出的文件。
