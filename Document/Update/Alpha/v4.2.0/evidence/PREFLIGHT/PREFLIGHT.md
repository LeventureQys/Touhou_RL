# PREFLIGHT — 执行前环境与旧契约实测

- 时间：2026-11-13T17:2x+08:00
- 作用：在派发 T00–T07 之前，先证明「资产主机能跑真实引擎评测」，并实测旧 S1 数据的
  时间/动作契约。**这不是任何 G420 门的通过证据**，只是执行前置与诊断输入。

## 1 环境实测

| 项 | 值 | 命令 / 来源 |
|---|---|---|
| 资产主机 | WSL2 Ubuntu，`cwd=/mnt/d/WorkShop/Touhou_RL` | 历史 v4.1.0 证据一致 |
| Python | `/root/v400-venv/bin/python` → 3.10.12 | `python -V` |
| torch | 2.14.1+cu130，`cuda_available=True` | 同上 |
| GPU | RTX 4070 SUPER 12,282 MiB，驱动 617.42 | `nvidia-smi` |
| 引擎 rl_v3 | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` | `sha256sum` |
| 引擎 rl_laser | `1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f` | `sha256sum` |
| 游戏数据 | `/root/v400-data/game`，474 MB | `du -sh` |
| B0 checkpoint | `3bc0ec4461701bebb2ffc551d61dffff77f15cd486125baf9db6eaafc741b3a6` | `sha256sum` |
| Windows Python | 3.12.10，torch 2.14.1+cu132，`cuda_available=True` | 交叉检查用 |
| `/root/Touhou_RL` | HEAD `d55c26a`（**旧检出，无 v410 代码**），不要在那里跑 v4.2.0 代码 | `git log` |

### 1.1 执行期间的并发占用（必须记录）

v4.2.0 README §2.1 与 v4.1.0 `state.json` 都写「round3 暂停、4 shard / 14,000 decisions」。
实测发现**实际并非暂停**：一个 v4.1.0 采集进程仍在跑。

```text
PID 1884178 (PPID 1884157, xvfb-run 包装), 启动于 2026-11-13T16:28 +08:00
/root/v400-venv/bin/python tools/rl/v410_dagger_collect.py \
  --checkpoint .tmp_v410_wp05_train_beta05_20k/b0_s0_dagger_step20000.pt \
  --seeds 2044-2054 --beta 0.5 --decisions 3500 --max-frames 20000 \
  --stage 0 --difficulty normal --mode fast \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --data-root .tmp_v410_wp05_round3/data --shard-root .tmp_v410_wp05_round3 \
  --work-dir .tmp_v410_wp05_round3/logs \
  --continue-on-error --max-seconds 9000 \
  --out .tmp_v410_wp05_round3/round3_batchB.json
子进程 th08-modern env-2051 / shadow-2051
```

处理方式与理由：

- **不杀**。按 [AGENTS.md](../../../../AGENTS.md) §9「只管理自己拥有的进程，不根据旧 PID 杀其他任务」，
  这不是 v4.2.0 派发的进程；且它有 `--max-seconds 9000`（≈2.5 h）自限，约 19:00 自行结束。
- 它是**旧快照与实际状态不一致**的实例，按 T00 要求只在 v4.2.0 新记录中说明，
  **不修改** v4.1.0 的 `state.json` / `journal.jsonl` / `budget.jsonl`。
- 影响：WSL `load average ≈ 22`（20 vCPU）。它**不改变**确定性结论（frame 数、动作 SHA、
  HASH 都由引擎与权重决定），但会拉长墙钟；所以本版本记录的墙钟时间一律标注为
  「**争用条件下的观测值**」，不作为吞吐承诺。
- 对本版本的后续影响：T02 的复现只看帧数/动作 SHA/终止原因，不看墙钟；T05/T06 的
  采集与训练成本估算必须在争用结束后重测。


## 2 真实引擎单局 smoke（一次成功，非门）

脚本：`.tmp_v420_smoke_eval.sh`。命令：

```bash
cd /mnt/d/WorkShop/Touhou_RL
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_student_eval.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp04_arms/b0_seed0_last.pt \
  --seeds 1000 --stage 0 --difficulty normal \
  --max-frames 19887 --stop-on-death --device cpu \
  --out .tmp_v420_data/smoke/b0_seed1000_smoke.json
```

结果（原始 JSON：[`b0_seed1000_smoke.json`](b0_seed1000_smoke.json)）：

- **退出码 0**，墙钟 **61 s**（单局，`wall_s=54.951`）。
- seed 1000：**1,398 帧 / 699 决策 / 1 死亡 / `death_stop` / 0 过关**，`stage_clear=false`。
- `arm=B0`、731,802 参数、`obs_spec_digest=ef2fbd15…`、`checkpoint_sha256=3bc0ec44…`、`engine_sha256=691bfd77…`。
- **与 v4.1.0 历史记录一致**：`WP05/STATUS.md` §1.3 记载 B0 baseline seed1000 = **1,398 帧**，本次复现同值。这一条支持「原 evaluator 可重复运行」，但 T02 仍要按 3 seed × 2 重复正式复现。

### 2.0 三次重复的确定性（G420-0 的输入）

同一命令再跑两次（脚本 `.tmp_v420_repro.sh`，原始 JSON：
[`b0_seed1000_rep2.json`](b0_seed1000_rep2.json)、[`b0_seed1000_rep3.json`](b0_seed1000_rep3.json)）：

| 重复 | 退出码 | 墙钟 | frames | score | action_sequence_sha256 | policy_input_sha256 |
|---|---:|---:|---:|---:|---|---|
| #1 | 0 | 61 s | 1398 | 13102 | `85a528fd…` | `41d171cc…` |
| #2 | 0 | 59 s | 1398 | 13102 | `85a528fd…` | `41d171cc…` |
| #3 | 0 | 47 s | 1398 | 13102 | `85a528fd…` | `41d171cc…` |

**三项完全一致**：frames、score、动作序列 SHA、策略输入 SHA 全同 ⇒ 原 evaluator 在 CPU 上
逐决策可复现，且与 v4.1.0 历史值（1,398 帧）一致。这是「原模型可重复运行」的直接证据，
**但只覆盖 1 个 seed 1 个 checkpoint**；T02 仍须做 B0/20k × 3 seed × 2 重复的正式复现。

### 2.1 单局成本外推

61 s/局 ⇒ 36 局（T02：B0/20k × 3 seed × 2 重复）≈ **37 min**；G420-4 的 100 局 ≈ **1.7 h**。
这是 dev 短局（早死）成本，不是过面成本。

## 3 旧 S1 数据的时间/动作契约实测（G420-0 的直接输入）

对象：`.tmp_v410_s1c/d1/shards/d1-s0d1-n3300-seed10-shard`（3,300 行 = 13,200 帧）。
脚本：`.tmp_v420_slot_probe.py`、`.tmp_v420_hist_probe.py`、`.tmp_v420_mask_probe.py`、`.tmp_v420_release_probe.py`。

| 实测项 | 值 | 对 2 帧契约的含义 |
|---|---|---|
| 每行推进逻辑帧 | **4**（`decision_frame` 步长直方图 `{4: 3299}`） | 决策节拍是 4，不是 2 |
| 行内自述 | `frames_per_decision=4` 但 `decision_frames=2` | 同一行自相矛盾 |
| 槽位覆盖 | 6,600 槽，仅 **3,300** 有自己观测与标签（50%） | 槽 1 有动作、无观测、无 scores |
| 标签归属 | `shard.json.alignment.label_is_first_slot_not_continuation = true`；`scores[18]` 只评 slot 0，作用于 `[d, d+2)` | 标签只覆盖第一个槽 |
| `chosen_plan.duration` | 2 → 588 行；1 → 2,712 行 | 82.2% 的行教师计划只覆盖 1 槽，但采集器仍推进 4 帧 |
| 槽 1 动作 | 每行可从 `chosen_plan.slots[1]` 恢复（3,300/3,300） | 可恢复性 ≠ 可监督性 |
| `step_frames[0:2]` vs `cand_to_mask(slots[0], bomb)` | **3,300/3,300 相等** | 槽 0 掩码可复算 |
| `step_frames[2:4]` vs `cand_to_mask(slots[1], 0)` | **3,163/3,300 相等**，137 不等 | 见下一行 |
| 137 处差异性质 | **全部** `shoot_release=true`，**全部**只发生在块内第 4 帧（清开火位：0x85→0x84、0x41→0x40） | 开火释放是 **4 帧块周期**行为，2 帧节拍下无对应 |
| `act_hist` 组成 | `act_hist[r]` = `first_action[r-8..r-1]` one-hot，前补零，**0 失配** | 历史 = 最近 8 个**教师决策** |
| 历史时间跨度 | 8 × 4 帧 = **32 逻辑帧** | 部署侧 8 × 2 = **16 帧**，相差一倍 |
| 历史是否含槽 1 已执行动作 | **否** | 历史是「每隔一个动作」，不是「实际最近 8 个 2 帧动作」 |
| HASH 链 `hash_post[i] == hash_pre[i+1]` | **0 断点** | 这一项旧数据是干净的 |
| `bomb` | 全 0 | 旧 B0 无 bomb 策略，v4.2.0 固定 0 有依据 |
| 像素窗口 | `play96x112` (N,4,112,96)、`hud` (N,56,160) | 张量形状可原样沿用 obs-v3 |

### 3.1 结论

旧 S1 的**像素窗口**确实连续（4 帧真实连续渲染），但**时间/动作契约**与部署不一致，三项结构性缺口：

1. **槽位覆盖 50%**：槽 1 已执行动作，但没有属于它的观测、scores 或验证结果；
2. **历史语义 32 帧 vs 部署 16 帧**，且漏掉槽 1 实际执行的动作；
3. **开火释放是块周期**行为（每 4 帧末清一次开火位），2 帧节拍下不存在该边界。

因此旧数据不能按「2 帧样本」直接复用；`WP420-01` 的「新 S1 每个 2 帧决策槽都要有对应标签」是**必须新采/新查询**的工作，不能用复制上一条标签、插值 logits 或一条计两条补齐。

### 3.2 与 v4.1.0 evaluator 自述的冲突

v4.1.0 evaluator 在每局记录里写
`window_alignment.matches_training_obs = true`，理由只覆盖**像素窗口连续性**
（"one decision = two frames = two renders, so the stack is consecutive"）。
它没有涵盖 **决策节拍（4 vs 2）** 与 **历史跨度（32 vs 16 帧）**。
所以：`matches_training_obs=true` 不能作为 G420-0 的时间契约通过证据；这正是
[v4.2.0 计划](../README.md) §3 第 1 条要求的核查。

## 4 历史基准（T02 复现用的期望值，只读提取）

来源：`.tmp_v410_wp04_eval_arms/B0_seed0_full.json`（B0，20 dev）与
`.tmp_v410_wp05_train_beta05_20k/eval_dev20.json`（20k，20 dev）。两者均**只读**引用，
未修改。提取脚本 `.tmp_v420_expected.py`。

| checkpoint | seed | frames | steps | score | stop_reason | action_sequence_sha256 |
|---|---|---:|---:|---:|---|---|
| B0 `3bc0ec44…` | 1000 | 1398 | 699 | 13102 | death_stop | `85a528fd…` |
| B0 | 1001 | 818 | 409 | 1512 | death_stop | `57f61f99…` |
| B0 | 1002 | 1251 | 626 | 7438 | death_stop | `864101ae…` |
| 20k `15ae1fd5…` | 1000 | 1612 | 806 | 12471 | death_stop | `c2ee8fc1…` |
| 20k | 1001 | 1970 | 985 | 18920 | death_stop | `d2cd9917…` |
| 20k | 1002 | 1532 | 766 | 12124 | death_stop | `e86f54d5…` |

20 dev 汇总（历史）：B0 `frames_mean=1453.25`、20k `frames_mean=1714.65`、两者 `clears=0`。

**本 PREFLIGHT 已完成其中 1 格**：B0/seed1000 三次重复均为 1398 / 13102 / `85a528fd…`，
与历史基准逐项相同。其余 5 格由 T02 的复现腿补齐。

## 5 HASH 重建实测（G420-0 的「HASH 重建零未解释差异」腿）

脚本 [`replay/replay_probe.py`](replay/replay_probe.py)（原样保存）。做法：

1. 读 shard 的 `labels.jsonl`，取每行的 `step_frames`（4 个逐帧按钮掩码，已实测与
   `cand_to_mask` 逐槽一致）；
2. 用 `Th08Env(engine, data_dir, role="env")` + `StageStart(stage=0, difficulty="normal",
   team=0, seed=10, resources="standard")` 复现开局；
3. 逐决策推进 4 帧（`advance(mask, 1)` ×4），把第 4 帧的引擎 HASH 与记录的
   `hash_post` 逐条比对；
4. **引擎跑在 shard 自带 data 目录的副本上**（`.tmp_v420_data/replay/env-seed10/`，
   复制 `score.dat` / `th08.cfg`，`th08.dat`、`thbgm.dat` 走符号链接），
   避免写坏语料的 `score.dat`。

seed10 / 500 决策 / 2,000 帧的实测结果（原始 JSON
[`replay/seed10_500.json`](replay/seed10_500.json)，186 KB，含逐决策 4 个逐帧 HASH）：

| 项 | 值 |
|---|---|
| 退出码 | **0** |
| 墙钟 | 34.7 s（争用条件下） |
| `reset_hash` vs 首行 `hash_pre` | `c2b684a7a564c2d3` == `c2b684a7a564c2d3`，**match** |
| 重放决策 / 帧 | 500 / 2,000 |
| `hash_post` 不一致数 | **0**（`first_mismatch=null`） |
| 异常 | 无 |

含义：**旧 S1 的逐帧掩码可以逐比特重建引擎状态**，HASH 链在 2,000 帧切片上零差异。
这一条支持「旧语料在**已记录的那 4 帧节拍**上是可信的」；它**不能**替代 2 帧契约核查
（槽 1 仍无观测/无标签），两者是不同的问题。

其余 2 个 seed（11、12）× 2 次重复由 `.tmp_v420_replay_all.sh` 补齐，结果见
[`PREFLIGHT_REPLAY.md`](PREFLIGHT_REPLAY.md)。

## 6 下一步

- T00：资产主机路径映射与对象核验（进行中）；
- T01：`touhou_rl/v420/` 与 `tools/rl/v420.py`（进行中，规格见
  [`../experiments/T01-contract-spec.md`](../experiments/T01-contract-spec.md)）；
- T02：B0/20k 各 3 dev seed × 2 重复 ≥ 2,000 帧 + 2 帧契约在线审计 → G420-0 判定。
