# WP04.3 — v410 BC checkpoint 的真实引擎开发局评测入口 + B0_s0 双 dev seed smoke

- 时间：2026-11-13T05:00:00+08:00
- 结论：**评测入口已落地并通过 111 项无引擎单测 + compileall；B0_s0 checkpoint 在真实 `rl_v3` 引擎上跑完 2 个 dev seed（1000/1001），两次都在首死处停止（1,398 / 818 帧），0 次 stage_clear。** 这 2 个 episode **不是开发成绩**、**不是通过率**、**不构成 G410-2 结论**。
- 范围：只做评测入口与 smoke；**未训练、未采数据、未做正式门、未使用 confirm split**。

## 0. 关键数字一览

| 项 | 值 |
|---|---|
| checkpoint | `.tmp_v410_wp04_arms/b0_seed0_last.pt`（B0，train seed 0，20 epoch） |
| checkpoint SHA-256 | `3bc0ec4461701bebb2ffc551d61dffff77f15cd486125baf9db6eaafc741b3a6` |
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern` |
| 引擎 SHA-256 | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` |
| 评测口径 | S1N：`--stage 0`（0-based，Stage1）+ `normal` + `--max-frames 19887` + `--stop-on-death` |
| dev seeds | 1000, 1001（仅 dev；confirm 60000–60199 与语料 seed 1–20 被拒绝） |
| seed 1000 | **1,398 帧**，1 死亡（首死停止），score 13,102，**未过关** |
| seed 1001 | **818 帧**，1 死亡（首死停止），score 1,512，**未过关** |
| pytest | `tests/test_v410_student_eval.py` → **111 passed**，exit 0 |
| compileall | `tools/rl/v410_student_eval.py` → exit 0 |
| 真实 smoke | exit 0（两个 episode 都取得真实 stop_reason，无 `error`/`reset_failed`） |

## 1. 交付物

| 文件 | 作用 |
|---|---|
| [`tools/rl/v410_student_eval.py`](../../../../../../tools/rl/v410_student_eval.py) | 评测入口：加载 v410 checkpoint → 构造 ProbeNet（B0）→ 真实引擎 `env` 角色像素 readback → 每 2 帧决策 → 写 `eval_smoke.json` |
| [`tests/test_v410_student_eval.py`](../../../../../../tests/test_v410_student_eval.py) | 111 项无引擎单测（fake checkpoint / obs / 纯函数 / CLI fail-closed） |
| [`eval_smoke.json`](eval_smoke.json) | 本次 smoke 的机器可读证据（12,833 bytes） |
| `.tmp_v410_wp04_eval/` | 运行目录（dry-run、日志、探针脚本），**不入库** |

未修改 `tools/rl/v400_student_eval.py`、`touhou_rl/v410/`、`third_party/`、`state/journal/budget/assets` 或任何既有证据；未 commit/push。

## 2. 命令与退出码（实际运行）

所有 PowerShell 命令先设 `$env:PYTHONPATH='D:\WorkShop\Touhou_RL'`。

### 2.1 单测

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp043_pytest -q tests/test_v410_student_eval.py
# 111 passed in 1.59s
# EXIT_CODE=0
```

### 2.2 语法校验

```powershell
python -m compileall -q tools/rl/v410_student_eval.py
# EXIT_CODE=0
```

### 2.3 真实 smoke（WSL，`xvfb-run` 提供 SDL 显示）

```bash
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_student_eval.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp04_arms/b0_seed0_last.pt \
  --seeds 1000,1001 --stage 0 --difficulty normal --max-frames 19887 --stop-on-death \
  --out Document/Update/Alpha/v4.1.0/evidence/WP04/eval_smoke.json
# wrote .../eval_smoke.json (exit 0)
# EXIT_CODE=0
```

smoke 前另跑了一次 **60 帧 1 seed 的 dry run**（同一命令，`--max-frames 60`，输出到 `.tmp_v410_wp04_eval/dryrun_smoke.json`），用于先验证引擎/env-role/readback 接线，exit 0。它**不是证据**，只说明接线可用。

### 2.4 独立复核（WSL）

```bash
sha256sum .tmp_v410_wp04_arms/b0_seed0_last.pt \
  /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern
# 3bc0ec4461701bebb2ffc551d61dffff77f15cd486125baf9db6eaafc741b3a6  b0_seed0_last.pt
# 691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6  th08-modern
```

两个 SHA 与 `.tmp_v410_wp04_arms/B0_s0.json` 记录的 checkpoint SHA 及任务给定的引擎 SHA（`691bfd77…`）一致。

## 3. 实现要点

### 3.1 checkpoint 加载（fail-closed 阶梯）

`load_v410_checkpoint()` 按顺序拒绝，任一步失败都抛 `EvalError`，绝不部分加载：

1. 文件存在且可读 → 2. `torch.load` 可解析且 payload 是 dict → 3. `schema ∈ {v410-wp04-bc-arm-checkpoint/1, v410-wp04-bc-checkpoint/1}` → 4. 必需字段 `schema/arm/model_config/state_dict/seed/epochs` 齐全 → 5. `bc.formal_arm_spec(arm)` 认领该 arm → 6. **只接受 probe 家族且无 bomb head**（B1 = probe+bomb head 拒绝；B2/B3 = spatial 家族拒绝）→ 7. `model_class` 必须等于 `touhou_rl.v410.student.diagnostics.ProbeNet` → 8. `model_config` 6 个键齐全 → 9. `bc.build_model()` 参数钉死（B0 = **731,802**）→ 10. payload / arm_spec 的参数数三方一致 → 11. `state_dict` 严格 `load_state_dict(strict=True)` → 12. `model.eval()` + 全部参数 `requires_grad=False`。

### 3.2 观测与决策

- 引擎以 `--rl-control 0,1 --rl-role env` 启动（复用 `v400_student_eval.EnvHarnessClient`，**只读**），`READBACK 1` + `SHM` 挂载后从共享内存取 640×480 RGBA。
- policy 输入是冻结的 **obs-v3**：`play96x112` 4 帧栈 `(4,112,96)`、HUD `(1,56,160)`、动作历史 `(8,18)`；归一化直接调用 `obs_spec_v3.normalize_play / normalize_hud / normalize_action_history`（与 `s1_data.pack_batch` 完全同一函数，训练/部署同构）。
- **每 2 帧一次决策**（`DECISION_FRAMES = 2`，`--obs-window consecutive`：每个游戏帧渲染一次，4 帧栈=t-3..t，与训练窗口一致）；`units×want = 2` 帧，帧口径与 v400 S1N 相同。
- 动作 = 18 维 move logits 的 **argmax**（无采样、无温度；`NaN/Inf` 直接拒绝；并列取最小索引）；B0 无 bomb head，动作恒为 `(cand, bomb=0)`。
- 每步一次 `STATE` 作裁判：stage 前进 → `stage_clear` 结束；死亡且还有残机 → `--stop-on-death` 时 `death_stop` 结束；残机为 0 → `game_over`；帧数达 cap → `frame_cap`。
- 两类哈希写进记录：`policy_input_sha256`（每个决策的 uint8 观测平面拼接哈希）与 `action_sequence_sha256`（动作序列字节哈希），可复现比对。

### 3.3 seed 策略（fail-closed，无覆盖开关）

`check_seed_policy()` 依次拒绝：confirm split **60000–60199**（先报 confirm）→ D410-S1 语料 seed **1–20**（train 1–14 / val 15–17 / test 18–20）→ 非 **1000–1099** 的 dev 窗口。`parse_seeds()` 支持 `1000,1001` / `1000-1004` / 混合，逐个过策略；**一个坏 seed 拒绝整条命令**；上限 100（整个 dev 池）。CLI 在**任何引擎进程存在之前**完成 seed 校验（单测用不存在的 engine/data 路径验证了这一点）。

### 3.4 episode 记录字段

`seed / seed_role / frames / steps / renders / deaths / lives_end / bombs_used / score / stop_reason / exit / stage_start / stage_end / stage_clear / actions{histogram,decisions,action0_share,unique_actions} / action_sequence_sha256 / policy_input_sha256 / checkpoint(+schema/arm/model_class/parameters/train_seed/epochs,+SHA) / obs_spec_version+digest+obs_shapes / engine+SHA / wall_s / harness_role / not_a_gate / not_a_dev_score / not_the_confirm_split / error`。

`validate_episode_record()` 复核：类型、dev seed、obs spec/形状、`stage_clear == (stage_end > stage_start)`、直方图和 == `steps`、`frames ≤ max_frames`、SHA 十六进制、失败 episode 必带 `error` 且成功 episode 必无 `error`。

## 4. smoke 结果（真实，未修饰）

checkpoint：B0_s0，20 epoch，离线 **train top1 0.9769 / val top1 0.6280 / val macro recall 0.3080**（val 9,874 样本）。

| seed | frames | steps(=决策) | deaths | stop_reason | stage_clear | stage_start→end | score | lives_end | bombs_used | wall_s |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | **1,398** | 699 | 1.0 | `death_stop` | **false** | 0→0 | 13,102 | 2.0 | 0.0 | 20.104 |
| 1001 | **818** | 409 | 1.0 | `death_stop` | **false** | 0→0 | 1,512 | 2.0 | 0.0 | 14.258 |

- `deaths=1.0` 是 **`--stop-on-death` 口径的构造值**（首次死亡即停），不是"整局只死一次"；有信息量的是 **首死帧数**。两次 `lives_end=2.0` 说明首死发生在 3 残机开局后的第 1 次。
- 两次都 **未过关**；`stage_end` 仍为 0，`frames` 是纯 Stage-1 时间。`harness_role="env"`、`pixels_warmup_frames=0`（readback 首帧即有效）、`error=null`。
- 动作分布（699 / 409 次决策）：

| seed | idle(0) | N | NE | E | SE | S | SW | W | NW | focus-idle | focus NE | focus E | focus SW | 其它 focus | action0_share | unique |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 295 | 1 | 42 | 107 | 3 | 1 | 99 | 98 | 3 | 0 | 13 | 32 | 5 | 0 | 0.4220 | 12 |
| 1001 | 208 | 2 | 11 | 39 | 0 | 0 | 75 | 32 | 0 | 0 | 8 | 34 | 0 | 0 | 0.5086 | 8 |

  两次都明显偏向 **idle（42%/51%）+ 左右横移**，向下（S/SE）几乎不用（seed1001 的 S/SE 为 0）。方向上与离线 `val macro recall 0.3080`（18 类平均召回很低）一致；这里只记录直方图，不据此推断原因。
- 描述统计（**不是结果**）：frames n=2、mean 1,108.0、min 818、max 1,398、sd 410.12，`ci_reportable=false`；clears 0；stop_reasons `{death_stop: 2}`。`eval_smoke.json` 的 `verdict` 明确写着 `is_a_gate=false / is_a_development_score=false / is_a_pass_rate=false / is_a_four_arm_comparison=false / conclusion_allowed=false`。

**参考行（仅作上下文，不是比较）**：同一 S1N 口径下，`teacher_s1n`（20 dev seeds，EXP-P1-012）mean 13,565.2 帧、20/20 过关、0 死亡（来源 `Document/Update/Alpha/v4.0.0/experiments/EXP-P1-012-g1-evaluation.md`，由 `tools/rl/v400_student_eval.py` 的 `REFERENCE_STUDENT_ROWS` 转抄）；而 `bench/scoreboard.md` 的 random/still 行是 **game-over 口径**（deaths=3.0），与本次 `stop-on-death` 的首死帧数**不同口径，不可直接对比**。按 EXP-P3-009 的判据模板，两个臂要各自给出同水平 CI 且 CI 不重叠才能称为"有效应"；本次 n=2 无 CI，因此**不作任何效应/方向结论**。

## 5. 测试覆盖（111 项，`tests/test_v410_student_eval.py`，实际收集数）

- **模块常量（2 项）**：schema/seed 窗口/18 动作/obs 形状/stop_reason 集合/checkpoint schema 集合；`DECISION_FRAMES=2` 与 `window_alignment`（consecutive 2 renders/decision、strided 1）。
- **checkpoint 加载（19 项）**：真实 state_dict 往返 → 731,802 参数 B0、eval + 冻结梯度、逐张量等于 checkpoint、与直接 forward **bitwise** 一致；拒绝：缺文件、非 torch 文件、payload 非 dict、旧 schema、缺 6 个必需字段中任一、未知 arm、bomb-head arm(B1)、spatial arm(B2)、错误 model_class、参数漂移、model_config 缺键、state_dict 键不匹配、state_dict 条目非 tensor。
- **动作规则（13 项）**：0–17 全候选 argmax、`(1,18)`/`(18,)`/numpy、并列取最小索引、全负 logits、坏形状 `(17,)/(2,18)/(1,1,18)`、NaN/±Inf 拒绝；`ProbePolicy.act` 与直接 forward argmax 一致且可重复，坏 plane 形状（play/hud/act_hist）拒绝，非 ProbeNet 模型拒绝。
- **seed 策略（38 项）**：dev 1000/1050/1099 通过；confirm 60000/60050/60199 拒绝；语料 1/7/14/15/17/18/20 拒绝；0/999/1100/70000 拒绝；非 int（`"1000"`/`1000.0`/`True`/`None`）拒绝；解析 5 种合法写法、11 种拒绝（空、非数字、反向区间、混合 confirm/语料、超上限 "cap" 等）、上限=100。
- **episode 记录（31 项）**：合法记录通过；14 个必需字段逐一缺失拒绝；privilege/three 标注/confirm seed/语料 seed/seed_role/obs spec/obs 形状/decision_frames/stage_clear/直方图/decisions/frames 上限/SHA/失败带 error/非法 stop_reason 全部 fail-closed。
- **summary（2 项）**：`verdict` 五项全 false、`not_claimed` 含 `not G410-2`、limits 含 SMOKE 说明、n=2 时 `ci_reportable=false`、JSON 可序列化；n≥20 才标记 `ci_reportable=true`。
- **CLI fail-closed（6 项）**：confirm seed / 语料 seed / 缺 checkpoint / 缺 engine / 缺 data-dir 均在**引擎启动前**返回 exit 2 且**不写** `--out`；parser 默认值就是 S1N 口径（stage 0 / normal / 19887 / consecutive / cpu）。
- 全部测试 **不启动引擎**、不读真实 checkpoint、不写 `tmp_path` 之外。

## 6. 限制（不声称）

1. **2 个 episode 不是开发成绩**：n=2 无置信区间、无统计意义；不能写成"B0 平均 1,108 帧"当成绩，也不能据此排名。
2. **checkpoint 是 BC 基线**：B0 是 D410-S1 上的 chosen-one-hot 行为克隆（20 epoch，val top1 0.6280 / macro 0.3080），**不是 RL 训练策略、不是最终学生模型**；离线已见明显过拟合（train top1 0.9769 vs val 0.6280）。
3. **无 G410-2 结论**：没跑 confirm split（60000–60199 被硬拒绝）、没有 20 episode 协议、没有四臂矩阵、没有性能门/通过率。
4. **只测了 B0**：入口按设计拒绝 B1（bomb head）与 B2/B3（spatial）；四臂闭环评测需要各自 loader 与预注册。
5. **score/lives/bombs 只是记录**：本次没有 score 目标、没有 PSN、没有人类对齐指标；`--stop-on-death` 让 `deaths` 恒为 1，不能当游戏结束统计。
6. **未做 act_hist 消融**：obs-v3 第三平面在闭环里的贡献未验证（WP04.0 曾测到离线 act_hist 分支近乎冻结）。
7. **评测口径限制**：`--stage 0` + 19887 帧 + `stop-on-death` 与 v400 S1N 相同，但与 scoreboard 的 game-over 行**不同口径**。

## 7. 下一建议（按优先级）

1. **正式 20 dev episodes / 单臂**：`--seeds 1000-1019`，同 stage/cap/stop-on-death，先跑 B0_s0（把现在 2 条扩到 20 条），让 frames 有 n=20 的 CI；正式 runner 应把 `describe_frames` 的 n≥20 分支接上 v400 的 percentile bootstrap（复用 `v400_student_eval.frames_statistics`，只读），并同时报首死帧数分布。
2. **补齐 WP04.2 四臂**（B1_s2 / B2 / B3 尚未完成），然后每臂用同一 20-seed 协议评测；B1/B3 需要先给本入口加 bomb-head loader，或在独立入口中实现（不得偷偷复用本 B0 路径）。
3. **先决定过拟合处置**：预注册写的是 last epoch，val top1 在 epoch 4–5 左右最高（约 0.68）。若要评估"最佳 val epoch"，必须另立预注册，不能事后挑选。
4. **act_hist 闭环消融**：zero/shuffle 两种扰动下重跑同一批 seed，作为 obs-v3 第三平面是否被使用的验收项。
5. **confirm split 保持不动**，直到 20-episode 协议与四臂评测口径冻结；本入口的硬拒绝应保留。
6. 将 `eval_smoke.json` 与 WP04.4/正式评测的 20-episode 证据分文件存放，禁止把 smoke 合并进正式成绩表。

## 8. 复现清单

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp043_pytest -q tests/test_v410_student_eval.py
python -m compileall -q tools/rl/v410_student_eval.py
```

```bash
# WSL2 Ubuntu
cd /mnt/d/WorkShop/Touhou_RL
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_student_eval.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp04_arms/b0_seed0_last.pt \
  --seeds 1000,1001 --stage 0 --difficulty normal --max-frames 19887 --stop-on-death \
  --out Document/Update/Alpha/v4.1.0/evidence/WP04/eval_smoke.json
```
