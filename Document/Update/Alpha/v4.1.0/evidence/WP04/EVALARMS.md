# WP04.5 — 四臂（B0/B1/B2/B3）seed0 × 20 局 dev episodes 真实引擎对比

| 项 | 值 |
|---|---|
| 任务 | v4.1.0 WP04.5：扩展 v410 evaluator 到 B1/B2/B3，并对四臂 seed0 checkpoint 各跑 20 个 dev 局 |
| 状态 | **完成**：4 臂 × 20 局 = 80 episodes，全部 `death_stop`，0 失败，四臂退出码均 0 |
| 日期 | 2026-11-13 |
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，sha256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` |
| 数据 | `/root/v400-data/game` |
| 合并证据 | `Document/Update/Alpha/v4.1.0/evidence/WP04/eval_arms_seed0_20.json`（418,702 B，sha256 `2c8b008db0a32cacc570674063c5617ff27d78ed40116f1903a4686df0b49a6d`） |
| 工具 | `tools/rl/v410_student_eval.py`，sha256 `2db2caaf9089c1698cef9b6a682991a7594ebac47ae93244c3d9932591eddddc` |
| 测试 | `tests/test_v410_student_eval.py`，sha256 `ded040f64b3927998c6bf4652a4bde5441dab2a9650073f8348cb518a0c3503f` |

> **标签（写死在 JSON 里）**：`not_a_gate = true`、`not_a_development_score = true`、
> `not_g410_2 = true`；`verdict` 的 `is_a_gate` / `is_a_development_score` /
> `is_a_pass_rate` / `is_a_four_arm_gate_matrix` / `is_a_g410_2_measurement` /
> `conclusion_allowed` 全部为 `false`。本文件是**描述性开发测量**，不是门、不是开发
> 分数、不是过关率、不是 G410-2，也不是四臂门矩阵。

---

## 1. 结果摘要（描述性）

- 四臂各 20 局，seed 1000–1019，`--stage 0 --difficulty normal --max-frames 19887
  --stop-on-death`，每 2 帧一次决策，obs-v3 输入。
- **80/80 局都以 `death_stop` 结束**（首死即停），**0/80 过关**，8 个 `reset_failed`/
  `error` 都没有出现；`deaths` 恒为 1.0、`lives_end` 恒为 2.0、`bombs_used` 恒为 0。
- 因此每臂可比较的量是**首死帧数**（frames），不是整局死亡分布。

| 臂 | 参数量 | bomb head | clear | Wilson 95%（描述性） | frames mean | median | min | max | idle% | unique（均值） |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 731,802 | 否 | 0/20 | [0.0000, 0.1611] | 1453.25 | 1374.0 | 815 | 2,386 | 45.81 | 10.9 |
| B1 | 731,931 | 是 | 0/20 | [0.0000, 0.1611] | 1765.90 | 1615.0 | 1,209 | 3,021 | 57.18 | 11.7 |
| B2 | 2,600,951 | 否 | 0/20 | [0.0000, 0.1611] | 1161.20 | 1240.0 | 746 | 1,446 | 83.59 | 7.5 |
| B3 | 2,601,464 | 是 | 0/20 | [0.0000, 0.1611] | 1398.55 | 1396.0 | 777 | 2,171 | 63.69 | 11.0 |

读法（只允许这样读）：

1. **B0 复现 WP04.4**：20 个 seed 的 `frames/steps/score/stop_reason/stage_clear` 与
   `eval_b0_20.json` **逐项相同**，`action_sequence_sha256` 与 `policy_input_sha256`
   也 **20/20 逐字节相同**（见 §4.5）。这证明扩展 loader 没有改变 B0 的观测/决策/动作
   路径，也证明这一次运行没有被宿主负载污染成不同轨迹。
2. B1/B2/B3 是**首次**真实闭环数据，没有历史基线可复现。
3. 四臂的 Wilson 区间完全重叠（都是 0/20），**过关率不可区分**；首死帧的均值/中位数
   只是 20 局的描述统计，没有做区间比较、没有多重比较校正，也没有因果归因。
4. 可观察到的量级差（B1 首死最晚、B2 最早且 idle 最高）只能作为**下一轮预注册的假设
   来源**，不能作为“B1 更好 / B2 更差”的结论。

---

## 2. 口径与环境

### 2.1 Checkpoints（全部 seed0、last epoch = 20）

| 臂 | checkpoint | sha256 | 参数量 | bomb head | 家族 |
|---|---|---|---|---|---|
| B0 | `.tmp_v410_wp04_arms/b0_seed0_last.pt` | `3bc0ec4461701bebb2ffc551d61dffff77f15cd486125baf9db6eaafc741b3a6` | 731,802 | 否 | probe（`ProbeNet`） |
| B1 | `.tmp_v410_wp04_arms/b1_seed0_last.pt` | `b3762f100fec9fbd2975c9b9dea88ed5396cbe8e1b78436b8c9794525412e44f` | 731,931 | 是（129） | probe（`BombAuxNet`） |
| B2 | `.tmp_v410_wp04_arms/b2_seed0_last.pt` | `2958645e8c340475d4d551ad16bd710525d99c63eb75f0eac00c95a2b1c6f0ec` | 2,600,951 | 否 | spatial（`SpatialBombNet`） |
| B3 | `.tmp_v410_wp04_arms/b3_seed0_last.pt` | `ab82b5b1d6dd03e2edee68fa824c69f627c8d5c1f31f286843952c2ff235eab5` | 2,601,464 | 是（513） | spatial（`SpatialBombNet`） |

四个 checkpoint 的 `seed` 都是 0、`last_epoch` 都是 20；没有做 epoch 选择、没有 early
stopping、没有按 val 挑 checkpoint。checkpoint 是 **BC 行为克隆基线**，不是 RL 策略。

### 2.2 S1N 口径

| 项 | 值 |
|---|---|
| stage | `--stage 0`（RESET 的 stage 是 0-based，S1N = Stage 1） |
| difficulty | `normal` |
| frame cap | 19,887（human_reference cap） |
| stop rule | `--stop-on-death`：首死即停；stage 计数前进记为 `stage_clear` 并停 |
| 决策频率 | 每 2 个游戏帧一次（`DECISION_FRAMES = 2`），每帧 `STATE` 裁判 |
| 观测 | obs-v3：play `(4,112,96)` + HUD `(1,56,160)` + act_hist `(8,18)`；只有像素与自身动作历史 |
| 动作 | 18 个 move logits 的 `argmax`（无采样、无温度、无 bomb 通道） |
| obs spec | `obs_spec_v3`，逐 episode 记录 version + digest |

引擎 `env` 角色只回像素与 `STATE`；`STATE` 只用于裁判死亡/关卡/分数，从不喂给模型。

### 2.3 运行环境与串行原因

- WSL2 Ubuntu 22.04.5，`/root/v400-venv/bin/python`（CPython 3.10.12），torch
  `2.14.1+cu130`，numpy `2.2.6`，20 个逻辑核。四个臂都在 `xvfb-run -a` 下运行。
- **串行四臂**：引擎会往 `--data-dir` 写 `score.dat` / `th08.cfg`（本轮修改时间可见），
  四臂共享同一个 `/root/v400-data/game`，并行会互相覆盖这些文件，因此按 B0 → B1 →
  B2 → B3 串行执行。
- **宿主负载警告**：运行期间同一 WSL 上有一个并发的 WP05 dagger 采集进程
  （`v410_dagger_collect.py`，load average 21–27），所以 `wall_s` 比 WP04.4 的
  ~20 s/局慢约 2–3 倍。`frames/actions` 是确定性的（B0 20/20 逐字节复现就是证据），
  **wall_s 不是性能测量，也不进入任何比较**。

---

## 3. 变更（严格白名单内）

### 3.1 `tools/rl/v410_student_eval.py`

保留 B0 行为（单臂输出仍是 `v410-student-eval-smoke/1` / `wp=WP04.3`，字段只增不改），
新增：

1. **四臂 loader**：按 checkpoint 的 `arm` 认领 → `bc.formal_arm_spec` 预注册
   （family / model_class / parameters / bomb_head / bomb_aux_weight）→ 按 `model_config`
   实例化：
   - B0：`bc.build_model("B0", model_config=...)` → `ProbeNet`；
   - B1：`bc.build_model("B1", model_config=...)` → `BombAuxNet`；
   - B2/B3：`bc.build_formal_model(arm)` → `SpatialBombNet`（canonical `model_v3.StudentV3`
     适配层，只读 import）。
2. **严格加载**：`strict=True`；参数量在 **checkpoint `parameters` / `arm_spec.parameters`
   / 内置预注册 pin / 构建后实测** 四处一致；`model_class`、`family`、`bomb_head`、
   `bomb_aux_weight` 与 `arm_spec` 一致；probe `model_config` 必须等于冻结的 WP04.0 空间，
   spatial `model_config` 必须与构建出的 adapter 的 `config()` 逐字段相等（`adapter` /
   `source` / `arm` / `bomb_head` / `base`），否则拒绝。
3. **策略**：`ProbePolicy` 接受 `ProbeNet` / `BombAuxNet` / `SpatialBombNet`；B1/B3 的
   bomb head **只读记录、从不动用**——本 ruler 的动作空间没有 bomb 通道，四臂统一
   `(cand, bomb=0)`，bomb logit 逐决策记录（非有限值直接拒绝）。
4. **episode 记录**：新增 `arm_parameters_pinned` / `arm_bomb_head` /
   `arm_bomb_aux_weight` / `action_rule`，`actions` 增加 `labels`、`idle_definition`、
   `bomb{head_present, actuated, decisions_read, mean/sd/min/max, share_positive}`；
   record 校验会核对 arm/family/model_class/参数量/bomb 标志是否与预注册一致。
5. **per-arm summary**：`arm_summary` = episodes / clears / Wilson 95%（描述性）/
   frames mean+median+min+max+sd / deaths / stop_reasons / 动作分布（18 类 share、
   idle share、unique 均值）/ bomb logit 聚合；并显式写入
   `not_a_gate` / `not_a_development_score` / `not_g410_2`。
6. **合并模式**：`--merge-summaries B0.json B1.json B2.json B3.json --out ...` 生成
   `v410-student-eval-arms/1`。合并会重新校验每条 episode，并拒绝重复臂、不同
   engine/obs_spec/suite/seed 列表、非 dev seed、臂与记录不一致、外来 schema；合并模式
   不启动任何引擎。

### 3.2 `tests/test_v410_student_eval.py`

- 追加 B1/B2/B3 的真实 state dict 往返、参数量/arm 认领/`model_config`/obs spec/CLI/
  merge 用例；新增 B1/B2/B3 的拒绝路径（参数漂移、`model_config` 臂错配、`base`
  漂移、`arm_spec` 不一致、family/model_class 不一致、缺 `parameters` 等）。
- **旧 111 tests 全部仍通过**；其中 2 个用例因本任务**取代了旧契约**而改写了拒绝对象
  （旧的“B1/B2 属于范围外” → 现在的“臂认领与权重/配置不匹配仍必须拒绝”），断言仍是
  `EvalError` 拒绝，没有放松任何 fail-closed 行为。
- 统计见 §4.4：**152 passed**。

### 3.3 新增证据文件

- `Document/Update/Alpha/v4.1.0/evidence/WP04/eval_arms_seed0_20.json`（本任务的合并证据）
- `Document/Update/Alpha/v4.1.0/evidence/WP04/EVALARMS.md`（本文件）

**未修改**：v400/v380/v390、`tools/rl/v400_*`、`tests/test_v400_*`、`third_party/`、
其他 v410 文件、`tools/rl/v410*.py` 其他文件、`tests/test_v410_*` 其他文件、
state/journal/budget/assets、WP03/WP04 既有证据（`eval_b0_20.json`、`EVAL20.md` 等只读）。
**未训练、未采数据、未 commit/push、未删除/覆盖旧文件**。

---

## 4. 实际命令与退出码

### 4.1 四臂真实运行（WSL2，串行）

运行时驱动脚本在运行目录（不入库）`.tmp_v410_wp04_eval_arms/run_arms.sh`，核心命令：

```bash
cd /mnt/d/WorkShop/Touhou_RL
xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
  tools/rl/v410_student_eval.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --checkpoint .tmp_v410_wp04_arms/<arm>_seed0_last.pt \
  --seeds 1000-1019 --stage 0 --difficulty normal --max-frames 19887 \
  --stop-on-death \
  --out .tmp_v410_wp04_eval_arms/<ARM>_seed0_full.json
```

| 臂 | 退出码 | 墙钟（含引擎启动/20 局） | 日志 |
|---|---|---|---|
| B0 | **0** | 1,187 s（19.8 min） | `.tmp_v410_wp04_eval_arms/B0_full.log` |
| B1 | **0** | 1,339 s（22.3 min） | `.tmp_v410_wp04_eval_arms/B1_full.log` |
| B2 | **0** | 1,054 s（17.6 min） | `.tmp_v410_wp04_eval_arms/B2_full.log` |
| B3 | **0** | 1,183 s（19.7 min） | `.tmp_v410_wp04_eval_arms/B3_full.log` |
| 合计 | **0** | **4,763 s ≈ 79.4 min** | 80/80 episodes |

逐局 `wall_s` 已写入 JSON；因并发 WP05 采集，只作运行记录，不作性能/效率比较。

### 4.2 合并（生成证据 JSON）

```bash
/root/v400-venv/bin/python tools/rl/v410_student_eval.py \
  --merge-summaries \
  .tmp_v410_wp04_eval_arms/B0_seed0_full.json \
  .tmp_v410_wp04_eval_arms/B1_seed0_full.json \
  .tmp_v410_wp04_eval_arms/B2_seed0_full.json \
  .tmp_v410_wp04_eval_arms/B3_seed0_full.json \
  --out Document/Update/Alpha/v4.1.0/evidence/WP04/eval_arms_seed0_20.json
```

**退出码 0**；合并输出 `80 episodes / 4 arms / 0 failed / clears 0`，写入
418,702 B 的 `v410-student-eval-arms/1` 文件。

### 4.3 证据自检

```bash
cd /mnt/d/WorkShop/Touhou_RL
/root/v400-venv/bin/python .tmp_v410_wp04_eval_arms/verify_evidence.py
```

**退出码 0**，输出：

```text
OK: 80 episodes, 4 arms, 20 dev seeds each, all records valid
  B0: clears=0/20 frames_mean=1453.25 median=1374.0 idle=45.81% bomb_head=False
  B1: clears=0/20 frames_mean=1765.90 median=1615.0 idle=57.18% bomb_head=True
  B2: clears=0/20 frames_mean=1161.20 median=1240.0 idle=83.59% bomb_head=False
  B3: clears=0/20 frames_mean=1398.55 median=1396.0 idle=63.69% bomb_head=True
```

自检内容：schema/arm_order、80 条记录全部通过工具自身的 `validate_episode_record`、
每臂 seed 恰好 1000–1019、checkpoint sha256 与 `.pt` 文件一致、engine sha256 与引擎文件
一致、`verdict` 的 not-a-gate 标志、`exit_code == 0`。

### 4.4 测试与编译校验

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp045_pytest -q tests/test_v410_student_eval.py
# 152 passed in 2.93s   PYTEST_EXIT=0

python -m compileall -q tools/rl/v410_student_eval.py
# COMPILE_EXIT=0
```

额外回归（该模块被在跑的 WP05 采集器只读复用，验证未被本次扩展破坏）：

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp045_pytest_dagger -q tests/test_v410_dagger_collect.py
# 75 passed in 0.25s   DAGGER_EXIT=0
```

### 4.5 B0 复现 WP04.4

```powershell
python .tmp_v410_wp04_eval_arms/compare_b0.py
# compared 20 seeds, 20 identical, 0 different
# COMPARE_EXIT=0
```

比较字段：`frames / steps / score / stop_reason / stage_clear /
action_sequence_sha256 / policy_input_sha256`。20/20 全部相同。

---

## 5. Loader / record 的 fail-closed 契约

| 检查 | 行为 |
|---|---|
| checkpoint schema | 只接受 `v410-wp04-bc-arm-checkpoint/1`、`v410-wp04-bc-checkpoint/1` |
| arm 认领 | 必须是 `bc.FORMAL_ARMS` 的 B0/B1/B2/B3，未知 arm 拒绝 |
| family / model_class | 必须等于该臂预注册（B0 `ProbeNet`、B1 `BombAuxNet`、B2/B3 `SpatialBombNet`） |
| 参数量（三方一致） | checkpoint `parameters` = `arm_spec.parameters` = 构建后实测 = 内置 pin（731,802 / 731,931 / 2,600,951 / 2,601,464） |
| bomb 标志 | `bomb_head` / `bomb_aux_weight` 必须等于预注册（B0/B2 = False/0.0，B1/B3 = True/0.5） |
| `model_config`（probe） | 必须逐字段等于冻结的 WP04.0 probe 空间（24/16/64/128/128/8） |
| `model_config`（spatial） | 必须逐字段等于构建出的 `SpatialBombNet.config()`（adapter/source/arm/bomb_head/base，base.arch=`v3`） |
| state dict | 每个 entry 必须是 tensor，`load_state_dict(strict=True)`；缺键/多键/形状不符都拒绝 |
| obs spec | episode 记录并校验 `obs_spec_v3` version + digest + `play/hud/act_hist` 形状；不一致拒绝 |
| seed policy | 只允许 dev 1000–1099；confirm 60000–60199 与语料 seed 1–20 硬拒绝（无覆盖开关） |
| episode 记录 | 必需字段类型、`stage_clear` 与 stage 计数一致、histogram 和 = `steps`、失败局必须带 `error`、动作哈希/引擎哈希为 sha256 |
| 合并 | 重新校验每条记录；拒绝重复臂、跨 engine/obs_spec/suite/seed 列表、非 dev seed、臂与记录不一致 |

B1/B3 的 bomb head 在部署中**只读**：`actions.bomb.actuated = false`，`bombs_used = 0`；
这一点在 80 条记录里逐条成立。

---

## 6. 每臂 20 局明细

### 6.1 B0（`ProbeNet`，731,802）

| seed | frames | steps | stop_reason | clear | score | idle% | unique | wall_s |
|---|---|---|---|---|---|---|---|---|
| 1000 | 1,398 | 699 | `death_stop` | false | 13,102 | 42.20 | 12 | 47.1 |
| 1001 | 818 | 409 | `death_stop` | false | 1,512 | 50.86 | 8 | 35.0 |
| 1002 | 1,251 | 626 | `death_stop` | false | 7,438 | 49.68 | 11 | 45.7 |
| 1003 | 815 | 408 | `death_stop` | false | 1,507 | 61.52 | 7 | 31.5 |
| 1004 | 2,386 | 1,193 | `death_stop` | false | 30,207 | 44.09 | 14 | 79.2 |
| 1005 | 1,379 | 690 | `death_stop` | false | 5,165 | 38.55 | 13 | 50.0 |
| 1006 | 1,369 | 685 | `death_stop` | false | 5,030 | 47.45 | 9 | 53.9 |
| 1007 | 1,664 | 832 | `death_stop` | false | 8,774 | 50.00 | 14 | 64.8 |
| 1008 | 1,476 | 738 | `death_stop` | false | 8,520 | 39.70 | 11 | 57.0 |
| 1009 | 1,457 | 729 | `death_stop` | false | 5,629 | 38.41 | 10 | 57.3 |
| 1010 | 1,214 | 607 | `death_stop` | false | 5,414 | 45.47 | 12 | 48.1 |
| 1011 | 1,224 | 612 | `death_stop` | false | 7,589 | 34.80 | 9 | 40.1 |
| 1012 | 1,214 | 607 | `death_stop` | false | 7,456 | 49.75 | 10 | 46.2 |
| 1013 | 2,239 | 1,120 | `death_stop` | false | 33,295 | 49.11 | 14 | 82.2 |
| 1014 | 1,256 | 628 | `death_stop` | false | 9,512 | 45.38 | 11 | 48.3 |
| 1015 | 1,361 | 681 | `death_stop` | false | 8,167 | 48.31 | 11 | 49.8 |
| 1016 | 1,486 | 743 | `death_stop` | false | 8,924 | 50.47 | 8 | 50.7 |
| 1017 | 1,894 | 947 | `death_stop` | false | 24,666 | 49.21 | 13 | 65.2 |
| 1018 | 1,231 | 616 | `death_stop` | false | 7,475 | 43.67 | 9 | 43.8 |
| 1019 | 1,933 | 967 | `death_stop` | false | 19,457 | 43.85 | 12 | 68.7 |

### 6.2 B1（`BombAuxNet`，731,931，bomb head 只读）

| seed | frames | steps | stop_reason | clear | score | idle% | unique | wall_s |
|---|---|---|---|---|---|---|---|---|
| 1000 | 1,533 | 767 | `death_stop` | false | 10,420 | 58.67 | 11 | 51.5 |
| 1001 | 1,443 | 722 | `death_stop` | false | 4,309 | 58.86 | 11 | 47.2 |
| 1002 | 1,955 | 978 | `death_stop` | false | 19,877 | 52.35 | 10 | 63.8 |
| 1003 | 1,963 | 982 | `death_stop` | false | 17,232 | 56.21 | 14 | 64.5 |
| 1004 | 1,731 | 866 | `death_stop` | false | 11,282 | 56.81 | 12 | 57.0 |
| 1005 | 1,433 | 717 | `death_stop` | false | 13,512 | 60.81 | 10 | 49.1 |
| 1006 | 1,209 | 605 | `death_stop` | false | 6,872 | 58.68 | 10 | 43.8 |
| 1007 | 1,586 | 793 | `death_stop` | false | 8,968 | 61.54 | 10 | 59.4 |
| 1008 | 1,533 | 767 | `death_stop` | false | 11,537 | 52.15 | 11 | 57.7 |
| 1009 | 2,159 | 1,080 | `death_stop` | false | 29,175 | 59.17 | 13 | 77.8 |
| 1010 | 1,590 | 795 | `death_stop` | false | 10,123 | 56.35 | 11 | 58.0 |
| 1011 | 1,540 | 770 | `death_stop` | false | 10,877 | 55.97 | 10 | 57.1 |
| 1012 | 1,765 | 883 | `death_stop` | false | 13,970 | 57.98 | 11 | 45.3 |
| 1013 | 1,405 | 703 | `death_stop` | false | 9,758 | 56.61 | 13 | 53.0 |
| 1014 | 1,901 | 951 | `death_stop` | false | 17,247 | 62.15 | 11 | 72.0 |
| 1015 | 1,640 | 820 | `death_stop` | false | 16,175 | 60.12 | 11 | 57.8 |
| 1016 | 2,292 | 1,146 | `death_stop` | false | 25,581 | 54.28 | 14 | 74.8 |
| 1017 | 3,021 | 1,511 | `death_stop` | false | 54,634 | 53.21 | 15 | 94.3 |
| 1018 | 2,197 | 1,099 | `death_stop` | false | 21,812 | 58.87 | 13 | 73.1 |
| 1019 | 1,422 | 711 | `death_stop` | false | 4,217 | 57.24 | 13 | 48.2 |

### 6.3 B2（`SpatialBombNet`，2,600,951）

| seed | frames | steps | stop_reason | clear | score | idle% | unique | wall_s |
|---|---|---|---|---|---|---|---|---|
| 1000 | 1,252 | 626 | `death_stop` | false | 9,875 | 84.03 | 7 | 47.8 |
| 1001 | 1,265 | 633 | `death_stop` | false | 27,078 | 83.25 | 8 | 47.3 |
| 1002 | 1,101 | 551 | `death_stop` | false | 6,207 | 81.49 | 8 | 46.0 |
| 1003 | 1,129 | 565 | `death_stop` | false | 8,996 | 85.13 | 6 | 45.2 |
| 1004 | 1,377 | 689 | `death_stop` | false | 10,278 | 82.44 | 9 | 53.2 |
| 1005 | 1,438 | 719 | `death_stop` | false | 8,673 | 76.77 | 11 | 54.8 |
| 1006 | 1,285 | 643 | `death_stop` | false | 4,602 | 82.43 | 9 | 52.9 |
| 1007 | 761 | 381 | `death_stop` | false | 1,917 | 87.93 | 7 | 36.6 |
| 1008 | 746 | 373 | `death_stop` | false | 1,054 | 90.35 | 6 | 35.4 |
| 1009 | 1,397 | 699 | `death_stop` | false | 8,183 | 82.98 | 8 | 61.3 |
| 1010 | 1,090 | 545 | `death_stop` | false | 5,843 | 84.40 | 7 | 48.4 |
| 1011 | 992 | 496 | `death_stop` | false | 2,038 | 83.67 | 9 | 44.2 |
| 1012 | 746 | 373 | `death_stop` | false | 1,054 | 90.62 | 6 | 35.2 |
| 1013 | 758 | 379 | `death_stop` | false | 1,887 | 88.92 | 4 | 34.9 |
| 1014 | 1,347 | 674 | `death_stop` | false | 10,627 | 83.53 | 7 | 33.3 |
| 1015 | 1,354 | 677 | `death_stop` | false | 14,569 | 87.74 | 5 | 58.9 |
| 1016 | 1,123 | 562 | `death_stop` | false | 8,322 | 80.78 | 6 | 49.2 |
| 1017 | 1,389 | 695 | `death_stop` | false | 8,071 | 86.33 | 10 | 57.9 |
| 1018 | 1,228 | 614 | `death_stop` | false | 6,066 | 80.94 | 9 | 50.9 |
| 1019 | 1,446 | 723 | `death_stop` | false | 9,716 | 78.56 | 9 | 55.0 |

### 6.4 B3（`SpatialBombNet` + bomb head，2,601,464，bomb head 只读）

| seed | frames | steps | stop_reason | clear | score | idle% | unique | wall_s |
|---|---|---|---|---|---|---|---|---|
| 1000 | 1,575 | 788 | `death_stop` | false | 17,816 | 60.28 | 14 | 61.3 |
| 1001 | 1,452 | 726 | `death_stop` | false | 12,488 | 60.19 | 13 | 58.9 |
| 1002 | 1,427 | 714 | `death_stop` | false | 10,328 | 63.17 | 11 | 56.7 |
| 1003 | 1,994 | 997 | `death_stop` | false | 27,204 | 61.79 | 15 | 74.0 |
| 1004 | 1,320 | 660 | `death_stop` | false | 8,671 | 63.79 | 10 | 50.0 |
| 1005 | 1,553 | 777 | `death_stop` | false | 9,639 | 59.97 | 12 | 60.0 |
| 1006 | 1,369 | 685 | `death_stop` | false | 15,301 | 68.32 | 10 | 58.5 |
| 1007 | 1,403 | 702 | `death_stop` | false | 13,665 | 63.82 | 12 | 56.2 |
| 1008 | 2,123 | 1,062 | `death_stop` | false | 37,325 | 61.77 | 16 | 80.4 |
| 1009 | 1,104 | 552 | `death_stop` | false | 4,056 | 68.30 | 9 | 46.3 |
| 1010 | 777 | 389 | `death_stop` | false | 1,036 | 72.24 | 7 | 36.4 |
| 1011 | 1,389 | 695 | `death_stop` | false | 12,808 | 68.20 | 9 | 60.3 |
| 1012 | 1,474 | 737 | `death_stop` | false | 7,702 | 55.63 | 12 | 67.2 |
| 1013 | 1,104 | 552 | `death_stop` | false | 3,884 | 71.01 | 7 | 48.8 |
| 1014 | 790 | 395 | `death_stop` | false | 1,285 | 58.99 | 9 | 36.5 |
| 1015 | 1,614 | 807 | `death_stop` | false | 18,553 | 60.35 | 13 | 50.9 |
| 1016 | 1,270 | 635 | `death_stop` | false | 34,185 | 74.33 | 10 | 57.0 |
| 1017 | 969 | 485 | `death_stop` | false | 2,144 | 64.74 | 11 | 44.5 |
| 1018 | 2,171 | 1,086 | `death_stop` | false | 25,824 | 57.27 | 13 | 86.8 |
| 1019 | 1,093 | 547 | `death_stop` | false | 4,765 | 75.14 | 7 | 45.6 |

---

## 7. 汇总与动作分布

### 7.1 动作分布（pooled over 20 局；candidate 0 = idle）

| 臂 | decisions | idle% | E% | W% | SW% | NE% | 其他 move+focus | unique（均值） |
|---|---|---|---|---|---|---|---|---|
| B0 | 14,537 | 45.81 | 14.69 | 13.74 | 13.43 | 5.12 | 7.21 | 10.9 |
| B1 | 17,666 | 57.18 | 14.32 | 11.58 | 8.63 | 3.05 | 5.24 | 11.7 |
| B2 | 11,617 | 83.59 | 5.60 | 6.60 | 0.94 | 1.14 | 2.13 | 7.5 |
| B3 | 13,991 | 63.69 | 11.46 | 11.56 | 3.69 | 3.33 | 6.27 | 11.0 |

完整 18 类 share、直方图与逐局 `action0_share` 都在 JSON 的
`arms.<arm>.actions` 与 `episodes.<arm>[*].actions` 中。动作分布只是记录，**不能**据此
断言死因（例如“B2 死得早是因为 idle 高”需要预注册的消融/诊断才能说）。

### 7.2 B1/B3 的 bomb head（只读，从不动用）

| 臂 | 读取决策数 | mean logit | min | max | share_positive |
|---|---|---|---|---|---|
| B1 | 17,666 | -47.245 | -87.675 | -18.332 | **0.000** |
| B3 | 13,991 | -28.391 | -115.068 | -10.311 | **0.000** |

D410-S1 的 bomb 标签在全部 66,143 行上恒为 0，aux BCE 只能学负类先验；上表的
logit 全负、正例率 0 与之吻合。**B1−B0 / B3−B2 的差异不能解释为“辅助标签收益”**。

### 7.3 分数/死亡口径（仅记录）

- 各臂 mean score：B0 10,942.0 / B1 15,878.9 / B2 7,752.8 / B3 13,434.0。
- `deaths = 1.0`、`lives_end = 2.0`、`bombs_used = 0` 在 80 局全部成立；这是
  `--stop-on-death` 口径的直接结果，**没有整局死亡分布、没有 game-over 口径**。
- 所有局都远早于 19,887 帧上限（max 3,021 帧），没有 `frame_cap` 局。

---

## 8. 限制（明确不声称）

1. **只有 seed0、只有 last epoch**：每个臂只评了一个 BC 训练种子（seed 0）的 last
   epoch（epoch 20）；没有 BC 训练种子方差，没有 epoch 选择，没有按 val 挑 checkpoint。
2. **20 局是描述性的，不是门**：每臂 20 个 dev seed 只给出描述统计；Wilson 区间
   [0, 0.1611] 是在“失败算非过关”的口径下对 0/20 的描述性区间，**不是** G410-2 的
   过关率区间，也不能写成“过关率上限 16.1%”这类门结论。
3. **不是 G410-2、不是开发分数、不是过关率、不是四臂门矩阵**：没有预注册的判定规则、
   没有 confirm split（60000–60199 未读取）、没有阈值、没有 passing。JSON 的
   `verdict` 六项与 `not_a_gate`/`not_a_development_score`/`not_g410_2` 都是显式 false/true。
4. **bomb aux 全零**：D410-S1 的 bomb 标签恒为 0，B1/B3 的 auxiliary 只学到负类先验；
   bomb head 在本 ruler 中**只读记录、从不 actuate**（`bombs_used = 0`），四臂动作规则
   完全一致。因此 B1−B0 / B3−B2 的差异不能归因于辅助标签收益。
5. **容量与配方混杂**：B2/B3 是 2.6 M spatial 适配层，B0/B1 是 0.73 M probe 空间；
   B0-vs-B2 同时改变模型空间、感受野与训练配方，不是纯容量对比。空间臂还保留
   **25,557 个不参与 BC 训练的 perception 参数**（WP04.2 梯度覆盖结论），参数量会高估
   有效容量。
6. **`--stop-on-death` 口径**：每局首死即停，`deaths` 恒 1、`lives_end` 恒 2，因此只有
   “首死帧数”，没有整局死亡分布、没有 score/PSN/人类对齐指标。
7. **动作分布仅记录**：idle/E/W 等比例不能单独作为死因解释；B2 的高 idle（83.59%）
   与更早首死只是同时出现，不能据此断言因果。
8. **墙钟受并发负载污染**：运行期间有并发的 WP05 dagger 采集（load 21–27），
   `wall_s` 约比 WP04.4 慢 2–3 倍；本报告不拿它做性能或效率比较。
9. **可复现性覆盖**：只有 B0 有 WP04.4 的 20 局逐 seed 对标（20/20 逐字节相同）；
   B1/B2/B3 只跑了一次，没有重复运行证据。四臂的 `policy_input_sha256` /
   `action_sequence_sha256` 已逐局写入 JSON，可用于后续复跑比对。
10. **未训练、未采数据、未触 confirm**：本任务没有训练、没有生成新数据、没有读取
    confirm seeds、没有修改 WP03/WP04 既有证据，也没有 commit/push。

---

## 9. 下一建议（按优先级）

1. **预注册 seed1/seed2 与“最佳 val”规则，再跑**：当前只有 seed0/last epoch。若要评
   BC 训练种子方差或“最佳 val epoch”，必须**先**写清选择规则（例如“每臂按 val top1
   选 epoch，选择在跑 dev 之前固定”）和跑的臂×seed 网格，再执行；不得在本结果上事后
   挑 checkpoint、挑 seed 或挑 epoch。WP04.4 已记录 val top1 在 epoch 4–5 附近最高
   （~0.68），这只能作为**预注册的候选规则**，不能直接拿来替换 last epoch。
2. **四臂 × seed1/seed2 的 20 局**：同一 dev seeds 1000–1019、同一 S1N 口径，串行执行；
   每臂报告 clear 计数、首死帧 mean/median、动作分布与 bomb logit 统计，仍标注
   descriptive only。预算约 4 臂 × 2 seed × 20 局 ≈ 160 局；按本轮负载约 2.5–3 h。
3. **bomb 目标的处置先于扩评**：要么在预注册里明确“bomb aux 在 S1 无信号、B1/B3 仅作
   架构对照”，要么先构造带正例的 bomb 标签（或换成有信息量的辅助目标）再谈 aux 收益。
   若要 actuate bomb，必须另立预注册的 bomb 决策规则（阈值/冷却），不能在本 ruler 里
   临时加。
4. **可复现性补强**：对 B1/B2/B3 至少重跑 seeds 1000/1001，比对
   `policy_input_sha256` / `action_sequence_sha256`；当前只有 B0 有 20/20 的重复证据。
5. **诊断而非调参**：在不动正式口径的前提下，另立预注册做 act_hist zero/shuffle 消融与
   首死 forensics（20 局回放），验证 obs-v3 第三平面是否被使用、首死是否集中在固定
   阶段；结论只用于下一轮设计。
6. **confirm split 继续封存**：直到四臂 loader、20 局协议、checkpoint 选择规则与判定
   口径全部冻结并通过评审；`check_seed_policy` 的硬拒绝必须保留。

---

## 10. 复现清单

```bash
# WSL2 Ubuntu；仓库在 /mnt/d/WorkShop/Touhou_RL
cd /mnt/d/WorkShop/Touhou_RL
sha256sum .tmp_v410_wp04_arms/{b0,b1,b2,b3}_seed0_last.pt \
          /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern
# 与 §2.1 / §2 的 sha256 对照

# 四臂各 20 局（串行；输出到运行目录，不入库）
for arm in b0 b1 b2 b3; do
  xvfb-run -a -s "-screen 0 1024x768x24" /root/v400-venv/bin/python \
    tools/rl/v410_student_eval.py \
    --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
    --data-dir /root/v400-data/game \
    --checkpoint .tmp_v410_wp04_arms/${arm}_seed0_last.pt \
    --seeds 1000-1019 --stage 0 --difficulty normal --max-frames 19887 \
    --stop-on-death --out .tmp_v410_wp04_eval_arms/${arm}_seed0_full.json
done

# 合并（本任务的证据文件）
/root/v400-venv/bin/python tools/rl/v410_student_eval.py \
  --merge-summaries .tmp_v410_wp04_eval_arms/B0_seed0_full.json \
                    .tmp_v410_wp04_eval_arms/B1_seed0_full.json \
                    .tmp_v410_wp04_eval_arms/B2_seed0_full.json \
                    .tmp_v410_wp04_eval_arms/B3_seed0_full.json \
  --out Document/Update/Alpha/v4.1.0/evidence/WP04/eval_arms_seed0_20.json
# EXIT_CODE=0

/root/v400-venv/bin/python .tmp_v410_wp04_eval_arms/verify_evidence.py
# OK: 80 episodes ... EXIT_CODE=0

$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp045_pytest -q tests/test_v410_student_eval.py
# 152 passed  EXIT_CODE=0
python -m compileall -q tools/rl/v410_student_eval.py
# EXIT_CODE=0
```
