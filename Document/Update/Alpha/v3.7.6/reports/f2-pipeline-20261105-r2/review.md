# v3.7.6 真实 F2 链路：定向补采 → 冻结计划 → 数据集 → 一次更新

> **性质：执行记录。** 本文件记录 2026-11-05 第二轮（Stage 2 定向补采）从采集到一次真实更新的完整命令、退出码与身份；它**不改变**任何门状态，也不声称能力。真实长训练仍未授权，原 G4 仍未执行。
> 状态与计数仍以 [任务清单](<../../任务清单.md>) 为准。

- 配置：[f2-bc-v2.json](<../../configs/f2-bc-v2.json>)（Stage 2 预注册，取代 Stage 1 的 `f2-bc-v1.json` 作为本轮来源）
- 冻结计划：`v376-f2-bc-plan-20261105-r2.json`（仓库根；`/1` schema，`frozen=true`）
- 数据集：`third_party/th08/build/rl/f2/v376-f2-bc-dataset-20261105-r3`
- 缓存：`third_party/th08/build/rl/f2/v376-f2-bc-cache-20261105-r3`
- 一次更新：`third_party/th08/build/rl/f2/v376-f2-bc-train-20261105-r3`
- 身份清单：[sha256.json](<sha256.json>) · 审计：[dataset-audit.json](<dataset-audit.json>) · gate：[check-gate.txt](<check-gate.txt>) · 训练：[report](<train-report.json>)/[log](<train-log.jsonl>)

---

## 0. 结论

| 环节 | 结果 | 证据 |
|---|---|---|
| 定向补采（3 来源 × Stage 2） | **3000/3000 × 3，discarded=0，13 检查全过** | `frame-proof.json`、[深审](<../../reports/f2-capture-20261105-r2-deep-audit.json>) |
| 冻结计划（3 事件） | 已冻结，plan SHA 见 [sha256.json](<sha256.json>) | 仓库根 `v376-f2-bc-plan-20261105-r2.json` |
| build（含 upstream 深审） | **exit 0**，3 事件 × 48 行 | 数据集 `manifest.json` |
| audit | **passed=true，exit 0** | [dataset-audit.json](<dataset-audit.json>) |
| check-gate | **exit 0**：`F2 behavior-cloning content audited; research training eligible` | [check-gate.txt](<check-gate.txt>) |
| feature cache（GPU, batch16） | 96 行 / 2 事件，`test_tensor_loaded=false`、`test_cache_created=false` | `index.json` + `report.json` |
| **一次真实更新** | **updates=1**，validation_loss 3.2339，`backbone_unchanged=true`，`test_used=false` | [train-report.json](<train-report.json>)、[train-log.jsonl](<train-log.jsonl>) |

**没有解决的问题（必须同时读）：**

1. 复核由 AI 执行并**如实署名**（`reviewer.id = v376-wp-b-agent-visual-20261105`，`kind=ai_agent_visual`），用户已授权该签署链；**独立复核（WP-F，非作者执行）尚未进行**。
2. 数据集只有 **3 事件 × 48 行 = 144 行**，其中 **8 行**带 F2 标注（train 3、validation 2、test 3），**实际产生动作监督的只有 6 行**（test 有 2 行标注行当时没有按住射击，按合同落到 unknown）；其余 136 行是显式 `context_only`（零 mask）。这 6 行只有 4 个不同动作、且仅 validation 内有 1 处动作变化。这是**集成样本**，不是统计意义上的数据集；任何「学到了什么」的说法都无依据。
3. 一次更新只证明「真实数据 → 缓存 → 优化器 → 落后盘」链路可用，**不证明学习、泛化、闭环、玩法改善**，也不是原 G4。
4. `check-gate` 通过只表示内容审计通过，**不是 G3/real 通过**。
5. **放行审查**：[WP-F 自查](<../wp-f-self-audit-20261105/review.md>)（用户指定由实现者自查，非独立）结论为**「允许有限诊断，不允许正式训练」**，并列出四处必须先修的缺口（合同瞄准判据过弱、监督行按可测量性挑选、逐事件常数基线已达 83.3%、评估协议未冻结）。本报告的「一次更新成功」只在「有限诊断」意义上成立。

---

## 1. 为什么换到 Stage 2

[WP-B 第一轮筛选](<../wpb-pre-screen-20261105/review.md>) 对 Stage 1 的 133 个候选做了 133/133 内存几何排名 + 单帧联系表初筛 + 3 个窗口连续帧细看，结论是：三份来源的 Stage 1 片段是环形/星形网/扇形/竖向激光，**没有可确认的瞄准弹流**。按计划 §3 的停止条件，报告覆盖不足并申请定向补采；用户批准。

补采沿用**同一三份 replay 身份**（split 单位不变，都是 `full_replay_sha256`），只换 decoder stage：`--stage 1` → TH08 Stage 2。这样 `f2-bc-v1.json` 的来源与分组规则不需要重写。

---

## 2. 补采命令与结果

```powershell
python -B -m touhou_rl.pipeline collect --modality privileged --source replay `
  --replay <rpy> --out third_party/th08/build/rl/exact/v376-f2-<role>-st2-20261105-r1 `
  --stage 1 --max-seconds 900 --frame-proof exact `
  --exact-config Document/Update/Alpha/v3.7.6/configs/f2-exact-capture-v1.json
```

| 来源 | 帧 | discarded | stage | logic frames | 时钟滞后 | 深审 |
|---|---|---|---|---|---|---|
| train | 3000 | 0 | 1（Stage 2） | 4–6014 | 0 | 19.2 s，passed |
| validation | 3000 | 0 | 1 | 5–6027 | 0 | 18.8 s，passed |
| test | 3000 | 0 | 1 | 4–6010 | 0 | 18.7 s，passed |

- 原始日志：[train](<../../evidence/f2-capture-20261105-r2/train-st2.log>) · [validation](<../../evidence/f2-capture-20261105-r2/validation-st2.log>) · [test](<../../evidence/f2-capture-20261105-r2/test-st2.log>)
- 本次 `stage_clock_lag = 0`（Stage 2 窗口内没有中场对白），所以 WP-A 修复的 `logicFrame` 对齐在这批数据上等价于 `stageFrames`；修复本身仍逐行生效（消费输入复验）。
- 复核包（新）：`evidence/f2-review-20261105-r4/{train,validation,test}-st2/`（44/41/47 个候选，排除记录在各自 `candidates.json`）。

### 2.1 瞄准证据（预筛，仅排序）

同样用 [预筛工具](<../../../../../../tools/rl/v376_wpb_pre_screen.py>) 跑新窗口：出现「新弹 100% 指向自机」的窗口（train i900 `fresh=63/aimed=63`、validation i1200 `42/42`、test i1020 `69/69`），与 Stage 1 的「fresh 多、aimed 少」形成对照。该读数仍**只是排序**：预筛在密集环上会误报，接受与否必须看原图。

### 2.2 目视确认

三个最优窗口各取 6 个相邻采样帧、1:1 裁剪游戏区拼版（[train i900](<../../reports/wpb-pre-screen-20261105/montage/train-st2-i900.png>) 等）。可见敌机/中 boss 反复发射**成束白色箭头弹，朝自机方向成扇展开并推进**——这是 Stage 1 里没有的图案族。据此选定三个窗口各 1 个事件。

---

## 3. 复核（AI 署名，逐像素测量）

工具：[v376_wpb_author_review.py](<../../../../../../tools/rl/v376_wpb_author_review.py>)。

**每条标注行的证据链：**

1. **发射帧**：从当前帧回溯该槽位连续的样本，取「第一次能被单独测量」的帧（脉冲刚出膛时与发射体、同团弹重叠，通常不可测）。
2. **瞄准帧**：在发射帧之前 6 个采样帧内，取「弹道直线离自机位置最近且 ≤2 px」的那一帧，作为 `aim_player_frame`；此时刻就是「这发弹是对着自机打的」的可核对证据。
3. **像素测量**：以内存预测位置为中心取 15×15 窗口，用「亮度减窗口自身中位数」的平方加权质心求弹体中心；对比度不足/亮区铺满（无局部弹体）时判为不可测，绝不回退成内存坐标。实测偏差：train 2.98 px、validation 3.49 px、test 1.92 px（**容差由实测偏差反推**：3.98 / 4.49 / 2.92）。
4. **teacher 侧**：`slot:N` + 同 hold 内存坐标，`_verify_teacher` 逐点精确复算；`reviewer.id != producer_id`。
5. **两条发射**：同一 `origin-cluster`（两发首可见位置相距 ≤30 px，按 10 px 栅格命名）但在**不同帧**，满足「同一发射体 ≥2 次瞄准发射」。
6. 其余行为显式 `context_only`（`annotation=null`，零 mask），保留原因。

| 来源 | 事件 | 行数 | 标注行 | 其中有效动作 | 像素最大偏差 | 声明容差 | review SHA256 |
|---|---|---|---|---|---|---|---|
| train | f2-train-st2-e1 | 48 | 3 | 3（全为动作 1） | 2.984 px | 3.98 px | `1254d247…` |
| validation | f2-validation-st2-e1 | 48 | 2 | 2（动作 3→2，含 1 处动作变化） | 3.488 px | 4.49 px | `ef14aed0…` |
| test | f2-test-st2-e1 | 48 | 3 | 1（动作 0；另 2 行当时未按射击 → unknown） | 1.924 px | 2.92 px | `5152002d…` |

> 标注行按「分散 + 优先覆盖新动作值」挑选：四个相邻帧往往是同一个动作（第一次试验就是 4×动作 1），既看不到动作变化，也无法区分「学到」与「常数预测」。这一轮因此把标注行拉开，validation 出现了 3→2 的变化；同时也暴露 test 有两行标注行当时没有按住射击——按合同这些行只能是 unknown，不能补成 hold。

**合同中三处「不是独立测量」的地方，本报告不掩盖：**

- `observation.player_px` 被合同要求**精确等于**同 hold 内存自机坐标 → 自机位置不是像素测量。
- 轨迹 id 必须是 `slot:N`（teacher 侧按池槽绑定）→ 轨迹**身份**来自内存，只有坐标是测量的。
- `emitter_id` 是 origin 簇的标签，不是被验证的实体身份；合同也不校验它。

---

## 4. 冻结计划

```text
v376-f2-bc-plan-20261105-r2.json  (schema v376-f2-bc-plan/1, frozen=true)
  config = DEFAULT_CONFIG, config_sha256 = evidence_sha(config)
  sources: 3 (train/validation/test) — replay + frame-proof + collect-report + audit-state 全 SHA 绑定
  events:  3 — stage=1, split 各一, template/episode/derived 组按 replay 归属
```

- 计划放在**仓库根**：`f2_dataset` 要求所有来源引用相对计划父目录且不得逃逸，所有被引用文件（`third_party/...`、`Document/...`）的共同父目录就是仓库根。
- 计划 SHA、manifest SHA、cache index/report SHA、checkpoint SHA 全部登记在 [sha256.json](<sha256.json>)。

---

## 5. build / audit / gate

```text
python -B -m touhou_rl.f2_dataset build v376-f2-bc-plan-20261105-r2.json third_party/th08/build/rl/f2/v376-f2-bc-dataset-20261105-r3   → exit 0
python -B -m touhou_rl.f2_dataset audit .../manifest.json                                                                                  → exit 0, passed=true
python -B -m touhou_rl.tactical_train check-gate --manifest .../manifest.json                                                               → exit 0
```

build 期间实际执行（不是结构夹具）：

- 三个 upstream 的 `audit_exact_dataset(deep=True)` 全量重算（3000 帧 × 3，逐帧 PNG 与 pixelSha256 复算）；
- 每行：`gui_input_current/previous` 与解码回放逐帧比对（WP-A 修复）、review 图像 SHA/内容、player_px、teacher 槽位、review capture/proof 身份、`player_px` 与内存一致；
- 跨 split 逐帧像素内容去重、输入/标签物理隔离（含硬链接别名检查）；
- `alignment_diagnostics.stage_clock_lag` 落盘（本轮三来源均为 0）。

audit 重新计算同一批内容后 `passed=true`；`check-gate` 再次审计后返回 exit 0。

---

## 6. 缓存与一次更新

```text
python -B -m touhou_rl.tactical_feature_cache --manifest ... --baseline third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt `
  --out third_party/th08/build/rl/f2/v376-f2-bc-cache-20261105-r3 --device cuda --batch-size 16      → exit 0
python -B -m touhou_rl.tactical_sequence_train --manifest ... --cache-index <cache>/index.json `
  --baseline .../policy.pt --out third_party/th08/build/rl/f2/v376-f2-bc-train-20261105-r3 `
  --arm N1 --seed 7 --epochs 1 --max-updates 1 --device cuda                                        → exit 0
```

| 检查项 | 结果 |
|---|---|
| baseline 身份 | `74abde8a996c13fae832913e4e5533bc9490688bc4c0aa73560e695f065d14be`，与 [baseline-20261104-r3](<../baseline-20261104-r3/baseline.json>) 登记一致 |
| 缓存协议 | `frozen-backbone.forward_core/real_pixels`，float32/no autocast/no TF32，torch 2.14.1+cu132 |
| test 隔离 | `test_tensor_loaded=false`、`test_cache_created=false`、训练报告 `test_used=false` |
| 有效更新 | `updates=1`，loss 2.7521，grad_norm 4.16，监督计数 `action=3`（即该事件 3 条有效标注行），option/target/termination/slack/local 全 0；validation_loss 3.2339 |
| 骨干不动 | `backbone_unchanged=true`；baseline SHA 前后一致 |
| checkpoint 读回 | `research-heads.pt` 可 `torch.load(weights_only=True)`，schema `v373-sequence-research-only/1`，含 18 个 head 张量 + 报告 |
| 声明 | `deployable=false`，`original_G4=false`，`objective=f2_behavior_cloning` |

> 监督计数 `action=3` 精确对应 train 事件的 3 条有效标注行：其余 45 行是 `context_only`，零梯度。这说明 **一次更新确实由真实标注驱动**，同时也说明数据量之小（全数据集仅 6 条有效监督行、4 个动作值、1 处动作变化）。

---

## 7. 本轮明确**没有**做的事

- 没有做 WP-F 独立放行审查（必须由未参与实现者执行）。
- 没有长训练、没有超参搜索、没有选模循环（1 update）。
- 没有评估泛化、闭环、真机、默认模型替换、canonical 变更。
- 没有把 Stage 1 的 133 个候选"补标成 F2"；它们仍在 [第一轮报告](<../wpb-pre-screen-20261105/review.md>) 中保持"未发现可确认 F2"。
- 没有删除或覆盖任何旧 run（`v376-f2-*-20261104-r1`、旧 review pack、旧报告全部保留）。

---

## 8. 复现顺序

```powershell
# 1. 补采（每个 role 一次；新目录、新 run）
python -B -m touhou_rl.pipeline collect --modality privileged --source replay --replay <rpy> `
  --out third_party/th08/build/rl/exact/v376-f2-<role>-st2-20261105-r1 --stage 1 --max-seconds 900 `
  --frame-proof exact --exact-config Document/Update/Alpha/v3.7.6/configs/f2-exact-capture-v1.json

# 2. 深审（真实数据内容审计）
python -c "from pathlib import Path; from touhou_rl import exact_audit; exact_audit.audit_exact_dataset(Path('<capture>'), deep=True, expect_exact=True, require_proof=True)"

# 3. 复核包 + 预筛 + 目视（见 WP-B 报告）
python -B tools/rl/v376_review_pack.py --raw <capture> --out <review-pack> --source-id <id> --stage 1
python tools/rl/v376_wpb_pre_screen.py --capture <name>=<capture> --review <review-root> --table

# 4. 复核（逐像素测量；可重复）
python tools/rl/v376_wpb_author_review.py --capture <capture> --source-id <id> --event-id <eid> `
  --stage 1 --start <window start sample> --rows 48 --out <review.json>

# 5. build / audit / gate
python -B -m touhou_rl.f2_dataset build v376-f2-bc-plan-20261105-r2.json <dataset>
python -B -m touhou_rl.f2_dataset audit <dataset>/manifest.json
python -B -m touhou_rl.tactical_train check-gate --manifest <dataset>/manifest.json

# 6. cache + 一次更新
python -B -m touhou_rl.tactical_feature_cache --manifest <dataset>/manifest.json `
  --baseline third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt --out <cache> --device cuda --batch-size 16
python -B -m touhou_rl.tactical_sequence_train --manifest <dataset>/manifest.json `
  --cache-index <cache>/index.json --baseline third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt `
  --out <run> --arm N1 --seed 7 --epochs 1 --max-updates 1 --device cuda
```
