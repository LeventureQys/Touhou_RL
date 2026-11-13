# WP04 状态（学习链路诊断）

- 时间：2026-11-13T04:15:00+08:00
- 结论：**WP04.0 四项短诊断完成；WP04 正式四臂（B0–B3）尚未开始。** 诊断暴露两个必须先处理的决策：标签软目标规则与步数预算。

## 1. WP04.0 完成

证据：[`DIAG.md`](DIAG.md)、[`diag.json`](diag.json)；`pytest tests/test_v410_student_diagnostics.py` → **51 passed**；compileall exit 0；device CUDA RTX 4070 SUPER。

真实数据：20 shards / 66,143 decisions / seeds 1–20。

1. **小集合拟合**（256 tier-A 样本，16 动作，多数类基线 6.64%）：
   - 20 epoch / 40 step：loss 2.8856→1.5273，top1 **0.8047**，未过 95%；
   - 同超参 +18 step（≈9 epoch）：top1 **0.96875**，loss 0.4993；
   - 梯度 58/58 非零、0 个同输入冲突标签 → 缺口是**步数预算**，不是代码/标签/优化器缺陷。
2. **输入敏感性**（12 对 × 5 扰动）：top1 改变 19/60，L2 mean 2.003；roll 9/12 翻转、遮挡自机 1/12、平移弹幕 9/12、**act_hist 打乱 0/12（L2 0.086）**、act_hist 归零 0/12 → **act_hist 分支近乎冻结**。
3. **离线确定性**：重复/反序 forward bitwise 0.0；batch 1 vs 8 = **3.986e-4**（超 1e-5，CPU 同权重 1.43e-6）；第一处不一致已存 `.tmp_v410_wp04/determinism_first_mismatch.pt`。
4. **标签信号审计**（66,143 行）：每行恰好 1 个 verified（候选 5.56%，unknown 94.44%），verified_dead=0；τ=1 时 KL(raw‖uniform)=2.3169，**98.27% 来自 unverified discount**；undiscounted KL 仅 0.0437；82.10% 行折前 top-gap=0；top1 仅 0.0438% 翻转；各 τ 的 discount KL 占比 94.7–98.3%。

## 1.1 WP04.1 BC 训练 smoke（完成，非正式四臂）

证据：[`bc_smoke.json`](bc_smoke.json)（B0/B1 单种子 1 epoch）；`pytest tests/test_v410_student_bc.py` → **53 passed**；compileall exit 0。

- 预注册标签规则：**chosen one-hot CE**；B0 不用 bomb，B1 用 bomb BCE aux 0.5；unknown 不作负例。
- 切分：train seeds 1–14（46,397 samples）、val 15–17（9,874）、test 18–20；整 shard/seed 切分。
- 模型：B0 **731,802** 参数；B1 **731,931** 参数（+129 bomb head）。
- 单 epoch：B0 train top1 0.4895 / val top1 0.5613；B1 train 0.4849 / val 0.5482；bomb aux 0.0144。
- 吞吐：B0 363 steps、3,429.8 samples/s、26.83 steps/s；数据加载占训练 39%；batch 128 未 fallback。
- 结论：训练链路/标签/切分/checkpoint 可用；**1 epoch 不代表四臂结论**，需 20 epoch ×3 seeds 正式跑。
- **阻塞级发现**：S1 的 `bomb` 标签 66,143 行全为 0（老师从不 bomb），B1 的 bomb aux 退化为负先验（val BCE 0.000895、accuracy 1.0），因此 **B1/B3 不能用于 aux 有效性结论**。正式四臂若要比较“感知辅助”，必须换有信息量的辅助任务（如感知/重建或安全集），另立预注册；当前 WP04.2 的 B1/B3 只能作为“零辅助退化”记录。

## 1.2 WP04.2 正式四臂（完成，12/12）

证据：[ARMS.md](ARMS.md)、[arms.json](arms.json)；`pytest tests/test_v410_student_bc_arms.py` → **43 passed**；compileall exit 0；12/12 run ok，无 partial。

| 臂 | 参数 | val top1（3 seed mean） | val macroR（mean） |
|---|---:|---|---|
| B0 | 731,802 | **0.6254 ± 0.0038** | **0.3101 ± 0.0033** |
| B1 | 731,931 | **0.6241 ± 0.0023** | **0.3026 ± 0.0069** |
| B2 | 2,600,951 | **0.6594 ± 0.0049** | **0.3131 ± 0.0179** |
| B3 | 2,601,464 | **0.6605 ± 0.0062** | **0.3305 ± 0.0077** |

- 每 run 20×363=7,260 steps、927,940 samples_seen、batch 128、OOM fallback 0；矩阵 wall≈86 min。
- **全部臂在 ep3–5 达峰后过拟合**（3-seed 均值峰值 B0 0.6816@4、B1 0.6798@4、B2 0.6859@5、B3 0.6883@4），ep20 回落；正式按 ep20 报，不能事后挑峰值。
- B2 的 top1 优势几乎全来自多数类（动作0 recall 0.8196→0.8753、pred 占比 0.4517→0.5057），macroR 仅 +0.0030；不能称“空间结构更好”。
- B1−B0：Δtop1 −0.0012 / ΔmacroR −0.0074；B3−B2：+0.0011 / +0.0174（B3 最好但 bomb 标签恒 0，不能归因 aux；3 seeds 噪声同量级）。
- 仍未做四臂 20 局开发评测（WP04.5 进行中）；不能写 WP04 通过。
- val top1 在 epoch 4–5 附近最高（约 0.68），epoch 20 为 0.628，已出现明显过拟合；正式报告必须同时给 macro recall 与开发局，不能只看 train top1。
- B1/B3 的 bomb aux 因 S1 bomb 全零而退化，不能用于 aux 有效性结论；若要比较感知辅助，需另立预注册的辅助任务。

## 1.3 WP04.3 真实引擎开发局 smoke（完成，差）

证据：[`EVAL.md`](EVAL.md)、[`eval_smoke.json`](eval_smoke.json)；`pytest tests/test_v410_student_eval.py` → **111 passed**；compileall exit 0。

- 入口：v410 B0 checkpoint → ProbeNet → 真实引擎 env 角色像素 readback → 每 2 帧 argmax；fail-closed 加载/seed policy。
- B0_s0（val top1 0.6280）在 dev seeds 1000/1001 的真实结果：
  - seed1000：**1,398 帧首死**、stage_clear=false；
  - seed1001：**818 帧首死**、stage_clear=false；
  - 动作偏 idle（42%/51%），S/SE 几乎为 0；n=2 不构成开发成绩，也不是 G410-2。
- 结论：离线 BC 基线的闭环表现很差；需要正式 20 dev episodes/臂、过拟合处置与数据/模型改进，不能写 WP04 通过。
- **B0_s0 20 dev episodes（seeds 1000–1019）**：**0/20 过关、20/20 首死**；frames mean **1,453.25**、median **1,374**、min 815、max 2,386；动作 idle 45.81%、S/SE 仅 0.25%、focus 6.54%（14,537 决策）。该工具把 n=20 仍标为 smoke（`is_a_development_score=false`），不是正式四臂分数，但足以证明闭环落差。
- 可复现性：重叠 seeds 1000/1001 与 WP04.3 smoke **bitwise 一致**（frames/score/stop_reason/action_hist/policy_input_sha256/action_sequence_sha256）；只有 2 个 seed 有重复证据。
- 详见 [`EVAL20.md`](EVAL20.md)。
- 入口目前只接受 B0；B1/B3 需 bomb-head loader，B2/B3 为 spatial 家族，尚未评测。

## 1.4 WP04.5 四臂开发评测（进行中）

- `v410_student_eval.py` 已扩展 B0/B1/B2/B3 严格 loader；`tests/test_v410_student_eval.py` **152 passed**；B0/B1/B2/B3 各 1 个 dev episode smoke 均 exit 0。
- B0 seed1000 与 WP04.4 的 20 局记录 **bit-identical**（1398 帧，action/policy SHA 相同），无接线漂移。
- B1/B3 bomb logits 均值约 -51/-28、positive share 0.0（S1 bomb 全零的负先验），bomb head 只记录不执行。
- 正式四臂 ×20 episodes（seeds 1000–1019）正在串行运行：
  - **B0**：0/20、frames mean **1,453.25**、idle 45.81%；
  - **B1**：0/20、frames mean **1,765.9**（median 1,615、max 3,021）、idle **57.18%**；比 B0 长约 21.5%，但 n=20 无 CI，不能下“aux 有效”结论（bomb aux 全零）；
  - **B2**：0/20、frames mean **1,161.2**（median 1,240、max 1,446）、idle **83.59%**；比 B0 更差，离线 top1 0.6594 的优势未转化为闭环；
  - B3 尚未完成。
- 当前**无最终四臂数字**，不能写 WP04 通过或 G410-2。

## 1.5 四臂 seed0 20 局开发评测（完成，描述性）

证据：[`EVALARMS.md`](EVALARMS.md)、[`eval_arms_seed0_20.json`](eval_arms_seed0_20.json)（schema `v410-student-eval-arms/1`）；`pytest tests/test_v410_student_eval.py` → **152 passed**；四臂各 20 局全部 exit 0；B0 与 WP04.4 的 20/20 seeds **逐字节一致**。

| 臂 | clear | frames mean | median | min/max | idle 占比 |
|---|---:|---:|---:|---:|---:|
| B0 | 0/20 | 1,453.25 | 1,374 | 815/2,386 | 45.81% |
| B1 | 0/20 | 1,765.90 | 1,615 | 1,209/3,021 | 57.18% |
| B2 | 0/20 | 1,161.20 | 1,240 | 746/1,446 | 83.59% |
| B3 | 0/20 | 1,398.55 | 1,396 | 777/2,171 | 63.69% |

- 四臂全部 **0/20**；Wilson 95%（描述性）均为 [0, 0.1611]；deaths=1、lives_end=2、bombs_used=0。
- B1/B3 bomb logits 全负（mean −47.2/−28.4、share_positive 0），与 S1 bomb 全零一致。
- **离线/闭环脱节**：B2/B3 离线 top1 更高，闭环反而更差/更 idle。
- 仅 seed0 / last epoch；seed1/2 与最佳 val 选择规则尚未做，不能写 WP04 通过或 G410-2。

## 2. 对四臂的影响（必须先决策）

- S1 的 verified 支撑恒为单元素 → **“KL/集合软目标”臂在定义上退化为 one-hot CE**；软目标信息量约 98% 来自 discount，教师几何只贡献 0.0437 nats/行。
- 因此 B0–B3 的 soft/set 臂与 CE 臂如果直接开跑，无法区分“模型/容量”与“标签规则”的贡献。
- 必须先明确：使用 `undiscounted` 分数、原始几何软标签、还是 replay 软标签；并固定 step 预算（本次 40 step 确定不足，建议 ≥160 step 或 batch 64）。
- act_hist 消融（zero/shuffle）应作为 obs-v3 第三平面是否被使用的验收项。

## 3. 下一步

1. **WP04.1**：预注册标签规则与步数预算；在 780 行 tier-A 子集做 lr/steps 小扫描（不是正式四臂）。
2. 正式四臂 B0–B3 每臂 3 训练种子、20 有效 epoch/等价 steps、同数据/切分/曝光；输出开发局，而不是只看训练 acc。
3. G0.1 批量重试与 WP03.2 MULTI 仍并行。
4. 未 commit/push；未训练正式模型；未修改 v400。
