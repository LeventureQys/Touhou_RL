# 观测规格 obs-v1（P2-WP5 冻结交付物）

- 日期：2026-11-09
- 状态：**已冻结**（`OBS_SPEC_VERSION = "obs-v1"`，规范摘要 `obs_spec_digest()` =
  `24f0ddace59104938dbe991c0d8efbb6a8cd21dfd5a29ebcb6d023b5b364a381`）
- 代码：`touhou_rl/v400/student/obs_spec.py`（唯一实现；本文档是它的契约说明）
- 上游：P2-WP4 分辨率对照（[EXP-P2-003 §6](<experiments/EXP-P2-003-reward-detector-g2.md>)）、
  P2 标签几何（`touhou_rl/v400/labels/spec.py`）、13-P3 §2
- 下游：P3（BC/DAgger）、P4（自我改进）、P5/P6（部署适配）都必须按本规格构造输入

---

## 1. 冻结（先读这一节）

"冻结"在这里是**接口约束**，不是"以后不许改进"：

1. 任何**改变观测含义**的修改（分辨率、通道、帧堆叠、决策周期、归一化、HUD 是否包含、
   动作空间的语义、教师软标签的对应关系）都必须新开版本号（`obs-v2`）、新写一份
   `obs-spec-v2.md`，并让使用旧规格的 checkpoint/数据集显式不兼容。
2. **不改变观测含义**的训练超参数（例如教师软标签温度 `τ`、batch size、学习率、
   DAgger 的 β 日程）不算观测改动，记录在对应 checkpoint 的 `train` 字段即可。
3. 机器可读的摘要由 `obs_spec_digest()` 给出（`obs_spec()` 的规范化 JSON 的 sha256）。
   `StudentNet` 的 checkpoint 记录 `obs_spec_version` 与 `obs_spec_digest`，
   `load_checkpoint()` 用 `check_obs_spec()` **拒绝**加载不同规格的权重。
4. P3/P4/P5 不得"就地"改常量。发现规格不足时，正确动作是：记实验卡 → 新版本 →
   重训/重测；不得只改代码让旧 checkpoint 看起来还在工作。

---

## 2. 观测张量

| 项 | 冻结值 | 依据 |
|---|---|---|
| 场地分辨率 | **64×64**（`play64`，灰度，uint8） | P2-WP4：同参数下 64×64 训练快 1.56×、显存少 30%，事件级 death 召回 33.33% vs 3.13%（**只在 dive 域成立**，见 §7） |
| 通道 | 1（灰度）。亮度 = `(77R + 151G + 28B) >> 8`（整数 BT.601） | 与 P2 采集器 `encode_frame` 逐字节一致（`tests/test_v400_student.py` 对 P2 采集器做字节等价测试） |
| 下采样 | 场地 384×448 从屏幕 `(32,16)` 起，按 **6 列 × 7 行**块做整数均值 `sum // n`，原生 64×64（不是从 96×112 重采样） | `labels/spec.py` 的 viewport；P2 采集器 meta 记录 `block: {"play64": [7, 6]}` |
| 帧堆叠 | **4** 帧，最旧在前（`t-3, t-2, t-1, t`）；episode 开头不足 4 帧时**重复最早一帧**补齐 | 12-P2/EXP-P2-003 检测器同样的 4 帧堆叠；补帧是确定性行为，写进 spec |
| HUD 条带 | **包含**，独立输入流：`hud` 56×160，uint8，**原生分辨率**绿色通道；score 行 48–72、lives 80–96、bombs 96–112，列 480–640；只取**当前 1 帧** | 仅靠 playfield 读不出残机/bomb 数，而动作空间含 bomb；EXP-P2-003 §5 证明原生 HUD 的 lives/bombs 读数 100% |
| 归一化 | `float32(uint8) / 255.0`，每个平面独立，范围 [0,1]，**不减去均值/不除标准差** | 与 P2 检测器一致（`EXP-P2-003` §4） |
| 输入张量 | `play: (4,64,64) float32`、`hud: (1,56,160) float32` | `normalize_play` / `normalize_hud` |
| 排除输入 | 特权状态、最近动作历史、P2 奖励头输出 | 见 §6；动作历史保留给 `obs-v2`（13-P3 §2.1 要求，但它需要 DAgger/部署共享的决策历史契约，不能在 S9 里半实现） |

**帧率/决策周期**：一个决策 = **2 个逻辑帧**（`DECISION_FRAMES = 2`），与
`touhou_rl.v400.gym.env.DECISION_FRAMES`、`touhou_rl.v400.teacher.config.DECISION_FRAMES`
一致（03 §3.3）。`frames` 一律指引擎的逻辑帧数，不是墙钟时间，也不是渲染帧率。

**动作空间**：`cand = focus * 9 + move`，`move ∈ {0..8}` 是 9 个方向（idle/N/NE/E/SE/S/SW/W/NW），
`focus ∈ {0,1}`，共 **18** 类；`bomb ∈ {0,1}` 共 2 类；**shoot 恒开**（不是动作维度）。
按键掩码由 `touhou_rl/v400/gym/actions.py` 的 `cand_to_mask(cand, bomb)` 唯一生成，
与 `touhou_rl/spec.py` 的按键位一致；`mask_to_cand` 是它的逆，遇到不可表示掩码直接报错。

**动作持续时间**：学生每个决策输出 **1 个动作、持续 1 个决策 = 2 个逻辑帧**
（`ACTION_DURATION_DECISIONS = 1`）。教师搜索的根是 `(first_action, duration)`，
`duration` 的单位是**决策槽**（`config.durations = (1, 2)`）；因此教师 `duration=1`
与学生一个动作等价，`duration=2` 等于**连续两个**学生决策，不能当成一个。

---

## 3. 与教师的接口（学生输出 ↔ 教师 scores）

| 方向 | 契约 |
|---|---|
| 教师 → 学生 | 每个决策给出 `TeacherLabel`：`scores[18]`（`geo_score`，未验证动作已按 11-P1 §3 折扣）、`verified_mask[18]`、`bomb`、`chosen_plan`、`margin`。 |
| 软标签 | `π_teacher(a) = softmax(scores[18] / τ)`，默认 `τ = 1.0`（`TEACHER_TAU_DEFAULT`）。τ 是**训练配置**，P3 DAgger 可依据教师分数分布标定（13-P3 §2.4 要求"会死的动作概率 < 1e-3"），标定结果写进 checkpoint，不改 obs 版本。 |
| 学生 → 教师 | 学生输出 `move_logits[18] + bomb_logits[2]`；`π_student = softmax(move_logits)`。 |
| 主损失 | `KL(π_teacher || π_student)`（18 类，软标签）。 |
| bomb | `BCE(bomb_teacher, sigmoid(bomb_logit))`；教师 `bomb` 来自其决策流程，不是本文档推断的。 |
| verified_mask 的用法 | 只作**信任掩码/诊断**：未验证候选的分数已由教师折扣，KL 直接用 `scores` 即可；禁止把 mask=0 的候选当成"负样本 0 分"另造损失。 |
| 对齐单位 | 学生一个决策 = 教师 `duration=1` 的一个根。DAgger 记录 `(obs, TeacherLabel)` 时，`obs` 必须是**教师被查询的那个状态**，不是教师执行后到达的状态。 |

当前教师标签只有 stage 1 的 1500 个决策，且 JSONL **没有观测引用**（没有 `obs`/像素分片键），
所以 `dataset.load_teacher_samples()` 会 fail-closed（见 §5）。这是 P3-WP1 的输入缺口，
不是本规格的缺陷。

---

## 4. 数据集/checkpoint 的一致性约束

- 数据集 plane 形状必须是 `play64 (64,64)` + `hud (56,160)`，label 版本 `p2-v2-1`；
  `dataset.load_index()` 对 schema、label_version、plane_shapes、frames_total 逐项校验。
- 学生样本固定为 `frames[i-3..i] -> mask[i+1]`：采集器先写 mask 再 `STEP`、后读像素，
  所以第 `i` 行 mask 是**产生**第 `i` 帧的动作；用同一帧的 mask 当标签会把动作的即时效果
  泄漏给模型。`label_offset=1` 在 `BehaviorCloneDataset` 里是硬编码约束，改它必须新版本。
- checkpoint 必须记录：`obs_spec_version`、`obs_spec_digest`、模型结构、参数量、
  `train` 统计、`provenance`。`load_checkpoint()` 校验版本与摘要。
- 规格摘要：`python -c "from touhou_rl.v400.student.obs_spec import obs_spec_digest; print(obs_spec_digest())"`。

---

## 5. Fail-closed 规则（已实现并有测试）

| 情况 | 行为 |
|---|---|
| index/split schema、label_version、plane 形状/大小不符 | `DatasetError`，不训练 |
| 关卡整体切分被破坏（同一 stage 跨 split） | `DatasetError`（`load_split`） |
| 样本窗口不足、split 为空 | `DatasetError` |
| mask 不是学生动作（无 shoot 位/方向有歧义） | 该行计入 `actions_unmappable`；取到该标签时 `DatasetError` |
| 教师标签缺少 `scores/verified_mask/bomb` 或形状不对 | `DatasetError` |
| 教师标签无观测引用 / 样本量 < `MIN_TEACHER_SAMPLES`（50,000） | `DatasetInsufficient`，消息里带真实计数与 V0 免责声明 |

---

## 6. 为什么是这些选择（可审计的理由，不掩饰限制）

1. **64×64 而不是 96×112**：EXP-P2-003 §6 的同参数对照里 64×64 训练快 1.56×、显存少
   30%，且 death 事件级召回 33.33%（96×112 为 3.13%）。但该优势**只在 dive 采集域成立**
   （still 的 48 个死亡事件在两种分辨率下都是 0/48），因此本规格的措辞是"没有证据要求
   96×112，64×64 已足够且更省"，不是"64×64 普遍更强"。
2. **HUD 原生分辨率、不下采样**：把 HUD 降 2× 后 `000000000` 糊成团块、lives 星形不可数，
   P2 的 ≥99.5% 读数门不可能达成（EXP-P2-003 §3）。
3. **灰度而非 RGB**：P2 的存储平面与检测器都是灰度；本规格不引入第二套通道语义。
4. **4 帧堆叠**：与 P2 检测器一致，且决策周期 2 帧时 4 帧覆盖最近 8 个逻辑帧。
5. **不含动作历史**：13-P3 §2.1 的动作历史需要 DAgger/部署共享的决策历史契约；
   在 S9 里实现会得到一个与部署不一致的半成品，故推迟到 `obs-v2` 并明确记为排除项。
6. **τ 默认 1.0**：教师 `geo_score` 的量纲（样例约 111–117）下 τ=1.0 是保守起点；
   "致命动作概率 < 1e-3" 需要 P3 DAgger 用真实标签分布标定，不能凭猜测定值。

---

## 7. 已知限制（写进门报告，不能当成绩）

- 这是一份**观测**规格，不是能力证明：`play64` 让 P2 的 death 检测器在 dive 上可学，
  但 P2 的 death 门（≥99% 下界）**没有通过**，且 `still` 召回 0/48。
- 3 局 S1N 的 V0 BC 评测不是 G3，也不是 P3 成绩；`checkpoint-v0` 学的是
  `random/still/dive` 采集策略的掩码。
- HUD 只取 score/lives/bombs 三条带，不含 power/graze/point/time；如果 P3 发现
  bomb 决策需要更多 HUD 信息，走 `obs-v2`。

---

## 8. 复现/校验命令

```powershell
# Windows：规格常量、字节等价、dataset 校验、模型/checkpoint、评测 schema（19 用例）
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_p3/pytest -q tests/test_v400_student.py
```

```bash
# WSL：打印摘要
cd /mnt/d/WorkShop/Touhou_RL
PYTHONPATH=/mnt/d/WorkShop/Touhou_RL /root/v400-venv/bin/python -c \
  "from touhou_rl.v400.student.obs_spec import obs_spec_digest, obs_spec; print(obs_spec_digest())"
```

## 修订记录

| 日期 | 版本 | 修改 | 原因 |
|---|---|---|---|
| 2026-11-09 | obs-v1 | 初版冻结：64×64 灰度 + 4 帧堆叠 + 56×160 HUD + 18×2 动作 + 教师软标签接口 | P2-WP4 结论（EXP-P2-003 §6）+ P3 S9 规格冻结 |
