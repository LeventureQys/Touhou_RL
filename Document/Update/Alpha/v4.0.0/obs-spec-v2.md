# 观测规格 obs-v2（P3-WP7 / EXP-P3-008 冻结交付物）

- 日期：2026-11-10
- 状态：**已冻结**（`OBS_SPEC_VERSION = "obs-v2"`，规范摘要 `obs_spec_v2_digest()` =
  `95110743993c090e2b30fa40ac349a350e74021b35464e3204c784a69ae4dba3`）
- 代码：`touhou_rl/v400/student/obs_spec_v2.py`（唯一实现；本文档是它的契约说明）
- 基线：[obs-v1](<obs-spec-v1.md>)（`24f0ddac…364a381`，**未改动**，逐字节锁定）
- 上游：[EXP-P3-005 §8](<experiments/EXP-P3-005-margin-resolution.md>)（"输入信息不足"、
  标签路线两次被否证）、[EXP-P3-007](<experiments/EXP-P3-007-teacher-v2j-durations.md>)
  （标签分辨率 ×4.31 仍不转化为闭环帧数）、13-P3 §2.1（动作历史要求）
- 下游：P3（M2/M3 两臂）、P4/P5/P6 部署适配

---

## 1. 冻结（先读这一节）

obs-v2 是**只加不改的增量规格**，这一点比"新版本"更重要：

1. **obs-v1 的每一个字节都不动**。`touhou_rl/v400/student/obs_spec.py` 的 sha256
   在测试里是字面量 `0d47daf1…5578b`，`obs_spec_digest()` 仍是 `24f0ddac…364a381`。
   obs-v2 通过 `from touhou_rl.v400.student import obs_spec` **复用** v1 的实现，
   不复制、不重写，因此 v1 的编码器/归一化/对齐不可能"顺手"漂移。
2. **新增的只有 `act_hist`**（8 个历史决策 × 18 维 one-hot = 144 维）。
   `obs_spec_v2()` 的继承字段与 `obs_spec()` 逐值相等（测试逐键比对），只改
   `version` / `excluded_inputs` 并新增 4 个键。
3. 每个 obs-v2 checkpoint/数据集都记录 `obs_spec_version` 与 `obs_spec_digest`，
   加载器用 `check_obs_spec_v2()` **拒绝**跨规格加载；obs-v1 的 checkpoint 仍由
   `touhou_rl.v400.student.model.load_checkpoint()` 按 v1 规格加载，两条链互不污染。
4. 摘要：`python -c "from touhou_rl.v400.student.obs_spec_v2 import obs_spec_v2_digest; print(obs_spec_v2_digest())"`。

---

## 2. 与 obs-v1 的关系（表格对照）

| 项 | obs-v1 | obs-v2 | 说明 |
|---|---|---|---|
| `play` 平面 | 64×64 灰度，4 帧堆叠（`t-3..t`） | **相同** | 继承；无重采样、无新通道 |
| `hud` 平面 | 56×160 绿色通道，1 帧 | **相同** | 继承 |
| 归一化 | `float32/255.0`，不减去均值 | **相同** | 继承 |
| 决策周期 | 2 逻辑帧 | **相同** | 继承 |
| 动作空间 | `cand = focus*9+move`（18）+ bomb（2） | **相同** | 继承 |
| 教师接口 | `softmax(scores/τ)`、`verified_mask` 只作诊断 | **相同** | 继承 |
| **动作历史** | 排除项（"reserved for obs-v2"） | **新增 `act_hist` 8×18=144** | 本文档 §3 |
| 自机状态 `px/py` | 排除 | **仍不含**（见 §4，数据侧拿不到） | 保留为未来 `obs-v3` |
| 危险摘要 lives/bombs | 排除 | **仍不含**（见 §4，部署侧拿不到） | 字段留空并写进摘要 |

输入张量（模型侧）：

```text
play     (4,64,64)  float32 [0,1]     # obs-v1, unchanged
hud      (1,56,160) float32 [0,1]     # obs-v1, unchanged
act_hist (8,18)     float32 [0,1]     # obs-v2 addition (144 one-hot bits)
```

---

## 3. `act_hist`：8 个历史决策的 one-hot

| 项 | 冻结值 | 理由 |
|---|---|---|
| 长度 | **8 个决策**（`ACTION_HISTORY_DECISIONS = 8`） | 任务卡规定；8 决策 = 16 逻辑帧，覆盖 4 帧堆叠（8 逻辑帧）之外更早的运动 |
| 步长 | 1 个决策（`ACTION_HISTORY_STRIDE_DECISIONS = 1`） | 决策级历史，不做帧级，避免与决策周期耦合 |
| 编码 | 18 位 one-hot，索引 = `cand = focus*9 + move` | 与动作空间同一张表（`gym/actions.py`） |
| 维度 | 8×18 = **144**（`ACTION_HISTORY_DIM`） | |
| 顺序 | **最旧在前**：`row 0 = a[t-8] … row 7 = a[t-1]` | 与 4 帧堆叠"最旧在前"一致 |
| 当前动作 | **不含** `a[t]` | 标签就是 `a[t]`；含它等于把目标泄漏进输入 |
| 补齐 | 行全零 = "无记录"（`ACTION_HISTORY_PAD`） | 全零不是合法 `cand`，不会与动作 0（one-hot）混淆 |
| bomb 位 | **不进历史** | 任务卡规定 18 位；bomb 是独立的二分类头目标 |
| dtype | 数据集存 `uint8`，模型侧转 `float32` | 与 play/hud 同样的"存整数、喂浮点"约定 |

### 3.1 数据集侧重建规则

每个分片（一个分片 = 一局 = 一个种子）内**按决策序号**重建，起点与换面清零：

| 来源 | "已执行动作"取自 | 理由 |
|---|---|---|
| D0（教师自己跑） | `chosen_plan.first_action` | D0 是教师驱动的轨迹，教师在该决策提交的第一个槽就是执行动作 |
| DAgger R1 | `dagger.executed_cand` | DAgger 的执行动作是**环境实际执行的**（学生贪心，β=0.2 时被教师替换），教师 `chosen_plan.first_action` 只是"教师会选什么"，不是轨迹 |
| 两者都缺 | `ObsSpecV2Error` | 不猜 |

清零边界（`build_action_history` 的唯一实现）：

1. **分片起点**：`t < 7` 时更早的行是全零，绝不继承上一分片；
2. **换面**：`stage[t] != stage[t-1]` 的决策 `t` 起重新累计（换面 = 新一局，
   上一面的动作不是同一轨迹的上下文）；
3. 分片内 `stage` 恒定（D0/R1 实测 27/27 片都是 1 个 segment），
   但规则仍然实现并单测，R2 之后可能出现跨面分片。

### 3.2 部署侧（评测）语义

`ActionHistoryBuffer` 与 `ObservationBuffer` 对称：

- `reset()` 在 `RESET` 后调用；
- 每决策**执行完**才 `push(cand)`，因此 `rows()` 在决策 `t` 恰好是 `a[0..t-1]`
  （评测代码里 push 在 STEP/STATE 之后，顺序由单测锁住）；
- episode 开头不足 8 个动作时前导行全零，**不重复第一帧/第一动作**——
  训练数据在这些位置也是全零，两侧必须一致（这与 obs-v1 的帧补齐规则不同，因为
  帧补齐是"看不到更早的帧"，而这里"没有更早的动作"本身就是真实信息）。

---

## 4. 两个**明确不含**的输入（以及为什么）

### 4.1 自机状态 `px/py` —— 数据侧拿不到

D0 / DAgger 的 label 行字段是：`scores[18]`、`verified_mask[18]`、`bomb`、`margin`、
`chosen_plan`、`frame_event_flags`、`hash_*`、`obs{…}`。**没有 `px/py`**。
`chosen_plan.end_xy` 是**计划终点**（教师假设自己走到哪里），不是自机当前位置；
把它当自机状态会训练出一个错的输入。

从 `play64` 反推自机位置属于 P2 的感知问题：EXP-P2-003/004 的检测器是子弹/事件检测器，
没有自机定位输出，且自机精灵会被自机弹幕遮挡。因此 obs-v2 **不含自机状态**，
也不引入一个未验证的估计器。未来若要加，必须有独立的、可复现的位置源 → `obs-v3`。

### 4.2 危险摘要 `lives/bombs` 差分 —— 部署侧拿不到

任务卡把它列为可选项。能否做取决于**部署侧**（`deploy_pixel`）能不能从观测里读出
lives/bombs：

- obs-v1 的 `hud` 平面是**像素**（数字的字形），不是数值；
- 仓库里没有 HUD 数字读取器：`touhou_rl/v400/labels/hud.py` 明确标注"HUD label = 它画的那个
  状态值，不是 OCR"（12-P2 §3.2）；
- 评测器可以从引擎的 `HUD` 协议回包拿到 counters（`gym/protocol.py` 的 `hud()`），
  但那是**特权读数**，用它就改变了 `v400_student_eval.py` 的 `deploy_pixel` 契约
  （`policy_input` / `privilege` 字段会被验证器判为不一致）。

用两种不同的输入训练/部署（训练用引擎 state、部署用像素）正是本项目一直在避免的
train/deploy 错配。所以 obs-v2 把该字段列为 **reserved、不填充**，并在摘要里写
`danger_summary.present = false`——checkpoint 无法声称自己见过它。

---

## 5. 模型与训练契约（两臂）

| 臂 | 结构 | 输入 | 参数量 |
|---|---|---|---|
| **M2** `StudentNetV2` | 与 v1 相同的双流卷积（width 48）+ `act_hist` 编码 `144→64`（ReLU）+ `Linear(208,256)`（ReLU）+ 两个头 | 单决策（`S=1`） | **229,924**（≤300k 上限） |
| **M3** `StudentGRU` | 同上编码器 + `Linear(208,128)`（ReLU）+ **单层 GRU(hidden 256)** + 两个头；序列长度 **8 决策**，BPTT 截断 **8** | 8 决策窗口 | 499,620（不含 300k 限制） |

关键约束：

- **序列 = 8 个连续决策**，且窗口**不得跨分片或跨换面**（`valid_window_ends`）；
  训练时对窗口内 8 步**全部**计损失（每步都是教师标签），因此每步的算力与 M2 的一次
  单决策前向同量级；`bptt == seq_len` 是默认，`truncate_bptt` 支持更短的截断并单测梯度。
- **部署与训练同函数**：评测时 GRU 用 `GRUCache` 缓存最近 8 个决策的编码，
  新决策只做 1 次编码 + 1 次 8 步 GRU；离线评测用同一窗口（`predict_gru`），
  单测校验"缓存逐步前向"与"整窗前向取最后一步"给出同一动作。
- **对齐**：`act_hist` 与 play/hud 一样取**决策点之前**的信息（`frames[i-3..i] -> mask[i+1]`），
  因此 obs-v2 不改变 `decision_time_before_commit` 的对齐语义。

---

## 6. 数据集 root 契约（新增字段单独存）

obs-v2 **不复制** 1.9 GiB 的 `play64`/`hud` 平面，新 root 只存新增字段 + 溯源：

```text
.tmp_v400_obsv2/obsv2/
  manifest.json                     schema=v400-obs-v2-manifest-1
                                    obs_spec_version/digest（obs-v2）
                                    base_obs_spec_digest（obs-v1 字面量）
                                    sources[]（每个 v1 root + manifest sha256）
                                    totals{shards,decisions,act_hist_bytes,...}
  shards/<key>/act_hist.npy         (N,8,18) uint8
  shards/<key>/stage.npy            (N,) int32（换面边界，训练窗口用）
  shards/<key>/shard.json           source_dir、decisions、act_hist/stage sha256、
                                    v1 两个平面的 sha256、逐行对齐声明、
                                    history{segments, rows_with_padding, executed_source,…}
```

`build` 侧（`python -m touhou_rl.v400.student.obs_v2_dataset build`）对每片：

1. 用**冻结的 v1 loader**（`dataset.load_teacher_shard`）验 schema、obs-v1 摘要、
   对齐（`frames[d-4..d-1] -> mask[d]`）、平面形状/大小与**记录的两个平面 sha256**；
2. 从**原始** label 行重建执行动作（DAgger 的 `dagger` 块在 loader 的校验记录里被丢掉，
   所以必须读 raw 行），逐行校验 `decision == i` 和 `decision_frame == i * frames_per_decision`；
3. 写 `act_hist.npy`/`stage.npy`，回读比对字节；
4. 全部 shard 决策数之和必须等于各源 manifest 的 `totals.decisions`，否则报错。

`verify` 侧对**活源**重跑 1–3 并比对：

```text
27 片 / 76,912 决策 = D0 53,512（16 片）+ DAgger R1 23,400（11 片）
v1 平面 sha256：27/27 重算并匹配（obs-v1 字节未变）
逐行 decision/frame 对齐：27/27
act_hist 重建逐字节相等：27/27；stage 平面相等：27/27
→ verified: true（样本数/对齐/字节与 v1 完全一致，只有新增字段不同）
```

---

## 7. Fail-closed 规则（已实现并有测试）

| 情况 | 行为 |
|---|---|
| checkpoint 的 `obs_spec_version` 不是 `obs-v2` | `ObsSpecV2Mismatch`（obs-v1 checkpoint 不能被当 v2 读） |
| checkpoint 的 `obs_spec_digest` 不是冻结值（含被改成 v1 摘要） | `ObsSpecV2Mismatch` |
| `base_obs_spec_digest` 不是 `24f0ddac…` | `ValueError`（v2 必须声明它扩展的是哪一版 v1） |
| `act_hist` 形状不是 `(N,8,18)` / dtype 不是 uint8 / 行不是 one-hot 或全零 | `ObsSpecV2Error` |
| 某决策既无 `dagger.executed_cand` 也无 `chosen_plan.first_action` | `ObsSpecV2Error`（不猜） |
| obs-v2 root 的决策数与 v1 源 manifest 不符 | `DatasetError` |
| 分片已存在且非空（拒绝静默覆盖/合并） | `DatasetError` |
| 评测记录的 obs 规格与 checkpoint 不一致 | `EvalError` |
| 评测记录 `window_alignment.renders_per_decision` 与 `--obs-window` 不符 | `EvalError`（混尺拒绝） |
| GRU 序列长度 ≠ `seq_len`；`bptt` 不在 `[1, seq_len]` | `ValueError` |

---

## 8. 已知限制（写进报告，不能当成绩）

1. **动作历史只编码"按了哪个方向/是否 focus"，不含 bomb**：bomb 使用率极低
   （D0/R1 正例 27/12,008），历史里加 bomb 位只会多 8 个几乎恒零的维度；若后续
   bomb 策略变重要，需要 `obs-v2b` 明确加位。
2. **没有自机位置/速度**：本规格无法区分"我在哪"和"我朝哪走"，只能靠 4 帧堆叠的
   隐含运动 + 8 决策动作历史间接推断。若"被逼到角落"需要位置，正确动作是新观测源，
   不是更长的动作历史。
3. **没有敌弹/激光的显式摘要**：像素是唯一信息通道，与 obs-v1 相同。
4. **换面语义只做到"清零"**：obs-v2 不编码 stage id，也不跨面传播状态。

---

## 9. 复现/校验命令

```powershell
# 1) 建 obs-v2 root（只加字段，不复制平面）
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -X utf8 -m touhou_rl.v400.student.obs_v2_dataset build `
  --root .tmp_v400_d0/d0 --root .tmp_v400_dagger/d0_extra --out .tmp_v400_obsv2/obsv2

# 2) 对活源重验（v1 字节 / 样本数 / 对齐 / history 重建）
python -X utf8 -m touhou_rl.v400.student.obs_v2_dataset verify --root .tmp_v400_obsv2/obsv2

# 3) 契约测试（digest 冻结、obs-v1 逐字节、边界清零、GRU 打包/BPTT、评测 schema）
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_obsv2/pytest -q tests/test_v400_obs_v2.py
```

---

## 修订记录

| 日期 | 版本 | 修改 | 原因 |
|---|---|---|---|
| 2026-11-10 | obs-v2 | 初版冻结：obs-v1 + `act_hist`（8×18 one-hot，causal，边界清零）；`px/py` 与 lives/bombs 差分**不含**并写明原因 | EXP-P3-005/007 两次否证标签路线后，转向"输入信息不足"假设（EXP-P3-008） |
