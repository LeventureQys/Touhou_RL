# W03D — 观测编码器 + GRU + 四个动作头（actor 模型类）

**状态：完成（本切片范围内）**

**本切片只交付一个 `nn.Module`（actor 策略候选）及其合成单测，不构成任何游戏能力证据。**
未启动引擎、未读游戏数据 / checkpoint / rollout、未训练、未评测、未产生或删除任何资产。
测试里的观测是由 `observation.encode_o2` 对**种子噪声帧**编码得到的，因此形状是冻结的 O2
形状，像素本身没有任何含义。

---

## 1. 范围

只做 `DEVELOPMENT_GUIDE.md` §4.3「第一个可实现网络候选」这一个网络，以及 §5.1 的四个动作头
语义。**不做** critic、六个辅助头、优化器、rollout 循环、hidden 缓存、导出与延迟测量。

| 新增文件 | 作用 |
| --- | --- |
| `touhou_rl/v430/models/__init__.py` | 包入口，只导出 actor 相关符号 |
| `touhou_rl/v430/models/actor.py` | `ActorPolicy`、`HeadLogits`、`parameter_report()`、四个 logits 纯函数、`ModelError` 族 |
| `tests/test_v430_models.py` | 20 条合成测试（14 个测试函数，其中一个参数化 7 路） |
| `Document/Update/Alpha/v4.3.0/evidence/W03/EXIT_CODES_W03D.txt` | 三条验收命令的退出码与统计 |
| 本文件 | 本切片证据 |

**未改动**（只读复用）：`touhou_rl/v430/{contracts,observation,actions,sequence,session,evaluate}.py`、
`touhou_rl/v430/data/**`、`touhou_rl/v400|v420/**`、`tools/rl/**`、`configs/**`、
其它 `tests/test_v430_*.py`、`evidence/{W00,W01,W02,W09}/**`、`evidence/W03/W03A.md`、`W03B.md`、
`registries/**`、`ASSET_SYNC.*`、`PLAN_SPEC.json`（参数量范围未被修改，见 §4）。

## 2. 架构（逐条对齐 §4.3 表格）

网络**从** `MODEL_SPEC`（`PLAN_SPEC.model` 的逐字镜像）构造，因此不存在「另一份抄写出来的常量」。
`padding = kernel_size // 2`，每层 `Conv2d(bias=True)` → `GroupNorm(4 组)` → `SiLU`；投影与融合
线性层同样带 bias；GRU 保留标准两个 bias。

| 流 | 卷积序列 | 进池化前特征图 | 自适应池化 | flatten | 投影 |
| --- | --- | --- | --- | --- | --- |
| global | `3→16/5/2`，`16→32/3/2`，`32→32/3/2` | `30×40` | **6×8** | 1536 | → 128 |
| play | `4→32/5/2`，`32→64/3/2`，`64→64/3/2` | `28×24` | **14×12** | 10752 | → 128 |
| HUD | `1→16/3/1`，`16→32/3/2`，`32→32/3/2` | `112×40` | **28×10** | 8960 | → 64 |
| history | 65 = 56 按键位 + 8 有效位 + 1 采集间隔 | — | — | 65 | → 32 + SiLU |
| 融合 | 352 → 256 线性 → LayerNorm(256) → SiLU | | | | |
| 记忆 | 单层 `GRU(256, 256, bias=True)` | | | | |
| 动作头 | 256 → 9 / 2 / 2 / 2（四个独立线性） | | | | |

池化保留**空间格**再 `flatten(start_dim=1)`，没有一开始就做全局平均。构造时用解析公式
（`(H + 2p - k)//s + 1`）算出期望的进池化特征图尺寸，前向时再断言一次，所以卷积栈尺寸漂移
会立刻报错而不是被自适应池化悄悄吸收。

`/255.0` **只在** `ActorPolicy.forward` 内对三个图像流做一次。键位（0/1）、有效位（bool）、
采集间隔（float32 秒）**不除**。模型外传入 float 图像会被拒绝（说明它已经在别处被归一化过）。

## 3. 接口

| 符号 | 签名 / 说明 |
| --- | --- |
| `ActorPolicy(nn.Module)` | `forward(obs, hidden=None) -> (HeadLogits, hidden_next)` |
| `HeadLogits` | `move/focus/shoot/bomb` 四头**原始 logits** 的具名 frozen dataclass，`as_dict()` / `__getitem__` / `batched` / `batch_size` / `detach()` / `tolist()` |
| `joint_log_prob(head_logits, actions)` | 四头 log-prob **相加**（模块级函数，同名方法转发到同一实现） |
| `joint_entropy(head_logits)` | 四头熵**相加** |
| `sample(head_logits, *, generator=None, temperature=1.0) -> (JointAction, log_prob)` | 单条返回 `JointAction` + 0 维张量；批返回 `tuple[JointAction, ...]` + `(B,)` |
| `ActorPolicy.evaluate(obs, *, hidden=None, ids=None) -> Evaluation` | PPO 重算路径：无采样、无 RNG；给 `ids` 时同时给出 joint log-prob / entropy |
| `parameter_report()` | 各流 / 各分支 / 各头参数量与总数，并声明只含 actor |
| `build_actor()` | 唯一构造入口，**不接受**任何架构参数（传入即报错） |
| `ModelError` | 基类（`contracts.ContractError` 子类）；子类 `PrivilegedModelInputError` / `ShapeMismatchError` / `HiddenStateError` |

返回形状：单条观测 → 四头 `(9,) (2,) (2,) (2)`；批 → `(B,9) (B,2) (B,2) (B,2)`。
GRU 状态始终保留层轴：`(1, B, 256)`，单条也是 `(1, 1, 256)`。

## 4. 实际参数量（`sum(p.numel())`，只含 actor）

| 部分 | 参数量 |
| --- | ---: |
| global 流 | 212,000 |
| play 流 | 1,435,360 |
| HUD 流 | 587,712 |
| history 投影（65→32） | 2,112 |
| 融合（352→256 + LayerNorm） | 90,880 |
| GRU(256, 256, 1 层) | 394,752 |
| 四个动作头（9/2/2/2） | 3,855 |
| **合计** | **2,726,671（2.726671 M）** |

- 落在 `PLAN_SPEC.model.parameter_range_millions = [1, 3]` 内 → **是**（`within_parameter_range: true`）。
- 与 §4.3「约 2.73M」对照：`2,726,671` 距 `2,730,000` 约 **−0.12%**，测试断言 `rel=0.02`。
- `PLAN_SPEC` 的参数量范围**未被改动**，`parameter_range_is_gate` 仍为 `false`（本片不把它当门）。
- 报告显式含 `scope="actor"`、`includes_critic=false`、`includes_auxiliary_heads=false`、
  `auxiliary_heads_not_included=[player, visible_entities, motion_and_danger, visible_resources,
  tactical_goal, future_outcome]`，并且报告各部分之和必须等于对模块的独立重数，否则抛错
  —— 一个加不起来的报告比没有报告更糟。
- 参数量最大的两项是 play 流投影（1,376,384）与 global 流投影（196,736）：§4.3 要求保留
  空间格再投影，投影矩阵自然是大头。这与 HUD 细字/小弹被池化损伤的担忧无关，本片不据此扩网。

## 5. 六条硬约束的落实方式

| # | 约束 | 落实 |
| --- | --- | --- |
| 1 | 不做任何自动 OR / SHOOT | `shoot` 是普通二分类头；`contracts.forced_shoot()` / `truth_action_mask()` 若变 `True`，`forward`/`sample` 直接拒绝运行；采样与 log-prob 不改写任何头 |
| 2 | 不接受特权输入 | `forward` 只接受 `ActorObservation` / `O2Observation` / 六字段映射 / 等价张量束 + `hidden`；复用 `observation.assert_actor_inputs_clean` 哨兵；未知字段与未知关键字都报错 |
| 3 | 形状不符必须报错 | 每个字段必须**恰好**是 `O2_SHAPES` 形状或 `(B,)+O2_SHAPES`；不插值、不 pad、不 reshape、不 squeeze |
| 4 | `hidden=None` 零初始化；状态形状显式断言 | `hidden is None` → `torch.zeros(1, B, 256)`；否则必须逐位匹配 `(num_layers=1, B, 256)`、float32、有限、同 device |
| 5 | 只用 GroupNorm / LayerNorm | 卷积栈 `GroupNorm(4, C)`×9 + 融合 `LayerNorm(256)`；测试对模块树、buffers、AST 三路断言无 BatchNorm |
| 6 | 采样可用 `torch.Generator` 复现 | `sample(..., generator=g)` 全程只用传入的 generator，不触碰全局 RNG（有测试断言） |

一条**比 §4.3 更严**的刻意决定：`ActorObservation.memory` 只做形状/dtype/有限性校验，
**不**被当作 GRU 状态使用。递推状态必须由调用方显式传入 `hidden`，这样任何教师/真值填出的
`memory` 都无法操纵递推。已在 `actor.py` 模块 docstring 中写明。

## 6. 12 条要求 → 测试对号

| # | 要求 | 测试 |
| --- | --- | --- |
| 1 | 形状（单条 / 批） | `test_head_logit_shapes_unbatched_and_batched` |
| 2 | 参数量在 `[1,3]M`、打印实际值、声明只含 actor | `test_parameter_report_total_within_plan_range` |
| 3 | `joint_log_prob` = 四头**相加**（≠ 乘积、≠ 平均） | `test_joint_log_prob_is_the_sum_of_four_heads` |
| 4 | `joint_entropy` = 四头**相加** | `test_joint_entropy_is_the_sum_of_four_heads` |
| 5 | 72 组合概率和 ≈ 1 | `test_72_joint_probabilities_normalise` |
| 6 | 零更新确定性、`eval()` 无 dropout 等随机 | `test_eval_mode_is_deterministic_and_has_no_dropout` |
| 7 | 无 BatchNorm（遍历 `modules()`） | `test_no_batch_norm_anywhere` |
| 8 | 特权输入被拒 | `test_privileged_input_is_refused[stage_id/seed/engine_frame/teacher_action/resource_truth/player_center/enemy_table]` + `test_obs_must_be_an_observation_record` |
| 9 | 形状错误被拒（`(3,240,319)` 不自动 pad） | `test_wrong_image_shape_is_refused` |
| 10 | 采样可复现 | `test_sampling_is_reproducible_with_generator` + `test_sampling_does_not_need_global_rng` |
| 11 | hidden 传递 | `test_hidden_state_zero_init_and_passthrough` |
| 12 | 文件头注明「合成测试，不是游戏/能力证据」 | `tests/test_v430_models.py` 模块 docstring 首段 |
| 附 | 模型层的「无自动 OR SHOOT」 | `test_shoot_head_is_independent_and_never_forced` |

## 7. 三条关键断言的具体做法

**无 BatchNorm**（三路）：

1. 运行时遍历：`[(name, type(m).__name__) for name, m in policy.named_modules()` 若
   `isinstance(m, nn.BatchNorm1d/2d/3d/SyncBatchNorm)]` → 断言为空；
2. 机制层：断言没有任何 `named_buffers()` 名字含 `running_`，也没有 `track_running_stats=True`
   的归一化层；归一化层种类集合必须**恰好**等于 `{"GroupNorm", "LayerNorm"}`，且三条流的
   `GroupNorm.num_groups` 全为 `PLAN_SPEC` 的 `4`；
3. 源码层：`ast.parse` 后扫描 `ast.Attribute` / `ast.Name`，断言 `actor.py` 根本不*引用*
   `BatchNorm1d/2d/3d/SyncBatchNorm`（检测用结构化的 `running_*` buffer 判据，不靠类名），
   因此这条属性不是"测试时恰好没有"，而是"代码里不可能出现"。
   另外断言 `parameter_report()["has_mutable_batch_statistics"] is False` 且
   `["batch_norm_modules"] == []`。

**无自动 OR SHOOT**：

1. `contracts.forced_shoot() is False` 且 `contracts.truth_action_mask() is False`，
   并调用 `actor.assert_no_forced_shoot()` 通过；
2. 手算：`shoot` logits 取 `[0, ln3]`，`log_softmax` 给出 `shoot=0 → -ln4`，实测
   `joint_log_prob` 与 `-(ln9+ln2+ln4+ln2)` 在 `rel=1e-6` 内相等 → 没有任何 bias/mask 被偷偷加到该头；
   且 `exp(logp(shoot=1))/exp(logp(shoot=0)) == 3`（两个分支是不同的动作、不同的概率；若被 OR 上，两者必须相等）；
3. 采样层：`shoot=[20,-20]` 连采 64 次，`set(draws) == {0}`；镜像 `[-20,20]` 连采 64 次
   `set(draws) == {1}` —— 两端都不被"帮忙"；
4. 采样与 `joint_log_prob` / `joint_entropy` 前后逐位比较四头 logits，断言**未被就地修改**；
5. 结构层：`policy.heads` 恰为 `["bomb","focus","move","shoot"]`，各一个 `nn.Linear`，
   `shoot` 头 `out_features == 2`、`weight.shape == (2,256)`、`bias is not None`。

**特权输入被拒**（7 个名字参数化）：把 `{"stage_id": 3}` 之类塞进六字段映射 → 期望
`PrivilegedModelInputError`，并断言异常消息里出现该字段名、且异常同时是 `ModelError` 与
`contracts.ContractError`；同一名字作为**关键字参数**传入（`policy(obs, stage_id=3)`）同样被拒；
把名字藏进嵌套 dict（`{"extra": {"inner": {"stage_id": 3}}}`）同样被拒（递归哨兵）；
无特权含义的未知字段（`harmless_typo`）、未知关键字（`temperature=`）、以及
`build_actor(hidden_size=512)` 也被拒 —— 输入面是封闭的，不是黑名单。

## 8. 「72 组合概率和为 1」实测值

```text
SUM OVER 72 JOINT PROBABILITIES = 0.9999998848553472
```

`|sum − 1| = 1.151e-7`，在要求的 `1e-6` 容差内；同一 seed 下重跑取到同一数值。
测试同时对 72 个组合各断言 `log_prob` 有限且 `<= 0`，并额外用一次批式 `log_softmax`
重算，得到同量级的总和（容差 `1e-5`，float32 累加差异）。

## 9. 验收命令与退出码

见 `EXIT_CODES_W03D.txt`。摘要：

| 命令 | 退出码 | 统计 |
| --- | ---: | --- |
| `python -m compileall -q touhou_rl/v430/models` | 0 | 无输出 |
| `python -m pytest ... -q tests/test_v430_models.py` | 0 | **20 passed** |
| `python -m pytest ... -q tests/test_v430_contracts.py tests/test_v430_observation.py tests/test_v430_sequence.py` | 0 | **214 passed** |

## 10. 刻意不做的部分与 TODO

`TODO(W03C)` 清单（全部在本切片范围之外，未实现、未声称完成）：

- critic / 价值头与六个辅助头（`PLAN_SPEC.model.auxiliary_heads`）—— 本片报告只声明未包含；
- 优化器、PPO 损失、rollout 循环、hidden 状态缓存与 checkpoint 兼容；
- 参数量 / 显存 / 延迟的实机测量（§4.3 要求 critic 与辅助头**单列**这些指标）；
- §4.3 提到的 HUD 细字/小弹池化消融（需真实数据，本片无数据腿）；
- 导出时剔除辅助头并验证动作分布一致的流程。

本切片**不构成**任何证据表明该网络能玩游戏：没有任何引擎交互、没有真实帧、没有训练步、
没有 checkpoint。`compileall` 退出 0 只说明语法可解析。
