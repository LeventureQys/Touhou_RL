# P5 DAgger 数据补采 Round13（仅 train synthetic diagnostic）

## 范围与主线接口

独立实现 [tactical_synthetic_dagger.py](../../../../../touhou_rl/tactical_synthetic_dagger.py)，未修改 R10 renderer、R11 closedloop、R12 forensics、teacherV2 或旧 heads。本轮**只生产数据，不训练、不启动游戏、不创建正式标签、不打开真实 gate**。

- 主线合并入口：[records.json](../evidence/synthetic-dagger-round13/records/records.json)。schema 为 `v373-synthetic-rendered-diagnostic/1`。
- `events` 仅 8 个原 train event augmentation entries；像素、history、labels、raw、rollout 的 `ref.path` **相对该 records.json 所在目录**。主线混合原 records 时应分别解析各 records 的基准目录，不可把旧 validation 复制到此目录。
- augmentation event_id 为 `原event_id-dagger-r13`；`parent_event_id` 保留原 ID，`group == parent_group == 原 train template_group`。`augmentation=True`、`collection_checkpoint_sha` 位于 records 与 entries，rollout 亦保留来源。
- 4 个成功救援 entry `teacher_verified=True`；另 4 个无碰撞未尝试救援 entry `teacher_verified=False`，全部 labels invalid。它们保留用于审计，不提供额外自监督。

## 预注册与不可变输入

先写 [preregistration.json](../evidence/synthetic-dagger-round13/preregistration.json)，随后才初始化冻结网络并开始闭环采集。固定 R10 `run-s7-n1-200` 研究 heads，没有 checkpoint 选择或调参：

`85cc2f4ed93c6e513d76ffde559ca61e2c3061afb9522694d9831ee2838d81d7`

原 baseline：`74abde8a996c13fae832913e4e5533bc9490688bc4c0aa73560e695f065d14be`。

只使用 renderer `selected_specs()` 中原 train 8 events（F1=4/F2=2/F3=2，4 template groups）。读取 R10 records **索引**检查原 train group/scenario identity，未加载任何 validation 像素、标签或轨迹；未加载 R11 validation 轨迹或 R12 forensic validation 恢复点。复用 R11/R12 **代码**而非其 val 数据。采集结束所有预注册 source/checkpoint/index identities 均保持不变。

## 采集与监督规则

1. 每个事件从初始状态开始，执行网络自身 N1 `horizontal_masked` 闭环直到 160 logical frames。固定水平动作域 mask，不是 state-dependent shield；没有 fallback。网络的失败轨迹与像素完整保存于 [source-N1 目录](../evidence/synthetic-dagger-round13/source-N1/)，原网络 score 不被 rescue 覆写。
2. 对发生首 collision 的事件，独立重放检验整个原网络轨迹，取 `[reported firstcollision-16, firstcollision)` 内最早可信、未受击的偶数决策状态。只选该起点，不遍历起点择优。
3. 保持该状态之前的网络动作 prefix，teacherV2 从此状态续跑至 horizon。将**完整 prefix + teacher suffix** 从初始状态交给 R12 `independent_replay` 的独立逐逻辑帧物理验证；仅 `feasible` 的 rescue 可生成 suffix 监督。成功、失败 rescue 均写入各 source event 的 rescue JSON；本次 4 次全部成功。
4. prefix `action_valid=False` 且无 positive；失败 rescue/未尝试 rescue 全部 invalid。网络原动作绝不自监督。suffix 仅监督 teacher 的单路线动作，不冒充完整 viable set。
5. 真实混合轨迹渲染深度 3：`[max(0,f-4), max(0,f-2), f]`。raw 保存独立实际逐帧状态对应像素。history 使用**实际执行 candidate**（保留 candidate9 静止与 candidate0 的区别），prefix→suffix 边界亦连续；previous_option 永远 BOS3，target_valid=False，hold/delta=2，首行 BOS。无 teacher target/option/future hint 输入。

## 结果

详细结果：[collection_summary.json](../evidence/synthetic-dagger-round13/collection_summary.json)。8 事件全部采集完成，未扩充预算。

| 原 train 事件 | 原 N1 feasible | 首碰撞帧 | rescue 起点 | rescue 成功 | 有效 suffix rows |
|---|---:|---:|---:|---:|---:|
| F1 structure00 m0 | 是 | 无 | 无 | 未尝试 | 0 |
| F1 structure00 m1 | 是 | 无 | 无 | 未尝试 | 0 |
| F1 structure01 m0 | 是 | 无 | 无 | 未尝试 | 0 |
| F1 structure01 m1 | 是 | 无 | 无 | 未尝试 | 0 |
| F2 structure00 m0 | 否 | 92 | 76 | 是 | 42 |
| F2 structure00 m1 | 否 | 92 | 76 | 是 | 42 |
| F3 structure00 m0 | 否 | 152 | 136 | 是 | 12 |
| F3 structure00 m1 | 否 | 152 | 136 | 是 | 12 |

- 原始无 shield N1：**4/8 feasible**，不能把救援后 4/4 当作网络改进。
- Rescue：4 次尝试，4 次完整独立 replay 成功；本次 logical 首碰撞帧与 2-frame reported collision 相同。
- 新目录：8 entries ×80 rows=640 保存行，**108 有效 teacher suffix rows**；其余 532 行均不监督。
- collection training_steps=0、validation_asset_loads=0、forensic_restore_point_loads=0、game_execution=False、formal_gate=NOT_OPENED。
- 全部采集资产 SHA 清单：[sha256.json](../evidence/synthetic-dagger-round13/sha256.json)。

## 测试

独立基础测试：[test_tactical_synthetic_dagger.py](../../../../../touhou_rl/tests/test_tactical_synthetic_dagger.py)；像素/资产整合测试：[test_tactical_synthetic_dagger_export.py](../../../../../touhou_rl/tests/test_tactical_synthetic_dagger_export.py)。覆盖 train-only 限额与 groups、prefix 不自监督、实际 candidate history/BOS3、最早窗口起点、完整 prefix replay、无碰撞不调用 teacher、失败 rescue 不打标签、预注册身份不可变、R10 字段 refs SHA、actual mixed raw/causal stack。

使用两个从未使用过的 fresh basetemp，禁用 pytest cacheprovider：

```powershell
python -m pytest touhou_rl/tests/test_tactical_synthetic_dagger.py -q --basetemp .tmp_dagger_round13_fresh_a -p no:cacheprovider
# 6 passed in 2.40s
python -m pytest touhou_rl/tests/test_tactical_synthetic_dagger.py touhou_rl/tests/test_tactical_synthetic_dagger_export.py touhou_rl/tests/test_tactical_synthetic_pixels.py touhou_rl/tests/test_tactical_synthetic_closed_loop.py touhou_rl/tests/test_tactical_failure_forensics.py -q --basetemp .tmp_dagger_round13_fresh_b -p no:cacheprovider
# 30 passed in 5.15s
```

## 限制

这是有特权的 toy-world train 增强数据，不是正式 P5、T21、真实游戏或独立测试证据。F1 隐藏 wall onset、F3 隐藏 goal/window 的原 renderer 限制不变。主线之后将混合原 records 做公平训练；本模块没有执行训练或读取 validation 资产，不能据此声称网络改善。
