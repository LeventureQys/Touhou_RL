# P3 — synthetic rendered diagnostic round10（不是正式 P3 / T21）

2026-11-01。本轮仅在已知 F1/F2/F3 toy world 中制作真实栅格像素诊断数据。**不是游戏采集，不是正式训练授权，不开启、不声明通过任何 formal gate。** 主 agent 提特征和小规模学习另见 [P4 synthetic learning](<P4-synthetic-learning-round10.md>)；本报告只负责 renderer、拆分、完整轨迹与数据接口。

## 1. 预注册顺序与源冻结

读取 [R9 修订报告](<P2-revision-round9.md>)、[teacher V2](<../../../../../touhou_rl/tactical_teacher_v2.py>)、[ValidationWorld](<../../../../../touhou_rl/tactical_teacher_validation.py>) 与 [vision](<../../../../../touhou_rl/vision.py>) 后，新增 [renderer / CLI](<../../../../../touhou_rl/tactical_synthetic_pixels.py>) 及 [6 项合同测试](<../../../../../touhou_rl/tests/test_tactical_synthetic_pixels.py>)。测试先完成，再独立执行 `preregister` 写入 [preregistration](<../evidence/synthetic-learning-round10/preregistration.json>)，最后执行 `collect` 生成此前不存在的 `data` 目录；采集时重新核对源码哈希，拒绝覆盖旧目录，未在结果后调整 renderer 或 split。

| 冻结源 | SHA256 |
|---|---|
| renderer | `fde364c7d0da188023b60c540202c86f7d778fcde8f206d863df114bad60a09d` |
| teacher V2 | `47d39e1c02c32865ed5db46befdc1ab22d7178ded86eb1333409930ffd648e95` |
| 旧 teacher | `c23c9b91ba8c692265f4171dd430d08c508d5884f26370581943727680d0ad73` |
| validation/world/reference | `86608f2774553caa473e6d5d0e35c44fe42e12064e5ec6ebe1d3e4913bf3d92b` |
| scenarios | `471f73debcbb58ee3f184948b2368c6e453a59898c7286e68b29b9a863d84906` |

teacher V2 及旧 world 源码均不改。没有构造/执行封存 R8 test，也没有读取或执行 R9 新确认实例。只从 R8 已暴露 18 development + 24 validation 中按预注册取 16 例；这里的 validation **只是已暴露数据中的开发诊断 split，不是独立泛化确认**。

## 2. 分组与完整路线

| 本轮 split | 原始 template_group | 镜像实例数 |
|---|---|---:|
| train | development-F1-structure-00 / 01 | 4 |
| train | development-F2-structure-00 | 2 |
| train | development-F3-structure-00 | 2 |
| validation | validation-F1-structure-00 | 2 |
| validation | validation-F2-structure-00 / 01 | 4 |
| validation | validation-F3-structure-00 | 2 |

合计 8 train + 8 validation，8 个 template_group，镜像严格同 split，不作为独立组。每例 teacher 从初始状态完整执行 160 逻辑帧、80 次 2-frame action，未遇碰撞提前截断或只取短片段。每例结束由独立 `verify` 完整 replay 检查；**16/16 feasible**。本轮 [failure 列表](<../evidence/synthetic-learning-round10/data/failures.json>) 为空，但生成器遇失败仍保留失败事件、raw references，`action_valid=False`、`action_positive` 全零，不能当成功监督。

## 3. 栅格 renderer 与不可学习边界

PIL 原生 RGB 640×480，playfield 放在现有 ROI `(32,16,384,448)`，自机真实 `state.x` 与固定真实 y=400，出生后子弹按 `born/x/y/vx/vy` 推进到当前 state.frame。当前已生效的 F1 墙/F3 危险区域直接渲染几何；未发生的出生、不生效的危险区不画。背景和 HUD 在所有事件与时刻固定，无 hit/death 闪烁、种子/族/事件编码、分数变化或时钟。

**绝不渲染 teacher future target、option、倒计时、privileged labels、未来 bullet trajectory 或未来 wall coverage。** F1 在墙形成前没有任何可见墙预告；相同当前像素和因果历史可能对应不同 coverage / onset / gap，因此 privileged teacher 先行逃逸动作并非完全由该观测可辨识。F3 未来 window 与 goal 也不可见，危险区开关无预告。即使后续损失下降，也不能推论这些隐变量变得可见或策略具有完整状态最优性。本轮保留这一不可学习/部分可观察限制，**不为可学性加入 future hint**。

原生 raw 每例存 161 张帧：初始 frame0 加 frame1..160，覆盖全部160次逻辑推进。80个监督点为 frame0,2,...,158；80步完整动力学直接使用原 world.advance。奇数中间帧恢复精确 x、已出生 bullets、emissions 供 RGB 渲染；其 `hit` 字段继承下一偶数状态，**不是精确逐帧碰撞标签，既不渲染也不监督**。完整成功判定仅以原 world 的逐逻辑帧 collision 检查及完整 verify 为准。

## 4. 现有 vision 与 history 接口

统一采用现有 `vision.extract_regions`、`resize_area(Image.BOX)` 与 `regions_to_batch`。每个region输入为 float32 `[80,3,H,W,3]`、范围0..1：playfield 224×192、hud_score 34×216、hud_status 126×216。history depth3 为 `[max(0,f-4),max(0,f-2),f]`，缺失历史重复初始帧，只读过去与当前、不跨事件。固定 HUD 以 expanded shared tensor 存储，减少重复存储，不改变读取形状。主 agent 自行提取 backbone features，renderer 不提取也不训练。

teacher水平动作与 actual 候选 `focus*9 + move` 按现有 spec：

| teacher action | 实际 move / focus | candidate |
|---:|---|---:|
| -2 | W=7 / 不聚焦，4 px/frame | 7 |
| -1 | W=7 / 聚焦，2 px/frame | 16 |
| 0 | idle=0 / 不聚焦 | 0 |
| +1 | E=3 / 聚焦，2 px/frame | 12 |
| +2 | E=3 / 不聚焦，4 px/frame | 3 |

hold canonical candidate 选 0（不是 9）。每步 history 单独提供：`previous_candidate` long，首步 BOS18、随后实际上一已执行候选；`previous_hold` / `frame_delta` float32 首步0、之后2；`previous_option` 全程 BOS3，**不使用 teacher option 作为历史**；`previous_target` 零、`target_valid=False`；`history_valid` 首步False其余True。

labels 只有 `action_positive` bool `[80,18]` 与 `action_valid` bool `[80]`。每个正例仅标记该完整可靠 teacher 路线在当前 state 的一个首动作，**不是完整 viable action set，不表示其它候选不可行或不安全**。没有 option/target/slack 额外监督。

## 5. 文件合同与审计

[records](<../evidence/synthetic-learning-round10/data/records.json>) schema 严格 `v373-synthetic-rendered-diagnostic/1`，显式 `formal_manifest_compatible=False`、`real_pixel_training=False`、`formal_gate=NOT_OPENED`；不能作为正式 manifest 消费。每条 events 包含 `event_id/group/split/family/mirror/original_split/rows/teacher_verified/verify_verdict`，`pixels/history/labels/raw/rollout` 均含相对 data 根目录 path 与 SHA256，并含 scenario SHA。全局含 teacher 源SHA、renderer SHA、预注册 reference 与限制。

每例 `pixels.pt / history.pt / labels.pt` 独立；`raw.npz` 存原生 RGB + 逻辑帧索引；`rollout.json` 保留 scenario、完整 actions、teacher decisions 和161个state。raw/reference 属于 privileged 审计材料，**训练接口只读 pixels/history/labels，不把 raw metadata 喂网络**。

[独立完整性检查脚本](<../evidence/synthetic-learning-round10/integrity_check.py>) 对所有资产 SHA、每个偶数完整状态的真实 RGB、16个完整路线 verify、抽样因果堆叠与 dtype/range、action/history 契约进行核验；结果见 [integrity audit](<../evidence/synthetic-learning-round10/integrity-audit.json>)。它不 rerun teacher，不执行其它世界，不训练。

## 6. 执行命令与结论

```powershell
python -m pytest touhou_rl/tests/test_tactical_synthetic_pixels.py -q -p no:cacheprovider --basetemp .tmp_synthetic_pixels_round10
python -m touhou_rl.tactical_synthetic_pixels preregister --out Document/Update/Alpha/v3.7.3/evidence/synthetic-learning-round10
python -m touhou_rl.tactical_synthetic_pixels collect --out Document/Update/Alpha/v3.7.3/evidence/synthetic-learning-round10
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python Document/Update/Alpha/v3.7.3/evidence/synthetic-learning-round10/integrity_check.py
```

renderer合同 **6 passed in1.07s**。preregister/collect exit0，16个完整可行路线已生成，不是正式 gate passed。只证明诊断数据在指定 toy 几何下可重放、没有刻意未来渲染泄漏；不能证明真实截图识别、闭环改善、独立泛化、正式 P3 或 T21。主线学习指标必须与这些边界一并报告。
