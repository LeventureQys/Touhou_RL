# P4 sequence interface — round 3

日期：2026-11-01。**仅接口与 synthetic CPU 单测；尚未正式训练，无正式 checkpoint，无可部署候选。** 没有运行游戏，没有读取或训练正式训练数据。没有改动既有模块；本轮新增两个 Python 文件及本报告。仓库原有其他工作不属于本轮。

## 计划与实际实现

1. 阅读现有 [tactical_train.py](<../../../../../touhou_rl/tactical_train.py>)、[tactical_policy.py](<../../../../../touhou_rl/tactical_policy.py>)、[tactical_data.py](<../../../../../touhou_rl/tactical_data.py>)，复用门禁和独立多头损失。
2. 新增 [tactical_sequence_train.py](<../../../../../touhou_rl/tactical_sequence_train.py>)：严格前置、SHA/真实像素重算证明、单事件序列及统一预算训练接口。
3. 新增 [test_tactical_sequence_train.py](<../../../../../touhou_rl/tests/test_tactical_sequence_train.py>)：明确命名的 synthetic fixture，仅测试时 monkeypatch gates/backbone，CPU 执行；生产接口没有绕门或 synthetic CLI 参数。

### 严格前置与新 schema

- 必须先调用 `validate_training_gate`、`audit_manifest`；审计不通过不读 backbone、不建 optimizer、不创建输出目录。真实不合格 manifest 的拒绝有未 monkeypatch gate 的单测。
- 新索引 `v373-event-feature-cache/1` 绑定 manifest 与 baseline SHA256，并逐事件记录 cache path/SHA。缓存 `.pt` 只允许 schema、event_id、baseline_sha256、pixel_sha256、history_sha256、feature_source、features、base_logits 八个字段。
- features 为严格 finite `float32[T,F]`，base_logits 为 `float32[T,18]`；前向时增加 B=1 轴。拒绝 float64、错误来源、错误 baseline、错误 logits。缓存不是工程随机 feature 的包装。
- 要求 feature_source 为 `frozen-backbone.forward_core/real_pixels`，pixel/history SHA 必须匹配事件内容寻址资产。所有 train/validation 的每帧 feature 从 pixel region tensor 调用 frozen backbone.forward_core 重算比较；全部 base logits 与 backbone.policy_head(features) 比较。来源摘要记录事件、split、行数、pixel/cache SHA 和 extractor 标识。
- metadata event/template 分组不得跨 split；声明的 split_groups 与实际事件分组必须完全一致。物理资产受 dataset-root containment 与 SHA 检查。
- 输入历史读取独立 SHA 绑定 cache/history 资产，不从 teacher labels 构造。candidate、hold、effective-frame/delta、option、target/valid 均要求与事件 actual execution 证据一致；缺失实际 option/target 证据即拒绝，不回填 teacher 值。

### 训练语义

- 单事件 window：burn-in16（no_grad，hidden.detach），scored seq32。hidden 从每个窗口零状态开始，仅用该事件前16步重建，不跨事件。MVP 不 padding：首16行与不足32的尾段不计 loss，此限制明确保留。
- N0/N1/B1 同一 seed、窗口/采样顺序、epochs 上限30、max_updates、Adam learning rate 和 clip=1。公平的是数据/更新预算，不声称参数量、FLOPs相同。无 AMP 或 GPU 性能声明。
- 所有 train/val 行（含 burn-in/tail）先经过 history/label loss shape/mask 前置校验；每头 mask 独立，action 为正例集合 logsumexp loss，unknown batch 不 backward、不 step。全 unknown 训练不会保存权重。
- 仅 validation loss 选 epoch（当前为有监督窗口 loss 的等权均值，不能当作完整能力评估）。test 不进入 tensor loader、loss、metrics 或选择。为了履行既有 `audit_manifest`，test 的 JSON 元数据与资产 SHA 字节完整性审计允许 IO；这不等于加载 test tensor 或解释 test 标签。
- backbone 全冻结并保持 eval，结束逐 tensor 验证不变；检查 baseline/manifest/index 未变。输出必须新目录；日志、来源/config report、SHA sidecar 使用独立 research schema。
- 接口若未来通过全部前置，只能写 `v373-sequence-research-only/1` head artifact，不调用现有部署 sidecar 保存函数，不符合部署 loader schema。单测中产生的 temporary artifact 是 synthetic research-only 测试输出，不是候选。

## 命令与结果

首次：

```text
python -m pytest touhou_rl/tests/test_tactical_sequence_train.py -q
5 passed, 13 errors (pytest default temporary directory WinError 5; fixtures未运行)
```

原因：宿主已有 pytest temp/cache 目录不可访问，不是训练接口断言失败。未更改 ACL，使用新 workspace test-only basetemp，并关闭 cacheprovider。

完成版验证：

```text
python -m pytest touhou_rl/tests/test_tactical_sequence_train.py -q -p no:cacheprovider --basetemp .tmp_sequence_round3_cpu_verified_20261101
21 passed in 2.85s
python -m touhou_rl.tactical_sequence_train --help
exit 0; 显示 --manifest --cache-index --baseline --out --arm --seed --epochs --max-updates --device
```

测试覆盖：真实不合格 gate、audit拒绝、baseline/source/dtype/logits 拒绝、即使 logits 一致但像素 feature 不一致仍拒绝、test tensors 不加载、N0/N1/B1 统一一更新预算、冻结骨干、SHA sidecar、新目录、unknown不step且不导出权重、独立多头mask和action正例集合、实际history伪造拒绝、split声明漂移、配置上限。

## 尚未完成 / 不作声明

- **没有正式训练。** 没有验证当前真实 P4 数据是否具备全部 prerequisites；未打开生产训练数据，因此没有满足门禁的断言。前置缺失时只能拒绝，不产出正式 checkpoint。
- 未实现 cache-builder、真实图像采集与 freshness/teacher credibility 的独立认证、学习率搜索、完整三臂实验、能力评估、test评估、部署。
- 来源 SHA 与重算可以证明缓存与给定像素/backbone 一致，不能独立证明给定像素确实来自真实运行；仍依赖已有 pixel alignment/teacher/data gates 的可信证据。
- 第一版 B=1（不是原计划 batch4）、全事件 CPU 预检与逐帧像素重算较保守；没有吞吐/显存/正式预算达标结论。validation loss 仅工程诊断，不是行为事件指标，不构成正式 P4 选模结论。B1 使用既有 TacticalPolicy 的 memoryless 分支，不声称等于旧 macro baseline。尚未接统一 CLI，仅独立 module CLI。没有宣称 P4 完整训练完成或能力提升。
