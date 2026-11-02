# P4 cache contract — round 7

日期：2026-11-01。**定点修复缓存计算协议，不运行正式 cache / 正式训练，不改 audit 模块，不放宽严格容差。**

## 输入与故障

已读 [builder](<../../../../../touhou_rl/tactical_feature_cache.py>)、[sequence](<../../../../../touhou_rl/tactical_sequence_train.py>)、[round6 cache 报告](<P4-feature-cache-round6.md>)、[round6确认报告](<P0-confirmation-round6.md>)、[round6 pixel bank 报告](<P0-pixel-bank-round6.md>) 与 [真实numeric证据](<../evidence/cache-numeric-round7.json>)。真实 v3-st GPU batch16/32 对 CPU 逐帧的 feature 最大绝对差 `8.32974910736084e-6`，原 `atol=1e-6, rtol=1e-5` 不通过。这不是允许放宽容差的理由；原实现把不同计算协议当作同一来源复算才是问题。

主线随后提供的 [same-protocol repeat证据](<../evidence/cache-numeric-round7-repeat.json>) 已读取：同 device+batch 的两次计算，CPU16 / CUDA1 / CUDA16 / CUDA32 features/logits 最大绝对差均为0；跨 GPU/CPU 原严格失败保留。这只是32行 captured pixels 的 numeric 诊断，不是生产cache、像素同步或训练资格认证。

## 修复

- [builder](<../../../../../touhou_rl/tactical_feature_cache.py>) 与 [sequence](<../../../../../touhou_rl/tactical_sequence_train.py>) 仅接受 finite `float32` 预处理 region `[T,depth,H,W,C]`。删除 uint8 虚假支持；uint8/float64 单测拒绝且不发布index。
- cache schema 明确升级 `v373-event-feature-cache/2`，report schema 升级 `v373-feature-cache-build/2`。**CPU旧 /1 index 同样拒绝；没有缺字段兼容、自动迁移或伪协议默认值。**
- `compute_protocol` 必须存在于 index 和已SHA绑定的report且完全一致，包含 canonical device（例如`cuda:0`）、device_name、batch16/32、float32、autocast=false、tf32=false、torch_version、feature_source 与六个现有相关模块的source SHA。与当前源码/运行torch/设备不一致拒绝；report顶层计算字段与协议交叉验证。
- prepare 在加载backbone前解析并验证report路径/SHA和协议；按已绑定同device+batch（含相同tail划分）复算全部train/validation features和logits，policy_head检查也在相同device+batch完成。所有 allclose 仍为原 `atol=1e-6, rtol=1e-5`。CUDA不可用明确报错，无CPU回退。
- 复算后 features/logits 转为正常CPU tensors供训练，不把 inference tensors泄漏给梯度路径；返回backbone也转CPU，训练自行选择head训练设备。
- FP32/no-autocast/noTF32 helper置于sequence本地，builder单向导入它及protocol/pixel helper，sequence不导入builder，因此无circular import。成功/异常均恢复调用者TF32及autocast上下文。
- 生产没有bypass；gate/audit原顺序保留，不加载test tensors、不创建test cache。没有修改auditor。

## 测试

更新 [builder测试](<../../../../../touhou_rl/tests/test_tactical_feature_cache.py>) 和 [sequence测试](<../../../../../touhou_rl/tests/test_tactical_sequence_train.py>) fixtures为完整 /2 index+hashed report协议，不保留旧索引伪兼容。

```text
python -m pytest touhou_rl/tests/test_tactical_feature_cache.py touhou_rl/tests/test_tactical_sequence_train.py -q -p no:cacheprovider --basetemp .tmp_cache_contract_round7_a
37 passed in 3.43s

python -m pytest touhou_rl/tests/test_tactical_feature_cache.py touhou_rl/tests/test_tactical_sequence_train.py -q -p no:cacheprovider --basetemp .tmp_cache_contract_round7_final
50 passed in 3.77s
```

两个basetemp均全新。最终50项包含真实可用CUDA上的synthetic batch32 build+prepare同批次调用、CPU16/32+tail、协议缺失/旧schema/report缺失/hash损坏/report协议漂移/源码漂移/精度/device_name/torch版本/batch变更拒绝、CUDA不可用无回退、prepare成功与前向异常flags恢复、输出CPU普通tensor及synthetic一步head训练回归。生产gate仅测试fixture内monkeypatch，未在产品实现加绕门入口。

## 边界

没有运行正式cache或正式训练，没有修改numeric CLI（由主线独立负责）。协议绑定降低混用风险，但device_name+torch版本不宣称完整硬件/driver/cuDNN重现认证；若相同协议仍有数值漂移，严格检查继续拒绝，不放宽容差。source SHA包含整个相关模块，后续修改这些模块意味着需要重新构建cache，不能手工升级旧index掩盖来源变化。正式真实像素gate仍未因此获批。
