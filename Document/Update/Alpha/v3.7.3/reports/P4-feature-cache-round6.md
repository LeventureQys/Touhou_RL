# P4 feature cache — round 6

日期：2026-11-01。**仅新增缓存构建接口与 synthetic CPU/CUDA 单测；无正式训练、无游戏运行、无真实生产 cache、无能力/真实图像认证声明。**

## 阅读与范围

已读 [sequence trainer](<../../../../../touhou_rl/tactical_sequence_train.py>)、[data auditor](<../../../../../touhou_rl/tactical_data.py>)、[training gate](<../../../../../touhou_rl/tactical_train.py>)、[升级执行计划](<../升级执行计划.md>)、[round3接口报告](<P4-sequence-interface-round3.md>)、[硬件报告](<../evidence/hardware-windows.json>) 与 [环境报告](<../evidence/environment.json>)。硬件记录为 i5-14600KF / 32 GiB RAM / RTX 4070 SUPER，约12 GiB显存；这是既有记录，不是本轮重新完整硬件审计。

本轮新增 [tactical_feature_cache.py](<../../../../../touhou_rl/tactical_feature_cache.py>)、[test_tactical_feature_cache.py](<../../../../../touhou_rl/tests/test_tactical_feature_cache.py>) 和本报告；未修改已有代码。其他agent/工作树的改动不计入本轮。

## 实现

- `build(manifest, baseline, out, device='cpu', batch_size=16)` 及独立 module CLI；仅允许 CPU/CUDA 和 batch16/32；生产无 bypass/synthetic/AMP 选项。
- `validate_training_gate` → `audit_manifest` 且 audit manifest SHA一致 → event与asset实际路径/SHA与分组复核 → baseline SHA/load，最后才创建新输出目录。已有目录拒绝覆盖。失败可能留下partial event文件，但 `index.json` 最后发布；失败重试须另取新目录。
- 读取实际manifest event refs；pixels/history/labels refs首先相对event JSON解析，再规范到manifest根。拒绝越界、重复event、template泄漏、实际/声明split漂移；train/validation至少48行。
- auditor对包括test在内的全部event/asset做字节完整性hash；构建器只对train/validation的hashed pixel mapping执行 `torch.load(..., weights_only=True, map_location='cpu')`。不反序列化test像素、history、labels，也不生成test cache；构建器自身不读取任何split的history/labels数组。
- frozen/eval backbone，`inference_mode` 下按16或32批调用 `forward_core`，保留features与policy_logits；严格finite float32 `[T,F]` / `[T,18]`，并检查policy_head一致。仅接受uint8/float32、5维、行数一致的pixel region mapping。
- 默认且唯一精度为FP32：backbone转float32，显式关闭ambient autocast与CUDA matmul/cuDNN TF32，离开时恢复caller flags。本轮不提供autocast选项，也不凭float32存盘冒充计算精度。
- 每个 `event-NNNNNN.pt` 严格八字段：schema/event_id/baseline_sha256/pixel_sha256/history_sha256/feature_source/features/base_logits，沿用sequence `CACHE_SCHEMA` / `SOURCE`。固定数字文件名避免event_id路径注入。
- index绑定manifest、baseline、SOURCE、逐event文件SHA与report SHA，并保存builder/sequence/gate/auditor/vision loader/v3 backbone源码SHA；完成前复核输入event/全部资产、manifest、baseline、源码未变与backbone state逐tensor不变。
- report保存device/name、torch版本、batch、precision/autocast/TF32、每event耗时/rows、总耗时/吞吐、CUDA peak allocated bytes等。吞吐包含抽取与event序列化，不含前置完整审计/backbone加载及末尾hash复核，不能当作全流程壁钟或真实游戏延迟。

## 测试与实际拒绝

测试fixture明确synthetic，每split独立随机pixel源；**仅测试内monkeypatch生产gate和tiny backbone loader，使用真实audit**，生产没有绕门参数。tiny backbone是4维feature线性head，不是实际v3-st性能代理。

执行：

```text
python -m pytest touhou_rl/tests/test_tactical_feature_cache.py -q -p no:cacheprovider --basetemp .tmp_feature_round6_tests
11 passed in 1.39s  # 初版

python -m touhou_rl.tactical_feature_cache --help
exit 0; --manifest --baseline --out --device --batch-size {16,32}

python -m pytest touhou_rl/tests/test_tactical_feature_cache.py touhou_rl/tests/test_tactical_sequence_train.py -q -p no:cacheprovider --basetemp .tmp_feature_round6_final
36 passed in 3.53s  # 最终：builder 15项 + sequence 21项；包含synthetic CUDA batch32
```

最终覆盖：生产不合格gate先于baseline/output拒绝；audit先行；test pixel字节损坏拒绝且未tensor load；nested event-relative路径；batch16/32及tail；严格八字段/hash/FP32；冻结state；禁止test tensor load；CPU与CUDA构建完成后 `seq.prepare` 可读；已有目录拒绝；非法batch；split漂移；ambient autocast关闭和flags恢复；路径越界；构建中baseline变化不发布index；非法pixel dtype不发布index。

额外直接将真实 [live-calibration-01报告](<../evidence/live-calibration-01/report.json>) 作为manifest传给builder（baseline故意不存在）：

```text
ValueError: training gate rejected: schema must be 'v373-paired-events/1'
```

断言输出目录不存在。不尝试修补/伪造真实校准schema或gate，不读取其pixels为tensor，不创建真实cache。

## 明确保留的限制

1. SOURCE中的 `real_pixels` 是既有schema要求的来源标识；通过gate、hash与backbone重算仅证明所给资产的一致性，不能独立证明真实捕获、渲染新鲜度、教师可信度。本轮report明确 `real_image_source_independently_certified=false`。
2. synthetic CUDA测试验证设备路径与tiny数值兼容，不证明真实v3-st GPU批量结果必然满足sequence逐帧CPU严格allclose容差。sequence仍按原实现CPU逐帧重算；没有放宽容差。后续真实cache必须以同一精度模式通过原prepare，必要时统一精度上下文并另行实测；caller默认TF32设置与CPU/GPU批处理舍入也可能导致严格拒绝。
3. 本轮未实测真实v3-st batch16/32吞吐或显存上限，没有保证任意事件长度的RAM预算。当前一次load单事件完整pixel mapping，batch仅限制设备前向规模；没有streaming tensor文件格式。
4. builder不检查history tensor语义与label mask；这些仍由原sequence prepare与训练前置独立验证。没有训练、optimizer更新、部署checkpoint或test指标。
