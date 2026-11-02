# 真实基座缓存数值一致性失败（续轮7）

## 实际运行

```powershell
python -m tools.rl.v373_cache_numeric --source Document/Update/Alpha/v3.7.3/evidence/capture-round6/frames.npz --baseline third_party/th08/build/rl/runs/v104-v3-E5-s7/policy.pt --out Document/Update/Alpha/v3.7.3/evidence/cache-numeric-round7.json
```

退出码0。32个真实截图位置作计算诊断，使用3个因果相邻capture，按现有vision ROI/resize/regions_to_batch预处理。相邻capture不保证独立逻辑帧，这只是compute probe，不是训练序列认证。没有标签、优化步或正式cache文件。

## 实测

[原始报告](<../evidence/cache-numeric-round7.json>)绑定截图NPZ与真实基座SHA。

| 比较CPU逐帧FP32 | feature最大绝对差 | logits最大绝对差 | 既定atol1e-6/rtol1e-5 |
|---|---:|---:|---|
| CPU batch16 | 5.96e-7 | 2.38e-7 | 通过 |
| GPU batch1 | 3.64e-6 | 1.85e-6 | 失败 |
| GPU batch16 | 8.33e-6 | 5.84e-6 | 失败 |
| GPU batch32 | 8.33e-6 | 5.72e-6 | 失败 |

FP32且关闭TF32/autocast仍有设备/批量kernel舍入差异。GPU batch32本次短cold probe含transfer/copy约1293行/s，CPU batch16约141行/s；不能与先前纯GPU驻留tensor热身吞吐混为一谈。

## 同协议重复复测

同一脚本追加重复forward后写入全新[repeat报告](<../evidence/cache-numeric-round7-repeat.json>)，没有覆盖第一次失败证据。CPU batch16、GPU batch1/16/32的同device+batch第二次计算features/logits最大差均0，严格allclose均通过；跨设备相对CPU逐帧失败仍存在。这支持计算协议绑定方向，但只限本次32个像素输入，不作跨驱动/硬件/版本普遍逐位一致保证。

## 修复方向

不得为使缓存可读静默放宽严格容差。缓存构建与完整性复算应绑定相同计算协议（设备、批量、精度、源码/依赖），把计算协议写入已hash index/report；正式训练可再把校验过的features搬到训练设备。CPU-only复核需单列跨设备误差规则，不能假设逐位一致。

另外真实v3-st输入本来要求float32 region，旧builder声称支持uint8而未转换应纠正为明确预处理契约。该问题由tiny测试没覆盖真实模型输入检查引出。修复与回归由后续同轮报告记录，不覆盖本次失败证据。
