# 独立像素确认采集与审计加强（续轮6）

## 冻结与采样顺序

1. 在capture-round6不存在、未采集图像前，保存[capture预登记](<../evidence/capture-round6-prereg.md>)。
2. 定位器子任务保存[bank freeze](<../evidence/pixel-bank-round6/freeze.json>)与源码SHA后通知主线，主线才启动独立确认采集。模板仅旧capture-round5前60帧固定时点，不按定位成败筛。
3. 实际执行：

```powershell
python -m tools.rl.v373_capture_probe --confirm-foreground --protocol confirmation --out Document/Update/Alpha/v3.7.3/evidence/capture-round6
```

退出码0。289截图、122逻辑counter、250同frame括号，使用新非周期动作schedule（完整写入report）。未death/前台失去中断，游戏已关闭、按键已释放。新数据不与开发片段混成独立test。

[确认采样报告](<../evidence/capture-round6/report.json>)、[trace](<../evidence/capture-round6/trace.json>)。对这些像素的冻结bank诊断由单独报告记录，不能用同frame或帧数直接认证同步。

## 数据重复检查

强化[数据审计](<../../../../../touhou_rl/tactical_data.py>)：即便不同event_id/template_group，完全相同像素asset SHA跨split也拒绝。该检查防止通过改名绕过重复检查；不宣称能识别同录像不同压缩/裁剪或近重复，后续须来源分组与内容诊断。

```powershell
python -m pytest touhou_rl/tests/test_tactical_data.py touhou_rl/tests/test_tactical_sequence_train.py -q -p no:cacheprovider --basetemp .tmp/v373-duplicate-round6-01
```

退出码0，28passed in2.93s。新增断言复用已有fixture覆盖跨split同字节资产，未降低asset SHA校验。

GPU仍可用，约1GiB桌面显存占用。新批量缓存构建器将默认FP32、batch16/32，只有gate+audit通过才允许生产调用。本轮不绕过门禁生成真实生产cache或训练标签。
