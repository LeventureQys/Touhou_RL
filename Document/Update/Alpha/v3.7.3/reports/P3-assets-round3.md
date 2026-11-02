# P3因果资产审计强化（续轮3）

当前无正式P3数据，本轮只完善开跑前可靠性，不把工程fixture当采集事件。

## 具体修复

先前只校验事件JSON的SHA，像素/动作历史/标签文件可能悄悄变化。现要求三个资产都以`{path,sha256}`引用，审计实际文件字节哈希；拒绝dataset root之外的资产、输入/标签物理路径别名，以及pixels/history之外的未声明部署通道。事件JSON本身仍须SHA绑定，组间泄漏、未知同步等既有检查保留。

这仍是结构/字节审计，不代表图像新鲜度或教师标签真实性。test资产仅哈希检查属于完整性IO，不加载或解释test数组/结果，不用于checkpoint选择。

## 验证

```powershell
python -m pytest touhou_rl/tests/test_tactical_data.py -q -p no:cacheprovider --basetemp .tmp/v373-assets-round3-01
```

退出码0，7passed in0.08s。覆盖：SHA漂移、模板泄漏、未hash资产、input/label alias、因果历史、未执行动作和未知对齐。

后续序列训练接口还需从绑定像素重新计算features与base_logits，防止缓存伪造；只有data gate声明不足以认证数据，必须结合内容审计。尚不生成正式checkpoint。
