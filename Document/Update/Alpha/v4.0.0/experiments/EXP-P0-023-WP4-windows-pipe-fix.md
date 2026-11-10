# EXP-P0-023-WP4：Windows pipe reader thread + queue 修复记录

- **版本**：v4.0.0
- **日期**：2026-11-08
- **范围**：`touhou_rl/v400/gym/protocol.py`

## 修复内容

前一代理已修改 `touhou_rl/v400/gym/protocol.py`：在 Windows 环境下使用 reader thread + queue 读取协议管道，避免对不适用于该 pipe 类型的 `select` 调用，并规避 `WinError 10093`。

本记录仅用于记录该修复及其当前验收状态；本实验文档不包含任何额外代码变更。

## 已完成的验证

独立的协议、恢复和环境合同测试已运行并通过：

- **12 passed**
- `python -m compileall -q touhou_rl`：**exit 0**

上述结果证明对应的独立测试与 Python 编译检查通过，但不等同于真实 binary/data 运行验收。

## 真实验收状态

真实 binary/data 单环境验收目前仍 **blocked**。截至本记录生成时，尚未获得真实 binary/data 单环境成功运行的可复核结果，因此不宣称真实验收通过，也不伪造成功产物、日志或截图。

后续应在具备所需真实 binary 与 data、并能复现 Windows 运行环境后，重新执行单环境验收，记录原始命令、退出码、日志和产物；在此之前，WP4 的真实验收门保持未通过/阻塞状态。
