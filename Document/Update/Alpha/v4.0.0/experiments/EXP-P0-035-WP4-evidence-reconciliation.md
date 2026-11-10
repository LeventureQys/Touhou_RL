# EXP-P0-035 — WP4 原始 JSON 与代理摘要矛盾复核

- 日期：2026-11-08（America/Los_Angeles）
- Provider / model：DeepSeek / DeepSeek v4.1 flash
- 原始记录位置：仓库根目录 [`EXP-P0-035-WP4-evidence-reconciliation.md`](../../../../../EXP-P0-035-WP4-evidence-reconciliation.md)；本卡整理自该文件，原文件保留不删除。
- 本次归档范围：仅新增规范目录中的文档副本；以下实验、命令及结论均记录原始文件中的既有事实，本次未重跑实验，也未修改代码或 JSON。
- 原始实验范围：只读检查原始 JSON、stress 脚本与 EXP032；新增原始记录和一次 30 秒复现产物。未修改代码、原始长跑 JSON、协议、依赖或第三方。

## 检查对象

- [`.tmp_v400_wp4_long.json`](../../../../../.tmp_v400_wp4_long.json)：当前原始长跑 JSON。
- [`tools/rl/v400_wsl_vec_stress.py`](../../../../../tools/rl/v400_wsl_vec_stress.py)：实际执行脚本。
- [`EXP-P0-032-WP4-WSL-10min.md`](EXP-P0-032-WP4-WSL-10min.md)：与当前长跑 JSON 一致的 EXP032 记录。
- [`EXP-P0-032-WP4-WSL-16-instance-10min.md`](EXP-P0-032-WP4-WSL-16-instance-10min.md)：与当前长跑 JSON 矛盾的另一份同编号记录。

## 原始记录中的准确复现命令与退出码

在 Windows PowerShell、工作目录 `D:\WorkShop\Touhou_RL` 执行：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python tools\rl\v400_wsl_vec_stress.py --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-dir /root/v400-data/game --instances 16 --seconds 30 --out .tmp_v400_wp4_reconcile_30.json
```

原始记录所载命令退出码：**1**。

原始复现产物：[`.tmp_v400_wp4_reconcile_30.json`](../../../../../.tmp_v400_wp4_reconcile_30.json)。这是 30 秒切片，不是 600 秒验收，也不能外推 600 秒结论。

## 30 秒 JSON 摘要（摘自原始记录）

```json
{
  "provider": "DeepSeek",
  "model": "DeepSeek v4.1 flash",
  "instances": 16,
  "requested_seconds": 30.0,
  "wall_seconds": 30.113247632980347,
  "successful_instances": 11,
  "early_pipe_closed_instances": [3, 9, 10, 13, 15],
  "early_pipe_closed_exit_codes": [134, 134, 134, 134, 134],
  "successful_run_seconds": "29.986526012420654..29.997143507003784",
  "successful_steps_equals_frames": true,
  "remaining_wsl_process_lines": [],
  "all_instances_reached_deadline": false,
  "all_exit_zero": false,
  "no_wsl_processes": true,
  "all_error_free": false,
  "process_exit_code": 1
}
```

5/16 实例在约 2.904–4.378 秒提前收到 `HarnessError: harness closed the control pipe`，`steps=0`、`frames=0`，进程退出码 134；11/16 实例约 30 秒完成，`steps == frames`。本次没有残留 `wsl.exe`。

## 原始长跑 JSON 摘要

`.tmp_v400_wp4_long.json` 当前明确写入：

- `requested_seconds=600.0`，但 `wall_seconds=58.3528413772583`；
- 16/16 实例为 `HarnessError: harness closed the control pipe`、`exit_code=1`；
- `all_instances_reached_deadline=false`、`all_exit_zero=false`、`no_wsl_processes=false`、`all_error_free=false`；
- 实例运行时间约 58.205–58.243 秒；
- 因此它不是达到 600 秒的成功长跑。

## 时间语义与是否错写

脚本第 45–51 行将 `--seconds` 解析为浮点数，并设置 `deadline = time.monotonic() + a.seconds`；第 20 行循环至 deadline。复现中 `requested_seconds=30.0`、`wall_seconds≈30.113`，正常实例 `run_seconds≈30`，证明参数语义是实际有界运行秒数，不是被错误解释成 600 秒或立即退出。

脚本第 61–76 行按实际结果写入 `requested_seconds`、`wall_seconds`、逐实例错误/退出码及四个布尔门，并以四门全真才返回 0。30 秒复现和当前长跑 JSON 都按此逻辑返回 1。未观察到脚本把 30 秒结果错写成 600 秒成功；真正的矛盾来自同编号 EXP032 文档摘要与原始 JSON 不一致，而非当前脚本的时间字段语义。

## 矛盾裁决

`EXP-P0-032-WP4-WSL-16-instance-10min.md` 声称 `wall=600.299s`、15/16 成功、实例10 exit 134；这些数字与当前 `.tmp_v400_wp4_long.json`（58.35s、16/16 exit 1）不一致，也没有对应的当前 JSON 证据。应以可复核的原始 JSON 为权威：**EXP032 长门未通过；不能采用 600.299 秒/15 成功摘要，也不得伪造 600 秒结论。**

30 秒复现还显示同类控制管道提前关闭可在多个实例复现，因此“Windows 环境提前退出”已得到有界切片证据；根因仍未定位，本记录不声称已修复。

## 结论

**FAIL-CLOSED / 未通过。** 已确认：Windows→WSL 16 实例运行中存在提前控制管道关闭；`--seconds` 语义正常且数据门控没有把失败写成成功；原始 JSON 与一个同编号代理摘要矛盾时，应采用原始 JSON。没有执行、也没有声称 600 秒重跑成功。
