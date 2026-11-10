# EXP-P0-032：WP4 Windows→WSL 16环境连续10分钟真实运行

- 日期：2026-11-08（America/Los_Angeles）
- Provider/model：DeepSeek / DeepSeek v4.1 flash
- 允许修改范围：`tools/rl/v400_wsl_vec_stress.py`、`tests/test_v400_wsl_vec_stress.py`、本记录、`.tmp_v400_wp4_long.json`；未改引擎、协议、依赖或第三方。

## 目标与命令

目标是以真实 Windows Python 启动16个 `wsl.exe` launcher，各实例连续执行至严格600秒上限，检查帧连续性、错误、退出码、PID及清理后的 `wsl.exe`。

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python tools\rl\v400_wsl_vec_stress.py --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-dir /root/v400-data/game --instances 16 --seconds 600 --out .tmp_v400_wp4_long.json
```

- 命令退出码：**1**（真实运行，不是资源预检失败）。
- 实际墙钟：**600.299 s**；15个实例达到约600秒，实例10在**4.252 s**退出。
- 原始证据：[`.tmp_v400_wp4_long.json](<../../../.tmp_v400_wp4_long.json>)。

## 逐实例结果摘要

实例0–9、11–15：均连续运行约600秒，`steps == frames`，每实例约990,000–1,004,614帧，错误为空，退出码0。

实例10：PID **19476**，运行 **4.251650810241699 s**，`steps=0`、`frames=0`，错误：`HarnessError: harness closed the control pipe`，退出码 **134**。

因此本轮不满足“16/16连续600秒、全错误为零、全退出码为零”门槛，**不得记为PASS**。

## 清理与残留

压测报告在所有客户端 `close()` 后立即通过 `tasklist` 检查到32行残留 `wsl.exe`。这是失败证据的一部分，见原始 JSON 的 `remaining_wsl_process_lines`；报告字段为 `no_wsl_processes=false`。随后执行：

```powershell
wsl.exe --shutdown
Start-Sleep -Seconds 3
tasklist /FI "IMAGENAME eq wsl.exe" /FO CSV /NH
```

清理复核退出码 **0**，输出 `INFO: No tasks are running which match the specified criteria.`。故人工清理后无残留，但**原始验收报告的清理门仍为 false**，不将其改写为通过。

## 结论

**FAIL / WP4 10分钟门未通过。** 失败原因具体且可追溯：实例10在4.252秒收到 `harness closed the control pipe` 并以134退出；客户端批量关闭后报告仍观察到32个 `wsl.exe` 残留，虽随后 `wsl --shutdown` 清除。15/16实例确实完成约600秒真实运行，不能外推为16/16通过。后续需定位实例10控制管道关闭及 launcher 残留生命周期后重跑；本轮不伪造PASS、不重复无界运行。
