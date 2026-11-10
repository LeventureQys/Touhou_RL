# EXP-P0-038：WP4 isolated 16×30 s 实际运行

- 日期：2026-11-08（America/Los_Angeles）
- Provider / 模型：`DeepSeek` / `DeepSeek v4.1 flash`
- 结论：**PASS（仅本 16×30 s 短门）**

## 范围与命令

本实验验证的是 16 个实例在**请求 30 秒**、每实例独立数据目录条件下的真实连续运行；不是 600 秒门，也不改写或覆盖 EXP-P0-032 的失败结论。

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python tools\rl\v400_wsl_vec_stress.py --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-dir /root/v400-data/game --instances 16 --seconds 30 --out .tmp_v400_wp4_diag_038_30.json
```

- 上述真实运行命令退出码：**0**。
- 原始证据：[`.tmp_v400_wp4_diag_038_30.json`](../../../../.tmp_v400_wp4_diag_038_30.json)
- 执行器及隔离目录准备逻辑：[`v400_wsl_vec_stress.py`](../../../../tools/rl/v400_wsl_vec_stress.py)（每实例 `i-0`…`i-15` 独立目录；`th08.cfg`、`score.dat` 分别复制到实例目录，其余 `.dat` 以只读 symlink 接入）。

## 结果（以原始 JSON 为准）

- `instances=16`、`requested_seconds=30.0`、`continuous=true`。
- 墙钟：`wall_seconds=33.01343536376953`（约 **33.0134 s**）。
- **16/16** 实例达到约 30 秒：`all_instances_reached_deadline=true`；逐实例 `run_seconds` 范围约 **29.9776–30.0037 s**。
- **16/16 exit code=0**：`all_exit_zero=true`。
- **bad=0**：`setup_error=null`，所有实例 `error=null`，`all_error_free=true`；各实例 `steps == frames`。
- 清理退出码：`cleanup_exit_code=0`（**cleanup=0**）。
- 无残留：`remaining_wsl_process_lines=[]`、`no_wsl_processes=true`。
- 数据隔离：`instance_data_root` 为本轮唯一临时根目录；每个实例使用独立 `i-N` 目录，独立可写 `th08.cfg` 与 `score.dat`，其他数据通过 symlink 接入，未共享可写配置/score 文件。

## 门槛边界与 EXP032 关系

- **这不是 600 秒门**：本轮 `requested_seconds=30.0`；JSON 中的 `max_seconds=600` 只是脚本允许的上限，不代表本轮运行 600 秒。因此本记录**不得**表述为 16/16 连续 600 秒通过，也不替代 600 秒验收。
- **不覆盖 EXP-P0-032 失败**：EXP032 记录的是另一轮真实 600 秒运行，命令退出码 **1**，实例10约 4.252 秒以 **134** 退出并出现 `harness closed the control pipe`，且原始报告曾观察到残留；其失败事实和结论保持不变。详见：[EXP-P0-032-WP4-WSL-16-instance-10min.md](./EXP-P0-032-WP4-WSL-16-instance-10min.md)。
- 因此，EXP038 仅证明：在独立 `cfg/score/symlink` 数据布局下，**16×30 秒短门**本轮达到 exit0、近 30 秒、bad0、cleanup0、无残留；不能外推为 600 秒门通过。

## 可复核证据字段

- 原始 JSON：[`.tmp_v400_wp4_diag_038_30.json`](../../../../.tmp_v400_wp4_diag_038_30.json)（字段包括 `wall_seconds`、逐实例 `run_seconds`/`exit_code`/`error`、`cleanup_exit_code`、`remaining_wsl_process_lines`、四个布尔门）。
- 运行实现：[`tools/rl/v400_wsl_vec_stress.py`](../../../../tools/rl/v400_wsl_vec_stress.py)。
- 被明确保留的失败报告：[`EXP-P0-032-WP4-WSL-16-instance-10min.md`](./EXP-P0-032-WP4-WSL-16-instance-10min.md)。
