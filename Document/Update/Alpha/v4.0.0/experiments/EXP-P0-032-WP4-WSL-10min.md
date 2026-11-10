# EXP-P0-032：WP4 Windows→WSL 16实例 10分钟长门

- 日期：2026-11-08（America/Los_Angeles）
- Provider/model：DeepSeek / DeepSeek v4.1 flash
- 目标：真实 Windows→WSL Ubuntu 环境执行 16 实例、600 秒有界长门；不修改协议、第三方或依赖。
- 原始产物：[`.tmp_v400_wp4_long.json`](../../../.tmp_v400_wp4_long.json)

## 能力核查

脚本 `tools/rl/v400_wsl_vec_stress.py` **不提供 `--max-seconds` 参数**；对应参数是 `--seconds`。它的默认值为 600，并且源码约束 `0 < seconds <= 600`，因此本次使用 `--seconds 600` 执行了同等的 600 秒上限请求。脚本会在每个实例的 `finally` 中调用 `close(quit_timeout=3)`，并在结束时执行 Windows `tasklist` 残留检查。

WSL 资源门（执行前已核实）：`wsl.exe` 可用，发行版为 `Ubuntu`，真实 ELF 为 `/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern`，真实数据目录为 `/root/v400-data/game`；历史环境核查为 20 核。

## 实际命令与结果

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python tools/rl/v400_wsl_vec_stress.py --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-dir /root/v400-data/game --instances 16 --seconds 600 --out .tmp_v400_wp4_long.json
```

- 命令退出码：**1**
- 实际墙钟：**58.3528413772583 s**
- 请求上限：600 s；**未达到 600 秒长门**，不是 PASS。
- 16/16 实例均启动并实际推进；每实例约 72,892–75,198 steps/frames。
- 16/16 实例均报告 `HarnessError: harness closed the control pipe`。
- 16/16 子进程退出码均为 **1**。
- `all_instances_reached_deadline=false`、`all_exit_zero=false`、`all_error_free=false`。

逐实例原始 PID、steps、frames、时间、错误和退出码已完整保存在 `.tmp_v400_wp4_long.json`，未重写或伪造结果。

## 残留检查

脚本完成时的原始 `tasklist /FI "IMAGENAME eq wsl.exe" /FO CSV /NH` 仍发现 2 行 `wsl.exe`：PID **28544**（7,980 K）和 PID **23140**（12,788 K），因此原始报告的 `no_wsl_processes=false`。随后独立复核 `tasklist` 已显示 `INFO: No tasks are running which match the specified criteria.`；未发现仍在运行的 `wsl.exe`。一次附加 WSL `ps` 复核因当前命令字符串的 `grep` 在 WSL 内被解析为不接受该管道的程序而退出 **1**，不影响原始脚本结果，也不据此宣称长门成功。

## 结论

**BLOCKED / FAIL-CLOSED：EXP-P0-032 未通过。** 真实执行确已启动并写盘，但在约 58.35 秒时所有实例的控制管道关闭，全部退出码 1，未达到 600 秒；脚本结束瞬时残留检查还报告 2 个 `wsl.exe`，虽随后独立 `tasklist` 已清空。该结果不能冒充 10 分钟门通过；需要后续诊断控制管道提前关闭的真实原因后再重跑。未修改 `tools/rl/v400_wsl_vec_stress.py`、协议、第三方或依赖。
