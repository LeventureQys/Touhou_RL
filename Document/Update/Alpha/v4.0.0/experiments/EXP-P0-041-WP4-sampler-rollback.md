# EXP-P0-041 — WP4 采样器回退与有界 16×10 验证

> 规范副本说明：整理自根目录 [`EXP-P0-041-WP4-sampler-rollback.md`](../../../../../EXP-P0-041-WP4-sampler-rollback.md)；根目录原件保留不删。本次仅新增此文档，不重新执行长门或修改代码。

- Provider / model：DeepSeek / DeepSeek v4.1 flash。
- 范围：未验证的 Linux PID 采样器及其共享字典调用已从 `tools/rl/v400_wsl_vec_stress.py` 回退；保留既有 16 实例压力测试流程。该回退依据原始记录，不在本次纯文档整理中重新验证代码差异。

## 已有验证记录

根目录原件记录的单测命令：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests/test_v400_wsl_vec_stress.py
```

原件所载退出码 `0`，`4 passed in 0.07s`；本次没有重跑。

有界真实运行命令（根目录原件所载）：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python tools/rl/v400_wsl_vec_stress.py --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-dir /root/v400-data/game --instances 16 --seconds 10 --out .tmp_v400_wp4_diag_041_16x10.json
```

原件记录脚本退出码 **0**；原始证据 [`.tmp_v400_wp4_diag_041_16x10.json`](../../../../../.tmp_v400_wp4_diag_041_16x10.json) 记录 `instances=16`、`requested_seconds=10.0`、`wall_seconds=13.027353286743164`（约 **13.027 秒**）。`results` 共 16 条，16/16 的 `error=null`、`exit_code=0`；汇总字段 `all_instances_reached_deadline=true`、`all_exit_zero=true`、`all_error_free=true`。清理字段 `cleanup_exit_code=0`、`remaining_wsl_process_lines=[]`、`no_wsl_processes=true`：所检查的 WSL 进程无残留。

## 证据边界与结论

原始 JSON 的 `linux_samples={}`，`linux_post_cleanup.status=unknown`，原因是 `linux_pid_sampler_disabled_unverified`。因此 **Linux PID 唯一性和 zombie 状态均 unknown**；Windows/启动侧结果里的 `pid` 数值不能冒充 Linux PID，WSL 无残留也不能替代 Linux zombie 的专门验证。

本次有界 16×10 验证通过上述有限检查，**不构成 Linux PID/zombie 验收或 WP4 总体验收**。既有 [EXP-P0-039 16×600 长门记录](EXP-P0-039-WP4-16x600.md)保留其 **PASS-with-limit** 边界：长门可用性结果不因本次回退而撤销，但不能由其推导采样器已验、Linux PID/zombie 已验或 G0 总体验收通过。
