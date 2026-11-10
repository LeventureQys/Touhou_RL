# EXP-P0-024 — WP7 v4.0.0 random/still S1N 真实 runner 尝试

- **日期**：2026-11-08（America/Los_Angeles）
- **Provider / 模型**：DeepSeek / DeepSeek v4.1 flash
- **目标**：WP7 尝试通过 WSL/真实 binary runner 执行 1 局 random/still S1N；不得以合成数据替代真实 episode，不得伪造 score。
- **结论**：**BLOCKED；不得记为通过，不产生 scoreboard 或分数。**

## 前置 human reference gate

命令：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python tools/rl/v400_bench_run.py --agent random --suite S1N --seed 1000 --human-reference .tmp_v400_wp7_missing_human_reference.json
```

实测结果：

```text
{"status": "blocked", "reason": "human_reference.json is missing; supply a real human reference"}
EXIT=3
```

该命令未创建 reference，未输出 score/episode。由于 `human_reference.json` 缺失，按 fail-closed 规则不能进入 score 计算。

## WSL / binary runner 尝试

仓库工作区限定路径中未发现 Windows 原生 `th08-modern` binary；已尝试 WSL 真实路径：

```powershell
wsl.exe -d Ubuntu -- bash -lc 'if test -x /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern && test -d /root/v400-data/game; then echo WSL_RESOURCE_CHECK_RC=0; timeout 20s /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-dir /root/v400-data/game --rl-control 0,1 </dev/null; echo WSL_BINARY_EXIT=$?; else echo WSL_RESOURCE_CHECK_RC=1; fi'
```

实测输出：

```text
WSL_RESOURCE_CHECK_RC=0
```

该真实 binary 进程在命令超时（40 秒）后仍未返回，命令被 harness 终止；没有取得合法 episode、protocol completion 或 score，因此不能记录 `WSL_BINARY_EXIT`，也不能将其视为成功。该过程已停止，不继续进行无界运行。

## 缺口与解阻条件

- 缺少可用的真实 `human_reference.json`，前置 gate 未通过。
- WSL 资源检查虽为 0，但真实 binary 未在有界窗口内完成可记录的协议/episode。
- 未执行并未声称 random/still S1N 真实得分；未创建 synthetic episode 或 scoreboard。
- 解阻需要提供真实 human reference，并在可复现的有界命令中取得 runner 的协议完成、episode 记录和进程 exit code；随后才能重试 WP7。
