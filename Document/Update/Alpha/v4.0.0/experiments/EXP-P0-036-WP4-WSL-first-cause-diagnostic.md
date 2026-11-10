# EXP-P0-036 — WP4 WSL 16×30 s first-cause diagnostic

状态：进行 30 秒短门诊断；**不等同于 10 分钟验收**。

## 目的

针对 EXP032/035 中 shadow 实例 `exit 134/1` 与 `pipe closed`，在不改 C++、protocol、依赖或第三方的前提下，优先收集早死实例 stderr，并消除共享 `score.dat`/配置写冲突：每个实例使用唯一 WSL 数据目录；源目录不修改，`.dat` 文件复制后设为只读，配置与 score 各自独立。

## 变更

- `tools/rl/v400_wsl_vec_stress.py`：每实例独立临时 data-dir；stderr 后台有界 drain（末尾 16 KiB）；报告 PID、exit code、stderr tail、清理退出码和完整结果。
- `tests/test_v400_wsl_vec_stress.py`：既有 600 s 长门测试保持为长门，不把 30 s 结果冒充 600 s。

## 30 秒验收命令

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python tools\rl\v400_wsl_vec_stress.py --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-dir /root/v400-data/game --instances 16 --seconds 30 --out .tmp_v400_wp4_diag_036_30.json
```

记录要求：保存命令退出码；`.tmp_v400_wp4_diag_036_30.json` 中记录每实例 PID、exit code、首次捕获到的 bounded stderr tail、错误、请求时长、清理退出码；原始长门仍未运行。

## 判读

若 stderr 含 abort/assert/score 或数据文件写入冲突，第一因优先归因于共享可写 data-dir；若 stderr 为空且 pipe closed，第一因仍待更底层证据，不得声称已定位。短门失败不构成 10 分钟通过。
