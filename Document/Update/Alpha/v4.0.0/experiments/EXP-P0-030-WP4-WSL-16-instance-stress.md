# EXP-P0-030：WP4 Windows→WSL 16实例有界短压测

- 日期：2026-11-08（America/Los_Angeles）
- Provider/model：DeepSeek / DeepSeek v4.1 flash
- 允许修改范围：`tools/rl/v400_wsl_vec_stress.py`、本记录、`.tmp_v400_wp4_stress.json`；未改 protocol/依赖/第三方。

## 真实资源门

命令：
```powershell
wsl.exe -d Ubuntu -- sh -lc 'printf "WSL_OK\\n"; test -x /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern && echo BIN_OK || echo BIN_MISSING; test -d /root/v400-data/game && echo DATA_OK || echo DATA_MISSING; nproc'
```
退出码 0；输出 `WSL_OK`, `BIN_OK`, `DATA_OK`, `20` 核。

## 压测命令与结果

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python tools/rl/v400_wsl_vec_stress.py --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-dir /root/v400-data/game --instances 16 --seconds 60 --out .tmp_v400_wp4_stress.json
```
退出码 **0**；实际墙钟 **3.879 s**（有界上限 60 s）。16/16 实例均完成 100 STEP，每步 advanced=1，故每实例帧数 100；错误 0；16/16 子进程退出码 0；清理后 `wsl.exe` 进程行数 0（无残留子进程）。

PID / steps / frames / error / exit：

| 实例 | PID | steps | frames | error | exit |
|---:|---:|---:|---:|---|---:|
| 0 | 10308 | 100 | 100 | — | 0 |
| 1 | 33148 | 100 | 100 | — | 0 |
| 2 | 28532 | 100 | 100 | — | 0 |
| 3 | 33304 | 100 | 100 | — | 0 |
| 4 | 31296 | 100 | 100 | — | 0 |
| 5 | 27740 | 100 | 100 | — | 0 |
| 6 | 29520 | 100 | 100 | — | 0 |
| 7 | 27020 | 100 | 100 | — | 0 |
| 8 | 30248 | 100 | 100 | — | 0 |
| 9 | 23080 | 100 | 100 | — | 0 |
| 10 | 33584 | 100 | 100 | — | 0 |
| 11 | 19628 | 100 | 100 | — | 0 |
| 12 | 19508 | 100 | 100 | — | 0 |
| 13 | 29952 | 100 | 100 | — | 0 |
| 14 | 20304 | 100 | 100 | — | 0 |
| 15 | 8288 | 100 | 100 | — | 0 |

**Verdict: PASS（WP4 bounded stress only）**。没有运行10分钟门；本结论不冒充10分钟压力验收。
