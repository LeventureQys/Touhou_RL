# EXP-P0-021 — WP4 VecTh08Env 真实多进程隔离短压测

- **日期**：2026-11-08（America/Los_Angeles）
- **Provider / 模型**：DeepSeek / DeepSeek v4.1 flash
- **结论**：**BLOCKED；不得记为通过**。本轮未执行 16 环境 60 秒压测，也未执行 10 分钟完整压测。

## 资源核查

真实资源存在，未因缺少 binary/data 阻塞：

- `third_party/th08/build/play/th08.exe`：898048 bytes
- `third_party/th08/build/play/th08.dat`：46838025 bytes
- `third_party/th08/build/play/thbgm.dat`：449961024 bytes

## 真实启动证据

命令：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -c "from touhou_rl.v400.gym.env import Th08Env; b=r'D:\WorkShop\Touhou_RL\third_party\th08\build\play\th08.exe'; d=r'D:\WorkShop\Touhou_RL\third_party\th08\build\play'; e=Th08Env(b,d,shm_name='wp4_probe_one'); print('constructed'); e.__enter__(); print('started'); e.close()"
```

- **exit code**：1
- `Th08Env` 构造成功，但在 HELLO 阶段失败：`touhou_rl/v400/gym/protocol.py` 的 `select.select` 抛出 `OSError: [WinError 10093] WSAStartup failed`。
- 因此没有完成 HELLO/RESET，不能声称真实环境已建立，更不能推导多进程隔离、帧推进或崩溃恢复。

独立的原始管道探针也未得到协议回复：向实际 `th08.exe --data-dir ... --rl-control 0,1` 子进程发送 `HELLO`，等待 5 秒无 stdout/stderr；记录的子进程 PID 为 **25896**，随后 kill，return code **1**。该 PID 仅是失败探针，不是压力测试峰值 PID。

## 合同测试（非真实验收）

命令：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests/test_v400_vec_env.py
```

- **exit code**：0
- **结果**：4 passed in 0.15s
- 这些测试使用 `_FakeEnv`，仅验证 `VecTh08Env` 的调度/关闭合同，**不启动真实 TH08 binary，不证明 WP4 通过**。

## WP4 证据缺口

由于真实 harness 在 Windows 运行时无法完成 HELLO：

- 未运行 16 个真实环境的 60 秒短压测；
- 无合法的峰值 harness PID 统计；
- 无真实帧推进计数；
- 无崩溃恢复事件证据；
- 无无僵尸/泄漏检查证据；
- 未运行且不声称通过 10 分钟压测。

后续解阻需要在支持该控制管道的运行环境中修复/提供 WSAStartup 兼容路径，随后重新执行真实短压测。本轮未修改 `env.py`、`protocol.py`、第三方或依赖。
