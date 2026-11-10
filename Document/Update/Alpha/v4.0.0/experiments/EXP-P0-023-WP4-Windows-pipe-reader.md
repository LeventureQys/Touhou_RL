# EXP-P0-023 — WP4 Windows Th08Env HELLO 跨平台管道修复与单环境验收

- 日期：2026-11-08
- 目标：修复 Windows `select.select` 监控 subprocess anonymous pipe 导致的 WinError 10093/WSAStartup，并以真实 binary/data 执行单 `Th08Env` HELLO/RESET/STEP。

## 实现

允许范围内仅修改 `touhou_rl/v400/gym/protocol.py`：

- Linux 保留原有 `select.select` + `os.read` 行为。
- Windows 使用 daemon reader thread 阻塞读取 stdout，并通过 `queue.Queue` 向请求线程提供带 deadline 的读取；EOF、线程异常和子进程提前退出均转换为 `HarnessError`。
- 关闭时清理 reader 状态；未改 wire protocol、WSL C++、第三方或依赖。

## 回归证据

命令：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests/test_v400_protocol.py tests/test_v400_recovery.py tests/test_v400_env.py
```

结果：`12 passed in 0.11s`，退出码 0。

## 真实单环境尝试

真实资源存在：

- binary：`/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern`
- data：`/root/v400-data/game`

命令从 Windows 工作区调用 WSL Python，使用真实 `Th08Env`：

```text
wsl.exe -d Ubuntu -- bash -lc "PYTHONPATH=/mnt/d/WorkShop/Touhou_RL /root/v400-venv/bin/python ... Th08Env(...).hello/reset/step ..."
```

结果：退出码 1；在 `Th08Env.__enter__` 的 HELLO 阶段收到 `HarnessError: harness closed the control pipe`，因此 RESET/STEP 未执行。该结果不构成真实成功，未伪造多环境验收。该阻塞与 Windows `select` 修复分离：当前实际运行的是 WSL Linux binary，保留 Linux select 路径；未找到可直接在 Windows 原生启动的 `th08-modern` binary。需要后续诊断 harness 启动/显示或 WSL 运行条件后再验收真实 RESET/STEP。
