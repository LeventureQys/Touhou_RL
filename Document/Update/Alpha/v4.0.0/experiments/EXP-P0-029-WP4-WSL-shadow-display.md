# EXP-P0-029：WP4 Windows→WSL shadow null-device 启动验证

- 日期：2026-11-08（America/Los_Angeles）
- 执行模型：DeepSeek / DeepSeek v4.1 flash
- 范围：`tools/rl/v400_wsl_env.py`、`tests/test_v400_wsl_env.py`、本记录；未改协议、依赖、第三方。

## 修改

WSL argv 改成 `wsl.exe -d Ubuntu -- env -u DISPLAY SDL_AUDIODRIVER=dummy <ELF> --data-dir <data> --rl-control 0,1 --rl-role shadow`。显式取消 WSL 侧继承 `DISPLAY=:0`，避免 broken X connection；保持 Windows stdout reader-thread 原逻辑。添加 mock 回归断言 argv 与 reader-thread 调用。

## 先运行现有测试

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests/test_v400_protocol.py tests/test_v400_vec_env.py
```

退出码 **0**，输出：`11 passed in 0.18s`。当时尚不存在 `tests/test_v400_wsl_env.py`。

## 修改后回归

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests/test_v400_wsl_env.py tests/test_v400_protocol.py tests/test_v400_vec_env.py
```

退出码 **0**，输出：`12 passed in 0.19s`。

## Windows launcher → 真实 WSL ELF

真实 binary `/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern`；真实 data `/root/v400-data/game`。从 Windows Python 调用 `WslHarnessClient(...,timeout=25)`，依次 `start()`, `hello()`, `reset(stage=0,difficulty=1,team=0,seed=1)`, `step(0,2)`, `close(quit_timeout=3)`；使用 Windows reader-thread，没有直接改协议。

首次探针将已经关闭的 `proc.stderr` 再读，出现 `ValueError: I/O operation on closed file`，外层退出码 **1**；在该探针中真实 HELLO/RESET/STEP 均成功，QUIT child exit 0。移除探针自身关闭后读 stderr 的错误，再跑相同真实调用，原始输出：

```text
PID 29076
HELLO {'_verb': 'OK', 'proto': '1', 'harness': '1', 'role': 'shadow', 'frames': '0'}
RESET {'_verb': 'OK', 'reset': '1', 'ready': '1', 'setup_frames': '300', 'frames': '0', 'stage': '0', 'difficulty': '1', 'team': '0', 'seed': '1', 'setup_state': '0', 'hash': '7f04c7361749ed9f'}
STEP {'_verb': 'OK', 'advanced': '2', 'frame': '2', 'hash': '220b78e74341367b'}
QUIT exit 0
LAUNCHER_EXIT=0
```

结果：**单实例真实 Windows→WSL shadow HELLO/RESET/STEP/QUIT 通过**；不将该单实例 probe 冒充 WP4 多实例压力验收。
