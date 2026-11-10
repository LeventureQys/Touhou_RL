# EXP-P0-022 WP2 真实 shadow 分段 RESET 吞吐统计

- 日期：2026-11-08；环境：WSL Linux 真实 binary `/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern`，真实游戏数据 `/root/v400-data/game`，`/root/v400-venv/bin/python`。不借用 `LP_NO_RAST`、DISPLAY、WAYLAND_DISPLAY 或模拟引擎；`--rl-role shadow`；DEBUG `threads=1/subthread=0`。单核为单线程进程的名义吞吐口径，**未强制 CPU affinity**。
- 预定门槛：每核推演 ≥ 5,000 逻辑帧/s；三次独立样本，每次恰好 1,000 帧，样本以单独进程启动 + `RESET 0 1 0 1 1` 后 `curState=2, frames=0` 为起点，STEP 10×100，逐步 DEBUG 校验 `curState=2` 且 DEBUG frames 与 `advanced` 累计相等；结尾 `curState=2, frames=1000`。计时仅累计每条 STEP 调用的 perf_counter 墙钟；RESET、起进程与 DEBUG 检查不计入。未跨 `curState6`。

## 实际命令及原始证据

```powershell
wsl -- bash -lc 'PYTHONPATH=/mnt/d/WorkShop/Touhou_RL /root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v400_shadow_throughput_stats.py --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-dir /root/v400-data/game --samples 3 > /mnt/d/WorkShop/Touhou_RL/.tmp_v400_wp2_stats_3x1000.json'
# WSL_STATS_EXIT=0
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_wp2_stats_pytest -q tests/test_v400_shadow_throughput_stats.py tests/test_v400_shadow_throughput.py
# 23 passed in 0.09s; PYTEST_EXIT=0（仅合成逻辑测试，不是引擎性能证据）
```

| 样本 | 实际帧 | STEP 次数 | STEP 墙钟秒 | 逻辑帧/s/核 | ≥ 5,000/核 |
|---:|---:|---:|---:|---:|:---:|
| 1 | 1000 | 10 | 0.0684142470 | 14,616.84 | 是 |
| 2 | 1000 | 10 | 0.0673972490 | 14,837.40 | 是 |
| 3 | 1000 | 10 | 0.0689280780 | 14,507.88 | 是 |

- 三样本**算术均值 14,654.04 帧/s/核**；观测区间（样本 min–max，**不是置信区间**）**[14,507.88, 14,837.40] 帧/s/核**；三个样本全部单独超过 ≥ 5,000 门槛，**WP2 推演单核门通过**。原始逐段 DEBUG 与数字见 `.tmp_v400_wp2_stats_3x1000.json`。这不是全机渲染加读回 ≥ 2,000 门的通过证据；历史实测该门仍未达到，G0 不得称 PASS。

## 严格失败记录与调整

- 首次尝试在**同一个进程**连续 `RESET`：第一段 1000 帧有效（14,438.48 帧/s/核），但第二次 `RESET ready=0`，工具明确 `status=unavailable`、总指标 `null`、**exit 3**；原样保留 `.tmp_v400_wp2_stats_real.json`，该局部结果不计入三样本均值，也不作为通过证据。引擎在运行中不接受重复 RESET；因此修正 Python 工具为**每段关闭并重新启动真实 shadow 进程**，每段重新 HELLO/DEBUG 验证并仅 RESET 一次；未修改 C++。
- 历史 10000 连续帧测量因 `curState=6` 无效（EXP-P0-018、`.tmp_v400_wp2_10000.json`）；本实验不重新包装该失败，也不推断长程稳定性。测试以 `curState2` 每条 STEP 后检查，越界立即 exit 3 且不报告有效聚合指标。
