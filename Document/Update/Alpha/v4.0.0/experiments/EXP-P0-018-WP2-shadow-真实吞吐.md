# EXP-P0-018 WP2 真实 WSL shadow 空设备吞吐首测

- 日期：2026-11-08；条件：WSL 现有 Linux binary `/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern` 与真实游戏数据 `/root/v400-data/game`。脚本 `tools/rl/v400_shadow_throughput.py` 明确起 `--rl-role shadow`，移除 `DISPLAY`/`WAYLAND_DISPLAY`，禁止 `LP_NO_RAST` 代理。Windows 仓库路径为 `/mnt/d/WorkShop/Touhou_RL`；WSL Python `/root/v400-venv/bin/python`。
- 命令（先 1000 帧，后 10000 帧）：

```bash
PYTHONPATH=/mnt/d/WorkShop/Touhou_RL /root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v400_shadow_throughput.py --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-dir /root/v400-data/game --frames 1000 > /mnt/d/WorkShop/Touhou_RL/.tmp_v400_wp2_1000.json
# exit=0
PYTHONPATH=/mnt/d/WorkShop/Touhou_RL /root/v400-venv/bin/python /mnt/d/WorkShop/Touhou_RL/tools/rl/v400_shadow_throughput.py --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-dir /root/v400-data/game --frames 10000 > /mnt/d/WorkShop/Touhou_RL/.tmp_v400_wp2_10000.json
# exit=3 (unavailable)
```

- **有效的 1000 帧结果**：STEP 共 10 次、实际 `advanced` 总数 1000；STEP 墙钟 **0.069103362 秒**；**14,471.08 帧/s/核**（1 条进程线程的实测单核名义口径；非 CPU affinity 强制绑核）。HELLO role=shadow；RESET ready=1；DEBUG 初始、RESET 后、STEP 后均 threads=1/subthread=0；关卡状态 STEP 前后 `curState=2`；原始输出 `.tmp_v400_wp2_1000.json`。
- **10000 帧无效**：游戏 STEP 前 `curState=2`、结束后 `curState=6`，已跨出活跃关卡；加严脚本后重测明确输出 `status=unavailable`、`measured_frames_per_second_per_core=null`、退出码 **3**；原始 `.tmp_v400_wp2_10000.json`。不得把跨终局逻辑帧数包装成 10k 稳定推演吞吐或达标证据；若需长程测量应拆分重置/选择更持久的回放或路线，且必须保持相同证据门。
- 有效性限定：1000 帧单次短窗口，未估计波动，未绑定物理 CPU 核，且本测只覆盖 null-device 推演，不覆盖渲染与像素读回；因此仅为 WP2 单核推演短测，不代表 WP2 其它吞吐门达成。
- 测试：`$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_wp2_pytest_run -q tests/test_v400_shadow_throughput.py`；9 passed，exit=0（合成单测只验 fail-closed，不作为引擎性能证据）。
