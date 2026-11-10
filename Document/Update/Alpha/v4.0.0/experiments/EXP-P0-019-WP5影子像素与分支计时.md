# EXP-P0-019 — WP5 第 3/4/5 项真实验收（2026-11-08）

结论：第 3 项 **PASS**；第 4 项 **BLOCKED**；第 5 项 **PASS（引擎现有计时字段的口径限制见下）**；G0.2 五项未全过，不能放行。

## 来源与命令

- WSL Ubuntu 真游戏：`/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern`；SHA-256 `f9686a51a76cee5ae58da13d3a318f43360c0c7271aafcf917dc160e16558f05`；数据目录 `/root/v400-data/game`；`git rev-parse HEAD` 为 `f563df58c44037b44208c393a86540b434b0c83b`。该 commit 是 HEAD 身份，工作树有未提交修改，因此**不能将 HEAD 误称二进制源码的完整、干净版本身份**；二进制 SHA 单独记录。
- 真实运行：`wsl.exe -d Ubuntu -- xvfb-run -a python3 /mnt/d/WorkShop/Touhou_RL/tools/rl/v400_wp5_acceptance.py --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-dir /root/v400-data/game --harness-commit f563df58c44037b44208c393a86540b434b0c83b --timeout 30 --out /mnt/d/WorkShop/Touhou_RL/.tmp_v400_wp5_acceptance_report.json`。脚本退出 **3**：整体未通过（像素阻塞）；PowerShell 包装执行尾部 `echo ACCEPTANCE_EXIT=$?` 得到 `ACCEPTANCE_EXIT=3`、包装命令自身退出 0。引擎三个真实进程退出码全为 **0**，其 argv、逐条协议命令与响应见原始 JSON。
- 单测：`$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_wp5_acceptance_pytest -q tests/test_v400_wp5_acceptance.py`：**4 passed，退出 0**；它仅验证验收器门槛，不替代真机证据。
- 早期 `xvfb-run -a -s "-screen 0 640x480x24" ...` 经 PowerShell 传参出现 `xvfb-run: usage error: need a command to run`，退出 1；已改用上述不含 `-s` 的真实成功运行，勿将失败命令计入通过。

## 逐项结果

| 项 | 状态 | 真实观察与限制 |
|---|---|---|
| 3 影子/主环境每 60 帧核对 | **PASS** | `role=shadow` 与 `role=env` 两个真实二进制进程，影子 DEBUG `threads=1`；相同 `RESET 0 1 0 1 0` 后分别推进 **10000/10000 帧**；每 60 帧及最后第 10000 帧比 `HASH.hash` 和 `HASH.frames`，**167/167 检查点一致**。影子进程 exit 0，环境进程 exit 0。输入序列为全零 mask，覆盖一条真实 StageStart 轨迹，不声称随机轨迹覆盖。 |
| 4 轨迹 + commit 原始像素缓存字节再生成 | **BLOCKED** | 未提供可认证的原始轨迹输入文件与对应原始像素字节缓存；在限定的已有数据目录亦未找到二者。禁止生成伪造“原始”像素、禁止仅凭 state hash 或 HUD fps 不稳定的 `PIXELS` checksum 宣称字节相等。脚本即便传入形式上的文件，未有格式特定真实再生逻辑也严格 blocked。解阻需提供具来源、逐帧输入与原始像素字节的真实缓存，确定实际构建身份、实现并复测真实字节比较。 |
| 5 真实 BRANCH timing | **PASS（现有字段口径）** | 同一真实单线程 shadow 上 **20 次 BRANCH 0 300**，每次 `branch=1`、`advanced=300`、真实 pid，并核验父 `FRAME` 不变；两次分支间父 `STEP 0 1`。20 条 `fork_us`/`child_us`/`child_us_per_frame` 与请求端 `request_wall_us` 均在 JSON；`fork_us` 均值 **15470.3 μs**，子分支每帧均值 **47.757 μs**。**注意**引擎 `fork_us` 实为 BRANCH 请求总耗时（含子推进及 waitpid），不等于隔离出的 fork() 系统调用耗时；如果验收要求纯 fork syscall，须保留此细分指标 BLOCKED，不得改名冒充。进程 exit 0。 |

原始证据：[`.tmp_v400_wp5_acceptance_report.json`](../../../../../.tmp_v400_wp5_acceptance_report.json)；脚本：[`v400_wp5_acceptance.py`](../../../../../tools/rl/v400_wp5_acceptance.py)；单测：[`test_v400_wp5_acceptance.py`](../../../../../tests/test_v400_wp5_acceptance.py)。第 1 项完整重跑、原始像素源及**单独 fork() 计时**仍待独立证据，不能由这张卡代替。
