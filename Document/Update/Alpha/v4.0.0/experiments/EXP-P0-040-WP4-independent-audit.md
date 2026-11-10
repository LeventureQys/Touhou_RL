# EXP-P0-040 — WP4 独立审计

日期：2026-11-08；Provider=DeepSeek；模型=DeepSeek v4.1 flash。只读核验 [WP4 规格](../10-P0-训练场.md#L126-L137)、[EXP039 报告](EXP-P0-039-WP4-16x600.md)、[原始 JSON](../../../../../.tmp_v400_wp4_diag_039_600.json)、[G0 报告](../gates/G0-报告.md#L139-L166) 及 [压测脚本](../../../../../tools/rl/v400_wsl_vec_stress.py)、[WSL 启动器](../../../../../tools/rl/v400_wsl_env.py)。未重跑长门；不改原件/代码/G0。

## 裁决

**16 路真实 Windows→WSL shadow 控制链在共同 600 秒窗口持续 STEP、正常退出证据较强；但规格“16 环境并行跑 10 分钟，没有泄漏、没有僵尸进程”尚未完整证明。可称 `WP4 16×600 运行子门 PASS-with-limit（Linux 引擎 PID、内存趋势、僵尸/孤儿未证）`；若 WP4 PASS 指规格全部满足，不可无条件宣称 WP4 PASS。G0 仍 NOT PASS。** 缺证不等于已发现泄漏。WP4 还要求动作/协议单测与 reset 后 1000 step 集成验证，EXP039 本身不覆盖。G0 已认定长门 PASS，本审计仅限定其证据范围，不改历史结论；EXP032/035 早期失败保留。

| 主张 | 原始证据及脚本语义 | 强度 |
|---|---|---|
| 16 路长窗口 | JSON `instances=16`、`requested_seconds=600`、`wall_seconds=602.7674422264099`；16 条 `steps=frames>0`，总帧 **15,061,600**，全部 `error=null`、`exit_code=0`；EXP039 记录脚本 exit 0。 | **较强**：真实运行并推进大量帧；共同 deadline 与 16 worker 支持并发，但无 Linux 16 个 engine 同时在场快照。 |
| 每个 engine 精确 ≥600 秒 | `run_seconds=600.01057…600.01868`；计时从 `c.start()` 前到 `c.close()` 后；共享 deadline 在 worker 提交前设置，达标谓词实际为 `run_seconds >= seconds-2`。 | **有限定**：worker 生命周期约 600 秒，不证明每个 Linux ELF 自启动到最后 STEP 独立持续 ≥600 秒。 |
| PID/exit | `pid=c._proc.pid`；启动器 `_proc=subprocess.Popen(["wsl.exe",…])`。 | PID 是 **Windows wsl.exe PID**，非 Linux engine PID。exit 0 是 launcher 退出码；协议帧数间接证明 ELF 参与。 |
| 无僵尸/残留 | `remaining_wsl_process_lines=[]`、`no_wsl_processes=true` 来自一次 Windows `tasklist /FI "IMAGENAME eq wsl.exe"`，**未查 tasklist 返回码**，取样在 cleanup 前。`cleanup_exit_code=0` 仅为目录清理。 | **弱**：没有 Linux engine PID/PPID/State、退出后延时复查；无 Windows launcher 不能排除 WSL 孤儿/Z，tasklist 失败可假阴性。 |
| 无泄漏 | JSON 无 VmRSS/PSS、fd/线程、峰值和时间序列。 | **未证**：正常退出不排除运行中内存持续增长。 |
| 环境模式 | 启动器固定 `--rl-role shadow`，stderr 指示 null-device，无 SDL/GL/window/audio。 | 不外推 16 路渲染像素 `Th08Env/VecTh08Env` 或 G0.3 吞吐。 |

报告与 G0 记载 JSON SHA256 `731cfec8956e0642e65f3454f5786ba3b3819cd4e770a07046895d789d6aa2c5`；本次未重新计算哈希。

## 可复制验收（以下命令本次未运行，不虚构退出码）

```powershell
# D:\WorkShop\Touhou_RL 下仅只读核验
Get-FileHash -Algorithm SHA256 .\.tmp_v400_wp4_diag_039_600.json
$j = Get-Content .\.tmp_v400_wp4_diag_039_600.json -Raw | ConvertFrom-Json
$r = @($j.results)
[pscustomobject]@{Count=$r.Count; TotalFrames=($r | Measure-Object frames -Sum).Sum; MinWorkerSeconds=($r | Measure-Object run_seconds -Minimum).Minimum; MaxWorkerSeconds=($r | Measure-Object run_seconds -Maximum).Maximum; Errors=@($r | Where-Object { $null -ne $_.error }).Count; NonzeroExits=@($r | Where-Object { $_.exit_code -ne 0 }).Count; WallSeconds=$j.wall_seconds; CleanupExit=$j.cleanup_exit_code}
Select-String -Path .\tools\rl\v400_wsl_vec_stress.py,.\tools\rl\v400_wsl_env.py -Pattern 'r\["pid"\]|tasklist|run_seconds|all_instances_reached_deadline|Popen|--rl-role'
```

期望 SHA 同上，Count=16、TotalFrames=15061600、Errors=0、NonzeroExits=0、WallSeconds=602.7674422264099、CleanupExit=0；若不符暂停引用 PASS-with-limit。

**无条件验收须补证（下次经授权运行，不修改 EXP039）：** 建立 Windows launcher→Linux engine PID 映射，记录各 engine 的 PID/PPID、`/proc/<pid>/stat` starttime/State、`/proc/<pid>/status` VmRSS/VmHWM/Threads 与 fd 数；启动后 `ps -eo pid,ppid,stat,etime,args` 证明 16 个目标 ELF 同时存在，0/60/120/…/600 秒保留时间戳采样。退出/cleanup 后立即及 30 秒后复核**同 PID+starttime**是否消失、有无 `Z`/孤儿，记录 tasklist、ps、cleanup 的退出码及 stderr。预设稳定化窗口、末段内存增长斜率和峰值阈值，核对每引擎及 WSL 总量；没有预设阈值只报告曲线，不称无泄漏。若规格环境指渲染像素 VecTh08Env，shadow 长门不替代该模式。

## 可引用验收和风险

> EXP039：16 路真实 Windows→WSL shadow STEP 压测，共同 600 秒窗口，16/16 worker 无报告错误、launcher 退出 0，脚本报告退出 0；长时运行子门 PASS-with-limit。Linux 引擎 PID、内存趋势及 Linux 僵尸/残留未验证；完整 WP4“无泄漏/无僵尸”未证，G0 不放行。

禁止宣称“16 个 Linux engine 各精确连续满 600 秒”“已证明零泄漏/无 Linux 僵尸”“完整 WP4/G0 PASS”。后续发现孤儿/Z、超过预设增长阈值或不足 16 路并发，则对应断言 FAIL，保留原始证据；仅缺证时为**未证**而非实测 FAIL。
