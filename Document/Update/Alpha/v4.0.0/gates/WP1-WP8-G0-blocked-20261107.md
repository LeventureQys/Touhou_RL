# WP1 / WP8 / G0 严格阻塞记录（2026-11-07）

**结果：未完成 WP1、WP8、G0；无任何真实双端配对帧，不计伪造样例。**

## 实测资源核查

- Windows `third_party/th08/build/play/th08.exe` 存在，但仅为现有项目运行产物；未证明是零售版，不能以文件名代替零售身份。
- WSL `~/v400-data/game` 有 `th08.dat`、`thbgm.dat`、`score.dat`、`th08.cfg`，**无 `th08.exe`**；`command -v wine`、`wine32`、`xdotool` 均无结果。
- Windows 工作区 `D:\WorkShop\Touhou_RL\third_party\th08-rl`（对应 WSL 挂载路径 `/mnt/d/WorkShop/Touhou_RL/third_party/th08-rl`）不存在；**但 WSL 独立仓库 `/root/Touhou_RL/third_party/th08-rl` 存在，`build/rl/th08-modern` 和 `scripts/test-portable-replay-suite.py` 均存在**。本轮只核实它们的存在，未以此冒充回放验证通过。
- 当前工作区未发现可提供真实 Windows/Linux 同逻辑帧的参考轨迹清单。EXP-P0-001 已指出 `TH08_REPLAY_TRACE_REFERENCE` 缺失时 recorder 不启用，空文件也不能自举。
- 已有 `th08_replay` `.rpy` 不能替代零售执行结果或逐逻辑帧 PNG、状态指纹；因此未跑“两例”，WP1 的 108/108、WP8 的 3×200 均未达成。

## 可运行比较 harness

`python tools/rl/v400_pixel_pair.py --windows <windows-frames.json> --linux <linux-frames.json> --out <report.json>`；每个 manifest 包含 `source=windows-fence|linux-harness`、真实 `.rpy` 的 `replay_sha256`、200 个 `{logicFrame, image, state_sha256}`。所有 PNG 要相对 manifest 放置、严格 640×480 RGBA；拒绝缺帧、重复帧、hash 不同、尺寸/模式不符。按同一 logicFrame 逐像素 RGB 统计 playfield x=[32,416) 和其余 HUD 的平均绝对差、差异像素比、最大通道差与最差 10 帧清单。**FPS HUD 不静默遮盖；无人工检视和 3 份 Normal 回放不称 G0.4 PASS。**`--frames 1` 仅供单测，不能算验收。

## 此轮命令和退出码

| 命令 | 退出码 | 实际结果 |
|---|---:|---|
| `wsl.exe -- bash -lc 'command -v wine; command -v wine32; command -v xdotool; ls -ld ~/v400-data/game /mnt/d/WorkShop/Touhou_RL/third_party/th08-rl /mnt/d/WorkShop/Touhou_RL/third_party/th08/scripts/test-portable-replay-suite.py 2>&1'` | 2 | 前三个命令无输出；两个指定路径无文件；游戏数据目录存在。|
| `wsl.exe -- bash -lc 'ls -la ~/v400-data/game; ...; command -v wine || echo wine_missing; command -v xdotool || echo xdotool_missing'` | 0 | 游戏目录无 EXE；wine_missing / xdotool_missing。|
| `wsl.exe -- bash -lc 'ls -l /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern /root/Touhou_RL/third_party/th08-rl/scripts/test-portable-replay-suite.py 2>&1'` | 0 | WSL 独立仓库中的已构建 harness 可执行文件和回放 suite 脚本均实际存在；先前缺失的只是 Windows 工作区镜像路径。|
| `$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests -k 'replay or pixel or g0'` | 0 | **61 passed, 353 deselected**；新 6 个合成单测仅验证算法拒绝条件，不作真实回放证据。|
| `python tools/rl/v400_pixel_pair.py --windows .tmp_v400_wp1_wp8_missing_windows.json --linux .tmp_v400_wp1_wp8_missing_linux.json --out .tmp_v400_wp1_wp8_result.json` | 3 | 明确 BLOCKED：Windows manifest 不存在；未写出虚假比较报告。|

## G0 门槛证据判定

- **G0.1 FAIL/未测**：harness ON 的 108 份回放没有与零售参考逐位验证；两例也无法执行。
- **G0.2 未确认**：既有 EXP-P0-005 仅证明部分确定性，不等于 WP5 五项全通过；HUD fps 挂钟影响像素复现。
- **G0.3 未确认**：本轮不重新测吞吐，不以历史单项速率代替双阈值。
- **G0.4 FAIL/未测**：0/3 回放、0/600 真实配对帧，只有可运行严格分析器。
- 工程及 scoreboard 未全面验收；本次选择性 pytest 不能宣称 G0 工程门通过。

解阻需要：有授权、可确认哈希来源的**零售 TH08 原始游戏二进制** + Wine32/xdotool 或可信已保存零售轨迹；**WSL `/root/Touhou_RL/third_party/th08-rl` 的 harness 与 suite 已存在，无须据本报告重新恢复 worktree**。补齐零售参考后运行真实回放验证；随后由两端同一 `.rpy` 收集逐帧逻辑号、状态 SHA256、无损 PNG，选 3 份 Normal×200 帧执行比较、人工复核最差十帧，再跑完整 108/108、WP5/吞吐/工程/bench 门。缺这些条件不应翻转 G0。

## 权威审计覆盖（2026-11-08；Provider=DeepSeek；模型=DeepSeek v4.1 flash）

本节覆盖本卡中较早的环境探查摘要（例如“无 Wine/xdotool”或仅据 WSL 游戏目录推断无原件），不修改历史记录：

- **EXP-P0-029** 已证明单实例真实 Windows→WSL `HELLO/RESET/STEP/QUIT` 成功，launcher exit 0，相关回归 12/12；这不是 WP4 多实例长门通过。
- **EXP-P0-032 原始 JSON 权威 FAIL-CLOSED**：`.tmp_v400_wp4_long.json` 请求 600 秒但实际 wall `58.3528413772583 s`；16/16 `HarnessError: harness closed the control pipe`、exit 1，未达 deadline。与之矛盾的 600.299 秒/15 成功摘要不采纳。
- **EXP-P0-035 30 秒原始复现权威 FAIL-CLOSED**：16 实例仅 11/16 到达 deadline，5/16 约 2.904–4.378 秒提前 pipe closed，exit 134；wall `30.113247632980347 s`，结束无 WSL 残留。该结果不外推 600 秒通过。
- **EXP-P0-033/WP7、EXP-P0-034/WP8** 仍分别因缺少合法 human reference/scoreboard、真实 Windows capture/manifests 与双端 pixel pair 而阻塞。

**本卡更新后的结论：WP1/WP7/WP8/G0 未完成；WP4 长门 FAIL-CLOSED。**
