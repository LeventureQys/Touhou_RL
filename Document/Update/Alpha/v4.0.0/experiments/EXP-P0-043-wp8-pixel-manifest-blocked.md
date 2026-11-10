# EXP-P0-043 — WP8 双端 pixel manifest 预检（FAIL-CLOSED）

- 日期：2026-11-08（America/Los_Angeles）
- Provider / model：DeepSeek / DeepSeek v4.1 flash
- 范围：只读预检与证据记录；未修改零售 EXE、第三方、依赖或 replay；未生成 PNG，未生成半边 manifest。

## 有界命令与退出码

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python tools\rl\v400_retail_preflight.py --output .tmp_v400_wp8_retail_preflight.json
```

退出码：**3**（blocked）。报告：`.tmp_v400_wp8_retail_preflight.json`。

## 已核验

- `third_party/th08/resources/th08.exe` 存在，840704 bytes。
- SHA-256：`330fbdbf58a710829d65277b4f312cfbb38d5448b3df523e79350b879213d924`，与预检冻结零售身份一致。
- `th08_replay/` 存在真实 `.rpy` 文件（当前列出 40 个左右），但预检所需 fixture `th8_ud2cdc.rpy` 与 `th8_ud2c0a.rpy` 均不存在/不匹配。

## 阻塞证据

预检报告 gaps：

- `fixture_missing_or_hash_mismatch:th8_ud2cdc.rpy`
- `fixture_missing_or_hash_mismatch:th8_ud2c0a.rpy`
- `game_data_missing`
- `bgm_data_missing`
- `tool_missing:wine_loader_present`
- `tool_missing:wineboot`
- `tool_missing:xdotool`
- `tool_missing:Xvfb`

当前 PowerShell 预检的 `--wine`、`--game-data`、`--bgm-data` 默认值是 Linux/WSL 路径；因此 `tool_missing:*` 与 `*_missing` 的含义是“从当前 Windows 预检入口按这些默认路径未见”，**不等价于证明 WSL 内绝对不存在** Wine32 或数据。既有 EXP-P0-037 的历史证据表明，WSL/Wine32 曾有一条真实资源运行路径（Wine 6.0.3、`th08.dat`/`thbgm.dat` 的 hash 已记录），但其官方 ptrace observer 在首个 playback-control 边界前为 0 rows，外层有界 timeout 退出 124。当前也未发现可直接启动的 Windows native exact capture 调试/帧栅栏工具及其授权采集产物。现有上游 `capture-replay.py` 是 observer/trace 采集器，不是已落地的 PNG+逻辑帧号+状态哈希双端 manifest 生成器。

Linux harness 的 `v400_wsl_env.py` 仅提供 HELLO/RESET/STEP 探针，未提供与 Windows capture 共享 replay 的同语义逻辑帧 PNG+状态哈希输出；因此不能证明对应帧语义一致。

## 结论

**WP8 双端 manifest 本轮不能解除。** 当前缺少已核验零售 fixture/data、可启动 Windows exact 帧栅栏、以及可与其共享 replay 并证明同逻辑帧语义的 Linux capture。严格 fail-closed：不运行无界尝试，不造 PNG，不生成半边 manifest，不宣称至少一帧配对成功。

资源恢复后的最小下一步：先补齐同一已核验 replay、`th08.dat`/`thbgm.dat` 与授权 Windows exact capture 产物；单例有界运行，必须同时得到 Windows 与 Linux 同一 `logicFrame` 的真实 PNG 和 `state_sha256`，否则保持 exit 3。
