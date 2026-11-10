# EXP-P0-025 — WP8 Windows fence × Linux harness 配对诊断

日期：2026-11-08  
Provider/模型：DeepSeek / DeepSeek v4.1 flash  
结论：**严格前置阻塞（BLOCKED，未生成像素成绩）**。

## 目标与真实性边界

本实验只接受同一份真实回放的同一逻辑帧：Windows fence 与 Linux harness manifest 必须具有相同 `replay_sha256`，并逐帧提供 `logicFrame`、`state_sha256` 和 manifest 相对路径下的真实 PNG。工具要求默认 200 个精确配对帧、PNG 为 640×480 RGBA；缺任一前置时退出 3。不得用合成 PNG、重复运行、历史截图或空结果代替真实证据。

现有工具：`tools/rl/v400_pixel_pair.py`。它在输入缺失、来源/哈希/帧集合/状态指纹/PNG 属性不满足时 fail-closed；只有成功配对才写报告。`--frames 1` 仅用于单元测试，不是 WP8 验收。

## 现有零售原件/回放/工具核查

已运行 `tools/rl/v400_retail_preflight.py`（只读身份预检）：

- `third_party/th08/resources/th08.exe` 身份哈希匹配既定 TH08 1.00d（size 840704，SHA-256 `330fbdbf...213d924`）。这只证明文件身份，不证明授权或已完成采集。
- 两个 fixture 回放 `th8_ud2cdc.rpy`、`th8_ud2c0a.rpy` 均不存在。
- Windows 工作区对应预检所需的 `/root/v400-data/game/th08.dat`、`thbgm.dat` 路径在当前 Windows 调用中不可用。
- Wine loader、`wineboot`、`xdotool`、`Xvfb` 均不可用。
- 没有发现可用于 Windows/Linux 同一逻辑帧的双端 manifest；因此没有真实配对帧，也没有生成任何 PNG 或像素成绩。

## 可复制命令与退出码

在仓库根目录执行：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python tools/rl/v400_retail_preflight.py --output .tmp_v400_wp8_retail_preflight.json
# EXIT=3（身份文件匹配，但 fixture 回放、游戏数据和 Wine/X11 工具缺失）

python tools/rl/v400_pixel_pair.py `
  --windows .tmp_v400_wp8_missing_windows.json `
  --linux .tmp_v400_wp8_missing_linux.json `
  --out .tmp_v400_wp8_result.json
# EXIT=3；stderr: BLOCKED: Windows manifest 不存在；不写出结果

python -m pytest -p no:cacheprovider --basetemp .tmp_v400_wp8_pytest -q tests/test_v400_pixel_pair.py
# EXIT=0；6 passed
```

工具阻塞命令的实际输出为：`BLOCKED: [Errno 2] No such file or directory: '.tmp_v400_wp8_missing_windows.json'`，退出码 3；`.tmp_v400_wp8_result.json` 未产生。

## 解阻条件

需提供并核验：

1. 两端由同一真实 `.rpy` 产生的 manifest，且 `replay_sha256` 相同；
2. 至少 200 个相同 `logicFrame`，每帧 `state_sha256` 相同；
3. 两端各自真实、无损、相对 manifest 的 640×480 RGBA PNG；
4. 能实际运行 Windows fence 与 Linux harness 的回放环境（或可信保存的双端逐帧产物）；
5. WP8 要求的三份真实 Normal 回放各 200 帧以及人工复核最差帧。仅具备零售 EXE 哈希或既有 `.rpy` 文件，不能推出配对像素结果。

在上述条件补齐前，WP8 必须保持 BLOCKED；本记录不宣称 G0.4 通过，不造 PNG，不报告像素差异或成绩。
