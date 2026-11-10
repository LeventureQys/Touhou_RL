# EXP-P0-045：官方 108 fixture `retailTrace.sha256` 来源核验（WP1）

日期：2026-11-08。Provider=DeepSeek，模型=DeepSeek v4.1 flash。

## 结论

**WP1 保持 BLOCKED（0/108 可验收参考轨迹）**。在明确允许的路径 `third_party/th08/config/replay-fixtures.json`、`refs/`、`th08_replay/` 与官方 fixture replay 目录内，未找到任何可独立取得、包含完整 `rows.jsonl` + `complete.json` 且可核验 `retailTrace.sha256` 的参考轨迹。不得把 `.rpy` 解码或其输入记录当作参考轨迹。

## 已核验来源

- 官方 manifest：`third_party/th08/config/replay-fixtures.json`。PowerShell 解析命令提取到 **108** 个 `id/file/url/retailTrace` 记录；例如 `easy-0-b` 的 replay URL 为 `https://maribelhearn.com/replays/royalflare/th08/th8_ud2cdc.rpy`，`calculationFrames=107296`，`retailTrace.sha256=da8a6a0de9a79263a2ad13a9351eb467c2d0c2961585aa474149cecb010bbf67`。该字段只是期望指纹，不是本地轨迹文件路径。
- 官方页面 URL 只读 HEAD：replay URL 返回 **HTTP 200**、`Content-Type: application/octet-stream`、`Content-Length: 30418`；archive index `https://maribelhearn.com/royalflare/th08a` 返回 **HTTP 200**。这证明公开 `.rpy` 下载入口可达，不证明存在公开 retail trace rows。
- `third_party/th08/tests/fixtures/replays/th08-1.00d/README.md` 明确说明 accepted 目录有 108 个公开 recordings，suite 使用这些 recording 作为 replay 输入；其 provenance/rights 说明 recording rights 属作者，且需另行取得原始 EXE/data。README 未提供 retail trace rows 或 complete 文件。
- `refs/` 未发现 fixture manifest、rows、complete 或 retail trace；`th08_replay/` 仅有本地 `.rpy` 输入文件。现有 `.rpy` 不能替代 retail 执行产生的 trace。

## 可重复命令与退出码

1. 统计并提取 108 条 manifest 记录（退出码 **0**）：

```powershell
$p='third_party\\th08\\config\\replay-fixtures.json'; $raw=Get-Content -LiteralPath $p -Raw; $m=[regex]::Matches($raw,'"id"\s*:\s*"([^"]+)"[\s\S]*?"file"\s*:\s*"([^"]+)"[\s\S]*?"url"\s*:\s*"([^"]+)"[\s\S]*?"retailTrace"\s*:\s*\{[\s\S]*?"calculationFrames"\s*:\s*(\d+),\s*"sha256"\s*:\s*"([0-9a-f]{64})"'); $m.Count
```

预期输出：`108`。

2. 只读验证官方 URL（退出码 **0**；HTTP 结果见上）：

```powershell
Invoke-WebRequest -Uri 'https://maribelhearn.com/replays/royalflare/th08/th8_ud2cdc.rpy' -Method Head -UseBasicParsing
Invoke-WebRequest -Uri 'https://maribelhearn.com/royalflare/th08a' -Method Head -UseBasicParsing
```

3. 检查允许路径中是否存在参考产物（退出码 **0**，结果为空）：

```powershell
$roots=@('refs','th08_replay','third_party\\th08\\tests\\fixtures\\replays','Document\\Update\\Alpha\\v4.0.0'); Get-ChildItem -LiteralPath $roots -Recurse -File -ErrorAction SilentlyContinue | Where-Object {$_.Name -match 'rows\.jsonl|complete\.json|retail.*trace|reference.*trace'} | Select-Object -ExpandProperty FullName
```

## 首例可独立获取的非-Wine替代路径

首例为 **官方 native Linux portable harness 自己执行同一公开 `.rpy` 并生成 trace**，不是把 `.rpy` 自解码成参考：在 WSL worktree 构建 `TH08_ENABLE_REPLAY_TRACE=ON` 的 Linux harness，使用官方 `scripts/test-portable-replay-suite.py`/trace recorder 运行 `easy-0-b`，并将结果与 manifest 的 `retailTrace.sha256` 比对。它是可合法独立获取的非-Wine路径（仅使用已授权/可读游戏数据与公开 replay），但当前 recorder 需要预先存在 `TH08_REPLAY_TRACE_REFERENCE`，不能自举，因此本机尚未产生真实 rows+complete，不能宣称为参考或翻转 WP1。

最小执行证据命令（当前用于证明 fail-closed，预期退出码 **非 0**，不得伪造输出）：

```bash
cd /root/Touhou_RL/third_party/th08
cmake -S . -B build/portable-linux-x86_64-trace -DTH08_ENABLE_REPLAY_TRACE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/portable-linux-x86_64-trace -j2
TH08_REPLAY_TRACE_REFERENCE=/tmp/EXP-P0-045-missing-reference.rows.jsonl \
  python3 scripts/test-portable-replay-suite.py \
  --candidate build/portable-linux-x86_64-trace/th08-modern \
  --game-data-dir /root/v400-data/game \
  --fixture easy-0-b
printf 'exit=%s\n' "$?"
```

只有取得可信来源的完整 `rows.jsonl`、`complete.json`（schema/version、fixture/replay/EXE/data hash 均匹配）并计算出 manifest 指定的完整 SHA-256 后，才可解除 BLOCKED。不得使用 `.rpy` 解码、空 rows、缺 complete、合成数据或研究构建替代。
