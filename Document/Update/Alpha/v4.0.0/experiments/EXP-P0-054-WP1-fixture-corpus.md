# EXP-P0-054：WP1/G0.1 规模验证输入——108 份 fixture 回放全量下载与逐份校验

日期：2026-11-08。Provider=DeepSeek，模型=DeepSeek v4.1 flash。

## 结论

**108/108 全部下载并通过 SHA-256 与字节数双重校验**：`downloaded=108`、`verified=108`、`mismatch=0`、`failed=0`，全部 `HTTP 200`、全部首次尝试即通过（无重试）。本次只是把官方 manifest 列出的回放**输入**保存到本地并校验完整性，**没有**产生、也不声称存在任何 retail 参考轨迹。

> **这些 `.rpy` 只是输入，不是零售参考轨迹。** 它们是喂给 WP1/G0.1 的 replay 输入语料；`rows.jsonl`/`complete.json` 形式的零售参考轨迹并未取得，WP1 的可验收状态**不因本卡改变**（仍按 EXP-P0-045 保持 BLOCKED，0/108 可验收参考轨迹）。不得把 `.rpy` 的解码记录、输入流或本卡的校验结果冒充为参考轨迹。

## 输入与产物

| 项 | 路径/标识 |
| --- | --- |
| manifest | `third_party/th08/config/replay-fixtures.json`（version 1，108 条；sha256 `512835b74ab9c8b4104008a1f6f00d4ee731d70caf1c1b32c6554bf7e91e7808`） |
| 采集工具 | `tools/rl/v400_fixture_download.py`（sha256 `8f04057a621ef83fe1d6b7b50dee226b359c775939b79f7362d957326df3fe38`） |
| 工具测试 | `tests/test_v400_fixture_download.py`（sha256 `75db0eb92e4a8f134cbaddb3c98fe4f3100b75e8f32085508f0a4e3a23d88609`） |
| 回放语料目录（**新建**） | `.tmp_v400_retail_fixtures_all/`，108 个 `.rpy`，合计 3,504,743 字节 |
| 校验清单 | `.tmp_v400_retail_fixtures_all/manifest_check.json`（sha256 `99c5a1c1657c28ddf9b33d011a1e40d997aa5cee3703a79f2585ad396b779e69`，`generated_at=2026-11-08T15:38:56+00:00`） |

目录创建规则：目标目录若不存在则直接新建；若已存在则改用 `-<UTC时间戳>` 后缀的新目录，**从不删除、从不覆盖**既有目录。本次目标目录此前不存在，因此使用了规范路径本身。目录内仅含 108 个 `.rpy` 与 `manifest_check.json`（无 `.part`、无 `.mismatch` 残留）。

## 校验统计（真实执行结果）

| total | downloaded | verified | mismatch | failed |
| --- | --- | --- | --- | --- |
| 108 | 108 | 108 | 0 | 0 |

- 并发度 4（≤4）；内部总时限 1500 s（≤25 min 硬上限），实际墙钟 **27.2 s**（报告 `elapsed_seconds=27.094`）。
- 每条均为 `http=200`、`attempts=1`，`actual_bytes == manifest.bytes` 且 `actual_sha256 == manifest.sha256`。
- 语料规模分布：Easy/Normal/Hard/Lunatic 各 24，Extra 12；12 种机体（Border/Magic/Scarlet/Ghost Team + Reimu/Yukari/Marisa/Alice/Sakuya/Remilia/Youmu/Yuyuko）各 9 份；`-a` 48 份、`-b`/`extra` 60 份；全部 `practice=false`、`extended=false`；合计 588 个 stage 记录。
- 单文件字节数区间 12,596～43,917，总和 3,504,743，与 manifest 声明一致，也与磁盘实际字节数一致。

## 失败清单

**无失败条目。**

- `failed`：0 条（无 HTTP 错误、无异常、无超时、无触发总时限）。
- `mismatch`：0 条（无 SHA-256 或字节数不一致，故一次重试路径未被触发）。
- 未发生任何重试；`manifest_check.json` 的 `items[*].error` 全为 `null`。

## 独立复核（不复用工具逻辑）

用独立脚本从 manifest 重新读取每条 `file`，自行 `hashlib.sha256` 重新计算并比对，同时要求报告中的 `status/actual_sha256/actual_bytes` 与独立计算结果一致：

- `independent_verified=108/108`，`bad=[]`；
- 目录条目 109（108 `.rpy` + `manifest_check.json`），`extra=[]`；
- 磁盘 `*.rpy` 字节总和 3,504,743，等于 manifest 总和。

## 可重复命令与退出码

1. 语法编译校验（退出码 **0**）：

```powershell
cd D:\WorkShop\Touhou_RL
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m compileall -q tools\rl\v400_fixture_download.py tests\test_v400_fixture_download.py
"COMPILE_EXIT=$LASTEXITCODE"
```

2. 工具与失败路径单测（退出码 **0**，`30 passed`；含“不一致必须重试一次并记为 mismatch”“HTTP 404 记为 failed 且不落盘”“并发 >4 直接拒绝”“已存在目录不得复用/删除”“总时限到期条目记为 deadline exceeded”等断言）：

```powershell
cd D:\WorkShop\Touhou_RL
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests\test_v400_fixture_download.py
"PYTEST_EXIT=$LASTEXITCODE"
```

实际输出：`30 passed in 0.70s`，`PYTEST_EXIT=0`。

3. 全量下载 + 逐份校验（退出码 **0**）：

```powershell
cd D:\WorkShop\Touhou_RL
python tools\rl\v400_fixture_download.py --manifest third_party\th08\config\replay-fixtures.json `
  --out-dir .tmp_v400_retail_fixtures_all --concurrency 4 --deadline-seconds 1500 --timeout-seconds 60
"DOWNLOAD_EXIT=$LASTEXITCODE"
```

实际输出末两行：`total=108 downloaded=108 verified=108 mismatch=0 failed=0`、`report=.tmp_v400_retail_fixtures_all\manifest_check.json`；`DOWNLOAD_EXIT=0`；墙钟 27.2 s。

4. 独立复算（退出码 **0**）：

```powershell
cd D:\WorkShop\Touhou_RL
python "$env:TEMP\verify_v400_fixtures.py"
"VERIFY_EXIT=$LASTEXITCODE"
```

实际输出：`independent_verified= 108 / 108`、`bad= []`、`report_counts= {'total': 108, 'downloaded': 108, 'verified': 108, 'mismatch': 0, 'failed': 0}`；`VERIFY_EXIT=0`。

## 边界与未改变项

- 本卡只做**输入获取与完整性校验**：不解码 `.rpy`，不生成 `rows.jsonl`/`complete.json`，不计算 `retailTrace.sha256`，不翻转 WP1 的 BLOCKED 结论。
- 未修改任何第三方/游戏原件/依赖；未删除任何既有文件；只新增本卡、`tools/rl/v400_fixture_download.py`、`tests/test_v400_fixture_download.py` 与 `.tmp_v400_retail_fixtures_all/` 目录内文件。
- 工具对 manifest 做了 fail-closed 校验（缺键、非法 sha256、非 HTTP url、`bytes` 非正整数、路径逃逸型文件名、重复 id/文件名一律拒绝），退出码 2；任一条 mismatch/failed 时工具退出码为 1，不伪造成功。
- 回放权限/出处仍归原作者（见 `third_party/th08/tests/fixtures/replays/th08-1.00d/README.md`）；本地保存与校验不构成所有权声明。
- 下一入口（不在本卡范围）：用这批 108 份输入做 WP1/G0.1 规模验证时，仍需独立的零售参考轨迹来源来核验 `retailTrace.sha256`，不得以本卡语料替代。
