# EXP-P0-037 — 官方 Wine ptrace 0 rows 首断点前停点与替代路径设计

- 日期：2026-11-08（America/Los_Angeles）
- Provider / model：DeepSeek / DeepSeek v4.1 flash
- 范围：纯只读证；只新增本记录与 `.tmp_v400_retail_diag_exp037.json`，未改零售 EXE、官方采集器、第三方、系统配置、依赖，未删除/提交。

## 原件与 fixture 核验

证据文件 `.tmp_v400_retail_diag_easy0_display0/metadata.json` 与 `observer.json` 对同一运行给出：

| 对象 | 大小 | SHA-256 | 结论 |
|---|---:|---|---|
| `third_party/th08/resources/th08.exe` | 840704 B | `330fbdbf58a710829d65277b4f312cfbb38d5448b3df523e79350b879213d924` | 与冻结官方原件一致 |
| easy-0-b replay | 30418 B | `79739491e1cb1d735ef9a14814a79dfe8e2c4872d5f2bf0174c79f40ada36989` | 与 fixture 一致 |
| `th08.dat` | — | `9d7edf43b8ddd347cbb641836f6b5050745dd936f688daebbf9382ca557043bb` | 运行 metadata 已记录 |
| `thbgm.dat` | — | `2c2ef05f9ff6f43f752dae5da519a2477372c3f8875b7b44996a8f69fdad4d89` | 运行 metadata 已记录 |

`observer.json` 还验证了 EXE 版本 `0100d`、checksum `2724749753`；这不是用研究构建替代原件。来源授权仍未由文件 hash 建立，故不把本次变成合法授权结论。

## 首断点前停点与 Wine 证据

官方 launcher 命令及壳层结果已记录于 `Document/Update/Alpha/v4.0.0/experiments/EXP-P0-014-WP1-retail-capture-diagnostic.md`：独立 `WINEARCH=win32`、`DISPLAY=:0`、Wine `6.0.3`、Linux ptrace hardware execute breakpoints，外层有界 `timeout 150s` 返回 **124**。

可定位的停点如下：

1. `metadata.json` / `observer.json` 已生成，observer 已完成地址/字节 attestation；frame boundary 是 **0x00452490**，completion boundary 是 **0x00441F52**。
2. `clock-ready` 存在，说明启动器已到时钟/运行时准备阶段。
3. `rows.jsonl` 为 **0 bytes**（空 SHA `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`），`menu/rng/score/item/effect` 五类事件也全为 0；`complete.json` 缺失。
4. 因而首个 gameplay capture breakpoint **0x00452490 一次也未命中**；completion breakpoint **0x00441F52** 也未命中。停在“metadata/attestation + clock-ready 之后、首个 playback-control entry 之前”，不是“X 不可用”的泛化判断。
5. Wine 侧可归因错误是**未在 120 秒内达到 observer 边界**，最终由外层 timeout 杀死（exit 124）；现有证据没有 Wine fatal exception、page fault 或 GDB signal。故不能编造“Wine 崩溃”。这是 Wine32/WSLg `:0` 下运行/观察链未进入首断点的超时阻塞；与私有 Xvfb 路径此前同样 0 rows 的结果一致。

说明：仓库的 `REPLAY_TESTING.md` 明确规定 ptrace 使用与 GDB 相同的 Linux 硬件执行断点，并规定首个观测位置为 `ControlPlaybackFrameAdvance` 入口、终端位置另设 completion breakpoint。因此本记录把 ptrace observer log 作为官方 GDB-equivalent 断点证据，不把无对应原始 GDB transcript 的内容冒充 GDB 输出。

## 有界替代采集路径（只在真实资源可用时尝试）

按优先级设计，不自动伪造轨迹：

1. **Windows 原生零售帧栅栏（首选）**：在已授权的 Windows 采集机上，使用同一冻结 EXE、同一 replay/data/config，外部调试器在已 attested 的 playback-control 与 completion 地址设硬件断点；每次只允许单例、固定 120 s/fixture 上限，输出 `rows.jsonl`、`complete.json`、metadata 和 SHA。若首断点 0 次命中则 fail-closed，不尝试第二 fixture。
2. **Wine32 独立软件 X 显示**：仅当真实 `wine`, `wineboot`, 32-bit 支持、Xvfb/授权 DISPLAY 与 fixture/data 全部 preflight 为 present 且 hash 匹配时，使用新的独立 prefix 与新输出目录，单例 120 s；不能把 `DISPLAY=:0` 的 0-row 结果改写为成功。
3. **上游参考轨迹**：仅接受可追溯来源、schema v4、完整终端记录、fixture/data/EXE hash 全匹配的轨迹；不得用当前 `.rpy` 直接冒充零售执行轨迹。

本轮**未尝试**替代路径：Windows 原生调试资源/授权采集机未在当前文件证据中建立；现有 WSL preflight 证明 Wine 6.0.3、wineboot 与 xdotool 路径曾可用，但本次可复用的授权 DISPLAY/Xvfb、fixture/data 组合及完整采集门未同时建立，且官方路径已达到禁止无界重试的边界。未新增 rows，也未声称捕获成功。

## 下一条可执行实验命令（资源恢复后）

资源门通过后，先只跑 easy-0-b 一例；命令退出码 0 才允许检查 `rows>0`、`complete.json` 与 SHA，非 0 立即停止：

```powershell
wsl.exe -e bash -lc '
set +e
out=/mnt/d/WorkShop/Touhou_RL/.tmp_v400_retail_diag_exp037_wine32_fresh_20261108
prefix=/mnt/d/WorkShop/Touhou_RL/.tmp_v400_retail_diag_exp037_prefix_fresh_20261108
# Static safety gate: these exact absolute paths must not already exist; do not delete or overwrite old evidence.
test ! -e "$out" || { echo "REFUSE_EXISTING_OUT=$out" >&2; exit 10; }
test ! -e "$prefix" || { echo "REFUSE_EXISTING_PREFIX=$prefix" >&2; exit 11; }
export WINEPREFIX="$prefix" WINEARCH=win32 WINEDEBUG=-all DISPLAY=:88
timeout 120s python3 /mnt/d/WorkShop/Touhou_RL/tools/rl/v400_retail_py310_launcher.py --target /mnt/d/WorkShop/Touhou_RL/third_party/th08/resources/th08.exe --game-data /root/v400-data/game/th08.dat --bgm-data /root/v400-data/game/thbgm.dat --replay /mnt/d/WorkShop/Touhou_RL/.tmp_v400_retail_fixtures/th8_ud2cdc.rpy --output-dir "$out" --wine /usr/bin/wine --observer-backend ptrace --display :88 --timeout 100 --keep-runtime
ec=$?
test $ec -eq 0 || exit $ec
test -s "$out/rows.jsonl" || exit 2
test -s "$out/complete.json" || exit 3
'
```

预期退出语义：`10` = 输出目录已存在，拒绝执行；`11` = prefix 目录已存在，拒绝执行；`124` = 有界超时；`1` = launcher/observer 失败；`2` = exit 0 但首断点仍 0 rows（不可接受）；`3` = rows 有但未完成（不可作为完整轨迹）。只有 exit 0 且所有文件/hash/完整帧门通过，才可尝试 easy-1-b。

## 结论

**FAIL-CLOSED / EXP037 收束。** 真实原件与 easy-0-b fixture 身份已核验；官方 Wine ptrace 运行确实停在首个 playback-control 断点之前（0x00452490，0 rows），completion 也未命中，现象是有界超时而非已证实的 Wine 崩溃。替代路径已设计，但当前没有满足授权与工具门的真实资源，因此本轮不尝试、不伪造、不扩展。

## 副本声明

本文件是根目录 EXP-P0-037-wine-ptrace-boundary.md 的规范副本；原文件保留、不删。本副本不改变原记录。根因保持 **unknown**：现有证据只能确认首断点 0 hits、外层 timeout exit 124、rows 0，不能臆断 X。
