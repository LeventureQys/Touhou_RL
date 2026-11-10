# EXP-P0-042 — WP6/WP7 human_reference 真实回放路径阻塞报告

- 日期：2026-11-08（America/Los_Angeles）
- Provider / model：DeepSeek / DeepSeek v4.1 flash
- 结论：**BLOCKED / FAIL-CLOSED**
- 范围：本轮仅新增本报告；根目录同名报告为原件并保留。未修改第三方、游戏 EXE、依赖或既有 human-reference 实现；未删除、未提交。

## 目标与真实性门

WP6/WP7 需要解除 `human_reference` 缺失，且只能使用已构建 Linux TH08 harness 播放真实零售 `.rpy` 后取得逐面真实 gameplay frames、deaths、bombs。不能把 replay codec 的 `inputRecords` 或 `inputRecords` 减尾部记录推导为 gameplay frames，也不能从 `inputRecords` 捏造 death/bomb 成绩。

## 已核实的现成 autoplay 路径

现成可复制路径为 `third_party/th08/scripts/capture-replay.py`，而不是仅读取 codec 的 `replay_file.py`：

```powershell
python third_party/th08/scripts/capture-replay.py --help
```

退出码：**0**。帮助确认：`--replay REPLAY`（通过 Replay 菜单加载外部原版 1.00d replay）、`--start-stage START_STAGE`、`--playback-mode {0,2}`（普通播放/零售内置快进）、`--observer-backend {gdb,ptrace}`、`--timeout TIMEOUT`。

WSL 存在性检查：

```powershell
wsl.exe sh -lc 'ls -l /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern /root/v400-data/game 2>&1; printf EXIT=%s\\n $?'
```

退出码：**0**。binary 存在，大小 **4,807,320 bytes**；`/root/v400-data/game` 存在并包含 `th08.dat`、`thbgm.dat` 等数据。

## 观测能力审计

`replay-external-observer.py` 的真实播放观测会在 playback calculation boundary 读取 replay manager frame，输出 `rows.jsonl` 逐 calculation-frame 状态及 `complete.json`（包括 `stageFrames`、`endScores`、stage sequence 与 replay-tail 消耗校验）。但它没有 per-stage 权威 `deaths` 或 `bombs_used` 计数/事件采集；没有 death counter/reset 语义确认，也没有 bomb-use observer。`inputRecords` 只用于 replay 消耗与零售尾部校验，不是 gameplay frame 数，更不是 death/bomb 测量。

因此 autoplay 即使成功，现有 observer 最多提供真实 `stageFrames`/状态轨迹，不能提供 WP6/WP7 所需完整 human reference。

## 既有零售采集证据

EXP-P0-037 的真实零售采集诊断显示：metadata/attestation 与 clock-ready 已建立，但首个 gameplay capture breakpoint 未命中，`rows.jsonl` 为 0 bytes，`complete.json` 缺失；外层有界运行以 **exit 124** 超时结束。该证据不能改写为成功，也不能用空结果或 codec 长度补 metrics。

## 阻塞与收束

阻塞条件为：（1）现成 observer 无 per-stage death/bomb 权威观测；（2）当前 Wine/ptrace 零售路径在首 playback boundary 前有界超时（exit 124、0 rows），不可无界重试。解除需要已授权真实零售调试环境、绑定 replay/EXE/data hashes 的完整 rows/events、每面 death/bomb 事件或权威 counters（含 reset 语义）、stage boundaries 与 completion 证据。

本实验不产生 human reference，不修改既有实现，不把 inputRecords 当 gameplay/death/bomb 数据。当前保持 `human_reference` 缺失状态。
