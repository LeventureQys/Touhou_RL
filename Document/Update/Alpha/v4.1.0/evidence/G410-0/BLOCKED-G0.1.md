# G410-0 / G0.1 外部阻塞（桌面锁屏）

- 时间：2026-11-13T01:05:00+08:00
- 状态：**blocked on external desktop session**，不是代码失败。
- 当前 G0.1：90/108 identical、0 mismatch、18 failed、0 timeout。
- 18 个 exhausted fixture：`hard-8-b`、`hard-10-a`、`hard-10-b`、`hard-11-a`、`hard-11-b`、`lunatic-0-a`、`lunatic-1-a`、`lunatic-1-b`、`lunatic-2-a`、`lunatic-3-a`、`lunatic-4-a`、`lunatic-5-a`、`lunatic-6-a`、`lunatic-7-a`、`lunatic-8-a`、`lunatic-9-a`、`lunatic-10-a`、`lunatic-11-a`。

## 1. 证据

- 批量报告：`.tmp_v400_trace_batch/report.json`；journal：`.tmp_v400_trace_batch/journal.jsonl`。
- 单 fixture 重试：
  - `easy-2-b`：attempt 2 **ok**（说明失败并非数据本身错误）。
  - `hard-8-b`：attempt 3（`--max-attempts 5`）仍 `no row was captured: PermissionError [WinError 5]`，日志 `.tmp_v410_g01_hard8b2.log`。
- 桌面诊断：`LogonUI.exe` 正在运行 → Windows 桌面处于锁屏；`GetForegroundWindow` 无法取得有效窗口。零售件导航依赖 `SendInput` 前台焦点，锁屏时 `force_foreground`/导航必然失败。

## 2. 为什么这不是代码问题

- 同一批 108 个 fixture 中 90 个已经 identical；`easy-2-b` 重试成功。
- 失败集中在需要更复杂导航的 hard/lunatic fixture，且错误是 Win32 `WinError 5`（访问/前台权限），不是 trace hash mismatch。
- 锁屏状态下任何 `SendInput` 前台导航都不能可靠工作；继续重试不会改变结果。

## 3. 最小解阻动作

解锁交互式 Windows 桌面，保持解锁状态，然后运行：

```powershell
cd D:\WorkShop\Touhou_RL
python tools\rl\v400_retail_trace_batch.py --root .tmp_v400_trace_batch --max-attempts 5
```

预期：18 个 exhausted fixture 逐个串行重试；成功则 G0.1 108/108；仍失败则保存新 journal/attempt 证据，不走 `--skip-failed` 掩盖。

## 4. 影响范围

- 阻塞：G410-0 的 108/108 保真腿。
- 不阻塞：WP02 教师审计（WSL shadow 引擎 / 只读数据）、WP01 已通过证据、后续 WSL 实验。
- G410-0 状态仍为 `in_progress`；不得因为 90/108 而写成 passed。
