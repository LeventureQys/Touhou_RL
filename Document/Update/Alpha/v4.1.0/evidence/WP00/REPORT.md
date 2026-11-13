# WP00 进展报告（WP00.0 / WP00.1 / WP00.2）

- 时间：2026-11-13T00:06:07+08:00
- 状态：**WP00 已 passed（带 warnings）**；WP00.0（状态与证据入口）、WP00.1（资产审计）、WP00.2（确认集曝光与谱系审计）已完成。**G410-0 仍是 not_started**，因为 WP01 一致性还没做。
- 证据：`verdict.json`（机读）、`assets.json`、`resummed.json`、`exposure.json`、`protocol.json`、`commands.jsonl`、`logs/`。

## 1. 本轮实际做了什么

1. 创建 `touhou_rl/v410/evidence.py`、`tools/rl/v410.py`、`tests/test_v410_evidence.py`。
2. 创建 `state.json`、`journal.jsonl`、`budget.jsonl`、`configs/v410/local.json`、`evidence/WP00/`。
3. `preflight` 只探测环境，不启动游戏；确认 Windows HEAD `ca1ae8b…`、WSL HEAD `d55c26a…`、`rl_v3` SHA `691bfd77…`、零售原件 SHA `330fbdbf…`、Python 3.12.10 / torch 2.14.1+cu132 / RTX 4070 SUPER。
4. 对 16 个必需资产逐项给出 verified/missing/invalid；额外登记 D0、v3 训练 metrics、split、v3 逐局评测、m1 逐局、G0.1 trace journal。
5. 从原始记录重算三条结论：D1 完整 gate、G1 S1N、G1 PSN。
6. 运行并记录：pytest 11 passed（exit 0）、compileall（exit 0）、--help（0）、preflight（0）、status（0）、check WP00（0）。

## 2. 关键发现（已写入 assets.json / resummed.json）

- **D1 完整 gate 存在且通过**：`.tmp_v400_d1/work/gate.full.json`，137 片、469,216 决策、1,876,864 帧、HASH alignment 1.0（938,432/938,432）、load gate passed（469,216 samples）、planned_missing=0。顶层 `report.full.json`/`report.full.md` 是 smoke 结果（8 片/16,000 帧），**不能当完整 gate**；这一点已在证据中显式标注。
- **manifest 147 条 = 137 个磁盘分片 + 10 个 `-iNNN` 重复/隔离条目**；总数只按 137 条计。WSL `/root/v400-data/d1` 是 175 条/约 495 MB 的陈旧镜像，不是 canonical。
- **G1 S1N 重算**：20 局、20/20 过面、270,640 帧、均值 13,532.0、0 死。
- **G1 PSN 重算**：8 面、4/8 过面、4 死、均值恰 0.500；每面 n=1，仍不满足新确认协议，也不构成 G410-1 多面腿。
- **G0.1 原始 journal**：109 行（90 ok、19 failed）；P3 快照写 90/108、18 failed。**存在一行/一次尝试的差异，必须在 G410-0 关闭前对账**；19 份失败均为导航期 PermissionError（前台/权限）。
- 两份检出不同且都脏：Windows HEAD `ca1ae8b`（preflight 报 542 个 dirty 项）、WSL HEAD `d55c26a`（34 项）。只登记，不清理。
- 执行代理在本次 WP00.0 中长时间未落盘，按 AGENTS.md §6 中断；剩余入口由主代理以最小必要处置完成。后续 WP 会恢复子代理执行，并在报告中注明。

## 2.1 WP00.2 确认集曝光与谱系（exposure.json）

- **训练谱系**：m1 = D0 seeds 1–16 + DAgger seeds 2000–2010；v3 = D1 137 片、seeds 1–25（train 96 片/332,377 决策，val 21/70,961，test 20/65,878）。没有任何训练 split 含 1000–1019 或 60000–60199。
- **开发集**：1000–1019 只用于 G1 S1N、P3 开发评测、WP7 dev20、EXP-P3-009 功效；没有训练重叠。
- **旧确认集**：60000–60199 只在 `suites.CONFIRM_SEEDS` 定义、bench 披露文本引用；没有 run/training 使用。但 `train_v3.py` 没有 seed-range 断言，这是必须在正式确认前补的 fail-closed 缺口。
- **污染记录**：EXP-P3-010 的 test seeds 15–16 正是 m1（`dual-s0d1seed15/16` = train）与 v3（`d1-s0d1-n3371-seed15/16` = train）的训练数据。该 probe 的 holdout 只是局部 holdout，**不能当项目级 clean holdout**；G410-6 必须改用 1040–1099 或新的未暴露范围。
- **预算核查**：已知 GPU 0.22 h、已知墙钟 34.8 h（下界）、已知磁盘 11.1 GB（2 行 null）；首轮上限 160 GPU-h / 150 GB 不是新增授权。D1 26 GB、D0 1.39 GB、v3 checkpoint 0.042 GB；D 盘剩余约 909 GB。

## 3. 下一步

- **WP01.0**：`protocol.py` / adapters / bench 的最小实现与协议测试。
- **WP01.1**：真实引擎 3 训练种子 × 3 重开 × 1,000 帧一致性切片；30 分钟小规模运行；信息边界测试。
- **WP01 完成后**才可能把 G410-0 从 not_started 改为 passed；之后才进入 WP02。

## 4. 复现命令

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python tools/rl/v410.py preflight --config configs/v410/local.json --out Document/Update/Alpha/v4.1.0/evidence/WP00
python tools/rl/v410.py status --state Document/Update/Alpha/v4.1.0/state.json
python tools/rl/v410.py check --wp WP00 --evidence Document/Update/Alpha/v4.1.0/evidence/WP00
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp00_pytest -q tests/test_v410_evidence.py
python -m compileall -q touhou_rl tools/rl/v410.py
```
