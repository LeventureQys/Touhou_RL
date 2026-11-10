# 引擎构建清单（v4.0.0）

- 日期：2026-11-10（T0 建立；T7 统一构建 `rl_v3` 时追加本表）
- 位置：WSL Ubuntu，`/root/Touhou_RL/third_party/th08-rl/build/`
- 源树：`third_party/th08-rl`，分支 `rl-harness`。截至本表，分支 HEAD 为
  `bef369aaaf87d1fcd1cb568d2e5284da65abbc57`（T0 提交，收口此前全部未提交改动）；
  其上的 9 个提交已导出为 patch，备份在
  [evidence/harness-patches](<evidence/harness-patches/>)（10 份，含 T0 提交本身）。
- 规则（任务书 §2.3）：既有 `build/rl*` 目录**不重建、不覆盖**；新能力一律用新目录名。
  T0 只读取这些目录，没有编译、没有改动任何二进制。

## 1. 构建一览

SHA-256 于 2026-11-10 用 WSL `sha256sum` 实测，与各实验卡片记录的前缀一致。

| 目录 | 二进制 SHA-256（实测） | 字节 | 构建时间 | 基线 commit | 相对上一个构建新增的能力 | 使用过它的卡片 |
|---|---|---|---|---|---|---|
| `build/rl` | `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb` | 4,807,320 | 2026-11-08 23:29 | `f563df5`（干净树） | 基线：`HELLO`/`RESET`（确定性 StageStart）/`STEP`/`PIXELS`/`STATE`/`HASH`/`SHM`/`READBACK`/`DEBUG`，25 字段 HASH | P0 引擎侧卡片与 bench 的主要构建（54 份文档引用）：EXP-P0-008/023/024/027/029/030/032(×2)/035/036/038/039/041/042/044/046/048/049/050/052/056/057/058/062/063/064/068/069/070、EXP-P1-002、`bench/*` |
| `build/rl_retrace` | `c2ce960dff8cf6d5eee4062c7c88cbad3c00c42dc4ed9c32d3af240819c6f79d` | 4,807,664 | 2026-11-08 23:29 | `f563df5` + 未提交的 `replay_trace.cpp` 纯记录补丁（该补丁已随 `bef369a` 收口） | 回放播放时的纯记录模式（25 字段追踪行落盘），用于差分回放 | EXP-P0-051、EXP-P0-055、G0-报告、P0-WP1–WP9 执行计划 |
| `build/rl_pbo` | `a3ae118159fd3fb24091513cfb236dc83e7c61b7d1334fc6b79376b5b94f2465` | 4,817,776 | 2026-11-09 03:01 | `f563df5` + 未提交的异步读回改动（`d3d8_compat.cpp`，已随 `bef369a` 收口） | `TH08_ASYNC_READBACK` 异步读回路径与诊断行（默认路径不变） | EXP-P0-061、EXP-P0-067 |
| `build/rl_hazard` | `8053dfaa3d5cde0f720ba31d7f63aeba1db929a6721999c1b95e8ac00ba7b32a` | 4,826,176 | 2026-11-09 10:41 | `f563df5` + 未提交的 `HAZARDS`/`PLAYER` 扩展（已随 `bef369a` 收口） | 逐帧致死物几何：子弹全宽判定、敌人 `worldPosition`+`hitboxDimensions`+`flags1` bit2、自机 `hurtboxBounds` 与机体速度表；批量 `HAZARDS`、`PLAYER` | EXP-P1-001（本构建）、EXP-P1-003、EXP-P1-004、EXP-P1-006；另被 EXP-P1-005/007/021、EXP-P2-001/005、G1-报告 引用 |
| `build/rl_hazard2` | `f172399f61b75baa8e62be46041ef756a710759f439e7fb07111f4747c78264e` | 4,827,408 | 2026-11-09 11:22 | `f563df5` + 未提交的 `EVENTS`/`CHILD` 扩展（已随 `bef369a` 收口） | 在上一个基础上加 `EVENTS`（death/bomb/game_over）与 `CHILD <handle> HAZARDS` 子句柄读取；25 字段 HASH 前缀不变 | EXP-P1-005（本构建）、EXP-P1-007（SHA 闸门）、EXP-P1-008、EXP-P1-014、EXP-P1-021；另被 EXP-P2-002/003/004/005、EXP-P3-003/006、G1-报告 引用 |
| `build/rl_laser` | `1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f` | 4,828,752 | 2026-11-09 20:59 | 同上 + 未提交的激光判定窗扩展（已随 `bef369a` 收口） | 给 `LASERS`/`HAZARDS` 行尾追加 6 个激光判定窗字段，教师侧按 `damageable` 过滤；命令集不变，HASH 前 100 帧与 `rl_hazard2` 逐帧一致 | EXP-P1-021（本构建）、EXP-P3-005、EXP-P3-007、G1-报告（G1 的教师 `v2h` 即在此构建上评测） |
| `build/rl_score` | `d3386c9536888a1721ebcc93d16839c40a66eeee2f25aead52a107e1d9370070` | 4,829,776 | 2026-11-10 00:40 | 同上 + 未提交的分数字段扩展（已随 `bef369a` 收口） | 追加 HUD 分数字段（`STATE`/`HASH` 前缀不变），供 G2 的分辨率相关判定使用 | EXP-P2-005、G2-报告（G2 的引擎即此构建）、P1–P6 执行计划 |
| `build/rl_events` | `77278a8986fceb67929208cc082d18a69f518fb9d58960258ae1115578c9097a` | 4,834,704 | 2026-11-10 02:11 | 同上 + 未提交的 `EVENTS 0x0080` 扩展（已随 `bef369a` 收口） | 在上一个基础上加 `EVENTS` 的 `0x0080` = `stage_clear` 位，G2 过面识别依赖它 | EXP-P2-007、EXP-P2-008、EXP-P2-009、G2-报告 |

`rl_hazard` 之后各构建的「新增能力」由实验卡片与源码共同确定：`HAZARDS`/`PLAYER`/`EVENTS`
是新增命令，可在二进制里直接看到（`rl`/`rl_retrace`/`rl_pbo` 没有 `HAZARDS`/`PLAYER`/`EVENTS`
字面量）；`rl_laser`/`rl_score`/`rl_events` 是既有回复行尾的**字段级**追加，命令集与
`rl_hazard2` 相同，差异证据来自卡片记录的字段表与 HASH 前缀一致性。

## 2. 出处与可复现性（重要）

- 表里的「基线 commit」是 `rl-harness` 分支上的**最后一个提交**；除 `build/rl` 外，
  其余构建都编译自该提交 + **当时尚未提交**的工作树改动。这些改动现在已全部收口到
  `bef369a`，因此 `bef369a` 是「功能的并集」，**不是**任何一个历史二进制的逐字节来源。
- 逐字节身份由 SHA-256 认定：不要用「从当前源码重建」的方式去核对历史二进制，
  也不要重建这些目录（任务书 §2.3）。`build/rl` 是唯一的例外——它对应干净的
  `f563df5`（不含 `replay_trace.cpp` 纯记录补丁）。
- `build/rl_pbo` 曾被重建过一次：EXP-P0-061 记录的 `296a89737aaf…` 是旧二进制，
  EXP-P0-067 与当前磁盘上的是 `a3ae118159fd…`。以本表为准。
- EXP-P0-051 §「构建目录处置记录（build/rl 事故）」记过一次 `build/rl` 被误重建又按原
  SHA 复原的过程；复原后的 `591c521d…` 与本文一致。

## 3. 复核命令

```bash
wsl.exe -d Ubuntu -- bash -lc 'sha256sum /root/Touhou_RL/third_party/th08-rl/build/rl*/th08-modern'
```

## 4. 其他存在的构建目录（不是 `rl*`，本表不登记其身份）

`build/{measure,o2,o2c,portable-linux-x86_64,portable-replay,portable-retrace}`。
其中 `portable-retrace` 是 EXP-P0-051/055 的差分对照臂（OFF 构建，`f920e140…`），
其余为更早的吞吐/移植实验产物。

## 5. 后续

T7.2 的统一构建 `build/rl_v3` 建立后，在本表追加一行：新 SHA-256、构建用的 commit、
合并了哪些能力（`rl_laser` 的判定窗、`rl_events` 的 `0x0080`、`rl_score` 的分数字段、
T7.1 的进程内重开），以及有哪些卡片开始使用它。
