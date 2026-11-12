# EXP-P0-079（v4.0.0 G0.1）：7 份 `no row was captured` 的纯离线取证

- 日期：2026-11-11 22:26（+08:00）；写卡时批跑 pid 4944 仍在跑 `lunatic-8-b attempt-3`
- 范围：**只读** `.tmp_v400_trace_batch/**`、`third_party/th08/config/replay-fixtures.json`（本仓没有 `.../accepted/manifest.json`）、`tools/rl/v400_retail_trace_{debugapi,batch}.py`；**未启动零售游戏/引擎、未改代码、未改 journal、未删文件**
- 本次新增（仅诊断）：`.tmp_v400_g01diag/{collect,timeline,runfacts,manifest_facts,evidence}.py` + `fixtures.json / evidence.json / manifest-facts.json / runfacts.json / table.md` + 本卡
- 触发：journal 53 行 = 41 `ok` + 12 `failed`（其中 **11 条 `no row was captured`** 落在 7 份 fixture，1 条是 `easy-2-b` 的旧 `navigation hang`）；去重后 48 份 fixture 有终态

## 1. 结论（TL;DR）

| 问题 | 数据给出的答案 |
| --- | --- |
| 7 份的共同点 | **全部死在导航**，一行都没写：`playbackStarted=false`、`navPath=null`、`rows=0`、`rows.jsonl`=0 B、`keyTaps`=108–293（成功 8–10）、`blindFallbackAt≈40.0 s`、耗时 92–101 s、`exit 6`、`gameExitCode=1` |
| 是 stage / 难度 / 预算吗 | **全都不是**：H1/H2/H3 均被数据否决（§4） |
| 是「没进回放」吗（H4） | **是**。`status.json.phase` 停在 `navigating`；`finalProbe flags=14`（isDemo+isReplay）、`stage=5`、`gameFrame≈3000` → 那是**标题吸引 demo**，不是回放 |
| 直接原因 | 按键**从未到达零售窗口**：标题菜单 25 s 停在 `screen=0 substate=1 cursor=0 timer=1` 不动；随后 `TITLE_SCREEN` 指针失效、读数变垃圾（`screen=268374015`、`cursor=1347565652`）；40 s 进盲序列、90 s 撞 nav 截止 |
| 与 EXP-P0-075 的关系 | 上一卡把无上限前台转移改成 `--foreground-timeout`（默认 8 s），但超时后 **fail-open**：`continuing without it`（`debugapi.py:2356-2362`）。而 `SendInput` 只在前台窗口生效（`debugapi.py:176-178`）→ 这条路径**极易**空转 90 s（本次 10/11 次） |
| 相关性 | 11 条有 report 的失败里 **10 条** `parallel.foregroundOk=false`；41 条 `ok` 里 **40 条** `true`（`easy-0-a` 是旧构建无该字段）。不是 100% 决定论：`lunatic-8-b attempt-1` 前台超时却仍进回放（见 §3 注） |
| 两个反例（已记录，不掩盖） | `hard-10-b attempt-1`：`foregroundOk=true` 仍失败（快照为真但键仍没进菜单）；`lunatic-1-b attempt-2`：`totalHits=0`、`finalProbe` 全 0（引擎主线程被冻住，`AttachThreadInput` 卡死的另一面） |
| 现在能否跑通 | **能**：`lunatic-8-b attempt-3` 22:11:46 起，`foregroundOk=true`，22:11:54 进回放，22:30:48 已 92,160/143,679 行（≈80.8 行/s，ETA ≈22:41） |

## 2. 失败清单（journal 原样）

| fixture | attempt | beganAt → endedAt | elapsed | exit | 行数 | error |
| --- | --- | --- | --- | --- | --- | --- |
| hard-8-b | 1 / 2 | 11-10 02:24:03 → 02:25:44；19:58:33 → 20:00:14 | 99.9 / 101.2 | 6/6 | 0/0 | no row was captured |
| hard-10-b | 1 / 2 | 02:54:11 → 02:55:43；20:00:14 → 20:01:56 | 92.0 / 101.3 | 6/6 | 0/0 | no row was captured |
| hard-11-b | 1 / 2 | 02:55:43 → 02:57:23；20:01:56 → 20:03:37 | 99.7 / 101.0 | 6/6 | 0/0 | no row was captured |
| lunatic-1-b | 1 / 2 | 03:21:04 → 03:22:44；20:03:37 → 20:05:19 | 99.0 / 101.4 | 6/6 | 0/0 | no row was captured |
| lunatic-8-b | 2（1 无 journal） | 20:05:19 → 20:07:01 | 101.1 | 6 | 0 | no row was captured |
| lunatic-9-b | 1 | 20:07:01 → 20:08:42 | 101.1 | 6 | 0 | no row was captured |
| lunatic-10-b | 1 | 20:08:42 → 20:10:24 | 101.0 | 6 | 0 | no row was captured |

四条 20:0x 的尝试来自同一次会话（`batch5.stdout.log` 19:58:32 起，`resume: 41 terminal of 108`），7 份连续全挂；`hard-10-b/11-b/8-b/lunatic-1-b` 现已被批跑标 `exhausted`（2 attempts 用完）。

## 3. 证据摘要（每份 attempt 目录里到底有什么）

统一形态（11 个 attempt 全部一致）：只有 5 个文件 —— `report.json` 29.9–31.1 KB、`debugger.log` 1.26–1.62 KB、`status.json` 880–895 B、`complete.json` 239–246 B、`rows.jsonl` **0 B**；另加 `rundir/`（th08.exe/dat/bgm + 复制的 `th8_01.rpy`）。除 `debugger.log` 外**没有单独的 observer stdout 捕获**，`alignment.json` 在整棵树里 **0 个**。

| fixture | replay | stages / 开始 stage | frames | budget | taps | blindAt | fgOk | hits(filtered) | finalProbe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hard-8-b | th8_ud24f9.rpy | [0,1,2,3,5,7] / **0** | 143785 | 3176 | 108→114 | 40.01 | F,F | 6232(3075) / 6063(2990) | flags14 stage5 frame≈3000 |
| hard-10-b | th8_ud2ac5.rpy | [0,1,2,4,5,7] / **0** | 134771 | 2996 | 110→114 | 40.02 | **T**,F | 6090(3004) / 6073(2995) | flags14 stage5 |
| hard-11-b | th8_ud2854.rpy | [0,1,2,4,5,7] / **0** | 121434 | 2729 | 110→114 | 40.03 | F,F | 6214(3066) / 6070(2994) | flags14 stage5 |
| lunatic-1-b | th8_ud2c70.rpy | [0,1,2,3,5,7] / **0** | 106063 | 2422 | 108→**293** | 40.00 | F,F | 6266(3092) / **0(0)** | flags14 → **flags0 stage0** |
| lunatic-8-b | th8_ud2cc9.rpy | [0,1,2,3,5,7] / **0** | 143679 | 3174 | 114 | 40.01 | F | 6068(2993) | flags14 stage5 |
| lunatic-9-b | th8_ud2945.rpy | [0,1,2,3,5,7] / **0** | 132477 | 2950 | 114 | 40.01 | F | 6066(2992) | flags14 stage5 |
| lunatic-10-b | th8_ud2cac.rpy | [0,1,2,4,5,7] / **0** | 113660 | 2574 | 114 | 40.03 | F | 6056(2987) | flags14 stage5 |
| 对照 normal-0-b | th8_ud2cd5.rpy | [0,1,2,4,5,7] / 0 | 100979 | 2320 | **8** | — | T | 202453(0) | flags8712 stage7 |
| 对照 hard-0-b | — | [0,1,2,4,5,7] / 0 | 122914 | 2759 | **10** | — | T | 246322(0) | flags10760 stage7 |
| 对照 lunatic-0-b | — | [0,1,2,4,5,7] / 0 | 113221 | 2565 | **10** | — | T | 226936(0) | flags11016 stage7 |
| 对照 hard-9-b（同小时） | th8_ud2469.rpy | [0,1,2,3,5,7] / 0 | 129463 | 2890 | **10** | — | T | 259422(0) | flags11016 stage7 |

`hits(filtered)` 的差值说明失败 run 里那 6000 次断点命中全是 **demo 帧**，被 `isDemo` 过滤（`trace: row read mode=merged` 后有命中、`rows=0`）。`complete.json` 的 `errors` 统一是 `['phase budget exhausted before the replay ended', 'Stage sequence [] differs from [...]']` —— 前一句是误导：它只是 `debugapi.py:2482` 在导航线程自然结束后补写 `collector.finish(None, ...)` 的固定文案。

注（反例，必须保留）：`lunatic-8-b/`（=attempt-1，**无 journal 记录**，批跑当时被中断）留下 `status.json: foregroundOk=false, foregroundTimedOut=true, playbackStarted=true, navPath=closed_loop, rows=127,703` 与 23,373,263 B 的 `rows.jsonl`、`debugger.log` 0 B —— 说明**前台超时后键也可能进去**，所以 F1 的价值是「把不可靠路径变成显式可重试」，而不是声称必然失败。

## 4. 假设检验（H1–H4）

| 假设 | 判定 | 数据 |
| --- | --- | --- |
| H1 这些 replay 没有 stage 0 → `--start-stage 0` 导航失败 | **否决** | 7 份的 `stages[0]` 全是 **0**；全清单 108 份里只有 12 份 Extra（`stages=[8]`）非 0，且不在失败集；`report.arguments.start_stage=0` 与 `batch5` 启动日志「start-stage: passed from each fixture's first manifest stage index」一致 |
| H2 某个 stage 太长，预算内没走到第一行 | **否决** | 失败耗时 92–101 s，预算 2422–3176 s（20–30 倍余量）；最大的 stage 记录数 38,779（60 fps 计 ≈11 min 回放），而 demo 在 29–38 s 就开场；同预算下更大的 `normal-8-b`(148,719)/`lunatic-3-b`(149,986) 都跑完了 |
| H3 难度/Extra 菜单不匹配 | **否决** | 失败集 = Hard(3) + Lunatic(4)，`difficulty` 与清单 `difficultyName` 逐份吻合（Hard=2 / Lunatic=3），`extended/practice` 全 False，无 Extra（`stages=[8]` 的 12 份 Extra 一份都没跑）；同难度成功样本 hard-9-b、lunatic-2-b 用同一导航序列 |
| H4 observer 没进入回放（停在 attached/navigating） | **成立** | 11/11 `playbackStarted=false`、`phase=navigating`（status.json）、`navigation.path=null`、`menuLog` 停在 title；成功样本 100% `closed_loop` 且 5.0–5.4 s（窗口出现后）即进回放 |
| H4 的子原因：按键投递失败 | **成立（主因）** | 10/11 `foregroundOk=false`（`warning: foreground transfer still blocked after 8 s (EXP-P0-075)`）；标题菜单 25 s 零位移 + 菜单指针失效；盲序列（4×down、2×z、2×down、1×z）同样无效 |

## 5. 根因链（对着代码段）

1. `debugapi.py:2354-2362` 前台转移有界化 → 超时**只 warning**，`foreground_ok=false` 仍继续 → `2373` 启动 Navigator。
2. `SendInput` 只有目标窗口在前台才落到引擎（`debugapi.py:176-178`，EXP-P0-073 结论；`PostMessage` 在本引擎无效）→ 键落空。
3. `Navigator._closed_loop`（`1295-1343`）读 `title_state()`（`1226-1241`）：`state/sub_timer` 让 `menu_key()`（`679-693`）返回 `None` 时不发键；菜单指针失效后读数变垃圾，`screen/substate` 无范围校验 → 被当成菜单，`1309-1317` 的「ESC+Z 退出 demo」分支反而很少触发。
4. 40 s（`closed_timeout`）→ `_blind()`（`1345-1359`）→ 35 s 再闭环 → 线程自然结束（`1280-1285`）→ 主循环 `2461-2464` 打 `navigation thread ended without playback` → `2482` 补写 `phase budget exhausted...` → `_grade` 落 `no row was captured` + `EXIT_STATE=6`。
5. 期间引擎已空转进吸引 demo（`stageStart stage=5 rngOrigin=90`，`flags=14`），`isDemo` 把帧全过滤，`rows` 永远 0。

## 6. 修复建议（只给方案，未改代码）

- **F1（必须）禁止 fail-open**：`foreground_ok=false` 时不要继续导航。改成：① 在 `--foreground-timeout` 内**重试转移**（例如 3 次 × 10 s，每次之间 `_nudge_input()`）；仍失败则直接 `EXIT_BLOCKED(3)`/新增可重试状态，`reason="foreground not acquired"`，而不是耗 90 s 后报 `no row`。这样批跑的重试策略才有意义（现在 4 份已被判 exhausted）。
- **F2（必须）加输入活体门**：进入闭环前发 1 次 `down`（或 `z`），3 s 内要求 `title_state().subTimer/cursor/state` 变化；不变则判 `input not delivered` 立刻退出（exit 3/7），完全不进 40+35 s 空转、也不会触发吸引 demo。
- **F3（建议）菜单读数校验 + demo 恢复**：`screen<=16 && substate<=3 && cursor<=16 && state<=4` 之外的读视为「不在标题」，直接进 ESC+Z 恢复并重开闭环（最多 2 轮）；同时用 `playback_state().isDemo` 作为「demo 正在进行」的显式信号。
- **F4（建议）诊断字段**：把 `foregroundTransfer` 永远写进 report（现在只有**超时**才写），并把 `hits.filteredHits`、`menuLog` 全量（现只存 40 条）、`finalProbe.flags` 一并入 journal，便于以后一眼区分 `no row` 的两种亚型（keys-never-landed / engine-wedged）。
- **F5（批跑）重试语义**：把「输入/前台失败」从 `failed` 里分出来（独立 status 或 error code），`--max-attempts` 对它单独计数；否则 4 份已 exhausted 的 fixture 只能人工重跑。

## 7. 重跑这 7 份需要什么命令

前提（关键）：**交互式、未锁屏的桌面；跑的时候不要有别的窗口抢焦点**；不要与其他 fixture 并行。（下面用 **cmd** 语法；PowerShell 把 `set X=`→`$X=`、`%X%`→`$X`、`^`→反引号。）

```
set R=D:\WorkShop\Touhou_RL
set PY=C:\Users\a5935\AppData\Local\Programs\Python\Python312\python.exe
%PY% -u %R%\tools\rl\v400_retail_trace_debugapi.py --phase rows --out %R%\.tmp_v400_trace_batch\hard-8-b\attempt-3 ^
  --replay %R%\third_party\th08\tests\fixtures\replays\th08-1.00d\accepted\th8_ud24f9.rpy ^
  --budget-seconds 3176 --start-stage 0 --foreground-timeout 30 --nav-deadline 180 ^
  --status-json %R%\.tmp_v400_trace_batch\hard-8-b\attempt-3\status.json
```

| fixture | `--replay` | `--budget-seconds` | `--start-stage` |
| --- | --- | --- | --- |
| hard-8-b | th8_ud24f9.rpy | 3176 | 0 |
| hard-10-b | th8_ud2ac5.rpy | 2996 | 0 |
| hard-11-b | th8_ud2854.rpy | 2729 | 0 |
| lunatic-1-b | th8_ud2c70.rpy | 2422 | 0 |
| lunatic-8-b | th8_ud2cc9.rpy | 3174 | 0（attempt-3 正在跑，勿重复） |
| lunatic-9-b | th8_ud2945.rpy | 2950 | 0 |
| lunatic-10-b | th8_ud2cac.rpy | 2574 | 0 |

- 想一次交给批跑：`cd /d %R% && %PY% tools\rl\v400_retail_trace_batch.py --root .tmp_v400_trace_batch --only hard-8-b,hard-10-b,hard-11-b,lunatic-1-b,lunatic-9-b,lunatic-10-b --max-attempts 3`（前 4 份已 exhausted，必须抬到 3；`lunatic-8-b` 等 attempt-3 结束后再决定）。
- 前台门自检（开跑后 20 s 读一次）：`status.json.foregroundOk` 必须是 `true` 且 `playbackStarted` 应为 `true`（成功样本 5–8 s 就进回放）；否则 kill 掉重来，别等 90 s。

## 8. 复核命令与退出码（本次全部只读）

| 命令 | 退出码 |
| --- | --- |
| `python .tmp_v400_g01diag\collect.py` / `timeline.py` / `manifest_facts.py` / `evidence.py`（§3 表与时间线） | 0 / 0 / 0 / 0 |
| `python .tmp_v400_g01diag\runfacts.py`（每份 report 的 titleS/foreground/nav/rows 表） | 首次 1（脚本 key 名笔误）→ 修正后 0 |
| `Get-Content third_party\th08\tests\fixtures\replays\th08-1.00d\accepted\manifest.json` | 1（文件不存在：清单实际在 `third_party/th08/config/replay-fixtures.json`） |
| 读 `journal.jsonl` / `batch5.stdout.log` / `batch.log` / 每份 `report.json`、`debugger.log`、`status.json`、`complete.json` | 0 |

一句话：**这 7 份不是回放、stage、难度或预算的问题，是「键没进零售窗口 → 导航 90 s 空转 → demo 抢跑 → 0 行」；修 fail-open + 加输入活体门，然后在独占前台的环境里重跑。**
