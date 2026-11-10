# EXP-P0-046 — WP6/WP7 真实人类参考数据：真引擎回放逐面测量

- 日期：2026-11-09（Asia/Shanghai）
- Provider / model：DeepSeek / DeepSeek v4.1 flash
- 结论：**MEASURED（2 份真实零售回放 × 6 面 = 12 面，全部 `complete=true`，12/12 运行 exit 0）**
- 数据来源：真实 ELF 引擎播放真实 `.rpy` 时 `STATE`/`DEBUG` 的引擎计数器；**没有**用回放文件 `inputRecords` 推导帧数/deaths/bombs
- 范围：仅新增/修改 `tools/rl/v400_human_reference_autoplay.py`、`tests/test_v400_human_reference_autoplay.py`、本文件、`.tmp_v400_human_autoplay*`。未改 WSL C++/协议/第三方/依赖，未改零售 EXE，未删除、未提交。

## 1. 目标与真实性门

WP6/WP7 需要“每面长度（帧）+ 逐面死亡数 + bomb 数”。本实验用真实引擎播放真实零售回放，逐帧读取引擎自己的计数器，作为唯一来源：

- 逐面帧数 = `GameManager.gameplayFrameCounter`（STATE `f3`）在该面结束时的值；
- deaths = `ZunGlobals.deaths`（STATE `f6`，f32 bits）的累计增量；
- bomb 数 = `ZunGlobals.bombsRemaining`（STATE `f8`，f32 bits）的下降事件数（= 引擎 `bombsUsed` 的语义）。

禁止项已遵守：没有把 `.rpy` 的 input-record 数、尾部记录差、“inputRecords - trailing”当作帧数，也没有从输入流推断 death/bomb 成绩；`replay_input_frames` 只作为**一致性交叉校验**单独列示。

## 2. 数据来源与身份绑定

| 项 | 值 |
|---|---|
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern`，ELF 64-bit LSB pie executable（x86-64），**4,807,320 bytes**，sha256 `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb` |
| 引擎构建时间 | 2026-11-08 23:29:25 +0800（WSL 文件时间） |
| 工作树提交 | `/root/Touhou_RL` = `d55c26a6c465b26f11e26ba4fc1bdb888e8b130b`；`third_party/th08-rl` = `f563df58c44037b44208c393a86540b434b0c83b` |
| 游戏数据 | `/root/v400-data/game`（含 `th08.dat`/`thbgm.dat`），`th08.dat` sha256 `9d7edf43b8ddd347cbb641836f6b5050745dd936f688daebbf9382ca557043bb` |
| Lunatic 回放 | `th08_replay/Th8_ud0059.rpy`，**44,253 bytes**，sha256 `62b02a172780d846f4109fd0dcc762692c5205692670c0f7400c2026c409a5e9`（difficulty=3 Lunatic，shot=10 Youmu） |
| Normal 回放 | `th08_replay/Th8_ud2970.rpy`，**41,398 bytes**，sha256 `e621086d814fba4838a2c84e4081162524c3c7e84d1bc4ca3f6621f9ca52d96f`（difficulty=1 Normal，shot=6 Marisa） |

回放以只读方式从 WSL `/mnt/d/WorkShop/Touhou_RL/th08_replay/` 打开；数据目录与引擎都在 WSL 内真实运行，角色为 `--rl-role shadow`（null-device，`HELLO role=shadow`）。

### 2.1 播放路径（源码事实，本次未改动）

`third_party/th08-rl/src/TitleScreen.cpp:191 StartDeveloperReplayIfRequested()` 读取 `TH08_AUTOPLAY_REPLAY` / `TH08_AUTOPLAY_STAGE`，执行 `g_GameManager.SetIsReplayWeird(TRUE)`、`replayMode = REPLAY_MODE_NORMAL`、`currentStage = <stage>`，然后 `g_Supervisor.curState = SupervisorState_GameManager`。随后 `GameManager::RegisterChain()`（`GameManager.cpp:942`）因 `flags.isReplay` 调用 `ReplayManager::RegisterChain(1, replayFilename)`，`ReplayManager::PlaybackInputAndFps()` 把回放输入写入 `g_GuiMessageInputCurrent`，`Player` 直接读它。

因此：**harness 的 `STEP <mask>` mask 不参与回放输入，`STEP 0` 只是“泵一帧”**；回放输入完全来自引擎播放路径。每面都以独立进程、独立 `TH08_AUTOPLAY_STAGE=<面>` 从该面起点开始播放，所以 `TH08_AUTOPLAY` 起点、面内资源与 deaths 计数都是该面自己的。

### 2.2 观测点（每帧，无抽样推断）

`poll_frames=1`：每一个引擎帧都发一次 `STEP 0 1`，随后读 `DEBUG`（`curState`/`setupState`/`stage`/`difficulty`/`shot`/`isReplay`）与 `STATE`（25 字段），即逐帧判定而非抽样推断。12 次运行共泵 **278,918** 个引擎帧；两个掉命面的完整逐帧采样另存为 JSONL（见 §9），其余面保留聚合值、事件表与运行前 40 帧的 trace。

说明：`touhou_rl.v400.gym.protocol.HarnessClient.state()` 不能解析真实回放（`STATE` 把 `f6`..`f12` 打成 `%08x` float bits，`int(value)` 会抛 `ValueError`），本工具的 `read_state_fields()` 自己按 `%08x` 解码；未修改协议模块。

## 3. 计数器定义与判据

| 指标 | 引擎字段 | 判据 |
|---|---|---|
| `gameplay_frames` | `f3` gameplayFrameCounter | 该面最后一个 gameplay 采样值。计数器每完成一帧 gameplay 逻辑 +1、从 0 起，故终值即该面逻辑帧数 |
| `replay_input_frames` | `f2` `ReplayManager::frameCounter` | 回放输入驱动的逻辑帧数；`stageClearSequenceActive` 后停止递增（`ReplayManager::PlaybackInputAndFps`），故小于 `gameplay_frames` |
| `stage_clear_frames` | `f3 - f2` | 面 clear 序列帧数（无输入驱动） |
| `engine_frames` | `f0` 引擎帧 + `f3` | 从该面第一帧 gameplay 到最后一个 gameplay 采样的渲染帧跨度 |
| `dialogue_fast_forward_frames` | `f3 - engine_frames` | 回放对可跳过对白做 3 倍快进（`ReplayManager::ControlPlaybackFrameAdvance` → `RESTART_FROM_FIRST_JOB`），逻辑帧多于渲染帧 |
| `deaths` | `f6` `ZunGlobals.deaths` | 相邻采样增加的累计和（`Player::UpdateDeathAndRespawn` → `AddToDeaths(1)`） |
| `bombs_used` | `f8` `ZunGlobals.bombsRemaining` | **下降事件计数** = 引擎 `bombsUsed` 语义：普通 bomb `AddToBombCount(-1)`、deathbomb `AddToBombCount(-2)` 或清零，都记为 1 次 |
| `bombs_stock_spent` | `f8` | 下降量之和（普通 bomb 1、deathbomb 2），`> bombs_used` 即为 deathbomb |
| `bombs_gained` | `f8` | 上升量之和（boss/道具补弹，`GameManager::CollectExtend`/`AddToBombCount`） |
| `bombs_reset_on_death` | `f8` + `f7` | 掉命那帧 `Player::UpdateDeathAndRespawn` 同时 `AddLives(-1)` 与 `SetBombCount(primaryShtFile->initialBombCount)`，该下降是死亡惩罚，**不计入** `bombs_used` |
| 交叉校验 | `f7` lives | 掉命数与 `lives_start - lives_end` 一致（含 extend 时由表内数据自行核对） |

## 4. 真实命令与退出码

工具（每个面一条命令，逐面独立进程）：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python tools/rl/v400_human_reference_autoplay.py --replay "D:\WorkShop\Touhou_RL\th08_replay\Th8_ud0059.rpy" `
  --stage 0 --poll-frames 1 --max-frames 60000 --deadline 400 `
  --output .tmp_v400_human_autoplay\final_lunatic_s0.json `
  --stderr .tmp_v400_human_autoplay\final_lunatic_s0.stderr.txt
```

WSL 侧实际子进程（工具内部构造，未手工改参数）：

```text
wsl.exe -d Ubuntu -- env -u DISPLAY SDL_AUDIODRIVER=dummy \
  TH08_AUTOPLAY_REPLAY=/mnt/d/WorkShop/Touhou_RL/th08_replay/Th8_ud0059.rpy \
  TH08_AUTOPLAY_STAGE=<stage> \
  /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern \
  --data-dir /root/v400-data/game --rl-control 0,1 --rl-role shadow
```

| 运行 | 命令退出码 | `status` | `complete` | wall (s) | fps | 帧数泵入 |
|---|---|---|---|---|---|---|
| Lunatic stage 0 | **0** | measured | true | 10.266 | 1612.3 | 16,552 |
| Lunatic stage 1 | **0** | measured | true | 10.422 | 1785.0 | 18,603 |
| Lunatic stage 2 | **0** | measured | true | 10.250 | 1941.0 | 19,895 |
| Lunatic stage 4 | **0** | measured | true | 13.109 | 1870.9 | 24,525 |
| Lunatic stage 5 | **0** | measured | true | 14.422 | 1890.5 | 27,265 |
| Lunatic stage 7 | **0** | measured | true | 17.812 | 1929.7 | 34,372 |
| Normal stage 0 | **0** | measured | true | 6.891 | 1919.6 | 13,228 |
| Normal stage 1 | **0** | measured | true | 10.000 | 1836.3 | 18,363 |
| Normal stage 2 | **0** | measured | true | 10.672 | 1808.1 | 19,296 |
| Normal stage 3 | **0** | measured | true | 17.890 | 1503.0 | 26,888 |
| Normal stage 5 | **0** | measured | true | 15.906 | 1581.6 | 25,157 |
| Normal stage 7 | **0** | measured | true | 24.532 | 1417.5 | 34,774 |

进入 gameplay 的引擎帧：237–328（约 0.15 s，远低于 30 s 界）；所有运行都在 `deadline`/`max_frames` 之前自然结束（`next_stage` 或 `finish_replay`）。

验收命令与退出码：

```powershell
wsl -d Ubuntu -e bash -c 'cd /root/Touhou_RL/third_party/th08-rl && ls -l build/rl/th08-modern'
# -rwxr-xr-x 1 root root 4807320 Nov  8 23:29 build/rl/th08-modern  → exit 0

$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_human_autoplay_t -q tests/test_v400_human_reference_autoplay.py
# 13 passed → exit 0
```

## 5. 逐面真实结果

面索引按 `ScoreDat.hpp` 的 `Stage` 枚举：0=Stage1、1=Stage2、2=Stage3、3=Stage4A、4=Stage4B、5=Stage5、6=Stage6A、7=Stage6B、8=Extra。

### 5.1 Lunatic（`Th8_ud0059.rpy`，Youmu，difficulty=3，shot=10）

| 面 | 名称 | gameplay_frames | replay_input_frames | 面clear帧 | engine_frames | deaths | bombs_used | bombs_stock_spent | bombs_gained | death重置 | lives | score_end | graze_end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Stage1 | **16,732** | 16,663 | 69 | 16,315 | **0** | **1** | 1 | 1 | 0 | 2→2 | 18,456,495 | 6,954 |
| 1 | Stage2 | **19,009** | 18,940 | 69 | 18,347 | **0** | **0** | 0 | 1 | 0 | 2→3 | 56,979,542 | 12,002 |
| 2 | Stage3 | **22,394** | 22,303 | 91 | 19,635 | **0** | **0** | 0 | 1 | 0 | 3→6 | 139,791,135 | 18,533 |
| 4 | Stage4B | **26,909** | 26,818 | 91 | 24,246 | **0** | **2** | 2 | 4 | 0 | 6→7 | 248,792,077 | 20,888 |
| 5 | Stage5 | **30,943** | 30,852 | 91 | 26,988 | **0** | **1** | 1 | 1 | 0 | 7→8 | 360,304,141 | 26,581 |
| 7 | Stage6B | **35,959** | 35,518 | 441 | 34,044 | **1** | **2** | 2 | 1 | 4 | 8→7 | 635,583,645 | 40,396 |
| 合计 | — | **151,946** | 151,094 | 852 | — | **1** | **6** | 6 | 9 | 4 | — | 635,583,645 | — |

### 5.2 Normal（`Th8_ud2970.rpy`，Marisa，difficulty=1，shot=6）

| 面 | 名称 | gameplay_frames | replay_input_frames | 面clear帧 | engine_frames | deaths | bombs_used | bombs_stock_spent | bombs_gained | death重置 | lives | score_end | graze_end |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Stage1 | **13,258** | 13,189 | 69 | 12,987 | **0** | **0** | 0 | 0 | 0 | 2→2 | 8,179,952 | 3,671 |
| 1 | Stage2 | **18,525** | 18,434 | 91 | 18,115 | **0** | **0** | 0 | 1 | 0 | 2→3 | 28,480,266 | 6,229 |
| 2 | Stage3 | **20,009** | 19,940 | 69 | 19,037 | **0** | **0** | 0 | 1 | 0 | 3→6 | 69,474,118 | 9,299 |
| 3 | Stage4A | **27,722** | 27,631 | 91 | 26,608 | **0** | **3** | 4 | 4 | 0 | 6→7 | 132,290,333 | 13,855 |
| 5 | Stage5 | **25,675** | 25,606 | 69 | 24,886 | **1** | **0** | 0 | 2 | 2 | 7→8 | 203,568,518 | 19,714 |
| 7 | Stage6B | **36,403** | 35,903 | 500 | 34,472 | **0** | **0** | 0 | 1 | 0 | 8→8 | 396,652,042 | 29,036 |
| 合计 | — | **141,592** | 140,703 | 889 | — | **1** | **3** | 4 | 9 | 2 | — | 396,652,042 | — |

### 5.3 事件明细（括号内为 gameplay 帧 `f3`；标 engine 者为引擎帧）

- Lunatic Stage1：`bomb` f3=2（bombs 3→2；该面 input 流第 1 条非零记录即为 shoot+bomb，codec 交叉核对仅用于解释该事件，不参与计数）；`bomb_gained` f3=16,663（2→3，面结束补弹）。
- Lunatic Stage4B：`bomb_gained` f3=4,397（5→6）、f3=4,413（6→7）；`bomb` f3=9,856（7→6）、f3=13,662（6→5）；`bomb_gained` f3=14,441（5→6）、f3=14,461（6→7）。
- Lunatic Stage6B：`bomb` f3=4,462（7→6）；`bomb_gained` f3=7,020（6→7）；`death` f3=9,663（deaths 0→1，engine 8786）；`death_bomb_reset` f3=9,679（7→3，lives 8→7，engine 8802）；`bomb` f3=9,714（3→2，engine 8837）。
- Normal Stage4A：`bomb` f3=12,985（7→6）；`bomb` f3=16,634（**8→6，deathbomb**，故 `bombs_used=3` 而 `bombs_stock_spent=4`）；`bomb` f3=17,679（6→5）。
- Normal Stage5：`death` f3=4,082（0→1）；`death_bomb_reset` f3=4,098（5→3，lives 7→6）；`bomb_gained` f3=10,351（3→4）、f3=10,480（4→5）。

## 6. 交叉校验（不是数据来源）

1. **回放输入帧 vs codec 记录数**：该 `.rpy` 的 stage0 输入流为 16,665 条记录，引擎 `f2` 在面 clear 前停在 16,663，差 2 条与 retail 尾部记录 `[2,3]` 一致；其它面同样只差尾部 2–3 条（Lunatic 6 面 `f2` = 16663/18940/22303/26818/30852/35518，Normal 6 面 = 13189/18434/19940/27631/25606/35903）。此对照只用于验证“引擎确实按录制输入播放”，**未**参与任何帧数计算。
2. **deaths vs lives**：Lunatic Stage6B deaths=1 ↔ 事件表内 lives 8→7（无 extend）；Normal Stage5 deaths=1 ↔ 事件表内 lives 7→6，面末 lives=8（后续两次 extend），故净增 1，可按事件顺序复核。
3. **bombs 记账恒等式**（工具内强制）：`bombs_start - bombs_end + bombs_gained = bombs_stock_spent + bombs_reset_on_death`，任一不成立即报错退出 3，不产出数据。
4. 重复运行一致性：stage0 Lunatic 的 `gameplay_frames=16732`、`deaths=0`、`bombs_used=1`、`score_end=18,456,495` 在 `poll_frames=1` 的两次运行中完全一致（`poll_frames=2` 的一次早期探索运行得到 16,731，差 1 帧，原因是最后一个 gameplay 采样落在面切换前 1 帧：这正是把 `poll_frames` 固定为 1 的原因，正式数据全部为 `poll_frames=1`）。

## 7. 失败关闭条件（工具内置）

`play_stage()` / `aggregate()` 在下列情况抛 `AutoplayUnavailable`，CLI 退出码 **3**，报告 `status=unavailable`、`human_reference_created=false`，并保留 `--samples-output` 原始帧证据：

- `HELLO` 不是 harness（`harness!=1`）；
- 30 s 内未进入 gameplay（`enter_gameplay_seconds`）——本次 12/12 均在 0.2 s 内进入；
- 进入 gameplay 但 `DEBUG isReplay != 1`（拒绝把非回放局当人类参考）；
- 无 `f3==0` 基线采样、`f3` 回退、`f3` 与 `f2` 差值越界（>600）；
- bombs/deaths 记账恒等式不成立（采样漏掉一次下降）。

一次真实失败关闭已被观测并留证（开发早期版本，未产出参考）：`bomb accounting disagrees: sum of decreases=1.0 vs start-end+gained=4.0`，即把“开局 globals 未初始化（bombs=0）”当基线导致的账不平；修复方式是取最后一个 `f3==0` 采样作基线，而不是放宽校验。

## 8. 结论与剩余缺口

- **已完成**：WP6/WP7 所需的“逐面长度（帧）+ 逐面 deaths + 逐面 bombs”在 Normal 与 Lunatic 各一套真实零售回放上取得，来源为真实引擎计数器，12 次运行全部 exit 0、`complete=true`。
- **口径限制（必须按此引用）**：
  - `gameplay_frames` 是**逻辑帧**（含面 clear 序列），不是渲染帧；回放对白快进会让逻辑帧 > 渲染帧（差值见表 `gameplay_frames - engine_frames`，公开在 JSON 的 `dialogue_fast_forward_frames`）。
  - `bombs_used` 是**bomb 发动次数**（与引擎 `bombsUsed` 一致）；deathbomb 消耗 2 储备仍记 1 次，`bombs_stock_spent` 单独给出储备消耗。
  - 掉命那帧的 `SetBombCount(initialBombCount)` 不计入 `bombs_used`，单独记为 `bombs_reset_on_death`。
  - 每份回放只覆盖它实际录到的 6 面：Lunatic 为 1/2/3/4B/5/6B，Normal 为 1/2/3/4A/5/6B。没有 6A、4A(Lunatic)、4B(Normal)、Extra。
- **未做（不属本次授权范围）**：没有生成 `bench/human_reference.json`（该文件不在允许修改清单内）；WP7 scoreboard 仍需要合法 reference 文件与 runner 侧接入。
- **未改**：WSL C++/协议/第三方/依赖/零售 EXE 全部未改动；本实验只新增 Python 工具、单测、本文档与 `.tmp` 证据。

## 9. 证据文件（可复核）

- 逐面原始报告：`.tmp_v400_human_autoplay/final_lunatic_s{0,1,2,4,5,7}.json`、`.tmp_v400_human_autoplay/final_normal_s{0,1,2,3,5,7}.json`（含 `hello`、`stage` 聚合、`events`、`trace`、`engine_stderr`、`wall_seconds`、`frames_per_second`）
- 逐帧采样（掉命面）：`.tmp_v400_human_autoplay/final_lunatic_s7.samples.jsonl`、`.tmp_v400_human_autoplay/final_normal_s5.samples.jsonl`
- 合并后的参考数据：`.tmp_v400_human_autoplay/EXP-P0-046_measured_stages.json`
- 引擎 stderr：`.tmp_v400_human_autoplay/final_*.stderr.txt`（shadow preflight ok / startup setup_frames=8 / subthread=0）
