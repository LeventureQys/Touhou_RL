# EXP-P0-057 — WP7 正式开发集：人类参考帧上限 × 20 dev seeds × random/still

- 日期：2026-11-09（Asia/Shanghai；实测 episode 时间戳 `2026-11-08T15:53:06Z`–`2026-11-08T15:56:06Z`，即本地 23:53–23:56）
- Provider / model：DeepSeek / DeepSeek v4.1 flash
- 结论：**MEASURED — 40/40 真实 episode 全部 `status=ok`、进程 `exit 0`、全部 `game_over`；帧上限不再 provisional，而是 `ceil(1.5 × 13258) = 19887`，来自真实引擎回放测得的人类 Stage1(normal) 13258 gameplay 帧**
- 范围：仅修改 `touhou_rl/v400/bench/human_reference.py`、`touhou_rl/v400/bench/results.py`、`tools/rl/v400_human_reference.py`、`tools/rl/v400_bench_run.py`、`tests/test_v400_human_reference.py`、`tests/test_v400_bench_runner.py`；新增 `Document/Update/Alpha/v4.0.0/bench/human_reference.json`；追加 `bench/results.jsonl` 40 行；重写生成的 `bench/scoreboard.md`；本文件；`.tmp_v400_wp7_dev20*`。未改引擎 C++/协议/依赖/第三方/零售原件，未删除任何旧行，未提交。

## 1. 目标

把 WP7 从“provisional 帧上限切片（3 seeds/策略）”升级为**有人类参考上限的正式开发集**：

1. 用 EXP-P0-046 的**真实引擎实测**数据（`STATE`/`DEBUG` 在真实回放播放中逐帧采集）建立 `bench/human_reference.json`；
2. runner 按 `cap = ceil(1.5 × 该面人类 gameplay_frames)` 取上限，并在结果行写清来源；
3. 用正式 cap 真跑 20 个 dev seeds（1000–1019）× random/still = **40 局真实 episode**；
4. 重新生成 scoreboard，写本实验卡。

本实验**没有**使用回放文件 `inputRecords` 推导任何帧数/deaths/bombs；`not_from_input_records=true` 写进了参考文件本身。

## 2. 人类参考文件（`Document/Update/Alpha/v4.0.0/bench/human_reference.json`）

| 项 | 值 |
|---|---|
| schema | `v400-wp7-human-reference/1` |
| 文件大小 / sha256 | 8,691 bytes / `fb4ba91fe05af73de31ccdec79b09fb1800a814b671254e6b5aaf5e58298efb3` |
| `collection_method` | `engine STATE during real replay playback` |
| `not_from_input_records` | `true` |
| `measured_at` / 来源 | `2026-11-08T23:43:36+08:00`；来源为实测证据文件 mtime（EXP-P0-046 只在 §4 记时长、未在 JSON 内记时钟；证据文件时间戳显示测量发生在 2026-11-08 23:40–23:43） |
| 引擎 | `/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern`，**binary sha256 `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb`**（本次用 WSL `sha256sum` 复核，与 EXP-P0-046 一致） |
| 工作树提交 | `third_party/th08-rl` = `f563df58c44037b44208c393a86540b434b0c83b`；WSL kernel `6.18.54.1-microsoft-standard-WSL2` |
| 语料 | Lunatic `Th8_ud0059.rpy` sha256 `62b02a172780d846f4109fd0dcc762692c5205692670c0f7400c2026c409a5e9`（Youmu, shot 10）；Normal `Th8_ud2970.rpy` sha256 `e621086d814fba4838a2c84e4081162524c3c7e84d1bc4ca3f6621f9ca52d96f`（Marisa, shot 6） |

每套数据含 `source_replay_sha256`、`difficulty`、`shot`/`shot_name`、`stages[{stage_index, stage_name, gameplay_frames, deaths, bombs_used, replay_input_frames_cross_check, stage_cleared, end_reason, score_end, lives_start/end}]`；顶层含 `cap_multiplier=1.5`、`cap_rule`、`coverage`（含 `missing`）、`caveats[]`（6 条）。

实测值（与 EXP-P0-046 表一致，本次仅做机器可读化）：

| difficulty | Stage1 | Stage2 | Stage3 | Stage4A | Stage4B | Stage5 | Stage6A | Stage6B | Extra |
|---|---|---|---|---|---|---|---|---|---|
| normal | 13258/0/0 | 18525/0/0 | 20009/0/0 | 27722/0/3 | — | 25675/1/0 | — | 36403/0/0 | — |
| lunatic | 16732/0/1 | 19009/0/0 | 22394/0/0 | — | 26909/0/2 | 30943/0/1 | — | 35959/1/2 | — |

（格式：gameplay_frames / deaths / bombs_used；“—”= 该回放没有录到该面。）

构建命令与退出码：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python tools/rl/v400_human_reference.py `
  --measured .tmp_v400_human_autoplay/EXP-P0-046_measured_stages.json `
  --output Document/Update/Alpha/v4.0.0/bench/human_reference.json `
  --engine-sha256 591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb `
  --engine-sha256-source "sha256sum of /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern inside WSL Ubuntu, re-verified at build time"
# exit 0；输出 {"status":"ok","sha256":"fb4ba91f…98efb3","measured_at":"2026-11-08T23:43:36+08:00","sets":2,...}
```

fail-closed 行为（都有单测）：实测 JSON 缺 `gameplay_frames`/`deaths`/`bombs_used`、`kind` 不对、`not_from_input_records` 非 true、`cap_multiplier` 非 1.5、`measured_at` 无时区、回放 sha 非 64 hex、`gameplay_frames - replay_input_frames` 超出 `[0,600]`、缺引擎 binary sha256 → 一律拒绝，且**不写文件**（exit 3）。

## 3. cap 规则与被采用的 cap

- 规则：`cap(stage) = ceil(cap_multiplier × 该面实测人类 gameplay_frames)`，`cap_multiplier = 1.5`（**声明式基准策略，不是人类测量值**）。
- S1N（normal, Stage1）：`ceil(1.5 × 13258) = 19887`（整数运算 `-(-3×13258//2) = 19887`，无浮点误差）。
- 参考文件同时给出其他已测面的 cap（如 lunatic Stage1 16732 → 25098），供后续套件使用；**没有测到的面不给 cap**（`cap_for` 抛 `HumanReferenceError`，runner 退回 provisional 并在行内标注）。
- 结果行写：`cap_source="human_reference"`、`max_frames=19887`、`human_frames=13258`、`cap_multiplier=1.5`、`human_reference_stage="Stage1"`、`human_reference_difficulty="normal"`、`human_reference_shot=6`、`human_reference_source_sha256=<回放 sha>`、`human_reference_sha256=<参考文件 sha>`。

## 4. 代码变更与验收

| 文件 | 变更 |
|---|---|
| `touhou_rl/v400/bench/human_reference.py` | 新增 `build_reference_from_measurements` / `validate_reference` / `load_reference` / `write_reference` / `reference_sha256` / `cap_from_human_frames` / `cap_for` / `cap_for_suite` / `caps_for`；保留原 `inspect_human_replay` + `build_human_reference` 的 replay-证据 fail-closed 路径（仍 exit 3，不写参考文件） |
| `touhou_rl/v400/bench/results.py` | `summarize()` 分组键加入 `cap_source`，不同 cap 的 episode 不再混算到同一格；表多一列 `Cap source` |
| `tools/rl/v400_human_reference.py` | 新增 `--measured`（建参考）/`--dry-run`/`--measured-at`/`--engine-sha256`；位置参数 REPLAY 旧行为逐字保留（inspect + exit 3） |
| `tools/rl/v400_bench_run.py` | 新增 `resolve_cap()`（`human_reference` / `operator_override` / `provisional` 三级、来源写进行内）；`run_episode(..., cap=, collection=)`、`--dev20`；参考文件存在但不可信时 **exit 1 且不写行**（不静默降级）；scoreboard 按 cap 来源分表重写 |
| `tests/test_v400_human_reference.py` | 新增 24 个用例（真实实测构建、cap 算术、coverage/caveats、round-trip、15 种篡改拒绝、CLI 建/失败关闭）；文件共 29 |
| `tests/test_v400_bench_runner.py` | 新增 5 个用例（cap 解析、行内 human cap 字段、CLI 端到端、不可信参考 exit 1、scoreboard dev20 标注 + 旧行保留）；文件共 32 |

验收命令与退出码：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m compileall -q touhou_rl/v400/bench tools/rl                                            # exit 0
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_wp7_dev20_t -q `
  tests/test_v400_human_reference.py tests/test_v400_bench_runner.py `
  tests/test_v400_bench.py tests/test_v400_replay_starts.py                                       # 67 passed, exit 0
```

## 5. 40 局真实 episode

每局一条独立命令、独立进程（WSL 真实 ELF 引擎，`--rl-role shadow`，null device），**逐局进程退出码 0**：

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python tools/rl/v400_bench_run.py --agent <random|still> --suite S1N --seed <1000..1019> --execute --dev20
# 默认 --human-reference/--results/--scoreboard 指向 Document/Update/Alpha/v4.0.0/bench/
```

WSL 侧实际子进程（runner 内部构造，未手工改参数）：

```text
wsl.exe -d Ubuntu -- env -u DISPLAY SDL_AUDIODRIVER=dummy \
  /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern \
  --data-dir /root/v400-data/game --rl-control 0,1 --rl-role shadow
```

逐局结果（`frames` = STEP 实际推进帧；`gameplay_frame` = `STATE f3`；全部 `end=game_over`，即“掉命且 `livesRemaining<=0` 触发引擎 retry 分支”；`exit` 为进程退出码）：

| # | Agent | Seed | Frames | Gameplay frame | Deaths | Bombs used | Lives end | Score | End | exit | Wall s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | random | 1000 | 2586 | 2588 | 3 | 0 | 0 | 41193 | game_over | 0 | 3.578 |
| 2 | random | 1001 | 1790 | 1792 | 3 | 0 | 0 | 26031 | game_over | 0 | 3.078 |
| 3 | random | 1002 | 2004 | 2006 | 3 | 0 | 0 | 34615 | game_over | 0 | 3.375 |
| 4 | random | 1003 | 2490 | 2492 | 3 | 0 | 0 | 52750 | game_over | 0 | 3.640 |
| 5 | random | 1004 | 2026 | 2028 | 3 | 0 | 0 | 31261 | game_over | 0 | 3.297 |
| 6 | random | 1005 | 1968 | 1970 | 3 | -1 | 0 | 53654 | game_over | 0 | 3.281 |
| 7 | random | 1006 | 1640 | 1642 | 3 | 0 | 0 | 41229 | game_over | 0 | 3.140 |
| 8 | random | 1007 | 1658 | 1660 | 3 | 0 | 0 | 19369 | game_over | 0 | 2.985 |
| 9 | random | 1008 | 2170 | 2172 | 3 | -1 | 0 | 33319 | game_over | 0 | 3.469 |
| 10 | random | 1009 | 1834 | 1836 | 3 | 0 | 0 | 14577 | game_over | 0 | 3.219 |
| 11 | random | 1010 | 2008 | 2010 | 3 | 0 | 0 | 53184 | game_over | 0 | 3.250 |
| 12 | random | 1011 | 1998 | 2000 | 3 | -1 | 0 | 24837 | game_over | 0 | 4.312 |
| 13 | random | 1012 | 1918 | 1920 | 3 | 0 | 0 | 40194 | game_over | 0 | 3.297 |
| 14 | random | 1013 | 2092 | 2094 | 3 | 0 | 0 | 39980 | game_over | 0 | 3.438 |
| 15 | random | 1014 | 1750 | 1752 | 3 | 0 | 0 | 51788 | game_over | 0 | 4.031 |
| 16 | random | 1015 | 2074 | 2076 | 3 | 0 | 0 | 48211 | game_over | 0 | 4.734 |
| 17 | random | 1016 | 2280 | 2282 | 3 | 0 | 0 | 61781 | game_over | 0 | 4.063 |
| 18 | random | 1017 | 2710 | 2712 | 3 | 0 | 0 | 36028 | game_over | 0 | 3.672 |
| 19 | random | 1018 | 2100 | 2102 | 3 | -1 | 0 | 40871 | game_over | 0 | 4.594 |
| 20 | random | 1019 | 1832 | 1834 | 3 | 0 | 0 | 41752 | game_over | 0 | 3.485 |
| 21 | still | 1000 | 1748 | 1750 | 3 | 0 | 0 | 59953 | game_over | 0 | 2.921 |
| 22 | still | 1001 | 1692 | 1694 | 3 | 0 | 0 | 63693 | game_over | 0 | 3.610 |
| 23 | still | 1002 | 1748 | 1750 | 3 | 0 | 0 | 63762 | game_over | 0 | 4.156 |
| 24 | still | 1003 | 1692 | 1694 | 3 | 0 | 0 | 60704 | game_over | 0 | 4.250 |
| 25 | still | 1004 | 1662 | 1664 | 3 | 0 | 0 | 67926 | game_over | 0 | 3.078 |
| 26 | still | 1005 | 1724 | 1726 | 3 | 0 | 0 | 67787 | game_over | 0 | 3.110 |
| 27 | still | 1006 | 1720 | 1722 | 3 | 0 | 0 | 67345 | game_over | 0 | 3.156 |
| 28 | still | 1007 | 1724 | 1726 | 3 | 0 | 0 | 68189 | game_over | 0 | 3.063 |
| 29 | still | 1008 | 1724 | 1726 | 3 | 0 | 0 | 61363 | game_over | 0 | 3.094 |
| 30 | still | 1009 | 1692 | 1694 | 3 | 0 | 0 | 65967 | game_over | 0 | 2.938 |
| 31 | still | 1010 | 1692 | 1694 | 3 | 0 | 0 | 63559 | game_over | 0 | 2.953 |
| 32 | still | 1011 | 1720 | 1722 | 3 | 0 | 0 | 66965 | game_over | 0 | 3.031 |
| 33 | still | 1012 | 1662 | 1664 | 3 | 0 | 0 | 65190 | game_over | 0 | 2.938 |
| 34 | still | 1013 | 1748 | 1750 | 3 | 0 | 0 | 62602 | game_over | 0 | 3.016 |
| 35 | still | 1014 | 1646 | 1648 | 3 | 0 | 0 | 67989 | game_over | 0 | 2.828 |
| 36 | still | 1015 | 1724 | 1726 | 3 | 0 | 0 | 64776 | game_over | 0 | 3.047 |
| 37 | still | 1016 | 1692 | 1694 | 3 | 0 | 0 | 55551 | game_over | 0 | 2.890 |
| 38 | still | 1017 | 1724 | 1726 | 3 | 0 | 0 | 59148 | game_over | 0 | 3.015 |
| 39 | still | 1018 | 1692 | 1694 | 3 | 0 | 0 | 63122 | game_over | 0 | 2.625 |
| 40 | still | 1019 | 1692 | 1694 | 3 | 0 | 0 | 66193 | game_over | 0 | 2.765 |

## 6. 汇总统计

| 策略 | n | Frames mean [95% CI] | Frames median / min–max | Deaths | Score mean [95% CI] | Score median / min–max | Wall s mean (min–max) | End |
|---|---:|---|---|---|---|---|---|---|
| random | 20 | 2046.4 [1928.4, 2176.1] | 2006 / 1640–2710（sd 282.2） | 3.000 [3.000, 3.000] | 39331.2 [34053.2, 44562.4] | 40532 / 14577–61781（sd 11967.1） | 3.6 (2.985–4.734) | game_over 20/20 |
| still | 20 | 1705.9 [1693.5, 1718.1] | 1706 / 1646–1748（sd 28.4） | 3.000 [3.000, 3.000] | 64089.2 [62539.1, 65481.3] | 64269 / 55551–68189（sd 3361.0） | 3.1 (2.625–4.250) | game_over 20/20 |

同 seed 配对（`touhou_rl.v400.bench.stats.paired`，10,000 次重采样 + 符号置换）：

- frames（still − random）：均值 **−340.5** 帧，95% CI [−466.8, −224.1]，置换 p ≈ 2.0e-4 → 静止策略在 Stage1 存活时间更短且分离明显。
- score（still − random）：均值 **+24,758.0**，95% CI [18,785.2, 30,649.0]，置换 p ≈ 1.0e-4 → 静止策略得分更高（随机乱按导致移动/攻击效率下降）。

解读边界：这些差值只说明两个**无观测控制策略**在 S1N 上的差异，不构成对任何学习策略或人类的评估。

## 7. 真实性核对

- **40/40 真跑**：每局独立 `python ... --execute` 进程 + 独立 WSL 引擎进程；日志 `.tmp_v400_wp7_dev20_runlog.txt` 39 行 + 冒烟 1 局，全部 `exit=0 status=ok`；`bench/results.jsonl` 中 `collection="dev20"` 且 `status="ok"` 的行恰好 40 行（random 20 / still 20，20 个不同 seed），**0 条 error / invalid**。
- **每行 provenance**：`harness_binary_sha256=591c521d…25d3eb`（40/40 一致）、`harness_commit=f563df58…b0c83b`、`harness_wsl_kernel=6.18.54.1-microsoft-standard-WSL2`、`human_reference_sha256=fb4ba91f…98efb3`、`privilege=deploy_pixel`、`referee=touhou_rl.v400.bench.judge.judge`。
- **append-only**：旧 13 行（6 条 provisional 实测 + 1 条作废记录 + 6 条 EXP-P0-048 实测）逐行字节前缀未变（`append_row` 每次追加前后校验，且本次复核 1–13 行的 `schema/status/agent/seed/started_at` 与追加前完全一致）；本次只追加 40 行，未删除、未覆盖。
- **没有伪造 episode**：40 行的 `frames/deaths/bombs_used/lives_end/score/end` 全部来自真实 `STATE`；`cleared=false` 是引擎语义（未观测到 stage clear），不是猜测。
- **没有用 inputRecords**：`human_reference.json` 的 `not_from_input_records=true`；`replay_input_frames_cross_check` 字段只作一致性交叉校验，任何 cap 计算都不读它。
- **一致性旁证**：seed 1000 random 的 frames/score 在 provisional (3000 cap) 与 human_reference (19887 cap) 两次真实运行中完全相同（2586 / 41193），说明 episode 由引擎+seed 决定、cap 只是上界。

## 8. 限制（必须随数据一起引用）

1. **人类参考覆盖不全**：缺 Stage6A、Lunatic 的 Stage4A、Normal 的 Stage4B、Extra；每份回放只覆盖它实际录到的 6 面。因此只有 `S1N`/`S1L`（Stage1）有合法人类 cap；任何需要未测面的套件**必须**没有人类 cap（runner 会退回 provisional 并标注）。
2. **单份 Normal 语料**：全部 normal 数字来自 `Th8_ud2970.rpy` 一位玩家的一次录制，是参考点不是人群估计；`cap_multiplier=1.5` 也是策略选择，不是人类测量。
3. **cap 永远没有被触达**：40/40 都在 ~1.6k–2.7k 帧时 `game_over`，`cap_hit=false`，远未到 19887。所以本次 cap 的作用是“合法上界 + 可比性标签”，不是使能更长 episode。
4. **deaths 饱和、cleared=false**：episode 结束于“掉命且无剩余生命”，故 deaths 恒为 3、lives_end 恒为 0，无法区分策略；Clear rate 列不可解释为通过率；score 与 frames 才是区分量。
5. **bombs_used 可负**：`bombs_used = bombs_start − bombs_end`，random 有 4 局为 −1（道具/奖命补弹使储备上升）。两个控制策略从不按 bomb，因此该列只能度量计数器移动。
6. **无观测策略**：random/still 都不读 observation/privileged state，只是按钮掩码；本表**不是**学习策略 baseline 结论，也**不是** gate。
7. **测量口径**：`gameplay_frames` 是逻辑帧并含面 clear 序列（比回放输入帧多 69–500），不是渲染帧；回放对白 3 倍快进会让逻辑帧多于渲染帧。
8. **测量是回放播放不是真人实时操作**：无人类输入延迟、无菜单时间，每面由独立进程从该面起点开始，shadow/null-device 无画面音频。
9. **只跑 dev split**：seeds 1000–1019（`DEV_SEEDS` 前 20 个），**未**使用 confirm split（60000+），本表不是 confirm 评估，也不是门评估。
10. **wall_s 不是吞吐量指标**：批次运行期间同一 WSL 发行版上另有并发重放任务（VM load ≈ 12–15），wall 3.1–3.6 s/episode 只作运行代价参考，帧数/得分不受影响（引擎确定性 + 固定 seed）。

## 9. 证据文件（可复核）

- `Document/Update/Alpha/v4.0.0/bench/human_reference.json`（本次生成）、`bench/results.jsonl`（+40 行）、`bench/scoreboard.md`（重写）
- `.tmp_v400_wp7_dev20_runlog.txt`：39 条批处理日志（agent/seed/exit/status/frames/deaths/score/end/cap/src/wall）
- `.tmp_v400_wp7_dev20_runs/<agent>_<seed>.json`：39 份单局完整 runner 输出（含 cap 证明、harness sha、terminal_rule）
- `.tmp_v400_wp7_dev20_run_smoke.json`：seed 1000 random 的首次单局输出
- `.tmp_v400_wp7_dev20_run_batch.ps1`、`.tmp_v400_wp7_dev20_summary.py`/`.txt`、`.tmp_v400_wp7_dev20_reference_build.json`、`.tmp_v400_wp7_dev20_scoreboard_report.json`
- 人类参考的原始证据（上游，未改动）：`.tmp_v400_human_autoplay/EXP-P0-046_measured_stages.json` 与 `.tmp_v400_human_autoplay/final_*.json`

## 10. 结论与剩余工作

- **已完成**：`bench/human_reference.json` 从真实引擎实测数据建立并通过严格校验；runner 用 `ceil(1.5 × 人类 gameplay_frames)` 取 cap 并在行内留证；40 局 dev20 真实 episode 全部 exit 0；scoreboard 已升级为“human-referenced dev set (20 seeds)”并保留旧 provisional 行为独立表。
- **未完成/不在本次范围**：confirm split 评估、门评估（gate）、PSN/多面套件（runner 仍拒绝多面套件：需要 stage-clear 检测器）、6A/4A-L/Normal 4B/Extra 的人类参考（需要新的真实回放测量）、学习策略接入。
