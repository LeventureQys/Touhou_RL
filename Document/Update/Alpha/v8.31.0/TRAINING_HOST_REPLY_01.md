# V831-SLICE-01 回执：短任务提名与真机接线校准

**主机代号**：轩神开播了（Windows 11 家庭版 / i5-14600KF；WSL2 hostname `ubuntu`，kernel `6.18.58.1-microsoft-standard-WSL2`，20 vCPU，23 GiB，RTX 4070 SUPER 12,282 MiB，driver 617.42）
**时间及时区**：2026-11-26 08:46:37 – 08:56:47 +08:00（最终一轮）；本单全部执行窗口 2026-11-25 23:04 – 2026-11-26 08:57 +08:00
**执行代理及模型**：DSH 主代理（DeepSeek Flash）派发，Provider=DeepSeek、模型 DeepSeek v4.1 flash 执行代理完成编码；主代理独立复核范围、命令与统计
**对应工单**：[`TRAINING_HOST_REQUEST_01.md`](TRAINING_HOST_REQUEST_01.md)（v8.31.0 开发计划 · [README](README.md)）

---

## 0 一句话结论

B 节**两条真实轨迹都通过**：像素接线 = 600 受控帧 / exit 0；教师校准 = **真实 `stage_clear` @ 13,195 帧**，且三个动作摘要与冻结实现的独立正对照**逐位相同**（`mask_trace_sha256 = 66c81344…f179924`），双过关信号首次在**同一条记录**里并测：`EVENTS` 的 `0x0080` 电平/边沿在 **13,118** 帧出现，`STATE.stage` 前进在 **13,195** 帧，**顺序 events→state、间隔 77 帧**。
A 节**部分可冻结（缺项 8 条）**：现有登记材料里唯一的完成事件是**整段 stage 1**；不存在"短子阶段"的事件定义，也不存在可用的中局合法前缀（[任务卡](TASK_CARD_SLICE_A.md)）。
执行过程中发现并修复了**两处工具缺陷**与**两处工单口径问题**（见 §7），其中一处使前两次引擎启动作废——全部按"已启动尝试"如实计入（§5.0）。

---

## 1 实际检出与身份

| 项 | 值 | 来源 |
| --- | --- | --- |
| 实际检出 | `/mnt/d/WorkShop/Touhou_RL`（= Windows `D:\WorkShop\Touhou_RL`） | 训练端本次实测 |
| HEAD | `11807614ca8453d62e48eff6ca4c4df804cd1bae`（v8.31.0 文档提交，父 `3af5c40`＝v8.31 计划所记设计基线） | 训练端本次实测 |
| `git status --short --untracked-files=no` | **3,893 行**（`D` 3,493 / `M` 399 / `m` 1；`Document/Update` 3,698、`tools/rl` 82、`Document/Dialogs` 33…）。**本单开始前即存在**：v8.30 回执记 Windows 侧 3,516 行（同一批既有未提交改动/删除），本单未产生任何 `Document/` 或 `touhou_rl/` 改动 | 训练端本次实测 |
| Python（执行侧） | `/root/v400-venv/bin/python` → `/usr/bin/python3` 3.10.12；numpy 2.2.6；pytest 9.2.0 | 训练端本次实测 |
| Python（Windows 侧离线门） | `python` 3.12.10；numpy 2.5.4 | 训练端本次实测 |
| 引擎（登记的 rl_v3） | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，sha256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` —— 与工单第 67 行的期望值 `sha256sum -c` **OK** | 训练端本次实测 |
| 数据源 | `/root/v400-data/game`，`du -sb` = **496,968,814 B**，17 文件 / 2 目录 / 0 符号链接 | 训练端本次实测 |
| 源 `th08.cfg` | 60 B，sha256 `48bf7621da5879ef5ec1392fcac2cc0a85c313912a24fb33b4cddccf0d28f07a`，四项 `musicMode=1 playSounds=1 musicVolume=39 sfxVolume=36`（**可出声**，与 v8.30 回执第 255 行登记一致） | 训练端本次实测 |
| 静音副本 `th08.cfg` | 60 B，sha256 **`cfa292ae10ee1907df7c637d013386851486dadf7a97bb5adf12bcf21dd14638`**（与工单第 47 行期望一致）；`diff_byte_offsets = [31, 32, 39, 40]` = `0x1F/0x20/0x27/0x28`；其余 56 字节逐位相同 | 训练端本次实测 |
| obs-v3 digest | `ef2fbd15385f26f425db2bfe855d105ee009023acb08c26916e811963553a105` | 训练端本次实测 |
| 教师磁带 | `.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed12-shard/slots.jsonl`，sha256 `f759de7dacc78343dec8825da5de14942e749b8a5a2c5319304e513d1618b7b9` —— 工单第 82 行 `sha256sum -c` **OK** | 训练端本次实测 |
| 新工具 | `tools/rl/v8_31_slice.py`，sha256 `19888a274c3508c8fabc21452aa5959b9916b84b75ba39ce287d0b48026430d0`，3,624 行 | 训练端本次实测 |
| 新测试 | `tests/test_v8_31_slice.py`，sha256 `2c51143f8a22781c9af2cf8cb9ac1a0cb9a5e55066c9b5a7d08cbb43b3a630e3`，1,899 行 | 训练端本次实测 |

**代码写入范围**：本单只新增上述两个文件（工单第 10 行白名单），另只新增两份文档（本回执与 [A 节任务卡](TASK_CARD_SLICE_A.md)）。未改 `touhou_rl/**`、`conftest.py`、`configs/**`、`third_party/**`、依赖、协议/ABI、任何既有证据或资产。

---

## 2 A 节交付：短阶段任务卡

**交付物**：[`TASK_CARD_SLICE_A.md`](TASK_CARD_SLICE_A.md)（181 行，`Get-FileHash`/`sha256` 见 §9）。
**状态**：**部分可冻结**（缺项 8 条，逐条给"缺什么 / 为什么缺 / 最小补证动作 / 谁授权"）。

| 工单字段 | 结论 | 证据 |
| --- | --- | --- |
| 起止和分母 | 可冻结：阶段身份 `th08-stage1-normal-team0`（stage 0 / normal / shotType 0 / 非 practice）；起点**只能自然开局**（引擎无"从第 N 帧开始"入口）；完成点 13,195 帧；失败定义（首死 / 命尽）与截断定义（`max_frames`）可冻；每局上限登记过 19,887 / 600 / 13,195 三个值 | 任务卡 §1、§6 |
| 任务信号 | 可冻结：同约束成功 1 例（教师磁带重放，`stage_clear=true`、13,195 帧、2,631,430 分、`deaths=0`）；不同后果 2 例（no-pulse 同磁带 15,564 帧 `game_over`、13,195 帧处仅 141,379 分；策略分母 160 局 `stage_clear=0`、最深 3,914 帧）。"可观测完成"与"自动推进/超时"有硬读数区分（分数在 141,379 冻结 8,187 帧） | 任务卡 §2 |
| 入口和资源 | **不可冻结中局**：无起始帧入口；中局合法前缀与接管哈希**不存在**（R74 已排除四条解阻路径中的三条）。自然开局的**帧 0 实测资源由本单 B 节闭合**（见 §6.4） | 任务卡 §3；本回执 §6.4 |
| 祖先用途 | 可划出训练根 328 / dev 20 / 隔离 12 / confirm 0 / adapt 0；**但不存在"未参与拟合或选参"的开发根**：dev 1000–1019 已用于选参，1020–1039 只在 `reserved_seeds` 里 | 任务卡 §4 |
| 缺项 | **A1** 首受控帧编号基未钉死 → **本单已闭**（§6.3 并列四个数）；**A2** 双完成口径从未并测 → **本单已闭**（§6.1）；**A6** 帧 0 资源无登记读数 → **本单已闭**（§6.4）；**A3** 截断无真实见证；**A4** "短"与"完成可观测"不可同时成立（600 / 13,195 = 4.55%）；**A5** 中局合法前缀不存在；**A7** R74 首个冲突的具体掩码未登记；**A8** 无干净开发根 | 任务卡 §5 |

**研究端须裁定的分水岭（A4×A5 互咬）**：若要求"短 = 600 帧 + 阶段完成"，构造上不可观测（600 帧只覆盖完成点的 4.55%）；若要求观测完成，上限必须 ≥13,195（登记的 19,887 就够），但那就**不是短阶段**而是整段 stage 1。走"中局前缀 + PARK"则需要**资产重录授权**（一决策一 mask 的干净磁带），本单未做。

---

## 3 B 节：两条真机轨迹（最终一轮，训练端本次实测）

**运行根**：`/mnt/d/WorkShop/Touhou_RL/.tmp_v831_slice_01_YmVD4nBX`（`mktemp -d` 生成，未经清理）
**验收脚本**：`.tmp_v831_slice_01_probe/v831_acceptance_wsl_v2.sh`，日志 `.tmp_v831_slice_01_probe/acceptance_wsl_run3.log`（3,907 行，sha256 `c196f970bddfbe69a4535577351cbf15ad6a6a905076585aed5d5b1b73643afb`）
**脚本墙钟**：pixel **35.229 s**、teacher **504.745 s**（均由 `date +%s.%N` 夹在子进程外侧测得）

### 3.1 实际命令与退出码

| # | 命令 | 退出码 |
| --- | --- | --- |
| 1 | `git rev-parse --show-toplevel` / `--rev-parse HEAD` | 0 → `/mnt/d/WorkShop/Touhou_RL` / `11807614…` |
| 2 | `git status --short --untracked-files=no` | 0（3,887–3,893 行，见 §1） |
| 3 | `printf '691bfd77…' "$ENGINE" \| sha256sum -c` | 0 → `OK` |
| 4 | `python -B -c '... obs_spec_v3_digest()'` | 0 → `ef2fbd15…` |
| 5 | `python -B -m pytest -q tests/test_v8_31_slice.py tests/test_v400_obs_v3.py` | **0 → 79 passed in 19.39 s** |
| 6 | `python -B tools/rl/v8_31_slice.py pixel-smoke …`（工单原样 14 参数） | **0** |
| 7 | `printf 'f759de7d…' "$TAPE" \| sha256sum -c` | 0 → `OK` |
| 8 | `python -B tools/rl/v8_31_slice.py teacher-calibration …` | **0** |
| 9 | 终局断言 `[ pixel_rc -eq 0 ] && [ teacher_rc -eq 0 ]` | 0 |

**离线门（两侧合计 106 passed）**：

| 侧 | 命令 | 结果 |
| --- | --- | --- |
| WSL | `pytest -q tests/test_v8_31_slice.py tests/test_v400_obs_v3.py` | **79 passed / 19.39 s / rc 0** |
| Windows | `python -m pytest -q touhou_rl/tests/test_framefence.py touhou_rl/tests/test_frame_proof.py touhou_rl/tests/test_phase2_history_proof.py` | **27 passed / 0.36 s / rc 0** |
| WSL | `python -m compileall -q tools/rl/v8_31_slice.py tests/test_v8_31_slice.py` | rc 0 |
| WSL | `--help` / `pixel-smoke --help` / `teacher-calibration --help` | rc 0 / 0 / 0 |

### 3.2 计数

| 轨迹 | run ID | 计划 | 启动 | 有效 | 合法失败 | 技术无效 | 未启动 | exit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 像素接线 | `pixel-smoke-20261126T084700-528` | 1 | 1 | **1** | 0 | **0** | 0 | **0** |
| 教师校准 | `teacher-calibration-20261126T084735-633` | 1 | 1 | **1** | 0 | **0** | 0 | **0** |

`auto_resume` / `resume` 两条轨迹均为 0（本单不恢复）。

### 3.3 帧序与实际 STEP 序列

| 量 | 像素接线 | 教师校准 |
| --- | ---: | ---: |
| 受控帧 `controlled_frames` | **600** | **13,195** |
| 决策数 | 300 | 6,598 |
| `step_units`（= 实发 STEP 条数） | 600 | 13,195 |
| `renders`（= 每帧一次 buffer.push，含预热首次） | 601 | 13,195 |
| `pixels_warmup_frames` | 0 | 0 |
| 每决策 STEP 条数 | 2 条各 1 帧（`obs_window="consecutive"`） | 同 |
| 帧上限是否被突破 | 否（`unspent_frames=0`） | 否 |
| `pixel_frame == EVENTS.frames − 1` 校验 | 600/600 通过 | 13,195/13,195 通过 |
| 中继恢复 / 读回次数 | resume 0（未发生） | resume 0（未发生） |

逐决策的 `logical_frame` / `pixel_frame` / `pixel_frame_blake2b128` / `actor_input_digest` / `requested_mask` / `segments`（每段 `[mask, requested_frames, advanced]`）落在 `<out>/decisions.jsonl`（像素 300 行、教师 6,598 行）。

### 3.4 动作生效与射击脉冲

| 量 | 像素接线 | 教师校准 |
| --- | ---: | ---: |
| 候选序列 | 恒 `cand=12`（mask 133） | 冻结 `load_clear_schedule(tail=64)` 的 6,660 条 → 实际消费 6,598 |
| 释放决策（`release_due(i,48)`） | 6（47, 95, 143, 191, 239, 287） | 137（47, 95, …, 6575） |
| `releases_applied` | **6** | **137** |
| SHOOT 帧 / 松键帧 | 594 / 6 | **13,058 / 137** |
| SHOOT 上升沿 | 7 | **138** |
| 互异 mask 数 | 2（133、132） | **30** |
| `mask_trace_sha256` | `d438122dd8b7e8461ef12c2ad50a5fccabba5acbf82d2be9a2a1c35de48b127f` | **`66c81344b00f8766f7d9e860d1874b1141c699122f491ed1a8ed0aea9f179924`** |
| `action_sequence_sha256` | `689f40ffaa80db112aa61355eb5285138d9ce7f2889bc069e6ce9d90754f2af2` | **`5e9d66c4f5685da9f0ea56f3105656b7210721124f8b74a12813b49089995cd3`** |
| `observed_input_sha256` | `482b5221e008e3ee973016527eae4fe37e6a6d8b33b57aad4c3f27bcb0aaaa2c` | **`315a2ab6552edcba23a40f49069150e52eac67f034407e9480f48a66f47c6212`** |
| 动作历史 | 仅在 `advanced_total > 0` 后 push，每决策 1 行，前 8 行零行 | 同（前导 8 决策取磁带 `history`，全 `cand=7`） |

**等价性证明（本单最强的一条）**：教师轨迹的三个摘要与 `.tmp_v420_evalpulse/positive_control.json`（`tools/rl/v420_eval_shoot.py` 的独立实现，sha256 `9cdd9033…`）**逐位相同**，且 `expected/match` 表 **21/21 全部为 true**——`stage_clear`、`frames=13195`、`score=2631430`、`decisions=6598`、`state_frame=13195`、`gameplay_frame=13196`、`lives_end=2.0`、`deaths=0.0`、`renders=13195`、`pixels_warmup_frames=0`、`shoot.shoot_frames=13058`、`shoot.no_shoot_frames=137`、`shoot.releases_planned/applied=137/137`、`shoot.shoot_rising_edges=138`、`shoot.distinct_masks=30`、`shoot.mask_trace_sha256`、`action_sequence_sha256`、`observed_input_sha256`、`harness_role="env"`、`obs_window="consecutive"`。**两个独立实现给出同一动作轨迹与同一像素输入摘要**，这是"帧序/动作生效/像素进入 actor"三条验线的交叉证据。

---

## 4 终态判分：双过关信号首次并测

| 信号 | 帧 | 说明 |
| --- | ---: | --- |
| `EVENTS.stageClear`（电平）首次为 1 | **13,118** | `first_clear_event_source="stageClear"` |
| `EVENTS.stageClearEdgeFrame`（引擎锁存边沿） | **13,118** | 与上者一致 |
| `EVENTS.frameEventFlags & 0x0080`（电平位）首次置位 | **13,118** | 与上者一致（三者同帧） |
| `STATE.stage` 首次前进（0 → 1） | **13,195** | `terminal={"kind":"stage_clear","reason":"state_stage_advanced","frame":13195,"decision_index":6597}` |
| 顺序 | `["events_clear", "state_stage_advance"]` | 过关屏先出现，阶段计数后前进 |
| 间隔 | **77 帧** | |
| 第二信号是否缺失 | `missing_second_signal=false` | 两套信号都取得 |
| 观测窗 | 32 行原始小窗（帧 13,164 – 13,195）落盘 | `frame_event_flags` 在该窗内恒含 `0x0080`（128 或 1152 = 0x480） |
| 命中/死亡 | `player_hit_frames=[]`、`deaths=0.0` | 全程无中弹 |

**这条回答了三件此前无人并测的事**：(a) 两套完成口径在同一记录里**一致**（都判过关）；(b) 它们**不同帧**，相差 **77 帧**——"完成"这个词必须先声明端点是**过关屏出现**还是**阶段计数前进**，否则同一个 `stage_clear` 可以合法地报成 13,118 或 13,195；(c) 首受控帧编号基：`frames=13,195`、`STATE.frame=13,195`、`gameplay_frame=13,196`、磁带末录制决策帧 13,190 并排给出（A1 闭合）。

---

## 5 成本（训练端本次实测）与预算

### 5.0 引擎启动总账（含两轮被作废的尝试）

| # | 尝试 | 运行根 | 结果 | 作废原因 |
| ---: | --- | --- | --- | --- |
| 1 | 像素接线 | `.tmp_v831_slice_01_lV7zfs9S/pixel` | `technical_invalid` @ decision 0 / frame 1 | 工具审计顺序缺陷（§7 D2） |
| 2 | 教师校准 | `.tmp_v831_slice_01_lV7zfs9S/teacher` | `technical_invalid` @ decision 0 / frame 1 | 同上 |
| 3 | 像素接线 | `.tmp_v831_slice_01_eK6pxCuY/pixel` | **通过**（600 帧 / exit 0） | —— |
| 4 | 教师校准 | `.tmp_v831_slice_01_eK6pxCuY/teacher` | 墙钟超时 @ 9,582/13,195 帧 | 每帧 58 ms 纯 Python 像素摘要（§7 D3） |
| 5 | 像素接线 | `.tmp_v831_slice_01_YmVD4nBX/pixel` | **通过**（600 帧 / exit 0） | —— 最终工具 |
| 6 | 教师校准 | `.tmp_v831_slice_01_YmVD4nBX/teacher` | **通过**（13,195 帧 / exit 0） | —— 最终工具 |

工单第 36 行的"最多启动这两个引擎进程，各一条"被**超出**：实际启动 **6 次**（每轮 1 像素 + 1 教师）。原因全部是工具缺陷（D2、D3），不是为挑结果重试；第 5、6 次使用同一份最终工具，前 4 次的记录全部保留、未覆盖、未删除。**这两轮重跑经用户显式授权**（会话内），此处如实登记。

**写入量**：三个运行根合计 **3,006,505,559 B ≈ 2.80 GiB**（`du -cb` 连探针目录 `.tmp_v831_slice_01_probe` 一起算为 **3,008,965,603 B**），超出工单第 36 行的 **2 GiB** 上限（同样是两轮作废运行造成）。就"合计两份私有游戏目录"这一条本身：最终一轮 `pixel/game` 496,968,813 B + `teacher/game` 496,968,815 B = **993,937,628 B ≈ 0.93 GiB**，**未超**该分条。

### 5.1 最终一轮逐段墙钟（`result.json.cost`）

| 桶 | 像素接线 (600 unit) | 教师校准 (13,195 unit) | 桶定义 |
| --- | ---: | ---: | --- |
| `startup` | 6.615 s | 6.495 s | 进程启动 + HELLO + RESET |
| `resume` | **0**（未发生） | **0**（未发生） | 本单不恢复 |
| `readback` | 0.000201 s | 0.000183 s | `READBACK` / `SHM` 挂载 / 映射 |
| `actor` | 0.541 s | 11.282 s | `encode_planes_v3` + `ObservationBufferV3` + 摘要 |
| `step` | 4.101 s | 95.705 s | 每条 `STEP` |
| `audit` | 0.206 s | 3.490 s | `EVENTS` / `STATE`（+ gated 的 `HUD`/`PIXELS`） |
| `capture` | 17.102 s | **365.227 s** | 共享块捕获 + 本地摘要（每帧 blake2b-128，每决策 FNV-64） |
| `io` | 0.061 s | 1.164 s | JSON/JSONL 写入 |
| `session_close` | 0.0076 s | 0.0076 s | `QUIT` + 进程回收 |
| `copytree` | 2.703 s | 2.516 s | 私有游戏目录复制（474 MB） |
| `preflight` | 0.030 s | 0.015 s | 身份/空间/静音前置 |
| **`accounted`** | **31.367 s** | **485.901 s** | 叶子桶之和（互斥，不重复计数） |
| **`total`** | **31.503 s** | **487.657 s** | |
| `unaccounted` | 0.136 s（**0.43 %**） | 1.757 s（**0.36 %**） | |
| `unaccounted_by_unit` | 0.226 ms | 0.133 ms | |

**端到端总耗时**：pixel 35.229 s（脚本外侧计量，含 xvfb-run 重执行与提前退出）／teacher 504.745 s。
**每小时有效轨迹**：只有 1 条/类，**不估 P90/P95**（工单第 43 行的要求）。
**是否满足工单 600 秒上限**：教师 **504.745 s ≤ 600 s**。验收脚本里写的是 `--timeout-s 900`（见 §7 D3 的偏差说明），但**实际墙钟落在 600 s 以内**，故工单的 600 s 口径按实测**成立**；900 s 只作为防再超时的安全网，未被动用。

### 5.2 每条轨迹的 `EVENTS`/`STATE` 观测覆盖

| 量 | 像素接线 | 教师校准 |
| --- | ---: | ---: |
| `step` / `shm` / `events` / `state` 条数 | 600 / 600 / 600 / 600 | 13,195 / 13,195 / 13,195 / 13,195 |
| gated `hud` 条数 | 7 | 138 |
| gated `pixels` 条数 | 1 | 1 |
| `hud_mismatches` | 0 | 0 |
| `checksum_scope` | `per_decision` | `per_decision` |
| `frame_digest_scope` | `per_unit_blake2b128` | `per_unit_blake2b128` |
| `fnv_units` / `fnv_crosscheck_units` | 300 / 1 | 6,597 / 1 |
| `blake2b_units` | 600 | 13,195 |
| `audit.stride` | 1 | 1 |

---

## 6 资源、输入隔离、静音

### 6.1 帧 0 实测资源（闭合 A6）

| 字段 | 像素接线 | 教师校准 |
| --- | ---: | ---: |
| `frame` / `stage` / `gameplay_frame` | 0 / 0 / 2 | 0 / 0 / 2 |
| `lives_remaining` | 2.0 | 2.0 |
| `bombs_remaining` | 3.0 | 3.0 |
| `player_power` | 0.0 | 0.0 |
| `deaths` / `score` / `graze` | 0.0 / 0 / 0 | 0.0 / 0 / 0 |

末态：像素 = 第 600 帧、stage 0、`lives 2.0 / bombs 3.0 / power 0.0 / score 240 / graze 0`；教师 = 第 13,195 帧、**stage 1**、`lives 2.0 / bombs 3.0 / power 22.0 / score 2,631,430 / graze 467`。
**与 v8.30 回执"源码级初值"对照**：`livesRemaining=lifeCount=2`、`playerPower=0` 两条**实测相符**；`bombsRemaining` 实测 **3**（回执已提示它来自 `primaryShtFile->initialBombCount` 而非 `cfg.bombCount`，与 `cfg.bombCount=3` 数值巧合相同，本单无法区分二者，标**待核验**）。

### 6.2 静音证据（AGENTS.md §9.3 三件套 + 进程退出后复检）

| 检查 | 像素接线 | 教师校准 |
| --- | --- | --- |
| 私有目录（整目录 `copytree`，含 `th08.dat`/`thbgm.dat`） | `<out>/game` | `<out>/game` |
| 源 cfg sha / 四项 | `48bf7621…` / 1,1,39,36（audible） | 同 |
| 副本 **首启前** cfg sha / 四项 / 判定 | `cfa292ae…` / 0,0,0,0 / **muted（严格四项）** | 同 |
| `diff_byte_offsets` / 其余字节 | `[31,32,39,40]` / `other_bytes_identical=true` | 同 |
| **进程退出后**副本 cfg sha / 四项 | **`cfa292ae…` / 0,0,0,0（仍静音）** | **同** |
| `SDL_AUDIODRIVER` | `dummy` | `dummy` |
| `DISPLAY` | `:99`（`xvfb-run -a` 私有 Xvfb） | `:99` |
| `LIBGL_ALWAYS_SOFTWARE` / `GALLIUM_DRIVER` / `LP_NUM_THREADS` | `1` / `llvmpipe` / `1` | 同 |
| `problems` | `[]` | `[]` |

**两条新事实（比 AGENTS.md §9.2 的假设更强）**：
1. 引擎退出**没有**把副本 `th08.cfg` 写回可出声——`copy_after_exit` 的 sha 与首启前**逐位相同**。AGENTS.md §9.2 记的"引擎退出时把自己的 cfg 写了回去"在**这份副本 + 这份引擎**上**不成立**（可能因为源已被改成静音、或该写回路径只在窗口正常关闭时才走）。
2. 源目录 `/root/v400-data/game` **一字未改**：运行前后 `du -sb` 均 496,968,814 B、17 文件、`th08.cfg`/`score.dat` 的 mtime 与 sha 全部相同（`48bf7621…`、`147a7ef5…`）。

### 6.3 输入隔离（工单 §帧与判分验收 2 的第二半）

- 静态边界：`build_actor_inputs(pixels_rgba, history)` **恰好两个入参**（`inspect.signature` 断言），模块常量 `ACTOR_FIELDS=("pixels_rgba","history")`、`REFEREE_FIELDS`（lives/bombs/power/stage/deaths/frame_event_flags/stage_clear）只出现在审计侧；`actor_boundary_problems()` 每次运行自检并写进 `result.json.actor_boundary`（两条轨迹均 `[]`）。
- 动态边界（离线测试）：同一像素 + 同一动作历史、把裁判快照换成两组完全不同的值，actor 的三个输入摘要、`ObservationBufferV3` 内部状态（stack/hud 字节 + `frames_pumped`）、以及策略输出**逐位相同**。
- 真实轨迹侧：`observed_input_sha256` 与冻结独立实现 `315a2ab6…` **逐位相同**，说明"进入 actor 的正是那两个来源"，没有任何裁判字段混入。

---

## 7 偏差、缺陷与已修复项（全部训练端本次实测）

### D1 工单"验收命令"在 WSL 下**不可能通过**（工单口径问题，未修代码）

工单第 70–73 行的 pytest 目标里，**5 个文件有 3 个在 WSL 下连收集都失败**，因为它们在 import 链上碰 `ctypes.WinDLL`（`touhou_rl/memory.py:23`，Windows-only）：

| 文件 | WSL（3.10.12 / pytest 9.2.0） | Windows（3.12.10） |
| --- | --- | --- |
| `tests/test_v400_obs_v3.py` | **28 passed**（rc 0） | —— |
| `touhou_rl/tests/test_framefence.py` | collection **ERROR**（rc 2） | 27 passed（三文件合计） |
| `touhou_rl/tests/test_frame_proof.py` | **15 passed / 1 FAILED**（rc 1，`framefence.py:116`） | 同上 |
| `touhou_rl/tests/test_phase2_history_proof.py` | collection **ERROR**（rc 2） | 同上 |

把工单命令块**逐字**落成 `v831_acceptance.sh` 并实际执行：**退出码 2**，停在 pytest 收集错误，**两个引擎进程一个都没启动**（日志 `.tmp_v831_slice_01_probe/acceptance_verbatim.log`，3,925 行，sha256 `70822df013556b8741d181b5a1e9f02df8f079d4e2580f84f85b75e3c948db41`）。
**处理**：不改工单，改为"两侧合跑同一 5 个文件"（WSL 79 passed + Windows 27 passed = **106**，一个都没丢），并把这一行的偏差写在验收脚本头部注释里。建议下一单把该 pytest 行明确拆成 WSL 侧与 Windows 侧两段。

### D2 工单第 40 行的像素帧关系式对 rl_v3 **不成立**（工具缺陷，已修）

工单与 `tools/rl/v400_stage_clear_replay.py:547` 的写法是 `header.frame == EVENTS.frames − 1`。但 rl_v3 的 `HandleStep` **自己在返回前就 `PublishState`**：

- `src/rlharness/rlharness.cpp:899-902`：`// T7.3: STEP publishes too, so "STATE first" is no longer a prerequisite` → `PublishState(fields, hash);`
- `src/rlharness/rlharness.cpp:1362`：`header->frame = FrameCount();`（自增后的值）
- `src/main.cpp:143-146`：`AfterDrawFrame()`（`PublishPixels`，自增**前**）→ `Present()` → `++g_frameCount`
- `touhou_rl/v400/gym/shm.py` 模块 docstring 明写 T7.1/T7.3 起"engine publishes this block itself at the end of every RESET and every STEP… the header's frame number is the one the command's reply reports"；`touhou_rl/v400/gym/env.py::_assert_published` 就断言 `header["frame"] == reply["frame"]`

所以 STEP 一返回：`header.frame == EVENTS.frames`（自增值），而 shm 里的 **RGBA 字节仍是 `EVENTS.frames − 1` 那一帧**。第 1、2 次引擎启动因此都在 **decision 0 / frame 1** 判为 `technical_invalid`，`reason="pixel frame 1 != EVENTS frames 1 - 1"`。
**修复**：判定依据从"被 `STATE`/`PIXELS` 重写过的 header 字段"改为"STEP 刚返回那一刻捕获的 RGBA 字节所属帧 = `EVENTS.frames − 1`"；header 只作 `{F−1, F}` 的二值分类（`RELATION_STATE_PUBLISH` / `RELATION_DRAW_PUBLISH`），任何第三值仍 fail-closed 且只允许一次（在发 `STATE`/`PIXELS` 之前）重读。审计命令顺序固定为 `STEP → shm → EVENTS → STATE`（`AUDIT_COMMAND_SEQUENCE`），并把根因写进注释。
**给研究端的建议**：工单第 40 行与 `PIXEL_SLICE_PLAN` 里"像素帧 = EVENTS 帧 − 1（header）"的表述需要按 rl_v3 语义改写（**字节帧**是 −1，**header 标签**不是）。

### D3 工单的 600 s 教师上限是按**冻结三命令仪器**标定的（成本口径问题，已用算法置换解决）

冻结正对照（`v420_eval_shoot.py`）每帧只发 `STEP` + `STATE`（外加 shm 读与一次 `encode_planes_v3`），13,195 帧 **121.573 s = 9.2 ms/帧**。本单按工单 §帧与判分验收 2 还要求"原始画面校验值"，第 2 次教师启动实测 **63.6 ms/帧**，在 600 s 处停在 9,582 帧。
**成本归因（补测后暴露的真因）**：每帧 **58.2 ms 全部花在纯 Python 的 FNV-1a-64 遍历 1,228,800 B 像素**上（占 96.7 %），协议命令只占 1.0–1.7 ms/帧。**换算法而不是放宽上限**：
- 工单原文要求的是"**每次决策**记录…原始画面校验值"→ 引擎可比对的 FNV 改为**每决策一次**（6,598+1 次）；
- **每帧**仍保留一个 C 级摘要 `pixel_frame_blake2b128`（`hashlib.blake2b(digest_size=16)`，实测 **1.028 ms**/帧），所以 13,195 帧**帧帧**都有可复算的像素身份。
实测结果：教师 487.657 s（脚本外侧 504.745 s），**落进 600 s**。字段零缺失（对上一轮 `result.json` 的 keys 做差集：决策行/frame 行/`audit`/`cost` **0 删除**，只新增）。
**偏差**：验收脚本里教师写 `--timeout-s 900`（安全网）。工单的 600 s 由**实测墙钟 504.745 s** 满足；900 未被触发。

### D4 工作区既有差异（非本单产生，只报告）

`git status --untracked-files=no` 3,893 行（WSL 视角；Windows git 因 `core.autocrlf=true` 记 3,515–3,516 行）。内含 `Document/Update/Alpha` 3,698 条（大量历史版本目录被删除）、`tools/rl` 82 条、`Document/Dialogs` 33 条，以及 `touhou_rl/**` 若干 `M`。**本单开始前即存在**，v8.30 回执已记 3,516 行同一批。本单**未**清理、未提交、未修改其中任何条目（唯一的 `Document/` 写入是本回执与 A 节任务卡两份新文件）。

### D5 A 节任务卡的一处事实性错误（主代理复核后已更正）

任务卡初稿写"`configs/v430/splits.json` 全部 360 条 `ancestors` 只带 `task_family=th08-stage1-normal-team0`"。**实测不成立**：360 条跨 **8 个 stage 族**（stage1 242 / stage2 16 / stage3 16 / stage4a 17 / stage4b 16 / stage5 16 / stage6a 17 / stage6b 20）；train 328 条里 stage1 只有 **210** 条，来源是 `teacher_query` 225 / `dagger_round` 98 / `policy_rl` 5。已按实测改写三处（任务卡 §1.3、§4 两行），并把"dev 20 与 quarantine 12 全为 stage1、train 同族 210 条"作为"未见祖先"结论的新依据。同时复核通过未改的：`splits` 分类计数（train 328 / dev 20 / confirm 0 / adapt 0 / quarantine 12）、`reserved_seeds 1020–1039`、禁训 60000–60199、封存 `present_on_host=false`、`pretraining_sources` 25 条曝光直方图（17/2/3/1/1/1）。

---

## 8 本单**不**主张什么

1. **不主张像素策略有效**：像素轨迹是恒 `cand=12` 的固定策略，跑满 600 帧、`score=240`、`graze=0`、`stage=0`。它只证明"真实像素 → actor 输入 → 动作 → 实际帧 → 独立终态"这条链可运行。
2. **不主张教师成功是策略成绩**：13,195 帧的过关是**教师磁带重放**（`not_a_gate`/`not_a_policy_result`），只证明驱动与裁判的覆盖范围，不证明任何学生能过关。
3. **不主张"短阶段"存在**：全单没有找到段内完成事件；A 节只冻结不可变项，"短"仍是空的（§2 的分水岭）。
4. **不主张截断有真实见证**：本单像素轨迹的 `truncated` 是**设计上限**（600 帧到顶），不是教师 19,887 上限的真实触顶见证；仍是缺项 A3。
5. **不主张 `bombs_remaining=3` 的口径**：`initialBombCount` 与 `cfg.bombCount` 数值都是 3，本单无法区分，标**待核验**。
6. **不主张静音的哈希中性**：本单没有重跑 `R18` 那类中性实验；静音只按 §6.2 的**字节证据**成立。
7. **不主张 `observed_input_sha256` 是"策略"摘要**：它只是每决策一次 `play/hud/act_hist` 的 uint8 原字节 digest，用于与冻结实现对照。
8. **不主张两轮作废运行可以忽略**：它们占掉了工单的原预算、把写入量推过 2 GiB，并把引擎启动从 2 次推到 6 次（§5.0）。

---

## 9 证据索引（全部为训练端精确路径）

| 结论 | 路径 | sha256 / 身份 |
| --- | --- | --- |
| 新工具 | `tools/rl/v8_31_slice.py` | `19888a274c3508c8fabc21452aa5959b9916b84b75ba39ce287d0b48026430d0`（3,624 行） |
| 新测试 | `tests/test_v8_31_slice.py` | `2c51143f8a22781c9af2cf8cb9ac1a0cb9a5e55066c9b5a7d08cbb43b3a630e3`（1,899 行） |
| A 节任务卡 | `Document/Update/Alpha/v8.31.0/TASK_CARD_SLICE_A.md` | 见 `ASSET_SYNC.md` §38 |
| 像素接线结果 | `.tmp_v831_slice_01_YmVD4nBX/pixel/result.json` | `e5d17f3bfdd5a8f01720bf08f9e73305b956559cec9148e3382bc3972c9b948d` |
| 像素接线逐决策 | `.tmp_v831_slice_01_YmVD4nBX/pixel/decisions.jsonl` | `b426ce6d1584c69c27796500f6efc863e7d2899a2fc363a13d363ba0490de889`（300 行） |
| 像素接线静音 | `.tmp_v831_slice_01_YmVD4nBX/pixel/mute.json` | `b07dd6933b58c8a7ee3b9aa4cda352a4d021e902f144cd65ea3bf4177eb76c67` |
| 教师校准结果 | `.tmp_v831_slice_01_YmVD4nBX/teacher/result.json` | `64d91af4ba3e98ac0ecb18e94163ec79312e4b24c5a75a7fcb29410976130b01` |
| 教师校准逐决策 | `.tmp_v831_slice_01_YmVD4nBX/teacher/decisions.jsonl` | `d774338bd7ceef4585402cdd146841f7419f2a92d7d1a8771f1fdd96899d8498`（6,598 行） |
| 教师校准静音 | `.tmp_v831_slice_01_YmVD4nBX/teacher/mute.json` | `92b851b6ee4af65cb35c043052cdb0676b83a05c8e1b207f95fd6e99f63e9301` |
| 最终验收脚本 | `.tmp_v831_slice_01_probe/v831_acceptance_wsl_v2.sh` | 见 `ASSET_SYNC.md` §38 |
| 最终验收日志 | `.tmp_v831_slice_01_probe/acceptance_wsl_run3.log` | `c196f970bddfbe69a4535577351cbf15ad6a6a905076585aed5d5b1b73643afb`（3,907 行） |
| 工单命令块逐字执行的失败证据 | `.tmp_v831_slice_01_probe/acceptance_verbatim.log` | `70822df013556b8741d181b5a1e9f02df8f079d4e2580f84f85b75e3c948db41`（3,925 行，退出码 2） |
| 第 1 轮作废记录（技术无效） | `.tmp_v831_slice_01_lV7zfs9S/{pixel,teacher}/result.json` | 首冲突 `decision_index=0, frame=1` |
| 第 2 轮记录（像素通过 / 教师超时） | `.tmp_v831_slice_01_eK6pxCuY/{pixel,teacher}/result.json` | 教师 `terminal={"kind":"unknown","reason":"timeout","frame":9582}` |
| 历史正对照（外部实现） | `.tmp_v420_evalpulse/positive_control.json` | `9cdd90335edcc5a5d2afcf3b1e18d7713e921099967dbf4c62b5b84ef3dbaf63` |
| 引擎侧机理 | `/root/Touhou_RL/third_party/th08-rl/src/{rlharness/rlharness.cpp,main.cpp}` | 行号见 §7 D2 |

**未写入**：`/root/v400-data/game`（源）、`.tmp_v400_d1`、`.tmp_v410_*`、`.tmp_v420_data/**`、`.tmp_v830_*`、`Document/Update/Alpha/v4.*`、`configs/**`、`touhou_rl/**`（`git status` 可复核：本单只新增 2 个代码文件和 2 个文档）。

---

## 10 当前可用时段、可独立推进项、下一单必需输入

### 10.1 可用时段（训练端实测）

本单执行窗口 2026-11-25 23:04 – 2026-11-26 08:57 +08:00，期间无中断、无 OOM、无残留进程（结束时 `pgrep -af th08-modern` 为空、`/dev/shm` 残留 **0**）。磁盘：`/dev/sdd` 900 G 可用、`/mnt/d` 1.1 T 可用。本单占用约 2.80 GiB。**未对后续批次做任何并发或时长承诺。**

### 10.2 可独立推进项

1. **A 节三条缺项已被本单闭合**（A1 编号基、A2 双口径、A6 帧 0 资源）——请研究端据此重写任务卡 §5 的"随 B 一起回报"部分。
2. **A4/A5 需要一次产品决策**：接受整段 stage 1（19,887 上限），或授权重录"一决策一 mask"的干净磁带以切中局子窗。后者是独立资产工单，本单未做。
3. **A8 需要 split 变更**：把 1020–1039 登记为预注册开发根（或另立新种子段），并重算 `configs/v430/splits.json` 的登记摘要——属研究端口径，训练端只读。
4. **像素学习试点（W4）前置已满足一半**：接线真实通过（本单）、同约束教师真实成功（本单）。剩"短任务冻结"（A4/A5）与"开发根分配"（A8）。
5. **可并行的成本工作**：若要让教师轨迹回到 9.2 ms/帧量级，唯一已知手法是去掉每帧像素摘要；本单已给出"每决策 FNV + 每帧 blake2b"的折中实测（487.7 s）。是否冻结为契约由研究端定。

### 10.3 下一单必需的输入

| # | 需要什么 | 为什么 | 谁提供 |
| ---: | --- | --- | --- |
| 1 | **完成端点裁定**：`stage_clear` 以 `EVENTS` 过关屏（13,118）还是 `STATE.stage` 前进（13,195）为准，或"两者都记、以 X 为准" | 本单实测两者差 **77 帧**；不定就会各自报不同帧数 | 研究端 |
| 2 | **600 s 上限是否重标定**：本单实测 504.745 s（新仪器）vs 冻结仪器 121.573 s；若下一单要求"每帧都算引擎可比 FNV"，需 ≥950 s | 上限与仪器绑定，混用会误判"超时=失败" | 研究端 |
| 3 | **A4 路线选择**：整段 19,887（无需新资产）／重录干净磁带切子窗（需资产工单） | 决定 W3/W4 能否冻结 | 研究端 |
| 4 | **A8 开发根分配**：登记哪一段种子为"未参与拟合或选参" | 没有它，任何"泛化"结论都要降级为"同族复现" | 研究端 + split 变更授权 |
| 5 | **工单验收命令的 pytest 拆分**：把 5 个文件按解释器分成两段 | 现写法在 WSL 必然 rc 2，任何执行代理都会卡在这一行 | 研究端 |
| 6 | **`bombs_remaining` 口径确认**：`initialBombCount` 还是 `cfg.bombCount` | 两者数值都是 3，本单无法区分 | 研究端（源码侧） |

---

## 11 分类声明（工单第 109 行）

- **训练端本次实测**：§1 全部身份、§3 两条轨迹的全部计数/帧序/动作/摘要/退出码、§4 双信号与 77 帧间隔、§5 全部墙钟与观测覆盖、§6.1 帧 0 与末态资源、§6.2 静音与源目录不变、§6.3 真实轨迹侧隔离证据、§7 D1/D2/D3/D5 的命令与数字。
- **训练端本次离线检查**（无引擎）：§3 的 106 个离线测试、§6.3 的动态隔离扰动、§7 D3 的三个微基准（`fnv1a64` 51.268 ms / `blake2b-128` 1.028 ms / 合并 52.018 ms，60 次中位数）、D1 的逐文件 pytest 结果、D5 的 splits 直方图复核。
- **历史记录**：§2 与 A 节的教师见证、策略分母 0/160、R74 前缀结论、`CLEAR_REMEASURE` 的 19,887 上限、`pretraining_sources` 曝光分类、`splits.json` 分类计数。
- **待核验**：`bombs_remaining` 口径；`stageClearEdgeFrame` 与 `STATE.stage` 在同一次运行之外的稳定性（本单只 1 条教师轨迹）；rl_v3 是否在所有退出路径上都执行了 `PublishState`（本单只覆盖"正常跑满后过关"这一条）。
