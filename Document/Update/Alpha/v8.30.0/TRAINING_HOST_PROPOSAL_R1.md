# R1 实施提案（训练端草稿，**待研究端批准**）

**性质**：提案，不是实施单，也不是开工通知。训练端在本文件里把已经核实的事实填成可勾选的参数，让研究端从"写单"降为"勾选/改几处"。
**依据**：`TRAINING_HOST_REPLY_01.md`（V830-ROUTE-INFO-01 回执，已推送至 `origin/v400`）。
**未做**：本提案**没有**启动引擎、评测、采集或训练；WP05 仍暂停。

> **为什么现在给提案**：`origin/v400` 连续 8 轮没有研究端新提交，而 `ROUTE_DECISION.md` 规定"研究端先据回执冻结任务与比较方式，再下达受限实施单"。为了不让路线空转，训练端先把**只依赖已核实事实**的部分填好。下面每个 `D?` 都标注了我建议的取值与理由，研究端可以直接勾选、改值或否掉。

---

## 决策清单（请逐项勾选）

| 编号 | 决策 | 训练端建议 | 备选 |
| --- | --- | --- | --- |
| **D1** | 任务切片 | **整段 Stage 1**：`stage=0 / Normal / shotType=0(Border Team)`，完成 = `stage_clear` | 重录一决策一 mask 的干净磁带 → 前缀重放 + `PARK` 切后期子窗（**需资产授权**） |
| **D2** | 动作契约 | **(a) 外生 SHOOT 脉冲**：`shoot_pulse_period=48`、`decision_frames=2`（= 96 帧），复用 `touhou_rl/v400/teacher/shoot.py` | (b) 扩展动作空间给策略 shoot 位并重登记任务（会使冻结基线失效） |
| **D3** | 主指标口径 | **`EVENTS` 的 `stageClear`(bit `0x0080`) + `stageClearEdgeFrame` 为主**，`STATE.stage` 计数为交叉校验；两者不一致时单列为发现 | 仅用 `STATE.stage` 前进（只给"阶段变了"，无精确帧） |
| **D4** | 每局受控帧上限 | **19,887**（已登记值，= 13,195 过关帧 ×1.5） | 更小上限会在过关前截断 |
| **D5** | 开发祖先 | **dev 面板 seeds 1000–1019**（20 根，已核实 0 训练泄漏） | 用 v8.29 的 8 盘自采根（起点不中立） |
| **D6** | 接管时帧栈 burn-in | **前缀重放 3 帧填满 `t-3..t`，`act_hist` 前置全零行**；burn-in 长度写进运行身份 | 由研究端指定其他 burn-in |
| **D7** | 静音实现 | **最小四字节改法**：把源 cfg 的 `0x1F/0x20/0x27/0x28` 置 0（`48bf7621…`→`cfa292ae…`，diff=4，映射/opts/资源不变） | 整模板替换（会清空 controllerMapping、改 opts） |

---

## D1 任务定义（若批准"整段 Stage 1"）

| 字段 | 冻结值 | 来源/依据 |
| --- | --- | --- |
| 阶段 | `stage=0`（Stage 1）、`difficulty=normal`、`shotType=0` = Border Team | `configs/v430/splits.json` 任务族名；零售元数据射击对照表 |
| 完成事件 | `stage_clear`，帧级边沿 `stageClearEdgeFrame` | 回执 §1 |
| 已知真实成功 | 教师磁带 period 48：13,195 帧 / 2,631,430 分 / 0 死 | `T08/clear_reachability.json` |
| 已知真实失败 | 160 局（4 臂×2 口径×20 dev seed）`stage_clear=0`，最深 3,914 帧 | `T08/CLEAR_REMEASURE.md` |
| 初始资源 | `livesRemaining=2`（共 3 次死亡）、`bombs = primaryShtFile->initialBombCount`、`power=0` | `GameManagerSetup.cpp:105-130`、`Player.cpp:1393-1400` |
| 禁止 | **不得用 practice 模式**（`lifeCount` 会被强制成 8） | `GameManagerSetup.cpp:112-113` |

**若批准 (B) 重录磁带**，则本提案整体推迟到资产授权之后——它需要新采集批次、`ASSET_SYNC` 登记，并重新验证前缀可达性（现有磁带从 frame 92 就有双 mask 决策）。

## D2 动作契约（若批准"外生脉冲"）

```python
from touhou_rl.v400.teacher.shoot import release_due, commit_frames
# 每决策：release_due(i, 48) 为真时，该决策最后一帧发送 mask & ~(SHOOT|BOMB)
```

必须写进运行身份并接受合并校验的三项：`shoot_pulse_period=48`、`decision_frames=2`、`release_rule="last frame of every 48th decision"`。**不同 pulse period 的摘要不得合并**（T08 裁决的 `must_also`）。

**声明义务**：选 (a) 时，"过关" = **像素策略 + 固定外生脉冲**，脉冲不由策略决定。任何结果都不得表述为"策略学会了触发过关"。

## D3 指标记录格式（每局一行，最小字段）

```text
run_id, ancestor_id, seed, stage, difficulty, shot_type,
shoot_pulse_period, decision_frames, burn_in_frames,
takeover_frame, controlled_frames, total_frames,
stage_clear_events_bit, stage_clear_edge_frame, state_stage_start, state_stage_end,
terminal(kind), deaths, lives_remaining, bombs_remaining, power_final,
engine_sha256, obs_spec_digest, data_dir, exit_code
```

`stage_clear_events_bit` 与 `state_stage_end > state_stage_start` **两条都要记**——这是 D3 的交叉校验，也是将来能独立复核的前提。

## D4 预算（按已实测数推算，非新测）

| 项 | 值 | 依据 |
| --- | --- | --- |
| 单局受控帧上限 | 19,887 | 已登记 |
| 单局墙钟 | **23–45 s**（T08 25 批 / 160 局实测区间） | `CLEAR_REMEASURE.md` |
| 单决策磁盘 | **52,112 B**（`BYTES_PER_DECISION`） | `obs_spec_v3` |
| 600 帧接线冒烟 | ≈ 300 决策 ≈ **15.6 MB**/局 | 推算 |
| 单局数据量（满帧） | 9,943 决策 × 52,112 B ≈ **0.52 GB** | 按契约常量估算的上界 |
| 端到端 20 局开发 | ≈ 8–15 分钟 + ≈ **10.4 GB** | 推算，需实测确认 |

**注意**：上表墙钟来自旧节拍（每决策 2 帧）的历史批次；本次首批必须**重新实测一条端到端**，再把规模冻结。

## D5 祖先分配

- 可用：dev 面板 **1000–1019**（20 根；已核实这 20 个种子在 17 个 `used_for_training` 来源里出现 **0 次**）。
- 不用：`confirm`（当前为 0）、种子 **60000–60199**（永久禁训）、`quarantine` 的 12 根（S1b）。
- 首批只用 2–4 根做接线与可行性观察；**不看新成绩**再决定是否加根。

## D6 我建议的 burn-in（须批准）

`obs-v3` 只钉死栈形（4 帧 `t-3..t`）与 `act_hist` 清零时机，**没有 burn-in 定义**。建议：

- 接管前**重放 3 帧**填入 `t-3,t-2,t-1`，第 4 帧为首受控帧；
- `act_hist` 在接管前填**全零行**（契约的 `HISTORY_PAD`）；
- `burn_in_frames=3` 写进运行身份；
- 不跨回合残留隐藏状态（每次独立恢复都重建同样的历史）。

这是**设计决定，不是既有契约**；研究端若另有取值，请直接在 D6 上改。

## D7 静音（开工前必须完成，缺一不得开跑）

1. 私有 data 目录放静音 cfg：`python tools\rl\v400_make_cfg.py --out "<data-dir>\th08.cfg"`，并 `--inspect` 确认四个音频字段全 0。
2. 启动环境显式 `SDL_AUDIODRIVER=dummy`。
3. 启动后跑 `python tools\rl\v431_audio_cfg_audit.py`，目标目录必须落在 `muted`。

源 cfg 若要替换，用 D7 建议的最小四字节改法；**这是资产操作**，须在研究端的单子里明确授权，并同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`（当前登记 `48bf7621…`）。新运行**不得增加 `audible` 计数**。

---

## 验收命令（批准后由训练端执行，逐条留退出码）

```bash
# 0. 身份（不启引擎）
git rev-parse HEAD
python -B -c "import sys;print(sys.executable, sys.version.split()[0])"

# 1. 静音前置（不启引擎）
python tools/rl/v431_audio_cfg_audit.py            # 目标 data 目录须为 muted

# 2. 冒烟一局（≤600 受控帧；接线用，不作能力结论）
#    <engine>/<data-dir>/<obs 路径> 由实施单填写
<runner> --stage 0 --difficulty normal --shot-type 0 --seed 1000 \
         --max-controlled-frames 600 --shoot-pulse-period 48 \
         --out <new-path>/smoke-seed1000.json

# 3. 正式切片一局（≤19,887 帧）
<runner> --stage 0 --difficulty normal --shot-type 0 --seed 1000 \
         --max-controlled-frames 19887 --shoot-pulse-period 48 \
         --out <new-path>/ep-seed1000.json
```

**产出与判读**（须逐项写进回执）：计划/实际/有效局数、恢复次数、技术失败与合法失败数、受控帧、各环节墙钟、端到端总耗时，以及 `stage_clear` 两条口径是否一致。

**四种结果对应四种下一步**（不做事后改口径）：

| 结果 | 结论 | 下一步 |
| --- | --- | --- |
| 冒烟就恢复/帧序/动作对不上 | 接线问题 | 定位首个冲突点，不进入学习 |
| 满帧但 `stage_clear` 从不出现且 `stage` 不动 | 任务/契约问题 | 先查 D1/D2，不归罪像素 |
| 教师路径同参数能出 `stage_clear`，策略不能 | 任务可解、策略未达 | 这才是 R2 学习的起点 |
| 两条口径不一致 | 仪器问题（未见过） | 单列为发现，先修一致性 |

---

## 本提案明确**不**主张

- 不主张路线可行、像素可学、或任何能力门状态改变。
- 不主张 D6 的 burn-in 是既有契约（它是我提的设计值）。
- 不主张上表墙钟能外推到 20 局批次——首批必须实测。
- 不主张 D2 选 (a) 之后"过关"可被表述为策略能力。

## 批准后训练端立即能做的（不需要再问）

按 D1–D7 生成运行身份、建私有 data 目录（含静音三件套）、跑冒烟一局并回传命令/退出码/产物身份。**在收到批准前，本提案不启动任何引擎。**
