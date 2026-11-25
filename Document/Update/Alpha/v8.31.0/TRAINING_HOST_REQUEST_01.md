# V831-SLICE-01 短任务提名与真机接线校准

2026-11-25 · GPT 研究端 → DeepSeek 训练端 · 对应 [v8.31.0 开发计划](README.md)。用户转交并要求执行后，训练端主代理按 `AGENTS.md` 将实际操作派给 Provider=DeepSeek、模型=DeepSeek v4.1 flash 的执行代理。请在回执引用完整工单号。

## 目标和允许范围

从已登记材料提名一个可用于像素学习的真实短攻击阶段。同时交付一条自然开局的真实像素接线轨迹和一条独立的整面教师判分校准轨迹。两个工作可独立交付。

- 只读范围为 v8.30 回执与提案、T08 已登记教师记录、`configs/v430/splits.json`、R74 恢复证据，以及这些索引指向的精确文件。先查已知来源，按需读取少量原始行或已登记哈希。 [v8.30 回执](../v8.30.0/TRAINING_HOST_REPLY_01.md) · [T08 教师正对照](../v4.2.0/evidence/T08/EVAL_SHOOT_PULSE.md) · [R74 前缀检查](../v4.3.1/evidence/R74/pair_cand_scope.json)
- 代码写入白名单只有 `tools/rl/v8_31_slice.py` 和 `tests/test_v8_31_slice.py`。复用现有 harness 客户端、`ObservationBufferV3`、`ActionHistoryBuffer` 和 `teacher/shoot.py`。通过新工具适配原始 `EVENTS` 字段，保留已有协议、ABI、依赖及第三方。 [既有接线提案](../v8.30.0/TRAINING_HOST_PROPOSAL_R2_P1.md)
- 每次执行用唯一 `.tmp_v831_slice_01_*` 目录。两个进程分别使用自己的私有游戏目录。允许生成本次最少量的日志、清单和少量帧证据。新资产依 `AGENTS.md` 第 8 节只追加本机登记，关键产物写 SHA 到 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`。资产、完整原始日志、引擎和权重留在训练端。
- 本单禁止训练、选模、前缀重录、批量评测、恢复 WP05、提交或上传。既有资产、证据、暂停快照和源 `th08.cfg` 作为只读输入。新代码只清理本单自己创建的引擎进程。

## A 短阶段任务卡

从 `V830-ROUTE-INFO-01` 已回答的内容继续，只提名一个任务。优先在现有教师记录和脚本事件中寻找最早可辨识的攻击完成点。先冻结起点、终态和合法动作，再考虑候选策略。请给出以下字段。

| 字段 | 所需证据 |
| --- | --- |
| 起止和分母 | 攻击或阶段身份、起点类型、首受控帧、结束事件、失败与截断定义、每局候选受控帧上限 |
| 任务信号 | 现有成功及失败或不同后果各一例，给 run ID、原始事件和帧。说明可观测完成如何区别于自动推进或超时 |
| 入口和资源 | 自然起点或已登记前缀 SHA，实际姿态和资源、恢复哈希、像素帧栈与动作历史。有冲突时给首个帧号和具体掩码 |
| 祖先用途 | 已可划出的训练根和未参与拟合或选参的开发根，简列谱系、曝光及冲突。1000–1019 属旧开发面板。正式确认、适应保留及禁训种子按原登记约束 |

如只找到整面事件，或只有双 mask 冲突的中局前缀，交付缺项和最小补证动作。现有教师整面成功用于 B 的裁判校准。A 的缺项可以与 B 的接线结果一起回报。

## B 两条实机轨迹

两条轨迹使用登记的 `rl_v3`，从 Normal、Border Team 自然开局，`stage=0`，`shotType=0`，非 practice。诊断只用 18 类移动和 focus 动作，`bomb=0`。每决策 2 帧，每 48 决策最后一帧松开 SHOOT 和 BOMB。采用 `release_due` 与 `commit_frames` 的既有规则。播放历史教师表时保留其登记的起始决策和尾部处理，报告实际初始资源。

| 轨迹 | 固定输入 | 实机上限 | 验收 |
| --- | --- | --- | --- |
| 像素接线 | seed 1000，恒动作 #12 | 600 受控帧，120 秒 | 真实像素进入 actor，帧序、动作生效与历史一致。按实际状态结算合法死亡或截断 |
| 教师校准 | seed 12，T08 既有动作表 | 19,887 受控帧，600 秒 | 取得真实完成并记录 `EVENTS` 过关边沿与 `STATE.stage` 前进帧，报告两者的顺序、间隔及当时的动作和资源 |

最多启动这两个引擎进程，各一条。写入上限合计 2 GiB，含两份私有游戏目录。预算是本单上限。启动前记录空间和可用时段，资源不足时执行可独立容纳的一条并回报。每条轨迹计入所有已启动尝试，合法死亡保留原样，技术无效保留首个冲突。已冻结的端点与预算随运行身份记录。

### 帧与判分验收

1. 像素接线从自然开局使用 `obs-v3` 既有初始填充，重复最早可用画面，接管前动作历史用零行。填充不额外推进游戏帧。记录首张实际画面和首个受控帧。此后帧栈使用相邻实际画面。教师重放另保留 T08 的启动动作顺序。遇到中局恢复需求，将真实前缀和历史重建写入下一单任务卡。
2. 每次决策记录逻辑帧、像素帧、原始画面校验值、预处理后的 actor 输入摘要、请求动作、每段实际 `STEP` mask 和生效帧数。一次脉冲决策可有两段 mask，动作历史仅在候选实际生效后记一行。用固定像素与动作历史扰动裁判字段，检查 actor 输入摘要、记忆和输出均保持一致。
3. 判分使用 `EVENTS` 的 `0x0080` 与 `stageClearEdgeFrame`，独立记录 `STATE.stage` 首次前进帧。观察到过关边沿后，在原帧预算内继续观察交叉校验信号。记录实际终止时的两套信号和事件间隔。若第二信号在预算内缺失，保留缺项和原始小窗，由研究端判断差异。
4. 每条轨迹报告启动、恢复、读回、actor、步进、审计和 I/O 的墙钟与总时间。其中恢复或读回未发生时记相应项为零。另给环境终止、帧上限截断、未知和技术无效的计数。600 帧轨迹验收接线，教师轨迹验收整面判分，短任务仍需同入口、同契约的真实成功对照。 [像素与判分契约](../v8.30.0/PIXEL_SLICE_PLAN.md)

### 私有目录静音

复制所需游戏数据到新私有目录，只对副本的 `th08.cfg` 将 `0x1F`、`0x20`、`0x27`、`0x28` 置零。记录源与副本 SHA、实际字节差和其他字段一致性。历史源为 `48bf7621…` 时，副本应匹配 `cfa292ae…`。源身份变化时按当前登记核对并记录差异。首次启动前和进程退出后都针对本条轨迹的私有 cfg 核对四项为零，进程环境设 `SDL_AUDIODRIVER=dummy`。可复用既有四项解码函数，报告写入本次运行目录。 [静音字节和历史身份，回执第 10 节](../v8.30.0/TRAINING_HOST_REPLY_01.md)

## 验收命令

在实际执行检出根将下列命令作为独立 WSL Bash 脚本执行。命令中的二进制、Python 和数据路径来自既有登记。路径有变化时，先在回执绑定实际检出与资产身份。新工具使用下列 CLI，按 B 节逐轨迹核验空间、静音与进程时限，再启动真机。

```bash
set -eu
V831_PY=/root/v400-venv/bin/python
V831_ENGINE=/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern
V831_DATA=/root/v400-data/game
V831_TAPE="$PWD/.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed12-shard/slots.jsonl"
V831_RUN=$(mktemp -d "$PWD/.tmp_v831_slice_01_XXXXXXXX")
export PYTHONPATH="$PWD"
export SDL_AUDIODRIVER=dummy

git rev-parse --show-toplevel
git rev-parse HEAD
git status --short --untracked-files=no
"$V831_PY" -B -c 'import sys; from datetime import datetime; print(sys.executable, sys.version.split()[0], datetime.now().astimezone().isoformat())'
printf '%s  %s\n' '691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6' "$V831_ENGINE" | sha256sum -c
"$V831_PY" -B -c 'from touhou_rl.v400.student.obs_spec_v3 import obs_spec_v3_digest; print(obs_spec_v3_digest())'

"$V831_PY" -B -m pytest -p no:cacheprovider --basetemp "$V831_RUN/pytest" -q \
  tests/test_v8_31_slice.py tests/test_v400_obs_v3.py \
  touhou_rl/tests/test_framefence.py touhou_rl/tests/test_frame_proof.py \
  touhou_rl/tests/test_phase2_history_proof.py

V831_PIXEL_RC=0
"$V831_PY" -B tools/rl/v8_31_slice.py pixel-smoke \
  --engine "$V831_ENGINE" --data-source "$V831_DATA" --out "$V831_RUN/pixel" \
  --stage 0 --difficulty normal --shot-type 0 --seed 1000 --constant-cand 12 \
  --decision-frames 2 --shoot-pulse-period 48 --max-controlled-frames 600 --timeout-s 120 || V831_PIXEL_RC=$?

V831_TAPE_RC=0
printf '%s  %s\n' 'f759de7dacc78343dec8825da5de14942e749b8a5a2c5319304e513d1618b7b9' "$V831_TAPE" | sha256sum -c || V831_TAPE_RC=$?
V831_TEACHER_RC=not-started
if [ "$V831_TAPE_RC" -eq 0 ]; then
  V831_TEACHER_RC=0
  "$V831_PY" -B tools/rl/v8_31_slice.py teacher-calibration \
    --engine "$V831_ENGINE" --data-source "$V831_DATA" --out "$V831_RUN/teacher" \
    --stage 0 --difficulty normal --shot-type 0 --seed 12 --tape "$V831_TAPE" \
    --decision-frames 2 --shoot-pulse-period 48 --max-controlled-frames 19887 --timeout-s 600 || V831_TEACHER_RC=$?
fi
printf 'pixel_rc=%s tape_check_rc=%s teacher_rc=%s\n' "$V831_PIXEL_RC" "$V831_TAPE_RC" "$V831_TEACHER_RC"
[ "$V831_PIXEL_RC" -eq 0 ] && [ "$V831_TEACHER_RC" = 0 ]
```

逐项留实际命令与退出码。两条轨迹共用引擎、观测与静音前置检查。教师磁带校验失败时仅跳过教师轨迹。离线测试覆盖首帧填充、输入隔离、第 48 次决策的两段 mask、截断、合法死亡和缺失事件，真机证据另外验收。执行代理完成或遇阻后立即回执，训练端主代理独立复核范围、命令和统计。

## 回执格式

```text
V831-SLICE-01 / 主机代号 / 时间及时区 / 执行代理及模型
实际检出 / HEAD / 差异摘要 / Python / 引擎与数据身份 / cfg 与 obs SHA
A 任务卡 / 源路径与证据 / 谱系与曝光 / 是否可冻结 / 最小缺项
B 像素接线与教师校准各自的 run ID / 实际命令及退出码
  计划、启动、有效、合法失败、技术无效、未启动计数
  帧序与实际 STEP 序列 / 双过关证据与首个差异 / 资源 / 耗时 / 产物 SHA
当前可用时段 / 可独立推进项 / 下一单必需输入
```

请区分训练端本次实测、训练端本次离线检查、历史记录和待核验。回传小表和必要边界证据，文件和目录身份使用训练端精确路径。研究端据此冻结短任务与像素学习工单。
