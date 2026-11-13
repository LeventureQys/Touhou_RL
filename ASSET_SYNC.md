# ASSET_SYNC — 当前工程资产与外部运行同步说明

> 本文件描述 **本机当前实际可用** 的工程资产。v4.1.0 的资产（引擎、数据集、checkpoint、运行目录）**不会上传到 GitHub/远端**；远端只能同步代码与文档，资产必须在同一台机器上，或由你显式复制/挂载到相同路径。
>
> 机器上的状态：目标已 **paused**（第 102 轮），G410-0 已 passed；G410-1 教师 100/100 过面但 8px/数据腿未结；G410-2…G410-7 未通过。不要因为本文件存在就认为某个门通过。

## 1. 主机与路径

| 项 | 值 |
|---|---|
| Windows 工作区 | `D:\WorkShop\Touhou_RL` |
| Windows Git | 分支 `v400`，HEAD `d825f84`（工作树有既有 dirty/删除；不要清理） |
| Git remote | `https://github.com/LeventureQys/Touhou_RL.git`（只同步代码/文档，资产不推） |
| WSL | Ubuntu，`/root/Touhou_RL`（HEAD `d55c26a`，34 个 dirty 项；WSL 检出与 Windows 检出分开） |
| WSL 数据根 | `/root/v400-data` |
| WSL Python | `/root/v400-venv/bin/python`，Python 3.10.12，torch 2.14.1+cu130，numpy 2.2.6 |
| Windows Python | `python`，Python 3.12.10，torch 2.14.1+cu132，numpy 2.5.4 |
| 硬件 | 20 vCPU / 23 GiB WSL；RTX 4070 SUPER 12,282 MiB；驱动 617.42 |
| 仓库根资产索引 | `Document/Update/Alpha/v4.1.0/evidence/WP00/assets.json`（WP00 已核验） |

## 2. 引擎（WSL，最关键、不可从 Git 恢复）

| 名称 | 路径 | SHA-256 | 用途 |
|---|---|---|---|
| rl_v3 | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern` | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` | 学生评测/DAgger/rl_v3 数据 |
| rl_laser | `/root/Touhou_RL/third_party/th08-rl/build/rl_laser/th08-modern` | `1462829ba756bab49aae0878686988687dbaa109ea1750729d1202b555340d2f` | 教师 fast/v2h、G410-1 确认 |
| rl（旧） | `/root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern` | 见 WP00 assets；不要替换 | 旧 v400 实验 |
| Windows 零售/构建 | `third_party/th08/resources/th08.exe`、`third_party/th08/build/th08.exe` 等 | 见 WP00 assets | G0.1/Windows 腿 |
| Windows play 数据 | `third_party/th08/build/play/th08.dat`、`thbgm.dat` | 见 WP00 assets | 零售复验 |

WSL 的 `third_party/th08-rl` 被 `.gitignore` 忽略（`/third_party/th08-rl/`），**不在 Git 中**。

## 3. 数据集与运行目录（Windows `.tmp_*`，gitignored）

| 路径 | 大小（约） | 内容/状态 |
|---|---:|---|
| `.tmp_v400_d1` | **28.76 GB** | canonical D1：137 shards / 469,216 decisions / 1,876,864 frames（manifest 147 条，10 条为重复/隔离）；完整 gate 在 `work/gate.full.json`；D1 label 是旧 schema |
| `.tmp_v410_s1c` | **2.34 GB** | D410-S1：20 个 train seeds（1–20）/ 66,143 decisions；干净 train-seed 采集；S1 权威数据 |
| `.tmp_v410_wp05_round1` | **2.85 GB** | DAgger round1：16 episodes / 56,000 decisions / 112,000 frames；β=1.0；HASH 112,000/112,000 |
| `.tmp_v410_wp05_round2` | **2.67 GB** | DAgger round2：15 episodes / 52,500 decisions / 105,000 frames；β=0.5；injection 0.500857；HASH 105,000/105,000 |
| `.tmp_v410_wp05_round3` | **2.6 GB** | round3：**15 episodes / 50,787 decisions / 101,574 frames**（seeds 2040–2054；14 集 decisions、1 集 death_stop@2048）；β=0.5；injection 0.502629；HASH 101,574/101,574；已完成（计时探针在单独 timing/，不计入） |
| `.tmp_v410_wp04_arms` | **0.08 GB** | 四臂 12 个 checkpoint（B0/B1/B2/B3 × seed0–2） |
| `.tmp_v410_wp05_train50k` | **0.28 GB** | 50k round1 的 labels/checkpoint（`6ebe4ea9…`） |
| `.tmp_v410_wp05_train_beta05` | ~0.10 GB | 1k β=0.5 训练（`a11ba0b7…`） |
| `.tmp_v410_wp05_train_beta05_5k` | ~0.10 GB | 5k β=0.5 训练（`692fdc61…`） |
| `.tmp_v410_wp05_train_beta05_20k` | ~0.15 GB | 20k β=0.5 训练（`15ae1fd5…`） |
| `.tmp_v410_wp05_classweight` | ~0.10 GB | idle 降权 5k（`f447664c…`） |
| `.tmp_v410_wp05_tiebreak` | ~0.15 GB | tie-break 5k（`adc4fa9b…`）+ 20 dev |
| `.tmp_v410_g4101_tol8` | ~12 MB | 8px 校准审计（dev seed1000） |
| `.tmp_v410_wp05_idle_diag` | 小 | idle 诊断脚本/JSON |
| `.tmp_v400_trace_batch` | 74 KB report + 运行目录 | G0.1 108/108（`report.json`） |

WSL 数据：

| 路径 | 大小 | 说明 |
|---|---:|---|
| `/root/v400-data/game` | **474 MB** | 学生/DAgger 用的游戏数据（不要与 canonical D1 混淆） |
| `/root/v400-data/g1` | 1.2 MB | 教师 G410-1 的 inst0..5 小数据目录（score/cfg，不是数据集） |
| `/root/v400-data/d1` | 21 MB | **陈旧镜像，不是 canonical D1**；不要用它替代 `.tmp_v400_d1` |

## 4. 关键 checkpoint / SHA（本地，不入 Git）

| checkpoint | SHA-256 | 说明 |
|---|---|---|
| `.tmp_v410_wp04_arms/b0_seed0_last.pt` | `3bc0ec4461701bebb2ffc551d61dffff77f15cd486125baf9db6eaafc741b3a6` | 四臂 B0 基线（DAgger round1 的 rollout 学生） |
| `.tmp_v410_wp05_train50k/b0_s0_dagger_step1000.pt` | `6ebe4ea91d81d9333bfa1ccc065fc3515bb25b1d3b90a2684cdaa8a0a69b8086` | β=1.0 50k，1,000 步 |
| `.tmp_v410_wp05_train_beta05_5k/b0_s0_dagger_step5000.pt` | `692fdc61dc473ff10657d2a0b1a37434d3b24561fedb9798a41a38086bdbd120` | β=0.5，5k |
| `.tmp_v410_wp05_train_beta05_20k/b0_s0_dagger_step20000.pt` | `15ae1fd5b9a6e0cbbddfd668bb342444bdb44db123b5f7e56abf3c5e7215bd95` | β=0.5，20k；round3 的 rollout 学生 |
| `.tmp_v410_wp05_classweight/b0_s0_dagger_step5000.pt` | `f447664c660933af9c2840dd2c8e43051e9757c2f520fa9706178c92b53a4919` | idle 降权 5k |
| `.tmp_v410_wp05_tiebreak/b0_s0_dagger_step5000.pt` | `adc4fa9b8b5f2b57e059a99f81e0082156737593db92f962adabc1f07857e4d0` | tie-break 5k |

## 5. 证据与状态（可同步/可上传的是这些文档，不是资产）

- `Document/Update/Alpha/v4.1.0/README.md`：v4.1.0 唯一规格（877 行）。
- `Document/Update/Alpha/v4.1.0/state.json`：当前 WP/门状态；**目标 paused**。
- `journal.jsonl` / `budget.jsonl`：历史与预算记录。
- `Document/Update/Alpha/v4.1.0/evidence/`：
  - `G410-0/`：G0.1 收尾；
  - `WP00/`：资产索引（`assets.json`）；
  - `WP01/`：协议/一致性/边界；`G0.1.md`、`g0.1-108.json`（108/108）；
  - `WP02/`：教师；`G4101_100_TEACHER.md`（100/100）、四批 chunk + 8px 审计；
  - `WP03/`：S1/MULTI 计划与数据审计；
  - `WP04/`：四臂/20 dev；
  - `WP05/`：DAgger 数据、训练、20 dev、idle 诊断、tie-break/class-weight；`round3.json`/`ROUND3.md`（50,787 decisions，已完成）；
  - `WP06/`：事件/状态机契约（G410-3 未实现）。
- `ASSET_SYNC.sha256`：关键文档/checkpoint/引擎的校验清单。

## 6. 当前能力/缺口（给外部代码用）

**可用**：
- 真实 TH08 引擎 rl_v3/rl_laser、G0.1 108/108 零售证据；
- D410-S1 66,143、DAgger round1 56k / round2 52.5k、round3 50,787（已完成）；
- 学生 evaluator、DAgger 采集器、teacher label/verify、四臂/多种 DAgger checkpoint；
- v4.1 状态/证据/预算工具与 108/108 报告。

**不可用/未通过**：
- G410-1 的 100 种子 8px 全量审计、PSN/多面数据腿；
- G410-2：没有任何学生 20 dev 过关（最好 0/20）；G410-3…G410-7 未开始；
- 连续逐帧像素事件/终局 detector、Windows 部署、完整一周目、严格像素 RL、留出/第二作；
- 确认种子 61000–61099 与 61400–61499 仍未使用（不要读）。

## 7. 外部写代码、远端运行的同步协议

### 7.1 只同步代码/文档
资产（`.tmp_*`、`/root/v400-data`、引擎二进制、checkpoint）**不要上传、不要塞进 Git**。`.gitignore` 已忽略 `.tmp_*/`、`/third_party/th08-rl/`。

从本机同步代码到远端（示例，按需改 remote/路径）：
```powershell
# 只同步跟踪的代码/文档；排除 .tmp 与 build
git diff --binary -- touhou_rl tools tests configs Document > code.patch
# 或 rsync（WSL/Git Bash）
rsync -av --exclude .git --exclude '.tmp_*' --exclude 'third_party/th08-rl' `
  ./ user@remote:/workspace/Touhou_RL/
```
远端只运行代码时，unit tests 可跑；任何需要引擎/数据/checkpoint 的测试或实验会因缺资产失败——这是**资产缺失，不是代码失败**，不要用 fake/mock 冒充真实结果。

### 7.2 资产必须在远端可用
若远端要跑真实实验，需把以下之一挂载/复制到相同路径：
- 同机 WSL：直接使用 `/root/Touhou_RL` 与 `/root/v400-data`；
- 另一台机器：复制代码 + 对应引擎二进制 + 数据根 + checkpoint，并保持 CLI 路径一致；
- 至少复制 `third_party/th08-rl/build/rl_v3|rl_laser/th08-modern`、`/root/v400-data/game`、需要的 `.tmp_v410_*` 数据/ckpt。

### 7.3 校验
用 `ASSET_SYNC.sha256` 校验关键文件：
```bash
sha256sum -c ASSET_SYNC.sha256
```
引擎在 WSL 用绝对路径行校验；数据目录很大，按 `evidence/WP00/assets.json` 与各 `shard.json` 内部 SHA 校验，不要全目录哈希。

### 7.4 资产变更规则
- 新增/移动/删除任何 `.tmp_v*` 数据集、checkpoint、引擎构建，必须同步更新本文件与 `ASSET_SYNC.sha256`。
- 不得删除/覆盖 `evidence/`、`state.json`、`journal.jsonl`、`budget.jsonl`；外部代码只能写自己的新路径。
- 远端代码若产生新的资产，必须明确写回本机路径与 SHA，不能只留在远端。

## 8. 相关入口

- 规格：`Document/Update/Alpha/v4.1.0/README.md`
- 状态：`Document/Update/Alpha/v4.1.0/state.json`（paused）
- 资产索引：`Document/Update/Alpha/v4.1.0/evidence/WP00/assets.json`
- 当前进度摘要：`Document/Update/Alpha/v4.1.0/evidence/WP05/STATUS.md`
- 校验清单：`ASSET_SYNC.sha256`

> 校验快照：`ASSET_SYNC.sha256` 对应 **2026-11-13T16:20+08:00 的暂停状态**（已实测 `sha256sum -c` 全部 OK）。一旦 resume 并更新 `state.json`/`journal.jsonl`/`budget.jsonl`，必须重算对应行（或整表），否则校验会失败。
