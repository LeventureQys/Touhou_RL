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

## 9. v4.2.0 关键资产（成对登记，2026-11-15）

> **为什么有这一节**：v4.2.0 产出约 **16 GB** 承重新产物（实测 `du -sh --apparent-size .tmp_v420_data` = `16G`），
> 但在此之前 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256` **都没有任何 v4.2.0 条目**
> （`grep -c 'v420\|v4\.2\.0' ASSET_SYNC.sha256` = **0**，`ASSET_SYNC.md` 同样 **0**），
> 违反 AGENTS.md §8.2「新增/移动/删除任何资产，必须同步更新 `ASSET_SYNC.md` 与
> `ASSET_SYNC.sha256`」。本节与 `ASSET_SYNC.sha256` **末尾追加的 26 条**成对登记；
> 既有 20 条逐字节未变（`head -c 2375 ASSET_SYNC.sha256 | sha256sum` 仍为 `73ccf480d5db…`）。
>
> **26 条共同的新增原因**：v4.2.0 的承重 checkpoint / 语料 index / 状态快照此前要么只在内联
> 证据 md 里写过 SHA、要么完全没写，清单层面 **0 条校验记录**。本轮按 §8.2 补齐，登记体积合计
> **23.53 MiB**（对比 `16G` 的刻意最小集，占 **0.14%**）：只收「不可由 Git 或证据文档重建」的
> 二进制 checkpoint、每个语料的机器可读 index/汇总、以及三份状态快照；**不**收 1000 MiB 级
> `.npy`、`.npz` 观测样本、pytest basetemp、引擎 data 私有副本。
>
> **登记快照时刻**：`2026-11-15T04:18:27+08:00`（Asia/Shanghai）。26 条首次写入是 `04:17:26`；
> 其中 `state.json` / `journal.jsonl` 在写入后 **28 秒**就被 v4.2.0 自身推进追加过，已于
> `04:18:27` 重钉（见 §9.8）。这两份加上 `budget.jsonl` 是**可继续增长**的活文件，行内 SHA 只对
> 快照时刻成立；后续轮次一旦追加这三份文件，必须重算对应 3 行（见 §9.6 / §9.8，与第 8 节末尾
> 既有约定一致）。

### 9.1 七个承重 checkpoint（T04 / T05 / T06 / T08）

| 名称 | 路径 | 大小 (B) | SHA-256 | 用途一句话 | 新增原因 |
|---|---|---:|---|---|---|
| A0（T04） | `.tmp_v420_data/T04/checkpoints/a0_seed0_step5000.pt` | 2,940,699 | `96cb219eac83391f42b18ea656b5a6c5ac81e97cd1fd8c1e2c0d404aee6b89ad` | T04 受控学习第一轮的 A0 学生（5,000 步），T05 三轮 DAgger 的 rollout 起点 | v4.2.0 承重训练产物；`evidence/T04/TRAIN_CONTROLLED.md` 只内联 SHA，清单 0 条 |
| D2（T05） | `.tmp_v420_data/T05/checkpoints_d2/d2_seed0_step5000.pt` | 2,940,699 | `9f284e48c2304a519308b947f3d9e6705a49d9404dbc3ee5b596c2a4ea3751b6` | T05 第二轮 D2（β=0）学生自主访问轮的训练产物 | T05 轮次产物，清单 0 条 |
| D3（T05） | `.tmp_v420_data/T05/checkpoints_d3/d3_seed0_step5000.pt` | 2,940,699 | `91d11d22187bc25b15bc6eb36932aaf30f30c548ffa7ce076e7a57fbdd68a603` | T05 第三轮 D3（β=0）收尾轮学生；β=0 自主决策累计 51,671 的载体 | T05 最后一轮产物，清单 0 条 |
| r3@50k（T06） | `.tmp_v420_data/T06/checkpoints/r3_seed0_50000f.pt` | 2,948,551 | `83f1ac4436c7c0bcb3e03296b24dad915eecffa4a57f3e301f68927e4a4b4685` | 奖励 `v420-reward-r3/1` 下 50,000 帧 PPO checkpoint，advnorm / 200k 两档的对照基线 | T06 承重 RL 产物，清单 0 条 |
| r3@200k（T06） | `.tmp_v420_data/T06/checkpoints/r3b200_seed0_200000f.pt` | 2,948,761 | `35e92dfe90053a2dcee47bcf0f2d956efb5cd5d04a3ffa046ba2722b3dd09b01` | T06 预算单变量实验（50k→200k 帧）的 200,000 帧 checkpoint | 同上（预算维唯一变量档） |
| advoff（T06） | `.tmp_v420_data/T06/checkpoints/advoff_seed0_50000f.pt` | 2,948,719 | `f84e8b2e95a518d37f38d1552eca8065149c0b9b63ec63404a9ad86cf3cd4e73` | T06 N-a② `ppo.advantage_normalisation: true→false` 单变量 50k checkpoint（训练期特权诊断，**不是 R1**） | 同上（单变量消融档） |
| a1set（T08） | `.tmp_v420_data/T08/setsup/checkpoints/a1set_seed0_step5000.pt` | 2,941,273 | `7bacab5e23492f4bda7dc30cc8c256de8067ef25c6c08b559ed014219c2bcfba` | T08 A1 安全集合监督训练的 5,000 步学生，用于首次可报告 `stage_clear` 的过关测量 | T08 承重产物，清单 0 条 |

> **同名文件辨别（逐个实测，不是按名字猜）**：`a1set_seed0_step5000.pt` 有两份 ——
> `.tmp_v420_data/T08/setsup/checkpoints/` 是 `7bacab5e…`（setsup，A1 集合监督，**本条登记**）；
> `.tmp_v420_data/T08/setloss/checkpoints/` 是 `04d06400…`（setloss 另一实验，**不登记**）。
> 同样地，`.tmp_v420_data/T06/checkpoints/a0_seed0_step5000.pt` 是 `96cb219e…`，与 T04 的 A0
> **逐字节相同**（同一份 A0 的拷贝，不重复登记）；`.tmp_v420_data/T05/checkpoints/a0_seed0_step5000.pt`
> 是 `64509851…`（T05 副本，**不登记**）。A0 的权威登记条目取 **T04** 路径。
>
> 未登记的同族文件（最小集之外）：各 checkpoint 的 `.critic.pt` 兄弟文件、`T06/checkpoints/r2*`、
> `r2kl*`、`r4*`、`r4ctl*`、`VOID-w9e-08_r4*`、`smoke-kl*`、`T04/checkpoints_smoke/`、
> `T05/checkpoints_rehearsal/`、`T08/setsup/checkpoints_smoke/`、
> `T08/setsup/checkpoints/a1set_seed0_step5000.evaluated-243b23.pt`。

### 9.2 2 帧语料 index 与重建汇总（T02c）

| 路径 | 大小 (B) | SHA-256 | 用途一句话 | 新增原因 |
|---|---:|---|---|---|
| `.tmp_v420_data/T02c/rebuild/d1-s0d1-n3300-seed10-shard/2f/meta.json` | 10,760 | `3baf305647b0b8b00c218e3dbf0818668550c6a6bab1cfea09087a4718f7b192` | seed10 shard 的 2 帧语料索引：6,592 槽（6,600 应产、8 个冷启动丢弃）、13,200 帧 HASH 全等、self_check 0 违例 | T02c 是 2 帧语料的唯一来源，此前清单 0 条；只登记索引，不登记 1000 MiB 级 `.npy` |
| `.tmp_v420_data/T02c/rebuild/d1-s0d1-n3300-seed11-shard/2f/meta.json` | 10,760 | `efbd3ea022756f46f4ce4df19676336da5a5965c47b71894e4e526d6ac3f702d` | seed11 shard 的 2 帧语料索引（同上口径） | 同上 |
| `.tmp_v420_data/T02c/rebuild/d1-s0d1-n3300-seed12-shard/2f/meta.json` | 10,762 | `2b18e628ad7103d40e6003771d2f664f0e0d4e5b60ff04e4201d06d49f8f3a18` | seed12 shard 的 2 帧语料索引（同上口径） | 同上 |
| `.tmp_v420_data/T02c/logs/rebuild_seed10.json` | 13,224 | `6699660b88154eae3b8ae5cba16ccac6b1c1dd87a6b0c0ce5daf1412315f088c` | seed10 重建运行汇总：3,300 决策 / 13,200 帧 HASH 全等、legacy play+HUD 3,296/3,296 匹配、self_check 0 违例 | 该语料唯一的机器可读汇总，13 KB，与索引成对登记 |
| `.tmp_v420_data/T02c/logs/rebuild_seed11.json` | 13,224 | `9f65a1c7b384a43295875e35b6889372a7d5a9d058179477ae23fd45c942fe3d` | seed11 重建运行汇总（同上口径） | 同上 |
| `.tmp_v420_data/T02c/logs/rebuild_seed12.json` | 13,224 | `79e377151724a6b25d9d5693d94fa9ecbbafb5bb4a3060238bc422f5fc17a879` | seed12 重建运行汇总（同上口径） | 同上 |

### 9.3 教师重标 index 与汇总（T02d）

| 路径 | 大小 (B) | SHA-256 | 用途一句话 | 新增原因 |
|---|---:|---|---|---|
| `.tmp_v420_data/T02d/reslot/d1-s0d1-n3300-seed10-shard/meta.json` | 14,928 | `8098cfd733b9e47113f1da3a98fda4fa179f4c624805ae27e3a49c7bb1d5b979` | seed10 逐槽教师重标索引：6,600 槽全部查询并 verified（槽覆盖 1.0）、HASH 0 差异 | T02d 是教师重标的唯一产物，此前清单 0 条 |
| `.tmp_v420_data/T02d/reslot/d1-s0d1-n3300-seed11-shard/meta.json` | 14,928 | `4663d02c5554b8aaa7b063dd8dcec095312cf3149b335ffdb1fe578d87baa07d` | seed11 重标索引（同上口径） | 同上 |
| `.tmp_v420_data/T02d/reslot/d1-s0d1-n3300-seed12-shard/meta.json` | 14,930 | `a7a9ab4261177da994ad3e4e6b1964f14084b9eb2c5b1fce88131d2ff4f07af9` | seed12 重标索引（同上口径） | 同上 |
| `.tmp_v420_data/T02d/logs/reslot_2f_join.json` | 16,149 | `74ef2ebbf82137871d3ad2ef77b1d5750a5439dc29e233e044acdd2c831495a1` | T02d 的 2 帧 join 汇总：3 shard / 19,772 槽全部具备真实 8 帧历史与 verified 标签、HASH 0 差异、assertions passed | 该语料的 join 级汇总，清单 0 条 |
| `.tmp_v420_data/T02d/logs/reslot_seed10.json` | 17,165 | `727d21f51645a44bb48330e2bda0b760ba2d49b9a875abfb9aa87ff0b37ea8a7` | seed10 重标运行汇总：6,600 次教师查询、HASH 0 差异、self_check 通过 | 同上（逐 seed 汇总） |
| `.tmp_v420_data/T02d/logs/reslot_seed11.json` | 17,164 | `bfee4979a884cc734982a97a272dfecc43e6b3141004098733b42d3cbfe207ca` | seed11 重标运行汇总（同上口径） | 同上 |
| `.tmp_v420_data/T02d/logs/reslot_seed12.json` | 17,164 | `38a4424cb444b4bfd14004692eaf9d2c3a6abd300a78df8e36a0da62070c89c5` | seed12 重标运行汇总（同上口径） | 同上 |

### 9.4 join 后数据集 index、A1 安全集合表、v4.2.0 状态快照

| 路径 | 大小 (B) | SHA-256 | 用途一句话 | 新增原因 |
|---|---:|---|---|---|
| `.tmp_v420_data/T04/dataset/index.json` | 3,941 | `dbb17ec201a63a0c91112739c80e44de2cbd55d3712a25271ed17770672b09a8` | join 后 `v420-2f-dataset/1` 的唯一索引：3 shard / 19,772 槽（seed10 6,592、seed11 6,592、seed12 6,588）、全部 `chosen_verified`、HASH 链 0 断裂、self_check 0 违例 | 任务书点名的 `…/v420-2f-dataset/1/index.json` **实测不存在**；实际落盘于 `T04/dataset/index.json`（其 `dataset_version` 即 `v420-2f-dataset/1`，`schema` 为 `v420-dataset-2f/1/index`），故按**实际路径**登记 |
| `Document/Update/Alpha/v4.2.0/evidence/T08/a1_safe_set_full.json` | 2,718,805 | `d6da8ea80eab2a2bc967c7802a6aa4d47a590fc500afa4f02b575351e7c482ef` | A1 安全集合完整判定表（逐槽 safe/unsafe 标注），setsup 训练的目标来源 | 本轮点名要求；此前只有 `A1_SAFE_SET_FULL.md` 的文字结论，JSON 未入清单 |
| `Document/Update/Alpha/v4.2.0/evidence/T08/a1_safe_set_pilot.json` | 918,879 | `675fbffdd6949dc72a69ba4587c25b33db84ffc4fbcd771b662ca6c71b2babff` | A1 安全集合 pilot 版判定表（full 的前置试跑） | 同上（pilot 版） |
| `Document/Update/Alpha/v4.2.0/state.json` | 57,539 | `99e22af942b2c3eb89eed274fbfd38981a928954a6d9d17b8d359849989a6039` | v4.2.0 执行状态快照（WP/门/任务；G420-0 `passed`，G420-1…G420-4 `failed_valid`，G420-R `in_progress`，G420-5 `not_started`） | 清单已有 v4.1.0 的同名三件，v4.2.0 版本缺失；§8.2 要求状态文件入清单 |
| `Document/Update/Alpha/v4.2.0/journal.jsonl` | 189,337 | `5ed3693f38e4ac4a7f823d6ddd50c7eabbffa15c20acd13566798b3bafaf64e9` | v4.2.0 事件 journal 快照（`v420-journal/1`；含 asset-manifest 缺口被记为 `rule_gap`、以及本轮 repair 的 `dispatch` 记录） | 同上 |
| `Document/Update/Alpha/v4.2.0/budget.jsonl` | 9,627 | `01d8bd048b982bf19a2123c21265dac751cb79cf6a7e3e0550543784ddea4b16` | v4.2.0 预算记录快照 | 同上 |

### 9.5 明确**不**登记（有意排除，避免把 16 GB 塞进清单）

- `.tmp_v420_data/tests/**`（pytest basetemp，实测 3.9 GB，可随时重建）；
- 任何 `engine-data-*`（引擎 data 私有副本，可由 `/root/v400-data` 重建）；
- 任何 `*.npz` 观测样本；
- 2 帧语料的 1000 MiB 级 `.npy` 原始观测（实测 `T02c` 目录 1001 MB；由 §9.2 的 `meta.json`
  索引 + 各 `shard.json` 内部 SHA 校验，不做全目录哈希）；
- `.tmp_v420_data/T02c/smoke/**`、`.tmp_v420_data/T02d/smoke/**`、`T04/checkpoints_smoke/**`、
  `T05/checkpoints_rehearsal/**`、`T08/setsup/checkpoints_smoke/**`（smoke / rehearsal 试跑产物）；
- `.tmp_*` 脚本目录与 `*.sh` / `*.py` 探针（不是资产）；
- `.tmp_v420_data/T08/expand/**`（**扩容正在进行中**：`run_all` pid 397 与 3 个 `th08-modern`
  仍在写，快照无意义；本轮亦被禁止写入该目录）。

### 9.6 校验

```bash
cd /mnt/d/WorkShop/Touhou_RL          # 或 D:\WorkShop\Touhou_RL
sha256sum -c ASSET_SYNC.sha256        # 46 条 = 既有 20 条 + v4.2.0 26 条
```

- 登记后实测：**46 条 OK / 0 FAILED，退出码 0**。
- 双向一致性：§9.1–§9.4 表格里每条「路径 + 64 位 SHA-256」都能在 `ASSET_SYNC.sha256` 找到
  同名同哈希行；`.sha256` 末尾 26 条每条也都能在本节表格找到 → **不一致 0 项**。
- 既有 20 行未变的判据：`head -c 2375 ASSET_SYNC.sha256 | sha256sum` 仍等于
  `73ccf480d5dbd5ddc5d81d4d87b2ef15d204b76d111631ad99b2f19540637373`。
- 一致性脚本：`.tmp_v420_manifest/14_bidir_check.sh`（双向 + 逐哈希比对 + 既有 20 行字节证明）。
- **注意**：`state.json` / `journal.jsonl` / `budget.jsonl` 是活文件。v4.2.0 仍在推进
  （`G420-R in_progress`，扩容进程在跑），三份文件一旦被追加，上表 3 行的 SHA 会失效 ——
  届时**只需重算这 3 行**，不要把失败当成资产损坏。

### 9.7 附录：既有 20 条中「散文提到、无表格行」的 12 条索引

> **这不是本轮新增的资产，也不是 v4.2.0 的资产。** 这 12 条本来就在 `ASSET_SYNC.sha256`
> 的前 20 行里（v4.1.0 文档），只是既有第 5 节用散文提到它们（或只提文件名），没有
> 「路径 + 完整 SHA-256」的表格行，因此全量双向校验会把这 12 条算成「sha256 有、md 表格无」。
>
> 本表**只做索引**：哈希逐字取自 `ASSET_SYNC.sha256` 既有行，未重算、未改动任何既有行；
> 既有章节一字未动（`head -c 9946 ASSET_SYNC.md` 的 sha256 仍为
> `f3696ddcf881594eb…`，即在追加 §9 之前的原文件）。补上本表后，
> `md ↔ sha256` 全量双向不一致 = **0**。

| 路径 | 大小 (B) | SHA-256 | 在既有正文中的定位 |
|---|---:|---|---|
| `Document/Update/Alpha/v4.1.0/README.md` | 84790 | `8fc51064e5e54e7cdf6e0b924dcaefd0b3020d148ceeaf5bc0deb9294ae4876c` | §8 相关入口「规格」 |
| `Document/Update/Alpha/v4.1.0/state.json` | 2465 | `13e0e31476f2f44bd0c5eb54c93574583e7b59e305edbb2da2eeff5d0e19962c` | §5 / §8「状态（paused）」 |
| `Document/Update/Alpha/v4.1.0/journal.jsonl` | 67008 | `5ad6f2fdb9e8d6dc5a2e6422136422a41096e01d110814be7a74861a829d8f1d` | §5「journal.jsonl / budget.jsonl」 |
| `Document/Update/Alpha/v4.1.0/budget.jsonl` | 6630 | `2a766e230b6650328a4353d6512969f4fa8014ef6c17de6b96a5633ba54c11eb` | §5「journal.jsonl / budget.jsonl」 |
| `Document/Update/Alpha/v4.1.0/evidence/WP01/g0.1-108.json` | 74360 | `33274e174eda6cc4865b64ff359fd9587cc94f15bceca1a89caa4b1a6cc1c31f` | §5「g0.1-108.json（108/108）」 |
| `Document/Update/Alpha/v4.1.0/evidence/WP02/G4101_100_TEACHER.md` | 2279 | `aa3f5abcbd0f75a01b1b7e4413bb1b1933e84f097b64eec258dbd0e6d5956c49` | §5「G4101_100_TEACHER.md（100/100）」 |
| `Document/Update/Alpha/v4.1.0/evidence/WP05/round1.json` | 70416 | `8cc68b54778c1cedad30acaf8216b15eedf1c49846b325248047572c3b2a4f09` | §5 WP05 行（DAgger 数据） |
| `Document/Update/Alpha/v4.1.0/evidence/WP05/round2.json` | 72152 | `81bf1a5a34befa5a6d74ca022e1b69af26c444b9d6b9b03e074d8f091d42fb6f` | §5 WP05 行（DAgger 数据） |
| `Document/Update/Alpha/v4.1.0/evidence/WP05/train_beta05_20k.json` | 91445 | `f5d9590534cc28aaf942137b90bd91ec08c13108acd6190a8093c36a5319c8bf` | §5 WP05 行（训练） |
| `Document/Update/Alpha/v4.1.0/evidence/WP05/train_tiebreak.json` | 85818 | `d2698c359e89cad7c4a35575fd41b4e462244a844e4e321ec36f272d6ca5f62e` | §5 WP05 行（tie-break） |
| `Document/Update/Alpha/v4.1.0/evidence/WP05/round3.json` | 93330 | `f1c6b8dbef230699a23e9db63132d1ac52431be3c16ea72fad8f3c9e91cf303c` | §5「round3.json / ROUND3.md」 |
| `Document/Update/Alpha/v4.1.0/evidence/WP05/ROUND3.md` | 24701 | `75bc330b3bdbe8f2fdd97f48c18b2ee2c16ff7b888c60cd86cf4aed0eba9d21a` | §5「round3.json / ROUND3.md」 |

> 至此全量：`ASSET_SYNC.sha256` 46 条 = `ASSET_SYNC.md` 表格 46 行（§2 引擎 2 + §4 checkpoint 6
> + §9.1–§9.4 v4.2.0 26 + §9.7 既有 12），双向 0 不一致。

### 9.8 活文件（state / journal / budget）的重钉记录

> `Document/Update/Alpha/v4.2.0/state.json` 与 `journal.jsonl` 会被 v4.2.0 的推进**持续追加**
> （扩容进程 `run_all` pid 397 仍在跑），所以它们的 SHA 不是「资产属性」，而是「某个时刻的快照」。
> 实测证据：首次写入清单是 `04:17:26`，**28 秒后**（`04:17:54`，非本轮进程所写）这两份文件就被
> 追加过一次（`state.json` 56,064 → 56,149 B；`journal.jsonl` 168,461 → 169,748 B）；`04:18:27`
> 重钉时读到的就是 `04:17:54` 这一版。已在 `04:18:27` 重钉这 2 行，**重钉后立刻复验**：
> `sha256sum -c` = **46/46 OK，0 FAILED，exit 0**。

| 文件 | 首钉（04:17:26，已过期） | 重钉（04:18:27，当前） | 当前大小 (B) |
|---|---|---|---:|
| `state.json` | `ab36b344d20b…` | `87035123647d…` | 56,149 |
| `journal.jsonl` | `9890f86d7f9e…` | `6726def33c5c…` | 169,748 |
| `budget.jsonl` | `01d8bd048b98…` | `01d8bd048b98…`（未变，无需重钉） | 9,627 |

> 完整 64 位哈希以 `ASSET_SYNC.sha256` 第 44–46 行为唯一权威，本表只是审计线索，不参与校验。

**扩容期间的读法（重要）**：

1. 跑 `sha256sum -c ASSET_SYNC.sha256`；若 **只有**上面这 3 行 FAILED，那是**预期**的活文件推进，
   不是资产损坏 —— 跑 `.tmp_v420_manifest/17_repin_live.sh` 重钉（重算 → 只改这 3 行 → 立即复验，
   期间又被追加就自动重试，最多 6 次）。
2. 其余 **43 行是静态资产**（7 checkpoint + 20 个语料 index/汇总 + 2 个 A1 表 + 12 条 v4.1.0 文档
   + 2 个引擎二进制）。这 43 行里任何一条 FAILED 都是**真问题**，必须先查清，不要重钉了事。
3. 扩容（`run_all` / `th08-modern`）结束时会再写一次 `state.json` / `journal.jsonl`；
   届时重钉一次即可（`budget.jsonl` 通常不变）。

**重钉脚本**：`.tmp_v420_manifest/17_repin_live.sh`
**复验脚本**：`.tmp_v420_manifest/15_verify_all.sh`（sha256sum -c + 双向一致性 + 既有 20 行字节证明）

---

## 10. v4.3.2 fork 重判的新增资产（2026-11-21）

> **为什么有这一节**：v4.3.2 跑了一轮 16,944 条 episode 的实验（fork 池），产出了 3 个训练 checkpoint、
> 一个新的（**未使用**的）引擎构建目录，以及 5 份分析证据；按 AGENTS.md §8.2 必须成对登记。
>
> **引擎二进制未变**：`rl_v3` 仍为 `691bfd77…a7a568ef6`、`rl_laser` 仍为 `1462829b…b555340d2f`，
> 两轮实验全部用原 `rl_v3`；`third_party/th08-rl/src/**` **零改动**（`git status --porcelain` 为空）。

| 名称 | 路径 | 大小 (B) | SHA-256 | 用途一句话 | 新增原因 |
| --- | --- | ---: | --- | --- | --- |
| armA seed0 | `.tmp_v432_arm/armA_seed0.pt` | 34,619 | `dc301338673506fd1b729060263c18cf2972bf91c78f18fee9903afbaf85856d` | v4.3.2 臂 A 策略种子 0 的 4,000-episode 训练产物（塌缩到 cand 4） | 新训练产物，不可由 Git 重建 |
| armA seed1 | `.tmp_v432_arm/armA_seed1.pt` | 34,619 | `93064cf36e991c94238b24d16ecb2b55f31d3bb35b421b5c940b3a56b16306fe` | 策略种子 1（塌缩到 cand 12，= train 最优候选） | 同上 |
| armA seed2 | `.tmp_v432_arm/armA_seed2.pt` | 34,619 | `4abe8348f9cced5ceab0ae44b519219b6f50de9e1761d5200e54e4947c0f5aa9` | 策略种子 2（塌缩到 cand 3） | 同上 |
| fork 构建 | `/root/Touhou_RL/third_party/th08-rl/build/rl_fork/th08-modern` | 4,836,016 | `a866e63db4610c04154ee2d54e5b5f3e363ab9edeb775d8d8451742d362745d3` | 用来证明「同选项另建目录可复现构建」的**未打补丁**基线构建；**两轮实验都没有用它** | §8.2 新增引擎构建必须登记；与 `rl_v3` 字节不同（构建路径进 debug info，大小差 8 B） |
| 保真证据 | `Document/Update/Alpha/v4.3.2/evidence/FORK/FIDELITY.json` | 52,674 | `b591c4d6fcf0170369f6770b79ecccb3ac5acc0229d25d6777a02199efa2eb59` | 五条保真门的独立重算（2166/0、18/18、成本） | v4.3.2 的核心证据，需可校验 |
| 统计证据 | `Document/Update/Alpha/v4.3.2/evidence/FORK/analysis.json` | 71,709 | `8d3abf9404e3677a6455e73bf7a5ab8c127ec2c6dad3b47594eca688e1e2815e` | H1/H2/S2、伪重复诊断、塌缩指纹、R51 复现 | 同上 |
| 常量网格 | `Document/Update/Alpha/v4.3.2/evidence/FORK/constants.json` | 60,979 | `3313129d942ca199c6d5ba5abde76210230ab0f83c061d2a252346feac1a2890` | 18 候选 × 8 窗口的确定性存活（可部署基线 172.25 的来源） | 同上 |
| 执行记录 | `Document/Update/Alpha/v4.3.2/evidence/FORK/RUN.md` | 36,189 | `0e6ce5c0c6b094be6390409ef2c45530009dd31a63eabe5356861c3d9c29154e` | 26 条命令 + 退出码 + 墙钟 + 偏差清单 | 同上 |
| 分析报告 | `Document/Update/Alpha/v4.3.2/evidence/FORK/ANALYSIS.md` | 40,655 | `869a7ead1f7e7bf0522040e10dd05d644f070b6e0c1df432a1c0281c40c040dd` | 人读版分析与「能/不能声称」清单 | 同上 |

> **未登记（有意排除）**：`train_seed*.jsonl` / `eval_armA_seed*.jsonl` / `eval_cfair.jsonl`（合计 ~17 MB 原始 episode 记录，
> 已在 `Document/Update/Alpha/v4.3.2/evidence/FORK/` 内受 Git 跟踪）；`.tmp_v432_fork/`、`.tmp_v432_park/`、
> `.tmp_v432_arm/` 的其余探针与日志（可由脚本重跑，或已由上面的证据文件固化）。
>
> **校验**：`cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256`。新增 9 行后，
> 预期仍只有 `Document/Update/Alpha/v4.2.0/journal.jsonl` 一行 FAILED（§9.8 的活文件推进，非资产损坏）。

---

## 11. v4.3.3 学习器修复轮的新增资产（2026-11-21）

> 单因子修复（returns-to-go + 逐 episode 标准化 + 熵正则 β=0.01），其余全部冻结。
> **引擎未变**（`rl_v3` `691bfd77…`）；`tools/rl/v432_fork_arm.py` 只**新增** `--learner {frozen,repair}`
> （默认 `frozen`；等价性三条独立证据：A/B 60/60 条逐位相同、`constants.json` 144/144 格逐位相同、离版权重 5/5 逐位相同）。

| 名称 | 路径 | 大小 (B) | SHA-256 | 用途一句话 |
| --- | --- | ---: | --- | --- |
| repair seed0 | `.tmp_v433_repair/armA_repair_seed0.pt` | 34,689 | `deaacfc690e972f6e34b496b4beac09cda8eaeb0625ac56a550b8005e63d0817` | 4,000-episode repair 训练产物（策略种子 0） |
| repair seed1 | `.tmp_v433_repair/armA_repair_seed1.pt` | 34,689 | `3d8551b49903d697f615f8fca638d3793b51a3cab51d35480fe9751693e7d856` | 同上（种子 1） |
| repair seed2 | `.tmp_v433_repair/armA_repair_seed2.pt` | 34,689 | `7677b47b987906f4a8a55bd6ac170ad32e10a445d239b6a32f221ea06d8e16e6` | 同上（种子 2） |
| 保真门 | `Document/Update/Alpha/v4.3.3/evidence/REPAIR/fidelity_v433.json` | 26,411 | `e9653dbaad18f23da1d187aad511b22bde04289971902f2d3df77a4d8daf7690` | D3 五条门 |
| 诊断 | `Document/Update/Alpha/v4.3.3/evidence/REPAIR/diagnostics.json` | 70,462 | `2e1bff74f434e4ce4584204823151c0493ceea6d1dd2cbac63192cfcbd3c6f97` | D1/D2/D4 原始计数 |
| 执行记录 | `Document/Update/Alpha/v4.3.3/evidence/REPAIR/RUN.md` | 25,284 | `1531c4dd0e24a64528a32ac200f6c62ecb114f9e5411bd66012886ac1cbb28c0` | 命令/退出码/墙钟/偏差 |
| 统计证据 | `Document/Update/Alpha/v4.3.3/evidence/REPAIR/analysis.json` | 78,294 | `52074a68c3b59656b10954a3483d94384c2487252ed872e5f8bf9793b0181ff6` | D1–D4 / P1–P3 / 存活率 / 与 v4.3.2 并排 |
| 分析报告 | `Document/Update/Alpha/v4.3.3/evidence/REPAIR/ANALYSIS.md` | 25,090 | `3d8303ef098d10f5a12a42f34c29a6e62b5fd993232ae30aeb51b0d20c1c3b96` | 人读版 |

> **未登记（有意排除）**：`train_seed*.jsonl` / `eval_armA_seed*.jsonl`（约 14 MB 原始 episode 记录，已在
> `evidence/REPAIR/` 内受 Git 跟踪）、`v432_fork_arm.{before,after}.py` 与 `.diff`（审计副本）、生成脚本与日志。
> 证据目录自带的 `SHA256SUMS` 已重算并覆盖全部 **31** 个文件（`sha256sum -c` = 31/31 OK）。

---

## 12. v4.4.0–v5.1.0 新资产（成对登记，2026-11-21）

> **为什么有这一节**：v4.4.0 → v5.1.0 共 8 个版本轮次产出了 20 个训练 checkpoint（全部在 gitignored 的 `.tmp_v4*` / `.tmp_v5*` 下，**不可由 Git 重建**）与 14 份机器可读证据；按 AGENTS.md §8.2「新增/移动/删除任何资产，必须同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`」，本节与 `ASSET_SYNC.sha256` **末尾追加的 34 行**成对登记。
>
> **登记口径（最小集）**：只收 20 个「训练产物 / 部署候选」checkpoint 与每版本点名要求的机器可读证据；**不**收 `train_seed*.jsonl` / `eval_*.jsonl` / `labels_round*.jsonl` / `*_constants_*.jsonl`（每轮 1–17 MB 的原始 episode 记录，已在 `Document/Update/Alpha/<ver>/evidence/` 内受 Git 跟踪）、探针脚本、`.diff` 审计副本、`smoke*` / `e2e_init` / `negctl` / `ab` 试跑 ckpt。
>
> **快照时刻**：`2026-11-21T21:15` 前后（Asia/Shanghai）。本节点名的每一条都来自当次真实 `sha256sum` 输出，未手抄、未推算。
> **自校验**：20 个 checkpoint 的哈希与该版本自己的 `train_index.json` / `manifest.json` 内联值 **20/20 逐条相符**；14 份证据里
> **4 份**（`TERM/RUN.md`、`BCRL/RUN.md`、`LADDER/RUN.md`、`LADDER/constants_1200.json`）能在同版本 `manifest.json` 找到同值，
> 其余 10 份是该轮分析侧后加的产物，manifest 里本就没有条目，只能以本节实测值为准。
>
> **`.tmp_*` 行沿用既有写法**（相对路径 `.tmp_...`），不引入 `/mnt/d/...` 形式——`ASSET_SYNC.sha256` 前 63 行里只有 3 个引擎二进制用 WSL 绝对路径 `/root/...`，`.tmp_*` 一律是相对路径。

| 名称 | 路径 | 大小 (B) | SHA-256 | 用途一句话 |
| --- | --- | ---: | --- | --- |
| TERM seed0 | `.tmp_v440_term/term_seed0.pt` | 34,619 | `45915385dc6ea6bc1db3d66f8ab64817f2c34f7a016e09cce72b78a8a84ce195` | v4.4.0 终点化（terminal）奖励 RL 臂策略种子 0 的训练产物（4,000 episode × 300 决策、seeds 13–16）；P1 161/3,600 = 4.4722% 的载体 |
| TERM seed1 | `.tmp_v440_term/term_seed1.pt` | 34,619 | `5ddace76a5d3791c366308424694905b38cdeb5c3bd0b5a98700ba161436a191` | 同上（策略种子 1） |
| TERM seed2 | `.tmp_v440_term/term_seed2.pt` | 34,619 | `dd2b54c5d751062d0917e0fcb9c3d16c935ae26caac74843ee64494d6eabf8c4` | 同上（策略种子 2） |
| CF seed0 r2 | `.tmp_v450_cf/ckpt/seed0/round2.pt` | 34,387 | `4abee069d087872e0a2d726cc975850d760e90a705ddcbee60ea91268e5e61dc` | v4.5.0 反事实标签 3 轮监督 DAgger 的末轮 ckpt（seed0：1,302 状态 / 23,436 标签，final train BCE 0.3553） |
| CF seed1 r2 | `.tmp_v450_cf/ckpt/seed1/round2.pt` | 34,387 | `a93746bb5d123165e77e0271edfc1efcdb21931e9e2d706e5a22d613c5476ca4` | 同上（seed1：2,092 状态 / 37,656 标签） |
| CF seed2 r2 | `.tmp_v450_cf/ckpt/seed2/round2.pt` | 34,387 | `c69960fcbbd888e9b252d3fa685c22a6730f8dd637dcfc9edab45491d218fa01` | 同上（seed2：1,772 状态 / 31,896 标签） |
| MA seed0 r1 | `.tmp_v460_ma/ckpt/seed0/round1.pt` | 34,387 | `460fdd0c4848239d28e0236ed37135276264f52be8d4f1bcf5a47137f1ae1191` | v4.6.0 多锚点反事实监督 DAgger 第 2 轮的 ckpt（seed0：round0+1 共 574 状态 / 10,332 标签） |
| MA seed1 r1 | `.tmp_v460_ma/ckpt/seed1/round1.pt` | 34,387 | `e2558816f4ee4e782d010737857591322ff4d634fa20a711a1906ab7057455a2` | 同上（策略种子 1） |
| MA seed2 r1 | `.tmp_v460_ma/ckpt/seed2/round1.pt` | 34,387 | `7bd6f834b22068e7e6b3feffa458706a66640aec3f3f1dc27fb44664761225e8` | 同上（策略种子 2） |
| MA seed3 r1 | `.tmp_v460_ma/ckpt/seed3/round1.pt` | 34,387 | `d34e6d92b71fefcbd00ff601a62b63ee82cf89e763593b7807a7530c144ec841` | 同上（策略种子 3） |
| MA seed4 r1 | `.tmp_v460_ma/ckpt/seed4/round1.pt` | 34,387 | `3970c00979cacc8a1f794a8380fa87a22f59a0087dd2f2acc08867b9a3cf0953` | 同上（策略种子 4） |
| BCRL seed0 | `.tmp_v470_bcrl/bcrl_seed0.pt` | 34,619 | `3313b912fc16374ce1022445d9fb43293250e1936d60358c4169205de2e01cbe` | v4.7.0「只改初始化」臂的产物：以 v4.5.0 seed0/round2.pt 载入（max|Δparam|=0 已断言）后 4,000 episode 终点化 RL（seed0） |
| BCRL seed1 | `.tmp_v470_bcrl/bcrl_seed1.pt` | 34,619 | `7c0c522ceb2826c48b934dc00e4015b61df8b6330150f0977e092dccbe6bae11` | 同上（seed1 ← v4.5.0 seed1/round2.pt） |
| BCRL seed2 | `.tmp_v470_bcrl/bcrl_seed2.pt` | 34,619 | `1f605e0b1ad68b3f87bf66d0440d50d5106e875ec900793781f0d5087e4a9087` | 同上（seed2 ← v4.5.0 seed2/round2.pt） |
| HZ seed0 r2 | `.tmp_v480_hz/ckpt/seed0/round2.pt` | 34,387 | `15135b3c92b21143b179ed75fe6ff35cdba87e4db3a07089778ff7f3a7eee64d` | v4.8.0 标签视界 120→240 帧 + 18 路并行标注 + DAgger×3 的末轮 ckpt（seed0：2,103 状态 / 37,854 标签） |
| HZ seed1 r2 | `.tmp_v480_hz/ckpt/seed1/round2.pt` | 34,387 | `884cb32f3f7865e60de565cfb99982b63b6373b6011d1c4e5adf1c344201fff8` | 同上（seed1：2,010 状态 / 36,180 标签） |
| HZ seed2 r2 | `.tmp_v480_hz/ckpt/seed2/round2.pt` | 34,387 | `30c838c27e35c50ff5b33a17057a0d0f102e1d9174a6379e74181a41e11ccdf3` | 同上（seed2：1,685 状态 / 30,330 标签） |
| REG seed0 | `.tmp_v500_reg/ckpt/seed0/reg.pt` | 34,357 | `948a7da6127f7216c42524922fcd3cb108dfe9a97156fb4f2f5ba68802bdd741` | v5.0.0 把训练目标换成存活帧数回归（y = alive_frames/120，SmoothL1 β=0.1）的 seed0 产物（1,302 状态 / 23,436 标签） |
| REG seed1 | `.tmp_v500_reg/ckpt/seed1/reg.pt` | 34,357 | `c9d57625ec9f8bc195ca67b4b6d3f2d9ecaeab2cb9e2d9453814e64f9696bae6` | 同上（seed1：2,092 状态 / 37,656 标签） |
| REG seed2 | `.tmp_v500_reg/ckpt/seed2/reg.pt` | 34,357 | `c9e451198bbfa472e22603b8da38456f3cae928dbc5b91e36ae63b6b060a4471` | 同上（seed2：1,772 状态 / 31,896 标签） |
| TERM 统计 | `Document/Update/Alpha/v4.4.0/evidence/TERM/analysis.json` | 37,365 | `64f9e17797c0c445f6a89dd4e58edc293f0eb4871f24f995540f886c4f3aba2c` | v4.4.0 的离线统计重算（引擎 0 次）：P1 成立 3/3、P2 单侧 Fisher p = 3.908e-09、S-2x 到期未跑 |
| TERM 执行记录 | `Document/Update/Alpha/v4.4.0/evidence/TERM/RUN.md` | 43,127 | `070c6621b6951ec3482e08a5f17e2bea8d9dcbeb6961b932ff74ec7427b0dd17` | 命令/退出码/墙钟/门计数（全部门通过；含改动前后运行器 A/B 一致性 60/60） |
| CF 统计 | `Document/Update/Alpha/v4.5.0/evidence/CF/analysis.json` | 44,791 | `aa608cf6507fd9456c7363ba59e18c84f4c0e58619f95491a7faf63879d49d94` | v4.5.0 离线统计重算：P1 不成立（(a) 2/4 dev 窗口）、P2 p = 4.671e-46、S-no-signal 触发 |
| CF 跨轮 | `Document/Update/Alpha/v4.5.0/evidence/CF/CROSS_ROUND.md` | 19,662 | `a1b22380d0065522d8377dca142ad9d8592571d3ea2f831344eed0827c58133b` | v4.4.0 × v4.5.0 的唯一跨轮对照（论证均值与成功率必须同时报、且各注单位与 n） |
| MA 统计 | `Document/Update/Alpha/v4.6.0/evidence/MA/analysis.json` | 95,288 | `3db054cd91817216e28720bd2b4fc45db251fd8f5f7f4e639171a6704907a6c1` | v4.6.0 离线统计重算：P1 不成立（1/4 窗口、1/5 种子）、S-no-cover 触发、P2 率比 4.414× |
| MA 执行记录 | `Document/Update/Alpha/v4.6.0/evidence/MA/RUN.md` | 35,403 | `0e34dc38deb12b619623b662fd52a9821f2cafad0f69b2e286f7ea5318940a5e` | 原始数据/门/计数/路径/哈希；明确「本文件不跑统计，判定属独立分析侧」 |
| BCRL 统计 | `Document/Update/Alpha/v4.7.0/evidence/BCRL/analysis.json` | 67,212 | `dc817c87972cc6959b0b7294dbdc99a037f71952bfdf1f1d4ecfc892b692ab1c` | v4.7.0 离线统计重算：P1–P4 全不成立、S-degrade 触发（对 v4.4.0 基线退化 161×） |
| BCRL 执行记录 | `Document/Update/Alpha/v4.7.0/evidence/BCRL/RUN.md` | 35,308 | `66e35a36736d0e28f5bed4a7fe908553d26c993019ac4e4391f3e20a03b39388` | init 载入断言 3/3、3 条负对照 fail-closed、训练 890/12,000 = 7.417% 全 valid |
| HZ 执行记录 | `Document/Update/Alpha/v4.8.0/evidence/HZ/RUN.md` | 37,335 | `e2d32c4bc39ca93ef0eb6a1e6223f1b9e7494494269719328ef510bd40e43ef6` | H=240 + 并行 18 子进程 + DAgger×3 的原始计数与门（D3 5/5、selfcheck 18/18 两次、serial vs parallel 22/22） |
| ENS 统计 | `Document/Update/Alpha/v4.9.0/evidence/ENS/analysis.json` | 34,244 | `61ef52c781e18aa06d30febd6710b471f1b3dc7b52458fcc6da1386091c58a63` | v4.9.0 离线统计重算：P1 141/1,200 = 11.75% ≤ 13.4444% 不成立、P2 0/4 窗口、S-worse 触发 |
| ENS 跨轮 | `Document/Update/Alpha/v4.9.0/evidence/ENS/CROSS_ROUND_4.md` | 9,368 | `407d2f17d57b9576bd52e3f226c18f1f377497ee644d60dff677c649f517794c` | v4.3.2 → v4.9.0 并排对照（本批唯一跨轮文件；无任何一轮 stochastic 超过 v4.5.0 的 13.4444%） |
| REG 执行记录 | `Document/Update/Alpha/v5.0.0/evidence/REG/RUN.md` | 24,836 | `55bb51604de8fbe1c6999214c83e7a358736618db2867e2e36a85f9805c883f8` | v5.0.0 回归目标的执行记录（零新标签 / 零新采集；4 个 dev 窗口未复现 v4.5.0 水平） |
| LADDER 执行记录 | `Document/Update/Alpha/v5.1.0/evidence/LADDER/RUN.md` | 24,569 | `03d6f91264af0c413ed20c0992c0b4be74189d5f3a5ab204d98701f4b258d21c` | v5.1.0 单因子（段长 600 决策 / 1,200 帧）执行记录：D3 保真门 5/5 PASS、S-floor 逐字条件 |
| LADDER 常量网格 | `Document/Update/Alpha/v5.1.0/evidence/LADDER/constants_1200.json` | 38,226 | `91ea70303afb712d7ab2c06231068db4d51c4cc86705b25e6e2d88477eb186ba` | 1,200 帧段上 18 候选 × 8 窗口 = 144/144 格的确定性存活（部署基线 cand 12：train 186.25 / dev 172.25） |

### 12.1 跳过（实测缺失，未登记）

| 路径 | 实测 | 处理 |
| --- | --- | --- |
| `Document/Update/Alpha/v4.8.0/evidence/HZ/analysis.json` | **不存在**（该目录有 26 个文件，含 `RUN.md`、`manifest.json`、`fidelity_d3.json`、`train_index.json`，但没有 `analysis.json`） | **跳过不登记**，非 0 字节文件；该轮 `RUN.md` 原文即写明 `analysis.json` / `ANALYSIS.md` *deliberately absent*，统计判定归独立分析侧 |
| `Document/Update/Alpha/v5.0.0/evidence/REG/analysis.json` | **不存在**（该目录有 9 个文件：`RUN.md`、`manifest.json`、`fidelity_d3.json`、`features_recheck.json`、3 个 `train_seed*.json`、2 个 `eval_*.jsonl`） | **跳过不登记**，本轮没有产出独立分析文件 |

> 两处都是**目录里没有该文件**，不是 0 字节。另：`v4.9.0/evidence/ENS/analysis.json` 的 `v48_status` 还留着当时的口径 `NOT READY: no Document/Update/Alpha/v4.8.0/evidence/HZ/ directory`（HZ 目录后来才落盘，那一行未回改），引用时勿据此判断 v4.8.0 缺证据。

### 12.2 不重复登记（实测已核）

- **已在册路径 0 条**：追加前 `grep -nE 'v440|v450|v460|v470|v480|v490|v500|v510|v4\.4\.0|v4\.5\.0|v4\.6\.0|v4\.7\.0|v4\.8\.0|v4\.9\.0|v5\.0\.0|v5\.1\.0' ASSET_SYNC.sha256` 无任何匹配；追加的 34 条与既有 63 条无交集。
- **重复行 0 条**：追加前 `cut -c67- ASSET_SYNC.sha256 | sort | uniq -d` 为空；追加的 34 条自身 34 路径 / 34 哈希全唯一。
- **行数账**：63 + 34 = **97** 行（`wc -l ASSET_SYNC.sha256` = 97）。
- **Git 跟踪状态（实测，与任务前提不符，重要）**：这 8 个版本目录目前**全部未入 Git** ——
  `git status --porcelain -- Document/Update/Alpha/<ver>` 对 `v4.4.0` … `v5.1.0` 一律返回 `??`（整目录 untracked），
  `git check-ignore -v` 返回 `NOT_IGNORED`，`git ls-files` 计数 **0**。即这些证据**现在并不能由 Git 重建**，
  本节 + `ASSET_SYNC.sha256` 是它们目前**唯一**的字节身份记录；本轮按约束**未**执行 `git add`。

### 12.3 校验结论

`cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256`：**97 行 = 95 OK / 2 FAILED**（复验时刻 `2026-11-21T21:22+0800`）。
既有 63 行贡献 **62 OK / 1 FAILED**，新登记 34 行贡献 **33 OK / 1 FAILED**。

| # | FAILED 路径 | 判定 | 本轮处置 |
| --- | --- | --- | --- |
| 1 | `Document/Update/Alpha/v4.2.0/journal.jsonl` | §9.8 记载的活文件漂移，**已知且允许** | 未改该行 |
| 2 | `Document/Update/Alpha/v5.0.0/evidence/REG/RUN.md` | **登记后被并发执行方持续重发布**，不是资产损坏、也不是本轮算错：实测字节数 21,090（21:15 登记时；当时 `sha256sum -c` 为 96 OK / 1 FAILED，即该行通过）→ 24,329（21:21:02）→ 24,518（21:21:21）→ 24,836（21:22:04）；期间该轮 `manifest.json` 的 `RUN.md.sha256` 也跟着换 | 按任务约束**未改该行**；待 v5.0.0 发布进程停下后重钉这一行（现在重钉会立刻再次过期） |

> **一句话结论**：新登记的 34 行里 **33 行 OK**；唯一非预期的 FAILED 是 `v5.0.0/evidence/REG/RUN.md`，
> 成因是**并发写入**而非哈希错误（登记当时 34/34 OK，实测记录在上表）。该文件在复验期间仍在增长，
> 所以本轮**不做重钉** —— 重钉必须等写入方停下；停稳后只改这 1 行即可把结果拉回 **96 OK / 1 FAILED**
>
> **补记（2026-11-21 22:2x）**：写入方已停下，该行已于 21:24:51 重钉为 24,836 B / `55bb5160…`；§12 表格的同一行也已同步更正。当前整体为 **111 OK / 1 FAILED**（唯一 FAILED 是 §9.8 允许的活文件 \`v4.2.0/journal.jsonl），见 §13 末尾的复验。
> （只剩 §9.8 允许的 `v4.2.0/journal.jsonl`）。

---

## 13. v5.2.0–v5.4.0 新资产（2026-11-21）

> **为什么有这一节**：v5.2.0（XLADDER-001，零训练外推到 1,200 帧预算）与 v5.3.0（T20-001，600 决策预算训练）共新增 **3 个训练 checkpoint**（在 gitignored 的 `.tmp_v530_t20/` 下，**不可由 Git 重建**）与 **12 份已发布证据**；按 AGENTS.md §8.2「新增/移动/删除任何资产，必须同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`」，本节与 `ASSET_SYNC.sha256` **末尾追加的 15 行**成对登记。其中 **v5.4.0 的 `evidence/XL30/` 是在本轮运行期间（22:13–22:16）发布完成的**，其点名的 3 份证据已补登；`.tmp_v540_x30/` 内 `*.pt` 复查仍为 **0 个**（v5.4.0 零训练，见 §13.1）。
>
> **登记口径（最小集）**：只收 3 个 `t20_seed*.pt` 训练产物与点名要求的 **12 份机器可读证据**（v5.2.0 5 份 + v5.3.0 4 份 + v5.4.0 3 份）；**不**收 `train_seed*.jsonl` / `eval_*.jsonl`（0.86–3.48 MB 的原始 episode 记录）、`*.py` 生成/探针脚本、`exec.sh`、`verify.sh`、`commands.tsv`、`preflight.json`、`fidelity_*.json`、`manifest.json`，也不收 §13.2 列出的「存在但不在最小集」的那些文件（8 个单文件 + v5.4.0 的其余 20 个）。
>
> **快照时刻**：v5.2.0 / v5.3.0 的 12 条取于 `2026-11-21T22:12+0800`，v5.4.0 的 3 条取于 `2026-11-21T22:18+0800`。下表每条都来自当次真实 `sha256sum` 输出，未手抄、未推算；追加后 `sha256sum -c` 对这 15 行**全部 OK**。v5.4.0 的 `RUN.md` / `diagnostics.json` 在发布过程中被重写过两次（22:13:40 → 22:15:20 → 22:16:10，`RUN.md` 37,930 → 43,173 → 43,887 B），本节钉的是 **22:16:10 之后连续 105 s 无变化**的那一版。
>
> **自校验（15 条里 9 条能在该轮自己的产物里找到同值，9/9 相符）**：3 个 `.pt` ↔ `v5.3.0/evidence/T20/manifest.json` 的 `kept_in_tmp`（3/3）；`ckpt_assertions.json` ↔ `v5.2.0/evidence/XLADDER/manifest.json` 与 `analysis.json`（2/2）；`T20/RUN.md`、`T20/diagnostics.json` ↔ `T20/manifest.json`（2/2）；`XL30/ckpt_assertions.json`、`XL30/diagnostics.json` ↔ `v5.4.0/evidence/XL30/manifest.json` 的 `files`（2/2，`missing` = []）；`XLADDER/analysis.json` ↔ `v5.3.0/.../analysis.json` 的 `evidence_source` 引用。另有一条跨轮互证：`XL30/ckpt_assertions.json` 内记的 `v520_assertions_sha256` = `9782a842…`，与本节登记的 `v5.2.0/.../ckpt_assertions.json` **逐位相同**。其余 6 条（`XLADDER/ANALYSIS.md`、`XLADDER/CROSS_ROUND_5.md`、`v5.2.0/SUMMARY_FOR_DECISION.md`、`T20/analysis.json`、`T20/ANALYSIS.md`、`XL30/RUN.md`）是分析侧/发布侧后加的产物，`manifest.json` 里本就没有条目，只能以本节实测值为准。
>
> **`.tmp_*` 行沿用既有写法**（相对路径 `.tmp_...`），不引入 `/mnt/d/...` 形式。

| 名称 | 路径 | 大小 (B) | SHA-256 | 用途一句话 |
| --- | --- | ---: | --- | --- |
| T20 seed0 | `.tmp_v530_t20/t20_seed0.pt` | 34,609 | `69512dc3f2e51bbce2b9902584ee2356fbc46c49c92e7b5aec0e32efecbec272` | v5.3.0 T20-001 `--learner repair` 策略种子 0 的训练产物（4,000 episode × 600 决策、窗口 seeds 13–16、随机初始化不用 BC）；dev 池化 3,600 条均值 112.2333 / 18 条跨过 300 决策的载体 |
| T20 seed1 | `.tmp_v530_t20/t20_seed1.pt` | 34,609 | `41c5857299db491a63c2c4e6d294f60a885533809d9c2f211b7d83d50df9c051` | 同上（策略种子 1） |
| T20 seed2 | `.tmp_v530_t20/t20_seed2.pt` | 34,609 | `e1a40c875c946f603b5de0f7629a0a00a4e0e18b0b6f647591dde3164349d1b5` | 同上（策略种子 2） |
| XLADDER 统计 | `Document/Update/Alpha/v5.2.0/evidence/XLADDER/analysis.json` | 61,056 | `36be028a5514bafde7bfd675976146339d9d2f52e6f74d1521808e358dddb6e7` | v5.2.0 的离线统计重算（零训练、引擎 0 次；2,313 行、10,800 行原始 JSONL 逐行重算）：P1 T 90.1172 / S-stochastic 127.8628 > 基线 74.8575，P2 4.2500% / 14.0000% > 1.2500% ⇒ 命中 `S-ok` |
| XLADDER 分析 | `Document/Update/Alpha/v5.2.0/evidence/XLADDER/ANALYSIS.md` | 16,613 | `7dab22e3bce4d466183eeb025ab8ef89f7b5d0c5e6e1f243d9c0834ff3294912` | v5.2.0 独立分析正文（288 行）：P1 与 P2 都成立 ⇒ `S-ok`；P3 中位数 T 46.0 低于基线 54.0，满段 T 1/3,600、S 5/3,600（均为描述性） |
| XLADDER 跨轮 | `Document/Update/Alpha/v5.2.0/evidence/XLADDER/CROSS_ROUND_5.md` | 11,057 | `256d8dfa452b2d4144cedbe62e3f603d6853e610bf545b9a47d9a9e264b84143` | 600 帧代 × 1,200 帧代并排对照（本批唯一跨轮文件；每行数字都由各轮自己的原始 JSONL 重算，未沿用 `RUN.md` 转述值） |
| XLADDER ckpt 断言 | `Document/Update/Alpha/v5.2.0/evidence/XLADDER/ckpt_assertions.json` | 4,303 | `9782a842d91098650ab4ec332b5afcaa78a17b747da86684a06c10b08e9be383` | v5.2.0 的 `S-ckpt` 断言：T/S 各 3 个冻结 ckpt 的 `exists` + `sha256` + 字节数逐条比对（6/6 相符，`verdict` = PASS；缺失或不符即停轮，不许替换） |
| v5.2.0 决策摘要 | `Document/Update/Alpha/v5.2.0/SUMMARY_FOR_DECISION.md` | 24,016 | `dd40c2541a76e4d555d3661350073bfddc0e8dc893d5bb030dfd318f9b9265b4` | v5.2.0 的「10 分钟读完」决策摘要（零训练外推到 1,200 帧预算；P1/P2 均成立 ⇒ `S-ok`，并给出复现入口与跨代并排要点） |
| T20 统计 | `Document/Update/Alpha/v5.3.0/evidence/T20/analysis.json` | 73,810 | `99ccf854c5a6c4a186286645c6f0e7433714580d60525a810477035e7aa29a33` | v5.3.0 T20-001 的离线统计重算（2,443 行；引擎 0 次）：P1 112.2333 成立，P2 18/3,600 = 0.5000% 不成立 ⇒ 预注册未覆盖、三条停止条件都不适用 |
| T20 分析 | `Document/Update/Alpha/v5.3.0/evidence/T20/ANALYSIS.md` | 20,780 | `0696e0792e7613d34e5d18bf5498a77e20e7853b040c1c31b3b6c07bf1939e65` | v5.3.0 / T20 独立离线分析（242 行）：均值 112.2333（+49.93%）与中位数 92.0 抬高，但 300 决策率 0.5000% < 1.2500%、满段 dev 0/3,600 ⇒「奖励是均值，学到的也是均值」 |
| T20 执行记录 | `Document/Update/Alpha/v5.3.0/evidence/T20/RUN.md` | 17,856 | `9d516ff683b907be175da95f379f51f0525fa8aa6b2284016bf500b84bae4da9` | v5.3.0 T20-001 执行记录（215 行；1 次 `preflight` + 3 训练 + 3 评测，退出码全 0，墙钟合计 3,693.115 s；D3 保真门 5/5 PASS、引擎与运行器 sha256 与预注册一致）——明确不做任何统计推断 |
| T20 诊断 | `Document/Update/Alpha/v5.3.0/evidence/T20/diagnostics.json` | 22,620 | `716b9336ba8987dab793f12b5b1f0e6162b68f5ce773f7ce45266ed1921342de` | v5.3.0 T20 的原始诊断（1,176 行）：12 个 (种子,窗口) 格的 `return_decisions` 分布与 `S-collapse` 判定（单值格 0/12 < 10 ⇒ 未触发） |
| XL30 执行记录 | `Document/Update/Alpha/v5.4.0/evidence/XL30/RUN.md` | 43,887 | `8b339834cf9467482f159a7dc6b66f446b402d0d7fc5a34dc39c05025c44aa6f` | v5.4.0 XLADDER-30S-001 执行记录（445 行；900 决策 / 1,800 帧预算，**零训练 / 零新 checkpoint / 零代码改动**；D3 保真门 5/5 PASS、ckpt 断言 6/6 PASS、`S-floor` 触发 ⇒ P2 降级为「≥1 条」；统计推断不在本文件范围） |
| XL30 诊断 | `Document/Update/Alpha/v5.4.0/evidence/XL30/diagnostics.json` | 76,605 | `d2212f19019fbbd9119263e0661e2305ef4e8896c091ea6fa07050679f1dcfbc` | v5.4.0 X30 的原始诊断（3,753 行）：4 臂共 40 个 (种子,窗口) 格的分布（`B` 4 格 × 300、`T` / `S_stochastic` 各 12 格 × 300、`S_greedy` 12 格 × 1）、`vs_baseline_900`、`integrity`、greedy 确定性核对；`no_inference` 明示只做原始计数与描述统计（无 Fisher / 无置换 / 无区间 / 无 P1–P4 判定） |
| XL30 ckpt 断言 | `Document/Update/Alpha/v5.4.0/evidence/XL30/ckpt_assertions.json` | 6,350 | `a1bcc927af626ceb6c64e7b36a1d19f3ee9a0a230c5ad7ee13c08d0f14544803` | v5.4.0 的 `S-ckpt` 断言：6 个冻结 ckpt 与 v5.2.0 断言表（`v520_assertions_sha256` = `9782a842…`）及上游登记表**逐位相符**（`v520_all_match` = true，`verdict` = PASS） |

### 13.1 跳过（实测缺失，未登记）

| 路径 | 实测 | 处理 |
| --- | --- | --- |
| `.tmp_v540_x30/` 新 checkpoint | 递归 `*.pt` = **0 个**（22:08 与 22:17 两次扫；该目录下只有 `work/data/v540-*/w*/…` 引擎私有数据，以及 `work/runs/v540-*/{gate,summary,fidelity}.json`、`run.log`、`*-stderr.log`） | **跳过不登记**；v5.4.0 为零训练轮，与预期一致 |
| `Document/Update/Alpha/v5.3.0/SUMMARY_FOR_DECISION.md` | **不存在**（`v5.3.0/` 实测 19 个文件 = `PREREGISTRATION_T20_001.md` + `evidence/T20/` 18 个，无该摘要；22:18 复查仍不存在） | **跳过不登记** |

> **原本也缺失、但在本轮运行期间补齐的 3 项**：`v5.4.0/evidence/XL30/{RUN.md,diagnostics.json,ckpt_assertions.json}` 在首次检查（22:08 前后）时**整目录不存在**（`v5.4.0/` 当时只有 `PREREGISTRATION_XLADDER_30S_001.md` 4,226 B），22:13–22:16 发布完成、22:17:56 实测 3 份齐全 ⇒ **已补登到上表**，不计入跳过项。
>
> 现状：**1 条点名路径仍缺失**（v5.3.0 摘要）+ **1 项 checkpoint 为 0 个**；本次点名路径里 0 字节文件 **0 个**，没有「文件在但为空」的情况。

### 13.2 存在但不在最小集（未登记，供后续按需补登）

| 路径 | 大小 (B) | 说明 |
| --- | ---: | --- |
| `Document/Update/Alpha/v5.2.0/evidence/XLADDER/RUN.md` | 36,228 | 该轮执行记录（同目录 `SUMMARY_FOR_DECISION.md` 转述为 415 行 / 17 条命令 + 退出码 + 墙钟 + 9 条偏差） |
| `Document/Update/Alpha/v5.2.0/evidence/XLADDER/diagnostics.json` | 54,554 | 有效性诊断；`manifest.json` 内有其内联 sha256 `fc881f4b…` |
| `Document/Update/Alpha/v5.2.0/evidence/XLADDER/manifest.json` | 11,491 | 发布清单（`created_at` 21:40:34，早于 21:51 的分析侧产物） |
| `Document/Update/Alpha/v5.2.0/README_INDEX.md` | 10,826 | 该轮索引入口 |
| `Document/Update/Alpha/v5.3.0/evidence/T20/manifest.json` | 7,235 | T20 发布清单（含 `kept_in_tmp` 3 个 ckpt 的内联 sha256，本节已用作自校验） |
| `Document/Update/Alpha/v5.2.0/PREREGISTRATION_XLADDER_001.md` | 5,327 | 冻结预注册（判据来源，禁改；sha256 `74425994…`） |
| `Document/Update/Alpha/v5.3.0/PREREGISTRATION_T20_001.md` | 4,640 | 冻结预注册（`analysis.json` 记其 sha256 `5005580f…`） |
| `Document/Update/Alpha/v5.4.0/PREREGISTRATION_XLADDER_30S_001.md` | 4,226 | v5.4.0 预注册（判据来源，禁改；v5.4.0 发布前该目录只有这一个文件） |
| `Document/Update/Alpha/v5.4.0/evidence/XL30/` 其余 **20 个文件** | 6,852,340（合计） | `eval_{T,S_stochastic,S_greedy}_seed*.jsonl`（9 份原始评测）、`cfair_1800.jsonl`、`manifest.json`（`files` 20 条、`missing` = []）、`preflight.json`、`fidelity_v540.json`、`arms_vs_v520.json`、`baseline_repro.json`、`audio_cfg_audit_self.json`、`commands.tsv`、`status.json`、`runner_watch.json`、`verify_publish.json`（该目录实测共 23 个文件：上表 3 个 + 此处 20 个） |

### 13.3 不重复登记（实测已核）

- **已在册路径 0 条**：追加前 `grep -nE 'v5\.2\.0|v5\.3\.0|v5\.4\.0|XLADDER|T20|XL30' ASSET_SYNC.sha256` → **0 匹配**；`ASSET_SYNC.md` 同一 pattern（本节之前）→ **0 匹配**。新增 **15 条**与既有 97 条无交集（v5.2.0/v5.3.0 的 12 条 22:13 追加、v5.4.0 的 3 条 22:18 追加，两次追加前都复核过 0 匹配）。
- **重复行 0 条**：追加前与追加后 `head -97 ASSET_SYNC.sha256 | cut -c67- | sort | uniq -d`、`tail -15 …/cut -c67- | sort | uniq -d`、`tail -15 …/cut -c1-64 | sort | uniq -d` 实测**均为空**；新增 15 条自身 **15 路径 / 15 哈希全唯一**。
- **行数账**：97 + 15 = **112** 行（`wc -l ASSET_SYNC.sha256` = 112）。
- **Git 跟踪状态（实测）**：`git status --porcelain -- Document/Update/Alpha/v5.2.0 v5.3.0 v5.4.0` 三行全为 `??`（整目录 untracked），`git ls-files` 计数 **0** ⇒ 这些证据同样**无法由 Git 重建**，本节 + `ASSET_SYNC.sha256` 是它们目前的字节身份记录；本轮按约束**未** `git add`。
- **基线（追加前，97 行）**：`sha256sum -c` = **96 OK / 1 FAILED**，唯一 FAILED 是 §9.8 允许的 `v4.2.0/journal.jsonl`。
- **附带发现（超出本轮追加范围，未改）**：`ASSET_SYNC.md` §12 表格里 `v5.0.0/evidence/REG/RUN.md` 一行仍写 21,090 B / `048cbb95…`，而 `ASSET_SYNC.sha256` 第 95 行已在 21:24:51 被重钉为 24,836 B / `55bb5160…`（当前文件实测通过）⇒ §12 该行与 §12.3 的「2 FAILED」描述已过期，实际只剩 §9.8 那 1 条。需要时单独修 §12，本轮只追加、未动既有行。

### 13.4 校验结论

`cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256`：**112 行 = 111 OK / 1 FAILED**（复验时刻 `2026-11-21T22:18+0800`）。新登记 15 行贡献 **15/15 OK**（v5.2.0/v5.3.0 的 12 条自 22:13 起通过，v5.4.0 的 3 条自 22:18 起通过）；唯一 FAILED 是 §9.8 已知并允许的活文件 `Document/Update/Alpha/v4.2.0/journal.jsonl`，本轮**未改该行**、也**未重钉**。

> **一句话结论**：本次追加的 **15 条**资产校验全通过（15/15 OK），整体 FAILED 数**没有新增**，仍只有已知允许的 `v4.2.0/journal.jsonl` 那 1 条（111 OK / 1 FAILED）。

---

## 14. v5.5.0–v5.7.0 新资产（2026-11-21）

> **为什么有这一节**：v5.5.0（THRESH-001，阈值导向奖励：首次达到 300 决策 +1、600 决策再 +1）与 v5.6.0（CURR-001，课程式阈值 75/150/300/600 各 +1）各产出 **3 个训练 checkpoint**，共 6 个，全部在 gitignored 的 `.tmp_v550_th/`、`.tmp_v560_curr/` 下（`.gitignore:32:.tmp_*/`，**不可由 Git 重建**）；两轮另发布机器可读证据。按 AGENTS.md §8.2「新增/移动/删除任何资产，必须同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`」，本节与 `ASSET_SYNC.sha256` **末尾追加的 14 行**成对登记。v5.7.0（CF600-001）在快照时刻**尚未发布证据**，见 §14.1；v5.6.0 的 `analysis.json` 首次实测（23:46 / 23:48）时**不存在**，于**本轮运行期间 23:51:30 发布**，已按点名要求补登到下表（14 条而非 13 条）。
>
> **登记口径（最小集）**：只收 6 个 `thresh_seed*.pt` / `curr_seed*.pt` 训练产物 + 点名要求的 **8 份证据**（v5.5.0 的 `analysis.json` / `ANALYSIS.md` / `RUN.md` / `diagnostics.json` 4 份，v5.6.0 的 `analysis.json` / `RUN.md` / `diagnostics.json` / `manifest.json` 4 份）；**不**收每轮 6 份 `train_seed*.jsonl` / `eval_armA_seed*.jsonl`（v5.5.0 合计 13,595,319 B、v5.6.0 合计 13,589,564 B 的原始 episode 记录）、`*.py` / `*.sh` 生成与探针脚本、`commands.tsv`、`preflight.json`、`fidelity_*.json`、`ab_compare.json`、`selfcheck_*.json`、`diff_stats.json`、`verify.out`、`runner.diff`、`audio_cfg_audit_self.json`、v5.5.0 的 `manifest.json`、v5.6.0 的 `ANALYSIS.md`（任务未点名），也不收 §14.2 列出的那些文件。
>
> **快照时刻**：13 条取于 `2026-11-21T23:47+0800`，`v5.6.0/evidence/CURR/analysis.json` 1 条取于 `2026-11-21T23:53+0800`（该文件 23:51:30 才落盘，取前后两次重算值与字节数一致的那一版）。下表每条都来自当次真实 `sha256` 计算（Python `hashlib` 逐块读取 1 MiB，非手抄、非推算，也未取用任何 `manifest.json` 里的内联值代替实测）；追加后 `sha256sum -c` 对这 14 行**全部 OK**（见 §14.4）。
>
> **自校验（14 条里 10 条能在该轮自己的产物里找到同值，10/10 逐位相符）**：3 个 `thresh_seed*.pt` ↔ `v5.5.0/evidence/THRESH/manifest.json` 的 `kept_in_tmp`（3/3）；3 个 `curr_seed*.pt` ↔ `v5.6.0/evidence/CURR/manifest.json` 的 `checkpoints_not_published`（3/3）；`THRESH/RUN.md`、`THRESH/diagnostics.json` ↔ `THRESH/manifest.json` 的 `published`（2/2）；`CURR/RUN.md`、`CURR/diagnostics.json` ↔ `CURR/manifest.json` 的 `published`（2/2）。另有一条同源互证：`THRESH/manifest.json` 的 `refreshed_in_place` 记 `RUN.md` 的 `sha256_after` = `d3bc2597…`，与本节登记的 RUN.md **逐位相同**（该轮在首次发布后重写过自己的 RUN.md）。其余 4 条（`THRESH/analysis.json`、`THRESH/ANALYSIS.md`、`CURR/analysis.json`、`CURR/manifest.json`）里，前两者是分析侧 23:17 才生成的产物、晚于 22:55 的 manifest，第三个是 23:51:30 的分析侧产物、晚于 23:43:08 的 manifest，manifest 本就没有条目；manifest 当然不记录自身哈希 —— 这 4 条只能以本节实测值为准。
>
> **`.tmp_*` 行沿用既有写法**（相对路径 `.tmp_...`），不引入 `/mnt/d/...` 形式。

| 名称 | 路径 | 大小 (B) | SHA-256 | 用途一句话 |
| --- | --- | ---: | --- | --- |
| THRESH seed0 | `.tmp_v550_th/thresh_seed0.pt` | 34,639 | `b5132dfd422f9d337648b2a93b852a8255c65535dd021c97025c84232e19272c` | v5.5.0 THRESH-001 `--learner thresh`（奖励只在首次达到 300 决策 +1、600 决策再 +1）策略种子 0 的训练产物（4,000 episode × 600 决策、训练窗口 seeds 13–16、随机初始化不用 BC）；dev 池化 3,600 条均值 93.4347 / ≥300 阈值 49 条的载体 |
| THRESH seed1 | `.tmp_v550_th/thresh_seed1.pt` | 34,639 | `8868b4ed39742d9d78e22b3ed9be7512af6416dde923082011e4ef86c31454a6` | 同上（策略种子 1） |
| THRESH seed2 | `.tmp_v550_th/thresh_seed2.pt` | 34,639 | `c186c73db7a1d6c0e4c432aea5d7931667661168ab4a79bc2665b964b5ee24e1` | 同上（策略种子 2） |
| CURR seed0 | `.tmp_v560_curr/curr_seed0.pt` | 34,619 | `2d4f0bc720f248794987866cf67bbf9963bf3be2a9b9b60cfbb365330a950c9a` | v5.6.0 CURR-001 `--learner curr`（课程式阈值 75/150/300/600 各 +1，替代 v5.5.0 的硬阈值）策略种子 0 的训练产物（4,000 episode × 600 决策）；dev 池化均值 78.1453、≥150 654 条、≥300 29 条的载体 |
| CURR seed1 | `.tmp_v560_curr/curr_seed1.pt` | 34,619 | `bcce9774203e546f22bfb6a20a97202719ffe1bc1e3a77dd327bb1b4a8f0a04c` | 同上（策略种子 1） |
| CURR seed2 | `.tmp_v560_curr/curr_seed2.pt` | 34,619 | `64c526738153ea48e03d07bf2000463e91f936b438c926a0c6d92bb8f35c5cfe` | 同上（策略种子 2） |
| THRESH 统计 | `Document/Update/Alpha/v5.5.0/evidence/THRESH/analysis.json` | 63,350 | `d96f15bb73d8864765bd9a9d727b07fa1a64e9bc89765e227dd6fb16f143edcb` | v5.5.0 的离线统计重算（引擎 0 次；27,600 行 JSONL 逐行重算）：P1 点估计 49/3,600 = 1.3611% 成立、但单侧 Fisher 对 C-fair p = 0.4516 不显著；P2 93.4347 ≥ 74.8575 成立（置换 p = 4.9998e-05）；P3 满段 0/3,600；S-sparse 触发（dev 98.6389% > 80%） |
| THRESH 分析 | `Document/Update/Alpha/v5.5.0/evidence/THRESH/ANALYSIS.md` | 17,377 | `73d894168b1bd16a048f7c1077b08699a5d18e60211fad56844cd0c64f83b78b` | v5.5.0 / THRESH 独立分析正文（214 行）：S-sparse 触发 ⇒ 按预注册转课程式阈值路线；明确三重折扣（对基线不显著、12 格对 4 格精确置换 p = 0.1670、98.6389% 退化）故**不**写成「阈值导向奖励被证明有效」 |
| THRESH 执行记录 | `Document/Update/Alpha/v5.5.0/evidence/THRESH/RUN.md` | 28,702 | `d3bc2597b178c2dd670b1df4c0f41d19a44da519117fec3cf0da19b201b3fa7b` | v5.5.0 执行记录（360 行；6 条 A/B + 1 次 fidelity + 3 训练 + 3 评测，退出码 0 共 13 条、墙钟合计 5,291.6 s；D3 保真门 5/5 PASS、2,165 个前缀检查点 0 不匹配、音频 `audible` 0）—— 明确不做任何统计推断 |
| THRESH 诊断 | `Document/Update/Alpha/v5.5.0/evidence/THRESH/diagnostics.json` | 41,485 | `235970dcad5f10268ab996a5adcc037f0de875aae75831c0392fe00fe907dcc5` | v5.5.0 原始诊断（2,119 行；schema `v550-thresh-diagnostics/1`）：12 个 (种子,窗口) 格的分布、训练池化、`criteria_raw` 与阈值并排、`live_counters`；`what` 明示只做计数/均值/分位/去重计数（无检验、无区间、无 p 值） |
| CURR 执行记录 | `Document/Update/Alpha/v5.6.0/evidence/CURR/RUN.md` | 34,470 | `83485be0b8ee5f4d3b4f73169dd782157ad63fcac1be2c6d8a7e8f1248626812` | v5.6.0 执行记录（386 行；9 条 A/B + 1 次 fidelity + 3 训练 + 3 评测，退出码 0 共 15 条、墙钟合计 6,103.4 s；4 个 learner 的 A/B 各 60/60、selfcheck 32/32）；P1 0.8056% ≤ C-fair、P2 18.1667% > 8.7000%、P3 78.1453 ≥ 74.8575、P4 满段 0/3,600、S-sparse 68.6944% 触发 |
| CURR 诊断 | `Document/Update/Alpha/v5.6.0/evidence/CURR/diagnostics.json` | 60,393 | `d604db8673548bf6d5cac8b0b06486b2a56a233f71789596224c54dbc641813c` | v5.6.0 原始诊断（2,902 行；schema `v560-curr-diagnostics/1`）：75/150/300/600 四个阈值各自的达到条数、12 格分布、`degeneracy`、`criteria_raw`；明示无检验/无区间/无 p 值 |
| CURR 清单 | `Document/Update/Alpha/v5.6.0/evidence/CURR/manifest.json` | 19,027 | `8dc00e306c5e5baf3fe6d86e8c1b9c98bd715fc5349ce9c0f7fc453aece891aa` | v5.6.0 发布清单（`generated_at` 23:43:08）：31 条 `published` 的内联 sha256、3 个 ckpt 的 `checkpoints_not_published`、`verdicts`（D3 5/5、A/B 4 个 learner × 60/60、selfcheck 32/32、`audio_audible` 0）、`runner`（`--learner curr` 改动 2,127 → 2,407 行） |
| CURR 统计 | `Document/Update/Alpha/v5.6.0/evidence/CURR/analysis.json` | 74,723 | `922dfce16d1e129fb3708ee16a6aec5ada962b117212d98a1e8f99869f43a849` | v5.6.0 的离线统计重算（`.tmp_analysis_56/analyze_curr.py`，引擎 0 次；27,600 行 JSONL 逐行重算）：P1 29/3,600 = 0.8056% **不成立**（≤ C-fair 1.2500%）、P2 654/3,600 = 18.1667% 成立、P3 78.1453 ≥ 74.8575 成立、P4 描述性（满段 0/3,600、中位 54、max 445）；判定 `S-mixed`（P2 立 / P1 不立）+ `S-sparse` 在 **dev 口径** 68.6944% > 50% 触发、训练口径 11.5500% 不触发（口径歧义已逐条列出） |

### 14.1 跳过（实测缺失或未发布，未登记）

| 路径 | 实测 | 处理 |
| --- | --- | --- |
| `Document/Update/Alpha/v5.7.0/evidence/CF600/RUN.md` | **不存在**（`v5.7.0/evidence/` 整目录不存在） | **跳过不登记** |
| `Document/Update/Alpha/v5.7.0/evidence/CF600/labels_extension.jsonl` | **不存在** | **跳过不登记** |
| `Document/Update/Alpha/v5.7.0/evidence/CF600/labeller_selfcheck.json` | **不存在** | **跳过不登记** |
| `Document/Update/Alpha/v5.7.0/evidence/CF600/manifest.json` | **不存在** | **跳过不登记** |

> 点名路径里 0 字节文件 **0 个** —— 没有「文件在但为空」的情况；上述 4 条全部是**目录里没有该文件**。
>
> **原本缺失、但在本轮运行期间补齐的 1 项**：`Document/Update/Alpha/v5.6.0/evidence/CURR/analysis.json` 在 23:46:33 与 23:48:48 两次实测**不存在**（当时该目录 33 个文件），23:51:30 与同批的 `ANALYSIS.md`（22,703 B）一同落盘（生成器 `.tmp_analysis_56/analyze_curr.py`，104,490 B；`created_utc` = `2026-11-21T15:51:28Z`），23:53 复测字节数与 sha256 均未再变 ⇒ **已补登到上表**，**不计入**跳过项（因此本轮点名路径的跳过项是 4 条而不是 5 条）。
>
> **v5.7.0 运行现状（只作说明，不登记）**：`.tmp_v570_cf600/` 在快照时刻仍在推进 —— `labels_extension.jsonl` 8,427,235 B / `labels_stats.json` 8,902 B / `merge.json` 1,255 B 于 23:45:45 落盘，`ckpt/`、`train/` 目录时间戳 23:46:16，`eval/eval_{greedy,stochastic}_seed*.jsonl` 仍在 23:47:21 增长，而 `Document/Update/Alpha/v5.7.0/evidence/` 未创建（`v5.7.0/` 只有 `PREREGISTRATION_CF600_001.md` 5,147 B）。按约束（只登记已落盘且可按字节校验的文件、不登记进行中的半成品）本轮**不登记** v5.7.0 任何路径；待该轮发布后按 §14.2 的口径补登。

### 14.2 存在但不在最小集（未登记，供后续按需补登）

| 路径 | 大小 (B) | 说明 |
| --- | ---: | --- |
| `Document/Update/Alpha/v5.5.0/PREREGISTRATION_THRESH_001.md` | 4,409 | 冻结预注册（判据来源，禁改；sha256 `d9978d1b…`，与 `analysis.json` / `manifest.json` 内联值相符） |
| `Document/Update/Alpha/v5.5.0/evidence/THRESH/manifest.json` | 19,718 | v5.5.0 发布清单（`created_at` 22:55:12；`kept_in_tmp` 3 条 ckpt 内联 sha256 已用作本节自校验） |
| `Document/Update/Alpha/v5.5.0/evidence/THRESH/` 其余 **27 个文件** | 13,833,237（合计） | 6 份原始 JSONL（13,595,319）、`ab_compare.json`、`selfcheck_thresh.json`、`fidelity_v550.json`、`preflight.json`、`commands.tsv`、`runner.diff`、`verify.out`、`audio_cfg_audit_self.json`、9 个 `*.py`、3 个 `*.sh`（该目录实测共 31 个文件：本节登记 4 个 + 此处 27 个） |
| `Document/Update/Alpha/v5.6.0/PREREGISTRATION_CURR_001.md` | 3,925 | 冻结预注册（`manifest.json` 记其 sha256 `739c28e5…`） |
| `Document/Update/Alpha/v5.6.0/evidence/CURR/ANALYSIS.md` | 22,703 | v5.6.0 / CURR 独立分析正文（任务未点名，故不登记；与 `analysis.json` 同批 23:51:30 落盘） |
| `Document/Update/Alpha/v5.6.0/evidence/CURR/` 其余 **31 个文件** | 13,856,173（合计） | 6 份原始 JSONL（13,589,564）、`ab_compare.json`、`selfcheck_curr.json`、`diff_stats.json`、`fidelity_v560.json`、`preflight.json`、`commands.tsv`、`runner.diff`、`verify.out`、`audio_cfg_audit_self.json`、14 个 `*.py`、1 个 `*.sh`（该目录实测共 35 个文件：本节登记 4 个 + 此处 31 个） |
| `Document/Update/Alpha/v5.7.0/PREREGISTRATION_CF600_001.md` | 5,147 | v5.7.0 冻结预注册（判据来源，禁改；该目录发布前只有这一个文件） |

### 14.3 不重复登记（实测已核）

- **已在册路径 0 条**：追加前 `grep -nE 'v5\.5\.0|v5\.6\.0|v5\.7\.0|tmp_v550|tmp_v560|tmp_v570|THRESH|CURR|CF600|thresh_seed|curr_seed' ASSET_SYNC.sha256` → **0 匹配**；`ASSET_SYNC.md` 同一 pattern（本节之前）→ **0 匹配**。新增 **14 条**与既有 112 条无交集。
- **重复行 0 条**：追加后 `awk '{print $2}' ASSET_SYNC.sha256 | sort | uniq -d` 与 `awk '{print $1}' … | sort | uniq -d` 实测**均为空**；新增 14 条自身 **14 路径 / 14 哈希全唯一**。
- **行数账**：112 + 14 = **126** 行（`wc -l ASSET_SYNC.sha256` = 126）。分两次追加：23:47 追加 13 条（6 ckpt + v5.5.0 4 份 + v5.6.0 3 份），23:52 追加 v5.6.0 的 `analysis.json` 1 条（该文件在两次首次检查时确实不存在，见 §14.1 的补记）。
- **Git 跟踪状态（实测）**：`git status --porcelain -- Document/Update/Alpha/v5.5.0 v5.6.0 v5.7.0` 三行全为 `??`（整目录 untracked），三者 `git ls-files` 计数各为 **0**；6 个 `.pt` 命中 `.gitignore:32:.tmp_*/`（`.tmp_*/` 规则）。即这些证据与 checkpoint 都**无法由 Git 重建**，本节 + `ASSET_SYNC.sha256` 是它们目前的字节身份记录；本轮按约束**未** `git add` / `git commit`。
- **基线（追加前，112 行，23:47 实测）**：`sha256sum -c` = **111 OK / 1 FAILED**，唯一 FAILED 是 §9.8 允许的 `v4.2.0/journal.jsonl`。

### 14.4 校验结论

`cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256`：**126 行 = 125 OK / 1 FAILED**（复验时刻 `2026-11-21T23:52+0800`）。

- 既有 112 行贡献 **111 OK / 1 FAILED**（与 §13.4 的基线一致，未因本轮改动而变化）。
- 新登记 14 行贡献 **14/14 OK**：6 个 `.pt`、v5.5.0 的 4 份证据、v5.6.0 的 4 份证据（含 23:51:30 新发布的 `analysis.json`）。
- 唯一 FAILED 是 `Document/Update/Alpha/v4.2.0/journal.jsonl` —— §9.8 已知并**允许**的活文件漂移，本轮**未改该行**、也**未重钉**。
- **无并发写入导致的失败**：新发布的 `v5.6.0/evidence/CURR/analysis.json` 在 23:51:30 落盘后，23:52 与 23:53 两次独立重算的字节数（66,109）与 sha256 完全一致，且 `sha256sum -c` 通过，因此**不需要**等待 60 s 重判；其余 13 条自 23:47 起字节数与哈希均未变。本轮没有出现「登记后被重发布」的行（对照 §12.3 里 v5.0.0 的那次并发写入）。

> **一句话结论**：本轮成对登记 **14 条**（6 个 `.tmp_*` checkpoint + 8 份证据），`sha256sum -c` **125 OK / 1 FAILED**，新登记 14 条 **全部 OK**；整体 FAILED 数**没有新增**，仍只有已知允许的 `v4.2.0/journal.jsonl` 那 1 条。v5.7.0（CF600-001）的 4 条点名证据在快照时刻（`23:53:08` 最后复测：`v5.7.0/` 仍只有 1 个预注册文件）仍未发布，已按 §14.1 **如实列出、未登记**。

---

## 15. v5.7.0–v5.9.0 新资产（2026-11-22）

> **为什么有这一节**：v5.7.0（CF600-001，反事实标签覆盖 0–300 → 0–449 决策）产出 **5 个** `round_extension.pt`（`seed{0,1,2}` + `control_base` / `control_new`），v5.8.0（OBS48-001，观测轴 12 弹 → 48 弹）产出 **6 个** `model.pt`（`k12` / `k48` × `seed{0,1,2}`），共 **11 个训练 checkpoint**，全部在 gitignored 的 `.tmp_v570_cf600/`、`.tmp_v580_obs48/` 下（`.gitignore:32:.tmp_*/`，**不可由 Git 重建**）；两轮另发布机器可读证据，v5.9.0 发布跨轮综合报告。按 AGENTS.md §8.2「新增/移动/删除任何资产，必须同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`」，本节与 `ASSET_SYNC.sha256` **末尾追加的 22 行**成对登记。
>
> **登记口径（最小集）**：只收 11 个 checkpoint + 点名要求的 **11 份文档**（v5.7.0 的 `evidence/CF600/{RUN.md,analysis.json,ANALYSIS.md,labels_extension.jsonl,manifest.json}` 5 份 + `SUMMARY_FOR_DECISION.md`、`README_INDEX.md` 2 份；v5.8.0 的 `evidence/OBS48/{RUN.md,manifest.json}` 2 份；v5.9.0 的 `SYNTHESIS_2026-11-21.md`、`README_INDEX.md` 2 份）。**不**收每轮的原始 `eval_*.jsonl` / `labels_*.jsonl`、`*.py` / `*.sh` 生成与探针脚本、`commands.tsv`、`preflight.json`、`fidelity_d3.json`、`diagnostics.json`、`audio_audit.json`、`controls.json`、`train_*.json`（含 `train_seed*.jsonl`）、`verify_publish.json`、两轮的 `PREREGISTRATION_*`（禁改件，只引其 sha256），也不收 §15.2 列出的那些文件。
>
> **快照时刻**：第一遍实测 `2026-11-22T00:26+0800`，第二遍（间隔 ≥ 100 s）`00:28:15` **22 条字节数与 sha256 全部相同**，因此下表每条都是「连续 ≥ 100 s 无变化」的那一版；用例沿 `sha256sum -c` 同款口径逐文件重算（Windows 侧 `Get-FileHash -Algorithm SHA256` 与 WSL 侧 `sha256sum` 已在追加后用 `sha256sum -c` 交叉确认，见 §15.4）。**v5.8.0 的 `evidence/OBS48/{analysis.json,ANALYSIS.md}` 在两次实测（00:26、00:28）时都不存在**，按口径跳过并如实记入 §15.1；此外本轮点名路径里 **0 字节文件 0 个**。
>
> **自校验（22 条里 16 条能在该轮自己的产物里找到同值，16/16 逐位相符）**：6 个 `model.pt` ↔ `v5.8.0/evidence/OBS48/manifest.json` 的 `checkpoints`（6/6，含 `bytes` 与 `kept_in`）；5 个 `round_extension.pt` ↔ `v5.7.0/evidence/CF600/manifest.json` 的 `files.ckpt_seed*.pt` / `files.ckpt_control_*.pt` 的 `source_sha256`（5/5，且 `source` 字段逐字指向本节登记的 5 个 `.tmp` 路径）；`CF600/labels_extension.jsonl` 与 `CF600/RUN.md` ↔ 同一 manifest 的 `files`（2/2）；`OBS48/RUN.md` ↔ `OBS48/manifest.json` 的 `files`（1/1）；`CF600/analysis.json` ↔ `v5.7.0/SUMMARY_FOR_DECISION.md` 与 `v5.7.0/evidence/CF600/ANALYSIS.md` 同时记的 91,330 B / `e7a14776…`（2 份独立同值）；`CF600/manifest.json` ↔ `CF600/analysis.json` 内联的 `sha256` = `92a74801…`（1/1）。**另一条旁证**：`v5.7.0/evidence/CF600/` 里有 5 份 checkpoint 副本（`ckpt_seed*.pt` / `ckpt_control_*.pt`），与本节登记的 5 个 `.tmp` 原件 **5/5 逐位同哈希**（同一字节的两次落盘）。其余 6 条（`CF600/ANALYSIS.md`、`v5.7.0/SUMMARY_FOR_DECISION.md`、`v5.7.0/README_INDEX.md`、`OBS48/manifest.json`、`v5.9.0/SYNTHESIS_2026-11-21.md`、`v5.9.0/README_INDEX.md`）是分析侧/叙述侧后加的产物或清单自身，manifest 本就不记录自身、也不记录分析轮另写的文件 ⇒ 只能以本节实测值为准。
>
> **`.tmp_*` 行沿用既有写法**（相对路径 `.tmp_...`），不引入 `/mnt/d/...` 形式。

| 名称 | 路径 | 大小 (B) | SHA-256 | 用途一句话 |
| --- | --- | ---: | --- | --- |
| CF600 seed0 | `.tmp_v570_cf600/ckpt/seed0/round_extension.pt` | 34,669 | `feb072a43df3984e9a08539b4efbb3481877c6fae67d026860bd83d673d147f6` | v5.7.0 CF600-001 的 `round_extension` 训练产物（策略种子 0）：**5,166 基础状态（H=120，v4.5.0 只读复用）+ 2,400 新状态（H=240、决策 250–449、窗口 13–16）按 1:1 每 epoch 混合**，10,332 行 / 41 batch / 200 epoch；评测仍是 600 决策预算、dev 11/12/17/18、300 episode/格 |
| CF600 seed1 | `.tmp_v570_cf600/ckpt/seed1/round_extension.pt` | 34,669 | `4fd00dbda1f8174fbbdd2df8ce400f984b43bbe6bcb367bb574365382b0bab52` | 同上（策略种子 1） |
| CF600 seed2 | `.tmp_v570_cf600/ckpt/seed2/round_extension.pt` | 34,669 | `0d2d447a5e7030787bd97571c2514bdf06aec178463e10e4959eace9d1093a31` | 同上（策略种子 2） |
| CF600 对照 base | `.tmp_v570_cf600/ckpt/control_base/round_extension.pt` | 34,669 | `becc496c3bf00bc673a238fbb7fb2734c3bed7eccc81e6db817eb6cb4f5abcf7` | 两条**内部对照臂**之一的 `control_base` ckpt（与该轮 `train_control_base.json` 同批）：四臂描述性评测**各 1,200 条、只有 seed 0**，**不进 P1–P4**；用于 `G_mixed_horizon` 的 2×2 拆分，而该 2×2 缺「新覆盖 + H=120」格 ⇒ `coverage_leg` / `horizon_leg` 均为 `cannot_be_determined` |
| CF600 对照 new | `.tmp_v570_cf600/ckpt/control_new/round_extension.pt` | 34,669 | `e1794f5cac616576618397eb4900280086157d5e482e53a367e8967286022965` | 内部对照臂 `control_new`（**只用 H=240 新池**、1,200 条）的 ckpt：**天花板 270、0/1,200 越过 300**，而池化均值 134.7617 只比主臂低 5.5478 ⇒ 本轮「均值与 ≥300 率解耦」的直接载体；同样不进 P1–P4 |
| OBS48 k12 seed0 | `.tmp_v580_obs48/ckpt/k12/seed0/model.pt` | 34,377 | `3c3b3600aae7a23127f455be78137da01e2d4834f864e99d027ae57f095ea661` | v5.8.0 OBS48-001 的 **K=12 观测臂**（近 12 弹 × 6 = 72 特征 / 106 维，与 v4.5.0 `S` 同口径）种子 0 训练产物：同一批 **2,400 新状态**（随机行为采集、H=120）单池训练；池化 stochastic **1.0556%（38/3,600）**，作本轮的内部对照臂 |
| OBS48 k12 seed1 | `.tmp_v580_obs48/ckpt/k12/seed1/model.pt` | 34,377 | `acaf6f6881c42cc2fa9abfe250264ddf6ecf78cee6d30efee702c08a0865389e` | 同上（K=12，种子 1） |
| OBS48 k12 seed2 | `.tmp_v580_obs48/ckpt/k12/seed2/model.pt` | 34,377 | `ddea5c8fab4a9c7656560effab6a4e5272e305e9d22deda0b68340fd767409ff` | 同上（K=12，种子 2） |
| OBS48 k48 seed0 | `.tmp_v580_obs48/ckpt/k48/seed0/model.pt` | 89,673 | `0231e06fc472de355f6cc91e7f273538178bdfa4003c86bd54ec1cbdd49d5412` | v5.8.0 的 **K=48 观测臂**（近 48 弹 × 6 = 288 特征 / 322 维）种子 0 训练产物：观测轴是**唯一改动**，模型/优化器/评测协议冻结；池化 stochastic **2.5278%（91/3,600）** vs `S` 13.4444% ⇒ P1 不成立，K=48 − K=12 = **+1.4722 pp** ⇒ P3 成立 |
| OBS48 k48 seed1 | `.tmp_v580_obs48/ckpt/k48/seed1/model.pt` | 89,673 | `344ff2d1e1243773b6bfd27c2442c616a49c6c1ab4c5120ffe9bdb8bf96cda7a` | 同上（K=48，种子 1） |
| OBS48 k48 seed2 | `.tmp_v580_obs48/ckpt/k48/seed2/model.pt` | 89,673 | `faacdd3da78a0ecdfa9f6d4a29e8d32e0a38bfd2ca235e743e15016e28a89643` | 同上（K=48，种子 2） |
| CF600 执行记录 | `Document/Update/Alpha/v5.7.0/evidence/CF600/RUN.md` | 33,522 | `1853eefd87295d8bca0217cdd024acec5b2c9aa7bc5f31f4b668f5bd5a3e516c` | v5.7.0 执行记录（567 行；D3 保真门 **5/5 PASS**、前缀 2,165 checkpoint、labeller selfcheck **18/18** @ H=600、音频 **muted 71 / audible 0**、本轮引擎进程 **0**）；只列原始计数与门，**明确不做任何统计推断** |
| CF600 统计 | `Document/Update/Alpha/v5.7.0/evidence/CF600/analysis.json` | 91,330 | `e7a14776417233490c8aeecf93cf141805b1ca53483374036bac51ad5ddef698` | v5.7.0 独立分析层的机器可读重算（schema `v570-cf600-analysis/1`；`A_integrity` / `B_reference_recount` / `C_P1_reached_300` / `D_P2_reached_600` / `E_P3_mean` / `F_P4_descriptive` / `G_mixed_horizon` / `H_validity_discount` / `I_gate_and_labels` / `J_stop_conditions` / `K_claims` / `L_crosscheck` / `verdicts`）：**P1 7.1389% FAIL**（双侧 Fisher p = 1.9930013602e-21、CP95 [6.3190%, 8.0292%] 整段在门下方）、**P2 14 vs 5 PASS**（单侧 p = 0.0316116，双侧 0.0632232 须成对引）、**P3 140.309444 PASS**（+12.446667，置换 p = 4.99975e-05）⇒ 判定 **`S-mixed`**；折扣 ICC(1) 0.170433 / deff 51.9596 / n_eff 69.2846 |
| CF600 分析 | `Document/Update/Alpha/v5.7.0/evidence/CF600/ANALYSIS.md` | 28,645 | `8679a95f0c136e385a35206e6c93d1993eb9670eb6b5937cf5767039bccb3995` | v5.7.0 独立分析正文（517 行）：逐行重算 3,600 主臂 + 3,600 基线 + 4×1,200 对照；并明写三件限制 —— P2 不在 `S-*` 阶梯内（不能把 `S-mixed` 抬成 `S-ok`）、混合视界两条腿不可拆、四条折扣口径一致 |
| CF600 标签扩展 | `Document/Update/Alpha/v5.7.0/evidence/CF600/labels_extension.jsonl` | 8,427,235 | `216bc977b9ab0e9accfbed6603b92eef477053b891d1b2c8eb52cba665ac2dc3` | v5.7.0 新增标签池的原始记录（**2,400 行 / 43,200 条标签** = 2,400 状态 × 18 候选，H=240）：由 `S` 自身从 frame 8,663 起 rollout、只对「决策 ≥250 且 episode 曾越过 300」的状态打标；实测覆盖 **250–449**，450–599 无监督 |
| CF600 清单 | `Document/Update/Alpha/v5.7.0/evidence/CF600/manifest.json` | 10,906 | `92a74801da5373f66a09465b89096a1bcf1e8e8250670ccbd3b090a5ec9d3b54` | v5.7.0 发布清单（`created_at` `2026-11-22T00:07:53+0800`）：**35 条** `published` 的内联 sha256、`missing` = []、4 个 pin（`v450_cf_label` / `v432_fork_arm` / `v431_arm_a` / `v431_dense_driver`）全 `match`；5 个 ckpt 副本的 `source` 字段逐字指向本节登记的 5 个 `.tmp` 路径（`analysis.json` / `ANALYSIS.md` 不在其中：分析轮另写） |
| v5.7.0 决策摘要 | `Document/Update/Alpha/v5.7.0/SUMMARY_FOR_DECISION.md` | 37,000 | `cdedc96fbb2852134ea59e616a640d891c764c52266fbf7a98e424e37cbb2f37` | v5.7.0 的「10 分钟读完」决策摘要（342 行 / 6 节：§0 一句话 → §6 合规与冲突）；内含本批资产的二次记录（`analysis.json` 91,330 B / `e7a14776…`、`RUN.md` 33,522 B / `1853eefd…`、预注册 5,147 B / `5bdd2a28…`） |
| v5.7.0 索引 | `Document/Update/Alpha/v5.7.0/README_INDEX.md` | 15,648 | `1a24be7b67418ec57f3f172d40f346d6828d0c9adaf009a55c2f6e4a71097131` | v5.7.0 接手索引（102 行：「你是谁，先读哪份」+ 三十秒版本 + 复跑入口 + 姊妹轮）；同样记录 `analysis.json` 91,330 B / `e7a14776…` 与 `RUN.md` 33,522 B / 567 行 |
| OBS48 执行记录 | `Document/Update/Alpha/v5.8.0/evidence/OBS48/RUN.md` | 13,953 | `28d65a4d40f021a9e69d664aadd255998175be89cf6a074ac09e2c76885202f0` | v5.8.0 执行记录（264 行；执行窗口 00:02:15 → 00:23:00、墙钟 **20 min 45 s**；**28 个 runner 调用退出码全 0**（publish 后 29 个）；D3 5/5 PASS、前缀 **2,165/2,165** 零不匹配、labeller selfcheck + K=12 等价自证、音频 **muted 43 / audible 0**、本轮引擎进程 0）；§7 只陈述判据现状（P1 否 / P2 否 / P3 是 ⇒ 形状 `S-partial`），检验留给独立分析轮 |
| OBS48 清单 | `Document/Update/Alpha/v5.8.0/evidence/OBS48/manifest.json` | 14,265 | `440256615c7081fc0695ffa16afccf0d1dc0a2824e94ff0588fe7ee43969eeb4` | v5.8.0 发布清单（`created_at` `2026-11-22T00:24:42+0800`）：`files` **38 条**内联 sha256、`missing` = []、4 个 pin 全 `match`、`checkpoints` **6 条**（`k12`/`k48` × `seed0–2`，含 `bytes` 与 `kept_in` = `.tmp_v580_obs48`，即资产不复制进 `Document/`）；`analysis.json` / `ANALYSIS.md` 不在其中（该轮尚未跑独立分析） |
| v5.9.0 综合报告 | `Document/Update/Alpha/v5.9.0/SYNTHESIS_2026-11-21.md` | 62,904 | `ff131a5f44dcaf0baeb46af93056f41b5ab02142652687c058e3810d1ace6f32` | 跨轮综合（451 行；v4.4.0 → v5.7.0 共 **14 轮**）：§0 一句话总账、§1 十四轮判据结果表、§2 五个读数与三档能力、§3 已确立 15 条、§4 判不了 9 条、§5 八条测量陷阱、§6 三条路线、§8 十五条来源冲突（两侧并列**不裁决**）、§8bis 三条实测澄清；只读核对，未启引擎、未改既有文件 |
| v5.9.0 索引 | `Document/Update/Alpha/v5.9.0/README_INDEX.md` | 11,207 | `41e94aa5e8b6aa65449e50286862d3e02ee26d18110ba08c5163e8be5d468061` | v5.9.0 入口（163 行）：按目的分流 + 14 轮**证据地图**（主文件 / 字节 / sha256 前 16 位）+ 4 条离线复算命令 + 冲突速查 + 写作时刻环境快照；**写于 00:04:44**，其中 `v5.7.0/evidence/CF600/analysis.json` 仍记 63,283 B / `44c45be3…`、且称 `.tmp_v580_obs48` 只有空 `py/` —— 这两处叙述层快照已被后续发布取代（见 §15.4 的冲突第 1 条） |

### 15.1 跳过（实测缺失，未登记）

| 路径 | 实测 | 处理 |
| --- | --- | --- |
| `Document/Update/Alpha/v5.8.0/evidence/OBS48/analysis.json` | **不存在**（00:26:49 与 00:28:15 两次实测；该目录实测共 38 个文件，无此文件） | **跳过不登记**（点名时已标「若存在」）；该轮 §7 明写检验属独立分析轮，尚未跑 |
| `Document/Update/Alpha/v5.8.0/evidence/OBS48/ANALYSIS.md` | **不存在**（同上两次实测） | **跳过不登记**（点名时已标「若存在」） |

> 点名路径里 0 字节文件 **0 个** —— 没有「文件在但为空」的情况；上述 2 条全部是**目录里没有该文件**。
>
> **上一轮（§14.1）跳过的 4 条 v5.7.0 证据已全部补齐**：`evidence/CF600/{RUN.md,labels_extension.jsonl,manifest.json}` 与新增的 `analysis.json` / `ANALYSIS.md` 在 00:07:53–00:14:07 之间落盘，`SUMMARY_FOR_DECISION.md` / `README_INDEX.md` 于 00:20:19 / 00:20:30 落盘（`v5.7.0/` 现有 **42 个文件** = 顶层 4 个（`PREREGISTRATION_CF600_001.md` / `SUMMARY_FOR_DECISION.md` / `README_INDEX.md` / `verify_publish.json`）+ `evidence/CF600/` **38 个**；那 38 个里 35 个是 manifest 的 `published`，另 3 个是 `manifest.json` 自身与分析轮后加的 `analysis.json` / `ANALYSIS.md`）⇒ 本轮按点名要求**已补登**，**不计入**跳过项（因此本轮点名路径的跳过项是 2 条）。

### 15.2 存在但不在最小集（未登记，供后续按需补登）

| 路径 | 大小 (B) | 说明 |
| --- | ---: | --- |
| `Document/Update/Alpha/v5.7.0/PREREGISTRATION_CF600_001.md` | 5,147 | 冻结预注册（判据来源，**禁改**；sha256 `5bdd2a28…`，见 `SUMMARY_FOR_DECISION.md` / `README_INDEX.md` 两处同值引用） |
| `Document/Update/Alpha/v5.7.0/verify_publish.json` | 222 | v5.7.0 发布自校验（35 文件校验、`own_engine_processes = 0`） |
| `Document/Update/Alpha/v5.7.0/evidence/CF600/` 其余 **33 个文件** | 5,574,574（合计） | 5 份 checkpoint 副本（`ckpt_seed*.pt` / `ckpt_control_*.pt`，均 34,669，与本节登记的 `.tmp` 原件 **5/5 逐位同哈希**）、`eval_greedy.jsonl`（2,577,672）、`eval_stochastic.jsonl`（2,588,563）、`controls.json`、`control_eval.json`、`diagnostics.json`（27,713）、`labels_stats.json`、`labeller_selfcheck.json`、`fidelity_d3.json`、`preflight.json`、`audio_audit.json`、`collect_outcomes.json`、`merge.json`、`commands.tsv`、`exec.sh`、`train_*.json`、`eval_*_summary.json`（该目录实测共 38 个文件：本节登记 5 个 + 此处 33 个） |
| `Document/Update/Alpha/v5.8.0/PREREGISTRATION_OBS48_001.md` | 4,687 | v5.8.0 冻结预注册（判据来源，禁改；§7 的 P1/P2/P3 与 `S-partial` 定义出处） |
| `Document/Update/Alpha/v5.8.0/evidence/OBS48/` 其余 **36 个文件** | 29,324,514（合计） | `labels_obs.jsonl`（19,450,373）、3 份 `eval_*.jsonl`（2,709,006 / 2,809,737 / 2,689,872）、9 份 `eval_*_seed*.summary.json`、`train_summary.json` / `train_k{12,48}.json` / `train_k{12,48}_seed*.json`、`collect_w1{3,4,5,6}.json`、`diagnostics.json`（23,278）、`selfcheck_equiv_rows.jsonl`（1,564,379）、`fidelity_d3.json`、`preflight.json`、`audio_audit.json`、`labels_stats.json`、`merge.json`、`eval_merge.json`、`labeller_selfcheck.json`、`commands.tsv`（该目录实测共 38 个文件：本节登记 2 个 + 此处 36 个） |
| `.tmp_v570_cf600/` 与 `.tmp_v580_obs48/` 的其余中间产物 | — | 两轮的源侧副本与运行目录：`.tmp_v570_cf600/{RUN.md,diagnostics.json,eval_*.jsonl,labels_extension.jsonl,labels_stats.json,merge.json,labeller_selfcheck.json,fidelity_d3.json,preflight.json,collect_outcomes.json,control_eval.json,commands.tsv,exec.sh,status.sh,audio_audit.json,py/,train/,eval/,control_eval/,work/,logs/}`、`.tmp_v580_obs48/{RUN.md,diagnostics.json,labels_obs.jsonl,labels/,eval_*.jsonl,eval/,train/,collect/,selfcheck/,smoke/,runs/,data/,logs/,preflight.json,merge.json,train_summary.json,commands.tsv,py/}`；按最小集口径**不登记**（其中已发布的 33 + 36 个副本见上两行） |

### 15.3 不重复登记（实测已核）

- **已在册路径 0 条**：追加前 `Select-String -Path ASSET_SYNC.sha256 -Pattern 'v5\.7\.0|v5\.8\.0|v5\.9\.0|tmp_v570|tmp_v580|CF600|OBS48|round_extension|SYNTHESIS'` → **0 匹配**（`ASSET_SYNC.sha256` 的 126 行里一条都没有；`ASSET_SYNC.md` 里 `v5.7.0` 只出现在 §14.1「跳过」/ §14.2「未登记」的说明中，且全文 `round_extension` / `model.pt` 首次出现在第 606 行、即本节内 ⇒ 追加前没有任何登记行）。新增 **22 条**与既有 126 条无交集。
- **重复行 0 条**：追加后 `awk '{print $2}'` 去重 = 148、`awk '{print $1}'` 去重 = 148（`awk` 等价检查用 PowerShell `Substring(66)` / `Substring(0,64)` 的 `Sort-Object -Unique` 实测，**148 路径 / 148 哈希全唯一**）；新增 22 条自身 **22 路径 / 22 哈希全唯一**（11 个 ckpt 里 `k12` 三份同为 34,377 B、`k48` 三份同为 89,673 B、`CF600` 五份同为 34,669 B，但**哈希两两不同**，已逐对确认）。
- **行数账**：126 + 22 = **148** 行（`wc -l ASSET_SYNC.sha256` = 148；文件仍是 **LF / 无 BOM**，每行格式 `64 hex + 两个空格 + 相对路径`，格式异常行 0 条）。
- **Git 跟踪状态（实测）**：`git status --porcelain -- Document/Update/Alpha/v5.7.0 v5.8.0 v5.9.0` 三行全为 `??`（整目录 untracked），三者 `git ls-files` 计数各为 **0**；11 个 checkpoint 命中 `.gitignore:32:.tmp_*/`（`git check-ignore -v` 对 `.tmp_v570_cf600/ckpt/seed0/round_extension.pt` 与 `.tmp_v580_obs48/ckpt/k12/seed0/model.pt` 均返回 `.gitignore:32`）。即这些证据与 checkpoint 都**无法由 Git 重建**，本节 + `ASSET_SYNC.sha256` 是它们目前的字节身份记录；本轮按约束**未** `git add` / `git commit`。
- **基线（追加前，126 行，00:25:56 实测）**：`sha256sum -c` = **125 OK / 1 FAILED**，唯一 FAILED 是 §9.8 允许的 `v4.2.0/journal.jsonl`。

### 15.4 校验结论

`cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256`：**148 行 = 147 OK / 1 FAILED**（复验时刻 `2026-11-22T00:30+0800`，`sha256sum` 退出码 1 —— 仅因那 1 条已知活文件）。

- 既有 126 行贡献 **125 OK / 1 FAILED**（与 §15.3 的基线一致，未因本轮改动而变化）。
- 新登记 22 行贡献 **22/22 OK**：`tail -22 ASSET_SYNC.sha256 | sha256sum -c -` 实测 22 OK、0 FAILED（11 个 checkpoint + v5.7.0 的 7 份 + v5.8.0 的 2 份 + v5.9.0 的 2 份）。
- 唯一 FAILED 是 `Document/Update/Alpha/v4.2.0/journal.jsonl` —— §9.8 已知并**允许**的活文件漂移，本轮**未改该行**、也**未重钉**。
- **无并发写入导致的失败**：两遍实测（00:26 与 00:28:15，间隔 ≥ 100 s）22 条的字节数与 sha256 完全一致，且 `sha256sum -c` 通过 ⇒ **不需要**按「等 60 s 重算」处置；本轮没有出现「登记后被重发布」的行（对照 §12.3 里 v5.0.0 的那次并发写入）。
- **自校验 16/16**：22 条里 16 条在本轮自己的 `manifest.json` / 分析产物里能找到同值（明细见本节开头引言），另有 5 个 `.tmp` ckpt 与 `evidence/CF600/` 里的发布副本 **5/5 逐位同哈希**。

> **顺手发现的两处「来源冲突」（两侧并列，本轮未裁决、未改任何既有行）**：
>
> 1. **`v5.7.0/evidence/CF600/analysis.json` 在 v5.9.0 快照之后又被重写过**：`v5.9.0/README_INDEX.md` §3 与 `v5.9.0/SYNTHESIS_2026-11-21.md` 的来源标记表都记它为 **63,283 B / `44c45be3bf28ef78`**，而现盘实测是 **91,330 B / `e7a14776…`**（mtime 00:13:21；00:26 与 00:28 两次一致；`v5.7.0` 自己的 `ANALYSIS.md`（00:14:07）与 `SUMMARY_FOR_DECISION.md`（00:20:19）都记 91,330 B / `e7a14776…`，`v5.9.0/README_INDEX.md` §6 也自述该文件「18 分钟内被重写一次：69,335 → 63,283」）⇒ 00:04 的 v5.9.0 快照已过期，**不是本轮算错**；同理 `v5.9.0/README_INDEX.md` §6 记「`.tmp_v580_obs48` 只有一个空的 `py/` 子目录」，而 00:02:15–00:23:00 该轮已跑完、00:24:42 发布完成（现盘 6 个 `model.pt` + 38 份证据）。本节钉的是**现盘 00:26–00:28 的版本**。
> 2. **§14 表格的 `CURR/analysis.json` 一行已过期（本轮之前就存在，非本轮引入）**：`ASSET_SYNC.md` §14 记 66,109 B / `c47cb48e…`，而 `ASSET_SYNC.sha256` 第 126 行（23:58:58 重钉）记 `922dfce16d1e129f…` = 现盘 **74,723 B**，`sha256sum -c` 通过 ⇒ 与 `v5.9.0/README_INDEX.md` 的 `CURR/aj` 条目（`922dfce16d1e129f` / 74,723）一致，即 §14 的 md 表格落后于 sha256 文件。需要时单独修 §14，本轮只追加、未动既有行。

> **一句话结论**：本轮成对登记 **22 条**（11 个 `.tmp_*` checkpoint + 11 份文档），`sha256sum -c` **147 OK / 1 FAILED**，新登记 22 条 **全部 OK**；整体 FAILED 数**没有新增**，仍只有已知允许的 `v4.2.0/journal.jsonl` 那 1 条。唯一跳过项是 v5.8.0 的 `analysis.json` / `ANALYSIS.md`（两次实测均不存在，该轮独立分析尚未跑），已按 §15.1 **如实列出、未登记**。

### 15.5 补记（2026-11-22 00:3x）：v5.8.0 的独立分析产物

§15 登记时这两份尚未产出（独立分析轮后行），现补登（并已追加到 ASSET_SYNC.sha256 末尾两行）：

| 名称 | 路径 | 大小(B) | SHA-256 | 用途 |
| --- | --- | ---: | --- | --- |
| OBS48 分析（机读） | Document/Update/Alpha/v5.8.0/evidence/OBS48/analysis.json | 52,574 | 6e9afb6016dd0c07... | v5.8.0 的独立统计（P1/P2/P3 与折扣口径） |
| OBS48 分析（人读） | Document/Update/Alpha/v5.8.0/evidence/OBS48/ANALYSIS.md | 24,263 | 44d5dc581bbfa7e... | 同上，叙述版 |

复验：sha256sum -c ASSET_SYNC.sha256 = **150 行 / 149 OK / 1 FAILED**（唯一 FAILED 仍是 §9.8 允许的活文件 4.2.0/journal.jsonl）；
另把 §14 表格里 CURR/analysis.json 的过期行（66,109 B / c47cb48e...）同步更正为现盘值（74,723 B / 922dfce1...，与 .sha256 第 126 行一致）。

---

## 16. v6.0.0–v7.1.0 新资产（2026-11-22）

> **为什么有这一节**：v6.0.0（OBS-PIPE-001，把观测对照 **K=12 vs K=48** 放回 v4.5.0 的**完整 DAgger 管线**）产出 **18 个训练 checkpoint**（`k12` / `k48` × `seed{0,1,2}` × `round{0,1,2}`），全部在 gitignored 的 `.tmp_v600_obsp/` 下（`.gitignore` 的 `.tmp_*/`，**不可由 Git 重建**）；v7.0.0（CTRLFREQ-001，控制频率 **1 / 2 / 4 帧每决策**）**零训练**，不产出 checkpoint。两轮各自发布机器可读证据，并各写一份 `SUMMARY_FOR_DECISION.md` 与 `README_INDEX.md`。按 AGENTS.md §8.2「新增/移动/删除任何资产，必须同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`」，本节与 `ASSET_SYNC.sha256` **末尾追加的 17 行**成对登记。
>
> **登记口径（最小集）**：只收 **6 个被判据直接使用的 `round2.pt`**（`k{12,48}/seed{0,1,2}/round2.pt`，即评测载入的那 6 个；`round0` / `round1` 的 12 个中间 ckpt **不登记**）+ 点名要求的 **11 份文档**（v6.0.0 的 `evidence/OBSP/{RUN.md,analysis.json,ANALYSIS.md,manifest.json}` 4 份 + `SUMMARY_FOR_DECISION.md`、`README_INDEX.md` 2 份；v7.0.0 的 `evidence/CF/{RUN.md,diagnostics.json,manifest.json}` 3 份 + `SUMMARY_FOR_DECISION.md`、`README_INDEX.md` 2 份）。**不**收两轮的原始 `labels_*.jsonl` / `eval_*.jsonl`、`*.py` / `*.sh`、`commands.tsv`、`preflight.json`、`fidelity_d3.json`、`audio_audit.json`、`ckpt_assertions.json`、`ab_compare.json`、`train_*.json`、`labels_stats.json`、`merge.json`、两份 `PREREGISTRATION_*`（禁改件，只引其 sha256），也不收 §16.2 列出的那些文件。
>
> **`.tmp_v710_adapt/`：实测不存在 ⇒ 0 个 checkpoint 可登**。`Document/Update/Alpha/v7.1.0/` 实测只有 `PREREGISTRATION_ADAPT_001.md`（4,374 B，mtime 01:19:32），`evidence/` 与任何 `.tmp_v71*` 目录**都不存在** ⇒ 按「存在几个登几个」的口径登记 **0 个**（见 §16.1）。
>
> **快照时刻**：两轮证据在各自生产轮结束后未变（v6.0.0 的 `evidence/OBSP/` 除分析产物外 mtime 全为 01:06:41/01:06:43，v7.0.0 的 `evidence/CF/` 生产产物 mtime ≤ 01:18:28）。**例外是 v6.0.0 的两份分析产物**：`analysis.json` / `ANALYSIS.md` 在 **01:24:39** 首版、**01:27:02** 被分析轮重写（78,386 → **78,700** B；30,244 → **30,602** B），本节钉的是 **01:27:02 之后连续 ≥193 s 无变化**（01:27:02 → 01:30:15 两次实测字节与 sha256 全同）的那一版；四份新文档（下表第 **11–12、16–17** 行）按**最终成稿**的字节钉。所有值都是 `Get-FileHash -Algorithm SHA256` 与 WSL `sha256sum` **实测**，未手抄、也未取用任何 `manifest.json` 的内联值代替实测。
>
> **自校验（17 条里 8 条能在该轮自己的产物里找到同值）**：6 个 `.pt` ↔ `v6.0.0/evidence/OBSP/manifest.json` 的 `checkpoints`（**6/6**，含 `bytes` 与 `kept_in` = `.tmp_v600_obsp`，即资产不复制进 `Document/`）；其中 `k12` 的 3 个 `round2.pt` 与 v4.5.0 登记值 **3/3 逐位相同**（`OBSP/ANALYSIS.md` §8 的 9 对表），`k48` 的 3 个与 `OBSP/RUN.md` §7 列出的 sha256 **3/3 相符**；`OBSP/RUN.md` ↔ `OBSP/manifest.json` 的 `files`（**1/1**）；`CF/RUN.md` 与 `CF/diagnostics.json` ↔ `CF/manifest.json` 的 `files`（**2/2**）。其余 9 条（`OBSP/{analysis.json,ANALYSIS.md,manifest.json}`、`CF/manifest.json`、4 份新文档）属**分析侧 / 叙述侧 / 清单自身**：v6 的两份分析产物是分析轮后加的（manifest 发布在 01:06:43，早于 01:24:39），manifest 本就不记录自身、也不记录分析轮另写的文件 ⇒ 只能以本节实测值为准。
>
> **`.tmp_*` 行沿用既有写法**（相对路径 `.tmp_...`），不引入 `/mnt/d/...` 形式。

| 名称 | 路径 | 大小 (B) | SHA-256 | 用途一句话 |
| --- | --- | ---: | --- | --- |
| OBS-PIPE k12 seed0 | `.tmp_v600_obsp/ckpt/k12/seed0/round2.pt` | 34,387 | `4abee069d087872e0a2d726cc975850d760e90a705ddcbee60ea91268e5e61dc` | v6.0.0 评测载入的 **K=12 对照**（106 维）种子 0 `round2` checkpoint；与 `.tmp_v450_cf/ckpt/seed0/round2.pt` **逐位相同**（即 v4.5.0 `S` 的同一字节），池化 stochastic **484/3,600 = 13.4444%** ⇒ P3 的载体 |
| OBS-PIPE k12 seed1 | `.tmp_v600_obsp/ckpt/k12/seed1/round2.pt` | 34,387 | `a93746bb5d123165e77e0271edfc1efcdb21931e9e2d706e5a22d613c5476ca4` | 同上（K=12，种子 1；= v4.5.0 `S` seed1，逐位相同） |
| OBS-PIPE k12 seed2 | `.tmp_v600_obsp/ckpt/k12/seed2/round2.pt` | 34,387 | `c69960fcbbd888e9b252d3fa685c22a6730f8dd637dcfc9edab45491d218fa01` | 同上（K=12，种子 2；= v4.5.0 `S` seed2，逐位相同） |
| OBS-PIPE k48 seed0 | `.tmp_v600_obsp/ckpt/k48/seed0/round2.pt` | 89,683 | `0115160bc259ea063481e7171cf665f4cf294f65ee2e022ad2507c417cce9066` | v6.0.0 的 **K=48 观测臂**（近 48 弹 × 6 = 288 特征 / **322 维**）种子 0 `round2` 产物：与 K=12 臂**同一批 5,166 行**训练（数据被完全控制）；池化 **426/3,600 = 11.8333%** ⇒ P1 不成立 |
| OBS-PIPE k48 seed1 | `.tmp_v600_obsp/ckpt/k48/seed1/round2.pt` | 89,683 | `96725704797c43b901167847c52bd2da22d357f664ed68986616427bdc280f5c` | 同上（K=48，种子 1） |
| OBS-PIPE k48 seed2 | `.tmp_v600_obsp/ckpt/k48/seed2/round2.pt` | 89,683 | `f7cf927cfa8af83162ca6314513f9721f051b848d2253d6b7d8cf94f63b9e85c` | 同上（K=48，种子 2） |
| OBSP 执行记录 | `Document/Update/Alpha/v6.0.0/evidence/OBSP/RUN.md` | 37,442 | `c57bc1c3626a176026d9fe5ff32afd7a68192180b7a2605df0f525bdd11864c1` | v6.0.0 执行记录（650 行；24 步，其中 **1 次 rc=2**（pin 拒绝）后 resume，其余全 0；D3 保真 **5/5**、前缀 2,165、标签器 **18/18** @H=600、K=12 特征等价 **240/240**、音频 **muted 85 / audible 0**、自有引擎进程 **0**；§8 十条偏差 + §9 七条异常全列）—— **不做统计推断** |
| OBSP 统计（机读） | `Document/Update/Alpha/v6.0.0/evidence/OBSP/analysis.json` | 78,700 | `3531f4a2d893c263ba4d4fe7e82bfc4ce39b773e8a74ceffbd4eb054db5805c0` | v6.0.0 独立分析层（schema `v600-obsp-analysis/1`；离线、**0 次引擎**；除了 `created_utc` 逐字节确定）：**P1 不成立**（−1.6111 pp；预注册方向单侧 Fisher **p = 0.9818**；K=48 CP95 **[10.7956%, 12.9334%]**；MOVER **[−3.1660, −0.0567] pp**）、**P2 不成立**（格级 Fisher 双侧 p = 1.0000）、**P3 成立**（eval 行 3,600/3,600 + 标签 5,166/5,166 + ckpt 9/9）⇒ **`S-fail`**；折扣 ICC **0.1530/0.1026**、deff **46.75/31.68**、n_eff **77.0/113.6**，差值 deff **14.22** / n_eff **253.1** |
| OBSP 分析（人读） | `Document/Update/Alpha/v6.0.0/evidence/OBSP/ANALYSIS.md` | 30,602 | `aff3f0ae9178186268fd36c587010bea1d804b7d25e81090b092ed54f7740e16` | v6.0.0 独立分析正文（286 行 / 12 节）：L1–L4 四层保真、return 区间分解（`[29,60)` −223 条 / 中段 +281 / `==300` −58）、12 格 vs 4 格精确置换的分辨率上限（4 单位最小可达 p = 0.0625）、12 对 ckpt 的 sha256 全表、`RUN.md` **364 项**数值声明 **364 一致 / 0 不符** + 结构性闸门缺陷 D1 |
| OBSP 清单 | `Document/Update/Alpha/v6.0.0/evidence/OBSP/manifest.json` | 22,483 | `7e3fbdc08fbac854ad3b49a4101ff3d377d2acbfcfc49fe064f2ad9f7f0817f2` | v6.0.0 发布清单（`created_at` `2026-11-22T01:06:40+0800`）：`files` **57 条**内联 sha256、`missing = []`、`checkpoints` **18 条**（`k{12,48}` × `seed0–2` × `round0–2`，含 `bytes` 与 `kept_in`）、`pins_at_publish.all_match = false`（两个 pin 在生产结束后被并发 v7.0.0 改写）；分析轮复核 **57/57** 相符 |
| v6.0.0 决策摘要 | `Document/Update/Alpha/v6.0.0/SUMMARY_FOR_DECISION.md` | 43,721 | `d77a9130d0a8c65bf59dcc5940572fcd1b8bae06be6e46ba7c4bc7c19cbb6ac9` | v6.0.0 的「10 分钟读完」决策摘要（六节：§0 一句话（含九件事）→ §1 改了什么 → §2 判据表（P1/P2/P3 + 描述性 + return 区间分解 + 可加式）→ §3 有效性诊断（ICC/deff/n_eff + 三层读数）→ §4 并排 → §5 能/不能声称 → §6 合规、开放项与九条冲突） |
| v6.0.0 索引 | `Document/Update/Alpha/v6.0.0/README_INDEX.md` | 20,767 | `ce8fc19920c94da21b875722ae66f9d3a1a3923a99bea63cbc3e232a5b67c27a` | v6.0.0 接手索引（七节）：按目的分流（含 `analysis.json` / `ANALYSIS.md`）+ 三十秒版本 + 证据一览（60 文件分组）+ 判据与停止条件 + 五个开放项 + 与相邻版本的关系 + 路径与复现约定（含 `.tmp_analysis_60/run_analysis.py` 复跑） |
| CTRLFREQ 执行记录 | `Document/Update/Alpha/v7.0.0/evidence/CF/RUN.md` | 30,102 | `05e423e1642d81b4c896b55ece60cdd99e727f1a77c30d3602be30d6ee7ebce9` | v7.0.0 执行记录（466 行；**17/17 步 rc = 0**、执行窗口 01:02–01:20、整段 eval ≈9 min（9 单元并发）；D3 **5/5**、S-ab **60/60 + 240/240**、ckpt **3/3** + 收尾复检、音频 **57/57 muted / 0 audible**、`frames=2` 与 v4.5.0 冻结证据 **3,600/3,600 行逐条相同**；§9 九条偏差全列）—— **不做统计推断** |
| CTRLFREQ 诊断 | `Document/Update/Alpha/v7.0.0/evidence/CF/diagnostics.json` | 62,104 | `2ed0580f30a916aa3de0f63124058ba1aeee538663f0a719b4ad9d96ef0f6026` | v7.0.0 原始诊断（`pooled_by_level` / `risk` / `per_seed_by_level` / `per_window_by_level` / `survival_frames_definition` / `risk_definition` / `inputs`）：三档 10,800 条的分布、KM 风险（每决策 0.0037318 / 0.00666636 / 0.01757218；每帧 0.0037318 / 0.00333875 / 0.00442229）、两种存活口径差值；`problems = []`、**无检验** |
| CTRLFREQ 清单 | `Document/Update/Alpha/v7.0.0/evidence/CF/manifest.json` | 10,237 | `b25617a27ac25732344b5afb6899883137df3a278929cfad9820900f304aa6bd` | v7.0.0 发布清单（`created_at` `2026-11-22T01:18:28+0800`）：`file_count = 36`、三道门（D3 / S-ab / ckpt）、`checkpoints` **3 条**（v4.5.0 冻结 `S` 的 `round2.pt`）、`counts`、`no_statistics`（DESCRIPTIVE ONLY）、`problems = []`、`ok = true` |
| v7.0.0 决策摘要 | `Document/Update/Alpha/v7.0.0/SUMMARY_FOR_DECISION.md` | 32,820 | `4d35e2c7256833425a8fa30bcbaab9c92c5c5725bbbc42b4ae0c576a89040d50` | v7.0.0 的「10 分钟读完」决策摘要（六节）：1 帧 **382/3,600 = 10.6111%**（均值 196.89）/ 2 帧 **484/3,600 = 13.4444%**（216.57）/ 4 帧 **252/3,600 = 7.0000%**（203.94）⇒ 2 帧在均值与尾部都最优 ⇒ **P1/P2 不成立、控制频率不是瓶颈、该维度关闭**；并写明**预注册缺「基线自身最优」这一分支**（`S-null` 的 `|差| < 5 帧` 不满足）+ 八条文档间冲突 |
| v7.0.0 索引 | `Document/Update/Alpha/v7.0.0/README_INDEX.md` | 16,354 | `c415793e5fb221377b1e9d1530881dc77776531d4c12d45cdb02bd938b294ec3` | v7.0.0 接手索引（七节）：按目的分流（含 A/B、ckpt 断言、历史复现）+ 三十秒版本 + 证据一览 + 判据与停止条件（含「无分支字面命中」的边界）+ 五个开放项 + 与相邻版本的关系 + 路径与复现约定（含「不要写显著 / 不要写触发 `S-null`」的禁令） |

### 16.1 跳过（实测缺失 / 无产出，未登记）

| 路径 | 实测 | 处理 |
| --- | --- | --- |
| `.tmp_v710_adapt/` 新 checkpoint | **目录不存在**（`Test-Path` 为假；`Document/Update/Alpha/v7.1.0/` 实测只有 `PREREGISTRATION_ADAPT_001.md` 4,374 B） | **跳过不登记**；v7.1.0 尚未产出任何 ckpt，按「存在几个登几个」= **0 个** |

> 本节点名路径里 **0 字节文件 0 个** —— 没有「文件在但为空」的情况；唯一跳过项是**目录整个不存在**（不是空目录）。
>
> **原本不存在、后来补齐的 2 项**：v6.0.0 的 `evidence/OBSP/{analysis.json,ANALYSIS.md}` 在本次登记开始时**尚未产出**（首查时该目录 58 个文件），01:24:39 落盘、01:27:02 被重写，01:30:15 复测稳定 ⇒ **已补登到上表**（第 8–9 行），**不计入**跳过项。

### 16.2 存在但不在最小集（未登记，供后续按需补登）

| 路径 | 大小 (B) | 说明 |
| --- | ---: | --- |
| `Document/Update/Alpha/v6.0.0/PREREGISTRATION_OBS_PIPE_001.md` | 5,484 | v6.0.0 冻结预注册（判据来源，**禁改**；sha256 `47360e7d…`，文首含 00:5x 勘误） |
| `Document/Update/Alpha/v7.0.0/PREREGISTRATION_CTRLFREQ_001.md` | 4,532 | v7.0.0 冻结预注册（判据来源，**禁改**；sha256 `dcf215fd…`） |
| `Document/Update/Alpha/v7.1.0/PREREGISTRATION_ADAPT_001.md` | 4,374 | v7.1.0 冻结预注册（跨窗口自适应；该目录**唯一的文件**，尚无证据、无 ckpt） |
| `Document/Update/Alpha/v6.0.0/evidence/OBSP/` 其余 **56 个文件** | — | `labels_pipe.jsonl`（41,374,900，5,166 行）、4 份 `eval_*.jsonl`（2,685,345–2,702,567）、12 份 `eval_*.summary.json`、18 份 `train_*.json`、9 份 `collect_*.json`、`labels_stats.json`（16,382）、`merge.json`（6,488）、`eval_merge.json`、`diagnostics.json`（40,341）、`preflight.json`（4,103）、`fidelity_d3.json`（26,490）、`labeller_selfcheck.json`（17,676）、`selfcheck_equiv_rows.jsonl`（1,569,419）、`audio_audit.json`（2,906）、`train_summary.json`（12,884）（该目录实测 **60 个文件**：本节登记 4 个 + 此处 56 个） |
| `Document/Update/Alpha/v7.0.0/evidence/CF/` 其余 **24 个文件** | — | 9 份 `eval_frames{1,2,4}_seed{0,1,2}.jsonl`（各 860,757–862,518）、`eval_summaries.json`（43,713）、`eval_frames2_vs_v450_evidence.json`（2,816）、`ab_compare.json`（84,065）、5 份 `ab_*.jsonl`、`ckpt_assertions.json`（5,569）、`fidelity_d3.json`（26,495）、`audio_audit.json`（6,288）、`commands.tsv`（7,510）、3 份 `runner*.diff`、`MAIN_AGENT_VERIFICATION.md`（2,950，01:20:25，**不在 manifest 的 `files` 里**）（该目录实测 **27 个文件**：本节登记 3 个 + 此处 24 个；另有 `scripts/` 子目录 **11 个**：`assertions.py` / `ab_compare.py` / `ab_vs_v450_evidence.py` / `collect_facts.py` / `diagnostics.py` / `exec.sh` / `make_diff.sh` / `merge_hist.py` / `publish.py` / `tables.py` / `verify_publish.py`） |
| `.tmp_v600_obsp/` 的其余中间产物 | — | `labels_pipe.jsonl`、`labels/`、`eval_*.jsonl`、`eval/`、`train/`、`collect/`、`selfcheck/`、`d3/`、`data/`（85 个私有 `th08.cfg` 所在）、`logs/`、`py/`、`timeline.tsv`、`seed*/` 运行脚本；按最小集**不登记**（其中 6 个 `round2.pt` 已登，12 个 `round0/round1.pt` 未登） |
| `.tmp_v700_cf/` 的全部内容 | — | `ab/`、`data/`、`eval/`、`logs/`、`publish/`、`py/`、`smoke/`、`work/`、`tables.md`、`facts.json`、`hist_frames2_seed{0,1,2}.json`、`ab_compare.json` 等（该轮零训练，**无 `ckpt/`**）；按最小集**不登记** |

### 16.3 不重复登记（实测已核）

- **已在册路径 0 条**：追加前 `Select-String -Path ASSET_SYNC.sha256 -Pattern 'v6\.0\.0|v7\.0\.0|tmp_v600_obsp|tmp_v710'` → **0 匹配**（150 行里一条都没有；全文 `SUMMARY_FOR_DECISION` / `README_INDEX` 的既有条目只属 v5.7.0 / v5.9.0）。新增 **17 条**与既有 150 条无交集。
- **重复行 0 条**：新增 17 条自身 **17 路径 / 17 哈希全唯一**（6 个 ckpt 里 `k12` 三份同为 34,387 B、`k48` 三份同为 89,683 B，但**哈希两两不同**，已逐对确认）。
- **行数账**：150 + 17 = **167** 行（`wc -l ASSET_SYNC.sha256` = 167；文件仍是 **LF / 无 BOM**，每行格式 `64 hex + 两个空格 + 相对路径`，格式异常行 0 条）。
- **Git 跟踪状态（实测）**：`git status --porcelain -- Document/Update/Alpha/v6.0.0 v7.0.0 v7.1.0` 三行全为 `??`（整目录 untracked）；6 个 `.pt` 命中 `.gitignore` 的 `.tmp_*/` 规则 ⇒ 这些证据与 checkpoint 都**无法由 Git 重建**，本节 + `ASSET_SYNC.sha256` 是它们目前的字节身份记录；本轮按约束**未** `git add` / `git commit`。
- **基线（追加前，150 行，§15.5 末尾实测）**：`sha256sum -c` = **149 OK / 1 FAILED**，唯一 FAILED 是 §9.8 允许的 `v4.2.0/journal.jsonl`。

### 16.4 校验结论

`cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256`：**167 行 = 166 OK / 1 FAILED**（复验时刻 `2026-11-22T01:32+0800`；`sha256sum` 退出码 1 —— 仅因那 1 条已知活文件）。**Windows 侧独立复核**（Python `hashlib` 逐文件重算）得到同一组结论：163 OK / 1 FAILED / 3 个「文件不可见」；那 3 个是文件里本就在 WSL 侧登记的引擎路径（`/root/Touhou_RL/third_party/th08-rl/build/rl_{v3,laser,fork}/th08-modern`），在 WSL 内校验为 OK ⇒ **以 WSL 的 166 OK / 1 FAILED 为准**。

- 既有 150 行贡献 **149 OK / 1 FAILED**（与 §16.3 的基线一致，未因本轮改动而变化）。
- 新登记 17 行贡献 **17/17 OK**：`tail -17 ASSET_SYNC.sha256 | sha256sum -c -` 实测 **17 OK、0 FAILED**（6 个 ckpt + v6.0.0 的 6 份 + v7.0.0 的 5 份）。
- 唯一 FAILED 是 `Document/Update/Alpha/v4.2.0/journal.jsonl` —— §9.8 已知并**允许**的活文件漂移，本轮**未改该行**、也**未重钉**。
- **并发写入已处置并记录**：v6.0.0 的 `analysis.json` / `ANALYSIS.md` 在首次钉后被分析轮重写（01:27:02），导致最初两行 FAILED（78,386/30,244 → 78,700/30,602 B）；**已按 01:27:02 之后连续 ≥193 s 无变化的那一版重钉**，重钉后 `sha256sum -c` 通过。**首版的哈希已不保留**（避免留下无法校验的行）。
- **无并发写入的残余失败**：重钉后到最终复验之间，166 行全部 OK；本节与四份新文档的字节在钉后未再变。

> **一句话结论**：本轮成对登记 **17 条**（6 个 `.tmp_*` checkpoint + 11 份文档），`sha256sum -c` **166 OK / 1 FAILED**，新登记 17 条**全部 OK**；整体 FAILED 数**没有新增**，仍只有已知允许的 `v4.2.0/journal.jsonl` 那 1 条。唯一跳过项是 `.tmp_v710_adapt/`（**目录不存在 ⇒ 0 个 ckpt**），已按 §16.1 **如实列出、未登记**；v6.0.0 的两份分析产物在登记期间**先缺后补**（01:24:39 落盘、01:27:02 重写）已按 §16.1 的补记**补登**，不计入跳过项。

---

## 17. v7.1.0 新资产（2026-11-22）

> **为什么有这一节**：v7.1.0（ADAPT-001，在目标窗口上做 **200 条无标签在线自适应**）产出 **12 个自适应 checkpoint**（`seed{0,1,2}` × `w{11,12,17,18}`，共 412,284 B），全部在 gitignored 的 `.tmp_v710_adapt/` 下（`.gitignore:32:.tmp_*/`，**不可由 Git 重建**）；该轮另发布机器可读证据，并新写 `SUMMARY_FOR_DECISION.md` 与 `README_INDEX.md`。按 AGENTS.md §8.2「新增/移动/删除任何资产，必须同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`」，本节与 `ASSET_SYNC.sha256` **末尾追加的 18 行**成对登记。**§16 当时记录的「`.tmp_v710_adapt/` 目录不存在 ⇒ 0 个 ckpt」已被本轮取代**：该目录现已存在，12 个 ckpt 全部登记。
>
> **登记口径（最小集）**：只收 **12 个评测载入的自适应 ckpt** + 点名要求的 **4 份证据**（`evidence/ADAPT/{RUN.md,diagnostics.json,control_checks.json,manifest.json}`）+ **2 份新文档**（`SUMMARY_FOR_DECISION.md`、`README_INDEX.md`）。**不**收该轮的 15 份原始 `*.jsonl`（`eval_zero_*` / `adapt_*` / `eval_post_*` / `repro_zero_*` / `control_postfrozen_*`，合计 6,699,478 B）、9 份 `*.summary.json`（102,514 B）、7 份其余机读件（`audio_audit.json` / `ckpt_assertions.json` / `ckpt_manifest.json` / `commands.tsv` / `fidelity_d3.json` / `process_residue.json` / `selftest_update.json`，合计 53,744 B）、`logs/` 24 个（43,617 B）、`scripts/` 4 个（20,018 B），也不收 `PREREGISTRATION_ADAPT_001.md`（**禁改件**，只引其 sha256 `ba536306…`）。
>
> **快照时刻**：18 条取于 `2026-11-22T08:2x+0800`，Windows 侧 `Python hashlib` 与 WSL 侧 `sha256sum` 两侧独立实测一致（未手抄、未取用任何 `manifest.json` 的内联值代替实测）。该轮生产产物的 mtime 全在 01:23–01:38 之间（`control_checks.json` 01:34:36、`diagnostics.json` 01:34:41、`ckpt_manifest.json` 01:35:07、`manifest.json` 01:38:07、`RUN.md` 未变、12 个 `.pt` 同批），**自 01:38 起至本快照无变化**；两份新文档是本轮写作时刻的最终成稿。
>
> **自校验（18 条里 15 条能在该轮自己的产物里找到同值）**：12 个 `.pt` ↔ `evidence/ADAPT/ckpt_manifest.json` 的 `files`（**12/12**，含 `bytes = 34,357` 与 `total_bytes = 412,284`）；`RUN.md` / `diagnostics.json` / `control_checks.json` ↔ `evidence/ADAPT/manifest.json` 的 `files`（**3/3**，含 `bytes`）。其余 3 条（`evidence/ADAPT/manifest.json` 自身与 2 份新文档）属**清单自身 / 叙述侧**：`manifest.json` 的 `excludes = "manifest.json"` 本就不记录自身，两份新文档写于发布之后 ⇒ 以本节实测值为准。
>
> **`.tmp_*` 行沿用既有写法**（相对路径 `.tmp_...`），不引入 `/mnt/d/...` 形式。

| 名称 | 路径 | 大小 (B) | SHA-256 | 用途一句话 |
| --- | --- | ---: | --- | --- |
| ADAPT 自适应 ckpt seed0 w11 | `.tmp_v710_adapt/ckpt/seed0/w11.pt` | 34,357 | `301449b9ec7a685ed9b5584447f1a3d8adee5b29cb4518f096f892ae9673be6f` | seed0 × 窗口 11 的 **200 次在线更新**产物（起点 `4abee069…` = v4.5.0 `S` seed0；`ckpt_chain.adapt_in == zero_ckpt`）；该格零样本 **3/100** → 自适应后 **0/100** |
| ADAPT 自适应 ckpt seed0 w12 | `.tmp_v710_adapt/ckpt/seed0/w12.pt` | 34,357 | `001a174da521af3545dcb4917b4ffb8e2f50c380eefd180a2054012b5fa7b7d6` | 同上（seed0 × 窗口 12）；该格零样本 **26/100** → 自适应后 **0/100** |
| ADAPT 自适应 ckpt seed0 w17 | `.tmp_v710_adapt/ckpt/seed0/w17.pt` | 34,357 | `d6171110eb663cbda68acd42ddbdbc3dedcc6e40763c15b0b0c83d03093cbf74` | 同上（seed0 × 窗口 17）；该格零样本 **36/100** → 自适应后 **0/100** |
| ADAPT 自适应 ckpt seed0 w18 | `.tmp_v710_adapt/ckpt/seed0/w18.pt` | 34,357 | `68913b302cf5b9aeb64f685c5bf81fc1e47f46572e09fe9414fdf51c327b4e33` | 同上（seed0 × 窗口 18）；该格零样本 3/100（均值 53.47）→ 自适应后 **0/100**，但均值升到 **203.27 / 中位 224 / `max = 232 < 300`**（「能活很久但走不完」的极端样本） |
| ADAPT 自适应 ckpt seed1 w11 | `.tmp_v710_adapt/ckpt/seed1/w11.pt` | 34,357 | `3cb624a95929624093189a19735a6e1c63c0862505f7cc8fc621dba211ed9c5f` | seed1 × 窗口 11 的 200 次更新产物（起点 `a93746bb…`）；该格零样本 **3/100** → 自适应后 **0/100** |
| ADAPT 自适应 ckpt seed1 w12 | `.tmp_v710_adapt/ckpt/seed1/w12.pt` | 34,357 | `adbc5fff44537ea9a19a06373338cef108639d2fe9ea9ecf9503e887dff35582` | 同上（seed1 × 窗口 12）；该格零样本 **8/100** → 自适应后 **0/100**；该格块 3 的熵低到 **0.08 nats**（全轮最低的块均值） |
| ADAPT 自适应 ckpt seed1 w17 | `.tmp_v710_adapt/ckpt/seed1/w17.pt` | 34,357 | `ce5dbaeb25f19ce2a1779f0dd61b5a13d41d077f8c381c625b609a6f5b4e9621` | 同上（seed1 × 窗口 17）；该格零样本 **24/100** → 自适应后 **12/100** —— **12 格里唯一还有成功条数的格** |
| ADAPT 自适应 ckpt seed1 w18 | `.tmp_v710_adapt/ckpt/seed1/w18.pt` | 34,357 | `4e2ecdd4233445989dcd516ccc087e7326fde7a11e841c8fef07e97b01bf7d1a` | 同上（seed1 × 窗口 18）；该格零样本 **2/100** → 自适应后 **0/100** |
| ADAPT 自适应 ckpt seed2 w11 | `.tmp_v710_adapt/ckpt/seed2/w11.pt` | 34,357 | `4a8fead027d14ceb0b02d7c85ecfa6be3d35620b636276503b3f4f3885b2943e` | seed2 × 窗口 11 的 200 次更新产物（起点 `c69960fc…`）；该格零样本 **2/100** → 自适应后 **0/100** |
| ADAPT 自适应 ckpt seed2 w12 | `.tmp_v710_adapt/ckpt/seed2/w12.pt` | 34,357 | `1e77810c5dc392ff426ed2b7b7895c686c0156e00ff0ca2181cf0cc8800677b3` | 同上（seed2 × 窗口 12）；该格零样本 **4/100** → 自适应后 **0/100** |
| ADAPT 自适应 ckpt seed2 w17 | `.tmp_v710_adapt/ckpt/seed2/w17.pt` | 34,357 | `e05dc285ffbd989616e4ba008b00dfdf1f2529654c538b16ca2e3a3b8d5bc282` | 同上（seed2 × 窗口 17）；该格零样本 **19/100** → 自适应后 **0/100** |
| ADAPT 自适应 ckpt seed2 w18 | `.tmp_v710_adapt/ckpt/seed2/w18.pt` | 34,357 | `d56df8aab4b67967dd0af27bbbcaaecbc367348c07fc801ac75bb681f0025936` | 同上（seed2 × 窗口 18）；该格零样本 **3/100** → 自适应后 **0/100** |
| ADAPT 执行记录 | `Document/Update/Alpha/v7.1.0/evidence/ADAPT/RUN.md` | 26,024 | `0787d74cb8b23d16bf93e795c6cad6249f5fe83bac9de32fdbeeb43c01c0bfbb` | v7.1.0 执行记录（332 行；**13 个编号步骤 / 31 条 argv 记录全 exit = 0**，运行 01:23:56–01:34:58；D3 保真 **5/5**、ckpt 断言 **3/3** + 收尾复检、音频 **61 个 cfg / 61 muted / 0 audible**、自有引擎进程 **0**、61 次前缀走查 0 mismatch；§11 **六条偏差**：runner 轮内被改两次、publish 一度覆盖零样本文件、两个对照不属预注册三段等）—— **不做统计推断** |
| ADAPT 诊断 | `Document/Update/Alpha/v7.1.0/evidence/ADAPT/diagnostics.json` | 126,789 | `ad28d2763236468991adcf0ecd4b6b701f6f58d538996d428b9eb17df38bc0ee` | v7.1.0 原始诊断（`pooled` / `weak_windows` / `strong_windows` / `per_cell` **12 格** / `adapt_curves` / `raw_counts_side_by_side` / `integrity`）：零样本 **133/1,200 = 11.0833%**（均值 105.0808）→ 自适应后 **12/1,200 = 1.00%**（均值 79.1292），`descriptive_difference = −121 条 / −10.0833 pp`；弱窗口 2.67% → 0.00%、强窗口 19.50% → 2.00%；自适应段存活 106.91 → 72.23 → 80.44 → 76.44、熵 2.06 → 1.10 nats；`episode_indices_all_disjoint = true`、`ckpt_chain_all_ok = true`；`analysis_scope` 明写 **无 p 值 / 无检验统计量 / 无显著性声明**；`deviations = []` |
| ADAPT 对照汇总 | `Document/Update/Alpha/v7.1.0/evidence/ADAPT/control_checks.json` | 6,045 | `6c0901bded179535858eb72b6731312f7f03894125d98d459edcebab248e0846` | v7.1.0 的两个对照（**都不计入任何池化数字**，`note` 已写明）：①同参重跑零样本 **1,200/1,200 逐字段一致**（3 × 400，`episodes_differing = 0`、`first_differences = []`、7 个字段）⇒ 评测确定性；②**冻结 ckpt ＋ 同一后段协议** **162/1,200 = 13.50%**（18.25% / 11.75% / 10.50%）⇒ 协议 / shape / 段位 / 流都被排除；`errors = []`、`all_reproducible = true`、`clean = true` |
| ADAPT 发布清单 | `Document/Update/Alpha/v7.1.0/evidence/ADAPT/manifest.json` | 8,477 | `70fd3e31faeb2d74754ba77a63ff4033ab87a37cd253a4843bf30fc2d00515a0` | v7.1.0 发布清单（`created_at` `2026-11-22T01:38:07+0800`）：`file_count = 62`、`total_bytes = 7,078,229`、`excludes = "manifest.json"`、`files` 62 条内联 sha256（该目录实测 **63 个文件** = 清单 62 ＋ 自身）；**没有 `no_statistics` 字段**（统计边界引 `diagnostics.json::analysis_scope` 或 `RUN.md` §8/§12） |
| v7.1.0 决策摘要 | `Document/Update/Alpha/v7.1.0/SUMMARY_FOR_DECISION.md` | 41,229 | `8d06de09f9b91d4d41fec9596973bf196e9c9d98d382db2d74b10d271d9621a8` | v7.1.0 的「10 分钟读完」决策摘要（六节）：§0 一句话（含十件事：11.08% → **1.00%**、**−121 条 / −10.08 pp**、**12/12 格为负 0 格为正**、弱 **2.67% → 0.00%**、强 **19.50% → 2.00%**、熵 **2.06 → 1.10 nats**、两个对照、**`S-degrade` 同型 v4.7.0 的 0.0278%**、**本轮不做 P1/P2/P3**、**11.08% 不是 13.4444% 的复现**、**「自适应后」不是零样本泛化**）→ §1 改了什么 → §2 判据表 → §3 有效性诊断（含段位效应 2.42 pp）→ §4 并排 → §5 能/不能声称（11 条）→ §6 合规、六个开放项与**八条文档间冲突** |
| v7.1.0 索引 | `Document/Update/Alpha/v7.1.0/README_INDEX.md` | 20,247 | `18e1780c6a2bde1624fc52858463d26579cdea37d6fc7fb3976df98789451388` | v7.1.0 接手索引（七节）：按目的分流（含两个对照、自适应曲线、算子等价自检）+ 三十秒版本 + 证据一览（63 文件分组）+ 判据与停止条件（**`S-degrade` 是阈值规则，不是检验**）+ 六个开放项 + 与相邻版本的关系 + 路径与复现约定（含「不要写显著 / 不要写 P1/P2/P3 不成立 / 不要把 13.50% 说成复现」的禁令） |

### 17.1 跳过（实测缺失 / 无产出，未登记）

| 路径 | 实测 | 处理 |
| --- | --- | --- |
| — | **无** | 点名路径 **18 条全部存在且字节非 0** ⇒ 本次跳过 **0 条**；也没有「目录整个不存在」的情况（§16.1 记的「`.tmp_v710_adapt/` 不存在」已被本轮推翻） |

> 本节点名路径里 **0 字节文件 0 个**、缺失 **0 个**。`Document/Update/Alpha/v7.1.0/` 实测 **66 个文件** = `PREREGISTRATION_ADAPT_001.md`（4,374 B，**禁改件**）＋ `SUMMARY_FOR_DECISION.md` ＋ `README_INDEX.md` ＋ `evidence/ADAPT/` **63 个**（其中 4 个已登、59 个见 §17.2）。

### 17.2 存在但不在最小集（未登记，供后续按需补登）

| 路径 | 大小 (B) | 说明 |
| --- | ---: | --- |
| `Document/Update/Alpha/v7.1.0/PREREGISTRATION_ADAPT_001.md` | 4,374 | v7.1.0 冻结预注册（判据来源，**禁改**；sha256 `ba53630630f5ca0248fea5e4d3ac7ce8874ec7e7c1e368eafd440bdc96d3a2fd`）；`P1` 符号检验 + 精确置换 / `P2` Fisher vs 13.4444% / `P3` 弱窗口 Fisher；`S-fork` / **`S-degrade`** / `S-null` / `S-ok` / `S-mixed` |
| `Document/Update/Alpha/v7.1.0/evidence/ADAPT/` 其余 **15 份原始 `*.jsonl`** | 6,699,478（合计） | `eval_zero_seed{0,1,2}.jsonl`（各 400 行）、`adapt_seed{0,1,2}.jsonl`（各 816 行 = 800 episode + 16 个 50 条块）、`eval_post_seed{0,1,2}.jsonl`（各 400 行）、`repro_zero_seed{0,1,2}.jsonl`（各 400 行）、`control_postfrozen_seed{0,1,2}.jsonl`（各 400 行）；P1/P2/P3 与两个对照的**唯一**数据源，本轮只登记 sha256 的持有者（`RUN.md` §10.2 已逐份列出） |
| `Document/Update/Alpha/v7.1.0/evidence/ADAPT/` 其余 **9 份 `*.summary.json`** | 102,514（合计） | `eval_zero_*` / `adapt_*` / `eval_post_*` 各 3 份（17,648–17,670 B 与 7,757–7,759 B 与 8,750–8,752 B）；含 `launch_env`（`SDL_AUDIODRIVER=dummy`、`DISPLAY = null`）与逐格 gate 子断言 |
| `Document/Update/Alpha/v7.1.0/evidence/ADAPT/` 其余 **7 份机读件** | 53,744（合计） | `audio_audit.json`（6,663，`cfg_files = 61 / muted = 61 / audible = 0`）、`ckpt_assertions.json`（5,687，start/final 3/3 + 4 个受保护文件 start = final）、`ckpt_manifest.json`（1,745，12 个 ckpt 的 sha256 与 `total_bytes = 412,284`）、`commands.tsv`（11,192，**32 行 = 表头 + 31 条 argv**）、`fidelity_d3.json`（26,509，5/5）、`process_residue.json`（501，`matching = 0`）、`selftest_update.json`（1,447，`max_abs_diff = 0.0`、bitwise 相同） |
| `Document/Update/Alpha/v7.1.0/evidence/ADAPT/logs/` **24 个** | 43,617（合计） | 每个进程的完整 stdout（`zero-s*` / `adapt-s*` / `post-s*` / `repro-s*` / `control-s*` / `d3-fidelity` / `audio-audit` / `process-audit` / `diagnostics` / `manifest` / `assert-{start,final}` / `selftest` / `control-report`） |
| `Document/Update/Alpha/v7.1.0/evidence/ADAPT/scripts/` **4 个** | 20,018（合计） | `exec.sh`（12 个目标：`preflight`…`publish`，是 §13 的复跑入口）、`fact_check.py`、`make_tables.py`、`verify_publish.py` |
| `.tmp_v710_adapt/` 的其余中间产物 | — | `work/data/v710-{zero,adapt,post,repro,control,fidelity}-s*/`（**61 个私有 `th08.cfg`** 所在，内含指向 `/root/v400-data` 的 `th08.dat`/`thbgm.dat` 软链，Windows 侧不可遍历）、`zero/`、`adapt/`、`post/`、`repro/`、`control/`、`logs/`、`py/`、`publish_verification.json`；按最小集**不登记**（其中 `ckpt/` 的 12 个 `.pt` 已登） |

### 17.3 不重复登记（实测已核）

- **已在册路径 0 条**：追加前对前 167 行做 `v7\.1\.0|tmp_v710|ADAPT_001|ADAPT/` 匹配 → **0 匹配**（§16.2 只在正文提到 `v7.1.0/PREREGISTRATION_ADAPT_001.md` 与「`.tmp_v710_adapt/` 不存在」，**没有任何登记行**）。新增 **18 条**与既有 167 条无交集。
- **路径/哈希交集 0**：新增 18 条的 18 个路径、18 个 sha256 与前 167 行**逐一比对，均无命中**；新增 18 条自身 **18 路径 / 18 哈希全唯一**（12 个 ckpt 同为 34,357 B，但**哈希两两不同**，已逐对确认）。
- **全文件里存在的 3 组同哈希是既有的、已记录的一对**（不是本轮引入）：`.tmp_v450_cf/ckpt/seed{0,1,2}/round2.pt`（第 67–69 行）与 `.tmp_v600_obsp/ckpt/k12/seed{0,1,2}/round2.pt`（第 151–153 行）**逐位相同**（§16 已写明「与 v4.5.0 `S` 的同一字节」）⇒ 全文件 185 路径 / 182 唯一哈希，差异完全由这 3 对解释。
- **行数账**：167 + 18 = **185** 行（文件仍是 **LF / 无 BOM**，末行以 `\n` 结束，每行格式 `64 hex + 两个空格 + 相对路径`，格式异常行 **0** 条；文件从 19,102 B 增至 **21,030 B**）。
- **Git 跟踪状态（实测）**：`git status --porcelain -- Document/Update/Alpha/v7.1.0 ASSET_SYNC.md ASSET_SYNC.sha256` → `?? Document/Update/Alpha/v7.1.0/`（整目录 untracked）、`M ASSET_SYNC.md`、`M ASSET_SYNC.sha256`；`git ls-files -- Document/Update/Alpha/v7.1.0` = **0**；`git check-ignore -v .tmp_v710_adapt/ckpt/seed0/w11.pt` → `.gitignore:32:.tmp_*/` ⇒ 这些证据与 checkpoint 都**无法由 Git 重建**，本节 + `ASSET_SYNC.sha256` 是它们目前的字节身份记录；本轮按约束**未** `git add` / `git commit`。
- **基线（追加前，167 行，§16.4 记 166 OK / 1 FAILED @ 01:32）**：本轮复验时已漂移，见 §17.4。

### 17.4 校验结论

`cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256`（WSL，复验时刻 `2026-11-22T08:2x+0800`）：**185 行 = 183 OK / 2 FAILED**（`sha256sum` 退出码 1 —— 仅因那 2 条）。

- 既有 167 行贡献 **165 OK / 2 FAILED**；新登记 18 行贡献 **18/18 OK**：`tail -18 ASSET_SYNC.sha256 | sha256sum -c -` 实测 **18 OK、0 FAILED**（12 个 ckpt + 4 份证据 + 2 份新文档）。
- **FAILED 1（已知并允许）**：`Document/Update/Alpha/v4.2.0/journal.jsonl` —— §9.8 记载的活文件漂移，本轮**未改该行**、也**未重钉**。
- **FAILED 2（本轮新发现的既有漂移，不是本轮造成）**：`Document/Update/Alpha/v5.9.0/SYNTHESIS_2026-11-21.md` —— 登记值 `ff131a5f44dcaf0baeb46af93056f41b5ab02142652687c058e3810d1ace6f32`，现盘 **95,648 B / `80c7cb85959fb964d789e0973d0431bd6e8e25580533ed53af15185c40fa4cf7`**，mtime **2026-11-22 01:46:47**（晚于 §16.4 的复验时刻 01:32，当时该行通过、整体 166 OK / 1 FAILED）⇒ **该文件在 §16 复验之后被外部重写**，既不是本轮算错，也不是本轮改动（本轮只改 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`，未触 `v5.9.0/**`）。按「**只追加、未动既有行**」的约束，本轮**不重钉**；需要时单独修这一行（重钉会立刻把整体拉回 **184 OK / 1 FAILED**）。
- **两侧一致**：Windows 侧独立复验（Python `hashlib` 逐文件重算）对 **18 条新登记全部通过**，与 WSL 的 `18/18 OK` 相符；其余行本轮未逐一重算，以 WSL 的 183 OK / 2 FAILED 为准。
- **无并发写入造成的本轮失败**：18 条在登记前的实测哈希与登记值完全一致，登记后 `tail -18 | sha256sum -c -` 一次通过 ⇒ 本轮没有出现「登记后被重发布」的行。

> **一句话结论**：本轮成对登记 **18 条**（12 个 `.tmp_v710_adapt/` checkpoint + 4 份 v7.1.0 证据 + 2 份新文档），`sha256sum -c` **183 OK / 2 FAILED**；**新登记 18 条全部 OK**。整体 FAILED 数从 §16 的 1 条变为 2 条，**但新增的那 1 条不是本轮造成、也不属 §9.8 允许**：`v5.9.0/SYNTHESIS_2026-11-21.md` 在 01:46:47（§16 复验之后）被外部重写，本轮按「只追加」约束**未重钉**，已在 §17.4 如实列出。**跳过项 0 条**：§16.1 记的「`.tmp_v710_adapt/` 目录不存在 ⇒ 0 个 ckpt」已被本轮推翻，12 个 ckpt 全部登记。

---

## 18. v8.0.0–v8.1.0 新资产（2026-11-22）

> **为什么有这一节**：v8.0.0（BOMB-001，把「放雷」从被禁用的动作变成**外部脚本规则**，四臂 A0/A1/A2/A3）与 v8.1.0（BOMBFAR-001，把预算推到 **600 / 900 / 1,200 决策**）都是**零训练纯评测**轮次 ⇒ **不产出任何 checkpoint**；但它们各自发布了机器可读证据、**各自在轮后又补了一份独立离线分析**（`analysis.json` + `ANALYSIS.md`，`scope.engine_started = false`、0 次引擎），并各写一份 `SUMMARY_FOR_DECISION.md` 与 `README_INDEX.md`。按 AGENTS.md §8.2「新增/移动/删除任何资产，必须同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`」，本节与 `ASSET_SYNC.sha256` **末尾追加的 12 行**成对登记。
>
> **登记口径（最小集）**：只收点名要求的 **12 份文档** —— v8.0.0 的 `evidence/BOMB/{RUN.md,manifest.json,analysis.json,ANALYSIS.md}` 4 份 + `SUMMARY_FOR_DECISION.md`、`README_INDEX.md` 2 份；v8.1.0 的 `evidence/BOMBFAR/{RUN.md,manifest.json,analysis.json,ANALYSIS.md}` 4 份 + `SUMMARY_FOR_DECISION.md`、`README_INDEX.md` 2 份。**不**收两轮的原始 `eval_*.jsonl`（v8.0.0 12 份 / v8.1.0 13 份）、`*.summary.json`（12 / 13 份）、其余机读件（`bomb_audit.json` / `diagnostics.json` / `equiv_check.json` / `repro_v800.json` / `fidelity_d3.json` / `ckpt_assertions*.json` / `audio_audit.json` / `process_residue.json` / `commands.tsv` / `default_path_equivalence.json`）、`logs/`（22 / 23 个）、`scripts/`（各 7 个）、v8.1.0 的两份 runner 字节快照（`v800_bomb.used.py` / `v800_bomb.pre_v810_patch.py`），也不收两份 `PREREGISTRATION_*`（**禁改件**，只引其 sha256：`8629d73c…` / `b2609a35…`）。
>
> **两轮都零 checkpoint**：`.tmp_v800_bomb/` 与 `.tmp_v810_bomb/` 下**没有 `ckpt/` 目录**（纯评测、无训练）⇒ 本次登记 **0 个 `.pt`**。工作目录（含 v8.0.0 的 **52 个**、v8.1.0 的 **57 个**私有 `th08.cfg`）按 AGENTS.md §8.1 留在本机、不入 Git。
>
> **快照时刻**：12 条取于 `2026-11-22T10:0x+0800`，Windows 侧 `Python hashlib` 与 WSL 侧 `sha256sum` 两侧独立实测一致（未手抄、未取用任何 `manifest.json` 的内联值代替实测）。两轮的**生产产物** mtime 全在 08:20–09:36 之间；**4 份分析产物是后补的**（v8.0.0 首版 09:49:35、定稿 **10:03:28**；v8.1.0 首版 09:54:49、定稿 **10:03:33**），登记期间实测**被反复重写**（`analysis.json`：v8.0.0 76,294 → 78,196 → 80,736 → 80,731 → 81,197 → **81,317** B；v8.1.0 81,122 → 81,156 → **83,818** B。`ANALYSIS.md`：v8.0.0 17,932 → **0（重写中）** → 18,925 → 19,027 → 18,990 → 19,281 → **19,383** B；v8.1.0 **0** → 21,752 → 21,767 → 23,290 → **23,455** B）。本节钉的是**各自最后一版连续 ≥80 s 字节不变**的那一版；**0 字节的中间态一律不登记**，早期版本的 sha256 **不保留**（避免留下无法校验的行）。4 份新文档按**最终成稿**的字节钉。
>
> **自校验（12 条里 2 条能在该轮自己的产物里找到同值）**：两份 `RUN.md` ↔ 各自 `manifest.json` 的 `files`（**2/2**，含 `bytes`）；`manifest.json` 本身 2 条按 `excludes = "manifest.json"` **本就不记录自身**；4 份分析产物**晚于 manifest 发布**（v8.0.0 晚 83 min、v8.1.0 晚 27 min）⇒ **不可能**在清单里（v8.1.0 的分析轮自己在 `manifest_crosscheck.note` 写明「deliberately not in the manifest」）；4 份新文档写于两轮之后。**另一层（验证 manifest 本身可信、不是验证登记行）**：登记时对 v8.0.0 清单的 **9 个关键文件**、v8.1.0 清单的 **13 个关键文件**逐条重算 sha256，**22/22 与清单相符**。
>
> **`.tmp_*` 行**：本次 **0 条**（两轮零 checkpoint）；文中的 `.tmp_v800_bomb/` / `.tmp_v810_bomb/` / `.tmp_analysis_80/` 只在叙述里出现，不是登记行。

| 名称 | 路径 | 大小 (B) | SHA-256 | 用途一句话 |
| --- | --- | ---: | --- | --- |
| BOMB 执行记录 | `Document/Update/Alpha/v8.0.0/evidence/BOMB/RUN.md` | 21,059 | `01deca4164142b9bebaaf5563e6f9a93b626c84c0633bf2a07f3735c524d908d` | v8.0.0 执行记录（337 行；4 臂 × 3 种子 × 4 窗 × 300 = **14,400 episode**、`commands.tsv` **34 条 argv（31 条 exit 0）**、运行窗口 08:20:23–08:37:55；D3 保真 **5/5**、ckpt **3/3**、A0 循环等价 **4/4**、音频 **52 muted / 0 audible**、自有引擎进程 **0**；§6 **七条偏差**全列）—— **本文件不含统计推断**（那一步由分析轮补） |
| BOMB 发布清单 | `Document/Update/Alpha/v8.0.0/evidence/BOMB/manifest.json` | 8,452 | `e9d9eb5014053a796754cd255cc3f96ed09865801e446e2ba8ef9c328d9ac9a9` | v8.0.0 发布清单（`created_at` `2026-11-22T08:40:10+0800`）：`file_count = 62`、`total_bytes = 24,380,174`、`excludes = "manifest.json"`；该目录登记时实测 **65 个文件 / 24,489,326 B**（= 清单 62 ＋ `manifest.json` 自身 8,452 ＋ 2 份后补的分析产物） |
| BOMB 统计（机读） | `Document/Update/Alpha/v8.0.0/evidence/BOMB/analysis.json` | 81,317 | `f14ade7308231d15457248e4adf91c89a14ab4651bc7c2d54a18528c6c76d1a2` | v8.0.0 独立离线分析（schema `v800-bomb-analysis/1`、`created_utc = 2026-11-22T02:03:28Z`、`scope.engine_started = false`）：**P1 成立**（三个阈值的单侧 Fisher `log10 p = −1521.5834` ⇒ `p_sci = 2.609474e-1522`（double 下溢）、**Holm 后 7.828422e-1522**、CP95 不重叠）、**P2 成立**（置换 20,000、`count_ge = 0`、`p = 4.999750e-05` = 分辨率下限 `1/20,001`、Holm 后 1.499925e-4）、A0 vs v4.5.0 冻结评测 **3,600/3,600 × 17 字段 0 差异**、`caliber_checks`（决策 0 首雷 100%、决策 0 最近子弹 max **36.378 px**、臂间 Fisher p = 1.0 = 没有信息）、`stop_conditions` ⇒ **`S-ok`** |
| BOMB 分析（人读） | `Document/Update/Alpha/v8.0.0/evidence/BOMB/ANALYSIS.md` | 19,383 | `347f995e2e94020527f75b0078130065e67e279e2e66799517f66dc078114c1a` | v8.0.0 独立分析正文（238 行 / 11 节）：§0 结论摘要 → §1 主表 + **§1.0 行级自检（14 项 × 4 臂）** + P1 + P2 → §2 判定 → §3 P3（含「臂间差 ≠ 逐 episode 增量」的口径限制）→ §4 管线保真（**自己重算，不引 `equiv_check` / `diagnostics.a0_vs_v450`**）→ §5 决策 0 → §6 天花板饱和（**不是「阈值无效」**）→ §7 停止条件 → §8 与 `RUN.md` 逐条对数（9 条全 ✅、`mismatches = []`）→ §9 不能声称 → §10 文件与可复现性 |
| BOMBFAR 执行记录 | `Document/Update/Alpha/v8.1.0/evidence/BOMBFAR/RUN.md` | 19,610 | `e683b88aca9930d5b19124e403f731df9a56ff367020d89c67257e5fb93f8c04` | v8.1.0 执行记录（250 行；三档 **10,800** + A0 **1,200** + 控制臂 **3,600** = **15,600 episode**、`commands.tsv` **22 条 argv / 22 条 exit 0**、运行窗口 08:39:34–09:34:02；D3 **5/5**、ckpt start/final **各 3/3**、默认路径等价 **3/3 × 44 字段**、音频 **57 muted / 0 audible**、自有引擎进程 **0**（并发回合的 73 个在过滤器外）；§7 **十条偏差**全列，含「600 档未复现 100% 系预注册口径」与「runner 被并发回合改过三次」）—— **本文件不含统计推断** |
| BOMBFAR 发布清单 | `Document/Update/Alpha/v8.1.0/evidence/BOMBFAR/manifest.json` | 9,643 | `e1c72bddc7c737ad48f8725863beeecdd07c78f6454006ff067d7995800ed5c7` | v8.1.0 发布清单（`created_at` `2026-11-22T09:36:37+0800`）：`file_count = 69`、`total_bytes = 33,701,296`、`excludes = "manifest.json"`；该目录登记时实测 **72 个文件 / 33,818,212 B**（= 清单 69 ＋ 自身 ＋ 2 份后补的分析产物）；分析轮另对 **39 个顶层文件**逐个核对，`mismatch_count = 0`、`missing_count = 0` |
| BOMBFAR 统计（机读） | `Document/Update/Alpha/v8.1.0/evidence/BOMBFAR/analysis.json` | 83,818 | `19bf20122ded4552dca0d0489ac098cef40e1a70aca2850fca3676627794f9e5` | v8.1.0 独立离线分析（schema `v810-bombfar-analysis/1`、`created_utc = 2026-11-22T02:03:33Z`、`scope.engine_started = false`）：三档 **71.1111% / 13.3333% / 1.0000%** 与 A0 **0.0000%**、**P1 成立**（Fisher `p_sci = 3.865606e-575`、CP95 [69.5999%, 72.5879%]）、**`a0_survival_curve`**（活到 600/900/1,200/1,800/2,400 帧 = **235 / 67 / 2 / 0 / 0**，即 **19.5833% / 5.5833% / 0.1667% / 0% / 0%**）与 `tiers_vs_horizon_matched_A0`（同口径 Fisher 4.467089e-495 / 1.848037e-64 / 3.041551e-5）、`reproduction`（**3,600/3,600 × 15 字段、双向核对**）、`cross_budget_rng`（600 档 1,040 条有效行里仅 **6** 条同死亡帧）、`stop_conditions` ⇒ **`S-drop` + `S-floor` 触发** |
| BOMBFAR 分析（人读） | `Document/Update/Alpha/v8.1.0/evidence/BOMBFAR/ANALYSIS.md` | 23,455 | `71d34e736bf34d10a104644b1c809a12a744ed225005befa3263a91fe937d808` | v8.1.0 独立分析正文（269 行 / 8 节）：§0 结论摘要 → §1 三档主表 + 逐种子（**极差 73.17 pp**）+ 逐窗口 + 行级完整性 → §2 P1（**同口径警告** + A0 存活曲线）→ §3 复现检查（**双向核对** + 「为什么必须用 300 决策控制臂」的 RNG 论证）→ §4 机制量化（首/末雷位置、耗尽）→ §5 停止条件 → §6 与 `RUN.md` 对数（`all_claims_match = true`、`mismatches = []`）→ §7 不能声称 → §8 可复现性 |
| v8.0.0 决策摘要 | `Document/Update/Alpha/v8.0.0/SUMMARY_FOR_DECISION.md` | 44,023 | `84cc5488c5fef2cad3699068d3d056f99d16fe57988541891def654a8842c00f` | v8.0.0 的「10 分钟读完」决策摘要（六节）：§0 一句话（七件事：**决策 0 首雷** / **每局 3 颗** / **按键后第 2 帧扣雷** / **天花板饱和不是「阈值无效」** / **P1 成立** / **P2 成立** / **`S-ok`**）→ §1 改了什么（含分析层）→ §2 判据表（Fisher+Holm / 置换 / P3 / 停止条件）→ §3 三重保真 + 天花板 + all-bullets vs lethal-only → §4 并排 → §5 能/不能声称（11 条）→ §6 合规、七个开放项与**九条文档间冲突** |
| v8.0.0 索引 | `Document/Update/Alpha/v8.0.0/README_INDEX.md` | 25,370 | `11560af466e741b5b2f37c72be03465b3ae44c2fd13cd7ac2fd2fcc30e36204c` | v8.0.0 接手索引（七节）：按目的分流（含**分析产物**与复跑统计入口）+ 三十秒版本 + 证据一览（**65 文件**分组）+ 判据与停止条件（`S-ok` 已触发）+ 七个开放项（含「分析产物被反复重写」）+ 与相邻版本的关系 + 路径与复现约定（含「不要写阈值无效 / 不要说一颗雷值 383 帧」的禁令） |
| v8.1.0 决策摘要 | `Document/Update/Alpha/v8.1.0/SUMMARY_FOR_DECISION.md` | 45,045 | `e3f1f174f4c1805ef96bd40552d82b075180b47b04c70fa9b9d0de6f4cfbb4c3` | v8.1.0 的「10 分钟读完」决策摘要（六节）：§0 一句话（四件事：**600 档不是 100% 的原因=预注册口径** / **三颗雷 15 秒烧光** / **P1 成立 + `S-drop`/`S-floor`** / **同口径 A0 存活曲线**）→ §1 改了什么（含 runner 字节身份）→ §2 判据表（P1/P2/P3 + 停止条件）→ §3 H1–H4 归因 + 分母差异 → §4 并排 → §5 能/不能声称（11 条）→ §6 合规、七个开放项与**十一条文档间冲突** |
| v8.1.0 索引 | `Document/Update/Alpha/v8.1.0/README_INDEX.md` | 28,444 | `3f7f6a1564f959f16f301888958ad0c8dea879b5d0f2d8e41a0b33f36c5feeef` | v8.1.0 接手索引（七节）：按目的分流（含分析产物、同口径基线、跨预算 RNG 论证、runner 字节身份）+ 三十秒版本 + 证据一览（**72 文件**分组）+ 判据与停止条件 + 七个开放项 + 与相邻版本的关系 + 路径与复现约定（含「不要写 600 档复现了 100% / 不要把 71.1111% 与 13.4444% 当同口径」的禁令） |

### 18.1 跳过（实测缺失 / 无产出，未登记）

| 路径 | 实测 | 处理 |
| --- | --- | --- |
| — | **无** | 点名路径 **12 条全部存在且字节非 0** ⇒ 本次跳过 **0 条**；也没有「目录整个不存在」的情况 |

> **两轮都没有 checkpoint 可登**：`.tmp_v800_bomb/`（`arms/`、`work/data/` 52 个 cfg、`equiv_check.json`、`logs/`、`py/`）与 `.tmp_v810_bomb/`（`arms/`、`work/data/` 57 个 cfg、`regress/`、`logs/`、`py/`）**都没有 `ckpt/` 子目录** —— 这是「纯评测轮」的正常结果，不是缺失。
>
> **登记开始时 4 份分析产物确实不存在**：首查（约 09:40）时 `v8.0.0/evidence/BOMB/` 是 63 个文件、`v8.1.0/evidence/BOMBFAR/` 是 70 个文件，**两个目录里都没有 `analysis.json` / `ANALYSIS.md`**。它们在登记过程中先后落盘（09:49 / 09:54）并被反复重写，**已按定稿版补登到上表**（第 3–4、7–8 行），**不计入**跳过项（与 §16.1 记的 v6.0.0 情形同型）。

### 18.2 存在但不在最小集（未登记，供后续按需补登）

| 路径 | 大小 (B) | 说明 |
| --- | ---: | --- |
| `Document/Update/Alpha/v8.0.0/PREREGISTRATION_BOMB_001.md` | 4,456 | v8.0.0 冻结预注册（判据来源，**禁改**；sha256 `8629d73cf06d472833f438e54914e3b56c8c98dcbdb1d2eb99eb9b2cf20260ea`，80 行）；P1（三阈值单侧 Fisher + **Holm**）/ P2（**置换 20,000**）/ P3（描述）；`S-fork` / `S-nohelp` / `S-ok` / `S-harm` |
| `Document/Update/Alpha/v8.1.0/PREREGISTRATION_BOMBFAR_001.md` | 4,314 | v8.1.0 冻结预注册（判据来源，**禁改**；sha256 `b2609a35f21c325f3db3eb2d2da663a5e9ac5cab522f99d810486d92fec5f88e`，85 行）；P1（三档活满率 + 基线比较，**计数比较、无检验**）/ P2 / P3（描述）；`S-fork` / `S-100` / **`S-drop`** / **`S-floor`**；§0 把 v8.0.0 的 300 决策 / 600 帧写成「600 帧」（**已由两轮 RUN.md 与两份 ANALYSIS 如实披露，预注册未改**） |
| `Document/Update/Alpha/v8.0.0/evidence/BOMB/` 其余 **61 个文件** | 24,359,115（合计） | 12 份 `eval_A{0,1,2,3}_seed{0,1,2}.jsonl`（**23,632,760**，各 1,200 行）、12 份 `*.summary.json`（**82,320**）、8 份顶层机读件（**588,963**：`bomb_audit.json` 378,209 / `diagnostics.json` 148,597 / `equiv_check.json` 15,138 / `fidelity_d3.json` 26,500 / `ckpt_assertions.json` 2,675 / `audio_audit.json` 5,986 / `process_residue.json` 486 / `commands.tsv` 11,372）、`logs/` **22 个**（**37,476**）、`scripts/` **7 个**（**17,596**）。该目录实测 **65 个文件**：本表登记 2 个 + §18 上表登记 4 个 + 此处 61 个 |
| `Document/Update/Alpha/v8.1.0/evidence/BOMBFAR/` 其余 **68 个文件** | 33,681,686（合计） | 13 份 `eval_*.jsonl`（**32,870,838**：A3 三档 × 3 种子 + A0×1200 单种子 + 控制臂 3 份）、13 份 `*.summary.json`（**90,550**）、10 份顶层机读件（**483,824**：`diagnostics.json` 35,005 / `repro_v800.json` 391,100 / `bomb_audit.json` 9,159 / `fidelity_d3.json` 26,509 / `ckpt_assertions.json` 2,675 / `ckpt_assertions_final.json` 2,675 / `audio_audit.json` 6,888 / `process_residue.json` 487 / `commands.tsv` 8,415 / `default_path_equivalence.json` 911）、2 份 runner 字节（**155,539**：`v800_bomb.used.py` 77,827 = **本轮实际运行的字节**、`v800_bomb.pre_v810_patch.py` 77,712 = v8.0.0 的 runner）、`logs/` **23 个**（**39,327**）、`scripts/` **7 个**（**41,608**）。该目录实测 **72 个文件**：本表登记 2 个 + 上表登记 4 个 + 此处 68 个 |
| `.tmp_v800_bomb/`、`.tmp_v810_bomb/`、`.tmp_analysis_80/` 的全部内容 | — | `work/data/`（**52 / 57 个私有 `th08.cfg`** 所在）、`arms/`、`regress/`、`logs/`、`py/`；分析目录 `analyze_v800.py` / `analyze_v810.py` / `run_all.py` / `stats_core.py`（`analysis.json` / `ANALYSIS.md` 的**生成器**，未随证据发布）。按最小集**不登记**，且两轮**零 checkpoint** |

### 18.3 不重复登记（实测已核）

- **已在册路径 0 条**：追加前对前 **185 行**做 `v8\.0\.0|v8\.1\.0|BOMB|BOMBFAR` 匹配 → **0 匹配**；新增 **12 条**与既有 185 条无交集。
- **路径/哈希交集 0**：新增 12 条的 12 个路径、12 个 sha256 与前 185 行**逐一比对，均无命中**；新增 12 条自身 **12 路径 / 12 哈希全唯一**。全文件现为 **197 路径 / 194 唯一哈希**（本次**未**引入同哈希对；那 3 个重复值仍是 §16/§17 已记录的 `.tmp_v450_cf/ckpt/seed{0,1,2}/round2.pt` ↔ `.tmp_v600_obsp/ckpt/k12/seed{0,1,2}/round2.pt` 三对**逐位相同**，与 185 行时「182 唯一值」同源）。
- **行数账**：185 + 12 = **197** 行（文件仍是 **LF / 无 BOM**，末行以 `\n` 结束，每行格式 `64 hex + 两个空格 + 相对路径`，格式异常行 **0** 条；文件从 **21,030 B** 增至 **22,468 B**）。
- **Git 跟踪状态（实测）**：`git status --porcelain -- Document/Update/Alpha/v8.0.0 Document/Update/Alpha/v8.1.0 ASSET_SYNC.md ASSET_SYNC.sha256` → 两个版本目录均为 `??`（整目录 untracked）、`M ASSET_SYNC.md`、`M ASSET_SYNC.sha256`；`git ls-files -- Document/Update/Alpha/v8.0.0 Document/Update/Alpha/v8.1.0` = **0** ⇒ 这些证据**无法由 Git 重建**，本节 + `ASSET_SYNC.sha256` 是它们目前的字节身份记录；本轮按约束**未** `git add` / `git commit`。

### 18.4 校验结论

`cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256`（WSL，复验时刻 `2026-11-22T10:1x+0800`）：**197 行 = 196 OK / 1 FAILED**（`sha256sum` 退出码 1 —— 仅因那 1 条）。

- **基线（追加前，185 行）**：同一命令实测 **184 OK / 1 FAILED** —— 唯一 FAILED 是 `Document/Update/Alpha/v4.2.0/journal.jsonl`（§9.8 已知并**允许**的活文件漂移）。**注意与 §17.4 的差异**：§17.4 当时记的是 **183 OK / 2 FAILED**（第二条是 `v5.9.0/SYNTHESIS_2026-11-21.md` 被外部重写）；本次复验时那条**已经通过**（该文件现盘字节与登记值相符）⇒ 本轮**未改任何既有行**，只是基线自己回到了 184/1。
- **新登记 12 行贡献 12/12 OK**：`tail -n 12 ASSET_SYNC.sha256 | sha256sum -c` 实测 **12 OK、0 FAILED**（v8.0.0 的 4 份 + v8.1.0 的 4 份 + 4 份新文档）。
- **4 份分析产物是唯一有「登记期漂移」风险的行**：它们在登记窗口内被重写过多次（见本节开头的字节史）。本节采取的做法与 §16.4 记的 v6.0.0 情形一致 —— **等它们在连续 ≥80 s 内字节不变后钉版**，钉完再跑 `sha256sum -c`；**实测 12/12 OK ⇒ 钉版之后没有再被改写**。若未来分析轮再重写，这 4 行会 FAILED，需要单独重钉（**不要**用 `manifest.json` 的内联值代替实测）。
- **两侧一致**：Windows 侧独立复验（Python `hashlib` 逐文件重算）对 **12 条新登记全部通过**，与 WSL 的 12/12 OK 相符；其余行本轮未逐一重算，以 WSL 的 196 OK / 1 FAILED 为准。
- **唯一 FAILED** 是 `Document/Update/Alpha/v4.2.0/journal.jsonl` —— §9.8 已知并**允许**的活文件漂移，本轮**未改该行**、也**未重钉**。

> **一句话结论**：本轮成对登记 **12 条**（8 份证据 + 4 份新文档；两轮**零 checkpoint**），`sha256sum -c` **196 OK / 1 FAILED**，**新登记 12 条全部 OK**；整体 FAILED 数**没有新增**，仍只有已知允许的 `v4.2.0/journal.jsonl` 那 1 条。跳过项 **0 条**：点名 12 条全部存在；登记开始时缺失的 4 份分析产物在过程中落盘并被反复重写，已按**连续 ≥80 s 字节不变的定稿版**补登，早期版本与 0 字节中间态的 sha256 **未保留**。

---

## 19. R37/R38（.tmp_v431_seg001）资产登记（2026-11-22）

> **为什么有这一节**：`Document/Update/Alpha/v4.4.0/TRAINING_HOST_REPLY_01.md`（§6「缺项」、§八 8.1、§八 8.3 的实测行）指出 `ASSET_SYNC.md` / `ASSET_SYNC.sha256` **从未登记 `.tmp_v431_seg001/**`**，其中包含远端工单点名的 **R38 冻结权重** `armA-multi4-f8664-128.pt`——该文件此前的唯一身份记录只存在于那份回复文档的正文里，**校验表里没有任何一行**。按 AGENTS.md §8.2「新增/移动/删除任何资产，必须同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`」，本节与 `ASSET_SYNC.sha256` **末尾追加的 12 行**成对登记。
>
> **登记口径（最小集）**：只收 **12 条** —— **2 个 `.pt` 权重** ＋ **8 份 A 臂评估 JSONL** ＋ **2 份本轮新生成的 manifest 证据**（`Document/Update/Alpha/v4.4.0/evidence/E0/cfair_manifest.json`、`.../armA_manifest.json`）。**不**收：`.tmp_v431_seg001/` 顶层约 400 个 `patch_r*.py` / `fix_*.py` / `*_docs*.py` 补丁与探针脚本及 `*.txt`、`full/raw/armA/` 的其余 **10 份** JSONL（`train-*.jsonl` 5 份、`eval-seed11-f8664.jsonl`、`fixcheck-seed13-f8664.jsonl`、`smoke-seed13-f8664.jsonl` 等）、`full/raw/c4/c4.jsonl`、`full/raw/episodes.jsonl`、`full/raw/episode-*.json`（3 份）、`full/raw/state_000.json`、`full/states_frozen.json`、`full/logs/`、`full/run/`、`full/student_traces/`、`full/data/`（含私有 `th08.cfg`）、`rerun_snapshot/`。
>
> **`full/raw/cfair/**` 的 80 个文件不逐文件登记**，改用**manifest 里的 per-seed `combined_sha256`** 代表 —— 全部理由见 §19.1；逐文件的 `{name, bytes, sha256}` 仍完整保存在 `evidence/E0/cfair_manifest.json` 里，**没有丢任何一层的可校验性**。
>
> **只读实测（12 条没有一条取自内联值）**：本节 12 条 sha256 全部由 **Python 3.12.10 `hashlib`（1 MiB 分块流式）** 于 `2026-11-22T10:5x+0800` 在本机重算；随后用 **PowerShell `Get-FileHash`** 对 **90 个源文件**（80 cfair ＋ 8 JSONL ＋ 2 `.pt`）与 **4 个 per-seed `combined_sha256`** 独立复算一遍，**90/90 ＋ 4/4 全部相符**；两个 manifest 的 `rows` / `full_segments` / `early_deaths` 也逐行重数 **8/8 相符**。**未**取用任何 `manifest.json` / 日志 / 回复文档正文里的内联值代替实测。
>
> **`.tmp_v431_seg001/**` 全程只读、未修改**：登记窗口内实测该目录**最新一次写入仍是 `2026-11-21T12:40:20`**（`chk5.txt` / `chk5e.txt`），比本轮早约 **22 小时** ⇒ 本轮只有读取，**0 新增 / 0 改名 / 0 删除**。2 份新 manifest 写在**我们自己的** `Document/Update/Alpha/v4.4.0/evidence/E0/` 下，**没有**写进 `.tmp_v431_seg001/`。

| 名称 | 路径 | 大小 (B) | SHA-256 | 用途一句话 |
| --- | --- | ---: | --- | --- |
| R38 冻结权重（A 臂，128 决策档） | `.tmp_v431_seg001/full/raw/armA/armA-multi4-f8664-128.pt` | 34,729 | `1a43d18e2b3fe1e5dd645716219d2a79e811d6608286fc340fe08e8dd64bab61` | **远端工单点名的 R38 权重**（mtime `2026-11-16T02:54:36`）：`run_arm_a_multi4.sh` 先训 80 → 续训到 128 决策档；`run_eval_ext.sh` 用它跑 4 dev 窗口 × 10 episode、`--policy-seed 100`。`strict=True` 加载 exit 0、`<All keys matched successfully>`、`input_dim 106`、输出 `[1, 18]`、`finite True`（**加载结论出自 `TRAINING_HOST_REPLY_01.md` §三**，本节只负责字节身份）。与 `-80.pt` **不是同一文件**（34,729 vs 34,719 B，sha 不同）⇒ 续训确实改动了权重 |
| A 臂权重（80 决策档） | `.tmp_v431_seg001/full/raw/armA/armA-multi4-f8664-80.pt` | 34,719 | `bb5bc005728555ae0900408b82d62c230ff38bce1fadb8850c7ed0373d737afc` | 上一档中间产物（mtime `2026-11-16T01:38:47`），是 `-128.pt` 的**续训起点**。登记它的理由：**没有它就无法判定 `-128.pt` 是否真被续训改过**——实测 34,719 → 34,729 B、sha 全不同 ⇒ 改过；两者一起入册才能固定「80→128」这条链的字节身份 |
| A seed11 评估（128 决策） | `.tmp_v431_seg001/full/raw/armA/eval-seed11-armA-multi4-f8664-128.jsonl` | 1,265 | `296d707d7b08920bc335d7e50784f087fb9d87da01c9ae8be3331dd839b1cdf0` | seed11 的 A 臂评估（10 episode；mtime `2026-11-16T03:07:07`）：`rows=10`、`full_segments=1`（`return_decisions ≥ 300`）、`early_deaths=9`。**含 `TRAINING_HOST_REPLY_01.md` §4.1 的满段**（第 8 行 `episode=7 rd=300.0 died_at_offset=null survived=true`）与 **§4.2 的早死**（第 7 行 `episode=6 rd=43.0 offset=87`）。键集合固定 7 个，**无** `frames_executed`/`first_death_offset`/`terminal`/`error`/`restore` |
| A seed11 评估（ext） | `.tmp_v431_seg001/full/raw/armA/eval-seed11-multi4-128-ext.jsonl` | 1,261 | `6c3f81f7fe78e750e0dfdd4775ad9e2875af7a3014a698737f1d2b51855ba536` | seed11 扩展批（10 episode；mtime `2026-11-16T04:20:12`）：`rows=10`、`full_segments=0`、`early_deaths=10` —— **10/10 全早死，无一条 ≥300**；7 键同型，仍无帧级字段 |
| A seed12 评估（128 决策） | `.tmp_v431_seg001/full/raw/armA/eval-seed12-armA-multi4-f8664-128.jsonl` | 1,271 | `9d4e6173d77ab087758dfe9dcf18c0e8f5477bdfa49d318395cc9a031c8eb6e2` | seed12 的 A 臂评估（mtime `2026-11-16T03:21:06`）：`rows=10`、`full_segments=2`、`early_deaths=8` —— 8 份里**唯一有 2 条满段**的文件 |
| A seed12 评估（ext） | `.tmp_v431_seg001/full/raw/armA/eval-seed12-multi4-128-ext.jsonl` | 1,268 | `b5910c464e2e6f08acedd98918670ca9ae560ed19bf4d5339e57d7a4fb0203df` | seed12 扩展批（mtime `2026-11-16T04:32:40`）：`rows=10`、`full_segments=0`、`early_deaths=10` |
| A seed17 评估（128 决策） | `.tmp_v431_seg001/full/raw/armA/eval-seed17-armA-multi4-f8664-128.jsonl` | 1,275 | `e9ef878bda22a5c243ed6a4d0f00a58d28e81fd2209f602164b6e68b8e0b966a` | seed17 的 A 臂评估（mtime `2026-11-16T03:33:48`）：`rows=10`、`full_segments=1`、`early_deaths=9` |
| A seed17 评估（ext） | `.tmp_v431_seg001/full/raw/armA/eval-seed17-multi4-128-ext.jsonl` | 1,272 | `3f5190cea06337cf0348bd14264c379ed5b113225821d23b69b00776efa0246d` | seed17 扩展批（mtime `2026-11-16T04:45:19`）：`rows=10`、`full_segments=1`、`early_deaths=9` —— 与随机臂 `cfair-seed17-f8664-ep12` 的「名义满段」**不是同一件东西**（后者是 600 帧 `frames_executed`，前者只是 `return_decisions=300`） |
| A seed18 评估（128 决策） | `.tmp_v431_seg001/full/raw/armA/eval-seed18-armA-multi4-f8664-128.jsonl` | 1,262 | `a67445a0af84a7993680ebb2db9a0e32a6289a60b782c7b3d535e6995e1246c7` | seed18 的 A 臂评估（mtime `2026-11-16T03:46:12`）：`rows=10`、`full_segments=0`、`early_deaths=10` —— 128 决策档里**唯一 0 满段**的 dev seed |
| A seed18 评估（ext） | `.tmp_v431_seg001/full/raw/armA/eval-seed18-multi4-128-ext.jsonl` | 1,260 | `38abc5b91edce237be003cac4b465761fe0da118d86bdcdd005627c17eef2580` | seed18 扩展批（mtime `2026-11-16T04:57:39`）：`rows=10`、`full_segments=1`、`early_deaths=9` |
| E0 cfair 清单证据 | `Document/Update/Alpha/v4.4.0/evidence/E0/cfair_manifest.json` | 33,448 | `7f37df0b297a6f6ab9cee7bd06dcedca2cdf8add5e90672f454f62c9c13b3a47` | **本轮新生成**。schema `v440-e0-cfair-manifest/1`：`.tmp_v431_seg001/full/raw/cfair/` 全部 **80 个** `cfair-seed{S}-f8664-ep*.json` 的逐文件 `{name, bytes, sha256}`（4 个 dev seed × 20），每个 seed 另给 `file_count` / `total_bytes` / `nominal_full_count` / `frames_executed_600_count` / `combined_sha256`，并保留 `first_death_offset_by_file`、`frames_executed_by_file`、`restore_reset_hash_histogram` 与 80 条总 `combined_sha256_all_80`。**§19.1 的登记替代物就是它** |
| E0 A 臂清单证据 | `Document/Update/Alpha/v4.4.0/evidence/E0/armA_manifest.json` | 8,605 | `e19baec1413f0bb0b4a7963d32808be5abc2213895fdb4aed2be479dc501f43d` | **本轮新生成**。schema `v440-e0-arma-manifest/1`：8 份 A 臂 JSONL 的 `{name, bytes, sha256, rows, full_segments, early_deaths}`（另附逐行 `return_decisions_values`、`keys_present`、`mtime_ns`）＋ 2 个权重的 `{name, bytes, sha256}`；汇总 `files=8 / bytes=10,134 / rows=80 / full_segments=6 / early_deaths=74`。**上表第 1–10 行与它逐值一致**（双份登记，互为交叉校验） |

**8 份 A 臂 JSONL 的汇总（本节实测，与 `armA_manifest.json` 一致）**：文件 **8** ／字节 **10,134** ／行 **80**（每份恰好 10 行）／`full_segments` **6** ／`early_deaths` **74**。

### 19.1 cfair 的 80 个文件：为什么用 per-seed `combined_sha256` 代替逐文件登记

| seed | 文件数 | 合计字节 | `nominal_full_count`（`first_death_offset is null`） | `frames_executed_600_count` | `combined_sha256`（20 个 sha256 串接后再 sha256） |
| ---: | ---: | ---: | ---: | ---: | --- |
| 11 | 20 | 26,567 | **0** | **0** | `43bd611a365d2bab06ed6862c310779b549f89b61d22f0cd11af31ef9848483d` |
| 12 | 20 | 26,569 | **0** | **0** | `11d086263ee841d781bc30a7f0f40ce8932572c3e49d221647a41f15c5e539c4` |
| 17 | 20 | 26,547 | **1** | **1** | `8008613798fb19ce8cdde993c1178ad66e19c9947dfd12b7f541820ab6321653` |
| 18 | 20 | 26,563 | **0** | **0** | `8b4e1358170fa7f5acc007f39676349a51499173b6cd5568ce0c209bb5cea036` |
| **合计** | **80** | **106,246** | **1** | **1** | 80 条总 `combined_sha256_all_80` = `92fc56de56e6bcffc8891ed310190fe6c75bc58fe39513aae5dc2957421ef176` |

> **`combined_sha256` 的精确定义（可独立复现，不含任何隐含约定）**：把该 seed 的 **20 个文件名按升序（byte-wise `str` 序）** 排列 → 取出各自的 **64 字符小写 hex sha256** → **无分隔符**依次串接成 **1,280 字符**的 ASCII 串 → 对该串做 **sha256**。`cfair_manifest.json` 的 `method.combined_sha256` 与每 seed 的 `combined_sha256_file_order`（本次用的是哪 20 个名字、什么顺序）把这条定义固定下来；**任何人可以只凭 manifest 里的逐文件 sha256 重算出同一值**（本轮已用 PowerShell 独立重算 4/4 命中）。

**为什么这 80 个文件不逐文件进 `ASSET_SYNC.sha256`（五条理由）**：

1. **校验表是「每轮都要跑一遍」的逐行 `sha256sum -c` 表，80 行会永久抬高每轮校验成本**；而这批文件**总计只有 106,246 B**（平均 1,328 B/文件），单文件的信息量远低于它的登记成本。
2. **它们在语义上是一个整体，不是 80 个独立资产**：同一条 cfair runner、同一个 `target_frame=8664`、同一 `policy_spec="random"`、同一 `restore` 口径（`reset_frame=0` / `frames_walked=8663` / `engine_frame=8663`），差别只有 `--rng-seed {ep}`。**按 seed 分组的 combined 恰好对应「这一批有没有变」这个真实问题**。
3. **强度足够、且能二分**：combined 把 20 个**已实测**的 sha256 串接后再 sha256，任一文件哪怕 1 bit 变化都必然改变该 seed 的 combined；同时 manifest **逐文件保留了 `name` / `bytes` / `sha256`** ⇒ 一旦 combined 不符，**定位到具体文件只需一次本地重算，不需要任何外部信息**。这是「用 4 个综合值守住 80 个文件」，不是「放弃校验」。
4. **避免把一次性随机臂产物的噪声带进主校验表**：这 80 个文件 mtime 全落在 `2026-11-15T21:31:12`–`2026-11-16T05:48:43`（早已冻结），但若未来有人重跑/copy 该目录，**80 行会一次性报 80 个 FAILED**，淹没真正重要的权重行；分成 4 个 combined 则最多 4 条、且**立刻指明是哪个 seed**。
5. **与上游口径一致、同时不隐瞒**：`TRAINING_HOST_REPLY_01.md` §4.3 只点名 `cfair-seed17-f8664-ep12.json` **一个**文件；若按「最小集」只登这 1 行，会**隐瞒另外 79 个**。改用「4 个 combined ＋ 1 份完整 manifest ＋ 1 行 manifest 登记」＝**主校验表只加 1 行，字节身份一个不少**。

> **`combined_sha256` 能证明什么、不能证明什么**：它能证明**这 80 个文件的字节身份没有变**（配合 manifest 的逐文件 sha256，可定位到单个文件）；它**不能**证明这批 episode 的统计性质、也不构成任何新实验。`combined_sha256` **不是**「另一套哈希」——它就是 `sha256sum` 同族的 sha256，只是把**被哈希的对象**从「单个文件内容」换成「20 个已登记文件摘要的串接」。

> **本轮实测的一个数值提醒（不是新实验，只是把 manifest 的计数如实写下来）**：80 个 cfair 随机臂 episode 里，`first_death_offset` 为 `null` 的**只有 1 个**（`cfair-seed17-f8664-ep12.json`），`frames_executed == 600` 的也**只有 1 个**——**即「名义满段」是 1/80，不是 20/80**。这与 `TRAINING_HOST_REPLY_01.md` §4.3「`matched_files = 20`，满足条件的**恰好 1 个**」一致。**注意两个计数不等价**：`nominal_full_count` 看的是 `first_death_offset` 是否为空，`frames_executed_600_count` 看的是 `frames_executed` 是否为 600，一个 episode 理论上可以「顶着非空 offset 却记 600」——本次两者**恰好都只有 ep12**，但口径必须在 manifest 里分开写，不能互相替代。

### 19.2 不重复登记与行数账

- **既有行 0 命中**：追加前对前 **197 行**做 `v431_seg001|multi4|f8664` 匹配 → **0 命中**；`armA` 只命中 §12/§13 的 `.tmp_v432_arm/armA_seed{0,1,2}.pt` 与 `.tmp_v433_repair/armA_repair_seed{0,1,2}.pt` **六条完全不同的文件**（那是 v4.3.2/v4.3.3 的产物，不是 R37/R38 的 `.tmp_v431_seg001/`）。
- **新旧交集 0**：新增 12 条的 **12 个路径 / 12 个 sha256** 与前 197 行**逐一比对，路径交集 0、哈希交集 0**；新增 12 条自身**路径 12 唯一、哈希 12 唯一**。
- **行数账**：197 + 12 = **209 行**（文件仍是 **LF / 无 BOM**，末行以 `\n` 结束，每行 `64 hex + 两个空格 + 相对路径`，**格式异常行 0 条**；文件从 **22,468 B** 增至 **24,032 B**，+1,564 B）。
- **全文件现状**：**209 路径全部唯一**，**209 行里 206 个唯一哈希** —— 那 3 个重复值仍是 §16/§17 已记录的 `.tmp_v450_cf/ckpt/seed{0,1,2}/round2.pt` ↔ `.tmp_v600_obsp/ckpt/k12/seed{0,1,2}/round2.pt` 三对**逐位相同**，**本次未新增任何同哈希对**（与 §18.3 的「194 唯一值」同源）。
- **`ASSET_SYNC.md` 本身不在校验表内**（实测 `Select-String 'ASSET_SYNC'` → **0 行**）⇒ 本节追加不会让任何既有行 FAILED，也不需要为自己重钉。
- **未动既有行、未动既有证据/脚本/预注册**：本轮只 **append** 了 `ASSET_SYNC.md`（本节）与 `ASSET_SYNC.sha256`（12 行），新增了 `evidence/E0/` 下 2 个 JSON；**没有**修改前 197 行、没有改 `.tmp_v431_seg001/**`（只读）、没有改任何 v4.x/v5.x/v6.x/v7.x/v8.x 既有证据、没有改运行脚本或预注册；**未** `git add` / `git commit`；**未**启动引擎（本轮 0 次引擎调用、0 次游戏进程）。

### 19.3 校验结论

`cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256`（WSL，复验时刻 `2026-11-22T10:5x+0800`）：**209 行 = 208 OK / 1 FAILED**（`sha256sum` 退出码 **1** —— 仅因那 1 条）。

- **基线（追加前，197 行）**：§18.4 记的 **196 OK / 1 FAILED**；本次追加后 **208 OK / 1 FAILED** ⇒ **12 条新行 12/12 OK，整体 FAILED 数没有新增**。
- **唯一 FAILED 是已知并允许的那一条**：`Document/Update/Alpha/v4.2.0/journal.jsonl` —— §9.8 记载的活文件漂移，本轮**未改该行**、也**未重钉**。**除此以外没有第二个 FAILED**（§17.4 记过的 `v5.9.0/SYNTHESIS_2026-11-21.md` 那条本次**已通过**，与 §18.4 一致）。
- **新登记 12 行贡献 12/12 OK**：`tail -n 12 ASSET_SYNC.sha256 | sha256sum -c -` 实测 **12 OK、0 FAILED**（`rc = 0`）—— 2 个 `.pt` ＋ 8 份 A 臂 JSONL ＋ 2 份 manifest。
- **两侧一致**：登记前 Windows 侧 **Python `hashlib`** 实测 90 个源文件与 4 个 combined，**再**用 **PowerShell `Get-FileHash`** 独立复算 **90/90 + 4/4 全部相符**，与 WSL 的 **12/12 OK** 相符 ⇒ 不存在「Windows 侧算错、WSL 侧跟着错」的同源错误。
- **没有登记期漂移**：这 12 条对应文件的 mtime 全在 `2026-11-16T01:38:47`–`2026-11-22T10:52:12`，其中 10 个源文件是 **6 天前**的冻结产物、2 个 manifest 是**本轮刚写**的；钉版后重跑仍 **12/12 OK**。

> **一句话结论**：本轮补齐了 R37/R38 的资产册缺口 —— 追加 **12 条**（2 个权重 ＋ 8 份 A 臂评估 ＋ 2 份新 manifest），`sha256sum -c` **208 OK / 1 FAILED**，**新登记 12 条全部 OK**、整体 FAILED 数**保持为 1**（仍只有已知允许的 `v4.2.0/journal.jsonl`）。`.tmp_v431_seg001/full/raw/cfair/` 的 **80 个文件按设计不逐文件登记**，由 `cfair_manifest.json` 的 **4 个 per-seed `combined_sha256`**（`43bd611a…` / `11d08626…` / `80086137…` / `8b4e1358…`）＋ 逐文件 `{name, bytes, sha256}` 代表，理由见 §19.1；`.tmp_v431_seg001/**` **全程只读、0 修改**。

---

## 20. v8.4.0–v8.5.0 新资产（2026-11-22）

> **为什么有这一节**：AGENTS.md §8.2 要求「新增/移动/删除任何资产，必须同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`」。`Document/Update/Alpha/v8.4.0/`（BOMBRULE2）与 `Document/Update/Alpha/v8.5.0/`（DISTILL）两轮新资产在本次追加前 **一行都没有入册** —— `ASSET_SYNC.sha256` 前 209 行对 `v8\.4\.0|v8\.5\.0|BOMBRULE2|DISTILL|policy36` 的匹配是 **0 命中**。本节与 `ASSET_SYNC.sha256` 末尾追加的 **63 行**成对登记。
>
> **只读边界**：本轮只 **append** 两个清单文件。`Document/Update/Alpha/v8.4.0/**`、`Document/Update/Alpha/v8.5.0/**`、`.tmp_v850_distill/ckpt/**` 全程**只读** —— 0 修改 / 0 改名 / 0 删除 / 0 新建；登记完成后 63 个文件的 `st_mtime_ns` 与登记开始时**逐纳秒相同**（见 §20.4）。**未启动引擎**（本轮 0 次引擎调用、0 个游戏进程）。
>
> **实测口径**：63 条的 `bytes` 与 `sha256` **全部现场实测**，且用两套独立实现互相验证 —— Windows 侧 `PowerShell Get-FileHash -Algorithm SHA256`，WSL 侧 `sha256sum -c`。两轮自带的 `manifest.json` 里的内联 sha256 **只用于交叉核对，不代替实测**（核对结果见 §20.4）。
>
> **最小集口径**：只登记「判据文档 + 顶层证据 + 评测原始件 + checkpoint」。过程件（`logs/`、`runs/`、`scripts/`、`refs/commands/`、`smoke/`、`ab/`）不入册，理由与**字节身份仍有锚**的证明见 §20.2。

### 20.1 登记范围（**63 条** = v8.4.0 **28** 条 + v8.5.0 **35** 条）

| 分组 | 条数 | 字节合计 | 位置 |
| --- | ---: | ---: | --- |
| §20.1.1 v8.4.0 文档与顶层证据 | 7 | 324,005 | `Document/Update/Alpha/v8.4.0/**` |
| §20.1.2 v8.4.0 评测 JSONL（18 d600 + 3 d900） | 21 | 65,104,761 | 同上 |
| §20.1.3 v8.5.0 文档与顶层证据 | 14 | 2,168,458 | `Document/Update/Alpha/v8.5.0/**` |
| §20.1.4 v8.5.0 评测 JSONL + summary（9 + 9） | 18 | 73,521,074 | 同上 |
| §20.1.5 v8.5.0 checkpoint（36 决策框权重） | 3 | 117,813 | `.tmp_v850_distill/ckpt/seed{0,1,2}/policy36.pt` |
| **合计** | **63** | **141,236,111** | — |

#### §20.1.1 v8.4.0 文档与顶层证据（7 条）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.4.0/SUMMARY_FOR_DECISION.md` | 27,051 | `39a3318746b7cb66551bb61f060fcf42c8abad1f678532933fe9ba0b638e8433` | v8.4.0 决策摘要：R0–R5 六轮 d600 主表 + R5 d900 三档；**只登记字节身份，正文未改** |
| 2 | `Document/Update/Alpha/v8.4.0/README_INDEX.md` | 10,958 | `eb281035f0c5ef679db5992b5ca817462b1bb5d18db4957d91fdeed99ad2ec17` | v8.4.0 索引：阅读顺序与产物指向（RUN.md / manifest / CROSSCHECK） |
| 3 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/RUN.md` | 32,988 | `e5019b4c4a012c43ee8b0bde0a40ec27c54e7b7a967b41753a433cb0803da662` | v8.4.0 运行记录：命令表、退出码、六轮汇总与 d900 复现口径 |
| 4 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/manifest.json` | 23,003 | `5bf84a0bd7482bf98fc2ab8dfaa4b4ec179dc899ec0d372c1f5c04139aced2b2` | **清单锚**：`files` 逐条 `{relpath, bytes, sha256, lines}` 覆盖 BOMBRULE2 下 **115** 个文件（除自身）；`counts={files:115, jsonl_d600:18, jsonl_d900:3, bytes_total:69,410,012}`；`aggregate_sha256=46abf41ce9dd291a…`；`self_check` 可独立重算 |
| 5 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/diagnostics.json` | 175,407 | `9fef417dae058beae369209ab3cb0bef4e07333add9b0b36f322c88dcfdcd7f5` | 逐轮诊断 + P1（Fisher + Holm）三档检验表；`engine_started:false` 指该聚合脚本自身未起引擎 |
| 6 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/refs/CROSSCHECK.json` | 32,865 | `a27e04475bf538f78c778a9d1c6f36a02e169fe7d8a273d6aa7ec8cf3caa462f` | 交叉复核：`cells_checked=21` / `cells_failing=0` / `log_lines_missing=[]` / `runner_summary_missing=[]` |
| 7 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/refs/COPY_INDEX.json` | 21,733 | `088f7969f05909f4eb059e1d3e7ea977a7ac5cd2f078c0dfbf7206d4d7a82281` | 副本索引：**81** 个文件、`missing_sources=[]` |
| | **小计** | **324,005** | 7 条 | |

#### §20.1.2 v8.4.0 评测 JSONL（21 条 = 18 份 d600 + 3 份 d900）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R0_d600_seed0.jsonl` | 3,036,036 | `ceabf40ff6ab8992e8c60b7f8791f5d5f43d547d538700dae85606267b14b64a` | R0 / d600 / seed0 评测原始逐 episode JSONL；**1,200 行** |
| 2 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R0_d600_seed1.jsonl` | 2,979,598 | `6fad534cdc6c993e053e935e9aa4438b67c9b252640141673fd30272fd2d4037` | R0 / d600 / seed1 评测原始逐 episode JSONL；**1,200 行** |
| 3 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R0_d600_seed2.jsonl` | 4,365,289 | `92e72ca2b98ed3f8b862c4e56a7c24ac221ca7fcf50d25296f15a9f317ff30a0` | R0 / d600 / seed2 评测原始逐 episode JSONL；**1,200 行** |
| 4 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R1_d600_seed0.jsonl` | 3,647,087 | `8426b86e9823bba3c1490c952b2838ace7214a33890efa77fa20ee1d312a382c` | R1 / d600 / seed0 评测原始逐 episode JSONL；**1,200 行** |
| 5 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R1_d600_seed1.jsonl` | 3,559,357 | `3a7cb18edf069a9eb08227e6673912d298f3978e2fa960a0e9ca24b58d3f621d` | R1 / d600 / seed1 评测原始逐 episode JSONL；**1,200 行** |
| 6 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R1_d600_seed2.jsonl` | 5,498,662 | `219287e51a6f2187fa92cece4a351055fb9b4b6f8df37ab7c26708de391b8279` | R1 / d600 / seed2 评测原始逐 episode JSONL；**1,200 行** |
| 7 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R2_d600_seed0.jsonl` | 2,186,717 | `b24f8d0caa9f59b2b209ef2b2d18cf6604ea29b8607fe614390106d22c7c7b2e` | R2 / d600 / seed0 评测原始逐 episode JSONL；**1,200 行** |
| 8 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R2_d600_seed1.jsonl` | 2,114,880 | `19f3ea6d83fd5518f8a29803d4a36b8f80092c68236d0c47d30e408b012c0e38` | R2 / d600 / seed1 评测原始逐 episode JSONL；**1,200 行** |
| 9 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R2_d600_seed2.jsonl` | 3,144,536 | `baabd78e9ca8de45fffbfeb9f7339f9a19f96731f578e4bc699ee85fbe9f3cb6` | R2 / d600 / seed2 评测原始逐 episode JSONL；**1,200 行** |
| 10 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R3_d600_seed0.jsonl` | 2,902,964 | `0ee1ba427e52b8087f34fd9ea8eff225bbca8e7326ce47312a18a3af94ae2b2e` | R3 / d600 / seed0 评测原始逐 episode JSONL；**1,200 行** |
| 11 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R3_d600_seed1.jsonl` | 2,894,662 | `e18728d6122aaf64641f5fb1f2bf7bb43b0c47d634618afe6b24b867b0399172` | R3 / d600 / seed1 评测原始逐 episode JSONL；**1,200 行** |
| 12 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R3_d600_seed2.jsonl` | 3,311,566 | `9cc8d6f0b0341dccdc0ef26971823833246e9da974b454babc9d9e1b701f9aa9` | R3 / d600 / seed2 评测原始逐 episode JSONL；**1,200 行** |
| 13 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R4_d600_seed0.jsonl` | 2,436,059 | `505792b3e43469cecdb7c2fc900808e4eb2d3c2147a81c958f519385af369951` | R4 / d600 / seed0 评测原始逐 episode JSONL；**1,200 行** |
| 14 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R4_d600_seed1.jsonl` | 2,185,464 | `b711ac7b5ff5107468b82d37545463995e79416711ea824d97a758d9536f7a40` | R4 / d600 / seed1 评测原始逐 episode JSONL；**1,200 行** |
| 15 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R4_d600_seed2.jsonl` | 2,990,116 | `b4ca413b2407c859fe5eb5901f97f7532d1746967d230a9b9dabbfe7216d7478` | R4 / d600 / seed2 评测原始逐 episode JSONL；**1,200 行** |
| 16 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R5_d600_seed0.jsonl` | 2,630,968 | `2f40d89a0cb54b8f84730a29c4dc573565623a1dc0be659ec87c2bc47e5ead33` | R5 / d600 / seed0 评测原始逐 episode JSONL；**1,200 行** |
| 17 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R5_d600_seed1.jsonl` | 2,631,471 | `faada3645d534b8b6a88f5e3da1ba298961e01fa06eff046285a6f7e23469f9f` | R5 / d600 / seed1 评测原始逐 episode JSONL；**1,200 行** |
| 18 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R5_d600_seed2.jsonl` | 3,615,943 | `ff1ba5524a589e73af79d07e27315b38b3f738ba84e38ce9d26e1d01513bcd58` | R5 / d600 / seed2 评测原始逐 episode JSONL；**1,200 行** |
| 19 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R5_d900_seed0.jsonl` | 2,652,259 | `627233ec2ee7ba185b8fb1d34cf740d53a43d89fbbc8e28cd66aab519aa55123` | R5 / **d900**（预算加倍档）/ seed0 评测原始 JSONL；**1,200 行** |
| 20 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R5_d900_seed1.jsonl` | 2,646,975 | `1400e7f67043b8dddbc8e235f5e84c3487452da382e93ef3facbf4ef02665e55` | R5 / **d900**（预算加倍档）/ seed1 评测原始 JSONL；**1,200 行** |
| 21 | `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/eval_R5_d900_seed2.jsonl` | 3,674,152 | `eb7288dfd0d81b0dc468acc44b1ee088eec2a160b1c151b1833c54f97071634c` | R5 / **d900**（预算加倍档）/ seed2 评测原始 JSONL；**1,200 行** |
| | **小计** | **65,104,761** | 21 条 | |

#### §20.1.3 v8.5.0 文档与顶层证据（14 条）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.5.0/SUMMARY_FOR_DECISION.md` | 35,525 | `cbef998dba7011d0657b44382b6bad345a44b32cffd867c587c75cecceca3033` | v8.5.0 决策摘要：规则蒸馏 + 三 seed 训练 + d300/d600/d900 评测；**只登记字节身份，正文未改** |
| 2 | `Document/Update/Alpha/v8.5.0/README_INDEX.md` | 14,760 | `578eb9651ab20cf326a9801c5f29b374da0e168585c23b70992df0cc378d08ea` | v8.5.0 索引：阅读顺序与产物指向 |
| 3 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/RUN.md` | 32,935 | `713077c66479c1a5e60651c590697ed482063150e6be5056ba9998f1d2968e89` | v8.5.0 运行记录：预检 → 规则目标 → 训练 → 三档评测 → 发布自检 |
| 4 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/manifest.json` | 17,327 | `d8250577a5bc88177fd6aab3e13cb27ff31dc610aa661ffaf61c6704f728ea32` | **清单锚**：`files` 逐条 `{sha256, bytes}` 覆盖 DISTILL 下 **124** 个文件（`excludes="manifest.json"`）；`file_count=124` / `total_bytes=75,845,862` |
| 5 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/ckpt_manifest.json` | 1,715 | `4e84cb275c4f4281e3aae754664ae17d0130bc30eb610419ef27a72e3e3debb0` | 3 个 `policy36.pt` 的 `{sha256, bytes}` + 3 个输入 ckpt（`.tmp_v450_cf/ckpt/seed*/round2.pt`）；`file_count=3` / `total_bytes=117,813` |
| 6 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/diagnostics.json` | 40,410 | `2d321a9a8d5cefc6f115fd535713efc9fc77217c80fc53b8f64379b62e4cf02d` | 训练/评测诊断汇总：`all_pass:true`、`missing:[]` |
| 7 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/rule_targets.jsonl` | 1,794,237 | `a7b788e18687ce7fdab1c99d2b645b44d2499dc35427597285982b825c9fdc83` | 规则目标逐行数据；其 sha256 与 `rule_target_stats.targets_sha256` **逐位相同** → 两份互为交叉锚；**5,166 行** |
| 8 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/rule_target_stats.json` | 2,784 | `310564f0d779e29c331e853f2468e8ea2be1182abe8e1627f215dc6004c07180` | `targets_sha256=a7b788e18687ce7f…` / `targets_bytes=1,794,237` / `states=5166` / `status=complete` / `all_pass:true` |
| 9 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/bomb_usage.json` | 31,074 | `56f1457d6911d06b4ffc3d662ee3109228cf1ae8f8d3874a5bfd37b7741ee538` | 三档决策预算（300/600/900）的炸弹使用统计；`missing:[]` |
| 10 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/train_summary.json` | 3,065 | `e85dd0a0dbf8472470037c9ea9d697a90da2d5c5d4e2d3b14014c57dfa49c50f` | 三 seed 训练汇总 `seeds={0,1,2}`、`all_pass:true`；`design` = `BCE36 + 0.5*BCE_bomb(R5)`, Adam 0.001, batch 256, 200 epochs |
| 11 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/fidelity_d3.json` | 26,626 | `ab63c2b161558680a83664ad54e59f69869977763d0775728ae4cb7503e5e38d` | D3 保真度闸结果（与 v8.4.0 `refs/fidelity_d3.json` 同源，字节不同：26,626 vs 26,627） |
| 12 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/audio_audit.json` | 4,669 | `42c95d801a43ef14e6eaade95ef780055313d43d8ec4ba892e8a357595164a23` | **§9 静音审计**：`cfg_files=38 / decoded=38 / muted=38 / audible=0`、`all_muted:true`、`all_pass:true` |
| 13 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/verify.json` | 125,615 | `f078170f4e980001e999a4a1c0e0ecf472aff24000dac67ed92665c0c24e9add` | 发布前校验：`checks` 4 组、`all_pass:true` |
| 14 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/publish.json` | 37,716 | `f354144a13e9946e477006601858fe09b58e501bed3ad8112681489017390066` | 发布清单：`file_count=120`、`dest=…/v8.5.0/evidence/DISTILL`、`v850_distill_sha256=65e025139e68bd17…`（产链脚本身份） |
| | **小计** | **2,168,458** | 14 条 | |

#### §20.1.4 v8.5.0 评测 JSONL 与 summary（18 条 = 9 + 9）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d300_seed0.jsonl` | 6,330,256 | `6d8261e05836413c1bf52432cec21ef02bd4410a3890a4c4bb3a8e10ec2a1e85` | d300 / seed0 评测原始 JSONL；**1,200 行** |
| 2 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d300_seed0.summary.json` | 5,998 | `0a1f8782b3c1af83f8bf7421a54ccf78de10bef5c93b8ba750e3fc45dd2d0425` | d300 / seed0 评测 summary（与左侧 JSONL 成对） |
| 3 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d300_seed1.jsonl` | 5,959,352 | `404161b75d8618a69f11b8c682aca0bc2acf1b9984d28d1d285756bc60b3198b` | d300 / seed1 评测原始 JSONL；**1,200 行** |
| 4 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d300_seed1.summary.json` | 5,991 | `43f5436eab1f68616cb8441823f79e90e95dc76b432dcbcc0650808a985cf0f5` | d300 / seed1 评测 summary（与左侧 JSONL 成对） |
| 5 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d300_seed2.jsonl` | 5,446,991 | `afd2662d853c3cd44f053ef03711dc1a23d92c3521893af0ce7fb6b701812286` | d300 / seed2 评测原始 JSONL；**1,200 行** |
| 6 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d300_seed2.summary.json` | 5,980 | `cc8e4a5b8edec44645a2ab7aac9c0a40cd431959d97e5d13d88f7491257f1864` | d300 / seed2 评测 summary（与左侧 JSONL 成对） |
| 7 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d600_seed0.jsonl` | 9,273,352 | `2facea1d1842000873108763b8d767fbf6eddf3f9329bc1b4e5adc2b8bcc5b5d` | d600 / seed0 评测原始 JSONL；**1,200 行** |
| 8 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d600_seed0.summary.json` | 6,055 | `34fe9545df40b79e084903deba5eba8a78cbc797f9afbf7713175d902fa6249d` | d600 / seed0 评测 summary（与左侧 JSONL 成对） |
| 9 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d600_seed1.jsonl` | 8,891,953 | `2cefb62f184198ffe78a5dfe6c07fbd6cce8b8e1b67b1b79939b509964423518` | d600 / seed1 评测原始 JSONL；**1,200 行** |
| 10 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d600_seed1.summary.json` | 6,078 | `13170b5cd8ec73fd756ca780c3fecd391818d5f49fbc63dc7c6dd9f2004c2843` | d600 / seed1 评测 summary（与左侧 JSONL 成对） |
| 11 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d600_seed2.jsonl` | 8,432,507 | `1108b6e68939c825369ed1cc130692066c15e3c2f1865e025f38af5af960c64d` | d600 / seed2 评测原始 JSONL；**1,200 行** |
| 12 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d600_seed2.summary.json` | 6,079 | `6f44b7e4dafaed20a301ae52ba092385409bcbc92962d9f72ce5dc8ec286f979` | d600 / seed2 评测 summary（与左侧 JSONL 成对） |
| 13 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d900_seed0.jsonl` | 10,356,076 | `3fa5fb0f7af03f38bc09fe7233c271f8103e421c6bccd3d61f8f9731705aec07` | d900 / seed0 评测原始 JSONL；**1,200 行** |
| 14 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d900_seed0.summary.json` | 6,065 | `35f17b706354554b3d1a316f93b8bbfc516cee0bad638594b727106d808b32a2` | d900 / seed0 评测 summary（与左侧 JSONL 成对） |
| 15 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d900_seed1.jsonl` | 9,466,220 | `00bf021c39b362f4b3c3f4a32e324f6838cde4befca9869d5fc2ef0585bcac4d` | d900 / seed1 评测原始 JSONL；**1,200 行** |
| 16 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d900_seed1.summary.json` | 6,060 | `dddbd521e62c289abf133d4cde035a406e504d2d255f53539450693decd60a73` | d900 / seed1 评测 summary（与左侧 JSONL 成对） |
| 17 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d900_seed2.jsonl` | 9,309,992 | `bbf9009f2ff702c46e359ed2547eed2206d13cea8e07e58a5a45cbe7e0182124` | d900 / seed2 评测原始 JSONL；**1,200 行** |
| 18 | `Document/Update/Alpha/v8.5.0/evidence/DISTILL/eval_d900_seed2.summary.json` | 6,069 | `390e247b6ba63b11de7ed426017eba31718303d3676bcd356f9fa12236d55ffc` | d900 / seed2 评测 summary（与左侧 JSONL 成对） |
| | **小计** | **73,521,074** | 18 条 | |

#### §20.1.5 v8.5.0 checkpoint（3 条，`.tmp_v850_distill/`，不在 `Document/` 下）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `.tmp_v850_distill/ckpt/seed0/policy36.pt` | 39,271 | `326760dc53d97588211246420d06b02935b81eff9b891dc18ba72411f7cf70ab` | seed0 策略权重（36 决策框）；`ckpt_manifest.json` 记录值与本表**逐位相同** |
| 2 | `.tmp_v850_distill/ckpt/seed1/policy36.pt` | 39,271 | `32c38859cd5e4c2be2aa8638960942280401dfe3d4bc14df47455f0872e4b6b8` | seed1 策略权重（36 决策框）；`ckpt_manifest.json` 记录值与本表**逐位相同** |
| 3 | `.tmp_v850_distill/ckpt/seed2/policy36.pt` | 39,271 | `c997e4a288ae837c0d4ff062560277648822151f432482e90b14d18cb96ae5df` | seed2 策略权重（36 决策框）；`ckpt_manifest.json` 记录值与本表**逐位相同** |
| | **小计** | **117,813** | 3 条 | |

### 20.2 跳过说明（未入册 **187** 个文件 / **4,273,976 B**，逐组给理由）

两轮目录实测：v8.4.0 **119** 个文件 / 69,474,727 B（入册 28 / 65,428,766 B）；v8.5.0 **128** 个文件 / 75,917,547 B（入册 32 / 75,689,532 B）。差额即下列 91 + 96 = **187 个文件**。

| 组 | 文件数 | 字节 | 不入册理由 |
| --- | ---: | ---: | --- |
| v8.4.0 冻结预注册 `PREREGISTRATION_BOMBRULE2_001.md` | 1 | 3,703 | **冻结件（禁改）且不在 `BOMBRULE2/manifest.json` 覆盖范围内**（该清单只覆盖 `evidence/BOMBRULE2/**`）。按最小集未纳入 → **明确列为未入册缺口**，需要时单独补登。 |
| `evidence/BOMBRULE2/ab/`（`before/`、`after/`、`repro/` + `v800_bomb_pre.py` 快照 + `repro_before_check.json`） | 8 | 378,169 | v8.0.0 runner 的 A/B 前后对照与 2-episode 复现，属**过程性对照件**；判据表已由 `RUN.md` + `diagnostics.json` 承载，且整组 8 件逐文件 `{relpath,bytes,sha256}` 已在入册的 `BOMBRULE2/manifest.json` 里。 |
| `evidence/BOMBRULE2/logs/`（33 份 `.out`） | 33 | 64,404 | 运行日志，内容与 `diagnostics.json` / `manifest.json` 的计数重复；是**可重跑重建的输出**，不是资产。逐件 sha256 仍在入册 manifest 内。 |
| `evidence/BOMBRULE2/refs/commands/`（27 份 `commands*.tsv`） | 27 | 12,404 | 每次运行记录的 argv（单件 40–1,435 B），属**可重建元数据**；结论性内容已进 `RUN.md`。 |
| `evidence/BOMBRULE2/refs/` 其余 8 件（`ab_check.json`、`ckpt_assertions_start.json`、`deviations_executor_scratch.md`、`fidelity_d3.json`、`rulecheck.py`、`rulecheck.txt`、`v800_bomb_used.py`、`v840_patch_vs_pre.diff`） | 8 | 153,030 | 复核脚本 / 规则校验文本 / runner 快照 / 补丁 diff / 保真度副本。其中 `refs/fidelity_d3.json`（26,627 B）与已入册的 `v8.5.0/DISTILL/fidelity_d3.json`（26,626 B）**同源不同字节**，登记后者已足够定位；runner 与 diff 的更完整身份由 `BOMBRULE2/manifest.json` 的 `scripts/` 条目给出（`tools/rl/` 下 **没有** Git 跟踪的 v8.4.0/v8.5.0 runner，`git ls-files` = 0，所以 manifest 是这些脚本字节身份的唯一锚）。 |
| `evidence/BOMBRULE2/scripts/`（8 个 `.py`/`.sh` + `__pycache__/analyze.cpython-310.pyc`） | 9 | 113,134 | 生成器脚本本身不是证据；`__pycache__/` 是 Python 字节码缓存，**一律不入册**（理由见下）。8 个脚本的 sha256 已在入册 manifest 的 `scripts/*` 条目里。 |
| `evidence/BOMBRULE2/smoke/`（5 份 `bomb_smoke_R{1..5}.json`） | 5 | 3,321,117 | 冒烟测试原始大件（466,094–761,726 B/件，合计 3,321,117 B），**过程性中间数据**；其结论已进 `diagnostics.json` 与 `RUN.md`，逐件 sha256 仍在入册 manifest 内。 |
| v8.5.0 冻结预注册 `PREREGISTRATION_DISTILL_001.md` | 1 | 4,073 | **冻结件（禁改）且不在 `DISTILL/manifest.json` 覆盖范围内**（该清单 `excludes="manifest.json"` 且根为 `evidence/DISTILL`）。按最小集未纳入 → **明确列为未入册缺口**，需要时单独补登。 |
| `evidence/DISTILL/` 顶层其余 10 件：`ckpt_assertions{,_final}.json`（4,419 B×2）、`commands.tsv`（13,488 B）、`preflight.json`（4,426 B）、`process_residue{,_final}.json`（614 B×2）、`train_seed{0,1,2}.json`（12,178/12,186/12,165 B）、`v830_honoured.json`（5,648 B） | 10 | 70,157 | 预检 / 断言 / 残留进程快照 / 逐 seed 训练报告 / 命令行记录：都是**汇总表的展开或运行时元数据**，已由入册的 `train_summary.json`、`diagnostics.json`、`ckpt_manifest.json` 汇总；未入册不丢信息。 |
| `evidence/DISTILL/logs/`（25 份 `.out`） | 25 | 32,095 | 运行日志，与 `runs/**/run.log` 重复；统计已进 `diagnostics.json`。 |
| `evidence/DISTILL/runs/`（**11** 个运行目录、51 份文件：9 组评测各 1 份 `run.log` + 4 份 `w*-stderr.log`，另加 `v850-fidelity/` 4 份与 `v850-smoke/` 2 份） | 51 | 87,147 | 每次运行的 stdout/stderr 快照，属**过程件**；`run.log` 与 `logs/*.out` 内容同源（前者是后者的副本）。 |
| `evidence/DISTILL/scripts/`（9 个 `.py`/`.sh`） | 9 | 34,543 | 产链脚本（`chain.sh`、`grid.sh`、`tables.py`、`verify.py` 等）不是证据；其 sha256 全部在入册 `DISTILL/manifest.json` 的 `scripts/*` 条目里，且 `publish.json` 记录了产链身份 `v850_distill_sha256=65e025139e68bd17…`。 |
| **小计（v8.4.0）** | **91** | **4,045,961** | 119 − 28 = 91 |
| **小计（v8.5.0）** | **96** | **228,015** | 128 − 32 = 96 |
| **合计** | **187** | **4,273,976** | 247 − 60 = 187 |

**为什么「不入册」不等于「没有字节身份」** —— 两份入册的 `manifest.json` 就是这些过程件的锚，本节已实测复核：

| 清单 | 覆盖 | 记录形态 | 本轮实测复核 |
| --- | ---: | --- | --- |
| `v8.4.0/evidence/BOMBRULE2/manifest.json` | **115** 个文件（`evidence/BOMBRULE2/**` 除自身） | `files[]` 逐条 `{relpath, bytes, sha256, lines}` | **115/115 全等，0 不符** |
| `v8.5.0/evidence/DISTILL/manifest.json` | **124** 个文件（`evidence/DISTILL/**`，`excludes="manifest.json"`） | `files{}` 逐条 `{sha256, bytes}` | **124/124 全等，0 不符** |

即：主校验表只加 **2 行 manifest**，就把 **115 + 124 = 239 个**文件的逐文件 sha256 纳入了可校验范围；两个 manifest 各自的 `self_check` / `aggregate_sha256` 又对自身做了防篡改绑定。

对 187 个未入册文件逐项核对的结果：**185 个**的 `{bytes, sha256}` 已被上表两份入册 manifest 收录（v8.4.0 侧 90 个 + v8.5.0 侧 95 个），**只有 2 个**没有任何记录 —— 即两份 `PREREGISTRATION_*.md`，已在上表与 §20.4 里单列为缺口。

**`__pycache__` 实况（按要求只报告、不删）**：

- `Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/__pycache__/` —— **不存在**（已 `Get-ChildItem -Recurse -Directory -Filter __pycache__` 实测）；
- 但存在 **`Document/Update/Alpha/v8.4.0/evidence/BOMBRULE2/scripts/__pycache__/analyze.cpython-310.pyc`**（**22,932 B**，mtime `2026-11-22 10:14:24`）—— 属证据目录内的 Python 字节码缓存，**未删除、未移动、未入册**；它已被 `BOMBRULE2/manifest.json` 的 `scripts/__pycache__/analyze.cpython-310.pyc` 条目收进逐文件 sha256；
- `Document/Update/Alpha/v8.5.0/evidence/DISTILL/` 下 **没有** `__pycache__/`（实测 0 个）。

### 20.3 与 §1–§19 不重复的说明（实测已核）

- **既有 209 行 0 命中**：追加前对前 209 行做 `v8\.4\.0|v8\.5\.0|BOMBRULE2|DISTILL|policy36|v850` 匹配 → **0 命中**；`Document/Update/Alpha/v8.0.0`、`v8.1.0`（§18）与 `.tmp_v431_seg001`（§19）是**不同版本/不同目录**，本轮 63 条与之无交集。
- **路径 63 唯一 / 哈希 63 唯一**：63 条内部的 sha256 逐条比对 **0 组重复**（含 3 个 `policy36.pt` —— `326760dc…` / `32c38859…` / `c997e4a2…` 三者互不相同，说明三 seed 确实训出不同权重）。
- **与前 209 行 0 交集**：63 个路径、63 个 sha256 与前 209 行的 **209 个路径 / 206 个唯一哈希**逐一比对，**路径交集 0、哈希交集 0** → 本次**没有引入任何新的同哈希对**。
- **全文件现状**：**272 行 / 272 唯一路径 / 269 唯一哈希**（那 3 个重复值仍是 §16/§17 已记录的 `.tmp_v450_cf/ckpt/seed{0,1,2}/round2.pt` → `.tmp_v600_obsp/ckpt/k12/seed{0,1,2}/round2.pt` 三对逐位相同，与本轮无关）。
- **行数与格式**：209 + 63 = **272 行**；文件从 **24,032 B** 增到 **32,394 B**（+8,362 B）；仍是 **LF / 无 BOM**，末行以 `\n` 结束，每行 `64 hex + 两个空格 + 相对路径`，**格式异常行 0 条**。
- **`ASSET_SYNC.md` 本身不在校验表内**（`Select-String 'ASSET_SYNC'` → 0 行）→ 本节追加不会让任何既有行 FAILED，也不需要为自己重钉。
- **Git 跟踪状态（实测）**：`git status --porcelain -- Document/Update/Alpha/v8.4.0 Document/Update/Alpha/v8.5.0` → 两目录均为 `??`（整目录 untracked）；`git ls-files -- <两目录>` = **0**，`git ls-files | grep v8[45]0` = **0** ⇒ 这些证据**无法由 Git 重建**，本节 + `ASSET_SYNC.sha256` 是它们目前的字节身份记录。本轮按约束**未** `git add` / `git commit`。

### 20.4 校验结论

`cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256`（WSL，复验时刻 `2026-11-22T11:4x+0800`）：**272 行 = 271 OK / 1 FAILED**（`sha256sum` 退出码 **1** —— 仅因那 1 条；`271 + 1 = 272` 校验行全部参与）。

- **基线（追加前 209 行）**：同一命令实测 **208 OK / 1 FAILED**；追加后 **271 OK / 1 FAILED** ⇒ **63 条新行 63/63 OK，整体 FAILED 数没有新增**。
- **唯一 FAILED 是已知并允许的那一条**：`Document/Update/Alpha/v4.2.0/journal.jsonl`（§9.8 记录的活动文件漂移）。本轮**未改该行、未重钉**；`v5.9.0/SYNTHESIS_2026-11-21.md` 那条本次**已通过**（与 §18.4 / §19.3 一致）。
- **新增 63 行贡献 63/63 OK**：`tail -n 63 ASSET_SYNC.sha256 | sha256sum -c` 实测 **63 OK / 0 FAILED**（`rc=0`）—— 7 份 v8.4.0 文档/证据 + 21 份 v8.4.0 评测 + 14 份 v8.5.0 文档/证据 + 18 份 v8.5.0 评测 + 3 个 checkpoint。
- **两侧一致（不是同源错误）**：63 条的登记值由 **Windows `Get-FileHash`** 生成，WSL `sha256sum` 独立重算 **63/63 相同** ⇒ 不存在「Windows 侧算错、WSL 侧跟着错」。
- **与产出方 manifest 的交叉核对**：把两轮入册 `manifest.json` 里记录的 **239** 条内联 `{bytes, sha256}` 逐条与现场重算比对 —— **239/239 全等，0 处不符**（BOMBRULE2 115/115、DISTILL 124/124）。
- **两处额外的独立锚**：① `rule_target_stats.targets_sha256` = `a7b788e18687ce7f…` = 本表 `rule_targets.jsonl` 实测值，且 `targets_bytes=1,794,237` 与表格字节相同；② `ckpt_manifest.json` 记录的 3 个 `policy36.pt` 的 `{sha256, bytes}` 与本表实测**逐位相同**（3/3）。
- **登记期无漂移**：63 个文件的 `st_mtime_ns` 在登记开始与登记结束时**逐一相同**（0 个变化）；本轮**没有**任何写入落在 `v8.4.0/**`、`v8.5.0/**`、`.tmp_v850_distill/**` 上。
- **`__pycache__` 报告项**：`evidence/BOMBRULE2/__pycache__/` 不存在；但 `evidence/BOMBRULE2/scripts/__pycache__/analyze.cpython-310.pyc`（22,932 B）**存在**，按授权**只报告、未删除**（它已在 `BOMBRULE2/manifest.json` 的逐文件清单内，故其字节身份并未丢失）。
- **两处未入册缺口（如实列出）**：`v8.4.0/PREREGISTRATION_BOMBRULE2_001.md`（3,703 B）与 `v8.5.0/PREREGISTRATION_DISTILL_001.md`（4,073 B）既不在最小集内、也不在两份 manifest 的覆盖范围内 ⇒ **这两个文件目前没有可校验的字节身份记录**。它们按设计「禁改」，但若需要可校验，应单独补登 2 行。

> **一句话结论**：本轮补齐了 v8.4.0 / v8.5.0 的资产册缺口 —— 追加 **63 条**（v8.4.0 28 条：2 文档 + 5 顶层证据 + 21 评测；v8.5.0 35 条：3 文档/记录 + 11 顶层证据 + 18 评测/summary + 3 checkpoint），`sha256sum -c` **271 OK / 1 FAILED**，**新增 63 条全部 OK**、整体 FAILED 数**保持为 1**（仍只有已知允许的 `v4.2.0/journal.jsonl`）。187 个过程件按 §20.2 不入册，但其逐文件 sha256 已被入册的两份 `manifest.json` 覆盖（239/239 实测相符）；两轮目录**全程只读、0 修改**，`__pycache__` 只报告未删除。

## 21. v8.9.0–v8.10.0 取证轮资产（2026-11-22）

> **为什么有这一节**：AGENTS.md §8.2 要求「新增/移动/删除任何资产，必须同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`」。v8.9.0（RULEDIST2）与 v8.10.0（HAZFIX）两轮在本节追加前 **一行都没有入册** —— 追加前 **274 行** 对 `v8\.9\.0|v8\.10\.0|RULEDIST2|HAZFIX|v890_ruledist|preview_rule` 的匹配是 **0 命中**。本节与 `ASSET_SYNC.sha256` 末尾追加的 **10 行**成对登记。
>
> **两轮性质**：两轮都是 **`S-data`（P0 不成立、未训练）** 的取证轮。v8.9.0 在 v8.6.0 的 **5,040** 个 bomb-state 上离线算 R5 规则目标，P0（三段各 ≥50 正/负例）**不成立** ⇒ 只发布规则目标与统计，**未训练、未评测**；v8.10.0 是「补回 `nearest_all_px` 再蒸馏」的预注册件，**证据尚未发布**（见 §21.4）。
>
> **只读边界**：本轮只 **append** 两个清单文件。`Document/Update/Alpha/v8.9.0/**`、`Document/Update/Alpha/v8.10.0/**` 全程**只读** —— 0 修改 / 0 改名 / 0 删除 / 0 新建；登记结束后两目录共 **35** 个文件（v8.9.0 34 + v8.10.0 1）的 `st_mtime_ns` 与登记开始时**逐一相同**（见 §21.4）。**未启动引擎**（本轮 0 次引擎调用、0 个游戏进程）。
>
> **实测口径**：10 条的 `bytes` 与 `sha256` **全部现场实测**（Windows `Get-FileHash -Algorithm SHA256`），并用 WSL `sha256sum -c` 独立复算。两轮自带的 `manifest.json` / `verify.json` 内联 sha256 **只用于交叉核对，不代替实测**（核对见 §21.4）。
>
> **最小集口径**：只登记「顶层文档 + 核心证据 + 冻结预注册」。过程件（`logs/`、`runs/`、`scripts/` 与顶层运行时元数据）不入册，理由与**字节身份仍有锚**的证明见 §21.2。

### 21.1 登记范围（**10 条** = v8.9.0 **9** 条 + v8.10.0 **1** 条）

| 分组 | 条数 | 字节合计 | 位置 |
| --- | ---: | ---: | --- |
| §21.1.1 v8.9.0 RULEDIST2 判据文档与核心证据 | 8 | 2,618,189 | `Document/Update/Alpha/v8.9.0/evidence/RULEDIST2/**` |
| §21.1.2 两轮冻结预注册 | 2 | 8,877 | `Document/Update/Alpha/v8.{9,10}.0/PREREGISTRATION_*.md` |
| **合计** | **10** | **2,627,066** | — |

#### §21.1.1 v8.9.0 RULEDIST2 判据文档与核心证据（8 条 / 2,618,189 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.9.0/evidence/RULEDIST2/RUN.md` | 17,008 | `5f508bbb37ced66b2c6093567c799108399b83bb3520d375542bbaf82080d54b` | v8.9.0 运行记录（285 行）：状态**停在 `S-data`**，5,040 状态全部算完、**未训练未评测**、无统计推断；含命令表、退出码与两距离取证口径 |
| 2 | `Document/Update/Alpha/v8.9.0/evidence/RULEDIST2/manifest.json` | 4,424 | `2fdabd9fb0713228da74fcf01bc493249ffcc17137c400dd6593029bc992f6f4` | **清单锚**：`files{}` 逐条 `{sha256, bytes}` 覆盖 `evidence/RULEDIST2/**` 除自身与 `verify.json` 的 **31** 个文件；`file_count=31`、`total_bytes=2,708,713`、`excludes="manifest.json"`；本节实测 **31/31 全等** |
| 3 | `Document/Update/Alpha/v8.9.0/evidence/RULEDIST2/verify.json` | 3,578 | `86e80cd2e1988858cd6ed0c5b3f896a03b88d908b968d118b0e96c719d1dc2cb` | **独立复核结论**：`scripts/verify.py` 重算的 `checks=30` 全 `ok`、`failed=0`、`all_pass=true`；`p0_all_pass=false`、`p0_decision="S-data (publish targets only)"`；内含 `manifest_sha256` 与 `targets_sha256` 两个独立锚 |
| 4 | `Document/Update/Alpha/v8.9.0/evidence/RULEDIST2/p0_forensics.md` | 11,983 | `f63fce702fa8973bab0d12e0a42e0793cec11299bf263a41797733bcdac4d9fc` | P0 取证正文（212 行）：三段正/负例 **24/118/11** 与 **600/3,260/1,027**，判定 P0 不成立；归因为**状态分布**而非数据丢失，并写明 `nearest_all_px` 可离线恢复但**不能救回 P0** |
| 5 | `Document/Update/Alpha/v8.9.0/evidence/RULEDIST2/rule_targets_v86.jsonl` | 2,543,921 | `c013133c000c066aeecc9b97420d3022a6ce71630794e0037095adc51b4ccbae` | **核心证据**：**5,040 行** × 23 字段的逐状态规则目标（`nearest_all_px`、`rule_bomb`、`rule_bomb_obs`、`bombs_remaining` 等）；与 `rule_target_stats.json` 的 `targets_bytes` / `targets_sha256` **逐位相同** |
| 6 | `Document/Update/Alpha/v8.9.0/evidence/RULEDIST2/rule_target_stats.json` | 9,716 | `b9dc09c47eed9fa046637c1974f149970610f94455d33343cb5869801fa12454` | 三段统计与 P0 判定：`status="s_data"`、`p0.all_pass=false`、`p0.decision="S-data (publish targets only)"`；源数据锚 `labels_bombdag.jsonl` 的 `sha256=e1f47805…` / 35,189,092 B / 5,040 状态（**只读**） |
| 7 | `Document/Update/Alpha/v8.9.0/evidence/RULEDIST2/fidelity_d3.json` | 26,627 | `a52daf1044352586d83d229cedfcc28bced84e933941ae9514ad12af445729d6` | 保真门 d3：`all_pass=true`、`verdict="PASS: §2.1-§2.5 hold on this binary and data dir"`、`engine_expected_sha256=691bfd77…`（与 §2 的 `rl_v3` SHA **一致**）、`mode=fidelity` / `seed=12` |
| 8 | `Document/Update/Alpha/v8.9.0/evidence/RULEDIST2/audio_audit.json` | 932 | `913a7c2174d5b78e4d988c0e616266a840ebf7dc9ee8b3051f7663cb72d80736` | §9.3 静音审计：`cfg_files=1 / muted=1 / audible=0`、`all_muted=true`，作用域 `.tmp_v890_ruledist/data` ⇒ 本轮**未新增 `audible`** |
| | **小计** | **2,618,189** | 8 条 | |

#### §21.1.2 两轮冻结预注册（2 条 / 8,877 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.9.0/PREREGISTRATION_RULEDIST2_001.md` | 4,453 | `0a0121d54b2160af2e728d567ee74686819206ccd0dc53c52e503e3d7d58b136` | RULEDIST2 冻结预注册（**FROZEN**，在任何 v8.9.0 运行之前）：定义 P0 = 三段各 **≥50** 正/负例，任一不成立即 `S-data` |
| 2 | `Document/Update/Alpha/v8.10.0/PREREGISTRATION_HAZFIX_001.md` | 4,424 | `fbc2f1937d44166db2f81c9984b3fd2da6de57a8c230240dd5d9bf28eb06ce7b` | HAZFIX 冻结预注册（**FROZEN**）：重放 v8.6 的 5,040 状态补回 `nearest_all_px` 后再做规则蒸馏；其 §4 列出的 `evidence/HAZFIX/**` 产物**尚未产出**（见 §21.4） |
| | **小计** | **8,877** | 2 条 | |

**为什么把两份 `PREREGISTRATION_*.md` 也登记**：§20.4 曾把同类的 v8.4.0/v8.5.0 冻结件列为「未入册缺口」，而 `ASSET_SYNC.sha256` 第 **273–274** 行其实**已经收录**了这两份文件 ⇒ 既有口径事实上是「冻结预注册也入册」。本节沿用该口径，直接登记两轮冻结件，避免再造新的缺口（v8.10.0 侧尤其重要：它是该轮**唯一**存在的文件）。

### 21.2 跳过说明（未入册 **25** 个文件 / **98,526 B**，逐组给理由）

两目录实测：v8.9.0 **34** 个文件 / **2,721,168 B**（入册 9 / 2,622,642 B）；v8.10.0 **1** 个文件 / **4,424 B**（入册 1 / 4,424 B）。v8.9.0 侧差额 34 − 9 = **25 个文件 / 98,526 B** 即下表；v8.10.0 侧 **0 个未入册**。

| 组 | 文件数 | 字节 | 不入册理由 |
| --- | ---: | ---: | --- |
| `evidence/RULEDIST2/` 顶层其余 8 件：`ckpt_assertions{,_final}.json`（4,880 B×2）、`ckpt_manifest.json`（1,070）、`commands.tsv`（1,731）、`p0_recount_independent.json`（1,560）、`preflight.json`（5,086）、`process_residue{,_final}.json`（617 B×2） | 8 | 20,441 | 断言 / 预检 / 残留进程 / 命令行记录：都是**汇总表的展开或运行时元数据**，结论已由入册的 `RUN.md`、`verify.json`、`rule_target_stats.json` 承载；逐件 `{bytes, sha256}` 已在入册 `manifest.json` 内。 |
| `evidence/RULEDIST2/logs/`（10 份 `.out`） | 10 | 6,663 | 运行日志（单件 105–2,548 B），**可重跑重建的输出**，不是资产；统计结论已进 `verify.json`（30 项检查）与 `RUN.md`。 |
| `evidence/RULEDIST2/runs/v890-fidelity/`（4 件） | 4 | 56,117 | 其中 `fidelity.json`（26,627 B）与顶层入册 `fidelity_d3.json` **逐字节相同**（同为 `a52daf10…`）；`run.log`（2,722 B）与 `logs/d3-fidelity.out` **逐字节相同**（同为 `5d84e813…`）⇒ 两份是**纯副本**，已由入册件覆盖；`gate.json`（26,605 B）与 `w12-stderr.log`（163 B）是保真门的中间表 / stderr 快照，逐件 sha256 在入册 manifest 内。 |
| `evidence/RULEDIST2/scripts/`（3 件：`exec.sh`、`p0_recount.py`、`verify.py`） | 3 | 15,305 | 生成器与独立复核脚本本身不是证据；其 sha256 全在入册 `manifest.json` 的 `scripts/*` 条目里（`scripts/verify.py` 正是入册 `verify.json` 的产出者，`producer` 字段可对上）。 |
| **小计** | **25** | **98,526** | 33 − 8 = 25 |

**为什么「不入册」不等于「没有字节身份」**：入册的 `manifest.json` 覆盖 `evidence/RULEDIST2/**` 中除自身与 `verify.json` 之外的 **31** 个文件，本节已实测复核 **31/31 全等、0 处不符**（见 §21.4）。25 个未入册文件**全部**落在这 31 个之内（25 未入册 + 6 已入册且被 manifest 覆盖 = 31），故其逐件 `{bytes, sha256}` 仍可校验。

**`__pycache__` / `*.pyc` 实况（按要求只报告、不删）**：

- `Document/Update/Alpha/v8.9.0/` 全树 **没有** `__pycache__/`、**没有** `*.pyc`（`Get-ChildItem -Recurse -File -Force` 实测 34 个文件，0 命中）；`evidence/RULEDIST2/scripts/` 下只有 3 个文本脚本（`exec.sh`、`p0_recount.py`、`verify.py`），无字节码缓存；
- `Document/Update/Alpha/v8.10.0/` 只有 1 个 `.md`，**没有** `__pycache__/`、**没有** `*.pyc`；
- 若日后出现，按本节口径**一律不入册**（Python 字节码缓存不是证据）。

### 21.3 与 §1–§20 不重复的说明（实测已核）

- **既有 274 行 0 命中**：追加前对全部 274 行做 `v8\.9\.0|v8\.10\.0|RULEDIST2|HAZFIX|v890_ruledist|preview_rule` 匹配 → **0 命中**；`v8.4.0`/`v8.5.0`（§20）是**不同版本/不同目录**，本轮 10 条与之无交集。
- **路径 10 唯一 / 哈希 10 唯一**：10 条内部 sha256 逐条比对 **0 组重复**。
- **与既有 274 行 0 交集**：10 个路径、10 个 sha256 与既有 **274 个路径 / 271 个唯一哈希**逐一比对，**路径交集 0、哈希交集 0** → 本次没有引入任何新的同哈希对。
- **行数与格式**：274 + 10 = **284 行**；仍是 **LF / 无 BOM**，末行以 `\n` 结束，每行 `64 hex + 两个空格 + 相对路径`，**格式异常行 0 条**。
- **`ASSET_SYNC.md` 本身不在校验表内**（`Select-String 'ASSET_SYNC'` → 0 行）→ 本节追加不会让任何既有行 FAILED，也不需要为自身重钉。
- **Git 跟踪状态（实测）**：`git status --porcelain -- Document/Update/Alpha/v8.9.0 Document/Update/Alpha/v8.10.0` → 两目录均为 `??`（整目录 untracked）；`git ls-files -- <两目录>` = **0**，`git ls-files | grep -E 'v8\.9\.0|v8\.10\.0'` = **0** ⇒ 这些证据**无法由 Git 重建**，本节 + `ASSET_SYNC.sha256` 是它们目前的字节身份记录。本轮按约束**未** `git add` / `git commit`，分支仍为 `v400`。
- **一处既存口径不一致（只报告，未回改 §20 正文）**：§20.3 / §20.4 称校验表「**272 行**」且两份 `PREREGISTRATION_*.md` 属「未入册缺口」，但本轮实测 `ASSET_SYNC.sha256` 实际为 **274 行**，其第 **273–274** 行**正是** `v8.4.0/PREREGISTRATION_BOMBRULE2_001.md` 与 `v8.5.0/PREREGISTRATION_DISTILL_001.md` ⇒ 这两份文件**已被登记**，§20.4 那句「目前没有可校验的字节身份记录」**已过期**。本轮按「只追加」的约束不回改 §20 正文，故在此如实记录，供后续轮次订正。

### 21.4 校验结论

`cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256`（WSL）：**284 行 = 283 OK / 1 FAILED**（`sha256sum` 退出码 **1** —— 仅因那 1 条；`283 + 1 = 284` 校验行全部参与）。

- **基线（追加前 274 行）**：同一命令实测 **273 OK / 1 FAILED**；追加后 **283 OK / 1 FAILED** ⇒ **10 条新行 10/10 OK，整体 FAILED 数没有新增**。
- **唯一 FAILED 是已知并允许的那一条**：`Document/Update/Alpha/v4.2.0/journal.jsonl`（§9.8 记录的活动文件漂移）。本轮**未改该行、未重钉**。
- **与产出方 manifest 的交叉核对**：入册 `v8.9.0/evidence/RULEDIST2/manifest.json` 的 `files{}` 内联 **31** 条 `{bytes, sha256}` 与现场重算比对 —— **31/31 全等，0 处不符**；其 `file_count=31` / `total_bytes=2,708,713` 同时对上（31 件字节和 = 2,708,713；加 `manifest.json` 4,424 + `verify.json` 3,578 = **2,716,715** = 目录实测总字节）。
- **三处独立锚（互不依赖）**：① 入册 `verify.json` 的 `manifest_sha256` = `2fdabd9f…` = 本表 `manifest.json` 实测值；② 同文件 `targets_sha256` = `c013133c…` = 本表 `rule_targets_v86.jsonl` 实测值，且 `rule_target_stats.json` 的 `targets_sha256` 与 `targets_bytes=2,543,921` 与之**逐位相同**；③ `verify.json` 自报 `checks=30 / failed=0 / all_pass=true`，其中 `manifest_files=31`、`manifest_files_with_wrong_hash=[]`、`target_rows=5,040`、`target_rows_with_wrong_locator=0`、`target_rows_with_different_nearest_all_px=0`。
- **两轮结论口径（只登记，不重判）**：`rule_target_stats.json` 的 `p0.all_pass=false` 与 `verify.json` 的 `p0_all_pass=false`、`p0_decision="S-data (publish targets only)"` 三处一致 → 本轮**确为 S-data、未训练**；`rule_target_stats.json` 另有 `status="s_data"` 与 `all_pass=true`（后者指脚本自身跑通，**不是** P0 通过）。三段实测：`bombs_remaining` 3 → **24** 正 / **600** 负、2 → **118** / **3,260**、1 → **11** / **1,027**，仅 60 px 段满足预注册的 ≥50 正例门槛。
- **保真门与音频**：`fidelity_d3.json` 的 `all_pass=true`、`verdict="PASS: §2.1-§2.5 hold on this binary and data dir"`，且 `engine_expected_sha256=691bfd77…` 与 §2 记录的 `rl_v3` SHA **一致**；`audio_audit.json` 的 `counts={cfg_files:1, muted:1, audible:0}`、`all_muted=true` ⇒ 符合 §9.3。
- **`v8.10.0` 的缺口（如实列出，本轮最重要的一条）**：`Document/Update/Alpha/v8.10.0/` 实测**只有 1 个文件**（`PREREGISTRATION_HAZFIX_001.md`，4,424 B）—— **`evidence/HAZFIX/` 目录不存在**；在 `Document`、`tools`、`touhou_rl`、`tests`、`configs` 下递归搜索 `preview_rule*` **0 命中**（`preview_rule.json` 与 `preview_rule.py` **均不存在**）。该预注册 §4 列出的应有产物（`hazard_fields.jsonl`、`replay_reconciliation.json`、`rule_target_stats.json`、`train_A_*.json`、`train_B_*.json`、`eval_A_d{300,600,900}_seed*.jsonl`、`eval_B_d600_seed*.jsonl`、`bomb_usage.json`、`RUN.md`、`manifest.json`）**一件都不在盘上** ⇒ **HAZFIX 轮的证据尚未发布，本轮只登记其冻结预注册，v8.10.0 侧 0 条证据可登记**。这不是登记遗漏，是证据本身未产出。
- **一处文献口径分歧（只报告，不裁定）**：v8.10.0 预注册 §0 称 v8.9 的 P0 统计（0/10/24 正例）是「**数据缺失的假象**，已作废」，而 v8.9 自己的 `p0_forensics.md` 明确反对该解读 —— 它称原因是**状态分布而非数据丢失**，并写明 recovering it **does not rescue P0**。两份都是冻结/已发布件，本节**按原样登记字节身份，不对分歧做裁定**；如需结论应由研究端复核。
- **登记期无漂移**：两目录 **35** 个文件的 `st_mtime_ns` 在登记开始与登记结束时**逐一相同**（0 个变化）；本轮**没有**任何写入落在 `v8.9.0/**`、`v8.10.0/**` 上。

> **一句话结论**：本轮把 **v8.9.0（RULEDIST2）9 条 + v8.10.0（HAZFIX）1 条 = 10 条**成对登记进 `ASSET_SYNC.md` §21 与 `ASSET_SYNC.sha256`，`sha256sum -c` **283 OK / 1 FAILED**，**新增 10 条全部 OK**、整体 FAILED 数**保持为 1**（仍只有已知允许的 `v4.2.0/journal.jsonl`）。v8.9.0 的 25 个过程件按 §21.2 不入册，但逐文件 sha256 已被入册 `manifest.json` 覆盖（31/31 实测相符）；**v8.10.0 的 `evidence/HAZFIX/` 根本不存在，0 条证据可登记**，只登记其冻结预注册 —— 这是本轮最需要后续动作的缺口。两轮目录**全程只读、0 修改**，`__pycache__`/`*.pyc` 实测 0 个。

### 21.5 v8.10.0 证据补登（2026-11-22）

> **§21.3 / §21.4 中「v8.10.0 只有 1 个文件」的记录已被本小节取代。** 盘上实测，该类陈述共 **4 处**，全部按「不改既有记录」的约束**原样保留**（它们在 12:00 登记时**属实**，只是被 11:54–12:03 的实际产出取代）：§21.1 表第 2 行（第 1220 行，「`evidence/HAZFIX/**` 产物尚未产出」）、§21.2（第 1227 行「v8.10.0 1 个文件 / 4,424 B……v8.10.0 侧 0 个未入册」、第 1242 行「只有 1 个 `.md`」）、§21.4（第 1265 行「实测只有 1 个文件……`evidence/HAZFIX/` 目录不存在……0 条证据可登记」、第 1269 行「`evidence/HAZFIX/` 根本不存在，0 条证据可登记」）。其中 **§21.4 的两处**即通常所说的「§21.3/§21.4 记录」；**§21.3 自身**（第 1245–1253 行）是「与 §1–§20 不重复」的核验，**没有**「v8.10.0 只有 1 个文件」的陈述（如实说明，避免引错位置）。**自本节起，v8.10.0 的证据以本节为准。**

**为什么会有这次补登**：§21 与 `ASSET_SYNC.sha256` 第 **284** 行完成于 **12:00** 前后，当时 v8.10.0 侧只有冻结预注册 `PREREGISTRATION_HAZFIX_001.md`（4,424 B）；执行器 `tools/rl/v8a0_hazfix.py` 的重放、对账、规则统计、保真与审计产物在 **11:54:27–12:03:00** 才落盘（见各件 `created_at`）。§21.4 把「证据尚未发布」列为本轮**最重要的缺口**，本节按实测关闭它。**本轮只做登记**：0 次引擎调用、0 个游戏进程，`v8.10.0/**` 的 29 个文件**一个字节都没有改动**（见 §21.5.6）。

#### 21.5.1 盘上实测（登记时）

`Get-ChildItem Document\Update\Alpha\v8.10.0 -Recurse -File`：

| 范围 | 文件数 | 字节 |
| --- | --- | --- |
| `Document/Update/Alpha/v8.10.0/PREREGISTRATION_HAZFIX_001.md`（§21.1 第 2 条已登记） | 1 | 4,424 |
| `Document/Update/Alpha/v8.10.0/evidence/HAZFIX/**`（本节新增） | **28** | **15,995,936** |
| **合计** | **29** | **16,000,360** |

**「27 个文件 / 15,991,979 B」与「28 个 / 15,995,936 B」不矛盾**：前者是 `evidence/HAZFIX/manifest.json` 的**自述口径** —— 该文件写着 `"file_count": 27`、`"total_bytes": 15991979`、`"excludes": "manifest.json"`，即**除 manifest 自身**的计数。实测：`manifest.files{}` 共 **27** 条 `{bytes, sha256}`，逐条与现场重算比对 **27/27 全等、0 处不符**，`bytes` 求和 = **15,991,979**（与 `total_bytes` 逐位相同）；再加 `manifest.json` 自身 **3,957 B** ⇒ **28 个文件 / 15,995,936 B**。盘上 `HAZFIX/**` 里**不在** `manifest.files` 中的文件**只有 `manifest.json` 一个** ⇒ **无缺件、无多余件**，两组数字只是范围不同。

#### 21.5.2 登记范围（**16** 条 sha256，合计 **15,924,892 B**）

| # | 路径（相对仓库根） | 字节 | sha256 | 作用 / 关键读数 |
| --- | --- | --- | --- | --- |
| 1 | `Document/Update/Alpha/v8.10.0/evidence/HAZFIX/RUN.md` | 13,244 | `b4a9b31ec367af6a02db0e454a1e9e8e276c4e5e8fe64f3bf34f1c4b49bba56a` | 223 行运行记录：P0 判定、对预注册 §0 两条前提的纠正、重放对账、规则分布、门、偏差清单、「不能声称什么」 |
| 2 | `…/HAZFIX/audio_audit.json` | 2,388 | `691f11a31db0c8aa2e91e9eee1e61b0ece77adc8cf12db477c94b7b6f1abf953` | AGENTS.md §9.3 主运行审计：`cfg_files=12`、`muted=12`、`audible=0`、`errors=0`、`all_muted=true` |
| 3 | `…/HAZFIX/audio_audit_smoke.json` | 764 | `1c0db1fb51a6c8857acd21144144e218928e02738e6e9a60980dbebdd1ad48d1` | `smoke/data` 审计：1/1 muted、audible 0 |
| 4 | `…/HAZFIX/audio_audit_work.json` | 767 | `63dd5c2a7ba7bff44b32e9646c241c7a6b8704178b4ae5ceb811a1bf9ea4c1f5` | `work/data`（fidelity）审计：1/1 muted、audible 0 ⇒ 与 #2/#3 合计 **14/14 muted、audible 0、decode error 0** |
| 5 | `…/HAZFIX/ckpt_assertions.json` | 3,384 | `3df851107024df2a83be92941971d6b62da47db11048122452eb34e877072938` | `phase=start`（11:54:37）：3 个 `round2.pt` 逐字节 match（`4abee069…`/`a93746bb…`/`c69960fc…`，各 34,387 B）、引擎 `691bfd77…` match、`all_pass=true` |
| 6 | `…/HAZFIX/ckpt_assertions_final.json` | 3,384 | `465695f6e3d392adba39ced44706d2fd47b206883a6cab44f7a5f7b24cf81df3` | `phase=final`（12:02:16）；`diff` 实测**仅 `created_at` 与 `phase` 两行不同**，其余逐字节相同 ⇒ 全程 checkpoint 未漂移 |
| 7 | `…/HAZFIX/commands.tsv` | 7,888 | `7afe699bd6d8ce825861a83d1242f5afe69a7cf741d15d4167856a466f46081f` | 每条命令的 exit / wall / argv 记录 |
| 8 | `…/HAZFIX/exec.sh` | 6,244 | `d5b94c63d56d908ecaf9c6124ecb0de39a229a76db266fab85ddbd47ed228b52` | 本轮一键执行脚本（`p0` 阶段以 rc=5 收尾属**预期**，见 RUN.md §7 偏差清单第 7 条） |
| 9 | `…/HAZFIX/fidelity_d3.json` | 26,638 | `d86809c594eb73e93ba90f255994ccf2cc186919bf45467feb3968fa8f60bebf` | D3 保真门 **5/5 PASS**：`2165/2165` 前缀 checkpoint 0 mismatch、final frame **8,663**、R24 常量 **18/18**、2 个 duplicate 轨迹 bitwise 相同、`deviation_log=[]`；引擎 `691bfd77…` match |
| 10 | `…/HAZFIX/hazard_fields.jsonl` | 15,842,958 | `b6e800b392b682589f4fdc9db58d53b64823c1081d053518f309b067b57a5a93` | **5,040 行**，一行一状态；补记 `nearest_all_px`、最近 12 颗的距离/速度、`player_state` 等 v8.6.0 未存字段（全轮最大件，占 28 件的 **99.04%**） |
| 11 | `…/HAZFIX/manifest.json` | 3,957 | `ed8a37d1a7cfbd1f09149fb006a067093bb9a41df315b765cf14e8ae1cc8fea1` | 逐文件清单：`files{}` **27** 条 `{bytes, sha256}`、`total_bytes` 15,991,979、`excludes="manifest.json"`；**§21.5.3 那条合并说明的权威覆盖件** |
| 12 | `…/HAZFIX/p0.json` | 1,065 | `b3a4db54269ac8b3503ec073bb52af919424d0311e6e9469f3577bd8cbad3bac` | P0 判定：①`hit_rate=1.0`（5,040/5,040）**PASS**；②三段正例 24/118/11 **FAIL** ⇒ `p0_pass=false`、`verdict="S-data"` |
| 13 | `…/HAZFIX/preflight.json` | 3,747 | `17d5a448fabbaf528440d23a1f025105c33f311da7bc4c2c96c8f04a560a83dd` | 开跑前自检：**8/8** pin match、引擎 `691bfd77…` match、预注册 `fbc2f193…` match（4,424 B）、v8.6.0 标签 `e1f47805…` match 且 `labels_reused_not_recollected=true`、`relabeled_actions=0`、`SDL_AUDIODRIVER=dummy`、`feature_dim=106`、`all_pass=true` |
| 14 | `…/HAZFIX/process_residue.json` | 761 | `bb865bb44ab7e154f47e946ce23dccda5b531c843b3e9e626c192860622b8aaf` | 本进程残留 **0**（`--data-dir-filter .tmp_v8a0_hazfix`）；宿主另有 **61** 个属于**别轮**的 `th08-modern`，明确未触碰 |
| 15 | `…/HAZFIX/replay_reconciliation.json` | 3,230 | `68beae160544f8ad6bafdb830006831d25e71c40d75d32b98f177678fa939cbc` | 重放对账：5,040 重放 / **5,040 匹配**；`features_sha256_k12` **1.000000**（阈值 0.99）；`frame`、`bombs_remaining` 各 5,040/5,040；重放实测 `nearest_all_px` vs v8.6.0 标签 **5,040/5,040** 且 \|Δ\| 的 mean/median/min/max 全 **0.0**；`missing_keys=[]`、`feature_misses=[]` |
| 16 | `…/HAZFIX/rule_target_stats.json` | 4,473 | `8c81c207340831ddf5132fdc2fba9769194d3b3421a09135dcbe0856654aaf04` | R5 `staged` 阈值 80/60/40 px、严格 `<` 且 `bombs_remaining > 0`；三段 `n`=624/3,378/1,038、正例 **24/118/11**、负例 600/3,260/1,027、正例率 3.8462%/3.4932%/1.0597% ⇒ `p0_segment_pass=false`；`empty_bullet_states=0`、`states_with_laser=0`、`INVULNERABLE=3,416` |

> **16 条合计 15,924,892 B**；加上 §21.5.3 合并覆盖的 12 个 cell summary **71,044 B** ⇒ **15,995,936 B** = `evidence/HAZFIX/**` 实测总字节。**28 个证据件 100% 有字节身份记录**（16 条逐条 + 12 个由入册 `manifest.json` 覆盖）。

#### 21.5.3 12 个 `replay/seed*_w*.json` 合并为一条说明

`evidence/HAZFIX/replay/` 下 12 个 cell summary（seed ∈ {0,1,2} × window ∈ {13,14,15,16}，每 cell 420 状态 / 420 行、含 gate、cfg、wall_s）**共 71,044 B**：`seed0_w1{3,4,5,6}.json`、`seed1_w1{3,4,5,6}.json`、`seed2_w1{3,4,5,6}.json`（5,899–5,933 B/个）。

**12 个 cell summary 已由 `manifest.json` 覆盖**，故不逐条占用 `ASSET_SYNC.sha256` 行：`manifest.json` 的 `files{}` 完整收录这 12 条的 `{bytes, sha256}`，与现场重算 **12/12 全等**；而 `manifest.json` 本身已在 §21.5.2 第 **11** 条入册 ⇒ 身份链 **`ASSET_SYNC.sha256` → `manifest.json` → 12 条 cell summary** 闭合。这与 §21.2 处理 v8.9.0 过程件、§21.4 记录「25 个过程件由入册 `manifest.json` 覆盖」的口径**一致**。

#### 21.5.4 交叉核对（登记时实测，只登记不重判）

| 核对项 | 实测 |
| --- | --- |
| `manifest.json` 自述 vs 盘上 | `files{}` **27/27** 的 `{bytes, sha256}` 全等；`bytes` 求和 = `total_bytes` = 15,991,979；盘上多出的仅 `manifest.json` 自身 |
| 预注册未被改动 | `preflight.json`、`ckpt_assertions.json`、`ckpt_assertions_final.json` 三处独立记录预注册 `fbc2f1937d44166d…` / 4,424 B，与 §21.1 第 2 条入册值**逐位相同** ⇒ HAZFIX 轮**未改预注册一字** |
| 引擎身份 | `preflight` / `ckpt_assertions` / `fidelity_d3` 三处均 `691bfd77b7eca336…` 且 `match=true`，与 §2 记录的 `rl_v3` SHA 一致 |
| 行为策略身份 | 三个 `round2.pt` = `4abee069…` / `a93746bb…` / `c69960fc…`，start 与 final 两相**均 match**，各 34,387 B |
| 独立复算一致性 | 重放时行为策略**实际**触发 R5 **153** 次，按阈值分档 `{80 px: 24, 60 px: 118, 40 px: 11}` = **24+118+11 = 153**，与 `rule_target_stats.json` 离线重算的正例数**逐个相同** ⇒ 规则实现与 `v800_bomb.rule_gate` 语义一致 |
| 音频（§9 硬约束） | 三处审计合计 **14/14 muted**、`audible=0`、`errors=0`、`all_muted=true`；`launch_env` 记录 `SDL_AUDIODRIVER=dummy`、`DISPLAY`/`WAYLAND_DISPLAY` 为 null；共享审计件 `v4.3.1/evidence/R14/audio_cfg_audit.json` 明确**未触碰** |
| 进程残留 | 本轮残留 **0**；宿主 61 个 `th08-modern` 属别轮，未 `pkill -f`、未触碰 |
| 未重采标签 | `preflight.json` 的 `labels_reused_not_recollected=true`、`relabeled_actions=0`；v8.6.0 标签 `e1f47805…` 与 v8.9.0 独立复算的 pin 一致（未动） |

#### 21.5.5 本轮结论口径（照录，不重判）

* **`S-data`，未训练、未评测**：`p0.json` 的 `p0_pass=false` / `verdict="S-data"` 与 `rule_target_stats.json` 的 `p0_segment_pass=false` 一致。本轮**没有产生任何 checkpoint**，也**没有任何评测 episode**，因此**没有** `train_*` / `eval_*` 文件 —— 这是预注册 §2.1 的门不成立，**不是执行失败**。
* **P0② 在这条 5,040 状态轨迹上不成立是分布事实**：`bombs=3` 仅 624 状态（行为策略几乎立即放雷）、`bombs=1` 阈值只有 40 px（1,038 状态里 11 个跨过），只有 `bombs=2`（3,378 状态 / 60 px）凑到 118 个正例。要过 P0 须先改**取样设计**（按 bombs 分层、提高 quota、或改阈值），**不是**把同样 5,040 状态再跑一遍。
* **RUN.md 对预注册 §0 两条前提的纠正**（预注册**未改一字**，本节不重判，只登记其存在）：obs 对 106 维特征**无损**（5,040/5,040 特征哈希一致），`nearest_all_px`（全子弹）与 `nearest_lethal_px`（仅致命）本不该相等（44.36% 的状态有更近的非致命子弹）；`behaviour.nearest_all_px` 在每行里 ⇒ 规则目标**本来就可离线算**。⇒ 预注册的 **B 臂**（用最近 12 颗重算 106 维）**不是可检验因子**（会与 A 臂逐位相同）。
* **不能从本轮推出**：「规则蒸馏可行/不可行」（没训练任何模型）、「v8.6 的 obs 有 bug」（恰恰相反）、「观察退化已被修复」（本轮补的是**记录字段**）、「通关」。

#### 21.5.6 登记期零漂移（`v8.10.0/**` 只读）

登记**开始前**与**结束后**各采集一次 `v8.10.0/**` 全部 **29** 个文件的 `{size, mtime}`（`LastWriteTimeUtc`，精确到 0.1 µs 的 `fffffff` 位）并逐项比对：**29/29 的 `size` 与 `mtime` 完全相同，0 个变化**。本轮对 `v8.10.0/**` 的写操作 = **0**（只读 `Get-ChildItem` / `sha256sum` / `diff` / `head` 与 `manifest.json` 解析）；未新建、未改名、未删除、未移动任何文件；`__pycache__` / `*.pyc` **0 个**。写入**只落在** `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256` 两个清单文件上。

#### 21.5.7 校验结论

`cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256`（WSL）实测：

* **追加后**：**300 行 = 299 OK / 1 FAILED**，`sha256sum` 退出码 **1**（仅因那 1 条）。`299 + 1 = 300` 校验行全部参与。
* **基线（追加前 284 行）**：**283 OK / 1 FAILED** ⇒ **16 条新行 16/16 OK，整体 FAILED 数没有新增（仍为 1）**。单独对本节 16 行做 `sha256sum -c`：**16/16 OK，退出码 0**。
* **唯一 FAILED 仍是已知并允许的那一条**：`Document/Update/Alpha/v4.2.0/journal.jsonl`（§9.8 记录的活动文件漂移）。本轮**未改该行、未重钉、未触碰该文件**。
* **格式**：300 行仍为 **LF / 无 BOM**、末行以 `\n` 结束、每行 `64 hex + 两个空格 + 相对路径`、**空行 0 条、格式异常行 0 条**；新增 16 条的路径与哈希在表内**各自唯一**，与既有 284 行**路径交集 0、哈希交集 0**。
* **本节不重钉 `ASSET_SYNC.md` 自身**：`ASSET_SYNC.md` 不在校验表内（`grep -c 'ASSET_SYNC' ASSET_SYNC.sha256` = **0**），故追加 §21.5 不会让任何既有行转为 FAILED（与 §21.3 第 1251 行同一口径）。

> **一句话结论**：v8.10.0（HAZFIX）在 12:00 登记时**确实只有 1 个文件**，该缺口**已由 11:54–12:03 的实际产出关闭** —— 本节把 **16 条 sha256（15,924,892 B）+ 1 条覆盖 12 个 cell summary 的合并说明（71,044 B）** 成对写入 `ASSET_SYNC.md` §21.5 与 `ASSET_SYNC.sha256`，使 `evidence/HAZFIX/**` 的 **28 个文件 / 15,995,936 B 全部**具备字节身份记录，`v8.10.0/` 全目录实测 **29 个文件 / 16,000,360 B**。`sha256sum -c` 由 **283 OK / 1 FAILED** 变为 **299 OK / 1 FAILED**（新增 16 条全 OK，FAILED 数仍为 1，唯一 FAILED 仍为已知允许的 `v4.2.0/journal.jsonl`）。本轮**未启动引擎、未 `git add/commit`**，`v8.10.0/**` 的 29 个文件 `{size, mtime}` **逐个零漂移**，本轮**未训练、未评测**（`S-data`）。

## 22. v8.6.0 与 v8.12.0 资产（2026-11-22）

> **为什么有这一节**：AGENTS.md §8.2 要求「新增/移动/删除任何资产，必须同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`」。**v8.6.0（BOMBDAGGER）与 v8.12.0（HYBRID）到本节追加前一行都没有入册** —— 对既有 **300 行**做 `v8\.6\.0|v8\.12\.0|BOMBDAGGER|HYBRID|v860|v8d0` 匹配是 **0 命中**。本节与 `ASSET_SYNC.sha256` 末尾追加的 **131 行**成对登记。
>
> **两轮性质不同**：v8.6.0 是**真采集 + 真训练 + 真评测**（12 格 × 420 状态 = **5,040 状态 / 181,440 标签**，3 个 seed 各训 200 epoch，9 份评测共 **10,800 episode**），结论按预注册判为 **`S-partial`**（P0/P1 成立，P2/P3 不成立，见 §22.4.3）；v8.12.0 是**零训练、零采集**的部署评测轮（`H1`/`H2`/`H3`/`H2s` 四臂，执行器 `tools/rl/v8c0_hybrid.py` 与补跑的 `tools/rl/v8d0_h2s.py` 只做评测），结论 **`S-deploy`**（P1 成立，见 §22.4.3）。
>
> **只读边界**：本轮**只 append** 两个清单文件。`Document/Update/Alpha/v8.6.0/**`（57 个文件）全程只读；`Document/Update/Alpha/v8.12.0/**`（86 个文件）同样全程只读。登记开始前与结束后各采集一次这 **146 个文件**的 `{size, st_mtime_ns}` 并逐项比对：**146/146 完全相同，0 变化**（§22.4.4）。**未启动引擎**（本轮 0 次引擎调用、0 个游戏进程、未 `git add` / 未 `git commit`，分支仍为 `v400`）。
>
> **实测口径**：131 条的 `bytes` 与 `sha256` **全部现场实测**（Python `hashlib.sha256`，1 MiB 分块流式读取），再用 WSL `sha256sum -c` 独立复算 **131/131 OK**。三份产出方 `manifest.json` 的内联 sha256 **只用于交叉核对，不代替实测**（核对见 §22.4.2）。
>
> **两个口径都写（本册一贯做法）**：目录口径 = `Get-ChildItem -Recurse -File` 的实测总数；`manifest.json` 自述口径 = 该文件 `file_count` / `total_bytes` 字段，**`excludes="manifest.json"`**，即「除自身」。三处实测都自洽：v8.6.0 **57 / 133,935,659**（目录）vs **55 / 133,922,602**（manifest，除自身；再加 `manifest.json` 7,658 + 顶层 `PREREGISTRATION_BOMBDAGGER_001.md` 5,399 = 目录口径）；v8.12.0 的 `HYBRID_H2S` **25 / 1,125,666**（目录）vs **24 / 1,122,154**（manifest，除自身；+ `manifest.json` 3,512 + 顶层预注册 4,066 = v8.12.0 目录口径 132,395,489）；`HYBRID` **60 / 131,265,757**（目录）vs **59 / 131,368,485**（manifest，除自身；见 §22.2 的 4 处偏差）。
>
> **时区口径**：本节所有 `mtime` 一律 **UTC**（`os.stat().st_mtime_ns` 与 `LastWriteTimeUtc`）。换算本地（Asia/Shanghai）需 **+8 h**：例如 `BOMBDAGGER/RUN.md` 的 `2026-11-22T04:11:54Z` = 本地 `12:11:54`，与该文件内 `created_at` 的 `+0800` 口径一致。所以 v8.6.0 的 mtime 集中在 **03:06–04:12 UTC（11:06–12:12 本地）**是**正确**的，不是跨时区错误。
>
> **最小集口径**：过程件（v8.6.0 的 `collect/` 12 个 cell summary 与 `smoke/` 2 件；v8.12.0 那个被截断成 **0 字节**的 `eval_H2_d600_seed2.jsonl`）**不入册**，理由与「字节身份仍有锚」的证明见 §22.2。

### 22.1 登记范围（**131 条** = v8.6.0 **46 条** + v8.12.0 **26 条** + v8.12.0/HYBRID **59 条**）

| 分组 | 条数 | 字节合计 | 位置 |
| --- | ---: | ---: | --- |
| §22.1.1 v8.6.0 判据文档与运行记录 | 6 | 190,013 | `v8.6.0/`、`v8.6.0/evidence/BOMBDAGGER/` |
| §22.1.2 v8.6.0 标签集与统计 | 6 | 35,314,102 | `v8.6.0/evidence/BOMBDAGGER/` |
| §22.1.3 v8.6.0 训练报告与 3 个 ckpt 副本 | 7 | 141,172 | `v8.6.0/evidence/BOMBDAGGER/` |
| §22.1.4 v8.6.0 门、预检与审计 | 6 | 48,324 | `v8.6.0/evidence/BOMBDAGGER/` |
| §22.1.5 v8.6.0 九份评测：jsonl + summary 逐文件 | 18 | 98,076,113 | `v8.6.0/evidence/BOMBDAGGER/` |
| §22.1.6 v8.6.0 三个 ckpt 的运行目录原件 | 3 | 117,207 | `.tmp_v860_bombdag/ckpt/seed{0,1,2}/` |
| **v8.6.0 小计** | **46** | **133,886,931** | — |
| §22.1.7 v8.12.0 冻结预注册 | 1 | 4,066 | `v8.12.0/` |
| §22.1.8 `HYBRID_H2S` 的文档、门与审计 | 13 | 77,203 | `v8.12.0/evidence/HYBRID_H2S/` |
| §22.1.9 `HYBRID_H2S` 的 12 份逐窗口 jsonl | 12 | 1,048,463 | `v8.12.0/evidence/HYBRID_H2S/` |
| **v8.12.0 小计** | **26** | **1,129,732** | — |
| §22.1.10 `HYBRID` 的文档、判决与门 | 18 | 239,526 | `v8.12.0/evidence/HYBRID/` |
| §22.1.11 `HYBRID` 的 H1 评测（9 jsonl + 9 summary） | 18 | 98,081,678 | `v8.12.0/evidence/HYBRID/` |
| §22.1.12 `HYBRID` 的 H2 评测（9 份中的 8 份） | 8 | 827,988 | `v8.12.0/evidence/HYBRID/` |
| §22.1.13 `HYBRID` 的 H3 评测（9 份） | 9 | 908,136 | `v8.12.0/evidence/HYBRID/` |
| §22.1.14 `HYBRID` 的 H2s 评测（3 jsonl + 3 summary） | 6 | 31,208,429 | `v8.12.0/evidence/HYBRID/` |
| **HYBRID 小计** | **59** | **131,265,757** | — |
| **合计** | **131** | **266,282,420** | — |

#### 22.1.1 v8.6.0 判据文档与运行记录（6 条 / 190,013 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/RUN.md` | 22,112 | `ef0d3bdb94f5256bf3851cd45748a37837dde6097d1849906001fb4503032d1a` | 389 行运行记录：P0/P1/P2/P3 判定、两分支放雷不变量（honoured / no-op）、偏差清单、对「见雷就按」机制的解释；结论 `S-partial` |
| 2 | `Document/Update/Alpha/v8.6.0/PREREGISTRATION_BOMBDAGGER_001.md` | 5,399 | `2d10c50738cc1918ff60943cc42153293c59580b912653d1c9f2b178c54cb5b5` | 冻结预注册（FROZEN，任何 v8.6.0 运行之前）。**注意**：`ckpt_assertions.json`（start 相）记的是 `8df9bdc7…`，与本值不同 —— 见 §22.4.5 第 1 条 |
| 3 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/manifest.json` | 7,658 | `4a83bea7598149e74662d7f9a00146bdc863cc2c3bea0abdb85883bfd2cbb4f3` | **清单锚**：`files{}` **55** 条 `{bytes, sha256}`、`total_bytes=133,922,602`、`excludes="manifest.json"`；§22.2 那 14 件未入册件的覆盖件；本节实测 **55/55 全等** |
| 4 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/exec.sh` | 9,449 | `f5ac02b9328c4a885f15d054d2e21499f43f21519bb36275348dd7e2a690cc60` | 一键执行脚本：collect / train / eval / publish 各段的可复现入口（与 `commands.tsv` 一起构成命令级复现） |
| 5 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/commands.tsv` | 22,369 | `741423b301b7840ea134e5aeb0a3c5e3cb6d1f21c348b377d6ed3a717423c565` | 逐条命令的 exit / wall / argv 记录（22,369 B） |
| 6 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/runner_collect_8fab2e63.py` | 123,026 | `8fab2e633cf2d4be2cffb8d23380bad296b4f170c8b265c3fe13570ac7083b07` | 采集器**冻结副本**；文件名嵌入自身 sha 前缀 `8fab2e63`，与实测 sha 前缀一致 ⇒ 副本可自证身份 |

**`RUN.md` 的一句话结论（照录，不重判）**：行为策略换成「v4.5.0 `S` + R5 放雷规则」后，标签集第一次覆盖 `bombs_remaining ∈ {3,2,1}`（5,040 状态：3→624、2→3,378、1→1,038），**P0 以 16.9× / 5.2× 的余量通过**；同一 36 路标签、同一训练配方下 600 决策（20 s）活满率从 v8.3.0 的 **19.44 % 升到 29.58 %**（+10.14 pp，单侧 Fisher p = 4.98e-07），**P1 成立**；**P2（≥50 %）不成立**（29.58 %，距 R5 规则 77.05 % 差 47.47 pp），**P3 不成立**（10,800 episode 里 **0 个**只用 1–2 颗雷）⇒ **`S-partial`**。

**`runner_collect_8fab2e63.py` 的文件名自带自己的前缀**：该副本的实测 sha256 是 `8fab2e633cf2…`，与文件名里的 `8fab2e63` 一致 ⇒ 采集器副本可自证身份。`runner_collect_8fab2e63.py` 是**冻结副本**（123,026 B），与 `tools/rl/v860_bombdag.py` 的现行 sha（`ckpt_assertions.json` 记 `7b1f9119…`、`ckpt_assertions_final.json` 记 `3433ae74…`）**不是同一份**，两者不能互相替代（见 §22.4.5 第 2 条）。

#### 22.1.2 v8.6.0 标签集与统计（6 条 / 35,314,102 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 7 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/labels_bombdag.jsonl` | 35,189,092 | `e1f47805efdd37bb92d15c51898aacbef0ca13995a200d7d2e087cd919543c68` | **全轮唯一锚**：**5,040 行** × 36 路反事实标签 = 181,440 条；`e1f47805…` / 35,189,092 B，被 `labels_stats`、`train_seed{0,1,2}`、`train_summary` 三处引用为同一身份 |
| 8 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/labels_stats.json` | 21,123 | `79057ff783e104e5933b04be829071ea0b3823b703b7fa1fcfdeccae967413f1` | `p0.pass=true`（`bombs_remaining` 2→**3,378**、1→**1,038**，门槛 200 ⇒ 余量 16.9× / 5.2×）；`states=5,040`、`episodes=24`、`noop_state_frac=0.53631`、`anomaly_count=0`、`shape.complete_36_label_rows=5,040` |
| 9 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/labels_stats_smoke.json` | 8,119 | `b9a83aeb8d5599f849d53eed11195025e3dca024e3b771984b2d69b5afe92d38` | 冒烟统计；与未入册的 `smoke/labels_stats.json` **逐字节相同**（`b9a83aeb…`，8,119 B）⇒ 只登记这一份 |
| 10 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/diagnostics.json` | 51,263 | `8f90d47cc30d2005ea22d38e2a3f93e847affec72f704f1525fc571afca3935f` | 逐 cell 诊断 + `references`（v8.3.0 R5 参考 3,598 / 700 / 22）；含 `budget_comparison_warning`（三个 budget 不能当同一测量比均值）；`missing=[]`、`all_pass=true` |
| 11 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/bomb_usage.json` | 38,984 | `6acd3d2e2c1b59af2b69795b41e1cb78a4bf5c14a0c568dd1f98a9a2729b879a` | 三个 budget 的放雷事实（episodes 3,600/budget、`survived_fraction`、`alive_frames` 分布、`bomb_emits`）；`missing=[]` |
| 12 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/pilot.json` | 5,521 | `87a0d1d308b28028158100e8299f280eefd466535013fd051389ba7cbe56eb02` | 3,600 状态先导（quota 决策依据）：`bombs_remaining` 325 / 2,237 / 1,038；`states_for_200_each = {2: 322, 1: 694}` ⇒ 每格 420 的 quota 由此定出 |

`labels_bombdag.jsonl` 的 `e1f47805…` / 35,189,092 B / 5,040 行是**全轮共用的唯一锚**：`labels_stats.json` 的 `labels_file`、`train_seed{0,1,2}.json` 的 `labels_file`、`train_summary.json` 的 `labels`、以及 v8.12.0 `HYBRID_H2S/RUN.md` §2 的 `policy` 都指向同一 sha 与同一字节数 ⇒ 标签集**未在训练/评测/后续轮次中被重采或改写**。`labels_stats_smoke.json` 与未入册的 `smoke/labels_stats.json` **逐字节相同**（同为 `b9a83aeb…`，8,119 B），故只登记前者。

#### 22.1.3 v8.6.0 训练报告与 3 个 ckpt 副本（7 条 / 141,172 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 13 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/train_summary.json` | 2,746 | `93a9dda0b3b087493c0a460e8165d729536d54e711395084bb30a15292276834` | 3 seed × 200 epoch：loss `0.595201/0.574116/0.597460 → 0.039285/0.039133/0.039159`；`design="BCE, Adam 1e-3, batch 256"`、`missing=[]`、`all_pass=true` |
| 14 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/train_seed0.json` | 7,069 | `b306daeb82b177c49b09722eaaf709e9333241b4abcdfd3832c8295f06492399` | seed0 训练报告：`Policy(106,64,36)`、**9,188 参数**、`feature_dim=106`、`bombs_remaining_distribution` 与标签集一致 |
| 15 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/train_seed1.json` | 7,076 | `9723df5b34d4e9432bd890180569dbece91de644e8858dbe41c88dd626553462` | seed1 训练报告（同上结构） |
| 16 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/train_seed2.json` | 7,074 | `e029872ea71953aaf2274b56a7ccc8f581afa2135e2b9d26aa5880f629ab3069` | seed2 训练报告（同上结构） |
| 17 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/ckpt_seed0.pt` | 39,069 | `9c6d5d4ac7c9a6120ecd33aa9de7a0762c916413cc8f52e0c82f8ac41f8a483b` | seed0 训练产物**证据副本**（39,069 B）；与 §22.1.6 的 `.tmp` 原件**逐字节相同** |
| 18 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/ckpt_seed1.pt` | 39,069 | `36a51090d56149dc185d86e5b386f89304e5522c3612d9e919a4092e99f8e84c` | seed1 训练产物证据副本（39,069 B）；同上 |
| 19 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/ckpt_seed2.pt` | 39,069 | `885dde70c57f18bb7f1e3d0351737d239a52ce064aeb47fda025866bd6e45943` | seed2 训练产物证据副本（39,069 B）；同上 |

**训练实测读数**（`train_summary.json`）：`design="BCE, Adam 1e-3, batch 256, 200 epochs, 3 seeds"`、`missing=[]`、`all_pass=true`；三个 seed 的首/末 epoch loss = `0.595201→0.039285`、`0.574116→0.039133`、`0.597460→0.039159`，wall 1.934 / 2.164 / 1.858 s。`train_seed{k}.json` 记 `Policy` 形状 `Linear(106,64) → Tanh → Linear(64,36)`、**9,188 参数**、`feature_dim=106`、`actions=36`，且 `bombs_remaining_distribution` 与 `labels_stats.json` 逐项相同。

#### 22.1.4 v8.6.0 门、预检与审计（6 条 / 48,324 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 20 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/fidelity_d3.json` | 26,637 | `951fb9cea5383d9dbccf200eb64137b0eb6dcc9a8ccc51de4c5051409ee3d018` | D3 保真门 **5/5 PASS**（`verdict="PASS: §2.1-§2.5 hold on this binary and data dir"`）；2,165 前缀 checkpoint 0 mismatch、R24 常量 18/18、PARK frame 8,663、`deviations=[]`；引擎 `691bfd77…` match |
| 21 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/preflight.json` | 6,188 | `c7b15f178fb81a89ecf5b4085a06df7bedff5d601f9fdd35582ba04680c2ae70` | 开跑前自检：**8/8 pin match** + 引擎 match + `feature_dim=106` + `action_space={18 cand × 2 bomb = 36}` + 冻结协议 + `all_pass=true` |
| 22 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/ckpt_assertions.json` | 3,424 | `5c7ddb256b9434875057c21886eeb213b9b2567a4678187de877a9f05b29068e` | `phase=start`（`11:06:17+0800`）：v4.5.0 `round2.pt` **3/3 match**、引擎 match、`all_pass=true`；记预注册 `8df9bdc7…`（**与盘上现值不同**，见 §22.4.5 第 1 条） |
| 23 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/ckpt_assertions_final.json` | 3,424 | `73fef291f819d34d6cdae44243e3003da37a8c32b65109632f787d1acaa293b6` | `phase=final`（`12:12:09+0800`）：同上 `all_pass=true`；预注册改记 `2d10c507…`（= 盘上现值）、执行器改记 `3433ae74…`（见 §22.4.5 第 2 条） |
| 24 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/audio_audit.json` | 8,074 | `5c62975ccf222b5a47bfff80ea9b8c4fcda590d4a4a44bca3a1e5978e8e5270f` | AGENTS.md §9.3 审计：`cfg_files=62 / decoded=62 / muted=62 / audible=0 / errors=0`、`all_muted=true`；共享审计件 `v4.3.1/evidence/R14/audio_cfg_audit.json` 明确未触碰 |
| 25 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/process_residue.json` | 577 | `d9b25b80662d51d4b0a7b97f61ec48a1c586ae8fd2c431bb89d3a837628b3b60` | 本轮引擎进程残留 **`matching=0`**（`data_dir_filter=.tmp_v860_bombdag`）；宿主另有 48 个同名进程属**别轮**，未 `pkill`、未触碰 |

**四条门/审计读数**（全部 `all_pass=true`）：① `fidelity_d3.json` **5/5 PASS**（`verdict="PASS: §2.1-§2.5 hold on this binary and data dir"`、`2,165` 前缀 checkpoint 0 mismatch、R24 常量 18/18、PARK frame 8663、`deviations=[]`），引擎 `691bfd77…` 与 §2 记录的 `rl_v3` SHA **一致**；② `preflight.json` **8/8 pin match** + 引擎 match + `feature_dim=106` + `action_space=36`；③ 音频（AGENTS.md §9.3）**62/62 muted、`audible=0`、`errors=0`**、`all_muted=true`、未触碰共享审计件 `v4.3.1/evidence/R14/audio_cfg_audit.json`；④ 进程残留 **`matching=0`**（`data_dir_filter=.tmp_v860_bombdag`；宿主另有 48 个同名引擎进程属**别轮**，未触碰）。

#### 22.1.5 v8.6.0 九份评测：jsonl + summary 逐文件（18 条 / 98,076,113 B）

9 份 `eval_d{300,600,900}_seed{0,1,2}.jsonl` 各 **1,200 行**（300 episode × 4 个 dev window 11/12/17/18），每行一条 episode 记录；每份 jsonl 都有一份同名 `.summary.json` 记录逐窗口与合并存活率、`out_sha256` / `out_bytes`、`ckpt.sha256`、`sampler` 与 `episode_loop`。**9 份 summary 的 `out_sha256` / `out_bytes` 与现场实测 9/9 全等**（§22.4.2）。

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 26 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d300_seed0.jsonl` | 5,946,416 | `bb01986e69020a3f61705f021338cb70032f071555d35b2d1b6626c07340242b` | `d300` / seed0：**1,200 行**（300 episode × 4 dev window）；存活 **1,199/1,200（99.92 %）** |
| 27 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d300_seed0.summary.json` | 5,860 | `87d3d1e33fe2c2090fbe01b20007de63fa7155bfc37fe1586125bc74e0e8c656` | 其 summary：`out_sha256=bb01986e…` **match**、`out_bytes` match、`pooled_survived_fraction=0.999167`、`ckpt.sha256` 指回 §22.1.3/§22.1.6 |
| 28 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d300_seed1.jsonl` | 5,097,463 | `935fe7b2ee65a891323afd27f3eff2c123b59ea3cd6aae68e02e7c75f5e7d61b` | `d300` / seed1：**1,200 行**（300 episode × 4 dev window）；存活 **1,200/1,200（100 %）** |
| 29 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d300_seed1.summary.json` | 5,841 | `6623eb7aac8bbf36a03c90b2d010a377b74f4a853ad8df035a3b945b5da26551` | 其 summary：`out_sha256=935fe7b2…` **match**、`out_bytes` match、`pooled_survived_fraction=1.0`、`ckpt.sha256` 指回 §22.1.3/§22.1.6 |
| 30 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d300_seed2.jsonl` | 4,846,747 | `a6acbfc574063a1037f773d0ea0f8b2dea203d66052e53c1992983e1e8dafebc` | `d300` / seed2：**1,200 行**（300 episode × 4 dev window）；存活 **1,199/1,200（99.92 %）** |
| 31 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d300_seed2.summary.json` | 5,863 | `3e5b2f4011e92268ec434dc55495ab3c12fc6b59c4571245b9af1afd8a534edd` | 其 summary：`out_sha256=a6acbfc5…` **match**、`out_bytes` match、`pooled_survived_fraction=0.999167`、`ckpt.sha256` 指回 §22.1.3/§22.1.6 |
| 32 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d600_seed0.jsonl` | 13,993,113 | `1948690be207fa6338c3db8c208392c711a0ab909707f54f7c76cb7c042542cc` | `d600` / seed0：**1,200 行**（300 episode × 4 dev window）；存活 **343/1,200（28.58 %）** |
| 33 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d600_seed0.summary.json` | 5,945 | `74d88bfb38528727ed694eebbecc9f0922b8b0ff49d96cb08e9017754e5772bf` | 其 summary：`out_sha256=1948690b…` **match**、`out_bytes` match、`pooled_survived_fraction=0.285833`、`ckpt.sha256` 指回 §22.1.3/§22.1.6 |
| 34 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d600_seed1.jsonl` | 12,394,086 | `b285a81c5c6cf375e0f74e5d5e56fdff8a2ba89c6d96bf6b393a60925b4a8db8` | `d600` / seed1：**1,200 行**（300 episode × 4 dev window）；存活 **386/1,200（32.17 %）** |
| 35 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d600_seed1.summary.json` | 5,956 | `78fd51019cd1868cb8e6ff14aa8072c0d6ce226b237c988de354d94e40612bf8` | 其 summary：`out_sha256=b285a81c…` **match**、`out_bytes` match、`pooled_survived_fraction=0.321667`、`ckpt.sha256` 指回 §22.1.3/§22.1.6 |
| 36 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d600_seed2.jsonl` | 12,168,436 | `11298a93b28897652738c5d68224bedcc78d279deeac59a4556af2a38e897846` | `d600` / seed2：**1,200 行**（300 episode × 4 dev window）；存活 **336/1,200（28.00 %）** |
| 37 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d600_seed2.summary.json` | 5,934 | `32486d829eefc8126810107d7356ef0b971b7a3a0d1a89323fb5b3dc056cae2d` | 其 summary：`out_sha256=11298a93…` **match**、`out_bytes` match、`pooled_survived_fraction=0.28`、`ckpt.sha256` 指回 §22.1.3/§22.1.6 |
| 38 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d900_seed0.jsonl` | 15,712,124 | `e52f8b14249214fba979fd3b31e902726b2a4457ae082dba4d67bc1a1cec35e4` | `d900` / seed0：**1,200 行**（300 episode × 4 dev window）；存活 **49/1,200（4.08 %）** |
| 39 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d900_seed0.summary.json` | 5,955 | `2834adefc4389b6088b982edc38bf4e57834d917ca50d71224eef20a6c0392e6` | 其 summary：`out_sha256=e52f8b14…` **match**、`out_bytes` match、`pooled_survived_fraction=0.040833`、`ckpt.sha256` 指回 §22.1.3/§22.1.6 |
| 40 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d900_seed1.jsonl` | 14,551,116 | `e5dadc246cbe884e7153c878cf38275e46306b9c773e19459ecf48537851c626` | `d900` / seed1：**1,200 行**（300 episode × 4 dev window）；存活 **45/1,200（3.75 %）** |
| 41 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d900_seed1.summary.json` | 5,955 | `c69dd543ae4c109bfd0ef13d61b6d6eea611e400308bbba61d67aafad9e66783` | 其 summary：`out_sha256=e5dadc24…` **match**、`out_bytes` match、`pooled_survived_fraction=0.0375`、`ckpt.sha256` 指回 §22.1.3/§22.1.6 |
| 42 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d900_seed2.jsonl` | 13,313,364 | `8dc8afd09f1d91078e72ab2bb0df07d8c5038a980a5dff0b68e9918adf291f44` | `d900` / seed2：**1,200 行**（300 episode × 4 dev window）；存活 **12/1,200（1.00 %）** |
| 43 | `Document/Update/Alpha/v8.6.0/evidence/BOMBDAGGER/eval_d900_seed2.summary.json` | 5,939 | `2315581e66eb035a6d63634721d7e2f012aab05d641089c7d7ec2ec3c857c400` | 其 summary：`out_sha256=8dc8afd0…` **match**、`out_bytes` match、`pooled_survived_fraction=0.01`、`ckpt.sha256` 指回 §22.1.3/§22.1.6 |

**合并读数**：`d300` = 3,598/3,600 存活（**99.92 %**）、`d600` = **1,065/3,600（29.58 %）**、`d900` = 106/3,600（**2.94 %**）。`d600` 的 29.58 % 正是 `RUN.md` §0 与 v8.12.0 `HYBRID/verdict.json` 的 `H1_d600`（1,065/3,600 = 0.295833）**同一个数** ⇒ 两轮对同一批 episode 的口径闭合。

#### 22.1.6 v8.6.0 三个 ckpt 的运行目录原件（3 条 / 117,207 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 44 | `.tmp_v860_bombdag/ckpt/seed0/bombdag.pt` | 39,069 | `9c6d5d4ac7c9a6120ecd33aa9de7a0762c916413cc8f52e0c82f8ac41f8a483b` | **运行目录原件**（39,069 B）；与 `ckpt_seed0.pt` 逐字节相同（`9c6d5d4a…`） |
| 45 | `.tmp_v860_bombdag/ckpt/seed1/bombdag.pt` | 39,069 | `36a51090d56149dc185d86e5b386f89304e5522c3612d9e919a4092e99f8e84c` | 运行目录原件（39,069 B）；与 `ckpt_seed1.pt` 逐字节相同（`36a51090…`） |
| 46 | `.tmp_v860_bombdag/ckpt/seed2/bombdag.pt` | 39,069 | `885dde70c57f18bb7f1e3d0351737d239a52ce064aeb47fda025866bd6e45943` | 运行目录原件（39,069 B）；与 `ckpt_seed2.pt` 逐字节相同（`885dde70…`） |

**与 §22.1.3 的三份 `ckpt_seed{k}.pt` 逐字节相同**（`9c6d5d4a…` / `36a51090…` / `885dde70…`，各 39,069 B）⇒ 证据目录里的副本是 `.tmp_v860_bombdag/ckpt/` 原件的**精确副本**，不是重训产物。三处独立记录互相印证：`eval_*.summary.json` 的 `ckpt.sha256`、v8.12.0 `HYBRID_H2S/ckpt_assertions{,_final}.json` 的 `v860_bombdag`、v8.12.0 `HYBRID/RUN.md` §2.1，**三个前缀全部一致**。

#### 22.1.7 v8.12.0 冻结预注册（1 条 / 4,066 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 47 | `Document/Update/Alpha/v8.12.0/PREREGISTRATION_HYBRID_001.md` | 4,066 | `c937c1685af536b2d6aa9ac2cf378a9bfdcfb9603c5889516063be5d391636b4` | v8.12.0 冻结预注册（FROZEN）；`HYBRID/preflight.json`、`HYBRID/RUN.md`、`HYBRID_H2S/ckpt_assertions{,_final}.json` **三处独立记录同 sha** ⇒ 轮内一字未改 |

**三处独立记录都是同一个 sha**：`HYBRID/preflight.json` 的 `prereg.sha256`、`HYBRID/RUN.md` 头部、`HYBRID_H2S/ckpt_assertions{,_final}.json`（**start 与 final 两相相同**）⇒ 该预注册在整个 v8.12.0 轮（含补跑的 H2s）中**一字未改**。

#### 22.1.8 `HYBRID_H2S` 的文档、门与审计（13 条 / 77,203 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 48 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/RUN.md` | 20,068 | `429cb6666336b80fb17c03366b2efd37bd0f7f820a1a3aca945a10496f488e89` | 321 行运行记录：H2s 臂 240/240 episode 完成；四窗口 100.00 / 98.33 / 95.00 / 100.00 %、pooled **98.33 %**、CP95 **[95.79 %, 99.54 %]**；4 个 death；明写「本轮不训练、不采集、不换臂」 |
| 49 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/manifest.json` | 3,512 | `7f42798e61f9f060cb5ae77e1b392686fda364c330e7dfda5f412a3408a25e31` | **清单锚**：`files{}` **24** 条、`total_bytes=1,122,154`、`excludes="manifest.json"`；本节实测 **24/24 全等**；+ 自身 3,512 = 目录口径 1,125,666 |
| 50 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/summary.json` | 16,171 | `8dd6cf2aa1a9d2690cac958c213e3d49d6c56c206bd550a9798ceb3e82e527e6` | 12 个 cell 全量读数：`pooled={n:240, full:236, rate:0.983333, cp95:[0.957879, 0.995441]}`；`per_seed` 78/80、80/80、78/80；`bomb_usage.mean=2.9958`；`invalid_rows=0`；其 `files{}` 12/12 与盘上相符 |
| 51 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/publish.json` | 5,262 | `4f73df9bbee211a04476b5ad58ea4a5fe144239b114aaff1fee5d0adb03d6956` | 发布记录：**22 条**拷贝的 `{name, source, sha256, bytes}`（12 份 jsonl + 10 件文档） |
| 52 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/ckpt_assertions_start.json` | 4,492 | `942de94ee0c33b1875a284a1732725950e7c84de6a05543a6c10463a1b85d910` | `phase=start`（`12:35:40+0800`）：v8.6.0 `bombdag.pt` **3/3**（`9c6d5d4a/36a51090/885dde70`，各 39,069 B）+ v4.5.0 `round2.pt` **3/3**（`4abee069/a93746bb/c69960fc`，各 34,387 B）+ 引擎 + 预注册 `c937c168…`；`all_pass=true` |
| 53 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/ckpt_assertions_final.json` | 4,492 | `72bdabf5f4ad70c4174c918eda53d10326c81e120e29fd2f5539ef43405dc5ee` | `phase=final`（`13:17:07+0800`）：同上全部 match、`all_pass=true`；**预注册 sha 与 start 相同**，但执行器 sha 由 `8974d099…` 变 `1d25fe57…`（见 §22.4.5 第 3 条） |
| 54 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/d3_gate.json` | 5,033 | `b1c59559b8773d849ef9a7d1f9bcb421589165e0fa62f29ff46d31faeaec5bd2` | D3 门 **5/5 PASS**（`exit=0`、`passed=5/5`、2,165 前缀 checkpoint 0 mismatch、R24 18/18、PARK 8,663、`verdict="PASS: §2.1-§2.5 hold…"`）；引擎 `691bfd77…` match |
| 55 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/smoke.json` | 8,526 | `900a9a7b6dc2b0ce1133005c7d7bb1cf328aa27916b621ffc68e8a6e6aec7e5a` | 1 episode 冒烟（seed0 / w11，全量前的门）：`survived=true`、`bombs_used=3`、`alive_frames=1,200`、cfg 四零、`gate_prefix={hash_checks:2165, hash_mismatches:0}`；`all_pass=true` |
| 56 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/determinism_check.json` | 1,259 | `a4540e0545e19154c8c19fd4a53ee0ad9124bbdf64985f79cf3825b96238f77e` | w11 三格由两个进程各产一次：**逐字段相同**（`mismatches=[]`）；被中止尝试的死亡行（`first_death_frame=9,842`）在重跑中**逐字段复现**；`all_pass=true` |
| 57 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/audio_audit.json` | 3,635 | `db0aa62a5f54aeebf2f80cb0fd09c8c8f2147dba159a2fa89e9edf7954fc20a6` | §9.3 审计：`cfg_files=12 / decoded=12 / muted=12 / audible=0 / errors=0`；`all_pass=true` |
| 58 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/process_residue.json` | 508 | `c60206bc7d9630666c54af4a0b04183a162ce4b09b4024ee650492bb3bdbc01d` | 本轮残留 **`matching=0`**；宿主 17 个引擎进程属**别轮**（`other_round_processes=17`），未触碰；`all_pass=true` |
| 59 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/commands.tsv` | 4,156 | `a4d7891698cb2be39686ef0e0face290ebd592319f22ed768ef46c6dafb6458e` | 命令级记录（compile / assert-start / d3 / smoke / eval / publish 的 exit 与耗时） |
| 60 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/eval_exits.tsv` | 89 | `385304aa46c14c465dea5719a0192d56d15a5408f65ea3b3ba2eac2b849fd553` | 评测进程 exit 记录 **6 行**：首轮 `4 / 4 / 130` 三次失败后重试，重试后 s0/s1/s2 均 **exit 0**（660–1,389 s） |

#### 22.1.9 `HYBRID_H2S` 的 12 份逐窗口 jsonl（12 条 / 1,048,463 B）

12 份 = seed ∈ {0,1,2} × dev window ∈ {11,12,17,18}，每份 **20 行**（20 episode / window / seed），合计 **240 episode**、`budget_decisions=600`（= 1,200 帧），PARK frame 8663。

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 61 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/eval_H2s_w11_seed0.jsonl` | 87,143 | `6992f9f6ee2dcf74bd66e1f21f7b0b3a3505726103ee054176190fc6a6d1cbef` | **20 行**（20 episode / window / seed）；存活 **20/20**；`budget_decisions=600`（1,200 帧）、PARK 8,663 |
| 62 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/eval_H2s_w11_seed1.jsonl` | 87,978 | `f3b1efcaa9ad6ab38f65309d4da9e5b4b6bafdca08990f98ce682183837c9f3d` | **20 行**（20 episode / window / seed）；存活 **20/20**；`budget_decisions=600`（1,200 帧）、PARK 8,663 |
| 63 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/eval_H2s_w11_seed2.jsonl` | 87,798 | `2b8b57191642ee806c11192111831dfd5f1c1832e48193f10cca55cfd19fdd97` | **20 行**（20 episode / window / seed）；存活 **20/20**；`budget_decisions=600`（1,200 帧）、PARK 8,663 |
| 64 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/eval_H2s_w12_seed0.jsonl` | 86,743 | `8adf5c4bcf6292aa121a153768116a08b5d9018246fa7b1cdd8626d3a279bbd0` | **20 行**（20 episode / window / seed）；存活 **19/20**；`budget_decisions=600`（1,200 帧）、PARK 8,663 |
| 65 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/eval_H2s_w12_seed1.jsonl` | 87,839 | `88033e8fb0e449a741c9fe21ea528031709a1aa10911eab77e52aa8506b084ca` | **20 行**（20 episode / window / seed）；存活 **20/20**；`budget_decisions=600`（1,200 帧）、PARK 8,663 |
| 66 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/eval_H2s_w12_seed2.jsonl` | 87,924 | `cc82532eb5ae30888f06baee9e4e7c65ef670aa2636af774125ce53485950f52` | **20 行**（20 episode / window / seed）；存活 **20/20**；`budget_decisions=600`（1,200 帧）、PARK 8,663 |
| 67 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/eval_H2s_w17_seed0.jsonl` | 87,080 | `89b6a7e947200571aeef2f86f80fae3141a93dd485987d430a7de582fac16d31` | **20 行**（20 episode / window / seed）；存活 **19/20**；`budget_decisions=600`（1,200 帧）、PARK 8,663 |
| 68 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/eval_H2s_w17_seed1.jsonl` | 88,227 | `f0e7a9bc0c36c1ecd5c0032a15978f9e22e5741addead20a914d2c5744eb239e` | **20 行**（20 episode / window / seed）；存活 **20/20**；`budget_decisions=600`（1,200 帧）、PARK 8,663 |
| 69 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/eval_H2s_w17_seed2.jsonl` | 86,809 | `b52187a01be69c92b3d0fb7cfb8a0a639191744549342fb2f03fa2372c8dbbfa` | **20 行**（20 episode / window / seed）；存活 **18/20**；`budget_decisions=600`（1,200 帧）、PARK 8,663 |
| 70 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/eval_H2s_w18_seed0.jsonl` | 86,229 | `04a10692e52ac984896ac5ccb5409f7fbc2c47a0a653782e2cfa4efb99a23de4` | **20 行**（20 episode / window / seed）；存活 **20/20**；`budget_decisions=600`（1,200 帧）、PARK 8,663 |
| 71 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/eval_H2s_w18_seed1.jsonl` | 87,438 | `7cc8e76f8cb01dcd4f595606ccde3ae6b49a45b1edc127152701845f5269b6b9` | **20 行**（20 episode / window / seed）；存活 **20/20**；`budget_decisions=600`（1,200 帧）、PARK 8,663 |
| 72 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID_H2S/eval_H2s_w18_seed2.jsonl` | 87,255 | `20291b75033ba28b7f6082cad57ea2bd474aeffb4055c3cab3fd5b8719d74cfe` | **20 行**（20 episode / window / seed）；存活 **20/20**；`budget_decisions=600`（1,200 帧）、PARK 8,663 |

**合并读数**（`summary.json`）：`pooled = 236/240 = 98.333 %`、CP95 **[95.79 %, 99.54 %]**；四窗口 100.00 % / 98.33 % / 95.00 % / 100.00 %；`per_seed` = 78/80、80/80、78/80；4 个 death；`bomb_usage.mean=2.9958`（239/240 用满 3 颗）；`invalid_rows=0`。参考系 `r5_same_window_rate = {11: 0.811, 12: 0.792, 17: 0.704, 18: 0.777}` ⇒ 每个窗口均高于 R5。

#### 22.1.10 `HYBRID` 的文档、判决与门（18 条 / 239,526 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 73 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/RUN.md` | 17,605 | `bf92f27c6e194e147d1ed51ae45cbc65118974f55bd06e26cf61e89025c18fa4` | 347 行：四臂定义表、门表（8/8 pin、D3 5/5、音频 32/32）、§2.1 被评测 ckpt、结论；**mtime 晚于 `manifest.json` 29 秒**（§22.4.2） |
| 74 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/manifest.json` | 8,375 | `0c1e0e818fd59b017b6bb6fdb516c895b859f51077c7269c0207db0ab9910d2b` | **清单锚（有一处陈旧）**：`files{}` **59** 条、`total_bytes=131,368,485`、`excludes="manifest.json"`；与盘上 **55/59 相符**，4 处偏差见 §22.4.2 |
| 75 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/verdict.json` | 12,781 | `d6e90e2b8f3a29eae96a82ca97e72be0c7ef413187e87851dfb0d7a9037e3d85` | 判决：`P1` H2s d600 **0.979167**（705/720，CP95 [96.59 %, 98.83 %]）**pass**；`P2` H2s>H1（单侧 Fisher p=0.0）**pass**；`P3` `value=null / pass=false`；`P4` 放雷均值 **2.9889**；`stops={stop:"S-deploy"}` |
| 76 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/crosscheck.json` | 5,041 | `0e52ed7efed97d3dfde629087306a8bc656c0c66dbf3f4c80f3af204fe32d2a4` | H1 代码路径 **20/20 逐字段复现** v8.6.0 `eval_d600_seed0.jsonl`（参照 sha `1948690b…`）；`mismatches=[]`、`all_pass=true` |
| 77 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/h1_adoption.json` | 4,282 | `583e8b7aaffed2c4749f74560c065b8210980897121224ea3ff14f90f3a4dde2` | 「H1 = v8.6.0 评测网格」的**采纳决定**（前置条件 `crosscheck_all_pass=true`）；逐 cell 记 `source_sha256` ⇒ §22.1.11 的 9 份 jsonl 与 v8.6.0 逐字节相同是**设计使然** |
| 78 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/preflight.json` | 8,572 | `e460065994c937b613a765013956aa724389dc236e24077922a5a2ff95bf2db3` | **8/8 pin match** + 预注册 `c937c168…` + 引擎 match + `feature_dim=106` + 四臂定义 + `frozen` 协议 + `missing_checkpoints=[]`；`all_pass=true` |
| 79 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/fidelity_d3.json` | 26,640 | `407bf02ae355726b908d1e2dbac0ea1e1ef0fcb0bcbc3d61606de00ccdeae2e2` | D3 保真门 **5/5 PASS**（`all_pass=true`、`verdict="PASS: §2.1-§2.5 hold…"`）；引擎 `691bfd77…` match |
| 80 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/ckpt_assertions.json` | 5,626 | `e1c78a97a7dc96a6ec8aa97f79a27b3da9da3605d45a3ded87a3fdd5bdfc4780` | `phase=start`：`all_pass=true`（引擎与预注册的 start 相记录；与 `_final` 互为一对） |
| 81 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/ckpt_assertions_final.json` | 6,295 | `bff46619dd7913cd2a88517182b5d07637d32be8b11098a97116a82a836191a4` | `phase=final`：`all_pass=true`（两相共同证明整轮被评测 ckpt 未漂移） |
| 82 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/audio_audit.json` | 4,915 | `93756e7cae5044fe7811956c339f76eb86c9625cea684204ccc2e740cb279be4` | §9.3 审计：`cfg_files=32 / decoded=32 / muted=32 / audible=0 / errors=0`、`audible=[]` |
| 83 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/process_residue.json` | 533 | `285a2b192681e21e198596b079948a2fd11e30e2ccd1dbcad66df5c98fb4c1a2` | 本轮残留 **`matching=0`**；`all_pass=true` |
| 84 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/diagnostics.json` | 32,916 | `6b5289711e684ca7c52d2f666d333785c2eb102ee715c2987585a7c3b244cd93` | 四臂逐 cell 诊断：同时报 `episodes` 与 `distinct_trajectories` 两套口径（H2/H3 确定性臂的 `distinct=3` 由此可查） |
| 85 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/bomb_usage.json` | 5,797 | `27aa500509acd5f0df683909e394b466d6a36bc01f94170cd7e9f6757779d56f` | 四臂放雷分布（每 episode 的 `bombs_used` / `bomb_emits` / `bomb_decisions`） |
| 86 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/determinism_probe.json` | 711 | `0cf206778ae54bab1425e7e4ad5ae6263f1c1aacf64d5ddeccf4e092ac132089` | H2/H3 六格均 `distinct=1`（每格只有一条轨迹）⇒ 那 6 份 jsonl 只有 **2–6 行**不是中断，是确定性重放的必然结果 |
| 87 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/exec.sh` | 7,617 | `3e32d4ccf49eeea7d6e08afd5218402df48c1477d815d5bfbd0f4fc0df3919f9` | 一键执行脚本（preflight / d3 / crosscheck / adopt / smoke / eval / publish 各段） |
| 88 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/smoke.json` | 83,060 | `034ca8c318a6b66024abbdd29c745ba67c5a3b371e494ef15ae25598d20bb893` | 冒烟记录（83,060 B，含逐决策采样审计） |
| 89 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/commands.tsv` | 6,763 | `80113644e64fba14133defeca94a9e198cc9dabcf9406c4e0b95902050e843a3` | 命令级记录；**mtime 晚于 `manifest.json`**，比 manifest 多 167 B（§22.4.2） |
| 90 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/contention.tsv` | 1,997 | `a1f9ab1b98cccf1cf45cd3876fc5614ea3bfeabeb2aed4dc8b7cffcac9847b01` | 并发/抢占记录；**mtime 晚于 `manifest.json`**，比 manifest 多 88 B（§22.4.2） |

**四臂定义**（`RUN.md` §1 表）：`H1` = v8.6.0 36 路策略**自带的 bomb 头**；`H2` = 同策略 `argmax_c max(logit[2c],logit[2c+1])` + **外部 R5 规则**；`H3` = v8.3.0 策略 argmax + R5；`H2s` = 同 H2 但候选**采样**（`softmax/1.0`，每决策一次 `torch.multinomial`，generator `seed*1000+7`）。四臂共用**同一个** episode loop（`tools/rl/v830_bombpol.py::run_episode36`，import 而非重写），唯一差别是 `decide` 回调。

#### 22.1.11 `HYBRID` 的 H1 评测（18 条 / 98,081,678 B）

H1 的 9 份 jsonl **不是重跑**：`h1_adoption.json` 记录「H1 = v8.6.0 自有评测网格」的采纳决定（前置条件是 `crosscheck.json` 的 20 episode 逐字段复现），因此这 9 份 jsonl 与 §22.1.5 的 9 份**逐字节相同**；`.summary.json` 则是本轮**重新生成**的（带 `arm` 元信息，故 sha 与 v8.6.0 的 summary 不同）。9 份 jsonl 各 **1,200 行**。

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 91 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d300_seed0.jsonl` | 5,946,416 | `bb01986e69020a3f61705f021338cb70032f071555d35b2d1b6626c07340242b` | **1,200 行**；与 v8.6.0 `eval_d300_seed0.jsonl` **逐字节相同**（`bb01986e…`）⇒ 采纳而非重跑 |
| 92 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d300_seed0.summary.json` | 6,479 | `431dd1f6ee83255cfbd94b595a8b1efa27b5a466e0b29919453c4dc8c0e00b24` | H1 summary（本轮**重新生成**，带 `arm=H1` 元信息，故 sha 与 v8.6.0 的同名 summary 不同） |
| 93 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d300_seed1.jsonl` | 5,097,463 | `935fe7b2ee65a891323afd27f3eff2c123b59ea3cd6aae68e02e7c75f5e7d61b` | **1,200 行**；与 v8.6.0 `eval_d300_seed1.jsonl` **逐字节相同**（`935fe7b2…`）⇒ 采纳而非重跑 |
| 94 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d300_seed1.summary.json` | 6,460 | `50d6ed65d585ab3d4b2fea9a382fd73b019bbe3dec4c3dddfe1b869dbb42e35b` | H1 summary（本轮**重新生成**，带 `arm=H1` 元信息，故 sha 与 v8.6.0 的同名 summary 不同） |
| 95 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d300_seed2.jsonl` | 4,846,747 | `a6acbfc574063a1037f773d0ea0f8b2dea203d66052e53c1992983e1e8dafebc` | **1,200 行**；与 v8.6.0 `eval_d300_seed2.jsonl` **逐字节相同**（`a6acbfc5…`）⇒ 采纳而非重跑 |
| 96 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d300_seed2.summary.json` | 6,482 | `eb6776480e13d562f7defc9ba992f3e7c0349ec5cb7d8e35dc8da3400d266237` | H1 summary（本轮**重新生成**，带 `arm=H1` 元信息，故 sha 与 v8.6.0 的同名 summary 不同） |
| 97 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d600_seed0.jsonl` | 13,993,113 | `1948690be207fa6338c3db8c208392c711a0ab909707f54f7c76cb7c042542cc` | **1,200 行**；与 v8.6.0 `eval_d600_seed0.jsonl` **逐字节相同**（`1948690b…`）⇒ 采纳而非重跑 |
| 98 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d600_seed0.summary.json` | 6,563 | `3863137fbb91a577d97ba4d5bc799799c161d792ad6028f9e3f4d6cd16d8a471` | H1 summary（本轮**重新生成**，带 `arm=H1` 元信息，故 sha 与 v8.6.0 的同名 summary 不同） |
| 99 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d600_seed1.jsonl` | 12,394,086 | `b285a81c5c6cf375e0f74e5d5e56fdff8a2ba89c6d96bf6b393a60925b4a8db8` | **1,200 行**；与 v8.6.0 `eval_d600_seed1.jsonl` **逐字节相同**（`b285a81c…`）⇒ 采纳而非重跑 |
| 100 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d600_seed1.summary.json` | 6,574 | `3cbd2c984690c14afdeaa9d9ac3f3d2f8a3a05d9285ea1866acd4f28936c3f67` | H1 summary（本轮**重新生成**，带 `arm=H1` 元信息，故 sha 与 v8.6.0 的同名 summary 不同） |
| 101 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d600_seed2.jsonl` | 12,168,436 | `11298a93b28897652738c5d68224bedcc78d279deeac59a4556af2a38e897846` | **1,200 行**；与 v8.6.0 `eval_d600_seed2.jsonl` **逐字节相同**（`11298a93…`）⇒ 采纳而非重跑 |
| 102 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d600_seed2.summary.json` | 6,552 | `ef069d95c65fd2b843d99d47615e1a64ba2f45e7772ad6447db444cd09590be3` | H1 summary（本轮**重新生成**，带 `arm=H1` 元信息，故 sha 与 v8.6.0 的同名 summary 不同） |
| 103 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d900_seed0.jsonl` | 15,712,124 | `e52f8b14249214fba979fd3b31e902726b2a4457ae082dba4d67bc1a1cec35e4` | **1,200 行**；与 v8.6.0 `eval_d900_seed0.jsonl` **逐字节相同**（`e52f8b14…`）⇒ 采纳而非重跑 |
| 104 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d900_seed0.summary.json` | 6,573 | `687b4532a0342356549c1767e6eb3e7bb2a30c00c20f891642c891ac2c8c1959` | H1 summary（本轮**重新生成**，带 `arm=H1` 元信息，故 sha 与 v8.6.0 的同名 summary 不同） |
| 105 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d900_seed1.jsonl` | 14,551,116 | `e5dadc246cbe884e7153c878cf38275e46306b9c773e19459ecf48537851c626` | **1,200 行**；与 v8.6.0 `eval_d900_seed1.jsonl` **逐字节相同**（`e5dadc24…`）⇒ 采纳而非重跑 |
| 106 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d900_seed1.summary.json` | 6,573 | `fea9d93a3df8543111104efc946202a192d2169ed4f19b06487d78c73262186e` | H1 summary（本轮**重新生成**，带 `arm=H1` 元信息，故 sha 与 v8.6.0 的同名 summary 不同） |
| 107 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d900_seed2.jsonl` | 13,313,364 | `8dc8afd09f1d91078e72ab2bb0df07d8c5038a980a5dff0b68e9918adf291f44` | **1,200 行**；与 v8.6.0 `eval_d900_seed2.jsonl` **逐字节相同**（`8dc8afd0…`）⇒ 采纳而非重跑 |
| 108 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H1_d900_seed2.summary.json` | 6,557 | `ec079baed48ca3eec74d6836176bfdba4f65d2c22bbd241d3fcf0e98b2e0f607` | H1 summary（本轮**重新生成**，带 `arm=H1` 元信息，故 sha 与 v8.6.0 的同名 summary 不同） |

#### 22.1.12 `HYBRID` 的 H2 评测（8 条 / 827,988 B）

H2 是**确定性**臂（候选 argmax + 固定距离规则），`determinism_probe.json` 实测每个 cell 的 `distinct=1` ⇒ 同一格里的 300 个 episode 是**同一轨迹的重放**，故每份 jsonl 只有 **2–6 行**（跑到第一条重复轨迹即止），这不是缺件：`verdict.json` 的 `H2_d600` 因此只报 `episodes=9 / distinct=3`。**`eval_H2_d600_seed2.jsonl` 本轮不入册**（0 字节，见 §22.2）。

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 109 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H2_d300_seed0.jsonl` | 117,694 | `8589266b64622a780fce424afdd4b4eac408fc5494d9be258da68701cf1e015c` | **6 行**（确定性臂，`distinct=1`）；存活 6 |
| 110 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H2_d300_seed1.jsonl` | 98,334 | `baaf5b2bbc0a2972f1013deb6ab69b665d0722106081737c306098326b9fd0bd` | **5 行**（确定性臂，`distinct=1`）；存活 5 |
| 111 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H2_d300_seed2.jsonl` | 98,255 | `5ffe644cc7759573b8924de4f673d3dc4fba3fc7f92e995837de87c5318eaff8` | **5 行**（确定性臂，`distinct=1`）；存活 5 |
| 112 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H2_d600_seed0.jsonl` | 112,149 | `274ce2eafdd81db1269619565ce06ab79828262a431f4cfa7b683d3e5ebbeada` | **3 行**（确定性臂，`distinct=1`）；存活 3 |
| 113 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H2_d600_seed1.jsonl` | 112,428 | `226f13fe3225450824e1cd32f3d77ec10eaec97f18d93f75ca43537c562a2250` | **3 行**（确定性臂，`distinct=1`）；存活 3 |
| 114 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H2_d900_seed0.jsonl` | 89,224 | `fc9f703464d7349e04e1e2748af2173046f0a8fef6a212c06687a7f8dbe778e7` | **2 行**（确定性臂，`distinct=1`）；存活 0 |
| 115 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H2_d900_seed1.jsonl` | 90,282 | `c0e51061b1722eebc2371af75f9f5597fb41f5bda770e38cd9d9b62f548dde22` | **2 行**（确定性臂，`distinct=1`）；存活 0 |
| 116 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H2_d900_seed2.jsonl` | 109,622 | `36674e47e8c20638a479ecf3d645bd4079674a66e2388a3dc617ecfa8d48ab23` | **2 行**（确定性臂，`distinct=1`）；存活 2 |

#### 22.1.13 `HYBRID` 的 H3 评测（9 条 / 908,136 B）

H3 同 H2 是确定性臂（`v8.3.0` 策略 argmax + R5），同样每份 **2–6 行**；`verdict.json` 的 `H3_d600 = 9 episode / 3 distinct`。

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 117 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H3_d300_seed0.jsonl` | 116,844 | `51029b540207a1de12c2489591f01a09a310ebfabd411ad8dbd11de36c934dfa` | **6 行**（确定性臂，`distinct=1`）；存活 6 |
| 118 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H3_d300_seed1.jsonl` | 117,341 | `033d470123beb7656f1b309e53d9c038e5b6424765e50441745bb01452e4836b` | **6 行**（确定性臂，`distinct=1`）；存活 6 |
| 119 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H3_d300_seed2.jsonl` | 96,656 | `21abf7726d525805205a45e37dae79555d9881755584f3112b053ba0069fb94c` | **5 行**（确定性臂，`distinct=1`）；存活 5 |
| 120 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H3_d600_seed0.jsonl` | 110,765 | `8c5ecc49d8bcc4f45c1295ed13a23a37364ae90e70c3229887d8888622ae15e9` | **3 行**（确定性臂，`distinct=1`）；存活 3 |
| 121 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H3_d600_seed1.jsonl` | 111,873 | `8c9f9dc19816ecf2588d522f2cf5d6ff2d77541c0b080e9fb3b4904bc23d2d36` | **3 行**（确定性臂，`distinct=1`）；存活 3 |
| 122 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H3_d600_seed2.jsonl` | 110,751 | `30db902106f9621b34a9844dae34e982386b58a6b0e4f319de891d5256501851` | **3 行**（确定性臂，`distinct=1`）；存活 3 |
| 123 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H3_d900_seed0.jsonl` | 79,968 | `476034e36a277328e4f6d1d9b51d22779892181622b67bccc63e12c7d11acfa6` | **2 行**（确定性臂，`distinct=1`）；存活 0 |
| 124 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H3_d900_seed1.jsonl` | 82,040 | `632cbdffadb88589d6ac80892679903307fd2823226b893793c80cd8ef5dee1a` | **2 行**（确定性臂，`distinct=1`）；存活 0 |
| 125 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H3_d900_seed2.jsonl` | 81,898 | `cb782cd811dbaa76da7614bc3d1eac1d47216f2c5ab8ea052cb119e349ff94df` | **2 行**（确定性臂，`distinct=1`）；存活 0 |

#### 22.1.14 `HYBRID` 的 H2s 评测（6 条 / 31,208,429 B）

H2s 三份 jsonl 各 **240 行**（60 episode/window × 4 window），`budget_decisions=600`；存活 235 / 237 / 233（合计 **705/720**）。这三份是 **P1 的判据来源**（`verdict.json` 的 `P1.value = 0.979167`）。

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 126 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H2s_d600_seed0.jsonl` | 10,390,315 | `31749600d8ae2882bcc3e80bb47217720dbd7f21e17a94eb6c2e91fefcc9df2a` | **240 行**（60 episode/window × 4 window）；存活 **235/240（97.92 %）** |
| 127 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H2s_d600_seed0.summary.json` | 8,306 | `5109ea5bca7a47cc4e8db62155ad488081b1b84750c3404048868cf0e3c80731` | 其 summary：pooled 235/240、CP95、逐窗口与 `bomb_usage`；三份 seed 合计 705/720 ⇒ **P1 的判据** |
| 128 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H2s_d600_seed1.jsonl` | 10,399,132 | `afe2ae71b0fe4a102746e62789722238583ced08d581b2ea3f8cf6c75478c014` | **240 行**（60 episode/window × 4 window）；存活 **237/240（98.75 %）** |
| 129 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H2s_d600_seed1.summary.json` | 8,291 | `ca80d4cfd0201d17d36591859d071b2bca222ad562a33d3caac4228b245d9a4d` | 其 summary：pooled 237/240、CP95、逐窗口与 `bomb_usage`；三份 seed 合计 705/720 ⇒ **P1 的判据** |
| 130 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H2s_d600_seed2.jsonl` | 10,394,089 | `7a4b1c8d2bcd6ce68469515277fb9aa4d986f8c438316e8d40eac3355b41fc75` | **240 行**（60 episode/window × 4 window）；存活 **233/240（97.08 %）** |
| 131 | `Document/Update/Alpha/v8.12.0/evidence/HYBRID/eval_H2s_d600_seed2.summary.json` | 8,296 | `d3925788f203df176d047cc9017bfb5d7618b67e67da7e4ef129701d5179eadb` | 其 summary：pooled 233/240、CP95、逐窗口与 `bomb_usage`；三份 seed 合计 705/720 ⇒ **P1 的判据** |

### 22.2 跳过说明（未入册 **15** 个文件 / **165,935** B，逐组给理由）

三处实测：`v8.6.0/` **57** 个文件 / 133,935,659 B，其中**目录内**入册 **43** 条 / 133,769,724 B（42 件证据 + 1 份预注册），另有 **3 个 ckpt 在目录外**的 `.tmp_v860_bombdag/ckpt/` 也已入册（§22.1.6）；`v8.12.0/` **86** 个文件 / 132,395,489 B（入册 26 + 59 = 85 条 / 132,395,489 B）。两目录合计未入册 **15 个文件 / 165,935 B**（14 在 v8.6.0、1 在 v8.12.0）即下表。

| 组 | 文件数 | 字节 | 不入册理由 |
| --- | ---: | ---: | --- |
| v8.6.0 `evidence/BOMBDAGGER/collect/` 的 **12** 个 cell summary（`seed{0,1,2}_w{13,14,15,16}.json`，各 6,918–6,962 B） | 12 | 83,355 | 每格一份的**采集摘要**（`decisions` / `bomb_emits` / `deaths` / `bombs_remaining_counts` / `wall_s` / `status`），其聚合值已由入册的 `labels_stats.json`（`per_cell` 字段逐个 cell 收录）与 `pilot.json` 承载；**逐文件 `{bytes, sha256}` 已在入册 `manifest.json` 的 `files{}` 内**。 |
| v8.6.0 `evidence/BOMBDAGGER/smoke/` 的 **2** 件：`labels_bombdag.jsonl`（74,461 B，冒烟子集）+ `labels_stats.json`（8,119 B） | 2 | 82,580 | 冒烟子集与其统计；其统计与入册的 `labels_stats_smoke.json` **逐字节相同**（`b9a83aeb…`），故只登记后者；冒烟 jsonl 的 `{bytes, sha256}` 同样在入册 `manifest.json` 内。 |
| v8.12.0 `evidence/HYBRID/eval_H2_d600_seed2.jsonl` | 1 | 0 | **登记时是 0 字节的截断件**（`mtime 05:22:02Z`，199 秒后该目录停止变动）。产出方 `HYBRID/manifest.json` 自述它应为 **112,712 B / `d3dd8bd5…`**。给一个被截断的文件钉 sha256 等于把损坏态写成权威值，**故不登记**，只在此记录事实；修复/重跑后应在 §22.5 补登。 |
| **小计** | **15** | **165,935** | — |

**为什么「不入册」不等于「没有字节身份」**：v8.6.0 的两组 **14 件**全部落在入册 `manifest.json` 的 `files{}` 里 —— 该 manifest 覆盖 `evidence/BOMBDAGGER/**` 除自身以外的 **55** 个文件，本节已实测复核 **55/55 全等、0 处不符**（§22.4.2）。盘上 `BOMBDAGGER/**` 共 **56** 个文件 = 55（manifest 覆盖）+ `manifest.json` 自身；其中入册 **42** 件、未入册 **14** 件 ⇒ **56 件 100 % 有字节身份记录**（42 条逐条 + 14 条由入册 manifest 覆盖）。

**`__pycache__` / `*.pyc` 实况（按要求只报告、不删）**：`v8.6.0/`（57 文件）与 `v8.12.0/`（86 文件）全树 **没有** `__pycache__/`、**没有** `*.pyc`；唯一的 `.py` 是 v8.6.0 的 `runner_collect_8fab2e63.py`（冻结副本，无字节码缓存）。若日后出现，按 §21.2 口径**一律不入册**。

**`.tmp_v860_bombdag/**` 运行目录**：`logs/`、`collect/`、`runs/`、`work/`、`data/`、`labels/`、`train/`、`eval/`、`pilot/` 与顶层运行时脚本**不在 `Document/` 下**、不是发布件，故只登记其中 **3 个 ckpt**（§22.1.6）；其余过程件不入册，也不在本节范围内。

### 22.3 与 §1–§21 不重复的说明（实测已核）

- **既有 300 行 0 命中**：追加前对全部 300 行做 `v8\.6\.0|v8\.12\.0|BOMBDAGGER|HYBRID|v860|v8d0` 匹配 → **0 命中**；`§21` 登记的是 v8.9.0（RULEDIST2）/ v8.10.0（HAZFIX），与本轮**不同版本、不同目录**。
- **路径 131 唯一**：131 条路径逐条比对 **0 组重复**。
- **哈希 119 唯一 / 12 组有意重复**：131 条里只有 **12 组**同哈希对 —— ① `ckpt_seed{k}.pt`（证据副本）与 `.tmp_v860_bombdag/ckpt/seed{k}/bombdag.pt`（原件）**3 组**逐字节相同；② `HYBRID` 的 9 份 `eval_H1_*.jsonl` 与 v8.6.0 的 9 份 `eval_d*.jsonl` **9 组**逐字节相同（`h1_adoption.json` 的**采纳**决定，不是重跑）。这 12 组是**设计使然**，已在 §22.1.6 / §22.1.11 写明来源。
- **与既有 300 行 0 交集**：131 个路径、119 个唯一哈希与既有 **300 个路径 / 297 个唯一哈希**逐一比对，**路径交集 0、哈希交集 0** ⇒ 没有把任何既有行「换个路径再钉一遍」。
- **行数与格式**：300 + 131 = **431 行**；仍是 **LF / 无 BOM**，末行以 `\n` 结束，每行 `64 hex + 两个空格 + 相对路径`，**空行 0 条、格式异常行 0 条**。
- **`ASSET_SYNC.md` 本身不在校验表内**（`Select-String 'ASSET_SYNC' ASSET_SYNC.sha256` = **0** 行）⇒ 追加 §22 **不会**让任何既有行转 FAILED（与 §21.3 / §21.5.7 同一口径）。
- **Git 跟踪状态（实测）**：`git status --porcelain -- Document/Update/Alpha/v8.6.0 Document/Update/Alpha/v8.12.0` → 两目录均为 `??`（整目录 untracked）；`git ls-files -- <两目录>` = **0** ⇒ 这些证据**无法由 Git 重建**，本节 + `ASSET_SYNC.sha256` 是它们目前的字节身份记录。本轮按约束**未** `git add` / `git commit`，分支仍为 `v400`。

### 22.4 校验结论

#### 22.4.1 `sha256sum -c`（WSL，追加前后各一次）

命令（与 §21.5.7 同一口径）：

```bash
cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -c ': OK'; sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -v ': OK'
```

* **基线（追加前 300 行）**：**299 OK / 1 FAILED**，`sha256sum` 退出码 **1**（仅因那 1 条）。
* **追加后（431 行）**：**430 OK / 1 FAILED**，退出码 **1**（仅因那 1 条）。`430 + 1 = 431` 校验行全部参与。
* **新增 131 条 131/131 OK** ⇒ 整体 FAILED 数**没有新增（仍为 1）**；单独对本节 131 行做 `sha256sum -c`，**131/131 OK，退出码 0**。
* **唯一 FAILED 仍是已知并允许的那一条**：`Document/Update/Alpha/v4.2.0/journal.jsonl`（§9.8 记录的活动文件漂移）。本轮**未改该行、未重钉、未触碰该文件**。
* **格式**：431 行仍为 **LF / 无 BOM**、末行 `\n` 结束、每行 `64 hex + 两个空格 + 相对路径`；空行 **0**、格式异常 **0**；新增 131 条的路径在表内**各自唯一**。

#### 22.4.2 与三份产出方 `manifest.json` 的交叉核对（登记时实测）

| 清单 | 自述 | 实测比对 |
| --- | --- | --- |
| v8.6.0 `evidence/BOMBDAGGER/manifest.json` | `file_count=55`、`total_bytes=133,922,602`、`excludes="manifest.json"` | `files{}` **55/55** 的 `{bytes, sha256}` 全等、**0 处不符**；`bytes` 求和 = `total_bytes` = 133,922,602；盘上 `BOMBDAGGER/**` 的另一件只有 `manifest.json` 自身（56 − 55 = 1） |
| v8.12.0 `evidence/HYBRID_H2S/manifest.json` | `file_count=24`、`total_bytes=1,122,154`、`excludes="manifest.json"` | `files{}` **24/24** 全等、0 处不符；求和 = 1,122,154；盘上多出的只有 `manifest.json` 自身（25 − 24 = 1）；另：`summary.json` 的 `files{}` **12/12** 全等（与 manifest 互为独立锚） |
| v8.12.0 `evidence/HYBRID/manifest.json` | `file_count=59`、`total_bytes=131,368,485`、`excludes="manifest.json"` | **55/59 全等、4 处不符**（详见下） |
| v8.6.0 九份 `eval_*.summary.json` | 各自 `out_sha256` / `out_bytes` | **9/9** 与现场实测全等（§22.1.5 表内已逐条列出） |

`HYBRID` 的 4 处不符（**只报告，未改该目录任何一个字节**）：

| 文件 | 盘上 | manifest 自述 | 说明 |
| --- | ---: | ---: | --- |
| `RUN.md` | 17,605 | 16,251 | manifest 由 `v8c0_hybrid.py` 在 `13:21:28+0800` 生成；`RUN.md` 的 mtime 是 `05:21:57Z`（= `13:21:57`），**比 manifest 晚 29 秒** ⇒ manifest 记的是旧版 |
| `commands.tsv` | 6,763 | 6,596 | 同上，mtime `05:21:27Z`（与 manifest 同一秒内、之后） |
| `contention.tsv` | 1,997 | 1,909 | 同上，mtime `05:21:59Z` |
| `eval_H2_d600_seed2.jsonl` | **0** | 112,712 | 截断件，见 §22.2 |

⇒ `HYBRID` 的 manifest 是**发布时刻的快照**，其后的 3 处追写 + 1 处截断发生在同一分钟内（`05:21:27Z–05:22:02Z`，即本地 `13:21:27–13:22:02`），**随后该目录停止变动**（`13:22:11Z`→`13:26:12Z` 连续 6 次 15 s 轮询签名完全相同）。本节按「以盘上为准」登记**盘上值**，并把「manifest 与盘上不同」这件事实写在这里。

#### 22.4.3 身份链、门与读数（只登记，不重判）

| 核对项 | 实测 |
| --- | --- |
| 引擎身份 | 三处独立记录均 `691bfd77b7eca336…` 且 `match=true`：v8.6.0 `preflight.json` / `fidelity_d3.json` / `ckpt_assertions{,_final}.json`；v8.12.0 `HYBRID/preflight.json` / `fidelity_d3.json` / `HYBRID_H2S/d3_gate.json`。与 §2 记录的 `rl_v3` SHA 一致 |
| 行为策略身份 | v4.5.0 `round2.pt` = `4abee069…` / `a93746bb…` / `c69960fc…`（各 34,387 B），在 v8.6.0 `ckpt_assertions{,_final}.json` 与 v8.12.0 `HYBRID_H2S/ckpt_assertions{,_final}.json` **两轮各 3/3 match** |
| 被评测策略身份 | v8.6.0 `bombdag.pt` = `9c6d5d4a…` / `36a51090…` / `885dde70…`（各 39,069 B）：本节实测 6 个副本（3 证据 + 3 原件）**逐字节相同**；`eval_*.summary.json`、`HYBRID_H2S/ckpt_assertions*`、`HYBRID/RUN.md` §2.1 三处记录一致 |
| H3 策略身份 | `v8.3.0` `round2.pt` = `6e508206…` / `a04366c3…` / `5e9bcb60…`（各 39,059 B，`HYBRID/RUN.md` §2.1 记录；原件在 `.tmp_v830_bombpol/ckpt/`，**不在本轮登记范围**） |
| D3 保真门 | 三处 **5/5 PASS**：v8.6.0 `fidelity_d3.json`、`HYBRID/fidelity_d3.json`、`HYBRID_H2S/d3_gate.json`（后者 `exit=0`、`passed=5/5`、2,165 前缀 checkpoint 0 mismatch、R24 18/18、PARK 8663） |
| 音频（§9 硬约束） | 三处审计合计 **106/106 muted、`audible=0`、`errors=0`**：v8.6.0 **62**（`all_muted=true`）、`HYBRID` **32**、`HYBRID_H2S` **12**；三处 `launch_env` 均记 `SDL_AUDIODRIVER=dummy`、`DISPLAY`/`WAYLAND_DISPLAY` 为 null；共享审计件 `v4.3.1/evidence/R14/audio_cfg_audit.json` 明确**未触碰** |
| 进程残留 | 三处 `matching=0`：v8.6.0（宿主另有 48 个同名引擎进程属别轮）、`HYBRID`（`all_pass=true`）、`HYBRID_H2S`（宿主 17 个属别轮）；三处均未 `pkill -f` |
| v8.6.0 结论口径 | `labels_stats.json` 的 `p0.pass=true`（2→3,378、1→1,038，门槛 200，余量 16.9× / 5.2×）；`RUN.md` §0 判 **`S-partial`**（P1 成立、P2/P3 不成立）；`diagnostics.json` 的 `budget_comparison_warning` 明确「三个 budget 不能当同一测量比均值」 |
| v8.12.0 结论口径 | `HYBRID/verdict.json`：`P1` = H2s d600 **0.979167**（705/720，CP95 [96.59 %, 98.83 %]）**pass**；`P2` = H2s > H1（单侧 Fisher p = 0.0）**pass**；`P3` = `value=null / pass=false`（省雷未被检验）；`P4` 放雷均值 2.9889；`stops = {stop: "S-deploy"}`。`HYBRID_H2S/summary.json` 的 `pooled=236/240=98.333 %` 是 H2s 在**更小协议**下的复现 |
| H1 采纳链 | `h1_adoption.json` 的 `precondition.crosscheck_all_pass=true` + `crosscheck.json` 的 20/20 逐字段相同（参照 `eval_d600_seed0.jsonl` 的 `1948690b…`）+ 9 份 jsonl 与 v8.6.0 逐字节相同 ⇒ 「H1 = v8.6.0 评测网格」是**实测**而非声明 |

#### 22.4.4 登记期零漂移（`v8.6.0/**` 与 `v8.12.0/**` 只读）

登记**开始前**（`2026-11-22T13:31:51` 本地，`snap_pre`）与**结束后**（`2026-11-22T13:32:24` 本地，`snap_post`）各采集一次 **146** 个文件的 `{size, st_mtime_ns}`（并对每条重算 sha256，与本节表格逐一相同**后才**动手写清单）并逐项比对：

| 范围 | 文件数 | 差值 |
| --- | ---: | --- |
| `Document/Update/Alpha/v8.6.0/**` | 57 | `size` 0 变化 / `mtime_ns` 0 变化 |
| `Document/Update/Alpha/v8.12.0/**` | 86 | `size` 0 变化 / `mtime_ns` 0 变化 |
| `.tmp_v860_bombdag/ckpt/seed{0,1,2}/bombdag.pt` | 3 | `size` 0 变化 / `mtime_ns` 0 变化 |
| **合计** | **146** | **146/146 完全相同，0 个变化** |

本轮对上述路径的写操作 = **0**（只读 `Get-ChildItem` / `os.walk` / `sha256sum` / `hashlib` / `Get-Content` / `ConvertFrom-Json`）；**未新建、未改名、未删除、未移动**任何文件；写入**只落在** `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256` 两个清单文件上。

#### 22.4.5 只报告、不裁定的偏差清单

1. **v8.6.0 的冻结预注册在轮内被改过**：`ckpt_assertions.json`（`phase=start`，`11:06:17`）记预注册 sha256 = `8df9bdc7…`，而 `ckpt_assertions_final.json`（`phase=final`，`12:12:09`）记 **`2d10c507…`**；**盘上现值 = `2d10c507…`**（= final 相）⇒ 预注册文件在 `11:06–12:12` 之间**发生了变化**。本节按盘上值登记，只记录该事实。
2. **v8.6.0 的执行器在轮内被改过**：`ckpt_assertions.json` 记 `tools/rl/v860_bombdag.py` = `7b1f9119…`，`ckpt_assertions_final.json` 记 `3433ae74…` ⇒ 同一轮内执行器换过版本；证据目录里的 `runner_collect_8fab2e63.py`（`8fab2e63…`）是**第三份**，三者互不相同，不能互相替代。
3. **v8.12.0 H2s 补跑的执行器同样换过版本**：`HYBRID_H2S/ckpt_assertions_start.json` 记 `tools/rl/v8d0_h2s.py` = `8974d099…`，`ckpt_assertions_final.json` 记 `1d25fe57…`；两相的**预注册 sha 相同**（`c937c168…`），故这是执行器变更、不是预注册漂移。
4. **`HYBRID` 与自身 manifest 的 4 处偏差**（§22.4.2 表）：3 处追写 + 1 处 `0` 字节截断，发生在 `13:21:27–13:22:02`（本地）同一分钟内。
5. **`HYBRID` 的 H2/H3 每格只有 2–6 行**：这是**确定性臂**的必然结果（`determinism_probe.json` 的 `distinct=1`），不是采集中断；但因此 `H2_d600` 的 9 个 episode **只有 3 条不同轨迹**，`verdict.json` 同时报 `episode_pool` 与 `distinct` 两套口径，本册照录不重判。
6. **v8.12.0 目录在登记窗口之外发生过写入**：`evidence/HYBRID/` 在 `13:19–13:22`（本地）由**上一任 v8.12 执行代理**写出（各文件 mtime 跨度 `03:53:59Z–05:21:27Z` 是它们各自的产出时间，`eval_H2_d600_seed2.jsonl` 的 `05:22:02Z` 是截断时刻），`13:22:02Z` 后再无变动；本代理对该目录**只读**，且本节全部 131 条 sha256 都在 `13:31:51` 的快照中**逐条复算复核过**才写入清单，零漂移证据覆盖 `13:31:51` 之后的全部窗口（§22.4.4）。

> **一句话结论**：本轮把 **v8.6.0（BOMBDAGGER）46 条 + v8.12.0（`PREREGISTRATION_HYBRID_001.md` + `HYBRID_H2S`）26 条 + v8.12.0/HYBRID 59 条 = 131 条 / 266,282,420 B** 成对写入 `ASSET_SYNC.md` §22 与 `ASSET_SYNC.sha256`，`sha256sum -c` 由 **299 OK / 1 FAILED** 变为 **430 OK / 1 FAILED**（新增 131 条全 OK，FAILED 数仍为 1，唯一 FAILED 仍为已知允许的 `v4.2.0/journal.jsonl`）。两轮目录实测 **v8.6.0 57 文件 / 133,935,659 B**、**v8.12.0 86 文件 / 132,395,489 B**，其中未入册的 15 件（`collect/` 12、`smoke/` 2、被截断的 `eval_H2_d600_seed2.jsonl` 1）按 §22.2 逐组说明理由，前 14 件的字节身份由入册 `manifest.json` 覆盖（55/55 实测相符）。本轮**未启动引擎、未 `git add/commit`**，146 个受保护文件的 `{size, mtime}` **逐个零漂移**。

## 23. v8.11.0 与 v8.15.0 资产（2026-11-22）

> **为什么有这一节**：AGENTS.md §8.2 要求「新增/移动/删除任何资产，必须同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`」。**v8.11.0（EVENTSAMPLE）与 v8.15.0（CRITICAL）到本节追加前一行都没有入册** —— 对既有 **431 行**做 `v8\.11\.0|v8\.15\.0|EVENTSAMPLE|CRITICAL|v8b0_|v8f0_` 匹配是 **0 命中**。本节与 `ASSET_SYNC.sha256` 末尾追加的 **35 行**成对登记。
>
> **两轮性质不同**：v8.11.0 是 **live engine 真采集**轮 —— 12 格 × 834 状态 = **10,008 状态 / 360,288 标签**（160 局），P0（三段正例率 ≥25% 且各 ≥200）**不成立** ⇒ 按预注册 §2.1 判 **`S-data`**，**未训练、未评测**；v8.15.0 是 **零采集、零引擎**的**判据扫描**轮 —— 只读 v8.11.0 已发布的 `labels_events.jsonl` 重算「critical 状态」（`max(alive_frames) < H=120`），实测 **`critical_states = 0`** ⇒ 按预注册 §2.1 判 **`S-nodiff`**，同样**未训练、未评测**（`engine_started=false`、`training_performed=false`）。
>
> **只读边界**：本轮**只 append** 两个清单文件。`Document/Update/Alpha/v8.11.0/**`（**27** 个文件）、`Document/Update/Alpha/v8.15.0/**`（**6** 个文件）、`tools/rl/v8b0_eventsample.py`、`tools/rl/v8f0_critical.py` 全程只读；登记开始前与结束后各采集一次这 **35 个文件**的 `{size, st_mtime_ns}` 并逐项比对：**35/35 完全相同，0 变化**（§23.4.4）。**未启动引擎**（本节 0 次引擎调用、0 个游戏进程、未 `git add` / 未 `git commit`，分支仍为 `v400`）。
>
> **实测口径**：35 条的 `bytes` 与 `sha256` **全部现场实测**（Python `hashlib.sha256` 全文件流式读取），再用 WSL `sha256sum -c` 独立复算 **35/35 OK**。两份产出方 `manifest.json` 的内联 sha256 **只用于交叉核对，不代替实测**（核对见 §23.4.2）。
>
> **两个口径都写（本册一贯做法）**：目录口径 = `Get-ChildItem -Recurse -File -Force` 的实测总数；`manifest.json` 自述口径 = 该文件 `file_count` / `total_bytes`，**`excludes="manifest.json"`**，即「除自身」。v8.11.0 `evidence/EVENTSAMPLE`：**26 / 97,665,524**（目录，含 `manifest.json` 自身）vs **25 / 97,661,817**（manifest 口径，除自身；+ `manifest.json` 3,707 B = 目录口径 97,665,524 B）；v8.15.0 `evidence/CRITICAL`：**5 / 19,729**（目录）vs **4 / 18,381**（manifest `files{}` 求和 —— 该 manifest **没有** `total_bytes` 字段；+ 自身 1,348 B = 19,729 B），再加顶层 `PREREGISTRATION_CRITICAL_001.md` 4,356 B = v8.15.0 目录口径 **6 / 24,085 B**。
>
> **全量入册口径（与 §20–§22 的「最小集」不同）**：这两轮**目录里每一个文件都入册**，未入册 **0 个**。理由：v8.11.0 的 `collect/` 12 格摘要与 v8.15.0 的 5 件**都不是「可重跑重建的日志」**，而是该轮**唯一的统计出口**；v8.15.0 的 `labels_critical.jsonl` 虽然是 **0 字节**也照登（见 §23.2，**空文件是有意为之**）。

### 23.1 登记范围（**35 条** = v8.11.0 **27** 条 + v8.15.0 **6** 条 + 执行器 **2** 条）

| 分组 | 条数 | 字节合计 | 位置 |
| --- | ---: | ---: | --- |
| §23.1.1 v8.11.0 冻结预注册 | 1 | 4,211 | `Document/Update/Alpha/v8.11.0/` |
| §23.1.2 v8.11.0 运行记录、脚本与命令表 | 3 | 30,624 | `v8.11.0/evidence/EVENTSAMPLE/` |
| §23.1.3 v8.11.0 标签集与三项统计 | 4 | 97,331,147 | `v8.11.0/evidence/EVENTSAMPLE/` |
| §23.1.4 v8.11.0 门、断言与审计 | 6 | 45,333 | `v8.11.0/evidence/EVENTSAMPLE/` |
| §23.1.5 v8.11.0 逐格采集摘要（12 格） | 12 | 254,713 | `v8.11.0/evidence/EVENTSAMPLE/collect/` |
| §23.1.6 v8.11.0 清单锚 | 1 | 3,707 | `v8.11.0/evidence/EVENTSAMPLE/` |
| **v8.11.0 小计** | **27** | **97,669,735** | — |
| §23.1.7 v8.15.0 冻结预注册 | 1 | 4,356 | `Document/Update/Alpha/v8.15.0/` |
| §23.1.8 v8.15.0 判据统计与独立复核 | 2 | 7,068 | `v8.15.0/evidence/CRITICAL/` |
| §23.1.9 v8.15.0 空标签文件（**0 B**） | 1 | 0 | `v8.15.0/evidence/CRITICAL/` |
| §23.1.10 v8.15.0 运行记录与清单锚 | 2 | 12,661 | `v8.15.0/evidence/CRITICAL/` |
| **v8.15.0 小计** | **6** | **24,085** | — |
| §23.1.11 两轮执行器 | 2 | 153,353 | `tools/rl/` |
| **合计** | **35** | **97,847,173** | — |

#### 23.1.1 v8.11.0 冻结预注册（1 条 / 4,211 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.11.0/PREREGISTRATION_EVENTSAMPLE_001.md` | 4,211 | `ce8c23ff425a7311572ad9cc59bd1e87874f04f5bfcb4834870a64e46a72370e` | v8.11.0 冻结预注册（**FROZEN**，在任何 v8.11.0 采集之前）：单因子 = **采样分布**（R5 触发决策 **±10** 个决策加密采样 + 其余按 **1/10** 抽）；P0 = 三段正例率 **≥25%** **且** 各 **≥200** 正例，任一不成立即 `S-data`；§2.1 五档停止条件（S-data / S-ok / S-partial / S-fail / S-fork）；§5 列「不能声称什么」。**三处独立记录同 sha**（`RUN.md` 头、`ckpt_assertions.json` start 相、`ckpt_assertions_final.json` final 相）⇒ 轮内**一字未改** |

#### 23.1.2 v8.11.0 运行记录、脚本与命令表（3 条 / 30,624 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 2 | `Document/Update/Alpha/v8.11.0/evidence/EVENTSAMPLE/RUN.md` | 14,080 | `bf594a1ae02fed09a7be6b9ef653ad37c00c2fa3342011e28b3f246dbf368711` | **235 行**运行记录：§0 判 **`S-data`**（三段 17.74% / 13.35% / 6.86%，段 2 正例数 **852** 是唯一 PASS 项）；§1 D3 5/5 门 + ckpt 断言 + 音频 14/14；§4 记 `seed2 w15` 的 **RESET 哈希失配 fail-closed**（`d19bdbb84e21085d != 863212b903563a98`，129 状态作废后**单独重跑**）；§5 P0；§12 列 5 条偏差（含 quota 834 的取法与「未训练未评测」） |
| 3 | `Document/Update/Alpha/v8.11.0/evidence/EVENTSAMPLE/exec.sh` | 7,215 | `36f1941f7197e00473a9d02f731d1b1303df0d0044c386854c99ccfb98a52f51` | 一键执行脚本（preflight / assert-ckpt / project / d3-fidelity / smoke / audio / collect / assert-final / manifest / publish 各段的可复现入口）；头部注释写明 AGENTS.md §9.3 的**三处**静音位置：① 脚本自身设 `SDL_AUDIODRIVER=dummy` 并 unset `DISPLAY`/`WAYLAND_DISPLAY`；② runner 内 `v432_fork_arm.build_private_data_dir` 用 `tools/rl/v400_make_cfg.py` 重生成 `th08.cfg` 并 `--inspect` 四字段；③ 四字段非全 0 则**拒绝起引擎** |
| 4 | `Document/Update/Alpha/v8.11.0/evidence/EVENTSAMPLE/commands.tsv` | 9,329 | `802d03e57ee6431f9bacc9f6579af192e78a96724e4266a447bfc8aadbdf2f9c` | **34 行 = 表头 + 33 条命令**的 `label / exit / wall_s / started / finished / argv`（首条 `11:57:47`、末条 `13:44:00`）；其中 **6 条 exit 非 0**：`stats`×3 与 `smoke-stats` 记 **4**、`process-audit` 记 **6**、`collect-s2-w15` 记 **4** 后由 `collect-s2-w15-RELAUNCH` 重跑至 **exit 0**（wall 2,143.9 s）。**`RUN.md` §12 写「共 30 条命令」，与盘上 33 条不符**（只报告，见 §23.4.5 第 3 条） |

#### 23.1.3 v8.11.0 标签集与三项统计（4 条 / 97,331,147 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 5 | `Document/Update/Alpha/v8.11.0/evidence/EVENTSAMPLE/labels_events.jsonl` | 97,313,721 | `ec9c802a87a8beb954545952663b10b6582bc91c83674603ef34322420a0a5da` | **全轮唯一锚**：**10,008 行** × 36 路 = **360,288** 标签（H=120 帧）；每行含 live 计算的 `nearest_all_px` 与最近 **12** 颗弹；被 **三处**引用为同一 `{sha256, bytes}`：`sample_stats.labels_file`、v8.15.0 `critical_stats.source`、v8.15.0 `manifest.upstream` ⇒ 发布后未被改写。**与 `.tmp_v8b0_eventsample/labels/` 12 份逐格标签按文件名顺序 `cat` 的 sha256 逐字节相同**（§23.4.3） |
| 6 | `Document/Update/Alpha/v8.11.0/evidence/EVENTSAMPLE/sample_stats.json` | 8,857 | `6db5532c3abe36fa85373a52f1f0f50258971d561c9c1289fb46b648811ccfee` | **P0 主判据**：`states=10,008`、`labels=360,288`、pooled 正例 **1,199 / 11.98%**；三段 902（**160**，17.74%）/ 6,382（**852**，13.35%）/ 2,724（**187**，6.86%）；`p0_pass=false`；`label_checks`: `labels_36=10,008`、`bomb1_press_honoured=72,180`、`bomb1_noop_due_to_invuln=107,964`、`bomb1_anomalies=0`、`bomb0_spent_a_bomb=0`、`replay_checks=46,088` / `replay_mismatches=0`；`cells` 12 条逐格 `{file, sha256, bytes, states}`（本节已实测 **12/12** 相符）；`v810_preview` 记 v8.10 预览的池化正例率 **3.0%**（本轮 11.98%，≈4 倍） |
| 7 | `Document/Update/Alpha/v8.11.0/evidence/EVENTSAMPLE/rule_target_stats.json` | 3,614 | `b78da715bff3093ca431f151ee09b12c7ff3c67c8661666e497c1bc1b9926d24` | 规则**自身**目标的分布，以及 36 路反事实标签是否同意「放雷能买到帧」：`states=10,008`、`rule_bomb_counts={1:1,199, 0:8,809}`；`label_matrix`: 36 路全 safe **6,572（65.67%）**、`mean_alive_frames_over_36=118.5613 / 120`、`mean_safe_actions=34.3571`、**`best bomb=1` 更好 0 / 更差 0 / 相等 10,008**（三段逐段相同）⇒ 在本轮状态上反事实标签**无法表达雷的价值**；`labels_checks` 与 `sample_stats` 的 `label_checks` **逐字段相同** |
| 8 | `Document/Update/Alpha/v8.11.0/evidence/EVENTSAMPLE/projection.json` | 4,955 | `4443bf86bd258fc75466f89d58ded13502d66b74a92982a48baca9b13d2f3bdf` | **「投影，不是测量」**（`kind="projection, not a measurement"`）：把冻结采样规则重放到 v8.6.0 的 **5,040** 条**逐决策**轨迹（源 `e1f47805…` / 35,189,092 B）上，在采集**之前**预估 P0 可达性；`projected_p0` 三段 `positive_rate_met` / `min_positives_met` **全 false**（0.189 / 0.158 / 0.050）⇒ **采集前已预示 P0 不成立**；`episodes=24`、`per_episode_samples` 均值 45.54（21–79）、`wall_estimate_s_at_12_way=2704.9` |

#### 23.1.4 v8.11.0 门、断言与审计（6 条 / 45,333 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 9 | `Document/Update/Alpha/v8.11.0/evidence/EVENTSAMPLE/fidelity_d3.json` | 26,658 | `4681bfa43a9713a293d6f10116faf84a2de74e2d89349911df49cf1e2f73e035` | D3 保真门 **5/5 PASS**（`all_pass=true`、`verdict="PASS: §2.1-§2.5 hold on this binary and data dir"`）：前缀 **2,165** checkpoint、**0** mismatch；R24 常量 18/18；PARK 落在 frame **8,663**；引擎 `691bfd77…` match；`mode=fidelity` / `seed=12` / `deviations=[]` |
| 10 | `Document/Update/Alpha/v8.11.0/evidence/EVENTSAMPLE/preflight.json` | 8,005 | `35d25e26d30d7ecf4d7d5234ba98a7b2df3af630a0e6732f8c9ba61e0c9b33be` | 开跑前自检：**12 个 pinned import `all_match=true`**、`feature_dim=106`、动作 **36**（18 候选 × 2 bomb）、`frozen` 协议、采样规则、两臂定义、`references`；`all_pass=true`；`launch_env` 记 `SDL_AUDIODRIVER=dummy`、`DISPLAY`/`WAYLAND_DISPLAY` 为 **null** |
| 11 | `Document/Update/Alpha/v8.11.0/evidence/EVENTSAMPLE/ckpt_assertions.json` | 3,623 | `1ae7f15f3cfff5b22fffde8e39a1611ebefad7ba4b4f57688e62c2d58f57f62d` | `phase=start`（`11:57:49+0800`）：v4.5.0 `round2.pt` **3/3 match**（`4abee069…` / `a93746bb…` / `c69960fc…`，各 34,387 B）+ 引擎 `691bfd77…` + 预注册 `ce8c23ff…`，`all_pass=true`；**记执行器 `b0942d1c…`**（与 final 相不同，见 §23.4.5 第 1 条） |
| 12 | `Document/Update/Alpha/v8.11.0/evidence/EVENTSAMPLE/ckpt_assertions_final.json` | 3,623 | `2c1384b6b2c18e885c7e79e64e0a5c90f04789205eb9ae97c9b61e90bf3df854` | `phase=final`（`13:43:55+0800`）：同上全部 match、`all_pass=true`；**预注册 sha 与 start 相相同**（`ce8c23ff…`）；执行器改记 **`5b3252cf…`（= 盘上现值）** ⇒ 两相共同证明预注册与被评测 ckpt 未漂移，但**执行器在轮内换过版本** |
| 13 | `Document/Update/Alpha/v8.11.0/evidence/EVENTSAMPLE/audio_audit.json` | 2,811 | `365b8f0d82b0abeecd2d8b0e2bce45fe34dc496315ace315cc7d6ff2b95d2a1c` | AGENTS.md §9.3 审计：`cfg_files=14 / decoded=14 / **muted=14** / **audible=0** / errors=0`、`muted_paths` 14 条、`audible=[]`；共享审计件 `Document/Update/Alpha/v4.3.1/evidence/R14/audio_cfg_audit.json` 明确**未触碰** ⇒ 本轮**未新增 `audible`** |
| 14 | `Document/Update/Alpha/v8.11.0/evidence/EVENTSAMPLE/process_residue.json` | 613 | `b0ec1ed48de525b7312f68cb858510fba46a931d3e47406055e793fa8d065d8c` | 本轮引擎残留 **`matching=0`**（`data_dir_filter=.tmp_v8b0_eventsample`）；宿主按引擎名共 **12** 个 `th08-modern` 属**别轮**（`processes_total_by_engine_name=12`），**未 `pkill`**、未触碰；`all_pass=true` |

#### 23.1.5 v8.11.0 逐格采集摘要（12 条 / 254,713 B）

12 格 = seed ∈ {0,1,2} × **训练窗口** ∈ {13,14,15,16}（`preflight.json` 的 `frozen.TRAIN_WINDOWS`；`DEV_WINDOWS` 仍是 {11,12,17,18}，本轮只采训练窗口），**每格 quota 834 状态 / 30,024 标签**（834 × 36），合计 **10,008 状态 / 360,288 标签**；每格记 `status=complete`、`replay_checks`、`gate.prefix={hash_checks:2165, hash_mismatches:0}`、`cfg` 四字段全 0、以及该格标签文件的 `labels_sha256` / `labels_bytes`（本轮已逐格实测 12/12 相符）。

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 15 | `.../EVENTSAMPLE/collect/seed0_w13.json` | 21,023 | `3ece9e5cfa14cad93c9b45f0a8c03274cabb318f905f3608f9ba07585e75fa71` | seed0 / w13：834 状态、正例 **116**、wall **3,213.5 s**、`replay_checks=3,759`（0 mismatch）；格内标签 `06c5eef5…` / 7,969,097 B |
| 16 | `.../EVENTSAMPLE/collect/seed0_w14.json` | 22,727 | `943b5b1a693a005690007001aa2a8553be7c1d353c611825ba0e17f4ff36881c` | seed0 / w14：834 / **114** / 3,180.3 s / 4,348；标签 `04a2cfc9…` / 8,114,525 B |
| 17 | `.../EVENTSAMPLE/collect/seed0_w15.json` | 21,030 | `48e7a075d606dbbe72dc27018b6dbd0a4f4dba9fb50f2c9b57bdf87d3eacbfcd` | seed0 / w15：834 / **117** / 3,216.9 s / 3,791；标签 `7a9e6b4c…` / 8,181,244 B |
| 18 | `.../EVENTSAMPLE/collect/seed0_w16.json` | 21,171 | `3bd177485a89c798c75664e36219cd3c1ed6612c5b60bebc0a1d8062f2f05212` | seed0 / w16：834 / **27** / 3,295.7 s / 3,991；标签 `fb712b41…` / 8,139,097 B |
| 19 | `.../EVENTSAMPLE/collect/seed1_w13.json` | 21,403 | `a47b50822cb061854ad55e8bfa578e379523d82993bb70c02f2fc56cb11e7166` | seed1 / w13：834 / **143** / 3,300.3 s / 3,614；标签 `749c41df…` / 7,944,703 B |
| 20 | `.../EVENTSAMPLE/collect/seed1_w14.json` | 20,933 | `c8f2ea03dd7e39794e89c1d4c0b1ab5ec798a96734a4b5ed690e153efbd928c5` | seed1 / w14：834 / **107** / 3,200.9 s / 3,628；标签 `f1bdbac3…` / 8,170,562 B |
| 21 | `.../EVENTSAMPLE/collect/seed1_w15.json` | 20,998 | `f87973125996364192598e0cba82e494bfa36ca47d5a72db21e1566374ab2841` | seed1 / w15：834 / **113** / 3,233.9 s / 3,764；标签 `cf25257f…` / 8,241,804 B |
| 22 | `.../EVENTSAMPLE/collect/seed1_w16.json` | 22,074 | `9dfe15b4f800201d91ac27d130be3ff0ee4d08fce754b4562cdbb7511a96b819` | seed1 / w16：834 / **28** / 3,315.9 s / 4,306；标签 `40516864…` / 8,129,519 B |
| 23 | `.../EVENTSAMPLE/collect/seed2_w13.json` | 21,372 | `864f0f1dbfe72abb5b23cf6a132d523f48219d297d9ffb5364e776bbc02966f2` | seed2 / w13：834 / **143** / 3,283.3 s / 3,718；标签 `5ff80739…` / 7,975,905 B |
| 24 | `.../EVENTSAMPLE/collect/seed2_w14.json` | 20,951 | `e07cb5968e64a8e38c822e4e61e43045d1203698810395c030fb1863e0852400` | seed2 / w14：834 / **105** / 3,137.6 s / 3,790；标签 `7a93096e…` / 8,072,016 B |
| 25 | `.../EVENTSAMPLE/collect/seed2_w15.json` | 21,354 | `e51a3f7192559065333e39a28e69795dbe9414632ef22258a7509037f258ba0e` | seed2 / w15：834 / **139** / **2,143.9 s**（此格是 RESET 失配后**单独重跑**的结果，见 §23.1.2 第 2 条）/ 3,788；标签 `d1f2e311…` / 8,225,935 B |
| 26 | `.../EVENTSAMPLE/collect/seed2_w16.json` | 19,677 | `6d79eb187e0bd335199ebd938568d74b5c0b283106c792302b27ddf186663e44` | seed2 / w16：834 / **47** / 3,236.4 s / 3,591；标签 `963abd87…` / 8,149,314 B |

**逐格正例求和 = 1,199**，与 `sample_stats` 的 pooled 正例 **1,199** 逐位相同 ⇒ 12 格摘要与合并统计**口径闭合**（`replay_checks` 12 格求和 = 46,088 = `sample_stats.label_checks.replay_checks`）。

#### 23.1.6 v8.11.0 清单锚（1 条 / 3,707 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 27 | `Document/Update/Alpha/v8.11.0/evidence/EVENTSAMPLE/manifest.json` | 3,707 | `db12eb75a632e6412d5f6b8dc3beb781b74445f8e49264773616ab4880a9862c` | **清单锚**：`file_count=25`、`total_bytes=97,661,817`、`excludes="manifest.json"`；`files{}` **25** 条 `{sha256, bytes}` —— 本节实测 **25/25 全等、0 处不符**，`bytes` 求和 = 97,661,817（+ 自身 3,707 = 目录口径 97,665,524）。`created_at=13:44:15+0800`、mtime **13:44:14.548** 是目录内**最后一个写入**的文件（晚于 `commands.tsv` 13:44:11、`RUN.md` 13:43:56）⇒ **没有** §22 那种「manifest 是旧快照」的问题 |

#### 23.1.7 v8.15.0 冻结预注册（1 条 / 4,356 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 28 | `Document/Update/Alpha/v8.15.0/PREREGISTRATION_CRITICAL_001.md` | 4,356 | `90bae227354aabdd1a499e7e2c1498adac6392b170b136d01666887c16090f97` | v8.15.0 冻结预注册（**FROZEN**，在任何 v8.15.0 采集之前）：**只采「18 个候选全部在 H=120 内死亡」的 critical 状态**（`max(alive_frames) < 120`），每状态仍 36 路标签；P0 = **≥1,200** critical（三段各 **≥200**）**且** `best bomb=1 > best bomb=0` 占比 **≥20%**；§2.1 停止条件（S-nodiff / S-ok / S-partial）。**本轮判 `S-nodiff`** ⇒ §1 列出的训练 A/B 两臂**未执行**（不是漏做） |

#### 23.1.8 v8.15.0 判据统计与独立复核（2 条 / 7,068 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 29 | `Document/Update/Alpha/v8.15.0/evidence/CRITICAL/critical_stats.json` | 6,403 | `13f68c8c18773b2428e98d41867ef5f7dbd785ce61d0b48048d0801a5b387e59` | **P0 判据主件**：`engine_started=false`、`collection.kind="none (prereg §3 prescan)"`；`totals`: 10,008 状态 / **`critical_states=0`** / `diff_states_bomb1_better=0` / `equal_states=10,008` / `all_36_safe_states=6,572（65.67%）` / `states_max_alive_eq_H=10,008` / `mean_alive_frames_over_36=118.5613`；`segments` 三段 critical **各 0**、`p0_critical_target_met=false`；`p0={pass:false, verdict:"S-nodiff"}`；`source` 指 v8.11.0 `labels_events.jsonl`（**97,313,721 B / `ec9c802a…` / 10,008 行，`sha256_match=true`**）；`perflow_sha256=559b4ccd…`；`wall_s=1.186`；`deviations=[]` |
| 30 | `Document/Update/Alpha/v8.15.0/evidence/CRITICAL/independent_check.json` | 665 | `029cf223c77b44efd69035f39c5c5d9f9e2a49673bc52fbefd7e887623683188` | **独立复算**的同一组读数：`rows=10,008` / `critical_states=0` / `diff_states_bomb1_better=0` / `equal_states=10,008` / `mean_of_per_row_max_alive=120.0` / `max_alive_min/median/max=120` / `delta_distinct=[0]`；**`perflow_sha256=559b4ccd…` 与 `critical_stats.json` 相同** ⇒ 两条互不依赖的代码路径给出**同一结果**（`perflow` 定义 = 对每行 `max_alive,best_bomb1,best_bomb0,bombs_remaining` 按文件顺序 `;` 连接后求 sha256） |

#### 23.1.9 v8.15.0 空标签文件（1 条 / **0 B**）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 31 | `Document/Update/Alpha/v8.15.0/evidence/CRITICAL/labels_critical.jsonl` | **0** | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | **0 字节，有意为之，不是缺失**：`critical_states=0` ⇒ **没有任何状态**满足 `max(alive_frames) < H=120` ⇒ 该文件按设计写 **0 行 / 0 字节**；`e3b0c442…b855` **就是空文件的 sha256 常量值**。**三处独立记录同值**：`critical_stats.labels_critical={rows:0, bytes:0, sha256:e3b0c442…}`、`manifest.files{}` 同名条目、`RUN.md` 正文（记同一 sha） ⇒ **不是登记遗漏，也不是产出中断**（详见 §23.2） |

#### 23.1.10 v8.15.0 运行记录与清单锚（2 条 / 12,661 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 32 | `Document/Update/Alpha/v8.15.0/evidence/CRITICAL/RUN.md` | 11,313 | `5171f803039160b7db397a6f9b7088501dd2f59900903e37c62807792d6f13a8` | **210 行**运行记录：判定 **`S-nodiff`**；表头四行记「采集 **零**（prereg §3 预扫）/ 引擎**未启动**（`engine_started=false`，结束时 `th08*` 进程数 **0**）/ 训练评测**未进行**」，执行时刻 **13:51–13:57**；§1 三组 P0 数（critical **0** / 10,008、三段各 0、差异占比 **0.0000%**）；正文含 `ec9c802a…`（2 处）、`e3b0c442…`（空文件 sha）、`559b4ccd…`（`perflow`）三个锚 |
| 33 | `Document/Update/Alpha/v8.15.0/evidence/CRITICAL/manifest.json` | 1,348 | `4d440fe89b3beef2663a986b5dc1fd2fd0bf6d5bbae312abef8a5b20d4a13c9f` | **清单锚**：`file_count=4`、`excludes="manifest.json"`、`verdict="S-nodiff"`、`p0_pass=false`、`engine_started=false`、`training_performed=false`；`files{}` **4** 条（本节实测 **4/4 全等**，`bytes` 求和 = 18,381）；`tool.sha256=dc1150ef…`（= `tools/rl/v8f0_critical.py` 实测值）、`tool.bytes=19,597`；`upstream.labels_events.jsonl=ec9c802a…` + `match_v811_sample_stats=true`；`deviations=[]`。**注意：该 manifest 没有 `total_bytes` 字段**（与 §21/§22 三份不同），故字节口径用 `files{}` 求和 |

#### 23.1.11 两轮执行器（2 条 / 153,353 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 34 | `tools/rl/v8b0_eventsample.py` | 133,756 | `5b3252cf7d201b705f03baf1d73cb82d8c187240f7437b81b83cf340d4865390` | v8.11.0 采集 / 打标 / 统计 / 发布执行器（`--mode preflight|assert-ckpt|project|d3|collect|stats|audio-audit|process-audit|manifest|publish`）；与 `ckpt_assertions_final.json` 记录**一致**，与 `ckpt_assertions.json`（start 相）记录的 `b0942d1c…` **不一致** ⇒ 执行器在轮内换过版本（只报告，见 §23.4.5 第 1 条） |
| 35 | `tools/rl/v8f0_critical.py` | 19,597 | `dc1150ef9c4c79a0d2de098a4021d41c95b8b5b7c33f3f5ca24e81c0fdee131b` | v8.15.0 判据执行器（`mode=scan`：**零采集、不启引擎**，只读 v8.11.0 已发布的 `labels_events.jsonl`）；与 v8.15.0 `manifest.json` 的 `tool.sha256` / `tool.bytes` **均一致** |

### 23.2 跳过说明（未入册 **0** 个文件；含 **0 字节文件**的专门说明）

两轮实测：v8.11.0 **27** 个文件 / **97,669,735 B**、v8.15.0 **6** 个文件 / **24,085 B**，加两个执行器 **2** 个文件 / **153,353 B** ⇒ **35 个文件 / 97,847,173 B，入册 35、未入册 0**。

| 组 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `v8.11.0/**` | 27 | 97,669,735 | **全部入册**（含 `collect/` 12 格摘要与 `exec.sh`）；未入册 **0** |
| `v8.15.0/**` | 6 | 24,085 | **全部入册**（含 **0 字节**的 `labels_critical.jsonl`）；未入册 **0** |
| `tools/rl/v8b0_eventsample.py`、`tools/rl/v8f0_critical.py` | 2 | 153,353 | **全部入册**；未入册 **0** |
| **小计** | **35** | **97,847,173** | — |

**为什么这两轮全量入册、不像 §20–§22 那样列「过程件不入册」**：v8.11.0 的 `collect/` 12 格摘要**不是**可重跑重建的日志，而是每格唯一的状态/正例/回放/门读数（其逐格 `labels_sha256` 指向 `.tmp_v8b0_eventsample/labels/`，见下）；v8.15.0 只有 5 件证据 + 1 份预注册，**没有过程件目录**（无 `logs/`、无 `runs/`、无 `scripts/`）。这两轮**最小集 = 全集**。

**0 字节文件（`labels_critical.jsonl`）：有 意 为 之，不是缺失。**

- **成因链**：`critical_states = 0`（10,008 个状态里没有一个是 `max(alive_frames) < H=120`）⇒ 没有一行可写 ⇒ `rows=0` / `bytes=0`；
- **反向证据**：若这是「写了一半被截断」，`critical_stats.json` 不会同时记 `labels_critical={rows:0, bytes:0}`；该文件 `deviations=[]`、`wall_s=1.186`（1.2 s 跑完 10,008 行），说明扫描**完整结束**；
- **三处独立同值**：`critical_stats.labels_critical`、`CRITICAL/manifest.json` 的 `files{}`、`RUN.md` 正文 —— 三处都记 `e3b0c442…b855` / 0 B；
- **照登理由**：给它钉 sha256 **不会**把「损坏态」写成权威值（`e3b0c442…` 是空文件常量，任何空文件都等于它）；相反，**不入册**会让「`critical=0`」这个本轮最重要的结论失去字节锚。故**照登**，并在本节写明「空文件是有意为之，不是缺失」。
- **同一口径的另一处**：§22.2 拒绝登记的是 `v8.12.0/HYBRID/eval_H2_d600_seed2.jsonl`（**0 B，但 manifest 自述应为 112,712 B** ⇒ 截断件，钉上去等于把损坏态写成权威值）。两者的区别是**manifest 是否与盘上一致**：本例 `manifest` 明确记 0 B / 0 行 ⇒ 一致，故登记。

**运行目录 `.tmp_v8b0_eventsample/`（不在 `Document/` 下，不是发布件）**：实测 **408** 个文件 / **198,621,230 B**，其中 `labels/` 的 **12** 份逐格标签合计 **97,313,721 B**。**这 12 份按文件名顺序 `cat` 出来的 sha256 = `ec9c802a…`，与入册 `labels_events.jsonl` 逐字节相同** ⇒ 逐格标签的字节身份**已由入册件完全覆盖**（§23.4.3），故运行目录其余过程件不入册（与 §21.2 / §22.2 同口径）。

**`__pycache__` / `*.pyc` 实况（按要求只报告、不删）**：`v8.11.0/`（27 文件）与 `v8.15.0/`（6 文件）全树 **没有** `__pycache__/`、**没有** `*.pyc`；两轮目录内**唯一**的可执行文本是 `v8.11.0/evidence/EVENTSAMPLE/exec.sh`（bash，无字节码缓存）。若日后出现，按 §21.2 口径**一律不入册**。

### 23.3 与 §1–§22 不重复的说明（实测已核）

- **既有 431 行 0 命中**：追加前对全部 431 行做 `v8\.11\.0|v8\.15\.0|EVENTSAMPLE|CRITICAL|v8b0_|v8f0_` 匹配 → **0 命中**；§22 登记的是 v8.6.0（BOMBDAGGER）/ v8.12.0（HYBRID），与本轮**不同版本、不同目录**。
- **路径 35 唯一 / 哈希 35 唯一**：35 条路径与 35 个 sha256 逐条比对 **0 组重复**（`ckpt_assertions.json` 与 `ckpt_assertions_final.json` 字节数相同 3,623 B，但 sha **不同**：`1ae7f15f…` vs `2c1384b6…`）。
- **与既有 431 行 0 交集**：35 个路径、35 个 sha256 与既有 **431 个路径 / 416 个唯一哈希**逐一比对，**路径交集 0、哈希交集 0** ⇒ 没有把任何既有行「换个路径再钉一遍」。
- **行数与格式**：431 + 35 = **466 行**；仍是 **LF / 无 BOM**，末行以 `\n` 结束，每行 `64 hex + 两个空格 + 相对路径`，**空行 0 条、格式异常行 0 条**（`ASSET_SYNC.sha256` 实测 58,368 B、CR 字节 **0**）。
- **`ASSET_SYNC.md` 本身不在校验表内**（`ASSET_SYNC` 在 `ASSET_SYNC.sha256` 里 **0 行**）⇒ 追加 §23 **不会**让任何既有行转 FAILED（与 §21.3 / §22.3 同一口径）。
- **Git 跟踪状态（实测）**：`git status --porcelain` 对两目录与两个执行器均为 `??`（`Document/Update/Alpha/v8.11.0/`、`Document/Update/Alpha/v8.15.0/`、`tools/rl/v8b0_eventsample.py`、`tools/rl/v8f0_critical.py`）；`git ls-files -- <两目录>` = **0** ⇒ 这些证据**无法由 Git 重建**，本节 + `ASSET_SYNC.sha256` 是它们目前的字节身份记录。本轮按约束**未** `git add` / `git commit`，分支仍为 `v400`。

### 23.4 校验结论

#### 23.4.1 `sha256sum -c`（WSL，追加前后各一次）

命令（与 §22.4.1 同一口径）：

```bash
cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -c ': OK'; sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -v ': OK'
```

* **基线（追加前 431 行）**：**430 OK / 1 FAILED**。
* **追加后（466 行）**：**465 OK / 1 FAILED**。`465 + 1 = 466` 校验行全部参与。
* **新增 35 条 35/35 OK** ⇒ 整体 FAILED 数**没有新增（仍为 1）**。
* **唯一 FAILED 仍是已知并允许的那一条**：`Document/Update/Alpha/v4.2.0/journal.jsonl`（§9.8 记录的活动文件漂移）。本轮**未改该行、未重钉、未触碰该文件**。
* **格式**：466 行仍为 **LF / 无 BOM**、末行 `\n` 结束、每行 `64 hex + 两个空格 + 相对路径`；空行 **0**、格式异常 **0**；新增 35 条的路径在表内**各自唯一**。

#### 23.4.2 与两份产出方 `manifest.json` 的交叉核对（登记时实测）

| 清单 | 自述 | 实测比对 |
| --- | --- | --- |
| v8.11.0 `evidence/EVENTSAMPLE/manifest.json` | `file_count=25`、`total_bytes=97,661,817`、`excludes="manifest.json"` | `files{}` **25/25** 的 `{bytes, sha256}` 全等、**0 处不符**；`bytes` 求和 = `total_bytes` = 97,661,817；盘上 `EVENTSAMPLE/**` 的另一件只有 `manifest.json` 自身（26 − 25 = 1） |
| v8.15.0 `evidence/CRITICAL/manifest.json` | `file_count=4`、`excludes="manifest.json"`（**无 `total_bytes`**） | `files{}` **4/4** 全等、0 处不符；`bytes` 求和 = **18,381**（+ 自身 1,348 + 顶层预注册 4,356 = 24,085 = 目录口径）；`tool.sha256=dc1150ef…` / `tool.bytes=19,597` 与实测一致；`upstream.labels_events.jsonl=ec9c802a…` 与 v8.11.0 实测一致 |
| v8.11.0 `sample_stats.json` 的 `cells[]` | 12 格各自的 `{sha256, bytes, states}` | **12/12** 与 `.tmp_v8b0_eventsample/labels/` 现场重算全等；`states` 求和 = 10,008；`positive_counts` 求和 = **1,199** = pooled 正例 |
| v8.11.0 `collect/*.json` 的 `labels_sha256` / `labels_bytes` | 每格标签文件身份 | **12/12** 与其 `labels_path` 指向的 `.tmp` 文件现场重算全等、0 处不符 |

**一份 manifest 未覆盖的内容（如实列出）**：`manifest.json` **不含** `tools/rl/v8b0_eventsample.py` 与 `tools/rl/v8f0_critical.py` 的登记（v8.15.0 的 manifest 只记 `tool` 为 `v8f0_critical.py`）；两个执行器的字节身份由**本节第 34–35 条**提供（v8.11.0 侧另有 `ckpt_assertions_final.json` 的 `v8b0_eventsample_py` 作为第二锚）。

#### 23.4.3 身份链、门与读数（只登记，不重判）

| 核对项 | 实测 |
| --- | --- |
| 引擎身份 | `691bfd77b7eca336…` / `all_pass=true`，出现在 v8.11.0 `preflight.json`、`fidelity_d3.json`、`ckpt_assertions{,_final}.json` 与 12 份 `collect/*.json`；与 §2 / §22 记录的 `rl_v3` SHA **一致** |
| 行为策略身份 | v4.5.0 `round2.pt` = `4abee069…` / `a93746bb…` / `c69960fc…`（各 34,387 B），在 `ckpt_assertions{,_final}.json` **两相各 3/3 match** ⇒ 采集期间行为策略未换 |
| **跨轮标签锚** | v8.11.0 `labels_events.jsonl` = `ec9c802a…` / 97,313,721 B / **10,008 行**：`sample_stats.labels_file`、v8.15.0 `critical_stats.source`（`sha256_match=true`、`rows_match=true`）、v8.15.0 `manifest.upstream` **三处同值** ⇒ v8.15.0 读的确实是 v8.11.0 发布的那一份，**中间没有重新打标** |
| **逐格 → 合并的字节闭包** | 12 份 `.tmp_v8b0_eventsample/labels/seed*_w*.jsonl` 按文件名顺序 `cat` 后 `sha256sum` = **`ec9c802a…`**，与入册 `labels_events.jsonl` **逐字节相同** ⇒ 合并件不是重排/重写产物 |
| **独立复算锚（v8.15.0）** | `perflow_sha256` = `559b4ccd400bceced29868128708393a49338a659c19d0916a17420e36d38813`，在 `critical_stats.json`、`independent_check.json`、`RUN.md` **三处相同**；两条代码路径给出同一 `critical=0` |
| **v8.11 与 v8.15 的读数一致性** | 两轮各自算出的 `all_36_safe_states=6,572 / 65.67%`、`mean_alive_frames_over_36=118.5613`、`segments` 状态数 **902 / 6,382 / 2,724**、`equal_states=10,008` **逐项相同** ⇒ 两轮对同一批状态的统计**口径闭合** |
| D3 保真门 | `fidelity_d3.json` **5/5 PASS**（HELLO role=shadow、DEBUG threads=1、前缀 **2,165** checkpoint 0 mismatch、R24 18 个 offset 常量、PARK frame **8,663**），`verdict="PASS: §2.1-§2.5 hold on this binary and data dir"` |
| 音频（§9 硬约束） | v8.11.0：`audio_audit.json` **14/14 muted、`audible=0`、`errors=0`**、`all_muted=true`，12 份 `collect/*.json` 的 `launch_env` 均记 `SDL_AUDIODRIVER=dummy` 且 `DISPLAY`/`WAYLAND_DISPLAY` 为 null，逐格 `cfg.fields` 四零；共享审计件 `v4.3.1/evidence/R14/audio_cfg_audit.json` **未触碰**。v8.15.0：`engine_started=false`、**未起引擎** ⇒ 无新增 `audible` |
| 进程残留 | v8.11.0 `process_residue.json` **`matching=0`**（`data_dir_filter=.tmp_v8b0_eventsample`；宿主 **12** 个同名引擎进程属**别轮**，未 `pkill`）；v8.15.0 未起引擎，`RUN.md` 记结束时 `th08*` 进程数 **0** |
| v8.11.0 结论口径 | `sample_stats.json` 的 `p0_pass=false` 与 `RUN.md` §0 判定的 **`S-data`** 一致：段 3 **160/902 = 17.74%**、段 2 **852/6,382 = 13.35%**、段 1 **187/2,724 = 6.86%**，三段正例率**均 < 25%**（段 2 的正例数 852 ≥ 200 是本轮唯一 PASS 的单项）。**未训练、未评测**（预注册 §2.1 停止条件） |
| v8.15.0 结论口径 | `critical_stats.json` 的 `p0={pass:false, verdict:"S-nodiff"}` 与 `RUN.md` 的 **`S-nodiff`** 一致：**critical 0 / 10,008**、`diff` 占比 **0.0000%**、`equal_states=10,008` ⇒ 「H=120 内雷的价值不可表达」。按预注册 §2.1 转在线 RL，**未训练、未评测** |

#### 23.4.4 登记期零漂移（两轮目录与两个执行器只读）

登记**开始前**与**结束后**各采集一次这 **35** 个文件的 `{size, st_mtime_ns}` 并逐项比对（快照见 `.tmp_assetsync_v8/baseline.tsv`，本节写入前**先逐条重算 sha256 并与表格比对相同后才动手**）：

| 范围 | 文件数 | 差值 |
| --- | ---: | --- |
| `Document/Update/Alpha/v8.11.0/**` | 27 | `size` 0 变化 / `mtime_ns` 0 变化 |
| `Document/Update/Alpha/v8.15.0/**` | 6 | `size` 0 变化 / `mtime_ns` 0 变化 |
| `tools/rl/v8b0_eventsample.py`、`tools/rl/v8f0_critical.py` | 2 | `size` 0 变化 / `mtime_ns` 0 变化 |
| **合计** | **35** | **35/35 完全相同，0 个变化** |

本轮对上述路径的写操作 = **0**（只读 `Get-ChildItem` / `os.walk` / `sha256sum` / `hashlib` / `Get-Content` / `ConvertFrom-Json` / `Select-String`）；**未新建、未改名、未删除、未移动**任何文件；写入**只落在** `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256` 两个清单文件上。

#### 23.4.5 只报告、不裁定的偏差清单

1. **v8.11.0 的执行器在轮内被改过**：`ckpt_assertions.json`（`phase=start`，`11:57:49`）记 `tools/rl/v8b0_eventsample.py` = `b0942d1c…`，`ckpt_assertions_final.json`（`phase=final`，`13:43:55`）记 **`5b3252cf…`**；**盘上现值 = `5b3252cf…`**（= final 相）。同一轮内**预注册 sha 两相相同**（`ce8c23ff…`）⇒ 这是**执行器变更、不是预注册漂移**。本节按盘上值登记，只记录该事实。
2. **v8.11.0 的 `seed2 w15` 格是重跑结果，且发生过程序性 fail-closed**：`RUN.md` §4 记 `FidelityError: window 15: reset hash 'd19bdbb84e21085d' != recorded '863212b903563a98'`，该格 `*.partial` 保留、**未生成**标签文件（129 状态作废），随后单独重跑至 `status=complete` / 834 状态；`commands.tsv` 里对应 `collect-s2-w15`（exit **4**，576.8 s）与 `collect-s2-w15-RELAUNCH`（exit **0**，2,143.9 s）两条。`RUN.md` 把成因列为**推测**（`score.dat` 被引擎改写与 RESET 撞车，约 300 次重置出现 1 次），本节**照录不裁定**。
3. **`RUN.md` 与 `commands.tsv` 的命令条数不一致**：`RUN.md` §12 写「`commands.tsv` 共 **30** 条命令」，盘上实测 **34 行 = 表头 + 33 条命令**（多出 `collect-s2-w15-RELAUNCH`、第二次 `stats`、第二次 `assert-final`/`audio-audit`/`process-audit`/`manifest` 等重跑/补充条目）。两份文件都在本轮登记范围内、**按盘上值登记**，此差异**只报告、未改任何文件**。
4. **`commands.tsv` 有 6 条非 0 退出码**：`stats`×3 与 `smoke-stats` 记 **4**、`process-audit` 记 **6**、`collect-s2-w15` 记 **4**（后重跑至 0）。这些命令的**产物仍然存在且自洽**（`sample_stats.json` / `rule_target_stats.json` / `process_residue.json` 均 `all_pass` 或带完整读数），`RUN.md` 未逐条解释退出码语义 ⇒ 本节**只登记字节身份与退出码事实，不解释、不裁定**。
5. **`labels_critical.jsonl` 为 0 字节**：这是 `critical_states=0` 的直接后果，**有意为之**（§23.2 / §23.1.9 已给出三处独立同值证据），不是缺失、不是截断。
6. **v8.15.0 的 `manifest.json` 缺 `total_bytes` 字段**：与 §21/§22 的三份 manifest 不同，该文件只给 `files{}` 逐条 `{sha256, bytes}` 与 `file_count`。本节按 `files{}` 求和得 18,381 B 作为 manifest 口径，与目录口径 19,729 B 的差额恰为 `manifest.json` 自身 1,348 B ⇒ **无缺件、无多余件**。
7. **两轮都是「未训练、未评测」的轮次**：v8.11.0 判 `S-data`、v8.15.0 判 `S-nodiff`，两者都按**预注册写定的停止条件**停在统计发布处。本节只登记字节身份与已在盘上的读数，**不对结论重判**；两轮预注册 §1 列的 `train_A_*` / `train_B_*` / `eval_*` 产物**本就不存在**（不是登记遗漏）。

> **一句话结论**：本轮把 **v8.11.0（EVENTSAMPLE）27 条 / 97,669,735 B + v8.15.0（CRITICAL）6 条 / 24,085 B + 两个执行器 2 条 / 153,353 B = 35 条 / 97,847,173 B** 成对写入 `ASSET_SYNC.md` §23 与 `ASSET_SYNC.sha256`，`sha256sum -c` 由 **430 OK / 1 FAILED** 变为 **465 OK / 1 FAILED**（新增 35 条全 OK，整体 FAILED 数仍为 1，唯一 FAILED 仍为已知允许的 `v4.2.0/journal.jsonl`）。两轮实测 **v8.11.0 27 文件 / 97,669,735 B**、**v8.15.0 6 文件 / 24,085 B**，**未入册 0 个**（含 **0 字节**的 `labels_critical.jsonl` —— 空文件是有意为之，不是缺失，三处独立记录同值 `e3b0c442…`）；两份 manifest 逐条比对 **25/25 + 4/4 全等、0 处不符**；12 份逐格标签 `cat` 后与入册 `labels_events.jsonl` **逐字节相同**。本轮**未启动引擎、未训练、未评测、未 `git add/commit`**，35 个受保护文件的 `{size, mtime}` **逐个零漂移**。

## 24. v8.16.0 资产（2026-11-22）

> **为什么有这一节**：AGENTS.md §8.2 要求「新增/移动/删除任何资产，必须同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`」。**v8.16.0（DEATHADJ）到本节追加前一行都没有入册** —— 对既有 **466 行**做 `v8\.16\.0|DEATHADJ|v8g0_` 匹配是 **0 命中**。本节与 `ASSET_SYNC.sha256` 末尾追加的 **39 行**成对登记。
>
> **本轮性质**：**live engine 真采集轮** —— `tools/rl/v8g0_deathadj.py`（`HarnessClient role=shadow`，行为策略 = v4.5.0 `S` 随机 + R5 分级放雷 80/60/40 px）在 **900 决策预算**下反复滚到**死亡**，采「死亡前最后 20 个决策」邻域：**12 格 × 60 状态 = 720 状态 / 2,160 行 / 25,920 标签**、**36 条死亡 episode**（另 10 条活满 900 被丢弃），三个视界 **H = 120 / 240 / 600 全部落盘**。P0 判据 = critical（`max(alive_frames) < H`）占 **≥ 50 %** **且** critical 内「`best bomb=1 > best bomb=0`」占 **≥ 20 %**：H=600 的 critical **597 / 720 = 82.92 %（单项达标）**，但三个视界的**「有差异」全部 0** ⇒ 判 **`S-nodiff`**（`verdict.p0_all_pass=false`、`horizons_reaching_gate=[]`），**未训练、未评测**。
>
> **只读边界**：本轮**只 append** 两个清单文件。`Document/Update/Alpha/v8.16.0/**`（**38** 个文件）与 `tools/rl/v8g0_deathadj.py`（**1** 个）全程只读；登记快照采集于 **14:37–14:39**、登记结束后再采一次，这 **39 个文件**的 `{size, mtime_ns}` **39/39 完全相同、0 变化**（§24.4.4）。**未启动引擎**（0 次引擎调用；`Get-Process th08*` 实测 **0** 个；未 `git add` / 未 `git commit`，分支仍为 `v400`）。
>
> **实测口径**：39 条的 `bytes` 与 `sha256` **全部现场实测**（.NET `SHA256` 对全文件一次读入），再用 WSL `sha256sum -c` 独立复算 **39/39 OK**。产出方 `manifest.json` 的内联 sha256 **只用于交叉核对，不代替实测**（§24.4.2：**36/36 的 `{bytes, sha256}` 全等、0 处不符**）。
>
> **两个口径都写（本册一贯做法）**：目录口径 = `Get-ChildItem -Recurse -File -Force` 的实测总数；`manifest.json` 自述口径 = 该文件 `file_count` / `total_bytes`，**`excludes="manifest.json"`**（即「除自身」）。`evidence/DEATHADJ`：**37 / 24,275,217**（目录，含 `manifest.json` 自身）vs **36 / 24,269,969**（manifest 口径）—— 两数之差 **5,248 B 恰为 `manifest.json` 自身** ⇒ **无缺件、无多余件**；再加顶层 `PREREGISTRATION_DEATHADJ_001.md` **3,956 B** ⇒ v8.16.0 目录口径 **38 / 24,279,173 B**，再加执行器 **1 / 158,366 B** ⇒ **39 条 / 24,437,539 B**。
>
> **全量入册口径**：v8.16.0 目录里**每一个文件都入册**，未入册 **0 个**（含 `manifest.json` 自身、`work_manifest.json`、12 份逐格摘要、7 份离线复算脚本与产物）。理由：这 38 件**都不是可重跑重建的日志**，而是本轮唯一的标签 / 统计 / 门禁 / 摘要出口。

### 24.1 登记范围（**39 条** = v8.16.0 **38 条** + 执行器 **1 条**）

| 分组 | 条数 | 字节合计 | 位置 |
| --- | ---: | ---: | --- |
| §24.1.1 冻结预注册 | 1 | 3,956 | `Document/Update/Alpha/v8.16.0/` |
| §24.1.2 运行记录、命令表与清单锚 | 4 | 140,299 | `v8.16.0/evidence/DEATHADJ/` |
| §24.1.3 标签集与三项统计 | 5 | 23,871,525 | `v8.16.0/evidence/DEATHADJ/` |
| §24.1.4 门、断言与审计 | 9 | 52,122 | `v8.16.0/evidence/DEATHADJ/` |
| §24.1.5 离线复算、自测与执行脚本 | 7 | 60,913 | `v8.16.0/evidence/DEATHADJ/` |
| §24.1.6 逐格采集摘要（12 格） | 12 | 150,358 | `v8.16.0/evidence/DEATHADJ/collect/` |
| **v8.16.0 小计** | **38** | **24,279,173** | — |
| §24.1.7 执行器 | 1 | 158,366 | `tools/rl/` |
| **合计** | **39** | **24,437,539** | — |

#### 24.1.1 v8.16.0 冻结预注册（1 条 / 3,956 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.16.0/PREREGISTRATION_DEATHADJ_001.md` | 3,956 | `a820b606a842f3001c33450df2f13732944114a064fa5ef0b723ddc58810a753` | v8.16.0 冻结预注册（**FROZEN**，在任何 v8.16.0 采集之前）：单因子 = **采样位置**（v8.11 的「活轨迹触发邻域」→ 本轮的「死亡前最后 20 个决策」）；P0 = critical ≥ **50 %** **且** critical 内「有差异」≥ **20 %**；§2.1 停止条件（S-nodiff / S-ok / …）。**三处独立记录同 sha**（`preflight.pins.prereg`、`ckpt_assertions.json` start 相、`ckpt_assertions_final.json` final 相，均 `a820b606…` / 3,956 B）⇒ 轮内**一字未改** |

#### 24.1.2 v8.16.0 运行记录、命令表与清单锚（4 条 / 140,299 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 2 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/RUN.md` | 18,649 | `604d93efb9afa0692cc04455fb943d5e47e681fb8d4c1d9bc950d283a19bc67a` | **293 行**运行记录：§0 判 **`S-nodiff`**（「雷在死亡前已经用光」）；§1 记 900 决策为何承重（36 条死亡的死亡决策 **423 / 717 / 895**，**30 / 36 在 600 之后**）；§3 三个 H 的四组数；§4 机制（720 / 720 状态 `bombs_remaining=0` ⇒ `bomb=1` 是 no-op）；§6 门与约束逐项；§7 文件表；§9 **6 条偏差**（含「采了两遍」「采集窗 = dev 窗」「合成自测不是实验证据」）；§10 列「不能声称什么」 |
| 3 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/commands.tsv` | 13,753 | `f5c1ff23b658ae70cc608233802ad6bd093caf772cd8b0b5618c3514e9283a83` | **46 行 = 表头 + 45 条命令**的 `label / exit / wall_s / started / finished / argv`（首条 `assert-start` `13:59:12`、末条 `manifest2` `14:30:15 → 14:30:26`），覆盖 preflight / start / fidelity / recount / smoke / collect（12 路）/ stats / audit / final / manifest / publish 各段；**4 条非 0 退出码**：`smoke2-stats`=4、`stats2`=4、`process-audit2`=6、`stats2`=4（`RUN.md` §8 记 `stats` 在 `S-nodiff` 时 **rc=4 是设计值**）。**`RUN.md` §7 记的是 12,853 B / `5d20d5ed…`（发布前快照）**，本节按工单以 `manifest.json` 记的 **13,753 B / `f5c1ff23…`** 为准（只报告，见 §24.4.5 第 1 条） |
| 4 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/manifest.json` | 5,248 | `4f3b01de101e25f241d64b5f46d74cc574f6c60d835777e4cfde244f9663cc8d` | **清单锚**：`file_count=36`、`total_bytes=24,269,969`、`excludes="manifest.json"`；`files{}` **36** 条 `{sha256, bytes}` —— 本节实测 **36/36 全等、0 处不符**，`bytes` 求和 = `total_bytes` = 24,269,969（+ 自身 5,248 = 目录口径 37 / 24,275,217）。`created_at=14:30:28`、mtime **14:30:27** 是目录内**最后一个写入**的文件（晚于 `commands.tsv` 14:30:25、`RUN.md` 14:30:11）⇒ 没有「manifest 是旧快照」的问题 |
| 5 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/work_manifest.json` | 102,649 | `a4b35383c5f94ba06eca123fe4ab915f8754d395b85bd77740868dcc0fd8e557` | **运行目录清单**（`--mode manifest` 产出）：覆盖整个 `.tmp_v8g0_deathadj/`（含 `data/` 408 件、`logs/` 66 件、`runs/` 48 件、`work/` 42 件等）的逐文件 `{size, sha256, mtime}`。它是**运行现场的唯一全量字节记录**；但**不含**执行器 `tools/rl/v8g0_deathadj.py` 的登记（0 命中，执行器身份见第 39 条与 §24.4.5 第 3 条） |

#### 24.1.3 v8.16.0 标签集与三项统计（5 条 / 23,871,525 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 6 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/labels_deathadj.jsonl` | 23,745,483 | `2bffac3be71f8166f72afee0d47588b0ed03d30cb9226f5e56b6320f08bb252d` | **全轮唯一锚**：**2,160 行**（720 状态 × 3 视界）× 36 路 = **25,920 标签**。被 `deathadj_stats.labels_file` 以同一 `{path, sha256, bytes, rows}` 引用，且**与运行目录 `.tmp_v8g0_deathadj/labels/` 12 份逐格标签按文件名顺序 `cat` 的 sha256 逐字节相同**（§24.4.3）⇒ 合并件不是重排 / 重写产物 |
| 7 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/deathadj_stats.json` | 59,946 | `79f1a3b2de932e2b04765edaaf0125ac9670cd48775abb5071e2516f4bfa2bb7` | **P0 主判据**：`verdict={name:"S-nodiff", diff_gate:0.2, horizons_reaching_gate:[], p0_all_pass:false}`；逐 H：**120 → critical 59（8.19 %）/ 240 → 63（8.75 %）/ 600 → 597（82.92 %）**，「有差异」**三个 H 全 0**；`pooled.bombs_remaining_counts={"0":720}`、`rule_bomb_counts={"0":720}`、`death_decisions={n:36, mean:717.36, median:717, min:423, max:895}`；`label_checks`: `labels_36=720`、`bomb1_press_honoured=0`、`bomb1_noop_due_to_invuln=12,960`、`bomb1_anomalies=0`、`bomb0_spent_a_bomb=0`、`states_without_a_bomb_left=720`、`death_episodes=36`、`no_death_episodes=10`；`nesting_checks={states:720, mismatches:0, alive_is_truncation:true}`；`bombs_at_death.counts={"0":36}`；`reference_v815={states:10,008, critical:0}` |
| 8 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/deathadj_stats_pass1_quota50.json` | 59,855 | `0f7f4981572d9f0f30c64f4b586234b73bf66c699fcf71a3a4bdb688aef88947` | **第一遍**（12 格 × quota 50 = **600 状态 / 1,800 行**、36 条死亡 + 10 活满；每格第 3 条 episode 的窗口被 quota 截成**前 10 个**决策）的统计：`verdict` 同为 **`S-nodiff`**、critical **40 / 40 / 484**（**6.67 % / 6.67 % / 80.67 %**）、「有差异」**仍全 0**、`bomb1_equal=600`。按 `RUN.md` §9 偏差 1 保留为**过程证据**，**不是**入册标签的统计口径；其标签文件 `d33e5ee9…` / 19,798,905 B **已被第二遍覆盖、盘上不存在** |
| 9 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/p0_extras.json` | 4,761 | `be8dfe00cd6e55ab9c135daa66f65ecf3da051620f2d2dce53bed9eeb7e2d4b1` | 补充分布（离线复算，`rows=2160`）：36 条死亡决策逐个列出（`423…895`，`gt_600=30`、`gt_300=36`）；逐 H 的 `safe_actions_total` **9,772 / 6,306 / 378**、`states_all_36_safe` **4 / 0 / 0**、`states_all_36_dead` **16 / 16 / 16**；逐 H 的 `bombs_remaining` 与 `rule_bomb` 分布均为 `{"0":720}` |
| 10 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/p0_sensitivity.json` | 1,480 | `43ec27949865a96c53156cdc125049f301ae2bc47e62518e53465155fd932bc3` | 敏感性（剔除 36 个「死亡决策」退化状态后）：全部 720 → `8.19 % / 8.75 % / 82.92 %`；剔除后 **684** 状态 → **3.36 % / 3.95 % / 82.02 %**；仅死亡决策 36 → `100 % / 100 % / 100 %`；**三组子集的「有差异」全部 0** ⇒ 结论与主表一致 |

#### 24.1.4 v8.16.0 门、断言与审计（9 条 / 52,122 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 11 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/preflight.json` | 9,014 | `4fa116cc2b70c5dae9dd2889178a519dbfa2330fe2ea0f25b42ad2d86f76d6af` | 开跑前自检（`14:13:52`）：**12 个 pinned import `all_match=true`**、引擎 `691bfd77…` match、`feature_dim=106`、动作 **36**（18 候选 × 2 bomb）、`frozen` 协议（TRAIN 13–16 / DEV 11·12·17·18 / `decisions_cap=900` / `p0_critical_min=0.5` / `p0_diff_min=0.2`）、`launch_env` 记 `SDL_AUDIODRIVER=dummy` 且 `DISPLAY`/`WAYLAND_DISPLAY` 为 **null**；`all_pass=true`。**另记执行器 `5ff6c9fd…` / 157,602 B**（与盘上现值不同，见 §24.4.5 第 3 条） |
| 12 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/ckpt_assertions.json` | 4,013 | `6ad511857e1fa8eec2b2e3ae7eec00edcde5d32975def7e2346ece97a7b7f493` | `phase=start`（`14:13:55`）：v4.5.0 `round2.pt` **3/3 match**（`4abee069…` / `a93746bb…` / `c69960fc…`）+ 引擎 `691bfd77…` + 预注册 `a820b606…`，`all_pass=true`；记执行器 `5ff6c9fd…` |
| 13 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/ckpt_assertions_final.json` | 4,013 | `a428abfaf95fa172ae06c6ef79f1fb22976aa60b41dc5b5c4dc8455b4faabd9f` | `phase=final`（`14:27:45`）：同上 **全部 match**、`all_pass=true`；**预注册 sha 与 start 相同**（`a820b606…`）⇒ 采集期间预注册与被评测 ckpt **未漂移**；执行器改记 **`74dc5a26…`**（与 start 相、也与盘上现值都不同，见 §24.4.5 第 3 条） |
| 14 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/fidelity_d3.json` | 26,699 | `0580437cfce2d95b85a88c566cc9d635eaa5cd8ae6625037fe2f716cc50a0dc5` | D3 保真门 **5/5 PASS**（`all_pass=true`、`verdict="PASS: §2.1-§2.5 hold on this binary and data dir"`）：`1_hello_role_shadow`（`role=shadow`）、`2_debug_threads_1`、`3_prefix_2165_checkpoints`（**2,165 / 2,165**、**0 mismatch**、reset hash `83d95375…`、PARK frame 8663）、`4_r24_18_constants`（18/18）、`5_park_frame_8663`；引擎 sha 与 expected 相同 |
| 15 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/audio_audit.json` | 4,370 | `e993b0718d82078cfa82bfc2af4bb1baa61f012d1150fff6da719cd3547310d8` | AGENTS.md §9.3 主 run 审计（`scope=.tmp_v8g0_deathadj/data`）：`cfg_files=24 / decoded=24 / **muted=24** / **audible=0** / errors=0`、`all_muted=true`；`by_root` 24 个私有 data 目录逐目录 `{muted:1, audible:0}`；`muted_paths` 24 条；共享审计件 `v4.3.1/evidence/R14/audio_cfg_audit.json` 明确**未触碰** |
| 16 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/audio_audit_smoke.json` | 732 | `661237409d375a8404457d1a6e9818eb142c8adcfb45ab27c3ef602842c0a65f` | 冒烟轮的音频审计：`cfg_files=1 / decoded=1 / muted=1 / audible=0 / errors=0`、`all_muted=true` |
| 17 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/audio_audit_fidelity.json` | 840 | `7ccb07d32899aee70c6e1aa8a3c5a5b1989da053cedee8a8cd45d8a20ab4e714` | D3 保真轮的音频审计：`cfg_files=2 / decoded=2 / muted=2 / audible=0 / errors=0`、`all_muted=true`。**三处审计合计 27 个 cfg / 27 muted / 0 audible / 0 errors** |
| 18 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/process_residue.json` | 692 | `c4340eb15c0d08e26a14b43992db22b1f2fefdcb48243e24a3e6ce2f73f5a5d7` | 本轮引擎残留 **`matching=0`**（`data_dir_filter=.tmp_v8g0_deathadj`）、`processes_total_by_engine_name=0`、`pgrep_returncode=1`、`lines=[]`、`all_pass=true` ⇒ 本轮**没有**留下引擎进程，也**没有** `pkill` 别轮进程 |
| 19 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/recount_v811.json` | 1,749 | `4bb553df4f069aae8bdb4d96b83d57cadace8f19555b710ff11a55acd5dcb45c` | **零引擎复算**：本轮用**自己的归约代码**重算 v8.15.0 的 `critical = 0 / 10,008`（不直接引用旧数），结果是同一读数 ⇒ 跨轮基线可引用。它使「v8.16 的 critical 不是 0」这一对比**建立在两个独立实现上** |

#### 24.1.5 v8.16.0 离线复算、自测与执行脚本（7 条 / 60,913 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 20 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/cell_progress.py` | 3,159 | `8a3ddd5b420d5926f85c54a2ffa5da5db5d1e7c255aef35dc4ebc33f4e27d1a9` | 逐格进度 / 预算观察工具（离线，无引擎）：读 `collect/*.json` 与 `commands.tsv`，打印 12 格的 states / 死亡数 / wall 与剩余预算 |
| 21 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/selftest_stats.py` | 3,569 | `6aa6940af572bce7acc0050477cc5495163498d2c1411d9a4acc8823a3cffb23` | **合成自测**（预注册 §4 未列的额外产出，`RUN.md` §9 偏差 4）：用 3 个手工状态（全活满 / 全早死 / 仅 `bomb=1` 更久）验证 `deathadj_stats` 的算术 —— **是代码自测，不是实验证据** |
| 22 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/selftest_stats_synthetic.json` | 33,899 | `8d2b3762db4105ec5b511f310ae97c0c4301b5ca7e9823eba06695bb225ed291` | 上述合成自测的**输出**（合成输入 ⇒ 合成读数）。**不得**被当作实验读数引用；登记它是为了让它有字节身份、可被识别为合成件 |
| 23 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/p0_extras.py` | 3,602 | `8bd6627217bc34116831c9040e6fd480d7e3d7d57c8efeae6d0bdd9ea96b16f8` | `p0_extras.json` 的复算脚本（离线，只读 `labels_deathadj.jsonl`）：死亡决策分布、逐 H 的 safe 计数、`bombs_remaining`/`rule_bomb` 分布 |
| 24 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/p0_sensitivity.py` | 1,923 | `14bd092cdb759e7b9abebdfa963a2c0578a6f44ba3497fb0166aae5da900294e` | `p0_sensitivity.json` 的复算脚本（离线）：三组子集（全部 / 剔除死亡决策 / 仅死亡决策）的 critical 与差异占比 |
| 25 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/exec2.sh` | 7,448 | `df3254734b1e7fa04bc3ede78c7233037d26696c2bdebb291539bc28a82fb886` | 本轮一键执行脚本（REDO 版；`preflight / start / fidelity / recount / smoke / collect / stats / audit / final / publish` 各段入口）。头部写明 AGENTS.md §9.3 的**三处**静音位置：脚本自身 `SDL_AUDIODRIVER=dummy` + unset `DISPLAY`/`WAYLAND_DISPLAY`；`v400_make_cfg.py --out` 逐 data dir 生成 cfg 并 `--inspect` 四字段；四字段非全 0 则**拒绝起引擎** |
| 26 | `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/exec_previous_600decisions.sh` | 7,313 | `f68533f622e834b1aea4852423190eee0b3506d105d80b03188b893c13808185` | **上一任（600 决策预算版）的执行脚本**，发布时保留以便对照「为什么必须 900」。在运行目录里它的文件名是 `exec.sh`（**同 sha `f68533f6…`**）；改名见 §24.4.5 第 4 条 |

#### 24.1.6 v8.16.0 逐格采集摘要（12 条 / 150,358 B）

12 格 = seed ∈ {0,1,2} × **dev 窗口** ∈ {11,12,17,18}（`preflight.frozen.SAMPLE_WINDOWS`；按工单用 dev 窗），**每格 quota 60 状态 / 180 行**（60 × 3 视界），合计 **720 状态 / 2,160 行**；每格记 `status=complete`、`death_episodes`、`no_death_episodes`、`replay_checks`（pass-2 回放逐决策核对）、`nesting`（`all_match=true`）、`gate`、`cfg` 四字段全 0、以及该格标签文件的 `labels_sha256` / `labels_bytes`（本轮已逐格实测 12/12 相符）。

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 27 | `.../DEATHADJ/collect/seed0_w11.json` | 11,975 | `ba52f64e763752cff702f0ac7224519694e2caf1cc399dfa96916b6db4dc6a69` | seed0 / w11：60 状态、3 条死亡（+0 活满丢弃）、`replay_checks=2,475`、wall **285.5 s**；格内标签 `749d5783…` / 1,986,538 B |
| 28 | `.../DEATHADJ/collect/seed0_w12.json` | 13,275 | `59771f9b497d0753f708f5d6cddc8eb1335bdbf094192ad0929147a3da5a18d9` | seed0 / w12：60 / 3 死（+**2** 活满）、2,161、**328.8 s**；标签 `03b490d0…` / 1,988,141 B |
| 29 | `.../DEATHADJ/collect/seed0_w17.json` | 11,982 | `6b7e0f1a3882660365382f0d16357d7373b50c725ae163d7b75dfc150467b134` | seed0 / w17：60 / 3 死（+0）、2,435、**258.1 s**；标签 `c57a40ea…` / 1,980,476 B |
| 30 | `.../DEATHADJ/collect/seed0_w18.json` | 12,620 | `01e38ece786581288777d611ac09f254ff8cf1dca203176aba46aacf1d0e2452` | seed0 / w18：60 / 3 死（+**1**）、2,541、**252.2 s**；标签 `8757deaf…` / 1,981,163 B |
| 31 | `.../DEATHADJ/collect/seed1_w11.json` | 14,572 | `261853a3cf334c652bdb52d4fb74a48d2f54216f043ab34777412318917136a9` | seed1 / w11：60 / 3 死（+**4** 活满）、2,356、**334.3 s**；标签 `12b6e2ac…` / 1,992,917 B |
| 32 | `.../DEATHADJ/collect/seed1_w12.json` | 11,981 | `7944eaf7252b3087fc7d2fd19780ce9992ee64428156e0f64be4c77ee00dde2c` | seed1 / w12：60 / 3 死（+0）、2,361、**270.2 s**；标签 `1743cab4…` / 1,982,805 B |
| 33 | `.../DEATHADJ/collect/seed1_w17.json` | 13,923 | `d73af7094dbda5e6ba8f62dfa95220ff2f20ca0f6c9f98a05d5859d5b197bd8c` | seed1 / w17：60 / 3 死（+**3**）、1,977、**293.9 s**；标签 `d2724488…` / 1,984,538 B |
| 34 | `.../DEATHADJ/collect/seed1_w18.json` | 11,980 | `9277a7934a3abd397be51d041611804a3ef256adb40ebd553f5781eef222054e` | seed1 / w18：60 / 3 死（+0）、2,482、**244.2 s**；标签 `4095fc06…` / 1,983,227 B |
| 35 | `.../DEATHADJ/collect/seed2_w11.json` | 11,982 | `0f5cd928caea96d8f49962f541e69adb48012b03a88859c93a44d1a99d3c875e` | seed2 / w11：60 / 3 死（+0）、1,814、**270.7 s**；标签 `5b385f4b…` / 1,967,264 B |
| 36 | `.../DEATHADJ/collect/seed2_w12.json` | 12,023 | `9de2010896f3877a8ed110141ecda7f0a5653f83ee7f333404d2cf049bf279d4` | seed2 / w12：60 / 3 死（+0）、1,898、**225.9 s**；标签 `ac57e4db…` / 1,966,698 B |
| 37 | `.../DEATHADJ/collect/seed2_w17.json` | 12,018 | `f833818b1888bea376f8b9ade9151b55d7de39aa38140144369e147383a66dc2` | seed2 / w17：60 / 3 死（+0）、1,609、**239.0 s**；标签 `c3470afb…` / 1,964,812 B |
| 38 | `.../DEATHADJ/collect/seed2_w18.json` | 12,027 | `8ee73a994d047a491f77a90f8d25329304815282689431cbaa4da63a3e356c67` | seed2 / w18：60 / 3 死（+0）、1,752、**245.1 s**；标签 `463e03b6…` / 1,966,904 B |

**逐格闭合**：`states` 求和 = **720**、`label_rows` 求和 = **2,160**、`death_episodes` 求和 = **36**、`no_death_episodes` 求和 = **10**、`replay_checks` 求和 = **25,861**，与 `deathadj_stats.json` 的 `states=720` / `label_rows=2,160` / `death_episodes=36` / `no_death_episodes=10` **逐项相同**；12 格 `labels_bytes` 求和 = **23,745,483** = 入册 `labels_deathadj.jsonl` 的字节数（字节闭包见 §24.4.3）。单格 wall **225.9–334.3 s**，12 路并行。

#### 24.1.7 执行器（1 条 / 158,366 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 39 | `tools/rl/v8g0_deathadj.py` | 158,366 | `5b52c96ad3f19f17b262383669b5addb752a18ebd22e39b6110c41efcb2dff0b` | v8.16.0 采集 / 打标 / 统计 / 发布执行器（REDO 版：900 决策 + 三视界 + 连续 10 条不死跳过 + 逐 H P0；`--mode preflight|assert-ckpt|fidelity|recount-v811|smoke|collect|stats|audio-audit|process-audit|manifest|publish`）。盘上现值 = 运行目录 `publish.json` 的 `v8g0_deathadj_sha256`（`5b52c96a…`）；与 `preflight`/`ckpt_assertions`（`5ff6c9fd…`）、`ckpt_assertions_final`（`74dc5a26…`）**都不同** ⇒ 执行器在轮内换过三次状态（只报告，见 §24.4.5 第 3 条） |

### 24.2 跳过说明（未入册 **0** 个文件）

实测：v8.16.0 目录 **38** 个文件 / **24,279,173 B**，执行器 **1** 个 / **158,366 B** ⇒ **39 个文件 / 24,437,539 B，入册 39、未入册 0**。

| 组 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `Document/Update/Alpha/v8.16.0/**` | 38 | 24,279,173 | **全部入册**（含 `manifest.json` 自身、`work_manifest.json`、12 份逐格摘要、7 份离线复算脚本/产物、**合成自测件** `selftest_stats_synthetic.json`）；未入册 **0** |
| `tools/rl/v8g0_deathadj.py` | 1 | 158,366 | **入册**；未入册 **0** |
| **小计** | **39** | **24,437,539** | — |

**为什么全量入册（与 §20–§22 的「最小集」不同）**：本轮 `collect/` 12 份摘要是每格唯一的状态 / 死亡 / 回放 / 门读数（不是可重跑重建的日志）；`p0_extras.json` / `p0_sensitivity.json` 是两个**独立于主 stats 的复算出口**；`selftest_stats_synthetic.json` 虽为**合成**产物，但必须与 `p0_extras.json` 之类同处可辨识位置，否则日后会被误当实验读数。**本轮最小集 = 全集。**

**运行目录 `.tmp_v8g0_deathadj/`（不在 `Document/` 下，不是发布件）**：实测 **663** 个文件 / **53,244,011 B**（`data/` 408、`logs/` 66、`runs/` 48、`work/` 42、`collect/` 12、`labels/` 12、`smoke_work/` 20、`smoke2_work/` 23、`selftest/` 3、`legacy/` 1，其余为顶层 28 件）。其中：

- **标签字节身份已完全覆盖**：`labels/` 12 份逐格标签（合计 **23,745,483 B**）按文件名顺序 `cat` 的 sha256 = **`2bffac3be71f8166…`**，与入册 `labels_deathadj.jsonl` **逐字节相同**（§24.4.3）；
- **26 件已被入册件覆盖**：顶层 22 件同名同 sha，`exec.sh` 以 `exec_previous_600decisions.sh` 的名字入册（同 sha），已发布的 `commands.tsv` 是运行副本的**严格字节前缀**；
- **未覆盖的 6 件（46,239 B）按 §21.2 / §22.2 / §23.2 同口径不入册**：`commands.tsv`（运行副本，13,916 B，比发布件多 1 行 `publish2`）、`publish.json`（发布记录，11,411 B，36 件 + 执行器 sha，**其内容已在 §24.4.2 交叉引用**）、`new_mode_stats.py`（17,572 B）、`probe_evalcost.py`（1,549 B）、`show.py`（1,170 B）、`splice_stats.py`（621 B）—— 均为过程/试验件，**没有**进入 `Document/` 发布目录。

**`__pycache__` / `*.pyc` 实况（按要求只报告、不删）**：`v8.16.0/` 全树 **没有** `__pycache__/`、**没有** `*.pyc`（实测 0）。若日后出现，按 §21.2 口径**一律不入册**。

**`Document/Update/Alpha/v8.17.0/` 本轮不登记**：盘上现有 **1** 个文件（`PREREGISTRATION_CONSERVE_001.md`，4,431 B，mtime `14:18:33`），该轮**仍在采集中**，按工单**下一轮单独登记**。本节**没有**触碰该目录（`{size, mtime}` 前后一致）。

### 24.3 与 §1–§23 不重复的说明（实测已核）

- **既有 466 行 0 命中**：追加前对 `ASSET_SYNC.md` 全文做 `v8\.16\.0|DEATHADJ|v8g0_` 匹配 → **0 命中**；§23 登记的是 v8.11.0（EVENTSAMPLE）与 v8.15.0（CRITICAL），与本轮**不同版本、不同目录、不同执行器**。
- **路径 39 唯一 / 哈希 39 唯一**：39 条路径与 39 个 sha256 逐条比对 **0 组重复**（`ckpt_assertions.json` 与 `ckpt_assertions_final.json` 字节数相同 4,013 B，但 sha **不同**：`6ad51185…` vs `a428abfa…`；`audio_audit.json` / `_smoke` / `_fidelity` 三份互不相同）。
- **与既有 466 行 0 交集**：39 个路径、39 个 sha256 与既有 **466 个路径 / 451 个唯一哈希**逐一比对，**路径交集 0、哈希交集 0** ⇒ 没有把任何既有行「换个路径再钉一遍」。
- **行数与格式**：466 + 39 = **505 行**（`ASSET_SYNC.sha256` 实测 **63,566 B**）；仍是 **LF / 无 BOM**、末行以 `\n` 结束、每行 `64 hex + 两个空格 + 相对路径`，**空行 0 条、格式异常行 0 条**。
- **`ASSET_SYNC.md` 本身不在校验表内**（`ASSET_SYNC` 在 `ASSET_SYNC.sha256` 里 **0 行**）⇒ 追加 §24 **不会**让任何既有行转 FAILED（与 §21.3 / §22.3 / §23.3 同一口径）。
- **Git 跟踪状态（实测）**：`git status --porcelain` 对两路径为 `??`（`Document/Update/Alpha/v8.16.0/`、`tools/rl/v8g0_deathadj.py`）⇒ 这些证据**无法由 Git 重建**，本节 + `ASSET_SYNC.sha256` 是它们目前的字节身份记录。本轮按约束**未** `git add` / `git commit`，分支仍为 `v400`。

### 24.4 校验结论

#### 24.4.1 `sha256sum -c`（WSL，追加前后各一次）

命令（与 §22.4.1 / §23.4.1 同一口径）：

```bash
cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -c ': OK'; sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -v ': OK'
```

* **基线（追加前 466 行）**：**465 OK / 1 FAILED**。
* **追加后（505 行）**：**504 OK / 1 FAILED**。`504 + 1 = 505` 校验行全部参与。
* **新增 39 条 39/39 OK** ⇒ 整体 FAILED 数**没有新增（仍为 1）**。
* **唯一 FAILED 仍是已知并允许的那一条**：`Document/Update/Alpha/v4.2.0/journal.jsonl`（§9.8 记录的活动文件漂移）。本轮**未改该行、未重钉、未触碰该文件**，本节的三个校验时刻（追加 `sha256` 前 / 追加后 / 写完 §24 后）该行是**同一条 FAILED**。
* **格式**：505 行仍为 **LF / 无 BOM**、末行 `\n` 结束、每行 `64 hex + 两个空格 + 相对路径`；空行 **0**、格式异常 **0**；新增 39 条的路径在表内**各自唯一**。

#### 24.4.2 与产出方 `manifest.json` 的交叉核对（登记时实测）

| 清单 | 自述 | 实测比对 |
| --- | --- | --- |
| v8.16.0 `evidence/DEATHADJ/manifest.json` | `file_count=36`、`total_bytes=24,269,969`、`excludes="manifest.json"` | `files{}` **36/36** 的 `{bytes, sha256}` 全等、**0 处不符**；`bytes` 求和 = `total_bytes` = 24,269,969；盘上 `evidence/DEATHADJ/**` 共 **37** 件，manifest **未列**的**只有 `manifest.json` 自身**（5,248 B）⇒ **无缺件、无多余件** |
| 运行目录 `.tmp_v8g0_deathadj/publish.json`（**不属入册范围**，仅作第二锚） | `file_count=36`、36 条 `{source, target, sha256, bytes}`、`v8g0_deathadj_sha256=5b52c96a…` | 36 条的 `bytes` 求和 = **24,269,969** = manifest `total_bytes`；逐条 `sha256` 与入册件实测**全部相同**（36/36）；`v8g0_deathadj_sha256` = **盘上执行器现值** `5b52c96a…`（`{bytes:158,366}` 与实测一致） |
| 12 份 `collect/*.json` 的 `labels_sha256` / `labels_bytes` | 每格标签文件身份 | **12/12** 与 `.tmp_v8g0_deathadj/labels/` 现场重算全等、0 处不符；`labels_bytes` 求和 = **23,745,483** = 入册 `labels_deathadj.jsonl` 字节数 |
| 12 份 `collect/*.json` 的 `states` / `label_rows` / `replay_checks` | 每格读数 | `states` 求和 720、`label_rows` 求和 2,160、`replay_checks` 求和 **25,861** —— 与 `deathadj_stats.json` 的同名总项**逐项相同** |

**一份 manifest 未覆盖的内容（如实列出）**：`manifest.json` 与 `work_manifest.json` **都不含** `tools/rl/v8g0_deathadj.py` 的登记（后者对整个运行目录做 `v8g0_deathadj.py` 匹配 = **0 命中**）。执行器的字节身份由**本节第 39 条**（实测 `5b52c96a…` / 158,366 B）+ 运行目录 `publish.json.v8g0_deathadj_sha256`（同值）+ `ckpt_assertions{,_final}.json.v8g0_deathadj_py`（两相互不相同，见 §24.4.5 第 3 条）三处提供。

#### 24.4.3 身份链、门与读数（只登记，不重判）

| 核对项 | 实测 |
| --- | --- |
| 引擎身份 | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` / match=true，出现在 `preflight.json`、`fidelity_d3.json`、`ckpt_assertions{,_final}.json` 与 12 份 `collect/*.json`；与 §2 / §22 / §23 记录的 `rl_v3` SHA **一致** |
| 行为策略身份 | v4.5.0 `round2.pt` = `4abee069…` / `a93746bb…` / `c69960fc…`，在 `ckpt_assertions{,_final}.json` **两相各 3/3 match** ⇒ 采集期间行为策略未换；12 份 `collect/*.json` 的 `behaviour.ckpt_sha256` 与所属 seed 一致 |
| **预注册身份** | `a820b606…` / 3,956 B，在 `preflight.pins.prereg`、`ckpt_assertions{,_final}.json` **三处同值** ⇒ 轮内**一字未改** |
| **标签锚** | `labels_deathadj.jsonl` = `2bffac3be71f8166f72afee0d47588b0ed03d30cb9226f5e56b6320f08bb252d` / 23,745,483 B / **2,160 行**：`deathadj_stats.labels_file` 记同一 `{path, sha256, bytes, rows}`；运行目录同名件同 sha |
| **逐格 → 合并的字节闭包** | 12 份 `.tmp_v8g0_deathadj/labels/seed*_w*.jsonl` 按文件名顺序 `cat` 后 `sha256sum` = **`2bffac3b…`**，与入册 `labels_deathadj.jsonl` **逐字节相同**（本次现场重算）⇒ 合并件不是重排 / 重写产物 |
| **跨轮负基线可引用** | `recount_v811.json` 用本轮**自己的归约代码**重算 v8.15.0 的 `critical = 0 / 10,008`，与本轮 `critical = 59/63/597` 形成「同一实现下不同采样位置」的对照 |
| **H 嵌套** | `deathadj_stats.nesting_checks={states:720, mismatches:0, alive_is_truncation:true}`；冒烟 1 状态 + **12/12 格首状态**直接跑 H=120/240 对照（`collect/*.json` 的 `nesting.all_match=true`）⇒ `alive(H) = min(alive(600), H)` 在样本上成立 |
| **P0 主读数** | H=120 → critical **59 / 720 = 8.19 %**；H=240 → **63 / 720 = 8.75 %**；H=600 → **597 / 720 = 82.92 %**；三个 H 的「有差异」**都是 0**（`delta` 分布 `n=720, mean=0, stdev=0.0`、`bomb1_equal=720`）⇒ `S-nodiff` |
| **为什么恰好 0（机制）** | `bombs_remaining={0:720}`、`rule_bomb={0:720}`、`bombs_at_death={0:36}`、`bomb1_press_honoured=0`、`bomb1_noop_due_to_invuln=12,960`（= 720 × 18）⇒ 0 颗雷时 `cand_to_mask(cand,1)` 与 `cand_to_mask(cand,0)` 是同一动作，奇偶两半逐位相同 |
| D3 保真门 | `fidelity_d3.json` **5/5 PASS**（HELLO `role=shadow`、DEBUG `threads=1`、前缀 **2,165 / 2,165** checkpoint **0 mismatch**（reset hash `83d95375…`）、R24 **18/18** 常量、PARK `frame=8663`），`verdict="PASS: §2.1-§2.5 hold on this binary and data dir"` |
| 音频（§9 硬约束） | 三处审计 **合计 27 个 cfg / 27 muted / 0 audible / 0 errors**（主 run **24/24**、冒烟 **1/1**、D3 **2/2**），`all_muted=true`；`preflight.launch_env` 记 `SDL_AUDIODRIVER=dummy`、`DISPLAY`/`WAYLAND_DISPLAY` 为 null；共享审计件 `v4.3.1/evidence/R14/audio_cfg_audit.json` **未触碰** ⇒ 本轮**未新增 `audible`** |
| 进程残留 | `process_residue.json` **`matching=0`**、`processes_total_by_engine_name=0`、`pgrep_returncode=1`、`all_pass=true`；本轮**未 `pkill`** 任何别轮进程 |
| 结论口径 | `deathadj_stats.verdict={name:"S-nodiff", diff_gate:0.2, horizons_reaching_gate:[], p0_all_pass:false}` 与 `RUN.md` §0/§5 的 **`S-nodiff`** 一致 ⇒ 按预注册 §2.1 **只发布统计，不训练**；`train_A_*` / `train_B_*` / `eval_*` 产物**本就不存在**（不是登记遗漏） |
| 敏感性（只登记） | `p0_sensitivity.json`：剔除 36 个死亡决策状态后 **684** 状态 → critical **3.36 % / 3.95 % / 82.02 %**、「有差异」仍 **0** ⇒ 结论不依赖退化状态 |

#### 24.4.4 登记期零漂移（v8.16.0 目录与执行器只读）

登记**开始前**（`14:37–14:39`）与**结束后**各采集一次这 **39** 个文件的 `{size, st_mtime_ns}` 并逐项比对（快照保存在登记机的临时目录，不落仓库；本节写入前**先逐条实测 sha256 并与表格比对相同后才动手**）：

| 范围 | 文件数 | 差值 |
| --- | ---: | --- |
| `Document/Update/Alpha/v8.16.0/**` | 38 | `size` 0 变化 / `mtime_ns` 0 变化 |
| `tools/rl/v8g0_deathadj.py` | 1 | `size` 0 变化 / `mtime_ns` 0 变化 |
| **合计** | **39** | **39/39 完全相同，0 个变化** |

**是否需要「等 120 秒再钉」**：不需要。登记开始时目录内**最新** mtime = `manifest.json` 的 **`14:30:27`**，距快照时刻已 **> 7 分钟**（远超 5 分钟窗口），且 39 个文件的 `{size, mtime_ns}` 在两次快照间**没有任何变化** ⇒ 已静态，直接钉。

本轮对上述路径的写操作 = **0**（只读 `Get-ChildItem` / `Get-Process` / `SHA256` / `Get-Content` / `ConvertFrom-Json` / `Select-String` / `git status`）；**未新建、未改名、未删除、未移动**任何文件；写入**只落在** `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256` 两个清单文件上。同时确认 `Document/Update/Alpha/v8.17.0/` 内 1 个文件的 `{size, mtime}` **前后一致**（本轮未登记、未触碰该目录）。

#### 24.4.5 只报告、不裁定的偏差清单

1. **`commands.tsv` 三个长度并存（工单要求以 `manifest.json` 为准）**：`RUN.md` §7 记 **12,853 B / `5d20d5ed8f66d78f…`**；**发布件（盘上）+ `manifest.json` 记 13,753 B / `f5c1ff23b658ae70cc…`**；**运行目录**里的同名件 **13,916 B / `c2c98f76…`（47 行）**。已发布的 13,753 B 是运行副本的**严格字节前缀**，差 **163 B = 末行 `publish2`**（`14:30:26 → 14:30:28`，发布动作自身 append）。故本节按盘上 / `manifest.json` 的 13,753 B / `f5c1ff23…` 登记；`RUN.md` 与运行副本的差异**只记录、未改任何文件**。
2. **`manifest.json` 自述「除自身」**：`file_count=36` / `total_bytes=24,269,969` 与**目录口径 37 / 24,275,217** 的差 **5,248 B 恰为 `manifest.json` 自身** ⇒ 不是缺件，也不是多余件（与 §23.4.2 同一口径）。
3. **执行器在轮内换过三次状态**：`preflight.json`（`14:13:52`）、12 份 `collect/*.json`、`ckpt_assertions.json`（`phase=start`，`14:13:55`）都记 **`5ff6c9fd8103d064…` / 157,602 B**；`ckpt_assertions_final.json`（`phase=final`，`14:27:45`）记 **`74dc5a263961b25b…`**；**盘上现值 `5b52c96ad3f19f17…` / 158,366 B（mtime `14:29:00`）= 运行目录 `publish.json` 的 `v8g0_deathadj_sha256`**。即 **final 断言之后执行器又被编辑了一次**，因此「final 断言所记的执行器」≠「盘上发布所用执行器」。本节按盘上值登记，**只记录事实、不裁定影响**；`74dc5a26…` 这一状态在盘上**已不存在**（无法重建），这是本轮唯一无法用字节复核的环节。
4. **一次发布时改名**：运行目录 `exec.sh`（7,313 B / `f68533f622e834b1…`）在发布时改名为 `exec_previous_600decisions.sh`，**字节完全相同**（同 sha）⇒ 只是文件名变化，内容未改；`publish.json` 的 `source` / `target` 两条路径可对照。
5. **`work_manifest.json` 覆盖运行目录但不含执行器**（对 `v8g0_deathadj.py` 0 命中）：执行器身份靠第 39 条 + `publish.json` + 两份 `ckpt_assertions` 提供（后两者互相矛盾，见第 3 条）。
6. **采集跑了两遍，第一遍被第二遍覆盖**：第一遍 12 格 × quota 50 = **600 状态**，每格第 3 条 episode 的 20 决策窗口被 quota 截成**前 10 个**；第二遍 12 格 × 60 = **720 状态**、**36 条全部是完整 20 决策窗口**，并以第二遍为准（`RUN.md` §9 偏差 1）。第一遍只留下 `deathadj_stats_pass1_quota50.json`；其标签（`d33e5ee9…`）**盘上已不存在**。`manifest.json` 记的就是最终发布件 ⇒ **不是 manifest 不符**。
7. **采集窗 = dev 窗（11/12/17/18）**：按工单指定，与预注册的**评测窗是同一批 dev 窗**（`RUN.md` §9 偏差 2 / §10）。本轮**未训练**，所以没有实际泄漏的模型结果；但若日后用这批标签训练并在同样 4 个 dev 窗评测，必须写明重叠。本节**只登记事实**。
8. **预注册 §4 未列的额外产出**：`selftest_stats.py` + `selftest_stats_synthetic.json`、`p0_extras.{py,json}`、`p0_sensitivity.{py,json}`、`cell_progress.py`、`exec2.sh`、`exec_previous_600decisions.sh`（`RUN.md` §9 偏差 4）—— **只增不改**；其中 `selftest_stats_synthetic.json` 是**合成输入的代码自测输出**，**不得**当实验读数引用。本节照登以给它字节身份。

> **一句话结论**：本轮把 **v8.16.0（DEATHADJ）38 条 / 24,279,173 B + 执行器 1 条 / 158,366 B = 39 条 / 24,437,539 B** 成对写入 `ASSET_SYNC.md` §24 与 `ASSET_SYNC.sha256`，`sha256sum -c` 由 **465 OK / 1 FAILED（466 行）** 变为 **504 OK / 1 FAILED（505 行）**（新增 39 条全 OK，整体 FAILED 数仍为 1，唯一 FAILED 仍为已知允许的 `v4.2.0/journal.jsonl`）。两个口径都记：`evidence/DEATHADJ` **37 / 24,275,217 B**（目录）vs **36 / 24,269,969 B**（manifest 口径，`excludes="manifest.json"`）；manifest 逐条比对 **36/36 全等、0 处不符**；12 份逐格标签 `cat` 后与入册 `labels_deathadj.jsonl` **逐字节相同**（`2bffac3b…`）。本轮**未启动引擎、未训练、未评测、未 `git add/commit`**，39 个受保护文件的 `{size, mtime_ns}` **逐个零漂移**；`v8.17.0/` 按要求**未登记、未触碰**（下一轮单独登记）。

## 25. v8.19.0 与 v8.20.0 资产（2026-11-22）

> **为什么有这一节**：AGENTS.md §8.2 要求「新增/移动/删除任何资产，必须同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`」。**v8.19.0（RULESWEEP30）与 v8.20.0（UNIFIED）两轮已发布，但追加本节前一行都没有入册** —— 对既有 **505 行**校验表与 **2165 行**本册正文分别做 `v8\.19\.0|v8\.20\.0|v8k0_|v8l0_|RULESWEEP30|UNIFIED` 匹配，两份都是 **0 命中**。本节与 `ASSET_SYNC.sha256` 末尾追加的 **121 行**成对登记。

> **本轮性质：两轮都是 live engine 真评测轮，零训练、零采集**（两份 `summary.json` 的 `notes[0]` 都是 `"No training and no collection: this file only evaluates."`；`invalid_rows=0`）。两轮是**同一条研究线上的相邻两步**，单因子各自只动一个量：
>
> * **v8.19.0 = 单因子「放雷阈值组」**：候选头固定 v8.6.0 `bombdag.pt`（`Policy(106,64,36)`），只改 `THRESHOLDS[3/2/1]`。四臂 **S1 60/40/20、S2 100/80/60、S3 40/30/20、S4 120/90/60**，各 **3 种子 × 4 dev 窗 × 20 条 = 240 条 / 900 决策**；对照 R0 = 已发布 R5 的 **634/3,600 = 17.6111 %，按预注册不重跑**。判 **`S-win`**：S3 = **59/240 = 24.5833 %**（p = 0.0054）、S1 = **58/240 = 24.1667 %**（p = 0.0082）双双过 P1（≥24 %）与 P2（vs R5 单侧 Fisher p < 0.05）。
> * **v8.20.0 = 单因子「决策预算」**：候选头与规则都**固定为 v8.19 的 S1 = 60/40/20**，只改预算。三档 **U-600 / U-900 / U-1200**，各 **3 种子 × 4 窗 × 20 条 = 240 条**。读数 **237/240 = 98.7500 %** / **58/240 = 24.1667 %** / **5/240 = 2.0833 %**；P1（U-600 ≥ 95 %）成立、P2（U-900 复现 v8.19 S1）成立 ⇒ 判 **`S-unified`**（一套阈值覆盖 20 s 与 30 s）；P3 未达预注册的 +3 pp 线（**+1.0833 pp，p = 0.1097**）⇒ 40 s 的瓶颈不是阈值远近。

> **只读边界**：本轮**只 append** 两个清单文件。`Document/Update/Alpha/v8.19.0/**`（**67** 个文件）、`Document/Update/Alpha/v8.20.0/**`（**52** 个）与两个执行器（**2** 个）全程只读；登记前后各采一次快照，这 **121** 个文件的 `{size, mtime_ns}` **121/121 完全相同、0 变化**（§25.4.4）。**未启动引擎**（0 次引擎调用）、**未训练、未评测、未 `git add` / 未 `git commit`**。

> **实测口径**：121 条的 `bytes` 与 `sha256` **全部现场实测**（Python `hashlib` 对全文件流式读入），再用 WSL `sha256sum -c` 独立复算。两份产出方 `manifest.json` 的内联 `{bytes, sha256}` **只用于交叉核对，不代替实测**（§25.4.2：**67−2 条 + 52−2 条全等、0 处不符**）。

> **两个口径都写（本册一贯做法）**：目录口径 = 递归实测总数（含 `manifest.json` 自身与顶层预注册）；`manifest.json` 自述口径 = 该文件 `file_count` / `total_bytes`，**`excludes="manifest.json"`**，且因 `root` 在 `evidence/<TAG>/` 而**不含**顶层预注册。
>
> | 轮次 | 目录口径（含 `manifest.json` 自身与顶层预注册） | `manifest.json` 自述 | 差额闭合 |
> | --- | --- | --- | --- |
> | v8.19.0 `evidence/RULESWEEP30` | **66 / 5,473,409 B** | `file_count=65` / `total_bytes=5,463,957` | 差 **9,452 B = `manifest.json` 自身**；再加顶层预注册 **3,504 B** ⇒ v8.19.0 全目录 **67 / 5,476,913 B** |
> | v8.20.0 `evidence/UNIFIED` | **51 / 3,918,352 B** | `file_count=50` / `total_bytes=3,910,975` | 差 **7,377 B = `manifest.json` 自身**；再加顶层预注册 **2,790 B** ⇒ v8.20.0 全目录 **52 / 3,921,142 B** |
>
> 两行的「差额」都恰为 `manifest.json` 自身字节数（`5,473,409 − 5,463,957 = 9,452`、`3,918,352 − 3,910,975 = 7,377`）⇒ **无缺件、无多余件**（与 §23.4.2 / §24.4.2 同一口径）。

> **全量入册口径：两轮每一个文件都入册，未入册 0 个** —— 含两份 `manifest.json` 自身、两份 `publish.json`、v8.19.0 的 `attempt1/` 失败留档、v8.20.0 的三份 `preflight*.json`。理由：这 **119** 件（67 + 52）**都不是可重跑重建的日志**，而是本轮唯一的评测行数据 / 统计 / 门禁 / 摘要出口。**本轮最小集 = 全集。**

### 25.1 登记范围（**121 条** = v8.19.0 **67 条** + v8.20.0 **52 条** + 执行器 **2 条**）

| 分组 | 条数 | 字节合计 | 位置 |
| --- | ---: | ---: | --- |
| §25.1.1 v8.19.0 冻结预注册 | 1 | 3,504 | `Document/Update/Alpha/v8.19.0/` |
| §25.1.2 v8.19.0 运行记录、命令表、发布记录与清单锚 | 6 | 47,363 | `v8.19.0/evidence/RULESWEEP30/` |
| §25.1.3 v8.19.0 门、断言与审计 | 7 | 47,301 | `v8.19.0/evidence/RULESWEEP30/` |
| §25.1.4 v8.19.0 评测行数据（48 个 jsonl） | 48 | 5,105,509 | `v8.19.0/evidence/RULESWEEP30/` |
| §25.1.5 v8.19.0 `attempt1/` 失败留档 | 4 | 221,941 | `v8.19.0/evidence/RULESWEEP30/attempt1/` |
| §25.1.6 v8.19.0 统计 | 1 | 51,295 | `v8.19.0/evidence/RULESWEEP30/` |
| **v8.19.0 小计** | **67** | **5,476,913** | — |
| §25.1.7 v8.20.0 冻结预注册 | 1 | 2,790 | `Document/Update/Alpha/v8.20.0/` |
| §25.1.8 v8.20.0 运行记录、命令表、三次引擎普查、发布记录与清单锚 | 8 | 47,703 | `v8.20.0/evidence/UNIFIED/` |
| §25.1.9 v8.20.0 门、断言与审计 | 6 | 36,234 | `v8.20.0/evidence/UNIFIED/` |
| §25.1.10 v8.20.0 评测行数据（36 个 jsonl） | 36 | 3,787,935 | `v8.20.0/evidence/UNIFIED/` |
| §25.1.11 v8.20.0 统计 | 1 | 46,480 | `v8.20.0/evidence/UNIFIED/` |
| **v8.20.0 小计** | **52** | **3,921,142** | — |
| §25.1.12 执行器 | 2 | 157,775 | `tools/rl/` |
| **合计** | **121** | **9,555,830** | — |

#### 25.1.1 v8.19.0 冻结预注册（1 条 / 3,504 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.19.0/PREREGISTRATION_RULESWEEP30_001.md` | 3,504 | `39566638ad19ef5c363a0a4ec823b88e905f01be46c9f71d57c8d038fb64e5ff` | v8.19.0 冻结预注册（**FROZEN**，在任何 v8.19.0 评测之前）：单因子 = **放雷阈值组**，候选头固定 v8.6.0 `Policy(106,64,36)`；六臂 **R0 80/60/40（已发布，不重跑）/ S1 60/40/20 / S2 100/80/60 / S3 40/30/20 / S4 120/90/60 / S5（可选，本轮未做）**；P1 = 任一臂 30 s 活满率 **≥ 24 %**，P2 = vs R5 单侧 Fisher **p < 0.05**；§2.1 停止条件 S-win / S-flat / S-nonmono / S-fork。**三处独立记录同 sha**（`ckpt_assertions_{start,final}.json` 两相 + `RUN.md` §1）⇒ 轮内**一字未改** |

#### 25.1.2 v8.19.0 运行记录、命令表、发布记录与清单锚（6 条 / 47,363 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/RUN.md` | 19,187 | `49e070a1e05e1cfc60aa7aca470c0d6d4f1753557eaefc55b036aa9f99cee5cf` | **304 行** / 19,187 B 运行记录：§0 结论摘要（五臂表 + P1/P2/停止条件 **`S-win`**）；§1 身份门 2 相 `all_pass=true`；§2 D3 保真 **5/5 PASS**；§3 音频 **52/52 muted、0 audible**；§4 命令与 exit code（含 S4-s0 那次 **exit 4**）；§5 协议逐项对照；§6 结果（逐臂 / 逐窗口 / 逐种子 / 用雷分布 / 机制）；§7 判定；§8 **8 条偏差**（含整格重跑、105 台他轮引擎、一次打包事故）；§10「不能声称什么」5 条 |
| 2 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/commands.tsv` | 3,400 | `f9e6f1c6b2b061b5066024b1e38693822ae754fe8c68f3a84ec5a6ac99852105` | **20 行**（无表头）的 `label / argv` 两列命令原文：assert-start、d3、smoke、S1–S3 各 3 条、S4-s0、S4-s1/s2、**S4-s0 重跑**（tag `v8k0-eval-S4-d900-s0-r2`）、assert-final、summary、audio-audit、process-audit；与 `RUN.md` §4 的 16 行表展开后**逐条对应（20 条）** |
| 3 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/eval_exits.tsv` | 156 | `3f475bd3654c238b4d4031a474516420ab8a728659904477d019490142f17d12` | **12 行**评测退出码与墙钟：11 条 **exit 0**（424–455 s）+ **`S4-s0` = exit 4 / 235 s**（唯一非 0，已被整格重跑覆盖，见 §25.4.5 第 3 条） |
| 4 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/preflight.txt` | 423 | `e91b3054f874c3e62b80fb121d60f3c4c8684226dc725d92d85d9ba0e6a859af` | 开工前环境快照（423 B）：`nproc=20`、`loadavg 30.14 27.83 21.70`、内存 23 GB、`Python 3.10.12`；**`SDL_AUDIODRIVER: <unset>` 而 `DISPLAY: :0` / `WAYLAND_DISPLAY: wayland-0` 都是「存在」** ⇒ §9.3 的两个 unset 不是形式主义；`engine_processes_total=107`，按轮分组 **`105 tmp_v8i0_nobomb`** + 18 个他轮 python 驱动（本轮 `own=0`，未触碰） |
| 5 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/publish.json` | 14,745 | `97b8c4463713597e1b63b62689dcd9c35ee8998374792f2258b5bf5b7d822da6` | 发布记录：源 → 目标的逐条 `{source, target, sha256, bytes}` + 本轮清单，是「发布件 = 运行目录件」的第二锚（`RUN.md` §8.8 记该文件在打包事故后随 `manifest.json` 一起重建） |
| 6 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/manifest.json` | 9,452 | `cc9bc1217fd2c905357dae15ffb5a2e9f5cd262433a5dd1979ef6c22f115b1c7` | **清单锚**：`schema=v8k0-rulesweep30-manifest/1`、`file_count=65`、`total_bytes=5,463,957`、`excludes="manifest.json"`；`files{}` **65** 条 `{sha256, bytes}` —— 本节实测 **65/65 全等、0 处不符**；`bytes` 求和 = `total_bytes` = 5,463,957；`created_at=15:25:33`、mtime **15:25:33** 是目录内**最后一个写入**的文件（晚于 `publish.json` 15:25:32、`RUN.md` 15:25:27）⇒ **没有**「manifest 是旧快照」的问题 |

#### 25.1.3 v8.19.0 门、断言与审计（7 条 / 47,301 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/audio_audit.json` | 16,291 | `94a7235a88e34170c46d33849448ba0e6f883f3c08710bf07871192deb2d23e1` | AGENTS.md §9.3 主审计（`scope=/mnt/d/WorkShop/Touhou_RL/.tmp_v8k0_rulesweep30/data`）：`cfg_files=52 / decoded=52 / **muted=52** / **audible=0** / errors=0`；逐行 jsonl 都带 `cfg_sha256` 与 `cfg_muted_four_zero=true`；共享审计件 `v4.3.1/evidence/R14/audio_cfg_audit.json` **未触碰** |
| 2 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/ckpt_assertions_start.json` | 4,848 | `aafb04a7ceca22e8cd9122dc5964a9ac133a8fc1dd5a3e347523356352a674a8` | `phase=start`（`15:05:12`）：引擎 `691bfd77…` + `bombdag.pt` 3/3 + `round2.pt` 3/3 + 10 个受保护源文件**全部 match**，`all_pass=true`；`self.sha256=37b220a0…`、`prereg.sha256=39566638…` |
| 3 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/ckpt_assertions_final.json` | 4,848 | `f3c9d9b96a80d4fc4261c698021b6e62930d098ecd254f241c6edbfebc52a75f` | `phase=final`（`15:22:36`）：同上**全部 match**、`all_pass=true`；**`self` 与 `prereg` 的 sha 和 start 相逐位相同** ⇒ 整轮执行器与预注册**都未漂移**（与 §24.4.5 第 3 条那轮形成对比：本轮执行器 `37b220a0…` start = final = 盘上现值） |
| 4 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/d3_gate.json` | 5,140 | `815a8fdec1ad37f9a660a5ccca6ef3a1df61ca043ae5f15fe0c4ed68e5d09857` | D3 保真门 **5/5 PASS**（`all_pass=true`、`passed=5/total=5`、`exit=0`、`wall_s=16.487`），`verdict="PASS: §2.1-§2.5 hold on this binary and data dir"`：HELLO `role=shadow`、DEBUG `threads=1`、前缀 **2,165** checkpoint 到帧 8663 **0 mismatch**、R24 **18/18** 常量、PARK `frame=8663`。**其 `raw` 指向运行目录的 `.tmp_v8k0_rulesweep30/fidelity_d3.json`、`raw_sha256=5e05916f…`** —— 这正是 `RUN.md` §2 引的那个 sha（见 §25.5 第 4 条） |
| 5 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/process_residue.json` | 534 | `adab26561792e3130050a13fc3499e0221cd8adbe5bad2017350683c087b76e8` | 本轮引擎残留 **`matching=0`**、`lines=[]`、`pgrep_returncode=0`、`all_pass=true` ⇒ 本轮**没有**留下引擎进程；采集期间机器上的 105 个他轮（`v8i0_nobomb`）进程**只计数、未 `pkill`**。**与 §24.4 的同名件不同：本件没有 `data_dir_filter` 字段**（见 §25.5 第 8 条） |
| 6 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/smoke.json` | 10,365 | `573d931571e4dae985d93e56c8281844a0825eb706b2dd39b87f7b6e2e8ff021` | 1 条冒烟（S1 / seed0 / w11 / 900 决策）：`bombs_used=3`、`bomb_emits=9`、`bomb_decisions=[0,1,2,3,4,5,6,343,595]`、`distinct_cands=18`、`cfg_muted_four_zero=true`、`gate_all_pass=true`、5 项内部审计全 true；**`survived=false` / `alive_frames=1,584` / `decisions_executed=792`**（这条冒烟在 792 决策处死亡，**不是** 900 活满） |
| 7 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/within_sweep_contrasts.json` | 5,275 | `ea46f6004e7b282c7135ccd9203a4fe9e21db4fe64814ff7bd816f8e36f95992` | **离线复算**（`producer="within_sweep.py (offline)"`，只读已落盘的 960 行、不启引擎）：逐臂 `{n, survived, rate, cp95, triggering_decisions, alive_frames_dead_only, last_bomb_decision, bomb_ignored_total}` + `pairwise` 6 组 + `last_bomb_bands` 4 带 + `group_late_vs_early`（**晚臂组 S1+S3 = 117/480 = 24.38 % vs 早臂组 S2+S4 = 90/480 = 18.75 %，Δ = +5.62 pp，单侧 Fisher p = 0.0206**）+ `r5_published_cell` 的同带复现 ⇒ 「阈值 → 第三颗雷的时机 → 活满率」在两个不同候选头上都成立 |

#### 25.1.4 v8.19.0 评测行数据（48 条 / 5,105,509 B / **960 行**）

48 个 `eval_{S1,S2,S3,S4}_d900_w{11,12,17,18}_seed{0,1,2}.jsonl` = **4 臂 × 4 dev 窗 × 3 种子**，**每个 20 行**（实测 48/48 都是 20 行，合计 **960 行**，`summary.invalid_rows=0`）。逐文件 `bytes` 与 `sha256` 见下表（`sha256` 取前 12 位，**全集 121 条内无前缀碰撞**；完整 64 位在 `ASSET_SYNC.sha256`，与 `manifest.json` **48/48 全等**）。每格 20 条 episode 的 5 项自检（`one_sample_per_decision` / `rule_bomb_matches_emits` / `counter_bookkeeping_closes` / `frames_consistent` / `threshold_respected`）在 **960/960** 条上为 true。

**臂 S1（阈值 60/40/20）—— 12 格 / 1,288,537 B**

| 格（窗 / 种子） | 字节 | SHA-256（前 12 位） |
| --- | ---: | --- |
| w11 / seed0 | 109,517 | `74ae2d77c9ed…` |
| w11 / seed1 | 109,407 | `07b10ab032e6…` |
| w11 / seed2 | 107,547 | `da022ca8f663…` |
| w12 / seed0 | 106,043 | `45c5a7961a6c…` |
| w12 / seed1 | 107,072 | `5db93f097e77…` |
| w12 / seed2 | 106,415 | `8b352478d7aa…` |
| w17 / seed0 | 105,629 | `4b3919e0a159…` |
| w17 / seed1 | 104,457 | `e9b48c781b32…` |
| w17 / seed2 | 103,120 | `91a665c7a7f9…` |
| w18 / seed0 | 110,001 | `2b6b0886c825…` |
| w18 / seed1 | 110,493 | `437221becec1…` |
| w18 / seed2 | 108,836 | `85f3bd6a293c…` |

**臂 S2（阈值 100/80/60）—— 12 格 / 1,266,207 B**

| 格（窗 / 种子） | 字节 | SHA-256（前 12 位） |
| --- | ---: | --- |
| w11 / seed0 | 107,935 | `9d9f1a6c8750…` |
| w11 / seed1 | 106,694 | `05fdc5895545…` |
| w11 / seed2 | 106,112 | `1f8081aa588d…` |
| w12 / seed0 | 106,747 | `a08b3a5a9a38…` |
| w12 / seed1 | 105,393 | `71f141248d6b…` |
| w12 / seed2 | 103,881 | `e3197032a332…` |
| w17 / seed0 | 99,736 | `a9b304c0df41…` |
| w17 / seed1 | 102,946 | `69bcaaa2585b…` |
| w17 / seed2 | 100,108 | `f934ebbe2889…` |
| w18 / seed0 | 109,477 | `c7f7b76b25f4…` |
| w18 / seed1 | 108,655 | `fdd067e437c7…` |
| w18 / seed2 | 108,523 | `0b3c1178ef35…` |

**臂 S3（阈值 40/30/20）—— 12 格 / 1,287,405 B**

| 格（窗 / 种子） | 字节 | SHA-256（前 12 位） |
| --- | ---: | --- |
| w11 / seed0 | 110,790 | `2c2fd26879d6…` |
| w11 / seed1 | 109,746 | `0e791a6582cd…` |
| w11 / seed2 | 110,067 | `f512482389db…` |
| w12 / seed0 | 108,219 | `3e0dcd8716e9…` |
| w12 / seed1 | 107,568 | `59d9c6d0dc00…` |
| w12 / seed2 | 110,642 | `047666589e67…` |
| w17 / seed0 | 102,588 | `3451134045ae…` |
| w17 / seed1 | 100,858 | `d3fade055420…` |
| w17 / seed2 | 100,675 | `a10ca2402630…` |
| w18 / seed0 | 109,128 | `24d5b8e07ab5…` |
| w18 / seed1 | 107,707 | `eefa26db3873…` |
| w18 / seed2 | 109,417 | `c85ede18ea3e…` |

**臂 S4（阈值 120/90/60）—— 12 格 / 1,263,360 B**

| 格（窗 / 种子） | 字节 | SHA-256（前 12 位） |
| --- | ---: | --- |
| w11 / seed0 | 107,058 | `dc801641b80e…` |
| w11 / seed1 | 105,988 | `dbe77852cf69…` |
| w11 / seed2 | 106,703 | `c1c80720651e…` |
| w12 / seed0 | 107,237 | `e48eb9ea2cff…` |
| w12 / seed1 | 106,173 | `1765a986ebfe…` |
| w12 / seed2 | 105,280 | `1b443cbd90ea…` |
| w17 / seed0 | 101,428 | `14b5a069a025…` |
| w17 / seed1 | 100,846 | `f634b5204253…` |
| w17 / seed2 | 98,654 | `48fb734fbb65…` |
| w18 / seed0 | 108,247 | `85a37062f92a…` |
| w18 / seed1 | 107,899 | `1a3a4ac5b81d…` |
| w18 / seed2 | 107,847 | `1156e5ddea5c…` |

#### 25.1.5 v8.19.0 `attempt1/` 失败留档（4 条 / 221,941 B）

**这是失败那次（S4 / seed0 首跑，exit 4）的留档，照登为过程证据；它不参与最终统计。** `RUN.md` §8.2 记：原进程在 w17 起父进程时 `HarnessError: harness closed the control pipe`（引擎启动即关闭控制管道，当时机器上 168 个引擎进程、loadavg ≈ 50），跑到 40/80 条后 **exit 4**；处置是把已完成的 w11/w12 原始文件留档，再用新 tag `v8k0-eval-S4-d900-s0-r2` **整格重跑 4 个窗口**（不续跑，因为续跑会重置 `torch.Generator(seed*1000+7)` 的抽样流）。**发布目录里的 `eval_S4_d900_w*_seed0.jsonl`（§25.1.4）全部来自那次重跑（exit 0）**，与 `attempt1/` 的同名件**字节不同**（`attempt1` 的 w11 = `c8a5c4527000…` / w12 = `a450634a1ff8…`，发布件 w11 = `dc801641b80e…` / w12 = `e48eb9ea2cff…`）⇒ 两批可区分，不会被当成同一份。

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/attempt1/eval-S4-s0.out` | 4,889 | `0050e3acba147d9e35b411ab7ff5eaa24c6e765fbcd5e9c237ce7d2164c38286` | 失败那次的进程输出留档 —— 供复核 `HarnessError: harness closed the control pipe` 的现场 |
| 2 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/attempt1/eval_S4_d900_w11_seed0.jsonl` | 106,998 | `c8a5c45270008b0da1c80fe6b8420c0bb309e69f19d7c2c84d513c832421a802` | 失败那次（attempt1）的 w11 原始行数据 —— **过程证据，不参与最终统计**（§25.2.1） |
| 3 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/attempt1/eval_S4_d900_w12_seed0.jsonl` | 107,175 | `a450634a1ff8794290fad323fba929bd9527ad7f13c8833ff809081a68c1537d` | 失败那次（attempt1）的 w12 原始行数据 —— **过程证据，不参与最终统计**（§25.2.1） |
| 4 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/attempt1/S4_d900_s0_progress.json` | 2,879 | `dafd738916f8935df3f6a8ca08a754c466311f6579aafbb1a703f8e7cf626908` | 失败那次的进度快照（跑到 40/80 条后 exit 4）—— 与 `.out` 互为旁证 |

#### 25.1.6 v8.19.0 统计（1 条 / 51,295 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/summary.json` | 51,295 | `98d66fd3f79b60cdb4ee35f2fb1d3510c167af4cb45bfe321b560d1784d761ae` | **P1/P2 主判据**：`verdict.stop_condition="S-win"`、`best_arm="S3"`、`best_rate=0.245833`、`invalid_rows=0`；逐臂 `{thresholds, files{12 格各自的 rows/full/sha256/bytes}, pooled, cp95, bomb_usage, per_arm{P1,P2}}`；`S-flat.held=false`、`S-nonmono.monotone=false`（`order_late_to_early = [S3, S1, R0, S2, S4]`，唯一反序在 R0/S2/S4 之间、两两 Fisher 全不显著）；`control` = 已发布 R5 的 **634/3,600 = 17.6111 %**（`rerun=false`，逐窗口 198/143/107/186 求和 = 634 闭合）。**本文件引用的 48 格 `{bytes, sha256}` 与盘上实测 48/48 全等** |

#### 25.1.7 v8.20.0 冻结预注册（1 条 / 2,790 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.20.0/PREREGISTRATION_UNIFIED_001.md` | 2,790 | `199a567ebe26198ef6892a2f2e37201143a119d86c4a66f958f7ca0f4ec44c42` | v8.20.0 冻结预注册（**FROZEN**，在任何 v8.20.0 评测之前）：单因子 = **决策预算**，候选头与规则**都固定**（规则 = v8.19 的 S1 = 60/40/20）；三臂 **U-600 / U-900 / U-1200**，各 3 种子 × 4 窗 × 20 = 240 条；P1 = U-600 **≥ 95 %**、P2 = U-900 对 v8.19 S1（24.2 %）CP95 重叠、P3 = U-1200 与 R5 的 1.00 % 并排（**≥ 3 % 即实质提升**）、P4 = 用雷分布；§2.1 停止条件 S-unified / S-switch / S-mid / S-fork。**两相 `ckpt_assertions` 同 sha**（`199a567e…`）⇒ 轮内**一字未改** |

#### 25.1.8 v8.20.0 运行记录、命令表、三次引擎普查、发布记录与清单锚（8 条 / 47,703 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.20.0/evidence/UNIFIED/RUN.md` | 19,809 | `8627a6528fc684b4221c564c1e1354239e32200a71921e5c7e3f86cad4ed694d` | **294 行** / 19,809 B 运行记录：§0 一句话结论 + P1/P2/P3/P4 判定表（**`S-unified`**）；§1 协议；§2 保真门 5/5 + 身份断言 + 冒烟原文；§3 三档 × 逐窗口 / 逐种子 / 12 格原始计数 + 已发布对照并排；§4 P1/P2/P3 与停止条件；§5 用雷分布；§6 19 条命令与 exit code；§7 **8 条偏差**；§8「不能声称什么」5 条；§9 落盘清单 |
| 2 | `Document/Update/Alpha/v8.20.0/evidence/UNIFIED/commands.tsv` | 2,202 | `13301e5e2ccb3cde28b49ebdc58270c1f9ee3199b4fabf18f7d924ff334b5c37` | **9 行**（无表头）的 `label / argv` 两列：**只记 9 条评测命令原文**（3 档 × 3 种子），与 `RUN.md` §9「`commands.tsv` 9 条评测命令原文」自述一致；其余 10 条（preflight / assert / d3 / smoke / summary / audit / publish）记在 `RUN.md` §6 的表里，**不在本文件内** |
| 3 | `Document/Update/Alpha/v8.20.0/evidence/UNIFIED/eval_exits.tsv` | 465 | `0358eea5567cf1793ac8cc4fbf230fd90438c936806b3fcc3ab7d5fae5652945` | **9 行**四列 `label / exit / finished_at / wall_s`：**9/9 全部 exit 0**（198–293 s），结束时刻 **15:26:41 → 15:28:16**（`RUN.md` §6 记批次墙钟 4 min 53 s） |
| 4 | `Document/Update/Alpha/v8.20.0/evidence/UNIFIED/preflight.json` | 1,267 | `8ad5aea4fca287c6c509664595323cddaf9e2c30159a949486a384e6daf866c0` | 开工前**第一次** `--mode preflight`（`15:19:48`）：`census.engine_processes_total=**153**`、`other_round_processes=153`、**`matching=0`**、`lines_mine=[]`（全是别轮 `v8i0_nobomb`） |
| 5 | `Document/Update/Alpha/v8.20.0/evidence/UNIFIED/preflight_start.json` | 2,203 | `d124255a8c5ba10da4bde424e90e489df6fe6597acb1763fd4d2fbf6195cb268` | `15:23:20`：按「> 6 先等 5 分钟」等满后 `engine_processes_total=**90**`、`waited_s=300`；`history` 记 99 → 90 的两次普查 ⇒ **等待是真的发生了**，且等待期间只跑 d3 与 assert-ckpt（未启评测） |
| 6 | `Document/Update/Alpha/v8.20.0/evidence/UNIFIED/preflight_final.json` | 2,205 | `1ad6620b945088745c631e8d401be81ee771fba7f9d79aee9643d805f27957ba` | 结束普查（`15:31:53`）：`engine_processes_total=**161**`、`matching=0`、`lines_mine=[]` ⇒ 本轮进程已全部退出，161 台**全是别轮**、未触碰 |
| 7 | `Document/Update/Alpha/v8.20.0/evidence/UNIFIED/publish.json` | 12,175 | `c9ca741cd44ff25d99a2046f8c618e5e6553059f16b548d266dac7849b1e2b04` | 发布记录：逐条 `{source, target, sha256, bytes}` + 本轮清单，是「发布件 = 运行目录件」的第二锚 |
| 8 | `Document/Update/Alpha/v8.20.0/evidence/UNIFIED/manifest.json` | 7,377 | `81b479dea48920df1812060076ebb612ff91cb2f7742557285e74fdf2a3d5f4d` | **清单锚**：`schema=v8l0-unified-manifest/1`、`file_count=50`、`total_bytes=3,910,975`、`excludes="manifest.json"`；`files{}` **50** 条 `{sha256, bytes}` —— 本节实测 **50/50 全等、0 处不符**；`bytes` 求和 = `total_bytes` = 3,910,975；`created_at` / mtime 是目录内**最后一个写入**的文件⇒ 没有「旧快照」问题（**该 manifest 在 `15:38:38` 被重新发布过一次，见 §25.5 第 10 条**） |

#### 25.1.9 v8.20.0 门、断言与审计（6 条 / 36,234 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.20.0/evidence/UNIFIED/audio_audit.json` | 11,265 | `97c5a57d839ddb509489a957f0b80bcc89a9aaf0308ae6d7358d579caa32890b` | AGENTS.md §9.3 主审计（`scope=/mnt/d/WorkShop/Touhou_RL/.tmp_v8l0_unified/data`）：`cfg_files=37 / decoded=37 / **muted=37** / **audible=0** / errors=0`（36 个 (tag, window) 目录 + 1 个 fidelity 目录）；`RUN.md` §6 第 15 条同数；共享审计件 `v4.3.1/evidence/R14/audio_cfg_audit.json` **未触碰** |
| 2 | `Document/Update/Alpha/v8.20.0/evidence/UNIFIED/ckpt_assertions_start.json` | 4,999 | `a1e5892fb7824322930506a9d08c5d7d7fa956e154d3f26177dfe000366c1f90` | `phase=start`：引擎 `691bfd77…` + `bombdag.pt` 3/3 + `round2.pt` 3/3 + 10 个受保护源文件**全部 match**、`all_pass=true`；`rule={S1, 60/40/20}`、`budgets=[600,900,1200]`、`prereg.sha256=199a567e…`；**`self.sha256=c6f4c4b0…`（见 §25.5 第 3 条）** |
| 3 | `Document/Update/Alpha/v8.20.0/evidence/UNIFIED/ckpt_assertions_final.json` | 4,999 | `6a6a6fd1e20e7404ef6d687efb22d0601a208f94b5a003ce3c9138a1e0fcff1f` | `phase=final`：同上**全部 match**、`all_pass=true`；`prereg` / `rule` / `budgets` 与 start 相**相同** ⇒ 预注册与被评测 ckpt 未漂移；`self.sha256=**ceae240c…**` = **盘上执行器现值**（收尾版） |
| 4 | `Document/Update/Alpha/v8.20.0/evidence/UNIFIED/d3_gate.json` | 5,100 | `48da324bbbad6904031d4f10da02925fb35e5c5250ec89a373dbba67bb01e7db` | D3 保真门 **5/5 PASS**（`all_pass=true`、`passed=5/total=5`、`exit=0`、`wall_s=15.445`），同一 `verdict` 字符串；HELLO `role=shadow` / DEBUG `threads=1` / 前缀 **2,165** checkpoint **0 mismatch** / R24 **18/18** / PARK `8663`。**非 5/5 即整轮作废 —— 本轮未触发 `S-fork`**。其 `raw_sha256=ac35fcf8…` 指运行目录的 `fidelity_d3.json` |
| 5 | `Document/Update/Alpha/v8.20.0/evidence/UNIFIED/process_residue.json` | 522 | `88f4a762a02e29f97ae7b0e919d0f1e2eb53793584a3d7226f2487cf4743c323` | 本轮引擎残留 **`matching=0`**、`lines=[]`、`pgrep_returncode=0`、`all_pass=true` ⇒ 本轮**没有**留下引擎进程，也**没有** `pkill` 任何别轮进程（与三份 preflight 的 `matching=0` / `lines_mine=[]` 一致） |
| 6 | `Document/Update/Alpha/v8.20.0/evidence/UNIFIED/smoke.json` | 9,349 | `e8c80bb5c7c653d17b5a181ef7df01a0cb9a7cf6babeb2d2d406ef6861fc5e56` | 1 条冒烟（U-600 / seed0 / w11 / 600 决策）：**`survived=true`**、`bombs_used=3`、`alive_frames=1,200`、`frames=1,200`、`decisions_executed=600`、`bomb_emits=9`、`bomb_decisions=[0,1,2,3,4,5,6,343,595]`、`distinct_cands=18`、`cfg_muted_four_zero=true`、`gate_all_pass=true`、5 项内部审计全 true、`wall_s=6.789` —— **与 `RUN.md` §2.3 引用的原文逐项相同** |

#### 25.1.10 v8.20.0 评测行数据（36 条 / 3,787,935 B / **720 行**）

36 个 `eval_U{600,900,1200}_d{600,900,1200}_w{11,12,17,18}_seed{0,1,2}.jsonl` = **3 档 × 4 dev 窗 × 3 种子**，**每个 20 行**（实测 36/36 都是 20 行，合计 **720 行**，`summary.invalid_rows=0`）。逐文件 `bytes` 与 `sha256` 见下表（`sha256` 取前 12 位；完整 64 位在 `ASSET_SYNC.sha256`，与 `manifest.json` **36/36 全等**）。5 项内部审计 **720/720** 全过；每个窗口的 engine gate（`hello_role_shadow` / `park_ok` / `park_frame=8663` / `threads_1` / `all_pass`）**36/36** 全过。

**档 U-600（600 决策 / 1200 帧）—— 12 格 / 1,134,615 B**

| 格（窗 / 种子） | 字节 | SHA-256（前 12 位） |
| --- | ---: | --- |
| w11 / seed0 | 94,344 | `f2f41c3e255f…` |
| w11 / seed1 | 94,976 | `4e33803726cb…` |
| w11 / seed2 | 94,884 | `a4bbe307dc19…` |
| w12 / seed0 | 94,547 | `f47d065eb02d…` |
| w12 / seed1 | 95,223 | `3390dc35b84d…` |
| w12 / seed2 | 94,840 | `7ff62a3e7384…` |
| w17 / seed0 | 94,567 | `ba537436ad5d…` |
| w17 / seed1 | 95,815 | `eda2116cf74e…` |
| w17 / seed2 | 94,622 | `f224182e12af…` |
| w18 / seed0 | 92,507 | `79a9c602581a…` |
| w18 / seed1 | 94,207 | `9991b9c15829…` |
| w18 / seed2 | 94,083 | `212bce5bd890…` |

**档 U-900（900 决策 / 1800 帧）—— 12 格 / 1,312,536 B**

| 格（窗 / 种子） | 字节 | SHA-256（前 12 位） |
| --- | ---: | --- |
| w11 / seed0 | 111,514 | `11822d9b92cb…` |
| w11 / seed1 | 111,407 | `48c724dd1434…` |
| w11 / seed2 | 109,549 | `50bc938adde4…` |
| w12 / seed0 | 108,043 | `b65e0d278389…` |
| w12 / seed1 | 109,068 | `a95ea4b6dbdf…` |
| w12 / seed2 | 108,415 | `d646b1d35947…` |
| w17 / seed0 | 107,628 | `70740c30859b…` |
| w17 / seed1 | 106,453 | `f31431a7b0a7…` |
| w17 / seed2 | 105,125 | `c25182f203a4…` |
| w18 / seed0 | 111,996 | `b0036be7c02d…` |
| w18 / seed1 | 112,493 | `c43d0cc768b6…` |
| w18 / seed2 | 110,845 | `9b302a761e03…` |

**档 U-1200（1200 决策 / 2400 帧）—— 12 格 / 1,340,784 B**

| 格（窗 / 种子） | 字节 | SHA-256（前 12 位） |
| --- | ---: | --- |
| w11 / seed0 | 114,491 | `57fc20f27334…` |
| w11 / seed1 | 114,817 | `42671ceedbe9…` |
| w11 / seed2 | 110,507 | `3a6904bb8f0e…` |
| w12 / seed0 | 116,200 | `7981ff36d324…` |
| w12 / seed1 | 109,342 | `7834cfa68eca…` |
| w12 / seed2 | 111,110 | `5d36528be55a…` |
| w17 / seed0 | 104,758 | `7c831ca74416…` |
| w17 / seed1 | 108,393 | `78cb57fb7134…` |
| w17 / seed2 | 104,936 | `86ddff24af81…` |
| w18 / seed0 | 116,153 | `1dccd9cafce8…` |
| w18 / seed1 | 116,910 | `0dabb3afe6c8…` |
| w18 / seed2 | 113,167 | `f417c503ab47…` |

#### 25.1.11 v8.20.0 统计（1 条 / 46,480 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.20.0/evidence/UNIFIED/summary.json` | 46,480 | `63400fc89139ad7a6051ae62cf0f672ac10926a139f0f3845a1e41baf7586ac0` | **P1/P2/P3 主判据**：`verdict.stop_condition="S-unified"`、`invalid_rows=0`；`P1` = U-600 **237/240 = 98.7500 %**，CP95 [96.3906, 99.7415]，对 H2s（236/240 = 98.3333 %）Fisher「更差」**p = 0.7752** ⇒ 无损失；`P2` = U-900 **58/240 = 24.1667 %**，CP95 [18.8918, 30.0917] 与 v8.19 S1 **逐位相同**，且 `row_identity_vs_v819_S1` 记 240/240 条逐条一致；`P3` = U-1200 **5/240 = 2.0833 %** 对 R5 36/3,600 = 1.0000 %，**+1.0833 pp、Fisher p = 0.1097、`substantive=false`**（未过 3 pp 线）；`budgets{U600,U900,U1200}` 各含 12 格 `{name, path, sha256, bytes}`（与盘上实测 **36/36 全等**）、`references`（H2s / R5 并排）、`notes` 5 条。**内部有一处自相矛盾（`references.H2s.metadata.per_window` 求和 235 ≠ `pooled` 236），只报告不改，见 §25.5 第 2 条** |

#### 25.1.12 执行器（2 条 / 157,775 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `tools/rl/v8k0_rulesweep30.py` | 73,609 | `37b220a02b4c742fe322922c2b18daf1c76f38575279dbd011f97947c219056b` | v8.19.0 评测执行器（`--mode preflight|assert-ckpt|d3|smoke|eval|summary|audio-audit|process-audit|manifest|publish`）。盘上现值 **`37b220a0…`**，与 `ckpt_assertions_{start,final}.json` 的 `self.sha256`、`RUN.md` §1 记的「本脚本」**三处逐位相同** ⇒ 整轮**执行器未漂移**（与 §24.4.5 第 3 条那轮不同）。**两份 manifest 都不含执行器**，其字节身份只由本节第 1 条 + 两份 `ckpt_assertions` 提供 |
| 2 | `tools/rl/v8l0_unified.py` | 84,166 | `ceae240c461f3a06c9c44a9612e42c07f4559ef132c3f2bdf6c48bd9600d1fe9` | v8.20.0 评测执行器（同一组模式 + 三档预算）。盘上现值 **`ceae240c…`** = `RUN.md` §6 记的「收尾版 sha256」、也 = `ckpt_assertions_final.json` 的 `self.sha256`；但 `ckpt_assertions_start.json` 记的是 **`c6f4c4b0…`**，而 `RUN.md` §7.6 列的三版是 `51bb5bdc…` → `4f9ef984…` → `ceae240c…` —— **`c6f4c4b0…` 不在这份清单里**（只报告，见 §25.5 第 3 条） |

### 25.2 跳过说明（未入册 **0** 个文件；`attempt1/` 的专门说明）

实测：v8.19.0 目录 **67** 个文件 / **5,476,913 B**，v8.20.0 目录 **52** 个 / **3,921,142 B**，执行器 **2** 个 / **157,775 B** ⇒ **121 个文件 / 9,555,830 B，入册 121、未入册 0**。

| 组 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `Document/Update/Alpha/v8.19.0/**` | 67 | 5,476,913 | **全部入册**（含 `manifest.json` 自身、`publish.json`、`preflight.txt`、7 件门/断言/审计、48 个 jsonl、`attempt1/` 4 件）；未入册 **0** |
| `Document/Update/Alpha/v8.20.0/**` | 52 | 3,921,142 | **全部入册**（含 `manifest.json` 自身、`publish.json`、三份 `preflight*.json`、6 件门/断言/审计、36 个 jsonl）；未入册 **0** |
| `tools/rl/v8k0_rulesweep30.py`、`tools/rl/v8l0_unified.py` | 2 | 157,775 | **全部入册**；未入册 **0** |
| **小计** | **121** | **9,555,830** | — |

#### 25.2.1 `attempt1/` 是失败那次的留档 —— 照登，但**不参与最终统计**

`Document/Update/Alpha/v8.19.0/evidence/RULESWEEP30/attempt1/` 的 **4 个文件 / 221,941 B** 是 **S4 / seed0 首跑（exit 4）** 的现场留档，**它不参与任何最终统计**，理由与边界如下：

1. **它不是发布行数据**：`summary.json` 的逐臂 `files{}` 与 `RUN.md` §6 的表**都不引用** `attempt1/` 里的任何路径；该臂最终的 `eval_S4_d900_w11_seed0.jsonl` / `_w12_` 来自**重跑**（tag `v8k0-eval-S4-d900-s0-r2`，exit 0），与 `attempt1/` 的同名件 **sha 不同**（见 §25.1.5）⇒ **同名不同字节，两批可区分**。
2. **它不改变任何判定**：`attempt1/` 只含 w11 / w12 两个窗口的**部分**条数（进度快照记跑到 40/80 条），既没有 w17 / w18，也不构成 240 条的整格；把它计入只会得到**残缺的第三方读数**，不会影响 S4 = 46/240 = 19.1667 % 这个发布值。
3. **为什么仍要照登**：它是唯一能证明「那一次确实失败、失败形态是什么、重跑是从哪天起算」的字节证据（`eval-S4-s0.out` 里有 `HarnessError: harness closed the control pipe` 的现场，`S4_d900_s0_progress.json` 有退出前的进度）。**不登记它**会让 `eval_exits.tsv` 里那条 `S4-s0 → 4 → 235s` 失去可追溯的产物锚。这与 §23.2 登记 0 字节 `labels_critical.jsonl`、§24.2 登记合成自测件是**同一口径：过程/失败证据要给字节身份，但不混入结论口径**。
4. **口径纪律**：日后引用 S4 的 19.1667 % 时，**只能引 `eval_S4_d900_w*_seed{0,1,2}.jsonl`（发布件）**；`attempt1/` 只能作为「失败与重跑过程」的证据引用，**不得与发布件合并求和**。

#### 25.2.2 运行目录（不在 `Document/` 下，不是发布件）

两轮的运行目录 `.tmp_v8k0_rulesweep30/` 与 `.tmp_v8l0_unified/` **不在本轮登记范围**：它们被 `.gitignore` 忽略、是过程现场，其**唯一需要字节身份的部分（评测行数据、统计、门禁、摘要）已全部发布到 `Document/` 并被本节 121 条完全覆盖**。运行目录里的 `fidelity_d3.json`（两份 `d3_gate.raw_sha256` 指向的原始件）、`eval/` 下的同名 jsonl、`data/` 的私有 cfg 等按 §21.2 / §22.2 / §23.2 / §24.2 同口径**不单独入册**。

#### 25.2.3 `__pycache__` / `*.pyc` 实况（按要求只报告、不删）

`v8.19.0/`（67 文件）与 `v8.20.0/`（52 文件）全树 **没有** `__pycache__/`、**没有** `*.pyc`（实测 0）。两轮目录内都没有可执行脚本体（命令原文记在 `commands.tsv`）。若日后出现，按 §21.2 口径**一律不入册**。

#### 25.2.4 明确**不登记** `v8.17.0/` 与 `v8.18.0/`

按工单：`Document/Update/Alpha/v8.17.0/` 与 `Document/Update/Alpha/v8.18.0/` **本轮不登记、不触碰**，**下一轮单独登记**。**但工单「两者都只有预注册」这一前提在登记时已对 `v8.17.0` 不成立**（只报告、不改）：实测 `v8.17.0/` 已有 **82 个文件 / 419,911,442 B**，含 `evidence/CONSERVE/` 的 6 个 `ckpt_{A,B}_nobomb_seed{0,1,2}.pt`（各 39,059 B）、`RUN.md`（14,874 B）与 `manifest.json`，**最后一个写入的正是 `manifest.json`（`15:42:02`）⇒ 该轮已发布完毕**；`v8.18.0/` 仍只有 **1 个** `PREREGISTRATION_EARLYBOMB_001.md`（4,301 B，`14:49:14`）。两者在 `ASSET_SYNC.sha256` 里各 **0 行**。本节**没有**读改写这两个目录里的任何文件（旁证：这两轮的引擎进程正是 v8.19.0 / v8.20.0 开工时机器上那 105–161 台 `v8i0_nobomb` 他轮进程，本轮按硬约束**未 `pkill`**、只计数）。

### 25.3 与 §1–§24 不重复的说明（实测已核）

- **既有正文 0 命中**：追加前对 `ASSET_SYNC.md` 全文（**2165 行 / 375,708 B**）做 `v8\.19\.0|v8\.20\.0|v8k0_|v8l0_|RULESWEEP30|UNIFIED` 匹配 → **0 命中**；§24 登记的是 v8.16.0（DEATHADJ）与执行器 `v8g0_deathadj.py`，与本轮**不同版本、不同目录、不同执行器**。
- **既有校验表 0 命中**：追加前对 `ASSET_SYNC.sha256`（**505 行**）同样匹配 → **0 命中**。
- **路径 121 唯一**：121 条路径逐条比对 **0 组重复**（`eval_S4_d900_w11_seed0.jsonl` 在 `evidence/RULESWEEP30/` 与 `evidence/RULESWEEP30/attempt1/` 下**同名但路径不同、sha 也不同**，故不构成重复登记）。
- **哈希 121 唯一**：121 个 sha256 **0 组重复**（`ckpt_assertions_start.json` 与 `_final.json` 在 v8.19.0 都是 4,848 B、在 v8.20.0 都是 4,999 B，但 sha **各自不同**：v8.19.0 `aafb04a7ceca…` vs `f3c9d9b96a80…`、v8.20.0 `a1e5892fb782…` vs `6a6a6fd1e20e…`）。
- **与既有 505 行 0 交集**：121 个**路径**与既有 **505 个路径**逐一比对 → **路径交集 0**；121 个 **sha256** 与既有 **490 个唯一哈希**逐一比对 → **哈希交集 0** ⇒ 没有把任何既有行「换个路径再钉一遍」，也没有把既有资产重复入册。
- **行数与格式**：505 + 121 = **626 行**（`ASSET_SYNC.sha256` 实测 **626 行**）；仍为 **LF / 无 BOM**、末行以 `\n` 结束、每行 `64 hex + 两个空格 + 相对路径（`/` 分隔、仓库根相对）`，**空行 0 条、格式异常行 0 条**、**CR 字节 0**。
- **`ASSET_SYNC.md` 本身不在校验表内**（`ASSET_SYNC` 在 `ASSET_SYNC.sha256` 里 **0 行**）⇒ 追加 §25 **不会**让任何既有行转 FAILED（与 §21.3 / §22.3 / §23.3 / §24.3 同一口径）。
- **Git 跟踪状态**：本轮按硬约束**未** `git add` / `git commit`，分支仍为 `v400`；两轮证据与两个执行器都**未入库** ⇒ 这些证据**无法由 Git 重建**，本节 + `ASSET_SYNC.sha256` 是它们目前的字节身份记录。

### 25.4 校验结论

#### 25.4.1 `sha256sum -c`（WSL，追加前后各一次）

命令（与 §22.4.1 / §23.4.1 / §24.4.1 同一口径）：

```bash
cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -c ': OK'; sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -v ': OK'
```

* **基线（追加前 505 行）**：**504 OK / 1 FAILED**。
* **追加后（626 行）**：**625 OK / 1 FAILED**。`625 + 1 = 626` 校验行全部参与。
* **新增 121 条 121/121 OK** ⇒ 整体 FAILED 数**没有新增（仍为 1）**。
* **唯一 FAILED 仍是已知并允许的那一条**：`Document/Update/Alpha/v4.2.0/journal.jsonl`（§9.8 记录的活动文件漂移）。本轮**未改该行、未重钉、未触碰该文件**，三个校验时刻（追加前 / 追加后 / 写完 §25 后）该行是**同一条 FAILED**。
* **格式**：626 行仍为 **LF / 无 BOM**、末行 `\n` 结束、每行 `64 hex + 两个空格 + 相对路径`；空行 **0**、格式异常 **0**；新增 121 条的路径在表内**各自唯一**。

#### 25.4.2 与两份产出方 `manifest.json` 的交叉核对（登记时实测）

| 清单 | 自述 | 实测比对 |
| --- | --- | --- |
| v8.19.0 `evidence/RULESWEEP30/manifest.json` | `file_count=65`、`total_bytes=5,463,957`、`excludes="manifest.json"` | `files{}` **65/65** 的 `{bytes, sha256}` **全等、0 处不符**；`bytes` 求和 = `total_bytes` = 5,463,957；盘上 `evidence/RULESWEEP30/**` 共 **66** 件，manifest **未列**的**只有 `manifest.json` 自身**（9,452 B）⇒ **无缺件、无多余件**（5,473,409 − 5,463,957 = 9,452） |
| v8.20.0 `evidence/UNIFIED/manifest.json` | `file_count=50`、`total_bytes=3,910,975`、`excludes="manifest.json"` | `files{}` **50/50** 全等、**0 处不符**；`bytes` 求和 = `total_bytes` = 3,910,975；盘上 `evidence/UNIFIED/**` 共 **51** 件，manifest **未列**的**只有 `manifest.json` 自身**（7,377 B）⇒ **无缺件、无多余件**（3,918,352 − 3,910,975 = 7,377） |
| v8.19.0 `summary.json` 的逐臂 `files{}` | 每格 `{rows, full, sha256, bytes}` | **48/48** 与盘上实测全等、0 处不符；`rows` 全为 20 |
| v8.20.0 `summary.json` 的 `budgets{U600,U900,U1200}` 12 格 | 每格 `{name, path, sha256, bytes}` | **36/36** 与盘上实测全等、0 处不符 |
| 两份 `ckpt_assertions_*.json` 的 `prereg.sha256` | 冻结预注册身份 | v8.19.0 `39566638…`、v8.20.0 `199a567e…`；与**盘上实测预注册 sha 逐位相同**，两轮都是 start = final |

**两份 manifest 都未覆盖的内容（如实列出）**：`manifest.json` **不含** `tools/rl/v8k0_rulesweep30.py` 与 `tools/rl/v8l0_unified.py` 的登记（对两份 `files{}` 做 `v8k0_` / `v8l0_` 匹配 = **0 命中**）。两个执行器的字节身份由**§25.1.12 的第 1–2 条**（实测）+ 两份 `ckpt_assertions` 的 `self.sha256` 提供（v8.19.0 三处同值；v8.20.0 见 §25.5 第 3 条的分歧）。

#### 25.4.3 身份链、门与读数（只登记，不重判）

| 核对项 | 实测 |
| --- | --- |
| 引擎身份 | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` / match=true，出现在两轮的 `d3_gate.json`、`ckpt_assertions{,_final}.json` 与两份 `summary.json`；与 §2 / §22 / §23 / §24 记录的 `rl_v3` SHA **一致** |
| 行为策略身份 | v8.6.0 `bombdag.pt` = `9c6d5d4a…` / `36a51090…` / `885dde70…`（各 39,069 B），两轮各自的 start/final 两相 **各 3/3 match**；v4.5.0 `round2.pt` = `4abee069…` / `a93746bb…` / `c69960fc…` 同样两相 3/3 match ⇒ 两轮评测期间**候选头未换** |
| **预注册身份** | v8.19.0 `39566638ad19ef5c…` / 3,504 B、v8.20.0 `199a567ebe26198e…` / 2,790 B —— 各自在 `ckpt_assertions{,_final}.json` **两相同值**、并与盘上实测 sha 相同 ⇒ 两轮都是「预注册先冻结、再开跑」 |
| **跨轮可重复性锚（v8.19 → v8.20）** | v8.20.0 的 U-900 **不是新臂，是对 v8.19 S1 的可重复性检查**：同候选头、同规则（60/40/20）、同 `seed*1000+7`、同 12 格循环 ⇒ 池化同为 **58/240 = 24.1667 %**、CP95 **逐位相同** [18.8918, 30.0917]、逐窗口 16/11/11/20 **完全相同**，且 `row_identity_vs_v819_S1` 记 **240/240 条 episode 在 5 个字段上 0 不匹配**、**240/240 条的 900 个 `actions_sampled` 抽样序列完全相同** |
| D3 保真门 | 两轮都是 **5/5 PASS**（`all_pass=true`）：HELLO `role=shadow`、DEBUG `threads=1`、前缀 **2,165 / 2,165** checkpoint **0 mismatch**（到帧 8663）、R24 **18/18** 常量、PARK `frame=8663`；同一 `verdict` 字符串 |
| 音频（§9 硬约束） | v8.19.0 **52 / 52 muted / 0 audible / 0 errors**；v8.20.0 **37 / 37 muted / 0 audible / 0 errors** ⇒ 合计 **89 个 cfg / 89 muted / 0 audible**；两轮都在进程内 `SDL_AUDIODRIVER=dummy` 并 `pop` 掉 `DISPLAY` / `WAYLAND_DISPLAY`；v8.19.0 的 `preflight.txt` 还留下「这两条变量在脚本外是**存在**的」的旁证；共享审计件 `v4.3.1/evidence/R14/audio_cfg_audit.json` **未触碰** ⇒ 两轮**未新增 `audible`** |
| 进程残留 | 两轮 `process_residue.json` 都是 **`matching=0` / `lines=[]` / `all_pass=true`**；开工时机器上分别有 **107**（v8.19.0 `preflight.txt`，其中 105 台 `v8i0_nobomb`）与 **153 → 90**（v8.20.0，等满 5 分钟后）台**他轮**引擎进程，结束时他轮仍 **161** 台（v8.20.0 `preflight_final.json`）—— 两轮都**只计数、未 `pkill`** |
| v8.19.0 结论口径 | `summary.verdict.stop_condition="S-win"` 与 `RUN.md` §0/§7 的 **`S-win`** 一致：S3 = 59/240 = 24.5833 %（p = 0.0054）、S1 = 58/240 = 24.1667 %（p = 0.0082）过 P1/P2；**P3 的「更省雷」预期被证伪**（四臂 240/240 都恰好用掉 3 颗雷）；同头对照晚臂组 vs 早臂组 **+5.62 pp（p = 0.0206）** ⇒ 阈值动的是「第三颗雷的时机」 |
| v8.20.0 结论口径 | `summary.verdict.stop_condition="S-unified"` 与 `RUN.md` §0/§4 的 **`S-unified`** 一致：U-600 **98.7500 %**（≥95 %，P1 成立）、U-900 **24.1667 %**（P2 成立）、U-1200 **2.0833 %**（P3 **未达** +3 pp 线，`substantive=false`）⇒ 20 s / 30 s 用同一套阈值，**40 s 的瓶颈不是阈值远近** |
| 两轮的「不能声称什么」都照预注册保留 | 两轮都明写「不能声称学会了资源分配」「不能把不同预算的均值直接比较」「不能把 12 格当独立样本」；v8.20.0 另加「不能声称 60/40/20 在 40 s 上更好」⇒ 本节**照录不重判** |

#### 25.4.4 登记期零漂移（两轮目录与两个执行器只读）

登记**开始前**与**结束后**各采集一次这 **121** 个文件的 `{size, mtime_ns}` 并逐项比对：

| 范围 | 文件数 | 差值 |
| --- | ---: | --- |
| `Document/Update/Alpha/v8.19.0/**` | 67 | `size` 0 变化 / `mtime_ns` 0 变化 |
| `Document/Update/Alpha/v8.20.0/**` | 52 | `size` 0 变化 / `mtime_ns` 0 变化 |
| `tools/rl/v8k0_rulesweep30.py`、`tools/rl/v8l0_unified.py` | 2 | `size` 0 变化 / `mtime_ns` 0 变化 |
| **合计** | **121** | **121/121 完全相同，0 个变化** |

**「等 120 秒再钉」被执行了两次，第二次是因为 v8.20.0 在第一次等待之后被重新发布。**

* **第一次**：登记开始时 v8.20.0 目录内最新 mtime = `manifest.json` 的 `15:33:44`，距首次快照（`15:34:55`）不足 5 分钟 ⇒ 等 125 s：快照 1（`15:34:55`，119 文件，sha `7a11535d07848004…`）与快照 2（`15:37:00`，同 sha）**逐位相同**，`DRIFT=none`。
* **但是**：`15:38:28–15:38:38` 之间 v8.20.0 的 **`RUN.md`、`publish.json`、`manifest.json`** 三个文件被**重新发布**（`RUN.md` 19,809 B、`total_bytes` 由 3,910,948 变为 3,910,975）⇒ 第一次等待的结论**已作废**。
* **第二次**：对**全部 121 个文件**重新快照并再等 130 s（`{size, mtime_ns}` 逐项比对）⇒ **两次快照逐位相同、`QUIESCENT=YES`**，此后才把 121 条的数字与 sha256 **全部重新实测**并钉入本节与 `ASSET_SYNC.sha256`。
* 结论：本节所有 `{bytes, sha256}` 都是**重新发布之后**的状态；两轮目录在最终等待窗口内**没有任何一个文件被写**，且两个执行器的 `{size, mtime_ns}` 在登记前后同样一致。

本轮对上述路径的写操作 = **0**（只读 `os.walk` / `Get-ChildItem` / `hashlib` / `sha256sum` / `Get-Content` / `ConvertFrom-Json` / `Select-String`）；**未新建、未改名、未删除、未移动**任何文件；写入**只落在** `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256` 两个清单文件上。同时确认 `v8.17.0/` 与 `v8.18.0/` **未被读取改写**（本轮未登记、未触碰该两目录）。

#### 25.4.5 只报告、不裁定的偏差清单

1. **v8.19.0 的对照 R0 与 S1–S4 是不同候选头**（预注册设计自带，`RUN.md` §8.1 已写明）：R0 用 v4.5.0 `S` 学生（`4abee069…`）、本轮 S1–S4 用 `bombdag.pt`。所以「臂 vs R5」的 Fisher **同时混了候选头与阈值两个因子**，不是纯阈值因果；纯阈值效应只能用 `within_sweep_contrasts.json` 的**同头对照**（+5.62 pp，p = 0.0206）支持。本节照录，**不重判**。
2. **v8.19.0 `RUN.md` §2 引的 `fidelity_d3.json` 不在发布目录里**：`RUN.md` §2 写「`fidelity_d3.json` sha256 `5e05916fbdcd47ba…`」，但盘上该 sha 是**运行目录** `.tmp_v8k0_rulesweep30/fidelity_d3.json` 的（= `d3_gate.json` 的 `raw_sha256`，**同值**）；发布目录里的对应件叫 **`d3_gate.json`**（`815a8fde…`）。**这是发布时的改名，不是缺件** —— v8.20.0 的 `RUN.md` §2.1 就按 `d3_gate.json`（`48da324b…`）记。本节按盘上**按名登记 `d3_gate.json`**，并在此记录该改名。
3. **v8.19.0 的 `S4/seed0` 首跑失败并整格重跑**：`eval_exits.tsv` 记 `S4-s0` **exit 4 / 235 s**，`RUN.md` §8.2 记成因 `HarnessError: harness closed the control pipe`（当时 168 台引擎、loadavg ≈ 50）；处置是把已完成的 w11/w12 留档到 `attempt1/`，再以新 tag `…-r2` **整格重跑 4 窗**（不续跑，避免重置抽样流）。发行的 48 个 jsonl **全部来自成功批次**；`attempt1/` 照登但**不参与最终统计**（§25.2.1）。
4. **一次打包事故（v8.19.0 `RUN.md` §8.8 自述）**：第一次 `publish` 前的 `cp` 里证据目录变量被外层 PowerShell 引号转义吃掉，导致 `within_sweep_contrasts.json` 与 `attempt1/` 的 4 个审计文件被写到**文件系统根 `/`**；发现后逐项确认绝对路径、移入正确目录、删除根下误放副本并**重建 `manifest.json`**。`RUN.md` 称 `verify_publish.py` 复核「48 个 eval 文件与 WORK 逐字节相同、manifest 全部 65 个 sha256 与磁盘一致」。**本节实测正好印证了这一说法**：manifest 65/65 与盘上全等（§25.4.2）⇒ **当前发布件自洽**；事故本身**只记录、不重判**。
5. **v8.20.0 的执行器在轮内被改过**：`RUN.md` §7.6 记三版 `51bb5bdc…`（批次）→ `4f9ef984…`（加 P2 逐条检查）→ **`ceae240c…`（收尾，= 盘上现值）**，并声明「评测路径未动，只改 `--mode summary` / preflight 历史 / publish 清单」。本节按盘上值登记，**只记录事实**；`51bb5bdc…` 与 `4f9ef984…` 两个状态**盘上已不存在**（无法重建）。另注意 `ckpt_assertions_start.json` 记的是 **`c6f4c4b0…`**，**不在这三个版本里**（§25.5 第 3 条）。
6. **v8.19.0 的冒烟是在 900 决策配额内死亡的**：`smoke.json` 的 `survived=false` / `alive_frames=1584` / `decisions_executed=792`；`RUN.md` §4 只记该命令 exit 0 / 22 s，**未记这条读数**。这不构成失败（冒烟只验证装置与门），但**不能**把该冒烟当作「900 决策可活满」的证据 —— 活满率的证据是 48 个 jsonl（960 条）。对照 v8.20.0 的冒烟 `survived=true` / `decisions_executed=600`。
7. **两轮都是「零训练、只评测」轮次**：`summary.notes[0]` 均写 `"No training and no collection: this file only evaluates."`；预注册 §4 列的 `train_*` 产物**本就不存在**（不是登记遗漏）。本节只登记字节身份与已在盘上的读数，**不对结论重判**。
8. **v8.19.0 `process_residue.json` 缺 `data_dir_filter` 字段**：与 §24.4 登记的 v8.16.0 同名件（有 `data_dir_filter`、`processes_total_by_engine_name`、`pgrep_returncode=1`）结构不同；本件只有 `matching=0` / `lines=[]` / `pgrep_returncode=0` / `all_pass=true`。`pgrep_returncode=0` 配合 `lines=[]` 说明它是**过滤后**的普查，但文件里没有写明过滤条件 ⇒ 「0 残留」这一读数的**过滤口径无法从本件自证**，需与 `preflight.txt`（`engine_processes_total=107`、按轮分组）配合读。**只报告、不裁定。**
9. **v8.20.0 在登记期间被重新发布过一次**（见 §25.4.4 与 §25.5 第 10 条）：`15:38:28–15:38:38` 之间 `RUN.md`（+27 B）、`publish.json`、`manifest.json` 被重写，`total_bytes` 3,910,948 → 3,910,975。本节钉的是**重发布之后**的状态；重发布**没有**改变任何评测行数据（36 个 jsonl 与 `summary.json` 的 mtime 都 ≤ `15:32:18`），也没有改变本文引用的任何结论句（`RUN.md` 里「运行窗 15:41」「其他轮 110 台」「三版执行器」等措辞在重发布后**依然存在**）。
10. **`v8.17.0/` 在本轮登记时已经发布，工单「只有预注册」的前提已过期**：实测 **82 文件 / 419,911,442 B**，`manifest.json` 是目录内最后一个写入（`15:42:02`）⇒ 该轮发布完毕；`v8.18.0/` 仍只有 1 个预注册（4,301 B）。按工单**本轮不登记**（下一轮单独登记），本节**未触碰**这两目录；这条只报告事实，供下一轮派单时修正前提。（同一时刻 `v8.20.0` 也在 `15:38:28–15:38:38` 被重新发布过，见 §25.5 第 10 条与 §25.4.4。）

### 25.5 与 manifest 不符项（**只报告、不改**）

> **总结论：两份 `manifest.json` 与盘上 0 不符。** v8.19.0 `files{}` **65/65**、v8.20.0 `files{}` **50/50** 的 `{bytes, sha256}` **逐条全等**，`bytes` 求和分别等于各自的 `total_bytes`，未列出的文件**只有 `manifest.json` 自身**。下面列的**不是** manifest 的错，而是「manifest 覆盖不到」或「manifest 之外的文件自相矛盾」的 **10 项**，按工单**只报告、未改任何文件**。

| # | 项 | 事实（实测） | manifest 侧 | 处置 |
| ---: | --- | --- | --- | --- |
| 1 | **manifest 自述「除自身」且不含顶层预注册** | v8.19.0 `evidence/RULESWEEP30` 目录 **66 / 5,473,409 B** vs 自述 **65 / 5,463,957**，差 **9,452 B = `manifest.json` 自身**；再加顶层预注册 3,504 B ⇒ 全目录 **67 / 5,476,913 B**（v8.20.0 同理：51 / 3,918,352 vs 50 / 3,910,975，差 **7,377 B**，+ 2,790 = **52 / 3,921,142 B**） | **不是不符**，口径已闭合 | 两个口径都写进 §25 起首表 |
| 2 | **v8.20.0 `summary.json` 内部 H2s 参考块自相矛盾** | `references.H2s_600_80_60_40.metadata.pooled = [236, 240]`，但 `metadata.per_window = {11:[60,60], 12:**[58**,60], 17:[57,60], 18:[60,60]}` **求和 = 235 ≠ 236**；同一文件的 `scan.per_window[12] = [**59**,60]`（求和 236，与 `scan.full=236` / `pooled` 一致）。`RUN.md` §3.3 也写 236/240。⇒ **`metadata.per_window[12]=58` 与同文件 `scan` 及 `RUN.md` 都不一致（差 1 条）** | manifest 对 `summary.json` 的**字节身份记录正确**（`63400fc89139ad7a` / 46,480 B）；矛盾在文件**内容**里，不在 manifest | **只报告、不改** |
| 3 | **v8.20.0 `ckpt_assertions_start.json` 记的执行器 sha 不在 `RUN.md` 的版本清单里** | 该文件 `self.sha256 = **c6f4c4b0…**`；`RUN.md` §7.6 列的三版是 `51bb5bdc…` → `4f9ef984…` → `ceae240c…`（末者 = `ckpt_assertions_final.self` = **盘上现值**）。⇒ **实际至少有 4 个状态**，`RUN.md` 只列了 3 个；`c6f4c4b0…` 这一版**盘上已不存在**，无法用字节复核 | manifest 不含执行器（0 命中），故 manifest **不冲突**；执行器身份由 §25.1.12 实测提供 | **只报告、不改**；按盘上值 `ceae240c…` 登记 |
| 4 | **v8.19.0 `RUN.md` §2 引的 `fidelity_d3.json` 不在发布目录** | 该 sha `5e05916fbdcd47ba…` = `d3_gate.json` 的 `raw_sha256`（运行目录原始件）；发布目录里对应件名为 **`d3_gate.json`**（`815a8fde…`）。**没有缺件**，是发布改名 | manifest 只登记 `d3_gate.json`（正确） | **只报告、不改**（按名登记 `d3_gate.json`） |
| 5 | **v8.20.0 `RUN.md` 首段的「运行窗 15:18–15:41」无盘上支撑** | 目录内**最大** mtime = `manifest.json` 的 **15:38:38**；评测批次按 `eval_exits.tsv` 在 **15:28:16** 结束、`publish.json` 15:38:37。⇒ **15:41 之后没有任何一个文件被写** | manifest 与盘上一致（15:38:38） | **只报告、不改** |
| 6 | **v8.20.0「其他轮 110 台」与三份 preflight 不一致** | `RUN.md` §7.1 与 §6 第 16 条都写「其他轮 **110** 台」（2 处）；三份普查实测为 `preflight.json` **153**、`preflight_start.json` **90**、`preflight_final.json` **161**，**没有 110** | manifest 不涉及 | **只报告、不改** |
| 7 | **两轮 manifest 都不含执行器** | 对两份 `files{}` 做 `v8k0_` / `v8l0_` 匹配 = **0 命中** | **不是不符**（manifest 的 `root` 只在 `evidence/<TAG>/`） | 执行器身份由 §25.1.12 + `ckpt_assertions.self` 提供 |
| 8 | **v8.19.0 `process_residue.json` 缺 `data_dir_filter`** | 只有 `{matching:0, lines:[], pgrep_returncode:0, all_pass:true}`，**没有**过滤条件字段（§24.4 的 v8.16.0 同件有）⇒「0 残留」的过滤口径**无法从本件自证** | manifest 记录该文件字节**正确** | **只报告、不改** |
| 9 | **v8.19.0 `attempt1/` 与发布件同名不同字节** | `attempt1/eval_S4_d900_w11_seed0.jsonl`（`c8a5c45270008b0d…`）与其发布件（`dc801641b80e4872…`）不同；`_w12_` 同理（`a450634a1ff87942…` vs `e48eb9ea2cffcd79…`）⇒ 同名文件存在两代 | manifest **两代都登记**（`attempt1/*` 4 条 + 发布件 48 条），**无冲突** | 照登 `attempt1/`，并在 §25.2.1 写明**不参与最终统计** |
| 10 | **v8.20.0 在登记期间被重新发布** | `15:38:28–15:38:38` 之间 `RUN.md`（19,782 → 19,809 B）、`publish.json`、`manifest.json` 被重写，`total_bytes` 3,910,948 → 3,910,975、`file_count` 仍 50。⇒ 本节以**重发布后**状态为准；重发布**前后 manifest 都与盘上自洽**（各自 50/50 全等），所以这不是「manifest 不符」，而是**同一轮存在两个发布快照**、旧快照的数字**已不可引用** | 重发布后的 manifest 与盘上 50/50 全等 | **只报告、不改**；已按重发布后状态重测全部 121 条（§25.4.4） |

**逐条核对口径（可复算）**：把两份 `manifest.json` 的 `files{}` 逐项与 `sha256sum` 实测比对，`{bytes, sha256}` **不符数 = 0**（v8.19.0 65 项全过、v8.20.0 50 项全过）；`bytes` 求和与 `total_bytes` **差 0**；盘上「有而 manifest 无」= **1 项**（各自的 `manifest.json`）、「manifest 有而盘上无」= **0 项** ⇒ **无缺件、无多余件**。

> **一句话结论**：本轮把 **v8.19.0（RULESWEEP30）67 条 / 5,476,913 B + v8.20.0（UNIFIED）52 条 / 3,921,142 B + 两个执行器 2 条 / 157,775 B = 121 条 / 9,555,830 B** 成对写入 `ASSET_SYNC.md` §25 与 `ASSET_SYNC.sha256`，`sha256sum -c` 由 **504 OK / 1 FAILED（505 行）** 变为 **625 OK / 1 FAILED（626 行）**（新增 121 条全 OK，整体 FAILED 数仍为 1，唯一 FAILED 仍为已知允许的 `v4.2.0/journal.jsonl`）。两个口径都记：v8.19.0 `evidence/RULESWEEP30` **66 / 5,473,409 B**（目录）vs **65 / 5,463,957 B**（manifest 口径）；v8.20.0 `evidence/UNIFIED` **51 / 3,918,352 B**（目录）vs **50 / 3,910,975 B**（manifest 口径），两处差额都恰为 `manifest.json` 自身 ⇒ **无缺件、无多余件**；manifest 逐条比对 **65/65 + 50/50 全等、0 处不符**。两轮**未启动引擎、未训练、未评测、未 `git add/commit`**；121 个受保护文件的 `{size, mtime_ns}` **逐个零漂移**（且按要求先后等了 125 s 与 130 s 两轮才确认静态）；`v8.17.0/` 与 `v8.18.0/` 按要求**未登记、未触碰**（下一轮单独登记）。

## 26. v8.17.0 资产（2026-11-22）

> **为什么有这一节**：AGENTS.md §8.2 要求「新增/移动/删除任何资产，必须同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`」。**v8.17.0（CONSERVE-001 的 `none` 分支 / NOBOMB）已发布，但追加本节前一行都没有入册** —— 追加前对既有 **626 行**校验表做 `v8\.17\.0|CONSERVE|v8i0_|nobomb` 匹配 = **0 命中**；本册正文里 `v8.17.0` 只出现在 §25.2.4、§25.4.5 第 10 条、§25.5 第 10 条三处，三处都明写「**本轮不登记、下一轮单独登记**」⇒ 本节就是那一轮，**与 §1–§25 不重复**（§26.3）。本节与 `ASSET_SYNC.sha256` 末尾追加的 **83 行**成对登记。

> **本轮性质：`--bomb-rule none` 分支的「采集 + 训练 + 18 格评测」轮，全部 live engine**（预注册 §1 的保守 25 px 规则在 §2.1 的 `S-nobomb` 兜底条款下被换成「一次不放雷」）：12 格 × 50 状态 = **600 个死亡邻域状态**（每状态 36 路 × H=120/240/600 ⇒ 12 格合计 **21,600 条标签 / 1,800 行**，池化到 `labels_nobomb.jsonl`），A/B 两臂 × 3 种子训练（`Policy(106,64,36)`，`Linear(106,64)->Tanh->Linear(64,36)`，9,188 参数，200 epoch），A/B × 300/600/900 × 3 种子 × 4 dev 窗 × 300 episode = **18 格 / 21,600 episode**。判定 **`S-weak`**：P0a **600/600 = 100 %**、P0b **588/600 = 98.0 %**（标签第一次能表达「死那一刻手里那颗雷」的价值），但 **P1 不成立**（B d600 **18.500 %** ≤ v8.3.0 的 **19.444 %**）、**P3 不成立**（**0/3,600** 局只用 1–2 颗，每局都把 3 颗花光）。

> **只读边界**：本轮**只 append** 两个清单文件。`Document/Update/Alpha/v8.17.0/**`（**82** 个文件 / **419,911,442 B**）与执行器 `tools/rl/v8i0_nobomb.py`（**1** 个 / **177,079 B**）全程只读；登记前后在 **15:49:00.837** 与 **15:51:10.926**（间隔 **130 s**）各采一次 `{size, mtime}` 快照，83 条 sha256 全部算完后再采第三次（**15:51:11.781**），**83/83 的 `{size, mtime_ns}` 三次逐项相同、0 变化**（§26.4.4）。**未启动引擎**（0 次引擎调用）、**未训练、未评测、未 `git add` / 未 `git commit`**。

> **实测口径**：83 条的 `bytes` 与 `sha256` **全部现场实测**，且**两条独立路径各算一遍** —— PowerShell `Get-FileHash`（83/83）与 Python `hashlib` 流式读入（83/83），两者**逐位相同**；再用 WSL `sha256sum -c` 独立复算（§26.4.1）。发布方 `manifest.json` 的内联 `{bytes, sha256}` **只用于交叉核对，不代替实测**（§26.4.2：**80/80 全等、0 处不符**）。

> **两个口径都写（本册一贯做法）**：目录口径 = 递归实测总数（含 `manifest.json` 自身与顶层预注册）；`manifest.json` 自述口径 = 该文件 `file_count` / `total_bytes`，**`excludes="manifest.json"`**，且因 `root` 在 `evidence/CONSERVE/` 而**不含**顶层预注册。
>
> | 口径 | 文件数 | 字节 | 差额闭合 |
> | --- | ---: | ---: | --- |
> | v8.17.0 全目录（含顶层预注册 + `manifest.json` 自身） | **82** | **419,911,442** | — |
> | `evidence/CONSERVE/`（含 `manifest.json` 自身） | **81** | **419,907,011** | 419,907,011 − 419,895,193 = **11,818 B = `manifest.json` 自身** |
> | `manifest.json` 自述 | **80** | **419,895,193** | `excludes="manifest.json"`、`root=…/v8.17.0/evidence/CONSERVE` |
> | 顶层预注册（manifest 覆盖不到） | **1** | **4,431** | 82 − 81 = 1；419,911,442 − 419,907,011 = 4,431 |
>
> 两处差额都恰好闭合（**4,431 + 11,818 = 16,249**；**419,911,442 − 419,895,193 = 16,249**）⇒ **无缺件、无多余件**（与 §23.4.2 / §24.4.2 / §25.4.2 同一口径）。

> **全量入册口径：82 个文件全部入册，未入册 0 个** —— 含 `manifest.json` 自身、运行目录清单 `nobomb_work_manifest.json`、`nobomb_exec.sh` / `nobomb_make_summary.py` 两个脚本、6 个 `.pt`、12 个采集格记录、18 份逐格 `.summary.json`。理由：这些**都不是可重跑重建的日志**，而是本轮唯一的采集 / 训练 / 评测 / 门禁 / 摘要出口。**本轮最小集 = 全集 + 执行器。**

### 26.1 登记范围（**83 条** = v8.17.0 **82 条** + 执行器 **1 条** / **420,088,521 B**）

| 分组 | 条数 | 字节合计 | 位置 |
| --- | ---: | ---: | --- |
| §26.1.1 v8.17.0 冻结预注册 | 1 | 4,431 | `Document/Update/Alpha/v8.17.0/` |
| §26.1.2 运行记录、执行脚本与命令表 | 5 | 56,664 | `v8.17.0/evidence/CONSERVE/` |
| §26.1.3 身份门、保真门与三项审计 | 6 | 48,866 | `v8.17.0/evidence/CONSERVE/` |
| §26.1.4 死亡邻域采集记录、标签集与召回对照 | 15 | 19,833,031 | `v8.17.0/evidence/CONSERVE/`（12 件在 `collect/`） |
| §26.1.5 训练记录、6 个 checkpoint 与训练汇总 | 13 | 301,538 | `v8.17.0/evidence/CONSERVE/` |
| §26.1.6 评测行数据与逐格摘要（18 格 × 2 件） | 36 | 399,280,730 | `v8.17.0/evidence/CONSERVE/` |
| §26.1.7 统计与诊断出口 | 4 | 73,193 | `v8.17.0/evidence/CONSERVE/` |
| §26.1.8 清单锚（发布清单 + 运行目录清单） | 2 | 312,989 | `v8.17.0/evidence/CONSERVE/` |
| **v8.17.0 小计** | **82** | **419,911,442** | — |
| §26.1.9 执行器 | 1 | 177,079 | `tools/rl/` |
| **合计** | **83** | **420,088,521** | — |

#### 26.1.1 v8.17.0 冻结预注册（1 条 / 4,431 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.17.0/PREREGISTRATION_CONSERVE_001.md` | 4,431 | `79294a44c5d3a066fddcd5174e744cc3b9004b74533c454cd75c1056d99343c7` | v8.17.0 冻结预注册（**FROZEN**，在任何 v8.17.0 采集之前，`14:18:33` 落盘）：单因子 = **行为策略的放雷阈值**；P0a 死亡邻域 `bombs_remaining > 0` **≥50 %**、P0b 其中 `best bomb1 > best bomb0` **≥20 %**、P1 = 臂 B 的 600 决策活满率 **>19.44 %**（v8.3.0）、P3 = 出现「只用 1–2 颗雷」的 episode、P4 = 900 档并排；§2.1 停止条件 `S-nobomb` / `S-nodiff` / `S-ok` / `S-partial` / `S-fork`。**三处独立记录同 sha**（`nobomb_preflight.json` 的 `pins.prereg`、`nobomb_ckpt_assertions{,_final}.json` 的 `prereg`、盘上实测）⇒ 轮内**一字未改**；本轮走的正是 §2.1 授权的 `S-nobomb` 兜底（文件名随之全部加 `nobomb` 前缀，见 §26.5 第 7 条） |

#### 26.1.2 运行记录、执行脚本与命令表（5 条 / 56,664 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/RUN.md` | 14,874 | `fa0282fbc11db5b9ac70127b837d643c6d9cd62fd6fa8313a78a658724f30cf2` | **248 行** / 14,874 B 运行记录：§0 为什么开 `none` 分支（三行表：v8.16.0 死亡邻域 `0/1,431` 带雷、25 px 规则 `0/27` 带雷、本轮 `600/600` 带 3 颗雷）→ 只有「一次不放」能保证死时手里有雷；§1 冒烟（死亡决策 45、窗口 26..45、`bombs 3->3`、`emits=0`、H 嵌套 0 mismatch）；§2 P0a/P0b（三个 H 都 600/600 与 588/600）与「`critical` 不作门」的判据说明；§3 机制（600/600 状态都是 `DYING`，deathbomb **一次扣 2 颗**，`bomb1_press_honoured=10,584`）；§4 训练 A/B 与 final BCE；§5 **18 格评测**（21,600 episode、逐种子表、与 v8.3/v8.6/R5 并排）；§6 用雷分布（P3）；§7 落盘与身份；§8 音频 **13/13 muted**；§9 判定表（P0a PASS / P0b PASS / P1 不成立 / P3 不成立 ⇒ **`S-weak`**）；§10 **6 条偏差**；§11「不能声称什么」4 条 |
| 2 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/nobomb_RUN.md` | 14,874 | `fa0282fbc11db5b9ac70127b837d643c6d9cd62fd6fa8313a78a658724f30cf2` | **与第 1 条逐字节相同**（sha 相同、字节相同，mtime 晚 23 ms）—— 发布方把同一份运行记录用 `RUN.md` 与 `nobomb_RUN.md` 两个名字各放了一份，见 §26.5 第 5 条。**照登**（路径不同，不构成重复登记） |
| 3 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/nobomb_commands.tsv` | 10,773 | `b62d3139ca3e408a2ad67d658aea9350a679fb1e794d7b142ec0ae2922b93eb2` | **70 行**（1 行表头 + **69 条**命令）六列 `label / exit / wall_s / started / finished / argv`：preflight 1、assert 2、recall 1、fidelity 1、smoke 3、collect 12、stats 1、train 7、audio 1、**eval 37**、diagnostics 1、process 1、manifest 1 = 69。**非 0 退出共 18 条**：`smoke` **exit 4 / 88 s**（首版账本假设，§26.5 第 10 条）+ 首轮 18 格评测里的 16 条（**10 条 exit 4 / 6 条 exit 130**，§26.5 第 9 条）。是「命令原文」的唯一锚（`RUN.md` 未逐条列全 69 条） |
| 4 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/nobomb_exec.sh` | 7,718 | `705b6b83ab2e2368455d1e365897374c4937f42feda4e32cc756b5ad9d0a6b9d` | **233 行** WSL 侧执行脚本（`bash nobomb.sh <step>`，一次一个步骤）：每个 data dir 现场生成静音 `th08.cfg`（`v400_make_cfg.py --out` + `--inspect`）、启动环境 `SDL_AUDIODRIVER=dummy` 且 `DISPLAY` / `WAYLAND_DISPLAY` 未设置、每个 eval 格显式 `--tag`（首轮就是漏了这个参数，§26.5 第 9 条） |
| 5 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/nobomb_make_summary.py` | 8,425 | `c07a8451c6fb48aa914719f328207a7ce2e7f2b07eac7a217a0a184084f65b9e` | **164 行** 离线汇总脚本（`--mode summary` 的实现，只读已落盘的 18 个 jsonl 与 12 个采集格，**不启引擎**）⇒ `summary.json` / `nobomb_diagnostics.json` / `nobomb_bomb_usage.json` 都可从本件 + 已发布行数据**离线复算** |

#### 26.1.3 身份门、保真门与三项审计（6 条 / 48,866 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/nobomb_preflight.json` | 10,230 | `06899b7d9c4fae4e97757b3fdd3ba3e432873c4aed7878c9bf3cbdb68cf39624` | 开工前身份门（`14:28:52`，`all_pass=true`）：**12 个受保护源文件全部 match**（含 `tools/rl/v450_cf_label.py` `b7ab0ebe…`、`tools/rl/v830_bombpol.py` `a8c6d306…`、`touhou_rl/v420/contracts.py` `1357c69c…`、`touhou_rl/v400/gym/protocol.py` `9081402d…`）；`prereg.sha256=79294a44…` / 4,431 B；`feature_dim=106`、`action_space{18 候选 × 2 放雷位 = 36}`、`behaviour=S stochastic / torch.Generator(seed*1000+7)`、`frozen{TRAIN_WINDOWS 13–16, DEV 11/12/17/18, H=120, horizons 120/240/600, 200 epoch, lr 1e-3, batch 256}`、`references{v830,v860,rule_r5,v815_critical}`。**`pins.self` 记执行器 `28fa6731…` / 175,189 B —— 与盘上现值不同**（§26.5 第 2 条） |
| 2 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/nobomb_ckpt_assertions.json` | 4,228 | `f64a10fff023db2f39c469d72c69a347340df8781650853e3d916872073cb754` | `phase=start`（`14:28:53`）：引擎 `691bfd77…` match=true；行为策略 `round2.pt` **3/3 match**（`4abee069…` / `a93746bb…` / `c69960fc…`，各 34,387 B）；12 个 `protected` 源文件只记了 `sha256` 与 `exists`、**没有 `expected`**（`match` 为 `null`，§26.5 第 3 条）；`all_pass=true`；`prereg.sha256=79294a44…`。**字段名是 `v8h0_conserve_py`，但路径指向 `tools/rl/v8i0_nobomb.py`，值 `28fa6731…` / 175,189 B**（§26.5 第 2 条） |
| 3 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/nobomb_ckpt_assertions_final.json` | 4,226 | `d06f740e6d89f52f631114af3f25b294c85e8e3b6aeaa48f6e19dae82deb2dd5` | `phase=final`（`14:44:02`）：引擎与 3 个 `round2.pt` 同上 **3/3 match**、`all_pass=true`、`prereg` 与 start 相**相同**；**字段名改成了 `v8i0_nobomb_py`，值 `605d332f…` / 177,079 B = 盘上现值** ⇒ 执行器在轮内被改过（§26.5 第 2 条）。12 个 `protected` 里 **`tools/rl/v8g0_deathadj.py` 由 start 的 `74dc5a26…` 变成 `5b52c96a…`**（§26.5 第 3 条） |
| 4 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/nobomb_fidelity_d3.json` | 26,700 | `ac1e2ad7e7faf468c13351727fe87a9ea20fdc0269f7b92ec1c16c9fd994b5c0` | D3 保真门 **5/5 PASS**（`all_pass=true`、`error=null`）：① HELLO `role=shadow`；② DEBUG `threads=1`（PARK 前后）；③ 前缀 **2,165 / 2,165** checkpoint 到帧 **8663**、`hash_mismatches=0`、`first_mismatch=null`（reset hash `83d95375` = 冻结值、final step hash `478b642e`）；④ R24 **18/18** 常量；⑤ PARK `frame=8663`；PARK 后 `advance` 帧 8664 的 hash `3ee41231` 与冻结值 **match**；`verdict="PASS: §2.1-§2.5 hold on this binary and data dir"`；`launch_env{SDL_AUDIODRIVER: dummy, DISPLAY: null, WAYLAND_DISPLAY: null}`；`wall_s=1.827`、`frame_ms_mean=0.2109`。**非 5/5 即整轮作废 —— 本轮未触发 `S-fork`** |
| 5 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/nobomb_audio_audit.json` | 2,731 | `cb42f2495fa45772c09e696b050635d2d3e1490bcec82ffe5e711d26a3d76c20` | AGENTS.md §9.3 主审计（`14:43:29`，`scope=…/.tmp_v8i0_nobomb/data`）：`cfg_files=13 / decoded=13 / **muted=13** / **audible=0** / errors=0`，`audible=[]`、`by_root` 13 个目录；共享审计件 `v4.3.1/evidence/R14/audio_cfg_audit.json` **未触碰**。**它只覆盖 13 个 data dir（12 个采集格 + 1 个首轮作废评测残留），不含 15:0x 之后才建的 72 个评测窗**（§26.5 第 11 条） |
| 6 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/nobomb_process_residue.json` | 751 | `812d95e52688e9cf5ba011a2162d43f2b89c32af4c11b7b89dc83b277e576eff` | 本轮引擎残留审计（`15:38:50`）：`engine_name=th08-modern`、`data_dir_filter=/mnt/d/WorkShop/Touhou_RL/.tmp_v8i0_nobomb`、`processes_total_by_engine_name=**66**`、**`matching=0`**、`lines=[]`、`pgrep_returncode=0`、`all_pass=true` ⇒ 本轮**没有**留下引擎进程，机器上那 66 台他轮引擎**只计数、未 `pkill`**。与 §25.5 第 8 条不同：**本件有 `data_dir_filter`** |

#### 26.1.4 死亡邻域采集记录、标签集与召回对照（15 条 / 19,833,031 B）

12 个 `collect/nobomb_seed{S}_w{W}.json` = **3 种子 × 4 dev 窗**，每格 **50 状态 / 150 行 / `labels_total=1,800`**（12 格合计 600 状态 / 1,800 行 / `labels_total` 求和 21,600），`status=complete`、`invalid_episodes=0`、`no_death_episodes=0`、`replay_mismatches=0`、`gate.all_pass=true`、`cfg` 四字段全 0（12/12）。逐格 `bytes` 与 `sha256`（前 12 位；完整 64 位在 `ASSET_SYNC.sha256`）如下，合计 **143,232 B**：

| 格（种子 / 窗） | 字节 | SHA-256（前 12 位） | 状态 | 死亡 episode | 死时带雷 | 标签条数 |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| seed0 / w11 | 11,849 | `c106713f450a…` | 50 | 3 | `{3: 3}` | 1,800 |
| seed0 / w12 | 12,028 | `a1960e563300…` | 50 | 3 | `{3: 3}` | 1,800 |
| seed0 / w17 | 12,006 | `cc9de667c7c0…` | 50 | 3 | `{3: 3}` | 1,800 |
| seed0 / w18 | 11,894 | `7307e79a9daa…` | 50 | 3 | `{3: 3}` | 1,800 |
| seed1 / w11 | 11,893 | `36333d257a12…` | 50 | 3 | `{3: 3}` | 1,800 |
| seed1 / w12 | 11,994 | `1237c8e29322…` | 50 | 3 | `{3: 3}` | 1,800 |
| seed1 / w17 | 11,856 | `c7febc25f318…` | 50 | 3 | `{3: 3}` | 1,800 |
| seed1 / w18 | 11,893 | `bfa1236dca04…` | 50 | 3 | `{3: 3}` | 1,800 |
| seed2 / w11 | 11,839 | `e1eba3ecf061…` | 50 | 3 | `{3: 3}` | 1,800 |
| seed2 / w12 | 12,012 | `350d2106ffe4…` | 50 | 3 | `{3: 3}` | 1,800 |
| seed2 / w17 | 12,028 | `6498e1d3d0f6…` | 50 | 3 | `{3: 3}` | 1,800 |
| seed2 / w18 | 11,940 | `7cf9a5ae8ad3…` | 50 | 3 | `{3: 3}` | 1,800 |

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 13 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/labels_nobomb.jsonl` | 19,628,830 | `df1d2e944e42b6047c79d1a07cce5c8d419c52d103a6b96838b7870c23b31dc2` | **池化标签集**（`14:40:29`）：**1,800 行** = 12 格 × 150 行（600 状态 × 3 个 H），每行 36 路 `alive_frames` + 元数据。其 sha **与 `train_*/nobomb_train_summary.json` 里记录的 `labels_file.sha256`、`nobomb_stats.json` 的 `labels_file` 逐位相同** ⇒ 训练/统计用的是同一份字节 |
| 14 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/nobomb_recall_v816.json` | 2,318 | `e5c40cbccce7201a26267299b3babf637d56ad579157bcb1fbee5827a0ea2ee8` | **归因对照**（`14:28:54`，`rows=2,160` / 720 状态，源 `.tmp_v8g0_deathadj/labels_deathadj.jsonl` `2bffac3b…` / 23,745,483 B）：把 v8.16.0（R5 分阶段 80/60/40 px）的死亡邻域**重算一遍 P0a** ⇒ 三个 H 的 `bombs_remaining_counts` 都是 **`{0:720}`、`armed=0`、`p0a_met=false`**、`critical` 59/63/597。⇒ 本轮的 **P0a=100 %** 只能归因于**放雷规则**，不是采样口径 |
| 15 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/nobomb_stats.json` | 58,651 | `a4eae4ae859ca66830f0b4af8428ad9b97e66c1a0cae37b32a742d92fd56c7dc` | 12 个采集格的**合并统计**（`14:40:29`）：`states=600`、`labels=21,600`、`death_episodes=36`、`no_death_episodes=0`、`by_H` 三个 H 的 P0a/P0b、`cell_count=12`、`label_checks` 池化 = `labels_36 600 / bomb1_press_honoured **10,584** / bomb1_noop_due_to_invuln 216 / bomb1_anomalies 0 / bomb0_spent_a_bomb 0 / states_without_a_bomb_left 0 / death_episodes 36 / no_death_episodes 0`。**合并层没有 `bomb1_spent_two_or_more`**（12 个 cell 内合计 10,584 正确，§26.5 第 8 条） |

#### 26.1.5 训练记录、6 个 checkpoint 与训练汇总（13 条 / 301,538 B）

6 个 `ckpt_{A,B}_nobomb_seed{0,1,2}.pt` 各 **39,059 B**（`Policy(106,64,36)`，9,188 参数），其 sha256 与 `summary.json` 的 `train.arms[*][*].ckpt_sha256` **6/6 全等**、且被 18 份评测 `.summary.json` 的 `ckpt.sha256` 引用 **18/18 全等**：

| checkpoint | 字节 | SHA-256（前 12 位） | 训练记录 | 训练记录字节 | 训练记录 SHA-256（前 12 位） |
| --- | ---: | --- | --- | ---: | --- |
| `ckpt_A_nobomb_seed0.pt` | 39,059 | `58e7f8f50a52…` | `train_A_nobomb_seed0.json` | 11,697 | `8e4b58e702eb…` |
| `ckpt_A_nobomb_seed1.pt` | 39,059 | `d97f3ec83558…` | `train_A_nobomb_seed1.json` | 11,680 | `885631669214…` |
| `ckpt_A_nobomb_seed2.pt` | 39,059 | `9765a360d398…` | `train_A_nobomb_seed2.json` | 11,689 | `69f89cf0114e…` |
| `ckpt_B_nobomb_seed0.pt` | 39,059 | `522b1a46486f…` | `train_B_nobomb_seed0.json` | 9,047 | `6c9828d41ece…` |
| `ckpt_B_nobomb_seed1.pt` | 39,059 | `8f9746ef4af9…` | `train_B_nobomb_seed1.json` | 9,053 | `ea597b4f5145…` |
| `ckpt_B_nobomb_seed2.pt` | 39,059 | `b44622eb4880…` | `train_B_nobomb_seed2.json` | 9,049 | `5a3a7266166e…` |

6 份 `train_*.json` 的读数（`states=600`、`labels=21,600`、200 epoch、Adam 1e-3 / batch 256、`deviations=[]`）：A 臂（`BCE(36) + 0.5 × BCE_bomb(best-bomb)`）final BCE **0.060321 / 0.059073 / 0.059985**、`aux_target_positive_states=588`、wall 141.351 / 141.598 / 141.633 s；B 臂（仅 `BCE(36)`）final BCE **0.048950 / 0.048945 / 0.048936**、wall 115.541 / 118.003 / 120.022 s；两臂 `train_accuracy` 均为 **0.99**。

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 13 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/nobomb_train_summary.json` | 4,969 | `27bfcd5aa4b7a9144a136b5f8da2657e9493700b5278af9cc8e08af1a6c923fc` | A/B 六格的**训练汇总出口**（`14:43:11`）：逐臂逐种子的 `{states, loss_first_epoch, loss_last_epoch, final_train_bce, final_train_aux_bce, train_accuracy, wall_s, ckpt}`；A 臂 `final_train_aux_bce` 0.135638 / 0.138113 / 0.136168，B 臂为 `null` |

#### 26.1.6 评测行数据与逐格摘要（36 条 / 399,280,730 B / **21,600 行 episode**）

18 格 = **2 臂（A / B）× 3 预算（300 / 600 / 900）× 3 种子**，每格 **1,200 episode**（4 dev 窗 × 300）、每格一个 `.jsonl`（**实测 18/18 都是 1,200 行**，合计 **21,600 行**）+ 同名 `.summary.json`。下面的表把**每格的 jsonl 与 summary 合成一行**（`sha256` 取前 12 位；完整 64 位在 `ASSET_SYNC.sha256`；83 条内无前缀碰撞）。

| 格（臂 / 预算 / 种子） | jsonl 字节 | jsonl SHA-256（前 12 位） | summary 字节 | summary SHA-256（前 12 位） | 活满 / 1,200 | 活满率 |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| A / 300 / seed0 | 16,151,439 | `6f0de55d575e…` | 6,461 | `db5ec458dcb2…` | 1,177 | **0.980833** |
| A / 300 / seed1 | 16,169,835 | `579ccd9c6612…` | 6,482 | `6afae82a0bcb…` | 1,186 | **0.988333** |
| A / 300 / seed2 | 16,166,118 | `f1631754b833…` | 6,481 | `273ed9a1dd7f…` | 1,186 | **0.988333** |
| A / 600 / seed0 | 24,684,512 | `55e91e2f0855…` | 6,501 | `1cb8c205ed16…` | 207 | **0.172500** |
| A / 600 / seed1 | 24,789,325 | `b1696246792f…` | 6,517 | `1d1637c52a38…` | 197 | **0.164167** |
| A / 600 / seed2 | 24,523,983 | `98e37b4d8758…` | 6,521 | `e2ad7bd55089…` | 212 | **0.176667** |
| A / 900 / seed0 | 26,044,646 | `82f8e43316c9…` | 6,512 | `e3a2d28c96d1…` | 26 | **0.021667** |
| A / 900 / seed1 | 25,581,418 | `20dcf1b9d423…` | 6,517 | `075d2f71ab92…` | 18 | **0.015000** |
| A / 900 / seed2 | 25,630,052 | `9f63103dac3b…` | 6,525 | `3faa4afb6b28…` | 27 | **0.022500** |
| B / 300 / seed0 | 16,217,141 | `3e1b57067492…` | 6,377 | `f21256d87e71…` | 1,173 | **0.977500** |
| B / 300 / seed1 | 16,229,086 | `18ec9e1c3bc5…` | 6,373 | `e399e5d52933…` | 1,185 | **0.987500** |
| B / 300 / seed2 | 16,223,324 | `64b61e82b2e7…` | 6,368 | `ad278479d73b…` | 1,182 | **0.985000** |
| B / 600 / seed0 | 24,760,988 | `db44eaa19daf…` | 6,408 | `26f2deeab172…` | 217 | **0.180833** |
| B / 600 / seed1 | 24,807,084 | `495b72b24937…` | 6,409 | `d60f9b44a588…` | 231 | **0.192500** |
| B / 600 / seed2 | 24,644,109 | `b4627c8212d4…` | 6,404 | `cd3bac0c475a…` | 218 | **0.181667** |
| B / 900 / seed0 | 25,385,480 | `5efc99614cf9…` | 6,398 | `23fae4831dec…` | 27 | **0.022500** |
| B / 900 / seed1 | 25,557,288 | `49d7309b327b…` | 6,412 | `f4812be20822…` | 26 | **0.021667** |
| B / 900 / seed2 | 25,598,831 | `42635ee7f254…` | 6,405 | `3a1973f3ab0d…` | 22 | **0.018333** |
| **合计** | **399,164,659** | — | **116,071** | — | **8,517 / 21,600** | — |

**逐格读数与自检（18/18 全过）**：每份 `.summary.json` 都是 `status=complete`、`error=null`、`episodes_total=1,200`、`bombs_used_total=3,600`（每局都把 3 颗雷花光）；每格的 4 个窗口都带 `cfg_muted=true`、`gate{hello_role_shadow, park_ok, park_frame=8663, threads_1, all_pass}=true`、`gate_prefix{hash_checks=2165, hash_mismatches=0, final_frame=8663}` ⇒ **72/72 个窗口**全过；`engine.match=true` **18/18**，`launch_env{SDL_AUDIODRIVER: dummy, DISPLAY: null, WAYLAND_DISPLAY: null}` **18/18**。逐臂逐预算的池化读数：**A** d300 **3,549/3,600 = 98.5833 %**、d600 **616/3,600 = 17.1111 %**、d900 **71/3,600 = 1.9722 %**；**B** d300 **3,540/3,600 = 98.3333 %**、d600 **666/3,600 = 18.5000 %**、d900 **75/3,600 = 2.0833 %**。

#### 26.1.7 统计与诊断出口（4 条 / 73,193 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/summary.json` | 16,838 | `e87e4b72153c38614dc943e5a513f65e68f4942086a46f5c29f87447218ca64e` | **主判据出口**（`14:40:29`，`schema=v8170-nobomb-summary/1`）：`verdict` = **`S-weak`**（P0a 100 %、P0b 98.0 % 成立；P1 不成立、P3 不成立）；`p0.by_H{120,240,600}` 各 `states 600 / armed 600 / diff 588`、`critical_context` 12/12/42、`delta_alive_frames` mean 96.07 / 213.67 / 565.27；`p0.bombs_at_death={3:36}`、`states_by_player_state={DYING:600}`；`mechanism{bomb1_presses_honoured 10,584, bomb1_noop_due_to_invuln 216, bomb1_anomalies 0}`；`train{…}`；`eval{18 格、21,600 episode}`；`baselines{v830/v860/r5}`；`artifacts{6 条}`；`deviations` **5 条** |
| 2 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/nobomb_summary.json` | 16,838 | `e87e4b72153c38614dc943e5a513f65e68f4942086a46f5c29f87447218ca64e` | **与第 1 条逐字节相同**（sha 相同、字节相同，mtime 早 26 ms）—— 同一份统计用两个名字各放一份，见 §26.5 第 5 条。**照登** |
| 3 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/nobomb_diagnostics.json` | 35,216 | `97591e7f5fc6f916bd6123fb8e84e11974afe246f915c932acff620ce7bffa0d` | 逐臂 / 逐预算 / 逐种子的诊断汇总（`15:38:35`，`schema=v8170-conserve-diagnostics/1`）：`arms{A,B}{d300,d600,d900}.per_seed{0,1,2}` 的 `{episodes, survived, survived_fraction, alive_frames 分位, bombs_used}`、`bomb_usage`、`references{v830,v860,rule_r5}`、`side_by_side_survived_fraction{v830: .999444/.194444/.006111, v860: .999444/.295833/.029444, rule_r5: 1.0/.7711/.1761, arm_A, arm_B}`。`requirement` 明写「**本文件不含检验、不含区间**：P1–P4 是计数与分布，检验与置信区间属于独立分析轮」 |
| 4 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/nobomb_bomb_usage.json` | 4,301 | `584c891117babaa3e9ce2554bda767316f7deb15211f72b582087c42cbfb76ad` | **P3 专项读数**（`15:38:45`，`schema=v8170-conserve-bomb-usage/1`）：6 格（A/B × 300/600/900）每格 3,600 episode，`bombs_used_counts` 全部是 **`{3: 3600}`**、`episodes_using_1_or_2=0`、`episodes_using_0=0`、`episodes_using_3=3,600`、分位 mean/median/q1/q3 全 3.0；`bomb_emits_total` 1,074,227 / 1,708,143 / 1,787,827（A）与 1,079,131 / 1,713,933 / 1,770,687（B）⇒ **P3 不成立：没有任何一局只用 1–2 颗** |

#### 26.1.8 清单锚（发布清单 + 运行目录清单，2 条 / 312,989 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/manifest.json` | 11,818 | `dea08a53cc7ab4c9154aa85f10b65253af26f5d39329e1ff2a27698fc888e92d` | **发布清单锚**：`schema=v8170-conserve-manifest/1`、`file_count=80`、`total_bytes=419,895,193`、`excludes="manifest.json"`、`root=…/v8.17.0/evidence/CONSERVE`、`created_at=2026-11-22T15:42:01+0800`；`files{}` **80** 条 `{sha256, bytes}` —— 本节实测 **80/80 全等、0 处不符**，`bytes` 求和 = `total_bytes` = 419,895,193（§26.4.2）；**mtime `15:42:02.326` 是目录内最后一个写入**（晚于 `nobomb_summary.json` 15:41:53.258 / `RUN.md` 15:41:53.306 / `nobomb_RUN.md` 15:41:53.329 / `nobomb_make_summary.py` 15:41:53.363）⇒ **没有**「manifest 是旧快照」的问题 |
| 2 | `Document/Update/Alpha/v8.17.0/evidence/CONSERVE/nobomb_work_manifest.json` | 301,171 | `6870614cd8d2543b3bc2cb58eae816e8a41b941a91425914b23594fb2de95135` | **运行目录清单**（`15:40:35`，同一 schema）：`root=…/.tmp_v8i0_nobomb`、`file_count=**1,862**`、`total_bytes=**45,664,203,257**`、`excludes="work_manifest.json"`，`files{}` 1,862 条的 `bytes` 求和 = `total_bytes`（内部自洽）。**它是运行现场清单，不是发布目录清单**（两者 root 不同、覆盖集不同，不构成冲突，§26.5 第 6 条）；运行目录本体按 §26.2 不入册 |

#### 26.1.9 执行器（1 条 / 177,079 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `tools/rl/v8i0_nobomb.py` | 177,079 | `605d332f7989a6856b49f99fc3c11c334ff7c9276a509450d8dd3103aa7a264c` | v8.17.0 执行器（`--mode` 12 个取值：preflight / assert-ckpt / recall-v816 / collect / stats / train / eval / audio-audit / process-audit / diagnostics / summary / manifest）。`RUN.md` §0 记：**由 `tools/rl/v8h0_conserve.py` 复制生成**（`v8h0_conserve.py` 本身一字节未改），逐条替换清单在运行目录 `make_v8i0.py`（每条替换都断言「在源文件里恰好出现一次」）。盘上现值 **`605d332f…` / 177,079 B**（mtime `14:31:47.408`）= `nobomb_ckpt_assertions_final.json` 的 `v8i0_nobomb_py`；但 `nobomb_preflight.json` 的 `pins.self` 与 `nobomb_ckpt_assertions.json`（start 相）记的是 **`28fa6731…` / 175,189 B** ⇒ **轮内被改过**，旧状态盘上已不存在（§26.5 第 2 条）。**两份 manifest 都不含执行器**（对两份 `files{}` 做 `v8i0_` 匹配 = 0 命中），其字节身份只由本条 + 两份 `ckpt_assertions` 提供 |

### 26.2 跳过说明（未入册 **0** 个文件；`.tmp_v8i0_nobomb/` 运行件不入册）

实测：v8.17.0 目录 **82** 个文件 / **419,911,442 B**，执行器 **1** 个 / **177,079 B** ⇒ **83 个文件 / 420,088,521 B，入册 83、未入册 0**。

| 组 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `Document/Update/Alpha/v8.17.0/**` | 82 | 419,911,442 | **全部入册**（含 `manifest.json` 自身、`nobomb_work_manifest.json`、2 个脚本、6 个 `.pt`、12 个采集格、36 个评测件）；未入册 **0** |
| `tools/rl/v8i0_nobomb.py` | 1 | 177,079 | **全部入册**；未入册 **0** |
| **小计** | **83** | **420,088,521** | — |

#### 26.2.1 `.tmp_v8i0_nobomb/` 是 gitignored 的运行件（不入册）

`.gitignore` 有一条 **`.tmp_*/`**，`.tmp_v8i0_nobomb/` 整棵树因此**不进 Git**，是过程现场而非发布件；其**唯一需要字节身份的部分（采集格记录、池化标签、训练记录与 ckpt、18 格评测行数据与逐格摘要、统计/诊断/门禁出口、两份清单）已全部发布到 `Document/Update/Alpha/v8.17.0/` 并被本节 82 条完全覆盖**。运行目录里剩下的 `data/`（私有 cfg 与软链的 `th08.dat`）、`work/`、`logs/`、`runs/`、`smoke_work/`、`eval/` 下与发布件同名的 jsonl、以及 8 个未发布的辅助脚本（`make_v8i0.py`、`eval_table.py`、`inspect_smoke.py`、`peek.py`、`progress.sh`、`show_stats.py`、`stop_eval.sh`、`finish.sh`）按 §21.2 / §22.2 / §23.2 / §24.2 / §25.2.2 同口径**不单独入册**。

两个口径只报数、不登记：**Windows 侧递归 = 1,868 个文件条目 / 455,844,889 B**，其中 **182** 个是 `*.dat`（`th08.dat` / `thbgm.dat` 各 91 份，逐 data dir 一份）在 Windows 侧**不可解析**（`os.path.getsize` 报 `[WinError 1920]`，例如 `data/eval-20261122-144319/w11/th08.dat`），其尺寸在 Windows 口径里被计为 **0**；**运行目录自带的 `nobomb_work_manifest.json` = 1,862 条 / 45,664,203,257 B**（WSL 侧把软链目标尺寸算进去，故远大于 Windows 侧读数）。两个口径的 **6 条差额也不是缺件**：work manifest 建于 `15:40:35`，此后运行目录又写了 6 个发布收尾文件（`finish.sh`、`make_summary.py`、`publish.json`、`summary.json`、`work_manifest.json`、`logs/publish.out`）⇒ 「盘上有而 manifest 无」= **6**、「manifest 有而盘上无」= **0**。**差异全部来自「跨 Windows/WSL 的软链尺寸口径」与「清单是 15:40 的快照」，不是缺件**。

#### 26.2.2 `__pycache__` / `*.pyc` / 隐藏文件实况（按要求只报告、不删）

`v8.17.0/` 全树只有 **3** 个目录（`evidence/`、`evidence/CONSERVE/`、`evidence/CONSERVE/collect/`）、**0** 个 `__pycache__/`、**0** 个 `*.pyc`、**0** 个隐藏文件（`Get-ChildItem -Force` 与不加 `-Force` 两种口径都是 **82 个文件 / 419,911,442 B**）。目录内的两个脚本（`nobomb_exec.sh`、`nobomb_make_summary.py`）是**本轮按证据发布的脚本本体**，不是运行残留，已入册。若日后出现 `__pycache__/` 或 `*.pyc`，按 §21.2 口径**一律不入册**。

#### 26.2.3 明确**不登记** `v8.18.0/`

按工单：`Document/Update/Alpha/v8.18.0/` **本轮不登记、不触碰**，**下一轮单独登记**。实测它仍然只有 **1 个**文件 —— `PREREGISTRATION_EARLYBOMB_001.md`（**4,301 B**，`14:49:14`）—— 在 `ASSET_SYNC.sha256` 里 **0 行**、在本册里**没有任何路径/字节/sha 登记**。本节对 `v8.18.0/` 的写操作 = **0**（只做了一次 `os.walk` 计数）。

### 26.3 与 §1–§25 不重复的说明（实测已核）

- **既有校验表 0 命中**：追加前对 `ASSET_SYNC.sha256`（**626 行**）做 `v8\.17\.0|CONSERVE|v8i0_|nobomb` 匹配 → **0 命中**（既没有 v8.17.0 的路径，也没有执行器）。
- **新增 83 条路径与既有表 0 交集**：83 个**路径**与既有 **626 个路径**逐一比对 → **路径交集 0**。
- **哈希 0 交集**：83 条里有 **81 个唯一 sha256**（两条重复见下），与既有 **611 个唯一哈希**逐一比对 → **哈希交集 0** ⇒ 没有把任何既有行「换个路径再钉一遍」。
- **本册正文的 10 处提及都不是登记**：`v8.17.0` 在本册出现 **10 次**、`v8i0_` **5 次**、`CONSERVE` **2 次**、`nobomb` **6 次**，全部集中在 §24（`v8.16.0` 的「采集期间机器上有 105 台 v8i0_nobomb 他轮引擎」）与 §25 的 §25.2.4 / §25.4.5 第 10 条 / §25.5 第 10 条 —— 后三处**都明写「本轮不登记、下一轮单独登记」**，且**只给了目录口径的数字（82 / 419,911,442 B），没有给任何单文件的路径、字节或 sha256**。本节首次给出逐文件字节身份 ⇒ **不重复**。
- **两条内容重复（83 条 → 81 个唯一哈希）**：`RUN.md` 与 `nobomb_RUN.md` 逐字节相同（`fa0282fb…`）、`summary.json` 与 `nobomb_summary.json` 逐字节相同（`e87e4b72…`）。**照登**（4 条各自独立路径），但本节不把它们当两件不同的证据引用（§26.5 第 5 条）。
- **行数与格式**：626 + 83 = **709 行**（`ASSET_SYNC.sha256` 实测 **709 行**）；仍为 **LF / 无 BOM**、末行以 `\n` 结束、每行 `64 hex + 两个空格 + 相对路径（`/` 分隔、仓库根相对）`，**空行 0 条、格式异常行 0 条**、**CR 字节 0**。
- **`ASSET_SYNC.md` 本身不在校验表内**（`ASSET_SYNC` 在 `ASSET_SYNC.sha256` 里 **0 行**）⇒ 追加 §26 **不会**让任何既有行转 FAILED（与 §21.3 / §22.3 / §23.3 / §24.3 / §25.3 同一口径）。
- **Git 跟踪状态**：本轮按硬约束**未** `git add` / `git commit`（没有执行任何 git 写命令），分支仍为 `v400`；v8.17.0 证据与执行器都**未入库** ⇒ 这些证据**无法由 Git 重建**，本节 + `ASSET_SYNC.sha256` 是它们目前的字节身份记录。

### 26.4 校验结论

#### 26.4.1 `sha256sum -c`（WSL，追加前后各一次）

命令（与 §22.4.1 / §23.4.1 / §24.4.1 / §25.4.1 同一口径）：

```bash
cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -c ': OK'; sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -v ': OK'
```

* **基线（追加前 626 行）**：**625 OK / 1 FAILED**。
* **追加后（709 行）**：**708 OK / 1 FAILED**。`708 + 1 = 709` 校验行全部参与。
* **新增 83 条 83/83 OK** ⇒ 整体 FAILED 数**没有新增（仍为 1）**。
* **唯一 FAILED 仍是已知并允许的那一条**：`Document/Update/Alpha/v4.2.0/journal.jsonl`（§9.8 记录的活动文件漂移）。本轮**未改该行、未重钉、未触碰该文件**，三个校验时刻（追加前 / 追加后 / 写完 §26 后）该行是**同一条 FAILED**。
* **格式**：709 行仍为 **LF / 无 BOM**、末行 `\n` 结束、每行 `64 hex + 两个空格 + 相对路径`；空行 **0**、格式异常 **0**；新增 83 条的路径在表内**各自唯一**。

#### 26.4.2 与 `manifest.json` 的交叉核对（登记时实测）

| 清单 | 自述 | 实测比对 |
| --- | --- | --- |
| v8.17.0 `evidence/CONSERVE/manifest.json` | `file_count=80`、`total_bytes=419,895,193`、`excludes="manifest.json"` | `files{}` **80/80** 的 `{bytes, sha256}` **全等、0 处不符**；`bytes` 求和 = `total_bytes` = 419,895,193；盘上 `evidence/CONSERVE/**` 共 **81** 件，manifest **未列**的**只有 `manifest.json` 自身**（11,818 B）⇒ **无缺件、无多余件**（419,907,011 − 419,895,193 = 11,818） |
| 18 份 `eval_*.summary.json` 的 `out_sha256` / `out_bytes` | 每格 `{status, episodes_total, out_sha256, out_bytes}` | **18/18** 与盘上 jsonl 的实测 `{sha256, bytes}` 全等、`status` 全为 `complete`、`episodes_total` 全为 1,200 ⇒ 逐格摘要**指向的就是发布件字节** |
| 发布件 vs 运行目录同名件（`.tmp_v8i0_nobomb/eval/<臂>/`） | 18 个 jsonl + 18 个 summary | **36/36 逐字节相同**（sha 与字节都相同）⇒ 「发布件 = 运行目录件」的第二锚（与 §25.4.2 的 `publish.json` 角色相同） |
| `labels_nobomb.jsonl` 的 sha | 被 `train_*/nobomb_train_summary.json`、`nobomb_stats.json`、`summary.json` 记录 | **1/1 全等**（`df1d2e94…` / 19,628,830 B / 1,800 行） |
| 6 个 `ckpt_*.pt` 的 sha | 被 `summary.train.arms[*][*].ckpt_sha256` 与 18 份评测 summary 的 `ckpt.sha256` 引用 | **6/6 + 18/18 全等** |
| 预注册 sha | `nobomb_preflight.pins.prereg`、`nobomb_ckpt_assertions{,_final}.prereg`、盘上 | **三处同值** `79294a44…` / 4,431 B ⇒ 预注册在全轮前后**一字未改** |

**两份清单都未覆盖的内容（如实列出）**：`manifest.json` **不含** `tools/rl/v8i0_nobomb.py`（对 `files{}` 做 `v8i0_` 匹配 = **0 命中**）⇒ 执行器的字节身份由 **§26.1.9**（实测）+ 两份 `ckpt_assertions` 的 `v8h0_conserve_py` / `v8i0_nobomb_py` 提供（两者不同值，见 §26.5 第 2 条）。

#### 26.4.3 身份链、门与读数（只登记，不重判）

| 核对项 | 实测 |
| --- | --- |
| 引擎身份 | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` / match=true，出现在 `nobomb_preflight.json`、`nobomb_fidelity_d3.json`、`nobomb_ckpt_assertions{,_final}.json`、**18/18** 份评测 summary 与 **12/12** 个采集格；与 §2 / §22 / §23 / §24 / §25 记录的 `rl_v3` SHA **一致** |
| 行为策略身份 | v4.5.0 `S` 的 `round2.pt` = `4abee069…` / `a93746bb…` / `c69960fc…`（各 34,387 B），start 与 final 两相 **各 3/3 match** ⇒ 600 个状态的采样策略没换。本轮的 A/B 学生 ckpt 各 39,059 B，**6/6** 与 summary 记录一致 |
| **预注册身份** | `79294a44c5d3a066…` / 4,431 B —— 在 `nobomb_preflight.json`、`nobomb_ckpt_assertions{,_final}.json` **两相**、以及盘上实测**三处同值** ⇒ 本轮是「预注册先冻结、再开跑」 |
| D3 保真门 | **5/5 PASS**（`all_pass=true`）：HELLO `role=shadow`、DEBUG `threads=1`、前缀 **2,165 / 2,165** checkpoint **0 mismatch**（到帧 8663）、R24 **18/18** 常量、PARK `frame=8663`；PARK 后帧 8664 的 hash `3ee41231` 与冻结值 match。**非 5/5 即整轮作废 —— 本轮未触发 `S-fork`** |
| 逐格门 | 18 格 × 4 窗 = **72/72** 个窗口 `gate.all_pass=true` + `gate_prefix{hash_checks 2,165, hash_mismatches 0, final_frame 8663}`；12 个采集格 **12/12** `gate.all_pass=true` |
| 音频（§9 硬约束） | 审计件 **13 / 13 muted / 0 audible / 0 errors**；另 72 个评测窗 `cfg_muted=true`（同一模板 `cc062cfb…` / 60 B）、12 个采集格 cfg 四字段全 0 ⇒ **去重后 85 个独立 data dir 全静音、`audible` = 0**；全部 `launch_env{SDL_AUDIODRIVER=dummy, DISPLAY=null, WAYLAND_DISPLAY=null}`；共享审计件 `v4.3.1/evidence/R14/audio_cfg_audit.json` **未触碰** ⇒ 本轮**未新增 `audible`** |
| 进程残留 | `matching=0` / `lines=[]` / `pgrep_returncode=0` / `processes_total_by_engine_name=66` / `data_dir_filter=…/.tmp_v8i0_nobomb` / `all_pass=true` ⇒ 本轮**没有**留下引擎进程，也**没有** `pkill` 任何他轮进程（只计数） |
| P0a / P0b 读数 | 三个 H 各 **600/600 = 100 %** 带雷（`bombs_at_death={3:36}`、`states_by_player_state={DYING:600}`）、**588/600 = 98.0 %** 出现 diff；`delta_alive_frames` mean 96.07 / 213.67 / 565.27（H=120/240/600）⇒ 判据 `P0a`、`P0b` **PASS** |
| 结论口径 | `summary.verdict` = **`S-weak`**，与 `RUN.md` §9 的判定表**一致**：P0a/P0b PASS、**P1 不成立**（B d600 18.500 % ≤ 19.444 %）、**P3 不成立**（0/3,600 只用 1–2 颗）；900 档 **A 1.9722 % / B 2.0833 %** 对不放雷的 v8.3.0 **0.6111 %** 是 **3 倍**，但仍低于 v8.6.0 的 **2.9444 %** |
| 「不能声称什么」照预注册保留 | `RUN.md` §11 明写：不能声称「学会了资源分配」、不能把 12 格当独立样本、不能把不同预算的均值直接比较、不能把 `critical` 当作「雷没价值」的证据 ⇒ 本节**照录不重判** |

#### 26.4.4 登记期零漂移（v8.17.0 目录与执行器只读）

登记**开始前**、**等待窗口结束时**、**83 条 sha256 全部算完之后**各采集一次这 **83** 个文件的 `{size, mtime_ns}`（PowerShell `LastWriteTimeUtc.Ticks`，100 ns 分辨率）并逐项比对：

| 快照 | 时刻 | 条目 | 比对结果 |
| --- | --- | ---: | --- |
| S1 | `15:49:00.837` | 83 | 基线 |
| S2 | `15:51:10.926`（S1 + **130 s**） | 83 | **S1 == S2，83/83 相同** ⇒ `QUIESCENT=YES` |
| S3 | `15:51:11.781`（83 条 sha256 全部算完） | 83 | **S1 == S3，83/83 相同** ⇒ `AFTERHASH_DRIFT=NO` |

| 范围 | 文件数 | 差值 |
| --- | ---: | --- |
| `Document/Update/Alpha/v8.17.0/**` | 82 | `size` 0 变化 / `mtime_ns` 0 变化 |
| `tools/rl/v8i0_nobomb.py` | 1 | `size` 0 变化 / `mtime_ns` 0 变化 |
| **合计** | **83** | **83/83 完全相同，0 个变化** |

**「等 120 秒再钉」这一条不需要触发、但仍然执行了 130 s 复核。** 登记开始时目录内最新 mtime 是 `manifest.json` 的 **`15:42:02.326`**，距首次快照（`15:49:00.837`）已 **6 分 58 秒** > 5 分钟 ⇒ 不存在「最近 5 分钟内仍在变」的文件，**不需要等待**；为排除 `15:42` 之后仍有迟写，仍按要求等了 **130 s** 并比对 **S1 / S2 逐位相同**，此后才把 83 条数字与 sha256 钉入本节与 `ASSET_SYNC.sha256`。**本轮目录没有被重写**（对比 v8.20 那次「登记期被重新发布」，见 §25.4.4 / §25.5 第 10 条）。

本轮对上述路径的写操作 = **0**（只读 `Get-ChildItem` / `Get-FileHash` / `os.walk` / `hashlib` / `sha256sum` / `Get-Content` / `ConvertFrom-Json` / `json.load`）；**未新建、未改名、未删除、未移动**任何文件；写入**只落在** `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256` 两个清单文件上。**把两处清单写完之后的第四次实测**（`sha256sum -c` 全表通过之后）：目录仍是 **82 个文件 / 419,911,442 B**、目录内**最大 mtime 仍是 `manifest.json` 的 `15:42:02.326`**（即没有任何文件的 mtime 晚于发布完成时刻）、执行器仍是 **177,079 B / mtime `14:31:47.408`** ⇒ 登记全过程的 `{size, mtime}` **一次也没动**。同时确认 `v8.18.0/` **未被读写改写**（本轮未登记，只对它做过 `os.walk` 计数）。

#### 26.4.5 本章只登记字节身份与已在盘上的读数（**不重判**）

1. **不重算检验**：`nobomb_diagnostics.json` 的 `requirement` 明写「P1–P4 是计数与分布，**检验与置信区间属于独立分析轮**」，本轮发布件里**没有**独立的检验件（对照 v8.19.0 的 `within_sweep_contrasts.json`、v8.20.0 的 `summary.references` 并排）。本节因此**只照录** 18.500 % / 19.444 % / 2.0833 % 与 `S-weak`，**不重新做 Fisher、不给区间**。
2. **12 格不是独立样本**：`nobomb_diagnostics.json` 的 `note` 明写「12 个 (seed, window) 格不是独立样本；逐格数字列在池化数字旁边，**不得把它们平均成「均值的均值」**」。本节一律用**池化**口径（x/3,600、x/21,600）引用。
3. **雷的语义是 deathbomb**：`nobomb_stats.json` / `summary.json` 的 `mechanism` 记 600/600 状态都是 `DYING`、按雷=死亡炸弹**一次扣 2 颗**（`bomb1_presses_honoured=10,584`、`bomb1_noop_due_to_invuln=216`）。所以本轮的标签表达的是「**死那一刻手里那颗雷的价值**」，**不能外推**为「活着时提前放雷的价值」—— `RUN.md` §11 第 1 条已写明，本节照录。

### 26.5 与 manifest 不符项（**只报告、不改**）

> **总结论：`manifest.json` 与盘上 0 不符。** `files{}` **80/80** 的 `{bytes, sha256}` **逐条全等**，`bytes` 求和 = `total_bytes` = 419,895,193，未列出的文件**只有 `manifest.json` 自身**与**顶层预注册**（后者在 `root` 之外）。下面列的**不是** manifest 的错，而是「manifest 覆盖不到」或「manifest 之外的证据自相矛盾 / 口径不同」的 **11 项**，按工单**只报告、未改任何文件**。

| # | 项 | 事实（实测） | manifest 侧 | 处置 |
| ---: | --- | --- | --- | --- |
| 1 | **manifest 自述「除自身」且不含顶层预注册** | `evidence/CONSERVE/**` 目录 **81 / 419,907,011 B** vs 自述 **80 / 419,895,193**，差 **11,818 B = `manifest.json` 自身**；再加顶层预注册 4,431 B ⇒ 全目录 **82 / 419,911,442 B** | **不是不符**，口径已闭合 | 两个口径都写进 §26 起首表 |
| 2 | **执行器在轮内被改过** | `nobomb_preflight.json`（`14:28:52`，`pins.self`）与 `nobomb_ckpt_assertions.json`（start，`14:28:53`，字段名 `v8h0_conserve_py`）都记 **`28fa6731…` / 175,189 B**；`nobomb_ckpt_assertions_final.json`（final，`14:44:02`，字段名改成 `v8i0_nobomb_py`）记 **`605d332f…` / 177,079 B** = 盘上现值（mtime **`14:31:47.408`**，在 start 之后 **2 分 54 秒**）⇒ **轮内至少 2 个状态**，旧状态 `28fa6731…` **盘上已不存在**，无法用字节复核 | manifest **不含执行器**（0 命中），故**不冲突**；执行器身份由 §26.1.9 实测 + 两份 `ckpt_assertions` 提供 | **只报告、不改**；按盘上值 `605d332f…` 登记 |
| 3 | **`protected` 12 件没有 `expected`，`all_pass=true` 看不见它们变** | 两份 `ckpt_assertions` 的 `protected{}` 只有 `{path, exists, sha256}`，**没有 `expected`** ⇒ `match` 为 `null`、`all_pass` 对它**没有约束力**。实测：`tools/rl/v8g0_deathadj.py` 在 start 相是 **`74dc5a26…`**、在 final 相是 **`5b52c96a…`**（其 mtime **`14:29:00.796`** = start 之后 **7 秒**）⇒ **轮内确实有一个「受保护」文件被改动，而两相 `all_pass` 仍是 true**。盘上现值 `5b52c96a…` / 158,366 B = §24 已登记的值（即该改动发生在本轮登记之前、且已被 §24 固定） | manifest 不涉 | **只报告、不改** |
| 4 | **18 份 eval `.summary.json` 的 `created_at` 全是同一个 `14:57:44`** | 18/18 的 `created_at = 2026-11-22T14:57:44+0800`，而它们的 mtime 跨 **`15:20:18`–`15:35:14`**；`RUN.md` §10 第 1 条记 `14:57` 那批 18 格**全部作废**（`episodes_total=0`）⇒ 这个字段**不可能是发布件的写入时刻**。另外每份的 `out` 仍指向**运行目录**（`.tmp_v8i0_nobomb/eval/<臂>/…jsonl`），不是发布路径 | manifest 对 18 份 summary 的**字节身份记录正确**（18/18 与盘上全等） | **只报告、不改**；时间锚以 mtime 与 `nobomb_commands.tsv` 的 `started/finished` 为准 |
| 5 | **两组文件逐字节重复** | `RUN.md` == `nobomb_RUN.md`（`fa0282fb…` / 14,874 B）、`summary.json` == `nobomb_summary.json`（`e87e4b72…` / 16,838 B）⇒ 83 条里只有 **81 个唯一 sha256**。**同一份内容被两个文件名各放了一份**，不是缺件也不是清单错误 | manifest **把两条路径都登记**（各 1 条 `files{}`，sha 相同），**无冲突** | 照登 4 条；引用时只用其一（§26.3 已注明） |
| 6 | **`nobomb_work_manifest.json` 是「运行目录清单」而不是发布清单** | 其 `root=…/.tmp_v8i0_nobomb`、`file_count=1,862`、`total_bytes=45,664,203,257`、`excludes="work_manifest.json"`，`bytes` 求和 = `total_bytes`（内部自洽）；但它是 **`15:40:35` 的快照**，此后运行目录又写了 6 个发布收尾文件 ⇒ 盘上「有而清单无」= **6**、「清单有而盘上无」= **0** | 与 `manifest.json`（root 在 `evidence/CONSERVE`）**覆盖集不同**，两者**不冲突** | **只报告、不改**；运行目录本体按 §26.2 不入册 |
| 7 | **预注册 §4 列的结果文件名与实际发布名不符** | 预注册 §4 写 `labels_conserve.jsonl` / `conserve_stats.json` / `bomb_usage.json`；实际发布件是 **`labels_nobomb.jsonl`** / **`nobomb_stats.json`** / **`nobomb_bomb_usage.json`**（其余同名字段齐备）。原因是本轮按预注册 §2.1 从 25 px 保守规则切到 **`--bomb-rule none`** 兜底分支，发布时统一加 `nobomb` 前缀 | manifest 按**实际文件名**登记（正确） | **只报告、不改** |
| 8 | **`nobomb_stats.json` 合并层缺 `bomb1_spent_two_or_more`** | 合并后的 `label_checks` 只有 `labels_36 / bomb1_press_honoured / bomb1_noop_due_to_invuln / bomb1_anomalies / bomb0_spent_a_bomb / states_without_a_bomb_left / death_episodes / no_death_episodes`，**没有** `bomb1_spent_two_or_more`；但 12 个 `collect_cells[*].label_checks` 里都有，合计 **10,584** 正确。`RUN.md` §10 第 5 条已自述「合并用的 key 列表早于本轮新增字段」 | manifest 记录该文件字节**正确** | **只报告、不改** |
| 9 | **首轮 18 格评测的失败账与 `RUN.md` 叙述不齐** | `nobomb_commands.tsv` 里首轮 18 行的 `started` 都是 `14:43:18/19`、`finished` 都是 **`14:57:23`**、`wall_s` **844–845**，退出码分布 **`0`×2 / `4`×10 / `130`×6**；而 `RUN.md` §5 记「首轮 **5 个**『跑完』的 summary 同样作废重跑」、§10 第 1 条记 18 个 summary 全是 `episodes_total=0` / `wall 0.02–0.27 s` ⇒ **「5 个跑完」与 tsv 的 2 条 exit 0 不一致**；且 tsv 的 `wall_s` 是**批次级**（18 行完全相同）、summary 里的 0.02–0.27 s 是**单格级**，两者**不是同一个量**。作废结论本身有 tsv 支撑（18 行里 16 行非 0，其中 6 行 **exit 130 = SIGINT**） | manifest 只登记**最终发布件**（18 格全 `status=complete`），**不含首轮废件** | **只报告、不改** |
| 10 | **首版冒烟 exit 4（fail-closed）** | `nobomb_commands.tsv` 唯一一条 `smoke` 非零行：`--mode collect --bomb-rule none … --quota 20 --verify-h 1` **exit 4 / 88 s**（`14:29:13→14:30:41`）。`RUN.md` §10 第 2 条记成因：v8h0 的 `spent = after == before - 1` 假设「按一次扣一颗」，而 deathbomb **一次扣 2 颗** ⇒ 18/18 被判 anomaly、fail-closed 中止（`*.partial` 保留）。此后改为「扣 ≥1 颗即 honoured」并保留 fail-closed 门 | manifest 不含该次 `.partial` 运行件 | **只报告、不改** |
| 11 | **音频审计件只覆盖 13 个 data dir，72 个评测窗不在它里面** | `nobomb_audio_audit.json`（`14:43:29`）的 `scope` 是 `…/.tmp_v8i0_nobomb/data`，`cfg_files=13` = 12 个采集格 + 1 个首轮作废评测残留（`eval-20261122-144319`）；**15:0x 之后才创建的 18×4 = 72 个评测 data dir 完全不在该审计件内**，它们的静音只有各自 `.summary.json` 里的 `cfg_muted=true` / `cfg_sha256=cc062cfb…`（同一 60 B 模板）作旁证 ⇒ 「全轮 85 个 cfg 全静音」是**两份不同来源拼起来**的结论，**不是单一审计件覆盖** | manifest 不含 cfg（运行件） | **只报告、不改** |

**逐条核对口径（可复算）**：把 `manifest.json` 的 `files{}` 逐项与 `sha256sum` 实测比对，`{bytes, sha256}` **不符数 = 0**（80 项全过）；`bytes` 求和与 `total_bytes` **差 0**；盘上「有而 manifest 无」= **2 项**（各自 `manifest.json` + 顶层预注册）、「manifest 有而盘上无」= **0 项** ⇒ **无缺件、无多余件**。

> **一句话结论**：本轮把 **v8.17.0（CONSERVE-001 `none` / NOBOMB）82 条 / 419,911,442 B + 执行器 1 条 / 177,079 B = 83 条 / 420,088,521 B** 成对写入 `ASSET_SYNC.md` §26 与 `ASSET_SYNC.sha256`，`sha256sum -c` 由 **625 OK / 1 FAILED（626 行）** 变为 **708 OK / 1 FAILED（709 行）**（新增 83 条全 OK，整体 FAILED 数仍为 1，唯一 FAILED 仍为已知允许的 `v4.2.0/journal.jsonl`）。两个口径都记：`evidence/CONSERVE` **81 / 419,907,011 B**（目录）vs **80 / 419,895,193 B**（manifest 口径），差额恰为 `manifest.json` 自身 11,818 B，再加顶层预注册 4,431 B 即全目录 **82 / 419,911,442 B** ⇒ **无缺件、无多余件**；manifest 逐条比对 **80/80 全等、0 处不符**。**未启动引擎、未训练、未评测、未 `git add/commit`**；83 个**登记文件**的 `{size, mtime_ns}` 在登记前 / 等 130 s 后 / sha256 算完后**三次快照逐个零漂移**，写完两个清单后再实测一次仍是 `82 / 419,911,442 B` + `177,079 B`、最大 mtime 仍是 `manifest.json` 的 `15:42:02.326`；`v8.18.0/` 按要求**未登记、未触碰**（仍只有 1 个 4,301 B 的预注册，下一轮单独登记）。

## 27. v8.18.0 资产（2026-11-22）

> **为什么有这一节**：AGENTS.md §8.2 要求「新增/移动/删除任何资产，必须同步更新 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256`」。**v8.18.0（EARLYBOMB-001，「提前放雷」采样位置轮）已发布，但直到追加本节之前，它在两个清单里一行都没有** —— 追加前对既有 **709 行**校验表做 `v8\.18\.0|EARLYBOMB|v8j0` 匹配 = **0 命中**；本册正文里 `v8.18.0` 只出现 **9 次**，全部是 §25.2.4 / §26.2.3 的「**本轮不登记、下一轮单独登记**」以及 §25 / §26 收束段里对这同一句话的引述，**没有任何单文件的路径、字节或 sha256**（§27.3 逐条核对）⇒ 本节就是那一轮，**与 §1–§26 不重复**。本节与 `ASSET_SYNC.sha256` 末尾追加的 **81 行**成对登记。

> **本轮性质：`EARLYBOMB-001`（单因子 = 采样位置相对死亡的提前量），采集 + 训练 + 18 格评测全部 live engine**。行为策略仍是冻结的 v4.5.0 `S`（stochastic、`--bomb-rule none`、900 决策死亡即停），36 路 × H=120/240/600；12 个 (seed, window) 格 × 50 状态目标 = **447 个状态**（`ALIVE` **314**）、**1,341 行标签**；A/B 两臂 × 3 种子 × 200 epoch 训练（`Policy(106,64,36)`，9,188 参数，`train_horizon = H=600`）；A/B × 300/600/900 × 3 种子 × 4 dev 窗 × 300 episode = **18 格 / 21,600 episode**。判定 **`S-2bomb` → 按 `player_state == ALIVE` 过滤 → 四门在 `ALIVE` 池全过**，`summary.verdict` = **`train`**、`train_horizon = 600`。

> **本轮最重要的一条负结果（照录，不重判）**：**P3 不成立** —— `bomb_usage.json` 的 6 个池（A/B × 300/600/900）**每个池 3,600 episode 全部是 `{3: 3600}`**，`episodes_using_1_or_2 = 0`、`episodes_using_0 = 0`、分位 mean/median/q1/q3 全 **3.0**、stdev **0** ⇒ **21,600/21,600 局都把 3 颗雷用光，没有一局只用 1–2 颗**，平均用雷 **3.000** = 上限。也就是说：**这一轮没有学到「省着用」的资源配置**，两臂都退化成「一有机会就放」。同时 **P1 成立**（B 的 600 决策活满率 **31.11 %** > v8.3.0 的 **19.44 %**、> v8.17.0 的 **14.1 %**），但 900 决策最好也只有 **5.00 %**，仍远低于 R5 脚本规则的 **17.61 %**（§27.4.3 逐项列读数）。

> **只读边界**：本轮**只 append** 两个清单文件。`Document/Update/Alpha/v8.18.0/**`（**80** 个文件 / **378,445,951 B**）与执行器 `tools/rl/v8j0_earlybomb.py`（**1** 个 / **178,482 B**）全程只读；**未启动引擎**（0 次引擎调用）、**未训练、未评测、未 `git add` / 未 `git commit`**。登记前后各采一次 `{size, mtime_ns}` 快照（`16:07:32.058` 与 `16:10:32.058`，间隔 **180 s**），两个清单写完后再采第三次，**81/81 逐项相同、0 变化**（§27.4.4）。

> **实测口径**：81 条的 `bytes` 与 `sha256` **全部现场实测**（Python `hashlib` 流式读入，81/81），再用 WSL `sha256sum -c` 独立复算（§27.4.1）。发布方 `manifest.json` 的内联 `{bytes, sha256}` **只用于交叉核对，不代替实测**（§27.4.2：**78/78 全等、0 处不符**）。

> **两个口径都写（本册一贯做法）**：目录口径 = 递归实测总数（含 `manifest.json` 自身与顶层预注册）；`manifest.json` 自述口径 = 该文件 `file_count` / `total_bytes`，**`excludes="manifest.json"`**，且因 `root` 在 `evidence/EARLYBOMB/` 而**不含**顶层预注册。
>
> | 口径 | 文件数 | 字节 | 差额闭合 |
> | --- | ---: | ---: | --- |
> | v8.18.0 全目录（含顶层预注册 + `manifest.json` 自身） | **80** | **378,445,951** | — |
> | `evidence/EARLYBOMB/`（含 `manifest.json` 自身） | **79** | **378,441,650** | 378,441,650 − 378,430,106 = **11,544 B = `manifest.json` 自身** |
> | `manifest.json` 自述 | **78** | **378,430,106** | `excludes="manifest.json"`、`root=…/v8.18.0/evidence/EARLYBOMB` |
> | 顶层预注册（manifest 覆盖不到） | **1** | **4,301** | 80 − 79 = 1；378,445,951 − 378,441,650 = 4,301 |
>
> 两处差额都恰好闭合（**4,301 + 11,544 = 15,845**；**378,445,951 − 378,430,106 = 15,845**）⇒ **无缺件、无多余件**（与 §23.4.2 / §24.4.2 / §25.4.2 / §26 起首表同一口径）。

> **全量入册口径：80 个文件全部入册，未入册 0 个** —— 含 `manifest.json` 自身、18 份评测 `.summary.json`、6 个 `.pt`、12 个采集格、3 份统计出口、7 份门与审计件、2 份冒烟件、2 份运行记录。理由：这些**都不是可重跑重建的日志**，而是本轮唯一的采集 / 训练 / 评测 / 门禁 / 摘要出口。**本轮最小集 = 全集 + 执行器。**

> **⚠ 与工单的估值不符（只报告）**：工单估「约 **62** 文件」，**盘上实测 80 个**（+18）。原因是发布收尾阶段（`16:06:58`–`16:07:23`）又写了/重写了 3 件：`RUN.md` 由 9,257 B 重写为 **19,557 B**、新增 `summary.json` **28,608 B**、`manifest.json` 由 11,419 B 重写为 **11,544 B**；执行器同期也由 178,433 B 变为 **178,482 B**（§27.5 第 1、2 条）。**本节一律以盘上为准。**

### 27.1 登记范围（**81 条** = v8.18.0 **80 条** + 执行器 **1 条** / **378,624,433 B**）

| 分组 | 条数 | 字节合计 | 位置 |
| --- | ---: | ---: | --- |
| §27.1.1 v8.18.0 冻结预注册 | 1 | 4,301 | `Document/Update/Alpha/v8.18.0/` |
| §27.1.2 运行记录与命令表 | 2 | 21,044 | `v8.18.0/evidence/EARLYBOMB/` |
| §27.1.3 身份门、保真门与三项审计 | 7 | 60,653 | `v8.18.0/evidence/EARLYBOMB/` |
| §27.1.4 冒烟件 | 2 | 58,252 | `v8.18.0/evidence/EARLYBOMB/`（1 件在 `smoke/`） |
| §27.1.5 12 个采集格记录 | 12 | 1,168,792 | `v8.18.0/evidence/EARLYBOMB/collect/` |
| §27.1.6 标签集、训练池与 P0 统计 | 3 | 21,266,731 | `v8.18.0/evidence/EARLYBOMB/` |
| §27.1.7 训练记录、6 个 checkpoint 与训练汇总 | 13 | 301,033 | `v8.18.0/evidence/EARLYBOMB/` |
| §27.1.8 评测行数据与逐格摘要（18 格 × 2 件） | 36 | 355,484,182 | `v8.18.0/evidence/EARLYBOMB/` |
| §27.1.9 统计与诊断出口 | 3 | 69,419 | `v8.18.0/evidence/EARLYBOMB/` |
| §27.1.10 发布清单锚 | 1 | 11,544 | `v8.18.0/evidence/EARLYBOMB/` |
| **v8.18.0 小计** | **80** | **378,445,951** | — |
| §27.1.11 执行器 | 1 | 178,482 | `tools/rl/` |
| **合计** | **81** | **378,624,433** | — |

#### 27.1.1 v8.18.0 冻结预注册（1 条 / 4,301 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.18.0/PREREGISTRATION_EARLYBOMB_001.md` | 4,301 | `d4bbb00946ce184914981d306ea540288c39cf06d3d6b77c2d7cb7737f00a57e` | v8.18.0 冻结预注册（**FROZEN**，`14:49:14` 落盘，早于本轮任何采集）：单因子 = **采样位置相对死亡的提前量**（v8.17.0 的「最后 20 决策 = 40 帧」→ `decisions_before_death ∈ [30, 60]` = 60–120 帧）；P0a `player_state == ALIVE` **≥90 %**、P0b `bombs_remaining > 0` **≥50 %**、**P0c honoured 分支里 `spent == 1` ≥90 %**（与 v8.17 的判别式）、P0d diff **≥20 %**、P1 臂 B 的 600 决策活满率 **>19.44 %**（v8.3.0）且 **>14.1 %**（v8.17.0）、P3 用雷分布、P4 900 档并排；§2.1 停止条件 `S-dead` / `S-2bomb` / `S-nodiff` / `S-ok` / `S-fork`。**三处独立记录同 sha**（`preflight.json` 的 `pins.prereg` 带 `bytes: 4301`、`ckpt_assertions{,_final}.json` 的 `prereg`、盘上实测）⇒ 轮内**一字未改**；本轮走的正是 §2.1 授权的 **`S-2bomb`** 兜底（按 `player_state == ALIVE` 过滤，见 §27.4.3） |

#### 27.1.2 运行记录与命令表（2 条 / 21,044 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/RUN.md` | 19,557 | `8350732ca9a048625b611c8d2dae19c6494aa1b50b0f199822201aa3153a26c1` | **247 行** / **19,557 B** 运行记录（`16:06:58`，发布收尾阶段被重写过一次，见 §27.5 第 1 条）：§0 本轮最重要的实测 **死亡检测滞后 = 27 决策 = 54 帧（≈1.8 s @30 fps）**（`death_decision 125` − `last_ALIVE 98`），并给出「普通放雷扣 1 颗」与「死亡炸弹扣 2 颗」的分界 ≈ 死前 54 帧；§1 单因子设计（采样窗由预注册 `[30,60]` 收窄到 **`[20,40]` = 40–80 帧**，理由是先测量后收窄）；§2 P0a–P0d 两套（整窗 447 / `ALIVE` 314）；§3 训练 A/B 逐种子表；§4 评测（18 组、21,600 episode、三档 × 池化 / 逐窗 / 逐种子 / 用雷分布 / 并排 / 判据）；§5 保真门 / 静音 / 进程；§6 **8 条偏差**；§7「不能声称什么」7 条；§8 产物表；§9 复现命令。**§8 自述「77 个文件」与盘上 79 不符**（§27.5 第 4 条） |
| 2 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/commands.tsv` | 1,487 | `f8c39fb3ebbc0335cfb1204300574cd4907d83610a338bd9f7e356a81da00ac5` | **17 行**（1 行表头 + **16 条**汇总行）三列 `mode / command / exit`：preflight 1、smoke 2、collect 2、assert-ckpt 2、fidelity 1、stats 1、train 1、train-summary 1、**eval 1**、diagnostics 1、audio-audit 1、process-audit 1、publish 1 = 16。**exit 列 14 个 `0`、1 个 `4`**（`smoke-tight`）、**1 个非数值** `(killed: …)`（`smoke-first`，`window=auto` 死亡决策 45 的退化窗被主动中止）。它与 v8.17.0 的 69 条**逐条**日志**不是同一口径**：12 个采集格压成 1 行、18 组评测压成 1 行（§27.5 第 8 条） |

#### 27.1.3 身份门、保真门与三项审计（7 条 / 60,653 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/preflight.json` | 9,153 | `7503a398407e795943dee505167fab4607544ef4ee307ec4fdefae258b9912ac` | 开工前身份门（`14:54:40`，`all_pass=true`）：**12 个 pin 全部 match**（`tools/rl/v450_cf_label.py` `b7ab0ebe…`、`v432_fork_arm.py` `33235506…`、`v431_arm_a.py` `96923930…`、`v431_dense_driver.py` `1079d93a…`、`v800_bomb.py` `5d85bcb0…`、`v830_bombpol.py` `a8c6d306…`、`v860_bombdag.py` `3433ae74…`、`v400_make_cfg.py` `a8c2aabc…`、`touhou_rl/v420/contracts.py` `1357c69c…`、`touhou_rl/v420/counterfactual.py` `c1f8cae4…`、`touhou_rl/v400/gym/actions.py` `a67145b2…`、`touhou_rl/v400/gym/protocol.py` `9081402d…`）；`prereg.sha256=d4bbb009…` / **4,301 B**；`feature_dim=106`、`action_space{18 候选 × 2 放雷位 = 36}`、`behaviour=S stochastic / bomb_rule=none`、`frozen{TRAIN_WINDOWS 13–16, DEV/SAMPLE_WINDOWS 11/12/17/18, horizons 120/240/600, H_max 600, decisions_cap 900, 200 epoch, lr 1e-3, batch 256, hidden 64, target_states 600, cells 12, quota 50}`、`references{v830_bombpol, v860_bombdag, v8170_conserve, rule_r5}`。**`pins.self` 记执行器 `692db4cb…` / 164,111 B —— 与盘上现值不同**（§27.5 第 2 条） |
| 2 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/ckpt_assertions.json` | 3,735 | `3c3260cb67cab5d70ead8f5f21b758df51b0559efe0767f0605080de75c18252` | `phase=start`（`15:31:22`）：引擎 `691bfd77…` match=true；行为策略 `round2.pt` **3/3 match**（`4abee069…` / `a93746bb…` / `c69960fc…`，各 **34,387 B**）；`protected{}` **9 件**只记 `{exists, sha256, bytes}`、**没有 `expected`**（`match` 不参与 `all_pass`，§27.5 第 9 条）；`all_pass=true`；`prereg=d4bbb009…`；`v8j0_earlybomb_py` = **`639f54ce…`（无 `bytes` 字段）**（§27.5 第 2 条） |
| 3 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/ckpt_assertions_final.json` | 3,735 | `a72cc02780363683b408d583b5b926238aae7ac0c9e52737623b07d9f4a653e4` | `phase=final`（`16:05:53`）：引擎与 3 个 `round2.pt` 同上 **3/3 match**、`all_pass=true`、`prereg` 与 start 相**相同**；**9 件 `protected` 与 start 相逐条相同**（本轮与 v8.17 不同：受保护文件**没有**在轮内漂移）；`v8j0_earlybomb_py` 仍是 **`639f54ce…`**，而盘上现值是 **`655d0c1f…` / 178,482 B / mtime `16:07:03.062`**（晚于本件 **70 s**）⇒ 执行器在 final 门之后又被改过（§27.5 第 2 条） |
| 4 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/fidelity_d3.json` | 26,740 | `e1b3654bea50a844b6b6792790b1c6718bd28979db1a4e046a15c9497e18e03b` | D3 保真门 **5/5 PASS**（`all_pass=true`、`error=null`，`15:31:26`，`schema=v432-fork-fidelity-live/1`）：① HELLO `role=shadow`；② DEBUG `threads=1`（PARK 前后）；③ 前缀 **2,165 / 2,165** checkpoint 到帧 **8663**、`hash_mismatches=0`；④ R24 **18/18** 常量；⑤ PARK `frame=8663`。源是 `.tmp_v410_s1c` 的 seed12 shard（`rows_in_shard 3,298` / `frames_built 13,192` / `target 8664`）；`launch_env{SDL_AUDIODRIVER: dummy, DISPLAY: null, WAYLAND_DISPLAY: null}`。**非 5/5 即整轮作废 —— 本轮未触发 `S-fork`** |
| 5 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/fidelity_d3_wrapper.json` | 5,781 | `b13badc8b53963a2975f1e4b5505f0660a96f1dbfc6ad59bc7fe54f37bc0a456` | **门禁委托件**（`15:31:41`，`schema=v8180-earlybomb-fidelity/1`）：为免在本轮代码里重实现判据，`--mode fidelity` 把门**原样委托**给 `tools/rl/v432_fork_arm.py --mode fidelity --seeds 12 --learner frozen`（子进程，`returncode=0`），`d3_all_pass=true`、`d3_wall_s=14.473`、`wall_s=17.713` 一并落盘 ⇒ 判据出处单一、可复核 |
| 6 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/audio_audit.json` | 10,807 | `d0c20dc115100cc6f395e8d74a7681d4f9ec747e78117fa64bf59d66ffaac130` | AGENTS.md §9.3 主审计（`16:05:54`，`schema=v450-cf-audio-audit/1`，`scope=…/.tmp_v8j0_earlybomb/data`）：`cfg_files=89 / decoded=89 / **muted=89** / **audible=0** / errors=0`，`audible=[]`、`by_root` **35 个根目录**（12 个采集格 + 20 个评测根 + 3 个冒烟窗），`all_muted=true`；共享审计件 `v4.3.1/evidence/R14/audio_cfg_audit.json` **未触碰** ⇒ 本轮**未新增 `audible`**。覆盖明细：12 采集 + **74 评测 cfg** + 3 冒烟 = 89；74 = 17 个正式格 × 4 窗（68）+ `B-d600-s1-retry` 4 窗 + 该格失败首跑 1 窗 + `eval-smoke` 1 窗 |
| 7 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/process_residue.json` | 702 | `e86b89dc9216973d01588abdf03498dbe7096f892a0437c590cd2765c7da903f` | 本轮引擎残留审计（`16:05:55`，`schema=v8180-earlybomb-process-audit/1`）：`engine_name=th08-modern`、`data_dir_filter=.tmp_v8j0_earlybomb`、**`matching=0`**、`lines=[]`、`pgrep_returncode=1`、`processes_total_by_engine_name=0`、`all_pass=true` ⇒ 本轮**没有**留下引擎进程，也**没有** `pkill` 任何他轮进程。（与 v8.17.0 的 `66` 不同：那次审计时全机有他轮引擎；本轮审计时刻 `16:05:55` 全机确有 0 台 th08-modern，本轮评测 `16:04:46` 结束） |

#### 27.1.4 冒烟件（2 条 / 58,252 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/smoke.json` | 9,007 | `e4791f72477738a4d10ed801e9f9b96955d8ab8e0a0527ab150720bc32371003` | 冒烟结论件（`15:02:03`，`schema=v8180-earlybomb-smoke/1`，`window=12`、`states=21`）：**死亡检测滞后** `death_decision 125` / `last_ALIVE 98` / **`frames_of_DYING_before_detection = 27`**；21 状态的 `p0a 14/21 = 66.7 %（met=false）`、`p0b 21/21 = 100 %`、`p0c bomb1_spent_amount_counts {1: 252, 2: 126}` = **252/378 = 66.7 %（met=false）**；三视界嵌套校验 `nesting_verified`（H=600 一次跑、`alive(H)=min(alive(600),H)`）⇒ **冒烟不是判定件**，它只用来定位采样窗与验证 H 嵌套（判定走 `--mode stats` 的 447/314） |
| 2 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/smoke/smoke_seed0_w12.json` | 49,245 | `ef2c58e5efafec7126ba914ec08a183872c0154b3be016e52922d193b5f357ab` | 冒烟采集格原始记录（seed0 / w12 / `window=tight`，`16:05:0x` 前落盘）：逐 episode 的 `death_decision` 与逐状态的 36 路标签，是 `smoke.json` 数字的唯一来源；12 个正式采集格由同一代码路径产出 |

#### 27.1.5 12 个采集格记录（12 条 / 1,168,792 B）

12 个 `collect/seed{S}_w{W}.json` = **3 种子 × 4 dev 窗**：`status` 有 **5 个 `complete` / 7 个 `skipped`**（`skipped` 指该格在 `min_death_decision=300` 第一轮被守卫清空、随后用 **120** 档重取），**8 个格有状态、4 个格 0 状态**（`s0w18`、`s1w11`、`s1w17`、`s2w18` 两档都是「连续 10 条早死」）；`states` 求和 = **447**、`labels_total` 求和 = **16,092**（36 路）；12/12 `gate.all_pass=true` + `prefix{hash_checks 2,165, hash_mismatches 0, final_frame 8663}`、12/12 `invalid_episodes=0`、`replay_mismatches=0`、`cfg` 四字段全 0（同一模板 `cc062cfb…` / 60 B）。

| 格（种子 / 窗） | 字节 | SHA-256（前 12 位） | status | 状态数 | 标签条数 | `min_death_decision` | 早死丢弃 episode |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| seed0 / w11 | 70,128 | `14c420b5cd12…` | skipped | 21 | 756 | 120 | 14 |
| seed0 / w12 | 147,819 | `a30b7ea4351d…` | complete | 75 | 2,700 | 300 | 4 |
| seed0 / w17 | 127,336 | `36f8f0b5b854…` | complete | 75 | 2,700 | 300 | 7 |
| seed0 / w18 | 34,119 | `a43cbb2b9293…` | skipped | 0 | 0 | 120 | 10 |
| seed1 / w11 | 40,550 | `909d6e7b5f56…` | skipped | 0 | 0 | 120 | 10 |
| seed1 / w12 | 77,057 | `1c40f6c5907d…` | complete | 75 | 2,700 | 120 | 5 |
| seed1 / w17 | 48,628 | `ec675d184874…` | skipped | 0 | 0 | 120 | 10 |
| seed1 / w18 | 53,265 | `77a96b3ecd4b…` | skipped | 21 | 756 | 120 | 13 |
| seed2 / w11 | 153,598 | `a7f82e97cd7d…` | skipped | 63 | 2,268 | 300 | 27 |
| seed2 / w12 | 259,071 | `fb5d975e9d27…` | skipped | 42 | 1,512 | 300 | 20 |
| seed2 / w17 | 113,213 | `bd7266ea5de9…` | complete | 75 | 2,700 | 300 | 1 |
| seed2 / w18 | 44,008 | `8eb00eb18957…` | skipped | 0 | 0 | 120 | 10 |

（`early_death_episodes` 合计 **131**，与 `summary.collection.early_death_episodes` 一致；4 个 0 状态格的标签 jsonl 在运行目录里也存在，但**长度为 0**，按 §27.2 不入册。）

#### 27.1.6 标签集、训练池与 P0 统计（3 条 / 21,266,731 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/labels_earlybomb.jsonl` | 12,462,772 | `2e8cd349f0eb56d7990328e638f56af8f95147ea30a0ba963beb548b5cf01871` | **池化标签集**（`15:25:01`）：**1,341 行** = 447 状态 × 3 个 H（每行 36 路 `alive_frames`）。其 sha **与 `earlybomb_stats.json` 的 `labels_file` 逐位相同**（`2e8cd349…` / 12,462,772 B / 1,341 行）⇒ 统计用的是同一份字节 |
| 2 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/labels_earlybomb_alive.jsonl` | 8,762,992 | `305639df0bfabc895c639113a368bbb7be42a5490719b36aae31a1087cbe393b` | **训练池**（`15:25:03`）：`filter=ALIVE` 后的 **942 行** = **314 状态** × 3 个 H；sha 被 `train_summary` 与 `earlybomb_stats.training_pool_file` 记录（`305639df…` / 8,762,992 B / 942 行）⇒ 训练样本不是整窗 447，而是判定池 314 |
| 3 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/earlybomb_stats.json` | 40,967 | `171d887f6d808c1e7a53474ae05fe3b5e165dedfaae8afb00ce4abba717f4184` | **P0 主出口**（`15:25:04`，`schema=v8180-earlybomb-stats/1`）：`states=447`、`label_rows=1,341`、`labels=16,092`、`episodes=23`、`cell_count=12`、`failed_cells=0`、`early_death_episodes=131`；`filter{states_all 447, states_alive 314, alive_fraction 0.702461}`；`p0_all`（整窗）P0a **314/447 = 70.25 %（不达 90 %）**、P0b 447/447、P0c **5,652/8,046 = 70.25 %（不达 90 %）**、P0d 151/447 = 33.78 % / 183/447 = 40.94 % / **360/447 = 80.54 %**；`p0_alive`（判定池）P0a **314/314 = 100 %**、P0b 314/314、P0c **5,652/5,652 = 100 %**、P0d 18/314 = 5.73 % / 50/314 = 15.92 % / **227/314 = 72.29 %**；`verdict{name: train, train_horizon: 600, p0d_horizons_met: [600], filter: ALIVE}`；`label_checks{labels_36 447, bomb1_press_honoured 8,046, bomb1_spent_one 5,652, bomb1_spent_two_or_more 2,394, bomb1_noop_due_to_invuln 0, bomb0_spent_a_bomb 0, states_without_a_bomb_left 0, death_episodes 23, no_death_episodes 0}`；`bombs_at_death{3: 23}`、`states_by_player_state{ALIVE: 314, DYING: 133}`；`sampling.window_bounds.tight = {lo: 20, hi: 40, frames: [40, 80]}`（**不是**预注册的 `[30, 60]`） |

#### 27.1.7 训练记录、6 个 checkpoint 与训练汇总（13 条 / 301,033 B）

6 个 `ckpt_{A,B}_earlybomb_seed{0,1,2}.pt` 各 **39,281 B**（`Policy(106,64,36)`，9,188 参数），其 sha256 与 `train_summary.json` 的 `arms[*][*].ckpt.sha256` **6/6 全等**、且被 18 份评测 `.summary.json` 的 `ckpt.sha256` 引用 **18/18 全等**：

| checkpoint | 字节 | SHA-256（前 12 位） | 训练记录 | 训练记录字节 | 训练记录 SHA-256（前 12 位） |
| --- | ---: | --- | --- | ---: | --- |
| `ckpt_A_earlybomb_seed0.pt` | 39,281 | `4d96a536448a…` | `train_A_earlybomb_seed0.json` | 11,493 | `f96e4ad906a7…` |
| `ckpt_A_earlybomb_seed1.pt` | 39,281 | `b4675badee55…` | `train_A_earlybomb_seed1.json` | 11,488 | `a14b144ff441…` |
| `ckpt_A_earlybomb_seed2.pt` | 39,281 | `df676ef4bdb8…` | `train_A_earlybomb_seed2.json` | 11,477 | `192eb1c5771e…` |
| `ckpt_B_earlybomb_seed0.pt` | 39,281 | `d6b0681e4cd7…` | `train_B_earlybomb_seed0.json` | 8,846 | `df13b4d707c4…` |
| `ckpt_B_earlybomb_seed1.pt` | 39,281 | `44ea6ac1ba55…` | `train_B_earlybomb_seed1.json` | 8,839 | `377a913a7f7c…` |
| `ckpt_B_earlybomb_seed2.pt` | 39,281 | `8bc5b1a9bd4c…` | `train_B_earlybomb_seed2.json` | 8,841 | `7abe8171fd82…` |

6 份 `train_*.json` 的读数（`states=314`、`H=600`、200 epoch、Adam 1e-3 / batch 256）：**A 臂**（`BCE(36) + 0.5 × BCE_bomb(best-bomb)`）final BCE **0.219553 / 0.220798 / 0.224351**、aux BCE **0.212780 / 0.208831 / 0.208815**、`train_accuracy` **0.906051 / 0.905609 / 0.903751**、`aux_bomb_agreement.accuracy` **0.882166 / 0.891720 / 0.885350**、wall **51.669 / 44.866 / 40.955 s**；**B 臂**（仅 `BCE(36)`）final BCE **0.212176 / 0.210473 / 0.209420**、`train_accuracy` **0.914013 / 0.913924 / 0.915340**、`aux` 为 `null`、wall **11.774 / 2.872 / 2.794 s**。**B 臂三项都比 A 臂好**（BCE 更低、acc 更高），与 §27.4.3 的 600 决策 2.17 pp 描述性差同向。

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 13 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/train_summary.json` | 4,363 | `769bdb71093febd5b1d8b4d9556c40c4a42ce0b99e09bae9e27b9ca884feb117` | A/B 六格的**训练汇总出口**（`15:28:28`，`schema=v8180-earlybomb-train-summary/1`）：`arms{A,B}{0,1,2}` 的 `{states 314, H 600, loss_first_epoch, loss_last_epoch, final_train_bce, final_train_aux_bce, train_accuracy, aux_bomb_agreement, wall_s, ckpt{path, sha256, bytes 39281}`；`producer` 里仍写采样窗 `[30, 60]`（§27.5 第 3 条） |

#### 27.1.8 评测行数据与逐格摘要（36 条 / 355,484,182 B / **21,600 行 episode**）

18 格 = **2 臂（A / B）× 3 预算（300 / 600 / 900）× 3 种子**，每格 **1,200 episode**（4 dev 窗 × 300），每格一个 `.jsonl`（`out_bytes` 与盘上一致）加一个同名 `.summary.json`。**18/18 `status=complete`、`error=null`、`episodes_total=1,200`、`bombs_used_total=3,600`**；每格 4 个窗口都带 `cfg_muted=true`（同一模板 `cc062cfb…` / 60 B）、`gate.all_pass=true`、`gate_prefix{hash_checks 2,165, hash_mismatches 0, final_frame 8663}` ⇒ **72/72 个窗口全过**；`engine.match=true` **18/18**、`launch_env{SDL_AUDIODRIVER: dummy, DISPLAY: null, WAYLAND_DISPLAY: null}` **18/18**。下表把**每格的 jsonl 与 summary 合成一行**（`sha256` 取前 12 位；完整 64 位在 `ASSET_SYNC.sha256`）。

| 格（臂 / 预算 / 种子） | jsonl 字节 | jsonl SHA-256（前 12 位） | summary 字节 | summary SHA-256（前 12 位） | 活满 / 1,200 | 活满率 |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| A / 300 / seed0 | 11,633,176 | `a016811a3dc2…` | 6,417 | `aa07b39663a3…` | 1,200 | **1.000000** |
| A / 300 / seed1 | 12,358,115 | `42b1ed446e71…` | 6,450 | `0fa587086e43…` | 1,193 | **0.994167** |
| A / 300 / seed2 | 11,296,860 | `74e84213c740…` | 6,439 | `143f66822384…` | 1,199 | **0.999167** |
| A / 600 / seed0 | 19,705,078 | `ba96d1d3fc31…` | 6,537 | `f2ebda7224c0…` | 331 | **0.275833** |
| A / 600 / seed1 | 19,230,981 | `c6a6edd5ed19…` | 6,532 | `e18df0bc19ee…` | 354 | **0.295000** |
| A / 600 / seed2 | 19,253,090 | `0b995a8e2ddd…` | 6,525 | `dd455a147780…` | 357 | **0.297500** |
| A / 900 / seed0 | 21,650,498 | `b33f55276b35…` | 6,531 | `72713433c5e7…` | 42 | **0.035000** |
| A / 900 / seed1 | 20,456,903 | `8ecb83e9d9fa…` | 6,534 | `56c02b51fdd6…` | 51 | **0.042500** |
| A / 900 / seed2 | 20,928,687 | `4f99d98c90f9…` | 6,525 | `4fde45d6aafe…` | 58 | **0.048333** |
| B / 300 / seed0 | 14,919,026 | `374a5aae15d3…` | 6,331 | `0774259a8447…` | 1,199 | **0.999167** |
| B / 300 / seed1 | 14,989,705 | `d40df264e511…` | 6,334 | `f49f6ea48551…` | 1,198 | **0.998333** |
| B / 300 / seed2 | 14,751,765 | `e4a7f296f123…` | 6,334 | `66f46b8d9fdd…` | 1,199 | **0.999167** |
| B / 600 / seed0 | 24,745,605 | `da288111b86d…` | 6,425 | `64c2c319a718…` | 383 | **0.319167** |
| B / 600 / seed1 | 24,749,929 | `462fa65d65d7…` | 6,447 | `8d49199c789c…` | 372 | **0.310000** |
| B / 600 / seed2 | 24,510,197 | `5a9789787601…` | 6,418 | `b1470be670d4…` | 365 | **0.304167** |
| B / 900 / seed0 | 26,546,843 | `9898f1444463…` | 6,414 | `8c16d49975f8…` | 58 | **0.048333** |
| B / 900 / seed1 | 27,020,643 | `9956b3f9971b…` | 6,422 | `1d8b72564d02…` | 60 | **0.050000** |
| B / 900 / seed2 | 26,621,042 | `0ca0908b8698…` | 6,424 | `d8e33abf72f4…` | 62 | **0.051667** |

**逐格读数与自检（18/18 全过）**：逐臂逐预算的池化读数（`summary.eval`，每池 3,600 episode）—— **A** d300 **3,592/3,600 = 99.7778 %**、d600 **1,042/3,600 = 28.9444 %**、d900 **151/3,600 = 4.1944 %**；**B** d300 **3,596/3,600 = 99.8889 %**、d600 **1,120/3,600 = 31.1111 %**、d900 **180/3,600 = 5.0000 %**。18 个 jsonl 的实测字节合计 **355,368,143 B**、18 份 summary 合计 **116,039 B** ⇒ **355,484,182 B**（36 条）。体积主因是每行含 `actions_sampled`（300/600/900 个整数），见 `RUN.md` §6.8。

#### 27.1.9 统计与诊断出口（3 条 / 69,419 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/summary.json` | 28,608 | `9d9a3e8b021da1d688fcaf32c92ddd0ad65910e1eb7d8b3341a267774aac1365` | **一页汇总 / 主判据出口**（`16:07:13`，`schema=v8180-earlybomb-summary/1`）：`verdict` = **`{name: train, rule: "the smallest horizon reaching the P0d gate is the one to train with", train_horizon: 600, p0d_horizons_met: [600], filter: ALIVE}`**；`p0{chosen 314, all 447, alive 314}` 三套；`smoke`（滞后 27 决策）；`collection{states 447, label_rows 1,341, cells 12, cells_with_states 8, early_death_episodes 131, min_death_decisions_used [120, 300]}`；`training`（6 格）；`eval{A,B}{d300,d600,d900}` 池化 + `per_window` + `per_seed`；`references{v830_bombpol, v860_bombdag, v8170_conserve, rule_r5}`；`preflight_all_pass=true`、`ckpt_assertions_all_pass=true`、`d3_fidelity.all_pass=true`（`wall_s 17.713`）；`artifacts{}` **13 条**（**不含** 18×2 评测件、6 个 `.pt`、`ckpt_assertions_final`、`audio_audit`、`process_residue`、`manifest.json` 自身 —— §27.5 第 7 条） |
| 2 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/diagnostics.json` | 36,333 | `a74f9b29c2cd683bd10eeeca2740675c4bb42e608eeefb3fd03c1507f78b51e2` | 逐臂 / 逐预算 / 逐种子的诊断汇总（`16:05:37`，`schema=v8180-earlybomb-diagnostics/1`）：`arms{A,B}{d300,d600,d900}.per_seed{0,1,2}` 的 `{episodes, survived, survived_fraction, alive_frames 分位, bombs_used, per_window}`；`bomb_usage`；`references{v830, v860, v8170, rule_r5}`；`side_by_side_survived_fraction` = `{v830: .999444/.194444/.006111, v860: .999444/.295833/.029444, v8170_20s: .141, rule_r5: 1.0/.7711/.1761, arm_A: .997778/.289444/.041944, arm_B: .998889/.311111/.05}`。`requirement` 明写「**本文件不含检验、不含区间**：P1–P4 是计数与分布，检验与置信区间属于独立分析轮」；`note` 明写「12 个 (seed, window) 格不是独立样本，逐格数字不得平均成均值的均值」 |
| 3 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/bomb_usage.json` | 4,478 | `a7de5ea2f9174bd3c2004eb4beeea0ece166493701c971bf754ec4b7012bd21b` | **P3 专项读数**（`16:05:41`，`schema=v8180-earlybomb-bomb-usage/1`）：6 个池（A/B × 300/600/900）**每池 3,600 episode**，`bombs_used_counts` 全部是 **`{3: 3600}`**、**`episodes_using_1_or_2 = 0`**、`episodes_using_0 = 0`、`episodes_using_3 = 3,600`、分位 mean/median/q1/q3 全 **3.0**、stdev **0.0**；`bomb_emits_total` **716,912 / 1,268,434 / 1,383,976**（A）与 **972,993 / 1,701,075 / 1,852,222**（B）⇒ **P3 不成立：21,600 局里没有任何一局只用了 1–2 颗雷，全部用光 3 颗** |

#### 27.1.10 发布清单锚（1 条 / 11,544 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB/manifest.json` | 11,544 | `5a98c2faff470714fc74b4b1f620c59634b21246f132d2341b5c0e6393162dff` | **发布清单锚**（`created_at=2026-11-22T16:07:23+0800`，`schema=v8180-earlybomb-manifest/1`）：`file_count=78`、`total_bytes=378,430,106`、`excludes="manifest.json"`、`root=/mnt/d/WorkShop/Touhou_RL/Document/Update/Alpha/v8.18.0/evidence/EARLYBOMB`；`files{}` **78** 条 `{sha256, bytes}` —— 本节实测 **78/78 全等、0 处不符**，`bytes` 求和 = `total_bytes` = 378,430,106（§27.4.2）；**mtime `16:07:22.692` 是目录内最后一个写入**（晚于 `summary.json` `16:07:11.000`、`RUN.md` `16:06:58.000`）⇒ **没有**「manifest 是旧快照」的问题 |

#### 27.1.11 执行器（1 条 / 178,482 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 | 用途 / 事实 |
| ---: | --- | ---: | --- | --- |
| 1 | `tools/rl/v8j0_earlybomb.py` | 178,482 | `655d0c1f98da2bb67a260a46f22e8945e33ad469e58bcddb832e9b252dd1861e` | v8.18.0 执行器（`--mode` 13 个取值：preflight / assert-ckpt / fidelity / smoke / collect / stats / train / train-summary / eval / diagnostics / audio-audit / process-audit / publish）。**盘上现值 `655d0c1f…` / 178,482 B / mtime `16:07:03.062`**（`RUN.md` §6 说明它是本轮唯一新增代码，`touhou_rl/**`、`third_party/**`、既有版本目录与 `v8i0_nobomb.py` 一律未改）。**轮内至少三个状态**：`preflight.pins.self`（`14:54:40`）= `692db4cb…` / **164,111 B** → `ckpt_assertions{,_final}.v8j0_earlybomb_py`（`15:31:22` / `16:05:53`）= **`639f54ce…`（无 `bytes`）** → 盘上 `655d0c1f…` / **178,482 B**（`16:07:03`，晚于 final 门 **70 s**）⇒ **final 门看到的不是最终字节**（§27.5 第 2 条）。**两份清单都不含执行器**（对 `manifest.json` 的 `files{}` 做 `v8j0_` 匹配 = **0 命中**；`summary.artifacts{}` 也没有它），其字节身份只由本条 + 三处 JSON 的 `pins.self` / `v8j0_earlybomb_py` 提供 |

### 27.2 跳过说明（未入册 **0** 个文件；`.tmp_v8j0_earlybomb/` 运行件不入册）

实测：v8.18.0 目录 **80** 个文件 / **378,445,951 B**，执行器 **1** 个 / **178,482 B** ⇒ **81 个文件 / 378,624,433 B，入册 81、未入册 0**。

| 组 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `Document/Update/Alpha/v8.18.0/**` | 80 | 378,445,951 | **全部入册**（含 `manifest.json` 自身、18 份 `.summary.json`、6 个 `.pt`、12 个采集格、9 份门/审计/统计件、2 份冒烟件）；未入册 **0** |
| `tools/rl/v8j0_earlybomb.py` | 1 | 178,482 | **全部入册**；未入册 **0** |
| **小计** | **81** | **378,624,433** | — |

#### 27.2.1 `.tmp_v8j0_earlybomb/` 是 gitignored 的运行件（不入册）

`.gitignore` 里有一条 **`.tmp_*/`**（第 32 行），`.tmp_v8j0_earlybomb/` 整棵树因此**不进 Git**，是过程现场而非发布件；其**唯一需要字节身份的部分（12 个采集格记录、两份标签池、6 份训练记录与 6 个 ckpt、18 格评测行数据与逐格摘要、统计/诊断/命令表/门禁出口、发布清单）已全部发布到 `Document/Update/Alpha/v8.18.0/` 并被本节 80 条完全覆盖**。运行目录里剩下的 `data/`（私有 cfg 与软链的 `th08.dat`）、`work/`、`logs/`、`runs/`、`ckpt/`、`labels/`、`eval/`、`smoke/` 下与发布件同名的 jsonl，以及 **10 个未发布的辅助脚本**（`analyze_smoke.py`、`make_summary.py`、`report.py`、`verify.sh`、`finalize.sh`、`publish_final.sh`、`probe.sh`、`probe_eval.sh`、`probe_host.sh`、`probe_rate.sh`）+ 4 个 `run_*.sh` 按 §21.2 / §22.2 / §23.2 / §24.2 / §25.2.2 / §26.2.1 同口径**不单独入册**。

运行目录里另有 **3 个未发布的中间件**，与发布件的区别已核实：`collect/firstpass_min300/`（`RUN.md` §8 提到，但发布树 `collect/` 只有 12 个 json，无此子目录）、`smoke/labels_smoke_seed0_w11.jsonl.partial`（139,982 B）与 `smoke/labels_smoke_seed0_w12_EARLY30-60.partial`（194,193 B，`[30,60]` 原始窗的部分标注）—— 三者**只在运行目录里**，按本节口径不单独入册（§27.5 第 5 条）。运行目录的 Windows 侧递归计数**只报数、不登记**。

#### 27.2.2 `__pycache__` / `*.pyc` / 隐藏文件实况（按要求只报告、不删）

`v8.18.0/` 全树只有 **4** 个目录（`evidence/`、`evidence/EARLYBOMB/`、`evidence/EARLYBOMB/collect/`、`evidence/EARLYBOMB/smoke/`）、**0** 个 `__pycache__/`、**0** 个 `*.pyc`、**0** 个隐藏文件（`Get-ChildItem -Force` 与不加 `-Force` 两种口径都是 **80 个文件 / 378,445,951 B**）。目录内没有任何脚本本体（本轮的脚本与 `run_*.sh` 都留在运行目录，未随证据发布），所以**不存在**前几轮那种「发布树里夹着脚本」的情况。若日后出现 `__pycache__/` 或 `*.pyc`，按 §21.2 口径**一律不入册**。

#### 27.2.3 §26 的「不登记 `v8.18.0/`」被本节接续

§26.2.3 写的是「`Document/Update/Alpha/v8.18.0/` **本轮不登记、不触碰**，**下一轮单独登记**；实测它仍然只有 **1 个**文件 —— `PREREGISTRATION_EARLYBOMB_001.md`（**4,301 B**，`14:49:14`）」。本节就是那一轮：该预注册**字节未变**（同一 `4,301 B` / sha `d4bbb00946ce…`，mtime 仍 `14:49:14`），其余 **79** 个文件都是本节首次入册。

### 27.3 与 §1–§26 不重复的说明（实测已核）

- **既有校验表 0 命中**：追加前对 `ASSET_SYNC.sha256`（**709 行**）做 `v8\.18\.0|EARLYBOMB|v8j0` 匹配 → **0 命中**（既没有 v8.18.0 的路径，也没有执行器）。
- **新增 81 条路径与既有表 0 交集**：81 个**路径**与既有 **709 个路径**逐一比对 → **路径交集 0**；新增路径在表内**各自唯一**。
- **哈希 0 交集**：81 条里有 **81 个唯一 sha256**（本轮没有逐字节重复的两份文件），与既有 **692 个唯一哈希**逐一比对 → **哈希交集 0** ⇒ 没有把任何既有行「换个路径再钉一遍」。
- **本册正文的 9 处提及都不是登记**：`v8.18.0` 在追加前的本册里出现 **9 次**（`v8j0` / `EARLYBOMB` 各 0 次），分布在 §25.2.4（「本轮不登记 `v8.17.0/` 与 `v8.18.0/`」）、§25.4.4 与 §25.5 第 10 条的引述、§25 收束段、§26.2.3、§26.4.4、§26 收束段 —— **全部只给句子、不给任何单文件路径/字节/sha**（§26.2.3 唯一给出的是顶层预注册的 4,301 B，本节 §27.2.3 已注明并接续）。本节首次给出逐文件字节身份 ⇒ **不重复**。
- **与 v8.17.0（§26）的关系**：两轮共用同一个冻结预注册池、同一个引擎 `691bfd77…`、同一批 34,387 B 的 `round2.pt`，但**证据文件是两套独立字节**（§26 的 83 条与本节 81 条无一条 sha 相同）⇒ 不构成重复登记；两轮的可比性限制由 `RUN.md` §7 第 2 条写明「**同一单因子的两个水平，不是同一实验的两次重复**」。
- **行数与格式**：709 + 81 = **790 行**（`ASSET_SYNC.sha256` 实测 **790 行**）；仍为 **LF / 无 BOM**、末行以 `\n` 结束、每行 `64 hex + 两个空格 + 相对路径（`/` 分隔、仓库根相对）`，**空行 0 条、格式异常行 0 条**、**CR 字节 0**。
- **`ASSET_SYNC.md` 本身不在校验表内**（`ASSET_SYNC` 在 `ASSET_SYNC.sha256` 里 **0 行**）⇒ 追加 §27 **不会**让任何既有行转 FAILED（与 §21.3 / §22.3 / §23.3 / §24.3 / §25.3 / §26.3 同一口径）。
- **Git 跟踪状态**：本轮按硬约束**未** `git add` / `git commit`（没有执行任何 git 写命令），分支仍为 `v400`；`Document/Update/Alpha/v8.18.0/` 在 `git status` 里仍是 **`??`（未跟踪）** ⇒ 这些证据**无法由 Git 重建**，本节 + `ASSET_SYNC.sha256` 是它们目前的字节身份记录。

### 27.4 校验结论

#### 27.4.1 `sha256sum -c`（WSL，追加前后各一次）

命令（与 §22.4.1 / §23.4.1 / §24.4.1 / §25.4.1 / §26.4.1 同一口径）：

```bash
cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -c ': OK'; sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -v ': OK'
```

* **基线（追加前 709 行）**：**708 OK / 1 FAILED**。
* **追加后（790 行）**：**789 OK / 1 FAILED**。`789 + 1 = 790` 校验行全部参与。
* **新增 81 条 81/81 OK** ⇒ 整体 FAILED 数**没有新增（仍为 1）**。
* **唯一 FAILED 仍是已知并允许的那一条**：`Document/Update/Alpha/v4.2.0/journal.jsonl`（§9.8 记录的活动文件漂移）。本轮**未改该行、未重钉、未触碰该文件**。
* **格式**：790 行仍为 **LF / 无 BOM**、末行 `\n` 结束、每行 `64 hex + 两个空格 + 相对路径`；空行 **0**、格式异常 **0**。

#### 27.4.2 与 `manifest.json` 的交叉核对（登记时实测）

| 清单 | 自述 | 实测比对 |
| --- | --- | --- |
| v8.18.0 `evidence/EARLYBOMB/manifest.json` | `file_count=78`、`total_bytes=378,430,106`、`excludes="manifest.json"` | `files{}` **78/78** 的 `{bytes, sha256}` **全等、0 处不符**；`bytes` 求和 = `total_bytes` = 378,430,106；盘上 `evidence/EARLYBOMB/**` 共 **79** 件，manifest **未列**的**只有 `manifest.json` 自身**（11,544 B）⇒ **无缺件、无多余件**（378,441,650 − 378,430,106 = 11,544） |
| 18 份 `eval_*.summary.json` 的 `out_sha256` / `out_bytes` | 每格 `{status, episodes_total, out_sha256, out_bytes}` | **18/18** 与盘上 jsonl 的实测 `{sha256, bytes}` 全等、`status` 全为 `complete`、`episodes_total` 全为 1,200 ⇒ 逐格摘要**指向的就是发布件字节** |
| 6 个 `ckpt_*.pt` 的 sha | 被 `train_summary.arms[*][*].ckpt.sha256` 与 18 份评测 summary 的 `ckpt.sha256` 引用 | **6/6 + 18/18 全等**（各 39,281 B） |
| 两份标签文件的 sha | 被 `earlybomb_stats.labels_file` / `training_pool_file`、`train_summary` 引用 | **2/2 全等**（`2e8cd349…` / 12,462,772 B / 1,341 行；`305639df…` / 8,762,992 B / 942 行） |
| 预注册 sha | `preflight.pins.prereg`（带 `bytes: 4301`）、`ckpt_assertions{,_final}.prereg`、盘上 | **三处同值** `d4bbb009…` / 4,301 B ⇒ 预注册在全轮前后**一字未改** |

**两份清单都未覆盖的内容（如实列出）**：`manifest.json` **不含**执行器 `tools/rl/v8j0_earlybomb.py`（对 `files{}` 做 `v8j0_` 匹配 = **0 命中**）；`summary.artifacts{}` **只有 13 条**，**不含** 18×2 评测件、6 个 `.pt`、`ckpt_assertions_final.json`、`audio_audit.json`、`process_residue.json` 与 `manifest.json` 自身 ⇒ 这些的字节身份**只由本节 80 条实测提供**（§27.5 第 7 条）。

#### 27.4.3 身份链、门与读数（只登记，不重判）

| 核对项 | 实测 |
| --- | --- |
| 引擎身份 | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` / match=true，出现在 `preflight.json`、`fidelity_d3.json`、`ckpt_assertions{,_final}.json`、**18/18** 份评测 summary 与 **12/12** 个采集格；与 §2 / §22–§26 记录的 `rl_v3` SHA **一致** |
| 行为策略身份 | v4.5.0 `S` 的 `round2.pt` = `4abee069…` / `a93746bb…` / `c69960fc…`（各 **34,387 B**），start 与 final 两相 **各 3/3 match** ⇒ 447 个状态的采样策略没换。本轮的 A/B 学生 ckpt 各 **39,281 B**，**6/6** 与 `train_summary` 记录一致、**18/18** 与评测 summary 引用一致 |
| **预注册身份** | `d4bbb00946ce…` / **4,301 B** —— 在 `preflight.pins.prereg`、`ckpt_assertions{,_final}` **两相**、以及盘上实测**三处同值** ⇒ 本轮是「预注册先冻结、再开跑」 |
| D3 保真门 | **5/5 PASS**（`all_pass=true`、`error=null`、`wall_s 17.713`）：HELLO `role=shadow`、DEBUG `threads=1`、前缀 **2,165 / 2,165** checkpoint **0 mismatch**（到帧 8663）、R24 **18/18** 常量、PARK `frame=8663`。**非 5/5 即整轮作废 —— 本轮未触发 `S-fork`** |
| 逐格门 | 18 格 × 4 窗 = **72/72** 个窗口 `gate.all_pass=true` + `gate_prefix{hash_checks 2,165, hash_mismatches 0, final_frame 8663}`；12 个采集格 **12/12** `gate.all_pass=true` |
| 音频（§9 硬约束） | 审计件 **89 / 89 muted / 0 audible / 0 errors**（`by_root` 35 个根、`all_muted=true`）；另 72 个评测窗的 `cfg_muted=true` 与 `cfg_sha256=cc062cfb…`（同一 60 B 模板）、12 个采集格 cfg 四字段全 0 ⇒ **全轮 89 个独立 data dir 全静音、`audible` = 0**；全部 `launch_env{SDL_AUDIODRIVER=dummy, DISPLAY=null, WAYLAND_DISPLAY=null}`；共享审计件 `v4.3.1/evidence/R14/audio_cfg_audit.json` **未触碰** ⇒ 本轮**未新增 `audible`** |
| 进程残留 | `matching=0` / `lines=[]` / `pgrep_returncode=1` / `processes_total_by_engine_name=0` / `data_dir_filter=…/.tmp_v8j0_earlybomb` / `all_pass=true` ⇒ 本轮**没有**留下引擎进程，也**没有** `pkill` 任何他轮进程 |
| **P0a / P0b / P0c（判定池 `ALIVE` 314）** | P0a **314/314 = 100 %**（门 ≥90 %，`player_state_counts={ALIVE: 314}`）；P0b **314/314 = 100 %**（`bombs_remaining_counts={3: 314}`）；P0c honoured 分支里 `spent == 1` **5,652/5,652 = 100 %**（`{1: 5652}`，`spent_two_or_more = 0`）⇒ 三项**全过**。**与 v8.17.0 的判别式成立**：v8.17 的 600 状态全是 `DYING`、按雷扣 **2** 颗；本轮 314 个判定状态全是 `ALIVE`、按雷扣 **1** 颗 |
| **P0d（视界）** | 判定池：H=120 **18/314 = 5.73 %**（不达 20 %）、H=240 **50/314 = 15.92 %**（不达）、**H=600 227/314 = 72.29 %（达门）**；`critical`（bomb1 反而更差）H=600 **63/314 = 20.06 %**、`worse_states` **45**、`delta_alive_frames` mean **153.0064** / median **151.5** / q1 0 / q3 264。整窗池（447）则是 151/183/360 = **33.78 % / 40.94 % / 80.54 %**（三档都达门）⇒ **雷买到的是「再活 3–10 秒」，不是「多活一口气」** |
| 结论口径 | `summary.verdict` = **`train`**，`train_horizon = 600`（`p0d_horizons_met = [600]`，`filter = ALIVE`，`filter_reason` 明写「P0a/P0c 在整窗失败：下沿落在死亡动画内，按雷要扣 2 颗」）；与 `RUN.md` §2 的判定表**一致** |
| **评测池化读数（3,600 episode/池）** | **A** d300 **3,592 = 99.7778 %**（Wilson 99.56–99.89 %）、d600 **1,042 = 28.9444 %**（27.49–30.45 %）、d900 **151 = 4.1944 %**（3.59–4.90 %）；**B** d300 **3,596 = 99.8889 %**（99.71–99.96 %）、d600 **1,120 = 31.1111 %**（29.62–32.64 %）、d900 **180 = 5.0000 %**（4.33–5.76 %）（区间取自 `RUN.md` §4.1，本文件不重算） |
| **P1** | 臂 B 的 600 决策活满率 **31.11 %** > v8.3.0 的 **19.44 %**（+11.67 pp）且 > v8.17.0 的 **14.1 %**（+17.01 pp）⇒ **成立** |
| **P3（负结果，重点）** | `bomb_usage.json`：6 个池 **每池 3,600/3,600 局全部 `{3: 3600}`**、**`episodes_using_1_or_2 = 0`**、`episodes_using_0 = 0`、分位全 **3.0**、stdev **0**；即 **21,600/21,600 局都把 3 颗雷用光，平均用雷 3.000 = 上限**，`RUN.md` §4.4 同表逐格印证（A/B × 300/600/900 六行「用 3 颗」全是 3,600、其余全 0）⇒ **P3 不成立：两臂都没有学会「省着用」，`aux` 目标（83.4 % 为正）等价于「只要有机会就放雷」** |
| **P4 / 并排** | 900 档 A **4.19 %** / B **5.00 %**，高于 v8.3.0 的 **0.6111 %** 与 v8.6.0 的 **2.9444 %**，但远低于 R5 脚本规则的 **17.61 %**（`diagnostics.side_by_side_survived_fraction`） |
| A/B 差 | 600 决策 B **31.11 %** 高于 A **28.94 %** 共 **2.17 pp**，但两者 Wilson 区间**重叠**（A 27.49–30.45 % vs B 29.62–32.64 %）⇒ **只是描述性差异，不是显著性结论**（`RUN.md` §4.6 已明写） |
| 「不能声称什么」照录 | `RUN.md` §7 七条：不能声称「学会了资源分配」、不能把本轮与 v8.17 直接比较而不说明采样位置不同、不能声称跨窗口泛化（w12 最差，A 18.56 %）、不能声称「通关」（30 s 档最好 31.11 %）、不能把小样本 A/B 差当显著性、不能把 900 决策与 R5 直接比、不能把「H=600 训练」外推为「提前放雷有用」⇒ 本节**照录不重判** |

#### 27.4.4 登记期零漂移（v8.18.0 目录与执行器只读）

**⚠ 本轮确实撞上了「登记期目录被重写」的场景（v8.20 那次同型）**：第一次盘点（`16:06:25`）是 **79 个文件 / 378,406,918 B**、目录内最新 mtime 是 `manifest.json` 的 `16:06:07`；此后发布方**仍在收尾**——`RUN.md` 于 `16:06:58` 由 9,257 B 重写为 **19,557 B**、`summary.json` 于 `16:07:11` 新增（28,608 B）、`manifest.json` 于 `16:07:22` 重写为 11,544 B、执行器于 `16:07:03` 变为 178,482 B。**按工单要求，第一批快照直接作废、不钉**，改为「等目录静默 ≥120 s 再取」：

| 快照 | 时刻 | 条目 | 比对结果 |
| --- | --- | ---: | --- |
| W（轮询） | `16:07:42` → `16:10:12`，**15 s 一次 × 11 次（跨 150 s）** | 81 | 相邻样本**全部零 delta**（added 0 / removed 0 / modified 0）⇒ `QUIESCENT=YES @ 16:10:12` |
| S1 | `16:07:32.058` | 81 | 基线（含 sha256 计算开始前） |
| S2 | `16:10:32.058`（S1 + **180 s**） | 81 | **S1 == S2，81/81 逐项相同** ⇒ `QUIESCENT_CONFIRMED` |
| S3 | 两个清单写完、`sha256sum -c` 全表通过之后 | 81 | **S2 == S3，81/81 逐项相同** ⇒ `AFTERWRITE_DRIFT=NO` |

| 范围 | 文件数 | 差值 |
| --- | ---: | --- |
| `Document/Update/Alpha/v8.18.0/**` | 80 | `size` 0 变化 / `mtime_ns` 0 变化 |
| `tools/rl/v8j0_earlybomb.py` | 1 | `size` 0 变化 / `mtime_ns` 0 变化（`178,482 B` / `16:07:03.062`） |
| **合计** | **81** | **81/81 完全相同，0 个变化** |

登记开始**之前**的最后一个写入是 `manifest.json` 的 `16:07:22.692`；首次锚定快照 S1 是 `16:07:32.058`（**9.4 s** 之后），静默确认过 150 s 才取 S2（`16:10:32`），此时距最后一个写入已 **3 分 9 秒**。**「等 120 秒再钉」这一条被真正触发并执行**（不是走形式）：从第一次盘点（`16:06:25`，79 文件）到最终钉入（`16:10:32` 之后）共 **4 分 7 秒**，中间作废了一次快照。

本轮对上述路径的写操作 = **0**（只读 `Get-ChildItem` / `Get-FileHash` / `os.walk` / `hashlib` / `sha256sum` / `Get-Content` / `ConvertFrom-Json` / `json.load` / `Select-String` / `ps`/`pgrep` 计数）；**未新建、未改名、未删除、未移动**任何文件；写入**只落在** `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256` 两个清单文件上。**未启动引擎**（全程 0 次 `th08-modern` 调用）、**未训练、未评测**。

#### 27.4.5 本章只登记字节身份与已在盘上的读数（**不重判**）

1. **不重算检验**：`diagnostics.json` 的 `requirement` 明写「P1–P4 是计数与分布，**检验与置信区间属于独立分析轮**；本文件无检验、无区间」。本节因此**只照录** 31.11 % / 19.44 % / 14.1 % / 5.00 % / 17.61 % 与 `train` 判定，**不重新做 Fisher、不给区间**（Wilson 区间也照 `RUN.md` §4.1 抄录，不重算）。
2. **12 格不是独立样本**：`diagnostics.note` 明写「12 个 (seed, window) 格不是独立样本；逐格数字列在池化数字旁边，**不得把它们平均成「均值的均值」**」。本节一律用**池化**口径（x/3,600、x/21,600）引用。
3. **判定池与整窗池两套都写**：`earlybomb_stats.json` 同时落盘 `p0_all`（447）与 `p0_alive`（314），`summary.p0` 还有 `chosen` / `all` / `alive` 三份。本节**照录两套**，并明写判定用的是 `ALIVE` 池；**不把整窗的 80.54 % 与判定池的 72.29 % 混用**。
4. **雷的语义已变（本轮的核心）**：v8.17.0 的标签表达「死那一刻手里那颗雷（扣 2 颗）」，本轮表达「**死前 40–80 帧、还活着时按下去（扣 1 颗）**」——`spent` 分布从 `{1: 5652, 2: 2394}`（整窗）变成判定池的 `{1: 5652}`（100 %）。所以 **P0c 是两轮的判别式**，不能把两轮的存活率当成同一实验的重复。
5. **`P3` 是负结果，不是缺数据**：6 个池 × 3,600 局的 `bombs_used_counts` 全为 `{3: 3600}`，`episodes_using_1_or_2 = 0`。这意味着「本轮学到的策略在 300/600/900 三档都把 3 颗雷花光」——与 `RUN.md` §4.6 的「预期成立」一致（H=600 标签几乎总说 `bomb=1` 更优），**不能读成「资源分配成功」**。

### 27.5 与 manifest 不符项（**只报告、不改**）

> **总结论：`manifest.json` 与盘上 0 不符。** `files{}` **78/78** 的 `{bytes, sha256}` **逐条全等**，`bytes` 求和 = `total_bytes` = 378,430,106，未列出的文件**只有 `manifest.json` 自身**与**顶层预注册**（后者在 `root` 之外）。下面列的**不是** manifest 的错，而是「manifest 覆盖不到」或「manifest 之外的证据自相矛盾 / 口径不同」的 **10 项**，按工单**只报告、未改任何文件**。

| # | 项 | 事实（实测） | manifest 侧 | 处置 |
| ---: | --- | --- | --- | --- |
| 1 | **目录在登记期被发布方继续写过（本轮真实发生）** | 第一次盘点 `16:06:25` = **79 文件 / 378,406,918 B**、最新 mtime `manifest.json` `16:06:07`；收尾阶段又写：`RUN.md` `16:06:58`（9,257 → **19,557 B**）、`summary.json` `16:07:11`（新增 28,608 B）、`manifest.json` `16:07:22`（11,419 → **11,544 B**）、执行器 `16:07:03`（178,433 → **178,482 B**）⇒ 最终 **80 文件 / 378,445,951 B** | manifest 是 **`16:07:23`** 的最终快照，与盘上一致（78/78 全等） | **只报告、不改**；第一批快照作废、静默 150 s 后重取（§27.4.4） |
| 2 | **执行器轮内至少三个状态，final 门看到的不是最终字节** | `preflight.pins.self`（`14:54:40`）= **`692db4cb…` / 164,111 B**；`ckpt_assertions{,_final}.v8j0_earlybomb_py`（`15:31:22` / `16:05:53`）= **`639f54ce…`（**没有 `bytes` 字段**）**；盘上现值 **`655d0c1f…` / 178,482 B / mtime `16:07:03.062`**（晚于 final 门 **70 s**）⇒ `all_pass=true` 的 final 断言**并不覆盖最终字节** | manifest **不含执行器**（`v8j0_` 匹配 0 命中），**不冲突**；执行器身份由 §27.1.11 实测 + 三处 JSON 提供 | **只报告、不改**；按盘上值 `655d0c1f…` 登记 |
| 3 | **多处 `producer` 字符串仍写旧的采样窗 `[30, 60]`** | 至少 **7 类**文件（`earlybomb_stats.json`、`train_summary.json`、`diagnostics.json`、`bomb_usage.json`、`process_residue.json`、`fidelity_d3_wrapper.json`、**18/18** 份 `eval_*.summary.json`）的 `producer` 里仍写 `sampling window decisions_before_death in [30, 60] = 60-120 frames`；而实际采样窗是 **`[20, 40]` = 40–80 帧**（`earlybomb_stats.sampling.window_bounds.tight = {lo: 20, hi: 40, frames: [40, 80]}`；`RUN.md` §6.1 自述「先测量后收窄」）⇒ **元数据陈旧**，不是数据错（`smoke.json` / `stats` 的 `window_bounds` 才是准的） | manifest 只记字节，不受影响 | **只报告、不改** |
| 4 | **`RUN.md` §8 自述的文件数与盘上不符** | `RUN.md` §8 写「`evidence/EARLYBOMB/`（**77 个文件**，manifest 见同目录 `manifest.json`）」；盘上 `evidence/EARLYBOMB/**` 实测 **79** 件（含 `manifest.json` 自身），manifest 自述 **78** 件（`excludes="manifest.json"`）⇒ **77 既不是 78 也不是 79** | manifest 本身**正确**（78/78 全等） | **只报告、不改**；以盘上 80（全目录）/79（EARLYBOMB）/78（manifest 口径）为准 |
| 5 | **`RUN.md` 引用了 3 个未随证据发布的运行件** | `RUN.md` §8 提到 `collect/firstpass_min300/`，§6.1 / §6.7 提到 `smoke/labels_smoke_seed0_w12_EARLY30-60.partial` 与 `smoke/labels_smoke_seed0_w11.jsonl.partial`；实测这三者**只在运行目录** `.tmp_v8j0_earlybomb/{collect,smoke}/`（`139,982 B` / `194,193 B` / 子目录 12 件），发布树 `collect/` 只有 **12 个 json**、`smoke/` 只有 **1 个 json**，**无子目录、无 `.partial`** | manifest 只登记**已发布件**，与之一致 | **只报告、不改**；按 §27.2.1 不入册 |
| 6 | **`RUN.md` §2 的 P0d「整窗（447）」列与 `earlybomb_stats.p0_all` 不一致** | `RUN.md` §2 写整窗 P0d = **18/447 = 4.0 %**、**50/447 = 11.2 %**、**227/447 = 50.8 %**；而 `earlybomb_stats.json` 的 `p0_all.by_H` 实测是 **151/447 = 33.78 %**、**183/447 = 40.94 %**、**360/447 = 80.54 %**（三档都达 20 % 门）。原因：RUN.md 该列用了 **`ALIVE` 池的分子**（18/50/227）去除以 447 的分母 | manifest 只记字节 | **只报告、不改**；判定用的是 `p0_alive`（18/50/227 ÷ 314），**结论不受影响**，但引用整窗数字时应以 `p0_all` 为准 |
| 7 | **`summary.artifacts{}` 只覆盖 13 件，且两份清单都不含执行器** | `summary.artifacts{}` 13 条 = `earlybomb_stats` / `labels_earlybomb` / `labels_earlybomb_alive` / `smoke` / `preflight` / `ckpt_assertions`(start) / `fidelity_d3` / `fidelity_d3_wrapper` / `train_summary` / `diagnostics` / `bomb_usage` / `RUN.md` / `commands.tsv`；**不含** 18×2 评测件、6 个 `.pt`、`ckpt_assertions_final`、`audio_audit`、`process_residue`、`manifest.json` 自身，也不含执行器 | manifest 覆盖其中 78 件（含评测件与 `.pt`），**两份清单互补但都不完整** | **只报告、不改**；缺口由本节 80 条实测补齐 |
| 8 | **`commands.tsv` 不是逐条命令日志，且有非数值 `exit`** | **17 行**（1 表头 + **16 条**）：12 个采集格压成 1 行、18 组评测压成 1 行、`train` 1 行覆盖 6 次训练；`exit` 列 14 个 `0` + `smoke-tight` **`4`** + `smoke-first` **`(killed: …)`**（非数值）。`smoke-tight` 的 exit 4 与 `smoke.json` 的 `p0a 14/21`、`p0c 252/378`（各 66.7 % < 90 %）一致 —— 冒烟本就**不是判定件**，判定走 `--mode stats` 的 447/314 | manifest 只记字节 | **只报告、不改**；时间锚以各文件的 mtime 为准 |
| 9 | **`protected{}` 9 件没有 `expected`，`all_pass` 对它们没有约束力** | 两份 `ckpt_assertions` 的 `protected{}` 只有 `{exists, sha256, bytes}`，**没有 `expected`** ⇒ 即使受保护文件被改，`all_pass` 仍是 `true`。**本轮实测 start 与 final 的 9 件逐条相同（0 漂移）**，与 §26 那轮（`v8g0_deathadj.py` 轮内被改）不同 | manifest 不涉 | **只报告、不改** |
| 10 | **18 份评测 summary 的 `created_at` 是启动时刻、不是写盘时刻；且有 1 份重试件** | 18/18 的 `created_at` 都在 **`15:30:39`–`15:30:41`**（`arm B d600 seed 1` 重试件是 **`15:36:17`**），而它们的 mtime 是 **`15:49:37`–`16:05:32`**（d300 `15:49`、d600 `16:02`、d900 `16:04`）⇒ 该字段是**进程启动时刻**。`B-d600-s1` 的 `tag` 是 `v8j0-eval-B-d600-s1-retry`（首次因 `FidelityError: window 11: reset hash 'af6314501cf1a19e' != recorded '7c8705fc88f94c17'` 被丢弃并重跑，`RUN.md` §6.6 自述），发布件是**重跑结果**（`survived 372`） | manifest 对 18 件的**字节身份记录正确**（18/18 与盘上全等） | **只报告、不改**；时间锚以 mtime 为准 |

**逐条核对口径（可复算）**：把 `manifest.json` 的 `files{}` 逐项与 `sha256sum` 实测比对，`{bytes, sha256}` **不符数 = 0**（78 项全过）；`bytes` 求和与 `total_bytes` **差 0**；盘上「有而 manifest 无」= **2 项**（`manifest.json` 自身 + 顶层预注册）、「manifest 有而盘上无」= **0 项** ⇒ **无缺件、无多余件**。

> **一句话结论**：本轮把 **v8.18.0（EARLYBOMB-001，提前放雷采样）80 条 / 378,445,951 B + 执行器 1 条 / 178,482 B = 81 条 / 378,624,433 B** 成对写入 `ASSET_SYNC.md` §27 与 `ASSET_SYNC.sha256`，`sha256sum -c` 由 **708 OK / 1 FAILED（709 行）** 变为 **789 OK / 1 FAILED（790 行）**（新增 81 条全 OK，整体 FAILED 数仍为 1，唯一 FAILED 仍为已知允许的 `v4.2.0/journal.jsonl`）。三个口径都记：`evidence/EARLYBOMB` **79 / 378,441,650 B**（含 manifest 自身）vs **78 / 378,430,106 B**（manifest 口径），差额恰为 `manifest.json` 自身 **11,544 B**，再加顶层预注册 4,301 B 即全目录 **80 / 378,445,951 B** ⇒ **无缺件、无多余件**；manifest 逐条比对 **78/78 全等、0 处不符**。**本轮最重要的读数**：判定走 **`S-2bomb` → `ALIVE` 过滤**，判定池 314 状态上 P0a/P0b/P0c **全 100 %**、P0d 由 **H=600 72.29 %** 达门 ⇒ `train` @H=600；**P1 成立**（B d600 **31.11 %** > 19.44 % > 14.1 %）；但 **P3 明确不成立** —— `bomb_usage.json` 的 6 池 **3,600/3,600 局全部用光 3 颗**、**`episodes_using_1_or_2 = 0`**、均值 **3.000**，**21,600 局没有一局只用 1–2 颗**，即「没有学会省雷」。**未启动引擎、未训练、未评测、未 `git add/commit`**；81 个登记文件的 `{size, mtime_ns}` 在静默 150 s 后的 S1 / S2 / 写完清单后的 S3 **三次快照逐个零漂移**；登记期目录确实被发布方继续写过（79 → 80 文件、`RUN.md` 9,257 → 19,557 B），**第一批快照已作废并按工单重新快照**（§27.4.4 / §27.5 第 1 条）。

## 32. v8.21.0 / v8.22.0 / v8.23.0 资产（2026-11-22）

### 32.1 登记范围与逐文件表

本轮登记 `Document/Update/Alpha/v8.21.0/`、`Document/Update/Alpha/v8.22.0/`、`Document/Update/Alpha/v8.23.0/` 三个**已发布**版本**整目录**（含 `manifest.json` 自身、顶层预注册稿 `PREREGISTRATION_*_001.md`、`publish.json`、`commands.tsv`、`eval_exits.tsv`、`preflight_{start,final}.json`、`ckpt_assertions_{start,final}.json`、`audio_audit.json`、`process_residue.json`、`d3_gate.json`、`smoke.json`、`train_smoke.json`、`train_curve.json`、`bomb_usage.json`、`bomb_head_seed{0,1,2}{,_bc}.pt`、`eval_*.jsonl` 等），并登记三者的**执行器**，合计 **229 条 / 55,309,941 B**。**入册 229、未入册 0**（见 §32.2）。

| 组 | 文件数 | 字节 | 目录实测 | manifest 覆盖 |
| --- | ---: | ---: | --- | --- |
| `Document/Update/Alpha/v8.21.0/**` | **110** | **46,745,090** | `Get-ChildItem -Recurse -File` 110 件 / 46,745,090 B | `evidence/BOMBRL/manifest.json` 自述 **108** 件 / 46,725,306 B（`excludes` 空、以 `manifest.json` 自身排除） |
| `Document/Update/Alpha/v8.22.0/**` | **55** | **4,102,233** | 55 件 / 4,102,233 B | `evidence/RULEFINE/manifest.json` 自述 **53** 件 / 4,091,698 B |
| `Document/Update/Alpha/v8.23.0/**` | **61** | **4,180,094** | 61 件 / 4,180,094 B | `evidence/RULETIGHT/manifest.json` 自述 **59** 件 / 4,168,922 B |
| `tools/rl/v8m0_bombrl.py`、`tools/rl/v8n0_rulefine.py`、`tools/rl/v8o0_ruletight.py` | **3** | **282,524** | — | 三份 manifest **均不含执行器**（目录 `root` 之外） |
| **合计** | **229** | **55,309,941** | — | — |

口径自洽（三个版本都成立、逐位可复算）：

| 版本 | 目录 | `evidence/<X>/` | `manifest.json` | manifest 件数/字节 | 顶层预注册稿 | 差额 |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| v8.21.0 | 110 / 46,745,090 B | 109 / 46,741,008 B | 15,702 B | 108 / 46,725,306 B | `PREREGISTRATION_BOMBRL_001.md` 4,082 B | 46,725,306 + 15,702 = **46,741,008**（+ 4,082 = 46,745,090） |
| v8.22.0 | 55 / 4,102,233 B | 54 / 4,099,405 B | 7,707 B | 53 / 4,091,698 B | `PREREGISTRATION_RULEFINE_001.md` 2,828 B | 4,091,698 + 7,707 = **4,099,405**（+ 2,828 = 4,102,233） |
| v8.23.0 | 61 / 4,180,094 B | 60 / 4,177,443 B | 8,521 B | 59 / 4,168,922 B | `PREREGISTRATION_RULETIGHT_001.md` 2,651 B | 4,168,922 + 8,521 = **4,177,443**（+ 2,651 = 4,180,094） |

文件类型构成（三个目录实测，**无 `.tmp_*`、无 `.partial`**）：

| 版本 | `.jsonl` | `.json` | `.md` | `.pt` | `.tsv` | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v8.21.0 | 80 / 45,688,000 B | 20 / 844,815 B | 2 / 24,363 B | 6 / 182,760 B | 2 / 5,152 B | **110 / 46,745,090 B** |
| v8.22.0 | 36 / 3,952,306 B | 15 / 120,898 B | 2 / 24,318 B | 0 | 2 / 4,711 B | **55 / 4,102,233 B** |
| v8.23.0 | 36 / 3,970,158 B | 21 / 175,343 B | 2 / 30,528 B | 0 | 2 / 4,065 B | **61 / 4,180,094 B** |

> **v8.21.0 的字节重心**是 80 个 `eval_*.jsonl` 共 **45,688,000 B**（占该目录 97.75 %）；其中 `actions_sampled` 字段是主要体积来源，本轮**逐个算 sha256、无一跳过**。

#### 32.1.1 v8.21.0（110 文件 / 46,745,090 B；BOMBRL）

| # | 相对路径（仓库根） | 字节 | SHA-256 |
| ---: | --- | ---: | --- |
| 1 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/audio_audit.json` | 33,898 | `231fe2efbac0fffd41aaa9cf41717f3b9e4474da99f08db6956221c52b1b4a99` |
| 2 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/bomb_head_seed0.pt` | 30,445 | `337e2228ce0a4d4172373d37d1d7a8fdc85c1a175b0cb738d68b1bbec13afde4` |
| 3 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/bomb_head_seed0_bc.pt` | 30,475 | `b52c1dfb5b01214fd4fb221dd394e4e0b51ee2d18aa0d7742f9bf546b225ff61` |
| 4 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/bomb_head_seed1.pt` | 30,445 | `2e7bf5a4dd2ea3b2b097a2fb910bcb06bfa5776a46e454fabb8a716bd2aae7ae` |
| 5 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/bomb_head_seed1_bc.pt` | 30,475 | `ce1ef1ed64f977ff02c015bec038f9d8a0ef7053189687517c1b9084724202df` |
| 6 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/bomb_head_seed2.pt` | 30,445 | `52fb5310c26882dc4b1e2a200cf94af24d6adc0665596a08feb244bad5916c4d` |
| 7 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/bomb_head_seed2_bc.pt` | 30,475 | `d0ab8204e61c2ebe48a28eb0e3c7fd1e44365f8cbc630148adb55d5198139e5e` |
| 8 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/bomb_usage.json` | 12,486 | `6b27229c5aa800ba723e07e06fd9d45a096a10f44a0e45b53b7055613725525d` |
| 9 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/ckpt_assertions_final.json` | 5,380 | `bbbd674087a05055b5239a6f60c179b1e891f7a5aa18bc9054a7d3c7ae4a7ab8` |
| 10 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/ckpt_assertions_start.json` | 5,194 | `2bae55a589821b217e4fc8bac85df719af749edd56104afc667b32186a88a16e` |
| 11 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/commands.tsv` | 4,257 | `e4b1277e5a28336e3684bfa08da02b4464c1b9ef3035dc92ccf262e68053a825` |
| 12 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/d3_gate.json` | 5,097 | `dd49e6b59262dfbb95672e1d83122c0b87a80ccd3fa8141cc3a89dffa3c4477a` |
| 13 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_exits.tsv` | 895 | `99c07033164d481bcb004ad9a58f2d4482eaa4e089183478eb068c3fa705c2b8` |
| 14 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d300_w11_seed0.jsonl` | 65,705 | `d9b07dd196e8a6ad964383256158cd41b58fa12cd6eb92ad16e3ba318cbb80f8` |
| 15 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d300_w11_seed1.jsonl` | 65,988 | `3df07c3687bad38e1946202f5ea0c2bf97df7c92bfa12ca8e8077ff8472b8919` |
| 16 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d300_w11_seed2.jsonl` | 66,008 | `2ef3c6abbd469d559058836430cb92709071bf1e9be5c7aa5fdcde6ab6852779` |
| 17 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d300_w12_seed0.jsonl` | 66,066 | `ec02ce606ed543aaece3ad140ca1c8072451050411fe324f9c6bc390f382a13e` |
| 18 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d300_w12_seed1.jsonl` | 65,866 | `ebc5e8e22ba1294c511d75c1df1753c8b536bea26ebe592891cd8e5582816be8` |
| 19 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d300_w12_seed2.jsonl` | 66,505 | `eda8dbfb64050dadaafd1d4fad08a1213418d73c40b16e7777c53300c56bbfc2` |
| 20 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d300_w17_seed0.jsonl` | 98,889 | `f69133528221f518a8fda593a7ebcf5feb0bad34da666d1d4ed2aeb37c8426cc` |
| 21 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d300_w17_seed1.jsonl` | 64,629 | `a4ea476b0cd4a0663aaa13bc58ea4d5c7f62e2e638f88f034b7b6c1ba4bc5379` |
| 22 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d300_w17_seed2.jsonl` | 64,865 | `7d26f3493881cded40feae4546774e30558bfb465f06f5fcd2d1a6d30a3fb26f` |
| 23 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d300_w18_seed0.jsonl` | 65,009 | `13a4e9cfc28d01db6b9dca1615cffadbf7a4462eb9ca0af77a2c8f6905f0b0fc` |
| 24 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d300_w18_seed1.jsonl` | 64,865 | `c7cc6da719d022deabff3f921b0ec2a262bc22f64d8cbd3c7ef62e91dcc65e8e` |
| 25 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d300_w18_seed2.jsonl` | 67,968 | `88d2d6149c186b2e66c5874c8491cef09624f1a496a8e7c0cd89eaadd065544e` |
| 26 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d600_w11_seed0.jsonl` | 90,065 | `0c74cfceed0f8d291213aa86d650dfb9a64f080bd6ee4071942a33e2d70d2dda` |
| 27 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d600_w11_seed1.jsonl` | 90,447 | `3418fb9464df52bb141a832d0a715c62a32b779cbb6b03e1cf1a3188371dba5f` |
| 28 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d600_w11_seed2.jsonl` | 87,286 | `0258edff10f998f8bb43b8dded482130a73a674ff905ca0bd747f60cba830636` |
| 29 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d600_w12_seed0.jsonl` | 90,963 | `700c36c9b6a0efed831a00c81bd44a12846254339a17d8fe1780179c1bdfbfb1` |
| 30 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d600_w12_seed1.jsonl` | 87,984 | `e40793274922fb8ff2c3cd5d0a345a857cda81bb406ac12bfaf7a95c485ada73` |
| 31 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d600_w12_seed2.jsonl` | 88,684 | `a7b828b676c3310b6756762aed27949531d3f5ed117b08c0230c17d03fa9b5cf` |
| 32 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d600_w17_seed0.jsonl` | 133,646 | `4ede00f6fa16cd5cfa37892d2a871daea89694b61478c7652e5155cbdbd7c3a4` |
| 33 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d600_w17_seed1.jsonl` | 86,607 | `43ca2719c4c6f31974181ffe45c495f04cefcb6a70a667e9cf8a036b6c191848` |
| 34 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d600_w17_seed2.jsonl` | 88,083 | `a0cdc266971a8471ede46be4cb4a31609899c28b8d98657ab0b45911fbea8266` |
| 35 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d600_w18_seed0.jsonl` | 86,991 | `b1250765cadba56d7167fa2e86f948a83f2bcbe27c6a3573ae7df14de9fcb17c` |
| 36 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d600_w18_seed1.jsonl` | 88,185 | `caec8e1fd56cb0d2e807f78983efb9038f8b58f2dd0378f72c4373e035541f37` |
| 37 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d600_w18_seed2.jsonl` | 92,106 | `eae04c20cbcde845cd6f0cbba3bfa19952ad574a001ce389c85cd2134af2684b` |
| 38 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d900_w11_seed0.jsonl` | 108,508 | `0755e16ad37a80ffa26dc27c7bb0c5753832140347354272b201dac91dd0121b` |
| 39 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d900_w11_seed1.jsonl` | 113,928 | `64dfbb704b2c00e12bfe6bcbda97a5c47bb7152719548f982c5fdc877bb542da` |
| 40 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d900_w11_seed2.jsonl` | 97,306 | `0f0076c39e788a39ee737229ea896acec89d563d0d230b7e300c1b3ea3aed0e9` |
| 41 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d900_w12_seed0.jsonl` | 103,906 | `3287b5fd0b42dfad7fc607d57da717f08df654fd888202fde762f08b5e1d912c` |
| 42 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d900_w12_seed1.jsonl` | 92,946 | `3de1aceaf61fad8f56ebe044098d51dd25e466ac09f8809b314e5df5959719aa` |
| 43 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d900_w12_seed2.jsonl` | 96,788 | `5c0f2cddfcf59d8342c7f47703354afbb27d8e913f2f22658e0b779302225b01` |
| 44 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d900_w17_seed0.jsonl` | 133,644 | `dd48638c53fd78abac9984d111c6463461485fda6d253bbba125e491134ffc7c` |
| 45 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d900_w17_seed1.jsonl` | 95,224 | `9327c3c12bb08ca414e1e0ab16c94bc1e38bb8e2e55bb23bb35becb59b89928e` |
| 46 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d900_w17_seed2.jsonl` | 93,202 | `aef193c619104fe2f0719db2d54440c3bf5724ae48a9b59ee947484bf465cb3b` |
| 47 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d900_w18_seed0.jsonl` | 108,608 | `a424d4e63713c8c27a50db43bb7ec8a8ab569a51681699d76aa770847d85f48f` |
| 48 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d900_w18_seed1.jsonl` | 104,846 | `3b173e206925b6918f3ec9be536304ca689cbfbaa9aafd08c53343cfda9fd7b5` |
| 49 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_R0-argmax_d900_w18_seed2.jsonl` | 104,305 | `02584a744c4c9c55175ac2edb5f5f586bfda8f8ec348d4cf345b67fca9e81f6c` |
| 50 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d300_w11_seed0.jsonl` | 1,018,303 | `905c6835228d44c114a534b45cbd9479ad718e92c1a2be6d539795697399daf0` |
| 51 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d300_w11_seed1.jsonl` | 1,022,506 | `744972de92931946f360ffd555236d5a5fcaae3c6096e44c400890c607b35e00` |
| 52 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d300_w11_seed2.jsonl` | 1,015,833 | `e8f492fc0770536e23170505c1aa8820ef55669801c27cd6970928699a6164f5` |
| 53 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d300_w12_seed0.jsonl` | 1,003,484 | `ffaa3f29c1d568650be0126f070c7480a14d952466cf87f93d614b6f9aafe303` |
| 54 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d300_w12_seed1.jsonl` | 1,017,065 | `fb39468b89efc0dc4a31b75fe569e73ae85e8e405fd733ea081fe36e475a12d1` |
| 55 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d300_w12_seed2.jsonl` | 1,011,150 | `57a2f67f82c581399aa96c0ed9cd126a7bb444f9be6d3563e471d4a47352dd1c` |
| 56 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d300_w17_seed0.jsonl` | 1,003,471 | `a2bcebeedbdb18c18c15fd2b5928a5628d34b3da9cf9f020c7eb85cea40f828c` |
| 57 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d300_w17_seed1.jsonl` | 904,712 | `55d04e7341c707c00e700ac51d0da2d53f2c7008968bf47892386f15b76a1010` |
| 58 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d300_w17_seed2.jsonl` | 931,373 | `351d0fe8b9c18d8d470c0145e3d6e10e34a1d88e6f02a613c71d8d6d191fb56d` |
| 59 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d600_w11_seed0.jsonl` | 1,380,169 | `13ff971f856c95f405a35743f1c5f38962662093212e6ffe4731ed64ef761875` |
| 60 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d600_w11_seed1.jsonl` | 1,449,896 | `6ce8c082ed565fe07b647b6f9e510af97941eadfcb5d40385afd79f419b8a341` |
| 61 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d600_w11_seed2.jsonl` | 1,373,879 | `5e2a6900f241df3e9da9d3320827b7e4e649ffd92856978b0a17639236d97a22` |
| 62 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d600_w12_seed0.jsonl` | 637,871 | `dfa7134956423733c4fdb24557008d8c277db6ff6d2fe8344a88d6356867f488` |
| 63 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d600_w12_seed1.jsonl` | 710,913 | `3bc2825465ce610e7022c035d2d8f43057192528a0d0e32afe862bf8125ada88` |
| 64 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d600_w12_seed2.jsonl` | 630,630 | `2c7540401c5ace6c269b7bf24add7a31bd93d5a9d64a7e4c9b167d7296c46d15` |
| 65 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d900_w11_seed0.jsonl` | 1,528,863 | `be49a2b7c11895b78cbd1f900659d1e78dd8485c861f23b50e1bf641d9d22e6c` |
| 66 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d900_w11_seed1.jsonl` | 1,789,745 | `b9079771dc3af23f0d5ebad2d4aa458be3484c4ca4bd83a9b1462e84a9f92c34` |
| 67 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d900_w11_seed2.jsonl` | 1,570,167 | `32ac819488a95058a3668fb44f1650b989aa2020cf4100d91643f09a22920baf` |
| 68 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d900_w12_seed0.jsonl` | 293,608 | `9e1ec72371aac61276e6d66d2079ea70ca278835c4e093fe95e3631dc9714955` |
| 69 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL_d900_w12_seed1.jsonl` | 418,104 | `453115342216c31ad350a70f49ed3c458b7e8b452629420204b1d4e8d4dd2e09` |
| 70 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d300_w11_seed0.jsonl` | 1,019,827 | `3e377c6bdad6da8bf93cecdac4d3d8a5f755666e05cf6243bb1d5dce3b0a6f96` |
| 71 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d300_w11_seed1.jsonl` | 1,023,705 | `fa1c37c2cf52137651658c01c806861866898fce322daa90a31dbf0848c1901c` |
| 72 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d300_w11_seed2.jsonl` | 1,024,027 | `c7e8b97a8b149662abab9ba5005962fd5c26af2e25ce0f3150b617a7d05885c4` |
| 73 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d300_w12_seed0.jsonl` | 1,001,527 | `d8ec2d432acc6e39992ce18a8962da8d93a02471b9ef9701bc22291222c92246` |
| 74 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d300_w12_seed1.jsonl` | 1,017,414 | `cc3f62fba05fa3083e569353ce6858970d12ae12f6e277b32b9edc3839c37b23` |
| 75 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d300_w12_seed2.jsonl` | 1,007,223 | `4cd30dfbe690eeb526ec2ce2181de27c7b62bb95628fd6dce5b74c44a88f60b7` |
| 76 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d300_w17_seed0.jsonl` | 1,141,255 | `4093428122c8aadfc88a329bac6c2921c5e1e88cfdbb0f0848d6dec32257302d` |
| 77 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d300_w17_seed1.jsonl` | 1,010,155 | `6d8dbd84392f08d986066df2ce2b5baa055686459830451d036c7c2eb22c98fd` |
| 78 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d300_w17_seed2.jsonl` | 1,024,029 | `392695ffc98104e46c40c78d188b36daa0870130bed91ab2c31ab581be3e95c3` |
| 79 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d600_w11_seed0.jsonl` | 1,389,720 | `9488f916c4ecf40f1be900ed935b5f84eba9e46042d80a19c13312e138a77d1a` |
| 80 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d600_w11_seed1.jsonl` | 1,391,844 | `79108679f2ff6839c504f7ee4a26383ad275f7d725ca7033d4eff1a7a2829159` |
| 81 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d600_w11_seed2.jsonl` | 1,369,639 | `2eb286e1b9068e8d27110576c5128c14add4fa191ccd76343d6eaa20820854bc` |
| 82 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d600_w12_seed0.jsonl` | 703,800 | `0e24a28a51f6c46bd67d69c83c20012324f634fb14c40ceaeef24102fc48d5c1` |
| 83 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d600_w12_seed1.jsonl` | 686,363 | `7a052adbab406a8caed989ba8c2d6f6fa2d317d03856d37e20f09d315f3394f1` |
| 84 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d600_w12_seed2.jsonl` | 737,281 | `f0528bb39e64a6c8a8822cc0ba918b6ace742448118d98981281ebd0877695e0` |
| 85 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d900_w11_seed0.jsonl` | 1,522,925 | `ab4bf32ad3a45d1ec65f617d4c6b5c00d3ed46f74b524cedbab187bfd3a53dfb` |
| 86 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d900_w11_seed1.jsonl` | 1,745,528 | `d2cc0119fadaa6eb3f86a03ffc6929f9cec5eae3a0b9102ef010c006921fc388` |
| 87 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d900_w11_seed2.jsonl` | 1,617,719 | `6451f3f4f8178b2f1b1745ddfd1fdeaa44c336abbd57b500c358d49ef0060169` |
| 88 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d900_w12_seed0.jsonl` | 344,160 | `e9ff3f653f3b728ad997c341908cf43df6222f639efd8e6a7ec0680f4139c3cd` |
| 89 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d900_w12_seed1.jsonl` | 391,277 | `8bba1f9dfac74d6bc2a14ee3aac92daa2f12ac60933821cbbb2ac70278b81e69` |
| 90 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_RL-argmax_d900_w12_seed2.jsonl` | 188,384 | `51dddaa0e79f0338e85de1f1266977a77fe73877c1fc1e83f841ae9e78227595` |
| 91 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/eval_smoke.json` | 9,919 | `b5e7ddfc37f961a1d66d145a714305e3cd9305c14a9604944c85aec9633290c7` |
| 92 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/manifest.json` | 15,702 | `7b42458673416a86721dae45b6e96b806924acf76ba39f0ab94e1732ca4f23d0` |
| 93 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/preflight_final.json` | 668 | `c790aec16b872313ac9f45d9704c2c029cfb0b640f0a2bf59e31a4482f683e37` |
| 94 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/preflight_start.json` | 666 | `300fe24f4f54c96f676ff0d3438743b13e3ecfc8a29f5dc39f67bdf830d66e57` |
| 95 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/pretrain_seed0.json` | 5,132 | `4c40a96fbbc56cfb01caf719af01c7379e0f3e146e77d6fe0237895f4961213b` |
| 96 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/pretrain_seed1.json` | 5,115 | `1c04fd35f7754545cbfc80ae0ba75f4905923506c3c53c75f922d8401d4d26e9` |
| 97 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/pretrain_seed2.json` | 5,123 | `6128046aef5f0b8427ec81c336fa50fa621529fa221ab12d9edffc4f4e153cd9` |
| 98 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/process_residue.json` | 538 | `8adccee657ed106aad51803eadc6a0f19bcd3172b6f6c62d3e9e2320ab2e09cc` |
| 99 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/publish.json` | 26,625 | `5f6beccb14817b5f2784e09ce3cb5cd92e40aff4b9f36c4c6c257cd33bac353e` |
| 100 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/RUN.md` | 20,281 | `f091983568f204f7c86d94df63ea13e58086c3b2fac6336eeebdb822e25f9c10` |
| 101 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/summary.json` | 113,721 | `c574c1f4ba592172a6f263aca26a99e13ef2995355604cc23a1fc39cd8b2bebb` |
| 102 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/train_curve.json` | 6,753 | `76d3d6def46d27b5251d2ccedb90e032218e575a8754aaf2488fe42ff4eefb8d` |
| 103 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/train_rows_seed0.jsonl` | 155,935 | `eb9b41bcb61748f35db865801afc07c15d5b66ce4a18af28f7709179efea5efe` |
| 104 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/train_rows_seed1.jsonl` | 139,156 | `f12fb19ed7b33c4d017e82d170febfc161ddc833f9cd80fcc72627fbe0cd9f20` |
| 105 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/train_rows_seed2.jsonl` | 136,744 | `c3244e8e0dbb9cff6ecb513b830e67dec357a8434f9419d149b80ff2a341b98a` |
| 106 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/train_seed0.json` | 223,968 | `010389ce3d297c7ff5e6c0c653060b59ca2f1fb1c976b9f1c114e90af2298e44` |
| 107 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/train_seed1.json` | 185,071 | `5d72c6f39a166858a7a35c027b4b56fb8f0847ddba4b0dd78d7b839dce26f75b` |
| 108 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/train_seed2.json` | 180,056 | `a3292720ba188850df7b03cd4c500654740a42999ac76c13013bf530e151db7f` |
| 109 | `Document/Update/Alpha/v8.21.0/evidence/BOMBRL/train_smoke.json` | 3,703 | `9318a6c9ca24b38dc21b3c5b1f6beb641d31255a45a6a5d8dde5a8b18e945a7a` |
| 110 | `Document/Update/Alpha/v8.21.0/PREREGISTRATION_BOMBRL_001.md` | 4,082 | `004b658898cddf920401e720a09f3b1f407d614c0e88e24286df1acf539f31b3` |

#### 32.1.2 v8.22.0（55 文件 / 4,102,233 B；RULEFINE）

| # | 相对路径（仓库根） | 字节 | SHA-256 |
| ---: | --- | ---: | --- |
| 1 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/audio_audit.json` | 11,146 | `c1fbf1c5afba4f66f6ea74000a78d26fb1ef44524ed278a50415fbe4375cbba9` |
| 2 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/ckpt_assertions_final.json` | 5,257 | `4b2c44ebac31a1d25c5476a1c3496e313e98ca30a69e1bdafcfb0f3794cfe94a` |
| 3 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/ckpt_assertions_start.json` | 5,257 | `fd6ba3a8da450b8f4cdf9a346e92f70476c07c5b52d264a6b4e02e4508ad4cd0` |
| 4 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/commands.tsv` | 3,769 | `de04bdb8873c5a6a44aeffaf12b1e03b7a3e70624666157355508a7dba1c455e` |
| 5 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/d3_gate.json` | 5,107 | `c28c895bdc5912bdce3f8d946e31bab225e56ee6ecc1655fce8e5797a74d4ca6` |
| 6 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_exits.tsv` | 942 | `165d49941d491b43ac3ed5911f7e3ddb838fda07369b4030b2cfba28ef397001` |
| 7 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F1_d900_w11_seed0.jsonl` | 111,525 | `51e740f391413b9772efa0d1393306819b8c3336848180250ceec8b88a013771` |
| 8 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F1_d900_w11_seed1.jsonl` | 111,522 | `317ad99ccc3a5cd08dc3902ae88a22a05ed110096b594bd84e32517852e55a74` |
| 9 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F1_d900_w11_seed2.jsonl` | 111,515 | `89656ca4182aaf785bdfd427746049841d51dac3e2766f2b25dc7e1e5e2ea4f5` |
| 10 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F1_d900_w12_seed0.jsonl` | 110,832 | `12304d911e74ad80dbb8134b37a3f74b21a3f5f054023ea860727d2e71c68421` |
| 11 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F1_d900_w12_seed1.jsonl` | 112,032 | `3d58541eecc101c0ee2602ffd37d6d823360d641b201755d2068388585d7f5ea` |
| 12 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F1_d900_w12_seed2.jsonl` | 110,868 | `961033ea558b6cabee62194793441de0832aff10b10950252b6d14123f1f4183` |
| 13 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F1_d900_w17_seed0.jsonl` | 106,336 | `40deaf70496ad760cd477cca292d151d4009d948b57a7919dbda2b888957514c` |
| 14 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F1_d900_w17_seed1.jsonl` | 104,387 | `258ee2d7fc4ec271fba588259a218100cac04996d7e1e0f2788a6cb4bffd7051` |
| 15 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F1_d900_w17_seed2.jsonl` | 103,597 | `2e3938a2860ca556b30b608dd277c40f927b8a8bf10419454aa2aedde8afdad2` |
| 16 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F1_d900_w18_seed0.jsonl` | 111,642 | `9da63909f64ae4d87d14037353ba745497111ead002e5aef597d1172aa27ac89` |
| 17 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F1_d900_w18_seed1.jsonl` | 111,238 | `fbb08babcb93a97f1e7ba34fc2d186f920967932927e122c479d8c49599f8f41` |
| 18 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F1_d900_w18_seed2.jsonl` | 112,169 | `a1f3fc149510860a9066d01c16eb52945d0e47f79f443506892922175d1ed5ac` |
| 19 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F2_d900_w11_seed0.jsonl` | 113,374 | `ffff5f10ad442b0b69f096de352d18a5901b031406a2677d4cf84bd57a9d5da5` |
| 20 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F2_d900_w11_seed1.jsonl` | 112,606 | `666bec970b52a5e7d708fdffde483fc266465d782c0e2e5e02a6bb665b7ecaef` |
| 21 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F2_d900_w11_seed2.jsonl` | 110,000 | `c678a1d11ee53b1cbc87f5c067466666548626d2ab7779ed6ff1c670c9d00ea3` |
| 22 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F2_d900_w12_seed0.jsonl` | 110,918 | `5052a70b7c74e6a620b6cd02b39b263c3709fff93f365ea89635505390056850` |
| 23 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F2_d900_w12_seed1.jsonl` | 113,183 | `75a43a2154c39df8268db49117873991f053ec381e4d7a7edb9a380fd43ebb36` |
| 24 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F2_d900_w12_seed2.jsonl` | 112,540 | `f40800d1053d380d0d5af8cc1d0c439d4276711dd333df5bf018ccc7e5935759` |
| 25 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F2_d900_w17_seed0.jsonl` | 106,237 | `02bdc68636fca13bc3e2c74c2f03ef4d7ea6ac493bfd76c00975a11dc8a3b89a` |
| 26 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F2_d900_w17_seed1.jsonl` | 107,215 | `38ce3ff8adda91c470feed4808950af728b7c06a8a67de25275869185e498a06` |
| 27 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F2_d900_w17_seed2.jsonl` | 104,544 | `0c1668d328533efdc5e1bc3a5bde30de67ffa18d1ca8f8f1b9a817b1f588d216` |
| 28 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F2_d900_w18_seed0.jsonl` | 110,958 | `468108766cfe1295e2c24fa785159a4db800e2a50d5ec1b851c1ae52f7b33ef7` |
| 29 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F2_d900_w18_seed1.jsonl` | 113,898 | `d0befee048ab33b85d47ee2fd76ecbda0f114bbab56a45ee53a7a507a44218a2` |
| 30 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F2_d900_w18_seed2.jsonl` | 112,840 | `624e39e711a028284713d20c81f21c9a17a77b0fdf6dc7b708ba8bb8c941a1f3` |
| 31 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F3_d900_w11_seed0.jsonl` | 112,063 | `4267b0bcf910338db42914c9f2c41c6c2f511ad5fe00e1c7b16afaa2ef7d72aa` |
| 32 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F3_d900_w11_seed1.jsonl` | 108,869 | `a8229671e0da00d0bbdaebff72c116c0f2b6892dc6dd77079b5bc8128342e48d` |
| 33 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F3_d900_w11_seed2.jsonl` | 110,144 | `a4310fa07640f04c71e6b30fd7aeb63cca645cacc5438da382d8fec3e10cc8f1` |
| 34 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F3_d900_w12_seed0.jsonl` | 107,116 | `ba91b16ad4a597bee4601523371a59a7f1aca7f84a8af78f4ae66a85e499d352` |
| 35 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F3_d900_w12_seed1.jsonl` | 111,038 | `a91d26b6fc9a745c42f78b69c2b2011be23f035dcc0d911e838fd2d7886bf4a9` |
| 36 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F3_d900_w12_seed2.jsonl` | 110,453 | `df2cf52068ac368f1d1163194c2fa98d766c81f1b356d3a9026f228df28e7f59` |
| 37 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F3_d900_w17_seed0.jsonl` | 105,605 | `4cdae542fea02cd9ac812324b602b37adf4fcbf06d6b33c52c2b5eca15d87ace` |
| 38 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F3_d900_w17_seed1.jsonl` | 105,632 | `e28e1b6dc1d8d9356e53a710f481c51545cb26f68fc6060b2a40d64b6e12878e` |
| 39 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F3_d900_w17_seed2.jsonl` | 101,175 | `78de5fd6a78fde97434816384f1a4e61e7383402a8811e5e151a4504f7528cdc` |
| 40 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F3_d900_w18_seed0.jsonl` | 111,741 | `a602252133c731099c0140ec1f7dd995dfa50a774334c7b6d450dd6ebb60b8b3` |
| 41 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F3_d900_w18_seed1.jsonl` | 112,043 | `e8c548b63d9665d97f761a82206ff3e55f2e72d08216f6ac1d332e2b6b1e49f6` |
| 42 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/eval_F3_d900_w18_seed2.jsonl` | 110,451 | `7508944c070f05dde151988304ddf40df9567242b3189624aa8139886d29f880` |
| 43 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/extra_numbers.json` | 3,356 | `6fc1d273dd6db67b4a6b2dab6bd6f22e64b22c35da86a27be67137afde64237f` |
| 44 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/F3_d900_s0_progress.json` | 4,526 | `f56ec9a55910469fa095a6ba7c5ec61b67a7816a574c2860c436f6ee55b3793c` |
| 45 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/F3_d900_s1_progress.json` | 4,529 | `0ea4b672de84787d87918d68eda4a1eaa2e4d602f3f1f03132a891d014f39312` |
| 46 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/F3_d900_s2_progress.json` | 4,526 | `d9fdbb5c5ceb67413bd1c6de55425a9d295342207cbd31f086cdfa677c67a90f` |
| 47 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/manifest.json` | 7,707 | `6d0aff72a30a4cbba7499de66df24f3693fac8ca333833374f3bd86d31dedec9` |
| 48 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/preflight_final.json` | 1,250 | `8e5d955efa46729abcb61d53e4c2406f4fed513723f6ba9983c8293388454322` |
| 49 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/preflight_start.json` | 1,250 | `0de369289672cb21280bb17265f7ba80550dc974d6647b64f6e0fe1511c390d9` |
| 50 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/process_residue.json` | 521 | `97c064ec3d3cf99b5da5513dbb8a76319813f3e8f461ab3111abde382bbaff3c` |
| 51 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/publish.json` | 12,547 | `76a83dd97f2d0834948f90bcb27e84765ec6f261986d39f5aee11e9d41d284ee` |
| 52 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/RUN.md` | 21,490 | `be15b758c58d1d7e5412eae08cf10e1090786280918a496c6f1c52ced41de70f` |
| 53 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/smoke.json` | 10,666 | `6a38743bb47429432bd2683bbddb4105f61641570b78e9b824707db9eec88bff` |
| 54 | `Document/Update/Alpha/v8.22.0/evidence/RULEFINE/summary.json` | 43,253 | `35c8407d1d0b31d33e711b2fff40ebf9c7457cfdfd969840a7a7683b0b048a6e` |
| 55 | `Document/Update/Alpha/v8.22.0/PREREGISTRATION_RULEFINE_001.md` | 2,828 | `4c79fc7faa7a92cbbfe466335a57f01063e06e2ca5a72dfc72ec471394646c81` |

#### 32.1.3 v8.23.0（61 文件 / 4,180,094 B；RULETIGHT）

| # | 相对路径（仓库根） | 字节 | SHA-256 |
| ---: | --- | ---: | --- |
| 1 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/audio_audit.json` | 11,223 | `31a3d95883dc162311106a13d9a88b05d3cccebe8afe6a20c5c304f9441bcc8c` |
| 2 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/ckpt_assertions_final.json` | 5,255 | `29b15df870b619c4f32e59c7eb2a65e791e8f9b97f33e1ecf580d8de05f378f9` |
| 3 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/ckpt_assertions_start.json` | 5,255 | `b5dc99c2a51fc9e1d9c2d15be86c3d9b4139a4346ccd821502fe2fd82e12307a` |
| 4 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/commands.tsv` | 3,770 | `6aaf397f3c2f711dcbe062981435254ddde5031c413e4fd1e04b0b15ec2753f5` |
| 5 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/d3_gate.json` | 5,116 | `7d9105a47b3ed2e299173b9090ed259854dfb905e746d4d5c0d1a7fbf74256ec` |
| 6 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_exits.tsv` | 295 | `71a57ae6fe5c0ea212c38a0880d766f6a3c5af8d441301ddb830df6614f2f28b` |
| 7 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T1_d900_w11_seed0.jsonl` | 114,029 | `f683507a3044a8d36bdf90db3b28a208495bfa9c9771295171647b1f8b93833e` |
| 8 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T1_d900_w11_seed1.jsonl` | 113,547 | `8d49b219770614f071a1d4587103e1b8b48c950add2074a65f0f569ec691a723` |
| 9 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T1_d900_w11_seed2.jsonl` | 113,431 | `3e48873bff4ddabdb7f84076541aaf805270dd6e39f6d8f24b08cdfc726c81a0` |
| 10 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T1_d900_w12_seed0.jsonl` | 113,350 | `ab8e0bb02225bfd90fa1309b0c3c4ee7522e10239a3c82132735460d9691586e` |
| 11 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T1_d900_w12_seed1.jsonl` | 112,119 | `5b92fee3140d661a286d4e0bbba9a85d9b85eca63fcbc98519bdd5400233e5b6` |
| 12 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T1_d900_w12_seed2.jsonl` | 111,749 | `090a8fcaa11e429659b6fde0e21701c87640197a298fe308837507df452ad3b0` |
| 13 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T1_d900_w17_seed0.jsonl` | 109,503 | `669dd9deb1fe73dbc6250d85901ac38a1821759feb7b1c3f1c6d4056482da471` |
| 14 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T1_d900_w17_seed1.jsonl` | 106,244 | `c35f5fd03a4d1d802f79fb172357842bd3111c6556ed3b50d7a4cf4a68c45e50` |
| 15 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T1_d900_w17_seed2.jsonl` | 104,414 | `aebc25ee6bb93a1603f8ff68ba680bf09051a8f92233d7bd092f7ab97aef0f28` |
| 16 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T1_d900_w18_seed0.jsonl` | 112,581 | `2e58d11f61cf381a4349e30a100d8ae696d2bb722b5f41cac2c8b7dbe5400c5a` |
| 17 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T1_d900_w18_seed1.jsonl` | 115,528 | `153deb229bd66bdf2fa2767092ddf8f9fb01483f864a90931bb9d09be29aeb47` |
| 18 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T1_d900_w18_seed2.jsonl` | 112,991 | `902240c00dde0f1c67903bd0090369627854be5e014c7794c7a6e12a01d6e17f` |
| 19 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T2_d900_w11_seed0.jsonl` | 113,556 | `29ea44da013ee73021846020a35919fe75d0235f3e223ac57995f5a9a479f8fc` |
| 20 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T2_d900_w11_seed1.jsonl` | 113,061 | `091b7b34a9dad86c5940c1411fd08e2244e8014893d50dc2ca8c24a7f15015a6` |
| 21 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T2_d900_w11_seed2.jsonl` | 115,003 | `debc9a957597824ddf3d7d947bb6c8b017037281621b51ced6ba8ab833ce5e82` |
| 22 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T2_d900_w12_seed0.jsonl` | 112,523 | `793f1f76222a6ecb56d162050b15be19b856d39444ef1fb733ac9be611b57c7d` |
| 23 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T2_d900_w12_seed1.jsonl` | 113,173 | `70f58b7702452e21d4f0bcea3571b95f2e73db11e2a711c193d8e18c4748e1af` |
| 24 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T2_d900_w12_seed2.jsonl` | 111,951 | `a6d65558297bc1d0af758b930a7e63f566a0af9dbf028e3310a9b566c96972e2` |
| 25 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T2_d900_w17_seed0.jsonl` | 109,477 | `94145a97fb14f5f1d2d46d22e773e7ddebe244703109851ff053f52c98413715` |
| 26 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T2_d900_w17_seed1.jsonl` | 111,298 | `fc8dfa8ce028955e6addc3ce9b1e923b66d7bb412d4c8161fce0dbb9018cf021` |
| 27 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T2_d900_w17_seed2.jsonl` | 102,042 | `e1fe52a3211a673349a2ad3c421f6793a1d7f144b5a910980cb04902ab38e52d` |
| 28 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T2_d900_w18_seed0.jsonl` | 113,969 | `5b9bdf19bc1bf39d034b417eccdbb48f8bdc6e8f97439d306c9f9749339b9509` |
| 29 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T2_d900_w18_seed1.jsonl` | 114,091 | `2cced92c58c72da5ad0006ce791e75ed797eed75935afb6dc57d13ee1a0f7bae` |
| 30 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T2_d900_w18_seed2.jsonl` | 111,912 | `09029efcefb8bf3c22535b7b8462dbe0ff79daf5aca9dbf9daf1b027c6072ca4` |
| 31 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T3_d900_w11_seed0.jsonl` | 110,041 | `9820ca33c862d992aed5f30b1b4f441e6e524781f7d6baa806c43f55db3a96fc` |
| 32 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T3_d900_w11_seed1.jsonl` | 108,182 | `32d3f972b53b5389fd243f81076320a5dcbd0a411ef3e8c8fd0bd302c46491f0` |
| 33 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T3_d900_w11_seed2.jsonl` | 109,634 | `706bb782419ae9d1d8c4909d29260909a48ab297dca5156fd17df94a186d2d50` |
| 34 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T3_d900_w12_seed0.jsonl` | 109,388 | `ca3e69beaa8132217fbc3d04cf280ead7f613a66203464b111e8fdd089a44a75` |
| 35 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T3_d900_w12_seed1.jsonl` | 110,725 | `4cd1bfd9c54beed64e4784d1139949a527bf93adf4be136be69804c30a20bdca` |
| 36 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T3_d900_w12_seed2.jsonl` | 108,611 | `fa5e68a157303211c95669a207c804d9516b9d7ee61fe7c3f89c84fb036b1b30` |
| 37 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T3_d900_w17_seed0.jsonl` | 103,024 | `7de1e9cab78f3c4be26b08f818d84506b0c655be6f1da98494a83416a8d7e0c6` |
| 38 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T3_d900_w17_seed1.jsonl` | 102,374 | `36ef56eb3d6b67fce50a54b6028ed8bd349421d5014306209b60d1994daa97e2` |
| 39 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T3_d900_w17_seed2.jsonl` | 99,071 | `676fd60a018237ce40f8437f11c9a64c3e1627b16bbe24ca41163be30803ad6e` |
| 40 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T3_d900_w18_seed0.jsonl` | 107,438 | `625b36bea4e4134ac75b9df6516125f179a51333c6cd604f6b961bdbca79ad79` |
| 41 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T3_d900_w18_seed1.jsonl` | 110,323 | `e47e27d345f1ba0bdd88320a756986fa3fa52f164566d1203895f57a2f321bb3` |
| 42 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/eval_T3_d900_w18_seed2.jsonl` | 109,805 | `506b69823c4f440e06d9b054b50eeb0c54d2b4015cdd9b561a20f22ac5bcaf06` |
| 43 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/extra_numbers.json` | 18,265 | `e1df2b6864a8b1e3082b6e7a06117c0a976283669d9871e55bbb27fa2a668baa` |
| 44 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/manifest.json` | 8,521 | `fdc985697385e582edffab46427115abd7dd9209fe0a42886821d499b2c2fcce` |
| 45 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/preflight_final.json` | 1,254 | `2bfc9a9bb9dbd08ba536df2035983a976fc7e933e97f2e78545e21d79514cb5f` |
| 46 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/preflight_start.json` | 2,131 | `999c04b83e91170cec81f793af34966fe781bf2bc141758ba8ba4f0d3ba43428` |
| 47 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/process_residue.json` | 526 | `94aafe92ee477703868f04cbba66dfdb79c3c724ddb27eee2051389136071bf4` |
| 48 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/publish.json` | 14,250 | `4b6d1858c0b861bdac7e3a2b07065ab156c21ea4f4f7b88175e69e34b1695603` |
| 49 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/RUN.md` | 27,877 | `bd3157ded9fdc4389f70d38c58bb642f639964b8fffbc80616434b93c600b69c` |
| 50 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/smoke.json` | 11,382 | `2327b8e132cfbaff611921a4b5c5dd0a6e49babb9e9f69b700a6421e38b5872f` |
| 51 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/summary.json` | 51,361 | `e7d11625542f215af5b90b5c0e156cacf80aacb40d4252eb88d1cd2eba8a7ebd` |
| 52 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/T1_d900_s0_progress.json` | 4,532 | `6889867f7b197aa25843925becc28b20fe2bd0520478e4965f077e825b81af22` |
| 53 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/T1_d900_s1_progress.json` | 4,534 | `af129d5e680f94ffc8a9ae2820ef44d225e3c259c857ae5d8c5c4bb712e9f018` |
| 54 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/T1_d900_s2_progress.json` | 4,532 | `ea13cecc7b04719c68c9b23b7cae9090a3c536dae851628946ef783888949b98` |
| 55 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/T2_d900_s0_progress.json` | 4,534 | `337e64a050e5d7cae6c0f100fb3cacd7a3f572fdc8377c3cd23c4ee74beb2fd8` |
| 56 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/T2_d900_s1_progress.json` | 4,534 | `94d192134a1b0e184d79f2a84f9d10f0a2a380e875689415cded6c78b9e37b01` |
| 57 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/T2_d900_s2_progress.json` | 4,536 | `02b7b5e7b7b4237e18a771e711b68c66d1fe9db24833a452e7a662a64d26ee89` |
| 58 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/T3_d900_s0_progress.json` | 4,532 | `d572ab510d644988c07f04d6b8634b30f85ed4d3aa3316d9f64e0c2f886aadde` |
| 59 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/T3_d900_s1_progress.json` | 4,536 | `046d6c4654bde10ab045592fa0ec6e53277055b069bd94262db05847085508b6` |
| 60 | `Document/Update/Alpha/v8.23.0/evidence/RULETIGHT/T3_d900_s2_progress.json` | 4,534 | `a6b0687a368efbc9556ee918eaca5b6966b9337f0cc203c9c7542d10051b13ea` |
| 61 | `Document/Update/Alpha/v8.23.0/PREREGISTRATION_RULETIGHT_001.md` | 2,651 | `36cf9cec19a52d93cbf1ff633feb8d8f1d9509a430d4b755aae82d1f07e57249` |

#### 32.1.4 执行器（3 文件 / 282,524 B）

| # | 相对路径（仓库根） | 字节 | SHA-256 |
| ---: | --- | ---: | --- |
| 1 | `tools/rl/v8m0_bombrl.py` | 116,848 | `e3d5bd63994433044026369b287d4d111cab1a0d32203ed972bb2bbda659a942` |
| 2 | `tools/rl/v8n0_rulefine.py` | 80,601 | `46f6f452687b3559f0f213a34e12afbfd4347881068aecc94c7318433007c01d` |
| 3 | `tools/rl/v8o0_ruletight.py` | 85,075 | `a8f699c4cc3073f38589e00b2aee2f2017c01d519028a2d2f60319713eaf838c` |

### 32.2 跳过说明（未入册 **0** 个文件）

**本章没有跳过任何文件**：三个发布目录内 **`.tmp_*` / `*.partial` / `*.tmp` 计数 = 0**（`Get-ChildItem -Recurse -Force` 逐项匹配，含隐藏项；三个目录的条目总数 112 / 57 / 63，减去各自的 110 / 55 / 61 个**文件**后恰为各自的 2 个目录 `evidence/`、`evidence/<X>/`，**没有第 3 类条目**）。

| 组 | 目录内条目 | 其中文件 | 其中目录 | `.tmp_*`/`.partial` | 入册 | 未入册 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Document/Update/Alpha/v8.21.0/**` | 112 | **110** | 2 | **0** | 110 | **0** |
| `Document/Update/Alpha/v8.22.0/**` | 57 | **55** | 2 | **0** | 55 | **0** |
| `Document/Update/Alpha/v8.23.0/**` | 63 | **61** | 2 | **0** | 61 | **0** |
| `tools/rl/v8m0_bombrl.py`、`v8n0_rulefine.py`、`v8o0_ruletight.py` | — | 3 | — | — | 3 | **0** |
| **合计** | — | **229** | — | **0** | **229** | **0** |

三条**不在**本章范围、且**已确认未登记**的路径（ledger 中命中 **0**）：

1. `.tmp_v8m0_bombrl/`、`.tmp_v8n0_rulefine/`、`.tmp_v8o0_ruletight/` 等**运行件目录**——它们都在发布树之外，按工单「`.tmp_*` 运行件不入册」一律不登记。本章**未进入、未枚举**这些目录；文中出现的 `.tmp_…` 路径只作为 `ckpt_assertions*.bomb_heads.path` 的**字符串引用**照录。
2. `Document/Update/Alpha/v8.24.0/`（**训练中**）与 `Document/Update/Alpha/v8.25.0/`（已出现在 `Alpha/` 下）——**均不登记**，本轮只覆盖 v8.21.0 / v8.22.0 / v8.23.0。
3. `v8.19.0/`、`v8.20.0/` 已由 §25 登记（ledger 各 67 / 52 条），本章**不重复登记**。

> 本轮**未启引擎**（0 次 `th08-modern` 调用）、**未训练、未评测**、**未触碰任何 `.tmp_v8p0_bombrl900` 进程**（未 `ps`/`pgrep` 其 pid，也未发信号）。

### 32.3 与 §1–§31 不重复（路径交集 **0**）

| 检查项 | 实测 | 结论 |
| --- | --- | --- |
| 追加前 ledger 行数 | **790** | §1–§31 已登记的基线 |
| 追加前含 `v8.21.0` / `v8.22.0` / `v8.23.0` 的行数 | **0 / 0 / 0** | 三个版本**从未登记过** |
| 追加条数 | **229** | 226 目录件 + 3 执行器 |
| 追加后 ledger 行数 | **1019** | 790 + 229 |
| 追加后**不同路径**数 | **1019** | **重复路径 = 0** |
| 与 §1–§31 的路径交集 | **0** | 新增路径全部落在 `Document/Update/Alpha/v8.2{1,2,3}.0/**` 与 `tools/rl/v8{m,n,o}0_*.py`，这 229 条在追加前**逐条不存在** |

另核对：追加后的 1019 条里出现 **0** 个重复路径（`Sort-Object -Unique` 后仍为 1019）；不同 SHA-256 值 **1002** 个 ≠ 1019，差额 **17 条**来自**同内容不同路径**（典型是早期轮次的别名对，如 `eval_d600_seed2.jsonl` 与 `eval_H1_d600_seed2.jsonl`、`RUN.md` 与 `nobomb_RUN.md` 等），**与本章三个版本无关**——本章的 `audio_audit.json` 三版**互不相同**（`231fe2ef…` / `c1fbf1c5…` / `31a3d958…`），`process_residue.json` 三版字节数也不同（538 / 521 / 526 B），`ckpt_assertions_*` 亦逐版不同。因此本节**不合并、不折叠**，229 条逐条登记。

### 32.4 校验结论

```
wsl.exe -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256"
```

| 指标 | 追加前 | 追加后 | 变化 |
| --- | ---: | ---: | --- |
| ledger 行数 | 790 | **1019** | +229 |
| `: OK` | 789 | **1018** | **+229（新增 229 条全 OK）** |
| 非 OK 行 | 1 | **1** | **0**（未新增失败） |

**唯一 FAILED 行**：

```
Document/Update/Alpha/v4.2.0/journal.jsonl: FAILED
```

—— 这是工单**明确允许**的唯一 FAILED（已知的 §4.2.0 追加式 journal，属预期内），**不是本章引入的问题**，本轮**未改、未删、未重建**该文件。∴ **本章 229 条 = 229 OK / 0 FAILED**，整体口径 **1018 OK / 1 FAILED（1019 行）**，与工单验收条件**一致**。

### 32.5 与 manifest 不符项（**只报告、不改**）

> **总结论：三份 `manifest.json` 与盘上 0 不符。** 逐项比对 `{bytes, sha256}`：v8.21.0 **108/108 全等**、v8.22.0 **53/53 全等**、v8.23.0 **59/59 全等**，`bytes` 求和 = `total_bytes`（46,725,306 / 4,091,698 / 4,168,922）**差 0**；「manifest 有而盘上无」**三版各 0 项**；「盘上有而 manifest 无」**三版各 1 项，且都是 `manifest.json` 自身**（另加 `root` 之外的顶层预注册稿 1 件）⇒ **无缺件、无多余件**。下面 6 项**不是 manifest 的错**，而是「manifest 覆盖不到」或「manifest 之外的证据自相矛盾 / 口径不同」，按工单**只报告、未改任何文件**。

| # | 项 | 事实（实测） | manifest 侧 | 处置 |
| ---: | --- | --- | --- | --- |
| 1 | **执行器轮内有 3 个不同字节，`all_pass=true` 的 final 门看到的不是盘上值（v8.21.0）** | `ckpt_assertions_start.self`（`f2b438f8…`）→ `ckpt_assertions_final.self`（`5e1f7375…`）→ **盘上现值 `e3d5bd63…` / 116,848 B**；`RUN.md` 同时引 `5e1f7375…` 与 `e3d5bd63…` ⇒ 轮内**至少 3 个版本**，最后的写盘发生在 final 门之后 | manifest **不含执行器**（`root` 之外），**不冲突** | **只报告、不改**；按盘上值 `e3d5bd63…` 登记（§32.1.4 第 1 条） |
| 2 | **同型漂移、且 final 门的值在 `RUN.md` 里根本没出现（v8.23.0）** | `start.self`（`e5827cf7…`）→ `final.self`（`0df0534b…`）→ **盘上现值 `a8f699c4…` / 85,075 B**；`RUN.md` 只引 `e5827cf7…` 与 `a8f699c4…`，**未提 `0df0534b…`** ⇒ 与第 1 条同型，但**多一处「文档未覆盖的中间态」** | manifest 不含执行器 | **只报告、不改**；按盘上值 `a8f699c4…` 登记。**这正是工单预告的「盘上现值可能与代理报告的 `a8f699c4…` 不同」——实测盘上就是 `a8f699c4…`，与代理报告一致；不一致的是 `ckpt_assertions` 里记的另外两个值** |
| 3 | **对照组：v8.22.0 执行器轮内零漂移** | `start.self` = `final.self` = **盘上 `46f6f452…` / 80,601 B**，三处**逐位相同**；`RUN.md` 也引 `46f6f452…` ⇒ **四个来源一致** | manifest 不含执行器 | **只报告**；这是本轮唯一一处「门看到的字节 = 最终字节」的执行器 |
| 4 | **`protected{}` 无 `expected` 字段，`all_pass` 对它们没有约束力** | 三份 final 的 `protected{}` 各 **9 / 10 / 10** 件，字段只有 `{exists, sha256, bytes}`，**`expected` 计数 = 0** ⇒ 即使受保护文件被改，`all_pass` 仍是 `true`。**实测 start 与 final 逐条相同（0 漂移）** | manifest 不涉 | **只报告、不改**（与 §27.5 第 9 条同型） |
| 5 | **v8.21.0 的 `bomb_heads` 在 start 是 `exists:false`** | `start.bomb_heads.{0,1,2}` 全部 `{exists:false, sha256:null, bytes:null}`，`final` 全部 `{exists:true, …}` —— 这是**训练前后**的生命周期（start 在训练前、final 在训练后），**不是漂移**；final 的 3 条与**已发布的 3 个 `.pt` 逐位相同**：`337e2228…`(30,445 B) / `2e7bf5a4…`(30,445 B) / `52fb5310…`(30,445 B)；另 3 个 `_bc` 件不在 `bomb_heads` 记录内，仅由 `manifest.json` 与 §32.1.1 登记 | manifest 覆盖 6 个 `.pt` 全部 | **只报告、不改** |
| 6 | **`engine` / `prereg` 两个 pin 三方一致（无问题，作为对照记录）** | `engine` 三版均为 `rl_v3` `691bfd77…`、`expected` = 同值、`match:true`（与 §8 已知 SHA 相同）；`prereg` 三份均为 `start` = `final` = **盘上实测**（`004b6588…` 4,082 B / `4c79fc7f…` 2,828 B / `36cf9cec…` 2,651 B） | manifest 不含顶层预注册稿（`root` 之外） | **只报告**；预注册稿已由 §32.1 逐文件登记 |

**逐条核对口径（可复算）**：把三份 `manifest.json` 的 `files{}` 逐项与 `sha256sum` 实测比对，`{bytes, sha256}` **不符数 = 0**（108 + 53 + 59 = 220 项全过）；`bytes` 求和与 `total_bytes` **差 0**；「盘上有而 manifest 无」= **3 项**（三个 `manifest.json` 自身）+ **3 项**（三个顶层预注册稿）、「manifest 有而盘上无」= **0 项**。

> **一句话结论**：本轮把 **v8.21.0（110 条 / 46,745,090 B）+ v8.22.0（55 条 / 4,102,233 B）+ v8.23.0（61 条 / 4,180,094 B）+ 执行器 3 条 / 282,524 B = 229 条 / 55,309,941 B** 成对写入 `ASSET_SYNC.md` §32 与 `ASSET_SYNC.sha256`，`sha256sum -c` 由 **789 OK / 1 FAILED（790 行）** 变为 **1018 OK / 1 FAILED（1019 行）**（新增 229 条**全 OK**，FAILED 数仍为 1、仍为已知允许的 `v4.2.0/journal.jsonl`）。三份 manifest 逐条比对 **220/220 全等、0 处不符**，`bytes` 求和差 **0**，「无缺件、无多余件」，差额恰为三个 `manifest.json` 自身（15,702 + 7,707 + 8,521 = 31,930 B）与三个顶层预注册稿（4,082 + 2,828 + 2,651 = 9,561 B）。**本章最重要的发现是 3 份执行器的字节史**：只有 **v8.22.0 的 `46f6f452…` 三处一致**，而 **v8.21.0（`f2b438f8…`→`5e1f7375…`→盘上 `e3d5bd63…`）与 v8.23.0（`e5827cf7…`→`0df0534b…`→盘上 `a8f699c4…`）都在轮内被改过、且 final 门看到的不是最终字节** —— 一律**按盘上现值登记，只报告不改**。**未启动引擎、未训练、未评测、未 `git add/commit`**，未改动任何证据文件。

## 33. v8.25.0 资产（2026-11-22）

### 33.1 登记范围与逐文件表

本轮登记 `Document/Update/Alpha/v8.25.0/` **整目录**（`Get-ChildItem -Recurse -File` 实测 **61 件 / 3,955,008 B = 3.77 MB**，17:26 发布完毕），含 `manifest.json`、`publish.json` 自身，顶层预注册稿 `PREREGISTRATION_T33TIERS_001.md`，以及 `RUN.md`、`summary.json`、`extra_numbers.json`、`commands.tsv`、`eval_exits.tsv`、`preflight{,_final}.json`、`ckpt_assertions_{start,final}.json`、`audio_audit.json`、`process_residue.json`、`d3_gate.json`、`smoke.json`、9 件 `U3-*_progress.json`、36 件 `eval_U3-*_d*_w*_seed*.jsonl`；并登记**执行器** `tools/rl/v8q0_t33tiers.py`，合计 **62 条 / 4,044,768 B**。**入册 62、未入册 0**（见 §33.2）。

| 组 | 文件数 | 字节 | 目录实测 | manifest 覆盖 |
| --- | ---: | ---: | --- | --- |
| `Document/Update/Alpha/v8.25.0/**` | **61** | **3,955,008** | `Get-ChildItem -Recurse -File` 61 件 / 3,955,008 B（3.77 MB） | `evidence/T33TIERS/manifest.json` 自述 **59** 件 / 3,943,754 B（`root` = `…/v8.25.0/evidence/T33TIERS`，以 `manifest.json` 自身排除） |
| `tools/rl/v8q0_t33tiers.py` | **1** | **89,760** | — | manifest **不含执行器**（`root` 之外） |
| **合计** | **62** | **4,044,768** | — | — |

口径自洽（逐位可复算）：

| 目录 | 件数 / 字节 | `evidence/T33TIERS/` | `manifest.json` | manifest 件数 / 字节 | 顶层预注册稿 | 差额 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v8.25.0 | 61 / 3,955,008 B | 60 / 3,952,461 B | 8,707 B | 59 / 3,943,754 B | `PREREGISTRATION_T33TIERS_001.md` 2,547 B | 3,943,754 + 8,707 = **3,952,461**（+ 2,547 = **3,955,008**） |

文件类型构成（实测，**无 `.tmp_*`、无 `.partial`、无 `.pt`、无二进制**）：

| 目录 | `.jsonl` | `.json` | `.md` | `.tsv` | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v8.25.0 | 36 / 3,762,790 B | 21 / 170,423 B | 2 / 17,619 B | 2 / 4,176 B | **61 / 3,955,008 B** |

> 字节重心是 36 个 `eval_*.jsonl` 共 **3,762,790 B**（占该目录 **95.14 %**），每个文件恰 **20 条** episode 行、**合计 720 条**（3 档 × 3 种子 × 4 窗 × 20 条），**逐个算 sha256、无一跳过**（§33.1.1 第 3–38 条）。

#### 33.1.1 逐文件表（62 条：61 件发布目录 + 1 件执行器）

| # | 相对路径（仓库根） | 字节 | SHA-256 |
| ---: | --- | ---: | --- |
| 1 | `Document/Update/Alpha/v8.25.0/PREREGISTRATION_T33TIERS_001.md` | 2,547 | `d81458a66abb571eb01e893d38d48383df715a4ee041460402bf01994cc06ace` |
| 2 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/RUN.md` | 15,072 | `42826f3e8588bd13a2bf9ca7ac798b4dff5db4a03c412732542a30d5e0186a6a` |
| 3 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/U3-1200_d1200_s0_progress.json` | 4,840 | `dfeb2ad536f1c9c429c430411845783b120ccdf36f5673966d6497363c088a5d` |
| 4 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/U3-1200_d1200_s1_progress.json` | 4,846 | `d0a2c2380bd7e0e70b4febb790aa9886757a8f7e0ec6f0c3a82840c68e5b724c` |
| 5 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/U3-1200_d1200_s2_progress.json` | 4,846 | `d82bb64a5847cb2a39cfa884fdb9ff5359c619a5cbba1e529308b7713675c861` |
| 6 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/U3-600_d600_s0_progress.json` | 4,828 | `c1dea0f0b739618ca49e156d7adc3afba1ff0ec5e24f0daa8ad75efbe23b16ff` |
| 7 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/U3-600_d600_s1_progress.json` | 4,828 | `b3e7a68b1980b56727829f340a0b7851a42a2cddc8b4dc8435d004f548a591cd` |
| 8 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/U3-600_d600_s2_progress.json` | 4,829 | `90fdba3e30e16610723b91c6406133711b7894ccc36c6d2f41ab70fdea1643b8` |
| 9 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/U3-900_d900_s0_progress.json` | 4,829 | `43a4f44a8b50bb756f99d03a44d1691bf0b82806c72eeb0780db89f919b3f5bf` |
| 10 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/U3-900_d900_s1_progress.json` | 4,833 | `e6430e5fe35e87631832e3dc4daf39a1d2b0d1542089c75c7379519ecb67e854` |
| 11 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/U3-900_d900_s2_progress.json` | 4,833 | `edf2faae76f1ec7518ac3fb26696fb49321a16b775edb9e34df6f736d65db764` |
| 12 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/audio_audit.json` | 11,098 | `173f91787f3d8d6fb5ee618cc73ea573fcc9d2766a69a639872b15529c5032c8` |
| 13 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/ckpt_assertions_final.json` | 4,985 | `2cfa697e3c833aedf6f9ec18c5bbdba8b84d611be7db5dd3a836150e22effa51` |
| 14 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/ckpt_assertions_start.json` | 4,985 | `00df92b0786f98f2973ae6a754d73c861f476c801c20715a1e8aa458884c00c9` |
| 15 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/commands.tsv` | 3,839 | `be0c2edba71c0e8fc39fa182f7b88b04ce3b2556e7cfe0f1937c14e3fad21cea` |
| 16 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/d3_gate.json` | 5,107 | `040ba6f7612e05fd07c4a4d2aebf2c346fc6fa11ee79cef0d063ae35b7fd26e8` |
| 17 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-1200_d1200_w11_seed0.jsonl` | 121,122 | `8c25b5e47e1bf1e78416a54051b60defc87d2a2a96db7cb6db53627b65b57be7` |
| 18 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-1200_d1200_w11_seed1.jsonl` | 120,073 | `d786213abc0de9b73efb768deea4f67f490093761a1e3803e62e25819917dc67` |
| 19 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-1200_d1200_w11_seed2.jsonl` | 116,305 | `b714dc1f9dd3fb7551c645424b2e9119830cc2334e21875702279beee9b5fe39` |
| 20 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-1200_d1200_w12_seed0.jsonl` | 114,827 | `308b4173d1b9f4cf21275eb626adb47038d2be13fda7096882df491f2820e84c` |
| 21 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-1200_d1200_w12_seed1.jsonl` | 113,690 | `6bc6bc6bfd2c934da11df30efe767f19e83ce9aecf18975835a1b7a3f1c38fc5` |
| 22 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-1200_d1200_w12_seed2.jsonl` | 114,470 | `4d6d0c5b59b71b9fd80da0873c4ee92da21c05a05944fd9001c376a6b1e78a82` |
| 23 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-1200_d1200_w17_seed0.jsonl` | 106,254 | `33f879139f63bfd01ba3b36cab410749d4690f70e63cde236cc9b700eaf797ca` |
| 24 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-1200_d1200_w17_seed1.jsonl` | 102,213 | `4e76b63b0c4deea523abc055e6ff1e31dd40762fb531817d1c6e44bd7bc70330` |
| 25 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-1200_d1200_w17_seed2.jsonl` | 100,687 | `00a8622406f243b781ac2e6b0cb1dbe3590c5852182200facd46c44ca6d883bb` |
| 26 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-1200_d1200_w18_seed0.jsonl` | 121,299 | `f06bf91e0424beeac740709979733a6abf3cd766d2ddba2a6958d8a71d848985` |
| 27 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-1200_d1200_w18_seed1.jsonl` | 124,028 | `a3b00672c442819c3a6cf7580a92967e80ec659635d6c892ebeee084845c60af` |
| 28 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-1200_d1200_w18_seed2.jsonl` | 111,615 | `ed37d4ed6b6c9ff9084aca59f5a28087d7b8006074327be4972b083f08ec581e` |
| 29 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-600_d600_w11_seed0.jsonl` | 92,085 | `eaf64437d890b0733c832b116a0cf9b90fe706588ebd430b7515561b2d201848` |
| 30 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-600_d600_w11_seed1.jsonl` | 92,758 | `7b21ceefba353d564e912b06c0c439eae7868b8d3aa230685ad80d7281580f60` |
| 31 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-600_d600_w11_seed2.jsonl` | 93,244 | `b7eeaeb786621f09642c943a63eb0bbe8b07767a405cddaa7010e28551679552` |
| 32 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-600_d600_w12_seed0.jsonl` | 91,792 | `49db082d7844c22965fd78ad071d222f20864721e14d199c72cd121d0c9f4db4` |
| 33 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-600_d600_w12_seed1.jsonl` | 91,974 | `7b71396e7fc3fb6408cc0e40974b16c3a506b0a49c0dbfa9ef1d81e1edcb2b49` |
| 34 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-600_d600_w12_seed2.jsonl` | 92,536 | `337e3db498dcfdc46bfbb172a84ebdbd074d5ab90b81bc5db58872836e0e5cda` |
| 35 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-600_d600_w17_seed0.jsonl` | 90,035 | `2a49e717ea7bd7fd5b793c4b01850d29a0cf13663cb04c64f8eb018fc9743136` |
| 36 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-600_d600_w17_seed1.jsonl` | 89,948 | `9bb208313c5a22534835fbbd9347c3789669670492b1ad38fdda4a24cef8191c` |
| 37 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-600_d600_w17_seed2.jsonl` | 89,504 | `d14bfacc0d93b08437e5a3e46759b96027abf3021f24f27ff8c4f96ee847b500` |
| 38 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-600_d600_w18_seed0.jsonl` | 91,731 | `44ca3b7abc7f4af1bdc51942feb2c3f41a22c83705ac92e7337b84d96fb78e3f` |
| 39 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-600_d600_w18_seed1.jsonl` | 92,195 | `ab9028591c09919a87a196bf5724f449dfdbc611b8d0d3612bd9fb0d085515d8` |
| 40 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-600_d600_w18_seed2.jsonl` | 93,078 | `eef5990692c55b44fea318ec46a6b2d077913d35ec6b00a17556f68577d9d252` |
| 41 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-900_d900_w11_seed0.jsonl` | 110,601 | `376659aa246517d5c37696f0d5bf7702514ea881954cac982cd46b8f9e138f81` |
| 42 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-900_d900_w11_seed1.jsonl` | 108,735 | `4ae350a496d1fdfd2dbb47bd1f9ddcd1636c7d695338e429b2cf609678c5b370` |
| 43 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-900_d900_w11_seed2.jsonl` | 110,198 | `b3c6f7db961e12455efd1a87fc24cb9885b7bebda2f21aee3772979fa1855571` |
| 44 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-900_d900_w12_seed0.jsonl` | 109,945 | `43381525c14799729bb4fe7d4bef901346ad8a408c3202813734abce51764e1b` |
| 45 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-900_d900_w12_seed1.jsonl` | 111,285 | `0263f2b98efae5ce0c9f06e380823d3c0d7e5bce1f90501c4870260bb6f2ffa4` |
| 46 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-900_d900_w12_seed2.jsonl` | 109,167 | `d7a2dc4fa376986e720f00ff0d94b59d67e8aa6568dc58f4cd400b5bd497191e` |
| 47 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-900_d900_w17_seed0.jsonl` | 103,584 | `570dcd2944304ffca95f1f22e369e1ef712e12f61f81f9fb099af9a14fb2f2bf` |
| 48 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-900_d900_w17_seed1.jsonl` | 102,936 | `44dd714eb822a600e550f2abf9834b55c4261d82e3e7c0e47edec3dc4f5f5d11` |
| 49 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-900_d900_w17_seed2.jsonl` | 99,628 | `f3fb14aa4f1bd0242aeb3e1cd6ad311209e71cf6c0e2c65a9edf429507afcde0` |
| 50 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-900_d900_w18_seed0.jsonl` | 108,000 | `3670307133e4e10b254d6f627403743af21f81863c66296ac9a94e7f57463ddc` |
| 51 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-900_d900_w18_seed1.jsonl` | 110,886 | `86f22fd943ed4771bd8a35ad51674f23796f73e9dbe50b5830ead51905e749cc` |
| 52 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_U3-900_d900_w18_seed2.jsonl` | 110,362 | `638d2e1b3718bf84e09886922bfb4fed69eabdfe63e08ae3d03a09a9a6671f06` |
| 53 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/eval_exits.tsv` | 337 | `846cbb95bd64a89e6cfe0b502ef2c7e9964fa421d19f2004e6fe95d360b929ee` |
| 54 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/extra_numbers.json` | 3,467 | `e0f07e297df6a90c801fd9b0514abb3eb11a03aa9ebee15dbc76b356077d5679` |
| 55 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/manifest.json` | 8,707 | `5d305e8ada4c2fd67582e5aaaee06b61c484038cb00685acfa810f31f4de8bc3` |
| 56 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/preflight.json` | 2,290 | `a34ea1c32f38ac7fa04a2d27a545744abd3a1f8290a2258e0adbac85895bd0db` |
| 57 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/preflight_final.json` | 1,331 | `27a5b61a79d9ea375c25a27f4ddeab9253861e60237bdf607df0376c2d6d7b4f` |
| 58 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/process_residue.json` | 574 | `87b980d19c7e92cea79df25db78ef578168769e2720f0ddf20fe8f48448f81a7` |
| 59 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/publish.json` | 14,583 | `2abd28a3c3f2b8e3b0c908f43e4fff6deb7e557127cf946eaf98cb4efa7cfa26` |
| 60 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/smoke.json` | 9,157 | `4eb475657483bc9139805096b07765c6e498df29ef1bc638fe4acd862500fd1f` |
| 61 | `Document/Update/Alpha/v8.25.0/evidence/T33TIERS/summary.json` | 60,627 | `94dfbbe11234e4c5f053eba4ab00f29977dc2085fef505826af9e8fa639904ee` |
| 62 | `tools/rl/v8q0_t33tiers.py` | 89,760 | `1eec3351e1298bee08ef73528c6f54bc3f372edf51f3d8d1b475b7079015fbad` |

#### 33.1.2 三档结果（登记对象的读数，供核对）

| 档 | n | 池化 | CP95（精确） | 种子极差 | 用雷均值 | distinct（均 / 最小–最大） |
| --- | ---: | --- | --- | ---: | ---: | --- |
| **U3-600** | 240 | **222/240 = 92.50 %** | [88.41 %, 95.49 %] | 3.75 pp | **2.3500** | 17.5875（16–18） |
| **U3-900** | 240 | **117/240 = 48.75 %** | [42.27 %, 55.26 %] | 6.25 pp | **2.7417** | 17.7458（16–18） |
| **U3-1200** | 240 | **30/240 = 12.50 %** | [8.60 %, 17.36 %] | 18.75 pp | **2.7375** | 17.8208（16–18） |

判定 **`S-switch`**：P1 不成立（92.50 % < 95 %，Fisher「更差」**p = 5.898e-4**）；P2 成立（CP95 与 v8.23.0 T3 完全重合，但属**确定性复现**，见 §33.5 第 5 条）；P3 成立（12.50 % vs 60/40/20 的 2.0833 %，**+10.4167 pp**，Fisher「更好」**p = 5.650e-6**）；P4 已报。`S-unified` **未成立**、`S-fork` **未成立**（D3 门 **5/5 PASS**，wall 8.481 s）。**720 行全部 `invalid=0`，每条 6 项 checks（`one_sample_per_decision`、`rule_bomb_matches_emits`、`counter_bookkeeping_closes`、`frames_consistent`、`threshold_respected`、`candidate_sampled`）全 True。**

### 33.2 跳过说明（未入册 **0** 个文件）

- **发布目录内 `.tmp_*` / `*.tmp` / `tmp_*` / `.partial` 实测 0 个**（`Get-ChildItem -Recurse -File -Force` 连隐藏一并枚举）⇒ 本目录**没有**需要跳过的运行件。
- 按工单口径，`.tmp_*` **运行件一律不入册**：本轮真实运行根 `/mnt/d/WorkShop/Touhou_RL/.tmp_v8q0_t33tiers/`（`data/`、`work/`、`eval/`、`fidelity_d3.json`）按 §8.1 属本机运行资产、被 `.gitignore` 忽略，**不登记**；同理候选头 `.tmp_v860_bombdag/ckpt/seed{0,1,2}/bombdag.pt`（`9c6d5d4a…` / `36a51090…` / `885dde70…`，各 39,069 B）与另一轮评测根 `.tmp_v8p0_bombrl900/` 也不入册。
- 未入册合计：**0 个文件 / 0 B**。

### 33.3 与 §1–§32 不重复（路径交集 **0**）

- 追加前 ledger **1019 行**，去重后仍 **1019 行**（重复路径 **0**）；
- 新增 62 条（61 件发布目录 + 1 件执行器）与既有 1019 条的**路径交集 = 0**（`set(旧) & set(新)` 实测，非目视）；
- 追加前 `Document/Update/Alpha/v8.25.0/` 前缀出现 **0** 次、`v8q0` 出现 **0** 次 ⇒ 本章**不与 §1–§32 任何一条重叠**；§32 登记的是 v8.21.0 / v8.22.0 / v8.23.0，本章是 v8.25.0，**版本目录互斥**；
- 新增 62 条的 SHA-256 **互不相同**（**62 个不同值 / 62 条**）⇒ 本章**无**「同内容不同路径」的别名对（与 §32 那轮的 17 条别名不同）；
- 追加后 ledger **1081 行**，去重后仍 **1081 行**（重复路径仍为 **0**）。

> 本机另有 **v8.24.0** 评测在跑（`.tmp_v8p0_bombrl900`，§33.5 第 2 条有它的引擎计数），按工单**本轮不登记** `v8.24.0/`，其归属为另一章 §34。

### 33.4 校验结论

```
wsl.exe -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -c ': OK'; sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -v ': OK'"
```

| 指标 | 追加前 | 追加后 | 变化 |
| --- | ---: | ---: | --- |
| ledger 行数 | 1019 | **1081** | +62 |
| `: OK` | 1018 | **1080** | **+62（新增 62 条全 OK）** |
| 非 OK 行 | 1 | **1** | **0**（未新增失败） |

**唯一 FAILED 行**：

```
Document/Update/Alpha/v4.2.0/journal.jsonl: FAILED
```

—— 这是工单**明确允许**的唯一 FAILED（已知的 §4.2.0 追加式 journal，属预期内），**不是本章引入的问题**，本轮**未改、未删、未重建**该文件。∴ **本章 62 条 = 62 OK / 0 FAILED**，整体口径 **1080 OK / 1 FAILED（1081 行）**，与工单验收条件**一致**。

**用雷读数（工单指定引用点）**：`35/15/3` 在 **600 决策（20 s）** 档平均只用 **2.3500 颗雷**（564 颗 / 240 条），直方图 **`{0:0, 1:0, 2:156, 3:84}`** —— **156 条只用 2 颗、84 条用满 3 颗、0 条只用 0 或 1 颗**；逐位复算 `2×156 + 3×84 = 564`、`564 / 240 = 2.3500`，与 `summary.verdict.P4.per_tier["U3-600"]` 的 `mean = 2.35`、`median = 2.0`、`stdev = 0.478` 一致。对照：同一三元组在 **900** 与 **1200** 档分别是 **2.7417**（`{2:62, 3:178}`）与 **2.7375**（`{2:63, 3:177}`），三档的 `0 颗 / 1 颗` 条数**都是 0**。

> **两条口径并列记录，不合并**：工单把这一档的读法定为「本线第一次真正做到省雷」；而 `RUN.md` §4 对同一个 2.3500 的读法**相反** —— 原文「20 s 里雷还没花完就已经打死了 —— 省雷不是优点，是**没活到该用雷的时刻**」。本章按实测登记**数值事实**（2.3500、`{2:156, 3:84}`、564/240）与**作者的解释**（未及用雷）两条，「是否为本线第一次」需与 §1–§32 各 600 档逐章比对，**本章不另作断言**，读者不应只引其一。

### 33.5 与 manifest 不符项（**只报告、不改**）

> **总结论：`evidence/T33TIERS/manifest.json` 与盘上 0 不符。** 逐项比对 `{bytes, sha256}`：**59/59 全等**；`bytes` 求和 = `total_bytes`（**3,943,754**）**差 0**；「manifest 有而盘上无」**0 项**；「盘上有而 manifest 无」= **1 项**（`manifest.json` 自身，8,707 B，在 `root` 之内但不可能自哈希）+ **1 项**（顶层预注册稿 2,547 B，在 `root` 之外）⇒ **无缺件、无多余件**。下面 6 项**不是 manifest 的错**，而是「manifest 覆盖不到」或「manifest 之外的证据之间口径不同」，按工单**只报告、未改任何文件**。

| # | 项 | 事实（实测） | manifest 侧 | 处置 |
| ---: | --- | --- | --- | --- |
| 1 | **执行器本轮零漂移（正对照）** | `ckpt_assertions_start.self` = `ckpt_assertions_final.self` = **盘上现值** = `1eec3351e1298bee08ef73528c6f54bc3f372edf51f3d8d1b475b7079015fbad` / **89,760 B**，**三处逐位相同**，且与 §33.1.1 第 62 条同值 | manifest **不含执行器**（`root` 之外），**不冲突** | **只报告**；这是与 §32 **相反的对照**：v8.21.0（`f2b438f8…`→`5e1f7375…`→盘上 `e3d5bd63…`）与 v8.23.0（`e5827cf7…`→`0df0534b…`→盘上 `a8f699c4…`）都在轮内被改过、final 门看到的不是最终字节；**本轮 final 门看到的字节就是最终字节** |
| 2 | **两份 `preflight` 都自称 `phase: final`，他轮引擎总数不同（61 vs 65），`RUN.md` 引的是较早那一拍** | `preflight.json`：`created_at` **17:25:33**、`phase` **final**、`census.engine_processes_total` **61**，且内嵌 `history[0]`（`phase: start`，17:15:55，**33** 个他轮引擎）；`preflight_final.json`：**17:26:36**、`phase` **final**、**65**，`history` 为 `[]`；`process_residue.json`（**17:26:36**）同样记 **65**。`RUN.md` §0「收尾时 61 个」、§7「final：61 个」引的是**前者** ⇒ 他轮在 **63 s** 内把引擎从 61 扩到 65 | manifest 只记字节 | **只报告、不改**；两轮的 `matching` 均为 **0**、`lines_mine` 均为 **`[]`**（**本轮 0 引擎残留**），`RUN.md` 的「33 → 61」与 `preflight.json` 完全自洽，只是**不是最后一拍**；结论不受影响 |
| 3 | **`protected{}` 10 件没有 `expected`，`all_pass` 对它们没有约束力** | 两份 `ckpt_assertions` 的 `protected{}` 各 **10** 件，字段只有 `{exists, sha256, bytes}`，**`expected` 计数 = 0** ⇒ 即使受保护文件被改，`all_pass` 仍是 `true`。**实测 start 与 final 的 10 件逐条相同（0 漂移）** | manifest 不涉 | **只报告、不改**（与 §27.5 第 9 条、§32.5 第 4 条同型） |
| 4 | **`manifest` 与 `publish` 覆盖面不同（59 vs 58），差额可精确闭合** | `manifest.file_count` = **59**（排除自身）；`publish.count` = **58**（排除自身 + `manifest.json`，因 manifest 在 publish **之后**才写）。`publish.copied` 的 **58 条 `sha256` 与盘上 58/58 全等**，`count` = `len(copied)` = **58** | 两者互不矛盾 | **只报告**；闭合：**59 + `manifest.json` = 60**（evidence 全量）、**58 + `manifest.json` + `publish.json` = 60** |
| 5 | **P2 是「确定性复现」，不是独立重复（`extra_numbers.json`）** | `U3-900` 与 v8.23.0 T3 的 240 条 raw 行逐字段比对（`survived`、`alive_frames`、`bombs_used`、`decisions_executed`、`decisions_measured`、`first_death_frame`、`bomb_emits`、`distinct_cands`）**240/240 全同、mismatches = 0**；协议本身确定性（生成器只由 `seed*1000+7` 决定、每决策恰好消费一次）⇒ 同一 (档, 种子) 必重放出逐位相同结果 | manifest 不涉 | **只报告**；`RUN.md` §3.1 已**如实**标注「P2 只能证明『实现可复现』，不能当成『v8.23 的 48.75 % 被独立复核了』」，本章照录该限定，**不把 P2 当两个独立实验互证** |
| 6 | **`commands.tsv` ↔ `eval_exits.tsv` 逐行一致（9/9，正对照）** | 两表各 **9 行**（1 表头 + 9 个评测进程），`wall_s` **9/9 全等**（157.731 / 157.434 / 156.617 / 208.898 / 208.273 / 207.059 / 222.502 / 218.077 / 206.416），`status` 全 `complete`、`exit` 全 `0`，**无 `(killed: …)` 之类非数值**；每行命令都带 `SDL_AUDIODRIVER=dummy` 且 `unset DISPLAY WAYLAND_DISPLAY` | manifest 只记字节 | **只报告**（与 §27.5 第 8 条那轮「16 条压成 1 行、带非数值 exit」**形成对照**：本轮是**逐进程一行**、可逐条复算） |

**逐条核对口径（可复算）**：把 `manifest.json` 的 `files{}` 逐项与 `sha256sum` 实测比对，`{bytes, sha256}` **不符数 = 0**（59 项全过）；`bytes` 求和与 `total_bytes` **差 0**（3,943,754）；「盘上有而 manifest 无」= **1 项**（`manifest.json` 自身）+ **1 项**（顶层预注册稿）、「manifest 有而盘上无」= **0 项**。`publish.json` 的 58 条 `{name, sha256}` 亦与盘上 **58/58 全等**。旁证：`summary.json` 与两份 `ckpt_assertions` 都记 `prereg.sha256 = d81458a6…`，与盘上预注册稿（2,547 B）**逐位相同**；`engine` 在三份文件里都是 `rl_v3` `691bfd77…`、`match: true`（与 §8 已知 SHA 相同）。

> **一句话结论**：本轮把 **v8.25.0 发布目录 61 件 / 3,955,008 B + 执行器 1 件 / 89,760 B = 62 条 / 4,044,768 B** 成对写入 `ASSET_SYNC.md` §33 与 `ASSET_SYNC.sha256`，`sha256sum -c` 由 **1018 OK / 1 FAILED（1019 行）** 变为 **1080 OK / 1 FAILED（1081 行）**（新增 62 条**全 OK**，FAILED 数仍为 1、仍为已知允许的 `v4.2.0/journal.jsonl`）；三档判定 **`S-switch`**（**20 s 92.50 % / 30 s 48.75 % / 40 s 12.50 %**），`35/15/3` 在 600 档用雷均值 **2.3500**（`{2:156, 3:84}`）；manifest 逐条比对 **59/59 全等、0 处不符**，差额恰为 `manifest.json` 自身（8,707 B）与顶层预注册稿（2,547 B）⇒ **无缺件、无多余件**；执行器 **`1eec3351…` 在 start / final / 盘上三处零漂移**，与 §32 的 v8.21.0 / v8.23.0 轮内漂移形成**正对照**。**未启动引擎、未训练、未评测、未 `git add/commit`**，未改动、删除或移动任何证据文件（62 件的 `{size, mtime_ns}` 在登记前后两次快照**逐条相同**）。

## 34. v8.24.0 资产（2026-11-22）

### 34.1 登记范围与逐文件表

本轮登记 `Document/Update/Alpha/v8.24.0/` **整目录**（`Get-ChildItem -Recurse -File` 实测 **69 件 / 67,691,695 B = 64.56 MB**，17:47 发布完毕），含 `manifest.json`、`publish.json` 自身，顶层预注册稿 `PREREGISTRATION_BOMBRL900_001.md`，以及 `RUN.md`、`deviations.md`、`summary.json`、`bomb_usage.json`、`train_curve.json`、`d3_gate.json`、`eval_smoke.json`、`train_smoke.json`、`preflight.json`、`ckpt_assertions_{start,final}.json`、`audio_audit.json`、`process_residue.json`、`commands.tsv`、`eval_exits.tsv`、`pretrain_seed{0,1,2}.json`、`train_seed{0,1,2}.json`、`seed{0,1,2}.jsonl`、6 件 `bomb_head_seed{0,1,2}_{bc,d900}.pt`、36 件 `eval_RL900_d{300,600,900}_w{11,12,17,18}_seed{0,1,2}.jsonl`；并登记**执行器** `tools/rl/v8p0_bombrl900.py`，合计 **70 条 / 67,817,000 B**。**入册 70、未入册 0**（见 §34.2）。

| 组 | 文件数 | 字节 | 目录实测 | manifest 覆盖 |
| --- | ---: | ---: | --- | --- |
| `Document/Update/Alpha/v8.24.0/**` | **69** | **67,691,695** | `Get-ChildItem -Recurse -File` 69 件 / 67,691,695 B（64.56 MB） | `evidence/BOMBRL900/manifest.json` 自述 **67** 件 / 67,678,732 B（`root` = `…/v8.24.0/evidence/BOMBRL900`，以 `manifest.json` 自身排除） |
| `tools/rl/v8p0_bombrl900.py` | **1** | **125,305** | — | manifest **不含执行器**（`root` 之外） |
| **合计** | **70** | **67,817,000** | — | — |

口径自洽（逐位可复算）：

| 目录 | 件数 / 字节 | `evidence/BOMBRL900/` | `manifest.json` | manifest 件数 / 字节 | 顶层预注册稿 | 差额 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v8.24.0 | 69 / 67,691,695 B | 68 / 67,688,378 B | 9,646 B | 67 / 67,678,732 B | `PREREGISTRATION_BOMBRL900_001.md` 3,317 B | 67,678,732 + 9,646 = **67,688,378**（+ 3,317 = **67,691,695**） |

文件类型构成（实测，**无 `.tmp_*`、无 `*.tmp`、无 `tmp_*`、无 `.partial`、无隐藏件**）：

| 目录 | `.jsonl` | `.json` | `.md` | `.tsv` | `.pt` | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v8.24.0 | 39 / 66,479,228 B | 19 / 1,003,080 B | 3 / 24,098 B | 2 / 2,529 B | 6 / 182,760 B | **69 / 67,691,695 B** |

> 字节重心是 **39 个 `.jsonl` 共 66,479,228 B（占该目录 98.21 %）**：36 个 `eval_RL900_*.jsonl` **每个恰 300 条** episode 行、**合计 10,800 条**（3 档 × 4 窗 × 3 种子 × 300 条），另有 3 个 `seed{0,1,2}.jsonl` 为训练逐回合日志（148,838 / 174,572 / 213,056 B）。6 个 `.pt` 是 **3 个 BCE 暖启头 `_bc.pt`（各 30,475 B）+ 3 个训练后头 `_d900.pt`（各 30,445 B）**，合计 182,760 B，**逐个算 sha256、无一跳过**（§34.1.1 第 2–70 条）。

#### 34.1.1 逐文件表（70 条：69 件发布目录 + 1 件执行器）

| 1 | `Document/Update/Alpha/v8.24.0/PREREGISTRATION_BOMBRL900_001.md` | 3,317 | `2ead6917e702f2af4c2d7c0df4abb91e6244bfa197ccaf31ea63aea2edcead14` |
| 2 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/RUN.md` | 16,545 | `9519c1de00fc7ca4fd740bef8bab688b006e3ea6b9d15f71e0bdadbdb1575a1c` |
| 3 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/audio_audit.json` | 20,554 | `9e19669f494e66ccfe605502b0c96ebd2bb7e2850987b2df06f993a093839a7a` |
| 4 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/bomb_head_seed0_bc.pt` | 30,475 | `38e4f6e47d4ac14917d2f90c5ce544f8840a190fb8aeadc90c581f1f61936c29` |
| 5 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/bomb_head_seed0_d900.pt` | 30,445 | `eb58332da441128acf5d71723c8cbcf969b5d56cc1d85737bea6656eaaf0b17f` |
| 6 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/bomb_head_seed1_bc.pt` | 30,475 | `e662ed408ecee56a5ea5dbafd059175a8ff2ca5a345ced02d1a729ed3822f90e` |
| 7 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/bomb_head_seed1_d900.pt` | 30,445 | `d17597255227ae856d9f0b394d8654ab714d6c0be1caeee1c5f96685d00f0670` |
| 8 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/bomb_head_seed2_bc.pt` | 30,475 | `7d926f735f8b7b0f307c64d9cd6ff527d08f256edce9822fe8241fe40e6fc0b0` |
| 9 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/bomb_head_seed2_d900.pt` | 30,445 | `98845f8eb29fe744e52bc3ab092d7e8e71c459d8ef788134fa4777eb1e4f2190` |
| 10 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/bomb_usage.json` | 6,161 | `41aa1c04cba06f8558d5cf55f08e04eac189e3d36601af648840478e2cef9e82` |
| 11 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/ckpt_assertions_final.json` | 5,412 | `783e3230dd9e553a6c5283c97026ec880529ecaf60618999fdae3e17281abbdc` |
| 12 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/ckpt_assertions_start.json` | 5,226 | `b439421c404b164a0639f72383b9f15105ce41f72de29bc241f885b96f28280b` |
| 13 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/commands.tsv` | 2,124 | `08be3bbc975e457eb452cb42d86929204564ed923ee1a634439b5d0db2e957d0` |
| 14 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/d3_gate.json` | 5,127 | `7495c17c93e5dbbf33daa5c9db450213c3eecc011b24639a26ed18479ac6763e` |
| 15 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/deviations.md` | 4,236 | `ef19f7cfc73251d94bc1b28da9590e7cdfe2ff06de0b7bbbe2268e74265b156e` |
| 16 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d300_w11_seed0.jsonl` | 1,040,447 | `41f6f181b5de8b183b5d792610157df2e1879e0c79e83e1d45ffad68f8179f76` |
| 17 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d300_w11_seed1.jsonl` | 1,559,836 | `de61f7c2099c448dd035f30057bcdea0d2a28388b54b63c24f2470750cbac89b` |
| 18 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d300_w11_seed2.jsonl` | 1,724,712 | `d501f09d8f172ef48898e5f97d151fe3d863c72de4445cda78cd28d8e41c546e` |
| 19 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d300_w12_seed0.jsonl` | 1,050,594 | `f1118c619829eec8e7a611c955cf41175cbc9686177e3528f97ace7d2779fd66` |
| 20 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d300_w12_seed1.jsonl` | 1,726,481 | `dd5843d0c0c655647f412bb4b7ef600f93b344618dd930727bd705e031309117` |
| 21 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d300_w12_seed2.jsonl` | 1,809,434 | `231256f9bb4e478191da4c367e9b3a8df02dd3959c9d900866955c29a4242da7` |
| 22 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d300_w17_seed0.jsonl` | 1,077,400 | `9a7034ab73c91e927cca904f5792dea8fe4f1a3f2604018b65fe4f3c3c974fdd` |
| 23 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d300_w17_seed1.jsonl` | 1,079,828 | `0f0490e9fc975c563ae5d2d7830df82f0f83756b7c98ffc368269a6755f50641` |
| 24 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d300_w17_seed2.jsonl` | 1,349,122 | `316aa655045446e571a867ebcfaa4e33aaab890882dea0d0f747bbba111bae3d` |
| 25 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d300_w18_seed0.jsonl` | 1,024,190 | `964ef5f14539f08fea6f7b3f39eb02895691376be53f08c117c16d1bf266262b` |
| 26 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d300_w18_seed1.jsonl` | 1,247,532 | `5a48e3a14f1c8a175a350e60d5004622bd4d97c32304e3ca6c8c5fafb6c7122c` |
| 27 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d300_w18_seed2.jsonl` | 1,570,471 | `23baf395fe87d25089daa2453654aa9dcd55a059563119e8a9dbbcb259be6709` |
| 28 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d600_w11_seed0.jsonl` | 1,418,697 | `d3c04e5d80ca95f760ec9cee501c3f287655cabb5a8d7dc30379d056086f4187` |
| 29 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d600_w11_seed1.jsonl` | 2,497,490 | `2a39ce9ee052847bb651792202f5a611a7677dd9314a8a9c18af2e5f70ad0733` |
| 30 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d600_w11_seed2.jsonl` | 3,163,941 | `c6e4ece38681de698579337fc62d3b664e2ee8eac11c57dae677e7d86a90b157` |
| 31 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d600_w12_seed0.jsonl` | 1,459,789 | `243bd81e3877d51cc3cf8544d71a095a82e5d12e05aca43cdb943f3f2d814647` |
| 32 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d600_w12_seed1.jsonl` | 2,682,713 | `6ffb3085a540922561a9df305a7629a9f5327337d44991c6277a379503f795d1` |
| 33 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d600_w12_seed2.jsonl` | 2,426,709 | `e6729c354fd15f4e01d910cf3a01c641b5e61956bc72a7ad67ade954b03eb3a9` |
| 34 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d600_w17_seed0.jsonl` | 1,437,349 | `737ded9d602334a709f1b33289390fd6e23f8dad74c34481a52d3e508321e704` |
| 35 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d600_w17_seed1.jsonl` | 1,449,908 | `cf836c4aada3a0cac4939d0c5fff82757ce2108fdd8da4e1f1bfb02f32092c62` |
| 36 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d600_w17_seed2.jsonl` | 2,323,552 | `98960324156cd9078494402dd44ad589abe68698e5a8d59e4efcbeccf03181b4` |
| 37 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d600_w18_seed0.jsonl` | 1,373,007 | `c6a3c79423c2c012e610acad032c5e65844c63f031ac5a132b1ecd98963ed07f` |
| 38 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d600_w18_seed1.jsonl` | 1,805,918 | `585638b5c31589e13869aa448357d5a0937f511a74eaf6adf064128e3eefceac` |
| 39 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d600_w18_seed2.jsonl` | 2,314,218 | `03c10e186cb7e32c69cad25a84489d476288fdecc865dc3411df7487d228397f` |
| 40 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d900_w11_seed0.jsonl` | 1,555,942 | `62e712b1000f47b9977624e4e9df533c75d8492f728b712bd5ad35b94ebaacb9` |
| 41 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d900_w11_seed1.jsonl` | 2,465,269 | `cfd52cbe449fbe08a2db9a18ae828d29223add508eace6061921d5a78cb89655` |
| 42 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d900_w11_seed2.jsonl` | 3,163,950 | `34d2b70742d762e170c2a134e005f1db8b8737475af4ca515e43e9212ee0b21f` |
| 43 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d900_w12_seed0.jsonl` | 1,475,747 | `0c3aafd0e8c063c0243dd8c841b6846d0bbd3001bd01e3392b39e08829a6ce5c` |
| 44 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d900_w12_seed1.jsonl` | 2,774,354 | `42784027499bee1383ca3485668d3f04675ee02e507038a6d7f6dae5df99aee0` |
| 45 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d900_w12_seed2.jsonl` | 2,426,709 | `30ea36bd3c01919d724baa374ec859d409d10122e301b2d374aed4756c81ac4d` |
| 46 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d900_w17_seed0.jsonl` | 1,536,644 | `37686d8778a83d22465b6631a8b14e4683c9809ea59e2566abfd2244eae0a7a3` |
| 47 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d900_w17_seed1.jsonl` | 1,667,302 | `be49349a45fd8211f26a03cd1c5497a414370a7b4b6419ead921a2c2cff6a924` |
| 48 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d900_w17_seed2.jsonl` | 2,367,734 | `bb7331c41fcc74267f43c09b0c2e1751a331578d74b223c0db78e1ca0270aac7` |
| 49 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d900_w18_seed0.jsonl` | 1,644,899 | `0e4396cad3302e055052d4cca0408d26146a1102ab84825dec1868630b279e21` |
| 50 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d900_w18_seed1.jsonl` | 1,916,608 | `8b0185ed20cf501a30ad458f5d8eed03c8df3957c9fbc327311eab99d93667e4` |
| 51 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_RL900_d900_w18_seed2.jsonl` | 2,334,266 | `9ea8ad9e37c25ab1bd45140193fc7200e277eb05337a24256fdda28ddbfd8ac3` |
| 52 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_exits.tsv` | 405 | `89890786eddb56515075cf28d33cb6a9123ad851a8e3607b9d424cb73167e9bf` |
| 53 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/eval_smoke.json` | 11,612 | `20627296d0a172c3c5489f96232c3ec6ec5d2c7f842b7d25963480b1118844f6` |
| 54 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/manifest.json` | 9,646 | `150f3052ae72c9f3618b6aafe2e7db89c234bf37e31a2cb6126e91eeb8e5da6b` |
| 55 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/preflight.json` | 674 | `63271fdbf8c12a08160e7d9bc18e66aa21933e8bf038db6064c418eb2007286b` |
| 56 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/pretrain_seed0.json` | 5,199 | `d1f35c6f3b8b7f64a49e74099e1ebffc6a44eb2e17c49271d9354384786edd4d` |
| 57 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/pretrain_seed1.json` | 5,194 | `f17ef6a56a9f6499e8e521537d61942e208197e517b0ca1426b7be169fa649c7` |
| 58 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/pretrain_seed2.json` | 5,198 | `f9bc0aca3ba545c83b4abe6d3bc58512df1fa6ea9578af4fc9579988d329497e` |
| 59 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/process_residue.json` | 549 | `e0354e47d6687525eb4e8eef21fddbd0b8ec20f0eebd53e416de9503c8b3699d` |
| 60 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/publish.json` | 15,998 | `60156ff790f5ef73adb83c4f1f3172e348da7e51c938ef9ad9e4b40c35063ee4` |
| 61 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/seed0.jsonl` | 148,838 | `c73301988c603dd19c7a9985f8b880cc52376b598d6dcee5119965bbdbb00f0e` |
| 62 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/seed1.jsonl` | 174,572 | `c9b4b0d2a723f47c42ccc7efa17535f2c7b6ddfedc88e0d6723b2bfee661e498` |
| 63 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/seed2.jsonl` | 213,056 | `d0d61ea59e2b2aff20dfc832a96562fe87162d69ae3f2f415d026303fc445d22` |
| 64 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/summary.json` | 69,354 | `322fd9df1946f73ac0315c95f23cc326df3f88fba9f20f9cf56436c2182e999d` |
| 65 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/train_curve.json` | 6,730 | `7bbaec7f551515e7b10bdc41d84f07b6e6d160a28ae0b655e2d60fa1002cf8f0` |
| 66 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/train_seed0.json` | 205,633 | `4b78b9f9a1d8839c9c489f00e3f8767e00f7a8dab7d4c30b108fe83896c7183e` |
| 67 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/train_seed1.json` | 265,699 | `223506bb17801917c30b0c6ec06ee2b835ea59f4a149d30660d46410509ccdf7` |
| 68 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/train_seed2.json` | 355,383 | `73366eb20c944a5252a0d056751d82f57dc843b101f18ffdc0d5f248c7659a14` |
| 69 | `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/train_smoke.json` | 3,731 | `f55b8b035af9b2127f16dcc3c25110810e44c8626b03f416a9dbe088c4881a62` |
| 70 | `tools/rl/v8p0_bombrl900.py` | 125,305 | `71bfbeb6011179d0373896c1efdd973a760eb3f1d7adad55478cc2a6511d1222` |

### 34.2 跳过说明（未入册 **0** 个文件）

- **发布目录内 `.tmp_*` / `*.tmp` / `tmp_*` / `.partial` / 隐藏件实测 0 个**（`Get-ChildItem -Recurse -File -Force` 连隐藏一并枚举，另用 `os.walk` 以文件名前缀/后缀再筛一次）⇒ 本目录**没有**需要跳过的运行件，**69 件全部入册**。
- 按工单口径，`.tmp_*` **运行件一律不入册**：本轮真实运行根 `/mnt/d/WorkShop/Touhou_RL/.tmp_v8p0_bombrl900/`（`data/`、`work/`、`eval/`、`launch/`、`summary/`、`train_curve.json` 等）按 §8.1 属本机运行资产、被 `.gitignore` 忽略，**不登记**；同理候选头 `.tmp_v860_bombdag/ckpt/seed{0,1,2}/bombdag.pt`（`9c6d5d4a…` / `36a51090…` / `885dde70…`，各 39,069 B）、行为策略 `v4.5.0 round2.pt`（`4abee069…` / `a93746bb…` / `c69960fc…`）、以及引擎 `rl_v3` `691bfd77…` 均不入册（前三者已在 §33.2 登记过一轮，本轮不重复）。
- 发布目录**外的同名资产**只有执行器 1 件，已随本章入册（第 70 条），**不属跳过项**。
- 未入册合计：**0 个文件 / 0 B**。

### 34.3 与 §1–§33 的路径交集（**0**）

- 追加前 ledger **1081 行**，去重后仍 **1081 行**（重复路径 **0**）；
- 新增 **70** 条（69 件发布目录 + 1 件执行器）与既有 1081 条的**路径交集 = 0**（`set(旧) & set(新)` 实测，非目视）；
- 追加前 `Document/Update/Alpha/v8.24.0/` 前缀出现 **0** 次、`v8p0_bombrl900` 出现 **0** 次、子串 `v8.24.0` 出现 **0** 次 ⇒ 本章**不与 §1–§33 任何一条重叠**；§33 登记的是 v8.25.0，并在 §33.3 末尾明确写着「本机另有 **v8.24.0** 评测在跑（`.tmp_v8p0_bombrl900`）…其归属为另一章 **§34**」——本章正是该章，**版本目录互斥**；
- 新增 70 条的 SHA-256 **互不相同**（**70 个不同值 / 70 条**）⇒ 本章**无**「同内容不同路径」的别名对；且新增 70 条与既有 1081 条的**哈希交集也为 0**（`set(旧 sha) & set(新 sha)` 实测）；
- 追加后 ledger **1151 行**，去重后仍 **1151 行**（重复路径仍为 **0**）。

### 34.4 校验结论

```
wsl.exe -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -c ': OK'; sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -v ': OK'"
```

| 指标 | 追加前 | 追加后 | 变化 |
| --- | ---: | ---: | --- |
| ledger 行数 | 1081 | **1151** | +70 |
| `: OK` | 1080 | **1150** | **+70（新增 70 条全 OK）** |
| 非 OK 行 | 1 | **1** | **0**（未新增失败） |

**唯一 FAILED 行**：

```
Document/Update/Alpha/v4.2.0/journal.jsonl: FAILED
```

—— 这是工单**明确允许**的唯一 FAILED（已知的 §4.2.0 追加式 journal，属预期内），**不是本章引入的问题**，本轮**未改、未删、未重建**该文件。∴ **本章 70 条 = 70 OK / 0 FAILED**，整体口径 **1150 OK / 1 FAILED（1151 行）**，与工单验收条件**一致**。

#### 34.4.1 主表（`summary.json` → `table`，三档**全部满额 3,600 行**）

| 档 | 活满/总数 | 活满率 | CP95 | 用雷均值 | distinct（池化） | 最小格 distinct | 作废格 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| **10 s**（300 决策） | 3,595/3,600 | **99.86 %**（0.998611） | [99.68 %, 99.95 %] | 2.5211 | 3,194 | 73 | 0 |
| **20 s**（600 决策） | 1,250/3,600 | **34.72 %**（0.347222） | [33.17 %, 36.30 %] | 2.9747 | 3,508 | 247 | 0 |
| **30 s**（900 决策） | 147/3,600 | **4.08 %**（0.040833） | [3.46 %, 4.78 %] | 2.9783 | 3,545 | 263 | 0 |

`rows = expected_rows = 3,600`、`invalid_rows = 0`、`cells_complete = 12/12`（三档皆是）；`distinct` 的去重键 = 预注册 §2 P4 的 `(alive_frames, bombs_used, bomb_emits, tuple(bomb_decisions))`；**没有一格 distinct < 10**（最小 73）⇒ **P4 成立、`void_cells = []`**，`pooled_valid` 与 `pooled` 完全相同（3,600 行）。逐档 `wall_s_total` = **2,821.313 / 4,586.519 / 4,962.899 s**。

**逐窗口**（每格 n = 900 = 3 种子 × 300）：

| 档 | w11 | w12 | w17 | w18 |
| --- | ---: | ---: | ---: | ---: |
| 10 s | 900/900（100 %） | 895/900（99.44 %） | 900/900（100 %） | 900/900（100 %） |
| 20 s | 267/900（29.67 %） | 110/900（12.22 %） | 440/900（48.89 %） | 433/900（48.11 %） |
| 30 s | **2/900（0.22 %）** | **0/900（0.00 %）** | **3/900（0.33 %）** | **142/900（15.78 %）** |

**逐种子**（每格 n = 1,200 = 4 窗 × 300）：

| 档 | seed0 | seed1 | seed2 |
| --- | ---: | ---: | ---: |
| 10 s | 1,195/1,200（99.58 %） | 1,200/1,200（100 %） | 1,200/1,200（100 %） |
| 20 s | 818/1,200（68.17 %） | 407/1,200（33.92 %） | 25/1,200（2.08 %） |
| 30 s | **131/1,200（10.92 %）** | 16/1,200（1.33 %） | **0/1,200（0.00 %）** |

⇒ 本轮的分裂**几乎全在种子上**：30 s 档 **142/147（96.6 %）的活满集中在 w18**（w11 2、w12 0、w17 3）；而 w18 的 142 条 = **seed0 127 + seed1 15 + seed2 0**。三个种子在 30 s 档的贡献是 **seed0 131（占全档 89.1 %）/ seed1 16 / seed2 0**，其中 seed0 的 131 = w11 2 + w17 2 + w18 127；**seed2 在 30 s 档 0/1,200**。这是「**学成一直按雷、把 20 s 换成 30 s 的假存活**」的直接读数。

#### 34.4.2 判定（`summary.json` → `verdict`）

| 门 | 预注册要求 | 门槛 | 实测 | 差值 | 单侧 Fisher | 结论 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| **P1**（主） | 900 决策（30 s）池化存活 > v8.23.0 40/20/5 点估计 112/194 = 57.7319 %，`p < 0.05` | **57.73 %** | **4.08 %**（147/3,600） | **−53.65 pp** | p = **3.075e-88**（更差） | **不成立** |
| **P3** | 600 决策（20 s）池化存活 ≥ 85 %（v8.21.0 为 90.26 %） | **85 %** | **34.72 %**（1,250/3,600） | **−55.54 pp** | p = **3.107e-181**（更差） | **不成立** |
| **P4** | 每格报 distinct，< 10 作废 | ≥ 10 | min 73 / 作废格 0 | — | — | **成立** |
| **P2** | 报用雷分布并对 v8.21.0 的 2.964（20 s）/ 1.315（10 s） | 描述性 | 30 s 均值 2.9783（Δ = +0.0143） | — | — | **已报**（非门） |

判定 **`S-no`**（`S-no.held = true`；`S-win` / `S-long-only` / `S-fork` 均 **false**）：
- `P1.rate_gate = false`、`P1.fisher_significant = false` ⇒ `P1.passed = false`；
- `P3.passed = false`；
- **`S-long-only`（P1 成立且 P3 < 85 %）不成立**，因为 P1 本身就不成立——**不是「只顾长档」的失败模式**；
- **`S-fork` 不成立**：**D3 保真门 5/5 PASS**（`d3_gate.json` `verdict = "PASS: §2.1-§2.5 hold on this binary and data dir"`，`exit = 0`、`wall_s = 8.226`），引擎 `rl_v3` `691bfd77…` `match: true` ⇒ 失败**不是**二进制或数据目录的问题；
- `stop_text`：*"P1 does not hold: the hand-written thresholds still win the long tier; the final deployment configuration of this line is 40/20/5"*。

#### 34.4.3 机制：学成「一直按雷」（`bomb_usage.json` + `train_curve.json`）

| 档 | `bombs_used` 均值 | 用雷总数 | 直方图 `{0,1,2,3}` | 用满 3 颗 | `bomb_emits` 总数 | **每回合 `bomb_emits`** | `bomb_prob` 均值 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 10 s | 2.5211 | 9,076 | `{0, 444, 836, 2,320}` | 2,320/3,600 | 232,526 | **64.59** | 0.2153 |
| 20 s | 2.9747 | 10,709 | `{0, 0, 91, 3,509}` | 3,509/3,600 | 474,675 | **131.85** | 0.2692 |
| 30 s | 2.9783 | 10,722 | `{0, 0, 78, 3,522}` | 3,522/3,600 | 489,813 | **136.06** | 0.2704 |

- **`bombs_used` 基本恒为 3**：20 s / 30 s 档 **`episodes_never_bombed = 0`**、`episodes_exhausted` 分别 **3,509 / 3,522**（占 97.5 % / 97.8 %），只有 91 / 78 条只用 2 颗；10 s 档因预算短，2,320 条用满、1,280 条用 1–2 颗。
- **`bomb_emits` 远超 3 次上限**（每回合 **64.6 / 131.9 / 136.1** 次）⇒ 策略**不是「该用才用」，而是「能放就放」**：三颗雷在很早就被按掉，剩下的时间靠 `alive_frames` 撑。
- **熵单调上升、alive 单调下降**：池化 entropy **0.0258 → 0.0314 → 0.0443 → 0.0688 → 0.1044 → 0.1396**（0.0258 → 0.1396，**×5.4**）；`alive_frames_mean` **1,354 → 1,301 → 1,272 → 1,279 → 1,197 → 1,190**。逐种子末端熵：s0 **0.0370 → 0.0307**、s1 **0.0198 → 0.1808**、s2 **0.0207 → 0.2072**。
- **奖励没升**：`first_50_reward = 0.4028`、`last_50_reward = 0.3773`、**`rise = False`（−0.0255）**；每 50 回合池化存活 4/1/3/5/4/7（合计 **24/900 = 2.67 %**）。**只有 s0 在学**（R +0.074、存活 3→6/50、alive 1,385→1,434），s1 平、**s2 反向崩塌**（R 0.374→0.258、alive 1,275→**929**）。`grad_finite_all = True` ⇒ 不是数值爆炸。

⇒ 这三条合起来是**「熵在涨、奖励不动、用雷顶到上限」**：模型学到的不是「什么时候该按雷」，而是**「一直按雷」**。

#### 34.4.4 与 v8.21（600 决策训练）的对照 —— **工单口径需纠正两处**

| 档 | RL900（本轮，n = 3,600） | RL v8.21（600 决策训练） | **实测 Δ** | 单侧 Fisher |
| --- | ---: | ---: | ---: | --- |
| 10 s | 99.86 %（3,595/3,600） | 99.86 %（1,415/1,417 = 99.8589 %） | **+0.00 pp**（+0.0023 pp） | p = 0.63（**无差别**） |
| 20 s | 34.72 %（1,250/3,600） | 90.26 %（658/729 = 90.2606 %） | **−55.54 pp** | p = 3.107e-181 |
| 30 s | 4.08 %（147/3,600） | 4.79 %（28/585 = 4.7863 %） | **−0.70 pp** | p = 0.245（**不显著**） |

> **工单「10 s +9.7 pp」→ 实测 +0.00 pp**（`comparison.10s.versus_v8_21_RL_10s.difference_pp = 0.0023`）；**工单「30 s −0.8 pp」→ 实测 −0.70 pp 且不显著**。30 s 的正确读法是**「没有变好」**，不是「变差」。真正的差异**只在 20 s（−55.54 pp）**；10 s 变的是**用雷**（1.315 → 2.521），不是存活。工单这三处数字来自 `RUN.md` §11 已**自行纠正**的口头读数，本章按 `summary.json` 终值登记，**以终值为准**。

#### 34.4.5 部分采样读数不可用于判定（实证，`RUN.md` §11）

评测 9 路并行时盘上的 `eval_*.jsonl` 是**边跑边写**的，任何中途池化都是**采样未满**的读数，会**随窗口完成顺序单调漂移**。本轮实测：

| 档 | 中途读数（n） | 终值（n = 3,600） |
| --- | ---: | ---: |
| 10 s | 100.00 %（n=620）→ 99.85 %（n=3,387） | **99.86 %** |
| 20 s | 26.21 %（n=335）→ 21.34 %（n=1,954）→ 34.72 %（n=3,600） | **34.72 %** |
| 30 s | 0.34 %（n=291）→ 0.11 %（n=1,868）→ **0.14 % / 0.21 %**（n≈2,000–3,400） | **4.08 %** |

⇒ **`d900 0.14 %` / `0.21 %` 之类的数字是部分采样，不是结果**；终值是 **4.08 %**（147/3,600），中途读数与之相差**最多 37 倍**（0.11 % → 4.08 %）。任何「30 s 只有 0.1 %」的说法都应作废。**本表的分母 n 逐档递增且不单调收敛（先降后升），说明漂移方向取决于「哪几个窗先完成」，不是「先高后低」的固定模式**——这是本轮新增的**可复用教训**：登记/引用**只能取 `n = 3,600` 的终值**，或标明 n 并注明「部分采样」。

另附两处**同类自我纠正**（`deviations.md` §2 第 6 条）：17:06 一次 mid-flight `--mode summary` 干跑曾用 **47–51 行**训练数据写过 `train_curve.json`，已被终版覆盖；**终版 `train_curve.json` 是每种子 300 行、共 900 行**（`train_rows = 900`，`episodes_per_seed = 300`，`decisions_per_episode = 900`）。

#### 34.4.6 旁证（与 §8 已知身份对齐）

| 项 | 实测 | 说明 |
| --- | --- | --- |
| 引擎 | `rl_v3` `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`，`match: true` | 与 §8 已知 SHA **相同** |
| 候选头 | `.tmp_v860_bombdag/ckpt/seed{0,1,2}/bombdag.pt` = `9c6d5d4a…` / `36a51090…` / `885dde70…`（各 39,069 B），`match: true` | 与 §33.2 记的同值 |
| 行为策略 | v4.5.0 `round2.pt` = `4abee069…` / `a93746bb…` / `c69960fc…`，`match: true` | — |
| 预注册稿 | `PREREGISTRATION_BOMBRL900_001.md` = `2ead6917…`（3,317 B） | `summary.prereg` / `ckpt_assertions.prereg` / 盘上**三处同值** |
| 音频 | `.tmp_v8p0_bombrl900/data` 下 **69 个 `th08.cfg`，解码 69、errors 0、muted 69、audible 0** | **AGENTS §9.3 合规**（`musicMode/playSounds/musicVolume/sfxVolume` 四字段全 0） |
| 引擎进程 | `process_residue.json`：他轮 **18** 个 `th08-modern`，`matching = 0`、`lines = []`；`preflight.json`（17:02:36，`phase: start`）本轮 **0** 个、`nproc = 20`、`waited_s = 0` | **本轮 0 残留**，他轮进程**未被碰**（未 `pkill`） |
| 评测退出 | `eval_exits.tsv` **9 行 / 9 个进程 / `exit` 全 0**，最后一拍 **17:45:25**；`commands.tsv` 同 **9 行** | `invalid_rows = 0` ⇒ 10,800 行无坏行 |

### 34.5 与 manifest 不符项（**只报告、不改**）

> **总结论：`evidence/BOMBRL900/manifest.json` 与盘上 0 不符。** 逐项比对 `{bytes, sha256}`：**67/67 全等**；`bytes` 求和 = `total_bytes`（**67,678,732**）**差 0**；「manifest 有而盘上无」**0 项**；「盘上有而 manifest 无」= **1 项**（`manifest.json` 自身，9,646 B，在 `root` 之内但不可能自哈希）+ **1 项**（顶层预注册稿 3,317 B，在 `root` 之外）⇒ **无缺件、无多余件**。下面各项**不是 manifest 的错**，而是「manifest 覆盖不到」或「manifest 之外的证据之间口径不同」，按工单**只报告、未改任何文件**。

| # | 项 | 事实（实测） | manifest 侧 | 处置 |
| ---: | --- | --- | --- | --- |
| 1 | **覆盖齐口可精确闭合（65 / 67 / 68 / 69）** | `publish.count = 65`（排除自身 + `manifest.json`，因 manifest 在 publish **之后**才写，17:47:49 同一秒）；`manifest.file_count = 67`（排除自身）；`evidence/BOMBRL900/` 盘上 **68**；整目录 **69**。闭合：**65 + `publish.json` + `manifest.json` = 67**、**67 + `manifest.json` = 68**、**68 + 预注册稿 = 69** | 三者互不矛盾 | **只报告**；`publish.copied` 的 **65 条 `sha256` 与盘上 65/65 全等** |
| 2 | **manifest 不含执行器，且执行器 `self` 在轮内漂移过（但 final = 盘上）** | `ckpt_assertions_start.self` = **`00fda3fc…`**（17:02:36）→ `ckpt_assertions_final.self` = **`71bfbeb6…`**（17:12:36）= **盘上现值 `71bfbeb6…` / 125,305 B**。即轮内执行器**被改过**，**但 final 门看到的字节就是最终字节** | manifest **不含执行器**（`root` 之外），**不冲突** | **只报告**；这对最终判定**无影响**（`deviations.md` §1 明写执行器是 v8.21.0 `v8m0_bombrl.py` 的**授权副本**），但**不能**像 §33.5 第 1 条那样把本轮当「轮内零漂移」的正对照——**本轮是漂移、§33 是不漂移**，两章口径相反，切勿互引 |
| 3 | **`protected{}` 9 件没有 `expected`，`all_pass` 对它们没有约束力** | 两份 `ckpt_assertions` 的 `protected{}` 各 **9** 件（`v432_fork_arm.py` 123,448 B、`v450_cf_label.py` 67,077 B、`v431_arm_a.py` 15,697 B、`v800_bomb.py` 92,126 B、`v830_bombpol.py` 138,057 B、`v860_bombdag.py` 126,406 B、`v8l0_unified.py` 84,166 B、`touhou_rl/v420/contracts.py` 56,321 B、`touhou_rl/v400/gym/actions.py` 3,813 B），字段只有 `{exists, sha256, bytes}`，**`expected` 计数 = 0** ⇒ 即使受保护文件被改，`all_pass` 仍是 `true`。**实测 start 与 final 的 9 件逐条相同（0 漂移）** | manifest 不涉 | **只报告、不改**（与 §27.5 第 9 条、§32.5 第 4 条、§33.5 第 3 条同型；本轮 `expected` 件数从 §33 的 10 降到 **9**） |
| 4 | **`manifest.root` 写的是 WSL 绝对路径** | `root = /mnt/d/WorkShop/Touhou_RL/Document/Update/Alpha/v8.24.0/evidence/BOMBRL900`（`publish.publish_dir` 同型） | 仓库内相对路径需自行拼 `Document/Update/Alpha/v8.24.0/evidence/BOMBRL900/` | **只报告**（与 §33 同型）；本表 §34.1.1 **已统一写成仓库根相对路径**，可直接与 ledger 对拼 |
| 5 | **工单列举的文件名与盘上不完全同名，另一批文件工单未列** | 工单写 `smoke.json` → 盘上是 **`eval_smoke.json`（11,612 B）+ `train_smoke.json`（3,731 B）**；工单写 `preflight_*` → 盘上**只有 `preflight.json` 1 件**（674 B，`phase: start`）；工单写 `bomb_head_seed{0,1,2}_d900.pt` → 盘上实为 **6 件 `.pt`**（另 3 件 `_bc.pt`）；盘上另有工单未列出的 **`deviations.md`（4,236 B）、`seed{0,1,2}.jsonl`、`train_seed{0,1,2}.json`、`pretrain_seed{0,1,2}.json`、`train_smoke.json`、`eval_smoke.json`** | 只记字节 | **只报告**；**盘上实测 69 件与工单「69 文件」一致**，差异只在**逐个文件名**，本章**一律以盘上为准**入册，未按工单清单裁剪 |
| 6 | **`commands.tsv` / `eval_exits.tsv` 与 §33 那轮不同口径** | 两表各 **9 行**（**无表头**），`commands.tsv` **无 `wall_s` 列**（§33 那轮是「1 表头 + 9 进程」且含 `wall_s`、9/9 全等）；本轮 9 个 tag 集合一一对应，`eval_exits.tsv` 的 `exit` **全 0**、时间戳 **17:28:01 → 17:45:25** 单调；`wall_s_total` 只能在 `summary.arms.<budget>` 里读（2,821.313 / 4,586.519 / 4,962.899 s） | manifest 只记字节 | **只报告**；**不影响可复算性**，但**不能**照 §33.5 第 6 条那样「逐行对 `wall_s` 9/9 相等」——**本轮没有逐进程 wall_s** |
| 7 | **「最小格 distinct = 73」是跨档最小值，不是 30 s 档的值** | 三档的最小格 distinct 分别是 **73（10 s）/ 247（20 s）/ 263（30 s）**，且**三档都落在 `w18_seed0`**；`summary.P4.min_distinct = 73` 取的是三档最小，而 `P4.per_budget["900"].per_cell_min = 263` | 不涉 | **只报告**；引用「最小格 distinct」时必须**标明档**，否则会把 10 s 的 73 误挂到 30 s 的 263 上（本表 §34.4.1 已逐档分列 73 / 247 / 263） |
| 8 | **`audio_audit` 与 `process_residue` 是**本轮**自证，不是他证** | `audio_audit`：`scope = .tmp_v8p0_bombrl900/data`、**69 cfg / 69 muted / 0 audible / errors 0**；`process_residue`：他轮 **18** 个引擎、`matching = 0`、`lines = []` | 不涉 | **只报告**；两者都是**本执行器自报**（`producer = tools/rl/v8p0_bombrl900.py`），不是外部审计；但 `preflight.json`（17:02:36，本轮 0 引擎）与 `deviations.md` §3（16:57:45 他轮 97 引擎 → 17:02:36 本轮 preflight 0 → 17:02:40 本轮首引擎）**两条独立时间线自洽** |

**逐条核对口径（可复算）**：把 `manifest.json` 的 `files{}` 逐项与 `sha256sum` 实测比对，`{bytes, sha256}` **不符数 = 0**（67 项全过）；`bytes` 求和与 `total_bytes` **差 0**（67,678,732）；「盘上有而 manifest 无」= **1 项**（`manifest.json` 自身）+ **1 项**（顶层预注册稿）、「manifest 有而盘上无」= **0 项**。`publish.json` 的 65 条 `{name, sha256}` 亦与盘上 **65/65 全等**。旁证：`summary.json`、两份 `ckpt_assertions`、`bomb_usage.json`、`d3_gate.json` 的 `prereg.sha256` 都是 `2ead6917…`，与盘上预注册稿（3,317 B）**逐位相同**；`engine` 在各处都是 `rl_v3` `691bfd77…`、`match: true`（与 §8 已知 SHA 相同）。

> **一句话结论**：本轮把 **v8.24.0 发布目录 69 件 / 67,691,695 B + 执行器 1 件 / 125,305 B = 70 条 / 67,817,000 B** 成对写入 `ASSET_SYNC.md` §34 与 `ASSET_SYNC.sha256`，`sha256sum -c` 由 **1080 OK / 1 FAILED（1081 行）** 变为 **1150 OK / 1 FAILED（1151 行）**（新增 70 条**全 OK**，FAILED 数仍为 1、仍为已知允许的 `v4.2.0/journal.jsonl`）；三档**全部满额 3,600 行**、判定 **`S-no`**（P1 4.08 % vs 57.73 % 门槛、p = 3.075e-88；P3 34.72 % vs 85 %、p = 3.107e-181；**P4 成立**；**D3 门 5/5 PASS**），机制是**学成「一直按雷」**（`bombs_used` 2.9783、30 s 档 3,522/3,600 用满 3 颗，`bomb_emits` 136.1 次/回合，池化熵 0.0258→0.1396 而奖励 −0.0255 未升）；与 v8.21 的实测差是 **10 s +0.00 pp / 20 s −55.54 pp / 30 s −0.70 pp（后两者中 30 s 不显著）**，**工单的「10 s +9.7 pp」与「30 s −0.8 pp」按终值不成立**；**部分采样漂移已实证**（30 s 中途 0.11 %/0.14 %/0.21 % → 终值 4.08 %，相差至多 37 倍）；manifest 逐条比对 **67/67 全等、0 处不符**，差额恰为 `manifest.json` 自身（9,646 B）与顶层预注册稿（3,317 B）⇒ **无缺件、无多余件**。**未启动引擎、未训练、未评测、未 `git add/commit`**，未改动、删除或移动任何证据文件。

## 35. v8.26.0 资产（2026-11-22）

### 35.1 登记范围与逐文件表

本轮登记 `Document/Update/Alpha/v8.26.0/` **整目录**（`Get-ChildItem -Recurse -File` 实测 **155 件 / 18,831,813 B = 17.96 MB**，17:58 发布完毕），含 `manifest.json`、`publish.json` 自身，顶层预注册稿 `PREREGISTRATION_URGENCY_001.md`（4,272 B，`9d462c1f…`），以及 `evidence/URGENCY/` 下的 `RUN.md`、`summary.json`、`notes.md`、`extra_numbers.{md,json}`、`extra.py`、`verify_fisher.py`、`d3_gate.json`、`smoke.json`、`audio_audit.json`、`process_residue.json`、`preflight_{start,final}.json`、`ckpt_assertions_{start,final}.json`、`commands.tsv`、`eval_exits.tsv`、**108 件 `eval_G{1,2,3}_d{600,900,1200}_w{11,12,17,18}_seed{0,1,2}.jsonl`**（3 臂 × 3 档 × 4 窗 × 3 种子 = **108 格全齐、缺格 0**，每件恰 20 行 = **2,160 行**）与 **27 件 `G{1,2,3}_d{600,900,1200}_s{0,1,2}_progress.json`**；并登记**执行器** `tools/rl/v8r0_urgency.py`（115,154 B，`4e81f911…`），合计 **156 条 / 18,946,967 B**。**入册 156、未入册 0**（见 §35.2）。

| 组 | 文件数 | 字节 | 目录实测 | manifest 覆盖 |
| --- | ---: | ---: | --- | --- |
| `Document/Update/Alpha/v8.26.0/**` | **155** | **18,831,813** | `Get-ChildItem -Recurse -File` 155 件 / 18,831,813 B（17.96 MB） | `evidence/URGENCY/manifest.json` 自述 **153** 件 / **18,805,573** B（`root` = `Document/Update/Alpha/v8.26.0/evidence/URGENCY`，以 `manifest.json` 自身排除） |
| `tools/rl/v8r0_urgency.py` | **1** | **115,154** | — | manifest **不含执行器**（`root` 之外） |
| **合计** | **156** | **18,946,967** | — | — |

口径自洽（逐位可复算）：

| 目录 | 件数 / 字节 | `evidence/URGENCY/` | `manifest.json` | manifest 件数 / 字节 | 顶层预注册稿 | 差额 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v8.26.0 | 155 / 18,831,813 B | 154 / 18,827,541 B | 21,968 B | 153 / 18,805,573 B | `PREREGISTRATION_URGENCY_001.md` 4,272 B | 18,805,573 + 21,968 = **18,827,541**（+ 4,272 = **18,831,813**） |

文件类型构成（实测，**无 `.tmp_*`、无 `*.tmp`、无 `tmp_*`、无 `.partial`、无 `~` 结尾、无点开头件、无 Hidden 属性件**）：

| 目录 | `.jsonl` | `.json` | `.md` | `.tsv` | `.py` | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v8.26.0 | 108 / 18,228,913 B | 39 / 520,041 B | 4 / 65,155 B | 2 / 2,840 B | 2 / 14,864 B | **155 / 18,831,813 B** |

> 字节重心是 **108 个 `.jsonl` 共 18,228,913 B（占该目录 96.80 %）**，全部是**评测原始行**（每件 20 行、**132,643–195,124 B/件**），合计 **2,160 行 = 3 臂 × 3 档 × 240 条**；**27 个 progress `.json`**（各 5,076–5,093 B）是逐 `(arm, tier, seed)` 进度快照，与 27 个评测进程一一对应；其余 **19 件**（`evidence/URGENCY/` 内 **12 个 `.json` + 3 个 `.md` + 2 个 `.tsv` + 2 个 `.py`**，另加顶层 1 个 `.md` 预注册稿）中体量最大的是 `summary.json` 229,925 B、`RUN.md` 44,647 B、`publish.json` 35,961 B、`audio_audit.json` 31,594 B、`smoke.json` 30,989 B，最小的是 `process_residue.json` 539 B、`preflight_final.json` 763 B；`extra.py` 13,020 B 与 `verify_fisher.py` 1,844 B 是只读复算脚本（属 manifest 的 `supplementary`，**不计入 `publish.copied`**，见 §35.5 第 1 条）。发布时序：`preflight_start` 17:40:21 → `ckpt_assertions_start` 17:40:23 → `d3_gate` 17:40:32 → `smoke` 17:40:40 → 首个评测进程 17:40:52 → 最后一个退出 17:54:57 → `ckpt_assertions_final` 17:55:19 → `audio_audit` 17:55:20 → `preflight_final`/`process_residue` 17:55:21 → `summary` 17:55:24 → `RUN.md` 17:57:50 → `publish` 17:58:19 → `manifest` 17:58:23。

#### 35.1.1 逐文件表（156 条：155 件发布目录 + 1 件执行器）

| 1 | `Document/Update/Alpha/v8.26.0/PREREGISTRATION_URGENCY_001.md` | 4,272 | `9d462c1f4b0f125212121c1e49e27b87802a5beed2ea39e60183e597f8a6be7f` |
| 2 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G1_d1200_s0_progress.json` | 5,084 | `5dc30b8a8e711209337680b3bb5edacfdf41bbdd4d17d6a3e804487c16276b57` |
| 3 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G1_d1200_s1_progress.json` | 5,091 | `bc84b26725930925402460c896362e46fe22f2149d1f5312d83e367a9e1e3e9b` |
| 4 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G1_d1200_s2_progress.json` | 5,090 | `79694bd59f41989644279c296f17d4e8b2da4713b4fddd0d81f49cc5c02d40fe` |
| 5 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G1_d600_s0_progress.json` | 5,076 | `a801b222aecbbfd8f09978f0fc755dc3bf9d35c5d44089eb1105ffbd74d78269` |
| 6 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G1_d600_s1_progress.json` | 5,085 | `d783d510e3dde3bc6f54ad274c82198d2e44d2fcb64af6e6d23ceac63657c765` |
| 7 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G1_d600_s2_progress.json` | 5,080 | `6eadb56b25a983b95f6751b6c39ec5b1b5db33f4a6bc34512a7139edce4a6026` |
| 8 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G1_d900_s0_progress.json` | 5,083 | `1dfde6fa1fdfcac0539ef842f51983ddcdb9c7f515d57bf442c0554517421a2d` |
| 9 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G1_d900_s1_progress.json` | 5,078 | `2e2b2c60ba7ec4f76097b94858073052ad02ee39eda5489a1fa08874b02b815e` |
| 10 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G1_d900_s2_progress.json` | 5,080 | `9f9ddd0b302c444e984251fa3bcd8f985da7e133a8a1c4dcbf18726afe5ce554` |
| 11 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G2_d1200_s0_progress.json` | 5,087 | `888f72e68ffd61b2545310977ef3a95458ab76662355c884b60594bf7d2e342f` |
| 12 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G2_d1200_s1_progress.json` | 5,089 | `efb5145716c86f71e75684e02d3c72bbdc9344a12464523b57a4cf7c1fa41a72` |
| 13 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G2_d1200_s2_progress.json` | 5,091 | `ffd5a3b8510998dfb2dd1a1b8ec64c016d04fa8625f61ea84b2c06cc7424dd45` |
| 14 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G2_d600_s0_progress.json` | 5,078 | `7ed9b3e7b2a61f1184ac13b826840402049179235c9dce343fa678ee1d58d4cb` |
| 15 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G2_d600_s1_progress.json` | 5,083 | `5548dbfcf6128cb5e8b6c6f58f3f31e1655222dcdb409ac7d27ec57d43e77fc5` |
| 16 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G2_d600_s2_progress.json` | 5,083 | `9e8bb5a2d469bd9f8cc19fa4de7f0f81120ad3379f8ed11504cd9cce7ba8fdcd` |
| 17 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G2_d900_s0_progress.json` | 5,078 | `5708c5132b4f211cb1bfb4ee24f5cc7dd4cbde1dc31e69c92dea4741cba973f4` |
| 18 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G2_d900_s1_progress.json` | 5,082 | `cbe5a3ad93a6791a41de838e13b08d6c31130e50d6fe580c73ff6d53d48af864` |
| 19 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G2_d900_s2_progress.json` | 5,080 | `ae00b44e9496553b7da8de4f6db72b953ac33c48dae804a68abebca0aa7bfef3` |
| 20 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G3_d1200_s0_progress.json` | 5,089 | `f76b8ed8b30535250ee7cc272939ae84a3373431c734d9b101b212a53316c2f4` |
| 21 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G3_d1200_s1_progress.json` | 5,093 | `84048c68ba1117e6c09ce73ad59113451e34c180cb13268e8780e5d528d2656d` |
| 22 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G3_d1200_s2_progress.json` | 5,089 | `489f571559ec69bc2dd5afbb8e55c281d745e671947f7dd12e904a4acb31bbf7` |
| 23 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G3_d600_s0_progress.json` | 5,079 | `61f0f7993cbd001d91cf0283a706e499883298850585f71935f7f8d532850a32` |
| 24 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G3_d600_s1_progress.json` | 5,086 | `a2cc1f7821de2a3b4e835f6abc314a3d9e1bb0dcb238a81db420c479ea52070c` |
| 25 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G3_d600_s2_progress.json` | 5,081 | `b61861e86cb3fa0458fba803cac54e96568232953e7478ec78b6258e558c7567` |
| 26 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G3_d900_s0_progress.json` | 5,080 | `631577d9c26090e3549c29ab1516b1008dc67a11a873a4d8a57f4be1502f9077` |
| 27 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G3_d900_s1_progress.json` | 5,084 | `0db6fab9c0734cd43ce8e487067647490b46d3aeb7b89d4cf0138ea2c49b1387` |
| 28 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/G3_d900_s2_progress.json` | 5,080 | `9bb293d77374e55c89ed5031eab3b77dd823196a71a0aae1b12d27a49ace382a` |
| 29 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/RUN.md` | 44,647 | `d9762058d43a44d1c6ce2cb688b19eb5b3b3f318187cfa48ffc1dc7ef9fb512d` |
| 30 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/audio_audit.json` | 31,594 | `fcd2d75f9017a73e4fa67ebe7576650bf073883a2491dcb28ca6c07426d645b3` |
| 31 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/ckpt_assertions_final.json` | 5,228 | `c9680942efcf3e40f80158f1d6a69dad3a890e388f908a60346165e85ae3f36c` |
| 32 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/ckpt_assertions_start.json` | 5,228 | `44e1e2f49fb9aa24ca2a7a1f71c891e6454a8d0771bf90d02e224f1eab975dc9` |
| 33 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/commands.tsv` | 1,643 | `665428b49e67a239434255f5aed58691284970741fd5a26c37b15f81069e7574` |
| 34 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/d3_gate.json` | 5,097 | `a2a61fa3f83d802fe56174ecf7d5789ad94c1255edadd8a4964b489a4fb48bc6` |
| 35 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d1200_w11_seed0.jsonl` | 195,124 | `a04170f3e7d4c04c31da3bbf3cc713fed8f0ea1ee0afc9c877a2ff5b930896a4` |
| 36 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d1200_w11_seed1.jsonl` | 180,046 | `6494508f1329986144b09b6fbd028f1f81edc1af6b036d938802ac54a678ac29` |
| 37 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d1200_w11_seed2.jsonl` | 187,188 | `bd4883a86461c769c5e913f56e17d97cde35310b0c511d0ec6cedbcbf503e2ee` |
| 38 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d1200_w12_seed0.jsonl` | 182,183 | `b6eb21a57fd795c84e377808bcbe3fce077376c1021708edfa92002bcf430cd9` |
| 39 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d1200_w12_seed1.jsonl` | 186,804 | `08ce6b22717c3f431429d5d9019e55f27eafcb88fc6a7c304dd806c2194b05c4` |
| 40 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d1200_w12_seed2.jsonl` | 173,893 | `c25c333163bbf3f62c52e104a6465038f92dd08e73748ffa491abaa55df10abd` |
| 41 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d1200_w17_seed0.jsonl` | 166,353 | `be5eb657a252c66c7f0bdb5a963540d9c30531ffd7ac6effb8f4d7c2a4d63d19` |
| 42 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d1200_w17_seed1.jsonl` | 155,981 | `81664a11a286d8368fbacf52e70d9d9c2bbf7bff8e374abcb9b79ab91f65a584` |
| 43 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d1200_w17_seed2.jsonl` | 146,911 | `d0cb9fb88f74f45b0cb188a6f134908bd321903412ec4027de04f4898fa2ecfb` |
| 44 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d1200_w18_seed0.jsonl` | 188,662 | `7e7728490daf09e766d54586a46493431f4eb86f4164e54c2e96c8f5aee25c66` |
| 45 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d1200_w18_seed1.jsonl` | 189,254 | `82d530ff9f70c8345a7a186f8982e282c7ba897f355f4def9f0e8db9cd66f1c7` |
| 46 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d1200_w18_seed2.jsonl` | 182,630 | `d872805c7d8c4b68ce934dd22fbeda2be1973337b2cb2eced3da4909d851865f` |
| 47 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d600_w11_seed0.jsonl` | 141,102 | `8c90d2abcee2b458a37ff7b73648c0c662cd6c9306889e867f367be887dc1198` |
| 48 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d600_w11_seed1.jsonl` | 137,088 | `32034d145da9ceecc35fa5e75009c3ee13e4aa8444be4d5807e56df2d868d82c` |
| 49 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d600_w11_seed2.jsonl` | 145,834 | `fb9890df166e4c05e0900c7ea674ec4a08c0d9d5c9e4dacd09c41e05f48c88be` |
| 50 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d600_w12_seed0.jsonl` | 137,101 | `2ae3ad0c7dfaf2b771824e479598250f407ab02c3652f428cbc06bb3104f6d0a` |
| 51 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d600_w12_seed1.jsonl` | 136,218 | `3a9b64aa49e83cbe6cfbc1e8a1c849fc562ce67c581c310315e379d1179e855a` |
| 52 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d600_w12_seed2.jsonl` | 133,218 | `7e1844e60dc54cb1fd60fbb24e7ad674ea3ec0e346e33c35fefcb73990e1387f` |
| 53 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d600_w17_seed0.jsonl` | 137,847 | `d8cf41301f489c844d08ee2bafd3f358d3f1587c70247d2f0d0481cfcffe4a11` |
| 54 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d600_w17_seed1.jsonl` | 132,643 | `41972dff580abf5845621fe27c2bf3bdf620ff2ad349a6c713ae6b73435b4f31` |
| 55 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d600_w17_seed2.jsonl` | 134,983 | `f3d9a253c509f9698e1bdf5e47db4a89c03119ae7e41b41f17317ab67942dcf8` |
| 56 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d600_w18_seed0.jsonl` | 148,859 | `5f08cd23d8eca84acecd0b86c90e2345909bd46ad4cacfb541d30b4b85632d8c` |
| 57 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d600_w18_seed1.jsonl` | 147,226 | `fbe53eda5c2aba13ed421fa76ee37b29101fe1ec13f2c83af75d4ac297a77d59` |
| 58 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d600_w18_seed2.jsonl` | 152,400 | `c2c4f1ea0e6f57afd1707d3190bc0976f1cc00288529c0a2ddc936f1c7317008` |
| 59 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d900_w11_seed0.jsonl` | 173,049 | `09bdab19e7f7d19694d3dcba280bb6e894010bf2c632ebd7a4c12a2868f940ef` |
| 60 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d900_w11_seed1.jsonl` | 171,778 | `ae0885849f31db74d1a5895942151fbe288f015953767ff2fb20a11295a5cb91` |
| 61 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d900_w11_seed2.jsonl` | 170,028 | `16c09253b4389ded5537f964dec95f7682779a164037ff07e09d28235015c44d` |
| 62 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d900_w12_seed0.jsonl` | 171,812 | `77ba5b9698ae1e0799a4095e2a61951b695760381ac37a8f8963ac5c5c39d7b8` |
| 63 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d900_w12_seed1.jsonl` | 165,246 | `6076ce2677a45004ed46ea063cff7fb998e562d808afb8d3a5af1335ab83e961` |
| 64 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d900_w12_seed2.jsonl` | 163,518 | `751b37df4517f4e9f21561f943630e1495d61917d7144103c146f44ae77677e0` |
| 65 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d900_w17_seed0.jsonl` | 153,200 | `28ba8be6a65318b98f8bbcad428c8ef161c36fd119c66650d18d88a5837a81d2` |
| 66 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d900_w17_seed1.jsonl` | 149,107 | `cae2c34b33604d87c0103781ffc08bf2db0cd46c900ca6855b62204527434b37` |
| 67 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d900_w17_seed2.jsonl` | 148,271 | `f5d4b19db42406b0448c17766b5accba71634648cb6fbb58509e63e5f58de357` |
| 68 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d900_w18_seed0.jsonl` | 181,566 | `ea1f64d6c30c93b6c4aa5859aad966e050ef81dbd3263e8933f4f36d6c0d27a2` |
| 69 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d900_w18_seed1.jsonl` | 180,559 | `e7856ab12b89efcb2cd303e73abc7c5c302da31ef43f683891a04f99f456e41a` |
| 70 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G1_d900_w18_seed2.jsonl` | 177,568 | `bf760b3913226bc5ccf6403e5b4507a302642dbbead40123d8cc7e8628d29864` |
| 71 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d1200_w11_seed0.jsonl` | 181,032 | `5e031c701ef306cc7c0b8bbd63e9d0ca702c3e44ff2f6c5e46533ddb833aaaeb` |
| 72 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d1200_w11_seed1.jsonl` | 181,455 | `3dfb70e9f8da1854b4ecf05b2380b3f60995ba90696bc3e0ca142b1da9703b48` |
| 73 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d1200_w11_seed2.jsonl` | 183,537 | `ac3788e09f483c1bfe646a257acb02fa189a56b0ab91d160d048cd40b18b754d` |
| 74 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d1200_w12_seed0.jsonl` | 176,866 | `4a19fff5d0c2d6c96ee6ed4a317632191fa7df7d6de4aee5574126513f5638de` |
| 75 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d1200_w12_seed1.jsonl` | 185,027 | `19d6e9f3cc3a82917385eef6edc211e89b3b333c7cdbe82534e056ca55bcd92e` |
| 76 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d1200_w12_seed2.jsonl` | 177,435 | `14b25ccd4a28d7e8a159708377515e1bddc231b1b3978a5d28e16e408e47a802` |
| 77 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d1200_w17_seed0.jsonl` | 179,429 | `8ac50615530e976ccd4d8f74665b5d291e2c6b16bc5913a45378e89a6ebb0071` |
| 78 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d1200_w17_seed1.jsonl` | 181,566 | `7396a42c6ac0aab997017dcf4f3ebc0a3e952b9e53230d913eef2c40852fe725` |
| 79 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d1200_w17_seed2.jsonl` | 176,481 | `98d948456ce1777553d0812caae423c65d552f927760462683ad5bc6e93d8d62` |
| 80 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d1200_w18_seed0.jsonl` | 181,035 | `e5aec0cfa4181400e948f2babfbbe5716ed808b38364879f1548eb93ff6dc17e` |
| 81 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d1200_w18_seed1.jsonl` | 188,799 | `9a7647be6634c3d36bfd6595d22eeff68f6bd584f4bd24e0ffb7bdbeac993f1b` |
| 82 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d1200_w18_seed2.jsonl` | 179,691 | `e137f576e3e77090cb31d00276c0c942e3150029e9e43c1e0407240dac844d85` |
| 83 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d600_w11_seed0.jsonl` | 151,424 | `f4f86c8db1aee64c03f032dc6b3ce4a02ca75dc6c5e4cc4ce97fe8c0200db9b3` |
| 84 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d600_w11_seed1.jsonl` | 146,464 | `17c2db6836e635b0e3c3adf90a42efa429fae06b13f911b9ffed3fd72f713f09` |
| 85 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d600_w11_seed2.jsonl` | 152,398 | `070f39dc51be6ef1619a559a8e9a35d188dd1a9e315a50464d784b13d6b9825a` |
| 86 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d600_w12_seed0.jsonl` | 144,837 | `ada8c92c562865fea6c60e0528615fdf11fe16167cc6f1255aa32f8fbb6ed810` |
| 87 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d600_w12_seed1.jsonl` | 142,843 | `db780bf063c0d04a6822f537fc50aeda4c01b42aca773ddfa54362a544580fe4` |
| 88 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d600_w12_seed2.jsonl` | 141,499 | `56ddbbdfda8227b0b488a37babee8409b1cb80bb230c3b4a76126f8ad5e92871` |
| 89 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d600_w17_seed0.jsonl` | 141,883 | `7db1a4b7f554fe40190b3643d6bf07a5108f94e72f9f3a80c40aaef24efc8338` |
| 90 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d600_w17_seed1.jsonl` | 134,434 | `53836d742aa5c66ef1bf4c20b6221ba9e6658f60ff942df171aab5abbab9bc1f` |
| 91 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d600_w17_seed2.jsonl` | 139,723 | `8dcdd58a5301d7eaca8fe310c24a8bb37a82e105f20464badb449c1d98805001` |
| 92 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d600_w18_seed0.jsonl` | 150,694 | `06c65d98ba3be14c56ba07fbbfc5cc8b07b90cfb545583a52a902837ef494f7a` |
| 93 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d600_w18_seed1.jsonl` | 150,453 | `dad70b81e043d9e1feed8ff4415698b4f82b2fc0e00f4b29a09944e04d85d0f5` |
| 94 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d600_w18_seed2.jsonl` | 153,706 | `0ef980b0b53f54bfe0d53146c276a4a5191069abf3ef5683090e5710a2435089` |
| 95 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d900_w11_seed0.jsonl` | 180,754 | `561f4fce61dd48c8786d56f8197e759b89cfd56b93d7084179eae629d90a8db5` |
| 96 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d900_w11_seed1.jsonl` | 179,837 | `162f06f341bdbd3b9bfc7187d76130f2527e15563f65fe83ee03bc9e53d787e1` |
| 97 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d900_w11_seed2.jsonl` | 182,688 | `7c5340031829676d414fcf3714e19cc92ee0a7f5b54d096df5bcce3ad4a240df` |
| 98 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d900_w12_seed0.jsonl` | 180,688 | `91c2e440fe3378596e8d16d845b8732f8631b9479a8976ec548c4f3ac93dcfa1` |
| 99 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d900_w12_seed1.jsonl` | 178,282 | `e98f5857bf36c269be429d6c332e60c0f036dcccd280088f86248f92b33e992f` |
| 100 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d900_w12_seed2.jsonl` | 179,981 | `5d2332c374f60a64df014025650541fbe92f04c7026d3a34045fdd73e8311ada` |
| 101 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d900_w17_seed0.jsonl` | 174,188 | `a47426e97e1968798d4c70ff54e64a5c094ee175415dc070eb666e7217698bef` |
| 102 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d900_w17_seed1.jsonl` | 174,005 | `0402e51b3dbb8720e6a3f710217489fd9f919a827516c14a2861dcb229aada12` |
| 103 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d900_w17_seed2.jsonl` | 170,899 | `3cb90c5145540061a8ae01f0eb2231f6d265921734544d399d911cc38c89ea07` |
| 104 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d900_w18_seed0.jsonl` | 182,297 | `6ee5c663ae42f39cef4d10d3f5881857e4dc70ad21aba5d2bc7fc8b6d0d340c2` |
| 105 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d900_w18_seed1.jsonl` | 179,307 | `4821e12c5792505f629d9738e9c82d678c26add381259326495f9a1baa170594` |
| 106 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G2_d900_w18_seed2.jsonl` | 179,945 | `7c7b1ee93bf703a92c6749b314e99ea98db68b5c5eaac54e125379d45ca8ecb9` |
| 107 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d1200_w11_seed0.jsonl` | 186,063 | `7d6577b52aa8770d8e9c9d2c89e510ec86dd517bbe83c82a91565f73d0381212` |
| 108 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d1200_w11_seed1.jsonl` | 179,166 | `fe8fb96aaec8321f3837475e6ae0e56c35fb59282be9e7dd3f09bd0e06666c9c` |
| 109 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d1200_w11_seed2.jsonl` | 181,128 | `d20da0c54eb5f3b9cdd50248c2e18875ded963922dd633ccd85be216b6a7b75f` |
| 110 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d1200_w12_seed0.jsonl` | 180,921 | `e95bdf6285b531c7b86abcc16877a0203da8806cea2bf6f2458bdf4740cccd42` |
| 111 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d1200_w12_seed1.jsonl` | 187,321 | `b37d8e654b2dbd6a89bcbaa67e9ec925a39776e928963ec03f1c85fe08a25368` |
| 112 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d1200_w12_seed2.jsonl` | 181,563 | `459ec1745275a854b536b7daeed5475a696e856566a1d7b133fa616d047092fd` |
| 113 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d1200_w17_seed0.jsonl` | 179,736 | `59b1c3e3ecfa3e24d9010794077bf3af1a1f7c6a3c9e234c403dfd1dc1e2cc5d` |
| 114 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d1200_w17_seed1.jsonl` | 182,190 | `ffbfd6b0dfcefe7385408c88e0facd961f1142cad3eff88397fe93c8f36e707e` |
| 115 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d1200_w17_seed2.jsonl` | 174,169 | `7681ef8bcd801c93ac52ee7b62a34e466b3a42a9ccfc49c477458c0e22c36c70` |
| 116 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d1200_w18_seed0.jsonl` | 186,780 | `25d4163c0235605bd031d99894f8e1da79ccecff8946c978ff066363031f88a9` |
| 117 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d1200_w18_seed1.jsonl` | 189,377 | `e60d0e299d7adfbc429da04399367071548248cbede64ba6613563adcff878ff` |
| 118 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d1200_w18_seed2.jsonl` | 184,942 | `5caae2cb9882239ad245bed161f63247fc38cef59daaf3fb7ffeeb47f11a353a` |
| 119 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d600_w11_seed0.jsonl` | 159,036 | `954a3a7ec3252bde48dc63f6dc4536fc499dc48b00b7a85540e8e32fe58dadbe` |
| 120 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d600_w11_seed1.jsonl` | 160,548 | `29f2343b804e10b9095a5802c733ae1a3f5371d6193cbafdee35927c2b47457b` |
| 121 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d600_w11_seed2.jsonl` | 155,463 | `bcdcb882c0e83b73e4f2579f2f66d42fd33d286f5af1d8ecf8ec63b3e39de809` |
| 122 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d600_w12_seed0.jsonl` | 156,260 | `af4c402cd8d75fb576ca12bfbb3975e226a32d77a72d418eee65f6ebc8c2a092` |
| 123 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d600_w12_seed1.jsonl` | 159,382 | `031fac80ecbf8d1ccd4864db191f56adf8dd39787af5cd30c796bf0d00fba722` |
| 124 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d600_w12_seed2.jsonl` | 157,269 | `963242bda8d762794f62ee9510e2dc73b32da49e32c7b3d8275ab96f2e52fd43` |
| 125 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d600_w17_seed0.jsonl` | 150,954 | `371dd70acd2864b2a924a39e34163a67229c789b1af4be98de87524b9c170112` |
| 126 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d600_w17_seed1.jsonl` | 147,570 | `2718f2a1eda8eaa50cd882170fc307b4602bc31216988195b9f7bbc8759aa584` |
| 127 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d600_w17_seed2.jsonl` | 160,098 | `5d3ca24eb4385bff5da276d0274c75a9e31076e3845a2818d472afffe5d61afb` |
| 128 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d600_w18_seed0.jsonl` | 155,312 | `f6022159ba24b5c2d8519efc65781080bdc91806742a40a5abfa6f9747854634` |
| 129 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d600_w18_seed1.jsonl` | 154,449 | `715e89622b8e18e880dc590fbdd2b6e25e67f9a4e47c150df32402f9c9def2b0` |
| 130 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d600_w18_seed2.jsonl` | 155,794 | `b1b4e6e4a5d13721170cfef8eec20ca9f3c4cc85a2c00543ec56d489a4ed4c7e` |
| 131 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d900_w11_seed0.jsonl` | 186,673 | `97548e81674b5c4e212695dadae1741e31c5df5457a9956217ea4c1f1607fc12` |
| 132 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d900_w11_seed1.jsonl` | 186,689 | `7dd0ebbf7ee44c0089a85130ca821012607f40468bf8194d5743dce92749e072` |
| 133 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d900_w11_seed2.jsonl` | 185,597 | `26d0cb87c55cf6237238712356cfcb7705767e4ef8cbdfbc24644147bd6098ac` |
| 134 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d900_w12_seed0.jsonl` | 191,184 | `e9dfe69a3eb5f0b080d9602854f9374327f223da97fa5b7a124075c8c17f120c` |
| 135 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d900_w12_seed1.jsonl` | 190,899 | `16c3fa8eecedd5515366f474bff443cbf3c8c79d8cd2b3778b32005afb5372e6` |
| 136 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d900_w12_seed2.jsonl` | 189,628 | `832aa73ea1ef5d4a8a9c2c00ff713cffcefdece87bafaf096984ec1f417dfb24` |
| 137 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d900_w17_seed0.jsonl` | 193,420 | `95eef119c8d3577a618d676a697b0683b1771be6da7c6f42bd77b44f7d976569` |
| 138 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d900_w17_seed1.jsonl` | 194,775 | `c751f72f431701cab11fb414de689300f5ab93655610b89a9df7d27de4283a2e` |
| 139 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d900_w17_seed2.jsonl` | 191,967 | `8de2148e6548550a66eb66dcbeed2aae920fcf2cceeda6788970a040ceb213cd` |
| 140 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d900_w18_seed0.jsonl` | 188,804 | `aecf3177d6500490427c3789400c4be0003f51551e2dfca3d80db11a5dcd1d24` |
| 141 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d900_w18_seed1.jsonl` | 188,458 | `fa65059026706de92c05d8551aa6b267980db8aa8c1fe3d4df4d3cb26549469a` |
| 142 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_G3_d900_w18_seed2.jsonl` | 184,496 | `bb1174fd4cf134e3bb876b88838a52c97dd3dfdf0e6748ed5bb797ee0f09a09a` |
| 143 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/eval_exits.tsv` | 1,197 | `5cb109ef25985150442cacce0622c296497fbd6a177eee142adf6219d0da563f` |
| 144 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/extra.py` | 13,020 | `2d786d847fc34f097831a60fdead6a7582882c4c99a9a1a125ac24036c57a6f6` |
| 145 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/extra_numbers.json` | 14,115 | `f16c660b1b0289fbec0b20b611262eb1bbaf6b59db6def4d2781deb4dbfa6393` |
| 146 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/extra_numbers.md` | 6,267 | `e55d7d8252dcdc57e18ea36e42b5eb4b84856afe62a6ad5aec92f8dee1a4c652` |
| 147 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/manifest.json` | 21,968 | `753e934b63942cf43cb35bf4001b839973885de967e5f3da110f0b89e3841d55` |
| 148 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/notes.md` | 9,969 | `57ebce1bf5c9f825ad222b6a8ba9e1e348d2fafa2479ecf26bcc05ff5559ed3e` |
| 149 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/preflight_final.json` | 763 | `1e063890351cf5aa4cafb1efd97e3e9a5bd91ef30c3e155805c1aae1daec53f4` |
| 150 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/preflight_start.json` | 1,375 | `9f98c6876d3a06b16df91904ae77dd9d525ad0473dab2b163703626d375e1f26` |
| 151 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/process_residue.json` | 539 | `9285501c29cf7cee90b826845e288630ef5381edb799053080c9d067479b4751` |
| 152 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/publish.json` | 35,961 | `195dba9550c39834814a754816263a98b5d59efd50b51c76980b3252cf2bd061` |
| 153 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/smoke.json` | 30,989 | `a768d57d5855349105db8a655cb0176e096c4758da98f24a59c8ed291d130be1` |
| 154 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/summary.json` | 229,925 | `9025e3660fd844072bb63ab8bba1f30af9b2ff22c43975f8d3322fbfd6327137` |
| 155 | `Document/Update/Alpha/v8.26.0/evidence/URGENCY/verify_fisher.py` | 1,844 | `9c41cf0d4b278a52622a871d829af81de50aedeb280f565274d2bbd06fbf2fbf` |
| 156 | `tools/rl/v8r0_urgency.py` | 115,154 | `4e81f9111293148ff0221e5df4618702c4ad4069f4f070a50ffe2a26fe2ebf0e` |

### 35.2 跳过说明（未入册 **0** 个文件）

- **发布目录内 `.tmp_*` / `*.tmp` / `tmp_*` / `.partial` / `~` 结尾 / 点开头 / Hidden 属性实测 0 个**（`Get-ChildItem -Recurse -File -Force` 155 件 = 不加 `-Force` 的 155 件；按名筛前缀/后缀、按 `Attributes` 筛 Hidden 全部 0 命中）⇒ 本目录**没有**需要跳过的运行件，**155 件全部入册**。
- 按工单口径与 §8.1，`.tmp_*` **运行件一律不入册**：本轮真实运行根 `/mnt/d/WorkShop/Touhou_RL/.tmp_v8r0_urgency/`（`data/`、`work/`、`eval/`、`fidelity_d3.json`、`extra.py` 的同源副本等）按 §8.1 属本机运行资产、被 `.gitignore` 忽略，**不登记**；同理候选头 `.tmp_v860_bombdag/ckpt/seed{0,1,2}/bombdag.pt`（`9c6d5d4a…` / `36a51090…` / `885dde70…`，各 39,069 B）、行为策略 `.tmp_v450_cf/ckpt/seed{0,1,2}/round2.pt`（`4abee069…` / `a93746bb…` / `c69960fc…`，各 34,387 B）、以及引擎 `rl_v3` `691bfd77…` 均不入册（前三者已在 §33.2 登记，本轮不重复；引擎见 §2 / §8）。
- 发布目录**外的同名资产**只有执行器 1 件，已随本章入册（第 156 条），**不属跳过项**。
- 未入册合计：**0 个文件 / 0 B**。

### 35.3 与 §1–§34 的路径交集（**0**）

- 追加前 ledger **1151 行**，去重后仍 **1151 行**（重复路径 **0**）；
- 新增 **156** 条（155 件发布目录 + 1 件执行器）与既有 1151 条的**路径交集 = 0**（`set(旧) & set(新)` 实测，非目视）；
- 追加前 `Document/Update/Alpha/v8.26.0/` 前缀出现 **0** 次、`v8r0_urgency` 出现 **0** 次、`v8r0-urgency` 出现 **0** 次、子串 `v8.26` 出现 **0** 次、`URGENCY` 出现 **0** 次 ⇒ 本章**不与 §1–§34 任何一条重叠**；既有 44 个版本目录前缀（v4.1.0 … v8.25.0）中**不含 v8.26.0**；§33 登记 v8.25.0、§34 登记 v8.24.0，**版本目录互斥**；
- 新增 156 条的 SHA-256 **互不相同**（**156 个不同值 / 156 条**）⇒ 本章**无**「同内容不同路径」的别名对；且新增 156 条与既有 1151 条的**哈希交集也为 0**（`set(旧 sha) & set(新 sha)` 实测）；
- 追加后 ledger **1307 行**，去重后仍 **1307 行**（重复路径仍为 **0**）。

### 35.4 校验结论

```
wsl.exe -e bash -lc "cd /mnt/d/WorkShop/Touhou_RL && sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -c ': OK'; sha256sum -c ASSET_SYNC.sha256 2>/dev/null | grep -v ': OK'"
```

| 指标 | 追加前 | 追加后 | 变化 |
| --- | ---: | ---: | --- |
| ledger 行数 | 1151 | **1307** | +156 |
| `: OK` | 1150 | **1306** | **+156（新增 156 条全 OK）** |
| 非 OK 行 | 1 | **1** | **0**（未新增失败） |

**唯一 FAILED 行**：

```
Document/Update/Alpha/v4.2.0/journal.jsonl: FAILED
```

—— 这是工单**明确允许**的唯一 FAILED（已知的 v4.2.0 追加式 journal，属预期内），**不是本章引入的问题**，本轮**未改、未删、未重建**该文件。WSL 侧附加核对：`grep v8.26.0 | grep -c ': OK'` = **155**、`grep v8r0_urgency` = **`tools/rl/v8r0_urgency.py: OK`** ⇒ **本章 156 条 = 156 OK / 0 FAILED**，整体口径 **1306 OK / 1 FAILED（1307 行）**，与工单验收条件**一致**。

#### 35.4.1 主表（9 个 arm × tier 格，**全部满额 240 行**）

| 臂 | α | 档 | 存活/总数 | 存活率 | CP95 | bar（该档最优固定阈值） | Δpp | Fisher p(>) | p(<) | bar met | distinct 均值 | 用雷 全/存活/死亡 |
| --- | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| **G1** | 0.50 | 20 s（600） | 239/240 | **99.58 %**（0.995833） | [97.70 %, 99.99 %] | FIX 237/240（98.75 %） | +0.83 | 0.3117 | 0.9383 | **是** | 17.5792 | 2.4833 / 2.4812 / 3.00 |
| **G1** | 0.50 | 30 s（900） | 146/240 | **60.83 %**（0.608333） | [54.35 %, 67.05 %] | G0 117/240（48.75 %） | **+12.08** | **0.0051** | 0.9971 | **是** | 17.8292 | 2.9792 / 2.9658 / 3.00 |
| **G1** | 0.50 | 40 s（1200） | 34/240 | **14.17 %**（0.141667） | [10.01 %, 19.23 %] | G0 30/240（12.50 %） | +1.67 | 0.3437 | 0.7489 | **是** | 17.8708 | 2.9958 / 3.00 / 2.9951 |
| G2 | 1.00 | 20 s（600） | 239/240 | 99.58 % | [97.70 %, 99.99 %] | FIX 237/240（98.75 %） | +0.83 | 0.3117 | 0.9383 | 是 | 17.5667 | 2.5542 / 2.5523 / 3.00 |
| G2 | 1.00 | 30 s（900） | 120/240 | 50.00 % | [43.50 %, 56.50 %] | G0 117/240（48.75 %） | +1.25 | 0.4276 | 0.6425 | 是 | 17.8292 | 2.9667 / 2.9583 / 2.975 |
| G2 | 1.00 | 40 s（1200） | 9/240 | **3.75 %** | [1.73 %, 7.00 %] | G0 30/240（12.50 %） | **−8.75** | 0.9999 | **3.26e-04** | **否** | 17.8333 | 3.00 / 3.00 / 3.00 |
| G3 | 1.50 | 20 s（600） | 232/240 | **96.67 %** | [93.54 %, 98.55 %] | FIX 237/240（98.75 %） | **−2.08** | 0.9688 | 0.1106 | **否** | 17.6708 | 2.5667 / 2.5517 / 3.00 |
| G3 | 1.50 | 30 s（900） | 123/240 | 51.25 % | [44.74 %, 57.73 %] | G0 117/240（48.75 %） | +2.50 | 0.3241 | 0.7386 | 是 | 17.7917 | 2.9792 / 2.9675 / 2.9915 |
| G3 | 1.50 | 40 s（1200） | 5/240 | **2.08 %** | [0.68 %, 4.79 %] | G0 30/240（12.50 %） | **−10.42** | 1.0000 | **5.65e-06** | **否** | 17.7333 | 3.00 / 3.00 / 3.00 |

**逐格自洽**：9 个 (arm, tier) 格的 `rows = expected_rows = 240`、`cells_complete = 12/12`（= 4 窗 × 3 种子）、`complete = true`、`invalid_rows = 0` ⇒ **2,160 行无坏行**；`pooled` 合计 **1,147/2,160 = 53.10 %**。逐 (arm, tier) `wall_s_total` = **333.059 / 448.531 / 463.839** s（G1）、**330.482 / 443.843 / 443.421** s（G2）、**317.119 / 424.017 / 342.350** s（G3），**合计 3,546.661 s**（27 个评测进程按 6 个一批、5 批 6/6/6/6/3）。`distinct` 的去重键 = 每局 softmax 采样器触到的候选下标个数（共 18 个候选）；**108 个单元无一格 < 10**（逐单元 `distinct_mean` 介于 **17.15**（G1/600/w18/seed1）与 **18.00** 之间，9 个 arm×tier 均值 17.5667–17.8708）⇒ **P4 成立，无 argmax 塌缩**。

#### 35.4.2 判定链（预注册 §2 / §2.1 → `S-adaptive`）

| 判据 | 结果（`summary.verdict`） |
| --- | --- |
| **P1** 三档**同时** ≥ 该档最优固定阈值（98.75 / 48.75 / 12.50 %） | **成立，且只对 G1**：`per_arm.G1.P1 = true`（99.5833 / 60.8333 / 14.1667 三档全 met）；`per_arm.G2.P1 = false`（40 s 3.75 % < 12.50 %）；`per_arm.G3.P1 = false`（20 s 96.67 % < 98.75 %、40 s 2.08 % < 12.50 %）；`arms_bar_met_all_tiers = ["G1"]` |
| **P2** 至少一档**严格更好**（单侧 Fisher p < 0.05） | **成立，且只对 G1**：`G1.P2 = true`（`named_tiers = [900]`、`improved_tiers = [900]`，p = 0.0050783082）；G2 / G3 均 `false`、`improved_tiers = []` |
| **P3** 用雷（存活局单独报，§35.12 口径） | 已报（`summary.verdict.P3.per_arm_per_tier`，见 §35.4.1 末列） |
| **P4** `distinct` 每格 | 已报四处（arm×tier、逐窗口、逐种子、逐单元格），无格 < 10 |

⇒ `stop_condition = **S-adaptive**`；`S-adaptive.held = true`、`S-adaptive.arms = ["G1"]`；`S-partial.held = false`（`arms = []`）、`S-null.held = false`、`S-fork.held = false`（`d3_all_pass = true`，D3 **5/5**）。`stop_text` 原文：*“P1 + P2 hold for ['G1']: the urgency formula clears all three fixed-threshold bars at once (20 s >= 98.75 %, 30 s >= 48.75 %, 40 s >= 12.50 %) and is strictly better than the fixed bar on {'G1': [900]} (one-sided Fisher p < 0.05). The urgency-adaptive threshold is the unified deployment configuration.”*

#### 35.4.3 预注册读法的敏感性（`S-adaptive` vs `S-partial`，唯一分歧点）

预注册 §2 **P1** 写作「每档都 ≥ **max**(固定阈值的两臂在该档的结果)」随后枚举 **98.75 / 48.75 / 12.50 %**。盘上两个固定臂是 **G0**（35/15/3：92.50 / 48.75 / 12.50 %，来自 `.tmp_v8q0_t33tiers/eval` 720 行复算 222/117/30）与 **FIX**（60/40/20：98.75 / 24.17 / 2.08 %，来自 `.tmp_v8l0_unified/eval` 720 行复算 237/58/5），**逐档取 max 恰为 98.75 / 48.75 / 12.50**，与枚举值**完全一致**；`summary.reference.bar_note` 亦明写 *“P1's bar per tier is the better of the two fixed arms on that tier”*，`summary.reference.bar[900] = {arm: "G0", full: 117, n: 240, rate_pct: 48.75}`。⇒ **本章按枚举值 48.75 %（30 s）判定**，两处读法在该轮**不冲突**。

真正会翻成 `S-partial` 的是 **`RUN.md` §「敏感性：更严读法」** 记的另一种读法：把 **P2** 读成「**三档都要** p < 0.05」。该读法下只有 30 s 显著（0.0050783），20 s（0.31171）与 40 s（0.34367）不显著 ⇒ 判 `S-partial`。`RUN.md` 明写「本文件以预注册字面为主，严读法并列存档」，并指出两种读法给出**同一个部署结论**：只有 G1 可用，且「自适应 ≥ 固定」只在 30 s 被统计证明；差别仅在「统一配置」这四个字是否成立。

> **两种读法下的不变量**：**G1（α=0.50）是唯一三档 bar 全 met 的臂**（G2 在 40 s 差 8.75 pp、G3 在 20 s 差 2.08 pp 与 40 s 差 10.42 pp），且 G2/G3 的 40 s 崩塌在两个读法下都不变。
>
> **工单所称「另一读法 max = 58.75 %」在盘上找不到依据**：`58.75` 全文唯一出现在 `summary.json` 的 **G2@30 s seed0 = 47/80 = 58.75 %**（逐种子格，不是任何固定臂的 bar；G0@900 = 48.75 %、FIX@900 = 24.17 %）。若强行把 30 s bar 设为 58.75 %，G1 的 60.83 % **仍 ≥ bar**（P1 仍成立），变化的只是 P2 的显著性（146/240 vs 141/240 不显著）⇒ 那是 **`S-null`**，不是 `S-partial`。§35.5 第 7–8 条同此口径，**只报告、不改任何文件**。

#### 35.4.4 机制：α 越大，长段越崩（可复核因果链）

1. **开局阈值随 α 指数抬高**。40 s 开局 `urgency = R/B = 1200/3 = 400` ⇒ `scale = 2**α` ⇒ `thr[3]` = **49.50 px（G1）/ 70.00 px（G2）/ 98.99 px（G3）**，而 PARK 后第一帧最近弹距约 **24.96 px**（`smoke.json` 决策 0 `nearest_all_px = 24.9619`）⇒ 三臂在 40 s 的 `first_bomb_decision` 均值都是 **0.00**（开局立刻放雷）。
2. **公式自带正反馈**：放掉一颗后 `B` 减 1、`R` 几乎不变 ⇒ `urgency` 变大 ⇒ `scale` 变大 ⇒ 阈值更松。`B=1` 时 40 s `urgency ≈ 1200` ⇒ `scale = 6**α` = 2.45 / 6.00 / 14.70 ⇒ `thr[3] = 85.8 / 120（封顶）/ 120（封顶）`。
3. **实测后果**（40 s，三档同一生成器流）：

| α | 开局 `thr[3]` | 放雷时阈值均值 | 放雷时最近距离均值 | 最后一颗雷的中位决策 | 用尽 3 颗雷的局 | 存活 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50（G1） | 49.50 px | 25.18 px | **16.91 px** | 600.5 / 1200 | 239/240 | **34/240 = 14.17 %** |
| 1.00（G2） | 70.00 px | 40.53 px | 21.23 px | 534.0 / 1200 | 240/240 | 9/240 = 3.75 % |
| 1.50（G3） | 98.99 px | 68.58 px | **25.58 px** | 500.5 / 1200 | 240/240 | 5/240 = 2.08 % |

即：α 越大 ⇒ 雷在**更远**的距离被丢掉（16.9 → 21.2 → 25.6 px）、且**更早**丢完（中位 601 → 534 → 501），剩下的决策段完全无雷。两个因素的相对贡献本轮**不做分解**（没有消融臂，α 同时控制两者）。

**实际落地的阈值调度**（`RUN.md` 表，`thr@B=3` 为放第一颗雷时的阈值）：G1@600 **34.99 [34.88, 35.00]**、G1@900 **42.87**、G1@1200 **49.50**；G2@1200 **70.00**；G3@1200 **98.99** px；`thr@B=1` 依次被压到 **5.00 / 5.07 / 5.81 / 11.82 / 24.89** px ⇒ **下限 5 px 与上限 120 px 两条硬裁剪都在实测中被触到**（预注册 §1.1 的下限保护生效）。

**α 的单调性**：20 s 单调（99.5833 → 99.5833 → 96.6667，`YES`）、40 s 单调（14.1667 → 3.7500 → 2.0833，`YES`），**30 s 不单调**（60.8333 → 50.0000 → **51.2500**，`NO`；G2 vs G3 精确 p = 0.642489）⇒ α≥1 之后 30 s 档**平台化**，增加 α 不再有益。

#### 35.4.5 测量纪律：部分采样不可判定（实证）

评测 27 路并行时盘上的 `eval_*.jsonl` 是**边跑边写**的，任何中途池化都是**采样未满**的读数（`RUN.md` §「Measurement discipline」）：

| 臂 | 档 | 仅 w11 | w11+w12 | 3 窗 | 满 4 窗（终值） | 摆幅 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| G1 | 600 | 100.00 % | 100.00 % | 99.44 % | **99.58 %** | 0.56 pp |
| G1 | 900 | 63.33 % | 68.33 % | 57.22 % | **60.83 %** | 11.11 pp |
| G1 | 1200 | 23.33 % | 20.83 % | 16.11 % | **14.17 %** | 9.17 pp |
| G2 | 900 | 66.67 % | 56.67 % | 48.33 % | **50.00 %** | 18.33 pp |
| G2 | 1200 | 5.00 % | 5.00 % | 3.89 % | **3.75 %** | 1.25 pp |
| G3 | 600 | 93.33 % | 94.17 % | 96.11 % | **96.67 %** | 3.33 pp |
| G3 | 900 | 58.33 % | 58.33 % | 46.67 % | **51.25 %** | 11.67 pp |
| G3 | 1200 | 3.33 % | 3.33 % | 2.22 % | **2.08 %** | 1.25 pp |

⇒ **登记/引用只能取满 4 窗的终值**：本轮主代理过程中曾读到 **G3@1200 `n = 180`**（当时 w18 尚未落盘），跑满后是 **5/240 = 2.08 %**；G3@900 也出现过「n=87 时 62.07 % → 满额 51.25 %」的摆动。凡未标 `n` 的中途数字**不得用于判定**（与 §34.4.5 同型教训）。

#### 35.4.6 旁证（与 §8 / §33 / §34 已知身份对齐）

| 项 | 实测 | 说明 |
| --- | --- | --- |
| 引擎 | `rl_v3` `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`，`match: true` | 与 §2 / §8 已知 SHA **相同**（`ckpt_assertions_{start,final}` 与 `smoke.json` 三处同值） |
| 候选头 | `.tmp_v860_bombdag/ckpt/seed{0,1,2}/bombdag.pt` = `9c6d5d4a…` / `36a51090…` / `885dde70…`（各 39,069 B），`match: true` | 与 §33.2 / §34.4.6 **同值** |
| 行为策略 | `.tmp_v450_cf/ckpt/seed{0,1,2}/round2.pt` = `4abee069…` / `a93746bb…` / `c69960fc…`（各 34,387 B），`match: true` | 与 §33.2 / §34.4.6 **同值** |
| 预注册稿 | `PREREGISTRATION_URGENCY_001.md` = `9d462c1f4b0f125212121c1e49e27b87802a5beed2ea39e60183e597f8a6be7f`（4,272 B） | `summary.prereg` / `ckpt_assertions_{start,final}.prereg` / 盘上**四处同值**；`RUN.md` 头部亦引用同值 |
| 执行器 | `tools/rl/v8r0_urgency.py` = `4e81f9111293148ff0221e5df4618702c4ad4069f4f070a50ffe2a26fe2ebf0e`（115,154 B） | `ckpt_assertions_start.self` = `ckpt_assertions_final.self` = **盘上现值** ⇒ **本轮轮内零漂移**（与 §34「轮内漂移」相反，可作正对照） |
| D3 保真门 | `d3_gate.json`：**5/5 PASS**（`1_hello_role_shadow`、`2_debug_threads_1`、`3_prefix_2165_checkpoints`（2,165 个检查点 0 不符、末帧 8663）、`4_r24_18_constants`（18/18）、`5_park_frame_8663`），`exit = 0`、`wall_s = 5.646`、`raw_sha256 = ecc7e088…` | `S-fork` 不触发 |
| 冒烟 | `smoke.json`：G2 / α=1.00 / seed0 / w11 / 600 决策，**2 局 / 16 次放雷 / 前 5 次全打印**，`checks` **18/18 True**（含 `formula_reproduced`、`thresholds_in_bounds`、`one_sample_per_decision`、`cfg_muted_four_zero`），`gate_prefix` 2,165 检查点 0 不符 | 首帧 `nearest_all_px = 24.9619`、决策 0 即 `thr[3] = 35.00` px |
| 音频 | `.tmp_v8r0_urgency/data` 下 **109 个 `th08.cfg`，解码 109、errors 0、muted 109、audible 0**，`audible_paths = []`，`all_pass = true` | **AGENTS §9.3 合规**（`musicMode` / `playSounds` / `musicVolume` / `sfxVolume` 四字段全 0；冒烟那件的 `sha256 = cc062cfb…`、60 B） |
| 引擎残留 | `process_residue.json`：`needle = .tmp_v8r0_urgency`、本轮 **0** 个、`matching = 0`、`lines = []`、`other_round_processes = 0`、`all_pass = true`；`preflight_start`（17:40:21）记**他轮 18 个**引擎（全属 `.tmp_v8p0_bombrl900`）、`matching = 0`、`lines_mine = []`、`nproc = 20`、`waited_s = 0`；`preflight_final`（17:55:21）全 0 | **本轮 0 残留**，他轮进程**未被碰**（未 `pkill`） |
| 评测退出 | `eval_exits.tsv` **27 行 / 27 个进程 / `exit` 全 0**，时间戳 17:43:07 → 17:54:57 单调；`commands.tsv` **1 表头 + 27 行**（pid 13.11M–13.14M、`started` 17:40:52） | `invalid_rows = 0` ⇒ 2,160 行无坏行 |

### 35.5 与 manifest 不符项（**只报告、不改**）

> **总结论：`evidence/URGENCY/manifest.json` 与盘上 0 不符。** 逐项比对 `{bytes, sha256}`：**153/153 全等**；`bytes` 求和 = `total_bytes`（**18,805,573**）**差 0**；「manifest 有而盘上无」**0 项**；「盘上有而 manifest 无」= **1 项**（`manifest.json` 自身，21,968 B，在 `root` 之内但不可能自哈希）⇒ **无缺件、无多余件**。`publish.json` 的 **147 条 `{name, sha256}` 与盘上 147/147 全等**（合计 18,724,397 B）。下面各项**不是 manifest 的错**，而是「manifest 覆盖不到」或「manifest 之外的证据之间口径不同」，按工单**只报告、未改任何文件**。

| # | 项 | 事实（实测） | manifest 侧 | 处置 |
| ---: | --- | --- | --- | --- |
| 1 | **覆盖齐口可精确闭合（147 / 153 / 154 / 155 / 156）** | `publish.count = 147`（= 108 eval + 27 progress + 12 件其它分析/门/日志；`publish.json` 17:58:19 先写、`manifest.json` 17:58:23 后写）；`manifest.file_count = 153`（排除自身）；`evidence/URGENCY/` 盘上 **154**；整目录 **155**。闭合：**147 + `publish.json` + 5 件 `supplementary` = 153**（= manifest 件数）、**153 + `manifest.json` = 154**、**154 + 顶层预注册稿 = 155**、**155 + 执行器 = 156** | 三者互不矛盾 | **只报告**；147 条 `sha256` 与盘上 **147/147 全等**（0 不符） |
| 2 | **manifest 不含执行器，且执行器本轮「轮内零漂移」** | `ckpt_assertions_start.self` = `ckpt_assertions_final.self` = 盘上现值 **`4e81f911…` / 115,154 B**（三次同值）⇒ 与 §34.5 第 2 条（start `00fda3fc…` → final `71bfbeb6…` **漂移**）**恰好相反** | manifest **不含执行器**（`root` 之外），**不冲突** | **只报告**；本章可作「轮内零漂移」正对照，**但不要**与 §34 互引口径 |
| 3 | **`protected{}` 11 件没有 `expected`，`all_pass` 对它们没有约束力** | 两份 `ckpt_assertions` 的 `protected{}` 各 **11** 件（`v432_fork_arm.py` 123,448 B、`v450_cf_label.py` 67,077 B、`v431_arm_a.py` 15,697 B、`v800_bomb.py` 92,126 B、`v830_bombpol.py` 138,057 B、`v860_bombdag.py` 126,406 B、`v8l0_unified.py` 84,166 B、**`v8o0_ruletight.py` 85,075 B（本轮新增）**、**`v8q0_t33tiers.py` 89,760 B（本轮新增）**、`touhou_rl/v420/contracts.py` 56,321 B、`touhou_rl/v400/gym/actions.py` 3,813 B），字段只有 `{exists, sha256, bytes}`，**`expected` 计数 = 0** ⇒ 即使受保护文件被改，`all_pass` 仍是 `true`。**实测 start 与 final 的 11 件逐条相同（0 漂移）** | manifest 不涉 | **只报告、不改**（与 §27.5 第 9 条、§32.5 第 4 条、§33.5 第 3 条、§34.5 第 3 条同型；本轮 `expected` 件数 = **0**、受保护件数从 §34 的 9 升到 **11**） |
| 4 | **`manifest.root` 是仓库相对路径，`publish.publish_dir` 是 WSL 绝对路径** | `manifest.root = Document/Update/Alpha/v8.26.0/evidence/URGENCY`（**可直接拼接，与 §34 的 WSL 绝对路径不同**）；`publish.publish_dir = /mnt/d/WorkShop/Touhou_RL/Document/Update/Alpha/v8.26.0/evidence/URGENCY`；`summary.prereg.path` / `ckpt_assertions.prereg.path` 亦为仓库相对路径 | 只记字节 | **只报告**；本章 §35.1.1 **已统一写成仓库根相对路径**，可直接与 ledger 对拼 |
| 5 | **`RUN.md` 尾部整块自重复（发布件自身瑕疵）** | `RUN.md` **44,647 B / 695 行**。`## Extra numbers (read-only re-derivation…)` 至文末在 **1-based 第 215–454 行**与**第 456–695 行**各出现一次，`difflib` ratio = **1.0000**（203/203 行零差异、4 个 opcode 全 `equal`）⇒ 实为同一块重复两遍。**工单写「12 KB」与盘上 44,647 B 不符**（有效内容约 22 KB） | manifest 只记字节（44,647 B 与盘上一致） | **只报告、不改**；引用 `RUN.md` 时注意同一结论在文中出现**两次**，勿误当成两组独立证据 |
| 6 | **工单「81 个 eval jsonl」与盘上 108 不符** | 盘上 `eval_G{1,2,3}_d{600,900,1200}_w{11,12,17,18}_seed{0,1,2}.jsonl` = **108 个**（3 臂 × 3 档 × 4 窗 × 3 种子 **全齐**，缺格 **0**），每件恰 20 行 = **2,160 行**；另 **27 个 `G*_d*_s*_progress.json`** 是独立 `.json`（不是 `.jsonl`）。合起来恰是 `.jsonl` 108 件 + `.json` 39 件 | manifest 153 件**全部收录** | **只报告**；本章**以盘上为准登记 108 个**，未按工单的 81 裁剪 |
| 7 | **预注册 §2 P1 的 `max(...)` 措辞与枚举值** | 两个固定臂逐档取 max = **98.75 / 48.75 / 12.50 %**，与预注册枚举值**完全一致**（`summary.reference.bar_note` 亦如此读，`bar[900].arm = "G0"`、`full = 117`、`n = 240`）⇒ **按枚举值 48.75 %（30 s）判定**。真正给出 `S-partial` 的是 `RUN.md` §敏感性 的**严读法**（P2 要求三档都 p<0.05），详见 §35.4.3 | 不涉 | **只报告**；两种读法下「**G1 是唯一三档 bar 全 met 的臂**」不变 |
| 8 | **`58.75 %` 不是任何固定臂的 bar** | `58.75` 在发布目录内**唯一出处**是 `summary.json` 的 **G2@30 s seed0 = 47/80 = 58.75 %**（逐种子格）。盘上 30 s 的两个固定臂值是 **G0 = 48.75 %**、**FIX = 24.17 %**。若强把 30 s bar 当 58.75 %：G1 的 60.83 % **仍 ≥ bar**（P1 仍成立），只是 P2 不再显著（146/240 vs 141/240）⇒ 那会判 **`S-null`**，**不是** `S-partial` | 不涉 | **只报告**；引用 30 s bar 一律写 **48.75 %**，不要写 58.75 % |
| 9 | **冒烟「前 5 次放雷」来自 2 局而不是 1 局** | `smoke.json`：`episodes_requested = 1` 但 `rows` **2** 条、`emissions` **16** 次、`first_5_emissions` **5** 条。`RUN.md` §偏差 1 说明：ep0 只有 **4** 次放雷（无敌帧使部分放雷被丢弃 → `bomb_ignored`），凑不满 5 次，故实现为「最多 3 局、凑满 5 次即停」 | manifest 只记字节 | **只报告**；**不影响**判定（`checks` 18/18 True，正式 eval 仍是 20 局/窗、每格 240 条） |
| 10 | **`commands.tsv` 有表头、`eval_exits.tsv` 无表头（与 §34 那轮口径不同）** | `commands.tsv` = **1 表头 + 27 行**（`arm/budget/seed/tag/pid/started`，**无 `wall_s` 列**）；`eval_exits.tsv` = **27 行无表头**（`tag/exit/finished_at`），`exit` **全 0**、时间戳 17:43:07 → 17:54:57 单调。逐进程 `wall_s` 只能在 `summary.arms.<arm>.<tier>.wall_s_total` 读（G1 333.059/448.531/463.839 s 等） | 只记字节 | **只报告**；**不影响可复算性**，但**不能**照 §34.5 第 6 条那样逐行对 `wall_s` |
| 11 | **`audio_audit` / `process_residue` / `preflight_*` 是**本轮**自证，不是他证** | `audio_audit`：`scope = .tmp_v8r0_urgency/data`、**109 cfg / 109 muted / 0 audible / errors 0**；`process_residue`：本轮 **0**、`lines = []`；`preflight_start`：**他轮 18 个**引擎（`.tmp_v8p0_bombrl900`）、`matching = 0`；`preflight_final`：全 0 | 不涉 | **只报告**；三者都是**本执行器自报**（`producer = tools/rl/v8r0_urgency.py`），不是外部审计；但 `preflight_start`（17:40:21，他轮 18）→ 首进程（17:40:52）→ `preflight_final`（17:55:21，全 0）→ `process_residue`（17:55:21，本轮 0）**四条时间线自洽** |

**逐条核对口径（可复算）**：把 `manifest.json` 的 `files{}` 逐项与 `sha256sum` 实测比对，`{bytes, sha256}` **不符数 = 0**（153 项全过）；`bytes` 求和与 `total_bytes` **差 0**（18,805,573）；「盘上有而 manifest 无」= **1 项**（`manifest.json` 自身）、「manifest 有而盘上无」= **0 项**。`publish.json` 的 147 条 `{name, sha256}` 亦与盘上 **147/147 全等**。旁证：`summary.json`、两份 `ckpt_assertions` 的 `prereg.sha256`、`RUN.md` 头部引用的都是 `9d462c1f…`，与盘上预注册稿（4,272 B）**逐位相同**；`engine` 在各处都是 `rl_v3` `691bfd77…`、`match: true`（与 §2 / §8 已知 SHA 相同）；`protected` 11 件 start/final **0 漂移**。

> **一句话结论**：本轮把 **v8.26.0 发布目录 155 件 / 18,831,813 B + 执行器 1 件 / 115,154 B = 156 条 / 18,946,967 B** 成对写入 `ASSET_SYNC.md` §35 与 `ASSET_SYNC.sha256`，`sha256sum -c` 由 **1150 OK / 1 FAILED（1151 行）** 变为 **1306 OK / 1 FAILED（1307 行）**（新增 156 条**全 OK**，FAILED 数仍为 1、仍为已知允许的 `Document/Update/Alpha/v4.2.0/journal.jsonl`）；9 个 (arm, tier) 格**全部满额 240 行**、`invalid_rows = 0`、`cells_complete = 12/12`，判定 **`S-adaptive`**（`S-adaptive.held = true`、`arms = [G1]`；`S-partial` / `S-null` / `S-fork` 均 false，D3 **5/5**）：**G1（α=0.50）三档 bar 全 met** —— 20 s **99.58 %**（239/240，p = 0.3117 不显著）、30 s **60.83 %**（146/240，**p = 0.0051 显著**、+12.08 pp）、40 s **14.17 %**（34/240，p = 0.3437 不显著）；**G2（α=1.00）= 99.58 / 50.00 / 3.75 %**（40 s 显著更差 `p_less = 3.26e-04`、bar **未 met**）、**G3（α=1.50）= 96.67 / 51.25 / 2.08 %**（20 s 与 40 s 都未 met）；存活局用雷 **2.48 / 2.97 / 3.00**、`distinct` 9 格均值 **17.57–17.87**（逐单元 17.15–18.00，**无一格 < 10**，非 argmax）；音频 **109 cfg muted / 0 audible**、本轮引擎残留 **0**、`invalid = 0`、评测 **27 进程 exit 全 0**；**预注册 §2 P1 的 `max` 措辞与枚举值在本轮不冲突（逐档 max 恰为 98.75 / 48.75 / 12.50）**，本章**按枚举值 48.75 % 判定**，另一读法（`RUN.md` 严读法：P2 要求三档都显著）会得到 `S-partial`，两种读法下「**G1 是唯一三档 bar 全 met 的臂**」不变；manifest 逐条比对 **153/153 全等、0 处不符**，差额恰为 `manifest.json` 自身（21,968 B）⇒ **无缺件、无多余件**。**未启动引擎、未训练、未评测、未 `git add/commit`**，未改动、删除或移动任何证据文件。

## 36. v8.27.0 资产（2026-11-22）

### 36.1 登记范围与逐文件表

本轮登记 `Document/Update/Alpha/v8.27.0/` **整目录**（`Get-ChildItem -Recurse -File` 实测 **70 件 / 22,098,054 B = 21.0743 MB**）与**执行器** `tools/rl/v8t0_densereward.py`（148,000 B，`877f4581…`），合计 **71 条 / 22,246,054 B**。**入册 71、未入册 0**（跳过项见 §36.2）。目录内 `evidence/DENSEREWARD/` **68 件 / 22,092,372 B**；顶层 2 件为本轮预注册稿 `PREREGISTRATION_DENSEREWARD_001.md`（4,024 B，`ad8e6416…`）与范围修正 `ERRATA_SCOPE_REDUCTION.md`（1,658 B，`277bda6a…`，两者 sha256 与 `summary` 自述逐位相同）。最后落盘 **18:30:31**（`manifest.json`，其 `created_at = 2026-11-22T18:30:32+0800`），即工单所称「18:31 发布完毕」。

| 组 | 文件数 | 字节 | 目录实测 | manifest 覆盖 |
| --- | ---: | ---: | --- | --- |
| `Document/Update/Alpha/v8.27.0/**` | **70** | **22,098,054** | `Get-ChildItem -Recurse -File` 70 件 / 22,098,054 B（21.0743 MB） | `evidence/DENSEREWARD/manifest.json` 自述 **67** 件 / **22,082,868** B（`root` = `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD`，以 `manifest.json` 自身排除） |
| `tools/rl/v8t0_densereward.py` | **1** | **148,000** | — | manifest **不含执行器**（`root` 之外） |
| **合计** | **71** | **22,246,054** | — | — |

口径自洽（逐位可复算）：

| 目录 | 件数 / 字节 | `evidence/DENSEREWARD/` | `manifest.json` | manifest 件数 / 字节 | 顶层 2 件 | 差额 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v8.27.0 | 70 / 22,098,054 B | 68 / 22,092,372 B | 9,504 B | 67 / 22,082,868 B | `PREREGISTRATION_DENSEREWARD_001.md` 4,024 B + `ERRATA_SCOPE_REDUCTION.md` 1,658 B | 22,082,868 + 9,504 = **22,092,372**（+ 5,682 = **22,098,054**） |

文件类型构成（实测，**无 `.tmp_*`、无 `*.tmp`、无 `tmp_*`、无 `.partial`、无 `~` 结尾、无点开头件、无 Hidden/System 属性件**——`-Force` 实测 70 件、可疑 0）：

| 目录 | `.jsonl` | `.json` | `.md` | `.pt` | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v8.27.0 | 30 / 20,219,443 B | 25 / 1,490,239 B | 3 / 22,852 B | 12 / 365,520 B | **70 / 22,098,054 B** |

> 字节重心是 **30 个 `.jsonl` 共 20,219,443 B（占该目录 91.50 %）**：其中 **24 件评测原始行** `eval_D{1,2}_d900_w{11,12,17,18}_seed{0,1,2}.jsonl` 共 **19,317,937 B**（每件恰 150 行、**657,922–1,061,357 B/件**，合计 **3,600 行 = 2 臂 × 3 种子 × 4 窗 × 150**），另有 **6 件训练逐回合** `seed{0,1,2}_{D1,D2}.jsonl` 共 **901,506 B**（每件 300 行、共 **1,800 行**），两类合计 **5,400 行**；**12 个 `.pt` 共 365,520 B** = 6 个 BC 热启动头（30,475 B 各）+ 6 个 d900 训练后头（30,445 B 各）；其余 **25 个 `.json`（1,490,239 B）+ 3 个 `.md`（22,852 B）** 中体量最大的是 `train_seed2_D2.json` 223,738 B、`summary_D{1,2}.json` 各 62,431 B、`audio_audit.json` 24,835 B、`RUN.md` 17,170 B、`train_curve.json` 15,151 B、`manifest.json` 9,504 B、`publish.json` 8,948 B，最小的是 `process_residue.json` 553 B。发布时序（**已发布件的 mtime 是复制时刻，不是产生时刻**，见 §36.4 第 6 条）：BC 头 18:09:52 → 预训练元数据 18:10:09 → 训练冒烟 18:10:23 → 训练产物 18:17:02–18:17:58 → 评测 24 格 18:21:41–18:27:44 → `d3_gate`/`ckpt_assertions_start` 18:28:00 → `audio_audit` 18:28:31 → `summary`/`train_curve`/`bomb_usage` 18:28:32 → `ckpt_assertions_final` 18:29:54 → `RUN.md` 18:30:05 → `process_residue` 18:30:24 → `publish` 18:30:30 → `manifest` 18:30:31。

#### 36.1.1 逐文件表（71 条：70 件发布目录 + 1 件执行器）

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `Document/Update/Alpha/v8.27.0/ERRATA_SCOPE_REDUCTION.md` | 1,658 | `277bda6aff3524074463030f9e177e410dca472d9b1c58e30b4f552ad8508c40` |
| 2 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/audio_audit.json` | 24,835 | `780363ffdd829241c863c81edc734009fa2c4608a5c4f8413e800342f8c08fd8` |
| 3 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/bomb_head_seed0_bc_D1.pt` | 30,475 | `38e4f6e47d4ac14917d2f90c5ce544f8840a190fb8aeadc90c581f1f61936c29` |
| 4 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/bomb_head_seed0_bc_D2.pt` | 30,475 | `38e4f6e47d4ac14917d2f90c5ce544f8840a190fb8aeadc90c581f1f61936c29` |
| 5 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/bomb_head_seed0_d900_D1.pt` | 30,445 | `af62c20ed30cbbf616ef44cd5fdc9ed7c48875975e0bf54e5590d20b0dc57964` |
| 6 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/bomb_head_seed0_d900_D2.pt` | 30,445 | `02d3122493d0a498feb8ee43008157cccc7391c11fd00c82089d319959fc2d49` |
| 7 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/bomb_head_seed1_bc_D1.pt` | 30,475 | `e662ed408ecee56a5ea5dbafd059175a8ff2ca5a345ced02d1a729ed3822f90e` |
| 8 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/bomb_head_seed1_bc_D2.pt` | 30,475 | `e662ed408ecee56a5ea5dbafd059175a8ff2ca5a345ced02d1a729ed3822f90e` |
| 9 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/bomb_head_seed1_d900_D1.pt` | 30,445 | `351cc04d8c47e43b0e2d1249cc2907eb6d105b0421fc59f05c2d8fb24d282162` |
| 10 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/bomb_head_seed1_d900_D2.pt` | 30,445 | `86a573caa3340f99594b77ca44a8d64dccb4422b7db726952911bc9fcebc4043` |
| 11 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/bomb_head_seed2_bc_D1.pt` | 30,475 | `7d926f735f8b7b0f307c64d9cd6ff527d08f256edce9822fe8241fe40e6fc0b0` |
| 12 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/bomb_head_seed2_bc_D2.pt` | 30,475 | `7d926f735f8b7b0f307c64d9cd6ff527d08f256edce9822fe8241fe40e6fc0b0` |
| 13 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/bomb_head_seed2_d900_D1.pt` | 30,445 | `86a0cb5d8f14a3e1b1c69bf3b2456a9b444eb2d3b135cf8b757ad873ce71fd17` |
| 14 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/bomb_head_seed2_d900_D2.pt` | 30,445 | `c2a7df406ab1c901d9102adf055104ea75b782c79ed4878d041f54c8d1f33d98` |
| 15 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/bomb_usage.json` | 4,003 | `8d1ca7726f8642cae36be963514b401b54565274ab7bf1fbd17c89ad1f18821f` |
| 16 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/ckpt_assertions_final.json` | 5,749 | `efd1e87025e8e44b9f0e3bb1235a29291f180e5c3835e46d0b2a5f9fcc714451` |
| 17 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/ckpt_assertions_start.json` | 5,563 | `5de8611006b1d668fa17c6a3d520d4e4fe06cbb6f253027c6da9e2a525ec387a` |
| 18 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/d3_gate.json` | 5,170 | `f04b803dd832198cc0b960714a9b48c94b66b90a77eeb7fdcfd7f9bdf145e362` |
| 19 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D1_d900_w11_seed0.jsonl` | 771,202 | `593e0fa74603ed3f6c2e02b4171bf89ff78d3945f2d6225142eb6cc78bae39ad` |
| 20 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D1_d900_w11_seed1.jsonl` | 815,356 | `65ede490f86c7b5cf1bf5d872d95b62a1934826adf5fb4e03b53479d18cfe90a` |
| 21 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D1_d900_w11_seed2.jsonl` | 789,052 | `3c9e4f536418965eb302586924aeabc95366f313d3b91dd46629537336c0ef94` |
| 22 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D1_d900_w12_seed0.jsonl` | 733,641 | `e25b998ffc9cc9e1d7c0198a4aa177ec1b3c9d0facfa8a133f4890cf9b654a63` |
| 23 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D1_d900_w12_seed1.jsonl` | 762,915 | `041a3051f15e67664255fb4c3ce04c815348a4e4dcd26deb2c512bad7ceb71b9` |
| 24 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D1_d900_w12_seed2.jsonl` | 774,208 | `44f985a16a1c0f95dbb126acd7399331a11ba3dd75188df3dc7240ec0bb6895c` |
| 25 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D1_d900_w17_seed0.jsonl` | 791,974 | `9d21913fe32856ae8c20c12640b9ee218bc5bd161b08338b29e7fcd034a74a22` |
| 26 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D1_d900_w17_seed1.jsonl` | 657,922 | `735105fd09b6d0a1d966be35ac6a3319227218cb11e12903ce206bde15009ee9` |
| 27 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D1_d900_w17_seed2.jsonl` | 711,550 | `31307f4ff987085c8bb5c00ad9ec291977822116b41c5f25d84e2734b8b5582d` |
| 28 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D1_d900_w18_seed0.jsonl` | 840,737 | `2c1040ebb3f5c2f02094921ba07950bcf665c8f8b7c5d3dd2eb59418de95ed66` |
| 29 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D1_d900_w18_seed1.jsonl` | 809,690 | `6538fda524a0861f60384c08cbaf6aa900843d2c2672d943ba8648446c5bd0d9` |
| 30 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D1_d900_w18_seed2.jsonl` | 790,613 | `11453af88178ccfc3bc898991d833487f110b3061bd215a76963a39fe4450d25` |
| 31 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D2_d900_w11_seed0.jsonl` | 766,906 | `751437f04f8b9db9f2691a6afb5f80e9ebaea4756d4bc47986535b87ebd52545` |
| 32 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D2_d900_w11_seed1.jsonl` | 814,183 | `f2522d4fc79def0baafc4c8a4c21242d7e800d467af51ff34f0275f28f1435b6` |
| 33 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D2_d900_w11_seed2.jsonl` | 1,033,687 | `2c5a34395faf41047b7c67ce35ae982fc8e0a0983dcdf392e94f9a2dcc07ff71` |
| 34 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D2_d900_w12_seed0.jsonl` | 738,951 | `426a3f3f1e191b06be6f1aee4bb28c96bf6fe981268ac705e1a7129af017cda8` |
| 35 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D2_d900_w12_seed1.jsonl` | 789,388 | `a18b80472d79354ec19ae46edc9fa6a2589fab385a74da2f154bf803d91742dc` |
| 36 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D2_d900_w12_seed2.jsonl` | 1,061,357 | `7ad4d78d66ebe8980654ab7593b24a59b4fd356d76b40cc91d85aa024fd84ec7` |
| 37 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D2_d900_w17_seed0.jsonl` | 781,265 | `1daf2dea4dcfb911a8af96b525a0bf38b2e0e53c64cc102dacf2954bb87a60f2` |
| 38 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D2_d900_w17_seed1.jsonl` | 674,958 | `a075f095ce7ccf50afd9ea4ed3ad193ca9987cba6dbdfe408266715014b15ec2` |
| 39 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D2_d900_w17_seed2.jsonl` | 849,862 | `4e4f686765b2e7c211e6a69ca13efd481ede0bc7fbec42691b8da77441c6ebfb` |
| 40 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D2_d900_w18_seed0.jsonl` | 839,465 | `6ef3a0c74c5aeca56925f73281abe056cfb12f65e2cf7f4541ca619bf0b6c146` |
| 41 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D2_d900_w18_seed1.jsonl` | 811,458 | `38319a48503ffb9f572476f5f4a97702254210fcc0ab370bae44a5f54edc38ef` |
| 42 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/eval_D2_d900_w18_seed2.jsonl` | 907,597 | `90f22c5b01c436da4f60db5214b3bd543cc9e4b226ef8bad038a2d8417b8df99` |
| 43 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/manifest.json` | 9,504 | `461a122cbdc0856cc026b9d0e56f389bbec4ec6f007d505f215582f79f76ab72` |
| 44 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/pretrain_seed0_D1.json` | 5,206 | `a8e0dd6be6a51425d94fd1b7d7bd34ed216a630f1fb0466abde2f51062282894` |
| 45 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/pretrain_seed0_D2.json` | 5,205 | `4c8f31bc56957c30fcad2546f7ba892bd7889ec15eefec627e9747c85b1368ce` |
| 46 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/pretrain_seed1_D1.json` | 5,198 | `e018d0a49e8d4710b1b187ae36a73a24814ce3a04221df59354b0079252f097b` |
| 47 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/pretrain_seed1_D2.json` | 5,201 | `ede3cf10fc0380ca8dabcecf053ba8f75a8af4e6cdf04f7785ff1a92578dbb84` |
| 48 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/pretrain_seed2_D1.json` | 5,204 | `717727a890b698f02f6bef73f27562b636578cbc863408c60661efb21caa9371` |
| 49 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/pretrain_seed2_D2.json` | 5,203 | `44969b837ed8a1cfae6fa7dfdb1858bcb5e95f7bcd3dcd57d2fdf94ef980e7e8` |
| 50 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/process_residue.json` | 553 | `7d3aca57045500663c3cfc107923d491e803e103048e9bcdc6968e36d12eeff0` |
| 51 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/publish.json` | 8,948 | `ecb87638c09618f195b16f919e329f16531f679139ddb90877fed850ca6faca1` |
| 52 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/RUN.md` | 17,170 | `cbe0c36b4bffe0c842eeb847921fa6f0161662e9dd4b2e652c6f29be3b3a5a20` |
| 53 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/seed0_D1.jsonl` | 151,706 | `155e32a59a910632b8f8f5df6abfdb84443b11b032adc549d0f4c6da9ff83203` |
| 54 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/seed0_D2.jsonl` | 155,893 | `9a0da199d1b1209b1b135b0941621a5fbe1f327f017fa6554d364be6fd68d4f2` |
| 55 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/seed1_D1.jsonl` | 143,266 | `faf2bdb3842fb314d435694a0903c218e708870af4da99c97abead03138a2161` |
| 56 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/seed1_D2.jsonl` | 144,746 | `8fc922d4a503ad4bd9fa529e2c0e07e9d8a5de98ce3e4102d08224d60786eebd` |
| 57 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/seed2_D1.jsonl` | 148,846 | `50ef3f64955595292c5b857066b85d3789f767b4965cbb0a30f79ba2d21ba417` |
| 58 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/seed2_D2.jsonl` | 157,049 | `777d04afaa952067cd978d8f3cf0392e09fe1333c428a804f601d0728e928192` |
| 59 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/summary_D1.json` | 62,431 | `1e2de30697143ba19f591cffdbf7de0f0c205d19b7ad9d64d0c69b729ae3efc3` |
| 60 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/summary_D2.json` | 62,431 | `1e2de30697143ba19f591cffdbf7de0f0c205d19b7ad9d64d0c69b729ae3efc3` |
| 61 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/train_curve.json` | 15,151 | `f051100d47350a021518870a4e4cd54319552f157b96d1856a0b9a630ee20c76` |
| 62 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/train_seed0_D1.json` | 211,147 | `f274813ae86c1d415147bbf0a4cafcf7b995f7503553a9c624bb709ad4ec6eb3` |
| 63 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/train_seed0_D2.json` | 221,037 | `54097e39c0444e30b5bced0fc3413490271fe04e0d8b84db1d6478ee8805cfbe` |
| 64 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/train_seed1_D1.json` | 191,569 | `9526523a7c43a1e019c37c0c676475dd52cc03ead620a13dd100a4a254dd8f40` |
| 65 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/train_seed1_D2.json` | 194,938 | `f282ace3bb94da5f2ce3e0e74ef265eec32e36f1d0989713567e6876ef3c5c80` |
| 66 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/train_seed2_D1.json` | 204,628 | `22784cb9e672b6be1d2d075bfbb37a226638626840bf177d828b197854dc90ec` |
| 67 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/train_seed2_D2.json` | 223,738 | `daa437688f85eaaebddc22c1eb8a857bf57d79c6d6c03750d7f8cc6a4a804bf3` |
| 68 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/train_smoke_D1.json` | 3,813 | `db2fcdaa17490ba9afffd5de7a04396efce56d26e9d2e77ca601532585f0ef9a` |
| 69 | `Document/Update/Alpha/v8.27.0/evidence/DENSEREWARD/train_smoke_D2.json` | 3,814 | `0304fefa843cb9211ad4f9b228c991ec39f67a67b766e0d9ec9f40703123eb79` |
| 70 | `Document/Update/Alpha/v8.27.0/PREREGISTRATION_DENSEREWARD_001.md` | 4,024 | `ad8e6416dc30f706aa1db48a3e3cd061964d1153de5663e842b95c1924fa7345` |
| 71 | `tools/rl/v8t0_densereward.py` | 148,000 | `877f4581b010712a1380db4b6978db740b8cc333e1cb6916396c33be9b9b152d` |


### 36.2 跳过说明

本目录**无跳过件**：70 件发布目录文件 + 1 件执行器 = 71 条**全部入册**，未入册 **0**。以下物件**不属登记范围**，其存在/缺失都不构成缺件：

| 物件 | 处理 | 理由 |
| --- | --- | --- |
| `.tmp_v8t0_densereward/`（`D1/`、`D2/` 各含 `ckpt/`、`data/`、`eval/`、`train/`、`runs/`，及 `summary.json`） | 不入册，仅引用路径 | §8.1：本机原始运行目录，不是发布件；其 `summary.json` 与已发布 `summary_D1.json`/`summary_D2.json` **逐字节相同**（`1e2de306…`，62,431 B），不入册不损失任何信息 |
| D3（折扣密集臂）、20 s / 600 档、40 s / 1,200 档、确定性臂 | 无对应产物 | `ERRATA_SCOPE_REDUCTION.md` 在产生 v8.27.0 数据**之前**已把本轮删减为「2 臂 × 900 档 × 150 回合/格」，故这些格**本轮本就没有**，不是丢件 |
| `preflight.json` | 未入册 | `RUN.md` §10 列了它，但盘上与 `manifest.json` 都没有此件（见 §36.5 第 3 条） |
| 引擎 / 数据集 / checkpoint（`bombdag.pt`、`round2.pt`） | 不复制、不入册 | §8.2 红线：既有资产不移动、不覆盖；其身份由 `ckpt_assertions_{start,final}.json` 固定（`all_pass=True`） |
| 既有 §1–§35 条目 | 不动 | 本轮**只追加**，未删/未改/未重排任何既有行（零漂移证据见 §36.3） |

### 36.3 与 §1–§35 的交集

`v8.27.0` 是**新增版本号**：登记前 `ASSET_SYNC.sha256` 共 1307 条非空行，其中 `v8.27.0` 命中 **0 次**、`v8t0_densereward` 命中 **0 次**；§33 / §34 / §35 分别登记 `v8.25.0` / `v8.24.0` / `v8.26.0`，与 `v8.27.0/` 的路径前缀无交集。本轮 71 条路径**两两互不相同**（重复 0），且**无一命中既有 1307 条**。

- **与 §1–§35 交集 = 0 条**；
- `ASSET_SYNC.sha256` 的写入方式是「读原文 + 末尾拼接」，**原 1307 条被完整保留**，新增 71 条在其后（1378 = 1307 + 71）；
- **零漂移证据**：追加后 `sha256sum -c` 为 **1377 OK / 1 FAILED（1378 条）**，减去本轮 71 条（**71/71 全 OK**）得既有部分 **1306 OK / 1 FAILED（1307 条）**——与 §35 记录的 **1306 OK / 1 FAILED** 逐数一致，即既有条目**一条未变、一条未坏**；
- 本轮**未**新增/删除/移动任何资产，`ASSET_SYNC.md` §1–§35 **一字未动**，证据目录**未被写入**（只读 + 只算 sha256）。

### 36.4 校验结论

**1. `manifest.json` 逐条比对（零不符）**：`evidence/DENSEREWARD/manifest.json` 的 **67** 条 `(sha256, bytes)` 与盘上逐件复算结果 **67/67 全等、0 处不符、0 缺件、0 多余**；件数差额恰为 `manifest.json` 自身（9,504 B），22,092,372 − 22,082,868 = 9,504 ⇒ 口径闭合。

**2. `sha256sum -c`**：**1377 OK / 1 FAILED（1378 条）**。唯一 FAILED 是 `Document/Update/Alpha/v4.2.0/journal.jsonl`（**唯一允许项**，与 §35 记录相同、未因本轮而变）；本轮新增 **71/71 全 OK**。

**3. 判定 `S-not-reward`**：D1（纯密集 `alive/1800`）**11/1,800 = 0.61 %**、D2（`alive/1800 + 0.5×survived`）**14/1,800 = 0.78 %**。P1（30 s > 24.17 % = 58/240，v8.20.0 U-900 60/40/20）与 P2（30 s > 60.83 % = 146/240，v8.26.0 G1）**两臂都不成立**；P3（20 s ≥ 85 %）按 ERRATA **本轮未测**，不得推断。

**4. vs v8.24 的 4.08 %（147/3,600）是显著更差**：单侧 Fisher（H1：密集臂 < D0）**D1 p = 1.11e-15**（−3.47 pp）、**D2 p = 8.62e-14**（−3.31 pp）；对 G1 的 60.83 % 为 p = 6.79e-143 / 1.07e-139。CP95：D1 [0.31 %, 1.09 %]、D2 [0.43 %, 1.30 %]，两臂区间大幅重叠 ⇒ **两臂之间不构成差异证据**。

**5. 一个重要否证**：v8.24 §33.3 把 30 s 崩溃解释为「稀疏奖励 ⇒ advantage ≈ 0 ⇒ 梯度变噪声」。**本轮否证该解释**——梯度密度最高的 D1（每一步都有非零奖励、无稀疏终局项）**反而最差（0.61 %）**，比稀疏的 D0（4.08 %）低 3.47 pp 且 p = 1.11e-15。⇒ 30 s 崩溃**不是奖励形状问题**，把奖励改密也救不活它；§33.3 的这条机制解释**不得再被引用**。

**6. 两臂都收敛到同一病态稳态**：存活局**首雷决策恒为 0**（D1 11 局、D2 14 局，**全部**）、存活局 **100 % 用满 3 颗**（`{3:11}` / `{3:14}`，均值 **3.000**）、`bomb_emits` **29.73 / 33.29 次/局**；全体回合用雷均值 **2.963 / 2.947**、打空（3 颗用完）**1,735/1,800** 与 **1,704/1,800**。对照 v8.21 的 2.962、v8.24 的 2.978 ⇒ **用雷总量没变**，「少用雷」完全没学会。

**7. 训练三件同向退化**（`train_curve.json`，每桶 150 回合）：`R` 前 50 → 后 50 为 **0.7500 → 0.7277（−0.0222）/ 0.7692 → 0.6968（−0.0724）**；`alive_frames` **1,349.9 → 1,310.0 / 1,342.6 → 1,248.3**（单调下降）；熵 **0.0241 → 0.0394 / 0.0241 → 0.0575**（单调上升）；训练内活满 **11/900**（D1）与 **19/900**（D2）。

**8. 门禁全过**：D3 保真门 **5/5 PASS**（`1_hello_role_shadow`、`2_debug_threads_1`、`3_prefix_2165_checkpoints`、`4_r24_18_constants`、`5_park_frame_8663`；wall **4.14 s**、`exit=0`、`all_pass=true`）；音频 **81 cfg 全 muted / 0 audible**（AGENTS.md §9.3 口径）；本轮引擎残留 **0**（`engine_processes_total=0`、`other_round_processes=0`）；`invalid` 行 **0/3,600**；`ckpt_assertions_{start,final}` 均 `all_pass=True`；训练梯度 6/6 `grad_finite=True`。

**9. `distinct` 与作废格**：池化 `distinct` D1 **1,716** / D2 **1,750**；逐格最小 **77**（D1 w18 s1）、次低 **127**（D2 w18 s0），**无一格 < 10**、**作废格 0** ⇒ P5 成立，`pooled_valid` 与 `pooled` 完全相同。

**10. mtime 口径（易误读）**：已发布件的 mtime 是**复制时刻**（18:28:00–18:30:31），**不是产生时刻**；`RUN.md` §9.1 的内部时间线为 18:08 preflight → 18:09 D3 门 → 18:10 训练 → 18:18–18:26 评测 → 18:27 verdict。两者**不矛盾**，但**不得用 mtime 反推执行顺序**。

### 36.5 与 manifest 不符项

以下逐条**只报告，不改动任何证据文件**。

**1. 「`summary` 因门未落盘而误判 `S-fork`」——已发布证据不支持该说法，须更正。** 盘上三份 `summary`（`summary_D1.json`、`summary_D2.json`、`.tmp_v8t0_densereward/summary.json`）**逐字节相同**（`1e2de30697143ba19f591cffdbf7de0f0c205d19b7ad9d64d0c69b729ae3efc3`，62,431 B，`created_at = 2026-11-22T18:28:31+0800`），三者的 `d3_gate` 都是 `present=true / passed=5 / total=5 / all_pass=true`，`verdict.stop_condition` 都是 **`S-not-reward`**、`S-fork.held=false`。且 `d3_gate.json` 落盘 **18:28:00**、`summary` 落盘 **18:28:32**——门比摘要**早 32 s**；`RUN.md` §9.1 的内部时间线同样是门 18:09 早于 verdict 18:27。⇒ **既不存在 `d3_gate.present=false` 的摘要，也不存在 `S-fork` 判定**。若该状态确曾出现，其产物已被 18:28:31/18:28:32 的重写**覆盖而未留痕**；因此它**不能**作为「自动判定必须在门证据落盘后生成」的实例引用——本轮的门 → 摘要顺序在盘上**已经是正确的**。

**2. 执行器身份与 `RUN.md` §9 不符**：`RUN.md` 记载 `tools/rl/v8t0_densereward.py` = **147,737 B / 2,927 行**（sha256 `d528031f…`）；盘上实测 **148,000 B / 2,931 行**，sha256 **`877f4581b010712a1380db4b6978db740b8cc333e1cb6916396c33be9b9b152d`**，差 **+263 B / +4 行**。§36.1.1 第 71 条以**盘上**为准入册；`RUN.md` 自身不自洽。

**3. `preflight.json` 缺件**：`RUN.md` §10 将其列入门禁产物，但 `v8.27.0` 目录与 `manifest.json` **都没有**此件；本轮 preflight 仅存 `RUN.md` §9.1 的叙述（`engines=0`、`assert-ckpt start all_pass=True`）。

**4. `publish.copied` 只覆盖 D1 臂**：`publish.json` `count=36`，36 条 = **D1 臂 29 件 + 7 件轮级产物**（`audio_audit`、`bomb_usage`、`ckpt_assertions_{start,final}`、`d3_gate`、`process_residue`、`train_curve`）；**D2 臂 29 件 + `manifest.json` / `publish.json` / `RUN.md` 共 32 件不在 `copied` 内**。完整清单以 `manifest.json`（67 件）为准。**工单所称「36 个 eval jsonl」实为 `publish.copied` 的 36 条之误**（36 = 24 个 `eval_*.jsonl` + 12 个 `.pt`）。

**5. 「24 格 exit 全 0」无机读证据**：`summary` 内**无 `exit` 字段**（`"exit"` 命中 **0** 次），目录内**无 `eval_exits.tsv`**；该结论仅见于 `RUN.md` §9.1 的叙述（「评测 24 格，12 路并行，全部 `exit=0`」）。作为门禁证据，它弱于 D3 门的 `exit=0`（后者有 `d3_gate.json` 落盘）。

**6. 两份「逐臂」摘要是同一份文件**：`summary_D1.json` 与 `summary_D2.json` **逐字节相同**（`1e2de306…`），且与工作目录 `summary.json` 相同。`RUN.md` §10 已自述「两次发布各一份，内容都是整轮跨臂结果」，故**非矛盾**；但文件名易被误读为两臂的独立摘要。

**7. 编码瑕疵（只报告不改）**：58 个文本件中 **3 件**含缺陷——`RUN.md` 第 44 行 `| 回�合 |` 含 **1 个 U+FFFD**；`summary_D1.json` / `summary_D2.json` 各含 **4 个 U+6402（「搂」）**，系「§」的 UTF-8→GBK 误码（如 `prereg 搂2 P4`）。12 个 `.pt` 为二进制件，不在此列。

**8. 工单清单与盘上不符（清单口径，非证据缺陷）**：`gate.json`、`fidelity_d3.json`、`commands.tsv`、`eval_exits.tsv`、`preflight_*`、`smoke.json` 盘上**均不存在**（`smoke` 实为 `train_smoke_D{1,2}.json`）；`eval_*.jsonl` 为 **24 件**（2 臂 × 4 窗 × 3 种子）而非 36 件；训练头 `.pt` 为 **12 件**（6 BC + 6 d900）而非「3+3」。以上**一律以盘上为准**，而盘上与 `manifest.json` **零不符**。

**9. `manifest.json` 的覆盖边界**：其 `root` 是 `evidence/DENSEREWARD`，**不含**顶层 `PREREGISTRATION_DENSEREWARD_001.md` 与 `ERRATA_SCOPE_REDUCTION.md`，也**不含**执行器；这两份顶层文件由 `summary.prereg` / `summary.errata` 声明的 sha256 交叉固定，已核对**逐位相同**（`ad8e6416…` / `277bda6a…`）。
## 37. v8.28.0 资产（2026-11-22）

### 37.1 登记范围与逐文件表

本轮登记 `Document/Update/Alpha/v8.28.0/` **整目录**（`Get-ChildItem -Recurse -File` 实测 **45 件 / 12,380,966 B = 11.8074 MB**）与**执行器** `tools/rl/v8u0_features.py`（147,355 B，`c4cde38f…`），合计 **46 条 / 12,528,321 B = 11.9479 MB**。**入册 46、未入册 0**（跳过项见 §37.2）。

> **与工单口述的 41 件 / 11.78 MB 不符，以盘上为准。** 该目录在本次登记过程中**仍在增长**：18:52 首次盘点为 **42 件 / 12,374,077 B**（11.8 MB），再等 120 s 后变为 **45 件 / 12,380,966 B**——新增 `commands.tsv`、`eval_exits.tsv`、`deviations.md`，且 `manifest.json` / `publish.json` / `RUN.md` 被**重写**（体积均变大）。此后连续 **120 s + 150 s 两段采样字节数与件数零变化**才钉住；最后一笔 `manifest.json` 落盘 **18:53:33**（其 `created_at = 2026-11-22T18:53:34+0800`）。

| 组 | 文件数 | 字节 | 目录实测 | manifest 覆盖 |
| --- | ---: | ---: | --- | --- |
| `Document/Update/Alpha/v8.28.0/**` | **45** | **12,380,966** | `Get-ChildItem -Recurse -File` 45 件 / 12,380,966 B（11.8074 MB）；`-Force` 亦 **45 件**，Hidden/System **0**、`.tmp_*`/`*.tmp`/`*.partial`/`~`/点开头 **0** | `evidence/FEATURES/manifest.json` 自述 **43** 件 / **12,371,313** B（`root` = `/mnt/d/…/evidence/FEATURES`，以 `manifest.json` 自身排除） |
| `tools/rl/v8u0_features.py` | **1** | **147,355** | — | manifest **不含执行器**（`root` 之外） |
| **合计** | **46** | **12,528,321** | — | — |

口径自洽（逐位可复算）：

| 目录 | 件数 / 字节 | `evidence/FEATURES/` | `manifest.json` | manifest 件数 / 字节 | 顶层 1 件 | 差额 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| v8.28.0 | 45 / 12,380,966 B | 44 / 12,377,422 B | 6,109 B | 43 / 12,371,313 B | `PREREGISTRATION_FEATURES_001.md` 3,544 B | 12,371,313 + 6,109 = **12,377,422**（+ 3,544 = **12,380,966**） |

> 目录结构只有两层：顶层 **1 件** `PREREGISTRATION_FEATURES_001.md`（3,544 B）+ `evidence/`（仅 `FEATURES/`，**44 件**）。与 v8.27.0 的**两臂**结构不同，本轮是**单臂（`F`）**，故没有 `<arm>/` 子目录。

文件类型构成（实测）：

| 目录 | `.jsonl` | `.json` | `.pt` | `.md` | `.tsv` | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v8.28.0 | 15 / 11,426,829 B | 19 / 740,378 B | 6 / 187,368 B | 3 / 24,501 B | 2 / 1,890 B | **45 / 12,380,966 B** |

> 字节重心是 **15 个 `.jsonl` 共 11,426,829 B（占该目录 92.2935 %）**：其中 **12 件评测原始行** `eval_F_d900_w{11,12,17,18}_seed{0,1,2}.jsonl` 共 **10,993,736 B（88.7955 %）**，**逐件实测恰 150 行**、**808,880–983,858 B/件**，合计 **1,800 行 = 3 种子 × 4 窗 × 150**；另有 **3 件训练逐回合** `seed{0,1,2}.jsonl` 共 **433,093 B**，**逐件实测恰 300 行**，合计 **900 行**；两类合计 **2,700 行**，**15 件全部以 LF 结尾**。**6 个 `.pt` 共 187,368 B** = 3 个 BC 热启动头（31,243 B 各）+ 3 个 d900 训练后头（31,213 B 各）。其余 **19 个 `.json`（740,378 B）** 中体量最大的是 `summary.json` 71,322 B、`publish.json` 9,743 B、`train_curve.json` 8,450 B、`train_smoke.json` 6,402 B、`manifest.json` 6,109 B、`ckpt_assertions_final.json` 5,824 B（`train_seed{0,1,2}.json` 合计 **585,714 B**），最小的是 `process_residue.json` 544 B；**2 个 `.tsv`（1,890 B）** 为 `commands.tsv` 1,736 B 与 `eval_exits.tsv` 154 B；**3 个 `.md`（24,501 B）** 为 `RUN.md` 17,433 B、`deviations.md` 3,524 B 与顶层预注册稿 3,544 B。
>
> 发布时序（**已发布件的 mtime 是复制时刻，不是产生时刻**）：`PREREGISTRATION_FEATURES_001.md` 18:32:17 → `preflight` 18:37:50 → `ckpt_assertions_start` 18:38:25 → **`d3_gate` 18:38:31** → 3 个 BC 头 18:38:56 → `pretrain_seed{0,1,2}` 18:39:09–18:39:10 → `train_smoke` 18:39:23 → 训练产物 `seed{0,1,2}.jsonl` 与 3 个 d900 头 18:44:43–18:45:23 → 评测 12 格 18:49:43–18:51:04 → `audio_audit`/`ckpt_assertions_final`/`process_residue` 18:51:09 → `summary`/`train_curve`/`bomb_usage`/`first_bomb_decision` 18:51:31 → `RUN.md` 18:53:16 → `deviations.md` 18:53:24 → `commands.tsv`/`eval_exits.tsv` 18:53:31 → `manifest.json`/`publish.json` 18:53:33。

#### 37.1.1 逐文件表（46 条：45 件发布目录 + 1 件执行器）

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/audio_audit.json` | 12,130 | `67d6ddad800252ed15e313d53e6310fdfa496ce7880ec6b20c2be1beebfcf420` |
| 2 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/bomb_head_seed0_bc.pt` | 31,243 | `02b3137172b95d3066b74d4bc09a0b4ada76362bee104a73bab79368dca93976` |
| 3 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/bomb_head_seed0_d900.pt` | 31,213 | `b7c7036183464e47660f71dab65e6cf5a2aab220b07fca3132e0822498149bc5` |
| 4 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/bomb_head_seed1_bc.pt` | 31,243 | `5f1e3a610fd81fb76583c888c378abb491cb4d4fe6997cce347a8b9900dc9905` |
| 5 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/bomb_head_seed1_d900.pt` | 31,213 | `ac31bca61fc6c2ff7b3059e54f77fb2d236e96215056ac7c95e7d62ee253aac6` |
| 6 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/bomb_head_seed2_bc.pt` | 31,243 | `cfa0cad34e9d3eb8b52905ca298fa284e0eec61a6b7a5cacbf1f614e524ccab7` |
| 7 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/bomb_head_seed2_d900.pt` | 31,213 | `aedc56c060589093b36692e37b6b278e26ad8ffcca7cec612acd648a5aeb469f` |
| 8 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/bomb_usage.json` | 3,591 | `2734e74f9468beb6854db4b1a6bf3ccd3c3f5d942710446185483dabab715837` |
| 9 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/ckpt_assertions_final.json` | 5,824 | `ea88ad8b32f70d357320bb2d7ecae2bcbe8b512b341901cf64bac3cc5a245c9e` |
| 10 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/ckpt_assertions_start.json` | 5,638 | `efe75e325462a0f5129a814b96338a787cb21fe32749af97b7d9cf200759718f` |
| 11 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/commands.tsv` | 1,736 | `81abf2359c1485b8c8eac5e24711d0dd18742e3ed4631b9807df0eb9c35621b6` |
| 12 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/d3_gate.json` | 5,086 | `b102f0a910f3fe8edf51b3c4eaa1456b422ebf4dddcdf1828a6a4fb88e58fa66` |
| 13 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/deviations.md` | 3,524 | `4e9d217ed67b4e47787fa35c45804de9a84d40538616708c0f767e728801a2fd` |
| 14 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/eval_exits.tsv` | 154 | `c229488033daa0baf1d1a272409bf9a450d47e4cf2e7ef15cc62474eb9a4e8f8` |
| 15 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/eval_F_d900_w11_seed0.jsonl` | 950,010 | `aa6ba99d47f2cec347286bc0c56ff9ed97f85b0442e87488028aaba898b75578` |
| 16 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/eval_F_d900_w11_seed1.jsonl` | 983,858 | `19102ac74ada1c4e22160ea530a4bc9b24fc06ec78ccf31f1b090e529c089626` |
| 17 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/eval_F_d900_w11_seed2.jsonl` | 919,446 | `91d2e916de48971dd2534308f19394b9e4ca3cbaac463c7fe270e8f26f081541` |
| 18 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/eval_F_d900_w12_seed0.jsonl` | 894,958 | `c7c3973b6f03b3402f53b4b3befa32da3283c02bbfa8feff632915269c75f42e` |
| 19 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/eval_F_d900_w12_seed1.jsonl` | 883,462 | `546a34a81bf66d38ba60d252ddd1adcd6f669e31edd5fdcc51c1cd95d4e19a0d` |
| 20 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/eval_F_d900_w12_seed2.jsonl` | 894,927 | `06167356bb1e46f1dc2ed806e62cf015b17dd9b64783418163d7431cd6448930` |
| 21 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/eval_F_d900_w17_seed0.jsonl` | 942,534 | `45c0d5ab0a4d8abed6c94da889452b20466d7dbb1f05532a560bc3259943166a` |
| 22 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/eval_F_d900_w17_seed1.jsonl` | 808,880 | `3ad1b2c6dc862e93c78cbddd8a8ac346075fdc9bdf145c64b811010cd8eb4382` |
| 23 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/eval_F_d900_w17_seed2.jsonl` | 890,383 | `6a61664af69bade3c29f5cff7f9e2b42dcd06490e7ac5cd7501eeb20c460e9e9` |
| 24 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/eval_F_d900_w18_seed0.jsonl` | 972,965 | `615e90741e7e4bc65d1625b551c2644150b08a5ce2272e114a1380a5039ef273` |
| 25 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/eval_F_d900_w18_seed1.jsonl` | 928,049 | `255a0f97e9675dc45a7da99d7be0c826368f00e408d18679b01eae5042bfb0cf` |
| 26 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/eval_F_d900_w18_seed2.jsonl` | 924,264 | `60a193b2bac381c786e662c89b0340d32c50a7ccadf5a033d3335333a3676f0d` |
| 27 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/first_bomb_decision.json` | 3,540 | `98c26da3c589f4eac18276cd506c971db4908c9ea0f6e7a4b140b1cfa544cf98` |
| 28 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/manifest.json` | 6,109 | `fbd62a5b55fef3d0e8bb7a3562e7f9bbeab5d62b95d87c78899d962fb9cba551` |
| 29 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/preflight.json` | 670 | `44e0f9295d349ad15a24086629d5c28eb1fbf1dd106e798058776998d9a52c47` |
| 30 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/pretrain_seed0.json` | 5,209 | `361f298d680a664aaf09752de70421d2879541eb72968bd883b18bbaab956dae` |
| 31 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/pretrain_seed1.json` | 5,202 | `14fe62b094978767a697263c1ead2ee888f22b32f5346d0d034c3c191c477e3c` |
| 32 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/pretrain_seed2.json` | 5,204 | `9c1394556245f9322538e3df31f6fd11a724a6144768ccebdfe685df7424968a` |
| 33 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/process_residue.json` | 544 | `b01a03efe2bd5ffb509afc43c200d4f904f990a67e91e5e9052912f0293a1e14` |
| 34 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/publish.json` | 9,743 | `5c6bafda7769da59ad066aec14c387efeec5ea68ae7707cb8c011ee27c3d27b5` |
| 35 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/RUN.md` | 17,433 | `2291ad7947c64a60d86e8957cd407009b0ff08ba2f5bf22a443e26cac9e1c3be` |
| 36 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/seed0.jsonl` | 154,611 | `5da6ec49f3185274df42ee5a89651fb7cf8f68e9e5b26222c0b82fb3a560ccd5` |
| 37 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/seed1.jsonl` | 134,985 | `c637b0efe12574e06684178d87fdc4c7247a05091db9c17025099ac8cf0e657a` |
| 38 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/seed2.jsonl` | 143,497 | `b860fdb4311835b4768cf1be137fdecbc1b87d3b57bbf131fbcb4d49c5ed0528` |
| 39 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/summary.json` | 71,322 | `480fdd0df15df76d650eb4e4825b8c7e1c4f0e06f0940b29e521d5d439e43a71` |
| 40 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/train_curve.json` | 8,450 | `078deff4f9474323af33e961bc48433ba5a8e0a3e0b35a993cb7c1d361210ee4` |
| 41 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/train_seed0.json` | 219,201 | `c482712e2a41dd7c8709131676a2e94dc92c26cd14e302c61888fbc30ecd76b0` |
| 42 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/train_seed1.json` | 173,381 | `860349503ab3229976192d9ff97745926de71638936ae6027036b8ad965f9a0e` |
| 43 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/train_seed2.json` | 193,132 | `eb608b487e05329401089835cccbefc50a0f5b02ea24408143a8860f0f85e74c` |
| 44 | `Document/Update/Alpha/v8.28.0/evidence/FEATURES/train_smoke.json` | 6,402 | `9af42431fa2af5c9f0604a0a113b9eac4d3a99a2dec3a9e4f72e27eee78057e5` |
| 45 | `Document/Update/Alpha/v8.28.0/PREREGISTRATION_FEATURES_001.md` | 3,544 | `926779db365ef04487497be7de9b44cae178a09e01621070e50dda928d97286a` |
| 46 | `tools/rl/v8u0_features.py` | 147,355 | `c4cde38f954f20433932c3dade77afb5882c0a42bb0c94070c8bee380a848aac` |

> 第 45 条的 sha256 与 `ckpt_assertions_{start,final}.json` 的 `prereg.sha256` **逐位相同**（`926779db…7286a`）⇒ 预注册稿自冻结后**未被改动**。第 46 条**不**等于 `ckpt_assertions` 的 `self.sha256`（`79f9f713…`，见 §37.5 第 1 条）。

### 37.2 跳过说明

本目录**无跳过件**：45 件发布目录文件 + 1 件执行器 = 46 条**全部入册**，未入册 **0**。以下物件**不属登记范围**，其存在/缺失都不构成缺件：

| 物件 | 处理 | 理由 |
| --- | --- | --- |
| `.tmp_v8u0_features/`（`ckpt/`、`data/`、`eval/`、`train/`、`runs/`、`logs/`、`work/`、`d3_gate.json`、`fidelity_d3.json`） | 不入册，仅引用路径 | §8.1：本机原始运行目录，不是发布件。其文件由 `manifest.json` 的 `root`（`evidence/FEATURES`）固定，且 `ckpt_assertions` 已用 sha256 钉住引擎/候选头/行为策略，不入册不损失身份信息 |
| 引擎 `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`（`691bfd77…`）、候选头 `bombdag.pt`（`9c6d5d4a…` / `36a51090…` / `885dde70…`）、行为策略 `round2.pt`（`4abee069…` / `a93746bb…` / `c69960fc…`） | 不复制、不入册 | §8.2 红线：既有资产不移动、不覆盖；其身份由 `ckpt_assertions_{start,final}.json` 固定（两者均 `all_pass=True`，且引擎 sha 与本文件 §8.1 记录的 `rl_v3 = 691bfd77…` **一致**） |
| 20 s / 600 档、40 s / 1,200 档、密集奖励臂、D1/D2 双臂 | 无对应产物 | 预注册 §1 把本轮定为**单因子单臂 `F` + 只测 900 档**，故这些格**本轮本就没有**，不是丢件（`summary.notes` 第 1 条自述「只有 900 档」） |
| `ERRATA_SCOPE_REDUCTION.md` 一类范围修正稿 | 本轮**没有**此件 | v8.27.0 有该稿；v8.28.0 不需要——预注册 §1 已把范围收在单臂 900 档，`deviations.md` 只记执行器偏离 |
| 既有 §1–§36 条目 | 不动 | 本轮**只追加**，未删/未改/未重排任何既有行（零漂移证据见 §37.3） |

### 37.3 与 §1–§36 的交集

`v8.28.0` 是**新增版本号**：登记前 `ASSET_SYNC.sha256` 共 **1378** 条非空行，其中 `v8.28.0` 命中 **0 次**、`v8u0_features` 命中 **0 次**、`v8u0` 命中 **0 次**、`evidence/FEATURES` 命中 **0 次**；§33 / §34 / §35 / §36 分别登记 `v8.25.0` / `v8.24.0` / `v8.26.0` / `v8.27.0`，与 `v8.28.0/` 的路径前缀无交集。本轮 46 条路径**两两互不相同**（重复 0），且**无一命中既有 1378 条**。

- **与 §1–§36 交集 = 0 条**；
- `ASSET_SYNC.sha256` 的写入方式是「读原文 + 末尾拼接」，**原 1378 条被完整保留**（写入前 `chars=186,309`、`lines=1378`、`endsWithLF=true`、`CRLF=0`、无 BOM；写入后前缀 `StartsWith(原文)` = **True**），新增 46 条在其后（**1424 = 1378 + 46**，`chars=192,450`、`CRLF=0`、首字节 `56,102,99` 即 `Vfc` 无 BOM）；
- **零漂移证据**：追加后 `sha256sum -c` 为 **1423 OK / 1 FAILED（1424 条）**，减去本轮 46 条（**46/46 全 OK**）得既有部分 **1377 OK / 1 FAILED（1378 条）**——与 §36 记录的 **1377 OK / 1 FAILED（1378 条）** 逐数一致，即既有条目**一条未变、一条未坏**；
- 本轮**未**新增/删除/移动任何资产，`ASSET_SYNC.md` §1–§36 **一字未动**（追加前 `chars=567,940`、`lines=4413`，追加后前缀 `StartsWith(原文)` = **True**），证据目录**未被写入**（只读盘点 + 只算 sha256；`manifest.json` 的 43 条 sha256 与盘上复算**全等**，可反证证据文件未被我方触碰）。

### 37.4 校验结论

**1. `manifest.json` 逐条比对（零不符）**：`evidence/FEATURES/manifest.json` 的 **43** 条 `(sha256, bytes)` 与盘上逐件复算结果 **43/43 全等、0 处不符、0 缺件、0 多余**；件数差额恰为 `manifest.json` 自身（6,109 B），12,377,422 − 12,371,313 = **6,109** ⇒ 口径闭合。其 `file_count=43`、`total_bytes=12,371,313` 与自述**自洽**。覆盖面与 v8.27 一样是「`root` 内除自身」，即**不含**顶层 `PREREGISTRATION_FEATURES_001.md`、**不含**执行器。

**2. `sha256sum -c`**：**1423 OK / 1 FAILED（1424 条）**。唯一 FAILED 是 `Document/Update/Alpha/v4.2.0/journal.jsonl`（**唯一允许项**，与 §35/§36 记录相同、未因本轮而变）；本轮新增 **46/46 全 OK**。退出码 1 仅由该既有项引起，`WARNING: 1 computed checksum did NOT match`。

**3. 判定 `S-features-partial`（与工单一致）**：`summary.verdict.stop_condition = "S-features-partial"`，`stop_text = "P2 holds but P1 does not: better than the sparse/dense 106-dim arms (4.08 %) but still far below the hand-written rule"`；四个判定位 **`S-features-work.held=false`、`S-features-partial.held=true`、`S-representation-hopeless.held=false`、`S-fork.held=false`**。`summary.complete=true`、`invalid_rows=0`、`arm="F"`、`rows=1800=expected`。

**4. 单因子 = 放雷头输入维度 106 → 109（与工单一致）**：`protocol.bomb_head = "Linear(109,64)->Tanh->Linear(64,2)"`，`bomb_head_input = "109 = 106 (frozen obs) + 3 explicit: ['bombs_left/3', 'decisions_left/budget', 'clip(nearest_all_px,0,200)/200']"`；**候选头仍只吃 106 维**（`candidate_head = v8.6.0 bombdag.pt Policy(106,64,36), frozen, argmax`）；奖励**回到 v8.24 的稀疏式**（`R = survived(900) + 0.5 × alive_frames/1800`，`REWARD_BY_ARM = {"F": (1.0, 0.5)}`），以与已发布的 4.08 % 可比。`f3` 的测距与 60/40/20 规则**同源**（`v800_bomb.nearest_bullet_distances`）。

**5. P1 不成立（与工单一致）**：**150/1,800 = 8.3333 %** vs 阈值 **24.1667 %（58/240）**，**−15.8333 pp**；正向单侧 Fisher `p = 0.9999999999965`（不显著），反向 `p = 1.1074618606825624e-11`；`rate_gate=false`、`passed=false`。CP95 **[7.0975 %, 9.7069 %]**，上界仍远低于 24.17 %。

**6. P2 成立（与工单一致）**：**8.3333 % > 4.0833 %（147/3,600，v8.24 同框架 D0）**，**+4.25 pp**；正向单侧 Fisher **`p = 2.4733605799136241e-10`**，优势比 `odds_ratio = 2.1354`（Haldane 2.1350）；`rate_gate=true`、`passed=true`。**速率比 8.3333 / 4.0833 = 2.0408 ≈ 2.0 倍**（工单所称「2.0 倍」按速率比计）；若按 Fisher 优势比则 **2.14 倍**。两者都远超 1，P2 的「确实更好」有统计支撑。

**7. P3 不成立 —— 本轮最重要的发现（与工单一致）**：`first_bomb_decision` **中位数 0.0、均值 0.0**、`q1=q3=min=max=0.0`、`stdev=0.0`；**1,800/1,800 局全部在 decision 0 放雷**（`at_decision_zero=1800`、`fraction=1.0`），`episodes_bombed=1800`、**`episodes_never_bombed=0`**；8 个桶中只有 `"0"` 桶 = **1,800**（100 %），`1-5`/`6-10`/`11-20`/`21-50`/`51-100`/`101-300`/`>300` **全为 0**；逐窗 w11/w12/w17/w18 **各 450/450 也都在 0**（`at_decision_zero=450` ×4）；`median_gt_threshold=false`、`passed=false`（阈值 10）。⇒ **加了 3 个显式特征之后，首雷仍然恒为 0。**

> **这一条是「学习放雷」第 10 次实验的决定性证据。** 把本轮计入，本线已用 **10 次实验、4 类途径**——**反事实标签 5 次**、**终点回报 2 档**（600 / 900 决策）、**密集奖励 2 次**、**显式特征 1 次（本轮）**——反复检验同一问题，结果都是 **首雷决策恒为 0**。计数以 `PREREGISTRATION_FEATURES_001.md` §0 的自述「本线已用 **9 次实验**否定『学习放雷』的三种做法」为基数，加本轮 = **10**。
>
> 决定性在于：前三类途径动的是**标签质量**与**奖励形状**，本轮动的是**输入信息本身**——把「还剩几颗雷 / 还剩多少决策 / 距最近弹幕多远」三个量**直接喂进放雷头**（且测距与手写规则同源），首雷位置**仍然逐位相同（全 0）**。⇒ 「首雷恒为 0」**不是**奖励形状问题（v8.27 已否证）、**不是**标签质量问题、**也不是**输入信息缺失问题。**这三类解释都已被实验排除**；至少在本框架（`Linear(·,64)→Tanh→Linear(64,2)` 2 动作头 + argmax 冻结候选头 + 900 决策回合）内，「学习放雷」**不成立**。
>
> **机制补充（`RUN.md` §6.1，不改变结论）**：被推迟的只有**第 3 颗**雷——首次在「还剩 1 颗」时放雷的 decision 中位数 s0/s1/s2 = **430.5 / 474 / 250**，而**第 1、2 颗仍在 decision 0 与 1**；且 v8.27 的 **106 维**头已经有该行为（173 / 253），故「留一颗」**不是**显式特征带来的新能力。另：头几乎**一直输出 bomb**（`bomb_emits` 均值 23.35 vs 仅兑现 2.98），引擎只在还有雷时兑现——这是「用雷恒 3.00」的机械原因，不是学会了时机。

**8. P4 用雷（与工单一致）**：**存活局 150 局 `bombs_used_mean = 3.0000`**，`counts = {3: 150}`（**100 % 用满 3 颗**）、`bombs_used_total = 450`、**`bomb_emits_mean = 13.3867`**（total 2,008）；全体 1,800 局均值 **2.9794**、`{3: 1763, 2: 37}`、`bombs_used_total 5,363`、`bomb_emits_total 42,038`（均值 23.354）、`episodes_never_bombed 0`、`bomb_prob_mean 0.0369`、`last_bomb_decision` 中位数 **555.0**、`alive_frames` 均值 **1,383.9**。对照 v8.21 的 30 s 用雷 **2.962** ⇒ **用雷总量没变**，「少用雷」依旧没学会。存活局首雷中位数同为 **0.0**（与 P3 交叉一致）。

**9. P5 成立，但**工单口述的「distinct 最小 98」与盘上不符**：**池化 `distinct = 1,559`**（= 逐窗和）、`replays_per_outcome = 1.155`；逐格 `distinct` 为 w11 **142 / 98 / 150**、w12 **123 / 117 / 150**、w17 **144 / 120 / 150**、w18 **147 / 68 / 150**（s0/s1/s2），`per_cell_min = 68`、`per_cell_max = 150`、`per_cell_mean = 129.9167`。**98 是 `w11_s1` 的值，真正的最小值是 `w18_s1` 的 68**（`RUN.md` §5.4 与 `RUN.md` §5.1 都记 68，仅工单写成 98）。全部 ≥ 10（`cells_below_min=0`、`void_cells=[]`）⇒ **`passed=true`、作废格 0**，`pooled_valid` 与 `pooled` 完全相同。

**10. 训练曲线（与工单一致）**：`R` 前 50 → 后 50：**s0 0.533398 → 0.540361（+0.006963，上升）**、**s1 0.544278 → 0.553339（+0.009061，上升）**、**s2 0.373669 → 0.343502（−0.030167，下降）**；池化 **0.476798 → 0.473157（−0.003641）**，`rises=false`。训练内活满 **s0 44 / s1 43 / s2 2 = 89/900 = 9.889 %**（v8.24 为 **24/900 = 2.67 %**）。熵**单调上升 0.013047 → 0.026743**（≈2.05 倍）；存活率 9.33 %–10.67 % 无趋势；用雷从第一桶就 2.88 → 2.99（BC 热启动即「能放就放」，训练没拉回来）。

**11. 门时序正确（`S-fork.held=false` 有据）**：`d3_gate.created_at = 2026-11-22T18:38:30+0800` **早于** `summary.created_at = 2026-11-22T18:51:30+0800`（早 **13 分 00 秒**）；`summary.d3_gate` 显式记 **`present=true`、`written_before_summary=true`、`passed=5`、`total=5`、`all_pass=true`**，且其 `sha256 = b102f0a9…` 与盘上 `d3_gate.json` 复算**逐位相同**。门本体 `exit=0`、`wall_s=5.187`、`verdict="PASS: §2.1-§2.5 hold on this binary and data dir"`、`checks` 五条**全部 `passed=true`**（`1_hello_role_shadow`、`2_debug_threads_1`、`3_prefix_2165_checkpoints`、`4_r24_18_constants`、`5_park_frame_8663`）。⇒ 本轮**不存在**「门晚于摘要」的情形。

**12. 其余门禁全过**：音频 **41 个 `th08.cfg` 全 `muted` / 0 `audible`**（AGENTS.md §9.3 口径，`all_pass=true`）；本轮引擎残留 **0**（`engine_processes_total=0`、`other_round_processes=0`、`pgrep_returncode=1`）；`preflight` 记 `engines=0`、`nproc=20`、`waited_s=0`、`loadavg 0.11/1.82/4.60`；`invalid_rows = 0 / 1,800`；`ckpt_assertions_{start,final}` 均 **`all_pass=True`**，引擎 `691bfd77…` 与其 `expected` **相等**（与本文件 §8.1 记录的 `rl_v3` 身份一致）；`eval_exits.tsv` **12 格 `exit=0`、各 150 行**；`commands.tsv` **27 条命令 `exit` 全 0**。

**13. mtime 口径（易误读）**：已发布件的 mtime 是**复制时刻**（18:32:17–18:53:33），**不是产生时刻**；`RUN.md` §10.1 的内部时间线为 18:38 preflight/D3 门 → 18:39 BC → 18:40 smoke → 18:41–18:47 训练 → 18:47–18:53 评测 → 18:51 verdict/summary → 18:52 audit/publish。两者**不矛盾**（门 18:38:30 早于 verdict 18:51:30 在两条时间线上都成立），但**不得用 mtime 反推执行顺序**。

### 37.5 与 manifest 不符项

以下逐条**只报告，不改动任何证据文件**。先给结论：**盘上与 `manifest.json` 零不符**（§37.4 第 1 条，43/43 全等）；下列「不符」全部是**证据文件自述之间**或**工单口述与盘上**的不符，不是盘上与 manifest 的不符。

**1. 执行器身份三方互不相符，且盘上文件**不是**产生本轮证据的那一份（本轮最需要下游注意的一条）。** 同一文件 `tools/rl/v8u0_features.py` 有**三个互不相等的身份**：

| 来源 | 字节 | 行数 | sha256 | 是否在盘上 |
| --- | ---: | ---: | --- | --- |
| **盘上实测** | **147,355** | **2,859** | `c4cde38f954f20433932c3dade77afb5882c0a42bb0c94070c8bee380a848aac` | ✅ 是 |
| `RUN.md` §10 | 140,559 | 2,847 | 「见 `manifest.json`」——而 **manifest 的 `root` 是 `evidence/FEATURES`，根本不覆盖执行器** | ❌ 无 |
| `ckpt_assertions_{start,final}.json` 的 `self.sha256` | —（未记字节） | — | `79f9f713266439f51b539342056b9eaceeaedcce7fb313a148f5a76904e056d8` | ❌ **盘上不存在** |

三条互不相等，且**盘上文件与 `assert` 记录不符**（`MATCH = False`）。旁证：执行器 mtime = **18:51:29**，而 `ckpt_assertions_final.created_at = 18:51:10`——**盘上文件在 final 断言之后 19 秒被改过**；`RUN.md` 落盘（18:53:16）虽晚于该 mtime，却仍写 140,559 B，与 147,355 B 也不符。⇒ **产生证据的执行器字节序列已无法从盘上恢复**（唯一记录 `79f9f713…` 在盘上没有对应文件，manifest 也不含执行器）。本轮已按工单口径以**盘上**为准入册第 46 条（§37.1.1），但**「用盘上执行器复现本轮数字」这一前提不成立**，须由下游显式确认后再生效。对照：**预注册稿未漂移**——第 45 条的 sha256 与 `ckpt_assertions.prereg.sha256` **逐位相同**（`926779db…`），漂移只发生在执行器上。

**2. `publish.copied` 只有 41 条，盘上 `FEATURES/` 有 44 件**：`publish.json` `count=41`、`copied` 41 条，其 `(sha256, bytes)` 与盘上**逐条全等（0 不符）**；未进入 `copied` 的 3 件为 **`manifest.json`、`publish.json`、`RUN.md`**——这三件是发布动作**之后**才生成的（`RUN.md` 18:53:16、`manifest.json`/`publish.json` 18:53:33），逻辑上不可能被「复制」清单收录，**非丢件**。完整清单以 `manifest.json`（43 件）+ 其自身为准。

**3. `RUN.md` §11 产物表漏列 3 件盘上文件**：`deviations.md`、`commands.tsv`、`eval_exits.tsv` 在盘上、在 `manifest.json` 内，但 `RUN.md` 全文对这 3 个文件名的命中数均为 **0**（§11 表格未列，§10.1 时间线也未提）。属**文档不完整**，非证据缺陷。

**4. `first_bomb_decision.json` 的 `in_training` 块字面读作「训练 900 局从未放雷」，与 `train_curve.json` 直接矛盾。** 前者记 `episodes=900`、`episodes_bombed=0`、**`episodes_never_bombed=900`**、`median_with_never_bombed_at_budget=900.0`、`degenerate_low_bomb_fraction=true`、`n=0`；后者同批训练回合的 `bombs_used_mean` 为 **2.88 / 2.92 / 2.96 / 3.00 / 2.98 / 3.00**（六个桶，seed0）等——**训练局显然放了雷**。真因见 `RUN.md` §11.1 与 `deviations.md`「Known gaps」第 1 条自述：**训练侧逐回合 jsonl 没有 `first_bomb_decision` 字段**，故 `n=0`、派生标志**全部为空转**。⇒ 该块的 `episodes_never_bombed=900` **不得**被引用为「训练期不放雷」的证据；P3 只能按定义取**评测侧 1,800 局**（评测侧 1,800/1,800 全部放雷，与「从未放雷」正好相反）。

**5. 工单口述与盘上不符（清单口径，非证据缺陷）——一律以盘上为准**：
- **文件数**：工单称 **41 件 / 11.78 MB**；盘上 **45 件 / 12,380,966 B（11.8074 MB）**。差额来自登记期间目录**仍在增长**（42 → 45；见 §37.1 首段）。
- **`eval_*.jsonl` 件数**：工单称 **18 个**；盘上 **12 个**（`eval_F_d900_w{11,12,17,18}_seed{0,1,2}.jsonl`，3 种子 × 4 窗）。**12 件 × 150 行 = 1,800 行**与 `summary` 的 `rows=1800` 一致。
- **`.pt` 件数**：工单称 3 个 `bomb_head_seed*_d900.pt`；盘上 **6 个**（3 个 `_bc` 热启动头 + 3 个 `_d900`）。
- **执行器体积**：工单称「约 140 KB」；盘上 **147,355 B（约 144 KB）**，且身份见第 1 条。
- **distinct 最小值**：工单称 **98**；盘上 `per_cell_min = 68`（`w18_s1`）。98 是 `w11_s1` 的值（详见 §37.4 第 9 条）。
- **途径数**：工单称「**5 种途径**」但只列出 4 项（标签 / 终点回报 / 密集奖励 / 显式特征）；按 `PREREGISTRATION_FEATURES_001.md` §0 的枚举，本线实为 **4 类途径 / 10 次实验**（标签 5 + 终点回报 2 + 密集奖励 2 + 显式特征 1）。§37.4 第 7 条按 **4 类** 写入。

**6. `manifest.json` 的覆盖边界**：其 `root` 是 `/mnt/d/WorkShop/Touhou_RL/Document/Update/Alpha/v8.28.0/evidence/FEATURES`（**绝对 WSL 路径**，与 v8.27.0 的写法**一致**，非本轮变化），**不含**顶层 `PREREGISTRATION_FEATURES_001.md`，**不含**执行器，**也不含自身**（`file_count=43` 而目录 45 件）。⇒ 「manifest 是整目录清单」**不成立**；它只是「`FEATURES/` 内除自身」的 43 件。顶层预注册稿由 `ckpt_assertions.prereg.sha256` 独立固定（且与盘上逐位相同），执行器则**没有任何盘上可校验的固定**（见第 1 条）。

**7. 本轮的记录优点（与 v8.27.0 的对照，供下游取舍）**：v8.27.0 的 `preflight.json`、`eval_exits.tsv`、`commands.tsv` **缺件或无机读证据**（§36.5 第 3、5 条）；本轮这 3 件**全部落盘且入册**（`preflight.json` 670 B、`eval_exits.tsv` 12 格 `exit=0`、`commands.tsv` 27 条 `exit=0`），并且 `verdict` 新增了 **D3 门序断言**（`written_before_summary`）与 **P3 显式判据**，把 v8.27 踩过的「摘要早于门 ⇒ 误判 `S-fork`」变成**机检项**而非人读项。这两处是本轮相对 v8.27 的实质改进，`S-fork.held=false` 因此有**自动**依据。
## 38. v8.31.0 资产（2026-11-26）

### 38.1 登记范围与逐文件表

本轮登记 **30 条 / 17,316,326 B = 16.5126 MiB**：v8.31.0 目录 **6 件**（含本轮回执与 A 节任务卡）+ 本单代码 **2 件** + 验收脚本与日志 **6 件** + 三个运行根的结果/静音/逐决策 **18 件**（6 件 × 3 轮）。**入册 30、未入册 0**（跳过项见 §38.2）。

> 口径：v8.31.0 目录在本单开始时已有 `README.md`（6,695 B）与 `TRAINING_HOST_REQUEST_01.md`（10,385 B）两件（研发端提交 `1180761` 带来的），本轮**只追加**两份新文档，未改动前者；`README.md`/`TRAINING_HOST_REQUEST_01.md` 一并入册以便下游复算同版身份。

| 组 | 文件数 | 字节 |
| --- | ---: | ---: |
| `Document/Update/Alpha/v8.31.0/**` | 4 | 87,603 |
| `tools/rl/v8_31_slice.py` + `tests/test_v8_31_slice.py` | 2 | 261,776 |
| `.tmp_v831_slice_01_probe/`（3 个验收脚本 + 3 份日志） | 6 | 1,706,853 |
| `.tmp_v831_slice_01_lV7zfs9S/**`（第 1 轮，技术无效） | 6 | 42,548 |
| `.tmp_v831_slice_01_eK6pxCuY/**`（第 2 轮，像素通过 / 教师超时） | 6 | 5,763,824 |
| `.tmp_v831_slice_01_YmVD4nBX/**`（**第 3 轮，两条轨迹均通过**） | 6 | 9,453,722 |
| **合计** | **30** | **17,316,326** |

> `v8.31.0/` 的 4 件 = `README.md` 6,695 + `TRAINING_HOST_REQUEST_01.md` 10,385 + `TASK_CARD_SLICE_A.md` 35,046 + `TRAINING_HOST_REPLY_01.md` 35,393。**未入册**：两个空运行目录 `.tmp_v831_slice_01_l6OqKAS4/`、`.tmp_v831_slice_01_NANazung/`（各 4,096 B，`mktemp -d` 建目录后在 pytest 行即中止，**未启动引擎、未写任何文件**，见 §38.4 第 3 条）；三份运行根的 `game/` 私有游戏目录（各 474 MB，`.tmp_*` 运行件，身份由 `mute.json` 的 `copy_before_start`/`copy_after_exit` sha256 固定）。

#### 38.1.1 逐文件表（30 条）

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `Document/Update/Alpha/v8.31.0/README.md` | 6,695 | `6512fe5f4d452226726b7d2ed9ea1483213c45190361a41b9b511abc13c47227` |
| 2 | `Document/Update/Alpha/v8.31.0/TRAINING_HOST_REQUEST_01.md` | 10,385 | `74da5f38246e1657e5db817bb6b1bd28eb96e81c6325e511806d6470f3c8a010` |
| 3 | `Document/Update/Alpha/v8.31.0/TASK_CARD_SLICE_A.md` | 35,046 | `c3d2b470b57654b2991102f2516a68847abd688c0fe7d622f1246d7eb2caf5db` |
| 4 | `Document/Update/Alpha/v8.31.0/TRAINING_HOST_REPLY_01.md` | 35,477 | `3c1d2a963d6b9e774f8fcc8ad8fc0ff6873d71ae4438c496edcd8ccc924c2435` |
| 5 | `tools/rl/v8_31_slice.py` | 173,807 | `19888a274c3508c8fabc21452aa5959b9916b84b75ba39ce287d0b48026430d0` |
| 6 | `tests/test_v8_31_slice.py` | 87,969 | `2c51143f8a22781c9af2cf8cb9ac1a0cb9a5e55066c9b5a7d08cbb43b3a630e3` |
| 7 | `.tmp_v831_slice_01_probe/v831_acceptance.sh` | 2,301 | `4ac8452d36e5a8fda32aad58048b90a09cea03d1749babc1339ed1088e64ad50` |
| 8 | `.tmp_v831_slice_01_probe/v831_acceptance_wsl.sh` | 3,179 | `70e3a7c633db581fb4201815ecd3bf4780d2babd6169842cb04067f48b4821ed` |
| 9 | `.tmp_v831_slice_01_probe/v831_acceptance_wsl_v2.sh` | 3,320 | `33ad65ff73349ec13319a7f0fe0308cde870e358a5612558f04332549a5a9bb8` |
| 10 | `.tmp_v831_slice_01_probe/acceptance_verbatim.log` | 337,233 | `62b28fc0fb7dd740a6924c073d6658960f618190d0b36181119f08144afd969d` |
| 11 | `.tmp_v831_slice_01_probe/acceptance_wsl_run2.log` | 680,094 | `3caa85db25a957a2092d88a3b8029c57e46768db240c2ca22f46ccdf9d786d25` |
| 12 | `.tmp_v831_slice_01_probe/acceptance_wsl_run3.log` | 680,726 | `c196f970bddfbe69a4535577351cbf15ad6a6a905076585aed5d5b1b73643afb` |
| 13 | `.tmp_v831_slice_01_lV7zfs9S/pixel/result.json` | 14,022 | `aa0eb02edfe27faab57f647d7ea86e58c67d006065073fcc237ff0c5253fd1e7` |
| 14 | `.tmp_v831_slice_01_lV7zfs9S/pixel/mute.json` | 4,579 | `a2985821b8dd49f4c72efa6038ff334c7542aca2ff87ce2fa5e1f793b1e728bf` |
| 15 | `.tmp_v831_slice_01_lV7zfs9S/pixel/decisions.jsonl` | 807 | `fb8b8936f4c77824362723395d3a40b359c5f285a857940250188619f7526b54` |
| 16 | `.tmp_v831_slice_01_lV7zfs9S/teacher/result.json` | 17,754 | `b4e677fc6c65d7d7a8abbf0d6c28503b44a3dcde436219b414644707fcda123c` |
| 17 | `.tmp_v831_slice_01_lV7zfs9S/teacher/mute.json` | 4,583 | `f38e80ae44f66b81e0d9055063bd13b58e9e7cdac3b92ba670d193f959cfe483` |
| 18 | `.tmp_v831_slice_01_lV7zfs9S/teacher/decisions.jsonl` | 803 | `8032586a7c168c87644e6d40f8f816b2fb23215d5b3d1380c295988b9d483357` |
| 19 | `.tmp_v831_slice_01_eK6pxCuY/pixel/result.json` | 57,752 | `e8b65c403a2a298b8be0789293ea30b90743491fd712c9d98fb43beed3e65f7e` |
| 20 | `.tmp_v831_slice_01_eK6pxCuY/pixel/mute.json` | 4,579 | `3a91b3feae6f3320c578c6f8e977336a2564670457c869edaa9917759ad61d85` |
| 21 | `.tmp_v831_slice_01_eK6pxCuY/pixel/decisions.jsonl` | 329,765 | `6fd73bfe8de1b80557a9848b024db3042014e987176be5b8abf68d7c0126adef` |
| 22 | `.tmp_v831_slice_01_eK6pxCuY/teacher/result.json` | 62,801 | `11386b331c906310f3a0a7740e622bb150f15c24f768d7b349dd4cec7fe5c874` |
| 23 | `.tmp_v831_slice_01_eK6pxCuY/teacher/mute.json` | 4,583 | `06df6c83a09687ea806b2f71f43471c116876420e3a1d18647792d448f673ab1` |
| 24 | `.tmp_v831_slice_01_eK6pxCuY/teacher/decisions.jsonl` | 5,264,344 | `634fc709ec159f91a98885312c1447c178b4da80cb54438a457f860620681d83` |
| 25 | `.tmp_v831_slice_01_YmVD4nBX/pixel/result.json` | 72,985 | `e5d17f3bfdd5a8f01720bf08f9e73305b956559cec9148e3382bc3972c9b948d` |
| 26 | `.tmp_v831_slice_01_YmVD4nBX/pixel/mute.json` | 4,579 | `b07dd6933b58c8a7ee3b9aa4cda352a4d021e902f144cd65ea3bf4177eb76c67` |
| 27 | `.tmp_v831_slice_01_YmVD4nBX/pixel/decisions.jsonl` | 404,085 | `b426ce6d1584c69c27796500f6efc863e7d2899a2fc363a13d363ba0490de889` |
| 28 | `.tmp_v831_slice_01_YmVD4nBX/teacher/result.json` | 77,058 | `64d91af4ba3e98ac0ecb18e94163ec79312e4b24c5a75a7fcb29410976130b01` |
| 29 | `.tmp_v831_slice_01_YmVD4nBX/teacher/mute.json` | 4,583 | `92b851b6ee4af65cb35c043052cdb0676b83a05c8e1b207f95fd6e99f63e9301` |
| 30 | `.tmp_v831_slice_01_YmVD4nBX/teacher/decisions.jsonl` | 8,890,432 | `d774338bd7ceef4585402cdd146841f7419f2a92d7d1a8771f1fdd96899d8498` |

### 38.2 跳过说明（未入册 0，以下物件不属登记范围）

| 物件 | 处理 | 理由 |
| --- | --- | --- |
| 三个运行根下的 `pixel/game/`、`teacher/game/`（各 496,968,813–496,968,815 B，合计约 2.85 GiB/9 份） | 不入册，只引路径与 sha | §8.1：本机私有游戏目录，是 `--data-source` 的整目录副本；其身份已由各 `mute.json` 的 `copy_before_start`/`copy_after_exit` sha256（`cfa292ae…`）+ `diff_byte_offsets` 固定，逐字节复制不提供额外信息 |
| `.tmp_v831_slice_01_l6OqKAS4/`、`.tmp_v831_slice_01_NANazung/`（各 4,096 B） | 不入册 | `mktemp -d` 建目录后脚本在 pytest 行即中止（§38.4 第 3 条），**目录为空、未启动引擎、未写文件**；目录空这一事实已由 `du -sb` = 4,096 B 记录在回执 §5.0 |
| `.tmp_v831_slice_01_probe/pytest_win/`、`pytest_win2/`、各 `*.sh` 探针脚本 | 不入册 | 本单临时探针与 pytest `--basetemp` 产物，非证据；所有结论已由入册的 6 件脚本/日志承载 |
| `/root/v400-data/game`（496,968,814 B）、引擎 `rl_v3`、教师磁带 `slots.jsonl` | 不复制、不入册 | §8.2 红线：既有资产不移动、不覆盖。身份在本节与本回执 §1 逐项记录（引擎 `691bfd77…`、磁带 `f759de7d…`、源 cfg `48bf7621…`），运行前后 `du -sb` 与 mtime 逐项相同 |
| `Document/Update/Alpha/v4.2.0/evidence/T08/**`、`.tmp_v420_evalpulse/positive_control.json` | 不动、不重复登记 | 只读对照物；`positive_control.json` 的 sha256 `9cdd9033…` 已记在回执 §9，其内容自 v4.2.0 起未变 |

### 38.3 与 §1–§37 的交集

登记前 `ASSET_SYNC.sha256` 共 **1,424** 条非空行；本轮 30 条路径**两两互不相同**，且与既有 1,424 条**交集 0**（`v8.31.0` / `v831` / `v8_31_slice` 三个串在既有清单里命中 0 次；三个 `.tmp_v831_slice_01_*` 运行根为本轮新建）。`ASSET_SYNC.md` §1–§37 **一字未动**（追加前 `chars=772,305`、`lines=4,585`、末字节 `0a`、sha256 `75dadd2e…`；追加方式为"读原文 + 末尾拼接"，前缀 `StartsWith(原文)` = **True**）。

### 38.4 校验结论

**1. 发现的真实缺陷：`ASSET_SYNC.sha256` 是 CRLF，`sha256sum -c` 对它 1,424 条全部失败。** 追加前实测：

```
$ sha256sum -c ASSET_SYNC.sha256
sha256sum: 'Document/Update/Alpha/v4.1.0/README.md'$'\r': No such file or directory
… 1,424 条全部 "FAILED open or read"（OK = 0）
```

原因是该文件 **1,424 行全部以 `0d 0a` 结尾**（`tr -dc '\r' | wc -c` = 1,424），GNU `sha256sum` **不剥离** 行尾 `\r`，于是把文件名读成 `README.md\r`。**AGENTS.md §8.3 让下游执行 `sha256sum -c ASSET_SYNC.sha256`，而这条命令在本机当前状态下一个文件也校验不了。** §37 记录的"1,423 OK / 1 FAILED"与本机现况不符——该文件此后被以 CRLF 重写过（本机 Windows 检出 `core.autocrlf=true`，一次 `git checkout`/编辑器保存即可造成）。

**2. 本轮处理（只改行尾，不改任何 (sha256, path) 对，且先留底）。** 先把原字节备份为 `.tmp_v831_slice_01_probe/ASSET_SYNC.sha256.pre-lf.bak`，再按"去掉全部 `0x0D`"把整车清单规范到 LF 并追加本轮 30 条。可核对的零信息损失判据：

- 去 CR 后的内容 `==` 原文去 CR 后的内容 `+` 本轮 30 条（`StartsWith(original_without_CR)` = **True**）；
- 1,424 条既有 `(sha256, path)` 对的**字符内容一字节未改**（只删了行尾 CR）；
- 追加后 `sha256sum -c` 的 OK/FAILED 计数见第 3 条。

**3. 追加后的实际校验计数**：

| 量 | 值 |
| --- | ---: |
| 清单总条数 | 1,454 = 1,424 + 30 |
| 本轮 30 条 | **30 / 30 OK** |
| 既有 1,424 条 | **1,350 OK / 74 FAILED**，退出码 1，`WARNING: 74 computed checksums did NOT match`——74 条**全部是"内容不符"**（`FAILED open or read` = **0**），即文件在盘上但字节已变；本轮**未修改其中任何一条**，只报告（分类见第 5 条） |

**4. 运行产物自洽性（本单三项独立复核）**：① 最终一轮两条轨迹的 `result.json` 里 `accepted=true`、`exit_code=0`、`problems=[]`；② 教师轨迹 `expected/match` 21 项**全 true**，且三个动作摘要与外部独立实现 `positive_control.json`（`9cdd9033…`）**逐位相同**；③ 源数据目录 `/root/v400-data/game` 运行前后 `du -sb` = 496,968,814 B、17 文件、`th08.cfg`/`score.dat` 的 sha256 与 mtime 全部相同 ⇒ 本轮**没有触碰任何受保护资产**。

**5. 74 条既有失败的分类（只报告，不改条目）**：`Document/Update/Alpha` **63** 条（61 个 `.md` + 2 个 `journal.jsonl`）＋ `tools/rl` **11** 条 `v8*.py`（`v8b0_eventsample` / `v8f0_critical` / `v8j0_earlybomb` / `v8k0_rulesweep30` / `v8l0_unified` / `v8m0_bombrl` / `v8n0_rulefine` / `v8o0_ruletight` / `v8q0_t33tiers` / `v8r0_urgency` / `v8t0_densereward`）。逐条 `git status --porcelain` 复判：**73 条在本机工作树里本身就是 dirty**，与 §38.3 记录的"3,893 行既有未提交改动/删除"同源；剩 **1 条**工作树 clean 但仍不符（`Document/Update/Alpha/v4.2.0/journal.jsonl`，与 §35/§36/§37 记的"唯一允许失败项"一致）。**没有一条属于本单**（本单 30 条 = 30 OK / 0 FAILED），也没有一条落在引擎、数据、checkpoint、`configs/` 或 `third_party/` 上。口径提示：§37 的"1,423 OK / 1 FAILED"是在清单仍为 LF 时得的；该文件此后被以 CRLF 重写，故本节 **1,350 OK / 74 FAILED** 是本机**当前**口径，两者不矛盾但不可混用；清单 LF 化后的 `sha256 = b5ed071300aabadb976c2e8297fcffc39d86e640a3c82a494ab57bb51ae79161`（`chars=195,833`、`lines=1,454`、`cr=0`）。
## 39. v8.31.0 补遗与 A7/A5 复核资产（2026-11-26）

### 39.1 登记范围与逐文件表

本轮登记 **12 条 / 34,843 B = 34.0 KiB**：v8.31.0 补遗回执 **1 件** + 只读复核脚本 **11 件**。**入册 12、未入册 0**。本节全部内容在**未启动引擎**的前提下产生（只读源码、只读既有产物、离线复算）。

| 组 | 文件数 | 字节 |
| --- | ---: | ---: |
| `Document/Update/Alpha/v8.31.0/TRAINING_HOST_REPLY_02.md` | 1 | 16,148 |
| `.tmp_v831_slice_02a/*.sh`（A7 / 脉冲 / 炸弹 / 77 帧四组探针） | 11 | 18,695 |
| **合计** | **12** | **34,843** |

#### 39.1.1 逐文件表（12 条）

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `Document/Update/Alpha/v8.31.0/TRAINING_HOST_REPLY_02.md` | 16,148 | `b62b25b31074c0582742c91ef412fbb14ebb1226e0afdc6b808d02e60871e6a4` |
| 2 | `.tmp_v831_slice_02a/a7_pair_conflict.sh` | 3,343 | `0e278fae3f6df390d5c6112db7ae631fd0056a3a618c08beeb2521ad24d576c9` |
| 3 | `.tmp_v831_slice_02a/a7_all_shards.sh` | 2,773 | `06417b714f0ab2250e5d8bb58c02167e12798481a54789ae74f12ae33dd6eda7` |
| 4 | `.tmp_v831_slice_02a/a7_field_semantics.sh` | 2,913 | `6f2f9e3b971bd6dc511d0fc71dedbbe64f6e111ca1fb69fdb8ed07fecf53d357` |
| 5 | `.tmp_v831_slice_02a/a7_pulse_hypothesis.sh` | 3,899 | `fa84d3b1336e1f651e1cad90645fa9326eb5ffde89a01232b47a72c62390372d` |
| 6 | `.tmp_v831_slice_02a/bombs_probe.sh` | 541 | `a0e7dc0cff544c6fd43cc43812040dbc3f27068dedf1c71b411e92ad5b29954d` |
| 7 | `.tmp_v831_slice_02a/bombs_and_gap.sh` | 1,633 | `50446b7fa688a409a7203263790f38cd58ed996aca10c63d7f1b68adb5b4bd96` |
| 8 | `.tmp_v831_slice_02a/gap_mechanism.sh` | 461 | `88775215299b85a33a1be24a3f6551ecb38ba65e9e4f486e2bfd95070c6ff73e` |
| 9 | `.tmp_v831_slice_02a/gap_mechanism2.sh` | 579 | `b6cb2786422e943105373ce4929eb6e427cfa3d448616977e60ff2e59653a190` |
| 10 | `.tmp_v831_slice_02a/gap_mechanism3.sh` | 270 | `c0ac794b5542fcc8b0ced5a335adc9aad3678b64adfa78041ea3c374f1cbdee5` |
| 11 | `.tmp_v831_slice_02a/gap_mechanism4.sh` | 173 | `3b5b77c133c3de9391e411709308000c8a81417c2fcaf88ca71f8e9eea81de17` |
| 12 | `.tmp_v831_slice_02a/gap_timeline.sh` | 2,110 | `f8462e53be8879f1165cf8757d256e947b499ecc259119fda319df54fdd916b8` |

### 39.2 跳过说明

| 物件 | 处理 | 理由 |
| --- | --- | --- |
| `D1` 语料 8 个 seed-12 shard 的 `labels.jsonl` / `shard.json` | **不入册**，只引身份 | §8.2 红线：既有数据资产不动。身份已在回执 §1.1/§1.3 逐件记录（s0d1 `labels.jsonl` = 5,822,072 B / `457728a9…ccbd`；8 个 shard 的 `labels.jsonl` sha256 全部列表化） |
| `.tmp_v400_d1/**`、`.tmp_v420_data/**`、`.tmp_v831_slice_01_*/**` | 不重复登记 | 前两者是既有资产；后者已由 §38 登记，本轮只读 |
| 引擎源码 `/root/Touhou_RL/third_party/th08-rl/src/**` | 不复制、不入册 | 既有资产；本轮回执只引行号（`GameManagerSetup.cpp:123-130`、`GameManager.cpp:357-359`/`:388-396`、`Gui.cpp:687-688`、`rlharness.cpp:107`/`:1362`、`main.cpp:143-146`） |

### 39.3 与 §1–§38 的交集

本轮 12 条路径与既有 **1,454** 条清单**交集 0**（`v8.31.0/TRAINING_HOST_REPLY_02.md` 与 `.tmp_v831_slice_02a/` 均为本轮新建）。`ASSET_SYNC.md` §1–§38 一字未动（追加前 `chars=784,164`、`lines=4,683`、`cr=4,683`、sha256 `872e5ebc…`；追加后前缀 `StartsWith(原文)` = **True**）。

### 39.4 校验结论

1. 清单在 §38 已 LF 化（`cr=0`）。本轮追加 12 条后：**1,466 条**，本轮 **12/12 OK**。
2. 全清单 `sha256sum -c`（1,466 条）= **1,392 OK / 74 FAILED**，退出码 1；74 条与 §38.4 第 5 条**完全相同**（63 条 `Document/Update/Alpha` + 11 条 `tools/rl/v8*.py`，全部为既有工作树漂移，`open or read` = 0），**未新增任何失败项**；1,466 行全部 well-formed（`bad=0`），`cr=0`。
3. 本轮回执的三项结论都可**离线复现**：`a7_pulse_hypothesis.sh` 打印 8 个 shard 的"原始/仅差 SHOOT/真正候选级"三列；`gap_timeline.sh` 从 `decisions.jsonl` 重放出 13,195 帧与 137 个上升沿；`bombs_probe.sh` + 源码 `sed` 给出 `initialBombCount` 的三处调用点。
## 40 V831-SLICE-02 / V831-PILOT-03（2026-11-26）新增登记

本轮登记 **V831-SLICE-02 / V831-PILOT-03** 的交付文件、运行证据与脚本/中间报告。登记过程**未启动引擎**（26 次引擎启动全部发生在 `.tmp_v831_followup_02a_gwZ4fA8u/` 的既有台账里，本节只引身份），只读既有产物，并且只对 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256` 两个登记文件做**追加**。

| 组 | 条数 | 说明 |
| --- | ---: | --- |
| 交付文件 | 4 | 脚本 / 测试 / 配置各 1 件 + 训练端回执 1 件 |
| 运行证据目录 | 2 | 1 个有效运行 + 1 个失败尝试；只登记目录身份 |
| 关键证据文件 | 38 | 逐个 sha256，不复制（含 `03/strength/` 3 条、`03/diag/teacher_arm/` 5 条） |
| 离线复拟合权重 | 9 | 一次离线 BC 的初始/最终权重 6 件 + loss 3 件 |
| 脚本与中间报告 | 132 | 非资产，只登记身份 |
| **合计 sha256 条目** | **183** | 交付 + 关键文件 + 离线复拟合 + 脚本（目录不入 `sha256`） |

### 40.1 交付文件

本轮新增交付 **4 件 / 955,227 B**，全部为本轮新建路径，入册 4、未入册 0。其中 3 件是代码/配置/测试，1 件是本轮训练端回执 `Document/Update/Alpha/v8.31.0/TRAINING_HOST_REPLY_03.md`。工具与测试在两轮里各改过两版，此处登记**第三批定稿版本**，不是任何中间版本；回执哈希取父代理更正后的定稿值。

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `tools/rl/v8_31_followup.py` | 766,502 | `6520ca935c789da0fd846c48af7676e3988baa9912b764f659cfd7c357de5bde` |
| 2 | `tests/test_v8_31_followup.py` | 123,807 | `cf6194145965c2a11ebe3c094a8ecc32bc03bcc016bf7174ce8955e082d66acc` |
| 3 | `configs/v8_31/pilot_03_roots.json` | 18,071 | `c15b1b43fdd827f22cb56493b20be1bd0177773c003ba21ac030e6791844cc1b` |
| 4 | `Document/Update/Alpha/v8.31.0/TRAINING_HOST_REPLY_03.md` | 46,847 | `63dcc5d26cb5d62877921d64f5f24ca40088f21786c9223ee29f07c3230f135a` |

口径：`tools/rl/v8_31_followup.py` 是 SLICE-02/PILOT-03 的打包执行器；`tests/test_v8_31_followup.py` 是离线单测（不需要引擎与数据）；`configs/v8_31/pilot_03_roots.json` 是 PILOT-03 的 4 个 dev root（seeds 1020–1023）配置。

### 40.2 运行证据目录

#### 40.2.1 目录身份（`du -sb` 表观字节，与本轮工单同口径）

| 目录 | 条目数 | 总字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v831_followup_02a_gwZ4fA8u/` | 664 | 13,044,502,699 | **有效运行**：`02a`/`02b`/`02c`/`03` 四阶段齐全，另有 `ledger.jsonl`、`run.json`、`verify.json`；台账累计 `engine_starts` = **26**（末条 `committed_before.starts` = 26，bookkeeping 行 `requested.starts` = 0），与 26 个 `mute.json` 一一对应；目录内 `pytest-wsl/` 79 项为 WSL 侧 pytest 临时工作目录，不计入证据 |
| `.tmp_v831_followup_02a_gZ3CwofM/` | 4 | 866,423 | **首次尝试失败**：`verify.json` 的 `inputs_present` 缺 `02a/prefix/result.json`、`02a/reference/result.json`，`gate.passed=false`；目录内**无 `ledger.jsonl`、无 `run.json`** ⇒ 未启动引擎、无台账行，其余字节全部是 `02a` 三个 JSON |

条目数 = 常规文件 641 + 无法在 Windows 侧 stat 的 WSL 符号链接 23（均在 `pytest-wsl/` 下，`WinError 1920`）。

#### 40.2.2 关键证据文件（逐文件，38 条）

| # | 路径（相对 `.tmp_v831_followup_02a_gwZ4fA8u/`） | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `ledger.jsonl` | 106,292 | `f4b1cb83481b75f23e70b66fab2088ae9e56e7ae60c9bbda1fdd02e1da69ca08` |
| 2 | `run.json` | 382,171 | `fe6b3e67fb16752318084f04ff8cc40c50eb8b5e5ba9c3a6dd0f12d9ee0dcd02` |
| 3 | `verify.json` | 674,978 | `fcf8805054acf1b7a6188c72a335ccb7c14cf4027288c54562a6eeb125ea596f` |
| 4 | `02a/sources.json` | 431,444 | `2de1c50c8c2d21cd65461bb80263254a917e69635ed6208bd924eb109fb7d872` |
| 5 | `02a/adapter.json` | 421,843 | `043e9b4faa38e9ab27f5f543e9cc2db76ddc8f314915b108eb58b59a9222dec4` |
| 6 | `02a/compare.json` | 36,773 | `4fbac360e7b7c90bc1db0bcfae76703cf008bfe31eedd0f8aff5b60cc7e661ed` |
| 7 | `02a/prefix/result.json` | 499,392 | `158d2fcca37b985aad34b85500bd9a5b9844587551cc23c3036745dff271c596` |
| 8 | `02a/prefix/mute.json` | 4,598 | `d424e731c4a83b08ab79d9b59af43f48f6fbf895ef5b066615e6fda5c4a3c430` |
| 9 | `02a/prefix/history.jsonl` | 1,064,458 | `a988e12f311137369c8d03cb4884693a1ddbeeba1197a9632f2c7a3665f835b4` |
| 10 | `02a/prefix/hashes.jsonl` | 380,285 | `54ca3eab3ba5de21e41b36965654ad20fce136a5f627ea145b61bec16bd1a74f` |
| 11 | `02a/reference/result.json` | 499,183 | `f0a563be8c69dad2e1950842c4146373a3774d266bd264234027ab4c93632333` |
| 12 | `02a/reference/mute.json` | 4,604 | `07df970dc77a713831fecbe295d21e405f1ef360fce92ec36d8aa298c85bbe5a` |
| 13 | `02b/recon/result.json` | 420,686 | `d916ca28048590162f13947d94c974a419624b9a7256d1bb11a35241ee4596ad` |
| 14 | `02b/recon/decisions.jsonl` | 20,093,418 | `584ccd1e3fb0d979ad892f8a683128b2c774fa1daec436d7e63163990ce828ec` |
| 15 | `02b/task_card.json` | 17,044 | `52a08005eb7f8a56a1dbf739a595d3047043eb9f8fa19401000ed78cd834c002` |
| 16 | `02b/task_card.md` | 4,500 | `8260f508052a3fb4e41bf45553d44d628b67b9c94eecee3cb2c3af45adde2144` |
| 17 | `02b/nomination.json` | 492,432 | `56ca51b48e385e7414b6c7fa8550cacbbc8b7bc39b03477ca1a203790c4a5fa0` |
| 18 | `02c/intervene.json` | 452,438 | `992672612aa162548d80d8644ed5207bd8957e19a776eca1ee9676ac7635a0c0` |
| 19 | `02c/arm_T/decisions.jsonl` | 726,657 | `684a7d069704f9d15fa5e0d6d39ac1fcb0266eaca6a90a7c0cad7959f1d00056` |
| 20 | `02c/arm_T/mute.json` | 4,596 | `60b010fcf9387ffcabb4ec964ddaf0e8b0135c94552756f0bf7c85c3ee1de2d5` |
| 21 | `02c/arm_T/prefix.jsonl` | 1,096,264 | `8f0a955f6010c57b507630ca0a2e3020312813a6f84e6bd7226ece865a9f7484` |
| 22 | `02c/arm_C/decisions.jsonl` | 323,237 | `9dac84a02d2ed69b07f716c11043d4fc1afdbbe574d2dbf7c226aad7d7853fa1` |
| 23 | `02c/arm_C/mute.json` | 4,596 | `9782b3a23220edfd42f1d3e6e8cee16de2352702bbbd98ca11e228cc33675c77` |
| 24 | `02c/arm_C/prefix.jsonl` | 1,096,264 | `8f0a955f6010c57b507630ca0a2e3020312813a6f84e6bd7226ece865a9f7484` |
| 25 | `03/pilot.json` | 1,241,538 | `ef84047a763828f18eac419ed4bc360d232e7fde490d0bfcc2cbba9e89027db9` |
| 26 | `03/train/train.json` | 384,456 | `4ee8f9e84c7722ee78b5249c282341f044dda1340bd48240572b8730d5fc6b89` |
| 27 | `03/train/weights_initial.pt` | 10,468,055 | `5ef59855b41558d8d1742ac3963a2011c161d3cb0c02bb48f66a11ab68dff675` |
| 28 | `03/train/weights_final.pt` | 10,467,859 | `336397ed6f3adc738264d9dd1cc9299b6345348306c77aa33450f867286c4084` |
| 29 | `03/open_loop/table.json` | 572,383 | `16bc87225c15b89455ac7b32b20a6fd27375b9099146f937b7f41d833813a061` |
| 30 | `03/open_loop/rows.jsonl` | 152,916 | `11cb3b44df2ac01f1982c21caf694cccfeb73944dc69355301e39be50ff88f26` |
| 31 | `03/strength/result.json` | 407,317 | `2c3304e2486ba47cc59ca46529513c46e570f39e5fee15721e8ab13b4ed1f867` |
| 32 | `03/strength/arm_open_loop/result.json` | 409,116 | `c250c2eb72db109b81ea5639e052b2c24d1fbb2b83cb45e738fcd22142ebe08a` |
| 33 | `03/strength/arm_constant_12/result.json` | 412,676 | `1262f8cef5463dea0532926148e9688acbb02fa221cef0d2d0d77b8f8bad5e5f` |
| 34 | `03/diag/teacher_arm/result.json` | 434,182 | `96823ee2fd9ef9fd2dff2efa20f24b5b96820a8d2ee090633594fa393925dccc` |
| 35 | `03/diag/teacher_arm/1020/result.json` | 414,922 | `eb17efe23770e0030e401a56d66127bd9c534a65b1f30a5272df06b3841ba311` |
| 36 | `03/diag/teacher_arm/1021/result.json` | 414,974 | `2ca2ef902da869a7df1ef551a21f13c942fb256dfab6eca8c77f7d9ad72ac428` |
| 37 | `03/diag/teacher_arm/1022/result.json` | 412,621 | `231659132c13cb8f7b10544766ad4fe5e3cab6af1c5956a4e4e01a126e3b7b9b` |
| 38 | `03/diag/teacher_arm/1023/result.json` | 413,961 | `e0352555ecdc14463f4694a4b50d4cdfbacc9c06f04242b853003fd931c57bc5` |

38 条合计 **55,845,129 B**。只登记身份，**未复制**任何一个文件。

第 31–33 条是 **L51 开环强度检查**证据：`03/strength/result.json` 里 `strength_passed` = **false**（既作字段 `"strength_passed": false`，也作 gate 检查项 `strength_passed` / `ok=false`），顶层 `verdict` = `strength_failed`（`seed=13`），两条 arm 分别 `open_loop` / `constant_12` ⇒ **强度门未通过**，按失败记。

第 34–38 条是**教师续带诊断**证据（4 次引擎启动）：`03/diag/teacher_arm/result.json` + 4 个 seed 逐根 `result.json`；汇总的 `teacher_completed_roots` = **`[]`**、`interpretation` = **`entry_or_task_seed_specific`**，即 4 个 dev root 上教师续带均未达成 ⇒ 该诊断判**负**，不是通过。每个 seed 目录里另有 `decisions.jsonl` / `prefix.jsonl` / `mute.json` / `game/`，按目录身份登记，不逐文件列。

#### 40.2.3 `03/dev/`、`03/smoke/`、`03/strength/`、`03/diag/`：只按目录身份登记

这些目录里是引擎运行副本（`game/`）与轨迹，文件数过多，按工单要求**不逐文件列**，只给文件数 + 总字节 + 一句说明。

| 目录 | 常规文件数 | 总字节（`du -sb`） | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v831_followup_02a_gwZ4fA8u/03/dev/` | 252 | 5,982,862,573 | 4 个 dev root（seeds 1020–1023），每个 root 63 个常规文件（`open_loop/` + `train/` + `game/`）；`03/pilot.json` 的 `gate.passed=false`、`verdict=no_nomination` |
| `.tmp_v831_followup_02a_gwZ4fA8u/03/smoke/` | 69 | 1,519,373,888 | 1 个 smoke root（seed 1000）63 个常规文件 + `open_loop/` 2 + `train/` 3 + 根 1 |
| `.tmp_v831_followup_02a_gwZ4fA8u/03/strength/` | 43 | 996,873,140 | L51 开环强度检查：顶层 `result.json` + 2 个 arm 目录（`arm_open_loop/`、`arm_constant_12/`），seed 13，`verdict=strength_failed` |
| `.tmp_v831_followup_02a_gwZ4fA8u/03/diag/` | 85 | 1,993,537,281 | 教师续带诊断（`teacher_arm/`，4 个 seed 1020–1023 各一套 `game/` + 轨迹 + `result.json`，共 4 次引擎启动）；`teacher_completed_roots=[]` |

#### 40.2.4 与工单路径清单的三处口径修正

| 工单所列路径 | 实测 | 处理 |
| --- | --- | --- |
| `.tmp_v831_followup_02a_gwZ4fA8u/02c/arm_T/result.json` | **不存在** | 改登该 arm 的 3 个实际产物：`decisions.jsonl`、`mute.json`、`prefix.jsonl` |
| `.tmp_v831_followup_02a_gwZ4fA8u/02c/arm_C/result.json` | **不存在** | 同上 |
| `.tmp_v831_followup_02a_gwZ4fA8u/03/open_loop/masks.json` | **不存在** | 按工单「不存在就跳过并注明」处理；同目录改登 `table.json` + `rows.jsonl` |

#### 40.2.5 离线复拟合权重（一次离线 BC，`.tmp_v831_followup_probe/diag_refit/`）

3 组配置（A 原生 2f / B 类别平衡 / C D1-seed13 半密度）各一对初始/最终权重 + 一条 loss 曲线，共 **9 件 / 62,910,494 B**。这是**一次离线 BC** 的权重，不是线上训练 checkpoint：只登记身份，未复制。`diag_refit/` 里其余 `artifact_sha256.json`、`refit_bc.log`、`t04_self_check/*.json` 不属本节登记范围（见 §40.4）。

| # | 路径（相对 `.tmp_v831_followup_probe/`） | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `diag_refit/A_t04_seed10_native_2f/weights_initial.pt` | 10,467,863 | `38de77824f714d0ef5832fa19acbc914c68e2644cebc5e889682e31ffed90fdb` |
| 2 | `diag_refit/A_t04_seed10_native_2f/weights_final.pt` | 10,467,731 | `e633193948bc0e8bf5d5f2d1aacb36201c42a8f17ebb4b56d9b664bcadd2a38b` |
| 3 | `diag_refit/A_t04_seed10_native_2f/loss.jsonl` | 34,652 | `ed03c2488e5c919fa3dffc1aed3f6ca30a626523fc87cfdcf3270d0026f0b3ad` |
| 4 | `diag_refit/B_t04_seed10_class_balanced/weights_initial.pt` | 10,467,863 | `d6e7e62200e63e3f7acea73ae42f39cbefda6bebb09e61a9dd3f84bd0bd465da` |
| 5 | `diag_refit/B_t04_seed10_class_balanced/weights_final.pt` | 10,467,731 | `b7268751bbd09f2d5fd31bedb9aea60f4d7593a4368eccd44aa8f8c7cf29c306` |
| 6 | `diag_refit/B_t04_seed10_class_balanced/loss.jsonl` | 34,421 | `4fe7dbc56f9644bb18b4ad63bc264cc6edecc897b24860e5aef20dd315e5648c` |
| 7 | `diag_refit/C_d1_seed13_half_density/weights_initial.pt` | 10,467,863 | `d4dfb9cf9c841b42a1ce0e31ea4c1d1cee96fc3a4bf473ca1393f7341a476f5e` |
| 8 | `diag_refit/C_d1_seed13_half_density/weights_final.pt` | 10,467,731 | `46a9cd5044c8e9f87cd48a581bb1948dad619060ad0b02fc3e5a4785babdb97f` |
| 9 | `diag_refit/C_d1_seed13_half_density/loss.jsonl` | 34,639 | `fbf9d8e0b4eba7eb9db9b90a2cacc9178bb42e80610151d7c6f26a7fbe8b3de3` |

### 40.3 脚本与中间报告（非资产）

`.tmp_v831_followup_probe/` 下 `*.sh` / `*.py` / `*.md` / `*.json` 共 **132 件**（已排除 `pytest-*/` 临时目录；同目录的 `*.log` / `*.txt` 是日志与校验输出，不属脚本/报告类，未登记）。这些是**脚本与中间报告，非资产**：可随时删除或重生成，登记只为让本轮的探针、报告与复算入口可追溯。此表为本轮**最后一次**枚举的结果：诊断阶段新增的 `run_diag_teacher.sh`、`ff_*.py`、`refit_*.py`、`finalize_diag.sh`、`ledger_now.py`、`03_diag_refit.*`、`03_first_failure.*` 及 `diag_dry/` 下的 JSON 已计入，过期条目已替换、删除的条目已移除；`diag_refit/` 子目录按工单排除（其权重与 loss 见 §40.2.5）。

| # | 路径（相对 `.tmp_v831_followup_probe/`） | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `02b_eventscan.json` | 694,932 | `f036fd9cc8b8c67e4bc0698bb790a2b951036da1f0e2762fc0085e81c7f86ac9` |
| 2 | `02b_eventscan.md` | 50,933 | `ebb7168e8bde2019074b7fbb74d7f4bcdc83a020c7cfe5ac9fb95aca506136b2` |
| 3 | `03_diag_refit.json` | 55,608 | `73fce6d5e4e36e0898d6caaae8bd162ca4472c14e0a69f434944c3a9312dd9f7` |
| 4 | `03_diag_refit.md` | 14,099 | `0fd61afeec82ce62bef78a5a17c302faaa0de74fe5b70d32de5721ea8bb02174` |
| 5 | `03_first_failure.json` | 566,514 | `614b3cef24b6a9d406fa50ca7f24a331643a8acc9c216d7287bdaec1ace37f8b` |
| 6 | `03_first_failure.md` | 25,249 | `c6f185413455fa3aced44eeced1f43c706dc91754e836a173e07f571e0f01a86` |
| 7 | `03_recipe.json` | 30,596 | `3cf06eb32a9baab752df46502759180c5d8a8485c3c188c1498d8eb80eb39ff4` |
| 8 | `03_recipe.md` | 30,178 | `99f1b9724549bb441755ce82ec8539ee464181d795a46fdca048a2a812540480` |
| 9 | `03_roots_analysis.json` | 110,780 | `df04738b5f2efc35243bb90b3b10ca1f31f7c7c6ce4b6db5040e5d8073a09f9f` |
| 10 | `03_roots_analysis.md` | 24,817 | `2fc237434760cf97666d570aa6a2b28ea632f25377d83b52901fd142ba8488b8` |
| 11 | `SPEC.md` | 16,252 | `307cbf6268c616464f80f30b3ccb98e04e44842b98a35751a2440e48f8dcdbd3` |
| 12 | `_probe1.py` | 891 | `a604f78c2eaa4de160eb837541d9817ddc0e36bc2402a4d1ad2b1feb089a43f6` |
| 13 | `_probe10.py` | 1,143 | `7015ac574506a67ff81d5fe713842817d50270edff0c6c491897fdb7bd2b7016` |
| 14 | `_probe11.py` | 337 | `c3f484850728297f0b2635e411d60b851db1d57d6fbfcbbf70c90853ea5e9a45` |
| 15 | `_probe12.py` | 383 | `ff913e48626cfaf7e03841d2446e2ea646cd7f6aee96abac3a56532aad596ce6` |
| 16 | `_probe13.py` | 248 | `54753ade309412c0b139c6fb9f42b864e6b7ee22f38252d78624971c50dfb639` |
| 17 | `_probe2.py` | 695 | `a9665d9e099a6b26b17a49900f8ae048c5c3072b2340ab83758fb0333dad0b79` |
| 18 | `_probe3.py` | 522 | `d826ed927f752d5f7914b219f494a60f4f228277247df5ce055ecb657e5c7217` |
| 19 | `_probe4.py` | 610 | `c147b8318647ae0b680ad2181319b4bf7adbc04ebee1169c6364d337f342d09b` |
| 20 | `_probe5.py` | 625 | `a305b67899b50a518cff195b04e1a2997dd552e4d564251b4e92e264573c871f` |
| 21 | `_probe6.py` | 625 | `70ffcf1f3bd291ffe0a72e2510662a5cd5dc83c7c39597cb3032da0181fa0361` |
| 22 | `_probe7.py` | 341 | `1d8fe98f830ab5ea27930553ee69aaec32d5763fbcb2d0e1a9b8cf181c5a3c52` |
| 23 | `_probe8.py` | 690 | `c190d6873ed48bdfb45f297d17d8a100ee1768a6ad16f76322fb05d7557c577f` |
| 24 | `_probe9.py` | 378 | `d8f3b22adf8afdfc773e4d3fe39e59186fdd79406ad4b272312e80c0c20788bd` |
| 25 | `accept_full.sh` | 3,595 | `95faa504b4244f8619cad7b9d3583444ffe37e107132a6542d6e0c39bd2ecb07` |
| 26 | `add_roots_config.py` | 1,870 | `8dde2591f509f7d6e8c52dfd27f8432c731969c46eea27415c0dfd73524dd4a3` |
| 27 | `arm_table.py` | 2,385 | `ffb872b8ced513a6edebc92c325f6ec1290e42c2e3c90439c9130ede61e83f2c` |
| 28 | `check_offline.py` | 582 | `37961e375f531bd080b8ff3bc776ebe4c2d4a80bbb4401511349e900b7abd604` |
| 29 | `check_offline2.py` | 1,168 | `b8665a1aaa87da50199696e4890f91d3f801e8a7fa6766166a41ad65ffdc4b66` |
| 30 | `config_facts.py` | 1,560 | `beab977543078f41682d8c7c753289a89f2959f4db63ca4ff5e781047d0f625f` |
| 31 | `diag_dry.py` | 2,736 | `54e73842e39472cba089a0efdb72f23ef5807549470790fa84800e958dd09ab8` |
| 32 | `diag_dry/02b/task_card.json` | 17,044 | `52a08005eb7f8a56a1dbf739a595d3047043eb9f8fa19401000ed78cd834c002` |
| 33 | `diag_dry/02b/task_card.md` | 4,500 | `8260f508052a3fb4e41bf45553d44d628b67b9c94eecee3cb2c3af45adde2144` |
| 34 | `diag_dry/03/diag/teacher_arm/1020/result.json` | 3,120 | `c6d966b9c357c5bb3e2e414fde3e5ac96daa121bf11df3017d81970431a507e5` |
| 35 | `diag_dry/03/diag/teacher_arm/1021/result.json` | 3,120 | `7b8d28d95cec272882f1e35959f075976b3a796eb4418a75cfbd9dd8a8c9901d` |
| 36 | `diag_dry/03/diag/teacher_arm/1022/result.json` | 3,120 | `38faf6dc69410885872773aed918353081ade63e7518e91c564a5095f2e0c7f2` |
| 37 | `diag_dry/03/diag/teacher_arm/1023/result.json` | 3,120 | `3e18837d9abdca24cabcedfd374afbe537715c542c545f518bcfb7f481f818f0` |
| 38 | `diag_dry/03/diag/teacher_arm/result.json` | 409,693 | `09481388bd3aefa7ae72cbcba19b787b08244b5912194dd3e5d0d206d440d8d2` |
| 39 | `diag_dry/03/pilot.json` | 1,241,538 | `ef84047a763828f18eac419ed4bc360d232e7fde490d0bfcc2cbba9e89027db9` |
| 40 | `diag_dry_verify.py` | 1,217 | `44fa67920ea7954cc7f0a9efe2a2ea975243db42be77058df8c77ee2a3bd11e3` |
| 41 | `diag_teacher_block.py` | 80,507 | `b57863fa0feafc2dbdf12630e3c3cc8be447b3f9a2879ef666c53df5e4e978dd` |
| 42 | `diag_tests_block.py` | 14,356 | `a7b98b5932683b7969415a87a54761936226a8fb4a8ffaa87537bc4e013521b2` |
| 43 | `enemy_semantics.json` | 14,713 | `420b434de683d4b1dd81fc467c63dd9f8387ba22eecc7ae81c2741f262f9c52c` |
| 44 | `enemy_semantics.md` | 19,321 | `654e5866402a1c227ce8e194413ca5bafa0b81612b11a26d2845ff2a57e6bfd6` |
| 45 | `env_audit.sh` | 1,435 | `5f2a6a05b1473a7bbdd5785bf133e9234b0c0759c12817317efb8f1da0a791ee` |
| 46 | `es_check.py` | 5,597 | `76f547bddda3bfd19a1099967b6a75605ead738eb7396f7cb5bb739d439d504d` |
| 47 | `ff_probe_ckpt.py` | 1,093 | `3db8b1a976981d80f05d14c8c95d1dd1c64de4b4bedd5f0173dbf3a47f896bf1` |
| 48 | `ff_report.py` | 78,963 | `1113b958b56f9299a4270bf742d2659877722f36b37214df279d3e2e7753efcc` |
| 49 | `ff_validate.py` | 7,819 | `7aa9468711edbda7f0b00a0c3a824a4b9f009658b9a687dfc0b8eeab526467bf` |
| 50 | `final_check.py` | 1,953 | `c3f00499df7d0767e43434dc6785a8f436900e34ea946a232d5b5c192152f842` |
| 51 | `final_check.sh` | 1,127 | `dee41be2b13a978eb482bb50c859d0b2c29cbb3f95b57c2c43aec755ce1fb5f6` |
| 52 | `final_summary.py` | 2,177 | `9a2272019a4a5f3e445de890555d96d56d7d1acb927e63902afb7f21383c56d2` |
| 53 | `final_summary2.py` | 1,582 | `247b0e2e30334fb2ea806451104ffcea07453fc1d3db03c6d4be686700fbdca1` |
| 54 | `finalize_diag.sh` | 1,052 | `de9e998755e85e06a1eeb317fe0ceff1b9f71f28fef6250f32ab5f37a8127ac0` |
| 55 | `finalize_run.sh` | 691 | `77907c3da31171c8cec64fd4dba49205d8ae682df48d46774bfbdfc3f6880606` |
| 56 | `hash_pairs_probe.sh` | 1,110 | `24c9a4c5c5bb9bd5e21c54aaa33c7d7f248047cace50e344b80a75746b7710ae` |
| 57 | `ledger_now.py` | 524 | `4c32634375deae93a485e2f91c8931403073a9cd4413900ef6f02a8d66117f62` |
| 58 | `main_check_sources.py` | 3,236 | `04a0ed9c162cc336cd419e88963e33b17f9c0a5c779acda294a32e30a55eb895` |
| 59 | `main_verify_out/02a/adapter.json` | 404,272 | `0086ee660ab61e955c9bed42dca56b5cab4c15158505a5d8ba971aacecbf7ad8` |
| 60 | `main_verify_out/02a/sources.json` | 415,800 | `7699a2b8d4bd30f2eaaec92a17ee1815a6cd4a0a378866acf68a8db1ca525040` |
| 61 | `main_verify_out/verify.json` | 387,608 | `c532a66fa973d5d44bed85d6fa03bb66d69837f2cfbb8baa8fbea64ebe91579a` |
| 62 | `patch_meter.py` | 750 | `c8fcbd02d1ad7e8fa12fe29ffcf3886c15f348661e031648708f6d5ebb855b34` |
| 63 | `patch_run.py` | 738 | `88eb7deafe4e376dce2565ed26cdf7bd3550761edc888587727dbcc8ae598649` |
| 64 | `patch_stage.py` | 703 | `addc553cc11f660496a11a19f848d25461c39903fbd11cd2d983bc172ba2634f` |
| 65 | `patch_verify.py` | 887 | `76f72cbcc18fc587f3f02ba4f105b4c3157f3230987632df96b7771cca53a606` |
| 66 | `peek_s1.py` | 1,154 | `dd35afbb44abeed26d0c6f2e39883e4974bfc9673dd3660aa5c43efddb67d2b3` |
| 67 | `peek_s2.py` | 1,073 | `3631fa2c259cd361770cceda74187c6f9254339ca16a7806bdffda27c1c0cee1` |
| 68 | `peek_s2b.py` | 1,925 | `21239a7917123a2ee55761a15c8853af9602c75c8baca837450f38187c37a66a` |
| 69 | `peek_s3.py` | 1,630 | `17facd52ee5a8e6c81133f83cf00e9e3d44c76d47b20fc30148d2489da14006b` |
| 70 | `pilot_block.py` | 114,695 | `bbdbf54155404e77ef97e9931f3ee11b8d590ef57348a3e6a7b9d464de1bf29e` |
| 71 | `pilot_dry.py` | 4,739 | `a8094c5e2ce69b38b9e8d85c6af52e5d120251f10556b21440c425fb6731001d` |
| 72 | `pilot_dry/out/03/smoke/open_loop/table.json` | 572,389 | `13f1bf6d89f873b67ec223864d11377c9b4b32180602d8cfc8dfe2120a2fba67` |
| 73 | `pilot_dry/out/03/smoke/train/train.json` | 384,509 | `57e817b4719a9c77afacf4044d8333a459f762f29f399b07c9b0e3080f758052` |
| 74 | `pilot_dry/pilot_03_roots.json` | 18,071 | `142fa5329d3b8d2a4e73165633e955b66af43ec3b5d4f7debf2eb8ea94b39a95` |
| 75 | `pilot_report.py` | 3,556 | `409dd8c7ee156201ba26b4c09d129de5ddd30d4da967fe990143a370695bbd9a` |
| 76 | `preflight_02bc.sh` | 1,316 | `fe3eb8064b507908e75f31577c4b976a6521b0b95ea25b0157a66c5c8e80139b` |
| 77 | `probe_d1.py` | 1,821 | `959d48f68c7207ffd125c7922bd6cd439ac1ea71fec8e0485598a6ec24a4233b` |
| 78 | `probe_d1b.py` | 1,818 | `10afac1455bacfebb7058bb2bbd4927f0b7ec3fdb522da682fd975d362252763` |
| 79 | `probe_imports.py` | 1,219 | `cef66bfa7dd4c3b423669627360ac73a8a8ebcb7a92fa14c23cfa176ec03639a` |
| 80 | `probe_inspect_adapter.py` | 1,431 | `e214c2fa874d698331285daf13efeae2e01d8e2ff4b67084a8cf5eae6b70390a` |
| 81 | `probe_inspect_sources.py` | 1,496 | `59c43776ef295febcceb328542007f3d7e1c43a4e28738c1a44be257bb0aebb4` |
| 82 | `probe_meta.py` | 2,707 | `0098088018d3621d86b9a5713383b63557ccd751df62e99cf4581750c8c99769` |
| 83 | `probe_out/02a/adapter.json` | 404,283 | `0e900b1be9bbd0535ddc8a13b045c1b154f0e44c988be911e387588a50d12dab` |
| 84 | `probe_out/02a/sources.json` | 415,732 | `2053236f5cf2edbaef43bcb8d22a6c8c18e8e247f2974795963a032bca59cc5f` |
| 85 | `probe_report.py` | 3,265 | `97d4b6babf1277a9334e69dad76c31e8084a5a4bb12db66654445751cf3fa5ba` |
| 86 | `probe_shapes.py` | 595 | `83754258be57f51b9240c2ae88dc3d309eb396b3b6829eca8513ff604d881d1c` |
| 87 | `probe_slots.py` | 1,607 | `34f79e768ce48c1a6463497af23243468a6f49e5913b32f8e5a62d86e9114e3b` |
| 88 | `probe_slots2.py` | 1,508 | `d67445b9e728ac4e5f56ba63feae3e072768eb45153965ac6bcd8a2864468808` |
| 89 | `recipe_probe.py` | 19,825 | `fc038aaf66a2fc1d4eb85441b1d65107867e1dbe2b83697a99a3cc866a34e8e4` |
| 90 | `recipe_sha_test.py` | 1,465 | `f31298b4ed56d247fcdd0720fc58cc6869be8f6ad969d89bdb541f09fd474f1b` |
| 91 | `refit_bc.py` | 66,470 | `e91d2dbdaebd23f710c72bd572f4e2dce1422062615a579ddb26fd0993a4dc76` |
| 92 | `refit_data_probe.py` | 7,791 | `26c64cad42f54b7f36a5993eb297904a9e67e4ce4da6c4ab8434f0b70be85c08` |
| 93 | `refit_env_probe.py` | 1,866 | `4b7f5a3beb3d72860e9a6a434c368c69fcf9422ee243eb16462f17b14cd6c1a3` |
| 94 | `refit_prior_check.py` | 3,512 | `b4f7c6b586a40a9fdb41a9541fe3b5399708d48f91ed46d1b23ca31603f708ba` |
| 95 | `refresh_run.sh` | 428 | `9bddeb2ff93240c39ea24157d30bfdf1446a1073bacf73a6faca9589fb0e6547` |
| 96 | `roots_check.py` | 10,401 | `fd3e3fa1d0f4cd27363ab4c41c8f230ae8fbb5755967ac40633bb1f3af7d0bd3` |
| 97 | `roots_collect.py` | 81,087 | `b16726e29179b76423cb5bf179bd1c0902d276963ef850bf94b63ad9e08b9f7f` |
| 98 | `run_02a.sh` | 9,086 | `4607b65a58c2caec68ef872c95141b9e41fe3ca21ac62fe435771d20bf1797ad` |
| 99 | `run_02bc.sh` | 15,989 | `ae64ff193c4b2cf02e74ea6be555e0a061bc5bd00461dac0cabe2828ef6ff803` |
| 100 | `run_block.py` | 18,054 | `f58349d5783d71082b1b95974a9b7dbeb21a568185410553a0b6d53380755a1a` |
| 101 | `run_diag_teacher.sh` | 11,273 | `0c204995ec1a680bd30deb15f6021bcbbb614b2e399927928636bb7aaa007967` |
| 102 | `run_pilot_formal.sh` | 2,697 | `218601dc342384055572f3c6af6499b1aa63a3ba2e5f281edb61e9db8dcbe889` |
| 103 | `run_pilot_smoke.sh` | 4,441 | `6cb91e4713afafd250db5bd5858c94737fb9ffe23d6f847bbb632709abcefc7a` |
| 104 | `run_strength.sh` | 10,267 | `e54d28eff1cfdea2523f17609430be626c5a2880cb21548ec11f5c6a2063f1f5` |
| 105 | `s1_out/02a/adapter.json` | 404,265 | `75a6e9b1d9ad6470b856dc518fe2233c0db9c05f4c35c4e0274d0dc9da3ef638` |
| 106 | `s1_out/02a/sources.json` | 415,791 | `01d69844ab1554f28eb9accff84f2ebd57eabe8a731d83b50efe03bb7c34e40c` |
| 107 | `s1_out/verify.json` | 387,582 | `d0e6df2855ca8c5dd6b39ce5dc515444aa109741ffd028d7e807eb03577a789d` |
| 108 | `s2_env.sh` | 228 | `29a9e4943ab4b4c7bd6198e92f766e74e75e7e83e90a2765bef1ad7f2794fa0c` |
| 109 | `s2_fix_frames.py` | 841 | `194c503c1407dd0c242f5e607956f0f9608dafae10f8e09c71daa2c5797008a0` |
| 110 | `s2_helpers.py` | 4,798 | `8dd7f1fec6c8a7c85105e2491eb48b3dcd22ee9b0578ebe79762f1dd9193395d` |
| 111 | `s2_mute_check.py` | 1,243 | `90ca6cd9ed16e874f8dd94f0542cf5db634ab02bc7eaee6c2a6945bad2084672` |
| 112 | `s2_probe_d1.py` | 2,049 | `407d9613de410f1ec818853389efb68151a789723cfe323af975ff0b7296f20b` |
| 113 | `s2_report.py` | 5,506 | `afceb3cc9d9b0d054cea193ea426e709080e85cd45787f13da82a42c99389955` |
| 114 | `s3_collect.py` | 3,769 | `6c552db8aa194a3203e490da5db75a5a9e8f89c75bd9442c41dad6e4cab33ae0` |
| 115 | `s3_gen_smoke.py` | 9,281 | `9d91883ae167cacf9e087729cd8eeaf064fc07f4c5d6fdeffd864ecb07639cc3` |
| 116 | `s3_look_boss.py` | 1,315 | `246675bbcd16b29860c41607b66f334089d66ed0691a7284ca82a044b4a1cb3b` |
| 117 | `s3_look_recon.py` | 3,714 | `4de9efa658e171fa3e9163bc6fb8659444bdad4f7204b1f95558b3985112f91a` |
| 118 | `scan_action_divergence.py` | 2,856 | `a6a10409799320115c58f805d344bfbe9ae8943ebd4f93284c9e738e73adea11` |
| 119 | `scan_digest.py` | 1,762 | `082270b614f2d60d2d032067c0f150553e954f98f5f33d832c9ad3c2544352a5` |
| 120 | `scan_events.py` | 73,421 | `5285a37ccc949985fbfe5663f97f759914c9d38028ca168fae7350847bd84e7a` |
| 121 | `scan_peek.py` | 2,102 | `ac6d8c98eff7c4cc3487f94f110d6e39f9688cdcb1ff18bbc32b31fcbf795742` |
| 122 | `shapes_probe.sh` | 1,397 | `b854bdbe5ef599c72e8b461a824d366ac8d7ee618f076eec09bd3cfada33cf08` |
| 123 | `shard_meta_probe.sh` | 1,130 | `26b91038bc9c97a3bc878b1bb5081a746f736af6b26e891935e9e70331038f37` |
| 124 | `source_probe.sh` | 2,662 | `75b4a4a518547af57ccbc6a5614b7386c98ca03bc604884e2af66f2fb483c930` |
| 125 | `splice_diag_teacher.py` | 643 | `a428be6b998df3b6d7831662dfc447e3e604ddf0c8f45b304a2a6cf7f7dff90c` |
| 126 | `splice_pilot.py` | 587 | `f1a9a4afc8b0145fdb3e8cc7dca7e646db7792e200de977f2effcef5eb7df1d3` |
| 127 | `splits_probe.sh` | 1,235 | `2def04a9125e9b6198253e5a9e15b10cb3b249dc892636260f83f81bbcb0eb0d` |
| 128 | `sync_append.py` | 35,054 | `e3b622b68f297bfb046f11364d1ffcf6339f08fb488643713d580d344e0e66e8` |
| 129 | `sync_verify.sh` | 682 | `6778e423da2dbc463e5ccb20dca1976b57e12bdda74e28c4f628b05252fbdff5` |
| 130 | `torch_probe.sh` | 566 | `8c3526b1b178dd3b52b89b69cd89b3c244445eb1407e480f36cb4eb9cba0c8ae` |
| 131 | `verify_block.py` | 14,755 | `309807690416ca834d72883449e82795ce5f473a46e61e4f57026c87237a3e59` |
| 132 | `wrapup.sh` | 1,284 | `4632084ed673a7d7a38ca1571e3927103b52c5469df7671ebc4e1e67f1e65885` |

其中 `_probe*.py` 与本节的追加脚本 `sync_append.py` 是登记动作本身使用的探针/工具，一并入册以保证可复现；`main_verify_out/`、`probe_out/`、`s1_out/`、`pilot_dry/` 下的 JSON 是早期尝试的中间输出。

### 40.4 跳过说明

| 物件 | 处理 | 理由 |
| --- | --- | --- |
| `D1` 语料（`.tmp_v400_d1/**`）、`.tmp_v410_s1c/**`、`.tmp_v410_wp05_round*/**`、`.tmp_v420_data/**` | 不重复登记 | 既有资产，§8.2 红线不动；本轮只按台账 `inputs` 引用其 sha256 |
| 引擎源码与二进制 `third_party/th08-rl/**`（`rl_v3/th08-modern` 等） | 不复制、不入册 | 既有资产；本轮只引用台账记录的引擎 sha256 与 `--rl-*` 参数 |
| `.tmp_v831_slice_01_*/**` | 不重复登记 | 已由 §38 登记，本轮只读 |
| `.tmp_v831_slice_02a/**` | 不重复登记 | 已由 §39 登记（12 条），本轮只读 |
| 各 run 下的 `game/` 引擎数据目录（`th08.dat`/`thbgm.dat` 软链、`th08.cfg`、`score.dat`） | 不逐文件登记 | 引擎运行副本，非本轮交付。静音按 §9 落盘且可核：26 个 `mute.json` 与 26 次启动一一对应，私有 `game/th08.cfg` 在**启动前**与**退出后**两次核验均为四字段全 0（`muted_strict_four_field=true`，两处 sha256 恒为 `cfa292ae10ee…`），`SDL_AUDIODRIVER=dummy`，`problems=[]` |
| `.tmp_v831_followup_02a_gwZ4fA8u/pytest-wsl/**` | 不逐文件登记 | pytest 临时工作目录；其中 23 个 WSL 符号链接在 Windows 侧 `WinError 1920` 无法 stat，无法给出可信 sha256 |
| `.tmp_v831_followup_02a_gwZ4fA8u/03/diag/teacher_arm/{1020,1021,1022,1023}/{decisions.jsonl,prefix.jsonl,mute.json,game/**}` | 不逐文件登记 | 按工单「只按目录身份登记」；每 root 的 `result.json` 已逐件登记，其余并入 `03/diag/` 的 85 文件 / 1,993,537,281 B |
| `.tmp_v831_followup_probe/diag_refit/{artifact_sha256.json,refit_bc.log,t04_self_check/*.json}` | 不逐文件登记 | 工单只要求登记 6 个权重 + 3 条 `loss.jsonl`（已入 §40.2.5）；其余为复算脚手架与日志 |
| `.tmp_v831_followup_probe/*.log`、`*.txt` | 不登记 | 非脚本/报告类扩展名（日志与校验输出） |

### 40.5 与既有清单的交集

本轮 **183** 条 `sha256` 路径与既有 **1,466** 条清单的交集 = **0**：`v8_31_followup`、`test_v8_31_followup`、`pilot_03_roots`、`.tmp_v831_followup`、`configs/v8_31` 五个模式在追加前的 `ASSET_SYNC.md` 与 `ASSET_SYNC.sha256` 里命中次数均为 **0**（脚本内已断言）。

`ASSET_SYNC.md` §1–§39 与 `ASSET_SYNC.sha256` 既有 1,466 行**一字未动**：追加前 md sha256 = `42a1906c1038f07013893b7878bd0a10dd7fb7bbb93ffc98b720b27b015c886c`；追加后重读前 **788,384 B**，sha256 仍为 `42a1906c1038f07013893b7878bd0a10dd7fb7bbb93ffc98b720b27b015c886c`。本条为追加前后逐字节比对的结果，不是推断。

### 40.6 校验结论

1. **`ASSET_SYNC.md`**：追加前 **788,384 B / 4,729 行 / cr=lf=4,729 / 无裸 LF / sha256 `42a1906c1038f07013893b7878bd0a10dd7fb7bbb93ffc98b720b27b015c886c`**；追加后 **820,342 B / 5,019 行 / cr=lf=5,019**（本节以 UTF-8 + CRLF 写入，逐行 `\r\n`）；前缀 788,384 B 重读后 sha256 与追加前相同 ⇒ **既有字节完全未变**。
2. **`ASSET_SYNC.sha256`**：追加前 **1,466 行 / 197,102 B / cr=0（纯 LF，行尾为既有风格）/ sha256 `7431feb07938fd5c698ec2938c8e4c562baf93377a80e77f0374554ece55e4d6`**；追加后 **1,649 行 / 217,846 B / cr=0**，每行 `<sha256>␠␠<path>`、路径为仓库相对 POSIX 形式，well-formed **1,649/1,649**、bad=0。
3. **本节新增 183 条 = 183/183 OK**，新增 FAILED **0**（交付 4 + 关键文件 38 + 离线复拟合 9 + 脚本 132；目录不入 sha256）。
4. **全表 `sha256sum -c`（1,649 条）= 1,575 OK / 74 FAILED，退出码 1**：FAILED **恰好 74**，与 §39.4 第 2 条**完全相同**（63 条 `Document/Update/Alpha` 漂移 + 11 条 `tools/rl/v8*.py` 漂移，均为本轮之前既有的工作树状态），**一条不增不减、新增 0 FAILED**（`comm` 双向比对：`new_failed=0`、`gone_failed=0`）；全表 `No such file` = 0。
5. 本节只读既有产物、只追加两个登记文件：未改 `touhou_rl/**`、`configs/**`、`Document/**`、`.tmp_*` 既有内容，未删除、未 `git` 写入、**未启动引擎**。

## 41 Easy 难度评测（V831-EASY-01，2026-11-27，训练端本机实测）

对当前训练成果做的一次 **Easy 难度闭环评测**：学生用已注册的 `tools/rl/v400_student_eval.py`
（`--obs v3 --arch v3`，未加 SHOOT 脉冲的 v4.1 标尺），教师参照用 `tools/rl/v420_eval_shoot.py --replay-tape`
（有脉冲，但它是一条 Normal 录像，shard 不记录录制难度）。10 个 episode / 10 次引擎启动 / 21,057 帧，
全部跑在 `game-muted` 私有副本上（四点静音字节逐 episode 前后 10/10 为 0，`SDL_AUDIODRIVER=dummy`，
源目录树 sha 前后不变）。

### 41.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v831_easy_01/` | 94 | 497179348 | 10 个 episode 的原始输出 + 共享 `game-muted` 私有副本 |

### 41.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `.tmp_v831_easy_01/summary.json` | 37,436 | `a684e3877c053c005b5c6a79f7b17e933aebd5a0bf4097a48b68a4f74bf1737e` |
| 2 | `.tmp_v831_easy_01/EASY_EVAL.md` | 17,024 | `efa65330d585883740f0bfcee1dc7e73c841e5da2d3e5eb97cb856aba3d0e670` |
| 3 | `.tmp_v831_easy_01/deviations.json` | 5,196 | `b2cd81c6c02f80fc3b4325e0eeb267eb8b5abf6e9971f980bd5af4a151b8e8f7` |
| 4 | `.tmp_v831_easy_01/run_easy_eval.log` | 4,205 | `1a05abc8ee1bd9c66d4ecaeff903037a70a937f722e86313ed4f435f68782bb2` |
| 5 | `.tmp_v831_easy_01/cost.txt` | 49 | `ea9abe75984a15bf5c6fbe98ae143ae5af17cb322cd7491ec28eb00cc1072d03` |
| 6 | `.tmp_v831_easy_01/cfg_inspect.json` | 787 | `9283c0a39d9b5517b2c1a3dfad5c09f608c34073bcaf46d6b474e3936a18b97c` |
| 7 | `.tmp_v831_easy_01/cfg_diff.json` | 2,439 | `8f0ceac5c6faa37208a0ea4c18ecf9030b6fa49a85ab15208cceb2672567f758` |
| 8 | `.tmp_v831_easy_01/audio_preflight.json` | 5,854 | `548cd7945d38ed882617b9f8e0e6e39844bd305c8f5621430bbb68f874b65624` |
| 9 | `.tmp_v831_easy_01/ep01/meta.json` | 682 | `c561c48be9ff506b15f5e92e6aa6406f1a7a333f8b1504747064ad6694a8f4bd` |
| 10 | `.tmp_v831_easy_01/ep01/summary.json` | 5,733 | `f64e222d0cef19a282a3f4d89da6e3b557c00628ecc81b544ca836d6def4f2e1` |
| 11 | `.tmp_v831_easy_01/ep01/episodes.jsonl` | 1,983 | `90e38ef0230ff14d0e9c5ec5189b158e61a0134d5c9f15ff3a8462e896351ed0` |
| 12 | `.tmp_v831_easy_01/ep01/post_audio.json` | 222 | `d512d03cc48320ef70c91392efb1cbb93ab0622f48d1f574dd665cdaa534d78a` |
| 13 | `.tmp_v831_easy_01/ep02/meta.json` | 684 | `33329a7f7d1f52c9cf394028e7f57594bc33ed4f081431b03d863d1b28534169` |
| 14 | `.tmp_v831_easy_01/ep02/summary.json` | 5,739 | `910832ac89694ed5873a10632f0361590aa94fa485b7a1cf4fe369f7834731b3` |
| 15 | `.tmp_v831_easy_01/ep02/episodes.jsonl` | 1,985 | `17d6395cca3eb386de38e7719d6b7ef9372306fecaccbc94db651766971a40e2` |
| 16 | `.tmp_v831_easy_01/ep02/post_audio.json` | 222 | `d512d03cc48320ef70c91392efb1cbb93ab0622f48d1f574dd665cdaa534d78a` |
| 17 | `.tmp_v831_easy_01/ep03/meta.json` | 683 | `5b072d5720317c39c65f1d97219022b24f1060af44b7fd00ca0c7e3ab918a6f2` |
| 18 | `.tmp_v831_easy_01/ep03/summary.json` | 5,715 | `2cf98053bb7105886fa444afaa58a068b2e02344f2ebefe6536d360bf0d4484c` |
| 19 | `.tmp_v831_easy_01/ep03/episodes.jsonl` | 1,970 | `c75aecd671274e2af0f9b74db4263bc0a243d12fde974b2774e684a5a33e120f` |
| 20 | `.tmp_v831_easy_01/ep03/post_audio.json` | 222 | `d512d03cc48320ef70c91392efb1cbb93ab0622f48d1f574dd665cdaa534d78a` |
| 21 | `.tmp_v831_easy_01/ep04/meta.json` | 685 | `6c72ca2a34448268c6dcbb49be83b1067f4a12dcc7db6b44c520a74955041435` |
| 22 | `.tmp_v831_easy_01/ep04/summary.json` | 5,721 | `1a40e6098350cf0911670d31a731b2dbbbe6a7f7812c68cb6f0c7e6a872dfdc8` |
| 23 | `.tmp_v831_easy_01/ep04/episodes.jsonl` | 1,972 | `3724ce37b48ed5158b2deade1522a64aae6ee54ac2283b0b9968c7bc98c9a830` |
| 24 | `.tmp_v831_easy_01/ep04/post_audio.json` | 222 | `d512d03cc48320ef70c91392efb1cbb93ab0622f48d1f574dd665cdaa534d78a` |
| 25 | `.tmp_v831_easy_01/ep05/meta.json` | 707 | `de909a2fc0cf636204053c2c44d981a8a6fbf997b537989a1a52a72922b31317` |
| 26 | `.tmp_v831_easy_01/ep05/summary.json` | 5,734 | `7fa055ccb9cab4dd09d2fa8e8d00d4a1c47011fed08adce9b33afdb970551f57` |
| 27 | `.tmp_v831_easy_01/ep05/episodes.jsonl` | 2,007 | `e9f33b6e2af5b451264b3a99df649f54733c6fcf3cf3b352affd1469ea1a9b47` |
| 28 | `.tmp_v831_easy_01/ep05/post_audio.json` | 222 | `d512d03cc48320ef70c91392efb1cbb93ab0622f48d1f574dd665cdaa534d78a` |
| 29 | `.tmp_v831_easy_01/ep06/meta.json` | 709 | `4c422daab874393a95d12300a3195a4f86da865cab07197f691804526c7bd942` |
| 30 | `.tmp_v831_easy_01/ep06/summary.json` | 5,740 | `c3bbd0546eafcb6edce4d106f631df90c0bc1c80b96573525aa9a329cf1e214d` |
| 31 | `.tmp_v831_easy_01/ep06/episodes.jsonl` | 2,008 | `d39460a4286cc1ff017f32d7520beb632af7759ff2f24ac55ced82f1a4e21736` |
| 32 | `.tmp_v831_easy_01/ep06/post_audio.json` | 222 | `d512d03cc48320ef70c91392efb1cbb93ab0622f48d1f574dd665cdaa534d78a` |
| 33 | `.tmp_v831_easy_01/ep07/meta.json` | 702 | `d79854f5fb9b40143c7ab1e64984f99a1f8077891ec075c5c814818dfa41db7a` |
| 34 | `.tmp_v831_easy_01/ep07/summary.json` | 5,733 | `479d4857ec06a35f33cedf170326b318a39cb56a6e2dbb34306209287b3d49ae` |
| 35 | `.tmp_v831_easy_01/ep07/episodes.jsonl` | 2,001 | `0f02d2ee82b61d4b830bacc8f729dc1e70123de24aaade00624ea14bd646df93` |
| 36 | `.tmp_v831_easy_01/ep07/post_audio.json` | 222 | `d512d03cc48320ef70c91392efb1cbb93ab0622f48d1f574dd665cdaa534d78a` |
| 37 | `.tmp_v831_easy_01/ep08/meta.json` | 684 | `935b70823e14626c612ecd56cdf32e8ca704ec82ac7a7c72c6f7b9a923235b50` |
| 38 | `.tmp_v831_easy_01/ep08/summary.json` | 5,721 | `416e82c5baa5e8c130fc76300493490ae07ad9dced9b62371ab22cd0ab1e28f3` |
| 39 | `.tmp_v831_easy_01/ep08/episodes.jsonl` | 1,979 | `a6ad6efe3b393b8c9fe6580e059af442b2bbc7f926a03192902bba860405aad0` |
| 40 | `.tmp_v831_easy_01/ep08/post_audio.json` | 222 | `d512d03cc48320ef70c91392efb1cbb93ab0622f48d1f574dd665cdaa534d78a` |
| 41 | `.tmp_v831_easy_01/ep09/meta.json` | 709 | `0a426cc0e9713293cd6e13686ff1cc42fc3b0dcb6f0cff4f5b565ef1a90027d2` |
| 42 | `.tmp_v831_easy_01/ep09/summary.json` | 5,722 | `2e7cd5e360e6e8bf2d6106f8a531f734c431ac1df7a6f2447d7d897b2a9c0cf1` |
| 43 | `.tmp_v831_easy_01/ep09/episodes.jsonl` | 2,003 | `733d114e3c5384464e5c578f23342e2fe8dfd3c50370e8fb42695f932eb4f671` |
| 44 | `.tmp_v831_easy_01/ep09/post_audio.json` | 222 | `d512d03cc48320ef70c91392efb1cbb93ab0622f48d1f574dd665cdaa534d78a` |
| 45 | `.tmp_v831_easy_01/ep10/meta.json` | 643 | `b561cbbb38b2bcdb358de48a525ee378121c0357e141bf5aa8710f85712b08e9` |
| 46 | `.tmp_v831_easy_01/ep10/tape.json` | 3,112 | `53a3eda3e0b39e5d525f1e14fb979e6ee991282e0f5e972243f668ae0dc95302` |
| 47 | `.tmp_v831_easy_01/ep10/post_audio.json` | 222 | `d512d03cc48320ef70c91392efb1cbb93ab0622f48d1f574dd665cdaa534d78a` |
| 48 | `.tmp_v831_followup_probe/run_easy_eval.sh` | 11,603 | `975bfebed6bc282a2ec08e915949ac4fe92c7dd54528035b54a57d531736ff53` |
| 49 | `.tmp_v831_followup_probe/easy_aggregate.py` | 15,165 | `adfcf07b4bcf1c9263d2ccfe9aa212e47c4d0d844fdd62c16e3387d4fb0b5440` |
| 50 | `.tmp_v831_followup_probe/easy_cfg_diff.py` | 2,265 | `7250deaca1b443cbb2c65ea6c7518802c4b08dedc103296b3413de5858e87f82` |
| 51 | `.tmp_v831_followup_probe/easy_audio_evidence.py` | 6,403 | `d7c74c7dde9127591a64b03715f7e374417d0d10b1aaca65981b17db8fc22f7f` |
| 52 | `.tmp_v831_followup_probe/easy_ckpt_check.py` | 2,465 | `99f97db9fbeda8d6e4f1ba0a1060ca49eabb496041e53f4144a8ff4d57b87d3f` |

### 41.3 本轮结论（只作描述，不构成能力门；单 episode/格、无 CI）

- **学生**（4 个检查点 × Easy × seed 12/1000，`--stop-on-death`）：首次死亡帧 **1,458–1,570**
  （约 729–785 决策，约为 13,195 帧整面的 **11–12%**），四个检查点之间只差约 110 帧，
  动作分布 dominant 78.6–97.1%（`pilot_final` 97.1% 是「不动」cand 0）。
- **训练后比 seed-0 初始权重还差 112 帧**：`pilot_final` 1,458 帧 vs `pilot_init` 1,570 帧；
  但 `pilot_final` 分数 21,577 vs 2,569（8.4×）——初始权重是「贴右下角斜射」的角落策略，
  训练后变成「站着不动」，分高但更早被弹。
- **Easy ≈ 2× Normal**（同检查点同 seed）：`pilot_final` 1,458 vs 731（1.99×），
  `refit_B_final` 1,468 vs 776（1.89×）；两个难度都活不过 Stage 1 的前 12%。
- **教师参照**：Normal 录像在 Easy 上 9,066 帧、3 次死亡、`game_over`、117,631 分、`stage_clear=false`
  （同一条磁带在 Normal 上 13,195 帧过关）；它是录音不是策略，且录制难度未登记，**不能当 Easy 上限**。

### 41.4 成本与限制

- 引擎启动 **10** 次、总帧 **21057**、
  墙钟 **240.658** s；单 episode 17.6–22.1 s，磁带 77.8 s。
- 仪器不记录 `power`/`graze`/`wave`（学生记录里没有这些字段），只报 stage + 帧号 + score + lives/bombs。
- 静音口径是「无出声」：引擎仍会打开样本表文件（`se_*.wav`/`.mid`），但不播放；
  `SDL_AUDIODRIVER=dummy` 与四点字节 0 是本轮的实际证据。

## 42 V831-ROADMAP-01 阶段 1 第一轮（V832-NAT01 / DATA01 / S20 / RUN01，2026-11-27，训练端本机实测）

按 `Document/Update/Alpha/v8.31.0/LONG_TERM_DEVELOPMENT.md` 的第一轮：离线取证两条（自然开局失败定位、T04 标签审计）、
学习器复拟合对照一条（`executed` vs `chosen`）、自然开局三臂真机对照一批（9 次启动）。
全部引擎运行使用 `rl_v3` `691bfd77…` 与私有静音目录 `.tmp_v832_nat01_runs/game-muted`（四点字节 18/18 次为 0）。

### 42.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_stage0/` | 5 | 6,946 | 身份探针与跨 seed 像素对照 |
| `.tmp_v832_nat01/` | 9 | 508,541 | 自然开局离线取证 + 冻结任务卡 S1N-MB1 |
| `.tmp_v832_data01/` | 6 | 159,682 | T04 两帧语料标签/历史审计 |
| `.tmp_v832_s20_refit/` | 43 | 88,444,728 | `executed` vs `chosen` 复拟合（含 8 份权重） |
| `.tmp_v832_nat01_runs/` | 67 | 16,882,884 | 三臂原始输出、账本、静音证据、复核脚本 |
| `.tmp_v832_run01/` | 486 | 19,434,087 | 新增单测的 pytest basetemp 与基线 |

### 42.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `tools/rl/v400_student_eval.py` | 70,317 | `1ae7ccde719a8ea1e26444892481662ee4847f34d698d802ade2448ed367f985` |
| 2 | `tests/test_v8_32_natural.py` | 32,476 | `12d4f597b09b4421212a22f2dfb4d3b454e7c95afa10d3a539b3a111412e9e4a` |
| 3 | `.tmp_v832_nat01/nat01.json` | 355,908 | `2222d7c245502d1efb042f92d3cb7787e56b5c009d906a45d73252c388b1f5be` |
| 4 | `.tmp_v832_nat01/nat01.md` | 16,236 | `3d612350b1dcd516b9a672db25faf4b7d98065cde9b88543caca8b40b6e19aa1` |
| 5 | `.tmp_v832_nat01/verify_nat01.py` | 28,220 | `c5d4813cce6dc2fd4a73cc4017c67aaa750376f190dacc1760cb2b56768065cd` |
| 6 | `.tmp_v832_nat01/build_nat01.py` | 72,228 | `c274bc669a84ae4ef4f6e0193c0cdb8af8f362abac54c9155ebc11e897210c5a` |
| 7 | `.tmp_v832_nat01/task_card.json` | 8,725 | `7eba7c5e932579e836b1a809b9776b0947eddaff13b2bb789e35cfcf34de8dcc` |
| 8 | `.tmp_v832_nat01/task_card.md` | 3,508 | `23ff4beed179a5150e759f7c5934c15e49a1fc0604383467246b895e896b0ab8` |
| 9 | `.tmp_v832_nat01/make_card.py` | 14,961 | `546b32b3c05f2dfeec2966ecead31c416b2b7ea76dcccfac7ce16ffa1bb3f007` |
| 10 | `.tmp_v832_data01/data01.json` | 82,570 | `06fcdc93fc6ff3a60e63bbd1e280abdcaca62e3f7ad430152327c184a17dfe6a` |
| 11 | `.tmp_v832_data01/data01.md` | 12,402 | `c08620c703a151677342be077d1b2e19c51d44f15f5510fd612c252d317d3280` |
| 12 | `.tmp_v832_data01/verify_data01.py` | 13,883 | `95f9cad56d31796b9e1fa580f27c389c41db13f98d0cfab05170e1cc2ae46035` |
| 13 | `.tmp_v832_s20_refit/refit_executed.json` | 52,545 | `9ed491240644e03236dd8c589aaa2d046455daf377abbfc3f2e90af4a52feea9` |
| 14 | `.tmp_v832_s20_refit/refit_executed.md` | 9,073 | `59a8092c50106427a97c96809ebaee41dc109a8c2fc5d43cdc4b40602d98a3c0` |
| 15 | `.tmp_v832_s20_refit/verify_refit.py` | 20,422 | `bfc442c3dee4af2ba74f9b650fb99ba4a2c9b02c092fea0330c0b576ecfafa31` |
| 16 | `.tmp_v832_nat01_runs/results.json` | 29,340 | `e4787d00aa42dfabe19bfbfd0fab28c15b49b941fad46651850d1cf2560d2f7a` |
| 17 | `.tmp_v832_nat01_runs/ledger.jsonl` | 7,386 | `47a3f22365f7bb4f3fcf1ea2e1851d52c4a3592265b94e4aee434ed0c9263b35` |
| 18 | `.tmp_v832_nat01_runs/verify_runs.py` | 12,984 | `a9d0b7bdff7a096b4b02b514108d44f5a6cab22bca2d55d4ad28d697bd4c81a4` |
| 19 | `.tmp_v832_nat01_runs/RUN_NOTES.md` | 6,030 | `94dc3d7f1904f18b90250269bd2c48ae531acdf0896b395fcc3f0545416a2a67` |
| 20 | `.tmp_v832_stage0/seed_pixel_probe.json` | 1,643 | `28147de67b98ceb3d47a8894175c2fb0a80b7b187527031a5245f47e2ea3e6be` |

### 42.3 本轮结论（只作描述，不构成能力门）

- **自然开局任务成立**：同一引擎同一静音目录下，Tier A（首中 boss `boss_life` 6000→0）教师 **3/3**（帧 3,684 / 3,740 / 3,652），
  当前学生 0/3（731 帧死亡）、恒动作 `cand 12` 0/3（746 帧死亡）；Tier B（帧 2,936 存活）同为 3/3 vs 0/3 vs 0/3。
- **三个 dev 种子不是三个独立实例**：学生与恒动作臂三 seed 逐位同结果；离线教师 20 根在帧 400–2400 的几何完全相同，
  首次分叉 2,580–2,584；`reset_hash` 随 seed 变但可玩几何不变。
- **数据可作 BC 标签（`executed` 口径）**：`history`/`act_hist` 100% 等于已执行序列，无未来帧；`chosen` 只可作验证目标。
- **学习器近恒定不是标签错位**：换 `executed` 后 dominant 0.9087→0.8864；类别平衡到 0.2734/17 类但 top-1 低于多数类基线。

### 42.4 成本与限制

- 引擎启动 **9** 次、总帧 **17,931**、启动墙钟 **383.808 s**；新增磁盘 **125,436,868 B ≈ 0.12 GiB**。
- 静音：9 次启动前后 18 次四点读数全 `0 0 0 0`，`audio_refused=0`；源目录树指纹前后不变。
- 限制：3 个 seed 非独立实例；Tier A 判分读数是 `enemies.boss_count/boss_life`，尚非 `HAZARDS` 正式口径；
  教师臂 4,500 帧为 `max_frames` 截断，不等于过关；历史 20 根教师轨迹跑在 `rl_hazard2` `f172399f…`，与本轮 `rl_v3` 不合并计数。

## 43 V831-ROADMAP-01 阶段 1 第二轮（HAZ01 / INDEP01 / READOUT01，2026-11-27，训练端离线 + GPU）

第二轮全部离线：把 Tier A 判分升级为行身份口径、量化独立根边界、并把"近恒定"定位到可修环节。**0 次引擎启动。**

### 43.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_haz01/` | 4 | 130,298 | 行身份判分器、三根教师包重算与复核 |
| `.tmp_v832_indep01/` | 6 | 289,790 | 独立根边界、独立根规格与 S1N-MB1 独立性附录 |
| `.tmp_v832_readout01/` | 10 | 132,752,154 | 学习器预算/采样/温度诊断（含 10 次拟合权重） |

### 43.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `.tmp_v832_haz01/score_hazards.py` | 38,936 | `d8cbff74372b6b21`（完整 64 位见 `ASSET_SYNC.sha256`） |
| 2 | `.tmp_v832_haz01/haz01.json` | 73,046 | `c86679d47de2942e` |
| 3 | `.tmp_v832_haz01/haz01.md` | 4,963 | `80719798bac47865` |
| 4 | `.tmp_v832_haz01/verify_haz01.py` | 13,353 | `303e36f60a04ee83` |
| 5 | `.tmp_v832_indep01/indep01.json` | 204,857 | `471fd8ce154c00ac` |
| 6 | `.tmp_v832_indep01/indep01.md` | 9,144 | `e7b1d5954bb46575` |
| 7 | `.tmp_v832_indep01/verify_indep01.py` | 16,861 | `df662140a8778dba` |
| 8 | `.tmp_v832_indep01/S1N-MB1_addendum.json` | 4,759 | content `47ddb81bfb34ff62973e77213369f009bc96567d60f4d2b049dd43b5d57be8cc` |
| 9 | `.tmp_v832_indep01/make_addendum.py` | 10,321 | `8fa9056dc1f82d25` |
| 10 | `.tmp_v832_readout01/readout01.py` | 100,141 | `a979b44bb720fac7` |
| 11 | `.tmp_v832_readout01/readout01.json` | 211,658 | `a642559518e227c0` |
| 12 | `.tmp_v832_readout01/readout01.md` | 9,798 | `072179de62caedca` |
| 13 | `.tmp_v832_readout01/verify_readout01.py` | 16,660 | `135adb4d2a60bea8` |
| 14 | `.tmp_v832_nat01_runs/T/seed{1000,1001,1002}/decisions.jsonl` | 4,698,111 等 | `35cc55f7…` / `a4847f2e…` / `5aba91cd…` |

### 43.3 本轮结论（只作描述，不构成能力门）

- **Tier A 行身份判分与汇总口径一致**：`boss_life` 6000→0 首见帧 3,684 / 3,740 / 3,652（Δ=0）；1,125 决策/seed 中 `truncated=0`、`available=false=0`。
  关键口径：`ID_MASK=0x00000003` + 出生断点；字面"任意 flags1 变化即新代"会把同一条 boss 命切成 14/13/13 段而**Tier A 不可达**（反例已归档）。
- **独立根边界**：共享前缀到帧 **2,576**，首次分叉 2,580（教师）/2,632（语料）；`n_indep` 在 cap≤2,576 为 **1**，2,580→4，2,632→18，2,640→20；
  入口 ≥2,632 的 1,200 帧窗有 20 条轨迹但只有 13 个单帧入口状态。根身份只能用 `reset_hash`。
- **学生臂有效根数 = 1**：三 seed 在帧 0–730 逐位相同；S1N-MB1 可作为"任务存在"证据，**不可**作为"多独立根"证据（另有独立性附录，未改冻结卡）。
- **跨引擎等价**：`rl_v3` 与 `rl_hazard2` 的 seed 1000 教师轨迹在 1,125 个决策上 `(px,py)` 逐点相同。
- **学习器可修**：2,000 步后，未参与训练的供体 seed11 上 top-1 0.5933（顺序）/0.5027（类别平衡），基线 0.3736；dominant 0.3638/0.2057、17 类。
  → 回执 04 把近恒定归因于"采样/读出"不完整，主因是预算；拟合根指标是样本内数字，今后不作判据。

### 43.4 成本与限制

- 引擎启动 **0** 次；GPU 10 次拟合 / **229.2 s**；新增磁盘 **133,172,242 B ≈ 0.124 GiB**。
- 限制：`flags1` 位名是转述；`life=0` 只到 4 帧采样精度；`flags1` 无实例 id 位；换难度独立根数未测；
  离线 argmax 不等于闭环能力。

## 44 V831-ROADMAP-01 阶段 1 第三轮（S2A / TAKEOVER，2026-11-27，训练端 GPU + 引擎）

第三轮：用官方训练器在 D1 上训 2,000 步 BC 做闭环前后对照（负结果），并交付/验证条件接管 runner。

### 44.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2a/` | 39 | 22,998,664 | 2,000 步与 1 步 checkpoint、训练指标、6 次闭环评测、账本、静音证据 |
| `.tmp_v832_takeover/` | 42 | 6,167,832 | 接管 runner 的离线校验与 4 次真机验证 |

### 44.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `tools/rl/v8_32_takeover.py` | 68,361 | `4894363e30e47d8e…`（完整值见 `ASSET_SYNC.sha256`） |
| 2 | `tests/test_v8_32_takeover.py` | 38,022 | `93cc1b07b1be20b0…` |
| 3 | `.tmp_v832_s2a/ckpt_2000.pt` | 10,482,699 | `19808e88cfe35902…`（best-by-val step 900） |
| 4 | `.tmp_v832_s2a/ckpt_init.pt` | 10,480,331 | `5cae894001e736a5…` |
| 5 | `.tmp_v832_s2a/train_2000.json` | 24,377 | `48df064002f66d1a…` |
| 6 | `.tmp_v832_s2a/results.json` | 32,586 | `d3341e99b441472c…` |
| 7 | `.tmp_v832_s2a/s2a.md` | 6,683 | `6072721f131fbb61…` |
| 8 | `.tmp_v832_s2a/verify_s2a.py` | 15,068 | `9128ac073d669b18…` |
| 9 | `.tmp_v832_takeover/verify/results.json` | 14,792 | `ed33d7f99c445fdd…` |
| 10 | `.tmp_v832_takeover/verify/TAKEOVER_VERIFY.md` | 3,739 | `7997857133bd9ee2…` |
| 11 | `.tmp_v832_takeover/verify/verify_takeover.py` | 10,744 | `d4dcb4179904fa7d…` |
| 12 | `.tmp_v832_takeover/verify/ledger.jsonl` | 3,429 | `ec365c8ee22340e9…` |

### 44.3 本轮结论（只作描述，不构成能力门）

- **2,000 步 BC 闭环负结果**：D1 上 2,000 步（64,000 样本 = 0.19 epoch，best-by-val step 900）与"初始化+1 步"在 S1N-MB1 上**都是 705 帧首死、353/353 `cand 0`、score 1,450**，Δ=0。
  离线 test top-1 0.5132→0.6069、非 idle 预测 0%→18.1%，但**闭环一次未触发** → 离线 argmax 不等于闭环；机制是 idle 自强化。
- **条件接管打通**：`v8_32_takeover.py` 在 3 条磁带上各核验 1,308 个 prefix 槽哈希后交还控制，入口哈希 `e2a84b01…`/`9edbdd05…`/`b8fdb779…`
  与磁带 `hash_post` 逐个相等且**互异**；tape 臂 3/3 跑满 4,500 帧 0 死亡，constant 臂入口后 630 帧死亡。
- **既有工具缺陷（登记未改）**：`v8_31_followup.py intervene` 在入口哈希不符时另一臂照样启动，不满足 fail-closed。

### 44.4 成本与限制

- 引擎启动 **10** 次（s2a 6 + takeover 4）、总帧 **20,992**、墙钟 **180.169 s**；GPU 训练 306.4 s；新增磁盘 **29,166,496 B ≈ 0.027 GiB**。
- 静音：20/20 读数全 `0 0 0 0`；账本 20 行（10 reserve + 10 settle），无未结。
- 限制：checkpoint 臂未实现（缺 5 件）；`S1N-MB2` 第 3 条前置验证只做 1/3 根；两臂 closed-loop 逐位相同说明 seed 在该窗口无方差。

## 45 V831-ROADMAP-01 阶段 1 第四轮（S2B / S2C / TAKEOVER-2，2026-11-27，训练端 GPU + 引擎）

第四轮：把 T04 规模训练的两个策略转成评测格式并做自然开局闭环评测（**首个正向学习收益**），同时补齐 `S1N-MB2` 的第 3 条前置验证。

### 45.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2b/` | 33 | 118,815,368 | T04 两策略 checkpoint（评测格式）、训练指标、兼容性自检 |
| `.tmp_v832_s2c/` | 41 | 3,868,014 | 自然开局六次闭环评测、账本、静音证据 |
| `.tmp_v832_takeover/verify2/` | 20 | 326,397 | `S1N-MB2` 退化臂（seed 11/12）与三条前置验证汇总 |

### 45.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `.tmp_v832_s2b/ckpt_t04_balanced.pt` | 10,470,555 | `46e536ecd35b9e39…` |
| 2 | `.tmp_v832_s2b/ckpt_t04_sequential.pt` | 10,470,879 | `c961b21ba3208771…` |
| 3 | `.tmp_v832_s2b/train_s2b.json` | 18,679 | `47dbe2fabb7e41c3…` |
| 4 | `.tmp_v832_s2b/compat_check.json` | 12,047 | `ce1a6db8c38ebfa0…` |
| 5 | `.tmp_v832_s2b/s2b.md` | 4,194 | `3359722d4166be14…` |
| 6 | `.tmp_v832_s2b/verify_s2b.py` | 15,561 | `eb4a6d3519d62c2f…` |
| 7 | `.tmp_v832_s2c/results.json` | 20,536 | `f7e5699fe6e2ac06…` |
| 8 | `.tmp_v832_s2c/s2c.md` | 4,373 | `1e7800ab1cba8d98…` |
| 9 | `.tmp_v832_s2c/verify_s2c.py` | 15,060 | `05670b5973189e1e…` |
| 10 | `.tmp_v832_s2c/ledger.jsonl` | 4,653 | `f647a2edad728ded…` |
| 11 | `.tmp_v832_takeover/verify2/results.json` | 36,931 | `f41b0e33aad7042a…` |
| 12 | `.tmp_v832_takeover/verify2/S1N-MB2_PRECHECK.md` | 2,750 | `e863aa3b19ac931c…` |
| 13 | `.tmp_v832_takeover/verify2/verify_takeover2.py` | 15,709 | `1d5d8e263a2066a3…` |

### 45.3 本轮结论（只作描述，不构成能力门）

- **首个正向闭环学习收益**：T04 顺序采样 2,000 步（9.7 epoch）在自然开局 S1N-MB1 上 1,504 / 1,808 / 2,224 帧（均值 **1,845.3**），
  相对 `pilot_final` 731 为 **+1,114 帧（2.52×）**；类别平衡臂 987.3（+256）。**但两臂 Tier B 仍 0/3**（最高 2,224 < 2,936）。
- **预算必须按语料规模换算**：同样 64,000 样本 = T04 的 9.7 epoch（学到策略）vs D1 的 0.19 epoch（idle 相）。
- **策略一活，seed 方差就回来**：坍缩臂三 seed 同值（705×3），T04 顺序臂 1,504/1,808/2,224、平衡臂 824/980/1,158。
- **`S1N-MB2` 三验全过**（前缀 3/3、入口 hash 3/3、退化臂 0/3 完成），但入口**玩家状态三根完全相同**，独立性只来自敌机/RNG，口径已更正。

### 45.4 成本与限制

- 引擎启动 **8** 次（s2c 6 + verify2 2）、总帧 **15,216**、墙钟 **173.521 s**；GPU 50.8 s；新增磁盘 **123,009,779 B ≈ 0.115 GiB**。
- 静音 16/16 读数全 `0 0 0 0`；账本 16 行（8 reserve + 8 settle），无未结。
- 限制：单训练种子、单语料片（T04 seed10）、3 个 dev 根；Tier A/B 未提升；checkpoint 臂未实现；500 字以内的"2.52×"只指生存帧，不指任务完成。

## 46 V831-ROADMAP-01 阶段 1 第五轮（S2D / CHECKPOINT-ARM / S2C2 / S1N-MB2，2026-11-27，训练端 GPU + 引擎）

第五轮：冻结 `S1N-MB2` 条件接管卡、实现并真机跑通 checkpoint 臂、把 T04 配方加长到 8k/16k 步做离线+闭环饱和测试。

### 46.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2d/` | 53 | 184,568,311 | 三个加长训练 checkpoint、离线指标、兼容性自检 |
| `.tmp_v832_s2c2/` | 73 | 3,375,957 | 双层真机评测（A 批 checkpoint 臂 3 根 + B 批自然开局 3 根） |

### 46.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `tools/rl/v8_32_takeover.py` | 112,013 | `0b3ab4ebc5668ec4…`（2,455 行，含 checkpoint 臂） |
| 2 | `tests/test_v8_32_takeover.py` | 80,263 | `6d9b5285dbdefc28…`（1,732 行，48 项） |
| 3 | `.tmp_v832_takeover/task_card_mb2.json` | 7,054 | content `2eee50461abd0710…cd2423c`（首冻 `70032a67…`） |
| 4 | `.tmp_v832_takeover/task_card_mb2.md` | 2,183 | `ab03aadc7cadaae1…` |
| 5 | `.tmp_v832_takeover/make_task_card_mb2.py` | 13,750 | `47a0e265697eddf8…` |
| 6 | `.tmp_v832_s2d/ckpt_l1.pt` | 10,468,615 | `d8f8912807a4bff5…`（8k 顺序） |
| 7 | `.tmp_v832_s2d/ckpt_l2.pt` | 10,468,615 | `bba0d3438151a472…`（8k 平衡） |
| 8 | `.tmp_v832_s2d/ckpt_l3.pt` | 10,468,615 | `312410d09e483619…`（16k 顺序） |
| 9 | `.tmp_v832_s2d/train_s2d.json` | 37,678 | `8576f5f7ac8666d3…` |
| 10 | `.tmp_v832_s2d/compat_s2d.json` | 17,353 | `8dc7972f9050ede5…` |
| 11 | `.tmp_v832_s2d/s2d.md` | 3,704 | `8873bf8f5ed7bab9…` |
| 12 | `.tmp_v832_s2d/verify_s2d.py` | 21,613 | `91f34ba19d36afd0…`（51/51） |
| 13 | `.tmp_v832_s2c2/results.json` | 10,876 | `c93e65a7eb7ca74c…` |
| 14 | `.tmp_v832_s2c2/s2c2.md` | 4,826 | `a85155c3eb653d3d…` |
| 15 | `.tmp_v832_s2c2/verify_s2c2.py` | 11,980 | `802081f08fcead53…`（51/51） |
| 16 | `.tmp_v832_s2c2/ledger.jsonl` | 5,696 | `880df7a04493992a…` |

### 46.3 本轮结论（只作描述，不构成能力门）

- **`S1N-MB2` 冻结**：3 个入口根（hash `e2a84b01…`/`9edbdd05…`/`b8fdb779…`），玩家状态三根相同，独立性只来自敌机/RNG。
- **checkpoint 臂真机通过**：策略从入口 2,632 起存活 **1,868 / 994 / 990** 帧（恒动作 630 / 674 / 780），`policy_calls` 934/497/495、SHM 像素源正常。
- **Tier A 不可判**（缺逐决策 `enemies`），**Tier B 无区分力**（磁带/恒动作/checkpoint 都是 3/3 在帧 2,936 存活）。
- **加长训练饱和**：donor top-1 0.5933→0.6211(8k)→0.6359(16k)，dominant 不再下降，16k 的 fit=1.0（记忆）；→ 下一瓶颈是 episode 多样性（当前只用 1 条 episode）。
- **闭环同向同幅**：8k 自然开局均值 1,917.7 vs 2,000 步 1,845.3（+3.9%，逐 seed −47/+1,087/−823，无 CI，不构成效应）。

### 46.4 成本与限制

- 引擎启动 **6** 次（+1 次预检拒绝 0 帧）、总帧 **17,501**、A 批墙钟 **107.238 s**（B 批未单列，ledger 事后重建）；GPU 训练 394.45 s；新增磁盘 **187,944,268 B ≈ 0.175 GiB**。
- 静音 14/14 读数全 `0 0 0 0`；账本 12 行（6 reserve + 6 settle），无未结。
- 限制：Tier A 无法判分；Tier B 不具区分力；只有 3 个条件接管根；训练仍只用单条 episode；B 批 wall 缺失。

## 47 V831-ROADMAP-01 阶段 1 第六轮（S2E / ENEMIES / S2C3，2026-11-27，训练端 GPU + 引擎）

第六轮：给 checkpoint 臂补逐决策 `enemies` 读数并判定 Tier A、把 T04 三片合并训练做"数据轴"检验、跑三片策略的自然开局闭环。

### 47.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2e/` | 62 | 127,722,398 | 三片合训（M1）与留出片（M2）checkpoint、离线指标、兼容性自检 |
| `.tmp_v832_s2c3/` | 78 | 3,713,996 | A 批 Tier A 判定（3 根）+ B 批三片策略自然开局（3 根） |

### 47.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `tools/rl/v8_32_takeover.py` | 122,486 | `ddeae1a5807f96d2…`（2,667 行，含逐决策 enemies + boss_summary） |
| 2 | `tests/test_v8_32_takeover.py` | 100,077 | `95973eb27c34cb16…`（2,118 行，58 项） |
| 3 | `.tmp_v832_s2e/ckpt_m1.pt` | 10,469,063 | `efab01f811c5a253…`（19,772 行，6,000 步） |
| 4 | `.tmp_v832_s2e/ckpt_m2.pt` | 10,468,935 | `27880ea5b99a9c49…`（seed11+12，14.57 epoch） |
| 5 | `.tmp_v832_s2e/train_s2e.json` | 44,641 | `21252bac55534b26…` |
| 6 | `.tmp_v832_s2e/compat_s2e.json` | 12,735 | `25ca0682b956cddb…` |
| 7 | `.tmp_v832_s2e/s2e.md` | 4,294 | `0408df507bb158f9…` |
| 8 | `.tmp_v832_s2e/verify_s2e.py` | 24,587 | `29eb18177a7f3162…`（43/43） |
| 9 | `.tmp_v832_s2c3/results.json` | 46,331 | `5b47a92f29ed9773…` |
| 10 | `.tmp_v832_s2c3/s2c3.md` | 5,376 | `ef4672d8c370f967…` |
| 11 | `.tmp_v832_s2c3/verify_s2c3.py` | 18,426 | `1cf60bda7036aa0d…`（69/69） |
| 12 | `.tmp_v832_s2c3/ledger.jsonl` | 9,901 | `96c56834e14bda8c…` |

### 47.3 本轮结论（只作描述，不构成能力门）

- **Tier A = 0/3（首次可判）**：策略从入口 2,632 起 3/3 进入 boss 战，`boss_life` 6,000 → **54 / 1,335 / 1,579**（99.1% / 77.8% / 73.7%），半血帧 3,284 / 3,274 / 3,270；无一根观测到 0（seed10 在 4,282 帧 boss 被移除、seed11/12 在 3,624/3,620 阵亡）。
- **DPS 诊断**：教师 6.2–8.4 HP/帧；策略 seed11 6.74、seed12 6.43（同档但只打了约 690 帧，需 900–960）、seed10 4.41（活得久但输出不足）。→ 2/3 根差 200–250 帧生存、1/3 根差约 30% 输出。
- **数据轴饱和**：1 片 vs 2 片留出后的 donor top-1 `0.6359` vs `0.6361`（**+0.0002**，比跨片抖动 1.2–1.3 pp 小约 60 倍）。
- **闭环同向**：三片合训自然开局 1,900.3，对单片 2k +55.0（+3.0%）、对单片 8k −17.3（−0.9%），逐 seed 方向不一致、无 CI。

### 47.4 成本与限制

- 引擎启动 **6** 次、总帧 **17,449**；GPU 训练 147.95 s；新增磁盘 **131,436,394 B ≈ 0.122 GiB**。
- 静音 12/12 读数全 `0 0 0 0`；账本 12 行（6 reserve + 6 settle），无未结。
- 限制：Tier A 的"击杀"口径缺无尸体帧读数；恒动作臂的 boss 血量未测（分级指标暂无对照）；M1 无 donor；三根均为条件接管。

## 48 V831-ROADMAP-01 阶段 1 第七轮（S2F / S2C4 / S2C5，2026-11-27，训练端 GPU + 引擎）

第七轮：给恒动作臂拿 boss 读数（退化对照）、放宽 cap 验证"54 血 + 超时"、boss 窗口微调与等量随机对照的闭环消融。**本轮出现第一个 Tier A 达成。**

### 48.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2f/` | 47 | 109,661,280 | b1（boss 窗口行）/ b2（等量随机行）微调 checkpoint 与离线消融 |
| `.tmp_v832_s2c4/` | 70 | 1,177,970 | 恒动作臂 boss 读数（3 根）+ 放宽 cap（1 根） |
| `.tmp_v832_s2c5/` | 106 | 1,876,170 | b1/b2 × 3 根的闭环消融（含 Tier A 达成那根） |

### 48.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `.tmp_v832_s2f/ckpt_b1.pt` | 10,469,959 | `9548e3a80aee47c1…` |
| 2 | `.tmp_v832_s2f/ckpt_b2.pt` | 10,470,215 | `dcaad643340af94f…` |
| 3 | `.tmp_v832_s2f/train_s2f.json` | 48,133 | `c828159be928196b…` |
| 4 | `.tmp_v832_s2f/compat_s2f.json` | 16,383 | `4995122047652120…` |
| 5 | `.tmp_v832_s2f/s2f.md` | 3,846 | `7b8303ca8951d277…` |
| 6 | `.tmp_v832_s2f/verify_s2f.py` | 22,317 | `bd938c3a4d3063f4…`（41 项） |
| 7 | `.tmp_v832_s2c4/results.json` | 17,431 | `5c9f431f85d40e4f…` |
| 8 | `.tmp_v832_s2c4/s2c4.md` | 3,870 | `19df1ff51422bb49…` |
| 9 | `.tmp_v832_s2c4/verify_s2c4.py` | 21,158 | `a75bd9af15d14a58…`（63/63） |
| 10 | `.tmp_v832_s2c5/results.json` | 37,476 | `a183b44c9e0e7edc…` |
| 11 | `.tmp_v832_s2c5/s2c5.md` | 3,815 | `40d79529bbfd1f0d…` |
| 12 | `.tmp_v832_s2c5/verify_s2c5.py` | 18,041 | `216a8d5eb8a4fb62…`（29/29） |

### 48.3 本轮结论（只作描述，不构成能力门）

- **Tier A 首次达成（1/6）**：`b2`/seed10 从入口 2,632 接管，boss 首帧 2,932、**帧 3,640 血量到 0**、0 血尸体持续到 3,730；
  击杀 708 帧、8.47 HP/帧（教师 716–964 帧 / 6.2–8.4）。玩家随后在 4,426 阵亡，`stop=legal_death`。
- **退化对照成立**：恒动作 `cand 12` 三根 `boss_life` 全程 6,000（**0.00%**）→ "boss 血量削减%"可区分策略与退化臂。
- **"54 血 + 超时"不是 cap 问题**：放宽到总 5,200 后前 934 决策逐字段相同，只多活 246 帧便在 4,746 阵亡；boss 早在 4,282 带 54 血退场。
- **boss 窗口微调无效**：b1 每帧伤害更高（6.67/9.18/9.44 vs 4.41/6.74/6.43）但更早阵亡，总伤害反而更低（14,117 vs 15,032）；
  b1 的离线 boss 子集 top-1 最高（0.9940）却打不死 → **离线不预测闭环的第三次独立复现**。
- **臂内离散 > 臂间差**：b2 三根削减 100% / 35.6% / 62.0%；n=3、每格一条确定性 rollout，无法分离训练效应与个体噪声。

### 48.4 成本与限制

- 引擎启动 **10** 次、总帧 **36,684**；GPU 训练 25.3 s；新增磁盘 **112,715,420 B ≈ 0.105 GiB**。
- 静音 20/20 读数全 `0 0 0 0`；账本 20 行（10 reserve + 10 settle），无未结。
- 限制：Tier A 只 1/6；b1/b2 的窗口定义占全数据 78.12%（原工单估算 800 行/片有误，6.44×）；三根均为条件接管；
  需要用户裁决 WP05/DAgger 的授权（见 `TRAINING_HOST_REPLY_10.md` §10）。

## 49 V831-ROADMAP-01 阶段 1 第八轮（S2C6 / SAVE-OBS，2026-11-27，训练端引擎 + 离线代码）

第八轮：测 Tier A 的确定性与"换入口"复现率，并给 runner 补上 `--save-obs`（WP05 定点补采的数据面）。
**WP05 已获用户授权**（2026-11-27 03:26，选项 A）；本轮的 `--save-obs` 是其前置，采集本身待预算确认。

### 49.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2c6/` | 527 | 29,445,605 | 6 次真机评测（3 次重复 + 3 次换入口 2,704）、冻结仪器副本、账本、静音证据 |

### 49.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `tools/rl/v8_32_takeover.py` | 163,092 | `4361a2077c2abbf3…`（3,452 行，含 `--save-obs`） |
| 2 | `tests/test_v8_32_takeover.py` | 127,646 | `8130a3a2286b2855…`（2,629 行，71 项） |
| 3 | `.tmp_v832_s2c6/results.json` | 339,573 | `a654157d1f021440…` |
| 4 | `.tmp_v832_s2c6/s2c6.md` | 4,985 | `0c6483c57226658a…` |
| 5 | `.tmp_v832_s2c6/verify_s2c6.py` | 21,198 | `90756a7f766f7fac…`（14/14） |
| 6 | `.tmp_v832_s2c6/ledger.jsonl` | 14,979 | `3cabf37fd3f6a529…` |

### 49.3 本轮结论（只作描述，不构成能力门）

- **确定性**：同一 (ckpt, root, entry) 重跑 3 次 **13/13 项逐位相同**（897 条决策数据 sha 相同），唯一差异是 `reset.setup_frames`（冷启动握手）→ Tier A 在给定配置上是 0/1 确定值。
- **Tier A = 2/6 个 (root, entry) 配置**：`(b2, seed10, 2632)` 帧 3,640 击杀；`(b2, seed12, 2704)` 帧 3,644 击杀且活到 cap 4,500。入口位移 72 帧把 seed12 从"剩 2,280 血"翻成"击杀"。
- **`--save-obs` 打通**：每决策落盘三平面（0–255 原始值，逐位等于策略输入）+ 与 T04 字段对齐的 `slots.jsonl` + `meta.json` + `README.md`；实测 **53.5 KB/决策**。
- **WP05 预算修正**：3 根 × 2 ckpt ≈ **281 MB**（原估 230 MB 只算 play 平面）。

### 49.4 成本与限制

- 引擎启动 **6** 次、总帧 **25,846**（**超 25,000 上限 846 帧 / 3.38%**，因 seed12 存活到 cap；未截断、显式 WARN）；新增磁盘 **29,445,605 B ≈ 0.027 GiB**。
- 静音 12/12 读数全 `0 0 0 0`；账本 12 行（6 reserve + 6 settle），无未结。
- 限制：Tier A 的"率"只能按 (root, entry) 组合计（2/6，无 CI）；`--save-obs` 只有 96 决策的离线窗口实测，**没有真 checkpoint 跑过**；
  教师重标注能否吃这种行还没验（下一轮第一步）。

## 50 V831-ROADMAP-01 阶段 1 第九轮（DAGGER01，2026-11-27，训练端只读分析）

第九轮：判定 WP05/DAgger 的**标签链路**可行性（不启动引擎），为定点补采定方案与成本。

### 50.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_dagger01/` | 20 | 10,632,831 | 可行性判定、实测异常原文、成本算术与验收脚本 |

### 50.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `.tmp_v832_dagger01/feasibility.json` | 17,976 | `f8164e18c7ca389f…` |
| 2 | `.tmp_v832_dagger01/feasibility.md` | 8,232 | `a14b40b91a994246…` |
| 3 | `.tmp_v832_dagger01/verify_feasibility.py` | 17,406 | `abba954f2f05a955…`（10/10） |
| 4 | `.tmp_v832_dagger01/probe_feasibility.py` | 22,521 | `5a397fbec7a00ee3…` |

### 50.3 本轮结论（只作描述，不构成能力门）

- **需最小适配**：既有 `reslot`/`dataset_2f` 加载路径对 `--save-obs` 产物**全部 fail-closed**（缺 shard 三件套 + 15 个行级键）；
  观测平面与帧尺是干净的（96/96 掩码算术通过）。
- **推荐方案 b**：约 120 行小工具 + 20 行前缀胶水，复用 `counterfactual.Prefix`/`_Session` 与 `teacher/decide.py:129`；
  方案 a（伪造成 legacy shard）要伪造 4 帧 `step_frames` 与 T04 `slot_id`，是对学生 rollout 的不实陈述，已否。
- **状态可重建**：磁带 `lead_in`(16 帧) + 磁带到 entry 的 slot mask，在入口与学生行 0 的 HASH 逐位相同；entry 后必须用学生 mask（84/96 行不同）。
- **成本**：5,400 查询 / 6 次引擎启动 / **≈377 s**；对照每状态 RESET+重放约 4,600 s（13×）。

### 50.4 成本与限制

- 引擎启动 **0** 次；新增磁盘 **10,632,831 B ≈ 0.010 GiB**。
- 限制：样例是桩 client 产物（HASH 鉴别力待真机复核）；前缀重放必须用 `TeacherEngine(role="shadow")`；
  磁带是 entry 前 mask 的唯一来源，采集时必须同时登记磁带身份。本轮**未做**卡 v2。

## 51 V831-ROADMAP-01 阶段 1 第十轮（RELABEL / S2C7 / CARD-V2，2026-11-27，训练端代码 + 引擎）

第十轮：实现 WP05/DAgger 的重标注工具、真机冒烟"采→标"整条链、并冻结 `S1N-MB2` v2 判分阶梯。

### 51.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_relabel/` | 1,667 | 31,836,241 | 重标注工具的离线测试夹具与样本 |
| `.tmp_v832_s2c7/` | 57 | 11,178,257 | 真机冒烟：采集（200 决策 obs）+ 重标注（20 行教师标签） |

### 51.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `tools/rl/v8_32_relabel.py` | 98,956 | `347b24ba6ee612c7…`（2,037 行） |
| 2 | `tests/test_v8_32_relabel.py` | 59,411 | `06caabcad654bc8b…`（1,393 行，63 项） |
| 3 | `.tmp_v832_s2c7/results.json` | 84,147 | `2db3ed61cd2b8564…` |
| 4 | `.tmp_v832_s2c7/s2c7.md` | 3,257 | `ef15ec03df21d275…` |
| 5 | `.tmp_v832_s2c7/verify_s2c7.py` | 9,735 | `e104bb22eed23fff…`（15/15） |
| 6 | `.tmp_v832_s2c7/labels/seed10/labels.jsonl` | 70,796 | `631ca21bc6694079…`（20 行） |
| 7 | `.tmp_v832_takeover/task_card_mb2_v2.json` | 10,495 | content `6e7da68a8bbd25db…` |
| 8 | `.tmp_v832_takeover/task_card_mb2_v2.md` | 2,550 | `2f455a589a3fe8a5…` |
| 9 | `.tmp_v832_takeover/make_task_card_mb2_v2.py` | 9,669 | `48b9c1185bba2061…` |

### 51.3 本轮结论（只作描述，不构成能力门）

- **DAgger 链路真机跑通**：采集（`--save-obs`，200 决策、53.5 KB/决策）→ 重标注（`--limit 20`，20/20 成功、`nonfinite=0`），
  入口三重证明成立（引擎帧 2,632 = 磁带 slot `hash_pre` = 学生 row0 `hash_pre` = `e2a84b01b1aadef4`）。
- **教师与学生只有 10% 一致**：`executed==chosen` = 2/20；`chosen` 直方图 `{3:15, 2:2, 7:2, 0:1}`。
  这是协变量偏移的定量证据，也是 DAgger 标签信息量的直接指标。
- **成本**：采集 200 决策 29.278 s；重标注 20 决策 3.437 s（含 2,632 帧前缀重放）→ 900 决策约 0.8–2.6 min。
- **`S1N-MB2` v2 冻结**：Tier A（强）+ boss 伤害%（分级，退化臂 0.00%）+ 确定性说明 + 12 个 (ckpt, root, entry) 配置（Tier A = 2/12）。

### 51.4 成本与限制

- 引擎启动 **2** 次；引擎帧 **5,704**（采集 3,032 + 重标注 2,672）——**超 5,000 上限 704 帧**，因重标注必须先重放 2,632 帧前缀；新增学生帧仅 440。
- 静音 4/4 读数全 `0 0 0 0`；账本 4 行（2 reserve + 2 settle），无未结；新增磁盘 **43,014,498 B ≈ 0.040 GiB**。
- 限制：`--limit 20` 的 `agree=0.10` 只是链路证据（n 太小）；下游仍无 joiner（`reslot`/`dataset_2f` 对该格式 fail-closed）。

## 52 V831-ROADMAP-01 阶段 1 第十一轮（WP05 / JOINER，2026-11-27，训练端引擎 + 代码）

第十一轮：WP05 数据集建成（6 个失败配置的学生态教师标签）+ joiner 落地（合并成可训练语料）。

### 52.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_wp05/` | 141 | 179,928,076 | 6 套采集（obs）+ 6 套教师标签 + 账本/静音/汇总 |
| `.tmp_v832_join/` | 696 | 4,293,909,977 | joiner 产物与**三份同 sha 中间语料**（见 §52.4） |

### 52.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `tools/rl/v8_32_join.py` | 102,247 | `8b6ea1382dff0d05…`（2,130 行） |
| 2 | `tests/test_v8_32_join.py` | 41,355 | `6349151732bcecc8…`（887 行，25 项） |
| 3 | `.tmp_v832_wp05/results.json` | 41,674 | `857dd237c70acabc…` |
| 4 | `.tmp_v832_wp05/wp05.md` | 3,399 | `0cb8596d60b852e1…` |
| 5 | `.tmp_v832_wp05/verify_wp05.py` | 16,924 | `2ea1ac40589707b7…`（48/48） |
| 6 | `.tmp_v832_wp05/ledger.jsonl` | 40,524 | `ae162ecb9998a7bc…`（24 行） |

### 52.3 本轮结论（只作描述，不构成能力门）

- **3,116 条学生态教师标签**（6 配置 × 410–679 决策），**0 失败、0 非有限分数**、`chosen` 100% 在 `verified_mask` 内。
- **一致率 331/3,116 = 10.62%**（逐配置 0.070–0.179）；教师 `chosen` 集中在 **`cand 3`（63.1%）**。
- **入口与 checkpoint 无关**：`b1`/`b2` 在同 (seed, entry) 上 `entry_hash` 相同；6/6 入口三重证明成立。
- **joiner**：T04 19,772 行与 dagger 3,116 行可合并成 `s2e_train.merge_shards` 能读的语料；
  `readout01.gate_corpus` 是单 episode 门，多 episode 合并**构造上**不能通过（已记入 meta）。

### 52.4 成本与限制

- 引擎启动 **12** 次、引擎帧 **44,336**、墙钟 **400.1 s**、WP05 新增磁盘 **179.9 MB**；静音 24/24；账本 24 行无未结。
- `.tmp_v832_join` 有 3 份同 sha 中间语料（`real3`/`real3b`/`corpus_t04_3shard`）占约 3 GiB，**待清理**（需主代理解析绝对路径后执行并登记）。
- 限制：标签尚未用于训练；一致率只覆盖 6 配置的 3,116 状态；6 次采集全为 `legal_death`（决策数低于 cap 估算）。

## 53 V831-ROADMAP-01 阶段 1 第十二轮（S2G / 标签通道修正，2026-11-27，训练端 GPU + 磁盘清理）

第十二轮：把 WP05 标签并入 T04 重训（D1），得到**负结果**并定位到标签通道接错；同时清理 joiner 的重复语料。

### 53.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2g/` | 26 | 2,162,725,000 量级 | 合并语料（22,888 行）、D1 checkpoint、离线对照 |
| `.tmp_v832_join/` | 12 | 1,088,519,102 | 清理后仅剩权威 `corpus/` 与测试夹具 |

### 53.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `.tmp_v832_s2g/corpus/slots.jsonl` | 38,556,000 量级 | `7d8ebe0d…`（22,888 行） |
| 2 | `.tmp_v832_s2g/ckpt_d1.pt` | 10,469,383 | `bcb9d4aaf1af69a1…` |
| 3 | `.tmp_v832_s2g/train_s2g.json` | 49,322 | 配置 + 55 项指标 |
| 4 | `.tmp_v832_s2g/s2g.md` | 4,624 | 21 行 |
| 5 | `.tmp_v832_s2g/verify_s2g.py` | 28,638 | 18/18 PASS |
| 6 | `.tmp_v832_s2g/compat_s2g.json` | 4,827 | 22/22 加载断言 |

### 53.3 本轮结论（只作描述，不构成能力门）

- **合并语料 22,888 行**（T04 19,772 + dagger 3,116），配方可复现（同控制器重训 ckpt_m1 → state 逐位相同，max_abs_diff 0.0）。
- **负结果**：D1 vs ckpt_m1（同预算 6,000 步）—— T04 `executed` **−1.54 pp**、dagger `chosen` **−1.06 pp**、
  在"学生执行≠教师首选"的 2,785 行上 **−6.10 pp**；唯一大涨的是 dagger 行 `executed`（+39.70 pp，样本内训练目标）。
- **根因**：训练目标一直是硬 `executed_cand`，而 dagger 行的 `executed_cand` 是**学生动作**（3,116 行里仅 331 行等于 `chosen`）；
  `bomb` 已走教师通道，**动作没走** → 等于用 13.6% 的数据教模型复现学生。
- **修法**：joiner 增 `--dagger-label chosen`（默认），把 dagger 行的 `executed_cand/mask/bomb` 改成教师标签，学生动作存入 `student_*`。

### 53.4 成本与限制

- 引擎启动 **0** 次；GPU 训练 74.2 s；新增磁盘约 2.1 GB（含 1 GB 语料 + 10 MB ckpt + 1 GB obs 拷贝）。
- **清理**：删除 `.tmp_v832_join/{real3,real3b,corpus_t04_3shard}`（5 个关键文件 sha 与 `corpus/` 全等、路径与目录名逐一校验）→ **回收 2.99 GiB**，`corpus/` 完好。
- 限制：D1 未做闭环评测（标签通道已错，评测无解释力）；语料未去重（24 个重复 key / 27 行）。


## 54 V831-ROADMAP-01 阶段 1 第十四轮（S2H / S2I，2026-11-27，训练端 GPU + 引擎）

第十四轮：修正 DAgger 标签通道（`--dagger-label chosen`）、重训 D2、与同预算基线做闭环对照。
**首次出现数据侧改动带来的闭环任务完成数变化。**

### 54.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2h/` | 500 | 1,286,485,734 | 修正后的合并语料（22,888 行）、D2 checkpoint、三模型离线对照 |
| `.tmp_v832_s2i/` | 139 | 4,468,852 | D2 vs m1 的 12 次闭环对照（2 模型 × 6 配置） |

### 54.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `tools/rl/v8_32_join.py` | 117,606 | `caee1c97e1889415…`（2,387 行） |
| 2 | `tests/test_v8_32_join.py` | 52,591 | `030307f402dcd894…`（1,111 行，31 项） |
| 3 | `.tmp_v832_s2h/ckpt_d2.pt` | 10,469,959 | `3500bb84dc8fb67a…` |
| 4 | `.tmp_v832_s2h/train_s2h.json` | 76,235 | `82eb50289982d804…` |
| 5 | `.tmp_v832_s2h/verify_s2h.py` | 38,625 | `41a690eefcba613d…`（27/27） |
| 6 | `.tmp_v832_s2i/results.json` | 82,941 | `7cdce822ee9e48f1…` |
| 7 | `.tmp_v832_s2i/s2i.md` | 4,422 | `61cd94753e22ad58…` |
| 8 | `.tmp_v832_s2i/verify_s2i.py` | 22,281 | `125de543bd76e69f…`（17/17） |
| 9 | `.tmp_v832_s2i/ledger.jsonl` | 27,850 | `12e1a10e6089fb47…`（24 行） |

### 54.3 本轮结论（只作描述，不构成能力门）

- **只改标签通道**：T04 块 19,772 行逐字节相同、obs 22,888 行整文件同 sha、`batch_indices` 与 D1 相同；`--dagger-label executed` 可逐字节重现旧语料。
- **离线方向掰正**：dagger `top1_vs_chosen` **D2 0.9692** vs D1 0.1069 / m1 0.1175（+85.17 / +86.23 pp），六套 6/6 升；
  分歧行 0.9706 vs 0.0025 / 0.0636（+90.70 / +96.80 pp）。代价 T04 `top1_vs_executed` −1.84 pp。
- **闭环首次正效应**：**D2 Tier A 3/6、平均伤害 95.46%**；基线 **m1 1/6、88.01%**。
  入口 2,704 三根 **D2 3/3 击破 vs m1 0/3**；入口 2,632 三根 D2 0/3（87–97% 伤害）、m1 1/3。
- **保留配置一胜一负**：`#5 seed10/2632` D2 输（777 vs 击破）、`#6 seed12/2704` D2 赢（击破 vs 1,239）。

### 54.4 成本与限制

- 引擎启动 **12** 次、引擎帧 **48,318**、静音 24/24、账本 24 行无未结；GPU 训练 75.6 s；新增磁盘 **1.20 GiB**。
- 限制：D2 优势集中在采集覆盖的入口 2,704；保留配置仅 2 个且方向相反；单次训练无方差；12 次全 `legal_death`。

## 55 V831-ROADMAP-01 阶段 1 第十五轮（S2J，2026-11-27，训练端引擎）

第十五轮：DAgger 模型 D2 的**自然开局回归**与**未见入口 hold-out**。结论：**取舍，不是净收益。**

### 55.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2j/` | 138 | 5,764,424 | A 批自然开局（D2/m1 × 3 seed）+ B 批未见入口（2,560/2,768 × 3 seed） |

### 55.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `.tmp_v832_s2j/results.json` | 24,872 | `d52408fdfd380762…` |
| 2 | `.tmp_v832_s2j/s2j.md` | 3,899 | `87b521c232cb60a6…`（59 行） |
| 3 | `.tmp_v832_s2j/verify_s2j.py` | 31,808 | `f20c86253d6c335f…`（20/20） |
| 4 | `.tmp_v832_s2j/ledger.jsonl` | 30,478 | `7866f0f6fda0a4d3…`（24 行） |

### 55.3 本轮结论（只作描述，不构成能力门）

- **自然开局崩掉**：D2 三 seed **1,007 / 943 / 880（均值 943.3）** vs m1 **1,966 / 2,350 / 1,385（1,900.3）** → **−50.4%**；
  只比最早的 `pilot_final`（731）好，比本项目任何一次 T04 训练都差。
- **未见入口上能打 boss**：入口 **2,768** 三根 **2/3 Tier A**（伤害 100% / 93.83% / 100%）；
  但入口 **2,560** 0/3（71.47% 均值）→ 不是平滑的"越远越差"。
- **净判断**：晚段（2,632–2,768 接管）变强、早段（0–1,000 帧）崩掉 → **按路线阶段 2 判据，当前不构成"像素闭环学习收益"通过**。
- 最可能原因：DAgger 行全部来自帧 ≥2,632，占比 13.6%，把策略往晚段分布拉；早段没有对应数据。

### 55.4 成本与限制

- 引擎启动 **12** 次、引擎帧 **31,299**（A 8,531 + B 22,768）、静音 24/24、账本 24 行无未结；新增磁盘 **5,764,424 B ≈ 0.005 GiB**。
- 限制：A/B cap 不同（4,500 vs 4,500−entry），帧数不可横比；单次训练、3 个 seed，无方差；
  验收首跑 19/20（报告行数上限），收敛后 20/20。

## 56 V831-ROADMAP-01 阶段 1 第十六轮（S2K / S2L，2026-11-27，训练端 GPU + 引擎）

第十六轮：给 DAgger 行加损失权重（W=0.05/0.15/0.00），并用 W=0.15 做闭环判决。结论：**降权重也救不回早段。**

### 56.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2k/` | 13 | 约 52 MB | 三个加权 ckpt + W=1.0 审计重训 + 离线并表 |
| `.tmp_v832_s2l/` | 140 | 约 5 MB | W=0.15 的 12 次闭环（自然开局 3 + 三入口各 3） |

### 56.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `.tmp_v832_s2k/ckpt_d3_w15.pt` | 10,470,735 | `4aa4441d58848234…` |
| 2 | `.tmp_v832_s2k/ckpt_d3_w05.pt` | 10,470,735 | `cada10d07c8a53ff…` |
| 3 | `.tmp_v832_s2k/ckpt_d3_w00.pt` | 10,470,735 | `5ed377463630af0e…` |
| 4 | `.tmp_v832_s2k/train_s2k.json` | 60,950 | 四模型离线并表 |
| 5 | `.tmp_v832_s2k/verify_s2k.py` | 25,703 | 22/22 PASS |
| 6 | `.tmp_v832_s2l/results.json` | 66,909 | 12 行全字段 |
| 7 | `.tmp_v832_s2l/verify_s2l.py` | 22,192 | 20/20 PASS |

### 56.3 本轮结论（只作描述，不构成能力门）

- **加权实现干净**：W=1.0 重训 ≡ D2、W=0.0 重训 ≡ m1，两者 `max_abs_diff` 均 **0.0**；三个 W>0 的 `batch_indices` 与 D2 同 sha。
- **离线曲线**：dagger `chosen` 严格单调（0.1175→0.6829→0.8601→0.9692）；T04 保留度非单调（1.0000→0.9885→**0.9986**→0.9809）→ **W=0.15 双向支配 W=0.05**。
- **闭环判决（W=0.15）**：自然开局 **1,176.7**（vs 门槛 1,845.3 差 −668.6、vs m1 差 −723.7）；
  2,768 **100% / Tier A 3/3**（最好）；2,632 70.18% / 0-3（D2 90.92%、m1 89.13%）；2,560 52.23% / 0-3（D2 71.47%）。
- **两端兼得不成立**；且**离线保留 99.86% 没能预测闭环 −38%**（"离线不预测闭环"的第四次复现，落差最大）。

### 56.4 成本与限制

- 引擎启动 **12** 次、引擎帧 **36,936**、静音 24/24、账本 24 行无未结；GPU 训练 **306.4 s**（4 × 6,000 步）。
- 限制：每个 W 只训一次、无种子方差；W=0.05 未做闭环（离线被支配）；m1 在 2,560/2,768 无产物；
  m1 在 2,632 的数字已按 s2i 原始产物更正为 100.0/78.33/89.07（均 89.13%，Tier A 1/3）。

## 57 V831-ROADMAP-01 阶段 1 第十七轮（S2M / S2N，2026-11-27，训练端 GPU + 引擎）

第十七轮：量化**训练种子方差**（离线与闭环），并据此重新校准所有既往结论的效力。

### 57.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2m/` | 约 15 | 约 52 MB | 纯 T04 的 init seed 1/2/3（+ s0 审计重训） |
| `.tmp_v832_s2n/` | 约 140 | 约 5 MB | 三模型 × 三环境种子的自然开局（9 次） |

### 57.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `.tmp_v832_s2m/ckpt_s1.pt` | 10,470,599 | `1414822f92382f38…` |
| 2 | `.tmp_v832_s2m/ckpt_s2.pt` | 10,470,599 | `b50c96381d1b03de…` |
| 3 | `.tmp_v832_s2m/ckpt_s3.pt` | 10,470,599 | `98811975fbd5ef18…` |
| 4 | `.tmp_v832_s2m/ckpt_s0_replay.pt` | 10,472,245 | `498c1e015c1e0e67…`（张量 ≡ m1） |
| 5 | `.tmp_v832_s2m/train_s2m.json` | 72,939 | 四模型离线 + 极差 + 等价性 |
| 6 | `.tmp_v832_s2n/results.json` | 约 60 KB | 9 行 + 均值/极差 |
| 7 | `.tmp_v832_s2n/verify_s2n.py` | 约 25 KB | 39/39 PASS |

### 57.3 本轮结论（只作描述，不构成能力门）

- **闭环种子方差**：同一配方只换初始化种子，自然开局均值 **1,601.7 – 2,115.7（极差 514 帧）**；
  同一模型跨环境种子极差 **66 – 965 帧**。
- **离线种子方差**：T04 top-1 极差 **1.23 pp**（sd 0.62 pp），≥ DAgger 的离线代价（−0.14 ～ −1.84 pp）。
- **仍然成立的结论**：BC 对 `pilot_final` 的 +1,385；DAgger 的 −425 ～ −659（超噪声但幅度有限）。
- **必须撤销的结论**：8k > 2k（+72）、三片 > 单片（+55）、"m1 是好基线"（实为中位种子；最好纯 T04 种子 2,115.7）。
- **度量基准改为**：每配方 ≥3 个初始化种子，报告均值与极差；单次训练数字不作结论。

### 57.4 成本与限制

- 引擎启动 **9** 次、引擎帧约 **15,500**、静音 18/18、账本 18 行无未结；GPU 训练 **301.6 s**。
- 限制：只测了纯 T04 一个配方；接管 Tier A 的种子分布未测；每档 3 个环境种子，标准误约 ±250 帧。

## 58 V831-ROADMAP-01 阶段 1 第十八轮（S2O，2026-11-27，训练端引擎）

第十八轮：**统计预算标定**——同一模型跑 10 个环境种子，定出"要分辨 500 帧需要多少种子"。

### 58.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2o/` | 约 60 | 约 10 MB | s1 模型的 10 个环境种子（1000–1009）自然开局 |

### 58.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `.tmp_v832_s2o/results.json` | 约 40 KB | 10 行 + 分布 + 所需 n |
| 2 | `.tmp_v832_s2o/s2o.md` | 约 3 KB | ≤40 行 |
| 3 | `.tmp_v832_s2o/verify_s2o.py` | 约 25 KB | 42/42 PASS |
| 4 | `.tmp_v832_s2o/ledger.jsonl` | 约 20 KB | 20 行（10+10） |

### 58.3 本轮结论（只作描述，不构成能力门）

- **10 个环境种子**：均值 **1,938.8**、sd **459.3**、极差 **1,284**（1,281–2,565）、均值 95% CI [1,610.3, 2,267.3]。
  3 个种子给出的 2,115.7 是高估（−177），极差 868 低估了尾巴。
- **标准误**：n=3 → **265.2 帧**；n=10 → 145.2（3 种子噪声是 10 种子的 1.83 倍）。
- **分辨 500 帧需要每组 n=15**（非中心 t，α=0.05、power=0.8；n=14 时 0.792、n=15 时 0.821）；n=10 只有 power 0.634。
- **DAgger 的 −957/−724 在 n=3 下 p=0.063/0.126（不显著）**；3 种子在 power 0.8 下只能分辨 ≥1,410 帧。
- **重复性成立**：seed 1000/1001/1002 逐位复现，且与既有 `episodes.jsonl` 50 字段全同（仅 `wall_s` 不同）。

### 58.4 成本与限制

- 引擎启动 **10** 次、引擎帧 **19,388**、静音 20/20、账本 20 行无未结。
- 限制：sd 只有 10 个样本（95% CI [316, 838]）→ 所需 n 的点估计 15、区间 8–46；
  只标定了自然开局这一个端点，接管 Tier A 的方差结构未标定。
- 执行代理披露并修正了一处统计实现错误（power 函数的 ν/W 与 W/ν 颠倒），已用教科书锚点与 MC 复核。

## 59 V831-ROADMAP-01 阶段 1 第十九轮（S2P，2026-11-27，训练端引擎）

第十九轮：**n=15 配对 A/B 定案**——纯 T04 vs T04+DAgger 的自然开局。

### 59.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2p/` | 约 200 | 约 20 MB | A 组补 5 个种子 + B 组 15 个种子（20 次启动） |

### 59.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `.tmp_v832_s2p/results.json` | 约 50 KB | 20 行 + 配对/Welch 统计 |
| 2 | `.tmp_v832_s2p/s2p.md` | 约 4 KB | 50 行 |
| 3 | `.tmp_v832_s2p/verify_s2p.py` | 约 25 KB | 22/22 PASS |
| 4 | `.tmp_v832_s2p/ledger.jsonl` | 约 30 KB | 40 行（20+20） |

### 59.3 本轮结论（只作描述，不构成能力门）

- **A（纯 T04）1,898.3 ± 396.9** vs **B（DAgger）910.6 ± 52.2**；
  配对差 **−987.7 帧**、95% CI [−1,209.9, −765.6]、配对 t −9.54（df 14）、**p = 1.68e-07**；Welch p = 1.22e-07；dz = −2.46。
- **B 组对种子几乎不敏感**（sd 52.2、极差 169）而 A 组 396.9 / 1,284 → **固定死法**，与回执 17 的 `dominant=E 60%` 一致。
- **判决**：DAgger 在当前配方下**净负且显著**（−52% 自然开局）。

### 59.4 成本与限制

- 引擎启动 **20** 次、引擎帧 **22,746**、静音 40/40、账本 40 行无未结。
- 限制：A 组 seed 1000–1009 为**产物级复用**（非本批复跑），B 组 1000–1002 逐位复现了更早批次；
  私有 `game-muted/` 被引擎写入 `score.dat`/5 个 backup（只读源未变）。

## 60 V831-ROADMAP-01 阶段 1 第二十轮（FORENSIC01 / S2Q / S2R，2026-11-27，训练端离线 + GPU + 引擎）

第二十轮：DAgger 失败的**机制归因**（纯离线）、DAgger 配置的复训件、纯 T04 三个复训件在接管任务上的基线。

### 60.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_forensic01/` | 约 30 | 约 1 MB | 15 份 B 组 + A 组决策日志的法证（箱表、动作词表校验） |
| `.tmp_v832_s2q/` | 约 20 | 约 55 MB | DAgger 配置的 init seed 1/2/3 复训 + s0 审计重训 |
| `.tmp_v832_s2r/` | 约 120 | 约 5 MB | 纯 T04 三复训件 × 4 配置的接管基线（12 次） |

### 60.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `.tmp_v832_forensic01/forensic.json` | 258,711 | 15×11 箱表 + 事件计数 |
| 2 | `.tmp_v832_forensic01/forensic.md` | 7,962 | 53 行（含纯文本箱表） |
| 3 | `.tmp_v832_forensic01/verify_forensic.py` | 15,141 | 21/21 PASS |
| 4 | `.tmp_v832_s2q/ckpt_d2_s1.pt` | 10,471,725 | `9e86259b4b2e33ea…` |
| 5 | `.tmp_v832_s2q/ckpt_d2_s2.pt` | 10,471,725 | `0974969893e15eea…` |
| 6 | `.tmp_v832_s2q/ckpt_d2_s3.pt` | 10,471,725 | `5ba083ac6d966415…` |
| 7 | `.tmp_v832_s2q/train_s2q.json` | 56,367 | 14/14 PASS |
| 8 | `.tmp_v832_s2r/results.json` | 约 60 KB | 12 行 + by_config + vs_reference |
| 9 | `.tmp_v832_s2r/verify_s2r.py` | 约 25 KB | 17/17 PASS |

### 60.3 本轮结论（只作描述，不构成能力门）

- **机制归因（法证，0 引擎）**：D2 锁死 `cand 3`（E/右，占 61.1%，帧 600 后 ≥94%），贴右墙 `x=376.0`（15/15），
  末端零位移 158–324 帧（15/15），炸弹自帧 402 起为 0，无敌在帧 662 结束、首弹幕帧 686（均 15/15 一致），
  随后 206–376 帧被首颗抵达角落的子弹打死。**对照：纯 T04 从不触右墙（0.000），且同种子上"完全静止"能活 1,692–1,748 → B 比站桩还差。**
- **DAgger 复训件离线**：四个初始化种子的 dagger `top1_vs_chosen` 极差 **2.15 pp**、T04 极差 **3.25 pp**（单片最大 4.02 pp）。
- **接管任务的复训方差**：纯 T04 三复训件在 4 个配置上 Tier A 合计 **4/12**，但按模型差异巨大
  （s2 95.85% / 2-4；s3 67.71% / 2-4；s1 58.20% / 0-4）；入口 2,632 上三复训件的伤害极差 **69.5–70.5 pp**。
  → **此前所有 Tier A 比较（2/12、3/6、1/6）也都在复训噪声里**。
- **两任务反相关**：自然开局最好的 s1（2,115.7 均）在接管上最差（0-4）；自然开局最差的 s2（1,601.7）在接管上最好（2-4）。

### 60.4 成本与限制

- 引擎启动 **12** 次（s2r）、GPU 训练 315.3 s（s2q）、法证 0 次；静音 24/24；账本 24 行无未结。
- 限制：法证无子弹坐标（"哪颗弹"由 graze/hit 推断）；m1 在 2,768 无本机产物；
  接管端每模型只有 4 个配置，Tier A 的分辨率 1/4。

## 61 V831-ROADMAP-01 阶段 1 第二十一轮（S2S，2026-11-27，训练端引擎）

第二十一轮：**墙角守卫的否证实验**——按上一轮法证给出的可修性假设，预先写死判据并检验。

### 61.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2s/` | 约 60 | 约 12 MB | 1 次等价性 + 15 个种子的守卫实验 |

### 61.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `tools/rl/v8_32_wallguard_eval.py` | 25,247 | 新建（未改 pinned 工具） |
| 2 | `.tmp_v832_s2s/results.json` | 83,548 | 15 行 + 触发统计 + 判定 |
| 3 | `.tmp_v832_s2s/s2s.md` | 4,660 | ≤50 行 |
| 4 | `.tmp_v832_s2s/verify_s2s.py` | 21,324 | 20/20 PASS |
| 5 | `.tmp_v832_s2s/ledger.jsonl` | 15,993 | 32 行（16+16） |

### 61.3 本轮结论（只作描述，不构成能力门）

- **判据预先写死**：`frames > 1,200` 需 ≥10/15；`≤5/15` 判否证。
- **实测**：守卫生效 **2,781/6,782 次决策（41%）**，其中 **2,193 次在 x=376.0 的墙上**；
  结果 **854–1,015 帧（均 903.6、sd 48.3）**，**0/15 > 1,200** → **MECHANISM FALSIFIED**。
- **原因**：在墙上 `cand 3`（E）与 `cand 0`（idle）位移都为 0 → 守卫把"空操作"换成"空操作"，轨迹几乎不变（Δ 均值 −7.0）。
- **中立性对照通过**：首次替换之前的决策行与基线逐字节相同。
- **更强的结论**：D2 在 ~900 帧时已是"几乎不动"的策略，比同种子 `still`（1,692–1,748）还差 800 帧 → **死因是"停在原地 + 不会躲"，不是位置约束**。

### 61.4 成本与限制

- 引擎启动 **16** 次、引擎帧约 **14,500**、静音 32/32、账本 32 行无未结。
- 限制：守卫只覆盖右墙；`reset.setup_frames` 在等价性比较中被遮蔽（setup 阶段实时读数，不入游戏输入）；
  本轮只否证了"可修性"，未否证法证描述本身。

## 62 V831-ROADMAP-01 阶段 1 第二十二轮（S2T，2026-11-27，训练端引擎 + 证据修复）

第二十二轮：**带滞回的反弹守卫**（把玩家真正推离右墙）与一处**证据完整性事件的修复**。

### 62.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2t/` | 约 70 | 约 12 MB | 1 次等价性 + 15 个种子的反弹守卫 |

### 62.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `tools/rl/v8_32_bounce_eval.py` | 新建 | `1f9ca58b…` |
| 2 | `.tmp_v832_s2t/results.json` | 约 80 KB | 15 行 + 反弹统计 + 判定 |
| 3 | `.tmp_v832_s2t/s2t.md` | 4,660 量级 | 44 行 |
| 4 | `.tmp_v832_s2t/verify_s2t.py` | 约 22 KB | 26/26 PASS |
| 5 | `.tmp_v832_s2t/ledger.jsonl` | 约 16 KB | 32 行（16+16） |

### 62.3 本轮结论（只作描述，不构成能力门）

- **按预设判据否证**：`>1,200` 的种子 **5/15**（判据 `≤5/15` 判否证）；均值 **1,010.3**（基线 910.6，+99.7），但**双峰**（5 个 1,238–1,353 / 10 个 823–934，8/15 低于自身基线）。
- **机械上有效**：基线 **15/15 死在 x=376.0 的墙上**，本批 **0/15 死在墙上**（死亡 x∈[309,368]、y∈[91,231]）。
- **真实缺口暴露**：argmax 里 **E+W 占 73.2%**、**纯 N/S 仅 26/7,582 = 0.34%**；51.0% 决策仍在旧贴墙带。
  → **墙角是症状，病是"只会左右走"（缺竖直控制）**。

### 62.4 证据完整性事件（已修复）

- 执行代理误跑 `v431_audio_cfg_audit.py --help`，该工具**忽略未知参数并重写**了
  `Document/Update/Alpha/v4.3.1/evidence/R14/audio_cfg_audit.json`（20 行增/20 行删）。
- 主代理用 `git checkout --` 恢复，`git status` 现为空；恢复后数值与 §9.2 的 R14 一致（105/105、muted 0、71/2/32），sha `2e433b6e…`。
- **工具缺陷未修**（对未知参数不报错、无条件写盘），已登记为后续任务。

### 62.5 成本与限制

- 引擎启动 **16** 次、引擎帧 **16,162**、静音 32/32、账本 32 行无未结。
- 限制：守卫只覆盖右墙；5/15 落在否证边界，未做事后改判；
  "竖直控制缺失"只是本轮暴露的缺口，尚未做因果检验（观测通道消融正在做）。

## 63 V831-ROADMAP-01 阶段 1 第二十三轮（S2U / S2V，2026-11-27，训练端引擎 + GPU）

第二十三轮：**观测通道消融**（哪一路输入在被使用）与**类别平衡重训**（罕见动作丢失是否来自损失）。

### 63.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2u/` | 约 90 | 约 20 MB | 3 组消融 × 5 种子 + 1 次等价性（17 次启动） |
| `.tmp_v832_s2v/` | 约 20 | 约 55 MB | 类别加权三件（cw00/cw10/cw50）与离线分布 |

### 63.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `tools/rl/v8_32_ablate_eval.py` | 新建 | pinned 工具未改 |
| 2 | `.tmp_v832_s2u/results.json` | 55,519 | 19/19 PASS |
| 3 | `.tmp_v832_s2u/s2u.md` | 5,742 | ≤50 行 |
| 4 | `.tmp_v832_s2v/ckpt_cw00.pt` | 10,470,283 | `3f4f529a…`（张量 ≡ m1） |
| 5 | `.tmp_v832_s2v/ckpt_cw10.pt` | 10,470,347 | `90b5ffb6…` |
| 6 | `.tmp_v832_s2v/ckpt_cw50.pt` | 10,470,347 | `051d1055…` |
| 7 | `.tmp_v832_s2v/s2v_recall.json` | 13,892 | 稀有类召回 |

### 63.3 本轮结论（只作描述，不构成能力门）

- **三路输入都在用**：`play` 置零 −1,209 帧（−3.05 sd，强）、`hud` −803（−2.02 sd，强）、`act_hist` −555（−1.40 sd，弱）。
  **没有"几乎未被使用"的通道** → 像素不是瓶颈。
- **`play` 置零的画像与 DAgger 同款**：5/5 种子恰好 891 帧、W 100%、竖直 0.00%（D2：910.6 帧、E 61%、竖直 0.12%）。
- **类别平衡不是杠杆**：m1 在**教师状态上**离线分布已与教师一致（竖直 3.46% vs 3.82%、focus 25.07% vs 25.54%、L1 0.0286），
  加权把竖直推到 4.8–5.9% 却让 top-1 掉 9.7–11.9 pp、L1 放大 6 倍 → **离开教师分布**。
  ⇒ 罕见动作丢失发生在**闭环状态漂移**，不在损失。
- **真正的缺口是 DAgger 采错状态区**：既有数据全在入口 2,560–2,704（boss 战），而崩溃在自然开局 500–700 帧。

### 63.4 成本与限制

- 引擎启动 **17** 次（消融 16 有效 + 1 次中止补测，**超上限 1 次**，中止那一对账本行未删改）、引擎帧 **21,054**、静音 34/34。
- GPU 训练 **246 s**（三次 6,000 步）；账本 34 行。
- 限制：消融每组 5 个种子；类别 9 在语料中 0 行；`play` 置零的 891 帧是策略坍塌（世界确实分叉），非引擎退化。

## 64 V831-ROADMAP-01 阶段 1 第二十四轮（WP06，2026-11-27，训练端引擎）

第二十四轮：**自然开局段的学生态教师标签**（入口 64），检验"补开局段学生态能否纠偏"。

### 64.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_wp06/` | 约 120 | 105 MB | 3 套采集（obs）+ 3 套开局段教师标签 |

### 64.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `.tmp_v832_wp06/results.json` | 3,697 | 六次运行全字段 |
| 2 | `.tmp_v832_wp06/comparison.json` | 4,329 | A/B/C 三段分布 + TV 距离 |
| 3 | `.tmp_v832_wp06/wp06.md` | 4,454 | 50 行 |
| 4 | `.tmp_v832_wp06/verify_wp06.py` | 15,640 | 28/28 PASS |
| 5 | `.tmp_v832_wp06/ledger.jsonl` | 10,240 | 12 行（6+6） |

### 64.3 本轮结论（只作描述，不构成能力门）

- **1,840 条开局段学生态标签**（入口 64、3 种子、agree 0.259、全部 legal_death、nonfinite 0）。
- **教师的行为取决于"谁的状态"**：同一教师、同一帧区间——
  学生状态上 `cand3`（右）**43.3%**、`cand0` 15.7%；教师自己的状态上 `cand3` 13.5%、`cand0` **54.2%**。
  TV(A,B)=**0.424**、TV(A,C)=0.590、TV(B,C)=0.178。
- **倾向性结论**：学生正在右向撞死，而教师在这些状态上说"继续向右"——**这批标签会强化右偏**。
- **两处前提纠正**（执行代理用证据）：①"boss 段 63% 是 cand3"不成立（63–69% 是 **cand0**；cand3 仅 8.6%），
  63.1% 属 **WP05 的 DAgger 标签**；②"DAgger 从未覆盖开局段"不准确——语料有 38,400 条帧<2,560 的行，但那是**教师状态**，
  缺的是"**学生状态**"这一维。

### 64.4 成本与限制

- 引擎启动 **6** 次、引擎帧约 **13,100**、静音 12/12、新增磁盘 **105 MB**、账本 12 行无未结。
- 限制：**标签行没有玩家坐标**，无法判定"贴墙时才说右"；采集未跑满 cap（三次都在 2,500 帧内死亡）；
  `ckpt_s1` 的失败**不是**单候选冻结（分散在 10 个候选上，异于 D2 锁 cand3）。

## 65 V831-ROADMAP-01 阶段 1 第二十五轮（S2W + STAGE2 评估，2026-11-27，训练端 GPU）

第二十五轮：把开局段学生态标签并入重训（W1），检验"会强化右偏"的封存先验；并结算阶段 2。

### 65.1 目录身份

| 目录/文件 | 字节 | 说明 |
| --- | ---: | --- |
| `.tmp_v832_s2w/corpus/` | 约 1.1 GB | T04 19,772 + WP06 1,840 = **21,612 行** |
| `.tmp_v832_s2w/ckpt_w1.pt` | 10,469,127 | `77056fb7…`，init 1、6,000 步、W=1.0、74.6 s |
| `.tmp_v832_s2w/design_s2w.json` | 10,625 | 训练前签章的 P1–P5 先验（seal `9d3a0312…`） |
| `Document/Update/Alpha/v8.31.0/STAGE2_ASSESSMENT.md` | — | 阶段 2 逐条结算（**结论：不通过**） |

### 65.2 本轮结论（只作描述，不构成能力门）

- **先验方向命中**：在 1,840 行 WP06 学生态上，`cand3` 从 s1 的 **11.96%** 抬到 W1 的 **44.84%（+32.88 pp）**，
  预设门槛 +5 pp；三片同向（w1 58.9/45.8/32.6% vs s1 10.5/11.2/13.6%）。
- **超出教师**：教师在这些状态是 43.26%，W1 **44.84%**（P2 假设"不会超过"被否）。
- **代价很小**：T04 `top1_vs_executed` 0.9719 → 0.9625（−0.94 pp）；竖直 3.72→3.66%、focus 25.34→25.76%。
- **等价性闸门**：同训练器在旧 22,888 行语料 + init 0 重跑与 `ckpt_d2` **逐位相同**（state `9c5a6588…`），
  证明本轮只动了「语料 + 初始化种子」。
- **处置**：WP06 不得作为"自然开局修复"推广；同 W=1.0 的 d2 在这些行上 `cand3` 65.54%、闭环自然开局 910.6 帧。

### 65.3 成本与限制

- 引擎启动 **0**；GPU 训练 74.6 s；新增磁盘约 **1.1 GB**。
- 限制：本轮**未做闭环评测**（W1 的自然开局未测）；标签仍无玩家坐标（下一步 wp07 补）；
  1,840 行对 W1 是样本内、对 s1 是样本外——结论只依赖"同一批状态上的输出差"。

## 66 V831-ROADMAP-01 阶段 1 第二十六轮（WP07，2026-11-27，训练端引擎）

第二十六轮：给重标注行补**玩家坐标**（`PLAYER.px/py`，动作前读），并在 1,840 行开局段学生态上检验"位置条件"假设。**假设被反转。**

### 66.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_wp07/` | 约 40 | payload 7.4 MB | 3 套带坐标的标签 + 分箱表 + 验收 |
| `.tmp_v832_relabel_wp07/` | 约 200 | 约 30 MB | 新增 19 项坐标测试的 pytest 夹具 |

### 66.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `tools/rl/v8_32_relabel.py` | 119,085 | `db4ee78df4147a12…`（2,424 行，+387） |
| 2 | `.tmp_v832_wp07/results.json` | 16,852 | 分箱表 + px 分位 + 判定 |
| 3 | `.tmp_v832_wp07/wp07.md` | 3,805 | 47 行 |
| 4 | `.tmp_v832_wp07/verify_wp07.py` | 22,612 | 37/37 PASS |
| 5 | `.tmp_v832_wp07/ledger.jsonl` | 4,940 | 6 行（3+3） |

### 66.3 本轮结论（只作描述，不构成能力门）

- **"贴右墙"前提错误**：1,840 个自然开局学生状态里 **x ≥ 300 一行都没有**（max 263.2、中位 33.4、x<100 占 79.5%）；学生整段贴**左**侧（x=8.0 钳位 86 行）。
- **教师"说右"= 从左墙救回来**：`px ≤ 8.001` 的 86 行 **cand3 = 100%**；x<100 48.56%；100–200 28.64%（cand7 41.36%）；
  200–300 14.56%（cand7 34.81%、cand0 24.68%）。`cand3` 随 x **单调下降**（50.3% → 2.6%），`cand7` 反向上升。
  → **同一句"向右"在不同 x 上意思相反**；W1 的 +32.88 pp 是把条件映射学成了边缘分布。
- **学生没执行恢复建议**：x<100 的 1,462 行里教师说右 48.56%、学生执行右仅 **9.58%**、idle 28.80% → **失败在执行**。
- **判据按预设落"不成立"**，但分子为空（右带 0 行）；并列记录 `cand3_wall=null` / `diff_pp=null` / `diff_pp_arith=−43.26`，未把空带当 0%。

### 66.4 成本与限制

- 引擎启动 **3** 次（只重标注）、引擎帧约 2,000、静音 6/6、账本 6 行无未结。
- 限制：右墙带为空是语料属性，**右墙问题仍未回答**（需新采集让学生跑到右墙）；
  `--require-player` 默认关闭（既有 63 项测试用不含 PLAYER 的假引擎）。

## 67 V831-ROADMAP-01 阶段 1 第二十七轮（S2X，2026-11-27，训练端 GPU）

第二十七轮：把 WP06 的 1,840 行按 **(x 带 × cand) 逆频率重加权**后重训（W2），检验能否把"位置条件性"学回来。

### 67.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2x/` | 约 25 | 约 1.1 GB | 21,612 行语料、行权重、W2、等价性重训 |

### 67.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `.tmp_v832_s2x/ckpt_w2.pt` | 10,470,663 | 见 `train_s2x.json` |
| 2 | `.tmp_v832_s2x/weights.json` / `.npy` | 6,386 / 86,576 | 行权重与每带直方图 |
| 3 | `.tmp_v832_s2x/train_s2x.json` | 约 100 KB | 三带对照表 + 等价性 |
| 4 | `.tmp_v832_s2x/gate_s2x.json` | 1,722 | 全 1 权重重训 ≡ W1 |
| 5 | `.tmp_v832_s2x/verify_s2x.py` | 约 25 KB | 独立复算 |

### 67.3 本轮结论（只作描述，不构成能力门）

- **条件性学到了**：W2 在 x<100 / 100–200 / ≥200 三带上的 `cand3` = **41.45 / 29.09 / 14.56%**（单调下降），
  教师为 48.56 / 28.64 / **14.56%**（中高带完全相同）；`cand7` = 9.37 / 40.45 / 34.81% vs 教师 9.37 / 41.36 / 34.81%。
- **对照 W1（行权重全 1）**：50.48 / 29.55 / 13.92 —— 在 x<100 带**说右比教师还多**，是"到处都说右"。
- **代价小**：T04 `top1_vs_executed` 0.9632（s1 0.9719，−0.87 pp）。
- **含义**：同一批 1,840 行，**换行权重就从不条件变成条件**——这是本项目第一次"用标签权重把条件性买回来"。

### 67.4 成本与限制

- 引擎启动 **0**；GPU 训练两次（W2 + 等价性闸门，各约 78 s）；新增磁盘约 1.1 GB。
- 限制：三带对照是**离线、样本内**；"条件性能否转化为生存"是闭环问题，已派发 15 种子预注册评测。

## 68 V831-ROADMAP-01 阶段 1 第二十八轮（S2Y + 更正，2026-11-27，训练端引擎）

第二十八轮：W2（位置条件重加权）的 15 种子预注册闭环评测，以及**一处自我更正**。

### 68.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2y/` | 约 90 | 约 20 MB | 15 个种子的自然开局 + 预注册 |

### 68.2 关键文件

| # | 路径 | 字节 | sha256 |
| ---: | --- | ---: | --- |
| 1 | `.tmp_v832_s2y/results.json` | 49,938 | 15 行 + 配对统计 + P1 判定 |
| 2 | `.tmp_v832_s2y/prior.json` / `prior_lock.json` | 7,357 / 366 | 预注册（跑前锁定） |
| 3 | `.tmp_v832_s2y/verify_s2y.py` | 34,127 | 独立复算 |
| 4 | `.tmp_v832_s2y/ledger.jsonl` | 38,610 | 30 行（15+15） |

### 68.3 本轮结论（只作描述，不构成能力门）

- **W2 闭环失败**：15 种子均值 **1,560.1** vs s1 同种子 **1,898.3**，**配对 −338.3**；预注册 P1（≥+200）**失败**。
- **自我更正**：回执 29 的"W1 是全局偏置、W2 学出条件性"**错误**——W1 的三带 `cand3` 也严格单调（50.48/29.55/13.92）；
  重加权的真实作用是**压低右向总量**（44.84% → 37.66%，低于教师 43.26%），代价是低 x 带欠冲 −7.11 pp；
  逐带 |cand3−教师| **W2 2.52 pp vs W1 1.15 pp（W2 更远）**。已在 `REPLY_29` 顶部加更正块，原文保留。
- **三次标签注入对照**：D2（boss 段）−987.7、W2（开局段重加权）−338.3、W1（开局段全权重）未测闭环 → **任何学生态标签注入都让自然开局更差**。
- **该线结论**：在这个 BC 配方下，学生态教师标签注入**不再有收益**；下一方向为"改观测契约（让策略看到 px/py）"或"在线学习"。

### 68.4 成本与限制

- 引擎启动 **15** 次、引擎帧约 **24,000**、静音 30/30、账本 30 行无未结。
- 限制：W1 的闭环未测（不能把 W1/W2 的差直接归因于权重）；`s1` 是"已测最好"，不是"最好可能"。

## 69 V831-ROADMAP-01 阶段 1 第二十九轮（S2Z / OBS4，2026-11-27，训练端引擎 + 离线设计）

第二十九轮：左墙恢复守卫（从教师标签直接导出的控制规则）与 obs-v4 设计。**两次干预都更差**。

### 69.1 目录身份

| 目录 | 文件数 | 字节 | 说明 |
| --- | ---: | ---: | --- |
| `.tmp_v832_s2z/` | 约 90 | 约 20 MB | 1 次等价性 + 15 个种子的左墙恢复守卫 |
| `.tmp_v832_obs4/` | 约 6 | 约 60 KB | obs-v4 设计文档、原型、16 项测试 |

### 69.2 关键文件

| # | 路径 | 说明 |
| ---: | --- | --- |
| 1 | `tools/rl/v8_32_leftrecover_eval.py` | 新建守卫工具（未改 pinned） |
| 2 | `.tmp_v832_s2z/{results.json,verify_s2z.py,ledger.jsonl}` | 守卫实验 |
| 3 | `.tmp_v832_obs4/OBS4_DESIGN.md` | 112 行设计 |
| 4 | `.tmp_v832_obs4/proto_obs4.py` / `test_proto_obs4.py` | 原型 + 16 passed |
| 5 | `.tmp_v832_s2y/{results.json,verify_s2y.py}` | 尺寸已更正为 49,938 / 34,127 |

### 69.3 本轮结论（只作描述，不构成能力门）

- **左墙恢复守卫失败**：15 种子均值 **1,427.6** vs s1 **1,898.3**，配对 **−470.7**，P1 失败。
- **W2 的闭环判决（含统计）**：**1,560.07** vs **1,898.33**，配对 **−338.27**，t(14)=−3.41、**p=0.0042**、CI [−551.0, −125.5]；
  而 P2 成立（贴左钳位 0.37% vs 4.83%）→ **离开了角落但死得更早**。
- **综合机制结论**：**教师的局部标签以"教师自己的后续能力"为条件**；弱策略执行同样动作得不到同样保护，
  因此三次 DAgger 与两次守卫全部为负，「少动」在本能力水平下是承重的。
- **obs-v4 设计就绪**：`pos` 2 维、digest `6b360ccb…`、+8,240 参数、pos≡0 时严格等于 v3；
  但有两条边界：`--save-obs` 尚未记 px/py、pos 属**特权状态**（不能算像素收益）。

### 69.4 成本与限制

- 引擎启动 **31** 次、引擎帧约 **47,000**、静音 62/62、账本 62 行无未结。
- 限制：s1 基线由两批合并（10 + 5 个种子，已核对 28,475/1,898.333）；obs-v4 未施工。

## 70 清单状态登记（2026-11-27 13:25，主代理全量核对）

全量核对 `ASSET_SYNC.sha256`（1,973 行）的结果，如实登记于此：

| 项 | 数量 | 说明 |
| --- | ---: | --- |
| 哈希匹配 | 1,896 | — |
| **不匹配（会话前遗留）** | **74** | mtime 全为 **2026-11-15 / 11-23**，早于本会话首轮工单；集中在 `Document/Update/Alpha/v4.x/v8.x` 与 `tools/rl/v8b0…v8t0` |
| 不匹配（本会话产生） | **0** | 本轮把 `.tmp_v832_s2c/s2c.md`、`.tmp_v832_s2j/{results.json,s2j.md}` 三条刷新后归零 |
| 无法在 Windows 侧校验 | **3** | `/root/Touhou_RL/third_party/th08-rl/build/{rl_v3,rl_laser,rl_fork}/th08-modern`——WSL 路径，需在 WSL 内 `sha256sum -c` |

**处置**：74 条会话前遗留的不匹配**本轮不改写**（改写会让清单"看起来干净"而掩盖历史差异）；
它们与本研究线的资产无关（属 v4.x/v8.x 的旧证据与旧工具），且其 mtime 早于本会话，
故登记为**已知状态**而非"本轮修复项"。若研究端需要，可另开一次专门的清单重整工单。

## 71 V831-ROADMAP-01 阶段 1 第三十轮（V836 早窗分析 + 会话状态报告，2026-11-27，训练端离线）

第三十轮：**早期行为与存活的相关性分析**（纯离线），并出**会话级状态报告**。

### 71.1 目录身份

| 目录/文件 | 字节 | 说明 |
| --- | ---: | --- |
| `.tmp_v836_early01/` | 约 110 KB | 15 条轨迹 × 14 特征的关联分析（Spearman + 10,000 次置换 + Bonferroni + Mann-Whitney） |
| `Document/Update/Alpha/v8.31.0/SESSION_STATUS_2026_11_27.md` | — | 会话级状态报告（阶段 1 完成 / 阶段 2 未通过 / 下一步三选项） |

### 71.2 本轮结论（只作描述，不构成能力门）

- **11 个非恒定行为特征，0 个达到未校正 p<0.05**（最强 `x_sd` ρ=+0.443、置换 p=0.0998）；Bonferroni 后仍 0。
- **早期行为几乎确定性**：同一模型 15 个种子的**前 105 个决策逐位相同**；`vertical_frac = 0`（从不 N/S）、
  `distinct_classes = 9`、`bomb_count = 2`（首弹固定 decision 18）三个特征**零方差**。
- **因此运行间方差（sd 396.9）来自环境 RNG，不来自可辨认的行为差异**；
  这条与"n=15 才能分辨 500 帧"合起来，解释了为什么四次假设全部为负、三次干预都更差。
- **下一步建议（会话状态报告 §6）**：①把判分换成**离散到达事件**（立刻拉开信噪比，低成本）；②在线学习；③obs-v4（特权状态，不算像素收益）。

### 71.3 成本与限制

- 引擎启动 **0**、训练 0；新增磁盘约 110 KB。
- 限制：n=15 下 |ρ| 需 >0.5 才可能显著，本次最大 0.443 → **"未检出"不等于"无效应"**；
  环境种子同时决定弹幕与策略反应，**无因果结论**；`score_rate` 与存活机械耦合。

## 72 V831-ROADMAP-01 阶段 1 第三十一轮（V836 到达曲线，2026-11-27，训练端离线）

第三十一轮：用已有 69 条轨迹的 `frames` 检验"离散到达判分"是否比均值帧数更灵敏。**0 次引擎。**

### 72.1 目录身份

| 目录 | 字节 | 说明 |
| --- | ---: | --- |
| `.tmp_v836_reach/` | 约 2 KB | 到达曲线与 Fisher 精确检验脚本 |

### 72.2 本轮结论（只作描述，不构成能力门）

- **离散判分更灵敏（对大效应）**：1,750 阈值下 s1 **10/15** vs D2 **0/15（p=0.0002）**、W2 **2/15（p=0.0078）**、左墙守卫 **1/15（p=0.0017）**；
  1,500 阈值下 s1 12/15 vs 守卫 5/15（p=0.0253）。结论与配对 t 一致，但**不依赖分布形状**。
- **但对同配方复训无效**：同为纯 T04，s1 **10/15**、m1 2/3、s2 1/3、s3 **0/3**（p=0.07/0.53，n=3）。
- **建议**：把"到达 1,750 / 2,000 / 2,936"作为**主判分**替代均值帧数；但**同配方对照必须跑满 3 个训练种子**。
- 阶段 2 结论不变：**不通过**。

### 72.3 成本与限制

- 引擎启动 **0**；计算为一次 7 组 × 4 阈值的 Fisher 精确检验。
- 限制：s2/s3 只有 3 个环境种子；1,750 是本轮数据里最灵敏的点，不是普适最优阈值。

## 73 V831-ROADMAP-01 阶段 1 第三十二轮（V836 RWR 数据采集，2026-11-27，训练端引擎）

第三十二轮：为**回报加权回归（RWR）**采集 15 条带 obs 与剩余帧数的自然开局轨迹。

### 73.1 目录身份

| 目录 | 字节 | 说明 |
| --- | ---: | --- |
| `.tmp_v836_rwr/` | 663 MB | 12,265 条 (obs, 动作, 剩余帧) 三元组 + 15 条重建磁带 |

### 73.2 本轮结论（只作描述，不构成能力门）

- **12,265 个决策**、每决策三平面 + 所选动作 + `remaining_i = frames - frame_i`；`obs_rows == decisions` 15/15；`entry_verified` 15/15。
- 均值 1,699.3、sd 383.7、中位 1,548、1 条 `frame_cap`（censored）。
- **磁带来源偏差**：T04 只有 seed 10/11/12，1000–1014 无对应磁带 → 用 `.tmp_v420_data/T08/student_traces/` 重建（同引擎同 RESET、逐帧 HASH 通过）；
  代价是入口状态来自 **T08 B0 学生**，分布与历史 `s2o/s2p` 不同（中位 1,548 vs 1,939）。

### 73.3 成本与限制

- 引擎启动 **15** 次、静音 30/30、账本 30 行无未结、新增磁盘 **663 MB**。
- 限制：轨迹分布与历史不同（磁带来源偏差）；15 条共享决策 32–46，独立证据集中在决策 47 之后。

## 74 V831-ROADMAP-01 阶段 1 第三十三轮（RWRCHECK，2026-11-27，训练端离线）

第三十三轮：RWR 的**可学性预检**（纯离线），得到一条**识别性论证**。

### 74.1 目录身份

| 目录 | 字节 | 说明 |
| --- | ---: | --- |
| `.tmp_v836_rwrcheck/` | 约 0.5 MB | 分箱表、Kendall W、置换/打乱对照、识别性核验、13 项验收 |

### 74.2 本轮结论（只作描述，不构成能力门）

- **字面判据说"有稳定信号"（4/4 带 W=0.37–0.73、p=0.0001），但是假象**：
  实测 `rank(mean remaining) ≡ rank(−mean decision_frame)`（4/4 带最大差 **0.0**）→ **W 度量的是"动作出现的早晚"，不是"活多久"**。
- **打乱对照结构性失效**：同 seed 同带内 `remaining = frames − frame` 只差常数，常数不改排序 → 2,000 次打乱 |ΔW| = **0.000000**。
  按预设覆盖规则判 **"只有噪声"**。
- **根因**：12,265 行 `hash_pre` **全不同**（同状态不同动作对 = **0**）；15 条是 15 个不同 episode。
  带内回报方差 **97.6% 是"哪条轨迹"**、2.4% 是"带内何时"且 100% 由帧号解释 → **留给动作的可解释方差 = 0**。
- **结论**：**RWR 不训练**（最多作负对照）；离线动作价值估计在这套语料上**结构上不可识别**。
- 三处前提纠正（执行代理核实）：①本语料**没有逐决策坐标**（唯一 player 块是入场快照）；②`bomb` 分叉后 96.4% 恒为 1，不是动作维度；③seed 1006 撞 cap，剔除后结论不变。

### 74.3 成本与限制

- 引擎启动 **0**；计算为 4 带 ×（10,000 置换 + 2,000 打乱）+ 恒等式核验；验收 **13/13 PASS**。
- 限制：帧段只是关卡进度代理；跨种子同带非配对；置换检验的零假设是"动作↔带内时刻"。

## 75 V831-ROADMAP-01 阶段 1 第三十四轮（V836 SENS，2026-11-27，训练端引擎）

第三十四轮：**18 动作敏感性探针（cap 600）**——测"同一入口、同一 seed 下，固定动作是否影响存活"。

### 75.1 目录身份

| 目录 | 字节 | 说明 |
| --- | ---: | --- |
| `.tmp_v836_sens/` | 约 0.01 GiB | 18 个 `constant` cand × cap 600（入口 64、seed 10） |

### 75.2 本轮结论（只作描述，不构成能力门）

- **17/18 个固定动作都打满 664 帧**（cap 600 + 前缀 64），只有 **c8** 在 598 帧死亡；极差 **66 帧**。
- → **前 600 帧是"动作不敏感的安全窗"**，与回执 32 的"前 105 决策逐位相同"同源；
  也解释了"早期行为与存活无关联"与"离线留给动作的可解释方差 = 0"。
- 预设判据 P1（极差 ≥200）**不成立**；但探针只覆盖安全窗，已派发 **cap 2,000** 的第二探针在死亡时段重测。

### 75.3 成本与限制

- 引擎启动 **18**、引擎帧约 **11,900**、静音 36/36、账本 36 行无未结。
- 限制：只测了"持续单一动作"，未测"组合动作"；单 seed 单入口。

## 76 V831-ROADMAP-01 阶段 1 第三十五轮（V836 SENS2，2026-11-27，训练端引擎）

第三十五轮：**cap 2,000 的 18 动作探针**——在死亡时段重测动作敏感性。**推翻上一轮的"安全窗"结论。**

### 76.1 目录身份

| 目录 | 字节 | 说明 |
| --- | ---: | --- |
| `.tmp_v836_sens2/` | 约 0.01 GiB | 18 个 `constant` cand × cap 2,000（入口 64、seed 10） |

### 76.2 本轮结论（只作描述，不构成能力门）

- **动作有强信号**：18 个固定动作存活 **598–1,166 帧**，**极差 568 帧**（cap 600 时仅 66）。**P1（≥400）成立。**
- **"前 600 帧安全窗"是截断假象**：放开 cap 后 **18/18 全部在 598–1,166 内死亡**（0 个到 cap）。
- **P2 失败是主代理的阈值设计错误**：阈值"活过 1,200"落在可达范围（≤1,166）之外，Fisher 恒 p=1.0（算术恒等）；
  执行代理在首次启动前已把该缺陷写入 `prior.json`，未事后改判据。
- **结构**：focus 0/1 在 move∈{1..4} 上**完全同帧**，在 move∈{0,5,6,7,8} 上差 **+406～+466 帧**；位移 vs 不位移中位 762 vs 1,146（n=2，p=0.81）。
- **对在线学习的含义**：信号存在，但"持续单一动作"整体比策略差约 **700 帧**（策略 ~1,900 vs 常量族 598–1,166）
  → 长窗口常量探索是强 OOD 高致死区；探索须**短脉冲（≲100 帧）或带 teacher 回退**，判分窗口须 ≤1,166。

### 76.3 成本与限制

- 引擎启动 **18**、引擎帧约 **15,700**、静音 36/36、账本 36 行无未结。
- 限制：P2 阈值设计错误（已记录）；单 seed 单入口；只测"持续单一动作"，未测"短脉冲"。

## 77 V831-ROADMAP-01 阶段 1 第三十六轮（V836 PULSE，2026-11-27，训练端引擎）

第三十六轮：**短脉冲恢复性探针**——测"局部扰动是否可恢复"。这是在线学习可行性的最后一问。

### 77.1 目录身份

| 目录 | 字节 | 说明 |
| --- | ---: | --- |
| `.tmp_v836_pulse/` | 约 0.02 GiB | 1 次等价性 + 9 条脉冲（cand 0/3/14 × seed 1000/1001/1002） |

### 77.2 本轮结论（只作描述，不构成能力门）

- **短脉冲可恢复（预注册 P1 与 P2 都成立）**：决策 300–349（帧 600–699，约全程 4.4–7.9%）强制动作后交还策略，
  9 次均值 **1,749.8** vs 基线 **1,708.7**，逐次差均值 **+41.1**（中位 +122，6/9 不低于基线）。
- **与持续动作鲜明对比**：持续固定动作 **598–1,166 帧**（回执 37）→ **扰动时长是决定性变量**。
- **方向有强弱**：`cand0`（idle）**+360**、`cand14`（focus+S）**+164**、**`cand3`（E）−401**。
- **但单次不保证**：sd ≈ 723，两次约 −1,000（都在基线最强的 seed 1001）；剔除删失点后 8 次均值 1,648.0 vs 基线 1,763.5（−115.5）。
  → **"可恢复"成立、"更好"不成立**。
- **在线学习的最小可行参数**（§6）：探索 ≤100 帧、偏向 idle/focus+S、按状态门控、回报窗口 ≤1,166、判分用离散到达事件。

### 77.3 成本与限制

- 引擎启动 **10**（1 等价性 + 9 脉冲）、引擎帧 **17,018**、静音 20/20、账本 20 行无未结。
- 限制：只测了一个脉冲长度（100 帧）、3 个 seed、3 个动作；执行代理的工具来源偏差（改用 import takeover）已记录且等价性通过。

## 78 V831-ROADMAP-01 交接包与阶段状态（2026-11-27，训练端整理）

第三十七轮：出**交接包**（[`HANDOFF_V832.md`](../../../../Document/Update/Alpha/v8.31.0/HANDOFF_V832.md)），把 36 轮的回执、工具、语料、判分与施工参数整理成可直接接手的一包。

### 78.1 本轮结论（只作描述，不构成能力门）

- **冻结任务 2 个**（`S1N-MB1` 自然开局 / `S1N-MB2` v2 条件接管，后者卡 content `6e7da68a…`）。
- **工具链 8 个**（takeover / relabel / join / ablate / 三种守卫 / pulse），每个都有"关闭新功能 ≡ 原路径逐位相同"的等价性闸门与负对照。
- **语料 5 套**（T04 19,772 / WP05 3,116 / WP06+WP07 1,840 / RWR 12,265）。
- **已排除的路 6 条**（DAgger、预算、位置守卫、类别平衡、位置条件重加权、离线 RWR），每条都有量级与 p 值。
- **测量标准**：环境 sd 459.3、init 均值极差 514 → 分辨 500 帧需 n=15/臂；判分优先用离散到达事件；同配方对照 ≥3 个训练种子。
- **在线学习施工参数**：探索 ≤100 帧、偏向 idle/focus+S、按状态门控、回报窗口 ≤1,166、判分离散到达。
- **未解问题 4 条**：接管 Tier A 复训方差、obs-v4 未施工、`v431_audio_cfg_audit.py` 缺陷、清单 74 条会话前遗留不匹配。

### 78.2 阶段状态

| 阶段 | 状态 |
| --- | --- |
| 1 任务与数据 | **完成** |
| 2 像素学习收益 | **未通过**（唯一净收益：BC vs `pilot_final`，+1,167，2.6 sd） |
| 3–7 | 未开始 |

## 79 V831-ROADMAP-01 阶段 1 第三十八轮（V836 CF，2026-11-27，训练端引擎）

第三十八轮：**第一份干净的反事实数据**——同一状态、18 个单步动作。这是离线语料唯一缺失的数据形态。

### 79.1 目录身份

| 目录 | 字节 | 说明 |
| --- | ---: | --- |
| `.tmp_v836_cf/` | 约 0.03 GiB | 18 条同状态单步分支（入口 64、seed 1000、决策 300、`--pulse-len 1`） |

### 79.2 本轮结论（只作描述，不构成能力门）

- **反事实有强信号**：18 个动作的终局存活 **1,048–2,230 帧**，**极差 1,182**；P1（≥150）、P2（按 move 组间 ≥100，实测 344）、P3（≥3 个分支 ≥+100，实测 12 个）**全部成立**。
- **免费等价对照**：策略在该决策本身选 `cand 0`，故 `c0` = 无扰动，且**逐位复现基线**（1,270 帧、603 行决策 sha 相同）→ 18 条确为同一状态的分叉。
- **动作空间在该点高度冗余**：18 条后继轨迹只有 **11 种**（`c5/c8/c9/c14/c15/c16/c17` 从决策 305 起逐行相同，`c1/c10` 相同）→ 学习时必须把等值类**合并为一个样本**。
- **同向不同 focus 可反号**：`SE` 不聚焦 **+960** vs 聚焦 **−222** → 单步价值高度依赖上下文，须按完整 `cand` 建模。
- 限制：单 seed、单状态点；单条分支含大量混沌放大，不能直接当动作价值。

### 79.3 成本与限制

- 引擎启动 **18**、引擎帧 **26,310**、静音 36/36、账本 36 行无未结。
- 限制：单 seed、单状态点、单磁带；"18 个样本"实际只有 11 种结局。

## 80 V831-ROADMAP-01 阶段 1 第三十九轮（V836 CF2，2026-11-27，训练端引擎）

第三十九轮：**第二状态点（决策 400）的 18 分支反事实**——检验第一点的信号是否普遍。

### 80.1 目录身份

| 目录 | 字节 | 说明 |
| --- | ---: | --- |
| `.tmp_v836_cf2/` | 9.8 MB | 1 等价性 + 18 分支（入口 64、seed 1000、决策 400、`--pulse-len 1`） |

### 80.2 本轮结论（只作描述，不构成能力门）

- **信号复现且更强**：存活 **1,046–2,408 帧**，**极差 1,362**（第一点 1,182）；按 move 组间极差 **874**（第一点 344）；P1/P2/P3 全成立。
- **但跨点排序不一致**：逐 cand Spearman **ρ = −0.355**（p=0.149，n=18）→ **P4 不成立**。
  → **单决策动作价值是"状态点属性"，不能跨点外推**。最刺眼的是 SE：不聚焦在第一点 **+960**、第二点 **−192**；聚焦在第一点 **−222**、第二点 **+1,084**。
- **两点都有免费逐位对照**：第一点策略自选 `cand 0`、第二点自选 `cand 12`，两者都逐位复现 1,270 帧基线。
- **等值类不可迁移**：第一点 11 种后继轨迹、第二点 18 种全不同。
- **含义**：反事实采集配方打通；要学的是**状态条件价值函数**，因此必须多状态点采样。

### 80.3 成本与限制

- 引擎启动 **19**（1 等价性 + 18 分支）、引擎帧 **28,158**、静音 38/38、账本 38 行无未结。
- 限制：仅 2 个状态点；每分支只 1 次（未估重复噪声）；P4 点估计为负但未达显著，只能判"不复现"。

## 81 V831-ROADMAP-01 阶段 1 第四十轮（V836 CF3，2026-11-27，训练端引擎）

第四十轮：**相邻状态点（决策 301）的 18 分支反事实**——检验"排序不迁移"是不是距离效应。

### 81.1 目录身份

| 目录 | 字节 | 说明 |
| --- | ---: | --- |
| `.tmp_v836_cf3/` | 约 12 MB | 1 等价性 + 18 分支（入口 64、seed 1000、决策 301、`--pulse-len 1`） |

### 81.2 本轮结论（只作描述，不构成能力门）

- **相邻 1 决策的排序也不迁移**：ρ(301,300)=**0.238**（p=0.342）、ρ(301,400)=−0.246、ρ(300,400)=−0.355 —— 三者都不显著且同量级。
- **但点 301 的结构极其规整**：`SE/S/SW`（含 focus 版）**全部恰好等于基线 1,270**，其余 12 个动作几乎全部 **2,272**（N 为 2,432）。
  → 90% 的方差由"**是否向下**"解释，符合"玩家在底边、向下输入被钳制"的物理约束。
- **策略在 301 自选 `cand 6`（SW，被钳制）**，比任何非向下动作少活约 **1,000 帧** —— 本项目第一个有明确"该换成什么"的具体缺陷。
- **解释分歧如实记录**：执行代理判"价值混沌、状态条件价值函数几乎不可学"；主代理指出规整结构与混沌说法有张力，
  并给出可检验形式：**按"动作是否被位置钳制"重编码后再看排序是否稳定**（未做）。
- 两方都同意的一点：策略需要知道自己**在哪里、哪些动作被挡住**（指向 obs-v4 的位置标量）。

### 81.3 成本与限制

- 引擎启动 **19**（1 等价性 + 18 分支）、引擎帧 **34,730**、静音 38/38、账本 38 行无未结。
- 限制：n=18 且并列值多，ρ 的 CI 很宽（含 0.5）；三点都在同一条轨迹上；"按钳制重编码"检验未做。

## 82 V831-ROADMAP-01 阶段 1 第四十一轮（更正，2026-11-27，主代理离线查证）

第四十一轮：**主代理自我更正**——回执 41 §3 提出的"被位置钳制"解释被坐标数据否证。

### 82.1 本轮结论（只作描述，不构成能力门）

- `v8_32_pulse_eval.py` 的决策日志**不含坐标**（键为 bombs/cand/deaths/decision/end/enemies/events/frame/lives/mask/masks/repeat/schema/score/seed/stage）。
- 用同类运行的坐标代理（`.tmp_v832_s2o/seed1000/decisions.jsonl`，含 `player.position`）：
  决策 301 玩家在 **[14.0, 403.8]**，离底边 **28.2 px**、离左墙 **6 px**。
- 向下 2 帧只走 5.66 px → **不构成钳制**；向左会被钳到 x=8，但**钳制的是"左"，而 301 上等值的是"下"**。
  → **"向下被钳制"不成立。**
- 因此：**301 的六个向下分支为何精确等于基线，仍未解释**；执行代理的"价值混沌"解释也**未被证实**。
- **可观测性缺口**：反事实实验缺逐决策坐标。补齐方法明确（把 `v8_32_relabel.py` 的 `PLAYER` 读法搬进 pulse 工具的日志），本轮未做。

### 82.2 限制

- 坐标来自 `v400_student_eval` 从帧 0 起跑的运行，而 301 分支来自 takeover 从入口 64 起、T08 磁带重建的运行——**不是同一条轨迹**，只能当量级参考。

## 83 V831-ROADMAP-01 终稿登记（2026-11-27 15:30，训练端）

本会话 40 个自动续行轮次已用尽，四个收尾文档定稿：

| 文档 | 作用 |
| --- | --- |
| [`HANDOFF_V832.md`](../../../../Document/Update/Alpha/v8.31.0/HANDOFF_V832.md) | 交接包：可用件 / 已排除的路 / 测量标准 / 施工参数 / 反事实配方 / 复现命令 / 未解问题 |
| [`SESSION_STATUS_2026_11_27.md`](../../../../Document/Update/Alpha/v8.31.0/SESSION_STATUS_2026_11_27.md) | 会话总账：第 1–41 轮的全部结论、40 轮总账表、下一任第一周清单 |
| [`STAGE2_ASSESSMENT.md`](../../../../Document/Update/Alpha/v8.31.0/STAGE2_ASSESSMENT.md) | 阶段 2 逐条结算（判据 1/2/4 通过、判据 3 不通过 → 阶段 2 不通过） |
| [`V831_ROADMAP_STAGE_PLAN.md`](../../../../Document/Update/Alpha/v8.31.0/V831_ROADMAP_STAGE_PLAN.md) | 阶段状态与下一步三选项 |

**最终状态**：阶段 1 **完成**；阶段 2 **未通过**；阶段 3–7 未开始。
**唯一净收益**：BC 对 `pilot_final` 的 731 → 1,898.3 帧（**2.6 sd**）。
**最重要的三个负面结论**：①教师标签以教师自身能力为前提；②离线语料无同状态反事实对、动作价值结构上不可识别；③运行间方差来自环境（sd 459.3），n=3 无法分辨任何 <1,410 帧的效应。
**最重要的三个正面产出**：①两个冻结任务 + 8 个带等价性闸门的工具；②三点 × 18 动作的反事实数据（极差 1,162–1,362，每点自带逐位对照）；③一套可施工的在线探索参数（≤100 帧、偏 idle/focus+S、门控、回报窗 ≤1,166、离散到达判分）。

**留给下一任的第一个问题**：`v8_32_pulse_eval.py` 的决策日志缺逐决策坐标（补齐约 30 行 + 3 次启动），
它是判定"价值低维可学 vs 混沌不可学"的唯一前提。

## 84 V831-ROADMAP-01 阶段 1 第四十二轮（V836 PX，2026-11-27，训练端引擎）

第四十二轮：给反事实工具**补逐决策坐标**，并测出三个状态点的真实站位。**"钳制"解释被彻底否证。**

### 84.1 本轮结论（只作描述，不构成能力门）

| 点 | 位置 [x, y] | 到底边(432) | 到左墙(8) | 向下 2 帧 | 向左 2 帧 |
| ---: | --- | ---: | ---: | --- | --- |
| 300 | [31.31, 415.03] | 16.97 px | 23.31 px | 不钳制 | 不钳制 |
| 301 | [27.31, 415.03] | 16.97 px | 19.31 px | 不钳制 | 不钳制 |
| 400 | [9.37, 432.00] | 0.00 px | 1.37 px | **钳制** | **钳制** |

- **301 处任何方向都不会被钳制**（2 帧最大位移 8.0 px）→ **"钳制"解释否证**。
- **400 处玩家顶在左下角**（x=9.37、y=432.0），但三个"被钳制"的动作结果仍不同（SE 1,078 / S 1,528 / SW 1,348）→ **钳制存在但不决定价值**。
- **仍未解释**：301 上六个向下分支为何精确等于基线 1,270。
- **新线索**：cf3 观察到 8 个分支同落 2,272、6 个同落 1,270，但**只有 c6 与基线逐位相同** → 更像**轨迹吸收**，而不是钳制。

### 84.2 工具改动

- `tools/rl/v8_32_pulse_eval.py`：每决策行新增 `player{position, focus, invuln_state, player_state, frame, frame_matches, read_phase="before_action", ...}`；
  用出厂解码器 `parse_player_reply`，帧号不符即 fail-closed；新增 `--log-player`（默认开）/`--require-player`（默认关）。
- 等价性：`--pulse-len 0` 复跑 seed 1000 → **1,270 帧、603 行决策 sha 逐位相同**。

### 84.3 成本与限制

- 引擎启动 **3**、静音 6/6、账本 6 行无未结。
- 限制：三点同一 seed、同一条轨迹家族；"轨迹吸收"假设未验。

## 85 V831-ROADMAP-01 阶段 1 第四十三轮（V836 CORNER，2026-11-27，训练端引擎）

第四十三轮：**角吸收实测**——1 次启动把"301 六个向下分支为何等于基线"从推断变成实测。

### 85.1 本轮结论（只作描述，不构成能力门）

- 决策 302 起 `cand5` 确实向下移动：301→302 步进 **(0.000, +4.000)**（基线为 (−2.828, +2.828)）→ **C2 成立**。
- **决策 306 起逐位回到角点 (8.0, 432.0)**（raw px=`41000000`、py=`43d80000`，与基线同位）→ **角吸收成立**。
- 锁定口径 C1（决策 305–310 全在 ±0.5 px 内）**FALSE**：唯一失败点是 **305 处 x=12.000015（差 4.0 px）**；
  **306–310 全部精确** → 事后（post-hoc）口径 **TRUE**。执行代理如实并列报告两者，未改判据。
- `c5` 的 `frames=1270`、603 决策、`stop=legal_death` 与 cf3 完全一致。
- **最终机制（已实测）**：301 处的强制动作**真的生效**，但 **5 个决策后状态被墙角重新吸收** →
  六个向下分支终局相同。**不是钳制在 301，而是角点在约 5 个决策后把扰动吃掉。**

### 85.2 对下一步的含义

- 反事实分支的"等值类"有一部分是**吸收**造成的（扰动被物理边界吃掉），不是"动作等价"。
- 因此偏好/价值学习必须**排除或降权"会被吸收"的分支**，否则会把"无效扰动"当成"低价值动作"。
- 工具现在已能逐决策读坐标（`--log-player`），可以直接检测"分叉后 k 帧是否重汇"。

### 85.3 成本与限制

- 引擎启动 **1**、静音 2/2、账本 2 行无未结。
- 限制：只验了 `cand 5` 一个分支；"重汇帧数"只测了这一个点。
