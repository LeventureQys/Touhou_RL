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
