# W00 运行时探针：本机真实控制通路可用性

日期：2026-11-15，Asia/Shanghai。执行者：主代理（AGENTS.md §1 例外：环境侦察与汇报；所有业务实现由执行代理完成）。
本文件只记录**运行时事实**，不是能力证据，不改变任何 G430/G410 门状态。

## 1 结论摘要

| 项 | 结果 | 证据 |
| --- | --- | --- |
| 本检出源码可在 WSL 运行时直接导入 | 通过 | §2 探针 1，退出码 0 |
| rl_v3 引擎可启动并完成共享内存握手 | 通过 | §2 探针 1b，退出码 0 |
| 真实 RESET 进入 stage 1 可玩状态 | 通过 | §2 探针 2、3，`ready=1`，`setup_frames=168` |
| 真实 STEP 推进帧 | 通过 | 120 决策 = 240 帧 |
| 像素抓取 | 通过 | PNG 921,615 字节 |
| 端到端吞吐（单实例、软件渲染） | **126.3 帧/秒** | 240 帧 / 1.901 秒 |

**重要：端到端吞吐 126.3 fps 明显高于计划沿用的历史值 49.4 fps。** 按 PLAN_SPEC `budgets.replace_with_measured_throughput=true`，首轮预算核算应改用实测值；这属于 ST00 应交付的吞吐事实，不改变任何预算上限。

## 2 探针命令与原始输出

所有命令在 WSL `Ubuntu` 内、**以本检出为源码根**运行（`/mnt/d/WorkShop/Touhou_RL`）。
必须强调：`/root/Touhou_RL` 是**另一份陈旧检出**（commit `d55c26a`，v4.0.0 期，缺 `touhou_rl/v420/`），
**不能**作为源码根；只有引擎二进制与游戏数据来自 `/root`。

### 探针 1：源码可导入性

```bash
cd /mnt/d/WorkShop/Touhou_RL
PYTHONPATH=/mnt/d/WorkShop/Touhou_RL /root/v400-venv/bin/python .tmp_v430_probe_env.py
```

退出码 0，输出要点：

```text
python 3.10.12
numpy 2.2.6
torch 2.14.1+cu130 cuda_available True
N_CAND(v400 18 cand) 18
v420.contracts OK -> /mnt/d/WorkShop/Touhou_RL/touhou_rl/v420/contracts.py
CONTRACT_VERSION v420-contract/1
v420.eval OK -> /mnt/d/WorkShop/Touhou_RL/touhou_rl/v420/eval.py
v400.gym.env OK -> /mnt/d/WorkShop/Touhou_RL/touhou_rl/v400/gym/env.py
v400.gym.replay_starts OK -> /mnt/d/WorkShop/Touhou_RL/touhou_rl/v400/gym/replay_starts.py
```

### 探针 1b：引擎握手（shadow 角色，无窗口）

```bash
PYTHONPATH=/mnt/d/WorkShop/Touhou_RL /root/v400-venv/bin/python .tmp_v430_probe_engine.py shadow
```

退出码 0：

```text
started ok; pid = 107674
shm_name = dsh_v430_probe
HASH -> ('undefined', 0)
DEBUG keys ['_verb', 'curState', 'difficulty', 'frames', 'globals', 'isDemo', 'isReplay', 'setupState', 'shot', 'stage', 'subthread', 'threads']
READBACK -> {'_verb': 'OK', 'readback': '0', 'frames': '0', 'micros': '0'}
stats ProtocolStats(commands=5, frames=0, wall_seconds=1.218, replies={'OK': 5})
```

### 探针 2：真实 RESET（env 角色，Xvfb）

```bash
PYTHONPATH=/mnt/d/WorkShop/Touhou_RL xvfb-run -a /root/v400-venv/bin/python .tmp_v430_probe_engine2.py
```

`RESET` 成功（该探针随后在 `HarnessClient.state()` 上崩溃，见 §3 缺陷 1）：

```text
RESET -> {'_verb': 'OK', 'reset': '1', 'ready': '1', 'setup_frames': '168',
          'frames': '0', 'stage': '1', 'difficulty': '1', 'team': '0'}
reset wall s 0.89
after reset HASH -> ('3eafc035dc4fe6a9', 0)
```

### 探针 3：受支持封装 `Th08Env` 全通路

```bash
PYTHONPATH=/mnt/d/WorkShop/Touhou_RL xvfb-run -a /root/v400-venv/bin/python .tmp_v430_probe_engine3.py
```

退出码 0：

```text
reset wall s 0.877
reset hash 3eafc035dc4fe6a9 frame 0 events ['reset']
obs (480, 640, 4) uint8
state fields 25 sample {0: 0, 1: 1, 2: 0, 3: 2, 4: 0, 5: 0}
stepped decisions 120 frames 240 wall s 1.901 fps 126.3
last obs (480, 640, 4) uint8 reward 0.0 terminated False truncated False game_over False
after state {0: 240, 1: 1, 2: 0, 3: 242, 4: 0, 5: 0, 6: 0, 7: 1073741824, 8: 1077936128, 9: 0}
PIXELS reply keys ['_verb', 'pixels', 'nonzero', 'checksum', 'path'] wall s 0.002
png exists True bytes 921615
stats ProtocolStats(commands=125, frames=240, wall_seconds=8.218, replies={'OK': 125})
```

`StageStart(stage=1, difficulty="normal", team=0, seed=1000)`，动作 `cand=0, bomb=0`（静止 + 射击），
`decision_frames=2`。观测为 RGBA 640×480 uint8。

## 3 探针暴露的既有缺陷（不属于 W00 白名单，仅登记）

1. **`HarnessClient.state()` 不能处理浮点字段**：25 个跟踪字段中，`f7`/`f8`（残机/bomb）等以 IEEE-754 位模式以十六进制字符串返回（例：`'43c00000'`=384.0、`1073741824`=2.0、`1077936128`=2.5）。`protocol.py` 的 `state()` 无条件 `int(value)`，遇到十六进制会抛 `ValueError: invalid literal for int() with base 10`。`Th08Env` 用 `_f32_bits()` 规避了这一点，因此**受支持入口可用，裸 `HarnessClient.state()` 不可用**。
   影响：任何新代码若直接调用 `HarnessClient.state()` 会崩溃。W01 的契约层必须以 `Th08Env` 或自行解码浮点位模式，不能假定 25 个字段都是十进制整数。

2. **`/root/Touhou_RL` 与 `/mnt/d/WorkShop/Touhou_RL` 是两份不同检出**：前者停在 `d55c26a`（无 `v420`），后者是本检出。引擎二进制只在 `/root/Touhou_RL/third_party/th08-rl/build/`（Windows 侧 `third_party/th08-rl/build/` 不存在，属 gitignore 资产）。因此**语义是"源码用本检出、引擎与数据用 /root"**，任何配置都必须把这三者分开绑定，不能写成单一 root。

3. **`env` 角色需要 X display**（`shadow` 角色不需要）。探针用 `xvfb-run -a` 通过。W06 的 Windows 真实会话与 W01 的像素转场验证都需要渲染路径，必须显式提供 display，不能假定 headless 可用。

## 4 并发争用：一个已跑 10.7 小时的 v4.2.0 A1 扩容仍在占用 2–3 核

2026-11-15 约 11:15 只读查进程树时发现，本机除本次工作外还有**一个仍在活跃运行的 v4.2.0 任务**：

```text
PID 392  bash .tmp_v420_expand/run_all.sh                    started 2026-11-15 00:32:28
PID 397  bash .tmp_v420_expand/run_worker.sh w2 2 5 8 11 ...
PID 415  bash .tmp_v420_expand/run_worker.sh
PID 109536 bash .tmp_v420_expand/run_state.sh 530
PID 109544 timeout --signal=TERM --kill-after=20 580 xvfb-run -a ...
PID 109570 /root/v400-venv/bin/python .tmp_v420_expand/a1_expand.py \
           --config configs/v420/a1-safe-set-expand.json
PID 109572 / 109646 / 109731   rl_v3 引擎 ×3，各 ~94% / 94% / 47% CPU
```

事实与判读：

1. **起始时间 00:32:28，截至 11:15 已运行约 10.7 小时**；每个状态上限 `timeout ... 580` 秒，当前处理 state 530。
   它**不是**本次会话启动的，也不是任何执行代理启动的（任务书均禁止启动引擎）。
2. 它**持续占用 2–3 个 CPU 核**。因此本文件 §1 的 **126.3 fps 是在该争用存在的情况下测得的**，
   应视为**下限**而非干净基线；任何后续吞吐测量都必须先记录当时的并发引擎数。
3. 它**不会改坏已登记资产**：写入范围是 `.tmp_v420_data/T08/expand/**`，而 `ASSET_SYNC.sha256`
   的 46 条里该子树**一条都没有**（T08 下唯一注册资产是 `T08/setsup/checkpoints/a1set_seed0_step5000.pt`，
   不在 expand 子树内）。
4. 这与 v4.3.0 文档"没有接管 v4.2.0 在途任务"并不矛盾，但它意味着
   **"v4.2.0 在途任务已死/已暂停"这个印象是错的**——它活着、在写盘、在烧 CPU。
   与 W00 源码身份那一半记录的 `setsup` 测试 skip 原因（"扩容语料在途，750 中仅 merge 20，
   `merge status='failed'`"）互相印证。
5. 主代理**未杀、未暂停、未接管**该进程：`EXECUTION_PLAN.md` §3 与 AGENTS.md 都明确禁止
   动不属于本次任务的进程。该决定留给用户。

### 4.1 更正：它**不是**僵尸任务，而是进度 72% 的合法长跑（11:23 复核）

我最初把它当成"跑了一夜没人管的残留"。进一步只读查证后**必须更正这个印象**：

| 证据 | 实测 |
| --- | --- |
| `run_all.sh` 的规模来源 | 从 `states_frozen.json` **读取**状态总数（脚本注释明写"不得硬编码 750"，只以前缀冻结清单为准） |
| 三个 worker 的分配 | `w0/w1/w2` 各约 **251** 个状态 → 合计 **≈750** |
| 已完成 | `state_*.log` **541** 个、`shards.tsv` **538** 条 |
| 进度 | **≈541 / 750 ≈ 72%** |
| 活跃度 | 11:23:05–11:23:15 的 10 秒内，状态 532/535/540/536/539 相继完成——**在推进，不是卡死** |
| 断点续跑 | `run_all.sh` 会跳过 `status=pass` 的既有 shard，重跑只为未完成状态付费 |

结论修正：

1. 它是 **v4.2.0 遗留的、正在正常推进的**扩容任务，**不是**无人管理的僵尸。
2. 因此**不建议杀掉它**——它已到 72%，杀掉等于白烧掉前面 10 小时。
3. 但它确实是**并发争用的唯一来源**，会持续让本路线每次引擎运行慢约 3%。合理处置是
   **等它自然跑完（预计数十分钟量级）再重测吞吐**，而不是中途终止。
4. W00 源码身份那一半记录的 `setsup` skip（"750 中仅 merge 20"）是**更早时点**的快照；
   现已推进到 538，两者不矛盾，但**引用时必须注明时点**。

**给 W03/W04 排程的建议**：在它完成前，只做不占引擎的工作（观测编码、谱系、单测、文档）；
需要吞吐数字的预算核算，等它结束后再测一次干净值。

### 4.2 完成时间实测（更正我先前"数十分钟"的估计）

2026-11-15 11:38–11:40 做了一次 120 秒定间隔采样：

| 项 | 值 |
| --- | --- |
| 计划状态总数（`states_frozen.json`） | **750**（确认，不是估算） |
| 已完成 | **553**（state log）/ **550**（merged） |
| 进度 | **73.7%** |
| 速率 | **3 个状态 / 120 秒 = 1.5 个/分钟** |
| 剩余 | 197 个 |
| **预计还需** | **约 131 分钟 ≈ 2.2 小时** |
| load average | 3.56（20 核） |
| 活跃 worker / 引擎 | 3 / 3 |

**这里更正我先前的判断**：我在 §4.1 写过"预计数十分钟量级"，那是基于某 10 秒窗口内 5 个状态完成的观察——
那批恰好是短状态。按 120 秒稳定采样，真实速率只有 1.5 个/分钟，**剩余时间约 2.2 小时**，不是几十分钟。

含义：
- 约 3% 的引擎 slowdown 会**再持续约 2 小时**，不是"马上结束"。
- 任何依赖端到端帧率的预算核算，要么等它结束，要么明确标注"争用条件下的下限"。
- **不建议**为了跑得快去杀它：它已完成 73.7%，且 `run_all.sh` 的断点续跑只为未完成状态付费。

### 4.3 再次更正：速率随"剩余长状态"显著下降，ETA 约 6.2 小时

我在 §4.2 给的"约 2.2 小时"仍然偏乐观。继续观测 merged 计数后：

| 时刻 | merged | 区间速率 |
| --- | ---: | ---: |
| 11:40:00 | 547 | — |
| 11:42:00 | 550 | **1.50 /min**（我 §4.2 用的就是这个窗口） |
| 11:53:30 | 556 | **0.52 /min** |

剩余 194 个状态，按**最近**速率 0.52/min → **约 372 分钟 ≈ 6.2 小时**。

原因很清楚：`run_all.sh` 大致按序推进，**先完成的是短状态**。§4.2 那个 120 秒窗口恰好落在
一批短状态上，因此把速率高估了近 3 倍。**剩余的都是跑满预算的长状态。**

进程仍然健康（`engines=3`、`workers=3`、`run_all=1`、最新日志 11:53、load 2.82/20 核、无 partial 残留）。

排程含义（重要）：
- **不要指望几小时内解除争用**。凡需引擎的工作应按"争用常驻"设计。
- 纯代码切片（模块、单测、manifest、序列层）不受影响，应继续优先安排。
- 需要干净吞吐数字的测量，须等它结束或显式标注"争用下限"。

**这是我对同一件事的第二次更正**（先"数十分钟"、再"2.2 小时"、现"6.2 小时"）。
两次都源于用**短窗口**外推，而工作量分布是长尾的。教训：**报 ETA 要给出观测窗口与速率，并说明剩余任务的分布假设。**




- ST00 的"端到端帧/s"必须标注并发条件，正式预算核算前应在**无争用**条件下重测一次。
- W01c 的真实重放会因争用变慢；已要求执行代理把受限运行标为 `cost_contention`（资源不完整，exit 4 语义），
  **不得**标成 death/game_over，也不得为迁就争用修改判据。
- 若该扩容还要跑数小时，会持续污染 W03/W04 的吞吐与预算估计。

**已实测的争用代价与磁盘占用**（2026-11-15 11:19 前后）：

| 项 | 实测 |
| --- | --- |
| 扩容目录总占用 | `.tmp_v420_data/T08/expand` = **132 MB** |
| 增长速率 | 60 秒采样 **+689,956 字节**（约 0.7 MB/分钟） |
| D: 可用空间 | **1.2 TB**（41% 已用） |
| 争用对重放的实际影响 | 正对照 wall **125.9 s**、负对照 **143.0 s**，比 v4.2.0 参考慢 **3.6% / 3.0%**，未触发任何超时 |
| 负载 | load average 2.5–2.7（20 核） |

结论：**磁盘不是约束**（`新增持久资产 ≤40 GiB` 的额度远未触及），
**CPU 才是**——它让每次引擎运行慢约 3%，并会让 W03/W04 的吞吐估计偏低。


## 5 未做（not_run）

- 未做成功/失败**回放**重放（W01 交付项，需先实现契约入口）。
- 未跑任何训练、评测、反事实分支。
- 未测量 Windows 侧抓帧/发键时延（W06 交付项）。
- 未验证 `rl_laser` 构建；本次只用了 `rl_v3`。

## 6 对工作包的影响

- **W01 的真实腿在本机可执行**，无需等资产主机；验收命令应以 `/mnt/d/WorkShop/Touhou_RL` 为源码根、`xvfb-run -a` 为显示前缀。
- W01 必须覆盖缺陷 1（浮点字段解码）与缺陷 3（所有者/显示前提），否则"成功/失败回放"无法可信。
- 端到端吞吐以 126.3 fps 为当前实测起点，正式训练前须按实际并发实例数重测，不能把单实例数字当 4 实例吞吐。
