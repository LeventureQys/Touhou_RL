# V831-ROADMAP-01 阶段回执 08：S1N-MB2 冻结、checkpoint 臂打通与"预算饱和、数据不足"

**工单**：[`LONG_TERM_DEVELOPMENT.md`](LONG_TERM_DEVELOPMENT.md)（V831-ROADMAP-01）· 承接 [`TRAINING_HOST_REPLY_07.md`](TRAINING_HOST_REPLY_07.md)
**主机代号**：轩神开播了（Windows 11 家庭版 / i5-14600KF；WSL2 `ubuntu`，20 vCPU，23 GiB，RTX 4070 SUPER）
**时间及时区**：2026-11-27 02:19 – 02:44 +08:00（本回执全部时间为 +08:00）
**执行**：DSH 主代理派发 3 个执行代理（加长训练、checkpoint 臂、双层真机评测）；主代理独立复跑三份验收（51 / 48 / 51 项）并冻结任务卡
**检出**：`D:\WorkShop\Touhou_RL`，HEAD `1233598796b1b6f1d734950a4113af917da7b844`（分支 `v400`）
**引擎启动**：**6 次有效 + 1 次预检拒绝（0 帧）**

---

## 0 结论摘要（四条）

1. **`S1N-MB2` 已冻结，checkpoint 臂打通并在真机上跑出第一份学生结果。**
   卡 `content_sha256 = 2eee50461abd071049c461278f047cc5c26e0ca73a2f09cf737e2ae67cd2423c`
   （本轮先按三条前置验证冻结为 `70032a67…`，随后按 §4 的实测把"Tier A 不可判 + Tier B 无区分力 + checkpoint 臂结果"补进卡内重冻，故 content 变更为当前值）；
   runner 扩到 2,455 行（48 项离线测试，主代理复跑 48 passed）；真机 3 根全部 `entry_verified=true`、SHM 像素源正常（`readback=1`、`problems=[]`）、`policy_calls` 934 / 497 / 495。
   策略从入口 2,632 起 **存活 1,868（跑满 cap）/ 994 / 990 帧**，恒动作臂只有 **630 / 674 / 780**。
2. **但 Tier A 目前判不了，而且本卡的 Tier B 不具区分力。**
   checkpoint 臂的逐决策日志只记 `events`、**没有 `enemies` 读数**，所以 `boss_life` 6000→0 无法判定（`not_determinable`）；
   同时恒动作臂也在 3,262 / 3,412 / 3,306 才死——**都晚于 2,936**，于是 Tier B（帧 2,936 存活）在恒动作臂上也是 **3/3**，对本卡没有区分力。
   → 下一轮必须补逐决策 `enemies` 采样，并把本卡的成功口径换成 Tier A 或"入口后存活帧数"。
3. **训练预算已饱和。** 同一 episode（T04 seed10，6,592 行）上把 2,000 步加到 8,000 / 16,000：
   donor top-1 0.5933 → **0.6211**（8k 顺序）→ **0.6359**（16k），8k 平衡臂 0.6340；
   但 **dominant 不再下降**（顺序 0.3638 → 0.3736 → 0.3742，贴近先验 0.3736；平衡臂从 0.2057 回升到 0.3947），
   且 16k 的 fit top-1 = **1.0000**（6,592 行全记住）。→ 预算不是瓶颈，**episode 数量**才是（当前只用了 1 条 episode）。
4. **闭环与离线同向、同幅：加长没有带来可判定的收益。**
   L1（8k 顺序）自然开局 1,457 / 2,895 / 1,401 → 均值 **1,917.7**，对 2,000 步的 1,845.3 只 **+72.3 帧（+3.9%）**，
   逐种子 **−47 / +1,087 / −823**，n=3 无 CI → **不构成效应**；Tier B 仍 0/3。

---

## 1 身份

| 项 | 值 |
| --- | --- |
| 引擎 | `rl_v3/th08-modern` sha256 `691bfd77…a568ef6`（6 次启动 + 1 次预检全部一致） |
| 冻结卡 | `.tmp_v832_takeover/task_card_mb2.json`，content `2eee50461abd0710…cd2423c`（三验后首冻 `70032a67…`，按 §4 实测补录后重冻）；人读版 `task_card_mb2.md` |
| 判别器（卡生成器） | `.tmp_v832_takeover/make_task_card_mb2.py`，13,750 B，sha256 `47a0e265697eddf8…` |
| 接管 runner | `tools/rl/v8_32_takeover.py`，2,455 行，sha256 `0b3ab4ebc5668ec4…`（改前 1,565 行 `4894363e…`） |
| runner 测试 | `tests/test_v8_32_takeover.py`，1,732 行，sha256 `6d9b5285dbdefc28…`（48 项） |
| 加长训练 checkpoint | `ckpt_l1.pt` `d8f8912807a4bff5…`（8k 顺序）/ `ckpt_l2.pt` `bba0d3438151a472…`（8k 平衡）/ `ckpt_l3.pt` `312410d09e483619…`（16k 顺序），均 canonical、10,468,615 B |
| 静态身份 | 三次加长训练与 2,000 步版本共用同一 init（`6d6efb97…`）与前 2,000 步 batch 序列（`batch_indices` 逐位相同、前 2,000 loss `max|Δ|=0`） |

---

## 2 命令与退出码

| # | 命令 | rc | 结果 |
| ---: | --- | ---: | --- |
| 1 | `python -B .tmp_v832_s2d\verify_s2d.py`（主代理复跑） | **0** | **51/51 PASS** |
| 2 | `pytest tests\test_v8_32_takeover.py`（主代理复跑） | **0** | **48 passed in 5.12 s** |
| 3 | `python .tmp_v832_s2c2/verify_s2c2.py`（主代理复跑，WSL） | **0** | **51/51 PASS** |
| 4 | `python -B .tmp_v832_takeover\make_task_card_mb2.py` | **0** | 冻结卡 `content 70032a67…` |

**A 批（S1N-MB2，checkpoint 臂，ckpt_l1）**

| seed | rc | entry_verified | entry_hash | frames | 交还段 | stop | deaths | policy_calls | 墙钟 |
| ---: | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| 10 | 0 | true | `e2a84b01b1aadef4` | **4,500** | 1,868（跑满 cap） | `frame_cap` | 0 | 934 | 42.669 s |
| 11 | 0 | true | `9edbdd05aa4837a9` | 3,626 | 994 | `legal_death` | 1 | 497 | 32.793 s |
| 12 | 0 | true | `b8fdb7795775901f` | 3,622 | 990 | `legal_death` | 1 | 495 | 31.776 s |

**B 批（自然开局，ckpt_l1，`S1N-MB1` 尺）**

| seed | rc | frames | deaths | score | dominant | Tier B |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1000 | 0 | 1,457 | 1 | 28,090 | cand 7（23.5%） | 否 |
| 1001 | 0 | 2,895 | 1 | 67,650 | cand 7（28.9%） | 否（差 41 帧） |
| 1002 | 0 | 1,401 | 1 | 26,209 | cand 7（31.1%） | 否 |

---

## 3 额度

| 额度 | 实际 |
| --- | ---: |
| 引擎启动 | **6**（A 3 + B 3）+ **1 次预检拒绝（未起引擎、0 帧）** |
| 引擎帧 | **17,501**（A 11,748 + B 5,753） |
| 引擎墙钟 | A 批 **107.238 s**；B 批未单列（`s2c2/ledger.jsonl` 为事后重建，`wall_s` 字段缺失，见 §7-2） |
| GPU 训练墙钟 | **394.45 s**（L1 98.553 + L2 98.215 + L3 197.682） |
| 新增磁盘 | 187,944,268 B ≈ **0.175 GiB** |
| 静音 | **14/14** 读数全 `0 0 0 0`（含 1 次预检对）；`SDL_AUDIODRIVER=dummy`；源树 `/root/v400-data/game` 指纹前后一致 `0a6534e4…` |
| 账本 | 12 行 = 6 reserve + 6 settle，无未结 |

---

## 4 A 批：checkpoint 臂的首次真机结果

| 臂（同入口 2,632、同 cap 1,868） | 入口后存活帧数 | 是否跑满 cap |
| --- | ---: | --- |
| **checkpoint（ckpt_l1，8k 顺序）** | **1,868 / 994 / 990** | 1/3 |
| 恒动作 `cand 12` | 630 / 780 / 674 | 0/3 |
| 磁带续放（非策略） | 1,868 ×3 | 3/3 |

- **策略确实在玩**：`policy_calls` 934/497/495，动作 dominant 在日志中可见；`BOMB_STARTED` 出现 2 次（会用 bomb）；
  score 从入口的 40,829 涨到 99,820（seed10），`ENEMY_REMOVED` 38 次、`ITEM_COLLECTED` 91 次。
- **3/3 都活着进入 boss 窗口**（∈[2,936, …]），2/3 在窗口内死亡（3,620 / 3,624），1/3 存活到 4,500。
- **Tier A 判不了**：日志只有 `events`，没有 `enemies.boss_life`；`result.json` 只有入口单页 `entry_state.enemies`（`boss_count=0`）。
- **Tier B 没有区分力**：恒动作臂也死在 2,936 之后（3,262 / 3,412 / 3,306），所以两臂 Tier B 都是 3/3。
  → 冻结卡必须补这一条限制（见 §7-1），下一版卡的判分口径要改用 Tier A 或"入口后存活帧数"。

---

## 5 B 批：加长训练的自然开局对照

| 策略 | 步数 / epoch | 三 seed 帧数 | 均值 | 对 2,000 步 |
| --- | --- | --- | ---: | --- |
| 2,000 步顺序（回执 07） | 2,000 / 9.7 | 1,504 / 1,808 / 2,224 | 1,845.3 | — |
| **8,000 步顺序（L1）** | 8,000 / 38.8 | 1,457 / 2,895 / 1,401 | **1,917.7** | +72.3（+3.9%） |
| 逐种子差 | | | | **−47 / +1,087 / −823** |

- **不能算效应**：n=3、无 CI，且 3 个种子里 2 个下降；按项目"两个 CI 不重叠才算差异"的口径，本轮只说"同量级"。
- Tier B 仍 0/3；最高 2,895（差 41 帧到 2,936）是本项目自然开局最接近首 boss 的一次，但仍未到。

---

## 6 加长训练：预算饱和、数据不足

| 配置 | 步数 | epoch | fit top-1 | donor top-1 | donor dominant |
| --- | ---: | ---: | ---: | ---: | ---: |
| E3 顺序（2k） | 2,000 | 9.7 | 0.9606 | 0.5933 | 0.3638 |
| **L1 顺序（8k）** | 8,000 | 38.8 | 0.9892 | **0.6211** | 0.3736 |
| L2 平衡（8k） | 8,000 | 38.8 | 0.9683 | **0.6340** | 0.3947 |
| L3 顺序（16k） | 16,000 | 77.7 | **1.0000** | **0.6359** | 0.3742 |

- donor 在 8k 时已到 0.62–0.63，16k 只再 +1.5 pp；fit 到 1.0 说明开始记忆。
- dominant 不再随预算下降（顺序持平、平衡回升），说明"决策多样性"不是靠步数换来的。
- → **下一个瓶颈是 episode 多样性**：现在只用 T04 的 `seed10` 一条 episode（6,592 行）；
  `seed11` / `seed12` 两片（合计 19,772 行）已经在盘上、还没进训练。

---

## 7 偏差与异常

1. **冻结卡的 Tier B 不具区分力（已在卡内写入"弱子目标"，但实测比预期更弱）**：恒动作臂也在 2,936 之后死亡，因此两臂 Tier B 都是 3/3；下一版必须改用 Tier A 或"入口后存活帧数"，并补逐决策 `enemies` 读数。
2. **`s2c2/ledger.jsonl` 是事后重建**：A 的 reserve 时刻取 `result.json.created_at`、B 取 `episodes.jsonl mtime − wall_s`；`wall_s` 字段因此为 0，B 批墙钟未单列。
3. **首次 A/seed10 预检被拒**：脚本把 `--engine` 指向 `/mnt/d/...` 副本（不存在），工具以 `stop=engine_missing` rc=4 拒跑，**未起引擎、0 帧**；留档 `A/seed10/preflight_refusal_engine_missing.result.json`，改用 `/root/Touhou_RL` 引擎重跑成功。
4. **静音读数 14 条而非 12 条**：多出的 2 条来自那次未起引擎的预检，已在 `audio.json` 分列 `extra_readings`。
5. **verify 首版误报**：曾把 Tier B 当通过条件、并按 1e-6 比较四舍五入均值；已改为"结果只记 INFO、只校验产物一致性"，复跑 PASS。
6. **决策 0..7 的动作历史来自 slot0 的 `history`**：磁带没有这 8 个决策的 `executed_cand`，这是 checkpoint 臂的已知播种口径。

---

## 8 分类声明

- **训练端本次实测（引擎）**：§2 A/B 两批 6 次启动、静音 14/14、账本。
- **训练端本次 GPU 实测**：§1 三个 checkpoint 与 §6 的离线指标。
- **训练端本次离线**：runner 48 项测试、`verify_s2d` 51 项、冻结卡生成器。
- **历史记录**：2,000 步的 1,845.3、`pilot_final` 731、恒动作 746、教师 Tier A 3/3（回执 04/07）。
- **待核验**：Tier A（缺 `enemies` 读数）；B 批 B 墙钟；三片语料合训的效果。

---

## 9 本回执**不**主张

1. 不主张 checkpoint 臂击破了首中 boss：**判不了**，只能说"3/3 活着进入 boss 窗口，2/3 死在窗口内"。
2. 不主张 8,000 步比 2,000 步更好：+3.9% 且两 seed 下降、无 CI。
3. 不主张 `S1N-MB2` 的 Tier B 有区分力：恒动作臂同样 3/3。
4. 不主张加长训练能继续提升：16k 的 fit=1.0 是记忆，donor 只 +1.5 pp。
5. 不主张 1,917.7 可外推到新语料/新关卡。

---

## 10 下一项能改变判断的实验（按代价排序）

| # | 实验 | 代价 | 会区分什么 |
| ---: | --- | --- | --- |
| 1 | 给 checkpoint 臂加**逐决策 `enemies` 采样**，重跑 A 批 3 根 | 代码 + 3 次启动 | Tier A 到底成不成（本卡唯一的强成功口径） |
| 2 | **三片合训**（T04 seed10+11+12 = 19,772 行 = 3 条 episode） | 0.5–2 min GPU + 6 次启动 | "数据不足"是不是真瓶颈（对照单片 8k 的 1,917.7） |
| 3 | 按新口径修订 `S1N-MB2`（Tier A + 入口后存活帧数），冻结 v2 | 0 次启动 | 让条件接管卡具备区分力 |
| 4 | 对 seed11/12 的死亡点做逐决策法证（已在盘） | 0 次启动 | 死亡是撞弹还是被逼位；下一处该修什么 |
| 5 | 用 seed11/12 作训练片、seed10 作 donor 的反向泛化 | 0 次引擎，GPU | 排除"只在 seed10 上有效" |

---

## 11 证据索引

| 结论 | 路径 | 身份 |
| --- | --- | --- |
| 冻结卡 | `.tmp_v832_takeover/task_card_mb2.json` / `.md` / `make_task_card_mb2.py` | content `2eee5046…`（首冻 `70032a67…`）/ `ab03aadc…`（重冻后另存）/ `47a0e265…` |
| takeover runner | `tools/rl/v8_32_takeover.py` / `tests/test_v8_32_takeover.py` | `0b3ab4eb…`（2,455 行）/ `6d9b5285…`（1,732 行，48 项） |
| 加长训练 | `.tmp_v832_s2d/ckpt_l{1,2,3}.pt` / `train_s2d.json` / `compat_s2d.json` / `s2d.md` / `verify_s2d.py` | `d8f89128…` / `bba0d343…` / `312410d0…` / `8576f5f7…` / `8dc7972f…` / `8873bf8f…` / `91f34ba1…`（51/51） |
| 双层评测 | `.tmp_v832_s2c2/results.json` / `s2c2.md` / `verify_s2c2.py` / `ledger.jsonl` | `c93e65a7…` / `a85155c3…` / `802081f0…` / `880df7a0…`（51/51） |
| A 批原始 | `.tmp_v832_s2c2/A/seed{10,11,12}/` | `result.json` + `decisions.jsonl` |
| B 批原始 | `.tmp_v832_s2c2/B/seed{1000,1001,1002}/` | `summary.json` + `decisions.jsonl` |
| 预检拒绝留档 | `.tmp_v832_s2c2/A/seed10/preflight_refusal_engine_missing.result.json` | rc=4、0 帧 |
