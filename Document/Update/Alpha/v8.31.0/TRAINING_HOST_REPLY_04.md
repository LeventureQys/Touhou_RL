# V831-ROADMAP-01 阶段回执 04：自然开局任务卡 S1N-MB1、三臂对照与数据腿

**工单**：[`LONG_TERM_DEVELOPMENT.md`](LONG_TERM_DEVELOPMENT.md)（V831-ROADMAP-01）· 本轮实验记录 [`EXP_NAT01_RECORD.md`](EXP_NAT01_RECORD.md)
**主机代号**：轩神开播了（Windows 11 家庭版 / i5-14600KF；WSL2 `ubuntu`，kernel `6.18.58.1-microsoft-standard-WSL2`，20 vCPU，23 GiB，RTX 4070 SUPER 12,282 MiB）
**时间及时区**：2026-11-27 00:45 – 01:40 +08:00（本回执全部时间为 +08:00）
**执行**：DSH 主代理（DeepSeek Flash）派发 4 个执行代理完成离线取证、代码扩展与真机运行；主代理独立复算关键数字、复核范围与命令
**检出**：`D:\WorkShop\Touhou_RL`（WSL `/mnt/d/WorkShop/Touhou_RL`），HEAD `1233598796b1b6f1d734950a4113af917da7b844`（分支 `v400`）
**脏差异**：接单时 `git status --porcelain` = **5,150 行**（3,493 删除 + 1,635 未跟踪 + 22 修改，均为既有漂移）；本轮新增 2 个代码/测试文件、4 个文档、6 个证据根目录
**本轮新增磁盘**：**125,436,868 B ≈ 0.12 GiB**

---

## 0 结论摘要（五条）

1. **自然开局任务成立，且三臂可区分（H2 成立）。** 同一引擎（`rl_v3` `691bfd77…`）、同一静音私有目录、同一批 9 次启动：
   教师 **3/3** 在 4,500 帧内 **0 死亡**并击破首中 boss（`boss_life` 6,000→0 的帧：**3,684 / 3,740 / 3,652**）；
   当前学生 checkpoint **3/3 在 731 帧死亡**；恒动作 `cand 12` **3/3 在 746 帧死亡**。
   Tier A（击破首中 boss）= 3/3 vs 0/3 vs 0/3；Tier B（帧 2,936 时仍存活）= 3/3 vs 0/3 vs 0/3。
2. **三个 dev 种子不是三个独立任务实例——这是本轮最重要的结构性发现。** 学生臂三 seed 逐位相同（731 帧 / 7,535 分），
   恒动作臂三 seed 逐位相同（746 帧 / 2,630 分）；离线教师 20 根在帧 400–2400 的 `(px,py)`、敌机计数、`life_sum` **完全相同**，
   首次分叉在 **2,580–2,584**。入口 `reset_hash` 随 seed 变化（`a23f5ede` / `c061727e` / `c16c0976`），但 0–2,580 的可玩几何不变。
   → 在这一窗口里"多独立根"不能来自种子。
3. **离线失败定位**：另一 checkpoint（`96cb219e…`）在 normal 下自然开局首死 **1,397 / 1,562 / 2,185**（min/median/max，20 根），
   全部落在普通敌机段、比首 boss（帧 2,936）至少早 751 帧；**"位置冻结"先兆在自然开局的 0/20 命中**
   （末段恒定段 1–28 帧、死前 0–27 帧仍在换输入），失败形态是"13 字小动作词汇、约 30% 帧只按 bit0"的低多样性持续输出，不是卡输入。
4. **数据腿可以结**：T04 两帧语料**可作 BC 标签，口径必须是 `executed`**——`history` 与 `act_hist` 100% 等于已执行序列
   （19,748/19,748、19,772/19,772），且没有"看到未来帧"（像素末帧 `d-1`、动作生效帧 `d`）；`chosen` 与 `executed` 差 6,288/19,772 行，
   只可作验证目标，不能与 `executed` 混为同一训练目标。
5. **学习器的近恒定不是标签错位造成的，是采样/读出造成的。** 同 seed、同 200 步、同初始化下把标签从 `chosen` 换成 `executed`：
   dominant share 0.9087→0.8864（−2.23 pp）、distinct 4→6，**没有修好**；只改采样（按 `executed` 类别平衡）则 0.2734 / 17 类，
   但 top-1 掉到多数类基线以下。归因：采样项 dominant −0.613、distinct +11；标签项 dominant −0.022、distinct +2。

---

## 1 身份（本机实测）

| 项 | 值 |
| --- | --- |
| 引擎（本轮全部 9 次启动） | `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，4,836,008 B，sha256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` |
| 源数据 | `/root/v400-data/game`，`du -sb` = 496,968,814 B；树指纹 `483085552fd3d044ecfa6cb3e14652e0261658df8a8e0840d6e5054d57e67386`（运行前后相同） |
| 私有静音目录 | `.tmp_v832_nat01_runs/game-muted/`，`th08.cfg` sha256 `cc062cfbf9fd2c14a1225376e182beffc52f964c342c187b409045f495d5f0fa`（四点全 0） |
| 学生 checkpoint | `.tmp_v831_followup_02a_gwZ4fA8u/03/train/weights_final.pt`，10,467,859 B，sha256 `336397ed6f3adc738264d9dd1cc9299b6345348306c77aa33450f867286c4084` |
| 离线学生 checkpoint（nat01） | `a0_seed0_step5000.pt`，sha256 `96cb219eac83391f42b18ea656b5a6c5ac81e97cd1fd8c1e2c0d404aee6b89ad`（**与本轮三臂学生不同，不混池**） |
| 工具（本轮扩展） | `tools/rl/v400_student_eval.py` 1,503 行，sha256 `1ae7ccde719a8ea1e26444892481662ee4847f34d698d802ade2448ed367f985`（改前 `7bccf240…e843c1a2`、1,038 行） |
| 测试（本轮新增） | `tests/test_v8_32_natural.py` 777 行，sha256 `12d4f597b09b4421212a22f2dfb4d3b454e7c95afa10d3a539b3a111412e9e4a`（29 项） |
| Python | WSL `/root/v400-venv/bin/python` 3.10.12 / numpy 2.2.6 / torch 2.14.1+cu130；Windows `python` 3.12.10 / numpy 2.5.4 / torch 2.14.1+cu132 |
| GPU | RTX 4070 SUPER（Windows 侧 CUDA 可用） |
| 历史教师引擎（**不是**本轮引擎） | `/root/Touhou_RL/third_party/th08-rl/build/rl_hazard2/th08-modern`，sha256 `f172399f61b75baa8e62be46041ef756a710759f439e7fb07111f4747c78264e` |

**引擎身份纠正（重要）**：`.tmp_v400_g1/s1n/` 那 20 根教师轨迹的报告里 `engine.sha256 = f172399f…`（`rl_hazard2`），
而学生轨迹与 v8.x 各轮用的是 `rl_v3` `691bfd77…`。**历史上"教师 20/20 vs 学生"是跨引擎比较**。
本轮为此把三臂统一到 `rl_v3`（教师工具的 child 记录在 `rl_v3` 上可用：`probe.child_hazards.available=true`、`record_source=child`），
并在 `results.json.engine_batch` 里登记"一批一引擎"。

---

## 2 命令与退出码

| # | 命令 | rc | 结果 |
| ---: | --- | ---: | --- |
| 1 | `python -B .tmp_v832_nat01\verify_nat01.py`（主代理复跑） | **0** | 468 项复算全等 |
| 2 | `python -B .tmp_v832_data01\verify_data01.py`（主代理复跑） | **0** | 36 PASS / 0 FAIL |
| 3 | `python -B .tmp_v832_s20_refit\verify_refit.py` | **0** | 37 checks / 0 failures |
| 4 | `python -m pytest … tests\test_v8_32_natural.py tests\test_v400_student.py tests\test_v400_obs_v2.py tests\test_v400_eval_power.py tests\test_v400_model_v3.py tests\test_v400_labels_signal.py`（主代理复跑） | **0** | **163 passed**（改前同 5 个既有文件 134 passed，+29 新增） |
| 5 | `wsl … /root/v400-venv/bin/python .tmp_v832_nat01_runs/verify_runs.py` | **0** | **76 passed / 0 failed** |
| 6 | `wsl … /root/v400-venv/bin/python .tmp_v832_stage0/seed_pixel_probe.py` | 0 | 跨 seed 像素对照（见 §8-6） |
| 7 | `python -B .tmp_v832_nat01\make_card.py` | **0** | 冻结任务卡 `content_sha256 7f1e91c5…b420` |

九次引擎启动（全部 `xvfb-run`，`SDL_AUDIODRIVER=dummy`）：

| arm | seed | 命令要点 | rc | 帧 | 墙钟 s | 结局 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| T | 1000 | `v400_teacher.py --mode full --decisions 1200 --max-frames 4500` | 0 | 4,500 | 111.951 | max_frames，0 死亡 |
| T | 1001 | 同上 | 0 | 4,500 | 109.162 | max_frames，0 死亡 |
| T | 1002 | 同上 | 0 | 4,500 | 104.297 | max_frames，0 死亡 |
| S | 1000 | `v400_student_eval.py --obs v3 --arch v3 --max-frames 4500 --stop-on-death --decision-log` | 0 | **731** | 11.979 | death_stop，score 7,535 |
| S | 1001 | 同上 | 0 | 731 | 11.041 | death_stop，7,535 |
| S | 1002 | 同上 | 0 | 731 | 10.379 | death_stop，7,535 |
| C | 1000 | 同上 + `--arm constant --constant-cand 12` | 0 | **746** | 8.635 | death_stop，2,630 |
| C | 1001 | 同上 | 0 | 746 | 8.231 | death_stop，2,630 |
| C | 1002 | 同上 | 0 | 746 | 8.133 | death_stop，2,630 |

---

## 3 额度实耗

| 额度 | 本轮上限 | 实际 | 判定 |
| --- | ---: | ---: | --- |
| 引擎启动 | 12 | **9** | 未超 |
| 引擎帧 | 40,000 | **17,931** | 未超 |
| 启动墙钟合计 | — | **383.808 s** | — |
| 新增磁盘 | 8 GiB | **0.117 GiB**（125,436,868 B，六个新目录） | 未超 |
| 静音 | 每次前后两点 | **18/18 次读数全 `0 0 0 0`**，`audio_refused=0` | 通过 |
| 账本 | 无未结预留 | `ledger.jsonl` 18 行 = 9 reserve + 9 settle | 通过 |

---

## 4 R1-A 自然开局最早可控失败（离线，0 次启动）

数据源：教师 20 根 `.tmp_v400_g1/s1n/s1n_seed10{00..19}.jsonl`（`rl_hazard2`）+ 学生 20 根 `.tmp_v420_data/T08/student_traces/seed10{00..19}.json`
（`rl_v3`，`96cb219e…`，normal）。产物 [`.tmp_v832_nat01/nat01.json`](../../../../.tmp_v832_nat01/nat01.json)（355,908 B，468 项复算 PASS）。

| 指标 | 教师（20 根） | 学生（20 根） |
| --- | --- | --- |
| 结局 | `stage_clear` 20/20，`deaths_final=0` | `first_death` 20/20 |
| 生存 / 首死帧 | 12,812 / 13,532 / 15,204 | **1,397 / 1,562 / 2,185** |
| 帧 1,200 | 20/20 存活 | 20/20 存活 |
| 帧 1,400 | 20/20 存活 | 19/20 有轨迹（1006 死于 1,397） |
| 帧 2,000 | 20/20 存活 | 3/20 有轨迹 |
| 帧 2,400 / 2,936 | 20/20 存活 / boss 出现 | 0/20 |

- **首个 boss 代**：帧 **2,936**（决策 733，`boss_life=6000`），20 根帧号完全相同；首个 `enemies.count>=8` 在帧 860，`>=12` 在 2,632。
- **中 boss 击破**：`boss_life` 6,000→0 帧 **3,652–3,900**（中位 3,740），20/20 可给；**终 boss 不可用作事件**（第二池末 `life=1`，从不归零）。
- **失败形态（否证"冻结"）**：按"≥30 帧 mask 恒定且延续到死亡"的定义 **0/20 命中**；末段恒定段 1–28 帧（中位 6），
  死前最后一次换输入 0–27 帧（中位 5）。学生动作词汇只有 **13 个字**（全为奇数、bit0 恒置位），`mask==1` 占比中位 0.298，`distinct_masks` 9–12。
- **最早可控失败点估计**：帧 **1,397**（seed 1006）；起点区间 **[1,200, 1,397]**（1200 是全存活、1397 是首个死）；
  群体区间 1,397–2,185。判据：教师在同窗口 0 死亡通关 → 存在可达策略；学生最早死亡且无冻结 → 能力型失败，不是环境/输入故障。
- **样本独立性（关键负结果）**：教师 20 根在 400/800/1200/1400/2000/2400 的 `(px,py)` **唯一值 = 1**，敌机计数与 `life_sum` 全一致，
  直到帧 2,936 才 20 个唯一值。**"教师 20/20"在失败窗口内实质是 1 条轨迹 × 20 次复放**。

---

## 5 R1-B 数据与标签审计（离线，0 次启动）

产物 [`.tmp_v832_data01/data01.json`](../../../../.tmp_v832_data01/data01.json)（36 项复算 PASS）。T04 三片 19,772 行，`label_source` 全 `teacher_2f`。

| # | 问题 | 结论 | 关键数字 |
| ---: | --- | --- | --- |
| 1 | `executed≠chosen` 规模与结构 | 31.8%，几乎全在 legacy 块内第二槽 | 6,288/19,772；偏移 2 占 6,227（62.99%），偏移 0 仅 61（0.62%） |
| 2 | `chosen` 是否同一状态上的教师重查询 | 是（逐槽现查），但 `chosen_score ≥ scores[executed]` 是 argmax 的同义反复 | 19,772/19,772 为全局 argmax |
| 3 | **`history` 用哪条序列** | **已执行（executed）** | 窗口全等 executed 19,748/19,748 = 100%；对 chosen 仅 4,510 = 22.84% |
| 4 | `history` vs `act_hist.npy` | 完全一致 | 19,772/19,772 = 100%，坏 one-hot 0，WSL 复核同值 |
| 5 | 时序对齐 | 无"看到未来帧"；偏移 2 槽有"标签滞后一决策" | 像素末帧 `d-1`、动作生效帧 `d`；偏移 2 槽教师翻转率 62.99% |
| 6 | **可用性** | **`executed` 口径可作 BC 标签；`chosen` 只可作验证目标** | executed 无需修正；chosen 离线不可修 |

---

## 6 R1-C/D 任务卡 S1N-MB1 与三臂对照（真机实测）

冻结卡：[`.tmp_v832_nat01/task_card.json`](../../../../.tmp_v832_nat01/task_card.json)（`content_sha256 7f1e91c5e552be4de64afa02b8d8e3f6d0754ad9641b4c2a4210bcd17453b420`）
人读版 [`task_card.md`](../../../../.tmp_v832_nat01/task_card.md)。

| 项 | 定义 |
| --- | --- |
| 入口 | 自然 `RESET`：stage 0 / normal（参数 `1`）/ team 0 / seed ∈ dev；入口决策 0、帧 0 |
| 成功 A | 首个 boss 代 `boss_life` 观测到 6000 → 0（帧上限 4,500） |
| 成功 B | 帧 2,936（首 boss 出现）时仍存活（帧上限 3,200） |
| 合法失败 | `deaths` +1（首死即终止） |
| 截断 / 优先级 | 4,500 / 3,200 帧；`death > success > truncated > error` |
| 时钟 | 教师 4 帧/决策、学生 2 帧/决策；**比较只用游戏帧** |
| 射击 | 外生脉冲（教师 24、学生 48）；动作契约表达不了"不射击" |

**三臂结果**（`results.json`，9 次启动、17,931 帧、383.808 s，`verify_runs.py` 76/76 PASS）：

| 臂 | 根 | 帧 | 死亡 | Tier A | Tier B |
| --- | ---: | ---: | ---: | --- | --- |
| 教师（`rl_v3`，child 记录） | 1000/1001/1002 | 4,500 ×3 | 0 ×3 | **3/3**（3,684 / 3,740 / 3,652） | **3/3** |
| 学生（`336397ed…`，obs-v3） | 1000/1001/1002 | 731 ×3 | 1 ×3 | 0/3 | 0/3 |
| 恒动作 `cand 12` | 1000/1001/1002 | 746 ×3 | 1 ×3 | 0/3 | 0/3 |

**入口身份**：seed 1000 的 `reset_hash = a23f5edebd0e7454`，与独立 T08 学生轨迹记录的同一 seed 哈希**逐位相同**；
同 seed 的学生臂与恒动作臂 `reset` 字典只差 `setup_frames`（建局走帧，不进 HASH）。
**同 seed 内 S 与 C 的 `reset` 相同而动作不同** → 入口可复现、分歧来自策略。

**结构性限制（必须与结论一起读）**：三个 dev 种子在 0–2,580 帧的可玩几何相同，
学生三 seed 逐位同结果（731/7,535）、恒动作三 seed 逐位同结果（746/2,630）。
所以本轮的"3/3 vs 0/3 vs 0/3"在**根的意义上**是 **1 个任务实例 × 3 次复放**；
要拿到路线要求的独立根，必须换更晚入口、换难度，或注入起始状态。

---

## 7 S2-0 `executed` vs `chosen` 复拟合（离线，GPU 实测）

产物 [`.tmp_v832_s20_refit/refit_executed.json`](../../../../.tmp_v832_s20_refit/refit_executed.json)（37 checks / 0 failures）。

| 配置 | 标签 / 采样 | top-1（拟合根 / 供体） | dominant share | distinct |
| --- | --- | --- | ---: | ---: |
| C1 | `chosen` / 顺序（复现 §13.2 A） | 0.4897 / 0.4961 | 0.9087 / 0.9099 | 4 / 4 |
| C2 | `executed` / 顺序（同一索引序列） | 0.4251 / 0.4349 | **0.8864 / 0.8888** | 6 / 5 |
| C3 | `executed` / 类别平衡 | 0.3755 / 0.3051 | **0.2734 / 0.3117** | 17 / 15 |

- **判定：标签不一致不是近恒定主因。** 换 `executed` 只把 dominant 降 2.23 pp；distinct 4→6，仍 ≫ 多数类先验的 2.4 倍。
- **采样才是主因**：类别平衡把 dominant 降到 0.27、17 类，但 top-1 掉到多数类基线（0.3736）以下——不坍缩 ≠ 更好策略。
- **控制组 D**：直接加载上一轮的 `weights_initial.pt` 跑同样 200 步，**逐位复现 §13.2 A**（final 156/156 张量 max|Δ|=0）。
  C1 与 §13.2 A 的 ±1.7 pp 差来自两侧框架构建（cu130 vs cu132）在 37/156 个初始张量上的 1 ULP（max 1.49e-08），200 步 Adam 放大——
  不是配方/数据差异。

---

## 8 偏差、缺陷与首个失败记录

1. **第一个失败（R1-D 命令语义）**：`v400_teacher.py --engine` 的真实语义是"含 `th08-modern` 的**构建目录**"，传二进制会被拼成
   `th08-modern/th08-modern`。已改传 `.../build/rl_v3` + `--expected-sha`，同一文件同一 sha 门。
2. **教师必须显式 `--decisions 1200`**：默认 `--decisions 10` 只跑 40 帧，到不了 4,500 帧上限；`--max-frames` 未被改动。
3. **学生首死 731 与本轮实验记录里的 1,397–2,185 不符——这是口径混用，已更正**：
   1,397–2,185 是**另一个 checkpoint（`96cb219e…`）在 normal** 的首死（`nat01`）；
   `pilot_final`（`336397ed…`）在 **easy** 下是 1,458–1,570（`ASSET_SYNC` §41 的 ep01–ep07），在 **normal** 下是 **731**（§41 的 ep08，seed 12）。
   本轮三臂学生用 `pilot_final` + normal，实测 731 / 7,535，与 §41 ep08 逐位一致。**难度与 checkpoint 两个口径在起草时被混用，此处更正，不回改已验证数据。**
4. **历史教师引擎不是本轮引擎**：`s1n` 20 根跑在 `rl_hazard2` `f172399f…`；本轮三臂全在 `rl_v3` `691bfd77…`。
   因此 §4 的"教师 20/20"**不能**与本轮 §6 的教师臂当作同一份证据合并计数。
5. **`game-muted` 私有目录树 sha 变化**（`dcc5c7c8…`→`880d214b…`）：引擎在该目录重写 `score.dat` 并新增 5 个 `backup/score_*.261127.bak`；
   源目录 `/root/v400-data/game` 指纹前后不变，静音四点前后不变。
6. **跨 seed 像素对照（附带实测）**：T04 三片在**同一 slot/decision_frame** 上，`hud` 与 `act_hist` 前 64 行**完全相同**，
   `play96x112` 从第 0 行起就不同，但只差 **57/43,008 像素（0.13%）**，集中在顶部 0–3 行、0–66 列的一条窄带。
   这解释了"几何相同、像素略异"：像素策略在这种差异下就可能分叉，而几何不随 seed 变。
7. **`driver/results.json` 是首版构建器路径 bug 的误放副本**（内容为 null），因"不删除"保留并改写成指向 `../results.json` 的说明；
   唯一权威是 `.tmp_v832_nat01_runs/results.json`（sha256 `e4787d00…2f7a`）。
8. **教师臂 `score=null`**：教师工具不读 `score` 字段（其 STATE 无此项），未伪造。

---

## 9 分类声明

- **训练端本次实测（引擎）**：§2 九次启动的帧数/墙钟/结局/死亡/分数、§6 三臂 Tier A/B、入口 `reset_hash`、静音 18 次读数、源树指纹。
- **训练端本次离线**：§4 全部（`nat01` 468 项复算）、§5 全部（`data01` 36 项）、§7 全部（`refit` 37 项）、§8-6 跨 seed 像素对照。
- **历史记录**：`.tmp_v400_g1/s1n/` 20 根教师轨迹（本轮只读重算，不重跑）、EXP-P1-012 的 S1N 20/20、`ASSET_SYNC` §41 的 Easy/Normal 行。
- **待核验**：①学生侧 px/py（T08 轨迹不记录位置，所有几何结论只来自教师侧）；②`mask` 位语义（本轮未重新解码，`mask==1` 只表示"仅 bit0"）；
  ③T04 像素平面第 r 行与绝对帧的**逐帧哈希锚定**（只有网格恒等 + 尾部对齐 + 声明字段三重一致）。

---

## 10 本回执**不**主张

1. 不主张 S1N-MB1 可以直接进正式门：Tier A 的判分读数是逐决策 `enemies.boss_count/boss_life`，而项目口径要求 `HAZARDS` 行身份（slot+flags1、有符号 life）。
2. 不主张"3 根"是 3 个独立任务实例：见 §6 结构性限制，它是 1 个实例 × 3 次复放。
3. 不主张教师臂在 4,500 帧是"过关"：它是 `max_frames` 截断，只证明该窗口内 0 死亡且完成 Tier A。
4. 不主张 `pilot_final`（731）与 `96cb219e`（1,397–2,185）可比：不同 checkpoint，且 §8-3 的难度口径混用已更正。
5. 不主张 `chosen` 标签"不能用"：它可作验证/审计目标，只是不能与 `executed` 混为同一训练目标。
6. 不主张类别平衡修好了学习器：它只把动作分布摊平，top-1 反而低于多数类基线。
7. 不主张自然开局的失败机制是"位置冻结"：该先兆在自然开局 0/20 命中（与中局窗口 §13.3 的结论不同，可能是窗口差异，未归因）。

---

## 11 下一项能改变判断的实验（按代价排序）

| # | 实验 | 代价 | 会区分什么 |
| ---: | --- | --- | --- |
| 1 | 写 `HAZARDS` 口径的 Tier A 判分器，并用它重算本轮 3 根教师原始包 | 0 次启动 | 让 S1N-MB1 的 Tier A 具备正式口径；若重算不一致，说明 `ENEMIES` 分页漏过 boss |
| 2 | **构造独立根**：在帧 >2,580 的入口、或注入起始状态、或换难度，先离线验证入口可复现 | 0 次启动（先做接口/可行性） | 任务卡能否满足"多独立根"这一硬要求 |
| 3 | 学习器读出诊断：冻结编码器只训 readout + 温度/标签平滑 + 更长预算（200→2,000 步），先看离线 dominant/top-1 | 0 次启动，GPU ~10 min | 近恒定是"读出可达类只有 3 个"还是"训练不足/容量/正则" |
| 4 | 在 Tier B 窗口（≤3,200 帧）跑学习前后对照，用修好的学习器 | ~6–12 次启动 | 像素学习是否带来真实任务增益（阶段 2 的第一条证据） |
| 5 | 固定难度口径重跑 easy/normal 对照（同一 checkpoint、同一 cap） | 2–4 次启动 | 难度对首死的影响是否与既有 Easy 结果一致 |

---

## 12 证据索引（训练端精确路径）

| 结论 | 路径 | 身份 |
| --- | --- | --- |
| 离线失败定位 | `.tmp_v832_nat01/nat01.json` / `nat01.md` / `verify_nat01.py` | `2222d7c2…5be` / `3d612350…aa1` / `c5d4813c…5cd`（468 项 PASS） |
| 任务卡 | `.tmp_v832_nat01/task_card.json` / `task_card.md` / `make_card.py` | `7eba7c5e…8dcc`（content `7f1e91c5…b420`）/ `23ff4bee…ab8` / `546b32b3…007` |
| 数据审计 | `.tmp_v832_data01/data01.json` / `data01.md` / `verify_data01.py` | `06fcdc93…fe6a` / `c08620c7…167` / `95f9cad5…b6b`（36 项 PASS） |
| 复拟合 | `.tmp_v832_s20_refit/refit_executed.json` / `.md` / `verify_refit.py` | `9ed49124…ea9` / `59a8092c…27` / `bfc442c3…31`（37 项 PASS） |
| 三臂对照 | `.tmp_v832_nat01_runs/results.json` / `ledger.jsonl` / `verify_runs.py` / `RUN_NOTES.md` | `e4787d00…f7a` / `47a3f223…b35` / `a9d0b7bd…1a4` / `94dc3d7f…a67`（76 项 PASS） |
| 三臂原始 | `.tmp_v832_nat01_runs/{T,S,C}/seed{1000,1001,1002}/` | 见 `results.json.arms` |
| 静音证据 | `.tmp_v832_nat01_runs/preflight.json`、`post.json` | 18 次四点读数 |
| 工具 | `tools/rl/v400_student_eval.py` | `1ae7ccde…985`（1,503 行） |
| 测试 | `tests/test_v8_32_natural.py` | `12d4f597…4e9`（777 行，29 项） |
