# V831-ROADMAP-01 阶段回执 06：2,000 步 BC 的闭环负结果、条件接管打通与 S1N-MB2 草案

**工单**：[`LONG_TERM_DEVELOPMENT.md`](LONG_TERM_DEVELOPMENT.md)（V831-ROADMAP-01）· 承接 [`TRAINING_HOST_REPLY_05.md`](TRAINING_HOST_REPLY_05.md)
**主机代号**：轩神开播了（Windows 11 家庭版 / i5-14600KF；WSL2 `ubuntu`，20 vCPU，23 GiB，RTX 4070 SUPER）
**时间及时区**：2026-11-27 01:36 – 02:02 +08:00（本回执全部时间为 +08:00）
**执行**：DSH 主代理（DeepSeek Flash）派发 3 个执行代理（训练+闭环评测、接管 runner、真机验证）；主代理独立复跑三份验收（24 / 30 / 18 项）与账本汇总
**检出**：`D:\WorkShop\Touhou_RL`，HEAD `1233598796b1b6f1d734950a4113af917da7b844`（分支 `v400`）
**引擎启动**：**10 次**（6 次闭环评测 + 4 次接管验证）

---

## 0 结论摘要（四条）

1. **2,000 步 BC 在 D1 规模上没有改善自然开局闭环表现——负结果，且机制清楚。**
   `before`（初始化 + 1 步）与 `after`（2,000 步，D1，batch 32）在 S1N-MB1 上**都是 705 帧首死、score 1,450、
   353/353 决策恒为 `cand 0`**，逐位同一，Δ = **0 帧**。
   离线读数确实变了：test top-1 0.5132 → **0.6069**（多数类 0.5132）、非 idle 预测 0% → **18.1%**、预测类 1 → 7；
   **但闭环一次都没触发非 idle**。→ "离线 argmax 改善"不能推成"闭环行为改善"；机制是 idle 自强化（离线 idle 0.795 → 闭环 1.0）。
2. **预算口径必须按语料规模换算，"2,000 步可修"只在 T04 规模成立。**
   D1 训练集 **332,377 条决策**，64,000 样本 = **0.19 epoch**；T04 的 6,592 行 = **9.7 epoch**（回执 05 的 E3/E4）。
   而且训练器保存的是 **best-by-val 快照（step 900）**，val top-1 在 0.60 附近饱和 → **单纯加步数未必能解决闭环坍缩**，
   下一步要动的是采样/目标，而不是再加步数。
3. **条件接管打通：第一次拿到 >1 个真实独立入口状态。** 新 runner `tools/rl/v8_32_takeover.py`（1,565 行、30 项离线单测）
   在真机上对 3 条磁带各核验 **1,308 个 prefix 哈希**后交还控制：入口哈希 `e2a84b01…` / `9edbdd05…` / `b8fdb779…`
   与各自磁带 slot 1307 的 `hash_post` **逐个相等**、且**三 hash 互异**；tape 臂 3/3 跑满 4,500 帧、0 死亡；
   constant 臂从入口后 **630 帧**死亡（远小于 cap 1,868）。→ `S1N-MB2` 冻结前三条验证里的第 1、2 条已通过，第 3 条有 1 根证据。
4. **顺手查出一个既有工具缺陷**：`v8_31_followup.py intervene` 在入口哈希不符时只把该臂记成 `technical_invalid`，
   **另一臂照样启动**（`L8496-8502`、`L8626-8629`），且没有 `entry_verified`/`mismatch_slot` 字段——不满足 fail-closed。
   本轮新 runner 已按 fail-closed 实现，旧工具未改（不在授权写入范围），仅登记。

---

## 1 身份

| 项 | 值 |
| --- | --- |
| 引擎 | `rl_v3/th08-modern`，sha256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`（10 次启动全部一致） |
| 训练数据 | `.tmp_v400_d1/d1`，137 完整 obs-v3 shard / 469,216 决策；split train 96 / val 21 / test 20 |
| after checkpoint | `.tmp_v832_s2a/ckpt_2000.pt`，10,482,699 B，sha256 `19808e88cfe359028147bd954df1c6b2d36e767420cc092a74b7122be22cbb90`（**best-by-val step 900 快照**） |
| before checkpoint | `.tmp_v832_s2a/ckpt_init.pt`，10,480,331 B，sha256 `5cae894001e736a51aba0803e0e756bb56897ab5871c086f5d674db774977888` |
| 参数 | 2,602,490（canonical=True） |
| 训练器 | `touhou_rl/v400/student/train_v3.py`，batch 32，cuda，AdamW lr 3e-4；D1 门禁 `gate_waived=false` |
| 接管 runner | `tools/rl/v8_32_takeover.py` 1,565 行，sha256 `4894363e30e47d8e…` |
| 接管测试 | `tests/test_v8_32_takeover.py` 849 行，sha256 `93cc1b07b1be20b0…`（30 项） |
| 私有静音目录 | `.tmp_v832_s2a/game-muted/`、`.tmp_v832_takeover/game-muted/`，cfg sha256 `cc062cfbf9fd2c14…` |

---

## 2 命令与退出码

| # | 命令 | rc | 结果 |
| ---: | --- | ---: | --- |
| 1 | `train_v3.py … --max-steps 2000 --batch-size 32` | 0 | steps 2,000 / 64,000 样本 / wall 306.4 s / loss 1.32736→0.36900 / val top-1 0.5964 / test 0.6069 |
| 2 | `train_v3.py … --max-steps 1` | 0 | steps 1 / wall 191.7 s / loss 1.32740→1.32740 / test 0.5132 |
| 3–8 | `v400_student_eval.py --obs v3 --arch v3 --max-frames 4500 --stop-on-death --decision-log`（after/before × seed 1000/1001/1002） | **0/0/0/0/0/0** | 均 705 帧、`death_stop`、deaths 1、score 1,450 |
| 9 | `pytest tests\test_v8_32_takeover.py`（主代理复跑） | **0** | **30 passed in 1.34 s** |
| 10 | `python -B .tmp_v832_s2a/verify_s2a.py`（主代理复跑，WSL） | **0** | **24/24 PASS** |
| 11 | `python .tmp_v832_takeover/verify/verify_takeover.py`（主代理复跑，WSL） | **0** | **18/18 PASS** |

接管验证 4 次启动（`xvfb-run` + `SDL_AUDIODRIVER=dummy`）：

| arm | seed | rc | frames（prefix+交还） | stop | deaths | entry_verified | entry_hash |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| tape | 10 | 0 | 4,500 = 2,632 + 1,868 | `frame_cap` | 0 | **true** | `e2a84b01b1aadef4` |
| tape | 11 | 0 | 4,500 = 2,632 + 1,868 | `frame_cap` | 0 | **true** | `9edbdd05aa4837a9` |
| tape | 12 | 0 | 4,500 = 2,632 + 1,868 | `frame_cap` | 0 | **true** | `b8fdb7795775901f` |
| constant `cand 12` | 10 | 0 | 3,262 = 2,632 + 630 | `legal_death` | 1 | **true** | `e2a84b01b1aadef4` |

---

## 3 额度

| 额度 | 实际 |
| --- | ---: |
| 引擎启动 | **10**（s2a 6 + takeover 4） |
| 引擎帧 | **20,992**（s2a 4,230 + takeover 16,762，含 prefix） |
| 引擎墙钟 | s2a 60.944 s + takeover 119.225 s = **180.169 s** |
| GPU 训练墙钟 | 306.4 s（2,000 步）+ 191.7 s（1 步门禁开销） |
| 新增磁盘 | 29,166,496 B ≈ **0.027 GiB** |
| 静音 | s2a 12/12 读数 + takeover 8/8 读数 = **20/20 全 `0 0 0 0`**，`audio_refused=0` |
| 账本 | s2a 12 行（6+6）、takeover 8 行（4+4），**无未结 reserve** |

---

## 4 R3-A：2,000 步 BC 的闭环负结果

| 臂 | seed 1000 | 1001 | 1002 | 均值 | 动作分布 | 离线 test top-1 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| before（初始化+1 步） | 705 | 705 | 705 | **705.0** | 353/353 `cand 0` | 0.5132（=多数类） |
| after（2,000 步 D1） | 705 | 705 | 705 | **705.0** | 353/353 `cand 0` | 0.6069 |

- 差值 **0 帧（逐 seed 0/0/0）**；Tier B 0/3 vs 0/3；两臂死亡帧、分数、轨迹逐位相同。
- **不是复用错误**：两个 checkpoint sha 不同、路径各自对应、reset hash 与 seed 一一对应；两臂相同的原因是**都恒定 idle**。
- 对照（引用回执 04 的 `.tmp_v832_nat01_runs/results.json`，未重跑）：`pilot_final` 731 帧（94.3% idle + 5.7% cand 7）、
  恒 `cand 12` 746 帧、教师 Tier A 3/3。本批 705 帧**比旧学生低 26 帧、比恒动作低 41 帧**——因为旧学生还有 5.7% 非 idle。
- **归因**：D1 的教师动作分布本身 idle 占 46.9%（val 多数类）；BC 在 0.19 epoch 下把 idle 学得更"自信"（离线 idle 0.795），
  闭环自强化到 1.0。离线非 idle 18.1% 的预测**从未在自己的闭环状态里出现**——典型的协变量偏移/自强化。

---

## 5 R3-B/C：条件接管 runner 与真机验证

### 5.1 交付

| 项 | 结果 |
| --- | --- |
| 新 runner | `tools/rl/v8_32_takeover.py`（1,565 行）：`run --engine --expected-sha --data-dir --tape-slots --seed --entry-frame --arm {tape,constant} [--constant-cand] --cap-frames --out [--decision-log]` |
| 离线单测 | `tests/test_v8_32_takeover.py`（849 行，**30 项**）：入口映射、逐槽哈希、fail-closed、入口帧校验、cap 语义、两臂动作序列、日志 header、result 字段、sha 门、可出声 cfg 拒绝等 |
| fail-closed | 入口哈希不符 → `entry_verified=false` + `mismatch_slot` + 非零退出，**不进入交还控制** |
| 掩码规则交叉核对 | 磁带 6,588 槽的 `mask` 推导 vs 记录 **0 不符**；vs D1 逐帧 `step_frames` 13,176 帧 **0 不符**；链断 0 |
| checkpoint 臂 | **未实现**，只留接口与 `CheckpointArmNotImplemented`；缺 5 件（obs-v3 actor 加载、SHM/READBACK 像素源、动作历史播种、臂身份写入、逐决策读 HUD 的决策） |

### 5.2 真机验证（4 次启动）

- 3 条磁带各核验 **1,308 个 prefix 槽**的哈希（`verified_slots=1308`、`lead_in_verified=true`、`STATE.frame=2632`），
  入口哈希与磁带 slot 1307（frame 2,630 的 `hash_post`）**逐个相等且三 hash 互异** → **3 个真实独立入口状态**。
- tape 臂 3/3 跑满 4,500 帧、0 死亡（它本来就是录像的续放，属 `not_a_policy_result`）。
- constant 臂从入口后 **630 帧**死亡（deaths=1）→ 退化臂在入口窗口内不完成，判分有区分力。

### 5.3 对 `S1N-MB2` 草案的意义

[`TASK_CARD_S1N-MB2_DRAFT.md`](TASK_CARD_S1N-MB2_DRAFT.md) 的冻结前三验：

| # | 验证 | 状态 |
| ---: | --- | --- |
| 1 | 前缀回放 3 根，逐槽 `hash_post` 一致 | **通过**（3/3，1,308 槽/根） |
| 2 | 入口状态登记，三根互不相同 | **通过**（三 hash 互异；字段级登记已落 `result.json`） |
| 3 | 退化臂从入口起不完成 | **部分**（1/3 根：seed10 630 帧死亡；seed11/12 未跑） |

---

## 6 偏差与异常

1. **训练器保存的是 best-by-val 快照（step 900）**，不是第 2,000 步的权重；即 `ckpt_2000.pt` 的实际权重来自第 900 步。
   这解释了为什么"跑满 2,000 步"的 loss 与权重不对应；口径已在 `results.json` 里写明。
2. **before 臂是"初始化 + 1 步"**（`--max-steps 1`），不是纯初始化；1 步的权重变化可忽略，但必须如实标注。
3. **两臂逐位相同**不是复用 bug（见 §4），但这也说明**3 个 seed 在这条臂上无法提供方差**（都在共享前缀内）。
4. **工单前提有两处错误（已由执行代理纠正）**：①`b8fdb7795775901f` 是 seed **12** 的入口哈希，不是 seed 10；
  ②入口帧 2,632 是 slot **1,308** 的 `decision_frame`，而入口哈希取自最后一个 prefix 槽 slot 1,307（frame 2,630）的 `hash_post`
   （两者相等，且已验证 `slot1307.hash_post == slot1308.hash_pre`）。
5. **常数臂死亡帧 3,262 与参考的 705–746 不可比**：参考从帧 0 起算，本次从 2,632 起算（入口后 630 帧）。
6. **`v8_31_followup.py intervene` 的 fail-closed 缺陷**（§0-4）：登记，未修改。

---

## 7 分类声明

- **训练端本次实测（引擎）**：§2 的 10 次启动、§4 的六次闭环、§5.2 的四次接管、静音 20/20 读数、账本。
- **训练端本次 GPU 实测**：§1 的两个 checkpoint 与 §2 的训练指标。
- **训练端本次离线**：接管 runner 的 30 项单测、掩码规则交叉核对（0 不符）。
- **历史记录**：回执 04 的三臂结果（`pilot_final` 731 / 恒 746 / 教师 Tier A 3/3），本轮只引用未重跑。
- **待核验**：checkpoint 臂的五件缺口；`STATE.f0` 与 `HASH.frames` 的帧号同源（runner 会在不符时 fail-closed 到 `frame_relation_conflict`，本轮未触发）。

---

## 8 本回执**不**主张

1. 不主张"2,000 步无效"：只证明**在 D1 规模、0.19 epoch、best-by-val 0.60 的条件下**闭环无改善。
2. 不主张加步数就能解决：val 在 step 900 饱和，说明瓶颈可能已不在步数。
3. 不主张条件接管已经能测学生：**checkpoint 臂还没实现**（缺 5 件），本轮只验证了 tape/constant 两臂。
4. 不主张 3 条磁带 = 3 个确认根：它们是 3 个不同入口状态，但都来自教师自己的录像，属条件接管。
5. 不主张 tape 臂 4,500 帧 0 死亡是能力：它是录像续放，按项目规则记 `not_a_policy_result`。
6. 不主张 `S1N-MB2` 可以冻结：三条前置验证里第 3 条只做了 1/3 根。

---

## 9 下一项能改变判断的实验（按代价排序）

| # | 实验 | 代价 | 会区分什么 |
| ---: | --- | --- | --- |
| 1 | **改采样/目标**再训一次（D1 上做类别平衡或 idle 降权），闭环复测 | 代码 + ~6 min GPU + 6 次启动 | 闭环 idle 坍缩是否由**数据/目标**造成（而不是步数） |
| 2 | 补齐 `S1N-MB2` 第 3 条验证（seed 11/12 的 constant 臂） | 2 次启动 | 退化臂是否 3/3 不完成 |
| 3 | 实现 checkpoint 臂（obs-v3 + SHM + 动作历史播种） | 代码 + 3 次启动 | 能否在入口 2,632 上直接测学生 |
| 4 | 用 T04 规模（6,592 行）做 2,000 步闭环（对应 9.7 epoch） | 0.5 min GPU + 6 次启动 | 把"预算"与"语料规模"分开 |
| 5 | 逐决策 idle 抑制探针（argmax 排除 idle 或加先验惩罚） | 代码 + 6 次启动 | 闭环坍缩是否只是"idle 吸收态"，而非能力问题 |

---

## 10 证据索引

| 结论 | 路径 | 身份 |
| --- | --- | --- |
| 2,000 步训练 | `.tmp_v832_s2a/train_2000.json` / `ckpt_2000.pt` | `48df0640…` / `19808e88…`（best-by-val step 900） |
| 闭环两臂 | `.tmp_v832_s2a/results.json` / `s2a.md` / `verify_s2a.py` | `d3341e99…` / `6072721f…` / `9128ac07…`（24/24） |
| before 臂 | `.tmp_v832_s2a/ckpt_init.pt` / `train_init.json` | `5cae8940…` / — |
| 接管 runner | `tools/rl/v8_32_takeover.py` | `4894363e30e47d8e…`（1,565 行） |
| 接管单测 | `tests/test_v8_32_takeover.py` | `93cc1b07b1be20b0…`（849 行，30 项） |
| 接管真机 | `.tmp_v832_takeover/verify/results.json` / `TAKEOVER_VERIFY.md` / `verify_takeover.py` / `ledger.jsonl` | `ed33d7f9…` / `79978571…` / `d4dcb417…` / `ec365c8e…`（18/18） |
| 接管原始 | `.tmp_v832_takeover/verify/{tape_seed10,tape_seed11,tape_seed12,const_seed10}/` | `result.json` + `decisions.jsonl`（935/935/935/316 行） |
| 下一张卡草案 | `Document/Update/Alpha/v8.31.0/TASK_CARD_S1N-MB2_DRAFT.md` | 草案，未冻结 |
