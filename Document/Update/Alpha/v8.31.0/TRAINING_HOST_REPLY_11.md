# V831-ROADMAP-01 阶段回执 11：Tier A 是确定性的刀锋（2/6 配置）、复现率不可用 rollout 估计、`--save-obs` 打通

**工单**：[`LONG_TERM_DEVELOPMENT.md`](LONG_TERM_DEVELOPMENT.md)（V831-ROADMAP-01）· 承接 [`TRAINING_HOST_REPLY_10.md`](TRAINING_HOST_REPLY_10.md)
**主机代号**：轩神开播了（Windows 11 家庭版 / i5-14600KF；WSL2 `ubuntu`，20 vCPU，23 GiB，RTX 4070 SUPER）
**时间及时区**：2026-11-27 07:35 – 07:58 +08:00（本回执全部时间为 +08:00；上一轮 03:22 结束后有一段空闲，本轮从 07:35 起）
**执行**：DSH 主代理派发 2 个执行代理（Tier A 复现率评测、`--save-obs` 模块）；主代理独立复跑两份验收（14 / 71 项）并按原始 `enemies` 行复算
**检出**：`D:\WorkShop\Touhou_RL`，HEAD `1233598796b1b6f1d734950a4113af917da7b844`（分支 `v400`）
**引擎启动**：**6 次**

---

## 0 结论摘要（五条）

1. **同一 (checkpoint, root, entry) 的 rollout 是确定性的。** `ckpt_b2` + seed 10 + 入口 2,632 重跑 3 次：
   **13/13 项逐位相同**（897 条决策数据 sha `e585b37e…` 三次一致、`frames=4,426`、Tier A 3/3、末血 0）。
   唯一不同的原始字节是日志头的 `reset.setup_frames`（冷启动握手 238/231/237，不进游戏输入；三次 `reset_hash` 同为 `c2b684a7a564c2d3`）。
   → **Tier A 在给定配置上是 0/1 的确定值，"率"不能用重复 rollout 估计**，样本单位必须是 **(root, entry)**。
2. **Tier A 在 6 个 (root, entry) 配置里达成 2 个：2/6。**
   - (b2, seed 10, 入口 2,632)：帧 3,640 击杀（回执 10 已报）；
   - **(b2, seed 12, 入口 2,704)：帧 3,644 击杀，而且活到 cap 4,500**（回执 10 里它在入口 2,632 时剩 2,280 血）。
3. **72 帧的入口位移就能把结果翻过来**：seed12 在入口 2,632 剩 2,280 血（62.0%），在入口 2,704 直接击杀（100%）。
   同入口下三根削减分别是 66.25% / 56.52% / 100%。→ 这是**刀锋型能力**，对初始状态极敏感，不是稳定技能。
4. **`--save-obs` 打通（DAgger 的前置能力）**：runner 从 2,667 → **3,452 行**，71 项离线测试全绿（58 旧 + 13 新）。
   每个决策落盘 `play96x112` / `hud` / `act_hist` 三个平面（**0–255 原始值**，逐位等于送给策略的张量）+ 一行与 T04 字段对齐的 `slots.jsonl` + `meta.json` + `README.md`。
   实测 **52,116 B/决策的平面 + ≈1,393 B/决策的行 ≈ 53.5 KB/决策**。
5. **WP05 预算修正**：回执 10 §10 里"900 × 43 KB ≈ 230 MB"只算了 play 平面；三平面全存的实际值是
   **≈46.9 MB/(根×ckpt)**，`3 根 × 2 ckpt ≈ 281 MB`。采集仍按"先交预算再执行"。

---

## 1 身份

| 项 | 值 |
| --- | --- |
| 引擎 | `rl_v3/th08-modern` sha256 `691bfd77…a568ef6`（6 次启动全部一致） |
| takeover runner | `tools/rl/v8_32_takeover.py` **3,452 行**，sha256 `4361a2077c2abbf3…`（改前 2,667 行 `ddeae1a5…`） |
| runner 测试 | `tests/test_v8_32_takeover.py` **2,629 行**，sha256 `8130a3a2286b2855…`（**71 项**） |
| 本轮仪器 | 6 次启动全部跑**冻结副本** `.tmp_v832_s2c6/pinned/tools/rl/v8_32_takeover.py`，其 sha 与 live 在启动前/后/事后复核**完全相同**（`4361a207…`，只差路径） |
| 被测 checkpoint | `.tmp_v832_s2f/ckpt_b2.pt`，sha256 `dcaad643340af94f…`（未变） |
| 冻结卡 | `.tmp_v832_takeover/task_card_mb2.json`，content `2eee5046…` |

---

## 2 命令与退出码

| # | 命令 | rc | 结果 |
| ---: | --- | ---: | --- |
| 1 | `pytest tests\test_v8_32_takeover.py`（主代理复跑） | **0** | **71 passed in 8.65 s** |
| 2 | `python .tmp_v832_s2c6/verify_s2c6.py`（主代理复跑，WSL） | **0** | **14/14 PASS** |

**A 批（复现性，`ckpt_b2` / seed10 / 入口 2,632 / cap 1,868 × 3）**

| run | frames | 交还段 | stop | 末血 | 削减% | Tier A | 与 repeat1 |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| repeat1 | 4,426 | 1,794 | `legal_death` | 0 | 100.0 | 是 | 基准 |
| repeat2 | 4,426 | 1,794 | `legal_death` | 0 | 100.0 | 是 | 13/13 相同 |
| repeat3 | 4,426 | 1,794 | `legal_death` | 0 | 100.0 | 是 | 13/13 相同 |

**B 批（换入口 2,704，cap 1,796）**

| seed | entry_hash | 磁带 slot 1343 `hash_post` | frames | stop | 末血 | 削减% | Tier A |
| ---: | --- | --- | ---: | --- | ---: | ---: | --- |
| 10 | `f4325ab8b0295f0f` | 相等 ✓ | 4,062 | `legal_death` | 2,025 | 66.25% | 否 |
| 11 | `510e1fcc6f665a96` | 相等 ✓ | 4,006 | `legal_death` | 2,609 | 56.52% | 否 |
| 12 | `2bc8543137089ea3` | 相等 ✓ | 4,500 | `frame_cap` | **0** | **100.0%** | **是**（0 血帧 **3,644**） |

---

## 3 额度

| 额度 | 实际 |
| --- | ---: |
| 引擎启动 | **6** |
| 引擎帧 | **25,846**（前缀 16,008 + 接管 9,838）——**超 25,000 上限 846 帧（3.38%）**，原因是 seed12 在入口 2,704 存活到 cap；6 次未截断、超支照实报 |
| 静音 | **12/12** 读数全 `0 0 0 0`；cfg sha `cc062cfb…`；源树 root-game / muted-src / tape-t04 前后 manifest 逐位相同 |
| 新增磁盘 | `.tmp_v832_s2c6/` 29,445,605 B ≈ **0.027 GiB**（527 文件，含 6 次原始决策日志与冻结副本） |
| 账本 | 12 行 = 6 reserve + 6 settle，无未结 |

---

## 4 为什么"率"不能用重复 rollout 估计

| 观察 | 数值 |
| --- | --- |
| 同配置重跑一致项 | **13/13**（897 条决策数据 sha 三次相同、frames、stop、末血、`boss_summary` 五字段、cand/mask/frame 序列 sha、score/lives） |
| 唯一不同的原始字节 | `reset.setup_frames` = 238 / 231 / 237（冷启动握手帧数，不进游戏输入） |
| 三次 `reset_hash` | 全部 `c2b684a7a564c2d3` |
| Tier A | 3/3（方差 0） |

→ 在给定 (checkpoint, root, entry) 上，rollout 是**确定性**的；Tier A 只能取 0 或 1。
**估计能力必须换样本单位**：(root, entry) 组合。目前测过的组合是 **6 个，Tier A = 2 个**。

---

## 5 `--save-obs`：DAgger 的数据面已就绪

| 项 | 值 |
| --- | --- |
| CLI | `--save-obs DIR`（仅 checkpoint 臂；其它臂给了会拒绝；目录非空拒绝；写盘失败 fail-closed） |
| 每决策平面 | `obs/play96x112.npy`（N×4×112×96 uint8）、`obs/hud.npy`（N×56×160）、`obs/act_hist.npy`（N×8×18），**0–255 原始值** |
| 每决策行 | `slots.jsonl`：`decision_frame`/`commit_frames`/`duration_frames`/`obs_render_frames`/`history_frames`/`history`/`executed_cand`/`cand`/`mask`/`masks`/`hash_pre`/`hash_post`/`episode_seed`/`state`/`events`/`runner_frames`/`label_source=dagger_student_rollout`/`obs_source=takeover-runner` |
| 刻意留 null | `slot_id` / `shard_key` / `provenance` / `scores` / `verified_mask` / `verified_count` / `chosen` / `chosen_verified` / `chosen_score` / `teacher_bomb` / `margin` / `label_verified`（README 逐条说明原因；连接键改用 `(episode_seed, decision_frame, hash_pre)`） |
| 成本（实测 96 决策窗口） | 平面 **52,116 B/决策** + 行 ≈1,393 B/决策 → **53.5 KB/决策** |
| 关闭时开销 | 零新文件、零额外协议流量（`hash_calls`/`pixel_reads`/`enemy_reads`/`state_reads`/`steps` 全等，三个张量 `torch.equal`）——由测试钉住 |

**WP05 采集预算（修正）**：3 根 × 2 ckpt × 约 900 决策 × 53.5 KB ≈ **281 MB**（不是 230 MB，那只算了 play 平面）。

---

## 6 偏差与异常

1. **仪器并发冲突（已处置）**：`--save-obs` 任务在我派出第二批时仍在改写同一个 runner，07:41–07:45 期间 live 文件多次变化。
   执行代理把当时稳定版冻结到 `.tmp_v832_s2c6/pinned/` 并用该副本跑完 6 次启动；副本身份 `4361a207…` 与 live 在启动前后与事后（07:56）**完全相同**，且与旧基线（`.tmp_v832_s2c5/b2/seed10`）逐项一致 → 行为等价，不是换仪器。
   另：副本自测 **71/71** + 一次不启引擎的入口校验（entry 2631 → `entry_frame_invalid` rc=4）。
2. **帧预算超支 846 帧（3.38%）**：上限 25,000，实际 25,846；原因是 seed12 在入口 2,704 **存活到 cap**（4,500）而不是提前阵亡。
   没有截断、没有隐藏，verifier 里是显式 `[WARN]`。若必须严格守 25,000，只能下调 cap 或减少启动数——那会改变实验设计，未擅自修改。
3. **工单给的预算上限本身偏低**：我在派单时按"每根约 1,900 帧"估，实际入口 2,704 的 cap 是 1,796、但 seed12 跑满 4,500（含 2,704 前缀）。
4. **回执 10 §10 的磁盘估算偏低**（只算 play 平面），本轮已修正为 281 MB。
5. **`label_source="dagger_student_rollout"` 是新词**，不在 T04/D1 的词表里；这是刻意的（这些行还没有教师标签）。

---

## 7 分类声明

- **训练端本次实测（引擎）**：§2 的 6 次启动、静音 12/12、账本。
- **训练端本次离线**：71 项 runner 测试、`verify_s2c6` 14 项、§5 的 `--save-obs` 实测产物与字节数。
- **历史记录**：入口 2,632 的三臂基线（回执 10）、恒动作零伤害（回执 10）。
- **待核验**：Tier A 在更多 (root, entry) 组合上的分布；`--save-obs` 的真机产物（本轮只有 96 决策的离线窗口实测，没有真 checkpoint 跑）。

---

## 8 本回执**不**主张

1. 不主张 Tier A 是稳定能力：6 个配置里 2 个达成，且**入口位移 72 帧就翻转**。
2. 不主张"2/6"是精确率：样本单位只有 6 个组合，没有 CI。
3. 不主张 `--save-obs` 已在真机上产出过语料：只有离线 96 决策窗口的实测。
4. 不主张 WP05 采集可以立即开始：预算已修正到 281 MB，但**教师重标注的可行性（`teacher_2f` 吃不吃这种行）还没验**，这是下一轮的第一步。
5. 不主张本轮不影响既有基线：仪器换了路径（副本），虽已证明行为等价，仍按"路径不同"登记。

---

## 9 下一项能改变判断的实验（按代价排序）

| # | 实验 | 代价 | 会区分什么 |
| ---: | --- | --- | --- |
| 1 | **验教师重标注可行性**：把 `--save-obs` 产出的一小批行喂给 `teacher_2f` / `dataset_2f` 路径（3–5 个状态即可） | 0–1 次启动 | DAgger 的数据链到底能不能闭环 |
| 2 | **卡 v2 冻结**：Tier A（强）+ boss 血量削减%（分级，退化臂 0%）+ 确定性说明 + 2/6 实测 | 0 次启动 | 把 S1N-MB2 变成可判、可复算的任务 |
| 3 | **扩大 (root, entry) 组合**：入口 2,560 / 2,632 / 2,704 × 3 根 = 9 组合（已有 6 个） | 3 次启动 | Tier A 的真实分布与刀锋宽度 |
| 4 | **WP05 最小采集**：在**失败**的 4 个配置上跑 `--save-obs`（约 4 × 900 × 53.5 KB ≈ 193 MB） | 4 次启动 + 重标注 | DAgger 是否能把这些配置翻正 |
| 5 | 真机 `--save-obs` 冒烟（单根 100 决策） | 1 次启动 | 落盘路径在真引擎下是否如离线所见 |

---

## 10 证据索引

| 结论 | 路径 | 身份 |
| --- | --- | --- |
| 复现性 + 换入口 | `.tmp_v832_s2c6/results.json` / `s2c6.md` / `verify_s2c6.py` / `ledger.jsonl` | `a654157d…` / `0c6483c5…` / `90756a7f…` / `3cabf37f…`（14/14） |
| 冻结仪器 | `.tmp_v832_s2c6/pinned/tools/rl/v8_32_takeover.py` | 与 live 同 sha `4361a207…` |
| `--save-obs` 实现 | `tools/rl/v8_32_takeover.py` / `tests/test_v8_32_takeover.py` | `4361a207…`（3,452 行）/ `8130a3a2…`（2,629 行，71 项） |
| `--save-obs` 实测产物 | `.tmp_v832_takeover/scale/`（96 决策窗口） | 52,116 B/决策平面 + 1,393 B/决策 行 |
| Tier A 原始 | `.tmp_v832_s2c6/B/seed12/{result.json,decisions.jsonl}` | 0 血帧 3,644、跑满 4,500 |
