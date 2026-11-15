# T02 复审证据：G420-0 显式门复审（据实改判）

- 日期：2026-11-13，Asia/Shanghai。资产主机 WSL2，`cwd=/mnt/d/WorkShop/Touhou_RL`，
  `/root/v400-venv/bin/python` 3.10.12，git HEAD `c3e116fd45d708dc8cbe73d4bb49cd960f835405`。
- 执行：Provider=DeepSeek / DeepSeek Flash（子代理）；本文件与其机读镜像由本轮实测生成。
- 门的原文：[版本计划](../../README.md) §5 WP420-01 末段与 §7 G420-0 行。
- 原判定：[`G420_0.md`](G420_0.md)（**not passed**，L1 fail / L2 pass / L3 pass），
  本文件**不修改它**，只引用并说明「原判定基于旧语料，本轮以新分母重判」。
- 机器可读镜像：[`g420_0_readjudicated.json`](g420_0_readjudicated.json)
  （sha256 `c2546571e09c564c6c5b56712bb495086ee96ca1e0c3142bc0b82730ce5c668c`，46,221 B）。
- 性质：**这是据实改判，不是重跑实验**。没有启动引擎、没有训练、没有写入
  `.tmp_v420_data/T02c|T02d|T04/**`。新增的实测只有「只读复核」（§2.6、§3、§4、§6）。

---

## 0 裁定

| 腿 | 原判定（旧分母） | 本轮裁定（新分母） | 依据 |
|---|---|---|---|
| **L1 时间/动作索引一致** | fail | **pass（改判）** | §2 被消费链 + §3 自检重跑 + §4 逐条判据 |
| **L2 HASH 重建零未解释差异** | pass | **pass（不变）** | §5 五次独立测量，全 0 不一致 |
| **L3 部署输入边界** | pass | **pass（不变，另加新语料复核）** | §6 |
| **G420-0 整门** | **not passed** | **passed** | §7 逐条对照 + §8 分母变更声明 |

**一句话**：A0/A0v 那一轮训练真正消费的每一个样本都是「2 帧一槽、自带 4 帧观测 + 自带
HUD + 自带已执行动作 + 8 条真实 2 帧历史 + 自带 verified 教师标签」的槽；同前缀重放重建出
同一条 HASH 链、零未解释差异；部署输入边界成立。**旧 4 帧语料仍然不满足 L1，只是不再是被
消费的训练源。**

---

## 1 复审口径（先写清能做什么、不能做什么）

本轮要做的事只有三件：**证实「被消费」是真的**、**用新分母重算 L1**、**逐条对照门原文**。
因此：

| 允许 | 不允许（本轮未使用） |
|---|---|
| 只读读取数据集、checkpoint、训练报告、日志、旧证据 | 重跑引擎、重跑训练、重放前缀 |
| 重新执行数据集自检（`self_check_dataset`） | 修改 `dataset_2f.py` 或任何 `touhou_rl/**` |
| 逐槽重算契约判据（29 项 / 573,355 条断言） | 复制标签、插值 logits、平移分数 |
| 用**冻结** evaluator 读 checkpoint 做一次前向（无优化步） | 放宽容忍度、改分母口径、把旧语料算合格 |
| 计算 sha256 / mtime / 算术一致性 | `git add/commit/push` |

判据文字以 [README](../../README.md) §5 WP420-01 与 §7 G420-0 行为准；**判据本身一个字未改**，
改的只是「所有**已消费**样本」这个集合的成员（§8）。

---

## 2 「被消费」的证实链（全部可复算）

脚本 `.tmp_v420_g420re/a_consume_chain.py`（只读），报告
`.tmp_v420_g420re/a_consume_chain.json`，**30 项检查全 `ok=true`，退出码 0**。六条互不依赖的证据：

### 2.1 训练命令行与输入路径参数

`.tmp_v420_t04/train.sh`（sha256 `e65e0ca300ab542a4e85479dcef81ce880cdd6c8caf5454310682ca65be96693`）：

```bash
export PYTHONPATH="$ROOT420"
"$PY420" tools/rl/v420.py train --config "$CONFIG" --out "$OUT" --execute \
  > "$LOG" 2>&1            # CONFIG=configs/v420/learn-controlled.json
                           # OUT=.tmp_v420_data/T04/train.json
```

`configs/v420/learn-controlled.json`（sha256 `cc1ae9dcda76ad11de298b9c8722c80438ac8e87f606bcc19427fe2030d0577c`）
**没有 `dataset_dir` 键**。`touhou_rl/v420/dataset_2f.py` L207-209 推出
`dataset_root = <v420_data>/<dataset_run_id>/dataset`，配置里 `v420_data=/mnt/d/WorkShop/Touhou_RL/.tmp_v420_data`、
`dataset_run_id=T04` ⇒ **`/mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T04/dataset`**，与磁盘上的语料根逐字符相等。
训练日志第一行就是这条绝对路径：

```
[train] loading /mnt/d/WorkShop/Touhou_RL/.tmp_v420_data/T04/dataset
[train] corpus 19772 slots; train 15819 / holdout 3953
```

> **登记一个观察（不是门缺陷）**：`train.py` L413-415 只对 `config.dataset_dir` 做类型校验，
> 并不把它用于路径解析。所以「换一份别的语料」只能通过 `v420_data` / `dataset_run_id` 做到，
> 而这两个值在配置里、路径在日志里、内容在下面的 sha256 里 —— 这条链没有留下未登记的入口。
> 但一个写了 `dataset_dir` 的未来配置会被静默忽略，建议契约所有者后续改成「出现即拒绝」或「真的生效」。

### 2.2 语料身份：run 记录的值 vs 现在重算的值

| 对象 | run 里记录的 sha256 | 本轮重算 | 一致 |
|---|---|---|---|
| `dataset/index.json`（3,941 B） | `dbb17ec2…2b09a8`（全 `dbb17ec201a63a0c91112739c80e44de2cbd55d3712a25271ed17770672b09a8`） | 同值 | ✅ |
| `dataset_build.json`（36,789 B） | —（不是 run 记录的字段，本轮单列） | `3f48d9779cd06f4b0f88e773eb1c38225703c4df6988b662939d213ed1e414ed` | — |
| seed10 `slots.jsonl`（10,100,966 B） | `52402de44692fa2decd019da9701f57de6e27f193aea01c1e19d2b69fab781a2` | 同值 | ✅ |
| seed11 `slots.jsonl`（10,104,250 B） | `f75e632ba6990a372b4bae614f51bbc74049a75296380491b558b9071f09eb9e` | 同值 | ✅ |
| seed12 `slots.jsonl`（10,097,148 B） | `f759de7dacc78343dec8825da5de14942e749b8a5a2c5319304e513d1618b7b9` | 同值 | ✅ |
| seed10/11/12 `meta.json` | `321ca74c…a2007c` / `04ed2f11…9057b4` / `11d30d31…e92bc1` | 同值 | ✅ |
| `obs/play96x112.npy`×3、`hud.npy`×3、`act_hist.npy`×3 | —（symlink 指向 T02c，`SOURCE.json` 记录身份） | 逐文件 sha256 已登记（§11） | — |

`train.json` 的 `dataset.dataset_root`、`dataset_version=v420-2f-dataset/1`、`slots=19,772`、
`slots_with_verified=19,772` 与磁盘重算逐项相同；`index.json` 的 `totals.slots=19,772` 亦然。

### 2.3 split 复算

`row_index % 5 == 4` ⇒ **train 15,819 / holdout 3,953**（0.199929），与 `train.json.split` 和日志第二行相同；
本轮重算的不合规 holdout 行 **0**、train∩holdout **0**。

### 2.4 样本数算术（把 640,000 这个说法纠正清楚）

| 量 | 值 | 依据 |
|---|---:|---|
| 语料总槽 | 19,772 | 6,592 + 6,592 + 6,588 |
| 训练行（80%） | 15,819 | §2.3 |
| `steps_per_epoch = ceil(15,819/128)` | **124** | 日志 `steps/epoch=124`、两臂 `throughput`、两个 checkpoint 的 `budget` |
| 完整 epoch 的末批行数 | **75** | 15,819 − 123×128 |
| 5,000 步 = 40 完整 epoch + 40 步 | 40×15,819 + 40×128 | divmod(5000, 124) = (40, 40) |
| **`samples_seen`（可复算）** | **637,880** | 与两臂记录的 `samples_seen`、两个 checkpoint 的 `budget.samples_seen` 完全相等 |
| `epochs = ceil(5,000/124)` | **41** | **与 checkpoint 的 `epochs=41` / `last_epoch=41` 自洽** |

> 「5,000 × 128 = 640,000」只有在**每一批都满**时才成立。留出 20% 永不参与训练，所以一个
> epoch 是 15,819 行而不是 19,772 行：124 步/epoch，且完整 epoch 的最后一批只有 75 行。
> 真实呈现数是 **637,880**，与记录值逐位相同。若按全语料算，19,772/128 = **154.47**（向上取整
> 155）步/epoch —— 那是全语料口径，**不是这一轮实际用的节拍**。**没有任何一项需要写 `unknown`**：
> 每一步都能从记录的字段反推出来并相互印证。

### 2.5 两个 checkpoint

| 臂 | 目标列 | 路径 | 字节 | sha256 | epochs |
|---|---|---|---:|---|---:|
| A0 | `executed` | `.tmp_v420_data/T04/checkpoints/a0_seed0_step5000.pt` | 2,940,699 | `96cb219eac83391f42b18ea656b5a6c5ac81e97cd1fd8c1e2c0d404aee6b89ad` | 41 |
| A0v | `chosen_verified` | `.tmp_v420_data/T04/checkpoints/a0v_seed0_step5000.pt` | 2,940,741 | `f7188c0d773e1085299e797327fd62a2bcec35bf50c115cb203d2ad2fa1f4fd9` | 41 |

两者 schema 均 `v410-wp04-bc-arm-checkpoint/1`，`budget = {epochs:41, batch_size:128, steps_per_epoch:124,
steps:5000, samples_seen:637880}`，`split = {train_slots:15819, holdout_slots:3953, rule:"row_index % 5 == 4"}`，
`dataset = {dataset_version:"v420-2f-dataset/1", seeds:[10,11,12], slots:19772}`，`v420_arm`/`v420_target` 分列。
两个 sha256 与 `train.json.artifacts` 记录的值相等。

### 2.6 前向复现：checkpoint 与今天的语料是同一对

`.tmp_v420_g420re/d_forward_repro.py`（**只前向、无优化步**）用**冻结**的
`tools/rl/v410_student_eval.py::load_v410_checkpoint` 读回 checkpoint，在**今天的语料**上重算指标：

| 臂 | 块 | 记录（CUDA） | 本轮重算（CPU） | Δ |
|---|---|---:|---:|---:|
| A0 | holdout loss | 2.802158335165189 | 2.8021581137549676 | **2.21e-07** |
| A0 | holdout top1 | 0.5378193776878321 | 0.5378193776878321 | **0.0** |
| A0 | train loss | 0.05099899561102999 | 0.050999178759041475 | 1.83e-07 |
| A0 | train top1 | 0.9957013717681269 | 0.9957013717681269 | **0.0** |
| A0v | holdout loss | 2.64209283814292 | 2.642096250339191 | **3.41e-06** |
| A0v | holdout top1 | 0.5621047305843663 | 0.5621047305843663 | **0.0** |
| A0v | train loss | 0.09273530447685206 | 0.09273578449507329 | 4.80e-07 |
| A0v | train top1 | 0.9875466211517795 | 0.9875466211517795 | **0.0** |

`macro_recall` 与 `proper_safe_set_rate` 两侧**完全相同**（A0 0.4386541866936504 / A0v 0.5621047305843663）。
容忍度在**看数之前**声明：loss 绝对差 ≤ 5e-4、top1 差 ≤ 1/15,819（一行训练样本的量）。实测差比容忍度小两个数量级。

⇒ 这比「文件 sha 相同」更强：**同一对 (checkpoint, 语料) 在 15,819 + 3,953 个槽上给出同一组预测**，
argmax 一个都没变。语料若在训练后被换过，不可能同时复现 loss 与逐槽 argmax。

### 2.7 mtime 次序（弱旁证，不作证明）

观测 `.npy` 09:50:03 / 09:54:06 / 09:58:22 → 语料 `slots/targets/meta` 10:38:21–10:38:30 →
`index`+`build` 10:38:32 → 配置 10:49:05 → runner 10:51:38 → A0 checkpoint 10:54:29 →
A0v checkpoint 10:56:31 → `train.json`/`train.log` 10:59:04（均 UTC）。次序与「先建语料、再训练、最后写报告」一致。
**这一条只是旁证，不是证据**：mtime 可以被复制/触写改变。

---

## 3 数据集自检重跑（无引擎、无训练）

`touhou_rl/v420/dataset_2f.py` **没有独立 CLI**，所以本轮写了一个只读入口
`.tmp_v420_g420re/b_selfcheck.py`（**不改** `dataset_2f.py`，只调用它的
`load_dataset` / `self_check_dataset` / `dataset_summary`）。`dataset_2f.py` 在本次复核时的
sha256 是 `94341de6de35b25c7d9199d783f79b269b5dae5aa6bcc1bb5500bbd80e2cdbc9`（87,280 B）。

```bash
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v420_g420re/run_all.sh   # 内含 b_selfcheck
```

| shard | seed | 行 | targets 行 | `violations_total` | `unexplained_diffs` | `passed` | hash 对 | verified flags |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `d1-s0d1-n3300-seed10-shard` | 10 | 6,592 | 6,592 | **0** | **0** | true | 6,592 | 6,592 |
| `d1-s0d1-n3300-seed11-shard` | 11 | 6,592 | 6,592 | **0** | **0** | true | 6,592 | 6,592 |
| `d1-s0d1-n3300-seed12-shard` | 12 | 6,588 | 6,588 | **0** | **0** | true | 6,588 | 6,588 |
| **合计** | — | **19,772** | **19,772** | **0** | **0** | true | **19,772** | **19,772** |

- `label_source_counts = {teacher_2f: 19,772}`；`verified_flags_total = 19,772`；`explained_checks = []`
  （重建/join 语料没有「可解释的旧格式失败」，任何违反都会计入未解释）。
- **36 个检查项、514,117 条断言、0 违反**。逐项计数（`checked` / `violations`）已写进
  [`g420_0_readjudicated.json`](g420_0_readjudicated.json) 的 `self_check_rerun.per_check`。
  其中 26 项逐槽执行 19,772 次；文件级项：`hash_reconcile`/`join_*`/`obs_rows`/`targets_rows` 各 3 次、
  `obs_shape_dtype` 9 次、`source_unchanged` 12 次、`obs_blank_play`/`obs_blank_hud` 各 3 次。
- **退出码 0**。

---

## 4 L1 逐条判据（在新分母上逐槽重算）

脚本 `.tmp_v420_g420re/c_l1_audit.py`（只读）。它**不采信**自检结果，而是从
`slots.jsonl` + `targets.jsonl` + 三个 `.npy` 平面 + 冻结契约 `contracts.py` 重新算一遍。
**29 项判据 / 573,355 条断言 / 0 违反 / 退出码 0。**

| # | 判据（门原文的拆解） | 实测 | 违反 |
|---|---|---|---:|
| C1 | 每 2 帧一槽：`slot_id == 行号`、`decision_frame == 16 + 2·slot_id` | 19,772/19,772 | **0** |
| C1 | `commit_frames == [d, d+2)` 且 `duration_frames == 2` | 19,772/19,772 | **0** |
| C1 | 相邻槽帧号差恰为 2 | 19,769/19,769 | **0** |
| C2 | 每槽 obs 窗口 == `[d-4, d-3, d-2, d-1]`（4 帧连续） | 19,772/19,772 | **0** |
| C2 | **相邻槽的 4 帧栈重叠恰 2 帧**（`play[s-1][2:4] == play[s][0:2]`） | 19,769/19,769 | **0** |
| C2 | `play` 行非全零 | 19,772/19,772 | **0** |
| C3 | 每槽自有 HUD 行（帧 `d-1`）非全零 | 19,772/19,772 | **0** |
| C3 | 每槽自有已执行动作，且与 `executed{}` 三字段一致 | 19,772/19,772 | **0** |
| C3 | 存盘 `act_hist[s]` == `history[s]` 的 one-hot | 19,772/19,772 | **0** |
| C4 | history 长度 8 | 19,772/19,772 | **0** |
| C4 | history 帧窗 == `[d-16, d-14, …, d-2]`（8×2 = 16 帧） | 19,772/19,772 | **0** |
| C4 | 无未来：任何 history 帧 `>= d` 的槽 | 19,772/19,772 | **0** |
| C4 | 最近一条 history 就是帧 `d-2`（不含当前） | 19,772/19,772 | **0** |
| C4 | history 值全在 `[0,18)` | 19,772/19,772 | **0** |
| C4 | **跨槽一致**：槽 `s` 的 8 条 history == 槽 `s-8..s-1` 的已执行动作 | 19,748/19,748 | **0** |
| C5 | `mask ↔ (cand, bomb)` 双向往返 | 19,772/19,772 | **0** |
| C6 | `scores` 长度 18 / 全有限 | 19,772/19,772 | **0** |
| C6 | `verified_mask` 长度 18，`verified_count == sum(mask)` | 19,772/19,772 | **0** |
| C6 | `chosen` == verified 集合的 argmax（并列取最小索引） | 19,772/19,772 | **0** |
| C6 | `chosen_score` 与 `scores[chosen]` 一致（6 位小数） | 19,772/19,772 | **0** |
| C6 | `chosen_verified` / `label_verified` 与掩码一致 | 19,772/19,772 | **0** |
| C6 | 每槽至少有 1 个 verified 候选 | 19,772/19,772 | **0** |
| C7 | `hash_pre`/`hash_post` 均为 16 位 hex | 19,772/19,772 | **0** |
| C7 | 逐槽状态链 `hash_pre[s] == hash_post[s-1]` | 19,769/19,769 | **0** |
| C8 | `label_source ∈ contracts.LABEL_SOURCES`（5 源元组） | 19,772/19,772 | **0** |
| C9 | `targets.jsonl` 的两列分开且与记录相等（不得互换/填充） | 19,772/19,772 | **0** |

两条**本轮新增、前序证据没做过**的检查值得单独指出：

1. **相邻槽重叠 2 帧**（C2）：19,769/19,769 成立。它证明 `play` 的 4 帧窗口确实是
   「每 2 帧往前推 2 帧」抓出来的连续帧，而不是把同一个 4 帧块（或 4 帧块的错位拷贝）
   重复引用两次 —— 这正是旧语料结构上做不到的那一步。
2. **跨槽 history 一致**（C4）：槽 `s` 的 8 条历史等于槽 `s-8..s-1` 的已执行动作
   （19,748/19,748，前 8 槽没有前驱故不计入）。它证明 history 不是「另取的一串数」，
   而是**同一轨迹里已经提交过的动作**，即门要求的「已实际执行、非教师建议」。

契约源码：`LABEL_SOURCES = ("teacher_slot0", "teacher_slot1", "teacher_2f", "executed_slot1", "none")`
（`touhou_rl/v420/contracts.py` L138-139，5 源，另附 [T01b 契约缺口修复](../T01/CONTRACT_LABEL_SOURCES.md)）。
本轮消费语料的 `label_source` 全部是 `teacher_2f` —— 已注册且在规则表里
（`teacher_scored=true`、`verified_policy=optional`、允许携带 teacher 字段）。

---

## 5 L2：HASH 重建零未解释差异（pass，未变）

| 来源 | 规模 | 不一致 |
|---|---|---:|
| T02 在线腿（每 seed 两次独立重放互比） | 3 seed × 2,016 帧 = **6,048** 对 | **0** |
| [PREFLIGHT_REPLAY](../PREFLIGHT/PREFLIGHT_REPLAY.md)（旧语料前缀重放） | 3 seed × 2 次 × 2,000 帧 = **12,000** | **0** |
| [T02c](../T02c/REBUILD_2F.md)（重建语料逐边界复核） | `hash_post` **9,898/9,898** + **39,592** 逐帧 | **0** |
| [T02d](../T02d/RESLOT_2F.md)（逐槽标签重放） | **29,694** 个边界 + **79,184** 条帧 HASH | **0** |
| 本轮 part B/C（语料级链 + 与两侧 meta 对账） | **19,772** 个逐槽 hash 对、链断裂 **0**、链头 3/3 | **0** |

原始机读记录：`.tmp_v420_data/T02/time.json` → `g420_0.legs.l2_hash_reconstruction`
（`frames_compared 6048 / mismatches 0 / comparisons 3`，`status=pass`）。
**没有放宽容忍度**；`forward_cross_device` 仍是 `available=false`，不被计入任何一致数。

---

## 6 L3：部署输入边界（pass，未变，另在新语料上复核）

| 判据 | 原测量（T02 在线腿） | 本轮在新语料上复核 |
|---|---|---|
| 4 帧严格连续且 == `d-4..d-1` | 6,000/6,000，违反 0 | **19,772/19,772**（C2） |
| 历史 8 条 × 2 帧 = 16 帧窗 | 6,000/6,000 | **19,772/19,772**（C4） |
| 无未来 / 不含当前 | 6,000/6,000 | **19,772/19,772**（C4） |
| 决策跨度与真实持续帧数 | `duration_frames=2` 实测 | **19,772/19,772**（C1） |
| mask 往返与槽内稳定 | 6,000/6,000 | **19,772/19,772**（C5） |
| 形状/dtype | play `(4,112,96)`、hud `(56,160)`、act_hist `(8,18)` 全 uint8 | 3 个 shard 的 `play (N,4,112,96)` / `hud (N,56,160)` / `act_hist (N,8,18)` 全 **uint8** |
| 冻结模型在该输入上前向 | 在线/离线 6,000/6,000 逐比特相等（cpu→cpu，max abs diff **0.0**） | A0/A0v 在 15,819+3,953 槽上复现记录指标（§2.6） |
| 截断 / 无法归因 | 0 / 0 | 冷启动 24 槽**排除**而非填充 |

原始机读记录：`.tmp_v420_data/T02/time.json` → `g420_0.legs.l3_deploy_input_boundary`
（10 项检查各 6,000 次、违反全 0、`forward_mismatches 0`，`status=pass`）。
**边界含义**：这里成立的是「输入边界」，不是「像素够用」、不是「实时抓帧不陈旧」、不是「零售客户端时序」。

---

## 7 逐条对照门的原文要求

原文（[README](../../README.md) §7）：**G420-0 时间与运行有效 —— 所需证据：远端原模型复现；
2 帧样本/历史/输入/像素一致；零未解释 HASH 差异。**

| 门要求的条款 | 证据 | 本轮是否实测 | 判定 |
|---|---|---|---|
| 远端原模型复现 | [T02 复现腿](LEGACY_REPRODUCE.md)：4 次调用 12 局全退出码 0，6/6 个 (checkpoint, seed) × 7 字段与历史基准**完全相同**、`action_sequence_sha256` 相同（[PROGRESS](../../PROGRESS.md) §3.3） | 否（引用既有实测） | **满足** |
| 2 帧样本一致 | §4 C1（每 2 帧一槽、commit span 2、相邻槽差 2、commit/history 无未来） | **是**（19,772 槽） | **满足** |
| 历史一致 | §4 C4（8×2 帧、`d-16..d-2`、跨槽等于已执行动作、act_hist == one-hot） | **是** | **满足** |
| 输入/像素一致 | §4 C2/C3（4 帧连续、相邻槽重叠 2 帧、每槽自有 HUD/动作）+ §6 在线腿前向 6,000/6,000 | **是**（19,772 + 6,000） | **满足** |
| 零未解释 HASH 差异 | §5 五次独立测量，全 0；本轮 19,772 个逐槽 hash 对、链断裂 0 | **是** | **满足** |

另有 WP420-01 的原文要求「新 S1 数据每个 2 帧决策槽都要有对应标签；可用原掩码重放并在
缺失槽重新查询教师；仅复制上一条标签、插值 logits、把一条计成两条均不成立」：

- 每个槽都有**自己的** 18 路 `scores` + `verified_mask`（19,772/19,772）；
- 标签来自 [T02d](../T02d/RESLOT_2F.md) 的**逐槽重新查询**（19,796 次查询、1,266,944 内部帧、71.52 ms/query），
  且是**算出来的**证据：第 1 槽的排序与同块第 0 槽不同者 **7,718/9,886 = 78.07%**、
  verified 候选不同者 **5,406/9,886 = 54.68%** —— 复制/插值不可能产生这些值；
- 没有 `pad`、没有复制第一帧、没有白名单例外：24 个冷启动边界**被查询但不入语料**。

**整门裁定：G420-0 = passed。**

---

## 8 分母变更声明（这是本轮唯一改变的东西）

| 项 | 旧分母（原判定用的） | 新分母（本轮用的） |
|---|---|---|
| 语料 | v4.1.0 D410-S1 / `.tmp_v410_s1c` 的**4 帧** shard | `.tmp_v420_data/T04/dataset/`，版本 `v420-2f-dataset/1` |
| 逻辑槽 / 登记行 | 6,600 / 3,300 | **19,772 / 19,772** |
| 每槽自有 4 帧 play | 3,300（**50%**） | **19,772（100%）** |
| 每槽自有 HUD | 3,300（槽 1 无 `d+1` 的 HUD 行） | **19,772（100%）** |
| 每槽自有标签 | 3,300（槽 1 无 scores） | **19,772（100%，`teacher_2f`）** |
| history 跨度 | 8 × 4 = **32 帧**（部署是 16 帧） | 8 × 2 = **16 帧** |
| 谁在消费 | v4.1.0 的 S1c 训练源 | A0/A0v（5,000 步 × 128，seed 0，15,819 训练行） |
| 判定 | fail | **pass** |

**这不是放宽，也不是把旧语料算成合格：**

1. 判据一个字未改。2 帧节拍要求**每个槽**都有自己的观测/标签 —— 这是旧 4 帧写盘格式
   **结构上无法表达**的东西（槽 1 的观测与评分根本没被写下来）。换语料只会让要求更难过，不会更容易过。
2. **旧 4 帧语料仍然不满足 L1**，而且它的 6,600 槽分母**没有被合并**进新分母、没有被修补、
   没有被重算。它现在的身份是**只读历史材料**（`G420_0.md` §6.1.1 的 fail 结论继续有效）。
3. 新分母上的每个数字**由本轮独立重算**（§3 自检、§4 29 项判据），不是从产出方的文档里抄的。
4. 冷启动 24 个边界（每 shard 8 个，帧 0–14）**排除而不填充**；它们不是被消费样本。

---

## 9 这次复审**不能**主张什么

1. **G420-0 通过 ≠ 第一面能力达成。** 版本计划要求 G420-0、G420-4、G420-5 **一起**通过才能写
   「v4.2.0 第一面能力达成」。G420-4 **not passed**（四臂标准 S1N 全 **0/20**，Wilson 上界 0.1611），
   G420-5 **not started**（无候选）。见 [DEV_EVAL](../T04/DEV_EVAL.md)、[PROGRESS](../../PROGRESS.md) §1。
2. **不能把 A0/A0v 说成能力。** 两者在 DIAG-FIRST-DEATH 与标准 S1N 上都是 **0/20**；
   留出 top1 只有 0.5378 / 0.5621、留出 CE 2.802 / 2.642（均匀基线 `ln18 = 2.8904`），
   且 A0v 在两个口径上都低于 B0。本文件里出现的 loss/top1 **全部**是「数据被消费且能被优化」的
   证据，不是棋力。层 1 的 safe-set 0.6875 / 1.0 是**记忆**，见 [LAYER1](../T04/LAYER1.md)。
3. **不能把 L2 的通过推广到未测状态。** HASH 相等证明的是「这三条已录轨迹的同前缀重放
   在另一进程里重建出同一条链」。它**不**说明其它 seed、后面几面、其它难度/机体、
   其它引擎构建也成立。本次覆盖：stage 0 / Normal / team 0 / seeds 10-11-12，
   seed 12 还是 `stage_changed` 收尾（3,298 决策）。
4. **不能把 L3 的通过读成「像素够用」。** L3 成立的是**输入边界**（4+1 帧、8×2 历史、形状/dtype、
   无未来、冻结模型在该输入上的前向），它不涉及实时抓帧、截图陈旧、缩放/颜色/归一化、
   零售客户端输入延迟 —— 那些属 G420-5。
5. **不能主张跨设备逐比特一致。** 前向一致性是 cpu→cpu；§2.6 是 cpu（现在）对 cuda（当时记录），
   差到 3.41e-6（loss）且 top1 逐位相同，这是「同一对 (模型, 语料)」的强证据，**不是**逐比特等价证明。
   原审计的 `forward_cross_device.available=false` 依然成立，仍不计入任何一致数。
6. **不能主张「18 路 verified 排名」。** 本语料每槽恰好 **1** 个候选经引擎验证
   （`verify_k=1`，`verified_flags_total = 19,772`，即向量位的 5.56%），另外 17 个 score 带
   `-5.0` 的未验证折扣。「每槽都有自己的 verified 标签」成立；「18 路都验证过」不成立。
7. **L1 的范围就是被消费的那份语料**：seeds 10/11/12、9,898 决策、19,772 槽。
   S1c 其余 shard（13–20）既没重建也没重标，所以这不是「全部 S1 数据都合规」。
8. **不能主张「引擎今天还能产出这些像素」。** 本轮没有启动引擎，只重读了已发布的字节、
   重跑了契约检查器与冻结模型的前向。
9. **语料的 obs 是 symlink 到 T02c 的产物。** 身份只在本文记录的 sha256 上成立；
   移动或重编码 T02c 的 `.npy` 会作废它。
10. **不能用本轮算法层面的沉默当默认。** 本轮只审「时间/动作索引 + HASH + 输入边界」，
    没有审训练目标是否合理、没有审 5,000 步是否够、没有审标签密度是否要用 `verify_k>1`。

---

## 10 复算命令与退出码

全部在资产主机 WSL 内，`cwd=/mnt/d/WorkShop/Touhou_RL`，`PYTHONPATH=` 仓库根：

```bash
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v420_g420re/run_all.sh
wsl -e bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && PYTHONPATH=/mnt/d/WorkShop/Touhou_RL \
  /root/v400-venv/bin/python .tmp_v420_g420re/f_write_evidence.py'
```

| 部件 | 脚本 | 报告 | 退出码 | 覆盖 |
|---|---|---|---:|---|
| A 被消费链 | `.tmp_v420_g420re/a_consume_chain.py` | `a_consume_chain.json` | **0** | 30 项检查（配置/路径/身份/split/算术/checkpoint/日志） |
| B 自检重跑 | `.tmp_v420_g420re/b_selfcheck.py` | `b_selfcheck.json` | **0** | 3 shard / 36 检查 / 514,117 断言 / 0 违反 |
| C L1 判据 | `.tmp_v420_g420re/c_l1_audit.py` | `c_l1_audit.json` | **0** | 29 判据 / 573,355 断言 / 0 违反 |
| D 前向复现 | `.tmp_v420_g420re/d_forward_repro.py` | `d_forward_repro.json` | **0** | 两臂 × (train, holdout)，CPU，无优化步 |
| E 出处清单 | `.tmp_v420_g420re/e_provenance.py` | `e_provenance.json` | **0** | 45 个对象全在（missing 0）+ L2/L3 机读数 |
| F 写裁定 | `.tmp_v420_g420re/f_write_evidence.py` | `g420_0_readjudicated.json` | **0** | **fail-closed**：任一部件不通过即拒写 |

F 的守卫是硬的：只要 A/B/C/D/E 任一 `ok=false`、退出码非 0、`violations_total≠0`、
`unexplained_diffs≠0`、样本数≠19,772、或前向复现超出容忍度，它就**退出 3 并且不写文件** ——
「passed」这份文件不可能由一次失败的测量生成。

---

## 11 产物、出处与只读足迹

本轮新增/修改（严格白名单内）：

| 文件 | 内容 |
|---|---|
| `Document/Update/Alpha/v4.2.0/evidence/T02/G420_0_READJUDICATED.md` | 本文件 |
| `Document/Update/Alpha/v4.2.0/evidence/T02/g420_0_readjudicated.json` | 机读镜像（46,221 B，sha256 `c2546571…ce5c668c`） |
| `.tmp_v420_g420re/{a..f}_*.py`、`run_all.sh`、`*.json`、`*.log`、`env.txt`、`exit_codes.txt` | 只读复核脚本与逐部件报告 |

**未修改**（全部只读）：`touhou_rl/**`、`tools/**`、`configs/**`、`tests/**`、
`evidence/T02/G420_0.md`、`evidence/T03/**`、`evidence/T04/**`、`Document/Update/Alpha/v4.1.0/**`、
`ASSET_SYNC.*`；`.tmp_v420_data/T02c|T02d|T04/**` 只读。**未启动引擎、未训练、未 git 写操作。**

原判定文件的身份（读时记录，供后续核对未被触碰）：
`evidence/T02/G420_0.md` sha256 `c94f5afd440bceeebcd20a805ce107e182c446496a081581907cd08645426b0e`、
26,388 B、mtime `2026-11-13T10:21:44Z`。

关键对象 sha256（全部由 `.tmp_v420_g420re/e_provenance.py` 现算）：

| 对象 | 字节 | sha256 |
|---|---:|---|
| `T04/dataset/index.json` | 3,941 | `dbb17ec201a63a0c91112739c80e44de2cbd55d3712a25271ed17770672b09a8` |
| `T04/dataset_build.json` | 36,789 | `3f48d9779cd06f4b0f88e773eb1c38225703c4df6988b662939d213ed1e414ed` |
| seed10 `obs/play96x112.npy` | 283,508,864 | `5d3cf82a1ed70ca05e0dc1667c8b466016e461cb08c515b78ba51a07f0cf4eaf` |
| seed11 `obs/play96x112.npy` | 283,508,864 | `e94d6e8fcae1e6f7dabd1a2d93b86610292f76fcf42f768f4f155e29cfd78473` |
| seed12 `obs/play96x112.npy` | 283,336,832 | `e88be0561452287a57e0d1e9d3bf57771af48d2d10ab39c9b77ab84755f80ff1` |
| `T04/train.json` | 167,645 | `72afc751c783f7185e1e1e7c5e15a83b6153a32b83ad83a87b4bae6318f353b9` |
| `T04/train.log` | 176,386 | `846b4cdb3b3c98c0f526e23635c4c7d524796ab4ad05f1feb25c9c6e111b07c6` |
| `checkpoints/a0_seed0_step5000.pt` | 2,940,699 | `96cb219eac83391f42b18ea656b5a6c5ac81e97cd1fd8c1e2c0d404aee6b89ad` |
| `checkpoints/a0v_seed0_step5000.pt` | 2,940,741 | `f7188c0d773e1085299e797327fd62a2bcec35bf50c115cb203d2ad2fa1f4fd9` |
| `configs/v420/learn-controlled.json` | 4,477 | `cc1ae9dcda76ad11de298b9c8722c80438ac8e87f606bcc19427fe2030d0577c` |
| `touhou_rl/v420/dataset_2f.py` | 87,280 | `94341de6de35b25c7d9199d783f79b269b5dae5aa6bcc1bb5500bbd80e2cdbc9` |
| `touhou_rl/v420/train.py` | 55,803 | `bf4c2c1bf99015536127e5bf4f28b804ebcb74645b23b187011a722a4849a7b9` |
| `touhou_rl/v420/contracts.py` | 56,321 | `1357c69c32a74fda20553a3e6b290ca2102b7cc5c52e47e9fdc6eb9e08c946c1` |

---

## 12 剩余阻塞（与本次改判无关，照实登记）

1. **G420-1 not passed**（20/200 状态、proper **0/100**、祖先 3/10）：现有「固定教师续行」配方
   在同类状态上不可达，只有换性质（冻结学生续行 / 真危险状态 / 拉长 H）才可能达成。
2. **G420-2 not passed**：A1 无可用信号（T03 `proper=0`）、A2 未建、只有 seed 0。
3. **G420-3 / G420-R 未开始**。
4. **G420-4 not passed**：四臂标准 S1N 全 0/20；**G420-5 未开始**。
5. **规模缺口**：语料只覆盖 3 个 shard；seed 13–20 需另做 rebuild + reslot
   （约 +52,800 教师查询，占 650k 上限的 43.88%）。
