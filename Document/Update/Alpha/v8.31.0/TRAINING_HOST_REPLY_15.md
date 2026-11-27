# V831-ROADMAP-01 阶段回执 15：DAgger 首训的负结果——标签通道接错了，训成了"模仿学生"

**工单**：[`LONG_TERM_DEVELOPMENT.md`](LONG_TERM_DEVELOPMENT.md)（V831-ROADMAP-01）· 承接 [`TRAINING_HOST_REPLY_14.md`](TRAINING_HOST_REPLY_14.md)
**主机代号**：轩神开播了（Windows 11 家庭版 / i5-14600KF；WSL2 `ubuntu`，20 vCPU，23 GiB，RTX 4070 SUPER）
**时间及时区**：2026-11-27 08:38 – 08:50 +08:00
**执行**：DSH 主代理派发 1 个执行代理（合并语料 + DAgger 重训）；主代理自查标签通道并执行磁盘清理
**检出**：`D:\WorkShop\Touhou_RL`，HEAD `1233598796b1b6f1d734950a4113af917da7b844`（分支 `v400`）
**引擎启动**：**0 次**（本轮全离线）

---

## 0 结论摘要（五条）

1. **合并语料建成**：`.tmp_v832_s2g/corpus/` = **22,888 行**（T04 19,772 + dagger 3,116），9 源、`slot_id` 连号、无去重冲突损失（仅 24 个重复键/27 行，已记录）。
2. **配方可复现**：用同一控制器按 `ckpt_m1` 的原语料重训 → **state 与 `ckpt_m1` 逐位相同（max_abs_diff = 0.0）**。
   于是 D1 与 ckpt_m1 之间**只差语料**，对照是干净的。
3. **负结果：加了 DAgger 数据反而更差。** D1（22,888 行）vs ckpt_m1（19,772 行），同为 6,000 步：
   - T04 三片（两者都在样本内）：`top1_vs_executed` **0.9495 vs 0.9649（−1.54 pp）**；
   - dagger 六套上 `top1_vs_chosen`（教师首选）：**0.1069 vs 0.1175（−1.06 pp）**；
   - 在"学生执行 ≠ 教师首选"的 **2,785 行**上：**0.0025 vs 0.0636（−6.10 pp）**；
   - 唯一大涨的是 dagger 行上的 `top1_vs_executed`（+39.70 pp）——**那是它的训练目标**。
4. **根因（主代理自查确认）：标签通道接错了。** 训练脚本对所有行都用硬 `executed_cand` one-hot 作目标；
   而 dagger 行的 `executed_cand` 是**学生自己执行的动作**（3,116 行里只有 331 行等于教师 `chosen`）。
   于是这 13.6% 的数据在 8.39 个 epoch 里教模型**复现学生的动作**——与 DAgger 的目的正好相反。
   教师标签 `chosen` 在行里存在，但没被用作目标；`bomb` 通道已经走教师侧（`bomb_source=teacher_bomb`），**唯独动作没走**。
5. **磁盘清理**：删除 joiner 的三份同 sha 中间语料（`real3`/`real3b`/`corpus_t04_3shard`），
   删前逐文件校验 5 个关键文件 sha 与 `corpus/` 全等、路径在 `.tmp_v832_join` 内、目录名匹配 → **回收 2.99 GiB**，`corpus/` 完好。

---

## 1 身份

| 项 | 值 |
| --- | --- |
| 合并语料 | `.tmp_v832_s2g/corpus/`（slots `7d8ebe0d…`、targets `abe5fd5c…`、obs `c0d67b00…`/`744c2f06…`/`10d27505…`），22,888 行 |
| D1 checkpoint | `.tmp_v832_s2g/ckpt_d1.pt`，10,469,383 B，sha256 `bcb9d4aaf1af69a1…`；6,000 步、8.39 epoch、192,000 样本、loss 2.8856→0.2444、训练 74.2 s |
| 对照 checkpoint | `.tmp_v832_s2e/ckpt_m1.pt`，sha256 `efab01f811c5a253…`（19,772 行、6,000 步、同配方） |
| 配方等价性 | 同控制器重训 ckpt_m1 → state 逐位相同（max_abs_diff 0.0） |
| 采样比例 | T04 165,886（0.8640）/ dagger 26,114（0.1360），每源 coverage 1.0 |

---

## 2 命令与退出码

| # | 命令 | rc | 结果 |
| ---: | --- | ---: | --- |
| 1 | `python -B .tmp_v832_s2g\verify_s2g.py`（主代理复跑） | **0** | **18/18 PASS** |

18 项含：语料行数与来源分布、T04 块与原 shard 逐位一致（0 label/obs mismatch）、三个 obs 平面 sha、ckpt 载荷 13 字段、
`ckpt_m1` 同预算且未变、**配方复现 ckpt_m1（max_abs_diff=0.0）**、D1 与 ckpt_m1 的 55 项指标重算、headline 差值、加载兼容、`engine_starts==0`。

---

## 3 离线对照（Δ = D1 − ckpt_m1）

| 源 | n | D1 `executed` | m1 `executed` | Δ | D1 `chosen` | m1 `chosen` | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t04_seed10 | 6,592 | 0.9487 | 0.9640 | −1.53 pp | 0.6500 | 0.6611 | −1.11 pp |
| t04_seed11 | 6,592 | 0.9484 | 0.9677 | −1.93 pp | 0.6553 | 0.6707 | −1.53 pp |
| t04_seed12 | 6,588 | 0.9513 | 0.9630 | −1.17 pp | 0.6488 | 0.6595 | −1.08 pp |
| dagger ×6（池化） | 3,116 | **0.9856** | 0.5886 | **+39.70 pp** | **0.1069** | 0.1175 | **−1.06 pp** |

- **D1 的 dagger 列在样本内**，ckpt_m1 的 dagger 列是分布外——所以 +39.70 pp 不是泛化，而是"把训练标签背下来"。
- 两模型在 **41.4%（1,291 行）** 的 dagger 行上选了不同动作：策略确实被改了，方向是**学生已执行的动作**。

---

## 4 根因与修法

**根因**：`s2g_train.py` 对所有行使用 **硬 `executed_cand` one-hot**（与历次 BC 配方一致）。
但 dagger 行的语义是"学生走到这里、教师说该走 `chosen`"：

| 行来源 | `executed_cand` 的含义 | 应当作为训练标签的是 |
| --- | --- | --- |
| T04 | 教师提交的动作 ✅ | `executed_cand` |
| dagger | **学生执行的动作** ❌ | **`chosen`**（教师首选）+ 与之一致的 `mask`/`bomb` |

实测佐证：dagger 3,116 行里 `executed_cand == chosen` 只有 **331 行（10.6%）**。

**修法**（下一轮）：
1. joiner 增加 `--dagger-label chosen`（默认）：对 dagger 行输出
   `executed_cand := chosen`、`executed_bomb := teacher_bomb`、`mask := cand_to_mask(chosen, teacher_bomb)`，
   并把原学生动作保留到 `source.student_cand` / `student_bomb`（可追溯，不污染标签）。
2. 重建语料 → 重训 D2（同预算 6,000 步）→ 与 ckpt_m1 对照，重点看 **dagger 行上的 `top1_vs_chosen`** 与 **T04 上的退让**。
3. 只有 `top1_vs_chosen` 明显上升、T04 退让可控，才值得进闭环评测。

---

## 5 磁盘清理（主代理执行）

| 项 | 值 |
| --- | --- |
| 删除目标 | `.tmp_v832_join/{real3,real3b,corpus_t04_3shard}` |
| 删前校验 | 5 个关键文件（`slots.jsonl`/`targets.jsonl`/三个 obs 平面）sha **与 `corpus/` 全等**；路径在 `.tmp_v832_join` 内；目录名逐一匹配 |
| 结果 | **回收 2.99 GiB**（4,293,909,977 → 1,088,519,102 B）；`corpus/` 完好（`slots.jsonl` 仍在） |

---

## 6 偏差与异常

1. **本轮没有闭环评测**：D1 的 Tier A 表现未测。理由：离线已显示标签通道错误，闭环评测会浪费 12 次启动且结论不可解释。
2. **语料未去重**：24 个重复 join key / 27 行保留（T04 与 dagger 的 (seed, frame, hash_pre) 理论上不会重叠，这些是 T04 片间的重复），已记录。
3. **`top1_vs_chosen` 在 T04 行上与 `executed` 相同**：因为 T04 行的 `chosen` 已由教师验证且等于 `executed`（历史口径）。
4. **4 份语料的 meta.json 略有字节差**（1,068,462,627 vs 1,068,464,110），但 5 个关键文件 sha 全等，删的是同 sha 的重复。

---

## 7 分类声明

- **训练端本次 GPU 实测**：corpus 构建、D1 训练 74.2 s、55 项离线指标重算。
- **训练端本次离线**：18 项验收、配方复现性（max_abs_diff 0.0）、§4 的根因自查。
- **主代理自查**：从 `slots.jsonl` 直接读出 dagger 行的 `executed_cand` vs `chosen`（331/3,116 相等）。
- **待核验**：修正标签通道后的 D2 效果；闭环 Tier A 变化。

---

## 8 本回执**不**主张

1. 不主张 DAgger 无效：本轮的负结果**由标签通道错误造成**，不是方法本身的判决。
2. 不主张 D1 的 +39.70 pp 有任何价值：它在样本内，且方向是复现学生。
3. 不主张修好标签通道就一定能提升：要下一轮实测。
4. 不主张 2,785 行"分歧行"的 −6.10 pp 有统计效力：单次训练、无重复。

---

## 9 下一项（R13）

| # | 任务 | 代价 |
| ---: | --- | --- |
| 1 | joiner 加 `--dagger-label chosen`（含 mask/bomb 一致性与 `student_cand` 追溯） | 0 次引擎 |
| 2 | 重建语料 + 重训 D2（6,000 步）+ 与 ckpt_m1 对照 | 约 2 min GPU |
| 3 | 若 `top1_vs_chosen` 上升：12 次启动复测 6 配置 | 12 次启动 |
| 4 | 若仍无改善：改用 `chosen` 权重或按分歧行加权（下一层实验） | 待定 |

---

## 10 证据索引

| 结论 | 路径 | 身份 |
| --- | --- | --- |
| 合并语料 | `.tmp_v832_s2g/corpus/` | `slots.jsonl` `7d8ebe0d…`，22,888 行 |
| D1 | `.tmp_v832_s2g/ckpt_d1.pt` | `bcb9d4aaf1af69a1…` |
| 训练与指标 | `.tmp_v832_s2g/train_s2g.json` / `s2g.md` / `verify_s2g.py` / `compat_s2g.json` | 18/18 PASS |
| 标签通道自查 | `.tmp_v832_wp05/labels/*/meta.json`（`agreement.ratio`）+ `.tmp_v832_s2g/corpus/slots.jsonl` | 331/3,116 相等 |
