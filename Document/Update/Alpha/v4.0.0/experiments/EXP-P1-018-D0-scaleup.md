# EXP-P1-018 D0 规模化：60 → 50,000 教师决策（stage 0 / normal，双通道 + HASH 闸门）

- **日期**：2026-11-09
- **切片**：P3 D0 数据集放量（上游 [EXP-P1-014](<EXP-P1-014-D0-dual-channel.md>) 只做了 60 决策冒烟，`MIN_TEACHER_SAMPLES=50,000` 的训练闸门从未被触及）
- **依赖**：[EXP-P1-014](<EXP-P1-014-D0-dual-channel.md>)（双通道 + HASH 对齐 120/120）、[EXP-P1-013](<EXP-P1-013-teacher-obs-dataset.md>)（分片格式与 `dataset.py` teacher 模式）、[EXP-P1-012](<EXP-P1-012-g1-evaluation.md>)（stage 0 = 真 Stage 1，S1N 20/20 过面 0 死亡 → 该面可采）、[EXP-P1-010](<EXP-P1-010-search-numba-drift.md>)（`raster_numba.prange`）
- **一句话**：把 D0 从 60 决策放到 **53,512**（目标 50,000，stage 0 / difficulty normal，seeds 1–16，每片在 Stage 1 通关处如实提前收片），**16/16 片**都过 **HASH 逐检查点对齐**（107,024/107,024 = 1.000000，不等即 fail-closed）与 **`load_teacher_samples(require_observations=True)`** 加载闸门（16/16，摘要 `24f0ddac…`），总计 **1,356,224,832 B / 214,048 帧**，`partial=false`；本轮同时修掉三个规模化阻塞：**并行引擎闸门**、**分片/阶段级续跑**、**换面边界的 HASH 分歧**。

---

## 1. 目标与口径（先写死，后面对数）

| 项 | 值 |
|---|---|
| 面 | `--stage 0`（真 Stage 1）/ `--difficulty 1`（normal）/ `--mode fast` |
| 每片上限 | `--decisions 5000` + `--max-frames 20000`（4 帧/决策） |
| 提前收片 | `--stop-on-death` + 默认「换面即停」；**过面或死亡都如实结束该片并记录真实帧数** |
| 片数/seeds | seeds 1–20（不够可扩），每 seed 一片 |
| 通道 | A = `shadow` + `--labels-only`（出标签）；B = `env` + 私有 Xvfb + `SHM`/`READBACK 1`（出像素） |
| 并发 | ≤2 路；**同一片 A→B 串行**，不同 seed 可并行；我们的引擎 ≤4（2 路 ×(shadow parent+child 或 env)） |
| 质量门 | ①每片 assemble 前 HASH 全等 ②assemble 后 `dataset.load_teacher_samples(require_observations=True)` 样本数 = 决策数、摘要 `24f0ddac…` ③分片存在即拒绝覆盖 |
| 不做 | 不缩短帧数凑数、不跳过对齐、不伪造样本数、不用全 0 像素 |

---

## 2. 工具改动（`tools/rl/v400_d0_dual.py`，本轮）

| 新增 | 作用 | 为什么必须有 |
|---|---|---|
| `--engine-limit N` / `--engine-pattern P` | 等到本批引擎 < N 才启动，而不是等空机 | 原实现死等「全机 0 个 `th08-modern`」；本轮机器上始终有别的实验（v3/g1）在跑，原逻辑直接饿死 |
| `engine_processes()` 按 `/proc/<pid>/comm == th08-modern` 过滤 | 只数真引擎 | `pgrep -f` 会命中**本进程自己**（argv 里就有 `--engine-pattern …`）、`xvfb-run` 包装和 teacher；不过滤会把「自己的簿记」当引擎 |
| 分片级续跑 | 目标分片已存在且能被 `load_teacher_shard` 读回 → 同命令直接 `SKIPPED_COMPLETED`，不重跑 | 50k 要跑十几片、跨小时；中断后必须能同命令续跑 |
| 阶段级续跑（batch 侧） | `labels.jsonl` 在 → 从 `pixels` 续；`pixels.npz` 在 → 从 `assemble` 续 | 第一次失败点在最后一道闸门，A/B 都白跑了一遍 |
| `--max-frames/--stop-on-death/--allow-early-stop` | 把教师的「过面/死亡即停」透传，并**接受少于请求数**的决策，记录 `stop_reason`/`stage_clear`/`frames_committed` | 教师每片都会在 Stage 1 通关时停；原实现按「必须等于 5000」断言，会把**合法的提前收片**判成失败 |
| `--trim-transition`（见 §4） | 换面那一次提交的 `post` 不计入分片，其余全部要求相等 | 见 §4：这是唯一系统性分歧点 |
| `out_dir_lock()`（`flock`） | assemble 期间对 `manifest.json` 读改写加锁 | `manifest.json` 是读-改-写；双路并行 finalize 会丢条目 |
| `dual_channel.<key>.json` | 每片自己的双通道报告 | 原实现只写一个共享 `dual_channel.json`，双路会互相覆盖 |
| `.partial` 拒绝覆盖 + 存证 | 中断留 `.partial`（含 `incomplete.json`），重跑**不覆盖**、不静默删除 | 失败必须是证据，不是垃圾 |

`tests/test_v400_d0_dual.py`：**30 passed**（新增并行闸门/续跑/提前停/换面 trim/锁/元数据 11 例，全部无引擎、合成像素）。

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_d0_pytest_run -q tests/test_v400_d0_dual.py
# 30 passed
```

---

## 3. 发现一：39 线程的 numba 把引擎饿死；**单线程更快且标签逐行一致**

首轮按默认线程跑，两个 teacher 各 **39 线程**（`raster_numba.prange(frames)`），单进程烧 **330% CPU**，而 shadow 引擎只拿到 **8%**（178 s 墙钟里 15 s CPU）——负载 34 的机器上互相踩踏。改为 `NUMBA_NUM_THREADS=1`（连同 `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB/LP_NUM_THREADS=1`）后**墙钟反而更短**：

| 线程上限 | 120 决策墙钟（load ≈23） | 每决策 | 进程 CPU |
|---|---|---|---|
| 默认（39） | **12.53 s** | 104 ms | ~330% |
| 4 | 9.93 s | 83 ms | ~4 核 |
| 2 | 10.01 s | 83 ms | ~2 核 |
| **1** | **9.93–10.01 s** | **83 ms** | **1 核** |

**标签是否随线程数改变**：不。`labels-default/t4/t2/t1.jsonl` 去掉计时字段后逐行比对 → **0/120 差异**（`scores`、`verified_mask`、`chosen_plan`、`step_frames`、`hash_pre/hash_post` 全等），即 `prange` 只并行「同一张表的各帧」，不改变决策与轨迹。

**每决策实测耗时（channel A 单线程）**

| 环境 | 样本 | 墙钟 | 每决策 |
|---|---|---|---|
| 空载（load ≈2，smoke seed 91） | 150 决策 | 7.8 s（含启动 ~1.5 s） | **~42 ms** |
| 有负载（load 23–27，探针 seed 77） | 120 决策 | 9.93–10.01 s | **~83 ms** |
| 生产（load 27–42，1 路） | 3,323–3,371 决策 | 419–454 s | **~126–134 ms** |

生产比探针慢，是**外部负载**（同时段另有 v3 六面扫描与 Windows 零售 trace 批跑），不是线程配置回退。

---

## 4. 发现二：唯一系统性 HASH 分歧 = **换面那一次提交的 `post`**

前两片（seed 1、seed 2）在**最后一道闸门**失败：`exit 4`，`alignment.json` 显示 `checkpoints_mismatched = 1`：

| 片 | 决策 | 检查点 | 相等 | 首个不等 | 位置 |
|---|---|---|---|---|---|
| seed 1 | 3,371 | 6,742 | 6,741 | `decision 3370 post, frame 13484` | shadow `8a06e347…` / env `876bc773…` |
| seed 2 | 3,323 | 6,646 | 6,645 | `decision 3322 post, frame 13292` | shadow `21e9dc0a…` / env `feb1976f…` |

判据（不是猜测）：

1. 分歧**只**出现在最后一片「提交后」检查点；同一决策的 `pre` 相等（seed 1: `d026a264…`；seed 2: `4da6e9c9…`）；
2. 两侧**帧号相同**（13,484 / 13,292），且 A 的 `hash_post(k) == hash_pre(k+1)` 在整段内 3,369/3,321 处成立；
3. 教师在这一决策给出 `stop_reason=stage_changed`、`stage_clear=true`——**这一次 4 帧提交正好把 Stage 1 打通**，`post` 状态已经是**下一面的装载态**（shadow 经 `PARK`/`SPAWN` 子进程推进，env 在主循环推进，换面点两者不必一致）。

结论：**stage 内 13,480 帧逐帧一致，分歧点属于「下一面的 setup」，不是 stage 0 的玩法数据。** 处理方式（`--trim-transition`，默认关闭、显式开启）：

- 段**止于最后一片仍完全在 stage 内的决策**（seed 1 → 3,370；seed 2 → 3,322）；
- 触发条件严格：`stop_reason=stage_changed` 且 `stage_clear=true`、**恰好 1 个**不等、且该不等点**就是**最后决策的 `post`（另有 8 项计数交叉校验）；任何其它形态仍然 fail-closed（单测 3 例覆盖：无 flag、非最后点、未过面）；
- `alignment.json` **原样保留全部 6,742 个检查点（含被排除的那个）**；分片 `shard.json` 同时记录 `transition_trim{decision,frame,shadow_post,env_post,decisions_before/after,...}` 与 `alignment.hash_checkpoints.evidence{boundaries,equal,mismatched,sha256}`；
- 分片自身的检查点集（6,740 / 6,644）**全部相等**，`first_mismatch=null`。

> 这是**有据可查的排除**，不是把不等改成相等：被排除的那一条两侧哈希、帧号、原因都在 `shard.json` 里，原始证据文件没有被改写。

---

## 5. 质量门（每片都过，证据在分片里）

| 门 | 判据 | 实现位置 |
|---|---|---|
| HASH 对齐 | 分片内 `boundaries = 2×decisions`、`equal = boundaries`、`mismatched = 0`、`first_mismatch = null`；换面 trim 需额外满足 §4 的 9 项 | `assemble()` + `make_summary.py` 独立复核 |
| 换面排除可审计 | 原始证据恰 1 个不等、就是被排除点、两侧哈希都记下、保留前缀全等 | `make_summary.py::transition_trim_accounted` |
| A 侧连续性 | `hash_post(k) == hash_pre(k+1)` | `assemble()` + summary 独立复核 |
| 帧算术 | `hash_pre.frames = 4k`、`hash_post.frames = 4(k+1)`、每决策 4 个 `step_frames`、`decision_frame = 4k` | `load_labels()` + summary 独立复核 |
| 加载 | `dataset.load_teacher_samples(require_observations=True)`：样本数 = 决策数、`obs_unresolved = 0`、摘要以 `24f0ddac` 开头、`play (4,64,64)`/`hud (56,160)` | `verify_load()` + summary **重新加载一次** |
| 非空像素 | `require_rendered()` 拒全 0 读回；`zero_frames = 0` | `obs.py`（沿用 EXP-P1-013） |
| 不覆盖 | 分片目录已存在 → 写手直接拒绝；`.partial` 存在 → 本轮拒绝启动并保留 | `TeacherObsShardWriter` + `main()` |

### 5.1 数据真实性抽检（不是只看闸门）

`.tmp_v400_d0/audit_shards.py` 直接读分片（`load_teacher_shard` + 原始 `labels.jsonl`），确认观测是**真实渲染画面**、标签**非退化**。**16 片全量结果**（`.tmp_v400_d0/audit_shards.json`）：

| 指标 | 16 片的范围 / 合计 |
|---|---|
| 决策 | 3,274 – 3,466（**合计 53,512**，= `summary.json.decisions`） |
| 原始 label 行 | 3,274 – 3,466（**合计 53,512**，与决策一一对应） |
| `play` 均值 / 标准差 / 最大值 | 38.737–39.426 / 43.795–44.670 / 255（无全 0 片） |
| `play` **全 0 行** / **常量行** | **0 / 0**（16 片全为 0） |
| `hud` 全 0 行 | **0**（16 片全为 0） |
| 不同 4 帧掩码块 | **192 – 230** 种/片（合计 3,297 种） |
| 不同 `hash_pre` | 每片 = 该片决策数（**合计 53,512**，无重复状态） |
| `noop` 掩码 `(1,1,1,1)` 占比 | 884–1,110/片（约 27–32%） |
| `first_action` 种类 | **17**（每片都是 17） |
| 不同 18 维分数行 | 1,662 – 1,960 种/片 |
| `bomb=1` 决策 | **0**（stage 0 / normal 教师不需要放弹） |
| `stop_reason` / `stage_clear` | 全部 `stage_changed` / 全部 `true` |
| obs 摘要 | 16 片全部 `24f0ddac…` |

- 每片像素都**非空、非常量**；掩码 190+ 种、分数 1,600+ 种、状态哈希无重复 → 「全 0 像素」与「退化标签」都不是这批数据的形态。
- 空像素另有 `require_rendered` 在写入前拦截（`zero_frames=0`），不是靠事后统计。

---

## 6. 产物

```text
.tmp_v400_d0/
  batch.py                        # 分片调度（并行/续跑/里程碑/引擎与 load 采样）
  progress.json                   # 累计：决策/片数/HASH 率/加载校验/ETA/引擎数/loadavg
  campaign.json                   # 战役起点（重启不重置墙钟）
  summary.json                    # 汇总（每片 HASH 结果 + 独立重跑的加载校验）
  make_summary.py                # 汇总生成 + 独立复核
  logs/seed<N>.log               # 每片的完整驱动输出（含 COMMAND）
  work/seed<N>/                  # labels.jsonl / pixels.npz / alignment.json / channel_{a,b}.json
  d0/
    manifest.json                # v400-teacher-obs-manifest-1（totals 累计）
    shards/dual-s0d1seed<N>/     # labels.jsonl + obs/play64.npy + obs/hud.npy + shard.json
    dual_channel.dual-s0d1seed<N>.json
```

---

## 7. 规模结果（终稿，数字由 `make_summary.py` 从磁盘产物重新推导）

**结论：50,000 决策闸门已达成 —— 实际 53,512 决策 / 16 片，每片都过 §5 全部门，`partial=false`。**

| 指标 | 值 |
|---|---|
| **完成决策** | **53,512**（目标 50,000，达成率 107.0%） |
| 分片（成功/失败） | **16 / 0**（seeds 1–16 各一片） |
| 帧 | **214,048**（每片 13,096–13,864） |
| **HASH 检查点（分片保留集）** | **107,024 / 107,024 = 1.000000**，16/16 片 `first_mismatch=null` |
| 原始证据（含被排除的换面点） | 每片 `2×decisions+2` 个检查点、恰 1 个不等 = 该片的换面提交（16 例，全部有据排除） |
| **加载校验** | **16 / 16 片**：样本数 = 决策数、`obs_unresolved = 0`、摘要 `24f0ddace59104938dbe991c0d8efbb6a8cd21dfd5a29ebcb6d023b5b364a381` |
| **obs 总字节** | **1,356,224,832 B（1,293.4 MiB / 1.263 GiB）**，25,344 B/决策 |
| 墙钟 | 战役 **7,404 s（2.06 h）**，含首轮两次失败、换面诊断、1 路降速段与两次重启 |
| A+B 通道墙钟合计 | 9,029 s（每决策 **168.7 ms**：A ~128 ms + B ~70 ms） |
| 生产态速率 | 通道口径 **356 决策/分钟**；负载降到 loadavg≈2 的尾段约 **1,000 决策/分钟**（2 路） |
| 战役口径 | 53,512 / 7,404 s = **7.23 决策/秒**（含失败与诊断时间，保守） |
| 每决策字节 | 25,344 B（obs）+ ~2.6 KB（labels 行）≈ 27.9 KB |
| `partial` | **false** |

**每片明细**（A/B 为各自通道墙钟；「原始证据」= `alignment.json` 全量检查点，其中 1 个是换面排除点）：

| seed | 决策 | 帧 | A 墙钟 | B 墙钟 | 保留检查点 | 原始证据 | 加载 | 收片 |
|---|---|---|---|---|---|---|---|---|
| 1 | 3,370 | 13,480 | 431.5 s | 176.2 s | 6,740/6,740 | 6,741/6,742 | 3,370/3,370 | stage_changed（排除 3370） |
| 2 | 3,322 | 13,288 | 419.4 s | 177.6 s | 6,644/6,644 | 6,645/6,646 | 3,322/3,322 | stage_changed（排除 3322） |
| 3 | 3,346 | 13,384 | 453.7 s | 256.8 s | 6,692/6,692 | 6,693/6,694 | 3,346/3,346 | stage_changed（排除 3346） |
| 4 | 3,274 | 13,096 | 420.7 s | 231.8 s | 6,548/6,548 | 6,549/6,550 | 3,274/3,274 | stage_changed（排除 3274） |
| 5 | 3,394 | 13,576 | 433.1 s | 256.6 s | 6,788/6,788 | 6,789/6,790 | 3,394/3,394 | stage_changed（排除 3394） |
| 6 | 3,274 | 13,096 | 421.1 s | 236.2 s | 6,548/6,548 | 6,549/6,550 | 3,274/3,274 | stage_changed（排除 3274） |
| 7 | 3,346 | 13,384 | 420.1 s | 217.6 s | 6,692/6,692 | 6,693/6,694 | 3,346/3,346 | stage_changed（排除 3346） |
| 8 | 3,322 | 13,288 | 397.9 s | 233.5 s | 6,644/6,644 | 6,645/6,646 | 3,322/3,322 | stage_changed（排除 3322） |
| 9 | 3,466 | 13,864 | 341.1 s | 232.8 s | 6,932/6,932 | 6,933/6,934 | 3,466/3,466 | stage_changed（排除 3466） |
| 10 | 3,322 | 13,288 | 293.2 s | 154.9 s | 6,644/6,644 | 6,645/6,646 | 3,322/3,322 | stage_changed（排除 3322） |
| 11 | 3,322 | 13,288 | 243.4 s | 121.5 s | 6,644/6,644 | 6,645/6,646 | 3,322/3,322 | stage_changed（排除 3322） |
| 12 | 3,298 | 13,192 | 224.1 s | 117.1 s | 6,596/6,596 | 6,597/6,598 | 3,298/3,298 | stage_changed（排除 3298） |
| 13 | 3,274 | 13,096 | 208.6 s | 216.6 s | 6,548/6,548 | 6,549/6,550 | 3,274/3,274 | stage_changed（排除 3274） |
| 14 | 3,466 | 13,864 | 302.0 s | 222.7 s | 6,932/6,932 | 6,933/6,934 | 3,466/3,466 | stage_changed（排除 3466） |
| 15 | 3,322 | 13,288 | 369.0 s | 206.9 s | 6,644/6,644 | 6,645/6,646 | 3,322/3,322 | stage_changed（排除 3322） |
| 16 | 3,394 | 13,576 | 375.8 s | 215.5 s | 6,788/6,788 | 6,789/6,790 | 3,394/3,394 | stage_changed（排除 3394） |

**合计**：53,512 决策、214,048 帧、1,356,224,832 B；**16/16 片 HASH 全等、16/16 片加载通过**。

**这张表最值得注意的两件事**

1. **每片都在 Stage 1 通关处提前结束**（`stop_reason=stage_changed`、`stage_clear=true`、`deaths_observed=0`），实际 3,274–3,466 决策/片，而不是请求的 5,000。所以 50k 不是「10 片 × 5k」，而是 **16 片 × ~3.3k**——这与 EXP-P1-012 的人类 Stage 1 帧数 13,258 / 上限 19,887 完全一致（实测 13,096–13,864 帧）。
2. **每片恰好排除 1 个检查点**（换面提交的 `post`，§4）。16 片合计 16 个排除点，全部两侧哈希 + 帧号 + 原因记录在各自 `shard.json.transition_trim`，原始 `alignment.json` 未改写。若把 16 个排点也算成「不等」，则总不等率 = 16/107,040 = 0.015%，且每一处都可指向具体的换面帧——**不是散落的对齐失败**。

---

## 8. 与 P3 训练闸门的关系

- `dataset.MIN_TEACHER_SAMPLES = 50,000`：**只有决策数 ≥50,000 且每片都过 §5 的门，D0 才算可供 P3 的 BC/DAgger 训练**。本轮把闸门从「60（冒烟，不可训）」推进到 **53,512 决策 / 16 片 / 每片独立过门（HASH 107,024/107,024、加载 16/16）→ 闸门已满足**。
- `summary.json`：`decisions=53,512`、`complete=true`、`partial=false`、`shards_failed=0`、`hash_alignment_rate=1.0`、`load_verified_shards=16`。
- 与 EXP-P1-014 的差别：那轮只有 1 片 60 决策、只证明「通路 + 对齐」；这轮证明「规模化后每片仍逐检查点对齐」，并且发现了 60 决策冒烟**看不到**的换面边界（§4）与线程/负载问题（§3）。
- **整库一次性加载（P3 真正的入口）**：把 16 片的 `labels.jsonl` 一起交给 `dataset.load_teacher_samples(min_samples=50_000, obs_root=.tmp_v400_d0/d0, require_observations=True)` → **`count=53,512`、`obs_unresolved=0`、`obs_spec_digest=24f0ddac…`、`scores (18,)`、`play (4,64,64) uint8`、`hud (56,160)`、`alignment=decision_time_before_commit`**，`min_samples` 闸门通过（证据：`.tmp_v400_d0/dataset_load_16shards.json`）。
- 仍需注意的边界：本轮只覆盖 **stage 0 / normal**。P3 若要在其它面训练，需要按 §9 逐面再采；且每片都是**单 seed 的连续段**，多样性来自 seed（16 条不同轨迹），不是同一条轨迹的重复采样。

## 9. 下一步（DAgger）

1. 用 D0 训 BC（`dataset.BehaviorCloneDataset` + `load_teacher_samples`），先跑 `MIN_TEACHER_SAMPLES` 闸门内的 smoke（小步数、单面）。
2. DAgger 迭代：学生 rollout → 教师在这些状态上重新标注（`--labels-only` 同一通道，`RESET` 到学生到达的状态需要 replay 或 student-driven shadow）→ 聚合重训；**换面边界的排除规则同样适用**（每段止于换面前）。
3. 扩展面：stage 1..7 各采一批（每面仍需「教师可靠」前置；EXP-P1-012 的 PSN 已给每面死亡基线）。

## 10. 并发/环境记录（本轮真实状况）

| 时刻 | 全机 `th08-modern` | 我们的引擎 | loadavg | 备注 |
|---|---|---|---|---|
| 启动前 | 19 | 0 | ~34 | 另有 v3 六面扫描（stage 4/6/7，v3a/v3b）+ g1 残留 |
| 首轮（2 路，未限线程） | 23 | 4 | 34 | teacher 各 39 线程烧 330% CPU，引擎仅 8% |
| 限线程后（2 路） | 18–20 | 4 | 27 | 单线程 teacher，引擎恢复 |
| 协调后（1 路） | 12–28 | 1–2 | 13–42 | 按协调指令降为 1 路；g1 残留（`/root/v400-data/g1/inst*`）经确认为孤儿后由本任务精确清除（不碰 `d0s*`/`v3*`） |
| v3 停止后（2 路恢复） | 19–26 | 1–3 | 28–34 | resume 续跑，已完成片全部跳过 |
| v2f 收尾段（2 路） | 7–16 | 1–3 | 2–24 | 单片墙钟降到 350–375 s |
| 结束 | 15 | 0 | ~24 | 16 片完成 |

**并发口径**：本任务始终 ≤2 路，且**同一片的 A/B 严格串行**（驱动内部顺序执行）；我们的引擎实测 1–4 个（shadow parent + 瞬时 child 或 env），用 `--engine-pattern 'th08-modern --data-dir /root/v400-data/d0s'` 精确隔离，不会把别的实验算进来，也不会因别人的引擎而饿死。

**未触碰**：引擎/协议/依赖/第三方、`third_party`、`v3`/`v2f`/`g1` 的任何进程与数据（g1 孤儿仅在协调指令明确授权后按 `--data-dir /root/v400-data/g1/inst*` 精确清除）、Windows 侧零售 trace 批跑；未删除任何已完成分片或证据。

---

## 11. 复现与验收命令（含退出码）

```bash
# 0) 单测（无引擎，合成像素）
cd /mnt/d/WorkShop/Touhou_RL
PYTHONPATH=. /root/v400-venv/bin/python -m pytest -p no:cacheprovider \
  --basetemp .tmp_v400_d0_pytest_final -q tests/test_v400_d0_dual.py
# 30 passed

# 1) 全量批跑（resume：重跑同一条命令不会重做已完成片）
PYTHONPATH=. /root/v400-venv/bin/python -u .tmp_v400_d0/batch.py \
  --seeds 1-20 --workers 2 --target 50000 --threads 1
# exit 0；末行 DONE decisions=53512/50000 shards=16 partial=False

# 2) 汇总（磁盘独立复核：HASH + 重新 load_teacher_samples + 换面排除审计）
/root/v400-venv/bin/python .tmp_v400_d0/make_summary.py    # exit 0 → .tmp_v400_d0/summary.json

# 3) 数据真实性抽检（像素/标签非退化）
/root/v400-venv/bin/python .tmp_v400_d0/audit_shards.py    # exit 0 → .tmp_v400_d0/audit_shards.json
```

**失败路径也验证过**（不是只留成功日志）：首轮 seed 1/2 在换面点 `exit 4`（`ALIGNMENT_FAILED`），未写任何分片；`.partial` 拒绝覆盖；`--trim-transition` 关闭时，同样的输入仍然 fail-closed（单测 3 例）。
