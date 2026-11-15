# W12 交接说明（v4.3.0）

日期：2026-11-15。作者：主代理。
**本文件不是完成报告。G410-0…7 全部未通过，总目标未达成。**
它的用途是：让下一个接手的人**不必重新推演**，直接按 §3 的清单执行。

---

## 1 已经做成的事（有证据路径）

| 包 | 交付物 | 证据 |
| --- | --- | --- |
| W00 | 身份、资产账本、`tools/rl/v430.py`（18 子命令，4 已实现 14 fail-closed）、6 个归位脚本 | `evidence/W00/`（16 文件） |
| W01a/b/c | `contracts.py` / `actions.py` / `session.py` / `evaluate.py`；**真实引擎 3+3 双跑全 `diffs=0`** | `evidence/W01/`（38 文件） |
| W02a | `observation.py` O2 编码（`alpha_mode` 四值） | `evidence/W02/` |
| W03a/a2/a3 | `sequence.py`（GAE、分段、burn-in） | `evidence/W03/` |
| W03b/f | `data/manifest.py`（33 问题码）、观测契约守卫 | 同上 |
| W03c | `models/actor.py`（2,726,671 参数） | 同上 |
| W03d/g | `learn.py`（BC + checkpoint + 防伪闸门） | 同上 |
| W04a | `ppo.py`（PPO 损失与诊断） | `evidence/W04/` |
| W04b | 守卫接线进 `train_from_manifest`、`batch_source` 由调用方声明 | `evidence/W04/W04B_OBS_GUARD_WIRING.md` |
| W09/b | `data/lineage.py` + `validate-splits` CLI | `evidence/W09/`（5 文件） |
| W03h/k/l | 三批真实 O2 语料（64 / 1000 / 1099） | `evidence/W03/W03*_*.md` |

**测试：`557 passed / 0 failed / exit 0`**（10 个测试文件，见 AGENTS.md §5.1 的运行方式）。

## 2 没有做成的事（真实缺口，不得含糊）

| 项 | 状态 | 原因 |
| --- | --- | --- |
| **G410-0…7** | **全部 `not_run_for_v430`** | 无 rollout、无评估、无过面证据 |
| G430-T/C/P/D | `not_run` | W02 战术腿、W04 课程、W05/W08 像素、W06 Windows 均未开工 |
| G430-X | 仅谱系腿 | W10 holdout、W11 第二作未做 |
| W02 全包 | 仅 observation 编码 | 战术/图注意力未做 |
| W04C 端到端学习者 | **不存在** | `ppo.py` 与 `sequence.py` 是孤岛，无任何模块把它们接进 rollout 循环 |
| **BC 有效性** | **负面结论** | 三组配置均不优于"恒预测多数类"基线（见 §4） |

## 3 下一步该做什么（按序，可直接执行）

### 3.1 先做（不依赖用户决策，不耗引擎）

1. **复跑全套测试确认基线**：
   ```powershell
   $env:PYTHONPATH='D:\WorkShop\Touhou_RL'
   python -m pytest -p no:cacheprovider --basetemp .tmp_v430_pytest_run -q `
     tests/test_v430_contracts.py tests/test_v430_actions.py tests/test_v430_session.py `
     tests/test_v430_observation.py tests/test_v430_splits.py tests/test_v430_sequence.py `
     tests/test_v430_models.py tests/test_v430_data.py tests/test_v430_learn.py tests/test_v430_ppo.py
   ```
   期望 `557 passed`。

2. **补 `o2-64` 的记账字段**（纯加法，**不要**改 `counts.decisions`，否则 `valid=False`）：
   ```bash
   /root/v400-venv/bin/python .tmp_v430_o2/recompute_manifest_stats.py --out-dir .tmp_v430_o2/o2-64
   ```
   补完立刻复跑 `validate_manifest` 与 `assert_o2_compatible`，两者都必须通过。

3. **统一三份 manifest 的记账模型**（门表第 18 条）：
   - 权威模型 = **1 shard / 语料**（`counts.decisions == 样本数`），即 `o2-strided` 那套；
   - `o2-1k` 需要迁移（当前 8-shard、`counts.decisions=8000`）；
   - 迁移后**必须**重跑 `validate_manifest` + `assert_o2_compatible`，并在证据里记明模型变更。
   - **注意**：迁移是**显式动作**，不能让"顺手重跑一次脚本"意外完成。

4. **解决 BC 的"成因未定"**（见 §4）——这是 W03/W04 的前置条件。

### 3.2 然后做（需要用户决策或引擎）

5. **`ASSET_SYNC.md` / `ASSET_SYNC.sha256` 登记**四批语料（见 §5）。
6. **W04C**：写端到端学习者，把 `sequence.py`（GAE）+ `ppo.py` 接进 rollout 循环。**当前仓库里不存在这个模块。**
7. **W02 战术腿**：图注意力 / 战术编码。
8. **W05/W08 像素门**、**W06 Windows**、**W10 holdout**、**W11 第二作**。

## 4 最重要的一条负面结果（必读）

**在真实 O2 语料上，BC 不优于"恒预测多数类"这一平凡基线。**

| 配置 | 参数 | train CE | seed12-val | seed10 | 基线 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ActorPolicy` 全分辨率 | 2,726,671 | **0.0293** | 1.8807 | 1.8837 | 0.8537 |
| 小模型 16× 下采样 | 30,793 | 1.2827 | 1.5306 | 1.6020 | 0.8537 |
| 小模型 4× 下采样 | 487,753 | **0.1558** | 2.1299 | 2.1161 | 0.8537 |

**成因未定**（四条未区分）：数据量 / 训练步数 / 架构不适配 / 标签可预测性有限。
主代理曾断言"容量是主因"，缩模型后未获支持，**已降级为假设**。

**含义**：在成因查清之前，**W03 的"真实微段 ≥9/10"不可能达成**。
不要用"loss 在降"当作进展——**必须与平凡基线比较**。详见 `evidence/W03_DATA_LEG.md §8`。

## 5 资产清单（供 `ASSET_SYNC` 登记）

| 路径 | 字节 | 说明 |
| --- | ---: | --- |
| `.tmp_v430_o2/o2-64/` | 109,021,159 | 64 样本，8-shard 模型，缺 `sample_count` |
| `.tmp_v430_o2/o2-1k/` | 553,082,238 | 1000 样本，8-shard + `sample_count` |
| `.tmp_v430_o2/o2-strided/` | 600,058,042 | 1099 样本，**1-shard 模型（正确）**，含 `_dry-keep60` 预演目录 |
| `.tmp_v430_o2/o2-stride-seed10/` | 约 600,030,604 | 1099 样本，1-shard；**缺 `labels_provenance.json`** |
| `.tmp_v430_o2/o2-stride-seed11/` | 采集状态见 `evidence/W03/W03M_*.md` | 同上协议 |
| `.tmp_v430_o2/engine-data/` | 496,960,622 | 私有 data 目录（引擎会写 `score.dat`） |
| `.tmp_v430_o2/engine-data-strided/` | 496,960,622 约 | 同上，stride 采集专用 |

**建议**：只登记正式产物（`.npy`/`labels.jsonl`/`manifest.json`/`meta.json`），
**不登记** `_dry-keep60` 之类过程目录。总规模约 **2.4–3.0 GB**。

## 6 绝不能碰的东西

- `.tmp_v400_d1`、`.tmp_v410_s1c`、`.tmp_v410_wp05_round*`、`/root/v400-data`、引擎二进制、checkpoint；
- `state.json`、`journal.jsonl`、`budget.jsonl`、`ASSET_SYNC.*`、既有证据文件；
- `third_party/**`（含 `th08-rl` 构建）；
- 扩容进程（`run_all.sh` / `run_worker.sh` / 它们起的 `th08-modern`）；
- **不得 `git add -A`**：实测 209 个文件 / 5,575,696 B 应加入，另有 209 个文件（约 45 MiB，含 `.tmp_*`、`.claude/`、`.dsh-acl-recovery/`、一个名为 `%sn  ; done` 的文件）必须排除。

## 7 卡住时先看这里（我踩过的坑）

| 现象 | 真实原因 |
| --- | --- |
| `manifest_sha256` 对不上 | 口径不同：项目用 **`text/lf-normalized`**，不是 raw 字节 |
| 想把 `counts.decisions` 改成样本数 | 8-shard 模型下会触发 `P_COUNTS_MISMATCH`；要么加 `sample_count`，要么换 1-shard 模型 |
| `KeyError: 'executed_cand'` | T02c 无该字段，用 T04 的 `slots.jsonl` |
| `KeyError: 'lives'` | 状态键是 `lives_remaining` |
| `FrameAlphaError: alpha plane holds intermediate values` | 真实引擎 alpha 从不为 255，需 `alpha_mode="any"` |
| `encode_o2` 报"需要 4 帧" | 每决策 2 帧，必须按**帧**喂 `deque(maxlen=4)`，不能按决策 |
| `.pytest_cache` → `os error 5 拒绝访问` | 用 `-p no:cacheprovider --basetemp .tmp_*` |
| PowerShell 把嵌套引号/CJK 弄坏 | 写 `.py`/`.sh` 文件，不要用 `python -c` |
| 两个采集共用私有 data 目录 | 引擎互写 `score.dat` → **必须各用各的** |
| 采到"帧号对但画面错"的数据 | 用了 filter-tape（改写磁带）而非 filter-during-replay；用 `schedule[::N]` 逐项比对可发现 |
| **"跨磁带泛化接近基线"** | **验证集被污染**：seed10 与 seed12 的前 **219 个决策逐字节相同**（开局确定性），占 20%。剔除后 CE 从 2.12 恶化到 **2.51**。**跨磁带验证必须先算最长共同 `cand` 前缀并排除** |
| **"loss 在降，所以学到了"** | **没有跟平凡基线比**。`move=0` 占 43%，"恒预测多数类"的 CE 只有 0.85；训练 loss 从 4.34 降到 3.20 **仍远差于该基线**。任何 BC 结论都必须并列 `majority-class` 基线 |

## 8 一句话

**基础设施已经可用且经真实数据验证；能力一个都还没有。**
下一步的第一件事不是写新模型，而是查清 BC 为什么连平凡基线都赢不了（§3.1 第 4 项）——否则在上面盖的东西都会塌。

## 9 证据分层（每条结论的可信度来自哪一层）

主代理在本会话中出过多次判断错误，因此把结论按证据强度分层。**引用任何一条前，先看它在哪一层。**

### L1 主代理亲手实测（本会话内亲自跑过命令、看到退出码）

- `557 passed` 全套测试（多次复跑）
- 三份 manifest 的记账缺陷（`counts.decisions` 8×、`dataset_id` 的 `-64`、`manifest_sha256` 的 LF 口径）
- `o2-strided` / `o2-stride-seed10` 的覆盖率、空白、provenance、`counts.decisions` 与真实行数相等
- stride-6 黄金闸门（`schedule[::6][1:]`，帧间隔恒 12，0 处不符）
- BC 三配置 vs 平凡基线（0.85）——**全部劣于基线**
- 跨磁带共同前缀 **219**（语料层）与 **1308–1344**（磁带层），两者换算吻合
- 三盘磁带 sha 互不相同、seed12 与冻结值相符
- `o2-1k` 上守卫接线训练闭环（含 legacy 被拒）

### L2 子代理报告 + 主代理复核过关键字段

- W03K 的 1000 决策采集（我复核了形状、覆盖率、`sample_count`、`manifest_sha256` 口径）
- W03L 的 1099 决策采集（我复核了黄金比对、`counts`、provenance/guard 日志）
- W04B 的守卫接线（我复核了 `run=True`/`run=False` 都拒 legacy、`batch_source` 三态）
- W00/W01/W02a/W03a–g/W09 各包的测试统计（我复跑过全套）

### L3 子代理报告，主代理**未逐项复核**

- 各子代理自报的"逐行改动"清单（我只核了行为，没逐行对照）
- `EXIT_CODES_*.txt` 里的个别退出码（我只抽验了关键的几个）
- W03M 的 seed11 采集：**在本会话结束时可能仍未完成，或完成但我未及复核** —— 见 §10

### L4 未验证 / 明确未知

- **BC 失败的真实成因**（数据量/步数/架构/标签可预测性，四条未区分）
- `shoot`/`bomb` 在**任何**数据量下能否学到东西（当前零信号，从未测过有信号的情形）
- 预乘是否成立（`max(RGB) ≤ alpha` 仅 0.03–0.12% 违例，未裁决）
- 换面之后的观测长什么样（**没有任何样本**）

## 10 本会话结束时的未完成项（不要当成已完成）

- **seed11 的 O2 语料**：采集状态以 `.tmp_v430_o2/o2-stride-seed11/` 与
  `evidence/W03/W03M_*.md` 的实际内容为准，**不要仅凭本文件推断它已完成**。
- **`o2-64` / `o2-1k` 的记账迁移**：未做（§3.1 第 2、3 项）。
- **BC 成因诊断**：未做（§3.1 第 4 项）。
- **G410-0…7**：全部未通过。
- **W02/W04C/W05/W06/W07/W08/W10/W11**：未开工。
