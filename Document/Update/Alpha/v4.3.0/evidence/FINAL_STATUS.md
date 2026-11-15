# v4.3.0 最终状态（本会话收束）

日期：2026-11-15。作者：主代理。
**判定：总目标未达成。G410-0…G410-7 全部未通过。本文件不声称任何完成。**

---

## 1 一句话

**基础设施已经建成并经过真实数据验证；能力一个都没有取得。**
v4.3.0 路线的 W00/W01/W02a/W03/W04a/W04b/W09 已完成并有证据；
W02 其余、W04C、W05–W08、W10、W11 未开工；**没有任何一个能力门通过**。

## 2 门表真实状态

| 门 | 状态 | 依据 |
| --- | --- | --- |
| **G430-E0** | **pass** | W00 身份/账本/CLI + W01 完整动作评测通路；真实引擎 3+3 双跑 `diffs=0` |
| G430-T | **not_run** | 战术腿未做 |
| G430-C | **not_run** | 课程学习未做 |
| G430-P | **not_run** | 像素反馈门未做 |
| G430-D | **not_run** | W06 Windows / W08 严格像素未做 |
| G430-X | **partial（仅谱系腿）** | W09 完成；W10 holdout、W11 第二作未做 |
| **G410-0…G410-7** | **全部 not_run_for_v430** | **无 rollout、无评估、无过面证据** |

## 3 交付清单（均有证据路径）

### 3.1 代码（13 个模块，557 测试全通过）

| 模块 | 规模 | 测试 |
| --- | ---: | ---: |
| `contracts.py` | 82.5 KB | 103 |
| `actions.py` | 28.0 KB | 44 |
| `session.py` | 70.6 KB | 90 |
| `observation.py` | — | 56 |
| `data/splits.py` | — | 77 |
| `sequence.py` | 39.5 KB | 64 |
| `models/actor.py` | 55.3 KB | 20 |
| `data/manifest.py` | 90.0 KB | 36 |
| `learn.py` | 51.8 KB | 29 |
| `ppo.py` | 25.5 KB | 38 |
| `evaluate.py` | 54.0 KB | （含 session） |
| `data/lineage.py` | 77.6 KB | （含 splits） |
| `tools/rl/v430.py` | 18 子命令 | 4 实现 / 14 fail-closed |

**`557 passed / 0 failed / exit 0`**（复跑确认，见 §7 命令）。

### 3.2 真实 O2 语料（四批，三批 stride-6 全部验收）

| 语料 | 样本 | `move` | `cand` | 记账模型 | provenance |
| --- | ---: | ---: | ---: | --- | --- |
| `o2-64` | 64 | 5/9 | 9/18 | 8-shard，缺 `sample_count` | 64/64 |
| `o2-1k` | 1000 | 6/9 | 10/18 | 8-shard + `sample_count` | 1000/1000 |
| **`o2-strided`**（seed12） | **1099** | **9/9** | **17/18** | **1-shard** | **1098 核对 0 不符** |
| **`o2-stride-seed10`** | **1099** | **9/9** | **17/18** | **1-shard** | **1098 核对 0 不符** |
| **`o2-stride-seed11`** | **1099** | **9/9** | **17/18** | **1-shard** | **1098 核对 0 不符** |

三批 stride-6 语料均：`counts.decisions == 实际样本数`、`episodes=1`、`shards=1`、
`validate_manifest valid=True problems=[]`、`assert_o2_compatible` 放行、空白帧 0。
**三盘磁带 sha256 互不相同** + **三份 `move` 直方图互不相同** → 三段真不同轨迹。

### 3.3 文档

| 文件 | 内容 |
| --- | --- |
| `GATE_TABLE.md`（23 KB） | 21 条问题、门状态、语料对照 |
| `W03_DATA_LEG.md`（8.3 KB） | 数据腿状态、三种记账模型、BC 负面结果、污染方法学 |
| `W12_HANDOFF.md`（11 KB） | 可执行清单 + 12 条坑 + L1–L4 证据分层 + 未完成项 |
| `CORRECTIONS.md`（6.8 KB） | 本会话 10 类错误自我审计 |
| `W00/`–`W09/` 各包证据 | 含退出码、实测输出、逐行改动 |

## 4 最重要的一条负面结果

**BC 在真实 O2 语料上不优于"恒预测多数类"这一平凡基线。** 三组配置实测（`move` 头，基线 CE = 0.8537）：

| 配置 | 参数 | train CE | seed12-val | seed10 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| `ActorPolicy` 全分辨率 | 2,726,671 | **0.0293** | 1.8807 | 1.8837 | 不优于 |
| 小模型 16× 下采样 | 30,793 | 1.2827 | 1.5306 | 1.6020 | 不优于 |
| 小模型 4× 下采样 | 487,753 | **0.1558** | 2.1299 | 2.1161 | 不优于 |

**成因未定**（四条未区分）：数据量 / 训练步数 / 架构不适配 / 标签可预测性有限。
我曾断言"容量是主因"，缩小模型后**未获支持**，已降级为假设。

**并且跨磁带验证本身有偏**：seed10 与 seed12 的前 **219 个决策逐字节相同**（磁带层 1308，约 20%），
剔除后 CE 由 2.1161 **恶化到 2.5070**。→ 污染是**偏乐观**的，剔除后结论更强。

**含义：在成因查清之前，W03 的"真实微段 ≥9/10"不可能达成。**

## 5 尚未开始的工作（不得含糊）

| 项 | 状态 |
| --- | --- |
| W02 战术/图注意力（除 observation 编码） | 未开工 |
| **W04C 端到端学习者** | **不存在**——`ppo.py` 与 `sequence.py` 是孤岛，无模块把它们接进 rollout |
| W05 像素反馈 / W08 严格像素 | 未开工 |
| W06 Windows 前端 | 未开工 |
| W07 完整周目运行 | 未开工 |
| W10 holdout（必须先落实共同前缀排除） | 未开工 |
| W11 第二作 | 未开工 |
| `o2-64`/`o2-1k` 记账模型迁移 | 未做 |

## 6 仍需用户决策的 4 件事

1. **`git add` 范围**：实测 **209 个文件 / 5,575,696 B** 应加入；另有 209 个文件（约 45 MiB，含 `.tmp_*`、`.claude/`、`.dsh-acl-recovery/`）必须排除。**`git add -A` 禁止**（会带入上述内容）。
2. **换行符归一化策略**：现有 `core.autocrlf` 状态未定；摘要口径已统一为 `text/lf-normalized`，但仓库层面策略需确认。
3. **扩容是否继续**：`.tmp_v420_data/T08/expand` 当前 **573/750**，占 3 个引擎；观测速率在 **0.19–0.53 状态/分**之间波动，剩余 177 个 → 约 **5.6–15 小时**（报区间，不外推点估计）。
4. **`ASSET_SYNC` 重钉**：四份语料约 **2.4 GB** + 两个 474 MB 私有 data 目录需登记；建议只登记正式产物，不登记 `_dry-keep60` 之类过程目录。

## 7 复现命令

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v430_pytest_run -q `
  tests/test_v430_contracts.py tests/test_v430_actions.py tests/test_v430_session.py `
  tests/test_v430_observation.py tests/test_v430_splits.py tests/test_v430_sequence.py `
  tests/test_v430_models.py tests/test_v430_data.py tests/test_v430_learn.py tests/test_v430_ppo.py
# 期望：557 passed
```

语料验收（任意一份）：

```powershell
python .tmp_v430_triage\verify_o2_corpus.py .tmp_v430_o2/o2-strided
# 期望：move 9/9、counts.decisions == actual、valid=True、assert_o2_compatible accepted
```

## 8 结论

**该做的做完了（基础设施），不该说的没说（能力）。**

下一步的唯一入口是 `W12_HANDOFF.md §3.1 第 4 项`：查清 BC 为什么连平凡基线都赢不了。
在它解决之前，往上盖任何东西都会塌。

## 9 发布记录

| 项 | 值 |
| --- | --- |
| 提交 | `9d81825826ffa881bab2bdcb6ee9c8be5e394c75`（短 `9d81825`） |
| 分支 / 远端 | `v400` → `origin/v400`（`github.com/LeventureQys/Touhou_RL`） |
| 父提交 | `6eb76f2` |
| 提交内容 | **132 个文件，全部为新增（`A`），0 删除** |
| 推送结果 | `6eb76f2..9d81825  v400 -> v400`，exit 0，无 force |
| 校验 | `origin/v400 == HEAD`；`rev-list --left-right --count` = `0 0` |

**提交范围（严格白名单，共 132 个文件）**：

| 路径 | 文件数 |
| --- | ---: |
| `touhou_rl/v430/` | 14 |
| `tests/test_v430_*.py` | 10 |
| `configs/v430/` | 3 |
| `tools/rl/v430.py` | 1 |
| `Document/Update/Alpha/v4.3.0/` | 104 |

**本次提交刻意**未**包含以下内容（它们在本会话开始前就已存在于工作区，与 v4.3.0 无关）：
- **3,493 条 `Document/**` 的删除**（若提交等于永久删除 3,319 个旧文档）
- **21 条无关修改**，含 `Document/Update/Alpha/v4.1.0/state.json`、`journal.jsonl`、
  `third_party/th08` 子模块指针
- `refs/` 的 PDF、`tests/test_v410_*.py` 等其余 341 条未跟踪文件、根目录 `.tmp_*`

**因此：`git add -A` 在本仓库是被禁止的**——它会连带提交上述删除。
提交后工作区仍保有那 3,493 条未暂存删除与 21 条修改，未被触碰。

**资产未上传**：五批 O2 语料（2.35 GiB）、私有 data 目录、引擎二进制均在 `.tmp_*` 下，
被 `.gitignore` 忽略，**没有进入本次提交**（符合 AGENTS.md §8.1）。
