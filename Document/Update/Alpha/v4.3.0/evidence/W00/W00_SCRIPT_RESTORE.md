# W00 脚本归位：`.tmp_v420_expected.py` → `tools/rl/v420_expected.py`、`.tmp_v420_action_mix.py` → `tools/rl/v420_action_mix.py`

- 日期：2026-11-15（Asia/Shanghai）
- 范围：v4.3.0 W00 的「P1-07 脚本收尾」半 —— 把被 v4.2.0 报告引用、但滞留在仓库根临时命名空间的两个聚合脚本归位到可同步的源码路径
- 依据：`Document/Update/Alpha/v4.3.0/V420_REVIEW.md` P1-07（第 78–80 行）
- 原始退出码：同目录 [`EXIT_CODES_RESTORE.txt`](EXIT_CODES_RESTORE.txt)
- 本档**没有**启动游戏引擎、**没有**训练、**没有**评测、**没有** git add / commit；没有修改、删除、移动、重命名任何 `.tmp_v420_*` 原件

## 0 结论（一句话）

独立复核确认：v4.2.0 的 53 个 `.md` 里被点名的**根级** `.tmp_v420_*.py` 共 6 个，其中只有 **2 个**是报告表格数字的唯一产出脚本（`PREFLIGHT.md` §4 的期望值表、`ACTION_MIX.md` §1 的动作分布表），**与前序结论一致，不多不少**；两者都是纯只读 JSON 读取、无个人路径、无子进程/网络/写盘，**已安全归位**并经合成输入与真实数据双重验证（stdout 与原件逐字符相同，缺输入时 exit 2）。原件仍在仓库根、sha256 未变。

同时更正两处前序事实（§1.3）：60 个根级 `.tmp_v420_*` **并未被 `.gitignore` 忽略**（`git check-ignore` 0 命中，全部是 untracked `??`）；且 v4.2.0 里还有 **168** 个被点名的 `.tmp_v420_*/` 子目录脚本仍被 `.gitignore:32` 挡住 —— W00 只收口了其中 2 个根级文件，**P1-07 未被整体关闭**（§3.3）。

## 1 引用复核（独立重做，不照抄前序）

### 1.1 范围与方法

- 扫描对象：`Document/Update/Alpha/v4.2.0/` 下递归得到的 **53 个 `.md`**（含 `README.md`、`PROGRESS.md`、`ROUTE_CONCLUSION.md`、`evidence/**/*.md`、`experiments/**/*.md`）
- 方式一（精确文件名）：`Select-String -LiteralPath <53 md> -Pattern <脚本名> -SimpleMatch`，逐名统计命中处
- 方式二（无前缀变体）：`grep -E 'expected\.py|action_mix'` 覆盖 `.tmp_v420_` 之外可能的写法
- 方式三（全目录、不限扩展名）：对 `v4.2.0/` 全目录（含 json/jsonl/txt）grep 两个名字，确认没有漏在非 md 文件里的引用
- 计数复核：`Get-ChildItem -LiteralPath . -Filter '.tmp_v420_*' -File` → 60 个（`.sh` 42 / `.py` 18），与任务给的分布一致

### 1.2 确证被引用的脚本 → 出处

| 脚本 | 引用出处（文档:行号） | 引用原文 | 报告里它产出的东西 |
|---|---|---|---|
| `.tmp_v420_expected.py` | `evidence/PREFLIGHT/PREFLIGHT.md:144` | 「两者均**只读**引用，未修改。提取脚本 `.tmp_v420_expected.py`。」 | `PREFLIGHT.md` §4 期望值表（B0 / 20k × seed 1000/1001/1002 的 frames / steps / score / stop_reason / `action_sequence_sha256`，以及两个 20 dev 的 `frames_mean`），也是 T02 复现腿的判据来源 |
| `.tmp_v420_action_mix.py` | `evidence/T04/ACTION_MIX.md:5` | 「- 脚本：`.tmp_v420_action_mix.py`（只读）」 | `ACTION_MIX.md` §1 的 8 格表（局数 / 决策数 / idle 占比 / focus-idle / 出现动作数 / 前三动作） |

两个名字在 53 个 md 里各只出现 **1 次**；用无前缀写法（`expected.py`、`action_mix`）搜索**仍然只有这 2 处**，不存在第三种写法或第二个模块。→ **前序结论（只有 2 个真正被依赖）成立。**

### 1.3 与前序结论的差异（两处更正 + 一处补充）

**差异 1（更正，影响 P1-07 的定性）：60 个根级 `.tmp_v420_*` 并不被 `.gitignore` 忽略。**

- `.gitignore:32` 的规则是 `.tmp_*/`，**结尾斜杠只匹配目录**；根级普通文件 `.tmp_v420_expected.py` 不匹配
- `git check-ignore -- <60 个根级文件>` → **exit 1，输出 0 行**（一个都没命中）
- `git status --porcelain -- <60 个根级文件>` → **60 行，全部 `??`**
- 对照组：`git check-ignore -v .tmp_v420_adjudicate/adjudicate_corpus.py` → `exit 0`，命中 `.gitignore:32:.tmp_*/`（子目录**确实**被忽略）

即：根级 `tmp` 脚本的状态是「**untracked 但未被忽略**」，不是「被忽略」。后果有两面：① 它们确实不会随正常同步进入源码树（与原结论同向，归位决定不变）；② 但 `git add -A` / `git add .` **会把 60 个临时脚本一并扫进索引** —— 这比"被忽略"更危险，也让 `git status` 长期挂 60 行噪声。本档不改 `.gitignore`（不在白名单内），只登记这个事实供后续决策。

**差异 2（更正）：`W00_SOURCE_IDENTITY.md:199` 说「这些 `.tmp_v420_*` 脚本硬编码 WSL/`xvfb-run` 路径，在 Windows 端跑不起来」——对这两个不成立。**

两个脚本都不 import torch、不碰引擎、不调 shell，Windows `python`（3.12.10）直接跑通，退出码 0，并**逐格复现** v4.2.0 的两张表（§5.4）。该说法对 42 个 `.sh` 编排脚本大概率成立，但不能套到这两个纯 JSON 聚合脚本上。

**差异 3（补充，不改结论）：另外 4 个 probe 与 3 个 shell 其实也被文档点名。**

| 脚本 | 文档引用处 |
|---|---|
| `.tmp_v420_slot_probe.py` | `PREFLIGHT.md:101`、`experiments/T01-contract-spec.md:18`、`:28` |
| `.tmp_v420_mask_probe.py` | `PREFLIGHT.md:101`、`experiments/T01-contract-spec.md:23` |
| `.tmp_v420_release_probe.py` | `PREFLIGHT.md:101`、`experiments/T01-contract-spec.md:24` |
| `.tmp_v420_hist_probe.py` | `PREFLIGHT.md:101`、`experiments/T01-contract-spec.md:25`、`:29` |
| `.tmp_v420_smoke_eval.sh` | `PREFLIGHT.md:57` |
| `.tmp_v420_repro.sh` | `PREFLIGHT.md:80` |
| `.tmp_v420_replay_all.sh` | `PREFLIGHT.md:190`、`PREFLIGHT_REPLAY.md:6` |
| `.tmp_v420_replay_probe.py` | 正文未点名，由 `.tmp_v420_replay_all.sh` 调用（一跳闭包） |

这与任务给的前序分类（4 probe 溯源留存 / 3 shell 可丢 / 1 个已逐字节归档）**不冲突**：它们的性质不是「聚合/裁决脚本」，且报告正文已内联其命令或结论。我复核的是"谁被引用"，不是重新审定它们的处置（§3.2、§3.3）。归档副本一致性我实测过：`.tmp_v420_replay_probe.py` 与 `evidence/PREFLIGHT/replay/replay_probe.py` 的 sha256 均为 `44a218927794000900ad9b16c9c67408648684d7ba236705493b5fd3c88286af`，**逐字节相同**。

## 2 安全审查（逐脚本全文通读后）

| 审查维度 | `.tmp_v420_expected.py`（31 行） | `.tmp_v420_action_mix.py`（45 行） |
|---|---|---|
| 写文件 / 删文件 | **无**：只用 `read_text()`，无 `open(...,'w')`、`write_text`、`json.dump`、`os.remove`、`shutil` | **无**：同上 |
| 子进程 | **无**：无 `subprocess` / `os.system` / `popen` | **无** |
| 网络 | **无**：无 `socket` / `requests` / `urllib` / HTTP 客户端 | **无** |
| 个人路径 / 密钥 / 家目录 | **无**：无 `D:\Users\...`、无 `/root/...`、无 `Path.home()` / `expanduser` / 环境变量读取 | **无**：同上 |
| 实际读的路径 | 两条**相对于仓库根**的数据路径：`.tmp_v410_wp04_eval_arms/B0_seed0_full.json`、`.tmp_v410_wp05_train_beta05_20k/eval_dev20.json` | 一条相对路径：`.tmp_v420_data/T04/eval` 下的 `<tag>-<ruler>-s*.json` |
| 入口 / 参数解析 | **原版没有**：无 `if __name__ == "__main__"`，无 argparse，import 即执行；路径写死在模块级元组里 | **原版没有**：无入口、无 argparse，`BASE` 是模块级全局 |
| 引擎 / GPU / torch | 不 import（实证：Windows CPU python 直接跑通） | 不 import（同上） |
| 覆盖旧证据的风险 | **无**（脚本不写任何文件） | **无** |

危险 token 扫描（两个归位件）：`write_text|write_bytes|json.dump|subprocess|socket|requests|urllib|os.system|shutil|expanduser|Path.home|/root/|Users\|\D:\` → **0 命中**；`open(` → **0 命中**（全部走 `Path.read_text`）。

**结论：两个脚本都只读、都可安全纳入源码树。**任务给的"例外条款"（若某脚本会写覆盖旧证据则不归位、改在文档里写判定规则）**本次未触发** —— 两者的产物是 stdout 表格，不是落盘证据。

### 2.1 关于"个人路径硬编码"的实测结论

任务预设可能存在 `D:\Users\...` / `/root/...` 这类个人路径。**实测两个文件都没有**；唯一的硬编码是"相对于仓库根的项目数据目录"，与 v4.1.0/v4.2.0 证据里记录的路径一致。因此"必须由命令行/环境变量传入，缺参数报错"的**强制项为空集**。

归位后的处理是：把这三条路径**全部暴露成参数**（`--b0`、`--twenty-k`、`--base`），**默认值保留原来的硬编码相对路径**（否则 `PREFLIGHT.md`/`ACTION_MIX.md` 里"一条命令复现"的能力会退化，且这两个默认值不含个人目录），同时把"输入不存在"从**静默继续**改成**退出码 2**（§4 第 5 条、§5.5）。

## 3 归位决定

### 3.1 归位（2 个，均在白名单内）

| 目标路径（新建） | 来源（未动） | 来源 sha256 | 归位后 sha256 | 行数 |
|---|---|---|---|---|
| `tools/rl/v420_expected.py` | `.tmp_v420_expected.py` | `6cea35c04e7ed7717009157d71dffe0af9bc2ec0a327b950888c9d8ab6718e4b` | `f8b2cc0647ace4ad0d0c03bdc1fb956a8607d7b88b37dc40845e1d30e4d1e2aa` | 99 |
| `tools/rl/v420_action_mix.py` | `.tmp_v420_action_mix.py` | `f3b2f0359cbf51b40fdd14146e9209b5921bb8fe3e0c678e83c468537a6b7382` | `2ca24e44c8741f74b9dfad9ab1ce29b6e035459dc98419da9c0498016eef3005` | 102 |

两个文件头都带 provenance 注释块，含五项：原始路径、原始 sha256、归位日期 2026-11-15、归位原因（`V420_REVIEW.md` P1-07）、**"原始文件未删除，仍留在仓库根"**。

命名遵循 `tools/rl/` 既有约定（`v420.py`、`v420_eval_shoot.py`）：版本前缀 + 下划线功能名，不用 `.tmp_`。

### 3.2 不归位（58 个）及理由

| 类别 | 数量 | 处置 | 理由 |
|---|---:|---|---|
| 已被跟踪代码覆盖的 probe（slot / mask / release / hist） | 4 | 只溯源 | 报告引用它们的**结论**（`T01-contract-spec.md` 的表），而对应逻辑在 `touhou_rl/v420/**` 的已跟踪代码里；归位它们等于复制一份与源码重复的实现。前序已在 `source_identity.json` 登记 |
| 一次性 shell 编排（`smoke_eval.sh` / `repro.sh` / `replay_all.sh`） | 3 | 只溯源 | 正文已把命令内联（`PREFLIGHT.md:57/80/190`），shell 本身没有额外逻辑。`.sh` 硬编码 WSL/`xvfb-run`，Windows 端不可执行，归位进源码树只会制造"看起来能跑"的假象 |
| 已逐字节归档（`replay_probe.py`） | 1 | 无需归位 | `evidence/PREFLIGHT/replay/replay_probe.py` 与原件 sha256 完全相同（本次复核 `44a21892…`） |
| v4.2.0 文档未点名 | 50 | 丢弃（原件保留在原处） | 我的 grep 已确认 53 个 md 的点名集合 = 6 个 `.py` + 3 个 `.sh`（+1 个仅被 shell 调用）；其余 50 个无引用 |

**声明：我没有重做这 58 个文件的逐文件内容溯源。**我复核的是"被谁引用"这一面；类别划分沿用前序 `source_identity.json` 的登记，并已用引用计数交叉验证（§1.3）。这属于 W00 的边界，不属于本次归位职责。

### 3.3 剩余缺口（明确移交，不在本档白名单内）

v4.2.0 的 `.md` 里还点名了 **168 个**位于 `.tmp_v420_<sub>/` 下的脚本路径，分布在 **30 个子目录**（`adjudicate`、`t04`、`t05d2`、`setloss`、`setsup`、`clearcheck`、`remeasure`、`g420re`、`idle`、`r3/r4`、`a1/a1full`、`t03*`、`teacher2f` …）。实测：

- 168 / 168 **仍在磁盘上**（`Test-Path` 全 True）
- 168 / 168 **被 `.gitignore:32` 忽略**（`.tmp_*/` 匹配目录 → `git check-ignore` exit 0）
- 典型例子：`.tmp_v420_adjudicate/adjudicate_corpus.py`（`ROUTE_CONCLUSION.md:577` 称其为"预注册裁决器"，5/5 用例通过）、`.tmp_v420_t04/eval_dev.py`（`DEV_EVAL.md:31` 的运行脚本）

**W00 归位的这 2 个根级脚本不解决这部分。**如果 P1-07 的验收口径是"必要源码、配置、评测器和**裁判规则**都要能随 Git 同步"，那真正的缺口主体是这 168 个（尤其 `adjudicate_corpus.py` / `adjudicate_model.py` 这类判定规则），需要另开工作包按类别搬进 `tools/rl/` 或 `touhou_rl/v420/`。本档只如实登记，不越界处理。

## 4 逐处改动与理由（不做"顺手优化"）

### 4.1 `tools/rl/v420_expected.py`

| # | 改动 | 理由 | 是否改变可观察行为 |
|---:|---|---|---|
| 1 | 新增模块 docstring（原版没有） | 说明只读语义与用途；`--help` 的 description 保持独立短句 | 否 |
| 2 | 文件头 provenance 注释块 | 任务要求（原路径 / 原 sha256 / 2026-11-15 / P1-07 / 原件未删） | 否 |
| 3 | `import argparse`；`parse_args()`；`main(argv=None)`；`if __name__ == "__main__": raise SystemExit(main())` | 原版无入口、import 即执行且路径写死，无法在不改源文件的前提下重定向输入 | 否（新增能力，不改变既有行） |
| 4 | 主体搬进 `extract(name, path)`；`want = {1000,1001,1002}` 提为模块常量 `EXPECTED_SEEDS` | 只是为了让路径成为参数；**打印语句逐字保留**（含字段顺序、`[:16]` 截断、`episodes:` / `summary:` 行） | 否（§5.4 实证：真数据 stdout 与原件逐字符相同） |
| 5 | 缺输入时 `return 2`（stdout 仍打印同一行 `<name> MISSING <path>`） | 原版缺输入**仍 exit 0**，会让"归位可用"的检查假通过（§5.5 实测原件 exit 0 / 归位件 exit 2） | **是**（仅退出码；stdout 不变，已登记） |
| 6 | 默认值 = 原硬编码相对路径 | 保持报告"一条命令复现"；两个默认路径都不是个人路径 | 否 |

**刻意没做的事**：没有改 seed 集合、没有加"输出 JSON"选项、没有加 `--seeds`、没有改字段、没有改 `json.loads` 的编码、没有加日志。

### 4.2 `tools/rl/v420_action_mix.py`

| # | 改动 | 理由 | 是否改变可观察行为 |
|---:|---|---|---|
| 1 | 保留原 1 行 docstring 原文，其后加 provenance 注释块 | 原文照旧，满足 provenance 要求 | 否 |
| 2 | `BASE` 全局 → `DEFAULT_BASE` 常量 + `--base` 参数；`collect(tag, ruler)` → `collect(base, tag, ruler)` | 原版目录写死为模块全局，无法重定向 | 否（同一默认值） |
| 3 | `main(argv=None)` + `if __name__ == "__main__"`；`--base` 不是目录 → `parser.error(...)` = exit 2 | 原版无入口；缺目录时原版会打印 8 行 `no data` 并 exit 0 | **是**（仅退出码；stdout 不变） |
| 4 | 空目录（0 个直方图被聚合）→ `return 2` | 与 #3 同一 fail-closed 理由 | **是**（仅退出码） |
| 5 | `LETTERS`、`sorted(BASE.glob(...))`、`"driver"/"partial"` 过滤、`total=[sum(col) ...]`、`n=sum(total)`、`shares`、`sorted(range(18), key=-shares)[:4]`、`f"{:.4f}"` 格式 | **一字未改** | 否 |

**刻意没做的事**：没有改 18 个动作的顺序、没有改 top4 的排序稳定性处理、没有把 `focus-idle` 换成别的索引、没有加 `--tags`、没有动"跳过 driver/partial"的规则。

### 4.3 本档新增/改动的全部文件（4 个，均在白名单内）

| 文件 | 类型 | sha256 |
|---|---|---|
| `tools/rl/v420_expected.py` | 新建 | `f8b2cc0647ace4ad0d0c03bdc1fb956a8607d7b88b37dc40845e1d30e4d1e2aa` |
| `tools/rl/v420_action_mix.py` | 新建 | `2ca24e44c8741f74b9dfad9ab1ce29b6e035459dc98419da9c0498016eef3005` |
| `Document/Update/Alpha/v4.3.0/evidence/W00/W00_SCRIPT_RESTORE.md` | 新建 | 本文件 |
| `Document/Update/Alpha/v4.3.0/evidence/W00/EXIT_CODES_RESTORE.txt` | 新建 | 见该文件 |

未触碰 `tools/rl/v430.py`、`configs/v430/**`、`touhou_rl/**`、`tests/**`、W00 目录下其它文件、任何 `.tmp_v420_*`、`ASSET_SYNC.*`、旧证据/state/journal/budget。

## 5 验证（命令与真实退出码，全部实跑）

原始退出码逐条登记在 [`EXIT_CODES_RESTORE.txt`](EXIT_CODES_RESTORE.txt)。合成输入与驱动器放在仓库外：`%TEMP%\w00_restore_synth_20261115\`（含 `run_verify.ps1` 与完整 `transcript.txt`）。仓库内的唯一副产物是 `compileall` 必然写出的 `tools/rl/__pycache__/v420_expected.cpython-312.pyc` 与 `v420_action_mix.cpython-312.pyc`（`.gitignore:2` 的 `__pycache__/` 已覆盖；该目录运行前就有 310 个同类 `.pyc`），除此之外**没有**在仓库内产生额外文件。

### 5.1 静态检查

```text
python -m compileall -q tools/rl/v420_expected.py tools/rl/v420_action_mix.py   => exit 0
危险 token 扫描（写盘/子进程/网络/个人路径，两文件）                              => 0 匹配
"open(" 扫描（两文件）                                                           => 0 匹配
```

### 5.2 `--help`（两个都补了 argparse）

```text
$ python tools/rl/v420_expected.py --help       => exit 0
usage: v420_expected.py [-h] [--b0 B0] [--twenty-k TWENTY_K]
Extract the v4.2.0 T02 expectation table (read-only).
  --b0 B0              B0 20-dev eval JSON (default: .tmp_v410_wp04_eval_arms/B0_seed0_full.json)
  --twenty-k TWENTY_K  20k 20-dev eval JSON (default: .tmp_v410_wp05_train_beta05_20k/eval_dev20.json)

$ python tools/rl/v420_action_mix.py --help     => exit 0
usage: v420_action_mix.py [-h] [--base BASE]
Read-only per-checkpoint action-mix comparison (T04 dev eval).
  --base BASE  directory holding <tag>-<ruler>-s*.json (default: .tmp_v420_data/T04/eval)
```

（备注：首轮用 `--help | Select-Object -First 4` 取前几行时 PowerShell 报 `-1`，那是**管道被提前关闭**的测量 artifact，不是脚本退出码；不截断重跑后为 0。已记入 `EXIT_CODES_RESTORE.txt` §9。）

### 5.3 合成输入等价性（证明"归位件行为 = 原件行为"）

合成数据（**注意：这些数字是构造的，不是实验证据**）：

- `expected`：两份 JSON，各含 seed 1000/1001/1002 三行 + 一行 seed 999（**必须被过滤**，其 `stop_reason` 故意写成 `SYNTH-MUST-BE-SKIPPED`），以及 `arm_summary`；两文件 `checkpoint` 字段写 `SYNTHETIC-*-not-real-evidence`
- `action_mix`：`B0/20k/A0/A0v` × `diag/s1n` × 2 个 `-s*.json`，每个 18 bin 直方图；另放 `B0-diag-s9999-driver.json`、`B0-diag-s9999-partial.json` 两个**投毒文件**（直方图取值 100001+，若未被过滤会立刻抬高 `decisions`）

| 用例 | 命令 | 退出码 | 结果 |
|---|---|---:|---|
| 原件 expected | `python <root>\.tmp_v420_expected.py`（cwd=合成目录） | 0 | — |
| 归位件 expected | `python tools/rl/v420_expected.py --b0 <synth> --twenty-k <synth>` | 0 | stdout 与原件**逐字符相同**（case-sensitive compare = True）；seed 999 未出现 |
| 原件 action_mix | `python <root>\.tmp_v420_action_mix.py`（cwd=合成目录） | 0 | — |
| 归位件 action_mix | `python tools/rl/v420_action_mix.py --base <synth eval dir>` | 0 | stdout 与原件**逐字符相同** = True |
| 8 格 decisions 与独立真值比对 | 驱动器用 PowerShell 另算 `sum(hist)` | — | **8/8 match=True**（4842 / 4950 / 5094 / 5202 / 5346 / 5454 / 5598 / 5706） |
| 投毒过滤 | 磁盘上存在 2 个 driver/partial 文件 | — | 两个都未进入结果（否则 decisions 会多出 1,800,171） |

### 5.4 真实数据只读复现（最强证据：归位件重算出 v4.2.0 报告里的表）

命令（cwd = 仓库根，不传参数，用脚本自身默认路径；**只读**，不启引擎、不写任何文件）：

```text
python .tmp_v420_expected.py            => exit 0
python tools/rl/v420_expected.py        => exit 0    stdout 与原件逐字符相同 = True
python .tmp_v420_action_mix.py          => exit 0
python tools/rl/v420_action_mix.py      => exit 0    stdout 与原件逐字符相同 = True
```

**A. `PREFLIGHT.md` §4 期望值表（6 行逐格）**

| checkpoint | seed | 文档 frames/steps/score | 实跑 frames/steps/score | 文档 act_sha 前缀 | 实跑 act_sha 前缀 | 一致 |
|---|---:|---|---|---|---|---|
| B0 | 1000 | 1398 / 699 / 13102 | 1398 / 699 / 13102 | `85a528fd…` | `85a528fd29c83f51` | ✅ |
| B0 | 1001 | 818 / 409 / 1512 | 818 / 409 / 1512 | `57f61f99…` | `57f61f9958f4b4e5` | ✅ |
| B0 | 1002 | 1251 / 626 / 7438 | 1251 / 626 / 7438 | `864101ae…` | `864101aed6fe0d13` | ✅ |
| 20k | 1000 | 1612 / 806 / 12471 | 1612 / 806 / 12471 | `c2ee8fc1…` | `c2ee8fc1335dbfa7` | ✅ |
| 20k | 1001 | 1970 / 985 / 18920 | 1970 / 985 / 18920 | `d2cd9917…` | `d2cd9917d99b151e` | ✅ |
| 20k | 1002 | 1532 / 766 / 12124 | 1532 / 766 / 12124 | `e86f54d5…` | `e86f54d5964fb63c` | ✅ |
| 20 dev 汇总 | — | B0 `frames_mean=1453.25`、20k `1714.65`、两者 `clears=0` | B0 `1453.25`、20k `1714.65`、`clears 0` | — | — | ✅ |

**B. `ACTION_MIX.md` §1 动作分布表（8 格逐格）**

| 口径 | tag | 文档 局数/决策数/idle/unique/前3 | 实跑 局数/决策数/idle/unique/前3 | 一致 |
|---|---|---|---|---|
| diag | B0 | 20 / 14,537 / 0.4581 / 16 / idle,E,W | 20 / 14537 / 0.4581 / 16 / idle,E,W | ✅ |
| diag | 20k | 20 / 17,151 / 0.6087 / 17 / idle,W,E | 20 / 17151 / 0.6087 / 17 / idle,W,E | ✅ |
| diag | A0 | 20 / 16,374 / 0.2989 / 13 / idle,W,SW | 20 / 16374 / 0.2989 / 13 / idle,W,SW | ✅ |
| diag | A0v | 20 / 16,193 / 0.5055 / 13 / idle,E,W | 20 / 16193 / 0.5055 / 13 / idle,E,W | ✅ |
| s1n | B0 | 20 / 28,816 / 0.4610 / 17 / idle,W,E | 20 / 28816 / 0.4610 / 17 / idle,W,E | ✅ |
| s1n | 20k | 20 / 31,241 / 0.5061 / 17 / idle,W,E | 20 / 31241 / 0.5061 / 17 / idle,W,E | ✅ |
| s1n | A0 | 20 / 30,118 / 0.3303 / 16 / idle,W,E | 20 / 30118 / 0.3303 / 16 / idle,W,E | ✅ |
| s1n | A0v | 20 / 27,923 / 0.4952 / 16 / idle,E,W | 20 / 27923 / 0.4952 / 16 / idle,E,W | ✅ |

`focus-idle` 8 格全为 `0.0000`，与文档一致。

> 结论：归位后的两个脚本在真数据上**逐格重算出 v4.2.0 报告的两张表**，且 stdout 与原件逐字符相同 —— 这是"归位没有改变语义"的直接证据，而不是"看起来能跑"。

### 5.5 反向用例（fail-closed，两处行为变更的实证）

| 场景 | 命令 | 退出码 | 说明 |
|---|---|---:|---|
| 原件 + 缺输入 | `python .tmp_v420_expected.py`（cwd 里只有 20k 文件） | **0** | 旧行为：打印 `B0 MISSING …` 后仍然"成功" |
| 归位件 + 缺输入 | `python tools/rl/v420_expected.py --b0 <absent> --twenty-k <ok>` | **2** | 新行为：仍打印同一行 MISSING，但退出码 2 |
| 归位件 + 目录不存在 | `python tools/rl/v420_action_mix.py --base <absent>` | **2** | argparse：`error: --base is not a directory: …` |
| 归位件 + 目录存在但无数据 | `python tools/rl/v420_action_mix.py --base <empty>` | **2** | 8 行 `no data` 之后 fail-closed |

### 5.6 原件完整性（未删、未改、未移）

| 文件 | 运行前 sha256 | 全部验证跑完后 sha256 | 一致 |
|---|---|---|---|
| `.tmp_v420_expected.py` | `6cea35c0…6718e4b` | `6cea35c0…6718e4b` | ✅ |
| `.tmp_v420_action_mix.py` | `f3b2f035…6b7382` | `f3b2f035…6b7382` | ✅ |

`git status --porcelain` 里两条原件仍是 `??`（与归位前相同的 untracked 状态，未被 add、未被改）。

### 5.7 对现有测试的影响

我核对了唯一会对 `tools/rl` 做 `git status` 断言的测试 `tests/test_v420_eval_shoot.py::test_v410_evaluator_git_status_is_unchanged`（第 226–253 行）：它只 pin `tools/rl/v410_student_eval.py`（`??` 或干净）与 `tools/rl/v400_student_eval.py`（必须干净），**不枚举** `tools/rl` 目录，因此新增这两个文件不会改变它的判定。本档未运行该测试文件（见 §6）。

## 6 not_run（明确未做，且不冒充做过）

| 项 | 状态 | 原因 |
|---|---|---|
| 用 v4.2.0 记录用的 WSL 解释器 `/root/v400-venv/bin/python` 重跑 | **not_run** | W00 不允许启 WSL 引擎/资产腿；两个脚本无平台相关依赖，Windows `python 3.12.10` 已跑通并逐格复现报告表（§5.4）。**WSL 侧未验证**，如实登记 |
| v4.2.0 依赖引擎的实验腿（T02 复现 3 seed×2、T03、T05 DAgger、T06 RL、T08 评测） | **not_run** | 不在 W00 范围；需要 WSL 引擎 + `/root/v400-data` + `.tmp_v420_data` 资产，且会写运行目录 |
| 另外 58 个 `.tmp_v420_*` 的逐文件内容溯源 | **not_run** | 超出 W00 白名单/职责；本档只复核引用面（§1.3） |
| 168 个 `.tmp_v420_*/` 子目录脚本的归位 | **not_run** | 不在白名单内；已作为缺口移交（§3.3） |
| 全仓 pytest | **not_run** | W00 由其它代理并行推进；我新增的两个文件不被现有测试收集，且已单独核对唯一相关的 git-status 断言（§5.7） |
| 对 `.tmp_v420_*` 做任何清理/忽略规则调整 | **not_run（禁止）** | 原件是历史运行记录；`.gitignore` 不在白名单 |

## 7 剩余阻塞与移交

- **本次归位无阻塞**：两个脚本都归位、都可运行、都有真实退出码与逐格复现证据。
- **移交 1（口径）**：P1-07 的措辞应更新为"这 60 个根级临时脚本**既不在 Git、也不在 `ASSET_SYNC.sha256`、而且不被 `.gitignore` 忽略**"。若后续要止血，建议新增一条根级忽略规则（例如 `.tmp_v*.*`）——本档不改 `.gitignore`。
- **移交 2（缺口主体）**：168 个被 v4.2.0 文档点名的 `.tmp_v420_*/` 脚本（含"预注册裁决器" `.tmp_v420_adjudicate/adjudicate_corpus.py`）仍全部位于被忽略的子目录。W00 的这两次归位只覆盖根级 2 个文件，**不足以宣称 P1-07 关闭**。
- **移交 3（说法更正）**：`W00_SOURCE_IDENTITY.md:199` 关于"`.tmp_v420_*` 在 Windows 跑不起来"的表述应限定为 42 个 `.sh`；至少这两个 `.py` 是 Windows 可跑的（§1.3 差异 2）。
