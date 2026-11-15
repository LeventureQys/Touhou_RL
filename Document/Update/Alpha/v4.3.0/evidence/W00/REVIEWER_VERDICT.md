# W00 验收裁定（主代理复核记录）

日期：2026-11-15，Asia/Shanghai。裁定人：主代理。
本文件只做**验收裁定与独立复核记录**，不重复执行代理的证据内容，不修改任何被复核文件。

**裁定结论：W00 的三条腿（资产登记、源码身份、脚本归位）在「已解释 1 项例外」的口径下成立
并已全部验收；但 G430-E0 尚未通过，因为它的 W01 部分（完整动作与正负结果真实通过）还没做。**

**W00 的授权范围内已完成，无未收项。** 唯一遗留的 P1-07 主体（187 个被点名的子目录脚本）
不属 W00 白名单，已按独立工作包登记（见 §4bis）。

## 1 交付与裁定

| 腿 | 交付 | 裁定 |
| --- | --- | --- |
| 资产登记 + audit 入口 | `tools/rl/v430.py`、`configs/v430/assets.json`、`evidence/W00/audit-001.json`、`verify_audit.py/json`、`W00.md`、`EXIT_CODES.txt` | **接受**（附 1 项已解释例外，见 §3） |
| 源码身份 + 可复现 | `evidence/W00/verify_source_identity.py`、`source_identity.json`、`W00_SOURCE_IDENTITY.md`、`EXIT_CODES_SOURCE.txt`、`registries/code_identity.json` | **接受** |
| 引用脚本归位 | `tools/rl/v420_expected.py`、`tools/rl/v420_action_mix.py`、`tools/rl/v420_{slot,hist,mask,release}_probe.py`、`evidence/W00/W00_SCRIPT_RESTORE.md`、`W00_PROBE_RESTORE.md` | **接受**（共 6 个文件；范围仍窄于 P1-07 全量，见 §4bis） |
| 主代理独立复核 | `evidence/W00/RUNTIME_PROBE.md`、`ASSET_LEDGER_CHECK.md`、`SHA_LINE_ENDING_FINDING.md`、本文件 | 已落盘 |

### 1.1 脚本归位的独立复验（主代理实跑）

`tools/rl/v420_slot_probe.py`（默认参数、真实 shard）→ **exit 0**，且逐项复现 v4.2.0 已发布值：

```text
duration histogram: {2: 588, 1: 2712}
slots[0]==first_action: {True: 3300}
slots[1]==slots[0] (when duration>=2): {True: 588}
hash chain breaks: 0 []
decision_frame stride histogram: {4: 3299}
act_hist sha256: 76b3ff89a668a6d61ad6c3808be650a90592edbcdd15da9cfffe6d3f3f9aaa6d
shard.json: decision_count 3300 frames_committed 13200 deaths_observed 0 stage_clear False
hash_alignment: checkpoints_equal 6600 checkpoints_mismatched 0
```

另：4 个归位脚本 `compileall` 全 0、各自 `--help` 全 0、4 个 `.tmp_*` 原件的 sha256 与 provenance
声明逐一相符（原件未被删改）。

## 2 主代理独立复核（不是转述代理结论）

### 2.1 CLI 与 exit code 语义（我自己跑的）

| 检查 | 结果 |
| --- | --- |
| `python tools/rl/v430.py --help` | exit **0**，17 个子命令全部注册 |
| 16 个保留子命令（如 `validate-contract`、`train`） | exit **2** `not_implemented`，**且未创建输出文件**（`Test-Path` 为 False） |
| 对已存在 `--out` 重跑 `audit` | exit **2**，"refusing to overwrite existing output"，文件 sha256 前后不变 |
| `audit` 实跑 | exit **3**；53 项检查、WSL 可达、`missing=0`、`unreachable=0`，唯一不达标项 = `sync:...v4.2.0/journal.jsonl` |
| 结果 JSON 的 `capability_gate` | `{"status":"not_run","gates":{"G430-E0":"not_run","G410-0":"not_reassessed"}}` |

**关键红线守住**：资产审计的 exit 0 **不**自动等于能力通过，`capability_gate` 独立且默认 `not_run`。

### 2.2 合成自检的真实覆盖（我读了 `verify_audit.json`）

12 个用例 **12 通过 / 0 失败**，覆盖：`all_pass→0`、`missing_file→3`、`sha_mismatch→3`、
`plan_spec_drift→3`、`manifest_coverage_gap→3`、`oversize_blocked→3`、`oversize_stat_only→0`、
`wsl_unreachable→3`、`bad_schema→2`、`bad_frozen_at→2`、`out_exists→2`、`reserved_subcommand→2`。
该文件自带 `evidence_class=implementation_self_check`、`not_capability_evidence: true`，
并明写"不通过 G430-E0"。**分类诚实。**

### 2.3 源码身份（我自己重跑）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python Document/Update/Alpha/v4.3.0/evidence/W00/verify_source_identity.py   # exit 0
```

`reference_crosscheck` 计数：**verified 16 / mismatch 2 / not_referenced 11 / unverified_host_split 1 / missing 0**。
54 个文件 / 2,290,051 B / 清单摘要 `d5f2c4a3…56f509`；v420 测试 **468 passed / 3 skipped / 0 failed**；
两个 checkpoint 在独立进程里 **strict 绑定成功**（36 键、731,802 参数）。

`mismatch` 的 2 条都是 `ASSET_SYNC.sha256` 与 `ASSET_SYNC.md` 自身（2026-11-15 重钉后与 T00 记录不同），
**不是 v420 源码**。`engine420` 被判 `unverified_host_split` 而非 `missing`，且 `citation_holds=true`——
它每次运行会**回读**我的 `ASSET_LEDGER_CHECK.md` 与 `T01/audit-assets.json`，要求命令与 `691bfd77…` 仍在，
失效则自动退回 `missing` 并报错。**引用是被检查的，不是被转述的。**

### 2.4 脚本归位的 provenance（我逐字节核对）

| 文件 | 原路径 | 声明的原 sha256 | 实际 sha256 | 原件是否仍在 |
| --- | --- | --- | --- | --- |
| `tools/rl/v420_expected.py` | `.tmp_v420_expected.py` | `6cea35c0…8e4b` | 同 | 是 |
| `tools/rl/v420_action_mix.py` | `.tmp_v420_action_mix.py` | `f3b2f035…7382` | 同 | 是 |

两个 `.tmp_*` 原件都**未被删除/移动/重命名**，provenance 注释里写明了"只改了 4 处"。

### 2.5 引用面交叉核对（我自己扫的）

对 `Document/Update/Alpha/v4.2.0/` 下 149 个 `.md/.json/.txt/.jsonl` 扫 `.tmp_v420_*.py|.sh`，
得到 **9 个被引用脚本**：2 个 py（`expected`、`action_mix`）+ 4 个 probe（`slot`/`hist`/`mask`/`release`）
+ 3 个 shell（`replay_all`/`smoke_eval`/`repro`）。与执行代理的分类**一致**（它把这 9 个分成
"2 归位 / 4 溯源留存 / 3 一次性可丢"）。

## 3 唯一例外：`v4.2.0/journal.jsonl`（已解释，不重钉）

三条**互相独立**的证据指向同一结论——该文件在清单生成**之后**被追加，不是内容损坏：

1. **哈希**：pinned `5ed3693f…`，当前 `835b9e18…`；raw / 全 LF / 全 CRLF 三种口径**都不匹配**（`matches_under: none`）。
2. **mtime 顺序**：`ASSET_SYNC.sha256` 为 `08:24:47`，`journal.jsonl` 为 `08:26:39` —— 先有清单，后写日志。
3. **git blob**：`HEAD` 内 blob 与工作区字节相同（`835b9e18…`），说明该文件自 `05c4613` 后未再改。

按 `DEVELOPMENT_GUIDE.md` §12「不把 state/journal 的活文件漂移当静态资产损坏，也不任意重钉真正不匹配的静态文件」，
本次**不重钉**。重钉属 `ASSET_SYNC.md` / `ASSET_SYNC.sha256` 的改动，按 `EXECUTION_PLAN.md` §2 属
W12 的清单任务白名单，不属 W00。

**因此 `audit` 恒返回 3 是正确行为，不是缺陷。** W00 的验收口径按原文「SHA 与报告一致**或逐项解释差异**」，
本项属于"已逐项解释"，故判定 **W00 资产腿接受**；但 W12 结算时必须处理，否则 G430-E0 的资产腿永远停在 unmet。

## 4 我发现的、代理未覆盖的两个环境级问题

1. **3458 份历史文档被删（未提交）**：`git status` 有 3514 变更 / 8,241,913 行删除，全部在
   `Document/Update/Alpha/v1.0.0…v3.7.x` 与 `Document/Dialogs`。源码不受影响。直接后果是 v4.3.0
   自带的 `verify_plan.py`、`verify_review.py` 在本机 **exit 1**（断在 `V420_REVIEW.md → ../v3.6.0/总索引.md`），
   `plan_checks.json` 的通过状态**本机不可复现**。
2. **换行符口径污染**：`review_checks.json` 的 101 条源码 SHA 中 **96 条**只差 CRLF/LF 口径，
   **0 条无法解释**。详见 `SHA_LINE_ENDING_FINDING.md`；已据此对 v430 新代码下硬约束（文本产物的
   sha256 必须先 `CRLF->LF` 规范化并带 `digest_scope`）。

这两条都不是 W00 白名单内能改的，故只登记并给出方案，不擅自执行。

## 4bis P1-07 的真实规模（**更正我自己此前的一个错误判断**）

我在派发 W00 脚本归位任务时写过「`.tmp_v420_*` 都在 `.gitignore` 覆盖下，所以永远无法同步」。
**这句话是错的**，执行代理独立复核后指出，我又亲手验了一遍：

```text
.gitignore:32  ->  .tmp_*/          # 结尾斜杠只匹配目录
git check-ignore -v .tmp_v420_expected.py            -> exit 1（不匹配）
git status --porcelain .tmp_v420_expected.py         -> ?? .tmp_v420_expected.py
git check-ignore -v .tmp_v420_adjudicate/adjudicate_corpus.py -> exit 0（.gitignore:32 命中）
```

即根级 `.tmp_v420_*.py/.sh` 是「**untracked 但未被忽略**」，只有**子目录**里的才会被忽略。

按这个正确前提重新实测，P1-07 的实际规模是：

| 类别 | 数量 | git 状态 |
| --- | ---: | --- |
| 根级 `.tmp_v420_*.py/.sh` | **60** | untracked、**未**被忽略（`git add -A` 会把它们扫进索引） |
| `.tmp_v420_*/` 子目录内脚本总数（`.py`+`.sh`） | **2,389** | 全部被忽略 |
| 其中**被 v4.2.0 文档按路径点名**的脚本 | **187**（我的扫描口径；执行代理更严口径得 168） | **187/187 存在、187/187 被忽略** |
| 这些点名脚本分布的顶层目录 | **33** 个 `.tmp_v420_*` | —— |
| W00 实际归位的 | **2** | 新增于 `tools/rl/` |

点名密度最高的目录：`.tmp_v420_t05d3`(13)、`.tmp_v420_clearcheck`(11)、`.tmp_v420_r2diag`(11)、
`.tmp_v420_t05d2`(10)、`.tmp_v420_setloss`(10)、`.tmp_v420_t05`(10)。

**因此修正裁定：W00 的脚本归位腿「完成了它被授权的 2 个文件」，但 P1-07 远未关闭。**
把 187 个被点名脚本迁到可同步路径（其中含 WSL/xvfb 专属 shell、硬编码 `/root` 与 `/mnt/d` 的编排）
本身就是一个独立工作包，不能算在 W00 里；它与 W00「恢复可复现身份」相邻但不是同一件事。

**并且给出一个操作性警告**：因为 60 个根级临时脚本**未被忽略**，对本仓库执行
`git add -A` / `git add .` 会把这 60 个一次性脚本一并扫进索引。**任何 `git add` 都必须是显式路径**，
不能图省事用 `-A`。这条直接关系到 §5 里那个待用户决定的 `git add`。

## 5 仍未闭合（W00 → 下一包）

| 项 | 状态 | 归属 |
| --- | --- | --- |
| 4 个被引用的 probe 脚本（`slot`/`hist`/`mask`/`release`，共 8.5 KB、只读、无子进程）仍未归位 | 待办 | W00 收尾 |
| 3 个被引用的 shell 脚本未归位（命令已内联在报告，可接受） | 登记 | W00 收尾 |
| **187 个被文档按路径点名的 `.tmp_v420_*/` 脚本（187/187 被忽略）** | **未闭合** | **需独立工作包**，不在 W00 |
| 54 个 v420 源文件仍未 `git add`（P1-07 的核心） | **待用户决定** | 用户 |
| 换行符规范化方案（`.gitattributes` / `PLAN_SPEC` 规则 / 维持现状） | **待用户决定** | 用户 |
| `journal.jsonl` 重钉 | 待办 | W12 清单任务 |
| obs/action 协议 digest | 待办 | W01b |

## 6 G430-E0 判定

`G430-E0` = 「W00 身份与可复现源码；W01 完整动作、正负结果、时间/边界实际通过」。

- W00 部分：**已具备可裁定的证据**（本文件 §2、§3）。
- W01 部分：`contracts.py`/`actions.py` 已落地并经我实测（72 组合互异、shoot=0 不置 SHOOT 位、
  bomb 按下沿只触发一次、owner 独占、特权字段拒绝、`capability_gate` 默认 `not_run`）；
  但 **session / evaluate / `validate-contract` 在飞，真实成功与失败回放（W01c）尚未执行**。

**结论：G430-E0 = `not_run`（W01 腿未完成）。不得判定通过。**
