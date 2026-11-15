# W00 发现：`review_checks.json` 的 101 条源码 SHA 基线受换行符影响

日期：2026-11-15，Asia/Shanghai。执行者：主代理（AGENTS.md §1 例外：复核/汇报）。
只读核对，未修改任何被核对文件，未修改 `review_checks.json`，未改 `ASSET_SYNC.*`。

## 1 现象

`v4.3.0/evidence/verify_plan.py` 在本机 exit 1。它先在相对链接检查上 fail-fast：

```text
ValueError: local link V420_REVIEW.md: ../v3.6.0/总索引.md
```

把链接检查绕过单独复核它后面的**历史源码 SHA 一致性**检查（即 `review_checks.json` 的 `source_sha256`，101 条），
按 `verify_plan.py` 相同口径逐文件重算，结果是 **96 条不匹配、5 条匹配**。

## 2 根因：换行符状态，不是内容变更

对 101 条逐条做三种口径的 sha256（`raw` / `CRLF→LF` / `LF→CRLF`），分类结果：

| 类别 | 条数 | 说明 |
| --- | ---: | --- |
| 按当前字节直接匹配 | 5 | `T04/verified_set_idle.json`、`v4.2.0/README.md`、`state.json`、`journal.jsonl`、`budget.jsonl` |
| 仅「把该文件规范成 CRLF 后」匹配 | **96** | 这些文件当前是 LF，基线记录时是 CRLF |
| 两种口径都不匹配（真正无法解释） | **0** | —— |

**96 条全部可以用「基线按 CRLF 记录、当前工作区是 LF」解释，没有一条剩下。**
即：这 96 个文件的内容与 v4.3.0 计划复核时**逐字节相同（模换行符）**，不存在内容篡改或来源漂移。

复算方式（只读）：

```python
raw = path.read_bytes()
hashlib.sha256(raw).hexdigest()                                              # 当前口径
hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()  # CRLF 口径
```

本检出确实存在**混合换行状态**：`v4.2.0/journal.jsonl` 是 CRLF（`b"\r\n" in raw` 为真，190,979 B），
而绝大多数 `.py`/`.md`/`.json` 是 LF。`git` 本身也在提示：

```text
warning: in the working copy of 'touhou_rl/dataset.py', LF will be replaced by CRLF
         the next time Git touches it
```

## 3 这是缺陷，不是无害噪音

后果有三层：

1. **`verify_plan.py` 在本机不可复现**（外加被删历史文档导致的断链）。`plan_checks.json` 里记录的通过状态
   在当前工作区**无法重放**。
2. **任何 SHA 身份校验都会随换行符状态静默翻转**。同一份内容在 LF 机器上判 mismatch、在 CRLF 机器上判 verified。
   这不是"证据不足"，而是**校验器本身缺少规范化约定**。
3. 它会污染后续所有工作包：W00 的资产登记、W01 的 `contracts.json` 冻结、W03 的 manifest `code_snapshot`、
   W12 的结算，全部以 sha256 为身份。若不同机器换行符状态不同，跨机复现会得到互相矛盾的"通过"。

## 4 建议处置（未执行）

按优先级，任选其一，**但必须显式选定并记录在规格里**：

1. **在规格层固定规范化规则**（推荐）：所有身份 sha256 一律对「`CRLF→LF` 规范化后的字节」计算，
   并在 `PLAN_SPEC` 与各 manifest 的 `schema` 说明里写死。这样与 `core.autocrlf` 取值无关。
2. **固定仓库换行策略**：加 `.gitattributes`（`* text=auto eol=lf` 或 `eol=crlf`），让所有检出字节一致，
   然后**重算并重钉** `review_checks.json` 的 `source_sha256`（属新基线，需记为新 experiment_id，不能覆盖旧记录）。
3. 维持现状，但把 `verify_plan.py` 的 SHA 检查改为同时接受两种口径并报告实际口径——**最弱**，
   因为它仍然无法区分"纯换行差异"与"内容被改且恰好换行也变了"。

**主代理未执行任何一项**：方案 2 涉及写入 `.gitattributes` 与重钉证据（越出 W00 白名单，且 `AGENTS.md` §8.2
禁止在无授权时改清单）；方案 1 需要改 `PLAN_SPEC`（规格修订，须递增 `spec_revision`）。
两者都应由用户确认方向后，作为独立工作条目派发。

## 5 与另一条账本发现的区分（都很重要，但不是同一件事）

| 发现 | 对象 | 结论 |
| --- | --- | --- |
| 本节 | `v4.3.0/evidence/review_checks.json` 的 101 条源码 SHA | 96 条为**换行符口径差异**，内容未变，0 条无法解释 |
| `ASSET_LEDGER_CHECK.md` §1 | `ASSET_SYNC.sha256` 的 46 条资产 | **1 条真实过期**：`v4.2.0/journal.jsonl` 注册值 `5ed3693f…`，当前字节 `835b9e18…`（CRLF 规范化后 `25a5fd1c…`，也不等于注册值）。该文件自 `05c4613` 提交后未改，属登记后追加 |

两条互不否定：`review_checks.json` 对 `journal.jsonl` 的期望值与**当前字节**一致（它本身就在那 5 条"直接匹配"里），
而 `ASSET_SYNC.sha256` 对该文件的期望值与当前字节不一致。两个基线是在不同时刻、不同追加位置记录的。

## 6 对 v430 新代码的约束（主代理已决定，不涉及改历史证据）

§4 的三个方案都要动历史证据或规格，属于需要用户确认的方向。但**新代码不必等这个决定**：
v430 自己的身份 digest 从现在起必须自带规范化，不能把这个缺陷复制到下一代。

据此对后续工作包（W01b/W03/W12）下硬约束：

1. **文本类产物**（配置、manifest、JSON 证据、脚本源码）计算 sha256 时，必须先做
   `CRLF -> LF` 规范化（`bytes.replace(b"\r\n", b"\n")`），再对规范化后的字节哈希。
2. **二进制类产物**（checkpoint、`.dat`、引擎二进制、PNG）**禁止**做换行规范化，直接对原始字节哈希。
3. 每个 dict 里保存的身份字段必须显式带上计算口径，例如
   `{"sha256": "...", "digest_scope": "text/lf-normalized"}` 或 `"binary/raw"`。
   不允许出现"只有 sha256、没有口径"的身份字段。
4. 校验函数必须在**不一致时失败**，且失败消息要能区分「内容不同」与「口径不同」——否则等于把
   本次发现的问题重新藏起来。

这条约束只适用于 v430 新产物，**不改** `review_checks.json`、`PLAN_SPEC.json`、`ASSET_SYNC.*`
中的任何既有数值，因此不需要用户批准，也不会让历史记录失真。

## 7 未做 / 不声称

- 未改 `review_checks.json`、未改 `PLAN_SPEC`、未加 `.gitattributes`、未重钉 `ASSET_SYNC.*`。
- 未运行任何游戏/训练；本节不改变任何 G430/G410 门状态。
- 本节证明的是「96 条差异可用换行符解释」，**不证明**那些历史实验结论本身正确。
