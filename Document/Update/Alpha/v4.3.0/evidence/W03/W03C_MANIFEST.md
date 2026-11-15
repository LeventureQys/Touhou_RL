# W03C — 数据 manifest 层：加载 / fail-closed 校验 / partial 视图

**状态：完成（本切片范围内）**

**全部测试为合成测试（`tmp_path` 造 fixture），不是游戏/能力证据。**
未启动引擎、未训练、未读任何真实 `.tmp_*` 语料、未读 checkpoint/模型。
所有判定只针对「manifest 记录的字节/字段是否自洽」，不代表这些数据可用于训练或能提升能力。

---

## 1. 范围与交付

规范：`DEVELOPMENT_GUIDE.md` §6.1（Manifest，8 行）；§6.2（Transition 字段表）只用来
校准 shard 计数口径（`decisions` / `labels_valid` 是逐决策量，不是逐帧量）。

| 文件 | 说明 |
| --- | --- |
| `touhou_rl/v430/data/manifest.py` | **新建**。纯 stdlib：`dataclasses`/`os`/`pathlib`/`typing` + 同包 `lineage`、`contracts`。无 numpy / torch / 引擎 / 无 `hashlib` |
| `tests/test_v430_data.py` | **新建**。29 条行为测试，全部 `tmp_path` 合成 fixture |
| `Document/Update/Alpha/v4.3.0/evidence/W03/W03C_MANIFEST.md` | 本文件 |
| `Document/Update/Alpha/v4.3.0/evidence/W03/EXIT_CODES_W03C.txt` | 三条验收命令的真实退出码与统计 |

**只读复用，未修改**：`touhou_rl/v430/data/lineage.py`（摘要/分组/JSON 解析）、
`touhou_rl/v430/contracts.py`（`EXIT_OK`/`EXIT_INVALID`/`EXIT_INCOMPLETE`）。
本切片没有改动任何既有文件（白名单外零写入）。

公开接口：

```text
MANIFEST_SCHEMA = "v430-manifest/1"      REPORT_SCHEMA = "v430-manifest-validation/1"
load_manifest(path, *, roots=None) -> Manifest
validate_manifest(manifest, *, base_dir=None, strict=True, roots=None,
                  expected_obs_digest=None, expected_action_digest=None,
                  expected_label_digest=None, split_config=None,
                  split_registry_sha=None) -> dict
partial_view(manifest, *, completed_only=True, base_dir=None, strict=False) -> PartialView
exit_code_for(report) -> int              # 0 完整 / 4 合法但未完成 / 2 非法
PROBLEM_CODES (31 个稳定 P_* 码)          ADVISORY_CODES = {"P_PARTIAL_INCOMPLETE"}
```

`validate_manifest` 的返回**同时**给出两个布尔量，避免「partial 被当成完整」：

| 键 | 含义 |
| --- | --- |
| `valid` | 所有致命检查通过（`problems == []`） |
| `complete` | `status == "complete"` 且每个 shard 都 `completed` 且 `immutable is True`，且 `valid` |
| `problems` / `degradations` | 致命问题 / 仅 `strict=False` 下的可降级条目 |
| `checks_performed` / `not_checked` | 真正跑过的检查 / **没跑**的检查（缺检查必须自报，不得默认通过） |
| `statistics` | `shards_listed/readable`、按 split 计数、`bytes/decisions/labels_valid/episodes`、`ancestor_ids/ancestor_groups` |
| `manifest` | manifest 文件自身的身份：`sha256` + `digest_scope="text/lf-normalized"` + `size_bytes` |
| `split_registry` | 绑定的 `split_registry_sha`、来源、`split_config_valid` |
| `exit_code` | `exit_code_for(report)` |

## 2. 数据结构

`Manifest` 覆盖 §6.1 全部字段：
`schema, dataset_id, parent_ids, code_snapshot, engine_sha, game_data_manifest_sha,
obs_digest, action_digest, label_digest, split_registry_sha, shards, counts, status`
（外加可选的 `roots`、原始文档 `raw` 与文件身份 `source`）。

每个 `Shard` 至少含：相对路径 `path`、字节数 `size_bytes`、`sha256` + `digest_scope`、
`episodes`、`ancestors`、`frame_range`（`[start, end)`）、`decisions`（实际决策数）、
`labels_valid`（标签有效数）、`duplicates`（重复情况），以及本层判断 partial 必需的
`split`、`status`、`immutable`。

`load_manifest` **只做解析**（缺字段保留为 `None`，由校验按名报错），`validate_manifest`
才是唯一可以引用的结论。身份字段两种写法都接受，但都必须带 scope：

```text
{"sha256": "...", "digest_scope": "text/lf-normalized"}   # 规范记录
"<64 hex>" + 同级 "<field>_scope": "binary/raw"            # 扁平写法
```

## 3. 十条 fail-closed 规则 → 错误码 → 覆盖测试

| # | 要求 | 错误码 | 覆盖测试 |
| --- | --- | --- | --- |
| 1 | `schema` 不认识 | `P_SCHEMA` | `test_unknown_schema_is_rejected` |
| 2 | 缺必需字段（指出缺哪个） | `P_MISSING_FIELD` | `test_missing_field_is_rejected_and_named`、`test_missing_counts_key_is_rejected` |
| 3 | shard 文件不存在 | `P_SHARD_MISSING` | `test_missing_shard_file_is_rejected` |
| 4 | 字节数与 sha256 与磁盘不符（核心） | `P_SHARD_SIZE_MISMATCH`、`P_SHARD_SHA_MISMATCH` | `test_sha256_mismatch_names_shard_expected_and_actual`、`test_size_mismatch_is_rejected` |
| 5 | 未登记 shard（显式 `roots` 扫描） | `P_SHARD_UNREGISTERED` | `test_unregistered_shard_is_rejected` |
| 6 | train/dev ancestor 交叉（复用 `lineage.group_key`） | `P_ANCESTOR_CROSSING` | `test_train_dev_ancestor_crossing_is_rejected`（反例：`test_same_ancestor_group_inside_one_split_is_not_a_crossing`、`test_quarantine_is_exempt_from_the_crossing_rule`） |
| 7 | obs/action/label digest 与期望不符 | `P_OBS_DIGEST_MISMATCH` / `P_ACTION_DIGEST_MISMATCH` / `P_LABEL_DIGEST_MISMATCH` | `test_expected_digest_mismatch_is_rejected` |
| 8 | `split_registry_sha` 与 splits 配置算出的不一致 | `P_SPLIT_REGISTRY_SHA_MISMATCH`（配置读不出：`P_SPLIT_CONFIG_UNREADABLE`） | `test_split_registry_sha_mismatch_is_rejected` |
| 9 | `status="partial"` 不得整体判为合法完整数据集 | `P_PARTIAL_INCOMPLETE`（唯一 advisory）+ `P_SHARD_NOT_COMPLETED` | `test_partial_manifest_is_never_a_complete_dataset`、`test_complete_manifest_may_not_list_a_mutable_shard`、`test_unrecorded_shard_status_cannot_be_read` |
| 10 | 身份字段只有 `sha256` 没有 `digest_scope` | `P_DIGEST_SCOPE_MISSING`（未知 scope：`P_DIGEST_SCOPE_UNKNOWN`） | `test_bare_sha256_without_digest_scope_is_rejected`、`test_unknown_digest_scope_is_rejected` |

补充规则（同样致命）：`P_SHARD_DUPLICATE`（同一路径登记两次）、`P_SHARD_PATH_INVALID`
（绝对路径/`..` 逃逸）、`P_SHARD_NOT_FILE`、`P_SHARD_UNREADABLE`、`P_SHARD_SPLIT_UNKNOWN`、
`P_SHARD_STATUS_UNKNOWN`、`P_ANCESTOR_INVALID`、`P_LABELS_EXCEED_DECISIONS`、
`P_FRAME_RANGE_INVALID`、`P_DUPLICATES_INVALID`、`P_COUNTS_MISMATCH`、`P_ROOTS_INVALID`、
`P_FIELD_TYPE`、`P_STATUS_UNKNOWN`、`P_DIGEST_NOT_HEX`。
`test_every_emitted_code_is_registered` 断言**没有任何未登记码**可以逃出（`PROBLEM_CODES` 之外
的码在构造时直接抛 `ManifestError`，`test_problem_code_table_is_used_by_the_module`）。

`strict` 语义（与 `lineage.validate_splits` 同构的 strict/degraded 两档）：
`strict=True`（默认）＝「我要一份**完整**数据集」，partial 一律失败；`strict=False` ＝
「校验已完成部分」，`valid` 可为 `True` 但 `complete` 恒为 `False`，且 partial 只出现在
`degradations` 里。**没有任何参数可以把 partial 变成 complete。**

## 4. 换行口径（硬要求）

| 对象 | 摘要函数 | `digest_scope` |
| --- | --- | --- |
| manifest JSON（文本产物） | `lineage.text_lf_digest`（`CRLF -> LF` 后哈希） | `text/lf-normalized` |
| shard 数据文件（二进制） | `lineage.digest_bytes(raw, "binary/raw")` → `lineage.binary_raw_digest` | `binary/raw` |

实现**复用** `lineage.py`，没有第二份实现：`manifest.py` 从 `lineage` 导入
`text_lf_digest` / `digest_bytes` / `load_json_text` / `group_key` / `ancestor_id` /
`DIGEST_SCOPES`，并**不 import `hashlib`**（`Select-String -Pattern 'hashlib'` 无命中）。
JSON 解析也走 `lineage.load_json_text`（先 `CRLF->LF`，并容忍 UTF-8 BOM），所以 CRLF 与 LF
两个文件是同一份身份、可解析出同一 `Manifest`。

对应测试 `test_crlf_and_lf_manifest_copies_share_the_text_digest`：
同一份内容写成 CRLF / LF 两个文件 → `text_lf_digest` 相同、`binary_raw_digest` 不同、
两次 `load_manifest` 的 `source["sha256"]` 相同且等于 LF 摘要、两份都 `valid=True`；
同时断言 `digest_bytes(bytes, "raw")` 抛 `LineageError`（未标注的 scope 不被猜）。

## 5. 真实错误消息样例（本次运行的实际输出）

`sha256` 不符（规则 4，能看出是哪个 shard、期望与实测）：

```text
shard 'train/s1.bin' sha256 mismatch (binary/raw): manifest says
0000000000000000000000000000000000000000000000000000000000000000,
the file at C:/Users/.../data/train/s1.bin hashes to
9a366abe2deaced7fcfb8cd0391e084561d3ca640c8f25202b9a421b4e07e885
```

（`problems[i]` 同时带 `path="train/s1.bin"`、`expected`、`actual`、`digest_scope`、
`resolved`、`size_bytes`，测试逐项断言。）

ancestor 交叉（规则 6）：

```text
ancestor group 'th08-stage1-normal-team0|ep-0001' appears in splits ['dev', 'train']
(ancestors ['human_replay|th08-r2|ep-0001|th08-stage1-normal-team0']);
one task ancestor belongs to exactly one split, whatever the seeds say
(lineage.group_key collapses them)
```

partial（规则 9）：

```text
th08-w03c-synthetic has status 'partial': it is not a complete dataset and must not be
validated as one. Read only completed, immutable shards via partial_view(manifest) and
cite it as a partial view, never as coverage
```

## 6. partial 视图

`partial_view(manifest, *, completed_only=True)` 只保留 **`status == "completed"` 且
`immutable is True`** 的 shard；其余进入 `excluded` 并给出原因
（`status_writing` / `mutable` / `status_not_recorded` / `no_path` / `not_a_shard_object`），
不静默丢弃。`PartialView.complete` **恒为 `False`**（没有参数能把它改成 `True`），并携带
固定 `PARTIAL_DISCLAIMER` 文案。

视图的 `counts` 按**被包含的 shard** 重算（`derived_from: "included shards only (partial view)"`），
因此子 manifest 自洽；视图内部报告用 `strict=False` 校验，`complete` 仍为 `False`。
`partial_view` **没有** `roots` 参数：「roots 下有没有未登记文件」是针对整份数据集的属性，
只该在源 manifest 上判；对过滤后的视图跑该扫描会把每个被排除的 shard 误报成未登记数据。

## 7. 验收（详见 `EXIT_CODES_W03C.txt`）

| # | 命令 | 退出码 | 统计 |
| --- | --- | --- | --- |
| 1 | `python -m compileall -q touhou_rl/v430/data/manifest.py` | **0** | — |
| 2 | `python -m pytest -p no:cacheprovider --basetemp .tmp_v430_pytest_w03b -q tests/test_v430_data.py` | **0** | **29 passed** |
| 3 | `python -m pytest -p no:cacheprovider --basetemp .tmp_v430_pytest_w03b_reg -q tests/test_v430_splits.py tests/test_v430_sequence.py` | **0** | **141 passed**（splits 77 + sequence 64） |

回归口径说明：本切片期望「W09 73 条 + sequence 64 条」，实测 splits 当前为 **77 条**
（sequence 仍 64 条）。逐文件复跑同样是 77 + 64 = 141（见 `EXIT_CODES_W03C.txt` 的
Command 3b）。该文件本切片未改动：其 mtime 为 2026-11-15 11:45:46，早于本切片首次落盘的
11:49:49，且本切片从未以写方式打开 `tests/test_v430_splits.py`。

## 8. TODO 与剩余缺口

代码中留有 3 处 `TODO(W03B):`（按本切片的任务书原文编号）：

1. `manifest.py:823` — dataset 级 `obs/action/label digest` 无法发现「某一个 shard 是在别的
   观测编码下录的」；需要 per-shard digest 或跨 manifest 的 lineage 查询（属本包尚未创建的
   `index` 模块）。
2. `manifest.py:1481` — 通过 `lineage.validate_splits` 失败的分片配置仍会给出 sha256，
   本层只记录 `split_config_valid=False` 而**不**据此判失败；未来「只绑定冻结注册表」的
   更严模式应直接拒绝该情形。
3. `manifest.py:1709` — `partial_view` 尚未绑定源 manifest 身份（`parent_manifest_sha`），
   单独保存的视图文件无法证明来自哪份 manifest。

其余已知缺口（本切片明确不交付）：`index` / `stream` 加载器、shard 内容解码、
把 shard 实际字节与 `Transition`（§6.2）字段逐条对齐、CLI 子命令封装。

**剩余阻塞：无**（本切片全部验收命令已真实通过；上表缺口是后续切片的工作，不是阻塞）。

## 9. 本切片不证明什么

- 不证明任何真实数据集完整、不完整或不重复；fixture 只有两个几十字节的假 shard。
- 不证明任何观测/动作/标签语义正确：`obs_digest` 等值在 fixture 里是任意字符串的 sha256。
- 不证明 `counts`/`frame_range` 与真实游戏帧数一致——只证明它们与登记的 shard 自洽。
- 不证明 split 注册表本身合法：那由 W09 `lineage.validate_splits` 判，本层只绑定其 sha256。
- 不构成能力证据：未启动引擎、未训练、未评测、未做 rollout。
