# T01b 证据：`LABEL_SOURCES` 契约缺口修复

- 依据：[PROGRESS.md](../../PROGRESS.md) §6 阻塞 2；[journal.jsonl](../../journal.jsonl) 中 `defect_found`（2026-11-13T17:53:30）；[T02c REBUILD_2F.md](../T02c/REBUILD_2F.md)
- 执行：Provider=DeepSeek / DeepSeek Flash（子代理）。WSL2 `/root/v400-venv/bin/python` 3.10.12（资产主机）；Windows Python 3.12（编辑主机）
- 状态：**缺口已修**（枚举 + 显式规则表 + fail-closed 校验）。三个使用方**零改动**、六个测试文件全绿
- 本文件不含引擎启动、训练、采集；未写 `.tmp_v410_*` / `/root/v400-data/**` / `.tmp_v420_data/T02c|T02d|T04/**`；未 `git add/commit/push`
- 改动文件（严格白名单内 3 个）：`touhou_rl/v420/contracts.py`、`tests/test_v420_contracts.py`、本文件

---

## 1 缺口（改前）

```python
LABEL_SOURCES = ("teacher_slot0", "teacher_slot1", "none")
# build_slot_record():
if label_source != "none" and not label_verified:
    raise ContractError(...)          # 「非 none 必须 verified」
```

后果：两种**真实存在**的产物无法用 `SlotRecord` 表达。

| 使用方 | 写的源 | 冲突点 |
|---|---|---|
| `rebuild.py`（T02c） | `executed_slot1`，`label_verified=false`，无任何 `teacher_*` 键 | 源不在枚举里；且「非 none 必须 verified」正好拒绝它 |
| `reslot.py` / `dataset_2f.py`（T02d/T04） | `teacher_2f`（逐槽 18 路 scores） | 源不在枚举里；`dataset_2f` 只能在 meta 里登记 `unknown_to_contract=["teacher_2f"]` |

`build_slot_record()` 的语义本身没错（未评分的东西不能冒充 verified），错的是它把「verified」当成了「非 none 源」的统一要求，而不是**每个源各自的约束**。

## 2 改后

### 2.1 枚举

| | 值 |
|---|---|
| 改前 | `("teacher_slot0", "teacher_slot1", "none")` |
| 改后 | `("teacher_slot0", "teacher_slot1", "teacher_2f", "executed_slot1", "none")` |

- 新增 `LEGACY_LABEL_SOURCES = ("teacher_slot0", "teacher_slot1", "none")`，即改前的三元组。旧三源在新元组里**不是前缀**而是子序列；相对顺序不变（导入期断言 + 测试断言）。
- 导入期断言「只能加不能删」：旧源一旦从 `LABEL_SOURCES` 消失，模块 import 即 `RuntimeError`。
- `contract_constants()["label_sources"]` 自动变成新五元组（无新增 key、无删除 key）。

### 2.2 规则表 `LABEL_SOURCE_RULES`

显式表，`source -> LabelSourceRule(source, teacher_scored, verified_policy, allows_teacher_fields, semantics)`：

| source | teacher_scored | verified_policy | allows_teacher_fields | 语义 |
|---|---|---|---|---|
| `teacher_slot0` | True | `required` | True | 旧 4 帧块槽 0，教师 18 路评分；未 verified 仍拒绝（**旧规则不变**） |
| `teacher_slot1` | True | `required` | True | 旧 4 帧块槽 1，教师 18 路评分；同上 |
| `teacher_2f` | True | `optional` | True | T02d 逐槽 2 帧教师重查（T04 join 消费）；一次查询可能 0 条 verified，故 `false` 合法且诚实 |
| `executed_slot1` | False | `forbidden` | False | 教师**执行过但未评分**（T02c 重建槽 1）：BC 行为目标，**必须** `label_verified=false`，**不得**带 `teacher_scores`/`teacher_verified_mask`/`teacher_margin` |
| `none` | False | `forbidden` | False | 无标签 |

`VERIFIED_POLICIES = ("required", "optional", "forbidden")`。导入期一致性断言（任何一条不成立则不 import）：

1. `tuple(LABEL_SOURCE_RULES) == LABEL_SOURCES`（一一覆盖、同序）；
2. `LEGACY_LABEL_SOURCES` 全部仍在枚举内（只能加不能删）；
3. `key == rule.source`；
4. `verified_policy ∈ VERIFIED_POLICIES`；
5. `teacher_scored == allows_teacher_fields`（要么有 18 路评分且可带 teacher 字段，要么都没有；不存在「半评分」源）。

### 2.3 新增 fail-closed 校验（都在新增的 `__all__` 里）

模块此前**没有** `__all__`；本次新增，覆盖全部公开常量/类型/函数，并有测试断言 `__all__` 里每个名字都真实存在。

| 符号 | 作用 |
|---|---|
| `LABEL_SOURCE_RULES` / `LabelSourceRule` / `VERIFIED_POLICIES` | 规则表与类型 |
| `TEACHER_LABEL_FIELDS = ("teacher_scores", "teacher_verified_mask", "teacher_margin")` | 受规则约束的 teacher payload 名 |
| `label_source_rule(label_source)` | 取规则；未知源/非 str → `ContractError`（**无默认规则**） |
| `check_label_provenance(label_source, label_verified, *, teacher_fields=())` | 给定源 + verified + 持有的 teacher 字段，违反即 `ContractError`，通过则返回被应用的规则 |

`check_label_provenance` 的拒绝清单：

1. 源不在 `LABEL_SOURCES`；`label_verified` 不是 `bool`（`0` / `"false"` 也拒绝）；
2. `teacher_fields` 里出现**未注册**的名字（否则未知 payload 会绕过下面的规则）；
3. `executed_slot1` / `none` 带任何 `teacher_*` 字段；
4. `teacher_slot0` / `teacher_slot1` + `label_verified=false`（旧规则保留，未放宽）；
5. `executed_slot1` / `none` + `label_verified=true`（伪造）。

### 2.4 `build_slot_record()`

- 旧的「非 none 必须 verified」两条分支替换为一次 `check_label_provenance(...)` 调用；`label_source` 不在枚举、非 bool 的 `label_verified` 仍照旧拒绝。
- 新增 kw-only 参数 `teacher_fields: Iterable[str] = ()`，声明调用方**持有**哪些 teacher payload；`SlotRecord` 字段集冻结、不存 payload，所以这条声明就是规则可以拒绝的对象（`executed_slot1` + `teacher_scores` → 报错，而不是静默丢弃）。
- 向后兼容：`label_source="none"` / `label_verified=False` 默认值不变；`SLOT_RECORD` / `SlotRecord` 字段名一个没改；`as_json()` 输出键不变；新增参数有默认值，旧调用（`replay_audit` 的两处）零改动。

## 3 测试（只追加，未删既有用例）

### 3.1 新增 17 项（`tests/test_v420_contracts.py`，60 → 77）

| 用例 | 断言要点 |
|---|---|
| `test_label_sources_register_the_executed_and_2f_producers` | 新五元组字面量；`teacher_2f`/`executed_slot1` 在册；旧三源为子序列且顺序不变；`contract_constants()["label_sources"]` 同步 |
| `test_label_source_rules_cover_the_enumeration_one_to_one` | `tuple(LABEL_SOURCE_RULES) == LABEL_SOURCES`、集合相等、key==source、policy 合法、语义非空、`label_source_rule` 返回同一对象 |
| `test_every_registered_source_accepts_exactly_its_legal_flags[5 源]` | 参数化遍历枚举：forbidden 源只接受 `false` 且带 teacher 字段即报错；required/optional 源接受 `true`；optional 还接受 `false` |
| `test_verified_is_required_exactly_for_the_legacy_teacher_sources[5 源]` | `must_be_verified` 恰为 `teacher_slot0/1`；`may_be_verified` 恰为非 `none`/非 `executed_slot1` |
| `test_executed_slot1_must_not_be_verified` | `check_label_provenance` 与 `build_slot_record` 两条路径都拒绝 `executed_slot1 + true`；`executed_slot1 + false` 建记录成功 |
| `test_executed_slot1_must_not_carry_a_teacher_payload` | 三个 teacher 字段逐个试都报错；`none`+`teacher_margin` 报错；`build_slot_record(teacher_fields=("teacher_scores",))` 报错；`teacher_slot0` 带同字段**允许**；未注册字段 `teacher_logits` 报错 |
| `test_teacher_2f_may_be_verified` | `teacher_2f + label_verified=True` **允许**（含 `teacher_fields` 全量）；`teacher_2f + false` 也允许 |
| `test_unknown_label_sources_and_non_bool_flags_are_refused` | 未知源字符串（`teacher_snapshot`/`teacher_3f`/`executed_slot2`）、非 str 源、`0`/`"false"` 的 verified 全部报错 |
| `test_the_new_validators_are_exported` | 8 个新符号在 `__all__` 内；`__all__` 中不存在悬空名字 |

### 3.2 唯一被修改的既有断言：`test_protocol_json_mirrors_contract_constants`

`configs/**` 不在本任务写权限内，`configs/v420/protocol.json` 的 `label_sources` 仍是旧三元组，所以「除 `label_sources` 外逐键相等」这条旧断言按定义会失败。判定：**测试期望过时**（契约按本任务定义扩展），不是实现越界。

处理（不是跳过、不是放宽）：

- `payload["label_sources"]` 必须**精确等于** `LEGACY_LABEL_SOURCES`（把冻结文件的期望值钉死，任何人改 protocol.json 或回退枚举都会红）；
- 新增子序列断言：旧三源在 `LABEL_SOURCES` 里的相对顺序不变；
- 其余全部常量仍要求 `mirrored == expected` 逐键相等。

`contracts.py` 自身的 fail-closed 校验**没有**任何一条被放宽或跳过（`teacher_slot0/1` 未 verified 仍然报错，`none` 仍禁 teacher 字段）。

## 4 回归（改前基线 → 改后）

命令（同一组，六个测试文件）：

```bash
# WSL
cd /mnt/d/WorkShop/Touhou_RL
PYTHONPATH=$PWD /root/v400-venv/bin/python -m pytest -p no:cacheprovider \
  --basetemp $PWD/.tmp_v420_data/tests/T01b -q \
  tests/test_v420_contracts.py tests/test_v420_replay_audit.py \
  tests/test_v420_eval.py tests/test_v420_rebuild.py \
  tests/test_v420_reslot.py tests/test_v420_dataset_2f.py
```

```powershell
# Windows
cd D:\WorkShop\Touhou_RL
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v420_pytest_win6 -q `
  tests/test_v420_contracts.py tests/test_v420_replay_audit.py `
  tests/test_v420_eval.py tests/test_v420_rebuild.py `
  tests/test_v420_reslot.py tests/test_v420_dataset_2f.py
```

| 侧 | 改前 | 退出码 | 改后 | 退出码 |
|---|---|---:|---|---:|
| WSL 六文件 | **180 passed** | 0 | **197 passed** | **0** |
| Windows 六文件 | **179 passed, 1 skipped** | 0 | **196 passed, 1 skipped** | **0** |
| WSL `tests/test_v420_contracts.py` | 60 | 0 | **77** | **0** |
| Windows `python -m compileall -q touhou_rl\v420` | — | 0 | — | **0** |

差值与新增用例数一致（+17）；Windows 的 1 skipped 是「Windows 不允许建符号链接」的那条既有 skip，与本次改动无关。`.tmp_v420_contract_fix/{baseline_wsl, final_wsl}.sh` 是本次的复现脚本，日志同名 `.log`。

## 5 使用方影响

| 使用方 | 代码改动 | 测试需不需要改 | 结论 |
|---|---|---|---|
| `rebuild.py`（T02c） | 无 | 不需要 | 它的 `contract_label_sources()` 动态读 `C.LABEL_SOURCES`，`executed_slot1_known` 由 `false` 变 `true`（这正是修复效果）；`test_v420_rebuild.py` 全绿 |
| `dataset_2f.py`（T02d/T04） | 无 | 不需要 | `unknown_to_contract` 由 `["teacher_2f"]` 变 `[]`，`LABEL_SOURCES_DATASET` 全部落入契约；`test_v420_dataset_2f.py` 全绿 |
| `reslot.py`（T02d） | 无 | 不需要 | 不引用 contracts 的 label 枚举/规则，零影响；`test_v420_reslot.py` 全绿 |
| `replay_audit.py` / `eval.py` | 无 | 不需要 | `build_slot_record(... label_source="none", label_verified=False)` 仍合法 |

**结论**：三个使用方都不需要改测试；六个文件里唯一被改的既有断言是 `tests/test_v420_contracts.py` 自己的 protocol 镜像测试（原因见 §3.2）。

## 6 遗留（非阻塞，白名单外）

- `rebuild.contract_label_sources()` 与 `dataset_2f.contract_label_sources()` 里的 `note` 文本仍写着「契约还没登记这个源 / 用 check_* 逐字段绕过」。它们只是说明字符串（不影响任何校验），但下一次触碰这两个模块时应改为读 `LABEL_SOURCE_RULES`，让说明随表自动更新——否则新 `meta.json` 会出现「`executed_slot1_known: true` + 旧说明」的措辞不一致。
- G420-1 的「verified 标签不得含 `executed_slot1`」现在由契约层强制（`label_verified=true` 直接报错），但**统计口径**仍属 T04 的 join 自检，本次未验证（未跑 T02c/T02d/T04 产物）。

## 7 明确没做

- 没有启动引擎、没有训练、没有重跑 T02c/T02d/T04，没有写 `.tmp_v420_data/T02c|T02d|T04/**`、`.tmp_v410_*`、`/root/v400-data/**`。
- 没有改 `touhou_rl/v420/{replay_audit,eval,counterfactual,rebuild,reslot,dataset_2f,batch_2f,__init__}.py`、`tools/rl/v420.py`、`configs/**`、其它 tests、`touhou_rl/v400|v410/**`、旧证据、`ASSET_SYNC.*`。
- 没有 `git add/commit/push`（`contracts.py` 与 `tests/test_v420_contracts.py` 仍是未跟踪新文件）。
