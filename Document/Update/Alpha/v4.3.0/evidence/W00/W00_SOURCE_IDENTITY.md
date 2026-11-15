# W00 源码身份与可复现性核验

日期：2026-11-15，Asia/Shanghai。工作包：v4.3.0 W00「恢复可复现代码与谱系」的**源码身份与可复现性**一半。
主机：`D:\WorkShop\Touhou_RL`，分支 `v400`，HEAD `6eb76f26a3cb70478130ed8dd9b4c60fdccbb3ed`。
环境：Python 3.12.10，torch 2.14.1+cu132，CUDA 可用。

**本文件只做只读核验与证据记录。** 没有修改 `touhou_rl/**`、`configs/**`、`tools/**`、`tests/**`、`ASSET_SYNC.*`、旧证据、`state/journal/budget`；没有 `git add` / `commit` / 上传；没有删除、移动、重命名任何文件；没有启动引擎、没有训练。

**边界（先说清楚，防止被引用成能力结论）：** 本文件核的是**字节身份**。SHA 吻合只说明磁盘上的字节与报告记录的一致，**不构成**能力通过，**不构成**测试通过，**不构成** G430-E0 通过，也不追认任何 G410 门。

可复现入口：`verify_source_identity.py`（默认无参 = 自校验，`--write` = 重建清单）。
原始退出码：`EXIT_CODES_SOURCE.txt`。逐文件证据：`source_identity.json`。精简登记：`../../registries/code_identity.json`。

---

## 1 结论摘要

| 类别 | 条数 | 内容 |
|---|---:|---|
| 已验证 | 4 组 / 54 文件 + 16 条 SHA 参考 + 2 个 checkpoint + 12 个测试文件 | 见 §2 |
| 有差异 | 2 条 SHA mismatch + 2 条文档/计数差异 | 见 §3 |
| 缺失 | **0 条**（原 `t01.engine420` 已改判 `unverified_host_split`）+ **1 个交付缺口（v420 源码不可同步）** | 见 §4、§6 |
| not_run | 全部真实引擎/训练/评测；G430-E0 判定 | 见 §5 |

**判定分布（30 条参考）：** `verified` 16 / `mismatch` 2 / `not_referenced` 11 / `unverified_host_split` **1** / `missing` **0**。

**一句话结论：** v4.2.0 的 v420 实现与其配置、测试、评测器**在这台磁盘上完整存在且逐字节自洽**（54 文件、2,290,051 字节、清单摘要 `d5f2c4a31461502b88aaf4990cd9b20f2eb5e153655668fe49892c014256f509`），但**没有一行进入 Git，也没有一行进入 `ASSET_SYNC.sha256`**。P1-07 描述的交付缺口是**真实存在的**，只是性质不是"文件丢失"，而是"文件不可同步"。

---

## 2 已验证

### 2.1 四组源码存在性、字节数与清单摘要

| 组 | 文件数 | 总字节 |
|---|---:|---:|
| `touhou_rl/v420/*.py` | 14 | 1,327,322 |
| `tools/rl/v420.py` + `tools/rl/v420_eval_shoot.py` | 2 | 112,652 |
| `configs/v420/*.json` | 26 | 295,093 |
| `tests/test_v420_*.py` | 12 | 554,984 |
| **合计** | **54** | **2,290,051** |

命令：`python Document/Update/Alpha/v4.3.0/evidence/W00/verify_source_identity.py` → **exit 0**，`status=pass`，`problems=[]`。
该命令从字节重算全部 54 条 sha256/size/mtime，与 `source_identity.json` 的钉值逐条比对，并重抽取 §2.2 的每一条期望值。**手改清单无法通过。**

`touhou_rl/v420/` 下另有 `__pycache__/`（cpython-310 与 cpython-312 的 `.pyc`）。它们是构建产物，不计入清单；`git status` 也不报告它们，说明 `.pyc` 已被忽略。

### 2.2 与 v4.2.0 报告记录的 SHA 交叉核验：16 条 verified

期望值全部**在核验时从那四份证据文件重新抽取**（JSON 指针或文本行），不是抄进脚本的常量。

| 期望值来源 | 落在哪个文件 | 结果 |
|---|---|---|
| `T00/assets_verify.json` → `artifacts.local_json.sha256` = `6a104778…` | `configs/v420/local.json` | verified |
| `T00/assets_verify.json` → `artifacts.t00_report.sha256` = `77f9ed49…` | `…/T00/T00.md` | verified |
| `T00/assets_verify.json` → `objects[path=tools/rl/v410_student_eval.py]` = `2db2caaf…` | 评测器 | verified |
| `T00/assets_verify.json` → `objects[path=touhou_rl/v410/__init__.py]` = `f7419904…` | v410 包 | verified |
| `T01/audit-assets.json` → `checkpoint:B0.sha256_expected` = `3bc0ec44…` | `.tmp_v410_wp04_arms/b0_seed0_last.pt` | verified |
| `T01/audit-assets.json` → `checkpoint:20k.sha256_expected` = `15ae1fd5…` | `.tmp_v410_wp05_train_beta05_20k/b0_s0_dagger_step20000.pt` | verified |
| `T01/audit-assets.json` → `evaluator.sha256_expected` = `2db2caaf…` | 评测器 | verified |
| `T02/RAW_SHA256.txt` 5 行 | T02 的 5 份 evaluator 原始产物 | 5 × verified |
| `configs/v420/legacy-reproduce.json` → `evaluator_sha256` | 评测器 | verified |
| `configs/v420/rl-engine-r3-pilot200k.json` → `source_config_sha256` = `83f16c8a…` | `rl-engine-r3-pilot.json` | verified |
| `configs/v420/rl-engine-advnorm-off.json` → `source_config_sha256` = `83f16c8a…` | `rl-engine-r3-pilot.json` | verified |
| `configs/v420/rl-engine-r3-pilot1m.json` → `source_config_sha256` = `a341f0ad…` | `rl-engine-r3-pilot200k.json` | verified |

两个 checkpoint 的实测 SHA 与任务书/报告给出的值完全一致：`3bc0ec44…c741b3a6`、`15ae1fd5…7215bd95`。

### 2.3 路径翻译本身是"已验证"，不是假设

v4.2.0 证据把仓库路径写成 `/mnt/d/WorkShop/Touhou_RL/...`（WSL 侧视图）。本次把它翻译到 `D:\WorkShop\Touhou_RL`，**依据不是同主机假设**，而是：`configs/v420/local.json` 在证据里带有钉值 `6a104778…`，而该钉值在 `D:\...` 下的同一路径上逐字节复现。**内容对上了才允许翻译。** 无法翻译的 `/root/...` 路径保持原样并如实报告为 `missing`（§3.3）。

### 2.4 全新进程加载两个 checkpoint：strict 绑定成功

每个 checkpoint 一个独立 `python` 进程（pid 27080 / 27436），不是同进程 import 复用。
`torch.load(..., map_location='cpu', weights_only=False)` → 再绑定模型类：

| 项 | B0 | 20k |
|---|---|---|
| torch | 2.14.1+cu132 | 2.14.1+cu132 |
| payload schema | `v410-wp04-bc-arm-checkpoint/1` | 同 |
| `state_dict` 键数 | 36 | 36 |
| `state_dict` 总参数量 | 731,802 | 731,802 |
| 直接用 `ProbeNet(...).load_state_dict(strict=True)` | **成功**，参数量 731,802 | **成功**，731,802 |
| 生产加载器 `load_v410_checkpoint()` | **成功**，`parameters_pinned=731802` | **成功**，731,802 |
| v420 入口 `replay_audit.load_frozen_model()` | **成功**，返回同 sha256 | **成功** |
| 判定 | `verified` | `verified` |

模型类为 `touhou_rl.v410.student.diagnostics.ProbeNet`（v420 自身不定义网络类，`replay_audit.load_frozen_model` 明确委托给冻结的 v4.1.0 加载器）。
**注意：** `torch.load` 成功本身没有被计为通过；`strict_binding_ok` 要求**直接 strict 绑定**与**生产加载器**两条都成功。任一条失败即记 `failed` 并保留 traceback 摘录。本次两条都成功，故无 traceback 可摘。

### 2.5 现存 v420 测试（只读运行，未改动任何测试）

```
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v430_w00_src_pytest -q \
  tests/test_v420_contracts.py tests/test_v420_counterfactual.py tests/test_v420_dagger.py \
  tests/test_v420_dataset_2f.py tests/test_v420_eval.py tests/test_v420_eval_shoot.py \
  tests/test_v420_rebuild.py tests/test_v420_replay_audit.py tests/test_v420_reslot.py \
  tests/test_v420_rl.py tests/test_v420_setsup.py tests/test_v420_train.py
```

**`468 passed, 3 skipped`，0 failed，0 error，退出码 0（20.36 s）。** 收集 471 项。
3 个 skip 的测试名与原因（机器可读记录见 `source_identity.json` → `test_run`）：

| 测试 | skip 原因（原文摘要） | 性质 |
|---|---|---|
| `tests.test_v420_replay_audit::test_audit_live_episode_compares_repeats_and_forward_passes` | `host cannot create symlinks; the private engine data dir symlinks the two large blobs` | 主机能力（Windows 符号链接权限），非代码失败 |
| `tests.test_v420_setsup::test_the_expansion_corpus_duplicates_no_frozen_state` | `expansion corpus still being generated: 20 of 750 states merged, 520 of 750 raw shards written (merge status='failed')` | 资产未完成（v4.2.0 T08 扩容在途） |
| `tests.test_v420_setsup::test_the_recorded_expansion_evidence_agrees_with_the_frozen_criteria` | `the corpus-expansion evidence JSON is absent` | 资产缺失 |

**这三条 skip 不是"测试通过"的同义词。** 它们说明 T08 的扩容语料在 v4.2.0 结束时仍处于在途状态（`merge status='failed'`），与 P2-08 的版本状态漂移一致。本次按只读原则**如实记录、未修、未跳过**（skip 由测试自身条件触发）。

### 2.6 编译校验

`python -m compileall -q tools/rl/v420.py touhou_rl/v420` → **exit 0**。
这只代表语法可通过，**不等于**测试通过，**不等于**可运行。

---

## 3 有差异

### 3.1 SHA mismatch：2 条（都不是 v420 源码）

| 文件 | 证据记录 | 本地实测 | 定位 |
|---|---|---|---|
| `ASSET_SYNC.sha256` | `22fe78b3d78a3aa92a06c0b0c02d6ad7cdefd66cb309ac50e6d4311cf1701bd2`（T00） | `0d08ae4eac5fc7c8102244ec7a3fe97e27d3c6b1ff3ec731d97372f39ecf1681` | 变更后 |
| `ASSET_SYNC.md` | `340fb22ba252e2679c84c69e355c3e9fe4ae0cbef781f3d50edc849f184c1801`（T00） | `90fb60e855d30449145fa87fb977a7d93e344ec0607137af1a75bc7c7d9dce72` | 变更后 |

**原因已定位，不是篡改：** `ASSET_SYNC.md` 新增了 §9「v4.2.0 关键资产（成对登记，2026-11-15）」，在 `2026-11-15T04:18:27+08:00` 重钉，向 `ASSET_SYNC.sha256` 末尾追加 26 条 `.tmp_v420_data/**` 二进制条目。该文件体量因此从 T00 时的 2,135 B 增长到 46 行。
`verify_source_identity.py` 对这条解释做了**可证伪的自检**：要求 `ASSET_SYNC.md` 仍含 `2026-11-15T04:18:27` 与 §9 标题，且 `ASSET_SYNC.sha256` 仍有 v420 条目；否则该 mismatch 会被重新判为"未解释"并让校验失败。当前自检通过（`v420_asset_lines=21`）。

**但这也带出本文件最重要的一个反证：** 那次重钉只登记了**二进制**。同一时刻 `ASSET_SYNC.sha256` 里指向 `touhou_rl/v420`、`tools/rl/v420*`、`configs/v420`、`tests/test_v420_*` 的行数是 **0**（本次实测）。见 §6。

### 3.2 计划文档与磁盘现状不符（2 条）

| ID | 文档说 | 磁盘实测 | 判定 |
|---|---|---|---|
| D-1 | `v4.3.0/README.md` §2：「当前检出缺少 `touhou_rl/v420/`、`configs/v420/`、相关测试和评测脚本」 | 14 + 26 + 12 + 2 = 54 个文件全部存在 | **计划文本过时** |
| D-2 | W00 任务说明：`configs/v420/` 有 28 个 json | **实测 26，任务书数字偏大 2，以磁盘为准** | 任务书计数偏高 2 |

**不要盲信文档。** D-1 的后果是实质性的：如果下游按 README 去"从执行端恢复一套不存在的文件"，会去找一个不需要的恢复源，而真正的缺陷（Git 不可见）不会被修。`V420_REVIEW.md` P1-07 自己写的是「当前路径检查：… 均不存在」，同样与现状不符；但 P1-07 的**结论**（"Git 资产保护不应把必要源码、配置、评测器和裁判规则一并留在临时目录"）**被本次证据支持**。

D-2 的口径统一为：**`configs/v420/` 实测 26 个 json，任务书数字（28）偏大 2，以磁盘为准。** 本文件、`source_identity.json`（`known_discrepancies[D-2]`，含 `brief_number=28 / measured_number=26 / delta_brief_minus_measured=2`）、`registries/code_identity.json` 全部使用这一表述，不再出现"28"作为实测值。

### 3.3 `T01.md` 与 `T02/RAW_SHA256.txt` 未给出任何 v420 源码/测试的 SHA

- `T02/RAW_SHA256.txt` 只有 5 行，全部指向 `evidence/T02/*.json`（evaluator 产物），**0 行**指向 v420 源码。
- `T01.md` 以散文形式点名了 `touhou_rl/v420/{__init__,contracts,replay_audit,eval,counterfactual}.py`、`tools/rl/v420.py`、`configs/v420/{protocol,audit-time,legacy-reproduce,local}.json`，但**没有给出任何一个 sha256**。

这不是 mismatch，而是**期望值本身不存在**。见 §3.4。

### 3.4 交叉核验口径：51 / 54 个文件根本没有可比对的期望值

任务要求判定 `verified` / `mismatch` / `not_referenced` / `missing`。完整结果：

| 判定 | 条数 | 说明 |
|---|---:|---|
| `verified` | 16 | §2.2 的 16 条参考全部逐字节吻合 |
| `mismatch` | 2 | §3.1，均非 v420 源码 |
| `not_referenced` | 11（参考记录）+ 51（文件记录） | 见下 |
| `unverified_host_split` | 1 | §4.1，WSL 引擎路径；**不计入 missing** |
| `missing` | **0** | 本检出没有走失的文件 |

**参考记录层面的 11 条 `not_referenced`** = 文档**点名了文件但没记 sha**，因此没有可比对对象（`t01md.*` 10 条 + `t01.local_config_reference` 1 条）。

**文件层面的 51 / 54 条 `not_referenced`：**

| 组 | 有期望值 | 无期望值 |
|---|---:|---:|
| `touhou_rl/v420/*.py` | **0 / 14** | 14 |
| `tools/rl/v420*.py` | **0 / 2** | 2 |
| `configs/v420/*.json` | **3 / 26** | 23 |
| `tests/test_v420_*.py` | **0 / 12** | 12 |

有期望值的 3 个配置是：`local.json`（T00 钉）、`rl-engine-r3-pilot.json` 与 `rl-engine-r3-pilot200k.json`（被别的 v420 配置以 `source_config_sha256` 自钉）。

**这是本节最需要写清的结论：** v4.2.0 报告对 **14 个 v420 实现文件、2 个评测器入口、12 个测试文件、23 个配置** 都没有留下可核验的字节身份。因此"SHA 与报告一致"这句话在这 51 个文件上**无法成立也无法否证** —— 它只能被记为 `not_referenced`，不能写成 `verified`。把它们写成 verified 就是把"没有期望值"冒充成"核验通过"。

---

## 4 缺失：0 条；主机分裂 1 条

### 4.1 `t01.engine420` = `unverified_host_split`（**不是** `missing`）

`T01/audit-assets.json` 记录的 `engine420` = `/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`，在本 Windows 检出上不可见。**它不存在于本检出是因为它属于 WSL 资产主机，而不是因为它丢了。** 把它记成 `missing` 会让 W00 被误读成"引擎缺失"，所以改判为 `unverified_host_split`，并写清两侧事实：

| 字段 | 值 |
|---|---|
| `verdict` | `unverified_host_split` |
| `verdict_reason` | `host_split` |
| `windows_side_observed` | `false` |
| `windows_side_observed_sha256` | `null` |
| `windows_side_note` | `D:\...\third_party\th08-rl` 被 `.gitignore` 忽略，只有 `src/rlharness`，**无 `build/`** |
| `wsl_side_observed_sha256` | `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6` |
| `wsl_side_verified_by` | ① `evidence/W00/ASSET_LEDGER_CHECK.md` §4「引擎二进制身份（主代理在 WSL 侧实测）」，命令 `sha256sum /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`；② `v4.2.0/evidence/T01/audit-assets.json` → `checks_by_object['engine420'].sha256_expected` |
| `citation_holds` | `true` |

**这条引用是"被检查的引用"，不是"被转述的结论"。** `verify_source_identity.py` 每次运行都会回到上述两份文件里核对：`ASSET_LEDGER_CHECK.md` 必须仍然包含那条 `sha256sum` 命令与 `691bfd77…` 这个摘要；`T01/audit-assets.json` 里 `engine420.sha256_expected` 必须仍然等于该摘要。任一条件失效，该条会**自动退回 `missing`** 并让校验失败（`reference … host-split citation no longer holds`）。核对结果记录在 `citation_checks` 字段。

同时必须写明边界：`691bfd77…` 是**主代理在 WSL 侧的实测值**，本次 Windows 端没有、也无法重新测量它。所以是 `unverified`（本端未直接验证）+ `host_split`（原因），而不是 `verified`。两侧记录应互相引用 —— 本文与 `ASSET_LEDGER_CHECK.md` §4 互为引用点。

其余三个 `.tmp_v420_data/**` 资产目录条目（`T01` 的 `v420_data`、`data_dir`、`py420`）在证据里状态是 `present-unhashed`，本身没有期望 SHA，故不构成可比对项。

### 4.2 真正的交付缺口：v420 源码存在，但不在任何可同步范围

见 §6。

---

## 5 not_run

以下**没有做，也不声称做过**：

- 没有启动游戏引擎，没有抓帧，没有按键回放，没有跑任何评测或训练。
- 没有复现 T02 的 3 seed × 2 重复复现、T03 反事实、T05 DAgger、T06 RL、T08 安全集合。这些需要 WSL 引擎 + `/root/v400-data` + `.tmp_v420_data` 资产，**编辑端不具备条件**。
- **G430-E0 未判定，也不由本文件判定。** G430-E0 要求「W00 身份与可复现源码；W01 完整动作、正负结果、时间/边界实际通过」。本文件只覆盖前半句的**字节身份**部分；W01 全部未做。
- 没有对 12 个测试文件做覆盖率、变异测试或语义审查；只跑了它们。
- 没有验证 `.tmp_v420_*` 脚本现在还能不能跑（它们硬编码 WSL/`xvfb-run` 路径，在 Windows 端跑不起来）。

---

## 6 交付缺口：v420 源码「存在但不可同步」

这是 P1-07 的真实形态。它有三层，**每一层都是独立的**。

### 6.1 层一：Git 完全看不到这 54 个文件

```
git ls-files -- touhou_rl/v420                                → exit 0, 输出 0 行
git ls-files -- tools/rl/v420.py tools/rl/v420_eval_shoot.py   → exit 0, 输出 0 行
git ls-files -- configs/v420                                   → exit 0, 输出 0 行
git ls-files -- tests/test_v420_*.py                           → exit 0, 输出 0 行
git status --short --untracked-files=all -- <上述四组>          → exit 0, 54 条全为 "??"
```

### 6.2 层二：它们不是"被忽略"，是"未跟踪"——两者修复方式完全不同

```
git check-ignore -v touhou_rl/v420/__init__.py tools/rl/v420.py configs/v420/protocol.json tests/test_v420_contracts.py
→ exit 1
```

`git check-ignore` 的 **exit 1 = 无匹配**，即这四组**不在 `.gitignore` 里**。
这一点很关键：如果它们是被 `.gitignore` 忽略的（像 `.tmp_*` 那样，属于 AGENTS.md §8.1 的资产保护），那么"不进 Git"是**符合规则的设计**。但它们**没有**被忽略 —— 它们是**本该被提交却漏提交**的源码。这是一个**合规缺陷**，不是资产策略。

### 6.3 层三：`ASSET_SYNC.sha256` 也没有兜底

```
grep 'touhou_rl/v420|tools/rl/v420|configs/v420|tests/test_v420' ASSET_SYNC.sha256 → 0 行
```

`ASSET_SYNC.sha256` 现有 46 行，其中 21 行指向 `.tmp_v420_data/**` 二进制。**源码/配置/测试/评测器 0 行。**
所以这四组既不在 Git，也不在资产清单 —— **两个可同步通道都没有覆盖它们**。

### 6.4 影响

1. **远端无法复现。** 一次 `git clone` 拿不到任何 v420 实现、配置、测试或评测器。远端能跑的单测只剩 v400/v410 的；`tools/rl/v420.py` 的全部子命令、`tests/test_v420_*.py` 的 468 个通过项，在新克隆上**一个都不存在**。
2. **v4.2.0 的数字停留在本机。** T01–T08 的结论（含 3 个 skip 暴露的 T08 在途语料）作为历史记录有效，但"已复现"只能指**这台主机**。这与 v4.2.0 报告里那些 WSL 绝对路径的证据是同一类限制。
3. **身份不可传播。** 本次算出的 54 条 sha256 与清单摘要 `d5f2c4a3…` 目前是**唯一**的外部身份记录。一旦本机工作树被清理或改动，这 54 个文件将**没有任何其他地方可以恢复或校验**。
4. **但它不等于报告造假。** 文件在、字节对得上、模型能加载、测试能过。缺的是**交付方式**，不是**内容**。这正对应 `V420_REVIEW.md` P1-07 的措辞「Git 资产保护不应把必要源码、配置、评测器和裁判规则一并留在临时目录」。

### 6.5 关闭这个缺口需要什么（建议，本次不做）

需要有人对下列四组做出**跟踪决策**（`git add` 属于被禁止的操作，本次未做）：

```
touhou_rl/v420/          (14 .py)
tools/rl/v420.py  tools/rl/v420_eval_shoot.py
configs/v420/            (26 .json)
tests/test_v420_*.py     (12 .py)
```

若决定不提交，则必须改为登记进 `ASSET_SYNC.sha256` 并写明理由 —— 但那样远端仍无法运行。**推荐提交**：这些是纯文本源码/配置/测试，不含任何 `.tmp_*` 资产、checkpoint、引擎或数据，完全落在 AGENTS.md §8.3「只同步代码/文档」的允许范围内。

**关闭缺口后本校验器仍然 exit 0。** 这不只是声称，`verify_source_identity.py` 有一个可执行的证明：

```
python Document/Update/Alpha/v4.3.0/evidence/W00/verify_source_identity.py --self-test
```

它把**同一个**字节校验跑两遍，两遍只更换 git 跟踪注入：
- `before_add_real_git_state`：真实 git 状态（`tracked=0`，54 条 `??`）
- `after_add_synthetic_tracked_state`：模拟 `git add` 之后（`tracked=54`，`status` 无 `??`）

两种情形实测都是 **exit 0 / `problems=[]`**（见 `EXIT_CODES_SOURCE.txt` §6）。
同一个 `--self-test` 还会在**系统临时目录**里造一个一次性 git 仓库，做一次真实 `git add`，记录：`git ls-files` 0 → 2、`status` `?? a.py/?? b.py` → `A  a.py/A  b.py`、而两个文件的 sha256 **前后完全不变**。这就是"提交不改变字节、因此字节门禁不会因修复而翻转"的直接证据，且**不触碰本仓库**。

设计变更说明（相对第一版）：第一版把"四组 `tracked` 必须为 0"当成硬断言，会在缺口被修复后反而 fail —— 那是**反目标**，会让人不敢修复。现在 git 跟踪状态只作为**被记录的可观测事实**出现在 JSON 的 `observations` 与 stdout 里：

```json
"observations": {
  "git_tracking_state": "untracked",
  "git_tracked_in_groups": 0,
  "git_status_untracked_count": 79,
  "git_status_tracked_changed_count": 1,
  "git_check_ignore_exit_code": 1,
  "git_head": "6eb76f26a3cb70478130ed8dd9b4c60fdccbb3ed"
}
```

`pass`/`fail` **只**由 54 个文件的 sha256 / size / mtime 与清单是否一致决定（外加已钉参考判定与 checkpoint 摘要）。

---

## 7 `.tmp_v420_*` 脚本归位清单（只出清单，未拷贝、未移动）

仓库根共 **60** 个（`.py` 18、`.sh` 42）。判定依据：在 `Document/Update/Alpha/v4.2.0/` 下 **149 份** `.md/.json/.jsonl/.txt` 中精确搜索脚本文件名，并对命中的脚本做**一跳以上传递闭包**（读脚本内容找它又调用了哪个 `.tmp_v420_*`）。

| 建议动作 | 数量 | 文件 |
|---|---:|---|
| **需要归位进 `tools/rl/v420_*.py`** 才能让报告数字可复现 | **2** | `.tmp_v420_expected.py`、`.tmp_v420_action_mix.py` |
| 保留为溯源即可（逻辑已被跟踪代码覆盖） | 4 | `.tmp_v420_slot_probe.py`、`.tmp_v420_mask_probe.py`、`.tmp_v420_release_probe.py`、`.tmp_v420_hist_probe.py` |
| 一次性 shell 编排，命令已在报告正文内联，可丢弃 | 3 | `.tmp_v420_smoke_eval.sh`、`.tmp_v420_repro.sh`、`.tmp_v420_replay_all.sh` |
| 已在证据中留有逐字节副本，无需归位 | 1 | `.tmp_v420_replay_probe.py` |
| 未被任何 v4.2.0 文档引用，一次性，可丢弃 | 50 | 见 `source_identity.json` → `tmp_v420_scripts.scripts` |

明细与理由：

- **直接命中 9 个**：`slot_probe.py`、`mask_probe.py`、`release_probe.py`、`hist_probe.py`（`evidence/PREFLIGHT/PREFLIGHT.md` §3 + `experiments/T01-contract-spec.md`）、`expected.py`、`smoke_eval.sh`、`repro.sh`、`replay_all.sh`（`PREFLIGHT.md` §2/§2.0/§5、`PREFLIGHT_REPLAY.md`）、`action_mix.py`（`evidence/T04/ACTION_MIX.md`）。
- **传递闭包新增 1 个**：`.tmp_v420_replay_all.sh` 内部调用 `.tmp_v420_replay_probe.py`（文档正文没有点名它）。**它的 sha256 与 `Document/Update/Alpha/v4.2.0/evidence/PREFLIGHT/replay/replay_probe.py` 完全相同**，即已被归档，故不需要归位。这条是"按文件名 grep 文档"会漏、必须读脚本内容才能发现的一环。
- **需要归位的 2 个**（推荐拷进 `tools/rl/v420_*.py`，本次不拷）：
  - `.tmp_v420_expected.py` — 从 v4.1.0 的 `B0_seed0_full.json` 与 `eval_dev20.json` 抽取 seed 1000/1001/1002 的 frames/steps/deaths/stop_reason/score/`action_sequence_sha256`，正是 T02 复现的**期望值表**（`PREFLIGHT.md` §4 「未修改。提取脚本」）。`touhou_rl/v420/eval.py::summarize` 做归一化与 Wilson 区间，**不做**这张逐 seed 期望表。
  - `.tmp_v420_action_mix.py` — 从 T04 的 dev eval JSON 聚合 18 动作分布，产出 `ACTION_MIX.md` 的表。在 `touhou_rl/v420/*.py`、`tools/rl/v420*.py` 中搜不到 `action_mix` 相关实现。
- **4 个 probe 已被跟踪代码覆盖**：`touhou_rl/v420/replay_audit.py::audit_legacy_shard` 重算了同样四项（HASH 链、slot-1 mask/`shoot_release`、`act_hist` 来源、每决策帧数）—— 见该文件第 363、695、713–774 行。因此它们的**数字**在 v420 入库后可复算，脚本本身只需作为溯源留存。
- **3 个 shell 可丢弃**：都在 `xvfb-run` + `/root/v400-venv/bin/python` + `/root/Touhou_RL/...` 硬编码路径下工作。
  - 一个诚实的补充：`smoke_eval.sh` 的 evaluator 命令在 `PREFLIGHT.md` §2 里**已逐字内联**；`repro.sh` 的 evaluator 命令也已内联，但它的**后处理比较片段**（4 个字段相等性判断）没有内联 —— 丢弃它意味着要按表格重写这 4 行判断，成本近乎零。

**P1-07 的说法需要精确化：** P1-07 写「报告还依赖 `.tmp_v420_*` 的聚合和裁决脚本」。实测支持的是 **2 个逻辑脚本**（`expected.py`、`action_mix.py`）+ 4 个已被跟踪代码覆盖的 probe；`.tmp_v420_s1n_agg.py`、`.tmp_v420_t02_check.py`、`.tmp_v420_t0*c_check*.py` 这些名字看起来像"聚合/裁决"的脚本**反而没有被任何 v4.2.0 文档引用**。所以缺口方向是对的，但依赖面比 P1-07 描述的**更窄**。

---

## 8 复现方式

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'

# 自校验（默认；重算 54 条 sha256、重抽取 30 条参考、重算 checkpoint sha）
python Document/Update/Alpha/v4.3.0/evidence/W00/verify_source_identity.py     # 期望 exit 0

# 验收（W00 源码身份半）
python -m compileall -q tools/rl/v420.py touhou_rl/v420                        # 期望 exit 0
python Document/Update/Alpha/v4.3.0/evidence/W00/verify_source_identity.py     # 期望 exit 0

# 证明"git add 前后都通过"（不触碰本仓库；含一次性临时仓库的真实 git add 对照）
python Document/Update/Alpha/v4.3.0/evidence/W00/verify_source_identity.py --self-test   # 期望 exit 0

# 重建清单与登记（会重写 source_identity.json 与 registries/code_identity.json）
python Document/Update/Alpha/v4.3.0/evidence/W00/verify_source_identity.py --write

# 全新进程重做模型加载（会写回 source_identity.json 的 model_load）
python Document/Update/Alpha/v4.3.0/evidence/W00/verify_source_identity.py --load-checkpoint ".tmp_v410_wp04_arms/b0_seed0_last.pt" --record
python Document/Update/Alpha/v4.3.0/evidence/W00/verify_source_identity.py --load-checkpoint ".tmp_v410_wp05_train_beta05_20k/b0_s0_dagger_step20000.pt" --record
```

`verify_source_identity.py` 不 import torch/numpy（只有 `--load-checkpoint` 模式才 import），默认校验在秒级完成，可在 CI 里跑。

**模式一览：** 默认（自校验）/ `--write`（重建清单+登记）/ `--self-test`（两种 git 状态对照）/ `--load-checkpoint` + `--record`（全新进程模型加载）/ `--record-tests`（junit 统计）/ `--registry`（只重建登记）。

---

## 9 本次新增文件（严格白名单内）

| 路径 | 作用 |
|---|---|
| `Document/Update/Alpha/v4.3.0/evidence/W00/verify_source_identity.py` | 可重复自校验器（默认 / `--write` / `--self-test` / `--load-checkpoint` / `--record-tests` / `--registry`） |
| `Document/Update/Alpha/v4.3.0/evidence/W00/source_identity.json` | 逐文件 sha256/size/mtime + git 状态（observation）+ 30 条参考比对 + 60 个脚本清单 + 2 个模型加载记录 + 测试统计 |
| `Document/Update/Alpha/v4.3.0/evidence/W00/W00_SOURCE_IDENTITY.md` | 本文件 |
| `Document/Update/Alpha/v4.3.0/evidence/W00/EXIT_CODES_SOURCE.txt` | 原始命令与真实退出码 |
| `Document/Update/Alpha/v4.3.0/registries/code_identity.json` | 精简代码身份登记（由 `--write` 生成，含 `tracking_state`、`delivery_gap` 与同步覆盖缺口） |

只读引用的他人产物（**未修改**）：`evidence/W00/RUNTIME_PROBE.md`、`evidence/W00/ASSET_LEDGER_CHECK.md`（§4 作为 `t01.engine420` 的 WSL 侧实测引用源）；以及 `audit-001.json`、`EXIT_CODES.txt`、`W00.md`、`verify_audit.py`。
未触碰：`touhou_rl/**`、`configs/**`、`tools/**`、`tests/**`；`ASSET_SYNC.*`；旧证据与 `state/journal/budget`。

---

## 10 修正轮（2026-11-15 第二次提交）改了什么

| # | 修正 | 落点 |
|---|---|---|
| 1 | 删除"四组 `tracked` 必须为 0"的硬断言；git 跟踪状态改为 `observations` 里的可观测事实，不参与 pass/fail | `verify_source_identity.py::evaluate()`；`source_identity.json.git.tracking_state / status_untracked_count`；`code_identity.json.git.tracking_is_observed_not_gated` |
| 2 | `t01.engine420` 从 `missing` 改判 `unverified_host_split`（含 `windows_side_observed=false`、`wsl_side_observed_sha256=691bfd77…`、`wsl_side_verified_by` 双引用、`verdict_reason=host_split`），并对引用做可证伪自检 | §4.1；`source_identity.json.reference_crosscheck[t01.engine420]` |
| 3 | 26 / 28 口径统一为「实测 26，任务书数字偏大 2，以磁盘为准」 | §3.2 D-2；`known_discrepancies[D-2]`（新增 `brief_number/measured_number/delta`） |

新增证明：`--self-test`（两种 git 注入 + 一次性临时仓库真实 `git add` 对照），实测两次都 exit 0 / `problems=[]`。
