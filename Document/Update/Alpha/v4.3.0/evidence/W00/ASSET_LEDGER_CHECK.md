# W00 独立复核：资产账本与工作区完整性

日期：2026-11-15，Asia/Shanghai。执行者：主代理（AGENTS.md §1 例外：复核/汇报）。
本文件记录**主代理自己跑出来的**独立复核结果，用于交叉验证执行代理的结论，不重复其内容、不修改任何资产。

## 1 `ASSET_SYNC.sha256` 全量核对

命令（在 WSL `Ubuntu` 内、以本检出为根，路径与清单中的相对路径一致）：

```bash
cd /mnt/d/WorkShop/Touhou_RL
sha256sum -c ASSET_SYNC.sha256
```

结果：**46 条注册资产中 45 条 OK、1 条 FAILED。**

| 项 | 值 |
| --- | --- |
| 失败条目 | `Document/Update/Alpha/v4.2.0/journal.jsonl` |
| 清单注册值 | `5ed3693f38e4ac4a7f823d6ddd50c7eabbffa15c20acd13566798b3bafaf64e9` |
| 工作区实测值 | `835b9e18d22095a78761dd0773245b39d355cb1c61cdadc33bf6202002fdae38` |
| `HEAD` 提交内 blob 实测值 | `835b9e18d22095a78761dd0773245b39d355cb1c61cdadc33bf6202002fdae38` |

**结论：资产本身完好，账本条目过期。** 证据是 `git cat-file -p HEAD:...journal.jsonl` 的 sha256
与工作区完全一致，说明该文件自 `05c4613` 提交后**没有被改过**；而清单里的 `5ed3693f…`
既不匹配工作区、也不匹配提交内 blob，说明该条目是在日志最后一次追加**之前**生成、
之后随同一次提交被固定下来。这不是静态资产损坏，也不是工作区漂移。

`DEVELOPMENT_GUIDE.md` §12 明确「不把 state/journal 的活文件漂移当静态资产损坏，
也不任意重钉真正不匹配的静态文件」。因此本文件**只登记，不重钉**；重新生成清单属于
W12 结算或专门的清单任务，且必须先确认该日志是否还会追加。

## 2 工作区完整性：3458 份历史文档被删（未提交）

`git status` 有 **3514 个变更、8,241,913 行删除**，其中：

| 前缀 | 删除数 |
| --- | ---: |
| `Document/Update/Alpha/`（v1.0.0…v3.7.x 历史版本目录） | 3458 |
| `Document/Dialogs/`、根级 `v376-*json` 等 | 56 |

**源码未受影响**：`touhou_rl/`、`tools/`、`tests/`、`configs/` 全部完整。删除只发生在历史文档层。

### 2.1 直接后果：v4.3.0 自带的两个文档复核器在本机失败

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python Document/Update/Alpha/v4.3.0/evidence/verify_plan.py     # exit 1
python Document/Update/Alpha/v4.3.0/evidence/verify_review.py   # exit 1
```

两者都在同一条相对链接上抛错：

```text
ValueError: local link V420_REVIEW.md: ../v3.6.0/总索引.md
```

`Document/Update/Alpha/v3.6.0/` 在本工作区**不存在**（虽然在 `HEAD` 里存在），
`V420_REVIEW.md` 第 46 行的 `[旧策略总索引](../v3.6.0/总索引.md)` 因此断链。
`evidence/plan_checks.json`、`evidence/review_checks.json` 记录的通过状态**在本机不可复现**。

### 2.2 处置建议（未执行，需用户决定）

1. **恢复历史文档**：`git restore -- Document/Update/Alpha Document/Dialogs` 等可把 3458 份文档
   恢复到 `HEAD` 状态，两个复核器应随即通过。但这是一次性改动 8.24M 行的动作，
   且看起来像刻意的仓库瘦身，**主代理未擅自执行**。
2. **保留瘦身并修正链接**：把 `V420_REVIEW.md` 中对 `../v3.6.0/`、`../v3.7.x/` 的引用改为
   指向 Git 历史（如 commit + 路径）或注明"历史分支已裁剪"。这属文档修改，需在 W00 白名单外另开条目。
3. 无论选哪种，都应把「文档复核器在当前工作区不可复现」写进 W00 报告，不能沿用旧通过状态。

## 3 三方检出/资产的确切拓扑（避免用错源码根）

| 位置 | 内容 | 能否作源码根 |
| --- | --- | --- |
| `D:\WorkShop\Touhou_RL` = WSL `/mnt/d/WorkShop/Touhou_RL` | **本检出**，含 `touhou_rl/v420/`、`tools/rl/v420*.py` | **是** |
| WSL `/root/Touhou_RL` | 另一份陈旧检出，`HEAD d55c26a`（v4.0.0 期），**无 `v420`** | 否 |
| WSL `/root/Touhou_RL/third_party/th08-rl` | 引擎源码 `HEAD 1523df26` + 全部构建产物（`rl_v3`、`rl_laser`、`rl_events` …） | 引擎来源 |
| `D:\...\third_party\th08-rl` | 被 `.gitignore:97` 忽略，只含 `src/rlharness` 子集，**无 `build/`** | 否 |
| WSL `/root/v400-data/game` | 游戏数据 | 数据来源 |

含义：W430 的真实运行必须**源码取本检出、引擎与数据取 `/root`**，三者分开绑定。
把 `/root/Touhou_RL` 当源码根会缺 `v420`；把 `D:\...\third_party\th08-rl` 当引擎会缺二进制。

## 4 引擎二进制身份（主代理在 WSL 侧实测）

```bash
sha256sum /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern
# 691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6
```

与 `ASSET_SYNC.md` §8 及 v4.2.0 `T01/audit-assets.json` 的 `engine420.sha256_expected` **一致**。
`rl_laser/th08-modern` 与游戏数据目录同样存在。

W00 执行代理的 `source_identity.json` 把 `t01.engine420` 判为 `missing`，是因为它在 Windows 侧
无法解析 `/root/...` 路径并选择 fail-closed。该判定在**其声明范围内正确**（Windows 侧确实看不到），
但**不能**被读成"引擎缺失"。结合本节实测，权威结论是：**引擎存在且 SHA 已核验**。
两处记录应互相引用，不应只引用其中一处。

## 5 未做

- 未恢复任何被删历史文档（见 §2.2）。
- 未修改 `ASSET_SYNC.md` / `ASSET_SYNC.sha256`（见 §1）。
- 未跑训练/评测/能力实验；本节全部内容不改变任何 G430/G410 门状态。
