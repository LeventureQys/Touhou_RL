# G430-E0 门判定（主代理裁定）

日期：2026-11-15，Asia/Shanghai。裁定人：主代理。
范围：v4.3.0 `EXECUTION_PLAN.md` §7 的 `G430-E0` =「W00 身份与可复现源码；W01 完整动作、正负结果、时间/边界实际通过」。
本文件只做门判定，不改任何被判定文件；证据索引见 §4。

## 1 判定结论

**G430-E0 = `pass`（附两条范围限定与一条收尾项，均不削弱「通过」本身）。**

| 判定项 | 结果 | 依据 |
| --- | --- | --- |
| W00：身份与可复现源码 | **达成** | `W00/REVIEWER_VERDICT.md`：54 文件 / 2,290,051 B 字节自洽；468 passed / 3 skipped；两 checkpoint 独立进程 strict 绑定；引擎 SHA 已核验 |
| W01：完整动作 | **达成** | 72 组合互异掩码；`shoot=0` 置 SHOOT 位 **0 例**；bomb 按下沿只触发一次；owner 独占；8 个会话状态全覆盖 |
| W01：正负结果 | **达成** | 同一 `--replay-tape` 通道产出 `stage_clear` 与 `game_over`，且与 v4.2.0 记录**逐字段吻合** |
| W01：时间/边界 | **达成** | 时间契约、`terminated/truncated/segment_end` 互斥、bootstrap 规则、`playing` 不可发键、unknown 不补零 |

## 2 范围限定（必须随判定一起引用，不得单独引用"pass"）

### 2.1 这是工程门，不是能力门

`G430-E0` 只说明「通路与身份可信」。它**不**说明任何策略能力：

- 所有 v430 结果 JSON 的 `capability_gate` 一律 `not_run`；
- `G410-0…G410-7` 仍**全部未通过**，`PLAN_SPEC.evaluation.new_capability_gates` 保持 `not_run_for_v430`；
- 不得用本文件宣布任何通关率、第一面成绩或迁移能力。

### 2.2 W00 附一条已解释例外

`ASSET_SYNC.sha256` 中 `v4.2.0/journal.jsonl` 一条与当前字节不符（pinned `5ed3693f…` vs actual `835b9e18…`）。
三条独立证据证明这是**活文件在清单生成后被追加**，不是资产损坏（哈希三口径全不匹配、mtime 顺序、
`HEAD` blob 与工作区一致）。按 `DEVELOPMENT_GUIDE.md` §12 不重钉；重钉归 W12 清单任务。
W00 的验收原文允许"或逐项解释差异"，故不阻塞本门。

## 3 收尾项（不阻塞本门，但必须闭合）

`PLAN_SPEC.datasets.CONTRACT.repeats_per_source = 2`：

| 来源 | 磁带 | pulse | 结果 | 次数 |
| --- | --- | --- | --- | ---: |
| S1 | seed12 | 48 | `stage_clear` 13,195 帧 | **2**（diffs=0） |
| S2 | seed10 | 48 | `stage_clear` 13,291 帧 | 1 → 第二次进行中 |
| S3 | seed11 | 48 | `stage_clear` 13,291 帧 | 1 |
| F1 | seed12 | off | `game_over` 15,564 帧 | **2**（diffs=0） |
| F2 | seed10 | off | `game_over` 4,615 帧 | 1 |
| F3 | seed11 | off | `game_over` 13,944 帧 | 1 |

**3 success + 3 failure 的来源数已达成**；只有"每源 2 次"是 2/6。
已要求执行代理跑完剩余 4 次，每次只做逐字段比对、不重跑判据。

**为什么这不阻塞 G430-E0**：G430-E0 的原文要求是"正负结果实际通过"——同一通道能产出并区分
成功与失败，这一点已经由 6 条真实记录 + 2 对逐位重复证明。`repeats_per_source` 属
`CONTRACT` 数据集的完整度要求，归 W01 包验收清单，不属门判定项。
但它**必须**在 W12 结算前闭合，否则 `CONTRACT` 数据集不得标 `complete`。

## 4 证据索引

| 内容 | 路径 |
| --- | --- |
| W00 汇总裁定 | `Document/Update/Alpha/v4.3.0/evidence/W00/REVIEWER_VERDICT.md` |
| W00 运行时探针（含并发争用与实测吞吐） | `evidence/W00/RUNTIME_PROBE.md` |
| W00 资产账本核对 | `evidence/W00/ASSET_LEDGER_CHECK.md` |
| W00 换行符口径发现 | `evidence/W00/SHA_LINE_ENDING_FINDING.md` |
| W00 audit 结果（exit 3，1 项已解释例外） | `evidence/W00/audit-001.json` |
| W00 源码身份（54 文件、468 passed、strict 绑定） | `evidence/W00/source_identity.json` |
| W01 汇总裁定 | `evidence/W01/REVIEWER_VERDICT_W01.md` |
| W01 会话层复核（含我发现的阻断 bug 与修复复验） | `evidence/W01/W01B_REVIEW.md` |
| CONTRACT 来源清点 | `evidence/W01/W01C_CLEAR_INVENTORY.md` |
| 协议校验（173 项 / 13 组，`capability_gate=not_run`） | `evidence/W01/contract-001.json` |
| 真实重放原始记录（6 主跑 + 重复） | `.tmp_v430_replay/runs/`（本机资产，不入 Git） |

## 5 我独立执行过的验证（不依赖执行代理自述）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v430_review_w01all -q `
  tests/test_v430_contracts.py tests/test_v430_actions.py tests/test_v430_session.py   # 237 passed, exit 0
python tools/rl/v430.py audit --config configs/v430/assets.json --out .tmp_x.json     # exit 3, 53/51/1/1/0/0
python tools/rl/v430.py validate-contract --config configs/v430/contracts.json --out .tmp_y.json  # exit 0, 173/0
```

另外逐项手工验证过：72 组合互异、SHOOT 释放（对照旧 `v400.cand_to_mask(0,0)=0x1`）、bomb 沿、
owner 独占、5 个特权字段拒绝、`capability_gate` 防伪（篡改成 `pass` 被拒："nothing may promote itself"）、
Wilson `(0,20)→(0,0.16113)` 与我独立算的 16.1% 上界一致、digest 口径在**文件级**对 CRLF/LF 同摘要而对
`binary/raw` 不同、正负对照与 seed10/11 共 6 条记录的双哈希与 v4.2.0 记录逐位比对。

## 6 不影响本门的已知开放项（一并登记）

1. **P1-07 主体未闭合**：187 个被 v4.2.0 文档按路径点名的 `.tmp_v420_*/` 脚本（187/187 被 `.gitignore` 忽略）
   仍在临时目录；W00 只归位了 6 个根级脚本。需独立工作包。
2. **54 个 v420 源文件与全部 v430 新代码仍未被 Git 跟踪** —— 待用户决定是否按显式路径 `git add`（**禁用 `-A`**）。
3. **换行符规范化方案未定** —— 影响历史证据能否重算；v430 新代码已自带正确口径。
4. **CPU 争用**：一个 v4.2.0 A1 扩容（547/750）仍在跑，使每次引擎运行慢约 3%；吞吐预算需其结束后重测。
