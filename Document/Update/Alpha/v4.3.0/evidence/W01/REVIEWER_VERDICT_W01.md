# W01 验收裁定（主代理复核记录）

日期：2026-11-15，Asia/Shanghai。裁定人：主代理。
本文件只做验收裁定与独立复核记录，不重复执行代理的证据内容，不改任何被复核文件。
**本裁定在 failure 来源补测完成前为"有条件接受"**，缺口写在 §5。

## 1 交付清单（四个切片）

| 切片 | 交付 | 裁定 |
| --- | --- | --- |
| W01a | `touhou_rl/v430/{__init__,contracts,actions}.py`、`configs/v430/contracts.json`、`tests/test_v430_{contracts,actions}.py`、`evidence/W01/{W01A.md,EXIT_CODES_W01A.txt}` | **接受** |
| W01b | `touhou_rl/v430/{session,evaluate}.py`、`tools/rl/v430.py`（只增 `validate-contract`）、`tests/test_v430_session.py`、`evidence/W01/{contract-001.json,W01B.md,EXIT_CODES_W01B.txt}` | **接受** |
| W01c（前置） | 真实引擎正/负对照与三个磁带的重放、`.tmp_v430_replay/`（gitignored）、`evidence/W01/W01C_PREFLIGHT.md` | **接受**（failure 补测未完成，见 §5） |
| 主代理复核 | `evidence/W01/W01B_REVIEW.md`、`W01C_CLEAR_INVENTORY.md`、本文件 | 已落盘 |

## 2 主代理独立验证（不是转述代理结论）

### 2.1 测试

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v430_review_w01all -q `
  tests/test_v430_contracts.py tests/test_v430_actions.py tests/test_v430_session.py
# 237 passed in 0.64s   exit 0
```

分项：contracts 103 + actions 44 + session 90 = 237。

### 2.2 我亲手实测的行为（不依赖代理自述）

| 属性 | 我的实测 | 结论 |
| --- | --- | --- |
| 72 组合互异 | `all_joint_masks()` 得 72 个互异掩码 | 通过 |
| **SHOOT 释放** | 全部 72 组合中 `shoot=0` 却置 SHOOT 位的：**0 例**；`shoot=1` 未置位：**0 例** | 通过 |
| 对照旧实现 | `v400.cand_to_mask(0,0)` = `0x1`（强制常开），v430 同请求**不按射击** | 缺陷确已修复 |
| bomb 按下沿 | 首次 `edge_frames=(0,)`、`total=1`；持续按住 `edge=False`、`frames=()`、`total` 仍 1；松开后再按 `total=2` | 通过 |
| owner 独占 | 第二个 owner `acquire` → `OwnerConflictError`；释放后 session 可获 | 通过 |
| session `actor_loss_mask` | session=**0**、actor=1 | 通过 |
| 特权字段拒绝 | `engine_frame`/`stage_truth`/`resource_truth`/`seed`/`teacher_action` 全 `ContractError` | 通过 |
| 8 个会话状态 | 4 个「actor 驱动」返回 `None`，4 个「会话发键」返回合法动作（`src=session`、`loss_mask=0`） | 通过 |
| 场景→状态迁移 | `menu_overlay→menu`、`stage_banner→transition`、`game_over_prompt→game_over`、`ending_credits→ending`，`owns=True`、`loss_mask=0` | 通过 |
| **像素 digest 口径** | 传 `text/lf-normalized` 被拒："captured pixels are binary and are never line-ending-normalised" | 通过 |
| 时间链防造假 | `elapsed_capture_s=0` 被拒："a fake zero would drop time from the reward chain"；`>captured_at_s` 被拒 | 通过 |
| 不抢键盘 | `playing` 时 `compile_session_action()` → `SessionStateError`："an implicit acquire would race the actor" | 通过 |
| 过面奖励不丢 | `attach_reward('clear',1.0)` 后 `reward_rollup()` = `{"clear":{"sum":1.0,"count":1,"spans":[0],"states":["playing"],"elapsed_capture_s":0.016667}}` | 通过（保留 span/状态/耗时归因） |
| **Wilson 区间** | `(0,20) → (0.0, 0.16113)`；与我独立算的 0/20 上界 **16.1%** 完全一致；`(50,100)→(0.4038,0.5962)`、`(10,10)→(0.7225,1.0)` | 通过 |
| 分母为 0 | `(0,0)`/`(1,0)` → `ZeroDenominatorError`；`(-1,5)`/`(3,2)` → `EvaluationError` | 通过 |
| **digest 口径语义** | `text/lf-normalized` 对换行**不敏感**、`binary/raw` **敏感**；非法 scope 被拒 | 通过 |
| 裁判隔离 | `assert_judge_isolation()` → `isolated: True`，允许清单仅 stdlib + `touhou_rl.v430.contracts` | 通过 |
| `contract-001.json` | 喂 `assert_result_schema` → **PASS**；把 `capability_gate` 改成 `pass` → 被拒："nothing may promote itself" | 通过 |
| `audit` 未被破坏 | 重跑 → exit **3**，53/51/1/1/0/0、`wsl up`，与 `W00/audit-001.json` 一致 | 通过 |

### 2.3 真实引擎重放（W01c 的核心）

| 对象 | 结果 | 与 v4.2.0 记录比对 |
| --- | --- | --- |
| 正对照（seed12，pulse=48） | `stage_clear=True`、13,195 帧、2,631,430 分、0 死亡、`exit=0`、wall 125.9 s | **`frames`/`score`/`stage_clear`/`stop_reason`/`deaths`/`action_sequence_sha256`/`observed_input_sha256` 全部 MATCH** |
| 负对照（同磁带 pulse=1000000007） | `stage_clear=False`、`game_over`、15,564 帧、3 死亡、`score@13195=141379` | **全部 MATCH** |
| seed10（pulse=48） | `stage_clear=True`、13,291 帧、2,868,224 分、0 死亡、sha `3cb366a0…` | 新来源 |
| seed11（pulse=48） | `stage_clear=True`、13,291 帧、2,768,761 分、0 死亡、sha `19be3bc6…` | 新来源 |

`engine_sha256` 每次均为 `691bfd77…`，与登记一致。数据目录均为 `cp -a` 的私有副本（每个 474 MB），
未触碰共享树（引擎会写 `score.dat`）。

**注**：`mask_trace_sha256` 不在 per-run schema 内（由 T08 证据发射器另行计算），
两边都是 `None`，属"该字段不存在"而非"数值相等"，本裁定不把它当证据。

## 3 「同一生产控制入口」的判定

正负对照走的是 `tools/rl/v420_eval_shoot.py --replay-tape`：真实发键（`STEP mask n`）、真实读回、
真实裁判状态（不是 autoplay、不是内存注入）。这条**满足** EXECUTION_PLAN §2「同一控制入口」的要求：
正负结果由**同一条通道**产出，唯一的差别是 pulse 周期。

**但**：seed10/11 的分类运行绕过了该工具内一条把 shard sha 硬钉在 seed12 上的断言
（`shard_pin_bypassed=true`，其余全部调用工具自身函数）。因此：

- 它们**可以**作为「这三个磁带能否通关」的分类证据；
- 它们**不能**被无保留地当作「已登记的 CONTRACT 来源」——需在同一 instrument 身份下以未绕过
  路径确认，或把对应 shard sha 正式登记进来源清单。

## 4 CONTRACT 数据集现状

| 类别 | 要求 | 实测 | 结论 |
| --- | ---: | ---: | --- |
| success sources | 3 | **3**（seed12/10/11，动作序列哈希互异） | **达成**（附 §3 的 pin-bypass 限制） |
| failure sources | 3 | **3**（seed12/10/11 + pulse off，全 `game_over`） | **达成** |
| repeats per source | 2 | **6/6 全部 2 次，且每次比对 `diffs=0`** | **达成** |

六条来源的最终表（每条两次运行，9–12 个身份字段逐项相同）：

| # | 磁带 | pulse | 结果 | 帧 | 死亡 | `action_sequence_sha256` | 次数 | 一致性 |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |
| S1 | 12 | 48 | `stage_clear` | 13,195 | 0 | `5e9d66c4…` | 2 | **diffs=0** |
| S2 | 10 | 48 | `stage_clear` | 13,291 | 0 | `3cb366a0…` | 2 | **diffs=0** |
| S3 | 11 | 48 | `stage_clear` | 13,291 | 0 | `19be3bc6…` | 2 | **diffs=0** |
| F1 | 12 | off | `game_over` | 15,564 | 3 | `77455bef…` | 2 | **diffs=0** |
| F2 | 10 | off | `game_over` | 4,615 | 3 | `a9f3d28b…` | 2 | **diffs=0** |
| F3 | 11 | off | `game_over` | 13,944 | 3 | `4026e703…` | 2 | **diffs=0** |

**`CONTRACT` 的行级要求（3+3 行、每行 2 次）与 `repeats_per_source: 2` 至此达成**，且六个来源各自的两次运行逐字段相同。

### 4.1 但"6 行"不等于"6 个独立来源"——这一点必须同时引用

执行代理在 `W01C_PREFLIGHT.md` §5.2 明确指出了我先前没有写清的一点，我复核后**接受它的更正**：

| 层面 | 实际 |
| --- | --- |
| 行数 | 6 行（S1/S2/S3/F1/F2/F3） |
| **独立录制 episode** | **只有 3 段**（seed10 / seed11 / seed12） |
| 成功与失败的关系 | **同一段磁带在两种仪器设置下的两次运行**，不是两批不同结果 |
| failure 的性质 | **仪器诱导**（把 SHOOT 释放通道关掉），**不是录制里真实发生过的失败** |

按 `TRAINING_PLAN.md` §4 的分组口径（"重复重放、由同一起点生成的多个分支属于同组"），
**独立来源是 3 组，不是 6 个**。

因此：

- 若 `CONTRACT` 的要求是"3 个不同的 success 见证 + 3 个不同的 failure 见证"（**行级**），
  当前是 **3 + 3，各 2 次**，成立；
- 若要求"success 与 failure 必须来自**不同的真实 episode**"（**来源级**），
  当前只有 **3/6**——缺 3 个"录制里真实失败"的 episode，需要**新采集**。

**我的裁定按行级要求记"达成"，但把来源级缺口明确写在这里，不许只引用"3+3 达成"而不说这句。**

### 4.2 还有一条更硬的限制：通关发生在录制区间**之后**

三段磁带都在**换面之前**就结束了（`shard.json` 的 `note`：*"the recording ends before the stage transition;
the tail repeats the last ..."*），通关是靠 `--tail 64` **重复最后一个动作**推过去的。

含义：
- 这仍然是 v4.2.0 官方正对照的**同一配方**（它自己也用 `--tail 64`），所以"复现"是成立的；
- 但**不能说"录制本身记录了换面"**，也**不能说"某策略学会了最后那段"**——最后 64 决策是同动作重复。

### 4.3 成本（供 ST00 预算核算）

12 次运行全部**串行、单引擎**，合计 **wrapper wall 1,410.769 s ≈ 23.5 分钟**；
单次最贵 144.007 s（F1），全部远低于 300 s 兜底，无一次 `cost_contention`；
同期另有 3 个扩容引擎在跑，故比 v4.2.0 记录慢 **3.0–4.0%**。
另加 9 份私有 data dir 的 `cp -a`（约 2.8 GB）。



### 4.1 确定性已验证两次（S1 与 F1，diff 均为 0）

两次独立运行在 9 个身份字段上全部相同：

```text
S1 (pos): frames 13195 | score 2631430 | stage_clear True  | stage_clear | deaths 0.0
          decisions 6598 | action 5e9d66c4… | input 315a2ab6… | engine 691bfd77…   diffs=0
          wall 126.440 s vs 125.241 s
F1 (neg): frames 15564 | score 163945  | stage_clear False | game_over  | deaths 3.0
          decisions 7782 | action 77455bef… | input a1b3bdd6… | engine 691bfd77…   diffs=0
          wall 144.447 s vs 144.969 s
```

**耗时可变、结果逐位不可变**——这正是"可复现"该有的形态。且 S1/F1 恰好就是与 v4.2.0 原始记录
逐字段吻合的那一对，所以重复验证落在最有价值的两个来源上。

### 4.2 failure 侧的配对结构与统计限制

三个 failure 都是"**同一磁带、只把 pulse 周期 48 改成 1000000007**"的受控操纵，
它们的**祖先与对应 success 相同**。因此：

- 对 W01 的**仪器校准**用途完全成立（正负本就该同源）；
- 但若将来用于**成功率统计**，必须按 same-ancestor 聚类，否则会把 6 条记录当成 6 个独立祖先
  （`TRAINING_PLAN.md` §4：由同一起点生成的多个分支属于同组）。



## 5 收口状态

| 项 | 状态 |
| --- | --- |
| failure sources 3 个（行级） | **已闭合**：seed12/10/11 + pulse off 全部 `game_over`（15,564 / 4,615 / 13,944 帧，各 3 死亡） |
| 确定性重跑（6 源各 2 次） | **已闭合**：6/6 全部逐字段相同 |
| **来源级独立性（3 组 vs 6 行）** | **未闭合**：failure 侧全部是仪器诱导，非录制真实失败（见 §4.1） |
| **通关发生在录制区间之后**（`--tail 64` 补上） | **限制已记录**（见 §4.2），不得据此声称录制含换面 |
| pin-bypass 的正式化 | **仍开放**：S2/F2/S3/F3 是 `classification` 证据；需未绕过路径确认或正式登记 shard sha |
| 真实过面回放的奖励链 | **仍开放**：合成测试覆盖了 span/耗时/事件/reward 归因机制，但**真实过面**的奖励链尚未在引擎上验证 |

**分类（重要，避免误读）**：

- **前两项**（failure 行数、确定性重跑）**已闭合**，且正是 G430-E0 所要求的"正负结果实际通过"。
- **第 3–6 项是 W01 包自身的完整度缺口**，不属 G430-E0 的判定文字，但**必须在 W12 结算前处理**，
  否则 `CONTRACT` 数据集不得标 `complete`。其中第 3 项（来源级只有 3 组、failure 为仪器诱导）
  是**最实质的一条**：它意味着"3 个失败来源"目前只满足行数，不满足"3 个不同的真实失败 episode"。

## 6 G430-E0 判定

`G430-E0` = 「W00 身份与可复现源码；**W01 完整动作、正负结果、时间/边界实际通过**」。

- W00：**已具备**（见 `W00/REVIEWER_VERDICT.md`，含 1 项已解释例外）。
- W01 的「完整动作」：**通过**（72 组合、SHOOT 释放、bomb 沿、owner 独占、特权隔离全部实测）。
- W01 的「正负结果」：**通过**——同一通道产出 `stage_clear` 与 `game_over`，与历史记录逐字段吻合，
  且 6 条来源各重复 2 次全部 `diffs=0`。
- W01 的「时间/边界」：**通过**（时间契约、`terminated/truncated/segment_end` 互斥、bootstrap 规则、
  owner 切换、`playing` 不可发键、unknown 不补零、8 状态全覆盖）。

**裁定：G430-E0 = `pass`。**（正式判定见 `evidence/G430_E0_VERDICT.md`；§5 的两项开放缺口
不属该门判定范围，但必须随判定一起引用。）

**无论如何不改变**：`capability_gate` 一律 `not_run`；W01 是工程门，不是能力门；
不得据此宣称任何 G410 能力。
