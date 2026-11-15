# v4.3.0 门表总览（截至 round 25）

日期：2026-11-15，Asia/Shanghai。维护者：主代理。
**本表是导航，不是能力结论。** 逐条门判定以引用的证据文件为准；本表不新增任何通过。

## 1 工程门（G430）

| 门 | 范围 | 状态 | 依据 / 缺口 |
| --- | --- | --- | --- |
| **G430-E0** | W00 身份与可复现源码；W01 完整动作、正负结果、时间/边界 | **pass** | `evidence/G430_E0_VERDICT.md`（含 2 条必须一起引用的范围限定） |
| G430-T | W02 可重复战术见证与可见线索；W03 真实微段 ≥9/10 及留出三 seed | **not_run** | W02 仅完成 **观测编码层**（`observation.py`，47 tests）；战术/反事实未做。W03 仅完成 **GAE/折扣**（45 tests）；分段/burn-in、表征、BC、微段均未做 |
| G430-C | W04 课程层级、真实连接、自然起点成绩与候选冻结 | **not_run** | W04 未开始 |
| G430-P | W05 像素反馈与隔离；W08 严格像素训练通路 | **not_run** | W05/W08 未开始 |
| G430-D | W06 真实时序与全流程运行 | **not_run** | W06 未开始；Windows 侧抓帧/发键时延未测 |
| G430-X | W09 谱系；W10/W11 干净任务与真实资源核验 | **partial：仅谱系腿** | W09 已完成（`splits.json` strict `passed=True`、73 tests、2 个冻结干净留出家族）。W10/W11 未开始，故 G430-X **不通过** |

## 2 能力门（G410）

| 门 | 状态 | 说明 |
| --- | --- | --- |
| G410-0 | **not_run_for_v430** | 实验可信性；v430 尚未重新裁定。历史 G410-0 passed 属 v4.1.0 范围，**不自动继承** |
| G410-1 | **not_run_for_v430** | 多面教师腿未做 |
| G410-2 | **not_run_for_v430** | 像素第一面：**0 次真实过面**；无候选 |
| G410-3 | **not_run_for_v430** | 像素反馈检测器未训练 |
| G410-4 | **not_run_for_v430** | R0/R1/R2 对照未做 |
| G410-5 | **not_run_for_v430** | 完整一周目未做 |
| G410-6 | **not_run_for_v430** | 干净留出**已冻结**（W09），但 Z/A/S/A-ad 适应未跑 |
| G410-7 | **not_run_for_v430** | 第二作 STRICT 未做 |

**所有 v430 结果 JSON 的 `capability_gate` 一律 `not_run`；本表不改变这一点。**

## 3 已完成的工作包（可验收）

| 包 | 交付 | 验收证据 |
| --- | --- | --- |
| **W00** | `tools/rl/v430.py`(audit)、`configs/v430/assets.json`、6 个归位脚本、身份与账本复核 | `evidence/W00/REVIEWER_VERDICT.md`、`audit-001.json`(exit 3, 1 项已解释例外)、`source_identity.json` |
| **W01a** | `contracts.py`(82KB)、`actions.py`(28KB)、`configs/v430/contracts.json` | 147 tests；72 组合互异、SHOOT 释放、bomb 沿、owner 独占实测 |
| **W01b** | `session.py`(70KB)、`evaluate.py`(54KB)、`validate-contract`、`contract-001.json` | 90 tests；173 项契约校验；防伪闸门实测；`W01B_REVIEW.md`（含我发现的阻断 bug 与修复复验） |
| **W01c** | 真实引擎重放：3 success + 3 failure 来源 | `W01C_CLEAR_INVENTORY.md`、`.tmp_v430_replay/runs/`；**6/6 来源各双跑且全部 `diffs=0`** |
| **W02a** | `observation.py`(54KB)（O2 编码） | 47 tests + 237 回归；整数均值/灰度/裁剪双向/复用契约实测；`evidence/W02/W02A.md` |
| **W09** | `data/lineage.py`(72KB)、4 个注册表、`configs/v430/splits.json` | 73 tests；strict `passed=True`；`split_registry_sha=9160cf84…`；`evidence/W09/W09.md` |
| W03a2 | `sequence.py`(GAE + 折扣) | 45 tests；gamma/lambda 与手算差 0.000e+00；跨集切断双向验证（精确 `==`） |
| W03a3 | `sequence.py`(分段 + burn-in + padding) | 64 tests（+19）；loss 区精确划分 `[0,n)`；`effective=512 == EFFECTIVE_MINIBATCH_STEPS`；`segment_end` 语义定案为"不改终止语义" |
| W03b | `data/manifest.py`(75.7 KB) | 29 tests；sha 不符报错指名 shard+口径+期望+实测；纯 stdlib；强制 `complete` 只列 immutable shard |
| W03c | `models/actor.py`(55.3 KB) + `models/__init__.py` | 20 tests；**参数 2,726,671（2.727M）**，与 §4.3「约 2.73M」差 −0.12%；72 组合概率和实测 1.000000（我）与 0.99999988（它）；无 BatchNorm；`memory` **不是**递推状态（实测） |
| W09b | `tools/rl/v430.py` 增 `validate-splits` | +4 CLI 测试；`splits-001.json` 已产出；`audit`/`validate-contract` 行为未变 |
| W03d | `learn.py`(46.5 KB) BC 编排 | 17 tests；**同 seed 两次训练每步 loss 与最终权重全等**（我实测）；`shuffle=False` 时 seed 无作用（正确）、`shuffle=True` 时不同 seed 确实不同；**强制 uint8 输入**（float 图像被拒："`/255` belongs inside the model"）；`load_checkpoint` 的 o2_digest 校验**无条件执行**（我篡改 digest 后不传参→被拒） |
| W03f | `data/manifest.py` 观测契约守卫 | +7 tests（data 29→36）；`KNOWN_OBS_CONTRACTS=('obs-v430-o2','obs-v420-2f')`；新码 `P_OBS_CONTRACT_UNKNOWN`/`P_OBS_SHAPE_CONTRACT_MISMATCH`（PROBLEM_CODES 31→33）；**AST 四层断言模块内不存在上采样/补零/通道合成**；反洗白：legacy 契约名即便声明 O2 几何仍判 `o2_compatible=False` |
| W03g | `tools/rl/v430.py` 增 `train-bc` | +7 tests（learn 17→24）；`train-bc-001.json` 实测 `command_class=training`、`capability_gate=not_run`、`dataset_kind=synthetic`、`research_result.passed=false`；**防伪四级构造 + evaluation 对照组**证明拒因确为 command_class；**发现并修复跨进程 float 不确定性**（多线程归约顺序 → 固定 `torch.set_num_threads(1)`） |
| W03h | 真实 O2 采集（64 样本） | `.tmp_v430_o2/o2-64/` 9 文件约 110 MB；5 路通道**形状全部等于 `O2_SHAPES`**；`blank 0/64`、`near_blank 0/64`；`meta.json` 含 `obs_contract=obs-v430-o2`、`alpha_mode=any`、`legacy_encoder_used=false`、`engine.sha256`、`tape.sha256_matches_frozen`、`orientation_probe`、`blank_audit.encoder_can_not_detect`；**原始 RGBA 已保留**（`frames_rgba (64,480,640,4)`，78.6 MB）。**主代理独立跑通闭环**：真实数据 → BC → loss **4.2716 → 3.3055**（8 步） |
| W03i | `observation.py` 增 `alpha_mode` | +9 tests（observation 47→56）；默认 `"strict"` 语义不变（实现后未加测试时单独跑过 47 passed）；`ALPHA_MODES=(opaque_255,unused_0,binary_mixed,any)`；**`o2_digest` 未变**（`cb1616e4…`），run 级身份改由 `encoder_digest` 区分（strict `37ebb8d4…` ≠ any `90e43af6…`）；`"any"` 下 alpha 255 vs 210 的 `global_rgb`/`play_gray`/`hud_gray` **逐字节相等** |
| W03k | 1000 决策真实 O2 语料 | 完整重放 13,195 帧 / `stage_clear` / 172.119 s / exit 0；1000 样本、12 文件 553,082,238 B；0 空白；provenance 1000/1000（守卫实测拒收过一条 `tail_repeat`）；`valid=True problems=[]`；BC loss 4.2716→3.2633；**覆盖率如实标注 6/9（缺 1,5,8，`move=4` 仅 3 次）**；两项记账缺陷已处置（`dataset_id` 推导、`sample_count` 权威、`manifest_sha256` 经 LF 口径核对相等） |
| W04a | `ppo.py`(25 KB) PPO 核心 | 38 tests；**`clip` 缺 `old_values` → 抛错**（不静默降级 MSE）；`kl_estimators` 强制返回两个（实测 `k1` 可取 −0.5、`k3` 恒 ≥0）；`clipped_surrogate` 手算 `0.149999991` vs 实测同值、`clip_fraction` 精确 0.5；`gradient_diagnostics` `[3,4]→norm 5.0`、反向参考 `cosine=-1.0 conflict=True`；六类非法输入全部 `PpoError` |
| W03l | **1099 样本跨盘 stride-6 语料** | 完整重放 13,195 帧 / `stage_clear` / **223.371 s** / exit 0；**1099 样本**（= 黄金参照 `schedule[::6][1:]`）；**`move` 9/9 无缺失**、`focus` 2/2、**`cand` 17/18**；直方图 `{0:468,1:28,2:203,3:172,4:32,5:21,6:78,7:76,8:21}` **与离线预测逐项相同**；0 空白/0 近空白；provenance 1099/1099；**黄金闸门 PASS**（间隔恒 12、帧与 cand **0 处不符**）；**1-shard 模型**（`decisions=1099`、`episodes=1`、ratio 1.0x、`three_way_equal=true`）；`dataset_id=…-strided-1099`；BC loss 4.3418→3.2010 |

**当前 v430 全测试集：557 passed / 0 failed / exit 0**（contracts 103 + actions 44 + session 90 + observation 56 + splits 77 + sequence 64 + models 20 + data 36 + learn 29 + ppo 38）。

### 3.1 三批真实 O2 语料对照（主代理实测）

| 语料 | 样本 | `move` | `cand` | 记账模型 | `counts.decisions` | 真实行数 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `o2-64` | 64 | 5/9 | 9/18 | 8-shard，**缺 `sample_count`** | 512 | 64 |
| `o2-1k` | 1000 | 6/9 | 10/18 | 8-shard + `sample_count` | 8000 | 1000 |
| **`o2-strided`**（seed12） | **1099** | **9/9** | **17/18** | **1-shard（正确）** | **1099** | **1099** |
| **`o2-stride-seed10`** | **1099** | **9/9** | **17/18** | **1-shard（正确）** | **1099** | **1099** |
| `o2-stride-seed11` | **1099** | **9/9** | **17/18** | **1-shard（正确）** | **1099** | **1099** |

**三批 stride-6 语料已全部完成并逐项验收**（主代理实测，`verify_o2_corpus.py`）：

| 语料 | `move` | `counts.decisions`=实际 | `validate_manifest` | `assert_o2_compatible` | provenance |
| --- | --- | --- | --- | --- | --- |
| `o2-strided`（seed12） | 9/9 | True | `valid=True problems=[]` | accepted | 1099/1099 |
| `o2-stride-seed10` | 9/9 | True | `valid=True problems=[]` | accepted | 1099/1099 |
| `o2-stride-seed11` | 9/9 | True | `valid=True problems=[]` | accepted | 1099/1099 |

三份 manifest 结构一致：`shards 1 / episodes 1 / decisions 1099 / sample_count 1099（顶级字段）/ observation_channels 6`，
`dataset_id` 各含自己的 seed。**三份 `move` 直方图互不相同**（`move=0` 分别 468/467/484），
加上三盘磁带 sha256 互不相同，双重证明它们是**三段不同的轨迹**。

`o2-strided` 是**首份 manifest 记账完全正确**的语料：`counts.decisions == 样本数`、`episodes=1`、`shards=1`、
ratio **1.0x**、`three_way_equal=true`，无需 `sample_count` 消歧。其 1099 与主代理黄金参照
`schedule[::6][1:]` **逐项相同**，帧间隔恒 12、帧与 cand 序列 **0 处不符** → 确证是
**filter-during-replay**（全程重放 6598 个决策、只保存每第 6 个），不是会把轨迹改写的 filter-tape。

## 4 我独立发现并已登记的问题（不影响 G430-E0，但影响后续）

| # | 问题 | 证据 |
| --- | --- | --- |
| 1 | **换行符口径污染**：`review_checks.json` 的 101 条源码 SHA 中 96 条仅差 CRLF/LF，0 条无法解释 | `evidence/W00/SHA_LINE_ENDING_FINDING.md` |
| 2 | **3458 份历史文档被删（未提交）**，导致 v4.3.0 自带复核器 `verify_plan.py`/`verify_review.py` 在本机 exit 1 | 同上 §2 |
| 3 | **`ASSET_SYNC.sha256` 1 条过期**（`v4.2.0/journal.jsonl`，活文件在清单生成后追加） | `evidence/W00/ASSET_LEDGER_CHECK.md` |
| 4 | **`.tmp_v410_s1b`（2.03 GB）在 `ASSET_SYNC` 中零条目** | `evidence/ASSET_LEDGER_DISCREPANCIES.md` |
| 5 | **`s1c` 规模被高估**：声称 66,143 decisions，实际约 39,600 | 同上 |
| 6 | **`61300–61399` 已消耗**（G410-1 100/100）但 `EXECUTION_PLAN §6` 未提 | 同上（W09 已标 `spent`） |
| 7 | **P1-07 主体未闭合**：187 个被点名的 `.tmp_v420_*/` 脚本仍不可同步 | `evidence/W00/REVIEWER_VERDICT.md §4bis` |
| 8 | 一个 v4.2.0 A1 扩容仍在跑（553/750，1.5 状态/分钟，预计还需约 2.2 小时），使每次引擎运行慢约 3–4% | `evidence/W00/RUNTIME_PROBE.md §4` |
| 9 | **源码跟踪缺口比 P1-07 描述的更大**：`git ls-files` 显示只有 `touhou_rl/v400` 被跟踪（71 文件）；v380(23)/v390(5)/v410(24)/v420(14)/v430(12) 与 `configs/v410|v420|v430` **全部未跟踪**。全新 clone 只会拿到 v400 代码 | 见 §5 第 1 行的实测分解 |
| 10 | **退出码有两套口径并存**：同一个"有效测量未达门"的情形，`lineage.exit_code_for()` 返回 **2**，而 CLI `validate-splits` 按 `EXECUTION_PLAN §4` 返回 **3**。两者都被写进结果（`lineage_exit_code` / `exit_code`） | `evidence/W09/W09B.md`。**需在 W12 统一**；在此之前，对外契约以 CLI 的 3 为准（它符合计划表），纯函数口径只作内部信号 |
| 11 | **既有 29 GB 语料不能给 O2 模型用**：存的是 v4.2.0 降采样观测。`global_rgb` **完全缺失**；play 同为 BT.601 但 **4×4 块平均**（O2 是 2×2）；hud **不是降采样而是窄带取样**（`HUD_BANDS` 只取 56 行，O2 要 448 行，**392 行 legacy 从未采集**）；history 18 vs 7 特征。**计划三个文件对此零提及** | `evidence/OBS_CONTRACT_GAP.md`（主代理实测数组头 + 读 `obs_spec_v3` 源码）。危险点：`o2_digest()` 只能查契约版本，查不出"形状对、内容降采样" |
| 12 | **`W02A.md` 记录的 `encoder_digest` 已过期**：W02a 时是 `4cc3c6cd…`，W03I 加入 `alpha_mode` 后 strict 变为 **`37ebb8d4…`**、any 为 `90e43af6…`。**按 ACID 不改旧证据**，此条仅作交叉引用 | `evidence/W03/W03I_ALPHA_MODE.md`。`o2_digest()` 仍是 `cb1616e4…`（未变，已实测 `O.o2_digest()==C.o2_digest()`）；run 级身份改用 `encoder_digest()` 区分 strict/any |
| 13 | **两个东西都叫 `obs_digest`，很容易误用**：`contracts.o2_digest()` = **编码器规格**摘要（`cb1616e4…`）；manifest 的 `obs_digest` = **观测数据**摘要（本次实测 `3db65809…`，`binary/raw`）。我实测把前者当 `expected_obs_digest` 传进去 → **`valid=False`、`P_OBS_DIGEST_MISMATCH`** | 主代理实测。**建议**：manifest 字段改名 `obs_data_digest`，或校验器另加 `expected_encoder_digest` 参数；在此之前调用方须知二者不可互换 |
| 14 | **CLI 的 reserved notice 文案已陈旧**：`train` 的 notice 写 `"implemented_here": ["audit"]`，但**实测已实现 4 个**（`audit` / `validate-contract` / `validate-splits` / `train-bc`）。`--help` 亦仍写 "audit only" | 主代理实测（`train` → exit 2 且报文含该字段）。**属误导性文案**，非功能缺陷；归 W12 统一清理 |
| 15 | ~~**W03f 的观测契约守卫没有被训练路径调用**~~ → **已修复（W04B）**：`learn.train_from_manifest` 现在在 gate 1b **无条件**调 `assert_o2_compatible`，`run=False` 干跑路径也调；`ManifestError` **不包装**（保持 manifest 层类型契约，调用侧需同时捕获两种类型）。附带修掉 `batch_source` 硬编码：改为调用方声明（`synthetic`/`captured_o2`），并以 `batch_source_verified: false` + note 明示"未经验证" | 主代理实测三条路径：`run=True`+legacy → `ManifestError`；`run=False`+legacy → `ManifestError`；`batch_source` 三态（`synthetic`→`synthetic_data_only=True`、`captured_o2`→`False`、`real`→`LearnError`）全部正确。**注意**：W04B 的测试用合成 fixture，**真实 64 样本语料上的全链路是我独立跑的**（`learn.py` 真实 manifest → 通过；legacy 改写 → 拒绝） |
| 16 | **`ppo.py` 与 `sequence.py` 是孤岛——仓库里没有任何"学习者"把它们接起来**：`ppo` 的非测试调用方为 **0**；`sequence`（GAE/分段）也没有被 `learn.py` 使用。现存的 `train_bc` 只做 BC，不做 rollout/GAE/PPO 更新 | 主代理接线审计（305 个文件，按 import 图统计）。**这不是缺陷而是进度事实**：W04A 的切片本就限定"纯函数"。但必须写明——**在 W04C 之前，仓库里不存在端到端学习者**，任何"PPO 已就绪"的说法都不成立 |
| 17 | **"取前 N 个决策"是错误的数据设计**：主代理只用磁带（未花引擎时间）算了覆盖率——`move` 去重数在 **first 64 → 5/9**、**first 1000 → 后 6/9**，而**整盘（6596）才是 9/9**、`cand` 17/18。磁带行为**前段高度集中、后段才分化** | 主代理实测（`predict_coverage.py`，仅读 `slots.jsonl`）。**结论**：按顺序截取前 1000 与截取前 64 差别很小；要覆盖度必须**均匀跨整盘抽样**（例如每 6 个决策取 1 个 ≈ 1100 样本），或直接全采。另：`bomb` 在**整盘 6588 行中全为 0**，再次确认该仪器从不放弹。**预测已被实测验证**：1000 样本实测 `move` 6/9、缺 `[1,5,8]`、`move=4` 恰 3 次，与预测逐项吻合 |
| 18 | **`counts.decisions` 的单位歧义**（原记"放大 8 倍"）→ **三批语料现在有三种记账状态，W12 必须统一**：见下表 | 主代理实测三方：我要求"改成 1000"→ W03K 实测会 `P_COUNTS_MISMATCH`（在 8-shard 模型下）；我进而断言"须改 `touhou_rl/**`"→ **错**，W03L 在生成器里换模型即解决。**结论**：修复不需要动库，但必须统一模型，否则 `decisions` 不可比 |

### 4.1 三批语料的 manifest 记账现状（主代理实测）

| 语料 | `n_shards` | `counts.decisions` | `episodes` | `sample_count` | `observation_channels` | 真实行数 |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `o2-64` | 8 | **512** | 8 | **缺字段** | **缺字段** | **64** |
| `o2-1k` | 8 | 8000 | 8 | 1000 | 8 | 1000 |
| `o2-strided` | 待采（预期 1） | — | — | — | — | — |

**`o2-64` 是三者中最难读的一份**：`counts.decisions=512` 而真实 64，且**没有 `sample_count` 字段**——读者只能靠加载 `.npy` 才知道真实样本数。它被 W03K 刻意冻结（我当时明令不得改 `o2-64`），所以未被 W03K 的加法修正覆盖。

**W12 的收口动作（两条）**：
1. 对 `o2-64` 施加**纯加法**修正（补 `sample_count`/`observation_channels`/`counts_note`，`counts.decisions` 保持 512 以免 `valid=False`）；
2. 三批统一到**一种**模型（建议 1-shard 模型：`counts.decisions` 直接等于样本数），并把 `o2-1k` 一并迁移；迁移后须重跑 `validate_manifest` 与 `assert_o2_compatible`。
| 19 | **`o2-1k` 的三个身份字段仍带 `-64` 旧身份**：`manifest.dataset_id = "w03h-o2-capture-seed12-64"`、`bc_smoke` 的 disclaimer 写"**64 real decisions**"、`assert_o2_compatible.returned_dataset` 同 | 主代理实测。`n_samples` 却是 **1000**。**dataset_id 是身份字段**，用错会让后人把 1000 样本语料与 64 样本语料混淆。需修：`dataset_id`/disclaimer 由实际样本数推导，不得硬编码。**已由 W03K 修复并经主代理复核**：`dataset_id=…seed12-1000`、disclaimer="1000 real decisions"、`returned_dataset=…-1000`、`manifest_sha256` 在 **text/lf-normalized** 口径下完全匹配（raw 口径不匹配是正常的，主代理曾误报） |
| 20 | **BC 在真实 O2 语料上不优于"恒预测多数类"这一平凡基线——三组配置全部失败，成因未定** | 主代理实测三组（`move` 头，seed12 训练 880 / 验证 219，基线 CE = 0.8537）：<br>① `ActorPolicy` 2,726,671 参数全分辨率：train **0.0293** → val **1.8807**、seed10 **1.8837**；<br>② 小模型 30,793 参数 / 16× 下采样（952 维）：val **1.5306**、seed10 **1.6020**；<br>③ 小模型 487,753 参数 / 4× 下采样（10,752 维）：train **0.1558** → val **2.1299**、seed10 **2.1161**。<br>**无一配置优于基线**；典型形态是 train CE 塌向 0 而 val CE 升至基线之上。**成因未定**：数据量 / 训练步数 / 架构不适配 / 标签可预测性有限，四条未区分（主代理曾断言"容量是主因"，缩模型后未获支持，已降级为假设）。详见 `W03_DATA_LEG.md §8` |
| 21 | **三盘通关磁带彼此共享约 20% 的完全相同开头——跨磁带验证若不排除，会把"记住开局"误读成"迁移"** | 主代理实测（磁带层，逐位置比较 `executed_cand`）：seed10–seed11 共同前缀 **1308**、seed10–seed12 **1308**、seed11–seed12 **1344**，折合 2616–2688 帧（约 44 秒），占较短磁带 19.8%–20.4%；全盘一致率 44.8%–46.3%。**与 stride-6 语料层的 219 相互印证**（1308÷6=218，加相位差 = 219）。剔除污染后 seed10 后缀测试 CE 由 2.1161 **恶化到 2.5070**（基线 0.8293）→ 污染是**偏乐观**的。**方法学要求**：跨磁带验证前把两条序列切到 `[最长共同前缀:]`；W10 holdout 开工前必须先生效。数据：`.tmp_v430_triage/tape_overlap.json` |

### 4.1 `git add` 的精确分解（实测，非估算）

`git status --untracked-files=all` 展开后共 **418** 个未跟踪文件；其中**应加入**的是 **209 个 / ≈5.32 MiB**：

| 来源 | 文件 | 字节 |
| --- | ---: | ---: |
| `touhou_rl/v380` | 23 | 195,330 |
| `touhou_rl/v390` | 5 | 74,446 |
| `touhou_rl/v410` | 24 | 1,092,650 |
| `touhou_rl/v420` | 14 | 1,327,322 |
| `touhou_rl/v430` | 12 | 445,010 |
| `configs/v410` / `v420` / `v430` | 5 / 26 / 3 | 15,167 / 295,093 / 157,288 |
| 路线产物（`tools/rl/v42x|v430`、`tests/test_v42x|v43x`、`evidence/v4.3.0`） | 97 | 1,973,390 |
| **合计** | **209** | **5,575,696（≈5.32 MiB）** |

**另外 209 个未跟踪文件必须排除**（`.tmp_*` 根级脚本、`.claude/`、`.dsh-acl-recovery/`、一个名字就是
`%sn  ; done` 的垃圾文件等，共约 45 MiB）。因此 **`git add -A` 绝对不可用**。

交叉校验：`touhou_rl/v420` 实测 **14 文件 / 1,327,322 B**，与 W00 源码身份那一半独立测得的
"14 个 .py / 1,327,322 B"**逐字节相同**——两个独立统计口径互相印证。

## 5 待用户决定（阻塞项）

| # | 决定 | 影响 |
| --- | --- | --- |
| 1 | 是否按**显式路径** `git add`：实测需 add **209 个文件 / ≈5.32 MiB**（见下方分解），而 `git add -A` 会额外扫入另外 **209 个**临时/杂项文件 | 不 add 则"源码不可同步"的缺口仍在。**新发现：只有 `touhou_rl/v400` 在 Git 里（71 文件），v380/v390/v410/v420/v430 与整个 `configs/` 树全部未被跟踪** |
| 2 | 换行符规范化方案：`.gitattributes` / 规格层规则 / 维持现状 | 只影响历史证据能否重算；v430 新代码已自带正确口径 |
| 3 | 那个 11 小时的 v4.2.0 扩容是否让它跑完 | 建议让它跑完（已 72%，杀掉等于白烧前功） |
| 4 | `ASSET_SYNC` 重钉策略（含 journal 1 条 + s1b 补登 + s1c 数字更正） | 归 W12 清单任务；需先定策略 |
| 5 | **O2 观测缺口怎么走**：既有语料是 v4.2.0 降采样（见 §4 第 11 条），三条路——**(a)** 用 v430 采集通路重新收真实 O2（每局约 130 s 引擎时间，干净但慢）、**(b)** 上采样+编造 `global_rgb`（便宜但训练/部署不一致，**不得冒充 O2**）、**(c)** 给 O2 加显式 legacy 模式并如实登记 | 决定 W03 数据腿与 W04 能否开工。**未定期间我按 (c) 推进**：只登记与拒绝，不做任何转换、不编造通道 |

## 6 未做的（不得声称）

- 未启动任何训练、未跑任何策略评估、**0 次真实过面由 v430 模型达成**。
- 未验证 Hard/Lunatic 留出家族的能力区间（可能全臂 0%，届时结论只能是"证据不足"）。
- 未测 Windows 侧端到端时延。
- 未做第二作 STRICT 任何核查。
