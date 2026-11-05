# v3.7.6：真实F2 BC准备与清理收口

- 当前状态：**WP-F 自查提出的四项不通过已逐项修复，冻结协议后训练已执行；训练结果是否定的（低于"保持上一动作"基线，也低于均匀随机）。**
- 决策：用户选择真实F2回放示范行为克隆（BC），已授权三独立replay研究变体补采；画面复核另交接。长训练、默认模型替换及发布未授权。
- 规格来源：[v3.7.5](<../v3.7.5/README.md>)、[原验收](<../v3.7.5/验收标准.md>)；本轮新增路线以 [训练修订](<训练修订.md>) 为准，**BC不是原U05/G4关系目标实验**，不静默改旧规格。

## 1. 当前事实与边界

1. 报告契约修复、F2标签/数据集、readiness、训练gate、缓存/序列训练复用、同hold完整活动弹池审计及采集身份绑定已经实现。代码可用不等于真实准入通过，最终日志/基线仍待统一登记。
2. 新候选三 replay 各 3000 帧，共 **9000/9000 有效、各 0 丢弃、13 检查全过**；耗时 124.9/124.6/124.9 秒，三组 deep 审计均成功。[统一采集报告](<reports/f2-capture-20261104-r1/report.json>)、[完整命令](<evidence/f2-capture-20261104-r1/commands.jsonl>)。最终待复核联系表：[train 43 个候选](<evidence/f2-review-20261104-r3/train/index.html>)、[validation 45 个](<evidence/f2-review-20261104-r3/validation/index.html>)、[test 45 个](<evidence/f2-review-20261104-r3/test/index.html>)；它们仍是候选，不是已认证 F2 训练集。后续见 [真实 F2 复核交接](<真实F2复核交接.md>)。
3. F2为瞄准弹流，没有固有gap。gap_center、relative_dx、deadline、route等关系/规划标签必须不适用、null且关闭mask；示范动作只称演示，不称最优/安全动作。
4. 至少三独立来源组只是必要条件；仍需独立画面review、合格事件覆盖、回放对齐、物理隔离、分组及逐帧像素泄漏审计。当前 `real_training_ready=false`，不能写G3/real passed。
5. 默认模型、canonical镜像及原资源目标镜像保持不变。研究变体采集授权不等于部署或新策略控制授权。

## 2. 执行路线与门状态

| 阶段 | 当前状态 | 下一步/门边界 |
|---|---|---|
| A/WP0 | in_progress | 已补 [有限历史分类](<历史结果分类.md>)，未遍历全部run；新baseline待齐，G0未通过 |
| B/WP3 | in_progress | 报告与F2标签契约已有实现；原六CLI、完整proposal/receipt/trace及P4–P7指标未因此完成 |
| C/WP1 | 历史G1/G2保留；新采集已产出候选 | 旧G1四工件漂移，新代码需新engineering证据；新三来源须deep审与review |
| D/WP2 | in_progress；可观测性验收blocked | **D1+B3联合冻结**，不再循环等待；画面review另交接，合格事件plan待齐 |
| BC新增分支 | 真实准入blocked | build → audit → check-gate → cache → 一更新；只有前置齐备才执行 |
| 原E/U05/G4 | not_run | gap目标对当前F2不适用；另行重设计，BC不代验原8事件/四臂/三seed |
| 原F/U06/G5 | not_started | 原事务/闭环尚无本轮完整验收 |
| 原G/U07/G6/G7 | 未完成，真机控制blocked | 需协议/统计/授权，补采不是确认实验 |
| U08/G8 | out_of_scope | 本版本不做跨作品 |

唯一逐任务状态及计数见 [任务清单](<任务清单.md>)；详细依赖、停止条件见 [执行计划](<执行计划.md>)。

## 3. 历史指纹与当前基线

### 3.1 历史快照（historical，不作为当前实现身份）

原登记日期2026-11-05予以保留为历史记录，不据此推断本轮运行时间；本轮新run使用20261104标识。以下SHA/测试数属于当时快照，代码继续变化后不能沿用为当前基线。

| 项目 | historical记录 | 限制 |
|---|---|---|
| HEAD | `a45589a0e9ebd6821e840c3750071ce0eabe678c` | 不能标识未提交/未跟踪文件及当前dirty内容 |
| 全量回归 | 2356 passed、1 xfailed、4 subtests，144s | 历史，不是本轮最终数 |
| 报告契约及测试 | `8ef8b4c3…5427` / `d7ac3a81…4191`，38项 | 已发生修复，不再是当前代码指纹 |
| scope配置 | `b30ba84f…c6deb`，real_game_available=true、status=active | 旧登记；当前SHA以新baseline为准 |
| exact配置 | `0dd084c5…04e2e` / `c69e3eaf…d4ec0` | 对应旧G2，不自动适用于新完整审计采集配置 |
| [旧G1报告](<../v3.7.5/reports/wp1-v375-20261105/report.json>) | `f2461e29…e5e8f` | 历史engineering passed；capture_dataset、exact_capture、test_exact_capture、exact-capture-v1四工件相对报告漂移，不重写旧报告 |
| [旧G2报告](<../v3.7.5/reports/wp1-g2-v375-20261105/report.json>) | `3a0ae88e…308f8` | 当次3000/3000、0丢弃、24.29帧/s；单replay无F2标签，不能充当独立三划分 |
| 旧G2数据proof/manifest/proofs | `05d5d24c…0dc8` / `a3bee298…bad05` / `cb8f6be6…4242` | 对应旧数据，不是新三来源数据摘要 |

### 3.2 当前证据与待补项

- [开工readiness快照](<reports/readiness-20261104-r1/readiness.json>)：旧G2深审可用、采集可开工，但真实训练不就绪；它是开工快照，不替代新增实现状态或最终数据审计。
- 最终锁定源码全量：**2658 passed、2 skipped、1 xfailed、4 subtests，202.39 秒，exit 0**；[原始日志](<reports/regression-20261104-r3/pytest.log>)、[命令及退出码](<reports/regression-20261104-r3/command.json>)。两个 skip 是 Windows symlink 权限用例；确定性 resolve 路径逃逸负例已执行。随后新增基线工具的独立测试 **1 passed**，不并入前述全量计数。工程夹具端到端四项已包含于全量：真实 materialize/deep audit/gate/cache/prepare/Adam 一更新均执行，仅替换 tiny backbone；不是人工真实画面复核或 G3 通过。
- [最终源码基线 r3](<reports/baseline-20261104-r3/baseline.json>)：source snapshot SHA `2b2fae8227ecdae3e448b6edc8f69f0f2e5828c4e7e9d999e6ee6f084c340b43`，含未跟踪源码、解释器/依赖、训练基线模型、三套游戏镜像和地址表身份。r1/r2 是中间快照；文档后续结算不改变源码快照所列文件字节。默认部署选择没有从训练基线路径推断，详见基线 limitations。
- train 的 stageFrame 5451 重复 55 行，validation 的 5706 重复 94 行，test 无重复。dataset/reviewpack 已实现**整组歧义帧排除**并持久化诊断，任何当前/历史引用均拒绝，不取 first/last、不改 raw、不降级 exact。[最终复核交接](<真实F2复核交接.md>)。
- 中间 full-r2：1 failed、2654 passed；失败是测试构建 cache 后源码仍在并发修改，触发 `cache compute protocol/source mismatch`。保护生效，不放宽来源哈希；锁定源码后重新运行最终 full-r3。早期 reviewpack 也因 float 帧号及真实 stage-frame 暂停失败，均已补回归修复，失败 run 不覆盖。
- 最终真实训练状态：**blocked / pending_visual_review**。没有合法review、plan及内容审计，不生成真实训练checkpoint或虚构通过。
- **WP-A 合同与动作对齐审查（2026-11-05）**：[报告](<reports/wpa-contract-review-20261105/review.md>)。发现并修复 P0——回放记录索引原被当作 `stageFrames`，而回放游标实为 `logicFrame`（`stageActiveFrames` 在对白期间停走，游标照旧每逻辑帧消费一条记录）。用游戏自己发布的 `gui_input_current`/`gui_input_previous` 逐帧复验：**train/validation/test 各 3000/3000 命中**，自机速度方向与已消费记录 100% 一致（与下一条记录仅 92%）；train 对白后滞后 **149** 记录（≈2.5 s）、validation **393**（≈6.5 s），test 因无对白不受影响——这解释了为什么该错误此前未被任何测试发现。修复把标签索引、历史索引与行时间基准改为逻辑时钟，并在 build/audit 时逐行复验消费输入；[对齐证据](<reports/wpa-contract-review-20261105/alignment-probe.json>)。重跑全量 **2662 passed / 2 skipped / 1 xfailed / 4 subtests，207.34s，exit 0**：[日志](<reports/regression-20261105-wpa/pytest.log>)。
- **WP-B 真实 F2 视觉复核（第一轮，未完成）**：[报告](<reports/wpb-pre-screen-20261105/review.md>)。133/133 候选已按特权弹池几何排名（[pre-screen.json](<reports/wpb-pre-screen-20261105/pre-screen.json>)），目视检查 3/133，**0 个可确认 F2**；三份冻结来源都只覆盖 Stage 1，画面为环形/扇形/螺旋扩散而非瞄准弹流。按计划停止条件报**覆盖不足并申请定向补采**，不用放宽容忍度凑数。
- **Stage 2 定向补采 → 冻结计划 → 数据集 → 一次真实更新（2026-11-05）**：[报告](<reports/f2-pipeline-20261105-r2/review.md>)。同一三份 replay 身份、换 decoder stage 1（TH08 Stage 2）重采：**3×3000/3000、discarded=0、deep 审计通过、lag=0**（[深审](<reports/f2-capture-20261105-r2-deep-audit.json>)）。新窗口出现「新弹 100% 指向自机」（train 63/63、validation 42/42、test 69/69），目视可见成束白色箭头弹朝自机推进 → 每来源选 1 个 48 行窗口。复核由 AI 审查者**逐像素测量并如实署名**（`reviewer.id=v376-wp-b-agent-visual-20261105`，用户授权该签署链）：3 事件共 8 条标注行、每行 2 次同源瞄准发射，实测像素偏差 ≤3.49 px、容差由实测反推，其中 6 行产生有效动作监督（test 2 行未按射击 → unknown）。冻结计划（仓库根 `v376-f2-bc-plan-20261105-r2.json`）→ **build / audit / check-gate 全 exit 0** → GPU 缓存 + **一次真实更新成功**（`updates=1`、loss 2.7521、监督 `action=3`、validation_loss 3.2339、骨干不变、`test_used=false`、checkpoint 可读回），[训练报告](<reports/f2-pipeline-20261105-r2/train-report.json>)、[审计](<reports/f2-pipeline-20261105-r2/dataset-audit.json>)、[SHA 清单](<reports/f2-pipeline-20261105-r2/sha256.json>)。预注册更新为 [f2-bc-v2.json](<configs/f2-bc-v2.json>)。
- **仍未完成（不因上一条而消失）**：WP-F 独立放行审查（必须由未参与实现者执行）；数据集只有 3 事件 × 48 行，其中仅 6 行有效动作监督、4 个动作值、1 处动作变化，属集成样本而非统计基础；长训练、选模、闭环、真机、默认模型替换均未授权。BC-03 因此保持 `in_progress`；全量回归再跑 **2662 passed / 2 skipped / 1 xfailed / 4 subtests，211.35s，exit 0**（[日志](<reports/regression-20261105-wpb/pytest.log>)）。
- **WP-F 放行审查（用户指定由实现者自查，非独立）**：[自查报告](<reports/wp-f-self-audit-20261105/review.md>)、[机器读数](<reports/wp-f-self-audit-20261105/self-audit.json>)、[审计脚本](<../../../../tools/rl/v376_wpf_self_audit.py>)。从字节重算而非复述报告：身份/时钟（144/144）/标签（6/6）/像素（48/48，≤2.86 px）/隔离/优化均**通过**；瞄准由 teacher 几何支撑（漏距 ≤0.5 px，中位 0.09 px）。结论 **「允许有限诊断，不允许正式训练」**，并自曝四处缺口：① `validate_event_annotation` 的瞄准判据是速度空间角度检验、**证明不了瞄准**（我第一版审计脚本误当漏距算，得到 0/16 的假失败，已记录）；② 监督行按「可测量性」挑选，存在选择偏差；③ 逐事件常数基线已达 **83.3%**，6 行无法区分学习与常数；④ 评估协议未在看结果前冻结。签名/审查同源，独立性不成立。
- **第三轮：修掉上述四项 → 冻结协议 → 训练（2026-11-05，结果否定）**：[第三轮报告](<reports/f2-pipeline-20261105-r3/review.md>)。逐项处置：F-1 瞄准改为**命中半径漏距** `|cross|/|v_teacher| ≤ 1.5 px`（瞄准帧回看 ≤2、发射恢复窗 ≤64、每发申报并被合同与 `_verify_teacher` 两条独立路径重算）；F-5 改为**冻结窗口 + 逐行穷举**（窗口由特权几何在像素工作前定死，失败行如实记 `context_only` 不换行）；F-4 数据从 6 行扩到 **340 行有效监督**（9 事件 × 96 行、13 个动作值、49 处动作变化）；G-1 [预注册 v3](<configs/f2-preregistration-v3.json>) 与[评估协议 v1](<configs/f2-eval-protocol-v1.json>) 训练前入库（提交 `2cd5d0d`）；F-6 由**另写的、不 import 生产链**的脚本做双重实现复核（[audit.json](<reports/wp-f-independent-audit-20261105/audit.json>)：14 项 0 失败，时钟 856/856、标签 340/340、漏距最大 1.396 px、像素中位 0.50 px）。数据集 `…-dataset-20261105-r4`：build/audit/check-gate 全 exit 0。冻结预算训练实际 **120 次更新**（epochs 30、上限 1000），验证损失 epoch 2 最优 2.7589 后单调恶化至 5.4640。冻结指标：验证 **4.96%**、封存测试 **1.47%**，保持上一动作基线 **97.52% / 86.76%**、均匀随机 5.56%、逐事件宏平均 4.2% / 1.3%；像素消融（playfield 置零）准确率 **5.79% ≥ 4.96%**，说明模型完全未利用弹幕画面。**判 fail（无技能证据）**，模型权重仅作研究产物保留，不换默认模型、不作玩法改善声明。
- **仍未完成（不因第三轮而消失）**：外部执行者（非实现者、非同一开发者）的独立只读复核（F-6 按原文仍未关闭）；选窗规则带来的分布位移（按「存在瞄准弹流」选窗 → 训练段恰好是人类频繁变向的区段，而验证/测试以长按为主），需在下一轮重新预注册；数据量仍需上一个量级才能谈泛化；长训练在当前选模规则下无意义（120 次更新即已单调恶化）。原关系目标 G4、闭环 G5、真机 G6/G7 状态未变。

## 4. 本轮出口与停止条件

本轮可交付：实现及负例、三来源候选采集/深审、review交接资产、新baseline与最终工程回归。真实review未齐时，清楚停在blocked即为诚实交接，不承诺G3/G4/G5全部完成。

- G0：历史分类覆盖与基线仍未全部闭合，不假passed。
- BC：Stage 2 定向补采已完成；真实事件 plan 已冻结、build/audit/check-gate 已过、缓存与**一次真实更新**已跑通。第三轮把 WP-F 的四项不通过修掉并冻结协议后**执行了冻结预算训练**，结果否定（模型低于保持上一动作基线）——**训练"跑通"不等于"学到"**，本轮只交付后者不成立这一结论。
- 原G4：关系目标/四臂/三seed未执行，BC成功也不代替。
- 任何exact、身份、路径、像素分组或标签因果检查失败，停止并保留失败资产；unknown/unmapped/budget不变负例或hold。
- 不覆盖旧run，不看test后改阈值，不把测试成功或一次更新称作泛化、玩法改善或发布。

## 5. 历史解释与复用

[历史分类](<历史结果分类.md>)按有限范围登记passed/failed/void/not_run；[v3.8证据重置](<../v3.8.0/证据重置.md>) X4/X5约束本轮：关系目标唯一根因未证、旧F2 mirror代理审计不能直接证明真实监督合法。复用回放decode/align及候选映射经验，不引入PPO/critic/新actor，也不继承v3.8真机能力结论。

## 6. 导航

- [任务/计数](<任务清单.md>) · [执行计划](<执行计划.md>) · [训练修订与命令](<训练修订.md>) · [历史分类](<历史结果分类.md>)
- **后继工作分派**：[后续工作分派计划](<后续工作分派计划.md>)（WP-A～WP-F、判据、停止条件、可粘贴的执行约束） · [真实F2复核交接](<真实F2复核交接.md>)
- [BC冻结候选配置](<configs/f2-bc-v1.json>) · [新exact配置](<configs/f2-exact-capture-v1.json>)
- [原规格与验收](<../v3.7.5/验收标准.md>) · [旧G1](<../v3.7.5/reports/wp1-v375-20261105/report.json>) · [旧G2](<../v3.7.5/reports/wp1-g2-v375-20261105/report.json>)
