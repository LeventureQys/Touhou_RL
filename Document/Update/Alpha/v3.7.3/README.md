# v3.7.3 当前执行状态

更新：2026-11-02。已开始实施与GPU工程优化，尚未进入正式真实像素战术训练。**2026-11-02 补：P0 已在帧栅栏变体上重跑并通过（round26），像素—状态同帧从 `unknown` 改为"有条件 exact"，见下。**

| 阶段 | 当前事实 | 判定 |
|---|---|---|
| P0 | 输入通道已修复并实测发送/释放/四边界；**round26：在变体镜像上 10 条判据全过**（ticket/logic 逐帧 +1、发送→掩码 1 帧、掩码→位移 1 帧、四类动力学各 20/20、四边界、像素 780/780 检查） | **同帧性解冻（仅变体镜像 + 环境指纹）**；canonical/GDI 仍 unknown；P3/P6/P7 的"同帧"这一条阻断解除 |
| P1 | 有效课程 28 例（六基线全 0、参考 28/28）；旧集仅 8/32 过硬判据且都能被单动作常量过 | 可用；F2 仅 4 例，多样性不足 |
| P2 | 有效世界上：旧教师 14/28（F1 0/12 硬编码几何）、V2 28/28，**0 次退化成功**；train7组/val7组冻结 | 教师按族量化完成；V2 的 F1 成功依赖被告知 gap，非战术理解 |
| P3 | 新增像素/历史/标签资产SHA、物理隔离与因果审计；**R28 特权审计：28 个世界里只有 F2（4 例）同时满足"教师不读世界参数"与"对象在真机存在"；F1/F3 的标签在真实会话里不存在** | 数据门 pending：合法标签源仅 F2 4 例，需扩 F2/F5 类课程或收窄范围（新预注册） |
| P4 | R25 规模对照：F1 训练组 6→12（24 事件），新增留出"需向左"21 格翻转 **0/21**，"需向右"21/21 保持；**R27 像素信息探针**：预告渲染上方向可直接读出（12/12 + 480/480），遮挡后 0/492；小 CNN 只训 base→镜像 0.860（<0.90），两朝向混合→组留出 0.896–0.917 | 规模不是主因；**"信息没画出来"这一解释已被排除**，卡点改为"目标没有要求做缺口相对自机的比较"，下一步转向关系型读出/目标 |
| P5 | 旧闭环/补采/加权对照全部含常量可解例，聚合结论作废 | 保留的唯一真动作证据：R11 F2 上 N1 1/4 |
| P6—P7 | 未执行 | pending（阻断理由已从"P0 同帧性"变为"P3 无预注册数据集"） |

## 硬件结论

RTX 4070 SUPER 12GB + i5-14600KF +32GB内存足够开展计划的小型冻结骨干/GRU训练。当前瓶颈是玩法任务、标签和真实像素同步，不是模型容量或显存。以批量特征编码、缓存和合理CPU并行利用硬件，不扩大无效训练。

## 已执行验证

- **round26 P0 重判（真机）**：10/10 判据通过；采样器 A/B 吞吐 3.8→45.2 张/s；单元测试 `touhou_rl/tests/test_p0_fence_round26.py` + `test_framefence.py` 共 **36 passed**（不启动游戏）。
- **round27 像素信息探针（离线）**：直读器 12/12 + 480/480、遮挡 0/492；单元测试 `touhou_rl/tests/test_pixel_info.py` **7 passed**。
- **round28 特权审计（离线）**：旧教师 14/28、V2 28/28 复现；F1/F3 在参数喂错时 0/24；仅 F2 4/4 通过两种模式；单元测试 `test_privilege_audit.py` **5 passed**。
- **P3 接入规则（离线）**：`frame_proof` schema 与 `test_frame_proof.py` **16 passed**；预注册与接入方案已交付。
- 共同执行管线：48 passed。
- 新策略/runner/训练门：25 passed。
- 合成场景/教师：11 passed（R2最终针对测试）；最终全部新tactical测试36 passed，1.64s。
- 最新全量（touhou_rl/tests + tests/）：**1958 passed，1 xfailed，4 subtests passed，126.72s**。
- N1真实GPU人造特征优化100步：38.81步/s，骨干严格不变；不能作为玩法改善证据。

## 报告入口

- **[P0 栅栏重判：10 条判据全过、解冻规则 v2](<reports/P0-fence-round26.md>)**（2026-11-02，含三次未完成执行的披露）
- **[F1 方向信息在像素里：可读、可学，但不是"再加预告"能解决](<reports/P1-pixel-info-round27.md>)**（2026-11-02，含两次自证失败的披露）
- **[谁能在真实会话里产出标签：28 个有效世界的特权审计](<reports/P2-privilege-round28.md>)**（2026-11-02，只有 F2 4 例合格）
- **[P3 接入方案：把 frame_proof=exact 接进 capture_dataset](<reports/P3-exact-integration-plan.md>)**（2026-11-02，含[预注册](<preregistration/p3-exact-capture-v1.md>)与可执行 schema）
- [硬件与P0](<reports/P0-clock.md>)
- [合成学习线收敛结论与下一步](<reports/synthesis-round25.md>)
- [F1 规模对照：规模不是主因](<reports/P4-f1scale-round25.md>)
- [玩家相对标签：improved 只发生在 base 组](<reports/P4-prelaux-round24.md>)
- [特权辅助任务：学得到但用反](<reports/P4-privaux-round23.md>)
- [空间融合无效：信息在表征里，行为不用它](<reports/P4-spatialfuse-round22.md>)
- [危险预告修复失败：方向先验而非信息缺失](<reports/P4-telegraph-round21.md>)
- [F1 不可学：缺口在致命前从未被渲染](<reports/P1-f1-visibility-round20.md>)
- [有效课程蒸馏与留出闭环（负结果）](<reports/P4-validcurriculum-round20.md>)
- [有效世界上的教师评估与 7/7 分组冻结](<reports/P2-validteacher-round19.md>)
- [非常量可解课程重建（28 例，含撤回披露）](<reports/P1-worldfilter-round18.md>)
- [KL 锚定：过滤后世界的首个方向性信号](<reports/P4-kl-anchor-round17.md>)
- [常量探针：旧世界大量可被单一动作通过](<reports/P4-degenerate-round17.md>)
- [能力台账：每条声称的状态与证据](<reports/ledger-round17.md>)
- [多种子与新世界确认（B0 仍最强）](<reports/P4-seed-confirm-round16.md>)
- [P0出口判定与冻结规则](<reports/P0-exit-round15.md>)
- [单因素加权动作监督（单seed，F2退化）](<reports/P4-weighted-round15.md>)
- [三臂fresh等预算对照](<reports/P4-arms-round14.md>)
- [关键移动标签召回为0的诊断](<reports/P4-recall-round14.md>)
- [首轮补采＋等预算训练负结果](<reports/P5-dagger-training-round13.md>)
- [仅训练来源失败状态采集](<reports/P5-dagger-data-round13.md>)
- [闭环失败独立重放与恢复见证](<reports/P5-forensics-round12.md>)
- [精确因果输入别名/标签冲突](<reports/P3-alias-round12.md>)
- [合成闭环负结果：N1与B0均2/8](<reports/P5-synthetic-round11.md>)
- [离线频率/历史/视觉控制](<reports/P4-controls-round11.md>)
- [合成像素首次GPU学习实测](<reports/P4-synthetic-learning-round10.md>)
- [合成渲染数据与因果审计](<reports/P3-synthetic-diagnostic-round10.md>)
- [几何教师v2修复与新确认](<reports/P2-revision-round9.md>)
- [事件级离线指标接口](<reports/P4-metrics-round9.md>)
- [F1/F2/F3冻结教师留出与延迟见证窗口](<reports/P2-validation-round8.md>)
- [左/上/下边界与旧动作保持](<reports/P0-boundaries-round8.md>)
- [逐图候选frame与几何误差集合](<reports/P0-alignment-round7.md>)
- [冻结bank与独立像素确认](<reports/P0-pixel-bank-round6.md>)
- [确认采集与重复像素跨split拒绝](<reports/P0-confirmation-round6.md>)
- [计算协议绑定修复cache /2](<reports/P4-cache-contract-round7.md>)
- [真实v3-st跨设备数值差异与同协议重复](<reports/P4-numeric-round7.md>)
- [CPU/CUDA批量特征缓存构建器](<reports/P4-feature-cache-round6.md>)
- [真实像素定位与同步unknown结论](<reports/P0-pixels-round5.md>)
- [连续真实像素采样](<reports/P0-capture-round5.md>)
- [两帧动作、释放与右边界实测](<reports/P0-motion-round4.md>)
- [截图新鲜度源码审查与lag估计器](<reports/P0-capture-round4.md>)
- [修复后真机核验与因果数据审计](<reports/P0-round2.md>)
- [输入读回纠错与第二次真机分析](<reports/P0-input-readback.md>)
- [结构模板冻结验证](<reports/P1-template-round3.md>)
- [真实数据资产SHA审计强化](<reports/P3-assets-round3.md>)
- [F4/F5规则教师开发复验](<reports/P2-task-teacher-round2.md>)
- [F4/F5带目标任务构造](<reports/P1-tasks-round1.md>)
- [场景/教师与失败](<reports/P1-P2-tactical-development.md>)
- [受门禁约束序列训练接口](<reports/P4-sequence-interface-round3.md>)
- [网络工程与GPU诊断](<reports/P4-learning.md>)
- [教学过程记录](<../../../Dialogs/2026-11-01_v3.7.3_实施与硬件训练审计.md>)

## 下一任务卡

1. **P0 已重判并通过（round26）**：像素派生几何标签在**帧栅栏变体 + 13 项检查 + 哈希 + 环境指纹**下可为 `exact`；canonical/GDI 仍是 `alignment_unknown`。规则见 [round26 报告](<reports/P0-fence-round26.md#解冻规则-v2可执行版>)、[P0 出口判定补记](<reports/P0-exit-round15.md#补记2026-11-02round26解除条件已满足冻结改为按条件解冻>)。
2. **接入规则已就绪（方案，未实现）**：`frame_proof=exact` 的[预注册](<preregistration/p3-exact-capture-v1.md>)、[接入方案](<reports/P3-exact-integration-plan.md>)与可执行 schema [frame_proof.py](<../../../../touhou_rl/frame_proof.py>)（16 单测）已交付；**`capture_dataset.py` 本体尚未改**。实现后第一件要测的是"回放模式下 60 帧 13 项全过"，再跑 3,000 帧试采集（E1–E10）。在此之前**不宣布任何 P3 数据集合格**。
3. **工程项（已完成并实测）**：`FenceFrameSampler` 每样张烧掉一个 250 ms hold 预算（round26 实测 `wait` 中位 235 ms、60 次预算耗尽）；新增 `accept_current=True` 消费当前 hold 后，同一场景同 60 张：**3.8 → 45.2 张/s（11.9×）、`wait` 中位 232.9 ms → 3.1 µs、预算耗尽 59 → 0**，13 项检查两边都是 780/780。默认行为未改，round26 正式数据由默认模式产生。
4. **评估集必须先有效**：R17 证明旧世界可被常量通过；R18 已产出 28 个"六基线全失败 + 参考有见证"的世界（F1 12 / F2 4 / F3 12）。F2 只有 2 组，需另立预注册、更大预算重造，不能放宽判据。
5. **P4 的下一个单因子已由 R27 定方向**：不要再加预告/加通道；做一个**关系型读出或目标**（直接回归 `gap_center − player_x`，而不是分类"屏幕左/右"），训练分布必须同时覆盖两个朝向与边缘位置，并在相同的冻结分组上留出闭环。预注册后才训练，3 seeds。
6. **P3 的标签源已被 R28 收窄到 F2（4 例）**：要么扩 F2/F5 这类"对象可观测"的课程到可用规模，要么收窄 P3 范围并明说只做 F2 类行为；把 F1/F3 教师轨迹当真机标签等于把世界参数偷偷带进监督。两条路都要新预注册。
7. **之后才训练**：在冻结划分上渲染教师轨迹 + KL 锚定，按组留出后测闭环；在此之前不再用旧 8/24 例的聚合数字做任何结论。
8. P1/P2 的教师与课程仍未达标项保持原状（见台账）；P6/P7 的阻断理由是 P3 未预注册，不再是"同帧性"。

默认模型和部署未改变。当前没有可发布的新玩法模型，不宣称死亡率改善或门禁通过。
