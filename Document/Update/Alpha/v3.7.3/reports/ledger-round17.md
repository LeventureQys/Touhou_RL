# v3.7.3 能力台账（claim ledger）

用途：把每一个"声称"与它的证据、状态放在同一张表里，避免用后续成功覆盖早先失败。状态只有五种：**passed / partial / failed / unknown / blocked**。本表不引入新实验；数字全部引用既有报告，可回溯到原始 JSON。

最后更新：2026-11-02（补入 round26 的 P0 重判；旧行未删除）。

## P0 可审计执行与同步

| 声称 | 状态 | 证据 | 备注 |
|---|---|---|---|
| 基座真实加载、架构/SHA 登记 | passed | [environment.json](<../evidence/environment.json>)、[source-sha256.json](<../evidence/source-sha256.json>) | v3-st，1,507,162 参数 |
| 两臂同 schema、同曝光、同动作保持（假环境） | passed | [P0-clock.md](<P0-clock.md>)、48 项 harness 回归 | 仅假环境 |
| 输入读回通道正确（engine current / GUI previous） | passed | [P0-input-readback.md](<P0-input-readback.md>)、非零 padding 单测 | 修复前 3625 个轮询恒 0 |
| 发送→生效为区间、释放可观察 | passed | [P0-motion-round4.md](<P0-motion-round4.md>)：20/20 动作、20/20 释放 | 首次匹配在发送后 1 逻辑帧 |
| 四边界 | partial | [P0-boundaries-round8.md](<P0-boundaries-round8.md>)：x[8,376]、y[16,432] | 每边界仅 1 次，非 20 次误差分布 |
| 每类 20 次独立动力学重复 | failed | — | 未做；稳定子集仅 9–36 个转移 |
| 截图与状态精确同帧 | **unknown（已冻结）** | [P0-capture-round4.md](<P0-capture-round4.md>)、[P0-alignment-round7.md](<P0-alignment-round7.md>) | GDI 无 Present fence；289 图中 223 unknown |
| 处理规则冻结 | passed | [P0-exit-round15.md](<P0-exit-round15.md>) | 像素派生几何标签禁入精确监督 |

## P0 重判（round26，帧栅栏变体上重跑）

本表不覆盖上面的旧行：旧行在"canonical + GDI"范围内仍然成立。下面是**在变体镜像 `build_fence/th08.exe` 上**的新证据（[P0-fence-round26](<P0-fence-round26.md>)），只对该镜像 + 记录的环境指纹成立。

| 声称 | 状态 | 证据 | 备注 |
|---|---|---|---|
| ticket/logicFrame 逐帧 +1 | passed（变体） | 239（demo）+ 7911（practice，4 局）次转移全为 +1 | 死亡/重开也不改变 1:1 |
| 呈现速率 | passed（变体） | 59.991 Hz（游戏侧 QPC，240 帧） | hold 仅读 telemetry 时中位 20 µs |
| 发送→掩码 = 1 帧 | passed（变体） | 80/80 试验，四类各 20 | 之前只能写"区间" |
| 掩码→位移 = 1 帧 | passed（变体） | 80/80，`lag_mv = 2 = lag_in+1` | 解释了旧"1 帧滞后"读数 |
| 每类 20 次独立动力学重复 | **passed**（变体） | 4.0 / 4.0 / 2.0 / 2.82843017578125，各 20/20 逐位相同 | 斜向相对 4/√2 偏差 1.08e-6 |
| 反向旧方向恰 1 帧 4 px | passed（变体） | 20/20 反向 + 10/10 切轴 | 旧规则首次有逐帧证据 |
| 四边界 | passed（变体，样本级） | 376 / 8 / 16 / 432，各 2 个连续静止帧 | 每方向仍只 1 次饱和，不是标定模型 |
| 截图与状态精确同帧 | **passed（有条件）** | 60/60 样张、780/780 检查、GDI == 后台缓冲 60/60 | 条件是变体镜像 + 13 项 + 哈希 + 环境指纹 |
| 旧读法仍然不可用（对照） | unknown（本轮太弱） | 60 次读取仅 2 次跨 ticket、0 摘要不一致；场景 0–5 发弹 | 靠 v3.7.4 的 43 发弹对照，不靠本轮 |
| `frame_proof=exact` 字段进入正式数据集 | **blocked** | `capture_dataset.py` 未改，无新预注册 | 解冻规则已给出，接入未做 |
| 精确采集吞吐 | passed（工程，变体） | 逐样张落盘：默认模式 3.8 张/s、`wait` 中位 232.9 ms、59/60 预算耗尽；`accept_current` 45.2 张/s（11.9×）、3.1 µs、0 耗尽，13 项 780/780 | 只改吞吐，不改 `exact` 条件；默认行为未改 |

## P1 任务与标签

| 声称 | 状态 | 证据 |
|---|---|---|
| 非常量可解课程（28 例） | passed（构造） | [P1-worldfilter-round18.md](<P1-worldfilter-round18.md>)：六基线全 0、参考 28/28 |
| 教师能否在有效世界上玩 | passed（合成）：旧 14/28、V2 28/28，退化成功 0 | [P2-validteacher-round19.md](<P2-validteacher-round19.md>) |
| V2 的 F1 成功是否等于战术理解 | **no**：直接被告知 `s.gap`，停在 `gap[0]+4` | 同上 |
| 旧教师 F1 | **failed**：0/12，硬编码缺口坐标 | 同上 |
| 旧世界能区分玩法 | **failed**：32 例里 28 例可被单动作常量通过 | 同上 |
| F2 课程数量 | partial：仅 2 组/4 例，预算所限未放宽判据 | 同上 |
| 六族可重放、F2/F3 发射依赖实际自机 | partial | [P1-P2-tactical-development.md](<P1-P2-tactical-development.md>) |
| F4 等待通过 / 立即撤离反例 | passed（合成） | [P1-tasks-round1.md](<P1-tasks-round1.md>) |
| F5 局部恢复（含位移约束） | passed（合成） | 同上 |
| 结构模板开发 24/24、双通道验证 24/24 | passed（合成） | [P1-template-round3.md](<P1-template-round3.md>) |
| 封存 test 12 条未执行 | passed | 同上，仅 manifest |
| 独立验证/测试模板 + 双人复核 | failed | 未做；镜像非独立 |

## P2 教师

| 声称 | 状态 | 证据 |
|---|---|---|
| 旧教师 R1 开发 42 例 | partial，含失败 | [P1-P2-tactical-development.md](<P1-P2-tactical-development.md>) |
| R2 必要换边窄任务 12/12 vs 对照 0/12 | passed（窄合成） | 同上 |
| 独立参数留出：旧教师 F1 仅 2/8 | **failed（保留）** | [P2-validation-round8.md](<P2-validation-round8.md>) |
| 几何 v2 修订：开发 42/42、新确认 24/24 | passed（合成修复） | [P2-revision-round9.md](<P2-revision-round9.md>) |
| 精确"最迟启动"上界 | unknown | 4536 条 delay 扫描，全部 `exact_latest_upper_bound=null` |
| 对照原视觉 B0 | failed | 未做 |

## P3 数据

| 声称 | 状态 | 证据 |
|---|---|---|
| 资产级 SHA、物理隔离、因果/泄漏审计 | passed（工具） | [P3-assets-round3.md](<P3-assets-round3.md>)、`tactical_data.py` |
| 本轮合格真实配对事件集 | **blocked（P0 冻结）→ 仅解除"同帧性"这一条** | 无；校准数据被审计拒绝。round26 给出了 exact 采集的解冻规则，但正式数据集仍无预注册 |
| 跨 split 重复像素拒绝 | passed | 同上 |
| F1/F3 教师能否产出"真实会话可用"的标签 | **no** | [P2-privilege-round28.md](<P2-privilege-round28.md>)：V2 读 `coverage`/`window`，参数喂错 24/24 全灭；F1 墙在原始渲染里致命前不可见，F3 耗尽窗口在真机无对应对象 |
| F2 教师是否 state-only 且对象真实存在 | **yes（4/4，三种模式一致）** | 同上：旧教师与 V2 在 masked 与 corrupted 下都 4/4 |
| P3 的合法标签源规模 | **partial**：仅 F2，4 例 / 2 组 | 同上；需要扩 F2/F5 类课程或收窄 P3 范围，均需新预注册 |
| `frame_proof=exact` 的接入规则 | passed（预注册 + 方案 + 可执行 schema） | [p3-exact-capture-v1.md](<../preregistration/p3-exact-capture-v1.md>)、[P3-exact-integration-plan.md](<P3-exact-integration-plan.md>)、`touhou_rl/frame_proof.py`（16 单测） |
| exact 数据集本体 | **blocked（未实现、未采集）** | 方案与判据 E1–E10 已写死；`capture_dataset.py` 未改，回放模式下的 60 帧验证未做 |

## P4 学习诊断（全部合成）

| 声称 | 状态 | 证据 |
|---|---|---|
| 时序策略/runner/loss/sidecar 工程 | passed（单测） | [P4-learning.md](<P4-learning.md>) |
| 真实基座 GPU 批量 = CPU 逐帧（严格容差） | **failed**：8.3e-6 | [P4-numeric-round7.md](<P4-numeric-round7.md>) |
| 同设备同批量重复逐位一致 | passed | 同上；修复为 cache /2 绑定计算协议 |
| 合成渲染 16 事件可学（离线 93.75%） | passed（离线） | [P4-synthetic-learning-round10.md](<P4-synthetic-learning-round10.md>) |
| 离线一致率 ≥95% 小集门 | **failed**：93.75% | 同上 |
| 三臂等预算：B1 0/8、N0 1/8、N1 1/8 | failed（无结构收益） | [P4-arms-round14.md](<P4-arms-round14.md>) |
| 关键移动召回 | **failed**：F2 两类聚焦召回 0 | [P4-recall-round14.md](<P4-recall-round14.md>) |
| 加权动作监督 | partial：加权在新世界均值为 7.33 vs 4.00，但受 F1 退化污染 | [P4-weighted-round15.md](<P4-weighted-round15.md>)、[P4-seed-confirm-round16.md](<P4-seed-confirm-round16.md>)、[P4-degenerate-round17.md](<P4-degenerate-round17.md>) |
| 未训练 B0 在同一确认渲染 | 9/24，但 **99.7% 恒定 right、0 切换**；F1 8/8 靠贴墙退化 | [P4-degenerate-round17.md](<P4-degenerate-round17.md>) |
| F2（需要区分动作的族）上训练臂 vs B0 | 训练臂更好：B0 0/8、unweighted 2/8、weighted 3/8 | 同上 |
| 聚合 9/24 作为"能力" | **invalid**：F1 退化分污染聚合 | 同上 |

## P5 闭环与补采

| 声称 | 状态 | 证据 |
|---|---|---|
| 冻结合成 N1 闭环 | **invalid**：R11 F1 两例恒 right/left 都可过，不计；F2 1/4 仍算有限真动作证据 | [P4-degenerate-round17.md](<P4-degenerate-round17.md>) |
| 失败状态独立恢复见证 | passed（存在性，非策略） | [P5-forensics-round12.md](<P5-forensics-round12.md>)：12/12 状态有替代路线 |
| 首轮补采 | **invalid（聚合）**：对照/补采的差异里含常量可解的 F1 例 | [P5-dagger-training-round13.md](<P5-dagger-training-round13.md>)、[P4-degenerate-round17.md](<P4-degenerate-round17.md>) |
| 多种子稳定性 | partial：unweighted 0–3/8 波动；逐事件表曾错位，汇总已复核 | [P4-seed-confirm-round16.md](<P4-seed-confirm-round16.md>)（含更正节） |
| KL 锚定（过滤后非常量可解世界） | partial：F2+F3 子集 A 3/16 → B 7/16，配对 0 负；n=3 | [P4-kl-anchor-round17.md](<P4-kl-anchor-round17.md>) |
| 过滤后未训练 B0 | **0/24**（旧 9/24 是常量贴墙） | 同上 + [P4-degenerate-round17.md](<P4-degenerate-round17.md>) |
| 冻结划分上的蒸馏与留出闭环 | **failed**：A 26.2% / B 19.0% / B0 0/14，F3 全 0 | [P4-validcurriculum-round20.md](<P4-validcurriculum-round20.md>) |
| 离线准确率能否预测闭环 | **no**：0.913–0.942 对 14%–29% | 同上 |
| F1 在像素上是否可学 | **no（构造缺陷）**：缺口在致命前从未渲染 | [P1-f1-visibility-round20.md](<P1-f1-visibility-round20.md>) |
| 补上预告后 F1 是否可学 | **no（更差）**：A 3/3/3→2/2/2、B 2/3/3→2/2/2 | [P4-telegraph-round21.md](<P4-telegraph-round21.md>) |
| F1 失败机制 | **方向先验**：base 15/18 走对、镜像 15/18 走反；死亡仍 100% 在 wall_frame | 同上 |
| 空间特征融合能否修复方向 | **no**：A=S=2/2/2，14 事件 first-hit 全同；融合 94.4% 学成常数偏置 | [P4-spatialfuse-round22.md](<P4-spatialfuse-round22.md>) |
| 256 维 features 是否含缺口侧信息 | **需修正读法**：R23 的"缺口相对屏幕中心"标签在留出 6/6 等价于"玩家 x>192"，98.5% 可解码性有相当部分来自玩家坐标 | [P4-prelaux-round24.md](<P4-prelaux-round24.md>) |
| 底座是否 mirror-blind | **yes**：镜像世界仍 16–22/22 投屏幕向右 | [P4-spatialfuse-round22.md](<P4-spatialfuse-round22.md>) |
| 玩家相对标签辅助任务 | **improved 但只限 base**：F1 2/2/2→3/3/3，镜像 3/9→0/9，屏幕方向 18右/0左/0hold | [P4-prelaux-round24.md](<P4-prelaux-round24.md>) |
| 数据规模是否为 F1 失败主因 | **no**：训练组 6→12（24 事件），"需向左"21 格翻转 0/21 | [P4-f1scale-round25.md](<P4-f1scale-round25.md>) |
| 合成学习线总体结论 | **卡在镜像方向**；详见收敛报告 | [synthesis-round25.md](<synthesis-round25.md>) |
| 特权辅助任务（屏幕相对标签） | **failed**：F1 2/2/2→1/3/2，合计 3→1/3/2，镜像 3/9→1/9（先验被加强） | [P4-privaux-round23.md](<P4-privaux-round23.md>) |
| 辅助头是否学到 | 训练 0.943、留出 0.636/0.682/0.500 → 按预注册规则**不能宣称学会** | 同上 |
| 学到是否被用 | **no，且用反**：镜像组辅助头读对 8/9、策略走反 8/9 | 同上 |
| 行为形态 | 恒定 0/84，两段式 84/84；KL 只改 hold 占比 | [P4-validcurriculum-round20.md](<P4-validcurriculum-round20.md>) |
| 辅助标签与教师首动作方向 | 12/12 同向，但教师 6/12 在锚定行 hold ⇒ 方向真值≠时机真值 | [P4-prelaux-round24.md](<P4-prelaux-round24.md>) |
| 底座行为保持 | **invalid（聚合）**：B0 的 F1 是常量贴墙，非能力；F2 上训练臂反超 | [P4-degenerate-round17.md](<P4-degenerate-round17.md>) |
| 方向信息是否在像素里（预告渲染） | **passed（可读）**：无学习直读器 12/12 有效世界、480/480 生成；无预告 0/492；遮挡后 0/492 | [P1-pixel-info-round27.md](<P1-pixel-info-round27.md>) |
| 小 CNN 能否从像素学"缺口相对自机" | **partial**：只训 base→镜像 0.860（<预注册 0.90）；两朝向混合→组留出 0.896–0.917；边缘位置留出 0.50（分布外） | 同上 |
| "再加预告/信息不足"这个解释 | **已排除**：R21 的负结果不能用"预告不够"解释 | 同上 |

## P6 / P7

| 声称 | 状态 |
|---|---|
| 真机共同管线小试 | blocked（不再是"P0 同帧性"，而是 P3 数据集未预注册） |
| 独立确认与发布 | blocked |

## 读法

- **passed** 只表示"该证据范围内的该条声称成立"，不跨范围外推。例如教师 v2 的 24/24 只限固定 y 合成几何。
- 任何 failed/unknown 行都保留在原位；后续修复只能新增一行，不能删除旧行。
- 目前**没有任何一项**达到"真实像素闭环能力"级别，默认模型未替换。
