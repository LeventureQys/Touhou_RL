# P0 像素定位与 counter-lag 离线诊断（round5）

## 结论

**本轮是特权起始模板的离线计算诊断，不是部署视觉，也不是正式截图同步证明。`real_pixel_gate_passed=False`，相位与同步结论始终 unknown。**

新增[实现](<../../../../../touhou_rl/tactical_pixel_probe.py>)、[合成测试](<../../../../../touhou_rl/tests/test_tactical_pixel_probe.py>)。不修改现有 capture/audit/game 模块，不运行游戏；仅消费主 agent 已保存的 [RGB](<../evidence/capture-round5/frames.npz>) 和 [trace](<../evidence/capture-round5/trace.json>)。本报告不依赖 image tool 或肉眼 sprite 身份复核。

## 方法与固定边界

- 首个 pre/post 的 gameplay_frame、position、player_state、focus_mode 均稳定、图像非常量的采样，用内存位置加 origin=(32,16) **一次性提取** 24×24 模板。支持预先指定 32×32，但本次不尝试第二模板或调参。
- 本次 anchor 为 capture index=0、counter=240，memory=(192,384)，crop=(212,388,24,24)，player_state=0/focus=0。稳定 bracket 不是 sprite 身份或显示相位证明。
- 后续每帧独立搜索固定全 playfield=(32,16,384,448)，不以当前内存点限制窗口。纯 numpy FFT zero-mean NCC，比较灰度原始模板与梯度模长模板；不依赖 scipy/torch。
- 固定 NCC≥0.65、主峰与远处次峰差≥0.06、raw/gradient 位置差≤3px，且状态/focus 与起始相同，才记为 localization candidate。低置信、多峰、动画/状态变化及两模式分歧均 unknown。两模式源于相同图像，**不构成两份独立证据**。
- 初始模板 capture 不参与 lag 证据。counter 合并时不挑第一/最后：同 counter 有多个 memory position 则剔除；同 counter 像素候选差>1px 亦剔除。只用稳定 bracket 且像素高置信样本，不插值。
- lag 扫描[-8,+8]，正数表示图像落后内存；所有 lag 用完全相同的公共 counter 样本。逐 lag 拟合固定 anchor 偏移、计算 RMSE，候选阈值为 best RMSE+0.35px。固定最少12点、max error=2px；恒速运动可被固定 anchor 吸收，必须保留不可辨识。
- trace 的 phase_verified=False 不被偷偷升级；单独输出 counter_based_candidate_only，不冒充现有审计器所需的 phase_verified 输入。

## 首次固定阈值实测

[完整输出](<../evidence/capture-round5/pixel-probe.json>)含每 capture 原始 RGB SHA256、raw/gradient confidence、次峰、位置、原因、候选 lag 分数与输入 SHA256。

| 指标 | 结果 |
|---|---:|
| captures | 290 |
| 独立于 anchor 的双模式高置信候选 | 43 / 289 |
| unknown（不含 anchor） | 246 / 289 |
| raw 单模式候选（含 anchor） | 53 |
| gradient 单模式候选（含 anchor） | 44 |
| raw 低置信 unknown | 237 |
| gradient 低置信 unknown | 246 |
| 高置信但不稳定 bracket 排除 | 7 captures |
| 同 counter 冲突 memory / pixel | 0 / 0（仅此 trace） |
| 全[-8,+8]共用 counter 数 | **10，少于固定12** |

首次运行的直接结果：unknown / insufficient_unambiguous_common_counter_coverage，scores=[]、lag_candidates=[]。之后仅改变“样本不足时也保留探索性分数”的输出逻辑，**没有改变模板、confidence、size、ROI、最少样本量**，候选数量完全相同。

[首次结果快照](<../evidence/capture-round5/pixel-probe-initial.json>)在第二次覆盖主输出后，依据不变的逐帧定位结果和首次 pwsh-133 stdout 重建；文件内明确标注 artifact_provenance，**不是未经修改的首次原始artifact**。

10点的探索性counter分数：

| lag | RMSE px |
|---:|---:|
| 0 | 0.356697 |
| +1 | 2.776961 |
| -1 | 2.865636 |
| +2 | 5.300546 |
| -2 | 5.413864 |
| +3 | 7.828158 |

探索性候选集合=[0]；lag0拟合 anchor offset≈(32.1127,16.1127)。**这不是同步=0的判定**：样本只有10<12，且相位未验证，绝大多数图像低置信，单模板可能锁定背景或错误身份，选入样本受同一模板动画状态偏好影响。最终 status=unknown，scores_are_exploratory_only=True，real gate=False。未降低门槛来“通过”。

### 输入文件哈希

- RGB npz SHA256：`2e9e2050aaac547eb6bba791a79b65c73193892b0f9b8aa7afa5069b79d53186`
- trace SHA256：`479bdc184ec3b1199a0a2036ebd5bf1815c006f76720c8f7c61db2e35c22d973`

逐图 hash 只用于重现/重复关联，变化不证明自机新鲜或显示相位。

## 验证

```powershell
python -m pytest touhou_rl/tests/test_tactical_pixel_probe.py touhou_rl/tests/test_tactical_capture_audit.py -q -p no:cacheprovider --basetemp .tmp/pixel-round5-final-tests
python -m touhou_rl.tactical_pixel_probe --frames Document/Update/Alpha/v3.7.3/evidence/capture-round5/frames.npz --trace Document/Update/Alpha/v3.7.3/evidence/capture-round5/trace.json --output Document/Update/Alpha/v3.7.3/evidence/capture-round5/pixel-probe.json
```

最终退出0，**23 passed in 2.04s**（本次新probe 14例，原capture audit 9例）。覆盖 NCC 与 brute-force 一致、全场远距离 sprite 跳变、两个远处相同峰 unknown、背景/低conf unknown、全场合成 sprite +3lag、恒速 -3/0/+3 全候选不可辨识、冲突memory/pixel counter保守剔除、一次特权anchor排除、样本不足仍可给探索分数但unknown、无稳定模板/非原尺寸拒绝。

## 局限与后续

1. 第一帧内存辅助模板有明确特权，可能包含背景；本轮无人工核实自机身份、无独立模板标注开发段。
2. 单模板不能覆盖动画、透明闪烁、死亡、focus切换；246 unknown是实测失败边界，不是可以靠事后调阈值修掉的噪音。
3. NCC高分/两模式同意不是 sprite 语义证明，也不排除背景特征恰好相关运动。
4. 稳定RAM counter/position bracket没有D3D Present fence，也没有像素独立可读frame；无法从此升级±1帧正式门。
5. 下一轮若完善多动画模板，应预先划分独立开发/确认段、固定模板与阈值后再确认；需要更多非周期转向/停启高置信公共counter，而不是降低本轮最小样本标准。
