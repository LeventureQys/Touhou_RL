# WP04.2 — D410-S1 正式受控 BC 四臂（B0/B1/B2/B3 × 3 seeds × 20 epoch）

- 机器可读证据：[`arms.json`](arms.json)；本文件是人读版本，所有数字都由 `arms.json` 生成（生成脚本 `.tmp_v410_wp04_arms/make_arms_md2.py`，不入库）
- runner manifest：`.tmp_v410_wp04_arms/arms_manifest.json`（每个 run 的真实退出码）
- 生成时间：2026-11-12T21:55:28Z
- 范围：离线 BC，train seeds 1–14、val seeds [15, 16, 17]；**test seeds [18, 19, 20] 从未读取**；不接引擎、不采数据、不启动游戏、不做 G410-2 结论
- 数据：20 shards / 66,143 decisions；train 46,397 行、val 9,874 行（无 chosen 目标的行被排除：train 0 / val 0 行）

## 0. 结论摘要（不夸大）

- **12/12 个 run 真实跑完**（无缺失、无失败）：每个 run 20 × 363 = 7,260 step；真实 `samples_seen` = 927,940（= 20 × 46,397，最后一个 batch 不满 128）；batch 128，OOM fallback 0 次。
- epoch 20 的 val top1（3 seeds 均值）：B0 0.6254、B1 0.6241、B2 0.6594、B3 0.6605；macro recall：B0 0.3101、B1 0.3026、B2 0.3131、B3 0.3305。
- **所有臂都明显过拟合**：逐 epoch 的 val top1 均值峰值出现在 B0 ep4（0.6816）、B1 ep4（0.6798）、B2 ep5（0.6859）、B3 ep4（0.6883）；到 epoch 20 全部回落。预注册协议没有早停/选模，因此正式数字按 epoch 20 报，峰值只作诊断记录，**不作为选型依据**。
- 空间臂（B2/B3，约 2.60M 参数）在 epoch 20 的 val top1（B2 0.6594）高于 ProbeNet 臂（B0 0.6254）；但两者**模型族、容量、吞吐都不同**，这个差值不能归因为“空间结构有效”。
- 辅助炸弹项（B1、B3 的 BCE@0.5）没有信息量：D410-S1 的 bomb 标签全为 0（train+val 阳性行 0），aux 只学到负类先验；B1−B0 / B3−B2 的差值**不是**辅助标签收益的证据。

## 1. 预注册（训练前冻结，之后未改）

| 项目 | 值 |
| --- | --- |
| 主目标 | `chosen_plan.first_action` one-hot 交叉熵；soft/KL 目标不使用 |
| 辅助目标 | bomb BCE-with-logits，权重 B0/B2 = 0.0、B1/B3 = 0.5 |
| unknown / query / survival | 仅诊断；不作负例、不进 loss、无类权重 |
| 切分 | 整 seed，70/15/15：train 1–14、val 15–17、test 18–20（test 不读） |
| 优化器 | AdamW，lr 0.0003，weight_decay 0.0001，grad clip 5.0 |
| 预算 | 20 epoch × 363 step/epoch（= ceil(46397/128)），batch 128，eval batch 128 |
| 训练种子 | [0, 1, 2]（每臂 3 个，固定） |
| 增强 / 早停 / 选模 / 调参 | 全部无；固定跑满 20 epoch，报最后一轮 |

| 臂 | 模型 | 参数 | bomb head | aux 权重 | 定义 |
| --- | --- | ---: | --- | ---: | --- |
| B0 | `ProbeNet` | 731,802 | 否 | 0.0 | WP04.0 `ProbeNet`，仅 chosen one-hot CE |
| B1 | `BombAuxNet` | 731,931 | 是 | 0.5 | B0 + `Linear(128,1)` bomb head（129）+ BCE@0.5 |
| B2 | `SpatialBombNet` | 2,600,951 | 否 | 0.0 | `model_v3.StudentV3` 规范空间模型适配：去掉 2-way bomb head 与 value head，无 aux |
| B3 | `SpatialBombNet` | 2,601,464 | 是 | 0.5 | B2 + `Linear(512,1)` binary bomb head（513）+ BCE@0.5 |

空间臂细节：canonical `model_v3.StudentV3()` 实测 **2,602,490** 参数（其中 perception head group 25,557）；B2 去掉 2-way bomb head（1,026）与 value head（513）→ **2,600,951**；B3 再加 513 参数的 binary head → **2,601,464**。两者都在预注册的 2–6M 带内。perception heads 保留在参数计数里，但**没有任何 BC 臂训练它**：梯度覆盖探针显示空间臂只有 2,575,394 / 2,600,951（B2）与 2,575,907 / 2,601,464（B3）个参数拿到非零梯度。

## 2. 实际命令与退出码

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
# 验收 1：新测试（exit 0，43 passed）
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp042_pytest -q `
    tests/test_v410_student_bc_arms.py
# 验收 2：编译（exit 0）
python -m compileall -q touhou_rl/v410/student
# 验收 3：12 个正式 run，真实训练（exit 0）
python -u -m touhou_rl.v410.student.bc --arms B0 B1 B2 B3 --seeds 0 1 2 --epochs 20 `
    --audit Document/Update/Alpha/v4.1.0/evidence/WP03/s1c_full_audit.json `
    --repo-root . --arms-config configs/v410/bc_s1_arms.json `
    --workdir .tmp_v410_wp04_arms --runs-dir .tmp_v410_wp04_arms --device cuda
# 验收 4：聚合证据（exit 0）
python -m touhou_rl.v410.student.bc --assemble-arms .tmp_v410_wp04_arms/B0_s0.json `
    .tmp_v410_wp04_arms/B0_s1.json ... --out Document/Update/Alpha/v4.1.0/evidence/WP04/arms.json
# 回归：WP04.1 既有测试（exit 0，53 passed）
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp042_pytest_existing -q `
    tests/test_v410_student_bc.py
```

每个 (arm, seed) 的真实退出码记录在 `arms.json.runs[].exit_code`（来源：runner manifest）。本次 `failed_runs` = {}，`missing_runs` = 0 个，`partial` = false。

## 3. 每臂每种子 val 指标（epoch 20）

| 臂 | seed | 参数 | train loss | train top1 | val loss | val top1 | val macroR | bomb aux BCE | bomb acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0 | 731,802 | 0.0875 | 0.9769 | 2.9367 | 0.6280 | 0.3080 | 0.0000 | 0.0000 |
| B0 | 1 | 731,802 | 0.0818 | 0.9770 | 3.1153 | 0.6210 | 0.3084 | 0.0000 | 0.0000 |
| B0 | 2 | 731,802 | 0.1033 | 0.9717 | 2.9225 | 0.6271 | 0.3139 | 0.0000 | 0.0000 |
| B1 | 0 | 731,931 | 0.0784 | 0.9786 | 3.0247 | 0.6261 | 0.3044 | 0.0000 | 1.0000 |
| B1 | 1 | 731,931 | 0.0762 | 0.9795 | 3.1602 | 0.6216 | 0.2950 | 0.0000 | 1.0000 |
| B1 | 2 | 731,931 | 0.1111 | 0.9691 | 2.9297 | 0.6247 | 0.3085 | 0.0000 | 1.0000 |
| B2 | 0 | 2,600,951 | 0.0537 | 0.9828 | 3.3528 | 0.6576 | 0.2923 | 0.0000 | 0.0000 |
| B2 | 1 | 2,600,951 | 0.0496 | 0.9843 | 2.8052 | 0.6557 | 0.3234 | 0.0000 | 0.0000 |
| B2 | 2 | 2,600,951 | 0.0536 | 0.9826 | 2.8039 | 0.6649 | 0.3234 | 0.0000 | 0.0000 |
| B3 | 0 | 2,601,464 | 0.0549 | 0.9831 | 2.4785 | 0.6595 | 0.3355 | 0.0000 | 1.0000 |
| B3 | 1 | 2,601,464 | 0.0454 | 0.9859 | 2.8417 | 0.6547 | 0.3216 | 0.0000 | 1.0000 |
| B3 | 2 | 2,601,464 | 0.0591 | 0.9811 | 2.7320 | 0.6671 | 0.3344 | 0.0000 | 1.0000 |

### 3.1 聚合（3 seeds，mean ± 样本标准差；描述性，不是置信区间）

| 臂 | params | val top1 | val macro recall | val loss | train top1 | samples/s | GPU 峰值 MiB |
| --- | ---: | --- | --- | --- | --- | ---: | ---: |
| B0 | 731,802 | 0.6254 ± 0.0038 | 0.3101 ± 0.0033 | 2.9915 ± 0.1074 | 0.9752 ± 0.0030 | 4165.6 ± 40.3 | 252.7 |
| B1 | 731,931 | 0.6241 ± 0.0023 | 0.3026 ± 0.0069 | 3.0382 ± 0.1159 | 0.9757 ± 0.0058 | 4113.0 ± 90.4 | 252.7 |
| B2 | 2,600,951 | 0.6594 ± 0.0049 | 0.3131 ± 0.0179 | 2.9873 ± 0.3165 | 0.9832 ± 0.0009 | 1792.2 ± 48.2 | 1467.9 |
| B3 | 2,601,464 | 0.6605 ± 0.0062 | 0.3305 ± 0.0077 | 2.6841 ± 0.1863 | 0.9834 ± 0.0024 | 1583.1 ± 84.6 | 1467.9 |

### 3.2 动作分布与辅助项

val 共 9,874 行，多数类基线 top1 = 0.4412；四臂都高于基线，但 macro recall 只有 0.303–0.331，说明 18 个动作里仍有若干类别几乎不被召回。

下表按**每个臂的 seed 0** 取一行（下面的多数类效应表才用 3 seeds 均值），完整的逐 seed 分布见 `arms.json`。

| 臂 | 覆盖类别数 | 最高频类的目标占比 | 该类的 predicted 占比 | 该类 recall | bomb aux BCE (val) | bomb 正例率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 17 | 0.4412 | 0.4712 | 0.8324 | 0.0000 | 0.0000 |
| B1 | 17 | 0.4412 | 0.4735 | 0.8372 | 0.0000 | 0.0000 |
| B2 | 17 | 0.4412 | 0.5777 | 0.9144 | 0.0000 | 0.0000 |
| B3 | 17 | 0.4412 | 0.4508 | 0.8359 | 0.0000 | 0.0000 |

（逐 seed、逐 18 类的完整 recall / support / target / predicted 分布与混淆矩阵都在 `arms.json.arms.<arm>.seeds.<seed>.val` 里。）

**多数类效应**（动作 0，占 val 的 0.4412）：

| 臂 | 该类 recall（seed 均值） | 该类预测占比（seed 均值） | top1 − macro recall |
| --- | ---: | ---: | ---: |
| B0 | 0.8196 | 0.4517 | 0.3153 |
| B1 | 0.8266 | 0.4611 | 0.3215 |
| B2 | 0.8753 | 0.5057 | 0.3463 |
| B3 | 0.8506 | 0.4715 | 0.3300 |

空间臂的 top1 优势主要来自这个多数类：B2 在该类的 recall 0.8753 / 预测占比 0.5057，B0 为 0.8196 / 0.4517（Δrecall +0.0558、Δpred +0.0539）。但整臂 macro recall 只从 0.3101 变到 0.3131（Δ +0.0030），而 top1 的差值是 +0.0340：更高容量的空间模型把概率更多压给了最高频动作，少数类没有同等改善，且 B2 的 macro recall 逐种子为 0.2923、0.3234、0.3234（波动大于两臂均值之差）。因此“B2 top1 更高”不等于“B2 策略更好”，必须看 macro recall 与闭环评估。

## 4. 训练曲线（逐 epoch，3 seeds 均值，val）

| epoch | B0 top1 | B1 top1 | B2 top1 | B3 top1 | B0 loss | B1 loss | B2 loss | B3 loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5761 | 0.5675 | 0.6189 | 0.6215 | 1.3989 | 1.4214 | 1.2784 | 1.2841 |
| 2 | 0.6580 | 0.6543 | 0.6725 | 0.6612 | 1.1943 | 1.2059 | 1.1181 | 1.1478 |
| 3 | 0.6753 | 0.6747 | 0.6838 | 0.6517 | 1.1379 | 1.1444 | 1.0766 | 1.1920 |
| 4 | 0.6816 | 0.6798 | 0.6822 | 0.6883 | 1.1109 | 1.1097 | 1.0920 | 1.0485 |
| 5 | 0.6797 | 0.6791 | 0.6859 | 0.6807 | 1.1208 | 1.1243 | 1.1159 | 1.1249 |
| 6 | 0.6747 | 0.6752 | 0.6728 | 0.6537 | 1.1416 | 1.1407 | 1.1364 | 1.2192 |
| 7 | 0.6701 | 0.6695 | 0.6732 | 0.6738 | 1.1613 | 1.1597 | 1.2243 | 1.1799 |
| 8 | 0.6678 | 0.6677 | 0.6700 | 0.6587 | 1.2278 | 1.2225 | 1.2875 | 1.3945 |
| 9 | 0.6576 | 0.6593 | 0.6619 | 0.6510 | 1.2947 | 1.2777 | 1.5138 | 1.5559 |
| 10 | 0.6557 | 0.6521 | 0.6579 | 0.6566 | 1.3907 | 1.3771 | 1.6907 | 1.7113 |
| 11 | 0.6467 | 0.6460 | 0.6578 | 0.6573 | 1.4904 | 1.4986 | 1.8983 | 1.8469 |
| 12 | 0.6455 | 0.6422 | 0.6596 | 0.6557 | 1.6332 | 1.6169 | 2.1219 | 2.1861 |
| 13 | 0.6449 | 0.6390 | 0.6578 | 0.6601 | 1.8015 | 1.7847 | 2.3469 | 2.2488 |
| 14 | 0.6323 | 0.6309 | 0.6608 | 0.6525 | 1.9608 | 1.9569 | 2.3064 | 2.3364 |
| 15 | 0.6334 | 0.6295 | 0.6585 | 0.6531 | 2.1481 | 2.1412 | 2.4831 | 2.5204 |
| 16 | 0.6317 | 0.6297 | 0.6551 | 0.6540 | 2.3472 | 2.3561 | 2.5278 | 2.5642 |
| 17 | 0.6267 | 0.6261 | 0.6539 | 0.6557 | 2.5544 | 2.5581 | 2.6153 | 2.5343 |
| 18 | 0.6245 | 0.6216 | 0.6600 | 0.6558 | 2.7119 | 2.7434 | 2.5847 | 2.6029 |
| 19 | 0.6285 | 0.6239 | 0.6606 | 0.6590 | 2.9257 | 2.9046 | 2.6453 | 2.7510 |
| 20 | 0.6254 | 0.6241 | 0.6594 | 0.6605 | 2.9915 | 3.0382 | 2.9873 | 2.6841 |

train loss 在 epoch 20 降到 ~0.08–0.10、train top1 > 0.97，而 val loss 持续上升——这是过拟合，不是“训练还没收敛”。

## 5. 吞吐与 GPU

| 臂 | steps/run | samples_seen/run | samples/s（均值） | steps/s（均值） | train loop 秒 | eval 秒/run | GPU 峰值 MiB（max） | 显存 reserved MiB（max） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 7,260 | 927,940 | 4165.6 | 32.59 | 222.0 | 28.4 | 252.7 | 432.0 |
| B1 | 7,260 | 927,940 | 4113.0 | 32.18 | 224.5 | 27.9 | 252.7 | 428.0 |
| B2 | 7,260 | 927,940 | 1792.2 | 14.02 | 509.4 | 43.9 | 1467.9 | 2436.0 |
| B3 | 7,260 | 927,940 | 1583.1 | 12.39 | 552.1 | 51.8 | 1467.9 | 2436.0 |

设备：NVIDIA GeForce RTX 4070 SUPER（12281.5 MiB 总显存），torch 2.14.1+cu132，CUDA 13.2，Python 3.12.10。所有 run **串行**执行（不并发）；samples/s 只统计训练 batch 循环，数据加载约占其 41–46%（`throughput.data_fraction_of_train`），wall_seconds 还包含 21 次 val 评估。

**显存口径**：`torch.cuda.max_memory_allocated/reserved` 只统计本进程，即使机器上同时有别的 GPU 进程，这里的数字仍是本 run 的真实占用。**吞吐口径**：矩阵运行期间机器上同时有 v400 retail trace 的 `th08.exe` 在跑（不是本工作包启动的），所以 samples/s 是共享 GPU 下的保守值；B0/B1 的 4.1–4.3k samples/s 与 WP04.1 smoke 的 4,013 samples/s 同量级，说明影响有限。

## 6. 与 WP04.1 smoke 的对照

| 臂 | WP04.1 smoke（1 epoch）val top1 | WP04.2 ep1（3 seeds 均值） | WP04.2 ep20（3 seeds 均值） | WP04.1 参数 | WP04.2 参数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.5613 | 0.5761 | 0.6254 | 731,802 | 731,802 |
| B1 | 0.5482 | 0.5675 | 0.6241 | 731,931 | 731,931 |
| B2 | 未训练 | 0.6189 | 0.6594 | — | 2,600,951 |
| B3 | 未训练 | 0.6215 | 0.6605 | — | 2,601,464 |

- 可复现性：WP04.1 的 B0 单 epoch val top1 = 0.5613；WP04.2 的 B0 seed 0 第 1 epoch = 0.5613、3 seeds 均值 = 0.5761。同协议/同数据/同预算下量级一致；差异来自 WP04.1 只有 1 个种子（0），这里报 3 种子均值。
- 20 epoch 的**最后一轮**确实比第 1 轮好，但都低于第 3–5 轮的峰值：B0 ep1→ep20 0.5761→0.6254（峰值 0.6816 @ep4），B1 0.5675→0.6241（峰值 0.6798 @ep4），B2 0.6189→0.6594（峰值 0.6859 @ep5），B3 0.6215→0.6605（峰值 0.6883 @ep4）。也就是说：训练是有效的，但**预注册的 20 epoch 预算在 S1 上偏大**，多出来的 epoch 只增加过拟合；这是本轮最重要的负面结论。

## 7. 辅助项差值（仅描述，不做显著性声明）

| 对比 | Δ val top1（均值） | Δ val macro recall（均值） | Δ val loss（均值） |
| --- | ---: | ---: | ---: |
| B1 − B0 | -0.0012 | -0.0074 | 0.0467 |
| B3 − B2 | 0.0011 | 0.0174 | -0.3032 |

逐种子差值见 `arms.json.deltas.*.per_seed`。解读边界：

- bomb 标签在 66,143 行上恒为 0，aux 项退化成正类先验，**不能**说明“辅助标签有用/无用”；
- B1−B0（+129 参数 head 与 0.5 aux）和 B3−B2（+513 参数 head 与 0.5 aux）是同一模型族内唯一干净的对照，但只有 3 个配对种子，差值落在种子噪声量级内就不应解读；
- B0/B1 与 B2/B3 之间还差模型族、容量与训练配方，**不可**当作“空间结构”的因果证据。

## 8. 限制

1. **过拟合**：4 个臂都在 epoch 3–5 后 val loss 上升、top1 回落；协议不做早停/选模，epoch 20 是唯一允许报告的点，因此这些数字是“预算上限下的表现”，不是各臂的最佳表现。
2. **预算偏大**：20 epoch × 363 step ≈ 929k 次样本曝光（约 20× 数据量），对 S1 明显过量。
3. **标签上限**：目标只有 chosen 的 first_action（18 类），unknown 不作负例、soft/KL 目标不使用，所以这里测的是“模仿单一 chosen 动作”，不是教师几何的拟合质量。
4. **辅助项无信号**：bomb 恒 0（见 §7）。
5. **种子数 3**：mean/std 是描述性统计，不是置信区间，也不做显著性检验。
6. **只看离线指标**：top1 / macro recall 与游戏存活/得分没有已知映射，闭环价值必须由开发局/引擎评估给出。
7. **机器共享**：矩阵期间 GPU 上另有 v400 trace 的 `th08.exe` 在跑，吞吐偏保守（显存数字不受影响，见 §5）。
8. **B2/B3 是适配层**：空间臂由 `bc.SpatialBombNet` 包装 canonical `StudentV3`，移除 value head 与（B2 的）bomb head，perception heads 保留却不训练；参数计数因此包含 25,557 个无梯度参数，判断“容量”时应扣除。

## 9. 下一步建议

1. **开发局（引擎）评估**：用本轮 12 个 checkpoint（`.tmp_v410_wp04_arms/*_last.pt`，SHA-256 见 `arms.json`）在固定开发局种子上做闭环评估，比较存活/得分，而不是只看离线 top1；这是判定“哪个臂真的更好”的唯一途径，也是 G410-2 的前置。
2. **预算重定**：以 §4 的 val 曲线为证据，把下一轮预注册的 epoch 预算降到 3–6（或固定 ~1.5k–2k step），写死在预注册里，不再事后挑轮次。
3. **辅助项**：bomb 恒 0 使 B1/B3 的 aux 在 S1 上无意义；要验证辅助学习，需要先有带正例的 bomb 标签或换成真正有信息量的辅助目标（如 perception 标签），并在 WP04.3 预注册中固定。
4. **同容量对照**：若要把 B2/B3 的优势归因于空间结构，需要同容量、同配方的非空间臂（例如 2.6M 的 MLP/ProbeNet 放大版）或同族消融，不能直接用 B0↔B2 比较。
5. **测试集**：seeds 18–20 至今未读；应只在四臂方案冻结、开发局评估给出方向后一次性使用。

## 10. 未声明（not claimed）

- not G410-2: no closed-loop evaluation, no survival/score metric, no engine and no game launch; this is an offline train/val BC result
- not the final student model: these arms are the WP04.2 controlled comparison, not a deployable policy
- the test split (seeds 18-20) is never read, so no test metric exists and no test-split tuning happened
- B0/B1 (731,802 / 731,931) and B2/B3 (2,600,951 / 2,601,464) differ in capacity as well as in the auxiliary term, so a B0-vs-B2 gap confounds model space and training recipe; only B1-B0 and B3-B2 isolate the aux term inside one model space
- the bomb label is constant 0, so the auxiliary BCE degenerates to the negative prior and no auxiliary-label benefit can be shown here
- three training seeds per arm: the mean/std are descriptive, not a confidence statement, and no significance test is performed
- no hyper-parameter search, no early stopping, no val-based checkpoint selection: epoch 20 is reported as-is
- the closed-loop (development-game) value of these policies is unknown until the engine evaluation of a later work package

## 11. 产物与校验

| 项 | 值 |
| --- | --- |
| 证据 schema | `v410-wp04-bc-arms/1` |
| 预注册 schema | `v410-bc-s1-arms/1`（`configs\v410\bc_s1_arms.json`，sha256 `7ca369eccb8758a2…`） |
| 审计输入 | WP03 `s1c_full_audit.json`；`arms.json.split_seeds.integrity` 记录 20 个 shard 的归属与跨切分 shard 数（0） |
| runs | 12/12 |
| checks | test_split_read=False、val_used_for_tuning=False、formal_four_arm_trained=True、game_launched=False、engine_connected=False、checkpoints_in_repo=False |
| 每 run 的 checkpoint SHA-256 | `arms.json.arms.<arm>.seeds.<seed>.checkpoint.sha256` |

checkpoint 只存在于 `.tmp_v410_wp04_arms/`（`.tmp_*/` 已在 `.gitignore`，实测 `git check-ignore` 命中），可由 §2 的命令复现。

