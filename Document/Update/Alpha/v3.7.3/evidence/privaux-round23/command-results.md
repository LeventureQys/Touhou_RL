# round23 命令、退出码与迭代记录

本文件是 [P4-privaux-round23.md](<../../reports/P4-privaux-round23.md>) 的执行台账。所有命令都在
`D:\WorkShop\Touhou_RL` 下运行；`python` 是 CPython 3.12 + torch 2.14.1+cu132（RTX 4070 SUPER），
FP32、无 AMP、`torch.backends.cuda.matmul.allow_tf32 = False`、`torch.backends.cudnn.allow_tf32 = False`、
`float32_matmul_precision = highest`（由 `_fp32_mode` 与工具内的硬断言共同保证）。

## 1) 单测（写工具期间共 8 轮）

```powershell
python -m pytest tests/test_v373_privileged_aux.py -q -p no:cacheprovider --basetemp .tmp/v373-privaux-tests-1
#    8 failed, 19 passed          exit 1   ← 见下面 4 个真实缺陷 1/2/3/4
python -m pytest tests/test_v373_privileged_aux.py -q -p no:cacheprovider --basetemp .tmp/v373-privaux-tests-2
python -m pytest tests/test_v373_privileged_aux.py -q -p no:cacheprovider --basetemp .tmp/v373-privaux-tests-3
python -m pytest tests/test_v373_privileged_aux.py -q -p no:cacheprovider --basetemp .tmp/v373-privaux-tests-4
#    2 failed, 25 passed          exit 1   ← 缺陷 5/6
python -m pytest tests/test_v373_privileged_aux.py -q -p no:cacheprovider --basetemp .tmp/v373-privaux-tests-5
#    27 passed in 4.28s           exit 0
python -m pytest tests/test_v373_privileged_aux.py -q -p no:cacheprovider --basetemp .tmp/v373-privaux-tests-6
#    27 passed in 4.27s           exit 0   （修 verify_prereg 之后）
python -m pytest tests/test_v373_privileged_aux.py -q -p no:cacheprovider --basetemp .tmp/v373-privaux-tests-7
#    27 passed in 4.31s           exit 0
python -m pytest tests/test_v373_privileged_aux.py -q -p no:cacheprovider --basetemp .tmp/v373-privaux-tests-final
#    27 passed in 4.20s           exit 0   （正式 run 之后复跑，冻结的测试文件仍通过）
```

**真实缺陷（工具 4 个、测试 4 个）**，全部由失败信息定位，没有一个是靠"看起来对"通过的：

| # | 位置 | 症状 | 原因与修法 |
|---|---|---|---|
| 1 | 工具 `r22_reference()` | 单测 `test_round22_reference_reproduces` 报 `KeyError: 'B-s7'` | 用了 `r20.ARMS`（round20 的 `A/B`），而 round22 的两臂是 `A/S` → 改用 `r22.ARMS`。这正是 round22 报告里记的缺陷 ①（`r20.all_names()` 返回 round20 的臂名）在我这里的同构再现 |
| 2 | 工具 `verify_prereg()` | 第一次 `run` 在开跑前就 `ValueError: the auxiliary supervision table changed` | 拿 **14 事件全表**和预注册里存的 **6 个有标签事件子表**比较 → 先过滤 `target_defined` 再比 |
| 3 | 工具 `write_records()` | 第二次 `run` 在写 `records.json` 时 `KeyError: 'bytes'` | 资产字节数从 round22 的 `run/records.json` 取，而那里只存 `path`/`sha256`；round21 的 `run/records.json` 才是冻结了 `bytes` 的地方 → 改为读 R21 |
| 4 | 工具 `run_probes()` | 第四次 `run` 在探针阶段 `KeyError: ('S-s7','validation')` | 复用了 `r22.stored_actions`，它内部硬编码 round22 的臂名 → 在本轮工具里按名字重表达同一段逻辑（`stored_actions`），并把该 helper 也纳入单测 |
| 5 | 测试 `test_the_captured_tensor_is_the_gru_state` | `RuntimeError: Inference tensors cannot be saved for backward` | 在 `inference_mode()` 外拿 hook 捕获的 inference tensor 去过 `nn.Linear` → 把四个 head 的重算放回 `inference_mode()` 内 |
| 6 | 测试 `test_the_policy_input_surface_is_unchanged` | `torch.Size([1,1,285]) == (1,80,285)` 失败 | 父类 `forward_features` 是**逐行**喂 GRU 的（`inp[:,i:i+1]`），只捕获到一次调用；顺带发现前一个真 bug：lambda 形式的 pre-hook 返回了 `setdefault` 的返回值（一个 Tensor），PyTorch 会把它当成替换后的 args → 改成显式 `return None` 的具名函数，并断言 hook 被调用 80 次 |
| 7 | 测试 `test_labels_are_not_an_input_surface` | `{'path','sha256'} <= {history 键}` 失败 | `train_records` 返回的 `history` 是**资产引用**，张量键在编码后的 `data` 里 → 改用编码后的行，并断言 `labels` 恰好是 `action_positive`/`action_valid` |
| 8 | 测试 `test_the_encode_pass_is_round22_s` | 94 维向量最后一位 `0.17768769` vs `0.17768766` | 存盘的 `spatial/*.npz` 是 round22 **在 CUDA 上**算的，单测在 CPU 上跑；逐位相等只在同设备成立 → 单测改为测 `max|Δ| < 1e-5` 并把该差值打印出来，**逐位**的 digest 相等由 `preregister`/`verify_prereg`/`run`（都在 CUDA 上）硬断言 |

一处**在产生任何结果之前**就调整过的实现选择，如实登记：`usage.aux_correct` 最初定义为"该事件
22 行全部正确"，在第一次成功的 run **之前**改成"该事件的平均 logit 的 argmax 与标签一致"（事件级读数），
理由与预注册里 `aux_accuracy` 的 caveat 一致（标签在事件内恒定，事件才是诚实的单位）。
第 1–3 次 run 都在训练之前或训练之后、探针之前失败，**没有任何行为学输出被看到**。

## 2) 旧轮回归（旧模块未被改动）

```powershell
python -m pytest tests/test_v373_spatial_fuse.py       -q -p no:cacheprovider --basetemp .tmp/v373-r22-regression-r23
#    22 passed in 6.00s   exit 0
python -m pytest tests/test_v373_telegraph.py          -q -p no:cacheprovider --basetemp .tmp/v373-r21-regression-r23
#    27 passed in 15.29s  exit 0
python -m pytest tests/test_v373_valid_curriculum.py   -q -p no:cacheprovider --basetemp .tmp/v373-r20-regression-r23
#    21 passed in 18.86s  exit 0
```

正式 run 之后又跑了一遍（日志在 `logs/regression-round{20,21,22}.log`）：22 / 27 / 21 全绿，exit 0。

## 3) 冒烟（scratch，预先披露）

`preregister` → `run` → `verify` 在 `.tmp/privaux-smoke` 上完整跑过一次（4 次尝试，前 3 次被上表缺陷 2/3/4 挡住）。
冒烟产物**没有用于任何结论**，但它的存在使得"两次独立进程逐位相同"可以被测量：

```powershell
python -m tools.rl.v373_privileged_aux preregister --out .tmp/privaux-smoke   # exit 0
python -m tools.rl.v373_privileged_aux run          --out .tmp/privaux-smoke   # exit 0
python -m tools.rl.v373_privileged_aux verify       --out .tmp/privaux-smoke   # exit 0
```

正式 run 与冒烟 run 的比对（`70` 个文件）：**59 个逐字节相同**；其余 11 个全部只差墙钟字段
（`run/train/*/training.json` 的 `optimization_seconds`，以及由它派生的 `run/checkpoints.json`、
`run/summary.json`、`run/timings.json`、两份 `sha256.json`）。去掉 `timings`/`optimization_seconds`
之后 `run/checkpoints.json` 与 `run/summary.json` **完全相等**。6 个 `research-heads.pt`、14 个闭环文件、
`f1-gap.json`、`probes.json`、`records.json`、`baselines.json`、28 个事件文件、14 个空间向量**逐字节相同**。

## 4) 正式冻结预注册

```powershell
python -m tools.rl.v373_privileged_aux preregister
#    exit 0；stdout 为空
#    criterion sha256 = fd382ac499fce04670f68c11c7dcb29a4c6106c390ab7f5dcaa6e720edae0a64
#    preregistration.json sha256 = f0dede1261e7875171b605874436403760b959abc08125f4fd91c5e85e8c3563
#    35 个身份 SHA；aux 权重 0.5；损失臂 A（control）；train 128 行 / validation 132 行
```

preregister 里**硬断言**的前置条件（任一不符即抛异常、不写文件）：

1. 14 个 train 资产的 `pixels/history/labels/rollout` 等于 R21 冻结 SHA（56 个绑定）；
2. round22 的 94 维空间向量 digest 重算等于 `83a75f73211b1964d5ff3cbb21220d09ed4b8d0e92fe181c010cd59909f58cd5`
   —— 这是"本轮头看到的 features 就是 round22 头看到的 features"的**可测量**形式；
3. round22 预注册里逐事件的 feature L2 范数表重算**完全相等**；
4. 每个 seed 的初始 head SHA 等于 R16/R20/R21/R22 冻结表；初始辅助头 SHA 在**两臂之间相同**；
5. 辅助监督表：6 个 F1 train 事件、128 行（3 左 + 3 右）、132 行 validation（3 + 3）；
6. R18 manifest / R19 split / worlds digest（`b47dd5f4b7001e48…`）与 round20/21/22 相同；
7. round22 的六件产物逐一 SHA 冻结，并把 round22 的 F1 留出（A 2/2/2、S 2/2/2）、合计（3/3/3）、
   控制头 SHA、方向计数（A base 6/3/0、mirror 3/0/6；六头 base 12/3/3、mirror 6/1/11）、
   `comparison.deltas = [0,0,0]`、`identical_first_hit_cells = [14,14,14]` 全部重算比对。

## 5) 唯一一次正式 run（后台 job，自行收取）

```powershell
python -m tools.rl.v373_privileged_aux run
#    exit 0
#    墙钟 ≈ 260 s：data 1.271 s / 训练 66.375 s（其中纯优化 63.025 s）/ 闭环 160.763 s / 探针 31.321 s
#    6 个 checkpoint 的 optimization_seconds：10.49 / 10.39 / 10.26 / 10.61 / 10.63 / 10.65
```

控制臂的三次训练**逐位复现 round22 的 A**：`d52551bb43e4…` / `fb9233f11bb2…` / `49ff13d9384f…`，
`matches_round22_control = true`（3/3）；`equivalence.all_equal = true`（6 个 (arm,split) 的闭环
**每一步** `candidate/actual_action/x_after/hit` 都与 round22 存盘相同）；`control_not_reproduced = false`。

## 6) 只读复核

```powershell
python -m tools.rl.v373_privileged_aux verify
#    exit 0
#    67 个 run 产物 / 69 个根产物 / 56 个资产 / 6 checkpoint / 14 闭环文件 / 28 事件文件
#    168 基线行 / 84 条 F1 机制行 / 35 个身份 SHA / aux 参数 514 / aux 权重 0.5
```

run 之后写入了 `logs/` 与 `command-results.md`，它们不在根 `sha256.json` 表内（该表在 run 结束时生成），
写入之后重跑 `verify` 仍 exit 0。

## 7) 退出码汇总

| 步骤 | 退出码 |
|---|---|
| 单测（最终） | 0 |
| 旧轮回归 ×3（最终） | 0 / 0 / 0 |
| 冒烟 preregister / run / verify | 0 / 0 / 0 |
| 正式 preregister | 0 |
| 正式 run（1 次） | 0 |
| 正式 verify（run 后 1 次、写日志后 1 次） | 0 / 0 |
| run 之后复跑单测 | 0 |
