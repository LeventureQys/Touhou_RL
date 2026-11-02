# round24 命令与退出码（command-results）

所有命令都在仓库根 `D:\WorkShop\Touhou_RL` 下执行，解释器为 `python`（torch 2.14.1+cu132，
RTX 4070 SUPER，FP32、`matmul_tf32=False`、`cudnn_tf32=False`、`float32_matmul_precision=highest`）。

## 1) 单测（写工具期间共 7 轮）

```powershell
python -m pytest tests/test_v373_player_relative_aux.py -q -p no:cacheprovider --basetemp .tmp/v373-prelaux-tests-1
# 5 failed, 22 passed  -> 逐条归因见下
python -m pytest tests/test_v373_player_relative_aux.py -q -p no:cacheprovider --basetemp .tmp/v373-prelaux-tests-5
# 28 passed in 3.27s, exit 0
```

测试迭代里修掉的 **4 个真实缺陷**（都在本轮自己的代码/测试里，旧模块一行未动）：

1. `judgement()` 在 `replaces` 字段里递归调用自己 → `RecursionError`。改为纯函数
   `rule_verdict(higher, lower, k1, k2, k2_all)`。
2. `train_all` 用 `targets[event_id]` 取目标，而 `targets` 只含有标签的 6 个 F1 事件，
   F2/F3 的 8 个事件会 `KeyError`（会在数据准备完成后、开始训练时炸）。改为
   `targets.get(event_id, (None, None))`。
3. `agreement_table` 把**符号**（±1）与**类别**（0/1）直接比较，
   于是"标签 left、教师在锚定行投 −2"被记成 `disagree`。改为 `action_match(action, sign)`，
   之后 train 4 agree / 0 disagree / 2 hold、validation 2 agree / 0 disagree / 4 hold。
   **这个缺陷是在第一次正式 run 已经开跑之后发现的，因此那次 run 被主动中止并整目录删除**（见 §4）。
4. 测试自身 4 处符号/类别混用的断言（`per_row_signs[0]` 与 `target_side`、
   `007-m1` 的正确标签方向、`judgement([1,1,1],[2,2,0])` 的 K1 期望值、以及假 probes 缺字段）。

## 2) 旧轮回归（旧模块未被改动）

```powershell
python -m pytest tests/test_v373_privileged_aux.py -q -p no:cacheprovider --basetemp .tmp/v373-r23-regression-r24
# 27 passed in 4.28s, exit 0   -> logs/regression-round23.log
python -m pytest tests/test_v373_spatial_fuse.py -q -p no:cacheprovider --basetemp .tmp/v373-r22-regression-r24
# 22 passed in 6.12s, exit 0   -> logs/regression-round22.log
python -m pytest tests/test_v373_telegraph.py -q -p no:cacheprovider --basetemp .tmp/v373-r21-regression-r24
# 27 passed in 15.36s, exit 0  -> logs/regression-round21.log
python -m pytest tests/test_v373_valid_curriculum.py -q -p no:cacheprovider --basetemp .tmp/v373-r20-regression-r24
# 21 passed in 19.04s, exit 0  -> logs/regression-round20.log
```

## 3) 冒烟（scratch，**预先披露**）

预注册之前在 `.tmp/prelaux-smoke/` 用同一份冻结代码跑过一次**完整**流水线（preregister → run →
verify，一次通过，exit 0，约 300 s），其数值不用于任何结论。它与正式 run 的逐字节比较：

- 两个目录各 **68** 个文件，**45 个逐字节相同**，其中包含**全部 6 个 `research-heads.pt`、
  14 个闭环文件、`baselines.json`、`f1-gap.json`、`records.json`**；
- 23 个不同，全部可归因：`timings.json`、`checkpoints.json`、`train/*/training.json`（含墙钟秒数）、
  `events/*.json`（含 `agreement` 表）、`probes.json` / `summary.json`（含 `agreement` 表与
  本次新增的 `overall_vs_frozen_teacher`、`label_tie`、`own_first_tie` 字段）、`sha256.json`（因上述而变）。

因此"两个独立进程给出逐位相同的 checkpoint 与闭环"在本轮是被测出来的，而不是声明的。

## 4) 第一次正式 run：**主动中止**

```powershell
python -m tools.rl.v373_player_relative_aux preregister           # exit 0（prereg SHA 76f9a659…）
python -m tools.rl.v373_player_relative_aux run                   # 已跑到训练第 1 个 checkpoint 后被 kill
```

中止理由（缺陷 3）：预注册里冻结的 `agreement` 表把符号当类别比较，`anchored_action` 记成
train 2 agree / 2 disagree、validation 1 agree / 1 disagree，而正确读数是 4/0/2 与 2/0/4。
这个错误会**写进证据与报告**，所以不能在它上面继续。处理：
`job_kill` → 删除 `evidence/prelaux-round24/`（**已验证绝对路径为
`D:\WorkShop\Touhou_RL\Document\Update\Alpha\v3.7.3\evidence\prelaux-round24`**，当时只有 6 个文件、
无 checkpoint）→ 修 `action_match` → 复跑单测 → 重新冻结预注册 → 重新执行唯一一次正式 run。
被中止那次的日志留在 `logs/aborted-run-1.log`（436 B，只到 `A-s7` 训练完成）以保留失败痕迹。

## 5) 正式冻结预注册

```powershell
python -m tools.rl.v373_player_relative_aux preregister           # exit 0，无 stdout
```

- `criterion_sha256 = 6a2c5014f60d3d088502dab84d3ebcad958f2dd93a61b138ce64496e00e21bf9`
- `preregistration.json` sha256 = `166bc70ed1c0ee8f6ed9d1a32bed63cf890887998e56271da538d16cfa4f0041`
  （197 840 B，**57 个身份 SHA**）
- 见 `logs/preregister.log`（该命令成功时不打印任何内容，日志里记录了冻结值）。

## 6) 唯一一次正式 run（后台 job，自行收取）

```powershell
python -m tools.rl.v373_player_relative_aux run                   # exit 0，墙钟 ≈ 258 s
# data 1.288 s / 训练 66.285 s（其中纯优化 62.846 s）/ 闭环 158.044 s / 探针 31.799 s
```

`logs/execution.log`（36 006 B）。6 个 checkpoint 的初始/末 head SHA 见报告 §训练。

## 7) 只读复核

```powershell
python -m tools.rl.v373_player_relative_aux verify                # exit 0
# 67 run 产物 / 69 根产物 / 56 资产 / 6 checkpoint / 14 闭环文件 / 28 事件文件
# 168 基线行 / 84 条 F1 机制行 / 57 个身份 SHA / aux 参数 514 / aux 权重 0.5
# K1 = A 2/2/2 对 X 3/3/3、K2 F1 +3 / 合计 0、verdict = improved
```

`logs/verify.log`。写入 §3–§7 的日志之后**再跑一次 verify 仍 exit 0**（日志文件在 run 之后写入，
因此不在根 `sha256.json` 表内，与 R22/R23 的做法一致）。

## 8) run 之后复跑冻结测试

```powershell
python -m pytest tests/test_v373_player_relative_aux.py -q -p no:cacheprovider --basetemp .tmp/v373-prelaux-tests-final
# 28 passed in 3.12s, exit 0   -> logs/tests-after-run.log
```

退出码依次 **0 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0**（4 个旧轮回归 + 正式 preregister + 正式 run +
正式 verify + run 后测试 + 中止的那次 run 之外的复跑），另有 1 次被主动中止的正式 run（未计为完成）。
