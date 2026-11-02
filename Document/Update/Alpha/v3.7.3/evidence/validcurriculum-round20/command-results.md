# round20 命令、退出码与日志

工作目录固定为仓库根 `D:\WorkShop\Touhou_RL`；`PYTHONPATH` 指向仓库根。
执行顺序就是本文件自上而下的顺序：**测试 → 预注册 → 测试 → 唯一一次 run → 只读复核**。

| # | 命令 | 结果 | 退出码 | 日志 |
|---|---|---|---:|---|
| 1 | `python -m pytest tests/test_v373_valid_curriculum.py -q -p no:cacheprovider --basetemp .tmp/v373-validcurriculum-round20-tests-01` | `21 passed in 18.94s` | **0** | [logs/tests-first.log](logs/tests-first.log) |
| 2 | `python -m tools.rl.v373_valid_curriculum preregister` | 冻结 `preregistration.json`（criterion SHA `377bcbf8…`，22 个身份 SHA），**无 stdout** | **0** | [logs/preregister.log](logs/preregister.log)（空文件，进程不打印） |
| 3 | `python -m pytest tests/test_v373_valid_curriculum.py -q -p no:cacheprovider --basetemp .tmp/v373-validcurriculum-round20-tests-02` | `21 passed in 18.84s`（冻结的测试文件预注册后仍通过） | **0** | [logs/tests-after-preregister.log](logs/tests-after-preregister.log) |
| 4 | `python -m tools.rl.v373_valid_curriculum run` | 后台 job 一次跑完：14 个事件渲染 → 共享 encode → 6 个 checkpoint → 168 行 R18 基线 → 196 条闭环 → summary/sha256 | **0** | [logs/execution.log](logs/execution.log)、[logs/execution-exit.log](logs/execution-exit.log) |
| 5 | `python -m tools.rl.v373_valid_curriculum verify` | 只读复核：121 个 run 产物 / 56 个资产 / 6 个 checkpoint / 14 个闭环文件 / 123 个根目录产物 SHA 全部一致 | **0** | [logs/verify.log](logs/verify.log) |

一次 `run` 的实测钟表时间 **223.8 s**（`execution-exit.log`）。工单里的四段耗时来自
[run/timings.json](run/timings.json)：数据 9.29 s、encode 1.39 s、训练 62.12 s、闭环评估 145.49 s。

## 测试阶段的迭代（保留，不粉饰）

第一次运行同一测试文件时是 **5 failed / 16 passed**。逐条归因：

| 失败 | 归因 | 处置 |
|---|---|---|
| `vc.SUPPORTED` 不存在 | **工具的真实缺陷**：重写模块时漏了 `SUPPORTED = tuple(sorted(HORIZONTAL))` 常量 | 在工具里补上 |
| `verify() takes from 0 to 1 positional arguments but 2 were given` | **工具的真实缺陷**：模块内新加的只读复核函数 `verify(out)` 覆盖了从 `tactical_teacher_validation` 导入的 `verify(world, actions)` | 把只读复核改名为 `verify_artifacts` |
| `kl_anchor` 在"策略=底座"下为 0 以外还断言了 `>0` | 测试自身写错：给底座加了**常数**偏移，常数偏移不改变 softmax，KL 必然为 0 | 改成在某个**类上**偏移 |
| `uint8/255` 往返逐位不等 | 测试自身写错：对照用 `x/255`（除法），实现用 `mul_(1/255)`，float32 舍入不同 | 对照改成容差 1e-6 |
| `depth_frames[-1]` 期望 `[156,158,160]` | 测试自身写错：第 80 个决策是 f=158，末帧应是 `[154,156,158]` | 改正期望 |

两个工具缺陷都在**预注册之前**被测试抓住并修好，因此冻结进预注册的分类器/渲染/封装代码里没有这两个缺陷。
测试修复过程中还写过两处临时占位断言（`k16_anchor_guard`、一条无信息的 `offline` 断言），已删除。
合计：**2 个工具真实缺陷 + 3 处测试自身写错**。
上面第 1 行的日志是通过那次（`21 passed`）；早期失败运行的日志已被同名 basetemp 覆盖，只在这里如实记录数量与归因。

## 一次性的边界

- `run` **只跑一次**：`exist_ok=False` 建目录、JSON 用 `x`、权重用 `xb`，第二次执行会直接失败而不是覆盖。
- 没有 rerun、没有 repair pass、没有补跑某个事件、没有改判据/种子/超参/权重。
- 预注册之前，在 `.tmp/v373-validcurriculum-smoke/` 用**同一份冻结代码**跑过一次完整冒烟（把产物写到 scratch，
  不进 evidence）。它只用于证明流水线能把所有产物写全；其数值不被本轮任何结论引用，冒烟与正式 run
  之间没有改动任何一行工具、测试、判据、种子或权重（工具 SHA `42c0d16a…` 两次相同，6 个 checkpoint 的
  SHA 两次也逐位相同，说明 CUDA 上是确定性的）。冒烟产物留在 `.tmp/`，未删除。
- 证据目录里 `sha256.json`（根）覆盖 123 个文件（`preregistration.json` + `run/` 全部产物）。
  本文件与 `logs/` 是 run 之后写的文档，**不在该表内**，也不被任何数值结论引用。
