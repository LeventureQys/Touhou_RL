# round22 执行记录（命令、退出码、测试迭代与真实缺陷）

本文件在正式 run 之后写入，因此不在 [sha256.json](sha256.json) / [run/sha256.json](run/sha256.json) 的 SHA 表内；
写入后重跑 `python -m tools.rl.v373_spatial_fuse verify` 仍为 exit 0（见 [logs/verify.log](logs/verify.log)）。

## 1. 命令与退出码（按执行顺序）

```powershell
# 写工具/测试期间的迭代（15 轮，见第 2 节）；最终一轮：
python -m pytest tests/test_v373_spatial_fuse.py -q -p no:cacheprovider --basetemp .tmp/v373-spatialfuse-tests-15
#   22 passed in 6.09s                                                 exit 0

# 旧轮回归（证明旧模块没被改动）
python -m pytest tests/test_v373_telegraph.py -q -p no:cacheprovider --basetemp .tmp/v373-r21-regression-r22
#   27 passed in 15.30s                                                exit 0
python -m pytest tests/test_v373_valid_curriculum.py -q -p no:cacheprovider --basetemp .tmp/v373-r20-regression-r22
#   21 passed in 18.87s                                                exit 0

# 预先披露的冒烟（scratch，共 5 轮；这里只记最后一轮）
python -m tools.rl.v373_spatial_fuse preregister --out .tmp/spatialfuse-smoke     # exit 0
python -m tools.rl.v373_spatial_fuse run --out .tmp/spatialfuse-smoke             # exit 0
python -m tools.rl.v373_spatial_fuse verify --out .tmp/spatialfuse-smoke          # exit 0

# 正式冻结预注册（无 stdout）
python -m tools.rl.v373_spatial_fuse preregister
#   exit 0；criterion SHA 8ff1f14b6c071cbe63b2eae9d153596ab1ba667d6779bea5dc72f3d487681297
#   preregistration.json SHA 9eed34e19aeaa510af65c2b0d703960d7cdbf567b6e5f31bca0aa0ac419185d7（64 553 B）

# 唯一一次正式 run（后台 job，自行收取）
python -m tools.rl.v373_spatial_fuse run
#   exit 0；墙钟 297.3 s
#   timings: data 1.355 s / training_loop 69.016 s（6 个 head 的纯优化 65.697 s）/ evaluation 168.614 s / probes 58.634 s

# 只读复核
python -m tools.rl.v373_spatial_fuse verify
#   exit 0；artifacts 81 / root_artifacts 83 / assets 70 / checkpoints 6 / closed_loop 14 /
#   events 28 / baseline_rows 168 / f1_gap_rows 84 / identities 33 / holdout_filter_valid true

# run 之后复跑冻结的测试
python -m pytest tests/test_v373_spatial_fuse.py -q -p no:cacheprovider --basetemp .tmp/v373-spatialfuse-tests-final
#   22 passed in 6.11s                                                 exit 0

# 写入日志后再复核一次（证明运行时新增文件不会破坏 SHA 表）
python -m tools.rl.v373_spatial_fuse verify                            # exit 0
```

`logs/` 内容：`preregister.log`（空，命令无 stdout）、`execution.log`（run 的完整 stdout，33 892 B）、`verify.log`、`tests-after-run.log`、`regression-round20.log`、`regression-round21.log`。

## 2. 测试迭代（15 轮）与其中的真实缺陷

每一轮都是 `python -m pytest tests/test_v373_spatial_fuse.py -q -p no:cacheprovider --basetemp .tmp/v373-spatialfuse-tests-NN`。
前 14 轮全部被我用来**造出并修掉真实缺陷**，不是格式调整：

| 轮 | 结果 | 真实缺陷 / 归因 |
|---|---|---|
| 01 | 8 failed / 13 passed | ① `spatial_vectors` 拿 `f4.shape[1:]`（含通道）比 `(56,48)` → 一律报 `frozen spatial map shape`；② `direction_counts` 迭代 dict 的 **键**（字符串）而不是 `values()` → `TypeError`；③ 测试里用 `r20.bos_history`（不存在，它在 `tactical_policy`）；④ 父类 detach 的单测期望 `0.0`，实际是 **`None`**（更彻底的"无梯度"）；⑤ 零参数 `projection(zeros)` 在训练后不为 0（bias 已动）——断言写错了 |
| 02 | 8 failed / 13 passed | 同上定位完成；新增：`signed_error` 的符号一致性断言写错（`value*distance >= 0` 恒不成立）、几何表比对用了 `frozen['train']`（无 `rollout` 字段） |
| 03 | 2 failed / 19 passed | ⑥ **真实设计缺陷**：`spatial_probe` 的 λ 扫描里把"目标向量"当成"特征矩阵"喂给 `ridge_predict` |
| 04 | 2 failed / 19 passed | ⑦ R21 的 `telegraph.window` 是**半开区间** `[31,76)`，我的 `preview` 是**闭区间** `[31,75]` → 统一成 `preview_window`（半开）+ `last_preview_frame` |
| 05 | 2 failed / 19 passed | ⑧ 植信号的 ridge 单测在 `p≈n` 下被 93 个噪声维压死（标准化把 `std=1e-3` 的噪声放大到单位方差）→ 单测只用 1 个有效维（其余维严格常数），把"探针机理"与"p≫n 的真实困难"分开 |
| 06 | 1 failed / 20 passed | ⑨ `decisions_in_preview` 对 `wall_frame=40` 的 `F1-006` 是 **20** 不是 22（预告窗口被 0 截断） |
| 07 | 21 passed | 全绿 |
| 08–10 | 21 passed（含重构） | ⑩ **真实缺陷**：`evaluate_all` 用 `r20.all_names()`（返回 R20 的 A/B 两臂）→ 闭环在 `B-s7` 上 `KeyError`；改为本轮自己的 `all_names()`（A/S × 3 seeds） |
| 11 | 2 failed / 19 passed | ⑪ 改 `all_names()` 时把 `r21_reference()` 的"六头方向计数"也换了 → R21 参照漂移；修法是显式用 R20 的 `ARMS`/`SEEDS` 构造 R21 的名字 |
| 12 | 21 passed | 全绿 |
| 13 | 1 failed / 21 passed | ⑫ **真实缺陷（探针）**：`replay_images` 多放了一张初始帧 → 输入镜像探针的每一栈整体**错位一格**；新增单测 `test_the_probe_rebuilds_the_runner_inputs_exactly`（跑一次真实 B0 闭环再重放，要求 **80/80** 逐决策复现，而不是"大多数"）。第一版该单测还漏了 `render=r21.render_state_telegraph`，于是"跑的是普通渲染器、重放的是 telegraph"→ 73/80，正好把这个问题也照出来了 |
| 14 | 22 passed | 修正渲染器参数后 **80/80** |
| 15 | 22 passed | 最终（criterion 文本 + 逐 seed 相同 first-hit 统计 + delta on/off 消融 + 条件翻转率都并入后复跑） |

正式 run 之前还做了 5 轮完整冒烟，其中 2 轮抓到 **run 路径**上的真实缺陷：

| 冒烟轮 | 缺陷 |
|---|---|
| 1 | 无（但暴露 `evaluate_all` 的 `B-s7` KeyError，见上 ⑩） |
| 2 | ⑬ **真实缺陷**：`inference_mode` 里创建的空间向量是 **inference tensor**，被 `Linear` 当 autograd 输入时抛 `Inference tensors cannot be saved for backward` → 在 `inference_mode(False)` 里 `clone()` 成普通 tensor（并加断言 `is_inference() == False`） |
| 3 | ⑭ **真实缺陷**：`fusion_diagnostics` 把 CPU 张量喂给 CUDA 上的融合层 → 设备不匹配；按权重的 device 搬移 |
| 4 | 无（得到全部机制读数；发现 flipping 探针的 239/240 保真） |
| 5 | 无；与正式 run 的产物逐字节比对通过（见下） |

## 3. 冒烟 vs 正式 run：逐字节对照

同一份冻结代码、两次独立进程：

| 产物 | 是否逐字节相同 |
|---|---|
| 6 个 `research-heads.pt` | **相同** |
| 14 个 `closed-loop/*.json` | **相同** |
| `f1-gap.json` / `probes.json` / `records.json` / `baselines.json` | **相同** |
| 14 个 `spatial/*.npz` | **相同** |
| `checkpoints.json` / `timings.json` | 不同（只含墙钟秒数） |

因此本轮可以声明"两次独立进程逐位相同"，且冒烟的数值没有用于任何结论。

## 4. 与预注册的一致性（`verify` 每次重算的东西）

- 33 个身份 SHA（含本轮工具与测试自身的 SHA）：一致，否则拒绝执行；
- `criterion` SHA：`8ff1f14b…`；
- worlds digest（train / validation / 合计）与 R20/R21 相同；
- **94 维空间向量**：重跑 14 事件编码后 SHA 仍为 `83a75f73…`，逐事件 min/max/mean/std 与预注册完全相同；
- 28 条几何（gap / gap_screen / preview_window / decisions_in_preview）；
- R21 参照记录（逐 seed F1 留出、六头方向计数 15/3/0 与 3/0/15、A 的三个末态 head SHA）；
- 初值：每 seed head SHA + 两臂融合层 SHA（`64a8389b…`，两臂相同）。
