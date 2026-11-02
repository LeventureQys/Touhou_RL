# round21 命令、退出码与测试迭代（F1 危险预告）

本文件是 [P4-telegraph-round21.md](../../../reports/P4-telegraph-round21.md) 的证据附件。
本目录下的 `logs/` 与 `sha256.json` 的关系：`sha256.json` 在 `run` 结束时只覆盖
`preregistration.json` 与 `run/**`（124 个文件），`logs/` 与本文是 run 之后写的，**不在**该表内。

## 环境

| 项 | 值 |
|---|---|
| Python | 3.12.10 |
| torch | 2.14.1+cu132 |
| GPU | NVIDIA GeForce RTX 4070 SUPER（真实 CUDA，非模拟） |
| 精度 | FP32，无 AMP，`autocast` 关闭，`matmul.allow_tf32 = false`，`cudnn.allow_tf32 = false`，`float32_matmul_precision = "highest"` |
| 判定方式 | 三个 flag 在 `run` 内、任何数值工作之前断言为 false；`summary.environment.precision` 记录同一组值 |

## 命令与退出码（严格按执行顺序）

| # | 命令 | 退出码 | 说明 |
|---:|---|---:|---|
| 1 | `python -m pytest tests/test_v373_telegraph.py -q -p no:cacheprovider --basetemp .tmp/v373-telegraph-tests-01` | 1 | 首次收集即失败（本文件下方第 1 轮）；修复后重跑 |
| 2 | 同上（同一 basetemp，第 2 轮） | 1 | 5 failed / 22 passed |
| 3 | 同上（`…-tests-02`，只选 2 个渲染用例） | 1 | 2 failed / 25 passed |
| 4 | 同上（`…-tests-03`，全量） | 1 | 2 failed / 25 passed |
| 5 | 同上（`…-tests-04`，全量） | 1 | 1 failed / 26 passed |
| 6 | 同上（`…-tests-05`，全量） | 0 | **27 passed** |
| 7 | `python -m pytest tests/test_v373_valid_curriculum.py -q -p no:cacheprovider --basetemp .tmp/v373-r20-regression` | 0 | R20 回归：21 passed（旧模块未被改动） |
| 8 | `python -m tools.rl.v373_telegraph preregister --out .tmp/telegraph-smoke` | 0 | 冒烟预注册（scratch），无 stdout |
| 9 | `python -m tools.rl.v373_telegraph run --out .tmp/telegraph-smoke` | 0 | 冒烟整链（scratch），墙钟 265 s |
| 10 | `python -m tools.rl.v373_telegraph verify --out .tmp/telegraph-smoke` | 0 | 冒烟只读复核：122 产物 / 56 资产 / 6 checkpoint / 14 闭环文件 |
| 11 | `python -m tools.rl.v373_telegraph preregister` | 0 | **正式冻结**；无 stdout；`preregistration.json` SHA `09d440d5a3053ade919279f295391ab19fa5962231496cd0a774b32a0c92556d` |
| 12 | `python -m pytest tests/test_v373_telegraph.py -q -p no:cacheprovider --basetemp .tmp/v373-telegraph-tests-06` | 0 | 预注册后复跑冻结的测试文件：27 passed |
| 13 | `python -m tools.rl.v373_telegraph run` | 0 | **唯一一次正式 run**；墙钟 266.3 s（数据 16.638 s / encode 1.382 s / 训练 60.202 s / 闭环 159.771 s） |
| 14 | `python -m tools.rl.v373_telegraph verify` | 0 | 只读复核：122 run 产物 / 124 根产物 / 56 资产 / 6 checkpoint / 14 闭环文件 / 168 基线行 / 84 条 F1 缺口诊断，全部 SHA 一致 |
| 15 | `python -m pytest tests/test_v373_telegraph.py -q -p no:cacheprovider --basetemp .tmp/v373-telegraph-tests-final` | 0 | run 之后复跑：27 passed |

退出码序列（正式流程）：**0（预注册）/ 0（预注册后测试）/ 0（run）/ 0（verify）/ 0（run 后测试）**。

`preregister` 无 stdout，因此 `logs/preregister.log` 不存在（不是缺失日志，是该命令不打印任何内容）；
`logs/execution.log` 与 `logs/execution-exit.log` 是第 13 步的原样输出与退出码记录。

## 测试迭代（预注册之前的 6 轮，逐条归因）

| 轮 | 结果 | 失败项 | 归因 |
|---:|---|---|---|
| 1 | 收集失败 | `SyntaxError` | **测试自身写错**：`assert x, (a, b), 'msg'` 三表达式非法；改成 f-string 消息 |
| 2 | 5 failed / 22 passed | `test_standing_in_the_preview_band_is_not_a_collision` | **测试自身写错**：用 2 帧网格的 trace 按逻辑帧下标取值（`trace[38]` 实际是帧 76） |
| 2 | 同上 | `test_gap_trace_reconstructs_advance_semantics` | **测试自身写错**：断言 `len(trace) == len(expected) == 161`，而 `_replay` 是 81 条（每 2 帧） |
| 2 | 同上 | `test_closed_loop_runner_reproduces_the_frozen_runner` | **工具真实缺陷**：副本没有算 `environment_metrics`，于是 `compact()` 的 `verdict`/`reached_goal`/`final_horizontal_space_px` 与冻结 runner 不同；改为与 R11/R18 一样调用冻结的 `metrics(s, actions)` |
| 2 | 同上 | `test_gap_is_an_empty_slot_bounded_by_markers` | **测试自身写错**：整列断言没有排除玩家精灵遮挡 |
| 2 | 同上 | `test_preview_survives_the_model_input_resize` | **测试自身写错**：`SIZES['playfield'] = (224, 192)` 是 `(out_h, out_w)`，断言写成了 `(192, 224, 3)` |
| 3 | 2 failed / 25 passed | 上表后两项（已修） | — |
| 4 | 2 failed / 25 passed | `test_gap_is_an_empty_slot_bounded_by_markers`；`test_standing_in_the_preview_band_is_not_a_collision` | 前者：镜像后 span 顺序按 `render_state` 的 span 顺序保留、不是排序后的顺序（测试期望写错）；后者：`hold` 轨迹只有偶数帧，取奇数帧（31）触发 `KeyError`（测试自身写错） |
| 5 | 1 failed / 26 passed | `test_gap_is_an_empty_slot_bounded_by_markers` | 上一轮修改时旧断言未删除（测试自身写错） |
| 6 | **27 passed** | — | 预注册前的最后状态 |

另外，在写测试之前的一次**渲染探针**发现了本轮唯一一个**在工具里**的真实缺陷：
`region[:, selected, :]` 是高级索引，返回副本，因此第一版 `render_state_telegraph`
一个像素都没画（`changed 0`）。修法是显式写回 `region[:, selected, :] = sub`。
这个缺陷在正式预注册之前就被 6 个训练事件的 diff 计数（22/22/20/20/22/22 帧）抓住了。

## 冒烟运行（预先披露）

预注册之前用**同一份冻结代码**在 `.tmp/telegraph-smoke/` 跑过一次完整流水线（第 8–10 步）。
它只用于证明产物能被写全、并提前发现真实缺陷；**它的数值不作为本轮任何结论的依据**。

冒烟与正式 run 的一致性（本轮因此可以声明"两次独立进程逐位相同"）：

| 比较项 | 结果 |
|---|---|
| 6 个 checkpoint 文件 SHA | 全部相同 |
| 6 个末态 head state SHA | 全部相同 |
| 14 个资产（pixels/history/labels/rollout）SHA | 全部相同 |
| validation / train 的 14×7 条闭环结果 | 全部相同 |
| 行为占比、切换数 | 全部相同 |
| F1 缺口诊断（84 行） | 全部相同 |
| B 臂锚末值 | 全部相同 |

## 产物清单（`verify` 的读数）

```json
{"artifacts": 122, "root_artifacts": 124, "assets": 56, "checkpoints": 6, "closed_loop_files": 14,
 "baseline_rows": 168, "baseline_passing_events_validation": [], "holdout_filter_valid": true,
 "f1_gap_rows": 84, "render_freeze_sha256": "1e3d40e7…", "physics_sha256": "ebd9e88d…",
 "train_events": 14, "validation_events": 14, "criterion_sha256": "eaa7a905…", "identities": 24}
```

## 没有做的事（与预注册一致）

- 没有改物理（`advance` 未被触碰，物理探针 SHA 不变）、没有改世界参数、没有改 R19 split、
  没有改教师、没有重新规划教师路线、没有改标签与打包。
- 没有调参：lr / clip / updates / batch / 权重 / 顺序 / 种子全部走 R20 的冻结代码路径。
- 没有早停、没有按 validation 选模、没有重跑、没有换基线。
- 没有开正式门、没有可部署 checkpoint、没有运行游戏。
