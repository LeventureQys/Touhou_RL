# WP01.1a — CPU/GPU 与前向一致性检查

- 日期：2026-11-13（UTC 2026-11-12T16:20Z）
- 范围：只写代码与测试。未启动游戏、未训练、未采集、未 commit/push。
- 环境：Windows 11，Python 3.12.10，numpy 2.5.4，torch 2.14.1+cu132，pytest 9.1.1，
  GPU = NVIDIA GeForce RTX 4070 SUPER（CUDA 13.2，cuDNN 92400）。

## 1. 完成 / 未完成

**完成**。WP01.1a 要求的三条路径（离线 batch / 在线逐 decision / CPU vs CUDA）已在
真实 D1 样本与真实 `v3-bc1` checkpoint 上实际执行并通过。

| 要求 | 位置 | 证据（测试） |
| --- | --- | --- |
| `load_v3`：复用 `load_checkpoint_v3`，`eval()` + `requires_grad_(False)`，错误 fail-closed | `touhou_rl/v410/parity.py` | `test_load_v3_is_eval_and_frozen`、`test_load_v3_rejects_bad_checkpoints`（missing/empty/garbage/wrong_schema）、`test_load_v3_rejects_unknown_device` |
| `build_batch`：同一 v410 adapter，B=1 与 B>1，返回 play/play_norm/hud/act_hist | 同上 | `test_build_batch_matches_per_row_adapter`、`test_real_d1_row_adapter_is_byte_exact[0..3]` |
| `forward_logits`：hud/act_hist float 约定、no_grad、返回 move logits + argmax | 同上 | `test_forward_logits_returns_argmax_and_keeps_batch`、`test_forward_logits_rejects_*` |
| `offline_vs_online`：离线磁盘数组 vs 在线逐帧 adapter；CPU 上 max_abs ≤ 1e-5、argmax 全等、uint8 play 逐字节相等 | 同上 | `test_offline_vs_online_cpu_batch4`、`test_offline_vs_online_cpu_batch1`、`test_offline_vs_online_does_not_mutate_inputs` |
| `cpu_gpu_parity`：CPU/CUDA 前向，max_abs ≤ 1e-4、argmax 全等、B=1 与 B=4、设备检查；无 CUDA 只 skip 不伪造通过 | 同上 | `test_cpu_gpu_parity_batch1`、`test_cpu_gpu_parity_batch4`、`test_cpu_gpu_parity_skips_without_faking_a_pass`、`test_cpu_gpu_parity_rejects_cpu_device` |
| `write_report`：UTF-8 / LF / JSON | 同上 | `test_write_report_is_utf8_lf_json`、`test_write_report_rejects_unserialisable_payload` |
| 坏 checkpoint / 坏 shape 被拒绝 | 同上 | `test_load_v3_rejects_bad_checkpoints`、`test_build_batch_rejects_*`、`test_forward_logits_rejects_*` |
| 输入不被改写 | 同上 | `test_real_d1_row_adapter_is_byte_exact`、`test_offline_vs_online_does_not_mutate_inputs`、`test_forward_logits_returns_argmax_and_keeps_batch` |
| 证据落盘 | `evidence/WP01/parity.json` | `test_build_parity_payload_records_every_check` + 第 3 节 CLI（退出码 0） |

未做（白名单外）：未改 `touhou_rl/v410/adapters.py::assert_cpu_gpu_parity`（仍是 WP01.0 的
`NotImplementedError` 占位）。CPU/GPU 检查由新模块 `touhou_rl/v410/parity.py` 提供，
不冒充 adapter 里那个占位函数已实现。

## 2. 变更文件（严格白名单）

| 文件 | 说明 | 行数 |
| --- | --- | ---: |
| `touhou_rl/v410/parity.py` | 新增：`load_v3`、`build_batch`、`forward_logits`、`offline_vs_online`、`cpu_gpu_parity`、`rnn_state_report`、`load_obs_rows`、`write_report`、`build_parity_payload` 与 `python -m` 入口 | 1120 |
| `tests/test_v410_parity.py` | 新增：37 个测试（含 4 个参数化行 + 4 个坏 checkpoint 参数化） | 441 |
| `Document/Update/Alpha/v4.1.0/evidence/WP01/parity.json` | 新增：真实运行生成的 JSON 证据（9493 字节，UTF-8，LF） | — |
| `Document/Update/Alpha/v4.1.0/evidence/WP01/PARITY.md` | 新增：本报告 | — |

白名单外无改动。运行 `git status --porcelain` 检查时出现的 `third_party/th08`（submodule 脏）、
`tools/rl/v410.py`、`touhou_rl/v380/`、`Document/Update/Alpha/v4.1.0/state.json` 等均为
本次任务开始前既有的未跟踪/未提交状态，本切片未写入。

## 3. 实际命令与退出码

全部在 `D:\WorkShop\Touhou_RL` 下执行，`$env:PYTHONPATH='D:\WorkShop\Touhou_RL'`。

| # | 命令（简写） | 退出码 | 实际结果 |
| --- | --- | ---: | --- |
| 1 | `python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp011_pytest -q tests/test_v410_parity.py`（首跑） | **1** | `1 failed, 36 passed in 2.71s`；失败是测试自身断言过严（对 `load_v3("tpu")` 要求文案含 `supported`，实际 torch 先抛 `cannot parse device ...`）。修正匹配词并让 `_resolve_device` 的报错含 `device` 后重跑 |
| 2 | 同上（修正后，验收跑） | **0** | `37 passed in 2.63s` |
| 3 | `python -m compileall -q touhou_rl/v410` | **0** | 无输出（仅语法通过，不等于测试通过） |
| 4 | `python -m touhou_rl.v410.parity --checkpoint D:\WorkShop\Touhou_RL\.tmp_v400_p3v3\checkpoint-v3-bc1.pt --obs-dir D:\WorkShop\Touhou_RL\.tmp_v400_d1\d1\shards\d1-s0d1-n3371-seed1-shard\obs --rows 0,1,2,3 --out D:\WorkShop\Touhou_RL\Document\Update\Alpha\v4.1.0\evidence\WP01\parity.json` | **0** | `{"passed": true, "cuda_skipped": false, "offline_vs_online_batch1_max_abs": 0.0, "offline_vs_online_batchN_max_abs": 1.1920928955078125e-07, "cpu_vs_cuda_batch1_max_abs": 2.384185791015625e-07, "cpu_vs_cuda_batchN_max_abs": 1.1920928955078125e-07}` |
| 5 | `python -c` 校验 parity.json（UTF-8 解码、无 CRLF、末尾 LF、JSON 可解析、关键字段） | **0** | `bytes 9493 CRLF False ends_LF True`；`无 RNN 状态可检查` 可按 UTF-8 读出 |
| 6 | `python -m pytest --version` / `python -c "…numpy/torch…"` | **0** | `pytest 9.1.1`；`py 3.12.10 numpy 2.5.4 torch 2.14.1+cu132` |

## 4. pytest 统计

- 最终：**37 passed，0 failed，0 skipped，2.63s**（本机 CUDA 可用，两个 CPU/GPU 测试真实执行，不是 skip）。
- 分组：checkpoint/loader 7、adapter 5、offline-vs-online 4、CPU-vs-CUDA 4、
  fail-closed 输入 8、磁盘读取/RNN/报告写入 8、完整证据 payload 1。
- skip 契约：`test_cpu_gpu_parity_skips_without_faking_a_pass` 用 monkeypatch 把
  `torch.cuda.is_available` 置 False，断言返回 `{"skipped": true, "passed": false}` 且不含
  `max_abs`；有 CUDA 的两个测试则带 `skipif(reason="CUDA is not available; …no pass is claimed")`。

## 5. parity 关键数字

样本：`.tmp_v400_d1/d1/shards/d1-s0d1-n3371-seed1-shard/obs/` 的第 0–3 行，
`np.load(..., mmap_mode='r')` 只取所需行（`play96x112.npy` 145 MB 未整片载入）。

| 项 | 值 |
| --- | --- |
| checkpoint SHA-256 | `47308386756f1997ac63d97cdb9a9699fffe502a50fb4722d16123de0325a404` |
| checkpoint 字节 / 参数量 | 10,478,427 / 2,602,490（arch=v3，canonical=True） |
| play96x112.npy SHA-256 | `ff9cda704d2aa1a5beacdbd3071875ae5edeb52c57a2f16371e1cab01b5070e0` |
| hud.npy SHA-256 | `c8a165990256e30201fa7b20fb9ab4aaa85455f51dc9b4109a12abfbbd1c5a0c` |
| act_hist.npy SHA-256 | `e6f85f6f986cd852fb62e972f4be97abc77f35f693dd8ca51cff9e10522b91a6` |
| 离线 vs 在线，B=1，CPU | max_abs = **0.0**（≤1e-5），argmax 1/1 相等 |
| 离线 vs 在线，B=4，CPU | max_abs = **1.1920928955078125e-07**（≤1e-5），argmax 4/4 相等 |
| CPU vs CUDA，B=1 | max_abs = **2.384185791015625e-07**（≤1e-4），argmax 1/1 相等 |
| CPU vs CUDA，B=4 | max_abs = **1.1920928955078125e-07**（≤1e-4），argmax 4/4 相等 |
| CPU/GPU 权重 | 逐位相同（`weights_identical=true`，`weights_max_abs_diff=0.0`）；参数全在 `cpu` / `cuda`，输入张量也在请求设备上 |
| uint8 play / play_norm | 离线与在线都逐字节等于磁盘行；`play_norm == astype(float32)/255.0` 逐字节相等 |
| argmax（4 行） | 四条路径一致，均为 `[7, 7, 7, 7]` |
| skip | `cuda_skipped=false`（CUDA 真实执行）；无 CUDA 时只允许 skip+reason，不能算通过 |
| RNN | `present=false`，`modules=[]`，note：`StudentV3 has no RNN/LSTM/GRU module; 无 RNN 状态可检查` |
| 生成/耗时 | `generated_at=2026-11-12T16:20:17+00:00`，payload 构建 0.881 s；离线 B=4（adapter+forward）0.0144 s，在线 4 次（stack+adapter+B=1 forward）0.0191 s；CPU/GPU B=4 检查 0.1003 s |

## 6. 输入约定（读代码确认，非猜测）

- `SpatialPlayEncoder` 首层是 Conv2d：`play` 必须是浮点，走 `play_norm`（float32）。
- `HudEncoder` 首层是 Conv2d：`hud` 必须转 float，且按 `train_v3.batch_tensors` 的
  `float().div_(255.0).unsqueeze(1)` 变成 `(B,1,56,160)`；`parity.forward_logits` 用的正是这一配方。
- `ActionHistoryEncoder.forward` 自己调用 `.float()`；本模块直接传 float32 `(B,8,18)`。
- 所有前向在 `eval()` + `torch.no_grad()` 下执行（BatchNorm 用 running stats；`requires_grad`
  已清空），否则 batch/设备差异会掺入训练态行为，parity 数字没有意义。
- `StudentV3.forward` 返回 `StudentV3Output(move, bomb, value, aux)`；本切片只比较
  `move` logits（18 类），未比较 `bomb`/`value`/perception 头。

## 7. 限制

1. **样本切片**：只用了 1 个 shard 的第 0–3 行（共 3370 个 decision）。第 0 行是任务要求的
   真实样本，第 1–3 行只为把 B 扩到 4；这不是完整 D1 数据集，也不覆盖其它 stage/difficulty。
2. **单一 checkpoint**：仅 `checkpoint-v3-bc1.pt`；未覆盖 bc2 / full / 其它训练轮次。
3. **只比较 move logits**：bomb、value、perception aux 未纳入 parity；未比较概率/采样路径。
4. **CPU/GPU 为单次前向**：同一 checkpoint 两次加载（权重逐位相等已断言）、单卡 4070 SUPER、
   单次运行；没有重复 N 次统计 cuDNN 算法选择带来的波动，也没有吞吐/延迟基准。
5. **无 RNN 状态可检查**：`StudentV3` 不含 RNN/LSTM/GRU，故只检查无状态前向；将来若引入
   循环结构，需要新增 hidden-state 跨设备/跨路径比较（当前报告显式记录 `present=false`）。
6. **adapter 占位未动**：`touhou_rl/v410/adapters.py::assert_cpu_gpu_parity` 仍抛
   `NotImplementedError`（白名单禁止改该文件）；若验收要求该函数本身可用，需另行授权。
7. 未启动游戏、未训练、未采集、未 commit/push；`compileall` 退出码 0 只代表语法通过。

## 8. 剩余阻塞

无（WP01.1a 范围内）。CUDA 可用，CPU/GPU 检查为真实执行，`parity.json` 的 `passed=true`
不是 skip 伪造。

## 9. 下一建议

1. WP01.1b：把 `parity.forward_logits` 接入实时 loop 的 batch=1 推理路径，在真实决策节奏下
   验证离线/在线一致性（本切片只离线复现该路径）。
2. 扩充行覆盖（如分层抽取 32–64 行）并把 bomb/value 纳入比较，降低“4 行同 argmax”带来的
   统计偶然性。
3. 若需要 `adapters.assert_cpu_gpu_parity` 契约成立，申请把 `adapters.py` 加入白名单，让其
   委托到 `parity.cpu_gpu_parity`（当前未做，避免越界）。
4. 增加 GPU 侧重复运行统计（同一 batch 跑 N 次记录 max_abs 分布），为后续性能门留下基线。
