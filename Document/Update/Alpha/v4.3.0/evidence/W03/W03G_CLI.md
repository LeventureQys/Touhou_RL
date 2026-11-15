# W03G — `train-bc`：把 `learn.train_bc` 接进 CLI，产出可审计的训练结果

日期：2026-11-15。执行代理（DeepSeek Flash）。范围：**小切片**，只做「子命令 + 合成批次 + 结果 JSON + 测试」。
**合成数据，不是游戏或能力证据。** 全程未启动引擎、未读取任何 `.tmp_*` 语料、未跑评测。

## 1 目标

`touhou_rl/v430/learn.train_bc()` 已经完成，但只能被 Python 代码调用；整条流程无法被命令行驱动与审计。
本切片在 `tools/rl/v430.py` **新增** `train-bc` 子命令：读一个合成批次规格文件，训练一次 O2 actor，
把结果写成一份**正式的培训结果 JSON**（写盘前必须通过 `contracts.assert_result_schema`），并如实标记
`dataset_kind="synthetic"`、`capability_gate="not_run"`。

## 2 改动（只增，不改既有三个子命令）

| 位置 | 内容 |
| --- | --- |
| `tools/rl/v430.py` | 新增 `train-bc` 段：`load_batches_spec` / `synthetic_batches` / `batches_signature` / `TrainBCRun` / `cmd_train_bc` |
| 同上 | `build_parser()` 追加 `train-bc` 子解析器（**不改** `audit` / `validate-contract` / `validate-splits` 的参数与 handler） |
| 同上 | 顶部只追加 `import math`（`load_batches_spec` 的有限性检查用），其余 import 未动 |
| `tests/test_v430_learn.py` | **增量追加** 7 条 W03G 测试（文件结尾）；未改动既有 17 条 |
| `Document/Update/Alpha/v4.3.0/evidence/W03/train-bc-001.json` | 真实命令产物（唯一一个结果 JSON） |
| `Document/Update/Alpha/v4.3.0/evidence/W03/EXIT_CODES_W03G.txt` | 命令与真实退出码 |

`RESERVED` 表**未改**：`train` 仍然是 reserved（fail-closed）。argparse 的子命令精确名优先，
`train` 仍解析到 reserved 解析器，`train-bc` 解析到新解析器（`train-` 等前缀才会触发歧义）。

## 3 `--batches` 格式（在 `load_batches_spec` 的 docstring 里定义）

```json
{
  "schema": "v430-synthetic-batches/1",
  "n_batches": 4,
  "batch_size": 2,
  "seed": 1234,
  "steps": 1,
  "elapsed_capture_s": 0.016,
  "observation_shapes": {
    "global_rgb": [3, 240, 320],
    "play_gray": [4, 224, 192],
    "hud_gray": [1, 448, 160],
    "executed_key_history": [8, 7],
    "history_valid": [8]
  }
}
```

| 键 | 必需 | 语义 |
| --- | --- | --- |
| `schema` | 否（有则必须相等） | `v430-synthetic-batches/1` |
| `n_batches` | **是** | 1..256 |
| `batch_size` | **是** | 1..64 |
| `seed` | **是** | 任意 int；喂给 `torch.Generator(device="cpu")` |
| `steps` | 否（默认 1） | 1..16；`gru` 的序列长度，`none` 必须为 1 |
| `elapsed_capture_s` | 否（默认 0.016） | 有限且 > 0 |
| `observation_shapes` | 否 | **声明而非覆盖**：逐项与 `observation.O2_SHAPES` 比对，一个不等就拒绝 |

生成规则（CLI 内部，无任何磁盘语料）：

- `global_rgb 3×240×320`、`play_gray 4×224×192`、`hud_gray 1×448×160` 一律 **uint8**（`0..255`）；
- `executed_key_history 8×7` 为 uint8 `0/1`，`history_valid 8` 为全 `False`（冷启动），
  `elapsed_capture_s` 为每样本一个 float32 区间；
- 标签为四个 head 的 int64 索引（`HEAD_SIZES` 内）；
- 所有张量都来自**同一个**显式 generator，全局 RNG 不参与 → 同 `seed` 逐位相同。

**uint8 是硬约束**：模型 `_check_field` 对浮点图像会报
「`a float image has already been normalised — /255 belongs inside the model`」，
CLI 不做任何 `/255`，也不把 uint8 改成 float 来“跑得快”。

## 4 结果 JSON（`train-bc-001.json`）

写盘前 `TrainBCRun.execute()` 与 `write_checked_result()` **各校验一次** `contracts.assert_result_schema`；
`--out` 用 `emit_result` 的 `"x"` 模式创建，已存在即拒绝。

| 字段 | 值（本次真实产物） |
| --- | --- |
| `schema` | `v430-train-bc/1` |
| `command` / `command_class` | `train-bc` / **`training`** |
| `execution_status` / `protocol_validity` / `exit_code` | `ok` / `valid` / `0` |
| `capability_gate` | **`not_run`** |
| `dataset_kind` | **`synthetic`** |
| `memory_mode` | `none` |
| `n_steps` / `epochs` / `lr` / `seed` | `4` / `1` / `0.001` / `7` |
| `shuffle_batches` / `max_steps` | `false` / `4` |
| `losses` | `[4.36763334274292, 4.401745796203613, 4.949324131011963, 4.736083984375]` |
| `steps` | 每步 `{step, epoch, loss, per_head_loss{move,focus,shoot,bomb}, n_samples}` |
| `o2_digest` | `cb1616e4cbfa928e691c2fb8497a84c813bded528064cf2b0aa1e30fa86b85b5` |
| `parameter_report` / `trainable_parameter_count` | 全量报告 / **2726671**（`within_parameter_range=true`） |
| `batches.signature_sha256` | `dc39ca75e13ec2cf6d1c8e18895dc4ee0bdb7684013ec8441dc0cd44e14e660f` |
| `batches.sha256` | 规格文件自身的 sha256 |
| `research_result.status` | `not_run`（训练命令不产出研究结论） |
| `determinism` | `torch_num_threads=1`、`global_seed_for_initial_weights=7`、`global_rng_used_for_training=false` |
| `identities` | CLI 自身 sha256、规格 sha256、`o2_digest`、contract/spec 版本、runtime |
| `checkpoint` | `null`（本次未给 `--checkpoint-out`） |
| `remaining_gaps` | 4 条：合成数据、无能力判定、未重载 checkpoint、未跑 episode |

## 5 确定性（同 seed 逐位相同）

两个**独立进程**、同参数（`seed 7`、`max-steps 4`）：

```
A: [4.36763334274292, 4.401745796203613, 4.949324131011963, 4.736083984375]
B: [4.36763334274292, 4.401745796203613, 4.949324131011963, 4.736083984375]
identical = True
batches signature: 两者均为 dc39ca75…e660f
```

两点让这件事成立，并且都写进结果：

1. **批次的位级一致**：只有一个 `torch.Generator(device="cpu").manual_seed(spec.seed)`，全局 RNG 不参与；
2. **初始权重是 seed 的函数**：`build_actor()` 前显式 `torch.manual_seed(--seed)`。

调试记录（诚实记录一次失败）：最初**跨进程**第 2 步 loss 末位不同（`4.401745796203613` vs `4.401745319366455`），
原因是 CPU 多线程归约顺序；`--batches` 的位级一致（signature 相同）已排除数据因素。
CLI 现在固定 `torch.set_num_threads(1)` 后才达成跨进程逐位一致，该值记录在 `result["determinism"]`。
注意：这会在**进程内**改变 torch 线程数（pytest 进程同样受影响），是刻意的确定性取舍。

## 6 fail-closed 矩阵（全部真实执行）

| 场景 | 退出码 | 副作用 |
| --- | --- | --- |
| 合法运行 | **0** | 只写 `--out` |
| `--out` 已存在 | **2** | 文件逐字节不变 |
| `--checkpoint-out` 已存在 | **2** | checkpoint 逐字节不变，且不写 `--out` |
| `--memory-mode lstm` | **2** | 无文件 |
| 规格缺 `n_batches` | **2** | 无文件 |
| 规格 `observation_shapes.play_gray=[4,224,128]` | **2** | 无文件 |
| `steps=2` + `--memory-mode none` | **2** | 无文件（提示需要 `gru`） |

非法参数/非法批次/`LearnError` 一律 exit 2；只有真实程序异常才会落到 exit 1。

## 7 防伪闸门（实测报文）

`contracts` 的闸门在 `assert_result_schema`，作用是「训练命令不得自我晋升为能力」。
用**真实产物** `train-bc-001.json` 做三次伪造，并做一次对照：

```
a) 只把 capability_gate 改成 "pass"：
   ContractError: capability_gate='pass' requires a capability_judgement record
   from an independent judge; nothing may promote itself

b) 再加一份完整的伪造 capability_judgement（judge + 64 位 sha256）：
   ContractError: a 'training' command may not report capability_gate='pass';
   training exit 0 is not a capability

c) contracts.new_result(..., capability_gate="pass")：
   ContractError: capability_gate='pass' cannot be set by a result constructor;
   only an independent judgement may set it (set_capability_gate with evidence),
   and an exit-0 command does not promote a capability

d) 对照：同一份伪造文档、command_class 改成 "evaluation" → 通过校验。
   这证明 (b) 的拒绝原因是 command_class='training'，不是 judgement 格式不合法。
```

因此 CLI 结构上不可能把 `capability_gate` 写成 `pass`：构造函数拒绝，写盘守卫拒绝，
就算有人手改 JSON 也必须先改 `command_class`——那就不是训练结果了。

## 8 测试（增量追加 7 条，17 → 24 passed）

`tests/test_v430_learn.py` 末尾新增：

1. `test_train_bc_writes_a_schema_valid_training_result` — exit 0、文件存在、
   `command_class="training"`、`capability_gate="not_run"`、`dataset_kind="synthetic"`、
   `o2_digest`/参数量/`losses` 长度自洽，并再喂一次 `assert_result_schema`；
2. `test_train_bc_same_seed_reproduces_the_loss_sequence` — 同 seed 两次，`losses`、`steps`、
   `batches.signature_sha256` 全等；
3. `test_train_bc_refuses_an_existing_out_and_leaves_it_byte_identical` — exit 2 + 逐字节不变；
4. `test_train_bc_refuses_an_unregistered_memory_mode` — exit 2 + 无输出文件；
5. `test_train_bc_refuses_an_invalid_batches_spec` — 缺字段、非 O2 形状，两条均 exit 2；
6. `test_a_training_result_cannot_promote_itself_to_a_capability` — 裸改 `pass` 被拒 +
   完整伪造被拒（断言报文含 `training exit 0 is not a capability`）+ 构造函数被拒 +
   `evaluation` 对照通过；
7. `test_train_bc_refuses_an_existing_checkpoint_and_writes_a_new_one` — 已存在 → exit 2 且不覆盖；
   新路径 → 写入并把 sha256、`o2_digest` 记进结果，`learn.load_checkpoint` 可回读
   `extra.command="train-bc"`、`extra.dataset_kind="synthetic"`。

测试通过 `importlib` 把 `tools/rl/v430.py` 当模块加载并调用 `main(argv)`（同一进程，快），
`main()` 的返回值就是进程退出码的来源；真实进程级退出码在 `EXIT_CODES_W03G.txt` 里另行实测。

## 9 既有子命令回归（行为未变）

| 命令 | 退出码 | 摘要 |
| --- | --- | --- |
| `audit --config configs/v430/assets.json` | **3** | checks 53 / verified 51 / unhashed 1 / mismatch 1 / missing 0 / unreachable 0 / wsl up，unmet 仍是 `sync:…/v4.2.0/journal.jsonl`（先前既有） |
| `validate-contract --config configs/v430/contracts.json` | **0** | checks 173 / failed 0 / groups 13 |
| `validate-splits --config configs/v430/splits.json` | **0** | mode strict / problems 0 / splits 5 / sha `9160cf841de1` |

## 10 语义分层（不许混淆）

- `exit 0` 只说**训练按协议跑完了**，不是能力；`capability_gate` 恒为 `not_run`；
- `dataset_kind="synthetic"` 说明这四个 loss 与 TH08 表现**无关**，不能当训练曲线证据；
- `o2_digest` 一致只说明用的是同一套 O2 编码参数，不说明像素内容真实；
- `parameter_report.total`（2726671）只证明模型规模在冻结范围，不证明它学会了任何东西。

## 11 剩余缺口（下一步的输入，不是本切片的问题）

1. 真实语料接入：需要先有 O2 契约的语料（legacy v4.2.0 契约按 W03F 不可喂 O2 模型），
   届时 `--batches` 需要第二套（非合成）来源与对应的 `dataset_kind`，并保留「合成/真实」不可混淆的标记；
2. checkpoint → 重新加载 → 评测的闭环本切片未做（结果里已如实写进 `remaining_gaps`）；
3. `train-bc` 目前只跑单进程 CPU；GPU 上跨进程逐位一致需要额外的确定性设置，本切片不承诺；
4. W03 之外的门（G430-*）一个都没有因为本切片而改变状态。
