# W04B — 把 O2 观测契约守卫真正接到训练路径，并让 `batch_source` 由调用方声明

**范围声明**：本文件只记录 `touhou_rl/v430/learn.py` 中 `train_from_manifest` 的两处修复
与 `tests/test_v430_learn.py` 的增量测试。全部证据来自**合成 fixture**：不启动引擎、
不读 `.tmp_*`、不训练真实数据。**合成测试不是游戏/能力证据**。

- 工作包：W04B（W04 的观测契约守卫接线切片）
- 缺陷来源：主代理实测（下 §0）
- 依赖：`torch` + stdlib + 既有 `touhou_rl.v430.{contracts,data.manifest,data.lineage,models.actor}`；
  **未**引入新依赖、**未**改 `manifest.py`（0 行）

## 0. 本片修掉的两个真实缺陷

| # | 之前的可观测行为 | 原因 | 现在 |
| --- | --- | --- | --- |
| 1 | 真实 O2 manifest → `protocol_validity: "valid"`；把 `obs_contract` 改成 `"obs-v420-2f"` → **完全相同的结果，未被拒绝** | `train_from_manifest` 只跑 `validate_manifest`，而它接受任何**已注册**的契约 —— `obs-v420-2f` 正是注册过的 legacy 契约（`KNOWN_OBS_CONTRACTS`）。`assert_o2_compatible` 存在、有 7 条测试，但**在这个入口上从未被调用** | gate 1（`validate_manifest`）之后、任何训练步之前，立即调用 `assert_o2_compatible`；`run=False` 干跑路径同样调用 |
| 2 | 喂真实采集的 64 个 O2 样本 → `batch_source: "synthetic"`、`synthetic_data_only: True` | 结果字段**硬编码** | 新增关键字参数 `batch_source`（默认 `"synthetic"`），`batch_source` / `synthetic_data_only` / `research_result.batch_source` / `research_result.scope` 全部由它推导 |

## 1. 变更文件（严格白名单内）

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `touhou_rl/v430/learn.py` | 修改（最小侵入，10 处） | 新增 `BATCH_SOURCES` + `_require_batch_source` + `train_from_manifest(batch_source=...)` + 接线 O2 守卫 + result 字段推导，见 §2 |
| `tests/test_v430_learn.py` | 增量追加（+5 条测试，175 行） | 4 条 legacy/dry-run/batch_source 行为测试 + 1 条非法值测试，见 §5；既有 24 条**一条未改**（仅 import 增补 2 个名字） |
| `Document/Update/Alpha/v4.3.0/evidence/W04/W04B_OBS_GUARD_WIRING.md` | 新建 | 本文件 |
| `Document/Update/Alpha/v4.3.0/evidence/W04/EXIT_CODES_W04B.txt` | 新建 | 验收命令退出码 |

未触碰：`touhou_rl/v430/data/manifest.py`（**0 行**）、`{contracts,actions,session,evaluate,observation,sequence,ppo}.py`、
`models/**`、`v400|v420/**`、`tools/rl/**`、`configs/**`、其它 `tests/test_v430_*.py`、
`evidence/{W00,W01,W02,W03,W09}/**`、`registries/**`、`GATE_TABLE.md`、`ASSET_SYNC.*`；
未删除/移动/`git add`/commit/上传；未读 `.tmp_*`。

## 2. `learn.py` 逐处改动

| # | 位置（选后行号） | 改动 |
| --- | --- | --- |
| 1 | L54–58（模块 docstring 第 4 条） | 把「`validate_manifest` → `validate_splits` → 训练」改写为「`validate_manifest` → **O2 守卫** → `validate_splits` → 训练」，并写明 `batch_source` 是调用方声明 |
| 2 | L96（`__all__`） | 导出 `"BATCH_SOURCES"` |
| 3 | L126 | 新增 `BATCH_SOURCES: tuple[str, ...] = ("synthetic", "captured_o2")`，注释明写「声明，不校验」 |
| 4 | L177–186 | 新增 `_require_batch_source(value)`：非法值抛 `LearnError`，消息列出合法值并复述「模块只记录声明」 |
| 5 | L927（签名） | 新增关键字参数 `batch_source: str = "synthetic"`（位于 `shuffle_batches` 与 `run` 之间；纯 keyword-only，不改变既有位置参数） |
| 6 | L933–1006（docstring） | 补三重说明：① 守卫顺序与「legacy 契约也能通过 `validate_manifest`」；② `batch_source` 参数含义与**这是声明、未经验证**；③ `Raises` 增加 `ManifestError` **原样传播**、不是 `LearnError` 子类、`run=True/False` 都抛 |
| 7 | L1014 | `batch_source = _require_batch_source(batch_source)`，在**任何 gate 之前**（非法配置不需要读到 manifest 就该失败） |
| 8 | L1026–1043 | **gate 1b**：`manifest_module.assert_o2_compatible(manifest_input)`。位置在 `validate_manifest` 合法性检查之后、`validate_splits` 之前、`if run:` 之前 —— 因此训练步与干跑都必然经过。**不捕获、不包装**该异常 |
| 9 | L1103–1109 | 推导块：`synthetic_data_only = batch_source == "synthetic"`；`scope` 二选一；`batch_source_note` 明写 "declared by the caller … this is NOT verified" |
| 10 | L1111–1143 | `research` 字典与 `new_result(**kwargs)` 的 `batch_source` / `synthetic_data_only` / `batch_source_verified` / `batch_source_note` / `scope` 全部改为由声明推导；另把 checkpoint 的 `extra["batch_source"]` 从硬编码 `"synthetic"` 改为声明值并加 `batch_source_verified: False`（L1078–1081） |

`train_bc` **一行未改**（其 24 条锁定测试全过）。`manifest.py` **一行未改**。

## 3. 实测：legacy 契约被拒（完整错误消息）

前置事实（这条缺陷的入口确实存在 —— 守卫是**唯一**拦截点）：

```text
validate_manifest(load_manifest(<legacy manifest>), base_dir=...)["valid"] == True
```

（`obs-v420-2f` 是**已注册**契约，manifest 只有 `obs_contract` 一个字段与 O2 fixture 不同，
sha256 / 计数 / splits 全部自洽，所以 gate 1 判它 valid。这正是修复前 `train_from_manifest`
会直接放行它的原因。）

实测拒绝，`run=True` 与 `run=False` **同一条**消息、**同一个**类型
`touhou_rl.v430.data.manifest.ManifestError`（`isinstance(exc, learn.LearnError) is False`）：

```text
manifest th08-w03e-synthetic: obs_contract is 'obs-v420-2f', but the O2 model requires
'obs-v430-o2'; it stores the v4.2.0 downsampled observation, not O2: global_rgb is absent
from it entirely, play_gray is (4,112,96) = exactly 2x downsampled from O2's (4,224,192),
and hud_gray is (56,160) = 8x vertically downsampled from O2's (1,448,160). There is no
upgrade path in this module (no resampling, no zero-fill, no channel synthesis), so the
corpus must be re-captured under 'obs-v430-o2'
```

`run=False` 实证：干跑同样抛上面这条 `ManifestError`，**没有返回任何 result** ——
因此不存在「干跑照样给出 `protocol_validity: "valid"`」这种误导产物。

「在任何训练步之前」实证：测试把 `_fresh_actor(seed=77)` 传进去，捕获异常后逐张量
`torch.equal(before[name], after[name])` 全为真，即权重逐位未变（`test_train_from_manifest_refuses_a_legacy_obs_contract`）。

## 4. `batch_source` 新参数 ↔ result 字段

| 声明的 `batch_source` | `result["batch_source"]` | `result["synthetic_data_only"]` | `research_result.batch_source` | `research_result.scope` |
| --- | --- | --- | --- | --- |
| `"synthetic"`（默认） | `"synthetic"` | `True` | `"synthetic"` | `bc-orchestration/synthetic-batches` |
| `"captured_o2"` | `"captured_o2"` | `False` | `"captured_o2"` | `bc-orchestration/captured-o2-batches` |
| 其它（如 `"real"`、`""`、`"Captured_O2"`） | — | — | — | 抛 `LearnError`，不产生 result |

对应关系是**推导**而不是另一次赋值：`synthetic_data_only = (batch_source == "synthetic")`。
`run=False` 的结果同样携带该声明（干跑也可以给批次分类）。

**这是声明，不是校验结果**（写进 docstring、result 与 research_result 三处）：

```json
{
  "batch_source": "captured_o2",
  "synthetic_data_only": false,
  "batch_source_verified": false,
  "batch_source_note": "declared by the caller as 'captured_o2'; this is NOT verified:
    train_from_manifest cannot inspect a batch's provenance, so a caller could label
    captured data synthetic or vice versa"
}
```

非法值实测消息：

```text
LearnError: batch_source 'real' not in ['synthetic', 'captured_o2']; declare 'synthetic'
for batches generated in code or 'captured_o2' for real O2 captures (the module records
the declaration, it cannot verify it)
```

## 5. 测试：+5 条（`tests/test_v430_learn.py`，增量追加）

| # | 要求 | 测试 | 关键断言 |
| --- | --- | --- | --- |
| 1 | legacy 契约 → `train_from_manifest` 抛错 | `test_train_from_manifest_refuses_a_legacy_obs_contract` | 先断言 legacy manifest 仍 `valid=True`（守卫是唯一拦截点）→ `pytest.raises(ManifestError)`、消息含 `obs-v420-2f` 与 `obs-v430-o2`、`not isinstance(..., LearnError)`、模型权重逐位不变 |
| 2 | 同一 legacy manifest 在 `run=False` 也抛错 | `test_dry_run_also_refuses_a_legacy_obs_contract` | `run=False` 抛同一 `ManifestError`；对照组（O2 manifest 干跑）仍 `protocol_validity=="valid"` 且 `training.n_steps==0` |
| 3 | 合法 O2 + 默认 `synthetic` | `test_batch_source_defaults_to_synthetic` | `synthetic_data_only is True`、`research.scope=="bc-orchestration/synthetic-batches"`、`batch_source_verified is False`、note 含 `NOT verified` |
| 4 | 同一 manifest + `captured_o2` | `test_batch_source_captured_o2_is_not_called_synthetic` | `synthetic_data_only is False`、`batch_source=="captured_o2"`、`scope=="bc-orchestration/captured-o2-batches"`、`batch_source_verified is False`、`n_steps==1`（声明不改变训练）、干跑同样保留声明 |
| 5 | 非法 `batch_source` | `test_an_unregistered_batch_source_is_refused` | `"real"` / `""` / `"Captured_O2"` 均 `pytest.raises(LearnError, match="batch_source")`，且**不读 manifest**也拒绝（连 `absent.json` 都先报 `batch_source`） |
| 6 | 既有 24 条继续全过 | 全部 24 条 | **零改动**：没有改任何 fixture、没有删任何断言。新增测试复用既有 `dataset` fixture（其 `obs_contract` 本来就是 `obs-v430-o2`），所以默认 `synthetic` 的既有断言（`research["batch_source"] == "synthetic"`、`synthetic_data_only is True`）原样成立 |

唯一对既有内容的触碰：import 块增加 `LEGACY_OBS_CONTRACTS` 与 `ManifestError` 两个名字
（只增不改，无断言变化）。测试**不依赖** `.tmp_v430_o2/o2-64/manifest.json` 或任何 `.tmp_*` 路径，
fixture 自包含（`tmp_path` 内真实写盘 shard 文件 + 合成 manifest dict）。

## 6. 验收命令与退出码（真实执行）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m compileall -q touhou_rl/v430/learn.py
# exit 0

python -m pytest -p no:cacheprovider --basetemp .tmp_v430_pytest_w04b -q tests/test_v430_learn.py
# 29 passed in 9.69s / exit 0   （此前 24 passed）

python -m pytest -p no:cacheprovider --basetemp .tmp_v430_pytest_w04b_reg -q tests/test_v430_data.py tests/test_v430_ppo.py tests/test_v430_models.py
# 94 passed in 2.53s / exit 0
```

与 `EXIT_CODES_W04B.txt` 同一轮捕获（最后一次 docstring 改动之后重跑）。
`compileall` 退出码 0 只代表语法通过；真实结论来自两条 pytest。

## 7. 诚实边界

1. `batch_source="captured_o2"` **只是声明**：本片没有任何手段校验批次来源
   （torch 张量不带 provenance），result 里以 `batch_source_verified: false` + note 明示，
   docstring 同样写明。**不要**把它读成「已确认是真实采集数据」。
2. `ManifestError` 故意**不**包装成 `LearnError`（否则会破坏 manifest 层 7 条锁定测试，
   也会丢掉「消息里点名实际契约」这一类型约定）。上层若要统一处理，需在调用侧同时捕获
   `ManifestError`；本模块只在 docstring 与 `Raises` 里声明这一点。
3. `run=False` 现在也会拒绝 legacy 语料 —— 这是**有意的行为变更**（干跑也输出
   `protocol_validity: "valid"`）。仓库内 `train_from_manifest` 无其它调用者
   （`tools/rl/v430.py` 走的是 `train_bc`），故无回归面。
4. 未用真实语料：`.tmp_v430_o2/o2-64/manifest.json` 未被测试引用；本片的 legacy/O2 两类
   manifest 都是合成 fixture。真实 64 样本 O2 语料的端到端训练仍属后续工作包。

## 8. 剩余阻塞

本片目标内**无**。遗留（不阻塞本片、也不由本片声称解决）：

- `captured_o2` 的自动校验需要一个能读语料 provenance 的组件（例如由采集端签发
  manifest 侧 `batch_source` 断言），本片不做；
- 真实 O2 语料上的 `train_from_manifest` 全链路（引擎 / 数据腿）仍未验证。
