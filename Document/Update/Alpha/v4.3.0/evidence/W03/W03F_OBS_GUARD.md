# W03F — 观测契约守卫（manifest 层声明与拒绝，无任何转换）

日期：2026-11-15。执行代理（DeepSeek Flash）。范围：**小切片**，只做「声明 + 拒绝」。

## 1 目标

让「非 O2 契约的语料」在 manifest 层**不可能**被交给 O2 模型。

`observation.o2_digest()` 只能比对契约参数块/版本，**查不出**「形状对、内容却是降采样」。
因此在 `touhou_rl/v430/data/manifest.py` 增加 `obs_contract` 必填声明与
`assert_o2_compatible()` 硬门；**不**做上采样、**不**补零、**不**拼接通道、
**不**提供 legacy → O2 的自动升级路径。

## 2 磁盘上 legacy 契约的真实名字

只读一处（不扫语料）：

```
.tmp_v420_data/T04/dataset/d1-s0d1-n3300-seed10-shard/meta.json
  -> meta["contract"]["obs_contract"] == "obs-v420-2f"
```

该值已登记为 `LEGACY_OBS_CONTRACTS["obs-v420-2f"]`，并在测试里以
`LEGACY_CONTRACT = "obs-v420-2f"` 固化（测试**不**读 29 GB 资产）。

## 3 改动（只增不改既有规则）

| 项 | 内容 |
| --- | --- |
| `OBS_CONTRACT_O2` | `"obs-v430-o2"`，唯一可用于 O2 模型的契约 |
| `LEGACY_OBS_CONTRACTS` | `{"obs-v420-2f": <不可升级的具体原因>}` |
| `KNOWN_OBS_CONTRACTS` | `(OBS_CONTRACT_O2,) + tuple(LEGACY_OBS_CONTRACTS)` |
| `O2_OBSERVATION_CHANNELS` | `global_rgb`/`play_gray`/`hud_gray`/`executed_key_history` |
| `REQUIRED_MANIFEST_FIELDS` | 追加 `obs_contract`（缺失 → 既有 `P_MISSING_FIELD`） |
| `Manifest` | 新字段 `obs_contract`、`obs_shapes`（可选几何声明） |
| `_check_obs_contract` / `_check_obs_shapes` | 在既有 `identity` 检查内追加（`checks_performed` 列表未变） |
| `assert_o2_compatible(manifest)` | 非 O2 → `ManifestError`；O2 但几何不符 → 也 `ManifestError` |
| `o2_compatible(manifest) -> bool` | 不抛异常的分流版 |
| report | 追加 `obs_contract`、`o2_compatible` 两个键（纯增量） |

新增错误码（两者均已登记进 `PROBLEM_CODES`，码表 31 → 33）：

- `P_OBS_CONTRACT_UNKNOWN` — `obs_contract` 不在 `KNOWN_OBS_CONTRACTS` 内
- `P_OBS_SHAPE_CONTRACT_MISMATCH` — 声明 `obs-v430-o2` 但 `obs_shapes` 与
  `observation.O2_SHAPES` 不符（或未声明 O2 四通道）

## 4 语义分层（关键，不许混淆）

- `validate_manifest(...)["valid"]` 只说**manifest 自身为真**。legacy 语料照样 `valid=True`
  （它确实是 `obs-v420-2f` 的数据）——`valid` **不是** O2 声明。
- 能不能喂 O2 模型只看 `assert_o2_compatible()` / `o2_compatible()`。
- `obs_shapes` 缺省时不构成形状主张：`not_checked` 会写明「按名字绑定、几何未验证」，
  不会静默当作已验证。
- 反过来，**形状声明不能洗白契约名**：`obs_contract="obs-v420-2f"` 即使声明了 O2 几何，
  仍被拒绝（有测试覆盖）。

## 5 legacy 被拒时的完整错误消息（实测）

```
manifest <dataset_id>: obs_contract is 'obs-v420-2f', but the O2 model requires
'obs-v430-o2'; it stores the v4.2.0 downsampled observation, not O2: global_rgb is
absent from it entirely, play_gray is (4,112,96) = exactly 2x downsampled from O2's
(4,224,192), and hud_gray is (56,160) = 8x vertically downsampled from O2's
(1,448,160). There is no upgrade path in this module (no resampling, no zero-fill,
no channel synthesis), so the corpus must be re-captured under 'obs-v430-o2'
```

同时给出：实际契约名、`obs-v430-o2`、以及三条形状差异（`global_rgb` 缺失、
play 2× 降采样、hud 8× 降采样）。

## 6 「无上采样 / 无伪造通道」是怎么断言的

`tests/test_v430_data.py::test_manifest_module_has_no_resampling_or_channel_synthesis`
对 `manifest.py` 做 **AST 扫描**（不是只看文档）：

1. **模块顶层导入**不得出现 `numpy`/`np`/`torch`/`cv2`/`PIL`/`Image`/`scipy`/`skimage`；
   且 `observation`（会把 numpy 拉进来）**只能**是函数内惰性导入
   （`_o2_shapes()`），顶层出现 `relative:observation` 即失败；
2. 全模块**任何 `ast.Call`** 的名字（点号全路径）不得以数组库为根，且最后一段不得是
   `repeat`/`repeat_interleave`/`tile`/`interpolate`/`resize`/`zoom`/`pad`/`zeros`/
   `zeros_like`/`ones`/`ones_like`/`full`/`full_like`/`empty`/`new_empty`/`kron`/
   `broadcast_to`/`upsample`/`resample`/`concatenate`/`concat`/`stack`/`vstack`/
   `hstack`/`dstack`；
3. **函数名**不得包含 `upsampl`/`resampl`/`interpolat`/`concat`/`convert`/`promote`/
   `synthes`/`pad_`/`_pad`；
4. 源码文本也不得含 `np.repeat`/`np.zeros`/`np.pad`/`np.resize`/`cv2.resize`/
   `F.interpolate`/`torch.nn.functional.interpolate` 这些审阅者会 grep 的字面量。

## 7 测试（`tests/test_v430_data.py`，单文件追加）

新增 7 条（29 → 36）：

1. `test_missing_obs_contract_is_rejected` — `P_MISSING_FIELD`，`field=obs_contract`
2. `test_unknown_obs_contract_is_rejected` — `P_OBS_CONTRACT_UNKNOWN`
3. `test_legacy_obs_contract_validates_but_is_refused_by_the_o2_gate` —
   legacy 仍 `valid=True` 但 `o2_compatible=False`；`assert_o2_compatible` 抛错，
   消息含 `obs-v420-2f`、`obs-v430-o2`、`global_rgb`、`2x`、`8x`、`112`、`56`
4. `test_o2_contract_with_non_o2_shapes_is_rejected` —
   `P_OBS_SHAPE_CONTRACT_MISMATCH`，消息含四个通道与 `(4,112,96)`/`(4,224,192)`
5. `test_o2_compatible_routes_every_other_case_to_false` — legacy/未知/缺失/非 manifest
   全 `False`；O2+正确几何 `True`；断言 `O2_OBSERVATION_CHANNELS` 与真实
   `observation.O2_SHAPES` 数值一致（防止测试自带一份假表）
6. `test_obs_contract_codes_are_registered` — 两个新码在 `PROBLEM_CODES`；契约白名单包含
   `obs-v430-o2` 与 `obs-v420-2f`；未登记的码仍抛 `ManifestError`
7. `test_manifest_module_has_no_resampling_or_channel_synthesis` — 见 §6

既有 29 条未删任何断言；唯一的 fixture 变更是 `_document()` 补 `obs_contract`
（新增必填字段），`checks_performed` 精确列表断言因此依旧成立。

## 8 未做 / 边界

- 未实现任何 legacy → O2 适配器（明确禁止）。
- 未重新采集 O2 语料；`obs-v430-o2` 语料在本仓库中**仍不存在**。
- `obs_shapes` 是「可选声明」：不声明则几何未验证（写进 `not_checked`），
  不声明**不**等于通过。若计划要求「O2 manifest 必须声明几何」，需另开切片把它变成必填。
- 该守卫只覆盖 manifest 层；shard 内部 `.npy` 的 dtype/shape 仍需采集侧或
  stream loader 自己校验（本切片不做）。

## 9 验收

见同目录 `EXIT_CODES_W03F.txt`。
