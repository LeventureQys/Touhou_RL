# EXP-P1-022 `lethal_counts()` 键集契约（既有红修复）

- **日期**：2026-11-10
- **状态**：**完成**：唯一既有红修绿，`-k v400` 0 failed / exit 0；实现侧未动，只改测试 + 本卡。
- **上游**：T0 报告 §8（唯一超出 T0 白名单的既有红）、[EXP-P1-021](EXP-P1-021-laser-damageable.md)（激光判定窗）
- **产物**：`tests/test_v400_teacher.py`（+3 行）、本卡
- **一句话**：测试字面量停在 EXP-P1-021 之前的三键，补上引擎 `damageable` 的第四键。

## 1. 现象（只读复现）

```powershell
$env:PYTHONPATH=(Get-Location).Path
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests touhou_rl/tests -k v400
# 1 failed, 1359 passed, 5 skipped（exit 1）
```

失败行：`E  AssertionError: ... Left contains 1 more item: {'lasers_damageable': 0}`，落点
`tests\test_v400_teacher.py:1117`；测试 mtime `2026-11-10 00:26`、实现 mtime `2026-11-09 21:12`，均早于本轮会话。

## 2. 根因

`hazard_tape.py:89-107` 的 `lethal_counts()` 返回 4 键（`bullets`/`lasers`/`lasers_damageable`/`enemies`），
docstring 明写第四键来自 EXP-P1-021；测试 1117 行字面量只有 3 键，且与两个 tape 三方 `==`，多键必红。

## 3. 判定：契约在实现侧（改测试）

- grep `lasers_damageable`：写方 `hazard_tape.py:92/102` + 实验卡（EXP-P3-004 §11.7、EXP-P3-006 §9、
  EXP-P3-007 §9 均记为 "v2h 激光字段" 的既有红）；`tools/rl/v400_teacher.py`、
  `tools/rl/v400_death_analysis.py` 内无直接读取方。
- 但它是导出契约：`hazard_tape.py:190` 把 `lethal_counts` 放进 `summary()`，`decide.py:76` 把它放进
  `Decision.as_dict()["tape"]`，`tools/rl/v400_teacher.py` 每决策写一行 JSONL —— 删键等于静默改导出
  schema 并丢掉激光判定窗审计计数。
- 底层 `damageable` 确被下游使用：`geometry.py:429-448`（`laser_hits` 门）、`search.py:179/355/908/925`
  （`laser_penalty`）、`config.py:117-154`（v2h）、`tools/rl/v400_teacher.py:111`。
- 语义明确（`damageable == 1` 行数，`lasers` 子集）、有意新增、有卡背书 → 改测试；EXP-P3-006 §9
  独立诊断同结论："测试里的字典字面量没跟上"。

## 4. 改动与断言强度（+3 行，只此一处）

`tests/test_v400_teacher.py:1117-1124`：

```diff
+    # EXP-P1-021: `lethal_counts()` also reports the engine's `damageable` laser
+    # rows (the laser judging window), so the exact contract is these four keys.
+        "lasers_damageable": 0,
```

实现/引擎/协议/依赖均未改。三方 `==` 保留：两个 tape 与字面量键集必须完全相同，多键/少键均报错，
即精确锁四键；无 `assert True`、无 `>= 0`、未改成"至少包含"。

## 5. 验证（真实输出）

- 单跑该测试：`1 passed in 0.39s`，exit 0
- 全量 `-k v400`：`1398 passed, 5 skipped, 2937 deselected in 35.03s`，exit 0
- `python -m compileall -q touhou_rl tests`：无输出，exit 0
- `1398 = 1359 + 1（本条转绿）+ 38`；38 来自并发任务未跟踪的 `tests/test_v400_p3_probe.py`（非本卡引入）。

边界：未重建 build、未动第三方；`git add` 逐文件，commit 只含 `tests/test_v400_teacher.py` 与本卡。
