# Template round3 命令结果

日期：2026-11-01；Python 3.12.10；仅合成，无游戏或训练进程。

## 首次单测（保留失败）

```text
python -m pytest tests/test_tactical_task_templates.py -q -p no:cacheprovider
5 passed, 1 error in 0.24s
exit code: 1
```

错误发生于pytest tmp_path fixture创建之前：系统临时目录 C:/Users/a5935/AppData/Local/Temp/pytest-of-a5935 的os.scandir触发PermissionError [WinError 5]。不是断言或模板失败；未修权限、未重复访问该目录。后续显式选用工作区新临时目录。

## 新模板单测

```text
python -m pytest tests/test_tactical_task_templates.py -q -p no:cacheprovider --basetemp .tmp_template_round3_pytest
6 passed in 0.11s
exit code: 0
```

## 冻结开发+验证评估（仅一次）

```text
python -m touhou_rl.tactical_task_templates --output Document/Update/Alpha/v3.7.3/evidence/template-round3
exit code: 0
```

- development：F4 goal/survived/reference=12/12；F5=12/12。
- validation：F4 goal/survived/reference=12/12；F5=12/12。
- manifest：development=24, validation=24, test=12；8个结构group；hash_integrity=true；group_disjoint=true。
- test_executed=false，test_status=sealed_manifest_only，formal_training_labels=false。
- source_hashes_unchanged=true；详细hash和逐事件结果见本目录产物。

## 已有接口回归

```text
python -m pytest tests/test_tactical_tasks.py tests/test_tactical_task_teacher.py tests/test_tactical_task_templates.py -q -p no:cacheprovider --basetemp .tmp_template_round3_regression
72 passed in 0.47s
exit code: 0
```

所有命令均前台完成，没有后台job。上述单测不instantiate或执行封存test。开发验证评估未按结果修改teacher、模板、参数或阈值，失败不覆盖策略已实现。
