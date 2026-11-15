# T04 验收命令与退出码（收尾实测）

全部在资产主机 WSL 内执行，`cwd=/mnt/d/WorkShop/Touhou_RL`，`PYTHONPATH=仓库根`。
脚本：`.tmp_v420_t04/{acceptance.sh,train.sh,run_eval.sh,run_all_evals.sh}`。

| # | 命令 | 结果 | 退出码 |
|---|---|---|---:|
| 1 | `PYTHONPATH=$PWD /root/v400-venv/bin/python -m pytest -p no:cacheprovider --basetemp $PWD/.tmp_v420_data/tests/T04b -q tests/test_v420_train.py` | **24 passed** in 5.97 s | **0** |
| 2 | `PYTHONPATH=$PWD /root/v400-venv/bin/python tools/rl/v420.py train --config configs/v420/learn-controlled.json --out $PWD/.tmp_v420_data/T04/train.json --execute` | 真实训练完成，墙钟 386.67 s；报告 167,529 B | **0** |
| 3 | `PYTHONPATH=$PWD /root/v400-venv/bin/python tools/rl/v420.py --help` | 8 个子命令列出，`train` 为可执行帮助文本 | **0** |
| 4 | `PYTHONPATH=$PWD /root/v400-venv/bin/python tools/rl/v420.py train --config configs/v420/learn-controlled.json`（dry-run，不带 `--execute`） | `mode=dry-run`、`executed=false`、两个臂的计划与 checkpoint 路径 | **0** |
| 5 | `PYTHONPATH=$PWD /root/v400-venv/bin/python -m compileall -q touhou_rl/v420` | 无输出 | **0** |
| 6 | `PYTHONPATH=$PWD … pytest … tests/test_v420_{contracts,replay_audit,eval,rebuild,reslot,dataset_2f,train}.py`（回归） | **221 passed** in 15.59 s | **0** |

Windows 侧（`python 3.12.10`，`PYTHONPATH=D:\WorkShop\Touhou_RL`）：

| # | 命令 | 结果 | 退出码 |
|---|---|---|---:|
| 7 | `python -m compileall -q touhou_rl\v420` | 无输出 | **0** |
| 8 | `python -m pytest -p no:cacheprovider --basetemp .tmp_v420_pytest_win5 -q tests/test_v420_train.py` | **24 passed** in 2.83 s | **0** |

引擎评测命令（32 条，逐条 argv 见 [`dev_eval.json`](dev_eval.json)）：

| suite | 条数 | 全部退出码 0 | 合计墙钟 |
|---|---:|---|---:|
| 标准 S1N（无 `--stop-on-death`） | 20 | 是 | 3,626.8 s |
| DIAG-FIRST-DEATH（`--stop-on-death`） | 12 | 是 | 1,952.0 s |

单条命令最长的墙钟远低于 10 min 上限（标准 S1N 每 batch ≤ 4 局、DIAG 每 batch ≤ 8 局）。

---

## 未改动（白名单外的只读对象）

- `touhou_rl/v420/{contracts,eval,dataset_2f,batch_2f,replay_audit,rebuild,reslot,__init__}.py`：
  mtime 仍是 17:21–18:38（本任务的文件是 18:47 之后）。`counterfactual.py` 在 19:59 被
  **T03 的执行代理**改过（并行任务），不是本任务。
- `tools/rl/v410_student_eval.py`：mtime 06:00，sha256
  `2db2caaf9089c1698cef9b6a682991a7594ebac47ae93244c3d9932591eddddc`（与 T00 登记一致）。
- `touhou_rl/v410/student/{bc,diagnostics,s1_data}.py`：mtime 04:xx（只读复用）。
- 冻结权重 `.tmp_v410_wp04_arms/b0_seed0_last.pt`（`3bc0ec44…`）与
  `.tmp_v410_wp05_train_beta05_20k/b0_s0_dagger_step20000.pt`（`15ae1fd5…`）：sha256 未变。
- `touhou_rl/v400/**`、`ASSET_SYNC.*`、旧证据、旧 configs：未由本任务写入
  （`git status` 里 `touhou_rl/v400/gym/trajectory_verify.py`、`touhou_rl/v400/teacher/search.py`
  的 `M` 是 T00 已登记的脏工作树，本任务没有写它们）。
- 没有 `git add/commit/push`；没有读 confirm 种子 61000–61499；没有碰 60000–60199；
  没有写 `.tmp_v410_*` / `.tmp_v400_*` / `/root/v400-data/**`。

## 新增资产（本机，不入 Git）

| 路径 | 大小 | 内容 |
|---|---:|---|
| `.tmp_v420_data/T04/checkpoints/a0_seed0_step5000.pt` | 2.94 MB | A0 checkpoint（`96cb219e…`） |
| `.tmp_v420_data/T04/checkpoints/a0v_seed0_step5000.pt` | 2.94 MB | A0v checkpoint（`f7188c0d…`） |
| `.tmp_v420_data/T04/{train.json,train_smoke.json,train.log}` | 0.4 MB | 训练报告、smoke 报告、进度日志 |
| `.tmp_v420_data/T04/eval/` | ≈ 2 MB | 32 个 batch 的原始 evaluator JSON + 运行记录 + 两份 suite 汇总 |
| `.tmp_v420_data/T04/evaldata/` | 180 KB + 2 符号链接 | 评测用的游戏数据副本（`th08.dat`/`thbgm.dat` 指向只读原件） |
| `.tmp_v420_data/T04/checkpoints_smoke/` | 5.9 MB | 20 步 smoke 的两个 checkpoint（仅为证明链路，不是结果） |
