# V829-INFO-01｜训练端只读盘点工单

**可整份转发。** 发件方为 GPT 研究端，收件方为 DeepSeek 训练端，用户转交。请判断现存权重、真实评测记录和未曝光起点能否支撑 v8.29 的零训练对照。两臂分别为 v8.26 的自适应放雷 G1 和 v8.23 的固定阈值 T2（40/20/5 px），移动均使用原 `bombdag.pt` 的 softmax 采样。

**范围。** 本单只授权读取下列路径及已登记清单指向的具体来源、离线核算、在聊天返回小表。引擎启动、采集、跑新评测、训练、依赖安装、旧文件写入和外部上传均超出本单。WP05 继续暂停。封存集只查切分登记中的用途。请由训练端主代理按本项目约定派给 Provider=DeepSeek、模型=DeepSeek v4.1 flash 的执行代理，主代理核对范围、实际命令和统计。缺项也请立即回报已核实的部分。

## 1. 四项交付

| 任务 | 允许读取的来源 | 回传最少证据 |
| --- | --- | --- |
| 确认运行身份 | 实际运行检出、两个历史 runner 及其明确引用的恢复/计分/动作模块、已登记引擎与三份 `bombdag.pt`、已有 D3/前缀身份记录 | 主机代号、时间和时区、实际运行根/HEAD/Python、引擎和三个权重的完整 SHA、runner 及关键依赖 SHA、历史 D3 文件身份。标出与发布记录的差异 |
| 复核历史 30 秒对照 | 两个版本各 4 窗 × 3 权重的 d900 逐局文件；先读已登记发布件，必要时查原运行的 `eval` 目录。登记参照为 `ASSET_SYNC.sha256` | 逐窗有效局、满段局、重复或坏行、文件与登记是否相符。历史预期 G1 为 38/44/21/43，T2 为 40/39/26/36，各窗每臂 60 局。尽量附两臂既有逐局墙钟中位数、P90 及前缀/恢复实测或历史估计 |
| 盘点新开发起点 | `configs/v430/splits.json` 及其登记的谱系表、训练机已知开发录像/前缀索引和生成配置。只沿已有索引查来源；请写明所查索引身份和谱系核对范围 | 独立对局根总数、道中/Boss 各阶段可用数、重复/曝光/未知数。每个提名项给匿名 ancestor ID、阶段、来源及父前缀摘要、接管帧和校验锚点、库存、剩余 2,400 帧的可判定依据、训练/选参曝光状态。只查已登记元数据，不按后续表现筛选 |
| 确定旧日志能回答哪种耗雷问题 | 上述 G1/T2 逐局文件，以及其配套已有轨迹（如有）；可补读 G1 d1200 的既有 12 个单元格 | 各给一段实际请求和库存的例子，注明能否定位实际扣雷决策/帧。分别统计有精确证据、仅能界定区间、没有足够信息的局数。给出可独立继续的诊断 |

## 2. 可直接运行的只读核对

以下命令在**训练端 WSL 的实际运行检出根目录**执行。若运行机路径不同，替换为已登记路径并随回执说明。分别保留退出码。缺少文件保留错误原文，继续可独立执行的检查。

```bash
git rev-parse --show-toplevel
git rev-parse HEAD
git branch --show-current
python -B -c "import sys; print(sys.executable, sys.version.split()[0])"
sha256sum -- third_party/th08-rl/build/rl_v3/th08-modern .tmp_v860_bombdag/ckpt/seed{0,1,2}/bombdag.pt
sha256sum -- tools/rl/{v8r0_urgency,v8o0_ruletight,v830_bombpol,v800_bomb,v431_arm_a,v432_fork_arm,v450_cf_label}.py touhou_rl/v420/{contracts,counterfactual}.py touhou_rl/v400/gym/actions.py
```

记录 SHA 时标注是 WSL 实跑文件、Git blob 还是 Windows 工作副本。T2 评测后更新过汇总/发布逻辑，回传时分别给历史实跑和当前版本的身份。判定起点同源时比较游戏初态、实际消费的有序动作前缀和接管锚点；引擎 SHA 或整份录像的 SHA 都不能单独证明祖先独立。[T2 的版本变动记录](../v8.23.0/evidence/RULETIGHT/RUN.md)

下段只使用 Python 标准库。它核对每个文件的登记 SHA、唯一局号、臂/窗口/checkpoint 种子、本档帧数和满段总数。某格缺失、摘要不匹配或身份冲突时，回报该格及其他可核格。两份同名副本只计一次。若研究规划文档尚未同步到训练机，可直接复制命令。

```bash
python -B - <<'PY'
import hashlib
import json
from pathlib import Path

manifest = {}
for line in Path("ASSET_SYNC.sha256").read_text(encoding="utf-8-sig").splitlines():
    parts = line.split(maxsplit=1)
    if len(parts) == 2:
        manifest[parts[1].lstrip("*").removeprefix("./")] = parts[0]

arms = [
    ("G1", "v8.26.0/evidence/URGENCY", ".tmp_v8r0_urgency", [38, 44, 21, 43]),
    ("T2", "v8.23.0/evidence/RULETIGHT", ".tmp_v8o0_ruletight", [40, 39, 26, 36]),
]
failed = False
for arm, evidence, work, expected in arms:
    for window, target in zip((11, 12, 17, 18), expected):
        completed = full = 0
        for seed in range(3):
            name = f"eval_{arm}_d900_w{window}_seed{seed}.jsonl"
            published = Path("Document/Update/Alpha") / evidence / name
            path = published if published.is_file() else Path(work) / "eval" / name
            try:
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if digest != manifest.get(published.as_posix()):
                    raise ValueError(f"unregistered or different SHA: {digest}")
                rows = [json.loads(line) for line in data.splitlines() if line.strip()]
                if len(rows) != 20 or {r["episode"] for r in rows} != set(range(20)):
                    raise ValueError("incomplete or duplicated episode IDs")
                for r in rows:
                    identity = (r["arm"], r["window"], r["policy_seed"], r["budget_decisions"])
                    if identity != (arm, window, seed, 900) or r["invalid"] or r["error"]:
                        raise ValueError(f"identity or invalid row at episode {r['episode']}")
                    survived = (r["frames"] == 1800 and r["decisions_executed"] == 900
                                and r["first_death_offset"] is None and r["terminal"] is None)
                    if r["survived"] is not survived:
                        raise ValueError(f"survival mismatch at episode {r['episode']}")
                completed += len(rows)
                full += sum(r["survived"] for r in rows)
            except (OSError, ValueError, KeyError, TypeError) as error:
                failed = True
                print("CHECK", path, error)
        print(arm, window, "full", full, "verified_rows", completed, "expected", target, 60)
        failed |= (full, completed) != (target, 60)
raise SystemExit(int(failed))
PY
```

**库存怎么报。** `bomb_decisions` 与 `bombs_remaining_at_emit` 给出请求步及请求前库存，`bomb_ignored` 是全局累计。逐次实际扣数优先用已有的两帧库存前后值核对；只有相邻请求前库存和 `bombs_end` 时，先报变化区间。只有在现存证据能唯一归属一次请求、并排除补给/掉命重置等其它库存变化时才归到该决策。扣数合计与 `bombs_start`、`bombs_end`、`bombs_used` 交叉核对。缺逐帧证据的“最后请求”不能直接填作“最后扣数帧”。[历史引擎循环](../../../../tools/rl/v830_bombpol.py)

## 3. 回执模板

~~~text
工单 V829-INFO-01　主机代号 / 时间与时区 / 执行代理
实际仓库 / HEAD / Python / 引擎、权重、runner 和恢复入口 SHA
本次实测命令 / 每项退出码 / 关键输出或缺失路径
历史 30 秒逐窗表（按 w11、w12、w17、w18，分列 G1 和 T2）
新开发祖先与阶段表（来源清单 SHA、可用数、曝光/重叠/未知数、接管验证能力）
耗雷例子与证据等级 / 历史吞吐 / 现有可用算时
可立即推进项 / 尚需补充的最小输入 / 已保留的封存用途
~~~

训练端本次实测、仅有历史记录和待核验项请分开填写。研究端据此给出窗口清单与预算；如符合条件，再由用户转交配对评测工单。
