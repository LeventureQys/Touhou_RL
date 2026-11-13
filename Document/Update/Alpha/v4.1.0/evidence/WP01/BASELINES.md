# WP01.2b — 非学习基线（random / still / fixed）真实引擎运行

- 工作包：v4.1.0 WP01.2b
- 工具：`tools/rl/v410_baselines.py`
- 测试：`tests/test_v410_baselines.py`
- 机器可读证据：`Document/Update/Alpha/v4.1.0/evidence/WP01/baselines.json`（schema `v410-baselines/1`）
- 引擎：`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`
  SHA256 `691bfd77b7eca3361aa8bb7af0977556a70cc73a070b29a2447222ba7a568ef6`
- 运行环境（WSL2 Ubuntu）：`Linux-6.18.58.1-microsoft-standard-WSL2-x86_64-with-glibc2.35` /
  Python 3.10.12 / numpy 2.2.6 / venv `/root/v400-venv/bin/python`
- 验收环境（Windows）：Python 3.12.10
- 活动声明：**未训练、未采集数据集、未修改 `touhou_rl/v400`·`v380`·`v390`·`third_party`·
  `v410` 协议与运行时代码、未 commit/push、未删除或覆盖旧文件**

结论一句话：4 个非学习策略 × 3 个 seed = 12 局真实引擎切片全部正常结束
（`status=pass`，退出码 0），**12/12 均以 `first_death` 终止（706–1806 帧）**，
没有一局撑到 `cap`、没有 `game_over`/`stage_clear` 样本；端到端吞吐
**6352 frame/s、3176 decision/s**（含 reset 的 wall 为 0.49–0.73 秒/局）。

## 1. 协议（与 v4.1.0 同一入口）

每个 `(policy, seed)` 独立一局：

| 项 | 值 |
| --- | --- |
| 环境 | `Th08Env(engine, data_dir, shm_name=..., privileged=True, decision_frames=2, max_frames=19887, role="shadow")` |
| 开局 | `env.reset(StageStart(stage=0, difficulty="normal", team=0, seed=SEED, resources="standard"))` |
| 决策周期 | 2 逻辑帧/次（`protocol.DECISION_FRAMES`） |
| 动作 | `(cand, bomb)`，cand∈0..17，本 WP 一律 `bomb=0` |
| 终止 | `boundary.classify_termination`（`game_over > stage_clear > first_death > cap`） |

策略映射（`policy_choice`）：

| policy | 动作 | 来源 |
| --- | --- | --- |
| `still` | `cand=0, bomb=0` | 固定 |
| `fixed_east` | `cand=3, bomb=0` | 固定（`cand 3` = E，`BTN_RIGHT`） |
| `fixed_west` | `cand=7, bomb=0` | 固定（`cand 7` = W，`BTN_LEFT`） |
| `random` | 每决策均匀取 `0..17`，`bomb=0` | `numpy.random.default_rng(seed * 1000 + 17)`，每局一条独立流，可复现 |

**特权等级**：`privilege = "engine_state_readonly"`。`privileged=True` 只用于终止判定与
资源日志（`game_over` 回复位、`F_DEATHS`、`F_STAGE`、`F_LIVES`/`F_BOMBS`/`F_SCORE`）。
终止事实被映射到 `classify_termination` 已有的像素事件名后再交给它判定，因此优先级
与该模块完全一致，没有第二份实现。**这是 engine-state 基线，不是严格像素能力测量**：
本 WP 没有读像素，也不声称像素策略能看到死亡或过面。

## 2. 实际执行的验收命令与退出码

### 2.1 单元测试与编译（Windows，仓库根）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v410_wp012b_pytest -q tests/test_v410_baselines.py
# 输出：39 passed in 0.17s
# 退出码：0

python -m compileall -q tools/rl/v410_baselines.py
# 输出：（无）
# 退出码：0
```

pytest 统计：**collected 39 / passed 39 / failed 0 / skipped 0 / error 0**。这 39 个是
**合成单元用例**（fake env，不启动引擎），不是游戏运行局数，也不构成能力证据。

### 2.2 真实引擎 smoke（WSL，实际 argv）

```bash
wsl.exe -d Ubuntu -- bash -c 'cd /mnt/d/WorkShop/Touhou_RL && xvfb-run -a /root/v400-venv/bin/python \
  tools/rl/v410_baselines.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --seeds 1 --policies still --max-frames 100 \
  --out /tmp/v410_base_smoke.json --quiet'
# SMOKE_EXIT=0（wsl.exe 传播的 Linux 退出码）
```

smoke 结果：`still/seed=1`，`end=cap`，`frames=100`，`decisions=50`，
`reset=0.420 s`，`loop=0.0148 s`，`wall=0.4345 s`，`fps_frames=6750.8`，
`fps_decisions=3375.4`，`status=pass`。
同一条命令先后跑过 5 次（含一次带进度输出、一次经管道），`end/frames/decisions`
每次一致，`fps_frames` 在 6750.8–7875.9 间波动（shadow 运行，短切片受进程启动与调度影响）。

### 2.3 主切片（WSL，写入证据文件，实际 argv）

```bash
wsl.exe -d Ubuntu -- bash -c 'cd /mnt/d/WorkShop/Touhou_RL && xvfb-run -a /root/v400-venv/bin/python \
  tools/rl/v410_baselines.py \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-dir /root/v400-data/game \
  --seeds 1,2,3 --policies random,still,fixed_east,fixed_west \
  --max-frames 19887 \
  --out Document/Update/Alpha/v4.1.0/evidence/WP01/baselines.json --quiet'
# MAIN_EXIT=0
```

主切片共执行三次（一次带进度输出、两次 `--quiet`，argv 除 `--quiet` 外相同），三次
**逐行 `frames`/`decisions` 完全一致**，`status` 均为 `pass`，退出码均为 0。本文件与
`baselines.json` 的数字取自最后一次（`--quiet`）运行。

## 3. 主切片逐局结果（12 局，全部真实引擎）

`end_frame` 即终止发生时的引擎帧号（`frames` 与它相同；这里没有补帧）。资源列取自
终止决策那一帧的状态快照。

| policy | seed | end | frames | decisions | deaths | lives | bombs | score | wall (s) | fps_frames | fps_decisions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 1 | first_death | 1410 | 705 | 1 | 2 | 3 | 7155 | 0.687 | 5360.0 | 2680.0 |
| random | 2 | first_death | 1806 | 903 | 1 | 2 | 3 | 19252 | 0.730 | 5467.1 | 2733.5 |
| random | 3 | first_death | 782 | 391 | 1 | 2 | 3 | 1200 | 0.508 | 6649.9 | 3324.9 |
| still | 1 | first_death | 706 | 353 | 1 | 2 | 3 | 1450 | 0.497 | 6980.5 | 3490.2 |
| still | 2 | first_death | 706 | 353 | 1 | 2 | 3 | 1450 | 0.490 | 7067.5 | 3533.7 |
| still | 3 | first_death | 706 | 353 | 1 | 2 | 3 | 1450 | 0.495 | 6876.5 | 3438.3 |
| fixed_east | 1 | first_death | 746 | 373 | 1 | 2 | 3 | 2060 | 0.499 | 6863.3 | 3431.7 |
| fixed_east | 2 | first_death | 746 | 373 | 1 | 2 | 3 | 2060 | 0.497 | 7012.9 | 3506.4 |
| fixed_east | 3 | first_death | 746 | 373 | 1 | 2 | 3 | 2060 | 0.498 | 6819.4 | 3409.7 |
| fixed_west | 1 | first_death | 714 | 357 | 1 | 2 | 3 | 1976 | 0.500 | 7016.6 | 3508.3 |
| fixed_west | 2 | first_death | 714 | 357 | 1 | 2 | 3 | 1976 | 0.499 | 6834.2 | 3417.1 |
| fixed_west | 3 | first_death | 714 | 357 | 1 | 2 | 3 | 1976 | 0.494 | 6678.6 | 3339.3 |

每局 `deaths_delta = 1.0`，`anomalies = []`，`status = "ok"`，`terminal_events = []`
（引擎在首次死亡帧没有给出事件名，终止由 `F_DEATHS` 增量判定）。

## 4. 每 policy 汇总

| policy | runs | frames mean | frames median | min–max | ends | decisions/s | frames/s | wall 合计 (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 3 | 1332.667 | 1410 | 782–1806 | `first_death`×3 | 2811.5 | 5623.1 | 1.926 |
| still | 3 | 706.0 | 706 | 706–706 | `first_death`×3 | 3487.0 | 6974.0 | 1.482 |
| fixed_east | 3 | 746.0 | 746 | 746–746 | `first_death`×3 | 3448.8 | 6897.5 | 1.494 |
| fixed_west | 3 | 714.0 | 714 | 714–714 | `first_death`×3 | 3420.2 | 6840.3 | 1.493 |
| **overall** | **12** | **874.667** | **730** | **706–1806** | **`first_death`×12** | **3176.2** | **6352.3** | **6.39** |

存活长度（帧，60 fps 下约 11.8–30.1 秒）：`random` 平均 1332.7 帧最长
（seed 之间差异来自随机动作流本身），三个固定策略在 706–746 帧之间，
`still` 最短。得分排序 `fixed_east (2060) > fixed_west (1976) > still (1450)`
在 3 个 seed 上完全一致：射击常开，横向移动改变了命中/被弹几何。

## 5. 终止类型

| 终止类型 | 次数 | 说明 |
| --- | --- | --- |
| `first_death` | 12 | `F_DEATHS` 首次增加，记该帧为 `end_frame` / `first_death_frame` |
| `game_over` | 0 | 12 局都在首次死亡就终止，没有消耗完残机，未出现 |
| `stage_clear` | 0 | `F_STAGE` 全程未变化（本切片在过面前就结束） |
| `cap` | 0 | smoke 的 100 帧切片是唯一 `cap` 样本；主切片无一局达到 19887 帧 |

按 `boundary.classify_termination` 的优先级（`game_over > stage_clear > first_death > cap`）
判定：本切片没有出现同一帧多终止事实的情况，`game_over` / `stage_clear` / `cap` 三条
真实路径**没有**被主切片覆盖，只在 `tests/test_v410_baselines.py` 里用 fake env 覆盖
（这不是真实运行证据，这里如实标注）。

## 6. 端到端性能

- 总吞吐（loop 计时，不含 reset）：**6352.3 frame/s、3176.2 decision/s**（12 局合计
  5248 次决策 / 1.652 s loop）。
- 单局 reset 0.387–0.424 s，是本切片 wall 的主要成分；loop 0.100–0.330 s。
- `fps_frames = frames / loop_seconds`、`fps_decisions = decisions / loop_seconds`；
  `wall_seconds` 是含 reset 的整局时间，两者在 JSON 里都记了，口径不混用。
- 最慢的单局是 `random/seed=1`（5360.0 fps；同一条命令的三次主切片分别为
  5792.5 / 5974.9 / 5360.0），仍 >5.3k frame/s；固定策略三个 seed 的 fps 区间为
  `still` 6876.5–7067.5、`fixed_east` 6819.4–7012.9、`fixed_west` 6678.6–7016.6，
  `random` 为 5360.0–6649.9；这些波动来自 loop 只有 0.10–0.33 s 时的进程调度抖动，
  不是策略差异。
- 口径提醒：这些数字是 shadow（null device）引擎的步进吞吐，**不含**渲染/像素编码
  成本，不能当作带渲染吞吐或训练吞吐。

## 7. 可复现性与观察

1. **动作可复现**：`random` 使用 `default_rng(seed*1000+17)`，同一 seed 的动作序列逐位
   相同（单测覆盖）；三次 `random` 各自覆盖了全部 18 个候选（`cand_counts` 的键数 = 18）。
2. **运行可复现**：三次主切片（不同 wall 时间）的 `frames`/`decisions` 逐局完全一致，
   `status` 均 `pass`。
3. **观察（不是结论）**：三个固定策略在 seed 1/2/3 上的 `frames` 与 `score` 完全相同
   （still 706/1450、fixed_east 746/2060、fixed_west 714/1976）。本切片无法区分
   “stage 0 normal 前 706 帧内 seed 不影响敌人行为” 与 “该 seed 没有被这条路径使用”。
   已有 WP01.1b 证据 `consistency.json` 显示 seed 1/2/3 的引擎状态哈希在第 0 帧就
   分叉，因此**不能**由本文件推断 seed 无效；这是需要单独验证的后续项。
4. **观察**：`StageStart(stage=0, resources="standard")` 开局时引擎发布的
   `F_LIVES = 2.0`，终止帧快照仍为 2.0。本工具只记录这两个真实快照，不解释该字段的
   语义（例如是否等于“可用残机库存”），也不据此推断残机结算发生在哪一帧。
5. 引擎 `sha256`、role、privilege、config、host、numpy 版本都写进 `baselines.json`，
   与引擎实际输出一致（`691bfd77…`）。

## 8. 限制（必读）

1. **engine-state 基线，不是严格像素能力**：终止来自 `game_over` 回复位与状态块里的
   `F_DEATHS`/`F_STAGE`，策略本身也只吃小动作集；本 WP 不读像素、不做像素奖励检测，
   不构成 v4.1.0 像素策略的能力证据。
2. **覆盖窄**：只跑了 stage 0 / difficulty normal / team 0 / `resources="standard"`；
   其他 stage、难度、team、practice 资源未覆盖。
3. **动作空间窄**：2 帧决策、仅 18 个候选、`bomb=0`；36 个学生动作
   （18 候选 × bomb∈{0,1}，射击恒开）未覆盖。
4. **终止路径不全**：主切片 12/12 都是 `first_death`，`game_over`/`stage_clear`/`cap`
   只有合成单测覆盖。
5. **没有渲染**：`role="shadow"`，`wall`/`fps` 不含渲染与像素编码，不能当作带渲染吞吐。
6. **基线不是能力门**：这些数字只描述“协议下不学习能活多久/多快”，不设最低存活阈值，
   也不能被当作“达到/未达到”某个能力等级的判据。
7. 短切片（100 帧 smoke、~700–1800 帧主切片）不构成稳定性结论；长时稳定性属于
   WP01.1b 的 `stability-2min.json` 范围，本文件不重复声称。

## 9. 变更文件

| 文件 | 类型 |
| --- | --- |
| `tools/rl/v410_baselines.py` | 新增 |
| `tests/test_v410_baselines.py` | 新增 |
| `Document/Update/Alpha/v4.1.0/evidence/WP01/BASELINES.md` | 新增（本文件） |
| `Document/Update/Alpha/v4.1.0/evidence/WP01/baselines.json` | 新增（真实运行产物） |

未修改白名单外任何文件；未改 `touhou_rl/v410/protocol.py`、`adapters.py`、`boundary.py`、
`evidence.py`、`parity.py`、`runtime.py`、`tools/rl/v410.py`、`tools/rl/v410_consistency.py`，
未改 `touhou_rl/v400/`、`v380/`、`v390/`、`tools/rl/v400_*`、`tests/test_v400_*`、
`third_party/`、`state.json`/`journal.jsonl`/`budget.jsonl`/`assets.json`。

## 10. 状态

`status = pass`：12/12 局真实引擎运行完成、无异常、无 `NaN`、退出码 0；
单元测试 39 passed、编译退出码 0。未训练、未采数据集、未 commit/push。
