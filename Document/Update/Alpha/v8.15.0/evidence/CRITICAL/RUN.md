# v8.15.0 CRITICAL-001 — P0 判定结果：**S-nodiff**

冻结预注册：[`PREREGISTRATION_CRITICAL_001.md`](../../PREREGISTRATION_CRITICAL_001.md)（FROZEN，2026-11-22，在任何采集之前）。

| 项 | 值 |
| --- | --- |
| 执行时刻 | 2026-11-22 13:51–13:57（Asia/Shanghai，训练端本机实测） |
| 训练端主机 | Windows `D:\WorkShop\Touhou_RL`，分支 `v400` |
| 采集 | **零**（prereg §3 预扫，复用 v8.11.0 已发布的 10,008 状态 × 36 路标签） |
| 引擎 | **未启动**（`engine_started = false`；结束时 `th08*` 进程数 **0**） |
| 训练 / 评测 | **未进行**（P0 不成立，按 prereg §2.1 `S-nodiff` 停） |
| 判定 | **`S-nodiff`** |

---

## 0. 一句话结论

> **在策略自己的轨迹上，不存在「必须放雷」的状态。**
> 10,008 个状态里，**critical 状态 = 0 个**（`max(alive_frames) < H=120` 一次都不成立），
> 因此 `best bomb=1 − best bomb=0` **恒为 0**，bomb 头拿不到任何信号。
> 按 prereg §2.1，判定 **`S-nodiff`：H=120 内雷的价值不可表达**，不训练、不补采。

这不是"采得不够密"，是**这批状态里没有可被表达的东西**：候选头在 2 秒（120 帧）内总能找到一个活满的方向。

---

## 1. P0 的三组数（训练端本次实测）

判据（prereg §2 P0）：critical ≥ **1,200** 且三段（`bombs_remaining` ∈ {3,2,1}）各 ≥ **200**，且
`best bomb=1 > best bomb=0` 的状态占比 ≥ **20%**。

### 1.1 critical 数量

| 指标 | 实测 | 判据 | 结果 |
| --- | ---: | ---: | :---: |
| states | **10,008** | — | — |
| **critical（`max(alive_frames) < 120`）** | **0** | ≥ 1,200 | **未达** |
| critical 占比 | 0.0000% | — | — |

### 1.2 critical 按 `bombs_remaining` 的三段分布

| 段 | states（全体） | critical | 目标 | 结果 |
| --- | ---: | ---: | ---: | :---: |
| `bombs_remaining = 3` | 902 | **0** | ≥ 200 | 未达 |
| `bombs_remaining = 2` | 6,382 | **0** | ≥ 200 | 未达 |
| `bombs_remaining = 1` | 2,724 | **0** | ≥ 200 | 未达 |
| 合计 | 10,008 | **0** | ≥ 1,200 | 未达 |

`critical_by_bombs = {}`（空：没有任何一段贡献过 critical 状态）。

### 1.3 「有差异」比例

| 指标 | 实测 | 判据 | 结果 |
| --- | ---: | ---: | :---: |
| `best bomb=1 > best bomb=0`（更好） | **0** | — | — |
| `best bomb=1 = best bomb=0`（相等） | **10,008** | — | — |
| `best bomb=1 < best bomb=0`（更差） | **0** | — | — |
| **有差异占比** | **0.0000%**（0 / 10,008） | ≥ 20% | **未达** |

---

## 2. 比 0 更深的一层：为什么恰好是 0

`max(alive_frames)` 的分布**不是"多数 120、少数不足 120"**，而是**退化成一个点**：

| 分布 | min | q1 | median | q3 | max | mean | stdev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `max(alive_frames)`（每状态取 36 路最大） | 120 | 120 | 120 | 120 | 120 | 120.0 | **0.0** |
| `best bomb=1`（只看奇数 18 路） | 120 | 120 | 120 | 120 | 120 | 120.0 | 0.0 |
| `best bomb=0`（只看偶数 18 路） | 120 | 120 | 120 | 120 | 120 | 120.0 | 0.0 |
| `best bomb=1 − best bomb=0` | 0 | 0 | 0 | 0 | 0 | 0.0 | **0.0** |

关键的一行是第三行：**即使把 bomb=1 的 18 路全部丢掉、只看 bomb=0 的 18 路，10,008 / 10,008 个状态仍然都有
一个候选活满 120 帧**（`states_best_bomb0_eq_H = 10008`，`states_best_bomb0_lt_H = 0`）。

⇒ 「必须放雷才能活」在这批状态上出现 **0 次**。`delta` 的可取值集合是 `{0}`（`delta_distinct = [0]`），
所以任何 36 路 BCE 的 bomb 维度都只有一个常数目标，`best bomb=1 > best bomb=0` 在任何权重、任何采样密度、
任何 loss 形式下都不可能出现——**这是标签的信息缺口，不是优化问题**。

### 2.1 与 v8.11 的逐项对账（证明读的就是同一份数据）

| 指标 | v8.11.0 `sample_stats.json` | 本轮复算 | 一致 |
| --- | ---: | ---: | :---: |
| states | 10,008 | 10,008 | ✅ |
| 段分布 3 / 2 / 1 | 902 / 6,382 / 2,724 | 902 / 6,382 / 2,724 | ✅ |
| 36 路全 safe 的状态 | 65.67% | 6,572 = 65.6675% | ✅ |
| 36 个动作平均 `alive_frames` | 118.56 | 118.5613 | ✅ |
| episodes | 160 | 160 | ✅ |
| labels | 360,288 | 10,008 × 36 | ✅ |

源文件身份：`labels_events.jsonl` **sha256 `ec9c802a87a8beb9…`**（97,313,721 bytes，10,008 行），
与 v8.11.0 `sample_stats.json` 的 `labels_file.sha256` **逐位相同**（`sha256_match = true`）。

---

## 3. 独立复核（两个不共享代码路径的实现）

主实现 `tools/rl/v8f0_critical.py` 走 `json.loads` 全量解析；独立实现
（`.tmp_v8f0_critical/independent_check.py`，一次性脚本）**不 import 主实现、不用 `json.loads`**，
而是从原始行文本里切出 `"alive_frames": [...]` 与 `"bombs_remaining"` 再重算。

| 复算项 | 主实现 | 独立实现 | 一致 |
| --- | ---: | ---: | :---: |
| rows | 10,008 | 10,008 | ✅ |
| critical_states | 0 | 0 | ✅ |
| diff（bomb=1 更好） | 0 | 0 | ✅ |
| equal | 10,008 | 10,008 | ✅ |
| worse | 0 | 0 | ✅ |
| delta 取值集合 | `{0}` | `{0}` | ✅ |
| **per-state digest** | `559b4ccd400bceced29868128708393a49338a659c19d0916a17420e36d38813` | 同左 | ✅ |

两份实现给出**逐位相同的 per-state digest**（`sha256` over 每行 `max_alive,best_bomb1,best_bomb0,bombs_remaining`），
所以 0 不是某一路解析或聚合的 bug。独立实现原始输出见
[`independent_check.json`](independent_check.json)。

### 3.1 布局约定不是承重的

`action = cand*2 + bomb`（`tools/rl/v830_bombpol.py::split_action` 即 `cand, bomb = divmod(action, 2)`），
`alive_frames` / `safe` 按该 action id 升序索引（`v830_bombpol.py::build_arrays` 把 `row["safe"]` 直接灌进标签矩阵，
并断言 `safe[a] == (alive_frames[a] == H)`）。因此 **bomb=1 是奇数半、bomb=0 是偶数半**。

主实现对**每一行**都重新验了这条对齐不变量（`safe[a] == (alive_frames[a] == H)`，10,008 × 36 个断言全过）。
另外，"交换两半"的镜像读法给出 `better = 0 / worse = 0`——**结论与约定无关**（`critical` 与 `equal` 在交换下不变）。

---

## 4. 为什么**不补采**

prereg §3 允许在 critical 数量不足但 > 0 时补采。**本轮不适用，而且补采没有意义**：

1. critical 数量是 **0**，不是"不足"；prereg §3 的补采触发条件（>0 但 <1,200）不成立。
2. 补采要在**同一条策略轨迹**（`S` stochastic + R5 规则）上继续采，而那批轨迹上 `best bomb=0 == 120` 是
   普遍成立的 ⇒ 再采 1,000 个同分布状态，预期 critical 仍是 **0**。
3. 预注册 §2.1 已把这种情况命名为 `S-nodiff` 并指定了动作：**判「H=120 内雷的价值不可表达」，转在线 RL（v8.8）**。

真正缺的不是数量，是**状态类型**：要采到 critical 状态，必须去「躲不掉」的局面里采——
即**死亡前的最后 ~120 帧**，那里 18 个候选才会全部死。这属于下一轮的设计变更（改采样分布），
不属于本预注册的授权范围，本代理不擅自改设计。

---

## 5. 本轮没有起引擎，所以哪些门不适用

| 门 | 状态 | 说明 |
| --- | --- | --- |
| D3 保真门 5 条 | **不适用** | 本轮没有任何新引擎会话；5 条门（`HELLO role=shadow`、`DEBUG threads=1`、`2165 前缀 checkpoint`、`R24 18 常数`、`PARK frame 8663`）只在启动引擎时才需要复核。v8.11.0 的 `fidelity_d3.json` 仍是最近的已记录 `all_pass = true` 会话，本轮未触碰。 |
| ckpt 断言 `4abee069… / a93746bb… / c69960fc…` | **不适用** | 没有加载任何 checkpoint（未训练、未评测）。 |
| 音频（AGENTS.md §9.3） | **不适用（无引擎可静音）** | 本轮零引擎启动；未生成/未修改任何 `th08.cfg`；结束时 `th08*` 进程 **0**。若下一轮起引擎，§9.3 三件事照旧必须在起之前做完。 |
| 进程残留 | **通过** | 结束时 `th08*` / 引擎进程数 = **0**。 |

---

## 6. 落盘与身份

全部位于 `Document/Update/Alpha/v8.15.0/evidence/CRITICAL/`（新增目录，未改任何既有版本目录）：

| 文件 | bytes | 说明 |
| --- | ---: | --- |
| `critical_stats.json` | 6,403 | **主产物**：P0 三组数 + 分布 + 段分布 + 判定 |
| `labels_critical.jsonl` | 0 | `rows = 0`（critical = 0），sha256 `e3b0c442…`（空载荷），**空缺可审计而非静默** |
| `independent_check.json` | 618 | 独立实现的原始复算结果 |
| `RUN.md` | 本文件 | 判定与机制 |
| `manifest.json` | — | 上述文件的 sha256 |

新增代码：`tools/rl/v8f0_critical.py`（只读扫描器，`--mode scan`；P0 通过退出 0，失败退出 4）。
未改 `touhou_rl/**`、`third_party/**`、任何既有版本目录/预注册/证据、任何既有 runner（只 import/参考其约定，未复制其写路径）。
镜像校验：`git status --porcelain -- tools/rl/v8f0_critical.py Document/Update/Alpha/v8.15.0` 仅两项**
`?? tools/rl/v8f0_critical.py`** 与 **`?? Document/Update/Alpha/v8.15.0/`**；`touhou_rl/` 下现存的 `M` 项都是
2026-11-05 之前的历史改动（mtime 可查），与本轮无关。

---

## 7. 复现命令

```powershell
cd D:\WorkShop\Touhou_RL
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'

# 主判定（零采集；退出码 4 = P0 不成立 = S-nodiff）
python tools\rl\v8f0_critical.py --mode scan
# -> [v8f0] labels ...\labels_events.jsonl bytes=97313721 sha256=ec9c802a87a8beb9
# -> [v8f0] read 10008 states in 1.186s
# -> [v8f0] P0 FAIL verdict=S-nodiff states=10008 critical=0 diff=0 (0.0000%)

# 独立复核（不共享代码路径）
python .tmp_v8f0_critical\independent_check.py
```

---

## 8. 不能声称什么

- **不能**声称"critical 采样不可行"——本轮**没有采到任何 critical 状态**，只是证明**这批轨迹上不存在**；
  换采样分布（死亡前 N 帧）后 critical 是否存在，仍未被检验。
- **不能**声称"雷没有价值"——只能声称"**在 `S`+R5 轨迹的 10,008 个状态上、H=120 内**，标签无法表达雷的价值"。
- **不能**把本条结论外推到 H > 120：`max(alive_frames) == 120` 是**被 horizon 截断**的观测，
  H 更长时 `best bomb=0` 可能先死。这正是下一轮该做 H 扫描（prereg §2.1 `S-ok` 的后续）而非直接判死的理由。
- **不能**声称 A/B 两臂有任何结果：本轮未训练。
- 不能声称通关、不能把不同预算的均值直接比较、不能把 12 格当独立样本。

---

## 9. 偏差（deviations）

| # | 偏差 | 影响 |
| --- | --- | --- |
| 1 | prereg §4 列出的 `train_A/B_*`、`eval_A/B_d{300,600,900}_seed*`、`bomb_usage.json` **未生成** | 无。P0 不成立，prereg §2.1 `S-nodiff` 明确"不训练"；这些产物在本分支下不存在是正确的 |
| 2 | 额外产出 `independent_check.json`（prereg §4 未列） | 无。为复核用，只增不改 |
| 3 | `labels_critical.jsonl` 为空文件（0 行） | 无。`critical = 0` 的直接后果；用 `rows=0` + 空载荷 sha256 显式记录，避免"文件缺失"被误读成"忘了写" |
| 4 | D3 5 条门 / ckpt 断言 3 个 hash 未执行 | 无。本轮零引擎、零 ckpt；见 §5 |
