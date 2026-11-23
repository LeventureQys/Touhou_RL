# v8.16.0 DEATHADJ-001 **REDO** — P0 判定：`S-nodiff`（**雷在死亡前已经用光**）

冻结预注册：[`PREREGISTRATION_DEATHADJ_001.md`](../../PREREGISTRATION_DEATHADJ_001.md)（FROZEN，2026-11-22，任何 v8.16.0 采集之前；本轮**未改**预注册）。

| 项 | 值 |
| --- | --- |
| 执行时刻 | 2026-11-22 14:06–14:45（Asia/Shanghai，训练端本机实测） |
| 训练端主机 | Windows `D:\WorkShop\Touhou_RL`，分支 `v400`，引擎经 WSL 启动 |
| 轨迹预算 | **900 决策 = 1,800 帧 = 30 s**（上一任写成 600，是它 0 状态的直接原因） |
| 标签视界 | **H = 120 / 240 / 600，三个都落盘** |
| 采集 | **720 状态 × 3 视界 = 2,160 行**，36 条死亡 episode × 完整 20 决策窗口 |
| 训练 / 评测 | **未进行**（判据：三个 H 的「有差异」都 < 20 % ⇒ `S-nodiff`） |
| 判定 | **`S-nodiff`** |

---

## 0. 一句话结论

> **死亡邻域确实采到了（H=600 时 82.92 % 的状态「36 路候选全部在 600 帧内死」），
> 但 `best bomb=1 − best bomb=0` 在三个视界上仍然恒为 0——
> 因为 720 / 720 个状态的 `bombs_remaining` 都是 0，`bomb=1` 只是 no-op。**

这不是「H 太短」的重复（那是 v8.15 的结论），而是**第二个、更硬的信息缺口**：
R5 行为策略在**死之前就把 3 颗雷用光**（36 / 36 条死亡 episode 的死亡决策处 `bombs_remaining = 0`），
所以「死亡前最后 20 个决策」这个采样区间里**没有一颗雷可花**，标签在结构上无法表达雷的价值。

---

## 1. 这轮改了什么，以及为什么 900 是承重的

| 项 | 上一任（已停） | 本轮 REDO |
| --- | --- | --- |
| 轨迹预算 | 600 决策（20 s） | **900 决策（30 s，1,800 帧）** |
| 每状态标签 | H = 120 | **H = 120 / 240 / 600** |
| 采样窗 | 死亡前 20 决策 | 不变 |
| 循环 | 12 格 × quota | `seed(0,1,2) → window(11,12,17,18) → episode(0..199)`，**连续 10 条不死就跳过该窗** |
| 状态目标 | 2,000 | **≥ 600 状态/视界**（12 格 × 60 = 720） |

**900 决策不是修辞上的修正，是必需条件**：

| 指标 | 实测（36 条死亡 episode） |
| --- | --- |
| 死亡决策 min / median / max | **423 / 717 / 895** |
| 死亡决策 > 600 的条数 | **30 / 36 = 83.3 %** |
| 死亡决策 > 300 的条数 | 36 / 36 |

即：若仍用 600 决策预算，**83 % 的死亡根本不会发生**，episode 会被「不死就换下一条」丢弃——
这正是上一任 19 / 19 条全活满、0 状态的机制（`.tmp_v8g0_deathadj/logs/smoke-collect.out`）。
上一任脚本另有一个**潜伏 bug**：`mode_collect` 在遇到死亡时执行 `rec["death_decisions"].append(...)`，
而 `rec` 里没有该 key（局部变量才叫 `death_decisions`）⇒ **一旦真的遇到死亡就 KeyError**。
它没暴露，是因为 19 条一条都没死。本轮已修（改用局部列表），并在 36 条死亡上验证通过。

---

## 2. 冒烟（先冒烟，再并行）

`./exec2.sh smoke`：seed 0、window 11，**1 条死亡 episode**，`--verify-h 1` 打开 H 嵌套直接对照。

```
[v450] window 11: parent up pid=1011777 prefix 2165/2165 checkpoints, 0 mismatches, PARK frame=8663, threads=1
[v8g0] s0 w11 ep0: DIED at decision 828 (829 decisions, 1657 frames, bombs 3->0, emits=9), window 809..828
[v8g0]   nesting verified at d809: {'120': 0, '240': 0} mismatches=0
[v8g0] s0 w11 ep0: DIED d828, 829 decisions, 3 states labelled, replay_checks=812, pass1=7.456s, pass2=12.74s
[v8g0] collect s0 w11: status=complete states=3 rows=9 states_by_H={'120': 3, '240': 3, '600': 3} ...
```

| 冒烟核对项 | 结果 |
| --- | --- |
| 900 决策内找到死亡 | ✅ 死亡决策 **828**（> 600：600 预算会丢弃它） |
| 三个 H 是否都落盘 | ✅ `rows=9`、`states_by_H={'120':3,'240':3,'600':3}` |
| H 嵌套（截断 = 直接跑） | ✅ 对第 1 个状态直接跑 H=120 与 H=240，36 动作逐个比对，**mismatch 0 / 0** |
| 死亡时剩余雷数 | `{0: 1}`（R5 3 颗放光，9 次触发） |
| 冒烟 P0（3 状态，仅打印） | H=120 critical 0/3、H=240 0/3、H=600 3/3；diff 0/0/0 |

---

## 3. P0：三个视界的四组数（训练端本次实测）

数据：`labels_deathadj.jsonl`（2,160 行 = 720 状态 × 3 视界，23,745,483 B，
sha256 `2bffac3be71f8166f72afee0d47588b0ed03d30cb9226f5e56b6320f08bb252d`）；
12 格 × 60 状态；**36 条死亡 episode + 10 条活满 900（丢弃）**；连续不死最多 3 条（阈值 10，未触发跳过）。
窗口完整性：`decisions_before_death = 0..19` 每一档都恰好 **36** 个状态 ⇒ 36 条窗口全部是完整 20 决策
（无 quota 截断）。

### 3.1 主表（每视界：critical 与「有差异」）

判据（预注册 §2 P0）：critical（`max(alive_frames) < H`）占 ≥ 50 %，**且**其中
`best bomb=1 > best bomb=0` 占 ≥ 20 %。

| H | states | **critical**（`max alive < H`） | critical 占比 | **有差异**（critical 内） | **有差异 占比** | 门槛 |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **120** | 720 | **59** | **8.19 %** | **0** | **0.00 %** | 未达 |
| **240** | 720 | **63** | **8.75 %** | **0** | **0.00 %** | 未达 |
| **600** | 720 | **597** | **82.92 %** | **0** | **0.00 %** | 未达 |

`delta = best bomb=1 − best bomb=0` 的分布（三个 H 相同）：
`n=720, mean=0.0, median=0.0, q1=0.0, q3=0.0, min=0, max=0, **stdev=0.0**`
（取值集合 `{0}`，`bomb1_better = 0`、`bomb1_equal = 720`、`bomb1_worse = 0`）。

补充分布（同一批 720 状态）：

| H | 36 路 safe 数 / 25,920 | 全 36 路活满的状态 | 全 36 路即死的状态 | 36 路平均 alive |
| ---: | ---: | ---: | ---: | ---: |
| 120 | 9,772（37.7 %） | 4 | 16 | 64.75 |
| 240 | 6,306（24.3 %） | 0 | 16 | 100.33 |
| 600 | 378（1.46 %） | 0 | 16 | 134.39 |

`max(alive_frames)` 在 H=600：mean 448.79、median 491.5、q1 404、q3 559.25、max 600。

### 3.2 按 `bombs_remaining` 的三段（+0）

| 段 | states | critical@120 | critical@240 | critical@600 | 有差异（任一 H） |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `bombs_remaining = 3` | **0** | 0 | 0 | 0 | 0 |
| `bombs_remaining = 2` | **0** | 0 | 0 | 0 | 0 |
| `bombs_remaining = 1` | **0** | 0 | 0 | 0 | 0 |
| `bombs_remaining = 0` | **720** | 59 | 63 | 597 | 0 |
| 合计 | 720 | 59 | 63 | 597 | **0** |

### 3.3 死亡时剩余雷数

| 指标 | 实测 | 预期 |
| --- | ---: | --- |
| 死亡决策处 `bombs_remaining` 分布（36 条 episode） | **`{0: 36}`** | 几乎全 0 ✅ |
| 采集区间内 `bombs_remaining` 分布（720 状态） | **`{0: 720}`** | — |
| `bomb=1` 标签真正扣掉一颗雷的次数 | **0** | — |
| `bomb=1` 标签的 no-op 次数 | **12,960 = 720 × 18**（全部） | — |
| `bomb=0` 标签误扣雷 / 计数异常 | **0 / 0** | — |

---

## 4. 为什么 `delta` 恰好是 0：机制而不是噪声

1. R5 规则在 3 颗雷时于 80 px 就触发、2 颗 60 px、1 颗 40 px（`v800_bomb.rule_gate`），
   一次死亡轨迹里触发 9–26 次（含无敌期内 no-op 的按压）；
2. 因此**在死亡前 20 个决策（40 帧）这个窗口里，雷已经被花光**：720 / 720 状态 `bombs_remaining = 0`；
3. 0 颗雷时 `cand_to_mask(cand, 1)` 与 `cand_to_mask(cand, 0)` 是**同一个动作**，
   36 路标签的奇偶两半逐位相同 ⇒ `best bomb=1 == best bomb=0`，`delta ≡ 0`（stdev 精确为 0.0）；
4. 连 R5 规则位 `rule_bomb` 在这 720 个状态上也**恒为 0**（没有雷可放）⇒ 若强行训练 A 臂，
   它的 `BCE_bomb(R5 规则)` 辅助项目标是常数 0，规则监督也是退化的。

所以 0 不是「标签解析 bug」或「聚合 bug」：`label_checks` 里 `bomb1_press_honoured = 0`
与 `states_without_a_bomb_left = 720` 是同一件事的两个视角，且 `safe == (alive_frames == H)`
在 2,160 行 × 36 路上逐条断言通过。

### 4.1 与 v8.15 的区别（不是同一个负结果）

| 轮次 | 采样 | H | critical | 有差异 | 机制 |
| --- | --- | ---: | ---: | ---: | --- |
| v8.15 | v8.11 触发点邻域（策略仍活着） | 120 | **0 / 10,008** | 0 % | 每个状态都有候选能活满 120 帧 |
| **v8.16 REDO** | **死亡前最后 20 决策** | 120 | **59 / 720 = 8.19 %** | 0 % | H=120 仍然太短（绝大多数状态有候选活过 120） |
| **v8.16 REDO** | 同上 | **600** | **597 / 720 = 82.92 %** | 0 % | **H 够长了，但雷已用光** |

即：H 扫描把 v8.15 的「H=120 内不可表达」拆分成了两件事——**H≥600 时 critical 成立，
但 bomb 分支在同一状态上没有可花的雷**。v8.15 的 RUN.md §8 明说「不能外推到 H > 120」，
本轮正好检验了这一点，并给出了不同的否决原因。

### 4.2 敏感性：去掉 36 个退化状态（死亡决策本身）

死亡决策处的状态是**死亡之后**读到的状态（36 路全部 0–1 帧即死，mean alive = 0.556），
属于退化状态。剔除它们后：

| 子集 | states | critical@120 | critical@240 | critical@600 | 有差异 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全部 | 720 | 8.19 % | 8.75 % | 82.92 % | **0** |
| 剔除死亡决策状态 | 684 | **3.36 %** | **3.95 %** | **82.02 %** | **0** |
| 仅死亡决策状态 | 36 | 100 % | 100 % | 100 % | **0** |

结论与主表一致；H=120 / 240 的 critical 占比在剔除后退到 3.4–4.0 %（更差），
H=600 基本不变（82.0 %）。`p0_sensitivity.json` 是原始输出。

---

## 5. 判定

按工单与预注册 §2.1：

> **三个 H 的「有差异」都 < 20 % ⇒ `S-nodiff`：只发布统计，不训练。**

因此本轮**没有**训练 A/B、**没有**评测 300/600/900、**没有** `bomb_usage.json`（P1–P4 全部未执行）。
这不是「采得不够」：720 状态、36 条完整死亡窗口、把 H 从 120 拉到 600，
`delta` 仍然是精确的 0；缺的是**带雷的死亡状态**。

### 5.1 下一轮的可检验假设（本轮**未**检验，需要新的预注册）

1. **换行为策略的「省雷」变体**：R5 的 80/60/40 px 太早放雷；改成「只在 ≤40 px 或 ≤20 px 放」，
   让它**带着 1–3 颗雷死**，死亡邻域才会出现「放雷能活 / 不放雷就死」的状态；
2. **或改用不放雷的 `S` 轨迹去死**（S 从不放雷）时采死亡前 20 决策——那样 `bombs_remaining` 天然 ≥ 1；
3. **采样窗改成 `death-1 … death-20`**（把死亡决策本身那个退化状态排除），并在记录里区分
   「死前状态」与「死后状态」。

这三条都是**设计变更**，不属本轮授权范围，本代理没有擅自执行。

---

## 6. 门与约束核对（全部本机实测）

| 门 / 约束 | 结果 | 证据 |
| --- | --- | --- |
| pinned imports（12 个文件 sha256） | **12/12 match**，`all_match=True` | `preflight.json` |
| 引擎 sha256 | `691bfd77…7a568ef6` match | `preflight.json` / `ckpt_assertions*.json` |
| 特征维 / 动作表 | 106 / 36（`action = cand*2 + bomb`） | `preflight.json` |
| ckpt 断言 `4abee069… / a93746bb… / c69960fc…` | **3/3 match**（start 与 final 两次） | `ckpt_assertions.json`、`ckpt_assertions_final.json` |
| **D3 保真门 5 条** | **5/5 passed**：`hello_role_shadow`、`debug_threads_1`、`prefix_2165_checkpoints`、`r24_18_constants`（18/18 常数）、`park_frame_8663` | `fidelity_d3.json`（`all_pass=True`） |
| 音频（AGENTS.md §9.3） | `SDL_AUDIODRIVER=dummy`、`DISPLAY/WAYLAND_DISPLAY` unset；每个 data dir 用 `v400_make_cfg.py --out` 生成并 `--inspect` 四字段全 0（否则拒启）；三处审计合计 **27 个 cfg / 27 muted / 0 audible**（主 run 24 + 冒烟 1 + D3 2） | `audio_audit.json`、`audio_audit_smoke.json`、`audio_audit_fidelity.json` |
| 本轮引擎进程残留 | **0**（`th08-modern` 0 个，filter = 本轮 work dir） | `process_residue.json` |
| pass-2 回放保真 | 25,861 次 frame+nearest 逐决策核对，**0 mismatch** | `collect/seed*_w*.json` |
| H 嵌套（截断 = 直接跑） | 冒烟 1 状态 + **12/12 格首状态**直接跑 H=120/240 对照：**936 个 action×H 比对，0 mismatch**；stats 又对 720 状态重验 `alive(H)=min(alive(600),H)` | `collect/*.json` 的 `nesting_verified`、`deathadj_stats.json` 的 `nesting_checks` |
| 标签不变量 | 2,160 行 × 36 路 `safe == (alive_frames == H)` 全过；`bomb0` 误扣雷 0；`bomb1` 计数异常 0 | `deathadj_stats.json` |
| 轮内并行 | 12 路（3 seed × 4 window），单格 wall 226–335 s，单状态标签 mean 3.1–4.4 s | `collect/*.json` |

---

## 7. 文件与 sha256

全部位于 `.tmp_v8g0_deathadj/`，并复制到 `Document/Update/Alpha/v8.16.0/evidence/DEATHADJ/`：

| 文件 | bytes | sha256（前 16） |
| --- | ---: | --- |
| `labels_deathadj.jsonl`（2,160 行） | 23,745,483 | `2bffac3be71f8166` |
| `deathadj_stats.json`（**主产物**） | 59,946 | `79f1a3b2de932e2b` |
| `deathadj_stats_pass1_quota50.json`（第一遍 600 状态） | 59,855 | `0f7f4981572d9f0f` |
| `p0_extras.json` | 4,761 | `be8dfe00cd6e55ab` |
| `p0_sensitivity.json` | 1,480 | `43ec27949865a96c` |
| `preflight.json` | 9,014 | `4fa116cc2b70c5da` |
| `ckpt_assertions.json` / `_final.json` | 4,013 / 4,013 | `6ad511857e1fa8ee` / `a428abfaf95fa172` |
| `fidelity_d3.json` | 26,699 | `0580437cfce2d95b` |
| `audio_audit.json` | 4,370 | `e993b0718d82078c` |
| `process_residue.json` | 692 | `c4340eb15c0d08e2` |
| `recount_v811.json`（v8.15 负基线复算 0/10,008） | 1,749 | `4bb553df4f069aae` |
| `commands.tsv`（每轮 append：采集/门禁 40 条 + publish 自身 2 条；其 sha 以 `manifest.json` 为准） | 12,853 | `5d20d5ed8f66d78f` |
| `collect/seed{0,1,2}_w{11,12,17,18}.json`（12 个格摘要） | — | 见 manifest |

完整清单（含本文件自身）见 `manifest.json`；`work_manifest.json` 覆盖整个 `.tmp_v8g0_deathadj/`。
新增/改写的代码只有 `tools/rl/v8g0_deathadj.py`（REDO 版：900 + 三 H + 10 条跳过 + 逐 H P0）
与 `.tmp_v8g0_deathadj/` 下的执行/分析脚本；**未改** `touhou_rl/**`、`third_party/**`、
任何既有版本目录/预注册/证据、其它 runner；**未** `git add/commit`；**未**删除/移动/覆盖任何资产。

---

## 8. 复现命令

```bash
cd /mnt/d/WorkShop/Touhou_RL
export SDL_AUDIODRIVER=dummy; unset DISPLAY WAYLAND_DISPLAY
export PYTHONPATH=/mnt/d/WorkShop/Touhou_RL

./.tmp_v8g0_deathadj/exec2.sh preflight    # pins + engine + 设计
./.tmp_v8g0_deathadj/exec2.sh start        # ckpt 断言（4abee069/a93746bb/c69960fc）
./.tmp_v8g0_deathadj/exec2.sh fidelity     # D3 5 条门
./.tmp_v8g0_deathadj/exec2.sh recount      # v8.15 的 0/10,008 复算（无引擎）
./.tmp_v8g0_deathadj/exec2.sh smoke        # 1 条死亡 episode + 三 H 落盘 + H 嵌套验证
QUOTA=60 RUNPREFIX=3 TAGPREFIX=v8g0r2b ./.tmp_v8g0_deathadj/exec2.sh collect   # 12 路
./.tmp_v8g0_deathadj/exec2.sh stats        # 逐 H P0；S-nodiff 时 rc=4
./.tmp_v8g0_deathadj/exec2.sh audit        # 音频 + 进程残留
./.tmp_v8g0_deathadj/exec2.sh final        # ckpt 断言（final）
./.tmp_v8g0_deathadj/exec2.sh publish      # 原始证据 -> Document/.../evidence/DEATHADJ/
```

离线分析（无引擎）：`python .tmp_v8g0_deathadj/p0_extras.py`、
`python .tmp_v8g0_deathadj/p0_sensitivity.py`、`python .tmp_v8g0_deathadj/cell_progress.py`、
`python .tmp_v8g0_deathadj/selftest_stats.py`（合成 3 状态的 stats 自测，见 §9 偏差 4）。

---

## 9. 偏差（deviations）

| # | 偏差 | 影响 |
| --- | --- | --- |
| 1 | **采了两遍**：第一遍 12 格 × 50 状态 = 600 状态，但每格第 3 条 episode 的 20 决策窗口被 quota 截成前 10 个；第二遍改成 12 格 × 60 = **720 状态，36 条全部是完整 20 决策窗口**，并以第二遍为准 | 无。第一遍的统计与标签 sha（`d33e5ee9…`）保留在 `deathadj_stats_pass1_quota50.json`；第一遍的标签文件已被第二遍覆盖，其逐格日志在 `logs/collect2-*.out`。两遍结论一致（diff 全 0），第二遍严格满足「死亡前最后 20 个决策」 |
| 2 | 采集窗用 **11/12/17/18（= dev 窗）**，按工单指定 | **训练/评测窗重叠**（见 §10）。本轮未训练，所以没有实际泄漏的模型结果；若未来用这批标签训练并在同样 4 个 dev 窗评测，必须写明重叠 |
| 3 | 10 条连续不死阈值**未触发**（每格连续不死最多 3 条），所以没有窗口被跳过；36 条死亡全部在 900 决策内 | 无 |
| 4 | 额外产出 `selftest_stats.py` + 合成 stats（`selftest_stats_synthetic.json`）、`p0_extras.json`、`p0_sensitivity.json`、`cell_progress.py`、`exec2.sh`（预注册 §4 未列） | 无，只增不改；合成自测用来验证 stats 的算术（3 个手工状态：全活满/全早死/仅 bomb=1 更久），**不是实验证据** |
| 5 | `bombs_at_death` 的 36 条全部为 0，导致 P0 的「三段分布」全是 0 段 | 无。这正是本轮要报的第三组数；预期就是「几乎全 0」，实测是「全部 0」 |
| 6 | 死亡决策处的 36 个状态是**死后状态**（36 路全部 0–1 帧即死），本轮仍按预注册把它们计入 | 已在 §4.2 给出剔除后的敏感性子集：结论不变 |

---

## 10. 不能声称什么

- **不能声称「雷没有价值」**——只能声称：在**这批 S+R5 死亡邻域状态**（720 个、dev 窗 11/12/17/18）、
  **这三个视界（120/240/600）**上，`bomb=1` 与 `bomb=0` 不可区分；原因是**雷已用光**，
  不是「雷无用」。带雷死亡的状态下标签会长什么样，本轮**未测**。
- **不能声称 H 扫描失败**——恰恰相反：H=600 把 critical 从 8.19 % 抬到 82.92 %，
  证明 120 帧确实截断了「必死」的观测；失败的是「雷还在不在」。
- **不能外推**到别的行为策略（尤其是不放雷的 `S`）、别的窗、别的 H。
- **不能把 900 决策与 600 决策做预算优劣比较**——本轮没有跑 600 的对照臂，
  只有「36 条死亡里 30 条在 600 之后」这一条分布事实。
- **不能把 12 个 (seed, window) 格当独立样本**：格间共享引擎/策略/窗口，只能并列，不能当 12 个独立重复。
- **训练/评测窗重叠**：本轮采集窗 11/12/17/18 与预注册的评测窗是**同一批 dev 窗**；
  若后续用这批标签训练并在同样窗评测，**不能**据此声称跨窗口泛化；泛化需要 held-out 窗
  （v4.4.0 的 train 13–16 或新窗）。
- **不能声称通关**、**不能**把不同预算的均值直接比较；P1–P4 本轮**全部未执行**（S-nodiff 分支）。
