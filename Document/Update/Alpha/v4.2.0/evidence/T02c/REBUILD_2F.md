# T02c — 2 帧语料重建（原掩码重放 + 逐帧抓像素）

日期：2026-11-13，Asia/Shanghai。资产主机（本机 WSL2），`cwd=/mnt/d/WorkShop/Touhou_RL`，
Python `/root/v400-venv/bin/python`（3.10.12），引擎 `rl_v3`
`/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern`
（sha256 `691bfd77…7a568ef6`，4,836,008 B），git HEAD `8684ae10…f58a274d`。

**范围声明**：本文件只回答 [README §5 WP420-01](../README.md) 的 L1 腿——
「新 S1 数据每个 2 帧决策槽都要有对应标签」中**时间/动作索引一致**的那一半，
以及 G420-0 的 HASH 重建腿在**重建语料**上的复核。它**不是** G420-0 整门通过：
部署输入边界测试、在线前向一致性等条目不在本文件范围内。
本文件不修改任何旧语料、旧证据或旧状态。

---

## 0 结论摘要

1. **槽 1 现在有自己的观测**：3 个 shard 共写出 **19,772 个 2 帧槽**
   （期望槽位 19,796，冷启动排除 24），每个槽都有自己的 4 帧连续 `play` 栈
   （帧 `d-4..d-1`）、自己的 HUD（帧 `d-1`）、自己的已执行动作、8 条真实已执行历史
   （帧 `d-16..d-2`，`a0,a1` 交错）。旧语料这四项对槽 1 全部缺失。
2. **L1 腿成立**（在重建语料上）：逐槽 span/history/4 帧连续/key timing 断言
   **0 违反、0 未解释差异**（写出进程与**另一新进程**各复核一遍，均是 0）；
   HASH 校验 **9,898/9,898 相等**（每决策 1 个 `hash_post` 加 3/3 `RESET` hash）；
   逐帧观察 39,592 个引擎 HASH。
3. **额外强证据**：重建的槽 0 观测与**已存旧语料逐字节相同**——
   `play` 9,886/9,886、`hud` 9,886/9,886（3 个 shard 的全部可比值）。
   这证明逐帧抓帧的索引没有平移/错位，不是"看起来像"的近似。
4. **标签边界**：槽 1 的 `label_source="executed_slot1"`、`label_verified=false`，
   且**不携带任何 scores 字段**（3 个 shard 实测 0 条）。它是"教师执行过但未评分"的动作，
   可作 BC 行为目标，**不得**计入 G420-1 的 verified 标签；
   verified 覆盖率仍为 **9,886/19,796 = 49.94%**（≈50%，未改变）。
5. **口径偏离（必须一起引用）**：(a) 每个 shard 有 8 个槽（决策帧 0,2,…,14）被
   **排除而非填充**，因为它们的 4 帧栈或 8 决策历史会越过第 0 帧；
   (b) seed 12 的旧 shard 只有 **3,298** 决策（`stop_reason=stage_changed`、
   `stage_clear=true`），不是任务书假设的 3,300。

---

## 1 方法（可复算）

### 1.1 一句话

对每个旧 shard：`RESET`（stage/difficulty/team/seed 与 `shard.json` 一致）
→ 逐决策把 `step_frames` 的 **4 个逐帧掩码**推进引擎 → **每帧**抓
`encode_planes_v3(env.shm.pixels())` → 每 4 帧产出**决策帧 `4i` 与 `4i+2` 两个槽**。
不复制标签、不插值 logits、不平移分数、不填充冷启动。

实现：`touhou_rl/v420/rebuild.py`（新文件）；入口
`python -m touhou_rl.v420.rebuild --seeds <seed>`。

### 1.2 每个槽写什么

| 字段 | 来源 | 说明 |
| --- | --- | --- |
| `play` 4 帧栈 | 该槽决策帧前 4 个**已渲染**帧 `d-4..d-1`（oldest-first） | 旧语料每块只存 1 行（帧 `4i-4..4i-1`），槽 1 无 |
| `hud` | 帧 `d-1` 的 HUD | **槽 1 新获得的东西**：帧 `4i+1` 而非 `4i-1` |
| `cand` / `mask` | `mask_to_cand(step_frames[0])`（槽 0）、`mask_to_cand(step_frames[2])`（槽 1） | mask 原样保留；`step_frames[3]` 在 137 行被 `shoot_release` 清 SHOOT 位，故只解前导帧 |
| `history` | 帧 `d-16,d-14,…,d-2` 的**已实际执行**候选 | 8 条、`a0,a1` 交错、16 帧跨度（部署节拍），不含当前/未来 |
| `label_source` | 槽 0 = `teacher_slot0`；槽 1 = `executed_slot1` | 见 §4 边界 |
| `label_verified` | 槽 0 = 原行 `verified_mask[first_action]`（3 个 shard 3,300/3,300 为真）；槽 1 = `false` | |
| `decision_frame` / `commit_frames` / `last_visible_render_frame` / `key_send_frame` / `key_effective_frame` / `duration_frames` | `d` / `(d,d+2)` / `d-1` / `d` / `d+1` / `2` | 与 `contracts.check_key_timing` 一致 |

槽 0 额外保留教师自己的 `teacher_scores`(18)、`teacher_margin`、
`teacher_verified_mask`(18)——那是**同一决策**的原始教师标签，不是平移；
槽 1 **一个 scores 键都没有**。

### 1.3 冷启动口径（排除，不填充）

`FIRST_DECISION_FRAME = max(FRAME_STACK, HISTORY_SPAN_FRAMES) = 16`。
决策帧 `0,2,…,14` 的槽被**排除并计数**：`d=0,2` 没有 4 个已渲染帧，
`d=4..14` 的 8 决策历史会要求帧 `-2` 以前。旧写入者当年用重复第一帧填充（`pad=3`），
`contracts.check_contiguous_render_frames` 明确拒绝；用假候选填 8 条历史更属于"凭空补"。
所以：**每 shard 期望槽位 6,600 / 6,596，写出 6,592 / 6,588，排除 8**（合计 24 / 19,796）。

### 1.4 HASH 规则（fail-closed）

1. `RESET` 后 `last_reset_hash` 必须等于首行 `hash_pre`（3/3 成立）；
2. 每决策第 4 帧的引擎 HASH 必须等于该行 `hash_post`——**先校验、后写该帧触发的槽**；
   不一致立刻抛错，本次输出写进 `2f.tmp-<pid>/`，异常路径删除临时目录，
   **最终路径不会出现**（单测 `test_hash_mismatch_*` 钉住这条）；
3. 输出先写在临时目录，**自检通过后**才 `rename` 成 `2f/`。

### 1.5 数据目录（只读源，绝不写语料）

每次运行把 `/root/v400-data/s1c/<shard>/env` 复制的**私有副本**给引擎：
`score.dat`、`th08.cfg` 复制（引擎会写），`th08.dat`/`thbgm.dat` 保持指向
`/root/v400-data/game/*` 的符号链接。源 `score.dat` 的 sha256 **运行前后相同**
（§6 `assets_readonly`），旧语料 `labels.jsonl` 的 sha256/mtime 未变。

### 1.6 命令

```bash
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v420_t02c/run_seed.sh 10   # 3.7 min
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v420_t02c/run_seed.sh 11   # 3.8 min
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v420_t02c/run_seed.sh 12   # 4.0 min
```

脚本内部：`xvfb-run -a -s "-screen 0 1024x768x24" env PYTHONPATH=$ROOT420 $PY420 -m
touhou_rl.v420.rebuild --repo-root $ROOT420 --v420-data $V420_DATA --seeds <seed>
--out-json $V420_DATA/T02c/logs/rebuild_seed<seed>.json`（`role="env"`，
每帧 `readback`；shadow 角色不能出像素，见 T03 成本试点）。

---

## 2 计数（第一组问题）

| seed | 决策 | 帧 | 写出槽 | 期望槽位 | 冷启动排除 | `teacher_slot0` | `executed_slot1` | HASH 校验(相等) | 逐帧 HASH | play / hud 逐字节 | 自检违反 | 墙钟 s | fps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 10 | 3,300 | 13,200 | 6,592 | 6,600 | 8 | 3,296 | 3,296 | 3,300 (3,300) | 13,200 | 3,296 / 3,296 全等 | 0 | 221.1 | 59.7 |
| 11 | 3,300 | 13,200 | 6,592 | 6,600 | 8 | 3,296 | 3,296 | 3,300 (3,300) | 13,200 | 3,296 / 3,296 全等 | 0 | 228.4 | 57.8 |
| 12 | 3,298 | 13,192 | 6,588 | 6,596 | 8 | 3,294 | 3,294 | 3,298 (3,298) | 13,192 | 3,294 / 3,294 全等 | 0 | 239.2 | 55.1 |
| **合计** | **9,898** | **39,592** | **19,772** | **19,796** | **24** | **9,886** | **9,886** | **9,898 (9,898)** | **39,592** | **9,886 / 9,886** | **0** | **688.7** | 57.6 均值 |

- **槽总数**：期望 19,796（`2 × 决策数`），写出 **19,772**，覆盖 **99.88%**；
  未写出的 24 个是 §1.3 的冷启动槽位，逐个列在每 shard `meta.json`
  的 `counts.cold_start_decision_frames`。
- **有自己 obs 的槽**：19,772（`play` 平面 19,772×4×112×96，`hud` 19,772×56×160，
  空行 0）。
- **有 HUD 的槽**：19,772（每槽 1 行 = 帧 `d-1`；旧语料是每块 1 行 = 帧 `4i-1`）。
- **有 label 的槽**：19,772 个都有动作目标，其中教师**评分过**的 9,886（槽 0）。
- **`label_source` 分布**：`teacher_slot0` 9,886 / `executed_slot1` 9,886。
  `label_verified` 分布：`teacher_slot0:true` 9,886、`executed_slot1:false` 9,886。
- **HASH 校验数**：9,898 个决策边界 `hash_post`（全等）+ 3 个 `RESET` hash（全等）；
  逐帧观察 39,592。首处不一致 `null`。
- **自检**：`violations_total=0`、`unexplained_diffs=0`、`passed=true`（3/3 shard）；
  决策帧步长直方图 `{2: 19,769}`（每 shard 行数 −1）；`history` 长度 8 / 全部帧号
  `< d` / 窗跨 14（`d-16..d-2`）/ `act_hist` 行 == 自身 `history` 的 one-hot：19,772/19,772。
- **独立复核（另一进程，只读）**：`self_check_corpus` 在写出进程之外重跑一遍，
  19,772 槽 **0 违反 / 0 未解释差异**；三个 `act_hist.npy`
  19,772/19,772 行全部通过 `obs_spec_v3.check_action_history`（one-hot 或全零行）；
  `play` `(N,4,112,96)`/`hud` `(N,56,160)`/`act_hist` `(N,8,18)` 全为 `uint8`。
  结果见 [`rebuild_2f.json`](rebuild_2f.json) 的 `independent_self_check`。
- 磁盘：3 个 shard 共 **1,048,346,825 B ≈ 1.05 GB**（`play` 283 MB + `hud` 59 MB +
  `act_hist` 0.9 MB + `slots.jsonl` 6.0 MB + `meta.json` 11 KB，每 shard）。

### 2.1 逐槽抽检（seed 10，可直接复算）

| 槽 | 决策帧 | 块 | `slot_in_block` | cand | `label_source` | HUD 帧 | HUD 校验和 | 有 scores |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| A | 400 | 100 | 0 | 11 | `teacher_slot0` | 399 | 286,606 | 是 |
| B | 402 | 100 | 2 | 16 | `executed_slot1` | **401** | 287,091 | **否** |

- `play[A] == 旧语料 play[100]` 且 `hud[A] == 旧语料 hud[100]`：**逐字节相同**；
- `hud[B] != hud[A]`（帧 401 vs 帧 399）——这正是旧语料无法表达的那个自由度；
- `play[B][0] == play[A][2]`：2 帧步长下相邻槽窗口重叠 2 帧，且窗口自身连续；
- `act_hist[B] == one_hot(history[B])`。

---

## 3 G420-0 的 L1 腿（时间/动作索引一致）

判据逐条（判据来自任务书与 README §5 WP420-01，证据来自 §2 与本文件 JSON）：

| # | 判据 | 结果 | 证据 |
| ---: | --- | --- | --- |
| 1 | 每个 2 帧槽都有自己连续 4 帧的 `play` | **成立** | 19,772 槽 × (4,112,96)；`obs_render_frames == [d-4..d-1]` 逐槽断言 0 违反；空行 0 |
| 2 | 每个槽都有自己的 HUD | **成立** | 19,772 行 HUD = 帧 `d-1`；槽 1 抽检帧 401 ≠ 槽 0 帧 399 |
| 3 | 每个槽都有自己的已执行动作 | **成立** | 槽 0 ← `step_frames[0]`，槽 1 ← `step_frames[2]`；3,300/3,300、3,298/3,298 与 `slots[1]` 一致；bomb 全 0 |
| 4 | 每个槽都有 8 条真实历史（不含当前/未来） | **成立** | 长度 8、帧号 `d-16..d-2`、全部 `< d`、16 帧跨度；`history` 由**已执行**的 `a0,a1` 交错构成，槽 1 的动作也进入后续槽的历史 |
| 5 | 逐槽 span/history 断言 0 违反 | **成立** | 自检 `violations_total=0`、`unexplained_diffs=0`（19,772 槽）；`EXPLAINED_CHECKS` 为空集——重建语料没有"可解释的旧格式失败"，任何违反都算未解释 |
| 6 | HASH 0 不一致 | **成立** | 9,898/9,898 `hash_post`、3/3 `RESET`、39,592 逐帧 HASH 观察；`hash_first_mismatch=null` |

**附加证据（不在判据内，但把"索引没平移"钉死）**：
槽 0 的观测与已存旧语料**逐字节相同**——`play` 9,886/9,886、`hud` 9,886/9,886。
旧语料是独立采集、独立渲染的另一条通道，这些字节相同说明重建抓的确实是
帧 `4i-4..4i-1` 与帧 `4i-1`，不是邻居、也不是插值。

**判定：L1 腿在重建语料上成立。** 但**不推出**：

- G420-0 整门通过（还缺部署输入边界测试、冻结模型在线/离线前向一致性等条目）；
- 语料可以直接投 2 帧全监督训练：verified 标签仍只有 50%（§4）；
- 任意其它 seed 都成立：本切片只覆盖 train split 的 seed 10/11/12。

---

## 4 边界（必须与上面的"成立"一起引用）

### 4.1 槽 1 的标签：行为目标，不是教师评分

- 槽 1 的 `cand` 是**教师实际执行**的动作（`step_frames[2]` 解码，与 `slots[1]`
  3,300/3,300 一致），但教师**没有**为它产生 `scores[18]`/`verified_mask`/`margin`。
- 因此：`label_source="executed_slot1"`、`label_verified=false`、**不含任何 scores 键**
  （3 个 shard 实测 0 条）。它可作 **BC 的行为目标**（"教师在此处这样动"），
  **不得**计入 G420-1 的 verified 标签、不得进入安全集合、不得当"教师认可"。
- **verified 覆盖率仍是 50%**：9,886/19,796 = **49.94%**（每 shard
  3,296/6,600 = 49.94%、3,294/6,596 = 49.94%）。重放掩码不可能凭空造出第二条评分向量；
  要真正的 2 帧标签，只能按 README §5 在缺失槽**重新查询教师**（后续工作包）。

### 4.2 冻结契约模块的命名缺口（需要 contracts 所有者处理）

`touhou_rl/v420/contracts.py` 的 `LABEL_SOURCES = ("teacher_slot0", "teacher_slot1",
"none")` **不包含** `executed_slot1`，且 `build_slot_record()` 要求"非 `none` 的
`label_source` 必须 `label_verified=true`"。本任务的契约要求是
`executed_slot1` + `label_verified=false`，两者直接冲突。本任务的禁止清单不允许改
`contracts.py`（另一代理正在改），所以 `rebuild.py` **不调用**
`build_slot_record()`，而是用同一组 `check_*`（span / render frames / history /
key timing / 动作表往返）逐字段校验，并把这条差异写进每 shard `meta.json` 的
`contract.executed_slot1_known=false` 与 `contract.note`。
建议 contracts 所有者：把 `executed_slot1` 加入 `LABEL_SOURCES`，并把
`build_slot_record` 的"必须 verified"规则改成"`teacher_*` 必须 verified、
`executed_slot1` **必须** unverified"。

### 4.3 冷启动 8 槽被排除

见 §1.3。代价：每 shard 少 8 个槽位（0.12%），换来 19,772 个槽的断言全清。
它们的位置（决策帧 `0,2,…,14`）逐 shard 记录在 `meta.json`；
不做白名单例外、不写 `pad`、不复制第一帧。

### 4.4 没有墙钟时间戳

旧语料只有引擎逻辑帧号（`shm` 头部只有 `frame` 一个时间量），重建语料同样只记
逻辑帧号。本文件任何"时间一致"都指**帧号对齐**，不含真实时间戳。

### 4.5 seed 12 是 `stage_changed` 收尾，且旧元数据多记 4 帧

- seed 12：3,298 决策 / 13,192 帧，`stop_reason=stage_changed`、`stage_clear=true`；
  3,298 个 `hash_post` 全部相等，说明换面就发生在记录的那一帧。
- 其 `shard.json` 的 `frames_committed=13,196` 比 `3,298×4=13,192` **多 4**，
  而 `last_decision_frame=13,188` 与 3,298 行自洽。这是旧元数据在
  `stage_changed` 收尾时的一处不一致（多计了一帧未标注的决策），
  不影响重建（重放自己的逐帧记账是权威值 13,192）。
- 同源现象：seed 13/17/20 的旧 shard 只有 3,274 行（同样 `stage_changed`）。
  所以"每个 shard 3,300 决策"只对 seed 9/10/11/14/15/16/19 成立；
  做 T04 覆盖率分母时要用**实际行数**。

---

## 5 与旧语料的差异表

| 项 | 旧语料（v4.1.0 4 帧 shard） | 重建语料（v4.2.0 2 帧 corpus） |
| --- | --- | --- |
| 决策节拍 | 每行提交 4 帧（2 个槽叠在一行） | 每 2 帧一槽，决策帧 `4i` 与 `4i+2` 各自成槽 |
| 槽数登记 | 6,600 个槽位，只登记 3,300 行 | 期望 19,796 / 写出 19,772（+24 冷启动排除） |
| 观测覆盖 | 3,300/6,600 = **50%**（每块 1 行 `play` = 帧 `4i-4..4i-1`） | 19,772/19,796 = **99.88%**（槽 1 有自己的 4 帧栈） |
| HUD 覆盖 | 3,300 行 = 每块 1 行（帧 `4i-1`）；**槽 1 缺 HUD** | 19,772 行 = 每槽 1 行（帧 `d-1`）；槽 1 用帧 `4i+1` |
| `history` 跨度 | 8 × 4 帧 = **32 帧**，只含槽 0 的 `first_action` | 8 × 2 帧 = **16 帧**，`a0,a1` 交错，含槽 1 的已执行动作 |
| `history` 含槽 1 动作 | 0 条 | 每槽 8 条中约 4 条来自槽 1 |
| 标签 | 3,300 个教师 18 路 `scores`（只覆盖槽 0） | 9,886 `teacher_slot0`（含 `teacher_scores`）+ 9,886 `executed_slot1`（无 scores） |
| verified 覆盖 | 3,300/6,600 = 50% | 9,886/19,796 = **49.94%（未改变）** |
| HASH 校验粒度 | 每决策 1 个边界（每 4 帧），6,600 对 | 每决策 1 个边界 + `RESET`，9,898；逐帧观察 39,592（每 2 帧边界都由 `d` 的栈覆盖） |
| 真实时间戳 | 无（只有逻辑帧号） | 无（只有逻辑帧号；同样不声称有时间戳） |
| 槽 1 可用性 | 有动作、无观测、无 HUD、无标签 → 不能当样本 | 有观测、有 HUD、有动作、有 8 条历史 → 可作 BC 样本（但不是 verified 标签） |

---

## 6 复算与退出码

| # | 命令 | 退出码 | 覆盖 |
| ---: | --- | ---: | --- |
| 1 | `wsl -e bash .tmp_v420_t02c/run_pytest.sh` | 0（13 passed） | WSL 侧单测 |
| 2 | `python -m compileall -q touhou_rl\v420`（Windows） | 0 | 语法 |
| 3 | `python -m pytest -p no:cacheprovider --basetemp .tmp_v420_pytest_win2 -q tests/test_v420_rebuild.py`（Windows） | 0（13 passed） | Windows 侧单测 |
| 4 | `wsl -e bash .tmp_v420_t02c/smoke.sh` | 0 | 8 决策冒烟（32 帧）先验正确性 |
| 5 | `wsl -e bash .tmp_v420_t02c/run_seed.sh 10` | 0 | seed 10 全量 |
| 6 | `wsl -e bash .tmp_v420_t02c/run_seed.sh 11` | 0 | seed 11 全量 |
| 7 | `wsl -e bash .tmp_v420_t02c/run_seed.sh 12` | 0 | seed 12 全量 |
| 8 | `wsl -e bash .tmp_v420_t02c/run_collect.sh` | 0 | 汇总 `rebuild_2f.json` |
| 9 | `wsl -e bash .tmp_v420_t02c/run_verify.sh` | 0 | 新进程独立复核（19,772 槽 0 违反）并写回 `independent_self_check` |

只读性核对（`rebuild_2f.json → assets_readonly`）：

| seed | 源 `env/score.dat` sha256 前 = 后 | 旧 `labels.jsonl` sha256 |
| ---: | --- | --- |
| 10 | `d7089834…81ae786` = 同值 | `011f6c1c…cff622a0`（未变） |
| 11 | `832f1813…f3d4b1b6` = 同值 | `bc663759…d7db149a`（未变） |
| 12 | `165fbd25…5317036` = 同值 | `f79f8ab6…b0033d028`（未变） |

`/root/v400-data/**`、`.tmp_v410_*`、`.tmp_v400_*` 全程未写；未读 confirm 种子
61000–61499，未触碰 60000–60199；未对 PID 1884178（v4.1.0 采集）发送任何信号；
未执行任何 git 写操作。

---

## 7 原始证据与产物

| 路径 | 内容 |
| --- | --- |
| `.tmp_v420_data/T02c/rebuild/<shard-key>/2f/` | 语料：`slots.jsonl`、`obs/{play96x112,hud,act_hist}.npy`、`meta.json` |
| `.tmp_v420_data/T02c/logs/rebuild_seed{10,11,12}.json` | 每 seed 的完整报告（含 `self_check` 全量、`data_dir` 清单、模块 sha256） |
| `.tmp_v420_data/T02c/logs/rebuild_seed{10,11,12}.log` | 运行 stdout/stderr |
| `.tmp_v420_data/T02c/replay_data/<shard-key>/` | 引擎数据目录副本（`score.dat`/`th08.cfg` 复制、两个 `.dat` 符号链接） |
| `touhou_rl/v420/rebuild.py` | 实现（新文件） |
| `tests/test_v420_rebuild.py` | 13 个纯逻辑单测（注入假 env） |
| [`rebuild_2f.json`](rebuild_2f.json) | 本文件全部计数的机器可读版本（含逐 shard `corpus_scan`、`legacy_baseline`、`plan_agreement`、新进程复核 `independent_self_check`） |

复算：

```bash
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v420_t02c/run_collect.sh   # 重算 rebuild_2f.json
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v420_t02c/run_verify.sh    # 新进程独立复核
wsl -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v420_t02c/run_seed.sh 10   # 重跑 seed 10
```

`meta.json` 里还登记了：契约版本（`v420-contract/1` / `obs-v420-2f` / `action-v420-2f`）、
每槽计数、HASH 统计、引擎 sha256、`data_dir` 副本清单、9 个源码模块的 sha256、
`git_head`、主机与 Python 版本、`plan_agreement`
（以 seed 10 为例：`slot0_matches_first_action` 3,300/3,300、
`slot1_matches_planned_slot1` 3,300/3,300、`slot1_distinct_from_slot0` 1,740/3,300、
`tail_mask_clears_shoot` 137；seed 11 = 1,734、seed 12 = 1,737）。

---

## 8 下一步（不在本文件范围）

1. 用本语料跑 T04 的 2 帧 BC 基线：**只把 `teacher_slot0` 当监督目标**，
   `executed_slot1` 只能作为可选的行为克隆目标单独登记，不进入 verified 统计。
2. 若要提高 verified 覆盖率，必须按 README §5 在槽 1 **重新查询教师**
   （新采集任务），不能用本语料"凑"。该工作已作为 **T02d** 在另一任务里开工
   （`touhou_rl/v420/reslot.py`，输出 `$V420_DATA/T02d/reslot/`，与本文件的
   `T02c` 目录无重叠），本文件的 `executed_slot1` 与 T02d 的 `scores` 是两种不同的
   东西，**不可逐值比较**。
3. G420-0 剩余腿：部署输入边界测试 + 冻结模型在线/离线前向一致性（T02 的在线审计）。
