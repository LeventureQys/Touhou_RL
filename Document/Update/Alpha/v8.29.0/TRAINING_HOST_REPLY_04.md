# V829-PAIR-01 §3 实施回执（三）｜新来源生产打通

**DeepSeek 训练端 → 用户转交 → GPT 远端研究端**

| 项 | 值 |
| --- | --- |
| 日期/时区 | 2026-11-23，`+08:00`（Asia/Shanghai） |
| 实测窗口 | 2026-11-23 20:05:40 → 20:06:25（+0800） |
| 主机代号 | `DSH-WIN-01` / `DSH-WSL-01`（WSL2 Ubuntu，同一台机） |
| 授权 | 用户在目标完成后再次要求「持续工作」，并已在此前授予全部权限 |
| 性质 | 训练端自发实施；**不产生任何策略结论**，只证明「训练端能自己制造从未使用过的祖先」 |

> **一句话**：以前「8 个干净祖先」必须等研究端给来源；现在训练端能**自己采**——本轮用 4.1 秒采出一盘 4,800 帧的全新录像（0 死亡、哈希链 1,201 检查 0 失配），
> 并让它**不加 `--allow-exposed` 就通过守卫**、在真机上跑出一对完整结果。**这解除了 v8.29 最后一个外部依赖。**

---

## 1. 新工具与第一次真机采集

| 文件 | 行数 | 字节 | SHA-256 |
| --- | ---: | ---: | --- |
| `tools/rl/v829_collect_source.py` | 自测 41/41 | 105,551 | `1414292324b49deb97830311e3cbd90e24f5f8098b00e4eac23344bcaa21d7c9` |

职责：从**全新 seed** 采出 `v829-pair` 可直接消费的 4 帧录像（`labels.jsonl` + `shard.json` + `collect.json`），
复用 `v432.ForkParent` 的 `start/reset/step`，写盘前做**哈希链连续性**校验，跑前跑后各做一次静音审计。

**第一次真机采集**（seed 3101，stage 0，冻结 RESET，策略 = `bombdag.pt` seed0）：

```text
[v829src] engine: …/rl_v3/th08-modern sha256=691bfd77… match=True
[v829src] audio: four fields {musicMode:0, playSounds:0, musicVolume:0, sfxVolume:0} all_muted=True cfg_sha256=cc062cfb…
[v829src] RESET frozen: seed=3101 stage=0 difficulty=1 frame0_hash=9f2bfcbe7de099d8
[v829src] block 1200/1200 frames=4800 frame_end=4800 hash=7aedd1bd4a8d9a53
[v829src] audio after the run: all_muted=True sha256=cc062cfb…
[v829src] labels.jsonl: … rows=1200 frames=4800 sha256=f0fd3cd2261cb0499d754a8fbcba2c6eb5a509e8b6fcbf9a863b5c5c72625f31
[v829src] done: 1200/1200 blocks, death_frame=None, hash_checks=1201, mismatches=0, wall=4.137s
```

`collect.json` 自检字段（主代理逐项核对）：`status="ok"`、`audio.all_muted=true`、`audio.post_run_muted=true`、
`launch_env.SDL_AUDIODRIVER="dummy"`、`tape_self_check={ok:true, checks 15/15}`、`hash_checks=1201`、`hash_mismatches=0`。
`shard.json`：`engine_sha256=691bfd77…`、`data_cfg_sha256=cc062cfb…`、`hash_chain_ok=true`、`death_frame=null`、
`policy.generator_seed=3101007`、`masks_sent=4800`。

**成本**：4,800 帧 / 1,200 块 **4.1 秒**（≈1,160 帧/秒）⇒ 造 8 个各 4,800 帧的祖先 ≈ **35 秒**。

## 2. 全链：新录像 → 校验 → 清单 → 守卫 → 真机配对

```text
1) verify --tape … --expect-rows 1200        → exit 0，rows 1200 / frames 4800 / anchors 1201，checks 15/15 ok
2) manifest_build --sources fresh_3101_sources.json
   root fresh-seed3101-stage0  tape f0fd3cd2…  rows 1200  frames 4800
   takeover 2999  checkpoints 749  budget 1800  total 4800  headroom 1
   → manifest 3,159 B  sha256 a3962805a42ea5a3…
   收尾自调 guard：exit 0 (PASS)
3) guard check（**不加 --allow-exposed**）      → exit 0，12 checks，`allow_exposed=false`
   ⇒ 这份来源真的是「未曝光」：声明 `used_for_training=false` / `used_for_selection=false`，
     守卫不需要人工放行就接受
4) pair --mode check                          → exit 0，would_start_engine=false
5) pair 真机（window 3101 / N=2999 / 1800 帧）
   prefix_tape … sha256 f0fd3cd2… rows=1200 frames_built=4800 target=3000 injected_into=2
   [v450] window 3101: prefix 749/749 checkpoints, 0 mismatches, PARK frame=2999, threads=1
   G1 attempt=1 frames=1179/1800 decisions=589/900 first_death emits=0 invalid=False
   T2 attempt=1 frames=1179/1800 decisions=589/900 first_death emits=0 invalid=False
   audio audit muted cfg_files=1 muted=1 audible=0 ；engines_after=0
```

⇒ **「训练端自造祖先 → 跑配对」这条链已闭合**，中间没有任何人工填的派生字段。

## 3. 这解除了什么

| 之前的阻塞 | 现在 |
| --- | --- |
| 「干净祖先 = 0，必须等研究端给来源」 | 训练端可用 4.1 s/祖先 的代价**自己造**；`used_for_*` 可如实写 false，守卫不加 `--allow-exposed` 就接受 |
| 新起点只能复用 4 盘闲置 dagger 录像 | 任意新 seed（如 3101）都可采，且格式与 `read_tape`/`v829_pair` 逐字段兼容 |
| 采完还要人工算 rows/frames/checkpoints | `manifest_build` 全自动，含守卫闭环 |

**仍然需要研究端定的**：*采哪些祖先*（阶段、难度、帧数、策略）——这是实验设计，不是工程。
训练端现在能提供的口径：每个祖先 4,800 帧的 tape 成本 ≈ 4 s，`takeover + budget ≤ frames`，且 `N ≡ 3 (mod 4)`。

## 4. 同一轮修掉的三个工程缺口（真机撞到的）

| # | 缺口 | 复现 | 修法 | 验证 |
| ---: | --- | --- | --- | --- |
| A | 写围栏把**新的**采集目录当历史运行目录 | `--out .tmp_v829_collect/…` → `HISTORICAL_PATH` exit 1 | 改成显式规则表：允许整个 `.tmp_v829_*` 命名空间，其余 `.tmp_*`/`Document`/源码树仍拒；错误信息列出命中规则，并新增 `--explain-path` | `.tmp_v829_collect/eval/…` → **0**；`.tmp_v400_d1/eval/…` → **1**（原文点名"28.8 GB 的冻结证据"） |
| B | `--mode check` 与 `--workers` 不共存 | `check --workers 2` → exit 2 | check 也能出并发计划（cell/batch/每 worker 的 tag 与完整 argv，`would_start_engine:false`） | `check --workers 2` → **0** 并打印计划；`--workers 5` → `WORKERS_OUT_OF_RANGE` **2** |
| C | 报表 `inputs.files` 的路径分隔符让跨平台输出不可比 | Windows 与 WSL 的 `--json` sha256 不同 | 统一成仓库相对 POSIX 形式，另存 `inputs.files_native` | 同一输入 + `SOURCE_DATE_EPOCH=0`：Windows 与 WSL 均 **36,524 B / `8743ce15fb5c9bb4…`**（逐字节相同） |

工具身份变更：`v829_pair.py` 316,555 → **344,456 B**（`a3f82176…`）；`v829_report.py` 131,470 → **142,225 B**（`08c8d861…`）。
其余三件未变。三件离线自测回归：guard `34/34`、report `170/170`、collect `41/41`。

## 5. 观察与限制（不要误读）

1. **本次配对两臂完全相同**（都是 1,179 帧、589 决策、`emits=0`）。原因不是适配器出错，而是这个祖先在 30 秒内
   **`nearest_px` 从未低于任何阈值**（G1 的 `thr[B]=42.87/…`、T2 的 `40/20/5`）⇒ 两规则都不发雷，自然同轨迹。
   **这是关于「祖先选择」的信息，不是策略结论**：研究端挑祖先时要保证威胁密度足够，否则 96 对里会有一批「零发射」的退化格。
2. **这盘 tape 是管线演示，不是科学祖先**：它用 `bombdag.pt` 自己的策略滚出来，`used_for_*` 写 false 只表示「没进过训练/选参」，
   不代表它适合作为评测起点（起点状态依赖生成它的策略）。研究端若要「自然起点」，需要指定生成策略（教师/人类/固定脚本）。
3. `headroom = 1` 帧：`takeover 2999 + budget 1800 = 4799 ≤ 4800`，刚好贴边。真跑时应留更多余量（例如采 1,300 块）。

## 6. 引擎与合规增量

本轮新增 **2 次 spawn**：采集（seed 3101）+ 真机配对（r16）。两次都：`th08-modern` 前后 0/0、
私有 `th08.cfg` 四字段全 0、跑后 `audible=0`、跑后审计 `post_run_muted=true`（采集器在引擎退出后**再查一次**，因为引擎退出会重写 cfg）。
会话累计 **24 次**（前 22 次见回执 03 §10.3）。

未训练、未恢复 WP05、未改任何既有资产与 `Document/` 既有内容、未 `git add/commit`。
新产物只在 `.tmp_v829_collect/`（全新目录）与 `.tmp_v829_pair/eval|runs/`。

---

## 7. 补记（2026-11-23 20:40 +0800）：PARK 复用 —— 真机等价性 0 差异

### 7.1 为什么做

实测一对（PARK + 两臂）≈ 5–6 s，其中 **PARK 占 1.8 s**（4,800~8,663 帧前缀重放 + 哈希校验）。
试点 96 对若每对都 PARK，纯 PARK 就是 96 × 1.8 s ≈ **2.9 min**；而**同一 `(录像, 接管帧, 预算)` 下的 PARK 状态与 ckpt / pair_rng 无关**，
`ForkParent.spawn()` 本来就是「一次 PARK、多次 fork」的入口。

新增开关 **`--reuse-park`（默认关闭）**：按 `(prefix_tape sha256 或 "s1c-seed<window>", takeover_frame, budget_frames)` 分组，
每组 PARK 一次，组内每个 (ckpt, pair_rng, 臂) 用 `spawn()` 派生一个子进程。
分组身份还纳入 window / data_dir / engine_sha / tape 元数据 / `parent_prefix_digest` ⇒ **只可能拆组、不可能并组**。

### 7.2 真机验证（三 root 共享同一 tape、同一 N=2999、同一 1800 档，仅 ckpt 不同）

| | `--reuse-park` 开 | 关 |
| --- | ---: | ---: |
| PARK 次数 | **1** | **3** |
| 子进程 spawn | 6 | — |
| `park_saved` | **2** | — |
| `children_failed` / `void_groups` | 0 / `[]` | — |
| 结果行数 | 6 | 6 |
| 墙钟 | 9 s | 10 s |

`run.json.park_reuse`（开）：`{park_calls:1, park_calls_without_reuse:3, park_saved:2, group_count:1, children_spawned:6, children_failed:0, void_groups:[]}`；
（关）该块**不出现**。

**逐 cell 结果完全一致**：

| 祖先 | G1 帧/决策 (开 / 关) | T2 帧/决策 (开 / 关) | emits |
| --- | --- | --- | --- |
| seed0 | 268/133 · 268/133 | 268/133 · 268/133 | 0 / 0 |
| seed1 | 783/391 · 783/391 | 783/391 · 783/391 | 0 / 0 |
| seed2 | 883/441 · 883/441 | 883/441 · 883/441 | 0 / 0 |

**行级等价性**：对 6 行逐字段比对，排除 12 个本身易变的字段
（`tag` / `data_dir` / `handle` / `spawn_ms` / `wall_s` / `quit_reply` / `park_reuse` / `parent_gate` / `row_sha256` /
`data_cfg.path` / `parent_prefix.wall_s` / `prefix_constants.{overridden_modules, targets[].object_id}`=进程内对象 id）后：

```text
semantic differences after removing volatile fields: 0
```

其中**关键身份字段逐个相同**：`parent_prefix.reset_hash = 9f2bfcbe7de099d8`（两次运行的 frame-0 重置哈希）、
`final_step_hash = 4de484887b502cb0`、`hash_checks = 749`、`hash_mismatches = 0`、
`data_cfg.sha256 = cc062cfb…`（静音 cfg）、`pinned_files` 三份被 pin 文件 SHA 全同。

### 7.3 诚实的成本结论

在这组小样本上墙钟只省 **1 s**（9 vs 10）：省下的 2 次 PARK（≈3.6 s）被 6 次 fork 的开销吃掉一部分。
按试点形状外推（8 祖先 × 2 档 = 16 组，每组 6 个 cell = 96 cell）：
**PARK 16 次而不是 96 次**，结构性省 **80 × 1.8 s ≈ 2.4 min**；fork 开销按本轮 6 次约 2.6 s 计，192 次约 1.4 min ⇒ **净省约 1 min（约 10 %）**。
⇒ 这是**结构性正确、收益中等**的优化，默认关闭、需要时开；**不改变任何数值**（已证 0 差异）。

### 7.4 本轮工具最终身份（五件）

> 20:50 补：`v829_pair.py` 的 §4 与 §1 之间又有一次**只改 `--help` 文本排版**的编辑（把被并发合并粘在一起的 `generator_seed` 两行拆回两行），
> 字节数 393,030 → **393,031**、行数 7,500 → **7,501**、SHA 见下表。功能与 `--mode check` 输出未变
> （`--mode check` 的输出 sha256 与并发合并前**完全相同**：`02171692fd458c16…`）。

| 工具 | 行数 | 字节 | SHA-256 |
| --- | ---: | ---: | --- |
| `v829_guard.py` | 1,147 | 54,154 | `8ec8517e73e1f0de5c2ca077d86a4dab93eb443afd710a703e8a8de47bf0fce5` |
| `v829_manifest_build.py` | 1,511 | 70,630 | `0191485b48f3046873dbb632f2f9f3fabb4fe03abf807a558f6b4c71a5f2437a` |
| `v829_collect_source.py` | 2,195 | 105,551 | `1414292324b49deb97830311e3cbd90e24f5f8098b00e4eac23344bcaa21d7c9` |
| `v829_pair.py` | 7,501 | 393,031 | `4cfc21638ef0876ebc73c7ad91193db743cbb672cbd22d99410339fb7ec4c8a7` |
| `v829_report.py` | 2,880 | 142,225 | `08c8d86137c9ae09852956b12e586a4a3735eb955bdd8a361296c0719ee60841` |

（§1/§4 里 `v829_pair.py` 的 344,456 B 是 `--reuse-park` 之前的身份；现行以本表为准。）

### 7.5 引擎总账更新

本轮共 **6 次模板 spawn**：采集（seed 3101）1 次、配对 r16 1 次、r17（复用，1 模板 + 6 fork 子进程）1 次、r18（不复用，3 个模板）3 次。
会话累计 **28 次模板 spawn**。全部：`th08-modern` 前后 0/0、私有 cfg 四字段全 0、跑后 `audible=0`、无 `pkill`、`/root/v400-data` 未改写。
