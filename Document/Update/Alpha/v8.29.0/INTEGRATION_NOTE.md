# v8.29 配对评测 · 训练端集成说明（给研究端写评测工单用）

**状态：训练端提案，未登记、未提交。** 研究端确认后建议把 `v829-pair-manifest/1` 写进预注册并登记 SHA。

本文件描述三件工具、清单 schema、可跑的起点类型、退出码与**已经真机验证过什么、没验证过什么**。
配套证据：[回执 02](TRAINING_HOST_REPLY_02.md)（§3.1–§3.3 冒烟）、[回执 03](TRAINING_HOST_REPLY_03.md)（阻塞解除与验收层）。

---

## 1. 五件工具与职责边界

| 工具 | 职责 | 会不会启动引擎 |
| --- | --- | --- |
| `tools/rl/v829_collect_source.py` | **从全新 seed 采出 4 帧录像**（`labels.jsonl` + `shard.json` + `collect.json`）；写盘前做哈希链校验，跑前跑后各一次静音审计 | **会**（`collect`）；`verify`/`selftest` 不会 |
| `tools/rl/v829_manifest_build.py` | **来源清单 → 通过守卫的清单**（顶层 `ckpts:[]` 与 `rngs` 可展开预注册的两根重复测量轴；幂等） | **不会** |
| `tools/rl/v829_guard.py` | 清单 `fail-closed` 校验（结构、预算、曝光、摘要、重复、RNG 复用、祖先自洽） | **不会**（纯离线，`selftest 52/52`） |
| `tools/rl/v829_pair.py` | 配对入口：`check / trace / pair / compare`，跑真实引擎；`--reuse-park` 可让同 `(录像, 接管帧, 预算)` 的多 cell 共用一次 PARK | `check`/`compare` 不会；`trace`/`pair` 会 |
| `tools/rl/v829_report.py` | §4 计分与报表（2×2、Δ30/Δ40、去一法、界限、判定） | **不会**（纯离线，`selftest 170/170`） |

**运行时解释器**：`/root/v400-venv/bin/python`（**Python 3.10.12**）。
⚠️ Windows 侧是 3.12；**任何编译/自测验收都必须用 3.10**，否则会漏掉只在 3.12 合法的语法。

---

## 2. 清单 schema `v829-pair-manifest/1`

### 2.1 必需字段（缺一个就是 `FIELD_MISSING`）

```json
{
  "schema": "v829-pair-manifest/1",
  "manifest_id": "v829-<用途>-<序号>",
  "purpose": "一句话说明这份清单要回答什么",
  "engine": "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern",
  "data_source": "/root/v400-data/game",
  "roots": [
    {
      "ancestor_id": "anon-01",
      "stage": "th08-stage1-normal-team0",
      "window": 11,
      "budget_frames": 2400,
      "ckpt_sha256": "9c6d5d4a…（39,069 B 的三份 bombdag.pt 之一）",
      "ckpt_path": ".tmp_v860_bombdag/ckpt/seed0/bombdag.pt",
      "engine_sha256": "691bfd77…",
      "pair_rng": 20261124,
      "takeover_frame": 8663,
      "total_frames": 13200,
      "total_frames_source": "这个数从哪来（必须写，不能凭空）",
      "parent_prefix_digest": "<录像 labels.jsonl 的 sha256>",
      "verify_anchor": {
        "prefix_digest": "<同上>",
        "prefix_record": ".tmp_v410_s1c/d1/shards/…/labels.jsonl",
        "recorded_frames": 8663,
        "hash_checkpoints": 2165
      },
      "used_for_training": false,
      "used_for_selection": false
    }
  ]
}
```

### 2.2 可选字段

| 字段 | 作用 |
| --- | --- |
| `prefix_tape` / `prefix_tape_sha256` / `prefix_tape_rows` / `prefix_tape_frames` | 换录像源（不声明则用 S1c 的 `seed<window>` 录像） |
| `lineage_ancestor_ids` | 与 `configs/v430/splits.json` 谱系交叉核对（`--lineage-config` 启用时必需） |
| `lineage_new` | 新祖先（不在冻结谱系里）：允许空 id，记 `LINEAGE_NEW` |
| `takeover_frame` 之外还想换档位 | ⚠️ **一份清单只登记一个档位**；换档位＝新 root 或新清单 |
| `ckpt_path` | 只在清单里给出路径（guard 忽略），让 `--mode check` 能打印完整 argv |

### 2.3 三条容易踩的硬约束

1. **接管帧必须是 `4k − 1`**：录像 4 帧一个锚点，`target = N+1` 必须落在锚点上（`v431_dense_driver.py:81-89`）。
   `8663 = 4×2166−1`；`4000` 不合法，`3999` 合法。期望检查点数 = `N // 4`。
2. **`takeover_frame + budget_frames ≤ total_frames`**，且两个档位（1800 / 2400）都会查。
3. **一份清单里 `budget_frames` 与 `ckpt_sha256` 是逐 root 钉死的**：`--budget-frames` 给不同值会被
   `BUDGET_ROOT_MISMATCH` 拒；`--pair-rng` 同理（`PAIR_RNG_CONFLICT`）。**清单就是运行表。**


---

## 2.4 来源清单 → 清单（最后一段路，已端到端验过）

研究端只需填 `v829-sources/1` 小 JSON（每源：`ancestor_id` / `stage` / `tape` / `takeover_frame` /
`pair_rng` / `used_for_training` / `used_for_selection`，可选 `window` / `lineage_new`），
训练端用 `v829_manifest_build.py` 生成清单，所有派生字段自动算：

| 输出字段 | 从哪来 |
| --- | --- |
| `parent_prefix_digest` / `verify_anchor.*` / `prefix_tape_sha256` | `sha256(tape)` 原始字节 |
| `total_frames` / `prefix_tape_frames` | `rows × 4`（工具数行，不许手填） |
| `verify_anchor.recorded_frames` / `hash_checkpoints` | `takeover` / `takeover // 4` |
| `window` | 源 `window` → `--window` → ancestor 哈希派生（会标 `window_derived`） |
| `ckpt_sha256` / `engine_sha256` | **实测**；声明值只做核验 |

**端到端实测**（`.tmp_v829_info/milestone_e2e.sh`，无引擎）：

```text
root milestone-old-w11    window 11   tape bc663759… rows 3300 frames 13200 takeover 8663 cp 2165 headroom 2737
root milestone-dagger-3006 window 3006 tape 55fd2a62… rows 2088 frames  8352 takeover 2999 cp  749 headroom 3553
wrote manifest.json 5290 B sha256 ee65f7f0…
guard: exit 0 (PASS, 18 checks)   ← 生成器自己调了一次，主代理又独立重跑一次，均 0
IDEMPOTENT: yes                   ← 连跑两次 sha256 相同（不写时间戳）
pair --mode check → rc=0, would_start_engine: false
```

**最容易填错的六件事**（生成器会挡，但提前知道更省事）：
1. `takeover_frame` 必须是 `4k−1`（末位 3/7；3000/4000/8664 全非法，报错会给最近两个合法值）；
2. 别手填 `rows`/`frames`/`total_frames`，只给 `tape` 路径；
3. `used_for_*` 是**正面声明**：不写/写 `"false"`/`0` 会被拒；写 `true` 必须加 `--allow-exposed`；
4. 同一祖先两个窗口不要各起一个 id（→ `WINDOW_DUPLICATE`）；换窗口/换档位 = 再交一份来源清单；
5. `pair_rng` 一个源一个（复制行最容易撞 → `RNG_DUPLICATE`）；
6. `budget_frames` 只有 1800/2400，且 `takeover + budget ≤ 录像帧数`（8,352 帧的 dagger 在 1800 档最大合法 N=6547）。


## 2.5 自造来源（当研究端要「全新祖先」时）

```bash
# 采一盘新录像（4 帧锚点、带哈希链），4,800 帧约 4 秒
python -B tools/rl/v829_collect_source.py collect \
  --out-dir .tmp_v829_collect/seed3101-frozen \
  --seed 3101 --stage 0 --difficulty 1 --blocks 1200 \
  --policy policy:.tmp_v860_bombdag/ckpt/seed0/bombdag.pt \
  --engine /root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern \
  --data-source /root/v400-data/game \
  --work-dir .tmp_v829_collect/work --tag v829-collect-3101

# 不启引擎地校验录像格式与哈希链
python -B tools/rl/v829_collect_source.py verify \
  --tape .tmp_v829_collect/seed3101-frozen/labels.jsonl --expect-rows 1200
```

产出三件：`labels.jsonl`（tape）、`shard.json`（seed/stage/difficulty/policy/引擎与 cfg SHA/哈希链）、
`collect.json`（argv、环境、静音审计、自检）。

写入围栏：允许整个 **`.tmp_v829_*`** 命名空间（`.tmp_v829_pair` / `.tmp_v829_info` / `.tmp_v829_collect`）；
其余 `.tmp_*`、`Document/`、源码树一律拒（`python -B tools/rl/v829_pair.py --explain-path <PATH>` 可离线查任意路径的判定）。

| 参数 | 语义 |
| --- | --- |
| `--seed` | 游戏 seed（`RESET` 的 seed 字段）；**不要用已登记段**（1–25 / 1000–1019 / 2000–2010 / 2020–2054 / 3000–3004 / 3010–3029 / 3042–3091 / 60000–60199 / 61000–61499） |
| `--stage` / `--difficulty` | 默认 `0 / 1`（= 冻结 `RESET 0 1 0 <seed> 0`，即 Normal）；显式传别的值会走 extended RESET，`collect.json.reset.mode="extended"` 会标出来 |
| `--blocks` | 每块 4 帧；`blocks 1200` → 4,800 帧 |
| `--policy` | `policy:<ckpt>` / `random` / `stay` / `const:N` |
| `--stop-on-death` | 死亡即停（默认）；否则会把已死的帧也记进去 |

**可用作祖先的硬约束**：`takeover + budget ≤ frames`，且 `takeover ≡ 3 (mod 4)`。
例：4,800 帧 + 1800 档 ⇒ 最大 `takeover = 2999`（`target 3000`、`749` 个检查点）。

⚠️ **新采录像只是「未曝光」，不等于「适合作为起点」**：它的状态依赖生成它的策略。
研究端若要「自然起点」，需指定生成策略（教师 / 人类录像 / 固定脚本）。


## 2.6 预注册的两根重复测量轴（`ckpts` / `rngs`）

预注册 §2 的形状是 **每祖先 × 3 checkpoint × 2 配对 RNG**。清单里这样表达：

```json
{
  "schema": "v829-sources/1",
  "budget_frames": 1800,
  "ckpts": [
    {"path": ".tmp_v860_bombdag/ckpt/seed0/bombdag.pt", "sha256": "9c6d…"},
    {"path": ".tmp_v860_bombdag/ckpt/seed1/bombdag.pt", "sha256": "36a5…"},
    {"path": ".tmp_v860_bombdag/ckpt/seed2/bombdag.pt", "sha256": "885d…"}
  ],
  "rngs": 2,
  "sources": [ { "ancestor_id": "anon-01", "tape": "…/seed3201/labels.jsonl", "takeover_frame": 2999, … } ]
}
```

⇒ 生成 `8 sources × 3 ckpt × 2 rng = 48 root`，**同一祖先的 6 个 root 共享同一 `ancestor_id`**，
用 `ckpt_sha256` 与 `pair_index` 区分。`pair_rng` 由 builder 按序号分配（`ckpts`/`rngs` 模式下**不许**再写行级 `pair_rng`）。

| 轴 | 字段 | 守卫里的作用 |
| --- | --- | --- |
| 祖先 | `ancestor_id` | 报表按它数祖先（`ANCESTOR_COUNT`） |
| checkpoint | `ckpt_sha256` | `DUPLICATE_ANCESTOR` 与窗口身份的键的一部分；报表按它分 ckpt 组 |
| 重复号 | `pair_index`（0..15，缺省 0） | 窗口身份与 `RNG_REUSED` 的槽键的一部分 |
| 配对 RNG | `pair_rng` | `RNG_REUSED` 必须唯一；`generator_seed = pair_rng*1000+7` |

**同一祖先的多个 root 必须自洽**：`stage` / `window` / `prefix_tape` / `takeover_frame` / `total_frames` /
`budget_frames` / `parent_prefix_digest` 任一不同 ⇒ `ANCESTOR_INCONSISTENT`（exit 1）。
`pair_index` 是唯一允许变化的字段。

实测：48 root 清单 `check_count=295`、`codes=[]`，**不加 `--allow-exposed`**（全部 `used_for_*: false`）。

---

## 3. 命令行

### 3.1 `v829_guard.py`
```bash
python -B tools/rl/v829_guard.py check <manifest> [--budget 1800|2400] [--allow-exposed] [--json]
python -B tools/rl/v829_guard.py selftest [--work DIR]
```
退出码：`0` 全通过 · `1` 拒绝 · `2` 无法评估。

### 3.2 `v829_pair.py`
```bash
# 不启引擎，只校验+打印将执行的 argv
python -B tools/rl/v829_pair.py --mode check --manifest M --window W --budget-frames 2400 \
  [--takeover-frame N] [--lineage-config configs/v430/splits.json] --allow-exposed

# 单臂
python -B tools/rl/v829_pair.py --mode trace --manifest M --window W --arm G1 --budget-frames 2400 ...

# 一对（两臂同起点；可选并发）
python -B tools/rl/v829_pair.py --mode pair --manifest M --window W --budget-frames 2400 \
  --ckpt PATH --ckpt-sha256 HEX --pair-rng INT --episodes 1 --arm-order G1,T2 \
  --work-dir .tmp_v829_pair --tag UNIQUE --out OUT.jsonl --log LOG --keep-trace \
  [--workers 2] [--allow-exposed] [--allow-legacy-window]

# 只比两份 trace（不启引擎）
python -B tools/rl/v829_pair.py --mode compare --trace-g1 A.json --trace-t2 B.json
```
`--workers N`（1..4，默认 1）：**一个 worker = 一个 cell = 一个 root**，tag 自动加 `-w<i>`，
每个 worker 写自己的 jsonl，协调器按 cell 顺序合并到 `--out` 并写 `runs/<tag>/workers.json`。

### 3.3 `v829_report.py`
```bash
python -B tools/rl/v829_report.py --pairs 'eval/*.jsonl' [--manifest M] \
  [--json OUT.json] [--markdown OUT.md] [--require-complete] [--expected-ancestors 8]
```
`Δ` 只在「每个预期祖先在该档 ≥1 完整配对」时给出；否则 `null` + 分母 + 界限。
`SOURCE_DATE_EPOCH` 可钉住 `generated_at`，使输出逐字节可复现。

⚠️ **不要把「合并后的 jsonl」和「worker 的 jsonl」一起喂**：`--workers` 会保留
`eval/<tag>-w<i>.jsonl` 并把它们合并成 `eval/<tag>.jsonl`，glob `eval/<tag>*.jsonl` 会把同一行读两遍。
工具会以 `DATA_INTEGRITY: DUPLICATE_ROW @ <file>#L<n>` 拒绝（已实测），只喂其中一个即可。

---

## 4. 关键退出码（`v829_pair.py`）

| 码 | 含义 | 何时出现 |
| ---: | --- | --- |
| 0 | 成功 | — |
| 1 | 被拒绝（可解释） | 曝光冲突、清单改档、帧不合法、谱系矛盾、tag 重复等 |
| 2 | 无法评估（环境/输入不可用） | 清单不可读、引擎缺失、`--out` 不可写、坏 budget |

常见码：`EXPOSURE_CONFLICT` `BUDGET_ROOT_MISMATCH` `PAIR_RNG_CONFLICT` `TAKEOVER_FRAME_NOT_ANCHOR`
`TAKEOVER_BEYOND_RECORDING` `PREFIX_TAPE_MISSING` `PREFIX_RECORDING_MISMATCH` `MANIFEST_CHANGED`
`DUPLICATE_WINDOW` `RNG_REUSED` `LINEAGE_MISSING` `LINEAGE_NEW` `EXPOSURE_CONTRADICTS_LINEAGE` `TAG_NOT_UNIQUE`。

---

## 5. 音频协议（AGENTS.md §9，工具内强制）

每次启动引擎前，工具链自动做三件事，并把证据写进 `runs/<tag>/`：
1. `v450.prepare_env()` → `SDL_AUDIODRIVER=dummy`，pop `DISPLAY`/`WAYLAND_DISPLAY`；
2. `v432.build_private_data_dir` → **重新生成** 60 字节 `th08.cfg`（不是复制），四项复核全 0；
3. 跑完 `audio_audit.json`：要求 `cfg_files ≥ 1`、`errors == 0`、`muted == cfg_files`、`audible == 0`、
   `four_zero == cfg_files`，否则该对作废、非 0 退出。

> 注意 `v431_audio_cfg_audit.decode()` 的 `muted` **不检查 `musicMode`**，所以工具另加 `four_zero` 判据。

---

## 6. 跑一次试点的最小步骤

```bash
# 0) 用 3.10，确认机器空闲
cd /mnt/d/WorkShop/Touhou_RL
ps -eo args= | grep -c '[t]h08-modern'      # 必须是 0

# 1) 研究端给：来源清单（`v829-sources/1` 小 JSON），训练端生成并校验清单
python -B tools/rl/v829_manifest_build.py build --sources S.json --out M.json --allow-exposed
#    → 自动算 sha256/frames/checkpoints，拒绝 4k−1 违规，收尾自己调 guard
#    产物：M.json + M.json.guard.json（sidecar，含 guard 的完整 stdout）
python -B tools/rl/v829_guard.py check M.json --budget 2400     # 必须 exit 0

# 2) 不启引擎预演
python -B tools/rl/v829_pair.py --mode check --manifest M.json --window <W> --budget-frames 2400

# 3) 一对冒烟（先 1 个祖先、1 个 ckpt）
python -B tools/rl/v829_pair.py --mode pair --manifest M.json --window <W> --budget-frames 2400 \
  --ckpt <PATH> --ckpt-sha256 <HEX> --pair-rng <INT> --episodes 1 --tag smoke-01 \
  --out eval/smoke-01.jsonl --log runs/smoke-01/run.log --keep-trace

# 4) 冒烟验收（逐项）
#    gate-*.json: park_frame == N, hash_checks == N//4, hash_mismatches == 0, all_pass true
#    run.json:    rng_same_initial_state true, invalid_rows 0, prefix_constants.override_applied true
#    audio_audit: muted == cfg_files, audible == 0, four_zero == cfg_files
#    compare:     首个字段差异 = 决策 0（threshold），首个行为差异按实际

# 5) 全批（可选并发）
python -B tools/rl/v829_pair.py --mode pair --manifest M.json --budget-frames 2400 \
  --ckpt ... --pair-rng ... --tag pilot-01 --out eval/pilot-01.jsonl --workers 2

# 6) 出表
python -B tools/rl/v829_report.py --pairs 'eval/pilot-01*.jsonl' --manifest M.json \
  --json report.json --markdown report.md
```

---

## 7. 已真机验证 / 未验证

| 项 | 状态 |
| --- | --- |
| 旧窗 8663 起点、两档 | ✅ r2/r3/r4 |
| 非 8663 起点（N=3999，999 检查点） | ✅ r5 |
| 非 S1c 录像（dagger seed3006，N=2999，749 检查点） | ✅ r6 |
| **接管帧边界 N=7（1 检查点）与 N=11399（2,849 检查点）** | ✅ r14（见下表） |
| 2 worker 并发（2 cell → 1 batch，4 行合并） | ✅ r7 |
| 跨进程逐决策确定性（同 seed 两次运行 0 差异） | ✅ r2 vs r4 |
| 满段计分路径 | ✅ r5（双臂 2400/2400） |
| 技术故障重跑一次（行级 / 恢复级 / 两次都失败） | ✅ r10 / r11 / r12（真机 kill 引擎注入） |
| 4 worker 并发（4 cell → 1 batch，8 行合并） | ✅ r13，但**比串行慢 35 %**，不建议 |
| 8 祖先 × 48 对的真实试点 | ⚠️ 没有干净祖先，跑不了 |

### 7.1 接管帧的可行范围（边界实测，window 11，13,200 帧录像，1800 帧预算）

| N | target | 检查点 | mismatches | PARK | reset_hash | G1 / T2 |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| **7**（最小合法） | 8 | **1** | 0 | 7 | `7c8705fc88f94c17` | 满段 / 死 1548 |
| 3,999 | 4,000 | 999 | 0 | 3,999 | `7c8705fc88f94c17` | — |
| 8,663（历史） | 8,664 | 2,165 | 0 | 8,663 | `7c8705fc88f94c17` | 见 r2/r3/r4 |
| **11,399**（最大可行） | 11,400 | **2,849** | 0 | 11,399 | `7c8705fc88f94c17` | 满段 / 满段 |

**四个不同接管帧的 `reset_hash` 完全相同**（frame 0 的重置哈希与 N 无关），而 `final_step_hash` 各自不同
（`daf305e8…` / `6568a22d…` / `0299ade9…` / `80cc0ac1…`）⇒ 「起点同一性」判据与接管帧**解耦**，
这是参数化正确性的直接证据。上界 `N = total_frames − budget` 且必须 `≡ 3 (mod 4)`。

---

## 8. 已知风险（研究端要知道）

1. **曝光半自证**：`used_for_training`/`used_for_selection` 由清单声明；`--lineage-config` 只能核对**已在谱系里**的祖先，
   新祖先走 `lineage_new` 时是**显式承认无法核对**。
2. **`total_frames` 无上界、无来源核对**：必须人工写 `total_frames_source`，工具只校验 `N + budget ≤ total_frames`。
3. **窗口身份键含 `budget_frames`**：同一物理窗口换标签 + 换档位仍可让它在 Δ 里算两次。
4. **录像与祖先的绑定是间接的**：清单声明 id、前缀门钉录像 SHA，两者不是同一张证明。
5. **`v829_guard.py` 的 `exit 0` 只证明「清单自洽地声明了这些数」，不证明这些数是真的。**
