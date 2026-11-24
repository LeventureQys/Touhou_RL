# v8.29 评审包（给研究院）

> 本文件是入口。它不重复回执里的细节，只说明**该看什么、结论是什么、还差什么决定**。
> 一句话状态：**工程链路已经跑到「完整两档主批 + 判定」；现在缺的不是代码，是三处实验设计决定（§4）。**

| 项 | 值 |
| --- | --- |
| 日期 | 2026-11-24（+08:00） |
| 远端 | `origin/v400`，提交 `75c431e` |
| 交付物 | 5 个工具（`tools/rl/v829_*.py`）+ 8 份文档（本目录） |
| 训练端合规 | 未训练、未恢复 WP05、未上传任何资产；引擎每次都静音审计 |

---

## 1. 三十秒摘要

| 能力 | 证据 | 状态 |
| --- | --- | --- |
| 历史对照可复算 | G1 146/240 · T2 141/240，8 个窗口逐格 +0 | ✅ 已验证 |
| 任意 `4k−1` 接管帧 | N=7 / 3999 / 11399，检查点 1 / 999 / 2849，0 失配 | ✅ 已验证 |
| 起点同一性不随接管帧变 | 四个 N 的 `reset_hash` 全是 `7c8705fc88f94c17` | ✅ 已验证 |
| 训练端能自造干净祖先 | 8 盘全新录像 34 s 采完，`used_for_*: false`，守卫不加 `--allow-exposed` 即接受 | ✅ 已验证 |
| 预注册的重复测量轴 | 8 祖先 × 3 ckpt × 2 RNG × 2 档 = 96 root，两档各 584 项守卫检查 | ✅ 已验证 |
| 完整两档主批能跑 | 192 单臂 344 s，16 次 PARK（省 80），0 无效，0 残留 | ✅ 已验证 |
| 报表五门全过并给出判定 | `CHECKPOINT_AXIS {1800:3, 2400:3}`；Δ30=Δ40=0.00 pp；`UNDETERMINED` | ✅ 已验证 |
| **能区分 G1 与 T2 的祖先** | 8 个演练祖先在 N=2399 下 **192 格零臂间差异** | ❌ **这正是 §4 要定的** |

## 2. 阅读路径

| 预算 | 读什么 |
| --- | --- |
| 3 分钟 | 本文件 §1 + §4 |
| 10 分钟 | [`TRAINING_HOST_REPLY_06.md`](TRAINING_HOST_REPLY_06.md)（两档形态、成本修正、接管帧约束）+ [`INTEGRATION_NOTE.md`](INTEGRATION_NOTE.md) §2.5–2.7 |
| 30 分钟 | `TRAINING_HOST_REPLY_01` → `06`（按轮次：盘点 → 冒烟 → 解阻塞 → 自造来源 → 单档演练 → 两档演练） |
| 要复算 | `python tools/rl/v829_report.py selftest`（170/170，含 §4 计分的两条黄金向量）；`python tools/rl/v829_guard.py selftest`（57/57） |
| 关心资产 | [`ASSET_REGISTRATION_PROPOSAL.md`](ASSET_REGISTRATION_PROPOSAL.md)（**提案，未应用**——`ASSET_SYNC.md` 尚未改） |

## 3. 清单长什么样（真实摘录）

顶层（96-root 演练清单，`v829-pair-manifest/1`）：

```json
{ "schema": "v829-pair-manifest/1",
  "manifest_id": "v829-rehearsal-8x3x2x2-001-manifest",
  "purpose": "… each root replays its own 4-frame prefix tape up to a 4k - 1 takeover frame, then runs 1800 and 2400 frames (one root per budget tier, the section 2 endpoint axis) per arm …",
  "engine": "/root/Touhou_RL/third_party/th08-rl/build/rl_v3/th08-modern",
  "data_source": "/root/v400-data/game",
  "budgets": [1800, 2400],
  "builder": { "tool": "tools/rl/v829_manifest_build.py", "version": "v8.30.0", … },
  "roots": [ … 96 个 … ] }
```

一个 root（1800 档，节选关键字段）：

```json
{ "ancestor_id": "reh-seed3201",
  "stage": "th08-stage1-normal-team0",
  "window": 3201,
  "budget_frames": 1800,
  "ckpt_sha256": "9c6d5d4ac7c9a612…", "ckpt_path": ".tmp_v860_bombdag/ckpt/seed0/bombdag.pt",
  "engine_sha256": "691bfd77b7eca336…",
  "pair_rng": 20265000,
  "takeover_frame": 2399, "total_frames": 4800,
  "parent_prefix_digest": "f07f4242be930f87…",
  "prefix_tape": ".tmp_v829_collect/seed3201/labels.jsonl",
  "prefix_tape_rows": 1200, "prefix_tape_frames": 4800,
  "verify_anchor": { "recorded_frames": 2399, "hash_checkpoints": 599,
                     "note": "N = 2399 = 4 x 599 + 3, so target = 2400 is a recorded 4-frame anchor …" },
  "used_for_training": false, "used_for_selection": false,
  "pair_index": 0, "budget_index": 0, "lineage_new": true }
```

同一祖先、同一 ckpt、同一重复号在 **2400 档**的兄弟 root，只有这些字段不同：

```json
{ "ancestor_id": "reh-seed3201", "budget_frames": 2400,
  "ckpt_sha256": "9c6d5d4ac7c9a612…", "pair_index": 0,
  "pair_rng": 20281000, "takeover_frame": 2399, "total_frames": 4800 }
```

**约定**：`ancestor_id` 是独立对局根；`ckpt_sha256` / `pair_index` / `budget_frames` 是三根**重复测量轴**；
`takeover_frame` / `total_frames` / tape / digest 在同一祖先内**必须一致**（守卫 `ANCESTOR_INCONSISTENT`）。
`--window` 与 `--budget-frames` 是运行时**选择器**；`--pair-rng` / `--takeover-frame` / `--prefix-tape` / `--ckpt-sha256` 是**登记值**，必须与清单相等。

## 4. 三处必须由研究端定的决定

### 4.1 接管帧取多晚 —— 有硬约束，且影响信息量

`takeover_frame + 最大档预算 ≤ total_frames`，且 `N ≡ 3 (mod 4)`（因为 `target = N+1` 必须是已记录锚点）。

| tape | 两档都可行的 N | 实测后果 |
| ---: | ---: | --- |
| 4,800 帧 | ≤ 2,399 | **2,399：两档都能跑，但 192 格零臂间差异**；2,999：40 s 档装不下（5,399 > 4,800） |
| 5,600 帧（1,400 块，9.6 s 采集） | ≤ 3,199 | build + guard 双档通过（余 601 / 1），**尚未真机跑** |
| 更大的 tape | 更大 | 采集成本线性：≈ 4.2 s / 4,800 帧 |

**要定**：正式试点用哪个 N、以及为了它是否接受更长的录像（每多 800 帧约多 1.5 s 采集、约多 16 KB/盘的 tape）。

### 4.2 祖先筛选准则 —— 现阶段最大的缺口

演练实测（8 个新采祖先，N=2399，192 格）：

| 现象 | 数字 |
| --- | --- |
| 全程零发射（`nearest_px` 从未低于任何阈值 ⇒ 两臂行为必然相同） | 曾出现 24/48 格（另一组起点） |
| 接管后立刻死（三个 ckpt 都在第 3 帧结束） | 1/8 祖先 |
| 两臂任何差异 | **0/192 格** |

**要定**：什么样的祖先是「有区分力」的。训练端可提供的操作性候选（供讨论，不是设计结论）：
接管后前 300 帧内至少 N 次进入 G1 的 40 px 阈值；至少一定比例的格能活过 600 帧；同祖先内 ckpt 间存活跨度不超过某倍数。

### 4.3 自然起点 vs 策略自生起点

现在 8 盘录像的前缀是**被评测策略自己滚出来的**（`bombdag.pt` seed0）。所以「起点状态」依赖被评测对象，严格说不是中立的对局根。
`used_for_training: false` 只表示**没进过训练/选参**，不表示起点中立。

**要定**：接受策略自生起点，还是要求教师/人类/固定脚本生成的起点（后者需要另一次采集批次）。

## 5. 成本（排期用，实测）

| 环节 | 时间 |
| --- | ---: |
| 采 1 盘 4,800 帧 | ≈ 4.2 s |
| 采 1 盘 5,600 帧 | ≈ 9.6 s |
| 48 cell / 96 单臂（单档） | **176 s**（30 s 档）/ **168 s**（40 s 档） |
| **完整两档试点（96 对 / 192 单臂）** | **≈ 344 s（≈ 6 分钟）** |
| 同上但关掉 `--reuse-park` | 每档多 40 次 PARK ⇒ 约 11 分钟 |

## 6. 已验 / 未验

**已验证（可复算）**：历史对照零偏差；任意 `4k−1` 接管帧；起点同一性；自造来源；三根重复测量轴的清单表达；
两档真机跑通；报表五门全过与判定路径；跨平台逐字节相同（Windows 3.12 与 WSL 3.10 的 `--json` 同 SHA）。

**未验证（诚实列出）**：
1. 5,600 帧 tape 在 N=3199 上的**真机**两档（只做了 build + guard）。
2. 任何**有区分力**的祖先（本轮 192 格全同）。
3. 48 cell 对应关系下 `pair_index` 的防伪：换标签 + 重复号 +1 的副本与合法第二测量同形，只能靠种子簿记区分。
4. 真实 `--mode pair --workers N` 多进程（只验过 `--mode check --workers`；此前 r7/r13 验过 2/4 worker 的合并）。

## 7. 我们刻意没做的事

- **没有在 4,800 帧 tape 上跑 N=2999 的两档**：物理上跑不了（5,399 > 4,800），builder/guard 都会拒。
- **没有把演练数值当作结论**：所有回执都标了 REHEARSAL；`UNDETERMINED` 是报表的真实输出，不是修辞。
- **没有改 `ASSET_SYNC.md`**：登记提案还需研究端接受。
- **没有训练、没有恢复 WP05**。

## 8. 复现命令（训练端环境）

```bash
PY=/root/v400-venv/bin/python          # Python 3.10.12，必须用它

# 1. 自检（不启引擎）
$PY -B tools/rl/v829_guard.py selftest          # 57/57
$PY -B tools/rl/v829_report.py selftest         # 170/170
$PY -B tools/rl/v829_collect_source.py selftest # 41/41

# 2. 从来源清单生成并校验清单（不启引擎）
$PY -B tools/rl/v829_manifest_build.py build --sources <sources>.json --out <manifest>.json
$PY -B tools/rl/v829_guard.py check <manifest>.json --budget 1800

# 3. 真机一档（每档 48 cell / 96 单臂；关声由工具强制）
$PY -B tools/rl/v829_pair.py --mode pair --manifest <manifest>.json \
    --budget-frames 1800 --reuse-park --tag <tag>-30s --allow-legacy-window

# 4. 出表
$PY -B tools/rl/v829_report.py --pairs <tag>-30s.jsonl --pairs <tag>-40s.jsonl --expected-ancestors 8
```
