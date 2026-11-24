# v8.29 资产登记提案（**未应用**，等研究端确认）

本文件只列出「若要按资产册规则登记，该往哪里加什么」。**我没有改 `ASSET_SYNC.md` / `ASSET_SYNC.sha256`，也没有 `git add/commit`。**
理由：三件工具与五份清单目前是**训练端提案**，研究端尚未确认它们是契约（见 [回执 03](TRAINING_HOST_REPLY_03.md) §7.1）。

---

## 1. 建议登记为「执行器」（与 §35 里 `tools/rl/v8r0_urgency.py` 同档处理）

| 路径 | 字节 | SHA-256 |
| --- | ---: | --- |
| `tools/rl/v829_guard.py` | 81,189 | `a234502ae27348839b72af2ca03570f605da400d6af4a71ac72dc6114470711a` |
| `tools/rl/v829_pair.py` | 393,031 | `4cfc21638ef0876ebc73c7ad91193db743cbb672cbd22d99410339fb7ec4c8a7` |
| `tools/rl/v829_report.py` | 142,225 | `08c8d86137c9ae09852956b12e586a4a3735eb955bdd8a361296c0719ee60841` |
| `tools/rl/v829_manifest_build.py` | 99,504 | `b8ec952b7872321b15fb31177242cd06f2a77641c8c6c73773a2a807a56e133c` |
| `tools/rl/v829_collect_source.py` | 105,551 | `1414292324b49deb97830311e3cbd90e24f5f8098b00e4eac23344bcaa21d7c9` |

> 五件工具都在演练中继续演进过（`--reuse-park`、路径规范化、`ckpts`/`rngs` 两根轴）。**以本表为准**。

## 2. 建议登记为「清单」（跑一次试点前必须先冻结）

| 路径 | 根数 | 字节 | SHA-256 前 16 |
| --- | ---: | ---: | --- |
| `.tmp_v829_pair/manifest/smoke-old-window.json` | 1 | 4,103 | `a84faca53333c0cb` |
| `.tmp_v829_pair/manifest/smoke-old-window-1800.json` | 1 | 4,311 | `6a1f9fd326d64171` |
| `.tmp_v829_pair/manifest/smoke-old-window-t3999.json` | 1 | 3,408 | `f81ed2a6e9b5af66` |
| `.tmp_v829_pair/manifest/smoke-two-windows.json` | 2 | 5,283 | `f4e74554c29c5c0b` |
| `.tmp_v829_info/s10/manifest-dagger-3006.json` | 1 | 4,603 | `9a8561b7993d3afc` |
| `.tmp_v829_collect/manifest.json`（**新采来源** seed 3101，N=2999） | 1 | 3,159 | `a3962805a42ea5a3` |
| `.tmp_v829_collect/manifest-reuse.json`（PARK 复用验证，3 root） | 3 | 6,442 | 见文件 |

> 注意：前五份是**冒烟清单**（`smoke_only: true` 或 `used_for_*: true`）；
> 后两份来自**本轮新采的、未曝光**来源（`used_for_*: false`），建议按「来源登记」而不是「冒烟证据」处理。

## 2b. 建议登记为「新采来源」（本轮新增，训练端自造）

| 路径 | 内容 | 字节 | SHA-256 |
| --- | --- | ---: | --- |
| `.tmp_v829_collect/seed3101-frozen/labels.jsonl` | 4 帧录像（1,200 行 / 4,800 帧 / 0 死亡） | 448,955 | `f0fd3cd2261cb0499d754a8fbcba2c6eb5a509e8b6fcbf9a863b5c5c72625f31` |
| `.tmp_v829_collect/seed3101-frozen/shard.json` | 采集元数据（seed / stage / policy / 引擎与 cfg SHA / 哈希链） | 见文件 | 见文件 |
| `.tmp_v829_collect/seed3101-frozen/collect.json` | 命令、环境、静音审计、自检 | 见文件 | 见文件 |

采集条件：`seed=3101 stage=0 difficulty=1`（冻结 RESET）、策略 `bombdag.pt seed0`、`wall=4.137 s`、
`hash_checks=1201 / mismatches=0`、`audio.all_muted` 与 `post_run_muted` 均为 true。
**它是管线演示，不是科学祖先**（起点状态依赖生成它的策略）；研究端若要自然起点需指定生成策略。

## 3. 建议登记为「运行证据」（`.tmp_v829_pair/runs/<tag>/` 下每份的 `run.json` + `gate-w*.json` + `audio_audit.json`）

| tag | 用途 | 关键结果 |
| --- | --- | --- |
| `v829-smoke-w11-c0-b2400-r2` | 旧窗 8663 基线 | 两臂同轨迹，死 10629 |
| `v829-smoke-w11-c0-b1800-r3` | 30 秒档 | 两臂分叉（1538 vs 1447 帧） |
| `v829-smoke-w11-c0-b2400-r4` | 跨进程确定性 | 与 r2 逐决策 0 差异 |
| `v829-smoke-w11-t3999-c0-r5` | 非 8663 接管帧 | 999 检查点、两臂满段 |
| `v829-smoke-dagger3006-t2999-r6` | 非 S1c 录像源 | 749 检查点、录像注入 2 对象 |
| `v829-conc-two-windows-r7` | 2 worker 并发 | 4 行合并、两 worker 各自静音 |
| `v829-retry-fault-r10` | 行级故障重跑 | attempts=2、4 行保留 |
| `v829-retry-parkkill-r11` | 恢复失败重跑 | attempts=2、rc=0 |
| `v829-retry-alwaysfail-r12` | 两次都失败 | rc=1、两条 recovery 记录 |

## 4. 如果研究端接受，登记时建议的两处补充

1. **`ASSET_SYNC.sha256` 是 LF 归一化 SHA 且自身为 CRLF**（见 [回执 01](TRAINING_HOST_REPLY_01.md) §9 勘误第 3 条）：
   WSL 里直接 `sha256sum -c` 会全线失败，追加行前请用 `tr -d '\r'` 口径核对。
2. **三件工具的 `__pycache__` 不应入册**（`tools/rl/__pycache__/v829_*.pyc` 由 `py_compile` 产生，已被 `.gitignore` 覆盖）。

---

## 5. 我没有做的事（明确记录）

- 没有执行任何 `sha256sum -c` 之外的写操作；`ASSET_SYNC.md` / `ASSET_SYNC.sha256` 未被修改。
- 没有把这 8 个工具/清单/运行目录 `git add`。
- 没有把 `.tmp_v829_pair/` 的任何二进制或 trace（~1.5 GB 级）纳入候选登记范围 —— 那些是运行件，不是证据件。
