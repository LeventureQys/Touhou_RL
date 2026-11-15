# W00 probe 归位报告（P1-07 收尾：4 个只读诊断脚本）

- 版本/工作包：v4.3.0 / **W00**（P1-07 的最后一个缺口）
- 执行：Provider=DeepSeek / DeepSeek Flash（子代理）；主代理指派的独立任务
- 日期：2026-11-15（Asia/Shanghai）
- 主机/仓库：`D:\WorkShop\Touhou_RL`；Windows；Python 3.12.10；numpy 2.5.4
- 依据：[`V420_REVIEW.md`](../../V420_REVIEW.md) P1-07；v4.2.0 报告中的实际引用（见 §1）
- 结论：**4 个脚本全部有引用、全部只读、全部归位**；编译、`--help`、合成等价、真实 shard 只读复现均有真实退出码；无 not_run 的归位项（§8）。
- 本文件只登记**实际执行过**的命令与它们的真实退出码，不登记“预期”结果。

---

## 0 结论摘要

| # | 原件（仓库根，未动） | 原 sha256 | v4.2.0 引用处 | 安全审查 | 归位 | 新路径（行数 / sha256） |
|---:|---|---|---|---|---|---|
| 1 | `.tmp_v420_slot_probe.py` | `ceff33d89616482bd91ae5dbc81fd91b47413087198d4339c1b60a68609f8427` | PREFLIGHT:101；T01:18,:28 | 只读 | ✅ | `tools/rl/v420_slot_probe.py`（144 行 / `53ed7f715b340c05d282bf9b11f6764787f04fb1eb12538d7f723a1ba02c5b09`） |
| 2 | `.tmp_v420_hist_probe.py` | `81820648ee00f1dc51c33f5915457ad40e08bcb38437042a2b9a8ec89796329f` | PREFLIGHT:101；T01:25,:29 | 只读 | ✅ | `tools/rl/v420_hist_probe.py`（130 行 / `c28b972ab283202c115566c56574cc062a38d8d39122581d164812d1dadc5353`） |
| 3 | `.tmp_v420_mask_probe.py` | `262fe55f7c8bd30816a7eae9b50293351b5d2d64fac0c1bb5510778eb11a738d` | PREFLIGHT:101；T01:23 | 只读（含 1 处硬编码绝对仓库根，已参数化） | ✅ | `tools/rl/v420_mask_probe.py`（108 行 / `32d78fcb6cbdfefde9770034fe1f654f06d9ec48423e4c9ee39864276e4dd5a2`） |
| 4 | `.tmp_v420_release_probe.py` | `8239b6773e5dfec0908bd72aff6c7d0f7ecefa822452b3def09dbe9789de391b` | PREFLIGHT:101；T01:24 | 只读（同上） | ✅ | `tools/rl/v420_release_probe.py`（122 行 / `749cf1f6c00dd6c586f4792d1980cdd233ca175f1958f61aa3f621916d1388bb`） |

**没有任何脚本被移出归位清单**（4 个都在 v4.2.0 正文里被点名，见 §1）。

---

## 1 引用复核（独立重做，精确文件名）

方法：对 `Document/Update/Alpha/v4.2.0/` **整目录**（`.md/.json/.txt/.jsonl`，不止 `.md`）用 4 个精确文件名 grep；再用 4 个文件名主干（`slot_probe|hist_probe|mask_probe|release_probe`）二次交叉确认，避免只匹配带 `.tmp_` 前缀的写法。命令在仓库根执行，路径显式指定，不对仓库根递归扫描（避开 `.pytest_cache` 的 `os error 5`）。

| 被引用的原件 | 文档:行号 | 引用内容（摘） |
|---|---|---|
| `.tmp_v420_slot_probe.py` | `experiments/T01-contract-spec.md:18` | 每行推进逻辑帧 **4**（`decision_frame` 步长 `{4: 3299}`） |
| | `experiments/T01-contract-spec.md:28` | HASH 链 `hash_post[i] == hash_pre[i+1]`，**0 断点** |
| | `evidence/PREFLIGHT/PREFLIGHT.md:101` | 「脚本：`.tmp_v420_slot_probe.py`、`.tmp_v420_hist_probe.py`、…」 |
| `.tmp_v420_hist_probe.py` | `experiments/T01-contract-spec.md:25` | `act_hist[r]` = `first_action[r-8..r-1]` one-hot，前补零，**0 失配** |
| | `experiments/T01-contract-spec.md:29` | `bomb` 全 0（`bomb nonzero: 0`） |
| | `evidence/PREFLIGHT/PREFLIGHT.md:101` | 同上 |
| `.tmp_v420_mask_probe.py` | `experiments/T01-contract-spec.md:23` | `3300/3300` 与 `3163/3300`，差 137 |
| | `evidence/PREFLIGHT/PREFLIGHT.md:101` | 同上 |
| `.tmp_v420_release_probe.py` | `experiments/T01-contract-spec.md:24` | 137 处差异全部 `shoot_release=True`、全部只发生在块内第 4 帧 |
| | `evidence/PREFLIGHT/PREFLIGHT.md:101` | 同上 |

- 合计 **10 处**引用：slot 3、hist 3、mask 2、release 2。
- 与 [`EXIT_CODES_RESTORE.txt`](EXIT_CODES_RESTORE.txt) §1（另一位代理对本任务同一批文件的独立扫描，计数 3/3/2/2）**逐条一致**，互为交叉验证。
- 主干 grep（`slot_probe|hist_probe|mask_probe|release_probe`）只命中上面两处文档，没有出现第三种写法；v4.2.0 报告**正文**点名的只读诊断 py 就是这 4 个。`.tmp_v420_replay_probe.py` 这个名字只出现在已保存副本 `replay/replay_probe.py` 自己的 docstring 第 7 行（不是报告正文的点名），见 §8。

---

## 2 安全审查（逐文件读全文后的结论）

审查项：是否写文件、覆盖旧证据、起子进程、连网络、读个人路径、改旧资产。逐行核对的结论：

| 脚本 | 读取操作（全部只读） | 写文件 | 子进程 | 网络 | 个人/机器绝对路径 | 改旧证据/资产 | 结论 |
|---|---|---|---|---|---|---|---|
| slot | `open(labels.jsonl, "r")`、`open(shard.json, "r")` | 无 | 无 | 无 | 无（`SHARD` 是**仓库相对**路径） | 无 | 可归位 |
| hist | `open(labels.jsonl, "r")`、`np.load(act_hist.npy)`（默认 `mode="r"`） | 无 | 无 | 无 | 无 | 无 | 可归位 |
| mask | `open(labels.jsonl, "r")`；`from touhou_rl.v400.gym.actions import cand_to_mask` | 无 | 无 | 无 | **有**：原第 5 行 `sys.path.insert(0, r"D:\WorkShop\Touhou_RL")` | 无 | 可归位（该路径必须参数化） |
| release | 同 mask，另加 `mask_to_cand` | 无 | 无 | 无 | **有**：同 mask 第 5 行 | 无 | 可归位（同上） |

补充说明：

- 4 个脚本的全部输出都只到 **stdout**（`print`）；唯一的 stderr 写入是本次新加的 fail-closed 错误行（§3(e)）。
- `mask_to_cand` 只在 `try/except ValueError` 里用于**打印**解码结果，不写回任何文件。
- 四个脚本都**没有** `__main__` 入口、没有 `argparse`、没有输出重定向、没有 `.write`/`json.dump`/`np.save`。
- 没有 `socket`/`requests`/`urllib`、没有 `subprocess`/`os.system`、没有 `shutil`/`expanduser`/`Path.home`、没有任何驱动盘绝对路径（机器扫描见 §4）。
- 运行对象是 `.tmp_v410_s1c/d1/shards/d1-s0d1-n3300-seed10-shard` 里 3 个文件（`labels.jsonl`、`shard.json`、`obs/act_hist.npy`）。**运行前后 sha256 与 mtime 均未变化**（§9）。
- 结论：**没有任何一个会写文件 / 覆盖旧证据 / 起子进程**，因此 §3 全部归位，无需保留在原位或改用替代方案。

---

## 3 归位决定与改动点（逐条理由）

决定：4 个全部归位为 `tools/rl/v420_{slot,hist,mask,release}_probe.py`。**原始逻辑一行未改**（分析循环、打印标签、计数器、切片、`json.dumps(...)[:600]` 截断、甚至 hist 里未被调用的 `onehot` 和 `first_bad_A` 诊断分支都原样保留）。改动只有下面 7 类：

| 编号 | 改动 | 理由 |
|---|---|---|
| (a) | 文件头新增 docstring + provenance 注释块：原始路径、原始 sha256、归位日期 2026-11-15、原因（P1-07 / v4.3.0 W00）、“原始文件未删除、仍在仓库根” | P1-07 要求可追溯；块内 sha256 用 `Get-FileHash -Algorithm SHA256` 实算，不是抄报告 |
| (b) | 新增 `--shard` 参数，默认值 = 原脚本硬编码的仓库相对路径 `.tmp_v410_s1c/d1/shards/d1-s0d1-n3300-seed10-shard` | 保留“无参数即可复现 v4.2.0 数字”的行为；同时允许在其它 shard 上复用，不必改源码 |
| (c) | 相对 `--shard` 按**仓库根**解析（原来按 CWD 解析） | 脚本已从仓库根搬到 `tools/rl/`；若仍按 CWD 解析，从 `tools/rl` 内运行会静默读错位置。从仓库根运行的语义与原件**完全等价**（§6 无参数运行验证 stdout 逐字节相同）；从别的 CWD 运行则从“报错”变成“正确”（§7 第 9 项） |
| (d) | mask/release 的硬编码绝对仓库根 `sys.path.insert(0, r"D:\WorkShop\Touhou_RL")` 改为 `_repo_root()`：优先环境变量 `TOUHOU_RL_ROOT`，否则由 `__file__` 推导（`tools/rl/x.py` 上两级）；随后校验 `touhou_rl/v400/gym/actions.py` 存在，不存在就 stderr + **exit 2** | 任务要求“个人/WSL 绝对路径改为参数或环境变量，缺参数时报错退出而不是猜默认值”。仓库根不是数据路径，可由 `__file__` 唯一确定，但仍保留 env 覆盖 + fail-closed 校验，避免“猜一个目录然后 import 到别的东西” |
| (e) | 缺失输入从“抛 `FileNotFoundError` traceback（exit 1）”改为“stderr 一行错误 + **exit 2**” | fail-closed：`exit 1` 与 traceback 容易被当成崩溃而忽略；`exit 2` 与 package 内既有约定（`v420_expected.py` 的 MISSING 分支）一致。**stdout 不受影响**（成功路径逐字节相同，§5/§6） |
| (f) | 补最小 `argparse` 入口：`main(argv=None) -> int` + `if __name__ == "__main__": raise SystemExit(main())` | 原件是没有入口的裸脚本，`--help` 与错误处理都无处挂载；这是任务显式要求的“缺 `__main__` 就补” |
| (g) | 删除“无用的 import/死代码/缩短打印”等一切“顺手优化” | 任务禁止顺手优化；保留全部原有输出，才能与原件做逐字节比对 |

**未做**：没有删除/移动/重命名任何 `.tmp_v420_*`；没有 `git add`/commit；没有改 `ASSET_SYNC.*`、旧证据、账本；没有启动引擎、没有训练。

---

## 4 静态验证（命令 + 真实退出码）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m compileall -q tools/rl/v420_slot_probe.py tools/rl/v420_hist_probe.py tools/rl/v420_mask_probe.py tools/rl/v420_release_probe.py
# => exit 0
```

| 命令（cwd = 仓库根） | 退出码 |
|---|---:|
| `python -m compileall -q tools/rl/v420_slot_probe.py tools/rl/v420_hist_probe.py tools/rl/v420_mask_probe.py tools/rl/v420_release_probe.py` | **0** |
| `python tools/rl/v420_slot_probe.py --help` | **0** |
| `python tools/rl/v420_hist_probe.py --help` | **0** |
| `python tools/rl/v420_mask_probe.py --help` | **0** |
| `python tools/rl/v420_release_probe.py --help` | **0** |

`--help` 实测首行（互不相同，说明入口各自生效）：
`usage: v420_slot_probe.py [-h] [--shard SHARD]` / `usage: v420_hist_probe.py [-h] [--shard SHARD]` / `usage: v420_mask_probe.py [-h] [--shard SHARD]` / `usage: v420_release_probe.py [-h] [--shard SHARD]`

静态扫描（`Select-String`，4 个新文件；避免歧义，`json.dump(` 带左括号、盘符用大小写敏感 `[A-Za-z]:\`）：

| 危险 token | 命中 |
|---|---:|
| `write_text` / `write_bytes` / `json.dump(` | 0 / 0 / 0 |
| `subprocess` / `os.system` / `shutil.` | 0 / 0 / 0 |
| `socket.` / `requests.` / `urllib` | 0 / 0 / 0 |
| `expanduser` / `Path.home` | 0 / 0 |
| 任意盘符绝对路径 `[A-Za-z]:\`（`-CaseSensitive`） | **0** |

`open(` 出现 4 处，全部带 `"r"`：`v420_slot_probe.py:75/131`、`v420_hist_probe.py:77`、`v420_mask_probe.py:77`、`v420_release_probe.py:78`；`np.load` 1 处（`v420_hist_probe.py:81`，numpy 默认 `mode="r"`，非 `mmap_mode` 写）。

---

## 5 合成输入等价性（不接触引擎、不写旧证据）

夹具放在**仓库外**（`%TEMP%\w00_probe_restore_20261115`），不是交付物：

- 生成器：`%TEMP%\w00_probe_restore_20261115\make_synth.py`
- clean 夹具：`…\clean\.tmp_v410_s1c\d1\shards\d1-s0d1-n3300-seed10-shard`（5 行、每行 4 逻辑帧、第 4 行故意做成 `shoot_release`：`step_frames[3]` 清掉开火位）
- bad 夹具：`…\bad\…`（同一 labels，`act_hist` 第 2 行被污染，用来证明 hist 的比较逻辑不是“恒真”）

做法：**同一份输入**分别喂给原件与归位件，逐字节比较 stdout。
原件按 CWD 解析相对 `SHARD`，所以让原件以夹具根为 CWD 运行；归位件用 `--shard <夹具绝对路径>`。

| 用例 | 原件退出码 | 归位件退出码 | stdout 逐字节相同 |
|---|---:|---:|---|
| `slot` / clean | 0 | 0 | **True** |
| `hist` / clean | 0 | 0 | **True** |
| `mask` / clean | 0 | 0 | **True** |
| `release` / clean | 0 | 0 | **True** |
| `hist` / bad（污染行） | 0 | 0 | **True** |

clean 夹具原样输出（归位件；与原件逐字节相同）：

```text
# python tools/rl/v420_slot_probe.py --shard <clean shard>            => exit 0
rows: 5
frames_per_decision: 4 decision_frames: 2
step_frames len: 4 [1, 1, 129, 129]
decision_frame of first 4: [100, 104, 108, 112]
frame of first 4: [100, 104, 108, 112]
duration histogram: {2: 3, 1: 2}
slots length histogram: {2: 5}
slots[0]==first_action: {True: 5}
first_action histogram: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
slots[1]==slots[0] (when duration>=2): {False: 3}
hash chain breaks: 0 []
decision_frame stride histogram: {4: 4}
act_hist obs source: None
obs keys: ['act_hist', 'observation_time', 'row']
obs observation_time: 10.5
obs row/frame fields: {'row': 0}
--- shard.json
decision_count 5 frames_committed 20 first_decision_frame 100 last_decision_frame 116 stage_frames 9999 deaths_observed 0 stage_clear False
alignment: {"label_is_first_slot_not_continuation": true, "note": "synthetic"}
act_hist block: {"length": 8, "source": "first_action", "note": "synthetic"}
d1 report hash_alignment: {"post_eq_next_pre": true, "breaks": 0}

# python tools/rl/v420_hist_probe.py --shard <clean shard>            => exit 0
act_hist shape: (5, 8, 18) uint8
hypothesis A (last 8 first_actions, front-padded) mismatches: 0 first: None
rows with duration>=2: 3
all slots[1]==slots[0] when duration>=2: False
label step_frames values: {(129, 129, 97, 97), (1, 1, 129, 129), (17, 17, 161, 161), (161, 161, 65, 64), (145, 145, 33, 33)}
shoot_release True count: 1
bomb nonzero: 0

# python tools/rl/v420_mask_probe.py --shard <clean shard>            => exit 0
rows: 5
slot0 mask matches step_frames[0:2]: 5
slot1 mask matches step_frames[2:4]: 4
neither matched: 0
sample: (0, [1, 1, 129, 129], 1, 129, 0, 3, 2)
sample: (1, [17, 17, 161, 161], 17, 161, 1, 4, 1)
sample: (2, [145, 145, 33, 33], 145, 33, 2, 5, 2)

# python tools/rl/v420_release_probe.py --shard <clean shard>         => exit 0
mismatch rows: 1
shoot_release True rows: 1
all mismatch rows have shoot_release: True
--- row 4 decision_frame 116 slots0/1 4 7 step_frames [161, 161, 65, 64] shoot_release 1 flags {'shoot_release': True} prev decision_frame 112 prev step_frames [129, 129, 97, 97]
   decode slot1 cand: (7, 0)
mismatch position: {'first_only': 0, 'second_only': 1, 'both': 0}
```

污染夹具（`act_hist` 第 2 行置零并点亮 cand 17）——证明失配检测真的在比较，而不是恒报 0：

```text
# python tools/rl/v420_hist_probe.py --shard <bad shard>              => exit 0
act_hist shape: (5, 8, 18) uint8
hypothesis A (last 8 first_actions, front-padded) mismatches: 1 first: 2
row 2 got nonzero rows: [0, 0, 0, 0, 1, 1, 1, 1]
expected nonzero rows: [0, 0, 0, 0, 1, 1, 1, 1]
got argmax: [-1, -1, -1, -1, 0, 1, 2, 3]
first_action window r-9..r: [0, 1, 2]
rows with duration>=2: 3
all slots[1]==slots[0] when duration>=2: False
label step_frames values: {(129, 129, 97, 97), (1, 1, 129, 129), (17, 17, 161, 161), (161, 161, 65, 64), (145, 145, 33, 33)}
shoot_release True count: 1
bomb nonzero: 0
```

> 合成输出**不代表真实数据结论**，只用于证明“归位件与原件在同一输入上行为一致”。

---

## 6 真实 S1 shard 只读复现（不启引擎、不写旧证据）

对象就是 v4.2.0 报告用的那一个 shard：`.tmp_v410_s1c/d1/shards/d1-s0d1-n3300-seed10-shard`。

```powershell
# cwd = 仓库根；原件与归位件都用默认参数（归位件因此仍在验证原路径语义）
python .tmp_v420_slot_probe.py     *> real_orig_slot.txt     # exit 0
python tools/rl/v420_slot_probe.py *> real_new_slot.txt      # exit 0
# hist / mask / release 同上
```

| 脚本 | 原件退出码 | 归位件退出码 | stdout 逐字节相同 |
|---|---:|---:|---|
| slot | 0 | 0 | **True** |
| hist | 0 | 0 | **True** |
| mask | 0 | 0 | **True** |
| release | 0 | 0 | **True** |

与 v4.2.0 已发布数字逐项对照（**归位件**实际输出）：

| 文档声明 | 文档位置 | 本次归位件输出 | 一致 |
|---|---|---|---|
| `decision_frame` 步长全部为 4，直方图 `{4: 3299}` | T01:18 | `decision_frame stride histogram: {4: 3299}` | ✅ |
| `step_frames[0:2] == cand_to_mask(slots[0], bomb)` 3300/3300 | T01:23 | `slot0 mask matches step_frames[0:2]: 3300` | ✅ |
| `step_frames[2:4] == cand_to_mask(slots[1], 0)` 3163/3300（差 137） | T01:23 | `slot1 mask matches step_frames[2:4]: 3163`（`neither matched: 0`） | ✅ |
| 137 处差异全部 `shoot_release=True`、全部只在块内第 4 帧 | T01:24 | `mismatch rows: 137`、`all mismatch rows have shoot_release: True`、`mismatch position: {'first_only': 0, 'second_only': 137, 'both': 0}` | ✅ |
| `act_hist[r]` = 最近 8 个 `first_action` one-hot，前补零，0 失配 | T01:25 | `hypothesis A … mismatches: 0 first: None` | ✅ |
| HASH 链 `hash_post[i] == hash_pre[i+1]`，0 断点 | T01:28 | `hash chain breaks: 0 []` | ✅ |
| `bomb` 全 0（`bomb nonzero: 0`） | T01:29 | `bomb nonzero: 0` | ✅ |
| `chosen_plan.duration` 2 → 588 行；1 → 2,712 行 | PREFLIGHT:109 | `duration histogram: {2: 588, 1: 2712}` | ✅ |
| `shoot_release` 行数 = 137 | T01:24 / PREFLIGHT:113 | `shoot_release True count: 137` | ✅ |

归位件关键输出（`slot` 的 `alignment` 行在终端里是 600 字符截断，此处按原样保留该行的前缀）：

```text
# python tools/rl/v420_slot_probe.py                                  => exit 0
rows: 3300
frames_per_decision: 4 decision_frames: 2
step_frames len: 4 [65, 65, 65, 65]
decision_frame of first 4: [0, 4, 8, 12]
frame of first 4: [0, 4, 8, 12]
duration histogram: {2: 588, 1: 2712}
slots length histogram: {16: 3300}
slots[0]==first_action: {True: 3300}
first_action histogram: {0: 1443, 1: 73, 2: 307, 3: 335, 4: 63, 5: 45, 6: 222, 7: 239, 8: 49, 10: 8, 11: 190, 12: 214, 13: 24, 14: 5, 15: 31, 16: 34, 17: 18}
slots[1]==slots[0] (when duration>=2): {True: 588}
hash chain breaks: 0 []
decision_frame stride histogram: {4: 3299}
act_hist obs source: obs/act_hist.npy
obs keys: ['shard', 'shard_dir', 'play', 'hud', 'act_hist', 'row', 'observation_time', 'stack', 'stack_order', 'render_frames', 'newest_render_frame', 'label_frame', 'label_frames', 'p3_window_last_row', 'p3_label_frame', 'pad']
obs observation_time: decision_time_before_commit
obs row/frame fields: {'shard': 'd1-s0d1-n3300-seed10-shard', 'row': 0}
--- shard.json
decision_count 3300 frames_committed 13200 first_decision_frame 0 last_decision_frame 13196 stage_frames [0, 13200] deaths_observed 0 stage_clear False
alignment: {"observation_time": "decision_time_before_commit", "observation_time_note": "the 4 most recent rendered frames at the instant the teacher is queried, before the decision's STEP; newest render = decision_frame - 1 because AfterDrawFrame publishes before ++g_frameCount", "label_time": "scores[18] ranks the plan's first student decision (slot 0), applied to frames [decision_frame, decision_frame + decision_frames)", "label_is_first_slot_not_continuation": true, "stack_order": "oldest-first t-3,t-2,t-1,t", "stack": 4, "hud": "green channel of the score/lives/bombs bands of the newest render (t)",

# python tools/rl/v420_hist_probe.py                                  => exit 0
act_hist shape: (3300, 8, 18) uint8
hypothesis A (last 8 first_actions, front-padded) mismatches: 0 first: None
rows with duration>=2: 588
all slots[1]==slots[0] when duration>=2: False
label step_frames values: <该行为 214 个 step_frames 元组的集合，共 4058 字符，此处省略；与原件逐字节相同>
shoot_release True count: 137
bomb nonzero: 0

# python tools/rl/v420_mask_probe.py                                  => exit 0
rows: 3300
slot0 mask matches step_frames[0:2]: 3300
slot1 mask matches step_frames[2:4]: 3163
neither matched: 0
sample: (0, [65, 65, 65, 65], 65, 65, 7, 7, 2)
sample: (1, [65, 65, 65, 65], 65, 65, 7, 7, 2)
sample: (2, [65, 65, 65, 65], 65, 65, 7, 7, 2)

# python tools/rl/v420_release_probe.py                               => exit 0
mismatch rows: 137
shoot_release True rows: 137
all mismatch rows have shoot_release: True
--- row 23 decision_frame 92 slots0/1 11 12 step_frames [149, 149, 133, 132] shoot_release True flags 0 prev decision_frame 88 prev step_frames [97, 97, 97, 97]
   decode slot1 cand: (12, 0)
--- row 47 decision_frame 188 slots0/1 2 7 step_frames [145, 145, 65, 64] shoot_release True flags 1024 prev decision_frame 184 prev step_frames [145, 145, 69, 69]
   decode slot1 cand: (7, 0)
--- row 71 decision_frame 284 slots0/1 2 7 step_frames [145, 145, 65, 64] shoot_release True flags 1024 prev decision_frame 280 prev step_frames [145, 145, 69, 69]
   decode slot1 cand: (7, 0)
--- row 95 decision_frame 380 slots0/1 2 7 step_frames [145, 145, 65, 64] shoot_release True flags 1024 prev decision_frame 376 prev step_frames [145, 145, 69, 69]
   decode slot1 cand: (7, 0)
mismatch position: {'first_only': 0, 'second_only': 137, 'both': 0}
```

**一个观察（不改变归位结论，也不重新裁定契约事实）**：`slot_probe` 打印 `slots length histogram: {16: 3300}`，即该 shard 的 `chosen_plan.slots` 是 16 项；`PREFLIGHT.md:107` 的「6,600 槽」不是由这一行数字得出的（属 v4.2.0 契约口径问题）。本任务只负责“脚本可同步、行为不变”，**未**复核该口径，也未改动任何 v4.2.0 数字。归位件的这一行与原件逐字节相同，说明它不是归位引入的。

---

## 7 反向用例（fail-closed）

| 命令（cwd = 仓库根） | 退出码 | stderr 首行（真实） |
|---|---:|---|
| `python tools/rl/v420_slot_probe.py --shard <不存在的目录>` | **2** | `error: not a readable S1 shard: …\absent_shard\labels.jsonl is missing` |
| `python tools/rl/v420_hist_probe.py --shard <不存在的目录>` | **2** | 同上 |
| `python tools/rl/v420_mask_probe.py --shard <不存在的目录>` | **2** | 同上 |
| `python tools/rl/v420_release_probe.py --shard <不存在的目录>` | **2** | 同上 |
| `$env:TOUHOU_RL_ROOT='<不是仓库的目录>'; python tools/rl/v420_slot_probe.py` | **2** | `error: … is not a Touhou_RL checkout (touhou_rl/v400/gym/actions.py missing); set TOUHOU_RL_ROOT to the repository root` |
| 同上 `hist` | **2** | 同上一行（文案相同） |
| 同上 `mask` | **2** | 同上（在 import 期就 fail-closed） |
| 同上 `release` | **2** | 同上（在 import 期就 fail-closed） |
| `cd %TEMP%; python D:\WorkShop\Touhou_RL\tools\rl\v420_slot_probe.py`（默认参数、异地 CWD） | **0** | — 首行 `rows: 3300`（原脚本在这种 CWD 下会因相对路径找不到而失败；这是 §3(c) 的预期差异） |

---

## 8 not_run / 明确未做（不伪造）

- **not_run**：未在**其它** shard（S1 其它 seed / 其它 stage、`.tmp_v400_d1` canonical、dev-eval 目录）上运行这 4 个脚本——它们只对 v4.2.0 报告引用的那一个 shard 有定义好的输入结构；本任务不需要更多样本，也不拿合成数据冒充真实结论。
- **not_run**：未跑 pytest、未跑 `tools/rl/v430.py`、未跑引擎/训练/评测（任务禁止；且本任务不涉及测试改动）。`compileall` 只代表语法通过，**不等于**测试通过。
- **未归位（任务明确排除或不属于本批）**：`.tmp_v420_replay_all.sh`、`.tmp_v420_smoke_eval.sh`、`.tmp_v420_repro.sh`（报告已把命令内联，本次不归位）；`.tmp_v420_replay_probe.py`（报告正文点名的是**已保存副本** `evidence/PREFLIGHT/replay/replay_probe.py`，见 `PREFLIGHT.md:162`；原件名字只出现在该副本 docstring 第 7 行。本次实测两者 sha256 相同 = `44a218927794000900ad9b16c9c67408648684d7ba236705493b5fd3c88286af`，即正文所需脚本已在可同步路径下，无需再归位）。
- **未重跑**：v4.2.0 其它数字（T01 契约、G420-0 结论）未重新裁定；本证据只证明“脚本可运行且与原件逐字节一致”。
- **未做**：`git add` / commit / 上传；未改 `ASSET_SYNC.*`、旧证据、账本。

---

## 9 原件与资产未被修改（前后 sha256 实测）

仓库根 4 个原件（运行前 = 运行后，与 §0 记录一致）：

```text
ceff33d89616482bd91ae5dbc81fd91b47413087198d4339c1b60a68609f8427  .tmp_v420_slot_probe.py
81820648ee00f1dc51c33f5915457ad40e08bcb38437042a2b9a8ec89796329f  .tmp_v420_hist_probe.py
262fe55f7c8bd30816a7eae9b50293351b5d2d64fac0c1bb5510778eb11a738d  .tmp_v420_mask_probe.py
8239b6773e5dfec0908bd72aff6c7d0f7ecefa822452b3def09dbe9789de391b  .tmp_v420_release_probe.py
```

被读的资产文件（`Get-FileHash` 在 4 个脚本各跑一遍的**前后**取值，`before_after_equal` 全为 True）：

| 文件 | before == after | sha256 | mtime |
|---|---|---|---|
| `.tmp_v410_s1c/d1/shards/d1-s0d1-n3300-seed10-shard/labels.jsonl` | True | `011f6c1cc3db232ca296177b9afc445a4922e94b1d5513f4e5746350cff622a0` | 2026-11-13 03:33:38 |
| `…/shard.json` | True | `883f3c56f94cda60f38232d12f0a1442859d326b8f2a986b73dcfeea182254ff` | 2026-11-13 03:33:38 |
| `…/obs/act_hist.npy` | True | `44cb1aabe90b55da68022d0d3d3266e28565d0adeae186230169026f633aa7ee` | 2026-11-13 03:33:38 |

新增文件（`git check-ignore` 退出码 1 = 未被忽略；`git status --porcelain` 显示 `??`，**未 add**）：

```text
53ed7f715b340c05d282bf9b11f6764787f04fb1eb12538d7f723a1ba02c5b09  tools/rl/v420_slot_probe.py        (144 行)
c28b972ab283202c115566c56574cc062a38d8d39122581d164812d1dadc5353  tools/rl/v420_hist_probe.py        (130 行)
32d78fcb6cbdfefde9770034fe1f654f06d9ec48423e4c9ee39864276e4dd5a2  tools/rl/v420_mask_probe.py        (108 行)
749cf1f6c00dd6c586f4792d1980cdd233ca175f1958f61aa3f621916d1388bb  tools/rl/v420_release_probe.py     (122 行)
```

---

## 10 复现材料位置（仓库外，不是交付物）

- 夹具生成器：`%TEMP%\w00_probe_restore_20261115\make_synth.py`
- 合成 shard：`%TEMP%\w00_probe_restore_20261115\clean\…`、`…\bad\…`
- 每次运行的 stdout / stderr 抓取：`%TEMP%\w00_probe_restore_20261115\out\`（`orig_*` / `new_*` / `real_orig_*` / `real_new_*`）

这些文件在仓库外，可能被系统清理；本文件已把关键输出与退出码内联，可独立复核。
