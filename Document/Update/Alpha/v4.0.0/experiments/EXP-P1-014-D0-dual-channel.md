# EXP-P1-014 D0 双通道（shadow 出标签 + env 出像素，HASH 对齐）

- 日期 / 切片：2026-11-09 / P3 D0 数据集打通（EXP-P1-013 遗留的"教师不渲染"阻塞）
- 依赖：[EXP-P1-013](<EXP-P1-013-teacher-obs-dataset.md>)（分片格式/加载器/fail-closed）、
  [EXP-P0-049](<EXP-P0-049-WP5-determinism-5000.md>)（shadow 逐帧确定性 5000 帧）、
  [EXP-P0-019](<EXP-P0-019-WP5影子像素与分支计时.md>)（shadow/env 167 检查点一致）、
  [obs-spec-v1](<../obs-spec-v1.md>)、13-P3 §2/§4、11-P1 §3
- 结论先说：**D0 的第一份真实分片已经存在**（`.tmp_v400_d0_dual/d0/shards/dual-s0d1seed1/`，
  60 决策），**对齐 120/120 检查点全等**（60 决策 × decide 前/提交后各一次；61 个唯一边界帧
  `0,4,…,240`），`dataset.load_teacher_samples(require_observations=True)` 读出 **60 样本 /
  `scores (18,)` / `play (4,64,64)` / `hud (56,160)` / 摘要 `24f0ddac…64a381`**。
  通道 A 与 B 是**两个独立进程**（shadow 出标签、env 出像素），用逐决策 `HASH` 证明同一条轨迹；
  任一处不等即 fail-closed（exit 4，不写 `shards/`）。
- 规模结论（§8，重要修正）：双通道的额外代价**不是**"读回 +20 ms/决策"，而是**多跑一遍
  env 回放**——实测 134.8 ms/决策（0.495× 实时），所以"0.57× 实时"的标签率要按
  **A+B 串行**重算：20 万决策 ≈ **8.8 h（实测 fast 模式）/ 14.0 h（按 0.57× 标签率）**，
  **5.39 GB**；不是原估的 6.5 h。

## 1. 范围与预注册

1. 目标：只打通"shadow 标签 + env 像素 + HASH 对齐 → D0 可加载分片"这一条通路，N=60 决策冒烟；
   规模化（20 万/50 万）只给方案与复核，不在本轮跑。
2. 白名单：新增 `tools/rl/v400_d0_dual.py`、`tests/test_v400_d0_dual.py`、本卡、
   `.tmp_v400_d0_dual/*`；改 `touhou_rl/v400/teacher/obs.py`（仅 6 行：`alignment_extra` 钩子）、
   `tools/rl/v400_teacher.py`（仅 `--labels-only` 及其 HASH/逐帧掩码/6 位分数字段）。
3. 未改引擎、未改协议（`SHM`/`READBACK`/`HASH`/`STEP`/`RESET` 都是既有命令）、未改依赖、
   未重建 build、未删除文件。
4. **不做**：训练、DAgger、`obs-v2`、教师提速、像素压缩。
5. 引擎并发 ≤1：启动前 `pgrep -x th08-modern`，最多等 300 s。本轮 G1 评测占用约 5 min，
   `wait_for_engine_slot()` 轮询到槽空（3→0）后启动；A、B 顺序执行，任一时刻只有 1 个引擎。

## 2. 为什么必须是两个进程

| 事实 | 证据 |
|---|---|
| 教师必须 `--rl-role shadow`（`PARK` 要求 shadow，env 会 `ERR park-unsafe`） | `fork.cpp:261-269`；EXP-P1-013 §6.2 |
| shadow 是 CPU-only null device：不建 SDL/GL，`Clear`/`Present`/`ReadBackbuffer` 直接返回 | `main.cpp:210-244`、`d3d8_compat.cpp:634-648`、`:373/:695/:797` |
| 像素来自 `AfterDrawFrame` 的 `glReadPixels`（shadow 无 GL 上下文 → 全 0） | `rlharness.cpp:978-986` |
| env 角色能渲染（`READBACK 1` + `SHM`），但不能 `PARK` | EXP-P1-001 §运行环境、EXP-P0-019 |
| shadow 与 env 在同一 `RESET`+同一输入序列下逐帧 HASH 一致 | EXP-P0-049（5000 帧零差异）、EXP-P0-019（167 检查点） |

因此本卡把"同一轨迹"变成**可执行的闸门**：两个进程各自记录**决策边界帧**的 `HASH`，
逐点比对；不依赖"应该一样"的假设。

## 3. 通道 A（shadow，出标签）

入口：`tools/rl/v400_teacher.py --labels-only`（新增最小开关）。语义：

- 角色仍是 `TeacherEngine` 的 shadow（默认），**不 attach SHM、不开 READBACK、不读像素**；
- 每个决策前后各记一次 `HASH`：`hash_pre`（`decide()` 之前，帧 `d`）、`hash_post`
  （4 帧提交之后，帧 `d+4`）；`HASH` 是只读查询，不会移动影子；
- 记录**逐帧掩码** `step_frames[4]`（`commit_block` 展开成 4 个 `(mask,1)`，与 env 回放
  逐帧 `STEP <mask> 1` 完全同序），另保留 `step_masks`（`0x…xN` 形式）；
- 记录 `scores[18]`（`TeacherLabel.scores`，6 位小数；`geo_score` 仍是 4 位 CLI 视图）、
  `verified_mask[18]`、`chosen_plan`、`margin`、`bomb`、`frame_event_flags`、`shoot_release`；
- `--labels-only` 与 `--save-observations`/`--follow-stage`/`--allow-legacy-restore`
  **入口直接拒绝**（exit 3）：前两者会换通道，后两者会让影子移动/插入未观测帧。

本轮实测（`--mode fast --stage 0 --difficulty 1 --seed 1 --decisions 60`）：

| 项 | 值 |
|---|---|
| 决策 / 提交帧 | 60 / 240（`RESET 0 1 0 1 0`） |
| 退出码 / 墙钟 | **0** / 3.304 s（含引擎启动与 RESET） |
| 决策率 / 实时倍率 | **42.4893 决策/s** / **2.8326× 实时**（`realtime_multiple`） |
| 每决策均时（ms） | record 13.165 / search 5.296 / verify 4.181 / commit 0.597 / observe 0.126 / events 0.040 |
| 标签文件 | `labels.jsonl` 96,067 B（1,601.1 B/行），sha256 `ea9366d6…6c18581e` |
| 掩码 | 17 个不同的 4 帧掩码块；决策 0–3 都是 `0x0041×4`（shoot+left） |
| 计划 | `chosen_plan.first_action=7`、`margin=16.0`、`bomb=0`、`no_plan=0` |
| 影子是否移动 | **`hash_post(k) == hash_pre(k+1)` 对 59 个相邻边界全部成立** |

> 说明：本轮的标签来自 `--mode fast`；教师自身一致性计数
> `consistency.trajectory_mismatch=60`（几何代理与本进程验证 rollout 的轨迹差异）是**标签质量**
> 问题，不是两通道对齐问题——对齐用的是 HASH，120/120 全等。60 决策只是格式/对齐冒烟，
> 不足以评价标签质量，也不构成训练数据（`MIN_TEACHER_SAMPLES=50,000` 未放宽）。

## 4. 通道 B（env，出像素）

入口：`tools/rl/v400_d0_dual.py --phase pixels`。流程：

1. 需要私有 Xvfb：`xvfb-run -a` 重新 exec（WSLg 的 `DISPLAY=:0` 会让 env 角色在 `HELLO` 前挂住，
   与 EXP-P0-052 的捕获工具同一处理），`SDL_AUDIODRIVER=dummy`、`LIBGL_ALWAYS_SOFTWARE=1`、
   `LP_NUM_THREADS=1`；
2. `HELLO` 必须 `role=env`；`SHM /<name>`；`READBACK 1`；映射 `/dev/shm/<name>`（`SharedBlock`）；
3. 同一 `RESET 0 1 0 1 0`，要求 `ready=1`、帧计数器 0；
4. `RESET` 之后共享内存里最后一张 setup 渲染 = decision 0 的补帧（`pad=3`）；
5. 逐决策：先比对 `hash_pre`，再按 A 的 `step_frames` 逐帧 `STEP <mask> 1`，**每帧读一次像素**
   （共享内存头帧必须等于 `d+offset`），最后比对 `hash_post`；边界上的 4 帧栈
   `d-4..d-1` 由 `DecisionObservationBuffer.stack_for()` 组装（与 EXP-P1-013 同一处窗口算术）；
6. 任一检查点/帧号/像素闸门失败 → 立即停（记录 `first_mismatch`、写 `alignment.json`、
   exit 4，不写 D0）。

本轮实测：

| 项 | 值 |
|---|---|
| 决策 / 帧 | 60 / 240（`frames_read=240`） |
| 退出码 / 墙钟 | **0** / 8.087 s（含 Xvfb 启动与 RESET） |
| 读回成本 | 201.072 ms 合计，**0.838 ms/帧**（4 帧/决策 = 3.35 ms/决策） |
| 种子帧 | 共享内存 `render_frame=169`（RESET 前计数器）、`state_frame=0`、`flags=2`（`PIXELS_VALID`） |
| 非空白证据 | 241 帧 `zero_frames=0`；`play_mean=36.46`（min 5.76）、`play_std=15.60`（min 0.77）、`hud_mean=55.28`、`hud_max=255` |
| 轨迹侧开销 | 8.087 s / 60 = **134.8 ms/决策 = 0.495× 实时**（软件 GL 渲染主导，读回只占 2.5%） |

## 5. 对齐校验（本卡的核心证据）

`alignment.json`（`v400-d0-dual-alignment-1`）：

| 项 | 值 |
|---|---|
| method | `hash-checkpoint` |
| 决策 / 检查点 | 60 / **120**（每决策 `pre`+`post`） |
| 唯一边界帧 | **61**：`0,4,8,…,240` |
| 相等 / 不等 | **120 / 0** |
| first_mismatch | `null` |
| pairs sha256 | `1c264692e9eba8394cf62fed6bd2ee608d8b21fac2eb455a9e757aa2bf1ca837` |
| A 侧内部连续性 | `hash_post(k) == hash_pre(k+1)`：59/59 成立（影子在两次检查点之间未移动） |

对应每决策的两侧值示例（decision 0）：

```text
hash_pre  = 7f04c7361749ed9f  frames=0
hash_post = e3b18537a740bc49  frames=4
```

未等时的行为（已单测，并在**真实分片上做了篡改实测**）：把 `alignment.json` 里
decision 3 `post`（帧 16）的 `env` 改成 `000000000000dead` 后跑 `--phase assemble`：打印
`{"status": "ALIGNMENT_FAILED", "reason": "alignment.json contains a mismatched checkpoint"}`，
**exit 4**，`d0/` 与 `d0/shards/` 都不存在（`D0_ABSENT`/`SHARDS_ABSENT`）。
`first_mismatch{decision,boundary,frame,shadow,env}` 已写入 `alignment.json`。
失败路径不会"写一半"。

## 6. D0 分片与加载往返

```text
.tmp_v400_d0_dual/d0/
  manifest.json                                   # v400-teacher-obs-manifest-1
  dual_channel.json                              # 双通道报告 + 加载复核
  shards/dual-s0d1seed1/
    labels.jsonl                                 # 60 行，schema v400-teacher-obs-1
    obs/play64.npy                               # (60,4,64,64) uint8，983,562 B
    obs/hud.npy                                  # (60,56,160)  uint8，538,122 B
    shard.json                                   # v400-teacher-obs-shard-1
```

`shard.json` 关键字段（本轮实际值）：

| 字段 | 值 |
|---|---|
| `label_channel` / `pixel_channel` | **`"shadow"` / `"env"`** |
| `alignment_method` | **`"hash-checkpoint"`** |
| `alignment.hash_alignment` | `"hash-checkpoint"`（另有 `alignment.hash_checkpoints{method,boundaries:120,checkpoints_equal:120,first_mismatch:null,sha256}`） |
| `alignment.pixel_source` / `label_source` | `"env_replay"` / `"shadow_teacher"` |
| `engine_shadow` / `engine_env` | 同一个 sha256 `f172399f61b75baa8e62be46041ef756a710759f439e7fb07111f4747c78264e`，role 分别为 shadow/env |
| `reset_command` | `RESET 0 1 0 1 0` |
| `obs_spec_version` / `digest` | `obs-v1` / `24f0ddace59104938dbe991c0d8efbb6a8cd21dfd5a29ebcb6d023b5b364a381` |
| `decision_count` / `frames_per_decision` | 60 / 4 |
| `first/last_decision_frame` / `stage_frames` | 0 / 236 / `[0,240]` |
| `obs.play64` | shape `[60,4,64,64]`, 983,562 B, sha256 `f5b5d6d3…bb1479` |
| `obs.hud` | shape `[60,56,160]`, 538,122 B, sha256 `c5b109e3…539c4d` |
| `obs.bytes_per_decision` / `bytes_total` | **25,361** / 1,521,684 |
| `labels.sha256` | `b55d23db45b8f092be7608c1b95e8fe8c38f6760cb9f18c537cdf09b8005ec4f` |

加载（`touhou_rl/v400/student/dataset.py` teacher 模式，真实分片、非合成）：

```python
dataset.load_teacher_samples([shards/dual-s0d1seed1/labels.jsonl], min_samples=60,
                            obs_root=".tmp_v400_d0_dual/d0", require_observations=True)
```

| 项 | 值 |
|---|---|
| 样本数 | **60** |
| `scores` / `verified_mask` | **(18,)** / (18,) |
| `play` / `hud` per sample | **(4,64,64)** uint8 / **(56,160)** uint8（归一化后 float32） |
| 摘要 | `obs-v1` / `24f0ddac…64a381`（= 冻结值 `FROZEN_OBS_SPEC_DIGEST`） |
| `alignment` / `obs_unresolved` | `decision_time_before_commit` / `[]` |
| 像素真实性 | 60 行 `blank_rows=0`；play 0..254（mean 38.98）、hud 0..255（mean 56.58） |
| 帧映射 | 行 0 `render_frames=[169,169,169,169]`、`pad=3`；行 1 `[0,1,2,3]`；行 k 为 `4k-4..4k-1` |
| manifest | 1 shard、60 decisions、240 帧、1,521,684 B |

> 格式偏差（有意，已记录）：`shard.json` 的 `alignment` **必须保持 dict**（
> `load_teacher_shard` 逐键断言 `observation_time`/`stack`/`p3_window_last_row` 等）。
> 因此 `label_channel`/`pixel_channel`/`alignment_method` 写在**顶层**，
> `"hash-checkpoint"` 同时写在 `alignment.hash_alignment` 与 `alignment.hash_checkpoints.method`。
> 若把 `alignment` 直接替换成字符串，加载器会拒绝——这是契约，不是漏项。

## 7. fail-closed 清单

| 情况 | 行为 | 用例 |
|---|---|---|
| 任一 `pre`/`post` 检查点 hash 不等（或 `env` 回报的 `frames` ≠ 边界帧） | `AlignError`，写 `first_mismatch`，**exit 4**，不建 `shards/` | `test_alignment_fails_closed_and_never_writes_a_shard` |
| A 侧 `hash_post(k) != hash_pre(k+1)`（影子被移动） | `AlignError`，不写分片 | `test_shard_assembly_refuses_a_shadow_that_moved_between_decisions` |
| A 的 `step_frames` 不是 4 个 / 不是合法 HASH / `hash_pre.frames != 4k` | `DualError`，不进入回放 | `test_load_labels_rejects_*`（3 个） |
| 像素全 0 / 无 `PIXELS_VALID` / 尺寸非 640×480 / 共享块大小不符 | `require_rendered` / `DualError`，停 | EXP-P1-013 用例 + 本工具 `_read_frame` |
| `STEP advanced != 1`、帧号 ≠ 期望边界、`RESET` 不 ready、帧计数非 0 | `DualError`，停 | 代码路径（真跑时同样断言） |
| 分片已存在 / 残留 `.partial` | `ObservationError`，拒绝覆盖/混写 | `test_shard_assembly_refuses_to_overwrite_or_resume` |
| `--labels-only` 与像素/legacy/follow-stage 组合 | 入口 exit 3 | `test_labels_only_cli_refuses_the_pixel_and_legacy_modes` |
| 加载侧（既有）：摘要漂移、alignment 被改、plane 篡改、帧号/决策号不符 | `DatasetError`（EXP-P1-013 的 6 个用例，本轮回归通过） | `tests/test_v400_teacher_obs.py` |

没有"用全 0 像素冒充观测"的路径：`zero_frames=0` 是**事后证据**，事前闸门是
`require_rendered`（play/hud 必须有方差），两者都在。

## 8. 规模化方案（200k / 500k）

### 8.1 体积复核（26.65 KB → 26.96 KB）

| 项 | 每决策字节 | 本轮实测 |
|---|---:|---|
| `play64` 4×(64×64) | 16,384 | 983,562/60 = 16,392.7 |
| `hud` 56×160 | 8,960 | 538,122/60 = 8,968.7 |
| obs 小计 | 25,344 | **25,361.4** |
| `labels.jsonl` | 1,310（EXP-P1-013 口径） | **1,601.1**（+291 B：`hash_pre`/`hash_post`/`step_frames`/6 位 `scores`/`child_record`） |
| 合计 | 26.65 KB | **26.96 KB**（26,962.5 B ≈ 26.33 KiB） |

- 20 万决策：obs 5.072 GB + labels 0.320 GB = **5.39 GB**；
- 50 万决策：**13.48 GB**；每 5k 决策分片 ≈ **134.8 MB**。
- 若确认不需要 `child_record` 溯源（~1.1 KB/行），labels 可回落到 ~500 B/行，
  合计 ~25.9 KB/决策（20 万 ≈ 5.18 GB）——默认**不**裁，溯源优先。

### 8.2 实时倍率复核（原估偏乐观，原因是"少算了一整条 env 回放"）

原方案把像素成本写成"读回 ~5 ms/帧 × 4 帧/决策 = +20 ms/决策"，这是**单进程**的算法：
教师进程自己开 `READBACK` 才成立。但 shadow 没有渲染器（§2），像素只能由**另一个 env 进程**
产出，所以真实成本是 **A + B 两条独立的墙钟**：

| 口径 | 标签通道 | 像素通道 | 合计 / 决策 | 实时倍率 | 20 万决策 | 50 万决策 |
|---|---:|---:|---:|---:|---:|---:|
| 实测 `fast`（本轮） | 23.5 ms | 134.8 ms | **158.3 ms** | **0.421×** | **8.79 h** | 21.98 h |
| 预注册 0.57× 标签率 | 117.0 ms | 134.8 ms | **251.8 ms** | **0.265×** | **13.99 h** | 34.97 h |
| 假想"同进程读回"（不成立） | 117.0 ms | +20 ms | 137.0 ms | 0.487× | 7.61 h | 19.0 h |
| 假想 双引擎并行 A‖B（当前 ≤1 不允许） | — | — | max(23.5,134.8)=134.8 ms | 0.495× | 7.49 h | 18.7 h |

- 换算：1 决策 = 4 逻辑帧 = 4/60 s 游戏时间；20 万决策 = 800,000 帧 = 3.70 h 游戏时间。
- **瓶颈是通道 B**（软件 GL 渲染 30 fps ≈ 0.5× 实时），不是读回（0.838 ms/帧，占 B 的 2.5%）。
  若将来允许 2 个引擎槽，A‖B 可把 20 万压到 **7.5 h**（仍受 B 限制）；想再快只能提渲染
  （`LIBGL_ALWAYS_SOFTWARE` 线程数/`llvmpipe`）或减帧，二者都要另立实验。
- 磁盘带宽：26.96 KB/158.3 ms ≈ **170 KB/s ≈ 0.6 GB/h**，不是瓶颈。

### 8.3 分片与断点续跑

阶段产物是三个独立文件，天然可续跑（每个都能单独重跑，不需要整段重来）：

```text
work/
  labels.jsonl + channel_a.json        # A：可用 --phase labels 单独重跑
  pixels.npz + channel_b.json + alignment.json   # B：--phase pixels
  d0/shards/<key>/ + d0/manifest.json  # 组装：--phase assemble（纯 CPU，无需引擎）
```

- **分片粒度 5,000 决策**（20,000 帧 ≈ 5.6 min 游戏时间）：200k → 40 片，500k → 100 片；
  单片 ~135 MB，崩溃最多损失 5k 决策（~13 min 墙钟），引擎重启/RESET 成本被摊薄。
- **一片一个 key**：`dual-s{stage}d{difficulty}seed{seed}-e{episode}-c{chunk}`。`manifest.json`
  由 writer 合并并拒绝 obs 摘要不一致的分片（既有行为）。
- **续跑规则**（`--phase labels|pixels|assemble|all`）：
  1. `labels.jsonl` 或 `channel_a.json` 缺失 → 只跑 A；
  2. `pixels.npz` 或 `alignment.json` 缺失 → 只跑 B（会重新核对 A 的 120 个检查点）；
  3. 分片目录存在 → writer 拒绝（不覆盖）；失败的 `.partial` 留在原地作为证据，换 key 重跑；
  4. `--phase assemble` 不碰引擎，可在任意空闲时段补算（例如引擎被占用时）。
- **每片必留证据**：`alignment.json`（120/2N 检查点 + pairs sha256）、`channel_a.json`、
  `channel_b.json`、`shard.json` 的 `hash_checkpoints.sha256`。跨片复核 = 逐片比对
  `labels_sha256/play_sha256/hud_sha256` 是否与 manifest 记录一致。
- **批跑脚本**（下一轮）：外层按 (stage,difficulty,seed,episode,chunk) 列表串行调用本工具，
  每片前 `pgrep -x th08-modern`（≤1），失败即停并保留最后一片证据；不做并发。
- 50 万 → 100 片 × 135 MB = 13.48 GB；按 0.421× 实时 ≈ 22 h，按 0.265× ≈ 35 h。

## 9. 测试

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_d0dual_pytest -q tests/test_v400_d0_dual.py
# → 13 passed in 2.05 s, exit 0
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_d0dual_final -q `
  tests/test_v400_d0_dual.py tests/test_v400_teacher_obs.py tests/test_v400_student.py `
  tests/test_v400_teacher_import.py
# → 72 passed in 3.96 s, exit 0（13 新增 + 59 既有回归）
python -m compileall -q tools\rl\v400_d0_dual.py tools\rl\v400_teacher.py `
  touhou_rl\v400\teacher\obs.py tests\test_v400_d0_dual.py
# → exit 0
```

覆盖（13 用例）：`load_labels` 接受检查点记录并拒绝缺 `step_frames`/非 HASH/帧号不符（4 个）、
`_pair` 的 hash+frame 双条件、**对齐失败 fail-closed 且不写 `shards/`**、首个不等点是
`first_mismatch`、A 侧影子移动被拒、决策号→`d-4..d-1` 帧映射（含行 0 的 `pad=3`）、
真实 writer/loader 往返（3/5 样本、形状、摘要、`obs_unresolved=[]`、归一化 float32）、
`shard.json` 双通道字段 + 冻结 alignment 键仍在 + manifest 合计、分片不覆盖不续写、
`--labels-only` 的三种模式拒绝。**像素是合成数组**（测试文件首行已声明），只证明代码路径；
真实 D0 证据是 §6 的 `.tmp_v400_d0_dual/d0` 分片。

## 10. 命令与退出码

| 命令 | 退出码 | 说明 |
|---|---:|---|
| `wsl -e bash -lc '… python tools/rl/v400_d0_dual.py --decisions 60 --stage 0 --difficulty 1 --seed 1 --work-dir .tmp_v400_d0_dual --out-dir .tmp_v400_d0_dual/d0 --engine-wait 300'` | **0** | 真实双通道；A 3.304 s、B 8.087 s、assemble 并加载复核 60 样本（后台作业 `pwsh-2544`） |
| `python -m pytest … tests/test_v400_d0_dual.py` | **0** | 13 passed |
| `python -m pytest … tests/test_v400_d0_dual.py tests/test_v400_teacher_obs.py tests/test_v400_student.py tests/test_v400_teacher_import.py` | **0** | 72 passed（13 + 59） |
| `python -m compileall -q tools/rl/v400_d0_dual.py tools/rl/v400_teacher.py touhou_rl/v400/teacher/obs.py tests/test_v400_d0_dual.py` | **0** | 语法 |
| `… v400_d0_dual.py --phase assemble …`（篡改 1 个检查点） | **4** | `ALIGNMENT_FAILED`；`d0/`、`d0/shards/` 均不存在 |

运行时环境：WSL2 / `/root/v400-venv/bin/python`、`xvfb-run -a` 私有 Xvfb、
`build/rl_hazard2/th08-modern`（sha256 `f172399f…7c78264e`）。

## 11. 限制与下一步

1. **60 决策 ≠ 数据集**：`MIN_TEACHER_SAMPLES=50,000` 未放宽；本片只证明通路、格式、对齐、
   加载。训练/BC 必须等 §8.3 的批量分片。
2. **标签质量未评估**：`--mode fast` 的 `trajectory_mismatch=60`（教师自身几何代理 vs 验证
   rollout）与本轮无关但必须在量产前处理；量产应改用 `--mode full`（0.57× 口径即来自它，
   届时实时倍率按 §8.2 第二行算）。
3. **掩码多样性偏低**：60 决策只有 17 个不同的 4 帧掩码块（决策 0–3 全是 `0x0041`），
   说明 fast 模式在这段早期关卡反复选同一动作；D0 需要跨 stage/seed 的分片来覆盖。
4. **分片 `render` 汇总为空**：通道 B 的渲染统计写在 `channel_b.json`（`render`），
   writer 的 `shard["render"]` 由本工具未填（`None`）。加载/对齐不受影响；下一轮把
   `render_stats` 透传进 writer（`TeacherObservationRun` 的做法）即可。
5. **未做 20 万实跑**：§8 的 8.8/14.0 h 是"实测单位成本 × 规模"的换算，不是已验证产能；
   真实批量还要验证长时间稳定性（引擎泄漏、Xvfb、`/dev/shm` 清理）。
6. **`obs-v2`/DAgger 不在本卡**；`--labels-only` 只加字段，不改既有 `--save-observations` 语义。

## 12. 变更与产物

- 新增 `tools/rl/v400_d0_dual.py`：两通道编排（`--phase labels|pixels|assemble|all`）、
  `wait_for_engine_slot`（≤1 引擎、300 s 上限）、`load_labels`（严格检查点解析）、
  `run_pixels`（env 回放 + 逐帧像素 + HASH 比对）、`_write_alignment`、`assemble`
  （复用 `TeacherObsShardWriter`）、`verify_load`（`load_teacher_samples` 复核）、
  `prepare_data_dir`、Xvfb 重入。
- 改 `tools/rl/v400_teacher.py`：`--labels-only`、`_hash_checkpoint`、`hash_pre/hash_post`、
  `step_frames`、6 位 `scores`、模式拒绝。
- 改 `touhou_rl/v400/teacher/obs.py`：`alignment_extra` 钩子（6 行），既有 `--save-observations`
  行为与 32 个用例不变。
- 新增 `tests/test_v400_d0_dual.py`（13 用例）。
- `.tmp_v400_d0_dual/`：`labels.jsonl` + `labels_teacher.log` + `channel_a.json`、
  `pixels.npz` + `channel_b.json` + `alignment.json`、`d0/`（真实分片 + manifest +
  `dual_channel.json`）、`inspect_d0.py`/`scalars.py`/`facts.py`/`row0.py`（只读复核脚本）。
- 未改引擎/协议/依赖，未重建 build，未删除文件；`git status` 里本卡涉及的文件只有上述
  新增/修改项（`tools/rl/v400_teacher.py`、`touhou_rl/v400/teacher/obs.py` 原本就是未跟踪文件）。
