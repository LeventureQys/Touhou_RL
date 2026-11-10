# EXP-P1-013 教师观测采集（D0 数据集格式 + 对齐语义 + fail-closed）

- 日期 / 切片：2026-11-09 / P3 硬阻塞（教师标签无观测引用）的采集侧切片
- 依赖：[EXP-P3-001](<EXP-P3-001-obs-spec-bc-smoke.md>)（D0 不存在的读数）、
  [obs-spec-v1](<../obs-spec-v1.md>)（冻结观测规格）、13-P3 §2、11-P1 §3/§4
- 结论先说：**`--save-observations` 已实现并单测通过**（编码一致、对齐语义、
  fail-closed、加载往返 32 用例）。**真实引擎冒烟本轮未跑成**：G1 评测/零售批跑全程占用
  WSL 引擎槽（等满 300 s 仍非空，按约定停手）。另外查清了一个更深的限制：
  **教师跑的 `shadow` 角色没有渲染器**（CPU-only null device，不建 SDL/GL），
  `AfterDrawFrame` 的 `glReadPixels` 读不到画面，所以采集侧现在会在**第一帧就 fail-closed**，
  而不是写入全 0 的假观测。真正的 D0 观测需要"shadow 出标签 + env 角色回放补像素"的配对通路
  （本轮越界，未实现）。**本卡不声称 D0 数据集已存在。**

## 1. 范围与预注册

1. 只补齐采集侧：观测编码、分片格式、对齐语义、fail-closed、单测、小冒烟。
2. 白名单：`tools/rl/v400_teacher.py`、`touhou_rl/v400/teacher/obs.py`（新增）、
   `touhou_rl/v400/student/dataset.py`（仅 teacher 模式）、`tests/test_v400_teacher_obs.py`（新增）、
   本卡、`.tmp_v400_d0_smoke/*`。
3. 未改引擎、未改协议（`READBACK`/`SHM`/`STEP` 都是既有命令）、未改依赖、未重建 build、未删除文件。
4. **不做**：DAgger 循环、env 角色回放配对、20 万决策量产、教师提速。
5. 冒烟预算 ≤60 决策；本轮实际 **0 决策**（原因见 §6），因此所有"真实读数"只有静态/单测证据，
   合成源往返明确标注 SYNTHETIC。

## 2. 交付物格式（`--save-observations DIR`）

```text
DIR/
  manifest.json                                  # v400-teacher-obs-manifest-1
  shards/<key>/                                  # key 默认 teacher-<mode>-s<stage>d<diff>t<team>-seed<seed>
    labels.jsonl                                 # 每决策一行，kind=decision，含 obs 引用
    obs/play64.npy                               # (N,4,64,64) uint8，最旧在前 t-3..t
    obs/hud.npy                                  # (N,56,160)   uint8，最新一帧的绿通道
    shard.json                                   # v400-teacher-obs-shard-1（对齐契约 + 摘要 + sha256）
```

- 写入是**流式**的：`.npy` 先用固定 512 B 头写 shape `(0,4,64,64)`，`finalize` 时原位改头
  （数据偏移不变），所以 20 万行不需要驻留内存；`labels.jsonl` 逐行追加。
- 分片先写到隐藏的 `shards/.<key>.partial/`，**只有**所有行都完整才 `rename` 成
  `shards/<key>/`；失败时留下 `.partial` + `incomplete.json`（既无 `shard.json` 也无 manifest 条目，
  `dataset.load_teacher_shard()` 直接拒绝）——不写半截分片、不覆盖已存在的分片。
- `shard.json` 关键字段：`schema`、`complete`、`obs_spec_version`、`obs_spec_digest`
  （`24f0ddac…64a381`）、`encoder=touhou_rl.v400.student.obs_spec.encode_planes`、
  `decision_count`、`frames_per_decision=4`、`first/last_decision_frame`、`stage_frames`、
  `labels{path,rows,bytes,sha256}`、`obs{play64,hud}{path,shape,dtype,bytes,sha256}`、
  `bytes_per_decision`、`alignment{…}`、`render{frames_sampled,play_mean,play_std,zero_frames}`、
  `dropped[]`。
- 标签行（与 `dataset.py` teacher 模式对齐）：`kind=decision`、`schema=v400-teacher-obs-1`、
  `decision`、`frame`/`decision_frame`、`scores[18]`、`verified_mask[18]`、`verified_count`、
  `bomb∈{0,1}`、`margin`、`chosen_plan`、`stage`、`difficulty`、
  `obs{shard,shard_dir,play,hud,row,observation_time,stack,stack_order,render_frames,
  newest_render_frame,label_frame,label_frames,p3_window_last_row,p3_label_frame,pad}`。
  `scores` 是 `TeacherLabel.scores` 原值（6 位小数），**不是** CLI JSONL 里 4 位小数的 `geo_score`。

## 3. 对齐语义（冻结，写进 `shard.json` 并在加载时复核）

| 事实 | 取值 |
|---|---|
| 教师 1 个决策提交 | `replan_decisions × DECISION_FRAMES = 2 × 2 = 4` 逻辑帧 = **2 个学生决策** |
| `scores[18]` 标注的是 | **该决策要执行**的 first action（plan slot 0，作用于帧 `[d, d+2)`），**不是下一决策**，也不是 plan 的第 2 槽（延续） |
| 观测时刻 | **决策时刻（提交动作前）** = `obs-v1.md` §3"必须是教师被查询的那个状态"；`observation_time = decision_time_before_commit` |
| 存的 4 帧 | `d-4, d-3, d-2, d-1`（`d` = 决策帧 = `FrameCount`）。`AfterDrawFrame` 在 `++g_frameCount` **之前**发布像素（`main.cpp:140-147`），所以 `STEP` 到 `d` 后共享内存里是 `d-1` 的画面 |
| HUD | 只取最新一帧（`t = d-1`），56×160 绿通道 |
| 与 P3 窗口的映射 | 令 `i = d-1`，则正是 `frames[i-3..i] -> mask[i+1]`：4 帧 = `i-3..i`，标签 = 作用在 `i+1 = d` 的动作。**教师样本就是一个 P3 窗口，不需要再移位** |
| 另一种约定（已否决） | "决策后"取像素（最新帧 `d+3`）会把**已经执行过**的动作配上它自己的结果，标签落后 1–2 个学生决策、且动作效果泄漏进输入。加载器对 `newest_render_frame + 1 != frame` / `render_frames != [d-4..d-1]` 直接报错（`test_dataset_refuses_a_stale_observation_window`） |
| episode 开头 | 按 obs-spec-v1 §2 重复最早一帧：`RESET` 完成后共享内存里的最后一帧（reset 前计数器编号）重复 4 次，`pad=3`，只在 decision 0 允许；`decision>0` 出现 `pad>0` 即报错 |
| 帧号/决策号 | 每决策 `d == 4k`；`engine.frame()` 在 `decide()` 前后都要等于 `4k`（`begin_decision` / `verify_frame_unchanged`），共享内存头帧必须等于 `STEP` 返回帧 −1 |

实现上只有一处算窗口（`obs.decision_window()`），它同时断言 `p3_label_frame == decision_frame`，
所以 off-by-one 会抛异常而不是进训练集。

## 4. fail-closed 清单（都已单测）

| 情况 | 行为 | 用例 |
|---|---|---|
| `obs_spec_digest()` 漂移 / 版本不符 | `ObservationError`，`--save-observations` 拒绝启动 | `test_frozen_gate_rejects_a_drifted_digest_or_version` |
| 共享内存无 `PIXELS_VALID`、magic 坏、尺寸非 640×480、attach 被拒 | `ObservationError` | `test_shm_source_fails_closed_on_frame_or_pixels`、`test_attach_requires_the_engine_to_confirm` |
| **整帧全 0（无渲染）** | `ObservationError`（消息写明 shadow null device 原因），分片停 | `test_capture_refuses_a_blank_readback`、`test_commit_refuses_a_blank_frame_mid_run` |
| 共享内存帧 ≠ 期望帧 | `ObservationError` | `test_shm_source_fails_closed_on_frame_or_pixels` |
| 引擎帧号 ≠ 决策号（`4k`） | `ObservationError` | `test_begin_decision_rejects_a_frame_decision_mismatch` |
| `decide()` 期间影子被移动（child 录制/验证泄漏） | `ObservationError` | `test_verify_frame_rejects_a_shadow_that_moved` |
| 缓冲区缺帧/重帧/顺序错 | `ObservationError` | `test_buffer_refuses_a_gap_or_a_short_window`、`test_buffer_rejects_a_duplicate_frame` |
| 提交中途被引擎截断（`exit=1`/`advanced<1`） | 该决策样本**丢弃**（不写行），分片在最后一个完整决策处结束 | `test_commit_reports_a_short_advance_instead_of_writing_a_row` |
| 0 决策 / 分片已存在 / 残留 `.partial` | `ObservationError`，不写 | `test_writer_refuses_to_overwrite_or_mix` |
| 加载侧：alignment 被改、`complete=false`、摘要不符、plane 被篡改、帧号与决策号不符、观测窗口过期 | `DatasetError` | `test_dataset_refuses_*`（6 个用例） |

`--save-observations` 与 `--follow-stage` / `--allow-legacy-restore` 组合被**入口直接拒绝**
（exit 3）：前者插入未观测的 stage-load 帧，后者每决策回放影子（帧不是教师被查询的那一帧）。
理由与拒绝行为有用例 `test_cli_exposes_save_observations_and_refuses_unsupported_modes`。

## 5. 逐帧读像素的代价与非侵入性

- 一个决策要 4 个**不同**的渲染帧；`STEP mask 2` 之后共享内存只剩最后一帧，所以观测模式下
  把提交拆成 4 次 `STEP mask 1`。这**不改变状态**：`HandleStep` 只设一次 `g_mask` 然后 pump 帧
  （`rlharness.cpp:654-694`），一个 slot 的 2 帧掩码相同；`advanced` 合计仍是 4，
  `advanced != 4` 的停止条件不变。有用例把拆分后的命令序列与 `step_commands(block)` 对齐比较。
- `READBACK 1` 只在提交窗口内打开（每个决策 1 次开、1 次关）：`decide()` 里 fork 的子进程
  （child 录制 48 帧 + 验证 rollout）**不继承**读回，既不付 `glReadPixels` 代价，也不会把
  子进程的帧号写进共享内存。RESET 前打开一次是为了拿到 episode 开头补帧的那一帧。
- 每帧额外 1 次 `STEP` + 1 次共享内存读（1.2 MB 拷贝 + 编码，单帧约 0.5–1.5 ms）。
  按 §6 的 4 帧/决策计，约 **3–6 ms/决策**，相对教师 500 ms 级决策 <1.5%。

## 6. 冒烟结果（本轮：未跑成，附原因与已取得的证据）

### 6.1 引擎并发

- 依赖 G1 评测与零售批跑，检查命令 `wsl -e bash -lc "pgrep -x th08-modern"`：
  06:02:52 起 5 个进程（`202715 261312 287497 287511 287515`），
  等待脚本每 5 s 轮询，**300 s 后仍为 2 个**（`292165 304749`），按约定停手。
  脚本：`.tmp_v400_d0_smoke/run_smoke.sh`（`ENGINE_BUSY_AFTER_300S`，exit 9）。
- 因此 §6.2 的真实 60 决策分片**不存在**；`.tmp_v400_d0_smoke/` 里没有真实引擎产物。

### 6.2 源码级发现：shadow 角色没有渲染器

- `main.cpp:210-244`：shadow 走 `NullDevicePreflight`（要求 `DISPLAY` 为空）+ **不建窗口**
  （"Shadow uses the software D3D null device and never creates SDL/GL"）。
- `d3d8_compat.cpp:634-648`：`window == NULL` → `g_linuxNullDevice = true`，
  "Shadow owns a CPU-only D3D slice: do not touch SDL or any GL entry point"；
  `Clear`/`Present`/`ReadBackbuffer` 全部直接返回（`:373`、`:695`、`:797`）。
- `rlharness.cpp:978-986`：像素是 `AfterDrawFrame` 里的 `glReadPixels(0,0,640,480,…)`。
  shadow 进程没有 GL 上下文 → 发布出去的只能是全 0 缓冲。
- 而 `fork.cpp:261-269` 的 `PARK` **要求 `DISPLAY` 为空**（"Forking a GL/SDL process is
  forbidden"），所以不可能用会渲染的 `env` 角色跑教师。
- 结论：**教师（shadow）自己拿不到像素**。已有实验卡的口径一致：EXP-P1-001 §运行环境
  "本次所有真实数据都来自 `env` 角色"、EXP-P0-018 "只覆盖 null-device 推演，不覆盖渲染与像素读回"、
  EXP-P0-064 "不覆盖像素渲染的 `Th08Env`"。
- 因此本轮把"像素读不到"做成了**首帧 fail-closed**（§4 第 3 行）：真跑起来时会在 `RESET` 后
  第一帧就报错退出（exit 3），留下 `.partial/incomplete.json`，不会产生任何"看起来像 D0"的分片。
  这个行为是**故意**的，不是缺陷：写入全 0 观测才是伪造。

### 6.3 已取得的加载证据（SYNTHETIC，明确标注）

引擎不可用，但"写出 60 行 → `dataset.py` teacher 模式读出"这一段不依赖引擎。用单测里的
假引擎/假共享块（**合成像素源，不是 D0**）跑 60 决策并经真实 writer/loader：

```powershell
python .tmp_v400_d0_smoke\synthetic_roundtrip.py 60    # exit 0
```

| 项 | 值 |
|---|---|
| 加载方式 | `dataset.load_teacher_samples([labels], min_samples=60, obs_root=DIR, require_observations=True)` |
| 样本数 | **60** |
| `scores` 形状 / `verified_mask` | **(18,)** / (18,) |
| obs 形状（单样本） | `play (4,64,64)` uint8、`hud (56,160)` uint8；归一化后 `float32` |
| 摘要校验 | `obs-spec version=obs-v1`、`digest=24f0ddac…64a381`，`digest_matches_frozen=true` |
| alignment / 未解析引用 | `decision_time_before_commit` / `[]` |
| manifest | 1 shard、60 decisions、240 帧、obs 1,521,684 B |
| 产物体积 | obs 1,521,684 B（60 行）+ labels 78,602 B |
| 标注 | 脚本输出首行即 `"synthetic": true` + `warning`，**不得当作 D0** |

注意：`MIN_TEACHER_SAMPLES = 50,000` 这道**训练**闸门没有放宽；60 行只能按 `min_samples=60`
加载（冒烟口径），50k 以下的"训练"仍然被 `DatasetInsufficient` 拒绝。

## 7. 规模化估算

单决策成本（实测，合成源同代码路径）：

| 项 | 字节 |
|---|---:|
| `play64` 4 帧（4×64×64） | 16,384 |
| `hud` 1 帧（56×160） | 8,960 |
| obs 小计 | **25,344**（24.75 KiB） |
| `labels.jsonl` 每行（1,310 B/行，实测 78,602/60） | ≈ 1,310 |
| 合计 | **≈ 26.65 KB/决策** |
| 分片固定开销（2 个 512 B `.npy` 头） | 1,044 B/分片 |

| 规模 | obs GB | labels GB | 合计 GB | 游戏帧 | 游戏时长 | 0.57× 实时墙钟 |
|---:|---:|---:|---:|---:|---:|---:|
| 20 万决策 | 5.069 | 0.262 | **≈5.33** | 800,000 | 13,333 s = 3.70 h | **23,392 s = 6.50 h** |
| 50 万决策（13-P3 §4 每轮上限） | 12.672 | 0.655 | ≈13.33 | 2,000,000 | 33,333 s = 9.26 h | 58,479 s = 16.24 h |
| 200 万帧（13-P3 WP1 目标）= 50 万决策 | 同上 | 同上 | 同上 | 同上 | 同上 | 同上 |

- 换算：1 决策 = 4 逻辑帧 = 4/60 s 游戏时间；`0.57×` 实时 → 每决策 0.117 s 墙钟。
- 读回开销未计入上表（§5：+3–6 ms/决策 ≈ 3–5%）；真实渲染色号下 `glReadPixels` 单帧
  0.5–1.5 ms 是 P0 实测口径，需在能做观测的角色上复测。
- 磁盘：20 万决策 ≈ 5.3 GB 可接受；50 万/轮 × 多轮要按轮归档（DAgger 最近 R 轮 + D0 20%）。

## 8. 测试

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_d0_pytest -q tests/test_v400_teacher_obs.py
# → 32 passed in 0.59 s, exit 0
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_d0_pytest -q tests/test_v400_student.py tests/test_v400_teacher_import.py
# → 27 passed in 1.82 s, exit 0（回归：teacher 模式新增未破坏既有 19 个 student 用例）
python -m compileall -q touhou_rl\v400\teacher\obs.py touhou_rl\v400\student\dataset.py tools\rl\v400_teacher.py
# → exit 0
```

测试覆盖：冻结摘要闸门（含 digest 漂移）、`decision_window` 与 P3 窗口恒等、
"决策后"约定被识别为不同窗口、`encode_planes` 逐字节（与 obs-spec 同一函数）、
`SHM` 帧闸门/像素闸门/尺寸闸门、滚动窗口（缺帧/重帧/顺序/补帧）、
流式 `.npy`（往返/中止后声明 0 行/坏行）、提交拆分与批式命令等价、
引擎帧号 ≠ 决策号、影子被移动、提交被截断、分片不覆盖、6 个加载侧拒绝、
以及 CLI 的两个模式拒绝。

## 9. 命令与退出码

| 命令 | 退出码 | 说明 |
|---|---:|---|
| `python -m pytest … tests/test_v400_teacher_obs.py` | **0** | 32 passed |
| `python -m pytest … tests/test_v400_student.py tests/test_v400_teacher_import.py` | **0** | 27 passed |
| `python -m compileall -q touhou_rl\v400\teacher\obs.py touhou_rl\v400\student\dataset.py tools\rl\v400_teacher.py` | **0** | 语法 |
| `python .tmp_v400_d0_smoke\synthetic_roundtrip.py 60` | **0** | 60 行加载（SYNTHETIC） |
| `bash .tmp_v400_d0_smoke/run_smoke.sh` | **9** | `ENGINE_BUSY_AFTER_300S`（冒烟未跑） |

真实冒烟（引擎空闲时）应执行的命令：

```bash
cd /mnt/d/WorkShop/Touhou_RL && export PYTHONPATH=/mnt/d/WorkShop/Touhou_RL
/root/v400-venv/bin/python tools/rl/v400_teacher.py \
  --decisions 60 --stage 1 --difficulty 1 --seed 1 --mode fast --quiet \
  --save-observations .tmp_v400_d0_smoke --out .tmp_v400_d0_smoke/teacher-smoke60.jsonl
# 预期：shadow 角色下 exit 3（首帧全 0 → fail-closed，.partial/incomplete.json）
# 只有在能拿到真实像素的角色/通路下才会 exit 0 并写出 shards/teacher-fast-s1d1t0-seed1/
```

## 10. 限制与下一步（按证据排序）

1. **D0 仍然不存在**：本卡只交付了格式、编码、对齐、fail-closed 和加载通路；
   没有一行真实引擎观测。任何"P3 数据已就绪"的说法都是错的。
2. **真正的阻塞已经换了一层**：从"标签没有观测引用"变成"教师所在角色不渲染"。
   可行路线只有一条：shadow 出标签 + 同一 seed/输入序列在 **env 角色**回放补像素，
   用 `HASH.hash`/`frames` 做逐检查点对齐（EXP-P0-049 已证 5000 帧零差异；
   EXP-P0-019 已证 shadow/env 同输入 167 个检查点一致）。这条通路是**新工具**，本轮未做。
3. 若走回放配对，`shard.json` 的 `alignment` 还要加一条 `pixel_source=env_replay` 与
   `hash_checkpoints`，加载器继续拒绝没有对齐证明的分片。
4. 读回成本与 0.57× 实时只有静态估算；真实"教师 + 读回"吞吐必须在能做观测的通路上实测，
   不能把本卡的 6.5 h/20 万决策当作已验证产能。
5. `MAX_TEACHER_SAMPLES` 闸门（50k）与 `MIN_TEACHER_SAMPLES` 不变：60 行只够冒烟，
   不够训练；DAgger 轮次规模（50 万/轮）的成本见 §7。
6. 观测仍是 obs-v1（无动作历史）；`obs-v2` 的决策历史契约与本卡无关，未改。

## 11. 变更与产物

- 新增 `touhou_rl/v400/teacher/obs.py`：`decision_window`、`DecisionObservationBuffer`、
  `ShmPixelSource`、`TeacherObservationRun`、`TeacherObsShardWriter`、`NpyAppendWriter`、
  `require_rendered`、`check_frozen_spec`。
- 改 `tools/rl/v400_teacher.py`：`--save-observations`、`--obs-shard-key`、模式拒绝、
  逐帧提交 + 读回、`_finish` 的 `observation` 字段。
- 改 `touhou_rl/v400/student/dataset.py`（teacher 模式）：`TEACHER_OBS_*` 常量、
  `load_teacher_shard`、`TeacherShard`、`load_teacher_manifest`、
  `load_teacher_samples(obs_root=…, require_observations=…)`、`margin=None` 容错。
- 新增 `tests/test_v400_teacher_obs.py`（32 用例）。
- `.tmp_v400_d0_smoke/`：`run_smoke.sh`（等 300 s 后 exit 9）、`probe_shadow_pixels.py`、
  `synthetic_roundtrip.py`、`synthetic-60/`（SYNTHETIC 分片）、`probe.json`（未生成）。
- 未改引擎/协议/依赖，未重建 build，未删除文件。
