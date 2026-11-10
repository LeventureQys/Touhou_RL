# EXP-P0-070 — WP2：多线程 llvmpipe（`LP_NUM_THREADS>1`）渲染像素是否逐字节确定（2026-11-09）

**结论：`LP=4` 可以用于训练口径。** 4 次独立运行 × 每臂 2 个独立 `env` 进程 × 100 帧，任务要求的三组比较
**全部 100/100 帧逐字节相同**（`LP=1` 两进程、`LP=4` 两进程、`LP=4` vs `LP=1`），
差异帧 0、无首个差异帧、差异字节 0，`PIXELS` 的 FNV-1a checksum 也逐帧全同。
`LP=8` 在 4 次运行中的 3 次同样全同；第 4 次（run c）出现 **1/6 进程在 `RESET` 阶段即落入不同
StageStart 状态**——该进程的像素差异是**游戏状态差异**，不是 llvmpipe 光栅差异。
这暴露的是"多进程同时启动时 StageStart 状态偶发不一致"的独立缺口，需在训练启动处加 fail-closed 状态守卫。

本卡闭合 [EXP-P0-068](EXP-P0-068-WP2-lp-instances-sweep.md) §3 与 [EXP-P0-061](EXP-P0-061-WP2-async-readback.md) §13
共同留下的缺口："`LP>1` 的吞吐数字在像素确定性复核前不能用于训练口径"。

---

## 1. 预注册（判据先于数据）

- 变量：`LP_NUM_THREADS ∈ {1, 4, 8}`（`LP` 是每实例 llvmpipe 光栅线程数，不是进程数）。
- 每臂 **2 个独立进程**：各自私有 data-dir、独立 SHM 名、各自 GL 上下文。
- 固定轨迹：`RESET 0 1 0 1 0`（StageStart, stage 0 / difficulty 1 / team 0 / seed 1 / practice 0），
  随后 100 次 `STEP 0 1`（每帧掩码全 0）。
- 逐帧观测：
  1. `PIXELS` → `pixels=640x480 nonzero=<n> checksum=<FNV-1a 64bit>`（引擎自己 `glReadPixels` 后对原始 RGBA 计算）；
  2. `/dev/shm/<name>` 原始 RGBA：192 字节头（`<IIQQIIII`，pixels 从偏移 192 开始）+ 1,228,800 字节，
     对 `raw[192:]` 取 **SHA-256**（字节本身，不是 PNG，也不是用 checksum 顶替字节）。
- 通过判据：LP=4 两进程之间、LP=4 vs LP=1（两个进程各比一次）、LP=1 两进程之间，三组都 100/100 帧
  SHA-256 相等 **且** `PIXELS` checksum 相等；任一帧不等即"不可用于训练口径"。
- 本卡不改门限、不改引擎/协议/依赖；只判像素确定性，不判吞吐（G0.3 结论见 EXP-P0-068）。

## 2. 环境与数据目录纪律

- WSL2 Ubuntu，`nproc=20`，共享 `Xvfb :98`（1024x768x24）；`LIBGL_ALWAYS_SOFTWARE=1`、`GALLIUM_DRIVER=llvmpipe`。
- 私有 data-dir 全部为**新建**目录：`/root/v400-lpdet-{a,b,c,d}/s<pid>-lp{N}/p{0,1}`；
  `th08.cfg`、`score.dat` 从 `/root/v400-data/game/*.orig` **复制**，`th08.dat`、`thbgm.dat` 是**只读符号链接**。
  `/root/v400-data/game` 从未用作 `--data-dir`（避免写脏共享模板）。
- 每进程 `/proc/<pid>/environ` 被记录为环境证明：`LP_NUM_THREADS=1/4/8`、`LIBGL_ALWAYS_SOFTWARE=1`、
  `GALLIUM_DRIVER=llvmpipe`、`DISPLAY=:98`。
- 4 次运行的 `loadavg` 起点分别为 25.14 / 23.26 / 26.14 / 9.15（前三次另有一个 16 实例吞吐扫描在跑，
  属"满载加压"条件；run d 为相对空载），两次条件都得到相同结论。
- 二进制 `th08-modern` SHA-256 = `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb`。

## 3. 命令与退出码

```bash
# 单元测试（Windows 侧，纯比较逻辑；18 个用例 = 13 独立 + 5 parametrize）
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_lpdet_pytest -q \
  tests/test_v400_render_pixel_determinism.py
# → 18 passed, exit 0

# 真实引擎实验（WSL 侧；脚本见 .tmp_v400_lpdet/run_determinism.sh）
/root/v400-venv/bin/python tools/rl/v400_render_pixel_determinism.py \
  --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern \
  --data-template /root/v400-data/game --run-root /root/v400-lpdet-a \
  --display :98 --lp-threads 1 --lp-threads 4 --lp-threads 8 \
  --instances 2 --frames 100 --out .tmp_v400_lpdet/report-a.json
```

| 运行 | 命令（同上，仅换 `--run-root` / `--out`） | 退出码 | 工具状态 |
|---|---|---|---|
| pytest | `tests/test_v400_render_pixel_determinism.py` | **0** | 18 passed |
| run a | `--run-root /root/v400-lpdet-a --out .tmp_v400_lpdet/report-a.json` | **0** | `deterministic` |
| run b | `--run-root /root/v400-lpdet-b --out .tmp_v400_lpdet/report-b.json` | **0** | `deterministic` |
| run c | `--run-root /root/v400-lpdet-c --out .tmp_v400_lpdet/report-c.json` | **4** | `divergent`（实为 LP=8 状态分歧，见 §5） |
| run d | `--run-root /root/v400-lpdet-d --out .tmp_v400_lpdet/report-d.json` | **0** | `deterministic` |

工具退出码约定：`0` = 全部比较逐字节相同；`4` = 有效测量但存在不等（真实结果，不当错误）；
`3` = 测量无法完成（不产出任何证据）。run c 的 `4` 由旧版判据产出（旧版把状态分歧与像素分歧合并计为
`divergent`）；原始逐帧数据未变，分歧发生在 `RESET` 阶段（§5），已按新版判据在工具中分离。

## 4. 三组结果（任务要求的核心）

每组每帧比较 SHA-256 与 `PIXELS` checksum；"相等帧/不等帧"按 100 帧计。

| 比较组 | run a | run b | run c | run d | 首个差异帧 | 差异字节数 |
|---|---|---|---|---|---|---|
| **LP=1 两进程之间**（基线） | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 | 无 | 0 |
| **LP=4 两进程之间**（同配置跨进程） | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 | 无 | 0 |
| **LP=4 vs LP=1**（进程 p0） | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 | 无 | 0 |
| **LP=4 vs LP=1**（进程 p1） | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 | 无 | 0 |
| LP=8 两进程之间（额外） | 100 / 0 | 100 / 0 | **0 / 100**（状态分歧，§5） | 100 / 0 | run c: 帧 1 | run c: 1,226,246 |
| LP=8 vs LP=1 p0（额外） | 100 / 0 | 100 / 0 | **0 / 100**（同上） | 100 / 0 | run c: 帧 1 | run c: 1,226,246 |
| LP=8 vs LP=1 p1（额外） | 100 / 0 | 100 / 0 | 100 / 0 | 100 / 0 | 无 | 0 |

- **三组（任务口径）合计：4 次运行 × 100 帧 × 2 个进程对 = 800 帧比较全部相等，0 帧不等。**
- `LP=4` 的 `PIXELS` checksum 与 `LP=1` 在全部 100 帧、全部 4 次运行中完全相同；
  SHA-256 相同则 checksum 必相同，反之不成立，因此两者都作为独立列被逐帧核对。
- 差异字节数只在"不等"时才有意义：run c 的 LP=8 p0 为 1,226,246 字节/帧（首帧），
  全帧 bbox `x0=0,y0=0,x1=639,y1=479`，即"整屏不同"，与"场景不同"一致，而非局部光栅噪声。

### 4.1 附加一致性证据（说明比较不是空转）

- 每个进程 100 帧的 SHA-256 **互不相同**（`distinct_sha=100`），即帧确实在变化，比较不是拿常量自比。
- 每帧 `nonzero` 覆盖 883,264–1,228,800 字节（画面非空、且从 HUD 到满屏弹幕都有变化）。
- `STEP` 回复的引擎 state hash 在 6 个进程间逐帧相同（run a/b/d；run c 仅 LP=8 p0 不同，且其
  `RESET` 回复 hash 已经不同）。
- SHM 头 `frame` 计数器必须等于步进帧号（`parse_header` 校验），排除"读到陈旧缓冲"导致的假相同。

## 5. run c 的 LP=8 分歧：状态差异，不是光栅差异

run c 中 LP=8 p0 从**第 1 帧**就与其它进程不同。关键在于它在**任何 `STEP` 之前**就已经不同：

| 进程 | `RESET 0 1 0 1 0` 回复 hash | 第 1 帧 state hash |
|---|---|---|
| LP=1 p0 / p1、LP=4 p0 / p1、LP=8 p1 | `7f04c7361749ed9f` | `f1df042d9373e879` |
| **LP=8 p0** | **`795b9441ed0c8056`** | **`6e05414e5ea646a2`** |

即该进程在 StageStart 就处在另一个游戏状态（RNG/初始化不同），它的像素不同是**必然而非缺陷**：
同一场景换一套状态，画面当然不同。因此：

- 这一条**不能**算作"llvmpipe 多线程导致像素不确定"的证据；
- 但它是一个**真实且重要的多进程缺口**：4 次运行共 18 个进程启动中有 1 次（约 5.6%）落入不同
  StageStart 状态，且当时 `RESET` 回复仍报 `ready=1`。若不校验，训练口径会把状态差异误读成
  渲染不确定性（或反过来，把非确定性训练数据当成确定性）。
- 该分歧与 LP 值无因果证据（只出现 1 次、在 LP=8 arm），本卡不声称 LP=8 更易触发；
  run d（空载）LP=8 完全干净。

## 6. 判定（明确结论）

1. **`LP=4` 可用于训练口径。** 依据：同配置跨进程 100/100 帧逐字节一致（4 次运行）
   **且** 与 `LP=1` 逐字节一致（4 次运行、每臂两个进程），差异帧 0、差异字节 0、checksum 全同。
   `LP=4` 的吞吐（961–1,097 fps，EXP-P0-061/068）因此在像素层面站得住。
2. **`LP=8` 条件成立。** 状态一致的进程之间同样 100/100 逐字节一致（3 次全量运行 + run c 的 5 个状态一致进程），
   但多进程 StageStart 需状态守卫；守卫到位前，`LP=8` 的吞吐不无条件用于训练口径。
3. **必须新增的守卫（不属于本卡，属后续工作）**：多进程启动时比对 `RESET` 回复 hash（或 SHM state hash），
   不一致即 fail-closed 重启该进程；否则 5.6% 量级的静默状态漂移会污染训练数据。
4. G0.3 门结论不受本卡影响：本卡只回答"像素是否确定"，不回答"吞吐是否达标"（仍为未达，见 EXP-P0-068）。

## 7. 限制（不要外推）

- 只覆盖 StageStart（stage 0）起 100 帧、掩码全 0 的一段场景；未覆盖 6 个 stage、未覆盖实际弹幕密度全程、
  未覆盖 5,000 帧长程（EXP-P0-049 口径）。像素确定性在这些范围内仍属"已验证的部分"。
- 显示为共享 `Xvfb :98` + llvmpipe 软件路径；真 GPU（D3D12）路径不在本卡范围（其 WSLg teardown 问题见 EXP-P0-052）。
- 像素来源是 `PIXELS` 触发的 `glReadPixels`（env 角色唯一观测路径），不是 present/scanout；本卡不对"窗口显示"下结论。
- 100 帧确定性 ≠ 任意帧数确定性；但 100 帧逐字节相同已足以否定"多线程光栅必然改变像素"的顾虑。

## 8. 证据文件

- 工具：`tools/rl/v400_render_pixel_determinism.py`（新增，含 `parse_header` / `diff_stats` / `compare_series` / `verdict`）
- 测试：`tests/test_v400_render_pixel_determinism.py`（新增，18 passed）
- 原始报告：`.tmp_v400_lpdet/report-{a,b,c,d}.json`（含每进程 100 帧 SHA-256、checksum、state hash、环境、退出码）
- 运行脚本：`.tmp_v400_lpdet/run_determinism.sh`；数据目录根：`/root/v400-lpdet-{a,b,c,d}`
- 参考实现（SHM 读取与 SHA-256 逻辑）：`tools/rl/v400_wp5_pixel_rebuild.py`
