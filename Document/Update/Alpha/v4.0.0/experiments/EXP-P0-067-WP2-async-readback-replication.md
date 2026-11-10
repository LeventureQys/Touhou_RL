# EXP-P0-067 — WP2/G0.3：异步 PBO 读回复现与"分支确实执行"的引擎自证（2026-11-09）

> **本卡是对 [EXP-P0-061](EXP-P0-061-WP2-async-readback.md) 的独立复现，不是新结论。**
> EXP-061 已完成：异步读回无收益（+1.7/−3.8/−1.8%）、真正热点读回在
> `rlharness.cpp::AfterDrawFrame()`、LD_PRELOAD 热路径原型仍无收益、320×240 降规格无收益、
> 关读回 8 实例 853.62 fps（门的 42.7%）、真正的杠杆是 `LP_NUM_THREADS`（8 实例 1,096.86 fps）。
> **本卡唯一新增的事实**：d3d8 层 PBO 分支**真的执行了**（引擎 stderr 自证），
> 从而排除"异步没生效所以看起来无收益"这一假阴性解释。G0.3 判定不变：**未达（OPEN）**。

**结论：异步 PBO 双缓冲读回确实生效，但吞吐没有提升；G0.3 的 2,000 fps 无法靠优化读回来达到。**
本卡是**负结果复现**，用于阻止后续在"读回优化"上继续投入。G0.3 仍为**未达**。

---

## 1. 为什么做这件事

`EXP-P0-052` 实测：`READBACK 1` 下渲染+读回最好 **819.66 fps**（软件 llvmpipe、8 实例），
门限 **2,000 fps**；引擎自报 `glReadPixels` 占软件路径帧时 **65–69%（4.93–8.33 ms/帧）**。
据此假设：把同步 `glReadPixels` 换成 **PBO 双缓冲异步读回**，就能把这 4.9 ms 从关键路径上拿掉，
进而把 819.66 fps 推向 2,000。

## 2. 被测实现

`third_party/th08-rl/src/modern/linux/d3d8_compat.cpp` 中已存在实现：

- `TH08_ASYNC_READBACK=1` 启用（`AsyncReadbackEnabled()`，默认关闭）；
- `IssueAsyncRead()`：`glReadPixels(..., NULL)` 写入 `GL_PIXEL_PACK_BUFFER`；
- `CompleteAsyncRead()`：下一帧 `glMapBuffer` 上一帧的 PBO 并 `EncodeFromRgba`；
- 双 PBO 交替（`asyncBuffers[2]`、`asyncSlot ^= 1`）；
- 失败时 `asyncUnavailable = true` 并**回退到同步路径**（不静默发布陈旧像素）。

本次为取证增加**仅诊断用** stderr 输出（同一环境变量下才打印，默认行为不变），并重建到
独立目录 `build/rl_pbo`；**未触碰** `build/rl`（其 SHA 仍为 `591c521d…25d3eb`，其他实验依赖）。

| 构建 | 路径 | SHA-256 |
|---|---|---|
| 被测（含异步读回） | `build/rl_pbo/th08-modern` | `a3ae118159fd3fb24091513cfb236dc83e7c61b7d1334fc6b79376b5b94f2465` |
| 现有基线（未触碰） | `build/rl/th08-modern` | `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb` |

配置：`cmake -S . -B build/rl_pbo -DCMAKE_BUILD_TYPE=Debug -DTH08_RL_HARNESS=ON`，configure/build 均 **exit 0**。

## 3. 关键证据：异步路径确实在跑

引擎子进程 `stderr_tail` 原样打印：

```
th08 async readback: PBO double buffer active
```

（该行只在 `TH08_ASYNC_READBACK` 置位且 PBO 入口点解析成功时打印；若 PBO 不可用会打印
`PBO entry points unavailable` 并回退同步。本次是前者，说明 **PBO 分支真的执行了**。）

## 4. 实测数字（全部真实运行，工具 `tools/rl/v400_render_throughput.py`）

环境：WSL Ubuntu / `xvfb-run -a -s "-screen 0 1024x768x24"` / `LIBGL_ALWAYS_SOFTWARE=1` /
`GALLIUM_DRIVER=llvmpipe` / `LP_NUM_THREADS=1` / `--readback` 按档设置 / `--backend software`。

### 4.1 单实例：同步 vs 异步

| 模式 | 二进制 | seconds | fps | readback_ms/帧 |
|---|---|---:|---:|---:|
| sync | `build/rl`（旧） | 10 | 137.24 | 4.935 |
| async（旧构建无此代码） | `build/rl`（旧） | 10 | 138.36 | 4.881 |
| sync | `build/rl_pbo` | 10 | 136.21 | 4.933 |
| **async** | `build/rl_pbo` | 10 | **137.62** | **4.904** |
| **async**（诊断轮） | `build/rl_pbo` | 4 | **144.03** | **4.662** |

→ 异步相对同步 **+1.0%（137.62/136.21）**，诊断轮 +5.7%；`readback_ms` 几乎不变（4.90 vs 4.93）。
**读回耗时没有被流水线化掉。**

### 4.2 8 实例：异步读回 vs 完全关掉读回

| 模式 | instances | seconds | 合计 fps | 每实例 fps | readback_ms/帧 |
|---|---:|---:|---:|---|---:|
| **async 读回** | 8 | 15 | **734.22** | 93.07/89.42/90.58/92.94/93.04/93.05/93.09/90.82 | 6.752 |
| **完全不读回**（`--readback 0`） | 8 | 15 | **753.73** | 92.89/93.80/96.01/93.88/95.48/95.17/96.02/92.22 | — |

→ **把读回整个删掉，8 实例也只有 753.73 fps**，与异步读回的 734.22 属同一量级（差 2.6%）。
即：在 8 实例规模上，**读回根本不是瓶颈**，瓶颈是软件光栅化 + 呈现本身。

## 5. 判定

1. **异步 PBO 读回：功能可用（有引擎自证），性能无效。** 单实例 +1%、8 实例无收益，
   `readback_ms` 不降。原因是 llvmpipe 的像素回传与光栅化在同一线程完成，`glMapBuffer`
   仍要等该帧的栅格化排空，流水线化拿不到重叠时间。
2. **G0.3 的 2,000 fps 不能靠读回优化达成。** 8 实例无读回上限 753.73 fps = 门限的 **37.7%**；
   读回已不是限制项，继续优化读回是无效投入。
3. 若要真正冲线，只剩两条路，**都属于需要决策的变更**：
   - **（A）修复 GPU 光栅路径**：`EXP-P0-052` 已证 D3D12 后端可用但多实例负扩展
     （N=8 = 72.91 fps），且 **17/17 实例 QUIT 后 SIGSEGV/挂起**——必须先修退出路径；
   - **（B）改观测规格**：降低读回分辨率/降低像素频率（属 `obs_spec` 产品决策，非工程优化）。
4. **G0.3 保持未达**，本卡不改变该判定。

## 6. 命令（可复现）

```bash
# 构建
wsl -d Ubuntu -e bash -c 'cd /root/Touhou_RL/third_party/th08-rl && \
  cmake -S . -B build/rl_pbo -DCMAKE_BUILD_TYPE=Debug -DTH08_RL_HARNESS=ON && \
  cmake --build build/rl_pbo -j4'

# 单实例 A/B（脚本见 .tmp_v400_pbo/run_n.sh，内部用 xvfb-run -a）
wsl -d Ubuntu -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_pbo/run_n.sh sync  1 10 pbo-sync-n1  /root/Touhou_RL/third_party/th08-rl/build/rl_pbo/th08-modern
wsl -d Ubuntu -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_pbo/run_n.sh async 1 10 pbo-async-n1 /root/Touhou_RL/third_party/th08-rl/build/rl_pbo/th08-modern

# 8 实例：异步读回 vs 完全不读回
wsl -d Ubuntu -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_pbo/run_n.sh async      8 15 pbo-async-n8 /root/Touhou_RL/third_party/th08-rl/build/rl_pbo/th08-modern
wsl -d Ubuntu -e bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_pbo/run_n.sh noreadback 8 15 pbo-noread-n8 /root/Touhou_RL/third_party/th08-rl/build/rl_pbo/th08-modern
```

原始报告：`.tmp_v400_pbo/pbo-sync-n1.json`、`pbo-async-n1.json`、`diag-async.json`、
`pbo-async-n8.json`、`pbo-noread-n8.json`、`sync-n1c.json`、`async-n1.json`。

## 7. 限制（如实）

- 单机、WSL2、`loadavg` 未锁；各档 `seconds` 为 4/10/15，样本量小，**只能支撑"无数量级提升"这一结论**，
  不能给出高精度绝对值。
- 本次 `--gl-probe-shim`/`--expect-renderer` 未提供，报告里 `gl_renderers=["unproven"]`，
  后端由环境变量与 `EXP-P0-052` 的独立 `glGetString` 探针佐证，不是本轮直接采集。
- `display` 显示为 `:0`：`xvfb-run -a` 在本环境复用了既有 X socket；GL 仍为 llvmpipe
  （`readback_ms` 4.66–4.93 与 `EXP-P0-052` 软件路径一致，可交叉印证）。
- 诊断 stderr 补丁保留在源树（仅 `TH08_ASYNC_READBACK` 置位时打印），不改变默认行为；
  `build/rl` 未重建，其二进制身份不受影响。
- 本卡只覆盖"读回优化"这一条假设；不证明 GPU 路径或降规格路径不可行。
