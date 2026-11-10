# EXP-P0-061：WP2 / G0.3 异步读回（PBO 双缓冲）实测——读回不是杠杆，异步读回无法把全机吞吐推过 2,000 帧/s

- 日期：2026-11-09。Provider=DeepSeek，模型=DeepSeek v4.1 flash。
- 工作包：P0-WP2；验收门：G0.3「渲染 + 读回全机合计 ≥ 2,000 帧/s」（`10-P0-训练场.md` §G0 第 363 行）。
- 前序：EXP-P0-052（同步读回基线 819.66 帧/s、读回占 65–69%）、EXP-P0-003（多实例扩展曲线 1,194 帧/s 上限）。

## 0. 结论（先说，逐条可指向证据）

1. **G0.3 仍未达到。** 门定义口径（640×480 RGBA 每帧读回、llvmpipe、`LP_NUM_THREADS=1`、共享 `Xvfb :98`）下 8 实例最好一次 **816.92 帧/s = 门槛的 40.8%**，与 EXP-P0-052 的 819.66 一致（本次机器空闲，`loadavg` 0.22–7.46 且几乎全是本次自己的负载）。
2. **授权范围内实现的异步读回（`d3d8_compat.cpp` PBO 双缓冲 + `TH08_ASYNC_READBACK=1`）对稳态吞吐没有可测量影响**：相对参考二进制 1/4/8 实例分别为 **+1.7% / −3.8% / −1.8%**，全部落在本次实测的噪声带内（见 §6.3 的重复测量与"默认路径二进制更慢 5.7%"的反证）。读回占比也没有变化（66–70%）。
3. **原因是一处事实纠正：每帧读回点不在 `LinuxSurface::ReadBackbuffer()`。** 引擎进程内 `glReadPixels` 计数（§2）显示：`ReadBackbuffer()` 只在 `RESET` 初始化期被调用（约 **7.2 次/初始化帧**），稳态 `STEP` 窗口内每帧调用数为 **≈0.014 次/帧**（500 帧 → 2,000 帧只多 21 次调用）。真正的每帧读回是 `src/rlharness/rlharness.cpp::AfterDrawFrame()` 里的 **1 次/帧** `glReadPixels(0,0,640,480,…)`，而它**不在本次授权的可改文件清单内**（授权只有 `d3d8_compat.cpp`）。
4. **为了仍能回答"异步读回能不能提吞吐"，用 LD_PRELOAD 原型（不改任何源码）把同一套 PBO 双缓冲套到真正的热点读回上**：8 实例 **−3.3%**（重复 −5.6%），仍无提升。原型确实走了 PBO 路径（每实例 1,678–1,698 次 issue / 1,676–1,696 次成功 map，§7.1）。
5. **320×240 降分辨率读回（**改变观测规格**，仅作对照）同样没有提升**（1/4/8 实例 +0.4%/+0.3%/+0.2%），而且引擎自报读回耗时仍是 **5.19 ms/帧，与 640×480 完全相同** → 读回成本是**光栅化排空**，不是像素搬运量。
6. **决定性对照**：把每帧读回整个关掉（`READBACK 0`，只作诊断、不参与判门）——1 实例 135.60（+2.0%）、8 实例 **853.62（+5.8%）**。**即使删掉每帧读回也过不了门**，所以任何"读回优化"路线都不可能让 G0.3 达标。
7. **真正的杠杆是每实例光栅线程数（EXP-P0-003 已指出的旋钮，本次在"读回开启"下补测）**：`LP_NUM_THREADS=4` 时 1 实例 **359.86 帧/s（2.71×）**、8 实例 **1,096.86 帧/s（+36%）**；线程/实例小扫描最好 **1,096.86 = 门槛的 54.8%**，仍不达标。这是**运行配置**改动、不是代码改动，且必须先补像素一致性复核才能采用。
8. **结论：异步读回足以"证明读回不是瓶颈"，但不足以让 G0.3 达标。G0.3 保持 OPEN。** 下一步最小改动建议见 §10（优先级：配置重调 → 渲染侧降分辨率/真 GPU 路径 → 若三者都不可行则由负责人改门或加机器）。

## 1. 预注册

- 主指标：`READBACK 1` 时真实 harness（`--rl-role env`）在稳定窗口内推进的逻辑帧数 ÷ 真实墙钟秒数，渲染 + 每帧 `glReadPixels(0,0,640,480,GL_RGBA,GL_UNSIGNED_BYTE)` + present 全部计入（与 EXP-P0-052 完全同一口径、同一工具 `tools/rl/v400_render_throughput.py`）。
- 通过阈值：`fps ≥ 2000` 通过；`500 ≤ fps < 2000` = 继续但按实测重估预算；`< 500` = 停下问负责人（工具内 `GATE_FPS=2000`、`GATE_RANGE_LOW=500` 常量，未改）。
- 对照臂（同一工具、同一后端、同一窗口长度 8 s、同一 `--warmup-frames 60`）：
  | 臂 | 二进制 | 开关 | 读回分辨率 | 性质 |
  |---|---|---|---|---|
  | `ref-sync` | `build/rl`（SHA 591c521d…，未改动） | `TH08_PBO_MODE=sync`（穿透） | 640×480 | 基线 |
  | `pbo-sync` | `build/rl_pbo` | 不设 `TH08_ASYNC_READBACK` | 640×480 | 本次**默认路径**不变证据 |
  | `pbo-async` | `build/rl_pbo` | `TH08_ASYNC_READBACK=1` | 640×480 | **本次实现的异步读回** |
  | `hot-async` | `build/rl` | `TH08_PBO_MODE=async`（LD_PRELOAD 原型） | 640×480 | 把异步读回套到真正的热点读回 |
  | `hot-down2` | `build/rl` | `TH08_PBO_MODE=down2`（LD_PRELOAD 原型） | **320×240 上采样** | **改变观测规格**，仅对照 |
- 判定规则预先写死：任何臂只要 ≥ 2,000 即"达标候选"，但 `hot-down2` 因改观测规格**不得与 640×480 混算、不得独立判门**。
- 样本量：每格一次 8 s 窗口；N=8 的四个 640×480 臂各补一次重复（§6.3）；窗口内由工具自适应 `STEP` 分段计时。
- 负载记录：每次运行前后写 `/proc/loadavg`（`matrix.log`/`extra.log`/`threads.log`）。

## 2. 读回点归因（本次最重要的新证据）

方法：`LD_PRELOAD` 计数器（`.tmp_v400_pbo/gl_probe_shim.c` → `libgl_probe.so`）在引擎进程内拦截 `glReadPixels`、按尺寸分桶计数，退出时写入 `TH08_RB_COUNT_OUT`。引擎与 harness 都链接 libGL 并直接调用 `glReadPixels`，所以计数覆盖两个候选读回点。

| 运行（`--rl-role env`，llvmpipe） | `RESET` setup_frames | `STEP` 帧数 | 640×480 `glReadPixels` 总次数 | harness 自身次数 |
|---|---|---|---|---|
| `RESET` 后直接 `QUIT` | 176 | 0 | **1337** | 0 |
| `STEP 0 40`，`READBACK 0` | 172 | 40 | 1267 | 0 |
| `STEP 0 40`，`READBACK 1` | 172 | 40 | 1348 | **40** |
| `STEP 0 200`，`READBACK 0` | 173 | 200 | 1920¹ | 0 |
| `STEP 0 500`，`READBACK 0` | 174 | 500 | **1259** | 0 |
| `STEP 0 2000`，`READBACK 0` | 181 | 2000 | **1280** | 0 |

¹ 该点是离群值（`loadavg` 3.2–4.1，紧接线程扫描之后；初始化期渲染的帧数与时机相关，单次总次数因此可抖动 30%）。决定性的两点是 `STEP 500 → STEP 2000`：**帧数多 4 倍，调用只多 21 次** → 斜率 **0.014 次/步进帧**，即稳态读回点数 = 0（除 harness 那 1 次/帧）。

结论：
- `LinuxSurface::ReadBackbuffer()`（`d3d8_compat.cpp:238`，`FlushBackbuffer()` → `glReadPixels`）**只在 `RESET` 初始化序列里被调用**（≈1259–1337 次 / ~175 初始化帧 ≈ **7.2 次/初始化帧**），在判门的稳态窗口内**不出现**。
- 每帧唯一的读回是 `rlharness.cpp:697 AfterDrawFrame()`（`main.cpp:143` 调用）→ 这也正是 EXP-P0-052 用 `READBACK … micros=` 计数器测到的那 4.93–8.33 ms/帧：那个计数器只包住 harness 自己的 `glReadPixels` + `PublishPixels`，所以它天然看不到 D3D8 层。
- 因此"把 `LinuxSurface::ReadBackbuffer()` 改成异步"对 G0.3 的稳态吞吐**在结构上不可能有影响**，本次实测（§6）与这个结构判断一致。

## 3. 实现（授权范围内：仅 `d3d8_compat.cpp` 读回路径 + 开关）

改动文件：`/root/Touhou_RL/third_party/th08-rl/src/modern/linux/d3d8_compat.cpp`
- 补丁：`.tmp_v400_pbo/d3d8_compat.patch`（+146 / −8 行；锚点式替换，每个锚点断言命中一次）
- 前置 SHA256：`26712dd52c23aaaf72addb406a684e53d8361ceab5e17704d445e130573cd145`
- 后置 SHA256：`5f72a088bd949e063e2c98561931b6f32188d24b08797ad32b30f36e06102d3e`
- 备份：`/root/d3d8_compat.cpp.prepatch`

做法：
1. `TH08_ASYNC_READBACK` 未设 / 为 `0` / 空 → 走 `ReadBackbufferSynchronous()`，内容与改前逐字节相同（原函数体原样搬进该函数，仅多一层调用）。
2. 设为 `1` → PBO 双缓冲：`glReadPixels(0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,NULL)` 记录进 `GL_PIXEL_PACK_BUFFER`（`GL_STREAM_READ`），**不等待**；下一次 `ReadBackbuffer()` 先 `glMapBuffer` 上一帧的 PBO，把像素编码进 `LinuxSurface::pixels`（**延迟一帧**），再发起本帧的读回。
3. 对外接口、像素编码路径（`EncodePixel` / `BackbufferY`）、`pixels` 布局、`dirty` 语义均未改；唯一变化是 `pixels` 里是"上一帧"的画面。
4. GL 函数指针全部用 `SDL_GL_GetProcAddress`（`glGenBuffers`/`glBindBuffer`/`glBufferData`/`glMapBuffer`/`glUnmapBuffer`/`glDeleteBuffers`，带 `ARB` 回退），与文件里既有的 `FramebufferApi`/`g_clipControl` 同一套做法；任何一步不可用就**永久降级回同步路径**（`asyncUnavailable`），不会静默给出旧像素。
5. `glPixelStorei(GL_PACK_ALIGNMENT)` 用完恢复；PBO 用后解绑，不泄漏 GL 状态。

**未改**：协议、ABI（`LinuxSurfaceAccess`、`th08_linux_surface_access`、D3D8 接口）、游戏逻辑、`rlharness.cpp`（不在授权清单）、`build/rl`（SHA 全程校验）。

## 4. 构建（独立目录 `build/rl_pbo`，绝不触碰 `build/rl`）

```bash
# 1) 配置（参数照抄 build/rl 的 CMakeCache 关键项）
cd /root/Touhou_RL/third_party/th08-rl
cmake -S . -B build/rl_pbo \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTH08_RL_HARNESS=ON \
  -DTH08_ENABLE_REPLAY_TRACE=OFF \
  -DTH08_DX8_SDK_ROOT=/root/Touhou_RL/third_party/th08-rl/scripts/prefix/mssdk
# → CONFIGURE_EXIT=0（.tmp_v400_pbo/configure.log）

# 2) 构建
cmake --build build/rl_pbo -j 20
# → BUILD_EXIT=0（.tmp_v400_pbo/build.log，0 条 error:，76 条既有 warning，d3d8_compat.cpp 无 warning）
```

| 二进制 | SHA256 | 说明 |
|---|---|---|
| `build/rl/th08-modern` | `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb` | **要求保持不变，构建前后与全部测量后均复核一致** |
| `build/rl_pbo/th08-modern` | `296a89737aafb7b5a9a3911f75d392b0c7d289945450bf760785f0cfab6837cf` | 本次新构建 |

## 5. 测试与"默认路径不变"

### 5.1 工具/协议单测（`tests/test_v400_render_throughput.py`）

```powershell
$env:PYTHONPATH='D:\WorkShop\Touhou_RL'
python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pbo_pytest -q tests/test_v400_render_throughput.py
# → 31 passed（原 23 项 + 本次为 --child-env 新增 8 项），退出码 0
```

本次给工具加的能力（在授权清单内）：`--child-env KEY=VALUE`（可重复），把实验开关传给子进程**并写进输出 JSON 的 `extra_child_env`**，同时**拒绝**任何 `GL_KNOBS`（后端一律由 `--backend` 决定、由子进程自己的 `glGetString` 证明），所以测量文档可审计、后端保证仍然 fail-closed。

### 5.2 默认路径：`TH08_ASYNC_READBACK` 未设时像素逐字节一致

`.tmp_v400_pbo/verify_pixels.sh`（每次 `RESET` + `READBACK 1` + `STEP 0 20` + `PIXELS`）：

| 用例 | 二进制 | 开关 | `PIXELS` 回包 | `glReadPixels` 计数（shim） |
|---|---|---|---|---|
| ref | `build/rl` | — | `OK pixels=640x480 nonzero=1228800 checksum=a1eff2690fe0f694` | 1295 次，`null_ptr=0` |
| **pbo_sync** | `build/rl_pbo` | **未设** | `OK pixels=640x480 nonzero=1228800 checksum=a1eff2690fe0f694` | 1315 次，`null_ptr=0` |
| pbo_async | `build/rl_pbo` | `=1` | `OK pixels=640x480 nonzero=1228800 checksum=a1eff2690fe0f694` | 1267 次，**`null_ptr=1246`** |

- `DEFAULT_PATH_IDENTICAL=yes`：未设开关时 **640×480、nonzero=1,228,800、checksum 完全相同**。
- `pbo_async` 的 `null_ptr=1246` 证明异步分支真的被执行（1246 次以 `NULL` 指针发起的 PBO 读回），且 `glBufferData`+`map`+`unmap` 全部成功（否则会永久降级、后续调用变回非 `NULL`，计数不会继续增长到 1246）。`pbo_async` 的 `PIXELS` 与前两者同为 `a1eff2690fe0f694`，因为 `PIXELS` 是 harness 直接读当前帧缓冲，不受 D3D8 层延迟影响。
- 退出码全 0；三个进程 `QUIT` 后均正常退出（软件后端）。

## 6. 测量矩阵：同步 vs 异步（640×480 门口径）

环境：WSL Ubuntu / `Xvfb :98`（1024×768×24，共享 X server）/ `LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=1` / 每实例独立 `--data-dir` / 8 s 稳定窗口 + 60 帧预热 / 工具自行 `glGetString` 指纹命中 `llvmpipe`。全部 15 次运行 `status=measured`、退出码 0（`.tmp_v400_pbo/matrix_exit_codes.txt`）。

### 6.1 全机合计帧/s（同机、同负载窗口、交错执行）

| 臂 | N=1 | N=4 | N=8 | 读回 ms/帧 @1/4/8 | 读回占比 @1/4/8 | 每帧墙钟 ms @1/4/8 |
|---|---|---|---|---|---|---|
| `ref-sync`（基线） | 132.99 | 524.04 | **806.76** | 5.19 / 5.33 / 6.59 | 69.0% / 69.8% / 66.4% | 7.52 / 7.63 / 9.92 |
| `pbo-sync`（默认路径） | 134.14 | 506.96 | 761.15 | 5.17 / 5.41 / 6.94 | 69.3% / 68.5% / 66.0% | 7.46 / 7.89 / 10.51 |
| **`pbo-async`（本次异步）** | **135.22** | **504.32** | **792.01** | 5.12 / 5.41 / 6.69 | 69.2% / 68.2% / 66.2% | 7.40 / 7.93 / 10.10 |
| `hot-async`（热点读回异步原型） | 134.46 | 507.14 | 779.95 | 5.19 / 5.49 / 6.92 | 69.8% / 69.6% / 67.4% | 7.44 / 7.89 / 10.26 |
| `hot-down2`（320×240，**改规格**） | 133.58 | 525.84 | 808.66 | 5.19 / 5.31 / 6.56 | 69.3% / 69.8% / 66.3% | 7.49 / 7.61 / 9.89 |

### 6.2 相对基线差值

| 臂 | N=1 | N=4 | N=8 |
|---|---|---|---|
| `pbo-async` | **+1.7%** | −3.8% | **−1.8%** |
| `hot-async` | +1.1% | −3.2% | **−3.3%** |
| `hot-down2`（改规格） | +0.4% | +0.3% | +0.2% |

### 6.3 噪声带（为什么上面的差值不可当因果）

- 同一臂 N=8 重复：`ref-sync` 806.76 → **816.92**（+1.3%）；`pbo-async` 792.01 → **793.24**（+0.2%）；`hot-async` 779.95 → **761.42**（−2.4%）；`hot-down2` 808.66 → **787.80**（−2.6%）。
- 最有力的反证：**关掉开关的 `pbo-sync` 反而是最慢的一组**（N=8 761.15，比基线低 5.7%），而打开异步的 `pbo-async` 只低 1.8%。同一份源码、只差一个环境变量，却给出"关闭比开启更慢 5.7%" → **N=8 的 ±6% 是机器漂移/臂序效应，不是本次改动**。
- 三个 640×480 臂的读回占比在每一档都停在 66–70%（与 EXP-P0-052 的 65–69% 一致）：**瓶颈归因没有任何变化**，异步没有把读回时间移出关键路径，因为根本没有可移出的重叠对象。

## 7. 附加对照

### 7.1 原型臂确实走了异步路径（不是"没生效"）

`counts-hot-async-n8.txt`（8 个实例，每行一个进程）：

```
readpixels_640x480≈1652–1698  pbo_issue≈1652–1698  pbo_complete≈1650–1696  pbo_mapfail=1  primes=1
```

每实例只有 1 次 map 失败（注入点已改为"失败即补一次同步读并继续下一个 PBO"，不会让流水线退化成同步），其余全部走 PBO 双缓冲 → **异步确实生效，但吞吐不动**。

### 7.2 `READBACK 0`（只作诊断，`gate_applicable=false`，不参与判门）

| 配置 | N=1 | N=8 |
|---|---|---|
| `READBACK 1`（基线 `ref-sync`） | 132.99 | 806.76 |
| `READBACK 0` | **135.60（+2.0%）** | **853.62（+5.8%）** |

**把每帧 640×480 读回整个删掉，8 实例也只有 853.62 帧/s（门的 42.7%）。** 这一条独立地否定了"优化读回能过门"的整条假设：读回全部成本也买不回 2,000。

### 7.3 降分辨率对照（`hot-down2`，**改变观测规格**，不得混算）

- 读回 320×240 后 2× 最近邻上采样回 640×480：1/4/8 实例 133.58 / 525.84 / 808.66，相对基线 **+0.4% / +0.3% / +0.2%**。
- 关键：引擎自报读回耗时 **5.189 ms/帧，与 640×480 的 5.191 ms 几乎完全相同** → 读回耗时与像素量（1.2 MB vs 0.3 MB）无关，**全部是光栅化排空**。所以"降低读回分辨率"这条路线即使被负责人批准也拿不到任何吞吐。
- 另注：`obs_spec` 分辨率按 `12-P2-像素感知与回报.md` §WP4 到 **G2 才冻结**，当前 harness 观测就是 640×480；本条只作对照，不构成替代口径。

### 7.4 真正的杠杆：每实例光栅线程数（诊断，改变运行配置）

640×480 读回**保持开启**，只动 `LP_NUM_THREADS` 与实例数（`threads.log`/`extra.log`，退出码全 0）：

| 配置 | 实例数 | 每实例线程 | 合计帧/s | 每实例帧/s | 读回 ms/帧 | 读回占比 |
|---|---|---|---|---|---|---|
| 基线 | 1 | 1 | 132.99 | 132.99 | 5.19 | 69.0% |
| 基线 | 8 | 1 | 806.76 | 100.85 | 6.59 | 66.4% |
| **LP=4** | **1** | 4 | **359.86** | 359.86 | 1.66 | 59.6% |
| LP=4 | 4 | 4 | 834.51 | 208.63 | 2.77 | — |
| LP=4 | 5 | 4 | 939.81 | 187.96 | 3.03 | — |
| **LP=4** | **8** | 4 | **1096.86** | 137.11 | 3.97 | 54.4% |
| LP=5 | 4 | 5 | 871.86 | 217.97 | 2.56 | — |
| LP=2 | 8 | 2 | 1035.72 | 129.46 | 4.90 | — |

- 单实例放开到 4 个光栅线程 = **2.71×**，与 EXP-P0-003 记录"单实例放开线程约 400 fps、锁 1 线程 99 fps"同一量级同一方向。
- 全机最好 **1,096.86 帧/s = 门的 54.8%**；仍不达标，而且这是**运行配置**层面的结论，必须补像素一致性（多线程 llvmpipe 的确定性）复核后才能用于训练。

## 8. 是否达到 2,000 帧/s

**没有。**

| 口径 | 最好一次 | 占门槛 |
|---|---|---|
| 门口径（640×480 读回、`LP_NUM_THREADS=1`、N=8） | **816.92** | **40.8%** |
| 门口径 + 异步读回（`pbo-async`/`hot-async`） | 793.24 / 761.42 | 39.7% / 38.1% |
| 读回关掉（诊断，不判门） | 853.62 | 42.7% |
| 降分辨率读回（**改规格**，不判门） | 808.66 | 40.4% |
| 线程调优（`LP=4`，N=8，读回开启） | **1,096.86** | **54.8%** |
| 历史渲染上限（EXP-P0-003，无每帧读回） | 1,194.3 | 59.7% |

按 `10-P0` WP2 决策表：落在 **500–2,000 区间** → `continue_but_rebudget_by_measurement`（继续，但按实测重估预算）；**G0.3 硬门状态 = 未通过 / OPEN，不得写成通过**。

## 9. 瓶颈变化

- **读回的时间没有变少，只是没有变多。** 三个 640×480 臂的读回占比逐档都是 66–70%，异步没有改变关键路径结构（llvmpipe `LP_NUM_THREADS=1` 下光栅化与读回拷贝都在同一个线程上，没有可重叠的硬件引擎）。
- **瓶颈归因被纠正**：EXP-P0-052 的"每帧同步 `glReadPixels` 占 65–69%"是**真实测量**，但它测的是"光栅化排空发生在读回调用内部"这一事实，不是"读回是可优化开销"。三条独立证据支持这个重述：
  1. `READBACK 0` 只快 2.0%（N=1）/5.8%（N=8）（§7.2）；
  2. 读回字节数降到 1/4（320×240）耗时完全不变（§7.3）；
  3. 每帧读回点根本不在 D3D8 层（§2）。
- **真正的瓶颈仍是"单核单光栅线程 + 总核数"**（与 EXP-P0-003 结论一致）：`LP_NUM_THREADS` 从 1 到 4 让单实例快 2.71×，但全机合计只到 1,096.86，说明 20 核上这条软件光栅化流水线的机器级上限就在 **1.1k–1.2k 帧/s** 量级。

## 10. 结论与下一步最小改动建议

**结论：异步读回（PBO 双缓冲）实现正确、默认路径不变、可安全保留为 opt-in 开关；但它对 G0.3 没有任何帮助，G0.3 不可能靠读回优化达成。**

按"最小改动 / 最大信息量"排序的建议：

1. **（配置，零代码）把 `实例数 × LP_NUM_THREADS ≈ 核数` 作为正式口径重测**：本次 `LP=4, N=8` 得 1,096.86（54.8%），`LP=2, N=8` 得 1,035.72，`LP=4, N=5` 得 939.81。先补"多线程 llvmpipe 像素一致性"复核（同 seed 的 `PIXELS` checksum 在 `LP=1/2/4` 下是否一致；若不一致则只能固定线程数、失去该杠杆）。**这一步不需要改引擎。**
2. **（读回侧已证无用，不要再投入）** `rlharness.cpp::AfterDrawFrame()` 的异步 PBO：本次用原型实测 −3.3%（N=8），且 `READBACK 0` 只值 +5.8%。如果将来要动，也只应作为"延迟一帧可接受"的语义选择，不能当性能优化卖。
3. **（结构性，需负责人批准）降低渲染侧分辨率**：把 640×480 渲染目标/FBO 降到 320×240 再上采样，光栅化工作量按像素数下降（≈4×），这是唯一可能把 1.1k 推向 2k 的量级杠杆；但它**改变 obs 规格**（`obs_spec` 到 G2 才冻结），必须由负责人决定，且**不能与 640×480 混算**。注意：本次已证明"只降读回分辨率"无效，降的必须是**渲染**分辨率。
4. **（环境）真 GPU 路径 + 能扩展的呈现**：EXP-P0-052 已证当前 WSLg/D3D12 客户端更慢且 N≥4 后串行化，并伴随 17/17 teardown SIGSEGV；在修好之前这条路不成立。
5. **（决策，需负责人）若 1–4 之后仍 < 2,000**：按 WP2 规则重估预算（软件后端按 **800–1,100 帧/s 全机**规划，而非 2,000），或下调 G0.3 门槛——两者都必须由负责人书面决定，本卡片不擅自改门。

## 11. 变更文件

- 改（授权清单内）：`/root/Touhou_RL/third_party/th08-rl/src/modern/linux/d3d8_compat.cpp`（+146/−8：读回路径 PBO 双缓冲 + `TH08_ASYNC_READBACK` 开关 + 同步回退）。补丁 `.tmp_v400_pbo/d3d8_compat.patch`，备份 `/root/d3d8_compat.cpp.prepatch`。
- 改（授权清单内）：`tools/rl/v400_render_throughput.py`（新增 `--child-env KEY=VALUE`，实验开关进子进程并写入 `extra_child_env`；拒绝 GL 后端旋钮）。
- 改（授权清单内）：`tests/test_v400_render_throughput.py`（+8 项，共 31 passed）。
- 新增：本卡片。
- 新增（`/mnt/d/WorkShop/Touhou_RL/.tmp_v400_pbo/`，已被 `.gitignore` 的 `.tmp_*/` 覆盖）：`gl_probe_shim.c`/`libgl_probe.so`（读回计数+指纹）、`gl_async_shim.c`/`libgl_async.so`（热点读回异步/降分辨率原型）、`patch_d3d8.py`、`apply_patch.sh`、`verify_pixels.sh`、`verify_pixels.txt`、`diag.sh`、`diag2.sh`、`attribution*.txt`、`build_rl_pbo.sh`、`reconfigure.sh`、`configure.log`、`build.log`、`matrix.sh`/`matrix.log`/`matrix_exit_codes.txt`、`extra.sh`/`extra.log`/`extra_exit_codes.txt`、`threads.sh`/`threads.log`/`threads_exit_codes.txt`、`*-n*.json`、`x-*.json`、`t-*.json`、`counts-*.txt`、`summarize.py`、`scripts/`。
- **未改**：协议/ABI、游戏逻辑、`rlharness.cpp`、`build/rl`（SHA 全程 `591c521d…`）、依赖、Windows/WSL 设置。
- **未删除任何文件**，**未提交**。

## 12. 有效性自检

- **V1 主指标是真实帧/s 吗**：是。帧数取引擎 `STEP` 回包的 `advanced=`，墙钟取工具对 `STEP` 的 `perf_counter()` 计时；工具输出 `status=measured`、`backend_proof.mode=in_process_gl_renderer`、`GL_RENDERER=llvmpipe (LLVM 15.0.7, 256 bits)`（子进程自身 `glGetString`），退出码 0。
- **V2 有合成/估算冒充吗**：没有。15+8+4 次运行全部是真实 ELF 的实测；`--expect-renderer llvmpipe` 不命中即 `status=invalid`（退出码 3），本次无一例。
- **V3 重复测量**：N=8 四个 640×480 臂各重复一次（§6.3），差值 ≤2.6%；基线 N=8 两次 806.76/816.92（1.3%）。
- **V4 平凡基线**：单实例 132.99 → 8 实例 806.76（6.1×）扩展真实存在；异步没有改变这个扩展比。
- **V5 异步真的开了吗**：是。`null_ptr=1246`（引擎路径，§5.2）与 `pbo_issue≈1652–1698`/实例（原型路径，§7.1）。
- **V6 瓶颈证据可信吗**：是，用**引擎进程内**的 `glReadPixels` 计数（不是 `/proc/maps`、不是 `glxinfo`），且用 `STEP 500 vs 2000` 的斜率消除初始化期干扰（§2）。
- **V7 数据隔离**：每实例独立 `--data-dir`（工具拒绝复用已有目录、从不删除任何东西）；`th08.dat`/`thbgm.dat` 只读链接，`th08.cfg`/`score.dat` 各一份。
- **V8 指标与门一致**：门口径臂 `READBACK 1`、640×480；改规格的 `hot-down2` 与诊断的 `READBACK 0` 都**单独列表、不参与判门**。
- **V9 阈值运行前固定**：是（工具内常量，未改）。
- **V10 负载**：启动前 `loadavg` 0.22（机器空闲）；矩阵过程中 0.22→5.79、附加扫描最高 7.46，**基本都是本次自己的 8 个实例**；`matrix.log`/`extra.log`/`threads.log` 逐次记录 `load_before`/`load_after`。EXP-P0-052 当年是在 `loadavg≈10–11.5` 下测的（819.66），本次空闲下的 816.92 与它一致 → 说明**绝对帧率对本次的空闲/负载差异不敏感**，结论更稳。

## 13. 剩余阻塞 / 对项目的直接含义

1. **G0.3 硬门 OPEN**：门口径最好 816.92（40.8%），即使线程调优也只 1,096.86（54.8%），且历史渲染上限（无读回）就 1,194.3（59.7%）。**这台 20 核机器的软件 llvmpipe 流水线上限 ≈1.1k–1.2k 帧/s。**
2. **读回优化路线可以关闭**：异步（两种落地方式）、降读回分辨率、关读回——三条都没有超出 +5.8%，且 `READBACK 0` 的 +5.8% 也远不够 2,000。
3. **待负责人决策**：是否批准"渲染侧 320×240 + 上采样"改变 obs 规格；是否接受把预算口径改为 800–1,100 帧/s；是否投入真 GPU 路径（需先修 WSLg 呈现/teardown）。
4. **需要补做的复核**：多线程 llvmpipe（`LP=2/4`）下的像素一致性与确定性——若不一致，§7.4 的 1,096.86 就不能用于训练口径。
## 14. 补充：d3d8 层 PBO 分支"确实执行"的引擎自证（2026-11-09 复现轮）

§2 的读回计数只能说明"d3d8 层不是热点"，但排除不了另一种可能：`TH08_ASYNC_READBACK=1` 时
PBO 入口点解析失败、`asyncUnavailable` 置位后又静默回退同步路径——那样"无收益"就只是
"根本没跑异步"。本节补上这条自证，结论与 §0 第 2 条一致（详见
[EXP-P0-067](EXP-P0-067-WP2-async-readback-replication.md)）。

做法：在 `d3d8_compat.cpp` 加**仅诊断用** stderr 输出（只在 `TH08_ASYNC_READBACK` 置位时打印，
默认行为不变），重建到独立目录 `build/rl_pbo`（**未触碰** `build/rl`，其 SHA 仍 `591c521d…25d3eb`）：

| 构建 | SHA-256 |
|---|---|
| `build/rl_pbo/th08-modern`（含异步读回 + 诊断行） | `a3ae118159fd3fb24091513cfb236dc83e7c61b7d1334fc6b79376b5b94f2465` |
| `build/rl/th08-modern`（基线，未重建） | `591c521df161cb9e5d5a5168d88207390ae4a3eb49b2d1c653f4ad775a25d3eb` |

引擎子进程 `stderr_tail` 原样输出 `th08 async readback: PBO double buffer active`，
即 PBO 分支**真的执行了**（入口点不可用时同一位置会打印 `PBO entry points unavailable` 并回退）。
在此前提下：`pbo-sync` N=1 136.21 fps（`readback_ms` 4.933）vs `pbo-async` N=1 **137.62**（4.904），
N=8 async **734.22**，而**完全不读回** N=8 **753.73** —— 与 §0 第 2/6 条、§7.3 的
"读回成本 = 光栅化排空"一致：**PBO 分支已执行且无效**，不是未生效造成的假阴性。

口径限制：本轮 8 实例绝对值低于 §0（机器有并发负载、`--seconds` 仅 15 s、未提供 `--gl-probe-shim`，
报告 `gl_renderers=["unproven"]`），**只用于"有/无提升"的定性对照，不替换 §0 的绝对值**。
