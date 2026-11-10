# EXP-P0-052：WP2 / G0.3 渲染+读回吞吐复测——软件 GL 与 WSLg D3D12 硬件后端的真实对照

- 日期：2026-11-08。Provider=DeepSeek，模型=DeepSeek v4.1 flash。
- 工作包：P0-WP2；验收门：G0.3「渲染 + 读回全机合计 ≥ 2,000 帧/s」（`10-P0-训练场.md` §WP2 决策表、§G0）。
- 结论：**G0.3 硬门仍未达到，且硬件 GL 后端不是提速路径**。
  - 单实例「渲染 + 读回」（640×480 RGBA，`READBACK 1`）实测：**软件 GL（llvmpipe）139.26 帧/s**，**硬件 GL（WSLg D3D12 / RTX 4070 SUPER）108.25 帧/s**（同窗口、同数据、同掩码）；单实例复测区间分别为 78.34–139.26 帧/s 与 45.72–108.25 帧/s（受外来负载影响，见 §6 第 5 条与 §8 第 4 条）。
  - 全机合计最好一次是**软件 GL 8 实例 819.66 帧/s**，为门槛的 **41.0%**；硬件 GL **完全不扩展**（N=4 → 142.23，N=8 → **72.91** 帧/s，比单实例还低）。
  - 因此 `31-决策记录` 里「WSLg 的 GL 是 D3D12」这一条**今天复现为真**（客户端上下文确实是 D3D12），但**它不是吞吐杠杆**：D3D12 路径的每帧固定开销（present/提交+同步读回）比 llvmpipe 高，且多实例时被 WSLg 的呈现路径串行化。
  - 按 `10-P0` WP2 决策表，实测落在 **500–2,000 帧/s 区间** → **继续，但 P3/P4 预算必须按实测重估**，不得按 2,000 帧/s 规划；同时该硬门保持 **OPEN**，不得写成通过。

## 1. 预注册

- 主指标：`READBACK 1` 时，真实 harness（`--rl-role env`）在稳定窗口内**实际推进的逻辑帧数 ÷ 该窗口的真实墙钟秒数**（渲染 + 每帧 `glReadPixels(0,0,640,480,GL_RGBA,GL_UNSIGNED_BYTE)` + present 全部包含在内）。
- 通过阈值：全机合计 **≥ 2,000 帧/s** 为通过；500–2,000 为「继续但按实测重估预算」；< 500 为「停下问负责人」。**本卡片同时记录单实例与多实例，两者都照实报，不择优发布。**
- 样本量：每个配置一个 10 s 稳定窗口（最终复核为 8 s），窗口内按 `STEP` 分段计时；另做 1 次 `READBACK 0` 对照（仅诊断，不参与判门）。
- 判定规则（预先写死，见 `tools/rl/v400_render_throughput.py::gate_verdict`）：`fps ≥ 2000 → continue_as_planned`；`500 ≤ fps < 2000 → continue_but_rebudget_by_measurement`；`fps < 500 → stop_and_ask_owner`；`readback=0 → 不适用`。

## 2. GL 后端探测（原始命令 + 退出码）

全部在 WSL（`Ubuntu`，内核 `6.18.54.1-microsoft-standard-WSL2`，`nproc=20`）内执行；脚本 `.tmp_v400_render_tp/evidence.sh`、`.tmp_v400_render_tp/evidence_fix.sh` 的完整输出落在 `.tmp_v400_render_tp/gl_backend_evidence.txt`。

| 编号 | 命令 | 退出码 | 关键输出 |
|---|---|---|---|
| E1 | `ls -la /dev/dri` | 0 | `card0`(226,0)、`renderD128`(226,128) 存在 |
| E2 | `cat /sys/class/drm/card0/device/uevent` | 0 | `DRIVER=faux_driver`（WSL 虚拟 GPU 节点） |
| E3 | `ls /usr/lib/x86_64-linux-gnu/dri/` | 0 | 17 个驱动，含 `d3d12_dri.so`、`swrast_dri.so`、`zink_dri.so` |
| E4 | `stat -c '%i %h %n' …/swrast_dri.so …/d3d12_dri.so …/zink_dri.so` | 0 | **三个名字同一个 inode `48557`、链接数 13** |
| E5 | `DISPLAY=:0 glxinfo -B` | 0 | `direct rendering: Yes`；`Device: D3D12 (NVIDIA GeForce RTX 4070 SUPER)`；`Accelerated: yes`；GL 4.2 compat |
| E6 | `DISPLAY=:98 glxinfo -B`（我自己起的 `Xvfb :98`，不强制软件） | 0 | 同样报 `D3D12 (NVIDIA GeForce RTX 4070 SUPER)`、`Accelerated: yes` |
| E7 | `DISPLAY=:98 LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=1 glxinfo -B` | 0 | `Device: llvmpipe (LLVM 15.0.7, 256 bits)`；`Accelerated: no` |
| E8 | `DISPLAY=:0 strace -f -e trace=openat glxinfo -B` | 0 | **只 openat `dri/swrast_dri.so`（1 次），却报 D3D12** |
| E9 | 引擎进程指纹，`DISPLAY=:0`，不设任何强制软件变量（见 §2.1） | 124（QUIT 后不退出，被 `timeout` 杀） | `GL_VENDOR=Microsoft Corporation`、`GL_RENDERER=D3D12 (NVIDIA GeForce RTX 4070 SUPER)`、GL 4.2、GLSL 4.20 |
| E10 | 引擎进程指纹，`DISPLAY=:98` + llvmpipe 三变量 | 0 | `GL_VENDOR=Mesa`、`GL_RENDERER=llvmpipe (LLVM 15.0.7, 256 bits)`、GL 4.5、GLSL 4.50 |
| E11 | `/usr/lib/wsl/lib/nvidia-smi -L` / `--query-gpu=name,utilization.gpu,memory.used` | 0 / 0 | `GPU 0: NVIDIA GeForce RTX 4070 SUPER`；空闲时 `1 %, 1865 MiB` |
| E12 | `nvidia-smi --query-gpu=driver_version,name` | 0 | `617.42, NVIDIA GeForce RTX 4070 SUPER` |
| E13 | `cat /proc/loadavg; nproc` | 0 | 最终复核时 `9.34 9.96 7.78`、`20` |

### 2.1 判定：存在硬件渲染路径，而且引擎真的用上了它

`glxinfo -B` 的 `D3D12` 字符串**不能**单独作为后端证据：E8 显示 `glxinfo` 进程只打开了 `dri/swrast_dri.so`，却仍打印 `OpenGL renderer string: D3D12 (NVIDIA GeForce RTX 4070 SUPER)`。原因有两层：

1. E4 证明这个 Mesa 构建把全部 gallium 驱动**硬链接成同一个文件**（`swrast_dri.so`/`d3d12_dri.so`/`zink_dri.so` 同 inode、链接数 13），所以「打开的文件名」与「实际驱动」不是一回事；`/proc/<pid>/maps` 也因此**不能**当作后端证据（本卡片最初据此误判为软件回退，后被 E9/E10 纠正）。
2. `GLX_MESA_query_renderer` 的信息由服务端/加载器给出，与客户端真正的上下文不一定一致。

可靠证据只能取**引擎进程自己的 `glGetString`**。用一个不改进程镜像的 `LD_PRELOAD` 探针（源码 `.tmp_v400_render_tp/gl_probe_shim.c`，`gcc -O2 -shared -fPIC -o /tmp/libgl_probe.so … -lGL -ldl`，退出码 0），在引擎第一次 `glReadPixels` 时记录指纹：

```powershell
wsl.exe -d Ubuntu -- bash -lc 'printf "HELLO\nRESET 0 1 0 11 0\nREADBACK 1\nSTEP 0 20\nREADBACK\nQUIT\n" | env -u LIBGL_ALWAYS_SOFTWARE -u GALLIUM_DRIVER -u LP_NUM_THREADS DISPLAY=:0 SDL_AUDIODRIVER=dummy LD_PRELOAD=/tmp/libgl_probe.so TH08_GL_PROBE_OUT=/tmp/fp_e9.txt timeout 60 /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-dir /root/v400-data/game --rl-control 0,1 --rl-role env'
```

结论：**硬件路径可用**（E9 指纹是 D3D12/RTX 4070 SUPER），`31-决策记录` 关于「WSLg 的 GL 实际是 D3D12」的记录今天复现为真。但**它慢**，见 §4/§5。

### 2.2 附带发现：硬件后端的退出路径是坏的

E9 的退出码 **124**（`QUIT` 后 60 s 内不退出，被 `timeout` 杀），没有强制软件的 E10 是 **0**。在本次全部 **17 个**硬件实例进程里，**17/17 都在 `QUIT` 之后以 SIGSEGV 结束**（`exit_code=-11`，见 §4 的退出码列），而软件后端的 **47** 个实例里 **45 个 `exit_code=0`**，另有 2 个在 12/16 实例超订载下也以 `139` 结束（`r2-a12` 的 inst=9、`r2-a16` 的 inst=1，两者在崩溃前已分别推进 437/380 帧，计入合计）。这是 WSLg/D3D12 路径上真实存在的缺陷，本次不修（禁止改引擎），但必须记录：**它是硬件后端无法用于长时间生产运行的阻塞项之一**。

## 3. 测量方法与真实命令

工具：`tools/rl/v400_render_throughput.py`（本次新增，允许清单内）。

它只做一件事：把真实 harness 跑起来，按真实墙钟数帧，并记录引擎自报的 `glReadPixels` 耗时与**进程内** GL 指纹。关键实现点：

- 启动 `th08-modern --data-dir <每实例独立目录> --rl-control 0,1 --rl-role env`（env 是唯一会渲染并读回像素的 role，`HELLO` 回包 `role=env` 被断言）。
- `READBACK 1` 打开后，用 `STEP 0 <chunk>` 推进；`chunk` 会自适应（目标每次 `STEP` 约 0.25 s），避免慢实例把共享窗口从 10 s 拖到 14 s。
- 每实例独立 `--data-dir`（`th08.dat`/`thbgm.dat` 只读链接、`th08.cfg`/`score.dat` 各自一份），工具**拒绝复用已有实例目录**，从不删除任何东西。
- 计时：`frames / Σ(STEP 命令墙钟)` 为单实例帧率；全机合计 = 各实例帧数之和 ÷ **同一段共享窗口**（所有实例经 `threading.Barrier` 对齐后开始、取 `max(end) - min(start)`），另报 `instance_fps_sum` 与 `parallel_efficiency`。
- 读回成本：对引擎 `READBACK` 命令的 `frames=`/`micros=` 计数器做差分，得到 **Δmicros / Δframes**，即引擎自己测的 `glReadPixels` 每帧耗时（不含 IPC、不含 Python）。
- 后端证明：`--gl-probe-shim /tmp/libgl_probe.so --expect-renderer <llvmpipe|D3D12>`，指纹必须命中，否则 `status=invalid`、退出码 3。env 侧另有 `/proc/<pid>/environ` 复核（硬件后端里不得存在 `LIBGL_ALWAYS_SOFTWARE`）。

单实例最终复核（每个后端一次，串行，先记录 load）：

```powershell
wsl.exe -d Ubuntu -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_render_tp/final_runs.sh
```

脚本内实际执行的工具命令（软件后端；硬件后端只改 `--backend/--display/--expect-renderer/--run-root`）：

```powershell
wsl.exe -d Ubuntu -- bash -lc 'cd /mnt/d/WorkShop/Touhou_RL && PYTHONPATH=/mnt/d/WorkShop/Touhou_RL /root/v400-venv/bin/python tools/rl/v400_render_throughput.py --binary /root/Touhou_RL/third_party/th08-rl/build/rl/th08-modern --data-template /root/v400-data/game --run-root /root/v400-render-tp-fin-sw --backend software --display :98 --instances 1 --seconds 8 --warmup-frames 60 --chunk 32 --chunk-seconds 0.25 --readback 1 --label final-sw --gl-probe-shim /tmp/libgl_probe.so --expect-renderer llvmpipe --out /mnt/d/WorkShop/Touhou_RL/.tmp_v400_render_tp/final-sw.json'
```

多实例矩阵（N=1/4/8，第 1 轮）：

```powershell
wsl.exe -d Ubuntu -- bash /mnt/d/WorkShop/Touhou_RL/.tmp_v400_render_tp/matrix_r1.sh
```

退出码：`matrix_r1.sh` 外层 0；8 个 case 的工具退出码全部 **0**（`CASE_EXIT_a1/a4/a8/b1/b4/b8/a1nr/b1nr=0`，记录在 `.tmp_v400_render_tp/r1_exit_codes.txt`）。`final_runs.sh` 外层 0，`FINAL_EXIT_sw=0`、`FINAL_EXIT_hw=0`。第 2 轮矩阵（`matrix_r2.sh`）在跑到 `b1_rep2` 之后**被负责人要求收缩、主动中止**（剩余 `b8_rep2`/`c1`/`c8`/`c16` 未跑），已完成的 `a1_rep2`/`a1_rep3`/`a1nr_rep2`/`a12`/`a16`/`b1_rep2` 结果照实列在 §4，不当作完整矩阵。

## 4. 实测结果

### 4.1 后端 A：软件 GL（`LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=1`，`Xvfb :98`）

`READBACK 1`，每帧读回 640×480 RGBA：

| 配置 | 实例数 | 全机合计 帧/s | 每实例 帧/s | 引擎自报 `glReadPixels` ms/帧 | 每帧墙钟 ms | 读回占比 | GPU util 峰值（系统级，仅参考） | 进程退出码 |
|---|---|---|---|---|---|---|---|---|
| r1-a1 | 1 | 139.26 | 139.26 | 4.93 | 7.18 | 68.6% | 25% | 0 |
| r2-a1_rep2 | 1 | 81.19 | 81.19 | 8.00 | 12.32 | 64.9% | 52% | 0 |
| r2-a1_rep3 | 1 | 78.34 | 78.34 | 8.33 | 12.76 | 65.3% | 45% | 0 |
| final-sw（8 s，load 11.5） | 1 | 106.89 | 106.89 | 6.07 | 9.36 | 64.9% | 24% | 0 |
| r1-a4 | 4 | **502.97** | 125.74 | 5.39 | 7.96 | 67.7% | 26% | 0,0,0,0 |
| r1-a8 | 8 | **819.66** | 102.46 | 6.46 | 9.77 | 66.1% | 23% | 0×8 |
| r2-a12 | 12 | 552.66 | 46.06 | 14.40 | 21.71 | 66.3% | 57% | 0×12 |
| r2-a16 | 16 | 618.88 | 38.68 | 17.04 | 25.85 | 65.9% | 54% | 0×16 |

（`r2-a12`/`r2-a16` 在机器已有外来负载时测得，单实例同期只有 78–81 帧/s，故这两个点不能与 `r1-a8` 直接连成曲线。）

### 4.2 后端 B：硬件 GL（不设任何强制软件变量，`DISPLAY=:0` = WSLg）

| 配置 | 实例数 | 全机合计 帧/s | 每实例 帧/s | 引擎自报 `glReadPixels` ms/帧 | 每帧墙钟 ms | 读回占比 | GPU util 峰值 | 进程退出码 |
|---|---|---|---|---|---|---|---|---|
| r1-b1 | 1 | 108.25 | 108.25 | 1.18 | 9.24 | 12.8% | 66% | **-11** |
| r2-b1_rep2 | 1 | 51.43 | 51.43 | 2.10 | 19.44 | 10.8% | 79% | **-11** |
| final-hw（8 s，load 10.9） | 1 | 45.72 | 45.72 | 3.06 | 21.87 | 14.0% | 77% | **-11** |
| r1-b4 | 4 | 142.23 | 35.56 | 1.91 | 28.12 | 6.8% | 74% | **-11 ×4** |
| r1-b8 | 8 | **72.91** | 9.11 | 2.61 | 109.8 | 2.4% | 57% | **-11 ×8** |

三个配置的 `GL_RENDERER` 指纹都是 `D3D12 (NVIDIA GeForce RTX 4070 SUPER)`（`backend_proof.mode=in_process_gl_renderer`），不是软件回退。

### 4.3 诊断对照：`READBACK 0`（**不参与判门**）

| 配置 | 后端 | 实例数 | 全机合计 帧/s |
|---|---|---|---|
| r1-a1nr | 软件 | 1 | 96.38 |
| r2-a1nr_rep2 | 软件 | 1 | 83.46 |
| r1-b1nr | 硬件 | 1 | 34.41 |

**关掉每帧读回并没有更快**（软件 96.4/83.5 对 139.3；硬件 34.4 对 108.3）。因此：

1. 软件路径约 100 帧/s 的墙上限**不是** `glReadPixels` 这一个调用造成的，而是整条「光栅化 + FBO 解析 + present」流水线；
2. **不能用 A/B 相减来反推读回成本**；唯一可信的读回成本是引擎自己的 `READBACK … micros=` 计数器（§4.1/§4.2 的「引擎自报 ms/帧」）。

### 4.4 隔离对照：同一份最小 GLX 程序，两种后端

`.tmp_v400_render_tp/glx_probe.c`（`gcc -O2 -o /tmp/glx_probe … -lGL -lX11`，退出码 0）用与引擎相同的上下文属性（`GLX_RGBA`、`GLX_DOUBLEBUFFER`、`GLX_DEPTH_SIZE 16`、直接渲染）做 `clear + swap + glReadPixels(640×480 RGBA)`，3 s：

| 后端 | 指纹 | 帧/s | `glReadPixels` ms/帧 | 首像素 |
|---|---|---|---|---|
| 不强制（`DISPLAY=:0`） | `D3D12 (NVIDIA GeForce RTX 4070 SUPER)` | **391.16** | 1.1265 | 25,51,76,255 |
| 不强制（`DISPLAY=:98`） | `D3D12 (NVIDIA GeForce RTX 4070 SUPER)` | **437.22** | 1.0564 | 25,51,76,255 |
| 强制软件（`LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=1`） | `llvmpipe (LLVM 15.0.7, 256 bits)` | **1748.88** | 0.2051 | 26,51,76,255 |

这是最干净的一条瓶颈证据：**同样是最小负载，D3D12 客户端的每帧固定开销是 llvmpipe 的 4.5 倍，而它的 `glReadPixels` 比 llvmpipe 贵 5.5 倍**（llvmpipe 的读回是内存拷贝，D3D12 的读回是 GPU 栅栏往返）。加上真实引擎里 D3D12 的 `glReadPixels` 只占 11–14% 的帧时间，说明 D3D12 的其余 86–89% 花在渲染提交与 WSLg 呈现上。

## 5. 是否达到 2,000 帧/s 全机门

**没有达到。**

- 单实例上限：软件 **139.26 帧/s**、硬件 **108.25 帧/s** → 分别只有门槛的 **6.96%** 与 **5.41%**。
- 全机最好一次：软件 8 实例 **819.66 帧/s** = 门槛的 **41.0%**；同一轮里硬件 8 实例只有 **72.91 帧/s**（3.6%），比它自己的单实例还低 33%。
- 判定：按 `10-P0` WP2 决策表落在 **500–2,000 区间** → `continue_but_rebudget_by_measurement`，即**继续但按实测重估 P3/P4 预算**；G0.3 硬门状态保持 **未通过 / OPEN**。
- 即使把负载影响按最乐观的方向修正（单实例软件从 78 拉到 139 帧/s，约 1.8 倍），8 实例的 820 帧/s 也只到约 1,470 帧/s，**仍不足以过 2,000 门**；且 12/16 实例的实测（552.66 / 618.88）显示再堆实例只会让每实例读回成本从 6.5 ms 涨到 17 ms，合计不再增长。

## 6. 瓶颈证据（按可信度排序）

1. **每帧同步读回本身就吃掉大部分帧时间（引擎自报，最可信）**：软件后端 4.93–8.33 ms/帧，占 65–69%；硬件后端 1.18–3.06 ms/帧，占 11–14%。引擎源码注释自己写明 `glReadPixels` 是同步的、会一直等到光栅化结束（`src/rlharness/rlharness.cpp:665` 附近），所以 llvmpipe 下的这个数字里包含栅格化排空时间——它高，正说明**软件光栅化是瓶颈**。
2. **硬件 D3D12 路径的每帧固定开销比软件还高**：隔离对照 §4.4（391.16 对 1,748.88 帧/s，同负载），以及真实引擎里 D3D12 的渲染+呈现占 86–89% 帧时间。
3. **D3D12/WSLg 呈现路径不扩展**：N=1 → 108.25，N=4 → 142.23，N=8 → **72.91** 帧/s（每实例 109.8 ms/帧）。GPU 利用率峰值只有 57–79%，说明**不是 GPU 算力被打满**，而是多窗口在 WSLg 呈现/提交侧串行化。
4. **软件后端的扩展上限由核数决定**：N=8 时 8 个光栅线程 + 引擎线程已占满 20 核的可用配额，合计 819.66 帧/s；N=12/16 时每实例读回从 6.46 ms 升到 14.40/17.04 ms，合计反而掉到 552.66/618.88。
5. **机械负载敏感性**：同一后端同一参数的单实例在本次会话里跨 78.34–139.26 帧/s（软件）、45.72–108.25 帧/s（硬件）。最终复核时 `/proc/loadavg = 11.47 → 10.88`（软件）、`10.88 → 9.65`（硬件），机器同时有 108 份差分回放 + 40 局 bench 在跑（负责人 2026-11-08 告知）。**因此本卡片所有绝对帧率都是「当时机器状态下」的实测值，属偏保守的下界**；判门结论（未达 2,000）不受影响，因为最好的一次也只有 41%。

## 7. 有效性自检

- **V1 主指标真的是实测帧/s 吗**：是。帧数来自引擎 `STEP` 回包的 `advanced=`，墙钟来自本进程 `perf_counter()` 对 `STEP` 命令的计时；`N=1` 时 1,408 帧 / 10.110 s = 139.26 与 `STEP` 逐段累加一致（`parallel_efficiency=0.99999`）。
- **V2 是否有合成/估算冒充**：否。工具对「后端指纹不符」「无有效实例」「env 与声明后端不符」一律 `status=invalid` + 退出码 3；本轮所有参与判门的数据 `status=measured`、退出码 0。
- **V3 是否重复测量**：软件单实例 5 次（139.08/139.26/81.19/78.34/106.89，其中 139.08 是最早的 3 s 冒烟），硬件单实例 4 次（73.92/108.25/51.43/45.72，73.92 是加指纹之前的冒烟，env 相同）；离散度大，已在 §6.5 归因为外来负载，未用单次数字下结论。
- **V4 平凡基线**：软件 8 实例 819.66 对单实例 139.26 是 5.9 倍，扩展真实存在；硬件 8 实例 72.91 对单实例 108.25 是 **0.67 倍**，负扩展也是真实测量。
- **V5 读回真的开了吗**：是。`READBACK 1` 回包 `readback=1`，且 `READBACK` 计数器的 `frames=` 与推进帧数一致（a1：`frames=1408`、Δmicros=6,940,610）。
- **V6 后端是真的吗**：是，用**引擎进程内** `glGetString(GL_RENDERER)` 指纹（§2.1），不是 `glxinfo` 字符串、不是 `/proc/maps`（后者已证明会被硬链接误导）。
- **V7 数据隔离**：每实例独立 `--data-dir`（工具拒绝复用已存在目录），`th08.dat`/`thbgm.dat` 只读链接、`th08.cfg`/`score.dat` 各自复制；未删除任何文件。
- **V8 指标定义是否与门一致**：是。门要「渲染 + 读回」，本轮 `READBACK 1`、640×480 RGBA 每帧读回；`READBACK 0` 的对照仅作诊断且 `gate_applicable=false`。
- **V9 指标与阈值是否运行前固定**：是。阈值与决策表来自 `10-P0` §WP2/§G0，工具里的 `GATE_FPS=2000`、`GATE_RANGE_LOW=500` 是常量，运行前写死。

## 8. 剩余阻塞

1. **G0.3 硬门本身**：全机 819.66 帧/s，距 2,000 差 2.44 倍。按现有引擎（每帧同步 `glReadPixels` 640×480 + llvmpipe 单光栅线程/实例）在 20 核上的实测上限就在 800–1,500 帧/s 量级，**靠加实例过不了门**。
2. **硬件 GL 不是解**：D3D12 客户端路径的每帧开销更高，且 WSLg 呈现侧在 N≥4 后串行化（N=8 掉到 72.91 帧/s）。
3. **硬件后端退出路径损坏**：17/17 个硬件实例在 `QUIT` 后 SIGSEGV（`exit_code=-11`）或挂起（`timeout` 124）；软件后端 47 个实例里 45 个正常退出、2 个在超订载下也崩（139）。硬件后端在修好之前不能用于长跑。
4. **本卡片的绝对帧率受外来负载污染**：最终复核期间 `/proc/loadavg≈10–11.5`（20 核），机器上同时跑着 108 份差分回放 + 40 局 bench。**要在空载机器上重测得由负责人安排**；但「未达 2,000」不需要重测就能成立。
5. **未完成的切片**（受负责人 2026-11-08 收缩指令影响，已主动中止）：第 2 轮 `b8_rep2`（自适应分段的硬件 8 实例复测）、以及 `c1/c8/c16`（D3D12 驱动 + Xvfb 呈现，用来隔离「WSLg 呈现」与「D3D12 提交」两个因素）。这三个是下一步最有信息量的实验，各只需单实例或短窗口。

## 9. 变更文件

- 新增 `tools/rl/v400_render_throughput.py`（真实 harness 渲染+读回吞吐工具；只跑 WSL 内的真实 ELF，合成输入一律 `invalid`）。
- 新增 `tests/test_v400_render_throughput.py`（23 项，纯记账/判定逻辑；不冒充引擎证据）。
- 新增 `.tmp_v400_render_tp/`（本次全部原始产物）：`glx_probe.c`、`gl_probe_shim.c`、`evidence.sh`、`evidence_fix.sh`、`smoke.sh`、`matrix_r1.sh`、`matrix_r2.sh`、`final_runs.sh`、`cleanup.sh`、`r1_exit_codes.txt`、`r2_exit_codes.txt`、`final_exit_codes.txt`、`r1-*.json`、`r2-*.json`、`final-*.json`、`gl_backend_evidence.txt`、`matrix_r1.log`、`matrix_r2.log`。
- 未改引擎 C++、未改协议、未改依赖或第三方、未删任何文件、未改 `.wslconfig` 或任何 Windows 设置。
- 测试命令与结果：`$env:PYTHONPATH='D:\WorkShop\Touhou_RL'; python -m pytest -p no:cacheprovider --basetemp .tmp_v400_pytest_run -q tests/test_v400_render_throughput.py` → **23 passed，退出码 0**。

## 10. 对项目的直接含义

- `31-决策记录` 第 46 行的「WSLg 的 GL 是 D3D12」今天复现为真，但该记录容易让人以为「换硬件后端就能过 WP2 门」。本轮实测把这个假设否掉了：**在这台机器、这条 WSLg 栈上，D3D12 客户端路径比 llvmpipe 更慢，且不能扩展**。
- 因此 WP2 的真正杠杆只剩两类（都需要改研究构建或改观测规格，超出本卡授权范围）：
  1. **去掉/降频每帧同步读回**（例如按需读回、降分辨率后在 harness 内缩放、或用共享内存写出代替阻塞式 `glReadPixels`）——但门定义的正是「渲染 + 读回」，改口径必须由负责人决定；
  2. **换渲染实现**（如把 D3D8 兼容层换到真正的 GPU 路径 + 异步拷贝，或改用能批处理的呈现方式），这是工程量级更大的改动。
- 在此之前的预算口径必须按实测重估：软件后端按 **800 帧/s 全机（N=8）** 规划，而不是 2,000。
