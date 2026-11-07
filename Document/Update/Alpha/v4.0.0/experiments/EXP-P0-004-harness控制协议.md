# EXP-P0-004 harness 首个纵向切片：控制协议能驱动帧推进

- 日期 / 工作包：2026-11-07 / P0-WP3
- 假设：把 `TH08_RL_HARNESS` 作为默认关闭的编译开关接进移植分支后，能构建出**不再按 60 Hz 节拍跑、而是由外部命令逐帧驱动**的引擎进程，并且能导出可用于确定性比对的 25 字段状态与哈希。
- 预注册：主指标 = ①`TH08_RL_HARNESS=ON` 构建通过分支自带的 `verify-portable-linux.sh`；②冒烟测试里 `HELLO/PING/FRAME/STEP/HASH/STATE/QUIT` 全部往返成功，且帧号按命令推进、哈希随状态变化。通过阈值 = 两条都成立。样本量 = 1 次构建 + 1 次冒烟。
- 设置：
  - worktree `third_party/th08-rl`，分支 `rl-harness`，提交 **`98ec3bd`**
  - 构建：`cmake -S . -B build/rl -G Ninja -DCMAKE_BUILD_TYPE=Debug -DTH08_RL_HARNESS=ON`
  - 运行：`xvfb-run -a -s "-screen 0 640x480x24" th08-modern --data-dir ~/v400-data/game --rl-control 0,1`
  - 环境：`LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=1 SDL_AUDIODRIVER=dummy`
  - 二进制：4,695,888 B，sha256 `b2297abf42cfa107c5b7a9b5a302b1fa85f2610b4f716b537570343c4cce17a0`
- 结果：**两条主指标都成立**。构建 4 个对象后链接成功并通过 `ELF64 PIE and native global ownership: OK`。冒烟输出：

  ```text
  >> HELLO            << OK hello proto=1 harness=1 role=env frames=0
  >> PING             << OK pong
  >> FRAME            << OK frame=0
  >> STEP 1 120       << OK advanced=120 frame=120 hash=49299c5ae622189e
  >> HASH             << OK hash=49299c5ae622189e frames=120
  >> STATE            << OK frames=120 f1=0 ... f14=3279 ...
  >> STEP 1 60        << OK advanced=60 frame=180 hash=cd679b2e57719b22
  >> QUIT             << OK bye
  ```

  帧号严格按命令推进（0 → 120 → 180），两次哈希不同，`f14` 是 `g_Rng.GetSeed()` 的非零值，说明读到的是真实引擎状态而不是空壳。
- 有效性自检：
  - V1 是：`STEP` 的返回值来自 `PumpOneFrame()` 的实际计数，不是估算；`STATE` 直接读引擎全局量
  - V2 不适用
  - V3 是：`f14=3279` 落在 u16 范围内且非零；标题画面下 `f1=0`（还在 Stage 0 默认值）符合预期
  - V4 不适用：单次冒烟，不是统计推断
  - V5 是：平凡基线是"进程不崩溃"；更强的是帧号按命令精确推进且哈希随状态变化
  - V6 不适用
  - V7 未做：同一命令序列跑两次比对哈希，留到 WP5 的 G0.2 第 1 项
  - V8 不适用
  - V9 是：指标在运行前由 `10-P0` WP3 的验收条款固定
- 结论：**成立**（首个纵向切片）。控制通道、逐帧驱动、输入注入点、状态导出四件事都通了。
- 成本：GPU 0 h；墙钟约 1.0 h；磁盘约 0.05 GB。
- 证据：[rlharness-skeleton.patch](<../evidence/EXP-P0-004/rlharness-skeleton.patch>)（19,003 B，sha256 `9e8d9ba7…c288`）。worktree 在父仓库里被 gitignore，所以把提交导出成 patch 存档，避免只存在于 WSL 磁盘上。
- 尚未实现（后续工作包）：`RESET`/`StageStart` 按种子直接进关、像素读回、共享内存布局、影子进程与空设备模式、`BRANCH`/`PARK`/`SPAWN`、`STATE` 的子弹/敌人/激光池。
- 一个实现上的坑：`linux_runtime.cpp` 把 `g_arguments` 放在匿名 namespace 里（内部链接），别的编译单元取不到；`ParseCommandLine()` 因此改读 `/proc/self/cmdline`，不需要改动平台层。另一处：`Rng::ReplayTraceGenerationCount()`（25 字段里的第 15 项）被 `TH08_ENABLE_REPLAY_TRACE` 包着，所以 harness 构建顺带打开该开关；记录器在没有 `TH08_REPLAY_TRACE_REFERENCE` 时完全不启用，运行时零成本。
