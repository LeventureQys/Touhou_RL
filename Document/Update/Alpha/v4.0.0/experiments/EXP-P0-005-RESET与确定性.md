# EXP-P0-005 RESET/StageStart 与逐帧确定性

- 日期 / 工作包：2026-11-07 / P0-WP3，兼作 P0-WP5 验收第 1 项
- 假设：给定 `StageStart`（面、难度、机体、种子），两个独立进程在相同输入序列下应当产生逐帧相同的状态哈希——这是整个训练场"可复现"的地基。
- 预注册：主指标 = ①`RESET` 能离开标题画面并真正进入关卡；②同种子同输入的两个进程，5,000 帧内状态哈希序列逐项相等。通过阈值 = 两条都成立。样本量 = 2 个进程 × 5,000 帧，重复两批（共 4 次运行）。
- 设置：
  - 子模块 `rl-harness` 分支，提交 **`0217306`**；二进制 4,714,920 B，sha256 `46fbcfdb…8d51`
  - `RESET 0 1 0 1234 1`（Stage 1 / Normal / 結界組 / seed 1234 / 练习资源）
  - 输入：固定种子的伪随机掩码序列，只用白名单位 `0x00F7`，每个掩码保持 2 帧，共 2,500 chunk
  - 环境：xvfb + `LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1`
- 结果：**通过**。

  ```text
  reset1: ready=1 setup_frames=176 ... hash=d97b43478cc24bae
  reset2: ready=1 setup_frames=175 ... hash=d97b43478cc24bae
  DETERMINISTIC: 501 samples identical over 5000 frames
  first: ('reset', 'd97b43478cc24bae')  last: (4992, '1ae73523eb15d77f')
  ```

  连跑两批、共 4 个进程，收敛到完全相同的末帧哈希 `1ae73523eb15d77f`。这正是 `10-P0` WP5 验收第 1 条。
- 达到这一步之前修掉的四个缺陷（都靠加诊断、逐字段比对定位）：
  1. **不能从外部切 `curState`**。直接设 `SupervisorState_GameManager` 会让标题画面的 calc 元素发现状态变了而要求退出，episode 只活约 330 帧。改为 `RESET` 只"登记请求"，由标题自己的 Ready 态调用 `ConsumePendingStart()` 并返回 `CHAIN_CALLBACK_RESULT_CONTINUE_AND_REMOVE_JOB`——标题自己摘掉自己，和开发者自动回放路径完全一致。episode 因此能跑满 5,000 帧。
  2. **就绪判定被"陈旧的 COMPLETE"骗了**。`GAMEPLAY_SETUP_COMPLETE` 在标题初始化时就已置位，`RESET` 一进来就判定成功（`setup_frames=35`、`curState` 还是 1、setup 子线程还没建），真正的 setup 反而落到 `STEP` 里按调度决定时机——这就是 142/152 帧处分叉的根源。改为必须**先观察到 `IN_PROGRESS`** 才接受 `COMPLETE`；现在 `setup_frames≈175`。
  3. **阶段 RNG 原点懒捕获导致下溢**。原先在第一次 `HASH`/`STATE` 时才记原点，调用时机不同结果就不同，第 15 字段溢出成 65534。改为在 `RESET` 里记。
  4. **`g_ReplayManager` 是悬垂指针**。回放链撤掉后它仍非空，无条件读取会把垃圾写进哈希。改为只在 `flags.isReplay` 为真时读。
- 有效性自检：
  - V1 是：哈希直接来自 25 个引擎字段，逐帧计算
  - V2 不适用
  - V3 是：4 个独立进程各自启动
  - V4 是：5,000 帧 × 4 次运行，501 个采样点全等
  - V5 是：平凡基线是"跑完不崩"；更强的是 501 个哈希点与末帧哈希完全一致
  - V6 是：进程启动顺序、时刻都不同
  - V7 是：重复两批，结果一致
  - V8 不适用
  - V9 是：指标在运行前由 `10-P0` WP5 验收第 1 条固定
- 结论：**成立**。确定性达标，`RESET` 可用。
- 遗留（属 WP5，不阻塞本项）：`DEBUG` 显示进程仍有 **6–7 个线程**（SDL/Mesa 的）。单核确定性不受影响，但 `10-P0` §4.7 要求 `fork` 前必须是单线程，所以 WP5 做 `PARK`/`SPAWN` 时必须先把线程收敛到 1（或改用别的快照机制）。
- 成本：GPU 0 h；墙钟约 3.0 h；磁盘约 0.05 GB。
- 证据：[rlharness-reset.patch](<../evidence/EXP-P0-005/rlharness-reset.patch>)（58,074 B，sha256 `60c3071e…43ea`，含 `98ec3bd..0217306` 五个提交）
- 下一步：像素读回与共享内存（WP3 余下部分），然后是影子进程与空设备模式。
