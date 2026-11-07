# EXP-P0-008 Python 客户端首发：Th08Env 驱动真实引擎

- 日期 / 工作包：2026-11-07 / P0-WP4
- 假设：`03-系统架构` §5 定义的 Python API 可以完全建立在已经跑通的控制协议 + 共享内存之上，不需要再改引擎。
- 预注册：主指标 = ①`Th08Env` 能完成 reset + 多步 step 并返回形状正确的像素观测；②零特权模式下 `state()` 抛 `PrivilegeError` 且奖励恒为 0；③动作编码在 18 个候选上双射。通过阈值 = 三项都成立。样本量 = 1 次端到端运行 + 30 项单元测试。
- 设置：
  - 二进制：子模块 `rl-harness` @ `f563df5` 的 `build/rl/th08-modern`
  - `Th08Env(binary, data_dir, shm_name="v400e2e", privileged=True, max_frames=400)`
  - `StageStart(stage=0, difficulty="normal", team=0, seed=1234)`
  - 环境：xvfb + `LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1`
- 结果：**三项都成立**。

  ```text
  === privileged ===
   obs shapes : {(480, 640, 4)}
   readback   : frames=290 micros=706406
   header     : version=1 frame=119 width=640 height=480 flags=2
   state len  : 25
  === zero-privilege ===
   obs shapes : {(480, 640, 4)}
   state refused: True
   reward sum : 0.0
  E2E OK
  ```

  单元测试：`30 passed`（动作编码 16 项 + 共享内存 14 项，纯 CPU，不需要引擎）。
- 本次新增的模块（`touhou_rl/v400/gym/`）：`actions.py`、`protocol.py`、`shm.py`、`env.py`，以及 `touhou_rl/tests/conftest.py`（只注册 `harness` 标记，不加任何钩子，避免影响既有测试）。
- 修掉的两个自己造的 bug（都靠测试抓到，不是靠读代码看出来）：
  1. **`cand = focus * 9 + move` 解反了**。原写成 `move, focus = divmod(cand, 9)`，导致 `focus` 变成 2 这种非法值、`move` 恒为 0，18 个候选塌成 4 个互不相同的掩码。现在有 `test_candidates_produce_eighteen_distinct_masks` 与逐候选双射测试盯着。
  2. **方向解码用了"子集匹配"**。`NE = UP|RIGHT` 会先命中 `N = UP`，所有斜向都解成 N。改成按位集合精确比较。
- 还暴露并修掉了子模块的一个真 bug（`f563df5`）：`AfterDrawFrame` 读了帧却从未写进共享内存，所以 `READBACK` 计数在涨而 header 的 `flags` 始终是 0，只有显式 `PIXELS` 命令才发布。这个 bug 是 `Th08Env` 读共享块（而不是轮询 `PIXELS`）才暴露出来的——换个写法就会被掩盖。
- 有效性自检：
  - V1 是：观测直接来自共享内存里的像素块，`flags=2` 表示引擎确认有效
  - V2 不适用
  - V3 是：形状 (480,640,4) 与 640×480 RGBA 一致；`state` 长度为 25，与 25 字段约定一致
  - V4 不适用：这是契约测试，不是统计推断
  - V5 是：平凡基线是"进程起得来但不推进"；实际帧号推进到 119、readback 记了 290 帧
  - V6 不适用
  - V7 未做：未重复多次；确定性由 EXP-P0-005 覆盖
  - V8 是：零特权模式下 `state()` 被拒绝、奖励恒 0，学生拿不到任何特权信息
  - V9 是：三项指标在运行前由 `03-系统架构` §5 固定
- 结论：**成立**（首个可用切片）。`Th08Env` 已经能作为训练接口使用。
- 成本：GPU 0 h；墙钟约 1.5 h；磁盘约 0.02 GB。
- 尚未做（WP4 剩余）：
  - `VecTh08Env` 向量化封装还没写；
  - 进程崩溃时"标记 abort + 自动重启 + 记录次数"没做；
  - `PIXELS` 的像素尚未按 `obs_spec` 缩放（G2 才冻结分辨率）；
  - reset 之后 header 的帧号要等第一个 step 才刷新（小瑕疵，读 `info.frame` 时值为上一轮的计数）。
- 下一步：`VecTh08Env` + 崩溃恢复，然后进 WP5 的空设备模式与 `fork`。
