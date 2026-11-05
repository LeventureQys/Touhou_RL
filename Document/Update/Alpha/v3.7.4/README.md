# v3.7.4 当前执行状态

更新：2026-11-02。本版本只做一件事：**把"画面与内存状态是否同一帧"从不可证明变成可证明、可复核、可采集**。没有训练、没有标签解冻、没有改动 canonical 镜像。
- 后续清理版本：[v3.7.5](<../v3.7.5/README.md>)

| 项 | 当前事实 | 判定 |
|---|---|---|
| P0 同帧证明 | 帧栅栏变体：Present 后 ticket + hold + 游戏侧摘要/后台缓冲复读；240/240 帧通过 13 项检查，GDI 截图与后台缓冲逐字节相同 | **技术阻塞已解决** |
| canonical 镜像 | 帧栅栏只存在于 `build_fence/` 变体；canonical `.text` SHA 不变（`7B5B3C8A…`），整文件仅 PE 时间戳不同，已恢复原字节（`CDD23BCA…`） | **未被改动** |
| 旧读法的同帧性 | 不 hold 时 120 次读取里 20.83% 跨 ticket，至少 4.17% 弹池读取可证明不是该帧；摘要只能证伪 | **仍不可用于精确标签** |
| 采集接入 | 新增 [frame_fence_capture.py](<../../../../tools/rl/frame_fence_capture.py>)（PNG + JSONL + manifest，每帧带校验结果）；实测 60 帧 0 丢弃 | **可用工具，未接入正式管线** |
| 标签侧 | `capture_dataset.py` 等未改；`frame_proof=exact` 需要新的预注册与清单字段 | pending |

## 已知边界

- 变体镜像不能用于 exact-match / 版本一致性声明；全局地址整体 `+0x3000`，读取必须用变体地址表（`TH08_ADDRESS_FILE`）。
- hold 占用帧时间（本次中位 7.44 ms）；读取方更慢时会丢 hold，由 `holdBudgetSkips` 与 ticket 增量显式暴露。
- "窗口 DC == 后台缓冲"是当前 GPU/驱动/窗口模式（D3D8 `COPY`、无 vsync）下的实测结论，换环境需重测。
- 摘要只能证伪，不能证明；只有 `ticket + holdPhase` 组合构成同帧证明。

## 报告入口

- [P0 同帧证明：帧栅栏变体（round1）](<reports/P0-frame-fence-round1.md>)
- 证据：[主轮 summary](<evidence/frame-fence-round1/summary.json>)、[复现轮 summary](<evidence/frame-fence-round1/replication/summary.json>)、[逐帧记录](<evidence/frame-fence-round1/frames.json>)
- 建库文档：[FRAME_FENCE.md](<../../../../third_party/th08/docs/FRAME_FENCE.md>)

## 下一任务卡

1. **把 fence 接进正式采集**：`capture_dataset.py` 改用 `FenceFrameSampler`，样本写 `frame_proof=exact` + 变体镜像/地址表哈希；先做预注册，再决定是否解冻像素几何标签。
2. **长期吞吐标定**：用 `holdEvery` 与并发会话测每分钟可用精确帧数，决定是否把 10 MB 敌人摘要移出逐帧路径。→ 已有第一段实测（2026-11-02）：逐样张 PNG 落盘时，等新 ticket 的模式只有 **3.8 张/s**（`wait` 中位 232.9 ms、60 张烧掉 59 个 hold 预算）；改为消费当前 hold（`accept_current=True`）后 **45.2 张/s（11.9×）**、预算耗尽 0，13 项检查 780/780 不变。上限由读取管线 8.16 ms + I/O 13.3 ms 决定，下一步可测把敌人摘要移出逐帧路径的收益。
3. **P0 出口重判**：解冻判据以"每张样本带 ticket + 全检查 + 采集环境指纹"为准，而不是 ±1 帧误差膨胀。→ **已完成 2026-11-02**：[v3.7.3 round26](<../v3.7.3/reports/P0-fence-round26.md>) 在该变体镜像上重跑 P0，10 条预注册判据全过（60/60 样张 780/780 检查、ticket/logic 逐帧 +1、发送→掩码 1 帧 / 掩码→位移 1 帧、四类动力学各 20/20、四边界），并给出解冻规则 v2；同时暴露本文件第 2 条的实测代价：`FenceFrameSampler` 每次等下一张 hold，单样张 `wait` 中位 235 ms、60 次预算耗尽。接入前先修这个再谈吞吐。
