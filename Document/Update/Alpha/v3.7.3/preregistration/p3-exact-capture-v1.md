# 预注册：精确同帧数据集（第一版，试采集）

- 写入时间：2026-11-02，**在任何 exact 采集之前**。
- 依据：[P0-fence-round26](<../reports/P0-fence-round26.md>) 给出的解冻规则 v2；[P0-exit-round15 补记](<../reports/P0-exit-round15.md>)。
- 本预注册只定义**数据门**：怎样的一批样本才配叫 `frame_proof=exact`。它不定义战术标签、不定义训练、不宣称任何能力提升。

## 1. 为什么需要单独的预注册

round26 证明了同帧性，但那是在**采集脚本自己写 JSONL** 的条件下。正式数据集走 `capture_dataset.py → ShardWriter → manifest → process/train` 这条老管线，它现在的字段里**没有**帧证明，默认语义是 `alignment_unknown`。不先定规则就接进去，会出现两种坏结果：

- 新数据带着"看起来像 exact"的字段，实际只过了部分检查；
- 新旧数据混在同一份 manifest 里，下游无法区分。

所以先写死：**exact 是一种逐帧、可复核、不可混用的属性**，不是一个数据集级的开关。

## 2. 数据集定义（试采集）

| 项 | 值 |
|---|---|
| 来源 | 单份回放、单面（回放自走，非人工操作） |
| 回放 | `th08_replay/Th8_ud3936.rpy`（Lunatic Youmu，本仓库既有资产），冻结前不替换 |
| 面 | 该回放第一面（`--stage -1` 自动解析） |
| 规模 | **3,000 个 exact 帧**（这是启动规模，不是统计功效保证） |
| 镜像 | `third_party/th08/build_fence/th08.exe` + `build_fence/frame-fence-addresses.json` |
| 图像 | 每帧一张 PNG（无损，客户区 640×480），文件名含 ticket |
| 观测 | 与现有 `obs.encode` 相同；**不新增任何观测字段** |
| 动作 | 与现有管线相同：`gui_input_current → buttons_to_action`（回放输入） |
| 标签 | **不产出战术标签**；标签问题另立预注册（round28 已把合法标签源收窄到 F2） |

## 3. 资格判据（先写死）

| 编号 | 判据 | 通过条件 |
|---|---|---|
| E1 | 逐帧 13 项检查 | 全部 3,000 帧通过全部 13 项，**0 例外**（不容忍"大部分通过"） |
| E2 | 逐帧证明字段完整 | 每帧有 `ticket`、`logicFrame`、`imageSha256`、`imagePath`、`build.{exeSha256,textSha256,addressTableSha256}`、`environment.{platform,gpu,python,gitHead}` |
| E3 | ticket 与逻辑帧 1:1 | 相邻被采帧的 `logicFrame` 增量 = `ticket` 增量 = 1（允许有明确记录的丢弃，丢弃帧必须单独计数） |
| E4 | 图像互不相同 | 3,000 帧的两两像素哈希去重后仍是 3,000（重复即"旧画面复用"，整批作废） |
| E5 | 图像与后台缓冲一致 | 每帧 `pixels_match_backbuffer` 为真（已含在 E1 的 13 项里，单列以便复核） |
| E6 | 镜像身份可复核 | 数据清单记录 exe / `.text` / 地址表 / map 四个 SHA；采集后复算一致 |
| E7 | 环境指纹 | 记录 OS、GPU 与驱动、窗口模式、客户区尺寸、输入后端；缺一项则整批降级为 `alignment_unknown` |
| E8 | 不混用 | 同一数据集内 `frameProof` 只允许一个取值；`process` 遇到混合直接失败 |
| E9 | canonical 拒绝 | 请求 `exact` 但模板不是 fence 构建（SHA 不符）时，**拒绝启动**，不允许"先跑再说" |
| E10 | 失败即停 | 任一帧检查失败或 hold 预算耗尽导致半帧：停止采集，写入失败原因，数据集标记为 `invalid_exact`，**不修补、不补齐** |

## 4. 不以本数据集声称的事

- 不声称"与 canonical 等价"（变体镜像，全局地址 `+0x3000`）。
- 不声称标签有效：动作来自回放，帧证明只说明像素↔状态同帧，不说明动作是对的策略标签。
- 不声称跨机器成立：E7 只是记录环境，换机器必须重采。
- 不进入训练：第一版只交付"能通过 E1–E10 的数据集"，训练需要另立预注册。

## 5. 产物

| 路径 | 内容 |
|---|---|
| `<dataset>/exact-frames.jsonl` | 逐帧证明记录（`touhou_rl.frame_proof.FrameProofRecord`） |
| `<dataset>/frames/*.png` | 每帧图像 |
| `<dataset>/capture-report.json` | 规模、丢弃、ticket/logic 增量、镜像与环境指纹、判定 |
| `<dataset>/shards/*.npz` | 与老管线相同的观测/动作分片（`frameProof` 写进 provenance） |
| manifest（由 `process` 生成） | 必须带 `frameProof`、`exactValidFrames`、镜像与环境 SHA |

## 6. 判定与失败处理

- E1–E10 全过 → 该数据集可标 `frame_proof=exact`，但仍标注"试采集、无标签、未训练"。
- 任一不过 → 数据集标 `invalid_exact`，**保留全部原始记录**，写清失败帧与原因；不重新跑到通过（重跑必须另开编号并披露）。
